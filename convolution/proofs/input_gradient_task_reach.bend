# Every band a call of an input gradient task copies lies in the gradient
# rows the task reads, its reach.

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
import ../../lib/tensor_tasks.bend as Tasks
import ../../lib/storage_buffer.bend as Storage
import ../../lib/storage_copy.bend as Copy
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
import ../../lib/proofs/image_steps.bend as Steps
import ../../lib/tensor_pieces.bend as Pieces
import ../../lib/proofs/tensor_task_runs.bend as Runs
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ./input_gradient_image_rows.bend as Rows
import ./input_gradient_batch_rows.bend as BatchRows
import ./input_gradient_cell_sources.bend as Sources
import ../input_gradient_specification.bend as Specification
import ../../lib/proofs/input_gradient_tiled_band_extents.bend as Bands
import ../../lib/proofs/input_gradient_tiled_band_order.bend as BandOrder
import ../../lib/proofs/input_gradient_tiled_copy_geometry.bend as CopyRows
import ../../lib/proofs/tensor_halo_windows.bend as Windows
import ../../lib/proofs/u32_comparison.bend as Comparison
import ./input_gradient_piece_cells.bend as PieceCells

# ---- The words of a reach ----

# The rows the first image of a task reads from, and the last image's.
def task_head(+call: Domain.Call,+first: U32,+count: U32) -> U32:
  BatchRows.count_word(Domain.geometry(call),PieceCells.piece_start(call,first),PieceCells.piece_end(call,first,count))
def task_low_first(+call: Domain.Call,+first: U32) -> U32:
  BatchRows.first_word(Domain.geometry(call),PieceCells.piece_start(call,first))
def task_last(+call: Domain.Call,+first: U32,+count: U32) -> U32:
  (PieceCells.piece_end(call,first,count) - 1 : U32)
def task_tail_first(+call: Domain.Call,+first: U32,+count: U32) -> U32:
  Bool.pick(U32,U32.is_eq(BatchRows.image_word(Domain.geometry(call),PieceCells.piece_start(call,first)),
    BatchRows.image_word(Domain.geometry(call),task_last(call,first,count))),task_low_first(call,first),0)
def task_tail(+call: Domain.Call,+first: U32,+count: U32) -> U32:
  (BatchRows.first_word(Domain.geometry(call),task_last(call,first,count)) + 1 - task_tail_first(call,first,count) : U32)

# The cells of a gradient row.
def row_cells(+call: Domain.Call) -> Nat:
  Nat.mul(Rows.columns_count(call),Rows.groups_count(call))

# The gradient rows of an image as a number.
def gradient_height_rows(+call: Domain.Call) -> {Domain.gradient_height(Domain.geometry(call)) == Rows.rows_count(call) : Nat}:
  Domain.Call{+geometry,+images,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  {==}
def gradient_word(+call: Domain.Call) -> {U32.to_nat(Kernel.gradient_height(Domain.geometry(call))) == Rows.rows_count(call) : Nat}:
  Domain.Call{+geometry,+images,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  {==}

# The cell a gradient row of an image starts at, as the runtime computes it.
def scaled_word(+call: Domain.Call,+image: U32,+start: U32) ->
  {((image * Kernel.gradient_height(Domain.geometry(call)) + start) * Kernel.gradient_row_cells(Domain.geometry(call)) : U32) ==
    U32.from_nat(Nat.mul(Nat.add(Nat.mul(U32.to_nat(image),Rows.rows_count(call)),U32.to_nat(start)),row_cells(call))) : U32}:
  Domain.Call{+geometry,+images,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  Rows.band_word(image,gh,start,gw,Tiled.groups(o))

# The last gradient row a call copies is that of its band query.
def stop_equal(+call: Domain.Call,+envelope: Geometry.WordEnvelope,+first: U32,+count: U32,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Domain.height(Domain.geometry(call))) == True{} : Bool}) ->
  {Kernel.copy_stop(Domain.geometry(call),first,count) == CopyRows.stop(Bands.caller_query(call,envelope,first,count,contained)) : U32}:
  Domain.Call{+geometry,+images,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  {==}

# The last gradient row a call copies is a row of its image.
def stop_below(+call: Domain.Call,+envelope: Geometry.WordEnvelope,+first: U32,+count: U32,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Domain.height(Domain.geometry(call))) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Kernel.copy_stop(Domain.geometry(call),first,count)),Rows.rows_count(call)) == True{} : Bool}:
  +query = Bands.caller_query(call,envelope,first,count,contained)
  +high = Kernel.band_high(Bands.geometry(query),Bands.first(query),Bands.last(query))
  +rows = Domain.gradient_height(Domain.geometry(call))
  +ended = Nat.sub(U32.to_nat(high),Domain.kernel(Domain.geometry(call)))
  %Equal.sym(U32,Kernel.copy_stop(Domain.geometry(call),first,count),CopyRows.stop(query),stop_equal(call,envelope,first,count,contained)) :
    {Nat.is_le(U32.to_nat(_),Rows.rows_count(call)) == True{} : Bool}
  %gradient_height_rows(call) : {Nat.is_le(U32.to_nat(CopyRows.stop(query)),_) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(CopyRows.stop(query)),Nat.min(ended,rows),CopyRows.stop_exact(query)) : {Nat.is_le(_,rows) == True{} : Bool}
  NatExtrema.minimum_right(ended,rows)

# One times a number is the number.
def one_times(+value: Nat) -> {Nat.mul(1n,value) == value : Nat}:
  Equal.trans(Nat,Nat.mul(1n,value),Nat.mul(value,1n),value,RowMajor.multiply_commutative(1n,value),Order.times_one(value))

# A gradient cell before the end of a row of an image of the batch fits
# the capacity.
def slot_inside(+call: Domain.Call,+image: Nat,+start: Nat,+inside: {Nat.is_lt(image,Domain.batch(call)) == True{} : Bool},
  +start_le: {Nat.is_le(start,Rows.rows_count(call)) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(Nat.add(Nat.mul(image,Rows.rows_count(call)),start),row_cells(call)),Rows.cells_count(call)) == True{} : Bool}:
  +rows = Rows.rows_count(call)
  +size = row_cells(call)
  +images = Domain.batch(call)
  +room = U32.to_nat(Machine.capacity(Domain.limits(call)))
  +offset = Nat.mul(image,rows)
  +slot = Nat.add(offset,start)
  +next = Nat.add(image,1n)
  +next_le = {%Equal.sym(Nat,Nat.add(image,1n),Nat.add(1n,image),Order.commutative(image,1n)) : {Nat.is_le(_,images) == True{} : Bool}
    Order.strict_follow(image,images,inside) : {Nat.is_le(next,images) == True{} : Bool}}
  +next_split = Equal.trans(Nat,Nat.mul(next,rows),Nat.add(offset,Nat.mul(1n,rows)),Nat.add(offset,rows),RowMajor.distrib(image,1n,rows),
    Equal.cong(Nat,Nat,number => Nat.add(offset,number),Nat.mul(1n,rows),rows,one_times(rows)))
  +slot_next = Order.transitive(slot,Nat.add(offset,rows),Nat.mul(next,rows),
    Encoded.sum_bounds(offset,start,offset,rows,Order.reflexive(offset),start_le),
    {%Equal.sym(Nat,Nat.mul(next,rows),Nat.add(offset,rows),next_split) : {Nat.is_le(Nat.add(offset,rows),_) == True{} : Bool}
      Order.reflexive(Nat.add(offset,rows)) : {Nat.is_le(Nat.add(offset,rows),Nat.mul(next,rows)) == True{} : Bool}})
  +slot_rows = Order.transitive(slot,Nat.mul(next,rows),Nat.mul(images,rows),slot_next,RowMajor.mul_le(next,images,rows,next_le))
  Order.transitive(Nat.mul(slot,size),Nat.mul(Nat.mul(images,rows),size),Rows.cells_count(call),RowMajor.mul_le(slot,Nat.mul(images,rows),size,slot_rows),
    {%Equal.sym(Nat,Nat.mul(Nat.mul(Nat.mul(images,rows),Rows.columns_count(call)),Rows.groups_count(call)),Nat.mul(Nat.mul(images,rows),size),
        RowMajor.assoc(Nat.mul(images,rows),Rows.columns_count(call),Rows.groups_count(call))) :
      {Nat.is_le(Nat.mul(Nat.mul(images,rows),size),_) == True{} : Bool}
      Order.reflexive(Nat.mul(Nat.mul(images,rows),size)) : {Nat.is_le(Nat.mul(Nat.mul(images,rows),size),Rows.cells_count(call)) == True{} : Bool}})

# A slot of the gradient rows inside the batch fits the capacity.
def slot_room(+call: Domain.Call,+image: Nat,+start: Nat,+inside: {Nat.is_lt(image,Domain.batch(call)) == True{} : Bool},
  +start_le: {Nat.is_le(start,Rows.rows_count(call)) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(Nat.add(Nat.mul(image,Rows.rows_count(call)),start),row_cells(call)),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool}:
  Order.transitive(Nat.mul(Nat.add(Nat.mul(image,Rows.rows_count(call)),start),row_cells(call)),Rows.cells_count(call),U32.to_nat(Machine.capacity(Domain.limits(call))),
    slot_inside(call,image,start,inside,start_le),Rows.cells_room(call))

# ---- The last row of a task ----

# The last row a task reads, its image and its row in the image.
type LastFacts<-call: Domain.Call,-first: U32,-count: U32> is Data:
  LastFacts{exact: {U32.to_nat(task_last(call,first,count)) == Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n) : Nat},
    image: {U32.to_nat(BatchRows.image_word(Domain.geometry(call),task_last(call,first,count))) ==
      Nat.div(Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n),Domain.height(Domain.geometry(call))) : Nat},
    rest: {U32.to_nat(BatchRows.first_word(Domain.geometry(call),task_last(call,first,count))) ==
      Nat.mod(Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n),Domain.height(Domain.geometry(call))) : Nat},
    inside: {Nat.is_lt(Nat.div(Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n),Domain.height(Domain.geometry(call))),Domain.batch(call)) == True{} : Bool},
    after: {Nat.is_le(U32.to_nat(PieceCells.piece_start(call,first)),Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n)) == True{} : Bool},
    restored: {Nat.add(Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n),1n) == U32.to_nat(PieceCells.piece_end(call,first,count)) : Nat}}
def last_image(-call: Domain.Call,-first: U32,-count: U32,facts: LastFacts<call,first,count>) ->
  {U32.to_nat(BatchRows.image_word(Domain.geometry(call),task_last(call,first,count))) ==
    Nat.div(Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n),Domain.height(Domain.geometry(call))) : Nat}:
  LastFacts{exact,image,rest,inside,after,restored} = facts
  image
def last_rest(-call: Domain.Call,-first: U32,-count: U32,facts: LastFacts<call,first,count>) ->
  {U32.to_nat(BatchRows.first_word(Domain.geometry(call),task_last(call,first,count))) ==
    Nat.mod(Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n),Domain.height(Domain.geometry(call))) : Nat}:
  LastFacts{exact,image,rest,inside,after,restored} = facts
  rest
def last_inside(-call: Domain.Call,-first: U32,-count: U32,facts: LastFacts<call,first,count>) ->
  {Nat.is_lt(Nat.div(Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n),Domain.height(Domain.geometry(call))),Domain.batch(call)) == True{} : Bool}:
  LastFacts{exact,image,rest,inside,after,restored} = facts
  inside
def last_after(-call: Domain.Call,-first: U32,-count: U32,facts: LastFacts<call,first,count>) ->
  {Nat.is_le(U32.to_nat(PieceCells.piece_start(call,first)),Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n)) == True{} : Bool}:
  LastFacts{exact,image,rest,inside,after,restored} = facts
  after
def last_restored(-call: Domain.Call,-first: U32,-count: U32,facts: LastFacts<call,first,count>) ->
  {Nat.add(Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n),1n) == U32.to_nat(PieceCells.piece_end(call,first,count)) : Nat}:
  LastFacts{exact,image,rest,inside,after,restored} = facts
  restored

# The last row of a task whose rows [start, end) lie in the batch.
def last_facts(+call: Domain.Call,+first: U32,+count: U32,
  +nonempty: {Nat.is_lt(U32.to_nat(PieceCells.piece_start(call,first)),U32.to_nat(PieceCells.piece_end(call,first,count))) == True{} : Bool},
  +inside: {Nat.is_le(U32.to_nat(PieceCells.piece_end(call,first,count)),Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call)))) == True{} : Bool},
  +rows_room: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool}) ->
  LastFacts<call,first,count>:
  +start = PieceCells.piece_start(call,first)
  +end = PieceCells.piece_end(call,first,count)
  +last = task_last(call,first,count)
  +s = U32.to_nat(start)
  +e = U32.to_nat(end)
  +previous = Nat.sub(e,1n)
  +cap = Machine.capacity(Domain.limits(call))
  +end_room = Order.transitive(e,Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),U32.to_nat(cap),inside,rows_room)
  +one = Order.positive_one(e,Order.le_lt_trans(0n,s,e,Order.zero_le(s),nonempty))
  +restore = Order.restore_shift(1n,e,one)
  +restored = Equal.trans(Nat,Nat.add(previous,1n),Nat.add(1n,previous),e,Order.commutative(previous,1n),restore)
  +exact = FullRange.subtract(end,1,cap,one,Order.transitive(previous,e,U32.to_nat(cap),Order.sub_le(e,1n),end_room))
  +previous_below = Bounds.predecessor_strict(previous,e,
    {%Equal.sym(Nat,Nat.add(1n,previous),e,restore) : {Nat.is_le(_,e) == True{} : Bool} Order.reflexive(e) : {Nat.is_le(1n+previous,e) == True{} : Bool}})
  +before = {%Equal.sym(Nat,U32.to_nat(last),previous,exact) : {Nat.is_lt(_,e) == True{} : Bool} previous_below : {Nat.is_lt(U32.to_nat(last),e) == True{} : Bool}}
  +step = BatchRows.row_step(call,last,end,before,inside,rows_room)
  +height = Domain.height(Domain.geometry(call))
  +after = Division.lt_succ_le(s,previous,{%Equal.sym(Nat,Nat.add(1n,previous),e,restore) : {Nat.is_lt(s,_) == True{} : Bool} nonempty : {Nat.is_lt(s,1n+previous) == True{} : Bool}})
  LastFacts{exact,
    {%exact : {U32.to_nat(BatchRows.image_word(Domain.geometry(call),last)) == Nat.div(_,height) : Nat} BatchRows.row_image(call,last,end,step) :
      {U32.to_nat(BatchRows.image_word(Domain.geometry(call),last)) == Nat.div(previous,height) : Nat}},
    {%exact : {U32.to_nat(BatchRows.first_word(Domain.geometry(call),last)) == Nat.mod(_,height) : Nat} BatchRows.row_first(call,last,end,step) :
      {U32.to_nat(BatchRows.first_word(Domain.geometry(call),last)) == Nat.mod(previous,height) : Nat}},
    {%exact : {Nat.is_lt(Nat.div(_,height),Domain.batch(call)) == True{} : Bool} BatchRows.row_image_inside(call,last,end,step) :
      {Nat.is_lt(Nat.div(previous,height),Domain.batch(call)) == True{} : Bool}},
    after,restored}

# ---- The rows the last image of a task reads ----

# The row the last image of a task reads from: the task's first row when
# the task holds one image, otherwise row 0.
type TailFirst<-call: Domain.Call,-first: U32,-count: U32,-word: U32> is Data:
  TailFirst{below: {Nat.is_le(U32.to_nat(word),Nat.add(Nat.mod(Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n),Domain.height(Domain.geometry(call))),1n)) == True{} : Bool},
    start: {Nat.is_le(Nat.add(Nat.mul(Nat.div(Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n),Domain.height(Domain.geometry(call))),Domain.height(Domain.geometry(call))),
      U32.to_nat(word)),Nat.max(U32.to_nat(PieceCells.piece_start(call,first)),
      Nat.mul(Nat.div(Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n),Domain.height(Domain.geometry(call))),Domain.height(Domain.geometry(call))))) == True{} : Bool}}
def tail_below(-call: Domain.Call,-first: U32,-count: U32,-word: U32,facts: TailFirst<call,first,count,word>) ->
  {Nat.is_le(U32.to_nat(word),Nat.add(Nat.mod(Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n),Domain.height(Domain.geometry(call))),1n)) == True{} : Bool}:
  TailFirst{below,start} = facts
  below
def tail_start(-call: Domain.Call,-first: U32,-count: U32,-word: U32,facts: TailFirst<call,first,count,word>) ->
  {Nat.is_le(Nat.add(Nat.mul(Nat.div(Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n),Domain.height(Domain.geometry(call))),Domain.height(Domain.geometry(call))),
    U32.to_nat(word)),Nat.max(U32.to_nat(PieceCells.piece_start(call,first)),
    Nat.mul(Nat.div(Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n),Domain.height(Domain.geometry(call))),Domain.height(Domain.geometry(call))))) == True{} : Bool}:
  TailFirst{below,start} = facts
  start

def tail_case(same: Bool,+call: Domain.Call,+first: U32,+count: U32,
  +nonempty: {Nat.is_lt(U32.to_nat(PieceCells.piece_start(call,first)),U32.to_nat(PieceCells.piece_end(call,first,count))) == True{} : Bool},
  +inside: {Nat.is_le(U32.to_nat(PieceCells.piece_end(call,first,count)),Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call)))) == True{} : Bool},
  +rows_room: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +classified: {U32.is_eq(BatchRows.image_word(Domain.geometry(call),PieceCells.piece_start(call,first)),
    BatchRows.image_word(Domain.geometry(call),task_last(call,first,count))) == same : Bool}) ->
  TailFirst<call,first,count,Bool.pick(U32,same,task_low_first(call,first),0)>:
  match same:
    case True{}:
      +geometry = Domain.geometry(call)
      +start = PieceCells.piece_start(call,first)
      +end = PieceCells.piece_end(call,first,count)
      +height = Domain.height(geometry)
      +positive = Domain.height_positive(call)
      +s = U32.to_nat(start)
      +previous = Nat.sub(U32.to_nat(end),1n)
      +step = BatchRows.row_step(call,start,end,nonempty,inside,rows_room)
      +last = last_facts(call,first,count,nonempty,inside,rows_room)
      +start_image = Nat.div(s,height)
      +final_image = Nat.div(previous,height)
      +start_rest = Nat.mod(s,height)
      +final_rest = Nat.mod(previous,height)
      +words_equal = Order.equal_of_is_eq(U32.to_nat(BatchRows.image_word(geometry,start)),U32.to_nat(BatchRows.image_word(geometry,task_last(call,first,count))),
        {%Comparison.eq(BatchRows.image_word(geometry,start),BatchRows.image_word(geometry,task_last(call,first,count))) : {_ == True{} : Bool}
          classified : {Nat.is_eq(U32.to_nat(BatchRows.image_word(geometry,start)),U32.to_nat(BatchRows.image_word(geometry,task_last(call,first,count)))) == True{} : Bool}})
      +same_image = Equal.trans(Nat,start_image,U32.to_nat(BatchRows.image_word(geometry,start)),final_image,
        Equal.sym(Nat,U32.to_nat(BatchRows.image_word(geometry,start)),start_image,BatchRows.row_image(call,start,end,step)),
        Equal.trans(Nat,U32.to_nat(BatchRows.image_word(geometry,start)),U32.to_nat(BatchRows.image_word(geometry,task_last(call,first,count))),final_image,
          words_equal,last_image(call,first,count,last)))
      +start_split = DivisionOrder.quotient_exact(s,height,positive)
      +last_split = DivisionOrder.quotient_exact(previous,height,positive)
      +offset = Nat.mul(final_image,height)
      +shifted_start = Equal.trans(Nat,s,Nat.add(Nat.mul(start_image,height),start_rest),Nat.add(offset,start_rest),start_split,
        Equal.cong(Nat,Nat,number => Nat.add(Nat.mul(number,height),start_rest),start_image,final_image,same_image))
      +ordered = {%shifted_start : {Nat.is_le(_,Nat.add(offset,final_rest)) == True{} : Bool}
        %last_split : {Nat.is_le(s,_) == True{} : Bool}
        last_after(call,first,count,last) : {Nat.is_le(Nat.add(offset,start_rest),Nat.add(offset,final_rest)) == True{} : Bool}}
      +rest_le = Order.cancel_offset_le(offset,start_rest,final_rest,ordered)
      +low_first = U32.to_nat(task_low_first(call,first))
      +rest_exact = Equal.sym(Nat,low_first,start_rest,BatchRows.row_first(call,start,end,step))
      TailFirst{
        {%rest_exact : {Nat.is_le(_,Nat.add(final_rest,1n)) == True{} : Bool}
          Order.transitive(start_rest,final_rest,Nat.add(final_rest,1n),rest_le,Order.le_add(final_rest,1n)) :
          {Nat.is_le(low_first,Nat.add(final_rest,1n)) == True{} : Bool}},
        {%rest_exact : {Nat.is_le(Nat.add(offset,_),Nat.max(s,offset)) == True{} : Bool}
          %shifted_start : {Nat.is_le(_,Nat.max(s,offset)) == True{} : Bool}
          NatExtrema.maximum_left(s,offset) : {Nat.is_le(Nat.add(offset,low_first),Nat.max(s,offset)) == True{} : Bool}}}
    case False{}:
      +height = Domain.height(Domain.geometry(call))
      +s = U32.to_nat(PieceCells.piece_start(call,first))
      +previous = Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n)
      +offset = Nat.mul(Nat.div(previous,height),height)
      TailFirst{Order.zero_le(Nat.add(Nat.mod(previous,height),1n)),
        {%Equal.sym(Nat,Nat.add(offset,0n),offset,Addition.zero_right(offset)) : {Nat.is_le(_,Nat.max(s,offset)) == True{} : Bool}
          NatExtrema.maximum_right(s,offset) : {Nat.is_le(Nat.add(offset,0n),Nat.max(s,offset)) == True{} : Bool}}}

# The rows the last image of a task reads: they stay in the image and end
# at the end of the task.
type TailFacts<-call: Domain.Call,-first: U32,-count: U32> is Data:
  TailFacts{room: {Nat.is_le(Nat.add(U32.to_nat(task_tail_first(call,first,count)),U32.to_nat(task_tail(call,first,count))),Domain.height(Domain.geometry(call))) == True{} : Bool},
    first_fact: TailFirst<call,first,count,task_tail_first(call,first,count)>,
    end: {Nat.add(Nat.mul(Nat.div(Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n),Domain.height(Domain.geometry(call))),Domain.height(Domain.geometry(call))),
      Nat.add(U32.to_nat(task_tail_first(call,first,count)),U32.to_nat(task_tail(call,first,count)))) == U32.to_nat(PieceCells.piece_end(call,first,count)) : Nat}}
def tails_room(-call: Domain.Call,-first: U32,-count: U32,facts: TailFacts<call,first,count>) ->
  {Nat.is_le(Nat.add(U32.to_nat(task_tail_first(call,first,count)),U32.to_nat(task_tail(call,first,count))),Domain.height(Domain.geometry(call))) == True{} : Bool}:
  TailFacts{room,first_fact,end} = facts
  room
def tails_first(-call: Domain.Call,-first: U32,-count: U32,facts: TailFacts<call,first,count>) -> TailFirst<call,first,count,task_tail_first(call,first,count)>:
  TailFacts{room,first_fact,end} = facts
  first_fact
def tails_end(-call: Domain.Call,-first: U32,-count: U32,facts: TailFacts<call,first,count>) ->
  {Nat.add(Nat.mul(Nat.div(Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n),Domain.height(Domain.geometry(call))),Domain.height(Domain.geometry(call))),
    Nat.add(U32.to_nat(task_tail_first(call,first,count)),U32.to_nat(task_tail(call,first,count)))) == U32.to_nat(PieceCells.piece_end(call,first,count)) : Nat}:
  TailFacts{room,first_fact,end} = facts
  end

def tail_facts(+call: Domain.Call,+first: U32,+count: U32,
  +nonempty: {Nat.is_lt(U32.to_nat(PieceCells.piece_start(call,first)),U32.to_nat(PieceCells.piece_end(call,first,count))) == True{} : Bool},
  +inside: {Nat.is_le(U32.to_nat(PieceCells.piece_end(call,first,count)),Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call)))) == True{} : Bool},
  +rows_room: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool}) ->
  TailFacts<call,first,count>:
  +geometry = Domain.geometry(call)
  +start = PieceCells.piece_start(call,first)
  +height = Domain.height(geometry)
  +positive = Domain.height_positive(call)
  +limits = Domain.limits(call)
  +cap = Machine.capacity(limits)
  +room = U32.to_nat(cap)
  +previous = Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n)
  +image = Nat.div(previous,height)
  +rest = Nat.mod(previous,height)
  +last = last_facts(call,first,count,nonempty,inside,rows_room)
  +first_fact = {tail_case(U32.is_eq(BatchRows.image_word(geometry,start),BatchRows.image_word(geometry,task_last(call,first,count))),call,first,count,
      nonempty,inside,rows_room,{==}) : TailFirst<call,first,count,task_tail_first(call,first,count)>}
  +tail_first = task_tail_first(call,first,count)
  +lead = U32.to_nat(tail_first)
  +rest_word = BatchRows.first_word(geometry,task_last(call,first,count))
  +rest_exact = last_rest(call,first,count,last)
  +height_room = Limits.dimension_capacity(limits,height,Domain.height_room(call))
  +rest_small = DivisionOrder.remainder_small(previous,height,positive)
  +up = Nat.add(rest,1n)
  +up_le = {%Equal.sym(Nat,Nat.add(rest,1n),Nat.add(1n,rest),Order.commutative(rest,1n)) : {Nat.is_le(_,height) == True{} : Bool}
    Order.strict_follow(rest,height,rest_small) : {Nat.is_le(up,height) == True{} : Bool}}
  +up_room = Order.transitive(up,height,room,up_le,height_room)
  +up_word = (rest_word + 1 : U32)
  +up_exact = Equal.trans(Nat,U32.to_nat(up_word),Nat.add(U32.to_nat(rest_word),1n),up,
    FullRange.add(rest_word,1,cap,
      {%Equal.sym(Nat,U32.to_nat(rest_word),rest,rest_exact) : {Nat.is_le(Nat.add(_,1n),room) == True{} : Bool}
        up_room : {Nat.is_le(Nat.add(U32.to_nat(rest_word),1n),room) == True{} : Bool}}),
    Equal.cong(Nat,Nat,number => Nat.add(number,1n),U32.to_nat(rest_word),rest,rest_exact))
  +below = tail_below(call,first,count,tail_first,first_fact)
  +tail_word = task_tail(call,first,count)
  +tail_exact = Equal.trans(Nat,U32.to_nat(tail_word),Nat.sub(U32.to_nat(up_word),lead),Nat.sub(up,lead),
    FullRange.subtract(up_word,tail_first,cap,
      {%Equal.sym(Nat,U32.to_nat(up_word),up,up_exact) : {Nat.is_le(lead,_) == True{} : Bool} below : {Nat.is_le(lead,U32.to_nat(up_word)) == True{} : Bool}},
      {%Equal.sym(Nat,U32.to_nat(up_word),up,up_exact) : {Nat.is_le(Nat.sub(_,lead),room) == True{} : Bool}
        Order.transitive(Nat.sub(up,lead),up,room,Order.sub_le(up,lead),up_room) : {Nat.is_le(Nat.sub(U32.to_nat(up_word),lead),room) == True{} : Bool}}),
    Equal.cong(Nat,Nat,number => Nat.sub(number,lead),U32.to_nat(up_word),up,up_exact))
  +sum = Equal.trans(Nat,Nat.add(lead,U32.to_nat(tail_word)),Nat.add(lead,Nat.sub(up,lead)),up,
    Equal.cong(Nat,Nat,number => Nat.add(lead,number),U32.to_nat(tail_word),Nat.sub(up,lead),tail_exact),
    Order.restore_shift(lead,up,below))
  +tail_room = {%Equal.sym(Nat,Nat.add(lead,U32.to_nat(tail_word)),up,sum) : {Nat.is_le(_,height) == True{} : Bool}
    up_le : {Nat.is_le(Nat.add(lead,U32.to_nat(tail_word)),height) == True{} : Bool}}
  +offset = Nat.mul(image,height)
  +end_exact = Equal.trans(Nat,Nat.add(offset,Nat.add(lead,U32.to_nat(tail_word))),Nat.add(offset,up),U32.to_nat(PieceCells.piece_end(call,first,count)),
    Equal.cong(Nat,Nat,number => Nat.add(offset,number),Nat.add(lead,U32.to_nat(tail_word)),up,sum),
    Equal.trans(Nat,Nat.add(offset,up),Nat.add(Nat.add(offset,rest),1n),U32.to_nat(PieceCells.piece_end(call,first,count)),
      Equal.sym(Nat,Nat.add(Nat.add(offset,rest),1n),Nat.add(offset,up),Addition.associative(offset,rest,1n)),
      Equal.trans(Nat,Nat.add(Nat.add(offset,rest),1n),Nat.add(previous,1n),U32.to_nat(PieceCells.piece_end(call,first,count)),
        Equal.cong(Nat,Nat,number => Nat.add(number,1n),Nat.add(offset,rest),previous,
          Equal.sym(Nat,previous,Nat.add(offset,rest),DivisionOrder.quotient_exact(previous,height,positive))),
        last_restored(call,first,count,last))))
  TailFacts{tail_room,first_fact,end_exact}

# ---- The reach of a task ----

# The first gradient cell a task reads: the first row its first image
# reads, which is a row of that image.
def reach_low(+call: Domain.Call,+first: U32,+count: U32,
  +nonempty: {Nat.is_lt(U32.to_nat(PieceCells.piece_start(call,first)),U32.to_nat(PieceCells.piece_end(call,first,count))) == True{} : Bool},
  +inside: {Nat.is_le(U32.to_nat(PieceCells.piece_end(call,first,count)),Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call)))) == True{} : Bool},
  +rows_room: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool}) ->
  KernelGemm.Both<{U32.to_nat(Windows.span_start(Kernel.reach(Rows.task_of(Domain.geometry(call)),first,count))) ==
      Nat.mul(Nat.add(Nat.mul(Nat.div(U32.to_nat(PieceCells.piece_start(call,first)),Domain.height(Domain.geometry(call))),Rows.rows_count(call)),
        Nat.min(U32.to_nat(Kernel.copy_start(Domain.geometry(call),task_low_first(call,first),task_head(call,first,count))),Rows.rows_count(call))),row_cells(call)) : Nat},
    {Nat.is_le(U32.to_nat(Windows.span_start(Kernel.reach(Rows.task_of(Domain.geometry(call)),first,count))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool}>:
  +geometry = Domain.geometry(call)
  +start = PieceCells.piece_start(call,first)
  +end = PieceCells.piece_end(call,first,count)
  +step = BatchRows.row_step(call,start,end,nonempty,inside,rows_room)
  +image = BatchRows.image_word(geometry,start)
  +rows = Rows.rows_count(call)
  +cap = Machine.capacity(Domain.limits(call))
  +height_word = Kernel.gradient_height(geometry)
  +copied = Kernel.copy_start(geometry,task_low_first(call,first),task_head(call,first,count))
  +clipped = U32.min(copied,height_word)
  +clip = Nat.min(U32.to_nat(copied),rows)
  +clip_exact = Equal.trans(Nat,U32.to_nat(clipped),Nat.min(U32.to_nat(copied),U32.to_nat(height_word)),clip,Extrema.minimum(copied,height_word),
    Equal.cong(Nat,Nat,number => Nat.min(U32.to_nat(copied),number),U32.to_nat(height_word),rows,gradient_word(call)))
  +image_n = Nat.div(U32.to_nat(start),Domain.height(geometry))
  +slot = Nat.mul(Nat.add(Nat.mul(image_n,rows),clip),row_cells(call))
  +slot_fits = slot_room(call,image_n,clip,BatchRows.row_image_inside(call,start,end,step),NatExtrema.minimum_right(U32.to_nat(copied),rows))
  +encoded = Nat.mul(Nat.add(Nat.mul(U32.to_nat(image),rows),U32.to_nat(clipped)),row_cells(call))
  +same = {%Equal.sym(Nat,U32.to_nat(image),image_n,BatchRows.row_image(call,start,end,step)) :
      {Nat.mul(Nat.add(Nat.mul(_,rows),U32.to_nat(clipped)),row_cells(call)) == slot : Nat}
    %Equal.sym(Nat,U32.to_nat(clipped),clip,clip_exact) : {Nat.mul(Nat.add(Nat.mul(image_n,rows),_),row_cells(call)) == slot : Nat}
    {==} : {encoded == slot : Nat}}
  +low_word = ((image * height_word + clipped) * Kernel.gradient_row_cells(geometry) : U32)
  +exact = {%Equal.sym(U32,low_word,U32.from_nat(encoded),scaled_word(call,image,clipped)) : {U32.to_nat(_) == slot : Nat}
    Equal.trans(Nat,U32.to_nat(U32.from_nat(encoded)),encoded,slot,
      FullRange.roundtrip(encoded,cap,{%Equal.sym(Nat,encoded,slot,same) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} slot_fits :
        {Nat.is_le(encoded,U32.to_nat(cap)) == True{} : Bool}}),same) :
    {U32.to_nat(low_word) == slot : Nat}}
  KernelGemm.Both{exact,{%Equal.sym(Nat,U32.to_nat(low_word),slot,exact) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} slot_fits :
    {Nat.is_le(U32.to_nat(low_word),U32.to_nat(cap)) == True{} : Bool}}}

# The end of a reach is at or after the end of the rows its last image reads.
def span_above(+low: U32,+high: U32,+capacity: U32,
  +low_room: {Nat.is_le(U32.to_nat(low),U32.to_nat(capacity)) == True{} : Bool},
  +high_room: {Nat.is_le(U32.to_nat(high),U32.to_nat(capacity)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(high),U32.to_nat(Windows.span_end(Tasks.Span{low,(U32.max(high,low) - low : U32)}))) == True{} : Bool}:
  +a = U32.to_nat(low)
  +b = U32.to_nat(high)
  +room = U32.to_nat(capacity)
  +most = Nat.max(b,a)
  +most_word = U32.max(high,low)
  +most_exact = Extrema.maximum(high,low)
  +most_room = NatExtrema.maximum_upper(b,a,room,high_room,low_room)
  +gap_word = (most_word - low : U32)
  +gap_exact = Equal.trans(Nat,U32.to_nat(gap_word),Nat.sub(U32.to_nat(most_word),a),Nat.sub(most,a),
    FullRange.subtract(most_word,low,capacity,
      {%Equal.sym(Nat,U32.to_nat(most_word),most,most_exact) : {Nat.is_le(a,_) == True{} : Bool} NatExtrema.maximum_right(b,a) :
        {Nat.is_le(a,U32.to_nat(most_word)) == True{} : Bool}},
      {%Equal.sym(Nat,U32.to_nat(most_word),most,most_exact) : {Nat.is_le(Nat.sub(_,a),room) == True{} : Bool}
        Order.transitive(Nat.sub(most,a),most,room,Order.sub_le(most,a),most_room) : {Nat.is_le(Nat.sub(U32.to_nat(most_word),a),room) == True{} : Bool}}),
    Equal.cong(Nat,Nat,number => Nat.sub(number,a),U32.to_nat(most_word),most,most_exact))
  +joined = Equal.trans(Nat,Nat.add(a,U32.to_nat(gap_word)),Nat.add(a,Nat.sub(most,a)),Nat.max(a,most),
    Equal.cong(Nat,Nat,number => Nat.add(a,number),U32.to_nat(gap_word),Nat.sub(most,a),gap_exact),NatExtrema.clipped_end(a,most))
  +joined_room = {%Equal.sym(Nat,Nat.add(a,U32.to_nat(gap_word)),Nat.max(a,most),joined) : {Nat.is_le(_,room) == True{} : Bool}
    NatExtrema.maximum_upper(a,most,room,low_room,most_room) : {Nat.is_le(Nat.add(a,U32.to_nat(gap_word)),room) == True{} : Bool}}
  +end_exact = Equal.trans(Nat,U32.to_nat((low + gap_word : U32)),Nat.add(a,U32.to_nat(gap_word)),Nat.max(a,most),
    FullRange.add(low,gap_word,capacity,joined_room),joined)
  %Equal.sym(Nat,U32.to_nat((low + gap_word : U32)),Nat.max(a,most),end_exact) : {Nat.is_le(b,_) == True{} : Bool}
  Order.transitive(b,most,Nat.max(a,most),NatExtrema.maximum_left(b,a),NatExtrema.maximum_right(a,most))

# The high end of the reach of a task: the end of the gradient rows its last
# image reads, inside the gradient cells.
def high_bound(+call: Domain.Call,+envelope: Geometry.WordEnvelope,+first: U32,+count: U32,
  +nonempty: {Nat.is_lt(U32.to_nat(PieceCells.piece_start(call,first)),U32.to_nat(PieceCells.piece_end(call,first,count))) == True{} : Bool},
  +inside: {Nat.is_le(U32.to_nat(PieceCells.piece_end(call,first,count)),Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call)))) == True{} : Bool},
  +rows_room: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +tails: TailFacts<call,first,count>) ->
  KernelGemm.Both<{U32.to_nat(((BatchRows.image_word(Domain.geometry(call),task_last(call,first,count)) * Kernel.gradient_height(Domain.geometry(call)) +
    Kernel.copy_stop(Domain.geometry(call),task_tail_first(call,first,count),task_tail(call,first,count))) * Kernel.gradient_row_cells(Domain.geometry(call)) : U32)) ==
    Nat.mul(Nat.add(Nat.mul(Nat.div(Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n),Domain.height(Domain.geometry(call))),Rows.rows_count(call)),
    U32.to_nat(Kernel.copy_stop(Domain.geometry(call),task_tail_first(call,first,count),task_tail(call,first,count)))),row_cells(call)) : Nat},
    {Nat.is_le(Nat.mul(Nat.add(Nat.mul(Nat.div(Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n),Domain.height(Domain.geometry(call))),Rows.rows_count(call)),
    U32.to_nat(Kernel.copy_stop(Domain.geometry(call),task_tail_first(call,first,count),task_tail(call,first,count)))),row_cells(call)),Rows.cells_count(call)) == True{} : Bool}>:
  +geometry = Domain.geometry(call)
  +cap = Machine.capacity(Domain.limits(call))
  +last = last_facts(call,first,count,nonempty,inside,rows_room)
  +image = BatchRows.image_word(geometry,task_last(call,first,count))
  +image_n = Nat.div(Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n),Domain.height(geometry))
  +rows = Rows.rows_count(call)
  +stopped = Kernel.copy_stop(geometry,task_tail_first(call,first,count),task_tail(call,first,count))
  +stop_le = stop_below(call,envelope,task_tail_first(call,first,count),task_tail(call,first,count),tails_room(call,first,count,tails))
  +slot = Nat.mul(Nat.add(Nat.mul(image_n,rows),U32.to_nat(stopped)),row_cells(call))
  +slot_inside_cells = slot_inside(call,image_n,U32.to_nat(stopped),last_inside(call,first,count,last),stop_le)
  +slot_fits = Order.transitive(slot,Rows.cells_count(call),U32.to_nat(cap),slot_inside_cells,Rows.cells_room(call))
  +encoded = Nat.mul(Nat.add(Nat.mul(U32.to_nat(image),rows),U32.to_nat(stopped)),row_cells(call))
  +same = {%Equal.sym(Nat,U32.to_nat(image),image_n,last_image(call,first,count,last)) :
      {Nat.mul(Nat.add(Nat.mul(_,rows),U32.to_nat(stopped)),row_cells(call)) == slot : Nat}
    {==} : {encoded == slot : Nat}}
  +high_word = ((image * Kernel.gradient_height(geometry) + stopped) * Kernel.gradient_row_cells(geometry) : U32)
  +high_exact = {%Equal.sym(U32,high_word,U32.from_nat(encoded),scaled_word(call,image,stopped)) : {U32.to_nat(_) == slot : Nat}
    Equal.trans(Nat,U32.to_nat(U32.from_nat(encoded)),encoded,slot,
      FullRange.roundtrip(encoded,cap,{%Equal.sym(Nat,encoded,slot,same) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} slot_fits :
        {Nat.is_le(encoded,U32.to_nat(cap)) == True{} : Bool}}),same) :
    {U32.to_nat(high_word) == slot : Nat}}
  KernelGemm.Both{high_exact,slot_inside_cells}

# The end of the reach of a task is at or after the end of the gradient rows
# its last image reads.
def reach_high(+call: Domain.Call,+envelope: Geometry.WordEnvelope,+first: U32,+count: U32,
  +nonempty: {Nat.is_lt(U32.to_nat(PieceCells.piece_start(call,first)),U32.to_nat(PieceCells.piece_end(call,first,count))) == True{} : Bool},
  +inside: {Nat.is_le(U32.to_nat(PieceCells.piece_end(call,first,count)),Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call)))) == True{} : Bool},
  +rows_room: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +tails: TailFacts<call,first,count>) ->
  {Nat.is_le(Nat.mul(Nat.add(Nat.mul(Nat.div(Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n),Domain.height(Domain.geometry(call))),Rows.rows_count(call)),
    U32.to_nat(Kernel.copy_stop(Domain.geometry(call),task_tail_first(call,first,count),task_tail(call,first,count)))),row_cells(call)),
    U32.to_nat(Windows.span_end(Kernel.reach(Rows.task_of(Domain.geometry(call)),first,count)))) == True{} : Bool}:
  +geometry = Domain.geometry(call)
  +start = PieceCells.piece_start(call,first)
  +cap = Machine.capacity(Domain.limits(call))
  +rows = Rows.rows_count(call)
  +slot = Nat.mul(Nat.add(Nat.mul(Nat.div(Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n),Domain.height(Domain.geometry(call))),Rows.rows_count(call)),
    U32.to_nat(Kernel.copy_stop(Domain.geometry(call),task_tail_first(call,first,count),task_tail(call,first,count)))),row_cells(call))
  +high_word = ((BatchRows.image_word(Domain.geometry(call),task_last(call,first,count)) * Kernel.gradient_height(Domain.geometry(call)) +
    Kernel.copy_stop(Domain.geometry(call),task_tail_first(call,first,count),task_tail(call,first,count))) * Kernel.gradient_row_cells(Domain.geometry(call)) : U32)
  +bound = high_bound(call,envelope,first,count,nonempty,inside,rows_room,tails)
  +high_exact = Shape.first({U32.to_nat(high_word) == slot : Nat},{Nat.is_le(slot,Rows.cells_count(call)) == True{} : Bool},bound)
  +high_room = {%Equal.sym(Nat,U32.to_nat(high_word),slot,high_exact) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
    Order.transitive(slot,Rows.cells_count(call),U32.to_nat(cap),Shape.second({U32.to_nat(high_word) == slot : Nat},{Nat.is_le(slot,Rows.cells_count(call)) == True{} : Bool},bound),
      Rows.cells_room(call)) :
    {Nat.is_le(U32.to_nat(high_word),U32.to_nat(cap)) == True{} : Bool}}
  +low_word = Windows.span_start(Kernel.reach(Rows.task_of(geometry),first,count))
  +low_room = Shape.second({U32.to_nat(low_word) ==
      Nat.mul(Nat.add(Nat.mul(Nat.div(U32.to_nat(start),Domain.height(geometry)),rows),
        Nat.min(U32.to_nat(Kernel.copy_start(geometry,task_low_first(call,first),task_head(call,first,count))),rows)),row_cells(call)) : Nat},
    {Nat.is_le(U32.to_nat(low_word),U32.to_nat(cap)) == True{} : Bool},reach_low(call,first,count,nonempty,inside,rows_room))
  %high_exact : {Nat.is_le(_,U32.to_nat(Windows.span_end(Kernel.reach(Rows.task_of(geometry),first,count)))) == True{} : Bool}
  span_above(low_word,high_word,cap,low_room,high_room)

# ---- The band of a call ----

# The rows of a number of images and the rows after them: the image of the
# rows and their row in it.
def image_of(+image: Nat,+row: Nat,+height: Nat,+positive: {Nat.is_lt(0n,height) == True{} : Bool},+below: {Nat.is_lt(row,height) == True{} : Bool}) ->
  {Nat.div(Nat.add(Nat.mul(image,height),row),height) == image : Nat}:
  Radix.quotient(image,row,height,positive,below)

# The rows of the next image start a height after the rows of an image.
def next_rows(+image: Nat,+height: Nat) -> {Nat.mul(Nat.add(image,1n),height) == Nat.add(Nat.mul(image,height),height) : Nat}:
  Equal.trans(Nat,Nat.mul(Nat.add(image,1n),height),Nat.add(Nat.mul(image,height),Nat.mul(1n,height)),Nat.add(Nat.mul(image,height),height),
    RowMajor.distrib(image,1n,height),Equal.cong(Nat,Nat,number => Nat.add(Nat.mul(image,height),number),Nat.mul(1n,height),height,one_times(height)))

# A slot in an image before a later image is before that later image.
def earlier_slot(+image: Nat,+later: Nat,+rows: Nat,+slot: Nat,+extra: Nat,+less: {Nat.is_lt(image,later) == True{} : Bool},
  +slot_le: {Nat.is_le(slot,rows) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(image,rows),slot),Nat.add(Nat.mul(later,rows),extra)) == True{} : Bool}:
  +next = Nat.add(image,1n)
  +next_le = {%Equal.sym(Nat,Nat.add(image,1n),Nat.add(1n,image),Order.commutative(image,1n)) : {Nat.is_le(_,later) == True{} : Bool}
    Order.strict_follow(image,later,less) : {Nat.is_le(next,later) == True{} : Bool}}
  Order.transitive(Nat.add(Nat.mul(image,rows),slot),Nat.mul(next,rows),Nat.add(Nat.mul(later,rows),extra),
    {%Equal.sym(Nat,Nat.mul(next,rows),Nat.add(Nat.mul(image,rows),rows),next_rows(image,rows)) : {Nat.is_le(Nat.add(Nat.mul(image,rows),slot),_) == True{} : Bool}
      Encoded.sum_bounds(Nat.mul(image,rows),slot,Nat.mul(image,rows),rows,Order.reflexive(Nat.mul(image,rows)),slot_le) :
      {Nat.is_le(Nat.add(Nat.mul(image,rows),slot),Nat.mul(next,rows)) == True{} : Bool}},
    Order.transitive(Nat.mul(next,rows),Nat.mul(later,rows),Nat.add(Nat.mul(later,rows),extra),RowMajor.mul_le(next,later,rows,next_le),
      Order.le_add(Nat.mul(later,rows),extra)))

# The first gradient row a call copies is at or after the first row of the
# reach of its task.
def low_case(later: Bool,+call: Domain.Call,+envelope: Geometry.WordEnvelope,+first: U32,+count: U32,
  +nonempty: {Nat.is_lt(U32.to_nat(PieceCells.piece_start(call,first)),U32.to_nat(PieceCells.piece_end(call,first,count))) == True{} : Bool},
  +inside: {Nat.is_le(U32.to_nat(PieceCells.piece_end(call,first,count)),Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call)))) == True{} : Bool},
  +rows_room: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +image: U32,+row: U32,+rows: U32,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(row),U32.to_nat(rows)),Domain.height(Domain.geometry(call))) == True{} : Bool},
  +row_below: {Nat.is_lt(U32.to_nat(row),Domain.height(Domain.geometry(call))) == True{} : Bool},
  +after: {Nat.is_le(U32.to_nat(PieceCells.piece_start(call,first)),Nat.add(Nat.mul(U32.to_nat(image),Domain.height(Domain.geometry(call))),U32.to_nat(row))) == True{} : Bool},
  +before: {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(image),Domain.height(Domain.geometry(call))),Nat.add(U32.to_nat(row),U32.to_nat(rows))),
    U32.to_nat(PieceCells.piece_end(call,first,count))) == True{} : Bool},
  +classified: {Nat.is_lt(Nat.div(U32.to_nat(PieceCells.piece_start(call,first)),Domain.height(Domain.geometry(call))),U32.to_nat(image)) == later : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(Nat.div(U32.to_nat(PieceCells.piece_start(call,first)),Domain.height(Domain.geometry(call))),Rows.rows_count(call)),
    Nat.min(U32.to_nat(Kernel.copy_start(Domain.geometry(call),task_low_first(call,first),task_head(call,first,count))),Rows.rows_count(call))),
    Nat.add(Nat.mul(U32.to_nat(image),Rows.rows_count(call)),U32.to_nat(CopyRows.begin(Bands.caller_query(call,envelope,row,rows,contained))))) == True{} : Bool}:
  match later:
    case True{}:
      +starting = Nat.div(U32.to_nat(PieceCells.piece_start(call,first)),Domain.height(Domain.geometry(call)))
      +gradient = Rows.rows_count(call)
      +copied = U32.to_nat(Kernel.copy_start(Domain.geometry(call),task_low_first(call,first),task_head(call,first,count)))
      earlier_slot(starting,U32.to_nat(image),gradient,Nat.min(copied,gradient),U32.to_nat(CopyRows.begin(Bands.caller_query(call,envelope,row,rows,contained))),
        classified,NatExtrema.minimum_right(copied,gradient))
    case False{}:
      +geometry = Domain.geometry(call)
      +start = PieceCells.piece_start(call,first)
      +end = PieceCells.piece_end(call,first,count)
      +height = Domain.height(geometry)
      +positive = Domain.height_positive(call)
      +gradient = Rows.rows_count(call)
      +s = U32.to_nat(start)
      +e = U32.to_nat(end)
      +n = U32.to_nat(image)
      +f = U32.to_nat(row)
      +c = U32.to_nat(rows)
      +starting = Nat.div(s,height)
      +rest = Nat.mod(s,height)
      +step = BatchRows.row_step(call,start,end,nonempty,inside,rows_room)
      +head = task_head(call,first,count)
      +low_first = task_low_first(call,first)
      +rest_exact = BatchRows.row_first(call,start,end,step)
      +head_room = {%Equal.sym(Nat,U32.to_nat(low_first),rest,rest_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(head)),height) == True{} : Bool}
        BatchRows.row_contained(call,start,end,step) : {Nat.is_le(Nat.add(U32.to_nat(low_first),U32.to_nat(head)),height) == True{} : Bool}}
      +up_bound = Order.transitive(starting,Nat.div(Nat.add(Nat.mul(n,height),f),height),n,DivisionOrder.quotient_le(s,Nat.add(Nat.mul(n,height),f),height,positive,after),
        {%Equal.sym(Nat,Nat.div(Nat.add(Nat.mul(n,height),f),height),n,image_of(n,f,height,positive,row_below)) :
          {Nat.is_le(_,n) == True{} : Bool} Order.reflexive(n) : {Nat.is_le(Nat.div(Nat.add(Nat.mul(n,height),f),height),n) == True{} : Bool}})
      +same = Order.antisymmetric(starting,n,up_bound,Order.not_less(starting,n,classified))
      +split = DivisionOrder.quotient_exact(s,height,positive)
      +offset = Nat.mul(n,height)
      +shifted = Equal.trans(Nat,s,Nat.add(Nat.mul(starting,height),rest),Nat.add(offset,rest),split,
        Equal.cong(Nat,Nat,number => Nat.add(Nat.mul(number,height),rest),starting,n,same))
      +lower_rest = Order.cancel_offset_le(offset,rest,f,{%shifted : {Nat.is_le(_,Nat.add(offset,f)) == True{} : Bool} after :
        {Nat.is_le(Nat.add(offset,rest),Nat.add(offset,f)) == True{} : Bool}})
      +lower = {%Equal.sym(Nat,U32.to_nat(low_first),rest,rest_exact) : {Nat.is_le(_,f) == True{} : Bool} lower_rest :
        {Nat.is_le(U32.to_nat(low_first),f) == True{} : Bool}}
      +limit = Nat.mul(Nat.add(starting,1n),height)
      +least = Nat.min(e,limit)
      +limit_split = Equal.trans(Nat,limit,Nat.add(Nat.mul(starting,height),height),Nat.add(offset,height),next_rows(starting,height),
        Equal.cong(Nat,Nat,number => Nat.add(Nat.mul(number,height),height),starting,n,same))
      +start_below = {%Equal.sym(Nat,limit,Nat.add(offset,height),limit_split) : {Nat.is_le(s,_) == True{} : Bool}
        %Equal.sym(Nat,s,Nat.add(offset,rest),shifted) : {Nat.is_le(_,Nat.add(offset,height)) == True{} : Bool}
        Order.offset_le(offset,rest,height,Bounds.strict_le(rest,height,DivisionOrder.remainder_small(s,height,positive))) :
        {Nat.is_le(s,limit) == True{} : Bool}}
      +start_least = NatExtrema.minimum_lower(e,limit,s,Bounds.strict_le(s,e,nonempty),start_below)
      +head_exact = BatchRows.row_count(call,start,end,step)
      +head_sum = Equal.trans(Nat,Nat.add(s,U32.to_nat(head)),Nat.add(s,Nat.sub(least,s)),least,
        Equal.cong(Nat,Nat,number => Nat.add(s,number),U32.to_nat(head),Nat.sub(least,s),head_exact),Order.restore_shift(s,least,start_least))
      +call_end = Nat.add(offset,Nat.add(f,c))
      +call_least = NatExtrema.minimum_lower(e,limit,call_end,before,
        {%Equal.sym(Nat,limit,Nat.add(offset,height),limit_split) : {Nat.is_le(call_end,_) == True{} : Bool}
          Order.offset_le(offset,Nat.add(f,c),height,contained) : {Nat.is_le(call_end,limit) == True{} : Bool}})
      +upper_shifted = {%Addition.associative(offset,rest,U32.to_nat(head)) : {Nat.is_le(call_end,_) == True{} : Bool}
        %shifted : {Nat.is_le(call_end,Nat.add(_,U32.to_nat(head))) == True{} : Bool}
        %Equal.sym(Nat,Nat.add(s,U32.to_nat(head)),least,head_sum) : {Nat.is_le(call_end,_) == True{} : Bool}
        call_least : {Nat.is_le(call_end,Nat.add(offset,Nat.add(rest,U32.to_nat(head)))) == True{} : Bool}}
      +upper_rest = Order.cancel_offset_le(offset,Nat.add(f,c),Nat.add(rest,U32.to_nat(head)),upper_shifted)
      +upper = {%Equal.sym(Nat,U32.to_nat(low_first),rest,rest_exact) : {Nat.is_le(Nat.add(f,c),Nat.add(_,U32.to_nat(head))) == True{} : Bool}
        upper_rest : {Nat.is_le(Nat.add(f,c),Nat.add(U32.to_nat(low_first),U32.to_nat(head))) == True{} : Bool}}
      +pair = {BandOrder.Nested{call,envelope,low_first,head,head_room,row,rows,contained,lower,upper} : BandOrder.Nested}
      +parent = Bands.caller_query(call,envelope,low_first,head,head_room)
      +child = Bands.caller_query(call,envelope,row,rows,contained)
      +copied = U32.to_nat(Kernel.copy_start(geometry,low_first,head))
      +begin_le = {%Equal.sym(U32,Kernel.copy_start(geometry,low_first,head),CopyRows.begin(parent),Rows.start_equal(call,envelope,low_first,head,head_room)) :
          {Nat.is_le(U32.to_nat(_),U32.to_nat(CopyRows.begin(child))) == True{} : Bool}
        BandOrder.begin_monotone(pair) : {Nat.is_le(copied,U32.to_nat(CopyRows.begin(child))) == True{} : Bool}}
      %Equal.sym(Nat,starting,n,same) : {Nat.is_le(Nat.add(Nat.mul(_,gradient),Nat.min(copied,gradient)),Nat.add(Nat.mul(n,gradient),U32.to_nat(CopyRows.begin(child)))) == True{} : Bool}
      Order.offset_le(Nat.mul(n,gradient),Nat.min(copied,gradient),U32.to_nat(CopyRows.begin(child)),
        Order.transitive(Nat.min(copied,gradient),copied,U32.to_nat(CopyRows.begin(child)),NatExtrema.minimum_left(copied,gradient),begin_le))

# The gradient row after the last one a call copies is at or before the
# end of the rows the last image of its task reads.
def high_case(later: Bool,+call: Domain.Call,+envelope: Geometry.WordEnvelope,+first: U32,+count: U32,
  +nonempty: {Nat.is_lt(U32.to_nat(PieceCells.piece_start(call,first)),U32.to_nat(PieceCells.piece_end(call,first,count))) == True{} : Bool},
  +inside: {Nat.is_le(U32.to_nat(PieceCells.piece_end(call,first,count)),Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call)))) == True{} : Bool},
  +rows_room: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +image: U32,+row: U32,+rows: U32,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(row),U32.to_nat(rows)),Domain.height(Domain.geometry(call))) == True{} : Bool},
  +row_below: {Nat.is_lt(U32.to_nat(row),Domain.height(Domain.geometry(call))) == True{} : Bool},
  +after: {Nat.is_le(U32.to_nat(PieceCells.piece_start(call,first)),Nat.add(Nat.mul(U32.to_nat(image),Domain.height(Domain.geometry(call))),U32.to_nat(row))) == True{} : Bool},
  +row_before: {Nat.is_lt(Nat.add(Nat.mul(U32.to_nat(image),Domain.height(Domain.geometry(call))),U32.to_nat(row)),U32.to_nat(PieceCells.piece_end(call,first,count))) == True{} : Bool},
  +before: {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(image),Domain.height(Domain.geometry(call))),Nat.add(U32.to_nat(row),U32.to_nat(rows))),
    U32.to_nat(PieceCells.piece_end(call,first,count))) == True{} : Bool},
  +classified: {Nat.is_lt(U32.to_nat(image),Nat.div(Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n),Domain.height(Domain.geometry(call)))) == later : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(image),Rows.rows_count(call)),U32.to_nat(CopyRows.stop(Bands.caller_query(call,envelope,row,rows,contained)))),
    Nat.add(Nat.mul(Nat.div(Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n),Domain.height(Domain.geometry(call))),Rows.rows_count(call)),
      U32.to_nat(Kernel.copy_stop(Domain.geometry(call),task_tail_first(call,first,count),task_tail(call,first,count))))) == True{} : Bool}:
  match later:
    case True{}:
      +gradient = Rows.rows_count(call)
      +child = Bands.caller_query(call,envelope,row,rows,contained)
      +ending = Nat.div(Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n),Domain.height(Domain.geometry(call)))
      +stopped = U32.to_nat(Kernel.copy_stop(Domain.geometry(call),task_tail_first(call,first,count),task_tail(call,first,count)))
      earlier_slot(U32.to_nat(image),ending,gradient,U32.to_nat(CopyRows.stop(child)),stopped,classified,
        {%stop_equal(call,envelope,row,rows,contained) : {Nat.is_le(U32.to_nat(_),gradient) == True{} : Bool}
          stop_below(call,envelope,row,rows,contained) : {Nat.is_le(U32.to_nat(CopyRows.stop(child)),gradient) == True{} : Bool}})
    case False{}:
      +geometry = Domain.geometry(call)
      +height = Domain.height(geometry)
      +positive = Domain.height_positive(call)
      +gradient = Rows.rows_count(call)
      +s = U32.to_nat(PieceCells.piece_start(call,first))
      +e = U32.to_nat(PieceCells.piece_end(call,first,count))
      +n = U32.to_nat(image)
      +f = U32.to_nat(row)
      +c = U32.to_nat(rows)
      +previous = Nat.sub(e,1n)
      +ending = Nat.div(previous,height)
      +last = last_facts(call,first,count,nonempty,inside,rows_room)
      +tails = tail_facts(call,first,count,nonempty,inside,rows_room)
      +tail_first = task_tail_first(call,first,count)
      +tail = task_tail(call,first,count)
      +lead = U32.to_nat(tail_first)
      +span = U32.to_nat(tail)
      +slot = Nat.add(Nat.mul(n,height),f)
      +one_previous = Equal.trans(Nat,Nat.add(1n,previous),Nat.add(previous,1n),e,
        Equal.sym(Nat,Nat.add(previous,1n),Nat.add(1n,previous),Order.commutative(previous,1n)),last_restored(call,first,count,last))
      +slot_previous = Division.lt_succ_le(slot,previous,{%Equal.sym(Nat,Nat.add(1n,previous),e,one_previous) : {Nat.is_lt(slot,_) == True{} : Bool} row_before :
        {Nat.is_lt(slot,1n+previous) == True{} : Bool}})
      +image_le = {%image_of(n,f,height,positive,row_below) : {Nat.is_le(_,ending) == True{} : Bool}
        DivisionOrder.quotient_le(slot,previous,height,positive,slot_previous) : {Nat.is_le(n,ending) == True{} : Bool}}
      +same = Order.antisymmetric(n,ending,image_le,Order.not_less(n,ending,classified))
      +offset = Nat.mul(ending,height)
      +slot_exact = Equal.cong(Nat,Nat,number => Nat.add(Nat.mul(number,height),f),n,ending,same)
      +start_le = Order.transitive(Nat.add(offset,lead),Nat.max(s,offset),Nat.add(offset,f),tail_start(call,first,count,tail_first,tails_first(call,first,count,tails)),
        NatExtrema.maximum_upper(s,offset,Nat.add(offset,f),{%slot_exact : {Nat.is_le(s,_) == True{} : Bool} after : {Nat.is_le(s,Nat.add(offset,f)) == True{} : Bool}},
          Order.le_add(offset,f)))
      +lower = Order.cancel_offset_le(offset,lead,f,start_le)
      +upper_shift = {%Equal.sym(Nat,Nat.add(offset,Nat.add(lead,span)),e,tails_end(call,first,count,tails)) : {Nat.is_le(Nat.add(offset,Nat.add(f,c)),_) == True{} : Bool}
        %Equal.cong(Nat,Nat,number => Nat.add(Nat.mul(number,height),Nat.add(f,c)),n,ending,same) : {Nat.is_le(_,e) == True{} : Bool}
        before : {Nat.is_le(Nat.add(offset,Nat.add(f,c)),Nat.add(offset,Nat.add(lead,span))) == True{} : Bool}}
      +upper = Order.cancel_offset_le(offset,Nat.add(f,c),Nat.add(lead,span),upper_shift)
      +room = tails_room(call,first,count,tails)
      +pair = {BandOrder.Nested{call,envelope,tail_first,tail,room,row,rows,contained,lower,upper} : BandOrder.Nested}
      +parent = Bands.caller_query(call,envelope,tail_first,tail,room)
      +child = Bands.caller_query(call,envelope,row,rows,contained)
      +stopped = U32.to_nat(Kernel.copy_stop(geometry,tail_first,tail))
      +stop_le = {%Equal.sym(U32,Kernel.copy_stop(geometry,tail_first,tail),CopyRows.stop(parent),stop_equal(call,envelope,tail_first,tail,room)) :
          {Nat.is_le(U32.to_nat(CopyRows.stop(child)),U32.to_nat(_)) == True{} : Bool}
        BandOrder.stop_monotone(pair) : {Nat.is_le(U32.to_nat(CopyRows.stop(child)),stopped) == True{} : Bool}}
      %Equal.sym(Nat,n,ending,same) : {Nat.is_le(Nat.add(Nat.mul(_,gradient),U32.to_nat(CopyRows.stop(child))),Nat.add(Nat.mul(ending,gradient),stopped)) == True{} : Bool}
      Order.offset_le(Nat.mul(ending,gradient),U32.to_nat(CopyRows.stop(child)),stopped,stop_le)

# ---- The band of a call lies in the holding ----

def placed_case(nonempty_band: Bool,+call: Domain.Call,+envelope: Geometry.WordEnvelope,+first: U32,+count: U32,
  +nonempty: {Nat.is_lt(U32.to_nat(PieceCells.piece_start(call,first)),U32.to_nat(PieceCells.piece_end(call,first,count))) == True{} : Bool},
  +inside: {Nat.is_le(U32.to_nat(PieceCells.piece_end(call,first,count)),Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call)))) == True{} : Bool},
  +rows_room: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +high: Nat,+reach_end: {Nat.is_le(U32.to_nat(Windows.span_end(Kernel.reach(Rows.task_of(Domain.geometry(call)),first,count))),high) == True{} : Bool},
  +image: U32,+row: U32,+rows: U32,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(row),U32.to_nat(rows)),Domain.height(Domain.geometry(call))) == True{} : Bool},
  +row_below: {Nat.is_lt(U32.to_nat(row),Domain.height(Domain.geometry(call))) == True{} : Bool},
  +image_inside: {Nat.is_lt(U32.to_nat(image),Domain.batch(call)) == True{} : Bool},
  +after: {Nat.is_le(U32.to_nat(PieceCells.piece_start(call,first)),Nat.add(Nat.mul(U32.to_nat(image),Domain.height(Domain.geometry(call))),U32.to_nat(row))) == True{} : Bool},
  +row_before: {Nat.is_lt(Nat.add(Nat.mul(U32.to_nat(image),Domain.height(Domain.geometry(call))),U32.to_nat(row)),U32.to_nat(PieceCells.piece_end(call,first,count))) == True{} : Bool},
  +before: {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(image),Domain.height(Domain.geometry(call))),Nat.add(U32.to_nat(row),U32.to_nat(rows))),
    U32.to_nat(PieceCells.piece_end(call,first,count))) == True{} : Bool},
  +classified: {U32.is_lt(CopyRows.begin(Bands.caller_query(call,envelope,row,rows,contained)),CopyRows.stop(Bands.caller_query(call,envelope,row,rows,contained))) == nonempty_band : Bool}) ->
  BandSource.Placed<U32.to_nat(Windows.span_start(Kernel.reach(Rows.task_of(Domain.geometry(call)),first,count))),high,
    Rows.band_first(Domain.geometry(call),image,row,rows),Rows.band_cells(Domain.geometry(call),row,rows)>:
  match nonempty_band:
    case False{}:
      +child = Bands.caller_query(call,envelope,row,rows,contained)
      +size = row_cells(call)
      +zero_count = Equal.cong(Bool,U32,less => Bool.pick(U32,less,(CopyRows.stop(child) - CopyRows.begin(child) : U32),0),
        U32.is_lt(CopyRows.begin(child),CopyRows.stop(child)),False{},classified)
      +positive = Order.le_lt_trans(0n,U32.to_nat(image),Domain.batch(call),Order.zero_le(U32.to_nat(image)),image_inside)
      +cells_zero = Equal.trans(Nat,U32.to_nat(Rows.band_cells(Domain.geometry(call),row,rows)),Nat.mul(U32.to_nat(CopyRows.row_count(child)),size),0n,
        Rows.band_count_exact(call,envelope,row,rows,contained,positive),
        Equal.cong(U32,Nat,word => Nat.mul(U32.to_nat(word),size),CopyRows.row_count(child),0,zero_count))
      BandSource.EmptyBand{Arithmetic.to_nat_injective(Rows.band_cells(Domain.geometry(call),row,rows),0,cells_zero)}
    case True{}:
      +geometry = Domain.geometry(call)
      +child = Bands.caller_query(call,envelope,row,rows,contained)
      +cap = Machine.capacity(Domain.limits(call))
      +gradient = Rows.rows_count(call)
      +size = row_cells(call)
      +n = U32.to_nat(image)
      +begun = U32.to_nat(CopyRows.begin(child))
      +stopped = U32.to_nat(CopyRows.stop(child))
      +ordered = Bounds.as_nat_lt(CopyRows.begin(child),CopyRows.stop(child),classified)
      +stop_le = {%stop_equal(call,envelope,row,rows,contained) : {Nat.is_le(U32.to_nat(_),gradient) == True{} : Bool}
        stop_below(call,envelope,row,rows,contained) : {Nat.is_le(stopped,gradient) == True{} : Bool}}
      +begin_le = Order.transitive(begun,stopped,gradient,Bounds.strict_le(begun,stopped,ordered),stop_le)
      +offset = Nat.mul(n,gradient)
      +first_slot = Nat.mul(Nat.add(offset,begun),size)
      +band_first = Rows.band_first(geometry,image,row,rows)
      +band_cells = Rows.band_cells(geometry,row,rows)
      +first_exact = Equal.trans(Nat,U32.to_nat(band_first),U32.to_nat(U32.from_nat(first_slot)),first_slot,
        Equal.cong(U32,Nat,word => U32.to_nat(word),band_first,U32.from_nat(first_slot),Rows.band_first_exact(call,envelope,image,row,rows,contained)),
        FullRange.roundtrip(first_slot,cap,slot_room(call,n,begun,image_inside,begin_le)))
      +gap = Nat.sub(stopped,begun)
      +positive = Order.le_lt_trans(0n,n,Domain.batch(call),Order.zero_le(n),image_inside)
      +cells_exact = Equal.trans(Nat,U32.to_nat(band_cells),Nat.mul(U32.to_nat(CopyRows.row_count(child)),size),Nat.mul(gap,size),
        Rows.band_count_exact(call,envelope,row,rows,contained,positive),
        Equal.cong(Nat,Nat,number => Nat.mul(number,size),U32.to_nat(CopyRows.row_count(child)),gap,CopyRows.count_exact(child)))
      +slot_end = Nat.add(offset,stopped)
      +joined = Equal.trans(Nat,Nat.add(first_slot,Nat.mul(gap,size)),Nat.mul(Nat.add(Nat.add(offset,begun),gap),size),Nat.mul(slot_end,size),
        Equal.sym(Nat,Nat.mul(Nat.add(Nat.add(offset,begun),gap),size),Nat.add(first_slot,Nat.mul(gap,size)),RowMajor.distrib(Nat.add(offset,begun),gap,size)),
        Equal.cong(Nat,Nat,number => Nat.mul(number,size),Nat.add(Nat.add(offset,begun),gap),slot_end,
          Equal.trans(Nat,Nat.add(Nat.add(offset,begun),gap),Nat.add(offset,Nat.add(begun,gap)),slot_end,Addition.associative(offset,begun,gap),
            Equal.cong(Nat,Nat,number => Nat.add(offset,number),Nat.add(begun,gap),stopped,Order.restore_shift(begun,stopped,Bounds.strict_le(begun,stopped,ordered))))))
      +total = Equal.trans(Nat,Nat.add(U32.to_nat(band_first),U32.to_nat(band_cells)),Nat.add(first_slot,U32.to_nat(band_cells)),Nat.mul(slot_end,size),
        Equal.cong(Nat,Nat,number => Nat.add(number,U32.to_nat(band_cells)),U32.to_nat(band_first),first_slot,first_exact),
        Equal.trans(Nat,Nat.add(first_slot,U32.to_nat(band_cells)),Nat.add(first_slot,Nat.mul(gap,size)),Nat.mul(slot_end,size),
          Equal.cong(Nat,Nat,number => Nat.add(first_slot,number),U32.to_nat(band_cells),Nat.mul(gap,size),cells_exact),joined))
      +start = PieceCells.piece_start(call,first)
      +starting = Nat.div(U32.to_nat(start),Domain.height(geometry))
      +copied = U32.to_nat(Kernel.copy_start(geometry,task_low_first(call,first),task_head(call,first,count)))
      +low_slot = Nat.add(Nat.mul(starting,gradient),Nat.min(copied,gradient))
      +low_word = Windows.span_start(Kernel.reach(Rows.task_of(geometry),first,count))
      +low_exact = Shape.first({U32.to_nat(low_word) == Nat.mul(low_slot,size) : Nat},{Nat.is_le(U32.to_nat(low_word),U32.to_nat(cap)) == True{} : Bool},
        reach_low(call,first,count,nonempty,inside,rows_room))
      +low_le = low_case(Nat.is_lt(starting,n),call,envelope,first,count,nonempty,inside,rows_room,image,row,rows,contained,row_below,after,before,{==})
      +after_band = {%Equal.sym(Nat,U32.to_nat(band_first),first_slot,first_exact) : {Nat.is_le(U32.to_nat(low_word),_) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(low_word),Nat.mul(low_slot,size),low_exact) : {Nat.is_le(_,first_slot) == True{} : Bool}
        RowMajor.mul_le(low_slot,Nat.add(offset,begun),size,low_le) : {Nat.is_le(U32.to_nat(low_word),U32.to_nat(band_first)) == True{} : Bool}}
      +ending = Nat.div(Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n),Domain.height(geometry))
      +tail_stop = U32.to_nat(Kernel.copy_stop(geometry,task_tail_first(call,first,count),task_tail(call,first,count)))
      +high_slot = Nat.add(Nat.mul(ending,gradient),tail_stop)
      +high_le = high_case(Nat.is_lt(n,ending),call,envelope,first,count,nonempty,inside,rows_room,image,row,rows,contained,row_below,after,row_before,before,{==})
      +reached = reach_high(call,envelope,first,count,nonempty,inside,rows_room,tail_facts(call,first,count,nonempty,inside,rows_room))
      +end_word = Windows.span_end(Kernel.reach(Rows.task_of(geometry),first,count))
      +fits = {%Equal.sym(Nat,Nat.add(U32.to_nat(band_first),U32.to_nat(band_cells)),Nat.mul(slot_end,size),total) : {Nat.is_le(_,high) == True{} : Bool}
        Order.transitive(Nat.mul(slot_end,size),Nat.mul(high_slot,size),high,RowMajor.mul_le(slot_end,high_slot,size,high_le),
          Order.transitive(Nat.mul(high_slot,size),U32.to_nat(end_word),high,reached,reach_end)) :
        {Nat.is_le(Nat.add(U32.to_nat(band_first),U32.to_nat(band_cells)),high) == True{} : Bool}}
      BandSource.HeldBand{after_band,fits}

# The band of a call of an image inside the rows of its task lies in the
# reach of the task.
def call_placed(+call: Domain.Call,+envelope: Geometry.WordEnvelope,+first: U32,+count: U32,
  +nonempty: {Nat.is_lt(U32.to_nat(PieceCells.piece_start(call,first)),U32.to_nat(PieceCells.piece_end(call,first,count))) == True{} : Bool},
  +inside: {Nat.is_le(U32.to_nat(PieceCells.piece_end(call,first,count)),Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call)))) == True{} : Bool},
  +rows_room: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +high: Nat,+reach_end: {Nat.is_le(U32.to_nat(Windows.span_end(Kernel.reach(Rows.task_of(Domain.geometry(call)),first,count))),high) == True{} : Bool},
  +image: U32,+row: U32,+rows: U32,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(row),U32.to_nat(rows)),Domain.height(Domain.geometry(call))) == True{} : Bool},
  +row_below: {Nat.is_lt(U32.to_nat(row),Domain.height(Domain.geometry(call))) == True{} : Bool},
  +image_inside: {Nat.is_lt(U32.to_nat(image),Domain.batch(call)) == True{} : Bool},
  +after: {Nat.is_le(U32.to_nat(PieceCells.piece_start(call,first)),Nat.add(Nat.mul(U32.to_nat(image),Domain.height(Domain.geometry(call))),U32.to_nat(row))) == True{} : Bool},
  +row_before: {Nat.is_lt(Nat.add(Nat.mul(U32.to_nat(image),Domain.height(Domain.geometry(call))),U32.to_nat(row)),U32.to_nat(PieceCells.piece_end(call,first,count))) == True{} : Bool},
  +before: {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(image),Domain.height(Domain.geometry(call))),Nat.add(U32.to_nat(row),U32.to_nat(rows))),
    U32.to_nat(PieceCells.piece_end(call,first,count))) == True{} : Bool}) ->
  BandSource.Placed<U32.to_nat(Windows.span_start(Kernel.reach(Rows.task_of(Domain.geometry(call)),first,count))),high,
    Rows.band_first(Domain.geometry(call),image,row,rows),Rows.band_cells(Domain.geometry(call),row,rows)>:
  +child = Bands.caller_query(call,envelope,row,rows,contained)
  placed_case(U32.is_lt(CopyRows.begin(child),CopyRows.stop(child)),call,envelope,first,count,nonempty,inside,rows_room,high,reach_end,image,row,rows,
    contained,row_below,image_inside,after,row_before,before,{==})

# ---- The facts of the calls of a piece ----

# The rows a call takes: up to the end of its image or to the end of the
# rows, whichever comes first.
def taken_rows(+row: Nat,+end: Nat,+height: Nat,+positive: {Nat.is_lt(0n,height) == True{} : Bool},+before: {Nat.is_lt(row,end) == True{} : Bool}) ->
  {Nat.sub(Nat.min(end,Nat.mul(Nat.add(Nat.div(row,height),1n),height)),row) == Nat.min(Nat.sub(height,Nat.mod(row,height)),Nat.sub(end,row)) : Nat}:
  +gap = Nat.sub(height,Nat.mod(row,height))
  +left = Nat.sub(end,row)
  +limit = Nat.mul(Nat.add(Nat.div(row,height),1n),height)
  +end_split = Equal.sym(Nat,Nat.add(row,left),end,Order.restore_shift(row,end,Bounds.strict_le(row,end,before)))
  +limit_split = Equal.trans(Nat,limit,Nat.mul(1n+Nat.div(row,height),height),Nat.add(row,gap),
    Equal.cong(Nat,Nat,value => Nat.mul(value,height),Nat.add(Nat.div(row,height),1n),Nat.add(1n,Nat.div(row,height)),Order.commutative(Nat.div(row,height),1n)),
    Equal.sym(Nat,Nat.add(row,gap),Nat.mul(1n+Nat.div(row,height),height),Steps.next_image(height,row,positive)))
  +least = Equal.trans(Nat,Nat.min(end,limit),Nat.min(Nat.add(row,left),limit),Nat.add(row,Nat.min(left,gap)),
    Equal.cong(Nat,Nat,value => Nat.min(value,limit),end,Nat.add(row,left),end_split),
    Equal.trans(Nat,Nat.min(Nat.add(row,left),limit),Nat.min(Nat.add(row,left),Nat.add(row,gap)),Nat.add(row,Nat.min(left,gap)),
      Equal.cong(Nat,Nat,value => Nat.min(Nat.add(row,left),value),limit,Nat.add(row,gap),limit_split),
      NatExtrema.minimum_offset(row,left,gap)))
  Equal.trans(Nat,Nat.sub(Nat.min(end,limit),row),Nat.sub(Nat.add(row,Nat.min(left,gap)),row),Nat.min(gap,left),
    Equal.cong(Nat,Nat,value => Nat.sub(value,row),Nat.min(end,limit),Nat.add(row,Nat.min(left,gap)),least),
    Equal.trans(Nat,Nat.sub(Nat.add(row,Nat.min(left,gap)),row),Nat.min(left,gap),Nat.min(gap,left),Order.subtract_shift(row,Nat.min(left,gap)),
      NatExtrema.minimum_commutative(left,gap)))

# The calls of a piece from row on, remaining of them: each comes before
# the end of the piece and copies a band in the reach of its task.
def call_facts(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,remaining: Nat,+holder: Holder,+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool},
  +weights: Buffer.Model<Tile.Tile8>,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},+layout: Held.Spans,
  +task_first: U32,+task_count: U32,
  +task_nonempty: {Nat.is_lt(U32.to_nat(PieceCells.piece_start(call,task_first)),U32.to_nat(PieceCells.piece_end(call,task_first,task_count))) == True{} : Bool},
  +task_inside: {Nat.is_le(U32.to_nat(PieceCells.piece_end(call,task_first,task_count)),Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call)))) == True{} : Bool},
  +reach_end: {Nat.is_le(U32.to_nat(Windows.span_end(Kernel.reach(Rows.task_of(Domain.geometry(call)),task_first,task_count))),high) == True{} : Bool},
  +first: U32,+count: U32,+rows: PieceCells.PieceRows<call,first,count>,
  +start_after: {Nat.is_le(U32.to_nat(PieceCells.piece_start(call,task_first)),U32.to_nat(PieceCells.piece_start(call,first))) == True{} : Bool},
  +end_before: {Nat.is_le(U32.to_nat(PieceCells.piece_end(call,first,count)),U32.to_nat(PieceCells.piece_end(call,task_first,task_count))) == True{} : Bool},
  +row: U32,
  +after: {Nat.is_le(U32.to_nat(PieceCells.piece_start(call,first)),U32.to_nat(row)) == True{} : Bool},
  +ordered: {Nat.is_le(U32.to_nat(row),U32.to_nat(PieceCells.piece_end(call,first,count))) == True{} : Bool},
  +counted: {remaining == Steps.calls_count(Nat.is_lt(U32.to_nat(row),U32.to_nat(PieceCells.piece_end(call,first,count))),Domain.height(Domain.geometry(call)),
    U32.to_nat(row),U32.to_nat(PieceCells.piece_end(call,first,count))) : Nat}) ->
  BatchRows.CallFacts(~Holder,~value,remaining,
    PieceCells.piece_batch(~Holder,~value,holder,call,envelope,fits,weights,U32.to_nat(Windows.span_start(Kernel.reach(Rows.task_of(Domain.geometry(call)),task_first,task_count))),
      high,capacity,capacity_ok,bounded,layout,first,count,rows),row):
  match remaining:
    case 0n:
      +r = U32.to_nat(row)
      +e = U32.to_nat(PieceCells.piece_end(call,first,count))
      +height = Domain.height(Domain.geometry(call))
      Steps.none_with(Nat.is_lt(r,e),height,r,e,{==},ordered,Equal.sym(Nat,0n,Steps.calls_count(Nat.is_lt(r,e),height,r,e),counted))
    case 1n+ +rest:
      +geometry = Domain.geometry(call)
      +end = PieceCells.piece_end(call,first,count)
      +r = U32.to_nat(row)
      +e = U32.to_nat(end)
      +task_start = U32.to_nat(PieceCells.piece_start(call,task_first))
      +task_end = U32.to_nat(PieceCells.piece_end(call,task_first,task_count))
      +height = Domain.height(geometry)
      +positive = Domain.height_positive(call)
      +before = Steps.some_with(Nat.is_lt(r,e),height,r,e,rest,{==},Equal.sym(Nat,1n+rest,Steps.calls_count(Nat.is_lt(r,e),height,r,e),counted))
      +rows_inside = PieceCells.rows_inside_of(call,first,count,rows)
      +rows_room = PieceCells.rows_room_of(call,first,count,rows)
      +step = BatchRows.row_step(call,row,end,before,rows_inside,rows_room)
      +image = BatchRows.image_word(geometry,row)
      +local = BatchRows.first_word(geometry,row)
      +taken = BatchRows.count_word(geometry,row,end)
      +n = Nat.div(r,height)
      +f = Nat.mod(r,height)
      +c = U32.to_nat(taken)
      +image_exact = BatchRows.row_image(call,row,end,step)
      +local_exact = BatchRows.row_first(call,row,end,step)
      +slot = Nat.add(Nat.mul(U32.to_nat(image),height),U32.to_nat(local))
      +slot_exact = {%Equal.sym(Nat,U32.to_nat(image),n,image_exact) : {Nat.add(Nat.mul(_,height),U32.to_nat(local)) == r : Nat}
        %Equal.sym(Nat,U32.to_nat(local),f,local_exact) : {Nat.add(Nat.mul(n,height),_) == r : Nat}
        Equal.sym(Nat,r,Nat.add(Nat.mul(n,height),f),DivisionOrder.quotient_exact(r,height,positive)) : {slot == r : Nat}}
      +contained = {%Equal.sym(Nat,U32.to_nat(local),f,local_exact) : {Nat.is_le(Nat.add(_,c),height) == True{} : Bool}
        BatchRows.row_contained(call,row,end,step) : {Nat.is_le(Nat.add(U32.to_nat(local),c),height) == True{} : Bool}}
      +row_below = {%Equal.sym(Nat,U32.to_nat(local),f,local_exact) : {Nat.is_lt(_,height) == True{} : Bool}
        DivisionOrder.remainder_small(r,height,positive) : {Nat.is_lt(U32.to_nat(local),height) == True{} : Bool}}
      +image_inside = {%Equal.sym(Nat,U32.to_nat(image),n,image_exact) : {Nat.is_lt(_,Domain.batch(call)) == True{} : Bool}
        BatchRows.row_image_inside(call,row,end,step) : {Nat.is_lt(U32.to_nat(image),Domain.batch(call)) == True{} : Bool}}
      +task_after = {%Equal.sym(Nat,slot,r,slot_exact) : {Nat.is_le(task_start,_) == True{} : Bool}
        Order.transitive(task_start,U32.to_nat(PieceCells.piece_start(call,first)),r,start_after,after) : {Nat.is_le(task_start,slot) == True{} : Bool}}
      +row_before = {%Equal.sym(Nat,slot,r,slot_exact) : {Nat.is_lt(_,task_end) == True{} : Bool}
        Bounds.nat_trans(r,e,task_end,before,end_before) : {Nat.is_lt(slot,task_end) == True{} : Bool}}
      +next_le = BatchRows.row_next_le(call,row,end,step)
      +call_end = Equal.trans(Nat,Nat.add(r,c),Nat.add(slot,c),Nat.add(Nat.mul(U32.to_nat(image),height),Nat.add(U32.to_nat(local),c)),
        Equal.cong(Nat,Nat,value => Nat.add(value,c),r,slot,Equal.sym(Nat,slot,r,slot_exact)),
        Addition.associative(Nat.mul(U32.to_nat(image),height),U32.to_nat(local),c))
      +task_before = {%call_end : {Nat.is_le(_,task_end) == True{} : Bool}
        Order.transitive(Nat.add(r,c),e,task_end,next_le,end_before) :
        {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(image),height),Nat.add(U32.to_nat(local),c)),task_end) == True{} : Bool}}
      +placed = call_placed(call,envelope,task_first,task_count,task_nonempty,task_inside,rows_room,high,reach_end,image,local,taken,
        contained,row_below,image_inside,task_after,row_before,task_before)
      +next = BatchRows.next_word(geometry,row,end)
      +next_exact = BatchRows.row_next(call,row,end,step)
      +next_after = {%Equal.sym(Nat,U32.to_nat(next),Nat.add(r,c),next_exact) : {Nat.is_le(U32.to_nat(PieceCells.piece_start(call,first)),_) == True{} : Bool}
        Order.transitive(U32.to_nat(PieceCells.piece_start(call,first)),r,Nat.add(r,c),after,Order.le_add(r,c)) :
        {Nat.is_le(U32.to_nat(PieceCells.piece_start(call,first)),U32.to_nat(next)) == True{} : Bool}}
      +next_ordered = {%Equal.sym(Nat,U32.to_nat(next),Nat.add(r,c),next_exact) : {Nat.is_le(_,e) == True{} : Bool} next_le :
        {Nat.is_le(U32.to_nat(next),e) == True{} : Bool}}
      +taken_exact = Equal.trans(Nat,c,Nat.sub(Nat.min(e,Nat.mul(Nat.add(n,1n),height)),r),Nat.min(Nat.sub(height,f),Nat.sub(e,r)),
        BatchRows.row_count(call,row,end,step),taken_rows(r,e,height,positive,before))
      +stepped = Steps.calls_step(Nat.is_lt(Nat.add(r,c),e),height,r,e,c,positive,before,taken_exact,next_le,{==})
      +more = {%before : {1n+rest == Steps.calls_count(_,height,r,e) : Nat} counted : {1n+rest == Steps.calls_count(True{},height,r,e) : Nat}}
      +later = Order.cancel_sum(1n,rest,Steps.calls_count(Nat.is_lt(Nat.add(r,c),e),height,Nat.add(r,c),e),
        Equal.trans(Nat,1n+rest,Steps.calls_count(True{},height,r,e),1n+Steps.calls_count(Nat.is_lt(Nat.add(r,c),e),height,Nat.add(r,c),e),more,stepped))
      +next_counted = {%Equal.sym(Nat,U32.to_nat(next),Nat.add(r,c),next_exact) : {rest == Steps.calls_count(Nat.is_lt(_,e),height,_,e) : Nat}
        later : {rest == Steps.calls_count(Nat.is_lt(U32.to_nat(next),e),height,U32.to_nat(next),e) : Nat}}
      KernelGemm.Both{BatchRows.CallFact{before,placed},
        call_facts(~Holder,~value,rest,holder,call,envelope,fits,weights,high,capacity,capacity_ok,bounded,layout,task_first,task_count,task_nonempty,task_inside,reach_end,
          first,count,rows,start_after,end_before,next,next_after,next_ordered,next_counted)}

# ---- The calls of a piece of a task ----

# The images a loop from row to end visits are its calls.
def images_count(+call: Domain.Call,+row: U32,+end: U32,
  +end_room: {Nat.is_le(U32.to_nat(end),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool}) ->
  {Kernel.images_between(Domain.geometry(call),row,end) ==
    Steps.calls_count(Nat.is_lt(U32.to_nat(row),U32.to_nat(end)),Domain.height(Domain.geometry(call)),U32.to_nat(row),U32.to_nat(end)) : Nat}:
  Domain.Call{+geometry,+images,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +whole = {Domain.Call{shape,images,limits,facts} : Domain.Call}
  +room = U32.to_nat(Machine.capacity(limits))
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
  +less = Nat.is_lt(r,e)
  %Equal.sym(Bool,U32.is_lt(row,end),less,Comparison.lt(row,end)) :
    {U32.to_nat(Bool.pick(U32,_,((end - 1) / word - row / word + 1 : U32),0)) == Steps.calls_count(less,height,r,e) : Nat}
  %word_exact : {U32.to_nat(Bool.pick(U32,less,((end - 1) / word - row / word + 1 : U32),0)) == Steps.calls_count(less,_,r,e) : Nat}
  Steps.calls_with(less,limits,word,row,end,word_positive,word_room,end_room,{==})

# The calls of a piece of the cells of a task, each before the end of the
# piece and copying a band in the reach of the task.
def piece_calls(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool},
  +weights: Buffer.Model<Tile.Tile8>,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},+layout: Held.Spans,
  +task_first: U32,+task_count: U32,+task_rows: PieceCells.PieceRows<call,task_first,task_count>,
  +task_nonempty: {Nat.is_lt(U32.to_nat(PieceCells.piece_start(call,task_first)),U32.to_nat(PieceCells.piece_end(call,task_first,task_count))) == True{} : Bool},
  +reach_end: {Nat.is_le(U32.to_nat(Windows.span_end(Kernel.reach(Rows.task_of(Domain.geometry(call)),task_first,task_count))),high) == True{} : Bool},
  +first: U32,+count: U32,+rows: PieceCells.PieceRows<call,first,count>,
  +first_after: {Nat.is_le(U32.to_nat(task_first),U32.to_nat(first)) == True{} : Bool},
  +cells_before: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(task_first),U32.to_nat(task_count))) == True{} : Bool}) ->
  BatchRows.CallFacts(~Holder,~value,Kernel.images_between(Domain.geometry(call),PieceCells.piece_start(call,first),PieceCells.piece_end(call,first,count)),
    PieceCells.piece_batch(~Holder,~value,holder,call,envelope,fits,weights,U32.to_nat(Windows.span_start(Kernel.reach(Rows.task_of(Domain.geometry(call)),task_first,task_count))),
      high,capacity,capacity_ok,bounded,layout,first,count,rows),PieceCells.piece_start(call,first)):
  +size = BatchRows.row_size(call)
  +positive = PieceCells.rows_size_positive(call,first,count,rows)
  +start = PieceCells.piece_start(call,first)
  +end = PieceCells.piece_end(call,first,count)
  +task_start = PieceCells.piece_start(call,task_first)
  +task_end = PieceCells.piece_end(call,task_first,task_count)
  +f = U32.to_nat(first)
  +tf = U32.to_nat(task_first)
  +stop = Nat.add(f,U32.to_nat(count))
  +task_stop = Nat.add(tf,U32.to_nat(task_count))
  +start_after = {%Equal.sym(Nat,U32.to_nat(task_start),Nat.div(tf,size),PieceCells.rows_start_exact(call,task_first,task_count,task_rows)) :
      {Nat.is_le(_,U32.to_nat(start)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(start),Nat.div(f,size),PieceCells.rows_start_exact(call,first,count,rows)) : {Nat.is_le(Nat.div(tf,size),_) == True{} : Bool}
    DivisionOrder.quotient_le(tf,f,size,positive,first_after) : {Nat.is_le(U32.to_nat(task_start),U32.to_nat(start)) == True{} : Bool}}
  +end_before = {%Equal.sym(Nat,U32.to_nat(end),Ceiling.ceiling(stop,size),PieceCells.rows_end_exact(call,first,count,rows)) :
      {Nat.is_le(_,U32.to_nat(task_end)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(task_end),Ceiling.ceiling(task_stop,size),PieceCells.rows_end_exact(call,task_first,task_count,task_rows)) :
      {Nat.is_le(Ceiling.ceiling(stop,size),_) == True{} : Bool}
    Ceiling.monotone(stop,task_stop,size,positive,cells_before) : {Nat.is_le(U32.to_nat(end),U32.to_nat(task_end)) == True{} : Bool}}
  +rows_inside = PieceCells.rows_inside_of(call,first,count,rows)
  +rows_room = PieceCells.rows_room_of(call,first,count,rows)
  +end_room = Order.transitive(U32.to_nat(end),Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call))),
    rows_inside,rows_room)
  call_facts(~Holder,~value,Kernel.images_between(Domain.geometry(call),start,end),holder,call,envelope,fits,weights,high,capacity,capacity_ok,bounded,layout,
    task_first,task_count,task_nonempty,PieceCells.rows_inside_of(call,task_first,task_count,task_rows),reach_end,first,count,rows,start_after,end_before,
    start,Order.reflexive(U32.to_nat(start)),PieceCells.rows_ordered(call,first,count,rows),images_count(call,start,end,end_room))

# ---- The reach of a task lies in the gradient cells ----

# The gradient cells a task reads lie in the gradient cells of the batch.
def reach_inside(+call: Domain.Call,+envelope: Geometry.WordEnvelope,+first: U32,+count: U32,
  +nonempty: {Nat.is_lt(U32.to_nat(PieceCells.piece_start(call,first)),U32.to_nat(PieceCells.piece_end(call,first,count))) == True{} : Bool},
  +inside: {Nat.is_le(U32.to_nat(PieceCells.piece_end(call,first,count)),Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call)))) == True{} : Bool},
  +rows_room: {Nat.is_le(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool}) ->
  Windows.Reaches(~Kernel.Task,~Kernel.reach_of,Rows.task_of(Domain.geometry(call)),Pieces.Range{first,count},Rows.cells_count(call)):
  +geometry = Domain.geometry(call)
  +total = Rows.cells_count(call)
  +rows = Rows.rows_count(call)
  +height = Domain.height(geometry)
  +start = U32.to_nat(PieceCells.piece_start(call,first))
  +tails = tail_facts(call,first,count,nonempty,inside,rows_room)
  +low_word = Windows.span_start(Kernel.reach(Rows.task_of(geometry),first,count))
  +copied = U32.to_nat(Kernel.copy_start(geometry,task_low_first(call,first),task_head(call,first,count)))
  +low_slot = Nat.mul(Nat.add(Nat.mul(Nat.div(start,height),rows),Nat.min(copied,rows)),row_cells(call))
  +low_facts = reach_low(call,first,count,nonempty,inside,rows_room)
  +image_inside = DivisionOrder.quotient_below(start,Domain.batch(call),height,Domain.height_positive(call),
    Bounds.nat_trans(start,U32.to_nat(PieceCells.piece_end(call,first,count)),Nat.mul(Domain.batch(call),height),nonempty,inside))
  +low_in = {%Equal.sym(Nat,U32.to_nat(low_word),low_slot,
        Shape.first({U32.to_nat(low_word) == low_slot : Nat},{Nat.is_le(U32.to_nat(low_word),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},low_facts)) :
      {Nat.is_le(_,total) == True{} : Bool}
    slot_inside(call,Nat.div(start,height),Nat.min(copied,rows),image_inside,NatExtrema.minimum_right(copied,rows)) :
    {Nat.is_le(U32.to_nat(low_word),total) == True{} : Bool}}
  +high_word = ((BatchRows.image_word(geometry,task_last(call,first,count)) * Kernel.gradient_height(geometry) +
    Kernel.copy_stop(geometry,task_tail_first(call,first,count),task_tail(call,first,count))) * Kernel.gradient_row_cells(geometry) : U32)
  +high_slot = Nat.mul(Nat.add(Nat.mul(Nat.div(Nat.sub(U32.to_nat(PieceCells.piece_end(call,first,count)),1n),height),rows),
    U32.to_nat(Kernel.copy_stop(geometry,task_tail_first(call,first,count),task_tail(call,first,count)))),row_cells(call))
  +bound = high_bound(call,envelope,first,count,nonempty,inside,rows_room,tails)
  +high_in = {%Equal.sym(Nat,U32.to_nat(high_word),high_slot,
        Shape.first({U32.to_nat(high_word) == high_slot : Nat},{Nat.is_le(high_slot,total) == True{} : Bool},bound)) :
      {Nat.is_le(_,total) == True{} : Bool}
    Shape.second({U32.to_nat(high_word) == high_slot : Nat},{Nat.is_le(high_slot,total) == True{} : Bool},bound) :
    {Nat.is_le(U32.to_nat(high_word),total) == True{} : Bool}}
  +most_word = U32.max(high_word,low_word)
  +most_exact = Extrema.maximum(high_word,low_word)
  +ordered = {%Equal.sym(Nat,U32.to_nat(most_word),Nat.max(U32.to_nat(high_word),U32.to_nat(low_word)),most_exact) :
      {Nat.is_le(U32.to_nat(low_word),_) == True{} : Bool}
    NatExtrema.maximum_right(U32.to_nat(high_word),U32.to_nat(low_word)) : {Nat.is_le(U32.to_nat(low_word),U32.to_nat(most_word)) == True{} : Bool}}
  +most_in = {%Equal.sym(Nat,U32.to_nat(most_word),Nat.max(U32.to_nat(high_word),U32.to_nat(low_word)),most_exact) : {Nat.is_le(_,total) == True{} : Bool}
    NatExtrema.maximum_upper(U32.to_nat(high_word),U32.to_nat(low_word),total,high_in,low_in) : {Nat.is_le(U32.to_nat(most_word),total) == True{} : Bool}}
  Windows.span_fits(Domain.limits(call),low_word,most_word,total,ordered,most_in,Rows.cells_room(call))

# A run of cells inside the input gradient reads inside the gradient.
def range_reaches(+call: Domain.Call,+envelope: Geometry.WordEnvelope,+unit: Nat,+bound: Nat,
  +bound_cells: {Nat.is_le(bound,Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call))) == True{} : Bool},
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call)),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  +first: U32,+count: U32,+fits: Runs.Fits<unit,U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),bound>) ->
  Windows.Reaches(~Kernel.Task,~Kernel.reach_of,Rows.task_of(Domain.geometry(call)),Pieces.Range{first,count},Rows.cells_count(call)):
  +f = U32.to_nat(first)
  +stop = Nat.add(f,U32.to_nat(count))
  +inside = Order.transitive(stop,bound,Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call)),
    Runs.fits_inside(unit,f,stop,bound,fits),bound_cells)
  +rows = PieceCells.piece_rows(call,first,count,inside,total_room)
  reach_inside(call,envelope,first,count,PieceCells.rows_nonempty(call,first,count,rows,Runs.fits_less(unit,f,stop,bound,fits)),
    PieceCells.rows_inside_of(call,first,count,rows),PieceCells.rows_room_of(call,first,count,rows))

# Tasks that are runs of cells inside the input gradient read inside the
# gradient.
def tasks_reach(+call: Domain.Call,+envelope: Geometry.WordEnvelope,+unit: Nat,+bound: Nat,
  +bound_cells: {Nat.is_le(bound,Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call))) == True{} : Bool},
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(Domain.batch(call),Domain.height(Domain.geometry(call))),BatchRows.row_size(call)),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool},
  tasks: Pieces.Ranges,+run: Runs.Run(tasks,unit,bound)) ->
  Windows.Reaches(~Kernel.Task,~Kernel.reach_of,Rows.task_of(Domain.geometry(call)),tasks,Rows.cells_count(call)):
  match tasks:
    case Pieces.Range{+first,+count}: range_reaches(call,envelope,unit,bound,bound_cells,total_room,first,count,run)
    case Pieces.Split{+left,+right}:
      Held.Both{+left_run,+rest} = run
      Held.Both{+right_run,+glue} = rest
      Held.Both{tasks_reach(call,envelope,unit,bound,bound_cells,total_room,left,left_run),tasks_reach(call,envelope,unit,bound,bound_cells,total_room,right,right_run)}
