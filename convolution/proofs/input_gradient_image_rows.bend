# One call of an input-gradient task on rows of an image: the band of
# gradient rows it reads, lent from the holding of the gradient pieces, the
# leaf it runs on the band, and what it gives back.
import Base
import ../../lib/kernel_input_gradient.bend as Kernel
import ../../lib/kernel_tile.bend as Tile
import ../../lib/tensor_move.bend as Move
import ../../lib/storage_buffer.bend as Storage
import ../../lib/proofs/input_gradient_phase_domain.bend as Domain
import ../../lib/proofs/input_gradient_tiled_chunk_geometry.bend as GeometryChunks
import ../../lib/proofs/input_gradient_tiled_leaf_execution.bend as Execution
import ../../lib/proofs/input_gradient_tiled_leaf_writes.bend as LeafWrites
import ../../lib/proofs/input_gradient_tiled_phase_index.bend as Index
import ../../lib/proofs/input_gradient_tiled_packing_bounds.bend as PackingBounds
import ../../lib/proofs/input_gradient_tiled_job_geometry.bend as Jobs
import ../../lib/proofs/input_gradient_tiled_weight_reads.bend as Weights
import ../../lib/proofs/input_gradient_tiled_pack_values.bend as Packed
import ../../lib/proofs/input_gradient_tiled_weights.bend as WeightPacking
import ../../lib/proofs/tensor_move_readback.bend as Held
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/machine_limits.bend as Machine
import ../../lib/tensor_grid.bend as Grid
import ../../lib/tensor_tiled.bend as Tiled
import ../../lib/kernel_gemm.bend as KernelGemm
import ../../lib/proofs/input_gradient_phase_packing.bend as Packing
import ../../lib/proofs/input_gradient_tiled_geometry.bend as Geometry
import ../../lib/proofs/input_gradient_tiled_chunk_reads.bend as Chunks
import ../../lib/proofs/input_gradient_tiled_producer.bend as Producer
import ../../lib/proofs/input_gradient_tiled_output_layout.bend as Output
import ../../lib/proofs/input_gradient_tiled_coverage.bend as Coverage
import ../../lib/proofs/input_gradient_tiled_fill.bend as Fill
import ../../lib/proofs/input_gradient_tiled_band_extents.bend as Bands
import ../../lib/proofs/input_gradient_tiled_copy_geometry.bend as Copy
import ../../lib/proofs/input_gradient_tiled_band_order.bend as BandOrder
import ../../lib/proofs/input_gradient_tiled_source_rows.bend as SourceRows
import ../../lib/proofs/input_gradient_tiled_band_source.bend as BandSource
import ../../lib/proofs/input_gradient_tiled_leaf_length.bend as LeafLength
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/write_program.bend as Writes
import ../../lib/proofs/write_program_every.bend as Every
import ../../lib/proofs/write_program_readback.bend as Readback
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ../../lib/proofs/tensor_tiled_values.bend as TiledValues
import ../../lib/proofs/tensor_allocation_depth.bend as Depth
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/nat_ceiling.bend as Ceiling
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/u32_full_range.bend as FullRange
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/word_multiplication.bend as Multiplication
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ./input_gradient_tiled_written.bend as Phases
import ./input_gradient_tiled_leaf_written.bend as LeafWritten
import ./input_gradient_tiled_leaf.bend as Leaf
import ./input_gradient_cell_sources.bend as Sources
import ../input_gradient_specification.bend as Specification

# ---- The packed weights ----

# The packed weights of a call: the kernel packs them once, before any
# image, and a leaf reads the tree they hold.
def packing(+call: Domain.Call,+weights: Buffer.Model<Tile.Tile8>) -> Kernel.Packed:
  Packed.native_complete(PackingBounds.Query{call},weights)
def native_tree(state: Kernel.Packed) -> Tree.Tree<Tile.Tile8>:
  Kernel.Packed{source,weights,weight_base} = state
  Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,weights))
def represent_tree(+state: WeightPacking.Packed) -> {native_tree(WeightPacking.represent(state)) == Packed.weight_tree(state) : Tree.Tree<Tile.Tile8>}:
  WeightPacking.Packed{source,+weights,weight_base} = state
  Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights))),
    Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,weights)),Buffer.tree(~Tile.Tile8,weights),
    Equal.cong(Buffer.Content<Tile.Tile8>,Tree.Tree<Tile.Tile8>,Buffer.content_tree(~Tile.Tile8),Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),
      Buffer.content(~Tile.Tile8,weights),Buffer.packed_contents(~Tile.Tile8,weights)),
    Buffer.content_tree_of(~Tile.Tile8,weights))

# The tree of the packed weights of a call is the tree of the full packing
# state of the first phase of any grid of the call.
def packed_tree(+call: Domain.Call,+weights: Buffer.Model<Tile.Tile8>,+grid: GeometryChunks.Grid,
  +same_call: {GeometryChunks.grid_call(grid) == call : Domain.Call}) ->
  {native_tree(packing(call,weights)) == Packed.weight_tree(Weights.full_state(Index.zero(grid),weights)) : Tree.Tree<Tile.Tile8>}:
  GeometryChunks.Grid{+grid_call,+e,+count,+range,+rp,+rr} = grid
  +whole = {GeometryChunks.Grid{grid_call,e,count,range,rp,rr} : GeometryChunks.Grid}
  +phase = Index.zero(whole)
  %same_call : {native_tree(Packed.native_complete(PackingBounds.Query{_},weights)) == Packed.weight_tree(Weights.full_state(phase,weights)) : Tree.Tree<Tile.Tile8>}
  +state = Weights.full_state(phase,weights)
  +matches = {Packed.native_complete_matches(PackingBounds.Query{grid_call},weights,Index.ordinal(phase),Nat.add(Nat.mul(0n,U32.to_nat(Kernel.phase_inner(Jobs.context(phase)))),0n)) :
    {Packed.native_complete(PackingBounds.Query{grid_call},weights) == WeightPacking.represent(state) : Kernel.Packed}}
  Equal.trans(Tree.Tree<Tile.Tile8>,native_tree(Packed.native_complete(PackingBounds.Query{grid_call},weights)),native_tree(WeightPacking.represent(state)),Packed.weight_tree(state),
    Equal.cong(Kernel.Packed,Tree.Tree<Tile.Tile8>,native_tree,Packed.native_complete(PackingBounds.Query{grid_call},weights),WeightPacking.represent(state),matches),
    represent_tree(state))

# ---- The band and the leaf of a call ----

# The cells of the gradient rows that output rows [first, first + count) of
# an image read, and the band of them lent from a holding.
def band_first(+geometry: Kernel.Geometry,+image: U32,+first: U32,+count: U32) -> U32:
  ((image * Kernel.gradient_height(geometry) + Kernel.copy_start(geometry,first,count)) * Kernel.gradient_row_cells(geometry) : U32)
def band_cells(+geometry: Kernel.Geometry,+first: U32,+count: U32) -> U32:
  (Kernel.copy_count(geometry,first,count) * Kernel.gradient_row_cells(geometry) : U32)
def call_band(holding: Held.Holding<Tile.Tile8>,spare: Held.SpareModel<Tile.Tile8>,+geometry: Kernel.Geometry,+image: U32,+first: U32,+count: U32) -> Move.Band<Tile.Tile8>:
  Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,holding),band_first(geometry,image,first,count),band_cells(geometry,first,count),Tile.filled(0.0),
    Held.pack_spare(~Tile.Tile8,spare))

# The leaf of a call: its context, the rows it holds as a model, and the part
# it gives back.
def task_of(+geometry: Kernel.Geometry) -> Kernel.Task:
  Kernel.Task{geometry,Kernel.tiling(geometry)}
def leaf_context(+geometry: Kernel.Geometry,+first: U32,+count: U32,+target: U32) -> Kernel.LeafContext:
  Kernel.LeafContext{task_of(geometry),first,count,target}
def call_held(+holding: Held.Holding<Tile.Tile8>,+spare: Held.SpareModel<Tile.Tile8>,+geometry: Kernel.Geometry,+image: U32,+first: U32,+count: U32) -> Execution.Held:
  Execution.Held{Buffer.remodel(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count))),
    Held.band_offset(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count)),Kernel.copy_start(geometry,first,count),Kernel.copy_count(geometry,first,count)}
def call_part(+holding: Held.Holding<Tile.Tile8>,+spare: Held.SpareModel<Tile.Tile8>,+geometry: Kernel.Geometry,+image: U32,+first: U32,+count: U32,+target: U32,
  +tiles: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) -> Execution.Part:
  Execution.leaf_band(leaf_context(geometry,first,count,target),call_held(holding,spare,geometry,image,first,count),tiles,output)

def banded_eta(+geometry: Kernel.Geometry,+first: U32,+count: U32,+target: U32,banded: Move.Band<Tile.Tile8>,weights: Storage.Buffer<Tile.Tile8>,
  output: Storage.Buffer<Tile.Tile8>) ->
  {Kernel.image_banded(task_of(geometry),first,count,target,banded,weights,output) ==
    Kernel.image_returned(Held.band_loan(~Tile.Tile8,banded),Kernel.leaf_band(leaf_context(geometry,first,count,target),
      Kernel.Source{Held.band_buffer(~Tile.Tile8,banded),Held.band_offset(~Tile.Tile8,banded),Kernel.copy_start(geometry,first,count),Kernel.copy_count(geometry,first,count)},
      weights,output)) : Kernel.Rows}:
  Move.Band{lent,input,base} = banded
  {==}

def rows_split(returned: Move.Holding<Tile.Tile8> & Move.Spare<Tile.Tile8>,weights: Storage.Buffer<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>) ->
  {Kernel.rows_back(returned,weights,output) ==
    Kernel.Rows{Pair.fst(Move.Holding<Tile.Tile8>,Move.Spare<Tile.Tile8>,returned),Pair.snd(Move.Holding<Tile.Tile8>,Move.Spare<Tile.Tile8>,returned),weights,output} : Kernel.Rows}:
  (gradient,spare) = returned
  {==}

def part_eta(lent: Move.Lent<Tile.Tile8>,+part: Execution.Part) ->
  {Kernel.image_returned(lent,Execution.represent_part(part)) ==
    Kernel.Rows{Move.returned_holding(~Tile.Tile8,lent,Buffer.pack(~Tile.Tile8,LeafWrites.part_source(part))),Held.spare_back(~Tile.Tile8,lent,Buffer.pack(~Tile.Tile8,LeafWrites.part_source(part))),
      Buffer.pack(~Tile.Tile8,LeafWrites.part_weights(part)),Buffer.pack(~Tile.Tile8,Execution.part_output(part))} : Kernel.Rows}:
  Execution.Part{+gradient,+weights,+output} = part
  rows_split(Move.returned(~Tile.Tile8,lent,Buffer.pack(~Tile.Tile8,gradient)),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,output))

# A call lends the band, runs the leaf on it and returns it: the rows it
# leaves are the models the leaf gives back.
def call_rows(+holding: Held.Holding<Tile.Tile8>,+spare: Held.SpareModel<Tile.Tile8>,+geometry: Kernel.Geometry,+image: U32,+first: U32,+count: U32,+target: U32,
  +tiles: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) ->
  {Kernel.image_rows(task_of(geometry),image,first,count,target,Kernel.Rows{Held.pack(~Tile.Tile8,holding),Held.pack_spare(~Tile.Tile8,spare),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)}) ==
    Kernel.Rows{Move.returned_holding(~Tile.Tile8,Held.band_loan(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count)),
        Buffer.pack(~Tile.Tile8,LeafWrites.part_source(call_part(holding,spare,geometry,image,first,count,target,tiles,output)))),
      Held.spare_back(~Tile.Tile8,Held.band_loan(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count)),
        Buffer.pack(~Tile.Tile8,LeafWrites.part_source(call_part(holding,spare,geometry,image,first,count,target,tiles,output)))),
      Buffer.pack(~Tile.Tile8,LeafWrites.part_weights(call_part(holding,spare,geometry,image,first,count,target,tiles,output))),
      Buffer.pack(~Tile.Tile8,Execution.part_output(call_part(holding,spare,geometry,image,first,count,target,tiles,output)))} : Kernel.Rows}:
  +context = leaf_context(geometry,first,count,target)
  +held = call_held(holding,spare,geometry,image,first,count)
  +part = call_part(holding,spare,geometry,image,first,count,target,tiles,output)
  +start = Kernel.copy_start(geometry,first,count)
  +rows = Kernel.copy_count(geometry,first,count)
  Equal.trans(Kernel.Rows,
    Kernel.image_rows(task_of(geometry),image,first,count,target,Kernel.Rows{Held.pack(~Tile.Tile8,holding),Held.pack_spare(~Tile.Tile8,spare),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)}),
    Kernel.image_returned(Held.band_loan(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count)),Execution.represent_part(part)),
    Kernel.Rows{Move.returned_holding(~Tile.Tile8,Held.band_loan(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count)),Buffer.pack(~Tile.Tile8,LeafWrites.part_source(part))),
      Held.spare_back(~Tile.Tile8,Held.band_loan(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count)),Buffer.pack(~Tile.Tile8,LeafWrites.part_source(part))),
      Buffer.pack(~Tile.Tile8,LeafWrites.part_weights(part)),Buffer.pack(~Tile.Tile8,Execution.part_output(part))},
    Equal.trans(Kernel.Rows,
      Kernel.image_rows(task_of(geometry),image,first,count,target,Kernel.Rows{Held.pack(~Tile.Tile8,holding),Held.pack_spare(~Tile.Tile8,spare),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)}),
      Kernel.image_returned(Held.band_loan(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count)),Kernel.leaf_band(context,
        Kernel.Source{Held.band_buffer(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count)),Held.band_offset(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count)),start,rows},
        Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output))),
      Kernel.image_returned(Held.band_loan(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count)),Execution.represent_part(part)),
      banded_eta(geometry,first,count,target,call_band(holding,spare,geometry,image,first,count),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)),
      Equal.cong(Kernel.Part,Kernel.Rows,result => Kernel.image_returned(Held.band_loan(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count)),result),
        Kernel.leaf_band(context,
          Kernel.Source{Held.band_buffer(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count)),Held.band_offset(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count)),start,rows},
          Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)),
        Execution.represent_part(part),
        Equal.trans(Kernel.Part,
          Kernel.leaf_band(context,
            Kernel.Source{Held.band_buffer(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count)),Held.band_offset(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count)),start,rows},
            Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)),
          Kernel.leaf_band(context,Execution.represent_held(held),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)),
          Execution.represent_part(part),
          Equal.cong(Storage.Buffer<Tile.Tile8>,Kernel.Part,
            data => Kernel.leaf_band(context,Kernel.Source{data,Held.band_offset(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count)),start,rows},
              Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)),
            Held.band_buffer(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count)),
            Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count)))),
            Equal.sym(Storage.Buffer<Tile.Tile8>,Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count)))),
              Held.band_buffer(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count)),
              Buffer.remodeled(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count))))),
          Execution.leaf_matches(context,held,tiles,output)))),
    part_eta(Held.band_loan(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count)),part))

# The buffer a call returns holds the band it was lent.
def call_same(+holding: Held.Holding<Tile.Tile8>,+spare: Held.SpareModel<Tile.Tile8>,+geometry: Kernel.Geometry,+image: U32,+first: U32,+count: U32,+target: U32,
  +tiles: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) ->
  {Buffer.content(~Tile.Tile8,LeafWrites.part_source(call_part(holding,spare,geometry,image,first,count,target,tiles,output))) ==
    Buffer.contents(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count))) : Buffer.Content<Tile.Tile8>}:
  +held = call_held(holding,spare,geometry,image,first,count)
  +part = call_part(holding,spare,geometry,image,first,count,target,tiles,output)
  Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,LeafWrites.part_source(part)),Buffer.content(~Tile.Tile8,Execution.held_data(held)),
    Buffer.contents(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count))),
    LeafWrites.leaf_content(leaf_context(geometry,first,count,target),held,tiles,output),
    Buffer.remodel_content(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count))))

# The band goes back into the holding with the cells it was lent with.
def call_covered(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+holding: Held.Holding<Tile.Tile8>,+spare: Held.SpareModel<Tile.Tile8>,
  +geometry: Kernel.Geometry,+image: U32,+first: U32,+count: U32,+target: U32,+tiles: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>,
  +low: Nat,+high: Nat,+tiled: Held.Tiled(~Tile.Tile8,holding,low,high),+values: Held.HoldingValues(~Tile.Tile8,~Holder,~value,holder,holding),
  +layout: Held.Spans,+ordered: {Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding)) == layout : Held.Spans}) ->
  Held.CoveredFact(~Tile.Tile8,~Holder,~value,holder,
    Move.returned_holding(~Tile.Tile8,Held.band_loan(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count)),
      Buffer.pack(~Tile.Tile8,LeafWrites.part_source(call_part(holding,spare,geometry,image,first,count,target,tiles,output)))),low,high,layout):
  Held.band_returned(~Tile.Tile8,~Holder,~value,holder,holding,band_first(geometry,image,first,count),band_cells(geometry,first,count),Tile.filled(0.0),spare,
    low,high,tiled,values,layout,ordered,LeafWrites.part_source(call_part(holding,spare,geometry,image,first,count,target,tiles,output)),
    call_same(holding,spare,geometry,image,first,count,target,tiles,output))

# The spare a call leaves is the model's, with room for what it held.
def call_spare(+holding: Held.Holding<Tile.Tile8>,+spare: Held.SpareModel<Tile.Tile8>,+geometry: Kernel.Geometry,+image: U32,+first: U32,+count: U32,+target: U32,
  +tiles: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) ->
  {Held.spare_back(~Tile.Tile8,Held.band_loan(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count)),
      Buffer.pack(~Tile.Tile8,LeafWrites.part_source(call_part(holding,spare,geometry,image,first,count,target,tiles,output)))) ==
    Held.pack_spare(~Tile.Tile8,Held.spare_after(~Tile.Tile8,holding,band_first(geometry,image,first,count),band_cells(geometry,first,count),spare,
      LeafWrites.part_source(call_part(holding,spare,geometry,image,first,count,target,tiles,output)))) : Move.Spare<Tile.Tile8>}:
  Held.band_spare(~Tile.Tile8,holding,band_first(geometry,image,first,count),band_cells(geometry,first,count),Tile.filled(0.0),spare,
    LeafWrites.part_source(call_part(holding,spare,geometry,image,first,count,target,tiles,output)))

def call_spare_room(+holding: Held.Holding<Tile.Tile8>,+spare: Held.SpareModel<Tile.Tile8>,+spare_room: Held.SpareRoom(~Tile.Tile8,spare),+geometry: Kernel.Geometry,
  +image: U32,+first: U32,+count: U32,+target: U32,+tiles: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>,
  +allocatable: {U32.is_le(band_cells(geometry,first,count),1073741824) == True{} : Bool}) ->
  Held.SpareRoom(~Tile.Tile8,Held.spare_after(~Tile.Tile8,holding,band_first(geometry,image,first,count),band_cells(geometry,first,count),spare,
    LeafWrites.part_source(call_part(holding,spare,geometry,image,first,count,target,tiles,output)))):
  Held.band_spare_room(~Tile.Tile8,holding,band_first(geometry,image,first,count),band_cells(geometry,first,count),Tile.filled(0.0),spare,spare_room,allocatable,
    LeafWrites.part_source(call_part(holding,spare,geometry,image,first,count,target,tiles,output)),call_same(holding,spare,geometry,image,first,count,target,tiles,output))

def call_weights(+holding: Held.Holding<Tile.Tile8>,+spare: Held.SpareModel<Tile.Tile8>,+geometry: Kernel.Geometry,+image: U32,+first: U32,+count: U32,+target: U32,
  +tiles: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) ->
  {Buffer.tree(~Tile.Tile8,LeafWrites.part_weights(call_part(holding,spare,geometry,image,first,count,target,tiles,output))) == Buffer.tree(~Tile.Tile8,tiles) : Tree.Tree<Tile.Tile8>}:
  LeafWrites.leaf_weights(leaf_context(geometry,first,count,target),call_held(holding,spare,geometry,image,first,count),tiles,output)

# ---- The words of a call ----

# The gradient rows and columns of a call, its groups of eight outputs and
# the cells of its batch.
def rows_count(+call: Domain.Call) -> Nat:
  U32.to_nat(Domain.gradient_rows(Domain.geometry(call)))
def columns_count(+call: Domain.Call) -> Nat:
  U32.to_nat(Domain.gradient_columns(Domain.geometry(call)))
def groups_count(+call: Domain.Call) -> Nat:
  U32.to_nat(Packing.output_groups(Domain.geometry(call)))
def cells_count(+call: Domain.Call) -> Nat:
  Nat.mul(Nat.mul(Nat.mul(Domain.batch(call),rows_count(call)),columns_count(call)),groups_count(call))
def cells_word(+call: Domain.Call) -> U32:
  U32.from_nat(cells_count(call))
# The cells of count output rows of a call, and the cell of a channel at a
# row and column counted from the first row.
def region_cells(+call: Domain.Call,+count: U32) -> Nat:
  Nat.mul(Nat.mul(U32.to_nat(count),Domain.width(Domain.geometry(call))),Ceiling.ceiling(Domain.channels(Domain.geometry(call)),8n))
def output_offset(+geometry: Kernel.Geometry,+row: Nat,+column: Nat,+channel: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.add(Nat.mul(row,Domain.width(geometry)),column),Ceiling.ceiling(Domain.channels(geometry),8n)),Nat.div(channel,8n))

def nested(+b: Nat,+h: Nat,+w: Nat,+g: Nat) -> {Nat.mul(Nat.mul(Nat.mul(b,h),w),g) == Nat.mul(b,Nat.mul(h,Nat.mul(w,g))) : Nat}:
  Equal.trans(Nat,Nat.mul(Nat.mul(Nat.mul(b,h),w),g),Nat.mul(Nat.mul(b,h),Nat.mul(w,g)),Nat.mul(b,Nat.mul(h,Nat.mul(w,g))),
    RowMajor.assoc(Nat.mul(b,h),w,g),RowMajor.assoc(b,h,Nat.mul(w,g)))
def turned(+h: Nat,+w: Nat,+o: Nat) -> {Nat.mul(h,Nat.mul(w,o)) == Nat.mul(o,Nat.mul(h,w)) : Nat}:
  Equal.trans(Nat,Nat.mul(h,Nat.mul(w,o)),Nat.mul(Nat.mul(h,w),o),Nat.mul(o,Nat.mul(h,w)),
    Equal.sym(Nat,Nat.mul(Nat.mul(h,w),o),Nat.mul(h,Nat.mul(w,o)),RowMajor.assoc(h,w,o)),RowMajor.multiply_commutative(Nat.mul(h,w),o))

# The outputs of a call fit its machine, fill no more groups than outputs and
# no more outputs than the lanes of the groups.
def outputs_capacity(+call: Domain.Call) -> {Nat.is_le(Domain.outputs(Domain.geometry(call)),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool}:
  +limits = Domain.limits(call)
  Order.transitive(Domain.outputs(Domain.geometry(call)),U32.to_nat(Machine.dimension(limits)),U32.to_nat(Machine.capacity(limits)),Domain.outputs_room(call),
    Bounds.as_nat_le(Machine.dimension(limits),Machine.capacity(limits),Limits.dimension_within_capacity(limits)))
def groups_cover(+limits: Machine.Limits,+channels: U32,facts: TiledValues.Groups<limits,channels>) ->
  {Nat.is_le(U32.to_nat(channels),Nat.mul(U32.to_nat(Grid.groups(channels)),8n)) == True{} : Bool}:
  TiledValues.Groups{+exact,fewer,lanes} = facts
  %Equal.sym(Nat,U32.to_nat(Grid.groups(channels)),Ceiling.ceiling(U32.to_nat(channels),8n),exact) :
    {Nat.is_le(U32.to_nat(channels),Nat.mul(_,8n)) == True{} : Bool}
  Ceiling.covers(U32.to_nat(channels),8n,{==})
def outputs_grouped(+call: Domain.Call) -> {Nat.is_le(groups_count(call),Domain.outputs(Domain.geometry(call))) == True{} : Bool}:
  Domain.Call{+geometry,+batch,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +whole = {Domain.Call{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},batch,limits,facts} : Domain.Call}
  TiledValues.fewer_of(limits,o,TiledValues.grouped(limits,o,outputs_capacity(whole)))
def outputs_covered(+call: Domain.Call) -> {Nat.is_le(Domain.outputs(Domain.geometry(call)),Nat.mul(groups_count(call),8n)) == True{} : Bool}:
  Domain.Call{+geometry,+batch,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +whole = {Domain.Call{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},batch,limits,facts} : Domain.Call}
  groups_cover(limits,o,TiledValues.grouped(limits,o,outputs_capacity(whole)))
def groups_exact(+call: Domain.Call) -> {Packing.output_groups(Domain.geometry(call)) == U32.from_nat(groups_count(call)) : U32}:
  Equal.sym(U32,U32.from_nat(groups_count(call)),Packing.output_groups(Domain.geometry(call)),Arithmetic.native_roundtrip(Packing.output_groups(Domain.geometry(call))))

# The cells of the batch fit the machine.
def cells_room(+call: Domain.Call) -> {Nat.is_le(cells_count(call),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool}:
  Domain.Call{+geometry,+batch,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +whole = {Domain.Call{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},batch,limits,facts} : Domain.Call}
  +rows = U32.to_nat(gh)
  +columns = U32.to_nat(gw)
  +groups = groups_count(whole)
  +outputs = U32.to_nat(o)
  +capacity = U32.to_nat(Machine.capacity(limits))
  +fewer = DivisionOrder.multiply_left_le(batch,Nat.mul(rows,Nat.mul(columns,groups)),Nat.mul(rows,Nat.mul(columns,outputs)),
    DivisionOrder.multiply_left_le(rows,Nat.mul(columns,groups),Nat.mul(columns,outputs),DivisionOrder.multiply_left_le(columns,groups,outputs,outputs_grouped(whole))))
  +room = {%Equal.sym(Nat,Nat.mul(rows,Nat.mul(columns,outputs)),Nat.mul(outputs,Nat.mul(rows,columns)),turned(rows,columns,outputs)) :
      {Nat.is_le(Nat.mul(batch,_),capacity) == True{} : Bool}
    Domain.gradient_room(whole) : {Nat.is_le(Nat.mul(batch,Nat.mul(rows,Nat.mul(columns,outputs))),capacity) == True{} : Bool}}
  %Equal.sym(Nat,cells_count(whole),Nat.mul(batch,Nat.mul(rows,Nat.mul(columns,groups))),nested(batch,rows,columns,groups)) :
    {Nat.is_le(_,capacity) == True{} : Bool}
  Order.transitive(Nat.mul(batch,Nat.mul(rows,Nat.mul(columns,groups))),Nat.mul(batch,Nat.mul(rows,Nat.mul(columns,outputs))),capacity,fewer,room)
def cells_exact(+call: Domain.Call) -> {U32.to_nat(cells_word(call)) == cells_count(call) : Nat}:
  FullRange.roundtrip(cells_count(call),Machine.capacity(Domain.limits(call)),cells_room(call))
def allocatable(+word: U32,+limits: Machine.Limits,+within: {Nat.is_le(U32.to_nat(word),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.is_le(word,1073741824) == True{} : Bool}:
  +capacity = Machine.capacity(limits)
  Order.word_le(word,capacity,1073741824,
    {%Equal.sym(Bool,U32.is_le(word,capacity),Nat.is_le(U32.to_nat(word),U32.to_nat(capacity)),Comparison.le(word,capacity)) : {_ == True{} : Bool}
      within : {U32.is_le(word,capacity) == True{} : Bool}},
    Limits.capacity_allocatable(limits))
def cells_sized(+call: Domain.Call) -> Depth.Sized<cells_word(call)>:
  Depth.bounded(cells_word(call),allocatable(cells_word(call),Domain.limits(call),
    {%Equal.sym(Nat,U32.to_nat(cells_word(call)),cells_count(call),cells_exact(call)) : {Nat.is_le(_,U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool}
      cells_room(call) : {Nat.is_le(U32.to_nat(cells_word(call)),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool}}))

# Products and sums of words as the word of the product or sum of their
# numbers.
def product_word(+left: U32,+right: U32) -> {(left * right : U32) == U32.from_nat(Nat.mul(U32.to_nat(left),U32.to_nat(right))) : U32}:
  +result = U32.from_nat(Nat.mul(U32.to_nat(left),U32.to_nat(right)))
  %Arithmetic.native_roundtrip(left) : {(_ * right : U32) == result : U32}
  %Arithmetic.native_roundtrip(right) : {(U32.from_nat(U32.to_nat(left)) * _ : U32) == result : U32}
  Multiplication.native_encoded_multiply(U32.to_nat(left),U32.to_nat(right))
def triple_word(+rows: U32,+width: U32,+groups: U32) ->
  {(rows * (width * groups) : U32) == U32.from_nat(Nat.mul(U32.to_nat(rows),Nat.mul(U32.to_nat(width),U32.to_nat(groups)))) : U32}:
  +size = Nat.mul(U32.to_nat(width),U32.to_nat(groups))
  +result = U32.from_nat(Nat.mul(U32.to_nat(rows),size))
  %Equal.sym(U32,(width * groups : U32),U32.from_nat(size),product_word(width,groups)) : {(rows * _ : U32) == result : U32}
  %Arithmetic.native_roundtrip(rows) : {(_ * U32.from_nat(size) : U32) == result : U32}
  Multiplication.native_encoded_multiply(U32.to_nat(rows),size)
def band_word(+image: U32,+height: U32,+start: U32,+width: U32,+groups: U32) ->
  {((image * height + start) * (width * groups) : U32) ==
    U32.from_nat(Nat.mul(Nat.add(Nat.mul(U32.to_nat(image),U32.to_nat(height)),U32.to_nat(start)),Nat.mul(U32.to_nat(width),U32.to_nat(groups)))) : U32}:
  +row = Nat.add(Nat.mul(U32.to_nat(image),U32.to_nat(height)),U32.to_nat(start))
  +size = Nat.mul(U32.to_nat(width),U32.to_nat(groups))
  +result = U32.from_nat(Nat.mul(row,size))
  %Equal.sym(U32,(image * height + start : U32),U32.from_nat(row),Encoded.native_madd(image,height,start)) : {(_ * (width * groups) : U32) == result : U32}
  %Equal.sym(U32,(width * groups : U32),U32.from_nat(size),product_word(width,groups)) : {(U32.from_nat(row) * _ : U32) == result : U32}
  Multiplication.native_encoded_multiply(row,size)
def base_word(+image: U32,+height: U32,+width: U32,+groups: U32) ->
  {(image * (height * width) * groups : U32) ==
    U32.from_nat(Nat.mul(Nat.mul(U32.to_nat(image),Nat.mul(U32.to_nat(height),U32.to_nat(width))),U32.to_nat(groups))) : U32}:
  +offset = Nat.mul(U32.to_nat(image),Nat.mul(U32.to_nat(height),U32.to_nat(width)))
  +result = U32.from_nat(Nat.mul(offset,U32.to_nat(groups)))
  %Equal.sym(U32,(image * (height * width) : U32),U32.from_nat(offset),triple_word(image,height,width)) : {(_ * groups : U32) == result : U32}
  %Arithmetic.native_roundtrip(groups) : {(U32.from_nat(offset) * _ : U32) == result : U32}
  Multiplication.native_encoded_multiply(offset,U32.to_nat(groups))

# ---- The band of a call ----

# The first gradient row a call copies and the rows it copies are those of
# its band query.
def start_equal(+call: Domain.Call,+envelope: Geometry.WordEnvelope,+first: U32,+count: U32,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Domain.height(Domain.geometry(call))) == True{} : Bool}) ->
  {Kernel.copy_start(Domain.geometry(call),first,count) == Copy.begin(Bands.caller_query(call,envelope,first,count,contained)) : U32}:
  +band = Bands.caller_query(call,envelope,first,count,contained)
  +geometry = Domain.geometry(call)
  Equal.trans(U32,Kernel.copy_start(geometry,first,count),Kernel.copy_start(geometry,first,Bands.rows_word(band)),Copy.begin(band),
    Equal.cong(U32,U32,rows => Kernel.copy_start(geometry,first,rows),count,Bands.rows_word(band),
      Equal.sym(U32,Bands.rows_word(band),count,Bands.caller_count_matches(call,envelope,first,count,contained))),
    Copy.begin_matches(band))
def count_equal(+call: Domain.Call,+envelope: Geometry.WordEnvelope,+first: U32,+count: U32,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Domain.height(Domain.geometry(call))) == True{} : Bool}) ->
  {Kernel.copy_count(Domain.geometry(call),first,count) == Copy.row_count(Bands.caller_query(call,envelope,first,count,contained)) : U32}:
  +band = Bands.caller_query(call,envelope,first,count,contained)
  +geometry = Domain.geometry(call)
  Equal.trans(U32,Kernel.copy_count(geometry,first,count),Kernel.copy_count(geometry,first,Bands.rows_word(band)),Copy.row_count(band),
    Equal.cong(U32,U32,rows => Kernel.copy_count(geometry,first,rows),count,Bands.rows_word(band),
      Equal.sym(U32,Bands.rows_word(band),count,Bands.caller_count_matches(call,envelope,first,count,contained))),
    Copy.count_matches(band))

def range_rows(+first: U32,+rows: U32,+height: U32,range: SourceRows.Range<first,rows,height>) -> {Nat.is_le(U32.to_nat(rows),U32.to_nat(height)) == True{} : Bool}:
  match range:
    case SourceRows.EmptyRows{+empty}:
      %Equal.sym(U32,rows,0,empty) : {Nat.is_le(U32.to_nat(_),U32.to_nat(height)) == True{} : Bool}
      Order.zero_le(U32.to_nat(height))
    case SourceRows.ImageRows{+contained}:
      Order.transitive(U32.to_nat(rows),Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(height),Order.right_le(U32.to_nat(first),U32.to_nat(rows)),contained)

# The cells a call copies fit the cells of the batch.
def band_count_room(+call: Domain.Call,+envelope: Geometry.WordEnvelope,+first: U32,+count: U32,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Domain.height(Domain.geometry(call))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,Domain.batch(call)) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(U32.to_nat(Copy.row_count(Bands.caller_query(call,envelope,first,count,contained))),Nat.mul(columns_count(call),groups_count(call))),cells_count(call)) == True{} : Bool}:
  +band = Bands.caller_query(call,envelope,first,count,contained)
  +rows = U32.to_nat(Copy.row_count(band))
  +size = Nat.mul(columns_count(call),groups_count(call))
  +image = Nat.mul(rows_count(call),size)
  %Equal.sym(Nat,cells_count(call),Nat.mul(Domain.batch(call),image),nested(Domain.batch(call),rows_count(call),columns_count(call),groups_count(call))) :
    {Nat.is_le(Nat.mul(rows,size),_) == True{} : Bool}
  Order.transitive(Nat.mul(rows,size),image,Nat.mul(Domain.batch(call),image),
    RowMajor.mul_le(rows,rows_count(call),size,range_rows(Copy.begin(band),Copy.row_count(band),Domain.gradient_rows(Bands.geometry(band)),BandOrder.copy_range(band))),
    DivisionOrder.le_multiple(Domain.batch(call),image,positive))

# The first cell, the cell count and the image base of a call as numbers.
def band_first_exact(+call: Domain.Call,+envelope: Geometry.WordEnvelope,+image: U32,+first: U32,+count: U32,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Domain.height(Domain.geometry(call))) == True{} : Bool}) ->
  {band_first(Domain.geometry(call),image,first,count) ==
    U32.from_nat(Nat.mul(Nat.add(Nat.mul(U32.to_nat(image),rows_count(call)),U32.to_nat(Copy.begin(Bands.caller_query(call,envelope,first,count,contained)))),
      Nat.mul(columns_count(call),groups_count(call)))) : U32}:
  Domain.Call{+geometry,+batch,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +whole = {Domain.Call{shape,batch,limits,facts} : Domain.Call}
  +begin = Copy.begin(Bands.caller_query(whole,envelope,first,count,contained))
  +result = U32.from_nat(Nat.mul(Nat.add(Nat.mul(U32.to_nat(image),U32.to_nat(gh)),U32.to_nat(begin)),Nat.mul(U32.to_nat(gw),U32.to_nat(Tiled.groups(o)))))
  %Equal.sym(U32,Kernel.copy_start(shape,first,count),begin,start_equal(whole,envelope,first,count,contained)) :
    {((image * gh + _) * (gw * Tiled.groups(o)) : U32) == result : U32}
  band_word(image,gh,begin,gw,Tiled.groups(o))
def band_count_exact(+call: Domain.Call,+envelope: Geometry.WordEnvelope,+first: U32,+count: U32,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Domain.height(Domain.geometry(call))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,Domain.batch(call)) == True{} : Bool}) ->
  {U32.to_nat(band_cells(Domain.geometry(call),first,count)) ==
    Nat.mul(U32.to_nat(Copy.row_count(Bands.caller_query(call,envelope,first,count,contained))),Nat.mul(columns_count(call),groups_count(call))) : Nat}:
  Domain.Call{+geometry,+batch,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +shape = {Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} : Kernel.Geometry}
  +whole = {Domain.Call{shape,batch,limits,facts} : Domain.Call}
  +rows = Copy.row_count(Bands.caller_query(whole,envelope,first,count,contained))
  +product = Nat.mul(U32.to_nat(rows),Nat.mul(U32.to_nat(gw),U32.to_nat(Tiled.groups(o))))
  %Equal.sym(U32,Kernel.copy_count(shape,first,count),rows,count_equal(whole,envelope,first,count,contained)) :
    {U32.to_nat((_ * (gw * Tiled.groups(o)) : U32)) == product : Nat}
  %Equal.sym(U32,(rows * (gw * Tiled.groups(o)) : U32),U32.from_nat(product),triple_word(rows,gw,Tiled.groups(o))) : {U32.to_nat(_) == product : Nat}
  FullRange.roundtrip(product,Machine.capacity(limits),
    Order.transitive(product,cells_count(whole),U32.to_nat(Machine.capacity(limits)),band_count_room(whole,envelope,first,count,contained,positive),cells_room(whole)))
def band_allocatable(+call: Domain.Call,+envelope: Geometry.WordEnvelope,+first: U32,+count: U32,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Domain.height(Domain.geometry(call))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,Domain.batch(call)) == True{} : Bool}) ->
  {U32.is_le(band_cells(Domain.geometry(call),first,count),1073741824) == True{} : Bool}:
  +word = band_cells(Domain.geometry(call),first,count)
  +product = Nat.mul(U32.to_nat(Copy.row_count(Bands.caller_query(call,envelope,first,count,contained))),Nat.mul(columns_count(call),groups_count(call)))
  +capacity = U32.to_nat(Machine.capacity(Domain.limits(call)))
  allocatable(word,Domain.limits(call),
    {%Equal.sym(Nat,U32.to_nat(word),product,band_count_exact(call,envelope,first,count,contained,positive)) : {Nat.is_le(_,capacity) == True{} : Bool}
      Order.transitive(product,cells_count(call),capacity,band_count_room(call,envelope,first,count,contained,positive),cells_room(call)) :
      {Nat.is_le(U32.to_nat(word),capacity) == True{} : Bool}})
def image_base_exact(+call: Domain.Call,+image: U32) ->
  {Packing.cells_image_base(Domain.geometry(call),image) ==
    U32.from_nat(Nat.mul(Nat.mul(U32.to_nat(image),Nat.mul(rows_count(call),columns_count(call))),groups_count(call))) : U32}:
  Domain.Call{+geometry,+batch,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  base_word(image,gh,gw,Tiled.groups(o))

# The outputs a query of a call fills, below the lanes of its groups.
def query_outputs(+call: Domain.Call,+envelope: Geometry.WordEnvelope,+first: U32,+count: U32,+target: U32,+source: Execution.Source,
  +within: {Nat.is_le(U32.to_nat(count),Domain.height(Domain.geometry(call))) == True{} : Bool},
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Fill.outputs(Chunks.filling(Chunks.Query{GeometryChunks.fitted_grid(call,envelope,count,within,fits),first,target,source,0n,0n,0}))),
    Nat.mul(groups_count(call),8n)) == True{} : Bool}:
  Domain.Call{+geometry,+batch,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +whole = {Domain.Call{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},batch,limits,facts} : Domain.Call}
  outputs_covered(whole)

# ---- A call and what it reads ----

# The holding a call lends its band from: the cells it holds, from low to
# high, are the cells of a reader.
type Lender<-Holder: Data,-value: Holder -> U32 -> Tile.Tile8> is Data:
  Lender{holder: Holder,holding: Held.Holding<Tile.Tile8>,low: Nat,high: Nat,capacity: U32,
    capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
    tiled: Held.Tiled(~Tile.Tile8,holding,low,high),values: Held.HoldingValues(~Tile.Tile8,~Holder,~value,holder,holding),
    layout: Held.Spans,ordered: {Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding)) == layout : Held.Spans},
    spare: Held.SpareModel<Tile.Tile8>,spare_room: Held.SpareRoom(~Tile.Tile8,spare)}
def lender_holder(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,lender: Lender<Holder,value>) -> Holder:
  Lender{holder,holding,low,high,capacity,capacity_ok,bounded,tiled,values,layout,ordered,spare,spare_room} = lender
  holder
def lender_holding(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,lender: Lender<Holder,value>) -> Held.Holding<Tile.Tile8>:
  Lender{holder,holding,low,high,capacity,capacity_ok,bounded,tiled,values,layout,ordered,spare,spare_room} = lender
  holding
def lender_low(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,lender: Lender<Holder,value>) -> Nat:
  Lender{holder,holding,low,high,capacity,capacity_ok,bounded,tiled,values,layout,ordered,spare,spare_room} = lender
  low
def lender_high(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,lender: Lender<Holder,value>) -> Nat:
  Lender{holder,holding,low,high,capacity,capacity_ok,bounded,tiled,values,layout,ordered,spare,spare_room} = lender
  high
def lender_capacity(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,lender: Lender<Holder,value>) -> U32:
  Lender{holder,holding,low,high,capacity,capacity_ok,bounded,tiled,values,layout,ordered,spare,spare_room} = lender
  capacity
def lender_capacity_ok(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,lender: Lender<Holder,value>) ->
  {U32.is_le(lender_capacity(~Holder,~value,lender),2147483648) == True{} : Bool}:
  Lender{holder,holding,low,high,capacity,capacity_ok,bounded,tiled,values,layout,ordered,spare,spare_room} = lender
  capacity_ok
def lender_bounded(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,lender: Lender<Holder,value>) ->
  {Nat.is_le(lender_high(~Holder,~value,lender),U32.to_nat(lender_capacity(~Holder,~value,lender))) == True{} : Bool}:
  Lender{holder,holding,low,high,capacity,capacity_ok,bounded,tiled,values,layout,ordered,spare,spare_room} = lender
  bounded
def lender_tiled(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,lender: Lender<Holder,value>) ->
  Held.Tiled(~Tile.Tile8,lender_holding(~Holder,~value,lender),lender_low(~Holder,~value,lender),lender_high(~Holder,~value,lender)):
  Lender{holder,holding,low,high,capacity,capacity_ok,bounded,tiled,values,layout,ordered,spare,spare_room} = lender
  tiled
def lender_values(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,lender: Lender<Holder,value>) ->
  Held.HoldingValues(~Tile.Tile8,~Holder,~value,lender_holder(~Holder,~value,lender),lender_holding(~Holder,~value,lender)):
  Lender{holder,holding,low,high,capacity,capacity_ok,bounded,tiled,values,layout,ordered,spare,spare_room} = lender
  values
def lender_spare(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,lender: Lender<Holder,value>) -> Held.SpareModel<Tile.Tile8>:
  Lender{holder,holding,low,high,capacity,capacity_ok,bounded,tiled,values,layout,ordered,spare,spare_room} = lender
  spare
def lender_spare_room(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,lender: Lender<Holder,value>) ->
  Held.SpareRoom(~Tile.Tile8,lender_spare(~Holder,~value,lender)):
  Lender{holder,holding,low,high,capacity,capacity_ok,bounded,tiled,values,layout,ordered,spare,spare_room} = lender
  spare_room
type Call<-Holder: Data,-value: Holder -> U32 -> Tile.Tile8> is Data:
  Call{lender: Lender<Holder,value>,call: Domain.Call,envelope: Geometry.WordEnvelope,
    fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool},
    weights: Buffer.Model<Tile.Tile8>,tiles: Buffer.Model<Tile.Tile8>,
    tiles_exact: {Buffer.tree(~Tile.Tile8,tiles) == native_tree(packing(call,weights)) : Tree.Tree<Tile.Tile8>},
    image: U32,image_inside: {Nat.is_lt(U32.to_nat(image),Domain.batch(call)) == True{} : Bool},
    first: U32,count: U32,contained: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Domain.height(Domain.geometry(call))) == True{} : Bool},
    target: U32,output: Buffer.Model<Tile.Tile8>,depth_ok: {Nat.is_le(Buffer.depth(~Tile.Tile8,output),30n) == True{} : Bool},
    room: {Nat.is_le(Nat.add(U32.to_nat(target),region_cells(call,count)),U32.to_nat(Tree.capacity(Tile.Tile8,Buffer.tree(~Tile.Tile8,output)))) == True{} : Bool},
    zeros: Nat,zeros_cover: {Nat.is_le(region_cells(call,count),zeros) == True{} : Bool},
    zero: Members.All(Nat,{cell => {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),U32.from_nat(Nat.add(U32.to_nat(target),cell))) == Tile.filled((0.0 + 0.0 : F32)) : Tile.Tile8} : Nat -> Data},
      Ranges.range(zeros,0n)),
    placed: BandSource.Placed<lender_low(~Holder,~value,lender),lender_high(~Holder,~value,lender),band_first(Domain.geometry(call),image,first,count),
      band_cells(Domain.geometry(call),first,count)>}
def record_lender(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,record: Call<Holder,value>) -> Lender<Holder,value>:
  Call{lender,call,envelope,fits,weights,tiles,tiles_exact,image,image_inside,first,count,contained,target,output,depth_ok,room,zeros,zeros_cover,zero,placed} = record
  lender
def record_call(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,record: Call<Holder,value>) -> Domain.Call:
  Call{lender,call,envelope,fits,weights,tiles,tiles_exact,image,image_inside,first,count,contained,target,output,depth_ok,room,zeros,zeros_cover,zero,placed} = record
  call
def record_envelope(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,record: Call<Holder,value>) -> Geometry.WordEnvelope:
  Call{lender,call,envelope,fits,weights,tiles,tiles_exact,image,image_inside,first,count,contained,target,output,depth_ok,room,zeros,zeros_cover,zero,placed} = record
  envelope
def record_fits(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,record: Call<Holder,value>) ->
  {Kernel.fits(Domain.geometry(record_call(~Holder,~value,record)),Machine.capacity(Domain.limits(record_call(~Holder,~value,record)))) == True{} : Bool}:
  Call{lender,call,envelope,fits,weights,tiles,tiles_exact,image,image_inside,first,count,contained,target,output,depth_ok,room,zeros,zeros_cover,zero,placed} = record
  fits
def record_weights(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,record: Call<Holder,value>) -> Buffer.Model<Tile.Tile8>:
  Call{lender,call,envelope,fits,weights,tiles,tiles_exact,image,image_inside,first,count,contained,target,output,depth_ok,room,zeros,zeros_cover,zero,placed} = record
  weights
def record_tiles(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,record: Call<Holder,value>) -> Buffer.Model<Tile.Tile8>:
  Call{lender,call,envelope,fits,weights,tiles,tiles_exact,image,image_inside,first,count,contained,target,output,depth_ok,room,zeros,zeros_cover,zero,placed} = record
  tiles
def record_tiles_packed(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,record: Call<Holder,value>) ->
  {Buffer.tree(~Tile.Tile8,record_tiles(~Holder,~value,record)) == native_tree(packing(record_call(~Holder,~value,record),record_weights(~Holder,~value,record))) : Tree.Tree<Tile.Tile8>}:
  Call{lender,call,envelope,fits,weights,tiles,tiles_exact,image,image_inside,first,count,contained,target,output,depth_ok,room,zeros,zeros_cover,zero,placed} = record
  tiles_exact
def record_image(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,record: Call<Holder,value>) -> U32:
  Call{lender,call,envelope,fits,weights,tiles,tiles_exact,image,image_inside,first,count,contained,target,output,depth_ok,room,zeros,zeros_cover,zero,placed} = record
  image
def record_image_inside(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,record: Call<Holder,value>) ->
  {Nat.is_lt(U32.to_nat(record_image(~Holder,~value,record)),Domain.batch(record_call(~Holder,~value,record))) == True{} : Bool}:
  Call{lender,call,envelope,fits,weights,tiles,tiles_exact,image,image_inside,first,count,contained,target,output,depth_ok,room,zeros,zeros_cover,zero,placed} = record
  image_inside
def record_first(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,record: Call<Holder,value>) -> U32:
  Call{lender,call,envelope,fits,weights,tiles,tiles_exact,image,image_inside,first,count,contained,target,output,depth_ok,room,zeros,zeros_cover,zero,placed} = record
  first
def record_count(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,record: Call<Holder,value>) -> U32:
  Call{lender,call,envelope,fits,weights,tiles,tiles_exact,image,image_inside,first,count,contained,target,output,depth_ok,room,zeros,zeros_cover,zero,placed} = record
  count
def record_contained(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,record: Call<Holder,value>) ->
  {Nat.is_le(Nat.add(U32.to_nat(record_first(~Holder,~value,record)),U32.to_nat(record_count(~Holder,~value,record))),
    Domain.height(Domain.geometry(record_call(~Holder,~value,record)))) == True{} : Bool}:
  Call{lender,call,envelope,fits,weights,tiles,tiles_exact,image,image_inside,first,count,contained,target,output,depth_ok,room,zeros,zeros_cover,zero,placed} = record
  contained
def record_target(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,record: Call<Holder,value>) -> U32:
  Call{lender,call,envelope,fits,weights,tiles,tiles_exact,image,image_inside,first,count,contained,target,output,depth_ok,room,zeros,zeros_cover,zero,placed} = record
  target
def record_output(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,record: Call<Holder,value>) -> Buffer.Model<Tile.Tile8>:
  Call{lender,call,envelope,fits,weights,tiles,tiles_exact,image,image_inside,first,count,contained,target,output,depth_ok,room,zeros,zeros_cover,zero,placed} = record
  output
def record_depth_ok(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,record: Call<Holder,value>) ->
  {Nat.is_le(Buffer.depth(~Tile.Tile8,record_output(~Holder,~value,record)),30n) == True{} : Bool}:
  Call{lender,call,envelope,fits,weights,tiles,tiles_exact,image,image_inside,first,count,contained,target,output,depth_ok,room,zeros,zeros_cover,zero,placed} = record
  depth_ok
def record_capacity(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) -> U32:
  Tree.capacity(Tile.Tile8,Buffer.tree(~Tile.Tile8,record_output(~Holder,~value,record)))
def record_room(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,record: Call<Holder,value>) ->
  {Nat.is_le(Nat.add(U32.to_nat(record_target(~Holder,~value,record)),region_cells(record_call(~Holder,~value,record),record_count(~Holder,~value,record))),
    U32.to_nat(Tree.capacity(Tile.Tile8,Buffer.tree(~Tile.Tile8,record_output(~Holder,~value,record))))) == True{} : Bool}:
  Call{lender,call,envelope,fits,weights,tiles,tiles_exact,image,image_inside,first,count,contained,target,output,depth_ok,room,zeros,zeros_cover,zero,placed} = record
  room
def record_zeros(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,record: Call<Holder,value>) -> Nat:
  Call{lender,call,envelope,fits,weights,tiles,tiles_exact,image,image_inside,first,count,contained,target,output,depth_ok,room,zeros,zeros_cover,zero,placed} = record
  zeros
def record_zeros_cover(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,record: Call<Holder,value>) ->
  {Nat.is_le(region_cells(record_call(~Holder,~value,record),record_count(~Holder,~value,record)),record_zeros(~Holder,~value,record)) == True{} : Bool}:
  Call{lender,call,envelope,fits,weights,tiles,tiles_exact,image,image_inside,first,count,contained,target,output,depth_ok,room,zeros,zeros_cover,zero,placed} = record
  zeros_cover
def record_zero(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,record: Call<Holder,value>) ->
  Members.All(Nat,{cell => {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,record_output(~Holder,~value,record)),
      U32.from_nat(Nat.add(U32.to_nat(record_target(~Holder,~value,record)),cell))) == Tile.filled((0.0 + 0.0 : F32)) : Tile.Tile8} : Nat -> Data},
    Ranges.range(record_zeros(~Holder,~value,record),0n)):
  Call{lender,call,envelope,fits,weights,tiles,tiles_exact,image,image_inside,first,count,contained,target,output,depth_ok,room,zeros,zeros_cover,zero,placed} = record
  zero
def record_placed(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,record: Call<Holder,value>) ->
  BandSource.Placed<lender_low(~Holder,~value,record_lender(~Holder,~value,record)),lender_high(~Holder,~value,record_lender(~Holder,~value,record)),
    band_first(Domain.geometry(record_call(~Holder,~value,record)),record_image(~Holder,~value,record),record_first(~Holder,~value,record),record_count(~Holder,~value,record)),
    band_cells(Domain.geometry(record_call(~Holder,~value,record)),record_first(~Holder,~value,record),record_count(~Holder,~value,record))>:
  Call{lender,call,envelope,fits,weights,tiles,tiles_exact,image,image_inside,first,count,contained,target,output,depth_ok,room,zeros,zeros_cover,zero,placed} = record
  placed

# ---- The leaf of a call ----

def record_geometry(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) -> Kernel.Geometry:
  Domain.geometry(record_call(~Holder,~value,record))
def record_holding(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) -> Held.Holding<Tile.Tile8>:
  lender_holding(~Holder,~value,record_lender(~Holder,~value,record))
def record_spare(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) -> Held.SpareModel<Tile.Tile8>:
  lender_spare(~Holder,~value,record_lender(~Holder,~value,record))
def record_band(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) -> Move.Band<Tile.Tile8>:
  call_band(record_holding(~Holder,~value,record),record_spare(~Holder,~value,record),record_geometry(~Holder,~value,record),record_image(~Holder,~value,record),
    record_first(~Holder,~value,record),record_count(~Holder,~value,record))
def record_held(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) -> Execution.Held:
  call_held(record_holding(~Holder,~value,record),record_spare(~Holder,~value,record),record_geometry(~Holder,~value,record),record_image(~Holder,~value,record),
    record_first(~Holder,~value,record),record_count(~Holder,~value,record))
def record_context(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) -> Kernel.LeafContext:
  leaf_context(record_geometry(~Holder,~value,record),record_first(~Holder,~value,record),record_count(~Holder,~value,record),record_target(~Holder,~value,record))
def record_part(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) -> Execution.Part:
  call_part(record_holding(~Holder,~value,record),record_spare(~Holder,~value,record),record_geometry(~Holder,~value,record),record_image(~Holder,~value,record),
    record_first(~Holder,~value,record),record_count(~Holder,~value,record),record_target(~Holder,~value,record),
    record_tiles(~Holder,~value,record),record_output(~Holder,~value,record))

# The grid, the band query, the rows and the query of the leaf of a call.
def record_within(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) ->
  {Nat.is_le(U32.to_nat(record_count(~Holder,~value,record)),Domain.height(record_geometry(~Holder,~value,record))) == True{} : Bool}:
  +first = U32.to_nat(record_first(~Holder,~value,record))
  +count = U32.to_nat(record_count(~Holder,~value,record))
  Order.transitive(count,Nat.add(first,count),Domain.height(record_geometry(~Holder,~value,record)),Order.right_le(first,count),record_contained(~Holder,~value,record))
def record_grid(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) -> GeometryChunks.Grid:
  GeometryChunks.fitted_grid(record_call(~Holder,~value,record),record_envelope(~Holder,~value,record),record_count(~Holder,~value,record),
    record_within(~Holder,~value,record),record_fits(~Holder,~value,record))
def record_rows(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) -> Bands.Query:
  Bands.caller_query(record_call(~Holder,~value,record),record_envelope(~Holder,~value,record),record_first(~Holder,~value,record),
    record_count(~Holder,~value,record),record_contained(~Holder,~value,record))
def record_source(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) -> Execution.Source:
  Execution.Source{BandSource.band_tree(record_band(~Holder,~value,record)),Held.band_offset(~Tile.Tile8,record_band(~Holder,~value,record)),
    Copy.begin(record_rows(~Holder,~value,record)),Copy.row_count(record_rows(~Holder,~value,record))}
def record_query(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) -> Chunks.Query:
  Chunks.Query{record_grid(~Holder,~value,record),record_first(~Holder,~value,record),record_target(~Holder,~value,record),record_source(~Holder,~value,record),0n,0n,0}
# The gradient of the whole batch the leaf reads its rows from.
def record_model(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) -> Buffer.Model<Tile.Tile8>:
  BandSource.gradient_model(~Holder,~value,lender_holder(~Holder,~value,record_lender(~Holder,~value,record)),cells_word(record_call(~Holder,~value,record)))

# The band of a call holds the gradient rows of its band query.
def record_positive(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) ->
  {Nat.is_lt(0n,Domain.batch(record_call(~Holder,~value,record))) == True{} : Bool}:
  Order.positive_bound(U32.to_nat(record_image(~Holder,~value,record)),Domain.batch(record_call(~Holder,~value,record)),record_image_inside(~Holder,~value,record))
def record_lent(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) -> BandSource.Lent<Holder,value>:
  +lender = record_lender(~Holder,~value,record)
  +call = record_call(~Holder,~value,record)
  +geometry = Domain.geometry(call)
  +image = record_image(~Holder,~value,record)
  +first = record_first(~Holder,~value,record)
  +count = record_count(~Holder,~value,record)
  BandSource.Lent{lender_holder(~Holder,~value,lender),lender_holding(~Holder,~value,lender),lender_low(~Holder,~value,lender),lender_high(~Holder,~value,lender),
    lender_capacity(~Holder,~value,lender),lender_capacity_ok(~Holder,~value,lender),lender_bounded(~Holder,~value,lender),
    lender_tiled(~Holder,~value,lender),lender_values(~Holder,~value,lender),
    band_first(geometry,image,first,count),band_cells(geometry,first,count),Tile.filled(0.0),
    lender_spare(~Holder,~value,lender),lender_spare_room(~Holder,~value,lender),record_placed(~Holder,~value,record),
    band_allocatable(call,record_envelope(~Holder,~value,record),first,count,record_contained(~Holder,~value,record),record_positive(~Holder,~value,record))}
def record_outputs(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) ->
  {Nat.is_le(U32.to_nat(Fill.outputs(Chunks.filling(record_query(~Holder,~value,record)))),Nat.mul(groups_count(record_call(~Holder,~value,record)),8n)) == True{} : Bool}:
  query_outputs(record_call(~Holder,~value,record),record_envelope(~Holder,~value,record),record_first(~Holder,~value,record),record_count(~Holder,~value,record),
    record_target(~Holder,~value,record),record_source(~Holder,~value,record),record_within(~Holder,~value,record),record_fits(~Holder,~value,record))
def record_banded(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) -> BandSource.Banded<Holder,value>:
  +call = record_call(~Holder,~value,record)
  +envelope = record_envelope(~Holder,~value,record)
  +image = record_image(~Holder,~value,record)
  +first = record_first(~Holder,~value,record)
  +count = record_count(~Holder,~value,record)
  +contained = record_contained(~Holder,~value,record)
  BandSource.Banded{record_lent(~Holder,~value,record),record_rows(~Holder,~value,record),groups_count(call),groups_exact(call),
    Domain.batch(call),U32.to_nat(image),record_image_inside(~Holder,~value,record),
    band_first_exact(call,envelope,image,first,count,contained),band_count_exact(call,envelope,first,count,contained,record_positive(~Holder,~value,record)),
    Packing.cells_image_base(Domain.geometry(call),image),image_base_exact(call,image),
    cells_word(call),cells_sized(call),cells_exact(call),
    Fill.outputs(Chunks.filling(record_query(~Holder,~value,record))),record_outputs(~Holder,~value,record)}
def record_witness(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) ->
  Phases.ChunkSource(record_query(~Holder,~value,record),record_contained(~Holder,~value,record),record_image(~Holder,~value,record),record_model(~Holder,~value,record)):
  KernelGemm.Both{BandSource.band_witness(~Holder,~value,record_banded(~Holder,~value,record)),record_image_inside(~Holder,~value,record)}

# The leaf of a call, the layout of its output and what it writes.
def record_leaf(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) -> LeafWritten.Leaf:
  LeafWritten.Leaf{record_query(~Holder,~value,record),record_contained(~Holder,~value,record),record_fits(~Holder,~value,record),
    record_image(~Holder,~value,record),record_model(~Holder,~value,record),record_weights(~Holder,~value,record),record_witness(~Holder,~value,record)}
def record_layout(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) -> Output.Layout<record_query(~Holder,~value,record)>:
  +output = record_output(~Holder,~value,record)
  Output.Layout{Tree.capacity(Tile.Tile8,Buffer.tree(~Tile.Tile8,output)),Buffer.capacity_bound(~Tile.Tile8,output,record_depth_ok(~Holder,~value,record)),record_room(~Holder,~value,record)}
def record_written(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) -> Phases.Target:
  LeafWritten.target(record_leaf(~Holder,~value,record))
def record_writes(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) -> List<&2,Writes.Assignment<Tile.Tile8>>:
  +leaf = record_leaf(~Holder,~value,record)
  LeafWrites.leaf_writes(LeafWritten.native(leaf),Chunks.source(record_query(~Holder,~value,record)),Packed.weight_tree(LeafWritten.packed(leaf)))

# The rows a leaf reads are the band, and its packed weights are those of
# the call.
def band_tree_exact(+holding: Held.Holding<Tile.Tile8>,+spare: Held.SpareModel<Tile.Tile8>,+geometry: Kernel.Geometry,+image: U32,+first: U32,+count: U32) ->
  {Buffer.tree(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count)))) ==
    BandSource.band_tree(call_band(holding,spare,geometry,image,first,count)) : Tree.Tree<Tile.Tile8>}:
  +model = Buffer.remodel(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count)))
  Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,model),Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,model)),
    BandSource.band_tree(call_band(holding,spare,geometry,image,first,count)),
    Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,model)),Buffer.tree(~Tile.Tile8,model),Buffer.content_tree_of(~Tile.Tile8,model)),
    Equal.cong(Buffer.Content<Tile.Tile8>,Tree.Tree<Tile.Tile8>,content => Buffer.content_tree(~Tile.Tile8,content),Buffer.content(~Tile.Tile8,model),
      Buffer.contents(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count))),
      Buffer.remodel_content(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count)))))
def record_held_exact(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) ->
  {Execution.held_source(record_held(~Holder,~value,record)) == Chunks.source(record_query(~Holder,~value,record)) : Execution.Source}:
  +holding = record_holding(~Holder,~value,record)
  +spare = record_spare(~Holder,~value,record)
  +geometry = record_geometry(~Holder,~value,record)
  +call = record_call(~Holder,~value,record)
  +envelope = record_envelope(~Holder,~value,record)
  +image = record_image(~Holder,~value,record)
  +first = record_first(~Holder,~value,record)
  +count = record_count(~Holder,~value,record)
  +contained = record_contained(~Holder,~value,record)
  +band = Bands.caller_query(call,envelope,first,count,contained)
  +tree = BandSource.band_tree(call_band(holding,spare,geometry,image,first,count))
  +offset = Held.band_offset(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count))
  +wanted = {Execution.Source{tree,offset,Copy.begin(band),Copy.row_count(band)} : Execution.Source}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,call_band(holding,spare,geometry,image,first,count)))),tree,
      band_tree_exact(holding,spare,geometry,image,first,count)) :
    {Execution.Source{_,offset,Kernel.copy_start(geometry,first,count),Kernel.copy_count(geometry,first,count)} == wanted : Execution.Source}
  %Equal.sym(U32,Kernel.copy_start(geometry,first,count),Copy.begin(band),start_equal(call,envelope,first,count,contained)) :
    {Execution.Source{tree,offset,_,Kernel.copy_count(geometry,first,count)} == wanted : Execution.Source}
  %Equal.sym(U32,Kernel.copy_count(geometry,first,count),Copy.row_count(band),count_equal(call,envelope,first,count,contained)) :
    {Execution.Source{tree,offset,Copy.begin(band),_} == wanted : Execution.Source}
  {==}
def record_tiles_exact(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) ->
  {Buffer.tree(~Tile.Tile8,record_tiles(~Holder,~value,record)) ==
    Packed.weight_tree(Weights.full_state(Index.zero(Chunks.grid(record_query(~Holder,~value,record))),record_weights(~Holder,~value,record))) : Tree.Tree<Tile.Tile8>}:
  +call = record_call(~Holder,~value,record)
  +weights = record_weights(~Holder,~value,record)
  +grid = record_grid(~Holder,~value,record)
  Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,record_tiles(~Holder,~value,record)),native_tree(packing(call,weights)),
    Packed.weight_tree(Weights.full_state(Index.zero(grid),weights)),
    record_tiles_packed(~Holder,~value,record),packed_tree(call,weights,grid,{==}))

# The tiles of the output after the leaf are its tiles after the leaf's
# writes.
def record_result(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) ->
  {Buffer.tree(~Tile.Tile8,Execution.part_output(record_part(~Holder,~value,record))) ==
    Writes.pure(Tile.Tile8,record_writes(~Holder,~value,record),Buffer.tree(~Tile.Tile8,record_output(~Holder,~value,record))) : Tree.Tree<Tile.Tile8>}:
  +context = record_context(~Holder,~value,record)
  +held = record_held(~Holder,~value,record)
  +tiles = record_tiles(~Holder,~value,record)
  +output = record_output(~Holder,~value,record)
  +start = Buffer.tree(~Tile.Tile8,output)
  +source = Chunks.source(record_query(~Holder,~value,record))
  +packed = Packed.weight_tree(LeafWritten.packed(record_leaf(~Holder,~value,record)))
  +direct = Writes.pure(Tile.Tile8,LeafWrites.leaf_writes(context,Execution.held_source(held),Buffer.tree(~Tile.Tile8,tiles)),start)
  +kept = Writes.pure(Tile.Tile8,LeafWrites.leaf_writes(context,source,Buffer.tree(~Tile.Tile8,tiles)),start)
  +wanted = Writes.pure(Tile.Tile8,record_writes(~Holder,~value,record),start)
  +tiled = Buffer.tree(~Tile.Tile8,LeafWrites.leaf_tiles(context,held,tiles,output))
  Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,Execution.part_output(record_part(~Holder,~value,record))),tiled,wanted,
    Equal.cong(Buffer.Model<Tile.Tile8>,Tree.Tree<Tile.Tile8>,model => Buffer.tree(~Tile.Tile8,model),Execution.part_output(record_part(~Holder,~value,record)),
      LeafWrites.leaf_tiles(context,held,tiles,output),LeafWrites.leaf_values(context,held,tiles,output)),
    Equal.trans(Tree.Tree<Tile.Tile8>,tiled,direct,wanted,LeafWrites.leaf_tiles_writes(context,held,tiles,output),
      Equal.trans(Tree.Tree<Tile.Tile8>,direct,kept,wanted,
        Equal.cong(Execution.Source,Tree.Tree<Tile.Tile8>,rows => Writes.pure(Tile.Tile8,LeafWrites.leaf_writes(context,rows,Buffer.tree(~Tile.Tile8,tiles)),start),
          Execution.held_source(held),source,record_held_exact(~Holder,~value,record)),
        Equal.cong(Tree.Tree<Tile.Tile8>,Tree.Tree<Tile.Tile8>,weights => Writes.pure(Tile.Tile8,LeafWrites.leaf_writes(context,source,weights),start),
          Buffer.tree(~Tile.Tile8,tiles),packed,record_tiles_exact(~Holder,~value,record)))))

# ---- What a call leaves ----

# The lane of a channel at a point of the rows of a call holds the input
# gradient the gradient and the weights give there.
def record_lane(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>,+channel: Nat,+row: Nat,+column: Nat,
  +channel_inside: {Nat.is_lt(channel,Domain.channels(record_geometry(~Holder,~value,record))) == True{} : Bool},
  +lower: {Nat.is_le(U32.to_nat(record_first(~Holder,~value,record)),row) == True{} : Bool},
  +upper: {Nat.is_lt(row,Nat.add(U32.to_nat(record_first(~Holder,~value,record)),U32.to_nat(record_count(~Holder,~value,record)))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,Domain.width(record_geometry(~Holder,~value,record))) == True{} : Bool}) ->
  {TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,Execution.part_output(record_part(~Holder,~value,record))),
      U32.from_nat(Nat.add(U32.to_nat(record_target(~Holder,~value,record)),
        output_offset(record_geometry(~Holder,~value,record),Nat.sub(row,U32.to_nat(record_first(~Holder,~value,record))),column,channel))))) ==
    Specification.value(Sources.problem(record_geometry(~Holder,~value,record),
      Sources.models_reader(record_geometry(~Holder,~value,record),record_model(~Holder,~value,record),record_weights(~Holder,~value,record)),
      U32.to_nat(record_image(~Holder,~value,record))),channel,row,column) : F32}:
  +geometry = record_geometry(~Holder,~value,record)
  +query = record_query(~Holder,~value,record)
  +contained = record_contained(~Holder,~value,record)
  +image = record_image(~Holder,~value,record)
  +first = U32.to_nat(record_first(~Holder,~value,record))
  +count = U32.to_nat(record_count(~Holder,~value,record))
  +target = U32.to_nat(record_target(~Holder,~value,record))
  +output = record_output(~Holder,~value,record)
  +held = record_held(~Holder,~value,record)
  +tiles = record_tiles(~Holder,~value,record)
  +context = record_context(~Holder,~value,record)
  +zero = (0.0 + 0.0 : F32)
  +lane = Nat.mod(channel,8n)
  +point = {Domain.Point{U32.to_nat(image),channel,row,column} : Domain.Point}
  +row_inside = Bounds.nat_trans(row,Nat.add(first,count),Domain.height(geometry),upper,contained)
  +problem = {Producer.Problem{query,point,Domain.Inside{record_image_inside(~Holder,~value,record),channel_inside,row_inside,column_inside},lower,upper,contained} : Producer.Problem}
  +requested = Coverage.requested(problem)
  +area = Output.region(query)
  +offset = Output.offset(area,requested)
  +cells = region_cells(record_call(~Holder,~value,record),record_count(~Holder,~value,record))
  +zeros = record_zeros(~Holder,~value,record)
  +cell_zero = Members.all_member(Nat,
    {cell => {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),U32.from_nat(Nat.add(target,cell))) == Tile.filled(zero) : Tile.Tile8} : Nat -> Data},
    offset,Ranges.range(zeros,0n),record_zero(~Holder,~value,record),
    Members.range_member(zeros,0n,offset,Bounds.nat_trans(offset,cells,zeros,Output.offset_inside(area,requested),record_zeros_cover(~Holder,~value,record))))
  +initial_zero = {%Equal.sym(Tile.Tile8,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),U32.from_nat(Nat.add(target,offset))),Tile.filled(zero),cell_zero) :
      {TileLaws.select_lane(lane,_) == zero : F32}
    TileLaws.filled_lane(lane,zero) : {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,output),Coverage.address(problem))) == zero : F32}}
  +leaf = {Leaf.Context{problem,image,record_model(~Holder,~value,record),record_weights(~Holder,~value,record),output,held,tiles,
    record_witness(~Holder,~value,record),record_fits(~Holder,~value,record),record_layout(~Holder,~value,record),{==},record_depth_ok(~Holder,~value,record),
    initial_zero,record_held_exact(~Holder,~value,record),record_tiles_exact(~Holder,~value,record)} : Leaf.Context}
  %Equal.sym(Buffer.Model<Tile.Tile8>,Execution.part_output(record_part(~Holder,~value,record)),LeafWrites.leaf_tiles(context,held,tiles,output),
      LeafWrites.leaf_values(context,held,tiles,output)) :
    {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,_),Coverage.address(problem))) == Leaf.wanted(leaf) : F32}
  Leaf.value(leaf)

# Rows of a call leave the output as balanced and as long as it was.
def record_balanced(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) ->
  Shape.Balanced(Tile.Tile8,Buffer.depth(~Tile.Tile8,record_output(~Holder,~value,record)),Buffer.tree(~Tile.Tile8,Execution.part_output(record_part(~Holder,~value,record)))):
  +output = record_output(~Holder,~value,record)
  +start = Buffer.tree(~Tile.Tile8,output)
  +writes = record_writes(~Holder,~value,record)
  %Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,Execution.part_output(record_part(~Holder,~value,record))),Writes.pure(Tile.Tile8,writes,start),
      record_result(~Holder,~value,record)) :
    Shape.Balanced(Tile.Tile8,Buffer.depth(~Tile.Tile8,output),_)
  Writes.writes_preserve_balance(Tile.Tile8,writes,start,Buffer.depth(~Tile.Tile8,output),Buffer.balanced(~Tile.Tile8,output))
def record_depth(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) ->
  {Buffer.depth(~Tile.Tile8,Execution.part_output(record_part(~Holder,~value,record))) == Buffer.depth(~Tile.Tile8,record_output(~Holder,~value,record)) : Nat}:
  +result = Execution.part_output(record_part(~Holder,~value,record))
  Shape.unique_depth(Tile.Tile8,Buffer.tree(~Tile.Tile8,result),Buffer.depth(~Tile.Tile8,result),Buffer.depth(~Tile.Tile8,record_output(~Holder,~value,record)),
    Buffer.balanced(~Tile.Tile8,result),record_balanced(~Holder,~value,record))
def record_kept_capacity(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) ->
  {Tree.capacity(Tile.Tile8,Buffer.tree(~Tile.Tile8,Execution.part_output(record_part(~Holder,~value,record)))) == record_capacity(~Holder,~value,record) : U32}:
  +output = record_output(~Holder,~value,record)
  +start = Buffer.tree(~Tile.Tile8,output)
  +writes = record_writes(~Holder,~value,record)
  %Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,Execution.part_output(record_part(~Holder,~value,record))),Writes.pure(Tile.Tile8,writes,start),
      record_result(~Holder,~value,record)) :
    {Tree.capacity(Tile.Tile8,_) == Tree.capacity(Tile.Tile8,start) : U32}
  Writes.writes_preserve_capacity(Tile.Tile8,writes,start,Buffer.depth(~Tile.Tile8,output),Buffer.balanced(~Tile.Tile8,output))
def record_length(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>) ->
  {Buffer.length(~Tile.Tile8,Execution.part_output(record_part(~Holder,~value,record))) == Buffer.length(~Tile.Tile8,record_output(~Holder,~value,record)) : Nat}:
  LeafLength.leaf_band_length(record_context(~Holder,~value,record),record_held(~Holder,~value,record),record_tiles(~Holder,~value,record),record_output(~Holder,~value,record))

# A cell before the cells of a call or past them.
type Outside<-start: Nat,-size: Nat,-index: Nat> is Data:
  Before{before: {Nat.is_lt(index,start) == True{} : Bool}}
  Beyond{beyond: {Nat.is_le(Nat.add(start,size),index) == True{} : Bool}}
def apart(+start: Nat,+size: Nat,+index: Nat,outside: Outside<start,size,index>,+offset: Nat,+inside: {Nat.is_lt(offset,size) == True{} : Bool}) ->
  {Nat.is_eq(Nat.add(start,offset),index) == False{} : Bool}:
  match outside:
    case Before{+before}:
      %Equal.sym(Bool,Nat.is_eq(Nat.add(start,offset),index),Nat.is_eq(index,Nat.add(start,offset)),Order.equal_symmetric(Nat.add(start,offset),index)) :
        {_ == False{} : Bool}
      Order.less_not_equal(index,Nat.add(start,offset),Bounds.nat_trans(index,start,Nat.add(start,offset),before,Order.le_add(start,offset)))
    case Beyond{+beyond}:
      Order.less_not_equal(Nat.add(start,offset),index,Bounds.nat_trans(Nat.add(start,offset),Nat.add(start,size),index,Order.offset_lt(start,offset,size,inside),beyond))

# Every write of a call lands in the output, and none on a cell outside its
# rows.
def record_inside(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>,+address: U32,+tile: Tile.Tile8,
  holds: Phases.Written(record_written(~Holder,~value,record),address,tile)) -> {U32.is_lt(address,record_capacity(~Holder,~value,record)) == True{} : Bool}:
  Phases.Witness{+point,+at,l0,l1,l2,l3,l4,l5,l6,l7} = holds
  +query = record_query(~Holder,~value,record)
  %at : {U32.is_lt(_,record_capacity(~Holder,~value,record)) == True{} : Bool}
  Leaf.native_inside(Output.region(query),Output.region_layout(query,record_layout(~Holder,~value,record)),point)
type Scope<-Holder: Data,-value: Holder -> U32 -> Tile.Tile8> is Data:
  Scope{record: Call<Holder,value>,index: Nat,
    outside: Outside<U32.to_nat(record_target(~Holder,~value,record)),region_cells(record_call(~Holder,~value,record),record_count(~Holder,~value,record)),index>,
    inside: {Nat.is_lt(index,U32.to_nat(record_capacity(~Holder,~value,record))) == True{} : Bool}}
def scope_record(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,scope: Scope<Holder,value>) -> Call<Holder,value>:
  Scope{record,index,outside,inside} = scope
  record
def scope_index(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,scope: Scope<Holder,value>) -> Nat:
  Scope{record,index,outside,inside} = scope
  index
def scope_written(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+scope: Scope<Holder,value>) -> Phases.Target:
  record_written(~Holder,~value,scope_record(~Holder,~value,scope))
def scope_word(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+scope: Scope<Holder,value>) -> U32:
  U32.from_nat(scope_index(~Holder,~value,scope))
def scope_apart(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+scope: Scope<Holder,value>,+address: U32,+tile: Tile.Tile8,
  holds: Phases.Written(scope_written(~Holder,~value,scope),address,tile)) -> {U32.is_eq(address,scope_word(~Holder,~value,scope)) == False{} : Bool}:
  Scope{+record,+index,+outside,+inside} = scope
  Phases.Witness{+point,+at,l0,l1,l2,l3,l4,l5,l6,l7} = holds
  +query = record_query(~Holder,~value,record)
  +area = Output.region(query)
  +layout = Output.region_layout(query,record_layout(~Holder,~value,record))
  +word = U32.from_nat(index)
  +capacity = record_capacity(~Holder,~value,record)
  +native = Output.native_address(area,point)
  %at : {U32.is_eq(_,word) == False{} : Bool}
  Equal.trans(Bool,U32.is_eq(native,word),Nat.is_eq(U32.to_nat(native),U32.to_nat(word)),False{},Comparison.eq(native,word),
    {%Equal.sym(Nat,U32.to_nat(native),Output.address(area,point),Output.native_address_exact(area,layout,point)) :
        {Nat.is_eq(_,U32.to_nat(word)) == False{} : Bool}
      %Equal.sym(Nat,U32.to_nat(word),index,FullRange.roundtrip(index,capacity,Bounds.strict_le(index,U32.to_nat(capacity),inside))) :
        {Nat.is_eq(Output.address(area,point),_) == False{} : Bool}
      apart(U32.to_nat(record_target(~Holder,~value,record)),region_cells(record_call(~Holder,~value,record),record_count(~Holder,~value,record)),index,outside,
        Output.offset(area,point),Output.offset_inside(area,point)) :
      {Nat.is_eq(U32.to_nat(native),U32.to_nat(word)) == False{} : Bool}})

# A cell of the output outside the rows of a call keeps its tile.
def record_kept(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+record: Call<Holder,value>,+index: Nat,
  +outside: Outside<U32.to_nat(record_target(~Holder,~value,record)),region_cells(record_call(~Holder,~value,record),record_count(~Holder,~value,record)),index>,
  +inside: {Nat.is_lt(index,U32.to_nat(record_capacity(~Holder,~value,record))) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,Execution.part_output(record_part(~Holder,~value,record))),U32.from_nat(index)) ==
    Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,record_output(~Holder,~value,record)),U32.from_nat(index)) : Tile.Tile8}:
  +output = record_output(~Holder,~value,record)
  +start = Buffer.tree(~Tile.Tile8,output)
  +writes = record_writes(~Holder,~value,record)
  +word = U32.from_nat(index)
  +capacity = record_capacity(~Holder,~value,record)
  +scope = {Scope{record,index,outside,inside} : Scope<Holder,value>}
  +holds = LeafWritten.leaf_written(record_leaf(~Holder,~value,record))
  +word_inside = Equal.trans(Bool,U32.is_lt(word,capacity),Nat.is_lt(U32.to_nat(word),U32.to_nat(capacity)),True{},Comparison.lt(word,capacity),
    {%Equal.sym(Nat,U32.to_nat(word),index,FullRange.roundtrip(index,capacity,Bounds.strict_le(index,U32.to_nat(capacity),inside))) :
        {Nat.is_lt(_,U32.to_nat(capacity)) == True{} : Bool}
      inside : {Nat.is_lt(U32.to_nat(word),U32.to_nat(capacity)) == True{} : Bool}})
  %Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,Execution.part_output(record_part(~Holder,~value,record))),Writes.pure(Tile.Tile8,writes,start),
      record_result(~Holder,~value,record)) :
    {Tree.read(Tile.Tile8,_,word) == Tree.read(Tile.Tile8,start,word) : Tile.Tile8}
  Readback.excluded_readback(~Tile.Tile8,writes,start,Buffer.depth(~Tile.Tile8,output),word,Buffer.balanced(~Tile.Tile8,output),record_depth_ok(~Holder,~value,record),
    Every.inside(~Tile.Tile8,~Phases.Target,~Phases.Written,~Call<Holder,value>,~record_written(~Holder,~value),~record_capacity(~Holder,~value),
      ~record_inside(~Holder,~value),record,writes,holds),
    word_inside,
    Every.excludes(~Tile.Tile8,~Phases.Target,~Phases.Written,~Scope<Holder,value>,~scope_written(~Holder,~value),~scope_word(~Holder,~value),
      ~scope_apart(~Holder,~value),scope,writes,holds))
