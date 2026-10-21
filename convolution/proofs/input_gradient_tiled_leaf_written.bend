# Every write of a leaf stores the specified value at its own output point:
# the chunks of the leaf in order, each one reading tiles filled from the
# source rows and the packed weights and table of the whole geometry.
import Base
import ../../lib/kernel_input_gradient.bend as Kernel
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/nat_order.bend as NatOrder
import ../../lib/proofs/write_program_every.bend as Every
import ../../lib/proofs/write_program_fold.bend as Fold
import ../../lib/proofs/input_gradient_tiled_chunk_reads.bend as Chunks
import ../../lib/proofs/input_gradient_tiled_chunk_geometry.bend as GeometryChunks
import ../../lib/proofs/input_gradient_tiled_chunk_indices.bend as Indices
import ../../lib/proofs/input_gradient_tiled_leaf_writes.bend as LeafWrites
import ../../lib/proofs/input_gradient_tiled_leaf_execution.bend as Execution
import ../../lib/proofs/input_gradient_tiled_output_layout.bend as Output
import ../../lib/proofs/input_gradient_tiled_phase_index.bend as Index
import ../../lib/proofs/input_gradient_tiled_pack_values.bend as Packed
import ../../lib/proofs/input_gradient_tiled_weight_reads.bend as Weights
import ../../lib/proofs/input_gradient_tiled_sweep_selected.bend as Selected
import ../../lib/proofs/input_gradient_tiled_weights.bend as WeightPacking
import ../../lib/proofs/input_gradient_tiled_jobs.bend as NativeJobs
import ../../lib/proofs/input_gradient_tiled_fill.bend as Fill
import ./input_gradient_sources.bend as Sources
import ./input_gradient_tiled_written.bend as Phases
import ./input_gradient_tiled_chunk_written.bend as ChunkWritten

# A leaf and what all its chunks share.
type Leaf is Data:
  Leaf{query: Chunks.Query,contained: Phases.Contained(query),fits: Phases.Fits(query),
    image: U32,gradient: Buffer.Model<F32>,weights: Buffer.Model<F32>,
    source: Phases.ChunkSource(query,contained,image,gradient)}
def query(leaf: Leaf) -> Chunks.Query:
  Leaf{query,contained,fits,image,gradient,weights,source} = leaf
  query
def contained(leaf: Leaf) -> Phases.Contained(query(leaf)):
  Leaf{query,contained,fits,image,gradient,weights,source} = leaf
  contained
def fits(leaf: Leaf) -> Phases.Fits(query(leaf)):
  Leaf{query,contained,fits,image,gradient,weights,source} = leaf
  fits
def image(leaf: Leaf) -> U32:
  Leaf{query,contained,fits,image,gradient,weights,source} = leaf
  image
def gradient(leaf: Leaf) -> Buffer.Model<F32>:
  Leaf{query,contained,fits,image,gradient,weights,source} = leaf
  gradient
def weights(leaf: Leaf) -> Buffer.Model<F32>:
  Leaf{query,contained,fits,image,gradient,weights,source} = leaf
  weights
def source(leaf: Leaf) -> Phases.ChunkSource(query(leaf),contained(leaf),image(leaf),gradient(leaf)):
  Leaf{query,contained,fits,image,gradient,weights,source} = leaf
  source

def target(+leaf: Leaf) -> Phases.Target:
  +query = query(leaf)
  Phases.Target{Output.region(query),Sources.problem(Chunks.geometry(query),Sources.models_reader(gradient(leaf),weights(leaf)),U32.to_nat(image(leaf)))}
def native(+leaf: Leaf) -> Kernel.LeafContext:
  Selected.native_context(query(leaf))
def chunk_count(+leaf: Leaf) -> Nat:
  Indices.count(Chunks.grid(query(leaf)))
# The weights and the table packed for every phase of the geometry, and the
# inputs every chunk reads: those and the source rows.
def packed(+leaf: Leaf) -> WeightPacking.Packed:
  Weights.full_state(Index.zero(Chunks.grid(query(leaf))),weights(leaf))
def inputs(+leaf: Leaf) -> LeafWrites.Inputs:
  LeafWrites.Inputs{Chunks.source(query(leaf)),Packed.weight_tree(packed(leaf)),Packed.offset_tree(packed(leaf))}

# The chunk at a position.
def chunk_query(+leaf: Leaf,+position: Nat) -> Chunks.Query:
  +query = query(leaf)
  +grid = Chunks.grid(query)
  Output.scheduled(query,Indices.row(grid,position),Indices.group(grid,position),0,0)
def chunk(+leaf: Leaf,+position: Nat,+inside: {Nat.is_lt(position,chunk_count(leaf)) == True{} : Bool}) -> ChunkWritten.Chunk:
  ChunkWritten.Chunk{chunk_query(leaf,position),Indices.inside(Chunks.grid(query(leaf)),position,inside),contained(leaf),fits(leaf),
    image(leaf),gradient(leaf),weights(leaf),source(leaf)}

# The writes of the chunk at a position are its phases' writes.
def chunk_written(+leaf: Leaf,+position: Nat,+inside: {Nat.is_lt(position,chunk_count(leaf)) == True{} : Bool}) ->
  Every.Every(~F32,~Phases.Target,~Phases.Written,target(leaf),LeafWrites.chunk_writes(U32.from_nat(position),native(leaf),inputs(leaf))):
  +query = query(leaf)
  +grid = Chunks.grid(query)
  +row = Indices.row(grid,position)
  +group = Indices.group(grid,position)
  +shape = Chunks.shape(chunk_query(leaf,position))
  +bases = {LeafWrites.Bases{NativeJobs.Readers{Fill.tree(Execution.fill_chunk(shape,Chunks.source(query))),Packed.weight_tree(packed(leaf)),Packed.offset_tree(packed(leaf))},0,0} : LeafWrites.Bases}
  %Indices.native_exact(grid,position,inside) :
    Every.Every(~F32,~Phases.Target,~Phases.Written,target(leaf),LeafWrites.chunk_writes(_,native(leaf),inputs(leaf)))
  %Equal.sym(Kernel.Chunk,Execution.shape_chunk(shape),GeometryChunks.selected_chunk(grid,Chunks.first(query),Chunks.target(query),Chunks.plane(query),row,group),
    GeometryChunks.selected_chunk_matches(grid,Chunks.first(query),Chunks.target(query),Chunks.plane(query),row,group,Chunks.source(query))) :
    Every.Every(~F32,~Phases.Target,~Phases.Written,target(leaf),LeafWrites.phases_writes(Kernel.window_jobs(LeafWrites.chunk_geometry(_)),0,_,bases))
  ChunkWritten.chunk_written(chunk(leaf,position,inside))

# What the fold over the chunks of a leaf keeps: the chunks left fit in the
# leaf, and the leaf context and the inputs are the leaf's.
def ChunksLeft(+leaf: Leaf,rest: Nat,position: Nat,context: Kernel.LeafContext,current: LeafWrites.Inputs) -> Data:
  Shape.Both<{Nat.is_le(Nat.add(position,rest),chunk_count(leaf)) == True{} : Bool},
    Shape.Both<{context == native(leaf) : Kernel.LeafContext},{current == inputs(leaf) : LeafWrites.Inputs}>>
def chunk_step(+leaf: Leaf,+rest: Nat,+position: Nat,+context: Kernel.LeafContext,+current: LeafWrites.Inputs,left: ChunksLeft(leaf,1n+rest,position,context,current)) ->
  Every.Every(~F32,~Phases.Target,~Phases.Written,target(leaf),LeafWrites.chunk_writes(U32.from_nat(position),context,current)):
  Shape.Both{room,same} = left
  Shape.Both{same_context,same_inputs} = same
  +inside = NatOrder.room_index(position,rest,chunk_count(leaf),room)
  %Equal.sym(Kernel.LeafContext,context,native(leaf),same_context) :
    Every.Every(~F32,~Phases.Target,~Phases.Written,target(leaf),LeafWrites.chunk_writes(U32.from_nat(position),_,current))
  %Equal.sym(LeafWrites.Inputs,current,inputs(leaf),same_inputs) :
    Every.Every(~F32,~Phases.Target,~Phases.Written,target(leaf),LeafWrites.chunk_writes(U32.from_nat(position),native(leaf),_))
  chunk_written(leaf,position,inside)
def chunks_left_kept(+leaf: Leaf,+rest: Nat,+position: Nat,+context: Kernel.LeafContext,+current: LeafWrites.Inputs,left: ChunksLeft(leaf,1n+rest,position,context,current)) ->
  ChunksLeft(leaf,rest,1n+position,context,LeafWrites.inputs_kept(U32.from_nat(position),context,current)):
  Shape.Both{room,same} = left
  Shape.Both{NatOrder.advance_room(position,rest,chunk_count(leaf),room),same}

# Every write of a leaf stores the specified value at its own output point.
def leaf_written(+leaf: Leaf) ->
  Every.Every(~F32,~Phases.Target,~Phases.Written,target(leaf),
    LeafWrites.leaf_writes(native(leaf),Chunks.source(query(leaf)),Packed.weight_tree(packed(leaf)),Packed.offset_tree(packed(leaf)))):
  +count = chunk_count(leaf)
  %Equal.sym(Nat,Execution.chunk_count(native(leaf)),count,Selected.count(query(leaf))) :
    Every.Every(~F32,~Phases.Target,~Phases.Written,target(leaf),LeafWrites.chunks_writes(_,0,native(leaf),inputs(leaf)))
  Fold.every(~F32,~Kernel.LeafContext,~LeafWrites.Inputs,~LeafWrites.chunk_writes,~LeafWrites.inputs_kept,~Leaf,~Phases.Target,~target,~Phases.Written,~ChunksLeft,~chunk_step,~chunks_left_kept,
    leaf,count,0n,native(leaf),inputs(leaf),Shape.Both{NatOrder.reflexive(count),Shape.Both{{==},{==}}})
