# Every write of a chunk stores the specified value at its own output point:
# the phases of the chunk in order, each one's jobs read from its packed
# weight and table bases.
import Base
import ../../lib/kernel_input_gradient.bend as Kernel
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/nat_order.bend as NatOrder
import ../../lib/proofs/write_program_every.bend as Every
import ../../lib/proofs/write_program_fold.bend as Fold
import ../../lib/proofs/input_gradient_tiled_chunk_reads.bend as Chunks
import ../../lib/proofs/input_gradient_tiled_jobs.bend as NativeJobs
import ../../lib/proofs/input_gradient_tiled_leaf_writes.bend as LeafWrites
import ../../lib/proofs/input_gradient_tiled_leaf_execution.bend as Execution
import ../../lib/proofs/input_gradient_tiled_output_layout.bend as Output
import ../../lib/proofs/input_gradient_tiled_phase_index.bend as Index
import ../../lib/proofs/input_gradient_tiled_product_reads.bend as Runtime
import ../../lib/proofs/input_gradient_tiled_packing_bounds.bend as PackingBounds
import ../../lib/proofs/input_gradient_tiled_pack_values.bend as Packed
import ../../lib/proofs/input_gradient_tiled_weight_reads.bend as Weights
import ../../lib/proofs/input_gradient_tiled_fill.bend as Fill
import ../../lib/proofs/input_gradient_tiled_job_geometry.bend as Jobs
import ./input_gradient_sources.bend as Sources
import ./input_gradient_tiled_written.bend as Phases

# A chunk of a leaf and what all its phases share.
type Chunk is Data:
  Chunk{query: Chunks.Query,inside: Phases.ChunkInside(query),contained: Phases.Contained(query),fits: Phases.Fits(query),
    image: U32,gradient: Buffer.Model<F32>,weights: Buffer.Model<F32>,
    source: Phases.ChunkSource(query,contained,image,gradient)}
def query(chunk: Chunk) -> Chunks.Query:
  Chunk{query,inside,contained,fits,image,gradient,weights,source} = chunk
  query
def chunk_inside(chunk: Chunk) -> Phases.ChunkInside(query(chunk)):
  Chunk{query,inside,contained,fits,image,gradient,weights,source} = chunk
  inside
def contained(chunk: Chunk) -> Phases.Contained(query(chunk)):
  Chunk{query,inside,contained,fits,image,gradient,weights,source} = chunk
  contained
def fits(chunk: Chunk) -> Phases.Fits(query(chunk)):
  Chunk{query,inside,contained,fits,image,gradient,weights,source} = chunk
  fits
def image(chunk: Chunk) -> U32:
  Chunk{query,inside,contained,fits,image,gradient,weights,source} = chunk
  image
def gradient(chunk: Chunk) -> Buffer.Model<F32>:
  Chunk{query,inside,contained,fits,image,gradient,weights,source} = chunk
  gradient
def weights(chunk: Chunk) -> Buffer.Model<F32>:
  Chunk{query,inside,contained,fits,image,gradient,weights,source} = chunk
  weights
def source(chunk: Chunk) -> Phases.ChunkSource(query(chunk),contained(chunk),image(chunk),gradient(chunk)):
  Chunk{query,inside,contained,fits,image,gradient,weights,source} = chunk
  source

def target(+chunk: Chunk) -> Phases.Target:
  +query = query(chunk)
  Phases.Target{Output.region(query),Sources.problem(Chunks.geometry(query),Sources.models_reader(gradient(chunk),weights(chunk)),U32.to_nat(image(chunk)))}
def native(+chunk: Chunk) -> Kernel.Chunk:
  Chunks.chunk(query(chunk))
def phase_count(+chunk: Chunk) -> Nat:
  Kernel.window_jobs(Chunks.geometry(query(chunk)))

# What the phases of a chunk read, and the packed weight and table bases
# before phase position.
def bounds(+chunk: Chunk) -> PackingBounds.Query:
  Index.packing_bounds(Index.zero(Chunks.grid(query(chunk))))
def readers(+chunk: Chunk) -> NativeJobs.Readers:
  +query = query(chunk)
  +packed = Weights.full_state(Index.zero(Chunks.grid(query)),weights(chunk))
  NativeJobs.Readers{Fill.tree(Execution.fill_chunk(Chunks.shape(query),Chunks.source(query))),Packed.weight_tree(packed),Packed.offset_tree(packed)}
def bases_at(+chunk: Chunk,+position: Nat) -> LeafWrites.Bases:
  +bounds = bounds(chunk)
  LeafWrites.Bases{readers(chunk),PackingBounds.prefix_word(Packed.reflag(True{},bounds),position),PackingBounds.prefix_word(Packed.reflag(False{},bounds),position)}

# The phase at a position: the chunk query with that phase's bases, and its
# selection.
def phase_query(+chunk: Chunk,+position: Nat) -> Chunks.Query:
  +query = query(chunk)
  +bounds = bounds(chunk)
  Output.scheduled(query,Chunks.row(query),Chunks.group(query),PackingBounds.prefix_word(Packed.reflag(True{},bounds),position),PackingBounds.prefix_word(Packed.reflag(False{},bounds),position))
def selection(+chunk: Chunk,+position: Nat,+inside: {Nat.is_lt(position,phase_count(chunk)) == True{} : Bool}) -> Chunks.Selection:
  Index.selected(phase_query(chunk,position),position,inside)
def phase_enabled(+chunk: Chunk,+position: Nat,+inside: {Nat.is_lt(position,phase_count(chunk)) == True{} : Bool},
  +enabled: {U32.is_eq(Kernel.phase_inner(Kernel.phase_of(Chunks.geometry(query(chunk)),U32.from_nat(position))),0) == False{} : Bool}) ->
  {U32.is_eq(Kernel.phase_inner(Jobs.context(Chunks.phase(selection(chunk,position,inside)))),0) == False{} : Bool}:
  %Index.selected_matches(phase_query(chunk,position),position,inside) : {U32.is_eq(Kernel.phase_inner(_),0) == False{} : Bool}
  enabled
# The packing bases of the phase at a position are the prefix sums there.
def phase_weight_base(+chunk: Chunk,+position: Nat,+inside: {Nat.is_lt(position,phase_count(chunk)) == True{} : Bool}) ->
  {Chunks.weight_base(Chunks.query(selection(chunk,position,inside))) == Runtime.weight_base(Chunks.phase(selection(chunk,position,inside))) : U32}:
  +phase = Chunks.phase(selection(chunk,position,inside))
  %Equal.sym(Nat,Index.ordinal(phase),position,Index.selected_ordinal(phase_query(chunk,position),position,inside)) :
    {PackingBounds.prefix_word(Packed.reflag(True{},bounds(chunk)),position) == PackingBounds.prefix_word(Runtime.weight_bounds(phase),_) : U32}
  {==}
def phase_offset_base(+chunk: Chunk,+position: Nat,+inside: {Nat.is_lt(position,phase_count(chunk)) == True{} : Bool}) ->
  {Chunks.offset_base(Chunks.query(selection(chunk,position,inside))) == Runtime.offset_base(Chunks.phase(selection(chunk,position,inside))) : U32}:
  +phase = Chunks.phase(selection(chunk,position,inside))
  %Equal.sym(Nat,Index.ordinal(phase),position,Index.selected_ordinal(phase_query(chunk,position),position,inside)) :
    {PackingBounds.prefix_word(Packed.reflag(False{},bounds(chunk)),position) == PackingBounds.prefix_word(Runtime.offset_bounds(phase),_) : U32}
  {==}
def phase(+chunk: Chunk,+position: Nat,+inside: {Nat.is_lt(position,phase_count(chunk)) == True{} : Bool},
  +enabled: {U32.is_eq(Kernel.phase_inner(Kernel.phase_of(Chunks.geometry(query(chunk)),U32.from_nat(position))),0) == False{} : Bool}) -> Phases.Phase:
  +selected = selection(chunk,position,inside)
  Phases.Phase{selected,chunk_inside(chunk),contained(chunk),fits(chunk),phase_enabled(chunk,position,inside,enabled),
    Runtime.Bases{phase_weight_base(chunk,position,inside),phase_offset_base(chunk,position,inside)},image(chunk),gradient(chunk),weights(chunk),
    Phases.selection_source(selected,contained(chunk),image(chunk),gradient(chunk),source(chunk))}

# The writes of a phase: none for an empty phase, its jobs' writes otherwise.
def active_phase(+chunk: Chunk,+position: Nat,+inside: {Nat.is_lt(position,phase_count(chunk)) == True{} : Bool},
  +enabled: {U32.is_eq(Kernel.phase_inner(Kernel.phase_of(Chunks.geometry(query(chunk)),U32.from_nat(position))),0) == False{} : Bool}) ->
  Every.Every(~F32,~Phases.Target,~Phases.Written,target(chunk),
    LeafWrites.selected_writes(False{},Kernel.phase_of(Chunks.geometry(query(chunk)),U32.from_nat(position)),native(chunk),bases_at(chunk,position))):
  +query = phase_query(chunk,position)
  +bases = bases_at(chunk,position)
  +weight_base = PackingBounds.prefix_word(Packed.reflag(True{},bounds(chunk)),position)
  +offset_base = PackingBounds.prefix_word(Packed.reflag(False{},bounds(chunk)),position)
  %Equal.sym(Kernel.Phase,Kernel.phase_of(Chunks.geometry(query),U32.from_nat(position)),Jobs.context(Chunks.phase(Index.selected(query,position,inside))),Index.selected_matches(query,position,inside)) :
    Every.Every(~F32,~Phases.Target,~Phases.Written,target(chunk),
      NativeJobs.jobs_writes(Execution.work_count(Execution.work(_,native(chunk),weight_base,offset_base)),0,Execution.work_plan(Execution.work(_,native(chunk),weight_base,offset_base)),readers(chunk)))
  Phases.phase_written(phase(chunk,position,inside,enabled))
def selected_phase(+chunk: Chunk,+position: Nat,+inside: {Nat.is_lt(position,phase_count(chunk)) == True{} : Bool},empty: Bool,
  +compared: {U32.is_eq(Kernel.phase_inner(Kernel.phase_of(Chunks.geometry(query(chunk)),U32.from_nat(position))),0) == empty : Bool}) ->
  Every.Every(~F32,~Phases.Target,~Phases.Written,target(chunk),
    LeafWrites.selected_writes(empty,Kernel.phase_of(Chunks.geometry(query(chunk)),U32.from_nat(position)),native(chunk),bases_at(chunk,position))):
  match empty:
    case True{}: Unit{}
    case False{}: active_phase(chunk,position,inside,compared)

# What the fold over the phases of a chunk keeps: the phases left fit in the
# chunk, the chunk is this one and the bases are those before the position.
def PhasesLeft(+chunk: Chunk,rest: Nat,position: Nat,native_chunk: Kernel.Chunk,bases: LeafWrites.Bases) -> Data:
  Shape.Both<{Nat.is_le(Nat.add(position,rest),phase_count(chunk)) == True{} : Bool},
    Shape.Both<{native_chunk == native(chunk) : Kernel.Chunk},{bases == bases_at(chunk,position) : LeafWrites.Bases}>>
def phase_step(+chunk: Chunk,+rest: Nat,+position: Nat,+native_chunk: Kernel.Chunk,+bases: LeafWrites.Bases,left: PhasesLeft(chunk,1n+rest,position,native_chunk,bases)) ->
  Every.Every(~F32,~Phases.Target,~Phases.Written,target(chunk),LeafWrites.phase_writes(U32.from_nat(position),native_chunk,bases)):
  Shape.Both{room,same} = left
  Shape.Both{same_native,same_bases} = same
  +inside = NatOrder.room_index(position,rest,phase_count(chunk),room)
  +phase = Kernel.phase_of(Chunks.geometry(query(chunk)),U32.from_nat(position))
  %Equal.sym(Kernel.Chunk,native_chunk,native(chunk),same_native) :
    Every.Every(~F32,~Phases.Target,~Phases.Written,target(chunk),LeafWrites.phase_writes(U32.from_nat(position),_,bases))
  %Equal.sym(LeafWrites.Bases,bases,bases_at(chunk,position),same_bases) :
    Every.Every(~F32,~Phases.Target,~Phases.Written,target(chunk),LeafWrites.phase_writes(U32.from_nat(position),native(chunk),_))
  selected_phase(chunk,position,inside,U32.is_eq(Kernel.phase_inner(phase),0),{==})

# The bases after a phase are those before the next position.
def next_bases(+chunk: Chunk,+position: Nat) ->
  {LeafWrites.phase_bases(U32.from_nat(position),native(chunk),bases_at(chunk,position)) == bases_at(chunk,1n+position) : LeafWrites.Bases}:
  +index = U32.from_nat(position)
  +weights = Packed.reflag(True{},bounds(chunk))
  +offsets = Packed.reflag(False{},bounds(chunk))
  +weight_base = PackingBounds.prefix_word(weights,position)
  +offset_base = PackingBounds.prefix_word(offsets,position)
  %Equal.sym(U32,PackingBounds.prefix_word(weights,1n+position),PackingBounds.update(index,weights,weight_base),PackingBounds.prefix_step(weights,position)) :
    {LeafWrites.phase_bases(index,native(chunk),bases_at(chunk,position)) == LeafWrites.Bases{readers(chunk),_,PackingBounds.prefix_word(offsets,1n+position)} : LeafWrites.Bases}
  %Equal.sym(U32,PackingBounds.prefix_word(offsets,1n+position),PackingBounds.update(index,offsets,offset_base),PackingBounds.prefix_step(offsets,position)) :
    {LeafWrites.phase_bases(index,native(chunk),bases_at(chunk,position)) == LeafWrites.Bases{readers(chunk),PackingBounds.update(index,weights,weight_base),_} : LeafWrites.Bases}
  LeafWrites.phase_totals(index,native(chunk),readers(chunk),weight_base,offset_base)
def bases_advanced(+chunk: Chunk,+position: Nat,+native_chunk: Kernel.Chunk,+bases: LeafWrites.Bases,
  +same_native: {native_chunk == native(chunk) : Kernel.Chunk},+same_bases: {bases == bases_at(chunk,position) : LeafWrites.Bases}) ->
  {LeafWrites.phase_bases(U32.from_nat(position),native_chunk,bases) == bases_at(chunk,1n+position) : LeafWrites.Bases}:
  %Equal.sym(Kernel.Chunk,native_chunk,native(chunk),same_native) : {LeafWrites.phase_bases(U32.from_nat(position),_,bases) == bases_at(chunk,1n+position) : LeafWrites.Bases}
  %Equal.sym(LeafWrites.Bases,bases,bases_at(chunk,position),same_bases) : {LeafWrites.phase_bases(U32.from_nat(position),native(chunk),_) == bases_at(chunk,1n+position) : LeafWrites.Bases}
  next_bases(chunk,position)
def phases_left_kept(+chunk: Chunk,+rest: Nat,+position: Nat,+native_chunk: Kernel.Chunk,+bases: LeafWrites.Bases,left: PhasesLeft(chunk,1n+rest,position,native_chunk,bases)) ->
  PhasesLeft(chunk,rest,1n+position,native_chunk,LeafWrites.phase_bases(U32.from_nat(position),native_chunk,bases)):
  Shape.Both{room,same} = left
  Shape.Both{+same_native,+same_bases} = same
  Shape.Both{NatOrder.advance_room(position,rest,phase_count(chunk),room),Shape.Both{same_native,bases_advanced(chunk,position,native_chunk,bases,same_native,same_bases)}}

# Every write of the phases of a chunk stores the specified value at its own
# output point.
def chunk_written(+chunk: Chunk) ->
  Every.Every(~F32,~Phases.Target,~Phases.Written,target(chunk),LeafWrites.phases_writes(phase_count(chunk),0,native(chunk),bases_at(chunk,0n))):
  Fold.every(~F32,~Kernel.Chunk,~LeafWrites.Bases,~LeafWrites.phase_writes,~LeafWrites.phase_bases,~Chunk,~Phases.Target,~target,~Phases.Written,~PhasesLeft,~phase_step,~phases_left_kept,
    chunk,phase_count(chunk),0n,native(chunk),bases_at(chunk,0n),Shape.Both{NatOrder.reflexive(phase_count(chunk)),Shape.Both{{==},{==}}})
