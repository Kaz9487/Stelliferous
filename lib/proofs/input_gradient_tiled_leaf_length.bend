# A leaf keeps the length of the output it writes: every phase and chunk
# returns a model of the same length as the one it was given.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../kernel_tile.bend as Tile
import ../traversal_array.bend as Traversal
import ./input_gradient_tiled_leaf_execution.bend as Execution
import ./input_gradient_tiled_sweep_selected.bend as Selected
import ./storage_buffer_model.bend as Buffer
import ./traversal_array_refinement.bend as Arrays
import ./input_gradient_tiled_leaf_writes.bend as LeafWrites

# Phases: every phase runs its jobs on the model of the leaf.
def leaf_length(state: Execution.Leaf) -> Nat:
  Execution.Leaf{tiles,weights,output,weight_base} = state
  Buffer.length(~Tile.Tile8,output)
def selected_length(empty: Bool,+phase: Kernel.Phase,+chunk: Kernel.Chunk,+state: Execution.Leaf) ->
  {leaf_length(Execution.selected(empty,phase,chunk,state)) == leaf_length(state) : Nat}:
  match empty:
    case True{}: {==}
    case False{}:
      Kernel.Phase{Kernel.Geometry{+c,o,h,w,k,s,p,gh,gw},y,x,+inner} = phase
      Execution.Leaf{+tiles,+weights,+output,+weight_base} = state
      +work = Execution.work(phase,chunk,weight_base)
      Execution.jobs_length(Execution.work_count(work),Execution.work_plan(work),Execution.Jobs{tiles,weights,output})
def phase_length(+index: U32,+chunk: Kernel.Chunk,+state: Execution.Leaf) -> {leaf_length(Execution.phase_step(index,chunk,state)) == leaf_length(state) : Nat}:
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{+geometry,layout},first,count,target},row,rows,low,group} = chunk
  +phase = Kernel.phase_of(geometry,index)
  selected_length(U32.is_eq(Kernel.phase_inner(phase),0),phase,chunk,state)
def kept_leaf_length(index: U32,chunk: Kernel.Chunk,length: Nat) -> Nat:
  length
def phase_length_step(+index: U32,+chunk: Kernel.Chunk,+state: Execution.Leaf) -> {leaf_length(Execution.phase_step(index,chunk,state)) == kept_leaf_length(index,chunk,leaf_length(state)) : Nat}:
  phase_length(index,chunk,state)
def phase_length_fixed(+index: U32,+chunk: Kernel.Chunk,+length: Nat) -> {kept_leaf_length(index,chunk,length) == length : Nat}:
  {==}
def phases_length(+count: Nat,+first: U32,+chunk: Kernel.Chunk,+state: Execution.Leaf) ->
  {leaf_length(Traversal.indexed_fold(~Execution.Leaf,~Kernel.Chunk,~Execution.phase_step,count,first,chunk,state)) == leaf_length(state) : Nat}:
  Equal.trans(Nat,leaf_length(Traversal.indexed_fold(~Execution.Leaf,~Kernel.Chunk,~Execution.phase_step,count,first,chunk,state)),
    Traversal.indexed_fold(~Nat,~Kernel.Chunk,~kept_leaf_length,count,first,chunk,leaf_length(state)),leaf_length(state),
    Arrays.indexed_fold_observes(~Execution.Leaf,~Nat,~Kernel.Chunk,~Execution.phase_step,~leaf_length,~kept_leaf_length,~phase_length_step,count,first,chunk,state),
    Arrays.indexed_fold_fixed(~Nat,~Kernel.Chunk,~kept_leaf_length,~phase_length_fixed,count,first,chunk,leaf_length(state)))

# Chunks: a chunk runs its phases on the model of the sweep.
def sweep_length(state: Execution.Sweep) -> Nat:
  Execution.Sweep{held,weights,output} = state
  Buffer.length(~Tile.Tile8,output)
def result_leaf(result: Execution.ChunkResult) -> Execution.Leaf:
  Execution.ChunkResult{gradient,leaf} = result
  leaf
def finished_chunk_length(+held: Execution.Held,+result: Execution.ChunkResult) ->
  {sweep_length(Execution.finished_chunk(held,result)) == leaf_length(result_leaf(result)) : Nat}:
  Execution.Held{data,+base,+first,+rows} = held
  Execution.ChunkResult{+gradient,+leaf} = result
  Execution.Leaf{tiles,weights,output,weight_base} = leaf
  {==}
def chunk_run_length(+chunk: Kernel.Chunk,+weights: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>,+owners: Execution.Filled) ->
  {leaf_length(result_leaf(Execution.chunk_run(chunk,weights,output,owners))) == Buffer.length(~Tile.Tile8,output) : Nat}:
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{+geometry,layout},first,count,target},row,rows,low,group} = chunk
  Execution.Filled{+input,+tiles} = owners
  phases_length(Kernel.window_jobs(geometry),0,chunk,Execution.Leaf{tiles,weights,output,0})
def chunk_length(+index: U32,+context: Kernel.LeafContext,+state: Execution.Sweep) ->
  {sweep_length(Execution.chunk_step(index,context,state)) == sweep_length(state) : Nat}:
  Execution.Sweep{+held,+weights,+output} = state
  +shape = Execution.chunk_shape(index,context,Execution.held_source(held))
  +filled = Execution.fill_held(shape,held)
  +result = Execution.chunk_run(Execution.shape_chunk(shape),weights,output,filled)
  Equal.trans(Nat,sweep_length(Execution.finished_chunk(held,result)),leaf_length(result_leaf(result)),Buffer.length(~Tile.Tile8,output),
    finished_chunk_length(held,result),chunk_run_length(Execution.shape_chunk(shape),weights,output,filled))
def kept_sweep_length(index: U32,context: Kernel.LeafContext,length: Nat) -> Nat:
  length
def chunk_length_step(+index: U32,+context: Kernel.LeafContext,+state: Execution.Sweep) ->
  {sweep_length(Execution.chunk_step(index,context,state)) == kept_sweep_length(index,context,sweep_length(state)) : Nat}:
  chunk_length(index,context,state)
def chunk_length_fixed(+index: U32,+context: Kernel.LeafContext,+length: Nat) -> {kept_sweep_length(index,context,length) == length : Nat}:
  {==}
def chunks_length(+count: Nat,+first: U32,+context: Kernel.LeafContext,+state: Execution.Sweep) ->
  {sweep_length(Traversal.indexed_fold(~Execution.Sweep,~Kernel.LeafContext,~Execution.chunk_step,count,first,context,state)) == sweep_length(state) : Nat}:
  Equal.trans(Nat,sweep_length(Traversal.indexed_fold(~Execution.Sweep,~Kernel.LeafContext,~Execution.chunk_step,count,first,context,state)),
    Traversal.indexed_fold(~Nat,~Kernel.LeafContext,~kept_sweep_length,count,first,context,sweep_length(state)),sweep_length(state),
    Arrays.indexed_fold_observes(~Execution.Sweep,~Nat,~Kernel.LeafContext,~Execution.chunk_step,~sweep_length,~kept_sweep_length,~chunk_length_step,count,first,context,state),
    Arrays.indexed_fold_fixed(~Nat,~Kernel.LeafContext,~kept_sweep_length,~chunk_length_fixed,count,first,context,sweep_length(state)))

def output_length(+state: Execution.Sweep) -> {Buffer.length(~Tile.Tile8,LeafWrites.sweep_output(state)) == sweep_length(state) : Nat}:
  Execution.Sweep{held,weights,+output} = state
  {==}

# The cells a leaf writes are as many as the output it was given.
def tiles_length(+context: Kernel.LeafContext,+held: Execution.Held,+weights: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) ->
  {Buffer.length(~Tile.Tile8,LeafWrites.leaf_tiles(context,held,weights,output)) == Buffer.length(~Tile.Tile8,output) : Nat}:
  +initial = {Execution.Sweep{held,weights,output} : Execution.Sweep}
  +final = Traversal.indexed_fold(~Execution.Sweep,~Kernel.LeafContext,~Execution.chunk_step,Execution.chunk_count(context),0,context,initial)
  Equal.trans(Nat,Buffer.length(~Tile.Tile8,LeafWrites.sweep_output(final)),sweep_length(final),sweep_length(initial),
    output_length(final),chunks_length(Execution.chunk_count(context),0,context,initial))

# The leaf: its part holds as many cells as the output it was given.
def leaf_band_length(+context: Kernel.LeafContext,+held: Execution.Held,+weights: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) ->
  {Buffer.length(~Tile.Tile8,Selected.part_output(Execution.leaf_band(context,held,weights,output))) == Buffer.length(~Tile.Tile8,output) : Nat}:
  %Equal.sym(Buffer.Model<Tile.Tile8>,Execution.part_output(Execution.leaf_band(context,held,weights,output)),LeafWrites.leaf_tiles(context,held,weights,output),
      LeafWrites.leaf_values(context,held,weights,output)) :
    {Buffer.length(~Tile.Tile8,_) == Buffer.length(~Tile.Tile8,output) : Nat}
  tiles_length(context,held,weights,output)
