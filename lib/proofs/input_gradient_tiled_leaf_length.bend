# A leaf keeps the length of the output it writes: every job store, phase and
# chunk returns a model of the same length as the one it was given.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../kernel_tile.bend as Tile
import ../traversal_array.bend as Traversal
import ../kernel_strided_store.bend as Stores
import ./input_gradient_tiled_reduction.bend as Reduction
import ./kernel_strided_store_writes.bend as StoreModel
import ./write_program.bend as Writes
import ./input_gradient_tiled_jobs.bend as Jobs
import ./input_gradient_tiled_refinement.bend as Filling
import ./input_gradient_tiled_leaf_execution.bend as Execution
import ./input_gradient_tiled_sweep_selected.bend as Selected
import ./storage_tree_model.bend as Tree
import ./storage_buffer_model.bend as Buffer
import ./traversal_array_refinement.bend as Arrays

# Jobs: a store rebuilds the model with the length it read.
def jobs_length(state: Jobs.Jobs) -> Nat:
  Buffer.length(~F32,Jobs.job_output(state))
def stored_length(+state: Reduction.Loop,+store: Stores.Store,+output: Buffer.Model<F32>) ->
  {Buffer.length(~F32,Jobs.job_output(Jobs.stored(state,store,output))) == Buffer.length(~F32,output) : Nat}:
  +written = Jobs.job_output(Jobs.stored(state,store,output))
  Equal.trans(Nat,Buffer.length(~F32,written),Buffer.content_length(~F32,Buffer.content(~F32,written)),Buffer.length(~F32,output),
    Equal.sym(Nat,Buffer.content_length(~F32,Buffer.content(~F32,written)),Buffer.length(~F32,written),Buffer.content_length_of(~F32,written)),
    Equal.cong(Buffer.Content<F32>,Nat,Buffer.content_length(~F32),Buffer.content(~F32,written),
      Buffer.Content{Writes.pure(F32,StoreModel.block_writes(StoreModel.Block{store,Reduction.accumulators(state)}),Buffer.tree(~F32,output)),Buffer.length(~F32,output)},
      Jobs.stored_content(state,store,output)))
def job_length(+index: U32,+plan: Kernel.Plan,+state: Jobs.Jobs) -> {jobs_length(Jobs.job(index,plan,state)) == jobs_length(state) : Nat}:
  stored_length(Jobs.final_loop(index,plan,Jobs.readers(state)),Jobs.job_store(index,plan),Jobs.job_output(state))
def kept_jobs_length(index: U32,plan: Kernel.Plan,length: Nat) -> Nat:
  length
def job_length_step(+index: U32,+plan: Kernel.Plan,+state: Jobs.Jobs) -> {jobs_length(Jobs.job(index,plan,state)) == kept_jobs_length(index,plan,jobs_length(state)) : Nat}:
  job_length(index,plan,state)
def job_length_fixed(+index: U32,+plan: Kernel.Plan,+length: Nat) -> {kept_jobs_length(index,plan,length) == length : Nat}:
  {==}
def every_job_length(+count: Nat,+first: U32,+plan: Kernel.Plan,+state: Jobs.Jobs) ->
  {jobs_length(Traversal.indexed_fold(~Jobs.Jobs,~Kernel.Plan,~Jobs.job,count,first,plan,state)) == jobs_length(state) : Nat}:
  Equal.trans(Nat,jobs_length(Traversal.indexed_fold(~Jobs.Jobs,~Kernel.Plan,~Jobs.job,count,first,plan,state)),
    Traversal.indexed_fold(~Nat,~Kernel.Plan,~kept_jobs_length,count,first,plan,jobs_length(state)),jobs_length(state),
    Arrays.indexed_fold_observes(~Jobs.Jobs,~Nat,~Kernel.Plan,~Jobs.job,~jobs_length,~kept_jobs_length,~job_length_step,count,first,plan,state),
    Arrays.indexed_fold_fixed(~Nat,~Kernel.Plan,~kept_jobs_length,~job_length_fixed,count,first,plan,jobs_length(state)))

# Phases: every phase runs its jobs on the model of the leaf.
def leaf_length(state: Execution.Leaf) -> Nat:
  Execution.Leaf{tiles,weights,offsets,output,wb,ob} = state
  Buffer.length(~F32,output)
def phase_jobs_length(+phase: Kernel.Phase,+chunk: Kernel.Chunk,+wb: U32,+ob: U32,+state: Jobs.Jobs) ->
  {jobs_length(Execution.phase_jobs(phase,chunk,wb,ob,state)) == jobs_length(state) : Nat}:
  +work = Execution.work(phase,chunk,wb,ob)
  every_job_length(Execution.work_count(work),0,Execution.work_plan(work),state)
def finished_length(+wb: U32,+ob: U32,+state: Jobs.Jobs) -> {leaf_length(Execution.leaf_from_jobs(wb,ob,state)) == jobs_length(state) : Nat}:
  Jobs.Jobs{tiles,weights,offsets,output} = state
  {==}
def as_jobs_length(+state: Execution.Leaf) -> {jobs_length(Execution.as_jobs(state)) == leaf_length(state) : Nat}:
  Execution.Leaf{tiles,weights,offsets,output,wb,ob} = state
  {==}
def selected_length(empty: Bool,+phase: Kernel.Phase,+chunk: Kernel.Chunk,+state: Execution.Leaf) ->
  {leaf_length(Execution.selected(empty,phase,chunk,state)) == leaf_length(state) : Nat}:
  match empty:
    case True{}: {==}
    case False{}:
      Kernel.Phase{Kernel.Geometry{+c,o,h,w,k,s,p,gh,gw},y,x,+inner} = phase
      Execution.Leaf{tiles,weights,offsets,output,+wb,+ob} = state
      +jobs = Execution.as_jobs(state)
      +written = Execution.phase_jobs(phase,chunk,wb,ob,jobs)
      Equal.trans(Nat,leaf_length(Execution.selected(False{},phase,chunk,state)),jobs_length(written),leaf_length(state),
        finished_length((wb + (c + 7) / 8 * inner : U32),(ob + inner : U32),written),
        Equal.trans(Nat,jobs_length(written),jobs_length(jobs),leaf_length(state),phase_jobs_length(phase,chunk,wb,ob,jobs),as_jobs_length(state)))
def phase_length(+index: U32,+chunk: Kernel.Chunk,+state: Execution.Leaf) -> {leaf_length(Execution.phase_step(index,chunk,state)) == leaf_length(state) : Nat}:
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{+geometry,layout},first,count,target,plane},row,rows,low,group} = chunk
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
  Execution.Sweep{source,weights,offsets,output} = state
  Buffer.length(~F32,output)
def result_leaf(result: Execution.ChunkResult) -> Execution.Leaf:
  Execution.ChunkResult{gradient,leaf} = result
  leaf
def finished_chunk_length(+source: Execution.Source,+result: Execution.ChunkResult) ->
  {sweep_length(Execution.finished_chunk(source,result)) == leaf_length(result_leaf(result)) : Nat}:
  Execution.Source{data,+base,+plane,+first,+rows} = source
  Execution.ChunkResult{+gradient,+leaf} = result
  Execution.Leaf{tiles,weights,offsets,output,wb,ob} = leaf
  {==}
def chunk_run_length(+chunk: Kernel.Chunk,+weights: Tree.Tree<Tile.Tile8>,+offsets: Tree.Tree<U32>,+output: Buffer.Model<F32>,+owners: Filling.Owners) ->
  {leaf_length(result_leaf(Execution.chunk_run(chunk,weights,offsets,output,owners))) == Buffer.length(~F32,output) : Nat}:
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{+geometry,layout},first,count,target,plane},row,rows,low,group} = chunk
  Filling.Owners{+input,+tiles} = owners
  phases_length(Kernel.window_jobs(geometry),0,chunk,Execution.Leaf{tiles,weights,offsets,output,0,0})
def chunk_length(+index: U32,+context: Kernel.LeafContext,+state: Execution.Sweep) ->
  {sweep_length(Execution.chunk_step(index,context,state)) == sweep_length(state) : Nat}:
  Execution.Sweep{+source,+weights,+offsets,+output} = state
  +shape = Execution.chunk_shape(index,context,source)
  +filled = Execution.fill_chunk(shape,source)
  +result = Execution.chunk_run(Execution.shape_chunk(shape),weights,offsets,output,filled)
  Equal.trans(Nat,sweep_length(Execution.finished_chunk(source,result)),leaf_length(result_leaf(result)),Buffer.length(~F32,output),
    finished_chunk_length(source,result),chunk_run_length(Execution.shape_chunk(shape),weights,offsets,output,filled))
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

# The leaf: its part holds the output of its last sweep.
def swept_length(+state: Execution.Sweep) -> {Buffer.length(~F32,Selected.part_output(Execution.swept(state))) == sweep_length(state) : Nat}:
  Execution.Sweep{Execution.Source{data,base,plane,first,rows},weights,offsets,output} = state
  {==}
def leaf_band_length(+context: Kernel.LeafContext,+source: Execution.Source,+weights: Tree.Tree<Tile.Tile8>,+offsets: Tree.Tree<U32>,+output: Buffer.Model<F32>) ->
  {Buffer.length(~F32,Selected.part_output(Execution.leaf_band(context,source,weights,offsets,output))) == Buffer.length(~F32,output) : Nat}:
  +initial = {Execution.Sweep{source,weights,offsets,output} : Execution.Sweep}
  +final = Traversal.indexed_fold(~Execution.Sweep,~Kernel.LeafContext,~Execution.chunk_step,Execution.chunk_count(context),0,context,initial)
  Equal.trans(Nat,Buffer.length(~F32,Selected.part_output(Execution.swept(final))),sweep_length(final),Buffer.length(~F32,output),
    swept_length(final),chunks_length(Execution.chunk_count(context),0,context,initial))
