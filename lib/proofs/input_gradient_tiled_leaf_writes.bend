# A leaf's output is one write list: every job's strided block, jobs in order
# inside a phase, phases in order inside a chunk, chunks in order. The list is
# computed from the source rows, the packed weights and the table; the output
# never feeds back into it.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../kernel_tile.bend as Tile
import ../traversal_array.bend as Traversal
import ./storage_tree_model.bend as Tree
import ./storage_buffer_model.bend as Buffer
import ./write_program.bend as Writes
import ./write_program_fold.bend as Fold
import ./input_gradient_tiled_jobs.bend as Jobs
import ./input_gradient_tiled_refinement.bend as Filling
import ./input_gradient_tiled_leaf_execution.bend as Execution
import ./input_gradient_tiled_fill.bend as Fill
import ./bounded_u32_arithmetic.bend as Arithmetic

# Phases: a phase step reads the job readers and the packed weight and table
# bases; an empty phase writes nothing and keeps the bases.
type Bases is Data:
  Bases{readers: Jobs.Readers,weight_base: U32,offset_base: U32}
def bases(state: Execution.Leaf) -> Bases:
  Execution.Leaf{tiles,weights,offsets,output,weight_base,offset_base} = state
  Bases{Jobs.Readers{tiles,weights,offsets},weight_base,offset_base}
def leaf_tree(state: Execution.Leaf) -> Tree.Tree<F32>:
  Execution.Leaf{tiles,weights,offsets,output,weight_base,offset_base} = state
  Buffer.tree(~F32,output)

def selected_writes(empty: Bool,+phase: Kernel.Phase,+chunk: Kernel.Chunk,bases: Bases) -> List<&2,Writes.Assignment<F32>>:
  match empty:
    case True{}: []
    case False{}:
      Bases{readers,+weight_base,+offset_base} = bases
      +work = Execution.work(phase,chunk,weight_base,offset_base)
      Jobs.jobs_writes(Execution.work_count(work),0,Execution.work_plan(work),readers)
# A phase adds its packed weight tiles and table entries to the bases.
def advanced(+phase: Kernel.Phase,bases: Bases) -> Bases:
  Kernel.Phase{Kernel.Geometry{+c,o,h,w,k,s,p,gh,gw},y,x,+inner} = phase
  Bases{readers,+weight_base,+offset_base} = bases
  Bases{readers,(weight_base + (c + 7) / 8 * inner : U32),(offset_base + inner : U32)}
def selected_bases(empty: Bool,+phase: Kernel.Phase,+chunk: Kernel.Chunk,bases: Bases) -> Bases:
  match empty:
    case True{}: bases
    case False{}: advanced(phase,bases)

def phase_writes(index: U32,chunk: Kernel.Chunk,bases: Bases) -> List<&2,Writes.Assignment<F32>>:
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{+geometry,layout},first,count,target,plane},row,rows,low,group} = chunk
  +phase = Kernel.phase_of(geometry,index)
  selected_writes(U32.is_eq(Kernel.phase_inner(phase),0),phase,chunk,bases)
def phase_bases(index: U32,chunk: Kernel.Chunk,bases: Bases) -> Bases:
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{+geometry,layout},first,count,target,plane},row,rows,low,group} = chunk
  +phase = Kernel.phase_of(geometry,index)
  selected_bases(U32.is_eq(Kernel.phase_inner(phase),0),phase,chunk,bases)

def finished_bases(+weight_base: U32,+offset_base: U32,+jobs: Jobs.Jobs) ->
  {bases(Execution.leaf_from_jobs(weight_base,offset_base,jobs)) == Bases{Jobs.readers(jobs),weight_base,offset_base} : Bases}:
  Jobs.Jobs{tiles,weights,offsets,output} = jobs
  {==}
def finished_tree(+weight_base: U32,+offset_base: U32,+jobs: Jobs.Jobs) ->
  {leaf_tree(Execution.leaf_from_jobs(weight_base,offset_base,jobs)) == Jobs.output_tree(jobs) : Tree.Tree<F32>}:
  Jobs.Jobs{tiles,weights,offsets,output} = jobs
  {==}

def selected_bases_step(empty: Bool,+phase: Kernel.Phase,+chunk: Kernel.Chunk,+state: Execution.Leaf) ->
  {bases(Execution.selected(empty,phase,chunk,state)) == selected_bases(empty,phase,chunk,bases(state)) : Bases}:
  match empty:
    case True{}: {==}
    case False{}:
      Kernel.Phase{Kernel.Geometry{+c,o,h,w,k,s,p,gh,gw},y,x,+inner} = phase
      Execution.Leaf{+tiles,+weights,+offsets,output,+weight_base,+offset_base} = state
      +jobs = Execution.as_jobs(state)
      +written = Execution.phase_jobs(phase,chunk,weight_base,offset_base,jobs)
      +next_weight = (weight_base + (c + 7) / 8 * inner : U32)
      +next_offset = (offset_base + inner : U32)
      Equal.trans(Bases,bases(Execution.leaf_from_jobs(next_weight,next_offset,written)),Bases{Jobs.readers(written),next_weight,next_offset},
        Bases{Jobs.Readers{tiles,weights,offsets},next_weight,next_offset},finished_bases(next_weight,next_offset,written),
        Equal.cong(Jobs.Readers,Bases,readers => Bases{readers,next_weight,next_offset},Jobs.readers(written),Jobs.Readers{tiles,weights,offsets},
          Execution.phase_jobs_readers(phase,chunk,weight_base,offset_base,jobs)))

def selected_output(empty: Bool,+phase: Kernel.Phase,+chunk: Kernel.Chunk,+state: Execution.Leaf) ->
  {leaf_tree(Execution.selected(empty,phase,chunk,state)) == Writes.pure(F32,selected_writes(empty,phase,chunk,bases(state)),leaf_tree(state)) : Tree.Tree<F32>}:
  match empty:
    case True{}: {==}
    case False{}:
      Kernel.Phase{Kernel.Geometry{+c,o,h,w,k,s,p,gh,gw},y,x,+inner} = phase
      Execution.Leaf{tiles,weights,offsets,output,+weight_base,+offset_base} = state
      +jobs = Execution.as_jobs(state)
      +work = Execution.work(phase,chunk,weight_base,offset_base)
      +written = Execution.phase_jobs(phase,chunk,weight_base,offset_base,jobs)
      Equal.trans(Tree.Tree<F32>,leaf_tree(Execution.leaf_from_jobs((weight_base + (c + 7) / 8 * inner : U32),(offset_base + inner : U32),written)),Jobs.output_tree(written),
        Writes.pure(F32,Jobs.jobs_writes(Execution.work_count(work),0,Execution.work_plan(work),Jobs.readers(jobs)),Jobs.output_tree(jobs)),
        finished_tree((weight_base + (c + 7) / 8 * inner : U32),(offset_base + inner : U32),written),
        Jobs.jobs_tree(Execution.work_count(work),0,Execution.work_plan(work),jobs))

def phase_bases_step(+index: U32,+chunk: Kernel.Chunk,+state: Execution.Leaf) ->
  {bases(Execution.phase_step(index,chunk,state)) == phase_bases(index,chunk,bases(state)) : Bases}:
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{+geometry,layout},first,count,target,plane},row,rows,low,group} = chunk
  +phase = Kernel.phase_of(geometry,index)
  selected_bases_step(U32.is_eq(Kernel.phase_inner(phase),0),phase,chunk,state)
def phase_output(+index: U32,+chunk: Kernel.Chunk,+state: Execution.Leaf) ->
  {leaf_tree(Execution.phase_step(index,chunk,state)) == Writes.pure(F32,phase_writes(index,chunk,bases(state)),leaf_tree(state)) : Tree.Tree<F32>}:
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{+geometry,layout},first,count,target,plane},row,rows,low,group} = chunk
  +phase = Kernel.phase_of(geometry,index)
  selected_output(U32.is_eq(Kernel.phase_inner(phase),0),phase,chunk,state)

def phases_writes(count: Nat,+first: U32,+chunk: Kernel.Chunk,+bases: Bases) -> List<&2,Writes.Assignment<F32>>:
  Fold.all(~F32,~Kernel.Chunk,~Bases,~phase_writes,~phase_bases,count,first,chunk,bases)
def phases_tree(+count: Nat,+first: U32,+chunk: Kernel.Chunk,+state: Execution.Leaf) ->
  {leaf_tree(Traversal.indexed_fold(~Execution.Leaf,~Kernel.Chunk,~Execution.phase_step,count,first,chunk,state)) ==
    Writes.pure(F32,phases_writes(count,first,chunk,bases(state)),leaf_tree(state)) : Tree.Tree<F32>}:
  Fold.output(~Execution.Leaf,~Kernel.Chunk,~F32,~Bases,~Execution.phase_step,~bases,~leaf_tree,~phase_writes,~phase_bases,~phase_bases_step,~phase_output,count,first,chunk,state)

# Chunks: a chunk step fills fresh tiles from the source rows and runs every
# phase from bases zero; the source, the weights and the table stay.
type Inputs is Data:
  Inputs{source: Execution.Source,weights: Tree.Tree<Tile.Tile8>,offsets: Tree.Tree<U32>}
def inputs(state: Execution.Sweep) -> Inputs:
  Execution.Sweep{source,weights,offsets,output} = state
  Inputs{source,weights,offsets}
def sweep_tree(state: Execution.Sweep) -> Tree.Tree<F32>:
  Execution.Sweep{source,weights,offsets,output} = state
  Buffer.tree(~F32,output)
def chunk_geometry(chunk: Kernel.Chunk) -> Kernel.Geometry:
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{geometry,layout},first,count,target,plane},row,rows,low,group} = chunk
  geometry

# A phase step adds the phase's packed sizes, which are zero for an empty
# phase.
def selected_advanced(empty: Bool,+phase: Kernel.Phase,+chunk: Kernel.Chunk,+bases: Bases,+compared: {U32.is_eq(Kernel.phase_inner(phase),0) == empty : Bool}) ->
  {selected_bases(empty,phase,chunk,bases) == advanced(phase,bases) : Bases}:
  match empty:
    case False{}: {==}
    case True{}:
      Kernel.Phase{Kernel.Geometry{+c,o,h,w,k,s,p,gh,gw},y,x,+inner} = phase
      Bases{readers,+weight_base,+offset_base} = bases
      +zero = Arithmetic.equal_of_is_eq(inner,0,compared)
      %Equal.sym(U32,inner,0,zero) : {Bases{readers,weight_base,offset_base} == Bases{readers,(weight_base + (c + 7) / 8 * _ : U32),(offset_base + _ : U32)} : Bases}
      %Equal.sym(U32,(((c + 7) / 8 : U32) * 0 : U32),0,Arithmetic.multiply_zero_right(((c + 7) / 8 : U32))) : {Bases{readers,weight_base,offset_base} == Bases{readers,(weight_base + _ : U32),(offset_base + 0 : U32)} : Bases}
      %Equal.sym(U32,(weight_base + 0 : U32),weight_base,Arithmetic.add_zero(weight_base)) : {Bases{readers,weight_base,offset_base} == Bases{readers,_,(offset_base + 0 : U32)} : Bases}
      %Equal.sym(U32,(offset_base + 0 : U32),offset_base,Arithmetic.add_zero(offset_base)) : {Bases{readers,weight_base,offset_base} == Bases{readers,weight_base,_} : Bases}
      {==}
def phase_totals(+index: U32,+chunk: Kernel.Chunk,+readers: Jobs.Readers,+weight_base: U32,+offset_base: U32) ->
  {phase_bases(index,chunk,Bases{readers,weight_base,offset_base}) ==
    Bases{readers,Kernel.weight_total(index,chunk_geometry(chunk),weight_base),Kernel.inner_total(index,chunk_geometry(chunk),offset_base)} : Bases}:
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{+geometry,layout},first,count,target,plane},row,rows,low,group} = chunk
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +phase = Kernel.phase_of(geometry,index)
  selected_advanced(U32.is_eq(Kernel.phase_inner(phase),0),phase,chunk,Bases{readers,weight_base,offset_base},{==})

def chunk_writes(index: U32,context: Kernel.LeafContext,inputs: Inputs) -> List<&2,Writes.Assignment<F32>>:
  Inputs{+source,weights,offsets} = inputs
  +shape = Execution.chunk_shape(index,context,source)
  +chunk = Execution.shape_chunk(shape)
  phases_writes(Kernel.window_jobs(chunk_geometry(chunk)),0,chunk,
    Bases{Jobs.Readers{Fill.tree(Execution.fill_chunk(shape,source)),weights,offsets},0,0})
def inputs_kept(index: U32,context: Kernel.LeafContext,inputs: Inputs) -> Inputs:
  inputs

def finished_leaf_tree(+base: U32,+plane: U32,+first: U32,+rows: U32,+data: Tree.Tree<F32>,+state: Execution.Leaf) ->
  {sweep_tree(Execution.finished_leaf(base,plane,first,rows,data,state)) == leaf_tree(state) : Tree.Tree<F32>}:
  Execution.Leaf{tiles,weights,offsets,output,weight_base,offset_base} = state
  {==}
def run_tree(+source: Execution.Source,+chunk: Kernel.Chunk,+weights: Tree.Tree<Tile.Tile8>,+offsets: Tree.Tree<U32>,+output: Buffer.Model<F32>,+owners: Filling.Owners) ->
  {sweep_tree(Execution.finished_chunk(source,Execution.chunk_run(chunk,weights,offsets,output,owners))) ==
    leaf_tree(Traversal.indexed_fold(~Execution.Leaf,~Kernel.Chunk,~Execution.phase_step,Kernel.window_jobs(chunk_geometry(chunk)),0,chunk,
      Execution.Leaf{Fill.tree(owners),weights,offsets,output,0,0})) : Tree.Tree<F32>}:
  Execution.Source{data,+base,+source_plane,+source_first,+source_rows} = source
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{+geometry,layout},first,count,target,plane},row,rows,low,group} = chunk
  Filling.Owners{+input,+tiles} = owners
  finished_leaf_tree(base,source_plane,source_first,source_rows,input,
    Traversal.indexed_fold(~Execution.Leaf,~Kernel.Chunk,~Execution.phase_step,Kernel.window_jobs(geometry),0,chunk,Execution.Leaf{tiles,weights,offsets,output,0,0}))

def chunk_output(+index: U32,+context: Kernel.LeafContext,+state: Execution.Sweep) ->
  {sweep_tree(Execution.chunk_step(index,context,state)) == Writes.pure(F32,chunk_writes(index,context,inputs(state)),sweep_tree(state)) : Tree.Tree<F32>}:
  Execution.Sweep{+source,+weights,+offsets,+output} = state
  +shape = Execution.chunk_shape(index,context,source)
  +filled = Execution.fill_chunk(shape,source)
  +chunk = Execution.shape_chunk(shape)
  +initial = {Execution.Leaf{Fill.tree(filled),weights,offsets,output,0,0} : Execution.Leaf}
  Equal.trans(Tree.Tree<F32>,sweep_tree(Execution.finished_chunk(source,Execution.chunk_run(chunk,weights,offsets,output,filled))),
    leaf_tree(Traversal.indexed_fold(~Execution.Leaf,~Kernel.Chunk,~Execution.phase_step,Kernel.window_jobs(chunk_geometry(chunk)),0,chunk,initial)),
    Writes.pure(F32,phases_writes(Kernel.window_jobs(chunk_geometry(chunk)),0,chunk,bases(initial)),leaf_tree(initial)),
    run_tree(source,chunk,weights,offsets,output,filled),
    phases_tree(Kernel.window_jobs(chunk_geometry(chunk)),0,chunk,initial))

# The inputs a finished chunk hands on: the source rows it kept, and the
# weights and the table its phases read.
def read_inputs(source: Execution.Source,readers: Jobs.Readers) -> Inputs:
  Jobs.Readers{tiles,weights,offsets} = readers
  Inputs{source,weights,offsets}

def finished_inputs(+base: U32,+plane: U32,+first: U32,+rows: U32,+data: Tree.Tree<F32>,+state: Execution.Leaf) ->
  {inputs(Execution.finished_leaf(base,plane,first,rows,data,state)) == read_inputs(Execution.Source{data,base,plane,first,rows},Execution.readers(state)) : Inputs}:
  Execution.Leaf{tiles,weights,offsets,output,weight_base,offset_base} = state
  {==}

def run_inputs(+source: Execution.Source,+chunk: Kernel.Chunk,+weights: Tree.Tree<Tile.Tile8>,+offsets: Tree.Tree<U32>,+output: Buffer.Model<F32>,+owners: Filling.Owners) ->
  {inputs(Execution.finished_chunk(source,Execution.chunk_run(chunk,weights,offsets,output,owners))) ==
    read_inputs(Execution.replaced_source(source,Filling.source(owners)),
      Execution.readers(Traversal.indexed_fold(~Execution.Leaf,~Kernel.Chunk,~Execution.phase_step,Kernel.window_jobs(chunk_geometry(chunk)),0,chunk,
        Execution.Leaf{Fill.tree(owners),weights,offsets,output,0,0}))) : Inputs}:
  Execution.Source{data,+base,+source_plane,+source_first,+source_rows} = source
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{+geometry,layout},first,count,target,plane},row,rows,low,group} = chunk
  Filling.Owners{+input,+tiles} = owners
  finished_inputs(base,source_plane,source_first,source_rows,input,
    Traversal.indexed_fold(~Execution.Leaf,~Kernel.Chunk,~Execution.phase_step,Kernel.window_jobs(geometry),0,chunk,Execution.Leaf{tiles,weights,offsets,output,0,0}))

def chunk_inputs(+index: U32,+context: Kernel.LeafContext,+state: Execution.Sweep) ->
  {inputs(Execution.chunk_step(index,context,state)) == inputs_kept(index,context,inputs(state)) : Inputs}:
  Execution.Sweep{+source,+weights,+offsets,+output} = state
  +shape = Execution.chunk_shape(index,context,source)
  +filled = Execution.fill_chunk(shape,source)
  +chunk = Execution.shape_chunk(shape)
  +initial = {Execution.Leaf{Fill.tree(filled),weights,offsets,output,0,0} : Execution.Leaf}
  +after = Traversal.indexed_fold(~Execution.Leaf,~Kernel.Chunk,~Execution.phase_step,Kernel.window_jobs(chunk_geometry(chunk)),0,chunk,initial)
  %Equal.sym(Inputs,inputs(Execution.finished_chunk(source,Execution.chunk_run(chunk,weights,offsets,output,filled))),
    read_inputs(Execution.replaced_source(source,Filling.source(filled)),Execution.readers(after)),run_inputs(source,chunk,weights,offsets,output,filled)) :
    {_ == Inputs{source,weights,offsets} : Inputs}
  %Equal.sym(Jobs.Readers,Execution.readers(after),Execution.readers(initial),Execution.phases_readers(Kernel.window_jobs(chunk_geometry(chunk)),0,chunk,initial)) :
    {read_inputs(Execution.replaced_source(source,Filling.source(filled)),_) == Inputs{source,weights,offsets} : Inputs}
  %Equal.sym(Tree.Tree<F32>,Filling.source(filled),Execution.source_data(source),Execution.fill_source(shape,source)) :
    {read_inputs(Execution.replaced_source(source,_),Jobs.Readers{Fill.tree(filled),weights,offsets}) == Inputs{source,weights,offsets} : Inputs}
  %Equal.sym(Execution.Source,Execution.replaced_source(source,Execution.source_data(source)),source,Execution.source_replaced_self(source)) :
    {read_inputs(_,Jobs.Readers{Fill.tree(filled),weights,offsets}) == Inputs{source,weights,offsets} : Inputs}
  {==}

def chunks_writes(count: Nat,+first: U32,+context: Kernel.LeafContext,+inputs: Inputs) -> List<&2,Writes.Assignment<F32>>:
  Fold.all(~F32,~Kernel.LeafContext,~Inputs,~chunk_writes,~inputs_kept,count,first,context,inputs)
def chunks_tree(+count: Nat,+first: U32,+context: Kernel.LeafContext,+state: Execution.Sweep) ->
  {sweep_tree(Traversal.indexed_fold(~Execution.Sweep,~Kernel.LeafContext,~Execution.chunk_step,count,first,context,state)) ==
    Writes.pure(F32,chunks_writes(count,first,context,inputs(state)),sweep_tree(state)) : Tree.Tree<F32>}:
  Fold.output(~Execution.Sweep,~Kernel.LeafContext,~F32,~Inputs,~Execution.chunk_step,~inputs,~sweep_tree,~chunk_writes,~inputs_kept,~chunk_inputs,~chunk_output,count,first,context,state)

# The whole leaf.
def leaf_writes(+context: Kernel.LeafContext,+source: Execution.Source,+weights: Tree.Tree<Tile.Tile8>,+offsets: Tree.Tree<U32>) -> List<&2,Writes.Assignment<F32>>:
  chunks_writes(Execution.chunk_count(context),0,context,Inputs{source,weights,offsets})
def part_tree(part: Execution.Part) -> Tree.Tree<F32>:
  Execution.Part{gradient,output} = part
  Buffer.tree(~F32,output)
def swept_tree(+state: Execution.Sweep) -> {part_tree(Execution.swept(state)) == sweep_tree(state) : Tree.Tree<F32>}:
  Execution.Sweep{Execution.Source{data,base,plane,first,rows},weights,offsets,output} = state
  {==}
def leaf_tree_writes(+context: Kernel.LeafContext,+source: Execution.Source,+weights: Tree.Tree<Tile.Tile8>,+offsets: Tree.Tree<U32>,+output: Buffer.Model<F32>) ->
  {part_tree(Execution.leaf_band(context,source,weights,offsets,output)) == Writes.pure(F32,leaf_writes(context,source,weights,offsets),Buffer.tree(~F32,output)) : Tree.Tree<F32>}:
  +initial = {Execution.Sweep{source,weights,offsets,output} : Execution.Sweep}
  +final = Traversal.indexed_fold(~Execution.Sweep,~Kernel.LeafContext,~Execution.chunk_step,Execution.chunk_count(context),0,context,initial)
  Equal.trans(Tree.Tree<F32>,part_tree(Execution.swept(final)),sweep_tree(final),Writes.pure(F32,leaf_writes(context,source,weights,offsets),Buffer.tree(~F32,output)),
    swept_tree(final),chunks_tree(Execution.chunk_count(context),0,context,initial))
