# A leaf's output is one write list: every job's block, jobs in order inside
# a phase, phases in order inside a chunk, chunks in order. The list is
# computed from the gradient rows and the packed weights; the output never
# feeds back into it.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../kernel_tile.bend as Tile
import ../traversal_array.bend as Traversal
import ./storage_tree_model.bend as Tree
import ./storage_buffer_model.bend as Buffer
import ./write_program.bend as Writes
import ./write_program_fold.bend as Fold
import ./kernel_gemm_jobs.bend as GemmJobs
import ./kernel_input_gradient_fill_refinement.bend as Filling
import ./input_gradient_tiled_leaf_execution.bend as Execution
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./traversal_array_refinement.bend as Arrays

# ---- Phases ----

# A phase step reads the job readers and the packed weight base; an empty
# phase writes nothing and keeps the base.
type Bases is Data:
  Bases{readers: GemmJobs.Readers,weight_base: U32}

def bases(state: Execution.Leaf) -> Bases:
  Execution.Leaf{+tiles,+weights,+output,+weight_base} = state
  Bases{Execution.readers(Execution.Jobs{tiles,weights,output}),weight_base}

def leaf_tree(state: Execution.Leaf) -> Tree.Tree<Tile.Tile8>:
  Execution.Leaf{tiles,weights,output,weight_base} = state
  Buffer.tree(~Tile.Tile8,output)

def selected_writes(empty: Bool,+phase: Kernel.Phase,+chunk: Kernel.Chunk,bases: Bases) -> List<&2,Writes.Assignment<Tile.Tile8>>:
  match empty:
    case True{}: []
    case False{}:
      Bases{readers,+weight_base} = bases
      +work = Execution.work(phase,chunk,weight_base)
      Execution.jobs_writes(Execution.work_count(work),Execution.work_plan(work),readers)

# A phase adds its packed weight tiles to the base.
def advanced(+phase: Kernel.Phase,bases: Bases) -> Bases:
  Kernel.Phase{Kernel.Geometry{+c,o,h,w,k,s,p,gh,gw},y,x,+inner} = phase
  Bases{readers,+weight_base} = bases
  Bases{readers,(weight_base + (c + 7) / 8 * inner : U32)}

def selected_bases(empty: Bool,+phase: Kernel.Phase,+chunk: Kernel.Chunk,bases: Bases) -> Bases:
  match empty:
    case True{}: bases
    case False{}: advanced(phase,bases)

def phase_writes(index: U32,chunk: Kernel.Chunk,bases: Bases) -> List<&2,Writes.Assignment<Tile.Tile8>>:
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{+geometry,layout},first,count,target},row,rows,low,group} = chunk
  +phase = Kernel.phase_of(geometry,index)
  selected_writes(U32.is_eq(Kernel.phase_inner(phase),0),phase,chunk,bases)

def phase_bases(index: U32,chunk: Kernel.Chunk,bases: Bases) -> Bases:
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{+geometry,layout},first,count,target},row,rows,low,group} = chunk
  +phase = Kernel.phase_of(geometry,index)
  selected_bases(U32.is_eq(Kernel.phase_inner(phase),0),phase,chunk,bases)

def finished_bases(+weight_base: U32,+jobs: Execution.Jobs) ->
  {bases(Execution.leaf_from_jobs(weight_base,jobs)) == Bases{Execution.readers(jobs),weight_base} : Bases}:
  Execution.Jobs{tiles,weights,output} = jobs
  {==}

def finished_tree(+weight_base: U32,+jobs: Execution.Jobs) ->
  {leaf_tree(Execution.leaf_from_jobs(weight_base,jobs)) == Buffer.tree(~Tile.Tile8,Execution.output_of(jobs)) : Tree.Tree<Tile.Tile8>}:
  Execution.Jobs{tiles,weights,output} = jobs
  {==}

def selected_bases_step(empty: Bool,+phase: Kernel.Phase,+chunk: Kernel.Chunk,+state: Execution.Leaf) ->
  {bases(Execution.selected(empty,phase,chunk,state)) == selected_bases(empty,phase,chunk,bases(state)) : Bases}:
  match empty:
    case True{}: {==}
    case False{}:
      Kernel.Phase{Kernel.Geometry{+c,o,h,w,k,s,p,gh,gw},y,x,+inner} = phase
      Execution.Leaf{+tiles,+weights,+output,+weight_base} = state
      +jobs = {Execution.Jobs{tiles,weights,output} : Execution.Jobs}
      +work = Execution.work(phase,chunk,weight_base)
      +written = Execution.phase_jobs(phase,chunk,weight_base,jobs)
      +next_weight = (weight_base + (c + 7) / 8 * inner : U32)
      Equal.trans(Bases,bases(Execution.leaf_from_jobs(next_weight,written)),Bases{Execution.readers(written),next_weight},
        Bases{Execution.readers(jobs),next_weight},finished_bases(next_weight,written),
        Equal.cong(GemmJobs.Readers,Bases,readers => Bases{readers,next_weight},Execution.readers(written),Execution.readers(jobs),
          Execution.jobs_readers(Execution.work_count(work),Execution.work_plan(work),jobs)))

def selected_output(empty: Bool,+phase: Kernel.Phase,+chunk: Kernel.Chunk,+state: Execution.Leaf) ->
  {leaf_tree(Execution.selected(empty,phase,chunk,state)) == Writes.pure(Tile.Tile8,selected_writes(empty,phase,chunk,bases(state)),leaf_tree(state)) : Tree.Tree<Tile.Tile8>}:
  match empty:
    case True{}: {==}
    case False{}:
      Kernel.Phase{Kernel.Geometry{+c,o,h,w,k,s,p,gh,gw},y,x,+inner} = phase
      Execution.Leaf{+tiles,+weights,+output,+weight_base} = state
      +jobs = {Execution.Jobs{tiles,weights,output} : Execution.Jobs}
      +work = Execution.work(phase,chunk,weight_base)
      +written = Execution.phase_jobs(phase,chunk,weight_base,jobs)
      Equal.trans(Tree.Tree<Tile.Tile8>,leaf_tree(Execution.leaf_from_jobs((weight_base + (c + 7) / 8 * inner : U32),written)),Buffer.tree(~Tile.Tile8,Execution.output_of(written)),
        Writes.pure(Tile.Tile8,Execution.jobs_writes(Execution.work_count(work),Execution.work_plan(work),Execution.readers(jobs)),Buffer.tree(~Tile.Tile8,output)),
        finished_tree((weight_base + (c + 7) / 8 * inner : U32),written),
        Execution.jobs_output(Execution.work_count(work),Execution.work_plan(work),jobs))

def phase_bases_step(+index: U32,+chunk: Kernel.Chunk,+state: Execution.Leaf) ->
  {bases(Execution.phase_step(index,chunk,state)) == phase_bases(index,chunk,bases(state)) : Bases}:
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{+geometry,layout},first,count,target},row,rows,low,group} = chunk
  +phase = Kernel.phase_of(geometry,index)
  selected_bases_step(U32.is_eq(Kernel.phase_inner(phase),0),phase,chunk,state)

def phase_output(+index: U32,+chunk: Kernel.Chunk,+state: Execution.Leaf) ->
  {leaf_tree(Execution.phase_step(index,chunk,state)) == Writes.pure(Tile.Tile8,phase_writes(index,chunk,bases(state)),leaf_tree(state)) : Tree.Tree<Tile.Tile8>}:
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{+geometry,layout},first,count,target},row,rows,low,group} = chunk
  +phase = Kernel.phase_of(geometry,index)
  selected_output(U32.is_eq(Kernel.phase_inner(phase),0),phase,chunk,state)

def phases_writes(count: Nat,+first: U32,+chunk: Kernel.Chunk,+bases: Bases) -> List<&2,Writes.Assignment<Tile.Tile8>>:
  Fold.all(~Tile.Tile8,~Kernel.Chunk,~Bases,~phase_writes,~phase_bases,count,first,chunk,bases)

def phases_tree(+count: Nat,+first: U32,+chunk: Kernel.Chunk,+state: Execution.Leaf) ->
  {leaf_tree(Traversal.indexed_fold(~Execution.Leaf,~Kernel.Chunk,~Execution.phase_step,count,first,chunk,state)) ==
    Writes.pure(Tile.Tile8,phases_writes(count,first,chunk,bases(state)),leaf_tree(state)) : Tree.Tree<Tile.Tile8>}:
  Fold.output(~Execution.Leaf,~Kernel.Chunk,~Tile.Tile8,~Bases,~Execution.phase_step,~bases,~leaf_tree,~phase_writes,~phase_bases,~phase_bases_step,~phase_output,count,first,chunk,state)

# A phase step adds the phase's packed size, which is zero for an empty phase.
def selected_advanced(empty: Bool,+phase: Kernel.Phase,+chunk: Kernel.Chunk,+bases: Bases,+compared: {U32.is_eq(Kernel.phase_inner(phase),0) == empty : Bool}) ->
  {selected_bases(empty,phase,chunk,bases) == advanced(phase,bases) : Bases}:
  match empty:
    case False{}: {==}
    case True{}:
      Kernel.Phase{Kernel.Geometry{+c,o,h,w,k,s,p,gh,gw},y,x,+inner} = phase
      Bases{readers,+weight_base} = bases
      +zero = Arithmetic.equal_of_is_eq(inner,0,compared)
      %Equal.sym(U32,inner,0,zero) : {Bases{readers,weight_base} == Bases{readers,(weight_base + (c + 7) / 8 * _ : U32)} : Bases}
      %Equal.sym(U32,(((c + 7) / 8 : U32) * 0 : U32),0,Arithmetic.multiply_zero_right(((c + 7) / 8 : U32))) : {Bases{readers,weight_base} == Bases{readers,(weight_base + _ : U32)} : Bases}
      %Equal.sym(U32,(weight_base + 0 : U32),weight_base,Arithmetic.add_zero(weight_base)) : {Bases{readers,weight_base} == Bases{readers,_} : Bases}
      {==}

def chunk_geometry(chunk: Kernel.Chunk) -> Kernel.Geometry:
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{geometry,layout},first,count,target},row,rows,low,group} = chunk
  geometry

def phase_totals(+index: U32,+chunk: Kernel.Chunk,+readers: GemmJobs.Readers,+weight_base: U32) ->
  {phase_bases(index,chunk,Bases{readers,weight_base}) == Bases{readers,Kernel.weight_total(index,chunk_geometry(chunk),weight_base)} : Bases}:
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{+geometry,layout},first,count,target},row,rows,low,group} = chunk
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +phase = Kernel.phase_of(geometry,index)
  selected_advanced(U32.is_eq(Kernel.phase_inner(phase),0),phase,chunk,Bases{readers,weight_base},{==})

# ---- Chunks ----

# A chunk step fills fresh tiles from the gradient rows and runs every phase
# from base zero; the rows and the weights stay.
type Inputs is Data:
  Inputs{source: Execution.Source,weights: Tree.Tree<Tile.Tile8>}

def inputs(state: Execution.Sweep) -> Inputs:
  Execution.Sweep{+held,+weights,output} = state
  Inputs{Execution.held_source(held),Buffer.tree(~Tile.Tile8,weights)}

def sweep_output(state: Execution.Sweep) -> Buffer.Model<Tile.Tile8>:
  Execution.Sweep{held,weights,output} = state
  output

def sweep_tree(state: Execution.Sweep) -> Tree.Tree<Tile.Tile8>:
  Buffer.tree(~Tile.Tile8,sweep_output(state))

# What the jobs of a chunk read.
def chunk_readers(+shape: Execution.ChunkShape,+source: Execution.Source,+weights: Tree.Tree<Tile.Tile8>) -> GemmJobs.Readers:
  GemmJobs.Readers{Filling.tiles(Execution.fill_chunk(shape,source)),weights,Buffer.tree(~F32,Execution.bias())}

def chunk_writes(index: U32,context: Kernel.LeafContext,inputs: Inputs) -> List<&2,Writes.Assignment<Tile.Tile8>>:
  Inputs{+source,+weights} = inputs
  +shape = Execution.chunk_shape(index,context,source)
  +chunk = Execution.shape_chunk(shape)
  phases_writes(Kernel.window_jobs(chunk_geometry(chunk)),0,chunk,Bases{chunk_readers(shape,source,weights),0})

def inputs_kept(index: U32,context: Kernel.LeafContext,inputs: Inputs) -> Inputs:
  inputs

def finished_leaf_tree(+base: U32,+first: U32,+rows: U32,+data: Buffer.Model<Tile.Tile8>,+state: Execution.Leaf) ->
  {sweep_tree(Execution.finished_leaf(base,first,rows,data,state)) == leaf_tree(state) : Tree.Tree<Tile.Tile8>}:
  Execution.Leaf{tiles,weights,output,weight_base} = state
  {==}

def run_tree(+held: Execution.Held,+chunk: Kernel.Chunk,+weights: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>,+owners: Execution.Filled) ->
  {sweep_tree(Execution.finished_chunk(held,Execution.chunk_run(chunk,weights,output,owners))) ==
    leaf_tree(Traversal.indexed_fold(~Execution.Leaf,~Kernel.Chunk,~Execution.phase_step,Kernel.window_jobs(chunk_geometry(chunk)),0,chunk,
      Execution.Leaf{Execution.filled_tiles(owners),weights,output,0})) : Tree.Tree<Tile.Tile8>}:
  Execution.Held{data,+base,+source_first,+source_rows} = held
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{+geometry,layout},first,count,target},row,rows,low,group} = chunk
  Execution.Filled{+input,+tiles} = owners
  finished_leaf_tree(base,source_first,source_rows,input,
    Traversal.indexed_fold(~Execution.Leaf,~Kernel.Chunk,~Execution.phase_step,Kernel.window_jobs(geometry),0,chunk,Execution.Leaf{tiles,weights,output,0}))

def chunk_output(+index: U32,+context: Kernel.LeafContext,+state: Execution.Sweep) ->
  {sweep_tree(Execution.chunk_step(index,context,state)) == Writes.pure(Tile.Tile8,chunk_writes(index,context,inputs(state)),sweep_tree(state)) : Tree.Tree<Tile.Tile8>}:
  Execution.Sweep{+held,+weights,+output} = state
  +source = Execution.held_source(held)
  +shape = Execution.chunk_shape(index,context,source)
  +owners = Execution.fill_held(shape,held)
  +chunk = Execution.shape_chunk(shape)
  +initial = {Execution.Leaf{Execution.filled_tiles(owners),weights,output,0} : Execution.Leaf}
  +count = Kernel.window_jobs(chunk_geometry(chunk))
  +same = Equal.cong(Tree.Tree<Tile.Tile8>,Bases,tiles => Bases{GemmJobs.Readers{tiles,Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~F32,Execution.bias())},0},
    Buffer.tree(~Tile.Tile8,Execution.filled_tiles(owners)),Filling.tiles(Execution.fill_chunk(shape,source)),Execution.held_tiles(shape,held))
  Equal.trans(Tree.Tree<Tile.Tile8>,sweep_tree(Execution.finished_chunk(held,Execution.chunk_run(chunk,weights,output,owners))),
    leaf_tree(Traversal.indexed_fold(~Execution.Leaf,~Kernel.Chunk,~Execution.phase_step,count,0,chunk,initial)),
    Writes.pure(Tile.Tile8,chunk_writes(index,context,Inputs{source,Buffer.tree(~Tile.Tile8,weights)}),Buffer.tree(~Tile.Tile8,output)),
    run_tree(held,chunk,weights,output,owners),
    Equal.trans(Tree.Tree<Tile.Tile8>,leaf_tree(Traversal.indexed_fold(~Execution.Leaf,~Kernel.Chunk,~Execution.phase_step,count,0,chunk,initial)),
      Writes.pure(Tile.Tile8,phases_writes(count,0,chunk,bases(initial)),leaf_tree(initial)),
      Writes.pure(Tile.Tile8,chunk_writes(index,context,Inputs{source,Buffer.tree(~Tile.Tile8,weights)}),Buffer.tree(~Tile.Tile8,output)),
      phases_tree(count,0,chunk,initial),
      Equal.cong(Bases,Tree.Tree<Tile.Tile8>,start => Writes.pure(Tile.Tile8,phases_writes(count,0,chunk,start),Buffer.tree(~Tile.Tile8,output)),bases(initial),
        Bases{chunk_readers(shape,source,Buffer.tree(~Tile.Tile8,weights)),0},same)))

# The inputs a finished chunk hands on: the rows it gave back and the weights
# its phases read.
def read_inputs(source: Execution.Source,readers: GemmJobs.Readers) -> Inputs:
  GemmJobs.Readers{tiles,weights,bias} = readers
  Inputs{source,weights}

def finished_inputs(+base: U32,+first: U32,+rows: U32,+data: Buffer.Model<Tile.Tile8>,+state: Execution.Leaf) ->
  {inputs(Execution.finished_leaf(base,first,rows,data,state)) ==
    read_inputs(Execution.Source{Buffer.tree(~Tile.Tile8,data),base,first,rows},Execution.leaf_readers(state)) : Inputs}:
  Execution.Leaf{tiles,weights,output,weight_base} = state
  {==}

def run_inputs(+held: Execution.Held,+chunk: Kernel.Chunk,+weights: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>,+owners: Execution.Filled) ->
  {inputs(Execution.finished_chunk(held,Execution.chunk_run(chunk,weights,output,owners))) ==
    read_inputs(Execution.Source{Buffer.tree(~Tile.Tile8,Execution.filled_source(owners)),Execution.source_base(Execution.held_source(held)),
        Execution.source_first(Execution.held_source(held)),Execution.source_rows(Execution.held_source(held))},
      Execution.leaf_readers(Traversal.indexed_fold(~Execution.Leaf,~Kernel.Chunk,~Execution.phase_step,Kernel.window_jobs(chunk_geometry(chunk)),0,chunk,
        Execution.Leaf{Execution.filled_tiles(owners),weights,output,0}))) : Inputs}:
  Execution.Held{data,+base,+source_first,+source_rows} = held
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{+geometry,layout},first,count,target},row,rows,low,group} = chunk
  Execution.Filled{+input,+tiles} = owners
  finished_inputs(base,source_first,source_rows,input,
    Traversal.indexed_fold(~Execution.Leaf,~Kernel.Chunk,~Execution.phase_step,Kernel.window_jobs(geometry),0,chunk,Execution.Leaf{tiles,weights,output,0}))

def chunk_inputs(+index: U32,+context: Kernel.LeafContext,+state: Execution.Sweep) ->
  {inputs(Execution.chunk_step(index,context,state)) == inputs_kept(index,context,inputs(state)) : Inputs}:
  Execution.Sweep{+held,+weights,+output} = state
  Execution.Held{+data,+base,+first,+rows} = held
  +model = {Execution.Held{data,base,first,rows} : Execution.Held}
  +source = Execution.held_source(model)
  +shape = Execution.chunk_shape(index,context,source)
  +owners = Execution.fill_held(shape,model)
  +chunk = Execution.shape_chunk(shape)
  +initial = {Execution.Leaf{Execution.filled_tiles(owners),weights,output,0} : Execution.Leaf}
  +after = Traversal.indexed_fold(~Execution.Leaf,~Kernel.Chunk,~Execution.phase_step,Kernel.window_jobs(chunk_geometry(chunk)),0,chunk,initial)
  %Equal.sym(Inputs,inputs(Execution.finished_chunk(model,Execution.chunk_run(chunk,weights,output,owners))),
      read_inputs(Execution.Source{Buffer.tree(~Tile.Tile8,Execution.filled_source(owners)),base,first,rows},Execution.leaf_readers(after)),
      run_inputs(model,chunk,weights,output,owners)) :
    {_ == Inputs{source,Buffer.tree(~Tile.Tile8,weights)} : Inputs}
  %Equal.sym(GemmJobs.Readers,Execution.leaf_readers(after),Execution.leaf_readers(initial),Execution.phases_readers(Kernel.window_jobs(chunk_geometry(chunk)),0,chunk,initial)) :
    {read_inputs(Execution.Source{Buffer.tree(~Tile.Tile8,Execution.filled_source(owners)),base,first,rows},_) == Inputs{source,Buffer.tree(~Tile.Tile8,weights)} : Inputs}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,Execution.filled_source(owners)),Buffer.tree(~Tile.Tile8,data),Execution.held_gradient(shape,model)) :
    {read_inputs(Execution.Source{_,base,first,rows},Execution.leaf_readers(initial)) == Inputs{source,Buffer.tree(~Tile.Tile8,weights)} : Inputs}
  {==}

def chunks_writes(count: Nat,+first: U32,+context: Kernel.LeafContext,+inputs: Inputs) -> List<&2,Writes.Assignment<Tile.Tile8>>:
  Fold.all(~Tile.Tile8,~Kernel.LeafContext,~Inputs,~chunk_writes,~inputs_kept,count,first,context,inputs)

def chunks_tree(+count: Nat,+first: U32,+context: Kernel.LeafContext,+state: Execution.Sweep) ->
  {sweep_tree(Traversal.indexed_fold(~Execution.Sweep,~Kernel.LeafContext,~Execution.chunk_step,count,first,context,state)) ==
    Writes.pure(Tile.Tile8,chunks_writes(count,first,context,inputs(state)),sweep_tree(state)) : Tree.Tree<Tile.Tile8>}:
  Fold.output(~Execution.Sweep,~Kernel.LeafContext,~Tile.Tile8,~Inputs,~Execution.chunk_step,~inputs,~sweep_tree,~chunk_writes,~inputs_kept,~chunk_inputs,~chunk_output,count,first,context,state)

def kept_fixed(+index: U32,+context: Kernel.LeafContext,+inputs: Inputs) -> {inputs_kept(index,context,inputs) == inputs : Inputs}:
  {==}

def chunks_inputs(+count: Nat,+first: U32,+context: Kernel.LeafContext,+state: Execution.Sweep) ->
  {inputs(Traversal.indexed_fold(~Execution.Sweep,~Kernel.LeafContext,~Execution.chunk_step,count,first,context,state)) == inputs(state) : Inputs}:
  Equal.trans(Inputs,inputs(Traversal.indexed_fold(~Execution.Sweep,~Kernel.LeafContext,~Execution.chunk_step,count,first,context,state)),
    Traversal.indexed_fold(~Inputs,~Kernel.LeafContext,~inputs_kept,count,first,context,inputs(state)),inputs(state),
    Arrays.indexed_fold_observes(~Execution.Sweep,~Inputs,~Kernel.LeafContext,~Execution.chunk_step,~inputs,~inputs_kept,~chunk_inputs,count,first,context,state),
    Arrays.indexed_fold_fixed(~Inputs,~Kernel.LeafContext,~inputs_kept,~kept_fixed,count,first,context,inputs(state)))

# ---- The leaf ----

def leaf_writes(+context: Kernel.LeafContext,+source: Execution.Source,+weights: Tree.Tree<Tile.Tile8>) -> List<&2,Writes.Assignment<Tile.Tile8>>:
  chunks_writes(Execution.chunk_count(context),0,context,Inputs{source,weights})

# The cells a leaf writes.
def leaf_tiles(+context: Kernel.LeafContext,+held: Execution.Held,+weights: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) -> Buffer.Model<Tile.Tile8>:
  sweep_output(Traversal.indexed_fold(~Execution.Sweep,~Kernel.LeafContext,~Execution.chunk_step,Execution.chunk_count(context),0,context,Execution.Sweep{held,weights,output}))

def leaf_tiles_writes(+context: Kernel.LeafContext,+held: Execution.Held,+weights: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) ->
  {Buffer.tree(~Tile.Tile8,leaf_tiles(context,held,weights,output)) ==
    Writes.pure(Tile.Tile8,leaf_writes(context,Execution.held_source(held),Buffer.tree(~Tile.Tile8,weights)),Buffer.tree(~Tile.Tile8,output)) : Tree.Tree<Tile.Tile8>}:
  chunks_tree(Execution.chunk_count(context),0,context,Execution.Sweep{held,weights,output})

def swept_values(+state: Execution.Sweep) ->
  {Execution.part_output(Execution.swept(state)) == sweep_output(state) : Buffer.Model<Tile.Tile8>}:
  Execution.Sweep{Execution.Held{data,base,first,rows},weights,output} = state
  {==}

# The cells of a leaf are the cells it wrote.
def leaf_values(+context: Kernel.LeafContext,+held: Execution.Held,+weights: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) ->
  {Execution.part_output(Execution.leaf_band(context,held,weights,output)) == leaf_tiles(context,held,weights,output) : Buffer.Model<Tile.Tile8>}:
  swept_values(Traversal.indexed_fold(~Execution.Sweep,~Kernel.LeafContext,~Execution.chunk_step,Execution.chunk_count(context),0,context,Execution.Sweep{held,weights,output}))

def part_source(part: Execution.Part) -> Buffer.Model<Tile.Tile8>:
  Execution.Part{gradient,weights,output} = part
  gradient

def part_weights(part: Execution.Part) -> Buffer.Model<Tile.Tile8>:
  Execution.Part{gradient,weights,output} = part
  weights

def held_content(state: Execution.Sweep) -> Buffer.Content<Tile.Tile8>:
  Execution.Sweep{held,weights,output} = state
  Buffer.content(~Tile.Tile8,Execution.held_data(held))

def content_kept(index: U32,context: Kernel.LeafContext,content: Buffer.Content<Tile.Tile8>) -> Buffer.Content<Tile.Tile8>:
  content

def content_fixed(+index: U32,+context: Kernel.LeafContext,+content: Buffer.Content<Tile.Tile8>) -> {content_kept(index,context,content) == content : Buffer.Content<Tile.Tile8>}:
  {==}

def finished_content(+base: U32,+first: U32,+rows: U32,+data: Buffer.Model<Tile.Tile8>,+state: Execution.Leaf) ->
  {held_content(Execution.finished_leaf(base,first,rows,data,state)) == Buffer.content(~Tile.Tile8,data) : Buffer.Content<Tile.Tile8>}:
  Execution.Leaf{tiles,weights,output,weight_base} = state
  {==}

def run_content(+held: Execution.Held,+chunk: Kernel.Chunk,+weights: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>,+owners: Execution.Filled) ->
  {held_content(Execution.finished_chunk(held,Execution.chunk_run(chunk,weights,output,owners))) == Buffer.content(~Tile.Tile8,Execution.filled_source(owners)) : Buffer.Content<Tile.Tile8>}:
  Execution.Held{data,+base,+source_first,+source_rows} = held
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{+geometry,layout},first,count,target},row,rows,low,group} = chunk
  Execution.Filled{+input,+tiles} = owners
  finished_content(base,source_first,source_rows,input,
    Traversal.indexed_fold(~Execution.Leaf,~Kernel.Chunk,~Execution.phase_step,Kernel.window_jobs(geometry),0,chunk,Execution.Leaf{tiles,weights,output,0}))

def chunk_content(+index: U32,+context: Kernel.LeafContext,+state: Execution.Sweep) ->
  {held_content(Execution.chunk_step(index,context,state)) == content_kept(index,context,held_content(state)) : Buffer.Content<Tile.Tile8>}:
  Execution.Sweep{+held,+weights,+output} = state
  Execution.Held{+data,+base,+first,+rows} = held
  +model = {Execution.Held{data,base,first,rows} : Execution.Held}
  +shape = Execution.chunk_shape(index,context,Execution.held_source(model))
  +owners = Execution.fill_held(shape,model)
  Equal.trans(Buffer.Content<Tile.Tile8>,held_content(Execution.finished_chunk(model,Execution.chunk_run(Execution.shape_chunk(shape),weights,output,owners))),
    Buffer.content(~Tile.Tile8,Execution.filled_source(owners)),Buffer.content(~Tile.Tile8,data),
    run_content(model,Execution.shape_chunk(shape),weights,output,owners),
    Equal.cong(Execution.PairView,Buffer.Content<Tile.Tile8>,Execution.view_source,Execution.filled_view(owners),
      Execution.PairView{Buffer.content(~Tile.Tile8,Execution.held_data(model)),
        Buffer.Content{Filling.tiles(Execution.fill_chunk(shape,Execution.held_source(model))),U32.to_nat((Execution.shape_tiles(shape) + 1 : U32))}},
      Execution.held_view(shape,model)))

def swept_content(+state: Execution.Sweep) -> {Buffer.content(~Tile.Tile8,part_source(Execution.swept(state))) == held_content(state) : Buffer.Content<Tile.Tile8>}:
  Execution.Sweep{Execution.Held{data,base,first,rows},weights,output} = state
  {==}

def leaf_content(+context: Kernel.LeafContext,+held: Execution.Held,+weights: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) ->
  {Buffer.content(~Tile.Tile8,part_source(Execution.leaf_band(context,held,weights,output))) == Buffer.content(~Tile.Tile8,Execution.held_data(held)) : Buffer.Content<Tile.Tile8>}:
  +count = Execution.chunk_count(context)
  +initial = {Execution.Sweep{held,weights,output} : Execution.Sweep}
  +final = Traversal.indexed_fold(~Execution.Sweep,~Kernel.LeafContext,~Execution.chunk_step,count,0,context,initial)
  Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,part_source(Execution.swept(final))),held_content(final),held_content(initial),
    swept_content(final),
    Equal.trans(Buffer.Content<Tile.Tile8>,held_content(final),
      Traversal.indexed_fold(~Buffer.Content<Tile.Tile8>,~Kernel.LeafContext,~content_kept,count,0,context,held_content(initial)),held_content(initial),
      Arrays.indexed_fold_observes(~Execution.Sweep,~Buffer.Content<Tile.Tile8>,~Kernel.LeafContext,~Execution.chunk_step,~held_content,~content_kept,~chunk_content,
        count,0,context,initial),
      Arrays.indexed_fold_fixed(~Buffer.Content<Tile.Tile8>,~Kernel.LeafContext,~content_kept,~content_fixed,count,0,context,held_content(initial))))

def inputs_weights(inputs: Inputs) -> Tree.Tree<Tile.Tile8>:
  Inputs{source,weights} = inputs
  weights

def swept_weights(+state: Execution.Sweep) -> {Buffer.tree(~Tile.Tile8,part_weights(Execution.swept(state))) == inputs_weights(inputs(state)) : Tree.Tree<Tile.Tile8>}:
  Execution.Sweep{Execution.Held{data,base,first,rows},weights,output} = state
  {==}

def leaf_weights(+context: Kernel.LeafContext,+held: Execution.Held,+weights: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) ->
  {Buffer.tree(~Tile.Tile8,part_weights(Execution.leaf_band(context,held,weights,output))) == Buffer.tree(~Tile.Tile8,weights) : Tree.Tree<Tile.Tile8>}:
  +initial = {Execution.Sweep{held,weights,output} : Execution.Sweep}
  +final = Traversal.indexed_fold(~Execution.Sweep,~Kernel.LeafContext,~Execution.chunk_step,Execution.chunk_count(context),0,context,initial)
  Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,part_weights(Execution.swept(final))),inputs_weights(inputs(final)),Buffer.tree(~Tile.Tile8,weights),
    swept_weights(final),
    Equal.cong(Inputs,Tree.Tree<Tile.Tile8>,inputs_weights,inputs(final),inputs(initial),chunks_inputs(Execution.chunk_count(context),0,context,initial)))
