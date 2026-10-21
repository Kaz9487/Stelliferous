# Actual leaf phase and chunk execution on storage-tree owners.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../kernel_tile.bend as Tile
import ../tensor.bend as Tensors
import ../traversal_array.bend as Traversal
import ../storage_allocation.bend as NativeAllocation
import ./input_gradient_tiled_jobs.bend as Jobs
import ./input_gradient_tiled_refinement.bend as Filling
import ./storage_tree_model.bend as Tree
import ./storage_buffer_model.bend as Buffer
import ./storage_allocation_refinement.bend as Allocation
import ./storage_refinement.bend as Storage
import ./traversal_array_refinement.bend as Arrays

type Leaf is Data:
  Leaf{tiles: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,offsets: Tree.Tree<U32>,output: Buffer.Model<F32>,weight_base: U32,offset_base: U32}
def represent_leaf(state: Leaf) -> Kernel.Leaf:
  Leaf{tiles,weights,offsets,output,wb,ob} = state
  Kernel.Leaf{Tree.pack(Tile.Tile8,tiles),Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),Buffer.pack(~F32,output),wb,ob}
def as_jobs(state: Leaf) -> Jobs.Jobs:
  Leaf{tiles,weights,offsets,output,wb,ob} = state
  Jobs.Jobs{tiles,weights,offsets,output}
def leaf_from_jobs(+wb: U32,+ob: U32,state: Jobs.Jobs) -> Leaf:
  Jobs.Jobs{tiles,weights,offsets,output} = state
  Leaf{tiles,weights,offsets,output,wb,ob}
def finished_matches(+wb: U32,+ob: U32,+state: Jobs.Jobs) ->
  {Kernel.leaf_finished(wb,ob,Jobs.represent(state)) == represent_leaf(leaf_from_jobs(wb,ob,state)) : Kernel.Leaf}:
  Jobs.Jobs{tiles,weights,offsets,output} = state
  {==}

type Work is Data:
  Work{plan: Kernel.Plan,count: Nat}
def work_plan(work: Work) -> Kernel.Plan:
  Work{plan,count} = work
  plan
def work_count(work: Work) -> Nat:
  Work{plan,count} = work
  count

def work(+phase: Kernel.Phase,+chunk: Kernel.Chunk,+wb: U32,+ob: U32) -> Work:
  Kernel.Phase{Kernel.Geometry{+c,o,h,+w,+k,+s,p,gh,gw},Kernel.Axis{+yf,+yc,yt,yr,+ys},Kernel.Axis{+xf,+xc,xt,xr,xs},+inner} = phase
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{geometry,+layout},+first,count,+target,+plane},+row,+rows,+low,+group} = chunk
  Kernel.Tiling{groups,+shifts,+outputs,column_low,kernel,height,width,+chunk_groups,chunk_rows} = layout
  +start = Bool.pick(U32,U32.is_lt(yf,row),((row - yf + s - 1) / s : U32),0)
  +end = U32.min(yc,Bool.pick(U32,U32.is_lt(yf,(row + rows : U32)),((row + rows - yf + s - 1) / s : U32),0))
  +units = ((c + 3) / 4 : U32)
  +phase_groups = ((xc + 7) / 8 : U32)
  +span = Bool.pick(U32,U32.is_lt(group,phase_groups),U32.min(chunk_groups,(phase_groups - group : U32)),0)
  +block = (outputs * shifts + 1 : U32)
  Work{Kernel.Plan{units,span,inner,((start + ys + k - low) * chunk_groups * block : U32),(chunk_groups * block : U32),block,
    (target + (yf + start * s - first) * w + xf + group * 8 * s : U32),(s * w : U32),s,plane,c,(xc - group * 8 : U32),(wb - ob : U32),ob},
    U32.to_nat(Bool.pick(U32,U32.is_lt(start,end),((end - start) * span * units : U32),0))}

def phase_jobs(+phase: Kernel.Phase,+chunk: Kernel.Chunk,+wb: U32,+ob: U32,state: Jobs.Jobs) -> Jobs.Jobs:
  +work = work(phase,chunk,wb,ob)
  Traversal.indexed_fold(~Jobs.Jobs,~Kernel.Plan,~Jobs.job,work_count(work),0,work_plan(work),state)
def phase_jobs_matches(+phase: Kernel.Phase,+chunk: Kernel.Chunk,+wb: U32,+ob: U32,+state: Jobs.Jobs) ->
  {Kernel.leaf_jobs(phase,chunk,wb,ob,Jobs.represent(state)) == Jobs.represent(phase_jobs(phase,chunk,wb,ob,state)) : Kernel.Jobs}:
  Kernel.Phase{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},Kernel.Axis{yf,yc,yt,yr,ys},Kernel.Axis{xf,xc,xt,xr,xs},inner} = phase
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{geometry,layout},first,count,target,plane},row,rows,low,group} = chunk
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,chunk_groups,chunk_rows} = layout
  +work = work(phase,chunk,wb,ob)
  Jobs.jobs_matches(work_count(work),0,work_plan(work),state)

def selected(empty: Bool,+phase: Kernel.Phase,+chunk: Kernel.Chunk,state: Leaf) -> Leaf:
  match empty:
    case True{}: state
    case False{}:
      Kernel.Phase{Kernel.Geometry{+c,o,h,w,k,s,p,gh,gw},y,x,+inner} = phase
      Leaf{tiles,weights,offsets,output,+wb,+ob} = state
      leaf_from_jobs((wb + (c + 7) / 8 * inner : U32),(ob + inner : U32),phase_jobs(phase,chunk,wb,ob,Jobs.Jobs{tiles,weights,offsets,output}))
def selected_matches(empty: Bool,+phase: Kernel.Phase,+chunk: Kernel.Chunk,+state: Leaf) ->
  {Kernel.leaf_selected(empty,phase,chunk,represent_leaf(state)) == represent_leaf(selected(empty,phase,chunk,state)) : Kernel.Leaf}:
  match empty:
    case True{}: {==}
    case False{}:
      Kernel.Phase{Kernel.Geometry{+c,o,h,w,k,s,p,gh,gw},y,x,+inner} = phase
      Leaf{tiles,weights,offsets,output,+wb,+ob} = state
      +jobs = as_jobs(state)
      +written = phase_jobs(phase,chunk,wb,ob,jobs)
      %Equal.sym(Kernel.Jobs,Kernel.leaf_jobs(phase,chunk,wb,ob,Jobs.represent(jobs)),Jobs.represent(written),phase_jobs_matches(phase,chunk,wb,ob,jobs)) :
        {Kernel.leaf_finished((wb + (c + 7) / 8 * inner : U32),(ob + inner : U32),_) == represent_leaf(selected(False{},phase,chunk,state)) : Kernel.Leaf}
      finished_matches((wb + (c + 7) / 8 * inner : U32),(ob + inner : U32),written)

def phase_step(index: U32,chunk: Kernel.Chunk,state: Leaf) -> Leaf:
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{+geometry,layout},first,count,target,plane},row,rows,low,group} = chunk
  +phase = Kernel.phase_of(geometry,index)
  selected(U32.is_eq(Kernel.phase_inner(phase),0),phase,chunk,state)
def phase_matches(+index: U32,+chunk: Kernel.Chunk,+state: Leaf) -> {Kernel.leaf_phase(index,chunk,represent_leaf(state)) == represent_leaf(phase_step(index,chunk,state)) : Kernel.Leaf}:
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{+geometry,layout},first,count,target,plane},row,rows,low,group} = chunk
  +phase = Kernel.phase_of(geometry,index)
  selected_matches(U32.is_eq(Kernel.phase_inner(phase),0),phase,chunk,state)
def phases_matches(+count: Nat,+first: U32,+chunk: Kernel.Chunk,+state: Leaf) ->
  {Traversal.indexed_fold(~Kernel.Leaf,~Kernel.Chunk,~Kernel.leaf_phase,count,first,chunk,represent_leaf(state)) == represent_leaf(Traversal.indexed_fold(~Leaf,~Kernel.Chunk,~phase_step,count,first,chunk,state)) : Kernel.Leaf}:
  Arrays.indexed_fold_refines_model(~Kernel.Leaf,~Leaf,~Kernel.Chunk,~Kernel.leaf_phase,~phase_step,~represent_leaf,~phase_matches,count,first,chunk,state)

def readers(state: Leaf) -> Jobs.Readers:
  Leaf{tiles,weights,offsets,output,wb,ob} = state
  Jobs.Readers{tiles,weights,offsets}
def phase_jobs_readers(+phase: Kernel.Phase,+chunk: Kernel.Chunk,+wb: U32,+ob: U32,+state: Jobs.Jobs) ->
  {Jobs.readers(phase_jobs(phase,chunk,wb,ob,state)) == Jobs.readers(state) : Jobs.Readers}:
  +work = work(phase,chunk,wb,ob)
  Jobs.jobs_keeps_readers(work_count(work),0,work_plan(work),state)
def finished_readers(+wb: U32,+ob: U32,+state: Jobs.Jobs) -> {readers(leaf_from_jobs(wb,ob,state)) == Jobs.readers(state) : Jobs.Readers}:
  Jobs.Jobs{tiles,weights,offsets,output} = state
  {==}
def as_jobs_readers(+state: Leaf) -> {Jobs.readers(as_jobs(state)) == readers(state) : Jobs.Readers}:
  Leaf{tiles,weights,offsets,output,wb,ob} = state
  {==}
def selected_readers(empty: Bool,+phase: Kernel.Phase,+chunk: Kernel.Chunk,+state: Leaf) -> {readers(selected(empty,phase,chunk,state)) == readers(state) : Jobs.Readers}:
  match empty:
    case True{}: {==}
    case False{}:
      Kernel.Phase{Kernel.Geometry{+c,o,h,w,k,s,p,gh,gw},y,x,+inner} = phase
      Leaf{tiles,weights,offsets,output,+wb,+ob} = state
      +jobs = as_jobs(state)
      +written = phase_jobs(phase,chunk,wb,ob,jobs)
      Equal.trans(Jobs.Readers,readers(selected(False{},phase,chunk,state)),Jobs.readers(written),readers(state),finished_readers((wb + (c + 7) / 8 * inner : U32),(ob + inner : U32),written),
        Equal.trans(Jobs.Readers,Jobs.readers(written),Jobs.readers(jobs),readers(state),phase_jobs_readers(phase,chunk,wb,ob,jobs),as_jobs_readers(state)))
def phase_readers(+index: U32,+chunk: Kernel.Chunk,+state: Leaf) -> {readers(phase_step(index,chunk,state)) == readers(state) : Jobs.Readers}:
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{+geometry,layout},first,count,target,plane},row,rows,low,group} = chunk
  +phase = Kernel.phase_of(geometry,index)
  selected_readers(U32.is_eq(Kernel.phase_inner(phase),0),phase,chunk,state)
def kept_readers(index: U32,chunk: Kernel.Chunk,readers: Jobs.Readers) -> Jobs.Readers:
  readers
def reader_step(+index: U32,+chunk: Kernel.Chunk,+state: Leaf) -> {readers(phase_step(index,chunk,state)) == kept_readers(index,chunk,readers(state)) : Jobs.Readers}:
  phase_readers(index,chunk,state)
def reader_fixed(+index: U32,+chunk: Kernel.Chunk,+readers: Jobs.Readers) -> {kept_readers(index,chunk,readers) == readers : Jobs.Readers}:
  {==}
def phases_readers(+count: Nat,+first: U32,+chunk: Kernel.Chunk,+state: Leaf) ->
  {readers(Traversal.indexed_fold(~Leaf,~Kernel.Chunk,~phase_step,count,first,chunk,state)) == readers(state) : Jobs.Readers}:
  Equal.trans(Jobs.Readers,readers(Traversal.indexed_fold(~Leaf,~Kernel.Chunk,~phase_step,count,first,chunk,state)),Traversal.indexed_fold(~Jobs.Readers,~Kernel.Chunk,~kept_readers,count,first,chunk,readers(state)),readers(state),
    Arrays.indexed_fold_observes(~Leaf,~Jobs.Readers,~Kernel.Chunk,~phase_step,~readers,~kept_readers,~reader_step,count,first,chunk,state),Arrays.indexed_fold_fixed(~Jobs.Readers,~Kernel.Chunk,~kept_readers,~reader_fixed,count,first,chunk,readers(state)))

type Source is Data:
  Source{data: Tree.Tree<F32>,base: U32,plane: U32,first: U32,rows: U32}
def represent_source(source: Source) -> Kernel.Source:
  Source{data,base,plane,first,rows} = source
  Kernel.Source{Tree.pack(F32,data),base,plane,first,rows}
def source_data(source: Source) -> Tree.Tree<F32>:
  Source{data,base,plane,first,rows} = source
  data

def source_base(source: Source) -> U32:
  Source{data,base,plane,first,rows} = source
  base
def source_plane(source: Source) -> U32:
  Source{data,base,plane,first,rows} = source
  plane
def source_first(source: Source) -> U32:
  Source{data,base,plane,first,rows} = source
  first
def source_rows(source: Source) -> U32:
  Source{data,base,plane,first,rows} = source
  rows

type Sweep is Data:
  Sweep{source: Source,weights: Tree.Tree<Tile.Tile8>,offsets: Tree.Tree<U32>,output: Buffer.Model<F32>}
def represent_sweep(state: Sweep) -> Kernel.Sweep:
  Sweep{source,weights,offsets,output} = state
  Kernel.Sweep{represent_source(source),Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),Buffer.pack(~F32,output)}
def sweep_source(state: Sweep) -> Source:
  Sweep{source,weights,offsets,output} = state
  source

type ChunkShape is Data:
  ChunkShape{chunk: Kernel.Chunk,filling: Kernel.Filling,fill_count: Nat,tile_count: U32}
def shape_chunk(shape: ChunkShape) -> Kernel.Chunk:
  ChunkShape{chunk,filling,count,tiles} = shape
  chunk
def shape_filling(shape: ChunkShape) -> Kernel.Filling:
  ChunkShape{chunk,filling,count,tiles} = shape
  filling
def shape_count(shape: ChunkShape) -> Nat:
  ChunkShape{chunk,filling,count,tiles} = shape
  count
def shape_tiles(shape: ChunkShape) -> U32:
  ChunkShape{chunk,filling,count,tiles} = shape
  tiles

def chunk_shape(index: U32,context: Kernel.LeafContext,source: Source) -> ChunkShape:
  Kernel.LeafContext{Kernel.Task{+geometry,+layout},+first,+count,target,plane} = context
  Kernel.Tiling{+groups,+shifts,+outputs,column_low,kernel,height,width,+chunk_groups,+chunk_rows} = layout
  Source{data,+base,+source_plane,+source_first,+source_rows} = source
  +index = index
  +group_chunks = ((groups + chunk_groups - 1) / chunk_groups : U32)
  +row = (first + index / group_chunks * chunk_rows : U32)
  +rows = U32.min(chunk_rows,(first + count - row : U32))
  +group = (index % group_chunks * chunk_groups : U32)
  +span = U32.min(chunk_groups,(groups - group : U32))
  +low = Kernel.band_low(geometry,row,(row + rows : U32))
  +tile_rows = Kernel.band_rows(geometry,row,rows)
  ChunkShape{Kernel.Chunk{context,row,rows,low,group},Kernel.Filling{layout,low,base,source_plane,source_first,source_rows,group,span},U32.to_nat((tile_rows * span : U32)),(tile_rows * chunk_groups * (outputs * shifts + 1) : U32)}
def fresh_tiles(+shape: ChunkShape) -> Tree.Tree<Tile.Tile8>:
  Tree.fresh(Tile.Tile8,Tensors.depth_for((shape_tiles(shape) + 1 : U32)),Tile.filled(0.0))
def fill_chunk(+shape: ChunkShape,+source: Source) -> Filling.Owners:
  Traversal.indexed_fold(~Filling.Owners,~Kernel.Filling,~Filling.fill_group,shape_count(shape),0,shape_filling(shape),Filling.Owners{source_data(source),fresh_tiles(shape)})

type ChunkResult is Data:
  ChunkResult{gradient: Tree.Tree<F32>,leaf: Leaf}
def represent_result(result: ChunkResult) -> Array<F32> & Kernel.Leaf:
  ChunkResult{gradient,leaf} = result
  (Tree.pack(F32,gradient),represent_leaf(leaf))
def chunk_run(+chunk: Kernel.Chunk,weights: Tree.Tree<Tile.Tile8>,offsets: Tree.Tree<U32>,output: Buffer.Model<F32>,owners: Filling.Owners) -> ChunkResult:
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{+geometry,layout},first,count,target,plane},row,rows,low,group} = chunk
  Filling.Owners{input,tiles} = owners
  ChunkResult{input,Traversal.indexed_fold(~Leaf,~Kernel.Chunk,~phase_step,Kernel.window_jobs(geometry),0,chunk,Leaf{tiles,weights,offsets,output,0,0})}
def chunk_run_matches(+chunk: Kernel.Chunk,+weights: Tree.Tree<Tile.Tile8>,+offsets: Tree.Tree<U32>,+output: Buffer.Model<F32>,+owners: Filling.Owners) ->
  {Kernel.chunk_run(chunk,Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),Buffer.pack(~F32,output),Filling.pack_owners(owners)) == represent_result(chunk_run(chunk,weights,offsets,output,owners)) : Array<F32> & Kernel.Leaf}:
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{+geometry,layout},first,count,target,plane},row,rows,low,group} = chunk
  Filling.Owners{+input,+tiles} = owners
  +initial = {Leaf{tiles,weights,offsets,output,0,0} : Leaf}
  Equal.cong(Kernel.Leaf,Array<F32> & Kernel.Leaf,leaf => (Tree.pack(F32,input),leaf),Traversal.indexed_fold(~Kernel.Leaf,~Kernel.Chunk,~Kernel.leaf_phase,Kernel.window_jobs(geometry),0,chunk,represent_leaf(initial)),
    represent_leaf(Traversal.indexed_fold(~Leaf,~Kernel.Chunk,~phase_step,Kernel.window_jobs(geometry),0,chunk,initial)),phases_matches(Kernel.window_jobs(geometry),0,chunk,initial))

def finished_leaf(+base: U32,+plane: U32,+first: U32,+rows: U32,data: Tree.Tree<F32>,state: Leaf) -> Sweep:
  Leaf{tiles,weights,offsets,output,wb,ob} = state
  Sweep{Source{data,base,plane,first,rows},weights,offsets,output}
def finished_chunk(+source: Source,result: ChunkResult) -> Sweep:
  Source{data,+base,+plane,+first,+rows} = source
  ChunkResult{gradient,leaf} = result
  finished_leaf(base,plane,first,rows,gradient,leaf)
def finished_leaf_matches(+base: U32,+plane: U32,+first: U32,+rows: U32,+data: Tree.Tree<F32>,+state: Leaf) ->
  {Kernel.chunk_leaf(base,plane,first,rows,Tree.pack(F32,data),represent_leaf(state)) == represent_sweep(finished_leaf(base,plane,first,rows,data,state)) : Kernel.Sweep}:
  Leaf{tiles,weights,offsets,output,wb,ob} = state
  {==}
def finished_chunk_matches(+source: Source,+result: ChunkResult) ->
  {Kernel.chunk_finished(source_base(source),source_plane(source),source_first(source),source_rows(source),represent_result(result)) == represent_sweep(finished_chunk(source,result)) : Kernel.Sweep}:
  Source{data,+base,+plane,+first,+rows} = source
  ChunkResult{+gradient,+leaf} = result
  finished_leaf_matches(base,plane,first,rows,gradient,leaf)

def chunk_step(index: U32,context: Kernel.LeafContext,state: Sweep) -> Sweep:
  Sweep{+source,weights,offsets,output} = state
  +shape = chunk_shape(index,context,source)
  finished_chunk(source,chunk_run(shape_chunk(shape),weights,offsets,output,fill_chunk(shape,source)))

def chunk_matches(+index: U32,+context: Kernel.LeafContext,+state: Sweep) ->
  {Kernel.chunk_step(index,context,represent_sweep(state)) == represent_sweep(chunk_step(index,context,state)) : Kernel.Sweep}:
  Kernel.LeafContext{Kernel.Task{geometry,layout},first,count,target,plane} = context
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,chunk_groups,chunk_rows} = layout
  Sweep{+source,+weights,+offsets,+output} = state
  Source{+data,+base,+source_plane,+source_first,+source_rows} = source
  +shape = chunk_shape(index,context,source)
  +depth = Tensors.depth_for((shape_tiles(shape) + 1 : U32))
  +initial = {Filling.Owners{data,fresh_tiles(shape)} : Filling.Owners}
  +filled = fill_chunk(shape,source)
  +result = chunk_run(shape_chunk(shape),weights,offsets,output,filled)
  %Equal.sym(Array<Tile.Tile8>,NativeAllocation.array(~Tile.Tile8,depth,Tile.filled(0.0)),Tree.pack(Tile.Tile8,fresh_tiles(shape)),Allocation.fresh(~Tile.Tile8,depth,Tile.filled(0.0))) :
    {Kernel.chunk_finished(base,source_plane,source_first,source_rows,Kernel.chunk_run(shape_chunk(shape),Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),Buffer.pack(~F32,output),
      Traversal.indexed_fold(~(Array<F32> & Array<Tile.Tile8>),~Kernel.Filling,~Kernel.fill_group,shape_count(shape),0,shape_filling(shape),(Tree.pack(F32,data),_)))) == represent_sweep(chunk_step(index,context,state)) : Kernel.Sweep}
  %Equal.sym(Array<F32> & Array<Tile.Tile8>,Traversal.indexed_fold(~(Array<F32> & Array<Tile.Tile8>),~Kernel.Filling,~Kernel.fill_group,shape_count(shape),0,shape_filling(shape),Filling.pack_owners(initial)),Filling.pack_owners(filled),Filling.groups_match(shape_count(shape),0,shape_filling(shape),initial)) :
    {Kernel.chunk_finished(base,source_plane,source_first,source_rows,Kernel.chunk_run(shape_chunk(shape),Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),Buffer.pack(~F32,output),_)) == represent_sweep(chunk_step(index,context,state)) : Kernel.Sweep}
  %Equal.sym(Array<F32> & Kernel.Leaf,Kernel.chunk_run(shape_chunk(shape),Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),Buffer.pack(~F32,output),Filling.pack_owners(filled)),represent_result(result),chunk_run_matches(shape_chunk(shape),weights,offsets,output,filled)) :
    {Kernel.chunk_finished(base,source_plane,source_first,source_rows,_) == represent_sweep(chunk_step(index,context,state)) : Kernel.Sweep}
  finished_chunk_matches(source,result)

def chunks_matches(+count: Nat,+first: U32,+context: Kernel.LeafContext,+state: Sweep) ->
  {Traversal.indexed_fold(~Kernel.Sweep,~Kernel.LeafContext,~Kernel.chunk_step,count,first,context,represent_sweep(state)) ==
    represent_sweep(Traversal.indexed_fold(~Sweep,~Kernel.LeafContext,~chunk_step,count,first,context,state)) : Kernel.Sweep}:
  Arrays.indexed_fold_refines_model(~Kernel.Sweep,~Sweep,~Kernel.LeafContext,~Kernel.chunk_step,~chunk_step,~represent_sweep,~chunk_matches,count,first,context,state)

def result_gradient(result: ChunkResult) -> Tree.Tree<F32>:
  ChunkResult{gradient,leaf} = result
  gradient
def chunk_gradient(+chunk: Kernel.Chunk,+weights: Tree.Tree<Tile.Tile8>,+offsets: Tree.Tree<U32>,+output: Buffer.Model<F32>,+owners: Filling.Owners) ->
  {result_gradient(chunk_run(chunk,weights,offsets,output,owners)) == Filling.source(owners) : Tree.Tree<F32>}:
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{geometry,layout},first,count,target,plane},row,rows,low,group} = chunk
  Filling.Owners{input,tiles} = owners
  {==}
def replaced_source(source: Source,data: Tree.Tree<F32>) -> Source:
  Source{old,base,plane,first,rows} = source
  Source{data,base,plane,first,rows}
def source_replaced_self(+source: Source) -> {replaced_source(source,source_data(source)) == source : Source}:
  Source{data,base,plane,first,rows} = source
  {==}
def finished_source(+source: Source,+result: ChunkResult) -> {sweep_source(finished_chunk(source,result)) == replaced_source(source,result_gradient(result)) : Source}:
  Source{data,base,plane,first,rows} = source
  ChunkResult{gradient,Leaf{tiles,weights,offsets,output,wb,ob}} = result
  {==}
def fill_source(+shape: ChunkShape,+source: Source) -> {Filling.source(fill_chunk(shape,source)) == source_data(source) : Tree.Tree<F32>}:
  Filling.groups_keep_source(shape_count(shape),0,shape_filling(shape),Filling.Owners{source_data(source),fresh_tiles(shape)})
def chunk_source(+index: U32,+context: Kernel.LeafContext,+state: Sweep) -> {sweep_source(chunk_step(index,context,state)) == sweep_source(state) : Source}:
  Sweep{+source,+weights,+offsets,+output} = state
  +shape = chunk_shape(index,context,source)
  +filled = fill_chunk(shape,source)
  +result = chunk_run(shape_chunk(shape),weights,offsets,output,filled)
  Equal.trans(Source,sweep_source(chunk_step(index,context,state)),replaced_source(source,result_gradient(result)),source,finished_source(source,result),
    Equal.trans(Source,replaced_source(source,result_gradient(result)),replaced_source(source,source_data(source)),source,
      Equal.cong(Tree.Tree<F32>,Source,data => replaced_source(source,data),result_gradient(result),source_data(source),
        Equal.trans(Tree.Tree<F32>,result_gradient(result),Filling.source(filled),source_data(source),chunk_gradient(shape_chunk(shape),weights,offsets,output,filled),fill_source(shape,source))),source_replaced_self(source)))

def kept_source(index: U32,context: Kernel.LeafContext,source: Source) -> Source:
  source
def source_step(+index: U32,+context: Kernel.LeafContext,+state: Sweep) -> {sweep_source(chunk_step(index,context,state)) == kept_source(index,context,sweep_source(state)) : Source}:
  chunk_source(index,context,state)
def source_fixed(+index: U32,+context: Kernel.LeafContext,+source: Source) -> {kept_source(index,context,source) == source : Source}:
  {==}
def chunks_source(+count: Nat,+first: U32,+context: Kernel.LeafContext,+state: Sweep) ->
  {sweep_source(Traversal.indexed_fold(~Sweep,~Kernel.LeafContext,~chunk_step,count,first,context,state)) == sweep_source(state) : Source}:
  Equal.trans(Source,sweep_source(Traversal.indexed_fold(~Sweep,~Kernel.LeafContext,~chunk_step,count,first,context,state)),Traversal.indexed_fold(~Source,~Kernel.LeafContext,~kept_source,count,first,context,sweep_source(state)),sweep_source(state),
    Arrays.indexed_fold_observes(~Sweep,~Source,~Kernel.LeafContext,~chunk_step,~sweep_source,~kept_source,~source_step,count,first,context,state),Arrays.indexed_fold_fixed(~Source,~Kernel.LeafContext,~kept_source,~source_fixed,count,first,context,sweep_source(state)))

type Part is Data:
  Part{gradient: Tree.Tree<F32>,output: Buffer.Model<F32>}
def represent_part(part: Part) -> Kernel.Part:
  Part{gradient,output} = part
  Kernel.Part{Tree.pack(F32,gradient),Buffer.pack(~F32,output)}
def swept(state: Sweep) -> Part:
  Sweep{Source{data,base,plane,first,rows},weights,offsets,output} = state
  Part{data,output}
def swept_matches(+state: Sweep) -> {Kernel.swept(represent_sweep(state)) == represent_part(swept(state)) : Kernel.Part}:
  Sweep{Source{data,base,plane,first,rows},weights,offsets,output} = state
  {==}
def chunk_count(context: Kernel.LeafContext) -> Nat:
  Kernel.LeafContext{Kernel.Task{geometry,layout},first,+count,target,plane} = context
  Kernel.Tiling{+groups,shifts,outputs,column_low,kernel,height,width,+chunk_groups,+chunk_rows} = layout
  U32.to_nat(((count + chunk_rows - 1) / chunk_rows * ((groups + chunk_groups - 1) / chunk_groups) : U32))
def leaf_band(+context: Kernel.LeafContext,source: Source,weights: Tree.Tree<Tile.Tile8>,offsets: Tree.Tree<U32>,output: Buffer.Model<F32>) -> Part:
  swept(Traversal.indexed_fold(~Sweep,~Kernel.LeafContext,~chunk_step,chunk_count(context),0,context,Sweep{source,weights,offsets,output}))
def leaf_matches(+context: Kernel.LeafContext,+source: Source,+weights: Tree.Tree<Tile.Tile8>,+offsets: Tree.Tree<U32>,+output: Buffer.Model<F32>) ->
  {Kernel.leaf_band(context,represent_source(source),Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),Buffer.pack(~F32,output)) == represent_part(leaf_band(context,source,weights,offsets,output)) : Kernel.Part}:
  Kernel.LeafContext{Kernel.Task{geometry,layout},first,count,target,plane} = context
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,chunk_groups,chunk_rows} = layout
  +initial = {Sweep{source,weights,offsets,output} : Sweep}
  +final = Traversal.indexed_fold(~Sweep,~Kernel.LeafContext,~chunk_step,chunk_count(context),0,context,initial)
  %Equal.sym(Kernel.Sweep,Traversal.indexed_fold(~Kernel.Sweep,~Kernel.LeafContext,~Kernel.chunk_step,chunk_count(context),0,context,represent_sweep(initial)),represent_sweep(final),chunks_matches(chunk_count(context),0,context,initial)) :
    {Kernel.swept(_) == represent_part(leaf_band(context,source,weights,offsets,output)) : Kernel.Part}
  swept_matches(final)

def gradient(part: Part) -> Tree.Tree<F32>:
  Part{gradient,output} = part
  gradient
def native_gradient(part: Kernel.Part) -> Array<F32>:
  Kernel.Part{gradient,output} = part
  gradient
def swept_gradient(+state: Sweep) -> {gradient(swept(state)) == source_data(sweep_source(state)) : Tree.Tree<F32>}:
  Sweep{Source{data,base,plane,first,rows},weights,offsets,output} = state
  {==}
def leaf_gradient(+context: Kernel.LeafContext,+source: Source,+weights: Tree.Tree<Tile.Tile8>,+offsets: Tree.Tree<U32>,+output: Buffer.Model<F32>) ->
  {gradient(leaf_band(context,source,weights,offsets,output)) == source_data(source) : Tree.Tree<F32>}:
  +initial = {Sweep{source,weights,offsets,output} : Sweep}
  +final = Traversal.indexed_fold(~Sweep,~Kernel.LeafContext,~chunk_step,chunk_count(context),0,context,initial)
  Equal.trans(Tree.Tree<F32>,gradient(leaf_band(context,source,weights,offsets,output)),source_data(sweep_source(final)),source_data(source),swept_gradient(final),
    Equal.cong(Source,Tree.Tree<F32>,source_data,sweep_source(final),source,chunks_source(chunk_count(context),0,context,initial)))

def represented_gradient(+part: Part) -> {native_gradient(represent_part(part)) == Tree.pack(F32,gradient(part)) : Array<F32>}:
  Part{gradient,output} = part
  {==}

# Returned gradient ownership is the original packed owner of this source.
def native_leaf_gradient(+context: Kernel.LeafContext,+source: Source,+weights: Tree.Tree<Tile.Tile8>,+offsets: Tree.Tree<U32>,+output: Buffer.Model<F32>) ->
  {native_gradient(Kernel.leaf_band(context,represent_source(source),Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),Buffer.pack(~F32,output))) == Tree.pack(F32,source_data(source)) : Array<F32>}:
  +part = leaf_band(context,source,weights,offsets,output)
  Equal.trans(Array<F32>,native_gradient(Kernel.leaf_band(context,represent_source(source),Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),Buffer.pack(~F32,output))),Tree.pack(F32,gradient(part)),Tree.pack(F32,source_data(source)),
    Equal.trans(Array<F32>,native_gradient(Kernel.leaf_band(context,represent_source(source),Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),Buffer.pack(~F32,output))),native_gradient(represent_part(part)),Tree.pack(F32,gradient(part)),
      Equal.cong(Kernel.Part,Array<F32>,native_gradient,Kernel.leaf_band(context,represent_source(source),Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),Buffer.pack(~F32,output)),represent_part(part),leaf_matches(context,source,weights,offsets,output)),represented_gradient(part)),
    Equal.cong(Tree.Tree<F32>,Array<F32>,Tree.pack(F32),gradient(part),source_data(source),leaf_gradient(context,source,weights,offsets,output)))

# Lift the ownership conclusion from packed trees to an arbitrary source array.
type GradientQuery is Data:
  GradientQuery{context: Kernel.LeafContext,base: U32,plane: U32,first: U32,rows: U32,weights: Tree.Tree<Tile.Tile8>,offsets: Tree.Tree<U32>,output: Buffer.Model<F32>}
def gradient_property(query: GradientQuery,-storage: Array<F32>,-tree: Tree.Tree<F32>) -> Type:
  GradientQuery{+context,+base,+plane,+first,+rows,+weights,+offsets,+output} = query
  {native_gradient(Kernel.leaf_band(context,Kernel.Source{storage,base,plane,first,rows},Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),Buffer.pack(~F32,output))) == storage : Array<F32>}
def canonical_gradient(+query: GradientQuery,+tree: Tree.Tree<F32>) -> gradient_property(query,Tree.pack(F32,tree),tree):
  GradientQuery{+context,+base,+plane,+first,+rows,+weights,+offsets,+output} = query
  native_leaf_gradient(context,Source{tree,base,plane,first,rows},weights,offsets,output)
def gradient_owner(+query: GradientQuery,storage: Array<F32>) -> gradient_property(query,storage,Tree.reflect(F32,storage)):
  Storage.lift_observation(~F32,~GradientQuery,~gradient_property,query,storage,Storage.observe(F32,storage),canonical_gradient(query))
