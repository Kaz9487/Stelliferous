# A leaf on buffer models: its chunks fill the tiles of the gradient rows they
# read, and its phases run their jobs on them. A step keeps the model of every
# buffer it returns, so what the buffers hold is told by their contents.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../kernel_input_gradient_fill.bend as Fill
import ../kernel_gemm.bend as Gemm
import ../kernel_gemm_buffer.bend as GemmBuffer
import ../kernel_tile.bend as Tile
import ../tensor.bend as Tensors
import ../traversal_array.bend as Traversal
import ../storage_buffer.bend as Storage
import ./kernel_gemm_certificate.bend as GemmCertificate
import ./kernel_gemm_refinement.bend as GemmLaws
import ./kernel_gemm_jobs.bend as GemmJobs
import ./kernel_gemm_buffer_refinement.bend as GemmContents
import ./kernel_input_gradient_fill_refinement.bend as Filling
import ./kernel_input_gradient_fill_certificate.bend as FillCertificate
import ./storage_certificate.bend as Certificate
import ./traversal_certificate.bend as TraversalCertificate
import ./storage_tree_model.bend as Tree
import ./storage_buffer_model.bend as Buffer
import ./storage_allocation_refinement.bend as Allocation
import ./storage_refinement.bend as StorageLaws
import ./traversal_array_refinement.bend as Arrays
import ./write_program.bend as Writes
import ../tensor_tiled.bend as Tiled

# ---- Jobs ----

type Jobs is Data:
  Jobs{tiles: Buffer.Model<Tile.Tile8>,weights: Buffer.Model<Tile.Tile8>,output: Buffer.Model<Tile.Tile8>}

def represent_jobs(state: Jobs) -> Kernel.Jobs:
  Jobs{tiles,weights,output} = state
  Kernel.Jobs{Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,output)}

# The gradient tiles are the rows of the run, the weights its lanes.
def run(+count: Nat,+plan: Kernel.Plan,+state: Jobs) -> GemmBuffer.Kept<Tile.Tile8>:
  Jobs{+tiles,+weights,+output} = state
  GemmBuffer.run(~Kernel.Plan,~Kernel.Cursor(),~Tile.Tile8,~Kernel.job,~Kernel.entries,~Gemm.row_times,~Gemm.write_tiles(~Gemm.unbiased,~Gemm.plus),
    ~GemmCertificate.write_tiles(~Gemm.unbiased,~GemmCertificate.unbiased_read,~Gemm.plus),count,plan,
    Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,weights),Tensors.buffer(~F32,0.0,1),Buffer.pack(~Tile.Tile8,output))

def kept_tiles(kept: GemmBuffer.Kept<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  GemmBuffer.Kept{tiles,weights,bias,output} = kept
  tiles

def kept_weights(kept: GemmBuffer.Kept<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  GemmBuffer.Kept{tiles,weights,bias,output} = kept
  weights

def kept_output(kept: GemmBuffer.Kept<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  GemmBuffer.Kept{tiles,weights,bias,output} = kept
  output

def kept_split(kept: GemmBuffer.Kept<Tile.Tile8>) ->
  {Kernel.jobs_kept(kept) == Kernel.Jobs{kept_tiles(kept),kept_weights(kept),kept_output(kept)} : Kernel.Jobs}:
  GemmBuffer.Kept{tiles,weights,bias,output} = kept
  {==}

# A run of count jobs of a plan, fresh.
def jobs(+count: Nat,+plan: Kernel.Plan,+state: Jobs) -> Jobs:
  Jobs{Buffer.remodel(~Tile.Tile8,kept_tiles(run(count,plan,state))),Buffer.remodel(~Tile.Tile8,kept_weights(run(count,plan,state))),
    Buffer.remodel(~Tile.Tile8,kept_output(run(count,plan,state)))}

def jobs_matches(+count: Nat,+plan: Kernel.Plan,+state: Jobs) -> {Kernel.jobs(count,plan,represent_jobs(state)) == represent_jobs(jobs(count,plan,state)) : Kernel.Jobs}:
  Jobs{+tiles,+weights,+output} = state
  +model = {Jobs{tiles,weights,output} : Jobs}
  %Equal.sym(Storage.Buffer<Tile.Tile8>,Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,kept_tiles(run(count,plan,model)))),kept_tiles(run(count,plan,model)),Buffer.remodeled(~Tile.Tile8,kept_tiles(run(count,plan,model)))) :
    {Kernel.jobs_kept(run(count,plan,model)) == Kernel.Jobs{_,Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,kept_weights(run(count,plan,model)))),Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,kept_output(run(count,plan,model))))} : Kernel.Jobs}
  %Equal.sym(Storage.Buffer<Tile.Tile8>,Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,kept_weights(run(count,plan,model)))),kept_weights(run(count,plan,model)),Buffer.remodeled(~Tile.Tile8,kept_weights(run(count,plan,model)))) :
    {Kernel.jobs_kept(run(count,plan,model)) == Kernel.Jobs{kept_tiles(run(count,plan,model)),_,Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,kept_output(run(count,plan,model))))} : Kernel.Jobs}
  %Equal.sym(Storage.Buffer<Tile.Tile8>,Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,kept_output(run(count,plan,model)))),kept_output(run(count,plan,model)),Buffer.remodeled(~Tile.Tile8,kept_output(run(count,plan,model)))) :
    {Kernel.jobs_kept(run(count,plan,model)) == Kernel.Jobs{kept_tiles(run(count,plan,model)),kept_weights(run(count,plan,model)),_} : Kernel.Jobs}
  kept_split(run(count,plan,model))

# The bias a run reads: one +0.
def bias() -> Buffer.Model<F32>:
  Buffer.remodel(~F32,Tensors.buffer(~F32,0.0,1))

# The trees the jobs of a state read: the gradient tiles as rows, the weights
# as lanes.
def readers(+state: Jobs) -> GemmJobs.Readers:
  Jobs{+tiles,+weights,output} = state
  GemmJobs.Readers{Buffer.tree(~Tile.Tile8,tiles),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~F32,bias())}

def jobs_writes(count: Nat,+plan: Kernel.Plan,+readers: GemmJobs.Readers) -> List<&2,Writes.Assignment<Tile.Tile8>>:
  GemmJobs.all_writes(~Kernel.Plan,~Kernel.Cursor(),~Kernel.job,~Kernel.entries,~GemmLaws.RowFirst{},~GemmLaws.Tiles{},~GemmLaws.unbiased_value,~Gemm.plus,count,0,plan,readers)

def tiles_of(state: Jobs) -> Buffer.Model<Tile.Tile8>:
  Jobs{tiles,weights,output} = state
  tiles

def weights_of(state: Jobs) -> Buffer.Model<Tile.Tile8>:
  Jobs{tiles,weights,output} = state
  weights

def output_of(state: Jobs) -> Buffer.Model<Tile.Tile8>:
  Jobs{tiles,weights,output} = state
  output

# A run keeps the tiles and the weights and writes its jobs to the output.
def run_contents(+count: Nat,+plan: Kernel.Plan,+state: Jobs) ->
  {GemmContents.contents(~Tile.Tile8,run(count,plan,state)) == GemmContents.Contents{Buffer.content(~Tile.Tile8,tiles_of(state)),Buffer.content(~Tile.Tile8,weights_of(state)),
    Buffer.content(~F32,bias()),Buffer.Content{Writes.pure(Tile.Tile8,jobs_writes(count,plan,readers(state)),Buffer.tree(~Tile.Tile8,output_of(state))),Buffer.length(~Tile.Tile8,output_of(state))}} :
    GemmContents.Contents<Tile.Tile8>}:
  Jobs{+tiles,+weights,+output} = state
  %Buffer.remodeled(~F32,Tensors.buffer(~F32,0.0,1)) :
    {GemmContents.contents(~Tile.Tile8,GemmBuffer.run(~Kernel.Plan,~Kernel.Cursor(),~Tile.Tile8,~Kernel.job,~Kernel.entries,~Gemm.row_times,~Gemm.write_tiles(~Gemm.unbiased,~Gemm.plus),
    ~GemmCertificate.write_tiles(~Gemm.unbiased,~GemmCertificate.unbiased_read,~Gemm.plus),count,plan,
        Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,weights),_,Buffer.pack(~Tile.Tile8,output))) ==
      GemmContents.Contents{Buffer.content(~Tile.Tile8,tiles),Buffer.content(~Tile.Tile8,weights),Buffer.content(~F32,bias()),
        Buffer.Content{Writes.pure(Tile.Tile8,jobs_writes(count,plan,readers(Jobs{tiles,weights,output})),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output)}} : GemmContents.Contents<Tile.Tile8>}
  GemmContents.plan_writes(~Kernel.Plan,~Kernel.Cursor(),~Kernel.job,~Kernel.entries,~Gemm.row_times,~GemmLaws.RowFirst{},~GemmLaws.row_first,~GemmLaws.Tiles{},~Gemm.write_tiles(~Gemm.unbiased,~Gemm.plus),
    ~GemmCertificate.write_tiles(~Gemm.unbiased,~GemmCertificate.unbiased_read,~Gemm.plus),~GemmLaws.unbiased_value,~Gemm.plus,
    ~GemmLaws.tiles_stored(~Gemm.unbiased,~GemmLaws.unbiased_value,~GemmLaws.unbiased_fetched,~Gemm.plus),count,plan,tiles,weights,bias(),output)

# What the buffers of jobs hold.
type View is Data:
  View{tiles: Buffer.Content<Tile.Tile8>,weights: Buffer.Content<Tile.Tile8>,output: Buffer.Content<Tile.Tile8>}

def view(+state: Jobs) -> View:
  Jobs{+tiles,+weights,+output} = state
  View{Buffer.content(~Tile.Tile8,tiles),Buffer.content(~Tile.Tile8,weights),Buffer.content(~Tile.Tile8,output)}

def contents_view(contents: GemmContents.Contents<Tile.Tile8>) -> View:
  GemmContents.Contents{tiles,weights,bias,output} = contents
  View{tiles,weights,output}

def kept_view(kept: GemmBuffer.Kept<Tile.Tile8>) ->
  {View{Buffer.contents(~Tile.Tile8,kept_tiles(kept)),Buffer.contents(~Tile.Tile8,kept_weights(kept)),Buffer.contents(~Tile.Tile8,kept_output(kept))} ==
    contents_view(GemmContents.contents(~Tile.Tile8,kept)) : View}:
  GemmBuffer.Kept{tiles,weights,bias,output} = kept
  {==}

# A run keeps the tiles and the weights and writes its jobs to the output.
def jobs_view(+count: Nat,+plan: Kernel.Plan,+state: Jobs) ->
  {view(jobs(count,plan,state)) == View{Buffer.content(~Tile.Tile8,tiles_of(state)),Buffer.content(~Tile.Tile8,weights_of(state)),
    Buffer.Content{Writes.pure(Tile.Tile8,jobs_writes(count,plan,readers(state)),Buffer.tree(~Tile.Tile8,output_of(state))),Buffer.length(~Tile.Tile8,output_of(state))}} : View}:
  +wanted = {View{Buffer.content(~Tile.Tile8,tiles_of(state)),Buffer.content(~Tile.Tile8,weights_of(state)),
    Buffer.Content{Writes.pure(Tile.Tile8,jobs_writes(count,plan,readers(state)),Buffer.tree(~Tile.Tile8,output_of(state))),Buffer.length(~Tile.Tile8,output_of(state))}} : View}
  %Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,kept_tiles(run(count,plan,state)))),Buffer.contents(~Tile.Tile8,kept_tiles(run(count,plan,state))),
      Buffer.remodel_content(~Tile.Tile8,kept_tiles(run(count,plan,state)))) :
    {View{_,Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,kept_weights(run(count,plan,state)))),Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,kept_output(run(count,plan,state))))} == wanted : View}
  %Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,kept_weights(run(count,plan,state)))),Buffer.contents(~Tile.Tile8,kept_weights(run(count,plan,state))),
      Buffer.remodel_content(~Tile.Tile8,kept_weights(run(count,plan,state)))) :
    {View{Buffer.contents(~Tile.Tile8,kept_tiles(run(count,plan,state))),_,Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,kept_output(run(count,plan,state))))} == wanted : View}
  %Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,kept_output(run(count,plan,state)))),Buffer.contents(~Tile.Tile8,kept_output(run(count,plan,state))),
      Buffer.remodel_content(~Tile.Tile8,kept_output(run(count,plan,state)))) :
    {View{Buffer.contents(~Tile.Tile8,kept_tiles(run(count,plan,state))),Buffer.contents(~Tile.Tile8,kept_weights(run(count,plan,state))),_} == wanted : View}
  Equal.trans(View,View{Buffer.contents(~Tile.Tile8,kept_tiles(run(count,plan,state))),Buffer.contents(~Tile.Tile8,kept_weights(run(count,plan,state))),Buffer.contents(~Tile.Tile8,kept_output(run(count,plan,state)))},
    contents_view(GemmContents.contents(~Tile.Tile8,run(count,plan,state))),wanted,kept_view(run(count,plan,state)),
    Equal.cong(GemmContents.Contents<Tile.Tile8>,View,contents_view,GemmContents.contents(~Tile.Tile8,run(count,plan,state)),
      GemmContents.Contents{Buffer.content(~Tile.Tile8,tiles_of(state)),Buffer.content(~Tile.Tile8,weights_of(state)),Buffer.content(~F32,bias()),
        Buffer.Content{Writes.pure(Tile.Tile8,jobs_writes(count,plan,readers(state)),Buffer.tree(~Tile.Tile8,output_of(state))),Buffer.length(~Tile.Tile8,output_of(state))}},
      run_contents(count,plan,state)))

# The readers of jobs, from what their buffers hold.
def view_readers(view: View) -> GemmJobs.Readers:
  View{tiles,weights,output} = view
  GemmJobs.Readers{Buffer.content_tree(~Tile.Tile8,tiles),Buffer.content_tree(~Tile.Tile8,weights),Buffer.tree(~F32,bias())}

def readers_view(+state: Jobs) -> {readers(state) == view_readers(view(state)) : GemmJobs.Readers}:
  Jobs{+tiles,+weights,+output} = state
  Buffer.Model{+tiles_tree,tiles_length,tiles_certificate} = tiles
  Buffer.Model{+weights_tree,weights_length,weights_certificate} = weights
  {==}

# A run keeps what its jobs read.
def jobs_readers(+count: Nat,+plan: Kernel.Plan,+state: Jobs) -> {readers(jobs(count,plan,state)) == readers(state) : GemmJobs.Readers}:
  Jobs{+tiles,+weights,+output} = state
  +model = {Jobs{tiles,weights,output} : Jobs}
  +after = jobs(count,plan,model)
  Equal.trans(GemmJobs.Readers,readers(after),view_readers(view(after)),readers(model),readers_view(after),
    Equal.trans(GemmJobs.Readers,view_readers(view(after)),view_readers(view(model)),readers(model),
      Equal.cong(View,GemmJobs.Readers,view_readers,view(after),
        View{Buffer.content(~Tile.Tile8,tiles),Buffer.content(~Tile.Tile8,weights),
          Buffer.Content{Writes.pure(Tile.Tile8,jobs_writes(count,plan,readers(model)),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output)}},
        jobs_view(count,plan,model)),
      Equal.sym(GemmJobs.Readers,readers(model),view_readers(view(model)),readers_view(model))))

def view_output(view: View) -> Buffer.Content<Tile.Tile8>:
  View{tiles,weights,output} = view
  output

def output_view(+state: Jobs) -> {Buffer.tree(~Tile.Tile8,output_of(state)) == Buffer.content_tree(~Tile.Tile8,view_output(view(state))) : Tree.Tree<Tile.Tile8>}:
  Jobs{+tiles,+weights,+output} = state
  Buffer.Model{output_tree,output_length,output_certificate} = output
  {==}

def length_view(+state: Jobs) -> {Buffer.length(~Tile.Tile8,output_of(state)) == Buffer.content_length(~Tile.Tile8,view_output(view(state))) : Nat}:
  Jobs{+tiles,+weights,+output} = state
  Buffer.Model{output_tree,output_length,output_certificate} = output
  {==}

# A run writes its jobs to the output.
def jobs_output(+count: Nat,+plan: Kernel.Plan,+state: Jobs) ->
  {Buffer.tree(~Tile.Tile8,output_of(jobs(count,plan,state))) == Writes.pure(Tile.Tile8,jobs_writes(count,plan,readers(state)),Buffer.tree(~Tile.Tile8,output_of(state))) : Tree.Tree<Tile.Tile8>}:
  +after = jobs(count,plan,state)
  Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,output_of(after)),Buffer.content_tree(~Tile.Tile8,view_output(view(after))),
    Writes.pure(Tile.Tile8,jobs_writes(count,plan,readers(state)),Buffer.tree(~Tile.Tile8,output_of(state))),
    output_view(after),
    Equal.cong(View,Tree.Tree<Tile.Tile8>,view => Buffer.content_tree(~Tile.Tile8,view_output(view)),view(after),
      View{Buffer.content(~Tile.Tile8,tiles_of(state)),Buffer.content(~Tile.Tile8,weights_of(state)),
        Buffer.Content{Writes.pure(Tile.Tile8,jobs_writes(count,plan,readers(state)),Buffer.tree(~Tile.Tile8,output_of(state))),Buffer.length(~Tile.Tile8,output_of(state))}},
      jobs_view(count,plan,state)))

# A run keeps the length of its output.
def jobs_length(+count: Nat,+plan: Kernel.Plan,+state: Jobs) -> {Buffer.length(~Tile.Tile8,output_of(jobs(count,plan,state))) == Buffer.length(~Tile.Tile8,output_of(state)) : Nat}:
  +after = jobs(count,plan,state)
  Equal.trans(Nat,Buffer.length(~Tile.Tile8,output_of(after)),Buffer.content_length(~Tile.Tile8,view_output(view(after))),Buffer.length(~Tile.Tile8,output_of(state)),
    length_view(after),
    Equal.cong(View,Nat,view => Buffer.content_length(~Tile.Tile8,view_output(view)),view(after),
      View{Buffer.content(~Tile.Tile8,tiles_of(state)),Buffer.content(~Tile.Tile8,weights_of(state)),
        Buffer.Content{Writes.pure(Tile.Tile8,jobs_writes(count,plan,readers(state)),Buffer.tree(~Tile.Tile8,output_of(state))),Buffer.length(~Tile.Tile8,output_of(state))}},
      jobs_view(count,plan,state)))

# ---- Phases ----

type Leaf is Data:
  Leaf{tiles: Buffer.Model<Tile.Tile8>,weights: Buffer.Model<Tile.Tile8>,output: Buffer.Model<Tile.Tile8>,weight_base: U32}

def represent_leaf(state: Leaf) -> Kernel.Leaf:
  Leaf{tiles,weights,output,weight_base} = state
  Kernel.Leaf{Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,output),weight_base}

def as_jobs(state: Leaf) -> Jobs:
  Leaf{tiles,weights,output,weight_base} = state
  Jobs{tiles,weights,output}

def leaf_from_jobs(+weight_base: U32,state: Jobs) -> Leaf:
  Jobs{tiles,weights,output} = state
  Leaf{tiles,weights,output,weight_base}

def finished_matches(+weight_base: U32,+state: Jobs) -> {Kernel.leaf_finished(weight_base,represent_jobs(state)) == represent_leaf(leaf_from_jobs(weight_base,state)) : Kernel.Leaf}:
  Jobs{tiles,weights,output} = state
  {==}

# The jobs of a phase in a chunk, the weights from weight base.
type Work is Data:
  Work{plan: Kernel.Plan,count: Nat}

def work_plan(work: Work) -> Kernel.Plan:
  Work{plan,count} = work
  plan

def work_count(work: Work) -> Nat:
  Work{plan,count} = work
  count

def work(+phase: Kernel.Phase,+chunk: Kernel.Chunk,+weight_base: U32) -> Work:
  Kernel.Phase{Kernel.Geometry{+c,o,+h,+w,+k,+s,p,gh,gw},Kernel.Axis{+yf,+yc,yt,+yr,+ys},Kernel.Axis{+xf,+xc,xt,+xr,+xs},+inner} = phase
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{geometry,+layout},+first,count,+target},+row,+rows,+low,+group} = chunk
  Kernel.Tiling{groups,+shifts,+outputs,+column_low,kernel,height,width,+chunk_groups,chunk_rows} = layout
  +start = Bool.pick(U32,U32.is_lt(yf,row),((row - yf + s - 1) / s : U32),0)
  +end = U32.min(yc,Bool.pick(U32,U32.is_lt(yf,(row + rows : U32)),((row + rows - yf + s - 1) / s : U32),0))
  +units = ((c + 7) / 8 : U32)
  +phase_groups = ((xc + 7) / 8 : U32)
  +span = Bool.pick(U32,U32.is_lt(group,phase_groups),U32.min(chunk_groups,(phase_groups - group : U32)),0)
  +group_step = (outputs * shifts + 1 : U32)
  +row_step = (chunk_groups * group_step : U32)
  Work{Kernel.Plan{units,span,inner,((start + ys + k - low) * row_step + xs + k - column_low : U32),row_step,group_step,
      (target + ((yf + start * s - first) * w + xf + group * 8 * s) * units : U32),(s * w : U32),s,units,c,(xc - group * 8 : U32),weight_base,
      xr,yr,(row_step - (xr - 1) : U32),(shifts - (yr - 1) * row_step - (xr - 1) : U32)},
    U32.to_nat(Bool.pick(U32,U32.is_lt(start,end),((end - start) * span * units : U32),0))}

def phase_jobs(+phase: Kernel.Phase,+chunk: Kernel.Chunk,+weight_base: U32,+state: Jobs) -> Jobs:
  +work = work(phase,chunk,weight_base)
  jobs(work_count(work),work_plan(work),state)

def phase_jobs_matches(+phase: Kernel.Phase,+chunk: Kernel.Chunk,+weight_base: U32,+state: Jobs) ->
  {Kernel.leaf_jobs(phase,chunk,weight_base,represent_jobs(state)) == represent_jobs(phase_jobs(phase,chunk,weight_base,state)) : Kernel.Jobs}:
  Kernel.Phase{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},Kernel.Axis{yf,yc,yt,yr,ys},Kernel.Axis{xf,xc,xt,xr,xs},inner} = phase
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{geometry,layout},first,count,target},row,rows,low,group} = chunk
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,chunk_groups,chunk_rows} = layout
  +work = work(phase,chunk,weight_base)
  jobs_matches(work_count(work),work_plan(work),state)

def selected(empty: Bool,+phase: Kernel.Phase,+chunk: Kernel.Chunk,state: Leaf) -> Leaf:
  match empty:
    case True{}: state
    case False{}:
      Kernel.Phase{Kernel.Geometry{+c,o,h,w,k,s,p,gh,gw},y,x,+inner} = phase
      Leaf{+tiles,+weights,+output,+weight_base} = state
      leaf_from_jobs((weight_base + (c + 7) / 8 * inner : U32),phase_jobs(phase,chunk,weight_base,Jobs{tiles,weights,output}))

def selected_matches(empty: Bool,+phase: Kernel.Phase,+chunk: Kernel.Chunk,+state: Leaf) ->
  {Kernel.leaf_selected(empty,phase,chunk,represent_leaf(state)) == represent_leaf(selected(empty,phase,chunk,state)) : Kernel.Leaf}:
  match empty:
    case True{}: {==}
    case False{}:
      Kernel.Phase{Kernel.Geometry{+c,o,h,w,k,s,p,gh,gw},y,x,+inner} = phase
      Leaf{+tiles,+weights,+output,+weight_base} = state
      +jobs = {Jobs{tiles,weights,output} : Jobs}
      +written = phase_jobs(phase,chunk,weight_base,jobs)
      %Equal.sym(Kernel.Jobs,Kernel.leaf_jobs(phase,chunk,weight_base,represent_jobs(jobs)),represent_jobs(written),phase_jobs_matches(phase,chunk,weight_base,jobs)) :
        {Kernel.leaf_finished((weight_base + (c + 7) / 8 * inner : U32),_) == represent_leaf(selected(False{},phase,chunk,state)) : Kernel.Leaf}
      finished_matches((weight_base + (c + 7) / 8 * inner : U32),written)

def phase_step(index: U32,chunk: Kernel.Chunk,state: Leaf) -> Leaf:
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{+geometry,layout},first,count,target},row,rows,low,group} = chunk
  +phase = Kernel.phase_of(geometry,index)
  selected(U32.is_eq(Kernel.phase_inner(phase),0),phase,chunk,state)

def phase_matches(+index: U32,+chunk: Kernel.Chunk,+state: Leaf) -> {Kernel.leaf_phase(index,chunk,represent_leaf(state)) == represent_leaf(phase_step(index,chunk,state)) : Kernel.Leaf}:
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{+geometry,layout},first,count,target},row,rows,low,group} = chunk
  +phase = Kernel.phase_of(geometry,index)
  selected_matches(U32.is_eq(Kernel.phase_inner(phase),0),phase,chunk,state)

def phases_matches(+count: Nat,+first: U32,+chunk: Kernel.Chunk,+state: Leaf) ->
  {Traversal.indexed_fold(~Kernel.Leaf,~Kernel.Chunk,~Kernel.leaf_phase,count,first,chunk,represent_leaf(state)) ==
    represent_leaf(Traversal.indexed_fold(~Leaf,~Kernel.Chunk,~phase_step,count,first,chunk,state)) : Kernel.Leaf}:
  Arrays.indexed_fold_refines_model(~Kernel.Leaf,~Leaf,~Kernel.Chunk,~Kernel.leaf_phase,~phase_step,~represent_leaf,~phase_matches,count,first,chunk,state)

# Phases keep what their jobs read.
def leaf_readers(state: Leaf) -> GemmJobs.Readers:

  readers(as_jobs(state))

def finished_readers(+weight_base: U32,+state: Jobs) -> {leaf_readers(leaf_from_jobs(weight_base,state)) == readers(state) : GemmJobs.Readers}:
  Jobs{tiles,weights,output} = state
  {==}

def selected_readers(empty: Bool,+phase: Kernel.Phase,+chunk: Kernel.Chunk,+state: Leaf) ->
  {leaf_readers(selected(empty,phase,chunk,state)) == leaf_readers(state) : GemmJobs.Readers}:
  match empty:
    case True{}: {==}
    case False{}:
      Kernel.Phase{Kernel.Geometry{+c,o,h,w,k,s,p,gh,gw},y,x,+inner} = phase
      Leaf{+tiles,+weights,+output,+weight_base} = state
      +jobs = {Jobs{tiles,weights,output} : Jobs}
      +written = phase_jobs(phase,chunk,weight_base,jobs)
      +work = work(phase,chunk,weight_base)
      Equal.trans(GemmJobs.Readers,leaf_readers(leaf_from_jobs((weight_base + (c + 7) / 8 * inner : U32),written)),readers(written),readers(jobs),
        finished_readers((weight_base + (c + 7) / 8 * inner : U32),written),jobs_readers(work_count(work),work_plan(work),jobs))

def phase_readers(+index: U32,+chunk: Kernel.Chunk,+state: Leaf) -> {leaf_readers(phase_step(index,chunk,state)) == leaf_readers(state) : GemmJobs.Readers}:
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{+geometry,layout},first,count,target},row,rows,low,group} = chunk
  +phase = Kernel.phase_of(geometry,index)
  selected_readers(U32.is_eq(Kernel.phase_inner(phase),0),phase,chunk,state)

def kept_readers(index: U32,chunk: Kernel.Chunk,readers: GemmJobs.Readers) -> GemmJobs.Readers:
  readers

def reader_step(+index: U32,+chunk: Kernel.Chunk,+state: Leaf) -> {leaf_readers(phase_step(index,chunk,state)) == kept_readers(index,chunk,leaf_readers(state)) : GemmJobs.Readers}:
  phase_readers(index,chunk,state)

def reader_fixed(+index: U32,+chunk: Kernel.Chunk,+readers: GemmJobs.Readers) -> {kept_readers(index,chunk,readers) == readers : GemmJobs.Readers}:
  {==}

def phases_readers(+count: Nat,+first: U32,+chunk: Kernel.Chunk,+state: Leaf) ->
  {leaf_readers(Traversal.indexed_fold(~Leaf,~Kernel.Chunk,~phase_step,count,first,chunk,state)) == leaf_readers(state) : GemmJobs.Readers}:
  Equal.trans(GemmJobs.Readers,leaf_readers(Traversal.indexed_fold(~Leaf,~Kernel.Chunk,~phase_step,count,first,chunk,state)),
    Traversal.indexed_fold(~GemmJobs.Readers,~Kernel.Chunk,~kept_readers,count,first,chunk,leaf_readers(state)),leaf_readers(state),
    Arrays.indexed_fold_observes(~Leaf,~GemmJobs.Readers,~Kernel.Chunk,~phase_step,~leaf_readers,~kept_readers,~reader_step,count,first,chunk,state),
    Arrays.indexed_fold_fixed(~GemmJobs.Readers,~Kernel.Chunk,~kept_readers,~reader_fixed,count,first,chunk,leaf_readers(state)))

# ---- Chunks ----

# The gradient rows a leaf reads: rows [first, first + rows) from base, the
# groups of a pixel side by side.
type Source is Data:
  Source{data: Tree.Tree<Tile.Tile8>,base: U32,first: U32,rows: U32}

def source_data(source: Source) -> Tree.Tree<Tile.Tile8>:
  Source{data,base,first,rows} = source
  data

def source_base(source: Source) -> U32:
  Source{data,base,first,rows} = source
  base

def source_first(source: Source) -> U32:
  Source{data,base,first,rows} = source
  first

def source_rows(source: Source) -> U32:
  Source{data,base,first,rows} = source
  rows

# The buffer holding them.
type Held is Data:
  Held{data: Buffer.Model<Tile.Tile8>,base: U32,first: U32,rows: U32}

def represent_held(held: Held) -> Kernel.Source:
  Held{data,base,first,rows} = held
  Kernel.Source{Buffer.pack(~Tile.Tile8,data),base,first,rows}

def held_source(held: Held) -> Source:
  Held{data,base,first,rows} = held
  Source{Buffer.tree(~Tile.Tile8,data),base,first,rows}

type Sweep is Data:
  Sweep{source: Held,weights: Buffer.Model<Tile.Tile8>,output: Buffer.Model<Tile.Tile8>}

def represent_sweep(state: Sweep) -> Kernel.Sweep:
  Sweep{source,weights,output} = state
  Kernel.Sweep{represent_held(source),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,output)}

type ChunkShape is Data:
  ChunkShape{chunk: Kernel.Chunk,filling: Fill.Filling,fill_count: Nat,tile_count: U32}

def shape_chunk(shape: ChunkShape) -> Kernel.Chunk:
  ChunkShape{chunk,filling,count,tiles} = shape
  chunk

def shape_filling(shape: ChunkShape) -> Fill.Filling:
  ChunkShape{chunk,filling,count,tiles} = shape
  filling

def shape_count(shape: ChunkShape) -> Nat:
  ChunkShape{chunk,filling,count,tiles} = shape
  count

def shape_tiles(shape: ChunkShape) -> U32:
  ChunkShape{chunk,filling,count,tiles} = shape
  tiles

def shape_call(+shape: ChunkShape) -> Fill.Call:
  Fill.Call{shape_count(shape),shape_filling(shape)}

def chunk_shape(index: U32,context: Kernel.LeafContext,source: Source) -> ChunkShape:
  Kernel.LeafContext{Kernel.Task{+geometry,+layout},+first,+count,target} = context
  Kernel.Geometry{c,+o,h,w,k,s,p,gh,gw} = geometry
  Kernel.Tiling{+groups,+shifts,+outputs,+column_low,+kernel,height,+width,+chunk_groups,+chunk_rows} = layout
  Source{data,+base,+source_first,+source_rows} = source
  +index = index
  +group_chunks = ((groups + chunk_groups - 1) / chunk_groups : U32)
  +row = (first + index / group_chunks * chunk_rows : U32)
  +rows = U32.min(chunk_rows,(first + count - row : U32))
  +group = (index % group_chunks * chunk_groups : U32)
  +span = U32.min(chunk_groups,(groups - group : U32))
  +low = Kernel.band_low(geometry,row,(row + rows : U32))
  +tile_rows = Kernel.band_rows(geometry,row,rows)
  ChunkShape{Kernel.Chunk{context,row,rows,low,group},
    Fill.Filling{shifts,outputs,column_low,kernel,width,chunk_groups,low,base,Tiled.groups(o),source_first,source_rows,group,span},
    U32.to_nat((tile_rows * span : U32)),(tile_rows * chunk_groups * (outputs * shifts + 1) : U32)}

# The tiles a chunk fills, zero first.
def fresh_buffer(+shape: ChunkShape) -> Storage.Buffer<Tile.Tile8>:
  Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),(shape_tiles(shape) + 1 : U32))

def fresh_tiles(+shape: ChunkShape) -> Tree.Tree<Tile.Tile8>:
  Tree.fresh(Tile.Tile8,Tensors.depth_for((shape_tiles(shape) + 1 : U32)),Tile.filled(0.0))

# The tiles a chunk fills from the gradient rows.
def fill_chunk(+shape: ChunkShape,+source: Source) -> Filling.Owners:
  Filling.run(shape_call(shape),Filling.Owners{source_data(source),fresh_tiles(shape)})

def filled(+shape: ChunkShape,+held: Held) -> Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  Held{+data,base,first,rows} = held
  Kernel.filled_tiles(shape_call(shape),(Buffer.pack(~Tile.Tile8,data),fresh_buffer(shape)))

def first_buffer(pair: Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  (source,tiles) = pair
  source

def second_buffer(pair: Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  (source,tiles) = pair
  tiles

def pair_split(pair: Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) ->
  {pair == (first_buffer(pair),second_buffer(pair)) : Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}:
  (source,tiles) = pair
  {==}

type Filled is Data:
  Filled{source: Buffer.Model<Tile.Tile8>,tiles: Buffer.Model<Tile.Tile8>}

def pack_filled(state: Filled) -> Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  Filled{source,tiles} = state
  (Buffer.pack(~Tile.Tile8,source),Buffer.pack(~Tile.Tile8,tiles))

def filled_source(state: Filled) -> Buffer.Model<Tile.Tile8>:
  Filled{source,tiles} = state
  source

def filled_tiles(state: Filled) -> Buffer.Model<Tile.Tile8>:
  Filled{source,tiles} = state
  tiles

def fill_held(+shape: ChunkShape,+held: Held) -> Filled:
  Filled{Buffer.remodel(~Tile.Tile8,first_buffer(filled(shape,held))),Buffer.remodel(~Tile.Tile8,second_buffer(filled(shape,held)))}

def fill_matches(+shape: ChunkShape,+held: Held) -> {filled(shape,held) == pack_filled(fill_held(shape,held)) : Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}:
  %Equal.sym(Storage.Buffer<Tile.Tile8>,Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,first_buffer(filled(shape,held)))),first_buffer(filled(shape,held)),Buffer.remodeled(~Tile.Tile8,first_buffer(filled(shape,held)))) :
    {filled(shape,held) == (_,Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,second_buffer(filled(shape,held))))) : Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
  %Equal.sym(Storage.Buffer<Tile.Tile8>,Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,second_buffer(filled(shape,held)))),second_buffer(filled(shape,held)),Buffer.remodeled(~Tile.Tile8,second_buffer(filled(shape,held)))) :
    {filled(shape,held) == (first_buffer(filled(shape,held)),_) : Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
  pair_split(filled(shape,held))

type ChunkResult is Data:
  ChunkResult{gradient: Buffer.Model<Tile.Tile8>,leaf: Leaf}

def represent_result(result: ChunkResult) -> Storage.Buffer<Tile.Tile8> & Kernel.Leaf:
  ChunkResult{gradient,leaf} = result
  (Buffer.pack(~Tile.Tile8,gradient),represent_leaf(leaf))

def chunk_run(+chunk: Kernel.Chunk,weights: Buffer.Model<Tile.Tile8>,output: Buffer.Model<Tile.Tile8>,owners: Filled) -> ChunkResult:
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{+geometry,layout},first,count,target},row,rows,low,group} = chunk
  Filled{input,tiles} = owners
  ChunkResult{input,Traversal.indexed_fold(~Leaf,~Kernel.Chunk,~phase_step,Kernel.window_jobs(geometry),0,chunk,Leaf{tiles,weights,output,0})}

def chunk_run_matches(+chunk: Kernel.Chunk,+weights: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>,+owners: Filled) ->
  {Kernel.chunk_run(chunk,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,output),pack_filled(owners)) == represent_result(chunk_run(chunk,weights,output,owners)) :
    Storage.Buffer<Tile.Tile8> & Kernel.Leaf}:
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{+geometry,layout},first,count,target},row,rows,low,group} = chunk
  Filled{+input,+tiles} = owners
  +initial = {Leaf{tiles,weights,output,0} : Leaf}
  Equal.cong(Kernel.Leaf,Storage.Buffer<Tile.Tile8> & Kernel.Leaf,leaf => (Buffer.pack(~Tile.Tile8,input),leaf),
    Traversal.indexed_fold(~Kernel.Leaf,~Kernel.Chunk,~Kernel.leaf_phase,Kernel.window_jobs(geometry),0,chunk,represent_leaf(initial)),
    represent_leaf(Traversal.indexed_fold(~Leaf,~Kernel.Chunk,~phase_step,Kernel.window_jobs(geometry),0,chunk,initial)),phases_matches(Kernel.window_jobs(geometry),0,chunk,initial))

def finished_leaf(+base: U32,+first: U32,+rows: U32,data: Buffer.Model<Tile.Tile8>,state: Leaf) -> Sweep:
  Leaf{tiles,weights,output,weight_base} = state
  Sweep{Held{data,base,first,rows},weights,output}

def finished_chunk(+held: Held,result: ChunkResult) -> Sweep:
  Held{data,+base,+first,+rows} = held
  ChunkResult{gradient,leaf} = result
  finished_leaf(base,first,rows,gradient,leaf)

def finished_leaf_matches(+base: U32,+first: U32,+rows: U32,+data: Buffer.Model<Tile.Tile8>,+state: Leaf) ->
  {Kernel.chunk_leaf(base,first,rows,Buffer.pack(~Tile.Tile8,data),represent_leaf(state)) == represent_sweep(finished_leaf(base,first,rows,data,state)) : Kernel.Sweep}:
  Leaf{tiles,weights,output,weight_base} = state
  {==}

def finished_chunk_matches(+held: Held,+result: ChunkResult) ->
  {Kernel.chunk_finished(source_base(held_source(held)),source_first(held_source(held)),source_rows(held_source(held)),represent_result(result)) ==
    represent_sweep(finished_chunk(held,result)) : Kernel.Sweep}:
  Held{data,+base,+first,+rows} = held
  ChunkResult{+gradient,+leaf} = result
  finished_leaf_matches(base,first,rows,gradient,leaf)

def chunk_step(index: U32,context: Kernel.LeafContext,state: Sweep) -> Sweep:
  Sweep{+held,weights,output} = state
  +shape = chunk_shape(index,context,held_source(held))
  finished_chunk(held,chunk_run(shape_chunk(shape),weights,output,fill_held(shape,held)))

def chunk_matches(+index: U32,+context: Kernel.LeafContext,+state: Sweep) ->
  {Kernel.chunk_step(index,context,represent_sweep(state)) == represent_sweep(chunk_step(index,context,state)) : Kernel.Sweep}:
  Kernel.LeafContext{Kernel.Task{geometry,layout},first,count,target} = context
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,chunk_groups,chunk_rows} = layout
  Sweep{+held,+weights,+output} = state
  Held{+data,+base,+source_first,+source_rows} = held
  +source = {Held{data,base,source_first,source_rows} : Held}
  +shape = chunk_shape(index,context,held_source(source))
  +owners = fill_held(shape,source)
  +result = chunk_run(shape_chunk(shape),weights,output,owners)
  %Equal.sym(Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>,filled(shape,source),pack_filled(owners),fill_matches(shape,source)) :
    {Kernel.chunk_finished(base,source_first,source_rows,Kernel.chunk_run(shape_chunk(shape),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,output),_)) ==
      represent_sweep(chunk_step(index,context,state)) : Kernel.Sweep}
  %Equal.sym(Storage.Buffer<Tile.Tile8> & Kernel.Leaf,Kernel.chunk_run(shape_chunk(shape),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,output),pack_filled(owners)),
      represent_result(result),chunk_run_matches(shape_chunk(shape),weights,output,owners)) :
    {Kernel.chunk_finished(base,source_first,source_rows,_) == represent_sweep(chunk_step(index,context,state)) : Kernel.Sweep}
  finished_chunk_matches(source,result)

def chunks_matches(+count: Nat,+first: U32,+context: Kernel.LeafContext,+state: Sweep) ->
  {Traversal.indexed_fold(~Kernel.Sweep,~Kernel.LeafContext,~Kernel.chunk_step,count,first,context,represent_sweep(state)) ==
    represent_sweep(Traversal.indexed_fold(~Sweep,~Kernel.LeafContext,~chunk_step,count,first,context,state)) : Kernel.Sweep}:
  Arrays.indexed_fold_refines_model(~Kernel.Sweep,~Sweep,~Kernel.LeafContext,~Kernel.chunk_step,~chunk_step,~represent_sweep,~chunk_matches,count,first,context,state)

# ---- The leaf ----

type Part is Data:
  Part{gradient: Buffer.Model<Tile.Tile8>,weights: Buffer.Model<Tile.Tile8>,output: Buffer.Model<Tile.Tile8>}

def part_output(part: Part) -> Buffer.Model<Tile.Tile8>:
  Part{gradient,weights,output} = part
  output

def represent_part(part: Part) -> Kernel.Part:
  Part{gradient,weights,output} = part
  Kernel.Part{Buffer.pack(~Tile.Tile8,gradient),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,output)}

def content_length_equal(~Element: Data,+model: Buffer.Model<Element>,+tree: Tree.Tree<Element>,+length: Nat,
  +equal: {Buffer.content(~Element,model) == Buffer.Content{tree,length} : Buffer.Content<Element>}) -> {Buffer.length(~Element,model) == length : Nat}:
  Buffer.Model{+model_tree,+model_length,model_certificate} = model
  Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),Buffer.Content{model_tree,model_length},Buffer.Content{tree,length},equal)

def swept(state: Sweep) -> Part:
  Sweep{Held{data,base,first,rows},weights,output} = state
  Part{data,weights,output}

def swept_matches(+state: Sweep) -> {Kernel.swept(represent_sweep(state)) == represent_part(swept(state)) : Kernel.Part}:
  Sweep{Held{data,base,first,rows},weights,output} = state
  {==}

def chunk_count(context: Kernel.LeafContext) -> Nat:
  Kernel.LeafContext{Kernel.Task{geometry,layout},first,+count,target} = context
  Kernel.Tiling{+groups,shifts,outputs,column_low,kernel,height,width,+chunk_groups,+chunk_rows} = layout
  U32.to_nat(((count + chunk_rows - 1) / chunk_rows * ((groups + chunk_groups - 1) / chunk_groups) : U32))

def leaf_band(+context: Kernel.LeafContext,held: Held,weights: Buffer.Model<Tile.Tile8>,output: Buffer.Model<Tile.Tile8>) -> Part:
  swept(Traversal.indexed_fold(~Sweep,~Kernel.LeafContext,~chunk_step,chunk_count(context),0,context,Sweep{held,weights,output}))

def leaf_matches(+context: Kernel.LeafContext,+held: Held,+weights: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) ->
  {Kernel.leaf_band(context,represent_held(held),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,output)) == represent_part(leaf_band(context,held,weights,output)) : Kernel.Part}:
  Kernel.LeafContext{Kernel.Task{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},layout},first,count,target} = context
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,chunk_groups,chunk_rows} = layout
  +initial = {Sweep{held,weights,output} : Sweep}
  +final = Traversal.indexed_fold(~Sweep,~Kernel.LeafContext,~chunk_step,chunk_count(context),0,context,initial)
  %Equal.sym(Kernel.Sweep,Traversal.indexed_fold(~Kernel.Sweep,~Kernel.LeafContext,~Kernel.chunk_step,chunk_count(context),0,context,represent_sweep(initial)),represent_sweep(final),
      chunks_matches(chunk_count(context),0,context,initial)) :
    {Kernel.swept(_) == represent_part(leaf_band(context,held,weights,output)) : Kernel.Part}
  swept_matches(final)

# ---- What a chunk fill holds ----

def held_data(held: Held) -> Buffer.Model<Tile.Tile8>:
  Held{data,base,first,rows} = held
  data

type PairView is Data:
  PairView{source: Buffer.Content<Tile.Tile8>,tiles: Buffer.Content<Tile.Tile8>}

def pair_view(pair: Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) -> PairView:
  (source,tiles) = pair
  PairView{Buffer.contents(~Tile.Tile8,source),Buffer.contents(~Tile.Tile8,tiles)}

def buffers_view(+source_length: Nat,+length: Nat,result: Array<Tile.Tile8> & Array<Tile.Tile8>,
  certificates: TraversalCertificate.Certificates<Array<Tile.Tile8>,Tile.Tile8,Certificate.family(~Tile.Tile8),result>) ->
  {pair_view(Kernel.fill_buffers(source_length,length,result,certificates)) ==
    PairView{Buffer.Content{Tree.reflect(Tile.Tile8,Pair.fst(Array<Tile.Tile8>,Array<Tile.Tile8>,result)),source_length},
      Buffer.Content{Tree.reflect(Tile.Tile8,Pair.snd(Array<Tile.Tile8>,Array<Tile.Tile8>,result)),length}} : PairView}:
  (source,tiles) = result
  TraversalCertificate.Certificates{source_certificate,certificate} = certificates
  {==}

# A fill of buffer models keeps the source and holds the fill of their trees.
def fill_view(+call: Fill.Call,+data: Buffer.Model<Tile.Tile8>,+tiles: Buffer.Model<Tile.Tile8>) ->
  {pair_view(Kernel.filled_tiles(call,(Buffer.pack(~Tile.Tile8,data),Buffer.pack(~Tile.Tile8,tiles)))) ==
    PairView{Buffer.content(~Tile.Tile8,data),
      Buffer.Content{Filling.tiles(Filling.run(call,Filling.Owners{Buffer.tree(~Tile.Tile8,data),Buffer.tree(~Tile.Tile8,tiles)})),Buffer.length(~Tile.Tile8,tiles)}} : PairView}:
  Buffer.Model{+data_tree,+data_length,+data_certificate} = data
  Buffer.Model{+tiles_tree,+tiles_length,+tiles_certificate} = tiles
  +owners = {Filling.Owners{data_tree,tiles_tree} : Filling.Owners}
  +result = Filling.run(call,owners)
  +wanted = {PairView{Buffer.Content{data_tree,data_length},Buffer.Content{Filling.tiles(result),tiles_length}} : PairView}
  Equal.trans(PairView,pair_view(Kernel.fill_buffers(data_length,tiles_length,Fill.run(call,Tree.pack(Tile.Tile8,data_tree),Tree.pack(Tile.Tile8,tiles_tree)),
      FillCertificate.certificates(call,Tree.pack(Tile.Tile8,data_tree),Tree.pack(Tile.Tile8,tiles_tree),data_certificate,tiles_certificate))),
    PairView{Buffer.Content{Tree.reflect(Tile.Tile8,Pair.fst(Array<Tile.Tile8>,Array<Tile.Tile8>,Fill.run(call,Tree.pack(Tile.Tile8,data_tree),Tree.pack(Tile.Tile8,tiles_tree)))),data_length},
      Buffer.Content{Tree.reflect(Tile.Tile8,Pair.snd(Array<Tile.Tile8>,Array<Tile.Tile8>,Fill.run(call,Tree.pack(Tile.Tile8,data_tree),Tree.pack(Tile.Tile8,tiles_tree)))),tiles_length}},wanted,
    buffers_view(data_length,tiles_length,Fill.run(call,Tree.pack(Tile.Tile8,data_tree),Tree.pack(Tile.Tile8,tiles_tree)),FillCertificate.certificates(call,Tree.pack(Tile.Tile8,data_tree),Tree.pack(Tile.Tile8,tiles_tree),data_certificate,tiles_certificate)),
    %Equal.sym(Array<Tile.Tile8> & Array<Tile.Tile8>,Fill.run(call,Tree.pack(Tile.Tile8,data_tree),Tree.pack(Tile.Tile8,tiles_tree)),Filling.pack_owners(result),Filling.run_matches(call,owners)) :
      {PairView{Buffer.Content{Tree.reflect(Tile.Tile8,Pair.fst(Array<Tile.Tile8>,Array<Tile.Tile8>,_)),data_length},
        Buffer.Content{Tree.reflect(Tile.Tile8,Pair.snd(Array<Tile.Tile8>,Array<Tile.Tile8>,_)),tiles_length}} == wanted : PairView}
    %Equal.sym(Array<Tile.Tile8>,Pair.fst(Array<Tile.Tile8>,Array<Tile.Tile8>,Filling.pack_owners(result)),Tree.pack(Tile.Tile8,Filling.source(result)),Filling.packed_source(result)) :
      {PairView{Buffer.Content{Tree.reflect(Tile.Tile8,_),data_length},
        Buffer.Content{Tree.reflect(Tile.Tile8,Pair.snd(Array<Tile.Tile8>,Array<Tile.Tile8>,Filling.pack_owners(result))),tiles_length}} == wanted : PairView}
    %Equal.sym(Array<Tile.Tile8>,Pair.snd(Array<Tile.Tile8>,Array<Tile.Tile8>,Filling.pack_owners(result)),Tree.pack(Tile.Tile8,Filling.tiles(result)),Filling.packed_tiles(result)) :
      {PairView{Buffer.Content{Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,Filling.source(result))),data_length},Buffer.Content{Tree.reflect(Tile.Tile8,_),tiles_length}} == wanted : PairView}
    %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,Filling.source(result))),Filling.source(result),StorageLaws.reflect_packed(Tile.Tile8,Filling.source(result))) :
      {PairView{Buffer.Content{_,data_length},Buffer.Content{Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,Filling.tiles(result))),tiles_length}} == wanted : PairView}
    %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,Filling.tiles(result))),Filling.tiles(result),StorageLaws.reflect_packed(Tile.Tile8,Filling.tiles(result))) :
      {PairView{Buffer.Content{Filling.source(result),data_length},Buffer.Content{_,tiles_length}} == wanted : PairView}
    %Equal.sym(Tree.Tree<Tile.Tile8>,Filling.source(result),data_tree,Filling.run_keeps_source(call,owners)) :
      {PairView{Buffer.Content{_,data_length},Buffer.Content{Filling.tiles(result),tiles_length}} == wanted : PairView}
    {==})

def filled_view(state: Filled) -> PairView:
  Filled{source,tiles} = state
  PairView{Buffer.content(~Tile.Tile8,source),Buffer.content(~Tile.Tile8,tiles)}

def pair_view_split(pair: Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) ->
  {pair_view(pair) == PairView{Buffer.contents(~Tile.Tile8,first_buffer(pair)),Buffer.contents(~Tile.Tile8,second_buffer(pair))} : PairView}:
  (source,tiles) = pair
  {==}

# A fresh buffer holds the fresh tree.
def fresh_contents(+shape: ChunkShape) ->
  {Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,fresh_buffer(shape))) == Buffer.Content{fresh_tiles(shape),U32.to_nat((shape_tiles(shape) + 1 : U32))} : Buffer.Content<Tile.Tile8>}:
  +count = (shape_tiles(shape) + 1 : U32)
  Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,fresh_buffer(shape))),Buffer.contents(~Tile.Tile8,fresh_buffer(shape)),
    Buffer.Content{fresh_tiles(shape),U32.to_nat(count)},Buffer.remodel_content(~Tile.Tile8,fresh_buffer(shape)),
    Allocation.contents(~Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0),U32.to_nat(count)))

def content_tree_equal(~Element: Data,+model: Buffer.Model<Element>,+tree: Tree.Tree<Element>,+length: Nat,
  +equal: {Buffer.content(~Element,model) == Buffer.Content{tree,length} : Buffer.Content<Element>}) -> {Buffer.tree(~Element,model) == tree : Tree.Tree<Element>}:
  Buffer.Model{+model_tree,+model_length,model_certificate} = model
  Equal.cong(Buffer.Content<Element>,Tree.Tree<Element>,Buffer.content_tree(~Element),Buffer.Content{model_tree,model_length},Buffer.Content{tree,length},equal)

# The fill of a chunk keeps the gradient rows and holds the tree-level fill.
def held_view(+shape: ChunkShape,+held: Held) ->
  {filled_view(fill_held(shape,held)) ==
    PairView{Buffer.content(~Tile.Tile8,held_data(held)),Buffer.Content{Filling.tiles(fill_chunk(shape,held_source(held))),U32.to_nat((shape_tiles(shape) + 1 : U32))}} : PairView}:
  Held{+data,+base,+first,+rows} = held
  +model = {Held{data,base,first,rows} : Held}
  +fresh = Buffer.remodel(~Tile.Tile8,fresh_buffer(shape))
  +call = shape_call(shape)
  +wanted = {PairView{Buffer.content(~Tile.Tile8,data),Buffer.Content{Filling.tiles(fill_chunk(shape,held_source(model))),U32.to_nat((shape_tiles(shape) + 1 : U32))}} : PairView}
  %Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,first_buffer(filled(shape,model)))),Buffer.contents(~Tile.Tile8,first_buffer(filled(shape,model))),Buffer.remodel_content(~Tile.Tile8,first_buffer(filled(shape,model)))) :
    {PairView{_,Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,second_buffer(filled(shape,model))))} == wanted : PairView}
  %Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,second_buffer(filled(shape,model)))),Buffer.contents(~Tile.Tile8,second_buffer(filled(shape,model))),
      Buffer.remodel_content(~Tile.Tile8,second_buffer(filled(shape,model)))) :
    {PairView{Buffer.contents(~Tile.Tile8,first_buffer(filled(shape,model))),_} == wanted : PairView}
  %pair_view_split(filled(shape,model)) :
    {_ == wanted : PairView}
  %Buffer.remodeled(~Tile.Tile8,fresh_buffer(shape)) :
    {pair_view(Kernel.filled_tiles(call,(Buffer.pack(~Tile.Tile8,data),_))) == wanted : PairView}
  +fresh_content = fresh_contents(shape)
  %content_tree_equal(~Tile.Tile8,fresh,fresh_tiles(shape),U32.to_nat((shape_tiles(shape) + 1 : U32)),fresh_content) :
    {pair_view(Kernel.filled_tiles(call,(Buffer.pack(~Tile.Tile8,data),Buffer.pack(~Tile.Tile8,fresh)))) ==
      PairView{Buffer.content(~Tile.Tile8,data),Buffer.Content{Filling.tiles(Filling.run(call,Filling.Owners{Buffer.tree(~Tile.Tile8,data),_})),U32.to_nat((shape_tiles(shape) + 1 : U32))}} : PairView}
  %content_length_equal(~Tile.Tile8,fresh,fresh_tiles(shape),U32.to_nat((shape_tiles(shape) + 1 : U32)),fresh_content) :
    {pair_view(Kernel.filled_tiles(call,(Buffer.pack(~Tile.Tile8,data),Buffer.pack(~Tile.Tile8,fresh)))) ==
      PairView{Buffer.content(~Tile.Tile8,data),Buffer.Content{Filling.tiles(Filling.run(call,Filling.Owners{Buffer.tree(~Tile.Tile8,data),Buffer.tree(~Tile.Tile8,fresh)})),_}} : PairView}
  fill_view(call,data,fresh)

def view_source(view: PairView) -> Buffer.Content<Tile.Tile8>:
  PairView{source,tiles} = view
  source

def view_tiles(view: PairView) -> Buffer.Content<Tile.Tile8>:
  PairView{source,tiles} = view
  tiles

# The tiles of a chunk are the tree-level fill.
def held_tiles(+shape: ChunkShape,+held: Held) ->
  {Buffer.tree(~Tile.Tile8,filled_tiles(fill_held(shape,held))) == Filling.tiles(fill_chunk(shape,held_source(held))) : Tree.Tree<Tile.Tile8>}:
  +owners = fill_held(shape,held)
  Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,filled_tiles(owners)),Buffer.content_tree(~Tile.Tile8,view_tiles(filled_view(owners))),
    Filling.tiles(fill_chunk(shape,held_source(held))),
    Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,filled_tiles(owners))),Buffer.tree(~Tile.Tile8,filled_tiles(owners)),
      Buffer.content_tree_of(~Tile.Tile8,filled_tiles(owners))),
    Equal.cong(PairView,Tree.Tree<Tile.Tile8>,view => Buffer.content_tree(~Tile.Tile8,view_tiles(view)),filled_view(owners),
      PairView{Buffer.content(~Tile.Tile8,held_data(held)),Buffer.Content{Filling.tiles(fill_chunk(shape,held_source(held))),U32.to_nat((shape_tiles(shape) + 1 : U32))}},
      held_view(shape,held)))

# A chunk gives back the gradient rows it read.
def held_gradient(+shape: ChunkShape,+held: Held) ->
  {Buffer.tree(~Tile.Tile8,filled_source(fill_held(shape,held))) == Buffer.tree(~Tile.Tile8,held_data(held)) : Tree.Tree<Tile.Tile8>}:
  +owners = fill_held(shape,held)
  Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,filled_source(owners)),Buffer.content_tree(~Tile.Tile8,view_source(filled_view(owners))),Buffer.tree(~Tile.Tile8,held_data(held)),
    Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,filled_source(owners))),Buffer.tree(~Tile.Tile8,filled_source(owners)),
      Buffer.content_tree_of(~Tile.Tile8,filled_source(owners))),
    Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.content_tree(~Tile.Tile8,view_source(filled_view(owners))),Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,held_data(held))),Buffer.tree(~Tile.Tile8,held_data(held)),
      Equal.cong(PairView,Tree.Tree<Tile.Tile8>,view => Buffer.content_tree(~Tile.Tile8,view_source(view)),filled_view(owners),
        PairView{Buffer.content(~Tile.Tile8,held_data(held)),Buffer.Content{Filling.tiles(fill_chunk(shape,held_source(held))),U32.to_nat((shape_tiles(shape) + 1 : U32))}},
        held_view(shape,held)),
      Buffer.content_tree_of(~Tile.Tile8,held_data(held))))
