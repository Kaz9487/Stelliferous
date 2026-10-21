# A job reads the tiles, the packed weights and the table, and writes its
# ordered MACs through one strided block store: its write list depends on
# what it reads, never on the output.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../kernel_strided_store.bend as Stores
import ../kernel_rows.bend as Rows
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ../traversal_loop.bend as Loop
import ../traversal_array.bend as Traversal
import ./input_gradient_tiled_reduction.bend as Reduction
import ./kernel_strided_store_writes.bend as StoreModel
import ./kernel_strided_store_certificate.bend as StoreCertificate
import ./storage_tree_model.bend as Tree
import ./storage_buffer_model.bend as Buffer
import ./storage_refinement.bend as NativeStorage
import ./traversal_array_refinement.bend as Arrays
import ./write_program.bend as Writes
import ./write_program_fold.bend as Fold

type Jobs is Data:
  Jobs{tiles: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,offsets: Tree.Tree<U32>,output: Buffer.Model<F32>}

def represent(state: Jobs) -> Kernel.Jobs:
  Jobs{tiles,weights,offsets,output} = state
  Kernel.Jobs{Tree.pack(Tile.Tile8,tiles),Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),Buffer.pack(~F32,output)}
def job_output(state: Jobs) -> Buffer.Model<F32>:
  Jobs{tiles,weights,offsets,output} = state
  output

# What a job reads.
type Readers is Data:
  Readers{tiles: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,offsets: Tree.Tree<U32>}
def readers(state: Jobs) -> Readers:
  Jobs{tiles,weights,offsets,output} = state
  Readers{tiles,weights,offsets}

def execute(upper: Bool,+count: Nat,state: Reduction.Loop) -> Reduction.Loop:
  match upper:
    case False{}: Loop.run(~Reduction.Loop,~Reduction.step(~False{}),count,state)
    case True{}: Loop.run(~Reduction.Loop,~Reduction.step(~True{}),count,state)

def execute_matches(upper: Bool,+count: Nat,+state: Reduction.Loop) ->
  {Kernel.execute(upper,count,Reduction.represent(state)) == Reduction.represent(execute(upper,count,state)) : Kernel.Loop}:
  match upper:
    case False{}: Reduction.loop_matches(~False{},count,state)
    case True{}: Reduction.loop_matches(~True{},count,state)

def native_store(+store: Stores.Store,+values: Rows.Rows<Tile.Tile8>,output: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  Storage.Buffer{storage,length,certificate} = output
  Storage.Buffer{Stores.block(store,values,storage),length,StoreCertificate.certificate(store,values,storage,certificate)}

def stored(state: Reduction.Loop,store: Stores.Store,output: Buffer.Model<F32>) -> Jobs:
  Reduction.Loop{tiles,weights,offsets,index,context,values} = state
  Jobs{tiles,weights,offsets,Buffer.remodel(~F32,native_store(store,values,Buffer.pack(~F32,output)))}

def stored_matches(+state: Reduction.Loop,+store: Stores.Store,+output: Buffer.Model<F32>) ->
  {Kernel.stored(Reduction.represent(state),store,Buffer.pack(~F32,output)) == represent(stored(state,store,output)) : Kernel.Jobs}:
  Reduction.Loop{+tiles,+weights,+offsets,index,context,+values} = state
  Buffer.Model{tree,length,certificate} = output
  %Buffer.remodeled(~F32,native_store(store,values,Buffer.pack(~F32,output))) :
    {Kernel.Jobs{Tree.pack(Tile.Tile8,tiles),Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),_} == represent(stored(state,store,output)) : Kernel.Jobs}
  {==}

# A job's block store is its write list applied to the output tree.
def stored_contents(+store: Stores.Store,+values: Rows.Rows<Tile.Tile8>,+output: Buffer.Model<F32>) ->
  {Buffer.contents(~F32,native_store(store,values,Buffer.pack(~F32,output))) ==
    Buffer.Content{Writes.pure(F32,StoreModel.block_writes(StoreModel.Block{store,values}),Buffer.tree(~F32,output)),Buffer.length(~F32,output)} : Buffer.Content<F32>}:
  Buffer.Model{+tree,+length,certificate} = output
  +writes = StoreModel.block_writes(StoreModel.Block{store,values})
  +written = Writes.pure(F32,writes,tree)
  Equal.cong(Tree.Tree<F32>,Buffer.Content<F32>,tree => Buffer.Content{tree,length},Tree.reflect(F32,Stores.block(store,values,Tree.pack(F32,tree))),written,
    Equal.trans(Tree.Tree<F32>,Tree.reflect(F32,Stores.block(store,values,Tree.pack(F32,tree))),Tree.reflect(F32,Tree.pack(F32,written)),written,
      Equal.cong(Array<F32>,Tree.Tree<F32>,Tree.reflect(F32),Stores.block(store,values,Tree.pack(F32,tree)),Tree.pack(F32,written),
        Equal.trans(Array<F32>,Stores.block(store,values,Tree.pack(F32,tree)),Writes.native(F32,writes,Tree.pack(F32,tree)),Tree.pack(F32,written),
          StoreModel.block_native(StoreModel.Block{store,values},Tree.pack(F32,tree)),Writes.refinement(F32,writes,tree))),
      NativeStorage.reflect_packed(F32,written)))

def stored_content(+state: Reduction.Loop,+store: Stores.Store,+output: Buffer.Model<F32>) ->
  {Buffer.content(~F32,job_output(stored(state,store,output))) ==
    Buffer.Content{Writes.pure(F32,StoreModel.block_writes(StoreModel.Block{store,Reduction.accumulators(state)}),Buffer.tree(~F32,output)),Buffer.length(~F32,output)} : Buffer.Content<F32>}:
  Reduction.Loop{tiles,weights,offsets,index,context,+values} = state
  Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,job_output(stored(state,store,output))),Buffer.contents(~F32,native_store(store,values,Buffer.pack(~F32,output))),
    Buffer.Content{Writes.pure(F32,StoreModel.block_writes(StoreModel.Block{store,values}),Buffer.tree(~F32,output)),Buffer.length(~F32,output)},
    Buffer.remodel_content(~F32,native_store(store,values,Buffer.pack(~F32,output))),stored_contents(store,values,output))

def tree_from_content(+model: Buffer.Model<F32>,+expected: Tree.Tree<F32>,+length: Nat,equation: {Buffer.content(~F32,model) == Buffer.Content{expected,length} : Buffer.Content<F32>}) ->
  {Buffer.tree(~F32,model) == expected : Tree.Tree<F32>}:
  Buffer.Model{tree,stored_length,certificate} = model
  Equal.cong(Buffer.Content<F32>,Tree.Tree<F32>,Buffer.content_tree(~F32),Buffer.content(~F32,model),Buffer.Content{expected,length},equation)

def stored_tree(+state: Reduction.Loop,+store: Stores.Store,+output: Buffer.Model<F32>) ->
  {Buffer.tree(~F32,job_output(stored(state,store,output))) == Writes.pure(F32,StoreModel.block_writes(StoreModel.Block{store,Reduction.accumulators(state)}),Buffer.tree(~F32,output)) : Tree.Tree<F32>}:
  tree_from_content(job_output(stored(state,store,output)),Writes.pure(F32,StoreModel.block_writes(StoreModel.Block{store,Reduction.accumulators(state)}),Buffer.tree(~F32,output)),Buffer.length(~F32,output),stored_content(state,store,output))

def initial_loop(index: U32,plan: Kernel.Plan,readers: Readers) -> Reduction.Loop:
  Kernel.Plan{+units,+groups,inner,+source_base,+row_step,+group_step,target_base,target_row,step,plane,channels,columns,+weight_base,offset_base} = plan
  Readers{tiles,weights,offsets} = readers
  +index = index
  +unit = (index % units : U32)
  +group = (index / units % groups : U32)
  +line = (index / units / groups : U32)
  Reduction.Loop{tiles,weights,offsets,offset_base,Kernel.Context{(source_base + line * row_step + group * group_step : U32),(weight_base + unit / 2 * inner : U32)},Rows.filled(~Tile.Tile8,Tile.filled(0.0))}

def job_store(index: U32,plan: Kernel.Plan) -> Stores.Store:
  Kernel.Plan{+units,+groups,inner,source_base,row_step,group_step,+target_base,+target_row,+step,+plane,+channels,+columns,weight_base,offset_base} = plan
  +index = index
  +unit = (index % units : U32)
  +group = (index / units % groups : U32)
  +line = (index / units / groups : U32)
  Stores.Store{(target_base + unit * 4 * plane + line * target_row + group * 8 * step : U32),step,plane,U32.min(4,(channels - unit * 4 : U32)),U32.min(8,(columns - group * 8 : U32))}

def job_upper(index: U32,plan: Kernel.Plan) -> Bool:
  Kernel.Plan{units,groups,inner,source_base,row_step,group_step,target_base,target_row,step,plane,channels,columns,weight_base,offset_base} = plan
  U32.is_eq((index % units % 2 : U32),1)
def job_count(plan: Kernel.Plan) -> Nat:
  Kernel.Plan{units,groups,inner,source_base,row_step,group_step,target_base,target_row,step,plane,channels,columns,weight_base,offset_base} = plan
  U32.to_nat(inner)

def final_loop(index: U32,plan: Kernel.Plan,readers: Readers) -> Reduction.Loop:
  +index = index
  +plan = plan
  execute(job_upper(index,plan),job_count(plan),initial_loop(index,plan,readers))

def job(index: U32,plan: Kernel.Plan,state: Jobs) -> Jobs:
  +index = index
  +plan = plan
  +state = state
  stored(final_loop(index,plan,readers(state)),job_store(index,plan),job_output(state))

def job_matches(+index: U32,+plan: Kernel.Plan,+state: Jobs) -> {Kernel.job(index,plan,represent(state)) == represent(job(index,plan,state)) : Kernel.Jobs}:
  Kernel.Plan{units,groups,inner,source_base,row_step,group_step,target_base,target_row,step,plane,channels,columns,weight_base,offset_base} = plan
  Jobs{tiles,weights,offsets,+output} = state
  +initial = initial_loop(index,plan,readers(state))
  +final = execute(job_upper(index,plan),job_count(plan),initial)
  +store = job_store(index,plan)
  %Equal.sym(Kernel.Loop,Kernel.execute(job_upper(index,plan),job_count(plan),Reduction.represent(initial)),Reduction.represent(final),execute_matches(job_upper(index,plan),job_count(plan),initial)) :
    {Kernel.stored(_,store,Buffer.pack(~F32,output)) == represent(job(index,plan,state)) : Kernel.Jobs}
  stored_matches(final,store,output)

# The writes of job index, from what it reads.
def job_writes(index: U32,plan: Kernel.Plan,readers: Readers) -> List<&2,Writes.Assignment<F32>>:
  +index = index
  +plan = plan
  StoreModel.block_writes(StoreModel.Block{job_store(index,plan),Reduction.accumulators(final_loop(index,plan,readers))})

def job_tree(+index: U32,+plan: Kernel.Plan,+state: Jobs) ->
  {Buffer.tree(~F32,job_output(job(index,plan,state))) == Writes.pure(F32,job_writes(index,plan,readers(state)),Buffer.tree(~F32,job_output(state))) : Tree.Tree<F32>}:
  stored_tree(final_loop(index,plan,readers(state)),job_store(index,plan),job_output(state))

def loop_readers(source: Reduction.Sources) -> Readers:
  Reduction.Sources{tiles,weights,offsets,context} = source
  Readers{tiles,weights,offsets}

def execute_keeps_sources(upper: Bool,+count: Nat,+state: Reduction.Loop) ->
  {Reduction.sources(execute(upper,count,state)) == Reduction.sources(state) : Reduction.Sources}:
  match upper:
    case False{}: Reduction.loop_keeps_sources(~False{},count,state)
    case True{}: Reduction.loop_keeps_sources(~True{},count,state)

def stored_readers(+state: Reduction.Loop,+store: Stores.Store,+output: Buffer.Model<F32>) ->
  {readers(stored(state,store,output)) == loop_readers(Reduction.sources(state)) : Readers}:
  Reduction.Loop{tiles,weights,offsets,index,context,values} = state
  {==}

def initial_readers(+index: U32,+plan: Kernel.Plan,+readers: Readers) -> {loop_readers(Reduction.sources(initial_loop(index,plan,readers))) == readers : Readers}:
  Kernel.Plan{units,groups,inner,source_base,row_step,group_step,target_base,target_row,step,plane,channels,columns,weight_base,offset_base} = plan
  Readers{tiles,weights,offsets} = readers
  {==}

def job_keeps_readers(+index: U32,+plan: Kernel.Plan,+state: Jobs) -> {readers(job(index,plan,state)) == readers(state) : Readers}:
  +initial = initial_loop(index,plan,readers(state))
  +final = execute(job_upper(index,plan),job_count(plan),initial)
  Equal.trans(Readers,readers(job(index,plan,state)),loop_readers(Reduction.sources(final)),readers(state),stored_readers(final,job_store(index,plan),job_output(state)),
    Equal.trans(Readers,loop_readers(Reduction.sources(final)),loop_readers(Reduction.sources(initial)),readers(state),
      Equal.cong(Reduction.Sources,Readers,loop_readers,Reduction.sources(final),Reduction.sources(initial),execute_keeps_sources(job_upper(index,plan),job_count(plan),initial)),initial_readers(index,plan,readers(state))))

def jobs_matches(+count: Nat,+first: U32,+plan: Kernel.Plan,+state: Jobs) ->
  {Traversal.indexed_fold(~Kernel.Jobs,~Kernel.Plan,~Kernel.job,count,first,plan,represent(state)) == represent(Traversal.indexed_fold(~Jobs,~Kernel.Plan,~job,count,first,plan,state)) : Kernel.Jobs}:
  Arrays.indexed_fold_refines_model(~Kernel.Jobs,~Jobs,~Kernel.Plan,~Kernel.job,~job,~represent,~job_matches,count,first,plan,state)

def readers_kept(index: U32,plan: Kernel.Plan,value: Readers) -> Readers:
  value
def reader_step(+index: U32,+plan: Kernel.Plan,+state: Jobs) ->
  {readers(job(index,plan,state)) == readers_kept(index,plan,readers(state)) : Readers}:
  job_keeps_readers(index,plan,state)
def reader_fixed(+index: U32,+plan: Kernel.Plan,+value: Readers) -> {readers_kept(index,plan,value) == value : Readers}:
  {==}

def output_tree(state: Jobs) -> Tree.Tree<F32>:
  Buffer.tree(~F32,job_output(state))
def job_output_tree(+index: U32,+plan: Kernel.Plan,+state: Jobs) ->
  {output_tree(job(index,plan,state)) == Writes.pure(F32,job_writes(index,plan,readers(state)),output_tree(state)) : Tree.Tree<F32>}:
  job_tree(index,plan,state)

# The writes of jobs first, first + 1, ... of one plan, in order.
def jobs_writes(count: Nat,+first: U32,+plan: Kernel.Plan,+readers: Readers) -> List<&2,Writes.Assignment<F32>>:
  Fold.all(~F32,~Kernel.Plan,~Readers,~job_writes,~readers_kept,count,first,plan,readers)

def jobs_tree(+count: Nat,+first: U32,+plan: Kernel.Plan,+state: Jobs) ->
  {output_tree(Traversal.indexed_fold(~Jobs,~Kernel.Plan,~job,count,first,plan,state)) == Writes.pure(F32,jobs_writes(count,first,plan,readers(state)),output_tree(state)) : Tree.Tree<F32>}:
  Fold.output(~Jobs,~Kernel.Plan,~F32,~Readers,~job,~readers,~output_tree,~job_writes,~readers_kept,~reader_step,~job_output_tree,count,first,plan,state)

def jobs_keeps_readers(+count: Nat,+first: U32,+plan: Kernel.Plan,+state: Jobs) ->
  {readers(Traversal.indexed_fold(~Jobs,~Kernel.Plan,~job,count,first,plan,state)) == readers(state) : Readers}:
  Equal.trans(Readers,readers(Traversal.indexed_fold(~Jobs,~Kernel.Plan,~job,count,first,plan,state)),Traversal.indexed_fold(~Readers,~Kernel.Plan,~readers_kept,count,first,plan,readers(state)),readers(state),
    Arrays.indexed_fold_observes(~Jobs,~Readers,~Kernel.Plan,~job,~readers,~readers_kept,~reader_step,count,first,plan,state),
    Arrays.indexed_fold_fixed(~Readers,~Kernel.Plan,~readers_kept,~reader_fixed,count,first,plan,readers(state)))

