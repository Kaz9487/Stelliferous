# A pointwise run over packed trees is a model whose staged output is a
# program of write lists: each job appends the stores of its block, row r
# of the block at base + r * pairs when r is below its rows, the cells of
# the row plus the biases of the job's pair.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./traversal_array_refinement.bend as ArrayLaws
import ./write_program.bend as Writes
import ./write_program_fold.bend as Fold
import ./kernel_gemm_refinement.bend as GemmLaws
import ./kernel_gemm_point_refinement.bend as PointLaws
import ./kernel_gemm_point_spec.bend as PointSpec

# ---- The model of a run ----

type Operands is Data:
  Operands{bias: Tree.Tree<Tile.Tile8>,cells: Tree.Tree<Tile.Tile8>,pairs: Tree.Tree<Tile.Tile16>,staged: Writes.Program<Tile.Tile16>}

def represent(operands: Operands) -> Gemm.PointOperands:
  Operands{bias,cells,pairs,staged} = operands
  Gemm.PointOperands{Tree.pack(Tile.Tile8,bias),Tree.pack(Tile.Tile8,cells),Tree.pack(Tile.Tile16,pairs),Tree.pack(Tile.Tile16,Writes.evaluate(Tile.Tile16,staged))}

# The trees a job reads.
type Readers is Data:
  Readers{bias: Tree.Tree<Tile.Tile8>,cells: Tree.Tree<Tile.Tile8>,pairs: Tree.Tree<Tile.Tile16>}

def readers_of(operands: Operands) -> Readers:
  Operands{bias,cells,pairs,staged} = operands
  Readers{bias,cells,pairs}

# The stores of a block: row r at base + r * stride when r < rows, plus the
# biases.
def stores(+base: U32,+stride: U32,+rows: U32,+biases: Tile.Tile16,+accumulators: Gemm.Quad) -> List<&2,Writes.Assignment<Tile.Tile16>>:
  w = GemmLaws.put(~Tile.Tile16,U32.is_lt(0,rows),base,Tile.plus_pair(Gemm.quad0(accumulators),biases))
  w = List.append(&2,Writes.Assignment<Tile.Tile16>,w,GemmLaws.put(~Tile.Tile16,U32.is_lt(1,rows),(base + stride : U32),Tile.plus_pair(Gemm.quad1(accumulators),biases)))
  w = List.append(&2,Writes.Assignment<Tile.Tile16>,w,GemmLaws.put(~Tile.Tile16,U32.is_lt(2,rows),(base + 2 * stride : U32),Tile.plus_pair(Gemm.quad2(accumulators),biases)))
  List.append(&2,Writes.Assignment<Tile.Tile16>,w,GemmLaws.put(~Tile.Tile16,U32.is_lt(3,rows),(base + 3 * stride : U32),Tile.plus_pair(Gemm.quad3(accumulators),biases)))

def PostInner() -> Type:
  @+matrix: Gemm.PointMatrix -> @+sources: PointLaws.Sources -> PointLaws.Reduced -> PointLaws.Reduced

def whole_inner(+matrix: Gemm.PointMatrix,+sources: PointLaws.Sources,state: PointLaws.Reduced) -> PointLaws.Reduced:
  state

# The accumulators of a job after its complete groups and remaining indices.
def job_accumulators_with(~times: F32 -> F32 -> F32,~finish: PostInner(),+matrix: Gemm.PointMatrix,+place: Gemm.Place,
  +cells: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>) -> Gemm.Quad:
  +sources = {PointLaws.Sources{cells,pairs} : PointLaws.Sources}
  PointSpec.accumulators_of(finish(matrix,sources,PointLaws.groups(~times,Gemm.point_groups_of(matrix),sources,
    PointLaws.Reduced{PointSpec.job_start(matrix,place),Gemm.quad_zero()})))

def job_accumulators(+matrix: Gemm.PointMatrix,+place: Gemm.Place,+cells: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>) -> Gemm.Quad:
  job_accumulators_with(~Gemm.row_times,~whole_inner,matrix,place,cells,pairs)

def Accumulate() -> Type:
  @+matrix: Gemm.PointMatrix -> @+place: Gemm.Place -> @+cells: Tree.Tree<Tile.Tile8> -> @+pairs: Tree.Tree<Tile.Tile16> -> Gemm.Quad

# The stores of the job at a place.
def place_writes_by(~accumulate: Accumulate(),+matrix: Gemm.PointMatrix,+place: Gemm.Place,+readers: Readers) ->
  List<&2,Writes.Assignment<Tile.Tile16>>:
  Gemm.PointMatrix{Gemm.TileMatrix{+channels,positions,inner,+start,+output_positions},cell,stride} = matrix
  Gemm.Place{+pair,+unit} = place
  Readers{+bias,+cells,+pairs} = readers
  +groups = Gemm.blocks(channels)
  +pairs_count = Gemm.pairs_of(channels)
  +position = (start + unit * 4 : U32)
  stores((position * pairs_count + pair : U32),pairs_count,U32.min(4,(output_positions - position : U32)),
    Tile.Tiles{Tree.read(Tile.Tile8,bias,(pair * 2 : U32)),Tree.read(Tile.Tile8,bias,U32.min((pair * 2 + 1 : U32),(groups - 1 : U32)))},
    accumulate(matrix,place,cells,pairs))

def place_writes_with(~times: F32 -> F32 -> F32,~finish: PostInner(),+matrix: Gemm.PointMatrix,+place: Gemm.Place,+readers: Readers) -> List<&2,Writes.Assignment<Tile.Tile16>>:
  place_writes_by(~job_accumulators_with(~times,~finish),matrix,place,readers)

def place_writes(+matrix: Gemm.PointMatrix,+place: Gemm.Place,+readers: Readers) -> List<&2,Writes.Assignment<Tile.Tile16>>:
  place_writes_with(~Gemm.row_times,~whole_inner,matrix,place,readers)

def job_writes(+index: U32,+matrix: Gemm.PointMatrix,+readers: Readers) -> List<&2,Writes.Assignment<Tile.Tile16>>:
  place_writes(matrix,Gemm.point_place(matrix,index),readers)

# A job keeps the trees it reads and appends its stores.
def job_by(~accumulate: Accumulate(),index: U32,matrix: Gemm.PointMatrix,operands: Operands) -> Operands:
  Operands{+bias,+cells,+pairs,staged} = operands
  +index = index
  +matrix = matrix
  Operands{bias,cells,pairs,Writes.Append{place_writes_by(~accumulate,matrix,Gemm.point_place(matrix,index),Readers{bias,cells,pairs}),staged}}

def job_with(~times: F32 -> F32 -> F32,~finish: PostInner(),index: U32,matrix: Gemm.PointMatrix,operands: Operands) -> Operands:
  job_by(~job_accumulators_with(~times,~finish),index,matrix,operands)

def job(index: U32,matrix: Gemm.PointMatrix,operands: Operands) -> Operands:
  job_with(~Gemm.row_times,~whole_inner,index,matrix,operands)

# ---- The native job is the model's ----

def loop_represented(+cells: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+state: PointLaws.Reduced) ->
  {PointLaws.represent(PointLaws.Sources{cells,pairs},state) == Gemm.PointLoop{Tree.pack(Tile.Tile8,cells),Tree.pack(Tile.Tile16,pairs),
    PointSpec.cursor_of(state),PointSpec.accumulators_of(state)} : Gemm.PointLoop}:
  PointLaws.Reduced{+cursor,+accumulators} = state
  {==}

# Four puts on a packed tree are the stores of the block.
def stored_matches(+tree: Tree.Tree<Tile.Tile16>,+base: U32,+stride: U32,+rows: U32,+biases: Tile.Tile16,+accumulators: Gemm.Quad) ->
  {Gemm.point_stored(Tree.pack(Tile.Tile16,tree),base,stride,rows,biases,accumulators) == Writes.native(Tile.Tile16,stores(base,stride,rows,biases,accumulators),Tree.pack(Tile.Tile16,tree)) :
    Array<Tile.Tile16>}:
  -packed = Tree.pack(Tile.Tile16,tree)
  +p0 = GemmLaws.put(~Tile.Tile16,U32.is_lt(0,rows),base,Tile.plus_pair(Gemm.quad0(accumulators),biases))
  +p1 = GemmLaws.put(~Tile.Tile16,U32.is_lt(1,rows),(base + stride : U32),Tile.plus_pair(Gemm.quad1(accumulators),biases))
  +p2 = GemmLaws.put(~Tile.Tile16,U32.is_lt(2,rows),(base + 2 * stride : U32),Tile.plus_pair(Gemm.quad2(accumulators),biases))
  +w1 = List.append(&2,Writes.Assignment<Tile.Tile16>,p0,p1)
  +w2 = List.append(&2,Writes.Assignment<Tile.Tile16>,w1,p2)
  -n0 = Gemm.put(~Tile.Tile16,U32.is_lt(0,rows),packed,base,Tile.plus_pair(Gemm.quad0(accumulators),biases))
  -n1 = Gemm.put(~Tile.Tile16,U32.is_lt(1,rows),n0,(base + stride : U32),Tile.plus_pair(Gemm.quad1(accumulators),biases))
  -n2 = Gemm.put(~Tile.Tile16,U32.is_lt(2,rows),n1,(base + 2 * stride : U32),Tile.plus_pair(Gemm.quad2(accumulators),biases))
  e0 = GemmLaws.put_next(~Tile.Tile16,U32.is_lt(0,rows),packed,tree,[],base,Tile.plus_pair(Gemm.quad0(accumulators),biases),{==})
  e1 = GemmLaws.put_next(~Tile.Tile16,U32.is_lt(1,rows),n0,tree,p0,(base + stride : U32),Tile.plus_pair(Gemm.quad1(accumulators),biases),e0)
  e2 = GemmLaws.put_next(~Tile.Tile16,U32.is_lt(2,rows),n1,tree,w1,(base + 2 * stride : U32),Tile.plus_pair(Gemm.quad2(accumulators),biases),e1)
  GemmLaws.put_next(~Tile.Tile16,U32.is_lt(3,rows),n2,tree,w2,(base + 3 * stride : U32),Tile.plus_pair(Gemm.quad3(accumulators),biases),e2)

def PostMatches(~native: @+matrix: Gemm.PointMatrix -> Gemm.PointLoop -> Gemm.PointLoop,~model: PostInner()) -> Type:
  @+matrix: Gemm.PointMatrix -> @+sources: PointLaws.Sources -> @+state: PointLaws.Reduced ->
    {native(matrix,PointLaws.represent(sources,state)) == PointLaws.represent(sources,model(matrix,sources,state)) : Gemm.PointLoop}

def whole_matches(+matrix: Gemm.PointMatrix,+sources: PointLaws.Sources,+state: PointLaws.Reduced) ->
  {Gemm.whole_inner(matrix,PointLaws.represent(sources,state)) == PointLaws.represent(sources,whole_inner(matrix,sources,state)) : Gemm.PointLoop}:
  {==}

def job_matches_with(~times: F32 -> F32 -> F32,~tail: @+matrix: Gemm.PointMatrix -> Gemm.PointLoop -> Gemm.PointLoop,
  ~finish: PostInner(),~finish_matches: PostMatches(~tail,~finish),+index: U32,+matrix: Gemm.PointMatrix,+operands: Operands) ->
  {Gemm.point_job_with(~Tile.plus_pair,~times,~tail,index,matrix,represent(operands)) == represent(job_with(~times,~finish,index,matrix,operands)) : Gemm.PointOperands}:
  Gemm.PointMatrix{Gemm.TileMatrix{+channels,+positions,+inner,+start,+output_positions},+cell,+stride} = matrix
  Operands{+bias,+cells,+pairs,+staged} = operands
  +whole = {Gemm.PointMatrix{Gemm.TileMatrix{channels,positions,inner,start,output_positions},cell,stride} : Gemm.PointMatrix}
  +pairs_count = Gemm.pairs_of(channels)
  +units = ((positions + 3) / 4 : U32)
  +once = Gemm.streamed(Gemm.blocks(channels),inner)
  +pair = Gemm.job_group(once,index,pairs_count,units)
  +unit = Gemm.job_unit(once,index,pairs_count,units)
  +place = {Gemm.Place{pair,unit} : Gemm.Place}
  +sources = {PointLaws.Sources{cells,pairs} : PointLaws.Sources}
  +model = {PointLaws.Reduced{PointSpec.job_start(whole,place),Gemm.quad_zero()} : PointLaws.Reduced}
  +count = Gemm.point_groups_of(whole)
  +grouped = PointLaws.groups(~times,count,sources,model)
  +ended = finish(whole,sources,grouped)
  +tree = Writes.evaluate(Tile.Tile16,staged)
  -bias_array = Tree.pack(Tile.Tile8,bias)
  -cell_array = Tree.pack(Tile.Tile8,cells)
  -pair_array = Tree.pack(Tile.Tile16,pairs)
  +groups = Gemm.blocks(channels)
  +position = (start + unit * 4 : U32)
  +base = (position * pairs_count + pair : U32)
  +rows = U32.min(4,(output_positions - position : U32))
  +above = U32.min((pair * 2 + 1 : U32),(groups - 1 : U32))
  +accumulators = PointSpec.accumulators_of(ended)
  +lower = Tree.read(Tile.Tile8,bias,(pair * 2 : U32))
  +upper = Tree.read(Tile.Tile8,bias,above)
  -wanted = represent(job_with(~times,~finish,index,whole,Operands{bias,cells,pairs,staged}))
  %Equal.sym(Gemm.PointLoop,Loop.run(~Gemm.PointLoop,~Gemm.point_group(~times),count,PointLaws.represent(sources,model)),PointLaws.represent(sources,grouped),
    PointLaws.groups_matched(~times,count,sources,PointLaws.represent(sources,model),model,{==})) :
    {Gemm.point_finish(bias_array,Tree.pack(Tile.Tile16,tree),whole,place,tail(whole,_)) == wanted : Gemm.PointOperands}
  %Equal.sym(Gemm.PointLoop,tail(whole,PointLaws.represent(sources,grouped)),PointLaws.represent(sources,ended),finish_matches(whole,sources,grouped)) :
    {Gemm.point_finish(bias_array,Tree.pack(Tile.Tile16,tree),whole,place,_) == wanted : Gemm.PointOperands}
  %Equal.sym(Gemm.PointLoop,PointLaws.represent(sources,ended),Gemm.PointLoop{cell_array,pair_array,PointSpec.cursor_of(ended),accumulators},loop_represented(cells,pairs,ended)) :
    {Gemm.point_finish(bias_array,Tree.pack(Tile.Tile16,tree),whole,place,_) == wanted : Gemm.PointOperands}
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,bias_array,(pair * 2 : U32)),(bias_array,lower),StorageLaws.read(Tile.Tile8,bias,(pair * 2 : U32))) :
    {Gemm.lower_biased(cell_array,pair_array,Tree.pack(Tile.Tile16,tree),base,pairs_count,rows,above,accumulators,_) == wanted : Gemm.PointOperands}
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,bias_array,above),(bias_array,upper),StorageLaws.read(Tile.Tile8,bias,above)) :
    {Gemm.upper_biased(cell_array,pair_array,Tree.pack(Tile.Tile16,tree),base,pairs_count,rows,lower,accumulators,_) == wanted : Gemm.PointOperands}
  +writes = stores(base,pairs_count,rows,Tile.Tiles{lower,upper},accumulators)
  %Equal.sym(Array<Tile.Tile16>,Gemm.point_stored(Tree.pack(Tile.Tile16,tree),base,pairs_count,rows,Tile.Tiles{lower,upper},accumulators),
    Writes.native(Tile.Tile16,writes,Tree.pack(Tile.Tile16,tree)),stored_matches(tree,base,pairs_count,rows,Tile.Tiles{lower,upper},accumulators)) :
    {Gemm.PointOperands{bias_array,cell_array,pair_array,_} == wanted : Gemm.PointOperands}
  Equal.cong(Array<Tile.Tile16>,Gemm.PointOperands,staged_array => Gemm.PointOperands{bias_array,cell_array,pair_array,staged_array},
    Writes.native(Tile.Tile16,writes,Tree.pack(Tile.Tile16,tree)),Tree.pack(Tile.Tile16,Writes.pure(Tile.Tile16,writes,tree)),Writes.refinement(Tile.Tile16,writes,tree))

def job_matches(+index: U32,+matrix: Gemm.PointMatrix,+operands: Operands) ->
  {Gemm.point_job(index,matrix,represent(operands)) == represent(job(index,matrix,operands)) : Gemm.PointOperands}:
  Operands{+bias,+cells,+pairs,+staged} = operands
  job_matches_with(~Gemm.row_times,~Gemm.whole_inner,~whole_inner,~whole_matches,index,matrix,operands)

def job_tail_matches(+index: U32,+matrix: Gemm.PointMatrix,+operands: Operands) ->
  {Gemm.point_job_with(~Gemm.panel_bias,~Gemm.panel_times,~Gemm.finish_inner,index,matrix,represent(operands)) ==
    represent(job_with(~Gemm.panel_times,~PointLaws.tail_reduced,index,matrix,operands)) : Gemm.PointOperands}:
  job_matches_with(~Gemm.panel_times,~Gemm.finish_inner,~PointLaws.tail_reduced,~PointLaws.tail_matches,index,matrix,operands)

# Jobs 0 to count - 1, each appending its stores.
def run(count: Nat,matrix: Gemm.PointMatrix,operands: Operands) -> Operands:
  Traversal.indexed_fold(~Operands,~Gemm.PointMatrix,~job,count,0,matrix,operands)

def run_matches(+count: Nat,+matrix: Gemm.PointMatrix,+operands: Operands) ->
  {Gemm.point_run(count,matrix,represent(operands)) == represent(run(count,matrix,operands)) : Gemm.PointOperands}:
  match count:
    case 0n: ArrayLaws.indexed_fold_refines_model(~Gemm.PointOperands,~Operands,~Gemm.PointMatrix,~Gemm.point_job,~job,~represent,~job_matches,0n,0,matrix,operands)
    case 1n: job_matches(0,matrix,operands)
    case 2n+ +rest: ArrayLaws.indexed_fold_refines_model(~Gemm.PointOperands,~Operands,~Gemm.PointMatrix,~Gemm.point_job,~job,~represent,~job_matches,2n+rest,0,matrix,operands)

def run_tail(count: Nat,matrix: Gemm.PointMatrix,operands: Operands) -> Operands:
  Traversal.indexed_fold(~Operands,~Gemm.PointMatrix,~job_with(~Gemm.panel_times,~PointLaws.tail_reduced),count,0,matrix,operands)

def run_tail_matches(+count: Nat,+matrix: Gemm.PointMatrix,+operands: Operands) ->
  {Gemm.point_run_indexed(~Gemm.point_job_with(~Gemm.panel_bias,~Gemm.panel_times,~Gemm.finish_inner),count,matrix,represent(operands)) ==
    represent(run_tail(count,matrix,operands)) : Gemm.PointOperands}:
  match count:
    case 0n: {==}
    case 1n: job_tail_matches(0,matrix,operands)
    case 2n+ +rest:
      ArrayLaws.indexed_fold_refines_model(~Gemm.PointOperands,~Operands,~Gemm.PointMatrix,~Gemm.point_job_with(~Gemm.panel_bias,~Gemm.panel_times,~Gemm.finish_inner),
        ~job_with(~Gemm.panel_times,~PointLaws.tail_reduced),~represent,~job_tail_matches,2n+rest,0,matrix,operands)

# ---- The writes of a run ----

def kept(index: U32,matrix: Gemm.PointMatrix,readers: Readers) -> Readers:
  readers

def step_writes_by(~accumulate: Accumulate(),index: U32,matrix: Gemm.PointMatrix,readers: Readers) -> List<&2,Writes.Assignment<Tile.Tile16>>:
  +matrix = matrix
  place_writes_by(~accumulate,matrix,Gemm.point_place(matrix,index),readers)

def step_writes(index: U32,matrix: Gemm.PointMatrix,readers: Readers) -> List<&2,Writes.Assignment<Tile.Tile16>>:
  step_writes_by(~job_accumulators_with(~Gemm.row_times,~whole_inner),index,matrix,readers)

def staged_tree(operands: Operands) -> Tree.Tree<Tile.Tile16>:
  Operands{bias,cells,pairs,staged} = operands
  Writes.evaluate(Tile.Tile16,staged)

# Jobs first, first + 1, ... (count of them).
def all_writes_by(~accumulate: Accumulate(),count: Nat,+first: U32,+matrix: Gemm.PointMatrix,+readers: Readers) -> List<&2,Writes.Assignment<Tile.Tile16>>:
  Fold.all(~Tile.Tile16,~Gemm.PointMatrix,~Readers,~step_writes_by(~accumulate),~kept,count,first,matrix,readers)

def all_writes(count: Nat,+first: U32,+matrix: Gemm.PointMatrix,+readers: Readers) -> List<&2,Writes.Assignment<Tile.Tile16>>:
  all_writes_by(~job_accumulators_with(~Gemm.row_times,~whole_inner),count,first,matrix,readers)

def readers_kept(+index: U32,+matrix: Gemm.PointMatrix,+operands: Operands) -> {readers_of(job(index,matrix,operands)) == kept(index,matrix,readers_of(operands)) : Readers}:
  Operands{+bias,+cells,+pairs,+staged} = operands
  {==}

def staged_applied(+index: U32,+matrix: Gemm.PointMatrix,+operands: Operands) ->
  {staged_tree(job(index,matrix,operands)) == Writes.pure(Tile.Tile16,step_writes(index,matrix,readers_of(operands)),staged_tree(operands)) : Tree.Tree<Tile.Tile16>}:
  Operands{+bias,+cells,+pairs,+staged} = operands
  {==}

# A run applies the writes of its jobs in order to the staged tree.
def run_writes(+count: Nat,+matrix: Gemm.PointMatrix,+operands: Operands) ->
  {staged_tree(run(count,matrix,operands)) == Writes.pure(Tile.Tile16,all_writes(count,0,matrix,readers_of(operands)),staged_tree(operands)) : Tree.Tree<Tile.Tile16>}:
  Fold.output(~Operands,~Gemm.PointMatrix,~Tile.Tile16,~Readers,~job,~readers_of,~staged_tree,
    ~step_writes_by(~job_accumulators_with(~Gemm.row_times,~whole_inner)),~kept,~readers_kept,~staged_applied,count,0,matrix,operands)

def step_writes_with(~times: F32 -> F32 -> F32,~finish: PostInner(),index: U32,matrix: Gemm.PointMatrix,readers: Readers) -> List<&2,Writes.Assignment<Tile.Tile16>>:
  step_writes_by(~job_accumulators_with(~times,~finish),index,matrix,readers)

def readers_kept_with(~times: F32 -> F32 -> F32,~finish: PostInner(),+index: U32,+matrix: Gemm.PointMatrix,+operands: Operands) ->
  {readers_of(job_with(~times,~finish,index,matrix,operands)) == kept(index,matrix,readers_of(operands)) : Readers}:
  Operands{+bias,+cells,+pairs,+staged} = operands
  {==}

def staged_applied_with(~times: F32 -> F32 -> F32,~finish: PostInner(),+index: U32,+matrix: Gemm.PointMatrix,+operands: Operands) ->
  {staged_tree(job_with(~times,~finish,index,matrix,operands)) ==
    Writes.pure(Tile.Tile16,step_writes_with(~times,~finish,index,matrix,readers_of(operands)),staged_tree(operands)) : Tree.Tree<Tile.Tile16>}:
  Operands{+bias,+cells,+pairs,+staged} = operands
  {==}

def tail_writes(count: Nat,+first: U32,+matrix: Gemm.PointMatrix,+readers: Readers) -> List<&2,Writes.Assignment<Tile.Tile16>>:
  all_writes_by(~job_accumulators_with(~Gemm.panel_times,~PointLaws.tail_reduced),count,first,matrix,readers)

def run_tail_writes(+count: Nat,+matrix: Gemm.PointMatrix,+operands: Operands) ->
  {staged_tree(run_tail(count,matrix,operands)) == Writes.pure(Tile.Tile16,tail_writes(count,0,matrix,readers_of(operands)),staged_tree(operands)) : Tree.Tree<Tile.Tile16>}:
  Fold.output(~Operands,~Gemm.PointMatrix,~Tile.Tile16,~Readers,
    ~job_with(~Gemm.panel_times,~PointLaws.tail_reduced),~readers_of,~staged_tree,~step_writes_by(~job_accumulators_with(~Gemm.panel_times,~PointLaws.tail_reduced)),~kept,
    ~readers_kept_with(~Gemm.panel_times,~PointLaws.tail_reduced),~staged_applied_with(~Gemm.panel_times,~PointLaws.tail_reduced),count,0,matrix,operands)
