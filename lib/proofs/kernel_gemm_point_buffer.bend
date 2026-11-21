# The buffers a pointwise product keeps hold the contents of its model run
# over the contents of the buffers it was given.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_gemm_buffer.bend as GemmBuffer
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ../traversal_array.bend as Traversal
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./write_program.bend as Writes
import ./kernel_gemm_point_jobs.bend as PointJobs
import ./kernel_gemm_certificate.bend as GemmCertificate
import ./kernel_gemm_read_into.bend as ReadInto
import ./traversal_update_readback.bend as Updates
import ./traversal_array_refinement.bend as ArrayLaws
import ./storage_shape.bend as Shape
import ./nat_order.bend as Order
import ./nat_extrema.bend as NatExtrema
import ./row_major_index.bend as RowMajor
import ./nat_division_order.bend as DivisionOrder
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_division.bend as Division32
import ./u32_extrema.bend as U32Extrema

# The contents of the four operands of a pointwise product.
type PointContents is Data:
  PointContents{bias: Buffer.Content<Tile.Tile8>,cells: Buffer.Content<Tile.Tile8>,pairs: Buffer.Content<Tile.Tile16>,staged: Buffer.Content<Tile.Tile16>}

def kept_contents(kept: GemmBuffer.PointKept) -> PointContents:
  GemmBuffer.PointKept{bias,cells,pairs,staged} = kept
  PointContents{Buffer.contents(~Tile.Tile8,bias),Buffer.contents(~Tile.Tile8,cells),Buffer.contents(~Tile.Tile16,pairs),Buffer.contents(~Tile.Tile16,staged)}

# The lengths of the four operands.
type PointLengths is Data:
  PointLengths{bias: Nat,cells: Nat,pairs: Nat,staged: Nat}

def operand_contents(operands: PointJobs.Operands,lengths: PointLengths) -> PointContents:
  PointJobs.Operands{bias,cells,pairs,staged} = operands
  PointLengths{bias_length,cell_length,pair_length,staged_length} = lengths
  PointContents{Buffer.Content{bias,bias_length},Buffer.Content{cells,cell_length},Buffer.Content{pairs,pair_length},Buffer.Content{Writes.evaluate(Tile.Tile16,staged),staged_length}}

def array_contents(operands: Gemm.PointOperands,lengths: PointLengths) -> PointContents:
  Gemm.PointOperands{bias,cells,pairs,staged} = operands
  PointLengths{bias_length,cell_length,pair_length,staged_length} = lengths
  PointContents{Buffer.Content{Tree.reflect(Tile.Tile8,bias),bias_length},Buffer.Content{Tree.reflect(Tile.Tile8,cells),cell_length},
    Buffer.Content{Tree.reflect(Tile.Tile16,pairs),pair_length},Buffer.Content{Tree.reflect(Tile.Tile16,staged),staged_length}}

# The buffers made from operands hold the contents of the operands.
def kept_array_contents(state: Gemm.PointOperands,+bias_length: Nat,+cell_length: Nat,+pair_length: Nat,+length: Nat,certificates: GemmCertificate.PointCertified<state>) ->
  {kept_contents(GemmBuffer.points_kept(state,bias_length,cell_length,pair_length,length,certificates)) == array_contents(state,PointLengths{bias_length,cell_length,pair_length,length}) : PointContents}:
  Gemm.PointOperands{bias,cells,pairs,staged} = state
  GemmCertificate.PointCertified{bias_certificate,cell_certificate,pair_certificate,staged_certificate} = certificates
  {==}

# Packed model operands reflect back to their trees.
def represented_contents(+operands: PointJobs.Operands,+lengths: PointLengths) ->
  {array_contents(PointJobs.represent(operands),lengths) == operand_contents(operands,lengths) : PointContents}:
  PointJobs.Operands{+bias,+cells,+pairs,+staged} = operands
  PointLengths{+bias_length,+cell_length,+pair_length,+staged_length} = lengths
  +tree = Writes.evaluate(Tile.Tile16,staged)
  %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,bias)),bias,StorageLaws.reflect_packed(Tile.Tile8,bias)) :
    {PointContents{Buffer.Content{_,bias_length},Buffer.Content{Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,cells)),cell_length},Buffer.Content{Tree.reflect(Tile.Tile16,Tree.pack(Tile.Tile16,pairs)),pair_length},
      Buffer.Content{Tree.reflect(Tile.Tile16,Tree.pack(Tile.Tile16,tree)),staged_length}} == operand_contents(PointJobs.Operands{bias,cells,pairs,staged},PointLengths{bias_length,cell_length,pair_length,staged_length}) : PointContents}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,cells)),cells,StorageLaws.reflect_packed(Tile.Tile8,cells)) :
    {PointContents{Buffer.Content{bias,bias_length},Buffer.Content{_,cell_length},Buffer.Content{Tree.reflect(Tile.Tile16,Tree.pack(Tile.Tile16,pairs)),pair_length},
      Buffer.Content{Tree.reflect(Tile.Tile16,Tree.pack(Tile.Tile16,tree)),staged_length}} == operand_contents(PointJobs.Operands{bias,cells,pairs,staged},PointLengths{bias_length,cell_length,pair_length,staged_length}) : PointContents}
  %Equal.sym(Tree.Tree<Tile.Tile16>,Tree.reflect(Tile.Tile16,Tree.pack(Tile.Tile16,pairs)),pairs,StorageLaws.reflect_packed(Tile.Tile16,pairs)) :
    {PointContents{Buffer.Content{bias,bias_length},Buffer.Content{cells,cell_length},Buffer.Content{_,pair_length},
      Buffer.Content{Tree.reflect(Tile.Tile16,Tree.pack(Tile.Tile16,tree)),staged_length}} == operand_contents(PointJobs.Operands{bias,cells,pairs,staged},PointLengths{bias_length,cell_length,pair_length,staged_length}) : PointContents}
  %Equal.sym(Tree.Tree<Tile.Tile16>,Tree.reflect(Tile.Tile16,Tree.pack(Tile.Tile16,tree)),tree,StorageLaws.reflect_packed(Tile.Tile16,tree)) :
    {PointContents{Buffer.Content{bias,bias_length},Buffer.Content{cells,cell_length},Buffer.Content{pairs,pair_length},
      Buffer.Content{_,staged_length}} == operand_contents(PointJobs.Operands{bias,cells,pairs,staged},PointLengths{bias_length,cell_length,pair_length,staged_length}) : PointContents}
  {==}

# The operands of buffer models, the staged buffer as a program that starts
# from its tree, and their lengths.
def model_operands(input: Buffer.Model<Tile.Tile8>,pairs: Buffer.Model<Tile.Tile16>,bias: Buffer.Model<Tile.Tile8>,staged: Buffer.Model<Tile.Tile16>) -> PointJobs.Operands:
  PointJobs.Operands{Buffer.tree(~Tile.Tile8,bias),Buffer.tree(~Tile.Tile8,input),Buffer.tree(~Tile.Tile16,pairs),Writes.Initial{Buffer.tree(~Tile.Tile16,staged)}}

def model_lengths(input: Buffer.Model<Tile.Tile8>,pairs: Buffer.Model<Tile.Tile16>,bias: Buffer.Model<Tile.Tile8>,staged: Buffer.Model<Tile.Tile16>) -> PointLengths:
  PointLengths{Buffer.length(~Tile.Tile8,bias),Buffer.length(~Tile.Tile8,input),Buffer.length(~Tile.Tile16,pairs),Buffer.length(~Tile.Tile16,staged)}

# The buffers a pointwise product keeps hold the contents of the model run
# of its jobs over the contents of its buffers.
def keeping_contents_with(~job: U32 -> Gemm.PointMatrix -> Gemm.PointOperands -> Gemm.PointOperands,~keeps: GemmCertificate.PointJob(~job),
  ~run: Nat -> Gemm.PointMatrix -> PointJobs.Operands -> PointJobs.Operands,
  ~matched: @+count: Nat -> @+matrix: Gemm.PointMatrix -> @+operands: PointJobs.Operands ->
    {Gemm.point_run_indexed(~job,count,matrix,PointJobs.represent(operands)) == PointJobs.represent(run(count,matrix,operands)) : Gemm.PointOperands},
  +matrix: Gemm.PointMatrix,+input: Buffer.Model<Tile.Tile8>,+pairs: Buffer.Model<Tile.Tile16>,+bias: Buffer.Model<Tile.Tile8>,+staged: Buffer.Model<Tile.Tile16>) ->
  {kept_contents(GemmBuffer.point_keeping_with(~job,~keeps,matrix,Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile16,pairs),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,staged))) ==
    operand_contents(run(U32.to_nat(Gemm.point_count(matrix)),matrix,model_operands(input,pairs,bias,staged)),model_lengths(input,pairs,bias,staged)) : PointContents}:
  Buffer.Model{+input_tree,+input_length,+input_certificate} = input
  Buffer.Model{+pair_tree,+pair_length,+pair_certificate} = pairs
  Buffer.Model{+bias_tree,+bias_length,+bias_certificate} = bias
  Buffer.Model{+staged_tree,+staged_length,+staged_certificate} = staged
  +count = U32.to_nat(Gemm.point_count(matrix))
  +operands = model_operands(Buffer.Model{input_tree,input_length,input_certificate},Buffer.Model{pair_tree,pair_length,pair_certificate},
    Buffer.Model{bias_tree,bias_length,bias_certificate},Buffer.Model{staged_tree,staged_length,staged_certificate})
  +lengths = {PointLengths{bias_length,input_length,pair_length,staged_length} : PointLengths}
  Equal.trans(PointContents,kept_contents(GemmBuffer.points_kept(Gemm.point_run_indexed(~job,count,matrix,PointJobs.represent(operands)),bias_length,input_length,pair_length,staged_length,
      GemmCertificate.point_certificates_with(~job,~keeps,count,matrix,Tree.pack(Tile.Tile8,bias_tree),Tree.pack(Tile.Tile8,input_tree),Tree.pack(Tile.Tile16,pair_tree),Tree.pack(Tile.Tile16,staged_tree),
        bias_certificate,input_certificate,pair_certificate,staged_certificate))),
    array_contents(Gemm.point_run_indexed(~job,count,matrix,PointJobs.represent(operands)),lengths),operand_contents(run(count,matrix,operands),lengths),
    kept_array_contents(Gemm.point_run_indexed(~job,count,matrix,PointJobs.represent(operands)),bias_length,input_length,pair_length,staged_length,
      GemmCertificate.point_certificates_with(~job,~keeps,count,matrix,Tree.pack(Tile.Tile8,bias_tree),Tree.pack(Tile.Tile8,input_tree),Tree.pack(Tile.Tile16,pair_tree),Tree.pack(Tile.Tile16,staged_tree),
        bias_certificate,input_certificate,pair_certificate,staged_certificate)),
    Equal.trans(PointContents,array_contents(Gemm.point_run_indexed(~job,count,matrix,PointJobs.represent(operands)),lengths),array_contents(PointJobs.represent(run(count,matrix,operands)),lengths),
      operand_contents(run(count,matrix,operands),lengths),
      Equal.cong(Gemm.PointOperands,PointContents,result => array_contents(result,lengths),Gemm.point_run_indexed(~job,count,matrix,PointJobs.represent(operands)),
        PointJobs.represent(run(count,matrix,operands)),matched(count,matrix,operands)),
      represented_contents(run(count,matrix,operands),lengths)))

def keeping_contents(+matrix: Gemm.PointMatrix,+input: Buffer.Model<Tile.Tile8>,+pairs: Buffer.Model<Tile.Tile16>,+bias: Buffer.Model<Tile.Tile8>,+staged: Buffer.Model<Tile.Tile16>) ->
  {kept_contents(GemmBuffer.point_keeping(matrix,Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile16,pairs),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,staged))) ==
    operand_contents(PointJobs.run(U32.to_nat(Gemm.point_count(matrix)),matrix,model_operands(input,pairs,bias,staged)),model_lengths(input,pairs,bias,staged)) : PointContents}:
  keeping_contents_with(~Gemm.point_job,~GemmCertificate.point_job_whole,~PointJobs.run,~PointJobs.run_matches,matrix,input,pairs,bias,staged)

def keeping_tail_contents(+matrix: Gemm.PointMatrix,+input: Buffer.Model<Tile.Tile8>,+pairs: Buffer.Model<Tile.Tile16>,+bias: Buffer.Model<Tile.Tile8>,+staged: Buffer.Model<Tile.Tile16>) ->
  {kept_contents(GemmBuffer.point_keeping_tail(matrix,Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile16,pairs),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,staged))) ==
    operand_contents(PointJobs.run_tail(U32.to_nat(Gemm.point_count(matrix)),matrix,model_operands(input,pairs,bias,staged)),model_lengths(input,pairs,bias,staged)) : PointContents}:
  keeping_contents_with(~Gemm.point_job_with(~Gemm.panel_bias,~Gemm.panel_times,~Gemm.finish_inner),
    ~GemmCertificate.point_job(~Gemm.panel_bias,~Gemm.panel_times,~Gemm.finish_inner,~GemmCertificate.finish_inner),
    ~PointJobs.run_tail,~PointJobs.run_tail_matches,matrix,input,pairs,bias,staged)

# The trees a run reads stay as they were.
def readers_fold(+count: Nat,+index: U32,+matrix: Gemm.PointMatrix,+operands: PointJobs.Operands) ->
  {PointJobs.readers_of(Traversal.indexed_fold(~PointJobs.Operands,~Gemm.PointMatrix,~PointJobs.job,count,index,matrix,operands)) == PointJobs.readers_of(operands) : PointJobs.Readers}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      Equal.trans(PointJobs.Readers,PointJobs.readers_of(Traversal.indexed_fold(~PointJobs.Operands,~Gemm.PointMatrix,~PointJobs.job,rest,(index + 1 : U32),matrix,PointJobs.job(index,matrix,operands))),
        PointJobs.readers_of(PointJobs.job(index,matrix,operands)),PointJobs.readers_of(operands),
        readers_fold(rest,(index + 1 : U32),matrix,PointJobs.job(index,matrix,operands)),PointJobs.readers_kept(index,matrix,operands))

def run_readers(+count: Nat,+matrix: Gemm.PointMatrix,+operands: PointJobs.Operands) ->
  {PointJobs.readers_of(PointJobs.run(count,matrix,operands)) == PointJobs.readers_of(operands) : PointJobs.Readers}:
  readers_fold(count,0,matrix,operands)

# The models of the buffers a pointwise product keeps.
type PointKeptModel is Data:
  PointKeptModel{bias: Buffer.Model<Tile.Tile8>,cells: Buffer.Model<Tile.Tile8>,pairs: Buffer.Model<Tile.Tile16>,staged: Buffer.Model<Tile.Tile16>}

def pack_point_kept(model: PointKeptModel) -> GemmBuffer.PointKept:
  PointKeptModel{bias,cells,pairs,staged} = model
  GemmBuffer.PointKept{Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile8,cells),Buffer.pack(~Tile.Tile16,pairs),Buffer.pack(~Tile.Tile16,staged)}

def remodel_point_kept(kept: GemmBuffer.PointKept) -> PointKeptModel:
  GemmBuffer.PointKept{bias,cells,pairs,staged} = kept
  PointKeptModel{Buffer.remodel(~Tile.Tile8,bias),Buffer.remodel(~Tile.Tile8,cells),Buffer.remodel(~Tile.Tile16,pairs),Buffer.remodel(~Tile.Tile16,staged)}

def point_kept_remodeled(kept: GemmBuffer.PointKept) -> {pack_point_kept(remodel_point_kept(kept)) == kept : GemmBuffer.PointKept}:
  GemmBuffer.PointKept{bias,cells,pairs,staged} = kept
  %Equal.sym(Storage.Buffer<Tile.Tile8>,Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,bias)),bias,Buffer.remodeled(~Tile.Tile8,bias)) : {GemmBuffer.PointKept{_,Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,cells)),Buffer.pack(~Tile.Tile16,Buffer.remodel(~Tile.Tile16,pairs)),
      Buffer.pack(~Tile.Tile16,Buffer.remodel(~Tile.Tile16,staged))} == GemmBuffer.PointKept{bias,cells,pairs,staged} : GemmBuffer.PointKept}
  %Equal.sym(Storage.Buffer<Tile.Tile8>,Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,cells)),cells,Buffer.remodeled(~Tile.Tile8,cells)) : {GemmBuffer.PointKept{bias,_,Buffer.pack(~Tile.Tile16,Buffer.remodel(~Tile.Tile16,pairs)),
      Buffer.pack(~Tile.Tile16,Buffer.remodel(~Tile.Tile16,staged))} == GemmBuffer.PointKept{bias,cells,pairs,staged} : GemmBuffer.PointKept}
  %Equal.sym(Storage.Buffer<Tile.Tile16>,Buffer.pack(~Tile.Tile16,Buffer.remodel(~Tile.Tile16,pairs)),pairs,Buffer.remodeled(~Tile.Tile16,pairs)) : {GemmBuffer.PointKept{bias,cells,_,Buffer.pack(~Tile.Tile16,Buffer.remodel(~Tile.Tile16,staged))} == GemmBuffer.PointKept{bias,cells,pairs,staged} : GemmBuffer.PointKept}
  %Equal.sym(Storage.Buffer<Tile.Tile16>,Buffer.pack(~Tile.Tile16,Buffer.remodel(~Tile.Tile16,staged)),staged,Buffer.remodeled(~Tile.Tile16,staged)) : {GemmBuffer.PointKept{bias,cells,pairs,_} == GemmBuffer.PointKept{bias,cells,pairs,staged} : GemmBuffer.PointKept}
  {==}

def model_contents(model: PointKeptModel) -> PointContents:
  PointKeptModel{bias,cells,pairs,staged} = model
  PointContents{Buffer.content(~Tile.Tile8,bias),Buffer.content(~Tile.Tile8,cells),Buffer.content(~Tile.Tile16,pairs),Buffer.content(~Tile.Tile16,staged)}

# The models of buffers hold what the buffers hold.
def remodeled_contents(kept: GemmBuffer.PointKept) -> {model_contents(remodel_point_kept(kept)) == kept_contents(kept) : PointContents}:
  GemmBuffer.PointKept{bias,cells,pairs,staged} = kept
  %Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,bias)),Buffer.contents(~Tile.Tile8,bias),Buffer.observed_content(~Tile.Tile8,bias,Buffer.observe(~Tile.Tile8,bias))) : {PointContents{_,Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,cells)),Buffer.content(~Tile.Tile16,Buffer.remodel(~Tile.Tile16,pairs)),
      Buffer.content(~Tile.Tile16,Buffer.remodel(~Tile.Tile16,staged))} == kept_contents(GemmBuffer.PointKept{bias,cells,pairs,staged}) : PointContents}
  %Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,cells)),Buffer.contents(~Tile.Tile8,cells),Buffer.observed_content(~Tile.Tile8,cells,Buffer.observe(~Tile.Tile8,cells))) : {PointContents{Buffer.contents(~Tile.Tile8,bias),_,Buffer.content(~Tile.Tile16,Buffer.remodel(~Tile.Tile16,pairs)),
      Buffer.content(~Tile.Tile16,Buffer.remodel(~Tile.Tile16,staged))} == kept_contents(GemmBuffer.PointKept{bias,cells,pairs,staged}) : PointContents}
  %Equal.sym(Buffer.Content<Tile.Tile16>,Buffer.content(~Tile.Tile16,Buffer.remodel(~Tile.Tile16,pairs)),Buffer.contents(~Tile.Tile16,pairs),Buffer.observed_content(~Tile.Tile16,pairs,Buffer.observe(~Tile.Tile16,pairs))) : {PointContents{Buffer.contents(~Tile.Tile8,bias),Buffer.contents(~Tile.Tile8,cells),_,
      Buffer.content(~Tile.Tile16,Buffer.remodel(~Tile.Tile16,staged))} == kept_contents(GemmBuffer.PointKept{bias,cells,pairs,staged}) : PointContents}
  %Equal.sym(Buffer.Content<Tile.Tile16>,Buffer.content(~Tile.Tile16,Buffer.remodel(~Tile.Tile16,staged)),Buffer.contents(~Tile.Tile16,staged),Buffer.observed_content(~Tile.Tile16,staged,Buffer.observe(~Tile.Tile16,staged))) : {PointContents{Buffer.contents(~Tile.Tile8,bias),Buffer.contents(~Tile.Tile8,cells),Buffer.contents(~Tile.Tile16,pairs),_} ==
      kept_contents(GemmBuffer.PointKept{bias,cells,pairs,staged}) : PointContents}
  {==}

# ---- The pairs of the weights ----

# Pair index / inner of the weights at tap index % inner: the tiles of its two
# groups, the last group again for a missing upper one.
def pair_tiles(+tree: Tree.Tree<Tile.Tile8>,+plan: Gemm.PairPlan,+index: U32) -> Tile.Tile16:
  Gemm.PairPlan{+groups,+inner} = plan
  +pair = (index / inner : U32)
  +tap = (index % inner : U32)
  Tile.Tiles{Tree.read(Tile.Tile8,tree,(pair * 2 * inner + tap : U32)),Tree.read(Tile.Tile8,tree,(U32.min((pair * 2 + 1 : U32),(groups - 1 : U32)) * inner + tap : U32))}

def pair_sample(tree: Tree.Tree<Tile.Tile8>,plan: Gemm.PairPlan,index: U32) -> Tile.Tile16:
  pair_tiles(tree,plan,index)

def paired_tiles(+reader: Tree.Tree<Tile.Tile8>,+low: U32,+high: U32) ->
  {Gemm.pair_lower(high,Array.get(Tile.Tile8,Tree.pack(Tile.Tile8,reader),low)) ==
    (Tree.pack(Tile.Tile8,reader),Tile.Tiles{Tree.read(Tile.Tile8,reader,low),Tree.read(Tile.Tile8,reader,high)}) : Array<Tile.Tile8> & Tile.Tile16}:
  -array = Tree.pack(Tile.Tile8,reader)
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,array,low),(array,Tree.read(Tile.Tile8,reader,low)),StorageLaws.read(Tile.Tile8,reader,low)) :
    {Gemm.pair_lower(high,_) == (array,Tile.Tiles{Tree.read(Tile.Tile8,reader,low),Tree.read(Tile.Tile8,reader,high)}) : Array<Tile.Tile8> & Tile.Tile16}
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,array,high),(array,Tree.read(Tile.Tile8,reader,high)),StorageLaws.read(Tile.Tile8,reader,high)) :
    {Gemm.pair_upper(Tree.read(Tile.Tile8,reader,low),_) == (array,Tile.Tiles{Tree.read(Tile.Tile8,reader,low),Tree.read(Tile.Tile8,reader,high)}) : Array<Tile.Tile8> & Tile.Tile16}
  {==}

def pair_matches(+reader: Tree.Tree<Tile.Tile8>,+plan: Gemm.PairPlan,+position: U32) ->
  {Gemm.pair_read(ArrayLaws.pack_tree(~Tile.Tile8,reader),plan,position) == (ArrayLaws.pack_tree(~Tile.Tile8,reader),pair_sample(reader,plan,position)) : Array<Tile.Tile8> & Tile.Tile16}:
  Gemm.PairPlan{+groups,+inner} = plan
  +pair = (position / inner : U32)
  +tap = (position % inner : U32)
  paired_tiles(reader,(pair * 2 * inner + tap : U32),(U32.min((pair * 2 + 1 : U32),(groups - 1 : U32)) * inner + tap : U32))

def pair_identity(+plan: Gemm.PairPlan,+position: U32) -> {Gemm.pair_placed(plan,position) == position : U32}:
  {==}

def pair_count(+channels: U32,+inner: U32) -> Nat:
  U32.to_nat((Gemm.pairs_of(channels) * inner : U32))

def pair_loop(+channels: U32,+inner: U32,+weights: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>) -> Tree.Tree<Tile.Tile16>:
  ReadInto.read_loop(~Tile.Tile8,~Tile.Tile16,~Gemm.PairPlan,~pair_sample,~Gemm.pair_placed,pair_count(channels,inner),Gemm.PairPlan{Gemm.blocks(channels),inner},weights,pairs)

# Building the pairs gives back the weights and pairs read from them.
def pairs_contents(+channels: U32,+inner: U32,+weights: Buffer.Model<Tile.Tile8>,+pairs: Buffer.Model<Tile.Tile16>) ->
  {ReadInto.buffers_contents(~Tile.Tile8,~Tile.Tile16,GemmBuffer.point_pairs(channels,inner,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile16,pairs))) ==
    (Buffer.content(~Tile.Tile8,weights),Buffer.Content{pair_loop(channels,inner,Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~Tile.Tile16,pairs)),Buffer.length(~Tile.Tile16,pairs)}) :
    Buffer.Content<Tile.Tile8> & Buffer.Content<Tile.Tile16>}:
  ReadInto.read_model(~Tile.Tile8,~Tile.Tile16,~Gemm.PairPlan,~Gemm.pair_read,~GemmCertificate.pair_read,~pair_sample,~pair_matches,~Gemm.pair_placed,
    pair_count(channels,inner),Gemm.PairPlan{Gemm.blocks(channels),inner},weights,pairs)

# Each pair read holds the tiles of its groups.
def pair_at(+channels: U32,+inner: U32,+weights: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+depth: Nat,+balanced: Shape.Balanced(Tile.Tile16,depth,pairs),
  +limit: {Nat.is_le(depth,31n) == True{} : Bool},+capacity_ok: {U32.is_le(Tree.capacity(Tile.Tile16,pairs),2147483648) == True{} : Bool},
  +room: {Nat.is_le(pair_count(channels,inner),U32.to_nat(Tree.capacity(Tile.Tile16,pairs))) == True{} : Bool},
  +index: Nat,+inside: {Nat.is_lt(index,pair_count(channels,inner)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile16,pair_loop(channels,inner,weights,pairs),U32.from_nat(index)) == pair_tiles(weights,Gemm.PairPlan{Gemm.blocks(channels),inner},U32.from_nat(index)) : Tile.Tile16}:
  Updates.identity_read(~Tree.Tree<Tile.Tile8>,~Tile.Tile16,~Tile.Tile16,~Gemm.PairPlan,~pair_sample,~Gemm.replaced(~Tile.Tile16),~Gemm.pair_placed,~pair_identity,
    pair_count(channels,inner),weights,Gemm.PairPlan{Gemm.blocks(channels),inner},pairs,depth,balanced,limit,capacity_ok,room,index,inside)

# The tile of a band at source + index.
def tile_sample(tree: Tree.Tree<Tile.Tile8>,plan: Gemm.TilePlan,index: U32) -> Tile.Tile8:
  Gemm.TilePlan{+source,target} = plan
  Tree.read(Tile.Tile8,tree,(source + index : U32))

def tile_matches(+reader: Tree.Tree<Tile.Tile8>,+plan: Gemm.TilePlan,+position: U32) ->
  {Gemm.tile_read(ArrayLaws.pack_tree(~Tile.Tile8,reader),plan,position) == (ArrayLaws.pack_tree(~Tile.Tile8,reader),tile_sample(reader,plan,position)) :
    Array<Tile.Tile8> & Tile.Tile8}:
  Gemm.TilePlan{+source,+target} = plan
  StorageLaws.read(Tile.Tile8,reader,(source + position : U32))

# A panel pair reads two output groups at one inner index.
def panel_sample(tree: Tree.Tree<Tile.Tile8>,plan: Gemm.PanelPlan,index: U32) -> Tile.Tile16:
  Gemm.PanelPlan{+groups,+inner,+base} = plan
  +index = index
  +tree = tree
  +lower = (2 * (index / inner) : U32)
  +row = (base + (index % inner) * groups : U32)
  Tile.Tiles{Tree.read(Tile.Tile8,tree,(row + lower : U32)),Tree.read(Tile.Tile8,tree,(row + U32.min((lower + 1 : U32),(groups - 1 : U32)) : U32))}

def panel_matches(+reader: Tree.Tree<Tile.Tile8>,+plan: Gemm.PanelPlan,+index: U32) ->
  {Gemm.panel_pair_read(ArrayLaws.pack_tree(~Tile.Tile8,reader),plan,index) ==
    (ArrayLaws.pack_tree(~Tile.Tile8,reader),panel_sample(reader,plan,index)) : Array<Tile.Tile8> & Tile.Tile16}:
  Gemm.PanelPlan{+groups,+inner,+base} = plan
  +lower = (2 * (index / inner) : U32)
  +row = (base + (index % inner) * groups : U32)
  paired_tiles(reader,(row + lower : U32),(row + U32.min((lower + 1 : U32),(groups - 1 : U32)) : U32))

def panel_loop(+channels: U32,+inner: U32,+base: U32,+weights: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>) -> Tree.Tree<Tile.Tile16>:
  ReadInto.read_loop(~Tile.Tile8,~Tile.Tile16,~Gemm.PanelPlan,~panel_sample,~Gemm.panel_pair_placed,
    pair_count(channels,inner),Gemm.PanelPlan{Gemm.blocks(channels),inner,base},weights,pairs)

def panel_contents(+channels: U32,+inner: U32,+base: U32,+weights: Buffer.Model<Tile.Tile8>,+pairs: Buffer.Model<Tile.Tile16>) ->
  {ReadInto.buffers_contents(~Tile.Tile8,~Tile.Tile16,GemmBuffer.panel_pairs(channels,inner,base,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile16,pairs))) ==
    (Buffer.content(~Tile.Tile8,weights),Buffer.Content{panel_loop(channels,inner,base,Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~Tile.Tile16,pairs)),Buffer.length(~Tile.Tile16,pairs)}) :
    Buffer.Content<Tile.Tile8> & Buffer.Content<Tile.Tile16>}:
  ReadInto.read_model(~Tile.Tile8,~Tile.Tile16,~Gemm.PanelPlan,~Gemm.panel_pair_read,~GemmCertificate.panel_pair_read,~panel_sample,~panel_matches,~Gemm.panel_pair_placed,
    pair_count(channels,inner),Gemm.PanelPlan{Gemm.blocks(channels),inner,base},weights,pairs)

def panel_identity(+plan: Gemm.PanelPlan,+index: U32) -> {Gemm.panel_pair_placed(plan,index) == index : U32}:
  {==}

def panel_at(+channels: U32,+inner: U32,+base: U32,+weights: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+depth: Nat,+balanced: Shape.Balanced(Tile.Tile16,depth,pairs),
  +limit: {Nat.is_le(depth,31n) == True{} : Bool},+capacity_ok: {U32.is_le(Tree.capacity(Tile.Tile16,pairs),2147483648) == True{} : Bool},
  +room: {Nat.is_le(pair_count(channels,inner),U32.to_nat(Tree.capacity(Tile.Tile16,pairs))) == True{} : Bool},
  +index: Nat,+inside: {Nat.is_lt(index,pair_count(channels,inner)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile16,panel_loop(channels,inner,base,weights,pairs),U32.from_nat(index)) ==
    panel_sample(weights,Gemm.PanelPlan{Gemm.blocks(channels),inner,base},U32.from_nat(index)) : Tile.Tile16}:
  Updates.identity_read(~Tree.Tree<Tile.Tile8>,~Tile.Tile16,~Tile.Tile16,~Gemm.PanelPlan,~panel_sample,~Gemm.replaced(~Tile.Tile16),~Gemm.panel_pair_placed,~panel_identity,
    pair_count(channels,inner),weights,Gemm.PanelPlan{Gemm.blocks(channels),inner,base},pairs,depth,balanced,limit,capacity_ok,room,index,inside)

# ---- The cells of the staged pairs ----

# Cell index of the staged pairs: the half of pair (index / groups) * pairs +
# group / 2 that holds group index % groups, mapped.
def staged_tile(~Context: Data,~map: Context -> Tile.Tile8 -> Tile.Tile8,+tree: Tree.Tree<Tile.Tile16>,+plan: Gemm.StagedPlan<Context>,+index: U32) -> Tile.Tile8:
  Gemm.StagedPlan{+groups,+pairs,+context} = plan
  +group = (index % groups : U32)
  map(context,Gemm.staged_half(Tree.read(Tile.Tile16,tree,((index / groups) * pairs + group / 2 : U32)),U32.is_eq((group % 2 : U32),1)))

def staged_sample(~Context: Data,~map: Context -> Tile.Tile8 -> Tile.Tile8,tree: Tree.Tree<Tile.Tile16>,plan: Gemm.StagedPlan<Context>,index: U32) -> Tile.Tile8:
  staged_tile(~Context,~map,tree,plan,index)

def staged_matches(~Context: Data,~map: Context -> Tile.Tile8 -> Tile.Tile8,+reader: Tree.Tree<Tile.Tile16>,+plan: Gemm.StagedPlan<Context>,+position: U32) ->
  {Gemm.staged_read(~Context,~map,ArrayLaws.pack_tree(~Tile.Tile16,reader),plan,position) == (ArrayLaws.pack_tree(~Tile.Tile16,reader),staged_sample(~Context,~map,reader,plan,position)) :
    Array<Tile.Tile16> & Tile.Tile8}:
  Gemm.StagedPlan{+groups,+pairs,+context} = plan
  +group = (position % groups : U32)
  +address = ((position / groups) * pairs + group / 2 : U32)
  -array = Tree.pack(Tile.Tile16,reader)
  %Equal.sym(Array<Tile.Tile16> & Tile.Tile16,Array.get(Tile.Tile16,array,address),(array,Tree.read(Tile.Tile16,reader,address)),StorageLaws.read(Tile.Tile16,reader,address)) :
    {Gemm.staged_mapped(~Context,~map,context,U32.is_eq((group % 2 : U32),1),_) ==
      (array,map(context,Gemm.staged_half(Tree.read(Tile.Tile16,reader,address),U32.is_eq((group % 2 : U32),1)))) : Array<Tile.Tile16> & Tile.Tile8}
  {==}

def staged_identity(~Context: Data,+plan: Gemm.StagedPlan<Context>,+position: U32) -> {Gemm.staged_placed(~Context,plan,position) == position : U32}:
  {==}

def staged_loop(~Context: Data,~map: Context -> Tile.Tile8 -> Tile.Tile8,+channels: U32,+count: U32,+context: Context,+staged: Tree.Tree<Tile.Tile16>,+output: Tree.Tree<Tile.Tile8>) ->
  Tree.Tree<Tile.Tile8>:
  ReadInto.read_loop(~Tile.Tile16,~Tile.Tile8,~Gemm.StagedPlan<Context>,~staged_sample(~Context,~map),~Gemm.staged_placed(~Context),U32.to_nat(count),
    Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),context},staged,output)

# Unstaging gives back the staged pairs and the cells read from them.
def staged_contents(~Context: Data,~map: Context -> Tile.Tile8 -> Tile.Tile8,+channels: U32,+count: U32,+context: Context,+staged: Buffer.Model<Tile.Tile16>,+output: Buffer.Model<Tile.Tile8>) ->
  {ReadInto.buffers_contents(~Tile.Tile16,~Tile.Tile8,GemmBuffer.staged_cells(~Context,~map,channels,count,context,Buffer.pack(~Tile.Tile16,staged),Buffer.pack(~Tile.Tile8,output))) ==
    (Buffer.content(~Tile.Tile16,staged),Buffer.Content{staged_loop(~Context,~map,channels,count,context,Buffer.tree(~Tile.Tile16,staged),Buffer.tree(~Tile.Tile8,output)),
      Buffer.length(~Tile.Tile8,output)}) : Buffer.Content<Tile.Tile16> & Buffer.Content<Tile.Tile8>}:
  ReadInto.read_model(~Tile.Tile16,~Tile.Tile8,~Gemm.StagedPlan<Context>,~Gemm.staged_read(~Context,~map),~GemmCertificate.staged_read(~Context,~map),~staged_sample(~Context,~map),
    ~staged_matches(~Context,~map),~Gemm.staged_placed(~Context),U32.to_nat(count),Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),context},staged,output)

# Each cell unstaged holds the mapped half of its pair.
def staged_at(~Context: Data,~map: Context -> Tile.Tile8 -> Tile.Tile8,+channels: U32,+count: U32,+context: Context,+staged: Tree.Tree<Tile.Tile16>,+output: Tree.Tree<Tile.Tile8>,
  +depth: Nat,+balanced: Shape.Balanced(Tile.Tile8,depth,output),+limit: {Nat.is_le(depth,31n) == True{} : Bool},
  +capacity_ok: {U32.is_le(Tree.capacity(Tile.Tile8,output),2147483648) == True{} : Bool},+room: {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Tile.Tile8,output))) == True{} : Bool},
  +index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,staged_loop(~Context,~map,channels,count,context,staged,output),U32.from_nat(index)) ==
    staged_tile(~Context,~map,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),context},U32.from_nat(index)) : Tile.Tile8}:
  Updates.identity_read(~Tree.Tree<Tile.Tile16>,~Tile.Tile8,~Tile.Tile8,~Gemm.StagedPlan<Context>,~staged_sample(~Context,~map),~Gemm.replaced(~Tile.Tile8),~Gemm.staged_placed(~Context),
    ~staged_identity(~Context),U32.to_nat(count),staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),context},output,depth,balanced,limit,capacity_ok,room,index,inside)

# A fragment uses the target order to select the half of a staged pair.
def slice_sample(staged: Tree.Tree<Tile.Tile16>,slice: Gemm.StagedSlice,index: U32) -> Tile.Tile8:
  Gemm.StagedSlice{+groups,+positions,+first,+group_major} = slice
  staged_tile(~Unit,~Gemm.unmapped_tile,staged,Gemm.StagedPlan{groups,((groups + 1) / 2 : U32),Unit{}},
    Gemm.staged_slice_index(Gemm.StagedSlice{groups,positions,first,group_major},index))

def slice_matches(+reader: Tree.Tree<Tile.Tile16>,+slice: Gemm.StagedSlice,+index: U32) ->
  {Gemm.staged_slice_read(ArrayLaws.pack_tree(~Tile.Tile16,reader),slice,index) ==
    (ArrayLaws.pack_tree(~Tile.Tile16,reader),slice_sample(reader,slice,index)) : Array<Tile.Tile16> & Tile.Tile8}:
  Gemm.StagedSlice{+groups,+positions,+first,+group_major} = slice
  staged_matches(~Unit,~Gemm.unmapped_tile,reader,Gemm.StagedPlan{groups,((groups + 1) / 2 : U32),Unit{}},
    Gemm.staged_slice_index(Gemm.StagedSlice{groups,positions,first,group_major},index))

def slice_identity(+slice: Gemm.StagedSlice,+index: U32) -> {Gemm.staged_slice_destination(slice,index) == index : U32}:
  {==}

def slice_loop(+slice: Gemm.StagedSlice,+count: U32,+staged: Tree.Tree<Tile.Tile16>,+output: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  ReadInto.read_loop(~Tile.Tile16,~Tile.Tile8,~Gemm.StagedSlice,~slice_sample,~Gemm.staged_slice_destination,U32.to_nat(count),slice,staged,output)

def slice_contents(+slice: Gemm.StagedSlice,+count: U32,+staged: Buffer.Model<Tile.Tile16>,+output: Buffer.Model<Tile.Tile8>) ->
  {ReadInto.buffers_contents(~Tile.Tile16,~Tile.Tile8,GemmBuffer.staged_slice(slice,count,Buffer.pack(~Tile.Tile16,staged),Buffer.pack(~Tile.Tile8,output))) ==
    (Buffer.content(~Tile.Tile16,staged),Buffer.Content{slice_loop(slice,count,Buffer.tree(~Tile.Tile16,staged),Buffer.tree(~Tile.Tile8,output)),
      Buffer.length(~Tile.Tile8,output)}) : Buffer.Content<Tile.Tile16> & Buffer.Content<Tile.Tile8>}:
  ReadInto.read_model(~Tile.Tile16,~Tile.Tile8,~Gemm.StagedSlice,~Gemm.staged_slice_read,~GemmCertificate.staged_slice_read,~slice_sample,
    ~slice_matches,~Gemm.staged_slice_destination,U32.to_nat(count),slice,staged,output)

def slice_at(+slice: Gemm.StagedSlice,+count: U32,+staged: Tree.Tree<Tile.Tile16>,+output: Tree.Tree<Tile.Tile8>,
  +depth: Nat,+balanced: Shape.Balanced(Tile.Tile8,depth,output),+limit: {Nat.is_le(depth,31n) == True{} : Bool},
  +capacity_ok: {U32.is_le(Tree.capacity(Tile.Tile8,output),2147483648) == True{} : Bool},+room: {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Tile.Tile8,output))) == True{} : Bool},
  +index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,slice_loop(slice,count,staged,output),U32.from_nat(index)) == slice_sample(staged,slice,U32.from_nat(index)) : Tile.Tile8}:
  Updates.identity_read(~Tree.Tree<Tile.Tile16>,~Tile.Tile8,~Tile.Tile8,~Gemm.StagedSlice,~slice_sample,~Gemm.replaced(~Tile.Tile8),~Gemm.staged_slice_destination,
    ~slice_identity,U32.to_nat(count),staged,slice,output,depth,balanced,limit,capacity_ok,room,index,inside)

# ---- The address of a pair read ----

# The pair read at tap of pair pair reads the tiles at that tap of the groups
# 2 pair and 2 pair + 1, or the last group for a missing upper one.
def pair_address(+tree: Tree.Tree<Tile.Tile8>,+groups: U32,+inner: U32,+group_count: Nat,+inner_count: Nat,
  +groups_exact: {U32.to_nat(groups) == group_count : Nat},+inner_exact: {U32.to_nat(inner) == inner_count : Nat},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(group_count,inner_count),U32.to_nat(cap)) == True{} : Bool},
  +divisor_room: {Nat.is_le(Nat.double(inner_count),U32.to_nat(cap)) == True{} : Bool},
  +pair: Nat,+tap: Nat,+tap_inside: {Nat.is_lt(tap,inner_count) == True{} : Bool},
  +pair_inside: {Nat.is_le(1n+Nat.mul(pair,2n),group_count) == True{} : Bool}) ->
  {pair_tiles(tree,Gemm.PairPlan{groups,inner},U32.from_nat(Nat.add(Nat.mul(pair,inner_count),tap))) ==
    Tile.Tiles{Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(Nat.mul(Nat.mul(pair,2n),inner_count),tap))),
      Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(Nat.mul(Nat.min(1n+Nat.mul(pair,2n),Nat.sub(group_count,1n)),inner_count),tap)))} : Tile.Tile16}:
  +double = Nat.mul(pair,2n)
  +last = Nat.sub(group_count,1n)
  +upper = Nat.min(1n+double,last)
  +total = Nat.mul(group_count,inner_count)
  +index = Nat.add(Nat.mul(pair,inner_count),tap)
  +low = Nat.add(Nat.mul(double,inner_count),tap)
  +high = Nat.add(Nat.mul(upper,inner_count),tap)
  +positive = {Order.le_lt_trans(0n,tap,inner_count,Order.zero_le(tap),tap_inside) : {Nat.is_lt(0n,inner_count) == True{} : Bool}}
  +groups_positive = {Order.lt_le(0n,1n+double,group_count,{==},pair_inside) : {Nat.is_lt(0n,group_count) == True{} : Bool}}
  +groups_cap = {Order.transitive(group_count,total,U32.to_nat(cap),RowMajor.product_covers(group_count,inner_count,positive),room) :
    {Nat.is_le(group_count,U32.to_nat(cap)) == True{} : Bool}}
  +span_cap = {Order.transitive(Nat.mul(1n+double,inner_count),total,U32.to_nat(cap),RowMajor.mul_le(1n+double,group_count,inner_count,pair_inside),room) :
    {Nat.is_le(Nat.mul(1n+double,inner_count),U32.to_nat(cap)) == True{} : Bool}}
  +low_below = {%Order.commutative(tap,Nat.mul(double,inner_count)) : {Nat.is_lt(_,Nat.mul(1n+double,inner_count)) == True{} : Bool}
    Order.add_lt(tap,inner_count,Nat.mul(double,inner_count),tap_inside) : {Nat.is_lt(low,Nat.mul(1n+double,inner_count)) == True{} : Bool}}
  +low_cap = {Bounds.strict_le(low,U32.to_nat(cap),Order.lt_le(low,Nat.mul(1n+double,inner_count),U32.to_nat(cap),low_below,span_cap)) :
    {Nat.is_le(low,U32.to_nat(cap)) == True{} : Bool}}
  +index_below = {%Order.commutative(tap,Nat.mul(pair,inner_count)) : {Nat.is_lt(_,Nat.mul(1n+pair,inner_count)) == True{} : Bool}
    Order.add_lt(tap,inner_count,Nat.mul(pair,inner_count),tap_inside) : {Nat.is_lt(index,Nat.mul(1n+pair,inner_count)) == True{} : Bool}}
  +pair_le = {RowMajor.product_covers(pair,2n,{==}) : {Nat.is_le(1n+pair,1n+double) == True{} : Bool}}
  +index_cap = {Bounds.strict_le(index,U32.to_nat(cap),Order.lt_le(index,Nat.mul(1n+pair,inner_count),U32.to_nat(cap),index_below,
      Order.transitive(Nat.mul(1n+pair,inner_count),Nat.mul(1n+double,inner_count),U32.to_nat(cap),RowMajor.mul_le(1n+pair,1n+double,inner_count,pair_le),span_cap))) :
    {Nat.is_le(index,U32.to_nat(cap)) == True{} : Bool}}
  +scaled_cap = {Order.transitive(Nat.mul(double,inner_count),low,U32.to_nat(cap),Order.le_add(Nat.mul(double,inner_count),tap),low_cap) :
    {Nat.is_le(Nat.mul(double,inner_count),U32.to_nat(cap)) == True{} : Bool}}
  +double_cap = {Order.transitive(double,Nat.mul(double,inner_count),U32.to_nat(cap),RowMajor.product_covers(double,inner_count,positive),scaled_cap) :
    {Nat.is_le(double,U32.to_nat(cap)) == True{} : Bool}}
  +next_cap = {%Order.commutative(1n,double) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
    Order.transitive(1n+double,group_count,U32.to_nat(cap),pair_inside,groups_cap) : {Nat.is_le(Nat.add(double,1n),U32.to_nat(cap)) == True{} : Bool}}
  +upper_follow = {%Order.positive_has_last(group_count,groups_positive) : {Nat.is_le(1n+upper,_) == True{} : Bool}
    NatExtrema.minimum_right(1n+double,last) : {Nat.is_le(1n+upper,group_count) == True{} : Bool}}
  +high_below = {%Order.commutative(tap,Nat.mul(upper,inner_count)) : {Nat.is_lt(_,Nat.mul(1n+upper,inner_count)) == True{} : Bool}
    Order.add_lt(tap,inner_count,Nat.mul(upper,inner_count),tap_inside) : {Nat.is_lt(high,Nat.mul(1n+upper,inner_count)) == True{} : Bool}}
  +high_cap = {Bounds.strict_le(high,U32.to_nat(cap),Order.lt_le(high,Nat.mul(1n+upper,inner_count),U32.to_nat(cap),high_below,
      Order.transitive(Nat.mul(1n+upper,inner_count),total,U32.to_nat(cap),RowMajor.mul_le(1n+upper,group_count,inner_count,upper_follow),room))) :
    {Nat.is_le(high,U32.to_nat(cap)) == True{} : Bool}}
  +upper_cap = {Order.transitive(Nat.mul(upper,inner_count),high,U32.to_nat(cap),Order.le_add(Nat.mul(upper,inner_count),tap),high_cap) :
    {Nat.is_le(Nat.mul(upper,inner_count),U32.to_nat(cap)) == True{} : Bool}}
  +index_word = U32.from_nat(index)
  +pair_word = (index_word / inner : U32)
  +tap_word = (index_word % inner : U32)
  +double_word = (pair_word * 2 : U32)
  +next_word = (double_word + 1 : U32)
  +last_word = (groups - 1 : U32)
  +upper_word = U32.min(next_word,last_word)
  +low_word = (double_word * inner + tap_word : U32)
  +high_word = (upper_word * inner + tap_word : U32)
  +index_exact = Bounds.roundtrip(index,cap,cap_ok,index_cap)
  +pair_exact = {Equal.trans(Nat,U32.to_nat(pair_word),Nat.div(index,inner_count),pair,
      Division32.quotient_exact(cap,cap_ok,index_word,inner,index,inner_count,index_exact,inner_exact,divisor_room,positive),
      %Order.commutative(tap,Nat.mul(pair,inner_count)) : {Nat.div(_,inner_count) == pair : Nat}
      DivisionOrder.shifted_small_quotient(tap,inner_count,pair,positive,tap_inside)) : {U32.to_nat(pair_word) == pair : Nat}}
  +tap_exact = {Equal.trans(Nat,U32.to_nat(tap_word),Nat.mod(index,inner_count),tap,
      Division32.remainder_exact(cap,cap_ok,index_word,inner,index,inner_count,index_exact,inner_exact,divisor_room,positive),
      %Order.commutative(tap,Nat.mul(pair,inner_count)) : {Nat.mod(_,inner_count) == tap : Nat}
      DivisionOrder.shifted_small_remainder(tap,inner_count,pair,positive,tap_inside)) : {U32.to_nat(tap_word) == tap : Nat}}
  +double_exact = Arithmetic.product_exact(cap,cap_ok,pair_word,2,pair,2n,pair_exact,{==},double_cap)
  +scaled_exact = Arithmetic.product_exact(cap,cap_ok,double_word,inner,double,inner_count,double_exact,inner_exact,scaled_cap)
  +low_exact = Arithmetic.sum_exact(cap,cap_ok,(double_word * inner : U32),tap_word,Nat.mul(double,inner_count),tap,scaled_exact,tap_exact,low_cap)
  +next_exact = {Equal.trans(Nat,U32.to_nat(next_word),Nat.add(double,1n),1n+double,
      Arithmetic.sum_exact(cap,cap_ok,double_word,1,double,1n,double_exact,{==},next_cap),Order.commutative(double,1n)) : {U32.to_nat(next_word) == 1n+double : Nat}}
  +last_exact = Arithmetic.difference_exact(cap,cap_ok,groups,1,group_count,1n,groups_exact,{==},Order.strict_follow(0n,group_count,groups_positive),groups_cap)
  +upper_exact = {%next_exact : {U32.to_nat(upper_word) == Nat.min(_,last) : Nat}
    %last_exact : {U32.to_nat(upper_word) == Nat.min(U32.to_nat(next_word),_) : Nat}
    U32Extrema.minimum(next_word,last_word) : {U32.to_nat(upper_word) == upper : Nat}}
  +upper_scaled = Arithmetic.product_exact(cap,cap_ok,upper_word,inner,upper,inner_count,upper_exact,inner_exact,upper_cap)
  +high_exact = Arithmetic.sum_exact(cap,cap_ok,(upper_word * inner : U32),tap_word,Nat.mul(upper,inner_count),tap,upper_scaled,tap_exact,high_cap)
  %Equal.sym(U32,U32.from_nat(low),low_word,Arithmetic.from_nat_exact(low_word,low,low_exact)) :
    {pair_tiles(tree,Gemm.PairPlan{groups,inner},index_word) == Tile.Tiles{Tree.read(Tile.Tile8,tree,_),Tree.read(Tile.Tile8,tree,U32.from_nat(high))} : Tile.Tile16}
  %Equal.sym(U32,U32.from_nat(high),high_word,Arithmetic.from_nat_exact(high_word,high,high_exact)) :
    {pair_tiles(tree,Gemm.PairPlan{groups,inner},index_word) == Tile.Tiles{Tree.read(Tile.Tile8,tree,low_word),Tree.read(Tile.Tile8,tree,_)} : Tile.Tile16}
  {==}
