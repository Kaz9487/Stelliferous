# The pointwise kernel of the window skeleton, one call: the product of a
# call reads its rows from the band, sixteen channels a block, and stages
# its outputs as pairs.
import Base
import ../../lib/kernel_window_pipeline.bend as WindowPipeline
import ../../lib/kernel_window.bend as KernelWindow
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/kernel_gemm_buffer.bend as GemmBuffer
import ../../lib/kernel_tile.bend as Tile
import ../../lib/storage_buffer.bend as Storage
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/write_program.bend as Writes
import ../../lib/proofs/kernel_gemm_point_jobs.bend as PointJobs
import ../../lib/proofs/kernel_gemm_point_buffer.bend as PointBuffer

# A pointwise output piece: the pairs of the weights, the bias and the
# staged pairs of its outputs.
type PointModel is Data:
  PointModel{pairs: Buffer.Model<Tile.Tile16>,bias: Buffer.Model<Tile.Tile8>,staged: Buffer.Model<Tile.Tile16>}

def represent(model: PointModel) -> WindowPipeline.PointWork:
  PointModel{pairs,bias,staged} = model
  WindowPipeline.PointWork{Buffer.pack(~Tile.Tile16,pairs),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,staged)}

def point_pairs(model: PointModel) -> Buffer.Model<Tile.Tile16>:
  PointModel{pairs,bias,staged} = model
  pairs

def point_bias(model: PointModel) -> Buffer.Model<Tile.Tile8>:
  PointModel{pairs,bias,staged} = model
  bias

def point_staged(model: PointModel) -> Buffer.Model<Tile.Tile16>:
  PointModel{pairs,bias,staged} = model
  staged

# ---- The buffers a call gives back ----

# The product of a call: the channels over its positions, the cells of its
# first position read from the band.
def call_matrix(+convolution: WindowPipeline.Convolution,+call: WindowPipeline.Call) -> Gemm.PointMatrix:
  WindowPipeline.Convolution{h,w,k,s,p,oh,+ow,+channels,+inner,+gi,go,activation} = convolution
  WindowPipeline.Call{band,image,+first,+offset,+count,+base} = call
  KernelWindow.Band{+row_first,rows} = band
  Gemm.PointMatrix{Gemm.TileMatrix{channels,count,inner,offset,(offset + count : U32)},(base + KernelWindow.start(first,1,ow,row_first) * gi : U32),gi}

# The models of the buffers a call gives back.
def point_after(+convolution: WindowPipeline.Convolution,+call: WindowPipeline.Call,band: Buffer.Model<Tile.Tile8>,work: PointModel) -> PointBuffer.PointKeptModel:
  PointModel{pairs,bias,staged} = work
  PointBuffer.remodel_point_kept(GemmBuffer.point_keeping(call_matrix(convolution,call),Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile16,pairs),Buffer.pack(~Tile.Tile8,bias),
    Buffer.pack(~Tile.Tile16,staged)))

def kept_cells(model: PointBuffer.PointKeptModel) -> Buffer.Model<Tile.Tile8>:
  PointBuffer.PointKeptModel{bias,cells,pairs,staged} = model
  cells

def kept_work(model: PointBuffer.PointKeptModel) -> PointModel:
  PointBuffer.PointKeptModel{bias,cells,pairs,staged} = model
  PointModel{pairs,bias,staged}

def band_after(convolution: WindowPipeline.Convolution,call: WindowPipeline.Call,band: Buffer.Model<Tile.Tile8>,work: PointModel) -> Buffer.Model<Tile.Tile8>:
  kept_cells(point_after(convolution,call,band,work))

def work_after(convolution: WindowPipeline.Convolution,call: WindowPipeline.Call,band: Buffer.Model<Tile.Tile8>,work: PointModel) -> PointModel:
  kept_work(point_after(convolution,call,band,work))

def returned_same(+model: PointBuffer.PointKeptModel) ->
  {WindowPipeline.point_returned(PointBuffer.pack_point_kept(model)) == (Buffer.pack(~Tile.Tile8,kept_cells(model)),represent(kept_work(model))) : Storage.Buffer<Tile.Tile8> & WindowPipeline.PointWork}:
  PointBuffer.PointKeptModel{+bias,+cells,+pairs,+staged} = model
  {==}

# A pointwise call is the call on the models.
def compute_same(+convolution: WindowPipeline.Convolution,+call: WindowPipeline.Call,+band: Buffer.Model<Tile.Tile8>,+work: PointModel) ->
  {WindowPipeline.point_call(convolution,call,Buffer.pack(~Tile.Tile8,band),represent(work)) ==
    (Buffer.pack(~Tile.Tile8,band_after(convolution,call,band,work)),represent(work_after(convolution,call,band,work))) : Storage.Buffer<Tile.Tile8> & WindowPipeline.PointWork}:
  WindowPipeline.Convolution{+h,+w,+k,+s,+p,+oh,+ow,+channels,+inner,+gi,+go,+activation} = convolution
  WindowPipeline.Call{+kband,+image,+first,+offset,+count,+base} = call
  KernelWindow.Band{+row_first,+rows} = kband
  PointModel{+pairs,+bias,+staged} = work
  +matrix = {Gemm.PointMatrix{Gemm.TileMatrix{channels,count,inner,offset,(offset + count : U32)},(base + KernelWindow.start(first,1,ow,row_first) * gi : U32),gi} : Gemm.PointMatrix}
  %PointBuffer.point_kept_remodeled(GemmBuffer.point_keeping(matrix,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile16,pairs),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,staged))) :
    {WindowPipeline.point_returned(_) ==
      (Buffer.pack(~Tile.Tile8,kept_cells(PointBuffer.remodel_point_kept(GemmBuffer.point_keeping(matrix,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile16,pairs),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,staged))))),
        represent(kept_work(PointBuffer.remodel_point_kept(GemmBuffer.point_keeping(matrix,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile16,pairs),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,staged)))))) :
      Storage.Buffer<Tile.Tile8> & WindowPipeline.PointWork}
  returned_same(PointBuffer.remodel_point_kept(GemmBuffer.point_keeping(matrix,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile16,pairs),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,staged))))

# ---- What a call leaves ----

# The contents of the buffers a call gives back: the model run of its jobs
# over the contents it was given.
def after_contents(+convolution: WindowPipeline.Convolution,+call: WindowPipeline.Call,+band: Buffer.Model<Tile.Tile8>,+work: PointModel) ->
  {PointBuffer.model_contents(point_after(convolution,call,band,work)) ==
    PointBuffer.operand_contents(PointJobs.run(U32.to_nat(Gemm.point_count(call_matrix(convolution,call))),call_matrix(convolution,call),
      PointBuffer.model_operands(band,point_pairs(work),point_bias(work),point_staged(work))),PointBuffer.model_lengths(band,point_pairs(work),point_bias(work),point_staged(work))) : PointBuffer.PointContents}:
  PointModel{+pairs,+bias,+staged} = work
  Equal.trans(PointBuffer.PointContents,
    PointBuffer.model_contents(PointBuffer.remodel_point_kept(GemmBuffer.point_keeping(call_matrix(convolution,call),Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile16,pairs),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,staged)))),
    PointBuffer.kept_contents(GemmBuffer.point_keeping(call_matrix(convolution,call),Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile16,pairs),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,staged))),
    PointBuffer.operand_contents(PointJobs.run(U32.to_nat(Gemm.point_count(call_matrix(convolution,call))),call_matrix(convolution,call),PointBuffer.model_operands(band,pairs,bias,staged)),
      PointBuffer.model_lengths(band,pairs,bias,staged)),
    PointBuffer.remodeled_contents(GemmBuffer.point_keeping(call_matrix(convolution,call),Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile16,pairs),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,staged))),
    PointBuffer.keeping_contents(call_matrix(convolution,call),band,pairs,bias,staged))

def contents_cells(contents: PointBuffer.PointContents) -> Buffer.Content<Tile.Tile8>:
  PointBuffer.PointContents{bias,cells,pairs,staged} = contents
  cells

def contents_pairs(contents: PointBuffer.PointContents) -> Buffer.Content<Tile.Tile16>:
  PointBuffer.PointContents{bias,cells,pairs,staged} = contents
  pairs

def contents_bias(contents: PointBuffer.PointContents) -> Buffer.Content<Tile.Tile8>:
  PointBuffer.PointContents{bias,cells,pairs,staged} = contents
  bias

def contents_staged(contents: PointBuffer.PointContents) -> Buffer.Content<Tile.Tile16>:
  PointBuffer.PointContents{bias,cells,pairs,staged} = contents
  staged

def readers_cells(readers: PointJobs.Readers) -> Tree.Tree<Tile.Tile8>:
  PointJobs.Readers{bias,cells,pairs} = readers
  cells

def readers_pairs(readers: PointJobs.Readers) -> Tree.Tree<Tile.Tile16>:
  PointJobs.Readers{bias,cells,pairs} = readers
  pairs

def readers_bias(readers: PointJobs.Readers) -> Tree.Tree<Tile.Tile8>:
  PointJobs.Readers{bias,cells,pairs} = readers
  bias


def lengths_cells(lengths: PointBuffer.PointLengths) -> Nat:
  PointBuffer.PointLengths{bias,cells,pairs,staged} = lengths
  cells

def operand_cells(+operands: PointJobs.Operands,+lengths: PointBuffer.PointLengths) ->
  {contents_cells(PointBuffer.operand_contents(operands,lengths)) == Buffer.Content{readers_cells(PointJobs.readers_of(operands)),lengths_cells(lengths)} : Buffer.Content<Tile.Tile8>}:
  PointJobs.Operands{+bias,+cells,+pairs,+staged} = operands
  PointBuffer.PointLengths{+bias_length,+cell_length,+pair_length,+staged_length} = lengths
  {==}

def kept_cells_content(+model: PointBuffer.PointKeptModel) -> {Buffer.content(~Tile.Tile8,kept_cells(model)) == contents_cells(PointBuffer.model_contents(model)) : Buffer.Content<Tile.Tile8>}:
  PointBuffer.PointKeptModel{+bias,+cells,+pairs,+staged} = model
  {==}

def model_content(+model: Buffer.Model<Tile.Tile8>) -> {Buffer.Content{Buffer.tree(~Tile.Tile8,model),Buffer.length(~Tile.Tile8,model)} == Buffer.content(~Tile.Tile8,model) : Buffer.Content<Tile.Tile8>}:
  Buffer.Model{+tree,+length,+certificate} = model
  {==}

# The band a call gives back has the cells it was lent with.
def band_kept(+convolution: WindowPipeline.Convolution,+call: WindowPipeline.Call,+band: Buffer.Model<Tile.Tile8>,+work: PointModel) ->
  {Buffer.content(~Tile.Tile8,band_after(convolution,call,band,work)) == Buffer.content(~Tile.Tile8,band) : Buffer.Content<Tile.Tile8>}:
  +matrix = call_matrix(convolution,call)
  +count = U32.to_nat(Gemm.point_count(matrix))
  +operands = PointBuffer.model_operands(band,point_pairs(work),point_bias(work),point_staged(work))
  +lengths = PointBuffer.model_lengths(band,point_pairs(work),point_bias(work),point_staged(work))
  +run = PointJobs.run(count,matrix,operands)
  Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,kept_cells(point_after(convolution,call,band,work))),contents_cells(PointBuffer.model_contents(point_after(convolution,call,band,work))),
    Buffer.content(~Tile.Tile8,band),kept_cells_content(point_after(convolution,call,band,work)),
    Equal.trans(Buffer.Content<Tile.Tile8>,contents_cells(PointBuffer.model_contents(point_after(convolution,call,band,work))),contents_cells(PointBuffer.operand_contents(run,lengths)),
      Buffer.content(~Tile.Tile8,band),
      Equal.cong(PointBuffer.PointContents,Buffer.Content<Tile.Tile8>,contents => contents_cells(contents),PointBuffer.model_contents(point_after(convolution,call,band,work)),
        PointBuffer.operand_contents(run,lengths),after_contents(convolution,call,band,work)),
      Equal.trans(Buffer.Content<Tile.Tile8>,contents_cells(PointBuffer.operand_contents(run,lengths)),Buffer.Content{readers_cells(PointJobs.readers_of(run)),lengths_cells(lengths)},
        Buffer.content(~Tile.Tile8,band),operand_cells(run,lengths),
        Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.Content{readers_cells(PointJobs.readers_of(run)),lengths_cells(lengths)},
          Buffer.Content{readers_cells(PointJobs.readers_of(operands)),lengths_cells(lengths)},Buffer.content(~Tile.Tile8,band),
          Equal.cong(PointJobs.Readers,Buffer.Content<Tile.Tile8>,readers => Buffer.Content{readers_cells(readers),lengths_cells(lengths)},PointJobs.readers_of(run),
            PointJobs.readers_of(operands),PointBuffer.run_readers(count,matrix,operands)),
          model_content(band)))))

# ---- The pairs, bias and staged pairs a call gives back ----

def kept_work_contents(+model: PointBuffer.PointKeptModel) ->
  {PointBuffer.PointContents{Buffer.content(~Tile.Tile8,point_bias(kept_work(model))),Buffer.content(~Tile.Tile8,kept_cells(model)),
      Buffer.content(~Tile.Tile16,point_pairs(kept_work(model))),Buffer.content(~Tile.Tile16,point_staged(kept_work(model)))} ==
    PointBuffer.model_contents(model) : PointBuffer.PointContents}:
  PointBuffer.PointKeptModel{+bias,+cells,+pairs,+staged} = model
  {==}

def lengths_bias(lengths: PointBuffer.PointLengths) -> Nat:
  PointBuffer.PointLengths{bias,cells,pairs,staged} = lengths
  bias

def lengths_pairs(lengths: PointBuffer.PointLengths) -> Nat:
  PointBuffer.PointLengths{bias,cells,pairs,staged} = lengths
  pairs

def lengths_staged(lengths: PointBuffer.PointLengths) -> Nat:
  PointBuffer.PointLengths{bias,cells,pairs,staged} = lengths
  staged

def operand_parts(+operands: PointJobs.Operands,+lengths: PointBuffer.PointLengths) ->
  {PointBuffer.operand_contents(operands,lengths) ==
    PointBuffer.PointContents{Buffer.Content{readers_bias(PointJobs.readers_of(operands)),lengths_bias(lengths)},Buffer.Content{readers_cells(PointJobs.readers_of(operands)),lengths_cells(lengths)},
      Buffer.Content{readers_pairs(PointJobs.readers_of(operands)),lengths_pairs(lengths)},Buffer.Content{PointJobs.staged_tree(operands),lengths_staged(lengths)}} : PointBuffer.PointContents}:
  PointJobs.Operands{+bias,+cells,+pairs,+staged} = operands
  PointBuffer.PointLengths{+bias_length,+cell_length,+pair_length,+staged_length} = lengths
  {==}

# What a call leaves: the pairs and bias it was given and its stores over the
# staged pairs it was given.
def call_after(+convolution: WindowPipeline.Convolution,+call: WindowPipeline.Call,+band: Buffer.Model<Tile.Tile8>,+work: PointModel) ->
  {PointBuffer.PointContents{Buffer.content(~Tile.Tile8,point_bias(work_after(convolution,call,band,work))),Buffer.content(~Tile.Tile8,band_after(convolution,call,band,work)),
      Buffer.content(~Tile.Tile16,point_pairs(work_after(convolution,call,band,work))),Buffer.content(~Tile.Tile16,point_staged(work_after(convolution,call,band,work)))} ==
    PointBuffer.PointContents{Buffer.Content{Buffer.tree(~Tile.Tile8,point_bias(work)),Buffer.length(~Tile.Tile8,point_bias(work))},
      Buffer.Content{Buffer.tree(~Tile.Tile8,band),Buffer.length(~Tile.Tile8,band)},
      Buffer.Content{Buffer.tree(~Tile.Tile16,point_pairs(work)),Buffer.length(~Tile.Tile16,point_pairs(work))},
      Buffer.Content{Writes.pure(Tile.Tile16,PointJobs.all_writes(U32.to_nat(Gemm.point_count(call_matrix(convolution,call))),0,call_matrix(convolution,call),
        PointJobs.Readers{Buffer.tree(~Tile.Tile8,point_bias(work)),Buffer.tree(~Tile.Tile8,band),Buffer.tree(~Tile.Tile16,point_pairs(work))}),
        Buffer.tree(~Tile.Tile16,point_staged(work))),Buffer.length(~Tile.Tile16,point_staged(work))}} : PointBuffer.PointContents}:
  PointModel{+pairs,+bias,+staged} = work
  +matrix = call_matrix(convolution,call)
  +count = U32.to_nat(Gemm.point_count(matrix))
  +operands = PointBuffer.model_operands(band,pairs,bias,staged)
  +lengths = PointBuffer.model_lengths(band,pairs,bias,staged)
  +run = PointJobs.run(count,matrix,operands)
  +after = point_after(convolution,call,band,PointModel{pairs,bias,staged})
  Equal.trans(PointBuffer.PointContents,
    PointBuffer.PointContents{Buffer.content(~Tile.Tile8,point_bias(kept_work(after))),Buffer.content(~Tile.Tile8,kept_cells(after)),
      Buffer.content(~Tile.Tile16,point_pairs(kept_work(after))),Buffer.content(~Tile.Tile16,point_staged(kept_work(after)))},
    PointBuffer.operand_contents(run,lengths),
    PointBuffer.PointContents{Buffer.Content{Buffer.tree(~Tile.Tile8,bias),Buffer.length(~Tile.Tile8,bias)},Buffer.Content{Buffer.tree(~Tile.Tile8,band),Buffer.length(~Tile.Tile8,band)},
      Buffer.Content{Buffer.tree(~Tile.Tile16,pairs),Buffer.length(~Tile.Tile16,pairs)},
      Buffer.Content{Writes.pure(Tile.Tile16,PointJobs.all_writes(count,0,matrix,PointJobs.readers_of(operands)),Buffer.tree(~Tile.Tile16,staged)),Buffer.length(~Tile.Tile16,staged)}},
    Equal.trans(PointBuffer.PointContents,
      PointBuffer.PointContents{Buffer.content(~Tile.Tile8,point_bias(kept_work(after))),Buffer.content(~Tile.Tile8,kept_cells(after)),
        Buffer.content(~Tile.Tile16,point_pairs(kept_work(after))),Buffer.content(~Tile.Tile16,point_staged(kept_work(after)))},
      PointBuffer.model_contents(after),PointBuffer.operand_contents(run,lengths),kept_work_contents(after),after_contents(convolution,call,band,PointModel{pairs,bias,staged})),
    Equal.trans(PointBuffer.PointContents,PointBuffer.operand_contents(run,lengths),
      PointBuffer.PointContents{Buffer.Content{readers_bias(PointJobs.readers_of(run)),lengths_bias(lengths)},Buffer.Content{readers_cells(PointJobs.readers_of(run)),lengths_cells(lengths)},
        Buffer.Content{readers_pairs(PointJobs.readers_of(run)),lengths_pairs(lengths)},Buffer.Content{PointJobs.staged_tree(run),lengths_staged(lengths)}},
      PointBuffer.PointContents{Buffer.Content{Buffer.tree(~Tile.Tile8,bias),Buffer.length(~Tile.Tile8,bias)},Buffer.Content{Buffer.tree(~Tile.Tile8,band),Buffer.length(~Tile.Tile8,band)},
        Buffer.Content{Buffer.tree(~Tile.Tile16,pairs),Buffer.length(~Tile.Tile16,pairs)},
        Buffer.Content{Writes.pure(Tile.Tile16,PointJobs.all_writes(count,0,matrix,PointJobs.readers_of(operands)),Buffer.tree(~Tile.Tile16,staged)),Buffer.length(~Tile.Tile16,staged)}},
      operand_parts(run,lengths),
      %Equal.sym(PointJobs.Readers,PointJobs.readers_of(run),PointJobs.readers_of(operands),PointBuffer.run_readers(count,matrix,operands)) :
        {PointBuffer.PointContents{Buffer.Content{readers_bias(_),lengths_bias(lengths)},Buffer.Content{readers_cells(_),lengths_cells(lengths)},
          Buffer.Content{readers_pairs(_),lengths_pairs(lengths)},Buffer.Content{PointJobs.staged_tree(run),lengths_staged(lengths)}} ==
          PointBuffer.PointContents{Buffer.Content{Buffer.tree(~Tile.Tile8,bias),Buffer.length(~Tile.Tile8,bias)},Buffer.Content{Buffer.tree(~Tile.Tile8,band),Buffer.length(~Tile.Tile8,band)},
            Buffer.Content{Buffer.tree(~Tile.Tile16,pairs),Buffer.length(~Tile.Tile16,pairs)},
            Buffer.Content{Writes.pure(Tile.Tile16,PointJobs.all_writes(count,0,matrix,PointJobs.readers_of(operands)),Buffer.tree(~Tile.Tile16,staged)),Buffer.length(~Tile.Tile16,staged)}} :
          PointBuffer.PointContents}
      %Equal.sym(Tree.Tree<Tile.Tile16>,PointJobs.staged_tree(run),Writes.pure(Tile.Tile16,PointJobs.all_writes(count,0,matrix,PointJobs.readers_of(operands)),Buffer.tree(~Tile.Tile16,staged)),PointJobs.run_writes(count,matrix,operands)) :
        {PointBuffer.PointContents{Buffer.Content{Buffer.tree(~Tile.Tile8,bias),Buffer.length(~Tile.Tile8,bias)},Buffer.Content{Buffer.tree(~Tile.Tile8,band),Buffer.length(~Tile.Tile8,band)},
          Buffer.Content{Buffer.tree(~Tile.Tile16,pairs),Buffer.length(~Tile.Tile16,pairs)},Buffer.Content{_,Buffer.length(~Tile.Tile16,staged)}} ==
          PointBuffer.PointContents{Buffer.Content{Buffer.tree(~Tile.Tile8,bias),Buffer.length(~Tile.Tile8,bias)},Buffer.Content{Buffer.tree(~Tile.Tile8,band),Buffer.length(~Tile.Tile8,band)},
            Buffer.Content{Buffer.tree(~Tile.Tile16,pairs),Buffer.length(~Tile.Tile16,pairs)},
            Buffer.Content{Writes.pure(Tile.Tile16,PointJobs.all_writes(count,0,matrix,PointJobs.readers_of(operands)),Buffer.tree(~Tile.Tile16,staged)),Buffer.length(~Tile.Tile16,staged)}} :
          PointBuffer.PointContents}
      {==}))

# ---- What a call leaves, part by part ----

# The staged pairs a call leaves are its stores over those it was given.
def staged_stores(+convolution: WindowPipeline.Convolution,+call: WindowPipeline.Call,+band: Buffer.Model<Tile.Tile8>,+work: PointModel) ->
  {Buffer.tree(~Tile.Tile16,point_staged(work_after(convolution,call,band,work))) == Writes.pure(Tile.Tile16,PointJobs.all_writes(U32.to_nat(Gemm.point_count(call_matrix(convolution,call))),0,call_matrix(convolution,call),PointJobs.Readers{Buffer.tree(~Tile.Tile8,point_bias(work)),Buffer.tree(~Tile.Tile8,band),Buffer.tree(~Tile.Tile16,point_pairs(work))}),Buffer.tree(~Tile.Tile16,point_staged(work))) : Tree.Tree<Tile.Tile16>}:
  Equal.trans(Tree.Tree<Tile.Tile16>,Buffer.tree(~Tile.Tile16,point_staged(work_after(convolution,call,band,work))),Buffer.content_tree(~Tile.Tile16,contents_staged(PointBuffer.PointContents{Buffer.content(~Tile.Tile8,point_bias(work_after(convolution,call,band,work))),Buffer.content(~Tile.Tile8,band_after(convolution,call,band,work)),Buffer.content(~Tile.Tile16,point_pairs(work_after(convolution,call,band,work))),Buffer.content(~Tile.Tile16,point_staged(work_after(convolution,call,band,work)))})),Writes.pure(Tile.Tile16,PointJobs.all_writes(U32.to_nat(Gemm.point_count(call_matrix(convolution,call))),0,call_matrix(convolution,call),PointJobs.Readers{Buffer.tree(~Tile.Tile8,point_bias(work)),Buffer.tree(~Tile.Tile8,band),Buffer.tree(~Tile.Tile16,point_pairs(work))}),Buffer.tree(~Tile.Tile16,point_staged(work))),
    Equal.sym(Tree.Tree<Tile.Tile16>,Buffer.content_tree(~Tile.Tile16,contents_staged(PointBuffer.PointContents{Buffer.content(~Tile.Tile8,point_bias(work_after(convolution,call,band,work))),Buffer.content(~Tile.Tile8,band_after(convolution,call,band,work)),Buffer.content(~Tile.Tile16,point_pairs(work_after(convolution,call,band,work))),Buffer.content(~Tile.Tile16,point_staged(work_after(convolution,call,band,work)))})),Buffer.tree(~Tile.Tile16,point_staged(work_after(convolution,call,band,work))),Buffer.content_tree_of(~Tile.Tile16,point_staged(work_after(convolution,call,band,work)))),
    Equal.cong(PointBuffer.PointContents,Tree.Tree<Tile.Tile16>,contents => Buffer.content_tree(~Tile.Tile16,contents_staged(contents)),PointBuffer.PointContents{Buffer.content(~Tile.Tile8,point_bias(work_after(convolution,call,band,work))),Buffer.content(~Tile.Tile8,band_after(convolution,call,band,work)),Buffer.content(~Tile.Tile16,point_pairs(work_after(convolution,call,band,work))),Buffer.content(~Tile.Tile16,point_staged(work_after(convolution,call,band,work)))},PointBuffer.PointContents{Buffer.Content{Buffer.tree(~Tile.Tile8,point_bias(work)),Buffer.length(~Tile.Tile8,point_bias(work))},Buffer.Content{Buffer.tree(~Tile.Tile8,band),Buffer.length(~Tile.Tile8,band)},Buffer.Content{Buffer.tree(~Tile.Tile16,point_pairs(work)),Buffer.length(~Tile.Tile16,point_pairs(work))},Buffer.Content{Writes.pure(Tile.Tile16,PointJobs.all_writes(U32.to_nat(Gemm.point_count(call_matrix(convolution,call))),0,call_matrix(convolution,call),PointJobs.Readers{Buffer.tree(~Tile.Tile8,point_bias(work)),Buffer.tree(~Tile.Tile8,band),Buffer.tree(~Tile.Tile16,point_pairs(work))}),Buffer.tree(~Tile.Tile16,point_staged(work))),Buffer.length(~Tile.Tile16,point_staged(work))}},call_after(convolution,call,band,work)))

# They keep their length.
def staged_length_kept(+convolution: WindowPipeline.Convolution,+call: WindowPipeline.Call,+band: Buffer.Model<Tile.Tile8>,+work: PointModel) ->
  {Buffer.length(~Tile.Tile16,point_staged(work_after(convolution,call,band,work))) == Buffer.length(~Tile.Tile16,point_staged(work)) : Nat}:
  Equal.trans(Nat,Buffer.length(~Tile.Tile16,point_staged(work_after(convolution,call,band,work))),Buffer.content_length(~Tile.Tile16,contents_staged(PointBuffer.PointContents{Buffer.content(~Tile.Tile8,point_bias(work_after(convolution,call,band,work))),Buffer.content(~Tile.Tile8,band_after(convolution,call,band,work)),Buffer.content(~Tile.Tile16,point_pairs(work_after(convolution,call,band,work))),Buffer.content(~Tile.Tile16,point_staged(work_after(convolution,call,band,work)))})),Buffer.length(~Tile.Tile16,point_staged(work)),
    Equal.sym(Nat,Buffer.content_length(~Tile.Tile16,contents_staged(PointBuffer.PointContents{Buffer.content(~Tile.Tile8,point_bias(work_after(convolution,call,band,work))),Buffer.content(~Tile.Tile8,band_after(convolution,call,band,work)),Buffer.content(~Tile.Tile16,point_pairs(work_after(convolution,call,band,work))),Buffer.content(~Tile.Tile16,point_staged(work_after(convolution,call,band,work)))})),Buffer.length(~Tile.Tile16,point_staged(work_after(convolution,call,band,work))),Buffer.content_length_of(~Tile.Tile16,point_staged(work_after(convolution,call,band,work)))),
    Equal.cong(PointBuffer.PointContents,Nat,contents => Buffer.content_length(~Tile.Tile16,contents_staged(contents)),PointBuffer.PointContents{Buffer.content(~Tile.Tile8,point_bias(work_after(convolution,call,band,work))),Buffer.content(~Tile.Tile8,band_after(convolution,call,band,work)),Buffer.content(~Tile.Tile16,point_pairs(work_after(convolution,call,band,work))),Buffer.content(~Tile.Tile16,point_staged(work_after(convolution,call,band,work)))},PointBuffer.PointContents{Buffer.Content{Buffer.tree(~Tile.Tile8,point_bias(work)),Buffer.length(~Tile.Tile8,point_bias(work))},Buffer.Content{Buffer.tree(~Tile.Tile8,band),Buffer.length(~Tile.Tile8,band)},Buffer.Content{Buffer.tree(~Tile.Tile16,point_pairs(work)),Buffer.length(~Tile.Tile16,point_pairs(work))},Buffer.Content{Writes.pure(Tile.Tile16,PointJobs.all_writes(U32.to_nat(Gemm.point_count(call_matrix(convolution,call))),0,call_matrix(convolution,call),PointJobs.Readers{Buffer.tree(~Tile.Tile8,point_bias(work)),Buffer.tree(~Tile.Tile8,band),Buffer.tree(~Tile.Tile16,point_pairs(work))}),Buffer.tree(~Tile.Tile16,point_staged(work))),Buffer.length(~Tile.Tile16,point_staged(work))}},call_after(convolution,call,band,work)))

# A call gives back the pairs of the weights it was given.
def pairs_returned(+convolution: WindowPipeline.Convolution,+call: WindowPipeline.Call,+band: Buffer.Model<Tile.Tile8>,+work: PointModel) ->
  {Buffer.tree(~Tile.Tile16,point_pairs(work_after(convolution,call,band,work))) == Buffer.tree(~Tile.Tile16,point_pairs(work)) : Tree.Tree<Tile.Tile16>}:
  Equal.trans(Tree.Tree<Tile.Tile16>,Buffer.tree(~Tile.Tile16,point_pairs(work_after(convolution,call,band,work))),Buffer.content_tree(~Tile.Tile16,contents_pairs(PointBuffer.PointContents{Buffer.content(~Tile.Tile8,point_bias(work_after(convolution,call,band,work))),Buffer.content(~Tile.Tile8,band_after(convolution,call,band,work)),Buffer.content(~Tile.Tile16,point_pairs(work_after(convolution,call,band,work))),Buffer.content(~Tile.Tile16,point_staged(work_after(convolution,call,band,work)))})),Buffer.tree(~Tile.Tile16,point_pairs(work)),
    Equal.sym(Tree.Tree<Tile.Tile16>,Buffer.content_tree(~Tile.Tile16,contents_pairs(PointBuffer.PointContents{Buffer.content(~Tile.Tile8,point_bias(work_after(convolution,call,band,work))),Buffer.content(~Tile.Tile8,band_after(convolution,call,band,work)),Buffer.content(~Tile.Tile16,point_pairs(work_after(convolution,call,band,work))),Buffer.content(~Tile.Tile16,point_staged(work_after(convolution,call,band,work)))})),Buffer.tree(~Tile.Tile16,point_pairs(work_after(convolution,call,band,work))),Buffer.content_tree_of(~Tile.Tile16,point_pairs(work_after(convolution,call,band,work)))),
    Equal.cong(PointBuffer.PointContents,Tree.Tree<Tile.Tile16>,contents => Buffer.content_tree(~Tile.Tile16,contents_pairs(contents)),PointBuffer.PointContents{Buffer.content(~Tile.Tile8,point_bias(work_after(convolution,call,band,work))),Buffer.content(~Tile.Tile8,band_after(convolution,call,band,work)),Buffer.content(~Tile.Tile16,point_pairs(work_after(convolution,call,band,work))),Buffer.content(~Tile.Tile16,point_staged(work_after(convolution,call,band,work)))},PointBuffer.PointContents{Buffer.Content{Buffer.tree(~Tile.Tile8,point_bias(work)),Buffer.length(~Tile.Tile8,point_bias(work))},Buffer.Content{Buffer.tree(~Tile.Tile8,band),Buffer.length(~Tile.Tile8,band)},Buffer.Content{Buffer.tree(~Tile.Tile16,point_pairs(work)),Buffer.length(~Tile.Tile16,point_pairs(work))},Buffer.Content{Writes.pure(Tile.Tile16,PointJobs.all_writes(U32.to_nat(Gemm.point_count(call_matrix(convolution,call))),0,call_matrix(convolution,call),PointJobs.Readers{Buffer.tree(~Tile.Tile8,point_bias(work)),Buffer.tree(~Tile.Tile8,band),Buffer.tree(~Tile.Tile16,point_pairs(work))}),Buffer.tree(~Tile.Tile16,point_staged(work))),Buffer.length(~Tile.Tile16,point_staged(work))}},call_after(convolution,call,band,work)))

# And the bias.
def bias_returned(+convolution: WindowPipeline.Convolution,+call: WindowPipeline.Call,+band: Buffer.Model<Tile.Tile8>,+work: PointModel) ->
  {Buffer.tree(~Tile.Tile8,point_bias(work_after(convolution,call,band,work))) == Buffer.tree(~Tile.Tile8,point_bias(work)) : Tree.Tree<Tile.Tile8>}:
  Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,point_bias(work_after(convolution,call,band,work))),Buffer.content_tree(~Tile.Tile8,contents_bias(PointBuffer.PointContents{Buffer.content(~Tile.Tile8,point_bias(work_after(convolution,call,band,work))),Buffer.content(~Tile.Tile8,band_after(convolution,call,band,work)),Buffer.content(~Tile.Tile16,point_pairs(work_after(convolution,call,band,work))),Buffer.content(~Tile.Tile16,point_staged(work_after(convolution,call,band,work)))})),Buffer.tree(~Tile.Tile8,point_bias(work)),
    Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.content_tree(~Tile.Tile8,contents_bias(PointBuffer.PointContents{Buffer.content(~Tile.Tile8,point_bias(work_after(convolution,call,band,work))),Buffer.content(~Tile.Tile8,band_after(convolution,call,band,work)),Buffer.content(~Tile.Tile16,point_pairs(work_after(convolution,call,band,work))),Buffer.content(~Tile.Tile16,point_staged(work_after(convolution,call,band,work)))})),Buffer.tree(~Tile.Tile8,point_bias(work_after(convolution,call,band,work))),Buffer.content_tree_of(~Tile.Tile8,point_bias(work_after(convolution,call,band,work)))),
    Equal.cong(PointBuffer.PointContents,Tree.Tree<Tile.Tile8>,contents => Buffer.content_tree(~Tile.Tile8,contents_bias(contents)),PointBuffer.PointContents{Buffer.content(~Tile.Tile8,point_bias(work_after(convolution,call,band,work))),Buffer.content(~Tile.Tile8,band_after(convolution,call,band,work)),Buffer.content(~Tile.Tile16,point_pairs(work_after(convolution,call,band,work))),Buffer.content(~Tile.Tile16,point_staged(work_after(convolution,call,band,work)))},PointBuffer.PointContents{Buffer.Content{Buffer.tree(~Tile.Tile8,point_bias(work)),Buffer.length(~Tile.Tile8,point_bias(work))},Buffer.Content{Buffer.tree(~Tile.Tile8,band),Buffer.length(~Tile.Tile8,band)},Buffer.Content{Buffer.tree(~Tile.Tile16,point_pairs(work)),Buffer.length(~Tile.Tile16,point_pairs(work))},Buffer.Content{Writes.pure(Tile.Tile16,PointJobs.all_writes(U32.to_nat(Gemm.point_count(call_matrix(convolution,call))),0,call_matrix(convolution,call),PointJobs.Readers{Buffer.tree(~Tile.Tile8,point_bias(work)),Buffer.tree(~Tile.Tile8,band),Buffer.tree(~Tile.Tile16,point_pairs(work))}),Buffer.tree(~Tile.Tile16,point_staged(work))),Buffer.length(~Tile.Tile16,point_staged(work))}},call_after(convolution,call,band,work)))
