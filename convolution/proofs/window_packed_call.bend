# A call of the packed window kernel, as models: it transposes the rows of
# its band into planes, packs them and multiplies them by the weights, and
# gives the band back with the cells it was lent with.
import Base
import ../../lib/kernel_window_pipeline.bend as WindowPipeline
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/kernel_spatial_packing.bend as Packing
import ../../lib/kernel_window.bend as KernelWindow
import ../../lib/kernel_tile.bend as Tile
import ../../lib/storage_buffer.bend as Storage
import ../../lib/storage_copy.bend as Copy
import ../../lib/tensor.bend as Tensors
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/kernel_pipeline_buffer_shape.bend as BufferShape
import ../../lib/proofs/kernel_pipeline_model.bend as Model
import ../../lib/proofs/kernel_spatial_packing_refinement.bend as PackingModel
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/contiguous_copy.bend as Contiguous
import ../../lib/proofs/write_program.bend as Writes
import ../../lib/proofs/storage_copy_refinement.bend as CopyLaws

# A packed output piece: the kept operands and its output cells.
type PackedModel is Data:
  PackedModel{weights: Buffer.Model<Tile.Tile8>,bias: Buffer.Model<Tile.Tile8>,planes: Buffer.Model<Tile.Tile16>,tiles: Buffer.Model<Tile.Tile8>,
    output: Buffer.Model<Tile.Tile8>}

def represent(model: PackedModel) -> WindowPipeline.PackedWork:
  PackedModel{weights,bias,planes,tiles,output} = model
  WindowPipeline.PackedWork{Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)}

def packed_weights(model: PackedModel) -> Buffer.Model<Tile.Tile8>:
  PackedModel{weights,bias,planes,tiles,output} = model
  weights

def packed_bias(model: PackedModel) -> Buffer.Model<Tile.Tile8>:
  PackedModel{weights,bias,planes,tiles,output} = model
  bias

def packed_planes(model: PackedModel) -> Buffer.Model<Tile.Tile16>:
  PackedModel{weights,bias,planes,tiles,output} = model
  planes

def packed_tiles(model: PackedModel) -> Buffer.Model<Tile.Tile8>:
  PackedModel{weights,bias,planes,tiles,output} = model
  tiles

def packed_output(model: PackedModel) -> Buffer.Model<Tile.Tile8>:
  PackedModel{weights,bias,planes,tiles,output} = model
  output

# ---- The buffers a call gives back ----

# The model of the buffers a call keeps.
def remodel_kept(kept: Pipeline.Kept) -> BufferShape.KeptModel:
  BufferShape.kept_model(kept,BufferShape.observe_kept(kept))

def kept_equation(-kept: Pipeline.Kept,+observation: BufferShape.KeptObservation<kept>) ->
  {BufferShape.pack_kept(BufferShape.kept_model(kept,observation)) == kept : Pipeline.Kept}:
  BufferShape.KeptObservation{+model,+equation} = observation
  equation

def kept_remodeled(kept: Pipeline.Kept) -> {BufferShape.pack_kept(remodel_kept(kept)) == kept : Pipeline.Kept}:
  kept_equation(kept,BufferShape.observe_kept(kept))

# The work a call leaves: the kept buffers without the band.
def work_of(model: BufferShape.KeptModel) -> PackedModel:
  BufferShape.KeptModel{input,weights,bias,planes,tiles,output} = model
  PackedModel{weights,bias,planes,tiles,output}

def returned_same(+model: BufferShape.KeptModel) ->
  {WindowPipeline.packed_returned(BufferShape.pack_kept(model)) == (Buffer.pack(~Tile.Tile8,BufferShape.model_input(model)),represent(work_of(model))) :
    Storage.Buffer<Tile.Tile8> & WindowPipeline.PackedWork}:
  BufferShape.KeptModel{+input,+weights,+bias,+planes,+tiles,+output} = model
  {==}

# ---- A call ----

# The buffers a call gives back.
def call_kept(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32,+band: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>,+planes: Buffer.Model<Tile.Tile16>,+tiles: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) -> BufferShape.KeptModel:
  remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))

def call_same(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32,+band: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>,+planes: Buffer.Model<Tile.Tile16>,+tiles: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) ->
  {Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)) == BufferShape.pack_kept(call_kept(work,pack_start,offset,band,weights,bias,planes,tiles,output)) : Pipeline.Kept}:
  Equal.sym(Pipeline.Kept,BufferShape.pack_kept(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))),Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)),kept_remodeled(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output))))

# The first row and the rows of a band.
def band_first(band: KernelWindow.Band) -> U32:
  KernelWindow.Band{row_first,rows} = band
  row_first

def band_rows(band: KernelWindow.Band) -> U32:
  KernelWindow.Band{row_first,rows} = band
  rows

# The window work of a call: its rows from the band offset on.
def call_work(+convolution: WindowPipeline.Convolution,+call: WindowPipeline.Call) -> Pipeline.WindowWork:
  WindowPipeline.Convolution{h,+w,+k,+s,+p,oh,+ow,+channels,+inner,+gi,go,activation} = convolution
  WindowPipeline.Call{band,image,first,+offset,+count,+base} = call
  Pipeline.WindowWork{channels,(offset + count : U32),Packing.Stencil{base,band_rows(band),w,k,k,s,(0 - p : U32),(0 - p : U32),ow,inner,gi}}

# The packing start of a call: the first row of its first position in the
# band.
def call_start(+convolution: WindowPipeline.Convolution,+call: WindowPipeline.Call) -> U32:
  WindowPipeline.Convolution{h,w,k,+s,p,oh,+ow,channels,inner,gi,go,activation} = convolution
  WindowPipeline.Call{band,image,+first,offset,count,base} = call
  KernelWindow.start(first,s,ow,band_first(band))

def call_image(+call: WindowPipeline.Call) -> U32:
  WindowPipeline.Call{band,+image,first,offset,count,base} = call
  image

def call_offset(+call: WindowPipeline.Call) -> U32:
  WindowPipeline.Call{band,image,first,+offset,count,base} = call
  offset

# The buffers a call gives back.
def packed_after(+convolution: WindowPipeline.Convolution,+call: WindowPipeline.Call,band: Buffer.Model<Tile.Tile8>,work: PackedModel) -> BufferShape.KeptModel:
  PackedModel{weights,bias,planes,tiles,output} = work
  call_kept(call_work(convolution,call),call_start(convolution,call),call_offset(call),band,weights,bias,planes,tiles,output)

def band_after(convolution: WindowPipeline.Convolution,call: WindowPipeline.Call,band: Buffer.Model<Tile.Tile8>,work: PackedModel) -> Buffer.Model<Tile.Tile8>:
  BufferShape.model_input(packed_after(convolution,call,band,work))

def work_after(convolution: WindowPipeline.Convolution,call: WindowPipeline.Call,band: Buffer.Model<Tile.Tile8>,work: PackedModel) -> PackedModel:
  work_of(packed_after(convolution,call,band,work))

# A packed call is the call on the models.
def compute_same(+convolution: WindowPipeline.Convolution,+call: WindowPipeline.Call,+band: Buffer.Model<Tile.Tile8>,+work: PackedModel) ->
  {WindowPipeline.packed_call(convolution,call,Buffer.pack(~Tile.Tile8,band),represent(work)) ==
    (Buffer.pack(~Tile.Tile8,band_after(convolution,call,band,work)),represent(work_after(convolution,call,band,work))) : Storage.Buffer<Tile.Tile8> & WindowPipeline.PackedWork}:
  WindowPipeline.Convolution{+h,+w,+k,+s,+p,+oh,+ow,+channels,+inner,+gi,+go,+activation} = convolution
  WindowPipeline.Call{+kband,+image,+first,+offset,+count,+base} = call
  KernelWindow.Band{+row_first,+rows} = kband
  PackedModel{+weights,+bias,+planes,+tiles,+output} = work
  +work = {Pipeline.WindowWork{channels,(offset + count : U32),Packing.Stencil{base,rows,w,k,k,s,(0 - p : U32),(0 - p : U32),ow,inner,gi}} : Pipeline.WindowWork}
  +pack_start = KernelWindow.start(first,s,ow,row_first)
  %Equal.sym(Pipeline.Kept,Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)),BufferShape.pack_kept(call_kept(work,pack_start,offset,band,weights,bias,planes,tiles,output)),call_same(work,pack_start,offset,band,weights,bias,planes,tiles,output)) :
    {WindowPipeline.packed_returned(_) == (Buffer.pack(~Tile.Tile8,BufferShape.model_input(call_kept(work,pack_start,offset,band,weights,bias,planes,tiles,output))),represent(work_of(call_kept(work,pack_start,offset,band,weights,bias,planes,tiles,output)))) :
      Storage.Buffer<Tile.Tile8> & WindowPipeline.PackedWork}
  returned_same(call_kept(work,pack_start,offset,band,weights,bias,planes,tiles,output))

# The band a call gives back has the cells it was lent with.
def blocked_input(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32,+band: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>,+planes: Buffer.Model<Tile.Tile16>,+tiles: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) ->
  {Buffer.content(~Tile.Tile8,BufferShape.model_input(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output))))) == Buffer.content(~Tile.Tile8,band) : Buffer.Content<Tile.Tile8>}:
  Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,BufferShape.model_input(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output))))),Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,band)),Buffer.content(~Tile.Tile8,band),
    Model.kept_input_content(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output),Buffer.Observation{band,{==}},Buffer.Observation{weights,{==}},Buffer.Observation{bias,{==}},Buffer.Observation{planes,{==}},Buffer.Observation{tiles,{==}},Buffer.Observation{output,{==}},BufferShape.observe_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))),
    Buffer.packed_contents(~Tile.Tile8,band))

def band_kept(+convolution: WindowPipeline.Convolution,+call: WindowPipeline.Call,+band: Buffer.Model<Tile.Tile8>,+work: PackedModel) ->
  {Buffer.content(~Tile.Tile8,band_after(convolution,call,band,work)) == Buffer.content(~Tile.Tile8,band) : Buffer.Content<Tile.Tile8>}:
  PackedModel{+weights,+bias,+planes,+tiles,+output} = work
  blocked_input(call_work(convolution,call),call_start(convolution,call),call_offset(call),band,weights,bias,planes,tiles,output)

# ---- What a call leaves ----

# The blocks of the blocked loop of a call.
def call_cells(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32,+band: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>,+planes: Buffer.Model<Tile.Tile16>,+tiles: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) -> Model.Cells:
  Model.call_cells(Model.cells_into(work,pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{Buffer.tree(~Tile.Tile16,planes),Buffer.tree(~Tile.Tile8,tiles),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~Tile.Tile8,bias),Buffer.tree(~Tile.Tile8,output)}))

def frame_pairs(frame: Model.Frame) -> Tree.Tree<Tile.Tile16>:
  Model.Frame{pairs,weights,bias} = frame
  pairs

def frame_weights(frame: Model.Frame) -> Tree.Tree<Tile.Tile8>:
  Model.Frame{pairs,weights,bias} = frame
  weights

def frame_bias(frame: Model.Frame) -> Tree.Tree<Tile.Tile8>:
  Model.Frame{pairs,weights,bias} = frame
  bias

def lengths_planes(lengths: BufferShape.Lengths) -> Nat:
  BufferShape.Lengths{input,planes,tiles,weights,bias,output} = lengths
  planes

def lengths_tiles(lengths: BufferShape.Lengths) -> Nat:
  BufferShape.Lengths{input,planes,tiles,weights,bias,output} = lengths
  tiles

def lengths_output(lengths: BufferShape.Lengths) -> Nat:
  BufferShape.Lengths{input,planes,tiles,weights,bias,output} = lengths
  output

def packed_length(~Element: Data,+model: Buffer.Model<Element>) -> {BufferShape.buffer_length(~Element,Buffer.pack(~Element,model)) == Buffer.length(~Element,model) : Nat}:
  Buffer.Model{+tree,+length,+certificate} = model
  {==}

# The work a call leaves, read from the kept buffers.
def kept_output(+model: BufferShape.KeptModel) -> {Buffer.tree(~Tile.Tile8,packed_output(work_of(model))) == Model.cells_result(Model.call_cells(Model.kept_trees(model))) : Tree.Tree<Tile.Tile8>}:
  BufferShape.KeptModel{+input,+weights,+bias,+planes,+tiles,+output} = model
  {==}

def kept_tiles(+model: BufferShape.KeptModel) -> {Buffer.tree(~Tile.Tile8,packed_tiles(work_of(model))) == Model.cells_tiles(Model.call_cells(Model.kept_trees(model))) : Tree.Tree<Tile.Tile8>}:
  BufferShape.KeptModel{+input,+weights,+bias,+planes,+tiles,+output} = model
  {==}

def kept_planes(+model: BufferShape.KeptModel) -> {Buffer.tree(~Tile.Tile16,packed_planes(work_of(model))) == Model.cells_pairs(Model.call_cells(Model.kept_trees(model))) : Tree.Tree<Tile.Tile16>}:
  BufferShape.KeptModel{+input,+weights,+bias,+planes,+tiles,+output} = model
  {==}

def kept_weights(+model: BufferShape.KeptModel) -> {Buffer.tree(~Tile.Tile8,packed_weights(work_of(model))) == Model.cells_weights(Model.call_cells(Model.kept_trees(model))) : Tree.Tree<Tile.Tile8>}:
  BufferShape.KeptModel{+input,+weights,+bias,+planes,+tiles,+output} = model
  {==}

def kept_bias(+model: BufferShape.KeptModel) -> {Buffer.tree(~Tile.Tile8,packed_bias(work_of(model))) == Model.cells_bias(Model.call_cells(Model.kept_trees(model))) : Tree.Tree<Tile.Tile8>}:
  BufferShape.KeptModel{+input,+weights,+bias,+planes,+tiles,+output} = model
  {==}

def kept_output_length(+model: BufferShape.KeptModel) -> {Buffer.length(~Tile.Tile8,packed_output(work_of(model))) == lengths_output(BufferShape.model_lengths(model)) : Nat}:
  BufferShape.KeptModel{+input,+weights,+bias,+planes,+tiles,+output} = model
  {==}

def kept_tiles_length(+model: BufferShape.KeptModel) -> {Buffer.length(~Tile.Tile8,packed_tiles(work_of(model))) == lengths_tiles(BufferShape.model_lengths(model)) : Nat}:
  BufferShape.KeptModel{+input,+weights,+bias,+planes,+tiles,+output} = model
  {==}

def kept_planes_length(+model: BufferShape.KeptModel) -> {Buffer.length(~Tile.Tile16,packed_planes(work_of(model))) == lengths_planes(BufferShape.model_lengths(model)) : Nat}:
  BufferShape.KeptModel{+input,+weights,+bias,+planes,+tiles,+output} = model
  {==}

# The trees of the buffers of a blocked call are the call of its loop.
def blocked_trees(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32,+band: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>,+planes: Buffer.Model<Tile.Tile16>,+tiles: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) -> {Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))) == Model.cells_into(work,pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{Buffer.tree(~Tile.Tile16,planes),Buffer.tree(~Tile.Tile8,tiles),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~Tile.Tile8,bias),Buffer.tree(~Tile.Tile8,output)}) : Model.Call}:
  Model.into_kept_trees(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output),Buffer.Observation{band,{==}},Buffer.Observation{weights,{==}},Buffer.Observation{bias,{==}},Buffer.Observation{planes,{==}},Buffer.Observation{tiles,{==}},Buffer.Observation{output,{==}},BufferShape.observe_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output))))

def blocked_output(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32,+band: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>,+planes: Buffer.Model<Tile.Tile16>,+tiles: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) -> {Buffer.tree(~Tile.Tile8,packed_output(work_of(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))) == Model.cells_result(Model.call_cells(Model.cells_into(work,pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{Buffer.tree(~Tile.Tile16,planes),Buffer.tree(~Tile.Tile8,tiles),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~Tile.Tile8,bias),Buffer.tree(~Tile.Tile8,output)}))) : Tree.Tree<Tile.Tile8>}:
  Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,packed_output(work_of(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))),Model.cells_result(Model.call_cells(Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))),Model.cells_result(Model.call_cells(Model.cells_into(work,pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{Buffer.tree(~Tile.Tile16,planes),Buffer.tree(~Tile.Tile8,tiles),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~Tile.Tile8,bias),Buffer.tree(~Tile.Tile8,output)}))),
    kept_output(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))),Equal.cong(Model.Call,Tree.Tree<Tile.Tile8>,call => Model.cells_result(Model.call_cells(call)),Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))),Model.cells_into(work,pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{Buffer.tree(~Tile.Tile16,planes),Buffer.tree(~Tile.Tile8,tiles),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~Tile.Tile8,bias),Buffer.tree(~Tile.Tile8,output)}),blocked_trees(work,pack_start,offset,band,weights,bias,planes,tiles,output)))

def blocked_tiles(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32,+band: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>,+planes: Buffer.Model<Tile.Tile16>,+tiles: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) -> {Buffer.tree(~Tile.Tile8,packed_tiles(work_of(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))) == Model.cells_tiles(Model.call_cells(Model.cells_into(work,pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{Buffer.tree(~Tile.Tile16,planes),Buffer.tree(~Tile.Tile8,tiles),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~Tile.Tile8,bias),Buffer.tree(~Tile.Tile8,output)}))) : Tree.Tree<Tile.Tile8>}:
  Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,packed_tiles(work_of(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))),Model.cells_tiles(Model.call_cells(Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))),Model.cells_tiles(Model.call_cells(Model.cells_into(work,pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{Buffer.tree(~Tile.Tile16,planes),Buffer.tree(~Tile.Tile8,tiles),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~Tile.Tile8,bias),Buffer.tree(~Tile.Tile8,output)}))),
    kept_tiles(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))),Equal.cong(Model.Call,Tree.Tree<Tile.Tile8>,call => Model.cells_tiles(Model.call_cells(call)),Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))),Model.cells_into(work,pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{Buffer.tree(~Tile.Tile16,planes),Buffer.tree(~Tile.Tile8,tiles),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~Tile.Tile8,bias),Buffer.tree(~Tile.Tile8,output)}),blocked_trees(work,pack_start,offset,band,weights,bias,planes,tiles,output)))

# A blocked call keeps its weights and bias and leaves the planes of its band.
def blocked_frame(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32,+band: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>,+planes: Buffer.Model<Tile.Tile16>,+tiles: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) -> {Model.Frame{Model.cells_pairs(Model.call_cells(Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))),Model.cells_weights(Model.call_cells(Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))),Model.cells_bias(Model.call_cells(Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output))))))} == Model.Frame{PackingModel.transposed(Model.call_window(work),Model.call_planes(work,pack_start,offset),Buffer.tree(~Tile.Tile8,band),Buffer.tree(~Tile.Tile16,planes)),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~Tile.Tile8,bias)} : Model.Frame}:
  Equal.trans(Model.Frame,Model.Frame{Model.cells_pairs(Model.call_cells(Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))),Model.cells_weights(Model.call_cells(Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))),Model.cells_bias(Model.call_cells(Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output))))))},Model.frame(Model.call_cells(Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))),Model.Frame{PackingModel.transposed(Model.call_window(work),Model.call_planes(work,pack_start,offset),Buffer.tree(~Tile.Tile8,band),Buffer.tree(~Tile.Tile16,planes)),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~Tile.Tile8,bias)},Equal.sym(Model.Frame,Model.frame(Model.call_cells(Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))),Model.Frame{Model.cells_pairs(Model.call_cells(Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))),Model.cells_weights(Model.call_cells(Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))),Model.cells_bias(Model.call_cells(Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output))))))},Model.framed(Model.call_cells(Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output))))))),
    {%Equal.sym(Model.Call,Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))),Model.cells_into(work,pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{Buffer.tree(~Tile.Tile16,planes),Buffer.tree(~Tile.Tile8,tiles),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~Tile.Tile8,bias),Buffer.tree(~Tile.Tile8,output)}),blocked_trees(work,pack_start,offset,band,weights,bias,planes,tiles,output)) : {Model.frame(Model.call_cells(_)) == Model.Frame{PackingModel.transposed(Model.call_window(work),Model.call_planes(work,pack_start,offset),Buffer.tree(~Tile.Tile8,band),Buffer.tree(~Tile.Tile16,planes)),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~Tile.Tile8,bias)} : Model.Frame}
      Model.into_frame(work,pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{Buffer.tree(~Tile.Tile16,planes),Buffer.tree(~Tile.Tile8,tiles),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~Tile.Tile8,bias),Buffer.tree(~Tile.Tile8,output)}) : {Model.frame(Model.call_cells(Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))) == Model.Frame{PackingModel.transposed(Model.call_window(work),Model.call_planes(work,pack_start,offset),Buffer.tree(~Tile.Tile8,band),Buffer.tree(~Tile.Tile16,planes)),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~Tile.Tile8,bias)} : Model.Frame}})

def blocked_weights(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32,+band: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>,+planes: Buffer.Model<Tile.Tile16>,+tiles: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) -> {Buffer.tree(~Tile.Tile8,packed_weights(work_of(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))) == Buffer.tree(~Tile.Tile8,weights) : Tree.Tree<Tile.Tile8>}:
  Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,packed_weights(work_of(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))),Model.cells_weights(Model.call_cells(Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))),Buffer.tree(~Tile.Tile8,weights),kept_weights(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))),
    Equal.cong(Model.Frame,Tree.Tree<Tile.Tile8>,frame => frame_weights(frame),Model.Frame{Model.cells_pairs(Model.call_cells(Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))),Model.cells_weights(Model.call_cells(Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))),Model.cells_bias(Model.call_cells(Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output))))))},Model.Frame{PackingModel.transposed(Model.call_window(work),Model.call_planes(work,pack_start,offset),Buffer.tree(~Tile.Tile8,band),Buffer.tree(~Tile.Tile16,planes)),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~Tile.Tile8,bias)},blocked_frame(work,pack_start,offset,band,weights,bias,planes,tiles,output)))

def blocked_bias(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32,+band: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>,+planes: Buffer.Model<Tile.Tile16>,+tiles: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) -> {Buffer.tree(~Tile.Tile8,packed_bias(work_of(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))) == Buffer.tree(~Tile.Tile8,bias) : Tree.Tree<Tile.Tile8>}:
  Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,packed_bias(work_of(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))),Model.cells_bias(Model.call_cells(Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))),Buffer.tree(~Tile.Tile8,bias),kept_bias(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))),
    Equal.cong(Model.Frame,Tree.Tree<Tile.Tile8>,frame => frame_bias(frame),Model.Frame{Model.cells_pairs(Model.call_cells(Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))),Model.cells_weights(Model.call_cells(Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))),Model.cells_bias(Model.call_cells(Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output))))))},Model.Frame{PackingModel.transposed(Model.call_window(work),Model.call_planes(work,pack_start,offset),Buffer.tree(~Tile.Tile8,band),Buffer.tree(~Tile.Tile16,planes)),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~Tile.Tile8,bias)},blocked_frame(work,pack_start,offset,band,weights,bias,planes,tiles,output)))

def blocked_planes(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32,+band: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>,+planes: Buffer.Model<Tile.Tile16>,+tiles: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) -> {Buffer.tree(~Tile.Tile16,packed_planes(work_of(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))) == PackingModel.transposed(Model.call_window(work),Model.call_planes(work,pack_start,offset),Buffer.tree(~Tile.Tile8,band),Buffer.tree(~Tile.Tile16,planes)) : Tree.Tree<Tile.Tile16>}:
  Equal.trans(Tree.Tree<Tile.Tile16>,Buffer.tree(~Tile.Tile16,packed_planes(work_of(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))),Model.cells_pairs(Model.call_cells(Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))),PackingModel.transposed(Model.call_window(work),Model.call_planes(work,pack_start,offset),Buffer.tree(~Tile.Tile8,band),Buffer.tree(~Tile.Tile16,planes)),kept_planes(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))),
    Equal.cong(Model.Frame,Tree.Tree<Tile.Tile16>,frame => frame_pairs(frame),Model.Frame{Model.cells_pairs(Model.call_cells(Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))),Model.cells_weights(Model.call_cells(Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))),Model.cells_bias(Model.call_cells(Model.kept_trees(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output))))))},Model.Frame{PackingModel.transposed(Model.call_window(work),Model.call_planes(work,pack_start,offset),Buffer.tree(~Tile.Tile8,band),Buffer.tree(~Tile.Tile16,planes)),Buffer.tree(~Tile.Tile8,weights),Buffer.tree(~Tile.Tile8,bias)},blocked_frame(work,pack_start,offset,band,weights,bias,planes,tiles,output)))

# A blocked call keeps the lengths of its buffers.
def blocked_lengths(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32,+band: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>,+planes: Buffer.Model<Tile.Tile16>,+tiles: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) -> {BufferShape.model_lengths(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))) == BufferShape.Lengths{Buffer.length(~Tile.Tile8,band),Buffer.length(~Tile.Tile16,planes),Buffer.length(~Tile.Tile8,tiles),Buffer.length(~Tile.Tile8,weights),Buffer.length(~Tile.Tile8,bias),Buffer.length(~Tile.Tile8,output)} : BufferShape.Lengths}:
  %packed_length(~Tile.Tile8,band) : {BufferShape.model_lengths(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))) == BufferShape.Lengths{_,Buffer.length(~Tile.Tile16,planes),Buffer.length(~Tile.Tile8,tiles),Buffer.length(~Tile.Tile8,weights),Buffer.length(~Tile.Tile8,bias),Buffer.length(~Tile.Tile8,output)} : BufferShape.Lengths}
  %packed_length(~Tile.Tile16,planes) : {BufferShape.model_lengths(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))) == BufferShape.Lengths{BufferShape.buffer_length(~Tile.Tile8,Buffer.pack(~Tile.Tile8,band)),_,Buffer.length(~Tile.Tile8,tiles),Buffer.length(~Tile.Tile8,weights),Buffer.length(~Tile.Tile8,bias),Buffer.length(~Tile.Tile8,output)} : BufferShape.Lengths}
  %packed_length(~Tile.Tile8,tiles) : {BufferShape.model_lengths(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))) == BufferShape.Lengths{BufferShape.buffer_length(~Tile.Tile8,Buffer.pack(~Tile.Tile8,band)),BufferShape.buffer_length(~Tile.Tile16,Buffer.pack(~Tile.Tile16,planes)),_,Buffer.length(~Tile.Tile8,weights),Buffer.length(~Tile.Tile8,bias),Buffer.length(~Tile.Tile8,output)} : BufferShape.Lengths}
  %packed_length(~Tile.Tile8,weights) : {BufferShape.model_lengths(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))) == BufferShape.Lengths{BufferShape.buffer_length(~Tile.Tile8,Buffer.pack(~Tile.Tile8,band)),BufferShape.buffer_length(~Tile.Tile16,Buffer.pack(~Tile.Tile16,planes)),BufferShape.buffer_length(~Tile.Tile8,Buffer.pack(~Tile.Tile8,tiles)),_,Buffer.length(~Tile.Tile8,bias),Buffer.length(~Tile.Tile8,output)} : BufferShape.Lengths}
  %packed_length(~Tile.Tile8,bias) : {BufferShape.model_lengths(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))) == BufferShape.Lengths{BufferShape.buffer_length(~Tile.Tile8,Buffer.pack(~Tile.Tile8,band)),BufferShape.buffer_length(~Tile.Tile16,Buffer.pack(~Tile.Tile16,planes)),BufferShape.buffer_length(~Tile.Tile8,Buffer.pack(~Tile.Tile8,tiles)),BufferShape.buffer_length(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),_,Buffer.length(~Tile.Tile8,output)} : BufferShape.Lengths}
  %packed_length(~Tile.Tile8,output) : {BufferShape.model_lengths(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))) == BufferShape.Lengths{BufferShape.buffer_length(~Tile.Tile8,Buffer.pack(~Tile.Tile8,band)),BufferShape.buffer_length(~Tile.Tile16,Buffer.pack(~Tile.Tile16,planes)),BufferShape.buffer_length(~Tile.Tile8,Buffer.pack(~Tile.Tile8,tiles)),BufferShape.buffer_length(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),BufferShape.buffer_length(~Tile.Tile8,Buffer.pack(~Tile.Tile8,bias)),_} : BufferShape.Lengths}
  BufferShape.into_kept_lengths(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output),Buffer.Observation{band,{==}},Buffer.Observation{weights,{==}},Buffer.Observation{bias,{==}},Buffer.Observation{planes,{==}},Buffer.Observation{tiles,{==}},Buffer.Observation{output,{==}},BufferShape.observe_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output))))

def blocked_output_length(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32,+band: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>,+planes: Buffer.Model<Tile.Tile16>,+tiles: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) -> {Buffer.length(~Tile.Tile8,packed_output(work_of(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))) == Buffer.length(~Tile.Tile8,output) : Nat}:
  Equal.trans(Nat,Buffer.length(~Tile.Tile8,packed_output(work_of(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))),lengths_output(BufferShape.model_lengths(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output))))),Buffer.length(~Tile.Tile8,output),kept_output_length(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))),
    Equal.cong(BufferShape.Lengths,Nat,lengths => lengths_output(lengths),BufferShape.model_lengths(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))),BufferShape.Lengths{Buffer.length(~Tile.Tile8,band),Buffer.length(~Tile.Tile16,planes),Buffer.length(~Tile.Tile8,tiles),Buffer.length(~Tile.Tile8,weights),Buffer.length(~Tile.Tile8,bias),Buffer.length(~Tile.Tile8,output)},blocked_lengths(work,pack_start,offset,band,weights,bias,planes,tiles,output)))

def blocked_tiles_length(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32,+band: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>,+planes: Buffer.Model<Tile.Tile16>,+tiles: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) -> {Buffer.length(~Tile.Tile8,packed_tiles(work_of(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))) == Buffer.length(~Tile.Tile8,tiles) : Nat}:
  Equal.trans(Nat,Buffer.length(~Tile.Tile8,packed_tiles(work_of(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))),lengths_tiles(BufferShape.model_lengths(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output))))),Buffer.length(~Tile.Tile8,tiles),kept_tiles_length(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))),
    Equal.cong(BufferShape.Lengths,Nat,lengths => lengths_tiles(lengths),BufferShape.model_lengths(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))),BufferShape.Lengths{Buffer.length(~Tile.Tile8,band),Buffer.length(~Tile.Tile16,planes),Buffer.length(~Tile.Tile8,tiles),Buffer.length(~Tile.Tile8,weights),Buffer.length(~Tile.Tile8,bias),Buffer.length(~Tile.Tile8,output)},blocked_lengths(work,pack_start,offset,band,weights,bias,planes,tiles,output)))

def blocked_planes_length(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32,+band: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>,+planes: Buffer.Model<Tile.Tile16>,+tiles: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) -> {Buffer.length(~Tile.Tile16,packed_planes(work_of(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))) == Buffer.length(~Tile.Tile16,planes) : Nat}:
  Equal.trans(Nat,Buffer.length(~Tile.Tile16,packed_planes(work_of(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))))),lengths_planes(BufferShape.model_lengths(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output))))),Buffer.length(~Tile.Tile16,planes),kept_planes_length(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))),
    Equal.cong(BufferShape.Lengths,Nat,lengths => lengths_planes(lengths),BufferShape.model_lengths(remodel_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,band),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output)))),BufferShape.Lengths{Buffer.length(~Tile.Tile8,band),Buffer.length(~Tile.Tile16,planes),Buffer.length(~Tile.Tile8,tiles),Buffer.length(~Tile.Tile8,weights),Buffer.length(~Tile.Tile8,bias),Buffer.length(~Tile.Tile8,output)},blocked_lengths(work,pack_start,offset,band,weights,bias,planes,tiles,output)))
