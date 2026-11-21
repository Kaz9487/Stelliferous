# Buffer footprint facts only: each output keeps its depth and logical length.
# The proof uses wrapper certificates, never MAC internals.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_spatial_packing.bend as Packing
import ../kernel_packing_buffer.bend as PackingBuffer
import ../kernel_window.bend as Window
import ./kernel_gemm_certificate.bend as GemmCertificate
import ../storage_buffer.bend as Storage
import ../kernel_gemm_buffer.bend as GemmBuffer
import ../kernel_pipeline.bend as Pipeline
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_map.bend as TiledMap
import ../tensor_tiled_cells.bend as Cells
import ../tensor_grid.bend as Grid
import ../kernel_tile.bend as Tile
import ../tensor.bend as Tensors
import ../math_activation.bend as Activations
import ./storage_certificate.bend as Certificate
import ./traversal_certificate.bend as TraversalCertificate
import ./kernel_packing_certificate.bend as PackingCertificate
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./storage_tree_model.bend as Tree
import ./storage_allocation_refinement.bend as AllocationLaws
import ../storage_allocation.bend as Allocation

type Footprint is Data:
  Footprint{depth: Nat,length: Nat}

def footprint(~Element: Data,buffer: Storage.Buffer<Element>) -> Footprint:
  Storage.Buffer{array,length,Certificate.Certificate{depth,equation}} = buffer
  Footprint{depth,length}

def depth(footprint: Footprint) -> Nat:
  Footprint{depth,length} = footprint
  depth

def length(footprint: Footprint) -> Nat:
  Footprint{depth,length} = footprint
  length

def model_ready(~Element: Data,+model: Buffer.Model<Element>,
  depth_ok: {Nat.is_le(depth(footprint(~Element,Buffer.pack(~Element,model))),30n) == True{} : Bool},
  room: {Nat.is_le(length(footprint(~Element,Buffer.pack(~Element,model))),U32.to_nat(U32.shln(1,depth(footprint(~Element,Buffer.pack(~Element,model)))))) == True{} : Bool}) ->
  Buffer.Sized<Element,model>:
  Buffer.Model{+tree,+length,Certificate.Certificate{+depth,+equation}} = model
  +model = {Buffer.Model{tree,length,Certificate.Certificate{depth,equation}} : Buffer.Model<Element>}
  Buffer.Sized{depth_ok,
    %Equal.sym(U32,Tree.capacity(Element,tree),U32.shln(1,depth),Shape.capacity(Element,depth,tree,Buffer.balanced(~Element,model))) :
      {Nat.is_le(length,U32.to_nat(_)) == True{} : Bool}
    room}

def content_ready(~Element: Data,+model: Buffer.Model<Element>,
  depth_ok: {Nat.is_le(depth(footprint(~Element,Buffer.pack(~Element,model))),30n) == True{} : Bool},
  room: {Nat.is_le(length(footprint(~Element,Buffer.pack(~Element,model))),U32.to_nat(U32.shln(1,depth(footprint(~Element,Buffer.pack(~Element,model)))))) == True{} : Bool}) ->
  Buffer.ContentReady<Element,Buffer.content(~Element,model)>:
  Buffer.content_ready(~Element,model,model_ready(~Element,model,depth_ok,room))

def allocated(~Element: Data,+fill: Element,+count: U32) ->
  {footprint(~Element,Tensors.buffer(~Element,fill,count)) == Footprint{Tensors.depth_for(count),U32.to_nat(count)} : Footprint}:
  %Equal.sym(Storage.Buffer<Element>,Allocation.allocate(~Element,Tensors.depth_for(count),fill,U32.to_nat(count)),Storage.allocate(~Element,Tensors.depth_for(count),fill,U32.to_nat(count)),
    AllocationLaws.dispatch_same(Element,Allocation.allocated(~Element),Tensors.depth_for(count),fill,U32.to_nat(count))) :
    {footprint(~Element,_) == Footprint{Tensors.depth_for(count),U32.to_nat(count)} : Footprint}
  {==}

type BlockModel is Data:
  BlockModel{planes: Buffer.Model<Tile.Tile16>,tiles: Buffer.Model<Tile.Tile8>,weights: Buffer.Model<Tile.Tile8>,bias: Buffer.Model<Tile.Tile8>,output: Buffer.Model<Tile.Tile8>}

def pack_blocks(model: BlockModel) -> Pipeline.BlockState:
  BlockModel{planes,tiles,weights,bias,output} = model
  Pipeline.BlockState{Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile8,output)}

type BlockObservation<-state: Pipeline.BlockState> is Data:
  BlockObservation{model: BlockModel,equation: {pack_blocks(model) == state : Pipeline.BlockState}}

def observed_blocks(-planes: Storage.Buffer<Tile.Tile16>,-tiles: Storage.Buffer<Tile.Tile8>,-weights: Storage.Buffer<Tile.Tile8>,-bias: Storage.Buffer<Tile.Tile8>,-output: Storage.Buffer<Tile.Tile8>,
  plane_observation: Buffer.Observation<Tile.Tile16,planes>,tile_observation: Buffer.Observation<Tile.Tile8,tiles>,weight_observation: Buffer.Observation<Tile.Tile8,weights>,
  bias_observation: Buffer.Observation<Tile.Tile8,bias>,output_observation: Buffer.Observation<Tile.Tile8,output>) -> BlockObservation<Pipeline.BlockState{planes,tiles,weights,bias,output}>:
  Buffer.Observation{+pm,pe} = plane_observation
  Buffer.Observation{+tm,te} = tile_observation
  Buffer.Observation{+wm,we} = weight_observation
  Buffer.Observation{+bm,be} = bias_observation
  Buffer.Observation{+om,oe} = output_observation
  BlockObservation{BlockModel{pm,tm,wm,bm,om},
    %pe : {pack_blocks(BlockModel{pm,tm,wm,bm,om}) == Pipeline.BlockState{_,tiles,weights,bias,output} : Pipeline.BlockState}
    %te : {pack_blocks(BlockModel{pm,tm,wm,bm,om}) == Pipeline.BlockState{Buffer.pack(~Tile.Tile16,pm),_,weights,bias,output} : Pipeline.BlockState}
    %we : {pack_blocks(BlockModel{pm,tm,wm,bm,om}) == Pipeline.BlockState{Buffer.pack(~Tile.Tile16,pm),Buffer.pack(~Tile.Tile8,tm),_,bias,output} : Pipeline.BlockState}
    %be : {pack_blocks(BlockModel{pm,tm,wm,bm,om}) == Pipeline.BlockState{Buffer.pack(~Tile.Tile16,pm),Buffer.pack(~Tile.Tile8,tm),Buffer.pack(~Tile.Tile8,wm),_,output} : Pipeline.BlockState}
    %oe : {pack_blocks(BlockModel{pm,tm,wm,bm,om}) == Pipeline.BlockState{Buffer.pack(~Tile.Tile16,pm),Buffer.pack(~Tile.Tile8,tm),Buffer.pack(~Tile.Tile8,wm),Buffer.pack(~Tile.Tile8,bm),_} :
      Pipeline.BlockState}
    {==}}

def observe_blocks(state: Pipeline.BlockState) -> BlockObservation<state>:
  Pipeline.BlockState{planes,tiles,weights,bias,output} = state
  observed_blocks(planes,tiles,weights,bias,output,Buffer.observe(~Tile.Tile16,planes),Buffer.observe(~Tile.Tile8,tiles),Buffer.observe(~Tile.Tile8,weights),
    Buffer.observe(~Tile.Tile8,bias),Buffer.observe(~Tile.Tile8,output))

# The buffers a call gives back, as models.
type KeptModel is Data:
  KeptModel{input: Buffer.Model<Tile.Tile8>,weights: Buffer.Model<Tile.Tile8>,bias: Buffer.Model<Tile.Tile8>,planes: Buffer.Model<Tile.Tile16>,tiles: Buffer.Model<Tile.Tile8>,
    output: Buffer.Model<Tile.Tile8>}

def pack_kept(model: KeptModel) -> Pipeline.Kept:
  KeptModel{input,weights,bias,planes,tiles,output} = model
  Pipeline.Kept{Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),
    Buffer.pack(~Tile.Tile8,output)}

type KeptObservation<-kept: Pipeline.Kept> is Data:
  KeptObservation{model: KeptModel,equation: {pack_kept(model) == kept : Pipeline.Kept}}

def observed_kept(-input: Storage.Buffer<Tile.Tile8>,-weights: Storage.Buffer<Tile.Tile8>,-bias: Storage.Buffer<Tile.Tile8>,-planes: Storage.Buffer<Tile.Tile16>,
  -tiles: Storage.Buffer<Tile.Tile8>,-output: Storage.Buffer<Tile.Tile8>,
  input_observation: Buffer.Observation<Tile.Tile8,input>,weight_observation: Buffer.Observation<Tile.Tile8,weights>,bias_observation: Buffer.Observation<Tile.Tile8,bias>,
  plane_observation: Buffer.Observation<Tile.Tile16,planes>,tile_observation: Buffer.Observation<Tile.Tile8,tiles>,output_observation: Buffer.Observation<Tile.Tile8,output>) ->
  KeptObservation<Pipeline.Kept{input,weights,bias,planes,tiles,output}>:
  Buffer.Observation{+im,ie} = input_observation
  Buffer.Observation{+wm,we} = weight_observation
  Buffer.Observation{+bm,be} = bias_observation
  Buffer.Observation{+pm,pe} = plane_observation
  Buffer.Observation{+tm,te} = tile_observation
  Buffer.Observation{+om,oe} = output_observation
  +model = {KeptModel{im,wm,bm,pm,tm,om} : KeptModel}
  KeptObservation{model,
    %ie : {pack_kept(model) == Pipeline.Kept{_,weights,bias,planes,tiles,output} : Pipeline.Kept}
    %we : {pack_kept(model) == Pipeline.Kept{Buffer.pack(~Tile.Tile8,im),_,bias,planes,tiles,output} : Pipeline.Kept}
    %be : {pack_kept(model) == Pipeline.Kept{Buffer.pack(~Tile.Tile8,im),Buffer.pack(~Tile.Tile8,wm),_,planes,tiles,output} : Pipeline.Kept}
    %pe : {pack_kept(model) == Pipeline.Kept{Buffer.pack(~Tile.Tile8,im),Buffer.pack(~Tile.Tile8,wm),Buffer.pack(~Tile.Tile8,bm),_,tiles,output} : Pipeline.Kept}
    %te : {pack_kept(model) == Pipeline.Kept{Buffer.pack(~Tile.Tile8,im),Buffer.pack(~Tile.Tile8,wm),Buffer.pack(~Tile.Tile8,bm),Buffer.pack(~Tile.Tile16,pm),_,output} : Pipeline.Kept}
    %oe : {pack_kept(model) == Pipeline.Kept{Buffer.pack(~Tile.Tile8,im),Buffer.pack(~Tile.Tile8,wm),Buffer.pack(~Tile.Tile8,bm),Buffer.pack(~Tile.Tile16,pm),Buffer.pack(~Tile.Tile8,tm),_} :
      Pipeline.Kept}
    {==}}

def observe_kept(kept: Pipeline.Kept) -> KeptObservation<kept>:
  Pipeline.Kept{input,weights,bias,planes,tiles,output} = kept
  observed_kept(input,weights,bias,planes,tiles,output,Buffer.observe(~Tile.Tile8,input),Buffer.observe(~Tile.Tile8,weights),Buffer.observe(~Tile.Tile8,bias),
    Buffer.observe(~Tile.Tile16,planes),Buffer.observe(~Tile.Tile8,tiles),Buffer.observe(~Tile.Tile8,output))

def kept_model(-kept: Pipeline.Kept,observation: KeptObservation<kept>) -> KeptModel:
  KeptObservation{model,equation} = observation
  model

# The lengths of the buffers of a call.
type Lengths is Data:
  Lengths{input: Nat,planes: Nat,tiles: Nat,weights: Nat,bias: Nat,output: Nat}

def buffer_length(~Element: Data,buffer: Storage.Buffer<Element>) -> Nat:
  Storage.Buffer{array,length,certificate} = buffer
  length

def lengths(+input: Nat,state: Pipeline.BlockState) -> Lengths:
  Pipeline.BlockState{planes,tiles,weights,bias,output} = state
  Lengths{input,buffer_length(~Tile.Tile16,planes),buffer_length(~Tile.Tile8,tiles),buffer_length(~Tile.Tile8,weights),buffer_length(~Tile.Tile8,bias),
    buffer_length(~Tile.Tile8,output)}

def kept_lengths(kept: Pipeline.Kept) -> Lengths:
  Pipeline.Kept{input,weights,bias,planes,tiles,output} = kept
  Lengths{buffer_length(~Tile.Tile8,input),buffer_length(~Tile.Tile16,planes),buffer_length(~Tile.Tile8,tiles),buffer_length(~Tile.Tile8,weights),
    buffer_length(~Tile.Tile8,bias),buffer_length(~Tile.Tile8,output)}

def cells_kept_lengths(-input: Nat,planes: Storage.Buffer<Tile.Tile16>,operands: Gemm.CellOperands,+row_length: Nat,+lane_length: Nat,+bias_length: Nat,+length: Nat,
  certificates: GemmCertificate.CellCertified<operands>) ->
  {lengths(input,Pipeline.block_kept(planes,GemmBuffer.cells_kept(operands,row_length,lane_length,bias_length,length,certificates))) ==
    Lengths{input,buffer_length(~Tile.Tile16,planes),row_length,lane_length,bias_length,length} : Lengths}:
  Gemm.CellOperands{bias,Gemm.Operands{row_tiles,lane_tiles,unused_bias,output}} = operands
  GemmCertificate.CellCertified{bias_certificate,rows,lanes,output_certificate} = certificates
  {==}

def kept_buffer_lengths(-input: Nat,+rows: U32,+end: U32,+inner: U32,+start: U32,+span: U32,planes: Storage.Buffer<Tile.Tile16>,tiles: Storage.Buffer<Tile.Tile8>,
  weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>) ->
  {lengths(input,Pipeline.block_packed(rows,end,inner,start,span,weights,bias,output,(planes,tiles))) ==
    Lengths{input,buffer_length(~Tile.Tile16,planes),buffer_length(~Tile.Tile8,tiles),buffer_length(~Tile.Tile8,weights),buffer_length(~Tile.Tile8,bias),
      buffer_length(~Tile.Tile8,output)} : Lengths}:
  match tiles weights bias output:
    case Storage.Buffer{ta,+tl,+tc} Storage.Buffer{wa,+wl,+wc} Storage.Buffer{ba,+bl,+bc} Storage.Buffer{oa,+ol,oc}:
      +matrix = {Gemm.TileMatrix{rows,span,inner,start,end} : Gemm.TileMatrix}
      +count = U32.to_nat(Gemm.tile_jobs(matrix))
      cells_kept_lengths(input,planes,Gemm.cell_run(count,matrix,Gemm.CellOperands{ba,Gemm.Operands{ta,wa,Unit{},oa}}),tl,wl,bl,ol,
        GemmCertificate.cell_certificates(count,matrix,ta,wa,ba,Unit{},oa,tc,wc,bc,oc))

def packed_buffer_lengths(-input: Nat,+rows: U32,+end: U32,+inner: U32,+start: U32,+span: U32,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<Tile.Tile8>,
  output: Storage.Buffer<Tile.Tile8>,+pairs_length: Nat,+tile_length: Nat,result: Array<Tile.Tile16> & Array<Tile.Tile8>,
  certificates: TraversalCertificate.Certificates<Array<Tile.Tile16>,Tile.Tile8,Certificate.family(~Tile.Tile16),result>) ->
  {lengths(input,Pipeline.block_packed(rows,end,inner,start,span,weights,bias,output,PackingBuffer.buffers(~Tile.Tile16,~Tile.Tile8,pairs_length,tile_length,result,certificates))) ==
    Lengths{input,pairs_length,tile_length,buffer_length(~Tile.Tile8,weights),buffer_length(~Tile.Tile8,bias),buffer_length(~Tile.Tile8,output)} : Lengths}:
  (pairs,tiles) = result
  TraversalCertificate.Certificates{pc,tc} = certificates
  kept_buffer_lengths(input,rows,end,inner,start,span,Storage.Buffer{pairs,pairs_length,pc},Storage.Buffer{tiles,tile_length,tc},weights,bias,output)

def step_lengths(-input: Nat,+work: Pipeline.WindowWork,+planes: Packing.Planes,+pack_start: U32,+output_start: U32,+span: U32,state: Pipeline.BlockState) ->
  {lengths(input,Pipeline.block_step_window(work,planes,pack_start,output_start,span,state)) == lengths(input,state) : Lengths}:
  Pipeline.WindowWork{+rows,+end,+window} = work
  Packing.Stencil{+base,+h,+w,+kr,+kc,+s,+ro,+co,+ow,+inner,+groups} = window
  Pipeline.BlockState{Storage.Buffer{pairs,+pl,pc},Storage.Buffer{tiles,+tl,tc},weights,bias,output} = state
  +window = {Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,inner,groups} : Packing.Stencil}
  +count = U32.to_nat((span / 8 * inner : U32))
  packed_buffer_lengths(input,rows,end,inner,output_start,span,weights,bias,output,pl,tl,Packing.pack(window,planes,count,pack_start,pairs,tiles),
    PackingCertificate.pack_certificates(window,planes,count,pack_start,pairs,tiles,pc,tc))

def blocks_lengths(-input: Nat,remaining: Nat,+work: Pipeline.WindowWork,+planes: Packing.Planes,+pack_start: U32,+output_start: U32,+padded: U32,+span: U32,
  -state: Pipeline.BlockState,observation: BlockObservation<state>) ->
  {lengths(input,Pipeline.blocks_window(remaining,work,planes,pack_start,output_start,padded,span,state)) == lengths(input,state) : Lengths}:
  match remaining:
    case 0n: {==}
    case 1n+rest:
      BlockObservation{+model,+equation} = observation
      %equation : {lengths(input,Pipeline.blocks_window(1n+rest,work,planes,pack_start,output_start,padded,span,_)) == lengths(input,_) : Lengths}
      Equal.trans(Lengths,lengths(input,Pipeline.blocks_window(rest,work,planes,(pack_start + span : U32),(output_start + span : U32),padded,span,Pipeline.block_step_window(work,planes,pack_start,output_start,U32.min(span,(padded - output_start : U32)),pack_blocks(model)))),
        lengths(input,Pipeline.block_step_window(work,planes,pack_start,output_start,U32.min(span,(padded - output_start : U32)),pack_blocks(model))),lengths(input,pack_blocks(model)),
        blocks_lengths(input,rest,work,planes,(pack_start + span : U32),(output_start + span : U32),padded,span,Pipeline.block_step_window(work,planes,pack_start,output_start,U32.min(span,(padded - output_start : U32)),pack_blocks(model)),observe_blocks(Pipeline.block_step_window(work,planes,pack_start,output_start,U32.min(span,(padded - output_start : U32)),pack_blocks(model)))),
        step_lengths(input,work,planes,pack_start,output_start,U32.min(span,(padded - output_start : U32)),pack_blocks(model)))

def kept_blocks_lengths(-input: Storage.Buffer<Tile.Tile8>,state: Pipeline.BlockState) ->
  {kept_lengths(Pipeline.kept_from_blocks(input,state)) == lengths(buffer_length(~Tile.Tile8,input),state) : Lengths}:
  Pipeline.BlockState{planes,tiles,weights,bias,output} = state
  {==}

def model_input(model: KeptModel) -> Buffer.Model<Tile.Tile8>:
  KeptModel{input,weights,bias,planes,tiles,output} = model
  input

def model_lengths(model: KeptModel) -> Lengths:
  KeptModel{input,weights,bias,planes,tiles,output} = model
  Lengths{Buffer.length(~Tile.Tile8,input),Buffer.length(~Tile.Tile16,planes),Buffer.length(~Tile.Tile8,tiles),Buffer.length(~Tile.Tile8,weights),
    Buffer.length(~Tile.Tile8,bias),Buffer.length(~Tile.Tile8,output)}

def input_length(lengths: Lengths) -> Nat:
  Lengths{input,planes,tiles,weights,bias,output} = lengths
  input

def model_input_length(+model: KeptModel) -> {Buffer.length(~Tile.Tile8,model_input(model)) == input_length(model_lengths(model)) : Nat}:
  KeptModel{input,weights,bias,planes,tiles,output} = model
  {==}

def footprint_length(~Element: Data,+model: Buffer.Model<Element>) -> {length(footprint(~Element,Buffer.pack(~Element,model))) == Buffer.length(~Element,model) : Nat}:
  Buffer.Model{+tree,+length,Certificate.Certificate{+depth,+equation}} = model
  {==}

# A write keeps the depth and the length of a buffer.
def write_footprint(~Element: Data,+index: U32,value: Element,+model: Buffer.Model<Element>) ->
  {footprint(~Element,Storage.write(~Element,index,value,Buffer.pack(~Element,model))) == footprint(~Element,Buffer.pack(~Element,model)) : Footprint}:
  Buffer.Model{+tree,+length,Certificate.Certificate{+depth,+equation}} = model
  {==}

# A fresh buffer, remodeled, holds as many cells as it has.
def allocated_length(~Element: Data,+fill: Element,+count: U32) ->
  {Buffer.content_length(~Element,Buffer.content(~Element,Buffer.remodel(~Element,Tensors.buffer(~Element,fill,count)))) == U32.to_nat(count) : Nat}:
  +start = Buffer.remodel(~Element,Tensors.buffer(~Element,fill,count))
  +allocation = {Footprint{Tensors.depth_for(count),U32.to_nat(count)} : Footprint}
  Equal.trans(Nat,Buffer.content_length(~Element,Buffer.content(~Element,start)),Buffer.length(~Element,start),U32.to_nat(count),
    Buffer.content_length_of(~Element,start),
    Equal.trans(Nat,Buffer.length(~Element,start),length(footprint(~Element,Buffer.pack(~Element,start))),U32.to_nat(count),
      Equal.sym(Nat,length(footprint(~Element,Buffer.pack(~Element,start))),Buffer.length(~Element,start),footprint_length(~Element,start)),
      Equal.cong(Footprint,Nat,value => length(value),footprint(~Element,Buffer.pack(~Element,start)),allocation,
        Equal.trans(Footprint,footprint(~Element,Buffer.pack(~Element,start)),footprint(~Element,Tensors.buffer(~Element,fill,count)),allocation,
          Equal.cong(Storage.Buffer<Element>,Footprint,buffer => footprint(~Element,buffer),Buffer.pack(~Element,start),Tensors.buffer(~Element,fill,count),
            Buffer.remodeled(~Element,Tensors.buffer(~Element,fill,count))),
          allocated(~Element,fill,count)))))

def packed_kept_lengths(+model: KeptModel) -> {kept_lengths(pack_kept(model)) == model_lengths(model) : Lengths}:
  KeptModel{+input,+weights,+bias,+planes,+tiles,+output} = model
  Buffer.Model{+it,+il,+ic} = input
  Buffer.Model{+wt,+wl,+wc} = weights
  Buffer.Model{+bt,+bl,+bc} = bias
  Buffer.Model{+pt,+pl,+pc} = planes
  Buffer.Model{+tt,+tl,+tc} = tiles
  Buffer.Model{+ot,+ol,+oc} = output
  {==}

# The blocks of a call keep the lengths of the planes and the input the
# transpose gives back.
def kept_window_lengths(-input: Storage.Buffer<Tile.Tile8>,+remaining: Nat,+work: Pipeline.WindowWork,+planes: Packing.Planes,+pack_start: U32,+output_start: U32,
  +padded: U32,+span: U32,-state: Pipeline.BlockState,observation: BlockObservation<state>) ->
  {kept_lengths(Pipeline.kept_from_blocks(input,Pipeline.blocks_window(remaining,work,planes,pack_start,output_start,padded,span,state))) ==
    lengths(buffer_length(~Tile.Tile8,input),state) : Lengths}:
  BlockObservation{+model,+equation} = observation
  %equation : {kept_lengths(Pipeline.kept_from_blocks(input,Pipeline.blocks_window(remaining,work,planes,pack_start,output_start,padded,span,_))) ==
    lengths(buffer_length(~Tile.Tile8,input),_) : Lengths}
  Equal.trans(Lengths,kept_lengths(Pipeline.kept_from_blocks(input,Pipeline.blocks_window(remaining,work,planes,pack_start,output_start,padded,span,pack_blocks(model)))),
    lengths(buffer_length(~Tile.Tile8,input),Pipeline.blocks_window(remaining,work,planes,pack_start,output_start,padded,span,pack_blocks(model))),
    lengths(buffer_length(~Tile.Tile8,input),pack_blocks(model)),
    kept_blocks_lengths(input,Pipeline.blocks_window(remaining,work,planes,pack_start,output_start,padded,span,pack_blocks(model))),
    blocks_lengths(buffer_length(~Tile.Tile8,input),remaining,work,planes,pack_start,output_start,padded,span,pack_blocks(model),observe_blocks(pack_blocks(model))))

def ready_lengths(+work: Pipeline.WindowWork,+planes: Packing.Planes,+pack_start: U32,+offset: U32,+padded: U32,-weights: Storage.Buffer<Tile.Tile8>,
  -bias: Storage.Buffer<Tile.Tile8>,-tiles: Storage.Buffer<Tile.Tile8>,-output: Storage.Buffer<Tile.Tile8>,-input: Storage.Buffer<Tile.Tile8>,-pairs: Storage.Buffer<Tile.Tile16>,
  pairs_seen: Buffer.Observation<Tile.Tile16,pairs>,weight_seen: Buffer.Observation<Tile.Tile8,weights>,bias_seen: Buffer.Observation<Tile.Tile8,bias>,
  tile_seen: Buffer.Observation<Tile.Tile8,tiles>,output_seen: Buffer.Observation<Tile.Tile8,output>) ->
  {kept_lengths(Pipeline.planes_ready(work,planes,pack_start,offset,padded,weights,bias,tiles,output,(input,pairs))) ==
    lengths(buffer_length(~Tile.Tile8,input),Pipeline.BlockState{pairs,tiles,weights,bias,output}) : Lengths}:
  Pipeline.WindowWork{+rows,+end,Packing.Stencil{+base,+h,+w,+kr,+kc,+s,+ro,+co,+ow,+inner,+groups}} = work
  +work = {Pipeline.WindowWork{rows,end,Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,inner,groups}} : Pipeline.WindowWork}
  +span = Pipeline.block_positions(inner)
  kept_window_lengths(input,U32.to_nat(((padded - offset + span - 1) / span : U32)),work,planes,pack_start,offset,padded,span,
    Pipeline.BlockState{pairs,tiles,weights,bias,output},observed_blocks(pairs,tiles,weights,bias,output,pairs_seen,tile_seen,weight_seen,bias_seen,output_seen))

def transposed_lengths(+work: Pipeline.WindowWork,+planes: Packing.Planes,+pack_start: U32,+offset: U32,+padded: U32,-weights: Storage.Buffer<Tile.Tile8>,
  -bias: Storage.Buffer<Tile.Tile8>,-tiles: Storage.Buffer<Tile.Tile8>,-output: Storage.Buffer<Tile.Tile8>,+input_length: Nat,+pairs_length: Nat,
  result: Array<Tile.Tile8> & Array<Tile.Tile16>,certificates: TraversalCertificate.Certificates<Array<Tile.Tile8>,Tile.Tile16,Certificate.family(~Tile.Tile8),result>,
  weight_seen: Buffer.Observation<Tile.Tile8,weights>,bias_seen: Buffer.Observation<Tile.Tile8,bias>,tile_seen: Buffer.Observation<Tile.Tile8,tiles>,
  output_seen: Buffer.Observation<Tile.Tile8,output>) ->
  {kept_lengths(Pipeline.planes_ready(work,planes,pack_start,offset,padded,weights,bias,tiles,output,
      PackingBuffer.buffers(~Tile.Tile8,~Tile.Tile16,input_length,pairs_length,result,certificates))) ==
    Lengths{input_length,pairs_length,buffer_length(~Tile.Tile8,tiles),buffer_length(~Tile.Tile8,weights),buffer_length(~Tile.Tile8,bias),buffer_length(~Tile.Tile8,output)} :
    Lengths}:
  (input,pairs) = result
  TraversalCertificate.Certificates{ic,pc} = certificates
  ready_lengths(work,planes,pack_start,offset,padded,weights,bias,tiles,output,Storage.Buffer{input,input_length,ic},Storage.Buffer{pairs,pairs_length,pc},
    Buffer.observe(~Tile.Tile16,Storage.Buffer{pairs,pairs_length,pc}),
    weight_seen,bias_seen,tile_seen,output_seen)

def call_lengths(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32,+input: Buffer.Model<Tile.Tile8>,+planes: Buffer.Model<Tile.Tile16>,-weights: Storage.Buffer<Tile.Tile8>,
  -bias: Storage.Buffer<Tile.Tile8>,-tiles: Storage.Buffer<Tile.Tile8>,-output: Storage.Buffer<Tile.Tile8>,
  weight_seen: Buffer.Observation<Tile.Tile8,weights>,bias_seen: Buffer.Observation<Tile.Tile8,bias>,tile_seen: Buffer.Observation<Tile.Tile8,tiles>,
  output_seen: Buffer.Observation<Tile.Tile8,output>) ->
  {kept_lengths(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,input),weights,bias,Buffer.pack(~Tile.Tile16,planes),tiles,output)) ==
    Lengths{buffer_length(~Tile.Tile8,Buffer.pack(~Tile.Tile8,input)),buffer_length(~Tile.Tile16,Buffer.pack(~Tile.Tile16,planes)),buffer_length(~Tile.Tile8,tiles),buffer_length(~Tile.Tile8,weights),buffer_length(~Tile.Tile8,bias),
      buffer_length(~Tile.Tile8,output)} : Lengths}:
  Pipeline.WindowWork{+rows,+end,+window} = work
  Buffer.Model{+it,+il,+ic} = input
  Buffer.Model{+pt,+pl,+pc} = planes
  +work = {Pipeline.WindowWork{rows,end,window} : Pipeline.WindowWork}
  +padded = (offset + Window.padded((end - offset : U32)) : U32)
  +geometry = Packing.planes_of(window,pack_start,(padded - offset : U32))
  transposed_lengths(work,geometry,pack_start,offset,padded,weights,bias,tiles,output,il,pl,Packing.transpose(window,geometry,Tree.pack(Tile.Tile8,it),Tree.pack(Tile.Tile16,pt)),
    PackingCertificate.transpose_certificates(window,geometry,Tree.pack(Tile.Tile8,it),Tree.pack(Tile.Tile16,pt),ic,pc),weight_seen,bias_seen,tile_seen,output_seen)

# A call keeps the lengths of all its buffers.
def blocked_lengths(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32,-input: Storage.Buffer<Tile.Tile8>,-weights: Storage.Buffer<Tile.Tile8>,-bias: Storage.Buffer<Tile.Tile8>,
  -planes: Storage.Buffer<Tile.Tile16>,-tiles: Storage.Buffer<Tile.Tile8>,-output: Storage.Buffer<Tile.Tile8>,
  input_seen: Buffer.Observation<Tile.Tile8,input>,weight_seen: Buffer.Observation<Tile.Tile8,weights>,bias_seen: Buffer.Observation<Tile.Tile8,bias>,
  plane_seen: Buffer.Observation<Tile.Tile16,planes>,tile_seen: Buffer.Observation<Tile.Tile8,tiles>,output_seen: Buffer.Observation<Tile.Tile8,output>) ->
  {kept_lengths(Pipeline.blocked_into(work,pack_start,offset,input,weights,bias,planes,tiles,output)) ==
    Lengths{buffer_length(~Tile.Tile8,input),buffer_length(~Tile.Tile16,planes),buffer_length(~Tile.Tile8,tiles),buffer_length(~Tile.Tile8,weights),buffer_length(~Tile.Tile8,bias),
      buffer_length(~Tile.Tile8,output)} : Lengths}:
  Buffer.Observation{+im,ie} = input_seen
  Buffer.Observation{+pm,pe} = plane_seen
  %ie : {kept_lengths(Pipeline.blocked_into(work,pack_start,offset,_,weights,bias,planes,tiles,output)) ==
    Lengths{buffer_length(~Tile.Tile8,_),buffer_length(~Tile.Tile16,planes),buffer_length(~Tile.Tile8,tiles),buffer_length(~Tile.Tile8,weights),buffer_length(~Tile.Tile8,bias),
      buffer_length(~Tile.Tile8,output)} : Lengths}
  %pe : {kept_lengths(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,im),weights,bias,_,tiles,output)) ==
    Lengths{buffer_length(~Tile.Tile8,Buffer.pack(~Tile.Tile8,im)),buffer_length(~Tile.Tile16,_),buffer_length(~Tile.Tile8,tiles),buffer_length(~Tile.Tile8,weights),
      buffer_length(~Tile.Tile8,bias),buffer_length(~Tile.Tile8,output)} : Lengths}
  call_lengths(work,pack_start,offset,im,pm,weights,bias,tiles,output,weight_seen,bias_seen,tile_seen,output_seen)

def into_kept_lengths(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32,-input: Storage.Buffer<Tile.Tile8>,-weights: Storage.Buffer<Tile.Tile8>,-bias: Storage.Buffer<Tile.Tile8>,
  -planes: Storage.Buffer<Tile.Tile16>,-tiles: Storage.Buffer<Tile.Tile8>,-output: Storage.Buffer<Tile.Tile8>,
  input_seen: Buffer.Observation<Tile.Tile8,input>,weight_seen: Buffer.Observation<Tile.Tile8,weights>,bias_seen: Buffer.Observation<Tile.Tile8,bias>,
  plane_seen: Buffer.Observation<Tile.Tile16,planes>,tile_seen: Buffer.Observation<Tile.Tile8,tiles>,output_seen: Buffer.Observation<Tile.Tile8,output>,
  +observation: KeptObservation<Pipeline.blocked_into(work,pack_start,offset,input,weights,bias,planes,tiles,output)>) ->
  {model_lengths(kept_model(Pipeline.blocked_into(work,pack_start,offset,input,weights,bias,planes,tiles,output),observation)) ==
    Lengths{buffer_length(~Tile.Tile8,input),buffer_length(~Tile.Tile16,planes),buffer_length(~Tile.Tile8,tiles),buffer_length(~Tile.Tile8,weights),buffer_length(~Tile.Tile8,bias),
      buffer_length(~Tile.Tile8,output)} : Lengths}:
  KeptObservation{+model,+equation} = observation
  -wanted = {Lengths{buffer_length(~Tile.Tile8,input),buffer_length(~Tile.Tile16,planes),buffer_length(~Tile.Tile8,tiles),buffer_length(~Tile.Tile8,weights),
    buffer_length(~Tile.Tile8,bias),buffer_length(~Tile.Tile8,output)} : Lengths}
  Equal.trans(Lengths,model_lengths(model),kept_lengths(pack_kept(model)),wanted,
    Equal.sym(Lengths,kept_lengths(pack_kept(model)),model_lengths(model),packed_kept_lengths(model)),
    %Equal.sym(Pipeline.Kept,pack_kept(model),Pipeline.blocked_into(work,pack_start,offset,input,weights,bias,planes,tiles,output),equation) : {kept_lengths(_) == wanted : Lengths}
    blocked_lengths(work,pack_start,offset,input,weights,bias,planes,tiles,output,input_seen,weight_seen,bias_seen,plane_seen,tile_seen,output_seen))
