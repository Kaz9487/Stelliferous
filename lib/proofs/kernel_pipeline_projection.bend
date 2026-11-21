# Erasing buffer certificates exposes the same transpose, packing and GEMM
# block loop of a call.
import Base
import ./kernel_pipeline_activation.bend as ActivationProofs
import ./kernel_pipeline_buffer_shape.bend as BufferShape
import ./storage_buffer_model.bend as Buffer
import ./kernel_packing_certificate.bend as PackingCertificate
import ../kernel_pipeline.bend as Pipeline
import ../kernel_gemm.bend as Gemm
import ../kernel_gemm_buffer.bend as GemmBuffer
import ../kernel_spatial_packing.bend as Packing
import ../kernel_packing_buffer.bend as PackingBuffer
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ../kernel_window.bend as Window
import ./kernel_gemm_certificate.bend as GemmCertificate
import ./traversal_certificate.bend as TraversalCertificate
import ./storage_certificate.bend as Certificate
import ./storage_tree_model.bend as Tree

# The arrays of the blocks of a call: the planes, the packed tiles and the
# operands of the tile products.
type State is Type:
  State{pairs: Array<Tile.Tile16>,tiles: Array<Tile.Tile8>,weights: Array<Tile.Tile8>,bias: Array<Tile.Tile8>,output: Array<Tile.Tile8>}

# The arrays a call gives back: its input and the arrays of its blocks.
type Called is Type:
  Called{input: Array<Tile.Tile8>,state: State}

def project(state: Pipeline.BlockState) -> State:
  Pipeline.BlockState{pairs,tiles,weights,bias,output} = state
  State{ActivationProofs.storage(~Tile.Tile16,pairs),ActivationProofs.storage(~Tile.Tile8,tiles),ActivationProofs.storage(~Tile.Tile8,weights),
    ActivationProofs.storage(~Tile.Tile8,bias),ActivationProofs.storage(~Tile.Tile8,output)}

# The operands of a tile product next to the planes.
def kept(pairs: Array<Tile.Tile16>,operands: Gemm.CellOperands) -> State:
  Gemm.CellOperands{bias,Gemm.Operands{tiles,weights,unused_bias,output}} = operands
  State{pairs,tiles,weights,bias,output}

def kept_projection(pairs: Storage.Buffer<Tile.Tile16>,operands: Gemm.CellOperands,+row_length: Nat,+lane_length: Nat,+bias_length: Nat,+length: Nat,
  certificates: GemmCertificate.CellCertified<operands>) ->
  {project(Pipeline.block_kept(pairs,GemmBuffer.cells_kept(operands,row_length,lane_length,bias_length,length,certificates))) ==
    kept(ActivationProofs.storage(~Tile.Tile16,pairs),operands) : State}:
  Gemm.CellOperands{bias,Gemm.Operands{tiles,weights,unused_bias,output}} = operands
  GemmCertificate.CellCertified{bias_certificate,tile_certificate,weight_certificate,certificate} = certificates
  {==}

def packed(+rows: U32,+end: U32,+inner: U32,+start: U32,+span: U32,weights: Array<Tile.Tile8>,bias: Array<Tile.Tile8>,output: Array<Tile.Tile8>,
  pair: Array<Tile.Tile16> & Array<Tile.Tile8>) -> State:
  (pairs,tiles) = pair
  +matrix = {Gemm.TileMatrix{rows,span,inner,start,end} : Gemm.TileMatrix}
  kept(pairs,Gemm.cell_run(U32.to_nat(Gemm.tile_jobs(matrix)),matrix,Gemm.CellOperands{bias,Gemm.Operands{tiles,weights,Unit{},output}}))

def packed_projection(+rows: U32,+end: U32,+inner: U32,+start: U32,+span: U32,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>,
  pairs: Storage.Buffer<Tile.Tile16>,tiles: Storage.Buffer<Tile.Tile8>) ->
  {project(Pipeline.block_packed(rows,end,inner,start,span,weights,bias,output,(pairs,tiles))) ==
    packed(rows,end,inner,start,span,ActivationProofs.storage(~Tile.Tile8,weights),ActivationProofs.storage(~Tile.Tile8,bias),ActivationProofs.storage(~Tile.Tile8,output),
      (ActivationProofs.storage(~Tile.Tile16,pairs),ActivationProofs.storage(~Tile.Tile8,tiles))) : State}:
  Storage.Buffer{weights,wl,wc} = weights
  Storage.Buffer{bias,bl,bc} = bias
  Storage.Buffer{output,ol,oc} = output
  Storage.Buffer{tiles,tl,tc} = tiles
  +matrix = {Gemm.TileMatrix{rows,span,inner,start,end} : Gemm.TileMatrix}
  +count = U32.to_nat(Gemm.tile_jobs(matrix))
  kept_projection(pairs,Gemm.cell_run(count,matrix,Gemm.CellOperands{bias,Gemm.Operands{tiles,weights,Unit{},output}}),tl,wl,bl,ol,
    GemmCertificate.cell_certificates(count,matrix,tiles,weights,bias,Unit{},output,tc,wc,bc,oc))

def packing_projection(+rows: U32,+end: U32,+inner: U32,+start: U32,+span: U32,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>,
  +pairs_length: Nat,+tile_length: Nat,result: Array<Tile.Tile16> & Array<Tile.Tile8>,
  certificates: TraversalCertificate.Certificates<Array<Tile.Tile16>,Tile.Tile8,Certificate.family(~Tile.Tile16),result>) ->
  {project(Pipeline.block_packed(rows,end,inner,start,span,weights,bias,output,PackingBuffer.buffers(~Tile.Tile16,~Tile.Tile8,pairs_length,tile_length,result,certificates))) ==
    packed(rows,end,inner,start,span,ActivationProofs.storage(~Tile.Tile8,weights),ActivationProofs.storage(~Tile.Tile8,bias),ActivationProofs.storage(~Tile.Tile8,output),result) :
    State}:
  (pairs,tiles) = result
  TraversalCertificate.Certificates{pc,tc} = certificates
  packed_projection(rows,end,inner,start,span,weights,bias,output,Storage.Buffer{pairs,pairs_length,pc},Storage.Buffer{tiles,tile_length,tc})

# A block packs its tiles from the planes and multiplies them.
def step_window(work: Pipeline.WindowWork,+planes: Packing.Planes,+pack_start: U32,+start: U32,+span: U32,state: State) -> State:
  Pipeline.WindowWork{rows,end,+window} = work
  Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,+inner,groups} = window
  State{pairs,tiles,weights,bias,output} = state
  packed(rows,end,inner,start,span,weights,bias,output,Packing.pack(window,planes,U32.to_nat((span / 8 * inner : U32)),pack_start,pairs,tiles))

def window_step_projection(+work: Pipeline.WindowWork,+planes: Packing.Planes,+pack_start: U32,+start: U32,+span: U32,state: Pipeline.BlockState) ->
  {project(Pipeline.block_step_window(work,planes,pack_start,start,span,state)) == step_window(work,planes,pack_start,start,span,project(state)) : State}:
  Pipeline.WindowWork{+rows,+end,+window} = work
  Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,+inner,groups} = window
  Pipeline.BlockState{Storage.Buffer{pairs,pl,pc},Storage.Buffer{tiles,tl,tc},weights,bias,output} = state
  packing_projection(rows,end,inner,start,span,weights,bias,output,pl,tl,Packing.pack(window,planes,U32.to_nat((span / 8 * inner : U32)),pack_start,pairs,tiles),
    PackingCertificate.pack_certificates(window,planes,U32.to_nat((span / 8 * inner : U32)),pack_start,pairs,tiles,pc,tc))

def blocks_window(remaining: Nat,+work: Pipeline.WindowWork,+planes: Packing.Planes,+pack_start: U32,+start: U32,+padded: U32,+span: U32,state: State) -> State:
  match remaining:
    case 0n: state
    case 1n+rest:
      blocks_window(rest,work,planes,(pack_start + span : U32),(start + span : U32),padded,span,step_window(work,planes,pack_start,start,U32.min(span,(padded - start : U32)),state))

def window_blocks_projection(+remaining: Nat,+work: Pipeline.WindowWork,+planes: Packing.Planes,+pack_start: U32,+start: U32,+padded: U32,+span: U32,-state: Pipeline.BlockState,
  observation: BufferShape.BlockObservation<state>) ->
  {project(Pipeline.blocks_window(remaining,work,planes,pack_start,start,padded,span,state)) == blocks_window(remaining,work,planes,pack_start,start,padded,span,project(state)) :
    State}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      BufferShape.BlockObservation{+model,equation} = observation
      %equation : {project(Pipeline.blocks_window(1n+rest,work,planes,pack_start,start,padded,span,_)) == blocks_window(1n+rest,work,planes,pack_start,start,padded,span,project(_)) :
        State}
      %window_step_projection(work,planes,pack_start,start,U32.min(span,(padded - start : U32)),BufferShape.pack_blocks(model)) :
        {project(Pipeline.blocks_window(rest,work,planes,(pack_start + span : U32),(start + span : U32),padded,span,
            Pipeline.block_step_window(work,planes,pack_start,start,U32.min(span,(padded - start : U32)),BufferShape.pack_blocks(model)))) ==
          blocks_window(rest,work,planes,(pack_start + span : U32),(start + span : U32),padded,span,_) : State}
      window_blocks_projection(rest,work,planes,(pack_start + span : U32),(start + span : U32),padded,span,
        Pipeline.block_step_window(work,planes,pack_start,start,U32.min(span,(padded - start : U32)),BufferShape.pack_blocks(model)),
        BufferShape.observe_blocks(Pipeline.block_step_window(work,planes,pack_start,start,U32.min(span,(padded - start : U32)),BufferShape.pack_blocks(model))))

def project_kept(kept: Pipeline.Kept) -> Called:
  Pipeline.Kept{input,weights,bias,planes,tiles,output} = kept
  Called{ActivationProofs.storage(~Tile.Tile8,input),State{ActivationProofs.storage(~Tile.Tile16,planes),ActivationProofs.storage(~Tile.Tile8,tiles),
    ActivationProofs.storage(~Tile.Tile8,weights),ActivationProofs.storage(~Tile.Tile8,bias),ActivationProofs.storage(~Tile.Tile8,output)}}

def kept_blocks_projection(-input: Storage.Buffer<Tile.Tile8>,state: Pipeline.BlockState) ->
  {project_kept(Pipeline.kept_from_blocks(input,state)) == Called{ActivationProofs.storage(~Tile.Tile8,input),project(state)} : Called}:
  Pipeline.BlockState{pairs,tiles,weights,bias,output} = state
  {==}

# The blocks of a call after the transpose: positions [offset, end) from the
# packing start on.
def blocks_of(+work: Pipeline.WindowWork,+planes: Packing.Planes,+pack_start: U32,+offset: U32,+padded: U32,tiles: Array<Tile.Tile8>,weights: Array<Tile.Tile8>,
  bias: Array<Tile.Tile8>,output: Array<Tile.Tile8>,transposed: Array<Tile.Tile8> & Array<Tile.Tile16>) -> Called:
  Pipeline.WindowWork{rows,end,Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,+inner,groups}} = work
  (input,pairs) = transposed
  +span = Pipeline.block_positions(inner)
  Called{input,blocks_window(U32.to_nat(((padded - offset + span - 1) / span : U32)),work,planes,pack_start,offset,padded,span,State{pairs,tiles,weights,bias,output})}

# A call: the transpose of its input into the planes of its positions, then
# its blocks.
def into(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32,input: Array<Tile.Tile8>,weights: Array<Tile.Tile8>,bias: Array<Tile.Tile8>,pairs: Array<Tile.Tile16>,
  tiles: Array<Tile.Tile8>,output: Array<Tile.Tile8>) -> Called:
  Pipeline.WindowWork{rows,+end,+window} = work
  +padded = (offset + Window.padded((end - offset : U32)) : U32)
  +planes = Packing.planes_of(window,pack_start,(padded - offset : U32))
  blocks_of(work,planes,pack_start,offset,padded,tiles,weights,bias,output,Packing.transpose(window,planes,input,pairs))

# The blocks of a call, planes observed, project to the blocks on the arrays.
def blocks_projection(-input: Storage.Buffer<Tile.Tile8>,+work: Pipeline.WindowWork,+planes: Packing.Planes,+pack_start: U32,+offset: U32,+padded: U32,
  +weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>,+tiles: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>,-pairs: Storage.Buffer<Tile.Tile16>,
  pairs_seen: Buffer.Observation<Tile.Tile16,pairs>) ->
  {project_kept(Pipeline.kept_from_blocks(input,Pipeline.blocked_window_with(work,planes,pack_start,offset,padded,
      Pipeline.BlockState{pairs,Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile8,output)}))) ==
    blocks_of(work,planes,pack_start,offset,padded,ActivationProofs.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,tiles)),ActivationProofs.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),
      ActivationProofs.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,bias)),ActivationProofs.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,output)),
      (ActivationProofs.storage(~Tile.Tile8,input),ActivationProofs.storage(~Tile.Tile16,pairs))) : Called}:
  Pipeline.WindowWork{+rows,+end,Packing.Stencil{+base,+h,+w,+kr,+kc,+s,+ro,+co,+ow,+inner,+groups}} = work
  Buffer.Observation{+pm,pe} = pairs_seen
  +work = {Pipeline.WindowWork{rows,end,Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,inner,groups}} : Pipeline.WindowWork}
  +span = Pipeline.block_positions(inner)
  +remaining = U32.to_nat(((padded - offset + span - 1) / span : U32))
  %pe : {project_kept(Pipeline.kept_from_blocks(input,Pipeline.blocks_window(remaining,work,planes,pack_start,offset,padded,span,
      Pipeline.BlockState{_,Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile8,output)}))) ==
    Called{ActivationProofs.storage(~Tile.Tile8,input),blocks_window(remaining,work,planes,pack_start,offset,padded,span,
      State{ActivationProofs.storage(~Tile.Tile16,_),ActivationProofs.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,tiles)),ActivationProofs.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),
        ActivationProofs.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,bias)),ActivationProofs.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,output))})} : Called}
  Equal.trans(Called,project_kept(Pipeline.kept_from_blocks(input,Pipeline.blocks_window(remaining,work,planes,pack_start,offset,padded,span,
      Pipeline.BlockState{Buffer.pack(~Tile.Tile16,pm),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile8,output)}))),
    Called{ActivationProofs.storage(~Tile.Tile8,input),project(Pipeline.blocks_window(remaining,work,planes,pack_start,offset,padded,span,
      Pipeline.BlockState{Buffer.pack(~Tile.Tile16,pm),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile8,output)}))},
    Called{ActivationProofs.storage(~Tile.Tile8,input),blocks_window(remaining,work,planes,pack_start,offset,padded,span,
      project(Pipeline.BlockState{Buffer.pack(~Tile.Tile16,pm),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile8,output)}))},
    kept_blocks_projection(input,Pipeline.blocks_window(remaining,work,planes,pack_start,offset,padded,span,
      Pipeline.BlockState{Buffer.pack(~Tile.Tile16,pm),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile8,output)})),
    Equal.cong(State,Called,blocks => Called{ActivationProofs.storage(~Tile.Tile8,input),blocks},
      project(Pipeline.blocks_window(remaining,work,planes,pack_start,offset,padded,span,
        Pipeline.BlockState{Buffer.pack(~Tile.Tile16,pm),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile8,output)})),
      blocks_window(remaining,work,planes,pack_start,offset,padded,span,
        project(Pipeline.BlockState{Buffer.pack(~Tile.Tile16,pm),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile8,output)})),
      window_blocks_projection(remaining,work,planes,pack_start,offset,padded,span,
        Pipeline.BlockState{Buffer.pack(~Tile.Tile16,pm),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile8,output)},
        BufferShape.observed_blocks(Buffer.pack(~Tile.Tile16,pm),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile8,output),
          Buffer.Observation{pm,{==}},Buffer.Observation{tiles,{==}},Buffer.Observation{weights,{==}},Buffer.Observation{bias,{==}},Buffer.Observation{output,{==}}))))

def ready_projection(+work: Pipeline.WindowWork,+planes: Packing.Planes,+pack_start: U32,+offset: U32,+padded: U32,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>,
  +tiles: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>,+input_length: Nat,+pairs_length: Nat,result: Array<Tile.Tile8> & Array<Tile.Tile16>,
  certificates: TraversalCertificate.Certificates<Array<Tile.Tile8>,Tile.Tile16,Certificate.family(~Tile.Tile8),result>) ->
  {project_kept(Pipeline.planes_ready(work,planes,pack_start,offset,padded,Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),Buffer.pack(~Tile.Tile8,tiles),
      Buffer.pack(~Tile.Tile8,output),PackingBuffer.buffers(~Tile.Tile8,~Tile.Tile16,input_length,pairs_length,result,certificates))) ==
    blocks_of(work,planes,pack_start,offset,padded,ActivationProofs.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,tiles)),ActivationProofs.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),
      ActivationProofs.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,bias)),ActivationProofs.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,output)),result) : Called}:
  (input,pairs) = result
  TraversalCertificate.Certificates{ic,pc} = certificates
  blocks_projection(Storage.Buffer{input,input_length,ic},work,planes,pack_start,offset,padded,weights,bias,tiles,output,Storage.Buffer{pairs,pairs_length,pc},
    Buffer.observe(~Tile.Tile16,Storage.Buffer{pairs,pairs_length,pc}))

# The arrays a call gives back are those of the call on the arrays of its
# buffers.
def into_projection(+work: Pipeline.WindowWork,+pack_start: U32,+offset: U32,+input: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<Tile.Tile8>,
  +planes: Buffer.Model<Tile.Tile16>,+tiles: Buffer.Model<Tile.Tile8>,+output: Buffer.Model<Tile.Tile8>) ->
  {project_kept(Pipeline.blocked_into(work,pack_start,offset,Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~Tile.Tile8,bias),
      Buffer.pack(~Tile.Tile16,planes),Buffer.pack(~Tile.Tile8,tiles),Buffer.pack(~Tile.Tile8,output))) ==
    into(work,pack_start,offset,ActivationProofs.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,input)),ActivationProofs.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),
      ActivationProofs.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,bias)),ActivationProofs.storage(~Tile.Tile16,Buffer.pack(~Tile.Tile16,planes)),
      ActivationProofs.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,tiles)),ActivationProofs.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,output))) : Called}:
  Pipeline.WindowWork{+rows,+end,+window} = work
  Buffer.Model{+it,+il,+ic} = input
  Buffer.Model{+pt,+pl,+pc} = planes
  +work = {Pipeline.WindowWork{rows,end,window} : Pipeline.WindowWork}
  +padded = (offset + Window.padded((end - offset : U32)) : U32)
  +geometry = Packing.planes_of(window,pack_start,(padded - offset : U32))
  ready_projection(work,geometry,pack_start,offset,padded,weights,bias,tiles,output,il,pl,Packing.transpose(window,geometry,Tree.pack(Tile.Tile8,it),Tree.pack(Tile.Tile16,pt)),
    PackingCertificate.transpose_certificates(window,geometry,Tree.pack(Tile.Tile8,it),Tree.pack(Tile.Tile16,pt),ic,pc))
