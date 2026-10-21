# Erasing buffer certificates exposes the same packing and GEMM block loop.
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
import ../tensor.bend as Tensors
import ../kernel_window.bend as Window
import ../math_activation.bend as Activations
import ./kernel_gemm_certificate.bend as GemmCertificate
import ./traversal_certificate.bend as TraversalCertificate
import ./storage_certificate.bend as Certificate

type State is Type:
  State{input: Array<F32>,tiles: Array<Tile.Tile8>,weights: Array<Tile.Tile8>,bias: Array<F32>,output: Array<F32>}

def project(state: Pipeline.BlockState) -> State:
  Pipeline.BlockState{input,tiles,weights,bias,output} = state
  State{ActivationProofs.storage(~F32,input),ActivationProofs.storage(~Tile.Tile8,tiles),ActivationProofs.storage(~Tile.Tile8,weights),ActivationProofs.storage(~F32,bias),ActivationProofs.storage(~F32,output)}

def kept(input: Array<F32>,state: Gemm.Buffers) -> State:
  Gemm.Buffers{tiles,weights,bias,output} = state
  State{input,tiles,weights,bias,output}

def kept_projection(input: Storage.Buffer<F32>,state: Gemm.Buffers,+input_length: Nat,+weight_length: Nat,+bias_length: Nat,+length: Nat,
  certificates: GemmCertificate.Kept<state>) ->
  {project(Pipeline.block_kept(input,GemmBuffer.kept(state,input_length,weight_length,bias_length,length,certificates))) == kept(ActivationProofs.storage(~F32,input),state) : State}:
  Gemm.Buffers{tiles,weights,bias,output} = state
  GemmCertificate.Kept{input_certificate,weight_certificate,bias_certificate,certificate} = certificates
  {==}

def packed(+rows: U32,+positions: U32,+inner: U32,+start: U32,+span: U32,weights: Array<Tile.Tile8>,bias: Array<F32>,output: Array<F32>,pair: Array<F32> & Array<Tile.Tile8>) -> State:
  (input,tiles) = pair
  kept(input,Gemm.run_keeping(~True{},rows,tiles,weights,bias,output,span,inner,start,positions))

def packed_projection(+rows: U32,+positions: U32,+inner: U32,+start: U32,+span: U32,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,output: Storage.Buffer<F32>,input: Storage.Buffer<F32>,tiles: Storage.Buffer<Tile.Tile8>) ->
  {project(Pipeline.block_packed(rows,positions,inner,start,span,weights,bias,output,(input,tiles))) ==
    packed(rows,positions,inner,start,span,ActivationProofs.storage(~Tile.Tile8,weights),ActivationProofs.storage(~F32,bias),ActivationProofs.storage(~F32,output),(ActivationProofs.storage(~F32,input),ActivationProofs.storage(~Tile.Tile8,tiles))) : State}:
  Storage.Buffer{weights,wl,wc} = weights
  Storage.Buffer{bias,bl,bc} = bias
  Storage.Buffer{output,ol,oc} = output
  Storage.Buffer{tiles,tl,tc} = tiles
  kept_projection(input,Gemm.run_keeping(~True{},rows,tiles,weights,bias,output,span,inner,start,positions),tl,wl,bl,ol,
    GemmCertificate.kept(rows,tiles,weights,bias,output,span,inner,start,positions,tc,wc,bc,oc))

def packing_projection(+rows: U32,+positions: U32,+inner: U32,+start: U32,+span: U32,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,output: Storage.Buffer<F32>,
  +input_length: Nat,+tile_length: Nat,result: Array<F32> & Array<Tile.Tile8>,
  certificates: TraversalCertificate.Certificates<Array<F32>,Tile.Tile8,Certificate.family(~F32),result>) ->
  {project(Pipeline.block_packed(rows,positions,inner,start,span,weights,bias,output,PackingBuffer.buffers(input_length,tile_length,result,certificates))) ==
    packed(rows,positions,inner,start,span,ActivationProofs.storage(~Tile.Tile8,weights),ActivationProofs.storage(~F32,bias),ActivationProofs.storage(~F32,output),result) : State}:
  (input,tiles) = result
  TraversalCertificate.Certificates{ic,tc} = certificates
  packed_projection(rows,positions,inner,start,span,weights,bias,output,Storage.Buffer{input,input_length,ic},Storage.Buffer{tiles,tile_length,tc})

def step_window(work: Pipeline.WindowWork,+pack_start: U32,+start: U32,+span: U32,state: State) -> State:
  Pipeline.WindowWork{rows,positions,+window} = work
  Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,+inner} = window
  State{input,tiles,weights,bias,output} = state
  packed(rows,positions,inner,start,span,weights,bias,output,Packing.pack_window(window,U32.to_nat((span / 8 * inner : U32)),pack_start,input,tiles))

def window_step_projection(+work: Pipeline.WindowWork,+pack_start: U32,+start: U32,+span: U32,state: Pipeline.BlockState) ->
  {project(Pipeline.block_step_window(work,pack_start,start,span,state)) == step_window(work,pack_start,start,span,project(state)) : State}:
  Pipeline.WindowWork{+rows,+positions,+window} = work
  Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,+inner} = window
  Pipeline.BlockState{Storage.Buffer{input,il,ic},Storage.Buffer{tiles,tl,tc},weights,bias,output} = state
  packing_projection(rows,positions,inner,start,span,weights,bias,output,il,tl,Packing.pack_window(window,U32.to_nat((span / 8 * inner : U32)),pack_start,input,tiles),
    PackingCertificate.window_certificates(window,U32.to_nat((span / 8 * inner : U32)),pack_start,input,tiles,ic,tc))

def blocks_window(remaining: Nat,+work: Pipeline.WindowWork,+pack_start: U32,+start: U32,+padded: U32,+span: U32,state: State) -> State:
  match remaining:
    case 0n: state
    case 1n+rest:
      blocks_window(rest,work,(pack_start + span : U32),(start + span : U32),padded,span,step_window(work,pack_start,start,U32.min(span,(padded - start : U32)),state))

def window_blocks_projection(+remaining: Nat,+work: Pipeline.WindowWork,+pack_start: U32,+start: U32,+padded: U32,+span: U32,-state: Pipeline.BlockState,observation: BufferShape.BlockObservation<state>) ->
  {project(Pipeline.blocks_window(remaining,work,pack_start,start,padded,span,state)) == blocks_window(remaining,work,pack_start,start,padded,span,project(state)) : State}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      BufferShape.BlockObservation{+model,equation} = observation
      %equation : {project(Pipeline.blocks_window(1n+rest,work,pack_start,start,padded,span,_)) == blocks_window(1n+rest,work,pack_start,start,padded,span,project(_)) : State}
      %window_step_projection(work,pack_start,start,U32.min(span,(padded - start : U32)),BufferShape.pack_blocks(model)) :
        {project(Pipeline.blocks_window(rest,work,(pack_start + span : U32),(start + span : U32),padded,span,Pipeline.block_step_window(work,pack_start,start,U32.min(span,(padded - start : U32)),BufferShape.pack_blocks(model)))) ==
          blocks_window(rest,work,(pack_start + span : U32),(start + span : U32),padded,span,_) : State}
      window_blocks_projection(rest,work,(pack_start + span : U32),(start + span : U32),padded,span,Pipeline.block_step_window(work,pack_start,start,U32.min(span,(padded - start : U32)),BufferShape.pack_blocks(model)),
        BufferShape.observe_blocks(Pipeline.block_step_window(work,pack_start,start,U32.min(span,(padded - start : U32)),BufferShape.pack_blocks(model))))

def finish(~custom: F32 -> F32,activation: Activations.Activation,count: U32,state: State) -> Array<F32>:
  State{input,tiles,weights,bias,output} = state
  ActivationProofs.activation_array(~custom,activation,count,output)

def kept_finish_projection(~custom: F32 -> F32,+activation: Activations.Activation,+count: U32,state: Pipeline.BlockState) ->
  {ActivationProofs.storage(~F32,Pipeline.kept_output(Pipeline.kept_finished(~custom,activation,count,Pipeline.kept_from_blocks(state)))) == finish(~custom,activation,count,project(state)) : Array<F32>}:
  Pipeline.BlockState{input,tiles,weights,bias,output} = state
  ActivationProofs.activated(~custom,activation,count,output)

def allocated(~Element: Data,fill: Element,count: U32) -> Array<Element>:
  ActivationProofs.storage(~Element,Tensors.buffer(~Element,fill,count))

def run_window_with(+work: Pipeline.WindowWork,pack_start: U32,state: State) -> State:
  Pipeline.WindowWork{rows,+positions,Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,+inner}} = work
  +padded = Window.padded(positions)
  +span = Pipeline.block_positions(inner)
  blocks_window(U32.to_nat(((padded + span - 1) / span : U32)),work,pack_start,0,padded,span,state)

def run_window(~custom: F32 -> F32,+activation: Activations.Activation,+work: Pipeline.WindowWork,+pack_start: U32,input: Array<F32>,weights: Array<Tile.Tile8>,bias: Array<F32>) -> Array<F32>:
  Pipeline.WindowWork{+rows,+positions,Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,+inner}} = work
  finish(~custom,activation,(rows * positions : U32),run_window_with(work,pack_start,
    State{input,allocated(~Tile.Tile8,Tile.filled(0.0),Pipeline.workspace_tiles(Pipeline.block_tiles(),inner,positions)),weights,bias,allocated(~F32,0.0,(rows * positions : U32))}))

def run(~custom: F32 -> F32,+activation: Activations.Activation,+work: Pipeline.Work,+pack_start: U32,input: Array<F32>,weights: Array<Tile.Tile8>,bias: Array<F32>) -> Array<F32>:
  run_window(~custom,activation,Pipeline.square_work(work),pack_start,input,weights,bias)

def window_run_projection(~custom: F32 -> F32,+activation: Activations.Activation,+work: Pipeline.WindowWork,+pack_start: U32,+input: Buffer.Model<F32>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>) ->
  {ActivationProofs.storage(~F32,Pipeline.kept_output(Pipeline.blocked_window_keeping(~custom,activation,work,pack_start,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias)))) == run_window(~custom,activation,work,pack_start,ActivationProofs.storage(~F32,Buffer.pack(~F32,input)),ActivationProofs.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),ActivationProofs.storage(~F32,Buffer.pack(~F32,bias))) : Array<F32>}:
  Pipeline.WindowWork{+rows,+positions,Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,+inner}} = work
  +padded = Window.padded(positions)
  +span = Pipeline.block_positions(inner)
  +remaining = U32.to_nat(((padded + span - 1) / span : U32))
  -state = {Pipeline.BlockState{Buffer.pack(~F32,input),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),Pipeline.workspace_tiles(Pipeline.block_tiles(),inner,positions)),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),Tensors.buffer(~F32,0.0,(rows * positions : U32))} : Pipeline.BlockState}
  %window_blocks_projection(remaining,work,pack_start,0,padded,span,state,BufferShape.observe_blocks(Pipeline.BlockState{Buffer.pack(~F32,input),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),Pipeline.workspace_tiles(Pipeline.block_tiles(),inner,positions)),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),Tensors.buffer(~F32,0.0,(rows * positions : U32))})) :
    {ActivationProofs.storage(~F32,Pipeline.kept_output(Pipeline.kept_finished(~custom,activation,(rows * positions : U32),Pipeline.kept_from_blocks(Pipeline.blocks_window(remaining,work,pack_start,0,padded,span,state))))) == finish(~custom,activation,(rows * positions : U32),_) : Array<F32>}
  kept_finish_projection(~custom,activation,(rows * positions : U32),Pipeline.blocks_window(remaining,work,pack_start,0,padded,span,Pipeline.BlockState{Buffer.pack(~F32,input),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),Pipeline.workspace_tiles(Pipeline.block_tiles(),inner,positions)),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias),Tensors.buffer(~F32,0.0,(rows * positions : U32))}))

def run_projection(~custom: F32 -> F32,+activation: Activations.Activation,+work: Pipeline.Work,+pack_start: U32,+input: Buffer.Model<F32>,+weights: Buffer.Model<Tile.Tile8>,+bias: Buffer.Model<F32>) ->
  {ActivationProofs.storage(~F32,Pipeline.blocked(~custom,activation,work,pack_start,Buffer.pack(~F32,input),Buffer.pack(~Tile.Tile8,weights),Buffer.pack(~F32,bias))) == run(~custom,activation,work,pack_start,ActivationProofs.storage(~F32,Buffer.pack(~F32,input)),ActivationProofs.storage(~Tile.Tile8,Buffer.pack(~Tile.Tile8,weights)),ActivationProofs.storage(~F32,Buffer.pack(~F32,bias))) : Array<F32>}:
  window_run_projection(~custom,activation,Pipeline.square_work(work),pack_start,input,weights,bias)
