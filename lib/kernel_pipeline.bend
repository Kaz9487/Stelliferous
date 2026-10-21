# One convolution leaf owns its input, packs it block by block, and computes
# before returning. The caller validates dimensions and chooses the parts.
import Base
import ./kernel_gemm_buffer.bend as GemmBuffer
import ./kernel_tile.bend as Tile
import ./kernel_spatial_packing.bend as Packing
import ./kernel_packing_buffer.bend as PackingBuffer
import ./storage_buffer.bend as Storage
import ./tensor.bend as Tensors
import ./kernel_window.bend as KernelWindow
import ./traversal_array.bend as Traversal
import ./math_activation.bend as Activations
import ./proofs/traversal_certificate.bend as TraversalCertificate
import ./tensor_operations.bend as Operations
import ./tensor_gelu.bend as Gelu

type Work is Data:
  Work{rows: U32,height: U32,width: U32,kernel: U32,stride: U32,padding: U32,output_width: U32,positions: U32,inner: U32}

# The matrix and its source stencil are independent. Forward convolution is
# the square specialization; an adjoint phase uses rectangular taps and origins.
type WindowWork is Data:
  WindowWork{rows: U32,positions: U32,window: Packing.Stencil}

def square_work(work: Work) -> WindowWork:
  Work{rows,h,w,k,s,p,ow,positions,inner} = work
  WindowWork{rows,positions,Packing.square(h,w,k,s,p,ow,inner)}

def activated_value(~activation: F32 -> F32,context: Unit,value: F32) -> F32:
  activation(value)

def mapped(~Context: Data,~operation: Context -> F32 -> F32,+count: U32,+context: Context,buffer: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  Storage.Buffer{storage,length,certificate} = buffer
  Storage.Buffer{Traversal.map_in_place(~F32,~Context,~operation,U32.to_nat(count),0,context,storage),length,
    TraversalCertificate.map_in_place(~F32,~Context,~operation,U32.to_nat(count),0,context,storage,certificate)}

def leaky_relu_value(slope: F32,value: F32) -> F32:
  Activations.leaky_relu_value(value,slope)

def unit_mapped(~operation: F32 -> F32,+count: U32,buffer: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  mapped(~Unit,~activated_value(~operation),count,Unit{},buffer)

def unit_lanes(~operation: F32 -> F32,+count: U32,buffer: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  Operations.lanes(~Unit,~activated_value(~operation),U32.to_nat(count),0,Unit{},buffer)

# One tight pass over each finished leaf while it is still in cache; applying
# the activation inside the store measured about twice as slow. The built-in
# is chosen once per leaf, so each pass is specialized; the identity skips it.
# Sigmoid, SiLU, tanh and GELU pass their values through tiles.
def activated(~custom: F32 -> F32,activation: Activations.Activation,+count: U32,buffer: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  match activation:
    case Activations.Identity{}: buffer
    case Activations.ReLU{}: unit_mapped(~Activations.relu_value,count,buffer)
    case Activations.ReLU6{}: unit_mapped(~Activations.relu6_value,count,buffer)
    case Activations.LeakyReLU{slope}: mapped(~F32,~leaky_relu_value,count,slope,buffer)
    case Activations.Sigmoid{}: unit_lanes(~Activations.sigmoid_value,count,buffer)
    case Activations.SiLU{}: unit_lanes(~Activations.silu_value,count,buffer)
    case Activations.Tanh{}: unit_lanes(~Activations.tanh_value,count,buffer)
    case Activations.HardSigmoid{}: unit_mapped(~Activations.hard_sigmoid_value,count,buffer)
    case Activations.HardSwish{}: unit_mapped(~Activations.hard_swish_value,count,buffer)
    case Activations.GELUTanh{}: Gelu.mapped(U32.to_nat(count),0,buffer)
    case Activations.Custom{operations}: unit_mapped(~custom,count,buffer)

# One tile is 32 bytes; 8192 tiles are 256 KB.
def block_tiles() -> U32:
  8192

def block_positions(inner: U32) -> U32:
  (8 * U32.max(1,(block_tiles() / inner : U32)) : U32)

def workspace_prefix(+target: U32,+inner: U32,positions: U32) -> U32:
  U32.min((8 * U32.max(1,(target / inner : U32)) : U32),KernelWindow.padded(positions))

def workspace_tiles(+target: U32,+inner: U32,+positions: U32) -> U32:
  (workspace_prefix(target,inner,positions) / 8 * inner : U32)

type BlockState is Type:
  BlockState{input: Storage.Buffer<F32>,tiles: Storage.Buffer<Tile.Tile8>,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,output: Storage.Buffer<F32>}

type Kept is Type:
  Kept{input: Storage.Buffer<F32>,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,output: Storage.Buffer<F32>}

def kept_output(state: Kept) -> Storage.Buffer<F32>:
  Kept{input,weights,bias,output} = state
  output

def kept_from_blocks(state: BlockState) -> Kept:
  BlockState{input,tiles,weights,bias,output} = state
  Kept{input,weights,bias,output}

def kept_finished(~custom: F32 -> F32,activation: Activations.Activation,count: U32,state: Kept) -> Kept:
  Kept{input,weights,bias,output} = state
  Kept{input,weights,bias,activated(~custom,activation,count,output)}

def block_kept(input: Storage.Buffer<F32>,kept: GemmBuffer.Kept) -> BlockState:
  GemmBuffer.Kept{tiles,weights,bias,output} = kept
  BlockState{input,tiles,weights,bias,output}

def block_packed(+rows: U32,+positions: U32,inner: U32,+start: U32,+span: U32,
  weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,output: Storage.Buffer<F32>,packed: Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>) -> BlockState:
  (input,tiles) = packed
  block_kept(input,
    GemmBuffer.run_keeping(rows,tiles,weights,bias,output,span,inner,start,positions))

# Packing addresses the input window; output positions are local to the leaf.
# Each block overwrites tiles from zero and writes directly into the leaf output.
def block_step_with(~Context: Data,
  ~pack: @+context: Context -> @+count: Nat -> @+start: U32 -> Storage.Buffer<F32> -> Storage.Buffer<Tile.Tile8> -> Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>,
  rows: U32,positions: U32,+inner: U32,context: Context,pack_start: U32,output_start: U32,+span: U32,state: BlockState) -> BlockState:
  BlockState{input,tiles,weights,bias,output} = state
  block_packed(rows,positions,inner,output_start,span,weights,bias,output,
    pack(context,U32.to_nat((span / 8 * inner : U32)),pack_start,input,tiles))

def block_step_window(work: WindowWork,pack_start: U32,output_start: U32,span: U32,state: BlockState) -> BlockState:
  WindowWork{rows,positions,+window} = work
  Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,inner} = window
  block_step_with(~Packing.Stencil,~PackingBuffer.run_window,rows,positions,inner,window,pack_start,output_start,span,state)

def blocks_window(remaining: Nat,+work: WindowWork,+pack_start: U32,+output_start: U32,+padded: U32,+span: U32,state: BlockState) -> BlockState:
  match remaining:
    case 0n: state
    case 1n+rest:
      blocks_window(rest,work,(pack_start + span : U32),(output_start + span : U32),padded,span,
        block_step_window(work,pack_start,output_start,U32.min(span,(padded - output_start : U32)),state))

# Exactly one reusable tile buffer and one output buffer per leaf.
# An inner larger than the target still needs one complete eight-position group.
def blocked_window_with(+work: WindowWork,pack_start: U32,state: BlockState) -> BlockState:
  WindowWork{rows,+positions,Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,+inner}} = work
  +padded = KernelWindow.padded(positions)
  +span = block_positions(inner)
  blocks_window(U32.to_nat(((padded + span - 1) / span : U32)),work,pack_start,0,padded,span,state)

def blocked_window_keeping(~custom: F32 -> F32,+activation: Activations.Activation,+work: WindowWork,pack_start: U32,input: Storage.Buffer<F32>,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>) -> Kept:
  WindowWork{+rows,+positions,Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,+inner}} = work
  kept_finished(~custom,activation,(rows * positions : U32),kept_from_blocks(
    blocked_window_with(work,pack_start,
      BlockState{input,Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),workspace_tiles(block_tiles(),inner,positions)),weights,bias,Tensors.buffer(~F32,0.0,(rows * positions : U32))})))

def blocked(~custom: F32 -> F32,activation: Activations.Activation,work: Work,pack_start: U32,input: Storage.Buffer<F32>,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  kept_output(blocked_window_keeping(~custom,activation,square_work(work),pack_start,input,weights,bias))

# The deepest split that leaves every leaf at least one eight-position block.
def depth_limit(+padded: U32) -> U32:
  (U32.from_nat(Tensors.significant_bits(32n,(padded / 8 : U32),0n,U32.is_zero((padded / 8 : U32)))) - 1 : U32)
