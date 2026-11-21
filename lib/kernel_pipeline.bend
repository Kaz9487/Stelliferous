# A convolution call owns the cells of its input window, packs them block by
# block and writes the cells of its output positions, the groups of a
# position side by side. The caller validates dimensions and chooses the
# calls.
import Base
import ./kernel_gemm.bend as Gemm
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
import ./math_gelu_lanes.bend as GeluLanes
import ./tensor_tiled_map.bend as TiledMap
import ./tensor_tiled.bend as Tiled
import ./tensor_grid.bend as Grid

type Work is Data:
  Work{rows: U32,height: U32,width: U32,kernel: U32,stride: U32,padding: U32,output_width: U32,positions: U32,inner: U32}

# The matrix and its source stencil are independent. Forward convolution is
# the square specialization; an adjoint phase uses rectangular taps and origins.
# end is the position after the last one a call writes.
type WindowWork is Data:
  WindowWork{rows: U32,end: U32,window: Packing.Stencil}

def activated_value(~activation: F32 -> F32,context: Unit,value: F32) -> F32:
  activation(value)

def mapped(~Context: Data,~operation: Context -> F32 -> F32,+count: U32,+context: Context,buffer: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  Storage.Buffer{storage,length,certificate} = buffer
  Storage.Buffer{Traversal.map_in_place(~Tile.Tile8,~Context,~TiledMap.lanes(~Context,~operation),U32.to_nat(count),0,context,storage),length,
    TraversalCertificate.map_in_place(~Tile.Tile8,~Context,~TiledMap.lanes(~Context,~operation),U32.to_nat(count),0,context,storage,certificate)}

def leaky_relu_value(slope: F32,value: F32) -> F32:
  Activations.leaky_relu_value(value,slope)

def unit_mapped(~operation: F32 -> F32,+count: U32,buffer: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  mapped(~Unit,~activated_value(~operation),count,Unit{},buffer)

def gelu_mapped(+count: U32,buffer: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  Storage.Buffer{storage,length,certificate} = buffer
  Storage.Buffer{Traversal.map_in_place(~Tile.Tile8,~Unit,~GeluLanes.tile,U32.to_nat(count),0,Unit{},storage),length,
    TraversalCertificate.map_in_place(~Tile.Tile8,~Unit,~GeluLanes.tile,U32.to_nat(count),0,Unit{},storage,certificate)}

# One tight pass over the cells of each finished leaf while they are still in
# cache; applying the activation inside the store measured about twice as
# slow. The built-in is chosen once per leaf, so each pass is specialized;
# the identity skips it.
def activated(~custom: F32 -> F32,activation: Activations.Activation,+count: U32,buffer: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  match activation:
    case Activations.Identity{}: buffer
    case Activations.ReLU{}: unit_mapped(~Activations.relu_value,count,buffer)
    case Activations.ReLU6{}: unit_mapped(~Activations.relu6_value,count,buffer)
    case Activations.LeakyReLU{slope}: mapped(~F32,~leaky_relu_value,count,slope,buffer)
    case Activations.Sigmoid{}: unit_mapped(~Activations.sigmoid_value,count,buffer)
    case Activations.SiLU{}: unit_mapped(~Activations.silu_value,count,buffer)
    case Activations.Tanh{}: unit_mapped(~Activations.tanh_value,count,buffer)
    case Activations.HardSigmoid{}: unit_mapped(~Activations.hard_sigmoid_value,count,buffer)
    case Activations.HardSwish{}: unit_mapped(~Activations.hard_swish_value,count,buffer)
    case Activations.GELUTanh{}: gelu_mapped(count,buffer)
    case Activations.GELU{}: unit_mapped(~Activations.gelu_value,count,buffer)
    case Activations.Custom{operations}: unit_mapped(~custom,count,buffer)

def block_tiles() -> U32:
  Gemm.block_tiles()

def block_positions(inner: U32) -> U32:
  (8 * U32.max(1,(block_tiles() / inner : U32)) : U32)

def workspace_prefix(+target: U32,+inner: U32,positions: U32) -> U32:
  U32.min((8 * U32.max(1,(target / inner : U32)) : U32),KernelWindow.padded(positions))

def workspace_tiles(+target: U32,+inner: U32,+positions: U32) -> U32:
  (workspace_prefix(target,inner,positions) / 8 * inner : U32)

# The blocks of a call pack their tiles from the planes of the call and store
# the channel groups of their positions, eight channels a tile.
type BlockState is Type:
  BlockState{planes: Storage.Buffer<Tile.Tile16>,tiles: Storage.Buffer<Tile.Tile8>,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<Tile.Tile8>,
    output: Storage.Buffer<Tile.Tile8>}

# What a call gives back: its input, weights and bias, its two workspaces and
# its output.
type Kept is Type:
  Kept{input: Storage.Buffer<Tile.Tile8>,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<Tile.Tile8>,planes: Storage.Buffer<Tile.Tile16>,tiles: Storage.Buffer<Tile.Tile8>,
    output: Storage.Buffer<Tile.Tile8>}

def kept_from_blocks(input: Storage.Buffer<Tile.Tile8>,state: BlockState) -> Kept:
  BlockState{planes,tiles,weights,bias,output} = state
  Kept{input,weights,bias,planes,tiles,output}

def output_tiles(+rows: U32,+positions: U32) -> U32:
  ((rows + 7) / 8 * positions : U32)

# The cells of the channels over positions positions, activated, with +0 in
# the lanes past the channels.
def finished(~custom: F32 -> F32,activation: Activations.Activation,+rows: U32,+positions: U32,output: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  +count = output_tiles(rows,positions)
  TiledMap.padded(TiledMap.Padding{rows,Tiled.groups(rows),positions,Grid.Innermost{}},0,count,activated(~custom,activation,count,output))

def block_kept(planes: Storage.Buffer<Tile.Tile16>,kept: GemmBuffer.CellKept) -> BlockState:
  GemmBuffer.CellKept{tiles,weights,bias,output} = kept
  BlockState{planes,tiles,weights,bias,output}

def block_packed(+rows: U32,+end: U32,inner: U32,+start: U32,+span: U32,
  weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>,packed: Storage.Buffer<Tile.Tile16> & Storage.Buffer<Tile.Tile8>) -> BlockState:
  (planes,tiles) = packed
  block_kept(planes,
    GemmBuffer.cell_keeping(rows,tiles,weights,bias,output,span,inner,start,end))

# Packing addresses the planes of the call; output positions are those of the
# output buffer, none at or after end. Each block overwrites tiles from zero
# and writes directly into the output.
def block_step_window(work: WindowWork,+planes: Packing.Planes,pack_start: U32,output_start: U32,+span: U32,state: BlockState) -> BlockState:
  WindowWork{rows,end,+window} = work
  Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,+inner,groups} = window
  BlockState{pairs,tiles,weights,bias,output} = state
  block_packed(rows,end,inner,output_start,span,weights,bias,output,
    PackingBuffer.pack(window,planes,U32.to_nat((span / 8 * inner : U32)),pack_start,pairs,tiles))

def blocks_window(remaining: Nat,+work: WindowWork,+planes: Packing.Planes,+pack_start: U32,+output_start: U32,+padded: U32,+span: U32,state: BlockState) -> BlockState:
  match remaining:
    case 0n: state
    case 1n+rest:
      blocks_window(rest,work,planes,(pack_start + span : U32),(output_start + span : U32),padded,span,
        block_step_window(work,planes,pack_start,output_start,U32.min(span,(padded - output_start : U32)),state))

# An inner larger than the target still needs one complete eight-position
# group.
def blocked_window_with(+work: WindowWork,+planes: Packing.Planes,pack_start: U32,+offset: U32,+padded: U32,state: BlockState) -> BlockState:
  WindowWork{rows,end,Packing.Stencil{base,h,w,kr,kc,s,ro,co,ow,+inner,groups}} = work
  +span = block_positions(inner)
  blocks_window(U32.to_nat(((padded - offset + span - 1) / span : U32)),work,planes,pack_start,offset,padded,span,state)

def planes_ready(+work: WindowWork,+planes: Packing.Planes,pack_start: U32,+offset: U32,+padded: U32,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<Tile.Tile8>,
  tiles: Storage.Buffer<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>,transposed: Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile16>) -> Kept:
  (input,pairs) = transposed
  kept_from_blocks(input,blocked_window_with(work,planes,pack_start,offset,padded,BlockState{pairs,tiles,weights,bias,output}))

# The cells of positions [offset, end) of output from the window, read from
# the packing start pack_start: the input is transposed once into planes, a
# workspace of at least Packing.plane_bound(window, end - offset rounded up
# to 8) pairs, and packed block by block into tiles, a workspace of at least
# workspace_tiles(block_tiles(), inner, end - offset) tiles; input, weights,
# bias and the workspaces come back.
def blocked_into(+work: WindowWork,+pack_start: U32,+offset: U32,input: Storage.Buffer<Tile.Tile8>,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<Tile.Tile8>,
  planes: Storage.Buffer<Tile.Tile16>,tiles: Storage.Buffer<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>) -> Kept:
  WindowWork{rows,+end,+window} = work
  +padded = (offset + KernelWindow.padded((end - offset : U32)) : U32)
  +geometry = Packing.planes_of(window,pack_start,(padded - offset : U32))
  planes_ready(work,geometry,pack_start,offset,padded,weights,bias,tiles,output,PackingBuffer.transpose(window,geometry,input,planes))

# The deepest split that leaves every leaf at least one eight-position block.
def depth_limit(+padded: U32) -> U32:
  (U32.from_nat(Tensors.significant_bits(32n,(padded / 8 : U32),0n,U32.is_zero((padded / 8 : U32)))) - 1 : U32)

# ---- Pointwise products staged as pairs ----

def staged_dropped(result: Storage.Buffer<Tile.Tile16> & Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  (staged,cells) = result
  cells

def kept_tile(context: Unit,tile: Tile.Tile8) -> Tile.Tile8:
  tile

def unit_converted(~operation: F32 -> F32,+channels: U32,+count: U32,staged: Storage.Buffer<Tile.Tile16>,output: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  staged_dropped(GemmBuffer.staged_cells(~Unit,~TiledMap.lanes(~Unit,~activated_value(~operation)),channels,count,Unit{},staged,output))

# The cells of the staged pairs, the activation applied in the same pass.
def converted(~custom: F32 -> F32,activation: Activations.Activation,+channels: U32,+count: U32,staged: Storage.Buffer<Tile.Tile16>,output: Storage.Buffer<Tile.Tile8>) ->
  Storage.Buffer<Tile.Tile8>:
  match activation:
    case Activations.Identity{}: staged_dropped(GemmBuffer.staged_cells(~Unit,~kept_tile,channels,count,Unit{},staged,output))
    case Activations.ReLU{}: unit_converted(~Activations.relu_value,channels,count,staged,output)
    case Activations.ReLU6{}: unit_converted(~Activations.relu6_value,channels,count,staged,output)
    case Activations.LeakyReLU{slope}: staged_dropped(GemmBuffer.staged_cells(~F32,~TiledMap.lanes(~F32,~leaky_relu_value),channels,count,slope,staged,output))
    case Activations.Sigmoid{}: unit_converted(~Activations.sigmoid_value,channels,count,staged,output)
    case Activations.SiLU{}: unit_converted(~Activations.silu_value,channels,count,staged,output)
    case Activations.Tanh{}: unit_converted(~Activations.tanh_value,channels,count,staged,output)
    case Activations.HardSigmoid{}: unit_converted(~Activations.hard_sigmoid_value,channels,count,staged,output)
    case Activations.HardSwish{}: unit_converted(~Activations.hard_swish_value,channels,count,staged,output)
    case Activations.GELUTanh{}: staged_dropped(GemmBuffer.staged_cells(~Unit,~GeluLanes.tile,channels,count,Unit{},staged,output))
    case Activations.GELU{}: unit_converted(~Activations.gelu_value,channels,count,staged,output)
    case Activations.Custom{operations}: unit_converted(~custom,channels,count,staged,output)

# The cells of the channels over positions positions from their staged pairs,
# activated, with +0 in the lanes past the channels.
def point_finished(~custom: F32 -> F32,activation: Activations.Activation,+rows: U32,+positions: U32,staged: Storage.Buffer<Tile.Tile16>) -> Storage.Buffer<Tile.Tile8>:
  +count = output_tiles(rows,positions)
  TiledMap.padded(TiledMap.Padding{rows,Tiled.groups(rows),positions,Grid.Innermost{}},0,count,
    converted(~custom,activation,rows,count,staged,Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count)))

# What packing one workspace tile counts: its reads, shift and store are four
# rows, its addressing eight steps.
def packed_tile_rows() -> F32:
  4.0

def packed_tile_steps() -> F32:
  8.0

# What transposing one pair of a plane counts: its gather, pair and store are
# four rows, its addressing eight steps.
def transposed_pair_rows() -> F32:
  4.0

def transposed_pair_steps() -> F32:
  8.0
