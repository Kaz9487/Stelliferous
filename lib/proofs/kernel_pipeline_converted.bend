# The cells a pointwise task converts from its staged pairs: each cell is the
# activation of the half of its pair that holds its group.
import Base
import ../kernel_tile.bend as Tile
import ../kernel_gemm.bend as Gemm
import ../kernel_gemm_buffer.bend as GemmBuffer
import ../kernel_pipeline.bend as Pipeline
import ../math_activation.bend as Activations
import ../math_gelu_lanes.bend as GeluLanes
import ../tensor_tiled_map.bend as TiledMap
import ../storage_buffer.bend as Storage
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./kernel_gemm_point_buffer.bend as PointBuffer
import ./kernel_gemm_read_into.bend as ReadInto
import ./traversal_update_readback.bend as Updates
import ./kernel_pipeline_activation.bend as ActivationProofs
import ./nat_order.bend as Order
import ./row_major_index.bend as RowMajor
import ./nat_division_order.bend as DivisionOrder
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_division.bend as U32Division

# The cells the staged pairs convert to, and the halves they convert.

def converted_tree(~custom: F32 -> F32,activation: Activations.Activation,+channels: U32,+count: U32,+staged: Tree.Tree<Tile.Tile16>,+output: Tree.Tree<Tile.Tile8>) ->
  Tree.Tree<Tile.Tile8>:
  match activation:
    case Activations.Identity{}: PointBuffer.staged_loop(~Unit,~Pipeline.kept_tile,channels,count,Unit{},staged,output)
    case Activations.ReLU{}: PointBuffer.staged_loop(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.relu_value)),channels,count,Unit{},staged,output)
    case Activations.ReLU6{}: PointBuffer.staged_loop(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.relu6_value)),channels,count,Unit{},staged,output)
    case Activations.LeakyReLU{slope}: PointBuffer.staged_loop(~F32,~TiledMap.lanes(~F32,~Pipeline.leaky_relu_value),channels,count,slope,staged,output)
    case Activations.Sigmoid{}: PointBuffer.staged_loop(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.sigmoid_value)),channels,count,Unit{},staged,output)
    case Activations.SiLU{}: PointBuffer.staged_loop(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.silu_value)),channels,count,Unit{},staged,output)
    case Activations.Tanh{}: PointBuffer.staged_loop(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.tanh_value)),channels,count,Unit{},staged,output)
    case Activations.HardSigmoid{}: PointBuffer.staged_loop(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.hard_sigmoid_value)),channels,count,Unit{},staged,output)
    case Activations.HardSwish{}: PointBuffer.staged_loop(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.hard_swish_value)),channels,count,Unit{},staged,output)
    case Activations.GELUTanh{}: PointBuffer.staged_loop(~Unit,~GeluLanes.tile,channels,count,Unit{},staged,output)
    case Activations.GELU{}: PointBuffer.staged_loop(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.gelu_value)),channels,count,Unit{},staged,output)
    case Activations.Custom{operations}: PointBuffer.staged_loop(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~custom)),channels,count,Unit{},staged,output)

def unpaired_tree(+channels: U32,+count: U32,+staged: Tree.Tree<Tile.Tile16>,+output: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  PointBuffer.staged_loop(~Unit,~Pipeline.kept_tile,channels,count,Unit{},staged,output)

# ---- Contents ----

def second(pair: Buffer.Content<Tile.Tile16> & Buffer.Content<Tile.Tile8>) -> Buffer.Content<Tile.Tile8>:
  (staged,cells) = pair
  cells

def dropped_contents(pair: Storage.Buffer<Tile.Tile16> & Storage.Buffer<Tile.Tile8>) ->
  {Buffer.contents(~Tile.Tile8,Pipeline.staged_dropped(pair)) == second(ReadInto.buffers_contents(~Tile.Tile16,~Tile.Tile8,pair)) : Buffer.Content<Tile.Tile8>}:
  (staged,cells) = pair
  {==}

# Dropping the staged pairs leaves the cells read from them.
def dropped_cells(~Context: Data,~map: Context -> Tile.Tile8 -> Tile.Tile8,+channels: U32,+count: U32,+context: Context,+staged: Buffer.Model<Tile.Tile16>,
  +output: Buffer.Model<Tile.Tile8>) ->
  {Buffer.contents(~Tile.Tile8,Pipeline.staged_dropped(GemmBuffer.staged_cells(~Context,~map,channels,count,context,Buffer.pack(~Tile.Tile16,staged),Buffer.pack(~Tile.Tile8,output)))) ==
    Buffer.Content{PointBuffer.staged_loop(~Context,~map,channels,count,context,Buffer.tree(~Tile.Tile16,staged),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output)} :
    Buffer.Content<Tile.Tile8>}:
  %PointBuffer.staged_contents(~Context,~map,channels,count,context,staged,output) :
    {Buffer.contents(~Tile.Tile8,Pipeline.staged_dropped(GemmBuffer.staged_cells(~Context,~map,channels,count,context,Buffer.pack(~Tile.Tile16,staged),Buffer.pack(~Tile.Tile8,output)))) ==
      second(_) : Buffer.Content<Tile.Tile8>}
  dropped_contents(GemmBuffer.staged_cells(~Context,~map,channels,count,context,Buffer.pack(~Tile.Tile16,staged),Buffer.pack(~Tile.Tile8,output)))


# The converted cells are the cells read from the staged pairs.
def converted_contents(~custom: F32 -> F32,+activation: Activations.Activation,+channels: U32,+count: U32,+staged: Buffer.Model<Tile.Tile16>,+output: Buffer.Model<Tile.Tile8>) ->
  {Buffer.contents(~Tile.Tile8,Pipeline.converted(~custom,activation,channels,count,Buffer.pack(~Tile.Tile16,staged),Buffer.pack(~Tile.Tile8,output))) ==
    Buffer.Content{converted_tree(~custom,activation,channels,count,Buffer.tree(~Tile.Tile16,staged),Buffer.tree(~Tile.Tile8,output)),Buffer.length(~Tile.Tile8,output)} :
    Buffer.Content<Tile.Tile8>}:
  match activation:
    case Activations.Identity{}: dropped_cells(~Unit,~Pipeline.kept_tile,channels,count,Unit{},staged,output)
    case Activations.ReLU{}: dropped_cells(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.relu_value)),channels,count,Unit{},staged,output)
    case Activations.ReLU6{}: dropped_cells(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.relu6_value)),channels,count,Unit{},staged,output)
    case Activations.LeakyReLU{+slope}: dropped_cells(~F32,~TiledMap.lanes(~F32,~Pipeline.leaky_relu_value),channels,count,slope,staged,output)
    case Activations.Sigmoid{}: dropped_cells(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.sigmoid_value)),channels,count,Unit{},staged,output)
    case Activations.SiLU{}: dropped_cells(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.silu_value)),channels,count,Unit{},staged,output)
    case Activations.Tanh{}: dropped_cells(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.tanh_value)),channels,count,Unit{},staged,output)
    case Activations.HardSigmoid{}: dropped_cells(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.hard_sigmoid_value)),channels,count,Unit{},staged,output)
    case Activations.HardSwish{}: dropped_cells(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.hard_swish_value)),channels,count,Unit{},staged,output)
    case Activations.GELUTanh{}: dropped_cells(~Unit,~GeluLanes.tile,channels,count,Unit{},staged,output)
    case Activations.GELU{}: dropped_cells(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.gelu_value)),channels,count,Unit{},staged,output)
    case Activations.Custom{operations}: dropped_cells(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~custom)),channels,count,Unit{},staged,output)

# ---- Shape ----

def converted_balanced(~custom: F32 -> F32,+activation: Activations.Activation,+channels: U32,+count: U32,+staged: Tree.Tree<Tile.Tile16>,+output: Tree.Tree<Tile.Tile8>,+depth: Nat,+balanced: Shape.Balanced(Tile.Tile8,depth,output)) ->
  Shape.Balanced(Tile.Tile8,depth,converted_tree(~custom,activation,channels,count,staged,output)):
  match activation:
    case Activations.Identity{}:
      Updates.update_balanced(~Tree.Tree<Tile.Tile16>,~Tile.Tile8,~Tile.Tile8,~Gemm.StagedPlan<Unit>,~PointBuffer.staged_sample(~Unit,~Pipeline.kept_tile),~Gemm.replaced(~Tile.Tile8),~Gemm.staged_placed(~Unit),U32.to_nat(count),0,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),Unit{}},output,depth,balanced)
    case Activations.ReLU{}:
      Updates.update_balanced(~Tree.Tree<Tile.Tile16>,~Tile.Tile8,~Tile.Tile8,~Gemm.StagedPlan<Unit>,~PointBuffer.staged_sample(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.relu_value))),~Gemm.replaced(~Tile.Tile8),~Gemm.staged_placed(~Unit),U32.to_nat(count),0,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),Unit{}},output,depth,balanced)
    case Activations.ReLU6{}:
      Updates.update_balanced(~Tree.Tree<Tile.Tile16>,~Tile.Tile8,~Tile.Tile8,~Gemm.StagedPlan<Unit>,~PointBuffer.staged_sample(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.relu6_value))),~Gemm.replaced(~Tile.Tile8),~Gemm.staged_placed(~Unit),U32.to_nat(count),0,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),Unit{}},output,depth,balanced)
    case Activations.LeakyReLU{+slope}:
      Updates.update_balanced(~Tree.Tree<Tile.Tile16>,~Tile.Tile8,~Tile.Tile8,~Gemm.StagedPlan<F32>,~PointBuffer.staged_sample(~F32,~TiledMap.lanes(~F32,~Pipeline.leaky_relu_value)),~Gemm.replaced(~Tile.Tile8),~Gemm.staged_placed(~F32),U32.to_nat(count),0,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),slope},output,depth,balanced)
    case Activations.Sigmoid{}:
      Updates.update_balanced(~Tree.Tree<Tile.Tile16>,~Tile.Tile8,~Tile.Tile8,~Gemm.StagedPlan<Unit>,~PointBuffer.staged_sample(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.sigmoid_value))),~Gemm.replaced(~Tile.Tile8),~Gemm.staged_placed(~Unit),U32.to_nat(count),0,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),Unit{}},output,depth,balanced)
    case Activations.SiLU{}:
      Updates.update_balanced(~Tree.Tree<Tile.Tile16>,~Tile.Tile8,~Tile.Tile8,~Gemm.StagedPlan<Unit>,~PointBuffer.staged_sample(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.silu_value))),~Gemm.replaced(~Tile.Tile8),~Gemm.staged_placed(~Unit),U32.to_nat(count),0,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),Unit{}},output,depth,balanced)
    case Activations.Tanh{}:
      Updates.update_balanced(~Tree.Tree<Tile.Tile16>,~Tile.Tile8,~Tile.Tile8,~Gemm.StagedPlan<Unit>,~PointBuffer.staged_sample(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.tanh_value))),~Gemm.replaced(~Tile.Tile8),~Gemm.staged_placed(~Unit),U32.to_nat(count),0,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),Unit{}},output,depth,balanced)
    case Activations.HardSigmoid{}:
      Updates.update_balanced(~Tree.Tree<Tile.Tile16>,~Tile.Tile8,~Tile.Tile8,~Gemm.StagedPlan<Unit>,~PointBuffer.staged_sample(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.hard_sigmoid_value))),~Gemm.replaced(~Tile.Tile8),~Gemm.staged_placed(~Unit),U32.to_nat(count),0,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),Unit{}},output,depth,balanced)
    case Activations.HardSwish{}:
      Updates.update_balanced(~Tree.Tree<Tile.Tile16>,~Tile.Tile8,~Tile.Tile8,~Gemm.StagedPlan<Unit>,~PointBuffer.staged_sample(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.hard_swish_value))),~Gemm.replaced(~Tile.Tile8),~Gemm.staged_placed(~Unit),U32.to_nat(count),0,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),Unit{}},output,depth,balanced)
    case Activations.GELUTanh{}:
      Updates.update_balanced(~Tree.Tree<Tile.Tile16>,~Tile.Tile8,~Tile.Tile8,~Gemm.StagedPlan<Unit>,~PointBuffer.staged_sample(~Unit,~GeluLanes.tile),~Gemm.replaced(~Tile.Tile8),~Gemm.staged_placed(~Unit),U32.to_nat(count),0,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),Unit{}},output,depth,balanced)
    case Activations.GELU{}:
      Updates.update_balanced(~Tree.Tree<Tile.Tile16>,~Tile.Tile8,~Tile.Tile8,~Gemm.StagedPlan<Unit>,~PointBuffer.staged_sample(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.gelu_value))),~Gemm.replaced(~Tile.Tile8),~Gemm.staged_placed(~Unit),U32.to_nat(count),0,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),Unit{}},output,depth,balanced)
    case Activations.Custom{operations}:
      Updates.update_balanced(~Tree.Tree<Tile.Tile16>,~Tile.Tile8,~Tile.Tile8,~Gemm.StagedPlan<Unit>,~PointBuffer.staged_sample(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~custom))),~Gemm.replaced(~Tile.Tile8),~Gemm.staged_placed(~Unit),U32.to_nat(count),0,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),Unit{}},output,depth,balanced)

def converted_capacity(~custom: F32 -> F32,+activation: Activations.Activation,+channels: U32,+count: U32,+staged: Tree.Tree<Tile.Tile16>,+output: Tree.Tree<Tile.Tile8>) ->
  {Tree.capacity(Tile.Tile8,converted_tree(~custom,activation,channels,count,staged,output)) == Tree.capacity(Tile.Tile8,output) : U32}:
  match activation:
    case Activations.Identity{}:
      Updates.update_capacity(~Tree.Tree<Tile.Tile16>,~Tile.Tile8,~Tile.Tile8,~Gemm.StagedPlan<Unit>,~PointBuffer.staged_sample(~Unit,~Pipeline.kept_tile),~Gemm.replaced(~Tile.Tile8),~Gemm.staged_placed(~Unit),U32.to_nat(count),0,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),Unit{}},output)
    case Activations.ReLU{}:
      Updates.update_capacity(~Tree.Tree<Tile.Tile16>,~Tile.Tile8,~Tile.Tile8,~Gemm.StagedPlan<Unit>,~PointBuffer.staged_sample(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.relu_value))),~Gemm.replaced(~Tile.Tile8),~Gemm.staged_placed(~Unit),U32.to_nat(count),0,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),Unit{}},output)
    case Activations.ReLU6{}:
      Updates.update_capacity(~Tree.Tree<Tile.Tile16>,~Tile.Tile8,~Tile.Tile8,~Gemm.StagedPlan<Unit>,~PointBuffer.staged_sample(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.relu6_value))),~Gemm.replaced(~Tile.Tile8),~Gemm.staged_placed(~Unit),U32.to_nat(count),0,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),Unit{}},output)
    case Activations.LeakyReLU{+slope}:
      Updates.update_capacity(~Tree.Tree<Tile.Tile16>,~Tile.Tile8,~Tile.Tile8,~Gemm.StagedPlan<F32>,~PointBuffer.staged_sample(~F32,~TiledMap.lanes(~F32,~Pipeline.leaky_relu_value)),~Gemm.replaced(~Tile.Tile8),~Gemm.staged_placed(~F32),U32.to_nat(count),0,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),slope},output)
    case Activations.Sigmoid{}:
      Updates.update_capacity(~Tree.Tree<Tile.Tile16>,~Tile.Tile8,~Tile.Tile8,~Gemm.StagedPlan<Unit>,~PointBuffer.staged_sample(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.sigmoid_value))),~Gemm.replaced(~Tile.Tile8),~Gemm.staged_placed(~Unit),U32.to_nat(count),0,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),Unit{}},output)
    case Activations.SiLU{}:
      Updates.update_capacity(~Tree.Tree<Tile.Tile16>,~Tile.Tile8,~Tile.Tile8,~Gemm.StagedPlan<Unit>,~PointBuffer.staged_sample(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.silu_value))),~Gemm.replaced(~Tile.Tile8),~Gemm.staged_placed(~Unit),U32.to_nat(count),0,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),Unit{}},output)
    case Activations.Tanh{}:
      Updates.update_capacity(~Tree.Tree<Tile.Tile16>,~Tile.Tile8,~Tile.Tile8,~Gemm.StagedPlan<Unit>,~PointBuffer.staged_sample(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.tanh_value))),~Gemm.replaced(~Tile.Tile8),~Gemm.staged_placed(~Unit),U32.to_nat(count),0,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),Unit{}},output)
    case Activations.HardSigmoid{}:
      Updates.update_capacity(~Tree.Tree<Tile.Tile16>,~Tile.Tile8,~Tile.Tile8,~Gemm.StagedPlan<Unit>,~PointBuffer.staged_sample(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.hard_sigmoid_value))),~Gemm.replaced(~Tile.Tile8),~Gemm.staged_placed(~Unit),U32.to_nat(count),0,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),Unit{}},output)
    case Activations.HardSwish{}:
      Updates.update_capacity(~Tree.Tree<Tile.Tile16>,~Tile.Tile8,~Tile.Tile8,~Gemm.StagedPlan<Unit>,~PointBuffer.staged_sample(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.hard_swish_value))),~Gemm.replaced(~Tile.Tile8),~Gemm.staged_placed(~Unit),U32.to_nat(count),0,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),Unit{}},output)
    case Activations.GELUTanh{}:
      Updates.update_capacity(~Tree.Tree<Tile.Tile16>,~Tile.Tile8,~Tile.Tile8,~Gemm.StagedPlan<Unit>,~PointBuffer.staged_sample(~Unit,~GeluLanes.tile),~Gemm.replaced(~Tile.Tile8),~Gemm.staged_placed(~Unit),U32.to_nat(count),0,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),Unit{}},output)
    case Activations.GELU{}:
      Updates.update_capacity(~Tree.Tree<Tile.Tile16>,~Tile.Tile8,~Tile.Tile8,~Gemm.StagedPlan<Unit>,~PointBuffer.staged_sample(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.gelu_value))),~Gemm.replaced(~Tile.Tile8),~Gemm.staged_placed(~Unit),U32.to_nat(count),0,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),Unit{}},output)
    case Activations.Custom{operations}:
      Updates.update_capacity(~Tree.Tree<Tile.Tile16>,~Tile.Tile8,~Tile.Tile8,~Gemm.StagedPlan<Unit>,~PointBuffer.staged_sample(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~custom))),~Gemm.replaced(~Tile.Tile8),~Gemm.staged_placed(~Unit),U32.to_nat(count),0,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),Unit{}},output)

# ---- Reads ----

# A cell read through a map is the map of the half it reads.
def mapped_read(~Context: Data,~map: Context -> Tile.Tile8 -> Tile.Tile8,+context: Context,+channels: U32,+count: U32,+staged: Tree.Tree<Tile.Tile16>,+output: Tree.Tree<Tile.Tile8>,+depth: Nat,+balanced: Shape.Balanced(Tile.Tile8,depth,output),+limit: {Nat.is_le(depth,31n) == True{} : Bool},
  +capacity_ok: {U32.is_le(Tree.capacity(Tile.Tile8,output),2147483648) == True{} : Bool},+room: {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Tile.Tile8,output))) == True{} : Bool},
  +index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,PointBuffer.staged_loop(~Context,~map,channels,count,context,staged,output),U32.from_nat(index)) == map(context,Tree.read(Tile.Tile8,unpaired_tree(channels,count,staged,output),U32.from_nat(index))) : Tile.Tile8}:
  %Equal.sym(Tile.Tile8,Tree.read(Tile.Tile8,unpaired_tree(channels,count,staged,output),U32.from_nat(index)),PointBuffer.staged_tile(~Unit,~Pipeline.kept_tile,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),Unit{}},U32.from_nat(index)),PointBuffer.staged_at(~Unit,~Pipeline.kept_tile,channels,count,Unit{},staged,output,depth,balanced,limit,capacity_ok,room,index,inside)) :
    {Tree.read(Tile.Tile8,PointBuffer.staged_loop(~Context,~map,channels,count,context,staged,output),U32.from_nat(index)) == map(context,_) : Tile.Tile8}
  PointBuffer.staged_at(~Context,~map,channels,count,context,staged,output,depth,balanced,limit,capacity_ok,room,index,inside)

def converted_native(~custom: F32 -> F32,+activation: Activations.Activation,+channels: U32,+count: U32,+staged: Tree.Tree<Tile.Tile16>,+output: Tree.Tree<Tile.Tile8>,+depth: Nat,+balanced: Shape.Balanced(Tile.Tile8,depth,output),+limit: {Nat.is_le(depth,31n) == True{} : Bool},
  +capacity_ok: {U32.is_le(Tree.capacity(Tile.Tile8,output),2147483648) == True{} : Bool},+room: {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Tile.Tile8,output))) == True{} : Bool},
  +index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,converted_tree(~custom,activation,channels,count,staged,output),U32.from_nat(index)) ==
    ActivationProofs.native_tile(~custom,activation,Tree.read(Tile.Tile8,unpaired_tree(channels,count,staged,output),U32.from_nat(index))) : Tile.Tile8}:
  match activation:
    case Activations.Identity{}: {==}
    case Activations.ReLU{}: mapped_read(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.relu_value)),Unit{},channels,count,staged,output,depth,balanced,limit,capacity_ok,room,index,inside)
    case Activations.ReLU6{}: mapped_read(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.relu6_value)),Unit{},channels,count,staged,output,depth,balanced,limit,capacity_ok,room,index,inside)
    case Activations.LeakyReLU{+slope}: mapped_read(~F32,~TiledMap.lanes(~F32,~Pipeline.leaky_relu_value),slope,channels,count,staged,output,depth,balanced,limit,capacity_ok,room,index,inside)
    case Activations.Sigmoid{}: mapped_read(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.sigmoid_value)),Unit{},channels,count,staged,output,depth,balanced,limit,capacity_ok,room,index,inside)
    case Activations.SiLU{}: mapped_read(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.silu_value)),Unit{},channels,count,staged,output,depth,balanced,limit,capacity_ok,room,index,inside)
    case Activations.Tanh{}: mapped_read(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.tanh_value)),Unit{},channels,count,staged,output,depth,balanced,limit,capacity_ok,room,index,inside)
    case Activations.HardSigmoid{}: mapped_read(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.hard_sigmoid_value)),Unit{},channels,count,staged,output,depth,balanced,limit,capacity_ok,room,index,inside)
    case Activations.HardSwish{}: mapped_read(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.hard_swish_value)),Unit{},channels,count,staged,output,depth,balanced,limit,capacity_ok,room,index,inside)
    case Activations.GELUTanh{}: mapped_read(~Unit,~GeluLanes.tile,Unit{},channels,count,staged,output,depth,balanced,limit,capacity_ok,room,index,inside)
    case Activations.GELU{}: mapped_read(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.gelu_value)),Unit{},channels,count,staged,output,depth,balanced,limit,capacity_ok,room,index,inside)
    case Activations.Custom{operations}: mapped_read(~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~custom)),Unit{},channels,count,staged,output,depth,balanced,limit,capacity_ok,room,index,inside)

# A converted cell is the activation of the half it reads.
def converted_read(~custom: F32 -> F32,+activation: Activations.Activation,+channels: U32,+count: U32,+staged: Tree.Tree<Tile.Tile16>,+output: Tree.Tree<Tile.Tile8>,+depth: Nat,+balanced: Shape.Balanced(Tile.Tile8,depth,output),+limit: {Nat.is_le(depth,31n) == True{} : Bool},
  +capacity_ok: {U32.is_le(Tree.capacity(Tile.Tile8,output),2147483648) == True{} : Bool},+room: {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Tile.Tile8,output))) == True{} : Bool},
  +index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,converted_tree(~custom,activation,channels,count,staged,output),U32.from_nat(index)) ==
    ActivationProofs.activation_tile(~custom,activation,Tree.read(Tile.Tile8,unpaired_tree(channels,count,staged,output),U32.from_nat(index))) : Tile.Tile8}:
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,converted_tree(~custom,activation,channels,count,staged,output),U32.from_nat(index)),
    ActivationProofs.native_tile(~custom,activation,Tree.read(Tile.Tile8,unpaired_tree(channels,count,staged,output),U32.from_nat(index))),ActivationProofs.activation_tile(~custom,activation,Tree.read(Tile.Tile8,unpaired_tree(channels,count,staged,output),U32.from_nat(index))),
    converted_native(~custom,activation,channels,count,staged,output,depth,balanced,limit,capacity_ok,room,index,inside),ActivationProofs.native_lanes(~custom,activation,Tree.read(Tile.Tile8,unpaired_tree(channels,count,staged,output),U32.from_nat(index))))

# ---- The pair a cell reads ----

# Half of a group lies below it.
def half_below(+group: Nat) -> {Nat.is_le(Nat.div(group,2n),group) == True{} : Bool}:
  Order.transitive(Nat.div(group,2n),Nat.mul(Nat.div(group,2n),2n),group,RowMajor.product_covers(Nat.div(group,2n),2n,{==}),
    DivisionOrder.product_below(group,2n,{==}))

# The pair of a cell lies at or below the cell.
def pair_below(+index: Nat,+groups: Nat,+pairs: Nat,+positive: {Nat.is_lt(0n,groups) == True{} : Bool},+fewer: {Nat.is_le(pairs,groups) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(Nat.div(index,groups),pairs),Nat.div(Nat.mod(index,groups),2n)),index) == True{} : Bool}:
  +quotient = Nat.div(index,groups)
  +group = Nat.mod(index,groups)
  +upper = Order.transitive(Nat.add(Nat.mul(quotient,pairs),Nat.div(group,2n)),Nat.add(Nat.mul(quotient,groups),Nat.div(group,2n)),Nat.add(Nat.mul(quotient,groups),group),
    Order.add_le(Nat.mul(quotient,pairs),Nat.mul(quotient,groups),Nat.div(group,2n),RowMajor.mul_le_left(quotient,pairs,groups,fewer)),
    Order.offset_le(Nat.mul(quotient,groups),Nat.div(group,2n),group,half_below(group)))
  %Equal.sym(Nat,index,Nat.add(Nat.mul(quotient,groups),group),DivisionOrder.quotient_exact(index,groups,positive)) :
    {Nat.is_le(Nat.add(Nat.mul(quotient,pairs),Nat.div(group,2n)),_) == True{} : Bool}
  upper

# A staged cell selects the upper half for an odd group.
def staged_cell(~Context: Data,~map: Context -> Tile.Tile8 -> Tile.Tile8,+context: Context,+channels: U32,+staged: Tree.Tree<Tile.Tile16>,+index: Nat,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+groups: Nat,+groups_exact: {U32.to_nat(Gemm.blocks(channels)) == groups : Nat},
  +pairs: Nat,+pairs_exact: {U32.to_nat(Gemm.pairs_of(channels)) == pairs : Nat},+positive: {Nat.is_lt(0n,groups) == True{} : Bool},
  +fewer: {Nat.is_le(pairs,groups) == True{} : Bool},+divisor_room: {Nat.is_le(Nat.double(groups),U32.to_nat(cap)) == True{} : Bool},
  +two_room: {Nat.is_le(4n,U32.to_nat(cap)) == True{} : Bool},+index_room: {Nat.is_le(index,U32.to_nat(cap)) == True{} : Bool}) ->
  {PointBuffer.staged_tile(~Context,~map,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),context},U32.from_nat(index)) ==
    map(context,Gemm.staged_half(Tree.read(Tile.Tile16,staged,U32.from_nat(Nat.add(Nat.mul(Nat.div(index,groups),pairs),Nat.div(Nat.mod(index,groups),2n)))),
      U32.is_eq(U32.from_nat(Nat.mod(Nat.mod(index,groups),2n)),1))) : Tile.Tile8}:
  +word = U32.from_nat(index)
  +group_word = (word % Gemm.blocks(channels) : U32)
  +quotient_word = (word / Gemm.blocks(channels) : U32)
  +quotient = Nat.div(index,groups)
  +group = Nat.mod(index,groups)
  +pair_word = ((quotient_word * Gemm.pairs_of(channels) : U32) + (group_word / 2 : U32) : U32)
  +parity_word = (group_word % 2 : U32)
  +pair = Nat.add(Nat.mul(quotient,pairs),Nat.div(group,2n))
  +index_exact = Bounds.roundtrip(index,cap,cap_ok,index_room)
  +quotient_exact = U32Division.quotient_exact(cap,cap_ok,word,Gemm.blocks(channels),index,groups,index_exact,groups_exact,divisor_room,positive)
  +group_exact = U32Division.remainder_exact(cap,cap_ok,word,Gemm.blocks(channels),index,groups,index_exact,groups_exact,divisor_room,positive)
  +half_exact = U32Division.quotient_exact(cap,cap_ok,group_word,2,group,2n,group_exact,{==},two_room,{==})
  +parity_exact = U32Division.remainder_exact(cap,cap_ok,group_word,2,group,2n,group_exact,{==},two_room,{==})
  +pair_room = Order.transitive(pair,index,U32.to_nat(cap),pair_below(index,groups,pairs,positive,fewer),index_room)
  +product_room = Order.transitive(Nat.mul(quotient,pairs),pair,U32.to_nat(cap),Order.le_add(Nat.mul(quotient,pairs),Nat.div(group,2n)),pair_room)
  +product_exact = Arithmetic.product_exact(cap,cap_ok,quotient_word,Gemm.pairs_of(channels),quotient,pairs,quotient_exact,pairs_exact,product_room)
  +pair_exact = Arithmetic.sum_exact(cap,cap_ok,(quotient_word * Gemm.pairs_of(channels) : U32),(group_word / 2 : U32),Nat.mul(quotient,pairs),Nat.div(group,2n),
    product_exact,half_exact,pair_room)
  +pair_same = Equal.sym(U32,U32.from_nat(pair),pair_word,Arithmetic.from_nat_exact(pair_word,pair,pair_exact))
  +parity_same = Equal.sym(U32,U32.from_nat(Nat.mod(group,2n)),parity_word,Arithmetic.from_nat_exact(parity_word,Nat.mod(group,2n),parity_exact))
  %pair_same : {PointBuffer.staged_tile(~Context,~map,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),context},word) ==
    map(context,Gemm.staged_half(Tree.read(Tile.Tile16,staged,_),U32.is_eq(U32.from_nat(Nat.mod(group,2n)),1))) : Tile.Tile8}
  %parity_same : {PointBuffer.staged_tile(~Context,~map,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),context},word) ==
    map(context,Gemm.staged_half(Tree.read(Tile.Tile16,staged,pair_word),U32.is_eq(_,1))) : Tile.Tile8}
  {==}

def unpaired_read(+channels: U32,+count: U32,+staged: Tree.Tree<Tile.Tile16>,+output: Tree.Tree<Tile.Tile8>,+depth: Nat,+balanced: Shape.Balanced(Tile.Tile8,depth,output),
  +limit: {Nat.is_le(depth,31n) == True{} : Bool},+capacity_ok: {U32.is_le(Tree.capacity(Tile.Tile8,output),2147483648) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Tile.Tile8,output))) == True{} : Bool},+index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+groups: Nat,+groups_exact: {U32.to_nat(Gemm.blocks(channels)) == groups : Nat},
  +pairs: Nat,+pairs_exact: {U32.to_nat(Gemm.pairs_of(channels)) == pairs : Nat},+positive: {Nat.is_lt(0n,groups) == True{} : Bool},
  +fewer: {Nat.is_le(pairs,groups) == True{} : Bool},+divisor_room: {Nat.is_le(Nat.double(groups),U32.to_nat(cap)) == True{} : Bool},
  +two_room: {Nat.is_le(4n,U32.to_nat(cap)) == True{} : Bool},+index_room: {Nat.is_le(index,U32.to_nat(cap)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,unpaired_tree(channels,count,staged,output),U32.from_nat(index)) ==
    Gemm.staged_half(Tree.read(Tile.Tile16,staged,U32.from_nat(Nat.add(Nat.mul(Nat.div(index,groups),pairs),Nat.div(Nat.mod(index,groups),2n)))),
      U32.is_eq(U32.from_nat(Nat.mod(Nat.mod(index,groups),2n)),1)) : Tile.Tile8}:
  +word = U32.from_nat(index)
  +half = PointBuffer.staged_tile(~Unit,~Pipeline.kept_tile,staged,Gemm.StagedPlan{Gemm.blocks(channels),Gemm.pairs_of(channels),Unit{}},word)
  +address = Nat.add(Nat.mul(Nat.div(index,groups),pairs),Nat.div(Nat.mod(index,groups),2n))
  +upper = U32.is_eq(U32.from_nat(Nat.mod(Nat.mod(index,groups),2n)),1)
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,unpaired_tree(channels,count,staged,output),word),half,Gemm.staged_half(Tree.read(Tile.Tile16,staged,U32.from_nat(address)),upper),
    PointBuffer.staged_at(~Unit,~Pipeline.kept_tile,channels,count,Unit{},staged,output,depth,balanced,limit,capacity_ok,room,index,inside),
    staged_cell(~Unit,~Pipeline.kept_tile,Unit{},channels,staged,index,cap,cap_ok,groups,groups_exact,pairs,pairs_exact,positive,fewer,divisor_room,two_room,index_room))
