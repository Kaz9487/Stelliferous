# Shared array/tree semantics and readback of the final activation pass.
import Base
import ../tensor_access.bend as Access
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./tensor_allocation_depth.bend as Depth
import ../storage_buffer.bend as Storage
import ../storage_allocation.bend as Allocation
import ../tensor.bend as Tensors
import ../kernel_pipeline.bend as Pipeline
import ../math_activation.bend as Activations
import ../traversal_array.bend as Traversal
import ./storage_allocation_refinement.bend as AllocationLaws
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./traversal_map_readback.bend as MapReadback
import ../kernel_tile.bend as Tile
import ../tensor_tiled_map.bend as TiledMap
import ../math_gelu_lanes.bend as GeluLanes
import ./kernel_tile_refinement.bend as TileLaws
import ./word_multiplication.bend as WordMultiplication
import ./storage_buffer_model.bend as Buffer
import ./storage_refinement.bend as StorageLaws

def storage(~Element: Data,buffer: Storage.Buffer<Element>) -> Array<Element>:
  Storage.Buffer{storage,length,certificate} = buffer
  storage

def activation_array(~custom: F32 -> F32,activation: Activations.Activation,+count: U32,cells: Array<Tile.Tile8>) -> Array<Tile.Tile8>:
  match activation:
    case Activations.Identity{}: cells
    case Activations.ReLU{}: Traversal.map_in_place(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.relu_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.ReLU6{}: Traversal.map_in_place(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.relu6_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.LeakyReLU{slope}: Traversal.map_in_place(~Tile.Tile8,~F32,~TiledMap.lanes(~F32,~Pipeline.leaky_relu_value),U32.to_nat(count),0,slope,cells)
    case Activations.Sigmoid{}: Traversal.map_in_place(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.sigmoid_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.SiLU{}: Traversal.map_in_place(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.silu_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.Tanh{}: Traversal.map_in_place(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.tanh_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.HardSigmoid{}: Traversal.map_in_place(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.hard_sigmoid_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.HardSwish{}: Traversal.map_in_place(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.hard_swish_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.GELUTanh{}: Traversal.map_in_place(~Tile.Tile8,~Unit,~GeluLanes.tile,U32.to_nat(count),0,Unit{},cells)
    case Activations.GELU{}: Traversal.map_in_place(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.gelu_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.Custom{operations}: Traversal.map_in_place(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~custom)),U32.to_nat(count),0,Unit{},cells)

def activation_tree(~custom: F32 -> F32,activation: Activations.Activation,+count: U32,cells: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  match activation:
    case Activations.Identity{}: cells
    case Activations.ReLU{}: MapReadback.run(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.relu_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.ReLU6{}: MapReadback.run(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.relu6_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.LeakyReLU{slope}: MapReadback.run(~Tile.Tile8,~F32,~TiledMap.lanes(~F32,~Pipeline.leaky_relu_value),U32.to_nat(count),0,slope,cells)
    case Activations.Sigmoid{}: MapReadback.run(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.sigmoid_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.SiLU{}: MapReadback.run(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.silu_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.Tanh{}: MapReadback.run(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.tanh_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.HardSigmoid{}: MapReadback.run(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.hard_sigmoid_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.HardSwish{}: MapReadback.run(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.hard_swish_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.GELUTanh{}: MapReadback.run(~Tile.Tile8,~Unit,~GeluLanes.tile,U32.to_nat(count),0,Unit{},cells)
    case Activations.GELU{}: MapReadback.run(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.gelu_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.Custom{operations}: MapReadback.run(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~custom)),U32.to_nat(count),0,Unit{},cells)

def activation_model(~custom: F32 -> F32,+activation: Activations.Activation,+count: U32,+cells: Tree.Tree<Tile.Tile8>) ->
  {activation_array(~custom,activation,count,Tree.pack(Tile.Tile8,cells)) == Tree.pack(Tile.Tile8,activation_tree(~custom,activation,count,cells)) : Array<Tile.Tile8>}:
  match activation:
    case Activations.Identity{}: {==}
    case Activations.ReLU{}: MapReadback.packed(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.relu_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.ReLU6{}: MapReadback.packed(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.relu6_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.LeakyReLU{+slope}: MapReadback.packed(~Tile.Tile8,~F32,~TiledMap.lanes(~F32,~Pipeline.leaky_relu_value),U32.to_nat(count),0,slope,cells)
    case Activations.Sigmoid{}: MapReadback.packed(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.sigmoid_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.SiLU{}: MapReadback.packed(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.silu_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.Tanh{}: MapReadback.packed(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.tanh_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.HardSigmoid{}: MapReadback.packed(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.hard_sigmoid_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.HardSwish{}: MapReadback.packed(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.hard_swish_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.GELUTanh{}: MapReadback.packed(~Tile.Tile8,~Unit,~GeluLanes.tile,U32.to_nat(count),0,Unit{},cells)
    case Activations.GELU{}: MapReadback.packed(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.gelu_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.Custom{operations}: MapReadback.packed(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~custom)),U32.to_nat(count),0,Unit{},cells)

def activation_value(~custom: F32 -> F32,activation: Activations.Activation,value: F32) -> F32:
  match activation:
    case Activations.Identity{}: value
    case Activations.ReLU{}: Activations.relu_value(value)
    case Activations.ReLU6{}: Activations.relu6_value(value)
    case Activations.LeakyReLU{slope}: Pipeline.leaky_relu_value(slope,value)
    case Activations.Sigmoid{}: Activations.sigmoid_value(value)
    case Activations.SiLU{}: Activations.silu_value(value)
    case Activations.Tanh{}: Activations.tanh_value(value)
    case Activations.HardSigmoid{}: Activations.hard_sigmoid_value(value)
    case Activations.HardSwish{}: Activations.hard_swish_value(value)
    case Activations.GELUTanh{}: Activations.gelu_tanh_value(value)
    case Activations.GELU{}: Activations.gelu_value(value)
    case Activations.Custom{operations}: custom(value)

# The activation of every lane of a cell.
def activation_tile(~custom: F32 -> F32,activation: Activations.Activation,tile: Tile.Tile8) -> Tile.Tile8:
  Tile.Tile{a,b,c,d,e,f,g,h} = tile
  +activation = activation
  Tile.Tile{activation_value(~custom,activation,a),activation_value(~custom,activation,b),activation_value(~custom,activation,c),activation_value(~custom,activation,d),
    activation_value(~custom,activation,e),activation_value(~custom,activation,f),activation_value(~custom,activation,g),activation_value(~custom,activation,h)}

# The operation the pass applies to one cell.
def native_tile(~custom: F32 -> F32,activation: Activations.Activation,tile: Tile.Tile8) -> Tile.Tile8:
  match activation:
    case Activations.Identity{}: tile
    case Activations.ReLU{}: TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.relu_value))(Unit{},tile)
    case Activations.ReLU6{}: TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.relu6_value))(Unit{},tile)
    case Activations.LeakyReLU{slope}: TiledMap.lanes(~F32,~Pipeline.leaky_relu_value)(slope,tile)
    case Activations.Sigmoid{}: TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.sigmoid_value))(Unit{},tile)
    case Activations.SiLU{}: TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.silu_value))(Unit{},tile)
    case Activations.Tanh{}: TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.tanh_value))(Unit{},tile)
    case Activations.HardSigmoid{}: TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.hard_sigmoid_value))(Unit{},tile)
    case Activations.HardSwish{}: TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.hard_swish_value))(Unit{},tile)
    case Activations.GELUTanh{}: GeluLanes.tile(Unit{},tile)
    case Activations.GELU{}: TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.gelu_value))(Unit{},tile)
    case Activations.Custom{operations}: TiledMap.lanes(~Unit,~Pipeline.activated_value(~custom))(Unit{},tile)

# The staged arithmetic changes only the implementation of one eight-value cell.
def gelu_tanh_tile(+context: Unit,+tile: Tile.Tile8) ->
  {GeluLanes.tile(context,tile) == TiledMap.lanes(~Unit,~Access.apply_element(~F32,~Activations.gelu_tanh_value),context,tile) : Tile.Tile8}:
  match tile:
    case Tile.Tile{+a,+b,+c,+d,+e,+f,+g,+h}: {==}

def native_lanes(~custom: F32 -> F32,+activation: Activations.Activation,+tile: Tile.Tile8) ->
  {native_tile(~custom,activation,tile) == activation_tile(~custom,activation,tile) : Tile.Tile8}:
  match activation tile:
    case Activations.Identity{} Tile.Tile{+a,+b,+c,+d,+e,+f,+g,+h}: {==}
    case Activations.ReLU{} Tile.Tile{+a,+b,+c,+d,+e,+f,+g,+h}: {==}
    case Activations.ReLU6{} Tile.Tile{+a,+b,+c,+d,+e,+f,+g,+h}: {==}
    case Activations.LeakyReLU{+slope} Tile.Tile{+a,+b,+c,+d,+e,+f,+g,+h}: {==}
    case Activations.Sigmoid{} Tile.Tile{+a,+b,+c,+d,+e,+f,+g,+h}: {==}
    case Activations.SiLU{} Tile.Tile{+a,+b,+c,+d,+e,+f,+g,+h}: {==}
    case Activations.Tanh{} Tile.Tile{+a,+b,+c,+d,+e,+f,+g,+h}: {==}
    case Activations.HardSigmoid{} Tile.Tile{+a,+b,+c,+d,+e,+f,+g,+h}: {==}
    case Activations.HardSwish{} Tile.Tile{+a,+b,+c,+d,+e,+f,+g,+h}: {==}
    case Activations.GELUTanh{} Tile.Tile{+a,+b,+c,+d,+e,+f,+g,+h}: {==}
    case Activations.GELU{} Tile.Tile{+a,+b,+c,+d,+e,+f,+g,+h}: {==}
    case Activations.Custom{operations} Tile.Tile{+a,+b,+c,+d,+e,+f,+g,+h}: {==}

def native_cases(~custom: F32 -> F32,+activation: Activations.Activation,
  +count: U32,+cells: Tree.Tree<Tile.Tile8>,+depth: Nat,
  +capacity_equal: {Tree.capacity(Tile.Tile8,cells) == U32.shln(1,depth) : U32},+balanced: Shape.Balanced(Tile.Tile8,depth,cells),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity_ok: {U32.is_le(U32.shln(1,depth),2147483648) == True{} : Bool},+room: {Nat.is_le(U32.to_nat(count),U32.to_nat(U32.shln(1,depth))) == True{} : Bool},
  +index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,activation_tree(~custom,activation,count,cells),U32.from_nat(index)) ==
    native_tile(~custom,activation,Tree.read(Tile.Tile8,cells,U32.from_nat(index))) : Tile.Tile8}:
  match activation:
    case Activations.Identity{}: {==}
    case Activations.ReLU{}: 
      MapReadback.inside(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.relu_value)),U32.to_nat(count),0n,index,Unit{},cells,depth,U32.shln(1,depth),capacity_equal,balanced,depth_ok,capacity_ok,room,inside)
    case Activations.ReLU6{}: 
      MapReadback.inside(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.relu6_value)),U32.to_nat(count),0n,index,Unit{},cells,depth,U32.shln(1,depth),capacity_equal,balanced,depth_ok,capacity_ok,room,inside)
    case Activations.LeakyReLU{+slope}: 
      MapReadback.inside(~Tile.Tile8,~F32,~TiledMap.lanes(~F32,~Pipeline.leaky_relu_value),U32.to_nat(count),0n,index,slope,cells,depth,U32.shln(1,depth),capacity_equal,balanced,depth_ok,capacity_ok,room,inside)
    case Activations.Sigmoid{}: 
      MapReadback.inside(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.sigmoid_value)),U32.to_nat(count),0n,index,Unit{},cells,depth,U32.shln(1,depth),capacity_equal,balanced,depth_ok,capacity_ok,room,inside)
    case Activations.SiLU{}: 
      MapReadback.inside(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.silu_value)),U32.to_nat(count),0n,index,Unit{},cells,depth,U32.shln(1,depth),capacity_equal,balanced,depth_ok,capacity_ok,room,inside)
    case Activations.Tanh{}: 
      MapReadback.inside(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.tanh_value)),U32.to_nat(count),0n,index,Unit{},cells,depth,U32.shln(1,depth),capacity_equal,balanced,depth_ok,capacity_ok,room,inside)
    case Activations.HardSigmoid{}: 
      MapReadback.inside(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.hard_sigmoid_value)),U32.to_nat(count),0n,index,Unit{},cells,depth,U32.shln(1,depth),capacity_equal,balanced,depth_ok,capacity_ok,room,inside)
    case Activations.HardSwish{}: 
      MapReadback.inside(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.hard_swish_value)),U32.to_nat(count),0n,index,Unit{},cells,depth,U32.shln(1,depth),capacity_equal,balanced,depth_ok,capacity_ok,room,inside)
    case Activations.GELUTanh{}: 
      MapReadback.inside(~Tile.Tile8,~Unit,~GeluLanes.tile,U32.to_nat(count),0n,index,Unit{},cells,depth,U32.shln(1,depth),capacity_equal,balanced,depth_ok,capacity_ok,room,inside)
    case Activations.GELU{}: 
      MapReadback.inside(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.gelu_value)),U32.to_nat(count),0n,index,Unit{},cells,depth,U32.shln(1,depth),capacity_equal,balanced,depth_ok,capacity_ok,room,inside)
    case Activations.Custom{operations}: 
      MapReadback.inside(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~custom)),U32.to_nat(count),0n,index,Unit{},cells,depth,U32.shln(1,depth),capacity_equal,balanced,depth_ok,capacity_ok,room,inside)

# A cell of the activated leaf is the activation of every lane of its product.
def activated_cases(~custom: F32 -> F32,+activation: Activations.Activation,
  +count: U32,+cells: Tree.Tree<Tile.Tile8>,+depth: Nat,
  +capacity_equal: {Tree.capacity(Tile.Tile8,cells) == U32.shln(1,depth) : U32},+balanced: Shape.Balanced(Tile.Tile8,depth,cells),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity_ok: {U32.is_le(U32.shln(1,depth),2147483648) == True{} : Bool},+room: {Nat.is_le(U32.to_nat(count),U32.to_nat(U32.shln(1,depth))) == True{} : Bool},
  +index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,activation_tree(~custom,activation,count,cells),U32.from_nat(index)) ==
    activation_tile(~custom,activation,Tree.read(Tile.Tile8,cells,U32.from_nat(index))) : Tile.Tile8}:
  +tile = Tree.read(Tile.Tile8,cells,U32.from_nat(index))
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,activation_tree(~custom,activation,count,cells),U32.from_nat(index)),native_tile(~custom,activation,tile),
    activation_tile(~custom,activation,tile),native_cases(~custom,activation,count,cells,depth,capacity_equal,balanced,depth_ok,capacity_ok,room,index,inside),
    native_lanes(~custom,activation,tile))

def activation_balanced(~custom: F32 -> F32,+activation: Activations.Activation,+count: U32,+cells: Tree.Tree<Tile.Tile8>,+depth: Nat,
  +balanced: Shape.Balanced(Tile.Tile8,depth,cells)) -> Shape.Balanced(Tile.Tile8,depth,activation_tree(~custom,activation,count,cells)):
  match activation:
    case Activations.Identity{}: balanced
    case Activations.ReLU{}: MapReadback.balanced(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.relu_value)),U32.to_nat(count),0,Unit{},cells,depth,balanced)
    case Activations.ReLU6{}: MapReadback.balanced(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.relu6_value)),U32.to_nat(count),0,Unit{},cells,depth,balanced)
    case Activations.LeakyReLU{+slope}: MapReadback.balanced(~Tile.Tile8,~F32,~TiledMap.lanes(~F32,~Pipeline.leaky_relu_value),U32.to_nat(count),0,slope,cells,depth,balanced)
    case Activations.Sigmoid{}: MapReadback.balanced(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.sigmoid_value)),U32.to_nat(count),0,Unit{},cells,depth,balanced)
    case Activations.SiLU{}: MapReadback.balanced(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.silu_value)),U32.to_nat(count),0,Unit{},cells,depth,balanced)
    case Activations.Tanh{}: MapReadback.balanced(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.tanh_value)),U32.to_nat(count),0,Unit{},cells,depth,balanced)
    case Activations.HardSigmoid{}: MapReadback.balanced(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.hard_sigmoid_value)),U32.to_nat(count),0,Unit{},cells,depth,balanced)
    case Activations.HardSwish{}: MapReadback.balanced(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.hard_swish_value)),U32.to_nat(count),0,Unit{},cells,depth,balanced)
    case Activations.GELUTanh{}: MapReadback.balanced(~Tile.Tile8,~Unit,~GeluLanes.tile,U32.to_nat(count),0,Unit{},cells,depth,balanced)
    case Activations.GELU{}: MapReadback.balanced(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.gelu_value)),U32.to_nat(count),0,Unit{},cells,depth,balanced)
    case Activations.Custom{operations}: MapReadback.balanced(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~custom)),U32.to_nat(count),0,Unit{},cells,depth,balanced)

def activation_capacity(~custom: F32 -> F32,+activation: Activations.Activation,+count: U32,+cells: Tree.Tree<Tile.Tile8>) ->
  {Tree.capacity(Tile.Tile8,activation_tree(~custom,activation,count,cells)) == Tree.capacity(Tile.Tile8,cells) : U32}:
  match activation:
    case Activations.Identity{}: {==}
    case Activations.ReLU{}: MapReadback.capacity(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.relu_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.ReLU6{}: MapReadback.capacity(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.relu6_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.LeakyReLU{+slope}: MapReadback.capacity(~Tile.Tile8,~F32,~TiledMap.lanes(~F32,~Pipeline.leaky_relu_value),U32.to_nat(count),0,slope,cells)
    case Activations.Sigmoid{}: MapReadback.capacity(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.sigmoid_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.SiLU{}: MapReadback.capacity(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.silu_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.Tanh{}: MapReadback.capacity(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.tanh_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.HardSigmoid{}: MapReadback.capacity(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.hard_sigmoid_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.HardSwish{}: MapReadback.capacity(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.hard_swish_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.GELUTanh{}: MapReadback.capacity(~Tile.Tile8,~Unit,~GeluLanes.tile,U32.to_nat(count),0,Unit{},cells)
    case Activations.GELU{}: MapReadback.capacity(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~Activations.gelu_value)),U32.to_nat(count),0,Unit{},cells)
    case Activations.Custom{operations}: MapReadback.capacity(~Tile.Tile8,~Unit,~TiledMap.lanes(~Unit,~Pipeline.activated_value(~custom)),U32.to_nat(count),0,Unit{},cells)

# A lane of an activated cell is the activation of the lane.
def activation_lane(~custom: F32 -> F32,+activation: Activations.Activation,+tile: Tile.Tile8,+lane: Nat) ->
  {TileLaws.select_lane(lane,activation_tile(~custom,activation,tile)) == activation_value(~custom,activation,TileLaws.select_lane(lane,tile)) : F32}:
  match tile:
    case Tile.Tile{+a,+b,+c,+d,+e,+f,+g,+h}:
      match lane:
        case 0n: {==}
        case 1n+ +l1:
          match l1:
            case 0n: {==}
            case 1n+ +l2:
              match l2:
                case 0n: {==}
                case 1n+ +l3:
                  match l3:
                    case 0n: {==}
                    case 1n+ +l4:
                      match l4:
                        case 0n: {==}
                        case 1n+ +l5:
                          match l5:
                            case 0n: {==}
                            case 1n+ +l6:
                              match l6:
                                case 0n: {==}
                                case 1n+ +l7:
                                  {==}

# The activated cells of a tree read back as the activation tree.
def reflected(~custom: F32 -> F32,+activation: Activations.Activation,+count: U32,+tree: Tree.Tree<Tile.Tile8>,+length: Nat) ->
  {Buffer.Content{Tree.reflect(Tile.Tile8,activation_array(~custom,activation,count,Tree.pack(Tile.Tile8,tree))),length} ==
    Buffer.Content{activation_tree(~custom,activation,count,tree),length} : Buffer.Content<Tile.Tile8>}:
  +activated = activation_tree(~custom,activation,count,tree)
  Equal.cong(Tree.Tree<Tile.Tile8>,Buffer.Content<Tile.Tile8>,cells => Buffer.Content{cells,length},
    Tree.reflect(Tile.Tile8,activation_array(~custom,activation,count,Tree.pack(Tile.Tile8,tree))),activated,
    Equal.trans(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,activation_array(~custom,activation,count,Tree.pack(Tile.Tile8,tree))),
      Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,activated)),activated,
      Equal.cong(Array<Tile.Tile8>,Tree.Tree<Tile.Tile8>,cells => Tree.reflect(Tile.Tile8,cells),activation_array(~custom,activation,count,Tree.pack(Tile.Tile8,tree)),
        Tree.pack(Tile.Tile8,activated),activation_model(~custom,activation,count,tree)),
      StorageLaws.reflect_packed(Tile.Tile8,activated)))

# The activated cells of a buffer: the activation tree of its cells, at its
# length.
def activated_contents(~custom: F32 -> F32,+activation: Activations.Activation,+count: U32,+model: Buffer.Model<Tile.Tile8>) ->
  {Buffer.contents(~Tile.Tile8,Pipeline.activated(~custom,activation,count,Buffer.pack(~Tile.Tile8,model))) ==
    Buffer.Content{activation_tree(~custom,activation,count,Buffer.tree(~Tile.Tile8,model)),Buffer.length(~Tile.Tile8,model)} : Buffer.Content<Tile.Tile8>}:
  match activation model:
    case Activations.Identity{} Buffer.Model{+tree,+length,certificate}: reflected(~custom,Activations.Identity{},count,tree,length)
    case Activations.ReLU{} Buffer.Model{+tree,+length,certificate}: reflected(~custom,Activations.ReLU{},count,tree,length)
    case Activations.ReLU6{} Buffer.Model{+tree,+length,certificate}: reflected(~custom,Activations.ReLU6{},count,tree,length)
    case Activations.LeakyReLU{+slope} Buffer.Model{+tree,+length,certificate}: reflected(~custom,Activations.LeakyReLU{slope},count,tree,length)
    case Activations.Sigmoid{} Buffer.Model{+tree,+length,certificate}: reflected(~custom,Activations.Sigmoid{},count,tree,length)
    case Activations.SiLU{} Buffer.Model{+tree,+length,certificate}: reflected(~custom,Activations.SiLU{},count,tree,length)
    case Activations.Tanh{} Buffer.Model{+tree,+length,certificate}: reflected(~custom,Activations.Tanh{},count,tree,length)
    case Activations.HardSigmoid{} Buffer.Model{+tree,+length,certificate}: reflected(~custom,Activations.HardSigmoid{},count,tree,length)
    case Activations.HardSwish{} Buffer.Model{+tree,+length,certificate}: reflected(~custom,Activations.HardSwish{},count,tree,length)
    case Activations.GELUTanh{} Buffer.Model{+tree,+length,certificate}: reflected(~custom,Activations.GELUTanh{},count,tree,length)
    case Activations.GELU{} Buffer.Model{+tree,+length,certificate}: reflected(~custom,Activations.GELU{},count,tree,length)
    case Activations.Custom{operations} Buffer.Model{+tree,+length,certificate}: reflected(~custom,Activations.Custom{operations},count,tree,length)
