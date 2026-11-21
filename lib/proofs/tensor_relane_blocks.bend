# The blocks of a target piece of a relane: block by block, each from the
# pieces the task holds, into a buffer of zero cells.

import Base
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ../tensor_move.bend as Move
import ../tensor_tiled.bend as Tiled
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./tensor_move_readback.bend as Held
import ./tensor_relane_block.bend as Block
import ./storage_shape.bend as Shape
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_extrema.bend as Extrema

# ---- The model of the blocks ----

def first_block() -> Tiled.Block:
  Tiled.Block{0,0,0,0,1,1,0}

def block_of(+axes: List<&2,Tiled.Ranged>,+index: U32) -> Tiled.Block:
  Tiled.block_at(axes,index,first_block())

# A block writes its cells up to an eighth at most.
def SmallHigh(+block: Tiled.Block) -> Data:
  Tiled.Block{source,target,stride,low,+high,reads,step} = block
  {U32.is_le(high,8) == True{} : Bool}

def axis_small(role: Tiled.Role,+block: Tiled.Block,+index: U32,+low: U32,+extent: U32,+source: U32,+lanes: U32,+target: U32,+small: SmallHigh(block)) ->
  SmallHigh(Tiled.block_axis(role,block,index,low,extent,source,lanes,target)):
  match role block:
    case Tiled.Shared{} Tiled.Block{+block_source,+block_target,+stride,+block_low,+block_high,+reads,+step}: small
    case Tiled.Groups{} Tiled.Block{+block_source,+block_target,+stride,+block_low,+block_high,+reads,+step}: small
    case Tiled.Lanes{} Tiled.Block{+block_source,+block_target,+stride,+block_low,+block_high,+reads,+step}:
      +first = ((low / 8 + index : U32) * 8 : U32)
      Extrema.minimum_right((low + extent - first : U32),8)

def at_small(axes: List<&2,Tiled.Ranged>,+index: U32,+block: Tiled.Block,+small: SmallHigh(block)) -> SmallHigh(Tiled.block_at(axes,index,block)):
  match axes:
    case Nil{}: small
    case Con{Tiled.Ranged{Tiled.Blocked{+role,+source,+lanes},+low,+extent,+count,+inner,+target},+rest}:
      +position = (index / U32.max(inner,1) % U32.max(count,1) : U32)
      at_small(rest,index,Tiled.block_axis(role,block,position,low,extent,source,lanes,target),axis_small(role,block,position,low,extent,source,lanes,target,small))

def block_of_small(+axes: List<&2,Tiled.Ranged>,+index: U32) -> SmallHigh(block_of(axes,index)):
  at_small(axes,index,first_block(),{==})

# The holding the blocks from index on leave.
def blocks_holding(fuel: Nat,+index: U32,+axes: List<&2,Tiled.Ranged>,+holding: Held.Holding<Tile.Tile8>) -> Held.Holding<Tile.Tile8>:
  match fuel:
    case 0n: holding
    case 1n+rest: blocks_holding(rest,(index + 1 : U32),axes,Block.block_holding(block_of(axes,index),holding))

# The buffer the blocks from index on leave.
def blocks_buffer(fuel: Nat,+index: U32,+axes: List<&2,Tiled.Ranged>,+holding: Held.Holding<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  match fuel:
    case 0n: output
    case 1n+rest:
      +block = block_of(axes,index)
      blocks_buffer(rest,(index + 1 : U32),axes,Block.block_holding(block,holding),Block.block_output(block,Block.block_eight(block,holding),output))

# The blocks on packed pieces leave the packed model holding and the buffer
# of the model.
def blocks_packed(fuel: Nat,+index: U32,+axes: List<&2,Tiled.Ranged>,+holding: Held.Holding<Tile.Tile8>,-output: Storage.Buffer<Tile.Tile8>) ->
  {Tiled.blocks(fuel,index,axes,(Held.pack(~Tile.Tile8,holding),output)) ==
    (Held.pack(~Tile.Tile8,blocks_holding(fuel,index,axes,holding)),blocks_buffer(fuel,index,axes,holding,output)) : Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}:
  match fuel:
    case 0n: {==}
    case 1n+ +rest:
      +block = block_of(axes,index)
      +next = Block.block_holding(block,holding)
      %Equal.sym(Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>,Tiled.block_written(block,Held.pack(~Tile.Tile8,holding),output),
          (Held.pack(~Tile.Tile8,next),Block.block_output(block,Block.block_eight(block,holding),output)),Block.block_written_packed(block,holding,output)) :
        {Tiled.blocks(rest,(index + 1 : U32),axes,_) ==
          (Held.pack(~Tile.Tile8,blocks_holding(rest,(index + 1 : U32),axes,next)),
            blocks_buffer(rest,(index + 1 : U32),axes,next,Block.block_output(block,Block.block_eight(block,holding),output))) :
          Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
      blocks_packed(rest,(index + 1 : U32),axes,next,Block.block_output(block,Block.block_eight(block,holding),output))

# ---- The tree the blocks leave ----

def block_tree(+block: Tiled.Block,+eight: Tiled.Eight,tree: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  Tiled.Block{source,+target,+stride,+low,+high,reads,step} = block
  Block.written_tree(8n,U32.is_lt(low,high),low,high,target,stride,Tiled.transposed(eight),tree)

def blocks_tree(fuel: Nat,+index: U32,+axes: List<&2,Tiled.Ranged>,+holding: Held.Holding<Tile.Tile8>,tree: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  match fuel:
    case 0n: tree
    case 1n+rest:
      +block = block_of(axes,index)
      blocks_tree(rest,(index + 1 : U32),axes,Block.block_holding(block,holding),block_tree(block,Block.block_eight(block,holding),tree))

def block_contents(+block: Tiled.Block,+eight: Tiled.Eight,output: Storage.Buffer<Tile.Tile8>,+small: SmallHigh(block)) ->
  {Buffer.contents(~Tile.Tile8,Block.block_output(block,eight,output)) ==
    Buffer.Content{block_tree(block,eight,Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,output))),Buffer.content_length(~Tile.Tile8,Buffer.contents(~Tile.Tile8,output))} :
    Buffer.Content<Tile.Tile8>}:
  match block:
    case Tiled.Block{+source,+target,+stride,+low,+high,+reads,+step}:
      Block.writes_contents(low,high,target,stride,Tiled.transposed(eight),output,small)

def blocks_contents_packed(fuel: Nat,+index: U32,+axes: List<&2,Tiled.Ranged>,+holding: Held.Holding<Tile.Tile8>,+model: Buffer.Model<Tile.Tile8>) ->
  {Buffer.contents(~Tile.Tile8,blocks_buffer(fuel,index,axes,holding,Buffer.pack(~Tile.Tile8,model))) ==
    Buffer.Content{blocks_tree(fuel,index,axes,holding,Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,model)))),
      Buffer.content_length(~Tile.Tile8,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,model)))} : Buffer.Content<Tile.Tile8>}:
  match fuel:
    case 0n: Buffer.content_eta(~Tile.Tile8,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,model)))
    case 1n+ +rest:
      +block = block_of(axes,index)
      +next = Block.block_holding(block,holding)
      +eight = Block.block_eight(block,holding)
      +old = Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,model))
      +observation = Buffer.observe(~Tile.Tile8,Block.block_output(block,eight,Buffer.pack(~Tile.Tile8,model)))
      +written = Buffer.observed_model(~Tile.Tile8,Block.block_output(block,eight,Buffer.pack(~Tile.Tile8,model)),observation)
      Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,blocks_buffer(rest,(index + 1 : U32),axes,next,Block.block_output(block,eight,Buffer.pack(~Tile.Tile8,model)))),
        Buffer.Content{blocks_tree(rest,(index + 1 : U32),axes,next,Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,Block.block_output(block,eight,Buffer.pack(~Tile.Tile8,model))))),
          Buffer.content_length(~Tile.Tile8,Buffer.contents(~Tile.Tile8,Block.block_output(block,eight,Buffer.pack(~Tile.Tile8,model))))},
        Buffer.Content{blocks_tree(rest,(index + 1 : U32),axes,next,block_tree(block,eight,Buffer.content_tree(~Tile.Tile8,old))),Buffer.content_length(~Tile.Tile8,old)},
        {%Buffer.observed_equation(~Tile.Tile8,Block.block_output(block,eight,Buffer.pack(~Tile.Tile8,model)),observation) :
            {Buffer.contents(~Tile.Tile8,blocks_buffer(rest,(index + 1 : U32),axes,next,_)) ==
              Buffer.Content{blocks_tree(rest,(index + 1 : U32),axes,next,Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,_))),
                Buffer.content_length(~Tile.Tile8,Buffer.contents(~Tile.Tile8,_))} : Buffer.Content<Tile.Tile8>}
          blocks_contents_packed(rest,(index + 1 : U32),axes,next,written) :
          {Buffer.contents(~Tile.Tile8,blocks_buffer(rest,(index + 1 : U32),axes,next,Block.block_output(block,eight,Buffer.pack(~Tile.Tile8,model)))) ==
            Buffer.Content{blocks_tree(rest,(index + 1 : U32),axes,next,Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,Block.block_output(block,eight,Buffer.pack(~Tile.Tile8,model))))),
              Buffer.content_length(~Tile.Tile8,Buffer.contents(~Tile.Tile8,Block.block_output(block,eight,Buffer.pack(~Tile.Tile8,model))))} : Buffer.Content<Tile.Tile8>}},
        Equal.cong(Buffer.Content<Tile.Tile8>,Buffer.Content<Tile.Tile8>,
          content => Buffer.Content{blocks_tree(rest,(index + 1 : U32),axes,next,Buffer.content_tree(~Tile.Tile8,content)),Buffer.content_length(~Tile.Tile8,content)},
          Buffer.contents(~Tile.Tile8,Block.block_output(block,eight,Buffer.pack(~Tile.Tile8,model))),
          Buffer.Content{block_tree(block,eight,Buffer.content_tree(~Tile.Tile8,old)),Buffer.content_length(~Tile.Tile8,old)},
          block_contents(block,eight,Buffer.pack(~Tile.Tile8,model),block_of_small(axes,index))))

# The contents the blocks leave: the tree the blocks leave.
def blocks_contents(fuel: Nat,+index: U32,+axes: List<&2,Tiled.Ranged>,+holding: Held.Holding<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>) ->
  {Buffer.contents(~Tile.Tile8,blocks_buffer(fuel,index,axes,holding,output)) ==
    Buffer.Content{blocks_tree(fuel,index,axes,holding,Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,output))),
      Buffer.content_length(~Tile.Tile8,Buffer.contents(~Tile.Tile8,output))} : Buffer.Content<Tile.Tile8>}:
  +observation = Buffer.observe(~Tile.Tile8,output)
  %Buffer.observed_equation(~Tile.Tile8,output,observation) :
    {Buffer.contents(~Tile.Tile8,blocks_buffer(fuel,index,axes,holding,_)) ==
      Buffer.Content{blocks_tree(fuel,index,axes,holding,Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,_))),
        Buffer.content_length(~Tile.Tile8,Buffer.contents(~Tile.Tile8,_))} : Buffer.Content<Tile.Tile8>}
  blocks_contents_packed(fuel,index,axes,holding,Buffer.observed_model(~Tile.Tile8,output,observation))

# ---- A cell after the blocks ----

def block_hit(+block: Tiled.Block,+query: U32) -> Bool:
  Tiled.Block{source,+target,+stride,+low,+high,reads,step} = block
  Block.writes_hit(8n,U32.is_lt(low,high),low,high,target,stride,query)

def BlockInside(+block: Tiled.Block,+capacity: U32) -> Data:
  Tiled.Block{source,+target,+stride,+low,+high,reads,step} = block
  Block.WritesInside(8n,U32.is_lt(low,high),low,high,target,stride,capacity)

def BlockSound(+block: Tiled.Block,+eight: Tiled.Eight,+query: U32,+expected: Tile.Tile8) -> Data:
  Tiled.Block{source,+target,+stride,+low,+high,reads,step} = block
  Block.WritesSound(8n,U32.is_lt(low,high),low,high,target,stride,Tiled.transposed(eight),query,expected)

def block_balanced(+block: Tiled.Block,+eight: Tiled.Eight,+depth: Nat,+tree: Tree.Tree<Tile.Tile8>,+balanced: Shape.Balanced(Tile.Tile8,depth,tree)) ->
  Shape.Balanced(Tile.Tile8,depth,block_tree(block,eight,tree)):
  match block:
    case Tiled.Block{+source,+target,+stride,+low,+high,+reads,+step}:
      Block.written_balanced(8n,U32.is_lt(low,high),low,high,target,stride,Tiled.transposed(eight),depth,tree,balanced)

def block_capacity(+block: Tiled.Block,+eight: Tiled.Eight,+tree: Tree.Tree<Tile.Tile8>) ->
  {Tree.capacity(Tile.Tile8,block_tree(block,eight,tree)) == Tree.capacity(Tile.Tile8,tree) : U32}:
  match block:
    case Tiled.Block{+source,+target,+stride,+low,+high,+reads,+step}:
      Block.written_capacity(8n,U32.is_lt(low,high),low,high,target,stride,Tiled.transposed(eight),tree)

# The blocks keep the depth and the capacity of the tree.
def blocks_balanced(fuel: Nat,+index: U32,+axes: List<&2,Tiled.Ranged>,+holding: Held.Holding<Tile.Tile8>,+depth: Nat,+tree: Tree.Tree<Tile.Tile8>,
  +balanced: Shape.Balanced(Tile.Tile8,depth,tree)) -> Shape.Balanced(Tile.Tile8,depth,blocks_tree(fuel,index,axes,holding,tree)):
  match fuel:
    case 0n: balanced
    case 1n+ +rest:
      +block = block_of(axes,index)
      +eight = Block.block_eight(block,holding)
      blocks_balanced(rest,(index + 1 : U32),axes,Block.block_holding(block,holding),depth,block_tree(block,eight,tree),block_balanced(block,eight,depth,tree,balanced))

def blocks_capacity(fuel: Nat,+index: U32,+axes: List<&2,Tiled.Ranged>,+holding: Held.Holding<Tile.Tile8>,+tree: Tree.Tree<Tile.Tile8>) ->
  {Tree.capacity(Tile.Tile8,blocks_tree(fuel,index,axes,holding,tree)) == Tree.capacity(Tile.Tile8,tree) : U32}:
  match fuel:
    case 0n: {==}
    case 1n+ +rest:
      +block = block_of(axes,index)
      +eight = Block.block_eight(block,holding)
      Equal.trans(U32,Tree.capacity(Tile.Tile8,blocks_tree(rest,(index + 1 : U32),axes,Block.block_holding(block,holding),block_tree(block,eight,tree))),
        Tree.capacity(Tile.Tile8,block_tree(block,eight,tree)),Tree.capacity(Tile.Tile8,tree),
        blocks_capacity(rest,(index + 1 : U32),axes,Block.block_holding(block,holding),block_tree(block,eight,tree)),block_capacity(block,eight,tree))

# The blocks keep the order of the holding.
def block_order(+block: Tiled.Block,+holding: Held.Holding<Tile.Tile8>) ->
  {Held.order(~Tile.Tile8,Block.block_holding(block,holding)) == Held.order(~Tile.Tile8,holding) : Held.Stack<Tile.Tile8>}:
  match block:
    case Tiled.Block{+source,+target,+stride,+low,+high,+reads,+step}:
      Block.block_order(reads,source,step,holding)

def blocks_order(fuel: Nat,+index: U32,+axes: List<&2,Tiled.Ranged>,+holding: Held.Holding<Tile.Tile8>) ->
  {Held.order(~Tile.Tile8,blocks_holding(fuel,index,axes,holding)) == Held.order(~Tile.Tile8,holding) : Held.Stack<Tile.Tile8>}:
  match fuel:
    case 0n: {==}
    case 1n+ +rest:
      +block = block_of(axes,index)
      Equal.trans(Held.Stack<Tile.Tile8>,Held.order(~Tile.Tile8,blocks_holding(rest,(index + 1 : U32),axes,Block.block_holding(block,holding))),
        Held.order(~Tile.Tile8,Block.block_holding(block,holding)),Held.order(~Tile.Tile8,holding),
        blocks_order(rest,(index + 1 : U32),axes,Block.block_holding(block,holding)),block_order(block,holding))

# A block that writes inside a capacity writes inside every wider one.
def block_wider(+block: Tiled.Block,+capacity: U32,+wider: U32,+le: {U32.is_le(capacity,wider) == True{} : Bool},+inside: BlockInside(block,capacity)) ->
  BlockInside(block,wider):
  match block:
    case Tiled.Block{+source,+target,+stride,+low,+high,+reads,+step}:
      Block.writes_wider(8n,U32.is_lt(low,high),low,high,target,stride,capacity,wider,le,inside)

# A cell after a block holds expected when the block hits it, and what it
# held otherwise.
def block_read(+block: Tiled.Block,+eight: Tiled.Eight,+depth: Nat,+tree: Tree.Tree<Tile.Tile8>,+balanced: Shape.Balanced(Tile.Tile8,depth,tree),
  +limit: {Nat.is_le(depth,30n) == True{} : Bool},+inside: BlockInside(block,Tree.capacity(Tile.Tile8,tree)),+query: U32,
  +query_inside: {U32.is_lt(query,Tree.capacity(Tile.Tile8,tree)) == True{} : Bool},+expected: Tile.Tile8,+sound: BlockSound(block,eight,query,expected)) ->
  {Tree.read(Tile.Tile8,block_tree(block,eight,tree),query) == Bool.pick(Tile.Tile8,block_hit(block,query),expected,Tree.read(Tile.Tile8,tree,query)) : Tile.Tile8}:
  match block:
    case Tiled.Block{+source,+target,+stride,+low,+high,+reads,+step}:
      +cells = Tiled.transposed(eight)
      Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Block.written_tree(8n,U32.is_lt(low,high),low,high,target,stride,cells,tree),query),
        Block.written_value(8n,U32.is_lt(low,high),low,high,target,stride,cells,query,Tree.read(Tile.Tile8,tree,query)),
        Bool.pick(Tile.Tile8,Block.writes_hit(8n,U32.is_lt(low,high),low,high,target,stride,query),expected,Tree.read(Tile.Tile8,tree,query)),
        Block.written_read(8n,U32.is_lt(low,high),low,high,target,stride,cells,depth,tree,balanced,limit,inside,query,query_inside),
        Block.written_sound(8n,U32.is_lt(low,high),low,high,target,stride,cells,query,expected,Tree.read(Tile.Tile8,tree,query),sound))

# Some block from index on hits query.
def blocks_hit(fuel: Nat,+index: U32,+axes: List<&2,Tiled.Ranged>,+query: U32) -> Bool:
  match fuel:
    case 0n: False{}
    case 1n+rest: block_hit(block_of(axes,index),query) || blocks_hit(rest,(index + 1 : U32),axes,query)

# Every block from index on writes inside a capacity, and every write of it
# that hits query writes expected.
def BlocksSound(fuel: Nat,+index: U32,+axes: List<&2,Tiled.Ranged>,+holding: Held.Holding<Tile.Tile8>,+capacity: U32,+query: U32,+expected: Tile.Tile8) -> Data:
  match fuel:
    case 0n: Unit
    case 1n+rest:
      +block = block_of(axes,index)
      Held.Both<BlockInside(block,capacity),Held.Both<BlockSound(block,Block.block_eight(block,holding),query,expected),
        BlocksSound(rest,(index + 1 : U32),axes,Block.block_holding(block,holding),capacity,query,expected)>>

# A cell after the blocks holds expected when a block hits it, and what it
# held otherwise.
def blocks_read(fuel: Nat,+index: U32,+axes: List<&2,Tiled.Ranged>,+holding: Held.Holding<Tile.Tile8>,+depth: Nat,+tree: Tree.Tree<Tile.Tile8>,
  +balanced: Shape.Balanced(Tile.Tile8,depth,tree),+limit: {Nat.is_le(depth,30n) == True{} : Bool},+capacity: U32,
  +fits: {U32.is_le(capacity,Tree.capacity(Tile.Tile8,tree)) == True{} : Bool},+query: U32,+query_inside: {U32.is_lt(query,capacity) == True{} : Bool},+expected: Tile.Tile8,
  +sound: BlocksSound(fuel,index,axes,holding,capacity,query,expected)) ->
  {Tree.read(Tile.Tile8,blocks_tree(fuel,index,axes,holding,tree),query) == Bool.pick(Tile.Tile8,blocks_hit(fuel,index,axes,query),expected,Tree.read(Tile.Tile8,tree,query)) : Tile.Tile8}:
  match fuel:
    case 0n: {==}
    case 1n+ +rest:
      Held.Both{+inside,Held.Both{+here,+later}} = sound
      +block = block_of(axes,index)
      +eight = Block.block_eight(block,holding)
      +next = Block.block_holding(block,holding)
      +written = block_tree(block,eight,tree)
      +hit = block_hit(block,query)
      +rest_hit = blocks_hit(rest,(index + 1 : U32),axes,query)
      Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,blocks_tree(rest,(index + 1 : U32),axes,next,written),query),
        Bool.pick(Tile.Tile8,rest_hit,expected,Tree.read(Tile.Tile8,written,query)),Bool.pick(Tile.Tile8,hit || rest_hit,expected,Tree.read(Tile.Tile8,tree,query)),
        blocks_read(rest,(index + 1 : U32),axes,next,depth,written,block_balanced(block,eight,depth,tree,balanced),limit,capacity,
          {%Equal.sym(U32,Tree.capacity(Tile.Tile8,written),Tree.capacity(Tile.Tile8,tree),block_capacity(block,eight,tree)) : {U32.is_le(capacity,_) == True{} : Bool}
            fits : {U32.is_le(capacity,Tree.capacity(Tile.Tile8,written)) == True{} : Bool}},query,query_inside,expected,later),
        Equal.trans(Tile.Tile8,Bool.pick(Tile.Tile8,rest_hit,expected,Tree.read(Tile.Tile8,written,query)),
          Bool.pick(Tile.Tile8,rest_hit,expected,Bool.pick(Tile.Tile8,hit,expected,Tree.read(Tile.Tile8,tree,query))),
          Bool.pick(Tile.Tile8,hit || rest_hit,expected,Tree.read(Tile.Tile8,tree,query)),
          Equal.cong(Tile.Tile8,Tile.Tile8,old => Bool.pick(Tile.Tile8,rest_hit,expected,old),Tree.read(Tile.Tile8,written,query),
            Bool.pick(Tile.Tile8,hit,expected,Tree.read(Tile.Tile8,tree,query)),
            block_read(block,eight,depth,tree,balanced,limit,block_wider(block,capacity,Tree.capacity(Tile.Tile8,tree),fits,inside),query,
              Bounds.uint_trans(query,capacity,Tree.capacity(Tile.Tile8,tree),query_inside,fits),expected,here)),
          Block.pick_merge(hit,rest_hit,expected,Tree.read(Tile.Tile8,tree,query))))
