# Measured flat FP32 tile representation; logical views stay out of hot loops.
import Base
import ./kernel_block.bend as Block
import ./math_sequence.bend as Sequence

type Tile8 is Data:
  Tile{a0: F32, a1: F32, a2: F32, a3: F32, a4: F32, a5: F32, a6: F32, a7: F32}

def filled(+value: F32) -> Tile8:
  Tile{value,value,value,value,value,value,value,value}

def lane0(tile: Tile8) -> F32:
  Tile{a,b,c,d,e,f,g,h} = tile
  a

def lane1(tile: Tile8) -> F32:
  Tile{a,b,c,d,e,f,g,h} = tile
  b

def lane2(tile: Tile8) -> F32:
  Tile{a,b,c,d,e,f,g,h} = tile
  c

def lane3(tile: Tile8) -> F32:
  Tile{a,b,c,d,e,f,g,h} = tile
  d

def lane4(tile: Tile8) -> F32:
  Tile{a,b,c,d,e,f,g,h} = tile
  e

def lane5(tile: Tile8) -> F32:
  Tile{a,b,c,d,e,f,g,h} = tile
  f

def lane6(tile: Tile8) -> F32:
  Tile{a,b,c,d,e,f,g,h} = tile
  g

def lane7(tile: Tile8) -> F32:
  Tile{a,b,c,d,e,f,g,h} = tile
  h

def plus(a: Tile8,b: Tile8) -> Tile8:
  Tile{a0,a1,a2,a3,a4,a5,a6,a7} = a
  Tile{b0,b1,b2,b3,b4,b5,b6,b7} = b
  Tile{(a0 + b0 : F32),(a1 + b1 : F32),(a2 + b2 : F32),(a3 + b3 : F32),(a4 + b4 : F32),(a5 + b5 : F32),(a6 + b6 : F32),(a7 + b7 : F32)}

def minus(a: Tile8,b: Tile8) -> Tile8:
  Tile{a0,a1,a2,a3,a4,a5,a6,a7} = a
  Tile{b0,b1,b2,b3,b4,b5,b6,b7} = b
  Tile{(a0 - b0 : F32),(a1 - b1 : F32),(a2 - b2 : F32),(a3 - b3 : F32),(a4 - b4 : F32),(a5 - b5 : F32),(a6 - b6 : F32),(a7 - b7 : F32)}

def times(a: Tile8,b: Tile8) -> Tile8:
  Tile{a0,a1,a2,a3,a4,a5,a6,a7} = a
  Tile{b0,b1,b2,b3,b4,b5,b6,b7} = b
  Tile{(a0 * b0 : F32),(a1 * b1 : F32),(a2 * b2 : F32),(a3 * b3 : F32),(a4 * b4 : F32),(a5 * b5 : F32),(a6 * b6 : F32),(a7 * b7 : F32)}

def scaled(tile: Tile8,+value: F32) -> Tile8:
  Tile{a,b,c,d,e,f,g,h} = tile
  Tile{(a * value : F32),(b * value : F32),(c * value : F32),(d * value : F32),(e * value : F32),(f * value : F32),(g * value : F32),(h * value : F32)}

# One ordered multiply-add per lane with one weight: times multiplies a lane
# value and the weight in the order the contraction's specification does.
def update(~times: F32 -> F32 -> F32,accumulator: Tile8,input: Tile8,weight: F32) -> Tile8:
  Tile{a,b,c,d,e,f,g,h} = accumulator
  Tile{x0,x1,x2,x3,x4,x5,x6,x7} = input
  +weight = weight
  Tile{(a + times(x0,weight) : F32),(b + times(x1,weight) : F32),(c + times(x2,weight) : F32),(d + times(x3,weight) : F32),
    (e + times(x4,weight) : F32),(f + times(x5,weight) : F32),(g + times(x6,weight) : F32),(h + times(x7,weight) : F32)}

# Two tiles stored as one array element, sixteen words read and written
# through one wrapped index. Clang proves adjacent only the words of one
# element, so a vector over sixteen lanes needs them in one element.
type Tile16 is Data:
  Tiles{lower: Tile8,upper: Tile8}

def filled_pair(+value: F32) -> Tile16:
  Tiles{filled(value),filled(value)}

# update on both tiles with the same weight.
def update_pair(~times: F32 -> F32 -> F32,accumulator: Tile16,+input: Tile16,+weight: F32) -> Tile16:
  Tiles{lower,upper} = accumulator
  Tiles{low,high} = input
  Tiles{update(~times,lower,low,weight),update(~times,upper,high,weight)}

def plus_pair(a: Tile16,b: Tile16) -> Tile16:
  Tiles{a0,a1} = a
  Tiles{b0,b1} = b
  Tiles{plus(a0,b0),plus(a1,b1)}

# One ordered multiply-add per lane: the lane of the input times the same
# lane of the weight, added to the lane of the accumulator.
def update_lanes(accumulator: Tile8,input: Tile8,weight: Tile8) -> Tile8:
  Tile{a,b,c,d,e,f,g,h} = accumulator
  Tile{x0,x1,x2,x3,x4,x5,x6,x7} = input
  Tile{w0,w1,w2,w3,w4,w5,w6,w7} = weight
  Tile{(a + (x0 * w0 : F32) : F32),(b + (x1 * w1 : F32) : F32),(c + (x2 * w2 : F32) : F32),(d + (x3 * w3 : F32) : F32),
    (e + (x4 * w4 : F32) : F32),(f + (x5 * w5 : F32) : F32),(g + (x6 * w6 : F32) : F32),(h + (x7 * w7 : F32) : F32)}

def from_sequence(vector: Sequence.Values(F32,8n)) -> Tile8:
  Sequence.Elements{a0,Sequence.Elements{a1,Sequence.Elements{a2,Sequence.Elements{a3,Sequence.Elements{a4,Sequence.Elements{a5,Sequence.Elements{a6,
          Sequence.Elements{a7,Unit{}}}}}}}}} = vector
  Tile{a0,a1,a2,a3,a4,a5,a6,a7}

type HeadContext<-Context: Data> is Data:
  HeadContext{head: F32,context: Context}

def read_with_head(
  ~State: Type,~Context: Data,~read: State -> Context -> Nat -> State & F32,
  state: State,combined: HeadContext<Context>,index: Nat
) -> State & F32:
  HeadContext{head,context} = combined
  match index:
    case 0n: (state,head)
    case 1n+rest: read(state,context,1n+rest)

def pairs_value(vector: Block.Twin<Block.Twin<Block.Twin<F32>>>) -> Tile8:
  Block.Twin{Block.Twin{Block.Twin{a0,a1},Block.Twin{a2,a3}},Block.Twin{Block.Twin{a4,a5},
      Block.Twin{a6,a7}}} = vector
  Tile{a0,a1,a2,a3,a4,a5,a6,a7}

def finish_pairs(~State: Type,result: State & Block.Twin<Block.Twin<Block.Twin<F32>>>) -> State & Tile8:
  (state,vector) = result
  (state,pairs_value(vector))

def read_tail(~State: Type,~Context: Data,~read: State -> Context -> Nat -> State & F32,first: State & F32,context: Context) -> State & Tile8:
  (state,head) = first
  finish_pairs(~State,
    Block.read_pair(~State,~HeadContext<Context>,~Block.Twin<Block.Twin<F32>>,
      ~Block.read_pair(~State,~HeadContext<Context>,~Block.Twin<F32>,
        ~Block.read_pair(~State,~HeadContext<Context>,~F32,
          ~read_with_head(~State,~Context,~read),~1n),~2n),~4n,
      state,HeadContext{head,context},0n))

# Logical view only; runtime kernels never traverse it.

def view(tile: Tile8) -> Sequence.Values(F32,8n):
  Tile{a0,a1,a2,a3,a4,a5,a6,a7} = tile
  Sequence.Elements{a0,Sequence.Elements{a1,Sequence.Elements{a2,Sequence.Elements{a3,Sequence.Elements{a4,Sequence.Elements{a5,Sequence.Elements{a6,
          Sequence.Elements{a7,Unit{}}}}}}}}}

def element_at(index: Sequence.Index(8n),tile: Tile8) -> F32:
  Sequence.get(F32,8n,index,view(tile))
