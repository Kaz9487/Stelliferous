# A traversal in blocks of size consecutive positions, one block after
# another. Block b reads and places with the context inner(context,b,start),
# start being the first position of the block, so a reader whose index is a
# quotient and a remainder of the position finds them once per block instead
# of once per element.
import Base
import ./traversal_array.bend as Traversal
import ./traversal_loop.bend as Loop

type Blocks<-Context: Data> is Data:
  Blocks{block: U32,size: U32,context: Context}

# Positions written where they are read.
def in_place(~Inner: Data,inner: Inner,position: U32) -> U32:
  position

def next_block(~Reader: Type,~Element: Data,~Context: Data,~Inner: Data,block: U32,size: U32,context: Context,
  state: Traversal.State<&1,&1,Reader,Array<Element>,Inner>) -> Traversal.State<&1,&1,Reader,Array<Element>,Blocks<Context>>:
  Traversal.State{input,output,position,inner} = state
  Traversal.State{input,output,position,Blocks{(block + 1 : U32),size,context}}

def block_step(~Reader: Type,~Element: Data,~Context: Data,~Inner: Data,~inner: Context -> U32 -> U32 -> Inner,
  ~read: Reader -> Inner -> U32 -> Reader & Element,~place: Inner -> U32 -> U32,
  state: Traversal.State<&1,&1,Reader,Array<Element>,Blocks<Context>>) -> Traversal.State<&1,&1,Reader,Array<Element>,Blocks<Context>>:
  Traversal.State{input,output,+position,Blocks{+block,+size,+context}} = state
  next_block(~Reader,~Element,~Context,~Inner,block,size,context,
    Traversal.run(~Reader,~Element,~Inner,~read,~place,U32.to_nat(size),Traversal.State{input,output,position,inner(context,block,position)}))

def run(~Reader: Type,~Element: Data,~Context: Data,~Inner: Data,~inner: Context -> U32 -> U32 -> Inner,
  ~read: Reader -> Inner -> U32 -> Reader & Element,~place: Inner -> U32 -> U32,
  blocks: Nat,size: U32,context: Context,input: Reader,output: Array<Element>) -> Reader & Array<Element>:
  Traversal.finish(~Reader,~Element,~Blocks<Context>,
    Loop.run(~Traversal.State<&1,&1,Reader,Array<Element>,Blocks<Context>>,~block_step(~Reader,~Element,~Context,~Inner,~inner,~read,~place),blocks,
      Traversal.State{input,output,0,Blocks{0,size,context}}))
