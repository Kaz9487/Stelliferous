# A traversal in blocks keeps the reader and the output depth: every block is
# the traversal of its own context.
import Base
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ../traversal_blocks.bend as Blocks
import ./storage_tree_model.bend as Tree
import ./storage_certificate.bend as Certificate
import ./traversal_witness.bend as TraversalWitness
import ./traversal_certificate.bend as TraversalCertificate

def next_block(~Reader: Type,~Element: Data,~Context: Data,~Inner: Data,+depth: Nat,block: U32,size: U32,context: Context,-input: Reader,
  -state: Traversal.State<&1,&1,Reader,Array<Element>,Inner>,evidence: TraversalWitness.StateEvidence<Reader,Element,Inner,input,depth,state>) ->
  TraversalWitness.StateEvidence<Reader,Element,Blocks.Blocks<Context>,input,depth,Blocks.next_block(~Reader,~Element,~Context,~Inner,block,size,context,state)>:
  match evidence:
    case TraversalWitness.StateEvidence{output,position,inner,equation,output_evidence}:
      %Equal.sym(Traversal.State<&1,&1,Reader,Array<Element>,Inner>,state,Traversal.State{input,output,position,inner},equation) :
        TraversalWitness.StateEvidence<Reader,Element,Blocks.Blocks<Context>,input,depth,Blocks.next_block(~Reader,~Element,~Context,~Inner,block,size,context,_)>
      TraversalWitness.StateEvidence{output,position,Blocks.Blocks{(block + 1 : U32),size,context},{==},output_evidence}

def block_step(~Reader: Type,~Element: Data,~Context: Data,~Inner: Data,~ReaderEvidence: @-reader: Reader -> Data,~inner: Context -> U32 -> U32 -> Inner,
  ~read: Reader -> Inner -> U32 -> Reader & Element,~place: Inner -> U32 -> U32,
  ~read_preserves: @-reader: Reader -> @context: Inner -> @index: U32 -> ReaderEvidence(reader) ->
    TraversalWitness.Sample<Reader,Element,reader,read(reader,context,index)>,
  +depth: Nat,-input: Reader,-state: Traversal.State<&1,&1,Reader,Array<Element>,Blocks.Blocks<Context>>,
  +input_evidence: ReaderEvidence(input),evidence: TraversalWitness.StateEvidence<Reader,Element,Blocks.Blocks<Context>,input,depth,state>) ->
  TraversalWitness.StateEvidence<Reader,Element,Blocks.Blocks<Context>,input,depth,Blocks.block_step(~Reader,~Element,~Context,~Inner,~inner,~read,~place,state)>:
  match evidence:
    case TraversalWitness.StateEvidence{output,+position,Blocks.Blocks{+block,+size,+context},equation,output_evidence}:
      %Equal.sym(Traversal.State<&1,&1,Reader,Array<Element>,Blocks.Blocks<Context>>,state,Traversal.State{input,output,position,Blocks.Blocks{block,size,context}},equation) :
        TraversalWitness.StateEvidence<Reader,Element,Blocks.Blocks<Context>,input,depth,Blocks.block_step(~Reader,~Element,~Context,~Inner,~inner,~read,~place,_)>
      next_block(~Reader,~Element,~Context,~Inner,depth,block,size,context,input,
        Traversal.run(~Reader,~Element,~Inner,~read,~place,U32.to_nat(size),Traversal.State{input,output,position,inner(context,block,position)}),
        TraversalWitness.run(~Reader,~Element,~Inner,~ReaderEvidence,~read,~place,~read_preserves,depth,U32.to_nat(size),input,
          Traversal.State{input,output,position,inner(context,block,position)},input_evidence,
          TraversalWitness.StateEvidence{output,position,inner(context,block,position),{==},output_evidence}))

def block_loop(~Reader: Type,~Element: Data,~Context: Data,~Inner: Data,~ReaderEvidence: @-reader: Reader -> Data,~inner: Context -> U32 -> U32 -> Inner,
  ~read: Reader -> Inner -> U32 -> Reader & Element,~place: Inner -> U32 -> U32,
  ~read_preserves: @-reader: Reader -> @context: Inner -> @index: U32 -> ReaderEvidence(reader) ->
    TraversalWitness.Sample<Reader,Element,reader,read(reader,context,index)>,
  +depth: Nat,count: Nat,-input: Reader,-state: Traversal.State<&1,&1,Reader,Array<Element>,Blocks.Blocks<Context>>,
  +input_evidence: ReaderEvidence(input),evidence: TraversalWitness.StateEvidence<Reader,Element,Blocks.Blocks<Context>,input,depth,state>) ->
  TraversalWitness.StateEvidence<Reader,Element,Blocks.Blocks<Context>,input,depth,
    Loop.run(~Traversal.State<&1,&1,Reader,Array<Element>,Blocks.Blocks<Context>>,~Blocks.block_step(~Reader,~Element,~Context,~Inner,~inner,~read,~place),count,state)>:
  match count:
    case 0n: evidence
    case 1n+rest:
      block_loop(~Reader,~Element,~Context,~Inner,~ReaderEvidence,~inner,~read,~place,~read_preserves,depth,rest,input,
        Blocks.block_step(~Reader,~Element,~Context,~Inner,~inner,~read,~place,state),input_evidence,
        block_step(~Reader,~Element,~Context,~Inner,~ReaderEvidence,~inner,~read,~place,~read_preserves,depth,input,state,input_evidence,evidence))

# The run hands the reader back and its output keeps the depth it had.
def run(~Reader: Type,~Element: Data,~Context: Data,~Inner: Data,~ReaderEvidence: @-reader: Reader -> Data,~inner: Context -> U32 -> U32 -> Inner,
  ~read: Reader -> Inner -> U32 -> Reader & Element,~place: Inner -> U32 -> U32,
  ~read_preserves: @-reader: Reader -> @context: Inner -> @index: U32 -> ReaderEvidence(reader) ->
    TraversalWitness.Sample<Reader,Element,reader,read(reader,context,index)>,
  blocks: Nat,size: U32,context: Context,-input: Reader,-output: Array<Element>,input_evidence: ReaderEvidence(input),+depth: Nat,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  TraversalWitness.ResultEvidence<Reader,Element,input,depth,Blocks.run(~Reader,~Element,~Context,~Inner,~inner,~read,~place,blocks,size,context,input,output)>:
  TraversalCertificate.completed(~Reader,~Element,~Blocks.Blocks<Context>,~ReaderEvidence,
    ~Loop.run(~Traversal.State<&1,&1,Reader,Array<Element>,Blocks.Blocks<Context>>,~Blocks.block_step(~Reader,~Element,~Context,~Inner,~inner,~read,~place)),
    ~block_loop(~Reader,~Element,~Context,~Inner,~ReaderEvidence,~inner,~read,~place,~read_preserves),
    blocks,0,Blocks.Blocks{0,size,context},input,output,input_evidence,depth,equation)
