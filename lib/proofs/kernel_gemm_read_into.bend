# The buffers a read into an output gives back: the source as it was and the
# output of the model run of its reads.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_gemm_buffer.bend as GemmBuffer
import ../storage_buffer.bend as Storage
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./storage_certificate.bend as Certificate
import ./traversal_certificate.bend as TraversalCertificate
import ./traversal_witness.bend as TraversalWitness
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./traversal_array_refinement.bend as ArrayLaws
import ./traversal_update_readback.bend as Updates

def buffers_contents(~R: Data,~E: Data,pair: Storage.Buffer<R> & Storage.Buffer<E>) -> Buffer.Content<R> & Buffer.Content<E>:
  (left,right) = pair
  (Buffer.contents(~R,left),Buffer.contents(~E,right))

def arrays_contents(~R: Data,~E: Data,+length: Nat,+output_length: Nat,result: Array<R> & Array<E>) -> Buffer.Content<R> & Buffer.Content<E>:
  (reader,output) = result
  (Buffer.Content{Tree.reflect(R,reader),length},Buffer.Content{Tree.reflect(E,output),output_length})

# Two buffers made of two arrays hold the contents of the arrays.
def buffers_law(~R: Data,~E: Data,+length: Nat,+output_length: Nat,result: Array<R> & Array<E>,
  certificates: TraversalCertificate.Certificates<Array<R>,E,Certificate.family(~R),result>,-target: Buffer.Content<R> & Buffer.Content<E>,
  arrays: {arrays_contents(~R,~E,length,output_length,result) == target : Buffer.Content<R> & Buffer.Content<E>}) ->
  {buffers_contents(~R,~E,GemmBuffer.read_buffers(~R,~E,length,output_length,result,certificates)) == target : Buffer.Content<R> & Buffer.Content<E>}:
  (reader,output) = result
  TraversalCertificate.Certificates{certificate,output_certificate} = certificates
  arrays

# The arrays of a model state are the trees of the state.
def represented_contents(~R: Data,~E: Data,~Context: Data,+length: Nat,+output_length: Nat,+state: Traversal.State<&2,&2,Tree.Tree<R>,Tree.Tree<E>,Context>) ->
  {arrays_contents(~R,~E,length,output_length,Traversal.finish(~Array<R>,~E,~Context,ArrayLaws.represent(~Array<R>,~Tree.Tree<R>,~E,~Context,~ArrayLaws.pack_tree(~R),state))) ==
    (Buffer.Content{ArrayLaws.model_reader(~Tree.Tree<R>,~E,~Context,state),length},Buffer.Content{Updates.state_tree(~Tree.Tree<R>,~E,~Context,state),output_length}) :
    Buffer.Content<R> & Buffer.Content<E>}:
  Traversal.State{+reader,+output,position,context} = state
  %Equal.sym(Tree.Tree<R>,Tree.reflect(R,Tree.pack(R,reader)),reader,StorageLaws.reflect_packed(R,reader)) :
    {(Buffer.Content{_,length},Buffer.Content{Tree.reflect(E,Tree.pack(E,output)),output_length}) == (Buffer.Content{reader,length},Buffer.Content{output,output_length}) :
      Buffer.Content<R> & Buffer.Content<E>}
  %Equal.sym(Tree.Tree<E>,Tree.reflect(E,Tree.pack(E,output)),output,StorageLaws.reflect_packed(E,output)) :
    {(Buffer.Content{reader,length},Buffer.Content{_,output_length}) == (Buffer.Content{reader,length},Buffer.Content{output,output_length}) : Buffer.Content<R> & Buffer.Content<E>}
  {==}

# The model run of count positions into an output tree, each combining the
# value at its destination with its sample.
def update_loop(~R: Data,~E: Data,~S: Data,~Context: Data,~sample: Tree.Tree<R> -> Context -> U32 -> S,~combine: E -> S -> E,~destination: Context -> U32 -> U32,
  count: Nat,context: Context,reader: Tree.Tree<R>,output: Tree.Tree<E>) -> Tree.Tree<E>:
  Updates.state_tree(~Tree.Tree<R>,~E,~Context,Loop.run(~Traversal.State<&2,&2,Tree.Tree<R>,Tree.Tree<E>,Context>,
    ~ArrayLaws.model_update_step(~Tree.Tree<R>,~E,~S,~Context,~sample,~combine,~destination),count,Traversal.State{reader,output,0,context}))

# The model run of the reads of count positions into an output tree.
def read_loop(~R: Data,~E: Data,~Context: Data,~sample: Tree.Tree<R> -> Context -> U32 -> E,~destination: Context -> U32 -> U32,count: Nat,
  context: Context,reader: Tree.Tree<R>,output: Tree.Tree<E>) -> Tree.Tree<E>:
  update_loop(~R,~E,~E,~Context,~sample,~Gemm.replaced(~E),~destination,count,context,reader,output)

# An update into an output gives back the source as it was and, in the
# output, what the model run of its positions leaves.
def update_model(~R: Data,~E: Data,~S: Data,~Context: Data,~read: Array<R> -> Context -> U32 -> Array<R> & S,
  ~read_preserves: @-reader: Array<R> -> @context: Context -> @index: U32 -> TraversalWitness.array_evidence(~R,reader) ->
    TraversalWitness.Sample<Array<R>,S,reader,read(reader,context,index)>,
  ~sample: Tree.Tree<R> -> Context -> U32 -> S,
  ~read_matches: @+reader: Tree.Tree<R> -> @+context: Context -> @+position: U32 ->
    {read(ArrayLaws.pack_tree(~R,reader),context,position) == (ArrayLaws.pack_tree(~R,reader),sample(reader,context,position)) : Array<R> & S},
  ~combine: E -> S -> E,~destination: Context -> U32 -> U32,+count: Nat,+context: Context,+source: Buffer.Model<R>,+output: Buffer.Model<E>) ->
  {buffers_contents(~R,~E,GemmBuffer.updated_into(~R,~E,~S,~Context,~read,~read_preserves,~combine,~destination,count,context,Buffer.pack(~R,source),Buffer.pack(~E,output))) ==
    (Buffer.content(~R,source),Buffer.Content{update_loop(~R,~E,~S,~Context,~sample,~combine,~destination,count,context,Buffer.tree(~R,source),Buffer.tree(~E,output)),
      Buffer.length(~E,output)}) : Buffer.Content<R> & Buffer.Content<E>}:
  Buffer.Model{+input,+input_length,input_certificate} = source
  Certificate.Certificate{+depth,+equation} = input_certificate
  Buffer.Model{+tree,+tree_length,tree_certificate} = output
  +run = {Traversal.Steps{count,0,context} : Traversal.Steps<Context>}
  +start = {Traversal.State{input,tree,0,context} : Traversal.State<&2,&2,Tree.Tree<R>,Tree.Tree<E>,Context>}
  +walked = Loop.run(~Traversal.State<&2,&2,Tree.Tree<R>,Tree.Tree<E>,Context>,~ArrayLaws.model_update_step(~Tree.Tree<R>,~E,~S,~Context,~sample,~combine,~destination),count,start)
  -target = {(Buffer.Content{input,input_length},Buffer.Content{Updates.state_tree(~Tree.Tree<R>,~E,~Context,walked),tree_length}) : Buffer.Content<R> & Buffer.Content<E>}
  buffers_law(~R,~E,input_length,tree_length,Traversal.updated(~Array<R>,~E,~S,~Context,~read,~combine,~destination,run,Tree.pack(R,input),Tree.pack(E,tree)),
    TraversalCertificate.updated_certificates(~Array<R>,~E,~S,~Context,~TraversalWitness.array_evidence(~R),~Certificate.family(~R),~read,~combine,~destination,
      ~read_preserves,~TraversalCertificate.array_observed(~R),run,Tree.pack(R,input),Tree.pack(E,tree),Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},
      Certificate.Certificate{depth,equation},tree_certificate),target,
    %Equal.sym(Traversal.State<&1,&1,Array<R>,Array<E>,Context>,
        Traversal.update(~Array<R>,~E,~S,~Context,~read,~combine,~destination,count,ArrayLaws.represent(~Array<R>,~Tree.Tree<R>,~E,~Context,~ArrayLaws.pack_tree(~R),start)),
        ArrayLaws.represent(~Array<R>,~Tree.Tree<R>,~E,~Context,~ArrayLaws.pack_tree(~R),walked),
      ArrayLaws.update_refines_model(~Array<R>,~Tree.Tree<R>,~E,~S,~Context,~ArrayLaws.pack_tree(~R),~read,~sample,~combine,~destination,~read_matches,count,start)) :
      {arrays_contents(~R,~E,input_length,tree_length,Traversal.finish(~Array<R>,~E,~Context,_)) == target : Buffer.Content<R> & Buffer.Content<E>}
    Equal.trans(Buffer.Content<R> & Buffer.Content<E>,
      arrays_contents(~R,~E,input_length,tree_length,Traversal.finish(~Array<R>,~E,~Context,ArrayLaws.represent(~Array<R>,~Tree.Tree<R>,~E,~Context,~ArrayLaws.pack_tree(~R),walked))),
      (Buffer.Content{ArrayLaws.model_reader(~Tree.Tree<R>,~E,~Context,walked),input_length},Buffer.Content{Updates.state_tree(~Tree.Tree<R>,~E,~Context,walked),tree_length}),target,
      represented_contents(~R,~E,~Context,input_length,tree_length,walked),
      %Equal.sym(Tree.Tree<R>,ArrayLaws.model_reader(~Tree.Tree<R>,~E,~Context,walked),input,
          Updates.update_reader(~Tree.Tree<R>,~E,~S,~Context,~sample,~combine,~destination,count,0,input,context,tree)) :
        {(Buffer.Content{_,input_length},Buffer.Content{Updates.state_tree(~Tree.Tree<R>,~E,~Context,walked),tree_length}) == target : Buffer.Content<R> & Buffer.Content<E>}
      {==}))

# A read into an output gives back the source as it was and, in the output,
# what the model run of its reads leaves.
def read_model(~R: Data,~E: Data,~Context: Data,~read: Array<R> -> Context -> U32 -> Array<R> & E,
  ~read_preserves: @-reader: Array<R> -> @context: Context -> @index: U32 -> TraversalWitness.array_evidence(~R,reader) ->
    TraversalWitness.Sample<Array<R>,E,reader,read(reader,context,index)>,
  ~sample: Tree.Tree<R> -> Context -> U32 -> E,
  ~read_matches: @+reader: Tree.Tree<R> -> @+context: Context -> @+position: U32 ->
    {read(ArrayLaws.pack_tree(~R,reader),context,position) == (ArrayLaws.pack_tree(~R,reader),sample(reader,context,position)) : Array<R> & E},
  ~destination: Context -> U32 -> U32,+count: Nat,+context: Context,+source: Buffer.Model<R>,+output: Buffer.Model<E>) ->
  {buffers_contents(~R,~E,GemmBuffer.read_into(~R,~E,~Context,~read,~read_preserves,~destination,count,context,Buffer.pack(~R,source),Buffer.pack(~E,output))) ==
    (Buffer.content(~R,source),Buffer.Content{read_loop(~R,~E,~Context,~sample,~destination,count,context,Buffer.tree(~R,source),Buffer.tree(~E,output)),Buffer.length(~E,output)}) :
    Buffer.Content<R> & Buffer.Content<E>}:
  update_model(~R,~E,~E,~Context,~read,~read_preserves,~sample,~read_matches,~Gemm.replaced(~E),~destination,count,context,source,output)
