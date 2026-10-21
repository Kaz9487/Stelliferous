# A convolution parameter (weights or bias) as the list the specification
# reads: every parameter sequence is an observation of the supplied physical
# owner. No constructor identity or equality between unused padding cells is
# needed.
import Base
import ../specification.bend as ConvSpec
import ../../lib/storage_buffer.bend as Storage
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/tensor_contract.bend as Contract
import ../../lib/proofs/nat_order.bend as Order
import ./input_storage.bend as Inputs

def sequence(content: Buffer.Content<F32>) -> List<&2,F32>:
  Buffer.Content{tree,length} = content
  Tree.read_sequence(~F32,length,0,tree)

# Storage that holds a reader's values reads back as the reader's values.
def listed_sequence(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+content: Buffer.Content<F32>,
  +values: Contract.Values(~F32,~Logical,~original,logical,content)) ->
  {sequence(content) == Contract.listed(~F32,~Logical,~original,Buffer.content_length(~F32,content),0n,logical) : List<&2,F32>}:
  Buffer.Content{+tree,+length} = content
  Contract.read_listed(~F32,~Logical,~original,logical,tree,length,values,length,0n,Order.reflexive(length))

def sequence_evidence(+content: Buffer.Content<F32>,ready: Buffer.ContentReady<F32,content>) ->
  StorageLaws.SequenceEvidence<F32,ConvSpec.length(sequence(content)),sequence(content)>:
  Buffer.Content{+tree,+length} = content
  Buffer.ContentReady{depth,balanced,depth_ok,room} = ready
  Inputs.prefix_evidence(length,tree,room)

def sequence_tree(+content: Buffer.Content<F32>,+ready: Buffer.ContentReady<F32,content>) ->
  {StorageLaws.sequence_storage(F32,ConvSpec.length(sequence(content)),sequence(content),sequence_evidence(content,ready)) == Buffer.content_tree(~F32,content) : Tree.Tree<F32>}:
  Buffer.Content{+tree,+length} = content
  Buffer.ContentReady{depth,balanced,depth_ok,room} = ready
  {==}

def sequence_length(+content: Buffer.Content<F32>) ->
  {ConvSpec.length(sequence(content)) == Buffer.content_length(~F32,content) : Nat}:
  Buffer.Content{tree,length} = content
  Inputs.observed_length(length,tree)

type Parameter<-buffer: Storage.Buffer<F32>> is Data:
  Parameter{model: Buffer.Model<F32>,equation: {Buffer.pack(~F32,model) == buffer : Storage.Buffer<F32>},ready: Buffer.Sized<F32,model>,
    values: List<&2,F32>,length_exact: {ConvSpec.length(values) == Buffer.length(~F32,model) : Nat},
    evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(values),values>,
    tree_exact: {Buffer.tree(~F32,model) == StorageLaws.sequence_storage(F32,ConvSpec.length(values),values,evidence) : Tree.Tree<F32>}}

def model_tree(+model: Buffer.Model<F32>) -> {Buffer.content_tree(~F32,Buffer.content(~F32,model)) == Buffer.tree(~F32,model) : Tree.Tree<F32>}:
  Buffer.Model{tree,length,certificate} = model
  {==}

def observed_parameter(-buffer: Storage.Buffer<F32>,+ready: Buffer.ContentReady<F32,Buffer.contents(~F32,buffer)>,observation: Buffer.Observation<F32,buffer>) -> Parameter<buffer>:
  Buffer.Observation{+model,+equation} = observation
  +model_ready = Buffer.ready_content(~F32,model,
    %Buffer.packed_contents(~F32,model) : Buffer.ContentReady<F32,_>
    %Equal.sym(Storage.Buffer<F32>,Buffer.pack(~F32,model),buffer,equation) : Buffer.ContentReady<F32,Buffer.contents(~F32,_)>
    ready)
  +content_ready = Buffer.content_ready(~F32,model,model_ready)
  Parameter{model,equation,model_ready,sequence(Buffer.content(~F32,model)),
    Equal.trans(Nat,ConvSpec.length(sequence(Buffer.content(~F32,model))),Buffer.content_length(~F32,Buffer.content(~F32,model)),Buffer.length(~F32,model),sequence_length(Buffer.content(~F32,model)),Buffer.model_length(model)),
    sequence_evidence(Buffer.content(~F32,model),content_ready),
    Equal.sym(Tree.Tree<F32>,StorageLaws.sequence_storage(F32,ConvSpec.length(sequence(Buffer.content(~F32,model))),sequence(Buffer.content(~F32,model)),sequence_evidence(Buffer.content(~F32,model),content_ready)),Buffer.tree(~F32,model),
      Equal.trans(Tree.Tree<F32>,StorageLaws.sequence_storage(F32,ConvSpec.length(sequence(Buffer.content(~F32,model))),sequence(Buffer.content(~F32,model)),sequence_evidence(Buffer.content(~F32,model),content_ready)),
        Buffer.content_tree(~F32,Buffer.content(~F32,model)),Buffer.tree(~F32,model),sequence_tree(Buffer.content(~F32,model),content_ready),model_tree(model)))}

def parameter(buffer: Storage.Buffer<F32>,ready: Buffer.ContentReady<F32,Buffer.contents(~F32,buffer)>) -> Parameter<buffer>:
  observed_parameter(buffer,ready,Buffer.observe(~F32,buffer))

def parameter_values(-buffer: Storage.Buffer<F32>,parameter: Parameter<buffer>) -> List<&2,F32>:
  Parameter{model,equation,ready,values,length_exact,evidence,tree_exact} = parameter
  values

def parameter_model(-buffer: Storage.Buffer<F32>,parameter: Parameter<buffer>) -> Buffer.Model<F32>:
  Parameter{model,equation,ready,values,length_exact,evidence,tree_exact} = parameter
  model

def parameter_evidence(-buffer: Storage.Buffer<F32>,+parameter: Parameter<buffer>) ->
  StorageLaws.SequenceEvidence<F32,ConvSpec.length(parameter_values(buffer,parameter)),parameter_values(buffer,parameter)>:
  Parameter{model,equation,ready,values,length_exact,evidence,tree_exact} = parameter
  evidence

def parameter_content(-buffer: Storage.Buffer<F32>,+parameter: Parameter<buffer>) ->
  {Buffer.content(~F32,parameter_model(buffer,parameter)) ==
    Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(parameter_values(buffer,parameter)),parameter_values(buffer,parameter),parameter_evidence(buffer,parameter)),ConvSpec.length(parameter_values(buffer,parameter))} : Buffer.Content<F32>}:
  Parameter{+model,equation,ready,+values,+length_exact,+evidence,+tree_exact} = parameter
  Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,model),Buffer.Content{Buffer.tree(~F32,model),Buffer.length(~F32,model)},
    Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(values),values,evidence),ConvSpec.length(values)},Buffer.content_parts(model),
    %Equal.sym(Tree.Tree<F32>,Buffer.tree(~F32,model),StorageLaws.sequence_storage(F32,ConvSpec.length(values),values,evidence),tree_exact) :
      {Buffer.Content{_,Buffer.length(~F32,model)} == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(values),values,evidence),ConvSpec.length(values)} : Buffer.Content<F32>}
    %length_exact : {Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(values),values,evidence),_} == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(values),values,evidence),ConvSpec.length(values)} : Buffer.Content<F32>}
    {==})
