# Certified copies have the raw traversal's exact write program.
# The Data observation retains tree contents and lengths, erasing certificates.
import Base
import ../storage_buffer.bend as Storage
import ../storage_copy.bend as Copy
import ../tensor_operations.bend as Operations
import ../tensor_access.bend as Access
import ../tensor_view.bend as Views
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./storage_certificate.bend as Certificate
import ./traversal_certificate.bend as TraversalCertificate
import ./tensor_access_certificate.bend as AccessCertificate
import ./contiguous_copy.bend as Contiguous
import ./write_program.bend as Writes
import ./tensor_access_refinement.bend as AccessLaws
import ./tensor_copy_values.bend as Values
import ./traversal_array_refinement.bend as ArrayLaws
import ./traversal_witness.bend as TraversalWitness

def returned_certificate(~Element: Data,-input: Array<Element>,-returned: Array<Element>,certificate: Certificate.Certificate<Element,input>,
  same: {returned == input : Array<Element>}) -> Certificate.Certificate<Element,returned>:
  %Equal.sym(Array<Element>,returned,input,same) : Certificate.Certificate<Element,_>
  certificate

def source_certificate(~Element: Data,+length: Nat,-input: Array<Element>,-returned: Array<Element>,+same: {returned == input : Array<Element>}) ->
  @certificate: Certificate.Certificate<Element,input> ->
  {Storage.Buffer{returned,length,returned_certificate(~Element,input,returned,certificate,same)} == Storage.Buffer{input,length,certificate} : Storage.Buffer<Element>}:
  %proof@same : @certificate: Certificate.Certificate<Element,_> ->
    {Storage.Buffer{returned,length,returned_certificate(~Element,_,returned,certificate,proof)} == Storage.Buffer{_,length,certificate} : Storage.Buffer<Element>}
  certificate => {==}

def returned_source(~Element: Data,+input_length: Nat,+output_length: Nat,-input: Array<Element>,result: Array<Element> & Array<Element>,
  +certificate: Certificate.Certificate<Element,input>,output_certificate: Certificate.Certificate<Element,TraversalWitness.second(~Array<Element>,~Array<Element>,result)>,
  +same: {TraversalWitness.first(~Array<Element>,~Array<Element>,result) == input : Array<Element>}) ->
  {Pair.fst(Storage.Buffer<Element>,Storage.Buffer<Element>,Operations.pair_buffers(~Element,input_length,output_length,result,
    TraversalCertificate.Certificates{returned_certificate(~Element,input,TraversalWitness.first(~Array<Element>,~Array<Element>,result),certificate,same),output_certificate})) ==
    Storage.Buffer{input,input_length,certificate} : Storage.Buffer<Element>}:
  (returned,output) = result
  source_certificate(~Element,input_length,input,returned,same,certificate)

# A view copy returns its source buffer, including its certificate.
def source(~Element: Data,+fast: Bool,+count: U32,+view: Views.View,+destination: U32,input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) ->
  {Pair.fst(Storage.Buffer<Element>,Storage.Buffer<Element>,Copy.view(~Element,fast,count,view,destination,input,output)) == input : Storage.Buffer<Element>}:
  Storage.Buffer{input,+input_length,Certificate.Certificate{+input_depth,+input_equation}} = input
  Storage.Buffer{output,+output_length,Certificate.Certificate{+depth,+equation}} = output
  +call = {AccessCertificate.CopyCall{fast,U32.to_nat(count),view,destination} : AccessCertificate.CopyCall}
  +certificate = {Certificate.Certificate{input_depth,input_equation} : Certificate.Certificate<Element,input>}
  returned_source(~Element,input_length,output_length,input,Access.copy_run(~Element,fast,U32.to_nat(count),view,destination,input,output),certificate,
    Certificate.Certificate{depth,TraversalCertificate.operation_equation(~Array<Element>,~Element,~AccessCertificate.CopyCall,~Certificate.family(~Element),
      ~AccessCertificate.copy_operation(~Element),~AccessCertificate.copied(~Element),call,input,output,certificate,depth,equation)},
    TraversalWitness.source_unchanged(~Array<Element>,~Element,input,depth,AccessCertificate.copy_operation(~Element,call,input,output),
      AccessCertificate.copied(~Element,depth,call,input,output,certificate,equation)))

type Owners<-Element: Data> is Data:
  Owners{source: Tree.Tree<Element>,source_length: Nat,output: Tree.Tree<Element>,output_length: Nat}

type Call is Data:
  Call{count: U32,source: U32,destination: U32,source_length: Nat,output_length: Nat}

def call_count(call: Call) -> U32:
  Call{count,source,destination,input_length,output_length} = call
  count

def call_source(call: Call) -> U32:
  Call{count,source,destination,input_length,output_length} = call
  source

def call_destination(call: Call) -> U32:
  Call{count,source,destination,input_length,output_length} = call
  destination

def call_input_length(call: Call) -> Nat:
  Call{count,source,destination,input_length,output_length} = call
  input_length

def call_output_length(call: Call) -> Nat:
  Call{count,source,destination,input_length,output_length} = call
  output_length

def observe(~Element: Data,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) -> Owners<Element>:
  (Storage.Buffer{input,input_length,input_certificate},Storage.Buffer{output,output_length,output_certificate}) = owners
  Owners{Tree.reflect(Element,input),input_length,Tree.reflect(Element,output),output_length}

def values(~Element: Data,input_length: Nat,output_length: Nat,result: Array<Element> & Array<Element>) -> Owners<Element>:
  (input,output) = result
  Owners{Tree.reflect(Element,input),input_length,Tree.reflect(Element,output),output_length}

def finished(~Element: Data,+input_length: Nat,+output_length: Nat,result: Array<Element> & Array<Element>,
  certificates: TraversalCertificate.Certificates<Array<Element>,Element,Certificate.family(~Element),result>) ->
  {observe(~Element,Operations.pair_buffers(~Element,input_length,output_length,result,certificates)) == values(~Element,input_length,output_length,result) : Owners<Element>}:
  (input,output) = result
  TraversalCertificate.Certificates{input_certificate,output_certificate} = certificates
  {==}

def represented_values(~Element: Data,+input: Tree.Tree<Element>,+input_length: Nat,+output: Tree.Tree<Element>,+output_length: Nat) ->
  {values(~Element,input_length,output_length,(Tree.pack(Element,input),Tree.pack(Element,output))) == Owners{input,input_length,output,output_length} : Owners<Element>}:
  %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,input)),input,StorageLaws.reflect_packed(Element,input)) :
    {Owners{_,input_length,Tree.reflect(Element,Tree.pack(Element,output)),output_length} == Owners{input,input_length,output,output_length} : Owners<Element>}
  %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,output)),output,StorageLaws.reflect_packed(Element,output)) :
    {Owners{input,input_length,_,output_length} == Owners{input,input_length,output,output_length} : Owners<Element>}
  {==}

def model(~Element: Data,count: U32,source: U32,destination: U32,owners: Owners<Element>) -> Owners<Element>:
  Owners{+input,input_length,output,output_length} = owners
  Owners{input,input_length,Writes.pure(Element,Contiguous.writes(~Element,U32.to_nat(count),input,0,source,destination),output),output_length}

def packed(~Element: Data,+call: Call,+axes: List<&2,Views.Axis>,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,
  +input_certificate: Certificate.Certificate<Element,Tree.pack(Element,input)>,+output_certificate: Certificate.Certificate<Element,Tree.pack(Element,output)>) ->
  {observe(~Element,Copy.view(~Element,True{},call_count(call),Views.View{axes,call_source(call)},call_destination(call),
      Storage.Buffer{Tree.pack(Element,input),call_input_length(call),input_certificate},Storage.Buffer{Tree.pack(Element,output),call_output_length(call),output_certificate})) ==
    model(~Element,call_count(call),call_source(call),call_destination(call),Owners{input,call_input_length(call),output,call_output_length(call)}) : Owners<Element>}:
  Call{+count,+source,+destination,+input_length,+output_length} = call
  +view = {Views.View{axes,source} : Views.View}
  %Equal.sym(Owners<Element>,observe(~Element,Operations.pair_buffers(~Element,input_length,output_length,
      Access.copy_run(~Element,True{},U32.to_nat(count),view,destination,Tree.pack(Element,input),Tree.pack(Element,output)),
      AccessCertificate.copy(~Element,True{},U32.to_nat(count),view,destination,Tree.pack(Element,input),Tree.pack(Element,output),input_certificate,output_certificate))),
    values(~Element,input_length,output_length,Access.copy_run(~Element,True{},U32.to_nat(count),view,destination,Tree.pack(Element,input),Tree.pack(Element,output))),
    finished(~Element,input_length,output_length,Access.copy_run(~Element,True{},U32.to_nat(count),view,destination,Tree.pack(Element,input),Tree.pack(Element,output)),
      AccessCertificate.copy(~Element,True{},U32.to_nat(count),view,destination,Tree.pack(Element,input),Tree.pack(Element,output),input_certificate,output_certificate))) :
    {_ == model(~Element,count,source,destination,Owners{input,input_length,output,output_length}) : Owners<Element>}
  %Equal.sym(Array<Element> & Array<Element>,Access.copy_run(~Element,True{},U32.to_nat(count),view,destination,Tree.pack(Element,input),Tree.pack(Element,output)),
    (Tree.pack(Element,input),Tree.pack(Element,Writes.pure(Element,Contiguous.writes(~Element,U32.to_nat(count),input,0,source,destination),output))),Contiguous.copy(~Element,U32.to_nat(count),input,output,source,destination)) :
    {values(~Element,input_length,output_length,_) == model(~Element,count,source,destination,Owners{input,input_length,output,output_length}) : Owners<Element>}
  represented_values(~Element,input,input_length,Writes.pure(Element,Contiguous.writes(~Element,U32.to_nat(count),input,0,source,destination),output),output_length)

# A copy through any view stores the copy writes of the view and keeps its
# source.
def model_values(~Element: Data,+input_length: Nat,+output_length: Nat,+result: AccessLaws.ModelResult<Tree.Tree<Element>,Element>) ->
  {values(~Element,input_length,output_length,AccessLaws.pack_result(~Array<Element>,~Tree.Tree<Element>,~Element,~ArrayLaws.pack_tree(~Element),result)) ==
    Owners{Values.reader_tree(~Element,result),input_length,Values.output_tree(~Element,result),output_length} : Owners<Element>}:
  AccessLaws.ModelResult{+reader,+output} = result
  represented_values(~Element,reader,input_length,output,output_length)

def view_packed(~Element: Data,+fast: Bool,+count: U32,+source: Views.View,+destination: U32,+input: Tree.Tree<Element>,+input_length: Nat,
  +output: Tree.Tree<Element>,+output_length: Nat,
  +input_certificate: Certificate.Certificate<Element,Tree.pack(Element,input)>,+output_certificate: Certificate.Certificate<Element,Tree.pack(Element,output)>) ->
  {observe(~Element,Copy.view(~Element,fast,count,source,destination,
      Storage.Buffer{Tree.pack(Element,input),input_length,input_certificate},Storage.Buffer{Tree.pack(Element,output),output_length,output_certificate})) ==
    Owners{input,input_length,Writes.pure(Element,Values.copy_writes(~Element,fast,U32.to_nat(count),source,destination,input),output),output_length} : Owners<Element>}:
  +steps = U32.to_nat(count)
  +model = AccessLaws.copy_model(~Element,fast,steps,source,destination,input,output)
  +written = Writes.pure(Element,Values.copy_writes(~Element,fast,steps,source,destination,input),output)
  %Equal.sym(Owners<Element>,observe(~Element,Operations.pair_buffers(~Element,input_length,output_length,Access.copy_run(~Element,fast,steps,source,destination,Tree.pack(Element,input),Tree.pack(Element,output)),
      AccessCertificate.copy(~Element,fast,steps,source,destination,Tree.pack(Element,input),Tree.pack(Element,output),input_certificate,output_certificate))),
    values(~Element,input_length,output_length,Access.copy_run(~Element,fast,steps,source,destination,Tree.pack(Element,input),Tree.pack(Element,output))),
    finished(~Element,input_length,output_length,Access.copy_run(~Element,fast,steps,source,destination,Tree.pack(Element,input),Tree.pack(Element,output)),
      AccessCertificate.copy(~Element,fast,steps,source,destination,Tree.pack(Element,input),Tree.pack(Element,output),input_certificate,output_certificate))) :
    {_ == Owners{input,input_length,written,output_length} : Owners<Element>}
  %Equal.sym(Array<Element> & Array<Element>,Access.copy_run(~Element,fast,steps,source,destination,Tree.pack(Element,input),Tree.pack(Element,output)),AccessLaws.pack_result(~Array<Element>,~Tree.Tree<Element>,~Element,~ArrayLaws.pack_tree(~Element),model),
    AccessLaws.copy(~Element,fast,steps,source,destination,input,output)) :
    {values(~Element,input_length,output_length,_) == Owners{input,input_length,written,output_length} : Owners<Element>}
  %Equal.sym(Owners<Element>,values(~Element,input_length,output_length,AccessLaws.pack_result(~Array<Element>,~Tree.Tree<Element>,~Element,~ArrayLaws.pack_tree(~Element),model)),
    Owners{Values.reader_tree(~Element,model),input_length,Values.output_tree(~Element,model),output_length},model_values(~Element,input_length,output_length,model)) :
    {_ == Owners{input,input_length,written,output_length} : Owners<Element>}
  %Equal.sym(Tree.Tree<Element>,Values.reader_tree(~Element,model),input,Values.copy_reader(~Element,fast,steps,source,destination,input,output)) :
    {Owners{_,input_length,Values.output_tree(~Element,model),output_length} == Owners{input,input_length,written,output_length} : Owners<Element>}
  %Equal.sym(Tree.Tree<Element>,Values.output_tree(~Element,model),written,Values.copy_program(~Element,fast,steps,source,destination,input,output)) :
    {Owners{input,input_length,_,output_length} == Owners{input,input_length,written,output_length} : Owners<Element>}
  {==}
