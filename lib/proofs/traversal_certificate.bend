# Flat certificates are updated once after a complete traversal. The witness
# traversal is evaluated only by the checker, beneath erased equality rewrites.
import Base
import ../traversal_array.bend as Traversal
import ./storage_tree_model.bend as Tree
import ./storage_certificate.bend as Certificate
import ./traversal_witness.bend as TraversalWitness

type Certificates<-Reader: Type,-Element: Data,-ReaderCertificate: @-reader: Reader -> Data,
  -result: Reader & Array<Element>> is Data:
  Certificates{source: ReaderCertificate(TraversalWitness.first(~Reader,~Array<Element>,result)),
    output: Certificate.Certificate<Element,TraversalWitness.second(~Reader,~Array<Element>,result)>}

def completed(~Reader: Type,~Element: Data,~Context: Data,~ReaderEvidence: @-reader: Reader -> Data,
  ~run: Nat -> Traversal.State<&1,&1,Reader,Array<Element>,Context> -> Traversal.State<&1,&1,Reader,Array<Element>,Context>,
  ~preserves: @+depth: Nat -> @count: Nat -> @-reader: Reader -> @-state: Traversal.State<&1,&1,Reader,Array<Element>,Context> ->
    @+reader_evidence: ReaderEvidence(reader) -> TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,state> ->
    TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,run(count,state)>,
  count: Nat,position: U32,context: Context,-reader: Reader,-output: Array<Element>,reader_evidence: ReaderEvidence(reader),+depth: Nat,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  TraversalWitness.ResultEvidence<Reader,Element,reader,depth,
    Traversal.finish(~Reader,~Element,~Context,run(count,Traversal.State{reader,output,position,context}))>:
  TraversalWitness.finish(~Reader,~Element,~Context,reader,depth,run(count,Traversal.State{reader,output,position,context}),
    preserves(depth,count,reader,Traversal.State{reader,output,position,context},reader_evidence,
      TraversalWitness.StateEvidence{output,position,context,{==},Certificate.witness_at(Element,depth,output,equation)}))

def map_equation(~Element: Data,~Context: Data,~map: Context -> Element -> Element,count: Nat,position: U32,context: Context,
  -output: Array<Element>,+depth: Nat,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  {Traversal.map_in_place(~Element,~Context,~map,count,position,context,output) ==
    Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,Traversal.map_in_place(~Element,~Context,~map,count,position,context,output)))) : Array<Element>}:
  %Certificate.restored(Element,depth,Traversal.map_in_place(~Element,~Context,~map,count,position,context,output),
    TraversalWitness.map_in_place(~Element,~Context,~map,depth,count,position,context,output,Certificate.witness_at(Element,depth,output,equation))) :
    {_ == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,Traversal.map_in_place(~Element,~Context,~map,count,position,context,output)))) : Array<Element>}
  {==}

def map_in_place(~Element: Data,~Context: Data,~map: Context -> Element -> Element,count: Nat,position: U32,context: Context,
  -output: Array<Element>,certificate: Certificate.Certificate<Element,output>) ->
  Certificate.Certificate<Element,Traversal.map_in_place(~Element,~Context,~map,count,position,context,output)>:
  Certificate.Certificate{+depth,equation} = certificate
  Certificate.Certificate{depth,map_equation(~Element,~Context,~map,count,position,context,output,depth,equation)}

# Metadata preparation belongs inside the erased preservation proof. Passing a
# prepared cursor to a live certificate helper would retain or rebuild it.
def operation_source(~Reader: Type,~Element: Data,~Metadata: Data,~ReaderCertificate: @-reader: Reader -> Data,
  ~operation: Metadata -> Reader -> Array<Element> -> Reader & Array<Element>,
  ~preserves: @+depth: Nat -> @metadata: Metadata -> @-reader: Reader -> @-output: Array<Element> -> ReaderCertificate(reader) ->
    {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>} ->
    TraversalWitness.ResultEvidence<Reader,Element,reader,depth,operation(metadata,reader,output)>,
  metadata: Metadata,-reader: Reader,-output: Array<Element>,proof_certificate: ReaderCertificate(reader),reader_certificate: ReaderCertificate(reader),depth: Nat,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  ReaderCertificate(TraversalWitness.first(~Reader,~Array<Element>,operation(metadata,reader,output))):
  %Equal.sym(Reader,TraversalWitness.first(~Reader,~Array<Element>,operation(metadata,reader,output)),reader,
    TraversalWitness.source_unchanged(~Reader,~Element,reader,depth,operation(metadata,reader,output),
      preserves(depth,metadata,reader,output,proof_certificate,equation))) : ReaderCertificate(_)
  reader_certificate

def operation_equation(~Reader: Type,~Element: Data,~Metadata: Data,~ReaderCertificate: @-reader: Reader -> Data,
  ~operation: Metadata -> Reader -> Array<Element> -> Reader & Array<Element>,
  ~preserves: @+depth: Nat -> @metadata: Metadata -> @-reader: Reader -> @-output: Array<Element> -> ReaderCertificate(reader) ->
    {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>} ->
    TraversalWitness.ResultEvidence<Reader,Element,reader,depth,operation(metadata,reader,output)>,
  metadata: Metadata,-reader: Reader,-output: Array<Element>,reader_certificate: ReaderCertificate(reader),+depth: Nat,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  {TraversalWitness.second(~Reader,~Array<Element>,operation(metadata,reader,output)) ==
    Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,TraversalWitness.second(~Reader,~Array<Element>,operation(metadata,reader,output))))) : Array<Element>}:
  %Certificate.restored(Element,depth,TraversalWitness.second(~Reader,~Array<Element>,operation(metadata,reader,output)),
    TraversalWitness.output_witness(~Reader,~Element,reader,depth,operation(metadata,reader,output),
      preserves(depth,metadata,reader,output,reader_certificate,equation))) :
    {_ == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,TraversalWitness.second(~Reader,~Array<Element>,operation(metadata,reader,output))))) : Array<Element>}
  {==}

def operation(~Reader: Type,~Element: Data,~Metadata: Data,~ReaderCertificate: @-reader: Reader -> Data,
  ~operation: Metadata -> Reader -> Array<Element> -> Reader & Array<Element>,
  ~preserves: @+depth: Nat -> @metadata: Metadata -> @-reader: Reader -> @-output: Array<Element> -> ReaderCertificate(reader) ->
    {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>} ->
    TraversalWitness.ResultEvidence<Reader,Element,reader,depth,operation(metadata,reader,output)>,
  +metadata: Metadata,-reader: Reader,-output: Array<Element>,source_certificate: ReaderCertificate(reader),equation_certificate: ReaderCertificate(reader),
  reader_certificate: ReaderCertificate(reader),output_certificate: Certificate.Certificate<Element,output>) ->
  Certificates<Reader,Element,ReaderCertificate,operation(metadata,reader,output)>:
  Certificate.Certificate{+depth,+equation} = output_certificate
  Certificates{operation_source(~Reader,~Element,~Metadata,~ReaderCertificate,~operation,~preserves,metadata,reader,output,source_certificate,reader_certificate,depth,equation),
    Certificate.Certificate{depth,operation_equation(~Reader,~Element,~Metadata,~ReaderCertificate,~operation,~preserves,metadata,reader,output,equation_certificate,depth,equation)}}

def array_observed(~Element: Data,-storage: Array<Element>,certificate: Certificate.Certificate<Element,storage>) -> TraversalWitness.array_evidence(~Element,storage):
  Certificate.witness(Element,storage,certificate)

# A run of updates hands its reader back and leaves its output balanced.
# observe gives the evidence the reads need from a certificate of the reader.
def update_preserves(~Reader: Type,~Element: Data,~Value: Data,~Context: Data,~ReaderEvidence: @-reader: Reader -> Data,~ReaderCertificate: @-reader: Reader -> Data,
  ~read: Reader -> Context -> U32 -> Reader & Value,~combine: Element -> Value -> Element,~destination: Context -> U32 -> U32,
  ~read_preserves: @-reader: Reader -> @context: Context -> @index: U32 -> ReaderEvidence(reader) -> TraversalWitness.Sample<Reader,Value,reader,read(reader,context,index)>,
  ~observe: @-reader: Reader -> ReaderCertificate(reader) -> ReaderEvidence(reader),
  +depth: Nat,run: Traversal.Steps<Context>,-reader: Reader,-output: Array<Element>,certificate: ReaderCertificate(reader),
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  TraversalWitness.ResultEvidence<Reader,Element,reader,depth,Traversal.updated(~Reader,~Element,~Value,~Context,~read,~combine,~destination,run,reader,output)>:
  Traversal.Steps{count,+position,+context} = run
  completed(~Reader,~Element,~Context,~ReaderEvidence,~Traversal.update(~Reader,~Element,~Value,~Context,~read,~combine,~destination),
    ~TraversalWitness.update(~Reader,~Element,~Value,~Context,~ReaderEvidence,~read,~combine,~destination,~read_preserves),
    count,position,context,reader,output,observe(reader,certificate),depth,equation)

# The certificates of the reader and the output of a run of updates.
def updated_certificates(~Reader: Type,~Element: Data,~Value: Data,~Context: Data,~ReaderEvidence: @-reader: Reader -> Data,~ReaderCertificate: @-reader: Reader -> Data,
  ~read: Reader -> Context -> U32 -> Reader & Value,~combine: Element -> Value -> Element,~destination: Context -> U32 -> U32,
  ~read_preserves: @-reader: Reader -> @context: Context -> @index: U32 -> ReaderEvidence(reader) -> TraversalWitness.Sample<Reader,Value,reader,read(reader,context,index)>,
  ~observe: @-reader: Reader -> ReaderCertificate(reader) -> ReaderEvidence(reader),
  +run: Traversal.Steps<Context>,-reader: Reader,-output: Array<Element>,source_certificate: ReaderCertificate(reader),equation_certificate: ReaderCertificate(reader),
  reader_certificate: ReaderCertificate(reader),output_certificate: Certificate.Certificate<Element,output>) ->
  Certificates<Reader,Element,ReaderCertificate,Traversal.updated(~Reader,~Element,~Value,~Context,~read,~combine,~destination,run,reader,output)>:
  operation(~Reader,~Element,~Traversal.Steps<Context>,~ReaderCertificate,~Traversal.updated(~Reader,~Element,~Value,~Context,~read,~combine,~destination),
    ~update_preserves(~Reader,~Element,~Value,~Context,~ReaderEvidence,~ReaderCertificate,~read,~combine,~destination,~read_preserves,~observe),
    run,reader,output,source_certificate,equation_certificate,reader_certificate,output_certificate)
