# Public-owner observations erase certificate representation while retaining
# storage and logical lengths. Kernel equations use the independent tree model.
import Base
import ./storage_refinement.bend as StorageLaws
import ../storage_buffer.bend as Storage
import ../tensor_operations.bend as Operations
import ../tensor_view.bend as Views
import ../tensor_shape.bend as Shapes
import ../tensor_access.bend as Access
import ../traversal_array.bend as Traversal
import ./storage_tree_model.bend as Tree
import ./storage_certificate.bend as Certificate
import ./traversal_certificate.bend as TraversalCertificate
import ./tensor_access_certificate.bend as AccessCertificate
import ./tensor_access_refinement.bend as AccessLaws
import ./traversal_array_refinement.bend as ArrayLaws
import ./tensor_operation_bounds.bend as OperationBounds
import ./storage_buffer_model.bend as Buffer
import ./storage_clone_refinement.bend as CloneLaws
import ./tensor_contiguous_bounds.bend as ContiguousBounds
import ./traversal_witness.bend as TraversalWitness
import ./ordered_fold_refinement.bend as Fold
import ../traversal_loop.bend as Loop

def Kernel(~Element: Data) -> Type:
  @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>

def KernelLaw(~Element: Data,~operation: Element -> Element -> Element,~kernel: Kernel(~Element)) -> Type:
  @left: Storage.Buffer<Element> -> @+left_view: Views.View -> @right: Storage.Buffer<Element> -> @+right_view: Views.View -> @+count: U32 ->
  @counted: {Views.numel(left_view) == Some{count} : Maybe<&2,U32>} ->
  @accepted: {OperationBounds.combine_guard(~Element,left,left_view,right,right_view) == True{} : Bool} ->
  {CloneLaws.contents(~Element,kernel(U32.to_nat(count),right_view,ContiguousBounds.offset(left_view),left,right)) ==
    CloneLaws.contents(~Element,Operations.combine_kernel(~Element,~operation,U32.to_nat(count),right_view,ContiguousBounds.offset(left_view),left,right)) : Buffer.Content<Element> & Buffer.Content<Element>}

def scalar_kernel(~Element: Data,~operation: Element -> Element -> Element,left: Storage.Buffer<Element>,+left_view: Views.View,right: Storage.Buffer<Element>,+right_view: Views.View,+count: U32,
  counted: {Views.numel(left_view) == Some{count} : Maybe<&2,U32>},accepted: {OperationBounds.combine_guard(~Element,left,left_view,right,right_view) == True{} : Bool}) ->
  {CloneLaws.contents(~Element,Operations.combine_kernel(~Element,~operation,U32.to_nat(count),right_view,ContiguousBounds.offset(left_view),left,right)) ==
    CloneLaws.contents(~Element,Operations.combine_kernel(~Element,~operation,U32.to_nat(count),right_view,ContiguousBounds.offset(left_view),left,right)) : Buffer.Content<Element> & Buffer.Content<Element>}:
  {==}

type Owner<kind,-Value: Kind(kind)> is Kind(kind):
  Owner{storage: Value,length: Nat}

def PairOwners(-Element: Data) -> Type:
  Owner<&1,Array<Element>> & Owner<&1,Array<Element>>

def observe_owner(~Element: Data,buffer: Storage.Buffer<Element>) -> Owner<&1,Array<Element>>:
  Storage.Buffer{storage,length,certificate} = buffer
  Owner{storage,length}

def observe_pair(~Element: Data,pair: Storage.Buffer<Element> & Storage.Buffer<Element>) -> PairOwners(Element):
  (input,output) = pair
  (observe_owner(~Element,input),observe_owner(~Element,output))

def owners_from_contents(~Element: Data,pair: Buffer.Content<Element> & Buffer.Content<Element>) -> PairOwners(Element):
  (Buffer.Content{left,left_length},Buffer.Content{right,right_length}) = pair
  (Owner{Tree.pack(Element,left),left_length},Owner{Tree.pack(Element,right),right_length})

def pair_reconstructed(~Element: Data,pair: Storage.Buffer<Element> & Storage.Buffer<Element>) ->
  {owners_from_contents(~Element,CloneLaws.contents(~Element,pair)) == observe_pair(~Element,pair) : PairOwners(Element)}:
  (Storage.Buffer{left,+left_length,left_certificate},Storage.Buffer{right,+right_length,right_certificate}) = pair
  %Equal.sym(Array<Element>,Tree.pack(Element,Tree.reflect(Element,left)),left,StorageLaws.pack_reflected(Element,left)) :
    {(Owner{_,left_length},Owner{Tree.pack(Element,Tree.reflect(Element,right)),right_length}) == (Owner{left,left_length},Owner{right,right_length}) : PairOwners(Element)}
  Equal.cong(Array<Element>,PairOwners(Element),value => (Owner{left,left_length},Owner{value,right_length}),
    Tree.pack(Element,Tree.reflect(Element,right)),right,StorageLaws.pack_reflected(Element,right))

def contents_congruent(~Element: Data,left: Storage.Buffer<Element> & Storage.Buffer<Element>,right: Storage.Buffer<Element> & Storage.Buffer<Element>,
  equation: {CloneLaws.contents(~Element,left) == CloneLaws.contents(~Element,right) : Buffer.Content<Element> & Buffer.Content<Element>}) ->
  {observe_pair(~Element,left) == observe_pair(~Element,right) : PairOwners(Element)}:
  Equal.trans(PairOwners(Element),observe_pair(~Element,left),owners_from_contents(~Element,CloneLaws.contents(~Element,left)),observe_pair(~Element,right),
    Equal.sym(PairOwners(Element),owners_from_contents(~Element,CloneLaws.contents(~Element,left)),observe_pair(~Element,left),pair_reconstructed(~Element,left)),
    Equal.trans(PairOwners(Element),owners_from_contents(~Element,CloneLaws.contents(~Element,left)),owners_from_contents(~Element,CloneLaws.contents(~Element,right)),observe_pair(~Element,right),
      Equal.cong(Buffer.Content<Element> & Buffer.Content<Element>,PairOwners(Element),contents => owners_from_contents(~Element,contents),CloneLaws.contents(~Element,left),CloneLaws.contents(~Element,right),equation),
      pair_reconstructed(~Element,right)))

def observe_pair_outcome(~Element: Data,result: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) -> Operations.Outcome<PairOwners(Element)>:
  match result:
    case Operations.Accepted{value}: Operations.Accepted{observe_pair(~Element,value)}
    case Operations.Rejected{value}: Operations.Rejected{observe_pair(~Element,value)}

type PairCall is Data:
  PairCall{valid: Bool,count: Maybe<&2,U32>,source: Views.View,target: Views.View,input_length: Nat,output_length: Nat}

def combined_pair(~Element: Data,left_length: Nat,right_length: Nat,result: AccessLaws.ModelResult<Tree.Tree<Element>,Element>) -> PairOwners(Element):
  AccessLaws.ModelResult{right,left} = result
  (Owner{Tree.pack(Element,left),left_length},Owner{Tree.pack(Element,right),right_length})

def combine_owned(~Element: Data,left_length: Nat,right_length: Nat,result: Array<Element> & Array<Element>) -> PairOwners(Element):
  (right,left) = result
  (Owner{left,left_length},Owner{right,right_length})

def combine_finished(~Element: Data,+left_length: Nat,+right_length: Nat,result: Array<Element> & Array<Element>,
  certificates: TraversalCertificate.Certificates<Array<Element>,Element,Certificate.family(~Element),result>,+model: AccessLaws.ModelResult<Tree.Tree<Element>,Element>,
  equation: {result == AccessLaws.pack_result(~Array<Element>,~Tree.Tree<Element>,~Element,~ArrayLaws.pack_tree(~Element),model) : Array<Element> & Array<Element>}) ->
  {observe_pair(~Element,Operations.combine_buffers(~Element,left_length,right_length,result,certificates)) == combined_pair(~Element,left_length,right_length,model) : PairOwners(Element)}:
  (right,left) = result
  TraversalCertificate.Certificates{right_certificate,left_certificate} = certificates
  AccessLaws.ModelResult{+right_tree,+left_tree} = model
  Equal.cong(Array<Element> & Array<Element>,PairOwners(Element),result => combine_owned(~Element,left_length,right_length,result),
    (right,left),(Tree.pack(Element,right_tree),Tree.pack(Element,left_tree)),equation)

# The kernel law is used only after the actual checked branch has accepted.
def checked_kernel_finish(~Element: Data,~operation: Element -> Element -> Element,~kernel: Kernel(~Element),~kernel_law: KernelLaw(~Element,~operation,~kernel),
  +left: Buffer.Model<Element>,+target: Views.View,+right: Buffer.Model<Element>,+source: Views.View,+count: U32,
  counted: {Views.numel(target) == Some{count} : Maybe<&2,U32>},
  accepted: {OperationBounds.combine_guard(~Element,Buffer.pack(~Element,left),target,Buffer.pack(~Element,right),source) == True{} : Bool}) ->
  {observe_pair(~Element,kernel(U32.to_nat(count),source,ContiguousBounds.offset(target),Buffer.pack(~Element,left),Buffer.pack(~Element,right))) ==
    combined_pair(~Element,Buffer.length(~Element,left),Buffer.length(~Element,right),AccessLaws.combine_model(~Element,~operation,
      Access.zero_contiguous(source) && U32.is_zero(ContiguousBounds.offset(target)),U32.to_nat(count),source,ContiguousBounds.offset(target),Buffer.tree(~Element,right),Buffer.tree(~Element,left))) : PairOwners(Element)}:
  Buffer.Model{+a,+al,+ac} = left
  Buffer.Model{+b,+bl,+bc} = right
  # the kernel builds its view again from the fields of source
  Views.View{+axes,+offset} = source
  +position = ContiguousBounds.offset(target)
  +fast = Access.zero_contiguous(source) && U32.is_zero(position)
  Equal.trans(PairOwners(Element),observe_pair(~Element,kernel(U32.to_nat(count),source,position,Storage.Buffer{Tree.pack(Element,a),al,ac},Storage.Buffer{Tree.pack(Element,b),bl,bc})),
    observe_pair(~Element,Operations.combine_kernel(~Element,~operation,U32.to_nat(count),source,position,Storage.Buffer{Tree.pack(Element,a),al,ac},Storage.Buffer{Tree.pack(Element,b),bl,bc})),
    combined_pair(~Element,al,bl,AccessLaws.combine_model(~Element,~operation,fast,U32.to_nat(count),source,position,b,a)),
    contents_congruent(~Element,kernel(U32.to_nat(count),source,position,Storage.Buffer{Tree.pack(Element,a),al,ac},Storage.Buffer{Tree.pack(Element,b),bl,bc}),
      Operations.combine_kernel(~Element,~operation,U32.to_nat(count),source,position,Storage.Buffer{Tree.pack(Element,a),al,ac},Storage.Buffer{Tree.pack(Element,b),bl,bc}),
      kernel_law(Storage.Buffer{Tree.pack(Element,a),al,ac},target,Storage.Buffer{Tree.pack(Element,b),bl,bc},source,count,counted,accepted)),
    combine_finished(~Element,al,bl,Access.combine_run(~Element,~operation,fast,U32.to_nat(count),source,position,Tree.pack(Element,b),Tree.pack(Element,a)),
      AccessCertificate.combine(~Element,~operation,fast,U32.to_nat(count),source,position,Tree.pack(Element,b),Tree.pack(Element,a),bc,ac),
      AccessLaws.combine_model(~Element,~operation,fast,U32.to_nat(count),source,position,b,a),AccessLaws.combine(~Element,~operation,fast,U32.to_nat(count),source,position,b,a)))

def combine_model(~Element: Data,~operation: Element -> Element -> Element,call: PairCall,right: Tree.Tree<Element>,left: Tree.Tree<Element>) -> Operations.Outcome<PairOwners(Element)>:
  PairCall{valid,count,source,target,right_length,left_length} = call
  match valid count:
    case True{} Some{count}:
      Views.View{axes,+destination} = target
      +source = source
      Operations.Accepted{combined_pair(~Element,left_length,right_length,AccessLaws.combine_model(~Element,~operation,
        Access.zero_contiguous(source) && U32.is_zero(destination),U32.to_nat(count),source,destination,right,left))}
    case valid count: Operations.Rejected{(Owner{Tree.pack(Element,left),left_length},Owner{Tree.pack(Element,right),right_length})}

def combine_proposition(~Element: Data,~operation: Element -> Element -> Element,~kernel: Kernel(~Element),call: PairCall,-right: Array<Element>,-left: Array<Element>,
  -right_tree: Tree.Tree<Element>,-left_tree: Tree.Tree<Element>) -> Type:
  PairCall{valid,count,source,target,right_length,left_length} = call
  @right_certificate: Certificate.Certificate<Element,right> -> @left_certificate: Certificate.Certificate<Element,left> ->
  @guarded: {OperationBounds.combine_guard(~Element,Storage.Buffer{left,left_length,left_certificate},target,Storage.Buffer{right,right_length,right_certificate},source) == valid : Bool} ->
  @counted: {Views.numel(target) == count : Maybe<&2,U32>} ->
  {observe_pair_outcome(~Element,Operations.combine_when_with(~Element,~kernel,valid,count,source,target,
      Storage.Buffer{left,left_length,left_certificate},Storage.Buffer{right,right_length,right_certificate})) ==
    combine_model(~Element,~operation,PairCall{valid,count,source,target,right_length,left_length},right_tree,left_tree) : Operations.Outcome<PairOwners(Element)>}

def packed_combine(~Element: Data,~operation: Element -> Element -> Element,~kernel: Kernel(~Element),~kernel_law: KernelLaw(~Element,~operation,~kernel),
  +call: PairCall,+right: Tree.Tree<Element>,+left: Tree.Tree<Element>) ->
  combine_proposition(~Element,~operation,~kernel,call,Tree.pack(Element,right),Tree.pack(Element,left),right,left):
  PairCall{valid,count,+source,+target,+right_length,+left_length} = call
  match valid count:
    case False{} None{}: right_certificate => left_certificate => guarded => counted => {==}
    case False{} Some{count}: right_certificate => left_certificate => guarded => counted => {==}
    case True{} None{}: right_certificate => left_certificate => guarded => counted => {==}
    case True{} Some{+count}:
      Views.View{+axes,+destination} = target
      +fast = Access.zero_contiguous(source) && U32.is_zero(destination)
      right_certificate => left_certificate => guarded => counted =>
        Equal.cong(PairOwners(Element),Operations.Outcome<PairOwners(Element)>,value => Operations.Accepted{value},
          observe_pair(~Element,kernel(U32.to_nat(count),source,destination,Storage.Buffer{Tree.pack(Element,left),left_length,left_certificate},Storage.Buffer{Tree.pack(Element,right),right_length,right_certificate})),
          combined_pair(~Element,left_length,right_length,AccessLaws.combine_model(~Element,~operation,fast,U32.to_nat(count),source,destination,right,left)),
          checked_kernel_finish(~Element,~operation,~kernel,~kernel_law,Buffer.Model{left,left_length,left_certificate},Views.View{axes,destination},Buffer.Model{right,right_length,right_certificate},source,count,counted,guarded))

def combine_denotation(~Element: Data,~operation: Element -> Element -> Element,left: Storage.Buffer<Element>,+left_view: Views.View,
  right: Storage.Buffer<Element>,+right_view: Views.View) -> Operations.Outcome<PairOwners(Element)>:
  Storage.Buffer{left,+left_length,left_certificate} = left
  Storage.Buffer{right,+right_length,right_certificate} = right
  combine_model(~Element,~operation,PairCall{Shapes.same(Views.shape(left_view),Views.shape(right_view)) && Views.is_contiguous(left_view) &&
    Operations.valid_buffer(left_length,Storage.capacity(~Element,left,left_certificate),left_view) &&
    Operations.valid_buffer(right_length,Storage.capacity(~Element,right,right_certificate),right_view),Views.numel(left_view),right_view,left_view,right_length,left_length},
    Tree.reflect(Element,right),Tree.reflect(Element,left))

def combine(~Element: Data,~operation: Element -> Element -> Element,~kernel: Kernel(~Element),~kernel_law: KernelLaw(~Element,~operation,~kernel),left: Storage.Buffer<Element>,+left_view: Views.View,
  right: Storage.Buffer<Element>,+right_view: Views.View) ->
  {observe_pair_outcome(~Element,Operations.combine_with(~Element,~kernel,left,left_view,right,right_view)) ==
    combine_denotation(~Element,~operation,left,left_view,right,right_view) : Operations.Outcome<PairOwners(Element)>}:
  Storage.Buffer{left,+left_length,left_certificate} = left
  Certificate.Certificate{+left_certificate_depth,+left_certificate_equation} = left_certificate
  # combine_with builds its views again from their fields
  Views.View{+left_axes,+left_offset} = left_view
  Storage.Buffer{right,+right_length,right_certificate} = right
  Certificate.Certificate{+right_certificate_depth,+right_certificate_equation} = right_certificate
  Views.View{+right_axes,+right_offset} = right_view
  +call = {PairCall{Shapes.same(Views.shape(left_view),Views.shape(right_view)) && Views.is_contiguous(left_view) &&
    Operations.valid_buffer(left_length,Storage.capacity(~Element,left,Certificate.Certificate{left_certificate_depth,left_certificate_equation}),left_view) &&
    Operations.valid_buffer(right_length,Storage.capacity(~Element,right,Certificate.Certificate{right_certificate_depth,right_certificate_equation}),right_view),Views.numel(left_view),right_view,left_view,right_length,left_length} : PairCall}
  StorageLaws.lift_pair_observation(~Element,~PairCall,~combine_proposition(~Element,~operation,~kernel),call,right,left,
    StorageLaws.observe(Element,right),StorageLaws.observe(Element,left),right => left => packed_combine(~Element,~operation,~kernel,~kernel_law,call,right,left))(Certificate.Certificate{right_certificate_depth,right_certificate_equation})(Certificate.Certificate{left_certificate_depth,left_certificate_equation})({==})({==})

def MapLaw_context(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,~mapper: @+count: Nat -> @+position: U32 -> @+context: Context -> Storage.Buffer<Element> -> Storage.Buffer<Element>) -> Type:
  @+count: Nat -> @+position: U32 -> @+context: Context -> @+tree: Tree.Tree<Element> -> @+length: Nat -> @+certificate: Certificate.Certificate<Element,Tree.pack(Element,tree)> ->
  {observe_owner(~Element,mapper(count,position,context,Storage.Buffer{Tree.pack(Element,tree),length,certificate})) ==
    Owner{Traversal.map_in_place(~Element,~Context,~operation,count,position,context,Tree.pack(Element,tree)),length} : Owner<&1,Array<Element>>}

def mapped_context(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,+count: Nat,+position: U32,+context: Context,+tree: Tree.Tree<Element>,+length: Nat,
  +certificate: Certificate.Certificate<Element,Tree.pack(Element,tree)>) ->
  {observe_owner(~Element,Operations.mapped_context(~Element,~Context,~operation,count,position,context,Storage.Buffer{Tree.pack(Element,tree),length,certificate})) ==
    Owner{Traversal.map_in_place(~Element,~Context,~operation,count,position,context,Tree.pack(Element,tree)),length} : Owner<&1,Array<Element>>}:
  {==}

# ---- A fold over a range of a buffer ----

# The certificate a fold of a range returns: the input's, carried to the
# finished state.
def stated_certificate(~Element: Data,~Accumulator: Data,-input: Array<Element>,-accumulator: Accumulator,-position: U32,-context: Traversal.Strided,
  -state: Traversal.State<&1,&2,Array<Element>,Accumulator,Traversal.Strided>,
  proof: {Traversal.State{input,accumulator,position,context} == state : Traversal.State<&1,&2,Array<Element>,Accumulator,Traversal.Strided>},
  certificate: Certificate.Certificate<Element,input>) ->
  Certificate.Certificate<Element,TraversalWitness.first(~Array<Element>,~Accumulator,Traversal.finish_fold(~Array<Element>,~Accumulator,~Traversal.Strided,state))>:
  %TraversalWitness.reader_kept(~Array<Element>,~Accumulator,~Traversal.Strided,input,accumulator,position,context,state,proof) : Certificate.Certificate<Element,_>
  certificate

# The buffer of a finished fold is the input with its own certificate.
def stated_kept(~Element: Data,~Accumulator: Data,+length: Nat,-input: Array<Element>,-accumulator: Accumulator,-position: U32,-context: Traversal.Strided,
  -state: Traversal.State<&1,&2,Array<Element>,Accumulator,Traversal.Strided>,
  +same: {Traversal.State{input,accumulator,position,context} == state : Traversal.State<&1,&2,Array<Element>,Accumulator,Traversal.Strided>},
  +certificate: Certificate.Certificate<Element,input>) ->
  {Operations.range_folded(~Element,~Accumulator,length,Traversal.finish_fold(~Array<Element>,~Accumulator,~Traversal.Strided,state),
      stated_certificate(~Element,~Accumulator,input,accumulator,position,context,state,same,certificate)) ==
    (Storage.Buffer{input,length,certificate},Pair.snd(Array<Element>,Accumulator,Traversal.finish_fold(~Array<Element>,~Accumulator,~Traversal.Strided,state))) :
    Storage.Buffer<Element> & Accumulator}:
  %proof@same :
    {Operations.range_folded(~Element,~Accumulator,length,Traversal.finish_fold(~Array<Element>,~Accumulator,~Traversal.Strided,_),
        stated_certificate(~Element,~Accumulator,input,accumulator,position,context,_,proof,certificate)) ==
      (Storage.Buffer{input,length,certificate},Pair.snd(Array<Element>,Accumulator,Traversal.finish_fold(~Array<Element>,~Accumulator,~Traversal.Strided,_))) :
      Storage.Buffer<Element> & Accumulator}
  {==}

# A fold of a range gives the input buffer back with its certificate.
def range_kept(~Element: Data,~Accumulator: Data,+length: Nat,-input: Array<Element>,-state: Traversal.State<&1,&2,Array<Element>,Accumulator,Traversal.Strided>,
  evidence: TraversalWitness.FoldEvidence<Array<Element>,Accumulator,Traversal.Strided,input,state>,+certificate: Certificate.Certificate<Element,input>) ->
  {Operations.range_folded(~Element,~Accumulator,length,Traversal.finish_fold(~Array<Element>,~Accumulator,~Traversal.Strided,state),
      AccessCertificate.range_stated(~Element,~Accumulator,input,state,evidence,certificate)) ==
    (Storage.Buffer{input,length,certificate},Pair.snd(Array<Element>,Accumulator,Traversal.finish_fold(~Array<Element>,~Accumulator,~Traversal.Strided,state))) :
    Storage.Buffer<Element> & Accumulator}:
  match evidence:
    case TraversalWitness.FoldEvidence{accumulator,position,context,equation}:
      stated_kept(~Element,~Accumulator,length,input,accumulator,position,context,state,
        Equal.sym(Traversal.State<&1,&2,Array<Element>,Accumulator,Traversal.Strided>,state,Traversal.State{input,accumulator,position,context},equation),certificate)

# The fold of count values of a buffer model from position on, step apart,
# keeps the buffer and folds the values of its tree.
def fold_range_model(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,+count: Nat,+position: U32,+step: U32,+initial: Accumulator,
  +model: Buffer.Model<Element>) ->
  {Operations.fold_range(~Element,~Accumulator,~operation,count,position,step,initial,Buffer.pack(~Element,model)) ==
    (Buffer.pack(~Element,model),Fold.values(~Tree.Tree<Element>,~Element,~Accumulator,~Traversal.Strided,~ArrayLaws.sample_strided(~Element),~operation,count,0,
      Buffer.tree(~Element,model),Traversal.Strided{position,step},initial)) : Storage.Buffer<Element> & Accumulator}:
  Buffer.Model{+tree,+length,+certificate} = model
  +context = {Traversal.Strided{position,step} : Traversal.Strided}
  +values = Fold.values(~Tree.Tree<Element>,~Element,~Accumulator,~Traversal.Strided,~ArrayLaws.sample_strided(~Element),~operation,count,0,tree,context,initial)
  Equal.trans(Storage.Buffer<Element> & Accumulator,
    Operations.fold_range(~Element,~Accumulator,~operation,count,position,step,initial,Storage.Buffer{Tree.pack(Element,tree),length,certificate}),
    (Storage.Buffer{Tree.pack(Element,tree),length,certificate},Pair.snd(Array<Element>,Accumulator,Traversal.fold(~Array<Element>,~Element,~Accumulator,~Traversal.Strided,~Traversal.strided(~Element),~operation,count,0,Tree.pack(Element,tree),initial,context))),
    (Storage.Buffer{Tree.pack(Element,tree),length,certificate},values),
    range_kept(~Element,~Accumulator,length,Tree.pack(Element,tree),
      Loop.run(~Traversal.State<&1,&2,Array<Element>,Accumulator,Traversal.Strided>,
        ~Traversal.fold_step(~Array<Element>,~Element,~Accumulator,~Traversal.Strided,~Traversal.strided(~Element),~operation),count,
        Traversal.State{Tree.pack(Element,tree),initial,0,context}),
      TraversalWitness.fold_run(~Array<Element>,~Element,~Accumulator,~Traversal.Strided,~TraversalWitness.array_evidence(~Element),~Traversal.strided(~Element),~operation,
        ~TraversalWitness.strided_read(~Element),count,Tree.pack(Element,tree),Traversal.State{Tree.pack(Element,tree),initial,0,context},
        Certificate.witness(Element,Tree.pack(Element,tree),certificate),TraversalWitness.FoldEvidence{initial,0,context,{==}}),
      certificate),
    Equal.cong(Accumulator,Storage.Buffer<Element> & Accumulator,value => (Storage.Buffer{Tree.pack(Element,tree),length,certificate},value),
      Pair.snd(Array<Element>,Accumulator,Traversal.fold(~Array<Element>,~Element,~Accumulator,~Traversal.Strided,~Traversal.strided(~Element),~operation,count,0,Tree.pack(Element,tree),initial,context)),values,
      Equal.cong(Array<Element> & Accumulator,Accumulator,pair => Pair.snd(Array<Element>,Accumulator,pair),Traversal.fold(~Array<Element>,~Element,~Accumulator,~Traversal.Strided,~Traversal.strided(~Element),~operation,count,0,Tree.pack(Element,tree),initial,context),(Tree.pack(Element,tree),values),
        Fold.fold(~Array<Element>,~Tree.Tree<Element>,~Element,~Accumulator,~Traversal.Strided,~ArrayLaws.pack_tree(~Element),~Traversal.strided(~Element),
          ~ArrayLaws.sample_strided(~Element),~operation,~ArrayLaws.strided_read_refines_storage(~Element),count,0,tree,context,initial))))
