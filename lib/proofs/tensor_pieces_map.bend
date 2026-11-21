# Piece map: a leaf on every piece in place. A leaf states, as a step, what it
# makes of the contents of its piece, and what the step keeps of a piece. The
# map of the cells of a buffer through an operation is one such step.
import Base
import ./tensor_pieces_model.bend as Model
import ./tensor_pieces_content.bend as Content
import ./tensor_pieces_each.bend as Each
import ./tensor_pieces_plan.bend as PiecesPlan
import ./tensor_global_shape_count.bend as Global
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_operations_refinement.bend as OperationLaws
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./storage_shape.bend as Shape
import ./word_power_of_two.bend as Power
import ./nat_order.bend as Order
import ./traversal_map_readback.bend as MapReadback
import ./ordered_ranges.bend as Ranges
import ./sequence_membership.bend as Members
import ../tensor_pieces.bend as Pieces
import ../tensor_tasks.bend as Tasks
import ../storage_buffer.bend as Storage
import ../tensor.bend as Tensors
import ../traversal_array.bend as Traversal
import ../kernel_gemm.bend as KernelGemm

# ---- A leaf on every piece ----

# What a leaf makes of the contents of its piece.
def Rule(~Element: Data,~Context: Data,~leaf: Tensors.PieceMap(~Element,~Context),~step: @+context: Context -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>) -> Type:
  @+context: Context -> @+first: U32 -> @+count: U32 -> @buffer: Storage.Buffer<Element> ->
  {Buffer.contents(~Element,leaf(context,first,count,buffer)) == step(context,first,count,Buffer.contents(~Element,buffer)) : Buffer.Content<Element>}

def stepped(~Element: Data,~Context: Data,~step: @+context: Context -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,+context: Context,content: Model.Content<Element>) -> Model.Content<Element>:
  match content:
    case Model.PieceContent{buffer,+first,+rows}: Model.PieceContent{step(context,first,rows,buffer),first,rows}
    case Model.Joined{left,right}: Model.Joined{stepped(~Element,~Context,~step,context,left),stepped(~Element,~Context,~step,context,right)}

def piece_model(~Element: Data,~Context: Data,~step: @+context: Context -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,context: Context,first: U32,count: U32,input: Buffer.Model<Element>) -> Model.Content<Element>:
  +first = first
  +count = count
  Model.PieceContent{step(context,first,count,Buffer.contents(~Element,Buffer.pack(~Element,input))),first,count}

def piece(~Element: Data,~Context: Data,~leaf: Tensors.PieceMap(~Element,~Context),~step: @+context: Context -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,~rule: Rule(~Element,~Context,~leaf,~step),
  +context: Context,+first: U32,+count: U32,+input: Buffer.Model<Element>) ->
  {Model.contents(~Element,Tensors.mapped_piece(~Element,~Context,~leaf,context,first,count,Buffer.pack(~Element,input),Unit{})) == piece_model(~Element,~Context,~step,context,first,count,input) : Model.Content<Element>}:
  Equal.cong(Buffer.Content<Element>,Model.Content<Element>,buffer => Model.PieceContent{buffer,first,count},
    Buffer.contents(~Element,leaf(context,first,count,Buffer.pack(~Element,input))),step(context,first,count,Buffer.contents(~Element,Buffer.pack(~Element,input))),
    rule(context,first,count,Buffer.pack(~Element,input)))

def modeled(~Element: Data,~Context: Data,~step: @+context: Context -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,+context: Context,pieces: Model.Model<Element>) ->
  {Each.content_map(~Element,~Element,~Context,~piece_model(~Element,~Context,~step),context,pieces) == stepped(~Element,~Context,~step,context,Model.contents(~Element,Model.pack(~Element,pieces))) : Model.Content<Element>}:
  match pieces:
    case Model.Leaf{+input,+first,+rows}: {==}
    case Model.Fork{+left,+right}:
      %modeled(~Element,~Context,~step,context,left) :
        {Model.Joined{Each.content_map(~Element,~Element,~Context,~piece_model(~Element,~Context,~step),context,left),Each.content_map(~Element,~Element,~Context,~piece_model(~Element,~Context,~step),context,right)} ==
          Model.Joined{_,stepped(~Element,~Context,~step,context,Model.contents(~Element,Model.pack(~Element,right)))} : Model.Content<Element>}
      %modeled(~Element,~Context,~step,context,right) :
        {Model.Joined{Each.content_map(~Element,~Element,~Context,~piece_model(~Element,~Context,~step),context,left),Each.content_map(~Element,~Element,~Context,~piece_model(~Element,~Context,~step),context,right)} ==
          Model.Joined{Each.content_map(~Element,~Element,~Context,~piece_model(~Element,~Context,~step),context,left),_} : Model.Content<Element>}
      {==}

# The leaf on any task plan, run in parallel or not, steps every piece.
def grouped(~Element: Data,~Context: Data,~leaf: Tensors.PieceMap(~Element,~Context),~step: @+context: Context -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,~rule: Rule(~Element,~Context,~leaf,~step),
  +context: Context,+tasks: Pieces.Ranges,+parallel: Bool,+pieces: Model.Model<Element>) ->
  {Model.contents(~Element,Pieces.grouped(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Context,~Tensors.mapped_piece(~Element,~Context,~leaf),tasks,parallel,context,Model.pack(~Element,pieces))) ==
    stepped(~Element,~Context,~step,context,Model.contents(~Element,Model.pack(~Element,pieces))) : Model.Content<Element>}:
  %Equal.sym(Pieces.Pieces(~Element),
    Pieces.grouped(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Context,~Tensors.mapped_piece(~Element,~Context,~leaf),tasks,parallel,context,Model.pack(~Element,pieces)),
    Pieces.each(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~Pieces.unshared,~Tensors.mapped_piece(~Element,~Context,~leaf),~Pieces.same(~Storage.Buffer<Element>),
      Model.layout(~Element,pieces),parallel,context,Model.pack(~Element,pieces),Unit{}),
    Each.grouped_each(~Element,~Storage.Buffer<Element>,~Context,~Tensors.mapped_piece(~Element,~Context,~leaf),~Pieces.same(~Storage.Buffer<Element>),tasks,parallel,context,pieces)) :
    {Model.contents(~Element,_) == stepped(~Element,~Context,~step,context,Model.contents(~Element,Model.pack(~Element,pieces))) : Model.Content<Element>}
  Equal.trans(Model.Content<Element>,
    Model.contents(~Element,Pieces.each(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~Pieces.unshared,~Tensors.mapped_piece(~Element,~Context,~leaf),~Pieces.same(~Storage.Buffer<Element>),
      Model.layout(~Element,pieces),parallel,context,Model.pack(~Element,pieces),Unit{})),
    Each.content_map(~Element,~Element,~Context,~piece_model(~Element,~Context,~step),context,pieces),stepped(~Element,~Context,~step,context,Model.contents(~Element,Model.pack(~Element,pieces))),
    Each.mapped(~Element,~Element,~Context,~Tensors.mapped_piece(~Element,~Context,~leaf),~Pieces.same(~Storage.Buffer<Element>),~piece_model(~Element,~Context,~step),
      ~piece(~Element,~Context,~leaf,~step,~rule),pieces,parallel,context),
    modeled(~Element,~Context,~step,context,pieces))

def stepped_layout(~Element: Data,~Context: Data,~step: @+context: Context -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,+context: Context,content: Model.Content<Element>) ->
  {Model.content_layout(~Element,stepped(~Element,~Context,~step,context,content)) == Model.content_layout(~Element,content) : Pieces.Ranges}:
  match content:
    case Model.PieceContent{buffer,+first,+rows}: {==}
    case Model.Joined{+left,+right}:
      %stepped_layout(~Element,~Context,~step,context,left) :
        {Pieces.Split{Model.content_layout(~Element,stepped(~Element,~Context,~step,context,left)),Model.content_layout(~Element,stepped(~Element,~Context,~step,context,right))} ==
          Pieces.Split{_,Model.content_layout(~Element,right)} : Pieces.Ranges}
      %stepped_layout(~Element,~Context,~step,context,right) :
        {Pieces.Split{Model.content_layout(~Element,stepped(~Element,~Context,~step,context,left)),Model.content_layout(~Element,stepped(~Element,~Context,~step,context,right))} ==
          Pieces.Split{Model.content_layout(~Element,stepped(~Element,~Context,~step,context,left)),_} : Pieces.Ranges}
      {==}

# What a step keeps of one piece: its storage ready, its length, and the
# values of another reader.
def Stepped(~Element: Data,~After: Data,~after_value: After -> U32 -> Element,+after: After,+first: U32,+count: U32,+input: Buffer.Content<Element>,+output: Buffer.Content<Element>) -> Data:
  KernelGemm.Both<Buffer.ContentReady<Element,output>,KernelGemm.Both<{Buffer.content_length(~Element,output) == Buffer.content_length(~Element,input) : Nat},
    Content.piece_values(~Element,~After,~after_value,after,first,count,output)>>

# What an operation proves of its step on one ready piece of a reader's
# values. A scope names the context of the leaf and the two readers.
def StepRule(~Element: Data,~Context: Data,~step: @+context: Context -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,
  ~Before: Data,~before_value: Before -> U32 -> Element,~After: Data,~after_value: After -> U32 -> Element,
  ~Scope: Data,~context_of: @+scope: Scope -> Context,~before_of: @+scope: Scope -> Before,~after_of: @+scope: Scope -> After,~total_of: @+scope: Scope -> Nat) -> Type:
  @+scope: Scope -> @+first: U32 -> @+count: U32 -> @+input: Buffer.Content<Element> -> @+ready: Buffer.ContentReady<Element,input> ->
  @+length_exact: {Buffer.content_length(~Element,input) == U32.to_nat(count) : Nat} ->
  @+inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),total_of(scope)) == True{} : Bool} ->
  @+values: Content.piece_values(~Element,~Before,~before_value,before_of(scope),first,count,input) ->
  Stepped(~Element,~After,~after_value,after_of(scope),first,count,input,step(context_of(scope),first,count,input))

# Every piece fits, owns ready storage and holds the reader's values.
def Facts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+total: Nat,content: Model.Content<Element>) -> Data:
  KernelGemm.Both<Content.Contents(~Element,total,content),KernelGemm.Both<Model.ContentReady(~Element,content),Content.Values(~Element,~Logical,~original,logical,content)>>

def piece_facts(~Element: Data,~After: Data,~after_value: After -> U32 -> Element,+after: After,+first: U32,+count: U32,+total: Nat,+input: Buffer.Content<Element>,+output: Buffer.Content<Element>,
  +length_exact: {Buffer.content_length(~Element,input) == U32.to_nat(count) : Nat},+inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),total) == True{} : Bool},
  made: Stepped(~Element,~After,~after_value,after,first,count,input,output)) ->
  Facts(~Element,~After,~after_value,after,total,Model.PieceContent{output,first,count}):
  KernelGemm.Both{+result_ready,KernelGemm.Both{+same_length,+result_values}} = made
  KernelGemm.Both{Content.Fields{Equal.trans(Nat,Buffer.content_length(~Element,output),Buffer.content_length(~Element,input),U32.to_nat(count),same_length,length_exact),inside},
    KernelGemm.Both{result_ready,result_values}}

def joined_facts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+total: Nat,+left: Model.Content<Element>,+right: Model.Content<Element>,
  left_facts: Facts(~Element,~Logical,~original,logical,total,left),right_facts: Facts(~Element,~Logical,~original,logical,total,right)) ->
  Facts(~Element,~Logical,~original,logical,total,Model.Joined{left,right}):
  KernelGemm.Both{left_fields,KernelGemm.Both{left_ready,left_values}} = left_facts
  KernelGemm.Both{right_fields,KernelGemm.Both{right_ready,right_values}} = right_facts
  KernelGemm.Both{KernelGemm.Both{left_fields,right_fields},KernelGemm.Both{KernelGemm.Both{left_ready,right_ready},KernelGemm.Both{left_values,right_values}}}

def stepped_facts(~Element: Data,~Context: Data,~step: @+context: Context -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,
  ~Before: Data,~before_value: Before -> U32 -> Element,~After: Data,~after_value: After -> U32 -> Element,
  ~Scope: Data,~context_of: @+scope: Scope -> Context,~before_of: @+scope: Scope -> Before,~after_of: @+scope: Scope -> After,~total_of: @+scope: Scope -> Nat,
  ~rule: StepRule(~Element,~Context,~step,~Before,~before_value,~After,~after_value,~Scope,~context_of,~before_of,~after_of,~total_of),+scope: Scope,content: Model.Content<Element>,
  fields: Content.Contents(~Element,total_of(scope),content),ready: Model.ContentReady(~Element,content),values: Content.Values(~Element,~Before,~before_value,before_of(scope),content)) ->
  Facts(~Element,~After,~after_value,after_of(scope),total_of(scope),stepped(~Element,~Context,~step,context_of(scope),content)):
  match content:
    case Model.PieceContent{+input,+first,+count}:
      Content.Fields{+length_exact,+inside} = fields
      piece_facts(~Element,~After,~after_value,after_of(scope),first,count,total_of(scope),input,step(context_of(scope),first,count,input),length_exact,inside,
        rule(scope,first,count,input,ready,length_exact,inside,values))
    case Model.Joined{+left,+right}:
      KernelGemm.Both{left_fields,right_fields} = fields
      KernelGemm.Both{left_ready,right_ready} = ready
      KernelGemm.Both{left_values,right_values} = values
      joined_facts(~Element,~After,~after_value,after_of(scope),total_of(scope),stepped(~Element,~Context,~step,context_of(scope),left),stepped(~Element,~Context,~step,context_of(scope),right),
        stepped_facts(~Element,~Context,~step,~Before,~before_value,~After,~after_value,~Scope,~context_of,~before_of,~after_of,~total_of,~rule,scope,left,left_fields,left_ready,left_values),
        stepped_facts(~Element,~Context,~step,~Before,~before_value,~After,~after_value,~Scope,~context_of,~before_of,~after_of,~total_of,~rule,scope,right,right_fields,right_ready,right_values))

# What the contract states of pieces of these contents.
def Holds(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,content: Model.Content<Element>) -> Data:
  KernelGemm.Both<KernelGemm.Both<Content.WellFormed<Element,ShapeLaws.elements(shape),content>,KernelGemm.Both<Global.Counted<shape>,Global.Canonical(~Element,shape,content)>>,
    Content.Values(~Element,~Logical,~original,logical,content)>

def holds(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,+content: Model.Content<Element>,
  facts: Facts(~Element,~Logical,~original,logical,ShapeLaws.elements(shape),content),coverage: PiecesPlan.Coverage(Model.content_layout(~Element,content),0n,ShapeLaws.elements(shape)),
  counted: Global.Counted<shape>,canonical: Global.Canonical(~Element,shape,content)) -> Holds(~Element,~Logical,~original,logical,shape,content):
  KernelGemm.Both{fields,KernelGemm.Both{ready,values}} = facts
  KernelGemm.Both{KernelGemm.Both{Content.WellFormed{fields,ready,coverage},KernelGemm.Both{counted,canonical}},values}

# The leaf on the tasks of any decision keeps the contract with the reader of
# its steps.
def decided(~Element: Data,~Context: Data,~leaf: Tensors.PieceMap(~Element,~Context),~step: @+context: Context -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,~rule: Rule(~Element,~Context,~leaf,~step),
  ~Before: Data,~before_value: Before -> U32 -> Element,~After: Data,~after_value: After -> U32 -> Element,
  ~Scope: Data,~context_of: @+scope: Scope -> Context,~before_of: @+scope: Scope -> Before,~after_of: @+scope: Scope -> After,~total_of: @+scope: Scope -> Nat,
  ~step_rule: StepRule(~Element,~Context,~step,~Before,~before_value,~After,~after_value,~Scope,~context_of,~before_of,~after_of,~total_of),
  +scope: Scope,decision: Tasks.Decision,+pieces: Model.Model<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  +total_exact: {total_of(scope) == ShapeLaws.elements(shape) : Nat},contract: Global.Contract(~Element,~Before,~before_value,before_of(scope),Tensors.Tensor{Model.pack(~Element,pieces),shape,fill,Tensors.Usable{},workers})) ->
  Global.Contract(~Element,~After,~after_value,after_of(scope),Tensors.mapped_decided(~Element,~Context,~leaf,context_of(scope),decision,Model.pack(~Element,pieces),shape,fill,workers)):
  match decision:
    case Tasks.Decision{+parallel,+tasks}:
      KernelGemm.Both{KernelGemm.Both{Content.WellFormed{+fields,+ready,+coverage},KernelGemm.Both{+counted,+canonical}},+values} = contract
      +context = context_of(scope)
      +input = Model.contents(~Element,Model.pack(~Element,pieces))
      %Equal.sym(Model.Content<Element>,
        Model.contents(~Element,Pieces.grouped(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Context,~Tensors.mapped_piece(~Element,~Context,~leaf),tasks,parallel,context,Model.pack(~Element,pieces))),
        stepped(~Element,~Context,~step,context,input),grouped(~Element,~Context,~leaf,~step,~rule,context,tasks,parallel,pieces)) :
        Holds(~Element,~After,~after_value,after_of(scope),shape,_)
      holds(~Element,~After,~after_value,after_of(scope),shape,stepped(~Element,~Context,~step,context,input),
        %total_exact : Facts(~Element,~After,~after_value,after_of(scope),_,stepped(~Element,~Context,~step,context,input))
          stepped_facts(~Element,~Context,~step,~Before,~before_value,~After,~after_value,~Scope,~context_of,~before_of,~after_of,~total_of,~step_rule,scope,input,
            %Equal.sym(Nat,total_of(scope),ShapeLaws.elements(shape),total_exact) : Content.Contents(~Element,_,input) fields,ready,values),
        %Equal.sym(Pieces.Ranges,Model.content_layout(~Element,stepped(~Element,~Context,~step,context,input)),Model.content_layout(~Element,input),
          stepped_layout(~Element,~Context,~step,context,input)) : PiecesPlan.Coverage(_,0n,ShapeLaws.elements(shape)) coverage,counted,
        %Equal.sym(Pieces.Ranges,Model.content_layout(~Element,stepped(~Element,~Context,~step,context,input)),Model.content_layout(~Element,input),
          stepped_layout(~Element,~Context,~step,context,input)) : {_ == Pieces.ranges_of_grid(Tensors.grid_of(shape)) : Pieces.Ranges} canonical)

def refinement_values(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,count: Nat,context: Context,input: Tree.Tree<Element>) -> Tree.Tree<Element>:
  MapReadback.run(~Element,~Context,~operation,count,0,context,input)

def raw(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,+count: Nat,+context: Context,+input: Tree.Tree<Element>) ->
  {Traversal.map_in_place(~Element,~Context,~operation,count,0,context,Tree.pack(Element,input)) == Tree.pack(Element,refinement_values(~Element,~Context,~operation,count,context,input)) : Array<Element>}:
  MapReadback.packed(~Element,~Context,~operation,count,0,context,input)

# The first count cells of a buffer through the operation.
def map_step(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,context: Context,count: Nat,input: Buffer.Content<Element>) -> Buffer.Content<Element>:
  Buffer.Content{tree,+length} = input
  Buffer.Content{refinement_values(~Element,~Context,~operation,count,context,tree),length}

def owner_content(~Element: Data,owner: OperationLaws.Owner<&1,Array<Element>>) -> Buffer.Content<Element>:
  OperationLaws.Owner{storage,length} = owner
  Buffer.Content{Tree.reflect(Element,storage),length}

def contents_of_owner(~Element: Data,buffer: Storage.Buffer<Element>) ->
  {Buffer.contents(~Element,buffer) == owner_content(~Element,OperationLaws.observe_owner(~Element,buffer)) : Buffer.Content<Element>}:
  Storage.Buffer{storage,length,certificate} = buffer
  {==}

def refinement_leaf(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,+context: Context,+tree: Tree.Tree<Element>,+count: Nat,+length: Nat) ->
  {owner_content(~Element,OperationLaws.Owner{Traversal.map_in_place(~Element,~Context,~operation,count,0,context,Tree.pack(Element,tree)),length}) ==
    map_step(~Element,~Context,~operation,context,count,Buffer.Content{tree,length}) : Buffer.Content<Element>}:
  Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{tree,length},
    Tree.reflect(Element,Traversal.map_in_place(~Element,~Context,~operation,count,0,context,Tree.pack(Element,tree))),refinement_values(~Element,~Context,~operation,count,context,tree),
    Equal.trans(Tree.Tree<Element>,Tree.reflect(Element,Traversal.map_in_place(~Element,~Context,~operation,count,0,context,Tree.pack(Element,tree))),
      Tree.reflect(Element,Tree.pack(Element,refinement_values(~Element,~Context,~operation,count,context,tree))),refinement_values(~Element,~Context,~operation,count,context,tree),
      Equal.cong(Array<Element>,Tree.Tree<Element>,storage => Tree.reflect(Element,storage),Traversal.map_in_place(~Element,~Context,~operation,count,0,context,Tree.pack(Element,tree)),
        Tree.pack(Element,refinement_values(~Element,~Context,~operation,count,context,tree)),raw(~Element,~Context,~operation,count,context,tree)),
      StorageLaws.reflect_packed(Element,refinement_values(~Element,~Context,~operation,count,context,tree))))

def mapped_model(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,~mapper: @+count: Nat -> @+position: U32 -> @+context: Context -> Storage.Buffer<Element> -> Storage.Buffer<Element>,
  ~lawful: OperationLaws.MapLaw_context(~Element,~Context,~operation,~mapper),+context: Context,+count: Nat,+input: Buffer.Model<Element>) ->
  {Buffer.contents(~Element,mapper(count,0,context,Buffer.pack(~Element,input))) == map_step(~Element,~Context,~operation,context,count,Buffer.contents(~Element,Buffer.pack(~Element,input))) : Buffer.Content<Element>}:
  Buffer.Model{+tree,+length,+certificate} = input
  %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,tree)),tree,StorageLaws.reflect_packed(Element,tree)) :
    {Buffer.contents(~Element,mapper(count,0,context,Storage.Buffer{Tree.pack(Element,tree),length,certificate})) == map_step(~Element,~Context,~operation,context,count,Buffer.Content{_,length}) : Buffer.Content<Element>}
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,mapper(count,0,context,Storage.Buffer{Tree.pack(Element,tree),length,certificate})),
    owner_content(~Element,OperationLaws.observe_owner(~Element,mapper(count,0,context,Storage.Buffer{Tree.pack(Element,tree),length,certificate}))),
    contents_of_owner(~Element,mapper(count,0,context,Storage.Buffer{Tree.pack(Element,tree),length,certificate}))) :
    {_ == map_step(~Element,~Context,~operation,context,count,Buffer.Content{tree,length}) : Buffer.Content<Element>}
  %Equal.sym(OperationLaws.Owner<&1,Array<Element>>,OperationLaws.observe_owner(~Element,mapper(count,0,context,Storage.Buffer{Tree.pack(Element,tree),length,certificate})),
    OperationLaws.Owner{Traversal.map_in_place(~Element,~Context,~operation,count,0,context,Tree.pack(Element,tree)),length},
    lawful(count,0,context,tree,length,certificate)) :
    {owner_content(~Element,_) == map_step(~Element,~Context,~operation,context,count,Buffer.Content{tree,length}) : Buffer.Content<Element>}
  refinement_leaf(~Element,~Context,~operation,context,tree,count,length)

def mapped_observed(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,~mapper: @+count: Nat -> @+position: U32 -> @+context: Context -> Storage.Buffer<Element> -> Storage.Buffer<Element>,
  ~lawful: OperationLaws.MapLaw_context(~Element,~Context,~operation,~mapper),+context: Context,+count: Nat,-buffer: Storage.Buffer<Element>,observation: Buffer.Observation<Element,buffer>) ->
  {Buffer.contents(~Element,mapper(count,0,context,buffer)) == map_step(~Element,~Context,~operation,context,count,Buffer.contents(~Element,buffer)) : Buffer.Content<Element>}:
  Buffer.Observation{+model,+equation} = observation
  %equation : {Buffer.contents(~Element,mapper(count,0,context,_)) == map_step(~Element,~Context,~operation,context,count,Buffer.contents(~Element,buffer)) : Buffer.Content<Element>}
  %equation : {Buffer.contents(~Element,mapper(count,0,context,Buffer.pack(~Element,model))) == map_step(~Element,~Context,~operation,context,count,Buffer.contents(~Element,_)) : Buffer.Content<Element>}
  mapped_model(~Element,~Context,~operation,~mapper,~lawful,context,count,model)

# A lawful mapper of the first count cells of a buffer.
def mapped(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,~mapper: @+count: Nat -> @+position: U32 -> @+context: Context -> Storage.Buffer<Element> -> Storage.Buffer<Element>,
  ~lawful: OperationLaws.MapLaw_context(~Element,~Context,~operation,~mapper),+context: Context,+count: Nat,buffer: Storage.Buffer<Element>) ->
  {Buffer.contents(~Element,mapper(count,0,context,buffer)) == map_step(~Element,~Context,~operation,context,count,Buffer.contents(~Element,buffer)) : Buffer.Content<Element>}:
  mapped_observed(~Element,~Context,~operation,~mapper,~lawful,context,count,buffer,Buffer.observe(~Element,buffer))

def map_ready(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,+context: Context,+count: Nat,+input: Buffer.Content<Element>,ready: Buffer.ContentReady<Element,input>) ->
  Buffer.ContentReady<Element,map_step(~Element,~Context,~operation,context,count,input)>:
  Buffer.Content{+tree,+length} = input
  Buffer.ContentReady{+depth,+balanced,+depth_ok,+room} = ready
  Buffer.ContentReady{depth,MapReadback.balanced(~Element,~Context,~operation,count,0,context,tree,depth,balanced),depth_ok,
    %Equal.sym(U32,Tree.capacity(Element,refinement_values(~Element,~Context,~operation,count,context,tree)),Tree.capacity(Element,tree),MapReadback.capacity(~Element,~Context,~operation,count,0,context,tree)) :
      {Nat.is_le(length,U32.to_nat(_)) == True{} : Bool} room}

# Maps keep the common logical content relation. The static operation may be
# arbitrary: no floating-point algebra or reassociation is used.
type Context<-Operation: Data,-Logical: Data> is Data:
  Context{operation: Operation,logical: Logical}

def value(~Element: Data,~Operation: Data,~Logical: Data,~operation: Operation -> Element -> Element,~original: Logical -> U32 -> Element,
  context: Context<Operation,Logical>,index: U32) -> Element:
  Context{operation_context,logical_context} = context
  operation(operation_context,original(logical_context,index))

# What one piece supplies to every one of its cells.
type CellScope<-Element: Data,-Operation: Data,-Logical: Data> is Data:
  CellScope{operation_context: Operation,logical_context: Logical,first: U32,count: U32,tree: Tree.Tree<Element>,length: Nat,
    ready: Buffer.ContentReady<Element,Buffer.Content{tree,length}>,length_exact: {length == U32.to_nat(count) : Nat}}

def Before(~Element: Data,~Operation: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: CellScope<Element,Operation,Logical>,+index: Nat) -> Data:
  CellScope{operation_context,logical_context,first,count,tree,length,ready,length_exact} = scope
  Content.At(~Element,~Logical,~original,logical_context,first,tree,length,index)

def After(~Element: Data,~Operation: Data,~Logical: Data,~operation: Operation -> Element -> Element,~original: Logical -> U32 -> Element,+scope: CellScope<Element,Operation,Logical>,+index: Nat) -> Data:
  CellScope{operation_context,logical_context,first,count,tree,length,ready,length_exact} = scope
  Content.At(~Element,~Context<Operation,Logical>,~value(~Element,~Operation,~Logical,~operation,~original),Context{operation_context,logical_context},first,
    refinement_values(~Element,~Operation,~operation,U32.to_nat(count),operation_context,tree),length,index)

# A mapped cell stays at its index and holds the operation applied to the
# value of the same cell.
def mapped_cell(~Element: Data,~Operation: Data,~Logical: Data,~operation: Operation -> Element -> Element,~original: Logical -> U32 -> Element,
  +scope: CellScope<Element,Operation,Logical>,+index: Nat,+cell: Before(~Element,~Operation,~Logical,~original,scope,index)) ->
  After(~Element,~Operation,~Logical,~operation,~original,scope,index):
  CellScope{+operation_context,+logical_context,+first,+count,+tree,+length,+ready,+length_exact} = scope
  Buffer.ContentReady{+depth,+balanced,+depth_ok,+room} = ready
  KernelGemm.Both{+inside,+equal} = cell
  +pixel = U32.from_nat(Nat.add(U32.to_nat(first),index))
  KernelGemm.Both{inside,
    Equal.trans(Element,Tree.read(Element,refinement_values(~Element,~Operation,~operation,U32.to_nat(count),operation_context,tree),U32.from_nat(index)),
      operation(operation_context,Tree.read(Element,tree,U32.from_nat(index))),operation(operation_context,original(logical_context,pixel)),
      MapReadback.inside(~Element,~Operation,~operation,U32.to_nat(count),0n,index,operation_context,tree,depth,Tree.capacity(Element,tree),{==},balanced,depth_ok,
        %Equal.sym(U32,Tree.capacity(Element,tree),U32.shln(1,depth),Shape.capacity(Element,depth,tree,balanced)) : {U32.is_le(_,2147483648) == True{} : Bool}
          Power.native_power_bound(depth,Order.successor(depth,30n,depth_ok)),
        %length_exact : {Nat.is_le(_,U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool} room,
        %length_exact : {Nat.is_lt(index,_) == True{} : Bool} inside),
      Equal.cong(Element,Element,element => operation(operation_context,element),Tree.read(Element,tree,U32.from_nat(index)),original(logical_context,pixel),equal))}

# The cells of a piece whose input holds a reader's values hold the operation
# of those values after the map of the piece.
def map_values(~Element: Data,~Operation: Data,~Logical: Data,~operation: Operation -> Element -> Element,~original: Logical -> U32 -> Element,
  +operation_context: Operation,+logical_context: Logical,+first: U32,+count: U32,+input: Buffer.Content<Element>,+ready: Buffer.ContentReady<Element,input>,
  +length_exact: {Buffer.content_length(~Element,input) == U32.to_nat(count) : Nat},+values: Content.piece_values(~Element,~Logical,~original,logical_context,first,count,input)) ->
  Content.piece_values(~Element,~Context<Operation,Logical>,~value(~Element,~Operation,~Logical,~operation,~original),Context{operation_context,logical_context},first,count,
    map_step(~Element,~Operation,~operation,operation_context,U32.to_nat(count),input)):
  Buffer.Content{+tree,+length} = input
  Members.transform_all(~Nat,~CellScope<Element,Operation,Logical>,~Before(~Element,~Operation,~Logical,~original),~After(~Element,~Operation,~Logical,~operation,~original),
    ~mapped_cell(~Element,~Operation,~Logical,~operation,~original),Ranges.range(U32.to_nat(count),0n),
    CellScope{operation_context,logical_context,first,count,tree,length,ready,length_exact},values)
