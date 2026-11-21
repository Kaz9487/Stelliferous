# Combinations of two tensors of one shape. Their pieces are the canonical
# pieces of the shape, so they zip piece by piece; every pair combines into
# the left piece and the step after it keeps the contract with its reader.
import Base
import ../tensor.bend as Tensors
import ../tensor_pieces.bend as Pieces
import ../tensor_tasks.bend as Tasks
import ../storage_buffer.bend as Storage
import ./tensor_pieces_model.bend as Model
import ./tensor_pieces_content.bend as Content
import ./tensor_pieces_each.bend as Each
import ./tensor_pieces_plan.bend as PiecesPlan
import ./tensor_pieces_map.bend as PiecesMap
import ./tensor_pieces_combine_leaf.bend as CombineLeaf
import ./tensor_vector_descriptor_refinement.bend as Descriptor
import ./tensor_global_shape_count.bend as Global
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_operations_refinement.bend as OperationsLaws
import ./storage_buffer_model.bend as Buffer
import ../kernel_gemm.bend as KernelGemm

# The two readers a combination reads at one index.
def combined_of(~Left: Data,~Right: Data,~Scope: Data,~left_of: @+scope: Scope -> Left,~right_of: @+scope: Scope -> Right,+scope: Scope) -> CombineLeaf.Context<Left,Right>:
  CombineLeaf.Context{left_of(scope),right_of(scope)}

# What an operation proves of its step after the combination: the step rule
# of a map from the combined reader.
def Rule(~Element: Data,~Later: Data,~after_step: @+later: Later -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,
  ~operation: Element -> Element -> Element,~Left: Data,~left_value: Left -> U32 -> Element,~Right: Data,~right_value: Right -> U32 -> Element,
  ~After: Data,~after_value: After -> U32 -> Element,~Scope: Data,~later_of: @+scope: Scope -> Later,~left_of: @+scope: Scope -> Left,~right_of: @+scope: Scope -> Right,
  ~after_of: @+scope: Scope -> After,~total_of: @+scope: Scope -> Nat) -> Type:
  PiecesMap.StepRule(~Element,~Later,~after_step,~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~Element,~Left,~Right,~operation,~left_value,~right_value),~After,~after_value,
    ~Scope,~later_of,~combined_of(~Left,~Right,~Scope,~left_of,~right_of),~after_of,~total_of)

# A pair of pieces of one range holding the values of the two readers
# combines and steps into a piece of the reader after the step.
def piece_facts(~Element: Data,~Later: Data,~after_step: @+later: Later -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,
  ~operation: Element -> Element -> Element,~Left: Data,~left_value: Left -> U32 -> Element,~Right: Data,~right_value: Right -> U32 -> Element,
  ~After: Data,~after_value: After -> U32 -> Element,~Scope: Data,~later_of: @+scope: Scope -> Later,~left_of: @+scope: Scope -> Left,~right_of: @+scope: Scope -> Right,
  ~after_of: @+scope: Scope -> After,~total_of: @+scope: Scope -> Nat,
  ~rule: Rule(~Element,~Later,~after_step,~operation,~Left,~left_value,~Right,~right_value,~After,~after_value,~Scope,~later_of,~left_of,~right_of,~after_of,~total_of),
  +scope: Scope,+first: U32,+count: U32,+left: Buffer.Model<Element>,+right: Buffer.Model<Element>,
  +left_fields: Content.piece_fields(~Element,total_of(scope),first,count,Buffer.content(~Element,left)),+left_ready: Buffer.ContentReady<Element,Buffer.content(~Element,left)>,
  +left_values: Content.piece_values(~Element,~Left,~left_value,left_of(scope),first,count,Buffer.content(~Element,left)),
  +right_fields: Content.piece_fields(~Element,total_of(scope),first,count,Buffer.content(~Element,right)),+right_ready: Buffer.ContentReady<Element,Buffer.content(~Element,right)>,
  +right_values: Content.piece_values(~Element,~Right,~right_value,right_of(scope),first,count,Buffer.content(~Element,right))) ->
  PiecesMap.Facts(~Element,~After,~after_value,after_of(scope),total_of(scope),CombineLeaf.piece_model(~Element,~Later,~operation,~after_step,later_of(scope),first,count,left,right)):
  Buffer.Model{+left_tree,+left_length,+left_certificate} = left
  Buffer.Model{+right_tree,+right_length,+right_certificate} = right
  Content.Fields{+left_exact,+left_inside} = left_fields
  Content.Fields{+right_exact,+right_inside} = right_fields
  +left_model = {Buffer.Model{left_tree,left_length,left_certificate} : Buffer.Model<Element>}
  +right_model = {Buffer.Model{right_tree,right_length,right_certificate} : Buffer.Model<Element>}
  +left_content = {Buffer.Content{left_tree,left_length} : Buffer.Content<Element>}
  +right_content = {Buffer.Content{right_tree,right_length} : Buffer.Content<Element>}
  +later = later_of(scope)
  +input = CombineLeaf.leaf_content(~Element,~operation,left_content,right_content)
  +same = CombineLeaf.checked_leaf_model(~Element,~operation,count,left_model,right_model,Equal.sym(Nat,left_length,U32.to_nat(count),left_exact),
    Equal.sym(Nat,right_length,U32.to_nat(count),right_exact),Buffer.ready_content(~Element,left_model,left_ready),Buffer.ready_content(~Element,right_model,right_ready),
    Descriptor.descriptor(count))
  +ready = CombineLeaf.leaf_ready(~Element,~operation,left_content,right_content,left_ready)
  +values = CombineLeaf.semantics_leaf(~Element,~Left,~Right,~operation,~left_value,~right_value,left_of(scope),right_of(scope),first,count,left_content,right_content,
    left_ready,left_values,right_values)
  +made = rule(scope,first,count,input,ready,left_exact,left_inside,values)
  %Equal.sym(Buffer.Content<Element>,CombineLeaf.refinement_leaf_model(~Element,~operation,count,left_model,right_model),input,same) :
    PiecesMap.Facts(~Element,~After,~after_value,after_of(scope),total_of(scope),Model.PieceContent{after_step(later,first,count,_),first,count})
    PiecesMap.piece_facts(~Element,~After,~after_value,after_of(scope),first,count,total_of(scope),input,after_step(later,first,count,input),left_exact,left_inside,made)

# Two trees of one layout holding the values of the two readers combine and
# step into pieces of the reader after the step.
def zipped_facts(~Element: Data,~Later: Data,~after_step: @+later: Later -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,
  ~operation: Element -> Element -> Element,~Left: Data,~left_value: Left -> U32 -> Element,~Right: Data,~right_value: Right -> U32 -> Element,
  ~After: Data,~after_value: After -> U32 -> Element,~Scope: Data,~later_of: @+scope: Scope -> Later,~left_of: @+scope: Scope -> Left,~right_of: @+scope: Scope -> Right,
  ~after_of: @+scope: Scope -> After,~total_of: @+scope: Scope -> Nat,
  ~rule: Rule(~Element,~Later,~after_step,~operation,~Left,~left_value,~Right,~right_value,~After,~after_value,~Scope,~later_of,~left_of,~right_of,~after_of,~total_of),
  +scope: Scope,left: Model.Model<Element>,right: Model.Model<Element>,common: Each.Common(~Element,left,right),
  left_fields: Content.Contents(~Element,total_of(scope),Model.content(~Element,left)),left_ready: Model.ContentReady(~Element,Model.content(~Element,left)),
  left_values: Content.Values(~Element,~Left,~left_value,left_of(scope),Model.content(~Element,left)),
  right_fields: Content.Contents(~Element,total_of(scope),Model.content(~Element,right)),right_ready: Model.ContentReady(~Element,Model.content(~Element,right)),
  right_values: Content.Values(~Element,~Right,~right_value,right_of(scope),Model.content(~Element,right))) ->
  PiecesMap.Facts(~Element,~After,~after_value,after_of(scope),total_of(scope),CombineLeaf.model(~Element,~Later,~operation,~after_step,later_of(scope),left,right)):
  match left right:
    case Model.Leaf{+left_buffer,+first,+rows} Model.Leaf{+right_buffer,+other_first,+other_rows}:
      KernelGemm.Both{+same_first,+same_rows} = common
      +total = total_of(scope)
      +right_content = Buffer.content(~Element,right_buffer)
      piece_facts(~Element,~Later,~after_step,~operation,~Left,~left_value,~Right,~right_value,~After,~after_value,~Scope,~later_of,~left_of,~right_of,~after_of,~total_of,~rule,
        scope,first,rows,left_buffer,right_buffer,left_fields,left_ready,left_values,
        %Equal.sym(U32,first,other_first,same_first) : Content.piece_fields(~Element,total,_,rows,right_content)
          %Equal.sym(U32,rows,other_rows,same_rows) : Content.piece_fields(~Element,total,other_first,_,right_content) right_fields,
        right_ready,
        %Equal.sym(U32,first,other_first,same_first) : Content.piece_values(~Element,~Right,~right_value,right_of(scope),_,rows,right_content)
          %Equal.sym(U32,rows,other_rows,same_rows) : Content.piece_values(~Element,~Right,~right_value,right_of(scope),other_first,_,right_content) right_values)
    case Model.Fork{+left_first,+left_second} Model.Fork{+right_first,+right_second}:
      KernelGemm.Both{+common_first,+common_second} = common
      KernelGemm.Both{+left_fields_first,+left_fields_second} = left_fields
      KernelGemm.Both{+left_ready_first,+left_ready_second} = left_ready
      KernelGemm.Both{+left_values_first,+left_values_second} = left_values
      KernelGemm.Both{+right_fields_first,+right_fields_second} = right_fields
      KernelGemm.Both{+right_ready_first,+right_ready_second} = right_ready
      KernelGemm.Both{+right_values_first,+right_values_second} = right_values
      +later = later_of(scope)
      PiecesMap.joined_facts(~Element,~After,~after_value,after_of(scope),total_of(scope),
        CombineLeaf.model(~Element,~Later,~operation,~after_step,later,left_first,right_first),CombineLeaf.model(~Element,~Later,~operation,~after_step,later,left_second,right_second),
        zipped_facts(~Element,~Later,~after_step,~operation,~Left,~left_value,~Right,~right_value,~After,~after_value,~Scope,~later_of,~left_of,~right_of,~after_of,~total_of,~rule,
          scope,left_first,right_first,common_first,left_fields_first,left_ready_first,left_values_first,right_fields_first,right_ready_first,right_values_first),
        zipped_facts(~Element,~Later,~after_step,~operation,~Left,~left_value,~Right,~right_value,~After,~after_value,~Scope,~later_of,~left_of,~right_of,~after_of,~total_of,~rule,
          scope,left_second,right_second,common_second,left_fields_second,left_ready_second,left_values_second,right_fields_second,right_ready_second,right_values_second))
    case Model.Leaf{+left_buffer,+first,+rows} Model.Fork{right_first,right_second}:
      Empty.absurd(PiecesMap.Facts(~Element,~After,~after_value,after_of(scope),total_of(scope),
        CombineLeaf.model(~Element,~Later,~operation,~after_step,later_of(scope),Model.Leaf{left_buffer,first,rows},Model.Fork{right_first,right_second})),common)
    case Model.Fork{+left_first,+left_second} Model.Leaf{right_buffer,other_first,other_rows}:
      Empty.absurd(PiecesMap.Facts(~Element,~After,~after_value,after_of(scope),total_of(scope),
        CombineLeaf.model(~Element,~Later,~operation,~after_step,later_of(scope),Model.Fork{left_first,left_second},Model.Leaf{right_buffer,other_first,other_rows})),common)

# The layout of a model is the layout of the contents of its pieces.
def packed_layout(~Element: Data,+model: Model.Model<Element>) ->
  {Model.layout(~Element,model) == Model.content_layout(~Element,Model.contents(~Element,Model.pack(~Element,model))) : Pieces.Ranges}:
  Equal.trans(Pieces.Ranges,Model.layout(~Element,model),Model.content_layout(~Element,Model.content(~Element,model)),
    Model.content_layout(~Element,Model.contents(~Element,Model.pack(~Element,model))),Model.layout_matches_content(~Element,model),
    Equal.cong(Model.Content<Element>,Pieces.Ranges,content => Model.content_layout(~Element,content),Model.content(~Element,model),
      Model.contents(~Element,Model.pack(~Element,model)),Equal.sym(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,model)),Model.content(~Element,model),
        Model.packed_contents(~Element,model))))

# Two models of canonical pieces of one shape have one layout.
def same_layout(~Element: Data,+shape: List<&2,U32>,+left: Model.Model<Element>,+right: Model.Model<Element>,
  +left_canonical: Global.Canonical(~Element,shape,Model.contents(~Element,Model.pack(~Element,left))),
  +right_canonical: Global.Canonical(~Element,shape,Model.contents(~Element,Model.pack(~Element,right)))) ->
  {Model.layout(~Element,left) == Model.layout(~Element,right) : Pieces.Ranges}:
  +canonical = Pieces.ranges_of_grid(Tensors.grid_of(shape))
  Equal.trans(Pieces.Ranges,Model.layout(~Element,left),canonical,Model.layout(~Element,right),
    Equal.trans(Pieces.Ranges,Model.layout(~Element,left),Model.content_layout(~Element,Model.contents(~Element,Model.pack(~Element,left))),canonical,
      packed_layout(~Element,left),left_canonical),
    Equal.sym(Pieces.Ranges,Model.layout(~Element,right),canonical,
      Equal.trans(Pieces.Ranges,Model.layout(~Element,right),Model.content_layout(~Element,Model.contents(~Element,Model.pack(~Element,right))),canonical,
        packed_layout(~Element,right),right_canonical)))

# Two tensors of one shape holding the values of two readers combine on the
# tasks of any decision into a tensor of the reader after the step.
def decided(~Element: Data,~Later: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),
  ~after: Tensors.PieceMap(~Element,~Later),~after_step: @+later: Later -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,
  ~after_rule: PiecesMap.Rule(~Element,~Later,~after,~after_step),~Left: Data,~left_value: Left -> U32 -> Element,~Right: Data,~right_value: Right -> U32 -> Element,
  ~After: Data,~after_value: After -> U32 -> Element,~Scope: Data,~later_of: @+scope: Scope -> Later,~left_of: @+scope: Scope -> Left,~right_of: @+scope: Scope -> Right,
  ~after_of: @+scope: Scope -> After,~total_of: @+scope: Scope -> Nat,
  ~rule: Rule(~Element,~Later,~after_step,~operation,~Left,~left_value,~Right,~right_value,~After,~after_value,~Scope,~later_of,~left_of,~right_of,~after_of,~total_of),
  +scope: Scope,decision: Tasks.Decision,+left: Model.Model<Element>,+right: Model.Model<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  +right_fill: Element,+right_workers: U32,+total_exact: {total_of(scope) == ShapeLaws.elements(shape) : Nat},
  left_contract: Global.Contract(~Element,~Left,~left_value,left_of(scope),Tensors.Tensor{Model.pack(~Element,left),shape,fill,Tensors.Usable{},workers}),
  right_contract: Global.Contract(~Element,~Right,~right_value,right_of(scope),Tensors.Tensor{Model.pack(~Element,right),shape,right_fill,Tensors.Usable{},right_workers})) ->
  Global.Contract(~Element,~After,~after_value,after_of(scope),
    Tensors.zipped_decided(~Element,~Later,~kernel,~after,later_of(scope),decision,shape,fill,workers,
      Pieces.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,left),Model.pack(~Element,right)))):
  match decision:
    case Tasks.Decision{+parallel,+tasks}:
      KernelGemm.Both{KernelGemm.Both{Content.WellFormed{+left_fields,+left_ready,+coverage},KernelGemm.Both{+counted,+left_canonical}},+left_values} = left_contract
      KernelGemm.Both{KernelGemm.Both{Content.WellFormed{+right_fields,+right_ready,right_coverage},KernelGemm.Both{right_counted,+right_canonical}},+right_values} = right_contract
      +later = later_of(scope)
      +total = ShapeLaws.elements(shape)
      +left_packed = Model.packed_contents(~Element,left)
      +right_packed = Model.packed_contents(~Element,right)
      +common = Each.matched(~Element,left,right,same_layout(~Element,shape,left,right,left_canonical,right_canonical))
      +output = CombineLeaf.model(~Element,~Later,~operation,~after_step,later,left,right)
      grouped = Pieces.grouped(~Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~Storage.Buffer<Element>,~Later,
        ~Pieces.paired(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Later,~Tensors.combine_buffers(~Element,~Later,~kernel,~after)),tasks,parallel,later,
        Pieces.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,left),Model.pack(~Element,right)))
      +output_layout = {Equal.trans(Pieces.Ranges,Model.content_layout(~Element,output),Model.layout(~Element,left),
          Model.content_layout(~Element,Model.contents(~Element,Model.pack(~Element,left))),
          CombineLeaf.model_layout(~Element,~Later,~operation,~after_step,later,left,right),packed_layout(~Element,left)) :
        {Model.content_layout(~Element,output) == Model.content_layout(~Element,Model.contents(~Element,Model.pack(~Element,left))) : Pieces.Ranges}}
      +total_back = Equal.sym(Nat,total_of(scope),total,total_exact)
      %Equal.sym(Model.Content<Element>,Model.contents(~Element,grouped),output,
        CombineLeaf.combine(~Element,~Later,~operation,~kernel,~kernel_law,~after,~after_step,~after_rule,later,tasks,parallel,left,right,common)) :
        PiecesMap.Holds(~Element,~After,~after_value,after_of(scope),shape,_)
      PiecesMap.holds(~Element,~After,~after_value,after_of(scope),shape,output,
        %total_exact : PiecesMap.Facts(~Element,~After,~after_value,after_of(scope),_,output)
          zipped_facts(~Element,~Later,~after_step,~operation,~Left,~left_value,~Right,~right_value,~After,~after_value,~Scope,~later_of,~left_of,~right_of,~after_of,~total_of,~rule,
            scope,left,right,common,
            %total_back : Content.Contents(~Element,_,Model.content(~Element,left)) %left_packed : Content.Contents(~Element,total,_) left_fields,
            %left_packed : Model.ContentReady(~Element,_) left_ready,
            %left_packed : Content.Values(~Element,~Left,~left_value,left_of(scope),_) left_values,
            %total_back : Content.Contents(~Element,_,Model.content(~Element,right)) %right_packed : Content.Contents(~Element,total,_) right_fields,
            %right_packed : Model.ContentReady(~Element,_) right_ready,
            %right_packed : Content.Values(~Element,~Right,~right_value,right_of(scope),_) right_values),
        %Equal.sym(Pieces.Ranges,Model.content_layout(~Element,output),Model.content_layout(~Element,Model.contents(~Element,Model.pack(~Element,left))),output_layout) :
          PiecesPlan.Coverage(_,0n,total) coverage,
        counted,
        %Equal.sym(Pieces.Ranges,Model.content_layout(~Element,output),Model.content_layout(~Element,Model.contents(~Element,Model.pack(~Element,left))),output_layout) :
          {_ == Pieces.ranges_of_grid(Tensors.grid_of(shape)) : Pieces.Ranges} left_canonical)
