# Combinations whose right operand has other extents. Every task reads the
# whole right operand through a view of it with the extents of the left:
# a piece combines in place in the left piece and gives the right operand back
# as it found it.
import Base
import ../default_machine_limits.bend as Default
import ./tensor_storage_count.bend as StorageCount
import ./tensor_count_refinement.bend as Count
import ./tensor_joined.bend as Joined
import ../tensor.bend as Tensors
import ../tensor_pieces.bend as Pieces
import ../tensor_grid.bend as Grid
import ../tensor_view.bend as Views
import ../tensor_shape.bend as Shapes
import ../tensor_operations.bend as Operations
import ../tensor_access.bend as Access
import ../storage_buffer.bend as Storage
import ../storage_clone.bend as Clone
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./storage_tree_model.bend as Tree
import ./storage_buffer_model.bend as Buffer
import ./storage_certificate.bend as Certificate
import ./storage_refinement.bend as StorageLaws
import ./storage_shape.bend as Shape
import ./storage_clone_refinement.bend as CloneLaws
import ./traversal_array_refinement.bend as ArrayLaws
import ./traversal_visit_refinement.bend as Visit
import ./tensor_visit_refinement.bend as TensorVisit
import ./tensor_cursor_refinement.bend as CursorLaws
import ./tensor_access_refinement.bend as AccessLaws
import ./tensor_operations_refinement.bend as OperationsLaws
import ./traversal_partition_refinement.bend as PartitionLaws
import ./tensor_pieces_model.bend as Model
import ./tensor_pieces_each.bend as Each
import ./tensor_pieces_map.bend as PiecesMap
import ./tensor_pieces_combine_leaf.bend as CombineLeaf
import ./tensor_pieces_combine.bend as PiecesCombine
import ./tensor_grid_ranges.bend as GridRanges
import ./tensor_move_readback.bend as Held
import ../tensor_tasks.bend as Tasks
import ./tensor_pieces_plan.bend as PiecesPlan
import ./tensor_global_shape_count.bend as Global
import ./tensor_pieces_content.bend as Content
import ./tensor_contract.bend as Contract
import ./tensor_copy_values.bend as CopyValues
import ./tensor_combine_values.bend as CombineValues
import ./tensor_view_gather.bend as ViewGather
import ./tensor_view_reads.bend as ViewReads
import ./tensor_view_part.bend as ViewPart
import ./tensor_target_cells.bend as Targets
import ./tensor_shape_refinement.bend as ShapeLaws
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as LimitBounds
import ./word_power_of_two.bend as Power
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_full_range.bend as FullRange
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./sequence_membership.bend as Members
import ./boolean_logic.bend as Logic
import ./tensor_grid_box.bend as Box
import ./natural_addition.bend as Addition
import ../kernel_gemm.bend as KernelGemm

# ---- The right operand comes back as it was ----

def reader_of(~Element: Data,result: AccessLaws.ModelResult<Tree.Tree<Element>,Element>) -> Tree.Tree<Element>:
  AccessLaws.ModelResult{reader,output} = result
  reader

def indexed_reader(~Element: Data,~operation: Element -> Element -> Element,count: Nat,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+position: U32) ->
  {reader_of(~Element,AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~Unit,Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Unit>,
      ~ArrayLaws.model_update_step(~Tree.Tree<Element>,~Element,~Element,~Unit,~AccessLaws.sample_index(~Element),~operation,~Traversal.index),count,
      Traversal.State{input,output,position,Unit{}}))) == input : Tree.Tree<Element>}:
  match count:
    case 0n: {==}
    case 1n+rest:
      indexed_reader(~Element,~operation,rest,input,
        Tree.write(Element,output,position,operation(Tree.read(Element,output,position),AccessLaws.sample_index(~Element,input,Unit{},position))),(position + 1 : U32))

def visited_reader(~Element: Data,~operation: Element -> Element -> Element,count: Nat,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+position: U32,
  +coordinates: CursorLaws.Coordinates) ->
  {reader_of(~Element,AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~CursorLaws.Coordinates,Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,CursorLaws.Coordinates>,
      ~Visit.model_update_step(~Tree.Tree<Element>,~Element,~CursorLaws.Coordinates,~TensorVisit.sample(~Element),~CursorLaws.next,~operation),count,
      Traversal.State{input,output,position,coordinates}))) == input : Tree.Tree<Element>}:
  match count:
    case 0n: {==}
    case 1n+rest:
      visited_reader(~Element,~operation,rest,input,
        Tree.write(Element,output,position,operation(Tree.read(Element,output,position),TensorVisit.sample(~Element,input,coordinates))),(position + 1 : U32),
        CursorLaws.next(coordinates))

# The checked combination reads its right operand and keeps it.
def combine_reader(~Element: Data,~operation: Element -> Element -> Element,fast: Bool,+count: Nat,+source: Views.View,+destination: U32,+input: Tree.Tree<Element>,
  +output: Tree.Tree<Element>) ->
  {reader_of(~Element,AccessLaws.combine_model(~Element,~operation,fast,count,source,destination,input,output)) == input : Tree.Tree<Element>}:
  match fast:
    case True{}: indexed_reader(~Element,~operation,count,input,output,0)
    case False{}: visited_reader(~Element,~operation,count,input,output,destination,CursorLaws.initial(source))

def right_content(~Element: Data,result: Operations.Outcome<OperationsLaws.PairOwners(Element)>) -> Buffer.Content<Element>:
  match result:
    case Operations.Accepted{(left,OperationsLaws.Owner{storage,length})}: Buffer.Content{Tree.reflect(Element,storage),length}
    case Operations.Rejected{(left,OperationsLaws.Owner{storage,length})}: Buffer.Content{Tree.reflect(Element,storage),length}

def accepted_right(~Element: Data,+left_length: Nat,+right_length: Nat,result: AccessLaws.ModelResult<Tree.Tree<Element>,Element>) ->
  {right_content(~Element,Operations.Accepted{OperationsLaws.combined_pair(~Element,left_length,right_length,result)}) ==
    Buffer.Content{Tree.reflect(Element,Tree.pack(Element,reader_of(~Element,result))),right_length} : Buffer.Content<Element>}:
  AccessLaws.ModelResult{reader,output} = result
  {==}

def called_right(~Element: Data,~operation: Element -> Element -> Element,valid: Bool,count: Maybe<&2,U32>,+source: Views.View,+axes: List<&2,Views.Axis>,+destination: U32,
  +right_length: Nat,+left_length: Nat,+right_tree: Tree.Tree<Element>,+left_tree: Tree.Tree<Element>) ->
  {right_content(~Element,OperationsLaws.combine_model(~Element,~operation,OperationsLaws.PairCall{valid,count,source,Views.View{axes,destination},right_length,left_length},right_tree,left_tree)) ==
    Buffer.Content{right_tree,right_length} : Buffer.Content<Element>}:
  match valid count:
    case True{} Some{+count}:
      +fast = Access.zero_contiguous(source) && U32.is_zero(destination)
      +result = AccessLaws.combine_model(~Element,~operation,fast,U32.to_nat(count),source,destination,right_tree,left_tree)
      Equal.trans(Buffer.Content<Element>,right_content(~Element,Operations.Accepted{OperationsLaws.combined_pair(~Element,left_length,right_length,result)}),
        Buffer.Content{Tree.reflect(Element,Tree.pack(Element,reader_of(~Element,result))),right_length},Buffer.Content{right_tree,right_length},
        accepted_right(~Element,left_length,right_length,result),
        %Equal.sym(Tree.Tree<Element>,reader_of(~Element,result),right_tree,combine_reader(~Element,~operation,fast,U32.to_nat(count),source,destination,right_tree,left_tree)) :
          {Buffer.Content{Tree.reflect(Element,Tree.pack(Element,_)),right_length} == Buffer.Content{right_tree,right_length} : Buffer.Content<Element>}
        %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,right_tree)),right_tree,StorageLaws.reflect_packed(Element,right_tree)) :
          {Buffer.Content{_,right_length} == Buffer.Content{right_tree,right_length} : Buffer.Content<Element>}
        {==})
    case True{} None{}:
      %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,right_tree)),right_tree,StorageLaws.reflect_packed(Element,right_tree)) :
        {Buffer.Content{_,right_length} == Buffer.Content{right_tree,right_length} : Buffer.Content<Element>}
      {==}
    case False{} count:
      %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,right_tree)),right_tree,StorageLaws.reflect_packed(Element,right_tree)) :
        {Buffer.Content{_,right_length} == Buffer.Content{right_tree,right_length} : Buffer.Content<Element>}
      {==}

# ---- One piece ----

# The checked combination of the contents of two buffers.
def denoted(~Element: Data,~operation: Element -> Element -> Element,+left_view: Views.View,+right_view: Views.View,left: Buffer.Content<Element>,right: Buffer.Content<Element>) -> Operations.Outcome<OperationsLaws.PairOwners(Element)>:
  Buffer.Content{+left_tree,+left_length} = left
  Buffer.Content{+right_tree,+right_length} = right
  OperationsLaws.combine_model(~Element,~operation,
    OperationsLaws.PairCall{Shapes.same(Views.shape(left_view),Views.shape(right_view)) && Views.is_contiguous(left_view) && Operations.valid_buffer(left_length,Tree.capacity(Element,left_tree),left_view) && Operations.valid_buffer(right_length,Tree.capacity(Element,right_tree),right_view),Views.numel(left_view),right_view,left_view,right_length,left_length},right_tree,left_tree)

def denoted_packed(~Element: Data,~operation: Element -> Element -> Element,+left: Buffer.Model<Element>,+left_view: Views.View,+right: Buffer.Model<Element>,+right_view: Views.View) ->
  {OperationsLaws.combine_denotation(~Element,~operation,Buffer.pack(~Element,left),left_view,Buffer.pack(~Element,right),right_view) ==
    denoted(~Element,~operation,left_view,right_view,Buffer.content(~Element,left),Buffer.content(~Element,right)) : Operations.Outcome<OperationsLaws.PairOwners(Element)>}:
  Buffer.Model{+left_tree,+left_length,Certificate.Certificate{+left_depth,+left_equation}} = left
  Buffer.Model{+right_tree,+right_length,Certificate.Certificate{+right_depth,+right_equation}} = right
  +left_model = {Buffer.Model{left_tree,left_length,Certificate.Certificate{left_depth,left_equation}} : Buffer.Model<Element>}
  +right_model = {Buffer.Model{right_tree,right_length,Certificate.Certificate{right_depth,right_equation}} : Buffer.Model<Element>}
  +model_call = {OperationsLaws.PairCall{Shapes.same(Views.shape(left_view),Views.shape(right_view)) && Views.is_contiguous(left_view) && Operations.valid_buffer(left_length,Tree.capacity(Element,left_tree),left_view) && Operations.valid_buffer(right_length,Tree.capacity(Element,right_tree),right_view),Views.numel(left_view),right_view,left_view,right_length,left_length} : OperationsLaws.PairCall}
  %Shape.capacity(Element,left_depth,left_tree,Buffer.balanced(~Element,left_model)) :
    {OperationsLaws.combine_model(~Element,~operation,OperationsLaws.PairCall{Shapes.same(Views.shape(left_view),Views.shape(right_view)) && Views.is_contiguous(left_view) && Operations.valid_buffer(left_length,_,left_view) && Operations.valid_buffer(right_length,U32.shln(1,right_depth),right_view),Views.numel(left_view),right_view,left_view,right_length,left_length},
      Tree.reflect(Element,Tree.pack(Element,right_tree)),Tree.reflect(Element,Tree.pack(Element,left_tree))) ==
      OperationsLaws.combine_model(~Element,~operation,model_call,right_tree,left_tree) : Operations.Outcome<OperationsLaws.PairOwners(Element)>}
  %Shape.capacity(Element,right_depth,right_tree,Buffer.balanced(~Element,right_model)) :
    {OperationsLaws.combine_model(~Element,~operation,OperationsLaws.PairCall{Shapes.same(Views.shape(left_view),Views.shape(right_view)) && Views.is_contiguous(left_view) && Operations.valid_buffer(left_length,Tree.capacity(Element,left_tree),left_view) && Operations.valid_buffer(right_length,_,right_view),Views.numel(left_view),right_view,left_view,right_length,left_length},
      Tree.reflect(Element,Tree.pack(Element,right_tree)),Tree.reflect(Element,Tree.pack(Element,left_tree))) ==
      OperationsLaws.combine_model(~Element,~operation,model_call,right_tree,left_tree) : Operations.Outcome<OperationsLaws.PairOwners(Element)>}
  %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,right_tree)),right_tree,StorageLaws.reflect_packed(Element,right_tree)) :
    {OperationsLaws.combine_model(~Element,~operation,model_call,_,Tree.reflect(Element,Tree.pack(Element,left_tree))) ==
      OperationsLaws.combine_model(~Element,~operation,model_call,right_tree,left_tree) : Operations.Outcome<OperationsLaws.PairOwners(Element)>}
  %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,left_tree)),left_tree,StorageLaws.reflect_packed(Element,left_tree)) :
    {OperationsLaws.combine_model(~Element,~operation,model_call,right_tree,_) ==
      OperationsLaws.combine_model(~Element,~operation,model_call,right_tree,left_tree) : Operations.Outcome<OperationsLaws.PairOwners(Element)>}
  {==}

# The combination gives the right contents back.
def denoted_right(~Element: Data,~operation: Element -> Element -> Element,+left_axes: List<&2,Views.Axis>,+left_offset: U32,+right_view: Views.View,
  +left: Buffer.Content<Element>,+right: Buffer.Content<Element>) ->
  {right_content(~Element,denoted(~Element,~operation,Views.View{left_axes,left_offset},right_view,left,right)) == right : Buffer.Content<Element>}:
  Buffer.Content{+left_tree,+left_length} = left
  Buffer.Content{+right_tree,+right_length} = right
  called_right(~Element,~operation,Shapes.same(Views.shape(Views.View{left_axes,left_offset}),Views.shape(right_view)) && Views.is_contiguous(Views.View{left_axes,left_offset}) && Operations.valid_buffer(left_length,Tree.capacity(Element,left_tree),Views.View{left_axes,left_offset}) && Operations.valid_buffer(right_length,Tree.capacity(Element,right_tree),right_view),
    Views.numel(Views.View{left_axes,left_offset}),right_view,left_axes,left_offset,right_length,left_length,right_tree,left_tree)

def released(~Element: Data,result: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) -> Storage.Buffer<Element>:
  match result:
    case Operations.Accepted{(left,right)}: right
    case Operations.Rejected{(left,right)}: right

def released_content(~Element: Data,result: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) ->
  {Buffer.contents(~Element,released(~Element,result)) == right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,result)) : Buffer.Content<Element>}:
  match result:
    case Operations.Accepted{(left,Storage.Buffer{storage,length,certificate})}: {==}
    case Operations.Rejected{(left,Storage.Buffer{storage,length,certificate})}: {==}

def outcome_seen(~Element: Data,~Later: Data,~after: Tensors.PieceMap(~Element,~Later),+later: Later,+first: U32,+rows: U32,result: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) ->
  {Each.walk_seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),Tensors.spread_outcome(~Element,~Later,~after,later,first,rows,result)) ==
    (Model.PieceContent{Buffer.contents(~Element,after(later,first,rows,Tensors.combined(~Element,result))),first,rows},Buffer.contents(~Element,released(~Element,result))) : Model.Content<Element> & Buffer.Content<Element>}:
  match result:
    case Operations.Accepted{(left,right)}: {==}
    case Operations.Rejected{(left,right)}: {==}

# What a piece of a spread holds: the left piece combined with the right read
# through the piece of its view, then the step after.
def spread_model(~Element: Data,~Later: Data,~operation: Element -> Element -> Element,
  ~after_step: @+later: Later -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,
  context: Tensors.Spread<Later>,right: Buffer.Content<Element>,+first: U32,+rows: U32,left: Buffer.Content<Element>) -> Buffer.Content<Element>:
  Tensors.Spread{+shape,+axes,+offset,+later} = context
  after_step(later,first,rows,CombineLeaf.left_content(~Element,denoted(~Element,~operation,Grid.box_view(Tensors.grid_of(shape),shape,rows),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset}),left,right)))

# A piece of a spread holds its model and gives the right operand back.
def spread_leaf(~Element: Data,~Later: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),
  ~after: Tensors.PieceMap(~Element,~Later),~after_step: @+later: Later -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,
  ~after_rule: PiecesMap.Rule(~Element,~Later,~after,~after_step),
  +context: Tensors.Spread<Later>,+first: U32,+rows: U32,+buffer: Buffer.Model<Element>,+kept: Buffer.Model<Element>) ->
  {Each.walk_seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),Tensors.spread_piece(~Element,~Later,~kernel,~after,context,first,rows,Buffer.pack(~Element,buffer),Buffer.pack(~Element,kept))) ==
    (Model.PieceContent{spread_model(~Element,~Later,~operation,~after_step,context,Buffer.contents(~Element,Buffer.pack(~Element,kept)),first,rows,Buffer.content(~Element,buffer)),first,rows},Buffer.contents(~Element,Buffer.pack(~Element,kept))) : Model.Content<Element> & Buffer.Content<Element>}:
  Tensors.Spread{+shape,+axes,+offset,+later} = context
  +packed = Buffer.packed_contents(~Element,kept)
  Equal.trans(Model.Content<Element> & Buffer.Content<Element>,
    Each.walk_seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),Tensors.spread_outcome(~Element,~Later,~after,later,first,rows,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,buffer),Grid.box_view(Tensors.grid_of(shape),shape,rows),Buffer.pack(~Element,kept),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset})))),
    (Model.PieceContent{Buffer.contents(~Element,after(later,first,rows,Tensors.combined(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,buffer),Grid.box_view(Tensors.grid_of(shape),shape,rows),Buffer.pack(~Element,kept),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset}))))),first,rows},Buffer.contents(~Element,released(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,buffer),Grid.box_view(Tensors.grid_of(shape),shape,rows),Buffer.pack(~Element,kept),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset}))))),
    (Model.PieceContent{after_step(later,first,rows,CombineLeaf.left_content(~Element,denoted(~Element,~operation,Grid.box_view(Tensors.grid_of(shape),shape,rows),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset}),Buffer.content(~Element,buffer),Buffer.contents(~Element,Buffer.pack(~Element,kept))))),first,rows},Buffer.contents(~Element,Buffer.pack(~Element,kept))),
    outcome_seen(~Element,~Later,~after,later,first,rows,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,buffer),Grid.box_view(Tensors.grid_of(shape),shape,rows),Buffer.pack(~Element,kept),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset}))),
    %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,after(later,first,rows,Tensors.combined(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,buffer),Grid.box_view(Tensors.grid_of(shape),shape,rows),Buffer.pack(~Element,kept),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset}))))),
        after_step(later,first,rows,Buffer.contents(~Element,Tensors.combined(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,buffer),Grid.box_view(Tensors.grid_of(shape),shape,rows),Buffer.pack(~Element,kept),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset}))))),after_rule(later,first,rows,Tensors.combined(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,buffer),Grid.box_view(Tensors.grid_of(shape),shape,rows),Buffer.pack(~Element,kept),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset}))))) :
      {(Model.PieceContent{_,first,rows},Buffer.contents(~Element,released(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,buffer),Grid.box_view(Tensors.grid_of(shape),shape,rows),Buffer.pack(~Element,kept),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset}))))) == (Model.PieceContent{after_step(later,first,rows,CombineLeaf.left_content(~Element,denoted(~Element,~operation,Grid.box_view(Tensors.grid_of(shape),shape,rows),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset}),Buffer.content(~Element,buffer),Buffer.contents(~Element,Buffer.pack(~Element,kept))))),first,rows},Buffer.contents(~Element,Buffer.pack(~Element,kept))) : Model.Content<Element> & Buffer.Content<Element>}
    %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Tensors.combined(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,buffer),Grid.box_view(Tensors.grid_of(shape),shape,rows),Buffer.pack(~Element,kept),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset})))),CombineLeaf.left_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,buffer),Grid.box_view(Tensors.grid_of(shape),shape,rows),Buffer.pack(~Element,kept),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset})))),
        CombineLeaf.selected_content(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,buffer),Grid.box_view(Tensors.grid_of(shape),shape,rows),Buffer.pack(~Element,kept),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset})))) :
      {(Model.PieceContent{after_step(later,first,rows,_),first,rows},Buffer.contents(~Element,released(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,buffer),Grid.box_view(Tensors.grid_of(shape),shape,rows),Buffer.pack(~Element,kept),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset}))))) == (Model.PieceContent{after_step(later,first,rows,CombineLeaf.left_content(~Element,denoted(~Element,~operation,Grid.box_view(Tensors.grid_of(shape),shape,rows),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset}),Buffer.content(~Element,buffer),Buffer.contents(~Element,Buffer.pack(~Element,kept))))),first,rows},Buffer.contents(~Element,Buffer.pack(~Element,kept))) : Model.Content<Element> & Buffer.Content<Element>}
    %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,released(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,buffer),Grid.box_view(Tensors.grid_of(shape),shape,rows),Buffer.pack(~Element,kept),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset})))),right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,buffer),Grid.box_view(Tensors.grid_of(shape),shape,rows),Buffer.pack(~Element,kept),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset})))),released_content(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,buffer),Grid.box_view(Tensors.grid_of(shape),shape,rows),Buffer.pack(~Element,kept),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset})))) :
      {(Model.PieceContent{after_step(later,first,rows,CombineLeaf.left_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,buffer),Grid.box_view(Tensors.grid_of(shape),shape,rows),Buffer.pack(~Element,kept),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset}))))),first,rows},_) ==
        (Model.PieceContent{after_step(later,first,rows,CombineLeaf.left_content(~Element,denoted(~Element,~operation,Grid.box_view(Tensors.grid_of(shape),shape,rows),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset}),Buffer.content(~Element,buffer),Buffer.contents(~Element,Buffer.pack(~Element,kept))))),first,rows},Buffer.contents(~Element,Buffer.pack(~Element,kept))) : Model.Content<Element> & Buffer.Content<Element>}
    %Equal.sym(Operations.Outcome<OperationsLaws.PairOwners(Element)>,OperationsLaws.observe_pair_outcome(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,buffer),Grid.box_view(Tensors.grid_of(shape),shape,rows),Buffer.pack(~Element,kept),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset}))),OperationsLaws.combine_denotation(~Element,~operation,Buffer.pack(~Element,buffer),Grid.box_view(Tensors.grid_of(shape),shape,rows),Buffer.pack(~Element,kept),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset})),
        OperationsLaws.combine(~Element,~operation,~kernel,~kernel_law,Buffer.pack(~Element,buffer),Grid.box_view(Tensors.grid_of(shape),shape,rows),Buffer.pack(~Element,kept),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset}))) :
      {(Model.PieceContent{after_step(later,first,rows,CombineLeaf.left_content(~Element,_)),first,rows},right_content(~Element,_)) ==
        (Model.PieceContent{after_step(later,first,rows,CombineLeaf.left_content(~Element,denoted(~Element,~operation,Grid.box_view(Tensors.grid_of(shape),shape,rows),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset}),Buffer.content(~Element,buffer),Buffer.contents(~Element,Buffer.pack(~Element,kept))))),first,rows},Buffer.contents(~Element,Buffer.pack(~Element,kept))) : Model.Content<Element> & Buffer.Content<Element>}
    %Equal.sym(Operations.Outcome<OperationsLaws.PairOwners(Element)>,OperationsLaws.combine_denotation(~Element,~operation,Buffer.pack(~Element,buffer),Grid.box_view(Tensors.grid_of(shape),shape,rows),Buffer.pack(~Element,kept),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset})),denoted(~Element,~operation,Grid.box_view(Tensors.grid_of(shape),shape,rows),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset}),Buffer.content(~Element,buffer),Buffer.content(~Element,kept)),denoted_packed(~Element,~operation,buffer,Grid.box_view(Tensors.grid_of(shape),shape,rows),kept,Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset}))) :
      {(Model.PieceContent{after_step(later,first,rows,CombineLeaf.left_content(~Element,_)),first,rows},right_content(~Element,_)) ==
        (Model.PieceContent{after_step(later,first,rows,CombineLeaf.left_content(~Element,denoted(~Element,~operation,Grid.box_view(Tensors.grid_of(shape),shape,rows),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset}),Buffer.content(~Element,buffer),Buffer.contents(~Element,Buffer.pack(~Element,kept))))),first,rows},Buffer.contents(~Element,Buffer.pack(~Element,kept))) : Model.Content<Element> & Buffer.Content<Element>}
    %Equal.sym(Buffer.Content<Element>,right_content(~Element,denoted(~Element,~operation,Grid.box_view(Tensors.grid_of(shape),shape,rows),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset}),Buffer.content(~Element,buffer),Buffer.content(~Element,kept))),Buffer.content(~Element,kept),
        denoted_right(~Element,~operation,Grid.dense_axes(Grid.box_of(Tensors.grid_of(shape),shape,rows)),0,Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset}),Buffer.content(~Element,buffer),Buffer.content(~Element,kept))) :
      {(Model.PieceContent{after_step(later,first,rows,CombineLeaf.left_content(~Element,denoted(~Element,~operation,Grid.box_view(Tensors.grid_of(shape),shape,rows),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset}),Buffer.content(~Element,buffer),Buffer.content(~Element,kept)))),first,rows},_) ==
        (Model.PieceContent{after_step(later,first,rows,CombineLeaf.left_content(~Element,denoted(~Element,~operation,Grid.box_view(Tensors.grid_of(shape),shape,rows),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset}),Buffer.content(~Element,buffer),Buffer.contents(~Element,Buffer.pack(~Element,kept))))),first,rows},Buffer.contents(~Element,Buffer.pack(~Element,kept))) : Model.Content<Element> & Buffer.Content<Element>}
    %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,kept)),Buffer.content(~Element,kept),packed) :
      {(Model.PieceContent{after_step(later,first,rows,CombineLeaf.left_content(~Element,denoted(~Element,~operation,Grid.box_view(Tensors.grid_of(shape),shape,rows),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset}),Buffer.content(~Element,buffer),Buffer.content(~Element,kept)))),first,rows},Buffer.content(~Element,kept)) == (Model.PieceContent{after_step(later,first,rows,CombineLeaf.left_content(~Element,denoted(~Element,~operation,Grid.box_view(Tensors.grid_of(shape),shape,rows),Grid.piece_view(Tensors.grid_of(shape),shape,first,rows,Views.View{axes,offset}),Buffer.content(~Element,buffer),_))),first,rows},_) : Model.Content<Element> & Buffer.Content<Element>}
    {==})

# ---- Every task holds its own right operand ----

def split_seen(~Element: Data,+model: Buffer.Model<Element>,observation: CloneLaws.Observation<Element,Clone.clone(~Element,Buffer.pack(~Element,model))>) ->
  PartitionLaws.Split<Buffer.Model<Element>>:
  CloneLaws.Observation{+left,+right,equation} = observation
  PartitionLaws.Split{left,right}

def split_buffer(~Element: Data,model: Buffer.Model<Element>) -> PartitionLaws.Split<Buffer.Model<Element>>:
  Buffer.Model{+tree,+length,+certificate} = model
  +model = {Buffer.Model{tree,length,certificate} : Buffer.Model<Element>}
  split_seen(~Element,model,CloneLaws.observe_pair(~Element,Clone.clone(~Element,Buffer.pack(~Element,model))))

def cloned_halves(~Element: Data,+model: Buffer.Model<Element>,observation: CloneLaws.Observation<Element,Clone.clone(~Element,Buffer.pack(~Element,model))>) ->
  {Clone.clone(~Element,Buffer.pack(~Element,model)) ==
    (Buffer.pack(~Element,PartitionLaws.first(~Buffer.Model<Element>,split_seen(~Element,model,observation))),Buffer.pack(~Element,PartitionLaws.second(~Buffer.Model<Element>,split_seen(~Element,model,observation)))) : Storage.Buffer<Element> & Storage.Buffer<Element>}:
  CloneLaws.Observation{+left,+right,+equation} = observation
  Equal.sym(Storage.Buffer<Element> & Storage.Buffer<Element>,(Buffer.pack(~Element,left),Buffer.pack(~Element,right)),Clone.clone(~Element,Buffer.pack(~Element,model)),equation)

def first_cloned(~Element: Data,+model: Buffer.Model<Element>,observation: CloneLaws.Observation<Element,Clone.clone(~Element,Buffer.pack(~Element,model))>) ->
  {Buffer.contents(~Element,Buffer.pack(~Element,PartitionLaws.first(~Buffer.Model<Element>,split_seen(~Element,model,observation)))) == Buffer.contents(~Element,Buffer.pack(~Element,model)) : Buffer.Content<Element>}:
  CloneLaws.Observation{+left,+right,+equation} = observation
  Equal.trans(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,left)),Buffer.content(~Element,left),Buffer.contents(~Element,Buffer.pack(~Element,model)),
    Buffer.packed_contents(~Element,left),
    Equal.trans(Buffer.Content<Element>,Buffer.content(~Element,left),Buffer.content(~Element,model),Buffer.contents(~Element,Buffer.pack(~Element,model)),
      CloneLaws.left_contents(~Element,model,left,right,equation),
      Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,model)),Buffer.content(~Element,model),Buffer.packed_contents(~Element,model))))

def second_cloned(~Element: Data,+model: Buffer.Model<Element>,observation: CloneLaws.Observation<Element,Clone.clone(~Element,Buffer.pack(~Element,model))>) ->
  {Buffer.contents(~Element,Buffer.pack(~Element,PartitionLaws.second(~Buffer.Model<Element>,split_seen(~Element,model,observation)))) == Buffer.contents(~Element,Buffer.pack(~Element,model)) : Buffer.Content<Element>}:
  CloneLaws.Observation{+left,+right,+equation} = observation
  Equal.trans(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,right)),Buffer.content(~Element,right),Buffer.contents(~Element,Buffer.pack(~Element,model)),
    Buffer.packed_contents(~Element,right),
    Equal.trans(Buffer.Content<Element>,Buffer.content(~Element,right),Buffer.content(~Element,model),Buffer.contents(~Element,Buffer.pack(~Element,model)),
      CloneLaws.right_contents(~Element,model,left,right,equation),
      Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,model)),Buffer.content(~Element,model),Buffer.packed_contents(~Element,model))))

# A task beyond the first clones the right operand.
def clone_shared(~Element: Data,+kept: Buffer.Model<Element>) ->
  {Tensors.shared_buffer(~Element,Buffer.pack(~Element,kept)) ==
    (Buffer.pack(~Element,PartitionLaws.first(~Buffer.Model<Element>,split_buffer(~Element,kept))),Buffer.pack(~Element,PartitionLaws.second(~Buffer.Model<Element>,split_buffer(~Element,kept)))) : Storage.Buffer<Element> & Storage.Buffer<Element>}:
  Buffer.Model{+tree,+length,+certificate} = kept
  cloned_halves(~Element,Buffer.Model{tree,length,certificate},CloneLaws.observe_pair(~Element,Clone.clone(~Element,Buffer.pack(~Element,Buffer.Model{tree,length,certificate}))))

def first_shared(~Element: Data,+kept: Buffer.Model<Element>) ->
  {Buffer.contents(~Element,Buffer.pack(~Element,PartitionLaws.first(~Buffer.Model<Element>,split_buffer(~Element,kept)))) == Buffer.contents(~Element,Buffer.pack(~Element,kept)) : Buffer.Content<Element>}:
  Buffer.Model{+tree,+length,+certificate} = kept
  first_cloned(~Element,Buffer.Model{tree,length,certificate},CloneLaws.observe_pair(~Element,Clone.clone(~Element,Buffer.pack(~Element,Buffer.Model{tree,length,certificate}))))

def second_shared(~Element: Data,+kept: Buffer.Model<Element>) ->
  {Buffer.contents(~Element,Buffer.pack(~Element,PartitionLaws.second(~Buffer.Model<Element>,split_buffer(~Element,kept)))) == Buffer.contents(~Element,Buffer.pack(~Element,kept)) : Buffer.Content<Element>}:
  Buffer.Model{+tree,+length,+certificate} = kept
  second_cloned(~Element,Buffer.Model{tree,length,certificate},CloneLaws.observe_pair(~Element,Clone.clone(~Element,Buffer.pack(~Element,Buffer.Model{tree,length,certificate}))))

# The tasks of any plan map every left piece with the right contents.
def spread_walked(~Element: Data,~Later: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),
  ~after: Tensors.PieceMap(~Element,~Later),~after_step: @+later: Later -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,
  ~after_rule: PiecesMap.Rule(~Element,~Later,~after,~after_step),
  layout: Pieces.Ranges,+enabled: Bool,+context: Tensors.Spread<Later>,+tree: Model.Model<Element>,+kept: Buffer.Model<Element>) ->
  {Model.contents(~Element,Pieces.grouped_with(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Tensors.Spread<Later>,~Tensors.shared_buffer(~Element),
      ~Tensors.spread_piece(~Element,~Later,~kernel,~after),layout,enabled,context,Model.pack(~Element,tree),Buffer.pack(~Element,kept))) ==
    Model.map(~Element,~spread_model(~Element,~Later,~operation,~after_step,context,Buffer.contents(~Element,Buffer.pack(~Element,kept))),Model.content(~Element,tree)) : Model.Content<Element>}:
  Each.grouped_walked(~Element,~Storage.Buffer<Element>,~Tensors.Spread<Later>,~Buffer.Model<Element>,~Buffer.pack(~Element),~Each.buffer_held(~Element),~Buffer.Content<Element>,~Buffer.contents(~Element),
    ~Tensors.spread_piece(~Element,~Later,~kernel,~after),~spread_model(~Element,~Later,~operation,~after_step),~spread_leaf(~Element,~Later,~operation,~kernel,~kernel_law,~after,~after_step,~after_rule),
    ~Tensors.shared_buffer(~Element),~split_buffer(~Element),~clone_shared(~Element),~first_shared(~Element),~second_shared(~Element),layout,enabled,enabled,context,tree,kept)

# ---- The values of a piece ----

# Ready storage stays within the largest capacity.
def content_capacity(~Element: Data,+tree: Tree.Tree<Element>,+length: Nat,ready: Buffer.ContentReady<Element,Buffer.Content{tree,length}>) ->
  {U32.is_le(Tree.capacity(Element,tree),2147483648) == True{} : Bool}:
  Buffer.ContentReady{+depth,+balanced,+depth_ok,room} = ready
  %Equal.sym(U32,Tree.capacity(Element,tree),U32.shln(1,depth),Shape.capacity(Element,depth,tree,balanced)) : {U32.is_le(_,2147483648) == True{} : Bool}
  Power.native_power_bound(depth,Order.successor(depth,30n,depth_ok))

def content_room(~Element: Data,+tree: Tree.Tree<Element>,+length: Nat,ready: Buffer.ContentReady<Element,Buffer.Content{tree,length}>) ->
  {Nat.is_le(length,U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool}:
  Buffer.ContentReady{depth,balanced,depth_ok,+room} = ready
  room

def with_true(+value: Bool) -> {value == (value && True{}) : Bool}:
  match value:
    case True{}: {==}
    case False{}: {==}

# A combination whose guard holds writes the kernel into the left contents.
def guarded_left(~Element: Data,~operation: Element -> Element -> Element,+left_axes: List<&2,Views.Axis>,+right_view: Views.View,+count: U32,
  +left_tree: Tree.Tree<Element>,+left_length: Nat,+right_tree: Tree.Tree<Element>,+right_length: Nat,
  +guarded: {Shapes.same(Views.shape(Views.View{left_axes,0}),Views.shape(right_view)) && Views.is_contiguous(Views.View{left_axes,0}) &&
    Operations.valid_buffer(left_length,Tree.capacity(Element,left_tree),Views.View{left_axes,0}) && Operations.valid_buffer(right_length,Tree.capacity(Element,right_tree),right_view) == True{} : Bool},
  +counted: {Views.numel(Views.View{left_axes,0}) == Some{count} : Maybe<&2,U32>}) ->
  {CombineLeaf.left_content(~Element,denoted(~Element,~operation,Views.View{left_axes,0},right_view,Buffer.Content{left_tree,left_length},Buffer.Content{right_tree,right_length})) ==
    Buffer.Content{CombineLeaf.readback_output(~Element,AccessLaws.combine_model(~Element,~operation,Access.zero_contiguous(right_view) && U32.is_zero(0),U32.to_nat(count),right_view,0,right_tree,left_tree)),left_length} : Buffer.Content<Element>}:
  +result = AccessLaws.combine_model(~Element,~operation,Access.zero_contiguous(right_view) && U32.is_zero(0),U32.to_nat(count),right_view,0,right_tree,left_tree)
  %Equal.sym(Bool,Shapes.same(Views.shape(Views.View{left_axes,0}),Views.shape(right_view)) && Views.is_contiguous(Views.View{left_axes,0}) &&
      Operations.valid_buffer(left_length,Tree.capacity(Element,left_tree),Views.View{left_axes,0}) && Operations.valid_buffer(right_length,Tree.capacity(Element,right_tree),right_view),True{},guarded) :
    {CombineLeaf.left_content(~Element,OperationsLaws.combine_model(~Element,~operation,OperationsLaws.PairCall{_,Views.numel(Views.View{left_axes,0}),right_view,Views.View{left_axes,0},right_length,left_length},right_tree,left_tree)) ==
      Buffer.Content{CombineLeaf.readback_output(~Element,result),left_length} : Buffer.Content<Element>}
  %Equal.sym(Maybe<&2,U32>,Views.numel(Views.View{left_axes,0}),Some{count},counted) :
    {CombineLeaf.left_content(~Element,OperationsLaws.combine_model(~Element,~operation,OperationsLaws.PairCall{True{},_,right_view,Views.View{left_axes,0},right_length,left_length},right_tree,left_tree)) ==
      Buffer.Content{CombineLeaf.readback_output(~Element,result),left_length} : Buffer.Content<Element>}
  CombineLeaf.model_contents(~Element,left_length,right_length,result)

# What a piece of a spread and the right operand supply to every cell.
type SpreadCell<-Element: Data,-Left: Data,-Right: Data,-left_value: Left -> U32 -> Element,-right_value: Right -> U32 -> Element> is Data:
  SpreadCell{limits: Machine.Limits,axes: List<&2,Views.Axis>,offset: U32,piece: U32,left_context: Left,right_context: Right,
    left_tree: Tree.Tree<Element>,length: Nat,right_tree: Tree.Tree<Element>,right_length: Nat,
    room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},
    rank_room: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target())))) == True{} : Bool},
    within: {Nat.is_le(Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))),ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},
    ready: Buffer.ContentReady<Element,Buffer.Content{left_tree,length}>,exact: {length == U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)) : Nat},
    right_ready: Buffer.ContentReady<Element,Buffer.Content{right_tree,right_length}>,right_room: {Nat.is_le(right_length,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    fits: {Views.fits_nonempty(False{},Views.View{axes,offset},U32.from_nat(right_length)) == True{} : Bool},
    left_values: Content.Leaf(~Element,~Left,~left_value,left_context,Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),left_tree,length),
    right_values: Contract.Values(~Element,~Right,~right_value,right_context,Buffer.Content{right_tree,right_length})}

def cell_count(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,scope: SpreadCell<Element,Left,Right,left_value,right_value>) -> Nat:
  SpreadCell{limits,axes,offset,piece,left_context,right_context,left_tree,length,right_tree,right_length,room,nonempty,rank_room,inside,within,ready,exact,right_ready,right_room,fits,left_values,right_values} = scope
  U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))

def SpreadAt(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,+scope: SpreadCell<Element,Left,Right,left_value,right_value>,+index: Nat) -> Data:
  SpreadCell{limits,axes,offset,piece,left_context,right_context,left_tree,length,right_tree,right_length,room,nonempty,rank_room,inside,within,ready,exact,right_ready,right_room,fits,left_values,right_values} = scope
  Content.At(~Element,~CombineLeaf.Context<Left,ViewGather.Viewing<Right>>,~CombineLeaf.value(~Element,~Left,~ViewGather.Viewing<Right>,~operation,~left_value,~ViewGather.viewed(~Element,~Right,~right_value)),CombineLeaf.Context{left_context,ViewGather.Viewing{right_context,axes,offset,[]}},Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),CombineLeaf.readback_output(~Element,AccessLaws.combine_model(~Element,~operation,Access.zero_contiguous(Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset})) && U32.is_zero(0),U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset}),0,right_tree,left_tree)),length,index)

# A cell of the piece combines the left cell with the right value at the
# address of the cell in the view.
def spread_cell_with(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,+limits: Machine.Limits,+axes: List<&2,Views.Axis>,+offset: U32,+piece: U32,+left_context: Left,+right_context: Right,+left_tree: Tree.Tree<Element>,+length: Nat,+right_tree: Tree.Tree<Element>,+right_length: Nat,
  +room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target())))) == True{} : Bool},+within: {Nat.is_le(Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))),ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},+ready: Buffer.ContentReady<Element,Buffer.Content{left_tree,length}>,+exact: {length == U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)) : Nat},+right_ready: Buffer.ContentReady<Element,Buffer.Content{right_tree,right_length}>,+fits: {Views.fits_nonempty(False{},Views.View{axes,offset},U32.from_nat(right_length)) == True{} : Bool},+right_values: Contract.Values(~Element,~Right,~right_value,right_context,Buffer.Content{right_tree,right_length}),
  +index: Nat,+below: {Nat.is_lt(index,U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))) == True{} : Bool},
  left_cell: Content.At(~Element,~Left,~left_value,left_context,Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),left_tree,length,index)) ->
  Content.At(~Element,~CombineLeaf.Context<Left,ViewGather.Viewing<Right>>,~CombineLeaf.value(~Element,~Left,~ViewGather.Viewing<Right>,~operation,~left_value,~ViewGather.viewed(~Element,~Right,~right_value)),CombineLeaf.Context{left_context,ViewGather.Viewing{right_context,axes,offset,[]}},Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),CombineLeaf.readback_output(~Element,AccessLaws.combine_model(~Element,~operation,Access.zero_contiguous(Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset})) && U32.is_zero(0),U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset}),0,right_tree,left_tree)),length,index):
  KernelGemm.Both{+selected,+left_equal} = left_cell
  +rows = U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))
  +start = U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))
  +cell = Nat.add(start,index)
  +position = U32.from_nat(index)
  +pixel = U32.from_nat(cell)
  +cell_in = Bounds.nat_trans(cell,Nat.add(start,rows),ShapeLaws.elements(Views.axis_extents(axes)),Order.offset_lt(start,index,rows,below),within)
  +start_in = Order.le_lt_trans(start,cell,ShapeLaws.elements(Views.axis_extents(axes)),Order.le_add(start,index),cell_in)
  +same = ViewPart.same_box_of(axes,Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))
  +boxed_eq = ViewReads.piece_view_boxed(limits,axes,offset,Grid.piece_target(),room,piece,start_in)
  +box_count = Equal.trans(Nat,ShapeLaws.elements(Views.axis_extents(ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))))),ShapeLaws.elements(Grid.box_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))),rows,
    Equal.cong(List<&2,U32>,Nat,extents => ShapeLaws.elements(extents),Views.axis_extents(ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)))),Grid.box_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),ViewReads.boxed_extents(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),same)),
    Targets.grid_box_cells(limits,Views.axis_extents(axes),Grid.piece_target(),room,nonempty,rank_room,piece,inside))
  +rows_room = {%exact : {Nat.is_le(_,U32.to_nat(Tree.capacity(Element,left_tree))) == True{} : Bool} content_room(~Element,left_tree,length,ready) :
    {Nat.is_le(rows,U32.to_nat(Tree.capacity(Element,left_tree))) == True{} : Bool}}
  +combined = CombineValues.combined_value(~Element,~operation,Access.zero_contiguous(Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset})) && U32.is_zero(0),Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset}),right_tree,left_tree,length,rows,index,with_true(Access.zero_contiguous(Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset}))),
    ready,content_capacity(~Element,left_tree,length,ready),rows_room,below,
    %Equal.sym(Views.View,Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset}),Views.View{ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))),(offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))) : U32)},boxed_eq) : {ShapeLaws.elements(Views.axis_extents_of(_)) == rows : Nat} box_count)
  +moved = ViewReads.piece_address(limits,axes,offset,Grid.piece_target(),room,nonempty,rank_room,piece,inside,index,below)
  +address = CopyValues.address(Views.View{axes,offset},cell)
  +address_inside = {%Bounds.roundtrip(right_length,Tree.capacity(Element,right_tree),content_capacity(~Element,right_tree,right_length,right_ready),
      content_room(~Element,right_tree,right_length,right_ready)) : {Nat.is_lt(U32.to_nat(address),_) == True{} : Bool}
    CopyValues.address_inside(Views.View{axes,offset},cell,U32.from_nat(right_length),fits) : {Nat.is_lt(U32.to_nat(address),right_length) == True{} : Bool}}
  +right_read = {%Arithmetic.native_roundtrip(address) : {Tree.read(Element,right_tree,_) == right_value(right_context,_) : Element}
    Contract.read(~Element,~Right,~right_value,right_context,Buffer.Content{right_tree,right_length},U32.to_nat(address),right_values,address_inside) :
    {Tree.read(Element,right_tree,address) == right_value(right_context,address) : Element}}
  +encoded = Bounds.roundtrip(cell,Machine.capacity(limits),LimitBounds.capacity_fits(limits),
    Order.transitive(cell,ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits)),Bounds.strict_le(cell,ShapeLaws.elements(Views.axis_extents(axes)),cell_in),room))
  +mirrored = (Views.mirrored_offset(axes,[],offset) + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(Views.mirrored_axes(axes,[])),cell) : U32)
  +viewed_value = {%Equal.sym(Nat,U32.to_nat(pixel),cell,encoded) :
      {right_value(right_context,(Views.mirrored_offset(axes,[],offset) + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(Views.mirrored_axes(axes,[])),_) : U32)) == right_value(right_context,address) : Element}
    Equal.cong(U32,Element,word => right_value(right_context,word),mirrored,address,ViewReads.viewed_address(axes,offset,cell,cell_in)) :
    {ViewGather.viewed(~Element,~Right,~right_value)(ViewGather.Viewing{right_context,axes,offset,[]},pixel) == right_value(right_context,address) : Element}}
  KernelGemm.Both{selected,
    Equal.trans(Element,Tree.read(Element,CombineLeaf.readback_output(~Element,AccessLaws.combine_model(~Element,~operation,Access.zero_contiguous(Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset})) && U32.is_zero(0),U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset}),0,right_tree,left_tree)),position),
      operation(Tree.read(Element,left_tree,position),Tree.read(Element,right_tree,CopyValues.address(Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset}),index))),
      operation(left_value(left_context,pixel),ViewGather.viewed(~Element,~Right,~right_value)(ViewGather.Viewing{right_context,axes,offset,[]},pixel)),
      combined,
      Equal.trans(Element,operation(Tree.read(Element,left_tree,position),Tree.read(Element,right_tree,CopyValues.address(Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset}),index))),
        operation(left_value(left_context,pixel),Tree.read(Element,right_tree,CopyValues.address(Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset}),index))),
        operation(left_value(left_context,pixel),ViewGather.viewed(~Element,~Right,~right_value)(ViewGather.Viewing{right_context,axes,offset,[]},pixel)),
        Equal.cong(Element,Element,element => operation(element,Tree.read(Element,right_tree,CopyValues.address(Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset}),index))),
          Tree.read(Element,left_tree,position),left_value(left_context,pixel),left_equal),
        Equal.cong(Element,Element,element => operation(left_value(left_context,pixel),element),
          Tree.read(Element,right_tree,CopyValues.address(Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset}),index)),ViewGather.viewed(~Element,~Right,~right_value)(ViewGather.Viewing{right_context,axes,offset,[]},pixel),
          Equal.trans(Element,Tree.read(Element,right_tree,CopyValues.address(Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset}),index)),right_value(right_context,address),ViewGather.viewed(~Element,~Right,~right_value)(ViewGather.Viewing{right_context,axes,offset,[]},pixel),
            %Equal.sym(U32,CopyValues.address(Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset}),index),address,moved) : {Tree.read(Element,right_tree,_) == right_value(right_context,address) : Element} right_read,
            Equal.sym(Element,ViewGather.viewed(~Element,~Right,~right_value)(ViewGather.Viewing{right_context,axes,offset,[]},pixel),right_value(right_context,address),viewed_value)))))}

def spread_cell(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,+scope: SpreadCell<Element,Left,Right,left_value,right_value>,+index: Nat,
  +below: {Nat.is_lt(index,cell_count(~Element,~Left,~Right,~operation,~left_value,~right_value,scope)) == True{} : Bool}) -> SpreadAt(~Element,~Left,~Right,~operation,~left_value,~right_value,scope,index):
  SpreadCell{+limits,+axes,+offset,+piece,+left_context,+right_context,+left_tree,+length,+right_tree,+right_length,+room,+nonempty,+rank_room,+inside,+within,+ready,+exact,+right_ready,+right_room,+fits,+left_values,+right_values} = scope
  spread_cell_with(~Element,~Left,~Right,~operation,~left_value,~right_value,limits,axes,offset,piece,left_context,right_context,left_tree,length,right_tree,right_length,
    room,nonempty,rank_room,inside,within,ready,exact,right_ready,fits,right_values,index,below,
    Content.leaf_cell(~Element,~Left,~left_value,left_context,Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),left_tree,length,index,left_values,below))

# The guard of a piece holds: the box view and the piece of the view have one
# shape, the box view is contiguous and both views fit their storage.
def spread_guard(~Element: Data,+limits: Machine.Limits,+axes: List<&2,Views.Axis>,+offset: U32,+piece: U32,+left_tree: Tree.Tree<Element>,+length: Nat,+right_tree: Tree.Tree<Element>,+right_length: Nat,+room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target())))) == True{} : Bool},+within: {Nat.is_le(Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))),ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},+ready: Buffer.ContentReady<Element,Buffer.Content{left_tree,length}>,+exact: {length == U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)) : Nat},+right_ready: Buffer.ContentReady<Element,Buffer.Content{right_tree,right_length}>,+right_room: {Nat.is_le(right_length,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: {Views.fits_nonempty(False{},Views.View{axes,offset},U32.from_nat(right_length)) == True{} : Bool}) ->
  {Shapes.same(Views.shape(Grid.box_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))),Views.shape(Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset}))) && Views.is_contiguous(Grid.box_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))) && Operations.valid_buffer(length,Tree.capacity(Element,left_tree),Grid.box_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))) && Operations.valid_buffer(right_length,Tree.capacity(Element,right_tree),Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset})) == True{} : Bool}:
  +rows = U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))
  +start = U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))
  +positive = Targets.grid_span_positive(limits,Views.axis_extents(axes),Grid.piece_target(),room,nonempty,piece,inside)
  +start_in = Bounds.nat_trans(start,Nat.add(start,rows),ShapeLaws.elements(Views.axis_extents(axes)),
    {%Addition.zero_right(start) : {Nat.is_lt(_,Nat.add(start,rows)) == True{} : Bool} Order.offset_lt(start,0n,rows,positive) :
      {Nat.is_lt(start,Nat.add(start,rows)) == True{} : Bool}},within)
  +same = ViewPart.same_box_of(axes,Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))
  +boxed_eq = ViewReads.piece_view_boxed(limits,axes,offset,Grid.piece_target(),room,piece,start_in)
  +box_cells = Targets.grid_box_cells(limits,Views.axis_extents(axes),Grid.piece_target(),room,nonempty,rank_room,piece,inside)
  +box_positive = {%Equal.sym(Nat,ShapeLaws.elements(Grid.box_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))),rows,box_cells) : {Nat.is_lt(0n,_) == True{} : Bool} positive :
    {Nat.is_lt(0n,ShapeLaws.elements(Grid.box_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)))) == True{} : Bool}}
  +box_room = Order.transitive(ShapeLaws.elements(Grid.box_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))),ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits)),
    {%Equal.sym(Nat,ShapeLaws.elements(Grid.box_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))),rows,box_cells) : {Nat.is_le(_,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}
      Order.transitive(rows,Nat.add(start,rows),ShapeLaws.elements(Views.axis_extents(axes)),Order.right_le(start,rows),within) :
      {Nat.is_le(ShapeLaws.elements(Grid.box_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))),ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}},room)
  +left_length = Equal.trans(Nat,U32.to_nat(U32.from_nat(length)),length,ShapeLaws.elements(Grid.box_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))),
    Bounds.roundtrip(length,Tree.capacity(Element,left_tree),content_capacity(~Element,left_tree,length,ready),content_room(~Element,left_tree,length,ready)),
    Equal.trans(Nat,length,rows,ShapeLaws.elements(Grid.box_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))),exact,Equal.sym(Nat,ShapeLaws.elements(Grid.box_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))),rows,box_cells)))
  +right_length_room = {%Equal.sym(Nat,U32.to_nat(U32.from_nat(right_length)),right_length,
      Bounds.roundtrip(right_length,Tree.capacity(Element,right_tree),content_capacity(~Element,right_tree,right_length,right_ready),
        content_room(~Element,right_tree,right_length,right_ready))) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    right_room : {Nat.is_le(U32.to_nat(U32.from_nat(right_length)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +same_shape = {%Equal.sym(Views.View,Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset}),Views.View{ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))),(offset + Box.stepped_sum(Views.axis_extents(axes),ViewPart.view_steps(axes),U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))) : U32)},boxed_eq) : {Shapes.same(Views.shape(Grid.box_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))),Views.shape(_)) == True{} : Bool}
    %Equal.sym(List<&2,U32>,Views.axis_extents(Grid.dense_axes(Grid.box_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)))),Grid.box_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),ViewReads.dense_extents(Grid.box_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)))) :
      {Shapes.same_extents(_,Views.axis_extents(ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))))) == True{} : Bool}
    %Equal.sym(List<&2,U32>,Views.axis_extents(ViewPart.boxed_axes(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)))),Grid.box_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),ViewReads.boxed_extents(axes,Grid.box_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),same)) :
      {Shapes.same_extents(Grid.box_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),_) == True{} : Bool}
    ShapeLaws.same_extents_reflexive(Grid.box_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))) : {Shapes.same(Views.shape(Grid.box_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))),Views.shape(Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset}))) == True{} : Bool}}
  +contiguous = ViewReads.dense_contiguous(Grid.box_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),limits,box_room,box_positive)
  +valid_left = Logic.and_true(Nat.is_le(length,U32.to_nat(Tree.capacity(Element,left_tree))),Views.fits(Grid.box_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),U32.from_nat(length)),
    content_room(~Element,left_tree,length,ready),ViewReads.dense_fits(Grid.box_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),U32.from_nat(length),limits,box_room,box_positive,left_length))
  +valid_right = Logic.and_true(Nat.is_le(right_length,U32.to_nat(Tree.capacity(Element,right_tree))),Views.fits(Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset}),U32.from_nat(right_length)),
    content_room(~Element,right_tree,right_length,right_ready),
    ViewReads.piece_fits(limits,axes,offset,Grid.piece_target(),room,nonempty,rank_room,piece,inside,U32.from_nat(right_length),right_length_room,fits))
  %Equal.sym(Bool,Shapes.same(Views.shape(Grid.box_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))),Views.shape(Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset}))),True{},same_shape) : {_ && Views.is_contiguous(Grid.box_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))) && Operations.valid_buffer(length,Tree.capacity(Element,left_tree),Grid.box_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))) && Operations.valid_buffer(right_length,Tree.capacity(Element,right_tree),Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset})) == True{} : Bool}
  %Equal.sym(Bool,Views.is_contiguous(Grid.box_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))),True{},contiguous) : {True{} && _ && Operations.valid_buffer(length,Tree.capacity(Element,left_tree),Grid.box_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))) && Operations.valid_buffer(right_length,Tree.capacity(Element,right_tree),Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset})) == True{} : Bool}
  %Equal.sym(Bool,Operations.valid_buffer(length,Tree.capacity(Element,left_tree),Grid.box_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))),True{},valid_left) : {True{} && True{} && _ && Operations.valid_buffer(right_length,Tree.capacity(Element,right_tree),Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset})) == True{} : Bool}
  %Equal.sym(Bool,Operations.valid_buffer(right_length,Tree.capacity(Element,right_tree),Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset})),True{},valid_right) : {True{} && True{} && True{} && _ == True{} : Bool}
  {==}

# The guard of a piece holds, so the piece holds the kernel written into it.
def spread_content(~Element: Data,~operation: Element -> Element -> Element,+limits: Machine.Limits,+axes: List<&2,Views.Axis>,+offset: U32,+piece: U32,+left_tree: Tree.Tree<Element>,+length: Nat,+right_tree: Tree.Tree<Element>,+right_length: Nat,
  +room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target())))) == True{} : Bool},+within: {Nat.is_le(Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))),ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},+ready: Buffer.ContentReady<Element,Buffer.Content{left_tree,length}>,+exact: {length == U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)) : Nat},+right_ready: Buffer.ContentReady<Element,Buffer.Content{right_tree,right_length}>,+right_room: {Nat.is_le(right_length,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: {Views.fits_nonempty(False{},Views.View{axes,offset},U32.from_nat(right_length)) == True{} : Bool}) ->
  {CombineLeaf.left_content(~Element,denoted(~Element,~operation,Grid.box_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset}),Buffer.Content{left_tree,length},Buffer.Content{right_tree,right_length})) == Buffer.Content{CombineLeaf.readback_output(~Element,AccessLaws.combine_model(~Element,~operation,Access.zero_contiguous(Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset})) && U32.is_zero(0),U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset}),0,right_tree,left_tree)),length} : Buffer.Content<Element>}:
  guarded_left(~Element,~operation,Grid.dense_axes(Grid.box_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))),Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset}),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),left_tree,length,right_tree,right_length,
    spread_guard(~Element,limits,axes,offset,piece,left_tree,length,right_tree,right_length,room,nonempty,rank_room,inside,within,ready,exact,right_ready,right_room,fits),
    ViewReads.dense_numel(Grid.box_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Targets.grid_box_cells(limits,Views.axis_extents(axes),Grid.piece_target(),room,nonempty,rank_room,piece,inside)))

# A piece of a spread holds the left piece combined with the right operand
# read through the view.
def spread_values(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,+limits: Machine.Limits,+axes: List<&2,Views.Axis>,+offset: U32,+piece: U32,+left_context: Left,+right_context: Right,+left_tree: Tree.Tree<Element>,+length: Nat,+right_tree: Tree.Tree<Element>,+right_length: Nat,
  +room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Grid.grid(Views.axis_extents(axes),Grid.piece_target())))) == True{} : Bool},+within: {Nat.is_le(Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))),ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},+ready: Buffer.ContentReady<Element,Buffer.Content{left_tree,length}>,+exact: {length == U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)) : Nat},+right_ready: Buffer.ContentReady<Element,Buffer.Content{right_tree,right_length}>,+right_room: {Nat.is_le(right_length,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: {Views.fits_nonempty(False{},Views.View{axes,offset},U32.from_nat(right_length)) == True{} : Bool},+left_values: Content.Leaf(~Element,~Left,~left_value,left_context,Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),left_tree,length),+right_values: Contract.Values(~Element,~Right,~right_value,right_context,Buffer.Content{right_tree,right_length})) ->
  Content.piece_values(~Element,~CombineLeaf.Context<Left,ViewGather.Viewing<Right>>,~CombineLeaf.value(~Element,~Left,~ViewGather.Viewing<Right>,~operation,~left_value,~ViewGather.viewed(~Element,~Right,~right_value)),CombineLeaf.Context{left_context,ViewGather.Viewing{right_context,axes,offset,[]}},Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),CombineLeaf.left_content(~Element,denoted(~Element,~operation,Grid.box_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset}),Buffer.Content{left_tree,length},Buffer.Content{right_tree,right_length}))):
  +rows = Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)
  %Equal.sym(Buffer.Content<Element>,CombineLeaf.left_content(~Element,denoted(~Element,~operation,Grid.box_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset}),Buffer.Content{left_tree,length},Buffer.Content{right_tree,right_length})),Buffer.Content{CombineLeaf.readback_output(~Element,AccessLaws.combine_model(~Element,~operation,Access.zero_contiguous(Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset})) && U32.is_zero(0),U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset}),0,right_tree,left_tree)),length},spread_content(~Element,~operation,limits,axes,offset,piece,left_tree,length,right_tree,right_length,room,nonempty,rank_room,inside,within,ready,exact,right_ready,right_room,fits)) :
    Content.piece_values(~Element,~CombineLeaf.Context<Left,ViewGather.Viewing<Right>>,~CombineLeaf.value(~Element,~Left,~ViewGather.Viewing<Right>,~operation,~left_value,~ViewGather.viewed(~Element,~Right,~right_value)),CombineLeaf.Context{left_context,ViewGather.Viewing{right_context,axes,offset,[]}},Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),rows,_)
  Members.range_all(~SpreadCell<Element,Left,Right,left_value,right_value>,~SpreadAt(~Element,~Left,~Right,~operation,~left_value,~right_value),~cell_count(~Element,~Left,~Right,~operation,~left_value,~right_value),~spread_cell(~Element,~Left,~Right,~operation,~left_value,~right_value),U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),0n,
    SpreadCell{limits,axes,offset,piece,left_context,right_context,left_tree,length,right_tree,right_length,room,nonempty,rank_room,inside,within,ready,exact,right_ready,right_room,fits,left_values,right_values},Order.reflexive(U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))))

# ---- Every piece of a spread ----

# A piece of the canonical pieces of the left operand combines with the right
# operand read through the view and steps into a piece of the reader after the
# step.
def spread_piece_facts(~Element: Data,~Later: Data,~operation: Element -> Element -> Element,~after_step: @+later: Later -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,~Left: Data,~left_value: Left -> U32 -> Element,~Right: Data,~right_value: Right -> U32 -> Element,~After: Data,~after_value: After -> U32 -> Element,~Scope: Data,~later_of: @+scope: Scope -> Later,~left_of: @+scope: Scope -> Left,~right_of: @+scope: Scope -> ViewGather.Viewing<Right>,~after_of: @+scope: Scope -> After,~total_of: @+scope: Scope -> Nat,~rule: PiecesCombine.Rule(~Element,~Later,~after_step,~operation,~Left,~left_value,~ViewGather.Viewing<Right>,~ViewGather.viewed(~Element,~Right,~right_value),~After,~after_value,~Scope,~later_of,~left_of,~right_of,~after_of,~total_of),+scope: Scope,+limits: Machine.Limits,+axes: List<&2,Views.Axis>,+offset: U32,+right_context: Right,+viewing: {right_of(scope) == ViewGather.Viewing{right_context,axes,offset,[]} : ViewGather.Viewing<Right>},+right_tree: Tree.Tree<Element>,+right_length: Nat,+total_exact: {total_of(scope) == ShapeLaws.elements(Views.axis_extents(axes)) : Nat},+room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+right_ready: Buffer.ContentReady<Element,Buffer.Content{right_tree,right_length}>,+right_room: {Nat.is_le(right_length,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: {Views.fits_nonempty(False{},Views.View{axes,offset},U32.from_nat(right_length)) == True{} : Bool},+right_values: Contract.Values(~Element,~Right,~right_value,right_context,Buffer.Content{right_tree,right_length}),
  +first: U32,+rows: U32,+left_tree: Tree.Tree<Element>,+length: Nat,
  +at: GridRanges.PieceAt<Grid.grid(Views.axis_extents(axes),Grid.piece_target()),first,rows>,+fields: Content.Fields<first,rows,length,total_of(scope)>,+ready: Buffer.ContentReady<Element,Buffer.Content{left_tree,length}>,
  +left_values: Content.Leaf(~Element,~Left,~left_value,left_of(scope),first,rows,left_tree,length)) ->
  PiecesMap.Facts(~Element,~After,~after_value,after_of(scope),total_of(scope),Model.PieceContent{spread_model(~Element,~Later,~operation,~after_step,Tensors.Spread{Views.axis_extents(axes),axes,offset,later_of(scope)},Buffer.Content{right_tree,right_length},first,rows,Buffer.Content{left_tree,length}),first,rows}):
  GridRanges.PieceAt{+piece,+inside,+start,+span} = at
  Content.Fields{+length_exact,+within_total} = fields
  +exact = {%Equal.sym(U32,Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),rows,span) : {length == U32.to_nat(_) : Nat} length_exact : {length == U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)) : Nat}}
  +within = {%Equal.sym(U32,Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),first,start) : {Nat.is_le(Nat.add(U32.to_nat(_),U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))),ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}
    %Equal.sym(U32,Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),rows,span) : {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(_)),ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}
    %total_exact : {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),_) == True{} : Bool} within_total :
    {Nat.is_le(Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))),ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}}
  +inside_total = {%Equal.sym(Nat,total_of(scope),ShapeLaws.elements(Views.axis_extents(axes)),total_exact) : {Nat.is_le(Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))),_) == True{} : Bool} within :
    {Nat.is_le(Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece))),total_of(scope)) == True{} : Bool}}
  +values = {%Equal.sym(U32,Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),first,start) : Content.Leaf(~Element,~Left,~left_value,left_of(scope),_,Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),left_tree,length)
    %Equal.sym(U32,Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),rows,span) : Content.Leaf(~Element,~Left,~left_value,left_of(scope),first,_,left_tree,length) left_values :
    Content.Leaf(~Element,~Left,~left_value,left_of(scope),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),left_tree,length)}
  +content = spread_content(~Element,~operation,limits,axes,offset,piece,left_tree,length,right_tree,right_length,room,nonempty,rank_room,inside,within,ready,exact,
    right_ready,right_room,fits)
  +input = CombineLeaf.left_content(~Element,denoted(~Element,~operation,Grid.box_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset}),Buffer.Content{left_tree,length},Buffer.Content{right_tree,right_length}))
  +input_ready = {%Equal.sym(Buffer.Content<Element>,input,Buffer.Content{CombineLeaf.readback_output(~Element,AccessLaws.combine_model(~Element,~operation,Access.zero_contiguous(Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset})) && U32.is_zero(0),U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset}),0,right_tree,left_tree)),length},content) : Buffer.ContentReady<Element,_>
    CombineValues.combined_ready(~Element,~operation,Access.zero_contiguous(Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset})) && U32.is_zero(0),U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset}),right_tree,left_tree,length,ready) : Buffer.ContentReady<Element,input>}
  +input_length = {%Equal.sym(Buffer.Content<Element>,input,Buffer.Content{CombineLeaf.readback_output(~Element,AccessLaws.combine_model(~Element,~operation,Access.zero_contiguous(Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset})) && U32.is_zero(0),U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)),Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Views.View{axes,offset}),0,right_tree,left_tree)),length},content) : {Buffer.content_length(~Element,_) == U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)) : Nat} exact :
    {Buffer.content_length(~Element,input) == U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece)) : Nat}}
  +input_values = {%Equal.sym(ViewGather.Viewing<Right>,right_of(scope),ViewGather.Viewing{right_context,axes,offset,[]},viewing) :
      Content.piece_values(~Element,~CombineLeaf.Context<Left,ViewGather.Viewing<Right>>,~CombineLeaf.value(~Element,~Left,~ViewGather.Viewing<Right>,~operation,~left_value,~ViewGather.viewed(~Element,~Right,~right_value)),CombineLeaf.Context{left_of(scope),_},Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),input)
    spread_values(~Element,~Left,~Right,~operation,~left_value,~right_value,limits,axes,offset,piece,left_of(scope),right_context,left_tree,length,right_tree,right_length,
      room,nonempty,rank_room,inside,within,ready,exact,right_ready,right_room,fits,values,right_values) :
    Content.piece_values(~Element,~CombineLeaf.Context<Left,ViewGather.Viewing<Right>>,~CombineLeaf.value(~Element,~Left,~ViewGather.Viewing<Right>,~operation,~left_value,~ViewGather.viewed(~Element,~Right,~right_value)),CombineLeaf.Context{left_of(scope),right_of(scope)},Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),input)}
  +made = rule(scope,Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),input,input_ready,input_length,inside_total,input_values)
  %start : PiecesMap.Facts(~Element,~After,~after_value,after_of(scope),total_of(scope),Model.PieceContent{after_step(later_of(scope),_,rows,CombineLeaf.left_content(~Element,denoted(~Element,~operation,Grid.box_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),rows),Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),_,rows,Views.View{axes,offset}),Buffer.Content{left_tree,length},Buffer.Content{right_tree,right_length}))),_,rows})
  %span : PiecesMap.Facts(~Element,~After,~after_value,after_of(scope),total_of(scope),Model.PieceContent{after_step(later_of(scope),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),_,CombineLeaf.left_content(~Element,denoted(~Element,~operation,Grid.box_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),_),Grid.piece_view(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),_,Views.View{axes,offset}),Buffer.Content{left_tree,length},Buffer.Content{right_tree,right_length}))),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),_})
  PiecesMap.piece_facts(~Element,~After,~after_value,after_of(scope),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),total_of(scope),input,after_step(later_of(scope),Grid.start(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),Grid.span(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),piece),input),input_length,inside_total,made)

# Every canonical piece of the left operand combines with the right operand
# read through the view.
def spread_facts(~Element: Data,~Later: Data,~operation: Element -> Element -> Element,~after_step: @+later: Later -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,~Left: Data,~left_value: Left -> U32 -> Element,~Right: Data,~right_value: Right -> U32 -> Element,~After: Data,~after_value: After -> U32 -> Element,~Scope: Data,~later_of: @+scope: Scope -> Later,~left_of: @+scope: Scope -> Left,~right_of: @+scope: Scope -> ViewGather.Viewing<Right>,~after_of: @+scope: Scope -> After,~total_of: @+scope: Scope -> Nat,~rule: PiecesCombine.Rule(~Element,~Later,~after_step,~operation,~Left,~left_value,~ViewGather.Viewing<Right>,~ViewGather.viewed(~Element,~Right,~right_value),~After,~after_value,~Scope,~later_of,~left_of,~right_of,~after_of,~total_of),+scope: Scope,+limits: Machine.Limits,+axes: List<&2,Views.Axis>,+offset: U32,+right_context: Right,+viewing: {right_of(scope) == ViewGather.Viewing{right_context,axes,offset,[]} : ViewGather.Viewing<Right>},+right_tree: Tree.Tree<Element>,+right_length: Nat,+total_exact: {total_of(scope) == ShapeLaws.elements(Views.axis_extents(axes)) : Nat},+room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+right_ready: Buffer.ContentReady<Element,Buffer.Content{right_tree,right_length}>,+right_room: {Nat.is_le(right_length,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: {Views.fits_nonempty(False{},Views.View{axes,offset},U32.from_nat(right_length)) == True{} : Bool},+right_values: Contract.Values(~Element,~Right,~right_value,right_context,Buffer.Content{right_tree,right_length}),content: Model.Content<Element>,
  at: GridRanges.PiecesAt(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Model.content_layout(~Element,content)),fields: Content.Contents(~Element,total_of(scope),content),
  ready: Model.ContentReady(~Element,content),values: Content.Values(~Element,~Left,~left_value,left_of(scope),content)) ->
  PiecesMap.Facts(~Element,~After,~after_value,after_of(scope),total_of(scope),Model.map(~Element,~spread_model(~Element,~Later,~operation,~after_step,Tensors.Spread{Views.axis_extents(axes),axes,offset,later_of(scope)},Buffer.Content{right_tree,right_length}),content)):
  match content:
    case Model.PieceContent{+buffer,+first,+rows}:
      Buffer.Content{+left_tree,+length} = buffer
      spread_piece_facts(~Element,~Later,~operation,~after_step,~Left,~left_value,~Right,~right_value,~After,~after_value,~Scope,~later_of,~left_of,~right_of,~after_of,~total_of,~rule,scope,limits,axes,offset,right_context,viewing,right_tree,right_length,total_exact,room,nonempty,rank_room,right_ready,right_room,fits,right_values,first,rows,left_tree,length,at,fields,ready,values)
    case Model.Joined{+head,+tail}:
      Held.Both{+head_at,+tail_at} = at
      KernelGemm.Both{+head_fields,+tail_fields} = fields
      KernelGemm.Both{+head_ready,+tail_ready} = ready
      KernelGemm.Both{+head_values,+tail_values} = values
      PiecesMap.joined_facts(~Element,~After,~after_value,after_of(scope),total_of(scope),Model.map(~Element,~spread_model(~Element,~Later,~operation,~after_step,Tensors.Spread{Views.axis_extents(axes),axes,offset,later_of(scope)},Buffer.Content{right_tree,right_length}),head),Model.map(~Element,~spread_model(~Element,~Later,~operation,~after_step,Tensors.Spread{Views.axis_extents(axes),axes,offset,later_of(scope)},Buffer.Content{right_tree,right_length}),tail),
        spread_facts(~Element,~Later,~operation,~after_step,~Left,~left_value,~Right,~right_value,~After,~after_value,~Scope,~later_of,~left_of,~right_of,~after_of,~total_of,~rule,scope,limits,axes,offset,right_context,viewing,right_tree,right_length,total_exact,room,nonempty,rank_room,right_ready,right_room,fits,right_values,head,head_at,head_fields,head_ready,head_values),
        spread_facts(~Element,~Later,~operation,~after_step,~Left,~left_value,~Right,~right_value,~After,~after_value,~Scope,~later_of,~left_of,~right_of,~after_of,~total_of,~rule,scope,limits,axes,offset,right_context,viewing,right_tree,right_length,total_exact,room,nonempty,rank_room,right_ready,right_room,fits,right_values,tail,tail_at,tail_fields,tail_ready,tail_values))

# ---- The contract of a spread ----

# A left operand of the canonical pieces of the left extents combines on the
# tasks of any decision with the right operand read through its view.
def spread_decided_with(~Element: Data,~Later: Data,~operation: Element -> Element -> Element,~after_step: @+later: Later -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,~Left: Data,~left_value: Left -> U32 -> Element,~Right: Data,~right_value: Right -> U32 -> Element,~After: Data,~after_value: After -> U32 -> Element,~Scope: Data,~later_of: @+scope: Scope -> Later,~left_of: @+scope: Scope -> Left,~right_of: @+scope: Scope -> ViewGather.Viewing<Right>,~after_of: @+scope: Scope -> After,~total_of: @+scope: Scope -> Nat,~rule: PiecesCombine.Rule(~Element,~Later,~after_step,~operation,~Left,~left_value,~ViewGather.Viewing<Right>,~ViewGather.viewed(~Element,~Right,~right_value),~After,~after_value,~Scope,~later_of,~left_of,~right_of,~after_of,~total_of),~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~after: Tensors.PieceMap(~Element,~Later),~after_rule: PiecesMap.Rule(~Element,~Later,~after,~after_step),+scope: Scope,decision: Tasks.Decision,+left: Model.Model<Element>,+fill: Element,+workers: U32,+limits: Machine.Limits,+axes: List<&2,Views.Axis>,+offset: U32,+right_context: Right,+viewing: {right_of(scope) == ViewGather.Viewing{right_context,axes,offset,[]} : ViewGather.Viewing<Right>},+right_shape: List<&2,U32>,+kept: Buffer.Model<Element>,+total_exact: {total_of(scope) == ShapeLaws.elements(Views.axis_extents(axes)) : Nat},+room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+right_formed: Contract.WellFormed<Element,right_shape,Buffer.content(~Element,kept)>,+right_room: {Nat.is_le(ShapeLaws.elements(right_shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: {Views.fits(Views.View{axes,offset},U32.from_nat(ShapeLaws.elements(right_shape))) == True{} : Bool},+right_values: Contract.Values(~Element,~Right,~right_value,right_context,Buffer.content(~Element,kept)),left_contract: Global.Contract(~Element,~Left,~left_value,left_of(scope),Tensors.Tensor{Model.pack(~Element,left),Views.axis_extents(axes),fill,Tensors.Usable{},workers})) ->
  Global.Contract(~Element,~After,~after_value,after_of(scope),Tensors.spread_decided(~Element,~Later,~kernel,~after,later_of(scope),decision,workers,Views.axis_extents(axes),Model.pack(~Element,left),fill,Buffer.pack(~Element,kept),axes,offset)):
  match decision:
    case Tasks.Decision{+parallel,+tasks}:
      Buffer.Model{+right_tree,+right_length,+certificate} = kept
      Contract.WellFormed{+length_exact,+right_ready} = right_formed
      KernelGemm.Both{KernelGemm.Both{Content.WellFormed{+fields,+ready,+coverage},KernelGemm.Both{+counted,+canonical}},+values} = left_contract
      +kept_model = {Buffer.Model{right_tree,right_length,certificate} : Buffer.Model<Element>}
      +context = {Tensors.Spread{Views.axis_extents(axes),axes,offset,later_of(scope)} : Tensors.Spread<Later>}
      +total = ShapeLaws.elements(Views.axis_extents(axes))
      +content = Model.content(~Element,left)
      +left_packed = Model.packed_contents(~Element,left)
      +output = Model.map(~Element,~spread_model(~Element,~Later,~operation,~after_step,context,Buffer.Content{right_tree,right_length}),content)
      +right_length_exact = Equal.sym(Nat,right_length,ShapeLaws.elements(right_shape),length_exact)
      +right_inside = {%right_length_exact : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} right_room : {Nat.is_le(right_length,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
      +right_fits = ViewReads.nonempty_fits(axes,offset,U32.from_nat(right_length),nonempty,
        %right_length_exact : {Views.fits(Views.View{axes,offset},U32.from_nat(_)) == True{} : Bool} fits)
      +layout_eq = Equal.trans(Pieces.Ranges,Model.content_layout(~Element,content),Model.layout(~Element,left),Pieces.ranges_of_grid(Tensors.grid_of(Views.axis_extents(axes))),
        Equal.sym(Pieces.Ranges,Model.layout(~Element,left),Model.content_layout(~Element,content),Model.layout_matches_content(~Element,left)),
        Equal.trans(Pieces.Ranges,Model.layout(~Element,left),Model.content_layout(~Element,Model.contents(~Element,Model.pack(~Element,left))),Pieces.ranges_of_grid(Tensors.grid_of(Views.axis_extents(axes))),
          PiecesCombine.packed_layout(~Element,left),canonical))
      +output_layout = Model.map_layout(~Element,~spread_model(~Element,~Later,~operation,~after_step,context,Buffer.Content{right_tree,right_length}),content)
      +output_canonical = Equal.trans(Pieces.Ranges,Model.content_layout(~Element,output),Model.content_layout(~Element,content),Pieces.ranges_of_grid(Tensors.grid_of(Views.axis_extents(axes))),output_layout,layout_eq)
      +output_packed = Equal.trans(Pieces.Ranges,Model.content_layout(~Element,output),Model.content_layout(~Element,content),Model.content_layout(~Element,Model.contents(~Element,Model.pack(~Element,left))),output_layout,
        Equal.cong(Model.Content<Element>,Pieces.Ranges,pieces => Model.content_layout(~Element,pieces),content,Model.contents(~Element,Model.pack(~Element,left)),Equal.sym(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,left)),content,left_packed)))
      +at = {%Equal.sym(Pieces.Ranges,Model.content_layout(~Element,content),Pieces.ranges_of_grid(Grid.grid(Views.axis_extents(axes),Grid.piece_target())),layout_eq) : GridRanges.PiecesAt(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),_)
        GridRanges.grid_pieces_at(limits,Grid.grid(Views.axis_extents(axes),Grid.piece_target()),GridRanges.grid_facts(limits,Views.axis_extents(axes),Grid.piece_target(),room,nonempty)) :
        GridRanges.PiecesAt(Grid.grid(Views.axis_extents(axes),Grid.piece_target()),Model.content_layout(~Element,content))}
      +total_back = Equal.sym(Nat,total_of(scope),total,total_exact)
      +facts = spread_facts(~Element,~Later,~operation,~after_step,~Left,~left_value,~Right,~right_value,~After,~after_value,~Scope,~later_of,~left_of,~right_of,~after_of,~total_of,~rule,scope,limits,axes,offset,right_context,viewing,right_tree,right_length,total_exact,room,nonempty,rank_room,
        right_ready,right_inside,right_fits,right_values,content,at,
        %total_back : Content.Contents(~Element,_,content) %left_packed : Content.Contents(~Element,total,_) fields,
        %left_packed : Model.ContentReady(~Element,_) ready,
        %left_packed : Content.Values(~Element,~Left,~left_value,left_of(scope),_) values)
      %Equal.sym(Model.Content<Element>,Model.contents(~Element,Pieces.grouped_with(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Tensors.Spread<Later>,~Tensors.shared_buffer(~Element),~Tensors.spread_piece(~Element,~Later,~kernel,~after),tasks,parallel,context,Model.pack(~Element,left),Buffer.pack(~Element,kept_model))),Model.map(~Element,~spread_model(~Element,~Later,~operation,~after_step,context,Buffer.contents(~Element,Buffer.pack(~Element,kept_model))),content),
          spread_walked(~Element,~Later,~operation,~kernel,~kernel_law,~after,~after_step,~after_rule,tasks,parallel,context,left,kept_model)) :
        PiecesMap.Holds(~Element,~After,~after_value,after_of(scope),Views.axis_extents(axes),_)
      %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,kept_model)),Buffer.Content{right_tree,right_length},Buffer.packed_contents(~Element,kept_model)) :
        PiecesMap.Holds(~Element,~After,~after_value,after_of(scope),Views.axis_extents(axes),Model.map(~Element,~spread_model(~Element,~Later,~operation,~after_step,context,_),content))
      PiecesMap.holds(~Element,~After,~after_value,after_of(scope),Views.axis_extents(axes),output,
        %total_exact : PiecesMap.Facts(~Element,~After,~after_value,after_of(scope),_,output) facts,
        %Equal.sym(Pieces.Ranges,Model.content_layout(~Element,output),Model.content_layout(~Element,Model.contents(~Element,Model.pack(~Element,left))),output_packed) : PiecesPlan.Coverage(_,0n,total) coverage,
        counted,output_canonical)

# A left operand of the canonical pieces of a shape combines on the tasks of
# any decision with a right operand read through a view of it with the extents
# of the shape.
def spread_decided(~Element: Data,~Later: Data,~operation: Element -> Element -> Element,~after_step: @+later: Later -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,~Left: Data,~left_value: Left -> U32 -> Element,~Right: Data,~right_value: Right -> U32 -> Element,~After: Data,~after_value: After -> U32 -> Element,~Scope: Data,~later_of: @+scope: Scope -> Later,~left_of: @+scope: Scope -> Left,~right_of: @+scope: Scope -> ViewGather.Viewing<Right>,~after_of: @+scope: Scope -> After,~total_of: @+scope: Scope -> Nat,~rule: PiecesCombine.Rule(~Element,~Later,~after_step,~operation,~Left,~left_value,~ViewGather.Viewing<Right>,~ViewGather.viewed(~Element,~Right,~right_value),~After,~after_value,~Scope,~later_of,~left_of,~right_of,~after_of,~total_of),~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~after: Tensors.PieceMap(~Element,~Later),~after_rule: PiecesMap.Rule(~Element,~Later,~after,~after_step),+scope: Scope,decision: Tasks.Decision,+left: Model.Model<Element>,+fill: Element,+workers: U32,+shape: List<&2,U32>,+limits: Machine.Limits,+axes: List<&2,Views.Axis>,+offset: U32,+right_context: Right,+viewing: {right_of(scope) == ViewGather.Viewing{right_context,axes,offset,[]} : ViewGather.Viewing<Right>},+right_shape: List<&2,U32>,+kept: Buffer.Model<Element>,+extents: {Views.axis_extents(axes) == shape : List<&2,U32>},+total_exact: {total_of(scope) == ShapeLaws.elements(shape) : Nat},+room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+right_formed: Contract.WellFormed<Element,right_shape,Buffer.content(~Element,kept)>,+right_room: {Nat.is_le(ShapeLaws.elements(right_shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: {Views.fits(Views.View{axes,offset},U32.from_nat(ShapeLaws.elements(right_shape))) == True{} : Bool},+right_values: Contract.Values(~Element,~Right,~right_value,right_context,Buffer.content(~Element,kept)),left_contract: Global.Contract(~Element,~Left,~left_value,left_of(scope),Tensors.Tensor{Model.pack(~Element,left),shape,fill,Tensors.Usable{},workers})) ->
  Global.Contract(~Element,~After,~after_value,after_of(scope),Tensors.spread_decided(~Element,~Later,~kernel,~after,later_of(scope),decision,workers,shape,Model.pack(~Element,left),fill,Buffer.pack(~Element,kept),axes,offset)):
  +back = Equal.sym(List<&2,U32>,Views.axis_extents(axes),shape,extents)
  %extents : Global.Contract(~Element,~After,~after_value,after_of(scope),Tensors.spread_decided(~Element,~Later,~kernel,~after,later_of(scope),decision,workers,_,
    Model.pack(~Element,left),fill,Buffer.pack(~Element,kept),axes,offset))
  spread_decided_with(~Element,~Later,~operation,~after_step,~Left,~left_value,~Right,~right_value,~After,~after_value,~Scope,~later_of,~left_of,~right_of,~after_of,~total_of,~rule,~kernel,~kernel_law,~after,~after_rule,scope,decision,left,fill,workers,limits,axes,offset,right_context,viewing,right_shape,kept,{%back : {total_of(scope) == ShapeLaws.elements(_) : Nat} total_exact : {total_of(scope) == ShapeLaws.elements(Views.axis_extents(axes)) : Nat}},{%back : {Nat.is_le(ShapeLaws.elements(_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} room : {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}},{%back : {Nat.is_lt(0n,ShapeLaws.elements(_)) == True{} : Bool} nonempty : {Nat.is_lt(0n,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}},{%back : {Nat.is_le(List.length(&2,U32,_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} rank_room : {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}},right_formed,right_room,fits,right_values,{%back : Global.Contract(~Element,~Left,~left_value,left_of(scope),Tensors.Tensor{Model.pack(~Element,left),_,fill,Tensors.Usable{},workers}) left_contract : Global.Contract(~Element,~Left,~left_value,left_of(scope),Tensors.Tensor{Model.pack(~Element,left),Views.axis_extents(axes),fill,Tensors.Usable{},workers})})

# The tasks of a spread divide the left operand and every one reads the whole
# right operand.
def spread_combined(~Element: Data,~Later: Data,~operation: Element -> Element -> Element,~after_step: @+later: Later -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,~Left: Data,~left_value: Left -> U32 -> Element,~Right: Data,~right_value: Right -> U32 -> Element,~After: Data,~after_value: After -> U32 -> Element,~Scope: Data,~later_of: @+scope: Scope -> Later,~left_of: @+scope: Scope -> Left,~right_of: @+scope: Scope -> ViewGather.Viewing<Right>,~after_of: @+scope: Scope -> After,~total_of: @+scope: Scope -> Nat,~rule: PiecesCombine.Rule(~Element,~Later,~after_step,~operation,~Left,~left_value,~ViewGather.Viewing<Right>,~ViewGather.viewed(~Element,~Right,~right_value),~After,~after_value,~Scope,~later_of,~left_of,~right_of,~after_of,~total_of),~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~after: Tensors.PieceMap(~Element,~Later),~after_rule: PiecesMap.Rule(~Element,~Later,~after,~after_step),+scope: Scope,+operations: F32,+left: Model.Model<Element>,+fill: Element,+workers: U32,+shape: List<&2,U32>,+limits: Machine.Limits,+axes: List<&2,Views.Axis>,+offset: U32,+right_context: Right,+viewing: {right_of(scope) == ViewGather.Viewing{right_context,axes,offset,[]} : ViewGather.Viewing<Right>},+right_shape: List<&2,U32>,+kept: Buffer.Model<Element>,+extents: {Views.axis_extents(axes) == shape : List<&2,U32>},+total_exact: {total_of(scope) == ShapeLaws.elements(shape) : Nat},+room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+right_formed: Contract.WellFormed<Element,right_shape,Buffer.content(~Element,kept)>,+right_room: {Nat.is_le(ShapeLaws.elements(right_shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: {Views.fits(Views.View{axes,offset},U32.from_nat(ShapeLaws.elements(right_shape))) == True{} : Bool},+right_values: Contract.Values(~Element,~Right,~right_value,right_context,Buffer.content(~Element,kept)),left_contract: Global.Contract(~Element,~Left,~left_value,left_of(scope),Tensors.Tensor{Model.pack(~Element,left),shape,fill,Tensors.Usable{},workers})) ->
  Global.Contract(~Element,~After,~after_value,after_of(scope),Tensors.combine_spread(~Element,~Later,~kernel,~after,later_of(scope),operations,workers,shape,Model.pack(~Element,left),fill,Buffer.pack(~Element,kept),right_shape,axes,offset)):
  spread_decided(~Element,~Later,~operation,~after_step,~Left,~left_value,~Right,~right_value,~After,~after_value,~Scope,~later_of,~left_of,~right_of,~after_of,~total_of,~rule,~kernel,~kernel_law,~after,~after_rule,scope,Tasks.decide(Tensors.grid_of(shape),workers,Tensors.pass_rows(2.0,operations),Tensors.pass_steps(),Tasks.Extra{(U32.to_f32(Tensors.count_or_zero(right_shape)) * 4.0 : F32),0.0,0.0}),left,fill,workers,shape,limits,axes,offset,right_context,viewing,right_shape,kept,extents,total_exact,room,nonempty,rank_room,right_formed,right_room,fits,right_values,left_contract)

# The tasks of a spread leave usable cells of the left extents.
def spread_usable(~Element: Data,~Later: Data,~kernel: OperationsLaws.Kernel(~Element),~after: Tensors.PieceMap(~Element,~Later),later: Later,decision: Tasks.Decision,
  +workers: U32,+shape: List<&2,U32>,left: Pieces.Pieces(~Element),fill: Element,right: Storage.Buffer<Element>,+axes: List<&2,Views.Axis>,+offset: U32) ->
  Global.Usable<Element,shape,Tensors.spread_decided(~Element,~Later,~kernel,~after,later,decision,workers,shape,left,fill,right,axes,offset)>:
  match decision:
    case Tasks.Decision{+parallel,+tasks}:
      Global.usable(~Element,Tensors.spread_decided(~Element,~Later,~kernel,~after,later,Tasks.Decision{parallel,tasks},workers,shape,left,fill,right,axes,offset),{==})

# ---- A right operand broadcast to the left ----

# A right operand of the contract joined into one buffer of all its cells is
# whole storage of its shape holding its values: the spread reads it so.
def joined_spread(~Element: Data,~Later: Data,~operation: Element -> Element -> Element,~after_step: @+later: Later -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,~Left: Data,~left_value: Left -> U32 -> Element,~Right: Data,~right_value: Right -> U32 -> Element,~After: Data,~after_value: After -> U32 -> Element,~Scope: Data,~later_of: @+scope: Scope -> Later,~left_of: @+scope: Scope -> Left,~right_of: @+scope: Scope -> ViewGather.Viewing<Right>,~after_of: @+scope: Scope -> After,~total_of: @+scope: Scope -> Nat,~rule: PiecesCombine.Rule(~Element,~Later,~after_step,~operation,~Left,~left_value,~ViewGather.Viewing<Right>,~ViewGather.viewed(~Element,~Right,~right_value),~After,~after_value,~Scope,~later_of,~left_of,~right_of,~after_of,~total_of),~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~after: Tensors.PieceMap(~Element,~Later),~after_rule: PiecesMap.Rule(~Element,~Later,~after,~after_step),+scope: Scope,+operations: F32,+left: Model.Model<Element>,+fill: Element,+workers: U32,+shape: List<&2,U32>,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +axes: List<&2,Views.Axis>,+offset: U32,+right_context: Right,+viewing: {right_of(scope) == ViewGather.Viewing{right_context,axes,offset,[]} : ViewGather.Viewing<Right>},
  +right_model: Model.Model<Element>,+right_shape: List<&2,U32>,+right_fill: Element,+right_workers: U32,
  +extents: {Views.axis_extents(axes) == shape : List<&2,U32>},+total_exact: {total_of(scope) == ShapeLaws.elements(shape) : Nat},
  +room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(right_shape)) == True{} : Bool},
  +right_contract: Global.Contract(~Element,~Right,~right_value,right_context,Tensors.Tensor{Model.pack(~Element,right_model),right_shape,right_fill,Tensors.Usable{},right_workers}),
  +left_contract: Global.Contract(~Element,~Left,~left_value,left_of(scope),Tensors.Tensor{Model.pack(~Element,left),shape,fill,Tensors.Usable{},U32.max(workers,right_workers)}),
  +observation: Buffer.Observation<Element,Tensors.joined(~Element,Tensors.count_or_zero(right_shape),right_fill,Model.pack(~Element,right_model))>) ->
  Global.Shaped(~Element,~After,~after_value,after_of(scope),shape,Tensors.combine_spread(~Element,~Later,~kernel,~after,later_of(scope),operations,U32.max(workers,right_workers),shape,Model.pack(~Element,left),fill,Tensors.joined(~Element,Tensors.count_or_zero(right_shape),right_fill,Model.pack(~Element,right_model)),right_shape,axes,offset)):
  KernelGemm.Both{KernelGemm.Both{+formed,KernelGemm.Both{+counted,+canonical}},+values} = right_contract
  Buffer.Observation{+kept,+equation} = observation
  +count = Global.count(right_shape,counted)
  +width = Global.whole_width(right_shape,count,Global.accepted(right_shape,counted))
  +evidence = Global.evidence(right_shape,counted)
  +exact = Count.count_exact(ShapeLaws.elements(right_shape),count,evidence)
  +back_width = Equal.sym(U32,Tensors.count_or_zero(right_shape),count,width)
  +joined_exact = {%back_width : {ShapeLaws.elements(right_shape) == U32.to_nat(_) : Nat} exact :
    {ShapeLaws.elements(right_shape) == U32.to_nat(Tensors.count_or_zero(right_shape)) : Nat}}
  +facts = Joined.joined_facts(~Element,~Right,~right_value,right_context,right_model,right_shape,right_fill,right_workers,right_contract,kept,equation)
  +right_formed = Held.first_of(Contract.WellFormed<Element,right_shape,Buffer.content(~Element,kept)>,Contract.Values(~Element,~Right,~right_value,right_context,Buffer.content(~Element,kept)),facts)
  +right_values = Held.second_of(Contract.WellFormed<Element,right_shape,Buffer.content(~Element,kept)>,Contract.Values(~Element,~Right,~right_value,right_context,Buffer.content(~Element,kept)),facts)
  +right_room = StorageCount.counted_room(limits,right_shape,counted,default_capacity)
  +fits_elements = {%Equal.sym(Nat,ShapeLaws.elements(right_shape),U32.to_nat(Tensors.count_or_zero(right_shape)),joined_exact) : {Views.fits(Views.View{axes,offset},U32.from_nat(_)) == True{} : Bool}
    %Equal.sym(U32,U32.from_nat(U32.to_nat(Tensors.count_or_zero(right_shape))),Tensors.count_or_zero(right_shape),Arithmetic.native_roundtrip(Tensors.count_or_zero(right_shape))) : {Views.fits(Views.View{axes,offset},_) == True{} : Bool}
    fits : {Views.fits(Views.View{axes,offset},U32.from_nat(ShapeLaws.elements(right_shape))) == True{} : Bool}}
  %equation : Global.Shaped(~Element,~After,~after_value,after_of(scope),shape,Tensors.combine_spread(~Element,~Later,~kernel,~after,later_of(scope),operations,U32.max(workers,right_workers),shape,Model.pack(~Element,left),fill,_,right_shape,axes,offset))
  KernelGemm.Both{spread_usable(~Element,~Later,~kernel,~after,later_of(scope),Tasks.decide(Tensors.grid_of(shape),U32.max(workers,right_workers),Tensors.pass_rows(2.0,operations),Tensors.pass_steps(),Tasks.Extra{(U32.to_f32(Tensors.count_or_zero(right_shape)) * 4.0 : F32),0.0,0.0}),U32.max(workers,right_workers),shape,Model.pack(~Element,left),fill,
      Buffer.pack(~Element,kept),axes,offset),
    spread_combined(~Element,~Later,~operation,~after_step,~Left,~left_value,~Right,~right_value,~After,~after_value,~Scope,~later_of,~left_of,~right_of,~after_of,~total_of,~rule,~kernel,~kernel_law,~after,~after_rule,scope,operations,left,fill,U32.max(workers,right_workers),shape,limits,axes,offset,right_context,viewing,right_shape,kept,extents,total_exact,room,nonempty,rank_room,
    right_formed,right_room,fits_elements,right_values,left_contract)}

# A left operand combined with a right operand read through a view of it with
# the left extents: every task combines its cells of the left with the right
# read through the view.
def viewed_combined(~Element: Data,~Later: Data,~operation: Element -> Element -> Element,~after_step: @+later: Later -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,~Left: Data,~left_value: Left -> U32 -> Element,~Right: Data,~right_value: Right -> U32 -> Element,~After: Data,~after_value: After -> U32 -> Element,~Scope: Data,~later_of: @+scope: Scope -> Later,~left_of: @+scope: Scope -> Left,~right_of: @+scope: Scope -> ViewGather.Viewing<Right>,~after_of: @+scope: Scope -> After,~total_of: @+scope: Scope -> Nat,~rule: PiecesCombine.Rule(~Element,~Later,~after_step,~operation,~Left,~left_value,~ViewGather.Viewing<Right>,~ViewGather.viewed(~Element,~Right,~right_value),~After,~after_value,~Scope,~later_of,~left_of,~right_of,~after_of,~total_of),~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~after: Tensors.PieceMap(~Element,~Later),~after_rule: PiecesMap.Rule(~Element,~Later,~after,~after_step),+scope: Scope,+operations: F32,+left: Model.Model<Element>,+fill: Element,+workers: U32,+shape: List<&2,U32>,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+axes: List<&2,Views.Axis>,+offset: U32,+right_context: Right,
  +viewing: {right_of(scope) == ViewGather.Viewing{right_context,axes,offset,[]} : ViewGather.Viewing<Right>},
  +right_model: Model.Model<Element>,+right_shape: List<&2,U32>,+right_fill: Element,+right_workers: U32,
  +extents: {Views.axis_extents(axes) == shape : List<&2,U32>},+total_exact: {total_of(scope) == ShapeLaws.elements(shape) : Nat},
  +room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(right_shape)) == True{} : Bool},
  +right_contract: Global.Contract(~Element,~Right,~right_value,right_context,Tensors.Tensor{Model.pack(~Element,right_model),right_shape,right_fill,Tensors.Usable{},right_workers}),
  +left_contract: Global.Contract(~Element,~Left,~left_value,left_of(scope),Tensors.Tensor{Model.pack(~Element,left),shape,fill,Tensors.Usable{},workers})) ->
  Global.Shaped(~Element,~After,~after_value,after_of(scope),shape,Tensors.combine_viewed_with(~Element,~Later,~kernel,~after,later_of(scope),operations,
    Tensors.Tensor{Model.pack(~Element,left),shape,fill,Tensors.Usable{},workers},Tensors.Tensor{Model.pack(~Element,right_model),right_shape,right_fill,Tensors.Usable{},right_workers},
    axes,offset)):
  joined_spread(~Element,~Later,~operation,~after_step,~Left,~left_value,~Right,~right_value,~After,~after_value,~Scope,~later_of,~left_of,~right_of,~after_of,~total_of,~rule,~kernel,~kernel_law,~after,~after_rule,scope,operations,left,fill,workers,shape,limits,default_capacity,axes,offset,right_context,viewing,
    right_model,right_shape,right_fill,right_workers,extents,total_exact,room,nonempty,rank_room,fits,right_contract,left_contract,
    Buffer.observe(~Element,Tensors.joined(~Element,Tensors.count_or_zero(right_shape),right_fill,Model.pack(~Element,right_model))))
