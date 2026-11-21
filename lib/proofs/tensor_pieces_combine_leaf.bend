# The binary piece leaf: storage guards, the checked traversal it refines,
# ordered readback and pointwise semantics.
import Base
import ../tensor.bend as Tensors
import ../tensor_pieces.bend as Pieces
import ./storage_tree_model.bend as Tree
import ./tensor_pieces_model.bend as Model
import ./tensor_pieces_content.bend as Content
import ./tensor_pieces_map.bend as PiecesMap
import ./nat_order.bend as Order
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./word_encoding.bend as WordEncoding
import ./word_power_of_two.bend as Power
import ./u32_comparison.bend as Comparison
import ../storage_buffer.bend as Storage
import ../tensor_operations.bend as Operations
import ../tensor_view.bend as Views
import ./tensor_access_refinement.bend as AccessLaws
import ./tensor_operations_refinement.bend as OperationsLaws
import ./tensor_vector_view_refinement.bend as Vector
import ../tensor_shape.bend as Shapes
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_division.bend as Division
import ./nat_interval.bend as Interval
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./storage_certificate.bend as Certificate
import ./storage_readback.bend as StorageReadback
import ./storage_refinement.bend as StorageLaws
import ./tensor_pieces_each.bend as Each
import ./traversal_array_refinement.bend as ArrayLaws
import ./traversal_map_readback.bend as MapReadback
import ./u32_full_range.bend as FullRange
import ./ordered_ranges.bend as Ranges
import ./sequence_membership.bend as Members
import ../tensor_access.bend as Access
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ../kernel_gemm.bend as KernelGemm

def certificate(~Element: Data,input: Buffer.Model<Element>) -> Certificate.Certificate<Element,Tree.pack(Element,Buffer.tree(~Element,input))>:
  Buffer.Model{tree,length,certificate} = input
  certificate

def valid(~Element: Data,+count: U32,+input: Buffer.Model<Element>,exact: {U32.to_nat(count) == Buffer.length(~Element,input) : Nat},ready: Buffer.Sized<Element,input>,
  fits: {Views.fits(Vector.vector(count),count) == True{} : Bool}) ->
  {Operations.valid_buffer(Buffer.length(~Element,input),Storage.capacity(~Element,Tree.pack(Element,Buffer.tree(~Element,input)),
      certificate(~Element,input)),Vector.vector(count)) == True{} : Bool}:
  Buffer.Model{+tree,+length,Certificate.Certificate{+depth,+equation}} = input
  Buffer.Sized{depth_ok,+room} = ready
  +model = {Buffer.Model{tree,length,Certificate.Certificate{depth,equation}} : Buffer.Model<Element>}
  %Shape.capacity(Element,depth,tree,Buffer.balanced(~Element,model)) :
    {Operations.valid_buffer(length,_,Vector.vector(count)) == True{} : Bool}
  %Equal.sym(U32,U32.from_nat(length),count,Arithmetic.from_nat_exact(count,length,exact)) :
    {Nat.is_le(length,U32.to_nat(Tree.capacity(Element,tree))) && Views.fits(Vector.vector(count),_) == True{} : Bool}
  %Equal.sym(Bool,Nat.is_le(length,U32.to_nat(Tree.capacity(Element,tree))),True{},room) :
    {_ && Views.fits(Vector.vector(count),count) == True{} : Bool}
  %Equal.sym(Bool,Views.fits(Vector.vector(count),count),True{},fits) : {True{} && _ == True{} : Bool}
  {==}

# Binary piece operations reuse the checked elementwise traversal refinement.
# Both scheduling modes have the same tree denotation, including rejection
# and malformed-shape ownership fallbacks. Logical Values are proved separately.
def left_content(~Element: Data,result: Operations.Outcome<OperationsLaws.PairOwners(Element)>) -> Buffer.Content<Element>:
  match result:
    case Operations.Accepted{(OperationsLaws.Owner{storage,length},right)}:
      Buffer.Content{Tree.reflect(Element,storage),length}
    case Operations.Rejected{(OperationsLaws.Owner{storage,length},right)}:
      Buffer.Content{Tree.reflect(Element,storage),length}

def selected_content(~Element: Data,result: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) ->
  {Buffer.contents(~Element,Tensors.combined(~Element,result)) == left_content(~Element,OperationsLaws.observe_pair_outcome(~Element,result)) : Buffer.Content<Element>}:
  match result:
    case Operations.Accepted{(Storage.Buffer{storage,length,certificate},right)}: {==}
    case Operations.Rejected{(Storage.Buffer{storage,length,certificate},right)}: {==}

def refinement_leaf_model(~Element: Data,~operation: Element -> Element -> Element,+count: U32,left: Buffer.Model<Element>,right: Buffer.Model<Element>) -> Buffer.Content<Element>:
  +view = {Views.View{[Views.Axis{count,1}],0} : Views.View}
  left_content(~Element,OperationsLaws.combine_denotation(~Element,~operation,Buffer.pack(~Element,left),view,Buffer.pack(~Element,right),view))

def refinement_leaf(~Element: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),+count: U32,+left: Buffer.Model<Element>,+right: Buffer.Model<Element>) ->
  {Buffer.contents(~Element,Tensors.combined(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,left),Views.View{[Views.Axis{count,1}],0},Buffer.pack(~Element,right),Views.View{[Views.Axis{count,1}],0}))) ==
    refinement_leaf_model(~Element,~operation,count,left,right) : Buffer.Content<Element>}:
  +view = {Views.View{[Views.Axis{count,1}],0} : Views.View}
  Equal.trans(Buffer.Content<Element>,
    Buffer.contents(~Element,Tensors.combined(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,left),view,Buffer.pack(~Element,right),view))),
    left_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,left),view,Buffer.pack(~Element,right),view))),
    refinement_leaf_model(~Element,~operation,count,left,right),
    selected_content(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,left),view,Buffer.pack(~Element,right),view)),
    Equal.cong(Operations.Outcome<OperationsLaws.PairOwners(Element)>,Buffer.Content<Element>,result => left_content(~Element,result),
      OperationsLaws.observe_pair_outcome(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,left),view,Buffer.pack(~Element,right),view)),
      OperationsLaws.combine_denotation(~Element,~operation,Buffer.pack(~Element,left),view,Buffer.pack(~Element,right),view),
      OperationsLaws.combine(~Element,~operation,~kernel,~kernel_law,Buffer.pack(~Element,left),view,Buffer.pack(~Element,right),view)))

# A pair of pieces of one range: the combination in the left piece, then the
# step after it.
def piece_model(~Element: Data,~Later: Data,~operation: Element -> Element -> Element,~after_step: @+later: Later -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,
  later: Later,first: U32,count: U32,left: Buffer.Model<Element>,right: Buffer.Model<Element>) -> Model.Content<Element>:
  +first = first
  +count = count
  Model.PieceContent{after_step(later,first,count,refinement_leaf_model(~Element,~operation,count,left,right)),first,count}

def model(~Element: Data,~Later: Data,~operation: Element -> Element -> Element,~after_step: @+later: Later -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,
  +later: Later,left: Model.Model<Element>,right: Model.Model<Element>) -> Model.Content<Element>:
  Each.content_zip(~Element,~Later,~piece_model(~Element,~Later,~operation,~after_step),later,left,right)

def piece(~Element: Data,~Later: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),
  ~after: Tensors.PieceMap(~Element,~Later),~after_step: @+later: Later -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,~after_rule: PiecesMap.Rule(~Element,~Later,~after,~after_step),
  +later: Later,+first: U32,+count: U32,+left: Buffer.Model<Element>,+right: Buffer.Model<Element>) ->
  {Model.contents(~Element,Tensors.combine_buffers(~Element,~Later,~kernel,~after,later,first,count,Buffer.pack(~Element,left),Buffer.pack(~Element,right),Unit{})) ==
    piece_model(~Element,~Later,~operation,~after_step,later,first,count,left,right) : Model.Content<Element>}:
  +view = {Views.View{[Views.Axis{count,1}],0} : Views.View}
  Equal.cong(Buffer.Content<Element>,Model.Content<Element>,buffer => Model.PieceContent{buffer,first,count},
    Buffer.contents(~Element,after(later,first,count,Tensors.combined(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,left),view,Buffer.pack(~Element,right),view)))),after_step(later,first,count,refinement_leaf_model(~Element,~operation,count,left,right)),
    Equal.trans(Buffer.Content<Element>,Buffer.contents(~Element,after(later,first,count,Tensors.combined(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,left),view,Buffer.pack(~Element,right),view)))),after_step(later,first,count,Buffer.contents(~Element,Tensors.combined(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,left),view,Buffer.pack(~Element,right),view)))),
      after_step(later,first,count,refinement_leaf_model(~Element,~operation,count,left,right)),after_rule(later,first,count,Tensors.combined(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,left),view,Buffer.pack(~Element,right),view))),
      Equal.cong(Buffer.Content<Element>,Buffer.Content<Element>,content => after_step(later,first,count,content),Buffer.contents(~Element,Tensors.combined(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,left),view,Buffer.pack(~Element,right),view))),
        refinement_leaf_model(~Element,~operation,count,left,right),refinement_leaf(~Element,~operation,~kernel,~kernel_law,count,left,right))))

# The combination of two trees of one shape on any task plan, run in parallel
# or not.
def combine(~Element: Data,~Later: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),
  ~after: Tensors.PieceMap(~Element,~Later),~after_step: @+later: Later -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,~after_rule: PiecesMap.Rule(~Element,~Later,~after,~after_step),
  +later: Later,+tasks: Pieces.Ranges,+parallel: Bool,+left: Model.Model<Element>,+right: Model.Model<Element>,+common: Each.Common(~Element,left,right)) ->
  {Model.contents(~Element,Pieces.grouped(~Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~Storage.Buffer<Element>,~Later,
      ~Pieces.paired(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Later,~Tensors.combine_buffers(~Element,~Later,~kernel,~after)),tasks,parallel,later,
      Pieces.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,left),Model.pack(~Element,right)))) ==
    model(~Element,~Later,~operation,~after_step,later,left,right) : Model.Content<Element>}:
  %Equal.sym(Pieces.Pieces(~Element),
    Pieces.grouped(~Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~Storage.Buffer<Element>,~Later,
      ~Pieces.paired(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Later,~Tensors.combine_buffers(~Element,~Later,~kernel,~after)),tasks,parallel,later,
      Pieces.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,left),Model.pack(~Element,right))),
    Each.Zipped(~Element,~Later,~Tensors.combine_buffers(~Element,~Later,~kernel,~after),Model.layout(~Element,left),parallel,later,Model.pack(~Element,left),Model.pack(~Element,right)),
    Each.grouped_pairs(~Element,~Storage.Buffer<Element>,~Later,~Pieces.paired(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Later,~Tensors.combine_buffers(~Element,~Later,~kernel,~after)),
      ~Pieces.left_of(~Storage.Buffer<Element>,~Storage.Buffer<Element>),tasks,parallel,later,left,right,common)) :
    {Model.contents(~Element,_) == model(~Element,~Later,~operation,~after_step,later,left,right) : Model.Content<Element>}
  Each.zipped(~Element,~Later,~Tensors.combine_buffers(~Element,~Later,~kernel,~after),~piece_model(~Element,~Later,~operation,~after_step),
    ~piece(~Element,~Later,~operation,~kernel,~kernel_law,~after,~after_step,~after_rule),left,right,parallel,later)

# The combination keeps the layout of the left tree, also where the two trees
# differ in shape.
def model_layout(~Element: Data,~Later: Data,~operation: Element -> Element -> Element,~after_step: @+later: Later -> @+first: U32 -> @+count: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,
  +later: Later,left: Model.Model<Element>,right: Model.Model<Element>) ->
  {Model.content_layout(~Element,model(~Element,~Later,~operation,~after_step,later,left,right)) == Model.layout(~Element,left) : Pieces.Ranges}:
  match left right:
    case Model.Leaf{buffer,first,rows} Model.Leaf{other,other_first,other_rows}: {==}
    case Model.Fork{+left,+right} Model.Fork{+other_left,+other_right}:
      %Equal.sym(Pieces.Ranges,Model.content_layout(~Element,model(~Element,~Later,~operation,~after_step,later,left,other_left)),Model.layout(~Element,left),
        model_layout(~Element,~Later,~operation,~after_step,later,left,other_left)) :
        {Pieces.Split{_,Model.content_layout(~Element,model(~Element,~Later,~operation,~after_step,later,right,other_right))} == Pieces.Split{Model.layout(~Element,left),Model.layout(~Element,right)} : Pieces.Ranges}
      %Equal.sym(Pieces.Ranges,Model.content_layout(~Element,model(~Element,~Later,~operation,~after_step,later,right,other_right)),Model.layout(~Element,right),
        model_layout(~Element,~Later,~operation,~after_step,later,right,other_right)) :
        {Pieces.Split{Model.layout(~Element,left),_} == Pieces.Split{Model.layout(~Element,left),Model.layout(~Element,right)} : Pieces.Ranges}
      {==}
    case Model.Leaf{+buffer,+first,+rows} Model.Fork{other_left,other_right}:
      Equal.sym(Pieces.Ranges,Model.layout(~Element,Model.Leaf{buffer,first,rows}),Model.content_layout(~Element,Model.content(~Element,Model.Leaf{buffer,first,rows})),
        Model.layout_matches_content(~Element,Model.Leaf{buffer,first,rows}))
    case Model.Fork{+left,+right} Model.Leaf{buffer,first,rows}:
      Equal.sym(Pieces.Ranges,Model.layout(~Element,Model.Fork{left,right}),Model.content_layout(~Element,Model.content(~Element,Model.Fork{left,right})),
        Model.layout_matches_content(~Element,Model.Fork{left,right}))

# Ordered in-place binary readback. A store changes only its own address;
# later addresses still contain their original left values. The right reader
# is immutable in this tree model. No algebraic law on operation is assumed.
def readback_reader(~Reader: Data,~Element: Data,result: AccessLaws.ModelResult<Reader,Element>) -> Tree.Tree<Element>:
  AccessLaws.ModelResult{reader,output} = result
  output

def readback_output(~Element: Data,result: AccessLaws.ModelResult<Tree.Tree<Element>,Element>) -> Tree.Tree<Element>:
  readback_reader(~Tree.Tree<Element>,~Element,result)

# The same readback for an update that samples any reader: the store at a
# position combines the output there with the sample of the reader there.
def updated(~Element: Data,~Reader: Data,~Sample: Data,~sample: Reader -> Unit -> U32 -> Sample,~operation: Element -> Sample -> Element,count: Nat,position: U32,reader: Reader,left: Tree.Tree<Element>) -> Tree.Tree<Element>:
  readback_reader(~Reader,~Element,AccessLaws.finish_model(~Reader,~Element,~Unit,
    Loop.run(~Traversal.State<&2,&2,Reader,Tree.Tree<Element>,Unit>,
      ~ArrayLaws.model_update_step(~Reader,~Element,~Sample,~Unit,~sample,~operation,~Traversal.index),
      count,Traversal.State{reader,left,position,Unit{}})))

def updated_capacity(~Element: Data,~Reader: Data,~Sample: Data,~sample: Reader -> Unit -> U32 -> Sample,~operation: Element -> Sample -> Element,count: Nat,+position: U32,+reader: Reader,+left: Tree.Tree<Element>) ->
  {Tree.capacity(Element,updated(~Element,~Reader,~Sample,~sample,~operation,count,position,reader,left)) == Tree.capacity(Element,left) : U32}:
  match count:
    case 0n: {==}
    case 1n+rest:
      +value = operation(Tree.read(Element,left,position),sample(reader,Unit{},position))
      +written = Tree.write(Element,left,position,value)
      Equal.trans(U32,Tree.capacity(Element,updated(~Element,~Reader,~Sample,~sample,~operation,rest,(position + 1 : U32),reader,written)),Tree.capacity(Element,written),Tree.capacity(Element,left),
        updated_capacity(~Element,~Reader,~Sample,~sample,~operation,rest,(position + 1 : U32),reader,written),StorageReadback.write_capacity(~Element,left,position,value))

def updated_balanced(~Element: Data,~Reader: Data,~Sample: Data,~sample: Reader -> Unit -> U32 -> Sample,~operation: Element -> Sample -> Element,count: Nat,+position: U32,+reader: Reader,+left: Tree.Tree<Element>,+depth: Nat,
  evidence: Shape.Balanced(Element,depth,left)) -> Shape.Balanced(Element,depth,updated(~Element,~Reader,~Sample,~sample,~operation,count,position,reader,left)):
  match count:
    case 0n: evidence
    case 1n+rest:
      +value = operation(Tree.read(Element,left,position),sample(reader,Unit{},position))
      updated_balanced(~Element,~Reader,~Sample,~sample,~operation,rest,(position + 1 : U32),reader,Tree.write(Element,left,position,value),depth,Shape.write(Element,depth,left,position,value,evidence))

def updated_outside(~Element: Data,~Reader: Data,~Sample: Data,~sample: Reader -> Unit -> U32 -> Sample,~operation: Element -> Sample -> Element,count: Nat,+position: Nat,+query: U32,+reader: Reader,+left: Tree.Tree<Element>,+depth: Nat,+capacity: U32,
  +capacity_equal: {Tree.capacity(Element,left) == capacity : U32},+balanced: Shape.Balanced(Element,depth,left),+limit: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(position,count),U32.to_nat(capacity)) == True{} : Bool},
  +query_inside: {U32.is_lt(query,capacity) == True{} : Bool},+separate: Interval.Outside<position,count,U32.to_nat(query)>) ->
  {Tree.read(Element,updated(~Element,~Reader,~Sample,~sample,~operation,count,U32.from_nat(position),reader,left),query) == Tree.read(Element,left,query) : Element}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +value = operation(Tree.read(Element,left,U32.from_nat(position)),sample(reader,Unit{},U32.from_nat(position)))
      +written = Tree.write(Element,left,U32.from_nat(position),value)
      +position_inside = Order.room_index(position,rest,U32.to_nat(capacity),room)
      +different = {
        %Equal.sym(Bool,U32.is_eq(U32.from_nat(position),query),Nat.is_eq(U32.to_nat(U32.from_nat(position)),U32.to_nat(query)),Comparison.eq(U32.from_nat(position),query)) : {_ == False{} : Bool}
        %Equal.sym(Nat,U32.to_nat(U32.from_nat(position)),position,Bounds.roundtrip(position,capacity,capacity_ok,Bounds.strict_le(position,U32.to_nat(capacity),position_inside))) :
          {Nat.is_eq(_,U32.to_nat(query)) == False{} : Bool}
        %Order.equal_symmetric(U32.to_nat(query),position) : {_ == False{} : Bool}
        Interval.outside_head(position,rest,U32.to_nat(query),separate) : {U32.is_eq(U32.from_nat(position),query) == False{} : Bool}}
      %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.add_one(U32.from_nat(position))) :
        {Tree.read(Element,updated(~Element,~Reader,~Sample,~sample,~operation,rest,_,reader,written),query) == Tree.read(Element,left,query) : Element}
      Equal.trans(Element,Tree.read(Element,updated(~Element,~Reader,~Sample,~sample,~operation,rest,U32.from_nat(1n+position),reader,written),query),Tree.read(Element,written,query),Tree.read(Element,left,query),
        updated_outside(~Element,~Reader,~Sample,~sample,~operation,rest,1n+position,query,reader,written,depth,capacity,
          Equal.trans(U32,Tree.capacity(Element,written),Tree.capacity(Element,left),capacity,StorageReadback.write_capacity(~Element,left,U32.from_nat(position),value),capacity_equal),
          Shape.write(Element,depth,left,U32.from_nat(position),value,balanced),limit,capacity_ok,Order.advance_room(position,rest,U32.to_nat(capacity),room),query_inside,
          Interval.outside_tail(position,rest,U32.to_nat(query),separate)),
        MapReadback.unchanged(~Element,depth,left,U32.from_nat(position),query,value,capacity,capacity_equal,balanced,limit,MapReadback.address_inside(position,capacity,capacity_ok,position_inside),query_inside,different))

def updated_inside(~Element: Data,~Reader: Data,~Sample: Data,~sample: Reader -> Unit -> U32 -> Sample,~operation: Element -> Sample -> Element,count: Nat,+position: Nat,index: Nat,+reader: Reader,+left: Tree.Tree<Element>,+depth: Nat,+capacity: U32,
  +capacity_equal: {Tree.capacity(Element,left) == capacity : U32},+balanced: Shape.Balanced(Element,depth,left),+limit: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(position,count),U32.to_nat(capacity)) == True{} : Bool},
  +selected: {Nat.is_lt(index,count) == True{} : Bool}) ->
  {Tree.read(Element,updated(~Element,~Reader,~Sample,~sample,~operation,count,U32.from_nat(position),reader,left),U32.from_nat(Nat.add(position,index))) ==
    operation(Tree.read(Element,left,U32.from_nat(Nat.add(position,index))),sample(reader,Unit{},U32.from_nat(Nat.add(position,index)))) : Element}:
  match count index:
    case 0n +index:
      Empty.absurd({Tree.read(Element,left,U32.from_nat(Nat.add(position,index))) ==
        operation(Tree.read(Element,left,U32.from_nat(Nat.add(position,index))),sample(reader,Unit{},U32.from_nat(Nat.add(position,index)))) : Element},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(index,0n),True{},Equal.sym(Bool,Nat.is_lt(index,0n),False{},Division.lt_zero(index)),selected)))
    case 1n+ +rest 0n:
      +value = operation(Tree.read(Element,left,U32.from_nat(position)),sample(reader,Unit{},U32.from_nat(position)))
      +written = Tree.write(Element,left,U32.from_nat(position),value)
      +position_inside = Order.room_index(position,rest,U32.to_nat(capacity),room)
      %Equal.sym(Nat,Nat.add(position,0n),position,Addition.zero_right(position)) :
        {Tree.read(Element,updated(~Element,~Reader,~Sample,~sample,~operation,1n+rest,U32.from_nat(position),reader,left),U32.from_nat(_)) ==
          operation(Tree.read(Element,left,U32.from_nat(_)),sample(reader,Unit{},U32.from_nat(_))) : Element}
      %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.add_one(U32.from_nat(position))) :
        {Tree.read(Element,updated(~Element,~Reader,~Sample,~sample,~operation,rest,_,reader,written),U32.from_nat(position)) == value : Element}
      Equal.trans(Element,Tree.read(Element,updated(~Element,~Reader,~Sample,~sample,~operation,rest,U32.from_nat(1n+position),reader,written),U32.from_nat(position)),Tree.read(Element,written,U32.from_nat(position)),value,
        updated_outside(~Element,~Reader,~Sample,~sample,~operation,rest,1n+position,U32.from_nat(position),reader,written,depth,capacity,
          Equal.trans(U32,Tree.capacity(Element,written),Tree.capacity(Element,left),capacity,StorageReadback.write_capacity(~Element,left,U32.from_nat(position),value),capacity_equal),
          Shape.write(Element,depth,left,U32.from_nat(position),value,balanced),limit,capacity_ok,Order.advance_room(position,rest,U32.to_nat(capacity),room),MapReadback.address_inside(position,capacity,capacity_ok,position_inside),
          Interval.Before{
            %Equal.sym(Nat,U32.to_nat(U32.from_nat(position)),position,Bounds.roundtrip(position,capacity,capacity_ok,Bounds.strict_le(position,U32.to_nat(capacity),position_inside))) : {Nat.is_lt(_,1n+position) == True{} : Bool}
            Order.lt_succ(position,position,Order.reflexive(position))}),
        StorageReadback.read_written(~Element,left,U32.from_nat(position),value))
    case 1n+ +rest 1n+ +index:
      +value = operation(Tree.read(Element,left,U32.from_nat(position)),sample(reader,Unit{},U32.from_nat(position)))
      +written = Tree.write(Element,left,U32.from_nat(position),value)
      +query_position = Nat.add(1n+position,index)
      +query = U32.from_nat(query_position)
      +position_inside = Order.room_index(position,rest,U32.to_nat(capacity),room)
      +query_inside = Bounds.nat_trans(query_position,Nat.add(1n+position,rest),U32.to_nat(capacity),Order.offset_lt(1n+position,index,rest,selected),Order.advance_room(position,rest,U32.to_nat(capacity),room))
      +different = {
        %Equal.sym(Bool,U32.is_eq(U32.from_nat(position),query),Nat.is_eq(U32.to_nat(U32.from_nat(position)),U32.to_nat(query)),Comparison.eq(U32.from_nat(position),query)) : {_ == False{} : Bool}
        %Equal.sym(Nat,U32.to_nat(U32.from_nat(position)),position,Bounds.roundtrip(position,capacity,capacity_ok,Bounds.strict_le(position,U32.to_nat(capacity),position_inside))) : {Nat.is_eq(_,U32.to_nat(query)) == False{} : Bool}
        %Equal.sym(Nat,U32.to_nat(query),query_position,Bounds.roundtrip(query_position,capacity,capacity_ok,Bounds.strict_le(query_position,U32.to_nat(capacity),query_inside))) : {Nat.is_eq(position,_) == False{} : Bool}
        Order.less_not_equal(position,query_position,Bounds.nat_trans(position,1n+position,query_position,Order.lt_succ(position,position,Order.reflexive(position)),Order.le_add(1n+position,index))) : {U32.is_eq(U32.from_nat(position),query) == False{} : Bool}}
      %Equal.sym(Nat,Nat.add(position,1n+index),query_position,Addition.successor_right(position,index)) :
        {Tree.read(Element,updated(~Element,~Reader,~Sample,~sample,~operation,1n+rest,U32.from_nat(position),reader,left),U32.from_nat(_)) ==
          operation(Tree.read(Element,left,U32.from_nat(_)),sample(reader,Unit{},U32.from_nat(_))) : Element}
      %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.add_one(U32.from_nat(position))) :
        {Tree.read(Element,updated(~Element,~Reader,~Sample,~sample,~operation,rest,_,reader,written),query) == operation(Tree.read(Element,left,query),sample(reader,Unit{},query)) : Element}
      Equal.trans(Element,Tree.read(Element,updated(~Element,~Reader,~Sample,~sample,~operation,rest,U32.from_nat(1n+position),reader,written),query),operation(Tree.read(Element,written,query),sample(reader,Unit{},query)),
        operation(Tree.read(Element,left,query),sample(reader,Unit{},query)),
        updated_inside(~Element,~Reader,~Sample,~sample,~operation,rest,1n+position,index,reader,written,depth,capacity,
          Equal.trans(U32,Tree.capacity(Element,written),Tree.capacity(Element,left),capacity,StorageReadback.write_capacity(~Element,left,U32.from_nat(position),value),capacity_equal),
          Shape.write(Element,depth,left,U32.from_nat(position),value,balanced),limit,capacity_ok,Order.advance_room(position,rest,U32.to_nat(capacity),room),selected),
        Equal.cong(Element,Element,element => operation(element,sample(reader,Unit{},query)),Tree.read(Element,written,query),Tree.read(Element,left,query),
          MapReadback.unchanged(~Element,depth,left,U32.from_nat(position),query,value,capacity,capacity_equal,balanced,limit,MapReadback.address_inside(position,capacity,capacity_ok,position_inside),MapReadback.address_inside(query_position,capacity,capacity_ok,query_inside),different)))

def run(~Element: Data,~operation: Element -> Element -> Element,count: Nat,position: U32,right: Tree.Tree<Element>,left: Tree.Tree<Element>) -> Tree.Tree<Element>:
  updated(~Element,~Tree.Tree<Element>,~Element,~AccessLaws.sample_index(~Element),~operation,count,position,right,left)

def capacity(~Element: Data,~operation: Element -> Element -> Element,count: Nat,+position: U32,+right: Tree.Tree<Element>,+left: Tree.Tree<Element>) ->
  {Tree.capacity(Element,run(~Element,~operation,count,position,right,left)) == Tree.capacity(Element,left) : U32}:
  updated_capacity(~Element,~Tree.Tree<Element>,~Element,~AccessLaws.sample_index(~Element),~operation,count,position,right,left)

def balanced(~Element: Data,~operation: Element -> Element -> Element,count: Nat,+position: U32,+right: Tree.Tree<Element>,+left: Tree.Tree<Element>,+depth: Nat,
  evidence: Shape.Balanced(Element,depth,left)) -> Shape.Balanced(Element,depth,run(~Element,~operation,count,position,right,left)):
  updated_balanced(~Element,~Tree.Tree<Element>,~Element,~AccessLaws.sample_index(~Element),~operation,count,position,right,left,depth,evidence)

def inside(~Element: Data,~operation: Element -> Element -> Element,count: Nat,+position: Nat,index: Nat,+right: Tree.Tree<Element>,+left: Tree.Tree<Element>,+depth: Nat,+capacity: U32,
  +capacity_equal: {Tree.capacity(Element,left) == capacity : U32},+balanced: Shape.Balanced(Element,depth,left),+limit: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(position,count),U32.to_nat(capacity)) == True{} : Bool},
  +selected: {Nat.is_lt(index,count) == True{} : Bool}) ->
  {Tree.read(Element,run(~Element,~operation,count,U32.from_nat(position),right,left),U32.from_nat(Nat.add(position,index))) ==
    operation(Tree.read(Element,left,U32.from_nat(Nat.add(position,index))),Tree.read(Element,right,U32.from_nat(Nat.add(position,index)))) : Element}:
  updated_inside(~Element,~Tree.Tree<Element>,~Element,~AccessLaws.sample_index(~Element),~operation,count,position,index,right,left,depth,capacity,capacity_equal,balanced,limit,capacity_ok,room,selected)

# Pointwise binary semantics over two pieces of one range. The operation keeps
# its exact left/right argument order; arbitrary Element operations are valid.
# Connecting the checked native leaf guard to this model is a separate step.
type Context<-Left: Data,-Right: Data> is Data:
  Context{left: Left,right: Right}

def value(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  context: Context<Left,Right>,index: U32) -> Element:
  Context{left,right} = context
  +index = index
  operation(left_value(left,index),right_value(right,index))

def leaf_content(~Element: Data,~operation: Element -> Element -> Element,left: Buffer.Content<Element>,right: Buffer.Content<Element>) -> Buffer.Content<Element>:
  Buffer.Content{left,+length} = left
  Buffer.Content{right,right_length} = right
  Buffer.Content{run(~Element,~operation,length,0,right,left),length}

def leaf_ready(~Element: Data,~operation: Element -> Element -> Element,+left: Buffer.Content<Element>,+right: Buffer.Content<Element>,ready: Buffer.ContentReady<Element,left>) ->
  Buffer.ContentReady<Element,leaf_content(~Element,~operation,left,right)>:
  Buffer.Content{+left_tree,+length} = left
  Buffer.Content{+right_tree,right_length} = right
  Buffer.ContentReady{+depth,+balanced_left,+depth_ok,+room} = ready
  Buffer.ContentReady{depth,balanced(~Element,~operation,length,0,right_tree,left_tree,depth,balanced_left),depth_ok,
    %Equal.sym(U32,Tree.capacity(Element,run(~Element,~operation,length,0,right_tree,left_tree)),Tree.capacity(Element,left_tree),
      capacity(~Element,~operation,length,0,right_tree,left_tree)) : {Nat.is_le(length,U32.to_nat(_)) == True{} : Bool}
    room}

# What two pieces of one range supply to every cell.
type CellScope<-Element: Data,-Left: Data,-Right: Data,-left_value: Left -> U32 -> Element,-right_value: Right -> U32 -> Element> is Data:
  CellScope{left_context: Left,right_context: Right,first: U32,count: U32,left_tree: Tree.Tree<Element>,length: Nat,right_tree: Tree.Tree<Element>,right_length: Nat,
    ready: Buffer.ContentReady<Element,Buffer.Content{left_tree,length}>,
    left_values: Content.Leaf(~Element,~Left,~left_value,left_context,first,count,left_tree,length),
    right_values: Content.Leaf(~Element,~Right,~right_value,right_context,first,count,right_tree,right_length)}

def cell_count(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,scope: CellScope<Element,Left,Right,left_value,right_value>) -> Nat:
  CellScope{left_context,right_context,first,count,left_tree,length,right_tree,right_length,ready,left_values,right_values} = scope
  U32.to_nat(count)

def Cell(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +scope: CellScope<Element,Left,Right,left_value,right_value>,+index: Nat) -> Data:
  CellScope{left_context,right_context,first,count,left_tree,length,right_tree,right_length,ready,left_values,right_values} = scope
  Content.At(~Element,~Context<Left,Right>,~value(~Element,~Left,~Right,~operation,~left_value,~right_value),Context{left_context,right_context},first,
    run(~Element,~operation,length,0,right_tree,left_tree),length,index)

# A combined cell stays at its index and applies the operation to the two
# values of the same cell.
def combined_cell(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+first: U32,+left_tree: Tree.Tree<Element>,+length: Nat,+right_tree: Tree.Tree<Element>,+right_length: Nat,
  +ready: Buffer.ContentReady<Element,Buffer.Content{left_tree,length}>,+index: Nat,
  left_cell: Content.At(~Element,~Left,~left_value,left_context,first,left_tree,length,index),
  right_cell: Content.At(~Element,~Right,~right_value,right_context,first,right_tree,right_length,index)) ->
  Content.At(~Element,~Context<Left,Right>,~value(~Element,~Left,~Right,~operation,~left_value,~right_value),Context{left_context,right_context},first,
    run(~Element,~operation,length,0,right_tree,left_tree),length,index):
  Buffer.ContentReady{+depth,+balanced_left,+depth_ok,+room} = ready
  KernelGemm.Both{+selected,+left_equal} = left_cell
  KernelGemm.Both{right_selected,+right_equal} = right_cell
  +position = U32.from_nat(index)
  +pixel = U32.from_nat(Nat.add(U32.to_nat(first),index))
  KernelGemm.Both{selected,
    Equal.trans(Element,Tree.read(Element,run(~Element,~operation,length,0,right_tree,left_tree),position),
      operation(Tree.read(Element,left_tree,position),Tree.read(Element,right_tree,position)),
      operation(left_value(left_context,pixel),right_value(right_context,pixel)),
      inside(~Element,~operation,length,0n,index,right_tree,left_tree,depth,Tree.capacity(Element,left_tree),{==},balanced_left,depth_ok,
        %Equal.sym(U32,Tree.capacity(Element,left_tree),U32.shln(1,depth),Shape.capacity(Element,depth,left_tree,balanced_left)) : {U32.is_le(_,2147483648) == True{} : Bool}
          Power.native_power_bound(depth,Order.successor(depth,30n,depth_ok)),
        room,selected),
      Equal.trans(Element,operation(Tree.read(Element,left_tree,position),Tree.read(Element,right_tree,position)),
        operation(left_value(left_context,pixel),Tree.read(Element,right_tree,position)),operation(left_value(left_context,pixel),right_value(right_context,pixel)),
        Equal.cong(Element,Element,element => operation(element,Tree.read(Element,right_tree,position)),Tree.read(Element,left_tree,position),left_value(left_context,pixel),left_equal),
        Equal.cong(Element,Element,element => operation(left_value(left_context,pixel),element),Tree.read(Element,right_tree,position),right_value(right_context,pixel),right_equal)))}

def leaf_cell(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +scope: CellScope<Element,Left,Right,left_value,right_value>,+index: Nat,+below: {Nat.is_lt(index,cell_count(~Element,~Left,~Right,~left_value,~right_value,scope)) == True{} : Bool}) ->
  Cell(~Element,~Left,~Right,~operation,~left_value,~right_value,scope,index):
  CellScope{+left_context,+right_context,+first,+count,+left_tree,+length,+right_tree,+right_length,+ready,+left_values,+right_values} = scope
  combined_cell(~Element,~Left,~Right,~operation,~left_value,~right_value,left_context,right_context,first,left_tree,length,right_tree,right_length,ready,index,
    Content.leaf_cell(~Element,~Left,~left_value,left_context,first,count,left_tree,length,index,left_values,below),
    Content.leaf_cell(~Element,~Right,~right_value,right_context,first,count,right_tree,right_length,index,right_values,below))

# Two pieces of one range holding two readers' values combine into the piece
# of the combined reader.
def semantics_leaf(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+first: U32,+count: U32,+left: Buffer.Content<Element>,+right: Buffer.Content<Element>,ready: Buffer.ContentReady<Element,left>,
  left_values: Content.piece_values(~Element,~Left,~left_value,left_context,first,count,left),
  right_values: Content.piece_values(~Element,~Right,~right_value,right_context,first,count,right)) ->
  Content.piece_values(~Element,~Context<Left,Right>,~value(~Element,~Left,~Right,~operation,~left_value,~right_value),Context{left_context,right_context},first,count,
    leaf_content(~Element,~operation,left,right)):
  Buffer.Content{+left_tree,+length} = left
  Buffer.Content{+right_tree,+right_length} = right
  Members.range_all(~CellScope<Element,Left,Right,left_value,right_value>,~Cell(~Element,~Left,~Right,~operation,~left_value,~right_value),
    ~cell_count(~Element,~Left,~Right,~left_value,~right_value),~leaf_cell(~Element,~Left,~Right,~operation,~left_value,~right_value),U32.to_nat(count),0n,
    CellScope{left_context,right_context,first,count,left_tree,length,right_tree,right_length,ready,left_values,right_values},Order.reflexive(U32.to_nat(count)))

# Connect the checked native binary leaf to its pointwise tree model. Its
# premises are the owners' Ready, exact local counts and the descriptor of the
# vector of the piece.
def same_shape(+count: U32) ->
  {Shapes.same(Views.shape(Vector.vector(count)),Views.shape(Vector.vector(count))) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_eq(count,count),True{},
    {%Equal.sym(Bool,U32.is_eq(count,count),Nat.is_eq(U32.to_nat(count),U32.to_nat(count)),Comparison.eq(count,count)) : {_ == True{} : Bool}
      Order.equal_reflexive(U32.to_nat(count)) : {U32.is_eq(count,count) == True{} : Bool}}) : {_ && True{} == True{} : Bool}
  {==}

def guard(~Element: Data,+count: U32,+left: Buffer.Model<Element>,+right: Buffer.Model<Element>) -> Bool:
  +view = Vector.vector(count)
  Shapes.same(Views.shape(view),Views.shape(view)) && Views.is_contiguous(view) &&
    Operations.valid_buffer(Buffer.length(~Element,left),Storage.capacity(~Element,Tree.pack(Element,Buffer.tree(~Element,left)),certificate(~Element,left)),view) &&
    Operations.valid_buffer(Buffer.length(~Element,right),Storage.capacity(~Element,Tree.pack(Element,Buffer.tree(~Element,right)),certificate(~Element,right)),view)

def descriptor_counted(+count: U32,descriptor: Vector.Descriptor<count>) -> {Views.numel(Vector.vector(count)) == Some{count} : Maybe<&2,U32>}:
  Vector.Descriptor{counted,contiguous,fits} = descriptor
  counted

def descriptor_contiguous(+count: U32,descriptor: Vector.Descriptor<count>) -> {Views.is_contiguous(Vector.vector(count)) == True{} : Bool}:
  Vector.Descriptor{counted,contiguous,fits} = descriptor
  contiguous

def call_guard(~Element: Data,+count: U32,+left: Buffer.Model<Element>,+right: Buffer.Model<Element>,
  left_exact: {U32.to_nat(count) == Buffer.length(~Element,left) : Nat},right_exact: {U32.to_nat(count) == Buffer.length(~Element,right) : Nat},
  left_ready: Buffer.Sized<Element,left>,right_ready: Buffer.Sized<Element,right>,+descriptor: Vector.Descriptor<count>) -> {guard(~Element,count,left,right) == True{} : Bool}:
  Vector.Descriptor{counted,+contiguous,+fits} = descriptor
  %Equal.sym(Bool,Shapes.same(Views.shape(Vector.vector(count)),Views.shape(Vector.vector(count))),True{},same_shape(count)) :
    {_ && Views.is_contiguous(Vector.vector(count)) &&
      Operations.valid_buffer(Buffer.length(~Element,left),Storage.capacity(~Element,Tree.pack(Element,Buffer.tree(~Element,left)),certificate(~Element,left)),Vector.vector(count)) &&
      Operations.valid_buffer(Buffer.length(~Element,right),Storage.capacity(~Element,Tree.pack(Element,Buffer.tree(~Element,right)),certificate(~Element,right)),Vector.vector(count)) == True{} : Bool}
  %Equal.sym(Bool,Views.is_contiguous(Vector.vector(count)),True{},contiguous) :
    {True{} && _ && Operations.valid_buffer(Buffer.length(~Element,left),Storage.capacity(~Element,Tree.pack(Element,Buffer.tree(~Element,left)),certificate(~Element,left)),Vector.vector(count)) &&
      Operations.valid_buffer(Buffer.length(~Element,right),Storage.capacity(~Element,Tree.pack(Element,Buffer.tree(~Element,right)),certificate(~Element,right)),Vector.vector(count)) == True{} : Bool}
  %Equal.sym(Bool,Operations.valid_buffer(Buffer.length(~Element,left),Storage.capacity(~Element,Tree.pack(Element,Buffer.tree(~Element,left)),certificate(~Element,left)),Vector.vector(count)),
    True{},valid(~Element,count,left,left_exact,left_ready,fits)) :
    {True{} && True{} && _ && Operations.valid_buffer(Buffer.length(~Element,right),Storage.capacity(~Element,Tree.pack(Element,Buffer.tree(~Element,right)),certificate(~Element,right)),Vector.vector(count)) == True{} : Bool}
  %Equal.sym(Bool,Operations.valid_buffer(Buffer.length(~Element,right),Storage.capacity(~Element,Tree.pack(Element,Buffer.tree(~Element,right)),certificate(~Element,right)),Vector.vector(count)),
    True{},valid(~Element,count,right,right_exact,right_ready,fits)) : {True{} && True{} && True{} && _ == True{} : Bool}
  {==}

def fast(+count: U32,contiguous: {Views.is_contiguous(Vector.vector(count)) == True{} : Bool}) -> {Access.zero_contiguous(Vector.vector(count)) && U32.is_zero(0) == True{} : Bool}:
  %Equal.sym(Bool,Views.is_contiguous(Vector.vector(count)),True{},contiguous) : {_ && True{} == True{} : Bool}
  {==}

def model_contents(~Element: Data,+left_length: Nat,+right_length: Nat,+result: AccessLaws.ModelResult<Tree.Tree<Element>,Element>) ->
  {left_content(~Element,Operations.Accepted{OperationsLaws.combined_pair(~Element,left_length,right_length,result)}) ==
    Buffer.Content{readback_output(~Element,result),left_length} : Buffer.Content<Element>}:
  AccessLaws.ModelResult{reader,+output} = result
  Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{tree,left_length},
    Tree.reflect(Element,Tree.pack(Element,output)),output,StorageLaws.reflect_packed(Element,output))

def accepted_content(~Element: Data,~operation: Element -> Element -> Element,+count: U32,+left_length: Nat,+right_length: Nat,+left: Tree.Tree<Element>,+right: Tree.Tree<Element>,
  contiguous: {Views.is_contiguous(Vector.vector(count)) == True{} : Bool}) ->
  {left_content(~Element,OperationsLaws.combine_model(~Element,~operation,
      OperationsLaws.PairCall{True{},Some{count},Vector.vector(count),Vector.vector(count),right_length,left_length},right,left)) ==
    Buffer.Content{run(~Element,~operation,U32.to_nat(count),0,right,left),left_length} : Buffer.Content<Element>}:
  %Equal.sym(Bool,Access.zero_contiguous(Vector.vector(count)) && U32.is_zero(0),True{},fast(count,contiguous)) :
    {left_content(~Element,Operations.Accepted{OperationsLaws.combined_pair(~Element,left_length,right_length,
      AccessLaws.combine_model(~Element,~operation,_,U32.to_nat(count),Vector.vector(count),0,right,left))}) ==
      Buffer.Content{run(~Element,~operation,U32.to_nat(count),0,right,left),left_length} : Buffer.Content<Element>}
  model_contents(~Element,left_length,right_length,AccessLaws.combine_model(~Element,~operation,True{},U32.to_nat(count),Vector.vector(count),0,right,left))

def checked_leaf_model(~Element: Data,~operation: Element -> Element -> Element,+count: U32,+left: Buffer.Model<Element>,+right: Buffer.Model<Element>,
  +left_exact: {U32.to_nat(count) == Buffer.length(~Element,left) : Nat},+right_exact: {U32.to_nat(count) == Buffer.length(~Element,right) : Nat},
  left_ready: Buffer.Sized<Element,left>,right_ready: Buffer.Sized<Element,right>,+descriptor: Vector.Descriptor<count>) ->
  {refinement_leaf_model(~Element,~operation,count,left,right) == leaf_content(~Element,~operation,Buffer.content(~Element,left),Buffer.content(~Element,right)) : Buffer.Content<Element>}:
  match left right:
    case Buffer.Model{+left_tree,+left_length,+left_certificate} Buffer.Model{+right_tree,+right_length,+right_certificate}:
      +left_model = {Buffer.Model{left_tree,left_length,left_certificate} : Buffer.Model<Element>}
      +right_model = {Buffer.Model{right_tree,right_length,right_certificate} : Buffer.Model<Element>}
      %Equal.sym(Bool,guard(~Element,count,left_model,right_model),True{},call_guard(~Element,count,left_model,right_model,left_exact,right_exact,left_ready,right_ready,descriptor)) :
        {left_content(~Element,OperationsLaws.combine_model(~Element,~operation,
          OperationsLaws.PairCall{_,Views.numel(Vector.vector(count)),Vector.vector(count),Vector.vector(count),right_length,left_length},
          Tree.reflect(Element,Tree.pack(Element,right_tree)),Tree.reflect(Element,Tree.pack(Element,left_tree)))) ==
          Buffer.Content{run(~Element,~operation,left_length,0,right_tree,left_tree),left_length} : Buffer.Content<Element>}
      %Equal.sym(Maybe<&2,U32>,Views.numel(Vector.vector(count)),Some{count},descriptor_counted(count,descriptor)) :
        {left_content(~Element,OperationsLaws.combine_model(~Element,~operation,
          OperationsLaws.PairCall{True{},_,Vector.vector(count),Vector.vector(count),right_length,left_length},Tree.reflect(Element,Tree.pack(Element,right_tree)),Tree.reflect(Element,Tree.pack(Element,left_tree)))) ==
          Buffer.Content{run(~Element,~operation,left_length,0,right_tree,left_tree),left_length} : Buffer.Content<Element>}
      %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,right_tree)),right_tree,StorageLaws.reflect_packed(Element,right_tree)) :
        {left_content(~Element,OperationsLaws.combine_model(~Element,~operation,
          OperationsLaws.PairCall{True{},Some{count},Vector.vector(count),Vector.vector(count),right_length,left_length},_,Tree.reflect(Element,Tree.pack(Element,left_tree)))) ==
          Buffer.Content{run(~Element,~operation,left_length,0,right_tree,left_tree),left_length} : Buffer.Content<Element>}
      %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,left_tree)),left_tree,StorageLaws.reflect_packed(Element,left_tree)) :
        {left_content(~Element,OperationsLaws.combine_model(~Element,~operation,
          OperationsLaws.PairCall{True{},Some{count},Vector.vector(count),Vector.vector(count),right_length,left_length},right_tree,_)) ==
          Buffer.Content{run(~Element,~operation,left_length,0,right_tree,left_tree),left_length} : Buffer.Content<Element>}
      %left_exact :
        {left_content(~Element,OperationsLaws.combine_model(~Element,~operation,
          OperationsLaws.PairCall{True{},Some{count},Vector.vector(count),Vector.vector(count),right_length,left_length},right_tree,left_tree)) ==
          Buffer.Content{run(~Element,~operation,_,0,right_tree,left_tree),left_length} : Buffer.Content<Element>}
      accepted_content(~Element,~operation,count,left_length,right_length,left_tree,right_tree,descriptor_contiguous(count,descriptor))
