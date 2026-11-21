# The shared piece recursion, by structure. A parallel split and a leaf task
# compute what the sequential recursion computes; every piece operation uses
# these equations instead of unfolding the recursion itself.
import Base
import ../tensor_pieces.bend as Pieces
import ../traversal_partition.bend as Partition
import ../storage_buffer.bend as Storage
import ./tensor_pieces_model.bend as Model
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./write_program.bend as Writes
import ./boolean_logic.bend as Boolean
import ./word_encoding.bend as WordEncoding
import ./tensor_pieces_walk.bend as Walk
import ./traversal_partition_refinement.bend as PartitionLaws
import ./tensor_pieces_plan.bend as PiecesPlan
import ./list_storage_readback.bend as ListRead
import ./word_arithmetic.bend as WordArithmetic
import ./row_major_index.bend as RowMajor
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ../kernel_gemm.bend as KernelGemm

def Tree(~Block: Type) -> Type:
  Partition.Tree<&1,Block,Unit>

# Both halves of a split, each with its own resource.
def forked(~Block: Type,~Output: Type,~Resource: Type,~Context: Data,
  ~share: Resource -> Resource & Resource,
  ~leaf: Context -> U32 -> U32 -> Block -> Resource -> Tree(~Output),
  ~keep: Tree(~Block) -> Tree(~Output),
  +left_plan: Pieces.Ranges,+right_plan: Pieces.Ranges,+enabled: Bool,+context: Context,
  left: Tree(~Block),right: Tree(~Block),resources: Resource & Resource) -> Tree(~Output):
  Walk.sequential(~Pieces.Pieced(~Block,~Resource),~Output,~Unit,~Context,~Bool,~Pieces.in_parallel,~Pieces.piece_divide(~Block,~Resource,~Context,~share),
    ~Pieces.piece_leaf(~Block,~Output,~Resource,~Context,~leaf,~keep),~Pieces.piece_kept(~Block,~Output,~Resource,~Context,~keep),
    left_plan,right_plan,enabled,context,Pieces.shared_halves(~Block,~Resource,left,right,resources))

# A split plan on a fork: both halves, each on its own plan. The parallel
# split and its leaf tasks equal the sequential recursion.
def fork(~Block: Type,~Output: Type,~Resource: Type,~Context: Data,
  ~share: Resource -> Resource & Resource,
  ~leaf: Context -> U32 -> U32 -> Block -> Resource -> Tree(~Output),
  ~keep: Tree(~Block) -> Tree(~Output),
  +left_plan: Pieces.Ranges,+right_plan: Pieces.Ranges,+enabled: Bool,+context: Context,
  -left: Tree(~Block),-right: Tree(~Block),-resource: Resource) ->
  {Pieces.each(~Block,~Output,~Resource,~Context,~share,~leaf,~keep,Pieces.Split{left_plan,right_plan},enabled,context,Partition.Fork{left,right},resource) ==
    forked(~Block,~Output,~Resource,~Context,~share,~leaf,~keep,left_plan,right_plan,enabled,context,left,right,share(resource)) : Tree(~Output)}:
  Walk.fork(~Pieces.Pieced(~Block,~Resource),~Output,~Unit,~Context,~Bool,~Pieces.in_parallel,~Pieces.piece_divide(~Block,~Resource,~Context,~share),
    ~Pieces.piece_leaf(~Block,~Output,~Resource,~Context,~leaf,~keep),~Pieces.piece_kept(~Block,~Output,~Resource,~Context,~keep),
    left_plan,right_plan,enabled,enabled,context,(Partition.Fork{left,right},resource))

# A split plan on one leaf keeps the leaf.
def split_leaf(~Block: Type,~Output: Type,~Resource: Type,~Context: Data,
  ~share: Resource -> Resource & Resource,
  ~leaf: Context -> U32 -> U32 -> Block -> Resource -> Tree(~Output),
  ~keep: Tree(~Block) -> Tree(~Output),
  +left_plan: Pieces.Ranges,+right_plan: Pieces.Ranges,+enabled: Bool,+context: Context,
  -block: Block,-first: U32,-rows: U32,-details: Unit,-resource: Resource) ->
  {Pieces.each(~Block,~Output,~Resource,~Context,~share,~leaf,~keep,Pieces.Split{left_plan,right_plan},enabled,context,Partition.Leaf{block,first,rows,details},resource) ==
    keep(Partition.Leaf{block,first,rows,details}) : Tree(~Output)}:
  Walk.fork(~Pieces.Pieced(~Block,~Resource),~Output,~Unit,~Context,~Bool,~Pieces.in_parallel,~Pieces.piece_divide(~Block,~Resource,~Context,~share),
    ~Pieces.piece_leaf(~Block,~Output,~Resource,~Context,~leaf,~keep),~Pieces.piece_kept(~Block,~Output,~Resource,~Context,~keep),
    left_plan,right_plan,enabled,enabled,context,(Partition.Leaf{block,first,rows,details},resource))

# An unpaired tree comes back unchanged.
def only_left(~Left: Type,~Right: Type,tree: Tree(~Left)) ->
  {Pieces.left_of(~Left,~Right,Pieces.only(~Left,~Right,tree)) == tree : Tree(~Left)}:
  match tree:
    case Partition.Leaf{block,first,rows,Unit{}}: {==}
    case Partition.Fork{left,right}:
      %only_left(~Left,~Right,left) :
        {Partition.Fork{Pieces.left_of(~Left,~Right,Pieces.only(~Left,~Right,left)),Pieces.left_of(~Left,~Right,Pieces.only(~Left,~Right,right))} == Partition.Fork{_,right} : Tree(~Left)}
      %only_left(~Left,~Right,right) :
        {Partition.Fork{Pieces.left_of(~Left,~Right,Pieces.only(~Left,~Right,left)),Pieces.left_of(~Left,~Right,Pieces.only(~Left,~Right,right))} ==
          Partition.Fork{Pieces.left_of(~Left,~Right,Pieces.only(~Left,~Right,left)),_} : Tree(~Left)}
      {==}

def KeptRule(~Left: Type,~Right: Type,~Resource: Type,~Context: Data,
  ~share: Resource -> Resource & Resource,
  ~leaf: Context -> U32 -> U32 -> Left -> Right -> Resource -> Tree(~Left),
  +layout: Pieces.Ranges,+enabled: Bool,+context: Context,-tree: Tree(~Left)) -> Type:
  @resource: Resource ->
  {Pieces.each(~Pieces.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Pieces.paired(~Left,~Right,~Resource,~Context,~leaf),~Pieces.left_of(~Left,~Right),
      layout,enabled,context,Pieces.only(~Left,~Right,tree),resource) == tree : Tree(~Left)}

def kept_fork(~Left: Type,~Right: Type,~Resource: Type,~Context: Data,
  ~share: Resource -> Resource & Resource,
  ~leaf: Context -> U32 -> U32 -> Left -> Right -> Resource -> Tree(~Left),
  +left_plan: Pieces.Ranges,+right_plan: Pieces.Ranges,+enabled: Bool,+context: Context,
  -left: Tree(~Left),-right: Tree(~Left),resources: Resource & Resource,
  left_rule: KeptRule(~Left,~Right,~Resource,~Context,~share,~leaf,left_plan,enabled,context,left),
  right_rule: KeptRule(~Left,~Right,~Resource,~Context,~share,~leaf,right_plan,enabled,context,right)) ->
  {forked(~Pieces.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Pieces.paired(~Left,~Right,~Resource,~Context,~leaf),~Pieces.left_of(~Left,~Right),
      left_plan,right_plan,enabled,context,Pieces.only(~Left,~Right,left),Pieces.only(~Left,~Right,right),resources) == Partition.Fork{left,right} : Tree(~Left)}:
  (left_resource,right_resource) = resources
  %left_rule(left_resource) :
    {Partition.Fork{Pieces.each(~Pieces.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Pieces.paired(~Left,~Right,~Resource,~Context,~leaf),~Pieces.left_of(~Left,~Right),
          left_plan,enabled,context,Pieces.only(~Left,~Right,left),left_resource),
        Pieces.each(~Pieces.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Pieces.paired(~Left,~Right,~Resource,~Context,~leaf),~Pieces.left_of(~Left,~Right),
          right_plan,enabled,context,Pieces.only(~Left,~Right,right),right_resource)} ==
      Partition.Fork{_,right} : Tree(~Left)}
  %right_rule(right_resource) :
    {Partition.Fork{Pieces.each(~Pieces.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Pieces.paired(~Left,~Right,~Resource,~Context,~leaf),~Pieces.left_of(~Left,~Right),
          left_plan,enabled,context,Pieces.only(~Left,~Right,left),left_resource),
        Pieces.each(~Pieces.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Pieces.paired(~Left,~Right,~Resource,~Context,~leaf),~Pieces.left_of(~Left,~Right),
          right_plan,enabled,context,Pieces.only(~Left,~Right,right),right_resource)} ==
      Partition.Fork{Pieces.each(~Pieces.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Pieces.paired(~Left,~Right,~Resource,~Context,~leaf),~Pieces.left_of(~Left,~Right),
          left_plan,enabled,context,Pieces.only(~Left,~Right,left),left_resource),_} : Tree(~Left)}
  {==}

# A paired operation on an unpaired tree returns the tree, on every plan.
def only_kept(~Left: Type,~Right: Type,~Resource: Type,~Context: Data,
  ~share: Resource -> Resource & Resource,
  ~leaf: Context -> U32 -> U32 -> Left -> Right -> Resource -> Tree(~Left),
  +layout: Pieces.Ranges,+enabled: Bool,+context: Context,tree: Tree(~Left),resource: Resource) ->
  {Pieces.each(~Pieces.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Pieces.paired(~Left,~Right,~Resource,~Context,~leaf),~Pieces.left_of(~Left,~Right),
      layout,enabled,context,Pieces.only(~Left,~Right,tree),resource) == tree : Tree(~Left)}:
  match layout tree:
    case Pieces.Range{first,rows} Partition.Leaf{block,own_first,own_rows,Unit{}}: {==}
    case Pieces.Range{first,rows} Partition.Fork{left,right}: only_left(~Left,~Right,Partition.Fork{left,right})
    case Pieces.Split{+left_plan,+right_plan} Partition.Leaf{block,own_first,own_rows,Unit{}}:
      split_leaf(~Pieces.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Pieces.paired(~Left,~Right,~Resource,~Context,~leaf),~Pieces.left_of(~Left,~Right),
        left_plan,right_plan,enabled,context,Pieces.Only{block,own_first,own_rows},own_first,own_rows,Unit{},resource)
    case Pieces.Split{+left_plan,+right_plan} Partition.Fork{left,right}:
      %Equal.sym(Tree(~Left),
        Pieces.each(~Pieces.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Pieces.paired(~Left,~Right,~Resource,~Context,~leaf),~Pieces.left_of(~Left,~Right),
          Pieces.Split{left_plan,right_plan},enabled,context,Partition.Fork{Pieces.only(~Left,~Right,left),Pieces.only(~Left,~Right,right)},resource),
        forked(~Pieces.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Pieces.paired(~Left,~Right,~Resource,~Context,~leaf),~Pieces.left_of(~Left,~Right),
          left_plan,right_plan,enabled,context,Pieces.only(~Left,~Right,left),Pieces.only(~Left,~Right,right),share(resource)),
        fork(~Pieces.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Pieces.paired(~Left,~Right,~Resource,~Context,~leaf),~Pieces.left_of(~Left,~Right),
          left_plan,right_plan,enabled,context,Pieces.only(~Left,~Right,left),Pieces.only(~Left,~Right,right),resource)) :
        {_ == Partition.Fork{left,right} : Tree(~Left)}
      kept_fork(~Left,~Right,~Resource,~Context,~share,~leaf,left_plan,right_plan,enabled,context,left,right,share(resource),
        left_resource => only_kept(~Left,~Right,~Resource,~Context,~share,~leaf,left_plan,enabled,context,left,left_resource),
        right_resource => only_kept(~Left,~Right,~Resource,~Context,~share,~leaf,right_plan,enabled,context,right,right_resource))

# The contents of an operation on every leaf of one tree. The leaf may hold
# elements of another type than the pieces it reads.
def content_map(~Input: Data,~Element: Data,~Context: Data,~leaf_model: Context -> U32 -> U32 -> Buffer.Model<Input> -> Model.Content<Element>,
  +context: Context,pieces: Model.Model<Input>) -> Model.Content<Element>:
  match pieces:
    case Model.Leaf{input,first,rows}: leaf_model(context,first,rows,input)
    case Model.Fork{left,right}: Model.Joined{content_map(~Input,~Element,~Context,~leaf_model,context,left),content_map(~Input,~Element,~Context,~leaf_model,context,right)}

def mapped_layout(~Input: Data,~Element: Data,~Context: Data,~leaf_model: Context -> U32 -> U32 -> Buffer.Model<Input> -> Model.Content<Element>,
  ~leaf: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+input: Buffer.Model<Input> -> {Model.content_layout(~Element,leaf_model(context,first,rows,input)) == Pieces.Range{first,rows} : Pieces.Ranges},
  +context: Context,pieces: Model.Model<Input>) ->
  {Model.content_layout(~Element,content_map(~Input,~Element,~Context,~leaf_model,context,pieces)) == Model.layout(~Input,pieces) : Pieces.Ranges}:
  match pieces:
    case Model.Leaf{+input,+first,+rows}: leaf(context,first,rows,input)
    case Model.Fork{+left,+right}:
      %mapped_layout(~Input,~Element,~Context,~leaf_model,~leaf,context,left) :
        {Pieces.Split{Model.content_layout(~Element,content_map(~Input,~Element,~Context,~leaf_model,context,left)),Model.content_layout(~Element,content_map(~Input,~Element,~Context,~leaf_model,context,right))} == Pieces.Split{_,Model.layout(~Input,right)} : Pieces.Ranges}
      %mapped_layout(~Input,~Element,~Context,~leaf_model,~leaf,context,right) :
        {Pieces.Split{Model.content_layout(~Element,content_map(~Input,~Element,~Context,~leaf_model,context,left)),Model.content_layout(~Element,content_map(~Input,~Element,~Context,~leaf_model,context,right))} == Pieces.Split{Model.content_layout(~Element,content_map(~Input,~Element,~Context,~leaf_model,context,left)),_} : Pieces.Ranges}
      {==}

def MapRule(~Input: Data,~Element: Data,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Input> -> Unit -> Pieces.Pieces(~Element),
  ~leaf_model: Context -> U32 -> U32 -> Buffer.Model<Input> -> Model.Content<Element>) -> Type:
  @+context: Context -> @+first: U32 -> @+rows: U32 -> @+input: Buffer.Model<Input> ->
  {Model.contents(~Element,leaf(context,first,rows,Buffer.pack(~Input,input),Unit{})) == leaf_model(context,first,rows,input) : Model.Content<Element>}

# A one-tree operation on its own plan: the tree induction, once. An
# operation supplies only the equation of its leaf; keep is never reached.
def mapped(~Input: Data,~Element: Data,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Input> -> Unit -> Pieces.Pieces(~Element),~keep: Pieces.Pieces(~Input) -> Pieces.Pieces(~Element),
  ~leaf_model: Context -> U32 -> U32 -> Buffer.Model<Input> -> Model.Content<Element>,
  ~leaf_rule: MapRule(~Input,~Element,~Context,~leaf,~leaf_model),
  pieces: Model.Model<Input>,+enabled: Bool,+context: Context) ->
  {Model.contents(~Element,Pieces.each(~Storage.Buffer<Input>,~Storage.Buffer<Element>,~Unit,~Context,~Pieces.unshared,~leaf,~keep,
      Model.layout(~Input,pieces),enabled,context,Model.pack(~Input,pieces),Unit{})) ==
    content_map(~Input,~Element,~Context,~leaf_model,context,pieces) : Model.Content<Element>}:
  match pieces:
    case Model.Leaf{+input,+first,+rows}: leaf_rule(context,first,rows,input)
    case Model.Fork{+left,+right}:
      %Equal.sym(Pieces.Pieces(~Element),
        Pieces.each(~Storage.Buffer<Input>,~Storage.Buffer<Element>,~Unit,~Context,~Pieces.unshared,~leaf,~keep,
          Pieces.Split{Model.layout(~Input,left),Model.layout(~Input,right)},enabled,context,Partition.Fork{Model.pack(~Input,left),Model.pack(~Input,right)},Unit{}),
        Partition.Fork{Pieces.each(~Storage.Buffer<Input>,~Storage.Buffer<Element>,~Unit,~Context,~Pieces.unshared,~leaf,~keep,
            Model.layout(~Input,left),enabled,context,Model.pack(~Input,left),Unit{}),
          Pieces.each(~Storage.Buffer<Input>,~Storage.Buffer<Element>,~Unit,~Context,~Pieces.unshared,~leaf,~keep,
            Model.layout(~Input,right),enabled,context,Model.pack(~Input,right),Unit{})},
        fork(~Storage.Buffer<Input>,~Storage.Buffer<Element>,~Unit,~Context,~Pieces.unshared,~leaf,~keep,
          Model.layout(~Input,left),Model.layout(~Input,right),enabled,context,Model.pack(~Input,left),Model.pack(~Input,right),Unit{})) :
        {Model.contents(~Element,_) == Model.Joined{content_map(~Input,~Element,~Context,~leaf_model,context,left),content_map(~Input,~Element,~Context,~leaf_model,context,right)} : Model.Content<Element>}
      %Equal.sym(Model.Content<Element>,
        Model.contents(~Element,Pieces.each(~Storage.Buffer<Input>,~Storage.Buffer<Element>,~Unit,~Context,~Pieces.unshared,~leaf,~keep,
          Model.layout(~Input,left),enabled,context,Model.pack(~Input,left),Unit{})),
        content_map(~Input,~Element,~Context,~leaf_model,context,left),
        mapped(~Input,~Element,~Context,~leaf,~keep,~leaf_model,~leaf_rule,left,enabled,context)) :
        {Model.Joined{_,Model.contents(~Element,Pieces.each(~Storage.Buffer<Input>,~Storage.Buffer<Element>,~Unit,~Context,~Pieces.unshared,~leaf,~keep,
            Model.layout(~Input,right),enabled,context,Model.pack(~Input,right),Unit{}))} ==
          Model.Joined{content_map(~Input,~Element,~Context,~leaf_model,context,left),content_map(~Input,~Element,~Context,~leaf_model,context,right)} : Model.Content<Element>}
      %Equal.sym(Model.Content<Element>,
        Model.contents(~Element,Pieces.each(~Storage.Buffer<Input>,~Storage.Buffer<Element>,~Unit,~Context,~Pieces.unshared,~leaf,~keep,
          Model.layout(~Input,right),enabled,context,Model.pack(~Input,right),Unit{})),
        content_map(~Input,~Element,~Context,~leaf_model,context,right),
        mapped(~Input,~Element,~Context,~leaf,~keep,~leaf_model,~leaf_rule,right,enabled,context)) :
        {Model.Joined{content_map(~Input,~Element,~Context,~leaf_model,context,left),_} ==
          Model.Joined{content_map(~Input,~Element,~Context,~leaf_model,context,left),content_map(~Input,~Element,~Context,~leaf_model,context,right)} : Model.Content<Element>}
      {==}

# Every leaf of a model through the leaf of an operation, in the shape of the
# model.
def leaf_map(~Element: Data,~Output: Type,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Unit -> Tree(~Output),
  +context: Context,pieces: Model.Model<Element>) -> Tree(~Output):
  match pieces:
    case Model.Leaf{input,first,rows}: leaf(context,first,rows,Buffer.pack(~Element,input),Unit{})
    case Model.Fork{left,right}: Partition.Fork{leaf_map(~Element,~Output,~Context,~leaf,context,left),leaf_map(~Element,~Output,~Context,~leaf,context,right)}

# On the plan of its own tree the recursion reaches every leaf, whatever it
# keeps elsewhere and whichever halves run in parallel.
def planned(~Element: Data,~Output: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Unit -> Tree(~Output),~keep: Tree(~Storage.Buffer<Element>) -> Tree(~Output),
  pieces: Model.Model<Element>,+enabled: Bool,+context: Context) ->
  {Pieces.each(~Storage.Buffer<Element>,~Output,~Unit,~Context,~Pieces.unshared,~leaf,~keep,Model.layout(~Element,pieces),enabled,context,Model.pack(~Element,pieces),Unit{}) ==
    leaf_map(~Element,~Output,~Context,~leaf,context,pieces) : Tree(~Output)}:
  match pieces:
    case Model.Leaf{+input,+first,+rows}: {==}
    case Model.Fork{+left,+right}:
      %Equal.sym(Tree(~Output),
        Pieces.each(~Storage.Buffer<Element>,~Output,~Unit,~Context,~Pieces.unshared,~leaf,~keep,
          Pieces.Split{Model.layout(~Element,left),Model.layout(~Element,right)},enabled,context,Partition.Fork{Model.pack(~Element,left),Model.pack(~Element,right)},Unit{}),
        Partition.Fork{Pieces.each(~Storage.Buffer<Element>,~Output,~Unit,~Context,~Pieces.unshared,~leaf,~keep,
            Model.layout(~Element,left),enabled,context,Model.pack(~Element,left),Unit{}),
          Pieces.each(~Storage.Buffer<Element>,~Output,~Unit,~Context,~Pieces.unshared,~leaf,~keep,
            Model.layout(~Element,right),enabled,context,Model.pack(~Element,right),Unit{})},
        fork(~Storage.Buffer<Element>,~Output,~Unit,~Context,~Pieces.unshared,~leaf,~keep,
          Model.layout(~Element,left),Model.layout(~Element,right),enabled,context,Model.pack(~Element,left),Model.pack(~Element,right),Unit{})) :
        {_ == Partition.Fork{leaf_map(~Element,~Output,~Context,~leaf,context,left),leaf_map(~Element,~Output,~Context,~leaf,context,right)} : Tree(~Output)}
      %Equal.sym(Tree(~Output),
        Pieces.each(~Storage.Buffer<Element>,~Output,~Unit,~Context,~Pieces.unshared,~leaf,~keep,
          Model.layout(~Element,left),enabled,context,Model.pack(~Element,left),Unit{}),
        leaf_map(~Element,~Output,~Context,~leaf,context,left),
        planned(~Element,~Output,~Context,~leaf,~keep,left,enabled,context)) :
        {Partition.Fork{_,Pieces.each(~Storage.Buffer<Element>,~Output,~Unit,~Context,~Pieces.unshared,~leaf,~keep,
            Model.layout(~Element,right),enabled,context,Model.pack(~Element,right),Unit{})} ==
          Partition.Fork{leaf_map(~Element,~Output,~Context,~leaf,context,left),leaf_map(~Element,~Output,~Context,~leaf,context,right)} : Tree(~Output)}
      %Equal.sym(Tree(~Output),
        Pieces.each(~Storage.Buffer<Element>,~Output,~Unit,~Context,~Pieces.unshared,~leaf,~keep,
          Model.layout(~Element,right),enabled,context,Model.pack(~Element,right),Unit{}),
        leaf_map(~Element,~Output,~Context,~leaf,context,right),
        planned(~Element,~Output,~Context,~leaf,~keep,right,enabled,context)) :
        {Partition.Fork{leaf_map(~Element,~Output,~Context,~leaf,context,left),_} ==
          Partition.Fork{leaf_map(~Element,~Output,~Context,~leaf,context,left),leaf_map(~Element,~Output,~Context,~leaf,context,right)} : Tree(~Output)}
      {==}

# A serial task walks all leaves it owns, preserving the leaf model.
def group_serial(~Element: Data,~Output: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Unit -> Tree(~Output),+context: Context,+pieces: Model.Model<Element>) ->
  {Pieces.group_run(~Storage.Buffer<Element>,~Output,~Context,~leaf,context,Model.pack(~Element,pieces)) ==
    leaf_map(~Element,~Output,~Context,~leaf,context,pieces) : Tree(~Output)}:
  match pieces:
    case Model.Leaf{input,first,rows}: {==}
    case Model.Fork{+left,+right}:
      Equal.trans(Tree(~Output),
        Pieces.group_run(~Storage.Buffer<Element>,~Output,~Context,~leaf,context,Model.pack(~Element,pieces)),
        Partition.Fork{Pieces.group_run(~Storage.Buffer<Element>,~Output,~Context,~leaf,context,Model.pack(~Element,left)),Pieces.group_run(~Storage.Buffer<Element>,~Output,~Context,~leaf,context,Model.pack(~Element,right))},
        leaf_map(~Element,~Output,~Context,~leaf,context,pieces),
        Walk.group_fork(~Storage.Buffer<Element>,~Output,~Context,~leaf,context,Model.pack(~Element,left),Model.pack(~Element,right)),
        Walk.group_children(~Output,
          Pieces.group_run(~Storage.Buffer<Element>,~Output,~Context,~leaf,context,Model.pack(~Element,left)),Pieces.group_run(~Storage.Buffer<Element>,~Output,~Context,~leaf,context,Model.pack(~Element,right)),
          leaf_map(~Element,~Output,~Context,~leaf,context,left),leaf_map(~Element,~Output,~Context,~leaf,context,right),
          group_serial(~Element,~Output,~Context,~leaf,context,left),group_serial(~Element,~Output,~Context,~leaf,context,right)))

def grouped(~Element: Data,~Output: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Unit -> Tree(~Output),+layout: Pieces.Ranges,+enabled: Bool,+context: Context,+pieces: Model.Model<Element>) ->
  {Pieces.grouped(~Storage.Buffer<Element>,~Output,~Context,~leaf,layout,enabled,context,Model.pack(~Element,pieces)) ==
    leaf_map(~Element,~Output,~Context,~leaf,context,pieces) : Tree(~Output)}:
  Equal.trans(Tree(~Output),
    Pieces.grouped(~Storage.Buffer<Element>,~Output,~Context,~leaf,layout,enabled,context,Model.pack(~Element,pieces)),
    Pieces.group_run(~Storage.Buffer<Element>,~Output,~Context,~leaf,context,Model.pack(~Element,pieces)),leaf_map(~Element,~Output,~Context,~leaf,context,pieces),
    %Model.buffer_parts_packed(~Element,pieces) :
      {Pieces.grouped(~Storage.Buffer<Element>,~Output,~Context,~leaf,layout,enabled,context,_) ==
        Pieces.group_run(~Storage.Buffer<Element>,~Output,~Context,~leaf,context,_) : Tree(~Output)}
    Walk.grouped(~Buffer.Model<Element>,~Storage.Buffer<Element>,~Buffer.pack(~Element),~Output,~Context,~leaf,layout,enabled,context,Model.buffer_parts(~Element,pieces)),
    group_serial(~Element,~Output,~Context,~leaf,context,pieces))

# A task grouping and the original piece plan give the same owned result.
def grouped_each(~Element: Data,~Output: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Unit -> Tree(~Output),~keep: Tree(~Storage.Buffer<Element>) -> Tree(~Output),
  +tasks: Pieces.Ranges,+enabled: Bool,+context: Context,+pieces: Model.Model<Element>) ->
  {Pieces.grouped(~Storage.Buffer<Element>,~Output,~Context,~leaf,tasks,enabled,context,Model.pack(~Element,pieces)) ==
    Pieces.each(~Storage.Buffer<Element>,~Output,~Unit,~Context,~Pieces.unshared,~leaf,~keep,Model.layout(~Element,pieces),enabled,context,Model.pack(~Element,pieces),Unit{}) : Tree(~Output)}:
  Equal.trans(Tree(~Output),
    Pieces.grouped(~Storage.Buffer<Element>,~Output,~Context,~leaf,tasks,enabled,context,Model.pack(~Element,pieces)),leaf_map(~Element,~Output,~Context,~leaf,context,pieces),
    Pieces.each(~Storage.Buffer<Element>,~Output,~Unit,~Context,~Pieces.unshared,~leaf,~keep,Model.layout(~Element,pieces),enabled,context,Model.pack(~Element,pieces),Unit{}),
    grouped(~Element,~Output,~Context,~leaf,tasks,enabled,context,pieces),
    Equal.sym(Tree(~Output),
      Pieces.each(~Storage.Buffer<Element>,~Output,~Unit,~Context,~Pieces.unshared,~leaf,~keep,Model.layout(~Element,pieces),enabled,context,Model.pack(~Element,pieces),Unit{}),
      leaf_map(~Element,~Output,~Context,~leaf,context,pieces),planned(~Element,~Output,~Context,~leaf,~keep,pieces,enabled,context)))

def piece_rows(~Part: Data,part: Part,+first: U32,+rows: U32) -> Pieces.Ranges:
  Pieces.Range{first,rows}

# Every piece of a tree of any parts through the leaf of an operation, in the
# shape of the tree.
def shaped_leaf_map(~Part: Data,~Block: Type,~pack: Part -> Block,~Output: Type,~Context: Data,~leaf: Context -> U32 -> U32 -> Block -> Unit -> Tree(~Output),
  +context: Context,tree: Model.Shaped<Part>) -> Tree(~Output):
  match tree:
    case Model.Piece{part,first,rows}: leaf(context,first,rows,pack(part),Unit{})
    case Model.Pieces{left,right}:
      Partition.Fork{shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,left),shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,right)}

# On the plan of the rows of its own pieces the recursion reaches every
# piece, whatever it keeps elsewhere and whichever halves run in parallel.
def shaped_planned(~Part: Data,~Block: Type,~pack: Part -> Block,~Output: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Block -> Unit -> Tree(~Output),~keep: Tree(~Block) -> Tree(~Output),
  tree: Model.Shaped<Part>,+enabled: Bool,+context: Context) ->
  {Pieces.each(~Block,~Output,~Unit,~Context,~Pieces.unshared,~leaf,~keep,Model.shaped_plan(~Part,~piece_rows(~Part),tree),enabled,context,
      Model.shaped_pack(~Part,~Block,~pack,tree),Unit{}) ==
    shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,tree) : Tree(~Output)}:
  match tree:
    case Model.Piece{+part,+first,+rows}: {==}
    case Model.Pieces{+left,+right}:
      +left_plan = Model.shaped_plan(~Part,~piece_rows(~Part),left)
      +right_plan = Model.shaped_plan(~Part,~piece_rows(~Part),right)
      %Equal.sym(Tree(~Output),
        Pieces.each(~Block,~Output,~Unit,~Context,~Pieces.unshared,~leaf,~keep,Pieces.Split{left_plan,right_plan},enabled,context,
          Partition.Fork{Model.shaped_pack(~Part,~Block,~pack,left),Model.shaped_pack(~Part,~Block,~pack,right)},Unit{}),
        Partition.Fork{Pieces.each(~Block,~Output,~Unit,~Context,~Pieces.unshared,~leaf,~keep,left_plan,enabled,context,Model.shaped_pack(~Part,~Block,~pack,left),Unit{}),
          Pieces.each(~Block,~Output,~Unit,~Context,~Pieces.unshared,~leaf,~keep,right_plan,enabled,context,Model.shaped_pack(~Part,~Block,~pack,right),Unit{})},
        fork(~Block,~Output,~Unit,~Context,~Pieces.unshared,~leaf,~keep,left_plan,right_plan,enabled,context,
          Model.shaped_pack(~Part,~Block,~pack,left),Model.shaped_pack(~Part,~Block,~pack,right),Unit{})) :
        {_ == Partition.Fork{shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,left),shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,right)} : Tree(~Output)}
      %Equal.sym(Tree(~Output),
        Pieces.each(~Block,~Output,~Unit,~Context,~Pieces.unshared,~leaf,~keep,left_plan,enabled,context,Model.shaped_pack(~Part,~Block,~pack,left),Unit{}),
        shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,left),
        shaped_planned(~Part,~Block,~pack,~Output,~Context,~leaf,~keep,left,enabled,context)) :
        {Partition.Fork{_,Pieces.each(~Block,~Output,~Unit,~Context,~Pieces.unshared,~leaf,~keep,right_plan,enabled,context,Model.shaped_pack(~Part,~Block,~pack,right),Unit{})} ==
          Partition.Fork{shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,left),shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,right)} : Tree(~Output)}
      %Equal.sym(Tree(~Output),
        Pieces.each(~Block,~Output,~Unit,~Context,~Pieces.unshared,~leaf,~keep,right_plan,enabled,context,Model.shaped_pack(~Part,~Block,~pack,right),Unit{}),
        shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,right),
        shaped_planned(~Part,~Block,~pack,~Output,~Context,~leaf,~keep,right,enabled,context)) :
        {Partition.Fork{shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,left),_} ==
          Partition.Fork{shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,left),shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,right)} : Tree(~Output)}
      {==}

# The serial group walks the same owned pieces as a plan of their rows.
def shaped_group_serial(~Part: Data,~Block: Type,~pack: Part -> Block,~Output: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Block -> Unit -> Tree(~Output),+context: Context,+tree: Model.Shaped<Part>) ->
  {Pieces.group_run(~Block,~Output,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,tree)) ==
    shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,tree) : Tree(~Output)}:
  match tree:
    case Model.Piece{part,first,rows}: {==}
    case Model.Pieces{+left,+right}:
      Equal.trans(Tree(~Output),
        Pieces.group_run(~Block,~Output,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,tree)),
        Partition.Fork{Pieces.group_run(~Block,~Output,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,left)),Pieces.group_run(~Block,~Output,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,right))},
        shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,tree),
        Walk.group_fork(~Block,~Output,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,left),Model.shaped_pack(~Part,~Block,~pack,right)),
        Walk.group_children(~Output,
          Pieces.group_run(~Block,~Output,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,left)),Pieces.group_run(~Block,~Output,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,right)),
          shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,left),shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,right),
          shaped_group_serial(~Part,~Block,~pack,~Output,~Context,~leaf,context,left),shaped_group_serial(~Part,~Block,~pack,~Output,~Context,~leaf,context,right)))

def shaped_grouped(~Part: Data,~Block: Type,~pack: Part -> Block,~Output: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Block -> Unit -> Tree(~Output),~keep: Tree(~Block) -> Tree(~Output),
  +tasks: Pieces.Ranges,+enabled: Bool,+context: Context,+tree: Model.Shaped<Part>) ->
  {Pieces.grouped(~Block,~Output,~Context,~leaf,tasks,enabled,context,Model.shaped_pack(~Part,~Block,~pack,tree)) ==
    Pieces.each(~Block,~Output,~Unit,~Context,~Pieces.unshared,~leaf,~keep,Model.shaped_plan(~Part,~piece_rows(~Part),tree),enabled,context,Model.shaped_pack(~Part,~Block,~pack,tree),Unit{}) : Tree(~Output)}:
  Equal.trans(Tree(~Output),
    Pieces.grouped(~Block,~Output,~Context,~leaf,tasks,enabled,context,Model.shaped_pack(~Part,~Block,~pack,tree)),
    shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,tree),
    Pieces.each(~Block,~Output,~Unit,~Context,~Pieces.unshared,~leaf,~keep,Model.shaped_plan(~Part,~piece_rows(~Part),tree),enabled,context,Model.shaped_pack(~Part,~Block,~pack,tree),Unit{}),
    Equal.trans(Tree(~Output),
      Pieces.grouped(~Block,~Output,~Context,~leaf,tasks,enabled,context,Model.shaped_pack(~Part,~Block,~pack,tree)),
      Pieces.group_run(~Block,~Output,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,tree)),shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,tree),
      Walk.grouped(~Part,~Block,~pack,~Output,~Context,~leaf,tasks,enabled,context,tree),shaped_group_serial(~Part,~Block,~pack,~Output,~Context,~leaf,context,tree)),
    Equal.sym(Tree(~Output),
      Pieces.each(~Block,~Output,~Unit,~Context,~Pieces.unshared,~leaf,~keep,Model.shaped_plan(~Part,~piece_rows(~Part),tree),enabled,context,Model.shaped_pack(~Part,~Block,~pack,tree),Unit{}),
      shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,tree),shaped_planned(~Part,~Block,~pack,~Output,~Context,~leaf,~keep,tree,enabled,context)))

# The contents of an operation on every pair of leaves of two trees of one
# shape; where the shapes differ, the left tree is kept.
def content_zip(~Element: Data,~Context: Data,~leaf_model: Context -> U32 -> U32 -> Buffer.Model<Element> -> Buffer.Model<Element> -> Model.Content<Element>,
  +context: Context,left: Model.Model<Element>,right: Model.Model<Element>) -> Model.Content<Element>:
  match left right:
    case Model.Leaf{left,first,rows} Model.Leaf{right,other_first,other_rows}: leaf_model(context,first,rows,left,right)
    case Model.Fork{left,right} Model.Fork{other_left,other_right}:
      Model.Joined{content_zip(~Element,~Context,~leaf_model,context,left,other_left),content_zip(~Element,~Context,~leaf_model,context,right,other_right)}
    case left right: Model.content(~Element,left)

def ZipRule(~Element: Data,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Unit -> Pieces.Pieces(~Element),
  ~leaf_model: Context -> U32 -> U32 -> Buffer.Model<Element> -> Buffer.Model<Element> -> Model.Content<Element>) -> Type:
  @+context: Context -> @+first: U32 -> @+rows: U32 -> @+left: Buffer.Model<Element> -> @+right: Buffer.Model<Element> ->
  {Model.contents(~Element,leaf(context,first,rows,Buffer.pack(~Element,left),Buffer.pack(~Element,right),Unit{})) == leaf_model(context,first,rows,left,right) : Model.Content<Element>}

def Zipped(~Element: Data,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Unit -> Pieces.Pieces(~Element),
  +layout: Pieces.Ranges,+enabled: Bool,+context: Context,left: Pieces.Pieces(~Element),right: Pieces.Pieces(~Element)) -> Pieces.Pieces(~Element):
  Pieces.each(~Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~Storage.Buffer<Element>,~Unit,~Context,~Pieces.unshared,
    ~Pieces.paired(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~leaf),~Pieces.left_of(~Storage.Buffer<Element>,~Storage.Buffer<Element>),
    layout,enabled,context,Pieces.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,left,right),Unit{})

# A two-tree operation on the plan of its left tree: the tree induction and
# the shape mismatches, once.
def zipped(~Element: Data,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Unit -> Pieces.Pieces(~Element),
  ~leaf_model: Context -> U32 -> U32 -> Buffer.Model<Element> -> Buffer.Model<Element> -> Model.Content<Element>,
  ~leaf_rule: ZipRule(~Element,~Context,~leaf,~leaf_model),
  left: Model.Model<Element>,right: Model.Model<Element>,+enabled: Bool,+context: Context) ->
  {Model.contents(~Element,Zipped(~Element,~Context,~leaf,Model.layout(~Element,left),enabled,context,Model.pack(~Element,left),Model.pack(~Element,right))) ==
    content_zip(~Element,~Context,~leaf_model,context,left,right) : Model.Content<Element>}:
  match left right:
    case Model.Leaf{+left,+first,+rows} Model.Leaf{+right,other_first,other_rows}: leaf_rule(context,first,rows,left,right)
    case Model.Leaf{+buffer,+first,+rows} Model.Fork{other_left,other_right}: Model.packed_contents(~Element,Model.Leaf{buffer,first,rows})
    case Model.Fork{+left,+right} Model.Leaf{buffer,other_first,other_rows}:
      Model.kept(~Element,Model.Fork{left,right},
        Zipped(~Element,~Context,~leaf,Pieces.Split{Model.layout(~Element,left),Model.layout(~Element,right)},enabled,context,
          Model.pack(~Element,Model.Fork{left,right}),Model.pack(~Element,Model.Leaf{buffer,other_first,other_rows})),
        only_kept(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~Pieces.unshared,~leaf,
          Pieces.Split{Model.layout(~Element,left),Model.layout(~Element,right)},enabled,context,Model.pack(~Element,Model.Fork{left,right}),Unit{}))
    case Model.Fork{+left,+right} Model.Fork{+other_left,+other_right}:
      %Equal.sym(Pieces.Pieces(~Element),
        Zipped(~Element,~Context,~leaf,Pieces.Split{Model.layout(~Element,left),Model.layout(~Element,right)},enabled,context,
          Model.pack(~Element,Model.Fork{left,right}),Model.pack(~Element,Model.Fork{other_left,other_right})),
        Partition.Fork{Zipped(~Element,~Context,~leaf,Model.layout(~Element,left),enabled,context,
            Model.pack(~Element,left),Model.pack(~Element,other_left)),
          Zipped(~Element,~Context,~leaf,Model.layout(~Element,right),enabled,context,
            Model.pack(~Element,right),Model.pack(~Element,other_right))},
        fork(~Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~Storage.Buffer<Element>,~Unit,~Context,~Pieces.unshared,
          ~Pieces.paired(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~leaf),~Pieces.left_of(~Storage.Buffer<Element>,~Storage.Buffer<Element>),
          Model.layout(~Element,left),Model.layout(~Element,right),enabled,context,
          Pieces.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,left),Model.pack(~Element,other_left)),
          Pieces.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,right),Model.pack(~Element,other_right)),Unit{})) :
        {Model.contents(~Element,_) == Model.Joined{content_zip(~Element,~Context,~leaf_model,context,left,other_left),content_zip(~Element,~Context,~leaf_model,context,right,other_right)} : Model.Content<Element>}
      %Equal.sym(Model.Content<Element>,
        Model.contents(~Element,Zipped(~Element,~Context,~leaf,Model.layout(~Element,left),enabled,context,
          Model.pack(~Element,left),Model.pack(~Element,other_left))),
        content_zip(~Element,~Context,~leaf_model,context,left,other_left),
        zipped(~Element,~Context,~leaf,~leaf_model,~leaf_rule,left,other_left,enabled,context)) :
        {Model.Joined{_,Model.contents(~Element,Zipped(~Element,~Context,~leaf,Model.layout(~Element,right),enabled,context,
            Model.pack(~Element,right),Model.pack(~Element,other_right)))} ==
          Model.Joined{content_zip(~Element,~Context,~leaf_model,context,left,other_left),content_zip(~Element,~Context,~leaf_model,context,right,other_right)} : Model.Content<Element>}
      %Equal.sym(Model.Content<Element>,
        Model.contents(~Element,Zipped(~Element,~Context,~leaf,Model.layout(~Element,right),enabled,context,
          Model.pack(~Element,right),Model.pack(~Element,other_right))),
        content_zip(~Element,~Context,~leaf_model,context,right,other_right),
        zipped(~Element,~Context,~leaf,~leaf_model,~leaf_rule,right,other_right,enabled,context)) :
        {Model.Joined{content_zip(~Element,~Context,~leaf_model,context,left,other_left),_} ==
          Model.Joined{content_zip(~Element,~Context,~leaf_model,context,left,other_left),content_zip(~Element,~Context,~leaf_model,context,right,other_right)} : Model.Content<Element>}
      {==}

# ---- Two trees of one shape ----

# Equal layouts are trees of one shape. Their constructor fields give the
# common first/rows needed by every checked leaf of two trees and every fork.
def row_first(layout: Pieces.Ranges) -> U32:
  match layout:
    case Pieces.Range{first,rows}: first
    case Pieces.Split{left,right}: 0

def row_count(layout: Pieces.Ranges) -> U32:
  match layout:
    case Pieces.Range{first,rows}: rows
    case Pieces.Split{left,right}: 0

def layout_match_leaf(layout: Pieces.Ranges) -> Bool:
  match layout:
    case Pieces.Range{first,rows}: True{}
    case Pieces.Split{left,right}: False{}

def left_plan(layout: Pieces.Ranges) -> Pieces.Ranges:
  match layout:
    case Pieces.Range{first,rows}: Pieces.Range{0,0}
    case Pieces.Split{left,right}: left

def right_plan(layout: Pieces.Ranges) -> Pieces.Ranges:
  match layout:
    case Pieces.Range{first,rows}: Pieces.Range{0,0}
    case Pieces.Split{left,right}: right

def Common(~Element: Data,left: Model.Model<Element>,right: Model.Model<Element>) -> Data:
  match left right:
    case Model.Leaf{left,+first,+rows} Model.Leaf{right,+other_first,+other_rows}:
      KernelGemm.Both<{first == other_first : U32},{rows == other_rows : U32}>
    case Model.Fork{left,right} Model.Fork{other_left,other_right}:
      KernelGemm.Both<Common(~Element,left,other_left),Common(~Element,right,other_right)>
    case left right: Empty

def matched(~Element: Data,+left: Model.Model<Element>,+right: Model.Model<Element>,+same: {Model.layout(~Element,left) == Model.layout(~Element,right) : Pieces.Ranges}) -> Common(~Element,left,right):
  match left right:
    case Model.Leaf{left,+first,+rows} Model.Leaf{right,+other_first,+other_rows}:
      KernelGemm.Both{Equal.cong(Pieces.Ranges,U32,layout => row_first(layout),Pieces.Range{first,rows},Pieces.Range{other_first,other_rows},same),
        Equal.cong(Pieces.Ranges,U32,layout => row_count(layout),Pieces.Range{first,rows},Pieces.Range{other_first,other_rows},same)}
    case Model.Fork{+left,+right} Model.Fork{+other_left,+other_right}:
      +first_plan = {Pieces.Split{Model.layout(~Element,left),Model.layout(~Element,right)} : Pieces.Ranges}
      +second_plan = {Pieces.Split{Model.layout(~Element,other_left),Model.layout(~Element,other_right)} : Pieces.Ranges}
      KernelGemm.Both{matched(~Element,left,other_left,Equal.cong(Pieces.Ranges,Pieces.Ranges,layout => left_plan(layout),first_plan,second_plan,same)),
        matched(~Element,right,other_right,Equal.cong(Pieces.Ranges,Pieces.Ranges,layout => right_plan(layout),first_plan,second_plan,same))}
    case Model.Leaf{buffer,+first,+rows} Model.Fork{+left,+right}:
      WordEncoding.false_true(Equal.sym(Bool,True{},False{},Equal.cong(Pieces.Ranges,Bool,layout => layout_match_leaf(layout),Pieces.Range{first,rows},Pieces.Split{Model.layout(~Element,left),Model.layout(~Element,right)},same)))
    case Model.Fork{+left,+right} Model.Leaf{buffer,+first,+rows}:
      WordEncoding.false_true(Equal.cong(Pieces.Ranges,Bool,layout => layout_match_leaf(layout),Pieces.Split{Model.layout(~Element,left),Model.layout(~Element,right)},Pieces.Range{first,rows},same))

# One piece of each of two trees of one shape.
type PiecePair<-Element: Data> is Data:
  PiecePair{left: Buffer.Model<Element>,right: Buffer.Model<Element>}

def pair_pack(~Element: Data,pair: PiecePair<Element>) -> Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>:
  PiecePair{left,right} = pair
  Pieces.Both{Buffer.pack(~Element,left),Buffer.pack(~Element,right)}

# The pieces of a tree, each paired with itself.
def self_pieces(~Element: Data,model: Model.Model<Element>) -> Model.Shaped<PiecePair<Element>>:
  match model:
    case Model.Leaf{+buffer,first,rows}: Model.Piece{PiecePair{buffer,buffer},first,rows}
    case Model.Fork{left,right}: Model.Pieces{self_pieces(~Element,left),self_pieces(~Element,right)}

# The pieces of two trees, each with the pieces of both, at the rows of the
# left tree; where the shapes differ, the left tree is paired with itself.
def paired_pieces(~Element: Data,left: Model.Model<Element>,right: Model.Model<Element>) -> Model.Shaped<PiecePair<Element>>:
  match left right:
    case Model.Leaf{left,first,rows} Model.Leaf{right,other_first,other_rows}: Model.Piece{PiecePair{left,right},first,rows}
    case Model.Fork{left,right} Model.Fork{other_left,other_right}:
      Model.Pieces{paired_pieces(~Element,left,other_left),paired_pieces(~Element,right,other_right)}
    case left right: self_pieces(~Element,left)

def pieces_pack(~Element: Data,left: Model.Model<Element>,right: Model.Model<Element>) -> Tree(~Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>):
  Model.shaped_pack(~PiecePair<Element>,~Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~pair_pack(~Element),paired_pieces(~Element,left,right))

# Two trees of one shape zip into the pairs of their pieces.
def paired_zip(~Element: Data,+left: Model.Model<Element>,+right: Model.Model<Element>,common: Common(~Element,left,right)) ->
  {Pieces.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,left),Model.pack(~Element,right)) ==
    Model.shaped_pack(~PiecePair<Element>,~Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~pair_pack(~Element),paired_pieces(~Element,left,right)) :
    Tree(~Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>)}:
  match left right:
    case Model.Leaf{+buffer,+first,+rows} Model.Leaf{+other,other_first,other_rows}: {==}
    case Model.Fork{+left,+right} Model.Fork{+other_left,+other_right}:
      KernelGemm.Both{left_common,right_common} = common
      %Equal.sym(Tree(~Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>),
          Pieces.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,left),Model.pack(~Element,other_left)),pieces_pack(~Element,left,other_left),
          paired_zip(~Element,left,other_left,left_common)) :
        {Partition.Fork{_,Pieces.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,right),Model.pack(~Element,other_right))} ==
          Partition.Fork{pieces_pack(~Element,left,other_left),pieces_pack(~Element,right,other_right)} : Tree(~Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>)}
      %Equal.sym(Tree(~Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>),
          Pieces.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,right),Model.pack(~Element,other_right)),pieces_pack(~Element,right,other_right),
          paired_zip(~Element,right,other_right,right_common)) :
        {Partition.Fork{pieces_pack(~Element,left,other_left),_} ==
          Partition.Fork{pieces_pack(~Element,left,other_left),pieces_pack(~Element,right,other_right)} : Tree(~Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>)}
      {==}
    case Model.Leaf{+buffer,+first,+rows} Model.Fork{+other_left,+other_right}:
      Empty.absurd({Pieces.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,Model.Leaf{buffer,first,rows}),Model.pack(~Element,Model.Fork{other_left,other_right})) ==
        Model.shaped_pack(~PiecePair<Element>,~Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~pair_pack(~Element),
          paired_pieces(~Element,Model.Leaf{buffer,first,rows},Model.Fork{other_left,other_right})) : Tree(~Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>)},common)
    case Model.Fork{+left,+right} Model.Leaf{+buffer,+first,+rows}:
      Empty.absurd({Pieces.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,Model.Fork{left,right}),Model.pack(~Element,Model.Leaf{buffer,first,rows})) ==
        Model.shaped_pack(~PiecePair<Element>,~Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~pair_pack(~Element),
          paired_pieces(~Element,Model.Fork{left,right},Model.Leaf{buffer,first,rows})) : Tree(~Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>)},common)

# The plan of the rows of the pieces of two trees of one shape is the layout
# of the left tree.
def paired_plan(~Element: Data,+left: Model.Model<Element>,+right: Model.Model<Element>,common: Common(~Element,left,right)) ->
  {Model.shaped_plan(~PiecePair<Element>,~piece_rows(~PiecePair<Element>),paired_pieces(~Element,left,right)) == Model.layout(~Element,left) : Pieces.Ranges}:
  match left right:
    case Model.Leaf{+buffer,+first,+rows} Model.Leaf{+other,other_first,other_rows}: {==}
    case Model.Fork{+left,+right} Model.Fork{+other_left,+other_right}:
      KernelGemm.Both{left_common,right_common} = common
      %Equal.sym(Pieces.Ranges,Model.shaped_plan(~PiecePair<Element>,~piece_rows(~PiecePair<Element>),paired_pieces(~Element,left,other_left)),Model.layout(~Element,left),
          paired_plan(~Element,left,other_left,left_common)) :
        {Pieces.Split{_,Model.shaped_plan(~PiecePair<Element>,~piece_rows(~PiecePair<Element>),paired_pieces(~Element,right,other_right))} ==
          Pieces.Split{Model.layout(~Element,left),Model.layout(~Element,right)} : Pieces.Ranges}
      %Equal.sym(Pieces.Ranges,Model.shaped_plan(~PiecePair<Element>,~piece_rows(~PiecePair<Element>),paired_pieces(~Element,right,other_right)),Model.layout(~Element,right),
          paired_plan(~Element,right,other_right,right_common)) :
        {Pieces.Split{Model.layout(~Element,left),_} == Pieces.Split{Model.layout(~Element,left),Model.layout(~Element,right)} : Pieces.Ranges}
      {==}
    case Model.Leaf{+buffer,+first,+rows} Model.Fork{+other_left,+other_right}:
      Empty.absurd({Model.shaped_plan(~PiecePair<Element>,~piece_rows(~PiecePair<Element>),paired_pieces(~Element,Model.Leaf{buffer,first,rows},Model.Fork{other_left,other_right})) ==
        Model.layout(~Element,Model.Leaf{buffer,first,rows}) : Pieces.Ranges},common)
    case Model.Fork{+left,+right} Model.Leaf{+buffer,+first,+rows}:
      Empty.absurd({Model.shaped_plan(~PiecePair<Element>,~piece_rows(~PiecePair<Element>),paired_pieces(~Element,Model.Fork{left,right},Model.Leaf{buffer,first,rows})) ==
        Model.layout(~Element,Model.Fork{left,right}) : Pieces.Ranges},common)

# Grouping a matched pair preserves the original piece-by-piece result.
def grouped_pairs(~Element: Data,~Output: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>> -> Unit -> Tree(~Output),
  ~keep: Tree(~Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>) -> Tree(~Output),
  +tasks: Pieces.Ranges,+enabled: Bool,+context: Context,+left: Model.Model<Element>,+right: Model.Model<Element>,+common: Common(~Element,left,right)) ->
  {Pieces.grouped(~Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~Output,~Context,~leaf,tasks,enabled,context,
      Pieces.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,left),Model.pack(~Element,right))) ==
    Pieces.each(~Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~Output,~Unit,~Context,~Pieces.unshared,~leaf,~keep,Model.layout(~Element,left),enabled,context,
      Pieces.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,left),Model.pack(~Element,right)),Unit{}) : Tree(~Output)}:
  %Equal.sym(Tree(~Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>),
    Pieces.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,left),Model.pack(~Element,right)),pieces_pack(~Element,left,right),paired_zip(~Element,left,right,common)) :
    {Pieces.grouped(~Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~Output,~Context,~leaf,tasks,enabled,context,_) ==
      Pieces.each(~Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~Output,~Unit,~Context,~Pieces.unshared,~leaf,~keep,Model.layout(~Element,left),enabled,context,_,Unit{}) : Tree(~Output)}
  %paired_plan(~Element,left,right,common) :
    {Pieces.grouped(~Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~Output,~Context,~leaf,tasks,enabled,context,pieces_pack(~Element,left,right)) ==
      Pieces.each(~Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~Output,~Unit,~Context,~Pieces.unshared,~leaf,~keep,_,enabled,context,pieces_pack(~Element,left,right),Unit{}) : Tree(~Output)}
  shaped_grouped(~PiecePair<Element>,~Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~pair_pack(~Element),~Output,~Context,~leaf,~keep,tasks,enabled,context,paired_pieces(~Element,left,right))

# What holds of the piece of each tree, at the rows of the pair.
def PairAt(~Element: Data,~Context: Data,~First: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Second: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+context: Context,+pair: PiecePair<Element>,+first: U32,+rows: U32) -> Data:
  PiecePair{+left,+right} = pair
  KernelGemm.Both<First(context,first,rows,Buffer.content(~Element,left)),Second(context,first,rows,Buffer.content(~Element,right))>

# What holds of every piece of each of two trees of one shape holds of every
# piece of their pairs.
def paired_all(~Element: Data,~Context: Data,~First: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Second: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+context: Context,
  +left: Model.Model<Element>,+right: Model.Model<Element>,common: Common(~Element,left,right),
  left_facts: Model.ContentAll(~Element,~First(context),Model.content(~Element,left)),right_facts: Model.ContentAll(~Element,~Second(context),Model.content(~Element,right))) ->
  Model.ShapedAll(~PiecePair<Element>,~Context,~PairAt(~Element,~Context,~First,~Second),context,paired_pieces(~Element,left,right)):
  match left right:
    case Model.Leaf{+buffer,+first,+rows} Model.Leaf{+other,+other_first,+other_rows}:
      KernelGemm.Both{+same_first,+same_rows} = common
      KernelGemm.Both{left_facts,
        %Equal.sym(U32,first,other_first,same_first) : Second(context,_,rows,Buffer.content(~Element,other))
        %Equal.sym(U32,rows,other_rows,same_rows) : Second(context,other_first,_,Buffer.content(~Element,other))
        right_facts}
    case Model.Fork{+left,+right} Model.Fork{+other_left,+other_right}:
      KernelGemm.Both{left_common,right_common} = common
      KernelGemm.Both{left_left,left_right} = left_facts
      KernelGemm.Both{right_left,right_right} = right_facts
      KernelGemm.Both{paired_all(~Element,~Context,~First,~Second,context,left,other_left,left_common,left_left,right_left),
        paired_all(~Element,~Context,~First,~Second,context,right,other_right,right_common,left_right,right_right)}
    case Model.Leaf{+buffer,+first,+rows} Model.Fork{+other_left,+other_right}:
      Empty.absurd(Model.ShapedAll(~PiecePair<Element>,~Context,~PairAt(~Element,~Context,~First,~Second),context,
        paired_pieces(~Element,Model.Leaf{buffer,first,rows},Model.Fork{other_left,other_right})),common)
    case Model.Fork{+left,+right} Model.Leaf{+buffer,+first,+rows}:
      Empty.absurd(Model.ShapedAll(~PiecePair<Element>,~Context,~PairAt(~Element,~Context,~First,~Second),context,
        paired_pieces(~Element,Model.Fork{left,right},Model.Leaf{buffer,first,rows})),common)

# A tree made from a plan with a state threaded through its leaves
# (Pieces.build). The state is a model held in its owner, and `see` reads from
# the owner what the proofs need. A leaf that keeps what is seen and makes one
# piece makes, over the plan, the tree of those pieces: an operation proves the
# rule of its leaf.
type Held<-State: Type,-Kept: Data,-hold: Kept -> State,-state: State> is Data:
  Held{kept: Kept,equation: {hold(kept) == state : State}}

def buffer_kept(~Element: Data,-buffer: Storage.Buffer<Element>,observation: Buffer.Observation<Element,buffer>) -> Held<Storage.Buffer<Element>,Buffer.Model<Element>,Buffer.pack(~Element),buffer>:
  Buffer.Observation{model,equation} = observation
  Held{model,equation}

def buffer_held(~Element: Data,buffer: Storage.Buffer<Element>) -> Held<Storage.Buffer<Element>,Buffer.Model<Element>,Buffer.pack(~Element),buffer>:
  buffer_kept(~Element,buffer,Buffer.observe(~Element,buffer))

def seen(~Element: Data,~State: Type,~See: Data,~see: State -> See,made: State & Pieces.Pieces(~Element)) -> See & Model.Content<Element>:
  (state,pieces) = made
  (see(state),Model.contents(~Element,pieces))

def BuildRule(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> State -> State & Pieces.Pieces(~Element),~part: Context -> See -> @+first: U32 -> @+rows: U32 -> Model.Content<Element>) -> Type:
  @+context: Context -> @+first: U32 -> @+rows: U32 -> @+kept: Kept ->
  {seen(~Element,~State,~See,~see,leaf(context,first,rows,hold(kept))) == (see(hold(kept)),part(context,see(hold(kept)),first,rows)) : See & Model.Content<Element>}

# A build compared by a key of what is seen and the contents of its pieces. The
# owners a copy or a clone gives back equal the ones it was given only in what
# they hold, so a leaf may give back a state of the same key; its parts then
# read what is seen only through the key.
def keyed(~Element: Data,~State: Type,~See: Data,~see: State -> See,~Key: Data,~key: See -> Key,made: State & Pieces.Pieces(~Element)) -> Key & Model.Content<Element>:
  (state,pieces) = made
  (key(see(state)),Model.contents(~Element,pieces))

def KeyedRule(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~Key: Data,~key: See -> Key,~leaf: Context -> U32 -> U32 -> State -> State & Pieces.Pieces(~Element),~part: Context -> See -> @+first: U32 -> @+rows: U32 -> Model.Content<Element>) -> Type:
  @+context: Context -> @+first: U32 -> @+rows: U32 -> @+kept: Kept ->
  {keyed(~Element,~State,~See,~see,~Key,~key,leaf(context,first,rows,hold(kept))) == (key(see(hold(kept))),part(context,see(hold(kept)),first,rows)) : Key & Model.Content<Element>}

def Respects(~Element: Data,~Context: Data,~See: Data,~Key: Data,~key: See -> Key,~part: Context -> See -> @+first: U32 -> @+rows: U32 -> Model.Content<Element>) -> Type:
  @+context: Context -> @+before: See -> @+after: See -> @+same: {key(before) == key(after) : Key} -> @+first: U32 -> @+rows: U32 ->
  {part(context,before,first,rows) == part(context,after,first,rows) : Model.Content<Element>}

def planned_respects(~Element: Data,~Context: Data,~See: Data,~Key: Data,~key: See -> Key,~part: Context -> See -> @+first: U32 -> @+rows: U32 -> Model.Content<Element>,
  ~respects: Respects(~Element,~Context,~See,~Key,~key,~part),+context: Context,+before: See,+after: See,+same: {key(before) == key(after) : Key},layout: Pieces.Ranges) ->
  {Model.planned(~Element,~part(context,before),layout) == Model.planned(~Element,~part(context,after),layout) : Model.Content<Element>}:
  match layout:
    case Pieces.Range{+first,+rows}: respects(context,before,after,same,first,rows)
    case Pieces.Split{+left,+right}:
      %Equal.sym(Model.Content<Element>,Model.planned(~Element,~part(context,before),left),Model.planned(~Element,~part(context,after),left),
        planned_respects(~Element,~Context,~See,~Key,~key,~part,~respects,context,before,after,same,left)) :
        {Model.Joined{_,Model.planned(~Element,~part(context,before),right)} ==
          Model.Joined{Model.planned(~Element,~part(context,after),left),Model.planned(~Element,~part(context,after),right)} : Model.Content<Element>}
      %Equal.sym(Model.Content<Element>,Model.planned(~Element,~part(context,before),right),Model.planned(~Element,~part(context,after),right),
        planned_respects(~Element,~Context,~See,~Key,~key,~part,~respects,context,before,after,same,right)) :
        {Model.Joined{Model.planned(~Element,~part(context,after),left),_} ==
          Model.Joined{Model.planned(~Element,~part(context,after),left),Model.planned(~Element,~part(context,after),right)} : Model.Content<Element>}
      {==}

def build_joined(~Element: Data,~See: Data,left: Model.Content<Element>,right: See & Model.Content<Element>) -> See & Model.Content<Element>:
  (state,right) = right
  (state,Model.Joined{left,right})

def build_join(~Element: Data,~State: Type,left: Pieces.Pieces(~Element),right: State & Pieces.Pieces(~Element)) -> State & Pieces.Pieces(~Element):
  Storage.bind_pair(~State,~Pieces.Pieces(~Element),~(State & Pieces.Pieces(~Element)),right,state => right => (state,Partition.Fork{left,right}))

def build_join_keyed(~Element: Data,~State: Type,~See: Data,~see: State -> See,~Key: Data,~key: See -> Key,left: Pieces.Pieces(~Element),right: State & Pieces.Pieces(~Element)) ->
  {keyed(~Element,~State,~See,~see,~Key,~key,build_join(~Element,~State,left,right)) == build_joined(~Element,~Key,Model.contents(~Element,left),keyed(~Element,~State,~See,~see,~Key,~key,right)) : Key & Model.Content<Element>}:
  (state,right) = right
  {==}

def build_spread(~Element: Data,~Context: Data,~See: Data,~Key: Data,~part: Context -> See -> @+first: U32 -> @+rows: U32 -> Model.Content<Element>,
  layout: Pieces.Ranges,context: Context,base: See,made: Key & Model.Content<Element>) -> Key & Model.Content<Element>:
  (key,left) = made
  (key,Model.Joined{left,Model.planned(~Element,~part(context,base),layout)})

def build_continued(~Element: Data,~State: Type,~Context: Data,~leaf: Context -> U32 -> U32 -> State -> State & Pieces.Pieces(~Element),layout: Pieces.Ranges,+context: Context,left: State & Pieces.Pieces(~Element)) -> State & Pieces.Pieces(~Element):
  Storage.bind_pair(~State,~Pieces.Pieces(~Element),~(State & Pieces.Pieces(~Element)),left,state => left => build_join(~Element,~State,left,Pieces.build(~Storage.Buffer<Element>,~State,~Context,~leaf,layout,context,state)))

def build_continue_packed(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~Key: Data,~key: See -> Key,~leaf: Context -> U32 -> U32 -> State -> State & Pieces.Pieces(~Element),~part: Context -> See -> @+first: U32 -> @+rows: U32 -> Model.Content<Element>,
  ~respects: Respects(~Element,~Context,~See,~Key,~key,~part),
  +layout: Pieces.Ranges,+context: Context,+base: See,-left_model: Model.Content<Element>,+kept: Kept,+left: Model.Model<Element>,
  +left_seen: {keyed(~Element,~State,~See,~see,~Key,~key,(hold(kept),Model.pack(~Element,left))) == (key(base),left_model) : Key & Model.Content<Element>},
  rule: @+kept: Kept -> {keyed(~Element,~State,~See,~see,~Key,~key,Pieces.build(~Storage.Buffer<Element>,~State,~Context,~leaf,layout,context,hold(kept))) ==
    (key(see(hold(kept))),Model.planned(~Element,~part(context,see(hold(kept))),layout)) : Key & Model.Content<Element>}) ->
  {keyed(~Element,~State,~See,~see,~Key,~key,build_continued(~Element,~State,~Context,~leaf,layout,context,(hold(kept),Model.pack(~Element,left)))) ==
    (key(base),Model.Joined{left_model,Model.planned(~Element,~part(context,base),layout)}) : Key & Model.Content<Element>}:
  +now = see(hold(kept))
  +pieces = Model.contents(~Element,Model.pack(~Element,left))
  +same = Equal.cong(Key & Model.Content<Element>,Key,made => Pair.fst(Key,Model.Content<Element>,made),(key(now),pieces),(key(base),left_model),left_seen)
  Equal.trans(Key & Model.Content<Element>,keyed(~Element,~State,~See,~see,~Key,~key,build_join(~Element,~State,Model.pack(~Element,left),Pieces.build(~Storage.Buffer<Element>,~State,~Context,~leaf,layout,context,hold(kept)))),
    build_joined(~Element,~Key,pieces,keyed(~Element,~State,~See,~see,~Key,~key,Pieces.build(~Storage.Buffer<Element>,~State,~Context,~leaf,layout,context,hold(kept)))),(key(base),Model.Joined{left_model,Model.planned(~Element,~part(context,base),layout)}),
    build_join_keyed(~Element,~State,~See,~see,~Key,~key,Model.pack(~Element,left),Pieces.build(~Storage.Buffer<Element>,~State,~Context,~leaf,layout,context,hold(kept))),
    Equal.trans(Key & Model.Content<Element>,build_joined(~Element,~Key,pieces,keyed(~Element,~State,~See,~see,~Key,~key,Pieces.build(~Storage.Buffer<Element>,~State,~Context,~leaf,layout,context,hold(kept)))),
      (key(now),Model.Joined{pieces,Model.planned(~Element,~part(context,now),layout)}),(key(base),Model.Joined{left_model,Model.planned(~Element,~part(context,base),layout)}),
      Equal.cong(Key & Model.Content<Element>,Key & Model.Content<Element>,made => build_joined(~Element,~Key,pieces,made),
        keyed(~Element,~State,~See,~see,~Key,~key,Pieces.build(~Storage.Buffer<Element>,~State,~Context,~leaf,layout,context,hold(kept))),(key(now),Model.planned(~Element,~part(context,now),layout)),rule(kept)),
      Equal.trans(Key & Model.Content<Element>,(key(now),Model.Joined{pieces,Model.planned(~Element,~part(context,now),layout)}),
        (key(now),Model.Joined{pieces,Model.planned(~Element,~part(context,base),layout)}),(key(base),Model.Joined{left_model,Model.planned(~Element,~part(context,base),layout)}),
        Equal.cong(Model.Content<Element>,Key & Model.Content<Element>,plan => (key(now),Model.Joined{pieces,plan}),
          Model.planned(~Element,~part(context,now),layout),Model.planned(~Element,~part(context,base),layout),
          planned_respects(~Element,~Context,~See,~Key,~key,~part,~respects,context,now,base,same,layout)),
        Equal.cong(Key & Model.Content<Element>,Key & Model.Content<Element>,made => build_spread(~Element,~Context,~See,~Key,~part,layout,context,base,made),
          (key(now),pieces),(key(base),left_model),left_seen))))

def build_continue_observed(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~Key: Data,~key: See -> Key,~leaf: Context -> U32 -> U32 -> State -> State & Pieces.Pieces(~Element),~part: Context -> See -> @+first: U32 -> @+rows: U32 -> Model.Content<Element>,
  ~respects: Respects(~Element,~Context,~See,~Key,~key,~part),
  +layout: Pieces.Ranges,+context: Context,+base: See,-left_model: Model.Content<Element>,-state: State,-left: Pieces.Pieces(~Element),
  state_held: Held<State,Kept,hold,state>,left_observation: Model.Observation<Element,left>,
  +left_seen: {keyed(~Element,~State,~See,~see,~Key,~key,(state,left)) == (key(base),left_model) : Key & Model.Content<Element>},
  rule: @+kept: Kept -> {keyed(~Element,~State,~See,~see,~Key,~key,Pieces.build(~Storage.Buffer<Element>,~State,~Context,~leaf,layout,context,hold(kept))) ==
    (key(see(hold(kept))),Model.planned(~Element,~part(context,see(hold(kept))),layout)) : Key & Model.Content<Element>}) ->
  {keyed(~Element,~State,~See,~see,~Key,~key,build_continued(~Element,~State,~Context,~leaf,layout,context,(state,left))) ==
    (key(base),Model.Joined{left_model,Model.planned(~Element,~part(context,base),layout)}) : Key & Model.Content<Element>}:
  Held{+kept,+state_equal} = state_held
  Model.Observation{+left_owner,+left_equal} = left_observation
  %state_equal : {keyed(~Element,~State,~See,~see,~Key,~key,build_continued(~Element,~State,~Context,~leaf,layout,context,(_,left))) ==
    (key(base),Model.Joined{left_model,Model.planned(~Element,~part(context,base),layout)}) : Key & Model.Content<Element>}
  %left_equal : {keyed(~Element,~State,~See,~see,~Key,~key,build_continued(~Element,~State,~Context,~leaf,layout,context,(hold(kept),_))) ==
    (key(base),Model.Joined{left_model,Model.planned(~Element,~part(context,base),layout)}) : Key & Model.Content<Element>}
  build_continue_packed(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~Key,~key,~leaf,~part,~respects,layout,context,base,left_model,kept,left_owner,
    %Equal.sym(Pieces.Pieces(~Element),Model.pack(~Element,left_owner),left,left_equal) : {keyed(~Element,~State,~See,~see,~Key,~key,(hold(kept),_)) == (key(base),left_model) : Key & Model.Content<Element>}
    %Equal.sym(State,hold(kept),state,state_equal) : {keyed(~Element,~State,~See,~see,~Key,~key,(_,left)) == (key(base),left_model) : Key & Model.Content<Element>}
    left_seen,rule)

def build_continue(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~Key: Data,~key: See -> Key,~leaf: Context -> U32 -> U32 -> State -> State & Pieces.Pieces(~Element),~part: Context -> See -> @+first: U32 -> @+rows: U32 -> Model.Content<Element>,
  ~respects: Respects(~Element,~Context,~See,~Key,~key,~part),
  +layout: Pieces.Ranges,+context: Context,+base: See,-left_model: Model.Content<Element>,left: State & Pieces.Pieces(~Element),
  +left_seen: {keyed(~Element,~State,~See,~see,~Key,~key,left) == (key(base),left_model) : Key & Model.Content<Element>},
  rule: @+kept: Kept -> {keyed(~Element,~State,~See,~see,~Key,~key,Pieces.build(~Storage.Buffer<Element>,~State,~Context,~leaf,layout,context,hold(kept))) ==
    (key(see(hold(kept))),Model.planned(~Element,~part(context,see(hold(kept))),layout)) : Key & Model.Content<Element>}) ->
  {keyed(~Element,~State,~See,~see,~Key,~key,build_continued(~Element,~State,~Context,~leaf,layout,context,left)) ==
    (key(base),Model.Joined{left_model,Model.planned(~Element,~part(context,base),layout)}) : Key & Model.Content<Element>}:
  (state,pieces) = left
  build_continue_observed(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~Key,~key,~leaf,~part,~respects,layout,context,base,left_model,state,pieces,observe(state),Model.observe(~Element,pieces),left_seen,rule)

def built_parts(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~Key: Data,~key: See -> Key,~leaf: Context -> U32 -> U32 -> State -> State & Pieces.Pieces(~Element),~part: Context -> See -> @+first: U32 -> @+rows: U32 -> Model.Content<Element>,
  ~respects: Respects(~Element,~Context,~See,~Key,~key,~part),
  ~leaf_rule: KeyedRule(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~Key,~key,~leaf,~part),layout: Pieces.Ranges,+context: Context,+kept: Kept) ->
  {keyed(~Element,~State,~See,~see,~Key,~key,Pieces.build(~Storage.Buffer<Element>,~State,~Context,~leaf,layout,context,hold(kept))) ==
    (key(see(hold(kept))),Model.planned(~Element,~part(context,see(hold(kept))),layout)) : Key & Model.Content<Element>}:
  match layout:
    case Pieces.Range{+first,+rows}: leaf_rule(context,first,rows,kept)
    case Pieces.Split{+left,+right}:
      build_continue(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~Key,~key,~leaf,~part,~respects,right,context,see(hold(kept)),Model.planned(~Element,~part(context,see(hold(kept))),left),
        Pieces.build(~Storage.Buffer<Element>,~State,~Context,~leaf,left,context,hold(kept)),built_parts(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~Key,~key,~leaf,~part,~respects,~leaf_rule,left,context,kept),
        next => built_parts(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~Key,~key,~leaf,~part,~respects,~leaf_rule,right,context,next))

# What is seen as its own key: the keyed build is the seen one, and parts
# that read what is seen respect it.
def same_seen(~See: Data,see: See) -> See:
  see
def seen_keyed(~Element: Data,~State: Type,~See: Data,~see: State -> See,made: State & Pieces.Pieces(~Element)) ->
  {keyed(~Element,~State,~See,~see,~See,~same_seen(~See),made) == seen(~Element,~State,~See,~see,made) : See & Model.Content<Element>}:
  (state,pieces) = made
  {==}
def seen_rule(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> State -> State & Pieces.Pieces(~Element),~part: Context -> See -> @+first: U32 -> @+rows: U32 -> Model.Content<Element>,
  ~leaf_rule: BuildRule(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~part),+context: Context,+first: U32,+rows: U32,+kept: Kept) ->
  {keyed(~Element,~State,~See,~see,~See,~same_seen(~See),leaf(context,first,rows,hold(kept))) == (see(hold(kept)),part(context,see(hold(kept)),first,rows)) : See & Model.Content<Element>}:
  Equal.trans(See & Model.Content<Element>,keyed(~Element,~State,~See,~see,~See,~same_seen(~See),leaf(context,first,rows,hold(kept))),
    seen(~Element,~State,~See,~see,leaf(context,first,rows,hold(kept))),(see(hold(kept)),part(context,see(hold(kept)),first,rows)),
    seen_keyed(~Element,~State,~See,~see,leaf(context,first,rows,hold(kept))),leaf_rule(context,first,rows,kept))
def same_respects(~Element: Data,~Context: Data,~See: Data,~part: Context -> See -> @+first: U32 -> @+rows: U32 -> Model.Content<Element>,
  +context: Context,+before: See,+after: See,+same: {same_seen(~See,before) == same_seen(~See,after) : See},+first: U32,+rows: U32) ->
  {part(context,before,first,rows) == part(context,after,first,rows) : Model.Content<Element>}:
  Equal.cong(See,Model.Content<Element>,see => part(context,see,first,rows),before,after,same)

# A leaf that makes one piece.
def row_part(~Element: Data,~Context: Data,~See: Data,~piece: Context -> See -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element>,context: Context,see: See,+first: U32,+rows: U32) -> Model.Content<Element>:
  Model.PieceContent{piece(context,see,first,rows),first,rows}

def planned_rows(~Element: Data,~Context: Data,~See: Data,~piece: Context -> See -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element>,+context: Context,+see: See,layout: Pieces.Ranges) ->
  {Model.planned(~Element,~row_part(~Element,~Context,~See,~piece,context,see),layout) == Model.build(~Element,~piece(context,see),layout) : Model.Content<Element>}:
  match layout:
    case Pieces.Range{first,rows}: {==}
    case Pieces.Split{+left,+right}:
      %Equal.sym(Model.Content<Element>,Model.planned(~Element,~row_part(~Element,~Context,~See,~piece,context,see),left),Model.build(~Element,~piece(context,see),left),
        planned_rows(~Element,~Context,~See,~piece,context,see,left)) :
        {Model.Joined{_,Model.planned(~Element,~row_part(~Element,~Context,~See,~piece,context,see),right)} == Model.Joined{Model.build(~Element,~piece(context,see),left),Model.build(~Element,~piece(context,see),right)} : Model.Content<Element>}
      %Equal.sym(Model.Content<Element>,Model.planned(~Element,~row_part(~Element,~Context,~See,~piece,context,see),right),Model.build(~Element,~piece(context,see),right),
        planned_rows(~Element,~Context,~See,~piece,context,see,right)) :
        {Model.Joined{Model.build(~Element,~piece(context,see),left),_} == Model.Joined{Model.build(~Element,~piece(context,see),left),Model.build(~Element,~piece(context,see),right)} : Model.Content<Element>}
      {==}

# A leaf that keeps what is seen and makes one piece makes, over the plan, the
# tree of those pieces.
def built(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> State -> State & Pieces.Pieces(~Element),~piece: Context -> See -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element>,
  ~leaf_rule: BuildRule(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~row_part(~Element,~Context,~See,~piece)),+layout: Pieces.Ranges,+context: Context,+kept: Kept) ->
  {seen(~Element,~State,~See,~see,Pieces.build(~Storage.Buffer<Element>,~State,~Context,~leaf,layout,context,hold(kept))) == (see(hold(kept)),Model.build(~Element,~piece(context,see(hold(kept))),layout)) : See & Model.Content<Element>}:
  %planned_rows(~Element,~Context,~See,~piece,context,see(hold(kept)),layout) :
    {seen(~Element,~State,~See,~see,Pieces.build(~Storage.Buffer<Element>,~State,~Context,~leaf,layout,context,hold(kept))) == (see(hold(kept)),_) : See & Model.Content<Element>}
  Equal.trans(See & Model.Content<Element>,seen(~Element,~State,~See,~see,Pieces.build(~Storage.Buffer<Element>,~State,~Context,~leaf,layout,context,hold(kept))),
    keyed(~Element,~State,~See,~see,~See,~same_seen(~See),Pieces.build(~Storage.Buffer<Element>,~State,~Context,~leaf,layout,context,hold(kept))),
    (see(hold(kept)),Model.planned(~Element,~row_part(~Element,~Context,~See,~piece)(context,see(hold(kept))),layout)),
    Equal.sym(See & Model.Content<Element>,keyed(~Element,~State,~See,~see,~See,~same_seen(~See),Pieces.build(~Storage.Buffer<Element>,~State,~Context,~leaf,layout,context,hold(kept))),seen(~Element,~State,~See,~see,Pieces.build(~Storage.Buffer<Element>,~State,~Context,~leaf,layout,context,hold(kept))),
      seen_keyed(~Element,~State,~See,~see,Pieces.build(~Storage.Buffer<Element>,~State,~Context,~leaf,layout,context,hold(kept)))),
    built_parts(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~See,~same_seen(~See),~leaf,~row_part(~Element,~Context,~See,~piece),
      ~same_respects(~Element,~Context,~See,~row_part(~Element,~Context,~See,~piece)),
      ~seen_rule(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~row_part(~Element,~Context,~See,~piece),~leaf_rule),layout,context,kept))

# A fold of the code over a tree of any parts, with the state held in its
# owner: on every piece it threads what is seen of the state. An operation
# supplies only the equation of its leaf.
def ShapedFoldRule(~Part: Data,~Block: Type,~pack: Part -> Block,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> Block -> State -> State,~step: Context -> @+first: U32 -> @+rows: U32 -> Part -> See -> See) -> Type:
  @+context: Context -> @+first: U32 -> @+rows: U32 -> @+part: Part -> @+kept: Kept ->
  {see(leaf(context,first,rows,pack(part),hold(kept))) == step(context,first,rows,part,see(hold(kept))) : See}

def fold_continue(~Part: Data,~Block: Type,~pack: Part -> Block,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> Block -> State -> State,~step: Context -> @+first: U32 -> @+rows: U32 -> Part -> See -> See,
  +right: Model.Shaped<Part>,+context: Context,-before: See,-state: State,held: Held<State,Kept,hold,state>,left_seen: {see(state) == before : See},
  rule: @+kept: Kept -> {see(Pieces.fold(~Block,~State,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,right),hold(kept))) == Model.shaped_thread(~Part,~See,~Context,~step,context,right,see(hold(kept))) : See}) ->
  {see(Pieces.fold(~Block,~State,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,right),state)) == Model.shaped_thread(~Part,~See,~Context,~step,context,right,before) : See}:
  Held{+kept,+equation} = held
  Equal.trans(See,see(Pieces.fold(~Block,~State,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,right),state)),
    Model.shaped_thread(~Part,~See,~Context,~step,context,right,see(state)),Model.shaped_thread(~Part,~See,~Context,~step,context,right,before),
    %equation : {see(Pieces.fold(~Block,~State,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,right),_)) == Model.shaped_thread(~Part,~See,~Context,~step,context,right,see(_)) : See}
    rule(kept),
    Equal.cong(See,See,value => Model.shaped_thread(~Part,~See,~Context,~step,context,right,value),see(state),before,left_seen))

def shaped_folded(~Part: Data,~Block: Type,~pack: Part -> Block,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> Block -> State -> State,~step: Context -> @+first: U32 -> @+rows: U32 -> Part -> See -> See,
  ~rule: ShapedFoldRule(~Part,~Block,~pack,~State,~Context,~Kept,~hold,~See,~see,~leaf,~step),tree: Model.Shaped<Part>,+context: Context,+kept: Kept) ->
  {see(Pieces.fold(~Block,~State,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,tree),hold(kept))) == Model.shaped_thread(~Part,~See,~Context,~step,context,tree,see(hold(kept))) : See}:
  match tree:
    case Model.Piece{+part,+first,+rows}: rule(context,first,rows,part,kept)
    case Model.Pieces{+left,+right}:
      fold_continue(~Part,~Block,~pack,~State,~Context,~Kept,~hold,~See,~see,~leaf,~step,right,context,Model.shaped_thread(~Part,~See,~Context,~step,context,left,see(hold(kept))),
        Pieces.fold(~Block,~State,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,left),hold(kept)),
        observe(Pieces.fold(~Block,~State,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,left),hold(kept))),
        shaped_folded(~Part,~Block,~pack,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~step,~rule,left,context,kept),
        next => shaped_folded(~Part,~Block,~pack,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~step,~rule,right,context,next))

# ---- The walk with the trees waiting on its left ----

# The trees waiting on the left of a walk, from their models.
def pack_lefts(~Element: Data,models: List<&2,Model.Model<Element>>) -> Pieces.Lefts<Storage.Buffer<Element>>:
  match models:
    case Nil{}: Pieces.NoLefts{}
    case Con{model,rest}: Pieces.Left{Model.pack(~Element,model),pack_lefts(~Element,rest)}

def WalkedThrough(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> State -> Pieces.Pieces(~Element) & State,
  +context: Context,+right: Model.Model<Element>) -> Type:
  @+models: List<&2,Model.Model<Element>> -> @+kept: Kept ->
    {Pieces.leaf_walked(~Storage.Buffer<Element>,~State,pack_lefts(~Element,models),Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,right),hold(kept))) == Pieces.walked(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,right),Pieces.Walking{pack_lefts(~Element,models),hold(kept)}) : Pieces.Walked<Storage.Buffer<Element>,State>}

# The right tree walks after the left one, observed, and joins it.
def walked_observed(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> State -> Pieces.Pieces(~Element) & State,
  +context: Context,+right: Model.Model<Element>,+models: List<&2,Model.Model<Element>>,-left: Pieces.Pieces(~Element),-state: State,
  left_observation: Model.Observation<Element,left>,state_held: Held<State,Kept,hold,state>,rule: WalkedThrough(~Element,~State,~Context,~Kept,~hold,~observe,~leaf,context,right)) ->
  {Pieces.leaf_walked(~Storage.Buffer<Element>,~State,pack_lefts(~Element,models),Storage.bind_pair(~Pieces.Pieces(~Element),~State,~(Pieces.Pieces(~Element) & State),(left,state),left => state =>
      Storage.bind_pair(~Pieces.Pieces(~Element),~State,~(Pieces.Pieces(~Element) & State),Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,right),state),right => state => (Partition.Fork{left,right},state)))) ==
    Pieces.joined(~Storage.Buffer<Element>,~State,Pieces.walked(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,right),Pieces.pushed(~Storage.Buffer<Element>,~State,Pieces.leaf_walked(~Storage.Buffer<Element>,~State,pack_lefts(~Element,models),(left,state))))) : Pieces.Walked<Storage.Buffer<Element>,State>}:
  Model.Observation{+owner,+left_equal} = left_observation
  Held{+kept,+state_equal} = state_held
  %left_equal : {Pieces.leaf_walked(~Storage.Buffer<Element>,~State,pack_lefts(~Element,models),Storage.bind_pair(~Pieces.Pieces(~Element),~State,~(Pieces.Pieces(~Element) & State),(_,state),left => state =>
      Storage.bind_pair(~Pieces.Pieces(~Element),~State,~(Pieces.Pieces(~Element) & State),Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,right),state),right => state => (Partition.Fork{left,right},state)))) ==
    Pieces.joined(~Storage.Buffer<Element>,~State,Pieces.walked(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,right),Pieces.pushed(~Storage.Buffer<Element>,~State,Pieces.leaf_walked(~Storage.Buffer<Element>,~State,pack_lefts(~Element,models),(_,state))))) : Pieces.Walked<Storage.Buffer<Element>,State>}
  %state_equal : {Pieces.leaf_walked(~Storage.Buffer<Element>,~State,pack_lefts(~Element,models),Storage.bind_pair(~Pieces.Pieces(~Element),~State,~(Pieces.Pieces(~Element) & State),(Model.pack(~Element,owner),_),left => state =>
      Storage.bind_pair(~Pieces.Pieces(~Element),~State,~(Pieces.Pieces(~Element) & State),Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,right),state),right => state => (Partition.Fork{left,right},state)))) ==
    Pieces.joined(~Storage.Buffer<Element>,~State,Pieces.walked(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,right),Pieces.pushed(~Storage.Buffer<Element>,~State,Pieces.leaf_walked(~Storage.Buffer<Element>,~State,pack_lefts(~Element,models),(Model.pack(~Element,owner),_))))) : Pieces.Walked<Storage.Buffer<Element>,State>}
  %rule(Con{owner,models},kept) : {Pieces.leaf_walked(~Storage.Buffer<Element>,~State,pack_lefts(~Element,models),Storage.bind_pair(~Pieces.Pieces(~Element),~State,~(Pieces.Pieces(~Element) & State),(Model.pack(~Element,owner),hold(kept)),left => state =>
      Storage.bind_pair(~Pieces.Pieces(~Element),~State,~(Pieces.Pieces(~Element) & State),Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,right),state),right => state => (Partition.Fork{left,right},state)))) == Pieces.joined(~Storage.Buffer<Element>,~State,_) : Pieces.Walked<Storage.Buffer<Element>,State>}
  Walk.joined_after(~Storage.Buffer<Element>,~State,Model.pack(~Element,owner),pack_lefts(~Element,models),Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,right),hold(kept)))

# The right tree walks after the left one and joins it.
def walked_fork(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> State -> Pieces.Pieces(~Element) & State,
  +context: Context,+right: Model.Model<Element>,+models: List<&2,Model.Model<Element>>,made: Pieces.Pieces(~Element) & State,
  rule: WalkedThrough(~Element,~State,~Context,~Kept,~hold,~observe,~leaf,context,right)) ->
  {Pieces.leaf_walked(~Storage.Buffer<Element>,~State,pack_lefts(~Element,models),Storage.bind_pair(~Pieces.Pieces(~Element),~State,~(Pieces.Pieces(~Element) & State),made,left => state =>
      Storage.bind_pair(~Pieces.Pieces(~Element),~State,~(Pieces.Pieces(~Element) & State),Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,right),state),right => state => (Partition.Fork{left,right},state)))) ==
    Pieces.joined(~Storage.Buffer<Element>,~State,Pieces.walked(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,right),Pieces.pushed(~Storage.Buffer<Element>,~State,Pieces.leaf_walked(~Storage.Buffer<Element>,~State,pack_lefts(~Element,models),made)))) : Pieces.Walked<Storage.Buffer<Element>,State>}:
  (left,state) = made
  walked_observed(~Element,~State,~Context,~Kept,~hold,~observe,~leaf,context,right,models,left,state,Model.observe(~Element,left),observe(state),rule)

# A walk keeps the trees waiting on its left and walks its tree by recursion.
def walked_through(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> State -> Pieces.Pieces(~Element) & State,
  tree: Model.Model<Element>,+context: Context,+models: List<&2,Model.Model<Element>>,+kept: Kept) ->
  {Pieces.leaf_walked(~Storage.Buffer<Element>,~State,pack_lefts(~Element,models),Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,tree),hold(kept))) == Pieces.walked(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,tree),Pieces.Walking{pack_lefts(~Element,models),hold(kept)}) : Pieces.Walked<Storage.Buffer<Element>,State>}:
  match tree:
    case Model.Leaf{+buffer,+first,+rows}: {==}
    case Model.Fork{+left,+right}:
      %walked_through(~Element,~State,~Context,~Kept,~hold,~observe,~leaf,left,context,models,kept) :
        {Pieces.leaf_walked(~Storage.Buffer<Element>,~State,pack_lefts(~Element,models),Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,Model.Fork{left,right}),hold(kept))) == Pieces.joined(~Storage.Buffer<Element>,~State,Pieces.walked(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,right),Pieces.pushed(~Storage.Buffer<Element>,~State,_))) : Pieces.Walked<Storage.Buffer<Element>,State>}
      walked_fork(~Element,~State,~Context,~Kept,~hold,~observe,~leaf,context,right,models,Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,left),hold(kept)),
        others => next => walked_through(~Element,~State,~Context,~Kept,~hold,~observe,~leaf,right,context,others,next))

# The walk of a tree is its walk by recursion.
def walk_model_same(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> State -> Pieces.Pieces(~Element) & State,
  +tree: Model.Model<Element>,+context: Context,+kept: Kept) ->
  {Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,tree),hold(kept)) == Pieces.walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,tree),hold(kept)) : Pieces.Pieces(~Element) & State}:
  %walked_through(~Element,~State,~Context,~Kept,~hold,~observe,~leaf,tree,context,Nil{},kept) :
    {Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,tree),hold(kept)) == Pieces.walk_result(~Storage.Buffer<Element>,~State,_) : Pieces.Pieces(~Element) & State}
  Equal.sym(Pieces.Pieces(~Element) & State,Pieces.walk_result(~Storage.Buffer<Element>,~State,Pieces.leaf_walked(~Storage.Buffer<Element>,~State,Pieces.NoLefts{},Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,tree),hold(kept)))),Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,tree),hold(kept)),
    Walk.result_walked(~Storage.Buffer<Element>,~State,Pieces.NoLefts{},Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,tree),hold(kept))))

# A walk of the code whose leaves give the state back as they found it maps
# every piece, each with what is seen of the state. An operation supplies only
# the equation of its leaf.
def walk_seen(~Element: Data,~State: Type,~See: Data,~see: State -> See,made: Pieces.Pieces(~Element) & State) -> Model.Content<Element> & See:
  (pieces,state) = made
  (Model.contents(~Element,pieces),see(state))

def WalkRule(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> State -> Pieces.Pieces(~Element) & State,~piece: Context -> See -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>) -> Type:
  @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Model<Element> -> @+kept: Kept ->
  {walk_seen(~Element,~State,~See,~see,leaf(context,first,rows,Buffer.pack(~Element,buffer),hold(kept))) == (Model.PieceContent{piece(context,see(hold(kept)),first,rows,Buffer.content(~Element,buffer)),first,rows},see(hold(kept))) : Model.Content<Element> & See}

def walk_joined(~Element: Data,~See: Data,left: Model.Content<Element>,right: Model.Content<Element> & See) -> Model.Content<Element> & See:
  (right,state) = right
  (Model.Joined{left,right},state)

def walk_join(~Element: Data,~State: Type,left: Pieces.Pieces(~Element),right: Pieces.Pieces(~Element) & State) -> Pieces.Pieces(~Element) & State:
  Storage.bind_pair(~Pieces.Pieces(~Element),~State,~(Pieces.Pieces(~Element) & State),right,right => state => (Partition.Fork{left,right},state))

def walk_join_seen(~Element: Data,~State: Type,~See: Data,~see: State -> See,left: Pieces.Pieces(~Element),right: Pieces.Pieces(~Element) & State) ->
  {walk_seen(~Element,~State,~See,~see,walk_join(~Element,~State,left,right)) == walk_joined(~Element,~See,Model.contents(~Element,left),walk_seen(~Element,~State,~See,~see,right)) : Model.Content<Element> & See}:
  (right,state) = right
  {==}

def walk_spread(~Element: Data,~Context: Data,~See: Data,~piece: Context -> See -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,right: Model.Content<Element>,context: Context,made: Model.Content<Element> & See) -> Model.Content<Element> & See:
  (left,state) = made
  +state = state
  (Model.Joined{left,Model.map(~Element,~piece(context,state),right)},state)

def walk_continued(~Element: Data,~State: Type,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> State -> Pieces.Pieces(~Element) & State,right: Pieces.Pieces(~Element),+context: Context,left: Pieces.Pieces(~Element) & State) -> Pieces.Pieces(~Element) & State:
  Storage.bind_pair(~Pieces.Pieces(~Element),~State,~(Pieces.Pieces(~Element) & State),left,left => state => walk_join(~Element,~State,left,Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,right,state)))

def walk_continue_packed(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> State -> Pieces.Pieces(~Element) & State,~piece: Context -> See -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,
  +right: Model.Model<Element>,+context: Context,-before: See,-left_model: Model.Content<Element>,+kept: Kept,+left: Model.Model<Element>,
  left_seen: {walk_seen(~Element,~State,~See,~see,(Model.pack(~Element,left),hold(kept))) == (left_model,before) : Model.Content<Element> & See},rule: @+kept: Kept -> {walk_seen(~Element,~State,~See,~see,Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,right),hold(kept))) == (Model.map(~Element,~piece(context,see(hold(kept))),Model.content(~Element,right)),see(hold(kept))) : Model.Content<Element> & See}) ->
  {walk_seen(~Element,~State,~See,~see,walk_continued(~Element,~State,~Context,~leaf,Model.pack(~Element,right),context,(Model.pack(~Element,left),hold(kept)))) == (Model.Joined{left_model,Model.map(~Element,~piece(context,before),Model.content(~Element,right))},before) : Model.Content<Element> & See}:
  Equal.trans(Model.Content<Element> & See,walk_seen(~Element,~State,~See,~see,walk_join(~Element,~State,Model.pack(~Element,left),Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,right),hold(kept)))),
    walk_joined(~Element,~See,Model.contents(~Element,Model.pack(~Element,left)),walk_seen(~Element,~State,~See,~see,Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,right),hold(kept)))),
    (Model.Joined{left_model,Model.map(~Element,~piece(context,before),Model.content(~Element,right))},before),
    walk_join_seen(~Element,~State,~See,~see,Model.pack(~Element,left),Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,right),hold(kept))),
    Equal.trans(Model.Content<Element> & See,walk_joined(~Element,~See,Model.contents(~Element,Model.pack(~Element,left)),walk_seen(~Element,~State,~See,~see,Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,right),hold(kept)))),
      walk_spread(~Element,~Context,~See,~piece,Model.content(~Element,right),context,(Model.contents(~Element,Model.pack(~Element,left)),see(hold(kept)))),
      (Model.Joined{left_model,Model.map(~Element,~piece(context,before),Model.content(~Element,right))},before),
      Equal.cong(Model.Content<Element> & See,Model.Content<Element> & See,made => walk_joined(~Element,~See,Model.contents(~Element,Model.pack(~Element,left)),made),
        walk_seen(~Element,~State,~See,~see,Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,right),hold(kept))),(Model.map(~Element,~piece(context,see(hold(kept))),Model.content(~Element,right)),see(hold(kept))),rule(kept)),
      Equal.cong(Model.Content<Element> & See,Model.Content<Element> & See,made => walk_spread(~Element,~Context,~See,~piece,Model.content(~Element,right),context,made),
        (Model.contents(~Element,Model.pack(~Element,left)),see(hold(kept))),(left_model,before),left_seen)))

def walk_continue_observed(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> State -> Pieces.Pieces(~Element) & State,~piece: Context -> See -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,
  +right: Model.Model<Element>,+context: Context,-before: See,-left_model: Model.Content<Element>,-state: State,-left: Pieces.Pieces(~Element),
  state_held: Held<State,Kept,hold,state>,left_observation: Model.Observation<Element,left>,
  left_seen: {walk_seen(~Element,~State,~See,~see,(left,state)) == (left_model,before) : Model.Content<Element> & See},rule: @+kept: Kept -> {walk_seen(~Element,~State,~See,~see,Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,right),hold(kept))) == (Model.map(~Element,~piece(context,see(hold(kept))),Model.content(~Element,right)),see(hold(kept))) : Model.Content<Element> & See}) ->
  {walk_seen(~Element,~State,~See,~see,walk_continued(~Element,~State,~Context,~leaf,Model.pack(~Element,right),context,(left,state))) == (Model.Joined{left_model,Model.map(~Element,~piece(context,before),Model.content(~Element,right))},before) : Model.Content<Element> & See}:
  Held{+kept,+state_equal} = state_held
  Model.Observation{+left_owner,+left_equal} = left_observation
  %state_equal : {walk_seen(~Element,~State,~See,~see,walk_continued(~Element,~State,~Context,~leaf,Model.pack(~Element,right),context,(left,_))) == (Model.Joined{left_model,Model.map(~Element,~piece(context,before),Model.content(~Element,right))},before) : Model.Content<Element> & See}
  %left_equal : {walk_seen(~Element,~State,~See,~see,walk_continued(~Element,~State,~Context,~leaf,Model.pack(~Element,right),context,(_,hold(kept)))) == (Model.Joined{left_model,Model.map(~Element,~piece(context,before),Model.content(~Element,right))},before) : Model.Content<Element> & See}
  walk_continue_packed(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~piece,right,context,before,left_model,kept,left_owner,
    %Equal.sym(Pieces.Pieces(~Element),Model.pack(~Element,left_owner),left,left_equal) : {walk_seen(~Element,~State,~See,~see,(_,hold(kept))) == (left_model,before) : Model.Content<Element> & See}
    %Equal.sym(State,hold(kept),state,state_equal) : {walk_seen(~Element,~State,~See,~see,(left,_)) == (left_model,before) : Model.Content<Element> & See}
    left_seen,rule)

def walk_continue(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> State -> Pieces.Pieces(~Element) & State,~piece: Context -> See -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,
  +right: Model.Model<Element>,+context: Context,-before: See,-left_model: Model.Content<Element>,left: Pieces.Pieces(~Element) & State,
  left_seen: {walk_seen(~Element,~State,~See,~see,left) == (left_model,before) : Model.Content<Element> & See},rule: @+kept: Kept -> {walk_seen(~Element,~State,~See,~see,Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,right),hold(kept))) == (Model.map(~Element,~piece(context,see(hold(kept))),Model.content(~Element,right)),see(hold(kept))) : Model.Content<Element> & See}) ->
  {walk_seen(~Element,~State,~See,~see,walk_continued(~Element,~State,~Context,~leaf,Model.pack(~Element,right),context,left)) == (Model.Joined{left_model,Model.map(~Element,~piece(context,before),Model.content(~Element,right))},before) : Model.Content<Element> & See}:
  (pieces,state) = left
  walk_continue_observed(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~piece,right,context,before,left_model,state,pieces,observe(state),Model.observe(~Element,pieces),left_seen,rule)

def walked(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> State -> Pieces.Pieces(~Element) & State,~piece: Context -> See -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,
  ~leaf_rule: WalkRule(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~piece),tree: Model.Model<Element>,+context: Context,+kept: Kept) ->
  {walk_seen(~Element,~State,~See,~see,Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,tree),hold(kept))) == (Model.map(~Element,~piece(context,see(hold(kept))),Model.content(~Element,tree)),see(hold(kept))) : Model.Content<Element> & See}:
  match tree:
    case Model.Leaf{+buffer,+first,+rows}: leaf_rule(context,first,rows,buffer,kept)
    case Model.Fork{+left,+right}:
      walk_continue(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~piece,right,context,see(hold(kept)),Model.map(~Element,~piece(context,see(hold(kept))),Model.content(~Element,left)),
        Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,left),hold(kept)),walked(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~piece,~leaf_rule,left,context,kept),
        next => walked(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~piece,~leaf_rule,right,context,next))

# ---- A walk on the tasks of a plan with a shared state ----

def contents_first(~Element: Data,~State: Type,~See: Data,~see: State -> See,made: Pieces.Pieces(~Element) & State) ->
  {Model.contents(~Element,Pair.fst(Pieces.Pieces(~Element),State,made)) == Pair.fst(Model.Content<Element>,See,walk_seen(~Element,~State,~See,~see,made)) : Model.Content<Element>}:
  (pieces,state) = made
  {==}

# The walk of one task maps every piece of its tree.
def task_walked(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> State -> Pieces.Pieces(~Element) & State,
  ~piece: Context -> See -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,
  ~leaf_rule: WalkRule(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~piece),+tree: Model.Model<Element>,+context: Context,+kept: Kept) ->
  {Model.contents(~Element,Pair.fst(Pieces.Pieces(~Element),State,Pieces.walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,tree),hold(kept)))) == Model.map(~Element,~piece(context,see(hold(kept))),Model.content(~Element,tree)) : Model.Content<Element>}:
  %walk_model_same(~Element,~State,~Context,~Kept,~hold,~observe,~leaf,tree,context,kept) :
    {Model.contents(~Element,Pair.fst(Pieces.Pieces(~Element),State,_)) == Model.map(~Element,~piece(context,see(hold(kept))),Model.content(~Element,tree)) : Model.Content<Element>}
  Equal.trans(Model.Content<Element>,Model.contents(~Element,Pair.fst(Pieces.Pieces(~Element),State,Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,tree),hold(kept)))),
    Pair.fst(Model.Content<Element>,See,walk_seen(~Element,~State,~See,~see,Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,tree),hold(kept)))),Model.map(~Element,~piece(context,see(hold(kept))),Model.content(~Element,tree)),
    contents_first(~Element,~State,~See,~see,Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,tree),hold(kept))),
    Equal.cong(Model.Content<Element> & See,Model.Content<Element>,made => Pair.fst(Model.Content<Element>,See,made),walk_seen(~Element,~State,~See,~see,Walk.model_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,tree),hold(kept))),
      (Model.map(~Element,~piece(context,see(hold(kept))),Model.content(~Element,tree)),see(hold(kept))),walked(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~piece,~leaf_rule,tree,context,kept)))

# The tasks of any plan sharing a state the leaves give back as they found it
# map every piece with what is seen of the state.
def grouped_walked(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> State -> Pieces.Pieces(~Element) & State,
  ~piece: Context -> See -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,
  ~leaf_rule: WalkRule(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~piece),
  ~share: State -> State & State,~split: Kept -> PartitionLaws.Split<Kept>,
  ~shared: @+kept: Kept -> {share(hold(kept)) == (hold(PartitionLaws.first(~Kept,split(kept))),hold(PartitionLaws.second(~Kept,split(kept)))) : State & State},
  ~left_seen: @+kept: Kept -> {see(hold(PartitionLaws.first(~Kept,split(kept)))) == see(hold(kept)) : See},~right_seen: @+kept: Kept -> {see(hold(PartitionLaws.second(~Kept,split(kept)))) == see(hold(kept)) : See},
  layout: Pieces.Ranges,+mode: Bool,+enabled: Bool,+context: Context,+tree: Model.Model<Element>,+kept: Kept) ->
  {Model.contents(~Element,Pieces.each_part(~Pieces.Pieced(~Storage.Buffer<Element>,~State),~Storage.Buffer<Element>,~Unit,~Context,~Bool,~Pieces.in_parallel,~Pieces.piece_divide(~Storage.Buffer<Element>,~State,~Context,~share),~Pieces.task_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf),~Pieces.task_kept(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf),layout,mode,enabled,context,(Model.pack(~Element,tree),hold(kept)))) == Model.map(~Element,~piece(context,see(hold(kept))),Model.content(~Element,tree)) : Model.Content<Element>}:
  match layout tree:
    case Pieces.Range{first,rows} tree: task_walked(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~piece,~leaf_rule,tree,context,kept)
    case Pieces.Split{+left_plan,+right_plan} Model.Leaf{+buffer,+first,+rows}:
      %Equal.sym(Pieces.Pieces(~Element),Pieces.each_part(~Pieces.Pieced(~Storage.Buffer<Element>,~State),~Storage.Buffer<Element>,~Unit,~Context,~Bool,~Pieces.in_parallel,~Pieces.piece_divide(~Storage.Buffer<Element>,~State,~Context,~share),~Pieces.task_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf),~Pieces.task_kept(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf),Pieces.Split{left_plan,right_plan},mode,enabled,context,(Model.pack(~Element,Model.Leaf{buffer,first,rows}),hold(kept))),
          Walk.sequential(~Pieces.Pieced(~Storage.Buffer<Element>,~State),~Storage.Buffer<Element>,~Unit,~Context,~Bool,~Pieces.in_parallel,~Pieces.piece_divide(~Storage.Buffer<Element>,~State,~Context,~share),~Pieces.task_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf),~Pieces.task_kept(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf),left_plan,right_plan,enabled,context,
            Pieces.piece_divide(~Storage.Buffer<Element>,~State,~Context,~share,context,left_plan,right_plan,(Model.pack(~Element,Model.Leaf{buffer,first,rows}),hold(kept)))),
          Walk.fork(~Pieces.Pieced(~Storage.Buffer<Element>,~State),~Storage.Buffer<Element>,~Unit,~Context,~Bool,~Pieces.in_parallel,~Pieces.piece_divide(~Storage.Buffer<Element>,~State,~Context,~share),~Pieces.task_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf),~Pieces.task_kept(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf),left_plan,right_plan,mode,enabled,context,(Model.pack(~Element,Model.Leaf{buffer,first,rows}),hold(kept)))) :
        {Model.contents(~Element,_) == Model.map(~Element,~piece(context,see(hold(kept))),Model.content(~Element,Model.Leaf{buffer,first,rows})) : Model.Content<Element>}
      task_walked(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~piece,~leaf_rule,Model.Leaf{buffer,first,rows},context,kept)
    case Pieces.Split{+left_plan,+right_plan} Model.Fork{+left,+right}:
      %Equal.sym(Pieces.Pieces(~Element),Pieces.each_part(~Pieces.Pieced(~Storage.Buffer<Element>,~State),~Storage.Buffer<Element>,~Unit,~Context,~Bool,~Pieces.in_parallel,~Pieces.piece_divide(~Storage.Buffer<Element>,~State,~Context,~share),~Pieces.task_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf),~Pieces.task_kept(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf),Pieces.Split{left_plan,right_plan},mode,enabled,context,(Model.pack(~Element,Model.Fork{left,right}),hold(kept))),
          Walk.sequential(~Pieces.Pieced(~Storage.Buffer<Element>,~State),~Storage.Buffer<Element>,~Unit,~Context,~Bool,~Pieces.in_parallel,~Pieces.piece_divide(~Storage.Buffer<Element>,~State,~Context,~share),~Pieces.task_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf),~Pieces.task_kept(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf),left_plan,right_plan,enabled,context,
            Pieces.piece_divide(~Storage.Buffer<Element>,~State,~Context,~share,context,left_plan,right_plan,(Model.pack(~Element,Model.Fork{left,right}),hold(kept)))),
          Walk.fork(~Pieces.Pieced(~Storage.Buffer<Element>,~State),~Storage.Buffer<Element>,~Unit,~Context,~Bool,~Pieces.in_parallel,~Pieces.piece_divide(~Storage.Buffer<Element>,~State,~Context,~share),~Pieces.task_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf),~Pieces.task_kept(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf),left_plan,right_plan,mode,enabled,context,(Model.pack(~Element,Model.Fork{left,right}),hold(kept)))) :
        {Model.contents(~Element,_) == Model.map(~Element,~piece(context,see(hold(kept))),Model.content(~Element,Model.Fork{left,right})) : Model.Content<Element>}
      %Equal.sym(State & State,share(hold(kept)),(hold(PartitionLaws.first(~Kept,split(kept))),hold(PartitionLaws.second(~Kept,split(kept)))),shared(kept)) :
        {Model.contents(~Element,Walk.sequential(~Pieces.Pieced(~Storage.Buffer<Element>,~State),~Storage.Buffer<Element>,~Unit,~Context,~Bool,~Pieces.in_parallel,~Pieces.piece_divide(~Storage.Buffer<Element>,~State,~Context,~share),~Pieces.task_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf),~Pieces.task_kept(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf),left_plan,right_plan,enabled,context,
          Pieces.shared_halves(~Storage.Buffer<Element>,~State,Model.pack(~Element,left),Model.pack(~Element,right),_))) == Model.map(~Element,~piece(context,see(hold(kept))),Model.content(~Element,Model.Fork{left,right})) : Model.Content<Element>}
      %Equal.sym(Model.Content<Element>,Model.contents(~Element,Pieces.each_part(~Pieces.Pieced(~Storage.Buffer<Element>,~State),~Storage.Buffer<Element>,~Unit,~Context,~Bool,~Pieces.in_parallel,~Pieces.piece_divide(~Storage.Buffer<Element>,~State,~Context,~share),~Pieces.task_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf),~Pieces.task_kept(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf),left_plan,enabled,enabled,context,(Model.pack(~Element,left),hold(PartitionLaws.first(~Kept,split(kept)))))),
          Model.map(~Element,~piece(context,see(hold(PartitionLaws.first(~Kept,split(kept))))),Model.content(~Element,left)),
          grouped_walked(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~piece,~leaf_rule,~share,~split,~shared,~left_seen,~right_seen,left_plan,enabled,enabled,context,left,PartitionLaws.first(~Kept,split(kept)))) :
        {Model.Joined{_,Model.contents(~Element,Pieces.each_part(~Pieces.Pieced(~Storage.Buffer<Element>,~State),~Storage.Buffer<Element>,~Unit,~Context,~Bool,~Pieces.in_parallel,~Pieces.piece_divide(~Storage.Buffer<Element>,~State,~Context,~share),~Pieces.task_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf),~Pieces.task_kept(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf),right_plan,enabled,enabled,context,(Model.pack(~Element,right),hold(PartitionLaws.second(~Kept,split(kept))))))} ==
          Model.map(~Element,~piece(context,see(hold(kept))),Model.content(~Element,Model.Fork{left,right})) : Model.Content<Element>}
      %Equal.sym(Model.Content<Element>,Model.contents(~Element,Pieces.each_part(~Pieces.Pieced(~Storage.Buffer<Element>,~State),~Storage.Buffer<Element>,~Unit,~Context,~Bool,~Pieces.in_parallel,~Pieces.piece_divide(~Storage.Buffer<Element>,~State,~Context,~share),~Pieces.task_walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf),~Pieces.task_kept(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf),right_plan,enabled,enabled,context,(Model.pack(~Element,right),hold(PartitionLaws.second(~Kept,split(kept)))))),
          Model.map(~Element,~piece(context,see(hold(PartitionLaws.second(~Kept,split(kept))))),Model.content(~Element,right)),
          grouped_walked(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~piece,~leaf_rule,~share,~split,~shared,~left_seen,~right_seen,right_plan,enabled,enabled,context,right,PartitionLaws.second(~Kept,split(kept)))) :
        {Model.Joined{Model.map(~Element,~piece(context,see(hold(PartitionLaws.first(~Kept,split(kept))))),Model.content(~Element,left)),_} == Model.map(~Element,~piece(context,see(hold(kept))),Model.content(~Element,Model.Fork{left,right})) : Model.Content<Element>}
      %Equal.sym(See,see(hold(PartitionLaws.first(~Kept,split(kept)))),see(hold(kept)),left_seen(kept)) :
        {Model.Joined{Model.map(~Element,~piece(context,_),Model.content(~Element,left)),Model.map(~Element,~piece(context,see(hold(PartitionLaws.second(~Kept,split(kept))))),Model.content(~Element,right))} ==
          Model.map(~Element,~piece(context,see(hold(kept))),Model.content(~Element,Model.Fork{left,right})) : Model.Content<Element>}
      %Equal.sym(See,see(hold(PartitionLaws.second(~Kept,split(kept)))),see(hold(kept)),right_seen(kept)) :
        {Model.Joined{Model.map(~Element,~piece(context,see(hold(kept))),Model.content(~Element,left)),Model.map(~Element,~piece(context,_),Model.content(~Element,right))} ==
          Model.map(~Element,~piece(context,see(hold(kept))),Model.content(~Element,Model.Fork{left,right})) : Model.Content<Element>}
      {==}

def shaped_valued_plan(~Part: Data,~Value: Data,~Context: Data,~value: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 -> Value,+context: Context,tree: Model.Shaped<Part>) ->
  {Model.shaped_plan(~Value,~piece_rows(~Value),Model.shaped_valued(~Part,~Value,~Context,~value,context,tree)) == Model.shaped_plan(~Part,~piece_rows(~Part),tree) : Pieces.Ranges}:
  match tree:
    case Model.Piece{+part,+first,+rows}: {==}
    case Model.Pieces{+left,+right}:
      +valued_left = Model.shaped_plan(~Value,~piece_rows(~Value),Model.shaped_valued(~Part,~Value,~Context,~value,context,left))
      +valued_right = Model.shaped_plan(~Value,~piece_rows(~Value),Model.shaped_valued(~Part,~Value,~Context,~value,context,right))
      +plain_left = Model.shaped_plan(~Part,~piece_rows(~Part),left)
      +plain_right = Model.shaped_plan(~Part,~piece_rows(~Part),right)
      Equal.trans(Pieces.Ranges,Pieces.Split{valued_left,valued_right},Pieces.Split{plain_left,valued_right},Pieces.Split{plain_left,plain_right},
        Equal.cong(Pieces.Ranges,Pieces.Ranges,plan => Pieces.Split{plan,valued_right},valued_left,plain_left,shaped_valued_plan(~Part,~Value,~Context,~value,context,left)),
        Equal.cong(Pieces.Ranges,Pieces.Ranges,plan => Pieces.Split{plain_left,plan},valued_right,plain_right,shaped_valued_plan(~Part,~Value,~Context,~value,context,right)))

def shaped_leaf_valued(~Part: Data,~Block: Type,~pack: Part -> Block,~Value: Data,~Output: Type,~pack_value: Value -> Output,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Block -> Unit -> Tree(~Output),~value: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 -> Value,
  ~rule: @+context: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 ->
    {leaf(context,first,rows,pack(part),Unit{}) == Partition.Leaf{pack_value(value(context,part,first,rows)),first,rows,Unit{}} : Tree(~Output)},
  +context: Context,tree: Model.Shaped<Part>) ->
  {shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,tree) ==
    Model.shaped_pack(~Value,~Output,~pack_value,Model.shaped_valued(~Part,~Value,~Context,~value,context,tree)) : Tree(~Output)}:
  match tree:
    case Model.Piece{+part,+first,+rows}: rule(context,part,first,rows)
    case Model.Pieces{+left,+right}:
      %Equal.sym(Tree(~Output),shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,left),
          Model.shaped_pack(~Value,~Output,~pack_value,Model.shaped_valued(~Part,~Value,~Context,~value,context,left)),
          shaped_leaf_valued(~Part,~Block,~pack,~Value,~Output,~pack_value,~Context,~leaf,~value,~rule,context,left)) :
        {Partition.Fork{_,shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,right)} ==
          Partition.Fork{Model.shaped_pack(~Value,~Output,~pack_value,Model.shaped_valued(~Part,~Value,~Context,~value,context,left)),
            Model.shaped_pack(~Value,~Output,~pack_value,Model.shaped_valued(~Part,~Value,~Context,~value,context,right))} : Tree(~Output)}
      %Equal.sym(Tree(~Output),shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,right),
          Model.shaped_pack(~Value,~Output,~pack_value,Model.shaped_valued(~Part,~Value,~Context,~value,context,right)),
          shaped_leaf_valued(~Part,~Block,~pack,~Value,~Output,~pack_value,~Context,~leaf,~value,~rule,context,right)) :
        {Partition.Fork{Model.shaped_pack(~Value,~Output,~pack_value,Model.shaped_valued(~Part,~Value,~Context,~value,context,left)),_} ==
          Partition.Fork{Model.shaped_pack(~Value,~Output,~pack_value,Model.shaped_valued(~Part,~Value,~Context,~value,context,left)),
            Model.shaped_pack(~Value,~Output,~pack_value,Model.shaped_valued(~Part,~Value,~Context,~value,context,right))} : Tree(~Output)}
      {==}

# ---- Lists written at the first row of their piece ----

# The lists of the pieces, one after the other.
def shaped_list(~Part: Data,~Element: Data,~Context: Data,~list: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 -> List<&2,Element>,+context: Context,tree: Model.Shaped<Part>) ->
  List<&2,Element>:
  match tree:
    case Model.Piece{+part,+first,+rows}: list(context,part,first,rows)
    case Model.Pieces{left,right}:
      List.append(&2,Element,shaped_list(~Part,~Element,~Context,~list,context,left),shaped_list(~Part,~Element,~Context,~list,context,right))

# The lists of valued pieces are the lists of the pieces read through their
# value.
def shaped_list_valued(~Part: Data,~Value: Data,~Element: Data,~Context: Data,~Outer: Data,
  ~value: Outer -> @+part: Part -> @+first: U32 -> @+rows: U32 -> Value,~list: Context -> @+part: Value -> @+first: U32 -> @+rows: U32 -> List<&2,Element>,
  ~context: @+outer: Outer -> Context,~composed: Outer -> @+part: Part -> @+first: U32 -> @+rows: U32 -> List<&2,Element>,
  ~rule: @+outer: Outer -> @+part: Part -> @+first: U32 -> @+rows: U32 -> {list(context(outer),value(outer,part,first,rows),first,rows) == composed(outer,part,first,rows) : List<&2,Element>},
  +outer: Outer,tree: Model.Shaped<Part>) ->
  {shaped_list(~Value,~Element,~Context,~list,context(outer),Model.shaped_valued(~Part,~Value,~Outer,~value,outer,tree)) == shaped_list(~Part,~Element,~Outer,~composed,outer,tree) : List<&2,Element>}:
  match tree:
    case Model.Piece{+part,+first,+rows}: rule(outer,part,first,rows)
    case Model.Pieces{+left,+right}:
      %Equal.sym(List<&2,Element>,shaped_list(~Value,~Element,~Context,~list,context(outer),Model.shaped_valued(~Part,~Value,~Outer,~value,outer,left)),
          shaped_list(~Part,~Element,~Outer,~composed,outer,left),shaped_list_valued(~Part,~Value,~Element,~Context,~Outer,~value,~list,~context,~composed,~rule,outer,left)) :
        {List.append(&2,Element,_,shaped_list(~Value,~Element,~Context,~list,context(outer),Model.shaped_valued(~Part,~Value,~Outer,~value,outer,right))) ==
          List.append(&2,Element,shaped_list(~Part,~Element,~Outer,~composed,outer,left),shaped_list(~Part,~Element,~Outer,~composed,outer,right)) : List<&2,Element>}
      %Equal.sym(List<&2,Element>,shaped_list(~Value,~Element,~Context,~list,context(outer),Model.shaped_valued(~Part,~Value,~Outer,~value,outer,right)),
          shaped_list(~Part,~Element,~Outer,~composed,outer,right),shaped_list_valued(~Part,~Value,~Element,~Context,~Outer,~value,~list,~context,~composed,~rule,outer,right)) :
        {List.append(&2,Element,shaped_list(~Part,~Element,~Outer,~composed,outer,left),_) ==
          List.append(&2,Element,shaped_list(~Part,~Element,~Outer,~composed,outer,left),shaped_list(~Part,~Element,~Outer,~composed,outer,right)) : List<&2,Element>}
      {==}

# A step that writes the list of its piece at its first row, width elements a
# row.
def WriteRule(~Part: Data,~Element: Data,~Context: Data,~step: Context -> @+first: U32 -> @+rows: U32 -> Part -> Buffer.Content<Element> -> Buffer.Content<Element>,
  ~list: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 -> List<&2,Element>,~width: Context -> Nat) -> Type:
  @+context: Context -> @+first: U32 -> @+rows: U32 -> @+part: Part -> @+tree: Tree.Tree<Element> -> @+length: Nat ->
  {step(context,first,rows,part,Buffer.Content{tree,length}) ==
    Buffer.Content{Tree.write_sequence(~Element,list(context,part,first,rows),U32.from_nat(Nat.mul(U32.to_nat(first),width(context))),tree),length} : Buffer.Content<Element>}

def LengthRule(~Part: Data,~Element: Data,~Context: Data,~list: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 -> List<&2,Element>,~width: Context -> Nat) -> Type:
  @+context: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 -> {List.length(&2,Element,list(context,part,first,rows)) == Nat.mul(U32.to_nat(rows),width(context)) : Nat}

# The lists of pieces that cover rows hold width elements a row.
def shaped_list_length(~Part: Data,~Element: Data,~Context: Data,~list: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 -> List<&2,Element>,~width: Context -> Nat,
  ~length_rule: LengthRule(~Part,~Element,~Context,~list,~width),+context: Context,tree: Model.Shaped<Part>,+first: Nat,+rows: Nat,
  coverage: PiecesPlan.Coverage(Model.shaped_plan(~Part,~piece_rows(~Part),tree),first,rows)) ->
  {List.length(&2,Element,shaped_list(~Part,~Element,~Context,~list,context,tree)) == Nat.mul(rows,width(context)) : Nat}:
  match tree:
    case Model.Piece{+part,+native_first,+native_rows}:
      PiecesPlan.LeafCoverage{first_equal,+rows_equal} = coverage
      %rows_equal : {List.length(&2,Element,list(context,part,native_first,native_rows)) == Nat.mul(_,width(context)) : Nat}
      length_rule(context,part,native_first,native_rows)
    case Model.Pieces{+left,+right}:
      PiecesPlan.ForkCoverage{+left_rows,+right_rows,+rows_equal,+left_coverage,+right_coverage} = coverage
      +width_value = width(context)
      +left_list = shaped_list(~Part,~Element,~Context,~list,context,left)
      +right_list = shaped_list(~Part,~Element,~Context,~list,context,right)
      %Equal.sym(Nat,rows,Nat.add(left_rows,right_rows),rows_equal) :
        {List.length(&2,Element,List.append(&2,Element,left_list,right_list)) == Nat.mul(_,width_value) : Nat}
      %Equal.sym(Nat,Nat.mul(Nat.add(left_rows,right_rows),width_value),Nat.add(Nat.mul(left_rows,width_value),Nat.mul(right_rows,width_value)),RowMajor.distrib(left_rows,right_rows,width_value)) :
        {List.length(&2,Element,List.append(&2,Element,left_list,right_list)) == _ : Nat}
      %Equal.sym(Nat,List.length(&2,Element,List.append(&2,Element,left_list,right_list)),Nat.add(List.length(&2,Element,left_list),List.length(&2,Element,right_list)),
          ListRead.append_length(~Element,left_list,right_list)) :
        {_ == Nat.add(Nat.mul(left_rows,width_value),Nat.mul(right_rows,width_value)) : Nat}
      %Equal.sym(Nat,List.length(&2,Element,left_list),Nat.mul(left_rows,width_value),
          shaped_list_length(~Part,~Element,~Context,~list,~width,~length_rule,context,left,first,left_rows,left_coverage)) :
        {Nat.add(_,List.length(&2,Element,right_list)) == Nat.add(Nat.mul(left_rows,width_value),Nat.mul(right_rows,width_value)) : Nat}
      %Equal.sym(Nat,List.length(&2,Element,right_list),Nat.mul(right_rows,width_value),
          shaped_list_length(~Part,~Element,~Context,~list,~width,~length_rule,context,right,Nat.add(first,left_rows),right_rows,right_coverage)) :
        {Nat.add(Nat.mul(left_rows,width_value),_) == Nat.add(Nat.mul(left_rows,width_value),Nat.mul(right_rows,width_value)) : Nat}
      {==}

def Written(~Part: Data,~Element: Data,~Context: Data,~step: Context -> @+first: U32 -> @+rows: U32 -> Part -> Buffer.Content<Element> -> Buffer.Content<Element>,
  ~list: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 -> List<&2,Element>,~width: Context -> Nat,
  +context: Context,tree: Model.Shaped<Part>,+first: Nat,+before: Tree.Tree<Element>,+length: Nat) -> Data:
  {Model.shaped_thread(~Part,~Buffer.Content<Element>,~Context,~step,context,tree,Buffer.Content{before,length}) ==
    Buffer.Content{Tree.write_sequence(~Element,shaped_list(~Part,~Element,~Context,~list,context,tree),U32.from_nat(Nat.mul(first,width(context))),before),length} : Buffer.Content<Element>}

def written_joined(~Part: Data,~Element: Data,~Context: Data,~step: Context -> @+first: U32 -> @+rows: U32 -> Part -> Buffer.Content<Element> -> Buffer.Content<Element>,
  ~list: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 -> List<&2,Element>,~width: Context -> Nat,
  +context: Context,+left: Model.Shaped<Part>,+right: Model.Shaped<Part>,+first: Nat,+left_rows: Nat,+before: Tree.Tree<Element>,+length: Nat,
  +left_length: {List.length(&2,Element,shaped_list(~Part,~Element,~Context,~list,context,left)) == Nat.mul(left_rows,width(context)) : Nat},
  +left_thread: Written(~Part,~Element,~Context,~step,~list,~width,context,left,first,before,length),
  right_thread: Written(~Part,~Element,~Context,~step,~list,~width,context,right,Nat.add(first,left_rows),
    Tree.write_sequence(~Element,shaped_list(~Part,~Element,~Context,~list,context,left),U32.from_nat(Nat.mul(first,width(context))),before),length)) ->
  Written(~Part,~Element,~Context,~step,~list,~width,context,Model.Pieces{left,right},first,before,length):
  +width_value = width(context)
  +left_list = shaped_list(~Part,~Element,~Context,~list,context,left)
  +right_list = shaped_list(~Part,~Element,~Context,~list,context,right)
  +offset = U32.from_nat(Nat.mul(first,width_value))
  +middle = Tree.write_sequence(~Element,left_list,offset,before)
  +next = U32.from_nat(Nat.mul(Nat.add(first,left_rows),width_value))
  +end = {
    %Equal.sym(U32,ListRead.end_position(~Element,left_list,offset),(offset + U32.from_nat(List.length(&2,Element,left_list)) : U32),ListRead.end_position_length(~Element,left_list,offset)) :
      {_ == next : U32}
    %Equal.sym(Nat,List.length(&2,Element,left_list),Nat.mul(left_rows,width_value),left_length) : {(offset + U32.from_nat(_) : U32) == next : U32}
    %Equal.sym(Nat,Nat.mul(Nat.add(first,left_rows),width_value),Nat.add(Nat.mul(first,width_value),Nat.mul(left_rows,width_value)),RowMajor.distrib(first,left_rows,width_value)) :
      {(offset + U32.from_nat(Nat.mul(left_rows,width_value)) : U32) == U32.from_nat(_) : U32}
    WordArithmetic.native_add(Nat.mul(first,width_value),Nat.mul(left_rows,width_value)) :
    {ListRead.end_position(~Element,left_list,offset) == next : U32}}
  %Equal.sym(Buffer.Content<Element>,Model.shaped_thread(~Part,~Buffer.Content<Element>,~Context,~step,context,left,Buffer.Content{before,length}),Buffer.Content{middle,length},left_thread) :
    {Model.shaped_thread(~Part,~Buffer.Content<Element>,~Context,~step,context,right,_) ==
      Buffer.Content{Tree.write_sequence(~Element,List.append(&2,Element,left_list,right_list),offset,before),length} : Buffer.Content<Element>}
  %Equal.sym(Tree.Tree<Element>,Tree.write_sequence(~Element,List.append(&2,Element,left_list,right_list),offset,before),
      Tree.write_sequence(~Element,right_list,ListRead.end_position(~Element,left_list,offset),middle),ListRead.concatenated(~Element,left_list,right_list,offset,before)) :
    {Model.shaped_thread(~Part,~Buffer.Content<Element>,~Context,~step,context,right,Buffer.Content{middle,length}) == Buffer.Content{_,length} : Buffer.Content<Element>}
  %Equal.sym(U32,ListRead.end_position(~Element,left_list,offset),next,end) :
    {Model.shaped_thread(~Part,~Buffer.Content<Element>,~Context,~step,context,right,Buffer.Content{middle,length}) ==
      Buffer.Content{Tree.write_sequence(~Element,right_list,_,middle),length} : Buffer.Content<Element>}
  right_thread

# Over pieces that cover rows [first, first + rows) in order, the steps write
# the lists of the pieces one after the other from the first row.
def shaped_written(~Part: Data,~Element: Data,~Context: Data,~step: Context -> @+first: U32 -> @+rows: U32 -> Part -> Buffer.Content<Element> -> Buffer.Content<Element>,
  ~list: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 -> List<&2,Element>,~width: Context -> Nat,
  ~rule: WriteRule(~Part,~Element,~Context,~step,~list,~width),~length_rule: LengthRule(~Part,~Element,~Context,~list,~width),
  +context: Context,tree: Model.Shaped<Part>,+first: Nat,+rows: Nat,coverage: PiecesPlan.Coverage(Model.shaped_plan(~Part,~piece_rows(~Part),tree),first,rows),
  +before: Tree.Tree<Element>,+length: Nat) ->
  Written(~Part,~Element,~Context,~step,~list,~width,context,tree,first,before,length):
  match tree:
    case Model.Piece{+part,+native_first,+native_rows}:
      PiecesPlan.LeafCoverage{+first_equal,rows_equal} = coverage
      %first_equal :
        {step(context,native_first,native_rows,part,Buffer.Content{before,length}) ==
          Buffer.Content{Tree.write_sequence(~Element,list(context,part,native_first,native_rows),U32.from_nat(Nat.mul(_,width(context))),before),length} : Buffer.Content<Element>}
      rule(context,native_first,native_rows,part,before,length)
    case Model.Pieces{+left,+right}:
      PiecesPlan.ForkCoverage{+left_rows,+right_rows,rows_equal,+left_coverage,+right_coverage} = coverage
      written_joined(~Part,~Element,~Context,~step,~list,~width,context,left,right,first,left_rows,before,length,
        shaped_list_length(~Part,~Element,~Context,~list,~width,~length_rule,context,left,first,left_rows,left_coverage),
        shaped_written(~Part,~Element,~Context,~step,~list,~width,~rule,~length_rule,context,left,first,left_rows,left_coverage,before,length),
        shaped_written(~Part,~Element,~Context,~step,~list,~width,~rule,~length_rule,context,right,Nat.add(first,left_rows),right_rows,right_coverage,
          Tree.write_sequence(~Element,shaped_list(~Part,~Element,~Context,~list,context,left),U32.from_nat(Nat.mul(first,width(context))),before),length))

# ---- Elements of the lists of pieces ----

# An element of the lists of pieces that cover rows [first, first + rows) is
# the element of its piece: a property of an element at its global position
# holds of every element when it holds of the elements of every piece.
def ListRule(~Part: Data,~Element: Data,~Context: Data,~list: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 -> List<&2,Element>,~width: Context -> Nat,
  ~Scope: Data,~context: @+scope: Scope -> Context,~fill: @+scope: Scope -> Element,~limit: @+scope: Scope -> Nat,
  ~Fact: @+scope: Scope -> @+part: Part -> @+first: U32 -> @+rows: U32 -> Data,~Holds: @+scope: Scope -> @+index: Nat -> @+element: Element -> Data) -> Type:
  @+scope: Scope -> @+part: Part -> @+first: U32 -> @+rows: U32 -> @+fact: Fact(scope,part,first,rows) ->
    @+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),limit(scope)) == True{} : Bool} ->
    @+local: Nat -> @+inside: {Nat.is_lt(local,Nat.mul(U32.to_nat(rows),width(context(scope)))) == True{} : Bool} ->
    Holds(scope,Nat.add(Nat.mul(U32.to_nat(first),width(context(scope))),local),ListRead.at(~Element,fill(scope),local,list(context(scope),part,first,rows)))

def ListAt(~Part: Data,~Element: Data,~Context: Data,~list: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 -> List<&2,Element>,~width: Context -> Nat,
  ~Scope: Data,~context: @+scope: Scope -> Context,~fill: @+scope: Scope -> Element,~Holds: @+scope: Scope -> @+index: Nat -> @+element: Element -> Data,
  +scope: Scope,tree: Model.Shaped<Part>,+first: Nat,+index: Nat) -> Data:
  Holds(scope,Nat.add(Nat.mul(first,width(context(scope))),index),ListRead.at(~Element,fill(scope),index,shaped_list(~Part,~Element,~Context,~list,context(scope),tree)))

def list_at_branch(~Part: Data,~Element: Data,~Context: Data,~list: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 -> List<&2,Element>,~width: Context -> Nat,
  ~Scope: Data,~context: @+scope: Scope -> Context,~fill: @+scope: Scope -> Element,~Holds: @+scope: Scope -> @+index: Nat -> @+element: Element -> Data,
  +scope: Scope,+left: Model.Shaped<Part>,+right: Model.Shaped<Part>,+first: Nat,+left_rows: Nat,+right_rows: Nat,+index: Nat,choose_left: Bool,
  +left_length: {List.length(&2,Element,shaped_list(~Part,~Element,~Context,~list,context(scope),left)) == Nat.mul(left_rows,width(context(scope))) : Nat},
  +choice: {Nat.is_lt(index,Nat.mul(left_rows,width(context(scope)))) == choose_left : Bool},
  +inside: {Nat.is_lt(index,Nat.add(Nat.mul(left_rows,width(context(scope))),Nat.mul(right_rows,width(context(scope))))) == True{} : Bool},
  left_case: {Nat.is_lt(index,Nat.mul(left_rows,width(context(scope)))) == True{} : Bool} ->
    ListAt(~Part,~Element,~Context,~list,~width,~Scope,~context,~fill,~Holds,scope,left,first,index),
  right_case: @+tail: Nat -> {Nat.is_lt(tail,Nat.mul(right_rows,width(context(scope)))) == True{} : Bool} ->
    ListAt(~Part,~Element,~Context,~list,~width,~Scope,~context,~fill,~Holds,scope,right,Nat.add(first,left_rows),tail)) ->
  ListAt(~Part,~Element,~Context,~list,~width,~Scope,~context,~fill,~Holds,scope,Model.Pieces{left,right},first,index):
  match choose_left:
    case True{}:
      +width_value = width(context(scope))
      +left_list = shaped_list(~Part,~Element,~Context,~list,context(scope),left)
      +right_list = shaped_list(~Part,~Element,~Context,~list,context(scope),right)
      +left_count = Nat.mul(left_rows,width_value)
      %Equal.sym(Element,ListRead.at(~Element,fill(scope),index,List.append(&2,Element,left_list,right_list)),ListRead.at(~Element,fill(scope),index,left_list),
          ListRead.appended_before(~Element,fill(scope),left_list,right_list,index,
            %Equal.sym(Nat,List.length(&2,Element,left_list),left_count,left_length) : {Nat.is_lt(index,_) == True{} : Bool} choice)) :
        Holds(scope,Nat.add(Nat.mul(first,width_value),index),_)
      left_case(choice)
    case False{}:
      +width_value = width(context(scope))
      +left_list = shaped_list(~Part,~Element,~Context,~list,context(scope),left)
      +right_list = shaped_list(~Part,~Element,~Context,~list,context(scope),right)
      +left_count = Nat.mul(left_rows,width_value)
      +above = Order.ge_true(index,left_count,Order.comparison_not_less(Nat.cmp(index,left_count),choice))
      +tail = Nat.sub(index,left_count)
      +restored = Order.restore_shift(left_count,index,above)
      +global = {
        %Equal.sym(Nat,Nat.mul(Nat.add(first,left_rows),width_value),Nat.add(Nat.mul(first,width_value),left_count),RowMajor.distrib(first,left_rows,width_value)) :
          {Nat.add(_,tail) == Nat.add(Nat.mul(first,width_value),index) : Nat}
        %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(first,width_value),left_count),tail),Nat.add(Nat.mul(first,width_value),Nat.add(left_count,tail)),
            Addition.associative(Nat.mul(first,width_value),left_count,tail)) :
          {_ == Nat.add(Nat.mul(first,width_value),index) : Nat}
        %Equal.sym(Nat,Nat.add(left_count,tail),index,restored) : {Nat.add(Nat.mul(first,width_value),_) == Nat.add(Nat.mul(first,width_value),index) : Nat}
        {==} : {Nat.add(Nat.mul(Nat.add(first,left_rows),width_value),tail) == Nat.add(Nat.mul(first,width_value),index) : Nat}}
      %restored : Holds(scope,Nat.add(Nat.mul(first,width_value),index),ListRead.at(~Element,fill(scope),_,List.append(&2,Element,left_list,right_list)))
      %left_length : Holds(scope,Nat.add(Nat.mul(first,width_value),index),ListRead.at(~Element,fill(scope),Nat.add(_,tail),List.append(&2,Element,left_list,right_list)))
      %global : Holds(scope,_,ListRead.at(~Element,fill(scope),Nat.add(List.length(&2,Element,left_list),tail),List.append(&2,Element,left_list,right_list)))
      %Equal.sym(Element,ListRead.at(~Element,fill(scope),Nat.add(List.length(&2,Element,left_list),tail),List.append(&2,Element,left_list,right_list)),
          ListRead.at(~Element,fill(scope),tail,right_list),ListRead.appended_after(~Element,fill(scope),left_list,right_list,tail)) :
        Holds(scope,Nat.add(Nat.mul(Nat.add(first,left_rows),width_value),tail),_)
      right_case(tail,Order.sub_lt(index,left_count,Nat.mul(right_rows,width_value),above,inside))

def shaped_list_at(~Part: Data,~Element: Data,~Context: Data,~list: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 -> List<&2,Element>,~width: Context -> Nat,
  ~length_rule: LengthRule(~Part,~Element,~Context,~list,~width),
  ~Scope: Data,~context: @+scope: Scope -> Context,~fill: @+scope: Scope -> Element,~limit: @+scope: Scope -> Nat,
  ~Fact: @+scope: Scope -> @+part: Part -> @+first: U32 -> @+rows: U32 -> Data,~Holds: @+scope: Scope -> @+index: Nat -> @+element: Element -> Data,
  ~rule: ListRule(~Part,~Element,~Context,~list,~width,~Scope,~context,~fill,~limit,~Fact,~Holds),
  +scope: Scope,tree: Model.Shaped<Part>,+first: Nat,+rows: Nat,coverage: PiecesPlan.Coverage(Model.shaped_plan(~Part,~piece_rows(~Part),tree),first,rows),
  facts: Model.ShapedAll(~Part,~Scope,~Fact,scope,tree),+room: {Nat.is_le(Nat.add(first,rows),limit(scope)) == True{} : Bool},
  +index: Nat,+inside: {Nat.is_lt(index,Nat.mul(rows,width(context(scope)))) == True{} : Bool}) ->
  ListAt(~Part,~Element,~Context,~list,~width,~Scope,~context,~fill,~Holds,scope,tree,first,index):
  match tree:
    case Model.Piece{+part,+native_first,+native_rows}:
      PiecesPlan.LeafCoverage{+first_equal,+rows_equal} = coverage
      %first_equal : Holds(scope,Nat.add(Nat.mul(_,width(context(scope))),index),ListRead.at(~Element,fill(scope),index,list(context(scope),part,native_first,native_rows)))
      rule(scope,part,native_first,native_rows,facts,
        %Equal.sym(Nat,U32.to_nat(native_first),first,first_equal) : {Nat.is_le(Nat.add(_,U32.to_nat(native_rows)),limit(scope)) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(native_rows),rows,rows_equal) : {Nat.is_le(Nat.add(first,_),limit(scope)) == True{} : Bool}
        room,
        index,
        %Equal.sym(Nat,U32.to_nat(native_rows),rows,rows_equal) : {Nat.is_lt(index,Nat.mul(_,width(context(scope)))) == True{} : Bool}
        inside)
    case Model.Pieces{+left,+right}:
      PiecesPlan.ForkCoverage{+left_rows,+right_rows,+rows_equal,+left_coverage,+right_coverage} = coverage
      KernelGemm.Both{+left_facts,+right_facts} = facts
      +width_value = width(context(scope))
      +left_room = Order.transitive(Nat.add(first,left_rows),Nat.add(first,rows),limit(scope),
        Order.offset_le(first,left_rows,rows,
          %Equal.sym(Nat,rows,Nat.add(left_rows,right_rows),rows_equal) : {Nat.is_le(left_rows,_) == True{} : Bool}
          Order.le_add(left_rows,right_rows)),
        room)
      +right_room = {
        %Equal.sym(Nat,Nat.add(Nat.add(first,left_rows),right_rows),Nat.add(first,Nat.add(left_rows,right_rows)),Addition.associative(first,left_rows,right_rows)) :
          {Nat.is_le(_,limit(scope)) == True{} : Bool}
        %rows_equal : {Nat.is_le(Nat.add(first,_),limit(scope)) == True{} : Bool}
        room : {Nat.is_le(Nat.add(Nat.add(first,left_rows),right_rows),limit(scope)) == True{} : Bool}}
      list_at_branch(~Part,~Element,~Context,~list,~width,~Scope,~context,~fill,~Holds,scope,left,right,first,left_rows,right_rows,index,
        Nat.is_lt(index,Nat.mul(left_rows,width_value)),
        shaped_list_length(~Part,~Element,~Context,~list,~width,~length_rule,context(scope),left,first,left_rows,left_coverage),{==},
        %RowMajor.distrib(left_rows,right_rows,width_value) : {Nat.is_lt(index,_) == True{} : Bool}
        %rows_equal : {Nat.is_lt(index,Nat.mul(_,width_value)) == True{} : Bool}
        inside,
        in_left => shaped_list_at(~Part,~Element,~Context,~list,~width,~length_rule,~Scope,~context,~fill,~limit,~Fact,~Holds,~rule,scope,left,first,left_rows,left_coverage,
          left_facts,left_room,index,in_left),
        tail => in_right => shaped_list_at(~Part,~Element,~Context,~list,~width,~length_rule,~Scope,~context,~fill,~limit,~Fact,~Holds,~rule,scope,right,Nat.add(first,left_rows),right_rows,
          right_coverage,right_facts,right_room,tail,in_right))

# ---- Pieces built along a plan ----
