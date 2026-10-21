# The shared band recursion, by structure. A parallel split and a leaf task
# compute what the sequential recursion computes; every band operation uses
# these equations instead of unfolding the recursion itself.
import Base
import ../tensor_band.bend as Bands
import ../tensor_band_schedule.bend as BandSchedule
import ../traversal_partition.bend as Partition
import ../storage_buffer.bend as Storage
import ./tensor_band_model.bend as Model
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./storage_tree_model.bend as Tree
import ./write_program.bend as Writes
import ./boolean_logic.bend as Boolean
import ./word_encoding.bend as WordEncoding
import ./tensor_band_walk.bend as Walk
import ./tensor_band_plan.bend as BandPlan
import ./list_storage_readback.bend as ListRead
import ./word_arithmetic.bend as WordArithmetic
import ./row_major_index.bend as RowMajor
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition

def Tree(~Block: Type) -> Type:
  Partition.Tree<&1,Block,Unit>

# Both halves of a split, each with its own resource.
def forked(~Block: Type,~Output: Type,~Resource: Type,~Context: Data,
  ~share: Resource -> Resource & Resource,
  ~leaf: Context -> U32 -> U32 -> Block -> Resource -> Tree(~Output),
  ~keep: Tree(~Block) -> Tree(~Output),
  +left_plan: Bands.Plan,+right_plan: Bands.Plan,+enabled: Bool,+context: Context,
  left: Tree(~Block),right: Tree(~Block),resources: Resource & Resource) -> Tree(~Output):
  Walk.sequential(~Bands.Banded(~Block,~Resource),~Output,~Unit,~Context,~Bool,~Bands.in_parallel,~Bands.band_divide(~Block,~Resource,~Context,~share),
    ~Bands.band_leaf(~Block,~Output,~Resource,~Context,~leaf,~keep),~Bands.band_kept(~Block,~Output,~Resource,~Context,~keep),
    left_plan,right_plan,enabled,context,Bands.shared_halves(~Block,~Resource,left,right,resources))

# A split plan on a fork: both halves, each on its own plan. The parallel
# split and its leaf tasks equal the sequential recursion.
def fork(~Block: Type,~Output: Type,~Resource: Type,~Context: Data,
  ~share: Resource -> Resource & Resource,
  ~leaf: Context -> U32 -> U32 -> Block -> Resource -> Tree(~Output),
  ~keep: Tree(~Block) -> Tree(~Output),
  +left_plan: Bands.Plan,+right_plan: Bands.Plan,+enabled: Bool,+context: Context,
  -left: Tree(~Block),-right: Tree(~Block),-resource: Resource) ->
  {Bands.each(~Block,~Output,~Resource,~Context,~share,~leaf,~keep,Bands.Split{left_plan,right_plan},enabled,context,Partition.Fork{left,right},resource) ==
    forked(~Block,~Output,~Resource,~Context,~share,~leaf,~keep,left_plan,right_plan,enabled,context,left,right,share(resource)) : Tree(~Output)}:
  Walk.fork(~Bands.Banded(~Block,~Resource),~Output,~Unit,~Context,~Bool,~Bands.in_parallel,~Bands.band_divide(~Block,~Resource,~Context,~share),
    ~Bands.band_leaf(~Block,~Output,~Resource,~Context,~leaf,~keep),~Bands.band_kept(~Block,~Output,~Resource,~Context,~keep),
    left_plan,right_plan,enabled,enabled,context,(Partition.Fork{left,right},resource))

# A split plan on one leaf keeps the leaf.
def split_leaf(~Block: Type,~Output: Type,~Resource: Type,~Context: Data,
  ~share: Resource -> Resource & Resource,
  ~leaf: Context -> U32 -> U32 -> Block -> Resource -> Tree(~Output),
  ~keep: Tree(~Block) -> Tree(~Output),
  +left_plan: Bands.Plan,+right_plan: Bands.Plan,+enabled: Bool,+context: Context,
  -block: Block,-first: U32,-rows: U32,-details: Unit,-resource: Resource) ->
  {Bands.each(~Block,~Output,~Resource,~Context,~share,~leaf,~keep,Bands.Split{left_plan,right_plan},enabled,context,Partition.Leaf{block,first,rows,details},resource) ==
    keep(Partition.Leaf{block,first,rows,details}) : Tree(~Output)}:
  Walk.fork(~Bands.Banded(~Block,~Resource),~Output,~Unit,~Context,~Bool,~Bands.in_parallel,~Bands.band_divide(~Block,~Resource,~Context,~share),
    ~Bands.band_leaf(~Block,~Output,~Resource,~Context,~leaf,~keep),~Bands.band_kept(~Block,~Output,~Resource,~Context,~keep),
    left_plan,right_plan,enabled,enabled,context,(Partition.Leaf{block,first,rows,details},resource))

# An unpaired tree comes back unchanged.
def only_left(~Left: Type,~Right: Type,tree: Tree(~Left)) ->
  {Bands.left_of(~Left,~Right,Bands.only(~Left,~Right,tree)) == tree : Tree(~Left)}:
  match tree:
    case Partition.Leaf{block,first,rows,Unit{}}: {==}
    case Partition.Fork{left,right}:
      %only_left(~Left,~Right,left) :
        {Partition.Fork{Bands.left_of(~Left,~Right,Bands.only(~Left,~Right,left)),Bands.left_of(~Left,~Right,Bands.only(~Left,~Right,right))} == Partition.Fork{_,right} : Tree(~Left)}
      %only_left(~Left,~Right,right) :
        {Partition.Fork{Bands.left_of(~Left,~Right,Bands.only(~Left,~Right,left)),Bands.left_of(~Left,~Right,Bands.only(~Left,~Right,right))} ==
          Partition.Fork{Bands.left_of(~Left,~Right,Bands.only(~Left,~Right,left)),_} : Tree(~Left)}
      {==}

# The left tree of a pairing comes back unchanged, whatever the right tree.
def zip_left(~Left: Type,~Right: Type,left: Tree(~Left),right: Tree(~Right)) ->
  {Bands.left_of(~Left,~Right,Bands.zip(~Left,~Right,left,right)) == left : Tree(~Left)}:
  match left right:
    case Partition.Leaf{block,first,rows,Unit{}} Partition.Leaf{other,other_first,other_rows,other_details}: {==}
    case Partition.Leaf{block,first,rows,Unit{}} Partition.Fork{other_left,other_right}: {==}
    case Partition.Fork{left,right} Partition.Leaf{other,other_first,other_rows,other_details}:
      only_left(~Left,~Right,Partition.Fork{left,right})
    case Partition.Fork{left,right} Partition.Fork{other_left,other_right}:
      %zip_left(~Left,~Right,left,other_left) :
        {Partition.Fork{Bands.left_of(~Left,~Right,Bands.zip(~Left,~Right,left,other_left)),Bands.left_of(~Left,~Right,Bands.zip(~Left,~Right,right,other_right))} == Partition.Fork{_,right} : Tree(~Left)}
      %zip_left(~Left,~Right,right,other_right) :
        {Partition.Fork{Bands.left_of(~Left,~Right,Bands.zip(~Left,~Right,left,other_left)),Bands.left_of(~Left,~Right,Bands.zip(~Left,~Right,right,other_right))} ==
          Partition.Fork{Bands.left_of(~Left,~Right,Bands.zip(~Left,~Right,left,other_left)),_} : Tree(~Left)}
      {==}

def KeptRule(~Left: Type,~Right: Type,~Resource: Type,~Context: Data,
  ~share: Resource -> Resource & Resource,
  ~leaf: Context -> U32 -> U32 -> Left -> Right -> Resource -> Tree(~Left),
  +layout: Bands.Plan,+enabled: Bool,+context: Context,-tree: Tree(~Left)) -> Type:
  @resource: Resource ->
  {Bands.each(~Bands.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Bands.paired(~Left,~Right,~Resource,~Context,~leaf),~Bands.left_of(~Left,~Right),
      layout,enabled,context,Bands.only(~Left,~Right,tree),resource) == tree : Tree(~Left)}

def kept_fork(~Left: Type,~Right: Type,~Resource: Type,~Context: Data,
  ~share: Resource -> Resource & Resource,
  ~leaf: Context -> U32 -> U32 -> Left -> Right -> Resource -> Tree(~Left),
  +left_plan: Bands.Plan,+right_plan: Bands.Plan,+enabled: Bool,+context: Context,
  -left: Tree(~Left),-right: Tree(~Left),resources: Resource & Resource,
  left_rule: KeptRule(~Left,~Right,~Resource,~Context,~share,~leaf,left_plan,enabled,context,left),
  right_rule: KeptRule(~Left,~Right,~Resource,~Context,~share,~leaf,right_plan,enabled,context,right)) ->
  {forked(~Bands.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Bands.paired(~Left,~Right,~Resource,~Context,~leaf),~Bands.left_of(~Left,~Right),
      left_plan,right_plan,enabled,context,Bands.only(~Left,~Right,left),Bands.only(~Left,~Right,right),resources) == Partition.Fork{left,right} : Tree(~Left)}:
  (left_resource,right_resource) = resources
  %left_rule(left_resource) :
    {Partition.Fork{Bands.each(~Bands.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Bands.paired(~Left,~Right,~Resource,~Context,~leaf),~Bands.left_of(~Left,~Right),
          left_plan,enabled,context,Bands.only(~Left,~Right,left),left_resource),
        Bands.each(~Bands.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Bands.paired(~Left,~Right,~Resource,~Context,~leaf),~Bands.left_of(~Left,~Right),
          right_plan,enabled,context,Bands.only(~Left,~Right,right),right_resource)} ==
      Partition.Fork{_,right} : Tree(~Left)}
  %right_rule(right_resource) :
    {Partition.Fork{Bands.each(~Bands.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Bands.paired(~Left,~Right,~Resource,~Context,~leaf),~Bands.left_of(~Left,~Right),
          left_plan,enabled,context,Bands.only(~Left,~Right,left),left_resource),
        Bands.each(~Bands.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Bands.paired(~Left,~Right,~Resource,~Context,~leaf),~Bands.left_of(~Left,~Right),
          right_plan,enabled,context,Bands.only(~Left,~Right,right),right_resource)} ==
      Partition.Fork{Bands.each(~Bands.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Bands.paired(~Left,~Right,~Resource,~Context,~leaf),~Bands.left_of(~Left,~Right),
          left_plan,enabled,context,Bands.only(~Left,~Right,left),left_resource),_} : Tree(~Left)}
  {==}

# A paired operation on an unpaired tree returns the tree, on every plan.
def only_kept(~Left: Type,~Right: Type,~Resource: Type,~Context: Data,
  ~share: Resource -> Resource & Resource,
  ~leaf: Context -> U32 -> U32 -> Left -> Right -> Resource -> Tree(~Left),
  +layout: Bands.Plan,+enabled: Bool,+context: Context,tree: Tree(~Left),resource: Resource) ->
  {Bands.each(~Bands.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Bands.paired(~Left,~Right,~Resource,~Context,~leaf),~Bands.left_of(~Left,~Right),
      layout,enabled,context,Bands.only(~Left,~Right,tree),resource) == tree : Tree(~Left)}:
  match layout tree:
    case Bands.Rows{first,rows} Partition.Leaf{block,own_first,own_rows,Unit{}}: {==}
    case Bands.Rows{first,rows} Partition.Fork{left,right}: only_left(~Left,~Right,Partition.Fork{left,right})
    case Bands.Split{+left_plan,+right_plan} Partition.Leaf{block,own_first,own_rows,Unit{}}:
      split_leaf(~Bands.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Bands.paired(~Left,~Right,~Resource,~Context,~leaf),~Bands.left_of(~Left,~Right),
        left_plan,right_plan,enabled,context,Bands.Only{block,own_first,own_rows},own_first,own_rows,Unit{},resource)
    case Bands.Split{+left_plan,+right_plan} Partition.Fork{left,right}:
      %Equal.sym(Tree(~Left),
        Bands.each(~Bands.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Bands.paired(~Left,~Right,~Resource,~Context,~leaf),~Bands.left_of(~Left,~Right),
          Bands.Split{left_plan,right_plan},enabled,context,Partition.Fork{Bands.only(~Left,~Right,left),Bands.only(~Left,~Right,right)},resource),
        forked(~Bands.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Bands.paired(~Left,~Right,~Resource,~Context,~leaf),~Bands.left_of(~Left,~Right),
          left_plan,right_plan,enabled,context,Bands.only(~Left,~Right,left),Bands.only(~Left,~Right,right),share(resource)),
        fork(~Bands.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Bands.paired(~Left,~Right,~Resource,~Context,~leaf),~Bands.left_of(~Left,~Right),
          left_plan,right_plan,enabled,context,Bands.only(~Left,~Right,left),Bands.only(~Left,~Right,right),resource)) :
        {_ == Partition.Fork{left,right} : Tree(~Left)}
      kept_fork(~Left,~Right,~Resource,~Context,~share,~leaf,left_plan,right_plan,enabled,context,left,right,share(resource),
        left_resource => only_kept(~Left,~Right,~Resource,~Context,~share,~leaf,left_plan,enabled,context,left,left_resource),
        right_resource => only_kept(~Left,~Right,~Resource,~Context,~share,~leaf,right_plan,enabled,context,right,right_resource))

# The contents of an operation on every leaf of one tree.
def content_map(~Element: Data,~Context: Data,~leaf_model: Context -> U32 -> U32 -> Buffer.Model<Element> -> Model.Content<Element>,
  +context: Context,bands: Model.Model<Element>) -> Model.Content<Element>:
  match bands:
    case Model.Leaf{input,first,rows}: leaf_model(context,first,rows,input)
    case Model.Fork{left,right}: Model.Joined{content_map(~Element,~Context,~leaf_model,context,left),content_map(~Element,~Context,~leaf_model,context,right)}

# The scope of an operation whose rule needs only the context of its leaf.
def own(~Context: Data,+context: Context) -> Context:
  context

# Facts about every band of the input give a fact about every band of the
# mapped contents. An operation proves its rule for one band and passes what
# the rule needs as one scope value, from which the context of its leaf is read.
def map_one(~Element: Data,~Context: Data,~leaf_model: Context -> U32 -> U32 -> Buffer.Model<Element> -> Model.Content<Element>,~Scope: Data,~context: @+scope: Scope -> Context,~First: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,~After: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~leaf: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Model<Element> -> @+first_fact: First(scope,first,rows,Buffer.content(~Element,buffer)) -> Model.ContentAll(~Element,~After(scope),leaf_model(context(scope),first,rows,buffer)),
  +scope: Scope,model: Model.Model<Element>,first_fact: Model.ContentAll(~Element,~First(scope),Model.content(~Element,model))) ->
  Model.ContentAll(~Element,~After(scope),content_map(~Element,~Context,~leaf_model,context(scope),model)):
  match model:
    case Model.Leaf{+buffer,+first,+rows}: leaf(scope,first,rows,buffer,first_fact)
    case Model.Fork{+left,+right}:
      Shape.Both{first_fact_left,first_fact_right} = first_fact
      Shape.Both{map_one(~Element,~Context,~leaf_model,~Scope,~context,~First,~After,~leaf,scope,left,first_fact_left),map_one(~Element,~Context,~leaf_model,~Scope,~context,~First,~After,~leaf,scope,right,first_fact_right)}

def map_two(~Element: Data,~Context: Data,~leaf_model: Context -> U32 -> U32 -> Buffer.Model<Element> -> Model.Content<Element>,~Scope: Data,~context: @+scope: Scope -> Context,~First: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,~Second: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,~After: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~leaf: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Model<Element> -> @+first_fact: First(scope,first,rows,Buffer.content(~Element,buffer)) -> @+second_fact: Second(scope,first,rows,Buffer.content(~Element,buffer)) -> Model.ContentAll(~Element,~After(scope),leaf_model(context(scope),first,rows,buffer)),
  +scope: Scope,model: Model.Model<Element>,first_fact: Model.ContentAll(~Element,~First(scope),Model.content(~Element,model)),second_fact: Model.ContentAll(~Element,~Second(scope),Model.content(~Element,model))) ->
  Model.ContentAll(~Element,~After(scope),content_map(~Element,~Context,~leaf_model,context(scope),model)):
  match model:
    case Model.Leaf{+buffer,+first,+rows}: leaf(scope,first,rows,buffer,first_fact,second_fact)
    case Model.Fork{+left,+right}:
      Shape.Both{first_fact_left,first_fact_right} = first_fact
      Shape.Both{second_fact_left,second_fact_right} = second_fact
      Shape.Both{map_two(~Element,~Context,~leaf_model,~Scope,~context,~First,~Second,~After,~leaf,scope,left,first_fact_left,second_fact_left),map_two(~Element,~Context,~leaf_model,~Scope,~context,~First,~Second,~After,~leaf,scope,right,first_fact_right,second_fact_right)}

def map_ready_zero(~Element: Data,~Context: Data,~leaf_model: Context -> U32 -> U32 -> Buffer.Model<Element> -> Model.Content<Element>,~Scope: Data,~context: @+scope: Scope -> Context,~After: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~leaf: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Model<Element> -> @+ready: Buffer.Sized<Element,buffer> -> Model.ContentAll(~Element,~After(scope),leaf_model(context(scope),first,rows,buffer)),
  +scope: Scope,model: Model.Model<Element>,ready: Model.Ready(~Element,model)) ->
  Model.ContentAll(~Element,~After(scope),content_map(~Element,~Context,~leaf_model,context(scope),model)):
  match model:
    case Model.Leaf{+buffer,+first,+rows}: leaf(scope,first,rows,buffer,ready)
    case Model.Fork{+left,+right}:
      Shape.Both{ready_left,ready_right} = ready
      Shape.Both{map_ready_zero(~Element,~Context,~leaf_model,~Scope,~context,~After,~leaf,scope,left,ready_left),map_ready_zero(~Element,~Context,~leaf_model,~Scope,~context,~After,~leaf,scope,right,ready_right)}

def map_ready_one(~Element: Data,~Context: Data,~leaf_model: Context -> U32 -> U32 -> Buffer.Model<Element> -> Model.Content<Element>,~Scope: Data,~context: @+scope: Scope -> Context,~First: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,~After: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~leaf: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Model<Element> -> @+ready: Buffer.Sized<Element,buffer> -> @+first_fact: First(scope,first,rows,Buffer.content(~Element,buffer)) -> Model.ContentAll(~Element,~After(scope),leaf_model(context(scope),first,rows,buffer)),
  +scope: Scope,model: Model.Model<Element>,ready: Model.Ready(~Element,model),first_fact: Model.ContentAll(~Element,~First(scope),Model.content(~Element,model))) ->
  Model.ContentAll(~Element,~After(scope),content_map(~Element,~Context,~leaf_model,context(scope),model)):
  match model:
    case Model.Leaf{+buffer,+first,+rows}: leaf(scope,first,rows,buffer,ready,first_fact)
    case Model.Fork{+left,+right}:
      Shape.Both{ready_left,ready_right} = ready
      Shape.Both{first_fact_left,first_fact_right} = first_fact
      Shape.Both{map_ready_one(~Element,~Context,~leaf_model,~Scope,~context,~First,~After,~leaf,scope,left,ready_left,first_fact_left),map_ready_one(~Element,~Context,~leaf_model,~Scope,~context,~First,~After,~leaf,scope,right,ready_right,first_fact_right)}

def map_ready_two(~Element: Data,~Context: Data,~leaf_model: Context -> U32 -> U32 -> Buffer.Model<Element> -> Model.Content<Element>,~Scope: Data,~context: @+scope: Scope -> Context,~First: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,~Second: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,~After: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~leaf: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Model<Element> -> @+ready: Buffer.Sized<Element,buffer> -> @+first_fact: First(scope,first,rows,Buffer.content(~Element,buffer)) -> @+second_fact: Second(scope,first,rows,Buffer.content(~Element,buffer)) -> Model.ContentAll(~Element,~After(scope),leaf_model(context(scope),first,rows,buffer)),
  +scope: Scope,model: Model.Model<Element>,ready: Model.Ready(~Element,model),first_fact: Model.ContentAll(~Element,~First(scope),Model.content(~Element,model)),second_fact: Model.ContentAll(~Element,~Second(scope),Model.content(~Element,model))) ->
  Model.ContentAll(~Element,~After(scope),content_map(~Element,~Context,~leaf_model,context(scope),model)):
  match model:
    case Model.Leaf{+buffer,+first,+rows}: leaf(scope,first,rows,buffer,ready,first_fact,second_fact)
    case Model.Fork{+left,+right}:
      Shape.Both{ready_left,ready_right} = ready
      Shape.Both{first_fact_left,first_fact_right} = first_fact
      Shape.Both{second_fact_left,second_fact_right} = second_fact
      Shape.Both{map_ready_two(~Element,~Context,~leaf_model,~Scope,~context,~First,~Second,~After,~leaf,scope,left,ready_left,first_fact_left,second_fact_left),map_ready_two(~Element,~Context,~leaf_model,~Scope,~context,~First,~Second,~After,~leaf,scope,right,ready_right,first_fact_right,second_fact_right)}

# The plan of the mapped contents, when every band gives a plan of its own.
def map_plan_one(~Element: Data,~Context: Data,~leaf_model: Context -> U32 -> U32 -> Buffer.Model<Element> -> Model.Content<Element>,~Scope: Data,~context: @+scope: Scope -> Context,~First: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,~Rows: Data,~band: @+context: Rows -> @+first: U32 -> @+rows: U32 -> Bands.Plan,~rows_context: @+scope: Scope -> Rows,
  ~leaf: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Model<Element> -> @+first_fact: First(scope,first,rows,Buffer.content(~Element,buffer)) -> {Model.content_layout(~Element,leaf_model(context(scope),first,rows,buffer)) == band(rows_context(scope),first,rows) : Bands.Plan},
  +scope: Scope,model: Model.Model<Element>,first_fact: Model.ContentAll(~Element,~First(scope),Model.content(~Element,model))) ->
  {Model.content_layout(~Element,content_map(~Element,~Context,~leaf_model,context(scope),model)) == Model.plan_map(~Rows,~band,rows_context(scope),Model.layout(~Element,model)) : Bands.Plan}:
  match model:
    case Model.Leaf{+buffer,+first,+rows}: leaf(scope,first,rows,buffer,first_fact)
    case Model.Fork{+left,+right}:
      Shape.Both{first_fact_left,first_fact_right} = first_fact
      %Equal.sym(Bands.Plan,Model.content_layout(~Element,content_map(~Element,~Context,~leaf_model,context(scope),left)),Model.plan_map(~Rows,~band,rows_context(scope),Model.layout(~Element,left)),map_plan_one(~Element,~Context,~leaf_model,~Scope,~context,~First,~Rows,~band,~rows_context,~leaf,scope,left,first_fact_left)) : {Bands.Split{_,Model.content_layout(~Element,content_map(~Element,~Context,~leaf_model,context(scope),right))} == Bands.Split{Model.plan_map(~Rows,~band,rows_context(scope),Model.layout(~Element,left)),Model.plan_map(~Rows,~band,rows_context(scope),Model.layout(~Element,right))} : Bands.Plan}
      %Equal.sym(Bands.Plan,Model.content_layout(~Element,content_map(~Element,~Context,~leaf_model,context(scope),right)),Model.plan_map(~Rows,~band,rows_context(scope),Model.layout(~Element,right)),map_plan_one(~Element,~Context,~leaf_model,~Scope,~context,~First,~Rows,~band,~rows_context,~leaf,scope,right,first_fact_right)) : {Bands.Split{Model.plan_map(~Rows,~band,rows_context(scope),Model.layout(~Element,left)),_} == Bands.Split{Model.plan_map(~Rows,~band,rows_context(scope),Model.layout(~Element,left)),Model.plan_map(~Rows,~band,rows_context(scope),Model.layout(~Element,right))} : Bands.Plan}
      {==}

# A leaf that keeps its first row and its row count keeps the layout of the tree.
def mapped_layout(~Element: Data,~Context: Data,~leaf_model: Context -> U32 -> U32 -> Buffer.Model<Element> -> Model.Content<Element>,
  ~leaf: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+input: Buffer.Model<Element> -> {Model.content_layout(~Element,leaf_model(context,first,rows,input)) == Bands.Rows{first,rows} : Bands.Plan},
  +context: Context,bands: Model.Model<Element>) ->
  {Model.content_layout(~Element,content_map(~Element,~Context,~leaf_model,context,bands)) == Model.layout(~Element,bands) : Bands.Plan}:
  match bands:
    case Model.Leaf{+input,+first,+rows}: leaf(context,first,rows,input)
    case Model.Fork{+left,+right}:
      %mapped_layout(~Element,~Context,~leaf_model,~leaf,context,left) :
        {Bands.Split{Model.content_layout(~Element,content_map(~Element,~Context,~leaf_model,context,left)),Model.content_layout(~Element,content_map(~Element,~Context,~leaf_model,context,right))} == Bands.Split{_,Model.layout(~Element,right)} : Bands.Plan}
      %mapped_layout(~Element,~Context,~leaf_model,~leaf,context,right) :
        {Bands.Split{Model.content_layout(~Element,content_map(~Element,~Context,~leaf_model,context,left)),Model.content_layout(~Element,content_map(~Element,~Context,~leaf_model,context,right))} == Bands.Split{Model.content_layout(~Element,content_map(~Element,~Context,~leaf_model,context,left)),_} : Bands.Plan}
      {==}

def MapRule(~Element: Data,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Unit -> Bands.Bands(~Element),
  ~leaf_model: Context -> U32 -> U32 -> Buffer.Model<Element> -> Model.Content<Element>) -> Type:
  @+context: Context -> @+first: U32 -> @+rows: U32 -> @+input: Buffer.Model<Element> ->
  {Model.contents(~Element,leaf(context,first,rows,Buffer.pack(~Element,input),Unit{})) == leaf_model(context,first,rows,input) : Model.Content<Element>}

# A one-tree operation on its own plan: the tree induction, once. An
# operation supplies only the equation of its leaf.
def mapped(~Element: Data,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Unit -> Bands.Bands(~Element),
  ~leaf_model: Context -> U32 -> U32 -> Buffer.Model<Element> -> Model.Content<Element>,
  ~leaf_rule: MapRule(~Element,~Context,~leaf,~leaf_model),
  bands: Model.Model<Element>,+enabled: Bool,+context: Context) ->
  {Model.contents(~Element,Bands.each(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~Bands.unshared,~leaf,~Bands.same(~Storage.Buffer<Element>),
      Model.layout(~Element,bands),enabled,context,Model.pack(~Element,bands),Unit{})) ==
    content_map(~Element,~Context,~leaf_model,context,bands) : Model.Content<Element>}:
  match bands:
    case Model.Leaf{+input,+first,+rows}: leaf_rule(context,first,rows,input)
    case Model.Fork{+left,+right}:
      %Equal.sym(Bands.Bands(~Element),
        Bands.each(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~Bands.unshared,~leaf,~Bands.same(~Storage.Buffer<Element>),
          Bands.Split{Model.layout(~Element,left),Model.layout(~Element,right)},enabled,context,Partition.Fork{Model.pack(~Element,left),Model.pack(~Element,right)},Unit{}),
        Partition.Fork{Bands.each(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~Bands.unshared,~leaf,~Bands.same(~Storage.Buffer<Element>),
            Model.layout(~Element,left),enabled,context,Model.pack(~Element,left),Unit{}),
          Bands.each(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~Bands.unshared,~leaf,~Bands.same(~Storage.Buffer<Element>),
            Model.layout(~Element,right),enabled,context,Model.pack(~Element,right),Unit{})},
        fork(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~Bands.unshared,~leaf,~Bands.same(~Storage.Buffer<Element>),
          Model.layout(~Element,left),Model.layout(~Element,right),enabled,context,Model.pack(~Element,left),Model.pack(~Element,right),Unit{})) :
        {Model.contents(~Element,_) == Model.Joined{content_map(~Element,~Context,~leaf_model,context,left),content_map(~Element,~Context,~leaf_model,context,right)} : Model.Content<Element>}
      %Equal.sym(Model.Content<Element>,
        Model.contents(~Element,Bands.each(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~Bands.unshared,~leaf,~Bands.same(~Storage.Buffer<Element>),
          Model.layout(~Element,left),enabled,context,Model.pack(~Element,left),Unit{})),
        content_map(~Element,~Context,~leaf_model,context,left),
        mapped(~Element,~Context,~leaf,~leaf_model,~leaf_rule,left,enabled,context)) :
        {Model.Joined{_,Model.contents(~Element,Bands.each(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~Bands.unshared,~leaf,~Bands.same(~Storage.Buffer<Element>),
            Model.layout(~Element,right),enabled,context,Model.pack(~Element,right),Unit{}))} ==
          Model.Joined{content_map(~Element,~Context,~leaf_model,context,left),content_map(~Element,~Context,~leaf_model,context,right)} : Model.Content<Element>}
      %Equal.sym(Model.Content<Element>,
        Model.contents(~Element,Bands.each(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~Bands.unshared,~leaf,~Bands.same(~Storage.Buffer<Element>),
          Model.layout(~Element,right),enabled,context,Model.pack(~Element,right),Unit{})),
        content_map(~Element,~Context,~leaf_model,context,right),
        mapped(~Element,~Context,~leaf,~leaf_model,~leaf_rule,right,enabled,context)) :
        {Model.Joined{content_map(~Element,~Context,~leaf_model,context,left),_} ==
          Model.Joined{content_map(~Element,~Context,~leaf_model,context,left),content_map(~Element,~Context,~leaf_model,context,right)} : Model.Content<Element>}
      {==}

# Every leaf of a model through the leaf of an operation, in the shape of the
# model.
def leaf_map(~Element: Data,~Output: Type,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Unit -> Tree(~Output),
  +context: Context,bands: Model.Model<Element>) -> Tree(~Output):
  match bands:
    case Model.Leaf{input,first,rows}: leaf(context,first,rows,Buffer.pack(~Element,input),Unit{})
    case Model.Fork{left,right}: Partition.Fork{leaf_map(~Element,~Output,~Context,~leaf,context,left),leaf_map(~Element,~Output,~Context,~leaf,context,right)}

# On the plan of its own tree the recursion reaches every leaf, whatever it
# keeps elsewhere and whichever halves run in parallel.
def planned(~Element: Data,~Output: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Unit -> Tree(~Output),~keep: Tree(~Storage.Buffer<Element>) -> Tree(~Output),
  bands: Model.Model<Element>,+enabled: Bool,+context: Context) ->
  {Bands.each(~Storage.Buffer<Element>,~Output,~Unit,~Context,~Bands.unshared,~leaf,~keep,Model.layout(~Element,bands),enabled,context,Model.pack(~Element,bands),Unit{}) ==
    leaf_map(~Element,~Output,~Context,~leaf,context,bands) : Tree(~Output)}:
  match bands:
    case Model.Leaf{+input,+first,+rows}: {==}
    case Model.Fork{+left,+right}:
      %Equal.sym(Tree(~Output),
        Bands.each(~Storage.Buffer<Element>,~Output,~Unit,~Context,~Bands.unshared,~leaf,~keep,
          Bands.Split{Model.layout(~Element,left),Model.layout(~Element,right)},enabled,context,Partition.Fork{Model.pack(~Element,left),Model.pack(~Element,right)},Unit{}),
        Partition.Fork{Bands.each(~Storage.Buffer<Element>,~Output,~Unit,~Context,~Bands.unshared,~leaf,~keep,
            Model.layout(~Element,left),enabled,context,Model.pack(~Element,left),Unit{}),
          Bands.each(~Storage.Buffer<Element>,~Output,~Unit,~Context,~Bands.unshared,~leaf,~keep,
            Model.layout(~Element,right),enabled,context,Model.pack(~Element,right),Unit{})},
        fork(~Storage.Buffer<Element>,~Output,~Unit,~Context,~Bands.unshared,~leaf,~keep,
          Model.layout(~Element,left),Model.layout(~Element,right),enabled,context,Model.pack(~Element,left),Model.pack(~Element,right),Unit{})) :
        {_ == Partition.Fork{leaf_map(~Element,~Output,~Context,~leaf,context,left),leaf_map(~Element,~Output,~Context,~leaf,context,right)} : Tree(~Output)}
      %Equal.sym(Tree(~Output),
        Bands.each(~Storage.Buffer<Element>,~Output,~Unit,~Context,~Bands.unshared,~leaf,~keep,
          Model.layout(~Element,left),enabled,context,Model.pack(~Element,left),Unit{}),
        leaf_map(~Element,~Output,~Context,~leaf,context,left),
        planned(~Element,~Output,~Context,~leaf,~keep,left,enabled,context)) :
        {Partition.Fork{_,Bands.each(~Storage.Buffer<Element>,~Output,~Unit,~Context,~Bands.unshared,~leaf,~keep,
            Model.layout(~Element,right),enabled,context,Model.pack(~Element,right),Unit{})} ==
          Partition.Fork{leaf_map(~Element,~Output,~Context,~leaf,context,left),leaf_map(~Element,~Output,~Context,~leaf,context,right)} : Tree(~Output)}
      %Equal.sym(Tree(~Output),
        Bands.each(~Storage.Buffer<Element>,~Output,~Unit,~Context,~Bands.unshared,~leaf,~keep,
          Model.layout(~Element,right),enabled,context,Model.pack(~Element,right),Unit{}),
        leaf_map(~Element,~Output,~Context,~leaf,context,right),
        planned(~Element,~Output,~Context,~leaf,~keep,right,enabled,context)) :
        {Partition.Fork{leaf_map(~Element,~Output,~Context,~leaf,context,left),_} ==
          Partition.Fork{leaf_map(~Element,~Output,~Context,~leaf,context,left),leaf_map(~Element,~Output,~Context,~leaf,context,right)} : Tree(~Output)}
      {==}

# What the checked transform gives back: its bands and whether every call was
# usable.
def made(~Element: Data,result: Bands.Bands(~Element) & Bool) -> Bands.Bands(~Element):
  (bands,usable) = result
  bands

def usable(~Element: Data,result: Bands.Bands(~Element) & Bool) -> Bool:
  (bands,usable) = result
  usable

def checked(~Element: Data,tree: Bands.Transformed(~Element)) -> Bands.Bands(~Element) & Bool:
  Bands.reduce(~Bands.Checked<Element>,~Storage.Buffer<Element>,~Bool,~Unit,~Bands.checked_band(~Element),~Bool.and,Unit{},tree)

# The checked transform on the plan of its own bands.
def transformed(~Element: Data,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Bands.Transformed(~Element),
  +parallel: Bool,+context: Context,+bands: Model.Model<Element>) ->
  {Bands.transform(~Element,~Context,~leaf,Model.layout(~Element,bands),parallel,context,Model.pack(~Element,bands)) ==
    checked(~Element,leaf_map(~Element,~Bands.Checked<Element>,~Context,~Bands.transform_band(~Element,~Context,~leaf),context,bands)) : Bands.Bands(~Element) & Bool}:
  %Equal.sym(Bands.Transformed(~Element),
    Bands.each(~Storage.Buffer<Element>,~Bands.Checked<Element>,~Unit,~Context,~Bands.unshared,~Bands.transform_band(~Element,~Context,~leaf),~Bands.untransformed(~Element),
      Model.layout(~Element,bands),parallel,context,Model.pack(~Element,bands),Unit{}),
    leaf_map(~Element,~Bands.Checked<Element>,~Context,~Bands.transform_band(~Element,~Context,~leaf),context,bands),
    planned(~Element,~Bands.Checked<Element>,~Context,~Bands.transform_band(~Element,~Context,~leaf),~Bands.untransformed(~Element),bands,
      parallel,context)) :
    {checked(~Element,_) == checked(~Element,leaf_map(~Element,~Bands.Checked<Element>,~Context,~Bands.transform_band(~Element,~Context,~leaf),context,bands)) : Bands.Bands(~Element) & Bool}
  {==}

# The scheduled checked transform plans its own bands.
def scheduled(~Element: Data,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Bands.Transformed(~Element),
  +context: Context,+geometry: Bands.Geometry,+workers: U32,+bytes_per_element: F32,+operations: F32,+bands: Model.Model<Element>) ->
  {BandSchedule.transform(~Element,~Context,~leaf,context,geometry,workers,bytes_per_element,operations,Model.pack(~Element,bands)) ==
    checked(~Element,leaf_map(~Element,~Bands.Checked<Element>,~Context,~Bands.transform_band(~Element,~Context,~leaf),context,bands)) : Bands.Bands(~Element) & Bool}:
  %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,bands)),(Model.pack(~Element,bands),Model.layout(~Element,bands)),Model.plan(~Element,bands)) :
    {BandSchedule.transformed(~Element,~Context,~leaf,context,geometry,workers,bytes_per_element,operations,_) ==
      checked(~Element,leaf_map(~Element,~Bands.Checked<Element>,~Context,~Bands.transform_band(~Element,~Context,~leaf),context,bands)) : Bands.Bands(~Element) & Bool}
  transformed(~Element,~Context,~leaf,BandSchedule.parallel(geometry,workers,bytes_per_element,operations,Model.layout(~Element,bands)),context,bands)

# Two halves joined: when both were usable, what holds of each half holds of
# the joined bands.
def checked_joined(~Element: Data,~Scope: Data,~After: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  +scope: Scope,left: Bands.Bands(~Element) & Bool,right: Bands.Bands(~Element) & Bool,
  left_fact: {usable(~Element,left) == True{} : Bool} -> Model.ContentAll(~Element,~After(scope),Model.contents(~Element,made(~Element,left))),
  right_fact: {usable(~Element,right) == True{} : Bool} -> Model.ContentAll(~Element,~After(scope),Model.contents(~Element,made(~Element,right))),
  +accepted: {usable(~Element,Bands.reduced(~Storage.Buffer<Element>,~Bool,~Bool.and,left,right)) == True{} : Bool}) ->
  Model.ContentAll(~Element,~After(scope),Model.contents(~Element,made(~Element,Bands.reduced(~Storage.Buffer<Element>,~Bool,~Bool.and,left,right)))):
  (left_bands,+left_usable) = left
  (right_bands,+right_usable) = right
  Shape.Both{left_fact(Boolean.left(left_usable,right_usable,accepted)),right_fact(Boolean.right(left_usable,right_usable,accepted))}

# Every band of a checked transform, from the rule for one call. A call that
# was not usable makes the whole result unusable, so the rule needs to hold
# only of usable calls. What the rule needs travels as one scope value, from
# which the context of the call is read.
def checked_all(~Element: Data,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Unit -> Bands.Transformed(~Element),~Scope: Data,~context: @+scope: Scope -> Context,
  ~First: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,~After: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~rule: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+input: Buffer.Model<Element> -> @+first_fact: First(scope,first,rows,Buffer.content(~Element,input)) ->
    @accepted: {usable(~Element,checked(~Element,leaf(context(scope),first,rows,Buffer.pack(~Element,input),Unit{}))) == True{} : Bool} ->
    Model.ContentAll(~Element,~After(scope),Model.contents(~Element,made(~Element,checked(~Element,leaf(context(scope),first,rows,Buffer.pack(~Element,input),Unit{}))))),
  +scope: Scope,bands: Model.Model<Element>,first_fact: Model.ContentAll(~Element,~First(scope),Model.content(~Element,bands)),
  accepted: {usable(~Element,checked(~Element,leaf_map(~Element,~Bands.Checked<Element>,~Context,~leaf,context(scope),bands))) == True{} : Bool}) ->
  Model.ContentAll(~Element,~After(scope),Model.contents(~Element,made(~Element,checked(~Element,leaf_map(~Element,~Bands.Checked<Element>,~Context,~leaf,context(scope),bands))))):
  match bands:
    case Model.Leaf{+input,+first,+rows}: rule(scope,first,rows,input,first_fact,accepted)
    case Model.Fork{+left,+right}:
      Shape.Both{left_fact,right_fact} = first_fact
      checked_joined(~Element,~Scope,~After,scope,checked(~Element,leaf_map(~Element,~Bands.Checked<Element>,~Context,~leaf,context(scope),left)),
        checked(~Element,leaf_map(~Element,~Bands.Checked<Element>,~Context,~leaf,context(scope),right)),
        accepted => checked_all(~Element,~Context,~leaf,~Scope,~context,~First,~After,~rule,scope,left,left_fact,accepted),
        accepted => checked_all(~Element,~Context,~leaf,~Scope,~context,~First,~After,~rule,scope,right,right_fact,accepted),accepted)

# The rows of a piece, as the plan of its band.
def piece_rows(~Part: Data,part: Part,+first: U32,+rows: U32) -> Bands.Plan:
  Bands.Rows{first,rows}

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
  {Bands.each(~Block,~Output,~Unit,~Context,~Bands.unshared,~leaf,~keep,Model.shaped_plan(~Part,~piece_rows(~Part),tree),enabled,context,
      Model.shaped_pack(~Part,~Block,~pack,tree),Unit{}) ==
    shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,tree) : Tree(~Output)}:
  match tree:
    case Model.Piece{+part,+first,+rows}: {==}
    case Model.Pieces{+left,+right}:
      +left_plan = Model.shaped_plan(~Part,~piece_rows(~Part),left)
      +right_plan = Model.shaped_plan(~Part,~piece_rows(~Part),right)
      %Equal.sym(Tree(~Output),
        Bands.each(~Block,~Output,~Unit,~Context,~Bands.unshared,~leaf,~keep,Bands.Split{left_plan,right_plan},enabled,context,
          Partition.Fork{Model.shaped_pack(~Part,~Block,~pack,left),Model.shaped_pack(~Part,~Block,~pack,right)},Unit{}),
        Partition.Fork{Bands.each(~Block,~Output,~Unit,~Context,~Bands.unshared,~leaf,~keep,left_plan,enabled,context,Model.shaped_pack(~Part,~Block,~pack,left),Unit{}),
          Bands.each(~Block,~Output,~Unit,~Context,~Bands.unshared,~leaf,~keep,right_plan,enabled,context,Model.shaped_pack(~Part,~Block,~pack,right),Unit{})},
        fork(~Block,~Output,~Unit,~Context,~Bands.unshared,~leaf,~keep,left_plan,right_plan,enabled,context,
          Model.shaped_pack(~Part,~Block,~pack,left),Model.shaped_pack(~Part,~Block,~pack,right),Unit{})) :
        {_ == Partition.Fork{shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,left),shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,right)} : Tree(~Output)}
      %Equal.sym(Tree(~Output),
        Bands.each(~Block,~Output,~Unit,~Context,~Bands.unshared,~leaf,~keep,left_plan,enabled,context,Model.shaped_pack(~Part,~Block,~pack,left),Unit{}),
        shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,left),
        shaped_planned(~Part,~Block,~pack,~Output,~Context,~leaf,~keep,left,enabled,context)) :
        {Partition.Fork{_,Bands.each(~Block,~Output,~Unit,~Context,~Bands.unshared,~leaf,~keep,right_plan,enabled,context,Model.shaped_pack(~Part,~Block,~pack,right),Unit{})} ==
          Partition.Fork{shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,left),shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,right)} : Tree(~Output)}
      %Equal.sym(Tree(~Output),
        Bands.each(~Block,~Output,~Unit,~Context,~Bands.unshared,~leaf,~keep,right_plan,enabled,context,Model.shaped_pack(~Part,~Block,~pack,right),Unit{}),
        shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,right),
        shaped_planned(~Part,~Block,~pack,~Output,~Context,~leaf,~keep,right,enabled,context)) :
        {Partition.Fork{shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,left),_} ==
          Partition.Fork{shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,left),shaped_leaf_map(~Part,~Block,~pack,~Output,~Context,~leaf,context,right)} : Tree(~Output)}
      {==}

# Every band of a checked transform over a tree of any parts, from the rule
# for one call on one piece. What the rule needs of a piece is stated of the
# whole tree; the rule needs to hold only of usable calls.
def shaped_checked_all(~Element: Data,~Part: Data,~Block: Type,~pack: Part -> Block,~Context: Data,~leaf: Context -> U32 -> U32 -> Block -> Unit -> Bands.Transformed(~Element),
  ~Scope: Data,~context: @+scope: Scope -> Context,~First: @+scope: Scope -> @+part: Part -> @+first: U32 -> @+rows: U32 -> Data,
  ~After: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~rule: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+part: Part -> @fact: First(scope,part,first,rows) ->
    @accepted: {usable(~Element,checked(~Element,leaf(context(scope),first,rows,pack(part),Unit{}))) == True{} : Bool} ->
    Model.ContentAll(~Element,~After(scope),Model.contents(~Element,made(~Element,checked(~Element,leaf(context(scope),first,rows,pack(part),Unit{}))))),
  +scope: Scope,tree: Model.Shaped<Part>,facts: Model.ShapedAll(~Part,~Scope,~First,scope,tree),
  accepted: {usable(~Element,checked(~Element,shaped_leaf_map(~Part,~Block,~pack,~Bands.Checked<Element>,~Context,~leaf,context(scope),tree))) == True{} : Bool}) ->
  Model.ContentAll(~Element,~After(scope),Model.contents(~Element,made(~Element,checked(~Element,shaped_leaf_map(~Part,~Block,~pack,~Bands.Checked<Element>,~Context,~leaf,context(scope),tree))))):
  match tree:
    case Model.Piece{+part,+first,+rows}: rule(scope,first,rows,part,facts,accepted)
    case Model.Pieces{+left,+right}:
      Shape.Both{left_facts,right_facts} = facts
      checked_joined(~Element,~Scope,~After,scope,checked(~Element,shaped_leaf_map(~Part,~Block,~pack,~Bands.Checked<Element>,~Context,~leaf,context(scope),left)),
        checked(~Element,shaped_leaf_map(~Part,~Block,~pack,~Bands.Checked<Element>,~Context,~leaf,context(scope),right)),
        accepted => shaped_checked_all(~Element,~Part,~Block,~pack,~Context,~leaf,~Scope,~context,~First,~After,~rule,scope,left,left_facts,accepted),
        accepted => shaped_checked_all(~Element,~Part,~Block,~pack,~Context,~leaf,~Scope,~context,~First,~After,~rule,scope,right,right_facts,accepted),accepted)

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
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Unit -> Bands.Bands(~Element),
  ~leaf_model: Context -> U32 -> U32 -> Buffer.Model<Element> -> Buffer.Model<Element> -> Model.Content<Element>) -> Type:
  @+context: Context -> @+first: U32 -> @+rows: U32 -> @+left: Buffer.Model<Element> -> @+right: Buffer.Model<Element> ->
  {Model.contents(~Element,leaf(context,first,rows,Buffer.pack(~Element,left),Buffer.pack(~Element,right),Unit{})) == leaf_model(context,first,rows,left,right) : Model.Content<Element>}

def Zipped(~Element: Data,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Unit -> Bands.Bands(~Element),
  +layout: Bands.Plan,+enabled: Bool,+context: Context,left: Bands.Bands(~Element),right: Bands.Bands(~Element)) -> Bands.Bands(~Element):
  Bands.each(~Bands.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~Storage.Buffer<Element>,~Unit,~Context,~Bands.unshared,
    ~Bands.paired(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~leaf),~Bands.left_of(~Storage.Buffer<Element>,~Storage.Buffer<Element>),
    layout,enabled,context,Bands.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,left,right),Unit{})

# A two-tree operation on the plan of its left tree: the tree induction and
# the shape mismatches, once.
def zipped(~Element: Data,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Unit -> Bands.Bands(~Element),
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
        Zipped(~Element,~Context,~leaf,Bands.Split{Model.layout(~Element,left),Model.layout(~Element,right)},enabled,context,
          Model.pack(~Element,Model.Fork{left,right}),Model.pack(~Element,Model.Leaf{buffer,other_first,other_rows})),
        only_kept(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~Bands.unshared,~leaf,
          Bands.Split{Model.layout(~Element,left),Model.layout(~Element,right)},enabled,context,Model.pack(~Element,Model.Fork{left,right}),Unit{}))
    case Model.Fork{+left,+right} Model.Fork{+other_left,+other_right}:
      %Equal.sym(Bands.Bands(~Element),
        Zipped(~Element,~Context,~leaf,Bands.Split{Model.layout(~Element,left),Model.layout(~Element,right)},enabled,context,
          Model.pack(~Element,Model.Fork{left,right}),Model.pack(~Element,Model.Fork{other_left,other_right})),
        Partition.Fork{Zipped(~Element,~Context,~leaf,Model.layout(~Element,left),enabled,context,
            Model.pack(~Element,left),Model.pack(~Element,other_left)),
          Zipped(~Element,~Context,~leaf,Model.layout(~Element,right),enabled,context,
            Model.pack(~Element,right),Model.pack(~Element,other_right))},
        fork(~Bands.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~Storage.Buffer<Element>,~Unit,~Context,~Bands.unshared,
          ~Bands.paired(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~leaf),~Bands.left_of(~Storage.Buffer<Element>,~Storage.Buffer<Element>),
          Model.layout(~Element,left),Model.layout(~Element,right),enabled,context,
          Bands.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,left),Model.pack(~Element,other_left)),
          Bands.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,right),Model.pack(~Element,other_right)),Unit{})) :
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
def row_first(layout: Bands.Plan) -> U32:
  match layout:
    case Bands.Rows{first,rows}: first
    case Bands.Split{left,right}: 0

def row_count(layout: Bands.Plan) -> U32:
  match layout:
    case Bands.Rows{first,rows}: rows
    case Bands.Split{left,right}: 0

def layout_match_leaf(layout: Bands.Plan) -> Bool:
  match layout:
    case Bands.Rows{first,rows}: True{}
    case Bands.Split{left,right}: False{}

def left_plan(layout: Bands.Plan) -> Bands.Plan:
  match layout:
    case Bands.Rows{first,rows}: Bands.Rows{0,0}
    case Bands.Split{left,right}: left

def right_plan(layout: Bands.Plan) -> Bands.Plan:
  match layout:
    case Bands.Rows{first,rows}: Bands.Rows{0,0}
    case Bands.Split{left,right}: right

def Common(~Element: Data,left: Model.Model<Element>,right: Model.Model<Element>) -> Data:
  match left right:
    case Model.Leaf{left,+first,+rows} Model.Leaf{right,+other_first,+other_rows}:
      Shape.Both<{first == other_first : U32},{rows == other_rows : U32}>
    case Model.Fork{left,right} Model.Fork{other_left,other_right}:
      Shape.Both<Common(~Element,left,other_left),Common(~Element,right,other_right)>
    case left right: Empty

def matched(~Element: Data,+left: Model.Model<Element>,+right: Model.Model<Element>,+same: {Model.layout(~Element,left) == Model.layout(~Element,right) : Bands.Plan}) -> Common(~Element,left,right):
  match left right:
    case Model.Leaf{left,+first,+rows} Model.Leaf{right,+other_first,+other_rows}:
      Shape.Both{Equal.cong(Bands.Plan,U32,layout => row_first(layout),Bands.Rows{first,rows},Bands.Rows{other_first,other_rows},same),
        Equal.cong(Bands.Plan,U32,layout => row_count(layout),Bands.Rows{first,rows},Bands.Rows{other_first,other_rows},same)}
    case Model.Fork{+left,+right} Model.Fork{+other_left,+other_right}:
      +first_plan = {Bands.Split{Model.layout(~Element,left),Model.layout(~Element,right)} : Bands.Plan}
      +second_plan = {Bands.Split{Model.layout(~Element,other_left),Model.layout(~Element,other_right)} : Bands.Plan}
      Shape.Both{matched(~Element,left,other_left,Equal.cong(Bands.Plan,Bands.Plan,layout => left_plan(layout),first_plan,second_plan,same)),
        matched(~Element,right,other_right,Equal.cong(Bands.Plan,Bands.Plan,layout => right_plan(layout),first_plan,second_plan,same))}
    case Model.Leaf{buffer,+first,+rows} Model.Fork{+left,+right}:
      WordEncoding.false_true(Equal.sym(Bool,True{},False{},Equal.cong(Bands.Plan,Bool,layout => layout_match_leaf(layout),Bands.Rows{first,rows},Bands.Split{Model.layout(~Element,left),Model.layout(~Element,right)},same)))
    case Model.Fork{+left,+right} Model.Leaf{buffer,+first,+rows}:
      WordEncoding.false_true(Equal.cong(Bands.Plan,Bool,layout => layout_match_leaf(layout),Bands.Split{Model.layout(~Element,left),Model.layout(~Element,right)},Bands.Rows{first,rows},same))

# One band of each of two trees of one shape.
type BandPair<-Element: Data> is Data:
  BandPair{left: Buffer.Model<Element>,right: Buffer.Model<Element>}

def pair_pack(~Element: Data,pair: BandPair<Element>) -> Bands.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>:
  BandPair{left,right} = pair
  Bands.Both{Buffer.pack(~Element,left),Buffer.pack(~Element,right)}

# The bands of a tree, each paired with itself.
def self_pieces(~Element: Data,model: Model.Model<Element>) -> Model.Shaped<BandPair<Element>>:
  match model:
    case Model.Leaf{+buffer,first,rows}: Model.Piece{BandPair{buffer,buffer},first,rows}
    case Model.Fork{left,right}: Model.Pieces{self_pieces(~Element,left),self_pieces(~Element,right)}

# The pieces of two trees, each with the bands of both, at the rows of the
# left tree; where the shapes differ, the left tree is paired with itself.
def paired_pieces(~Element: Data,left: Model.Model<Element>,right: Model.Model<Element>) -> Model.Shaped<BandPair<Element>>:
  match left right:
    case Model.Leaf{left,first,rows} Model.Leaf{right,other_first,other_rows}: Model.Piece{BandPair{left,right},first,rows}
    case Model.Fork{left,right} Model.Fork{other_left,other_right}:
      Model.Pieces{paired_pieces(~Element,left,other_left),paired_pieces(~Element,right,other_right)}
    case left right: self_pieces(~Element,left)

def pieces_pack(~Element: Data,left: Model.Model<Element>,right: Model.Model<Element>) -> Tree(~Bands.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>):
  Model.shaped_pack(~BandPair<Element>,~Bands.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~pair_pack(~Element),paired_pieces(~Element,left,right))

# Two trees of one shape zip into the pairs of their pieces.
def paired_zip(~Element: Data,+left: Model.Model<Element>,+right: Model.Model<Element>,common: Common(~Element,left,right)) ->
  {Bands.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,left),Model.pack(~Element,right)) ==
    Model.shaped_pack(~BandPair<Element>,~Bands.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~pair_pack(~Element),paired_pieces(~Element,left,right)) :
    Tree(~Bands.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>)}:
  match left right:
    case Model.Leaf{+buffer,+first,+rows} Model.Leaf{+other,other_first,other_rows}: {==}
    case Model.Fork{+left,+right} Model.Fork{+other_left,+other_right}:
      Shape.Both{left_common,right_common} = common
      %Equal.sym(Tree(~Bands.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>),
          Bands.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,left),Model.pack(~Element,other_left)),pieces_pack(~Element,left,other_left),
          paired_zip(~Element,left,other_left,left_common)) :
        {Partition.Fork{_,Bands.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,right),Model.pack(~Element,other_right))} ==
          Partition.Fork{pieces_pack(~Element,left,other_left),pieces_pack(~Element,right,other_right)} : Tree(~Bands.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>)}
      %Equal.sym(Tree(~Bands.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>),
          Bands.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,right),Model.pack(~Element,other_right)),pieces_pack(~Element,right,other_right),
          paired_zip(~Element,right,other_right,right_common)) :
        {Partition.Fork{pieces_pack(~Element,left,other_left),_} ==
          Partition.Fork{pieces_pack(~Element,left,other_left),pieces_pack(~Element,right,other_right)} : Tree(~Bands.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>)}
      {==}
    case Model.Leaf{+buffer,+first,+rows} Model.Fork{+other_left,+other_right}:
      Empty.absurd({Bands.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,Model.Leaf{buffer,first,rows}),Model.pack(~Element,Model.Fork{other_left,other_right})) ==
        Model.shaped_pack(~BandPair<Element>,~Bands.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~pair_pack(~Element),
          paired_pieces(~Element,Model.Leaf{buffer,first,rows},Model.Fork{other_left,other_right})) : Tree(~Bands.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>)},common)
    case Model.Fork{+left,+right} Model.Leaf{+buffer,+first,+rows}:
      Empty.absurd({Bands.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,Model.Fork{left,right}),Model.pack(~Element,Model.Leaf{buffer,first,rows})) ==
        Model.shaped_pack(~BandPair<Element>,~Bands.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~pair_pack(~Element),
          paired_pieces(~Element,Model.Fork{left,right},Model.Leaf{buffer,first,rows})) : Tree(~Bands.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>)},common)

# The plan of the rows of the pieces of two trees of one shape is the layout
# of the left tree.
def paired_plan(~Element: Data,+left: Model.Model<Element>,+right: Model.Model<Element>,common: Common(~Element,left,right)) ->
  {Model.shaped_plan(~BandPair<Element>,~piece_rows(~BandPair<Element>),paired_pieces(~Element,left,right)) == Model.layout(~Element,left) : Bands.Plan}:
  match left right:
    case Model.Leaf{+buffer,+first,+rows} Model.Leaf{+other,other_first,other_rows}: {==}
    case Model.Fork{+left,+right} Model.Fork{+other_left,+other_right}:
      Shape.Both{left_common,right_common} = common
      %Equal.sym(Bands.Plan,Model.shaped_plan(~BandPair<Element>,~piece_rows(~BandPair<Element>),paired_pieces(~Element,left,other_left)),Model.layout(~Element,left),
          paired_plan(~Element,left,other_left,left_common)) :
        {Bands.Split{_,Model.shaped_plan(~BandPair<Element>,~piece_rows(~BandPair<Element>),paired_pieces(~Element,right,other_right))} ==
          Bands.Split{Model.layout(~Element,left),Model.layout(~Element,right)} : Bands.Plan}
      %Equal.sym(Bands.Plan,Model.shaped_plan(~BandPair<Element>,~piece_rows(~BandPair<Element>),paired_pieces(~Element,right,other_right)),Model.layout(~Element,right),
          paired_plan(~Element,right,other_right,right_common)) :
        {Bands.Split{Model.layout(~Element,left),_} == Bands.Split{Model.layout(~Element,left),Model.layout(~Element,right)} : Bands.Plan}
      {==}
    case Model.Leaf{+buffer,+first,+rows} Model.Fork{+other_left,+other_right}:
      Empty.absurd({Model.shaped_plan(~BandPair<Element>,~piece_rows(~BandPair<Element>),paired_pieces(~Element,Model.Leaf{buffer,first,rows},Model.Fork{other_left,other_right})) ==
        Model.layout(~Element,Model.Leaf{buffer,first,rows}) : Bands.Plan},common)
    case Model.Fork{+left,+right} Model.Leaf{+buffer,+first,+rows}:
      Empty.absurd({Model.shaped_plan(~BandPair<Element>,~piece_rows(~BandPair<Element>),paired_pieces(~Element,Model.Fork{left,right},Model.Leaf{buffer,first,rows})) ==
        Model.layout(~Element,Model.Fork{left,right}) : Bands.Plan},common)

# What holds of the band of each tree, at the rows of the pair.
def PairAt(~Element: Data,~Context: Data,~First: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Second: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+context: Context,+pair: BandPair<Element>,+first: U32,+rows: U32) -> Data:
  BandPair{+left,+right} = pair
  Shape.Both<First(context,first,rows,Buffer.content(~Element,left)),Second(context,first,rows,Buffer.content(~Element,right))>

# What holds of every band of each of two trees of one shape holds of every
# piece of their pairs.
def paired_all(~Element: Data,~Context: Data,~First: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~Second: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+context: Context,
  +left: Model.Model<Element>,+right: Model.Model<Element>,common: Common(~Element,left,right),
  left_facts: Model.ContentAll(~Element,~First(context),Model.content(~Element,left)),right_facts: Model.ContentAll(~Element,~Second(context),Model.content(~Element,right))) ->
  Model.ShapedAll(~BandPair<Element>,~Context,~PairAt(~Element,~Context,~First,~Second),context,paired_pieces(~Element,left,right)):
  match left right:
    case Model.Leaf{+buffer,+first,+rows} Model.Leaf{+other,+other_first,+other_rows}:
      Shape.Both{+same_first,+same_rows} = common
      Shape.Both{left_facts,
        %Equal.sym(U32,first,other_first,same_first) : Second(context,_,rows,Buffer.content(~Element,other))
        %Equal.sym(U32,rows,other_rows,same_rows) : Second(context,other_first,_,Buffer.content(~Element,other))
        right_facts}
    case Model.Fork{+left,+right} Model.Fork{+other_left,+other_right}:
      Shape.Both{left_common,right_common} = common
      Shape.Both{left_left,left_right} = left_facts
      Shape.Both{right_left,right_right} = right_facts
      Shape.Both{paired_all(~Element,~Context,~First,~Second,context,left,other_left,left_common,left_left,right_left),
        paired_all(~Element,~Context,~First,~Second,context,right,other_right,right_common,left_right,right_right)}
    case Model.Leaf{+buffer,+first,+rows} Model.Fork{+other_left,+other_right}:
      Empty.absurd(Model.ShapedAll(~BandPair<Element>,~Context,~PairAt(~Element,~Context,~First,~Second),context,
        paired_pieces(~Element,Model.Leaf{buffer,first,rows},Model.Fork{other_left,other_right})),common)
    case Model.Fork{+left,+right} Model.Leaf{+buffer,+first,+rows}:
      Empty.absurd(Model.ShapedAll(~BandPair<Element>,~Context,~PairAt(~Element,~Context,~First,~Second),context,
        paired_pieces(~Element,Model.Fork{left,right},Model.Leaf{buffer,first,rows})),common)

# Every pair of bands of two trees through the leaf of an operation, in the
# shape of the left tree.
def paired_map(~Element: Data,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Bands.Transformed(~Element),
  +context: Context,left: Model.Model<Element>,right: Model.Model<Element>) -> Bands.Transformed(~Element):
  shaped_leaf_map(~BandPair<Element>,~Bands.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~pair_pack(~Element),~Bands.Checked<Element>,~Context,
    ~Bands.transform_pair(~Element,~Context,~leaf),context,paired_pieces(~Element,left,right))

# The checked transform of two trees of one shape on the plan of the left tree
# runs its leaf on every pair of bands, whichever halves run in parallel.
def paired_transformed(~Element: Data,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Bands.Transformed(~Element),
  +parallel: Bool,+context: Context,+left: Model.Model<Element>,+right: Model.Model<Element>,+common: Common(~Element,left,right)) ->
  {Bands.transform_paired(~Element,~Context,~leaf,Model.layout(~Element,left),parallel,context,
      Bands.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,left),Model.pack(~Element,right))) ==
    checked(~Element,paired_map(~Element,~Context,~leaf,context,left,right)) : Bands.Bands(~Element) & Bool}:
  +pieces = paired_pieces(~Element,left,right)
  %Equal.sym(Tree(~Bands.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>),
      Bands.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,left),Model.pack(~Element,right)),pieces_pack(~Element,left,right),
      paired_zip(~Element,left,right,common)) :
    {checked(~Element,Bands.each(~Bands.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~Bands.Checked<Element>,~Unit,~Context,~Bands.unshared,
        ~Bands.transform_pair(~Element,~Context,~leaf),~Bands.untransformed_pairs(~Element),Model.layout(~Element,left),parallel,context,_,Unit{})) ==
      checked(~Element,paired_map(~Element,~Context,~leaf,context,left,right)) : Bands.Bands(~Element) & Bool}
  %paired_plan(~Element,left,right,common) :
    {checked(~Element,Bands.each(~Bands.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~Bands.Checked<Element>,~Unit,~Context,~Bands.unshared,
        ~Bands.transform_pair(~Element,~Context,~leaf),~Bands.untransformed_pairs(~Element),_,parallel,context,pieces_pack(~Element,left,right),Unit{})) ==
      checked(~Element,paired_map(~Element,~Context,~leaf,context,left,right)) : Bands.Bands(~Element) & Bool}
  Equal.cong(Tree(~Bands.Checked<Element>),Bands.Bands(~Element) & Bool,tree => checked(~Element,tree),
    Bands.each(~Bands.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~Bands.Checked<Element>,~Unit,~Context,~Bands.unshared,
      ~Bands.transform_pair(~Element,~Context,~leaf),~Bands.untransformed_pairs(~Element),Model.shaped_plan(~BandPair<Element>,~piece_rows(~BandPair<Element>),pieces),parallel,
      context,pieces_pack(~Element,left,right),Unit{}),
    paired_map(~Element,~Context,~leaf,context,left,right),
    shaped_planned(~BandPair<Element>,~Bands.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~pair_pack(~Element),~Bands.Checked<Element>,~Context,
      ~Bands.transform_pair(~Element,~Context,~leaf),~Bands.untransformed_pairs(~Element),pieces,parallel,context))

# A tree made from a plan with a state threaded through its leaves
# (Bands.build). The state is a model held in its owner, and `see` reads from
# the owner what the proofs need. A leaf that keeps what is seen and makes one
# band makes, over the plan, the tree of those bands: an operation proves the
# rule of its leaf.
type Held<-State: Type,-Kept: Data,-hold: Kept -> State,-state: State> is Data:
  Held{kept: Kept,equation: {hold(kept) == state : State}}

def buffer_kept(~Element: Data,-buffer: Storage.Buffer<Element>,observation: Buffer.Observation<Element,buffer>) -> Held<Storage.Buffer<Element>,Buffer.Model<Element>,Buffer.pack(~Element),buffer>:
  Buffer.Observation{model,equation} = observation
  Held{model,equation}

def buffer_held(~Element: Data,buffer: Storage.Buffer<Element>) -> Held<Storage.Buffer<Element>,Buffer.Model<Element>,Buffer.pack(~Element),buffer>:
  buffer_kept(~Element,buffer,Buffer.observe(~Element,buffer))

def bands_kept(~Element: Data,-bands: Bands.Bands(~Element),observation: Model.Observation<Element,bands>) -> Held<Bands.Bands(~Element),Model.Model<Element>,Model.pack(~Element),bands>:
  Model.Observation{model,equation} = observation
  Held{model,equation}

def bands_held(~Element: Data,bands: Bands.Bands(~Element)) -> Held<Bands.Bands(~Element),Model.Model<Element>,Model.pack(~Element),bands>:
  bands_kept(~Element,bands,Model.observe(~Element,bands))

def seen(~Element: Data,~State: Type,~See: Data,~see: State -> See,made: State & Bands.Bands(~Element)) -> See & Model.Content<Element>:
  (state,bands) = made
  (see(state),Model.contents(~Element,bands))

def BuildRule(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> State -> State & Bands.Bands(~Element),~part: Context -> See -> @+first: U32 -> @+rows: U32 -> Model.Content<Element>) -> Type:
  @+context: Context -> @+first: U32 -> @+rows: U32 -> @+kept: Kept ->
  {seen(~Element,~State,~See,~see,leaf(context,first,rows,hold(kept))) == (see(hold(kept)),part(context,see(hold(kept)),first,rows)) : See & Model.Content<Element>}

def build_joined(~Element: Data,~See: Data,left: Model.Content<Element>,right: See & Model.Content<Element>) -> See & Model.Content<Element>:
  (state,right) = right
  (state,Model.Joined{left,right})

def build_join(~Element: Data,~State: Type,left: Bands.Bands(~Element),right: State & Bands.Bands(~Element)) -> State & Bands.Bands(~Element):
  Storage.bind_pair(~State,~Bands.Bands(~Element),~(State & Bands.Bands(~Element)),right,state => right => (state,Partition.Fork{left,right}))

def build_join_seen(~Element: Data,~State: Type,~See: Data,~see: State -> See,left: Bands.Bands(~Element),right: State & Bands.Bands(~Element)) ->
  {seen(~Element,~State,~See,~see,build_join(~Element,~State,left,right)) == build_joined(~Element,~See,Model.contents(~Element,left),seen(~Element,~State,~See,~see,right)) : See & Model.Content<Element>}:
  (state,right) = right
  {==}

def build_spread(~Element: Data,~Context: Data,~See: Data,~part: Context -> See -> @+first: U32 -> @+rows: U32 -> Model.Content<Element>,layout: Bands.Plan,context: Context,made: See & Model.Content<Element>) -> See & Model.Content<Element>:
  (state,left) = made
  +state = state
  (state,Model.Joined{left,Model.planned(~Element,~part(context,state),layout)})

def build_continued(~Element: Data,~State: Type,~Context: Data,~leaf: Context -> U32 -> U32 -> State -> State & Bands.Bands(~Element),layout: Bands.Plan,+context: Context,left: State & Bands.Bands(~Element)) -> State & Bands.Bands(~Element):
  Storage.bind_pair(~State,~Bands.Bands(~Element),~(State & Bands.Bands(~Element)),left,state => left => build_join(~Element,~State,left,Bands.build(~Storage.Buffer<Element>,~State,~Context,~leaf,layout,context,state)))

def build_continue_packed(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> State -> State & Bands.Bands(~Element),~part: Context -> See -> @+first: U32 -> @+rows: U32 -> Model.Content<Element>,
  +layout: Bands.Plan,+context: Context,-before: See,-left_model: Model.Content<Element>,+kept: Kept,+left: Model.Model<Element>,
  left_seen: {seen(~Element,~State,~See,~see,(hold(kept),Model.pack(~Element,left))) == (before,left_model) : See & Model.Content<Element>},rule: @+kept: Kept -> {seen(~Element,~State,~See,~see,Bands.build(~Storage.Buffer<Element>,~State,~Context,~leaf,layout,context,hold(kept))) == (see(hold(kept)),Model.planned(~Element,~part(context,see(hold(kept))),layout)) : See & Model.Content<Element>}) ->
  {seen(~Element,~State,~See,~see,build_continued(~Element,~State,~Context,~leaf,layout,context,(hold(kept),Model.pack(~Element,left)))) == (before,Model.Joined{left_model,Model.planned(~Element,~part(context,before),layout)}) : See & Model.Content<Element>}:
  Equal.trans(See & Model.Content<Element>,seen(~Element,~State,~See,~see,build_join(~Element,~State,Model.pack(~Element,left),Bands.build(~Storage.Buffer<Element>,~State,~Context,~leaf,layout,context,hold(kept)))),
    build_joined(~Element,~See,Model.contents(~Element,Model.pack(~Element,left)),seen(~Element,~State,~See,~see,Bands.build(~Storage.Buffer<Element>,~State,~Context,~leaf,layout,context,hold(kept)))),
    (before,Model.Joined{left_model,Model.planned(~Element,~part(context,before),layout)}),
    build_join_seen(~Element,~State,~See,~see,Model.pack(~Element,left),Bands.build(~Storage.Buffer<Element>,~State,~Context,~leaf,layout,context,hold(kept))),
    Equal.trans(See & Model.Content<Element>,build_joined(~Element,~See,Model.contents(~Element,Model.pack(~Element,left)),seen(~Element,~State,~See,~see,Bands.build(~Storage.Buffer<Element>,~State,~Context,~leaf,layout,context,hold(kept)))),
      build_spread(~Element,~Context,~See,~part,layout,context,(see(hold(kept)),Model.contents(~Element,Model.pack(~Element,left)))),
      (before,Model.Joined{left_model,Model.planned(~Element,~part(context,before),layout)}),
      Equal.cong(See & Model.Content<Element>,See & Model.Content<Element>,made => build_joined(~Element,~See,Model.contents(~Element,Model.pack(~Element,left)),made),
        seen(~Element,~State,~See,~see,Bands.build(~Storage.Buffer<Element>,~State,~Context,~leaf,layout,context,hold(kept))),(see(hold(kept)),Model.planned(~Element,~part(context,see(hold(kept))),layout)),rule(kept)),
      Equal.cong(See & Model.Content<Element>,See & Model.Content<Element>,made => build_spread(~Element,~Context,~See,~part,layout,context,made),
        (see(hold(kept)),Model.contents(~Element,Model.pack(~Element,left))),(before,left_model),left_seen)))

def build_continue_observed(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> State -> State & Bands.Bands(~Element),~part: Context -> See -> @+first: U32 -> @+rows: U32 -> Model.Content<Element>,
  +layout: Bands.Plan,+context: Context,-before: See,-left_model: Model.Content<Element>,-state: State,-left: Bands.Bands(~Element),
  state_held: Held<State,Kept,hold,state>,left_observation: Model.Observation<Element,left>,
  left_seen: {seen(~Element,~State,~See,~see,(state,left)) == (before,left_model) : See & Model.Content<Element>},rule: @+kept: Kept -> {seen(~Element,~State,~See,~see,Bands.build(~Storage.Buffer<Element>,~State,~Context,~leaf,layout,context,hold(kept))) == (see(hold(kept)),Model.planned(~Element,~part(context,see(hold(kept))),layout)) : See & Model.Content<Element>}) ->
  {seen(~Element,~State,~See,~see,build_continued(~Element,~State,~Context,~leaf,layout,context,(state,left))) == (before,Model.Joined{left_model,Model.planned(~Element,~part(context,before),layout)}) : See & Model.Content<Element>}:
  Held{+kept,+state_equal} = state_held
  Model.Observation{+left_owner,+left_equal} = left_observation
  %state_equal : {seen(~Element,~State,~See,~see,build_continued(~Element,~State,~Context,~leaf,layout,context,(_,left))) == (before,Model.Joined{left_model,Model.planned(~Element,~part(context,before),layout)}) : See & Model.Content<Element>}
  %left_equal : {seen(~Element,~State,~See,~see,build_continued(~Element,~State,~Context,~leaf,layout,context,(hold(kept),_))) == (before,Model.Joined{left_model,Model.planned(~Element,~part(context,before),layout)}) : See & Model.Content<Element>}
  build_continue_packed(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~part,layout,context,before,left_model,kept,left_owner,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,left_owner),left,left_equal) : {seen(~Element,~State,~See,~see,(hold(kept),_)) == (before,left_model) : See & Model.Content<Element>}
    %Equal.sym(State,hold(kept),state,state_equal) : {seen(~Element,~State,~See,~see,(_,left)) == (before,left_model) : See & Model.Content<Element>}
    left_seen,rule)

def build_continue(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> State -> State & Bands.Bands(~Element),~part: Context -> See -> @+first: U32 -> @+rows: U32 -> Model.Content<Element>,
  +layout: Bands.Plan,+context: Context,-before: See,-left_model: Model.Content<Element>,left: State & Bands.Bands(~Element),
  left_seen: {seen(~Element,~State,~See,~see,left) == (before,left_model) : See & Model.Content<Element>},rule: @+kept: Kept -> {seen(~Element,~State,~See,~see,Bands.build(~Storage.Buffer<Element>,~State,~Context,~leaf,layout,context,hold(kept))) == (see(hold(kept)),Model.planned(~Element,~part(context,see(hold(kept))),layout)) : See & Model.Content<Element>}) ->
  {seen(~Element,~State,~See,~see,build_continued(~Element,~State,~Context,~leaf,layout,context,left)) == (before,Model.Joined{left_model,Model.planned(~Element,~part(context,before),layout)}) : See & Model.Content<Element>}:
  (state,bands) = left
  build_continue_observed(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~part,layout,context,before,left_model,state,bands,observe(state),Model.observe(~Element,bands),left_seen,rule)

def built_parts(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> State -> State & Bands.Bands(~Element),~part: Context -> See -> @+first: U32 -> @+rows: U32 -> Model.Content<Element>,
  ~leaf_rule: BuildRule(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~part),layout: Bands.Plan,+context: Context,+kept: Kept) ->
  {seen(~Element,~State,~See,~see,Bands.build(~Storage.Buffer<Element>,~State,~Context,~leaf,layout,context,hold(kept))) == (see(hold(kept)),Model.planned(~Element,~part(context,see(hold(kept))),layout)) : See & Model.Content<Element>}:
  match layout:
    case Bands.Rows{+first,+rows}: leaf_rule(context,first,rows,kept)
    case Bands.Split{+left,+right}:
      build_continue(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~part,right,context,see(hold(kept)),Model.planned(~Element,~part(context,see(hold(kept))),left),
        Bands.build(~Storage.Buffer<Element>,~State,~Context,~leaf,left,context,hold(kept)),built_parts(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~part,~leaf_rule,left,context,kept),
        next => built_parts(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~part,~leaf_rule,right,context,next))

# A leaf that makes one band.
def row_part(~Element: Data,~Context: Data,~See: Data,~band: Context -> See -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element>,context: Context,see: See,+first: U32,+rows: U32) -> Model.Content<Element>:
  Model.RowBand{band(context,see,first,rows),first,rows}

def planned_rows(~Element: Data,~Context: Data,~See: Data,~band: Context -> See -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element>,+context: Context,+see: See,layout: Bands.Plan) ->
  {Model.planned(~Element,~row_part(~Element,~Context,~See,~band,context,see),layout) == Model.build(~Element,~band(context,see),layout) : Model.Content<Element>}:
  match layout:
    case Bands.Rows{first,rows}: {==}
    case Bands.Split{+left,+right}:
      %Equal.sym(Model.Content<Element>,Model.planned(~Element,~row_part(~Element,~Context,~See,~band,context,see),left),Model.build(~Element,~band(context,see),left),
        planned_rows(~Element,~Context,~See,~band,context,see,left)) :
        {Model.Joined{_,Model.planned(~Element,~row_part(~Element,~Context,~See,~band,context,see),right)} == Model.Joined{Model.build(~Element,~band(context,see),left),Model.build(~Element,~band(context,see),right)} : Model.Content<Element>}
      %Equal.sym(Model.Content<Element>,Model.planned(~Element,~row_part(~Element,~Context,~See,~band,context,see),right),Model.build(~Element,~band(context,see),right),
        planned_rows(~Element,~Context,~See,~band,context,see,right)) :
        {Model.Joined{Model.build(~Element,~band(context,see),left),_} == Model.Joined{Model.build(~Element,~band(context,see),left),Model.build(~Element,~band(context,see),right)} : Model.Content<Element>}
      {==}

# A leaf that keeps what is seen and makes one band makes, over the plan, the
# tree of those bands.
def built(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> State -> State & Bands.Bands(~Element),~band: Context -> See -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element>,
  ~leaf_rule: BuildRule(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~row_part(~Element,~Context,~See,~band)),+layout: Bands.Plan,+context: Context,+kept: Kept) ->
  {seen(~Element,~State,~See,~see,Bands.build(~Storage.Buffer<Element>,~State,~Context,~leaf,layout,context,hold(kept))) == (see(hold(kept)),Model.build(~Element,~band(context,see(hold(kept))),layout)) : See & Model.Content<Element>}:
  %planned_rows(~Element,~Context,~See,~band,context,see(hold(kept)),layout) :
    {seen(~Element,~State,~See,~see,Bands.build(~Storage.Buffer<Element>,~State,~Context,~leaf,layout,context,hold(kept))) == (see(hold(kept)),_) : See & Model.Content<Element>}
  built_parts(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~row_part(~Element,~Context,~See,~band),~leaf_rule,layout,context,kept)

# A fold of the code into a buffer threads its contents through the bands.
# An operation supplies only the equation of its leaf.
def FoldRule(~Element: Data,~Context: Data,~run: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element>,
  ~step: Context -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> Buffer.Content<Element> -> Buffer.Content<Element>) -> Type:
  @+context: Context -> @+first: U32 -> @+rows: U32 -> @+input: Buffer.Model<Element> -> @+output: Buffer.Model<Element> ->
  {Buffer.contents(~Element,run(context,first,rows,Buffer.pack(~Element,input),Buffer.pack(~Element,output))) == step(context,first,rows,Buffer.content(~Element,input),Buffer.content(~Element,output)) : Buffer.Content<Element>}

type Folding<-Element: Data,-Context: Data> is Data:
  Folding{context: Context,model: Model.Model<Element>}

def Folded(~Element: Data,~Context: Data,~run: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element>,
  ~step: Context -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> Buffer.Content<Element> -> Buffer.Content<Element>,folding: Folding<Element,Context>,-output: Storage.Buffer<Element>) -> Type:
  Folding{+context,+model} = folding
  {Buffer.contents(~Element,Bands.fold(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Context,~run,context,Model.pack(~Element,model),output)) == Model.thread(~Element,~Buffer.Content<Element>,~Context,~step,context,Model.content(~Element,model),Buffer.contents(~Element,output)) : Buffer.Content<Element>}

def folded(~Element: Data,~Context: Data,~run: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element>,
  ~step: Context -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> Buffer.Content<Element> -> Buffer.Content<Element>,~rule: FoldRule(~Element,~Context,~run,~step),
  +context: Context,model: Model.Model<Element>,output: Storage.Buffer<Element>) ->
  {Buffer.contents(~Element,Bands.fold(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Context,~run,context,Model.pack(~Element,model),output)) == Model.thread(~Element,~Buffer.Content<Element>,~Context,~step,context,Model.content(~Element,model),Buffer.contents(~Element,output)) : Buffer.Content<Element>}:
  match model:
    case Model.Leaf{+input,+first,+rows}:
      Buffer.lift(~Element,~Folding<Element,Context>,~Folded(~Element,~Context,~run,~step),Folding{context,Model.Leaf{input,first,rows}},output,Buffer.observe(~Element,output),
        output =>
          +output = output
          Equal.trans(Buffer.Content<Element>,Buffer.contents(~Element,run(context,first,rows,Buffer.pack(~Element,input),Buffer.pack(~Element,output))),
            step(context,first,rows,Buffer.content(~Element,input),Buffer.content(~Element,output)),
            step(context,first,rows,Buffer.content(~Element,input),Buffer.contents(~Element,Buffer.pack(~Element,output))),
            rule(context,first,rows,input,output),
            Equal.cong(Buffer.Content<Element>,Buffer.Content<Element>,state => step(context,first,rows,Buffer.content(~Element,input),state),Buffer.content(~Element,output),Buffer.contents(~Element,Buffer.pack(~Element,output)),
              Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,output)),Buffer.content(~Element,output),Buffer.packed_contents(~Element,output)))))
    case Model.Fork{+left,+right}:
      Buffer.lift(~Element,~Folding<Element,Context>,~Folded(~Element,~Context,~run,~step),Folding{context,Model.Fork{left,right}},output,Buffer.observe(~Element,output),
        output =>
          +output = output
          Equal.trans(Buffer.Content<Element>,Buffer.contents(~Element,Bands.fold(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Context,~run,context,Model.pack(~Element,right),Bands.fold(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Context,~run,context,Model.pack(~Element,left),Buffer.pack(~Element,output)))),
            Model.thread(~Element,~Buffer.Content<Element>,~Context,~step,context,Model.content(~Element,right),Buffer.contents(~Element,Bands.fold(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Context,~run,context,Model.pack(~Element,left),Buffer.pack(~Element,output)))),
            Model.thread(~Element,~Buffer.Content<Element>,~Context,~step,context,Model.content(~Element,right),Model.thread(~Element,~Buffer.Content<Element>,~Context,~step,context,Model.content(~Element,left),Buffer.contents(~Element,Buffer.pack(~Element,output)))),
            folded(~Element,~Context,~run,~step,~rule,context,right,Bands.fold(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Context,~run,context,Model.pack(~Element,left),Buffer.pack(~Element,output))),
            Equal.cong(Buffer.Content<Element>,Buffer.Content<Element>,state => Model.thread(~Element,~Buffer.Content<Element>,~Context,~step,context,Model.content(~Element,right),state),Buffer.contents(~Element,Bands.fold(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Context,~run,context,Model.pack(~Element,left),Buffer.pack(~Element,output))),
              Model.thread(~Element,~Buffer.Content<Element>,~Context,~step,context,Model.content(~Element,left),Buffer.contents(~Element,Buffer.pack(~Element,output))),folded(~Element,~Context,~run,~step,~rule,context,left,Buffer.pack(~Element,output)))))

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
  rule: @+kept: Kept -> {see(Bands.fold(~Block,~State,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,right),hold(kept))) == Model.shaped_thread(~Part,~See,~Context,~step,context,right,see(hold(kept))) : See}) ->
  {see(Bands.fold(~Block,~State,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,right),state)) == Model.shaped_thread(~Part,~See,~Context,~step,context,right,before) : See}:
  Held{+kept,+equation} = held
  Equal.trans(See,see(Bands.fold(~Block,~State,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,right),state)),
    Model.shaped_thread(~Part,~See,~Context,~step,context,right,see(state)),Model.shaped_thread(~Part,~See,~Context,~step,context,right,before),
    %equation : {see(Bands.fold(~Block,~State,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,right),_)) == Model.shaped_thread(~Part,~See,~Context,~step,context,right,see(_)) : See}
    rule(kept),
    Equal.cong(See,See,value => Model.shaped_thread(~Part,~See,~Context,~step,context,right,value),see(state),before,left_seen))

def shaped_folded(~Part: Data,~Block: Type,~pack: Part -> Block,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> Block -> State -> State,~step: Context -> @+first: U32 -> @+rows: U32 -> Part -> See -> See,
  ~rule: ShapedFoldRule(~Part,~Block,~pack,~State,~Context,~Kept,~hold,~See,~see,~leaf,~step),tree: Model.Shaped<Part>,+context: Context,+kept: Kept) ->
  {see(Bands.fold(~Block,~State,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,tree),hold(kept))) == Model.shaped_thread(~Part,~See,~Context,~step,context,tree,see(hold(kept))) : See}:
  match tree:
    case Model.Piece{+part,+first,+rows}: rule(context,first,rows,part,kept)
    case Model.Pieces{+left,+right}:
      fold_continue(~Part,~Block,~pack,~State,~Context,~Kept,~hold,~See,~see,~leaf,~step,right,context,Model.shaped_thread(~Part,~See,~Context,~step,context,left,see(hold(kept))),
        Bands.fold(~Block,~State,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,left),hold(kept)),
        observe(Bands.fold(~Block,~State,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,left),hold(kept))),
        shaped_folded(~Part,~Block,~pack,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~step,~rule,left,context,kept),
        next => shaped_folded(~Part,~Block,~pack,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~step,~rule,right,context,next))

# A step on the buffer of a piece, its rows counted from a start and its scope
# read from the context, threads the band model of the pieces.
def shaped_step(~Part: Data,~Element: Data,~State: Data,~Context: Data,~Scope: Data,~scope: Context -> Scope,~start: Context -> U32,~buffer: Part -> Buffer.Model<Element>,
  ~step: Scope -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> State -> State,context: Context,+first: U32,+rows: U32,part: Part,state: State) -> State:
  +context = context
  step(scope(context),(start(context) + first : U32),rows,Buffer.content(~Element,buffer(part)),state)

def shaped_stepped(~Part: Data,~Element: Data,~State: Data,~Context: Data,~Scope: Data,~scope: Context -> Scope,~start: Context -> U32,~buffer: Part -> Buffer.Model<Element>,
  ~step: Scope -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> State -> State,+context: Context,tree: Model.Shaped<Part>,+state: State) ->
  {Model.shaped_thread(~Part,~State,~Context,~shaped_step(~Part,~Element,~State,~Context,~Scope,~scope,~start,~buffer,~step),context,tree,state) ==
    Model.thread(~Element,~State,~Scope,~step,scope(context),Model.content(~Element,Model.shaped_model(~Part,~Element,~buffer,Model.shaped_shift(~Part,start(context),tree))),state) : State}:
  match tree:
    case Model.Piece{+part,+first,+rows}: {==}
    case Model.Pieces{+left,+right}:
      %Equal.sym(State,Model.shaped_thread(~Part,~State,~Context,~shaped_step(~Part,~Element,~State,~Context,~Scope,~scope,~start,~buffer,~step),context,left,state),
        Model.thread(~Element,~State,~Scope,~step,scope(context),Model.content(~Element,Model.shaped_model(~Part,~Element,~buffer,Model.shaped_shift(~Part,start(context),left))),state),
        shaped_stepped(~Part,~Element,~State,~Context,~Scope,~scope,~start,~buffer,~step,context,left,state)) :
        {Model.shaped_thread(~Part,~State,~Context,~shaped_step(~Part,~Element,~State,~Context,~Scope,~scope,~start,~buffer,~step),context,right,_) ==
          Model.thread(~Element,~State,~Scope,~step,scope(context),Model.content(~Element,Model.shaped_model(~Part,~Element,~buffer,Model.shaped_shift(~Part,start(context),right))),
            Model.thread(~Element,~State,~Scope,~step,scope(context),Model.content(~Element,Model.shaped_model(~Part,~Element,~buffer,Model.shaped_shift(~Part,start(context),left))),state)) : State}
      shaped_stepped(~Part,~Element,~State,~Context,~Scope,~scope,~start,~buffer,~step,context,right,
        Model.thread(~Element,~State,~Scope,~step,scope(context),Model.content(~Element,Model.shaped_model(~Part,~Element,~buffer,Model.shaped_shift(~Part,start(context),left))),state))

# A walk of the code whose leaves give the state back as they found it maps
# every band, each with what is seen of the state. An operation supplies only
# the equation of its leaf.
def walk_seen(~Element: Data,~State: Type,~See: Data,~see: State -> See,made: Bands.Bands(~Element) & State) -> Model.Content<Element> & See:
  (bands,state) = made
  (Model.contents(~Element,bands),see(state))

def WalkRule(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> State -> Bands.Bands(~Element) & State,~band: Context -> See -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>) -> Type:
  @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Model<Element> -> @+kept: Kept ->
  {walk_seen(~Element,~State,~See,~see,leaf(context,first,rows,Buffer.pack(~Element,buffer),hold(kept))) == (Model.RowBand{band(context,see(hold(kept)),first,rows,Buffer.content(~Element,buffer)),first,rows},see(hold(kept))) : Model.Content<Element> & See}

def walk_joined(~Element: Data,~See: Data,left: Model.Content<Element>,right: Model.Content<Element> & See) -> Model.Content<Element> & See:
  (right,state) = right
  (Model.Joined{left,right},state)

def walk_join(~Element: Data,~State: Type,left: Bands.Bands(~Element),right: Bands.Bands(~Element) & State) -> Bands.Bands(~Element) & State:
  Storage.bind_pair(~Bands.Bands(~Element),~State,~(Bands.Bands(~Element) & State),right,right => state => (Partition.Fork{left,right},state))

def walk_join_seen(~Element: Data,~State: Type,~See: Data,~see: State -> See,left: Bands.Bands(~Element),right: Bands.Bands(~Element) & State) ->
  {walk_seen(~Element,~State,~See,~see,walk_join(~Element,~State,left,right)) == walk_joined(~Element,~See,Model.contents(~Element,left),walk_seen(~Element,~State,~See,~see,right)) : Model.Content<Element> & See}:
  (right,state) = right
  {==}

def walk_spread(~Element: Data,~Context: Data,~See: Data,~band: Context -> See -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,right: Model.Content<Element>,context: Context,made: Model.Content<Element> & See) -> Model.Content<Element> & See:
  (left,state) = made
  +state = state
  (Model.Joined{left,Model.map(~Element,~band(context,state),right)},state)

def walk_continued(~Element: Data,~State: Type,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> State -> Bands.Bands(~Element) & State,right: Bands.Bands(~Element),+context: Context,left: Bands.Bands(~Element) & State) -> Bands.Bands(~Element) & State:
  Storage.bind_pair(~Bands.Bands(~Element),~State,~(Bands.Bands(~Element) & State),left,left => state => walk_join(~Element,~State,left,Bands.walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,right,state)))

def walk_continue_packed(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> State -> Bands.Bands(~Element) & State,~band: Context -> See -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,
  +right: Model.Model<Element>,+context: Context,-before: See,-left_model: Model.Content<Element>,+kept: Kept,+left: Model.Model<Element>,
  left_seen: {walk_seen(~Element,~State,~See,~see,(Model.pack(~Element,left),hold(kept))) == (left_model,before) : Model.Content<Element> & See},rule: @+kept: Kept -> {walk_seen(~Element,~State,~See,~see,Bands.walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,right),hold(kept))) == (Model.map(~Element,~band(context,see(hold(kept))),Model.content(~Element,right)),see(hold(kept))) : Model.Content<Element> & See}) ->
  {walk_seen(~Element,~State,~See,~see,walk_continued(~Element,~State,~Context,~leaf,Model.pack(~Element,right),context,(Model.pack(~Element,left),hold(kept)))) == (Model.Joined{left_model,Model.map(~Element,~band(context,before),Model.content(~Element,right))},before) : Model.Content<Element> & See}:
  Equal.trans(Model.Content<Element> & See,walk_seen(~Element,~State,~See,~see,walk_join(~Element,~State,Model.pack(~Element,left),Bands.walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,right),hold(kept)))),
    walk_joined(~Element,~See,Model.contents(~Element,Model.pack(~Element,left)),walk_seen(~Element,~State,~See,~see,Bands.walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,right),hold(kept)))),
    (Model.Joined{left_model,Model.map(~Element,~band(context,before),Model.content(~Element,right))},before),
    walk_join_seen(~Element,~State,~See,~see,Model.pack(~Element,left),Bands.walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,right),hold(kept))),
    Equal.trans(Model.Content<Element> & See,walk_joined(~Element,~See,Model.contents(~Element,Model.pack(~Element,left)),walk_seen(~Element,~State,~See,~see,Bands.walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,right),hold(kept)))),
      walk_spread(~Element,~Context,~See,~band,Model.content(~Element,right),context,(Model.contents(~Element,Model.pack(~Element,left)),see(hold(kept)))),
      (Model.Joined{left_model,Model.map(~Element,~band(context,before),Model.content(~Element,right))},before),
      Equal.cong(Model.Content<Element> & See,Model.Content<Element> & See,made => walk_joined(~Element,~See,Model.contents(~Element,Model.pack(~Element,left)),made),
        walk_seen(~Element,~State,~See,~see,Bands.walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,right),hold(kept))),(Model.map(~Element,~band(context,see(hold(kept))),Model.content(~Element,right)),see(hold(kept))),rule(kept)),
      Equal.cong(Model.Content<Element> & See,Model.Content<Element> & See,made => walk_spread(~Element,~Context,~See,~band,Model.content(~Element,right),context,made),
        (Model.contents(~Element,Model.pack(~Element,left)),see(hold(kept))),(left_model,before),left_seen)))

def walk_continue_observed(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> State -> Bands.Bands(~Element) & State,~band: Context -> See -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,
  +right: Model.Model<Element>,+context: Context,-before: See,-left_model: Model.Content<Element>,-state: State,-left: Bands.Bands(~Element),
  state_held: Held<State,Kept,hold,state>,left_observation: Model.Observation<Element,left>,
  left_seen: {walk_seen(~Element,~State,~See,~see,(left,state)) == (left_model,before) : Model.Content<Element> & See},rule: @+kept: Kept -> {walk_seen(~Element,~State,~See,~see,Bands.walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,right),hold(kept))) == (Model.map(~Element,~band(context,see(hold(kept))),Model.content(~Element,right)),see(hold(kept))) : Model.Content<Element> & See}) ->
  {walk_seen(~Element,~State,~See,~see,walk_continued(~Element,~State,~Context,~leaf,Model.pack(~Element,right),context,(left,state))) == (Model.Joined{left_model,Model.map(~Element,~band(context,before),Model.content(~Element,right))},before) : Model.Content<Element> & See}:
  Held{+kept,+state_equal} = state_held
  Model.Observation{+left_owner,+left_equal} = left_observation
  %state_equal : {walk_seen(~Element,~State,~See,~see,walk_continued(~Element,~State,~Context,~leaf,Model.pack(~Element,right),context,(left,_))) == (Model.Joined{left_model,Model.map(~Element,~band(context,before),Model.content(~Element,right))},before) : Model.Content<Element> & See}
  %left_equal : {walk_seen(~Element,~State,~See,~see,walk_continued(~Element,~State,~Context,~leaf,Model.pack(~Element,right),context,(_,hold(kept)))) == (Model.Joined{left_model,Model.map(~Element,~band(context,before),Model.content(~Element,right))},before) : Model.Content<Element> & See}
  walk_continue_packed(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~band,right,context,before,left_model,kept,left_owner,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,left_owner),left,left_equal) : {walk_seen(~Element,~State,~See,~see,(_,hold(kept))) == (left_model,before) : Model.Content<Element> & See}
    %Equal.sym(State,hold(kept),state,state_equal) : {walk_seen(~Element,~State,~See,~see,(left,_)) == (left_model,before) : Model.Content<Element> & See}
    left_seen,rule)

def walk_continue(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> State -> Bands.Bands(~Element) & State,~band: Context -> See -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,
  +right: Model.Model<Element>,+context: Context,-before: See,-left_model: Model.Content<Element>,left: Bands.Bands(~Element) & State,
  left_seen: {walk_seen(~Element,~State,~See,~see,left) == (left_model,before) : Model.Content<Element> & See},rule: @+kept: Kept -> {walk_seen(~Element,~State,~See,~see,Bands.walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,right),hold(kept))) == (Model.map(~Element,~band(context,see(hold(kept))),Model.content(~Element,right)),see(hold(kept))) : Model.Content<Element> & See}) ->
  {walk_seen(~Element,~State,~See,~see,walk_continued(~Element,~State,~Context,~leaf,Model.pack(~Element,right),context,left)) == (Model.Joined{left_model,Model.map(~Element,~band(context,before),Model.content(~Element,right))},before) : Model.Content<Element> & See}:
  (bands,state) = left
  walk_continue_observed(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~band,right,context,before,left_model,state,bands,observe(state),Model.observe(~Element,bands),left_seen,rule)

def walked(~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> State -> Bands.Bands(~Element) & State,~band: Context -> See -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> Buffer.Content<Element>,
  ~leaf_rule: WalkRule(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~band),tree: Model.Model<Element>,+context: Context,+kept: Kept) ->
  {walk_seen(~Element,~State,~See,~see,Bands.walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,tree),hold(kept))) == (Model.map(~Element,~band(context,see(hold(kept))),Model.content(~Element,tree)),see(hold(kept))) : Model.Content<Element> & See}:
  match tree:
    case Model.Leaf{+buffer,+first,+rows}: leaf_rule(context,first,rows,buffer,kept)
    case Model.Fork{+left,+right}:
      walk_continue(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~band,right,context,see(hold(kept)),Model.map(~Element,~band(context,see(hold(kept))),Model.content(~Element,left)),
        Bands.walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.pack(~Element,left),hold(kept)),walked(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~band,~leaf_rule,left,context,kept),
        next => walked(~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~band,~leaf_rule,right,context,next))

# A walk of the code over a tree of any parts whose leaf makes one band of its
# part, its rows counted from a start, and threads a state held in its owner:
# it makes the band model of the parts and threads what is seen of the state.
# An operation supplies only the equation of its leaf.
def shaped_bands(~Part: Data,~Element: Data,~buffer: Part -> Buffer.Model<Element>,+start: U32,tree: Model.Shaped<Part>) -> Model.Content<Element>:
  Model.content(~Element,Model.shaped_model(~Part,~Element,~buffer,Model.shaped_shift(~Part,start,tree)))

def ShapedWalkRule(~Part: Data,~Block: Type,~pack: Part -> Block,~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> Block -> State -> Bands.Bands(~Element) & State,~start: Context -> U32,~buffer: Part -> Buffer.Model<Element>,
  ~step: Context -> @+first: U32 -> @+rows: U32 -> Part -> See -> See) -> Type:
  @+context: Context -> @+first: U32 -> @+rows: U32 -> @+part: Part -> @+kept: Kept ->
  {walk_seen(~Element,~State,~See,~see,leaf(context,first,rows,pack(part),hold(kept))) ==
    (Model.RowBand{Buffer.content(~Element,buffer(part)),(start(context) + first : U32),rows},step(context,first,rows,part,see(hold(kept)))) : Model.Content<Element> & See}

def shaped_walk_continued(~Block: Type,~Element: Data,~State: Type,~Context: Data,~leaf: Context -> U32 -> U32 -> Block -> State -> Bands.Bands(~Element) & State,
  right: Partition.Tree<&1,Block,Unit>,+context: Context,left: Bands.Bands(~Element) & State) -> Bands.Bands(~Element) & State:
  Storage.bind_pair(~Bands.Bands(~Element),~State,~(Bands.Bands(~Element) & State),left,left => state => walk_join(~Element,~State,left,Bands.walk(~Block,~Storage.Buffer<Element>,~State,~Context,~leaf,context,right,state)))

def shaped_walk_spread(~Part: Data,~Element: Data,~See: Data,~Context: Data,~buffer: Part -> Buffer.Model<Element>,~step: Context -> @+first: U32 -> @+rows: U32 -> Part -> See -> See,
  +start: U32,+right: Model.Shaped<Part>,+context: Context,made: Model.Content<Element> & See) -> Model.Content<Element> & See:
  (left,state) = made
  (Model.Joined{left,shaped_bands(~Part,~Element,~buffer,start,right)},Model.shaped_thread(~Part,~See,~Context,~step,context,right,state))

def shaped_walk_packed(~Part: Data,~Block: Type,~pack: Part -> Block,~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> Block -> State -> Bands.Bands(~Element) & State,~start: Context -> U32,~buffer: Part -> Buffer.Model<Element>,
  ~step: Context -> @+first: U32 -> @+rows: U32 -> Part -> See -> See,
  +right: Model.Shaped<Part>,+context: Context,-before: See,-left_model: Model.Content<Element>,+kept: Kept,+left: Model.Model<Element>,
  left_seen: {walk_seen(~Element,~State,~See,~see,(Model.pack(~Element,left),hold(kept))) == (left_model,before) : Model.Content<Element> & See},
  rule: @+kept: Kept -> {walk_seen(~Element,~State,~See,~see,Bands.walk(~Block,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,right),hold(kept))) ==
    (shaped_bands(~Part,~Element,~buffer,start(context),right),Model.shaped_thread(~Part,~See,~Context,~step,context,right,see(hold(kept)))) : Model.Content<Element> & See}) ->
  {walk_seen(~Element,~State,~See,~see,shaped_walk_continued(~Block,~Element,~State,~Context,~leaf,Model.shaped_pack(~Part,~Block,~pack,right),context,(Model.pack(~Element,left),hold(kept)))) ==
    (Model.Joined{left_model,shaped_bands(~Part,~Element,~buffer,start(context),right)},Model.shaped_thread(~Part,~See,~Context,~step,context,right,before)) : Model.Content<Element> & See}:
  +packed_left = Model.contents(~Element,Model.pack(~Element,left))
  Equal.trans(Model.Content<Element> & See,walk_seen(~Element,~State,~See,~see,walk_join(~Element,~State,Model.pack(~Element,left),Bands.walk(~Block,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,right),hold(kept)))),
    walk_joined(~Element,~See,packed_left,walk_seen(~Element,~State,~See,~see,Bands.walk(~Block,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,right),hold(kept)))),
    (Model.Joined{left_model,shaped_bands(~Part,~Element,~buffer,start(context),right)},Model.shaped_thread(~Part,~See,~Context,~step,context,right,before)),
    walk_join_seen(~Element,~State,~See,~see,Model.pack(~Element,left),Bands.walk(~Block,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,right),hold(kept))),
    Equal.trans(Model.Content<Element> & See,walk_joined(~Element,~See,packed_left,walk_seen(~Element,~State,~See,~see,Bands.walk(~Block,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,right),hold(kept)))),
      shaped_walk_spread(~Part,~Element,~See,~Context,~buffer,~step,start(context),right,context,(packed_left,see(hold(kept)))),
      (Model.Joined{left_model,shaped_bands(~Part,~Element,~buffer,start(context),right)},Model.shaped_thread(~Part,~See,~Context,~step,context,right,before)),
      Equal.cong(Model.Content<Element> & See,Model.Content<Element> & See,made => walk_joined(~Element,~See,packed_left,made),
        walk_seen(~Element,~State,~See,~see,Bands.walk(~Block,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,right),hold(kept))),(shaped_bands(~Part,~Element,~buffer,start(context),right),Model.shaped_thread(~Part,~See,~Context,~step,context,right,see(hold(kept)))),rule(kept)),
      Equal.cong(Model.Content<Element> & See,Model.Content<Element> & See,made => shaped_walk_spread(~Part,~Element,~See,~Context,~buffer,~step,start(context),right,context,made),
        (packed_left,see(hold(kept))),(left_model,before),left_seen)))

def shaped_walk_observed(~Part: Data,~Block: Type,~pack: Part -> Block,~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> Block -> State -> Bands.Bands(~Element) & State,~start: Context -> U32,~buffer: Part -> Buffer.Model<Element>,
  ~step: Context -> @+first: U32 -> @+rows: U32 -> Part -> See -> See,
  +right: Model.Shaped<Part>,+context: Context,-before: See,-left_model: Model.Content<Element>,-state: State,-left: Bands.Bands(~Element),
  state_held: Held<State,Kept,hold,state>,left_observation: Model.Observation<Element,left>,
  left_seen: {walk_seen(~Element,~State,~See,~see,(left,state)) == (left_model,before) : Model.Content<Element> & See},
  rule: @+kept: Kept -> {walk_seen(~Element,~State,~See,~see,Bands.walk(~Block,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,right),hold(kept))) ==
    (shaped_bands(~Part,~Element,~buffer,start(context),right),Model.shaped_thread(~Part,~See,~Context,~step,context,right,see(hold(kept)))) : Model.Content<Element> & See}) ->
  {walk_seen(~Element,~State,~See,~see,shaped_walk_continued(~Block,~Element,~State,~Context,~leaf,Model.shaped_pack(~Part,~Block,~pack,right),context,(left,state))) ==
    (Model.Joined{left_model,shaped_bands(~Part,~Element,~buffer,start(context),right)},Model.shaped_thread(~Part,~See,~Context,~step,context,right,before)) : Model.Content<Element> & See}:
  Held{+kept,+state_equal} = state_held
  Model.Observation{+left_owner,+left_equal} = left_observation
  %state_equal : {walk_seen(~Element,~State,~See,~see,shaped_walk_continued(~Block,~Element,~State,~Context,~leaf,Model.shaped_pack(~Part,~Block,~pack,right),context,(left,_))) ==
    (Model.Joined{left_model,shaped_bands(~Part,~Element,~buffer,start(context),right)},Model.shaped_thread(~Part,~See,~Context,~step,context,right,before)) : Model.Content<Element> & See}
  %left_equal : {walk_seen(~Element,~State,~See,~see,shaped_walk_continued(~Block,~Element,~State,~Context,~leaf,Model.shaped_pack(~Part,~Block,~pack,right),context,(_,hold(kept)))) ==
    (Model.Joined{left_model,shaped_bands(~Part,~Element,~buffer,start(context),right)},Model.shaped_thread(~Part,~See,~Context,~step,context,right,before)) : Model.Content<Element> & See}
  shaped_walk_packed(~Part,~Block,~pack,~Element,~State,~Context,~Kept,~hold,~See,~see,~leaf,~start,~buffer,~step,right,context,before,left_model,kept,left_owner,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,left_owner),left,left_equal) : {walk_seen(~Element,~State,~See,~see,(_,hold(kept))) == (left_model,before) : Model.Content<Element> & See}
    %Equal.sym(State,hold(kept),state,state_equal) : {walk_seen(~Element,~State,~See,~see,(left,_)) == (left_model,before) : Model.Content<Element> & See}
    left_seen,rule)

def shaped_walk_continue(~Part: Data,~Block: Type,~pack: Part -> Block,~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> Block -> State -> Bands.Bands(~Element) & State,~start: Context -> U32,~buffer: Part -> Buffer.Model<Element>,
  ~step: Context -> @+first: U32 -> @+rows: U32 -> Part -> See -> See,
  +right: Model.Shaped<Part>,+context: Context,-before: See,-left_model: Model.Content<Element>,left: Bands.Bands(~Element) & State,
  left_seen: {walk_seen(~Element,~State,~See,~see,left) == (left_model,before) : Model.Content<Element> & See},
  rule: @+kept: Kept -> {walk_seen(~Element,~State,~See,~see,Bands.walk(~Block,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,right),hold(kept))) ==
    (shaped_bands(~Part,~Element,~buffer,start(context),right),Model.shaped_thread(~Part,~See,~Context,~step,context,right,see(hold(kept)))) : Model.Content<Element> & See}) ->
  {walk_seen(~Element,~State,~See,~see,shaped_walk_continued(~Block,~Element,~State,~Context,~leaf,Model.shaped_pack(~Part,~Block,~pack,right),context,left)) ==
    (Model.Joined{left_model,shaped_bands(~Part,~Element,~buffer,start(context),right)},Model.shaped_thread(~Part,~See,~Context,~step,context,right,before)) : Model.Content<Element> & See}:
  (bands,state) = left
  shaped_walk_observed(~Part,~Block,~pack,~Element,~State,~Context,~Kept,~hold,~See,~see,~leaf,~start,~buffer,~step,right,context,before,left_model,state,bands,observe(state),Model.observe(~Element,bands),left_seen,rule)

def shaped_walked(~Part: Data,~Block: Type,~pack: Part -> Block,~Element: Data,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,
  ~leaf: Context -> U32 -> U32 -> Block -> State -> Bands.Bands(~Element) & State,~start: Context -> U32,~buffer: Part -> Buffer.Model<Element>,
  ~step: Context -> @+first: U32 -> @+rows: U32 -> Part -> See -> See,
  ~rule: ShapedWalkRule(~Part,~Block,~pack,~Element,~State,~Context,~Kept,~hold,~See,~see,~leaf,~start,~buffer,~step),tree: Model.Shaped<Part>,+context: Context,+kept: Kept) ->
  {walk_seen(~Element,~State,~See,~see,Bands.walk(~Block,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,tree),hold(kept))) ==
    (shaped_bands(~Part,~Element,~buffer,start(context),tree),Model.shaped_thread(~Part,~See,~Context,~step,context,tree,see(hold(kept)))) : Model.Content<Element> & See}:
  match tree:
    case Model.Piece{+part,+first,+rows}: rule(context,first,rows,part,kept)
    case Model.Pieces{+left,+right}:
      shaped_walk_continue(~Part,~Block,~pack,~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~start,~buffer,~step,right,context,
        Model.shaped_thread(~Part,~See,~Context,~step,context,left,see(hold(kept))),shaped_bands(~Part,~Element,~buffer,start(context),left),
        Bands.walk(~Block,~Storage.Buffer<Element>,~State,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,left),hold(kept)),
        shaped_walked(~Part,~Block,~pack,~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~start,~buffer,~step,~rule,left,context,kept),
        next => shaped_walked(~Part,~Block,~pack,~Element,~State,~Context,~Kept,~hold,~observe,~See,~see,~leaf,~start,~buffer,~step,~rule,right,context,next))

# A walk of the code that leaves every band as it is and threads a buffer
# through them. An operation supplies only the equation of its leaf.
def ThreadRule(~Element: Data,~Context: Data,~run: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Bands.Bands(~Element) & Storage.Buffer<Element>,
  ~step: Context -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> Buffer.Content<Element> -> Buffer.Content<Element>) -> Type:
  @+context: Context -> @+first: U32 -> @+rows: U32 -> @+input: Buffer.Model<Element> -> @+output: Buffer.Model<Element> ->
  {walk_seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),run(context,first,rows,Buffer.pack(~Element,input),Buffer.pack(~Element,output))) ==
    (Model.RowBand{Buffer.content(~Element,input),first,rows},step(context,first,rows,Buffer.content(~Element,input),Buffer.content(~Element,output))) : Model.Content<Element> & Buffer.Content<Element>}

type Threading<-Element: Data,-Context: Data> is Data:
  Threading{context: Context,tree: Model.Model<Element>}

def Threaded(~Element: Data,~Context: Data,~run: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Bands.Bands(~Element) & Storage.Buffer<Element>,
  ~step: Context -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> Buffer.Content<Element> -> Buffer.Content<Element>,threading: Threading<Element,Context>,-output: Storage.Buffer<Element>) -> Type:
  Threading{+context,+tree} = threading
  {walk_seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),Bands.walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Context,~run,context,Model.pack(~Element,tree),output)) == (Model.content(~Element,tree),Model.thread(~Element,~Buffer.Content<Element>,~Context,~step,context,Model.content(~Element,tree),Buffer.contents(~Element,output))) : Model.Content<Element> & Buffer.Content<Element>}

def thread_spread(~Element: Data,~Context: Data,~step: Context -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> Buffer.Content<Element> -> Buffer.Content<Element>,+right: Model.Content<Element>,context: Context,made: Model.Content<Element> & Buffer.Content<Element>) -> Model.Content<Element> & Buffer.Content<Element>:
  (left,state) = made
  (Model.Joined{left,right},Model.thread(~Element,~Buffer.Content<Element>,~Context,~step,context,right,state))

def thread_continue_packed(~Element: Data,~Context: Data,~run: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Bands.Bands(~Element) & Storage.Buffer<Element>,
  ~step: Context -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> Buffer.Content<Element> -> Buffer.Content<Element>,+right: Model.Model<Element>,+context: Context,-before_tree: Model.Content<Element>,-before_state: Buffer.Content<Element>,+left: Model.Model<Element>,+state: Buffer.Model<Element>,
  made_seen: {walk_seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),(Model.pack(~Element,left),Buffer.pack(~Element,state))) == (before_tree,before_state) : Model.Content<Element> & Buffer.Content<Element>},rule: @output: Storage.Buffer<Element> -> {walk_seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),Bands.walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Context,~run,context,Model.pack(~Element,right),output)) == (Model.content(~Element,right),Model.thread(~Element,~Buffer.Content<Element>,~Context,~step,context,Model.content(~Element,right),Buffer.contents(~Element,output))) : Model.Content<Element> & Buffer.Content<Element>}) ->
  {walk_seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),walk_continued(~Element,~Storage.Buffer<Element>,~Context,~run,Model.pack(~Element,right),context,(Model.pack(~Element,left),Buffer.pack(~Element,state)))) == (Model.Joined{before_tree,Model.content(~Element,right)},Model.thread(~Element,~Buffer.Content<Element>,~Context,~step,context,Model.content(~Element,right),before_state)) : Model.Content<Element> & Buffer.Content<Element>}:
  Equal.trans(Model.Content<Element> & Buffer.Content<Element>,walk_seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),walk_join(~Element,~Storage.Buffer<Element>,Model.pack(~Element,left),Bands.walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Context,~run,context,Model.pack(~Element,right),Buffer.pack(~Element,state)))),
    walk_joined(~Element,~Buffer.Content<Element>,Model.contents(~Element,Model.pack(~Element,left)),walk_seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),Bands.walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Context,~run,context,Model.pack(~Element,right),Buffer.pack(~Element,state)))),
    (Model.Joined{before_tree,Model.content(~Element,right)},Model.thread(~Element,~Buffer.Content<Element>,~Context,~step,context,Model.content(~Element,right),before_state)),
    walk_join_seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),Model.pack(~Element,left),Bands.walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Context,~run,context,Model.pack(~Element,right),Buffer.pack(~Element,state))),
    Equal.trans(Model.Content<Element> & Buffer.Content<Element>,walk_joined(~Element,~Buffer.Content<Element>,Model.contents(~Element,Model.pack(~Element,left)),walk_seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),Bands.walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Context,~run,context,Model.pack(~Element,right),Buffer.pack(~Element,state)))),
      thread_spread(~Element,~Context,~step,Model.content(~Element,right),context,(Model.contents(~Element,Model.pack(~Element,left)),Buffer.contents(~Element,Buffer.pack(~Element,state)))),
      (Model.Joined{before_tree,Model.content(~Element,right)},Model.thread(~Element,~Buffer.Content<Element>,~Context,~step,context,Model.content(~Element,right),before_state)),
      Equal.cong(Model.Content<Element> & Buffer.Content<Element>,Model.Content<Element> & Buffer.Content<Element>,made => walk_joined(~Element,~Buffer.Content<Element>,Model.contents(~Element,Model.pack(~Element,left)),made),
        walk_seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),Bands.walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Context,~run,context,Model.pack(~Element,right),Buffer.pack(~Element,state))),
        (Model.content(~Element,right),Model.thread(~Element,~Buffer.Content<Element>,~Context,~step,context,Model.content(~Element,right),Buffer.contents(~Element,Buffer.pack(~Element,state)))),rule(Buffer.pack(~Element,state))),
      Equal.cong(Model.Content<Element> & Buffer.Content<Element>,Model.Content<Element> & Buffer.Content<Element>,made => thread_spread(~Element,~Context,~step,Model.content(~Element,right),context,made),
        (Model.contents(~Element,Model.pack(~Element,left)),Buffer.contents(~Element,Buffer.pack(~Element,state))),(before_tree,before_state),made_seen)))

def thread_continue_observed(~Element: Data,~Context: Data,~run: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Bands.Bands(~Element) & Storage.Buffer<Element>,
  ~step: Context -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> Buffer.Content<Element> -> Buffer.Content<Element>,+right: Model.Model<Element>,+context: Context,-before_tree: Model.Content<Element>,-before_state: Buffer.Content<Element>,-state: Storage.Buffer<Element>,-left: Bands.Bands(~Element),
  state_observation: Buffer.Observation<Element,state>,left_observation: Model.Observation<Element,left>,
  made_seen: {walk_seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),(left,state)) == (before_tree,before_state) : Model.Content<Element> & Buffer.Content<Element>},rule: @output: Storage.Buffer<Element> -> {walk_seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),Bands.walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Context,~run,context,Model.pack(~Element,right),output)) == (Model.content(~Element,right),Model.thread(~Element,~Buffer.Content<Element>,~Context,~step,context,Model.content(~Element,right),Buffer.contents(~Element,output))) : Model.Content<Element> & Buffer.Content<Element>}) ->
  {walk_seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),walk_continued(~Element,~Storage.Buffer<Element>,~Context,~run,Model.pack(~Element,right),context,(left,state))) == (Model.Joined{before_tree,Model.content(~Element,right)},Model.thread(~Element,~Buffer.Content<Element>,~Context,~step,context,Model.content(~Element,right),before_state)) : Model.Content<Element> & Buffer.Content<Element>}:
  Buffer.Observation{+state_owner,+state_equal} = state_observation
  Model.Observation{+left_owner,+left_equal} = left_observation
  %state_equal : {walk_seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),walk_continued(~Element,~Storage.Buffer<Element>,~Context,~run,Model.pack(~Element,right),context,(left,_))) == (Model.Joined{before_tree,Model.content(~Element,right)},Model.thread(~Element,~Buffer.Content<Element>,~Context,~step,context,Model.content(~Element,right),before_state)) : Model.Content<Element> & Buffer.Content<Element>}
  %left_equal : {walk_seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),walk_continued(~Element,~Storage.Buffer<Element>,~Context,~run,Model.pack(~Element,right),context,(_,Buffer.pack(~Element,state_owner)))) == (Model.Joined{before_tree,Model.content(~Element,right)},Model.thread(~Element,~Buffer.Content<Element>,~Context,~step,context,Model.content(~Element,right),before_state)) : Model.Content<Element> & Buffer.Content<Element>}
  thread_continue_packed(~Element,~Context,~run,~step,right,context,before_tree,before_state,left_owner,state_owner,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,left_owner),left,left_equal) : {walk_seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),(_,Buffer.pack(~Element,state_owner))) == (before_tree,before_state) : Model.Content<Element> & Buffer.Content<Element>}
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,state_owner),state,state_equal) : {walk_seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),(left,_)) == (before_tree,before_state) : Model.Content<Element> & Buffer.Content<Element>}
    made_seen,rule)

def thread_continue(~Element: Data,~Context: Data,~run: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Bands.Bands(~Element) & Storage.Buffer<Element>,
  ~step: Context -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> Buffer.Content<Element> -> Buffer.Content<Element>,+right: Model.Model<Element>,+context: Context,-before_tree: Model.Content<Element>,-before_state: Buffer.Content<Element>,made: Bands.Bands(~Element) & Storage.Buffer<Element>,
  made_seen: {walk_seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),made) == (before_tree,before_state) : Model.Content<Element> & Buffer.Content<Element>},rule: @output: Storage.Buffer<Element> -> {walk_seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),Bands.walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Context,~run,context,Model.pack(~Element,right),output)) == (Model.content(~Element,right),Model.thread(~Element,~Buffer.Content<Element>,~Context,~step,context,Model.content(~Element,right),Buffer.contents(~Element,output))) : Model.Content<Element> & Buffer.Content<Element>}) ->
  {walk_seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),walk_continued(~Element,~Storage.Buffer<Element>,~Context,~run,Model.pack(~Element,right),context,made)) == (Model.Joined{before_tree,Model.content(~Element,right)},Model.thread(~Element,~Buffer.Content<Element>,~Context,~step,context,Model.content(~Element,right),before_state)) : Model.Content<Element> & Buffer.Content<Element>}:
  (bands,state) = made
  thread_continue_observed(~Element,~Context,~run,~step,right,context,before_tree,before_state,state,bands,Buffer.observe(~Element,state),Model.observe(~Element,bands),made_seen,rule)

def threaded(~Element: Data,~Context: Data,~run: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Bands.Bands(~Element) & Storage.Buffer<Element>,
  ~step: Context -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> Buffer.Content<Element> -> Buffer.Content<Element>,~rule: ThreadRule(~Element,~Context,~run,~step),
  +context: Context,tree: Model.Model<Element>,output: Storage.Buffer<Element>) ->
  {walk_seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),Bands.walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Context,~run,context,Model.pack(~Element,tree),output)) == (Model.content(~Element,tree),Model.thread(~Element,~Buffer.Content<Element>,~Context,~step,context,Model.content(~Element,tree),Buffer.contents(~Element,output))) : Model.Content<Element> & Buffer.Content<Element>}:
  match tree:
    case Model.Leaf{+input,+first,+rows}:
      Buffer.lift(~Element,~Threading<Element,Context>,~Threaded(~Element,~Context,~run,~step),Threading{context,Model.Leaf{input,first,rows}},output,Buffer.observe(~Element,output),
        output =>
          +output = output
          %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,output)),Buffer.content(~Element,output),Buffer.packed_contents(~Element,output)) :
            {walk_seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),run(context,first,rows,Buffer.pack(~Element,input),Buffer.pack(~Element,output))) ==
              (Model.RowBand{Buffer.content(~Element,input),first,rows},step(context,first,rows,Buffer.content(~Element,input),_)) : Model.Content<Element> & Buffer.Content<Element>}
          rule(context,first,rows,input,output))
    case Model.Fork{+left,+right}:
      Buffer.lift(~Element,~Threading<Element,Context>,~Threaded(~Element,~Context,~run,~step),Threading{context,Model.Fork{left,right}},output,Buffer.observe(~Element,output),
        output =>
          +output = output
          thread_continue(~Element,~Context,~run,~step,right,context,Model.content(~Element,left),Model.thread(~Element,~Buffer.Content<Element>,~Context,~step,context,Model.content(~Element,left),Buffer.contents(~Element,Buffer.pack(~Element,output))),
            Bands.walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Context,~run,context,Model.pack(~Element,left),Buffer.pack(~Element,output)),threaded(~Element,~Context,~run,~step,~rule,context,left,Buffer.pack(~Element,output)),next => threaded(~Element,~Context,~run,~step,~rule,context,right,next)))

# A thread whose every step applies the writes of its band applies the writes
# of all the bands.
def WrittenRule(~Element: Data,~Context: Data,~step: Context -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> Buffer.Content<Element> -> Buffer.Content<Element>,
  ~writes: Context -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> List<&2,Writes.Assignment<Element>>) -> Type:
  @+context: Context -> @+first: U32 -> @+rows: U32 -> @+input: Buffer.Content<Element> -> @+tree: Tree.Tree<Element> -> @+length: Nat ->
  {step(context,first,rows,input,Buffer.Content{tree,length}) == Buffer.Content{Writes.pure(Element,writes(context,first,rows,input),tree),length} : Buffer.Content<Element>}

def thread_written(~Element: Data,~Context: Data,~step: Context -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> Buffer.Content<Element> -> Buffer.Content<Element>,
  ~writes: Context -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element> -> List<&2,Writes.Assignment<Element>>,~rule: WrittenRule(~Element,~Context,~step,~writes),
  +context: Context,content: Model.Content<Element>,+tree: Tree.Tree<Element>,+length: Nat) ->
  {Model.thread(~Element,~Buffer.Content<Element>,~Context,~step,context,content,Buffer.Content{tree,length}) == Buffer.Content{Writes.pure(Element,Model.collect(~Element,~Writes.Assignment<Element>,~Context,~writes,context,content),tree),length} : Buffer.Content<Element>}:
  match content:
    case Model.RowBand{+buffer,+first,+rows}: rule(context,first,rows,buffer,tree,length)
    case Model.Joined{+left,+right}:
      %Equal.sym(Buffer.Content<Element>,Model.thread(~Element,~Buffer.Content<Element>,~Context,~step,context,left,Buffer.Content{tree,length}),Buffer.Content{Writes.pure(Element,Model.collect(~Element,~Writes.Assignment<Element>,~Context,~writes,context,left),tree),length},
        thread_written(~Element,~Context,~step,~writes,~rule,context,left,tree,length)) :
        {Model.thread(~Element,~Buffer.Content<Element>,~Context,~step,context,right,_) == Buffer.Content{Writes.pure(Element,List.append(&2,Writes.Assignment<Element>,Model.collect(~Element,~Writes.Assignment<Element>,~Context,~writes,context,left),Model.collect(~Element,~Writes.Assignment<Element>,~Context,~writes,context,right)),tree),length} : Buffer.Content<Element>}
      %Equal.sym(Tree.Tree<Element>,Writes.pure(Element,List.append(&2,Writes.Assignment<Element>,Model.collect(~Element,~Writes.Assignment<Element>,~Context,~writes,context,left),Model.collect(~Element,~Writes.Assignment<Element>,~Context,~writes,context,right)),tree),
        Writes.pure(Element,Model.collect(~Element,~Writes.Assignment<Element>,~Context,~writes,context,right),Writes.pure(Element,Model.collect(~Element,~Writes.Assignment<Element>,~Context,~writes,context,left),tree)),
        Writes.concatenated_writes(Element,Model.collect(~Element,~Writes.Assignment<Element>,~Context,~writes,context,left),Model.collect(~Element,~Writes.Assignment<Element>,~Context,~writes,context,right),tree)) :
        {Model.thread(~Element,~Buffer.Content<Element>,~Context,~step,context,right,Buffer.Content{Writes.pure(Element,Model.collect(~Element,~Writes.Assignment<Element>,~Context,~writes,context,left),tree),length}) == Buffer.Content{_,length} : Buffer.Content<Element>}
      thread_written(~Element,~Context,~step,~writes,~rule,context,right,Writes.pure(Element,Model.collect(~Element,~Writes.Assignment<Element>,~Context,~writes,context,left),tree),length)

# ---- Trees of any blocks made from a plan ----

# What is seen of the state and of every block of a made tree.
def built_seen(~Block: Type,~State: Type,~See: Data,~see: State -> See,~Seen: Data,~view: Block -> Seen,made: State & Tree(~Block)) -> See & Model.Shaped<Seen>:
  (state,tree) = made
  (see(state),Model.shaped_seen(~Block,~Seen,~view,tree))

# A leaf that keeps what is seen of the state and makes one block.
def ShapedBuildRule(~Block: Type,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~See: Data,~see: State -> See,~Seen: Data,~view: Block -> Seen,
  ~leaf: Context -> U32 -> U32 -> State -> State & Tree(~Block),~piece: Context -> @+before: See -> @+first: U32 -> @+rows: U32 -> Seen) -> Type:
  @+context: Context -> @+first: U32 -> @+rows: U32 -> @+kept: Kept ->
  {built_seen(~Block,~State,~See,~see,~Seen,~view,leaf(context,first,rows,hold(kept))) == (see(hold(kept)),Model.Piece{piece(context,see(hold(kept)),first,rows),first,rows}) : See & Model.Shaped<Seen>}

def shaped_join(~Block: Type,~State: Type,left: Tree(~Block),right: State & Tree(~Block)) -> State & Tree(~Block):
  Storage.bind_pair(~State,~Tree(~Block),~(State & Tree(~Block)),right,state => right => (state,Partition.Fork{left,right}))

def shaped_continued(~Block: Type,~State: Type,~Context: Data,~leaf: Context -> U32 -> U32 -> State -> State & Tree(~Block),layout: Bands.Plan,+context: Context,left: State & Tree(~Block)) -> State & Tree(~Block):
  Storage.bind_pair(~State,~Tree(~Block),~(State & Tree(~Block)),left,state => left => shaped_join(~Block,~State,left,Bands.build(~Block,~State,~Context,~leaf,layout,context,state)))

def shaped_joined(~See: Data,~Seen: Data,left: Model.Shaped<Seen>,right: See & Model.Shaped<Seen>) -> See & Model.Shaped<Seen>:
  (state,right) = right
  (state,Model.Pieces{left,right})

def shaped_join_seen(~Block: Type,~State: Type,~See: Data,~see: State -> See,~Seen: Data,~view: Block -> Seen,-left: Tree(~Block),right: State & Tree(~Block)) ->
  {built_seen(~Block,~State,~See,~see,~Seen,~view,shaped_join(~Block,~State,left,right)) ==
    shaped_joined(~See,~Seen,Model.shaped_seen(~Block,~Seen,~view,left),built_seen(~Block,~State,~See,~see,~Seen,~view,right)) : See & Model.Shaped<Seen>}:
  (state,right) = right
  {==}

def shaped_spread(~See: Data,~Seen: Data,~Context: Data,~piece: Context -> @+before: See -> @+first: U32 -> @+rows: U32 -> Seen,layout: Bands.Plan,context: Context,made: See & Model.Shaped<Seen>) ->
  See & Model.Shaped<Seen>:
  (state,left) = made
  +state = state
  (state,Model.Pieces{left,Model.shaped_build(~Seen,~See,~piece(context),state,layout)})

def shaped_continue_held(~Block: Type,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~See: Data,~see: State -> See,~Seen: Data,~view: Block -> Seen,
  ~leaf: Context -> U32 -> U32 -> State -> State & Tree(~Block),~piece: Context -> @+before: See -> @+first: U32 -> @+rows: U32 -> Seen,
  +layout: Bands.Plan,+context: Context,-before: See,-left_seen: Model.Shaped<Seen>,-state: State,-tree: Tree(~Block),held: Held<State,Kept,hold,state>,
  left_equal: {built_seen(~Block,~State,~See,~see,~Seen,~view,(state,tree)) == (before,left_seen) : See & Model.Shaped<Seen>},
  rule: @+kept: Kept -> {built_seen(~Block,~State,~See,~see,~Seen,~view,Bands.build(~Block,~State,~Context,~leaf,layout,context,hold(kept))) ==
    (see(hold(kept)),Model.shaped_build(~Seen,~See,~piece(context),see(hold(kept)),layout)) : See & Model.Shaped<Seen>}) ->
  {built_seen(~Block,~State,~See,~see,~Seen,~view,shaped_continued(~Block,~State,~Context,~leaf,layout,context,(state,tree))) ==
    (before,Model.Pieces{left_seen,Model.shaped_build(~Seen,~See,~piece(context),before,layout)}) : See & Model.Shaped<Seen>}:
  Held{+kept,+equation} = held
  %equation : {built_seen(~Block,~State,~See,~see,~Seen,~view,shaped_continued(~Block,~State,~Context,~leaf,layout,context,(_,tree))) ==
    (before,Model.Pieces{left_seen,Model.shaped_build(~Seen,~See,~piece(context),before,layout)}) : See & Model.Shaped<Seen>}
  Equal.trans(See & Model.Shaped<Seen>,built_seen(~Block,~State,~See,~see,~Seen,~view,shaped_join(~Block,~State,tree,Bands.build(~Block,~State,~Context,~leaf,layout,context,hold(kept)))),
    shaped_joined(~See,~Seen,Model.shaped_seen(~Block,~Seen,~view,tree),built_seen(~Block,~State,~See,~see,~Seen,~view,Bands.build(~Block,~State,~Context,~leaf,layout,context,hold(kept)))),
    (before,Model.Pieces{left_seen,Model.shaped_build(~Seen,~See,~piece(context),before,layout)}),
    shaped_join_seen(~Block,~State,~See,~see,~Seen,~view,tree,Bands.build(~Block,~State,~Context,~leaf,layout,context,hold(kept))),
    Equal.trans(See & Model.Shaped<Seen>,shaped_joined(~See,~Seen,Model.shaped_seen(~Block,~Seen,~view,tree),built_seen(~Block,~State,~See,~see,~Seen,~view,Bands.build(~Block,~State,~Context,~leaf,layout,context,hold(kept)))),
      shaped_spread(~See,~Seen,~Context,~piece,layout,context,(see(hold(kept)),Model.shaped_seen(~Block,~Seen,~view,tree))),
      (before,Model.Pieces{left_seen,Model.shaped_build(~Seen,~See,~piece(context),before,layout)}),
      Equal.cong(See & Model.Shaped<Seen>,See & Model.Shaped<Seen>,result => shaped_joined(~See,~Seen,Model.shaped_seen(~Block,~Seen,~view,tree),result),
        built_seen(~Block,~State,~See,~see,~Seen,~view,Bands.build(~Block,~State,~Context,~leaf,layout,context,hold(kept))),(see(hold(kept)),Model.shaped_build(~Seen,~See,~piece(context),see(hold(kept)),layout)),rule(kept)),
      Equal.cong(See & Model.Shaped<Seen>,See & Model.Shaped<Seen>,result => shaped_spread(~See,~Seen,~Context,~piece,layout,context,result),
        (see(hold(kept)),Model.shaped_seen(~Block,~Seen,~view,tree)),(before,left_seen),
        %Equal.sym(State,hold(kept),state,equation) : {built_seen(~Block,~State,~See,~see,~Seen,~view,(_,tree)) == (before,left_seen) : See & Model.Shaped<Seen>} left_equal)))

def shaped_continue(~Block: Type,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,~Seen: Data,~view: Block -> Seen,
  ~leaf: Context -> U32 -> U32 -> State -> State & Tree(~Block),~piece: Context -> @+before: See -> @+first: U32 -> @+rows: U32 -> Seen,
  +layout: Bands.Plan,+context: Context,-before: See,-left_seen: Model.Shaped<Seen>,left: State & Tree(~Block),
  left_equal: {built_seen(~Block,~State,~See,~see,~Seen,~view,left) == (before,left_seen) : See & Model.Shaped<Seen>},
  rule: @+kept: Kept -> {built_seen(~Block,~State,~See,~see,~Seen,~view,Bands.build(~Block,~State,~Context,~leaf,layout,context,hold(kept))) ==
    (see(hold(kept)),Model.shaped_build(~Seen,~See,~piece(context),see(hold(kept)),layout)) : See & Model.Shaped<Seen>}) ->
  {built_seen(~Block,~State,~See,~see,~Seen,~view,shaped_continued(~Block,~State,~Context,~leaf,layout,context,left)) ==
    (before,Model.Pieces{left_seen,Model.shaped_build(~Seen,~See,~piece(context),before,layout)}) : See & Model.Shaped<Seen>}:
  (state,tree) = left
  shaped_continue_held(~Block,~State,~Context,~Kept,~hold,~See,~see,~Seen,~view,~leaf,~piece,layout,context,before,left_seen,state,tree,observe(state),left_equal,rule)

# A leaf that keeps what is seen of the state and makes one block makes, over
# the plan, the tree of those blocks: an operation proves the rule of its leaf.
def shaped_built(~Block: Type,~State: Type,~Context: Data,~Kept: Data,~hold: Kept -> State,~observe: @state: State -> Held<State,Kept,hold,state>,~See: Data,~see: State -> See,~Seen: Data,~view: Block -> Seen,
  ~leaf: Context -> U32 -> U32 -> State -> State & Tree(~Block),~piece: Context -> @+before: See -> @+first: U32 -> @+rows: U32 -> Seen,
  ~leaf_rule: ShapedBuildRule(~Block,~State,~Context,~Kept,~hold,~See,~see,~Seen,~view,~leaf,~piece),layout: Bands.Plan,+context: Context,+kept: Kept) ->
  {built_seen(~Block,~State,~See,~see,~Seen,~view,Bands.build(~Block,~State,~Context,~leaf,layout,context,hold(kept))) ==
    (see(hold(kept)),Model.shaped_build(~Seen,~See,~piece(context),see(hold(kept)),layout)) : See & Model.Shaped<Seen>}:
  match layout:
    case Bands.Rows{+first,+rows}: leaf_rule(context,first,rows,kept)
    case Bands.Split{+left,+right}:
      shaped_continue(~Block,~State,~Context,~Kept,~hold,~observe,~See,~see,~Seen,~view,~leaf,~piece,right,context,
        see(hold(kept)),Model.shaped_build(~Seen,~See,~piece(context),see(hold(kept)),left),
        Bands.build(~Block,~State,~Context,~leaf,left,context,hold(kept)),
        shaped_built(~Block,~State,~Context,~Kept,~hold,~observe,~See,~see,~Seen,~view,~leaf,~piece,~leaf_rule,left,context,kept),
        next => shaped_built(~Block,~State,~Context,~Kept,~hold,~observe,~See,~see,~Seen,~view,~leaf,~piece,~leaf_rule,right,context,next))

# A map keeps the rows of every piece, and the pieces built on a plan have its
# rows.
def shaped_map_plan(~From: Data,~To: Data,~each: From -> To,tree: Model.Shaped<From>) ->
  {Model.shaped_plan(~To,~piece_rows(~To),Model.shaped_map(~From,~To,~each,tree)) == Model.shaped_plan(~From,~piece_rows(~From),tree) : Bands.Plan}:
  match tree:
    case Model.Piece{part,+first,+rows}: {==}
    case Model.Pieces{+left,+right}:
      +mapped_left = Model.shaped_plan(~To,~piece_rows(~To),Model.shaped_map(~From,~To,~each,left))
      +mapped_right = Model.shaped_plan(~To,~piece_rows(~To),Model.shaped_map(~From,~To,~each,right))
      +plain_left = Model.shaped_plan(~From,~piece_rows(~From),left)
      +plain_right = Model.shaped_plan(~From,~piece_rows(~From),right)
      Equal.trans(Bands.Plan,Bands.Split{mapped_left,mapped_right},Bands.Split{plain_left,mapped_right},Bands.Split{plain_left,plain_right},
        Equal.cong(Bands.Plan,Bands.Plan,plan => Bands.Split{plan,mapped_right},mapped_left,plain_left,shaped_map_plan(~From,~To,~each,left)),
        Equal.cong(Bands.Plan,Bands.Plan,plan => Bands.Split{plain_left,plan},mapped_right,plain_right,shaped_map_plan(~From,~To,~each,right)))

def shaped_valued_plan(~Part: Data,~Value: Data,~Context: Data,~value: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 -> Value,+context: Context,tree: Model.Shaped<Part>) ->
  {Model.shaped_plan(~Value,~piece_rows(~Value),Model.shaped_valued(~Part,~Value,~Context,~value,context,tree)) == Model.shaped_plan(~Part,~piece_rows(~Part),tree) : Bands.Plan}:
  match tree:
    case Model.Piece{+part,+first,+rows}: {==}
    case Model.Pieces{+left,+right}:
      +valued_left = Model.shaped_plan(~Value,~piece_rows(~Value),Model.shaped_valued(~Part,~Value,~Context,~value,context,left))
      +valued_right = Model.shaped_plan(~Value,~piece_rows(~Value),Model.shaped_valued(~Part,~Value,~Context,~value,context,right))
      +plain_left = Model.shaped_plan(~Part,~piece_rows(~Part),left)
      +plain_right = Model.shaped_plan(~Part,~piece_rows(~Part),right)
      Equal.trans(Bands.Plan,Bands.Split{valued_left,valued_right},Bands.Split{plain_left,valued_right},Bands.Split{plain_left,plain_right},
        Equal.cong(Bands.Plan,Bands.Plan,plan => Bands.Split{plan,valued_right},valued_left,plain_left,shaped_valued_plan(~Part,~Value,~Context,~value,context,left)),
        Equal.cong(Bands.Plan,Bands.Plan,plan => Bands.Split{plain_left,plan},valued_right,plain_right,shaped_valued_plan(~Part,~Value,~Context,~value,context,right)))

def shaped_build_plan(~Part: Data,~See: Data,~Context: Data,~piece: Context -> @+before: See -> @+first: U32 -> @+rows: U32 -> Part,+context: Context,+before: See,layout: Bands.Plan) ->
  {Model.shaped_plan(~Part,~piece_rows(~Part),Model.shaped_build(~Part,~See,~piece(context),before,layout)) == layout : Bands.Plan}:
  match layout:
    case Bands.Rows{+first,+rows}: {==}
    case Bands.Split{+left,+right}:
      %Equal.sym(Bands.Plan,Model.shaped_plan(~Part,~piece_rows(~Part),Model.shaped_build(~Part,~See,~piece(context),before,left)),left,
          shaped_build_plan(~Part,~See,~Context,~piece,context,before,left)) :
        {Bands.Split{_,Model.shaped_plan(~Part,~piece_rows(~Part),Model.shaped_build(~Part,~See,~piece(context),before,right))} == Bands.Split{left,right} : Bands.Plan}
      %Equal.sym(Bands.Plan,Model.shaped_plan(~Part,~piece_rows(~Part),Model.shaped_build(~Part,~See,~piece(context),before,right)),right,
          shaped_build_plan(~Part,~See,~Context,~piece,context,before,right)) :
        {Bands.Split{left,_} == Bands.Split{left,right} : Bands.Plan}
      {==}

# A leaf that makes the pack of one value of its piece makes the pack of the
# valued pieces.
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
  coverage: BandPlan.Coverage(Model.shaped_plan(~Part,~piece_rows(~Part),tree),first,rows)) ->
  {List.length(&2,Element,shaped_list(~Part,~Element,~Context,~list,context,tree)) == Nat.mul(rows,width(context)) : Nat}:
  match tree:
    case Model.Piece{+part,+native_first,+native_rows}:
      BandPlan.LeafCoverage{first_equal,+rows_equal} = coverage
      %rows_equal : {List.length(&2,Element,list(context,part,native_first,native_rows)) == Nat.mul(_,width(context)) : Nat}
      length_rule(context,part,native_first,native_rows)
    case Model.Pieces{+left,+right}:
      BandPlan.ForkCoverage{+left_rows,+right_rows,+rows_equal,+left_coverage,+right_coverage} = coverage
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
  +context: Context,tree: Model.Shaped<Part>,+first: Nat,+rows: Nat,coverage: BandPlan.Coverage(Model.shaped_plan(~Part,~piece_rows(~Part),tree),first,rows),
  +before: Tree.Tree<Element>,+length: Nat) ->
  Written(~Part,~Element,~Context,~step,~list,~width,context,tree,first,before,length):
  match tree:
    case Model.Piece{+part,+native_first,+native_rows}:
      BandPlan.LeafCoverage{+first_equal,rows_equal} = coverage
      %first_equal :
        {step(context,native_first,native_rows,part,Buffer.Content{before,length}) ==
          Buffer.Content{Tree.write_sequence(~Element,list(context,part,native_first,native_rows),U32.from_nat(Nat.mul(_,width(context))),before),length} : Buffer.Content<Element>}
      rule(context,native_first,native_rows,part,before,length)
    case Model.Pieces{+left,+right}:
      BandPlan.ForkCoverage{+left_rows,+right_rows,rows_equal,+left_coverage,+right_coverage} = coverage
      written_joined(~Part,~Element,~Context,~step,~list,~width,context,left,right,first,left_rows,before,length,
        shaped_list_length(~Part,~Element,~Context,~list,~width,~length_rule,context,left,first,left_rows,left_coverage),
        shaped_written(~Part,~Element,~Context,~step,~list,~width,~rule,~length_rule,context,left,first,left_rows,left_coverage,before,length),
        shaped_written(~Part,~Element,~Context,~step,~list,~width,~rule,~length_rule,context,right,Nat.add(first,left_rows),right_rows,right_coverage,
          Tree.write_sequence(~Element,shaped_list(~Part,~Element,~Context,~list,context,left),U32.from_nat(Nat.mul(first,width(context))),before),length))

# ---- Elements of the lists of pieces ----

# An element of the lists of pieces that cover rows [first, first + rows) is
# the element of its piece: a value of its global position holds of every
# element when it holds of the element of every piece.
def ListRule(~Part: Data,~Element: Data,~Context: Data,~list: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 -> List<&2,Element>,~width: Context -> Nat,
  ~Scope: Data,~context: @+scope: Scope -> Context,~fill: @+scope: Scope -> Element,~limit: @+scope: Scope -> Nat,
  ~Fact: @+scope: Scope -> @+part: Part -> @+first: U32 -> @+rows: U32 -> Data,~value: @+scope: Scope -> @+index: Nat -> Element) -> Type:
  @+scope: Scope -> @+part: Part -> @+first: U32 -> @+rows: U32 -> @+fact: Fact(scope,part,first,rows) ->
    @+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),limit(scope)) == True{} : Bool} ->
    @+local: Nat -> @+inside: {Nat.is_lt(local,Nat.mul(U32.to_nat(rows),width(context(scope)))) == True{} : Bool} ->
    {ListRead.at(~Element,fill(scope),local,list(context(scope),part,first,rows)) == value(scope,Nat.add(Nat.mul(U32.to_nat(first),width(context(scope))),local)) : Element}

def ListAt(~Part: Data,~Element: Data,~Context: Data,~list: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 -> List<&2,Element>,~width: Context -> Nat,
  ~Scope: Data,~context: @+scope: Scope -> Context,~fill: @+scope: Scope -> Element,~value: @+scope: Scope -> @+index: Nat -> Element,
  +scope: Scope,tree: Model.Shaped<Part>,+first: Nat,+index: Nat) -> Data:
  {ListRead.at(~Element,fill(scope),index,shaped_list(~Part,~Element,~Context,~list,context(scope),tree)) == value(scope,Nat.add(Nat.mul(first,width(context(scope))),index)) : Element}

def list_at_branch(~Part: Data,~Element: Data,~Context: Data,~list: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 -> List<&2,Element>,~width: Context -> Nat,
  ~Scope: Data,~context: @+scope: Scope -> Context,~fill: @+scope: Scope -> Element,~value: @+scope: Scope -> @+index: Nat -> Element,
  +scope: Scope,+left: Model.Shaped<Part>,+right: Model.Shaped<Part>,+first: Nat,+left_rows: Nat,+right_rows: Nat,+index: Nat,choose_left: Bool,
  +left_length: {List.length(&2,Element,shaped_list(~Part,~Element,~Context,~list,context(scope),left)) == Nat.mul(left_rows,width(context(scope))) : Nat},
  +choice: {Nat.is_lt(index,Nat.mul(left_rows,width(context(scope)))) == choose_left : Bool},
  +inside: {Nat.is_lt(index,Nat.add(Nat.mul(left_rows,width(context(scope))),Nat.mul(right_rows,width(context(scope))))) == True{} : Bool},
  left_case: {Nat.is_lt(index,Nat.mul(left_rows,width(context(scope)))) == True{} : Bool} ->
    ListAt(~Part,~Element,~Context,~list,~width,~Scope,~context,~fill,~value,scope,left,first,index),
  right_case: @+tail: Nat -> {Nat.is_lt(tail,Nat.mul(right_rows,width(context(scope)))) == True{} : Bool} ->
    ListAt(~Part,~Element,~Context,~list,~width,~Scope,~context,~fill,~value,scope,right,Nat.add(first,left_rows),tail)) ->
  ListAt(~Part,~Element,~Context,~list,~width,~Scope,~context,~fill,~value,scope,Model.Pieces{left,right},first,index):
  match choose_left:
    case True{}:
      +width_value = width(context(scope))
      +left_list = shaped_list(~Part,~Element,~Context,~list,context(scope),left)
      +right_list = shaped_list(~Part,~Element,~Context,~list,context(scope),right)
      +left_count = Nat.mul(left_rows,width_value)
      Equal.trans(Element,ListRead.at(~Element,fill(scope),index,List.append(&2,Element,left_list,right_list)),ListRead.at(~Element,fill(scope),index,left_list),
        value(scope,Nat.add(Nat.mul(first,width_value),index)),
        ListRead.appended_before(~Element,fill(scope),left_list,right_list,index,
          %Equal.sym(Nat,List.length(&2,Element,left_list),left_count,left_length) : {Nat.is_lt(index,_) == True{} : Bool} choice),
        left_case(choice))
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
      %restored :
        {ListRead.at(~Element,fill(scope),_,List.append(&2,Element,left_list,right_list)) == value(scope,Nat.add(Nat.mul(first,width_value),index)) : Element}
      %left_length : {ListRead.at(~Element,fill(scope),Nat.add(_,tail),List.append(&2,Element,left_list,right_list)) == value(scope,Nat.add(Nat.mul(first,width_value),index)) : Element}
      %global : {ListRead.at(~Element,fill(scope),Nat.add(List.length(&2,Element,left_list),tail),List.append(&2,Element,left_list,right_list)) == value(scope,_) : Element}
      Equal.trans(Element,ListRead.at(~Element,fill(scope),Nat.add(List.length(&2,Element,left_list),tail),List.append(&2,Element,left_list,right_list)),
        ListRead.at(~Element,fill(scope),tail,right_list),value(scope,Nat.add(Nat.mul(Nat.add(first,left_rows),width_value),tail)),
        ListRead.appended_after(~Element,fill(scope),left_list,right_list,tail),
        right_case(tail,Order.sub_lt(index,left_count,Nat.mul(right_rows,width_value),above,inside)))

def shaped_list_at(~Part: Data,~Element: Data,~Context: Data,~list: Context -> @+part: Part -> @+first: U32 -> @+rows: U32 -> List<&2,Element>,~width: Context -> Nat,
  ~length_rule: LengthRule(~Part,~Element,~Context,~list,~width),
  ~Scope: Data,~context: @+scope: Scope -> Context,~fill: @+scope: Scope -> Element,~limit: @+scope: Scope -> Nat,
  ~Fact: @+scope: Scope -> @+part: Part -> @+first: U32 -> @+rows: U32 -> Data,~value: @+scope: Scope -> @+index: Nat -> Element,
  ~rule: ListRule(~Part,~Element,~Context,~list,~width,~Scope,~context,~fill,~limit,~Fact,~value),
  +scope: Scope,tree: Model.Shaped<Part>,+first: Nat,+rows: Nat,coverage: BandPlan.Coverage(Model.shaped_plan(~Part,~piece_rows(~Part),tree),first,rows),
  facts: Model.ShapedAll(~Part,~Scope,~Fact,scope,tree),+room: {Nat.is_le(Nat.add(first,rows),limit(scope)) == True{} : Bool},
  +index: Nat,+inside: {Nat.is_lt(index,Nat.mul(rows,width(context(scope)))) == True{} : Bool}) ->
  ListAt(~Part,~Element,~Context,~list,~width,~Scope,~context,~fill,~value,scope,tree,first,index):
  match tree:
    case Model.Piece{+part,+native_first,+native_rows}:
      BandPlan.LeafCoverage{+first_equal,+rows_equal} = coverage
      %first_equal :
        {ListRead.at(~Element,fill(scope),index,list(context(scope),part,native_first,native_rows)) == value(scope,Nat.add(Nat.mul(_,width(context(scope))),index)) : Element}
      rule(scope,part,native_first,native_rows,facts,
        %Equal.sym(Nat,U32.to_nat(native_first),first,first_equal) : {Nat.is_le(Nat.add(_,U32.to_nat(native_rows)),limit(scope)) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(native_rows),rows,rows_equal) : {Nat.is_le(Nat.add(first,_),limit(scope)) == True{} : Bool}
        room,
        index,
        %Equal.sym(Nat,U32.to_nat(native_rows),rows,rows_equal) : {Nat.is_lt(index,Nat.mul(_,width(context(scope)))) == True{} : Bool}
        inside)
    case Model.Pieces{+left,+right}:
      BandPlan.ForkCoverage{+left_rows,+right_rows,+rows_equal,+left_coverage,+right_coverage} = coverage
      Shape.Both{+left_facts,+right_facts} = facts
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
      list_at_branch(~Part,~Element,~Context,~list,~width,~Scope,~context,~fill,~value,scope,left,right,first,left_rows,right_rows,index,
        Nat.is_lt(index,Nat.mul(left_rows,width_value)),
        shaped_list_length(~Part,~Element,~Context,~list,~width,~length_rule,context(scope),left,first,left_rows,left_coverage),{==},
        %RowMajor.distrib(left_rows,right_rows,width_value) : {Nat.is_lt(index,_) == True{} : Bool}
        %rows_equal : {Nat.is_lt(index,Nat.mul(_,width_value)) == True{} : Bool}
        inside,
        in_left => shaped_list_at(~Part,~Element,~Context,~list,~width,~length_rule,~Scope,~context,~fill,~limit,~Fact,~value,~rule,scope,left,first,left_rows,left_coverage,
          left_facts,left_room,index,in_left),
        tail => in_right => shaped_list_at(~Part,~Element,~Context,~list,~width,~length_rule,~Scope,~context,~fill,~limit,~Fact,~value,~rule,scope,right,Nat.add(first,left_rows),right_rows,
          right_coverage,right_facts,right_room,tail,in_right))

# ---- Pieces built along a plan ----

def piece_first(~Part: Data,tree: Model.Shaped<Part>) -> U32:
  match tree:
    case Model.Piece{part,first,rows}: first
    case Model.Pieces{left,right}: 0

def piece_count(~Part: Data,tree: Model.Shaped<Part>) -> U32:
  match tree:
    case Model.Piece{part,first,rows}: rows
    case Model.Pieces{left,right}: 0

def piece_part(~Part: Data,tree: Model.Shaped<Part>,fallback: Part) -> Part:
  match tree:
    case Model.Piece{part,first,rows}: part
    case Model.Pieces{left,right}: fallback

def is_pieces(~Part: Data,tree: Model.Shaped<Part>) -> Bool:
  match tree:
    case Model.Piece{part,first,rows}: False{}
    case Model.Pieces{left,right}: True{}

def left_pieces(~Part: Data,tree: Model.Shaped<Part>) -> Model.Shaped<Part>:
  match tree:
    case Model.Piece{part,first,rows}: Model.Piece{part,first,rows}
    case Model.Pieces{left,right}: left

def right_pieces(~Part: Data,tree: Model.Shaped<Part>) -> Model.Shaped<Part>:
  match tree:
    case Model.Piece{part,first,rows}: Model.Piece{part,first,rows}
    case Model.Pieces{left,right}: right

# What is seen of a part is the piece built on its rows.
type BuiltScope<-Context: Data,-See: Data> is Data:
  BuiltScope{context: Context,before: See}

def built_piece(~Part: Data,~Seen: Data,~See: Data,~Context: Data,~view: Part -> Seen,~piece: Context -> @+before: See -> @+first: U32 -> @+rows: U32 -> Seen,
  +scope: BuiltScope<Context,See>,+part: Part,+first: U32,+rows: U32) -> Data:
  BuiltScope{context,before} = scope
  {view(part) == piece(context,before,first,rows) : Seen}

def built_piece_leaf(~Part: Data,~Seen: Data,~See: Data,~Context: Data,~view: Part -> Seen,~piece: Context -> @+before: See -> @+first: U32 -> @+rows: U32 -> Seen,
  +context: Context,+before: See,+part: Part,+first: U32,+rows: U32,+native_first: U32,+native_rows: U32,
  +equal: {Model.Piece{view(part),first,rows} == Model.Piece{piece(context,before,native_first,native_rows),native_first,native_rows} : Model.Shaped<Seen>}) ->
  built_piece(~Part,~Seen,~See,~Context,~view,~piece,BuiltScope{context,before},part,first,rows):
  +first_equal = Equal.cong(Model.Shaped<Seen>,U32,tree => piece_first(~Seen,tree),Model.Piece{view(part),first,rows},
    Model.Piece{piece(context,before,native_first,native_rows),native_first,native_rows},equal)
  +rows_equal = Equal.cong(Model.Shaped<Seen>,U32,tree => piece_count(~Seen,tree),Model.Piece{view(part),first,rows},
    Model.Piece{piece(context,before,native_first,native_rows),native_first,native_rows},equal)
  %Equal.sym(U32,first,native_first,first_equal) : {view(part) == piece(context,before,_,rows) : Seen}
  %Equal.sym(U32,rows,native_rows,rows_equal) : {view(part) == piece(context,before,native_first,_) : Seen}
  Equal.cong(Model.Shaped<Seen>,Seen,tree => piece_part(~Seen,tree,view(part)),Model.Piece{view(part),first,rows},
    Model.Piece{piece(context,before,native_first,native_rows),native_first,native_rows},equal)

# Parts whose views are the pieces built on a plan: the view of every part is
# the piece built on its rows.
def shaped_built_all(~Part: Data,~Seen: Data,~See: Data,~Context: Data,~view: Part -> Seen,~piece: Context -> @+before: See -> @+first: U32 -> @+rows: U32 -> Seen,
  +context: Context,+before: See,tree: Model.Shaped<Part>,layout: Bands.Plan,
  +equal: {Model.shaped_map(~Part,~Seen,~view,tree) == Model.shaped_build(~Seen,~See,~piece(context),before,layout) : Model.Shaped<Seen>}) ->
  Model.ShapedAll(~Part,~BuiltScope<Context,See>,~built_piece(~Part,~Seen,~See,~Context,~view,~piece),BuiltScope{context,before},tree):
  match tree layout:
    case Model.Piece{+part,+first,+rows} Bands.Rows{+native_first,+native_rows}:
      built_piece_leaf(~Part,~Seen,~See,~Context,~view,~piece,context,before,part,first,rows,native_first,native_rows,equal)
    case Model.Piece{+part,+first,+rows} Bands.Split{+left,+right}:
      Empty.absurd(built_piece(~Part,~Seen,~See,~Context,~view,~piece,BuiltScope{context,before},part,first,rows),
        WordEncoding.false_true(Equal.cong(Model.Shaped<Seen>,Bool,tree => is_pieces(~Seen,tree),Model.Piece{view(part),first,rows},
          Model.Pieces{Model.shaped_build(~Seen,~See,~piece(context),before,left),Model.shaped_build(~Seen,~See,~piece(context),before,right)},equal)))
    case Model.Pieces{+left,+right} Bands.Rows{+native_first,+native_rows}:
      Empty.absurd(Shape.Both<Model.ShapedAll(~Part,~BuiltScope<Context,See>,~built_piece(~Part,~Seen,~See,~Context,~view,~piece),BuiltScope{context,before},left),
          Model.ShapedAll(~Part,~BuiltScope<Context,See>,~built_piece(~Part,~Seen,~See,~Context,~view,~piece),BuiltScope{context,before},right)>,
        WordEncoding.false_true(Equal.cong(Model.Shaped<Seen>,Bool,tree => is_pieces(~Seen,tree),Model.Piece{piece(context,before,native_first,native_rows),native_first,native_rows},
          Model.Pieces{Model.shaped_map(~Part,~Seen,~view,left),Model.shaped_map(~Part,~Seen,~view,right)},
          Equal.sym(Model.Shaped<Seen>,Model.Pieces{Model.shaped_map(~Part,~Seen,~view,left),Model.shaped_map(~Part,~Seen,~view,right)},
            Model.Piece{piece(context,before,native_first,native_rows),native_first,native_rows},equal))))
    case Model.Pieces{+left,+right} Bands.Split{+left_layout,+right_layout}:
      +mapped = {Model.Pieces{Model.shaped_map(~Part,~Seen,~view,left),Model.shaped_map(~Part,~Seen,~view,right)} : Model.Shaped<Seen>}
      Shape.Both{
        shaped_built_all(~Part,~Seen,~See,~Context,~view,~piece,context,before,left,left_layout,
          Equal.cong(Model.Shaped<Seen>,Model.Shaped<Seen>,tree => left_pieces(~Seen,tree),mapped,
            Model.Pieces{Model.shaped_build(~Seen,~See,~piece(context),before,left_layout),Model.shaped_build(~Seen,~See,~piece(context),before,right_layout)},equal)),
        shaped_built_all(~Part,~Seen,~See,~Context,~view,~piece,context,before,right,right_layout,
          Equal.cong(Model.Shaped<Seen>,Model.Shaped<Seen>,tree => right_pieces(~Seen,tree),mapped,
            Model.Pieces{Model.shaped_build(~Seen,~See,~piece(context),before,left_layout),Model.shaped_build(~Seen,~See,~piece(context),before,right_layout)},equal))}
