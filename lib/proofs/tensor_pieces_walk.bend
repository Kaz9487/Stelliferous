# The walk over ranges, by structure. A parallel split computes what the
# sequential recursion computes, a walk over represented states is the
# represented model walk, and ranges halved at whole units start where their
# cells start.
import Base
import ../tensor_pieces.bend as Pieces
import ../storage_buffer.bend as Storage
import ../traversal_partition.bend as Partition
import ./traversal_partition_refinement.bend as PartitionLaws
import ../kernel_gemm.bend as KernelGemm
import ./tensor_pieces_model.bend as Model
import ./storage_shape.bend as Shape

def sequential(~State: Type,~Output: Type,~Details: Data,~Context: Data,~Schedule: Data,
  ~parallel: Schedule -> Pieces.Ranges -> Bool,
  ~divide: Context -> Pieces.Ranges -> Pieces.Ranges -> State -> Pieces.Division<State>,
  ~leaf: Context -> U32 -> U32 -> State -> Partition.Tree<&1,Output,Details>,
  ~keep: Context -> Pieces.Ranges -> State -> Partition.Tree<&1,Output,Details>,
  +left_plan: Pieces.Ranges,+right_plan: Pieces.Ranges,+schedule: Schedule,+context: Context,division: Pieces.Division<State>) -> Partition.Tree<&1,Output,Details>:
  Pieces.divided(~State,~Partition.Tree<&1,Output,Details>,division,whole => keep(context,Pieces.Split{left_plan,right_plan},whole),
    left_state => right_state =>
      Partition.Fork{Pieces.each_part(~State,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,left_plan,parallel(schedule,left_plan),schedule,context,left_state),
        Pieces.each_part(~State,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,right_plan,parallel(schedule,right_plan),schedule,context,right_state)})

def fork(~State: Type,~Output: Type,~Details: Data,~Context: Data,~Schedule: Data,
  ~parallel: Schedule -> Pieces.Ranges -> Bool,
  ~divide: Context -> Pieces.Ranges -> Pieces.Ranges -> State -> Pieces.Division<State>,
  ~leaf: Context -> U32 -> U32 -> State -> Partition.Tree<&1,Output,Details>,
  ~keep: Context -> Pieces.Ranges -> State -> Partition.Tree<&1,Output,Details>,
  +left_plan: Pieces.Ranges,+right_plan: Pieces.Ranges,+mode: Bool,+schedule: Schedule,+context: Context,-state: State) ->
  {Pieces.each_part(~State,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,Pieces.Split{left_plan,right_plan},mode,schedule,context,state) ==
    sequential(~State,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,left_plan,right_plan,schedule,context,divide(context,left_plan,right_plan,state)) :
    Partition.Tree<&1,Output,Details>}:
  match left_plan right_plan mode:
    case Pieces.Range{left_first,left_rows} Pieces.Range{right_first,right_rows} False{}: {==}
    case Pieces.Range{left_first,left_rows} Pieces.Range{right_first,right_rows} True{}: {==}
    case Pieces.Range{left_first,left_rows} Pieces.Split{right_left,right_right} False{}: {==}
    case Pieces.Range{left_first,left_rows} Pieces.Split{right_left,right_right} True{}: {==}
    case Pieces.Split{left_left,left_right} Pieces.Range{right_first,right_rows} False{}: {==}
    case Pieces.Split{left_left,left_right} Pieces.Range{right_first,right_rows} True{}: {==}
    case Pieces.Split{left_left,left_right} Pieces.Split{right_left,right_right} False{}: {==}
    case Pieces.Split{left_left,left_right} Pieces.Split{right_left,right_right} True{}: {==}

def model(~Model: Data,~Value: Data,~Details: Data,~Context: Data,
  ~split: Context -> Pieces.Ranges -> Pieces.Ranges -> Model -> PartitionLaws.Split<Model>,
  ~leaf: Context -> U32 -> U32 -> Model -> Partition.Tree<&2,Value,Details>,
  layout: Pieces.Ranges,+context: Context,+state: Model) -> Partition.Tree<&2,Value,Details>:
  match layout:
    case Pieces.Range{first,rows}: leaf(context,first,rows,state)
    case Pieces.Split{+left,+right}:
      +halves = split(context,left,right,state)
      Partition.Fork{model(~Model,~Value,~Details,~Context,~split,~leaf,left,context,PartitionLaws.first(~Model,halves)),
        model(~Model,~Value,~Details,~Context,~split,~leaf,right,context,PartitionLaws.second(~Model,halves))}

def walk_matches(~Native: Type,~Model: Data,~Output: Type,~Value: Data,~Details: Data,~Context: Data,~Schedule: Data,
  ~parallel: Schedule -> Pieces.Ranges -> Bool,
  ~divide: Context -> Pieces.Ranges -> Pieces.Ranges -> Native -> Pieces.Division<Native>,
  ~leaf: Context -> U32 -> U32 -> Native -> Partition.Tree<&1,Output,Details>,
  ~keep: Context -> Pieces.Ranges -> Native -> Partition.Tree<&1,Output,Details>,
  ~represent: Model -> Native,
  ~pack: Value -> Output,
  ~split: Context -> Pieces.Ranges -> Pieces.Ranges -> Model -> PartitionLaws.Split<Model>,
  ~model_leaf: Context -> U32 -> U32 -> Model -> Partition.Tree<&2,Value,Details>,
  ~divide_matches: @+context: Context -> @+left: Pieces.Ranges -> @+right: Pieces.Ranges -> @+state: Model ->
    {divide(context,left,right,represent(state)) ==
      Pieces.Divided{represent(PartitionLaws.first(~Model,split(context,left,right,state))),represent(PartitionLaws.second(~Model,split(context,left,right,state)))} :
      Pieces.Division<Native>},
  ~leaf_matches: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+state: Model ->
    {leaf(context,first,rows,represent(state)) == PartitionLaws.carry(~&2,~&1,~Value,~Output,~Details,~pack,model_leaf(context,first,rows,state)) :
      Partition.Tree<&1,Output,Details>},
  layout: Pieces.Ranges,+mode: Bool,+schedule: Schedule,+context: Context,+state: Model) ->
  {Pieces.each_part(~Native,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,layout,mode,schedule,context,represent(state)) ==
    PartitionLaws.carry(~&2,~&1,~Value,~Output,~Details,~pack,model(~Model,~Value,~Details,~Context,~split,~model_leaf,layout,context,state)) :
    Partition.Tree<&1,Output,Details>}:
  match layout:
    case Pieces.Range{+first,+rows}: leaf_matches(context,first,rows,state)
    case Pieces.Split{+left,+right}:
      +halves = split(context,left,right,state)
      +left_state = PartitionLaws.first(~Model,halves)
      +right_state = PartitionLaws.second(~Model,halves)
      %Equal.sym(Partition.Tree<&1,Output,Details>,
        Pieces.each_part(~Native,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,Pieces.Split{left,right},mode,schedule,context,represent(state)),
        sequential(~Native,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,left,right,schedule,context,divide(context,left,right,represent(state))),
        fork(~Native,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,left,right,mode,schedule,context,represent(state))) :
        {_ == PartitionLaws.carry(~&2,~&1,~Value,~Output,~Details,~pack,model(~Model,~Value,~Details,~Context,~split,~model_leaf,Pieces.Split{left,right},context,state)) :
          Partition.Tree<&1,Output,Details>}
      %Equal.sym(Pieces.Division<Native>,divide(context,left,right,represent(state)),Pieces.Divided{represent(left_state),represent(right_state)},
        divide_matches(context,left,right,state)) :
        {sequential(~Native,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,left,right,schedule,context,_) ==
          PartitionLaws.carry(~&2,~&1,~Value,~Output,~Details,~pack,model(~Model,~Value,~Details,~Context,~split,~model_leaf,Pieces.Split{left,right},context,state)) :
          Partition.Tree<&1,Output,Details>}
      %Equal.sym(Partition.Tree<&1,Output,Details>,
        Pieces.each_part(~Native,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,left,parallel(schedule,left),schedule,context,represent(left_state)),
        PartitionLaws.carry(~&2,~&1,~Value,~Output,~Details,~pack,model(~Model,~Value,~Details,~Context,~split,~model_leaf,left,context,left_state)),
        walk_matches(~Native,~Model,~Output,~Value,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,~represent,~pack,~split,~model_leaf,~divide_matches,~leaf_matches,
          left,parallel(schedule,left),schedule,context,left_state)) :
        {Partition.Fork{_,Pieces.each_part(~Native,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,right,parallel(schedule,right),schedule,context,represent(right_state))} ==
          PartitionLaws.carry(~&2,~&1,~Value,~Output,~Details,~pack,model(~Model,~Value,~Details,~Context,~split,~model_leaf,Pieces.Split{left,right},context,state)) :
          Partition.Tree<&1,Output,Details>}
      %Equal.sym(Partition.Tree<&1,Output,Details>,
        Pieces.each_part(~Native,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,right,parallel(schedule,right),schedule,context,represent(right_state)),
        PartitionLaws.carry(~&2,~&1,~Value,~Output,~Details,~pack,model(~Model,~Value,~Details,~Context,~split,~model_leaf,right,context,right_state)),
        walk_matches(~Native,~Model,~Output,~Value,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,~represent,~pack,~split,~model_leaf,~divide_matches,~leaf_matches,
          right,parallel(schedule,right),schedule,context,right_state)) :
        {Partition.Fork{PartitionLaws.carry(~&2,~&1,~Value,~Output,~Details,~pack,model(~Model,~Value,~Details,~Context,~split,~model_leaf,left,context,left_state)),_} ==
          PartitionLaws.carry(~&2,~&1,~Value,~Output,~Details,~pack,model(~Model,~Value,~Details,~Context,~split,~model_leaf,Pieces.Split{left,right},context,state)) :
          Partition.Tree<&1,Output,Details>}
      {==}

# The shape of ranges: a tree with nothing at its leaves.
def shape(layout: Pieces.Ranges) -> Partition.Tree<&2,Unit,Unit>:
  match layout:
    case Pieces.Range{first,rows}: Partition.Leaf{Unit{},first,rows,Unit{}}
    case Pieces.Split{left,right}: Partition.Fork{shape(left),shape(right)}

def unit_ranges_first(depth: Nat,+unit: U32,+first: U32,+count: U32) -> {Pieces.first(Pieces.unit_ranges(depth,unit,first,count)) == first : U32}:
  match depth:
    case 0n: {==}
    case 1n+ +rest: unit_ranges_first(rest,unit,first,(count / (unit * 2) * unit : U32))

# A serial group returns only a unit value. Exposing it lets a group split
# at an ownership fork without changing any leaf call.
def group_results(~Output: Type,left: Partition.Tree<&1,Output,Unit> & Unit,right: Partition.Tree<&1,Output,Unit> & Unit) ->
  {Pair.fst(Partition.Tree<&1,Output,Unit>,Unit,Pieces.reduced(~Output,~Unit,~Pieces.group_unit,left,right)) ==
    Partition.Fork{Pair.fst(Partition.Tree<&1,Output,Unit>,Unit,left),Pair.fst(Partition.Tree<&1,Output,Unit>,Unit,right)} : Partition.Tree<&1,Output,Unit>}:
  (left,Unit{}) = left
  (right,Unit{}) = right
  {==}

def group_fork(~Block: Type,~Output: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Block -> Unit -> Partition.Tree<&1,Output,Unit>,
  +context: Context,left: Partition.Tree<&1,Block,Unit>,right: Partition.Tree<&1,Block,Unit>) ->
  {Pieces.group_run(~Block,~Output,~Context,~leaf,context,Partition.Fork{left,right}) ==
    Partition.Fork{Pieces.group_run(~Block,~Output,~Context,~leaf,context,left),Pieces.group_run(~Block,~Output,~Context,~leaf,context,right)} : Partition.Tree<&1,Output,Unit>}:
  group_results(~Output,
    Pieces.reduce(~Block,~Output,~Unit,~Context,~Pieces.group_piece(~Block,~Output,~Context,~leaf),~Pieces.group_unit,context,left),
    Pieces.reduce(~Block,~Output,~Unit,~Context,~Pieces.group_piece(~Block,~Output,~Context,~leaf),~Pieces.group_unit,context,right))

# Grouping changes only where the shared walk hands work to a task. Every
# task plan computes the same serial leaf calls on the owned tree.
def group_children(~Output: Type,-left: Partition.Tree<&1,Output,Unit>,-right: Partition.Tree<&1,Output,Unit>,
  -expected_left: Partition.Tree<&1,Output,Unit>,-expected_right: Partition.Tree<&1,Output,Unit>,
  left_equal: {left == expected_left : Partition.Tree<&1,Output,Unit>},right_equal: {right == expected_right : Partition.Tree<&1,Output,Unit>}) ->
  {Partition.Fork{left,right} == Partition.Fork{expected_left,expected_right} : Partition.Tree<&1,Output,Unit>}:
  %left_equal : {Partition.Fork{left,right} == Partition.Fork{_,expected_right} : Partition.Tree<&1,Output,Unit>}
  %right_equal : {Partition.Fork{left,right} == Partition.Fork{left,_} : Partition.Tree<&1,Output,Unit>}
  {==}

def grouped(~Part: Data,~Block: Type,~pack: Part -> Block,~Output: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Block -> Unit -> Partition.Tree<&1,Output,Unit>,
  layout: Pieces.Ranges,+enabled: Bool,+context: Context,+tree: Model.Shaped<Part>) ->
  {Pieces.grouped(~Block,~Output,~Context,~leaf,layout,enabled,context,Model.shaped_pack(~Part,~Block,~pack,tree)) ==
    Pieces.group_run(~Block,~Output,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,tree)) : Partition.Tree<&1,Output,Unit>}:
  match layout tree:
    case Pieces.Range{first,rows} tree: {==}
    case Pieces.Split{+left_plan,+right_plan} Model.Piece{part,first,rows}:
      fork(~Partition.Tree<&1,Block,Unit>,~Output,~Unit,~Context,~Bool,~Pieces.in_parallel,~Pieces.group_divide(~Block,~Context),
        ~Pieces.group_task(~Block,~Output,~Context,~leaf),~Pieces.subtree_kept(~Block,~Output,~Context,~Pieces.group_task(~Block,~Output,~Context,~leaf)),
        left_plan,right_plan,enabled,enabled,context,Partition.Leaf{pack(part),first,rows,Unit{}})
    case Pieces.Split{+left_plan,+right_plan} Model.Pieces{+left,+right}:
      %Equal.sym(Partition.Tree<&1,Output,Unit>,
        Pieces.grouped(~Block,~Output,~Context,~leaf,Pieces.Split{left_plan,right_plan},enabled,context,Model.shaped_pack(~Part,~Block,~pack,Model.Pieces{left,right})),
        Partition.Fork{Pieces.grouped(~Block,~Output,~Context,~leaf,left_plan,enabled,context,Model.shaped_pack(~Part,~Block,~pack,left)),Pieces.grouped(~Block,~Output,~Context,~leaf,right_plan,enabled,context,Model.shaped_pack(~Part,~Block,~pack,right))},
        fork(~Partition.Tree<&1,Block,Unit>,~Output,~Unit,~Context,~Bool,~Pieces.in_parallel,~Pieces.group_divide(~Block,~Context),
          ~Pieces.group_task(~Block,~Output,~Context,~leaf),~Pieces.subtree_kept(~Block,~Output,~Context,~Pieces.group_task(~Block,~Output,~Context,~leaf)),
          left_plan,right_plan,enabled,enabled,context,Model.shaped_pack(~Part,~Block,~pack,Model.Pieces{left,right}))) :
        {_ == Pieces.group_run(~Block,~Output,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,Model.Pieces{left,right})) : Partition.Tree<&1,Output,Unit>}
      %Equal.sym(Partition.Tree<&1,Output,Unit>,
        Pieces.group_run(~Block,~Output,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,Model.Pieces{left,right})),
        Partition.Fork{Pieces.group_run(~Block,~Output,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,left)),Pieces.group_run(~Block,~Output,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,right))},
        group_fork(~Block,~Output,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,left),Model.shaped_pack(~Part,~Block,~pack,right))) :
        {Partition.Fork{Pieces.grouped(~Block,~Output,~Context,~leaf,left_plan,enabled,context,Model.shaped_pack(~Part,~Block,~pack,left)),Pieces.grouped(~Block,~Output,~Context,~leaf,right_plan,enabled,context,Model.shaped_pack(~Part,~Block,~pack,right))} ==
          _ : Partition.Tree<&1,Output,Unit>}
      group_children(~Output,
        Pieces.grouped(~Block,~Output,~Context,~leaf,left_plan,enabled,context,Model.shaped_pack(~Part,~Block,~pack,left)),Pieces.grouped(~Block,~Output,~Context,~leaf,right_plan,enabled,context,Model.shaped_pack(~Part,~Block,~pack,right)),
        Pieces.group_run(~Block,~Output,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,left)),Pieces.group_run(~Block,~Output,~Context,~leaf,context,Model.shaped_pack(~Part,~Block,~pack,right)),
        grouped(~Part,~Block,~pack,~Output,~Context,~leaf,left_plan,enabled,context,left),grouped(~Part,~Block,~pack,~Output,~Context,~leaf,right_plan,enabled,context,right))

# ---- A walk over a tree ----

# The walk of a tree by recursion: the left subtree first, then the right
# from the state it leaves.
def model_walk(~Input: Type,~Output: Type,~State: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Input -> State -> Partition.Tree<&1,Output,Unit> & State,
  +context: Context,tree: Partition.Tree<&1,Input,Unit>,state: State) -> Partition.Tree<&1,Output,Unit> & State:
  match tree:
    case Partition.Leaf{input,first,count,details}: leaf(context,first,count,input,state)
    case Partition.Fork{left,right}:
      Storage.bind_pair(~Partition.Tree<&1,Output,Unit>,~State,~(Partition.Tree<&1,Output,Unit> & State),
        model_walk(~Input,~Output,~State,~Context,~leaf,context,left,state),left => state =>
          Storage.bind_pair(~Partition.Tree<&1,Output,Unit>,~State,~(Partition.Tree<&1,Output,Unit> & State),
            model_walk(~Input,~Output,~State,~Context,~leaf,context,right,state),right => state => (Partition.Fork{left,right},state)))

# A right tree walked after a left one joins it.
def joined_after(~Output: Type,~State: Type,left: Partition.Tree<&1,Output,Unit>,lefts: Pieces.Lefts<Output>,made: Partition.Tree<&1,Output,Unit> & State) ->
  {Pieces.leaf_walked(~Output,~State,lefts,Storage.bind_pair(~Partition.Tree<&1,Output,Unit>,~State,~(Partition.Tree<&1,Output,Unit> & State),made,
      right => state => (Partition.Fork{left,right},state))) ==
    Pieces.joined(~Output,~State,Pieces.leaf_walked(~Output,~State,Pieces.Left{left,lefts},made)) : Pieces.Walked<Output,State>}:
  (right,state) = made
  {==}

def result_walked(~Output: Type,~State: Type,lefts: Pieces.Lefts<Output>,made: Partition.Tree<&1,Output,Unit> & State) ->
  {Pieces.walk_result(~Output,~State,Pieces.leaf_walked(~Output,~State,lefts,made)) == made : Partition.Tree<&1,Output,Unit> & State}:
  (tree,state) = made
  {==}
