# The plan recursion, by structure. A parallel split and its leaf tasks
# compute what the sequential recursion computes, a walk over represented
# states is the represented model walk, and a plan halved at whole units
# starts where its rows start.
import Base
import ../tensor_band.bend as Bands
import ../traversal_partition.bend as Partition
import ./traversal_partition_refinement.bend as PartitionLaws
import ./storage_shape.bend as Shape
import ./tensor_band_model.bend as Model

# Both halves of a split, the left one first.
def sequential(~State: Type,~Output: Type,~Details: Data,~Context: Data,~Schedule: Data,
  ~parallel: Schedule -> Bands.Plan -> Bool,
  ~divide: Context -> Bands.Plan -> Bands.Plan -> State -> Bands.Division<State>,
  ~leaf: Context -> U32 -> U32 -> State -> Partition.Tree<&1,Output,Details>,
  ~keep: Context -> Bands.Plan -> State -> Partition.Tree<&1,Output,Details>,
  +left_plan: Bands.Plan,+right_plan: Bands.Plan,+schedule: Schedule,+context: Context,division: Bands.Division<State>) -> Partition.Tree<&1,Output,Details>:
  Bands.divided(~State,~Partition.Tree<&1,Output,Details>,division,whole => keep(context,Bands.Split{left_plan,right_plan},whole),
    left_state => right_state =>
      Partition.Fork{Bands.each_part(~State,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,left_plan,parallel(schedule,left_plan),schedule,context,left_state),
        Bands.each_part(~State,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,right_plan,parallel(schedule,right_plan),schedule,context,right_state)})

# A split, in parallel or not, with its leaves called directly or not, is
# the sequential split.
def fork(~State: Type,~Output: Type,~Details: Data,~Context: Data,~Schedule: Data,
  ~parallel: Schedule -> Bands.Plan -> Bool,
  ~divide: Context -> Bands.Plan -> Bands.Plan -> State -> Bands.Division<State>,
  ~leaf: Context -> U32 -> U32 -> State -> Partition.Tree<&1,Output,Details>,
  ~keep: Context -> Bands.Plan -> State -> Partition.Tree<&1,Output,Details>,
  +left_plan: Bands.Plan,+right_plan: Bands.Plan,+mode: Bool,+schedule: Schedule,+context: Context,-state: State) ->
  {Bands.each_part(~State,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,Bands.Split{left_plan,right_plan},mode,schedule,context,state) ==
    sequential(~State,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,left_plan,right_plan,schedule,context,divide(context,left_plan,right_plan,state)) :
    Partition.Tree<&1,Output,Details>}:
  match left_plan right_plan mode:
    case Bands.Rows{left_first,left_rows} Bands.Rows{right_first,right_rows} False{}: {==}
    case Bands.Rows{left_first,left_rows} Bands.Rows{right_first,right_rows} True{}: {==}
    case Bands.Rows{left_first,left_rows} Bands.Split{right_left,right_right} False{}: {==}
    case Bands.Rows{left_first,left_rows} Bands.Split{right_left,right_right} True{}: {==}
    case Bands.Split{left_left,left_right} Bands.Rows{right_first,right_rows} False{}: {==}
    case Bands.Split{left_left,left_right} Bands.Rows{right_first,right_rows} True{}: {==}
    case Bands.Split{left_left,left_right} Bands.Split{right_left,right_right} False{}: {==}
    case Bands.Split{left_left,left_right} Bands.Split{right_left,right_right} True{}: {==}

# The parts of a plan over a model state that a pure rule splits.
def model(~Model: Data,~Value: Data,~Details: Data,~Context: Data,
  ~split: Context -> Bands.Plan -> Bands.Plan -> Model -> PartitionLaws.Split<Model>,
  ~leaf: Context -> U32 -> U32 -> Model -> Partition.Tree<&2,Value,Details>,
  layout: Bands.Plan,+context: Context,+state: Model) -> Partition.Tree<&2,Value,Details>:
  match layout:
    case Bands.Rows{first,rows}: leaf(context,first,rows,state)
    case Bands.Split{+left,+right}:
      +halves = split(context,left,right,state)
      Partition.Fork{model(~Model,~Value,~Details,~Context,~split,~leaf,left,context,PartitionLaws.first(~Model,halves)),
        model(~Model,~Value,~Details,~Context,~split,~leaf,right,context,PartitionLaws.second(~Model,halves))}

# A walk that always divides a represented state as the model splits it, and
# whose leaves are the represented model leaves, is the represented model
# walk on every plan.
def walk_matches(~Native: Type,~Model: Data,~Output: Type,~Value: Data,~Details: Data,~Context: Data,~Schedule: Data,
  ~parallel: Schedule -> Bands.Plan -> Bool,
  ~divide: Context -> Bands.Plan -> Bands.Plan -> Native -> Bands.Division<Native>,
  ~leaf: Context -> U32 -> U32 -> Native -> Partition.Tree<&1,Output,Details>,
  ~keep: Context -> Bands.Plan -> Native -> Partition.Tree<&1,Output,Details>,
  ~represent: Model -> Native,
  ~pack: Value -> Output,
  ~split: Context -> Bands.Plan -> Bands.Plan -> Model -> PartitionLaws.Split<Model>,
  ~model_leaf: Context -> U32 -> U32 -> Model -> Partition.Tree<&2,Value,Details>,
  ~divide_matches: @+context: Context -> @+left: Bands.Plan -> @+right: Bands.Plan -> @+state: Model ->
    {divide(context,left,right,represent(state)) ==
      Bands.Divided{represent(PartitionLaws.first(~Model,split(context,left,right,state))),represent(PartitionLaws.second(~Model,split(context,left,right,state)))} :
      Bands.Division<Native>},
  ~leaf_matches: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+state: Model ->
    {leaf(context,first,rows,represent(state)) == PartitionLaws.carry(~&2,~&1,~Value,~Output,~Details,~pack,model_leaf(context,first,rows,state)) :
      Partition.Tree<&1,Output,Details>},
  layout: Bands.Plan,+mode: Bool,+schedule: Schedule,+context: Context,+state: Model) ->
  {Bands.each_part(~Native,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,layout,mode,schedule,context,represent(state)) ==
    PartitionLaws.carry(~&2,~&1,~Value,~Output,~Details,~pack,model(~Model,~Value,~Details,~Context,~split,~model_leaf,layout,context,state)) :
    Partition.Tree<&1,Output,Details>}:
  match layout:
    case Bands.Rows{+first,+rows}: leaf_matches(context,first,rows,state)
    case Bands.Split{+left,+right}:
      +halves = split(context,left,right,state)
      +left_state = PartitionLaws.first(~Model,halves)
      +right_state = PartitionLaws.second(~Model,halves)
      %Equal.sym(Partition.Tree<&1,Output,Details>,
        Bands.each_part(~Native,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,Bands.Split{left,right},mode,schedule,context,represent(state)),
        sequential(~Native,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,left,right,schedule,context,divide(context,left,right,represent(state))),
        fork(~Native,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,left,right,mode,schedule,context,represent(state))) :
        {_ == PartitionLaws.carry(~&2,~&1,~Value,~Output,~Details,~pack,model(~Model,~Value,~Details,~Context,~split,~model_leaf,Bands.Split{left,right},context,state)) :
          Partition.Tree<&1,Output,Details>}
      %Equal.sym(Bands.Division<Native>,divide(context,left,right,represent(state)),Bands.Divided{represent(left_state),represent(right_state)},
        divide_matches(context,left,right,state)) :
        {sequential(~Native,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,left,right,schedule,context,_) ==
          PartitionLaws.carry(~&2,~&1,~Value,~Output,~Details,~pack,model(~Model,~Value,~Details,~Context,~split,~model_leaf,Bands.Split{left,right},context,state)) :
          Partition.Tree<&1,Output,Details>}
      %Equal.sym(Partition.Tree<&1,Output,Details>,
        Bands.each_part(~Native,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,left,parallel(schedule,left),schedule,context,represent(left_state)),
        PartitionLaws.carry(~&2,~&1,~Value,~Output,~Details,~pack,model(~Model,~Value,~Details,~Context,~split,~model_leaf,left,context,left_state)),
        walk_matches(~Native,~Model,~Output,~Value,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,~represent,~pack,~split,~model_leaf,~divide_matches,~leaf_matches,
          left,parallel(schedule,left),schedule,context,left_state)) :
        {Partition.Fork{_,Bands.each_part(~Native,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,right,parallel(schedule,right),schedule,context,represent(right_state))} ==
          PartitionLaws.carry(~&2,~&1,~Value,~Output,~Details,~pack,model(~Model,~Value,~Details,~Context,~split,~model_leaf,Bands.Split{left,right},context,state)) :
          Partition.Tree<&1,Output,Details>}
      %Equal.sym(Partition.Tree<&1,Output,Details>,
        Bands.each_part(~Native,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,right,parallel(schedule,right),schedule,context,represent(right_state)),
        PartitionLaws.carry(~&2,~&1,~Value,~Output,~Details,~pack,model(~Model,~Value,~Details,~Context,~split,~model_leaf,right,context,right_state)),
        walk_matches(~Native,~Model,~Output,~Value,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,~represent,~pack,~split,~model_leaf,~divide_matches,~leaf_matches,
          right,parallel(schedule,right),schedule,context,right_state)) :
        {Partition.Fork{PartitionLaws.carry(~&2,~&1,~Value,~Output,~Details,~pack,model(~Model,~Value,~Details,~Context,~split,~model_leaf,left,context,left_state)),_} ==
          PartitionLaws.carry(~&2,~&1,~Value,~Output,~Details,~pack,model(~Model,~Value,~Details,~Context,~split,~model_leaf,Bands.Split{left,right},context,state)) :
          Partition.Tree<&1,Output,Details>}
      {==}

# A fact about every band of a plan, with the state split as the model walk
# splits it. What the facts need travels as one scope value.
def StateAll(~Model: Data,~Context: Data,~Scope: Data,
  ~split: Context -> Bands.Plan -> Bands.Plan -> Model -> PartitionLaws.Split<Model>,
  ~Property: @+scope: Scope -> @+context: Context -> @+first: U32 -> @+rows: U32 -> @+state: Model -> Data,
  +scope: Scope,layout: Bands.Plan,+context: Context,+state: Model) -> Data:
  match layout:
    case Bands.Rows{+first,+rows}: Property(scope,context,first,rows,state)
    case Bands.Split{+left,+right}:
      +halves = split(context,left,right,state)
      Shape.Both<StateAll(~Model,~Context,~Scope,~split,~Property,scope,left,context,PartitionLaws.first(~Model,halves)),
        StateAll(~Model,~Context,~Scope,~split,~Property,scope,right,context,PartitionLaws.second(~Model,halves))>

# Every piece of the model walk, from the fact about each band and what one
# leaf makes of it.
def model_all(~Model: Data,~Value: Data,~Context: Data,~Scope: Data,
  ~split: Context -> Bands.Plan -> Bands.Plan -> Model -> PartitionLaws.Split<Model>,
  ~leaf: Context -> U32 -> U32 -> Model -> Partition.Tree<&2,Value,Unit>,
  ~Property: @+scope: Scope -> @+context: Context -> @+first: U32 -> @+rows: U32 -> @+state: Model -> Data,
  ~Piece: @+scope: Scope -> @+part: Value -> @+first: U32 -> @+rows: U32 -> Data,
  ~derive: @+scope: Scope -> @+context: Context -> @+first: U32 -> @+rows: U32 -> @+state: Model -> @fact: Property(scope,context,first,rows,state) ->
    Model.ShapedAll(~Value,~Scope,~Piece,scope,Model.shaped_parts(~Value,leaf(context,first,rows,state))),
  +scope: Scope,layout: Bands.Plan,+context: Context,+state: Model,facts: StateAll(~Model,~Context,~Scope,~split,~Property,scope,layout,context,state)) ->
  Model.ShapedAll(~Value,~Scope,~Piece,scope,Model.shaped_parts(~Value,model(~Model,~Value,~Unit,~Context,~split,~leaf,layout,context,state))):
  match layout:
    case Bands.Rows{+first,+rows}: derive(scope,context,first,rows,state,facts)
    case Bands.Split{+left,+right}:
      Shape.Both{left_facts,right_facts} = facts
      +halves = split(context,left,right,state)
      Shape.Both{model_all(~Model,~Value,~Context,~Scope,~split,~leaf,~Property,~Piece,~derive,scope,left,context,PartitionLaws.first(~Model,halves),left_facts),
        model_all(~Model,~Value,~Context,~Scope,~split,~leaf,~Property,~Piece,~derive,scope,right,context,PartitionLaws.second(~Model,halves),right_facts)}

# The rows of the pieces of the model walk are the plan, when every leaf keeps
# its band.
def model_rows(~Model: Data,~Value: Data,~Context: Data,
  ~split: Context -> Bands.Plan -> Bands.Plan -> Model -> PartitionLaws.Split<Model>,
  ~leaf: Context -> U32 -> U32 -> Model -> Partition.Tree<&2,Value,Unit>,
  ~leaf_rows: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+state: Model ->
    {Model.shaped_rows(~Value,Model.shaped_parts(~Value,leaf(context,first,rows,state))) == Bands.Rows{first,rows} : Bands.Plan},
  layout: Bands.Plan,+context: Context,+state: Model) ->
  {Model.shaped_rows(~Value,Model.shaped_parts(~Value,model(~Model,~Value,~Unit,~Context,~split,~leaf,layout,context,state))) == layout : Bands.Plan}:
  match layout:
    case Bands.Rows{+first,+rows}: leaf_rows(context,first,rows,state)
    case Bands.Split{+left,+right}:
      +halves = split(context,left,right,state)
      +left_state = PartitionLaws.first(~Model,halves)
      +right_state = PartitionLaws.second(~Model,halves)
      %Equal.sym(Bands.Plan,Model.shaped_rows(~Value,Model.shaped_parts(~Value,model(~Model,~Value,~Unit,~Context,~split,~leaf,left,context,left_state))),left,
        model_rows(~Model,~Value,~Context,~split,~leaf,~leaf_rows,left,context,left_state)) :
        {Bands.Split{_,Model.shaped_rows(~Value,Model.shaped_parts(~Value,model(~Model,~Value,~Unit,~Context,~split,~leaf,right,context,right_state)))} == Bands.Split{left,right} : Bands.Plan}
      %Equal.sym(Bands.Plan,Model.shaped_rows(~Value,Model.shaped_parts(~Value,model(~Model,~Value,~Unit,~Context,~split,~leaf,right,context,right_state))),right,
        model_rows(~Model,~Value,~Context,~split,~leaf,~leaf_rows,right,context,right_state)) :
        {Bands.Split{left,_} == Bands.Split{left,right} : Bands.Plan}
      {==}

# The shape of a plan: a tree with nothing at its leaves.
def shape(layout: Bands.Plan) -> Partition.Tree<&2,Unit,Unit>:
  match layout:
    case Bands.Rows{first,rows}: Partition.Leaf{Unit{},first,rows,Unit{}}
    case Bands.Split{left,right}: Partition.Fork{shape(left),shape(right)}

# A plan halved at whole units starts where its rows start.
def unit_plan_first(depth: Nat,+unit: U32,+first: U32,+count: U32) -> {Bands.first(Bands.unit_plan(depth,unit,first,count)) == first : U32}:
  match depth:
    case 0n: {==}
    case 1n+ +rest: unit_plan_first(rest,unit,first,(count / (unit * 2) * unit : U32))

# Two facts about every band of a plan are one fact about every band.
def StateBoth(~Model: Data,~Context: Data,~Scope: Data,
  ~First: @+scope: Scope -> @+context: Context -> @+first: U32 -> @+rows: U32 -> @+state: Model -> Data,
  ~Second: @+scope: Scope -> @+context: Context -> @+first: U32 -> @+rows: U32 -> @+state: Model -> Data,
  +scope: Scope,+context: Context,+first: U32,+rows: U32,+state: Model) -> Data:
  Shape.Both<First(scope,context,first,rows,state),Second(scope,context,first,rows,state)>

def state_both(~Model: Data,~Context: Data,~Scope: Data,
  ~split: Context -> Bands.Plan -> Bands.Plan -> Model -> PartitionLaws.Split<Model>,
  ~First: @+scope: Scope -> @+context: Context -> @+first: U32 -> @+rows: U32 -> @+state: Model -> Data,
  ~Second: @+scope: Scope -> @+context: Context -> @+first: U32 -> @+rows: U32 -> @+state: Model -> Data,
  +scope: Scope,layout: Bands.Plan,+context: Context,+state: Model,
  first_facts: StateAll(~Model,~Context,~Scope,~split,~First,scope,layout,context,state),
  second_facts: StateAll(~Model,~Context,~Scope,~split,~Second,scope,layout,context,state)) ->
  StateAll(~Model,~Context,~Scope,~split,~StateBoth(~Model,~Context,~Scope,~First,~Second),scope,layout,context,state):
  match layout:
    case Bands.Rows{+first,+rows}: Shape.Both{first_facts,second_facts}
    case Bands.Split{+left,+right}:
      Shape.Both{first_left,first_right} = first_facts
      Shape.Both{second_left,second_right} = second_facts
      +halves = split(context,left,right,state)
      Shape.Both{state_both(~Model,~Context,~Scope,~split,~First,~Second,scope,left,context,PartitionLaws.first(~Model,halves),first_left,second_left),
        state_both(~Model,~Context,~Scope,~split,~First,~Second,scope,right,context,PartitionLaws.second(~Model,halves),first_right,second_right)}

# A fact about the state and the context that every split keeps on both
# halves holds at every band of the plan.
def Kept(~Model: Data,~Context: Data,~Scope: Data,~Property: @+scope: Scope -> @+context: Context -> @+state: Model -> Data,
  +scope: Scope,+context: Context,+first: U32,+rows: U32,+state: Model) -> Data:
  Property(scope,context,state)

def split_kept(~Model: Data,~Context: Data,~Scope: Data,
  ~split: Context -> Bands.Plan -> Bands.Plan -> Model -> PartitionLaws.Split<Model>,
  ~Property: @+scope: Scope -> @+context: Context -> @+state: Model -> Data,
  ~kept: @+scope: Scope -> @+context: Context -> @+left: Bands.Plan -> @+right: Bands.Plan -> @+state: Model -> @+fact: Property(scope,context,state) ->
    Shape.Both<Property(scope,context,PartitionLaws.first(~Model,split(context,left,right,state))),Property(scope,context,PartitionLaws.second(~Model,split(context,left,right,state)))>,
  +scope: Scope,layout: Bands.Plan,+context: Context,+state: Model,+fact: Property(scope,context,state)) ->
  StateAll(~Model,~Context,~Scope,~split,~Kept(~Model,~Context,~Scope,~Property),scope,layout,context,state):
  match layout:
    case Bands.Rows{+first,+rows}: fact
    case Bands.Split{+left,+right}:
      +halves = split(context,left,right,state)
      Shape.Both{split_kept(~Model,~Context,~Scope,~split,~Property,~kept,scope,left,context,PartitionLaws.first(~Model,halves),
          Shape.first(Property(scope,context,PartitionLaws.first(~Model,halves)),Property(scope,context,PartitionLaws.second(~Model,halves)),kept(scope,context,left,right,state,fact))),
        split_kept(~Model,~Context,~Scope,~split,~Property,~kept,scope,right,context,PartitionLaws.second(~Model,halves),
          Shape.second(Property(scope,context,PartitionLaws.first(~Model,halves)),Property(scope,context,PartitionLaws.second(~Model,halves)),kept(scope,context,left,right,state,fact)))}
