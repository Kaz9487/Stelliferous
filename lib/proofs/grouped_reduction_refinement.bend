# A reader's local equation lifts through the actual shared loop, group
# traversal and adjacent-pair reduction. Instantiations discharge that equation.
import Base
import ../traversal_grouped.bend as Grouped
import ../traversal_loop.bend as Loop
import ./loop_projection.bend as Projection
import ./traversal_array_refinement.bend as Traversals
import ./sequence_membership.bend as Members
import ./storage_shape.bend as Shape
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_extrema.bend as Extrema
import ./nat_ceiling.bend as Ceiling
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ./natural_addition.bend as Addition

# Lift one combine equation through the same adjacent-pair tree for owners
# and their pure models.
def pack_values(~Runtime: Type,~Model: Data,~represent: Model -> Runtime,values: List<&2,Model>) -> List<&1,Runtime>:
  Traversals.collect_values(~&1,~Runtime,~Model,~represent,values)

def pairs_refines_model(~Runtime: Type,~Model: Data,~Context: Data,
  ~runtime_combine: Context -> Runtime -> Runtime -> Runtime,~model_combine: Context -> Model -> Model -> Model,~represent: Model -> Runtime,
  ~combine_matches: @+context: Context -> @+left: Model -> @+right: Model ->
    {runtime_combine(context,represent(left),represent(right)) == represent(model_combine(context,left,right)) : Runtime},
  +context: Context,+values: List<&2,Model>) ->
  {Grouped.pairs(~&1,~Runtime,~Context,~runtime_combine,context,pack_values(~Runtime,~Model,~represent,values)) ==
    pack_values(~Runtime,~Model,~represent,Grouped.pairs(~&2,~Model,~Context,~model_combine,context,values)) : List<&1,Runtime>}:
  match values:
    case Nil{}: {==}
    case head <> Nil{}: {==}
    case +left <> +right <> +tail:
      %Equal.sym(Runtime,runtime_combine(context,represent(left),represent(right)),represent(model_combine(context,left,right)),combine_matches(context,left,right)) :
        {_ <> Grouped.pairs(~&1,~Runtime,~Context,~runtime_combine,context,pack_values(~Runtime,~Model,~represent,tail)) ==
          pack_values(~Runtime,~Model,~represent,Grouped.pairs(~&2,~Model,~Context,~model_combine,context,values)) : List<&1,Runtime>}
      Equal.cong(List<&1,Runtime>,List<&1,Runtime>,tail => represent(model_combine(context,left,right)) <> tail,
        Grouped.pairs(~&1,~Runtime,~Context,~runtime_combine,context,pack_values(~Runtime,~Model,~represent,tail)),
        pack_values(~Runtime,~Model,~represent,Grouped.pairs(~&2,~Model,~Context,~model_combine,context,tail)),
        pairs_refines_model(~Runtime,~Model,~Context,~runtime_combine,~model_combine,~represent,~combine_matches,context,tail))

def levels_refines_model(~Runtime: Type,~Model: Data,~Context: Data,
  ~runtime_zero: Context -> Runtime,~model_zero: Context -> Model,
  ~runtime_combine: Context -> Runtime -> Runtime -> Runtime,~model_combine: Context -> Model -> Model -> Model,~represent: Model -> Runtime,
  ~zero_matches: @+context: Context -> {runtime_zero(context) == represent(model_zero(context)) : Runtime},
  ~combine_matches: @+context: Context -> @+left: Model -> @+right: Model ->
    {runtime_combine(context,represent(left),represent(right)) == represent(model_combine(context,left,right)) : Runtime},
  +fuel: Nat,+context: Context,+values: List<&2,Model>) ->
  {Grouped.levels(~&1,~Runtime,~Context,~runtime_zero,~runtime_combine,fuel,context,pack_values(~Runtime,~Model,~represent,values)) ==
    represent(Grouped.levels(~&2,~Model,~Context,~model_zero,~model_combine,fuel,context,values)) : Runtime}:
  match fuel values:
    case 0n Nil{}: zero_matches(context)
    case 0n head <> tail: {==}
    case 1n+rest Nil{}: zero_matches(context)
    case 1n+rest head <> Nil{}: {==}
    case 1n+ +rest +left <> +right <> +tail:
      %Equal.sym(List<&1,Runtime>,Grouped.pairs(~&1,~Runtime,~Context,~runtime_combine,context,pack_values(~Runtime,~Model,~represent,values)),
        pack_values(~Runtime,~Model,~represent,Grouped.pairs(~&2,~Model,~Context,~model_combine,context,values)),
        pairs_refines_model(~Runtime,~Model,~Context,~runtime_combine,~model_combine,~represent,~combine_matches,context,values)) :
        {Grouped.levels(~&1,~Runtime,~Context,~runtime_zero,~runtime_combine,rest,context,_) ==
          represent(Grouped.levels(~&2,~Model,~Context,~model_zero,~model_combine,fuel,context,values)) : Runtime}
      levels_refines_model(~Runtime,~Model,~Context,~runtime_zero,~model_zero,~runtime_combine,~model_combine,~represent,~zero_matches,~combine_matches,
        rest,context,Grouped.pairs(~&2,~Model,~Context,~model_combine,context,values))

type State<-Source: Data,-Context: Data> is Data:
  State{source: Source,context: Context,position: Nat,total: F32}

def packed(~Reader: Type,~Source: Data,~Context: Data,~pack: Source -> Reader,state: State<Source,Context>) -> Grouped.State<Reader,Context>:
  State{source,context,position,total} = state
  Grouped.State{pack(source),context,position,total}

def step(~Source: Data,~Context: Data,~sample: Source -> Context -> Nat -> F32,state: State<Source,Context>) -> State<Source,Context>:
  State{+source,+context,+position,total} = state
  State{source,context,1n+position,(total + sample(source,context,position) : F32)}

def step_matches(~Reader: Type,~Source: Data,~Context: Data,~pack: Source -> Reader,
  ~read: Reader -> Context -> Nat -> Reader & F32,~sample: Source -> Context -> Nat -> F32,
  ~matches: @+source: Source -> @+context: Context -> @+position: Nat -> {read(pack(source),context,position) == (pack(source),sample(source,context,position)) : Reader & F32},
  +state: State<Source,Context>) ->
  {Grouped.step(~Reader,~Context,~read,packed(~Reader,~Source,~Context,~pack,state)) == packed(~Reader,~Source,~Context,~pack,step(~Source,~Context,~sample,state)) : Grouped.State<Reader,Context>}:
  State{+source,+context,+position,+total} = state
  %Equal.sym(Reader & F32,read(pack(source),context,position),(pack(source),sample(source,context,position)),matches(source,context,position)) :
    {Grouped.advanced(~Reader,~Context,context,position,total,_) == Grouped.State{pack(source),context,1n+position,(total + sample(source,context,position) : F32)} : Grouped.State<Reader,Context>}
  {==}

def values(~Source: Data,~Context: Data,~sample: Source -> Context -> Nat -> F32,remaining: Nat,+position: Nat,+source: Source,+context: Context,total: F32) -> F32:
  match remaining:
    case 0n: total
    case 1n+rest: values(~Source,~Context,~sample,rest,1n+position,source,context,(total + sample(source,context,position) : F32))

def finished(~Reader: Type,~Source: Data,~Context: Data,~pack: Source -> Reader,state: State<Source,Context>) -> Reader & F32:
  State{source,context,position,total} = state
  (pack(source),total)

def finish_packed(~Reader: Type,~Source: Data,~Context: Data,~pack: Source -> Reader,+state: State<Source,Context>) ->
  {Grouped.finished(~Reader,~Context,packed(~Reader,~Source,~Context,~pack,state)) == finished(~Reader,~Source,~Context,~pack,state) : Reader & F32}:
  State{source,context,position,total} = state
  {==}

def model_fold(~Reader: Type,~Source: Data,~Context: Data,~pack: Source -> Reader,~sample: Source -> Context -> Nat -> F32,
  remaining: Nat,+position: Nat,+source: Source,+context: Context,+total: F32) ->
  {finished(~Reader,~Source,~Context,~pack,Loop.run(~State<Source,Context>,~step(~Source,~Context,~sample),remaining,State{source,context,position,total})) ==
    (pack(source),values(~Source,~Context,~sample,remaining,position,source,context,total)) : Reader & F32}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest: model_fold(~Reader,~Source,~Context,~pack,~sample,rest,1n+position,source,context,(total + sample(source,context,position) : F32))

def fold(~Reader: Type,~Source: Data,~Context: Data,~pack: Source -> Reader,
  ~read: Reader -> Context -> Nat -> Reader & F32,~sample: Source -> Context -> Nat -> F32,
  ~matches: @+source: Source -> @+context: Context -> @+position: Nat -> {read(pack(source),context,position) == (pack(source),sample(source,context,position)) : Reader & F32},
  +remaining: Nat,+position: Nat,+source: Source,+context: Context) ->
  {Grouped.fold(~Reader,~Context,~read,remaining,position,pack(source),context) ==
    (pack(source),values(~Source,~Context,~sample,remaining,position,source,context,0.0)) : Reader & F32}:
  +initial = {State{source,context,position,0.0} : State<Source,Context>}
  +model = Loop.run(~State<Source,Context>,~step(~Source,~Context,~sample),remaining,initial)
  %Equal.sym(Grouped.State<Reader,Context>,Loop.run(~Grouped.State<Reader,Context>,~Grouped.step(~Reader,~Context,~read),remaining,packed(~Reader,~Source,~Context,~pack,initial)),
    packed(~Reader,~Source,~Context,~pack,model),Projection.iteration_refines_model(~Grouped.State<Reader,Context>,~State<Source,Context>,
      ~Grouped.step(~Reader,~Context,~read),~step(~Source,~Context,~sample),~packed(~Reader,~Source,~Context,~pack),
      ~step_matches(~Reader,~Source,~Context,~pack,~read,~sample,~matches),remaining,initial)) :
    {Grouped.finished(~Reader,~Context,_) == (pack(source),values(~Source,~Context,~sample,remaining,position,source,context,0.0)) : Reader & F32}
  Equal.trans(Reader & F32,Grouped.finished(~Reader,~Context,packed(~Reader,~Source,~Context,~pack,model)),finished(~Reader,~Source,~Context,~pack,model),
    (pack(source),values(~Source,~Context,~sample,remaining,position,source,context,0.0)),finish_packed(~Reader,~Source,~Context,~pack,model),
    model_fold(~Reader,~Source,~Context,~pack,~sample,remaining,position,source,context,0.0))

def group_values(~Source: Data,~Context: Data,~sample: Source -> Context -> Nat -> F32,ranges: List<&2,Grouped.Range>,+source: Source,+context: Context) -> List<&2,F32>:
  match ranges:
    case []: []
    case Grouped.Range{first,count} <> rest:
      values(~Source,~Context,~sample,count,first,source,context,0.0) <> group_values(~Source,~Context,~sample,rest,source,context)

def groups(~Reader: Type,~Source: Data,~Context: Data,~pack: Source -> Reader,
  ~read: Reader -> Context -> Nat -> Reader & F32,~sample: Source -> Context -> Nat -> F32,
  ~matches: @+source: Source -> @+context: Context -> @+position: Nat -> {read(pack(source),context,position) == (pack(source),sample(source,context,position)) : Reader & F32},
  ranges: List<&2,Grouped.Range>,+source: Source,+context: Context) ->
  {Grouped.fold_ranges(~Reader,~Context,~read,ranges,pack(source),context) == (pack(source),group_values(~Source,~Context,~sample,ranges,source,context)) : Reader & List<&2,F32>}:
  match ranges:
    case []: {==}
    case Grouped.Range{+first,+count} <> +rest:
      %Equal.sym(Reader & F32,Grouped.fold(~Reader,~Context,~read,count,first,pack(source),context),
        (pack(source),values(~Source,~Context,~sample,count,first,source,context,0.0)),fold(~Reader,~Source,~Context,~pack,~read,~sample,~matches,count,first,source,context)) :
        {Grouped.grouped_tail(~Reader,_,reader => Grouped.fold_ranges(~Reader,~Context,~read,rest,reader,context)) == (pack(source),group_values(~Source,~Context,~sample,Grouped.Range{first,count} <> rest,source,context)) : Reader & List<&2,F32>}
      %Equal.sym(Reader & List<&2,F32>,Grouped.fold_ranges(~Reader,~Context,~read,rest,pack(source),context),
        (pack(source),group_values(~Source,~Context,~sample,rest,source,context)),groups(~Reader,~Source,~Context,~pack,~read,~sample,~matches,rest,source,context)) :
        {Grouped.prepend(~Reader,values(~Source,~Context,~sample,count,first,source,context,0.0),_) == (pack(source),group_values(~Source,~Context,~sample,Grouped.Range{first,count} <> rest,source,context)) : Reader & List<&2,F32>}
      {==}

def run(~Reader: Type,~Source: Data,~Context: Data,~pack: Source -> Reader,
  ~read: Reader -> Context -> Nat -> Reader & F32,~sample: Source -> Context -> Nat -> F32,
  ~matches: @+source: Source -> @+context: Context -> @+position: Nat -> {read(pack(source),context,position) == (pack(source),sample(source,context,position)) : Reader & F32},
  +ranges: List<&2,Grouped.Range>,+source: Source,+context: Context) ->
  {Grouped.run(~Reader,~Context,~read,ranges,pack(source),context) == (pack(source),Grouped.tree(group_values(~Source,~Context,~sample,ranges,source,context))) : Reader & F32}:
  Equal.cong(Reader & List<&2,F32>,Reader & F32,pair => Grouped.reduced(~Reader,pair),Grouped.fold_ranges(~Reader,~Context,~read,ranges,pack(source),context),
    (pack(source),group_values(~Source,~Context,~sample,ranges,source,context)),groups(~Reader,~Source,~Context,~pack,~read,~sample,~matches,ranges,source,context))

# Observations and preserved storage frames pass through the adjacent tree.
def observed_values(~Runtime: Data,~Model: Data,~Context: Data,~observe: Context -> Runtime -> Model,
  values: List<&2,Runtime>,context: Context) -> List<&2,Model>:
  Traversals.collect_values_context(~&2,~Model,~Runtime,~Context,~observe,values,context)

def observed_cons(~Runtime: Data,~Model: Data,~Context: Data,~observe: Context -> Runtime -> Model,
  +context: Context,+head: Runtime,+tail: List<&2,Runtime>) ->
  {observed_values(~Runtime,~Model,~Context,~observe,head <> tail,context) == observe(context,head) <> observed_values(~Runtime,~Model,~Context,~observe,tail,context) : List<&2,Model>}:
  {==}

def pairs_projected(~Runtime: Data,~Model: Data,~Context: Data,~RuntimeContext: Data,~project: Context -> RuntimeContext,~ModelContext: Data,~model_project: Context -> ModelContext,
  ~combine: RuntimeContext -> Runtime -> Runtime -> Runtime,~model_combine: ModelContext -> Model -> Model -> Model,
  ~observe: Context -> Runtime -> Model,~Frame: Context -> Runtime -> Data,
  ~combine_frame: @+context: Context -> @+left: Runtime -> @+right: Runtime -> Frame(context,left) -> Frame(context,right) -> Frame(context,combine(project(context),left,right)),
  ~combine_matches: @+context: Context -> @+left: Runtime -> @+right: Runtime -> Frame(context,left) -> Frame(context,right) ->
    {observe(context,combine(project(context),left,right)) == model_combine(model_project(context),observe(context,left),observe(context,right)) : Model},
  +context: Context,+values: List<&2,Runtime>,evidence: Members.All(Runtime,{value => Frame(context,value) : Runtime -> Data},values)) ->
  Shape.Both<Members.All(Runtime,{value => Frame(context,value) : Runtime -> Data},Grouped.pairs(~&2,~Runtime,~RuntimeContext,~combine,project(context),values)),
    {observed_values(~Runtime,~Model,~Context,~observe,Grouped.pairs(~&2,~Runtime,~RuntimeContext,~combine,project(context),values),context) ==
      Grouped.pairs(~&2,~Model,~ModelContext,~model_combine,model_project(context),observed_values(~Runtime,~Model,~Context,~observe,values,context)) : List<&2,Model>}>:
  match values:
    case Nil{}: Shape.Both{Unit{},{==}}
    case head <> Nil{}: Shape.Both{evidence,{==}}
    case +left <> +right <> +tail:
      Shape.Both{+left_frame,Shape.Both{+right_frame,tail_frame}} = evidence
      +tail_proof = pairs_projected(~Runtime,~Model,~Context,~RuntimeContext,~project,~ModelContext,~model_project,~combine,~model_combine,~observe,~Frame,~combine_frame,~combine_matches,context,tail,tail_frame)
      +paired = Grouped.pairs(~&2,~Runtime,~RuntimeContext,~combine,project(context),tail)
      +projected = Grouped.pairs(~&2,~Model,~ModelContext,~model_combine,model_project(context),observed_values(~Runtime,~Model,~Context,~observe,tail,context))
      %Equal.sym(List<&2,Model>,observed_values(~Runtime,~Model,~Context,~observe,combine(project(context),left,right) <> paired,context),
        observe(context,combine(project(context),left,right)) <> observed_values(~Runtime,~Model,~Context,~observe,paired,context),observed_cons(~Runtime,~Model,~Context,~observe,context,combine(project(context),left,right),paired)) :
        Shape.Both<Members.All(Runtime,{value => Frame(context,value) : Runtime -> Data},combine(project(context),left,right) <> paired),
          {_ == Grouped.pairs(~&2,~Model,~ModelContext,~model_combine,model_project(context),observed_values(~Runtime,~Model,~Context,~observe,values,context)) : List<&2,Model>}>
      %Equal.sym(List<&2,Model>,observed_values(~Runtime,~Model,~Context,~observe,values,context),
        observe(context,left) <> observed_values(~Runtime,~Model,~Context,~observe,right <> tail,context),observed_cons(~Runtime,~Model,~Context,~observe,context,left,right <> tail)) :
        Shape.Both<Members.All(Runtime,{value => Frame(context,value) : Runtime -> Data},combine(project(context),left,right) <> paired),
          {observe(context,combine(project(context),left,right)) <> observed_values(~Runtime,~Model,~Context,~observe,paired,context) == Grouped.pairs(~&2,~Model,~ModelContext,~model_combine,model_project(context),_) : List<&2,Model>}>
      %Equal.sym(List<&2,Model>,observed_values(~Runtime,~Model,~Context,~observe,right <> tail,context),
        observe(context,right) <> observed_values(~Runtime,~Model,~Context,~observe,tail,context),observed_cons(~Runtime,~Model,~Context,~observe,context,right,tail)) :
        Shape.Both<Members.All(Runtime,{value => Frame(context,value) : Runtime -> Data},combine(project(context),left,right) <> paired),
          {observe(context,combine(project(context),left,right)) <> observed_values(~Runtime,~Model,~Context,~observe,paired,context) == Grouped.pairs(~&2,~Model,~ModelContext,~model_combine,model_project(context),observe(context,left) <> _) : List<&2,Model>}>
      Shape.Both{Shape.Both{combine_frame(context,left,right,left_frame,right_frame),
          Shape.first(Members.All(Runtime,{value => Frame(context,value) : Runtime -> Data},paired),
            {observed_values(~Runtime,~Model,~Context,~observe,paired,context) == projected : List<&2,Model>},tail_proof)},
        %Equal.sym(Model,observe(context,combine(project(context),left,right)),model_combine(model_project(context),observe(context,left),observe(context,right)),combine_matches(context,left,right,left_frame,right_frame)) :
          {_ <> observed_values(~Runtime,~Model,~Context,~observe,paired,context) ==
            model_combine(model_project(context),observe(context,left),observe(context,right)) <> projected : List<&2,Model>}
        Equal.cong(List<&2,Model>,List<&2,Model>,rest => model_combine(model_project(context),observe(context,left),observe(context,right)) <> rest,
          observed_values(~Runtime,~Model,~Context,~observe,paired,context),projected,
          Shape.second(Members.All(Runtime,{value => Frame(context,value) : Runtime -> Data},paired),
            {observed_values(~Runtime,~Model,~Context,~observe,paired,context) == projected : List<&2,Model>},tail_proof))}

def observed_levels_advance(~Runtime: Data,~Model: Data,~Context: Data,~ModelContext: Data,~model_project: Context -> ModelContext,~observe: Context -> Runtime -> Model,
  ~zero: ModelContext -> Model,~combine: ModelContext -> Model -> Model -> Model,+fuel: Nat,+context: Context,+left: Runtime,+right: Runtime,+tail: List<&2,Runtime>) ->
  {Grouped.levels(~&2,~Model,~ModelContext,~zero,~combine,1n+fuel,model_project(context),observed_values(~Runtime,~Model,~Context,~observe,left <> right <> tail,context)) ==
    Grouped.levels(~&2,~Model,~ModelContext,~zero,~combine,fuel,model_project(context),Grouped.pairs(~&2,~Model,~ModelContext,~combine,model_project(context),observed_values(~Runtime,~Model,~Context,~observe,left <> right <> tail,context))) : Model}:
  %Equal.sym(List<&2,Model>,observed_values(~Runtime,~Model,~Context,~observe,left <> right <> tail,context),
    observe(context,left) <> observed_values(~Runtime,~Model,~Context,~observe,right <> tail,context),observed_cons(~Runtime,~Model,~Context,~observe,context,left,right <> tail)) :
    {Grouped.levels(~&2,~Model,~ModelContext,~zero,~combine,1n+fuel,model_project(context),_) == Grouped.levels(~&2,~Model,~ModelContext,~zero,~combine,fuel,model_project(context),Grouped.pairs(~&2,~Model,~ModelContext,~combine,model_project(context),_)) : Model}
  %Equal.sym(List<&2,Model>,observed_values(~Runtime,~Model,~Context,~observe,right <> tail,context),
    observe(context,right) <> observed_values(~Runtime,~Model,~Context,~observe,tail,context),observed_cons(~Runtime,~Model,~Context,~observe,context,right,tail)) :
    {Grouped.levels(~&2,~Model,~ModelContext,~zero,~combine,1n+fuel,model_project(context),observe(context,left) <> _) == Grouped.levels(~&2,~Model,~ModelContext,~zero,~combine,fuel,model_project(context),Grouped.pairs(~&2,~Model,~ModelContext,~combine,model_project(context),observe(context,left) <> _)) : Model}
  {==}

def levels_projected(~Runtime: Data,~Model: Data,~Context: Data,~RuntimeContext: Data,~project: Context -> RuntimeContext,~ModelContext: Data,~model_project: Context -> ModelContext,
  ~zero: RuntimeContext -> Runtime,~model_zero: ModelContext -> Model,
  ~combine: RuntimeContext -> Runtime -> Runtime -> Runtime,~model_combine: ModelContext -> Model -> Model -> Model,
  ~observe: Context -> Runtime -> Model,~Frame: Context -> Runtime -> Data,
  ~zero_frame: @+context: Context -> Frame(context,zero(project(context))),
  ~zero_matches: @+context: Context -> {observe(context,zero(project(context))) == model_zero(model_project(context)) : Model},
  ~combine_frame: @+context: Context -> @+left: Runtime -> @+right: Runtime -> Frame(context,left) -> Frame(context,right) -> Frame(context,combine(project(context),left,right)),
  ~combine_matches: @+context: Context -> @+left: Runtime -> @+right: Runtime -> Frame(context,left) -> Frame(context,right) ->
    {observe(context,combine(project(context),left,right)) == model_combine(model_project(context),observe(context,left),observe(context,right)) : Model},
  +fuel: Nat,+context: Context,+values: List<&2,Runtime>,evidence: Members.All(Runtime,{value => Frame(context,value) : Runtime -> Data},values)) ->
  Shape.Both<Frame(context,Grouped.levels(~&2,~Runtime,~RuntimeContext,~zero,~combine,fuel,project(context),values)),
    {observe(context,Grouped.levels(~&2,~Runtime,~RuntimeContext,~zero,~combine,fuel,project(context),values)) ==
      Grouped.levels(~&2,~Model,~ModelContext,~model_zero,~model_combine,fuel,model_project(context),observed_values(~Runtime,~Model,~Context,~observe,values,context)) : Model}>:
  match fuel values:
    case 0n Nil{}: Shape.Both{zero_frame(context),zero_matches(context)}
    case 0n +head <> +tail:
      Shape.Both{head_frame,tail_frame} = evidence
      %Equal.sym(List<&2,Model>,observed_values(~Runtime,~Model,~Context,~observe,head <> tail,context),
        observe(context,head) <> observed_values(~Runtime,~Model,~Context,~observe,tail,context),observed_cons(~Runtime,~Model,~Context,~observe,context,head,tail)) :
        Shape.Both<Frame(context,head),{observe(context,head) == Grouped.levels(~&2,~Model,~ModelContext,~model_zero,~model_combine,0n,model_project(context),_) : Model}>
      Shape.Both{head_frame,{==}}
    case 1n+rest Nil{}: Shape.Both{zero_frame(context),zero_matches(context)}
    case 1n+rest +head <> Nil{}:
      Shape.Both{head_frame,tail_frame} = evidence
      Shape.Both{head_frame,{==}}
    case 1n+ +rest +left <> +right <> +tail:
      +paired = Grouped.pairs(~&2,~Runtime,~RuntimeContext,~combine,project(context),values)
      +projected = Grouped.pairs(~&2,~Model,~ModelContext,~model_combine,model_project(context),observed_values(~Runtime,~Model,~Context,~observe,values,context))
      +pairs = pairs_projected(~Runtime,~Model,~Context,~RuntimeContext,~project,~ModelContext,~model_project,~combine,~model_combine,~observe,~Frame,~combine_frame,~combine_matches,context,values,evidence)
      +paired_frame = Shape.first(Members.All(Runtime,{value => Frame(context,value) : Runtime -> Data},paired),
        {observed_values(~Runtime,~Model,~Context,~observe,paired,context) == projected : List<&2,Model>},pairs)
      %Equal.sym(Model,Grouped.levels(~&2,~Model,~ModelContext,~model_zero,~model_combine,1n+rest,model_project(context),observed_values(~Runtime,~Model,~Context,~observe,values,context)),
        Grouped.levels(~&2,~Model,~ModelContext,~model_zero,~model_combine,rest,model_project(context),projected),observed_levels_advance(~Runtime,~Model,~Context,~ModelContext,~model_project,~observe,~model_zero,~model_combine,rest,context,left,right,tail)) :
        Shape.Both<Frame(context,Grouped.levels(~&2,~Runtime,~RuntimeContext,~zero,~combine,rest,project(context),paired)),
          {observe(context,Grouped.levels(~&2,~Runtime,~RuntimeContext,~zero,~combine,rest,project(context),paired)) == _ : Model}>
      %Shape.second(Members.All(Runtime,{value => Frame(context,value) : Runtime -> Data},paired),
        {observed_values(~Runtime,~Model,~Context,~observe,paired,context) == projected : List<&2,Model>},pairs) :
        Shape.Both<Frame(context,Grouped.levels(~&2,~Runtime,~RuntimeContext,~zero,~combine,rest,project(context),paired)),
          {observe(context,Grouped.levels(~&2,~Runtime,~RuntimeContext,~zero,~combine,rest,project(context),paired)) == Grouped.levels(~&2,~Model,~ModelContext,~model_zero,~model_combine,rest,model_project(context),_) : Model}>
      levels_projected(~Runtime,~Model,~Context,~RuntimeContext,~project,~ModelContext,~model_project,~zero,~model_zero,~combine,~model_combine,~observe,~Frame,~zero_frame,~zero_matches,~combine_frame,~combine_matches,rest,context,paired,paired_frame)

# Nonempty groups stay inside the spatial rectangle and below the group size.
def RangeBounded(+height: Nat,+width: Nat,+group: Nat,range: Grouped.Range) -> Data:
  Grouped.Range{+first,+count} = range
  Shape.Both<{Nat.is_lt(0n,count) == True{} : Bool},
    Shape.Both<{Nat.is_le(count,group) == True{} : Bool},{Nat.is_le(Nat.add(first,count),Nat.mul(height,width)) == True{} : Bool}>>

def clipped_room(+first: Nat,+size: Nat,+span: Nat,+inside: {Nat.is_lt(first,size) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(first,Nat.min(span,Nat.sub(size,first))),size) == True{} : Bool}:
  %Order.restore_shift(first,size,Bounds.strict_le(first,size,inside)) :
    {Nat.is_le(Nat.add(first,Nat.min(span,Nat.sub(size,first))),_) == True{} : Bool}
  Order.offset_le(first,Nat.min(span,Nat.sub(size,first)),Nat.sub(size,first),Extrema.minimum_right(span,Nat.sub(size,first)))

def rows_bounded(+remaining: Nat,+index: Nat,+height: Nat,+width: Nat,+span: Nat,+group: Nat,
  +width_positive: {Nat.is_lt(0n,width) == True{} : Bool},+span_positive: {Nat.is_lt(0n,span) == True{} : Bool},
  +span_bound: {Nat.is_le(Nat.mul(span,width),group) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(index,remaining),Ceiling.ceiling(height,span)) == True{} : Bool}) ->
  Members.All(Grouped.Range,{range => RangeBounded(height,width,group,range) : Grouped.Range -> Data},Grouped.rows(remaining,Nat.mul(index,span),height,width,span)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      +first = Nat.mul(index,span)
      +take = Nat.min(span,Nat.sub(height,first))
      +inside = {
        %Ceiling.upper_comparison(height,span,index,span_positive) : {_ == True{} : Bool}
        Order.room_index(index,rest,Ceiling.ceiling(height,span),room)
        : {Nat.is_lt(first,height) == True{} : Bool}}
      Shape.Both{Shape.Both{Order.product_positive(take,width,Extrema.minimum_strict_lower(span,Nat.sub(height,first),0n,span_positive,Order.positive_difference(height,first,inside)),width_positive),
          Shape.Both{Order.transitive(Nat.mul(take,width),Nat.mul(span,width),group,RowMajor.mul_le(take,span,width,Extrema.minimum_left(span,Nat.sub(height,first))),span_bound),
            %RowMajor.distrib(first,take,width) : {Nat.is_le(_,Nat.mul(height,width)) == True{} : Bool}
            RowMajor.mul_le(Nat.add(first,take),height,width,clipped_room(first,height,span,inside))}},
        %Order.commutative(span,first) :
          Members.All(Grouped.Range,{range => RangeBounded(height,width,group,range) : Grouped.Range -> Data},Grouped.rows(rest,_,height,width,span))
        rows_bounded(rest,1n+index,height,width,span,group,width_positive,span_positive,span_bound,Order.advance_room(index,rest,Ceiling.ceiling(height,span),room))}

def columns_bounded(+remaining: Nat,+index: Nat,+row: Nat,+height: Nat,+width: Nat,+group: Nat,
  +row_inside: {Nat.is_lt(row,height) == True{} : Bool},+group_positive: {Nat.is_lt(0n,group) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(index,remaining),Ceiling.ceiling(width,group)) == True{} : Bool}) ->
  Members.All(Grouped.Range,{range => RangeBounded(height,width,group,range) : Grouped.Range -> Data},Grouped.columns(remaining,row,Nat.mul(index,group),width,group)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      +first = Nat.mul(index,group)
      +take = Nat.min(group,Nat.sub(width,first))
      +base = Nat.mul(row,width)
      +inside = {
        %Ceiling.upper_comparison(width,group,index,group_positive) : {_ == True{} : Bool}
        Order.room_index(index,rest,Ceiling.ceiling(width,group),room)
        : {Nat.is_lt(first,width) == True{} : Bool}}
      Shape.Both{Shape.Both{Extrema.minimum_strict_lower(group,Nat.sub(width,first),0n,group_positive,Order.positive_difference(width,first,inside)),
          Shape.Both{Extrema.minimum_left(group,Nat.sub(width,first)),
            %Equal.sym(Nat,Nat.add(Nat.add(base,first),take),Nat.add(base,Nat.add(first,take)),Addition.associative(base,first,take)) : {Nat.is_le(_,Nat.mul(height,width)) == True{} : Bool}
            Order.transitive(Nat.add(base,Nat.add(first,take)),Nat.add(base,width),Nat.mul(height,width),
              Order.offset_le(base,Nat.add(first,take),width,clipped_room(first,width,group,inside)),
              %Order.commutative(width,base) : {Nat.is_le(_,Nat.mul(height,width)) == True{} : Bool}
              RowMajor.mul_le(1n+row,height,width,Order.strict_follow(row,height,row_inside)))}},
        %Order.commutative(group,first) :
          Members.All(Grouped.Range,{range => RangeBounded(height,width,group,range) : Grouped.Range -> Data},Grouped.columns(rest,row,_,width,group))
        columns_bounded(rest,1n+index,row,height,width,group,row_inside,group_positive,Order.advance_room(index,rest,Ceiling.ceiling(width,group),room))}

def bounded_cat(+height: Nat,+width: Nat,+group: Nat,left: List<&2,Grouped.Range>,+right: List<&2,Grouped.Range>,
  left_evidence: Members.All(Grouped.Range,{range => RangeBounded(height,width,group,range) : Grouped.Range -> Data},left),
  +right_evidence: Members.All(Grouped.Range,{range => RangeBounded(height,width,group,range) : Grouped.Range -> Data},right)) ->
  Members.All(Grouped.Range,{range => RangeBounded(height,width,group,range) : Grouped.Range -> Data},Grouped.cat(left,right)):
  match left:
    case Nil{}: right_evidence
    case +head <> +tail:
      Shape.Both{head_evidence,tail_evidence} = left_evidence
      Shape.Both{head_evidence,bounded_cat(height,width,group,tail,right,tail_evidence,right_evidence)}

def wide_rows_bounded(+remaining: Nat,+row: Nat,+height: Nat,+width: Nat,+group: Nat,
  +group_positive: {Nat.is_lt(0n,group) == True{} : Bool},+room: {Nat.is_le(Nat.add(row,remaining),height) == True{} : Bool}) ->
  Members.All(Grouped.Range,{range => RangeBounded(height,width,group,range) : Grouped.Range -> Data},Grouped.wide_rows(remaining,row,width,group)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      +count = Nat.div(Nat.add(width,Nat.sub(group,1n)),group)
      +head = Grouped.columns(count,row,0n,width,group)
      +tail = Grouped.wide_rows(rest,1n+row,width,group)
      +head_evidence = {
        %Equal.sym(Nat,count,Ceiling.ceiling(width,group),Ceiling.shifted_formula(width,group,group_positive)) :
          Members.All(Grouped.Range,{range => RangeBounded(height,width,group,range) : Grouped.Range -> Data},Grouped.columns(_,row,0n,width,group))
        columns_bounded(Ceiling.ceiling(width,group),0n,row,height,width,group,Order.room_index(row,rest,height,room),group_positive,Order.reflexive(Ceiling.ceiling(width,group)))
        : Members.All(Grouped.Range,{range => RangeBounded(height,width,group,range) : Grouped.Range -> Data},head)}
      bounded_cat(height,width,group,head,tail,head_evidence,wide_rows_bounded(rest,1n+row,height,width,group,group_positive,Order.advance_room(row,rest,height,room)))

def row_span_positive(+group: Nat,+width: Nat,+width_positive: {Nat.is_lt(0n,width) == True{} : Bool},+ordered: {Nat.is_le(width,group) == True{} : Bool}) ->
  {Nat.is_lt(0n,Nat.div(group,width)) == True{} : Bool}:
  +one = {
    %Addition.zero_right(width) : {Nat.div(_,width) == 1n : Nat}
    Division.exact_quotient(1n,width,width_positive)
    : {Nat.div(width,width) == 1n : Nat}}
  Bounds.nat_trans(0n,1n,Nat.div(group,width),{==},
    %one : {Nat.is_le(_,Nat.div(group,width)) == True{} : Bool}
    DivisionOrder.quotient_le(width,group,width,width_positive,ordered))

def ranges_selected_bounded(wide: Bool,+height: Nat,+width: Nat,+group: Nat,
  +width_positive: {Nat.is_lt(0n,width) == True{} : Bool},+group_positive: {Nat.is_lt(0n,group) == True{} : Bool},
  +choice: {Nat.is_lt(group,width) == wide : Bool}) ->
  Members.All(Grouped.Range,{range => RangeBounded(height,width,group,range) : Grouped.Range -> Data},Grouped.ranges_selected(wide,height,width,group)):
  match wide:
    case True{}: wide_rows_bounded(height,0n,height,width,group,group_positive,Order.reflexive(height))
    case False{}:
      +span = Nat.div(group,width)
      +positive = row_span_positive(group,width,width_positive,Order.not_less(group,width,choice))
      %Equal.sym(Nat,Nat.div(Nat.add(height,Nat.sub(span,1n)),span),Ceiling.ceiling(height,span),Ceiling.shifted_formula(height,span,positive)) :
        Members.All(Grouped.Range,{range => RangeBounded(height,width,group,range) : Grouped.Range -> Data},Grouped.rows(_,0n,height,width,span))
      rows_bounded(Ceiling.ceiling(height,span),0n,height,width,span,group,width_positive,positive,DivisionOrder.product_below(group,width,width_positive),Order.reflexive(Ceiling.ceiling(height,span)))

def ranges_bounded(+height: Nat,+width: Nat,+group: Nat,+width_positive: {Nat.is_lt(0n,width) == True{} : Bool},+group_positive: {Nat.is_lt(0n,group) == True{} : Bool}) ->
  Members.All(Grouped.Range,{range => RangeBounded(height,width,group,range) : Grouped.Range -> Data},Grouped.ranges(height,width,group)):
  ranges_selected_bounded(Nat.is_lt(group,width),height,width,group,width_positive,group_positive,{==})
