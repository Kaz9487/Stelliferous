# A batch writes the mapped values of consecutive positions after reading all
# of them from the original tree. Inside the tree it is the ordered in-place
# map of the same positions: no step of that map reads a position an earlier
# step wrote. For every element type and every batch length.
import Base
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./traversal_array_refinement.bend as ArrayLaws
import ./traversal_map_readback.bend as MapReadback
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./nat_interval.bend as Interval
import ./loop_projection.bend as Projection
import ./row_major_index.bend as RowMajor

def batch(~Element: Data,~Context: Data,~map: Context -> U32 -> Element -> Element,count: Nat,+position: U32,+context: Context,+source: Tree.Tree<Element>,
  target: Tree.Tree<Element>) -> Tree.Tree<Element>:
  match count:
    case 0n: target
    case 1n+rest:
      batch(~Element,~Context,~map,rest,(position + 1 : U32),context,source,Tree.write(Element,target,position,map(context,position,Tree.read(Element,source,position))))

def ModelState(~Element: Data,~Context: Data) -> Data:
  Traversal.State<&2,&2,Unit,Tree.Tree<Element>,Context>

# A loop's last step, for every state and count.
def last_step(~State: Data,~step: State -> State,+count: Nat,+state: State) ->
  {Loop.run(~State,~step,1n+count,state) == step(Loop.run(~State,~step,count,state)) : State}:
  match count:
    case 0n: {==}
    case 1n+ +rest: last_step(~State,~step,rest,step(state))

# The whole state after count steps of the model map.
def state_after(~Element: Data,~Context: Data,~map: Context -> U32 -> Element -> Element,+count: Nat,+start: Nat,+context: Context,+input: Tree.Tree<Element>) ->
  {Loop.run(~ModelState(~Element,~Context),~ArrayLaws.model_map_step(~Element,~Context,~map),count,Traversal.State{Unit{},input,U32.from_nat(start),context}) ==
    Traversal.State{Unit{},MapReadback.run_indexed(~Element,~Context,~map,count,U32.from_nat(start),context,input),U32.from_nat(Nat.add(start,count)),context} :
    ModelState(~Element,~Context)}:
  match count:
    case 0n:
      %Equal.sym(Nat,Nat.add(start,0n),start,Addition.zero_right(start)) :
        {Traversal.State{Unit{},input,U32.from_nat(start),context} == Traversal.State{Unit{},input,U32.from_nat(_),context} : ModelState(~Element,~Context)}
      {==}
    case 1n+ +rest:
      +written = Tree.write(Element,input,U32.from_nat(start),map(context,U32.from_nat(start),Tree.read(Element,input,U32.from_nat(start))))
      %Equal.sym(Nat,Nat.add(start,1n+rest),1n+Nat.add(start,rest),Addition.successor_right(start,rest)) :
        {Loop.run(~ModelState(~Element,~Context),~ArrayLaws.model_map_step(~Element,~Context,~map),rest,Traversal.State{Unit{},written,(U32.from_nat(start) + 1 : U32),context}) ==
          Traversal.State{Unit{},MapReadback.run_indexed(~Element,~Context,~map,rest,(U32.from_nat(start) + 1 : U32),context,written),U32.from_nat(_),context} : ModelState(~Element,~Context)}
      %Equal.sym(U32,(U32.from_nat(start) + 1 : U32),U32.from_nat(1n+start),Bounds.add_one(U32.from_nat(start))) :
        {Loop.run(~ModelState(~Element,~Context),~ArrayLaws.model_map_step(~Element,~Context,~map),rest,Traversal.State{Unit{},written,_,context}) ==
          Traversal.State{Unit{},MapReadback.run_indexed(~Element,~Context,~map,rest,_,context,written),U32.from_nat(1n+Nat.add(start,rest)),context} : ModelState(~Element,~Context)}
      state_after(~Element,~Context,~map,rest,1n+start,context,written)

# One more step of the model map writes the position after the ones it mapped.
def map_split(~Element: Data,~Context: Data,~map: Context -> U32 -> Element -> Element,+first: Nat,+second: Nat,+start: Nat,+context: Context,+input: Tree.Tree<Element>) ->
  {MapReadback.run_indexed(~Element,~Context,~map,Nat.add(first,second),U32.from_nat(start),context,input) ==
    MapReadback.run_indexed(~Element,~Context,~map,second,U32.from_nat(Nat.add(start,first)),context,MapReadback.run_indexed(~Element,~Context,~map,first,U32.from_nat(start),context,input)) : Tree.Tree<Element>}:
  +initial = {Traversal.State{Unit{},input,U32.from_nat(start),context} : ModelState(~Element,~Context)}
  %Equal.sym(ModelState(~Element,~Context),Loop.run(~ModelState(~Element,~Context),~ArrayLaws.model_map_step(~Element,~Context,~map),Nat.add(first,second),initial),
    Loop.run(~ModelState(~Element,~Context),~ArrayLaws.model_map_step(~Element,~Context,~map),second,Loop.run(~ModelState(~Element,~Context),~ArrayLaws.model_map_step(~Element,~Context,~map),first,initial)),
    Projection.iteration_split(~ModelState(~Element,~Context),~ArrayLaws.model_map_step(~Element,~Context,~map),first,second,initial)) :
    {MapReadback.output(~Element,~Context,_) == MapReadback.run_indexed(~Element,~Context,~map,second,U32.from_nat(Nat.add(start,first)),context,MapReadback.run_indexed(~Element,~Context,~map,first,U32.from_nat(start),context,input)) : Tree.Tree<Element>}
  %Equal.sym(ModelState(~Element,~Context),Loop.run(~ModelState(~Element,~Context),~ArrayLaws.model_map_step(~Element,~Context,~map),first,initial),
    Traversal.State{Unit{},MapReadback.run_indexed(~Element,~Context,~map,first,U32.from_nat(start),context,input),U32.from_nat(Nat.add(start,first)),context},state_after(~Element,~Context,~map,first,start,context,input)) :
    {MapReadback.output(~Element,~Context,Loop.run(~ModelState(~Element,~Context),~ArrayLaws.model_map_step(~Element,~Context,~map),second,_)) == MapReadback.run_indexed(~Element,~Context,~map,second,U32.from_nat(Nat.add(start,first)),context,MapReadback.run_indexed(~Element,~Context,~map,first,U32.from_nat(start),context,input)) : Tree.Tree<Element>}
  {==}

def extended(~Element: Data,~Context: Data,~map: Context -> U32 -> Element -> Element,+done: Nat,+start: Nat,+context: Context,+input: Tree.Tree<Element>) ->
  {MapReadback.run_indexed(~Element,~Context,~map,1n+done,U32.from_nat(start),context,input) ==
    Tree.write(Element,MapReadback.run_indexed(~Element,~Context,~map,done,U32.from_nat(start),context,input),U32.from_nat(Nat.add(start,done)),
      map(context,U32.from_nat(Nat.add(start,done)),Tree.read(Element,MapReadback.run_indexed(~Element,~Context,~map,done,U32.from_nat(start),context,input),U32.from_nat(Nat.add(start,done))))) : Tree.Tree<Element>}:
  +initial = {Traversal.State{Unit{},input,U32.from_nat(start),context} : ModelState(~Element,~Context)}
  %Equal.sym(ModelState(~Element,~Context),Loop.run(~ModelState(~Element,~Context),~ArrayLaws.model_map_step(~Element,~Context,~map),1n+done,initial),
    ArrayLaws.model_map_step(~Element,~Context,~map,Loop.run(~ModelState(~Element,~Context),~ArrayLaws.model_map_step(~Element,~Context,~map),done,initial)),
    last_step(~ModelState(~Element,~Context),~ArrayLaws.model_map_step(~Element,~Context,~map),done,initial)) :
    {MapReadback.output(~Element,~Context,_) ==
      Tree.write(Element,MapReadback.run_indexed(~Element,~Context,~map,done,U32.from_nat(start),context,input),U32.from_nat(Nat.add(start,done)),
        map(context,U32.from_nat(Nat.add(start,done)),Tree.read(Element,MapReadback.run_indexed(~Element,~Context,~map,done,U32.from_nat(start),context,input),U32.from_nat(Nat.add(start,done))))) : Tree.Tree<Element>}
  %Equal.sym(ModelState(~Element,~Context),Loop.run(~ModelState(~Element,~Context),~ArrayLaws.model_map_step(~Element,~Context,~map),done,initial),
    Traversal.State{Unit{},MapReadback.run_indexed(~Element,~Context,~map,done,U32.from_nat(start),context,input),U32.from_nat(Nat.add(start,done)),context},
    state_after(~Element,~Context,~map,done,start,context,input)) :
    {MapReadback.output(~Element,~Context,ArrayLaws.model_map_step(~Element,~Context,~map,_)) ==
      Tree.write(Element,MapReadback.run_indexed(~Element,~Context,~map,done,U32.from_nat(start),context,input),U32.from_nat(Nat.add(start,done)),
        map(context,U32.from_nat(Nat.add(start,done)),Tree.read(Element,MapReadback.run_indexed(~Element,~Context,~map,done,U32.from_nat(start),context,input),U32.from_nat(Nat.add(start,done))))) : Tree.Tree<Element>}
  {==}

# A batch that continues done steps of the map completes the map.
def continued(~Element: Data,~Context: Data,~map: Context -> U32 -> Element -> Element,+remaining: Nat,+done: Nat,+start: Nat,+context: Context,+input: Tree.Tree<Element>,
  +depth: Nat,+capacity: U32,+capacity_equal: {Tree.capacity(Element,input) == capacity : U32},+balanced: Shape.Balanced(Element,depth,input),
  +limit: {Nat.is_le(depth,31n) == True{} : Bool},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(Nat.add(start,done),remaining),U32.to_nat(capacity)) == True{} : Bool}) ->
  {batch(~Element,~Context,~map,remaining,U32.from_nat(Nat.add(start,done)),context,input,MapReadback.run_indexed(~Element,~Context,~map,done,U32.from_nat(start),context,input)) ==
    MapReadback.run_indexed(~Element,~Context,~map,Nat.add(done,remaining),U32.from_nat(start),context,input) : Tree.Tree<Element>}:
  match remaining:
    case 0n:
      %Equal.sym(Nat,Nat.add(done,0n),done,Addition.zero_right(done)) :
        {MapReadback.run_indexed(~Element,~Context,~map,done,U32.from_nat(start),context,input) == MapReadback.run_indexed(~Element,~Context,~map,_,U32.from_nat(start),context,input) : Tree.Tree<Element>}
      {==}
    case 1n+ +rest:
      +reached = Nat.add(start,done)
      +query = U32.from_nat(reached)
      +mapped = MapReadback.run_indexed(~Element,~Context,~map,done,U32.from_nat(start),context,input)
      +inside = Order.room_index(reached,rest,U32.to_nat(capacity),room)
      +within = Bounds.strict_le(reached,U32.to_nat(capacity),inside)
      +unread = MapReadback.outside_indexed(~Element,~Context,~map,done,start,query,context,input,depth,capacity,capacity_equal,balanced,limit,capacity_ok,within,
        MapReadback.address_inside(reached,capacity,capacity_ok,inside),
        Interval.After{
          %Equal.sym(Nat,U32.to_nat(query),reached,Bounds.roundtrip(reached,capacity,capacity_ok,within)) : {Nat.is_le(reached,_) == True{} : Bool}
          Order.reflexive(reached)})
      %Equal.sym(Nat,Nat.add(done,1n+rest),1n+Nat.add(done,rest),Addition.successor_right(done,rest)) :
        {batch(~Element,~Context,~map,rest,(query + 1 : U32),context,input,Tree.write(Element,mapped,query,map(context,query,Tree.read(Element,input,query)))) ==
          MapReadback.run_indexed(~Element,~Context,~map,_,U32.from_nat(start),context,input) : Tree.Tree<Element>}
      %unread :
        {batch(~Element,~Context,~map,rest,(query + 1 : U32),context,input,Tree.write(Element,mapped,query,map(context,query,_))) ==
          MapReadback.run_indexed(~Element,~Context,~map,1n+Nat.add(done,rest),U32.from_nat(start),context,input) : Tree.Tree<Element>}
      %extended(~Element,~Context,~map,done,start,context,input) :
        {batch(~Element,~Context,~map,rest,(query + 1 : U32),context,input,_) ==
          MapReadback.run_indexed(~Element,~Context,~map,1n+Nat.add(done,rest),U32.from_nat(start),context,input) : Tree.Tree<Element>}
      %Equal.sym(U32,(query + 1 : U32),U32.from_nat(1n+reached),Bounds.add_one(query)) :
        {batch(~Element,~Context,~map,rest,_,context,input,MapReadback.run_indexed(~Element,~Context,~map,1n+done,U32.from_nat(start),context,input)) ==
          MapReadback.run_indexed(~Element,~Context,~map,1n+Nat.add(done,rest),U32.from_nat(start),context,input) : Tree.Tree<Element>}
      %Addition.successor_right(start,done) :
        {batch(~Element,~Context,~map,rest,U32.from_nat(_),context,input,MapReadback.run_indexed(~Element,~Context,~map,1n+done,U32.from_nat(start),context,input)) ==
          MapReadback.run_indexed(~Element,~Context,~map,1n+Nat.add(done,rest),U32.from_nat(start),context,input) : Tree.Tree<Element>}
      continued(~Element,~Context,~map,rest,1n+done,start,context,input,depth,capacity,capacity_equal,balanced,limit,capacity_ok,
        %Equal.sym(Nat,Nat.add(start,1n+done),1n+reached,Addition.successor_right(start,done)) : {Nat.is_le(Nat.add(_,rest),U32.to_nat(capacity)) == True{} : Bool}
        Order.advance_room(reached,rest,U32.to_nat(capacity),room))

# A batch from the original tree is the map of its positions.
def mapped(~Element: Data,~Context: Data,~map: Context -> U32 -> Element -> Element,+count: Nat,+start: Nat,+context: Context,+input: Tree.Tree<Element>,
  +depth: Nat,+capacity: U32,+capacity_equal: {Tree.capacity(Element,input) == capacity : U32},+balanced: Shape.Balanced(Element,depth,input),
  +limit: {Nat.is_le(depth,31n) == True{} : Bool},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(start,count),U32.to_nat(capacity)) == True{} : Bool}) ->
  {batch(~Element,~Context,~map,count,U32.from_nat(start),context,input,input) == MapReadback.run_indexed(~Element,~Context,~map,count,U32.from_nat(start),context,input) : Tree.Tree<Element>}:
  %Addition.zero_right(start) :
    {batch(~Element,~Context,~map,count,U32.from_nat(_),context,input,input) == MapReadback.run_indexed(~Element,~Context,~map,count,U32.from_nat(start),context,input) : Tree.Tree<Element>}
  continued(~Element,~Context,~map,count,0n,start,context,input,depth,capacity,capacity_equal,balanced,limit,capacity_ok,
    %Equal.sym(Nat,Nat.add(start,0n),start,Addition.zero_right(start)) : {Nat.is_le(Nat.add(_,count),U32.to_nat(capacity)) == True{} : Bool}
    room)

# Consecutive fixed-width jobs use the indexed traversal invariant. A caller
# supplies the equation for one batch and its preserved reader frame.
type RoundPlan<-Element: Data,-Context: Data,-width: Nat> is Data:
  RoundPlan{count: Nat,context: Context,original: Tree.Tree<Element>,depth: Nat,capacity: U32,
    capacity_equal: {Tree.capacity(Element,original) == capacity : U32},
    balanced: Shape.Balanced(Element,depth,original),depth_ok: {Nat.is_le(depth,31n) == True{} : Bool},
    capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
    room: {Nat.is_le(Nat.mul(count,width),U32.to_nat(capacity)) == True{} : Bool}}

def round_count(~Element: Data,~Context: Data,~width: Nat,plan: RoundPlan<Element,Context,width>) -> Nat:
  RoundPlan{count,context,original,depth,capacity,capacity_equal,balanced,depth_ok,capacity_ok,room} = plan
  count

def round_context(~Element: Data,~Context: Data,~width: Nat,plan: RoundPlan<Element,Context,width>) -> Context:
  RoundPlan{count,context,original,depth,capacity,capacity_equal,balanced,depth_ok,capacity_ok,room} = plan
  context

def round_original(~Element: Data,~Context: Data,~width: Nat,plan: RoundPlan<Element,Context,width>) -> Tree.Tree<Element>:
  RoundPlan{count,context,original,depth,capacity,capacity_equal,balanced,depth_ok,capacity_ok,room} = plan
  original

def round_update(~Element: Data,~Context: Data,~State: Data,~width: Nat,~step: U32 -> Context -> State -> State,index: U32,plan: RoundPlan<Element,Context,width>,state: State) -> State:
  step(index,round_context(~Element,~Context,~width,plan),state)

def round_frame(~Element: Data,~Context: Data,~State: Data,~MapContext: Data,~width: Nat,
  ~output: State -> Tree.Tree<Element>,~map_context: Context -> MapContext,~map: MapContext -> U32 -> Element -> Element,~Frame: Context -> State -> Data,
  plan: RoundPlan<Element,Context,width>,index: Nat,state: State) -> Data:
  Shape.Both<Frame(round_context(~Element,~Context,~width,plan),state),
    {output(state) == MapReadback.run_indexed(~Element,~MapContext,~map,Nat.mul(index,width),0,map_context(round_context(~Element,~Context,~width,plan)),round_original(~Element,~Context,~width,plan)) : Tree.Tree<Element>}>

def round_frame_step(~Element: Data,~Context: Data,~State: Data,~MapContext: Data,~width: Nat,
  ~output: State -> Tree.Tree<Element>,~map_context: Context -> MapContext,~map: MapContext -> U32 -> Element -> Element,~Frame: Context -> State -> Data,
  ~step: U32 -> Context -> State -> State,
  ~frame_step: @+context: Context -> @+index: U32 -> @+state: State -> Frame(context,state) -> Frame(context,step(index,context,state)),
  ~batch_matches: @+context: Context -> @+index: Nat -> @+state: State -> @+depth: Nat -> @+capacity: U32 -> Frame(context,state) ->
    {Tree.capacity(Element,output(state)) == capacity : U32} -> Shape.Balanced(Element,depth,output(state)) ->
    {Nat.is_le(depth,31n) == True{} : Bool} -> {U32.is_le(capacity,2147483648) == True{} : Bool} ->
    {Nat.is_le(Nat.add(Nat.mul(index,width),width),U32.to_nat(capacity)) == True{} : Bool} ->
    {output(step(U32.from_nat(index),context,state)) == MapReadback.run_indexed(~Element,~MapContext,~map,width,U32.from_nat(Nat.mul(index,width)),map_context(context),output(state)) : Tree.Tree<Element>},
  +plan: RoundPlan<Element,Context,width>,+index: Nat,+state: State,
  inside: {Nat.is_lt(index,round_count(~Element,~Context,~width,plan)) == True{} : Bool},
  progress: round_frame(~Element,~Context,~State,~MapContext,~width,~output,~map_context,~map,~Frame,plan,index,state)) ->
  round_frame(~Element,~Context,~State,~MapContext,~width,~output,~map_context,~map,~Frame,plan,1n+index,round_update(~Element,~Context,~State,~width,~step,U32.from_nat(index),plan,state)):
  RoundPlan{+count,+context,+original,+depth,+capacity,+capacity_equal,+balanced,+depth_ok,+capacity_ok,+room} = plan
  Shape.Both{+frame,+mapped} = progress
  +done = Nat.mul(index,width)
  +values = MapReadback.run_indexed(~Element,~MapContext,~map,done,0,map_context(context),original)
  +batch_room = {
    %Order.commutative(width,done) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    Order.transitive(Nat.mul(1n+index,width),Nat.mul(count,width),U32.to_nat(capacity),
      RowMajor.mul_le(1n+index,count,width,Order.strict_follow(index,count,inside)),room)
    : {Nat.is_le(Nat.add(done,width),U32.to_nat(capacity)) == True{} : Bool}}
  +equal_capacity = Equal.trans(U32,Tree.capacity(Element,output(state)),Tree.capacity(Element,values),capacity,
    Equal.cong(Tree.Tree<Element>,U32,tree => Tree.capacity(Element,tree),output(state),values,mapped),
    Equal.trans(U32,Tree.capacity(Element,values),Tree.capacity(Element,original),capacity,MapReadback.capacity_indexed(~Element,~MapContext,~map,done,0,map_context(context),original),capacity_equal))
  +balance = {
    %Equal.sym(Tree.Tree<Element>,output(state),values,mapped) : Shape.Balanced(Element,depth,_)
    MapReadback.balanced_indexed(~Element,~MapContext,~map,done,0,map_context(context),original,depth,balanced)
    : Shape.Balanced(Element,depth,output(state))}
  +one = batch_matches(context,index,state,depth,capacity,frame,equal_capacity,balance,depth_ok,capacity_ok,batch_room)
  Shape.Both{frame_step(context,U32.from_nat(index),state,frame),
    %Order.commutative(done,width) : {output(step(U32.from_nat(index),context,state)) == MapReadback.run_indexed(~Element,~MapContext,~map,_,0,map_context(context),original) : Tree.Tree<Element>}
    Equal.trans(Tree.Tree<Element>,output(step(U32.from_nat(index),context,state)),
      MapReadback.run_indexed(~Element,~MapContext,~map,width,U32.from_nat(done),map_context(context),output(state)),
      MapReadback.run_indexed(~Element,~MapContext,~map,Nat.add(done,width),0,map_context(context),original),one,
      Equal.trans(Tree.Tree<Element>,MapReadback.run_indexed(~Element,~MapContext,~map,width,U32.from_nat(done),map_context(context),output(state)),
        MapReadback.run_indexed(~Element,~MapContext,~map,width,U32.from_nat(done),map_context(context),values),
        MapReadback.run_indexed(~Element,~MapContext,~map,Nat.add(done,width),0,map_context(context),original),
        Equal.cong(Tree.Tree<Element>,Tree.Tree<Element>,tree => MapReadback.run_indexed(~Element,~MapContext,~map,width,U32.from_nat(done),map_context(context),tree),output(state),values,mapped),
        Equal.sym(Tree.Tree<Element>,MapReadback.run_indexed(~Element,~MapContext,~map,Nat.add(done,width),0,map_context(context),original),
          MapReadback.run_indexed(~Element,~MapContext,~map,width,U32.from_nat(done),map_context(context),values),map_split(~Element,~MapContext,~map,done,width,0n,map_context(context),original))))}

def round_same_step(~Element: Data,~Context: Data,~State: Data,~width: Nat,~step: U32 -> Context -> State -> State,
  +index: U32,+plan: RoundPlan<Element,Context,width>,+state: State) ->
  {round_update(~Element,~Context,~State,~width,~step,index,plan,state) == step(index,round_context(~Element,~Context,~width,plan),state) : State}:
  {==}

def fold_mapped(~Element: Data,~Context: Data,~State: Data,~MapContext: Data,~width: Nat,
  ~output: State -> Tree.Tree<Element>,~map_context: Context -> MapContext,~map: MapContext -> U32 -> Element -> Element,~Frame: Context -> State -> Data,
  ~step: U32 -> Context -> State -> State,
  ~frame_step: @+context: Context -> @+index: U32 -> @+state: State -> Frame(context,state) -> Frame(context,step(index,context,state)),
  ~batch_matches: @+context: Context -> @+index: Nat -> @+state: State -> @+depth: Nat -> @+capacity: U32 -> Frame(context,state) ->
    {Tree.capacity(Element,output(state)) == capacity : U32} -> Shape.Balanced(Element,depth,output(state)) ->
    {Nat.is_le(depth,31n) == True{} : Bool} -> {U32.is_le(capacity,2147483648) == True{} : Bool} ->
    {Nat.is_le(Nat.add(Nat.mul(index,width),width),U32.to_nat(capacity)) == True{} : Bool} ->
    {output(step(U32.from_nat(index),context,state)) == MapReadback.run_indexed(~Element,~MapContext,~map,width,U32.from_nat(Nat.mul(index,width)),map_context(context),output(state)) : Tree.Tree<Element>},
  +count: Nat,+context: Context,+state: State,+depth: Nat,+capacity: U32,+frame: Frame(context,state),
  +capacity_equal: {Tree.capacity(Element,output(state)) == capacity : U32},+balanced: Shape.Balanced(Element,depth,output(state)),
  +depth_ok: {Nat.is_le(depth,31n) == True{} : Bool},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(count,width),U32.to_nat(capacity)) == True{} : Bool}) ->
  Shape.Both<Frame(context,Traversal.indexed_fold(~State,~Context,~step,count,0,context,state)),
    {output(Traversal.indexed_fold(~State,~Context,~step,count,0,context,state)) == MapReadback.run_indexed(~Element,~MapContext,~map,Nat.mul(count,width),0,map_context(context),output(state)) : Tree.Tree<Element>}>:
  +plan = {RoundPlan{count,context,output(state),depth,capacity,capacity_equal,balanced,depth_ok,capacity_ok,room} : RoundPlan<Element,Context,width>}
  %ArrayLaws.indexed_fold_change_context(~State,~RoundPlan<Element,Context,width>,~Context,~round_context(~Element,~Context,~width),
    ~round_update(~Element,~Context,~State,~width,~step),~step,~round_same_step(~Element,~Context,~State,~width,~step),count,0,plan,state) :
    Shape.Both<Frame(context,_),{output(_) == MapReadback.run_indexed(~Element,~MapContext,~map,Nat.mul(count,width),0,map_context(context),output(state)) : Tree.Tree<Element>}>
  ArrayLaws.indexed_fold_preserves_invariant(~State,~RoundPlan<Element,Context,width>,~round_count(~Element,~Context,~width),
    ~round_frame(~Element,~Context,~State,~MapContext,~width,~output,~map_context,~map,~Frame),~round_update(~Element,~Context,~State,~width,~step),
    ~round_frame_step(~Element,~Context,~State,~MapContext,~width,~output,~map_context,~map,~Frame,~step,~frame_step,~batch_matches),
    count,0n,plan,state,Order.reflexive(count),Shape.Both{frame,{==}})
