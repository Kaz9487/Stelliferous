# Actual unchecked entry points refine the independent sequential store models.
# Guarded callers discharge range/layout obligations in tensor_operation_bounds;
# tensor_cursor_order identifies contiguous samples with coordinate samples.
import Base
import ../tensor_access.bend as Access
import ../tensor_view.bend as Views
import ../tensor_cursor.bend as Cursor
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./traversal_array_refinement.bend as ArrayLaws
import ./traversal_visit_refinement.bend as Visit
import ./tensor_visit_refinement.bend as TensorVisit
import ./tensor_cursor_refinement.bend as CursorLaws
import ./nat_order.bend as Order
import ./tensor_cursor_model.bend as CursorModel
import ./ordered_fold_refinement.bend as Fold

type ModelResult<-Reader: Data,-Element: Data> is Data:
  ModelResult{reader: Reader,output: Tree.Tree<Element>}

def finish_model(~Reader: Data,~Element: Data,~Context: Data,state: Traversal.State<&2,&2,Reader,Tree.Tree<Element>,Context>) -> ModelResult<Reader,Element>:
  Traversal.State{reader,output,position,context} = state
  ModelResult{reader,output}

def pack_result(~Reader: Type,~Model: Data,~Element: Data,~pack_reader: Model -> Reader,result: ModelResult<Model,Element>) -> Reader & Array<Element>:
  ModelResult{reader,output} = result
  (pack_reader(reader),Tree.pack(Element,output))

def finished(~Reader: Type,~Model: Data,~Element: Data,~Context: Data,~Coordinates: Data,~pack_reader: Model -> Reader,~pack_context: Coordinates -> Context,
  +state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Coordinates>) ->
  {Traversal.finish(~Reader,~Element,~Context,Visit.represent(~Reader,~Model,~Element,~Context,~Coordinates,~pack_reader,~pack_context,state)) ==
    pack_result(~Reader,~Model,~Element,~pack_reader,finish_model(~Model,~Element,~Coordinates,state)) : Reader & Array<Element>}:
  Traversal.State{reader,output,position,context} = state
  {==}

def indexed_finished(~Reader: Type,~Model: Data,~Element: Data,~Context: Data,~pack_reader: Model -> Reader,
  +state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Context>) ->
  {Traversal.finish(~Reader,~Element,~Context,ArrayLaws.represent(~Reader,~Model,~Element,~Context,~pack_reader,state)) ==
    pack_result(~Reader,~Model,~Element,~pack_reader,finish_model(~Model,~Element,~Context,state)) : Reader & Array<Element>}:
  Traversal.State{reader,output,position,context} = state
  {==}

def finish_relation(~Reader: Type,~Model: Data,~Element: Data,~Context: Data,~Coordinates: Data,~pack_reader: Model -> Reader,
  ~represent: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Coordinates> -> Traversal.State<&1,&1,Reader,Array<Element>,Context>,
  ~finishes: @+state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Coordinates> ->
    {Traversal.finish(~Reader,~Element,~Context,represent(state)) == pack_result(~Reader,~Model,~Element,~pack_reader,finish_model(~Model,~Element,~Coordinates,state)) : Reader & Array<Element>},
  -actual: Traversal.State<&1,&1,Reader,Array<Element>,Context>,+state: Traversal.State<&2,&2,Model,Tree.Tree<Element>,Coordinates>,
  equation: {actual == represent(state) : Traversal.State<&1,&1,Reader,Array<Element>,Context>}) ->
  {Traversal.finish(~Reader,~Element,~Context,actual) == pack_result(~Reader,~Model,~Element,~pack_reader,finish_model(~Model,~Element,~Coordinates,state)) : Reader & Array<Element>}:
  Equal.trans(Reader & Array<Element>,Traversal.finish(~Reader,~Element,~Context,actual),
    Traversal.finish(~Reader,~Element,~Context,represent(state)),
    pack_result(~Reader,~Model,~Element,~pack_reader,finish_model(~Model,~Element,~Coordinates,state)),
    Equal.cong(Traversal.State<&1,&1,Reader,Array<Element>,Context>,Reader & Array<Element>,state => Traversal.finish(~Reader,~Element,~Context,state),actual,
      represent(state),equation),finishes(state))

def copy_model(~Element: Data,fast: Bool,count: Nat,source: Views.View,destination: U32,input: Tree.Tree<Element>,output: Tree.Tree<Element>) -> ModelResult<Tree.Tree<Element>,Element>:
  match fast:
    case True{}:
      Views.View{axes,offset} = source
      finish_model(~Tree.Tree<Element>,~Element,~Traversal.Offsets,
        Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Traversal.Offsets>,
          ~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Traversal.Offsets,~ArrayLaws.sample_offset(~Element),~Traversal.destination_offset),count,
          Traversal.State{input,output,0,Traversal.Offsets{offset,destination}}))
    case False{}:
      finish_model(~Tree.Tree<Element>,~Element,~CursorLaws.Coordinates,
        Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,CursorLaws.Coordinates>,
          ~Visit.model_step(~Tree.Tree<Element>,~Element,~CursorLaws.Coordinates,~TensorVisit.sample(~Element),~CursorLaws.next),count,
          Traversal.State{input,output,destination,CursorLaws.initial(source)}))

# A strided operation proceeds in runs of the innermost axis. The lemmas below
# are generic in the two element steps (inside on a stride, last on the whole
# cursor) and in what an element writes: write(current output element, element
# read). Three laws describe one step on packed trees, so each native read and
# write reduces to its model; copy and update supply them further down.
def CursorState(~Element: Data) -> Type:
  Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>

def StrideState(~Element: Data) -> Type:
  Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride>

def cursor_state(~Element: Data,extent: Nat,remaining: Nat,stride: U32,rewind: U32,rest: Cursor.Depth1(),address: U32,
  input: Tree.Tree<Element>,output: Tree.Tree<Element>,position: U32) -> CursorState(~Element):
  Traversal.State{Tree.pack(Element,input),Tree.pack(Element,output),position,Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,remaining,stride,rewind},rest},address}}

def stride_state(~Element: Data,address: U32,stride: U32,input: Tree.Tree<Element>,output: Tree.Tree<Element>,position: U32) -> StrideState(~Element):
  Traversal.State{Tree.pack(Element,input),Tree.pack(Element,output),position,Cursor.Stride{address,stride}}

def run_added(~State: Type,~step: State -> State,first: Nat,second: Nat,-state: State) ->
  {Loop.run(~State,~step,Nat.add(first,second),state) == Loop.run(~State,~step,second,Loop.run(~State,~step,first,state)) : State}:
  match first:
    case 0n: {==}
    case 1n+rest: run_added(~State,~step,rest,second,step(state))

def run_last(~State: Type,~step: State -> State,count: Nat,-state: State) ->
  {Loop.run(~State,~step,1n+count,state) == step(Loop.run(~State,~step,count,state)) : State}:
  match count:
    case 0n: {==}
    case 1n+rest: run_last(~State,~step,rest,step(state))

# The tree after one element: write(current, read) stored at position.
def written(~Element: Data,~write: Element -> Element -> Element,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+position: U32,+address: U32) -> Tree.Tree<Element>:
  Tree.write(Element,output,position,write(Tree.read(Element,output,position),Tree.read(Element,input,address)))

# What one element leaves in the output: stored(input, output, position, address).
def Stored(~Element: Data) -> Type:
  @+input: Tree.Tree<Element> -> @+output: Tree.Tree<Element> -> @+position: U32 -> @+address: U32 -> Tree.Tree<Element>

# Where the elements of a run before its last one leave the output, the
# position and the address.
def stride_output(~Element: Data,~stored: Stored(~Element),inner: Nat,+input: Tree.Tree<Element>,output: Tree.Tree<Element>,+position: U32,+address: U32,+stride: U32) -> Tree.Tree<Element>:
  match inner:
    case 0n: output
    case 1n+rest:
      stride_output(~Element,~stored,rest,input,stored(input,output,position,address),(position + 1 : U32),(address + stride : U32),stride)

def stride_position(inner: Nat,position: U32) -> U32:
  match inner:
    case 0n: position
    case 1n+rest: stride_position(rest,(position + 1 : U32))

def stride_address(inner: Nat,address: U32,+stride: U32) -> U32:
  match inner:
    case 0n: address
    case 1n+rest: stride_address(rest,(address + stride : U32),stride)

# The last step of a run carries into the outer axes and refills the innermost.
def carried(rest: Cursor.Depth1(),address: U32) -> Cursor.Depth1() & U32:
  Cursor.step(~Cursor.Depth0(),~Cursor.advance_outer,rest,address)

def carried_rest(pair: Cursor.Depth1() & U32) -> Cursor.Depth1():
  (rest,address) = pair
  rest

def carried_address(pair: Cursor.Depth1() & U32) -> U32:
  (rest,address) = pair
  address

def moved_reset(+extent: Nat,+stride: U32,+rewind: U32,pair: Cursor.Depth1() & U32) ->
  {Cursor.moved(Cursor.reset(~Cursor.Depth1(),extent,stride,rewind,pair)) ==
    Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,extent,stride,rewind},carried_rest(pair)},carried_address(pair)} : Cursor.Cursor}:
  (rest,address) = pair
  {==}

# The state one full run leaves: a full run again, one carry further.
def run_end(~Element: Data,~stored: Stored(~Element),+inner: Nat,+stride: U32,+rewind: U32,+rest: Cursor.Depth1(),+address: U32,
  +input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+position: U32) -> CursorState(~Element):
  cursor_state(~Element,1n+inner,1n+inner,stride,rewind,carried_rest(carried(rest,(stride_address(inner,address,stride) - rewind : U32))),
    carried_address(carried(rest,(stride_address(inner,address,stride) - rewind : U32))),input,
    stored(input,stride_output(~Element,~stored,inner,input,output,position,address,stride),stride_position(inner,position),stride_address(inner,address,stride)),(stride_position(inner,position) + 1 : U32))

# One element inside a run: the address moves by the stride.
def StrideLaw(~Element: Data,~inside: StrideState(~Element) -> StrideState(~Element),~stored: Stored(~Element)) -> Type:
  @+address: U32 -> @+stride: U32 -> @+input: Tree.Tree<Element> -> @+output: Tree.Tree<Element> -> @+position: U32 ->
    {inside(stride_state(~Element,address,stride,input,output,position)) ==
      stride_state(~Element,(address + stride : U32),stride,input,stored(input,output,position,address),(position + 1 : U32)) : StrideState(~Element)}

# The same element stepped with the whole cursor moves only the innermost axis.
def FreeLaw(~Element: Data,~last: CursorState(~Element) -> CursorState(~Element),~stored: Stored(~Element)) -> Type:
  @+extent: Nat -> @+inner_count: Nat -> @+stride: U32 -> @+rewind: U32 -> @+rest: Cursor.Depth1() -> @+address: U32 -> @+input: Tree.Tree<Element> -> @+output: Tree.Tree<Element> -> @+position: U32 ->
    {last(cursor_state(~Element,extent,2n+inner_count,stride,rewind,rest,address,input,output,position)) ==
      cursor_state(~Element,extent,1n+inner_count,stride,rewind,rest,(address + stride : U32),input,stored(input,output,position,address),(position + 1 : U32)) : CursorState(~Element)}

# The last element of a run carries into the outer axes and refills the innermost.
def CarryLaw(~Element: Data,~last: CursorState(~Element) -> CursorState(~Element),~stored: Stored(~Element)) -> Type:
  @+extent: Nat -> @+stride: U32 -> @+rewind: U32 -> @+rest: Cursor.Depth1() -> @+address: U32 -> @+input: Tree.Tree<Element> -> @+output: Tree.Tree<Element> -> @+position: U32 ->
    {last(cursor_state(~Element,extent,1n,stride,rewind,rest,address,input,output,position)) ==
      cursor_state(~Element,extent,extent,stride,rewind,carried_rest(carried(rest,(address - rewind : U32))),carried_address(carried(rest,(address - rewind : U32))),input,
        stored(input,output,position,address),(position + 1 : U32)) : CursorState(~Element)}

def stride_visited(~Element: Data,~inside: StrideState(~Element) -> StrideState(~Element),~last: CursorState(~Element) -> CursorState(~Element),~stored: Stored(~Element),
  ~stride_law: StrideLaw(~Element,~inside,~stored),~free_law: FreeLaw(~Element,~last,~stored),~carry_law: CarryLaw(~Element,~last,~stored),
  +inner: Nat,+address: U32,+stride: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+position: U32) ->
  {Loop.run(~StrideState(~Element),~inside,inner,stride_state(~Element,address,stride,input,output,position)) ==
    stride_state(~Element,stride_address(inner,address,stride),stride,input,stride_output(~Element,~stored,inner,input,output,position,address,stride),stride_position(inner,position)) : StrideState(~Element)}:
  match inner:
    case 0n: {==}
    case 1n+ +more:
      %Equal.sym(StrideState(~Element),inside(stride_state(~Element,address,stride,input,output,position)),
        stride_state(~Element,(address + stride : U32),stride,input,stored(input,output,position,address),(position + 1 : U32)),
        stride_law(address,stride,input,output,position)) :
        {Loop.run(~StrideState(~Element),~inside,more,_) ==
          stride_state(~Element,stride_address(1n+more,address,stride),stride,input,stride_output(~Element,~stored,1n+more,input,output,position,address,stride),
            stride_position(1n+more,position)) : StrideState(~Element)}
      stride_visited(~Element,~inside,~last,~stored,~stride_law,~free_law,~carry_law,more,(address + stride : U32),stride,input,stored(input,output,position,address),(position + 1 : U32))

def free_visited(~Element: Data,~inside: StrideState(~Element) -> StrideState(~Element),~last: CursorState(~Element) -> CursorState(~Element),~stored: Stored(~Element),
  ~stride_law: StrideLaw(~Element,~inside,~stored),~free_law: FreeLaw(~Element,~last,~stored),~carry_law: CarryLaw(~Element,~last,~stored),
  +inner: Nat,+extent: Nat,+stride: U32,+rewind: U32,+rest: Cursor.Depth1(),+address: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+position: U32) ->
  {Loop.run(~CursorState(~Element),~last,inner,cursor_state(~Element,extent,1n+inner,stride,rewind,rest,address,input,output,position)) ==
    cursor_state(~Element,extent,1n,stride,rewind,rest,stride_address(inner,address,stride),input,stride_output(~Element,~stored,inner,input,output,position,address,stride),stride_position(inner,position)) : CursorState(~Element)}:
  match inner:
    case 0n: {==}
    case 1n+ +more:
      %Equal.sym(CursorState(~Element),last(cursor_state(~Element,extent,2n+more,stride,rewind,rest,address,input,output,position)),
        cursor_state(~Element,extent,1n+more,stride,rewind,rest,(address + stride : U32),input,stored(input,output,position,address),(position + 1 : U32)),
        free_law(extent,more,stride,rewind,rest,address,input,output,position)) :
        {Loop.run(~CursorState(~Element),~last,more,_) ==
          cursor_state(~Element,extent,1n,stride,rewind,rest,stride_address(1n+more,address,stride),input,
            stride_output(~Element,~stored,1n+more,input,output,position,address,stride),stride_position(1n+more,position)) : CursorState(~Element)}
      free_visited(~Element,~inside,~last,~stored,~stride_law,~free_law,~carry_law,more,extent,stride,rewind,rest,(address + stride : U32),input,stored(input,output,position,address),(position + 1 : U32))

def run_closed(~Element: Data,~inside: StrideState(~Element) -> StrideState(~Element),~last: CursorState(~Element) -> CursorState(~Element),~stored: Stored(~Element),
  ~stride_law: StrideLaw(~Element,~inside,~stored),~free_law: FreeLaw(~Element,~last,~stored),~carry_law: CarryLaw(~Element,~last,~stored),
  +inner: Nat,+stride: U32,+rewind: U32,+rest: Cursor.Depth1(),+address: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+position: U32) ->
  {Access.run_step(~Element,~inside,~last,cursor_state(~Element,1n+inner,1n+inner,stride,rewind,rest,address,input,output,position)) ==
    run_end(~Element,~stored,inner,stride,rewind,rest,address,input,output,position) : CursorState(~Element)}:
  %Equal.sym(StrideState(~Element),Loop.run(~StrideState(~Element),~inside,inner,stride_state(~Element,address,stride,input,output,position)),
    stride_state(~Element,stride_address(inner,address,stride),stride,input,stride_output(~Element,~stored,inner,input,output,position,address,stride),stride_position(inner,position)),
    stride_visited(~Element,~inside,~last,~stored,~stride_law,~free_law,~carry_law,inner,address,stride,input,output,position)) :
    {last(Access.last_of_run(~Element,1n+inner,stride,rewind,rest,_)) ==
      run_end(~Element,~stored,inner,stride,rewind,rest,address,input,output,position) : CursorState(~Element)}
  carry_law(1n+inner,stride,rewind,rest,stride_address(inner,address,stride),input,stride_output(~Element,~stored,inner,input,output,position,address,stride),stride_position(inner,position))

def visit_last(~Element: Data,~inside: StrideState(~Element) -> StrideState(~Element),~last: CursorState(~Element) -> CursorState(~Element),~stored: Stored(~Element),
  ~stride_law: StrideLaw(~Element,~inside,~stored),~free_law: FreeLaw(~Element,~last,~stored),~carry_law: CarryLaw(~Element,~last,~stored),
  +inner: Nat,+stride: U32,+rewind: U32,+rest: Cursor.Depth1(),+address: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+position: U32) ->
  {last(Loop.run(~CursorState(~Element),~last,inner,cursor_state(~Element,1n+inner,1n+inner,stride,rewind,rest,address,input,output,position))) ==
    run_end(~Element,~stored,inner,stride,rewind,rest,address,input,output,position) : CursorState(~Element)}:
  %Equal.sym(CursorState(~Element),Loop.run(~CursorState(~Element),~last,inner,cursor_state(~Element,1n+inner,1n+inner,stride,rewind,rest,address,input,output,position)),
    cursor_state(~Element,1n+inner,1n,stride,rewind,rest,stride_address(inner,address,stride),input,stride_output(~Element,~stored,inner,input,output,position,address,stride),stride_position(inner,position)),
    free_visited(~Element,~inside,~last,~stored,~stride_law,~free_law,~carry_law,inner,1n+inner,stride,rewind,rest,address,input,output,position)) :
    {last(_) == run_end(~Element,~stored,inner,stride,rewind,rest,address,input,output,position) : CursorState(~Element)}
  carry_law(1n+inner,stride,rewind,rest,stride_address(inner,address,stride),input,stride_output(~Element,~stored,inner,input,output,position,address,stride),stride_position(inner,position))

def visit_closed(~Element: Data,~inside: StrideState(~Element) -> StrideState(~Element),~last: CursorState(~Element) -> CursorState(~Element),~stored: Stored(~Element),
  ~stride_law: StrideLaw(~Element,~inside,~stored),~free_law: FreeLaw(~Element,~last,~stored),~carry_law: CarryLaw(~Element,~last,~stored),
  +inner: Nat,+stride: U32,+rewind: U32,+rest: Cursor.Depth1(),+address: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+position: U32) ->
  {Loop.run(~CursorState(~Element),~last,1n+inner,cursor_state(~Element,1n+inner,1n+inner,stride,rewind,rest,address,input,output,position)) ==
    run_end(~Element,~stored,inner,stride,rewind,rest,address,input,output,position) : CursorState(~Element)}:
  Equal.trans(CursorState(~Element),Loop.run(~CursorState(~Element),~last,1n+inner,cursor_state(~Element,1n+inner,1n+inner,stride,rewind,rest,address,input,output,position)),
    last(Loop.run(~CursorState(~Element),~last,inner,cursor_state(~Element,1n+inner,1n+inner,stride,rewind,rest,address,input,output,position))),
    run_end(~Element,~stored,inner,stride,rewind,rest,address,input,output,position),
    run_last(~CursorState(~Element),~last,inner,cursor_state(~Element,1n+inner,1n+inner,stride,rewind,rest,address,input,output,position)),
    visit_last(~Element,~inside,~last,~stored,~stride_law,~free_law,~carry_law,inner,stride,rewind,rest,address,input,output,position))

def run_after(~Element: Data,~inside: StrideState(~Element) -> StrideState(~Element),~last: CursorState(~Element) -> CursorState(~Element),~stored: Stored(~Element),
  ~stride_law: StrideLaw(~Element,~inside,~stored),~free_law: FreeLaw(~Element,~last,~stored),~carry_law: CarryLaw(~Element,~last,~stored),
  +more: Nat,+inner: Nat,+stride: U32,+rewind: U32,+rest: Cursor.Depth1(),+address: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+position: U32,
  later: {Loop.run(~CursorState(~Element),~Access.run_step(~Element,~inside,~last),more,run_end(~Element,~stored,inner,stride,rewind,rest,address,input,output,position)) ==
    Loop.run(~CursorState(~Element),~last,Nat.mul(more,1n+inner),run_end(~Element,~stored,inner,stride,rewind,rest,address,input,output,position)) : CursorState(~Element)}) ->
  {Loop.run(~CursorState(~Element),~Access.run_step(~Element,~inside,~last),1n+more,cursor_state(~Element,1n+inner,1n+inner,stride,rewind,rest,address,input,output,position)) ==
    Loop.run(~CursorState(~Element),~last,Nat.mul(more,1n+inner),Loop.run(~CursorState(~Element),~last,1n+inner,cursor_state(~Element,1n+inner,1n+inner,stride,rewind,rest,address,input,output,position))) : CursorState(~Element)}:
  %Equal.sym(CursorState(~Element),Access.run_step(~Element,~inside,~last,cursor_state(~Element,1n+inner,1n+inner,stride,rewind,rest,address,input,output,position)),
    run_end(~Element,~stored,inner,stride,rewind,rest,address,input,output,position),run_closed(~Element,~inside,~last,~stored,~stride_law,~free_law,~carry_law,inner,stride,rewind,rest,address,input,output,position)) :
    {Loop.run(~CursorState(~Element),~Access.run_step(~Element,~inside,~last),more,_) ==
      Loop.run(~CursorState(~Element),~last,Nat.mul(more,1n+inner),Loop.run(~CursorState(~Element),~last,1n+inner,cursor_state(~Element,1n+inner,1n+inner,stride,rewind,rest,address,input,output,position))) : CursorState(~Element)}
  %Equal.sym(CursorState(~Element),Loop.run(~CursorState(~Element),~last,1n+inner,cursor_state(~Element,1n+inner,1n+inner,stride,rewind,rest,address,input,output,position)),
    run_end(~Element,~stored,inner,stride,rewind,rest,address,input,output,position),visit_closed(~Element,~inside,~last,~stored,~stride_law,~free_law,~carry_law,inner,stride,rewind,rest,address,input,output,position)) :
    {Loop.run(~CursorState(~Element),~Access.run_step(~Element,~inside,~last),more,run_end(~Element,~stored,inner,stride,rewind,rest,address,input,output,position)) ==
      Loop.run(~CursorState(~Element),~last,Nat.mul(more,1n+inner),_) : CursorState(~Element)}
  later

# Whole runs from a full run are the element visit of the same elements.
def runs_visit(~Element: Data,~inside: StrideState(~Element) -> StrideState(~Element),~last: CursorState(~Element) -> CursorState(~Element),~stored: Stored(~Element),
  ~stride_law: StrideLaw(~Element,~inside,~stored),~free_law: FreeLaw(~Element,~last,~stored),~carry_law: CarryLaw(~Element,~last,~stored),
  +runs: Nat,+inner: Nat,+stride: U32,+rewind: U32,+rest: Cursor.Depth1(),+address: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+position: U32) ->
  {Loop.run(~CursorState(~Element),~Access.run_step(~Element,~inside,~last),runs,cursor_state(~Element,1n+inner,1n+inner,stride,rewind,rest,address,input,output,position)) ==
    Loop.run(~CursorState(~Element),~last,Nat.mul(runs,1n+inner),cursor_state(~Element,1n+inner,1n+inner,stride,rewind,rest,address,input,output,position)) : CursorState(~Element)}:
  match runs:
    case 0n: {==}
    case 1n+ +more:
      Equal.trans(CursorState(~Element),
        Loop.run(~CursorState(~Element),~Access.run_step(~Element,~inside,~last),1n+more,cursor_state(~Element,1n+inner,1n+inner,stride,rewind,rest,address,input,output,position)),
        Loop.run(~CursorState(~Element),~last,Nat.mul(more,1n+inner),Loop.run(~CursorState(~Element),~last,1n+inner,cursor_state(~Element,1n+inner,1n+inner,stride,rewind,rest,address,input,output,position))),
        Loop.run(~CursorState(~Element),~last,Nat.add(1n+inner,Nat.mul(more,1n+inner)),cursor_state(~Element,1n+inner,1n+inner,stride,rewind,rest,address,input,output,position)),
        run_after(~Element,~inside,~last,~stored,~stride_law,~free_law,~carry_law,more,inner,stride,rewind,rest,address,input,output,position,
          runs_visit(~Element,~inside,~last,~stored,~stride_law,~free_law,~carry_law,more,inner,stride,rewind,carried_rest(carried(rest,(stride_address(inner,address,stride) - rewind : U32))),
            carried_address(carried(rest,(stride_address(inner,address,stride) - rewind : U32))),input,
            stored(input,stride_output(~Element,~stored,inner,input,output,position,address,stride),stride_position(inner,position),stride_address(inner,address,stride)),(stride_position(inner,position) + 1 : U32))),
        Equal.sym(CursorState(~Element),
          Loop.run(~CursorState(~Element),~last,Nat.add(1n+inner,Nat.mul(more,1n+inner)),cursor_state(~Element,1n+inner,1n+inner,stride,rewind,rest,address,input,output,position)),
          Loop.run(~CursorState(~Element),~last,Nat.mul(more,1n+inner),Loop.run(~CursorState(~Element),~last,1n+inner,cursor_state(~Element,1n+inner,1n+inner,stride,rewind,rest,address,input,output,position))),
          run_added(~CursorState(~Element),~last,1n+inner,Nat.mul(more,1n+inner),
            cursor_state(~Element,1n+inner,1n+inner,stride,rewind,rest,address,input,output,position))))

def guarded_visit(~Element: Data,~inside: StrideState(~Element) -> StrideState(~Element),~last: CursorState(~Element) -> CursorState(~Element),~stored: Stored(~Element),
  ~stride_law: StrideLaw(~Element,~inside,~stored),~free_law: FreeLaw(~Element,~last,~stored),~carry_law: CarryLaw(~Element,~last,~stored),
  +full_run: Bool,+whole_runs: Bool,+runs: Nat,+count: Nat,+extent: Nat,+inner: Nat,+stride: U32,+rewind: U32,+rest: Cursor.Depth1(),+address: U32,+destination: U32,
  +input: Tree.Tree<Element>,+output: Tree.Tree<Element>,
  full: {Nat.is_eq(extent,1n+inner) == full_run : Bool},whole: {Nat.is_eq(Nat.mul(runs,1n+inner),count) == whole_runs : Bool}) ->
  {Access.run_guarded(~Element,~inside,~last,full_run,whole_runs,runs,count,Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,1n+inner,stride,rewind},rest},address},destination,Tree.pack(Element,input),Tree.pack(Element,output)) ==
    Traversal.finish(~Array<Element>,~Element,~Cursor.Cursor,Loop.run(~CursorState(~Element),~last,count,cursor_state(~Element,extent,1n+inner,stride,rewind,rest,address,input,output,destination))) : Array<Element> & Array<Element>}:
  match full_run whole_runs:
    case True{} True{}:
      %Equal.sym(Nat,extent,1n+inner,Order.equal_of_is_eq(extent,1n+inner,full)) :
        {Traversal.finish(~Array<Element>,~Element,~Cursor.Cursor,Loop.run(~CursorState(~Element),~Access.run_step(~Element,~inside,~last),runs,cursor_state(~Element,_,1n+inner,stride,rewind,rest,address,input,output,destination))) ==
          Traversal.finish(~Array<Element>,~Element,~Cursor.Cursor,Loop.run(~CursorState(~Element),~last,count,cursor_state(~Element,_,1n+inner,stride,rewind,rest,address,input,output,destination))) : Array<Element> & Array<Element>}
      %Order.equal_of_is_eq(Nat.mul(runs,1n+inner),count,whole) :
        {Traversal.finish(~Array<Element>,~Element,~Cursor.Cursor,Loop.run(~CursorState(~Element),~Access.run_step(~Element,~inside,~last),runs,cursor_state(~Element,1n+inner,1n+inner,stride,rewind,rest,address,input,output,destination))) ==
          Traversal.finish(~Array<Element>,~Element,~Cursor.Cursor,Loop.run(~CursorState(~Element),~last,_,cursor_state(~Element,1n+inner,1n+inner,stride,rewind,rest,address,input,output,destination))) : Array<Element> & Array<Element>}
      Equal.cong(CursorState(~Element),Array<Element> & Array<Element>,state => Traversal.finish(~Array<Element>,~Element,~Cursor.Cursor,state),
        Loop.run(~CursorState(~Element),~Access.run_step(~Element,~inside,~last),runs,cursor_state(~Element,1n+inner,1n+inner,stride,rewind,rest,address,input,output,destination)),
        Loop.run(~CursorState(~Element),~last,Nat.mul(runs,1n+inner),cursor_state(~Element,1n+inner,1n+inner,stride,rewind,rest,address,input,output,destination)),
        runs_visit(~Element,~inside,~last,~stored,~stride_law,~free_law,~carry_law,runs,inner,stride,rewind,rest,address,input,output,destination))
    case True{} False{}: {==}
    case False{} whole_runs: {==}

def axis_visit(~Element: Data,~inside: StrideState(~Element) -> StrideState(~Element),~last: CursorState(~Element) -> CursorState(~Element),~stored: Stored(~Element),
  ~stride_law: StrideLaw(~Element,~inside,~stored),~free_law: FreeLaw(~Element,~last,~stored),~carry_law: CarryLaw(~Element,~last,~stored),
  +remaining: Nat,+extent: Nat,+stride: U32,+rewind: U32,+rest: Cursor.Depth1(),+address: U32,+count: Nat,+destination: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {Access.run_remaining(~Element,~inside,~last,remaining,extent,stride,rewind,rest,address,count,destination,Tree.pack(Element,input),Tree.pack(Element,output)) ==
    Traversal.finish(~Array<Element>,~Element,~Cursor.Cursor,Loop.run(~CursorState(~Element),~last,count,cursor_state(~Element,extent,remaining,stride,rewind,rest,address,input,output,destination))) : Array<Element> & Array<Element>}:
  match remaining:
    case 0n: {==}
    case 1n+ +inner:
      guarded_visit(~Element,~inside,~last,~stored,~stride_law,~free_law,~carry_law,Nat.is_eq(extent,1n+inner),Nat.is_eq(Nat.mul(Nat.div(count,1n+inner),1n+inner),count),Nat.div(count,1n+inner),count,
        extent,inner,stride,rewind,rest,address,destination,input,output,{==},{==})

def strided_visit(~Element: Data,~inside: StrideState(~Element) -> StrideState(~Element),~last: CursorState(~Element) -> CursorState(~Element),~stored: Stored(~Element),
  ~stride_law: StrideLaw(~Element,~inside,~stored),~free_law: FreeLaw(~Element,~last,~stored),~carry_law: CarryLaw(~Element,~last,~stored),
  +count: Nat,cursor: Cursor.Cursor,+destination: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {Access.run_strided(~Element,~inside,~last,count,cursor,destination,Tree.pack(Element,input),Tree.pack(Element,output)) ==
    Traversal.finish(~Array<Element>,~Element,~Cursor.Cursor,Loop.run(~CursorState(~Element),~last,count,Traversal.State{Tree.pack(Element,input),Tree.pack(Element,output),destination,cursor})) : Array<Element> & Array<Element>}:
  Cursor.Cursor{Cursor.Level{Cursor.Axis{+extent,+remaining,+stride,+rewind},+rest},+address} = cursor
  axis_visit(~Element,~inside,~last,~stored,~stride_law,~free_law,~carry_law,remaining,extent,stride,rewind,rest,address,count,destination,input,output)

# Copy: an element writes the value it read.
def second(~Element: Data,current: Element,value: Element) -> Element:
  value

def stride_step(~Element: Data,+address: U32,+stride: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+position: U32) ->
  {Traversal.visit_step(~Array<Element>,~Element,~Cursor.Stride,~Cursor.read_stride(~Element),stride_state(~Element,address,stride,input,output,position)) ==
    stride_state(~Element,(address + stride : U32),stride,input,Tree.write(Element,output,position,Tree.read(Element,input,address)),(position + 1 : U32)) : StrideState(~Element)}:
  %Equal.sym(Array<Element> & Element,Array.get(Element,Tree.pack(Element,input),address),(Tree.pack(Element,input),Tree.read(Element,input,address)),StorageLaws.read(Element,input,address)) :
    {Traversal.visit_store(~Array<Element>,~Element,~Cursor.Stride,position,Tree.pack(Element,output),Cursor.stride_value(~Element,address,stride,_)) ==
      stride_state(~Element,(address + stride : U32),stride,input,Tree.write(Element,output,position,Tree.read(Element,input,address)),(position + 1 : U32)) : StrideState(~Element)}
  %Equal.sym(Array<Element>,Array.set(Element,Tree.pack(Element,output),position,Tree.read(Element,input,address)),Tree.pack(Element,Tree.write(Element,output,position,Tree.read(Element,input,address))),StorageLaws.write(Element,output,position,Tree.read(Element,input,address))) :
    {Traversal.State{Tree.pack(Element,input),_,(position + 1 : U32),Cursor.Stride{(address + stride : U32),stride}} ==
      stride_state(~Element,(address + stride : U32),stride,input,Tree.write(Element,output,position,Tree.read(Element,input,address)),(position + 1 : U32)) : StrideState(~Element)}
  {==}

# A cursor step inside a run moves only the innermost axis.
def free_step(~Element: Data,+extent: Nat,+inner: Nat,+stride: U32,+rewind: U32,+rest: Cursor.Depth1(),+address: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+position: U32) ->
  {Traversal.visit_step(~Array<Element>,~Element,~Cursor.Cursor,~Cursor.read(~Element),cursor_state(~Element,extent,2n+inner,stride,rewind,rest,address,input,output,position)) ==
    cursor_state(~Element,extent,1n+inner,stride,rewind,rest,(address + stride : U32),input,Tree.write(Element,output,position,Tree.read(Element,input,address)),(position + 1 : U32)) : CursorState(~Element)}:
  %Equal.sym(Array<Element> & Element,Array.get(Element,Tree.pack(Element,input),address),(Tree.pack(Element,input),Tree.read(Element,input,address)),StorageLaws.read(Element,input,address)) :
    {Traversal.visit_store(~Array<Element>,~Element,~Cursor.Cursor,position,Tree.pack(Element,output),
        Cursor.read_value(~Element,Cursor.Level{Cursor.Axis{extent,2n+inner,stride,rewind},rest},address,_)) ==
      cursor_state(~Element,extent,1n+inner,stride,rewind,rest,(address + stride : U32),input,Tree.write(Element,output,position,Tree.read(Element,input,address)),(position + 1 : U32)) : CursorState(~Element)}
  %Equal.sym(Array<Element>,Array.set(Element,Tree.pack(Element,output),position,Tree.read(Element,input,address)),Tree.pack(Element,Tree.write(Element,output,position,Tree.read(Element,input,address))),StorageLaws.write(Element,output,position,Tree.read(Element,input,address))) :
    {Traversal.State{Tree.pack(Element,input),_,(position + 1 : U32),
        Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,1n+inner,stride,rewind},rest},(address + stride : U32)}} ==
      cursor_state(~Element,extent,1n+inner,stride,rewind,rest,(address + stride : U32),input,Tree.write(Element,output,position,Tree.read(Element,input,address)),(position + 1 : U32)) : CursorState(~Element)}
  {==}

def carry_step(~Element: Data,+extent: Nat,+stride: U32,+rewind: U32,+rest: Cursor.Depth1(),+address: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+position: U32) ->
  {Traversal.visit_step(~Array<Element>,~Element,~Cursor.Cursor,~Cursor.read(~Element),cursor_state(~Element,extent,1n,stride,rewind,rest,address,input,output,position)) ==
    cursor_state(~Element,extent,extent,stride,rewind,carried_rest(carried(rest,(address - rewind : U32))),carried_address(carried(rest,(address - rewind : U32))),input,
      Tree.write(Element,output,position,Tree.read(Element,input,address)),(position + 1 : U32)) : CursorState(~Element)}:
  %Equal.sym(Array<Element> & Element,Array.get(Element,Tree.pack(Element,input),address),(Tree.pack(Element,input),Tree.read(Element,input,address)),StorageLaws.read(Element,input,address)) :
    {Traversal.visit_store(~Array<Element>,~Element,~Cursor.Cursor,position,Tree.pack(Element,output),
        Cursor.read_value(~Element,Cursor.Level{Cursor.Axis{extent,1n,stride,rewind},rest},address,_)) ==
      cursor_state(~Element,extent,extent,stride,rewind,carried_rest(carried(rest,(address - rewind : U32))),carried_address(carried(rest,(address - rewind : U32))),input,
        Tree.write(Element,output,position,Tree.read(Element,input,address)),(position + 1 : U32)) : CursorState(~Element)}
  %Equal.sym(Array<Element>,Array.set(Element,Tree.pack(Element,output),position,Tree.read(Element,input,address)),Tree.pack(Element,Tree.write(Element,output,position,Tree.read(Element,input,address))),StorageLaws.write(Element,output,position,Tree.read(Element,input,address))) :
    {Traversal.State{Tree.pack(Element,input),_,(position + 1 : U32),
        Cursor.moved(Cursor.reset(~Cursor.Depth1(),extent,stride,rewind,carried(rest,(address - rewind : U32))))} ==
      cursor_state(~Element,extent,extent,stride,rewind,carried_rest(carried(rest,(address - rewind : U32))),carried_address(carried(rest,(address - rewind : U32))),input,
        Tree.write(Element,output,position,Tree.read(Element,input,address)),(position + 1 : U32)) : CursorState(~Element)}
  %Equal.sym(Cursor.Cursor,Cursor.moved(Cursor.reset(~Cursor.Depth1(),extent,stride,rewind,carried(rest,(address - rewind : U32)))),
    Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,extent,stride,rewind},carried_rest(carried(rest,(address - rewind : U32)))},carried_address(carried(rest,(address - rewind : U32)))},
    moved_reset(extent,stride,rewind,carried(rest,(address - rewind : U32)))) :
    {Traversal.State{Tree.pack(Element,input),Tree.pack(Element,Tree.write(Element,output,position,Tree.read(Element,input,address))),(position + 1 : U32),_} ==
      cursor_state(~Element,extent,extent,stride,rewind,carried_rest(carried(rest,(address - rewind : U32))),carried_address(carried(rest,(address - rewind : U32))),input,
        Tree.write(Element,output,position,Tree.read(Element,input,address)),(position + 1 : U32)) : CursorState(~Element)}
  {==}

def copy_strided_visit(~Element: Data,+count: Nat,cursor: Cursor.Cursor,+destination: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {Access.copy_strided(~Element,count,cursor,destination,Tree.pack(Element,input),Tree.pack(Element,output)) ==
    Traversal.finish(~Array<Element>,~Element,~Cursor.Cursor,Traversal.visit(~Array<Element>,~Element,~Cursor.Cursor,~Cursor.read(~Element),count,Traversal.State{Tree.pack(Element,input),Tree.pack(Element,output),destination,cursor})) : Array<Element> & Array<Element>}:
  strided_visit(~Element,~Traversal.visit_step(~Array<Element>,~Element,~Cursor.Stride,~Cursor.read_stride(~Element)),~Traversal.visit_step(~Array<Element>,~Element,~Cursor.Cursor,~Cursor.read(~Element)),~written(~Element,~second(~Element)),
    ~stride_step(~Element),~free_step(~Element),~carry_step(~Element),count,cursor,destination,input,output)

# Update: an element writes operation(current, read).
def update_stride_step(~Element: Data,~operation: Element -> Element -> Element,+address: U32,+stride: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+position: U32) ->
  {Traversal.visit_update_step(~Array<Element>,~Element,~Cursor.Stride,~Cursor.read_stride(~Element),~operation,stride_state(~Element,address,stride,input,output,position)) ==
    stride_state(~Element,(address + stride : U32),stride,input,written(~Element,~operation,input,output,position,address),(position + 1 : U32)) : StrideState(~Element)}:
  %Equal.sym(Array<Element> & Element,Array.get(Element,Tree.pack(Element,output),position),(Tree.pack(Element,output),Tree.read(Element,output,position)),StorageLaws.read(Element,output,position)) :
    {Traversal.visit_update_current(~Array<Element>,~Element,~Cursor.Stride,~Cursor.read_stride(~Element),~operation,position,Cursor.Stride{address,stride},Tree.pack(Element,input),_) ==
      stride_state(~Element,(address + stride : U32),stride,input,written(~Element,~operation,input,output,position,address),(position + 1 : U32)) : StrideState(~Element)}
  %Equal.sym(Array<Element> & Element,Array.get(Element,Tree.pack(Element,input),address),(Tree.pack(Element,input),Tree.read(Element,input,address)),StorageLaws.read(Element,input,address)) :
    {Traversal.visit_update_sample(~Array<Element>,~Element,~Cursor.Stride,~operation,position,Tree.pack(Element,output),Tree.read(Element,output,position),Cursor.stride_value(~Element,address,stride,_)) ==
      stride_state(~Element,(address + stride : U32),stride,input,written(~Element,~operation,input,output,position,address),(position + 1 : U32)) : StrideState(~Element)}
  %Equal.sym(Array<Element>,Array.set(Element,Tree.pack(Element,output),position,operation(Tree.read(Element,output,position),Tree.read(Element,input,address))),Tree.pack(Element,Tree.write(Element,output,position,operation(Tree.read(Element,output,position),Tree.read(Element,input,address)))),StorageLaws.write(Element,output,position,operation(Tree.read(Element,output,position),Tree.read(Element,input,address)))) :
    {Traversal.State{Tree.pack(Element,input),_,(position + 1 : U32),Cursor.Stride{(address + stride : U32),stride}} ==
      stride_state(~Element,(address + stride : U32),stride,input,written(~Element,~operation,input,output,position,address),(position + 1 : U32)) : StrideState(~Element)}
  {==}

def update_free_step(~Element: Data,~operation: Element -> Element -> Element,+extent: Nat,+inner: Nat,+stride: U32,+rewind: U32,+rest: Cursor.Depth1(),+address: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+position: U32) ->
  {Traversal.visit_update_step(~Array<Element>,~Element,~Cursor.Cursor,~Cursor.read(~Element),~operation,cursor_state(~Element,extent,2n+inner,stride,rewind,rest,address,input,output,position)) ==
    cursor_state(~Element,extent,1n+inner,stride,rewind,rest,(address + stride : U32),input,written(~Element,~operation,input,output,position,address),(position + 1 : U32)) : CursorState(~Element)}:
  %Equal.sym(Array<Element> & Element,Array.get(Element,Tree.pack(Element,output),position),(Tree.pack(Element,output),Tree.read(Element,output,position)),StorageLaws.read(Element,output,position)) :
    {Traversal.visit_update_current(~Array<Element>,~Element,~Cursor.Cursor,~Cursor.read(~Element),~operation,position,Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,2n+inner,stride,rewind},rest},address},Tree.pack(Element,input),_) ==
      cursor_state(~Element,extent,1n+inner,stride,rewind,rest,(address + stride : U32),input,written(~Element,~operation,input,output,position,address),(position + 1 : U32)) : CursorState(~Element)}
  %Equal.sym(Array<Element> & Element,Array.get(Element,Tree.pack(Element,input),address),(Tree.pack(Element,input),Tree.read(Element,input,address)),StorageLaws.read(Element,input,address)) :
    {Traversal.visit_update_sample(~Array<Element>,~Element,~Cursor.Cursor,~operation,position,Tree.pack(Element,output),Tree.read(Element,output,position),Cursor.read_value(~Element,Cursor.Level{Cursor.Axis{extent,2n+inner,stride,rewind},rest},address,_)) ==
      cursor_state(~Element,extent,1n+inner,stride,rewind,rest,(address + stride : U32),input,written(~Element,~operation,input,output,position,address),(position + 1 : U32)) : CursorState(~Element)}
  %Equal.sym(Array<Element>,Array.set(Element,Tree.pack(Element,output),position,operation(Tree.read(Element,output,position),Tree.read(Element,input,address))),Tree.pack(Element,Tree.write(Element,output,position,operation(Tree.read(Element,output,position),Tree.read(Element,input,address)))),StorageLaws.write(Element,output,position,operation(Tree.read(Element,output,position),Tree.read(Element,input,address)))) :
    {Traversal.State{Tree.pack(Element,input),_,(position + 1 : U32),
        Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,1n+inner,stride,rewind},rest},(address + stride : U32)}} ==
      cursor_state(~Element,extent,1n+inner,stride,rewind,rest,(address + stride : U32),input,written(~Element,~operation,input,output,position,address),(position + 1 : U32)) : CursorState(~Element)}
  {==}

def update_carry_step(~Element: Data,~operation: Element -> Element -> Element,+extent: Nat,+stride: U32,+rewind: U32,+rest: Cursor.Depth1(),+address: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+position: U32) ->
  {Traversal.visit_update_step(~Array<Element>,~Element,~Cursor.Cursor,~Cursor.read(~Element),~operation,cursor_state(~Element,extent,1n,stride,rewind,rest,address,input,output,position)) ==
    cursor_state(~Element,extent,extent,stride,rewind,carried_rest(carried(rest,(address - rewind : U32))),carried_address(carried(rest,(address - rewind : U32))),input,
      written(~Element,~operation,input,output,position,address),(position + 1 : U32)) : CursorState(~Element)}:
  %Equal.sym(Array<Element> & Element,Array.get(Element,Tree.pack(Element,output),position),(Tree.pack(Element,output),Tree.read(Element,output,position)),StorageLaws.read(Element,output,position)) :
    {Traversal.visit_update_current(~Array<Element>,~Element,~Cursor.Cursor,~Cursor.read(~Element),~operation,position,Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,1n,stride,rewind},rest},address},Tree.pack(Element,input),_) ==
      cursor_state(~Element,extent,extent,stride,rewind,carried_rest(carried(rest,(address - rewind : U32))),carried_address(carried(rest,(address - rewind : U32))),input,
      written(~Element,~operation,input,output,position,address),(position + 1 : U32)) : CursorState(~Element)}
  %Equal.sym(Array<Element> & Element,Array.get(Element,Tree.pack(Element,input),address),(Tree.pack(Element,input),Tree.read(Element,input,address)),StorageLaws.read(Element,input,address)) :
    {Traversal.visit_update_sample(~Array<Element>,~Element,~Cursor.Cursor,~operation,position,Tree.pack(Element,output),Tree.read(Element,output,position),Cursor.read_value(~Element,Cursor.Level{Cursor.Axis{extent,1n,stride,rewind},rest},address,_)) ==
      cursor_state(~Element,extent,extent,stride,rewind,carried_rest(carried(rest,(address - rewind : U32))),carried_address(carried(rest,(address - rewind : U32))),input,
      written(~Element,~operation,input,output,position,address),(position + 1 : U32)) : CursorState(~Element)}
  %Equal.sym(Array<Element>,Array.set(Element,Tree.pack(Element,output),position,operation(Tree.read(Element,output,position),Tree.read(Element,input,address))),Tree.pack(Element,Tree.write(Element,output,position,operation(Tree.read(Element,output,position),Tree.read(Element,input,address)))),StorageLaws.write(Element,output,position,operation(Tree.read(Element,output,position),Tree.read(Element,input,address)))) :
    {Traversal.State{Tree.pack(Element,input),_,(position + 1 : U32),
        Cursor.moved(Cursor.reset(~Cursor.Depth1(),extent,stride,rewind,carried(rest,(address - rewind : U32))))} ==
      cursor_state(~Element,extent,extent,stride,rewind,carried_rest(carried(rest,(address - rewind : U32))),carried_address(carried(rest,(address - rewind : U32))),input,
      written(~Element,~operation,input,output,position,address),(position + 1 : U32)) : CursorState(~Element)}
  %Equal.sym(Cursor.Cursor,Cursor.moved(Cursor.reset(~Cursor.Depth1(),extent,stride,rewind,carried(rest,(address - rewind : U32)))),
    Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,extent,stride,rewind},carried_rest(carried(rest,(address - rewind : U32)))},carried_address(carried(rest,(address - rewind : U32)))},
    moved_reset(extent,stride,rewind,carried(rest,(address - rewind : U32)))) :
    {Traversal.State{Tree.pack(Element,input),Tree.pack(Element,Tree.write(Element,output,position,operation(Tree.read(Element,output,position),Tree.read(Element,input,address)))),(position + 1 : U32),_} ==
      cursor_state(~Element,extent,extent,stride,rewind,carried_rest(carried(rest,(address - rewind : U32))),carried_address(carried(rest,(address - rewind : U32))),input,
      written(~Element,~operation,input,output,position,address),(position + 1 : U32)) : CursorState(~Element)}
  {==}

def update_strided_visit(~Element: Data,~operation: Element -> Element -> Element,+count: Nat,cursor: Cursor.Cursor,+destination: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {Access.update_strided(~Element,~operation,count,cursor,destination,Tree.pack(Element,input),Tree.pack(Element,output)) ==
    Traversal.finish(~Array<Element>,~Element,~Cursor.Cursor,Traversal.visit_update(~Array<Element>,~Element,~Cursor.Cursor,~Cursor.read(~Element),~operation,count,Traversal.State{Tree.pack(Element,input),Tree.pack(Element,output),destination,cursor})) : Array<Element> & Array<Element>}:
  strided_visit(~Element,~Traversal.visit_update_step(~Array<Element>,~Element,~Cursor.Stride,~Cursor.read_stride(~Element),~operation),~Traversal.visit_update_step(~Array<Element>,~Element,~Cursor.Cursor,~Cursor.read(~Element),~operation),~written(~Element,~operation),
    ~update_stride_step(~Element,~operation),~update_free_step(~Element,~operation),~update_carry_step(~Element,~operation),count,cursor,destination,input,output)

def visited_copy(~Element: Data,+count: Nat,+source: Views.View,+destination: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {Traversal.finish(~Array<Element>,~Element,~Cursor.Cursor,Traversal.visit(~Array<Element>,~Element,~Cursor.Cursor,~Cursor.read(~Element),count,Traversal.State{Tree.pack(Element,input),Tree.pack(Element,output),destination,Cursor.prepare(source)})) ==
    pack_result(~Array<Element>,~Tree.Tree<Element>,~Element,~ArrayLaws.pack_tree(~Element),copy_model(~Element,False{},count,source,destination,input,output)) : Array<Element> & Array<Element>}:
  %Equal.sym(Cursor.Cursor,Cursor.prepare(source),CursorLaws.pack(CursorLaws.initial(source)),CursorLaws.initial_matches(source)) :
    {Traversal.finish(~Array<Element>,~Element,~Cursor.Cursor,Traversal.visit(~Array<Element>,~Element,~Cursor.Cursor,~Cursor.read(~Element),count,
      Traversal.State{Tree.pack(Element,input),Tree.pack(Element,output),destination,_})) ==
      pack_result(~Array<Element>,~Tree.Tree<Element>,~Element,~ArrayLaws.pack_tree(~Element),copy_model(~Element,False{},count,source,destination,input,output)) : Array<Element> & Array<Element>}
  finish_relation(~Array<Element>,~Tree.Tree<Element>,~Element,~Cursor.Cursor,~CursorLaws.Coordinates,~ArrayLaws.pack_tree(~Element),
    ~Visit.represent(~Array<Element>,~Tree.Tree<Element>,~Element,~Cursor.Cursor,~CursorLaws.Coordinates,~ArrayLaws.pack_tree(~Element),~CursorLaws.pack),
    ~finished(~Array<Element>,~Tree.Tree<Element>,~Element,~Cursor.Cursor,~CursorLaws.Coordinates,~ArrayLaws.pack_tree(~Element),~CursorLaws.pack),
    Traversal.visit(~Array<Element>,~Element,~Cursor.Cursor,~Cursor.read(~Element),count,
      Traversal.State{Tree.pack(Element,input),Tree.pack(Element,output),destination,CursorLaws.pack(CursorLaws.initial(source))}),
    Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,CursorLaws.Coordinates>,
      ~Visit.model_step(~Tree.Tree<Element>,~Element,~CursorLaws.Coordinates,~TensorVisit.sample(~Element),~CursorLaws.next),count,Traversal.State{input,output,destination,CursorLaws.initial(source)}),
    TensorVisit.visit(~Element,count,Traversal.State{input,output,destination,CursorLaws.initial(source)}))

def copy(~Element: Data,+fast: Bool,+count: Nat,+source: Views.View,+destination: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {Access.copy_run(~Element,fast,count,source,destination,Tree.pack(Element,input),Tree.pack(Element,output)) ==
    pack_result(~Array<Element>,~Tree.Tree<Element>,~Element,~ArrayLaws.pack_tree(~Element),copy_model(~Element,fast,count,source,destination,input,output)) : Array<Element> & Array<Element>}:
  match fast:
    case True{}:
      Views.View{axes,+offset} = source
      finish_relation(~Array<Element>,~Tree.Tree<Element>,~Element,~Traversal.Offsets,~Traversal.Offsets,~ArrayLaws.pack_tree(~Element),
        ~ArrayLaws.represent(~Array<Element>,~Tree.Tree<Element>,~Element,~Traversal.Offsets,~ArrayLaws.pack_tree(~Element)),
        ~indexed_finished(~Array<Element>,~Tree.Tree<Element>,~Element,~Traversal.Offsets,~ArrayLaws.pack_tree(~Element)),
        Traversal.run(~Array<Element>,~Element,~Traversal.Offsets,~Traversal.source_offset(~Element),~Traversal.destination_offset,count,
          Traversal.State{Tree.pack(Element,input),Tree.pack(Element,output),0,Traversal.Offsets{offset,destination}}),
        Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Traversal.Offsets>,
          ~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Traversal.Offsets,~ArrayLaws.sample_offset(~Element),~Traversal.destination_offset),count,Traversal.State{input,output,0,Traversal.Offsets{offset,destination}}),
        ArrayLaws.copy_traversal_refines_storage(~Element,count,Traversal.State{input,output,0,Traversal.Offsets{offset,destination}}))
    case False{}:
      Equal.trans(Array<Element> & Array<Element>,Access.copy_strided(~Element,count,Cursor.prepare(source),destination,Tree.pack(Element,input),Tree.pack(Element,output)),
        Traversal.finish(~Array<Element>,~Element,~Cursor.Cursor,Traversal.visit(~Array<Element>,~Element,~Cursor.Cursor,~Cursor.read(~Element),count,Traversal.State{Tree.pack(Element,input),Tree.pack(Element,output),destination,Cursor.prepare(source)})),
        pack_result(~Array<Element>,~Tree.Tree<Element>,~Element,~ArrayLaws.pack_tree(~Element),copy_model(~Element,False{},count,source,destination,input,output)),
        copy_strided_visit(~Element,count,Cursor.prepare(source),destination,input,output),
        visited_copy(~Element,count,source,destination,input,output))

# Scatter: an element stores the source element at the state position to the
# address of the cursor.
def scattered(~Element: Data,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+position: U32,+address: U32) -> Tree.Tree<Element>:
  Tree.write(Element,output,address,Tree.read(Element,input,position))

def scatter_stride_step(~Element: Data,+address: U32,+stride: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+position: U32) ->
  {Traversal.spread_step(~Element,~Cursor.Stride,~Cursor.target_stride,stride_state(~Element,address,stride,input,output,position)) ==
    stride_state(~Element,(address + stride : U32),stride,input,scattered(~Element,input,output,position,address),(position + 1 : U32)) : StrideState(~Element)}:
  %Equal.sym(Array<Element> & Element,Array.get(Element,Tree.pack(Element,input),position),(Tree.pack(Element,input),Tree.read(Element,input,position)),StorageLaws.read(Element,input,position)) :
    {Traversal.spread_store(~Element,~Cursor.Stride,position,Tree.pack(Element,output),Traversal.Target{Cursor.Stride{(address + stride : U32),stride},address},_) ==
      stride_state(~Element,(address + stride : U32),stride,input,scattered(~Element,input,output,position,address),(position + 1 : U32)) : StrideState(~Element)}
  %Equal.sym(Array<Element>,Array.set(Element,Tree.pack(Element,output),address,Tree.read(Element,input,position)),Tree.pack(Element,Tree.write(Element,output,address,Tree.read(Element,input,position))),StorageLaws.write(Element,output,address,Tree.read(Element,input,position))) :
    {Traversal.State{Tree.pack(Element,input),_,(position + 1 : U32),Cursor.Stride{(address + stride : U32),stride}} ==
      stride_state(~Element,(address + stride : U32),stride,input,scattered(~Element,input,output,position,address),(position + 1 : U32)) : StrideState(~Element)}
  {==}

def scatter_free_step(~Element: Data,+extent: Nat,+inner: Nat,+stride: U32,+rewind: U32,+rest: Cursor.Depth1(),+address: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+position: U32) ->
  {Traversal.spread_step(~Element,~Cursor.Cursor,~Cursor.target,cursor_state(~Element,extent,2n+inner,stride,rewind,rest,address,input,output,position)) ==
    cursor_state(~Element,extent,1n+inner,stride,rewind,rest,(address + stride : U32),input,scattered(~Element,input,output,position,address),(position + 1 : U32)) : CursorState(~Element)}:
  %Equal.sym(Array<Element> & Element,Array.get(Element,Tree.pack(Element,input),position),(Tree.pack(Element,input),Tree.read(Element,input,position)),StorageLaws.read(Element,input,position)) :
    {Traversal.spread_store(~Element,~Cursor.Cursor,position,Tree.pack(Element,output),
        Traversal.Target{Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,1n+inner,stride,rewind},rest},(address + stride : U32)},address},_) ==
      cursor_state(~Element,extent,1n+inner,stride,rewind,rest,(address + stride : U32),input,scattered(~Element,input,output,position,address),(position + 1 : U32)) : CursorState(~Element)}
  %Equal.sym(Array<Element>,Array.set(Element,Tree.pack(Element,output),address,Tree.read(Element,input,position)),Tree.pack(Element,Tree.write(Element,output,address,Tree.read(Element,input,position))),StorageLaws.write(Element,output,address,Tree.read(Element,input,position))) :
    {Traversal.State{Tree.pack(Element,input),_,(position + 1 : U32),
        Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,1n+inner,stride,rewind},rest},(address + stride : U32)}} ==
      cursor_state(~Element,extent,1n+inner,stride,rewind,rest,(address + stride : U32),input,scattered(~Element,input,output,position,address),(position + 1 : U32)) : CursorState(~Element)}
  {==}

def scatter_carry_step(~Element: Data,+extent: Nat,+stride: U32,+rewind: U32,+rest: Cursor.Depth1(),+address: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+position: U32) ->
  {Traversal.spread_step(~Element,~Cursor.Cursor,~Cursor.target,cursor_state(~Element,extent,1n,stride,rewind,rest,address,input,output,position)) ==
    cursor_state(~Element,extent,extent,stride,rewind,carried_rest(carried(rest,(address - rewind : U32))),carried_address(carried(rest,(address - rewind : U32))),input,
        scattered(~Element,input,output,position,address),(position + 1 : U32)) : CursorState(~Element)}:
  %Equal.sym(Array<Element> & Element,Array.get(Element,Tree.pack(Element,input),position),(Tree.pack(Element,input),Tree.read(Element,input,position)),StorageLaws.read(Element,input,position)) :
    {Traversal.spread_store(~Element,~Cursor.Cursor,position,Tree.pack(Element,output),
        Traversal.Target{Cursor.moved(Cursor.reset(~Cursor.Depth1(),extent,stride,rewind,carried(rest,(address - rewind : U32)))),address},_) ==
      cursor_state(~Element,extent,extent,stride,rewind,carried_rest(carried(rest,(address - rewind : U32))),carried_address(carried(rest,(address - rewind : U32))),input,
        scattered(~Element,input,output,position,address),(position + 1 : U32)) : CursorState(~Element)}
  %Equal.sym(Array<Element>,Array.set(Element,Tree.pack(Element,output),address,Tree.read(Element,input,position)),Tree.pack(Element,Tree.write(Element,output,address,Tree.read(Element,input,position))),StorageLaws.write(Element,output,address,Tree.read(Element,input,position))) :
    {Traversal.State{Tree.pack(Element,input),_,(position + 1 : U32),
        Cursor.moved(Cursor.reset(~Cursor.Depth1(),extent,stride,rewind,carried(rest,(address - rewind : U32))))} ==
      cursor_state(~Element,extent,extent,stride,rewind,carried_rest(carried(rest,(address - rewind : U32))),carried_address(carried(rest,(address - rewind : U32))),input,
        scattered(~Element,input,output,position,address),(position + 1 : U32)) : CursorState(~Element)}
  %Equal.sym(Cursor.Cursor,Cursor.moved(Cursor.reset(~Cursor.Depth1(),extent,stride,rewind,carried(rest,(address - rewind : U32)))),
    Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,extent,stride,rewind},carried_rest(carried(rest,(address - rewind : U32)))},carried_address(carried(rest,(address - rewind : U32)))},
    moved_reset(extent,stride,rewind,carried(rest,(address - rewind : U32)))) :
    {Traversal.State{Tree.pack(Element,input),Tree.pack(Element,Tree.write(Element,output,address,Tree.read(Element,input,position))),(position + 1 : U32),_} ==
      cursor_state(~Element,extent,extent,stride,rewind,carried_rest(carried(rest,(address - rewind : U32))),carried_address(carried(rest,(address - rewind : U32))),input,
        scattered(~Element,input,output,position,address),(position + 1 : U32)) : CursorState(~Element)}
  {==}

def scatter_strided_visit(~Element: Data,+count: Nat,cursor: Cursor.Cursor,+source: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {Access.scatter_strided(~Element,count,cursor,source,Tree.pack(Element,input),Tree.pack(Element,output)) ==
    Traversal.finish(~Array<Element>,~Element,~Cursor.Cursor,Loop.run(~CursorState(~Element),~Traversal.spread_step(~Element,~Cursor.Cursor,~Cursor.target),count,Traversal.State{Tree.pack(Element,input),Tree.pack(Element,output),source,cursor})) : Array<Element> & Array<Element>}:
  strided_visit(~Element,~Traversal.spread_step(~Element,~Cursor.Stride,~Cursor.target_stride),~Traversal.spread_step(~Element,~Cursor.Cursor,~Cursor.target),~scattered(~Element),
    ~scatter_stride_step(~Element),~scatter_free_step(~Element),~scatter_carry_step(~Element),count,cursor,source,input,output)

# The element sequence of a scatter: the source from the offset on, each
# element stored at the address of the next coordinates of the target view.
def scatter_model(~Element: Data,count: Nat,target: Views.View,source: U32,input: Tree.Tree<Element>,output: Tree.Tree<Element>) -> ModelResult<Tree.Tree<Element>,Element>:
  finish_model(~Tree.Tree<Element>,~Element,~CursorLaws.Coordinates,
    Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,CursorLaws.Coordinates>,
      ~Visit.model_spread_step(~Element,~CursorLaws.Coordinates,~TensorVisit.address,~CursorLaws.next),count,
      Traversal.State{input,output,source,CursorLaws.initial(target)}))

def visited_scatter(~Element: Data,+count: Nat,+target: Views.View,+source: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {Traversal.finish(~Array<Element>,~Element,~Cursor.Cursor,Loop.run(~CursorState(~Element),~Traversal.spread_step(~Element,~Cursor.Cursor,~Cursor.target),count,Traversal.State{Tree.pack(Element,input),Tree.pack(Element,output),source,Cursor.prepare(target)})) ==
    pack_result(~Array<Element>,~Tree.Tree<Element>,~Element,~ArrayLaws.pack_tree(~Element),scatter_model(~Element,count,target,source,input,output)) : Array<Element> & Array<Element>}:
  %Equal.sym(Cursor.Cursor,Cursor.prepare(target),CursorLaws.pack(CursorLaws.initial(target)),CursorLaws.initial_matches(target)) :
    {Traversal.finish(~Array<Element>,~Element,~Cursor.Cursor,Loop.run(~CursorState(~Element),~Traversal.spread_step(~Element,~Cursor.Cursor,~Cursor.target),count,
      Traversal.State{Tree.pack(Element,input),Tree.pack(Element,output),source,_})) ==
      pack_result(~Array<Element>,~Tree.Tree<Element>,~Element,~ArrayLaws.pack_tree(~Element),scatter_model(~Element,count,target,source,input,output)) : Array<Element> & Array<Element>}
  finish_relation(~Array<Element>,~Tree.Tree<Element>,~Element,~Cursor.Cursor,~CursorLaws.Coordinates,~ArrayLaws.pack_tree(~Element),
    ~Visit.represent(~Array<Element>,~Tree.Tree<Element>,~Element,~Cursor.Cursor,~CursorLaws.Coordinates,~ArrayLaws.pack_tree(~Element),~CursorLaws.pack),
    ~finished(~Array<Element>,~Tree.Tree<Element>,~Element,~Cursor.Cursor,~CursorLaws.Coordinates,~ArrayLaws.pack_tree(~Element),~CursorLaws.pack),
    Loop.run(~CursorState(~Element),~Traversal.spread_step(~Element,~Cursor.Cursor,~Cursor.target),count,
      Traversal.State{Tree.pack(Element,input),Tree.pack(Element,output),source,CursorLaws.pack(CursorLaws.initial(target))}),
    Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,CursorLaws.Coordinates>,
      ~Visit.model_spread_step(~Element,~CursorLaws.Coordinates,~TensorVisit.address,~CursorLaws.next),count,Traversal.State{input,output,source,CursorLaws.initial(target)}),
    TensorVisit.spread(~Element,count,Traversal.State{input,output,source,CursorLaws.initial(target)}))

def scatter(~Element: Data,+count: Nat,+target: Views.View,+source: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {Access.scatter_run(~Element,count,target,source,Tree.pack(Element,input),Tree.pack(Element,output)) ==
    pack_result(~Array<Element>,~Tree.Tree<Element>,~Element,~ArrayLaws.pack_tree(~Element),scatter_model(~Element,count,target,source,input,output)) : Array<Element> & Array<Element>}:
  Equal.trans(Array<Element> & Array<Element>,Access.scatter_strided(~Element,count,Cursor.prepare(target),source,Tree.pack(Element,input),Tree.pack(Element,output)),
    Traversal.finish(~Array<Element>,~Element,~Cursor.Cursor,Loop.run(~CursorState(~Element),~Traversal.spread_step(~Element,~Cursor.Cursor,~Cursor.target),count,Traversal.State{Tree.pack(Element,input),Tree.pack(Element,output),source,Cursor.prepare(target)})),
    pack_result(~Array<Element>,~Tree.Tree<Element>,~Element,~ArrayLaws.pack_tree(~Element),scatter_model(~Element,count,target,source,input,output)),
    scatter_strided_visit(~Element,count,Cursor.prepare(target),source,input,output),
    visited_scatter(~Element,count,target,source,input,output))

def sample_index(~Element: Data,tree: Tree.Tree<Element>,context: Unit,index: U32) -> Element:
  Tree.read(Element,tree,index)

def read_index(~Element: Data,+tree: Tree.Tree<Element>,+context: Unit,+index: U32) ->
  {Traversal.read_index(~Element,Tree.pack(Element,tree),context,index) == (Tree.pack(Element,tree),sample_index(~Element,tree,context,index)) : Array<Element> & Element}:
  StorageLaws.read(Element,tree,index)

def combine_model(~Element: Data,~operation: Element -> Element -> Element,fast: Bool,count: Nat,source: Views.View,destination: U32,
  input: Tree.Tree<Element>,output: Tree.Tree<Element>) -> ModelResult<Tree.Tree<Element>,Element>:
  match fast:
    case True{}:
      finish_model(~Tree.Tree<Element>,~Element,~Unit,
        Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Unit>,
          ~ArrayLaws.model_update_step(~Tree.Tree<Element>,~Element,~Element,~Unit,~sample_index(~Element),~operation,~Traversal.index),count,Traversal.State{input,output,0,Unit{}}))
    case False{}:
      finish_model(~Tree.Tree<Element>,~Element,~CursorLaws.Coordinates,
        Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,CursorLaws.Coordinates>,
          ~Visit.model_update_step(~Tree.Tree<Element>,~Element,~CursorLaws.Coordinates,~TensorVisit.sample(~Element),~CursorLaws.next,~operation),count,
          Traversal.State{input,output,destination,CursorLaws.initial(source)}))

def visited_update(~Element: Data,~operation: Element -> Element -> Element,+count: Nat,+source: Views.View,+destination: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {Traversal.finish(~Array<Element>,~Element,~Cursor.Cursor,Traversal.visit_update(~Array<Element>,~Element,~Cursor.Cursor,~Cursor.read(~Element),~operation,count,Traversal.State{Tree.pack(Element,input),Tree.pack(Element,output),destination,Cursor.prepare(source)})) ==
    pack_result(~Array<Element>,~Tree.Tree<Element>,~Element,~ArrayLaws.pack_tree(~Element),combine_model(~Element,~operation,False{},count,source,destination,input,output)) : Array<Element> & Array<Element>}:
  %Equal.sym(Cursor.Cursor,Cursor.prepare(source),CursorLaws.pack(CursorLaws.initial(source)),CursorLaws.initial_matches(source)) :
    {Traversal.finish(~Array<Element>,~Element,~Cursor.Cursor,Traversal.visit_update(~Array<Element>,~Element,~Cursor.Cursor,~Cursor.read(~Element),~operation,count,
      Traversal.State{Tree.pack(Element,input),Tree.pack(Element,output),destination,_})) ==
      pack_result(~Array<Element>,~Tree.Tree<Element>,~Element,~ArrayLaws.pack_tree(~Element),combine_model(~Element,~operation,False{},count,source,destination,input,output)) : Array<Element> & Array<Element>}
  finish_relation(~Array<Element>,~Tree.Tree<Element>,~Element,~Cursor.Cursor,~CursorLaws.Coordinates,~ArrayLaws.pack_tree(~Element),
    ~Visit.represent(~Array<Element>,~Tree.Tree<Element>,~Element,~Cursor.Cursor,~CursorLaws.Coordinates,~ArrayLaws.pack_tree(~Element),~CursorLaws.pack),
    ~finished(~Array<Element>,~Tree.Tree<Element>,~Element,~Cursor.Cursor,~CursorLaws.Coordinates,~ArrayLaws.pack_tree(~Element),~CursorLaws.pack),
    Traversal.visit_update(~Array<Element>,~Element,~Cursor.Cursor,~Cursor.read(~Element),~operation,count,
      Traversal.State{Tree.pack(Element,input),Tree.pack(Element,output),destination,CursorLaws.pack(CursorLaws.initial(source))}),
    Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,CursorLaws.Coordinates>,
      ~Visit.model_update_step(~Tree.Tree<Element>,~Element,~CursorLaws.Coordinates,~TensorVisit.sample(~Element),~CursorLaws.next,~operation),count,Traversal.State{input,output,destination,CursorLaws.initial(source)}),
    TensorVisit.update(~Element,~operation,count,Traversal.State{input,output,destination,CursorLaws.initial(source)}))

def combine(~Element: Data,~operation: Element -> Element -> Element,+fast: Bool,+count: Nat,+source: Views.View,+destination: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {Access.combine_run(~Element,~operation,fast,count,source,destination,Tree.pack(Element,input),Tree.pack(Element,output)) ==
    pack_result(~Array<Element>,~Tree.Tree<Element>,~Element,~ArrayLaws.pack_tree(~Element),combine_model(~Element,~operation,fast,count,source,destination,input,output)) : Array<Element> & Array<Element>}:
  match fast:
    case True{}:
      finish_relation(~Array<Element>,~Tree.Tree<Element>,~Element,~Unit,~Unit,~ArrayLaws.pack_tree(~Element),
        ~ArrayLaws.represent(~Array<Element>,~Tree.Tree<Element>,~Element,~Unit,~ArrayLaws.pack_tree(~Element)),~indexed_finished(~Array<Element>,~Tree.Tree<Element>,~Element,~Unit,~ArrayLaws.pack_tree(~Element)),
        Traversal.update(~Array<Element>,~Element,~Element,~Unit,~Traversal.read_index(~Element),~operation,~Traversal.index,count,Traversal.State{Tree.pack(Element,input),Tree.pack(Element,output),0,Unit{}}),
        Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Unit>,
          ~ArrayLaws.model_update_step(~Tree.Tree<Element>,~Element,~Element,~Unit,~sample_index(~Element),~operation,~Traversal.index),count,Traversal.State{input,output,0,Unit{}}),
        ArrayLaws.update_refines_model(~Array<Element>,~Tree.Tree<Element>,~Element,~Element,~Unit,~ArrayLaws.pack_tree(~Element),~Traversal.read_index(~Element),~sample_index(~Element),~operation,~Traversal.index,~read_index(~Element),count,Traversal.State{input,output,0,Unit{}}))
    case False{}:
      Equal.trans(Array<Element> & Array<Element>,Access.update_strided(~Element,~operation,count,Cursor.prepare(source),destination,Tree.pack(Element,input),Tree.pack(Element,output)),
        Traversal.finish(~Array<Element>,~Element,~Cursor.Cursor,Traversal.visit_update(~Array<Element>,~Element,~Cursor.Cursor,~Cursor.read(~Element),~operation,count,Traversal.State{Tree.pack(Element,input),Tree.pack(Element,output),destination,Cursor.prepare(source)})),
        pack_result(~Array<Element>,~Tree.Tree<Element>,~Element,~ArrayLaws.pack_tree(~Element),combine_model(~Element,~operation,False{},count,source,destination,input,output)),
        update_strided_visit(~Element,~operation,count,Cursor.prepare(source),destination,input,output),
        visited_update(~Element,~operation,count,source,destination,input,output))

# The reduction of the lines of a view: the coordinates of the next run of
# the view without its last axis, and the runs of a line.
type LineCoordinates<-Accumulator: Data> is Data:
  LineCoordinates{coordinates: CursorLaws.Coordinates,runs: Nat,extent: U32,stride: U32,initial: Accumulator}

def pack_line(~Accumulator: Data,line: LineCoordinates<Accumulator>) -> Access.Lines<Accumulator>:
  LineCoordinates{coordinates,runs,extent,stride,initial} = line
  Access.Lines{CursorLaws.pack(coordinates),runs,extent,stride,initial}

def after_runs(count: Nat,coordinates: CursorLaws.Coordinates) -> CursorLaws.Coordinates:
  Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,count,coordinates)

def next_line(~Accumulator: Data,line: LineCoordinates<Accumulator>) -> LineCoordinates<Accumulator>:
  LineCoordinates{coordinates,+runs,extent,stride,initial} = line
  LineCoordinates{after_runs(runs,coordinates),runs,extent,stride,initial}

# The ordered fold of count runs from the coordinates: each run folds extent
# values stride apart from the address of its coordinates.
def run_values(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,count: Nat,+tree: Tree.Tree<Element>,
  coordinates: CursorLaws.Coordinates,+extent: U32,+stride: U32,accumulator: Accumulator) -> Accumulator:
  match count:
    case 0n: accumulator
    case 1n+rest:
      +coordinates = coordinates
      run_values(~Element,~Accumulator,~operation,rest,tree,CursorLaws.next(coordinates),extent,stride,
        Fold.values(~Tree.Tree<Element>,~Element,~Accumulator,~Traversal.Strided,~ArrayLaws.sample_strided(~Element),~operation,U32.to_nat(extent),0,tree,
          Traversal.Strided{TensorVisit.address(coordinates),stride},accumulator))

def run_position(count: Nat,+position: U32) -> U32:
  match count:
    case 0n: position
    case 1n+rest: run_position(rest,(position + 1 : U32))

def sample_line(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  tree: Tree.Tree<Element>,line: LineCoordinates<Accumulator>) -> Element:
  LineCoordinates{coordinates,runs,extent,stride,initial} = line
  finish(run_values(~Element,~Accumulator,~operation,runs,tree,coordinates,extent,stride,initial))

# The runs of a line fold the model's runs and move the cursor past them.
def runs_match(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,count: Nat,+tree: Tree.Tree<Element>,
  +coordinates: CursorLaws.Coordinates,+extent: U32,+stride: U32,+accumulator: Accumulator,+position: U32) ->
  {Loop.run(~Traversal.State<&1,&2,Array<Element>,Accumulator,Access.Runs>,~Access.fold_run(~Element,~Accumulator,~operation),count,
      Traversal.State{Tree.pack(Element,tree),accumulator,position,Access.Runs{CursorLaws.pack(coordinates),extent,stride}}) ==
    Traversal.State{Tree.pack(Element,tree),run_values(~Element,~Accumulator,~operation,count,tree,coordinates,extent,stride,accumulator),run_position(count,position),
      Access.Runs{CursorLaws.pack(after_runs(count,coordinates)),extent,stride}} : Traversal.State<&1,&2,Array<Element>,Accumulator,Access.Runs>}:
  match count:
    case 0n: {==}
    case 1n+rest:
      CursorLaws.Coordinates{+axes,+offset,+valid} = coordinates
      +address = U32.from_nat(CursorModel.address(axes,offset))
      +values = Fold.values(~Tree.Tree<Element>,~Element,~Accumulator,~Traversal.Strided,~ArrayLaws.sample_strided(~Element),~operation,U32.to_nat(extent),0,tree,
        Traversal.Strided{address,stride},accumulator)
      +next = CursorLaws.next(CursorLaws.Coordinates{axes,offset,valid})
      +stepped = {
        %Equal.sym(Array<Element> & Accumulator,
          Traversal.fold_rounds(~Array<Element>,~Element,~Accumulator,~Traversal.Strided,~Traversal.strided(~Element),~operation,U32.to_nat(extent),0,Tree.pack(Element,tree),accumulator,
            Traversal.Strided{address,stride}),
          (Tree.pack(Element,tree),values),
          Fold.rounds(~Array<Element>,~Tree.Tree<Element>,~Element,~Accumulator,~Traversal.Strided,~ArrayLaws.pack_tree(~Element),~Traversal.strided(~Element),
            ~ArrayLaws.sample_strided(~Element),~operation,~ArrayLaws.strided_read_refines_storage(~Element),U32.to_nat(extent),0,tree,Traversal.Strided{address,stride},accumulator)) :
          {Access.ran(~Element,~Accumulator,CursorLaws.represent_axes(axes),address,extent,stride,position,_) ==
            Traversal.State{Tree.pack(Element,tree),values,(position + 1 : U32),Access.Runs{CursorLaws.pack(next),extent,stride}} : Traversal.State<&1,&2,Array<Element>,Accumulator,Access.Runs>}
        %Equal.sym(Cursor.Cursor,Cursor.advance(CursorLaws.represent_axes(axes),address),CursorLaws.represent(CursorModel.advance(axes),offset),
          CursorLaws.advance(axes,offset,valid)) :
          {Traversal.State{Tree.pack(Element,tree),values,(position + 1 : U32),Access.Runs{_,extent,stride}} ==
            Traversal.State{Tree.pack(Element,tree),values,(position + 1 : U32),Access.Runs{CursorLaws.pack(next),extent,stride}} : Traversal.State<&1,&2,Array<Element>,Accumulator,Access.Runs>}
        {==} :
        {Access.fold_run(~Element,~Accumulator,~operation,Traversal.State{Tree.pack(Element,tree),accumulator,position,Access.Runs{CursorLaws.pack(CursorLaws.Coordinates{axes,offset,valid}),extent,stride}}) ==
          Traversal.State{Tree.pack(Element,tree),values,(position + 1 : U32),Access.Runs{CursorLaws.pack(next),extent,stride}} : Traversal.State<&1,&2,Array<Element>,Accumulator,Access.Runs>}}
      %Equal.sym(Traversal.State<&1,&2,Array<Element>,Accumulator,Access.Runs>,
          Access.fold_run(~Element,~Accumulator,~operation,Traversal.State{Tree.pack(Element,tree),accumulator,position,Access.Runs{CursorLaws.pack(CursorLaws.Coordinates{axes,offset,valid}),extent,stride}}),
          Traversal.State{Tree.pack(Element,tree),values,(position + 1 : U32),Access.Runs{CursorLaws.pack(next),extent,stride}},stepped) :
        {Loop.run(~Traversal.State<&1,&2,Array<Element>,Accumulator,Access.Runs>,~Access.fold_run(~Element,~Accumulator,~operation),rest,_) ==
          Traversal.State{Tree.pack(Element,tree),run_values(~Element,~Accumulator,~operation,1n+rest,tree,CursorLaws.Coordinates{axes,offset,valid},extent,stride,accumulator),
            run_position(1n+rest,position),Access.Runs{CursorLaws.pack(after_runs(1n+rest,CursorLaws.Coordinates{axes,offset,valid})),extent,stride}} :
          Traversal.State<&1,&2,Array<Element>,Accumulator,Access.Runs>}
      runs_match(~Element,~Accumulator,~operation,rest,tree,next,extent,stride,values,(position + 1 : U32))

def line_matches(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +tree: Tree.Tree<Element>,+line: LineCoordinates<Accumulator>) ->
  {Access.fold_line(~Element,~Accumulator,~operation,~finish,Tree.pack(Element,tree),pack_line(~Accumulator,line)) ==
    (Tree.pack(Element,tree),Traversal.Visited{pack_line(~Accumulator,next_line(~Accumulator,line)),sample_line(~Element,~Accumulator,~operation,~finish,tree,line)}) :
    Array<Element> & Traversal.Visited<Access.Lines<Accumulator>,Element>}:
  LineCoordinates{+coordinates,+runs,+extent,+stride,+initial} = line
  %Equal.sym(Traversal.State<&1,&2,Array<Element>,Accumulator,Access.Runs>,
      Loop.run(~Traversal.State<&1,&2,Array<Element>,Accumulator,Access.Runs>,~Access.fold_run(~Element,~Accumulator,~operation),runs,
        Traversal.State{Tree.pack(Element,tree),initial,0,Access.Runs{CursorLaws.pack(coordinates),extent,stride}}),
      Traversal.State{Tree.pack(Element,tree),run_values(~Element,~Accumulator,~operation,runs,tree,coordinates,extent,stride,initial),run_position(runs,0),
        Access.Runs{CursorLaws.pack(after_runs(runs,coordinates)),extent,stride}},
      runs_match(~Element,~Accumulator,~operation,runs,tree,coordinates,extent,stride,initial,0)) :
    {Access.folded_line(~Element,~Accumulator,~finish,runs,initial,_) ==
      (Tree.pack(Element,tree),Traversal.Visited{pack_line(~Accumulator,next_line(~Accumulator,LineCoordinates{coordinates,runs,extent,stride,initial})),
        sample_line(~Element,~Accumulator,~operation,~finish,tree,LineCoordinates{coordinates,runs,extent,stride,initial})}) :
      Array<Element> & Traversal.Visited<Access.Lines<Accumulator>,Element>}
  {==}

def lines_model(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  count: Nat,view: Views.View,runs: Nat,extent: U32,stride: U32,initial: Accumulator,input: Tree.Tree<Element>,output: Tree.Tree<Element>) -> ModelResult<Tree.Tree<Element>,Element>:
  finish_model(~Tree.Tree<Element>,~Element,~LineCoordinates<Accumulator>,
    Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,LineCoordinates<Accumulator>>,
      ~Visit.model_step(~Tree.Tree<Element>,~Element,~LineCoordinates<Accumulator>,~sample_line(~Element,~Accumulator,~operation,~finish),~next_line(~Accumulator)),count,
      Traversal.State{input,output,0,LineCoordinates{CursorLaws.initial(view),runs,extent,stride,initial}}))

def lines(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +count: Nat,+view: Views.View,+runs: Nat,+extent: U32,+stride: U32,+initial: Accumulator,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {Access.reduce_lines(~Element,~Accumulator,~operation,~finish,count,view,runs,extent,stride,initial,Tree.pack(Element,input),Tree.pack(Element,output)) ==
    pack_result(~Array<Element>,~Tree.Tree<Element>,~Element,~ArrayLaws.pack_tree(~Element),lines_model(~Element,~Accumulator,~operation,~finish,count,view,runs,extent,stride,initial,input,output)) :
    Array<Element> & Array<Element>}:
  +line = {LineCoordinates{CursorLaws.initial(view),runs,extent,stride,initial} : LineCoordinates<Accumulator>}
  %Equal.sym(Cursor.Cursor,Cursor.prepare(view),CursorLaws.pack(CursorLaws.initial(view)),CursorLaws.initial_matches(view)) :
    {Traversal.finish(~Array<Element>,~Element,~Access.Lines<Accumulator>,
      Traversal.visit(~Array<Element>,~Element,~Access.Lines<Accumulator>,~Access.fold_line(~Element,~Accumulator,~operation,~finish),count,
        Traversal.State{Tree.pack(Element,input),Tree.pack(Element,output),0,Access.Lines{_,runs,extent,stride,initial}})) ==
      pack_result(~Array<Element>,~Tree.Tree<Element>,~Element,~ArrayLaws.pack_tree(~Element),lines_model(~Element,~Accumulator,~operation,~finish,count,view,runs,extent,stride,initial,input,output)) :
      Array<Element> & Array<Element>}
  finish_relation(~Array<Element>,~Tree.Tree<Element>,~Element,~Access.Lines<Accumulator>,~LineCoordinates<Accumulator>,~ArrayLaws.pack_tree(~Element),
    ~Visit.represent(~Array<Element>,~Tree.Tree<Element>,~Element,~Access.Lines<Accumulator>,~LineCoordinates<Accumulator>,~ArrayLaws.pack_tree(~Element),~pack_line(~Accumulator)),
    ~finished(~Array<Element>,~Tree.Tree<Element>,~Element,~Access.Lines<Accumulator>,~LineCoordinates<Accumulator>,~ArrayLaws.pack_tree(~Element),~pack_line(~Accumulator)),
    Traversal.visit(~Array<Element>,~Element,~Access.Lines<Accumulator>,~Access.fold_line(~Element,~Accumulator,~operation,~finish),count,
      Traversal.State{Tree.pack(Element,input),Tree.pack(Element,output),0,pack_line(~Accumulator,line)}),
    Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,LineCoordinates<Accumulator>>,
      ~Visit.model_step(~Tree.Tree<Element>,~Element,~LineCoordinates<Accumulator>,~sample_line(~Element,~Accumulator,~operation,~finish),~next_line(~Accumulator)),count,
      Traversal.State{input,output,0,line}),
    Visit.visit(~Array<Element>,~Tree.Tree<Element>,~Element,~Access.Lines<Accumulator>,~LineCoordinates<Accumulator>,~ArrayLaws.pack_tree(~Element),~pack_line(~Accumulator),
      ~Access.fold_line(~Element,~Accumulator,~operation,~finish),~sample_line(~Element,~Accumulator,~operation,~finish),~next_line(~Accumulator),
      ~line_matches(~Element,~Accumulator,~operation,~finish),count,Traversal.State{input,output,0,line}))

