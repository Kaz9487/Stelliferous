# Values of the checked scatter. The run of a scatter is a write program: the
# source element at position i is stored at the address of the i-th
# coordinates of the target view, in row-major order. A storage index then
# holds the element of the last coordinates that address it, or its previous
# element when none does.
import Base
import ./boolean_logic.bend as Boolean
import ./tensor_access_refinement.bend as AccessLaws
import ./nat_to_u32_bounds.bend as Bounds
import ./storage_buffer_model.bend as Buffer
import ./storage_certificate.bend as Certificate
import ./u32_comparison.bend as Comparison
import ./tensor_cursor_refinement.bend as CursorLaws
import ./tensor_copy_values.bend as CopyValues
import ./view_writes.bend as ViewWrites
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_cursor_model.bend as CursorModel
import ./tensor_contract.bend as Contract
import ./tensor_gather_values.bend as GatherValues
import ../traversal_loop.bend as Loop
import ./sequence_membership.bend as Members
import ../storage_buffer.bend as Storage
import ../tensor_operations.bend as Operations
import ./nat_order.bend as Order
import ./tensor_operations_refinement.bend as OperationsLaws
import ./ordered_ranges.bend as Ranges
import ./write_program_readback.bend as Readback
import ./tensor_visit_refinement.bend as TensorVisit
import ./traversal_visit_refinement.bend as Visit
import ./tensor_shape_refinement.bend as ShapeLaws
import ../tensor_shape.bend as Shapes
import ./storage_shape.bend as Shape
import ../traversal_array.bend as Traversal
import ./storage_tree_model.bend as Tree
import ../tensor_view.bend as Views
import ./write_program.bend as Writes
import ./traversal_batch_refinement.bend as Batch
import ./storage_copy_refinement.bend as CopyModel
import ./traversal_array_refinement.bend as ArrayLaws
import ../tensor_access.bend as Access
import ./tensor_access_certificate.bend as AccessCertificate
import ./tensor_cursor_order.bend as CursorOrder

def scatter_writes(~Element: Data,count: Nat,+position: U32,+coordinates: CursorLaws.Coordinates,+input: Tree.Tree<Element>) -> List<&2,Writes.Assignment<Element>>:
  match count:
    case 0n: []
    case 1n+rest:
      Writes.Put{TensorVisit.address(coordinates),Tree.read(Element,input,position)} <>
        scatter_writes(~Element,rest,(position + 1 : U32),CursorLaws.next(coordinates),input)

def scatter_pair(~Element: Data,count: Nat,+position: U32,+coordinates: CursorLaws.Coordinates,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~CursorLaws.Coordinates,
      Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,CursorLaws.Coordinates>,
        ~Visit.model_spread_step(~Element,~CursorLaws.Coordinates,~TensorVisit.address,~CursorLaws.next),
        count,Traversal.State{input,output,position,coordinates})) ==
    AccessLaws.ModelResult{input,Writes.pure(Element,scatter_writes(~Element,count,position,coordinates,input),output)} : AccessLaws.ModelResult<Tree.Tree<Element>,Element>}:
  match count:
    case 0n: {==}
    case 1n+rest:
      scatter_pair(~Element,rest,(position + 1 : U32),CursorLaws.next(coordinates),input,
        Tree.write(Element,output,TensorVisit.address(coordinates),Tree.read(Element,input,position)))

def scatter_program(~Element: Data,+count: Nat,+position: U32,+coordinates: CursorLaws.Coordinates,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {CopyValues.output_tree(~Element,AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~CursorLaws.Coordinates,
      Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,CursorLaws.Coordinates>,
        ~Visit.model_spread_step(~Element,~CursorLaws.Coordinates,~TensorVisit.address,~CursorLaws.next),
        count,Traversal.State{input,output,position,coordinates}))) ==
    Writes.pure(Element,scatter_writes(~Element,count,position,coordinates,input),output) : Tree.Tree<Element>}:
  Equal.cong(AccessLaws.ModelResult<Tree.Tree<Element>,Element>,Tree.Tree<Element>,CopyValues.output_tree(~Element),
    AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~CursorLaws.Coordinates,Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,CursorLaws.Coordinates>,~Visit.model_spread_step(~Element,~CursorLaws.Coordinates,~TensorVisit.address,~CursorLaws.next),count,Traversal.State{input,output,position,coordinates})),
    AccessLaws.ModelResult{input,Writes.pure(Element,scatter_writes(~Element,count,position,coordinates,input),output)},scatter_pair(~Element,count,position,coordinates,input,output))

# The receipt observes the owners returned by the common certified operator.
def counted(~Element: Data,+count: Nat,+target: Views.View,+source: U32,input: Buffer.Model<Element>,output: Buffer.Model<Element>) -> CopyModel.ActualOwners<Element>:
  CopyModel.actual_model(~Element,Operations.scatter_counted(~Element,count,target,source,Buffer.pack(~Element,input),Buffer.pack(~Element,output)))

# The source element and the part element at a position.
def stored(~Element: Data,+input: Tree.Tree<Element>,+position: Nat) -> Element:
  Tree.read(Element,input,U32.from_nat(position))
def logical(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+part: Part,+position: Nat) -> Element:
  part_value(part,U32.from_nat(position))

# The run of a scatter writes the source through the target view: the
# coordinates after position steps address the view element at position.
def view_writes_at(~Element: Data,count: Nat,+position: Nat,+axes: List<&2,CursorModel.Coordinate>,+valid: CursorLaws.Valid(axes),
  +target_axes: List<&2,Views.Axis>,+offset: U32,+input: Tree.Tree<Element>,
  +placed: {axes == CursorOrder.iterate(position,CursorModel.prepare_axes(target_axes,[])) : List<&2,CursorModel.Coordinate>}) ->
  {scatter_writes(~Element,count,U32.from_nat(position),CursorLaws.Coordinates{axes,U32.to_nat(offset),valid},input) ==
    ViewWrites.writes_from(~Element,~Tree.Tree<Element>,~stored(~Element),count,position,Views.View{target_axes,offset},input) : List<&2,Writes.Assignment<Element>>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +start = CursorModel.prepare_axes(target_axes,[])
      +head = {Writes.Put{CopyValues.address(Views.View{target_axes,offset},position),stored(~Element,input,position)} : Writes.Assignment<Element>}
      +later = ViewWrites.writes_from(~Element,~Tree.Tree<Element>,~stored(~Element),rest,1n+position,Views.View{target_axes,offset},input)
      +next = Equal.trans(List<&2,CursorModel.Coordinate>,CursorModel.advance(axes),CursorModel.advance(CursorOrder.iterate(position,start)),CursorOrder.iterate(1n+position,start),
        Equal.cong(List<&2,CursorModel.Coordinate>,List<&2,CursorModel.Coordinate>,CursorModel.advance,axes,CursorOrder.iterate(position,start),placed),
        CursorOrder.advance_iterate(position,start))
      %Equal.sym(List<&2,CursorModel.Coordinate>,axes,CursorOrder.iterate(position,start),placed) :
        {Writes.Put{U32.from_nat(CursorModel.address(_,U32.to_nat(offset))),stored(~Element,input,position)} <>
          scatter_writes(~Element,rest,(U32.from_nat(position) + 1 : U32),CursorLaws.next(CursorLaws.Coordinates{axes,U32.to_nat(offset),valid}),input) ==
          head <> later : List<&2,Writes.Assignment<Element>>}
      %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.increment_encoded(position)) :
        {head <> scatter_writes(~Element,rest,_,CursorLaws.next(CursorLaws.Coordinates{axes,U32.to_nat(offset),valid}),input) == head <> later : List<&2,Writes.Assignment<Element>>}
      Equal.cong(List<&2,Writes.Assignment<Element>>,List<&2,Writes.Assignment<Element>>,tail => head <> tail,
        scatter_writes(~Element,rest,U32.from_nat(1n+position),CursorLaws.next(CursorLaws.Coordinates{axes,U32.to_nat(offset),valid}),input),later,
        view_writes_at(~Element,rest,1n+position,CursorModel.advance(axes),CursorLaws.advance_valid(axes,valid),target_axes,offset,input,next))
def view_writes(~Element: Data,count: Nat,+target: Views.View,+input: Tree.Tree<Element>) ->
  {scatter_writes(~Element,count,0,CursorLaws.initial(target),input) ==
    ViewWrites.writes(~Element,~Tree.Tree<Element>,~stored(~Element),count,target,input) : List<&2,Writes.Assignment<Element>>}:
  Views.View{+axes,+offset} = target
  view_writes_at(~Element,count,0n,CursorModel.prepare_axes(axes,[]),CursorLaws.prepared_valid(axes,[],Unit{}),axes,offset,input,{==})

# The element a storage index holds after count elements of the part went
# through the target view: the element of the last write at the index, or
# previous.
def scattered(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,count: Nat,+target: Views.View,+part: Part,previous: Element,+query: U32) -> Element:
  Readback.selected(~Element,ViewWrites.writes(~Element,~Part,~logical(~Element,~Part,~part_value),count,target,part),query,previous)

# The source and the part agree below the source length, so they write the
# same elements.
def logical_writes(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,count: Nat,+position: Nat,+target: Views.View,+part: Part,+input: Tree.Tree<Element>,+length: Nat,
  +values: Contract.Values(~Element,~Part,~part_value,part,Buffer.Content{input,length}),
  +room: {Nat.is_le(Nat.add(position,count),length) == True{} : Bool}) ->
  {ViewWrites.writes_from(~Element,~Tree.Tree<Element>,~stored(~Element),count,position,target,input) ==
    ViewWrites.writes_from(~Element,~Part,~logical(~Element,~Part,~part_value),count,position,target,part) : List<&2,Writes.Assignment<Element>>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +address = CopyValues.address(target,position)
      +later = ViewWrites.writes_from(~Element,~Part,~logical(~Element,~Part,~part_value),rest,1n+position,target,part)
      %Equal.sym(Element,Tree.read(Element,input,U32.from_nat(position)),part_value(part,U32.from_nat(position)),
        Contract.read(~Element,~Part,~part_value,part,Buffer.Content{input,length},position,values,Order.room_index(position,rest,length,room))) :
        {Writes.Put{address,_} <> ViewWrites.writes_from(~Element,~Tree.Tree<Element>,~stored(~Element),rest,1n+position,target,input) ==
          Writes.Put{address,part_value(part,U32.from_nat(position))} <> later : List<&2,Writes.Assignment<Element>>}
      Equal.cong(List<&2,Writes.Assignment<Element>>,List<&2,Writes.Assignment<Element>>,tail => Writes.Put{address,part_value(part,U32.from_nat(position))} <> tail,
        ViewWrites.writes_from(~Element,~Tree.Tree<Element>,~stored(~Element),rest,1n+position,target,input),later,
        logical_writes(~Element,~Part,~part_value,rest,1n+position,target,part,input,length,values,Order.advance_room(position,rest,length,room)))

# A view with a positive count has no zero extent; one with count zero stores
# nothing.
def counted_inside(~Element: Data,count: Nat,+target: Views.View,+input: Tree.Tree<Element>,+length: U32,+capacity: U32,
  exact: {ShapeLaws.elements(Views.axis_extents_of(target)) == count : Nat},
  fits: {Views.fits(target,length) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(length),U32.to_nat(capacity)) == True{} : Bool}) ->
  Readback.Inside(~Element,capacity,ViewWrites.writes(~Element,~Tree.Tree<Element>,~stored(~Element),count,target,input)):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      ViewWrites.inside_from(~Element,~Tree.Tree<Element>,~stored(~Element),1n+rest,0n,target,input,length,capacity,
        %CountCompletion.positive_not_zero(Views.axis_extents_of(target),
          %Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents_of(target)),1n+rest,exact) : {Nat.is_lt(0n,_) == True{} : Bool}
          {==}) : {Views.fits_nonempty(_,target,length) == True{} : Bool}
        fits,room)

# The reader of a scatter result: count elements of the part through the
# target view, over what the storage held before.
type Scatter<-Before: Data,-Part: Data> is Data:
  Scatter{before: Before,part: Part,target: Views.View,count: Nat}

def value(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,context: Scatter<Before,Part>,query: U32) -> Element:
  Scatter{before,part,target,count} = context
  +query = query
  scattered(~Element,~Part,~part_value,count,target,part,before_value(before,query),query)

def scatter_content(~Element: Data,+count: U32,+target: Views.View,input: Buffer.Content<Element>,output: Buffer.Content<Element>) -> Buffer.Content<Element>:
  Buffer.Content{input,input_length} = input
  Buffer.Content{output,output_length} = output
  Buffer.Content{Writes.pure(Element,scatter_writes(~Element,U32.to_nat(count),0,CursorLaws.initial(target),input),output),output_length}

# One index of the scatter program.
def cell(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,+before: Before,+part: Part,+target: Views.View,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+count: U32,+index: Nat,
  ready: Buffer.Sized<Element,output>,
  +part_values: Contract.Values(~Element,~Part,~part_value,part,Buffer.content(~Element,input)),
  before_values: Contract.Values(~Element,~Before,~before_value,before,Buffer.content(~Element,output)),
  +count_room: {Nat.is_le(U32.to_nat(count),Buffer.length(~Element,input)) == True{} : Bool},
  exact: {ShapeLaws.elements(Views.axis_extents_of(target)) == U32.to_nat(count) : Nat},
  fits: {Views.fits(target,U32.from_nat(Buffer.length(~Element,output))) == True{} : Bool},
  +inside: {Nat.is_lt(index,Buffer.length(~Element,output)) == True{} : Bool}) ->
  Contract.At(~Element,~Scatter<Before,Part>,~value(~Element,~Before,~Part,~before_value,~part_value),Scatter{before,part,target,U32.to_nat(count)},
    scatter_content(~Element,count,target,Buffer.content(~Element,input),Buffer.content(~Element,output)),index):
  Buffer.Model{+input_tree,+input_length,+input_certificate} = input
  Buffer.Model{+output_tree,+output_length,+output_certificate} = output
  Buffer.Sized{+depth_ok,+room} = ready
  +output = {Buffer.Model{output_tree,output_length,output_certificate} : Buffer.Model<Element>}
  +capacity = Tree.capacity(Element,output_tree)
  +capacity_ok = Buffer.capacity_bound(~Element,output,depth_ok)
  +query = U32.from_nat(index)
  +writes = ViewWrites.writes(~Element,~Tree.Tree<Element>,~stored(~Element),U32.to_nat(count),target,input_tree)
  +index_room = Bounds.nat_trans(index,output_length,U32.to_nat(capacity),inside,room)
  +query_word = {
    %Equal.sym(Bool,U32.is_lt(query,capacity),Nat.is_lt(U32.to_nat(query),U32.to_nat(capacity)),Comparison.lt(query,capacity)) : {_ == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,Bounds.roundtrip(index,capacity,capacity_ok,Bounds.strict_le(index,U32.to_nat(capacity),index_room))) :
      {Nat.is_lt(_,U32.to_nat(capacity)) == True{} : Bool}
    index_room : {U32.is_lt(query,capacity) == True{} : Bool}}
  +length_room = {
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(output_length)),output_length,Bounds.roundtrip(output_length,capacity,capacity_ok,room)) :
      {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    room : {Nat.is_le(U32.to_nat(U32.from_nat(output_length)),U32.to_nat(capacity)) == True{} : Bool}}
  +scattered_value = scattered(~Element,~Part,~part_value,U32.to_nat(count),target,part,before_value(before,query),query)
  %Equal.sym(List<&2,Writes.Assignment<Element>>,scatter_writes(~Element,U32.to_nat(count),0,CursorLaws.initial(target),input_tree),writes,
    view_writes(~Element,U32.to_nat(count),target,input_tree)) :
    {Tree.read(Element,Writes.pure(Element,_,output_tree),query) == scattered_value : Element}
  Equal.trans(Element,Tree.read(Element,Writes.pure(Element,writes,output_tree),query),
    Readback.selected(~Element,writes,query,Tree.read(Element,output_tree,query)),scattered_value,
    Readback.program(~Element,writes,Buffer.depth(~Element,output),output_tree,query,Buffer.balanced(~Element,output),depth_ok,query_word,
      counted_inside(~Element,U32.to_nat(count),target,input_tree,U32.from_nat(output_length),capacity,exact,fits,length_room)),
    %Equal.sym(Element,Tree.read(Element,output_tree,query),before_value(before,query),
      Contract.read(~Element,~Before,~before_value,before,Buffer.Content{output_tree,output_length},index,before_values,inside)) :
      {Readback.selected(~Element,writes,query,_) == scattered_value : Element}
    %Equal.sym(List<&2,Writes.Assignment<Element>>,writes,ViewWrites.writes(~Element,~Part,~logical(~Element,~Part,~part_value),U32.to_nat(count),target,part),
      logical_writes(~Element,~Part,~part_value,U32.to_nat(count),0n,target,part,input_tree,input_length,part_values,count_room)) :
      {Readback.selected(~Element,_,query,before_value(before,query)) == scattered_value : Element}
    {==})

# Every index of the scatter program.
def cells_range(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,remaining: Nat,+position: Nat,+before: Before,+part: Part,+target_axes: List<&2,Views.Axis>,+target_offset: U32,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+count: U32,
  +ready: Buffer.Sized<Element,output>,
  +part_values: Contract.Values(~Element,~Part,~part_value,part,Buffer.content(~Element,input)),
  +before_values: Contract.Values(~Element,~Before,~before_value,before,Buffer.content(~Element,output)),
  +count_room: {Nat.is_le(U32.to_nat(count),Buffer.length(~Element,input)) == True{} : Bool},
  +exact: {ShapeLaws.elements(Views.axis_extents(target_axes)) == U32.to_nat(count) : Nat},
  +fits: {Views.fits(Views.View{target_axes,target_offset},U32.from_nat(Buffer.length(~Element,output))) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(position,remaining),Buffer.length(~Element,output)) == True{} : Bool}) ->
  Members.All(Nat,{index => Contract.At(~Element,~Scatter<Before,Part>,~value(~Element,~Before,~Part,~before_value,~part_value),Scatter{before,part,Views.View{target_axes,target_offset},U32.to_nat(count)},scatter_content(~Element,count,Views.View{target_axes,target_offset},Buffer.content(~Element,input),Buffer.content(~Element,output)),index) : Nat -> Data},Ranges.range(remaining,position)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      Shape.Both{cell(~Element,~Before,~Part,~before_value,~part_value,before,part,Views.View{target_axes,target_offset},input,output,count,position,ready,part_values,before_values,count_room,exact,fits,Order.room_index(position,rest,Buffer.length(~Element,output),room)),
        cells_range(~Element,~Before,~Part,~before_value,~part_value,rest,1n+position,before,part,target_axes,target_offset,input,output,count,ready,part_values,before_values,count_room,exact,fits,Order.advance_room(position,rest,Buffer.length(~Element,output),room))}

def program_length(~Element: Data,+count: U32,+target: Views.View,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>) ->
  {Buffer.content_length(~Element,scatter_content(~Element,count,target,Buffer.content(~Element,input),Buffer.content(~Element,output))) == Buffer.length(~Element,output) : Nat}:
  Buffer.Model{input_tree,input_length,input_certificate} = input
  Buffer.Model{output_tree,output_length,output_certificate} = output
  {==}

def program_ready(~Element: Data,+count: U32,+target: Views.View,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,ready: Buffer.Sized<Element,output>) ->
  Buffer.ContentReady<Element,scatter_content(~Element,count,target,Buffer.content(~Element,input),Buffer.content(~Element,output))>:
  Buffer.Model{+input_tree,input_length,input_certificate} = input
  Buffer.Model{+output_tree,+output_length,+output_certificate} = output
  Writes.content_ready(~Element,scatter_writes(~Element,U32.to_nat(count),0,CursorLaws.initial(target),input_tree),output_tree,output_length,
    Buffer.content_ready(~Element,Buffer.Model{output_tree,output_length,output_certificate},ready))

def program_values(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,+before: Before,+part: Part,+target_axes: List<&2,Views.Axis>,+target_offset: U32,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+count: U32,
  +ready: Buffer.Sized<Element,output>,
  +part_values: Contract.Values(~Element,~Part,~part_value,part,Buffer.content(~Element,input)),
  +before_values: Contract.Values(~Element,~Before,~before_value,before,Buffer.content(~Element,output)),
  +count_room: {Nat.is_le(U32.to_nat(count),Buffer.length(~Element,input)) == True{} : Bool},
  +exact: {ShapeLaws.elements(Views.axis_extents(target_axes)) == U32.to_nat(count) : Nat},
  +fits: {Views.fits(Views.View{target_axes,target_offset},U32.from_nat(Buffer.length(~Element,output))) == True{} : Bool}) ->
  Contract.Values(~Element,~Scatter<Before,Part>,~value(~Element,~Before,~Part,~before_value,~part_value),Scatter{before,part,Views.View{target_axes,target_offset},U32.to_nat(count)},scatter_content(~Element,count,Views.View{target_axes,target_offset},Buffer.content(~Element,input),Buffer.content(~Element,output))):
  %Equal.sym(Nat,Buffer.content_length(~Element,scatter_content(~Element,count,Views.View{target_axes,target_offset},Buffer.content(~Element,input),Buffer.content(~Element,output))),Buffer.length(~Element,output),program_length(~Element,count,Views.View{target_axes,target_offset},input,output)) :
    Members.All(Nat,{index => Contract.At(~Element,~Scatter<Before,Part>,~value(~Element,~Before,~Part,~before_value,~part_value),Scatter{before,part,Views.View{target_axes,target_offset},U32.to_nat(count)},scatter_content(~Element,count,Views.View{target_axes,target_offset},Buffer.content(~Element,input),Buffer.content(~Element,output)),index) : Nat -> Data},Ranges.range(_,0n))
  cells_range(~Element,~Before,~Part,~before_value,~part_value,Buffer.length(~Element,output),0n,before,part,target_axes,target_offset,input,output,count,ready,part_values,before_values,count_room,exact,fits,Order.reflexive(Buffer.length(~Element,output)))

def program_facts(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,+before: Before,+part: Part,+target_axes: List<&2,Views.Axis>,+target_offset: U32,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+count: U32,+target_shape: List<&2,U32>,
  +ready: Buffer.Sized<Element,output>,
  +part_values: Contract.Values(~Element,~Part,~part_value,part,Buffer.content(~Element,input)),
  +before_values: Contract.Values(~Element,~Before,~before_value,before,Buffer.content(~Element,output)),
  +count_room: {Nat.is_le(U32.to_nat(count),Buffer.length(~Element,input)) == True{} : Bool},
  +exact: {ShapeLaws.elements(Views.axis_extents(target_axes)) == U32.to_nat(count) : Nat},
  +fits: {Views.fits(Views.View{target_axes,target_offset},U32.from_nat(Buffer.length(~Element,output))) == True{} : Bool},
  length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(target_shape) : Nat}) ->
  GatherValues.Facts(~Element,~Scatter<Before,Part>,~value(~Element,~Before,~Part,~before_value,~part_value),Scatter{before,part,Views.View{target_axes,target_offset},U32.to_nat(count)},target_shape,scatter_content(~Element,count,Views.View{target_axes,target_offset},Buffer.content(~Element,input),Buffer.content(~Element,output))):
  Shape.Both{
    Contract.WellFormed{
      Equal.trans(Nat,Buffer.content_length(~Element,scatter_content(~Element,count,Views.View{target_axes,target_offset},Buffer.content(~Element,input),Buffer.content(~Element,output))),Buffer.length(~Element,output),ShapeLaws.elements(target_shape),
        program_length(~Element,count,Views.View{target_axes,target_offset},input,output),length_exact),
      program_ready(~Element,count,Views.View{target_axes,target_offset},input,output,ready)},
    program_values(~Element,~Before,~Part,~before_value,~part_value,before,part,target_axes,target_offset,input,output,count,ready,part_values,before_values,count_room,exact,fits)}

# The content of an accepted scatter is its write program.
def scatter_guard(~Element: Data,input: Storage.Buffer<Element>,+source: Views.View,output: Storage.Buffer<Element>,+target: Views.View) -> Bool:
  Storage.Buffer{input,+input_length,input_certificate} = input
  Storage.Buffer{output,+output_length,output_certificate} = output
  Shapes.same(Views.shape(source),Views.shape(target)) && Views.is_contiguous(source) &&
    Operations.valid_buffer(input_length,Storage.capacity(~Element,input,input_certificate),source) &&
    Operations.valid_buffer(output_length,Storage.capacity(~Element,output,output_certificate),target)

def accepted_content(~Element: Data,+count: U32,+source_axes: List<&2,Views.Axis>,+target: Views.View,+input: Tree.Tree<Element>,+input_length: Nat,+output: Tree.Tree<Element>,+output_length: Nat) ->
  {CopyValues.right_content(~Element,OperationsLaws.scatter_model(~Element,True{},Some{count},Views.View{source_axes,0},target,input,input_length,output,output_length)) ==
    scatter_content(~Element,count,target,Buffer.Content{input,input_length},Buffer.Content{output,output_length}) : Buffer.Content<Element>}:
  Equal.trans(Buffer.Content<Element>,
    CopyValues.right_content(~Element,OperationsLaws.scatter_model(~Element,True{},Some{count},Views.View{source_axes,0},target,input,input_length,output,output_length)),
    Buffer.Content{CopyValues.output_tree(~Element,AccessLaws.scatter_model(~Element,U32.to_nat(count),target,0,input,output)),output_length},
    Buffer.Content{Writes.pure(Element,scatter_writes(~Element,U32.to_nat(count),0,CursorLaws.initial(target),input),output),output_length},
    CopyValues.modeled_content(~Element,input_length,output_length,AccessLaws.scatter_model(~Element,U32.to_nat(count),target,0,input,output)),
    Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{tree,output_length},
      CopyValues.output_tree(~Element,AccessLaws.scatter_model(~Element,U32.to_nat(count),target,0,input,output)),
      Writes.pure(Element,scatter_writes(~Element,U32.to_nat(count),0,CursorLaws.initial(target),input),output),
      scatter_program(~Element,U32.to_nat(count),0,CursorLaws.initial(target),input,output)))

def observed_content(~Element: Data,input: Storage.Buffer<Element>,+source_axes: List<&2,Views.Axis>,output: Storage.Buffer<Element>,+target: Views.View,+count: U32,
  guarded: {scatter_guard(~Element,input,Views.View{source_axes,0},output,target) == True{} : Bool},counted: {Views.numel(target) == Some{count} : Maybe<&2,U32>}) ->
  {CopyValues.right_content(~Element,OperationsLaws.scatter_denotation(~Element,input,Views.View{source_axes,0},output,target)) ==
    scatter_content(~Element,count,target,Buffer.contents(~Element,input),Buffer.contents(~Element,output)) : Buffer.Content<Element>}:
  Storage.Buffer{input,+input_length,input_certificate} = input
  Storage.Buffer{output,+output_length,output_certificate} = output
  %Equal.sym(Bool,Shapes.same(Views.shape(Views.View{source_axes,0}),Views.shape(target)) && Views.is_contiguous(Views.View{source_axes,0}) &&
      Operations.valid_buffer(input_length,Storage.capacity(~Element,input,input_certificate),Views.View{source_axes,0}) &&
      Operations.valid_buffer(output_length,Storage.capacity(~Element,output,output_certificate),target),True{},guarded) :
    {CopyValues.right_content(~Element,OperationsLaws.scatter_model(~Element,_,Views.numel(target),Views.View{source_axes,0},target,Tree.reflect(Element,input),input_length,Tree.reflect(Element,output),output_length)) ==
      scatter_content(~Element,count,target,Buffer.Content{Tree.reflect(Element,input),input_length},Buffer.Content{Tree.reflect(Element,output),output_length}) : Buffer.Content<Element>}
  %Equal.sym(Maybe<&2,U32>,Views.numel(target),Some{count},counted) :
    {CopyValues.right_content(~Element,OperationsLaws.scatter_model(~Element,True{},_,Views.View{source_axes,0},target,Tree.reflect(Element,input),input_length,Tree.reflect(Element,output),output_length)) ==
      scatter_content(~Element,count,target,Buffer.Content{Tree.reflect(Element,input),input_length},Buffer.Content{Tree.reflect(Element,output),output_length}) : Buffer.Content<Element>}
  accepted_content(~Element,count,source_axes,target,Tree.reflect(Element,input),input_length,Tree.reflect(Element,output),output_length)

# An accepted scatter model passed its guard.
def model_guard(~Element: Data,valid: Bool,count: Maybe<&2,U32>,+source: Views.View,+target: Views.View,+input: Tree.Tree<Element>,+input_length: Nat,
  +output: Tree.Tree<Element>,+output_length: Nat,
  accepted: {GatherValues.accepted_owners(~Element,OperationsLaws.scatter_model(~Element,valid,count,source,target,input,input_length,output,output_length)) == True{} : Bool}) ->
  {valid == True{} : Bool}:
  match valid count:
    case True{} Some{count}: {==}
    case True{} None{}: {==}
    case False{} Some{count}: accepted
    case False{} None{}: accepted

type Guarded<-Element: Data,-source: Views.View,-target: Views.View,-input: Buffer.Model<Element>,-output: Buffer.Model<Element>> is Data:
  Guarded{guard: {scatter_guard(~Element,Buffer.pack(~Element,input),source,Buffer.pack(~Element,output),target) == True{} : Bool},
    same: {Shapes.same(Views.shape(source),Views.shape(target)) == True{} : Bool},
    fits: {Views.fits(target,U32.from_nat(Buffer.length(~Element,output))) == True{} : Bool}}

def guarded(~Element: Data,+source: Views.View,+target: Views.View,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,
  accepted: {GatherValues.accepted_owners(~Element,OperationsLaws.scatter_denotation(~Element,Buffer.pack(~Element,input),source,Buffer.pack(~Element,output),target)) == True{} : Bool}) ->
  Guarded<Element,source,target,input,output>:
  Buffer.Model{+input_tree,+input_length,Certificate.Certificate{+input_depth,+input_equation}} = input
  Buffer.Model{+output_tree,+output_length,Certificate.Certificate{+output_depth,+output_equation}} = output
  +same = Shapes.same(Views.shape(source),Views.shape(target))
  +contiguous = Views.is_contiguous(source)
  +input_valid = Operations.valid_buffer(input_length,U32.shln(1,input_depth),source)
  +output_valid = Operations.valid_buffer(output_length,U32.shln(1,output_depth),target)
  +truth = model_guard(~Element,(same && contiguous && input_valid && output_valid : Bool),Views.numel(target),source,target,
    Tree.reflect(Element,Tree.pack(Element,input_tree)),input_length,Tree.reflect(Element,Tree.pack(Element,output_tree)),output_length,accepted)
  Guarded{truth,
    Boolean.left(same,contiguous,Boolean.left((same && contiguous : Bool),input_valid,Boolean.left((same && contiguous && input_valid : Bool),output_valid,truth))),
    Boolean.right(Nat.is_le(output_length,U32.to_nat(U32.shln(1,output_depth))),Views.fits(target,U32.from_nat(output_length)),
      Boolean.right((same && contiguous && input_valid : Bool),output_valid,truth))}

def guarded_facts(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,+before: Before,+part: Part,+target_axes: List<&2,Views.Axis>,+target_offset: U32,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+count: U32,+source_axes: List<&2,Views.Axis>,+target_shape: List<&2,U32>,
  +ready: Buffer.Sized<Element,output>,
  +part_values: Contract.Values(~Element,~Part,~part_value,part,Buffer.content(~Element,input)),
  +before_values: Contract.Values(~Element,~Before,~before_value,before,Buffer.content(~Element,output)),
  +source_exact: {ShapeLaws.elements(Views.axis_extents(source_axes)) == Buffer.length(~Element,input) : Nat},
  +exact: {ShapeLaws.elements(Views.axis_extents(target_axes)) == U32.to_nat(count) : Nat},
  +length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(target_shape) : Nat},
  facts: Guarded<Element,Views.View{source_axes,0},Views.View{target_axes,target_offset},input,output>) ->
  GatherValues.Facts(~Element,~Scatter<Before,Part>,~value(~Element,~Before,~Part,~before_value,~part_value),Scatter{before,part,Views.View{target_axes,target_offset},U32.to_nat(count)},target_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.scatter(~Element,Buffer.pack(~Element,input),Views.View{source_axes,0},Buffer.pack(~Element,output),Views.View{target_axes,target_offset})))):
  Guarded{+guard,same,+fits} = facts
  +count_room = {
    %exact : {Nat.is_le(_,Buffer.length(~Element,input)) == True{} : Bool}
    %ShapeLaws.same_extents(Views.axis_extents(source_axes),Views.axis_extents(target_axes),same) : {Nat.is_le(ShapeLaws.elements(_),Buffer.length(~Element,input)) == True{} : Bool}
    %Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(source_axes)),Buffer.length(~Element,input),source_exact) : {Nat.is_le(_,Buffer.length(~Element,input)) == True{} : Bool}
    Order.reflexive(Buffer.length(~Element,input)) : {Nat.is_le(U32.to_nat(count),Buffer.length(~Element,input)) == True{} : Bool}}
  %Equal.sym(Operations.Outcome<OperationsLaws.PairOwners(Element)>,OperationsLaws.observe_pair_outcome(~Element,Operations.scatter(~Element,Buffer.pack(~Element,input),Views.View{source_axes,0},Buffer.pack(~Element,output),Views.View{target_axes,target_offset})),OperationsLaws.scatter_denotation(~Element,Buffer.pack(~Element,input),Views.View{source_axes,0},Buffer.pack(~Element,output),Views.View{target_axes,target_offset}),
    OperationsLaws.scatter(~Element,Buffer.pack(~Element,input),Views.View{source_axes,0},Buffer.pack(~Element,output),Views.View{target_axes,target_offset})) :
    GatherValues.Facts(~Element,~Scatter<Before,Part>,~value(~Element,~Before,~Part,~before_value,~part_value),Scatter{before,part,Views.View{target_axes,target_offset},U32.to_nat(count)},target_shape,CopyValues.right_content(~Element,_))
  %Equal.sym(Buffer.Content<Element>,CopyValues.right_content(~Element,OperationsLaws.scatter_denotation(~Element,Buffer.pack(~Element,input),Views.View{source_axes,0},Buffer.pack(~Element,output),Views.View{target_axes,target_offset})),
    scatter_content(~Element,count,Views.View{target_axes,target_offset},Buffer.contents(~Element,Buffer.pack(~Element,input)),Buffer.contents(~Element,Buffer.pack(~Element,output))),
    observed_content(~Element,Buffer.pack(~Element,input),source_axes,Buffer.pack(~Element,output),Views.View{target_axes,target_offset},count,guard,CountCompletion.numel(Views.axis_extents(target_axes),count,exact))) :
    GatherValues.Facts(~Element,~Scatter<Before,Part>,~value(~Element,~Before,~Part,~before_value,~part_value),Scatter{before,part,Views.View{target_axes,target_offset},U32.to_nat(count)},target_shape,_)
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,input)),Buffer.content(~Element,input),Buffer.packed_contents(~Element,input)) :
    GatherValues.Facts(~Element,~Scatter<Before,Part>,~value(~Element,~Before,~Part,~before_value,~part_value),Scatter{before,part,Views.View{target_axes,target_offset},U32.to_nat(count)},target_shape,scatter_content(~Element,count,Views.View{target_axes,target_offset},_,Buffer.contents(~Element,Buffer.pack(~Element,output))))
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,output)),Buffer.content(~Element,output),Buffer.packed_contents(~Element,output)) :
    GatherValues.Facts(~Element,~Scatter<Before,Part>,~value(~Element,~Before,~Part,~before_value,~part_value),Scatter{before,part,Views.View{target_axes,target_offset},U32.to_nat(count)},target_shape,scatter_content(~Element,count,Views.View{target_axes,target_offset},Buffer.content(~Element,input),_))
  program_facts(~Element,~Before,~Part,~before_value,~part_value,before,part,target_axes,target_offset,input,output,count,target_shape,ready,part_values,before_values,count_room,exact,fits,length_exact)

# The storage content of the target after an accepted scatter of the part.
def modeled_facts(~Element: Data,~Before: Data,~Part: Data,~before_value: Before -> U32 -> Element,~part_value: Part -> U32 -> Element,+before: Before,+part: Part,+target_axes: List<&2,Views.Axis>,+target_offset: U32,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+count: U32,+source_axes: List<&2,Views.Axis>,+target_shape: List<&2,U32>,
  +ready: Buffer.Sized<Element,output>,
  +part_values: Contract.Values(~Element,~Part,~part_value,part,Buffer.content(~Element,input)),
  +before_values: Contract.Values(~Element,~Before,~before_value,before,Buffer.content(~Element,output)),
  +source_exact: {ShapeLaws.elements(Views.axis_extents(source_axes)) == Buffer.length(~Element,input) : Nat},
  +exact: {ShapeLaws.elements(Views.axis_extents(target_axes)) == U32.to_nat(count) : Nat},
  +length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(target_shape) : Nat},
  accepted: {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.scatter(~Element,Buffer.pack(~Element,input),Views.View{source_axes,0},Buffer.pack(~Element,output),Views.View{target_axes,target_offset}))) == True{} : Bool}) ->
  GatherValues.Facts(~Element,~Scatter<Before,Part>,~value(~Element,~Before,~Part,~before_value,~part_value),Scatter{before,part,Views.View{target_axes,target_offset},U32.to_nat(count)},target_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.scatter(~Element,Buffer.pack(~Element,input),Views.View{source_axes,0},Buffer.pack(~Element,output),Views.View{target_axes,target_offset})))):
  guarded_facts(~Element,~Before,~Part,~before_value,~part_value,before,part,target_axes,target_offset,input,output,count,source_axes,target_shape,ready,part_values,before_values,source_exact,exact,length_exact,
    guarded(~Element,Views.View{source_axes,0},Views.View{target_axes,target_offset},input,output,
      %OperationsLaws.scatter(~Element,Buffer.pack(~Element,input),Views.View{source_axes,0},Buffer.pack(~Element,output),Views.View{target_axes,target_offset}) : {GatherValues.accepted_owners(~Element,_) == True{} : Bool}
      accepted))

# A target view without a count is rejected.
def uncounted_model(~Element: Data,valid: Bool,+source: Views.View,+target: Views.View,+input: Tree.Tree<Element>,+input_length: Nat,+output: Tree.Tree<Element>,+output_length: Nat) ->
  {GatherValues.accepted_owners(~Element,OperationsLaws.scatter_model(~Element,valid,None{},source,target,input,input_length,output,output_length)) == False{} : Bool}:
  match valid:
    case True{}: {==}
    case False{}: {==}

def uncounted(~Element: Data,input: Storage.Buffer<Element>,+source: Views.View,output: Storage.Buffer<Element>,+target: Views.View,
  absent: {Views.numel(target) == None{} : Maybe<&2,U32>}) ->
  {GatherValues.accepted_owners(~Element,OperationsLaws.scatter_denotation(~Element,input,source,output,target)) == False{} : Bool}:
  Storage.Buffer{input,+input_length,input_certificate} = input
  Storage.Buffer{output,+output_length,output_certificate} = output
  %Equal.sym(Maybe<&2,U32>,Views.numel(target),None{},absent) :
    {GatherValues.accepted_owners(~Element,OperationsLaws.scatter_model(~Element,Shapes.same(Views.shape(source),Views.shape(target)) && Views.is_contiguous(source) &&
      Operations.valid_buffer(input_length,Storage.capacity(~Element,input,input_certificate),source) &&
      Operations.valid_buffer(output_length,Storage.capacity(~Element,output,output_certificate),target),_,source,target,
      Tree.reflect(Element,input),input_length,Tree.reflect(Element,output),output_length)) == False{} : Bool}
  uncounted_model(~Element,Shapes.same(Views.shape(source),Views.shape(target)) && Views.is_contiguous(source) &&
      Operations.valid_buffer(input_length,Storage.capacity(~Element,input,input_certificate),source) &&
      Operations.valid_buffer(output_length,Storage.capacity(~Element,output,output_certificate),target),source,target,
    Tree.reflect(Element,input),input_length,Tree.reflect(Element,output),output_length)
