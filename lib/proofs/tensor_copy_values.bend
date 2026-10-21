# Values of the checked copy. The run of a copy is a write program; a
# position inside the count holds the sample of its index, and the sample
# of a strided source is the storage at the address of the index in the
# coordinate model. Every operation that copies through a view proves
# only its own address map.
import Base
import ./boolean_logic.bend as Boolean
import ../tensor_access.bend as Access
import ./tensor_cursor_bounds.bend as CursorBounds
import ./tensor_access_refinement.bend as AccessLaws
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_to_u32_bounds.bend as Bounds
import ./storage_buffer_model.bend as Buffer
import ./contiguous_copy.bend as Contiguous
import ./tensor_cursor_refinement.bend as CursorLaws
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_cursor_model.bend as CursorModel
import ./tensor_cursor_order.bend as CursorOrder
import ./traversal_array_refinement.bend as ArrayLaws
import ../traversal_loop.bend as Loop
import ./storage_shape.bend as Shape
import ./sequence_membership.bend as Members
import ../storage_buffer.bend as Storage
import ../tensor_operations.bend as Operations
import ./nat_order.bend as Order
import ./tensor_operations_refinement.bend as OperationsLaws
import ./tensor_operation_bounds.bend as OperationBounds
import ./ordered_ranges.bend as Ranges
import ./tensor_visit_refinement.bend as TensorVisit
import ./traversal_visit_refinement.bend as Visit
import ./tensor_shape_refinement.bend as ShapeLaws
import ../tensor_shape.bend as Shapes
import ./storage_refinement.bend as StorageLaws
import ../traversal_array.bend as Traversal
import ./consecutive_writes.bend as Consecutive
import ./storage_tree_model.bend as Tree
import ../tensor_view.bend as Views
import ./write_program.bend as Writes
import ./traversal_batch_refinement.bend as TraversalBatchRefinement

# Nearest upsampling keeps the band representation and calls the same checked
# repeated-axis copy as whole upsampling. View/count premises below are the
# actual runtime computations; their successful producer guards remain to prove.
def right_content(~Element: Data,result: Operations.Outcome<OperationsLaws.PairOwners(Element)>) -> Buffer.Content<Element>:
  match result:
    case Operations.Accepted{(left,OperationsLaws.Owner{storage,length})}: Buffer.Content{Tree.reflect(Element,storage),length}
    case Operations.Rejected{(left,OperationsLaws.Owner{storage,length})}: Buffer.Content{Tree.reflect(Element,storage),length}

# Actual checked-copy output contracts for the nearest-neighbour band leaf.
# Ready includes both accepted and rejected outcomes. Values below require
# the copy guard, store bounds, sample matching, and coverage explicitly;
# those premises are not claimed to follow from the public caller yet.
def output_tree(~Element: Data,result: AccessLaws.ModelResult<Tree.Tree<Element>,Element>) -> Tree.Tree<Element>:
  AccessLaws.ModelResult{reader,output} = result
  output

def reader_tree(~Element: Data,result: AccessLaws.ModelResult<Tree.Tree<Element>,Element>) -> Tree.Tree<Element>:
  AccessLaws.ModelResult{reader,output} = result
  reader

def view_offset(view: Views.View) -> U32:
  Views.View{axes,offset} = view
  offset

# A visit writes the sample of its coordinates at each position in turn;
# copies and reductions along an axis differ only in sample and next.
def visit_writes(~Element: Data,~Coordinates: Data,~sample: Tree.Tree<Element> -> Coordinates -> Element,~next: Coordinates -> Coordinates,
  count: Nat,+position: U32,+coordinates: Coordinates,+input: Tree.Tree<Element>) -> List<&2,Writes.Assignment<Element>>:
  match count:
    case 0n: []
    case 1n+rest:
      Writes.Put{position,sample(input,coordinates)} <>
        visit_writes(~Element,~Coordinates,~sample,~next,rest,(position + 1 : U32),next(coordinates),input)

def fast_program(~Element: Data,count: Nat,+position: U32,+source: U32,+destination: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {output_tree(~Element,AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~Traversal.Offsets,
      Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Traversal.Offsets>,
        ~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Traversal.Offsets,~ArrayLaws.sample_offset(~Element),~Traversal.destination_offset),
        count,Traversal.State{input,output,position,Traversal.Offsets{source,destination}}))) ==
    Writes.pure(Element,Contiguous.writes(~Element,count,input,position,source,destination),output) : Tree.Tree<Element>}:
  match count:
    case 0n: {==}
    case 1n+rest:
      fast_program(~Element,rest,(position + 1 : U32),source,destination,input,
        Tree.write(Element,output,(destination + position : U32),Tree.read(Element,input,(source + position : U32))))

def visit_program(~Element: Data,~Coordinates: Data,~sample: Tree.Tree<Element> -> Coordinates -> Element,~next: Coordinates -> Coordinates,
  count: Nat,+position: U32,+coordinates: Coordinates,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {output_tree(~Element,AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~Coordinates,
      Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Coordinates>,
        ~Visit.model_step(~Tree.Tree<Element>,~Element,~Coordinates,~sample,~next),
        count,Traversal.State{input,output,position,coordinates}))) ==
    Writes.pure(Element,visit_writes(~Element,~Coordinates,~sample,~next,count,position,coordinates,input),output) : Tree.Tree<Element>}:
  match count:
    case 0n: {==}
    case 1n+rest:
      visit_program(~Element,~Coordinates,~sample,~next,rest,(position + 1 : U32),next(coordinates),input,
        Tree.write(Element,output,position,sample(input,coordinates)))

def copy_writes(~Element: Data,fast: Bool,count: Nat,+source: Views.View,destination: U32,input: Tree.Tree<Element>) -> List<&2,Writes.Assignment<Element>>:
  match fast:
    case True{}:
      Views.View{axes,offset} = source
      Contiguous.writes(~Element,count,input,0,offset,destination)
    case False{}: visit_writes(~Element,~CursorLaws.Coordinates,~TensorVisit.sample(~Element),~CursorLaws.next,count,destination,CursorLaws.initial(source),input)

def copy_program(~Element: Data,+fast: Bool,+count: Nat,+source: Views.View,+destination: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {output_tree(~Element,AccessLaws.copy_model(~Element,fast,count,source,destination,input,output)) ==
    Writes.pure(Element,copy_writes(~Element,fast,count,source,destination,input),output) : Tree.Tree<Element>}:
  match fast:
    case True{}:
      Views.View{+axes,+offset} = source
      fast_program(~Element,count,0,offset,destination,input,output)
    case False{}: visit_program(~Element,~CursorLaws.Coordinates,~TensorVisit.sample(~Element),~CursorLaws.next,count,destination,CursorLaws.initial(source),input,output)

# The source of a copy is only read: the model returns it unchanged.
def fast_reader(~Element: Data,count: Nat,+position: U32,+source: U32,+destination: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {reader_tree(~Element,AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~Traversal.Offsets,
      Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Traversal.Offsets>,
        ~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Traversal.Offsets,~ArrayLaws.sample_offset(~Element),~Traversal.destination_offset),
        count,Traversal.State{input,output,position,Traversal.Offsets{source,destination}}))) == input : Tree.Tree<Element>}:
  match count:
    case 0n: {==}
    case 1n+rest:
      fast_reader(~Element,rest,(position + 1 : U32),source,destination,input,
        Tree.write(Element,output,(destination + position : U32),Tree.read(Element,input,(source + position : U32))))

def visit_reader(~Element: Data,count: Nat,+position: U32,+coordinates: CursorLaws.Coordinates,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {reader_tree(~Element,AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~CursorLaws.Coordinates,
      Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,CursorLaws.Coordinates>,
        ~Visit.model_step(~Tree.Tree<Element>,~Element,~CursorLaws.Coordinates,~TensorVisit.sample(~Element),~CursorLaws.next),
        count,Traversal.State{input,output,position,coordinates}))) == input : Tree.Tree<Element>}:
  match count:
    case 0n: {==}
    case 1n+rest:
      visit_reader(~Element,rest,(position + 1 : U32),CursorLaws.next(coordinates),input,
        Tree.write(Element,output,position,TensorVisit.sample(~Element,input,coordinates)))

def copy_reader(~Element: Data,+fast: Bool,+count: Nat,+source: Views.View,+destination: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {reader_tree(~Element,AccessLaws.copy_model(~Element,fast,count,source,destination,input,output)) == input : Tree.Tree<Element>}:
  match fast:
    case True{}:
      Views.View{+axes,+offset} = source
      fast_reader(~Element,count,0,offset,destination,input,output)
    case False{}: visit_reader(~Element,count,destination,CursorLaws.initial(source),input,output)

def modeled_content(~Element: Data,+input_length: Nat,+output_length: Nat,+result: AccessLaws.ModelResult<Tree.Tree<Element>,Element>) ->
  {right_content(~Element,Operations.Accepted{OperationsLaws.modeled_pair(~Element,input_length,output_length,result)}) ==
    Buffer.Content{output_tree(~Element,result),output_length} : Buffer.Content<Element>}:
  AccessLaws.ModelResult{reader,+output} = result
  Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{tree,output_length},
    Tree.reflect(Element,Tree.pack(Element,output)),output,StorageLaws.reflect_packed(Element,output))

def accepted_content(~Element: Data,+count: U32,+source: Views.View,+target: Views.View,+input: Tree.Tree<Element>,+input_length: Nat,+output: Tree.Tree<Element>,+output_length: Nat) ->
  {right_content(~Element,OperationsLaws.copy_model(~Element,True{},Some{count},source,target,input,input_length,output,output_length)) ==
    Buffer.Content{Writes.pure(Element,copy_writes(~Element,Views.is_contiguous(source),U32.to_nat(count),source,view_offset(target),input),output),output_length} : Buffer.Content<Element>}:
  Views.View{+axes,+destination} = target
  Equal.trans(Buffer.Content<Element>,
    right_content(~Element,OperationsLaws.copy_model(~Element,True{},Some{count},source,target,input,input_length,output,output_length)),
    Buffer.Content{output_tree(~Element,AccessLaws.copy_model(~Element,Views.is_contiguous(source),U32.to_nat(count),source,destination,input,output)),output_length},
    Buffer.Content{Writes.pure(Element,copy_writes(~Element,Views.is_contiguous(source),U32.to_nat(count),source,destination,input),output),output_length},
    modeled_content(~Element,input_length,output_length,AccessLaws.copy_model(~Element,Views.is_contiguous(source),U32.to_nat(count),source,destination,input,output)),
    Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{tree,output_length},
      output_tree(~Element,AccessLaws.copy_model(~Element,Views.is_contiguous(source),U32.to_nat(count),source,destination,input,output)),
      Writes.pure(Element,copy_writes(~Element,Views.is_contiguous(source),U32.to_nat(count),source,destination,input),output),
      copy_program(~Element,Views.is_contiguous(source),U32.to_nat(count),source,destination,input,output)))

def program_content(~Element: Data,+count: U32,+source: Views.View,+target: Views.View,input: Buffer.Content<Element>,output: Buffer.Content<Element>) -> Buffer.Content<Element>:
  Buffer.Content{input,input_length} = input
  Buffer.Content{output,output_length} = output
  Buffer.Content{Writes.pure(Element,copy_writes(~Element,Views.is_contiguous(source),U32.to_nat(count),source,view_offset(target),input),output),output_length}

def copy_observed_content(~Element: Data,input: Storage.Buffer<Element>,+source: Views.View,+reversed: List<&2,Bool>,output: Storage.Buffer<Element>,+target: Views.View,+count: U32,
  guarded: {OperationBounds.copy_guard(~Element,input,source,output,target) == True{} : Bool},counted: {Views.numel(target) == Some{count} : Maybe<&2,U32>}) ->
  {right_content(~Element,OperationsLaws.copy_denotation(~Element,input,source,reversed,output,target)) ==
    program_content(~Element,count,Views.mirrored(source,reversed),target,Buffer.contents(~Element,input),Buffer.contents(~Element,output)) : Buffer.Content<Element>}:
  Storage.Buffer{input,+input_length,input_certificate} = input
  Storage.Buffer{output,+output_length,output_certificate} = output
  %Equal.sym(Bool,Shapes.same(Views.shape(source),Views.shape(target)) && Views.is_contiguous(target) &&
      Operations.valid_buffer(input_length,Storage.capacity(~Element,input,input_certificate),source) &&
      Operations.valid_buffer(output_length,Storage.capacity(~Element,output,output_certificate),target),True{},guarded) :
    {right_content(~Element,OperationsLaws.copy_model(~Element,_,Views.numel(target),Views.mirrored(source,reversed),target,Tree.reflect(Element,input),input_length,Tree.reflect(Element,output),output_length)) ==
      program_content(~Element,count,Views.mirrored(source,reversed),target,Buffer.Content{Tree.reflect(Element,input),input_length},Buffer.Content{Tree.reflect(Element,output),output_length}) : Buffer.Content<Element>}
  %Equal.sym(Maybe<&2,U32>,Views.numel(target),Some{count},counted) :
    {right_content(~Element,OperationsLaws.copy_model(~Element,True{},_,Views.mirrored(source,reversed),target,Tree.reflect(Element,input),input_length,Tree.reflect(Element,output),output_length)) ==
      program_content(~Element,count,Views.mirrored(source,reversed),target,Buffer.Content{Tree.reflect(Element,input),input_length},Buffer.Content{Tree.reflect(Element,output),output_length}) : Buffer.Content<Element>}
  accepted_content(~Element,count,Views.mirrored(source,reversed),target,Tree.reflect(Element,input),input_length,Tree.reflect(Element,output),output_length)

def cursor_address(coordinates: CursorLaws.Coordinates) -> U32:
  CursorLaws.Coordinates{axes,offset,valid} = coordinates
  U32.from_nat(CursorModel.address(axes,offset))

def cursor_sample_projection(~Element: Data,+input: Tree.Tree<Element>,+coordinates: CursorLaws.Coordinates) ->
  {Tree.read(Element,input,cursor_address(coordinates)) == TensorVisit.sample(~Element,input,coordinates) : Element}:
  CursorLaws.Coordinates{axes,offset,valid} = coordinates
  {==}

def iterated_address(count: Nat,+axes: List<&2,CursorModel.Coordinate>,+offset: Nat,+valid: CursorLaws.Valid(axes)) ->
  {cursor_address(Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,count,CursorLaws.Coordinates{axes,offset,valid})) == U32.from_nat(CursorModel.address(CursorOrder.iterate(count,axes),offset)) : U32}:
  match count:
    case 0n: {==}
    case 1n+remaining: iterated_address(remaining,CursorModel.advance(axes),offset,CursorLaws.advance_valid(axes,valid))

# The address the coordinate model gives to the index of a view.
def address(source: Views.View,index: Nat) -> U32:
  Views.View{axes,offset} = source
  U32.from_nat(CursorModel.address(CursorOrder.iterate(index,CursorModel.prepare_axes(axes,[])),U32.to_nat(offset)))

# The coordinates behind that address are the row-major digits of the index
# over the extents of the view (axes of extent one carry no digit); the
# address is their dot product with the strides, plus the offset.
def coordinates(source: Views.View,index: Nat) -> List<&2,CursorModel.Coordinate>:
  Views.View{axes,offset} = source
  CursorOrder.iterate(index,CursorModel.prepare_axes(axes,[]))

# The cursor of the strided traversal reads, at step index, the address of
# that index.
def sample_address(~Element: Data,+source: Views.View,+input: Tree.Tree<Element>,+index: Nat) ->
  {TensorVisit.sample(~Element,input,Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,CursorLaws.initial(source))) == Tree.read(Element,input,address(source,index)) : Element}:
  Views.View{+axes,+offset} = source
  +view = {Views.View{axes,offset} : Views.View}
  +coordinates = Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,CursorLaws.initial(view))
  Equal.trans(Element,TensorVisit.sample(~Element,input,coordinates),Tree.read(Element,input,cursor_address(coordinates)),Tree.read(Element,input,address(view,index)),
    Equal.sym(Element,Tree.read(Element,input,cursor_address(coordinates)),TensorVisit.sample(~Element,input,coordinates),cursor_sample_projection(~Element,input,coordinates)),
    Equal.cong(U32,Element,word => Tree.read(Element,input,word),cursor_address(coordinates),address(view,index),
      iterated_address(index,CursorModel.prepare_axes(axes,[]),U32.to_nat(offset),CursorLaws.prepared_valid(axes,[],Unit{}))))

# A contiguous view at offset zero addresses its indices by themselves.
def zero_contiguous_address(+source: Views.View,+index: Nat,+zero: {Access.zero_contiguous(source) == True{} : Bool},
  +inside: {Nat.is_lt(index,ShapeLaws.elements(Views.axis_extents_of(source))) == True{} : Bool}) -> {address(source,index) == U32.from_nat(index) : U32}:
  Views.View{+axes,+offset} = source
  +nonempty = CountCompletion.positive_not_zero(Views.axis_extents(axes),Order.le_lt_trans(0n,index,ShapeLaws.elements(Views.axis_extents(axes)),Order.zero_le(index),inside))
  %Equal.sym(U32,offset,0,Arithmetic.equal_of_is_eq(offset,0,Boolean.left(U32.is_zero(offset),Views.is_contiguous(Views.View{axes,offset}),zero))) :
    {U32.from_nat(CursorModel.address(CursorOrder.iterate(index,CursorModel.prepare_axes(axes,[])),U32.to_nat(_))) == U32.from_nat(index) : U32}
  Equal.cong(Nat,U32,U32.from_nat,CursorModel.address(CursorOrder.iterate(index,CursorModel.prepare_axes(axes,[])),U32.to_nat(0)),Nat.add(U32.to_nat(0),index),
    CursorOrder.contiguous_addresses(axes,0,index,nonempty,Boolean.right(U32.is_zero(offset),Views.is_contiguous(Views.View{axes,offset}),zero),inside))

# The elements of a view from position on, count of them, lie at base plus
# their position.
def Line(+view: Views.View,+base: Nat,count: Nat,+position: Nat) -> Data:
  match count:
    case 0n: Unit
    case 1n+rest: Shape.Both<{address(view,position) == U32.from_nat(Nat.add(base,position)) : U32},Line(view,base,rest,1n+position)>

# A contiguous view lies at consecutive addresses from its offset.
def contiguous_line(+axes: List<&2,Views.Axis>,+offset: U32,count: Nat,+position: Nat,
  +nonempty: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},+contiguous: {Views.is_contiguous(Views.View{axes,offset}) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(position,count),ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}) ->
  Line(Views.View{axes,offset},U32.to_nat(offset),count,position):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      +elements = ShapeLaws.elements(Views.axis_extents(axes))
      Shape.Both{
        Equal.cong(Nat,U32,U32.from_nat,CursorModel.address(CursorOrder.iterate(position,CursorModel.prepare_axes(axes,[])),U32.to_nat(offset)),Nat.add(U32.to_nat(offset),position),
          CursorOrder.contiguous_addresses(axes,offset,position,nonempty,contiguous,Order.room_index(position,rest,elements,room))),
        contiguous_line(axes,offset,rest,1n+position,nonempty,contiguous,Order.advance_room(position,rest,elements,room))}
def contiguous_start(+axes: List<&2,Views.Axis>,+offset: U32,+count: Nat,
  +nonempty: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},+contiguous: {Views.is_contiguous(Views.View{axes,offset}) == True{} : Bool},
  +exact: {ShapeLaws.elements(Views.axis_extents(axes)) == count : Nat}) ->
  Line(Views.View{axes,offset},U32.to_nat(offset),count,0n):
  contiguous_line(axes,offset,count,0n,nonempty,contiguous,
    %Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(axes)),count,exact) : {Nat.is_le(count,_) == True{} : Bool} Order.reflexive(count))

# The reader of a copy: the source storage at the address of each index.
type Sampled<-Element: Data> is Data:
  Sampled{source: Views.View,input: Tree.Tree<Element>}

def sampled(~Element: Data,+part: Sampled<Element>,+position: Nat) -> Element:
  Sampled{source,input} = part
  Tree.read(Element,input,address(source,position))

# A contiguous source read at its offset plus each position is the copy
# reader.
def fast_line(~Element: Data,count: Nat,+position: Nat,+axes: List<&2,Views.Axis>,+offset: U32,+input: Tree.Tree<Element>,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(position,count),U32.to_nat(capacity)) == True{} : Bool},
  line: Line(Views.View{axes,offset},U32.to_nat(offset),count,position)) ->
  {Contiguous.writes(~Element,count,input,U32.from_nat(position),offset,0) ==
    Consecutive.writes(~Element,~Sampled<Element>,~sampled(~Element),count,0n,position,Sampled{Views.View{axes,offset},input}) : List<&2,Writes.Assignment<Element>>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      Shape.Both{here,later} = line
      +view = {Views.View{axes,offset} : Views.View}
      +at = U32.from_nat(position)
      +value = Tree.read(Element,input,address(view,position))
      +later_writes = Consecutive.writes(~Element,~Sampled<Element>,~sampled(~Element),rest,0n,1n+position,Sampled{view,input})
      +natural = Nat.add(U32.to_nat(offset),position)
      +read_at = Equal.trans(U32,(offset + at : U32),U32.from_nat(Nat.add(U32.to_nat(offset),U32.to_nat(at))),address(view,position),
        Arithmetic.native_add(offset,at),
        Equal.trans(U32,U32.from_nat(Nat.add(U32.to_nat(offset),U32.to_nat(at))),U32.from_nat(natural),address(view,position),
          Equal.cong(Nat,U32,index => U32.from_nat(Nat.add(U32.to_nat(offset),index)),U32.to_nat(at),position,
            Bounds.roundtrip(position,capacity,capacity_ok,Order.transitive(position,Nat.add(position,1n+rest),U32.to_nat(capacity),Order.le_add(position,1n+rest),room))),
          Equal.sym(U32,address(view,position),U32.from_nat(natural),here)))
      %Equal.sym(U32,(0 + at : U32),at,Arithmetic.add_zero_left(at)) :
        {Writes.Put{_,Tree.read(Element,input,(offset + at : U32))} <> Contiguous.writes(~Element,rest,input,(at + 1 : U32),offset,0) ==
          Writes.Put{at,value} <> later_writes : List<&2,Writes.Assignment<Element>>}
      %Equal.sym(U32,(offset + at : U32),address(view,position),read_at) :
        {Writes.Put{at,Tree.read(Element,input,_)} <> Contiguous.writes(~Element,rest,input,(at + 1 : U32),offset,0) ==
          Writes.Put{at,value} <> later_writes : List<&2,Writes.Assignment<Element>>}
      %Equal.sym(U32,(at + 1 : U32),U32.from_nat(1n+position),Bounds.add_one(at)) :
        {Writes.Put{at,value} <> Contiguous.writes(~Element,rest,input,_,offset,0) == Writes.Put{at,value} <> later_writes : List<&2,Writes.Assignment<Element>>}
      Equal.cong(List<&2,Writes.Assignment<Element>>,List<&2,Writes.Assignment<Element>>,tail => Writes.Put{at,value} <> tail,
        Contiguous.writes(~Element,rest,input,U32.from_nat(1n+position),offset,0),later_writes,
        fast_line(~Element,rest,1n+position,axes,offset,input,capacity,capacity_ok,Order.advance_room(position,rest,U32.to_nat(capacity),room),later))

# A visit from position on writes, at consecutive addresses, the values of a
# part whenever the sample after index steps is the value of the part at index.
def visit_line(~Element: Data,~Coordinates: Data,~Part: Data,~sample: Tree.Tree<Element> -> Coordinates -> Element,~next: Coordinates -> Coordinates,
  ~value: @+part: Part -> @+position: Nat -> Element,~start: Coordinates,~input: Tree.Tree<Element>,~part: Part,
  ~agrees: @+index: Nat -> {sample(input,Loop.run(~Coordinates,~next,index,start)) == value(part,index) : Element},count: Nat,+position: Nat) ->
  {visit_writes(~Element,~Coordinates,~sample,~next,count,U32.from_nat(position),Loop.run(~Coordinates,~next,position,start),input) ==
    Consecutive.writes(~Element,~Part,~value,count,0n,position,part) : List<&2,Writes.Assignment<Element>>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +cursor = Loop.run(~Coordinates,~next,position,start)
      +at = U32.from_nat(position)
      +here = value(part,position)
      +later_writes = Consecutive.writes(~Element,~Part,~value,rest,0n,1n+position,part)
      %Equal.sym(Element,sample(input,cursor),here,agrees(position)) :
        {Writes.Put{at,_} <> visit_writes(~Element,~Coordinates,~sample,~next,rest,(at + 1 : U32),next(cursor),input) == Writes.Put{at,here} <> later_writes : List<&2,Writes.Assignment<Element>>}
      %Equal.sym(U32,(at + 1 : U32),U32.from_nat(1n+position),Bounds.add_one(at)) :
        {Writes.Put{at,here} <> visit_writes(~Element,~Coordinates,~sample,~next,rest,_,next(cursor),input) == Writes.Put{at,here} <> later_writes : List<&2,Writes.Assignment<Element>>}
      %TraversalBatchRefinement.last_step(~Coordinates,~next,position,start) :
        {Writes.Put{at,here} <> visit_writes(~Element,~Coordinates,~sample,~next,rest,U32.from_nat(1n+position),_,input) == Writes.Put{at,here} <> later_writes : List<&2,Writes.Assignment<Element>>}
      Equal.cong(List<&2,Writes.Assignment<Element>>,List<&2,Writes.Assignment<Element>>,tail => Writes.Put{at,here} <> tail,
        visit_writes(~Element,~Coordinates,~sample,~next,rest,U32.from_nat(1n+position),Loop.run(~Coordinates,~next,1n+position,start),input),later_writes,
        visit_line(~Element,~Coordinates,~Part,~sample,~next,~value,~start,~input,~part,~agrees,rest,1n+position))

# The writes of a copy to offset zero are the copy reader at consecutive
# addresses, for a contiguous and a strided source alike.
def copy_line(~Element: Data,fast: Bool,count: Nat,+source: Views.View,+input: Tree.Tree<Element>,+capacity: U32,
  +classified: {Views.is_contiguous(source) == fast : Bool},+exact: {ShapeLaws.elements(Views.axis_extents_of(source)) == count : Nat},
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(count,U32.to_nat(capacity)) == True{} : Bool}) ->
  {copy_writes(~Element,fast,count,source,0,input) ==
    Consecutive.writes(~Element,~Sampled<Element>,~sampled(~Element),count,0n,0n,Sampled{source,input}) : List<&2,Writes.Assignment<Element>>}:
  match fast count:
    case False{} count:
      visit_line(~Element,~CursorLaws.Coordinates,~Sampled<Element>,~TensorVisit.sample(~Element),~CursorLaws.next,~sampled(~Element),~CursorLaws.initial(source),~input,
        ~Sampled{source,input},~(index => sample_address(~Element,source,input,index)),count,0n)
    case True{} 0n:
      Views.View{axes,offset} = source
      {==}
    case True{} 1n+ +rest:
      Views.View{+axes,+offset} = source
      +nonempty = CountCompletion.positive_not_zero(Views.axis_extents(axes),
        %Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(axes)),1n+rest,exact) : {Nat.is_lt(0n,_) == True{} : Bool} {==})
      fast_line(~Element,1n+rest,0n,axes,offset,input,capacity,capacity_ok,room,contiguous_start(axes,offset,1n+rest,nonempty,classified,exact))

# A copy into a target at offset zero stores, at every index inside its
# count, the source storage at the address of that index. A contiguous and a
# strided source give the same statement.
def copied_value(~Element: Data,+source: Views.View,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+length: Nat,+count: Nat,+index: Nat,
  +ready: Buffer.ContentReady<Element,Buffer.Content{output,length}>,+capacity_ok: {U32.is_le(Tree.capacity(Element,output),2147483648) == True{} : Bool},
  +room: {Nat.is_le(count,U32.to_nat(Tree.capacity(Element,output))) == True{} : Bool},+inside: {Nat.is_lt(index,count) == True{} : Bool},
  +exact: {ShapeLaws.elements(Views.axis_extents_of(source)) == count : Nat}) ->
  {Tree.read(Element,Writes.pure(Element,copy_writes(~Element,Views.is_contiguous(source),count,source,0,input),output),U32.from_nat(index)) ==
    Tree.read(Element,input,address(source,index)) : Element}:
  Buffer.ContentReady{+depth,+balanced,+depth_ok,filled} = ready
  +fast = Views.is_contiguous(source)
  %Equal.sym(List<&2,Writes.Assignment<Element>>,copy_writes(~Element,fast,count,source,0,input),
    Consecutive.writes(~Element,~Sampled<Element>,~sampled(~Element),count,0n,0n,Sampled{source,input}),
    copy_line(~Element,fast,count,source,input,Tree.capacity(Element,output),{==},exact,capacity_ok,room)) :
    {Tree.read(Element,Writes.pure(Element,_,output),U32.from_nat(index)) == Tree.read(Element,input,address(source,index)) : Element}
  Consecutive.read(~Element,~Sampled<Element>,~sampled(~Element),count,0n,Sampled{source,input},output,depth,index,balanced,depth_ok,capacity_ok,room,inside)

# The span of the coordinates does not change along the iteration.
def iterate_span(count: Nat,+axes: List<&2,CursorModel.Coordinate>) -> {CursorModel.span(CursorOrder.iterate(count,axes)) == CursorModel.span(axes) : Nat}:
  match count:
    case 0n: {==}
    case 1n+rest:
      Equal.trans(Nat,CursorModel.span(CursorOrder.iterate(rest,CursorModel.advance(axes))),CursorModel.span(CursorModel.advance(axes)),CursorModel.span(axes),
        iterate_span(rest,CursorModel.advance(axes)),CursorBounds.advance_span(axes))

# The address of every index of a view that fits a length is inside it.
def address_inside(+source: Views.View,+index: Nat,+length: U32,fits: {Views.fits_nonempty(False{},source,length) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(address(source,index)),U32.to_nat(length)) == True{} : Bool}:
  Views.View{+axes,+offset} = source
  +start = CursorModel.prepare_axes(axes,[])
  CursorBounds.read_inside(CursorOrder.iterate(index,start),U32.to_nat(offset),length,CursorOrder.iterate_valid(index,start,CursorLaws.prepared_valid(axes,[],Unit{})),
    %Equal.sym(Nat,CursorModel.span(CursorOrder.iterate(index,start)),CursorModel.span(start),iterate_span(index,start)) :
      {Nat.is_lt(Nat.add(_,U32.to_nat(offset)),U32.to_nat(length)) == True{} : Bool}
    CursorBounds.fits_bound(axes,offset,length,fits))
