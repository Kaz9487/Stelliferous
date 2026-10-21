# Values of the checked line folds. Output k of the lines of a view holds
# finish of the fold, from the initial accumulator and in increasing
# position, of the values the view reads from k times the length of a line
# on.
import Base
import ../tensor_access.bend as Access
import ../tensor_operations.bend as Operations
import ../tensor_view.bend as Views
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ../storage_buffer.bend as Storage
import ./storage_tree_model.bend as Tree
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./tensor_access_refinement.bend as AccessLaws
import ./tensor_operations_refinement.bend as OperationsLaws
import ./tensor_copy_values.bend as CopyValues
import ./tensor_cursor_refinement.bend as CursorLaws
import ./tensor_cursor_model.bend as CursorModel
import ./tensor_visit_refinement.bend as TensorVisit
import ./traversal_array_refinement.bend as ArrayLaws
import ./ordered_fold_refinement.bend as Fold
import ./consecutive_writes.bend as Consecutive
import ./write_program.bend as Writes
import ./traversal_batch_refinement.bend as Batch
import ./tensor_row_major_coordinates.bend as Coordinates
import ./tensor_shape_refinement.bend as ShapeLaws
import ./natural_addition.bend as Addition
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_full_range.bend as FullRange
import ./word_arithmetic.bend as WordArithmetic
import ./word_multiplication.bend as WordMultiplication
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./row_major_index.bend as RowMajor

# ---- Folds of consecutive values ----

# The fold of count values of a reader from position on, in increasing
# position.
def line_fold(~Source: Data,~Element: Data,~Accumulator: Data,~reader: @+source: Source -> @+position: Nat -> Element,~operation: Accumulator -> Element -> Accumulator,
  count: Nat,+source: Source,+position: Nat,accumulator: Accumulator) -> Accumulator:
  match count:
    case 0n: accumulator
    case 1n+rest:
      line_fold(~Source,~Element,~Accumulator,~reader,~operation,rest,source,1n+position,operation(accumulator,reader(source,position)))

# The fold of first then second values continues the fold of the first with
# the second.
def line_split(~Source: Data,~Element: Data,~Accumulator: Data,~reader: @+source: Source -> @+position: Nat -> Element,~operation: Accumulator -> Element -> Accumulator,
  first: Nat,+second: Nat,+source: Source,+position: Nat,+accumulator: Accumulator) ->
  {line_fold(~Source,~Element,~Accumulator,~reader,~operation,Nat.add(first,second),source,position,accumulator) ==
    line_fold(~Source,~Element,~Accumulator,~reader,~operation,second,source,Nat.add(position,first),
      line_fold(~Source,~Element,~Accumulator,~reader,~operation,first,source,position,accumulator)) : Accumulator}:
  match first:
    case 0n:
      %Equal.sym(Nat,Nat.add(position,0n),position,Addition.zero_right(position)) :
        {line_fold(~Source,~Element,~Accumulator,~reader,~operation,second,source,position,accumulator) ==
          line_fold(~Source,~Element,~Accumulator,~reader,~operation,second,source,_,accumulator) : Accumulator}
      {==}
    case 1n+ +rest:
      +next = operation(accumulator,reader(source,position))
      %Equal.sym(Nat,Nat.add(position,1n+rest),1n+Nat.add(position,rest),Addition.successor_right(position,rest)) :
        {line_fold(~Source,~Element,~Accumulator,~reader,~operation,Nat.add(rest,second),source,1n+position,next) ==
          line_fold(~Source,~Element,~Accumulator,~reader,~operation,second,source,_,
            line_fold(~Source,~Element,~Accumulator,~reader,~operation,rest,source,1n+position,next)) : Accumulator}
      line_split(~Source,~Element,~Accumulator,~reader,~operation,rest,second,source,1n+position,next)

# Two folds whose readers agree at every offset below total fold to the same
# value, from any offset done on.
def line_agreed(~Element: Data,~Accumulator: Data,~Left: Data,~left: @+source: Left -> @+position: Nat -> Element,~Right: Data,~right: @+source: Right -> @+position: Nat -> Element,
  ~operation: Accumulator -> Element -> Accumulator,~left_source: Left,~left_position: Nat,~right_source: Right,~right_position: Nat,~total: Nat,
  ~agree: @+offset: Nat -> @inside: {Nat.is_lt(offset,total) == True{} : Bool} ->
    {left(left_source,Nat.add(left_position,offset)) == right(right_source,Nat.add(right_position,offset)) : Element},
  remaining: Nat,+done: Nat,+accumulator: Accumulator,+room: {Nat.is_le(Nat.add(done,remaining),total) == True{} : Bool}) ->
  {line_fold(~Left,~Element,~Accumulator,~left,~operation,remaining,left_source,Nat.add(left_position,done),accumulator) ==
    line_fold(~Right,~Element,~Accumulator,~right,~operation,remaining,right_source,Nat.add(right_position,done),accumulator) : Accumulator}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +value = right(right_source,Nat.add(right_position,done))
      %Equal.sym(Element,left(left_source,Nat.add(left_position,done)),value,agree(done,Order.room_index(done,rest,total,room))) :
        {line_fold(~Left,~Element,~Accumulator,~left,~operation,rest,left_source,1n+Nat.add(left_position,done),operation(accumulator,_)) ==
          line_fold(~Right,~Element,~Accumulator,~right,~operation,rest,right_source,1n+Nat.add(right_position,done),operation(accumulator,value)) : Accumulator}
      %Addition.successor_right(left_position,done) :
        {line_fold(~Left,~Element,~Accumulator,~left,~operation,rest,left_source,_,operation(accumulator,value)) ==
          line_fold(~Right,~Element,~Accumulator,~right,~operation,rest,right_source,1n+Nat.add(right_position,done),operation(accumulator,value)) : Accumulator}
      %Addition.successor_right(right_position,done) :
        {line_fold(~Left,~Element,~Accumulator,~left,~operation,rest,left_source,Nat.add(left_position,1n+done),operation(accumulator,value)) ==
          line_fold(~Right,~Element,~Accumulator,~right,~operation,rest,right_source,_,operation(accumulator,value)) : Accumulator}
      line_agreed(~Element,~Accumulator,~Left,~left,~Right,~right,~operation,~left_source,~left_position,~right_source,~right_position,~total,~agree,
        rest,1n+done,operation(accumulator,value),Order.advance_room(done,rest,total,room))

# ---- The address of an element of a run ----

# A view reads, at an index below its elements, the weight of the row-major
# digits of the index plus its offset.
def view_weight(+axes: List<&2,Views.Axis>,+offset: U32,+index: Nat,
  +inside: {Nat.is_lt(index,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}) ->
  {CopyValues.address(Views.View{axes,offset},index) == U32.from_nat(Nat.add(Coordinates.weight(Coordinates.points_of(axes,index)),U32.to_nat(offset))) : U32}:
  +points = Coordinates.points_of(axes,index)
  %Coordinates.axes_of(axes,index) :
    {CopyValues.address(Views.View{_,offset},index) == U32.from_nat(Nat.add(Coordinates.weight(points),U32.to_nat(offset))) : U32}
  %Coordinates.position_of(axes,index,0n,inside) :
    {CopyValues.address(Views.View{Coordinates.axes(points),offset},_) == U32.from_nat(Nat.add(Coordinates.weight(points),U32.to_nat(offset))) : U32}
  Coordinates.weighed(points,offset,Coordinates.valid_of(axes,index,inside))

# Element element of run run: the address of the run in the view without
# its last axis plus element strides is the address of
# run * extent + element in the whole view.
def run_element(+axes: List<&2,Views.Axis>,+extent: U32,+stride: U32,+offset: U32,+run: Nat,+element: Nat,
  +run_inside: {Nat.is_lt(run,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},
  +element_inside: {Nat.is_lt(element,U32.to_nat(extent)) == True{} : Bool}) ->
  {(CopyValues.address(Views.View{axes,offset},run) + U32.from_nat(element) * stride : U32) ==
    CopyValues.address(Views.View{List.append(&2,Views.Axis,axes,[Views.Axis{extent,stride}]),offset},Nat.add(Nat.mul(run,U32.to_nat(extent)),element)) : U32}:
  +points = Coordinates.points_of(axes,run)
  +here = {[Coordinates.Point{extent,element,stride,False{}}] : List<&2,Coordinates.Point>}
  +whole = List.append(&2,Coordinates.Point,points,here)
  +weight = Coordinates.weight(points)
  +moved = Nat.mul(element,U32.to_nat(stride))
  +base = Nat.add(weight,U32.to_nat(offset))
  +valid = Coordinates.valid_of(axes,run,run_inside)
  +right = {
    %Coordinates.axes_of(axes,run) :
      {CopyValues.address(Views.View{List.append(&2,Views.Axis,_,[Views.Axis{extent,stride}]),offset},Nat.add(Nat.mul(run,U32.to_nat(extent)),element)) ==
        U32.from_nat(Nat.add(Nat.add(weight,Nat.add(moved,0n)),U32.to_nat(offset))) : U32}
    %Coordinates.position_of(axes,run,0n,run_inside) :
      {CopyValues.address(Views.View{List.append(&2,Views.Axis,Coordinates.axes(points),[Views.Axis{extent,stride}]),offset},Nat.add(Nat.mul(_,U32.to_nat(extent)),element)) ==
        U32.from_nat(Nat.add(Nat.add(weight,Nat.add(moved,0n)),U32.to_nat(offset))) : U32}
    %Coordinates.axes_append(points,here) :
      {CopyValues.address(Views.View{_,offset},Nat.add(Nat.mul(Coordinates.position(points,0n),U32.to_nat(extent)),element)) ==
        U32.from_nat(Nat.add(Nat.add(weight,Nat.add(moved,0n)),U32.to_nat(offset))) : U32}
    %Coordinates.position_append(points,here,0n) :
      {CopyValues.address(Views.View{Coordinates.axes(whole),offset},_) == U32.from_nat(Nat.add(Nat.add(weight,Nat.add(moved,0n)),U32.to_nat(offset))) : U32}
    %Coordinates.weight_append(points,here) :
      {CopyValues.address(Views.View{Coordinates.axes(whole),offset},Coordinates.position(whole,0n)) == U32.from_nat(Nat.add(_,U32.to_nat(offset))) : U32}
    Coordinates.weighed(whole,offset,Coordinates.valid_append(points,here,valid,Shape.Both{element_inside,Unit{}})) :
    {CopyValues.address(Views.View{List.append(&2,Views.Axis,axes,[Views.Axis{extent,stride}]),offset},Nat.add(Nat.mul(run,U32.to_nat(extent)),element)) ==
      U32.from_nat(Nat.add(Nat.add(weight,Nat.add(moved,0n)),U32.to_nat(offset))) : U32}}
  +element_exact = FullRange.roundtrip(element,extent,Bounds.strict_le(element,U32.to_nat(extent),element_inside))
  +product = {
    %element_exact :
      {(U32.from_nat(element) * stride : U32) == U32.from_nat(Nat.mul(_,U32.to_nat(stride))) : U32}
    WordMultiplication.native_multiply(U32.from_nat(element),stride) : {(U32.from_nat(element) * stride : U32) == U32.from_nat(moved) : U32}}
  +sum = Equal.trans(Nat,Nat.add(base,moved),Nat.add(weight,Nat.add(U32.to_nat(offset),moved)),Nat.add(Nat.add(weight,Nat.add(moved,0n)),U32.to_nat(offset)),
    Addition.associative(weight,U32.to_nat(offset),moved),
    %Equal.sym(Nat,Nat.add(moved,0n),moved,Addition.zero_right(moved)) : {Nat.add(weight,Nat.add(U32.to_nat(offset),moved)) == Nat.add(Nat.add(weight,_),U32.to_nat(offset)) : Nat}
    %Order.commutative(moved,U32.to_nat(offset)) : {Nat.add(weight,_) == Nat.add(Nat.add(weight,moved),U32.to_nat(offset)) : Nat}
    Equal.sym(Nat,Nat.add(Nat.add(weight,moved),U32.to_nat(offset)),Nat.add(weight,Nat.add(moved,U32.to_nat(offset))),Addition.associative(weight,moved,U32.to_nat(offset))))
  +left = {
    %Equal.sym(U32,CopyValues.address(Views.View{axes,offset},run),U32.from_nat(base),view_weight(axes,offset,run,run_inside)) :
      {(_ + U32.from_nat(element) * stride : U32) == U32.from_nat(Nat.add(Nat.add(weight,Nat.add(moved,0n)),U32.to_nat(offset))) : U32}
    %Equal.sym(U32,(U32.from_nat(element) * stride : U32),U32.from_nat(moved),product) :
      {(U32.from_nat(base) + _ : U32) == U32.from_nat(Nat.add(Nat.add(weight,Nat.add(moved,0n)),U32.to_nat(offset))) : U32}
    %sum :
      {(U32.from_nat(base) + U32.from_nat(moved) : U32) == U32.from_nat(_) : U32}
    WordArithmetic.native_add(base,moved) :
    {(CopyValues.address(Views.View{axes,offset},run) + U32.from_nat(element) * stride : U32) == U32.from_nat(Nat.add(Nat.add(weight,Nat.add(moved,0n)),U32.to_nat(offset))) : U32}}
  Equal.trans(U32,(CopyValues.address(Views.View{axes,offset},run) + U32.from_nat(element) * stride : U32),U32.from_nat(Nat.add(Nat.add(weight,Nat.add(moved,0n)),U32.to_nat(offset))),
    CopyValues.address(Views.View{List.append(&2,Views.Axis,axes,[Views.Axis{extent,stride}]),offset},Nat.add(Nat.mul(run,U32.to_nat(extent)),element)),
    left,Equal.sym(U32,CopyValues.address(Views.View{List.append(&2,Views.Axis,axes,[Views.Axis{extent,stride}]),offset},Nat.add(Nat.mul(run,U32.to_nat(extent)),element)),
      U32.from_nat(Nat.add(Nat.add(weight,Nat.add(moved,0n)),U32.to_nat(offset))),right))

# The values of run run from element done on fold as the values of the whole
# view from run * extent + done on.
def run_strided(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,remaining: Nat,+done: Nat,
  +axes: List<&2,Views.Axis>,+extent: U32,+stride: U32,+offset: U32,+tree: Tree.Tree<Element>,+run: Nat,+accumulator: Accumulator,
  +run_inside: {Nat.is_lt(run,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(done,remaining),U32.to_nat(extent)) == True{} : Bool}) ->
  {Fold.values(~Tree.Tree<Element>,~Element,~Accumulator,~Traversal.Strided,~ArrayLaws.sample_strided(~Element),~operation,remaining,U32.from_nat(done),tree,
      Traversal.Strided{CopyValues.address(Views.View{axes,offset},run),stride},accumulator) ==
    line_fold(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,remaining,
      CopyValues.Sampled{Views.View{List.append(&2,Views.Axis,axes,[Views.Axis{extent,stride}]),offset},tree},Nat.add(Nat.mul(run,U32.to_nat(extent)),done),accumulator) : Accumulator}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +whole = {CopyValues.Sampled{Views.View{List.append(&2,Views.Axis,axes,[Views.Axis{extent,stride}]),offset},tree} : CopyValues.Sampled<Element>}
      +start = Nat.mul(run,U32.to_nat(extent))
      +context = {Traversal.Strided{CopyValues.address(Views.View{axes,offset},run),stride} : Traversal.Strided}
      +value = CopyValues.sampled(~Element,whole,Nat.add(start,done))
      +head = Equal.cong(U32,Element,word => Tree.read(Element,tree,word),(CopyValues.address(Views.View{axes,offset},run) + U32.from_nat(done) * stride : U32),
        CopyValues.address(Views.View{List.append(&2,Views.Axis,axes,[Views.Axis{extent,stride}]),offset},Nat.add(start,done)),
        run_element(axes,extent,stride,offset,run,done,run_inside,Order.room_index(done,rest,U32.to_nat(extent),room)))
      %Equal.sym(Element,ArrayLaws.sample_strided(~Element,tree,context,U32.from_nat(done)),value,head) :
        {Fold.values(~Tree.Tree<Element>,~Element,~Accumulator,~Traversal.Strided,~ArrayLaws.sample_strided(~Element),~operation,rest,(U32.from_nat(done) + 1 : U32),tree,context,
            operation(accumulator,_)) ==
          line_fold(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,rest,whole,1n+Nat.add(start,done),operation(accumulator,value)) : Accumulator}
      %Equal.sym(U32,(U32.from_nat(done) + 1 : U32),U32.from_nat(1n+done),Bounds.increment_encoded(done)) :
        {Fold.values(~Tree.Tree<Element>,~Element,~Accumulator,~Traversal.Strided,~ArrayLaws.sample_strided(~Element),~operation,rest,_,tree,context,operation(accumulator,value)) ==
          line_fold(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,rest,whole,1n+Nat.add(start,done),operation(accumulator,value)) : Accumulator}
      %Addition.successor_right(start,done) :
        {Fold.values(~Tree.Tree<Element>,~Element,~Accumulator,~Traversal.Strided,~ArrayLaws.sample_strided(~Element),~operation,rest,U32.from_nat(1n+done),tree,context,operation(accumulator,value)) ==
          line_fold(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,rest,whole,_,operation(accumulator,value)) : Accumulator}
      run_strided(~Element,~Accumulator,~operation,rest,1n+done,axes,extent,stride,offset,tree,run,operation(accumulator,value),run_inside,
        Order.advance_room(done,rest,U32.to_nat(extent),room))

# ---- The runs of a line ----

# Steps of a loop after steps of the same loop.
def run_after(~State: Data,~step: State -> State,+after: Nat,before: Nat,+state: State) ->
  {Loop.run(~State,~step,after,Loop.run(~State,~step,before,state)) == Loop.run(~State,~step,Nat.add(before,after),state) : State}:
  match before:
    case 0n: {==}
    case 1n+ +rest: run_after(~State,~step,after,rest,step(state))

# The cursor of a view after index steps.
def cursor_at(+view: Views.View,+index: Nat) -> CursorLaws.Coordinates:
  Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,index,CursorLaws.initial(view))

def cursor_word(+coordinates: CursorLaws.Coordinates) -> {TensorVisit.address(coordinates) == CopyValues.cursor_address(coordinates) : U32}:
  CursorLaws.Coordinates{axes,offset,valid} = coordinates
  {==}

# The cursor of a view at an index stands at the address of that index.
def cursor_address(+view: Views.View,+index: Nat) -> {TensorVisit.address(cursor_at(view,index)) == CopyValues.address(view,index) : U32}:
  Views.View{+axes,+offset} = view
  Equal.trans(U32,TensorVisit.address(cursor_at(Views.View{axes,offset},index)),CopyValues.cursor_address(cursor_at(Views.View{axes,offset},index)),
    CopyValues.address(Views.View{axes,offset},index),cursor_word(cursor_at(Views.View{axes,offset},index)),
    CopyValues.iterated_address(index,CursorModel.prepare_axes(axes,[]),U32.to_nat(offset),CursorLaws.prepared_valid(axes,[],Unit{})))

# remaining runs from run run fold as the values of the whole view from
# run * extent on.
def runs_fold(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,remaining: Nat,+run: Nat,
  +axes: List<&2,Views.Axis>,+extent: U32,+stride: U32,+offset: U32,+tree: Tree.Tree<Element>,+accumulator: Accumulator,
  +room: {Nat.is_le(Nat.add(run,remaining),ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}) ->
  {AccessLaws.run_values(~Element,~Accumulator,~operation,remaining,tree,cursor_at(Views.View{axes,offset},run),extent,stride,accumulator) ==
    line_fold(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,Nat.mul(remaining,U32.to_nat(extent)),
      CopyValues.Sampled{Views.View{List.append(&2,Views.Axis,axes,[Views.Axis{extent,stride}]),offset},tree},Nat.mul(run,U32.to_nat(extent)),accumulator) : Accumulator}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +view = {Views.View{axes,offset} : Views.View}
      +whole = {CopyValues.Sampled{Views.View{List.append(&2,Views.Axis,axes,[Views.Axis{extent,stride}]),offset},tree} : CopyValues.Sampled<Element>}
      +width = U32.to_nat(extent)
      +start = Nat.mul(run,width)
      +cursor = cursor_at(view,run)
      +first = line_fold(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,width,whole,start,accumulator)
      +folded = {
        %Equal.sym(U32,TensorVisit.address(cursor),CopyValues.address(view,run),cursor_address(view,run)) :
          {Fold.values(~Tree.Tree<Element>,~Element,~Accumulator,~Traversal.Strided,~ArrayLaws.sample_strided(~Element),~operation,width,0,tree,Traversal.Strided{_,stride},accumulator) ==
            first : Accumulator}
        %Addition.zero_right(start) :
          {Fold.values(~Tree.Tree<Element>,~Element,~Accumulator,~Traversal.Strided,~ArrayLaws.sample_strided(~Element),~operation,width,0,tree,
              Traversal.Strided{CopyValues.address(view,run),stride},accumulator) ==
            line_fold(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,width,whole,_,accumulator) : Accumulator}
        run_strided(~Element,~Accumulator,~operation,width,0n,axes,extent,stride,offset,tree,run,accumulator,
          Order.room_index(run,rest,ShapeLaws.elements(Views.axis_extents(axes)),room),Order.reflexive(width)) :
        {Fold.values(~Tree.Tree<Element>,~Element,~Accumulator,~Traversal.Strided,~ArrayLaws.sample_strided(~Element),~operation,width,0,tree,
            Traversal.Strided{TensorVisit.address(cursor),stride},accumulator) == first : Accumulator}}
      %Equal.sym(Accumulator,
          line_fold(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,Nat.add(width,Nat.mul(rest,width)),whole,start,accumulator),
          line_fold(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,Nat.mul(rest,width),whole,Nat.add(start,width),first),
          line_split(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,width,Nat.mul(rest,width),whole,start,accumulator)) :
        {AccessLaws.run_values(~Element,~Accumulator,~operation,rest,tree,CursorLaws.next(cursor),extent,stride,
            Fold.values(~Tree.Tree<Element>,~Element,~Accumulator,~Traversal.Strided,~ArrayLaws.sample_strided(~Element),~operation,width,0,tree,
              Traversal.Strided{TensorVisit.address(cursor),stride},accumulator)) == _ : Accumulator}
      %Order.commutative(width,start) :
        {AccessLaws.run_values(~Element,~Accumulator,~operation,rest,tree,CursorLaws.next(cursor),extent,stride,
            Fold.values(~Tree.Tree<Element>,~Element,~Accumulator,~Traversal.Strided,~ArrayLaws.sample_strided(~Element),~operation,width,0,tree,
              Traversal.Strided{TensorVisit.address(cursor),stride},accumulator)) ==
          line_fold(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,Nat.mul(rest,width),whole,_,first) : Accumulator}
      %Batch.last_step(~CursorLaws.Coordinates,~CursorLaws.next,run,CursorLaws.initial(view)) :
        {AccessLaws.run_values(~Element,~Accumulator,~operation,rest,tree,_,extent,stride,
            Fold.values(~Tree.Tree<Element>,~Element,~Accumulator,~Traversal.Strided,~ArrayLaws.sample_strided(~Element),~operation,width,0,tree,
              Traversal.Strided{TensorVisit.address(cursor),stride},accumulator)) ==
          line_fold(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,Nat.mul(rest,width),whole,Nat.add(width,start),first) : Accumulator}
      %Equal.sym(Accumulator,
          Fold.values(~Tree.Tree<Element>,~Element,~Accumulator,~Traversal.Strided,~ArrayLaws.sample_strided(~Element),~operation,width,0,tree,
            Traversal.Strided{TensorVisit.address(cursor),stride},accumulator),first,folded) :
        {AccessLaws.run_values(~Element,~Accumulator,~operation,rest,tree,cursor_at(view,1n+run),extent,stride,_) ==
          line_fold(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,Nat.mul(rest,width),whole,Nat.add(width,start),first) : Accumulator}
      runs_fold(~Element,~Accumulator,~operation,rest,1n+run,axes,extent,stride,offset,tree,first,
        Order.advance_room(run,rest,ShapeLaws.elements(Views.axis_extents(axes)),room))

# The lines of a view after count of them, from a cursor at position.
def lines_after(~Accumulator: Data,count: Nat,+position: Nat,+view: Views.View,+runs: Nat,+extent: U32,+stride: U32,+initial: Accumulator) ->
  {Loop.run(~AccessLaws.LineCoordinates<Accumulator>,~AccessLaws.next_line(~Accumulator),count,AccessLaws.LineCoordinates{cursor_at(view,position),runs,extent,stride,initial}) ==
    AccessLaws.LineCoordinates{cursor_at(view,Nat.add(position,Nat.mul(count,runs))),runs,extent,stride,initial} : AccessLaws.LineCoordinates<Accumulator>}:
  match count:
    case 0n:
      %Equal.sym(Nat,Nat.add(position,0n),position,Addition.zero_right(position)) :
        {AccessLaws.LineCoordinates{cursor_at(view,position),runs,extent,stride,initial} ==
          AccessLaws.LineCoordinates{cursor_at(view,_),runs,extent,stride,initial} : AccessLaws.LineCoordinates<Accumulator>}
      {==}
    case 1n+ +rest:
      %Equal.sym(CursorLaws.Coordinates,Loop.run(~CursorLaws.Coordinates,~CursorLaws.next,runs,cursor_at(view,position)),cursor_at(view,Nat.add(position,runs)),
          run_after(~CursorLaws.Coordinates,~CursorLaws.next,runs,position,CursorLaws.initial(view))) :
        {Loop.run(~AccessLaws.LineCoordinates<Accumulator>,~AccessLaws.next_line(~Accumulator),rest,AccessLaws.LineCoordinates{_,runs,extent,stride,initial}) ==
          AccessLaws.LineCoordinates{cursor_at(view,Nat.add(position,Nat.add(runs,Nat.mul(rest,runs)))),runs,extent,stride,initial} : AccessLaws.LineCoordinates<Accumulator>}
      %Addition.associative(position,runs,Nat.mul(rest,runs)) :
        {Loop.run(~AccessLaws.LineCoordinates<Accumulator>,~AccessLaws.next_line(~Accumulator),rest,AccessLaws.LineCoordinates{cursor_at(view,Nat.add(position,runs)),runs,extent,stride,initial}) ==
          AccessLaws.LineCoordinates{cursor_at(view,_),runs,extent,stride,initial} : AccessLaws.LineCoordinates<Accumulator>}
      lines_after(~Accumulator,rest,Nat.add(position,runs),view,runs,extent,stride,initial)

# ---- The value of a line ----

# The lines from the start as the part their visit writes.
type Folds<-Accumulator: Data,-Element: Data> is Data:
  Folds{start: AccessLaws.LineCoordinates<Accumulator>,input: Tree.Tree<Element>}

def fold_at(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +part: Folds<Accumulator,Element>,+position: Nat) -> Element:
  Folds{start,input} = part
  AccessLaws.sample_line(~Element,~Accumulator,~operation,~finish,input,Loop.run(~AccessLaws.LineCoordinates<Accumulator>,~AccessLaws.next_line(~Accumulator),position,start))

# Output index of the lines of a view of runs runs of extent values holds
# finish of the fold of the runs * extent values of the whole view from
# index * runs * extent on.
def line_value(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +count: Nat,+axes: List<&2,Views.Axis>,+offset: U32,+runs: Nat,+extent: U32,+stride: U32,+initial: Accumulator,
  +input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+output_length: Nat,+index: Nat,
  +ready: Buffer.ContentReady<Element,Buffer.Content{output,output_length}>,
  +capacity_ok: {U32.is_le(Tree.capacity(Element,output),2147483648) == True{} : Bool},
  +room: {Nat.is_le(count,U32.to_nat(Tree.capacity(Element,output))) == True{} : Bool},
  +inside: {Nat.is_lt(index,count) == True{} : Bool},
  +covered: {Nat.is_le(Nat.mul(count,runs),ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}) ->
  {Tree.read(Element,CopyValues.output_tree(~Element,AccessLaws.lines_model(~Element,~Accumulator,~operation,~finish,count,Views.View{axes,offset},runs,extent,stride,initial,input,output)),
      U32.from_nat(index)) ==
    finish(line_fold(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,Nat.mul(runs,U32.to_nat(extent)),
      CopyValues.Sampled{Views.View{List.append(&2,Views.Axis,axes,[Views.Axis{extent,stride}]),offset},input},Nat.mul(Nat.mul(index,runs),U32.to_nat(extent)),initial)) : Element}:
  Buffer.ContentReady{+depth,+balanced,+depth_ok,filled} = ready
  +view = {Views.View{axes,offset} : Views.View}
  +start = {AccessLaws.LineCoordinates{CursorLaws.initial(view),runs,extent,stride,initial} : AccessLaws.LineCoordinates<Accumulator>}
  +part = {Folds{start,input} : Folds<Accumulator,Element>}
  +visit = CopyValues.visit_writes(~Element,~AccessLaws.LineCoordinates<Accumulator>,~AccessLaws.sample_line(~Element,~Accumulator,~operation,~finish),
    ~AccessLaws.next_line(~Accumulator),count,0,start,input)
  +lines = Consecutive.writes(~Element,~Folds<Accumulator,Element>,~fold_at(~Element,~Accumulator,~operation,~finish),count,0n,0n,part)
  +line = Nat.mul(index,runs)
  +value = finish(line_fold(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,Nat.mul(runs,U32.to_nat(extent)),
    CopyValues.Sampled{Views.View{List.append(&2,Views.Axis,axes,[Views.Axis{extent,stride}]),offset},input},Nat.mul(line,U32.to_nat(extent)),initial))
  +reached = Order.transitive(Nat.add(line,runs),Nat.mul(count,runs),ShapeLaws.elements(Views.axis_extents(axes)),
    %Order.commutative(runs,line) : {Nat.is_le(_,Nat.mul(count,runs)) == True{} : Bool}
    RowMajor.mul_le(1n+index,count,runs,Order.strict_follow(index,count,inside)),covered)
  +folded = {
    %Equal.sym(AccessLaws.LineCoordinates<Accumulator>,Loop.run(~AccessLaws.LineCoordinates<Accumulator>,~AccessLaws.next_line(~Accumulator),index,start),
        AccessLaws.LineCoordinates{cursor_at(view,line),runs,extent,stride,initial},lines_after(~Accumulator,index,0n,view,runs,extent,stride,initial)) :
      {AccessLaws.sample_line(~Element,~Accumulator,~operation,~finish,input,_) == value : Element}
    Equal.cong(Accumulator,Element,finish,AccessLaws.run_values(~Element,~Accumulator,~operation,runs,input,cursor_at(view,line),extent,stride,initial),
      line_fold(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,Nat.mul(runs,U32.to_nat(extent)),
        CopyValues.Sampled{Views.View{List.append(&2,Views.Axis,axes,[Views.Axis{extent,stride}]),offset},input},Nat.mul(line,U32.to_nat(extent)),initial),
      runs_fold(~Element,~Accumulator,~operation,runs,line,axes,extent,stride,offset,input,initial,reached)) :
    {fold_at(~Element,~Accumulator,~operation,~finish,part,index) == value : Element}}
  %Equal.sym(Tree.Tree<Element>,
      CopyValues.output_tree(~Element,AccessLaws.lines_model(~Element,~Accumulator,~operation,~finish,count,view,runs,extent,stride,initial,input,output)),
      Writes.pure(Element,visit,output),
      CopyValues.visit_program(~Element,~AccessLaws.LineCoordinates<Accumulator>,~AccessLaws.sample_line(~Element,~Accumulator,~operation,~finish),
        ~AccessLaws.next_line(~Accumulator),count,0,start,input,output)) :
    {Tree.read(Element,_,U32.from_nat(index)) == value : Element}
  %Equal.sym(List<&2,Writes.Assignment<Element>>,visit,lines,
      CopyValues.visit_line(~Element,~AccessLaws.LineCoordinates<Accumulator>,~Folds<Accumulator,Element>,~AccessLaws.sample_line(~Element,~Accumulator,~operation,~finish),
        ~AccessLaws.next_line(~Accumulator),~fold_at(~Element,~Accumulator,~operation,~finish),~start,~input,~part,~(position => {==}),count,0n)) :
    {Tree.read(Element,Writes.pure(Element,_,output),U32.from_nat(index)) == value : Element}
  Equal.trans(Element,Tree.read(Element,Writes.pure(Element,lines,output),U32.from_nat(index)),fold_at(~Element,~Accumulator,~operation,~finish,part,index),value,
    Consecutive.read(~Element,~Folds<Accumulator,Element>,~fold_at(~Element,~Accumulator,~operation,~finish),count,0n,part,output,depth,index,balanced,depth_ok,capacity_ok,room,inside),
    folded)

# ---- The lines of a view after its kept axes ----

def extents_appended(+left: List<&2,Views.Axis>,+right: List<&2,Views.Axis>) ->
  {Views.axis_extents(List.append(&2,Views.Axis,left,right)) == List.append(&2,U32,Views.axis_extents(left),Views.axis_extents(right)) : List<&2,U32>}:
  match left:
    case []: {==}
    case Views.Axis{+extent,+stride} <> +tail:
      Equal.cong(List<&2,U32>,List<&2,U32>,others => extent <> others,Views.axis_extents(List.append(&2,Views.Axis,tail,right)),
        List.append(&2,U32,Views.axis_extents(tail),Views.axis_extents(right)),extents_appended(tail,right))

def axes_elements(+left: List<&2,Views.Axis>,+right: List<&2,Views.Axis>) ->
  {ShapeLaws.elements(Views.axis_extents(List.append(&2,Views.Axis,left,right))) ==
    Nat.mul(ShapeLaws.elements(Views.axis_extents(left)),ShapeLaws.elements(Views.axis_extents(right))) : Nat}:
  %Equal.sym(List<&2,U32>,Views.axis_extents(List.append(&2,Views.Axis,left,right)),List.append(&2,U32,Views.axis_extents(left),Views.axis_extents(right)),extents_appended(left,right)) :
    {ShapeLaws.elements(_) == Nat.mul(ShapeLaws.elements(Views.axis_extents(left)),ShapeLaws.elements(Views.axis_extents(right))) : Nat}
  ShapeLaws.elements_appended(Views.axis_extents(left),Views.axis_extents(right))

def elements_of_exact(+axes: List<&2,Views.Axis>) -> {Operations.elements_of(axes) == ShapeLaws.elements(Views.axis_extents(axes)) : Nat}:
  match axes:
    case []: {==}
    case Views.Axis{+extent,+stride} <> +tail:
      Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(extent),value),Operations.elements_of(tail),ShapeLaws.elements(Views.axis_extents(tail)),elements_of_exact(tail))

def append_associative(~Element: Data,+first: List<&2,Element>,+second: List<&2,Element>,+third: List<&2,Element>) ->
  {List.append(&2,Element,List.append(&2,Element,first,second),third) == List.append(&2,Element,first,List.append(&2,Element,second,third)) : List<&2,Element>}:
  match first:
    case []: {==}
    case +head <> +tail:
      Equal.cong(List<&2,Element>,List<&2,Element>,others => head <> others,List.append(&2,Element,List.append(&2,Element,tail,second),third),
        List.append(&2,Element,tail,List.append(&2,Element,second,third)),append_associative(~Element,tail,second,third))

def append_empty(~Element: Data,+items: List<&2,Element>) -> {List.append(&2,Element,items,[]) == items : List<&2,Element>}:
  match items:
    case []: {==}
    case +head <> +tail:
      Equal.cong(List<&2,Element>,List<&2,Element>,others => head <> others,List.append(&2,Element,tail,[]),tail,append_empty(~Element,tail))

def take_drop(~Element: Data,+items: List<&2,Element>,+count: Nat) ->
  {List.append(&2,Element,List.take(&2,Element,items,count),List.drop(&2,Element,items,count)) == items : List<&2,Element>}:
  match items count:
    case [] count: {==}
    case +head <> +tail 0n: {==}
    case +head <> +tail 1n+ +previous:
      Equal.cong(List<&2,Element>,List<&2,Element>,others => head <> others,
        List.append(&2,Element,List.take(&2,Element,tail,previous),List.drop(&2,Element,tail,previous)),tail,take_drop(~Element,tail,previous))

# The runs of a line joined back.
def joined(runs: List<&2,Views.Axis> & Views.Axis) -> List<&2,Views.Axis>:
  (outer,last) = runs
  List.append(&2,Views.Axis,outer,[last])

def prepended(+axis: Views.Axis,runs: List<&2,Views.Axis> & Views.Axis) -> {joined(Operations.prepend_run(axis,runs)) == axis <> joined(runs) : List<&2,Views.Axis>}:
  (outer,last) = runs
  {==}

def runs_joined(+rest: List<&2,Views.Axis>,+axis: Views.Axis) -> {joined(Operations.runs_after(rest,axis)) == axis <> rest : List<&2,Views.Axis>}:
  match rest:
    case []: {==}
    case +next <> +tail:
      Equal.trans(List<&2,Views.Axis>,joined(Operations.prepend_run(axis,Operations.runs_after(tail,next))),axis <> joined(Operations.runs_after(tail,next)),axis <> next <> tail,
        prepended(axis,Operations.runs_after(tail,next)),
        Equal.cong(List<&2,Views.Axis>,List<&2,Views.Axis>,others => axis <> others,joined(Operations.runs_after(tail,next)),next <> tail,runs_joined(tail,next)))

# Output index of the runs of a line holds finish of the fold of the values
# of its joined axes after the kept axes, from index times their elements on.
def runs_value(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +count: Nat,+kept: List<&2,Views.Axis>,+offset: U32,runs: List<&2,Views.Axis> & Views.Axis,+initial: Accumulator,
  +input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+output_length: Nat,+index: Nat,
  +ready: Buffer.ContentReady<Element,Buffer.Content{output,output_length}>,
  +capacity_ok: {U32.is_le(Tree.capacity(Element,output),2147483648) == True{} : Bool},
  +room: {Nat.is_le(count,U32.to_nat(Tree.capacity(Element,output))) == True{} : Bool},
  +inside: {Nat.is_lt(index,count) == True{} : Bool},
  +counted: {count == ShapeLaws.elements(Views.axis_extents(kept)) : Nat}) ->
  {Tree.read(Element,CopyValues.output_tree(~Element,OperationsLaws.runs_model(~Element,~Accumulator,~operation,~finish,count,kept,offset,runs,initial,input,output)),
      U32.from_nat(index)) ==
    finish(line_fold(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,ShapeLaws.elements(Views.axis_extents(joined(runs))),
      CopyValues.Sampled{Views.View{List.append(&2,Views.Axis,kept,joined(runs)),offset},input},Nat.mul(index,ShapeLaws.elements(Views.axis_extents(joined(runs)))),initial)) : Element}:
  (+outer,Views.Axis{+extent,+stride}) = runs
  +last = {[Views.Axis{extent,stride}] : List<&2,Views.Axis>}
  +groups = Operations.elements_of(outer)
  +width = U32.to_nat(extent)
  +length = {
    %Equal.sym(Nat,groups,ShapeLaws.elements(Views.axis_extents(outer)),elements_of_exact(outer)) : {ShapeLaws.elements(Views.axis_extents(List.append(&2,Views.Axis,outer,last))) == Nat.mul(_,width) : Nat}
    %Order.times_one(width) : {ShapeLaws.elements(Views.axis_extents(List.append(&2,Views.Axis,outer,last))) == Nat.mul(ShapeLaws.elements(Views.axis_extents(outer)),_) : Nat}
    axes_elements(outer,last) : {ShapeLaws.elements(Views.axis_extents(List.append(&2,Views.Axis,outer,last))) == Nat.mul(groups,width) : Nat}}
  +covered = {
    %Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(List.append(&2,Views.Axis,kept,outer))),Nat.mul(ShapeLaws.elements(Views.axis_extents(kept)),ShapeLaws.elements(Views.axis_extents(outer))),axes_elements(kept,outer)) : {Nat.is_le(Nat.mul(count,groups),_) == True{} : Bool}
    %counted : {Nat.is_le(Nat.mul(count,groups),Nat.mul(_,ShapeLaws.elements(Views.axis_extents(outer)))) == True{} : Bool}
    %elements_of_exact(outer) : {Nat.is_le(Nat.mul(count,groups),Nat.mul(count,_)) == True{} : Bool}
    Order.reflexive(Nat.mul(count,groups)) :
    {Nat.is_le(Nat.mul(count,groups),ShapeLaws.elements(Views.axis_extents(List.append(&2,Views.Axis,kept,outer)))) == True{} : Bool}}
  %append_associative(~Views.Axis,kept,outer,last) :
    {Tree.read(Element,CopyValues.output_tree(~Element,AccessLaws.lines_model(~Element,~Accumulator,~operation,~finish,count,Views.View{List.append(&2,Views.Axis,kept,outer),offset},
        groups,extent,stride,initial,input,output)),U32.from_nat(index)) ==
      finish(line_fold(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,ShapeLaws.elements(Views.axis_extents(List.append(&2,Views.Axis,outer,last))),
        CopyValues.Sampled{Views.View{_,offset},input},Nat.mul(index,ShapeLaws.elements(Views.axis_extents(List.append(&2,Views.Axis,outer,last)))),initial)) : Element}
  %Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(List.append(&2,Views.Axis,outer,last))),Nat.mul(groups,width),length) :
    {Tree.read(Element,CopyValues.output_tree(~Element,AccessLaws.lines_model(~Element,~Accumulator,~operation,~finish,count,Views.View{List.append(&2,Views.Axis,kept,outer),offset},
        groups,extent,stride,initial,input,output)),U32.from_nat(index)) ==
      finish(line_fold(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,_,
        CopyValues.Sampled{Views.View{List.append(&2,Views.Axis,List.append(&2,Views.Axis,kept,outer),last),offset},input},Nat.mul(index,_),initial)) : Element}
  %RowMajor.assoc(index,groups,width) :
    {Tree.read(Element,CopyValues.output_tree(~Element,AccessLaws.lines_model(~Element,~Accumulator,~operation,~finish,count,Views.View{List.append(&2,Views.Axis,kept,outer),offset},
        groups,extent,stride,initial,input,output)),U32.from_nat(index)) ==
      finish(line_fold(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,Nat.mul(groups,width),
        CopyValues.Sampled{Views.View{List.append(&2,Views.Axis,List.append(&2,Views.Axis,kept,outer),last),offset},input},_,initial)) : Element}
  line_value(~Element,~Accumulator,~operation,~finish,count,List.append(&2,Views.Axis,kept,outer),offset,groups,extent,stride,initial,input,output,output_length,index,
    ready,capacity_ok,room,inside,covered)

# A line of no axis holds the one value of the kept axes at its index.
def single_value(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +count: Nat,+kept: List<&2,Views.Axis>,+offset: U32,+initial: Accumulator,
  +input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+output_length: Nat,+index: Nat,
  +ready: Buffer.ContentReady<Element,Buffer.Content{output,output_length}>,
  +capacity_ok: {U32.is_le(Tree.capacity(Element,output),2147483648) == True{} : Bool},
  +room: {Nat.is_le(count,U32.to_nat(Tree.capacity(Element,output))) == True{} : Bool},
  +inside: {Nat.is_lt(index,count) == True{} : Bool},
  +counted: {count == ShapeLaws.elements(Views.axis_extents(kept)) : Nat}) ->
  {Tree.read(Element,CopyValues.output_tree(~Element,OperationsLaws.runs_model(~Element,~Accumulator,~operation,~finish,count,kept,offset,Operations.line_runs([]),initial,input,output)),
      U32.from_nat(index)) ==
    finish(line_fold(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,1n,
      CopyValues.Sampled{Views.View{List.append(&2,Views.Axis,kept,[]),offset},input},Nat.mul(index,1n),initial)) : Element}:
  +axes = List.append(&2,Views.Axis,kept,[])
  +unit = {[Views.Axis{1,0}] : List<&2,Views.Axis>}
  +single = Nat.mul(index,1n)
  +covered = {
    %Equal.sym(List<&2,Views.Axis>,axes,kept,append_empty(~Views.Axis,kept)) : {Nat.is_le(Nat.mul(count,1n),ShapeLaws.elements(Views.axis_extents(_))) == True{} : Bool}
    %counted : {Nat.is_le(Nat.mul(count,1n),_) == True{} : Bool}
    %Equal.sym(Nat,Nat.mul(count,1n),count,Order.times_one(count)) : {Nat.is_le(_,count) == True{} : Bool}
    Order.reflexive(count) : {Nat.is_le(Nat.mul(count,1n),ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}}
  +run_inside = {
    %Equal.sym(List<&2,Views.Axis>,axes,kept,append_empty(~Views.Axis,kept)) : {Nat.is_lt(single,ShapeLaws.elements(Views.axis_extents(_))) == True{} : Bool}
    %counted : {Nat.is_lt(single,_) == True{} : Bool}
    %Equal.sym(Nat,single,index,Order.times_one(index)) : {Nat.is_lt(_,count) == True{} : Bool}
    inside : {Nat.is_lt(single,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool}}
  +address = {
    %Arithmetic.add_zero(CopyValues.address(Views.View{axes,offset},single)) :
      {CopyValues.address(Views.View{List.append(&2,Views.Axis,axes,unit),offset},Nat.mul(Nat.mul(index,1n),1n)) == _ : U32}
    %Addition.zero_right(Nat.mul(single,1n)) :
      {CopyValues.address(Views.View{List.append(&2,Views.Axis,axes,unit),offset},_) == (CopyValues.address(Views.View{axes,offset},single) + 0 : U32) : U32}
    Equal.sym(U32,(CopyValues.address(Views.View{axes,offset},single) + U32.from_nat(0n) * 0 : U32),
      CopyValues.address(Views.View{List.append(&2,Views.Axis,axes,unit),offset},Nat.add(Nat.mul(single,1n),0n)),run_element(axes,1,0,offset,single,0n,run_inside,{==})) :
    {CopyValues.address(Views.View{List.append(&2,Views.Axis,axes,unit),offset},Nat.mul(Nat.mul(index,1n),1n)) == CopyValues.address(Views.View{axes,offset},single) : U32}}
  Equal.trans(Element,
    Tree.read(Element,CopyValues.output_tree(~Element,AccessLaws.lines_model(~Element,~Accumulator,~operation,~finish,count,Views.View{axes,offset},1n,1,0,initial,input,output)),
      U32.from_nat(index)),
    finish(line_fold(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,1n,
      CopyValues.Sampled{Views.View{List.append(&2,Views.Axis,axes,unit),offset},input},Nat.mul(Nat.mul(index,1n),1n),initial)),
    finish(line_fold(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,1n,
      CopyValues.Sampled{Views.View{axes,offset},input},single,initial)),
    line_value(~Element,~Accumulator,~operation,~finish,count,axes,offset,1n,1,0,initial,input,output,output_length,index,ready,capacity_ok,room,inside,covered),
    Equal.cong(U32,Element,word => finish(operation(initial,Tree.read(Element,input,word))),
      CopyValues.address(Views.View{List.append(&2,Views.Axis,axes,unit),offset},Nat.mul(Nat.mul(index,1n),1n)),CopyValues.address(Views.View{axes,offset},single),address))

# Output index of the lines of the line axes after the kept axes holds
# finish of the fold of the values of the line axes at that index of the
# kept axes, in row-major order.
def lines_value(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +count: Nat,+kept: List<&2,Views.Axis>,+line: List<&2,Views.Axis>,+offset: U32,+initial: Accumulator,
  +input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+output_length: Nat,+index: Nat,
  +ready: Buffer.ContentReady<Element,Buffer.Content{output,output_length}>,
  +capacity_ok: {U32.is_le(Tree.capacity(Element,output),2147483648) == True{} : Bool},
  +room: {Nat.is_le(count,U32.to_nat(Tree.capacity(Element,output))) == True{} : Bool},
  +inside: {Nat.is_lt(index,count) == True{} : Bool},
  +counted: {count == ShapeLaws.elements(Views.axis_extents(kept)) : Nat}) ->
  {Tree.read(Element,CopyValues.output_tree(~Element,OperationsLaws.runs_model(~Element,~Accumulator,~operation,~finish,count,kept,offset,Operations.line_runs(line),initial,input,output)),
      U32.from_nat(index)) ==
    finish(line_fold(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,ShapeLaws.elements(Views.axis_extents(line)),
      CopyValues.Sampled{Views.View{List.append(&2,Views.Axis,kept,line),offset},input},Nat.mul(index,ShapeLaws.elements(Views.axis_extents(line))),initial)) : Element}:
  match line:
    case []: single_value(~Element,~Accumulator,~operation,~finish,count,kept,offset,initial,input,output,output_length,index,ready,capacity_ok,room,inside,counted)
    case +axis <> +rest:
      %runs_joined(rest,axis) :
        {Tree.read(Element,CopyValues.output_tree(~Element,OperationsLaws.runs_model(~Element,~Accumulator,~operation,~finish,count,kept,offset,Operations.runs_after(rest,axis),initial,
            input,output)),U32.from_nat(index)) ==
          finish(line_fold(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,ShapeLaws.elements(Views.axis_extents(_)),
            CopyValues.Sampled{Views.View{List.append(&2,Views.Axis,kept,_),offset},input},Nat.mul(index,ShapeLaws.elements(Views.axis_extents(_))),initial)) : Element}
      runs_value(~Element,~Accumulator,~operation,~finish,count,kept,offset,Operations.runs_after(rest,axis),initial,input,output,output_length,index,
        ready,capacity_ok,room,inside,counted)
