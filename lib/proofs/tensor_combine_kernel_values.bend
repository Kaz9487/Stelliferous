# The values of the elementwise combination kernel on storage. The kernel
# reads the right operand through a view: at every index of the left operand
# it holds the operation applied to the left value at that index and the
# right value at the address of that index in the view.
import Base
import ../tensor.bend as Tensors
import ../tensor_shape.bend as Shapes
import ../tensor_view.bend as Views
import ../tensor_access.bend as Access
import ../tensor_operations.bend as Operations
import ../storage_buffer.bend as Storage
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_certificate.bend as Certificate
import ./storage_shape.bend as Shape
import ./storage_refinement.bend as StorageLaws
import ./tensor_contract.bend as Contract
import ./tensor_copy_values.bend as CopyValues
import ./tensor_combine_values.bend as CombineValues
import ./tensor_gather_values.bend as GatherValues
import ./tensor_band_combine_leaf.bend as BandCombineLeaf
import ./tensor_access_refinement.bend as AccessLaws
import ./tensor_operations_refinement.bend as OperationsLaws
import ./tensor_view_metadata_refinement.bend as Metadata
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_count_completion.bend as CountCompletion
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./boolean_logic.bend as Boolean
import ./word_encoding.bend as WordEncoding
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges

# The two logical operands and the view the right one is read through.
type Combine<-Left: Data,-Right: Data> is Data:
  Combine{left: Left,right: Right,source: Views.View}

def value(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,context: Combine<Left,Right>,index: U32) -> Element:
  Combine{left,right,source} = context
  +index = index
  operation(left_value(left,index),right_value(right,CopyValues.address(source,U32.to_nat(index))))

def combined_content(~Element: Data,~operation: Element -> Element -> Element,+count: U32,+source: Views.View,output: Buffer.Content<Element>,input: Buffer.Content<Element>) ->
  Buffer.Content<Element>:
  Buffer.Content{output,length} = output
  Buffer.Content{input,input_length} = input
  Buffer.Content{BandCombineLeaf.readback_output(~Element,AccessLaws.combine_model(~Element,~operation,Access.zero_contiguous(source) && U32.is_zero(0),U32.to_nat(count),source,0,input,output)),length}

def and_true(+value: Bool) -> {value == (value && True{}) : Bool}:
  match value:
    case True{}: {==}
    case False{}: {==}

# One index of the combination into a left operand at offset zero.
def cell(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,+left: Left,+right: Right,+source: Views.View,+output: Buffer.Model<Element>,+input: Buffer.Model<Element>,+count: U32,+index: Nat,
  ready: Buffer.Sized<Element,output>,input_ready: Buffer.Sized<Element,input>,left_values: Contract.Values(~Element,~Left,~left_value,left,Buffer.content(~Element,output)),right_values: Contract.Values(~Element,~Right,~right_value,right,Buffer.content(~Element,input)),
  +written: {U32.to_nat(count) == Buffer.length(~Element,output) : Nat},
  +exact: {ShapeLaws.elements(Views.axis_extents_of(source)) == U32.to_nat(count) : Nat},
  fits: {Views.fits(source,U32.from_nat(Buffer.length(~Element,input))) == True{} : Bool},
  +inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  Contract.At(~Element,~Combine<Left,Right>,~value(~Element,~Left,~Right,~left_value,~right_value,~operation),Combine{left,right,source},combined_content(~Element,~operation,count,source,Buffer.content(~Element,output),Buffer.content(~Element,input)),index):
  Buffer.Model{+output_tree,+output_length,+output_certificate} = output
  Buffer.Model{+input_tree,+input_length,+input_certificate} = input
  Buffer.Sized{+depth_ok,+room} = ready
  Buffer.Sized{+input_depth_ok,+input_room} = input_ready
  +output = {Buffer.Model{output_tree,output_length,output_certificate} : Buffer.Model<Element>}
  +input = {Buffer.Model{input_tree,input_length,input_certificate} : Buffer.Model<Element>}
  +capacity = Tree.capacity(Element,output_tree)
  +capacity_ok = Buffer.capacity_bound(~Element,output,depth_ok)
  +count_room = {
    %Equal.sym(Nat,U32.to_nat(count),output_length,written) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    room : {Nat.is_le(U32.to_nat(count),U32.to_nat(capacity)) == True{} : Bool}}
  +encoded = Bounds.roundtrip(index,capacity,capacity_ok,
    Bounds.strict_le(index,U32.to_nat(capacity),Bounds.nat_trans(index,U32.to_nat(count),U32.to_nat(capacity),inside,count_room)))
  +nonempty = CountCompletion.positive_not_zero(Views.axis_extents_of(source),
    %Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents_of(source)),U32.to_nat(count),exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.le_lt_trans(0n,index,U32.to_nat(count),Order.zero_le(index),inside))
  +address = CopyValues.address(source,index)
  +address_inside = {
    %Bounds.roundtrip(input_length,Tree.capacity(Element,input_tree),Buffer.capacity_bound(~Element,input,input_depth_ok),input_room) :
      {Nat.is_lt(U32.to_nat(address),_) == True{} : Bool}
    CopyValues.address_inside(source,index,U32.from_nat(input_length),
      %nonempty : {Views.fits_nonempty(_,source,U32.from_nat(input_length)) == True{} : Bool}
      fits) : {Nat.is_lt(U32.to_nat(address),input_length) == True{} : Bool}}
  +position = U32.from_nat(index)
  Equal.trans(Element,
    Tree.read(Element,BandCombineLeaf.readback_output(~Element,AccessLaws.combine_model(~Element,~operation,Access.zero_contiguous(source) && U32.is_zero(0),U32.to_nat(count),source,0,input_tree,output_tree)),position),
    operation(Tree.read(Element,output_tree,position),Tree.read(Element,input_tree,address)),
    operation(left_value(left,position),right_value(right,CopyValues.address(source,U32.to_nat(position)))),
    CombineValues.combined_value(~Element,~operation,Access.zero_contiguous(source) && U32.is_zero(0),source,input_tree,output_tree,output_length,U32.to_nat(count),index,and_true(Access.zero_contiguous(source)),
      Buffer.content_ready(~Element,output,Buffer.Sized{depth_ok,room}),capacity_ok,count_room,inside,exact),
    %Equal.sym(Nat,U32.to_nat(position),index,encoded) :
      {operation(Tree.read(Element,output_tree,position),Tree.read(Element,input_tree,address)) == operation(left_value(left,position),right_value(right,CopyValues.address(source,_))) : Element}
    Equal.trans(Element,operation(Tree.read(Element,output_tree,position),Tree.read(Element,input_tree,address)),
      operation(left_value(left,position),Tree.read(Element,input_tree,address)),operation(left_value(left,position),right_value(right,address)),
      Equal.cong(Element,Element,element => operation(element,Tree.read(Element,input_tree,address)),Tree.read(Element,output_tree,position),left_value(left,position),
        Contract.read(~Element,~Left,~left_value,left,Buffer.Content{output_tree,output_length},index,left_values,
          %written : {Nat.is_lt(index,_) == True{} : Bool} inside)),
      Equal.cong(Element,Element,element => operation(left_value(left,position),element),Tree.read(Element,input_tree,address),right_value(right,address),
        %Arithmetic.native_roundtrip(address) : {Tree.read(Element,input_tree,_) == right_value(right,_) : Element}
        Contract.read(~Element,~Right,~right_value,right,Buffer.Content{input_tree,input_length},U32.to_nat(address),right_values,address_inside))))

def cells_range(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,remaining: Nat,+position: Nat,+left: Left,+right: Right,+source: Views.View,+output: Buffer.Model<Element>,+input: Buffer.Model<Element>,+count: U32,
  +ready: Buffer.Sized<Element,output>,+input_ready: Buffer.Sized<Element,input>,+left_values: Contract.Values(~Element,~Left,~left_value,left,Buffer.content(~Element,output)),+right_values: Contract.Values(~Element,~Right,~right_value,right,Buffer.content(~Element,input)),
  +written: {U32.to_nat(count) == Buffer.length(~Element,output) : Nat},
  +exact: {ShapeLaws.elements(Views.axis_extents_of(source)) == U32.to_nat(count) : Nat},
  +fits: {Views.fits(source,U32.from_nat(Buffer.length(~Element,input))) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(position,remaining),U32.to_nat(count)) == True{} : Bool}) ->
  Members.All(Nat,{index => Contract.At(~Element,~Combine<Left,Right>,~value(~Element,~Left,~Right,~left_value,~right_value,~operation),Combine{left,right,source},combined_content(~Element,~operation,count,source,Buffer.content(~Element,output),Buffer.content(~Element,input)),index) : Nat -> Data},Ranges.range(remaining,position)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      Shape.Both{cell(~Element,~Left,~Right,~left_value,~right_value,~operation,left,right,source,output,input,count,position,ready,input_ready,left_values,right_values,written,exact,fits,Order.room_index(position,rest,U32.to_nat(count),room)),
        cells_range(~Element,~Left,~Right,~left_value,~right_value,~operation,rest,1n+position,left,right,source,output,input,count,ready,input_ready,left_values,right_values,written,exact,fits,Order.advance_room(position,rest,U32.to_nat(count),room))}

def program_length(~Element: Data,~operation: Element -> Element -> Element,+count: U32,+source: Views.View,+output: Buffer.Model<Element>,+input: Buffer.Model<Element>) ->
  {Buffer.content_length(~Element,combined_content(~Element,~operation,count,source,Buffer.content(~Element,output),Buffer.content(~Element,input))) == Buffer.length(~Element,output) : Nat}:
  Buffer.Model{output_tree,output_length,output_certificate} = output
  Buffer.Model{input_tree,input_length,input_certificate} = input
  {==}

def program_ready(~Element: Data,~operation: Element -> Element -> Element,+count: U32,+source: Views.View,+output: Buffer.Model<Element>,+input: Buffer.Model<Element>,ready: Buffer.Sized<Element,output>) ->
  Buffer.ContentReady<Element,combined_content(~Element,~operation,count,source,Buffer.content(~Element,output),Buffer.content(~Element,input))>:
  Buffer.Model{+output_tree,+output_length,+output_certificate} = output
  Buffer.Model{+input_tree,input_length,input_certificate} = input
  CombineValues.combined_ready(~Element,~operation,Access.zero_contiguous(source) && U32.is_zero(0),U32.to_nat(count),source,input_tree,output_tree,output_length,
    Buffer.content_ready(~Element,Buffer.Model{output_tree,output_length,output_certificate},ready))

def program_facts(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,+left: Left,+right: Right,+source: Views.View,+output: Buffer.Model<Element>,+input: Buffer.Model<Element>,+count: U32,+left_shape: List<&2,U32>,
  +ready: Buffer.Sized<Element,output>,+input_ready: Buffer.Sized<Element,input>,+left_values: Contract.Values(~Element,~Left,~left_value,left,Buffer.content(~Element,output)),+right_values: Contract.Values(~Element,~Right,~right_value,right,Buffer.content(~Element,input)),
  +written: {U32.to_nat(count) == Buffer.length(~Element,output) : Nat},
  +exact: {ShapeLaws.elements(Views.axis_extents_of(source)) == U32.to_nat(count) : Nat},
  +fits: {Views.fits(source,U32.from_nat(Buffer.length(~Element,input))) == True{} : Bool},
  length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(left_shape) : Nat}) ->
  GatherValues.Facts(~Element,~Combine<Left,Right>,~value(~Element,~Left,~Right,~left_value,~right_value,~operation),Combine{left,right,source},left_shape,combined_content(~Element,~operation,count,source,Buffer.content(~Element,output),Buffer.content(~Element,input))):
  Shape.Both{
    Contract.WellFormed{
      Equal.trans(Nat,Buffer.content_length(~Element,combined_content(~Element,~operation,count,source,Buffer.content(~Element,output),Buffer.content(~Element,input))),Buffer.length(~Element,output),ShapeLaws.elements(left_shape),
        program_length(~Element,~operation,count,source,output,input),length_exact),
      program_ready(~Element,~operation,count,source,output,input,ready)},
    %Equal.sym(Nat,Buffer.content_length(~Element,combined_content(~Element,~operation,count,source,Buffer.content(~Element,output),Buffer.content(~Element,input))),U32.to_nat(count),
      Equal.trans(Nat,Buffer.content_length(~Element,combined_content(~Element,~operation,count,source,Buffer.content(~Element,output),Buffer.content(~Element,input))),Buffer.length(~Element,output),U32.to_nat(count),
        program_length(~Element,~operation,count,source,output,input),Equal.sym(Nat,U32.to_nat(count),Buffer.length(~Element,output),written))) :
      Members.All(Nat,{index => Contract.At(~Element,~Combine<Left,Right>,~value(~Element,~Left,~Right,~left_value,~right_value,~operation),Combine{left,right,source},combined_content(~Element,~operation,count,source,Buffer.content(~Element,output),Buffer.content(~Element,input)),index) : Nat -> Data},Ranges.range(_,0n))
    cells_range(~Element,~Left,~Right,~left_value,~right_value,~operation,U32.to_nat(count),0n,left,right,source,output,input,count,ready,input_ready,left_values,right_values,written,exact,fits,Order.reflexive(U32.to_nat(count)))}

# The guard of the checked combination, over the models of its two owners.
def guard(~Element: Data,+target: Views.View,+source: Views.View,+output: Buffer.Model<Element>,+input: Buffer.Model<Element>) -> Bool:
  Shapes.same(Views.shape(target),Views.shape(source)) && Views.is_contiguous(target) &&
    Operations.valid_buffer(Buffer.length(~Element,output),U32.shln(1,Buffer.depth(~Element,output)),target) &&
    Operations.valid_buffer(Buffer.length(~Element,input),U32.shln(1,Buffer.depth(~Element,input)),source)

type Counted<-target: Views.View> is Data:
  Counted{count: U32,counted: {Views.numel(target) == Some{count} : Maybe<&2,U32>}}

# An accepted combination passed its guard and has the count of its target.
def model_guard(~Element: Data,~operation: Element -> Element -> Element,valid: Bool,count: Maybe<&2,U32>,+source: Views.View,+target: Views.View,+input_length: Nat,+output_length: Nat,
  +input: Tree.Tree<Element>,+output: Tree.Tree<Element>,+counts: {Views.numel(target) == count : Maybe<&2,U32>},
  accepted: {GatherValues.accepted_owners(~Element,OperationsLaws.combine_model(~Element,~operation,OperationsLaws.PairCall{valid,count,source,target,input_length,output_length},input,output)) == True{} : Bool}) ->
  Shape.Both<{valid == True{} : Bool},Counted<target>>:
  match valid count:
    case True{} Some{+count}: Shape.Both{{==},Counted{count,counts}}
    case True{} None{}: Empty.absurd(Shape.Both<{True{} == True{} : Bool},Counted<target>>,WordEncoding.false_true(accepted))
    case False{} Some{count}: Empty.absurd(Shape.Both<{False{} == True{} : Bool},Counted<target>>,WordEncoding.false_true(accepted))
    case False{} None{}: Empty.absurd(Shape.Both<{False{} == True{} : Bool},Counted<target>>,WordEncoding.false_true(accepted))

type Guarded<-Element: Data,-target: Views.View,-source: Views.View,-output: Buffer.Model<Element>,-input: Buffer.Model<Element>> is Data:
  Guarded{count: U32,counted: {Views.numel(target) == Some{count} : Maybe<&2,U32>},
    guard: {guard(~Element,target,source,output,input) == True{} : Bool},
    same: {Shapes.same(Views.shape(target),Views.shape(source)) == True{} : Bool},
    fits: {Views.fits(source,U32.from_nat(Buffer.length(~Element,input))) == True{} : Bool}}

def guarded_selected(~Element: Data,+target: Views.View,+source: Views.View,+output: Buffer.Model<Element>,+input: Buffer.Model<Element>,
  facts: Shape.Both<{guard(~Element,target,source,output,input) == True{} : Bool},Counted<target>>) -> Guarded<Element,target,source,output,input>:
  Buffer.Model{+output_tree,+output_length,Certificate.Certificate{+output_depth,+output_equation}} = output
  Buffer.Model{+input_tree,+input_length,Certificate.Certificate{+input_depth,+input_equation}} = input
  Shape.Both{+truth,Counted{+count,+counted}} = facts
  +same = Shapes.same(Views.shape(target),Views.shape(source))
  +contiguous = Views.is_contiguous(target)
  +output_valid = Operations.valid_buffer(output_length,U32.shln(1,output_depth),target)
  +input_valid = Operations.valid_buffer(input_length,U32.shln(1,input_depth),source)
  Guarded{count,counted,truth,
    Boolean.left(same,contiguous,Boolean.left((same && contiguous : Bool),output_valid,Boolean.left((same && contiguous && output_valid : Bool),input_valid,truth))),
    Boolean.right(Nat.is_le(input_length,U32.to_nat(U32.shln(1,input_depth))),Views.fits(source,U32.from_nat(input_length)),
      Boolean.right((same && contiguous && output_valid : Bool),input_valid,truth))}

def guarded(~Element: Data,~operation: Element -> Element -> Element,+target: Views.View,+source: Views.View,+output: Buffer.Model<Element>,+input: Buffer.Model<Element>,
  accepted: {GatherValues.accepted_owners(~Element,OperationsLaws.combine_denotation(~Element,~operation,Buffer.pack(~Element,output),target,Buffer.pack(~Element,input),source)) == True{} : Bool}) ->
  Guarded<Element,target,source,output,input>:
  Buffer.Model{+output_tree,+output_length,Certificate.Certificate{+output_depth,+output_equation}} = output
  Buffer.Model{+input_tree,+input_length,Certificate.Certificate{+input_depth,+input_equation}} = input
  guarded_selected(~Element,target,source,Buffer.Model{output_tree,output_length,Certificate.Certificate{output_depth,output_equation}},
    Buffer.Model{input_tree,input_length,Certificate.Certificate{input_depth,input_equation}},
    model_guard(~Element,~operation,(Shapes.same(Views.shape(target),Views.shape(source)) && Views.is_contiguous(target) && Operations.valid_buffer(output_length,U32.shln(1,output_depth),target) && Operations.valid_buffer(input_length,U32.shln(1,input_depth),source) : Bool),Views.numel(target),source,target,input_length,output_length,
      Tree.reflect(Element,Tree.pack(Element,input_tree)),Tree.reflect(Element,Tree.pack(Element,output_tree)),{==},accepted))

# The observed left content of an accepted combination is the model above.
def denoted_content(~Element: Data,~operation: Element -> Element -> Element,+count: U32,+target_axes: List<&2,Views.Axis>,+source: Views.View,+output: Buffer.Model<Element>,+input: Buffer.Model<Element>,
  guarded: {guard(~Element,Views.View{target_axes,0},source,output,input) == True{} : Bool},counted: {Views.numel(Views.View{target_axes,0}) == Some{count} : Maybe<&2,U32>}) ->
  {BandCombineLeaf.left_content(~Element,OperationsLaws.combine_denotation(~Element,~operation,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source)) == combined_content(~Element,~operation,count,source,Buffer.content(~Element,output),Buffer.content(~Element,input)) : Buffer.Content<Element>}:
  Buffer.Model{+output_tree,+output_length,Certificate.Certificate{+output_depth,+output_equation}} = output
  Buffer.Model{+input_tree,+input_length,Certificate.Certificate{+input_depth,+input_equation}} = input
  +target = {Views.View{target_axes,0} : Views.View}
  +model = {Buffer.Content{BandCombineLeaf.readback_output(~Element,AccessLaws.combine_model(~Element,~operation,Access.zero_contiguous(source) && U32.is_zero(0),U32.to_nat(count),source,0,input_tree,output_tree)),output_length} : Buffer.Content<Element>}
  %Equal.sym(Bool,(Shapes.same(Views.shape(target),Views.shape(source)) && Views.is_contiguous(target) && Operations.valid_buffer(output_length,U32.shln(1,output_depth),target) && Operations.valid_buffer(input_length,U32.shln(1,input_depth),source) : Bool),True{},guarded) :
    {BandCombineLeaf.left_content(~Element,OperationsLaws.combine_model(~Element,~operation,OperationsLaws.PairCall{_,Views.numel(target),source,target,input_length,output_length},
      Tree.reflect(Element,Tree.pack(Element,input_tree)),Tree.reflect(Element,Tree.pack(Element,output_tree)))) == model : Buffer.Content<Element>}
  %Equal.sym(Maybe<&2,U32>,Views.numel(target),Some{count},counted) :
    {BandCombineLeaf.left_content(~Element,OperationsLaws.combine_model(~Element,~operation,OperationsLaws.PairCall{True{},_,source,target,input_length,output_length},
      Tree.reflect(Element,Tree.pack(Element,input_tree)),Tree.reflect(Element,Tree.pack(Element,output_tree)))) == model : Buffer.Content<Element>}
  %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,input_tree)),input_tree,StorageLaws.reflect_packed(Element,input_tree)) :
    {BandCombineLeaf.left_content(~Element,OperationsLaws.combine_model(~Element,~operation,OperationsLaws.PairCall{True{},Some{count},source,target,input_length,output_length},
      _,Tree.reflect(Element,Tree.pack(Element,output_tree)))) == model : Buffer.Content<Element>}
  %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,output_tree)),output_tree,StorageLaws.reflect_packed(Element,output_tree)) :
    {BandCombineLeaf.left_content(~Element,OperationsLaws.combine_model(~Element,~operation,OperationsLaws.PairCall{True{},Some{count},source,target,input_length,output_length},
      input_tree,_)) == model : Buffer.Content<Element>}
  BandCombineLeaf.model_contents(~Element,output_length,input_length,AccessLaws.combine_model(~Element,~operation,Access.zero_contiguous(source) && U32.is_zero(0),U32.to_nat(count),source,0,input_tree,output_tree))

# The right operand comes back unchanged.
def model_right(~Element: Data,+left_length: Nat,+right_length: Nat,+result: AccessLaws.ModelResult<Tree.Tree<Element>,Element>) ->
  {CopyValues.right_content(~Element,Operations.Accepted{OperationsLaws.combined_pair(~Element,left_length,right_length,result)}) ==
    Buffer.Content{CopyValues.reader_tree(~Element,result),right_length} : Buffer.Content<Element>}:
  AccessLaws.ModelResult{+reader,output} = result
  Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{tree,right_length},
    Tree.reflect(Element,Tree.pack(Element,reader)),reader,StorageLaws.reflect_packed(Element,reader))

def right_unchanged(~Element: Data,~operation: Element -> Element -> Element,+count: U32,+target_axes: List<&2,Views.Axis>,+source: Views.View,+output: Buffer.Model<Element>,+input: Buffer.Model<Element>,
  guarded: {guard(~Element,Views.View{target_axes,0},source,output,input) == True{} : Bool},counted: {Views.numel(Views.View{target_axes,0}) == Some{count} : Maybe<&2,U32>}) ->
  {CopyValues.right_content(~Element,OperationsLaws.combine_denotation(~Element,~operation,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source)) == Buffer.content(~Element,input) : Buffer.Content<Element>}:
  Buffer.Model{+output_tree,+output_length,Certificate.Certificate{+output_depth,+output_equation}} = output
  Buffer.Model{+input_tree,+input_length,Certificate.Certificate{+input_depth,+input_equation}} = input
  +target = {Views.View{target_axes,0} : Views.View}
  +model = {Buffer.Content{input_tree,input_length} : Buffer.Content<Element>}
  +result = AccessLaws.combine_model(~Element,~operation,Access.zero_contiguous(source) && U32.is_zero(0),U32.to_nat(count),source,0,input_tree,output_tree)
  %Equal.sym(Bool,(Shapes.same(Views.shape(target),Views.shape(source)) && Views.is_contiguous(target) && Operations.valid_buffer(output_length,U32.shln(1,output_depth),target) && Operations.valid_buffer(input_length,U32.shln(1,input_depth),source) : Bool),True{},guarded) :
    {CopyValues.right_content(~Element,OperationsLaws.combine_model(~Element,~operation,OperationsLaws.PairCall{_,Views.numel(target),source,target,input_length,output_length},
      Tree.reflect(Element,Tree.pack(Element,input_tree)),Tree.reflect(Element,Tree.pack(Element,output_tree)))) == model : Buffer.Content<Element>}
  %Equal.sym(Maybe<&2,U32>,Views.numel(target),Some{count},counted) :
    {CopyValues.right_content(~Element,OperationsLaws.combine_model(~Element,~operation,OperationsLaws.PairCall{True{},_,source,target,input_length,output_length},
      Tree.reflect(Element,Tree.pack(Element,input_tree)),Tree.reflect(Element,Tree.pack(Element,output_tree)))) == model : Buffer.Content<Element>}
  %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,input_tree)),input_tree,StorageLaws.reflect_packed(Element,input_tree)) :
    {CopyValues.right_content(~Element,OperationsLaws.combine_model(~Element,~operation,OperationsLaws.PairCall{True{},Some{count},source,target,input_length,output_length},
      _,Tree.reflect(Element,Tree.pack(Element,output_tree)))) == model : Buffer.Content<Element>}
  %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,output_tree)),output_tree,StorageLaws.reflect_packed(Element,output_tree)) :
    {CopyValues.right_content(~Element,OperationsLaws.combine_model(~Element,~operation,OperationsLaws.PairCall{True{},Some{count},source,target,input_length,output_length},
      input_tree,_)) == model : Buffer.Content<Element>}
  Equal.trans(Buffer.Content<Element>,CopyValues.right_content(~Element,Operations.Accepted{OperationsLaws.combined_pair(~Element,output_length,input_length,result)}),
    Buffer.Content{CopyValues.reader_tree(~Element,result),input_length},model,model_right(~Element,output_length,input_length,result),
    Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{tree,input_length},CopyValues.reader_tree(~Element,result),input_tree,
      CombineValues.combined_reader(~Element,~operation,Access.zero_contiguous(source) && U32.is_zero(0),U32.to_nat(count),source,input_tree,output_tree)))

# What an accepted combination leaves in its two owners: the left one holds
# the combined values, the right one is unchanged.
def Step(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,+left: Left,+right: Right,+source: Views.View,+left_shape: List<&2,U32>,-owners: Operations.Outcome<OperationsLaws.PairOwners(Element)>,-right_content: Buffer.Content<Element>) -> Data:
  Shape.Both<GatherValues.Facts(~Element,~Combine<Left,Right>,~value(~Element,~Left,~Right,~left_value,~right_value,~operation),Combine{left,right,source},left_shape,BandCombineLeaf.left_content(~Element,owners)),{CopyValues.right_content(~Element,owners) == right_content : Buffer.Content<Element>}>

def guarded_facts(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),+left: Left,+right: Right,+target_axes: List<&2,Views.Axis>,+source: Views.View,+output: Buffer.Model<Element>,+input: Buffer.Model<Element>,+left_shape: List<&2,U32>,
  +ready: Buffer.Sized<Element,output>,+input_ready: Buffer.Sized<Element,input>,+left_values: Contract.Values(~Element,~Left,~left_value,left,Buffer.content(~Element,output)),+right_values: Contract.Values(~Element,~Right,~right_value,right,Buffer.content(~Element,input)),
  +target_exact: {ShapeLaws.elements(Views.axis_extents(target_axes)) == Buffer.length(~Element,output) : Nat},
  +length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(left_shape) : Nat},
  facts: Guarded<Element,Views.View{target_axes,0},source,output,input>) ->
  Step(~Element,~Left,~Right,~left_value,~right_value,~operation,left,right,source,left_shape,OperationsLaws.observe_pair_outcome(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source)),Buffer.content(~Element,input)):
  Guarded{+count,+counted,+guard,+same,+fits} = facts
  +count_exact = {
    %counted : ShapeLaws.Count(_,ShapeLaws.elements(Views.axis_extents(target_axes)))
    ShapeLaws.numel(Views.axis_extents(target_axes)) : {ShapeLaws.elements(Views.axis_extents(target_axes)) == U32.to_nat(count) : Nat}}
  +written = Equal.trans(Nat,U32.to_nat(count),ShapeLaws.elements(Views.axis_extents(target_axes)),Buffer.length(~Element,output),
    Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(target_axes)),U32.to_nat(count),count_exact),target_exact)
  +exact = {
    %GatherValues.same_extents(Views.View{target_axes,0},source,same) : {ShapeLaws.elements(_) == U32.to_nat(count) : Nat}
    count_exact : {ShapeLaws.elements(Views.axis_extents_of(source)) == U32.to_nat(count) : Nat}}
  Shape.Both{
    %Equal.sym(Operations.Outcome<OperationsLaws.PairOwners(Element)>,OperationsLaws.observe_pair_outcome(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source)),OperationsLaws.combine_denotation(~Element,~operation,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source),OperationsLaws.combine(~Element,~operation,~kernel,~kernel_law,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source)) :
      GatherValues.Facts(~Element,~Combine<Left,Right>,~value(~Element,~Left,~Right,~left_value,~right_value,~operation),Combine{left,right,source},left_shape,BandCombineLeaf.left_content(~Element,_))
    %Equal.sym(Buffer.Content<Element>,BandCombineLeaf.left_content(~Element,OperationsLaws.combine_denotation(~Element,~operation,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source)),combined_content(~Element,~operation,count,source,Buffer.content(~Element,output),Buffer.content(~Element,input)),
      denoted_content(~Element,~operation,count,target_axes,source,output,input,guard,counted)) : GatherValues.Facts(~Element,~Combine<Left,Right>,~value(~Element,~Left,~Right,~left_value,~right_value,~operation),Combine{left,right,source},left_shape,_)
    program_facts(~Element,~Left,~Right,~left_value,~right_value,~operation,left,right,source,output,input,count,left_shape,ready,input_ready,left_values,right_values,written,exact,fits,length_exact),
    %Equal.sym(Operations.Outcome<OperationsLaws.PairOwners(Element)>,OperationsLaws.observe_pair_outcome(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source)),OperationsLaws.combine_denotation(~Element,~operation,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source),OperationsLaws.combine(~Element,~operation,~kernel,~kernel_law,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source)) :
      {CopyValues.right_content(~Element,_) == Buffer.content(~Element,input) : Buffer.Content<Element>}
    right_unchanged(~Element,~operation,count,target_axes,source,output,input,guard,counted)}

def modeled_facts(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),+left: Left,+right: Right,+target_axes: List<&2,Views.Axis>,+source: Views.View,+output: Buffer.Model<Element>,+input: Buffer.Model<Element>,+left_shape: List<&2,U32>,
  +ready: Buffer.Sized<Element,output>,+input_ready: Buffer.Sized<Element,input>,+left_values: Contract.Values(~Element,~Left,~left_value,left,Buffer.content(~Element,output)),+right_values: Contract.Values(~Element,~Right,~right_value,right,Buffer.content(~Element,input)),
  +target_exact: {ShapeLaws.elements(Views.axis_extents(target_axes)) == Buffer.length(~Element,output) : Nat},
  +length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(left_shape) : Nat},
  accepted: {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source))) == True{} : Bool}) ->
  Step(~Element,~Left,~Right,~left_value,~right_value,~operation,left,right,source,left_shape,OperationsLaws.observe_pair_outcome(~Element,Operations.combine_with(~Element,~kernel,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source)),Buffer.content(~Element,input)):
  guarded_facts(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,left,right,target_axes,source,output,input,left_shape,ready,input_ready,left_values,right_values,target_exact,length_exact,
    guarded(~Element,~operation,Views.View{target_axes,0},source,output,input,
      %OperationsLaws.combine(~Element,~operation,~kernel,~kernel_law,Buffer.pack(~Element,output),Views.View{target_axes,0},Buffer.pack(~Element,input),source) : {GatherValues.accepted_owners(~Element,_) == True{} : Bool}
      accepted))

