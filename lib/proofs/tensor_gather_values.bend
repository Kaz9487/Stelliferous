# The values of the copy every band of a gather runs: one checked copy
# through a source view, with the flagged axes in reverse, into fresh
# storage. When the call is accepted, the storage is well formed and holds,
# at every index, the logical input at the address the source view gives to
# the digits of that index with the digit of every flagged axis reflected.
# The accepted call is the only premise; its guards are read from the call.
import Base
import ../tensor.bend as Tensors
import ../tensor_shape.bend as Shapes
import ../tensor_view.bend as Views
import ../tensor_layout.bend as Layout
import ../tensor_operations.bend as Operations
import ../storage_buffer.bend as Storage
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_certificate.bend as Certificate
import ./storage_shape.bend as Shape
import ./storage_allocation_refinement.bend as AllocationLaws
import ./tensor_contract.bend as Contract
import ./tensor_copy_values.bend as CopyValues
import ./tensor_mirror_address.bend as MirrorAddress
import ./tensor_operations_refinement.bend as OperationsLaws
import ./tensor_operation_bounds.bend as OperationBounds
import ./tensor_view_metadata_refinement.bend as Metadata
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_count_refinement.bend as Count
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_allocation_depth.bend as Depth
import ./tensor_band_count.bend as BandCount
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./boolean_logic.bend as Boolean
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges
import ./write_program.bend as Writes

# The logical input read through a view.
type Gather<-Logical: Data> is Data:
  Gather{logical: Logical,source: Views.View}

def value(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,context: Gather<Logical>,index: U32) -> Element:
  Gather{logical,source} = context
  original(logical,CopyValues.address(source,U32.to_nat(index)))

# The logical input read through a view with the flagged axes in reverse: at
# every index, the address the view gives to the digits of that index with
# the digit of every flagged axis reflected.
type Directed<-Logical: Data> is Data:
  Directed{logical: Logical,source: Views.View,reversed: List<&2,Bool>}

def directed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,context: Directed<Logical>,index: U32) -> Element:
  Directed{logical,source,reversed} = context
  original(logical,MirrorAddress.address(source,reversed,U32.to_nat(index)))

# The directed read of a gather with no flag.
def unflagged(~Logical: Data,context: Gather<Logical>) -> Directed<Logical>:
  Gather{logical,source} = context
  Directed{logical,source,[]}

# With no flag the directed read is the read through the view.
def unflagged_value(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+context: Gather<Logical>,+index: U32) ->
  {directed(~Element,~Logical,~original,unflagged(~Logical,context),index) == value(~Element,~Logical,~original,context,index) : Element}:
  Gather{+logical,+source} = context
  Equal.cong(U32,Element,word => original(logical,word),MirrorAddress.address(source,[],U32.to_nat(index)),CopyValues.address(source,U32.to_nat(index)),
    MirrorAddress.forward(source,U32.to_nat(index)))

def Forward(~Logical: Data,before: Directed<Logical>,after: Gather<Logical>) -> Data:
  {before == unflagged(~Logical,after) : Directed<Logical>}

def forward(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+before: Directed<Logical>,+after: Gather<Logical>,+related: Forward(~Logical,before,after),+index: U32) ->
  {directed(~Element,~Logical,~original,before,index) == value(~Element,~Logical,~original,after,index) : Element}:
  %Equal.sym(Directed<Logical>,before,unflagged(~Logical,after),related) :
    {directed(~Element,~Logical,~original,_,index) == value(~Element,~Logical,~original,after,index) : Element}
  unflagged_value(~Element,~Logical,~original,after,index)

def mirrored_axis_extents(+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>) -> {Views.axis_extents(Views.mirrored_axes(axes,flags)) == Views.axis_extents(axes) : List<&2,U32>}:
  match axes:
    case []: {==}
    case Views.Axis{+extent,+stride} <> +tail:
      Equal.cong(List<&2,U32>,List<&2,U32>,rest => extent <> rest,Views.axis_extents(Views.mirrored_axes(tail,Views.later_flags(flags))),Views.axis_extents(tail),mirrored_axis_extents(tail,Views.later_flags(flags)))

def mirrored_extents(+source: Views.View,+flags: List<&2,Bool>) -> {Views.axis_extents_of(Views.mirrored(source,flags)) == Views.axis_extents_of(source) : List<&2,U32>}:
  Views.View{+axes,+offset} = source
  mirrored_axis_extents(axes,flags)

# One index of the copy program through the mirrored view into a target at
# offset zero.
def cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+source: Views.View,+reversed: List<&2,Bool>,+target_axes: List<&2,Views.Axis>,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+count: U32,+index: Nat,
  input_ready: Buffer.Sized<Element,input>,ready: Buffer.Sized<Element,output>,values: Contract.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),
  +written: {U32.to_nat(count) == Buffer.length(~Element,output) : Nat},
  +exact: {ShapeLaws.elements(Views.axis_extents_of(source)) == U32.to_nat(count) : Nat},
  fits: {Views.fits(source,U32.from_nat(Buffer.length(~Element,input))) == True{} : Bool},
  +inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  Contract.At(~Element,~Directed<Logical>,~directed(~Element,~Logical,~original),Directed{logical,source,reversed},CopyValues.program_content(~Element,count,Views.mirrored(source,reversed),Views.View{target_axes,0},Buffer.content(~Element,input),Buffer.content(~Element,output)),index):
  Buffer.Model{+input_tree,+input_length,+input_certificate} = input
  Buffer.Model{+output_tree,+output_length,+output_certificate} = output
  Buffer.Sized{+input_depth_ok,+input_room} = input_ready
  Buffer.Sized{+depth_ok,+room} = ready
  +input = {Buffer.Model{input_tree,input_length,input_certificate} : Buffer.Model<Element>}
  +output = {Buffer.Model{output_tree,output_length,output_certificate} : Buffer.Model<Element>}
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
  +mirrored = Views.mirrored(source,reversed)
  +address = MirrorAddress.address(source,reversed,index)
  +address_inside = {
    %Bounds.roundtrip(input_length,Tree.capacity(Element,input_tree),Buffer.capacity_bound(~Element,input,input_depth_ok),input_room) :
      {Nat.is_lt(U32.to_nat(address),_) == True{} : Bool}
    MirrorAddress.inside(source,reversed,index,U32.from_nat(input_length),
      %nonempty : {Views.fits_nonempty(_,source,U32.from_nat(input_length)) == True{} : Bool}
      fits) : {Nat.is_lt(U32.to_nat(address),input_length) == True{} : Bool}}
  Equal.trans(Element,
    Tree.read(Element,Writes.pure(Element,CopyValues.copy_writes(~Element,Views.is_contiguous(mirrored),U32.to_nat(count),mirrored,0,input_tree),output_tree),U32.from_nat(index)),
    Tree.read(Element,input_tree,address),
    original(logical,MirrorAddress.address(source,reversed,U32.to_nat(U32.from_nat(index)))),
    Equal.trans(Element,
      Tree.read(Element,Writes.pure(Element,CopyValues.copy_writes(~Element,Views.is_contiguous(mirrored),U32.to_nat(count),mirrored,0,input_tree),output_tree),U32.from_nat(index)),
      Tree.read(Element,input_tree,CopyValues.address(mirrored,index)),
      Tree.read(Element,input_tree,address),
      CopyValues.copied_value(~Element,mirrored,input_tree,output_tree,output_length,U32.to_nat(count),index,
        Buffer.content_ready(~Element,output,Buffer.Sized{depth_ok,room}),capacity_ok,count_room,inside,{
          %Equal.sym(List<&2,U32>,Views.axis_extents_of(mirrored),Views.axis_extents_of(source),mirrored_extents(source,reversed)) : {ShapeLaws.elements(_) == U32.to_nat(count) : Nat}
          exact : {ShapeLaws.elements(Views.axis_extents_of(mirrored)) == U32.to_nat(count) : Nat}}),
      Equal.cong(U32,Element,word => Tree.read(Element,input_tree,word),CopyValues.address(mirrored,index),address,MirrorAddress.mirrored(source,reversed,index))),
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,encoded) : {Tree.read(Element,input_tree,address) == original(logical,MirrorAddress.address(source,reversed,_)) : Element}
    %Arithmetic.native_roundtrip(address) : {Tree.read(Element,input_tree,_) == original(logical,_) : Element}
    Contract.read(~Element,~Logical,~original,logical,Buffer.Content{input_tree,input_length},U32.to_nat(address),values,address_inside))

def cells_range(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,remaining: Nat,+position: Nat,+logical: Logical,+source: Views.View,+reversed: List<&2,Bool>,+target_axes: List<&2,Views.Axis>,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+count: U32,
  +input_ready: Buffer.Sized<Element,input>,+ready: Buffer.Sized<Element,output>,+values: Contract.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),
  +written: {U32.to_nat(count) == Buffer.length(~Element,output) : Nat},
  +exact: {ShapeLaws.elements(Views.axis_extents_of(source)) == U32.to_nat(count) : Nat},
  +fits: {Views.fits(source,U32.from_nat(Buffer.length(~Element,input))) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(position,remaining),U32.to_nat(count)) == True{} : Bool}) ->
  Members.All(Nat,{index => Contract.At(~Element,~Directed<Logical>,~directed(~Element,~Logical,~original),Directed{logical,source,reversed},CopyValues.program_content(~Element,count,Views.mirrored(source,reversed),Views.View{target_axes,0},Buffer.content(~Element,input),Buffer.content(~Element,output)),index) : Nat -> Data},Ranges.range(remaining,position)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      Shape.Both{cell(~Element,~Logical,~original,logical,source,reversed,target_axes,input,output,count,position,input_ready,ready,values,written,exact,fits,Order.room_index(position,rest,U32.to_nat(count),room)),
        cells_range(~Element,~Logical,~original,rest,1n+position,logical,source,reversed,target_axes,input,output,count,input_ready,ready,values,written,exact,fits,Order.advance_room(position,rest,U32.to_nat(count),room))}

def program_length(~Element: Data,+count: U32,+source: Views.View,+target: Views.View,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>) ->
  {Buffer.content_length(~Element,CopyValues.program_content(~Element,count,source,target,Buffer.content(~Element,input),Buffer.content(~Element,output))) ==
    Buffer.length(~Element,output) : Nat}:
  Buffer.Model{input_tree,input_length,input_certificate} = input
  Buffer.Model{output_tree,output_length,output_certificate} = output
  {==}

def program_ready(~Element: Data,+count: U32,+source: Views.View,+target: Views.View,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,ready: Buffer.Sized<Element,output>) ->
  Buffer.ContentReady<Element,CopyValues.program_content(~Element,count,source,target,Buffer.content(~Element,input),Buffer.content(~Element,output))>:
  Buffer.Model{+input_tree,input_length,input_certificate} = input
  Buffer.Model{+output_tree,+output_length,+output_certificate} = output
  Writes.content_ready(~Element,CopyValues.copy_writes(~Element,Views.is_contiguous(source),U32.to_nat(count),source,CopyValues.view_offset(target),input_tree),output_tree,output_length,
    Buffer.content_ready(~Element,Buffer.Model{output_tree,output_length,output_certificate},ready))

def program_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+source: Views.View,+reversed: List<&2,Bool>,+target_axes: List<&2,Views.Axis>,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+count: U32,
  +input_ready: Buffer.Sized<Element,input>,+ready: Buffer.Sized<Element,output>,+values: Contract.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),
  +written: {U32.to_nat(count) == Buffer.length(~Element,output) : Nat},
  +exact: {ShapeLaws.elements(Views.axis_extents_of(source)) == U32.to_nat(count) : Nat},
  +fits: {Views.fits(source,U32.from_nat(Buffer.length(~Element,input))) == True{} : Bool}) ->
  Contract.Values(~Element,~Directed<Logical>,~directed(~Element,~Logical,~original),Directed{logical,source,reversed},CopyValues.program_content(~Element,count,Views.mirrored(source,reversed),Views.View{target_axes,0},Buffer.content(~Element,input),Buffer.content(~Element,output))):
  %Equal.sym(Nat,Buffer.content_length(~Element,CopyValues.program_content(~Element,count,Views.mirrored(source,reversed),Views.View{target_axes,0},Buffer.content(~Element,input),Buffer.content(~Element,output))),U32.to_nat(count),
    Equal.trans(Nat,Buffer.content_length(~Element,CopyValues.program_content(~Element,count,Views.mirrored(source,reversed),Views.View{target_axes,0},Buffer.content(~Element,input),Buffer.content(~Element,output))),Buffer.length(~Element,output),U32.to_nat(count),
      program_length(~Element,count,Views.mirrored(source,reversed),Views.View{target_axes,0},input,output),Equal.sym(Nat,U32.to_nat(count),Buffer.length(~Element,output),written))) :
    Members.All(Nat,{index => Contract.At(~Element,~Directed<Logical>,~directed(~Element,~Logical,~original),Directed{logical,source,reversed},CopyValues.program_content(~Element,count,Views.mirrored(source,reversed),Views.View{target_axes,0},Buffer.content(~Element,input),Buffer.content(~Element,output)),index) : Nat -> Data},Ranges.range(_,0n))
  cells_range(~Element,~Logical,~original,U32.to_nat(count),0n,logical,source,reversed,target_axes,input,output,count,input_ready,ready,values,written,exact,fits,Order.reflexive(U32.to_nat(count)))

# What a copied tensor needs from its storage content.
def Facts(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,+shape: List<&2,U32>,-content: Buffer.Content<Element>) -> Data:
  Shape.Both<Contract.WellFormed<Element,shape,content>,Contract.Values(~Element,~Context,~reader,context,content)>

def program_facts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+source: Views.View,+reversed: List<&2,Bool>,+target_axes: List<&2,Views.Axis>,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+target_shape: List<&2,U32>,+count: U32,
  +input_ready: Buffer.Sized<Element,input>,+ready: Buffer.Sized<Element,output>,+values: Contract.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),
  +written: {U32.to_nat(count) == Buffer.length(~Element,output) : Nat},
  +exact: {ShapeLaws.elements(Views.axis_extents_of(source)) == U32.to_nat(count) : Nat},
  +fits: {Views.fits(source,U32.from_nat(Buffer.length(~Element,input))) == True{} : Bool},
  length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(target_shape) : Nat}) ->
  Facts(~Element,~Directed<Logical>,~directed(~Element,~Logical,~original),Directed{logical,source,reversed},target_shape,CopyValues.program_content(~Element,count,Views.mirrored(source,reversed),Views.View{target_axes,0},Buffer.content(~Element,input),Buffer.content(~Element,output))):
  Shape.Both{
    Contract.WellFormed{
      Equal.trans(Nat,Buffer.content_length(~Element,CopyValues.program_content(~Element,count,Views.mirrored(source,reversed),Views.View{target_axes,0},Buffer.content(~Element,input),Buffer.content(~Element,output))),Buffer.length(~Element,output),ShapeLaws.elements(target_shape),
        program_length(~Element,count,Views.mirrored(source,reversed),Views.View{target_axes,0},input,output),length_exact),
      program_ready(~Element,count,Views.mirrored(source,reversed),Views.View{target_axes,0},input,output,ready)},
    program_values(~Element,~Logical,~original,logical,source,reversed,target_axes,input,output,count,input_ready,ready,values,written,exact,fits)}

# The observed outcome of the checked copy decides the status of the result.
def accepted_owners(~Element: Data,outcome: Operations.Outcome<OperationsLaws.PairOwners(Element)>) -> Bool:
  match outcome:
    case Operations.Accepted{value}: True{}
    case Operations.Rejected{value}: False{}

def ready_accepted(~Element: Data,outcome: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>,+name: String,+source_shape: List<&2,U32>,+target_shape: List<&2,U32>,+fill: Element,+workers: U32) ->
  {Contract.tensor_ready(~Element,Tensors.result_of(~Element,Tensors.copied(~Element,name,source_shape,target_shape,fill,workers,outcome))) == accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,outcome)) : Bool}:
  match outcome:
    case Operations.Accepted{(input,output)}: {==}
    case Operations.Rejected{(input,output)}: {==}

# An accepted copy model passed its guard.
def model_guard(~Element: Data,valid: Bool,count: Maybe<&2,U32>,+source: Views.View,+target: Views.View,+input: Tree.Tree<Element>,+input_length: Nat,
  +output: Tree.Tree<Element>,+output_length: Nat,
  accepted: {accepted_owners(~Element,OperationsLaws.copy_model(~Element,valid,count,source,target,input,input_length,output,output_length)) == True{} : Bool}) ->
  {valid == True{} : Bool}:
  match valid count:
    case True{} Some{count}: {==}
    case True{} None{}: {==}
    case False{} Some{count}: accepted
    case False{} None{}: accepted

type Guarded<-Element: Data,-source: Views.View,-target: Views.View,-input: Buffer.Model<Element>,-output: Buffer.Model<Element>> is Data:
  Guarded{guard: {OperationBounds.copy_guard(~Element,Buffer.pack(~Element,input),source,Buffer.pack(~Element,output),target) == True{} : Bool},
    same: {Shapes.same(Views.shape(source),Views.shape(target)) == True{} : Bool},
    fits: {Views.fits(source,U32.from_nat(Buffer.length(~Element,input))) == True{} : Bool}}

def guarded(~Element: Data,+source: Views.View,+reversed: List<&2,Bool>,+target: Views.View,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,
  accepted: {accepted_owners(~Element,OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),source,reversed,Buffer.pack(~Element,output),target)) == True{} : Bool}) ->
  Guarded<Element,source,target,input,output>:
  Buffer.Model{+input_tree,+input_length,Certificate.Certificate{+input_depth,+input_equation}} = input
  Buffer.Model{+output_tree,+output_length,Certificate.Certificate{+output_depth,+output_equation}} = output
  +same = Shapes.same(Views.shape(source),Views.shape(target))
  +contiguous = Views.is_contiguous(target)
  +input_valid = Operations.valid_buffer(input_length,U32.shln(1,input_depth),source)
  +output_valid = Operations.valid_buffer(output_length,U32.shln(1,output_depth),target)
  +truth = model_guard(~Element,(same && contiguous && input_valid && output_valid : Bool),Views.numel(target),Views.mirrored(source,reversed),target,
    Tree.reflect(Element,Tree.pack(Element,input_tree)),input_length,Tree.reflect(Element,Tree.pack(Element,output_tree)),output_length,accepted)
  +leading = Boolean.left((same && contiguous && input_valid : Bool),output_valid,truth)
  Guarded{truth,
    Boolean.left(same,contiguous,Boolean.left((same && contiguous : Bool),input_valid,leading)),
    Boolean.right(Nat.is_le(input_length,U32.to_nat(U32.shln(1,input_depth))),Views.fits(source,U32.from_nat(input_length)),
      Boolean.right((same && contiguous : Bool),input_valid,leading))}

def same_extents(+source: Views.View,+target: Views.View,same: {Shapes.same(Views.shape(source),Views.shape(target)) == True{} : Bool}) ->
  {Views.axis_extents_of(source) == Views.axis_extents_of(target) : List<&2,U32>}:
  Views.View{+source_axes,source_offset} = source
  Views.View{+target_axes,target_offset} = target
  ShapeLaws.same_extents(Views.axis_extents(source_axes),Views.axis_extents(target_axes),same)

def guarded_facts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+source: Views.View,+reversed: List<&2,Bool>,+target_axes: List<&2,Views.Axis>,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+target_shape: List<&2,U32>,+count: U32,
  +input_ready: Buffer.Sized<Element,input>,+ready: Buffer.Sized<Element,output>,+values: Contract.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),
  +written: {U32.to_nat(count) == Buffer.length(~Element,output) : Nat},
  +target_exact: {ShapeLaws.elements(Views.axis_extents(target_axes)) == U32.to_nat(count) : Nat},
  +length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(target_shape) : Nat},
  facts: Guarded<Element,source,Views.View{target_axes,0},input,output>) ->
  Facts(~Element,~Directed<Logical>,~directed(~Element,~Logical,~original),Directed{logical,source,reversed},target_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,reversed,Buffer.pack(~Element,output),Views.View{target_axes,0})))):
  Guarded{+guard,+same,+fits} = facts
  +exact = {
    %Equal.sym(List<&2,U32>,Views.axis_extents_of(source),Views.axis_extents(target_axes),same_extents(source,Views.View{target_axes,0},same)) : {ShapeLaws.elements(_) == U32.to_nat(count) : Nat}
    target_exact : {ShapeLaws.elements(Views.axis_extents_of(source)) == U32.to_nat(count) : Nat}}
  %Equal.sym(Operations.Outcome<OperationsLaws.PairOwners(Element)>,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,reversed,Buffer.pack(~Element,output),Views.View{target_axes,0})),
      OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),source,reversed,Buffer.pack(~Element,output),Views.View{target_axes,0}),
      OperationsLaws.copy(~Element,Buffer.pack(~Element,input),source,reversed,Buffer.pack(~Element,output),Views.View{target_axes,0})) :
    Facts(~Element,~Directed<Logical>,~directed(~Element,~Logical,~original),Directed{logical,source,reversed},target_shape,CopyValues.right_content(~Element,_))
  %Equal.sym(Buffer.Content<Element>,CopyValues.right_content(~Element,OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),source,reversed,Buffer.pack(~Element,output),Views.View{target_axes,0})),
    CopyValues.program_content(~Element,count,Views.mirrored(source,reversed),Views.View{target_axes,0},Buffer.contents(~Element,Buffer.pack(~Element,input)),Buffer.contents(~Element,Buffer.pack(~Element,output))),
    CopyValues.copy_observed_content(~Element,Buffer.pack(~Element,input),source,reversed,Buffer.pack(~Element,output),Views.View{target_axes,0},count,guard,CountCompletion.numel(Views.axis_extents(target_axes),count,target_exact))) :
    Facts(~Element,~Directed<Logical>,~directed(~Element,~Logical,~original),Directed{logical,source,reversed},target_shape,_)
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,input)),Buffer.content(~Element,input),Buffer.packed_contents(~Element,input)) :
    Facts(~Element,~Directed<Logical>,~directed(~Element,~Logical,~original),Directed{logical,source,reversed},target_shape,CopyValues.program_content(~Element,count,Views.mirrored(source,reversed),Views.View{target_axes,0},_,Buffer.contents(~Element,Buffer.pack(~Element,output))))
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,output)),Buffer.content(~Element,output),Buffer.packed_contents(~Element,output)) :
    Facts(~Element,~Directed<Logical>,~directed(~Element,~Logical,~original),Directed{logical,source,reversed},target_shape,CopyValues.program_content(~Element,count,Views.mirrored(source,reversed),Views.View{target_axes,0},Buffer.content(~Element,input),_))
  program_facts(~Element,~Logical,~original,logical,source,reversed,target_axes,input,output,target_shape,count,input_ready,ready,values,written,exact,fits,length_exact)

def modeled_facts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+source: Views.View,+reversed: List<&2,Bool>,+target_axes: List<&2,Views.Axis>,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+target_shape: List<&2,U32>,+count: U32,
  +input_ready: Buffer.Sized<Element,input>,+ready: Buffer.Sized<Element,output>,+values: Contract.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),
  +written: {U32.to_nat(count) == Buffer.length(~Element,output) : Nat},
  +target_exact: {ShapeLaws.elements(Views.axis_extents(target_axes)) == U32.to_nat(count) : Nat},
  +length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(target_shape) : Nat},
  accepted: {accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,reversed,Buffer.pack(~Element,output),Views.View{target_axes,0}))) == True{} : Bool}) ->
  Facts(~Element,~Directed<Logical>,~directed(~Element,~Logical,~original),Directed{logical,source,reversed},target_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,reversed,Buffer.pack(~Element,output),Views.View{target_axes,0})))):
  guarded_facts(~Element,~Logical,~original,logical,source,reversed,target_axes,input,output,target_shape,count,input_ready,ready,values,written,target_exact,length_exact,
    guarded(~Element,source,reversed,Views.View{target_axes,0},input,output,
      %OperationsLaws.copy(~Element,Buffer.pack(~Element,input),source,reversed,Buffer.pack(~Element,output),Views.View{target_axes,0}) : {accepted_owners(~Element,_) == True{} : Bool}
      accepted))

# The fresh output of the public call, observed as a model.
def allocated_facts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+source: Views.View,+reversed: List<&2,Bool>,+target_axes: List<&2,Views.Axis>,+input: Buffer.Model<Element>,+view_shape: List<&2,U32>,
  +target_shape: List<&2,U32>,+fill: Element,+count: U32,
  observation: Buffer.Observation<Element,Tensors.buffer(~Element,fill,count)>,evidence: Count.Count<ShapeLaws.elements(view_shape),count>,
  +input_ready: Buffer.Sized<Element,input>,+values: Contract.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),
  +target_extents: {Views.axis_extents(target_axes) == view_shape : List<&2,U32>},
  +same_elements: {ShapeLaws.elements(view_shape) == ShapeLaws.elements(target_shape) : Nat},
  accepted: {accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,reversed,Tensors.buffer(~Element,fill,count),Views.View{target_axes,0}))) == True{} : Bool}) ->
  Facts(~Element,~Directed<Logical>,~directed(~Element,~Logical,~original),Directed{logical,source,reversed},target_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,reversed,Tensors.buffer(~Element,fill,count),Views.View{target_axes,0})))):
  Buffer.Observation{+output,+equation} = observation
  Count.Count{+exact,+bounded} = evidence
  +fresh = {Buffer.Content{Tree.fresh(Element,Tensors.depth_for(count),fill),U32.to_nat(count)} : Buffer.Content<Element>}
  +allocated = Equal.trans(Buffer.Content<Element>,Buffer.content(~Element,output),Buffer.contents(~Element,Buffer.pack(~Element,output)),fresh,
    Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,output)),Buffer.content(~Element,output),Buffer.packed_contents(~Element,output)),
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,output),Tensors.buffer(~Element,fill,count),equation) : {Buffer.contents(~Element,_) == fresh : Buffer.Content<Element>}
    AllocationLaws.contents(~Element,Tensors.depth_for(count),fill,U32.to_nat(count)))
  +length_count = Equal.trans(Nat,Buffer.length(~Element,output),Buffer.content_length(~Element,Buffer.content(~Element,output)),U32.to_nat(count),
    Equal.sym(Nat,Buffer.content_length(~Element,Buffer.content(~Element,output)),Buffer.length(~Element,output),Buffer.content_length_of(~Element,output)),
    Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),Buffer.content(~Element,output),fresh,allocated))
  %equation : Facts(~Element,~Directed<Logical>,~directed(~Element,~Logical,~original),Directed{logical,source,reversed},target_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,reversed,_,Views.View{target_axes,0}))))
  modeled_facts(~Element,~Logical,~original,logical,source,reversed,target_axes,input,output,target_shape,count,input_ready,
    Buffer.ready_observed(~Element,Tensors.buffer(~Element,fill,count),output,equation,BandCount.allocated_ready(~Element,count,fill,Depth.bounded(count,bounded))),
    values,Equal.sym(Nat,Buffer.length(~Element,output),U32.to_nat(count),length_count),
    %Equal.sym(List<&2,U32>,Views.axis_extents(target_axes),view_shape,target_extents) : {ShapeLaws.elements(_) == U32.to_nat(count) : Nat}
    exact,
    Equal.trans(Nat,Buffer.length(~Element,output),U32.to_nat(count),ShapeLaws.elements(target_shape),length_count,
      Equal.trans(Nat,U32.to_nat(count),ShapeLaws.elements(view_shape),ShapeLaws.elements(target_shape),
        Equal.sym(Nat,ShapeLaws.elements(view_shape),U32.to_nat(count),exact),same_elements)),
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,output),Tensors.buffer(~Element,fill,count),equation) : {accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,reversed,_,Views.View{target_axes,0}))) == True{} : Bool}
    accepted)

# The canonical view of the fresh storage starts at offset zero and has the
# extents it was built from.
def targeted_facts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,target: Views.View,+logical: Logical,+source: Views.View,+reversed: List<&2,Bool>,+input: Buffer.Model<Element>,+view_shape: List<&2,U32>,
  +target_shape: List<&2,U32>,+fill: Element,+count: U32,
  +input_ready: Buffer.Sized<Element,input>,+values: Contract.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),
  +targets: {Tensors.view_of(view_shape) == Some{target} : Maybe<&2,Views.View>},
  +counts: {Tensors.count_of(view_shape) == Some{count} : Maybe<&2,U32>},
  same_elements: @+count: U32 -> {Tensors.count_of(view_shape) == Some{count} : Maybe<&2,U32>} -> {ShapeLaws.elements(view_shape) == ShapeLaws.elements(target_shape) : Nat},
  accepted: {accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,reversed,Tensors.buffer(~Element,fill,count),target))) == True{} : Bool}) ->
  Facts(~Element,~Directed<Logical>,~directed(~Element,~Logical,~original),Directed{logical,source,reversed},target_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,reversed,Tensors.buffer(~Element,fill,count),target)))):
  Views.View{+target_axes,+target_offset} = target
  +offset_zero = {
    %targets : {Metadata.observed_offset(_,0) == 0 : U32}
    Metadata.contiguous_count_offset(Shapes.numel(Shapes.Shape{view_shape}),view_shape,0) : {target_offset == 0 : U32}}
  +target_extents = {
    %targets : {Metadata.observed_extents(_,view_shape) == view_shape : List<&2,U32>}
    Metadata.contiguous_shape(view_shape,0) : {Views.axis_extents(target_axes) == view_shape : List<&2,U32>}}
  %Equal.sym(U32,target_offset,0,offset_zero) : Facts(~Element,~Directed<Logical>,~directed(~Element,~Logical,~original),Directed{logical,source,reversed},target_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,reversed,Tensors.buffer(~Element,fill,count),Views.View{target_axes,_}))))
  allocated_facts(~Element,~Logical,~original,logical,source,reversed,target_axes,input,view_shape,target_shape,fill,count,
    Buffer.observe(~Element,Tensors.buffer(~Element,fill,count)),Count.count_of(view_shape,count,counts),input_ready,values,target_extents,same_elements(count,counts),
    %offset_zero : {accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,reversed,Tensors.buffer(~Element,fill,count),Views.View{target_axes,_}))) == True{} : Bool}
    accepted)
