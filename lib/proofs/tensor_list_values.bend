# from_list: the list written into one buffer is cut into the pieces of the
# grid of its shape, which hold the values of any reader that agrees with the
# list inside the shape.
import Base
import ../tensor.bend as Tensors
import ../storage_buffer.bend as Storage
import ../traversal_array.bend as Traversal
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./storage_tree_model.bend as Tree
import ./storage_buffer_model.bend as Buffer
import ./storage_refinement.bend as Native
import ./storage_certificate.bend as Certificate
import ./storage_witness.bend as Witness
import ./storage_shape.bend as Shape
import ./tensor_contract.bend as Contract
import ./tensor_whole_values.bend as Whole
import ./tensor_global_shape_count.bend as Global
import ./tensor_allocation_depth.bend as Depth
import ./tensor_count_refinement.bend as Count
import ./tensor_shape_refinement.bend as ShapeLaws
import ./list_storage_readback.bend as ListRead
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_full_range.bend as FullRange
import ./u32_comparison.bend as Comparison
import ./sequence_membership.bend as Members
import ./nat_order.bend as Order
import ./word_encoding.bend as Words
import ./tensor_move_readback.bend as Held
import ./storage_allocation_refinement.bend as AllocationLaws
import ../storage_allocation.bend as Allocation
import ../kernel_gemm.bend as KernelGemm

# A list written one value at a time holds what write_list writes.
def written_contents(~Element: Data,values: List<&2,Element>,+offset: U32,-storage: Array<Element>,+length: Nat,-certificate: Certificate.Certificate<Element,storage>) ->
  {Buffer.contents(~Element,Storage.write_at(~Element,values,offset,Storage.Buffer{storage,length,certificate})) ==
    Buffer.Content{Tree.reflect(Element,Traversal.write_list(~Element,values,offset,storage)),length} : Buffer.Content<Element>}:
  match values:
    case []: {==}
    case head <> tail:
      written_contents(~Element,tail,(offset + 1 : U32),Array.set(Element,storage,offset,head),length,Certificate.set(~Element,offset,head,storage,certificate))

# Each write keeps the depth of the certificate, so the buffer is ready when
# what write_list writes fits its length.
def written_native(~Element: Data,values: List<&2,Element>,+offset: U32,-storage: Array<Element>,+length: Nat,+depth: Nat,
  -equation: {storage == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,storage))) : Array<Element>},
  ready: KernelGemm.Both<{Nat.is_le(depth,30n) == True{} : Bool},
    {Nat.is_le(length,U32.to_nat(Tree.capacity(Element,Tree.reflect(Element,Traversal.write_list(~Element,values,offset,storage))))) == True{} : Bool}>) ->
  Buffer.NativeReady(~Element,Storage.write_at(~Element,values,offset,Storage.Buffer{storage,length,Certificate.Certificate{depth,equation}})):
  match values:
    case []: ready
    case head <> tail:
      written_native(~Element,tail,(offset + 1 : U32),Array.set(Element,storage,offset,head),length,depth,
        Certificate.set_equation(~Element,depth,offset,head,storage,equation),ready)

# What write_list writes into a fresh array is the sequence written into the
# fresh tree.
def listed_content(~Element: Data,+values: List<&2,Element>,+depth: Nat,+fill: Element,+count: Nat) ->
  {Buffer.Content{Tree.reflect(Element,Traversal.write_list(~Element,values,0,Array.new(Element,depth,fill))),count} ==
    Buffer.Content{Tree.write_sequence(~Element,values,0,Tree.fresh(Element,depth,fill)),count} : Buffer.Content<Element>}:
  %Equal.sym(Array<Element>,Array.new(Element,depth,fill),Tree.pack(Element,Tree.fresh(Element,depth,fill)),Native.fresh(Element,depth,fill)) :
    {Buffer.Content{Tree.reflect(Element,Storage.write_list(~Element,values,0,_)),count} == Buffer.Content{Tree.write_sequence(~Element,values,0,Tree.fresh(Element,depth,fill)),count} : Buffer.Content<Element>}
  %Equal.sym(Array<Element>,Storage.write_list(~Element,values,0,Tree.pack(Element,Tree.fresh(Element,depth,fill))),Tree.pack(Element,Tree.write_sequence(~Element,values,0,Tree.fresh(Element,depth,fill))),Native.write_sequence(~Element,values,0,Tree.fresh(Element,depth,fill))) :
    {Buffer.Content{Tree.reflect(Element,_),count} == Buffer.Content{Tree.write_sequence(~Element,values,0,Tree.fresh(Element,depth,fill)),count} : Buffer.Content<Element>}
  Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{tree,count},Tree.reflect(Element,Tree.pack(Element,Tree.write_sequence(~Element,values,0,Tree.fresh(Element,depth,fill)))),Tree.write_sequence(~Element,values,0,Tree.fresh(Element,depth,fill)),Native.reflect_packed(Element,Tree.write_sequence(~Element,values,0,Tree.fresh(Element,depth,fill))))

def contents(~Element: Data,+values: List<&2,Element>,+depth: Nat,+fill: Element,+count: Nat) -> {Buffer.contents(~Element,Storage.from_list(~Element,values,depth,fill,count)) == Buffer.Content{Tree.write_sequence(~Element,values,0,Tree.fresh(Element,depth,fill)),count} : Buffer.Content<Element>}:
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Storage.from_list(~Element,values,depth,fill,count)),
    Buffer.Content{Tree.reflect(Element,Traversal.write_list(~Element,values,0,Array.new(Element,depth,fill))),count},
    written_contents(~Element,values,0,Array.new(Element,depth,fill),count,Certificate.allocated(Element,depth,fill))) :
    {_ == Buffer.Content{Tree.write_sequence(~Element,values,0,Tree.fresh(Element,depth,fill)),count} : Buffer.Content<Element>}
  listed_content(~Element,values,depth,fill,count)

# A fresh buffer holds its fill.
def allocated(~Element: Data,+fill: Element,+count: U32) ->
  {Buffer.contents(~Element,Tensors.buffer(~Element,fill,count)) == Buffer.Content{Tree.fresh(Element,Tensors.depth_for(count),fill),U32.to_nat(count)} : Buffer.Content<Element>}:
  %Equal.sym(Storage.Buffer<Element>,Allocation.allocate(~Element,Tensors.depth_for(count),fill,U32.to_nat(count)),Storage.allocate(~Element,Tensors.depth_for(count),fill,U32.to_nat(count)),
    AllocationLaws.dispatch_same(Element,Allocation.allocated(~Element),Tensors.depth_for(count),fill,U32.to_nat(count))) :
    {Buffer.contents(~Element,_) == Buffer.Content{Tree.fresh(Element,Tensors.depth_for(count),fill),U32.to_nat(count)} : Buffer.Content<Element>}
  %Equal.sym(Array<Element>,Array.new(Element,Tensors.depth_for(count),fill),Tree.pack(Element,Tree.fresh(Element,Tensors.depth_for(count),fill)),Native.fresh(Element,Tensors.depth_for(count),fill)) :
    {Buffer.Content{Tree.reflect(Element,_),U32.to_nat(count)} == Buffer.Content{Tree.fresh(Element,Tensors.depth_for(count),fill),U32.to_nat(count)} : Buffer.Content<Element>}
  Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{tree,U32.to_nat(count)},Tree.reflect(Element,Tree.pack(Element,Tree.fresh(Element,Tensors.depth_for(count),fill))),
    Tree.fresh(Element,Tensors.depth_for(count),fill),Native.reflect_packed(Element,Tree.fresh(Element,Tensors.depth_for(count),fill)))

def written(~Element: Data,+values: List<&2,Element>,+count: U32,+fill: Element,+index: Nat,+equal: {List.length(&2,Element,values) == U32.to_nat(count) : Nat},within: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool},allocation: Depth.Sized<count>) ->
  {Buffer.value(~Element,U32.from_nat(index),Buffer.contents(~Element,Storage.from_list(~Element,values,Tensors.depth_for(count),fill,U32.to_nat(count)))) == ListRead.at(~Element,fill,index,values) : Element}:
  Depth.Sized{depth_ok,room} = allocation
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Storage.from_list(~Element,values,Tensors.depth_for(count),fill,U32.to_nat(count))),Buffer.Content{Tree.write_sequence(~Element,values,0,Tree.fresh(Element,Tensors.depth_for(count),fill)),U32.to_nat(count)},contents(~Element,values,Tensors.depth_for(count),fill,U32.to_nat(count))) :
    {Buffer.value(~Element,U32.from_nat(index),_) == ListRead.at(~Element,fill,index,values) : Element}
  ListRead.listed(~Element,values,index,0n,Tensors.depth_for(count),Tree.fresh(Element,Tensors.depth_for(count),fill),fill,Shape.fresh(Element,Tensors.depth_for(count),fill),depth_ok,
    %Equal.sym(U32,Tree.capacity(Element,Tree.fresh(Element,Tensors.depth_for(count),fill)),U32.shln(1,Tensors.depth_for(count)),Shape.capacity(Element,Tensors.depth_for(count),Tree.fresh(Element,Tensors.depth_for(count),fill),Shape.fresh(Element,Tensors.depth_for(count),fill))) : {Nat.is_le(List.length(&2,Element,values),U32.to_nat(_)) == True{} : Bool}
    %Equal.sym(Nat,List.length(&2,Element,values),U32.to_nat(count),equal) : {Nat.is_le(_,U32.to_nat(U32.shln(1,Tensors.depth_for(count)))) == True{} : Bool} room,
    %Equal.sym(Nat,List.length(&2,Element,values),U32.to_nat(count),equal) : {Nat.is_lt(index,_) == True{} : Bool} within)

def written_ready(~Element: Data,+values: List<&2,Element>,+count: U32,+fill: Element,allocation: Depth.Sized<count>) ->
  Buffer.NativeReady(~Element,Storage.from_list(~Element,values,Tensors.depth_for(count),fill,U32.to_nat(count))):
  Depth.Sized{depth_ok,room} = allocation
  written_native(~Element,values,0,Array.new(Element,Tensors.depth_for(count),fill),U32.to_nat(count),Tensors.depth_for(count),
    Certificate.fresh_equation(Element,Tensors.depth_for(count),fill),
    KernelGemm.Both{depth_ok,
      %Equal.sym(U32,Tree.capacity(Element,Tree.reflect(Element,Traversal.write_list(~Element,values,0,Array.new(Element,Tensors.depth_for(count),fill)))),U32.shln(1,Tensors.depth_for(count)),
        Shape.capacity(Element,Tensors.depth_for(count),Tree.reflect(Element,Traversal.write_list(~Element,values,0,Array.new(Element,Tensors.depth_for(count),fill))),
          Witness.balanced(Element,Tensors.depth_for(count),Traversal.write_list(~Element,values,0,Array.new(Element,Tensors.depth_for(count),fill)),
            Certificate.written(~Element,Tensors.depth_for(count),values,0,Array.new(Element,Tensors.depth_for(count),fill),Witness.fresh(Element,Tensors.depth_for(count),fill))))) :
        {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool}
      room})

# A list written into a fresh tree for a count within 2^30 is ready.
def sequence_ready(~Element: Data,+values: List<&2,Element>,+count: U32,+fill: Element,+within: {U32.is_le(count,1073741824) == True{} : Bool}) ->
  Buffer.ContentReady<Element,Buffer.Content{Tree.write_sequence(~Element,values,0,Tree.fresh(Element,Tensors.depth_for(count),fill)),U32.to_nat(count)}>:
  %contents(~Element,values,Tensors.depth_for(count),fill,U32.to_nat(count)) : Buffer.ContentReady<Element,_>
  Buffer.native_content_ready(~Element,Storage.from_list(~Element,values,Tensors.depth_for(count),fill,U32.to_nat(count)),
    written_ready(~Element,values,count,fill,Depth.bounded(count,within)))

# A list written from cell 0 into ready contents that hold as many cells
# reads back its values.
def sequence_value(~Element: Data,+values: List<&2,Element>,+tree: Tree.Tree<Element>,+length: Nat,+fill: Element,
  +ready: Buffer.ContentReady<Element,Buffer.Content{tree,length}>,+fits: {Nat.is_le(List.length(&2,Element,values),length) == True{} : Bool},+index: Nat,
  +within: {Nat.is_lt(index,List.length(&2,Element,values)) == True{} : Bool}) ->
  {Tree.read(Element,Tree.write_sequence(~Element,values,0,tree),U32.from_nat(index)) == ListRead.at(~Element,fill,index,values) : Element}:
  Buffer.ContentReady{+depth,+balanced,+depth_ok,+room} = ready
  ListRead.listed(~Element,values,index,0n,depth,tree,fill,balanced,depth_ok,
    Order.transitive(List.length(&2,Element,values),length,U32.to_nat(Tree.capacity(Element,tree)),fits,room),within)

# A list of the counted length written whole into a fresh tree holds the
# values of a reader that agrees with the list at every index of the shape.
def listed_facts(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,+shape: List<&2,U32>,+values: List<&2,Element>,+fill: Element,+count: U32,
  +counted: Count.Count<ShapeLaws.elements(shape),count>,+equal: {List.length(&2,Element,values) == U32.to_nat(count) : Nat},
  agrees: @+count: U32 -> @+counted: Count.Count<ShapeLaws.elements(shape),count> -> @+equal: {List.length(&2,Element,values) == U32.to_nat(count) : Nat} -> @+index: Nat -> @+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool} ->
    {ListRead.at(~Element,fill,index,values) == reader(context,U32.from_nat(index)) : Element}) ->
  Contract.Facts(~Element,~Context,~reader,context,shape,Buffer.Content{Tree.write_sequence(~Element,values,0,Tree.fresh(Element,Tensors.depth_for(count),fill)),U32.to_nat(count)}):
  Count.Count{+exact,+bounded} = counted
  %listed_content(~Element,values,Tensors.depth_for(count),fill,U32.to_nat(count)) : Contract.Facts(~Element,~Context,~reader,context,shape,_)
  KernelGemm.Both{
    Contract.WellFormed{Equal.sym(Nat,ShapeLaws.elements(shape),U32.to_nat(count),exact),
      %written_contents(~Element,values,0,Array.new(Element,Tensors.depth_for(count),fill),U32.to_nat(count),Certificate.allocated(Element,Tensors.depth_for(count),fill)) : Buffer.ContentReady<Element,_>
      Buffer.native_content_ready(~Element,Storage.from_list(~Element,values,Tensors.depth_for(count),fill,U32.to_nat(count)),
        written_ready(~Element,values,count,fill,Depth.bounded(count,bounded)))},
    Members.range_all(~Unit,
      ~(scope => index => Contract.At(~Element,~Context,~reader,context,Buffer.Content{Tree.reflect(Element,Traversal.write_list(~Element,values,0,Array.new(Element,Tensors.depth_for(count),fill))),U32.to_nat(count)},index)),
      ~(scope => U32.to_nat(count)),
      ~(scope => index => inside => Equal.trans(Element,
        Buffer.value(~Element,U32.from_nat(index),Buffer.Content{Tree.reflect(Element,Traversal.write_list(~Element,values,0,Array.new(Element,Tensors.depth_for(count),fill))),U32.to_nat(count)}),
        ListRead.at(~Element,fill,index,values),reader(context,U32.from_nat(index)),
        %written_contents(~Element,values,0,Array.new(Element,Tensors.depth_for(count),fill),U32.to_nat(count),Certificate.allocated(Element,Tensors.depth_for(count),fill)) : {Buffer.value(~Element,U32.from_nat(index),_) == ListRead.at(~Element,fill,index,values) : Element}
        written(~Element,values,count,fill,index,equal,inside,Depth.bounded(count,bounded)),agrees(count,Count.Count{exact,bounded},equal,index,inside))),
      U32.to_nat(count),0n,Unit{},Order.reflexive(U32.to_nat(count)))}

def count_bounded(+expected: Nat,+count: U32,+evidence: Count.Count<expected,count>) -> {U32.is_le(count,Machine.capacity(Default.limits())) == True{} : Bool}:
  Count.Count{+exact,+bounded} = evidence
  bounded

# A buffer that holds such a list, whole, is a tensor of the shape with the
# reader's values.
def stored_listed(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,+limits: Machine.Limits,-storage: Storage.Buffer<Element>,+shape: List<&2,U32>,
  +values: List<&2,Element>,+fill: Element,+count: U32,+workers: U32,observation: Buffer.Observation<Element,storage>,
  +stored: {Buffer.contents(~Element,storage) == Buffer.Content{Tree.write_sequence(~Element,values,0,Tree.fresh(Element,Tensors.depth_for(count),fill)),U32.to_nat(count)} : Buffer.Content<Element>},
  +accepted: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},+bounded: {U32.is_le(count,Machine.capacity(limits)) == True{} : Bool},
  +equal: {List.length(&2,Element,values) == U32.to_nat(count) : Nat},
  agrees: @+count: U32 -> @+counted: Count.Count<ShapeLaws.elements(shape),count> -> @+equal: {List.length(&2,Element,values) == U32.to_nat(count) : Nat} -> @+index: Nat -> @+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool} ->
    {ListRead.at(~Element,fill,index,values) == reader(context,U32.from_nat(index)) : Element},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool}) ->
  Global.Contract(~Element,~Context,~reader,context,Tensors.whole(~Element,storage,shape,fill,Tensors.Usable{},workers)):
  Buffer.Observation{+model,+equation} = observation
  +counted = Count.count_of(shape,count,accepted)
  +held = {%Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,model),storage,equation) : {Buffer.contents(~Element,_) == Buffer.Content{Tree.write_sequence(~Element,values,0,Tree.fresh(Element,Tensors.depth_for(count),fill)),U32.to_nat(count)} : Buffer.Content<Element>}
    stored : {Buffer.contents(~Element,Buffer.pack(~Element,model)) == Buffer.Content{Tree.write_sequence(~Element,values,0,Tree.fresh(Element,Tensors.depth_for(count),fill)),U32.to_nat(count)} : Buffer.Content<Element>}}
  +room = Held.nat_le_moved(ShapeLaws.elements(shape),U32.to_nat(count),U32.to_nat(Machine.capacity(limits)),Count.count_exact(ShapeLaws.elements(shape),count,counted),
    %Comparison.le(count,Machine.capacity(limits)) : {_ == True{} : Bool} bounded)
  %equation : Global.Contract(~Element,~Context,~reader,context,Tensors.whole(~Element,_,shape,fill,Tensors.Usable{},workers))
  Whole.whole_contract(~Element,~Context,~reader,context,limits,shape,fill,workers,model,Global.Counted{count,accepted},room,nonempty,
    %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,model)),Buffer.Content{Tree.write_sequence(~Element,values,0,Tree.fresh(Element,Tensors.depth_for(count),fill)),U32.to_nat(count)},held) : Contract.Facts(~Element,~Context,~reader,context,shape,_)
    listed_facts(~Element,~Context,~reader,context,shape,values,fill,count,counted,equal,agrees))

# A list of exactly expected values is written whole.
def exactly_matched(~Element: Data,values: List<&2,Element>,+offset: U32,expected: Nat,-buffer: Storage.Buffer<Element>,
  +matches: {Nat.is_eq(List.length(&2,Element,values),expected) == True{} : Bool}) ->
  {Storage.write_exactly(~Element,values,offset,expected,buffer) == (Storage.write_at(~Element,values,offset,buffer),True{}) : Storage.Buffer<Element> & Bool}:
  match values expected:
    case [] 0n: {==}
    case [] 1n+rest:
      Empty.absurd({Storage.write_exactly(~Element,[],offset,1n+rest,buffer) == (Storage.write_at(~Element,[],offset,buffer),True{}) : Storage.Buffer<Element> & Bool},Words.false_true(matches))
    case head <> tail 0n:
      Empty.absurd({Storage.write_exactly(~Element,head <> tail,offset,0n,buffer) == (Storage.write_at(~Element,head <> tail,offset,buffer),True{}) : Storage.Buffer<Element> & Bool},
        Words.false_true(matches))
    case head <> tail 1n+rest: exactly_matched(~Element,tail,(offset + 1 : U32),rest,Storage.write(~Element,offset,head,buffer),matches)

# A list of another count comes back with False.
def exactly_failed(~Element: Data,values: List<&2,Element>,+offset: U32,expected: Nat,-buffer: Storage.Buffer<Element>,
  +differs: {Nat.is_eq(List.length(&2,Element,values),expected) == False{} : Bool}) ->
  {Storage.write_exactly(~Element,values,offset,expected,buffer) ==
    (Pair.fst(Storage.Buffer<Element>,Bool,Storage.write_exactly(~Element,values,offset,expected,buffer)),False{}) : Storage.Buffer<Element> & Bool}:
  match values expected:
    case [] 0n:
      Empty.absurd({Storage.write_exactly(~Element,[],offset,0n,buffer) == (Pair.fst(Storage.Buffer<Element>,Bool,Storage.write_exactly(~Element,[],offset,0n,buffer)),False{}) : Storage.Buffer<Element> & Bool},
        Words.false_true(Equal.sym(Bool,True{},False{},differs)))
    case [] 1n+rest: {==}
    case head <> tail 0n: {==}
    case head <> tail 1n+rest: exactly_failed(~Element,tail,(offset + 1 : U32),rest,Storage.write(~Element,offset,head,buffer),differs)

def listed_matches(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,matches: Bool,+context: Context,+shape: List<&2,U32>,+values: List<&2,Element>,+fill: Element,+count: U32,
  +compared: {Nat.is_eq(List.length(&2,Element,values),U32.to_nat(count)) == matches : Bool},+accepted: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},
  agrees: @+count: U32 -> @+counted: Count.Count<ShapeLaws.elements(shape),count> -> @+equal: {List.length(&2,Element,values) == U32.to_nat(count) : Nat} -> @+index: Nat -> @+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool} ->
    {ListRead.at(~Element,fill,index,values) == reader(context,U32.from_nat(index)) : Element},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},
  ready: {Contract.tensor_ready(~Element,Tensors.listed_written(~Element,shape,Storage.write_exactly(~Element,values,0,U32.to_nat(count),Storage.allocate(~Element,Tensors.depth_for(count),fill,U32.to_nat(count))),fill)) == True{} : Bool}) ->
  Global.Contract(~Element,~Context,~reader,context,Tensors.listed_written(~Element,shape,Storage.write_exactly(~Element,values,0,U32.to_nat(count),Storage.allocate(~Element,Tensors.depth_for(count),fill,U32.to_nat(count))),fill)):
  match matches:
    case False{}:
      Empty.absurd(Global.Contract(~Element,~Context,~reader,context,Tensors.listed_written(~Element,shape,Storage.write_exactly(~Element,values,0,U32.to_nat(count),Storage.allocate(~Element,Tensors.depth_for(count),fill,U32.to_nat(count))),fill)),
        Words.false_true(%exactly_failed(~Element,values,0,U32.to_nat(count),Storage.allocate(~Element,Tensors.depth_for(count),fill,U32.to_nat(count)),compared) :
          {Contract.tensor_ready(~Element,Tensors.listed_written(~Element,shape,_,fill)) == True{} : Bool}
          ready))
    case True{}:
      %Equal.sym(Storage.Buffer<Element> & Bool,Storage.write_exactly(~Element,values,0,U32.to_nat(count),Storage.allocate(~Element,Tensors.depth_for(count),fill,U32.to_nat(count))),(Storage.write_at(~Element,values,0,Storage.allocate(~Element,Tensors.depth_for(count),fill,U32.to_nat(count))),True{}),
        exactly_matched(~Element,values,0,U32.to_nat(count),Storage.allocate(~Element,Tensors.depth_for(count),fill,U32.to_nat(count)),compared)) :
        Global.Contract(~Element,~Context,~reader,context,Tensors.listed_written(~Element,shape,_,fill))
      stored_listed(~Element,~Context,~reader,context,Default.limits(),Storage.from_list(~Element,values,Tensors.depth_for(count),fill,U32.to_nat(count)),shape,values,fill,count,1,
        Buffer.observe(~Element,Storage.from_list(~Element,values,Tensors.depth_for(count),fill,U32.to_nat(count))),
        contents(~Element,values,Tensors.depth_for(count),fill,U32.to_nat(count)),accepted,
        count_bounded(ShapeLaws.elements(shape),count,Count.count_of(shape,count,accepted)),
        Order.equal_of_is_eq(List.length(&2,Element,values),U32.to_nat(count),compared),agrees,nonempty)

def selected(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,result: Maybe<&2,U32>,+context: Context,+shape: List<&2,U32>,+values: List<&2,Element>,+fill: Element,
  +counts: {Tensors.count_of(shape) == result : Maybe<&2,U32>},
  agrees: @+count: U32 -> @+counted: Count.Count<ShapeLaws.elements(shape),count> -> @+equal: {List.length(&2,Element,values) == U32.to_nat(count) : Nat} -> @+index: Nat -> @+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool} ->
    {ListRead.at(~Element,fill,index,values) == reader(context,U32.from_nat(index)) : Element},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},
  ready: {Contract.tensor_ready(~Element,Tensors.listed(~Element,shape,result,values,fill)) == True{} : Bool}) ->
  Global.Contract(~Element,~Context,~reader,context,Tensors.listed(~Element,shape,result,values,fill)):
  match result:
    case None{}:
      Empty.absurd(Global.Contract(~Element,~Context,~reader,context,Tensors.listed(~Element,shape,None{},values,fill)),Words.false_true(ready))
    case Some{+count}:
      listed_matches(~Element,~Context,~reader,Nat.is_eq(List.length(&2,Element,values),U32.to_nat(count)),context,shape,values,fill,count,{==},counts,agrees,nonempty,ready)

# A Ready from_list tensor of a nonempty shape holds the values of a reader
# that agrees with the list at every index of the shape.
def from_list_read(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,+shape: List<&2,U32>,+values: List<&2,Element>,+fill: Element,
  agrees: @+count: U32 -> @+counted: Count.Count<ShapeLaws.elements(shape),count> -> @+equal: {List.length(&2,Element,values) == U32.to_nat(count) : Nat} -> @+index: Nat -> @+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool} ->
    {ListRead.at(~Element,fill,index,values) == reader(context,U32.from_nat(index)) : Element},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},
  ready: {Contract.tensor_ready(~Element,Tensors.from_list(~Element,shape,values,fill)) == True{} : Bool}) ->
  Global.Contract(~Element,~Context,~reader,context,Tensors.from_list(~Element,shape,values,fill)):
  selected(~Element,~Context,~reader,Tensors.count_of(shape),context,shape,values,fill,{==},agrees,nonempty,ready)

# from_list keeps the shape it is given, Ready or not.
def listed_count_extents(~Element: Data,matches: Bool,+shape: List<&2,U32>,-storage: Storage.Buffer<Element>,+fill: Element) ->
  {Contract.extents(~Element,Tensors.listed_count(~Element,matches,shape,storage,fill)) == shape : List<&2,U32>}:
  match matches:
    case True{}: {==}
    case False{}: {==}

def listed_written_extents(~Element: Data,+shape: List<&2,U32>,written: Storage.Buffer<Element> & Bool,+fill: Element) ->
  {Contract.extents(~Element,Tensors.listed_written(~Element,shape,written,fill)) == shape : List<&2,U32>}:
  (storage,matches) = written
  listed_count_extents(~Element,matches,shape,storage,fill)

def listed_extents(~Element: Data,+shape: List<&2,U32>,count: Maybe<&2,U32>,+values: List<&2,Element>,+fill: Element) ->
  {Contract.extents(~Element,Tensors.listed(~Element,shape,count,values,fill)) == shape : List<&2,U32>}:
  match count:
    case None{}: {==}
    case Some{+count}: listed_written_extents(~Element,shape,Storage.write_exactly(~Element,values,0,U32.to_nat(count),Storage.allocate(~Element,Tensors.depth_for(count),fill,U32.to_nat(count))),fill)

def from_list_extents(~Element: Data,+shape: List<&2,U32>,+values: List<&2,Element>,+fill: Element) ->
  {Contract.extents(~Element,Tensors.from_list(~Element,shape,values,fill)) == shape : List<&2,U32>}:
  listed_extents(~Element,shape,Tensors.count_of(shape),values,fill)

def listed_ready(~Element: Data,+shape: List<&2,U32>,+values: List<&2,Element>,+fill: Element,+workers: U32,+counted: Global.Counted<shape>,
  +length: {List.length(&2,Element,values) == ShapeLaws.elements(shape) : Nat}) ->
  {Contract.tensor_ready(~Element,Tensors.with_workers(~Element,Tensors.from_list(~Element,shape,values,fill),workers)) == True{} : Bool}:
  Global.Counted{+count,+accepted} = counted
  +exact = Count.count_exact(ShapeLaws.elements(shape),count,Count.count_of(shape,count,accepted))
  +compared = {%Equal.trans(Nat,List.length(&2,Element,values),ShapeLaws.elements(shape),U32.to_nat(count),length,exact) : {Nat.is_eq(List.length(&2,Element,values),_) == True{} : Bool}
    Order.equal_reflexive(List.length(&2,Element,values)) : {Nat.is_eq(List.length(&2,Element,values),U32.to_nat(count)) == True{} : Bool}}
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(shape),Some{count},accepted) :
    {Contract.tensor_ready(~Element,Tensors.with_workers(~Element,Tensors.listed(~Element,shape,_,values,fill),workers)) == True{} : Bool}
  %Equal.sym(Storage.Buffer<Element> & Bool,Storage.write_exactly(~Element,values,0,U32.to_nat(count),Storage.allocate(~Element,Tensors.depth_for(count),fill,U32.to_nat(count))),(Storage.write_at(~Element,values,0,Storage.allocate(~Element,Tensors.depth_for(count),fill,U32.to_nat(count))),True{}),
    exactly_matched(~Element,values,0,U32.to_nat(count),Storage.allocate(~Element,Tensors.depth_for(count),fill,U32.to_nat(count)),compared)) :
    {Contract.tensor_ready(~Element,Tensors.with_workers(~Element,Tensors.listed_written(~Element,shape,_,fill),workers)) == True{} : Bool}
  {==}
