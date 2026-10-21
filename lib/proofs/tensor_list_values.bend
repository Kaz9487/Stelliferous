# from_list: a Ready result has the Normal state of one leaf and holds the
# values of any reader that agrees with the list inside the shape.
import Base
import ../tensor.bend as Tensors
import ../storage_buffer.bend as Storage
import ../traversal_array.bend as Traversal
import ./storage_tree_model.bend as Tree
import ./storage_buffer_model.bend as Buffer
import ./storage_refinement.bend as Native
import ./storage_certificate.bend as Certificate
import ./storage_witness.bend as Witness
import ./storage_shape.bend as Shape
import ./tensor_contract.bend as Contract
import ./tensor_whole_leaf.bend as WholeLeaf
import ./tensor_gather_values.bend as GatherValues
import ./tensor_global_shape_count.bend as Global
import ./tensor_allocation_depth.bend as Depth
import ./tensor_count_refinement.bend as Count
import ./tensor_shape_refinement.bend as ShapeLaws
import ./list_storage_readback.bend as ListRead
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_full_range.bend as FullRange
import ./sequence_membership.bend as Members
import ./nat_order.bend as Order
import ./word_encoding.bend as Words
import ./storage_allocation_refinement.bend as AllocationLaws
import ../storage_allocation.bend as Allocation

# The items of a list read by index. The fill stands for an index past the
# end and is never read in a Ready result.
type Items<-Element: Data> is Data:
  Items{values: List<&2,Element>,fill: Element}

def item(~Element: Data,context: Items<Element>,index: U32) -> Element:
  Items{values,+fill} = context
  ListRead.at(~Element,fill,U32.to_nat(index),values)

def contents(~Element: Data,+values: List<&2,Element>,+depth: Nat,+fill: Element,+count: Nat) -> {Buffer.contents(~Element,Storage.from_list(~Element,values,depth,fill,count)) == Buffer.Content{Tree.write_sequence(~Element,values,0,Tree.fresh(Element,depth,fill)),count} : Buffer.Content<Element>}:
  %Equal.sym(Array<Element>,Array.new(Element,depth,fill),Tree.pack(Element,Tree.fresh(Element,depth,fill)),Native.fresh(Element,depth,fill)) :
    {Buffer.Content{Tree.reflect(Element,Storage.write_list(~Element,values,0,_)),count} == Buffer.Content{Tree.write_sequence(~Element,values,0,Tree.fresh(Element,depth,fill)),count} : Buffer.Content<Element>}
  %Equal.sym(Array<Element>,Storage.write_list(~Element,values,0,Tree.pack(Element,Tree.fresh(Element,depth,fill))),Tree.pack(Element,Tree.write_sequence(~Element,values,0,Tree.fresh(Element,depth,fill))),Native.write_sequence(~Element,values,0,Tree.fresh(Element,depth,fill))) :
    {Buffer.Content{Tree.reflect(Element,_),count} == Buffer.Content{Tree.write_sequence(~Element,values,0,Tree.fresh(Element,depth,fill)),count} : Buffer.Content<Element>}
  Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{tree,count},Tree.reflect(Element,Tree.pack(Element,Tree.write_sequence(~Element,values,0,Tree.fresh(Element,depth,fill)))),Tree.write_sequence(~Element,values,0,Tree.fresh(Element,depth,fill)),Native.reflect_packed(Element,Tree.write_sequence(~Element,values,0,Tree.fresh(Element,depth,fill))))

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
  Shape.Both{depth_ok,
    %Equal.sym(U32,Tree.capacity(Element,Tree.reflect(Element,Traversal.write_list(~Element,values,0,Array.new(Element,Tensors.depth_for(count),fill)))),U32.shln(1,Tensors.depth_for(count)),
      Shape.capacity(Element,Tensors.depth_for(count),Tree.reflect(Element,Traversal.write_list(~Element,values,0,Array.new(Element,Tensors.depth_for(count),fill))),
        Witness.balanced(Element,Tensors.depth_for(count),Traversal.write_list(~Element,values,0,Array.new(Element,Tensors.depth_for(count),fill)),
          Certificate.written(~Element,Tensors.depth_for(count),values,0,Array.new(Element,Tensors.depth_for(count),fill),Witness.fresh(Element,Tensors.depth_for(count),fill))))) :
      {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool}
    room}

# A list of the counted length written whole into a fresh tree holds the
# values of a reader that agrees with the list at every index of the shape.
def listed_facts(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,+shape: List<&2,U32>,+values: List<&2,Element>,+fill: Element,+count: U32,
  +counted: Count.Count<ShapeLaws.elements(shape),count>,+equal: {List.length(&2,Element,values) == U32.to_nat(count) : Nat},
  agrees: @+count: U32 -> @+counted: Count.Count<ShapeLaws.elements(shape),count> -> @+index: Nat -> @+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool} ->
    {ListRead.at(~Element,fill,index,values) == reader(context,U32.from_nat(index)) : Element}) ->
  GatherValues.Facts(~Element,~Context,~reader,context,shape,Buffer.Content{Tree.write_sequence(~Element,values,0,Tree.fresh(Element,Tensors.depth_for(count),fill)),U32.to_nat(count)}):
  Count.Count{+exact,+bounded} = counted
  %contents(~Element,values,Tensors.depth_for(count),fill,U32.to_nat(count)) : GatherValues.Facts(~Element,~Context,~reader,context,shape,_)
  Shape.Both{
    Contract.WellFormed{Equal.sym(Nat,ShapeLaws.elements(shape),U32.to_nat(count),exact),
      Buffer.native_content_ready(~Element,Storage.from_list(~Element,values,Tensors.depth_for(count),fill,U32.to_nat(count)),
        written_ready(~Element,values,count,fill,Depth.bounded(count,bounded)))},
    Members.range_all(~Unit,
      ~(scope => index => Contract.At(~Element,~Context,~reader,context,Buffer.contents(~Element,Storage.from_list(~Element,values,Tensors.depth_for(count),fill,U32.to_nat(count))),index)),
      ~(scope => U32.to_nat(count)),
      ~(scope => index => inside => Equal.trans(Element,
        Buffer.value(~Element,U32.from_nat(index),Buffer.contents(~Element,Storage.from_list(~Element,values,Tensors.depth_for(count),fill,U32.to_nat(count)))),
        ListRead.at(~Element,fill,index,values),reader(context,U32.from_nat(index)),
        written(~Element,values,count,fill,index,equal,inside,Depth.bounded(count,bounded)),agrees(count,Count.Count{exact,bounded},index,inside))),
      U32.to_nat(count),0n,Unit{},Order.reflexive(U32.to_nat(count)))}

def listed_ready(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,+shape: List<&2,U32>,+values: List<&2,Element>,+fill: Element,+count: U32,
  +counted: Count.Count<ShapeLaws.elements(shape),count>,+equal: {List.length(&2,Element,values) == U32.to_nat(count) : Nat},
  agrees: @+count: U32 -> @+counted: Count.Count<ShapeLaws.elements(shape),count> -> @+index: Nat -> @+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool} ->
    {ListRead.at(~Element,fill,index,values) == reader(context,U32.from_nat(index)) : Element}) ->
  Global.Contract(~Element,~Context,~reader,context,Tensors.listed_count(~Element,True{},shape,count,values,fill)):
  WholeLeaf.stored(~Element,~Context,~reader,context,Storage.from_list(~Element,values,Tensors.depth_for(count),fill,U32.to_nat(count)),shape,fill,1,
    Buffer.observe(~Element,Storage.from_list(~Element,values,Tensors.depth_for(count),fill,U32.to_nat(count))),
    %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Storage.from_list(~Element,values,Tensors.depth_for(count),fill,U32.to_nat(count))),Buffer.Content{Tree.write_sequence(~Element,values,0,Tree.fresh(Element,Tensors.depth_for(count),fill)),U32.to_nat(count)},
        contents(~Element,values,Tensors.depth_for(count),fill,U32.to_nat(count))) :
      GatherValues.Facts(~Element,~Context,~reader,context,shape,_)
    listed_facts(~Element,~Context,~reader,context,shape,values,fill,count,counted,equal,agrees))

# A buffer that holds such a list is a whole tensor of the shape with the
# reader's values.
def stored_listed(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,-storage: Storage.Buffer<Element>,+shape: List<&2,U32>,
  +values: List<&2,Element>,+fill: Element,+count: U32,+workers: U32,observation: Buffer.Observation<Element,storage>,
  +stored: {Buffer.contents(~Element,storage) == Buffer.Content{Tree.write_sequence(~Element,values,0,Tree.fresh(Element,Tensors.depth_for(count),fill)),U32.to_nat(count)} : Buffer.Content<Element>},
  +counted: Count.Count<ShapeLaws.elements(shape),count>,+equal: {List.length(&2,Element,values) == U32.to_nat(count) : Nat},
  agrees: @+count: U32 -> @+counted: Count.Count<ShapeLaws.elements(shape),count> -> @+index: Nat -> @+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool} ->
    {ListRead.at(~Element,fill,index,values) == reader(context,U32.from_nat(index)) : Element}) ->
  Global.Contract(~Element,~Context,~reader,context,Tensors.whole(~Element,storage,shape,fill,Tensors.Usable{},workers)):
  WholeLeaf.stored(~Element,~Context,~reader,context,storage,shape,fill,workers,observation,
    %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,storage),Buffer.Content{Tree.write_sequence(~Element,values,0,Tree.fresh(Element,Tensors.depth_for(count),fill)),U32.to_nat(count)},stored) : GatherValues.Facts(~Element,~Context,~reader,context,shape,_)
    listed_facts(~Element,~Context,~reader,context,shape,values,fill,count,counted,equal,agrees))

def listed_matches(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,matches: Bool,+context: Context,+shape: List<&2,U32>,+values: List<&2,Element>,+fill: Element,+count: U32,
  +compared: {Nat.is_eq(List.length(&2,Element,values),U32.to_nat(count)) == matches : Bool},+counted: Count.Count<ShapeLaws.elements(shape),count>,
  agrees: @+count: U32 -> @+counted: Count.Count<ShapeLaws.elements(shape),count> -> @+index: Nat -> @+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool} ->
    {ListRead.at(~Element,fill,index,values) == reader(context,U32.from_nat(index)) : Element},
  accepted: {Contract.tensor_ready(~Element,Tensors.listed_count(~Element,matches,shape,count,values,fill)) == True{} : Bool}) ->
  Global.Contract(~Element,~Context,~reader,context,Tensors.listed_count(~Element,matches,shape,count,values,fill)):
  match matches:
    case False{}:
      Empty.absurd(Global.Contract(~Element,~Context,~reader,context,Tensors.listed_count(~Element,False{},shape,count,values,fill)),Words.false_true(accepted))
    case True{}: listed_ready(~Element,~Context,~reader,context,shape,values,fill,count,counted,Order.equal_of_is_eq(List.length(&2,Element,values),U32.to_nat(count),compared),agrees)

def selected(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,result: Maybe<&2,U32>,+context: Context,+shape: List<&2,U32>,+values: List<&2,Element>,+fill: Element,
  +counts: {Tensors.count_of(shape) == result : Maybe<&2,U32>},
  agrees: @+count: U32 -> @+counted: Count.Count<ShapeLaws.elements(shape),count> -> @+index: Nat -> @+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool} ->
    {ListRead.at(~Element,fill,index,values) == reader(context,U32.from_nat(index)) : Element},
  accepted: {Contract.tensor_ready(~Element,Tensors.listed(~Element,shape,result,values,fill)) == True{} : Bool}) ->
  Global.Contract(~Element,~Context,~reader,context,Tensors.listed(~Element,shape,result,values,fill)):
  match result:
    case None{}:
      Empty.absurd(Global.Contract(~Element,~Context,~reader,context,Tensors.listed(~Element,shape,None{},values,fill)),Words.false_true(accepted))
    case Some{+count}:
      listed_matches(~Element,~Context,~reader,Nat.is_eq(List.length(&2,Element,values),U32.to_nat(count)),context,shape,values,fill,count,{==},Count.count_of(shape,count,counts),agrees,accepted)

# A Ready from_list tensor holds the values of a reader that agrees with
# the list at every index of the shape.
def from_list_read(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,+shape: List<&2,U32>,+values: List<&2,Element>,+fill: Element,
  agrees: @+count: U32 -> @+counted: Count.Count<ShapeLaws.elements(shape),count> -> @+index: Nat -> @+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool} ->
    {ListRead.at(~Element,fill,index,values) == reader(context,U32.from_nat(index)) : Element},
  accepted: {Contract.tensor_ready(~Element,Tensors.from_list(~Element,shape,values,fill)) == True{} : Bool}) ->
  Global.Contract(~Element,~Context,~reader,context,Tensors.from_list(~Element,shape,values,fill)):
  selected(~Element,~Context,~reader,Tensors.count_of(shape),context,shape,values,fill,{==},agrees,accepted)

# The item at an index of the shape is the item of the list there.
def item_agrees(~Element: Data,+shape: List<&2,U32>,+values: List<&2,Element>,+fill: Element,+count: U32,+counted: Count.Count<ShapeLaws.elements(shape),count>,
  +index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {ListRead.at(~Element,fill,index,values) == item(~Element,Items{values,fill},U32.from_nat(index)) : Element}:
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,FullRange.roundtrip(index,count,Bounds.strict_le(index,U32.to_nat(count),inside))) :
    {ListRead.at(~Element,fill,index,values) == ListRead.at(~Element,fill,_,values) : Element}
  {==}

# A Ready from_list tensor holds the items of the list in order.
def from_list(~Element: Data,+shape: List<&2,U32>,+values: List<&2,Element>,+fill: Element,
  accepted: {Contract.tensor_ready(~Element,Tensors.from_list(~Element,shape,values,fill)) == True{} : Bool}) ->
  Global.Contract(~Element,~Items<Element>,~item(~Element),Items{values,fill},Tensors.from_list(~Element,shape,values,fill)):
  from_list_read(~Element,~Items<Element>,~item(~Element),Items{values,fill},shape,values,fill,
    count => counted => index => inside => item_agrees(~Element,shape,values,fill,count,counted,index,inside),accepted)
