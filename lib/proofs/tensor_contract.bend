# The public contract of a tensor states the finite content and allocation of
# its whole storage, read at contiguous native indices. A whole tensor is its
# own whole storage. Band operations keep the band invariant beside it.
import Base
import ../tensor.bend as Tensors
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_allocation_refinement.bend as AllocationLaws
import ./tensor_shape_refinement.bend as ShapeLaws
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds

type WellFormed<-Element: Data,-shape: List<&2,U32>,-content: Buffer.Content<Element>> is Data:
  WellFormed{length_exact: {Buffer.content_length(~Element,content) == ShapeLaws.elements(shape) : Nat},ready: Buffer.ContentReady<Element,content>}

def At(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+content: Buffer.Content<Element>,+index: Nat) -> Data:
  {Buffer.value(~Element,U32.from_nat(index),content) == original(logical,U32.from_nat(index)) : Element}

def Values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+content: Buffer.Content<Element>) -> Data:
  Members.All(Nat,{index => At(~Element,~Logical,~original,logical,content,index) : Nat -> Data},Ranges.range(Buffer.content_length(~Element,content),0n))

def read(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+content: Buffer.Content<Element>,+index: Nat,
  values: Values(~Element,~Logical,~original,logical,content),inside: {Nat.is_lt(index,Buffer.content_length(~Element,content)) == True{} : Bool}) ->
  {Buffer.value(~Element,U32.from_nat(index),content) == original(logical,U32.from_nat(index)) : Element}:
  Members.all_member(Nat,{index => At(~Element,~Logical,~original,logical,content,index) : Nat -> Data},index,Ranges.range(Buffer.content_length(~Element,content),0n),values,
    Members.range_member(Buffer.content_length(~Element,content),0n,index,inside))

# The values a reader gives to count indices from offset, in order.
def listed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,count: Nat,+offset: Nat,+logical: Logical) -> List<&2,Element>:
  match count:
    case 0n: []
    case 1n+rest: original(logical,U32.from_nat(offset)) <> listed(~Element,~Logical,~original,rest,1n+offset,logical)

# Storage that holds the reader's values reads back as the reader's values.
def read_listed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+tree: Tree.Tree<Element>,+length: Nat,
  +values: Values(~Element,~Logical,~original,logical,Buffer.Content{tree,length}),count: Nat,+offset: Nat,
  +room: {Nat.is_le(Nat.add(offset,count),length) == True{} : Bool}) ->
  {Tree.read_sequence(~Element,count,U32.from_nat(offset),tree) == listed(~Element,~Logical,~original,count,offset,logical) : List<&2,Element>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(Element,Tree.read(Element,tree,U32.from_nat(offset)),original(logical,U32.from_nat(offset)),
        read(~Element,~Logical,~original,logical,Buffer.Content{tree,length},offset,values,Order.room_index(offset,rest,length,room))) :
        {_ <> Tree.read_sequence(~Element,rest,(U32.from_nat(offset) + 1 : U32),tree) == listed(~Element,~Logical,~original,1n+rest,offset,logical) : List<&2,Element>}
      %Equal.sym(U32,(U32.from_nat(offset) + 1 : U32),U32.from_nat(1n+offset),Bounds.add_one(U32.from_nat(offset))) :
        {original(logical,U32.from_nat(offset)) <> Tree.read_sequence(~Element,rest,_,tree) == listed(~Element,~Logical,~original,1n+rest,offset,logical) : List<&2,Element>}
      Equal.cong(List<&2,Element>,List<&2,Element>,tail => original(logical,U32.from_nat(offset)) <> tail,
        Tree.read_sequence(~Element,rest,U32.from_nat(1n+offset),tree),listed(~Element,~Logical,~original,rest,1n+offset,logical),
        read_listed(~Element,~Logical,~original,logical,tree,length,values,rest,1n+offset,Order.advance_room(offset,rest,length,room)))

# A buffer read as its own logical values: the reader with which a call on a
# buffer states its values when the caller relates the buffer to a larger
# input.
def stored(~Element: Data,content: Buffer.Content<Element>,index: U32) -> Element:
  Buffer.value(~Element,index,content)

def StoredAt(~Element: Data,+content: Buffer.Content<Element>,+index: Nat) -> Data:
  At(~Element,~Buffer.Content<Element>,~stored(~Element),content,content,index)

def stored_at(~Element: Data,+content: Buffer.Content<Element>,+index: Nat,+inside: {Nat.is_lt(index,Buffer.content_length(~Element,content)) == True{} : Bool}) ->
  StoredAt(~Element,content,index):
  {==}

def stored_values(~Element: Data,+content: Buffer.Content<Element>) -> Values(~Element,~Buffer.Content<Element>,~stored(~Element),content,content):
  Members.range_all(~Buffer.Content<Element>,~StoredAt(~Element),~Buffer.content_length(~Element),~stored_at(~Element),
    Buffer.content_length(~Element,content),0n,content,Order.reflexive(Buffer.content_length(~Element,content)))

def tensor_ready(~Element: Data,tensor: Tensors.Tensor<Element>) -> Bool:
  match tensor:
    case Tensors.Tensor{bands,geometry,shape,fill,status,workers}: Tensors.usable(status)

def extents(~Element: Data,tensor: Tensors.Tensor<Element>) -> List<&2,U32>:
  match tensor:
    case Tensors.Tensor{bands,geometry,shape,fill,status,workers}: shape

def empty_contents(~Element: Data,+fill: Element) ->
  {Buffer.contents(~Element,Tensors.buffer(~Element,fill,0)) == Buffer.Content{Tree.fresh(Element,Tensors.depth_for(0),fill),0n} : Buffer.Content<Element>}:
  AllocationLaws.contents(~Element,Tensors.depth_for(0),fill,0n)

def empty_ready(~Element: Data,+fill: Element) -> Buffer.ContentReady<Element,Buffer.contents(~Element,Tensors.buffer(~Element,fill,0))>:
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Tensors.buffer(~Element,fill,0)),Buffer.Content{Tree.fresh(Element,Tensors.depth_for(0),fill),0n},empty_contents(~Element,fill)) : Buffer.ContentReady<Element,_>
  Buffer.content_ready(~Element,Buffer.fresh(~Element,Tensors.depth_for(0),fill,0n),Buffer.fresh_ready(~Element,Tensors.depth_for(0),fill,0n,{==},{==}))
