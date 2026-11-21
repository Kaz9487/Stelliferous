# The ternary kernel updates its yes buffer in place: each of the first count
# values becomes choose of the mask, the value and the second operand there.
import Base
import ../tensor_access.bend as Access
import ../tensor_operations.bend as Operations
import ./tensor_access_refinement.bend as AccessLaws
import ./tensor_access_certificate.bend as AccessCertificate
import ./tensor_pieces_combine_leaf.bend as CombineLeaf
import ./traversal_witness.bend as TraversalWitness
import ./storage_certificate.bend as Certificate
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws

# The values the kernel leaves in the tree of the yes buffer.
def output(~Element: Data,~choose: Element -> Element -> Element -> Element,count: Nat,mask: Tree.Tree<Element>,no: Tree.Tree<Element>,yes: Tree.Tree<Element>) -> Tree.Tree<Element>:
  CombineLeaf.updated(~Element,~AccessLaws.Choices<Element>,~Access.Choice<Element>,~AccessLaws.sample_choice(~Element),~Access.chosen(~Element,~choose),count,0,
    AccessLaws.Choices{mask,no},yes)

def raw(~Element: Data,+length: Nat,result: (Array<Element> & Array<Element>) & Array<Element>) -> Buffer.Content<Element>:
  (reader,values) = result
  Buffer.Content{Tree.reflect(Element,values),length}

def chosen_contents(~Element: Data,+length: Nat,result: (Array<Element> & Array<Element>) & Array<Element>,
  certificate: Certificate.Certificate<Element,TraversalWitness.second(~(Array<Element> & Array<Element>),~Array<Element>,result)>) ->
  {Buffer.contents(~Element,Operations.chosen_buffer(~Element,length,result,certificate)) == raw(~Element,length,result) : Buffer.Content<Element>}:
  (reader,values) = result
  {==}

def packed_result(~Element: Data,+length: Nat,+result: AccessLaws.ModelResult<AccessLaws.Choices<Element>,Element>) ->
  {raw(~Element,length,AccessLaws.pack_result(~(Array<Element> & Array<Element>),~AccessLaws.Choices<Element>,~Element,~AccessLaws.pack_choices(~Element),result)) ==
    Buffer.Content{CombineLeaf.readback_reader(~AccessLaws.Choices<Element>,~Element,result),length} : Buffer.Content<Element>}:
  AccessLaws.ModelResult{+reader,+values} = result
  Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{tree,length},Tree.reflect(Element,Tree.pack(Element,values)),values,StorageLaws.reflect_packed(Element,values))

# The contents of the kernel's result.
def kernel(~Element: Data,~choose: Element -> Element -> Element -> Element,+count: Nat,+mask: Buffer.Model<Element>,+yes: Buffer.Model<Element>,+no: Buffer.Model<Element>) ->
  {Buffer.contents(~Element,Operations.choose_kernel(~Element,~choose,count,Buffer.pack(~Element,mask),Buffer.pack(~Element,yes),Buffer.pack(~Element,no))) ==
    Buffer.Content{output(~Element,~choose,count,Buffer.tree(~Element,mask),Buffer.tree(~Element,no),Buffer.tree(~Element,yes)),Buffer.length(~Element,yes)} : Buffer.Content<Element>}:
  Buffer.Model{+mask_tree,+mask_length,+mask_certificate} = mask
  Buffer.Model{+yes_tree,+yes_length,+yes_certificate} = yes
  Buffer.Model{+no_tree,+no_length,+no_certificate} = no
  Equal.trans(Buffer.Content<Element>,
    Buffer.contents(~Element,Operations.chosen_buffer(~Element,yes_length,Access.choose_run(~Element,~choose,count,Tree.pack(Element,mask_tree),Tree.pack(Element,no_tree),Tree.pack(Element,yes_tree)),
      AccessCertificate.choose(~Element,~choose,count,Tree.pack(Element,mask_tree),Tree.pack(Element,no_tree),Tree.pack(Element,yes_tree),mask_certificate,no_certificate,yes_certificate))),
    raw(~Element,yes_length,Access.choose_run(~Element,~choose,count,Tree.pack(Element,mask_tree),Tree.pack(Element,no_tree),Tree.pack(Element,yes_tree))),
    Buffer.Content{output(~Element,~choose,count,mask_tree,no_tree,yes_tree),yes_length},
    chosen_contents(~Element,yes_length,Access.choose_run(~Element,~choose,count,Tree.pack(Element,mask_tree),Tree.pack(Element,no_tree),Tree.pack(Element,yes_tree)),
      AccessCertificate.choose(~Element,~choose,count,Tree.pack(Element,mask_tree),Tree.pack(Element,no_tree),Tree.pack(Element,yes_tree),mask_certificate,no_certificate,yes_certificate)),
    %Equal.sym((Array<Element> & Array<Element>) & Array<Element>,
      Access.choose_run(~Element,~choose,count,Tree.pack(Element,mask_tree),Tree.pack(Element,no_tree),Tree.pack(Element,yes_tree)),
      AccessLaws.pack_result(~(Array<Element> & Array<Element>),~AccessLaws.Choices<Element>,~Element,~AccessLaws.pack_choices(~Element),
        AccessLaws.choose_model(~Element,~choose,count,mask_tree,no_tree,yes_tree)),
      AccessLaws.choose(~Element,~choose,count,mask_tree,no_tree,yes_tree)) :
      {raw(~Element,yes_length,_) == Buffer.Content{output(~Element,~choose,count,mask_tree,no_tree,yes_tree),yes_length} : Buffer.Content<Element>}
    packed_result(~Element,yes_length,AccessLaws.choose_model(~Element,~choose,count,mask_tree,no_tree,yes_tree)))

# The output keeps the shape of the yes tree.
def ready(~Element: Data,~choose: Element -> Element -> Element -> Element,+count: Nat,+mask: Buffer.Model<Element>,+yes: Buffer.Model<Element>,+no: Buffer.Model<Element>,
  ready: Buffer.Sized<Element,yes>) ->
  Buffer.ContentReady<Element,Buffer.Content{output(~Element,~choose,count,Buffer.tree(~Element,mask),Buffer.tree(~Element,no),Buffer.tree(~Element,yes)),Buffer.length(~Element,yes)}>:
  Buffer.Model{+mask_tree,mask_length,mask_certificate} = mask
  Buffer.Model{+yes_tree,+length,+certificate} = yes
  Buffer.Model{+no_tree,no_length,no_certificate} = no
  Buffer.Sized{+depth_ok,room} = ready
  +yes_model = {Buffer.Model{yes_tree,length,certificate} : Buffer.Model<Element>}
  +choices = {AccessLaws.Choices{mask_tree,no_tree} : AccessLaws.Choices<Element>}
  Buffer.ContentReady{Buffer.depth(~Element,yes_model),
    CombineLeaf.updated_balanced(~Element,~AccessLaws.Choices<Element>,~Access.Choice<Element>,~AccessLaws.sample_choice(~Element),~Access.chosen(~Element,~choose),count,0,choices,yes_tree,
      Buffer.depth(~Element,yes_model),Buffer.balanced(~Element,yes_model)),depth_ok,
    %Equal.sym(U32,Tree.capacity(Element,output(~Element,~choose,count,mask_tree,no_tree,yes_tree)),Tree.capacity(Element,yes_tree),
      CombineLeaf.updated_capacity(~Element,~AccessLaws.Choices<Element>,~Access.Choice<Element>,~AccessLaws.sample_choice(~Element),~Access.chosen(~Element,~choose),count,0,choices,yes_tree)) :
      {Nat.is_le(length,U32.to_nat(_)) == True{} : Bool}
    room}

# Every one of the first count values is the choice at its index.
def value(~Element: Data,~choose: Element -> Element -> Element -> Element,+count: Nat,+mask: Buffer.Model<Element>,+yes: Buffer.Model<Element>,+no: Buffer.Model<Element>,
  +index: Nat,ready: Buffer.Sized<Element,yes>,+inside: {Nat.is_lt(index,count) == True{} : Bool},
  +counted: {count == Buffer.length(~Element,yes) : Nat}) ->
  {Tree.read(Element,output(~Element,~choose,count,Buffer.tree(~Element,mask),Buffer.tree(~Element,no),Buffer.tree(~Element,yes)),U32.from_nat(index)) ==
    choose(Tree.read(Element,Buffer.tree(~Element,mask),U32.from_nat(index)),Tree.read(Element,Buffer.tree(~Element,yes),U32.from_nat(index)),
      Tree.read(Element,Buffer.tree(~Element,no),U32.from_nat(index))) : Element}:
  Buffer.Model{+mask_tree,mask_length,mask_certificate} = mask
  Buffer.Model{+yes_tree,+length,+certificate} = yes
  Buffer.Model{+no_tree,no_length,no_certificate} = no
  Buffer.Sized{+depth_ok,+sized_room} = ready
  +yes_model = {Buffer.Model{yes_tree,length,certificate} : Buffer.Model<Element>}
  CombineLeaf.updated_inside(~Element,~AccessLaws.Choices<Element>,~Access.Choice<Element>,~AccessLaws.sample_choice(~Element),~Access.chosen(~Element,~choose),count,0n,index,
    AccessLaws.Choices{mask_tree,no_tree},yes_tree,Buffer.depth(~Element,yes_model),Tree.capacity(Element,yes_tree),{==},Buffer.balanced(~Element,yes_model),depth_ok,
    Buffer.capacity_bound(~Element,yes_model,depth_ok),
    {%Equal.sym(Nat,count,length,counted) : {Nat.is_le(_,U32.to_nat(Tree.capacity(Element,yes_tree))) == True{} : Bool} sized_room :
      {Nat.is_le(count,U32.to_nat(Tree.capacity(Element,yes_tree))) == True{} : Bool}},inside)
