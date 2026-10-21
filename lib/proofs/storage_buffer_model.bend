# Proof-only canonical owners retain the actual certificate. They support
# induction through affine Buffer callers without duplicating native arrays.
import Base
import ../storage_buffer.bend as Storage
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./storage_certificate.bend as Certificate
import ./storage_shape.bend as Shape
import ./storage_witness.bend as Witness
import ./word_power_of_two.bend as Power
import ./nat_order.bend as Order

type Content<-Element: Data> is Data:
  Content{tree: Tree.Tree<Element>,length: Nat}

def contents(~Element: Data,buffer: Storage.Buffer<Element>) -> Content<Element>:
  Storage.Buffer{storage,length,certificate} = buffer
  Content{Tree.reflect(Element,storage),length}

def value(~Element: Data,index: U32,content: Content<Element>) -> Element:
  Content{tree,length} = content
  Tree.read(Element,tree,index)

def content_tree(~Element: Data,content: Content<Element>) -> Tree.Tree<Element>:
  Content{tree,length} = content
  tree

def content_length(~Element: Data,content: Content<Element>) -> Nat:
  Content{tree,length} = content
  length

type ContentReady<-Element: Data,-content: Content<Element>> is Data:
  ContentReady{depth: Nat,balanced: Shape.Balanced(Element,depth,content_tree(~Element,content)),
    depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
    room: {Nat.is_le(content_length(~Element,content),U32.to_nat(Tree.capacity(Element,content_tree(~Element,content)))) == True{} : Bool}}

type Model<-Element: Data> is Data:
  Model{tree: Tree.Tree<Element>,length: Nat,certificate: Certificate.Certificate<Element,Tree.pack(Element,tree)>}

def content(~Element: Data,model: Model<Element>) -> Content<Element>:
  Model{tree,length,certificate} = model
  Content{tree,length}

def pack(~Element: Data,model: Model<Element>) -> Storage.Buffer<Element>:
  Model{tree,length,certificate} = model
  Storage.Buffer{Tree.pack(Element,tree),length,certificate}

def tree(~Element: Data,model: Model<Element>) -> Tree.Tree<Element>:
  Model{tree,length,certificate} = model
  tree

def length(~Element: Data,model: Model<Element>) -> Nat:
  Model{tree,length,certificate} = model
  length

def content_length_of(~Element: Data,+model: Model<Element>) -> {content_length(~Element,content(~Element,model)) == length(~Element,model) : Nat}:
  Model{tree,length,certificate} = model
  {==}

def depth(~Element: Data,model: Model<Element>) -> Nat:
  Model{tree,length,Certificate.Certificate{depth,equation}} = model
  depth

def balanced(~Element: Data,+model: Model<Element>) -> Shape.Balanced(Element,depth(~Element,model),tree(~Element,model)):
  Model{+tree,length,Certificate.Certificate{+depth,equation}} = model
  %StorageLaws.reflect_packed(Element,tree) : Shape.Balanced(Element,depth,_)
  Witness.balanced(Element,depth,Tree.pack(Element,tree),Certificate.witness_at(Element,depth,Tree.pack(Element,tree),equation))

def capacity_bound(~Element: Data,+model: Model<Element>,+depth_ok: {Nat.is_le(depth(~Element,model),30n) == True{} : Bool}) ->
  {U32.is_le(Tree.capacity(Element,tree(~Element,model)),2147483648) == True{} : Bool}:
  %Equal.sym(U32,Tree.capacity(Element,tree(~Element,model)),U32.shln(1,depth(~Element,model)),
    Shape.capacity(Element,depth(~Element,model),tree(~Element,model),balanced(~Element,model))) : {U32.is_le(_,2147483648) == True{} : Bool}
  Power.native_power_bound(depth(~Element,model),Order.successor(depth(~Element,model),30n,depth_ok))

def capacity_allocatable(~Element: Data,+model: Model<Element>,depth_ok: {Nat.is_le(depth(~Element,model),30n) == True{} : Bool}) ->
  {U32.is_le(Tree.capacity(Element,tree(~Element,model)),1073741824) == True{} : Bool}:
  %Equal.sym(U32,Tree.capacity(Element,tree(~Element,model)),U32.shln(1,depth(~Element,model)),
    Shape.capacity(Element,depth(~Element,model),tree(~Element,model),balanced(~Element,model))) : {U32.is_le(_,1073741824) == True{} : Bool}
  Power.native_allocatable(depth(~Element,model),depth_ok)

# Logical length and the native machine depth are independent of the stored
# element values. Allocation establishes them; content-preserving operations
# carry them with the same owner.
type Sized<-Element: Data,-model: Model<Element>> is Data:
  Sized{depth_ok: {Nat.is_le(depth(~Element,model),30n) == True{} : Bool},
    room: {Nat.is_le(length(~Element,model),U32.to_nat(Tree.capacity(Element,tree(~Element,model)))) == True{} : Bool}}

def ready_content(~Element: Data,+model: Model<Element>,evidence: ContentReady<Element,content(~Element,model)>) -> Sized<Element,model>:
  Model{+tree,+length,Certificate.Certificate{+depth,+equation}} = model
  ContentReady{+proved_depth,proved_balanced,depth_ok,room} = evidence
  Sized{
    %Equal.sym(Nat,depth,proved_depth,Shape.unique_depth(Element,tree,depth,proved_depth,balanced(~Element,model),proved_balanced)) : {Nat.is_le(_,30n) == True{} : Bool} depth_ok,
    room}

def content_ready(~Element: Data,+model: Model<Element>,evidence: Sized<Element,model>) -> ContentReady<Element,content(~Element,model)>:
  Model{+tree,+length,Certificate.Certificate{+depth,+equation}} = model
  Sized{depth_ok,room} = evidence
  ContentReady{depth,balanced(~Element,model),depth_ok,room}

def NativeReady(~Element: Data,buffer: Storage.Buffer<Element>) -> Data:
  Storage.Buffer{storage,length,Certificate.Certificate{depth,equation}} = buffer
  Shape.Both<{Nat.is_le(depth,30n) == True{} : Bool},{Nat.is_le(length,U32.to_nat(Tree.capacity(Element,Tree.reflect(Element,storage)))) == True{} : Bool}>

def native_content_ready(~Element: Data,buffer: Storage.Buffer<Element>,ready: NativeReady(~Element,buffer)) -> ContentReady<Element,contents(~Element,buffer)>:
  Storage.Buffer{storage,length,Certificate.Certificate{+depth,equation}} = buffer
  Shape.Both{depth_ok,room} = ready
  ContentReady{depth,Witness.balanced(Element,depth,storage,Certificate.witness_at(Element,depth,storage,equation)),depth_ok,room}

def ready_packed(~Element: Data,+model: Model<Element>,ready: NativeReady(~Element,pack(~Element,model))) -> Sized<Element,model>:
  Model{+tree,+length,Certificate.Certificate{+depth,equation}} = model
  Shape.Both{depth_ok,room} = ready
  Sized{depth_ok,
    %StorageLaws.reflect_packed(Element,tree) : {Nat.is_le(length,U32.to_nat(Tree.capacity(Element,_))) == True{} : Bool} room}

def ready_observed(~Element: Data,-buffer: Storage.Buffer<Element>,+model: Model<Element>,equation: {pack(~Element,model) == buffer : Storage.Buffer<Element>},
  ready: NativeReady(~Element,buffer)) -> Sized<Element,model>:
  ready_packed(~Element,model,
    %Equal.sym(Storage.Buffer<Element>,pack(~Element,model),buffer,equation) : NativeReady(~Element,_) ready)

def fresh_certificate(~Element: Data,+depth: Nat,+fill: Element) -> Certificate.Certificate<Element,Tree.pack(Element,Tree.fresh(Element,depth,fill))>:
  Certificate.Certificate{depth,
    %StorageLaws.fresh(Element,depth,fill) : {_ == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,_))) : Array<Element>}
    Certificate.fresh_equation(Element,depth,fill)}

def fresh(~Element: Data,+depth: Nat,+fill: Element,length: Nat) -> Model<Element>:
  Model{Tree.fresh(Element,depth,fill),length,fresh_certificate(~Element,depth,fill)}

def fresh_ready(~Element: Data,+depth: Nat,+fill: Element,+length: Nat,depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  room: {Nat.is_le(length,U32.to_nat(U32.shln(1,depth))) == True{} : Bool}) -> Sized<Element,fresh(~Element,depth,fill,length)>:
  Sized{depth_ok,
    %Equal.sym(U32,Tree.capacity(Element,Tree.fresh(Element,depth,fill)),U32.shln(1,depth),Shape.capacity(Element,depth,Tree.fresh(Element,depth,fill),Shape.fresh(Element,depth,fill))) :
      {Nat.is_le(length,U32.to_nat(_)) == True{} : Bool} room}

def packed_contents(~Element: Data,+model: Model<Element>) -> {contents(~Element,pack(~Element,model)) == content(~Element,model) : Content<Element>}:
  Model{+tree,+length,certificate} = model
  Equal.cong(Tree.Tree<Element>,Content<Element>,tree => Content{tree,length},Tree.reflect(Element,Tree.pack(Element,tree)),tree,StorageLaws.reflect_packed(Element,tree))

type Observation<-Element: Data,-buffer: Storage.Buffer<Element>> is Data:
  Observation{model: Model<Element>,equation: {pack(~Element,model) == buffer : Storage.Buffer<Element>}}

def proposition(~Element: Data,length: Nat,-storage: Array<Element>,-tree: Tree.Tree<Element>) -> Type:
  @certificate: Certificate.Certificate<Element,storage> -> Observation<Element,Storage.Buffer{storage,length,certificate}>

def observe(~Element: Data,buffer: Storage.Buffer<Element>) -> Observation<Element,buffer>:
  Storage.Buffer{storage,+length,certificate} = buffer
  StorageLaws.lift_observation(~Element,~Nat,~proposition(~Element),length,storage,StorageLaws.observe(Element,storage),
    tree => certificate => Observation{Model{tree,length,certificate},{==}})(certificate)

# A model of any buffer.
def observed_model(~Element: Data,-buffer: Storage.Buffer<Element>,observation: Observation<Element,buffer>) -> Model<Element>:
  Observation{model,equation} = observation
  model

def observed_equation(~Element: Data,-buffer: Storage.Buffer<Element>,+observation: Observation<Element,buffer>) ->
  {pack(~Element,observed_model(~Element,buffer,observation)) == buffer : Storage.Buffer<Element>}:
  Observation{model,equation} = observation
  equation

def remodel(~Element: Data,buffer: Storage.Buffer<Element>) -> Model<Element>:
  observed_model(~Element,buffer,observe(~Element,buffer))

def remodeled(~Element: Data,buffer: Storage.Buffer<Element>) -> {pack(~Element,remodel(~Element,buffer)) == buffer : Storage.Buffer<Element>}:
  observed_equation(~Element,buffer,observe(~Element,buffer))
def lift(~Element: Data,~Context: Data,~Proposition: Context -> @-buffer: Storage.Buffer<Element> -> Type,+context: Context,-buffer: Storage.Buffer<Element>,
  observation: Observation<Element,buffer>,canonical: @+model: Model<Element> -> Proposition(context,pack(~Element,model))) -> Proposition(context,buffer):
  Observation{model,equation} = observation
  %equation : Proposition(context,_)
  canonical(model)

def certificate_equation(~Element: Data,model: Model<Element>) ->
  {Tree.pack(Element,tree(~Element,model)) == Tree.pack(Element,Certificate.normalized(Element,depth(~Element,model),Tree.reflect(Element,Tree.pack(Element,tree(~Element,model))))) : Array<Element>}:
  Model{tree,length,Certificate.Certificate{depth,equation}} = model
  equation

# Certificate transport follows the native read equation at the original depth.
def read_result(~Element: Data,+model: Model<Element>,-value: Element,-result: Array<Element> & Element,
  +same: {(Tree.pack(Element,tree(~Element,model)),value) == result : Array<Element> & Element}) ->
  {Storage.read_result(~Element,length(~Element,model),result,
    Certificate.retained(~Element,depth(~Element,model),Tree.pack(Element,tree(~Element,model)),value,result,same,
      certificate_equation(~Element,model))) == (pack(~Element,model),value) : Storage.Buffer<Element> & Element}:
  Model{+tree,+length,Certificate.Certificate{+depth,+original}} = model
  %proof@same :
    {Storage.read_result(~Element,length,_,Certificate.retained(~Element,depth,Tree.pack(Element,tree),value,_,proof,original)) == (pack(~Element,model),value) : Storage.Buffer<Element> & Element}
  {==}

def read_observed(~Element: Data,+model: Model<Element>,+index: U32,
  observed: Witness.Read<Element,Tree.pack(Element,tree(~Element,model)),Array.get(Element,Tree.pack(Element,tree(~Element,model)),index)>) ->
  {Storage.read_result(~Element,length(~Element,model),Array.get(Element,Tree.pack(Element,tree(~Element,model)),index),
    Certificate.Certificate{depth(~Element,model),Certificate.read_equation(~Element,depth(~Element,model),Tree.pack(Element,tree(~Element,model)),index,certificate_equation(~Element,model),observed)}) ==
    (pack(~Element,model),Tree.read(Element,tree(~Element,model),index)) : Storage.Buffer<Element> & Element}:
  Model{+tree,+length,Certificate.Certificate{+depth,+original}} = model
  Witness.Read{value,+same} = observed
  Equal.trans(Storage.Buffer<Element> & Element,
    Storage.read_result(~Element,length,Array.get(Element,Tree.pack(Element,tree),index),Certificate.Certificate{depth,Certificate.read_equation(~Element,depth,Tree.pack(Element,tree),index,original,observed)}),
    (pack(~Element,model),value),(pack(~Element,model),Tree.read(Element,tree,index)),
    read_result(~Element,model,value,Array.get(Element,Tree.pack(Element,tree),index),Equal.sym(Array<Element> & Element,Array.get(Element,Tree.pack(Element,tree),index),(Tree.pack(Element,tree),value),same)),
    Equal.cong(Element,Storage.Buffer<Element> & Element,value => (pack(~Element,model),value),value,Tree.read(Element,tree,index),
      Equal.cong(Array<Element> & Element,Element,pair => Pair.snd(Array<Element>,Element,pair),(Tree.pack(Element,tree),value),(Tree.pack(Element,tree),Tree.read(Element,tree,index)),
        Equal.trans(Array<Element> & Element,(Tree.pack(Element,tree),value),Array.get(Element,Tree.pack(Element,tree),index),(Tree.pack(Element,tree),Tree.read(Element,tree,index)),
          Equal.sym(Array<Element> & Element,Array.get(Element,Tree.pack(Element,tree),index),(Tree.pack(Element,tree),value),same),StorageLaws.read(Element,tree,index)))))

def read(~Element: Data,+model: Model<Element>,+index: U32) ->
  {Storage.read(~Element,pack(~Element,model),index) == (pack(~Element,model),Tree.read(Element,tree(~Element,model),index)) : Storage.Buffer<Element> & Element}:
  Model{+tree,+length,Certificate.Certificate{+depth,+equation}} = model
  read_observed(~Element,model,index,Witness.observed(Element,Tree.pack(Element,tree),index,Witness.ArrayWitness{depth,Certificate.witness_at(Element,depth,Tree.pack(Element,tree),equation)}))

def read_masked(~Element: Data,+fallback: Element,valid: Bool,+model: Model<Element>,+index: U32) ->
  {Storage.read_masked(~Element,fallback,valid,pack(~Element,model),index) ==
    (pack(~Element,model),Bool.pick(Element,valid,Tree.read(Element,tree(~Element,model),index),fallback)) : Storage.Buffer<Element> & Element}:
  match valid:
    case False{}: {==}
    case True{}: read(~Element,model,index)

def content_parts(+model: Model<F32>) -> {content(~F32,model) == Content{tree(~F32,model),length(~F32,model)} : Content<F32>}:
  Model{tree,length,certificate} = model
  {==}

def model_length(+model: Model<F32>) -> {content_length(~F32,content(~F32,model)) == length(~F32,model) : Nat}:
  Model{tree,length,certificate} = model
  {==}

def read_content(+index: U32,+content: Content<F32>) -> {Tree.read(F32,content_tree(~F32,content),index) == value(~F32,index,content) : F32}:
  Content{tree,length} = content
  {==}

def buffer_contents(~Element: Data,+model: Model<Element>,-buffer: Storage.Buffer<Element>,equal: {pack(~Element,model) == buffer : Storage.Buffer<Element>}) ->
  {content(~Element,model) == contents(~Element,buffer) : Content<Element>}:
  Equal.trans(Content<Element>,content(~Element,model),contents(~Element,pack(~Element,model)),contents(~Element,buffer),
    Equal.sym(Content<Element>,contents(~Element,pack(~Element,model)),content(~Element,model),packed_contents(~Element,model)),
    Equal.cong(Storage.Buffer<Element>,Content<Element>,contents(~Element),pack(~Element,model),buffer,equal))

def content_tree_of(~Element: Data,+model: Model<Element>) -> {content_tree(~Element,content(~Element,model)) == tree(~Element,model) : Tree.Tree<Element>}:
  Model{tree,length,certificate} = model
  {==}

def observed_content(~Element: Data,-buffer: Storage.Buffer<Element>,observation: Observation<Element,buffer>) ->
  {content(~Element,observed_model(~Element,buffer,observation)) == contents(~Element,buffer) : Content<Element>}:
  Observation{+model,equation} = observation
  buffer_contents(~Element,model,buffer,equation)

def remodel_content(~Element: Data,buffer: Storage.Buffer<Element>) ->
  {content(~Element,remodel(~Element,buffer)) == contents(~Element,buffer) : Content<Element>}:
  observed_content(~Element,buffer,observe(~Element,buffer))
# A value written into a buffer is written into its contents.
def write_value_packed(~Element: Data,+index: U32,+value: Element,+model: Model<Element>) ->
  {contents(~Element,Storage.write(~Element,index,value,pack(~Element,model))) ==
    Content{Tree.write(Element,content_tree(~Element,content(~Element,model)),index,value),content_length(~Element,content(~Element,model))} : Content<Element>}:
  Model{+tree,+length,certificate} = model
  %Equal.sym(Array<Element>,Array.set(Element,Tree.pack(Element,tree),index,value),Tree.pack(Element,Tree.write(Element,tree,index,value)),
    StorageLaws.write(Element,tree,index,value)) :
    {Content{Tree.reflect(Element,_),length} == Content{Tree.write(Element,tree,index,value),length} : Content<Element>}
  Equal.cong(Tree.Tree<Element>,Content<Element>,tree => Content{tree,length},Tree.reflect(Element,Tree.pack(Element,Tree.write(Element,tree,index,value))),
    Tree.write(Element,tree,index,value),StorageLaws.reflect_packed(Element,Tree.write(Element,tree,index,value)))
