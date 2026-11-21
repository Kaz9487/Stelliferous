# Normal usable tensors retain their actual public shape count, and their
# pieces are the canonical pieces of the shape and cover its cells. A failed
# tensor carries no layout guarantee.
import Base
import ../tensor.bend as Tensors
import ../tensor_pieces.bend as Pieces
import ./tensor_pieces_model.bend as Model
import ./tensor_pieces_content.bend as Content
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_count_refinement.bend as Count
import ./tensor_contract.bend as Contract
import ./word_encoding.bend as WordEncoding
import ../kernel_gemm.bend as KernelGemm

type Counted<-shape: List<&2,U32>> is Data:
  Counted{count: U32,accepted: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>}}

# The pieces of contents are the canonical pieces of a shape.
def Canonical(~Element: Data,+shape: List<&2,U32>,content: Model.Content<Element>) -> Data:
  {Model.content_layout(~Element,content) == Pieces.ranges_of_grid(Tensors.grid_of(shape)) : Pieces.Ranges}

def Normal(~Element: Data,tensor: Tensors.Tensor<Element>) -> Data:
  match tensor:
    case Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}:
      KernelGemm.Both<Content.WellFormed<Element,ShapeLaws.elements(shape),Model.contents(~Element,pieces)>,
        KernelGemm.Both<Counted<shape>,Canonical(~Element,shape,Model.contents(~Element,pieces))>>
    case Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers}: Unit

def LogicalValues(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,tensor: Tensors.Tensor<Element>) -> Data:
  match tensor:
    case Tensors.Tensor{pieces,shape,fill,status,workers}: Content.Values(~Element,~Logical,~original,logical,Model.contents(~Element,pieces))

# A Normal tensor whose cells hold the reader's values at their flat index.
def Contract(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,-tensor: Tensors.Tensor<Element>) -> Data:
  KernelGemm.Both<Normal(~Element,tensor),LogicalValues(~Element,~Logical,~original,logical,tensor)>

# Well formed canonical pieces of the cells of a counted shape that hold the
# reader's values make a tensor of the contract.
def contract(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  -pieces: Pieces.Pieces(~Element),counted: Counted<shape>,canonical: Canonical(~Element,shape,Model.contents(~Element,pieces)),
  evidence: KernelGemm.Both<Content.Values(~Element,~Logical,~original,logical,Model.contents(~Element,pieces)),Content.WellFormed<Element,ShapeLaws.elements(shape),Model.contents(~Element,pieces)>>) ->
  Contract(~Element,~Logical,~original,logical,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}):
  KernelGemm.Both{values,well_formed} = evidence
  KernelGemm.Both{KernelGemm.Both{well_formed,KernelGemm.Both{counted,canonical}},values}

# A tensor of the contract holds the values of another reader at a related
# context when the two readers agree at related contexts.
def relabeled(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~after_value: After -> U32 -> Element,~Related: Before -> After -> Data,
  ~agreement: Content.Agreement(~Element,~Before,~After,~before_value,~after_value,~Related),+before: Before,+after: After,+related: Related(before,after),
  tensor: Tensors.Tensor<Element>,contract: Contract(~Element,~Before,~before_value,before,tensor)) ->
  Contract(~Element,~After,~after_value,after,tensor):
  match tensor:
    case Tensors.Tensor{pieces,shape,fill,status,workers}:
      KernelGemm.Both{normal,values} = contract
      KernelGemm.Both{normal,
        Content.values(~Element,~Before,~After,~before_value,~after_value,~Related,~agreement,before,after,related,Model.contents(~Element,pieces),values)}

# A worker count leaves the contract of a tensor as it is.
def with_workers(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,tensor: Tensors.Tensor<Element>,+count: U32,
  accepted: {Contract.tensor_ready(~Element,Tensors.with_workers(~Element,tensor,count)) == True{} : Bool},
  contract: @ready: {Contract.tensor_ready(~Element,tensor) == True{} : Bool} -> Contract(~Element,~Logical,~original,logical,tensor)) ->
  Contract(~Element,~Logical,~original,logical,Tensors.with_workers(~Element,tensor,count)):
  match tensor:
    case Tensors.Tensor{pieces,+shape,+fill,status,+workers}:
      match status:
        case Tensors.Usable{}: contract(accepted)
        case Tensors.Failed{+problem}: Empty.absurd(Contract(~Element,~Logical,~original,logical,Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},U32.max(1,count)}),WordEncoding.false_true(accepted))

# A ready tensor is a usable tensor of the pieces of a model.
type Usable<-Element: Data,-shape: List<&2,U32>,-tensor: Tensors.Tensor<Element>> is Data:
  Usable{pieces: Model.Model<Element>,fill: Element,workers: U32,
    equation: {Tensors.Tensor{Model.pack(~Element,pieces),shape,fill,Tensors.Usable{},workers} == tensor : Tensors.Tensor<Element>}}

def usable_status(~Element: Data,+model: Model.Model<Element>,-pieces: Pieces.Pieces(~Element),+shape: List<&2,U32>,+fill: Element,status: Tensors.Status,+workers: U32,
  equation: {Model.pack(~Element,model) == pieces : Pieces.Pieces(~Element)},
  ready: {Contract.tensor_ready(~Element,Tensors.Tensor{pieces,shape,fill,status,workers}) == True{} : Bool}) ->
  Usable<Element,shape,Tensors.Tensor{pieces,shape,fill,status,workers}>:
  match status:
    case Tensors.Usable{}:
      Usable{model,fill,workers,
        Equal.cong(Pieces.Pieces(~Element),Tensors.Tensor<Element>,owner => Tensors.Tensor{owner,shape,fill,Tensors.Usable{},workers},Model.pack(~Element,model),pieces,equation)}
    case Tensors.Failed{problem}:
      Empty.absurd(Usable<Element,shape,Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers}>,WordEncoding.false_true(ready))

def usable_observed(~Element: Data,-pieces: Pieces.Pieces(~Element),+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,
  observation: Model.Observation<Element,pieces>,ready: {Contract.tensor_ready(~Element,Tensors.Tensor{pieces,shape,fill,status,workers}) == True{} : Bool}) ->
  Usable<Element,shape,Tensors.Tensor{pieces,shape,fill,status,workers}>:
  Model.Observation{+model,equation} = observation
  usable_status(~Element,model,pieces,shape,fill,status,workers,equation,ready)

def usable(~Element: Data,tensor: Tensors.Tensor<Element>,ready: {Contract.tensor_ready(~Element,tensor) == True{} : Bool}) -> Usable<Element,Contract.extents(~Element,tensor),tensor>:
  Tensors.Tensor{pieces,+shape,+fill,+status,+workers} = tensor
  usable_observed(~Element,pieces,shape,fill,status,workers,Model.observe(~Element,pieces),ready)

# A Ready tensor of the pieces of a model in a shape, of the contract of a
# reader.
def Shaped(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,-tensor: Tensors.Tensor<Element>) -> Data:
  KernelGemm.Both<Usable<Element,shape,tensor>,Contract(~Element,~Logical,~original,logical,tensor)>

def shaped(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,tensor: Tensors.Tensor<Element>,
  ready: {Contract.tensor_ready(~Element,tensor) == True{} : Bool},extents: {Contract.extents(~Element,tensor) == shape : List<&2,U32>},
  contract: Contract(~Element,~Logical,~original,logical,tensor)) -> Shaped(~Element,~Logical,~original,logical,shape,tensor):
  KernelGemm.Both{%extents : Usable<Element,_,tensor> usable(~Element,tensor,ready),contract}

# The contract of a Shaped tensor.
def contract_of(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,-tensor: Tensors.Tensor<Element>,
  shaped: Shaped(~Element,~Logical,~original,logical,shape,tensor)) -> Contract(~Element,~Logical,~original,logical,tensor):
  KernelGemm.Both{usable,contract} = shaped
  contract

# Two contexts, each the given one.
def Fixed(~Before: Data,~After: Data,+before: Before,+after: After,left: Before,right: After) -> Data:
  KernelGemm.Both<{left == before : Before},{right == after : After}>

def fixed_agrees(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~after_value: After -> U32 -> Element,+before: Before,+after: After,
  agrees: @+index: U32 -> {before_value(before,index) == after_value(after,index) : Element},
  +left: Before,+right: After,+related: Fixed(~Before,~After,before,after,left,right),+index: U32) ->
  {before_value(left,index) == after_value(right,index) : Element}:
  KernelGemm.Both{+left_equation,+right_equation} = related
  %Equal.sym(Before,left,before,left_equation) : {before_value(_,index) == after_value(right,index) : Element}
  %Equal.sym(After,right,after,right_equation) : {before_value(before,index) == after_value(_,index) : Element}
  agrees(index)

# A Shaped tensor of a reader at a context is a Shaped tensor of another
# reader at a context that agrees with it at every index.
def restated(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~after_value: After -> U32 -> Element,+before: Before,+after: After,
  agrees: @+index: U32 -> {before_value(before,index) == after_value(after,index) : Element},+shape: List<&2,U32>,-tensor: Tensors.Tensor<Element>,
  shaped: Shaped(~Element,~Before,~before_value,before,shape,tensor)) -> Shaped(~Element,~After,~after_value,after,shape,tensor):
  KernelGemm.Both{usable,contract} = shaped
  Usable{+pieces,+fill,+workers,+equation} = usable
  %equation : Shaped(~Element,~After,~after_value,after,shape,_)
  KernelGemm.Both{Usable{pieces,fill,workers,{==}},
    relabeled(~Element,~Before,~After,~before_value,~after_value,~(left => right => Fixed(~Before,~After,before,after,left,right)),
      ~(left => right => related => index => fixed_agrees(~Element,~Before,~After,~before_value,~after_value,before,after,agrees,left,right,related,index)),
      before,after,KernelGemm.Both{{==},{==}},Tensors.Tensor{Model.pack(~Element,pieces),shape,fill,Tensors.Usable{},workers},
      %Equal.sym(Tensors.Tensor<Element>,Tensors.Tensor{Model.pack(~Element,pieces),shape,fill,Tensors.Usable{},workers},tensor,equation) :
        Contract(~Element,~Before,~before_value,before,_)
      contract)}

# The count of a counted shape.
def whole_width(+shape: List<&2,U32>,+count: U32,accepted: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>}) -> {Tensors.count_or_zero(shape) == count : U32}:
  Equal.cong(Maybe<&2,U32>,U32,Tensors.known_count,Tensors.count_of(shape),Some{count},accepted)

def whole_counted(+shape: List<&2,U32>,+count: U32,accepted: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>}) ->
  {ShapeLaws.elements(shape) == U32.to_nat(count) : Nat}:
  Count.count_exact(ShapeLaws.elements(shape),count,Count.count_of(shape,count,accepted))

def count(+shape: List<&2,U32>,evidence: Counted<shape>) -> U32:
  Counted{count,accepted} = evidence
  count

def accepted(+shape: List<&2,U32>,+counted: Counted<shape>) -> {Tensors.count_of(shape) == Some{count(shape,counted)} : Maybe<&2,U32>}:
  Counted{+count,+accepted} = counted
  accepted

def evidence(+shape: List<&2,U32>,+counted: Counted<shape>) -> Count.Count<ShapeLaws.elements(shape),count(shape,counted)>:
  Counted{+count,accepted} = counted
  Count.count_of(shape,count,accepted)

# A tensor of the contract holds the values of another reader that agrees
# with it below a bound of at least the elements of its shape.
def relabeled_held(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~after_value: After -> U32 -> Element,~Related: Before -> After -> Data,
  ~bound: Nat,~agreement: Content.Below(~Element,~Before,~After,~before_value,~after_value,~Related,~bound),+before: Before,+after: After,+related: Related(before,after),
  +pieces: Model.Model<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Contract(~Element,~Before,~before_value,before,Tensors.Tensor{Model.pack(~Element,pieces),shape,fill,Tensors.Usable{},workers}),
  +room: {Nat.is_le(ShapeLaws.elements(shape),bound) == True{} : Bool},+word: U32,+fits: {Nat.is_le(bound,U32.to_nat(word)) == True{} : Bool}) ->
  Contract(~Element,~After,~after_value,after,Tensors.Tensor{Model.pack(~Element,pieces),shape,fill,Tensors.Usable{},workers}):
  KernelGemm.Both{KernelGemm.Both{Content.WellFormed{+fields,+ready,+coverage},+laid},values} = contract
  KernelGemm.Both{KernelGemm.Both{Content.WellFormed{fields,ready,coverage},laid},
    Content.values_below(~Element,~Before,~After,~before_value,~after_value,~Related,~bound,~agreement,before,after,related,ShapeLaws.elements(shape),
      Model.contents(~Element,Model.pack(~Element,pieces)),fields,room,word,fits,values)}

def relabeled_below(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~after_value: After -> U32 -> Element,~Related: Before -> After -> Data,
  ~bound: Nat,~agreement: Content.Below(~Element,~Before,~After,~before_value,~after_value,~Related,~bound),+before: Before,+after: After,+related: Related(before,after),
  +shape: List<&2,U32>,-tensor: Tensors.Tensor<Element>,usable: Usable<Element,shape,tensor>,contract: Contract(~Element,~Before,~before_value,before,tensor),
  +room: {Nat.is_le(ShapeLaws.elements(shape),bound) == True{} : Bool},+word: U32,+fits: {Nat.is_le(bound,U32.to_nat(word)) == True{} : Bool}) ->
  Contract(~Element,~After,~after_value,after,tensor):
  Usable{+pieces,+fill,+workers,+equation} = usable
  %equation : Contract(~Element,~After,~after_value,after,_)
  relabeled_held(~Element,~Before,~After,~before_value,~after_value,~Related,~bound,~agreement,before,after,related,pieces,shape,fill,workers,
    %Equal.sym(Tensors.Tensor<Element>,Tensors.Tensor{Model.pack(~Element,pieces),shape,fill,Tensors.Usable{},workers},tensor,equation) :
      Contract(~Element,~Before,~before_value,before,_) contract,room,word,fits)

def fixed_below(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~after_value: After -> U32 -> Element,~bound: Nat,+before: Before,+after: After,
  agrees: @+index: U32 -> @+inside: {Nat.is_lt(U32.to_nat(index),bound) == True{} : Bool} -> {before_value(before,index) == after_value(after,index) : Element},
  +left: Before,+right: After,+related: Fixed(~Before,~After,before,after,left,right),+index: U32,+inside: {Nat.is_lt(U32.to_nat(index),bound) == True{} : Bool}) ->
  {before_value(left,index) == after_value(right,index) : Element}:
  KernelGemm.Both{+left_equation,+right_equation} = related
  %Equal.sym(Before,left,before,left_equation) : {before_value(_,index) == after_value(right,index) : Element}
  %Equal.sym(After,right,after,right_equation) : {before_value(before,index) == after_value(_,index) : Element}
  agrees(index,inside)

# The same for readers that agree below a bound of at least the elements of
# the shape.
def restated_below(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~after_value: After -> U32 -> Element,~bound: Nat,+before: Before,+after: After,
  agrees: @+index: U32 -> @+inside: {Nat.is_lt(U32.to_nat(index),bound) == True{} : Bool} -> {before_value(before,index) == after_value(after,index) : Element},
  +shape: List<&2,U32>,-tensor: Tensors.Tensor<Element>,shaped: Shaped(~Element,~Before,~before_value,before,shape,tensor),
  +room: {Nat.is_le(ShapeLaws.elements(shape),bound) == True{} : Bool},+word: U32,+fits: {Nat.is_le(bound,U32.to_nat(word)) == True{} : Bool}) ->
  Shaped(~Element,~After,~after_value,after,shape,tensor):
  KernelGemm.Both{+usable,contract} = shaped
  KernelGemm.Both{usable,
    relabeled_below(~Element,~Before,~After,~before_value,~after_value,~(left => right => Fixed(~Before,~After,before,after,left,right)),~bound,
      ~(left => right => related => index => inside => fixed_below(~Element,~Before,~After,~before_value,~after_value,~bound,before,after,agrees,left,right,related,index,inside)),
      before,after,KernelGemm.Both{{==},{==}},shape,tensor,usable,contract,room,word,fits)}

# ---- The count of a Shaped tensor ----

def contract_counted(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,-pieces: Pieces.Pieces(~Element),+shape: List<&2,U32>,+fill: Element,+workers: U32,
  +contract: Contract(~Element,~Logical,~original,logical,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers})) -> Counted<shape>:
  KernelGemm.Both{normal,values} = contract
  KernelGemm.Both{well_formed,rest} = normal
  KernelGemm.Both{counted,canonical} = rest
  counted

# The shape of a Shaped tensor is counted.
def shaped_counted(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,-tensor: Tensors.Tensor<Element>,
  +shaped: Shaped(~Element,~Logical,~original,logical,shape,tensor)) -> Counted<shape>:
  KernelGemm.Both{Usable{+pieces,+fill,+workers,+equation},contract} = shaped
  contract_counted(~Element,~Logical,~original,logical,Model.pack(~Element,pieces),shape,fill,workers,
    %Equal.sym(Tensors.Tensor<Element>,Tensors.Tensor{Model.pack(~Element,pieces),shape,fill,Tensors.Usable{},workers},tensor,equation) : Contract(~Element,~Logical,~original,logical,_) contract)
