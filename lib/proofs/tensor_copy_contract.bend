# Copy returns two tensors with the contents of the copied one: every piece is
# cloned. A tiled tensor of the contract copies into two of the contract.
import Base
import ../kernel_tile.bend as Tile
import ../kernel_gemm.bend as KernelGemm
import ../storage_buffer.bend as Storage
import ../storage_clone.bend as Clone
import ../traversal_partition.bend as Partition
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_pieces.bend as Pieces
import ../tensor_tasks.bend as Tasks
import ../tensor_tiled.bend as Tiled
import ./storage_buffer_model.bend as Buffer
import ./storage_clone_refinement.bend as CloneLaws
import ./tensor_pieces_model.bend as Model
import ./tensor_pieces_each.bend as Each
import ./tensor_pieces_content.bend as Content
import ./tensor_global_shape_count.bend as Global
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_tiled_public.bend as Public
import ./tensor_tiled_representation.bend as Representation

def Owners(~Element: Data) -> Type:
  Storage.Buffer<Element> & Storage.Buffer<Element>

# The pairs of clones of every piece of a model, split into two trees.
def cloned(~Element: Data,+model: Model.Model<Element>) -> Pieces.Pieces(~Element) & Pieces.Pieces(~Element):
  Tensors.unzipped(~Element,Each.leaf_map(~Element,~Owners(~Element),~Unit,~Tensors.copy_leaf(~Element),Unit{},model))

# The copies of the pieces of a model are the pieces of two models of its
# content.
type Copied<-Element: Data,-model: Model.Model<Element>> is Data:
  Copied{first: Model.Model<Element>,second: Model.Model<Element>,
    equation: {(Model.pack(~Element,first),Model.pack(~Element,second)) == cloned(~Element,model) : Pieces.Pieces(~Element) & Pieces.Pieces(~Element)},
    first_content: {Model.content(~Element,first) == Model.content(~Element,model) : Model.Content<Element>},
    second_content: {Model.content(~Element,second) == Model.content(~Element,model) : Model.Content<Element>}}

def leaf_copied(~Element: Data,+buffer: Buffer.Model<Element>,+first: U32,+rows: U32,
  observation: CloneLaws.Observation<Element,Clone.clone(~Element,Buffer.pack(~Element,buffer))>) -> Copied<Element,Model.Leaf{buffer,first,rows}>:
  CloneLaws.Observation{+left,+right,+equation} = observation
  Copied{Model.Leaf{left,first,rows},Model.Leaf{right,first,rows},
    Equal.cong(Owners(~Element),Pieces.Pieces(~Element) & Pieces.Pieces(~Element),owners => Tensors.copy_pair(~Element,first,rows,owners),
      (Buffer.pack(~Element,left),Buffer.pack(~Element,right)),Clone.clone(~Element,Buffer.pack(~Element,buffer)),equation),
    Equal.cong(Buffer.Content<Element>,Model.Content<Element>,content => Model.PieceContent{content,first,rows},Buffer.content(~Element,left),Buffer.content(~Element,buffer),
      CloneLaws.left_contents(~Element,buffer,left,right,equation)),
    Equal.cong(Buffer.Content<Element>,Model.Content<Element>,content => Model.PieceContent{content,first,rows},Buffer.content(~Element,right),Buffer.content(~Element,buffer),
      CloneLaws.right_contents(~Element,buffer,left,right,equation))}

def fork_copied(~Element: Data,+left: Model.Model<Element>,+right: Model.Model<Element>,left_copies: Copied<Element,left>,right_copies: Copied<Element,right>) ->
  Copied<Element,Model.Fork{left,right}>:
  Copied{+left_first,+left_second,+left_equation,+left_first_content,+left_second_content} = left_copies
  Copied{+right_first,+right_second,+right_equation,+right_first_content,+right_second_content} = right_copies
  Copied{Model.Fork{left_first,right_first},Model.Fork{left_second,right_second},
    %left_equation : {(Partition.Fork{Model.pack(~Element,left_first),Model.pack(~Element,right_first)},Partition.Fork{Model.pack(~Element,left_second),Model.pack(~Element,right_second)}) ==
      Tensors.copy_halves(~Element,_,cloned(~Element,right)) : Pieces.Pieces(~Element) & Pieces.Pieces(~Element)}
    %right_equation : {(Partition.Fork{Model.pack(~Element,left_first),Model.pack(~Element,right_first)},Partition.Fork{Model.pack(~Element,left_second),Model.pack(~Element,right_second)}) ==
      Tensors.copy_halves(~Element,(Model.pack(~Element,left_first),Model.pack(~Element,left_second)),_) : Pieces.Pieces(~Element) & Pieces.Pieces(~Element)}
    {==},
    %Equal.sym(Model.Content<Element>,Model.content(~Element,left_first),Model.content(~Element,left),left_first_content) :
      {Model.Joined{_,Model.content(~Element,right_first)} == Model.Joined{Model.content(~Element,left),Model.content(~Element,right)} : Model.Content<Element>}
    %Equal.sym(Model.Content<Element>,Model.content(~Element,right_first),Model.content(~Element,right),right_first_content) :
      {Model.Joined{Model.content(~Element,left),_} == Model.Joined{Model.content(~Element,left),Model.content(~Element,right)} : Model.Content<Element>}
    {==},
    %Equal.sym(Model.Content<Element>,Model.content(~Element,left_second),Model.content(~Element,left),left_second_content) :
      {Model.Joined{_,Model.content(~Element,right_second)} == Model.Joined{Model.content(~Element,left),Model.content(~Element,right)} : Model.Content<Element>}
    %Equal.sym(Model.Content<Element>,Model.content(~Element,right_second),Model.content(~Element,right),right_second_content) :
      {Model.Joined{Model.content(~Element,left),_} == Model.Joined{Model.content(~Element,left),Model.content(~Element,right)} : Model.Content<Element>}
    {==}}

def copied(~Element: Data,model: Model.Model<Element>) -> Copied<Element,model>:
  match model:
    case Model.Leaf{+buffer,+first,+rows}: leaf_copied(~Element,buffer,first,rows,CloneLaws.observe_pair(~Element,Clone.clone(~Element,Buffer.pack(~Element,buffer))))
    case Model.Fork{+left,+right}: fork_copied(~Element,left,right,copied(~Element,left),copied(~Element,right))

# Whatever tasks copy them, the pieces of a model are cloned piece by piece.
def decided(~Element: Data,decision: Tasks.Decision,+model: Model.Model<Element>) ->
  {Tensors.copy_decided(~Element,decision,Model.pack(~Element,model)) == cloned(~Element,model) : Pieces.Pieces(~Element) & Pieces.Pieces(~Element)}:
  match decision:
    case Tasks.Decision{+parallel,+tasks}:
      Equal.cong(Each.Tree(~Owners(~Element)),Pieces.Pieces(~Element) & Pieces.Pieces(~Element),tree => Tensors.unzipped(~Element,tree),
        Pieces.grouped(~Storage.Buffer<Element>,~Owners(~Element),~Unit,~Tensors.copy_leaf(~Element),tasks,parallel,Unit{},Model.pack(~Element,model)),
        Each.leaf_map(~Element,~Owners(~Element),~Unit,~Tensors.copy_leaf(~Element),Unit{},model),
        Each.grouped(~Element,~Owners(~Element),~Unit,~Tensors.copy_leaf(~Element),tasks,parallel,Unit{},model))

# Pieces of the content of a model keep the contract of its tensor.
def kept(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+model: Model.Model<Element>,+copy: Model.Model<Element>,
  +shape: List<&2,U32>,+fill: Element,+workers: U32,content: {Model.content(~Element,copy) == Model.content(~Element,model) : Model.Content<Element>},
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,model),shape,fill,Tensors.Usable{},workers})) ->
  Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,copy),shape,fill,Tensors.Usable{},workers}):
  %Equal.trans(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,model)),Model.content(~Element,model),Model.contents(~Element,Model.pack(~Element,copy)),
      Model.packed_contents(~Element,model),
      Equal.trans(Model.Content<Element>,Model.content(~Element,model),Model.content(~Element,copy),Model.contents(~Element,Model.pack(~Element,copy)),
        Equal.sym(Model.Content<Element>,Model.content(~Element,copy),Model.content(~Element,model),content),
        Equal.sym(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,copy)),Model.content(~Element,copy),Model.packed_contents(~Element,copy)))) :
    KernelGemm.Both<KernelGemm.Both<Content.WellFormed<Element,ShapeLaws.elements(shape),_>,KernelGemm.Both<Global.Counted<shape>,Global.Canonical(~Element,shape,_)>>,
      Content.Values(~Element,~Logical,~original,logical,_)>
  contract


# The copy of the pieces of a model is the two tensors of the copies.
def copied_equation(~Element: Data,+bytes_per_element: F32,+model: Model.Model<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  +first: Model.Model<Element>,+second: Model.Model<Element>,
  +equation: {(Model.pack(~Element,first),Model.pack(~Element,second)) == cloned(~Element,model) : Pieces.Pieces(~Element) & Pieces.Pieces(~Element)}) ->
  {Tensors.copied(~Element,bytes_per_element,Tensors.Tensor{Model.pack(~Element,model),shape,fill,Tensors.Usable{},workers}) ==
    (Tensors.Tensor{Model.pack(~Element,first),shape,fill,Tensors.Usable{},workers},Tensors.Tensor{Model.pack(~Element,second),shape,fill,Tensors.Usable{},workers}) :
    Tensors.Tensor<Element> & Tensors.Tensor<Element>}:
  +decision = Tasks.decide_copy(Tensors.grid_of(shape),workers,bytes_per_element)
  %Equal.sym(Pieces.Pieces(~Element) & Pieces.Pieces(~Element),Tensors.copy_decided(~Element,decision,Model.pack(~Element,model)),cloned(~Element,model),
      decided(~Element,decision,model)) :
    {Tensors.copies_of(~Element,shape,fill,workers,_) ==
      (Tensors.Tensor{Model.pack(~Element,first),shape,fill,Tensors.Usable{},workers},Tensors.Tensor{Model.pack(~Element,second),shape,fill,Tensors.Usable{},workers}) :
      Tensors.Tensor<Element> & Tensors.Tensor<Element>}
  %equation :
    {Tensors.copies_of(~Element,shape,fill,workers,_) ==
      (Tensors.Tensor{Model.pack(~Element,first),shape,fill,Tensors.Usable{},workers},Tensors.Tensor{Model.pack(~Element,second),shape,fill,Tensors.Usable{},workers}) :
      Tensors.Tensor<Element> & Tensors.Tensor<Element>}
  {==}

def PairContract(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,pair: Tensors.Tensor<Element> & Tensors.Tensor<Element>) -> Data:
  (left,right) = pair
  KernelGemm.Both<Global.Contract(~Element,~Logical,~original,logical,left),Global.Contract(~Element,~Logical,~original,logical,right)>

def copies(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+bytes_per_element: F32,+logical: Logical,+model: Model.Model<Element>,+shape: List<&2,U32>,
  +fill: Element,+workers: U32,evidence: Copied<Element,model>,
  +contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,model),shape,fill,Tensors.Usable{},workers})) ->
  PairContract(~Element,~Logical,~original,logical,Tensors.copied(~Element,bytes_per_element,Tensors.Tensor{Model.pack(~Element,model),shape,fill,Tensors.Usable{},workers})):
  Copied{+first,+second,+equation,+first_content,+second_content} = evidence
  %Equal.sym(Tensors.Tensor<Element> & Tensors.Tensor<Element>,
      Tensors.copied(~Element,bytes_per_element,Tensors.Tensor{Model.pack(~Element,model),shape,fill,Tensors.Usable{},workers}),
      (Tensors.Tensor{Model.pack(~Element,first),shape,fill,Tensors.Usable{},workers},Tensors.Tensor{Model.pack(~Element,second),shape,fill,Tensors.Usable{},workers}),
      copied_equation(~Element,bytes_per_element,model,shape,fill,workers,first,second,equation)) :
    PairContract(~Element,~Logical,~original,logical,_)
  KernelGemm.Both{kept(~Element,~Logical,~original,logical,model,first,shape,fill,workers,first_content,contract),
    kept(~Element,~Logical,~original,logical,model,second,shape,fill,workers,second_content,contract)}

def observed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+bytes_per_element: F32,+logical: Logical,-pieces: Pieces.Pieces(~Element),+shape: List<&2,U32>,
  +fill: Element,+workers: U32,observation: Model.Observation<Element,pieces>,
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers})) ->
  PairContract(~Element,~Logical,~original,logical,Tensors.copied(~Element,bytes_per_element,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers})):
  Model.Observation{+model,+equation} = observation
  %equation : PairContract(~Element,~Logical,~original,logical,Tensors.copied(~Element,bytes_per_element,Tensors.Tensor{_,shape,fill,Tensors.Usable{},workers}))
  copies(~Element,~Logical,~original,bytes_per_element,logical,model,shape,fill,workers,copied(~Element,model),
    %Equal.sym(Pieces.Pieces(~Element),Model.pack(~Element,model),pieces,equation) :
      Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{_,shape,fill,Tensors.Usable{},workers}) contract)

# Both results of copy hold the values of the copied tensor.
def copy(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+bytes_per_element: F32,+logical: Logical,pieces: Pieces.Pieces(~Element),+shape: List<&2,U32>,
  +fill: Element,+workers: U32,contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers})) ->
  PairContract(~Element,~Logical,~original,logical,Tensors.copied(~Element,bytes_per_element,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers})):
  observed(~Element,~Logical,~original,bytes_per_element,logical,pieces,shape,fill,workers,Model.observe(~Element,pieces),contract)

# ---- Tiled tensors ----

def Copies(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,pair: Tiled.Tiled & Tiled.Tiled) -> Data:
  (first,second) = pair
  KernelGemm.Both<Public.Contract(~Logical,~original,logical,values,first),Public.Contract(~Logical,~original,logical,values,second)>

# What holds of two tiled tensors of the contract holds of the copies of
# packed cells of the contract.
def copies_then(~Logical: Data,~original: Logical -> U32 -> F32,~Goal: Tiled.Tiled & Tiled.Tiled -> Data,+logical: Logical,+values: List<&2,U32>,
  +model: Model.Model<Tile.Tile8>,+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+axis: Nat,+lanes: U32,+position: Grid.Position,evidence: Copied<Tile.Tile8,model>,
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},axis,lanes,position}),
  body: @first: Tiled.Tiled -> @second: Tiled.Tiled -> @+first_contract: Public.Contract(~Logical,~original,logical,values,first) ->
    @+second_contract: Public.Contract(~Logical,~original,logical,values,second) -> Goal((first,second))) ->
  Goal(Tiled.copy(Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},axis,lanes,position})):
  Copied{+first,+second,+equation,+first_content,+second_content} = evidence
  KernelGemm.Both{+global,+laid} = contract
  %Equal.sym(Tensors.Tensor<Tile.Tile8> & Tensors.Tensor<Tile.Tile8>,
      Tensors.copied(~Tile.Tile8,32.0,Tensors.Tensor{Model.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers}),
      (Tensors.Tensor{Model.pack(~Tile.Tile8,first),shape,fill,Tensors.Usable{},workers},Tensors.Tensor{Model.pack(~Tile.Tile8,second),shape,fill,Tensors.Usable{},workers}),
      copied_equation(~Tile.Tile8,32.0,model,shape,fill,workers,first,second,equation)) :
    Goal(Tiled.copied(axis,lanes,position,_))
  +before = {Representation.Lifted{logical,Public.geometry_of(values,axis,position)} : Representation.Lifted<Logical>}
  body(Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,first),shape,fill,Tensors.Usable{},workers},axis,lanes,position},
    Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,second),shape,fill,Tensors.Usable{},workers},axis,lanes,position},
    KernelGemm.Both{kept(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),before,model,first,shape,fill,workers,first_content,global),laid},
    KernelGemm.Both{kept(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),before,model,second,shape,fill,workers,second_content,global),laid})

# What holds of two tiled tensors of the contract holds of the copies of a
# tiled tensor of the contract.
def tiled_copied(~Logical: Data,~original: Logical -> U32 -> F32,~Goal: Tiled.Tiled & Tiled.Tiled -> Data,+logical: Logical,+values: List<&2,U32>,tiled: Tiled.Tiled,
  +contract: Public.Contract(~Logical,~original,logical,values,tiled),
  body: @first: Tiled.Tiled -> @second: Tiled.Tiled -> @+first_contract: Public.Contract(~Logical,~original,logical,values,first) ->
    @+second_contract: Public.Contract(~Logical,~original,logical,values,second) -> Goal((first,second))) -> Goal(Tiled.copy(tiled)):
  Public.by_packed(~Logical,~original,(that => Goal(Tiled.copy(that))),logical,values,tiled,contract,
    model => shape => fill => workers => axis => lanes => position => packed =>
      copies_then(~Logical,~original,~Goal,logical,values,model,shape,fill,workers,axis,lanes,position,copied(~Tile.Tile8,model),packed,body))

# Both copies of a tiled tensor of the contract are tensors of the contract.
def tiled(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,tiled: Tiled.Tiled,+contract: Public.Contract(~Logical,~original,logical,values,tiled)) ->
  Copies(~Logical,~original,logical,values,Tiled.copy(tiled)):
  tiled_copied(~Logical,~original,~Copies(~Logical,~original,logical,values),logical,values,tiled,contract,
    first => second => first_contract => second_contract => KernelGemm.Both{first_contract,second_contract})
