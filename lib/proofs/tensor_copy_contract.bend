# Copy returns two tensors with the same ordered logical content: the storage
# of every band is cloned.
import Base
import ../tensor.bend as Tensors
import ../tensor_layout.bend as Layout
import ../tensor_band.bend as Bands
import ../storage_buffer.bend as Storage
import ../storage_clone.bend as Clone
import ./storage_clone_refinement.bend as CloneLaws
import ./tensor_contract.bend as Contract
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./tensor_band_model.bend as Model
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_values.bend as BandValues
import ./tensor_global_shape_count.bend as Global
import ../traversal_partition.bend as Partition

def pair_ready(~Element: Data,pair: Tensors.Tensor<Element> & Tensors.Tensor<Element>) -> Bool:
  (left,right) = pair
  Contract.tensor_ready(~Element,right)

# The two copies of the bands of a model are the bands of two models of its
# content.
type Copied<-Element: Data,-model: Model.Model<Element>> is Data:
  Copied{left: Model.Model<Element>,right: Model.Model<Element>,
    equation: {(Model.pack(~Element,left),Model.pack(~Element,right)) == Bands.copy(~Element,Model.pack(~Element,model)) : Bands.Bands(~Element) & Bands.Bands(~Element)},
    left_content: {Model.content(~Element,left) == Model.content(~Element,model) : Model.Content<Element>},
    right_content: {Model.content(~Element,right) == Model.content(~Element,model) : Model.Content<Element>}}

def leaf_copied(~Element: Data,+buffer: Buffer.Model<Element>,+first: U32,+rows: U32,
  observation: CloneLaws.Observation<Element,Clone.clone(~Element,Buffer.pack(~Element,buffer))>) -> Copied<Element,Model.Leaf{buffer,first,rows}>:
  CloneLaws.Observation{+left,+right,+equation} = observation
  Copied{Model.Leaf{left,first,rows},Model.Leaf{right,first,rows},
    Equal.cong(Storage.Buffer<Element> & Storage.Buffer<Element>,Bands.Bands(~Element) & Bands.Bands(~Element),owners => Bands.copied_leaf(~Element,first,rows,owners),
      (Buffer.pack(~Element,left),Buffer.pack(~Element,right)),Clone.clone(~Element,Buffer.pack(~Element,buffer)),equation),
    Equal.cong(Buffer.Content<Element>,Model.Content<Element>,content => Model.RowBand{content,first,rows},Buffer.content(~Element,left),Buffer.content(~Element,buffer),
      CloneLaws.left_contents(~Element,buffer,left,right,equation)),
    Equal.cong(Buffer.Content<Element>,Model.Content<Element>,content => Model.RowBand{content,first,rows},Buffer.content(~Element,right),Buffer.content(~Element,buffer),
      CloneLaws.right_contents(~Element,buffer,left,right,equation))}

def fork_copied(~Element: Data,+left: Model.Model<Element>,+right: Model.Model<Element>,left_copies: Copied<Element,left>,right_copies: Copied<Element,right>) ->
  Copied<Element,Model.Fork{left,right}>:
  Copied{+left_first,+left_second,+left_equation,+left_first_content,+left_second_content} = left_copies
  Copied{+right_first,+right_second,+right_equation,+right_first_content,+right_second_content} = right_copies
  Copied{Model.Fork{left_first,right_first},Model.Fork{left_second,right_second},
    %left_equation : {(Partition.Fork{Model.pack(~Element,left_first),Model.pack(~Element,right_first)},Partition.Fork{Model.pack(~Element,left_second),Model.pack(~Element,right_second)}) ==
      Bands.reduced(~Storage.Buffer<Element>,~Bands.Bands(~Element),~Bands.fork(~Element),_,Bands.copy(~Element,Model.pack(~Element,right))) : Bands.Bands(~Element) & Bands.Bands(~Element)}
    %right_equation : {(Partition.Fork{Model.pack(~Element,left_first),Model.pack(~Element,right_first)},Partition.Fork{Model.pack(~Element,left_second),Model.pack(~Element,right_second)}) ==
      Bands.reduced(~Storage.Buffer<Element>,~Bands.Bands(~Element),~Bands.fork(~Element),(Model.pack(~Element,left_first),Model.pack(~Element,left_second)),_) : Bands.Bands(~Element) & Bands.Bands(~Element)}
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

# A copy of a model's content keeps the contract of its tensor.
def kept(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+model: Model.Model<Element>,+copy: Model.Model<Element>,
  +geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,content: {Model.content(~Element,copy) == Model.content(~Element,model) : Model.Content<Element>},
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,model),geometry,shape,fill,Tensors.Usable{},workers})) ->
  Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,copy),geometry,shape,fill,Tensors.Usable{},workers}):
  Shape.Both{Shape.Both{well_formed,Shape.Both{counted,coherent}},values} = contract
  %Equal.trans(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,model)),Model.content(~Element,model),Model.contents(~Element,Model.pack(~Element,copy)),
      Model.packed_contents(~Element,model),
      Equal.trans(Model.Content<Element>,Model.content(~Element,model),Model.content(~Element,copy),Model.contents(~Element,Model.pack(~Element,copy)),
        Equal.sym(Model.Content<Element>,Model.content(~Element,copy),Model.content(~Element,model),content),
        Equal.sym(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,copy)),Model.content(~Element,copy),Model.packed_contents(~Element,copy)))) :
    Shape.Both<Shape.Both<BandShape.WellFormed<Element,geometry,_>,Shape.Both<Global.Counted<shape>,Global.Coherent(shape,geometry)>>,BandValues.Values(~Element,~Logical,~original,logical,geometry,_)>
  Shape.Both{Shape.Both{well_formed,Shape.Both{counted,coherent}},values}

def PairContract(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,pair: Tensors.Tensor<Element> & Tensors.Tensor<Element>) -> Data:
  (left,right) = pair
  Shape.Both<Global.Contract(~Element,~Logical,~original,logical,left),Global.Contract(~Element,~Logical,~original,logical,right)>

def copies(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+model: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  evidence: Copied<Element,model>,+contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,model),geometry,shape,fill,Tensors.Usable{},workers})) ->
  PairContract(~Element,~Logical,~original,logical,Layout.copy(~Element,Tensors.Tensor{Model.pack(~Element,model),geometry,shape,fill,Tensors.Usable{},workers})):
  Copied{+left,+right,+equation,+left_content,+right_content} = evidence
  %equation : PairContract(~Element,~Logical,~original,logical,Storage.bind_pair(~Bands.Bands(~Element),~Bands.Bands(~Element),~(Tensors.Tensor<Element> & Tensors.Tensor<Element>),_,
    first => second => (Tensors.Tensor{first,geometry,shape,fill,Tensors.Usable{},workers},Tensors.Tensor{second,geometry,shape,fill,Tensors.Usable{},workers})))
  Shape.Both{kept(~Element,~Logical,~original,logical,model,left,geometry,shape,fill,workers,left_content,contract),
    kept(~Element,~Logical,~original,logical,model,right,geometry,shape,fill,workers,right_content,contract)}

def observed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,-bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  observation: Model.Observation<Element,bands>,contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers})) ->
  PairContract(~Element,~Logical,~original,logical,Layout.copy(~Element,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers})):
  Model.Observation{+model,+equation} = observation
  %equation : PairContract(~Element,~Logical,~original,logical,Layout.copy(~Element,Tensors.Tensor{_,geometry,shape,fill,Tensors.Usable{},workers}))
  copies(~Element,~Logical,~original,logical,model,geometry,shape,fill,workers,copied(~Element,model),
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,model),bands,equation) : Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{_,geometry,shape,fill,Tensors.Usable{},workers}) contract)

# Both results of copy hold the values of the copied tensor.
def copy(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers})) ->
  PairContract(~Element,~Logical,~original,logical,Layout.copy(~Element,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers})):
  observed(~Element,~Logical,~original,logical,bands,geometry,shape,fill,workers,Model.observe(~Element,bands),contract)
