# The exits read a tensor of the contract of a reader, in any layout, as the
# reader's values in flat index order: into_storage gives storage that holds
# them, to_list their list, item the one value, and save and save_npy hand
# that storage to the writer and return a tensor of the same contract.
import Base
import ../tensor.bend as Tensors
import ../tensor_band.bend as Bands
import ../tensor_file.bend as Files
import ../storage_buffer.bend as Storage
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./storage_shape.bend as Shape
import ./tensor_band_model.bend as Model
import ./tensor_contract.bend as Contract
import ./tensor_whole_leaf.bend as WholeLeaf
import ./tensor_gather_values.bend as GatherValues
import ./tensor_global_shape_count.bend as Global
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_copy_contract.bend as CopyContract
import ./nat_order.bend as Order

# Storage that holds the reader's values in the shape's element count.
type Stored<-Element: Data,-Context: Data,-reader: Context -> U32 -> Element,-context: Context,-shape: List<&2,U32>,
  -result: Result<&2,&1,Tensors.Problem,Storage.Buffer<Element>>> is Data:
  Stored{model: Buffer.Model<Element>,equation: {result == Done{Buffer.pack(~Element,model)} : Result<&2,&1,Tensors.Problem,Storage.Buffer<Element>>},
    facts: GatherValues.Facts(~Element,~Context,~reader,context,shape,Buffer.content(~Element,model))}

def StoredResult(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,+shape: List<&2,U32>,
  result: Result<&2,&1,Tensors.Problem,Storage.Buffer<Element>>) -> Type:
  Stored<Element,Context,reader,context,shape,result>

# into_storage of a tensor of the contract gives storage that holds the
# reader's values.
def into_storage(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,
  +bands: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Global.Contract(~Element,~Context,~reader,context,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})) ->
  Stored<Element,Context,reader,context,shape,Tensors.into_storage(~Element,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})>:
  WholeLeaf.with_whole(~Element,~Context,~reader,~Result<&2,&1,Tensors.Problem,Storage.Buffer<Element>>,
    ~StoredResult(~Element,~Context,~reader,context,shape),
    ~(storage => shape => fill => status => workers => Tensors.released(~Element,status,storage)),
    context,bands,geometry,shape,fill,workers,contract,
    model => facts => Stored{model,{==},facts})

# Storage that holds the reader's values in the shape's element count reads
# as their list.
def listed(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,+shape: List<&2,U32>,+model: Buffer.Model<Element>,
  +facts: GatherValues.Facts(~Element,~Context,~reader,context,shape,Buffer.content(~Element,model))) ->
  {Tensors.listed_buffer(~Element,Tensors.Usable{},Buffer.pack(~Element,model)) ==
    Done{Contract.listed(~Element,~Context,~reader,ShapeLaws.elements(shape),0n,context)} : Result<&2,&2,Tensors.Problem,List<&2,Element>>}:
  Buffer.Model{+tree,+length,certificate} = model
  Shape.Both{Contract.WellFormed{+length_exact,ready},+values} = facts
  %length_exact : {Done{Storage.read_list(~Element,length,0,Tree.pack(Element,tree))} ==
    Done{Contract.listed(~Element,~Context,~reader,_,0n,context)} : Result<&2,&2,Tensors.Problem,List<&2,Element>>}
  %Equal.sym(List<&2,Element>,Storage.read_list(~Element,length,0,Tree.pack(Element,tree)),Tree.read_sequence(~Element,length,0,tree),
      StorageLaws.read_sequence(~Element,length,0,tree)) :
    {Done{_} == Done{Contract.listed(~Element,~Context,~reader,length,0n,context)} : Result<&2,&2,Tensors.Problem,List<&2,Element>>}
  Equal.cong(List<&2,Element>,Result<&2,&2,Tensors.Problem,List<&2,Element>>,list => Done{list},
    Tree.read_sequence(~Element,length,0,tree),Contract.listed(~Element,~Context,~reader,length,0n,context),
    Contract.read_listed(~Element,~Context,~reader,context,tree,length,values,length,0n,Order.reflexive(length)))

def Listed(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,+shape: List<&2,U32>,
  result: Result<&2,&2,Tensors.Problem,List<&2,Element>>) -> Type:
  {result == Done{Contract.listed(~Element,~Context,~reader,ShapeLaws.elements(shape),0n,context)} : Result<&2,&2,Tensors.Problem,List<&2,Element>>}

# to_list of a tensor of the contract is the list of the reader's values at
# the flat indices from 0.
def to_list(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,
  +bands: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Global.Contract(~Element,~Context,~reader,context,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})) ->
  Listed(~Element,~Context,~reader,context,shape,Tensors.to_list(~Element,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})):
  WholeLeaf.with_whole(~Element,~Context,~reader,~Result<&2,&2,Tensors.Problem,List<&2,Element>>,
    ~Listed(~Element,~Context,~reader,context,shape),
    ~(buffer => shape => fill => status => workers => Tensors.listed_buffer(~Element,status,buffer)),
    context,bands,geometry,shape,fill,workers,contract,
    model => facts => listed(~Element,~Context,~reader,context,shape,model,facts))

def Single(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,result: Result<&2,&2,Tensors.Problem,Element>) -> Type:
  {result == Done{reader(context,U32.from_nat(0n))} : Result<&2,&2,Tensors.Problem,Element>}

# Storage that holds the reader's values in a shape of one element gives the
# reader's first value.
def single(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,+shape: List<&2,U32>,
  +one: {ShapeLaws.elements(shape) == 1n : Nat},+model: Buffer.Model<Element>,
  +facts: GatherValues.Facts(~Element,~Context,~reader,context,shape,Buffer.content(~Element,model))) ->
  {Tensors.single(~Element,shape,Tensors.listed_buffer(~Element,Tensors.Usable{},Buffer.pack(~Element,model))) ==
    Done{reader(context,U32.from_nat(0n))} : Result<&2,&2,Tensors.Problem,Element>}:
  Equal.trans(Result<&2,&2,Tensors.Problem,Element>,
    Tensors.single(~Element,shape,Tensors.listed_buffer(~Element,Tensors.Usable{},Buffer.pack(~Element,model))),
    Tensors.single(~Element,shape,Done{Contract.listed(~Element,~Context,~reader,ShapeLaws.elements(shape),0n,context)}),
    Done{reader(context,U32.from_nat(0n))},
    Equal.cong(Result<&2,&2,Tensors.Problem,List<&2,Element>>,Result<&2,&2,Tensors.Problem,Element>,values => Tensors.single(~Element,shape,values),
      Tensors.listed_buffer(~Element,Tensors.Usable{},Buffer.pack(~Element,model)),Done{Contract.listed(~Element,~Context,~reader,ShapeLaws.elements(shape),0n,context)},
      listed(~Element,~Context,~reader,context,shape,model,facts)),
    %Equal.sym(Nat,ShapeLaws.elements(shape),1n,one) :
      {Tensors.single(~Element,shape,Done{Contract.listed(~Element,~Context,~reader,_,0n,context)}) == Done{reader(context,U32.from_nat(0n))} : Result<&2,&2,Tensors.Problem,Element>}
    {==})

# item of a tensor of one element of the contract is the reader's first value.
def item(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,
  +bands: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  +one: {ShapeLaws.elements(shape) == 1n : Nat},
  contract: Global.Contract(~Element,~Context,~reader,context,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})) ->
  Single(~Element,~Context,~reader,context,Tensors.item(~Element,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})):
  WholeLeaf.with_whole(~Element,~Context,~reader,~Result<&2,&2,Tensors.Problem,Element>,
    ~Single(~Element,~Context,~reader,context),
    ~(storage => shape => fill => status => workers =>
      +shape = shape
      Tensors.single(~Element,shape,Tensors.listed_buffer(~Element,status,storage))),
    context,bands,geometry,shape,fill,workers,contract,
    model => facts => single(~Element,~Context,~reader,context,shape,one,model,facts))

# What a save hands to the writer: the tensor it returns, which keeps the
# contract, and storage that holds the reader's values.
type Saved<-Context: Data,-reader: Context -> U32 -> F32,-context: Context,-geometry: Bands.Geometry,-shape: List<&2,U32>,-fill: F32,-workers: U32,
  -operation: String,-path: String,-prefix: List<&2,U32>,-saved: IO(Tensors.Tensor<F32>)> is Data:
  Saved{kept: Model.Model<F32>,model: Buffer.Model<F32>,
    equation: {saved == Files.saving(operation,path,Tensors.Tensor{Model.pack(~F32,kept),geometry,shape,fill,Tensors.Usable{},workers},prefix,Done{Buffer.pack(~F32,model)}) :
      IO(Tensors.Tensor<F32>)},
    contract: Global.Contract(~F32,~Context,~reader,context,Tensors.Tensor{Model.pack(~F32,kept),geometry,shape,fill,Tensors.Usable{},workers}),
    facts: GatherValues.Facts(~F32,~Context,~reader,context,shape,Buffer.content(~F32,model))}

# The kept tensor of a save and the storage of the result of into_storage.
def kept_saved(~Context: Data,~reader: Context -> U32 -> F32,+context: Context,+geometry: Bands.Geometry,+shape: List<&2,U32>,
  +fill: F32,+workers: U32,+operation: String,+path: String,+prefix: List<&2,U32>,+kept: Model.Model<F32>,
  +contract: Global.Contract(~F32,~Context,~reader,context,Tensors.Tensor{Model.pack(~F32,kept),geometry,shape,fill,Tensors.Usable{},workers}),
  -result: Result<&2,&1,Tensors.Problem,Storage.Buffer<F32>>,stored: Stored<F32,Context,reader,context,shape,result>) ->
  Saved<Context,reader,context,geometry,shape,fill,workers,operation,path,prefix,
    Files.saving(operation,path,Tensors.Tensor{Model.pack(~F32,kept),geometry,shape,fill,Tensors.Usable{},workers},prefix,result)>:
  Stored{+model,+equation,+facts} = stored
  Saved{kept,model,
    Equal.cong(Result<&2,&1,Tensors.Problem,Storage.Buffer<F32>>,IO(Tensors.Tensor<F32>),
      storage => Files.saving(operation,path,Tensors.Tensor{Model.pack(~F32,kept),geometry,shape,fill,Tensors.Usable{},workers},prefix,storage),
      result,Done{Buffer.pack(~F32,model)},equation),
    contract,facts}

# A save writes the storage of one copy of a tensor of the contract and
# returns the other copy.
def copies_saved(~Context: Data,~reader: Context -> U32 -> F32,+context: Context,+bands: Model.Model<F32>,+geometry: Bands.Geometry,+shape: List<&2,U32>,
  +fill: F32,+workers: U32,+operation: String,+path: String,+prefix: List<&2,U32>,+left: Model.Model<F32>,+right: Model.Model<F32>,
  +left_content: {Model.content(~F32,left) == Model.content(~F32,bands) : Model.Content<F32>},
  +right_content: {Model.content(~F32,right) == Model.content(~F32,bands) : Model.Content<F32>},
  +contract: Global.Contract(~F32,~Context,~reader,context,Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers})) ->
  Saved<Context,reader,context,geometry,shape,fill,workers,operation,path,prefix,
    Files.saving(operation,path,Tensors.Tensor{Model.pack(~F32,left),geometry,shape,fill,Tensors.Usable{},workers},prefix,
      Tensors.into_storage(~F32,Tensors.Tensor{Model.pack(~F32,right),geometry,shape,fill,Tensors.Usable{},workers}))>:
  kept_saved(~Context,~reader,context,geometry,shape,fill,workers,operation,path,prefix,left,
    CopyContract.kept(~F32,~Context,~reader,context,bands,left,geometry,shape,fill,workers,left_content,contract),
    Tensors.into_storage(~F32,Tensors.Tensor{Model.pack(~F32,right),geometry,shape,fill,Tensors.Usable{},workers}),
    into_storage(~F32,~Context,~reader,context,right,geometry,shape,fill,workers,
      CopyContract.kept(~F32,~Context,~reader,context,bands,right,geometry,shape,fill,workers,right_content,contract)))

# save of the copies of a tensor of the contract.
def save_copied(~Context: Data,~reader: Context -> U32 -> F32,+context: Context,+path: String,
  +bands: Model.Model<F32>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  +contract: Global.Contract(~F32,~Context,~reader,context,Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers}),
  copies: CopyContract.Copied<F32,bands>) ->
  Saved<Context,reader,context,geometry,shape,fill,workers,"save",path,[],
    Files.save(path,Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers})>:
  CopyContract.Copied{+left,+right,+equation,+left_content,+right_content} = copies
  %equation : Saved<Context,reader,context,geometry,shape,fill,workers,"save",path,[],
    Tensors.then(~Tensors.Tensor<F32>,~Tensors.Tensor<F32>,~IO(Tensors.Tensor<F32>),
      Storage.bind_pair(~Bands.Bands(~F32),~Bands.Bands(~F32),~(Tensors.Tensor<F32> & Tensors.Tensor<F32>),_,
        first => second => (Tensors.Tensor{first,geometry,shape,fill,Tensors.Usable{},workers},Tensors.Tensor{second,geometry,shape,fill,Tensors.Usable{},workers})),
      kept => copy => Files.saving("save",path,kept,[],Tensors.into_storage(~F32,copy)))>
  copies_saved(~Context,~reader,context,bands,geometry,shape,fill,workers,"save",path,[],left,right,left_content,right_content,contract)

# save of a tensor of the contract writes storage that holds the reader's
# values and returns a tensor of the same contract.
def save(~Context: Data,~reader: Context -> U32 -> F32,+context: Context,+path: String,
  +bands: Model.Model<F32>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  +contract: Global.Contract(~F32,~Context,~reader,context,Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers})) ->
  Saved<Context,reader,context,geometry,shape,fill,workers,"save",path,[],
    Files.save(path,Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers})>:
  save_copied(~Context,~reader,context,path,bands,geometry,shape,fill,workers,contract,CopyContract.copied(~F32,bands))

# save_npy of the copies of a tensor of the contract.
def save_npy_copied(~Context: Data,~reader: Context -> U32 -> F32,+context: Context,+path: String,
  +bands: Model.Model<F32>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  +contract: Global.Contract(~F32,~Context,~reader,context,Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers}),
  copies: CopyContract.Copied<F32,bands>) ->
  Saved<Context,reader,context,geometry,shape,fill,workers,"save_npy",path,Files.npy_header(shape),
    Files.save_npy(path,Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers})>:
  CopyContract.Copied{+left,+right,+equation,+left_content,+right_content} = copies
  %equation : Saved<Context,reader,context,geometry,shape,fill,workers,"save_npy",path,Files.npy_header(shape),
    Files.npy_saving(path,
      Storage.bind_pair(~Bands.Bands(~F32),~Bands.Bands(~F32),~(Tensors.Tensor<F32> & Tensors.Tensor<F32>),_,
        first => second => (Tensors.Tensor{first,geometry,shape,fill,Tensors.Usable{},workers},Tensors.Tensor{second,geometry,shape,fill,Tensors.Usable{},workers})),
      shape)>
  copies_saved(~Context,~reader,context,bands,geometry,shape,fill,workers,"save_npy",path,Files.npy_header(shape),left,right,left_content,right_content,contract)

# save_npy of a tensor of the contract writes the header of its shape and
# storage that holds the reader's values, and returns a tensor of the same
# contract.
def save_npy(~Context: Data,~reader: Context -> U32 -> F32,+context: Context,+path: String,
  +bands: Model.Model<F32>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  +contract: Global.Contract(~F32,~Context,~reader,context,Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers})) ->
  Saved<Context,reader,context,geometry,shape,fill,workers,"save_npy",path,Files.npy_header(shape),
    Files.save_npy(path,Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers})>:
  save_npy_copied(~Context,~reader,context,path,bands,geometry,shape,fill,workers,contract,CopyContract.copied(~F32,bands))
