# The exits read the tiled form of a tensor of the contract of a reader as
# the reader's values in flat index order: into_storage gives storage that
# holds them, to_list their list, item the one value, and save_raw and
# save_npy hand that storage to the writer and return a tensor of the same
# contract.
import Base
import ../tensor.bend as Tensors
import ../tensor_f32.bend as F32Tensor
import ../tensor_file.bend as Files
import ../tensor_grid.bend as Grid
import ../tensor_pieces.bend as Pieces
import ../tensor_tiled.bend as Tiled
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ../storage_clone.bend as Clone
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./storage_clone_refinement.bend as CloneLaws
import ./tensor_contract.bend as Contract
import ./tensor_global_shape_count.bend as Global
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_copy_contract.bend as CopyContract
import ./tensor_tiled_public.bend as Public
import ./tensor_tiled_exit.bend as Exit
import ./tensor_tiled_retiled.bend as Retiled
import ./nat_order.bend as Order
import ./list_storage_readback.bend as ListRead
import ./natural_addition.bend as Addition
import ./word_encoding.bend as WordEncoding
import ../kernel_gemm.bend as KernelGemm

# Storage that holds the reader's values in the shape's element count.
type Stored<-Element: Data,-Context: Data,-reader: Context -> U32 -> Element,-context: Context,-shape: List<&2,U32>,
  -result: Result<&2,&1,Tensors.Problem,Storage.Buffer<Element>>> is Data:
  Stored{model: Buffer.Model<Element>,equation: {result == Done{Buffer.pack(~Element,model)} : Result<&2,&1,Tensors.Problem,Storage.Buffer<Element>>},
    facts: Contract.Facts(~Element,~Context,~reader,context,shape,Buffer.content(~Element,model))}

def listed(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,+shape: List<&2,U32>,+model: Buffer.Model<Element>,
  +facts: Contract.Facts(~Element,~Context,~reader,context,shape,Buffer.content(~Element,model))) ->
  {Tensors.listed_buffer(~Element,Tensors.Usable{},Buffer.pack(~Element,model)) ==
    Done{Contract.listed(~Element,~Context,~reader,ShapeLaws.elements(shape),0n,context)} : Result<&2,&2,Tensors.Problem,List<&2,Element>>}:
  Buffer.Model{+tree,+length,certificate} = model
  KernelGemm.Both{Contract.WellFormed{+length_exact,ready},+values} = facts
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

# The list of a reader holds count values, and the one at an index below the
# count is the reader's value at offset plus the index.
def listed_length(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,count: Nat,+offset: Nat,+context: Context) ->
  {List.length(&2,Element,Contract.listed(~Element,~Context,~reader,count,offset,context)) == count : Nat}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,Element,Contract.listed(~Element,~Context,~reader,rest,1n+offset,context)),rest,
        listed_length(~Element,~Context,~reader,rest,1n+offset,context))

def listed_at(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+fallback: Element,count: Nat,+offset: Nat,+context: Context,index: Nat,
  inside: {Nat.is_lt(index,count) == True{} : Bool}) ->
  {ListRead.at(~Element,fallback,index,Contract.listed(~Element,~Context,~reader,count,offset,context)) == reader(context,U32.from_nat(Nat.add(offset,index))) : Element}:
  match count index:
    case 0n 0n: Empty.absurd({fallback == reader(context,U32.from_nat(Nat.add(offset,0n))) : Element},WordEncoding.false_true(inside))
    case 0n 1n+previous: Empty.absurd({fallback == reader(context,U32.from_nat(Nat.add(offset,1n+previous))) : Element},WordEncoding.false_true(inside))
    case 1n+rest 0n:
      %Equal.sym(Nat,Nat.add(offset,0n),offset,Addition.zero_right(offset)) : {reader(context,U32.from_nat(offset)) == reader(context,U32.from_nat(_)) : Element}
      {==}
    case 1n+ +rest 1n+ +previous:
      %Equal.sym(Nat,Nat.add(offset,1n+previous),1n+Nat.add(offset,previous),Addition.successor_right(offset,previous)) : {ListRead.at(~Element,fallback,previous,Contract.listed(~Element,~Context,~reader,rest,1n+offset,context)) == reader(context,U32.from_nat(_)) : Element}
      listed_at(~Element,~Context,~reader,fallback,rest,1n+offset,context,previous,inside)

def Single(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,result: Result<&2,&2,Tensors.Problem,Element>) -> Type:
  {result == Done{reader(context,U32.from_nat(0n))} : Result<&2,&2,Tensors.Problem,Element>}

# Storage that holds the reader's values in a shape of one element gives the
# reader's first value.
def single(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,+shape: List<&2,U32>,
  +one: {ShapeLaws.elements(shape) == 1n : Nat},+model: Buffer.Model<Element>,
  +facts: Contract.Facts(~Element,~Context,~reader,context,shape,Buffer.content(~Element,model))) ->
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

# ---- The tiled form ----

# The values the untiled form stores are storage that holds the reader's
# values, their list and their one value.
def stored_untiled(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,-tensor: Tensors.Tensor<F32>,
  +untiled: Exit.Untiled<Logical,original,logical,values,tensor>) -> Stored<F32,Logical,original,logical,values,Tensors.into_storage(~F32,tensor)>:
  Exit.Untiled{+model,+workers,+equation,+facts} = untiled
  %Equal.sym(Tensors.Tensor<F32>,tensor,Tensors.values(~F32,Buffer.pack(~F32,model),values,0.0,Tensors.Usable{},workers),equation) :
    Stored<F32,Logical,original,logical,values,Tensors.into_storage(~F32,_)>
  Stored{model,{==},facts}

def listed_untiled(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,-tensor: Tensors.Tensor<F32>,
  +untiled: Exit.Untiled<Logical,original,logical,values,tensor>) -> Listed(~F32,~Logical,~original,logical,values,Tensors.to_list(~F32,tensor)):
  Exit.Untiled{+model,+workers,+equation,+facts} = untiled
  %Equal.sym(Tensors.Tensor<F32>,tensor,Tensors.values(~F32,Buffer.pack(~F32,model),values,0.0,Tensors.Usable{},workers),equation) :
    Listed(~F32,~Logical,~original,logical,values,Tensors.to_list(~F32,_))
  listed(~F32,~Logical,~original,logical,values,model,facts)

def single_untiled(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,-tensor: Tensors.Tensor<F32>,
  +one: {ShapeLaws.elements(values) == 1n : Nat},+untiled: Exit.Untiled<Logical,original,logical,values,tensor>) ->
  Single(~F32,~Logical,~original,logical,Tensors.item(~F32,tensor)):
  Exit.Untiled{+model,+workers,+equation,+facts} = untiled
  %Equal.sym(Tensors.Tensor<F32>,tensor,Tensors.values(~F32,Buffer.pack(~F32,model),values,0.0,Tensors.Usable{},workers),equation) :
    Single(~F32,~Logical,~original,logical,Tensors.item(~F32,_))
  single(~F32,~Logical,~original,logical,values,one,model,facts)

# F.into_storage of a tensor of the contract gives storage that holds the
# reader's values.
def into_storage(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,tiled: Tiled.Tiled,
  +contract: Public.Contract(~Logical,~original,logical,values,tiled)) -> Stored<F32,Logical,original,logical,values,F32Tensor.into_storage(tiled)>:
  stored_untiled(~Logical,~original,logical,values,Tiled.untiled(tiled),Exit.untiled(~Logical,~original,logical,values,tiled,contract))

# F.to_list of a tensor of the contract is the list of the reader's values at
# the flat indices from 0.
def to_list(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,tiled: Tiled.Tiled,
  +contract: Public.Contract(~Logical,~original,logical,values,tiled)) -> Listed(~F32,~Logical,~original,logical,values,F32Tensor.to_list(tiled)):
  listed_untiled(~Logical,~original,logical,values,Tiled.untiled(tiled),Exit.untiled(~Logical,~original,logical,values,tiled,contract))

# F.item of a tensor of one element of the contract is the reader's first
# value.
def item(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,tiled: Tiled.Tiled,+one: {ShapeLaws.elements(values) == 1n : Nat},
  +contract: Public.Contract(~Logical,~original,logical,values,tiled)) -> Single(~F32,~Logical,~original,logical,F32Tensor.item(tiled)):
  single_untiled(~Logical,~original,logical,values,Tiled.untiled(tiled),one,Exit.untiled(~Logical,~original,logical,values,tiled,contract))

# What a save hands to the writer: the kept values, whose tiled form keeps
# the contract, and storage that holds the reader's values.
type Saved<-Logical: Data,-original: Logical -> U32 -> F32,-logical: Logical,-values: List<&2,U32>,
  -operation: String,-path: String,-prefix: List<&2,U32>,-saved: IO(Tiled.Tiled)> is Data:
  Saved{kept: Buffer.Model<F32>,workers: U32,model: Buffer.Model<F32>,
    equation: {saved == F32Tensor.loaded(Files.saving(operation,path,Tensors.values(~F32,Buffer.pack(~F32,kept),values,0.0,Tensors.Usable{},workers),prefix,Done{Buffer.pack(~F32,model)})) :
      IO(Tiled.Tiled)},
    kept_facts: Contract.Facts(~F32,~Logical,~original,logical,values,Buffer.content(~F32,kept)),
    contract: Public.Contract(~Logical,~original,logical,values,Tiled.tiled(Tensors.values(~F32,Buffer.pack(~F32,kept),values,0.0,Tensors.Usable{},workers))),
    facts: Contract.Facts(~F32,~Logical,~original,logical,values,Buffer.content(~F32,model))}

def saved_observed(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+workers: U32,
  +operation: String,+path: String,+prefix: List<&2,U32>,+counted: Global.Counted<values>,+model: Buffer.Model<F32>,
  +facts: Contract.Facts(~F32,~Logical,~original,logical,values,Buffer.content(~F32,model)),
  observation: CloneLaws.Observation<F32,Clone.clone(~F32,Buffer.pack(~F32,model))>,-saved: IO(Tiled.Tiled),
  step: @+left: Buffer.Model<F32> -> @+right: Buffer.Model<F32> ->
    @+copied: {Tensors.copied(~F32,4.0,Tensors.values(~F32,Buffer.pack(~F32,model),values,0.0,Tensors.Usable{},workers)) ==
      (Tensors.values(~F32,Buffer.pack(~F32,left),values,0.0,Tensors.Usable{},workers),Tensors.values(~F32,Buffer.pack(~F32,right),values,0.0,Tensors.Usable{},workers)) :
      Tensors.Tensor<F32> & Tensors.Tensor<F32>} ->
    {saved == F32Tensor.loaded(Files.saving(operation,path,Tensors.values(~F32,Buffer.pack(~F32,left),values,0.0,Tensors.Usable{},workers),prefix,Done{Buffer.pack(~F32,right)})) :
      IO(Tiled.Tiled)}) -> Saved<Logical,original,logical,values,operation,path,prefix,saved>:
  CloneLaws.Observation{+left,+right,+clone_eq} = observation
  +count = Tensors.count_or_zero(values)
  +pair_eq = Equal.cong(CopyContract.Owners(~F32),Pieces.Pieces(~F32) & Pieces.Pieces(~F32),owners => Tensors.copy_pair(~F32,0,count,owners),
    (Buffer.pack(~F32,left),Buffer.pack(~F32,right)),Clone.clone(~F32,Buffer.pack(~F32,model)),clone_eq)
  +copied_eq = CopyContract.copied_equation(~F32,4.0,PiecesModel.Leaf{model,0,count},values,0.0,workers,
    PiecesModel.Leaf{left,0,count},PiecesModel.Leaf{right,0,count},pair_eq)
  +left_facts = {%Equal.sym(Buffer.Content<F32>,Buffer.content(~F32,left),Buffer.content(~F32,model),CloneLaws.left_contents(~F32,model,left,right,clone_eq)) :
    Contract.Facts(~F32,~Logical,~original,logical,values,_) facts : Contract.Facts(~F32,~Logical,~original,logical,values,Buffer.content(~F32,left))}
  +right_facts = {%Equal.sym(Buffer.Content<F32>,Buffer.content(~F32,right),Buffer.content(~F32,model),CloneLaws.right_contents(~F32,model,left,right,clone_eq)) :
    Contract.Facts(~F32,~Logical,~original,logical,values,_) facts : Contract.Facts(~F32,~Logical,~original,logical,values,Buffer.content(~F32,right))}
  Saved{left,workers,right,step(left,right,copied_eq),left_facts,Retiled.tiled(~Logical,~original,logical,values,workers,left,counted,left_facts),right_facts}

def save_raw_untiled(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+path: String,
  +counted: Global.Counted<values>,-tensor: Tensors.Tensor<F32>,+untiled: Exit.Untiled<Logical,original,logical,values,tensor>) ->
  Saved<Logical,original,logical,values,"save",path,[],F32Tensor.loaded(Files.save(path,tensor))>:
  Exit.Untiled{+model,+workers,+equation,+facts} = untiled
  %Equal.sym(Tensors.Tensor<F32>,tensor,Tensors.values(~F32,Buffer.pack(~F32,model),values,0.0,Tensors.Usable{},workers),equation) :
    Saved<Logical,original,logical,values,"save",path,[],F32Tensor.loaded(Files.save(path,_))>
  saved_observed(~Logical,~original,logical,values,workers,"save",path,[],counted,model,facts,
    CloneLaws.observe_pair(~F32,Clone.clone(~F32,Buffer.pack(~F32,model))),
    F32Tensor.loaded(Files.save(path,Tensors.values(~F32,Buffer.pack(~F32,model),values,0.0,Tensors.Usable{},workers))),
    left => right => copied_eq =>
      Equal.cong(Tensors.Tensor<F32> & Tensors.Tensor<F32>,IO(Tiled.Tiled),
        pair => F32Tensor.loaded(Storage.bind_pair(~Files.Tensor(),~Files.Tensor(),~IO(Files.Tensor()),pair,kept => copy => Files.saving("save",path,kept,[],Tensors.into_storage(~F32,copy)))),
        Tensors.copied(~F32,4.0,Tensors.values(~F32,Buffer.pack(~F32,model),values,0.0,Tensors.Usable{},workers)),
        (Tensors.values(~F32,Buffer.pack(~F32,left),values,0.0,Tensors.Usable{},workers),Tensors.values(~F32,Buffer.pack(~F32,right),values,0.0,Tensors.Usable{},workers)),
        copied_eq))

def save_npy_untiled(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+path: String,
  +counted: Global.Counted<values>,-tensor: Tensors.Tensor<F32>,+untiled: Exit.Untiled<Logical,original,logical,values,tensor>) ->
  Saved<Logical,original,logical,values,"save_npy",path,Files.npy_header(values),F32Tensor.loaded(Files.save_npy(path,tensor))>:
  Exit.Untiled{+model,+workers,+equation,+facts} = untiled
  %Equal.sym(Tensors.Tensor<F32>,tensor,Tensors.values(~F32,Buffer.pack(~F32,model),values,0.0,Tensors.Usable{},workers),equation) :
    Saved<Logical,original,logical,values,"save_npy",path,Files.npy_header(values),F32Tensor.loaded(Files.save_npy(path,_))>
  saved_observed(~Logical,~original,logical,values,workers,"save_npy",path,Files.npy_header(values),counted,model,facts,
    CloneLaws.observe_pair(~F32,Clone.clone(~F32,Buffer.pack(~F32,model))),
    F32Tensor.loaded(Files.save_npy(path,Tensors.values(~F32,Buffer.pack(~F32,model),values,0.0,Tensors.Usable{},workers))),
    left => right => copied_eq =>
      Equal.cong(Tensors.Tensor<F32> & Tensors.Tensor<F32>,IO(Tiled.Tiled),
        pair => F32Tensor.loaded(Files.npy_saving(path,pair,values)),
        Tensors.copied(~F32,4.0,Tensors.values(~F32,Buffer.pack(~F32,model),values,0.0,Tensors.Usable{},workers)),
        (Tensors.values(~F32,Buffer.pack(~F32,left),values,0.0,Tensors.Usable{},workers),Tensors.values(~F32,Buffer.pack(~F32,right),values,0.0,Tensors.Usable{},workers)),
        copied_eq))

def save_raw_packed(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+path: String,
  +model: PiecesModel.Model<Tile.Tile8>,+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+axis: Nat,+lanes: U32,+position: Grid.Position,
  +contract: Public.Contract(~Logical,~original,logical,values,
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},axis,lanes,position})) ->
  Saved<Logical,original,logical,values,"save",path,[],
    F32Tensor.save_raw(path,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},axis,lanes,position})>:
  KernelGemm.Both{global,Public.Laid{inside,shaped,restored,+counted,pinned}} = contract
  save_raw_untiled(~Logical,~original,logical,values,path,counted,
    Tiled.untiled(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},axis,lanes,position}),
    Exit.packed_untiled(~Logical,~original,logical,values,model,shape,fill,workers,axis,lanes,position,contract))

def save_npy_packed(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+path: String,
  +model: PiecesModel.Model<Tile.Tile8>,+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+axis: Nat,+lanes: U32,+position: Grid.Position,
  +contract: Public.Contract(~Logical,~original,logical,values,
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},axis,lanes,position})) ->
  Saved<Logical,original,logical,values,"save_npy",path,Files.npy_header(values),
    F32Tensor.save_npy(path,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},axis,lanes,position})>:
  KernelGemm.Both{global,Public.Laid{inside,shaped,restored,+counted,pinned}} = contract
  save_npy_untiled(~Logical,~original,logical,values,path,counted,
    Tiled.untiled(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},axis,lanes,position}),
    Exit.packed_untiled(~Logical,~original,logical,values,model,shape,fill,workers,axis,lanes,position,contract))

# F.save_raw of a tensor of the contract writes storage that holds the reader's
# values and returns a tensor of the same contract.
def save_raw(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+path: String,tiled: Tiled.Tiled,
  +contract: Public.Contract(~Logical,~original,logical,values,tiled)) ->
  Saved<Logical,original,logical,values,"save",path,[],F32Tensor.save_raw(path,tiled)>:
  Public.by_packed(~Logical,~original,(that => Saved<Logical,original,logical,values,"save",path,[],F32Tensor.save_raw(path,that)>),logical,values,tiled,contract,
    model => shape => fill => workers => axis => lanes => position => packed =>
      save_raw_packed(~Logical,~original,logical,values,path,model,shape,fill,workers,axis,lanes,position,packed))

# F.save_npy of a tensor of the contract writes the header of its shape and
# storage that holds the reader's values, and returns a tensor of the same
# contract.
def save_npy(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+path: String,tiled: Tiled.Tiled,
  +contract: Public.Contract(~Logical,~original,logical,values,tiled)) ->
  Saved<Logical,original,logical,values,"save_npy",path,Files.npy_header(values),F32Tensor.save_npy(path,tiled)>:
  Public.by_packed(~Logical,~original,(that => Saved<Logical,original,logical,values,"save_npy",path,Files.npy_header(values),F32Tensor.save_npy(path,that)>),logical,values,tiled,contract,
    model => shape => fill => workers => axis => lanes => position => packed =>
      save_npy_packed(~Logical,~original,logical,values,path,model,shape,fill,workers,axis,lanes,position,packed))
