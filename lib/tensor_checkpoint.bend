# A PyTorch checkpoint: the ZIP archive torch.save writes, whose data.pkl
# holds the saved object and whose data/<key> entries hold the storages. The
# saved module tree, or a state dict, gives one tensor per state-dict entry
# under its PyTorch name, read from its storage in the stored format.
import Base
import ./storage_bytes.bend as Bytes
import ./checkpoint_zip.bend as Archive
import ./checkpoint_pickle.bend as Pickle
import ./storage_element_format.bend as Formats
import ./tensor.bend as Tensors
import ./tensor_file.bend as Files
import ./tensor_model.bend as Models
import ./tensor_view.bend as Views
import ./tensor_shape.bend as Shapes

def Tensor() -> Type:
  Tensors.Tensor<F32>

def operation() -> String:
  "load_checkpoint"

# What the pickle says about one state-dict entry: the storage class, the key
# of its entry, the element offset in the storage, the shape and the axes of
# the view of the storage, one extent and stride each.
type Described is Data:
  Described{name: String,kind: String,key: String,offset: U32,shape: List<&2,U32>,axes: List<&2,Views.Axis>}
  Unusable{name: String,detail: String}

# A module still to walk, with the prefix of its tensors' names.
type Task is Data:
  Task{prefix: String,module: Pickle.Value}

type Loading is Type:
  Loading{file: File,store: Models.Store,remaining: List<&2,Described>}

# Tensors as torch._utils rebuilds them: a parameter wraps a tensor, and a tensor
# is a storage reference, an offset, a size and a stride.

def numbers(items: List<&2,Pickle.Value>) -> List<&2,U32>:
  match items:
    case Nil{}: []
    case Con{Pickle.Integer{word},rest}: word <> numbers(rest)
    case Con{other,rest}: 0 <> numbers(rest)

def all_numbers(items: List<&2,Pickle.Value>) -> Bool:
  match items:
    case Nil{}: True{}
    case Con{Pickle.Integer{word},rest}: all_numbers(rest)
    case Con{other,rest}: False{}

# The axes of the tensor as the library views them.
def prepended(+extent: U32,+stride: U32,axes: Maybe<&2,List<&2,Views.Axis>>) -> Maybe<&2,List<&2,Views.Axis>>:
  match axes:
    case None{}: None{}
    case Some{axes}: Some{Views.Axis{extent,stride} <> axes}

def axes_of(extents: List<&2,U32>,strides: List<&2,U32>) -> Maybe<&2,List<&2,Views.Axis>>:
  match extents strides:
    case Nil{} Nil{}: Some{[]}
    case Nil{} Con{stride,more}: None{}
    case Con{extent,rest} Nil{}: None{}
    case Con{extent,rest} Con{stride,more}: prepended(extent,stride,axes_of(rest,more))

def described_layout(axes: Maybe<&2,List<&2,Views.Axis>>,+name: String,+kind: String,+key: String,+offset: U32,+shape: List<&2,U32>) -> Described:
  match axes:
    case Some{axes}: Described{name,kind,key,offset,shape,axes}
    case None{}: Unusable{name,"has a different number of sizes and strides"}

def described_numbers(plain: Bool,+name: String,+kind: String,+key: String,+offset: U32,+sizes: List<&2,Pickle.Value>,+strides: List<&2,Pickle.Value>) -> Described:
  match plain:
    case False{}: Unusable{name,"has a size or stride that is not an integer"}
    case True{}: described_layout(axes_of(numbers(sizes),numbers(strides)),name,kind,key,offset,numbers(sizes))

# The persistent id of a storage: ('storage', torch.<Kind>Storage, key, location, count).
def described_storage(+name: String,id: Pickle.Value,+offset: U32,+sizes: List<&2,Pickle.Value>,+strides: List<&2,Pickle.Value>) -> Described:
  match id:
    case Pickle.Sequence{Con{Pickle.Text{tag},Con{Pickle.Global{module,kind},Con{Pickle.Text{key},others}}}}:
      described_numbers(all_numbers(sizes) && all_numbers(strides),name,kind,key,offset,sizes,strides)
    case id: Unusable{name,"has a storage reference of an unknown form"}

def described_arguments(+name: String,arguments: Pickle.Value) -> Described:
  match arguments:
    case Pickle.Sequence{Con{Pickle.Persistent{id},Con{Pickle.Integer{offset},Con{Pickle.Sequence{sizes},Con{Pickle.Sequence{strides},others}}}}}:
      described_storage(name,id,offset,sizes,strides)
    case arguments: Unusable{name,"is rebuilt with unexpected arguments"}

def described_call(rebuilt: Bool,+name: String,arguments: Pickle.Value) -> Described:
  match rebuilt:
    case True{}: described_arguments(name,arguments)
    case False{}: Unusable{name,"is not rebuilt by torch._utils._rebuild_tensor_v2"}

def tensor_of(+name: String,value: Pickle.Value) -> Described:
  match value:
    case Pickle.Reduced{Pickle.Global{module,+function},arguments}:
      described_call(String.eq(function,"_rebuild_tensor_v2"),name,arguments)
    case value: Unusable{name,"is not a tensor"}

def unwrapped(parameter: Bool,+name: String,inner: Pickle.Value,+whole: Pickle.Value) -> Described:
  match parameter:
    case True{}: tensor_of(name,inner)
    case False{}: tensor_of(name,whole)

# A parameter is _rebuild_parameter(tensor, requires_grad, hooks).
def described(+name: String,+value: Pickle.Value) -> Described:
  match value:
    case Pickle.Reduced{Pickle.Global{module,+function},Pickle.Sequence{Con{+inner,others}}}:
      unwrapped(String.eq(function,"_rebuild_parameter"),name,inner,Pickle.Reduced{Pickle.Global{module,function},Pickle.Sequence{inner <> others}})
    case value: tensor_of(name,value)

# The module tree. A module's state holds _parameters and _buffers, names to
# tensors or None, _non_persistent_buffers_set, and _modules, names to child
# modules. The state dict lists parameters, then persistent buffers, then each
# child in order, so a walk takes a module's own tensors and then its children.

def listed(set: Pickle.Value) -> List<&2,Pickle.Value>:
  match set:
    case Pickle.Reduced{callable,Pickle.Sequence{Con{Pickle.Items{names},others}}}: names
    case Pickle.Items{names}: names
    case set: []

def named(names: List<&2,Pickle.Value>,+name: String) -> Bool:
  match names:
    case Nil{}: False{}
    case Con{Pickle.Text{+text},rest}: String.eq(text,name) || named(rest,name)
    case Con{other,rest}: named(rest,name)

def entries_of(value: Pickle.Value) -> List<&2,Pickle.Value>:
  match value:
    case Pickle.Dict{entries}: entries
    case value: []

# The tensors of one dictionary, newest first onto the found list; None and the
# excluded names are skipped.
def own(entries: List<&2,Pickle.Value>,+prefix: String,+excluded: List<&2,Pickle.Value>,+found: List<&2,Described>) -> List<&2,Described>:
  match entries:
    case Nil{}: found
    case Con{Pickle.Sequence{Con{Pickle.Text{+name},Con{Pickle.Absent{},others}}},rest}: own(rest,prefix,excluded,found)
    case Con{Pickle.Sequence{Con{Pickle.Text{+name},Con{value,others}}},rest}:
      own(rest,prefix,excluded,Bool.pick(List<&2,Described>,named(excluded,name),found,described(prefix ++ name,value) <> found))
    case Con{entry,rest}: own(rest,prefix,excluded,found)

def children(entries: List<&2,Pickle.Value>,+prefix: String) -> List<&2,Task>:
  match entries:
    case Nil{}: []
    case Con{Pickle.Sequence{Con{Pickle.Text{name},Con{Pickle.Absent{},others}}},rest}: children(rest,prefix)
    case Con{Pickle.Sequence{Con{Pickle.Text{name},Con{module,others}}},rest}: Task{prefix ++ name ++ ".",module} <> children(rest,prefix)
    case Con{entry,rest}: children(rest,prefix)

def state_of(module: Pickle.Value) -> List<&2,Pickle.Value>:
  match module:
    case Pickle.Object{class,arguments,state}: entries_of(state)
    case module: []

# Every module is one step, so the byte count of the pickle bounds the walk.
def walk(fuel: Nat,todo: List<&2,Task>,found: List<&2,Described>) -> List<&2,Described>:
  match fuel todo:
    case fuel Nil{}: List.reverse(&2,Described,found)
    case 0n todo: List.reverse(&2,Described,found)
    case 1n+rest Con{Task{+prefix,module},more}:
      +state = state_of(module)
      walk(rest,List.append(&2,Task,children(entries_of(Pickle.get(state,"_modules")),prefix),more),
        own(entries_of(Pickle.get(state,"_buffers")),prefix,listed(Pickle.get(state,"_non_persistent_buffers_set")),
          own(entries_of(Pickle.get(state,"_parameters")),prefix,[],found)))

# A state dict: names to tensors, in order.
def plain(entries: List<&2,Pickle.Value>,found: List<&2,Described>) -> List<&2,Described>:
  match entries:
    case Nil{}: List.reverse(&2,Described,found)
    case Con{Pickle.Sequence{Con{Pickle.Text{+name},Con{value,others}}},rest}: plain(rest,described(name,value) <> found)
    case Con{entry,rest}: plain(rest,found)

def is_module(value: Pickle.Value) -> Bool:
  match value:
    case Pickle.Object{class,arguments,Pickle.Dict{entries}}: True{}
    case value: False{}

def rooted(module: Bool,+fuel: Nat,+value: Pickle.Value) -> List<&2,Described>:
  match module:
    case True{}: walk(fuel,[Task{"",value}],[])
    case False{}: plain(entries_of(value),[])

# The saved object: a module, a checkpoint dictionary whose "model" is one, or
# a state dict.
def tensors_of(+fuel: Nat,+value: Pickle.Value) -> List<&2,Described>:
  match value:
    case Pickle.Dict{+entries}: rooted(is_module(Pickle.get(entries,"model")),fuel,Bool.pick(Pickle.Value,is_module(Pickle.get(entries,"model")),Pickle.get(entries,"model"),Pickle.Dict{entries}))
    case value: rooted(is_module(value),fuel,value)

# Reading the storages

def format_classified(half: Bool,bfloat: Bool,float: Bool,double: Bool,long: Bool,int: Bool) -> Maybe<&2,Formats.Format>:
  match half bfloat float double long int:
    case True{} bfloat float double long int: Some{Formats.Fp16{}}
    case False{} True{} float double long int: Some{Formats.BFloat16{}}
    case False{} False{} True{} double long int: Some{Formats.Fp32{}}
    case False{} False{} False{} True{} long int: Some{Formats.Fp64{}}
    case False{} False{} False{} False{} True{} int: Some{Formats.Int64{}}
    case False{} False{} False{} False{} False{} True{}: Some{Formats.Int32{}}
    case False{} False{} False{} False{} False{} False{}: None{}

def format_of(+kind: String) -> Maybe<&2,Formats.Format>:
  format_classified(String.eq(kind,"HalfStorage"),String.eq(kind,"BFloat16Storage"),String.eq(kind,"FloatStorage"),String.eq(kind,"DoubleStorage"),
    String.eq(kind,"LongStorage"),String.eq(kind,"IntStorage"))

def fault(+path: String,+name: String,+detail: String,shape: List<&2,U32>) -> Tensor():
  Tensors.failed(~F32,operation(),Tensors.FileFault{path,"holds " ++ name ++ ", which " ++ detail},shape,0.0,1)

def storage_name(+prefix: List<&2,U32>,+key: String) -> List<&2,U32>:
  List.append(&2,U32,prefix,List.append(&2,U32,Bytes.codes("data/"),Bytes.codes(key)))

def bounded(+count: U32,+width: U32,+offset: U32,+length: U32) -> Bool:
  +first = (offset * width : U32)
  +bytes = (count * width : U32)
  U32.is_eq((first / width : U32),offset) && U32.is_eq((bytes / width : U32),count) && U32.is_le(first,length) && U32.is_le(bytes,(length - first : U32))

# The elements a view reaches from its offset: one past its last address, or
# none when an extent is zero. None when that does not fit a word.
def joined(product: Maybe<&2,U32>,rest: Maybe<&2,U32>) -> Maybe<&2,U32>:
  match product:
    case Some{product}: Shapes.checked_add_to(rest,product)
    case None{}: None{}

def last_address(axes: List<&2,Views.Axis>) -> Maybe<&2,U32>:
  match axes:
    case []: Some{0}
    case Views.Axis{extent,stride} <> rest: joined(Shapes.checked_multiply((extent - 1 : U32),stride),last_address(rest))

def reach(empty: Bool,axes: List<&2,Views.Axis>) -> Maybe<&2,U32>:
  match empty:
    case True{}: Some{0}
    case False{}: Shapes.checked_add_to(last_address(axes),1)

# A row-major view is read as the tensor itself. Any other view is read as the
# elements it reaches and copied through the view into row-major storage.
def strided(tensor: Tensor(),+axes: List<&2,Views.Axis>,+shape: List<&2,U32>) -> Tensor():
  Tensors.with_whole(~F32,~Tensor(),tensor,storage => reached => fill => status => workers =>
    Tensors.result_of(~F32,Tensors.gather_status(~F32,operation(),status,Some{Views.View{axes,0}},[],shape,shape,storage,reached,fill,workers)))

def rearranged(pair: File & Tensor(),+axes: List<&2,Views.Axis>,+shape: List<&2,U32>) -> File & Tensor():
  (file,tensor) = pair
  (file,strided(tensor,axes,shape))

def arranged(row_major: Bool,file: File,+path: String,+shape: List<&2,U32>,+axes: List<&2,Views.Axis>,+span: U32,+start: U32,+format: Formats.Format) -> IO(File & Tensor()):
  match row_major:
    case True{}: Files.load_range(file,operation(),path,shape,start,format)
    case False{}:
      do IO<File & Tensor()>:
        pair : File & Tensor() <- Files.load_range(file,operation(),path,[span],start,format)
        IO.pure(File & Tensor(),rearranged(pair,axes,shape))

# The elements the view reaches lie within the entry.
def within(fits: Bool,file: File,+path: String,+name: String,+shape: List<&2,U32>,+axes: List<&2,Views.Axis>,+span: U32,+start: U32,+format: Formats.Format) -> IO(File & Tensor()):
  match fits:
    case False{}: IO.pure(File & Tensor(),(file,fault(path,name,"reaches beyond its storage",shape)))
    case True{}: arranged(Maybe.is_some(&2,U32,Views.contiguous_size(axes)),file,path,shape,axes,span,start,format)

def spanned(span: Maybe<&2,U32>,file: File,+path: String,+name: String,+shape: List<&2,U32>,+axes: List<&2,Views.Axis>,+data: U32,+offset: U32,+length: U32,
  +format: Formats.Format) -> IO(File & Tensor()):
  match span:
    case None{}: IO.pure(File & Tensor(),(file,fault(path,name,"reaches beyond any storage",shape)))
    case Some{+span}:
      within(bounded(span,Formats.width(format),offset,length),file,path,name,shape,axes,span,(data + offset * Formats.width(format) : U32),format)

def located(data: Maybe<&2,U32>,file: File,+path: String,+name: String,+shape: List<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32,+length: U32,+format: Formats.Format) -> IO(File & Tensor()):
  match data:
    case None{}: IO.pure(File & Tensor(),(file,fault(path,name,"has a storage entry that cannot be read",shape)))
    case Some{+data}: spanned(reach(Shapes.has_zero(shape),axes),file,path,name,shape,axes,data,offset,length,format)

def stored(pair: File & Maybe<&2,U32>,+path: String,+name: String,+shape: List<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32,+length: U32,+format: Formats.Format) -> IO(File & Tensor()):
  (file,data) = pair
  located(data,file,path,name,shape,axes,offset,length,format)

def recorded(record: Maybe<&2,Archive.Record>,file: File,+path: String,+name: String,+shape: List<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32,+format: Formats.Format) -> IO(File & Tensor()):
  match record:
    case None{}: IO.pure(File & Tensor(),(file,fault(path,name,"has no storage entry",shape)))
    case Some{Archive.Record{entry,header,+length,method}}:
      do IO<File & Tensor()>:
        pair : File & Maybe<&2,U32> <- Archive.data_offset(file,Archive.Record{entry,header,length,method})
        stored(pair,path,name,shape,axes,offset,length,format)

def formatted(format: Maybe<&2,Formats.Format>,file: File,+path: String,+name: String,+kind: String,+key: String,+offset: U32,+shape: List<&2,U32>,+axes: List<&2,Views.Axis>,
  +records: List<&2,Archive.Record>,+prefix: List<&2,U32>) -> IO(File & Tensor()):
  match format:
    case None{}: IO.pure(File & Tensor(),(file,fault(path,name,"is stored as " ++ kind ++ ", a format that is not supported",shape)))
    case Some{format}: recorded(Archive.find(records,storage_name(prefix,key)),file,path,name,shape,axes,offset,format)

def load_one(described: Described,file: File,+path: String,+records: List<&2,Archive.Record>,+prefix: List<&2,U32>) -> IO(File & Tensor()):
  match described:
    case Unusable{+name,+detail}: IO.pure(File & Tensor(),(file,fault(path,name,detail,[])))
    case Described{+name,+kind,+key,+offset,+shape,+axes}: formatted(format_of(kind),file,path,name,kind,key,offset,shape,axes,records,prefix)

def name_of(described: Described) -> String:
  match described:
    case Unusable{name,detail}: name
    case Described{name,kind,key,offset,shape,axes}: name

def entered(pair: File & Tensor(),+name: String,store: Models.Store,remaining: List<&2,Described>) -> Loading:
  (file,tensor) = pair
  Loading{file,Models.Entry{name,tensor,store},remaining}

# The described tensors are visited last first, so the store lists them in order.
def load_all(fuel: Nat,loading: Loading,+path: String,+records: List<&2,Archive.Record>,+prefix: List<&2,U32>) -> IO(Loading):
  match fuel loading:
    case 0n loading: IO.pure(Loading,loading)
    case 1n+rest Loading{file,store,Nil{}}: IO.pure(Loading,Loading{file,store,[]})
    case 1n+rest Loading{file,store,Con{+first,more}}:
      do IO<Loading>:
        pair : File & Tensor() <- load_one(first,file,path,records,prefix)
        load_all(rest,entered(pair,name_of(first),store,more),path,records,prefix)

def closed(loading: Loading) -> IO(Models.Store):
  Loading{file,store,remaining} = loading
  do IO<Models.Store>:
    File.close(file)
    IO.pure(Models.Store,store)

def refused(file: File,+path: String,+detail: String) -> IO(Models.Store):
  do IO<Models.Store>:
    File.close(file)
    IO.pure(Models.Store,Models.Unavailable{Tensors.Problem{operation(),Tensors.FileFault{path,detail}}})

def parsed(outcome: Pickle.Parsed,file: File,+path: String,+records: List<&2,Archive.Record>,+prefix: List<&2,U32>,+count: Nat) -> IO(Models.Store):
  match outcome:
    case Pickle.Rejected{detail}: refused(file,path,"has a data.pkl that " ++ detail)
    case Pickle.Parsed{value}:
      +described = List.reverse(&2,Described,tensors_of(count,value))
      do IO<Models.Store>:
        loading : Loading <- load_all(Nat.add(List.length(&2,Described,described),1n),Loading{file,Models.Empty{},described},path,records,prefix)
        closed(loading)

def fetched(outcome: Archive.Fetched,+path: String,+records: List<&2,Archive.Record>,+prefix: List<&2,U32>) -> IO(Models.Store):
  Archive.Fetched{file,bytes} = outcome
  match bytes:
    case None{}: refused(file,path,"has a data.pkl that cannot be read")
    case Some{+bytes}: parsed(Pickle.parse(bytes),file,path,records,prefix,List.length(&2,U32,bytes))

def pickled(record: Maybe<&2,Archive.Record>,file: File,+path: String,+records: List<&2,Archive.Record>) -> IO(Models.Store):
  match record:
    case None{}: refused(file,path,"has no data.pkl entry")
    case Some{Archive.Record{+entry,header,length,method}}:
      do IO<Models.Store>:
        read : Archive.Fetched <- Archive.entry(file,Archive.Record{entry,header,length,method})
        fetched(read,path,records,List.take(&2,U32,entry,Nat.sub(List.length(&2,U32,entry),8n)))

def listed_archive(listing: Archive.Listing,file: File,+path: String) -> IO(Models.Store):
  match listing:
    case Archive.Unreadable{detail}: refused(file,path,detail)
    case Archive.Records{+records}: pickled(Archive.first_matching(records,Bytes.codes("data.pkl")),file,path,records)

def directory(pair: File & Archive.Listing,+path: String) -> IO(Models.Store):
  (file,listing) = pair
  listed_archive(listing,file,path)

def opened(result: Result<&1,&1,U32 & String,File>,+path: String) -> IO(Models.Store):
  match result:
    case Fail{error}: IO.pure(Models.Store,Models.Unavailable{Tensors.Problem{operation(),Tensors.FileFault{path,"cannot be opened"}}})
    case Done{file}:
      do IO<Models.Store>:
        pair : File & Archive.Listing <- Archive.directory(file)
        directory(pair,path)

# One tensor per state-dict entry under its PyTorch name. FP16 and BFloat16
# values widen exactly; integers round to the nearest FP32. A file that is not
# a checkpoint gives a store whose every take reports why.
def load(+path: String) -> IO(Models.Store):
  do IO<Models.Store>:
    result : Result<&1,&1,U32 & String,File> <- File.open(path,"r")
    opened(result,path)
