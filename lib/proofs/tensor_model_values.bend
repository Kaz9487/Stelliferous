# A store of a model holds named tensors in order: take gives the first
# tensor of the name, holding its values, and leaves the others, and keep
# gives the tensor back and stores a copy, both holding its values.
import Base
import ../tensor_model.bend as Models
import ../tensor_tiled.bend as Tiled
import ../tensor.bend as Tensors
import ../kernel_gemm.bend as KernelGemm
import ./tensor_copy_contract.bend as CopyContract
import ./tensor_tiled_public.bend as Public

# A tensor of the store: its name, and the values and extents it holds.
type Named<-X: Data> is Data:
  Named{name: String,value: X,shape: List<&2,U32>}

# A store whose entries hold the named values in their order. A store that
# could not be loaded holds nothing.
def StoreHolds(~X: Data,~x_value: X -> U32 -> F32,store: Models.Store,+entries: List<&2,Named<X>>) -> Data:
  match store:
    case Models.Empty{}: {entries == [] : List<&2,Named<X>>}
    case Models.Unavailable{problem}: {entries == [] : List<&2,Named<X>>}
    case Models.Entry{name,tensor,rest}:
      match entries:
        case Nil{}: Empty
        case Con{Named{+entry_name,+value,+shape},+more}:
          KernelGemm.Both<{entry_name == name : String},KernelGemm.Both<Public.Contract(~X,~x_value,value,shape,tensor),StoreHolds(~X,~x_value,rest,more)>>

# The first entry of the name, and the entries without it.
def found(~X: Data,+entries: List<&2,Named<X>>,+wanted: String) -> Maybe<&2,Named<X>>:
  match entries:
    case Nil{}: None{}
    case Con{Named{+name,+value,+shape},+more}: Bool.pick(Maybe<&2,Named<X>>,String.eq(name,wanted),Some{Named{name,value,shape}},found(~X,more,wanted))

def without(~X: Data,+entries: List<&2,Named<X>>,+wanted: String) -> List<&2,Named<X>>:
  match entries:
    case Nil{}: []
    case Con{Named{+name,+value,+shape},+more}: Bool.pick(List<&2,Named<X>>,String.eq(name,wanted),more,Named{name,value,shape} <> without(~X,more,wanted))

# The tensor a take gives holds the values of the entry it found.
def TakenHolds(~X: Data,~x_value: X -> U32 -> F32,+entry: Maybe<&2,Named<X>>,tensor: Tiled.Tiled) -> Data:
  match entry:
    case None{}: Unit
    case Some{Named{name,+value,+shape}}: Public.Contract(~X,~x_value,value,shape,tensor)

def Taken(~X: Data,~x_value: X -> U32 -> F32,+entries: List<&2,Named<X>>,+wanted: String,result: Tiled.Tiled & Models.Store) -> Data:
  (tensor,store) = result
  KernelGemm.Both<TakenHolds(~X,~x_value,found(~X,entries,wanted),tensor),StoreHolds(~X,~x_value,store,without(~X,entries,wanted))>

# What the scan of a take hands on: the tensor it gives and the store it
# leaves, holding what the entry it is at and the entries after it state.
def ScanBody(~X: Data,~x_value: X -> U32 -> F32,~Goal: Tiled.Tiled & Models.Store -> Data,+same: Bool,+name: String,+value: X,+shape: List<&2,U32>,
  +more: List<&2,Named<X>>,+wanted: String) -> Type:
  @-given: Tiled.Tiled -> @-left: Models.Store ->
    @+given_holds: TakenHolds(~X,~x_value,Bool.pick(Maybe<&2,Named<X>>,same,Some{Named{name,value,shape}},found(~X,more,wanted)),given) ->
    @+left_holds: StoreHolds(~X,~x_value,left,Bool.pick(List<&2,Named<X>>,same,more,Named{name,value,shape} <> without(~X,more,wanted))) -> Goal((given,left))

# The scan of a take from an entry on: an entry passed over stays in front
# of what the rest of the scan leaves. Past the last entry the names left
# are none, so the failed tensor states nothing.
def scan_taken(~X: Data,~x_value: X -> U32 -> F32,~Goal: Tiled.Tiled & Models.Store -> Data,rest: Models.Store,same: Bool,+name: String,tensor: Tiled.Tiled,+wanted: String,
  +value: X,+shape: List<&2,U32>,+more: List<&2,Named<X>>,+holds: Public.Contract(~X,~x_value,value,shape,tensor),+rest_holds: StoreHolds(~X,~x_value,rest,more),
  body: ScanBody(~X,~x_value,~Goal,same,name,value,shape,more,wanted)) ->
  Goal(Models.scan(rest,same,name,tensor,wanted)):
  match rest same:
    case rest True{}: body(tensor,rest,holds,rest_holds)
    case Models.Empty{} False{}:
      body(Models.missing(wanted),Models.Entry{name,tensor,Models.Empty{}},
        %Equal.sym(List<&2,Named<X>>,more,[],rest_holds) : TakenHolds(~X,~x_value,found(~X,_,wanted),Models.missing(wanted)) Unit{},
        KernelGemm.Both{{==},KernelGemm.Both{holds,Equal.cong(List<&2,Named<X>>,List<&2,Named<X>>,entries => without(~X,entries,wanted),more,[],rest_holds)}})
    case Models.Unavailable{+problem} False{}:
      body(Models.unavailable(problem),Models.Entry{name,tensor,Models.Unavailable{problem}},
        %Equal.sym(List<&2,Named<X>>,more,[],rest_holds) : TakenHolds(~X,~x_value,found(~X,_,wanted),Models.unavailable(problem)) Unit{},
        KernelGemm.Both{{==},KernelGemm.Both{holds,Equal.cong(List<&2,Named<X>>,List<&2,Named<X>>,entries => without(~X,entries,wanted),more,[],rest_holds)}})
    case Models.Entry{+next,other,deeper} False{}:
      match more:
        case Nil{}: Empty.absurd(Goal(Models.scan(Models.Entry{next,other,deeper},False{},name,tensor,wanted)),rest_holds)
        case Con{Named{+next_name,+next_value,+next_shape},+further}:
          KernelGemm.Both{+named,KernelGemm.Both{+next_holds,+deeper_holds}} = rest_holds
          +renamed = Equal.sym(String,next_name,next,named)
          scan_taken(~X,~x_value,~(pair => Goal(Models.passed(name,tensor,pair))),deeper,String.eq(next,wanted),next,other,wanted,next_value,next_shape,further,
            next_holds,deeper_holds,
            given => left => given_holds => left_holds =>
              body(given,Models.Entry{name,tensor,left},
                %renamed : TakenHolds(~X,~x_value,found(~X,Named{_,next_value,next_shape} <> further,wanted),given) given_holds,
                KernelGemm.Both{{==},KernelGemm.Both{holds,%renamed : StoreHolds(~X,~x_value,left,without(~X,Named{_,next_value,next_shape} <> further,wanted)) left_holds}}))

# take of a name from a store that holds its entries gives the first entry
# of the name, holding its values, and the store holds the other entries.
def take(~X: Data,~x_value: X -> U32 -> F32,store: Models.Store,+entries: List<&2,Named<X>>,+wanted: String,+holds: StoreHolds(~X,~x_value,store,entries)) ->
  Taken(~X,~x_value,entries,wanted,Models.take(wanted)(store)):
  match store:
    case Models.Empty{}:
      %Equal.sym(List<&2,Named<X>>,entries,[],holds) : Taken(~X,~x_value,_,wanted,(Models.missing(wanted),Models.Empty{}))
      KernelGemm.Both{Unit{},{==}}
    case Models.Unavailable{+problem}:
      %Equal.sym(List<&2,Named<X>>,entries,[],holds) : Taken(~X,~x_value,_,wanted,(Models.unavailable(problem),Models.Unavailable{problem}))
      KernelGemm.Both{Unit{},{==}}
    case Models.Entry{+name,tensor,rest}:
      match entries:
        case Nil{}: Empty.absurd(Taken(~X,~x_value,[],wanted,Models.take(wanted)(Models.Entry{name,tensor,rest})),holds)
        case Con{Named{+entry_name,+value,+shape},+more}:
          KernelGemm.Both{+named,KernelGemm.Both{+tensor_holds,+rest_holds}} = holds
          %Equal.sym(String,entry_name,name,named) : Taken(~X,~x_value,Named{_,value,shape} <> more,wanted,Models.take(wanted)(Models.Entry{name,tensor,rest}))
          scan_taken(~X,~x_value,~(pair => Taken(~X,~x_value,Named{name,value,shape} <> more,wanted,pair)),rest,String.eq(name,wanted),name,tensor,wanted,value,shape,more,
            tensor_holds,rest_holds,given => left => given_holds => left_holds => KernelGemm.Both{given_holds,left_holds})

# What keep gives and leaves: the tensor and the store with its copy in
# front.
def Kept(~X: Data,~x_value: X -> U32 -> F32,+name: String,+value: X,+shape: List<&2,U32>,+entries: List<&2,Named<X>>,result: Tiled.Tiled & Models.Store) -> Data:
  (given,store) = result
  KernelGemm.Both<Public.Contract(~X,~x_value,value,shape,given),StoreHolds(~X,~x_value,store,Named{name,value,shape} <> entries)>

# keep of a tensor that holds its values gives it back and stores a copy in
# front of the entries.
def keep(~X: Data,~x_value: X -> U32 -> F32,+name: String,+value: X,+shape: List<&2,U32>,tensor: Tiled.Tiled,-store: Models.Store,+entries: List<&2,Named<X>>,
  +holds: Public.Contract(~X,~x_value,value,shape,tensor),+store_holds: StoreHolds(~X,~x_value,store,entries)) ->
  Kept(~X,~x_value,name,value,shape,entries,Models.keep(name,tensor)(store)):
  CopyContract.tiled_copied(~X,~x_value,~(pair => Kept(~X,~x_value,name,value,shape,entries,Models.kept(name,pair,store))),value,shape,tensor,holds,
    first => second => first_holds => second_holds => KernelGemm.Both{first_holds,KernelGemm.Both{{==},KernelGemm.Both{second_holds,store_holds}}})
