# The entries make the tiled form of the values they are given: from_storage
# holds the values of the storage and from_list the values its list writes,
# both at their flat index in a nonempty counted shape. read_storage reads one
# value of such storage.
import Base
import ../tensor.bend as Tensors
import ../tensor_f32.bend as F32Tensor
import ../tensor_tiled.bend as Tiled
import ../storage_buffer.bend as Storage
import ./storage_buffer_model.bend as Buffer
import ./tensor_contract.bend as Contract
import ./tensor_global_shape_count.bend as Global
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_tiled_public.bend as Public
import ./tensor_tiled_entry.bend as Entry
import ./nat_order.bend as Order

# The values a list writes into the buffer of a count.
def list_buffer(+values: List<&2,F32>,+count: U32) -> Storage.Buffer<F32>:
  Storage.from_list(~F32,values,Tensors.depth_for(count),0.0,U32.to_nat(count))

# F.from_storage of storage that holds a model of as many values as a nonempty
# counted shape holds the values of the storage.
def tiled_from_storage(+shape: List<&2,U32>,+model: Buffer.Model<F32>,+counted: Global.Counted<shape>,
  +matches: {Buffer.length(~F32,model) == U32.to_nat(Global.count(shape,counted)) : Nat},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool}) ->
  Public.Contract(~Buffer.Content<F32>,~Contract.stored(~F32),Buffer.content(~F32,model),shape,F32Tensor.from_storage(shape,Buffer.pack(~F32,model))):
  Buffer.Model{+tree,+length,+certificate} = model
  Global.Counted{+count,+accepted} = counted
  +source = {Buffer.Model{tree,length,certificate} : Buffer.Model<F32>}
  +classified = {%matches : {Nat.is_eq(length,_) == True{} : Bool} Order.equal_reflexive(length) : {Nat.is_eq(length,U32.to_nat(count)) == True{} : Bool}}
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(shape),Some{count},accepted) :
    Public.Contract(~Buffer.Content<F32>,~Contract.stored(~F32),Buffer.content(~F32,source),shape,
      Tiled.tiled(Tensors.storage_of(~F32,shape,_,Buffer.pack(~F32,source),0.0)))
  %Equal.sym(Bool,Nat.is_eq(length,U32.to_nat(count)),True{},classified) :
    Public.Contract(~Buffer.Content<F32>,~Contract.stored(~F32),Buffer.content(~F32,source),shape,Tiled.tiled(Tensors.stored(~F32,_,Buffer.pack(~F32,source),shape,0.0)))
  %Buffer.packed_contents(~F32,source) :
    Public.Contract(~Buffer.Content<F32>,~Contract.stored(~F32),_,shape,Tiled.tiled_shape(Tensors.Usable{},shape,1,Buffer.pack(~F32,source)))
  Entry.tiled_stored(shape,1,source,Global.Counted{count,accepted},nonempty)

# F.from_list of as many values as a nonempty counted shape holds the values
# the list writes into its buffer.
def tiled_from_list(+shape: List<&2,U32>,+values: List<&2,F32>,+counted: Global.Counted<shape>,
  +matches: {List.length(&2,F32,values) == U32.to_nat(Global.count(shape,counted)) : Nat},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool}) ->
  Public.Contract(~Buffer.Content<F32>,~Contract.stored(~F32),Buffer.contents(~F32,list_buffer(values,Global.count(shape,counted))),shape,F32Tensor.from_list(shape,values)):
  Global.Counted{+count,+accepted} = counted
  +buffer_model = Buffer.remodel(~F32,list_buffer(values,count))
  +length = List.length(&2,F32,values)
  +classified = {%matches : {Nat.is_eq(length,_) == True{} : Bool} Order.equal_reflexive(length) : {Nat.is_eq(length,U32.to_nat(count)) == True{} : Bool}}
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(shape),Some{count},accepted) :
    Public.Contract(~Buffer.Content<F32>,~Contract.stored(~F32),Buffer.contents(~F32,list_buffer(values,count)),shape,
      Tiled.tiled(Tensors.values_listed(~F32,shape,_,values,0.0)))
  %Equal.sym(Bool,Nat.is_eq(length,U32.to_nat(count)),True{},classified) :
    Public.Contract(~Buffer.Content<F32>,~Contract.stored(~F32),Buffer.contents(~F32,list_buffer(values,count)),shape,
      Tiled.tiled(Tensors.values_count(~F32,_,shape,count,values,0.0)))
  %Buffer.remodeled(~F32,list_buffer(values,count)) :
    Public.Contract(~Buffer.Content<F32>,~Contract.stored(~F32),Buffer.contents(~F32,_),shape,Tiled.tiled_shape(Tensors.Usable{},shape,1,_))
  Entry.tiled_stored(shape,1,buffer_model,Global.Counted{count,accepted},nonempty)

# read_storage of storage that holds a model gives the storage back with the
# value at the index when the index is below the length, and 0.0 past it.
def read_storage(+model: Buffer.Model<F32>,+index: U32) ->
  {F32Tensor.read_storage(Buffer.pack(~F32,model),index) ==
    (Buffer.pack(~F32,model),Bool.pick(F32,U32.is_lt(index,U32.from_nat(Buffer.length(~F32,model))),Contract.stored(~F32,Buffer.content(~F32,model),index),0.0)) :
    Storage.Buffer<F32> & F32}:
  Buffer.Model{+tree,+length,+certificate} = model
  Buffer.read_masked(~F32,0.0,U32.is_lt(index,U32.from_nat(length)),Buffer.Model{tree,length,certificate},index)
