# Transpose followed by flip of a tensor of any layout: the public transpose
# result read by the public flip.
import Base
import ../tensor.bend as Tensors
import ../tensor_layout.bend as Layout
import ../tensor_view.bend as Views
import ../tensor_band.bend as Bands
import ./tensor_band_model.bend as Model
import ./tensor_contract.bend as Contract
import ./tensor_gather_values.bend as Gather
import ./tensor_gather_public.bend as GatherPublic
import ./tensor_global_shape_count.bend as Global
import ./word_encoding.bend as Words

# A gather of a failed tensor is not Ready, whatever its view.
def failed_gather(~Element: Data,source: Maybe<&2,Views.View>,+reversed: List<&2,Bool>,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,
  +fill: Element,+workers: U32,+problem: Tensors.Problem,
  accepted: {Contract.tensor_ready(~Element,Tensors.gather_extents(~Element,"flip",Tensors.Failed{problem},source,reversed,bands,geometry,shape,fill,workers)) == True{} : Bool}) ->
  {False{} == True{} : Bool}:
  match source:
    case Some{Views.View{axes,offset}}: accepted
    case None{}: accepted

# Flip keeps a failure, so a Ready flip had a Ready argument.
def flipped_ready(~Element: Data,+axes: List<&2,U32>,tensor: Tensors.Tensor<Element>,
  accepted: {Contract.tensor_ready(~Element,Layout.flip(~Element,tensor,axes)) == True{} : Bool}) ->
  {Contract.tensor_ready(~Element,tensor) == True{} : Bool}:
  Tensors.Tensor{bands,geometry,+shape,+fill,status,+workers} = tensor
  match status:
    case Tensors.Usable{}: {==}
    case Tensors.Failed{+problem}:
      Empty.absurd({Contract.tensor_ready(~Element,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers}) == True{} : Bool},
        Words.false_true(failed_gather(~Element,Layout.flipped(Views.unique_axes(Layout.naturals(axes)) && Layout.all_below(axes,List.length(&2,U32,shape)),Tensors.view_of(shape)),
          Layout.reversed_flags(List.length(&2,U32,shape),0,axes),bands,geometry,shape,fill,workers,problem,accepted)))

def flipped_usable(~Element: Data,+axes: List<&2,U32>,tensor: Tensors.Tensor<Element>,
  accepted: {Contract.tensor_ready(~Element,Layout.flip(~Element,tensor,axes)) == True{} : Bool}) ->
  Global.Usable<Element,Contract.extents(~Element,tensor),tensor>:
  Tensors.Tensor{bands,+geometry,+shape,+fill,status,+workers} = tensor
  match status:
    case Tensors.Usable{}:
      Global.usable_observed(~Element,bands,geometry,shape,fill,Tensors.Usable{},workers,Model.observe(~Element,bands),{==})
    case Tensors.Failed{+problem}:
      Empty.absurd(Global.Usable<Element,shape,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers}>,
        Words.false_true(failed_gather(~Element,Layout.flipped(Views.unique_axes(Layout.naturals(axes)) && Layout.all_below(axes,List.length(&2,U32,shape)),Tensors.view_of(shape)),
          Layout.reversed_flags(List.length(&2,U32,shape),0,axes),bands,geometry,shape,fill,workers,problem,accepted)))

# The result reads the logical input through the transposed view with the
# listed axes of the transposed extents reflected.
def Contract(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,source: Maybe<&2,Views.View>,+axes: List<&2,U32>,
  transposed: Tensors.Tensor<Element>) -> Data:
  match source:
    case Some{source}:
      Tensors.Tensor{bands,+geometry,+shape,+fill,+status,+workers} = transposed
      GatherPublic.Directed(~Element,~Gather.Gather<Logical>,~Gather.value(~Element,~Logical,~original),Gather.Gather{logical,source},
        Tensors.view_of(shape),Layout.reversed_flags(List.length(&2,U32,shape),0,axes),
        Layout.flip(~Element,Tensors.Tensor{bands,geometry,shape,fill,status,workers},axes))
    case None{}: Empty

def held(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+source: Views.View,+axes: List<&2,U32>,+shape: List<&2,U32>,
  -tensor: Tensors.Tensor<Element>,usable: Global.Usable<Element,shape,tensor>,
  contract: Global.Contract(~Element,~Gather.Gather<Logical>,~Gather.value(~Element,~Logical,~original),Gather.Gather{logical,source},tensor),
  accepted: {Contract.tensor_ready(~Element,Layout.flip(~Element,tensor,axes)) == True{} : Bool}) ->
  GatherPublic.Directed(~Element,~Gather.Gather<Logical>,~Gather.value(~Element,~Logical,~original),Gather.Gather{logical,source},
    Tensors.view_of(shape),Layout.reversed_flags(List.length(&2,U32,shape),0,axes),Layout.flip(~Element,tensor,axes)):
  Global.Usable{+bands,+geometry,+fill,+workers,+equation} = usable
  %equation : GatherPublic.Directed(~Element,~Gather.Gather<Logical>,~Gather.value(~Element,~Logical,~original),Gather.Gather{logical,source},
    Tensors.view_of(shape),Layout.reversed_flags(List.length(&2,U32,shape),0,axes),Layout.flip(~Element,_,axes))
  GatherPublic.flip(~Element,~Gather.Gather<Logical>,~Gather.value(~Element,~Logical,~original),Gather.Gather{logical,source},axes,Model.pack(~Element,bands),geometry,shape,fill,workers,
    %Equal.sym(Tensors.Tensor<Element>,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},tensor,equation) :
      Global.Contract(~Element,~Gather.Gather<Logical>,~Gather.value(~Element,~Logical,~original),Gather.Gather{logical,source},_) contract,
    %Equal.sym(Tensors.Tensor<Element>,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},tensor,equation) :
      {Contract.tensor_ready(~Element,Layout.flip(~Element,_,axes)) == True{} : Bool} accepted)

def flipped(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,source: Maybe<&2,Views.View>,+axes: List<&2,U32>,
  transposed: Tensors.Tensor<Element>,viewed: GatherPublic.Viewed(~Element,~Logical,~original,logical,source,transposed),
  +accepted: {Contract.tensor_ready(~Element,Layout.flip(~Element,transposed,axes)) == True{} : Bool}) ->
  Contract(~Element,~Logical,~original,logical,source,axes,transposed):
  match source:
    case Some{+source}:
      Tensors.Tensor{bands,+geometry,+shape,+fill,+status,+workers} = transposed
      held(~Element,~Logical,~original,logical,source,axes,shape,Tensors.Tensor{bands,geometry,shape,fill,status,workers},
        flipped_usable(~Element,axes,Tensors.Tensor{bands,geometry,shape,fill,status,workers},accepted),viewed,accepted)
    case None{}: viewed

# Transpose then flip of a tensor of the contract: when the call returns
# Ready, the result holds at every index the input at the address the
# transposed view gives to the digits of that index with the digit of every
# listed axis reflected.
def transpose_flip(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+order: List<&2,U32>,+axes: List<&2,U32>,
  +bands: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}),
  +accepted: {Contract.tensor_ready(~Element,Layout.flip(~Element,Layout.transpose(~Element,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},order),axes)) == True{} : Bool}) ->
  Contract(~Element,~Logical,~original,logical,Layout.permuted(Tensors.view_of(shape),Layout.naturals(order)),axes,
    Layout.transpose(~Element,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},order)):
  flipped(~Element,~Logical,~original,logical,Layout.permuted(Tensors.view_of(shape),Layout.naturals(order)),axes,
    Layout.transpose(~Element,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},order),
    GatherPublic.transpose(~Element,~Logical,~original,logical,order,Model.pack(~Element,bands),geometry,shape,fill,workers,contract,
      flipped_ready(~Element,axes,Layout.transpose(~Element,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},order),accepted)),
    accepted)
