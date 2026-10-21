# reshape keeps the bands, their geometry and their values, and changes only
# the shape.
import Base
import ../tensor.bend as Tensors
import ../tensor_layout.bend as Layout
import ../tensor_band.bend as Bands
import ./tensor_contract.bend as Contract
import ./tensor_global_shape_count.bend as Global
import ./tensor_band_model.bend as Model
import ./tensor_band_gather.bend as BandGather
import ./tensor_count_refinement.bend as Count
import ./tensor_shape_refinement.bend as ShapeLaws
import ./storage_shape.bend as Shape
import ./word_encoding.bend as WordEncoding
import ./bounded_u32_arithmetic.bend as Arithmetic

def counts(+source: Maybe<&2,U32>,+target: Maybe<&2,U32>,+source_shape: List<&2,U32>,+target_shape: List<&2,U32>,
  +sources: {Tensors.count_of(source_shape) == source : Maybe<&2,U32>},+targets: {Tensors.count_of(target_shape) == target : Maybe<&2,U32>},
  same: {Layout.same_count(source,target) == True{} : Bool}) -> {ShapeLaws.elements(source_shape) == ShapeLaws.elements(target_shape) : Nat}:
  match source target:
    case Some{+left} Some{+right}:
      +left_exact = Count.count_exact(ShapeLaws.elements(source_shape),left,Count.count_of(source_shape,left,sources))
      +right_exact = Count.count_exact(ShapeLaws.elements(target_shape),right,Count.count_of(target_shape,right,targets))
      Equal.trans(Nat,ShapeLaws.elements(source_shape),U32.to_nat(left),ShapeLaws.elements(target_shape),left_exact,
        Equal.trans(Nat,U32.to_nat(left),U32.to_nat(right),ShapeLaws.elements(target_shape),
          Equal.cong(U32,Nat,U32.to_nat,left,right,Arithmetic.equal_of_is_eq(left,right,same)),Equal.sym(Nat,ShapeLaws.elements(target_shape),U32.to_nat(right),right_exact)))
    case None{} None{}: Empty.absurd({ShapeLaws.elements(source_shape) == ShapeLaws.elements(target_shape) : Nat},WordEncoding.false_true(same))
    case None{} Some{right}: Empty.absurd({ShapeLaws.elements(source_shape) == ShapeLaws.elements(target_shape) : Nat},WordEncoding.false_true(same))
    case Some{left} None{}: Empty.absurd({ShapeLaws.elements(source_shape) == ShapeLaws.elements(target_shape) : Nat},WordEncoding.false_true(same))

# A count the source shares is a count of the target.
def counted(+source: Maybe<&2,U32>,+target: Maybe<&2,U32>,+target_shape: List<&2,U32>,+targets: {Tensors.count_of(target_shape) == target : Maybe<&2,U32>},
  same: {Layout.same_count(source,target) == True{} : Bool}) -> Global.Counted<target_shape>:
  match source target:
    case Some{left} Some{+right}: Global.Counted{right,targets}
    case None{} None{}: Empty.absurd(Global.Counted<target_shape>,WordEncoding.false_true(same))
    case None{} Some{right}: Empty.absurd(Global.Counted<target_shape>,WordEncoding.false_true(same))
    case Some{left} None{}: Empty.absurd(Global.Counted<target_shape>,WordEncoding.false_true(same))

def relabeled(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,same: Bool,+logical: Logical,+bands: Model.Model<Element>,+geometry: Bands.Geometry,
  +source: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32,
  +computed: {Layout.same_count(Tensors.count_of(source),Tensors.count_of(target)) == same : Bool},
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,bands),geometry,source,fill,Tensors.Usable{},workers}),
  accepted: {Contract.tensor_ready(~Element,Layout.relabel_ready(~Element,same,Model.pack(~Element,bands),geometry,source,target,fill,workers)) == True{} : Bool}) ->
  Global.Shaped(~Element,~Logical,~original,logical,target,Layout.relabel_ready(~Element,same,Model.pack(~Element,bands),geometry,source,target,fill,workers)):
  match same:
    case True{}:
      Shape.Both{Shape.Both{well_formed,Shape.Both{source_counted,coherent}},values} = contract
      +same_elements = counts(Tensors.count_of(source),Tensors.count_of(target),source,target,{==},{==},computed)
      Global.shaped(~Element,~Logical,~original,logical,target,Tensors.Tensor{Model.pack(~Element,bands),geometry,target,fill,Tensors.Usable{},workers},accepted,{==},
        Shape.Both{Shape.Both{well_formed,Shape.Both{counted(Tensors.count_of(source),Tensors.count_of(target),target,{==},computed),
          Equal.trans(Nat,ShapeLaws.elements(target),ShapeLaws.elements(source),BandGather.elements(geometry),
            Equal.sym(Nat,ShapeLaws.elements(source),ShapeLaws.elements(target),same_elements),coherent)}},values})
    case False{}:
      Empty.absurd(Global.Shaped(~Element,~Logical,~original,logical,target,Layout.relabel_ready(~Element,False{},Model.pack(~Element,bands),geometry,source,target,fill,workers)),
        WordEncoding.false_true(accepted))

# reshape of an input of any layout: when the call returns Ready, the result
# has the target shape and holds the input's values at every index.
def reshape(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+target: List<&2,U32>,
  +bands: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}),
  accepted: {Contract.tensor_ready(~Element,Layout.reshape(~Element,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},target)) == True{} : Bool}) ->
  Global.Shaped(~Element,~Logical,~original,logical,target,Layout.reshape(~Element,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},target)):
  relabeled(~Element,~Logical,~original,Layout.same_count(Tensors.count_of(shape),Tensors.count_of(target)),logical,bands,geometry,shape,target,fill,workers,{==},contract,accepted)
