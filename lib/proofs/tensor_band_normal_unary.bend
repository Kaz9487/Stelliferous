# Maps of one tensor, with or without an operation context, and an operation
# with a scalar keep the contract: the result holds the operation of the input
# value at every index, in any layout.
# An indexed map keeps the result Normal.
import Base
import ../tensor.bend as Tensors
import ../tensor_band.bend as Bands
import ../storage_buffer.bend as Storage
import ./tensor_operations_refinement.bend as OperationLaws
import ./tensor_band_model.bend as Model
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_map.bend as BandMap
import ./tensor_band_indexed.bend as BandIndexed
import ./tensor_global_shape_count.bend as Global
import ./storage_shape.bend as Shape
import ./tensor_band_values.bend as BandValues
import ../tensor_operations.bend as Operations
import ../tensor_access.bend as Access
import ./tensor_contract.bend as Contract

def apply_context(~Element: Data,~Context: Data,~Logical: Data,~operation: Context -> Element -> Element,
  ~mapper: @+count: Nat -> @+position: U32 -> @+context: Context -> Storage.Buffer<Element> -> Storage.Buffer<Element>,~lawful: OperationLaws.MapLaw_context(~Element,~Context,~operation,~mapper),
  ~original: Logical -> U32 -> Element,+context: Context,+logical: Logical,+operations: F32,+bands: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})) ->
  Global.Contract(~Element,~BandMap.Context<Context,Logical>,~BandMap.value(~Element,~Context,~Logical,~operation,~original),BandMap.Context{context,logical},
    Tensors.apply_context(~Element,~Context,~mapper,context,operations,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})):
  Shape.Both{Shape.Both{+well_formed,Shape.Both{+counted,+coherent}},values} = contract
  +source_shape = {%Model.packed_contents(~Element,bands) : BandShape.WellFormed<Element,geometry,_> well_formed : BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>}
  Shape.Both{
    Shape.Both{BandMap.scheduled_map_well_formed_context(~Element,~Context,~operation,~mapper,~lawful,context,operations,geometry,workers,bands,source_shape),Shape.Both{counted,coherent}},
    Shape.second(BandMap.WellFormed(~Element,Tensors.apply_context(~Element,~Context,~mapper,context,operations,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})),
      BandMap.LogicalValues(~Element,~BandMap.Context<Context,Logical>,~BandMap.value(~Element,~Context,~Logical,~operation,~original),BandMap.Context{context,logical},
        Tensors.apply_context(~Element,~Context,~mapper,context,operations,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})),
      BandMap.apply_ready(~Element,~Context,~Logical,~operation,~mapper,~lawful,~original,context,operations,logical,geometry,shape,fill,workers,bands,source_shape,
        %Model.packed_contents(~Element,bands) : BandValues.Values(~Element,~Logical,~original,logical,geometry,_) values))}

def apply(~Element: Data,~Logical: Data,~operation: Element -> Element,~mapper: @+count: Nat -> @+position: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element>,~lawful: OperationLaws.MapLaw(~Element,~operation,~mapper),
  ~original: Logical -> U32 -> Element,+logical: Logical,+operations: F32,+bands: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})) ->
  Global.Contract(~Element,~BandMap.Context<Unit,Logical>,~BandMap.value(~Element,~Unit,~Logical,~Access.apply_element(~Element,~operation),~original),BandMap.Context{Unit{},logical},
    Tensors.apply(~Element,~mapper,operations,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})):
  apply_context(~Element,~Unit,~Logical,~Access.apply_element(~Element,~operation),~Operations.without_context(~Element,~mapper),~OperationLaws.unit_map_law(~Element,~operation,~mapper,~lawful),
    ~original,Unit{},logical,operations,bands,geometry,shape,fill,workers,contract)

def combine_scalar(~Element: Data,~Logical: Data,~operation: Element -> Element -> Element,~original: Logical -> U32 -> Element,+logical: Logical,+operations: F32,+scalar: Element,
  +bands: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})) ->
  Global.Contract(~Element,~BandMap.Context<Element,Logical>,~BandMap.value(~Element,~Element,~Logical,~Bands.scalar_value(~Element,~operation),~original),BandMap.Context{scalar,logical},
    Tensors.combine_scalar(~Element,~operation,operations,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},scalar)):
  Shape.Both{Shape.Both{+well_formed,Shape.Both{+counted,+coherent}},values} = contract
  +source_shape = {%Model.packed_contents(~Element,bands) : BandShape.WellFormed<Element,geometry,_> well_formed : BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>}
  Shape.Both{
    Shape.Both{BandMap.scheduled_scalar_well_formed(~Element,~operation,operations,scalar,geometry,workers,bands,source_shape),Shape.Both{counted,coherent}},
    Shape.second(BandMap.WellFormed(~Element,Tensors.combine_scalar(~Element,~operation,operations,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},scalar)),
      BandMap.LogicalValues(~Element,~BandMap.Context<Element,Logical>,~BandMap.value(~Element,~Element,~Logical,~Bands.scalar_value(~Element,~operation),~original),BandMap.Context{scalar,logical},
        Tensors.combine_scalar(~Element,~operation,operations,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},scalar)),
      BandMap.scalar_ready(~Element,~Logical,~operation,~original,operations,logical,geometry,shape,fill,workers,bands,scalar,source_shape,
        %Model.packed_contents(~Element,bands) : BandValues.Values(~Element,~Logical,~original,logical,geometry,_) values))}

# An operation with a scalar keeps the status of its tensor.
def scalar_status(~Element: Data,~operation: Element -> Element -> Element,+operations: F32,tensor: Tensors.Tensor<Element>,+scalar: Element,
  ready: {Contract.tensor_ready(~Element,Tensors.combine_scalar(~Element,~operation,operations,tensor,scalar)) == True{} : Bool}) ->
  {Contract.tensor_ready(~Element,tensor) == True{} : Bool}:
  match tensor:
    case Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}: {==}
    case Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers}: ready

def indexed_ready(~Element: Data,~operation: U32 -> Element -> Element,+operations: F32,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,+source: Model.Model<Element>,
  +width_positive: {BandShape.width_positive(geometry) == True{} : Bool},source_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,source)>,counted: Global.Counted<shape>,coherent: Global.Coherent(shape,geometry)) ->
  Global.Normal(~Element,Tensors.apply_indexed(~Element,~operation,operations,Tensors.Tensor{Model.pack(~Element,source),geometry,shape,fill,Tensors.Usable{},workers})):
  %Equal.sym(Tensors.Tensor<Element>,Tensors.apply_indexed(~Element,~operation,operations,Tensors.Tensor{Model.pack(~Element,source),geometry,shape,fill,Tensors.Usable{},workers}),
    Tensors.apply_indexed_positive(~Element,~operation,True{},operations,Model.pack(~Element,source),geometry,shape,fill,workers),
    BandMap.indexed_selected(~Element,~operation,operations,geometry,shape,fill,workers,Model.pack(~Element,source),width_positive)) : Global.Normal(~Element,_)
  Shape.Both{BandIndexed.scheduled_well_formed(~Element,~operation,operations,geometry,workers,fill,source,width_positive,source_shape),Shape.Both{counted,coherent}}

# An empty width holds no element and keeps the bands.
def indexed_empty(~Element: Data,~operation: U32 -> Element -> Element,+operations: F32,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,bands: Bands.Bands(~Element),
  rejected: {Tensors.indexed_width_positive(geometry) == False{} : Bool}) ->
  {Tensors.apply_indexed(~Element,~operation,operations,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}) ==
    Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers} : Tensors.Tensor<Element>}:
  %Equal.sym(Bool,Tensors.indexed_width_positive(geometry),False{},rejected) :
    {Tensors.apply_indexed_positive(~Element,~operation,_,operations,bands,geometry,shape,fill,workers) ==
      Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers} : Tensors.Tensor<Element>}
  {==}
