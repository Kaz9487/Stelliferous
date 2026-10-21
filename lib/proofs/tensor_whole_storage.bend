# Whole storage preserves the actual allocation and shape of a ready Normal
# tensor. Gather is a sequence of writes into its real counted allocation;
# zero extents use the same allocation and need no positive-domain premise.
import Base
import ../tensor.bend as Tensors
import ../tensor_band.bend as Bands
import ../tensor_window.bend as Window
import ./tensor_band_concat_execution.bend as BandConcatExecution
import ./tensor_global_shape_count.bend as Global
import ./tensor_band_shape_count.bend as BandShapeCount
import ./tensor_band_from_whole.bend as FromWhole
import ./tensor_count_refinement.bend as Count
import ./tensor_allocation_depth.bend as Depth
import ./tensor_contract.bend as Contract
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_band_model.bend as Model
import ./tensor_band_gather.bend as BandGather
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_allocation_refinement.bend as AllocationLaws
import ./write_program.bend as Writes
import ./tensor_band_shape_refinement.bend as BandShape
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./natural_addition.bend as Addition

# The content a gather writes into output: every band's cells at their
# row-major positions.
def gather_content(~Element: Data,+geometry: Bands.Geometry,bands: Model.Content<Element>,+output: Buffer.Content<Element>) -> Buffer.Content<Element>:
  Model.thread(~Element,~Buffer.Content<Element>,~Bands.Geometry,~BandGather.leaf_model(~Element),geometry,bands,output)

def gather_model_content(~Element: Data,+geometry: Bands.Geometry,bands: Model.Model<Element>,+output: Buffer.Content<Element>) ->
  {BandGather.model(~Element,geometry,bands,output) == gather_content(~Element,geometry,Model.content(~Element,bands),output) : Buffer.Content<Element>}:
  {==}

def fresh_counted(~Element: Data,+shape: List<&2,U32>,+fill: Element,+count: U32,
  counted: Count.Count<ShapeLaws.elements(shape),count>,allocation: Depth.Sized<count>) ->
  Contract.WellFormed<Element,shape,Buffer.contents(~Element,Tensors.buffer(~Element,fill,count))>:
  Count.Count{exact,bounded} = counted
  Depth.Sized{depth_ok,room} = allocation
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Tensors.buffer(~Element,fill,count)),Buffer.Content{Tree.fresh(Element,Tensors.depth_for(count),fill),U32.to_nat(count)},
    AllocationLaws.contents(~Element,Tensors.depth_for(count),fill,U32.to_nat(count))) : Contract.WellFormed<Element,shape,_>
  Contract.WellFormed{Equal.sym(Nat,ShapeLaws.elements(shape),U32.to_nat(count),exact),
    Buffer.content_ready(~Element,Buffer.fresh(~Element,Tensors.depth_for(count),fill,U32.to_nat(count)),
      Buffer.fresh_ready(~Element,Tensors.depth_for(count),fill,U32.to_nat(count),depth_ok,room))}

def counted_allocation(+shape: List<&2,U32>,+count: U32,counted: Count.Count<ShapeLaws.elements(shape),count>) -> Depth.Sized<count>:
  Count.Count{exact,bounded} = counted
  Depth.bounded(count,bounded)

def fresh(~Element: Data,+shape: List<&2,U32>,+fill: Element,+counted: Global.Counted<shape>) ->
  Contract.WellFormed<Element,shape,Buffer.contents(~Element,Tensors.buffer(~Element,fill,Global.count(shape,counted)))>:
  fresh_counted(~Element,shape,fill,Global.count(shape,counted),Global.evidence(shape,counted),
    counted_allocation(shape,Global.count(shape,counted),Global.evidence(shape,counted)))

def gathered_model(~Element: Data,+geometry: Bands.Geometry,+bands: Model.Model<Element>,+shape: List<&2,U32>,output: Buffer.Content<Element>,
  well_formed: Contract.WellFormed<Element,shape,output>) ->
  Contract.WellFormed<Element,shape,gather_content(~Element,geometry,Model.content(~Element,bands),output)>:
  Buffer.Content{+tree,+length} = output
  Contract.WellFormed{length_exact,ready} = well_formed
  %gather_model_content(~Element,geometry,bands,Buffer.Content{tree,length}) : Contract.WellFormed<Element,shape,_>
  %Equal.sym(Buffer.Content<Element>,BandGather.model(~Element,geometry,bands,Buffer.Content{tree,length}),
    Buffer.Content{Writes.pure(Element,BandGather.refinement_writes(~Element,geometry,bands),tree),length},BandGather.model_writes(~Element,geometry,bands,tree,length)) : Contract.WellFormed<Element,shape,_>
  Contract.WellFormed{length_exact,Writes.content_ready(~Element,BandGather.refinement_writes(~Element,geometry,bands),tree,length,ready)}

def gathered_contents(~Element: Data,+geometry: Bands.Geometry,+bands: Model.Model<Element>,+fill: Element) ->
  {Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry)))) ==
    gather_content(~Element,geometry,Model.content(~Element,bands),Buffer.contents(~Element,Tensors.buffer(~Element,fill,Bands.elements(geometry)))) : Buffer.Content<Element>}:
  Equal.trans(Buffer.Content<Element>,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry)))),
    BandGather.model(~Element,geometry,bands,Buffer.contents(~Element,Tensors.buffer(~Element,fill,Bands.elements(geometry)))),
    gather_content(~Element,geometry,Model.content(~Element,bands),Buffer.contents(~Element,Tensors.buffer(~Element,fill,Bands.elements(geometry)))),
    BandGather.gather(~Element,geometry,bands,Tensors.buffer(~Element,fill,Bands.elements(geometry))),
    gather_model_content(~Element,geometry,bands,Buffer.contents(~Element,Tensors.buffer(~Element,fill,Bands.elements(geometry)))))

def gathered(~Element: Data,+geometry: Bands.Geometry,+bands: Model.Model<Element>,+shape: List<&2,U32>,+fill: Element,+counted: Global.Counted<shape>,
  coherent: Global.Coherent(shape,geometry)) ->
  Contract.WellFormed<Element,shape,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry))))>:
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry)))),
    gather_content(~Element,geometry,Model.content(~Element,bands),Buffer.contents(~Element,Tensors.buffer(~Element,fill,Bands.elements(geometry)))),gathered_contents(~Element,geometry,bands,fill)) : Contract.WellFormed<Element,shape,_>
  gathered_model(~Element,geometry,bands,shape,Buffer.contents(~Element,Tensors.buffer(~Element,fill,Bands.elements(geometry))),
    %Equal.sym(U32,Bands.elements(geometry),Global.count(shape,counted),BandShapeCount.global_element_count_exact(shape,geometry,counted,coherent)) :
      Contract.WellFormed<Element,shape,Buffer.contents(~Element,Tensors.buffer(~Element,fill,_))>
    fresh(~Element,shape,fill,counted))

# The cells of one image are its elements.
def image_elements(+channels: U32,+height: U32,+width: U32,+length: Nat,
  exact: {length == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width))) : Nat}) ->
  {length == BandGather.elements(Bands.Geometry{1,channels,height,width}) : Nat}:
  +elements = Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))
  Equal.trans(Nat,length,elements,BandGather.elements(Bands.Geometry{1,channels,height,width}),exact,
    Equal.sym(Nat,BandGather.elements(Bands.Geometry{1,channels,height,width}),elements,Addition.zero_right(elements)))

# The whole storage of one leaf: the leaf itself when it holds the whole image,
# otherwise the gathered allocation.
def leaf_well_formed(~Element: Data,+whole: Bool,+geometry: Bands.Geometry,+buffer: Buffer.Model<Element>,+first: U32,+rows: U32,+shape: List<&2,U32>,+fill: Element,
  +counted: Global.Counted<shape>,+held: {Bands.whole_image(geometry,first,rows) == whole : Bool},coherent: Global.Coherent(shape,geometry),
  metadata: BandShape.WellFormed<Element,geometry,Model.content(~Element,Model.Leaf{buffer,first,rows})>) ->
  Contract.WellFormed<Element,shape,Buffer.contents(~Element,Tensors.leaf_whole_storage(~Element,whole,geometry,fill,Buffer.pack(~Element,buffer),first,rows))>:
  match whole:
    case False{}: gathered(~Element,geometry,Model.Leaf{buffer,first,rows},shape,fill,counted,coherent)
    case True{}:
      Bands.Geometry{+batch,+channels,+height,+width} = geometry
      BandShape.WellFormed{fields,ready,coverage} = metadata
      BandShape.Fields{local,rows_fit,length_exact} = fields
      +batch_one = FromWhole.held_batch(batch,first,rows,height,held)
      +rows_height = FromWhole.held_rows(batch,first,rows,height,held)
      +length = Buffer.content_length(~Element,Buffer.content(~Element,buffer))
      %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,buffer)),Buffer.content(~Element,buffer),Buffer.packed_contents(~Element,buffer)) : Contract.WellFormed<Element,shape,_>
      Contract.WellFormed{
        Equal.trans(Nat,length,BandGather.elements(Bands.Geometry{1,channels,height,width}),ShapeLaws.elements(shape),
          image_elements(channels,height,width,length,
            %rows_height : {length == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(_),U32.to_nat(width))) : Nat} length_exact),
          Equal.sym(Nat,ShapeLaws.elements(shape),BandGather.elements(Bands.Geometry{1,channels,height,width}),
            %batch_one : Global.Coherent(shape,Bands.Geometry{_,channels,height,width}) coherent)),
        ready}

def well_formed(~Element: Data,+geometry: Bands.Geometry,+bands: Model.Model<Element>,+shape: List<&2,U32>,+fill: Element,+counted: Global.Counted<shape>,
  coherent: Global.Coherent(shape,geometry),metadata: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>) ->
  Contract.WellFormed<Element,shape,Buffer.contents(~Element,Tensors.whole_storage(~Element,geometry,fill,Model.pack(~Element,bands)))>:
  match bands:
    case Model.Fork{+left,+right}: gathered(~Element,geometry,Model.Fork{left,right},shape,fill,counted,coherent)
    case Model.Leaf{+buffer,+first,+rows}:
      leaf_well_formed(~Element,Bands.whole_image(geometry,first,rows),geometry,buffer,first,rows,shape,fill,counted,{==},coherent,metadata)

