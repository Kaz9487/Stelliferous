# Band concatenation, as owners: the tensors and parts a proof observes, and
# the tree the concat gathers from the bands of its parts, one piece for every
# band of the output.
import Base
import ./storage_buffer_model.bend as Buffer
import ./tensor_band_model.bend as Model
import ../tensor_band.bend as Bands
import ../tensor_band_concat.bend as BandConcat
import ../tensor.bend as Tensors
import ../storage_buffer.bend as Storage
import ../traversal_partition.bend as Partition
import ./tensor_band_each.bend as Each
import ./boolean_logic.bend as Boolean
import ./word_encoding.bend as WordEncoding

# Contents retain tensor metadata while erasing only the certificates.
type TensorContent<-Element: Data> is Data:
  TensorContent{bands: Model.Content<Element>,geometry: Bands.Geometry,shape: List<&2,U32>,fill: Element,status: Tensors.Status,workers: U32}

def tensor_contents(~Element: Data,input: Tensors.Tensor<Element>) -> TensorContent<Element>:
  match input:
    case Tensors.Tensor{bands,geometry,shape,fill,status,workers}: TensorContent{Model.contents(~Element,bands),geometry,shape,fill,status,workers}

# Proof observations keep the bands of a tensor and their actual
# certificates. Each affine owner is observed once before proof reuse.
type TensorOwnerModel<-Element: Data> is Data:
  TensorOwner{bands: Model.Model<Element>,geometry: Bands.Geometry,shape: List<&2,U32>,fill: Element,status: Tensors.Status,workers: U32}

def tensor_owner_pack(~Element: Data,owner: TensorOwnerModel<Element>) -> Tensors.Tensor<Element>:
  match owner:
    case TensorOwner{bands,geometry,shape,fill,status,workers}: Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,status,workers}

type TensorObservation<-Element: Data,-tensor: Tensors.Tensor<Element>> is Data:
  TensorObservation{model: TensorOwnerModel<Element>,equation: {tensor_owner_pack(~Element,model) == tensor : Tensors.Tensor<Element>}}

def tensor_observed_bands(~Element: Data,-bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32,
  observation: Model.Observation<Element,bands>) -> TensorObservation<Element,Tensors.Tensor{bands,geometry,shape,fill,status,workers}>:
  Model.Observation{+model,equation} = observation
  TensorObservation{TensorOwner{model,geometry,shape,fill,status,workers},
    Equal.cong(Bands.Bands(~Element),Tensors.Tensor<Element>,owner => Tensors.Tensor{owner,geometry,shape,fill,status,workers},Model.pack(~Element,model),bands,equation)}

def observe_tensor(~Element: Data,tensor: Tensors.Tensor<Element>) -> TensorObservation<Element,tensor>:
  Tensors.Tensor{bands,+geometry,+shape,+fill,+status,+workers} = tensor
  tensor_observed_bands(~Element,bands,geometry,shape,fill,status,workers,Model.observe(~Element,bands))

def tensor_shape(~Element: Data,input: TensorContent<Element>) -> List<&2,U32>:
  TensorContent{bands,geometry,shape,fill,status,workers} = input
  shape

type PartContents<-Element: Data> is Data:
  SingleContent{input: TensorContent<Element>}
  ConsContent{input: TensorContent<Element>,rest: PartContents<Element>}

def parts_contents(~Element: Data,parts: Tensors.Parts<Element>) -> PartContents<Element>:
  match parts:
    case Tensors.Single{input}: SingleContent{tensor_contents(~Element,input)}
    case Tensors.Cons{input,rest}: ConsContent{tensor_contents(~Element,input),parts_contents(~Element,rest)}

type PartsOwnerModel<-Element: Data> is Data:
  SingleOwner{input: TensorOwnerModel<Element>}
  ConsOwner{input: TensorOwnerModel<Element>,rest: PartsOwnerModel<Element>}

def parts_owner_pack(~Element: Data,parts: PartsOwnerModel<Element>) -> Tensors.Parts<Element>:
  match parts:
    case SingleOwner{input}: Tensors.Single{tensor_owner_pack(~Element,input)}
    case ConsOwner{input,rest}: Tensors.Cons{tensor_owner_pack(~Element,input),parts_owner_pack(~Element,rest)}

# ---- The bands of the parts, band by band ----

# The owners of the bands of every part at one band of the output.
type Joined<-Element: Data> is Data:
  Joined{parts: PartsOwnerModel<Element>,fill: Element,valid: Bool}

def joined_pack(~Element: Data,joined: Joined<Element>) -> BandConcat.Gathered<Element>:
  Joined{parts,fill,valid} = joined
  BandConcat.Gathered{parts_owner_pack(~Element,parts),fill,valid}

def Gathered(~Element: Data) -> Data:
  Model.Shaped<Joined<Element>>

# The whole tensor of one band: one row of one channel holding the band.
def band_owner(~Element: Data,+joining: BandConcat.Joining,+extent: U32,+fill: Element,+rows: U32,buffer: Buffer.Model<Element>) -> TensorOwnerModel<Element>:
  TensorOwner{Model.Leaf{buffer,0,1},Bands.Geometry{1,1,1,Tensors.count_or_zero(BandConcat.band_shape(joining,extent,rows))},BandConcat.band_shape(joining,extent,rows),fill,Tensors.Usable{},1}

def last_bands(~Element: Data,+joining: BandConcat.Joining,+extent: U32,+fill: Element,bands: Model.Model<Element>) -> Gathered(~Element):
  match bands:
    case Model.Leaf{buffer,+first,+rows}: Model.Piece{Joined{SingleOwner{band_owner(~Element,joining,extent,fill,rows,buffer)},fill,True{}},first,rows}
    case Model.Fork{left,right}: Model.Pieces{last_bands(~Element,joining,extent,fill,left),last_bands(~Element,joining,extent,fill,right)}

def invalid_band(~Element: Data,joined: Joined<Element>) -> Joined<Element>:
  Joined{parts,fill,valid} = joined
  Joined{parts,fill,False{}}

def invalid(~Element: Data,gathered: Gathered(~Element)) -> Gathered(~Element):
  Model.shaped_map(~Joined<Element>,~Joined<Element>,~invalid_band(~Element),gathered)

def joined_band(~Element: Data,+joining: BandConcat.Joining,+extent: U32,+fill: Element,buffer: Buffer.Model<Element>,+first: U32,+rows: U32,
  other: Joined<Element>,+other_first: U32,+other_rows: U32) -> Gathered(~Element):
  Joined{parts,other_fill,valid} = other
  Model.Piece{Joined{ConsOwner{band_owner(~Element,joining,extent,fill,rows,buffer),parts},other_fill,
    valid && U32.is_eq(first,other_first) && U32.is_eq(rows,other_rows)},first,rows}

def joined_bands(~Element: Data,+joining: BandConcat.Joining,+extent: U32,+fill: Element,bands: Model.Model<Element>,gathered: Gathered(~Element)) -> Gathered(~Element):
  match bands gathered:
    case Model.Leaf{buffer,first,rows} Model.Piece{other,other_first,other_rows}:
      joined_band(~Element,joining,extent,fill,buffer,first,rows,other,other_first,other_rows)
    case Model.Fork{left,right} Model.Pieces{other_left,other_right}:
      Model.Pieces{joined_bands(~Element,joining,extent,fill,left,other_left),joined_bands(~Element,joining,extent,fill,right,other_right)}
    case bands gathered: invalid(~Element,gathered)

def last_part(~Element: Data,+joining: BandConcat.Joining,owner: TensorOwnerModel<Element>) -> Gathered(~Element):
  TensorOwner{bands,geometry,+shape,+fill,status,workers} = owner
  last_bands(~Element,joining,BandConcat.extent(joining,shape),fill,bands)

def joined_part(~Element: Data,+joining: BandConcat.Joining,owner: TensorOwnerModel<Element>,gathered: Gathered(~Element)) -> Gathered(~Element):
  TensorOwner{bands,geometry,+shape,+fill,status,workers} = owner
  joined_bands(~Element,joining,BandConcat.extent(joining,shape),fill,bands,gathered)

# One tree whose pieces hold the band owners of every part.
def gather(~Element: Data,+joining: BandConcat.Joining,sources: PartsOwnerModel<Element>) -> Gathered(~Element):
  match sources:
    case SingleOwner{input}: last_part(~Element,joining,input)
    case ConsOwner{input,rest}: joined_part(~Element,joining,input,gather(~Element,joining,rest))

def gathered_pack(~Element: Data,gathered: Gathered(~Element)) -> BandConcat.Tree(~Element):
  Model.shaped_pack(~Joined<Element>,~BandConcat.Gathered<Element>,~joined_pack(~Element),gathered)

def last_bands_packed(~Element: Data,+joining: BandConcat.Joining,+extent: U32,+fill: Element,+bands: Model.Model<Element>) ->
  {BandConcat.last_bands(~Element,joining,extent,fill,Model.pack(~Element,bands)) == gathered_pack(~Element,last_bands(~Element,joining,extent,fill,bands)) : BandConcat.Tree(~Element)}:
  match bands:
    case Model.Leaf{buffer,first,rows}: {==}
    case Model.Fork{+left,+right}:
      %Equal.sym(BandConcat.Tree(~Element),BandConcat.last_bands(~Element,joining,extent,fill,Model.pack(~Element,left)),gathered_pack(~Element,last_bands(~Element,joining,extent,fill,left)),
        last_bands_packed(~Element,joining,extent,fill,left)) :
        {Partition.Fork{_,BandConcat.last_bands(~Element,joining,extent,fill,Model.pack(~Element,right))} ==
          Partition.Fork{gathered_pack(~Element,last_bands(~Element,joining,extent,fill,left)),gathered_pack(~Element,last_bands(~Element,joining,extent,fill,right))} : BandConcat.Tree(~Element)}
      %Equal.sym(BandConcat.Tree(~Element),BandConcat.last_bands(~Element,joining,extent,fill,Model.pack(~Element,right)),gathered_pack(~Element,last_bands(~Element,joining,extent,fill,right)),
        last_bands_packed(~Element,joining,extent,fill,right)) :
        {Partition.Fork{gathered_pack(~Element,last_bands(~Element,joining,extent,fill,left)),_} ==
          Partition.Fork{gathered_pack(~Element,last_bands(~Element,joining,extent,fill,left)),gathered_pack(~Element,last_bands(~Element,joining,extent,fill,right))} : BandConcat.Tree(~Element)}
      {==}

def invalid_packed(~Element: Data,+gathered: Gathered(~Element)) ->
  {BandConcat.invalid(~Element,gathered_pack(~Element,gathered)) == gathered_pack(~Element,invalid(~Element,gathered)) : BandConcat.Tree(~Element)}:
  match gathered:
    case Model.Piece{joined,first,rows}:
      match joined:
        case Joined{parts,fill,valid}: {==}
    case Model.Pieces{+left,+right}:
      %Equal.sym(BandConcat.Tree(~Element),BandConcat.invalid(~Element,gathered_pack(~Element,left)),gathered_pack(~Element,invalid(~Element,left)),invalid_packed(~Element,left)) :
        {Partition.Fork{_,BandConcat.invalid(~Element,gathered_pack(~Element,right))} ==
          Partition.Fork{gathered_pack(~Element,invalid(~Element,left)),gathered_pack(~Element,invalid(~Element,right))} : BandConcat.Tree(~Element)}
      %Equal.sym(BandConcat.Tree(~Element),BandConcat.invalid(~Element,gathered_pack(~Element,right)),gathered_pack(~Element,invalid(~Element,right)),invalid_packed(~Element,right)) :
        {Partition.Fork{gathered_pack(~Element,invalid(~Element,left)),_} ==
          Partition.Fork{gathered_pack(~Element,invalid(~Element,left)),gathered_pack(~Element,invalid(~Element,right))} : BandConcat.Tree(~Element)}
      {==}

def joined_bands_packed(~Element: Data,+joining: BandConcat.Joining,+extent: U32,+fill: Element,+bands: Model.Model<Element>,+gathered: Gathered(~Element)) ->
  {BandConcat.joined_bands(~Element,joining,extent,fill,Model.pack(~Element,bands),gathered_pack(~Element,gathered)) ==
    gathered_pack(~Element,joined_bands(~Element,joining,extent,fill,bands,gathered)) : BandConcat.Tree(~Element)}:
  match bands gathered:
    case Model.Leaf{buffer,first,rows} Model.Piece{other,other_first,other_rows}:
      match other:
        case Joined{parts,other_fill,valid}: {==}
    case Model.Fork{+left,+right} Model.Pieces{+other_left,+other_right}:
      %Equal.sym(BandConcat.Tree(~Element),BandConcat.joined_bands(~Element,joining,extent,fill,Model.pack(~Element,left),gathered_pack(~Element,other_left)),
        gathered_pack(~Element,joined_bands(~Element,joining,extent,fill,left,other_left)),joined_bands_packed(~Element,joining,extent,fill,left,other_left)) :
        {Partition.Fork{_,BandConcat.joined_bands(~Element,joining,extent,fill,Model.pack(~Element,right),gathered_pack(~Element,other_right))} ==
          Partition.Fork{gathered_pack(~Element,joined_bands(~Element,joining,extent,fill,left,other_left)),gathered_pack(~Element,joined_bands(~Element,joining,extent,fill,right,other_right))} : BandConcat.Tree(~Element)}
      %Equal.sym(BandConcat.Tree(~Element),BandConcat.joined_bands(~Element,joining,extent,fill,Model.pack(~Element,right),gathered_pack(~Element,other_right)),
        gathered_pack(~Element,joined_bands(~Element,joining,extent,fill,right,other_right)),joined_bands_packed(~Element,joining,extent,fill,right,other_right)) :
        {Partition.Fork{gathered_pack(~Element,joined_bands(~Element,joining,extent,fill,left,other_left)),_} ==
          Partition.Fork{gathered_pack(~Element,joined_bands(~Element,joining,extent,fill,left,other_left)),gathered_pack(~Element,joined_bands(~Element,joining,extent,fill,right,other_right))} : BandConcat.Tree(~Element)}
      {==}
    case Model.Leaf{buffer,first,rows} Model.Pieces{+other_left,+other_right}: invalid_packed(~Element,Model.Pieces{other_left,other_right})
    case Model.Fork{left,right} Model.Piece{+other,+other_first,+other_rows}: invalid_packed(~Element,Model.Piece{other,other_first,other_rows})

def last_part_packed(~Element: Data,+joining: BandConcat.Joining,+owner: TensorOwnerModel<Element>) ->
  {BandConcat.last_part(~Element,joining,tensor_owner_pack(~Element,owner)) == gathered_pack(~Element,last_part(~Element,joining,owner)) : BandConcat.Tree(~Element)}:
  match owner:
    case TensorOwner{+bands,geometry,+shape,+fill,status,workers}: last_bands_packed(~Element,joining,BandConcat.extent(joining,shape),fill,bands)

def joined_part_packed(~Element: Data,+joining: BandConcat.Joining,+owner: TensorOwnerModel<Element>,+gathered: Gathered(~Element)) ->
  {BandConcat.joined_part(~Element,joining,tensor_owner_pack(~Element,owner),gathered_pack(~Element,gathered)) == gathered_pack(~Element,joined_part(~Element,joining,owner,gathered)) : BandConcat.Tree(~Element)}:
  match owner:
    case TensorOwner{+bands,geometry,+shape,+fill,status,workers}: joined_bands_packed(~Element,joining,BandConcat.extent(joining,shape),fill,bands,gathered)

# The gather of the packed parts is the pack of the gathered owners.
def gathered(~Element: Data,+joining: BandConcat.Joining,+sources: PartsOwnerModel<Element>) ->
  {BandConcat.gather(~Element,joining,parts_owner_pack(~Element,sources)) == gathered_pack(~Element,gather(~Element,joining,sources)) : BandConcat.Tree(~Element)}:
  match sources:
    case SingleOwner{+input}: last_part_packed(~Element,joining,input)
    case ConsOwner{+input,+rest}:
      %Equal.sym(BandConcat.Tree(~Element),BandConcat.gather(~Element,joining,parts_owner_pack(~Element,rest)),gathered_pack(~Element,gather(~Element,joining,rest)),gathered(~Element,joining,rest)) :
        {BandConcat.joined_part(~Element,joining,tensor_owner_pack(~Element,input),_) == gathered_pack(~Element,joined_part(~Element,joining,input,gather(~Element,joining,rest))) : BandConcat.Tree(~Element)}
      joined_part_packed(~Element,joining,input,gather(~Element,joining,rest))

# The rows of the pieces of a tree.
def gathered_plan(~Element: Data,gathered: Gathered(~Element)) -> Bands.Plan:
  Model.shaped_plan(~Joined<Element>,~Each.piece_rows(~Joined<Element>),gathered)

def last_bands_plan(~Element: Data,+joining: BandConcat.Joining,+extent: U32,+fill: Element,+bands: Model.Model<Element>) ->
  {gathered_plan(~Element,last_bands(~Element,joining,extent,fill,bands)) == Model.layout(~Element,bands) : Bands.Plan}:
  match bands:
    case Model.Leaf{buffer,first,rows}: {==}
    case Model.Fork{+left,+right}:
      %Equal.sym(Bands.Plan,gathered_plan(~Element,last_bands(~Element,joining,extent,fill,left)),Model.layout(~Element,left),last_bands_plan(~Element,joining,extent,fill,left)) :
        {Bands.Split{_,gathered_plan(~Element,last_bands(~Element,joining,extent,fill,right))} == Bands.Split{Model.layout(~Element,left),Model.layout(~Element,right)} : Bands.Plan}
      %Equal.sym(Bands.Plan,gathered_plan(~Element,last_bands(~Element,joining,extent,fill,right)),Model.layout(~Element,right),last_bands_plan(~Element,joining,extent,fill,right)) :
        {Bands.Split{Model.layout(~Element,left),_} == Bands.Split{Model.layout(~Element,left),Model.layout(~Element,right)} : Bands.Plan}
      {==}

# Bands of the plan of the tree keep its rows: every piece takes the rows of
# the band joined into it.
def joined_bands_plan(~Element: Data,+joining: BandConcat.Joining,+extent: U32,+fill: Element,+bands: Model.Model<Element>,+gathered: Gathered(~Element),
  +same: {Bands.same_plan(Model.layout(~Element,bands),gathered_plan(~Element,gathered)) == True{} : Bool}) ->
  {gathered_plan(~Element,joined_bands(~Element,joining,extent,fill,bands,gathered)) == Model.layout(~Element,bands) : Bands.Plan}:
  match bands gathered:
    case Model.Leaf{buffer,first,rows} Model.Piece{other,other_first,other_rows}:
      match other:
        case Joined{parts,other_fill,valid}: {==}
    case Model.Fork{+left,+right} Model.Pieces{+other_left,+other_right}:
      +left_same = Boolean.left(Bands.same_plan(Model.layout(~Element,left),gathered_plan(~Element,other_left)),Bands.same_plan(Model.layout(~Element,right),gathered_plan(~Element,other_right)),same)
      +right_same = Boolean.right(Bands.same_plan(Model.layout(~Element,left),gathered_plan(~Element,other_left)),Bands.same_plan(Model.layout(~Element,right),gathered_plan(~Element,other_right)),same)
      %Equal.sym(Bands.Plan,gathered_plan(~Element,joined_bands(~Element,joining,extent,fill,left,other_left)),Model.layout(~Element,left),
        joined_bands_plan(~Element,joining,extent,fill,left,other_left,left_same)) :
        {Bands.Split{_,gathered_plan(~Element,joined_bands(~Element,joining,extent,fill,right,other_right))} == Bands.Split{Model.layout(~Element,left),Model.layout(~Element,right)} : Bands.Plan}
      %Equal.sym(Bands.Plan,gathered_plan(~Element,joined_bands(~Element,joining,extent,fill,right,other_right)),Model.layout(~Element,right),
        joined_bands_plan(~Element,joining,extent,fill,right,other_right,right_same)) :
        {Bands.Split{Model.layout(~Element,left),_} == Bands.Split{Model.layout(~Element,left),Model.layout(~Element,right)} : Bands.Plan}
      {==}
    case Model.Leaf{buffer,first,rows} Model.Pieces{other_left,other_right}:
      Empty.absurd({gathered_plan(~Element,joined_bands(~Element,joining,extent,fill,Model.Leaf{buffer,first,rows},Model.Pieces{other_left,other_right})) == Bands.Rows{first,rows} : Bands.Plan},
        WordEncoding.false_true(same))
    case Model.Fork{left,right} Model.Piece{other,other_first,other_rows}:
      Empty.absurd({gathered_plan(~Element,joined_bands(~Element,joining,extent,fill,Model.Fork{left,right},Model.Piece{other,other_first,other_rows})) ==
        Bands.Split{Model.layout(~Element,left),Model.layout(~Element,right)} : Bands.Plan},WordEncoding.false_true(same))

