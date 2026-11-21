# The sources of a moved tensor: a tensor of the contract laid as the
# canonical pieces of its shape, its pieces good before anything is taken,
# and the empty piece, all holding the values of the reader.

import Base
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_tasks.bend as Tasks
import ../tensor_pieces.bend as Pieces
import ../kernel_gemm.bend as KernelGemm
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_pieces_content.bend as Content
import ./tensor_placed.bend as Placed
import ./tensor_moved.bend as Moved
import ./tensor_move_readback.bend as Held
import ./tensor_components.bend as Components
import ./tensor_global_shape_count.bend as Global
import ./tensor_contract.bend as Contract
import ./tensor_shape_refinement.bend as ShapeLaws
import ./boolean_logic.bend as Logic
import ../machine_limits.bend as Machine

# Pieces whose contents fit their fields, are ready and hold the values of a
# reader hold those values piece by piece.
def holds_of(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,content: PiecesModel.Content<Element>,+total: Nat,
  +values: Content.Values(~Element,~Logical,~original,logical,content),+fields: Content.Contents(~Element,total,content),
  +ready: PiecesModel.ContentReady(~Element,content)) -> PiecesModel.ContentAll(~Element,~Placed.PieceHolds(~Element,~Logical,~original,logical),content):
  match content:
    case PiecesModel.PieceContent{+buffer,+first,+rows}:
      Content.Fields{+length_exact,+inside} = fields
      KernelGemm.Both{values,KernelGemm.Both{length_exact,ready}}
    case PiecesModel.Joined{+left,+right}:
      KernelGemm.Both{+values_left,+values_right} = values
      KernelGemm.Both{+fields_left,+fields_right} = fields
      KernelGemm.Both{+ready_left,+ready_right} = ready
      KernelGemm.Both{holds_of(~Element,~Logical,~original,logical,left,total,values_left,fields_left,ready_left),
        holds_of(~Element,~Logical,~original,logical,right,total,values_right,fields_right,ready_right)}

# The pieces of a model are laid as the canonical pieces of a shape and every
# piece holds the reader's values.
def SourceHeld(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+model: PiecesModel.Model<Element>,+shape: List<&2,U32>) -> Data:
  Held.Both<{PiecesModel.layout(~Element,model) == Pieces.ranges_of_grid(Tensors.grid_of(shape)) : Pieces.Ranges},
    PiecesModel.ContentAll(~Element,~Placed.PieceHolds(~Element,~Logical,~original,logical),PiecesModel.content(~Element,model))>

# A tensor of the contract is laid as the canonical pieces of its shape,
# every piece holding the reader's values.
def source_held(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+model: PiecesModel.Model<Element>,+shape: List<&2,U32>,
  +fill: Element,+workers: U32,
  +contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers})) ->
  SourceHeld(~Element,~Logical,~original,logical,model,shape):
  KernelGemm.Both{KernelGemm.Both{Content.WellFormed{+fields,+ready,+coverage},KernelGemm.Both{+counted,+canonical}},+values} = contract
  +packed = PiecesModel.packed_contents(~Element,model)
  +content = PiecesModel.content(~Element,model)
  +total = ShapeLaws.elements(shape)
  +layout = Pieces.ranges_of_grid(Tensors.grid_of(shape))
  Held.Both{
    Equal.trans(Pieces.Ranges,PiecesModel.layout(~Element,model),PiecesModel.content_layout(~Element,content),layout,PiecesModel.layout_matches_content(~Element,model),
      {%packed : {PiecesModel.content_layout(~Element,_) == layout : Pieces.Ranges} canonical : {PiecesModel.content_layout(~Element,content) == layout : Pieces.Ranges}}),
    holds_of(~Element,~Logical,~original,logical,content,total,
      {%packed : Content.Values(~Element,~Logical,~original,logical,_) values : Content.Values(~Element,~Logical,~original,logical,content)},
      {%packed : Content.Contents(~Element,total,_) fields : Content.Contents(~Element,total,content)},
      {%packed : PiecesModel.ContentReady(~Element,_) ready : PiecesModel.ContentReady(~Element,content)})}

# Pieces laid and held so are the placed model of the model.
def placed_of(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+model: PiecesModel.Model<Element>,+shape: List<&2,U32>,
  +held: SourceHeld(~Element,~Logical,~original,logical,model,shape)) ->
  Placed.PlacedModel<Element,Placed.PieceHolds(~Element,~Logical,~original,logical),Pieces.ranges_of_grid(Tensors.grid_of(shape)),PiecesModel.pack(~Element,model)>:
  Held.Both{+laid,+holds} = held
  Placed.PlacedModel{model,{==},laid,holds}

# A tensor of the contract is laid as the canonical pieces of its shape,
# every piece holding the reader's values.
def source_placed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+model: PiecesModel.Model<Element>,+shape: List<&2,U32>,
  +fill: Element,+workers: U32,
  +contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers})) ->
  Placed.PlacedModel<Element,Placed.PieceHolds(~Element,~Logical,~original,logical),Pieces.ranges_of_grid(Tensors.grid_of(shape)),PiecesModel.pack(~Element,model)>:
  placed_of(~Element,~Logical,~original,logical,model,shape,source_held(~Element,~Logical,~original,logical,model,shape,fill,workers,contract))

def zero_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+first: U32,content: Buffer.Content<Element>) ->
  Content.piece_values(~Element,~Logical,~original,logical,first,0,content):
  match content:
    case Buffer.Content{+tree,+length}: Unit{}

# An empty piece holds the values of any reader.
def empty_holds(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+fill: Element) ->
  Placed.PieceHolds(~Element,~Logical,~original,logical,0,0,Buffer.content(~Element,Moved.empty(~Element,fill))):
  %Equal.sym(Buffer.Content<Element>,Buffer.content(~Element,Buffer.remodel(~Element,Tensors.buffer(~Element,fill,0))),Buffer.contents(~Element,Tensors.buffer(~Element,fill,0)),
      Buffer.remodel_content(~Element,Tensors.buffer(~Element,fill,0))) :
    Placed.PieceHolds(~Element,~Logical,~original,logical,0,0,_)
  KernelGemm.Both{zero_values(~Element,~Logical,~original,logical,0,Buffer.contents(~Element,Tensors.buffer(~Element,fill,0))),
    KernelGemm.Both{Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),Buffer.contents(~Element,Tensors.buffer(~Element,fill,0)),
        Buffer.Content{Tree.fresh(Element,Tensors.depth_for(0),fill),0n},Contract.empty_contents(~Element,fill)),Contract.empty_ready(~Element,fill)}}

# Pieces that all hold a property are good for it before anything is taken.
def good_all(~Element: Data,~Property: @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+model: PiecesModel.Model<Element>,+remaining: Held.Spans,
  +holds: PiecesModel.ContentAll(~Element,~Property,PiecesModel.content(~Element,model))) -> Placed.Good(~Element,~Property,model,remaining):
  match model:
    case PiecesModel.Leaf{+buffer,+first,+rows}:
      Placed.OnRight{holds}
    case PiecesModel.Fork{+left,+right}:
      KernelGemm.Both{+left_holds,+right_holds} = holds
      Held.Both{good_all(~Element,~Property,left,remaining,left_holds),good_all(~Element,~Property,right,remaining,right_holds)}

# A piece that holds the reader's values holds the values of a source.
def held_source(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+first: U32,+rows: U32,+buffer: Buffer.Content<Element>,
  held: Placed.PieceHolds(~Element,~Logical,~original,logical,first,rows,buffer)) -> Placed.SourceHolds(~Element,~Logical,~original,logical,0,first,rows,buffer):
  held

# Every piece of a source that holds the reader's values is good for the
# first source before anything is taken.
def good_of(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+model: PiecesModel.Model<Element>,+remaining: Held.Spans,
  +holds: PiecesModel.ContentAll(~Element,~Placed.PieceHolds(~Element,~Logical,~original,logical),PiecesModel.content(~Element,model))) ->
  Placed.Good(~Element,~Placed.SourceHolds(~Element,~Logical,~original,logical,0),model,remaining):
  good_all(~Element,~Placed.SourceHolds(~Element,~Logical,~original,logical,0),model,remaining,
    PiecesModel.content_map(~Element,~Placed.PieceHolds(~Element,~Logical,~original,logical),~Placed.SourceHolds(~Element,~Logical,~original,logical,0),
      ~held_source(~Element,~Logical,~original,logical),PiecesModel.content(~Element,model),holds))

# The one source reads its row of readings when the components are separate.
def origin_when(enabled: Bool,+limits: Machine.Limits,+row: List<&2,Tasks.Reading>,+source: List<&2,U32>,+offset: U32,
  +row_ok: Logic.When(enabled,Components.RowOk(limits,row,Grid.grid(source,Grid.piece_target()),source))) ->
  Logic.When(enabled,Held.Both<{Nat.is_lt(U32.to_nat(0),List.length(&2,List<&2,Tasks.Reading>,[row])) == True{} : Bool},
    Components.RowOk(limits,Components.readings_word([row],0),Moved.origin_grid(Tensors.origin(source,offset)),Components.origin_axes(Tensors.origin(source,offset)))>):
  match enabled:
    case False{}: Unit{}
    case True{}: Held.Both{{==},row_ok}
