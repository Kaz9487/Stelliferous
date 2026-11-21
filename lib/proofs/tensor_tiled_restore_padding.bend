# Clear the inactive lanes of tiles without operating on their active lanes.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_tasks.bend as Tasks
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_cells.bend as Cells
import ../tensor_tiled_map.bend as TiledMap
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_order.bend as Order
import ./storage_buffer_model.bend as Buffer
import ./tensor_global_shape_count.bend as Global
import ./tensor_padded_pieces.bend as Padded
import ./tensor_pieces_content.bend as Content
import ./tensor_pieces_map.bend as Map
import ./tensor_pieces_model.bend as Model
import ./tensor_placed.bend as Placed
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_tiled_padding.bend as Padding

type Scope<-Reader: Data> is Data:
  Scope{reader: Reader,padding: TiledMap.Padding,cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
    fits: {Padded.groups_fit(cap,padding) == True{} : Bool}}

def before_of(~Reader: Data,+scope: Scope<Reader>) -> Reader:
  Scope{reader,padding,cap,cap_ok,fits} = scope
  reader

def padding_of(~Reader: Data,+scope: Scope<Reader>) -> TiledMap.Padding:
  Scope{reader,padding,cap,cap_ok,fits} = scope
  padding

def after_of(~Reader: Data,+scope: Scope<Reader>) -> Padded.Padded<Reader>:
  Scope{reader,padding,cap,cap_ok,fits} = scope
  Padded.Padded{padding,reader}

def total_of(~Reader: Data,+scope: Scope<Reader>) -> Nat:
  Padded.padded_cells(padding_of(~Reader,scope))

def lanes_of(+padding: TiledMap.Padding) -> U32:
  TiledMap.Padding{lanes,groups,outer,position} = padding
  lanes

def kept_step(~Reader: Data,~value: Reader -> U32 -> Tile.Tile8,+reader: Reader,+padding: TiledMap.Padding,+first: U32,+count: U32,
  +input: Buffer.Content<Tile.Tile8>,+holds: Placed.PieceHolds(~Tile.Tile8,~Padded.Padded<Reader>,~Padded.padded_value(~Reader,~value),Padded.Padded{padding,reader},first,count,
    Padding.step(padding,first,count,input)),+length: {Buffer.content_length(~Tile.Tile8,input) == U32.to_nat(count) : Nat}) ->
  Map.Stepped(~Tile.Tile8,~Padded.Padded<Reader>,~Padded.padded_value(~Reader,~value),Padded.Padded{padding,reader},first,count,input,Padding.step(padding,first,count,input)):
  Gemm.Both{+values,Gemm.Both{+size,+ready}} = holds
  +output = Padding.step(padding,first,count,input)
  Gemm.Both{ready,Gemm.Both{Equal.trans(Nat,Buffer.content_length(~Tile.Tile8,output),U32.to_nat(count),Buffer.content_length(~Tile.Tile8,input),size,
    Equal.sym(Nat,Buffer.content_length(~Tile.Tile8,input),U32.to_nat(count),length)),values}}

def piece_rule(~Reader: Data,~value: Reader -> U32 -> Tile.Tile8,+scope: Scope<Reader>,+first: U32,+count: U32,+input: Buffer.Content<Tile.Tile8>,
  +ready: Buffer.ContentReady<Tile.Tile8,input>,+length: {Buffer.content_length(~Tile.Tile8,input) == U32.to_nat(count) : Nat},
  +inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),total_of(~Reader,scope)) == True{} : Bool},
  +values: Content.piece_values(~Tile.Tile8,~Reader,~value,before_of(~Reader,scope),first,count,input)) ->
  Map.Stepped(~Tile.Tile8,~Padded.Padded<Reader>,~Padded.padded_value(~Reader,~value),after_of(~Reader,scope),first,count,input,Padding.step(padding_of(~Reader,scope),first,count,input)):
  Scope{+reader,TiledMap.Padding{+lanes,+groups,+outer,+position},+cap,+cap_ok,+fit} = scope
  +padding = {TiledMap.Padding{lanes,groups,outer,position} : TiledMap.Padding}
  kept_step(~Reader,~value,reader,padding,first,count,input,
    Padded.padded_piece(~Reader,~value,padding,reader,cap,cap_ok,first,count,input,
      Padded.fits_of(cap,lanes,groups,outer,position,first,count,fit,inside),Padded.span_of(cap,groups,outer,first,count,fit,inside),Gemm.Both{values,Gemm.Both{length,ready}}),length)

def masked(~Reader: Data,~value: Reader -> U32 -> Tile.Tile8,+scope: Scope<Reader>,+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,
  +model: Model.Model<Tile.Tile8>,+exact: {total_of(~Reader,scope) == ShapeLaws.elements(shape) : Nat},
  +contract: Global.Contract(~Tile.Tile8,~Reader,~value,before_of(~Reader,scope),Tensors.Tensor{Model.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers})) ->
  Global.Contract(~Tile.Tile8,~Padded.Padded<Reader>,~Padded.padded_value(~Reader,~value),after_of(~Reader,scope),
    Tensors.map_pieces(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,padding_of(~Reader,scope),0.0,Tensors.Tensor{Model.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers})):
  +decision = Tasks.decide(Tensors.grid_of(shape),workers,Tensors.pass_rows(1.0,0.0),Tensors.pass_steps(),Tasks.no_extra())
  Map.decided(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,~Padding.step,~Padding.rule,
    ~Reader,~value,~Padded.Padded<Reader>,~Padded.padded_value(~Reader,~value),~Scope<Reader>,~padding_of(~Reader),~before_of(~Reader),~after_of(~Reader),~total_of(~Reader),
    ~piece_rule(~Reader,~value),scope,decision,model,shape,fill,workers,exact,contract)

def read_aligned(~Reader: Data,~value: Reader -> U32 -> Tile.Tile8,+reader: Reader,+padding: TiledMap.Padding,+index: U32,
  +aligned: {U32.is_zero((lanes_of(padding) % 8 : U32)) == True{} : Bool}) ->
  {value(reader,index) == Padded.padded_value(~Reader,~value,Padded.Padded{padding,reader},index) : Tile.Tile8}:
  TiledMap.Padding{+lanes,+groups,+outer,+position} = padding
  +cell = U32.to_nat(index)
  +tile = value(reader,index)
  %Equal.sym(U32,U32.from_nat(cell),index,Arithmetic.native_roundtrip(index)) :
    {tile == Padding.cell(padding,cell,value(reader,_)) : Tile.Tile8}
  %Equal.sym(Bool,U32.is_zero((lanes % 8 : U32)),True{},aligned) :
    {tile == Bool.pick(Tile.Tile8,_,tile,Padding.kept(Padding.last(position,U32.to_nat(groups),U32.to_nat(outer),cell),Cells.Tail{((groups - 1) * 8 : U32),lanes},tile)) : Tile.Tile8}
  {==}

type Aligned<-Reader: Data,-reader: Reader,-padded: Padded.Padded<Reader>> is Data:
  Aligned{padding: TiledMap.Padding,equal: {Padded.Padded{padding,reader} == padded : Padded.Padded<Reader>},
    exact: {U32.is_zero((lanes_of(padding) % 8 : U32)) == True{} : Bool}}

def AlignedWith(~Reader: Data,reader: Reader,padded: Padded.Padded<Reader>) -> Data:
  Aligned<Reader,reader,padded>

def aligned_reads(~Reader: Data,~value: Reader -> U32 -> Tile.Tile8,+reader: Reader,+padded: Padded.Padded<Reader>,+relation: Aligned<Reader,reader,padded>,+index: U32) ->
  {value(reader,index) == Padded.padded_value(~Reader,~value,padded,index) : Tile.Tile8}:
  Aligned{+padding,+equal,+exact} = relation
  %equal : {value(reader,index) == Padded.padded_value(~Reader,~value,_,index) : Tile.Tile8}
  read_aligned(~Reader,~value,reader,padding,index,exact)

def clear_contract(~Reader: Data,~value: Reader -> U32 -> Tile.Tile8,aligned: Bool,+scope: Scope<Reader>,+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,
  +axis: Nat,+lanes: U32,+position: Grid.Position,+model: Model.Model<Tile.Tile8>,
  +checked: {U32.is_zero((lanes_of(padding_of(~Reader,scope)) % 8 : U32)) == aligned : Bool},
  +exact: {total_of(~Reader,scope) == ShapeLaws.elements(shape) : Nat},
  +contract: Global.Contract(~Tile.Tile8,~Reader,~value,before_of(~Reader,scope),Tensors.Tensor{Model.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers})) ->
  Global.Contract(~Tile.Tile8,~Padded.Padded<Reader>,~Padded.padded_value(~Reader,~value),after_of(~Reader,scope),
    Tiled.cells_of(TiledMap.clear_tail(aligned,padding_of(~Reader,scope),Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},axis,lanes,position}))):
  match aligned:
    case True{}:
      Scope{+reader,+padding,+cap,+cap_ok,+fits} = scope
      Global.relabeled(~Tile.Tile8,~Reader,~Padded.Padded<Reader>,~value,~Padded.padded_value(~Reader,~value),~AlignedWith(~Reader),~aligned_reads(~Reader,~value),
        reader,Padded.Padded{padding,reader},Aligned{padding,{==},checked},Tensors.Tensor{Model.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},contract)
    case False{}:
      masked(~Reader,~value,scope,shape,fill,workers,model,exact,contract)

def mapped_usable(+padding: TiledMap.Padding,decision: Tasks.Decision,+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,
  +model: Model.Model<Tile.Tile8>) ->
  Global.Usable<Tile.Tile8,shape,Tensors.mapped_decided(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,padding,decision,Model.pack(~Tile.Tile8,model),shape,fill,workers)>:
  Tasks.Decision{+parallel,+tasks} = decision
  Global.usable(~Tile.Tile8,Tensors.mapped_decided(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,padding,Tasks.Decision{parallel,tasks},Model.pack(~Tile.Tile8,model),shape,fill,workers),{==})

def clear_usable(aligned: Bool,+padding: TiledMap.Padding,+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,
  +axis: Nat,+lanes: U32,+position: Grid.Position,+model: Model.Model<Tile.Tile8>) ->
  Global.Usable<Tile.Tile8,shape,Tiled.cells_of(TiledMap.clear_tail(aligned,padding,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},axis,lanes,position}))>:
  match aligned:
    case True{}: Global.Usable{model,fill,workers,{==}}
    case False{}:
      mapped_usable(padding,Tasks.decide(Tensors.grid_of(shape),workers,Tensors.pass_rows(1.0,0.0),Tensors.pass_steps(),Tasks.no_extra()),shape,fill,workers,model)

def clear_shaped(~Reader: Data,~value: Reader -> U32 -> Tile.Tile8,+aligned: Bool,+scope: Scope<Reader>,+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,
  +axis: Nat,+lanes: U32,+position: Grid.Position,+model: Model.Model<Tile.Tile8>,
  +checked: {U32.is_zero((lanes_of(padding_of(~Reader,scope)) % 8 : U32)) == aligned : Bool},
  +exact: {total_of(~Reader,scope) == ShapeLaws.elements(shape) : Nat},
  +contract: Global.Contract(~Tile.Tile8,~Reader,~value,before_of(~Reader,scope),Tensors.Tensor{Model.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers})) ->
  Global.Shaped(~Tile.Tile8,~Padded.Padded<Reader>,~Padded.padded_value(~Reader,~value),after_of(~Reader,scope),shape,
    Tiled.cells_of(TiledMap.clear_tail(aligned,padding_of(~Reader,scope),Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},axis,lanes,position}))):
  Gemm.Both{clear_usable(aligned,padding_of(~Reader,scope),shape,fill,workers,axis,lanes,position,model),
    clear_contract(~Reader,~value,aligned,scope,shape,fill,workers,axis,lanes,position,model,checked,exact,contract)}
