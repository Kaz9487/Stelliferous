# Maps and combinations of tiled tensors. Every leaf maps or combines the
# tiles of its piece and puts +0 back in the lanes past the lane count, so the
# cells hold the lifted values of the mapped or combined reader in the same
# geometry.
import Base
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ../tensor.bend as Tensors
import ../tensor_tasks.bend as Tasks
import ../tensor_pieces.bend as Pieces
import ../tensor_grid.bend as Grid
import ../tensor_shape.bend as Shapes
import ../tensor_operations.bend as Operations
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_cells.bend as Cells
import ../tensor_tiled_map.bend as TiledMap
import ./tensor_tiled_padding.bend as Padding
import ./tensor_tiled_padded_cells.bend as PaddedCells
import ./tensor_tiled_public.bend as Public
import ./tensor_tiled_representation.bend as Representation
import ./tensor_pieces_map.bend as PiecesMap
import ./tensor_pieces_combine.bend as PiecesCombine
import ./tensor_pieces_combine_leaf.bend as CombineLeaf
import ./tensor_pieces_model.bend as Model
import ./tensor_pieces_content.bend as Content
import ./tensor_global_shape_count.bend as Global
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_operations_refinement.bend as OperationLaws
import ./storage_buffer_model.bend as Buffer
import ../kernel_gemm.bend as KernelGemm
import ../tensor_view.bend as Views
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./machine_limit_bounds.bend as Limits
import ./tensor_view_gather.bend as ViewGather
import ./tensor_padded_pieces.bend as PaddedPieces
import ./tensor_pieces_spread.bend as PiecesSpread
import ./tensor_cell_room.bend as Rooms
import ./tensor_storage_count.bend as StorageCount

# ---- A mapped cell ----

# A tile function that is the scalar operation at every lane.
def LaneLaw(~Operation: Data,~transform: Operation -> Tile.Tile8 -> Tile.Tile8,~operation: Operation -> F32 -> F32) -> Type:
  @+context: Operation -> @+tile: Tile.Tile8 -> {transform(context,tile) == TiledMap.lanes(~Operation,~operation,context,tile) : Tile.Tile8}

# The lanes of the scalar operation are their own law.
def lanes_law(~Operation: Data,~operation: Operation -> F32 -> F32,+context: Operation,+tile: Tile.Tile8) ->
  {TiledMap.lanes(~Operation,~operation,context,tile) == TiledMap.lanes(~Operation,~operation,context,tile) : Tile.Tile8}:
  {==}

# A mapped lane is the mapped value below the count, and +0 like it past the
# count.
def tail_mapped(~Operation: Data,~operation: Operation -> F32 -> F32,+value: F32,+context: Operation,valid: Bool) ->
  {Bool.pick(F32,valid,operation(context,Bool.pick(F32,valid,value,0.0)),0.0) == Bool.pick(F32,valid,operation(context,value),0.0) : F32}:
  match valid:
    case True{}: {==}
    case False{}: {==}

# The lanes of a mapped cell agree with the cell of the mapped reader.
def mapped_valid(~Operation: Data,~Logical: Data,~operation: Operation -> F32 -> F32,~original: Logical -> U32 -> F32,+context: Operation,+logical: Logical,
  +geometry: Cells.Cells,+outer: Nat,+group: Nat) ->
  PaddedCells.Valid<Representation.lane_count(geometry),group,
    TiledMap.lanes(~Operation,~operation,context,Representation.cell(~Logical,~original,logical,geometry,outer,group)),
    Representation.cell(~PiecesMap.Context<Operation,Logical>,~PiecesMap.value(~F32,~Operation,~Logical,~operation,~original),PiecesMap.Context{context,logical},geometry,outer,group)>:
  +lanes = Representation.lane_count(geometry)
  PaddedCells.Valid{
    tail_mapped(~Operation,~operation,original(logical,U32.from_nat(Representation.logical_index(geometry,outer,Nat.add(0n,Nat.mul(group,8n))))),context,
      Nat.is_lt(Nat.add(0n,Nat.mul(group,8n)),lanes)),
    tail_mapped(~Operation,~operation,original(logical,U32.from_nat(Representation.logical_index(geometry,outer,Nat.add(1n,Nat.mul(group,8n))))),context,
      Nat.is_lt(Nat.add(1n,Nat.mul(group,8n)),lanes)),
    tail_mapped(~Operation,~operation,original(logical,U32.from_nat(Representation.logical_index(geometry,outer,Nat.add(2n,Nat.mul(group,8n))))),context,
      Nat.is_lt(Nat.add(2n,Nat.mul(group,8n)),lanes)),
    tail_mapped(~Operation,~operation,original(logical,U32.from_nat(Representation.logical_index(geometry,outer,Nat.add(3n,Nat.mul(group,8n))))),context,
      Nat.is_lt(Nat.add(3n,Nat.mul(group,8n)),lanes)),
    tail_mapped(~Operation,~operation,original(logical,U32.from_nat(Representation.logical_index(geometry,outer,Nat.add(4n,Nat.mul(group,8n))))),context,
      Nat.is_lt(Nat.add(4n,Nat.mul(group,8n)),lanes)),
    tail_mapped(~Operation,~operation,original(logical,U32.from_nat(Representation.logical_index(geometry,outer,Nat.add(5n,Nat.mul(group,8n))))),context,
      Nat.is_lt(Nat.add(5n,Nat.mul(group,8n)),lanes)),
    tail_mapped(~Operation,~operation,original(logical,U32.from_nat(Representation.logical_index(geometry,outer,Nat.add(6n,Nat.mul(group,8n))))),context,
      Nat.is_lt(Nat.add(6n,Nat.mul(group,8n)),lanes)),
    tail_mapped(~Operation,~operation,original(logical,U32.from_nat(Representation.logical_index(geometry,outer,Nat.add(7n,Nat.mul(group,8n))))),context,
      Nat.is_lt(Nat.add(7n,Nat.mul(group,8n)),lanes))}

# ---- The leaf of a map ----

# What padding makes of the map of the cells of a piece.
def step(~Operation: Data,~transform: Operation -> Tile.Tile8 -> Tile.Tile8,+mapping: TiledMap.Mapping<Operation>,+first: U32,+count: U32,content: Buffer.Content<Tile.Tile8>) ->
  Buffer.Content<Tile.Tile8>:
  TiledMap.Mapping{+context,+padding} = mapping
  Padding.step(padding,first,count,PiecesMap.map_step(~Tile.Tile8,~Operation,~transform,context,U32.to_nat(count),content))

def rule_model(~Operation: Data,~transform: Operation -> Tile.Tile8 -> Tile.Tile8,+context: Operation,+padding: TiledMap.Padding,+first: U32,+count: U32,+input: Buffer.Model<Tile.Tile8>) ->
  {Buffer.contents(~Tile.Tile8,TiledMap.map_leaf(~Operation,~transform,TiledMap.Mapping{context,padding},first,count,Buffer.pack(~Tile.Tile8,input))) ==
    step(~Operation,~transform,TiledMap.Mapping{context,padding},first,count,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,input))) : Buffer.Content<Tile.Tile8>}:
  Equal.trans(Buffer.Content<Tile.Tile8>,
    Buffer.contents(~Tile.Tile8,TiledMap.padded(padding,first,count,Operations.mapped_context(~Tile.Tile8,~Operation,~transform,U32.to_nat(count),0,context,Buffer.pack(~Tile.Tile8,input)))),
    Padding.step(padding,first,count,Buffer.contents(~Tile.Tile8,Operations.mapped_context(~Tile.Tile8,~Operation,~transform,U32.to_nat(count),0,context,Buffer.pack(~Tile.Tile8,input)))),
    Padding.step(padding,first,count,PiecesMap.map_step(~Tile.Tile8,~Operation,~transform,context,U32.to_nat(count),Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,input)))),
    Padding.rule(padding,first,count,Operations.mapped_context(~Tile.Tile8,~Operation,~transform,U32.to_nat(count),0,context,Buffer.pack(~Tile.Tile8,input))),
    Equal.cong(Buffer.Content<Tile.Tile8>,Buffer.Content<Tile.Tile8>,content => Padding.step(padding,first,count,content),
      Buffer.contents(~Tile.Tile8,Operations.mapped_context(~Tile.Tile8,~Operation,~transform,U32.to_nat(count),0,context,Buffer.pack(~Tile.Tile8,input))),
      PiecesMap.map_step(~Tile.Tile8,~Operation,~transform,context,U32.to_nat(count),Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,input))),
      PiecesMap.mapped(~Tile.Tile8,~Operation,~transform,~Operations.mapped_context(~Tile.Tile8,~Operation,~transform),~OperationLaws.mapped_context(~Tile.Tile8,~Operation,~transform),
        context,U32.to_nat(count),Buffer.pack(~Tile.Tile8,input))))

def rule_observed(~Operation: Data,~transform: Operation -> Tile.Tile8 -> Tile.Tile8,+context: Operation,+padding: TiledMap.Padding,+first: U32,+count: U32,
  -buffer: Storage.Buffer<Tile.Tile8>,observation: Buffer.Observation<Tile.Tile8,buffer>) ->
  {Buffer.contents(~Tile.Tile8,TiledMap.map_leaf(~Operation,~transform,TiledMap.Mapping{context,padding},first,count,buffer)) ==
    step(~Operation,~transform,TiledMap.Mapping{context,padding},first,count,Buffer.contents(~Tile.Tile8,buffer)) : Buffer.Content<Tile.Tile8>}:
  Buffer.Observation{+model,+equation} = observation
  %equation : {Buffer.contents(~Tile.Tile8,TiledMap.map_leaf(~Operation,~transform,TiledMap.Mapping{context,padding},first,count,_)) ==
    step(~Operation,~transform,TiledMap.Mapping{context,padding},first,count,Buffer.contents(~Tile.Tile8,buffer)) : Buffer.Content<Tile.Tile8>}
  %equation : {Buffer.contents(~Tile.Tile8,TiledMap.map_leaf(~Operation,~transform,TiledMap.Mapping{context,padding},first,count,Buffer.pack(~Tile.Tile8,model))) ==
    step(~Operation,~transform,TiledMap.Mapping{context,padding},first,count,Buffer.contents(~Tile.Tile8,_)) : Buffer.Content<Tile.Tile8>}
  rule_model(~Operation,~transform,context,padding,first,count,model)

# The leaf of a map steps the contents of its piece.
def rule(~Operation: Data,~transform: Operation -> Tile.Tile8 -> Tile.Tile8,+mapping: TiledMap.Mapping<Operation>,+first: U32,+count: U32,buffer: Storage.Buffer<Tile.Tile8>) ->
  {Buffer.contents(~Tile.Tile8,TiledMap.map_leaf(~Operation,~transform,mapping,first,count,buffer)) == step(~Operation,~transform,mapping,first,count,Buffer.contents(~Tile.Tile8,buffer)) :
    Buffer.Content<Tile.Tile8>}:
  TiledMap.Mapping{+context,+padding} = mapping
  rule_observed(~Operation,~transform,context,padding,first,count,buffer,Buffer.observe(~Tile.Tile8,buffer))

# ---- The cells of a piece ----

# What every piece of a map reads: the function, the logical reader and the
# layout of the cells.
type Scope<-Operation: Data,-Logical: Data> is Data:
  Scope{context: Operation,logical: Logical,layout: PaddedCells.Layout}

def context_of(~Operation: Data,~Logical: Data,+scope: Scope<Operation,Logical>) -> TiledMap.Mapping<Operation>:
  Scope{+context,+logical,+layout} = scope
  TiledMap.Mapping{context,PaddedCells.layout_padding(layout)}

def layout_of(~Operation: Data,~Logical: Data,+scope: Scope<Operation,Logical>) -> PaddedCells.Layout:
  Scope{+context,+logical,+layout} = scope
  layout

def before_of(~Operation: Data,~Logical: Data,+scope: Scope<Operation,Logical>) -> Representation.Lifted<Logical>:
  Scope{+context,+logical,+layout} = scope
  Representation.Lifted{logical,PaddedCells.layout_geometry(layout)}

# The tiles of a piece after the map of its cells.
def mapped_of(~Operation: Data,~Logical: Data,+scope: Scope<Operation,Logical>) -> PiecesMap.Context<Operation,Representation.Lifted<Logical>>:
  Scope{+context,+logical,+layout} = scope
  PiecesMap.Context{context,Representation.Lifted{logical,PaddedCells.layout_geometry(layout)}}

def after_of(~Operation: Data,~Logical: Data,+scope: Scope<Operation,Logical>) -> Representation.Lifted<PiecesMap.Context<Operation,Logical>>:
  Scope{+context,+logical,+layout} = scope
  Representation.Lifted{PiecesMap.Context{context,logical},PaddedCells.layout_geometry(layout)}

def total_of(~Operation: Data,~Logical: Data,+scope: Scope<Operation,Logical>) -> Nat:
  Scope{+context,+logical,+layout} = scope
  PaddedCells.layout_total(layout)

# The lanes of every mapped cell agree with the cell of the mapped reader.
def mapped_agree(~Operation: Data,~Logical: Data,~operation: Operation -> F32 -> F32,~transform: Operation -> Tile.Tile8 -> Tile.Tile8,~lane_law: LaneLaw(~Operation,~transform,~operation),~original: Logical -> U32 -> F32,+scope: Scope<Operation,Logical>,+word: U32,
  +inside: {Nat.is_lt(U32.to_nat(word),PaddedCells.layout_total(layout_of(~Operation,~Logical,scope))) == True{} : Bool}) ->
  PaddedCells.Valid<Representation.lane_count(PaddedCells.layout_geometry(layout_of(~Operation,~Logical,scope))),
    Representation.cell_group(PaddedCells.layout_geometry(layout_of(~Operation,~Logical,scope)),U32.to_nat(word)),
    PiecesMap.value(~Tile.Tile8,~Operation,~Representation.Lifted<Logical>,~transform,~Representation.lifted(~Logical,~original),
      mapped_of(~Operation,~Logical,scope),word),
    Representation.lifted(~PiecesMap.Context<Operation,Logical>,~PiecesMap.value(~F32,~Operation,~Logical,~operation,~original),after_of(~Operation,~Logical,scope),word)>:
  Scope{+context,+logical,+layout} = scope
  +geometry = PaddedCells.layout_geometry(layout)
  +number = U32.to_nat(word)
  +outer = Representation.cell_outer(geometry,number)
  +group = Representation.cell_group(geometry,number)
  +cell = Representation.cell(~Logical,~original,logical,geometry,outer,group)
  %Equal.sym(Tile.Tile8,transform(context,cell),TiledMap.lanes(~Operation,~operation,context,cell),lane_law(context,cell)) :
    PaddedCells.Valid<Representation.lane_count(geometry),group,_,
      Representation.cell(~PiecesMap.Context<Operation,Logical>,~PiecesMap.value(~F32,~Operation,~Logical,~operation,~original),PiecesMap.Context{context,logical},geometry,outer,group)>
  mapped_valid(~Operation,~Logical,~operation,~original,context,logical,geometry,Representation.cell_outer(geometry,number),Representation.cell_group(geometry,number))

# What the map step keeps of a piece of the lifted values: the map of its
# tiles, then the padding.
def piece_rule(~Operation: Data,~Logical: Data,~operation: Operation -> F32 -> F32,~transform: Operation -> Tile.Tile8 -> Tile.Tile8,~lane_law: LaneLaw(~Operation,~transform,~operation),~original: Logical -> U32 -> F32,+scope: Scope<Operation,Logical>,+first: U32,+count: U32,
  +input: Buffer.Content<Tile.Tile8>,+ready: Buffer.ContentReady<Tile.Tile8,input>,+length_exact: {Buffer.content_length(~Tile.Tile8,input) == U32.to_nat(count) : Nat},
  +inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),total_of(~Operation,~Logical,scope)) == True{} : Bool},
  +values: Content.piece_values(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),before_of(~Operation,~Logical,scope),first,count,input)) ->
  PiecesMap.Stepped(~Tile.Tile8,~Representation.Lifted<PiecesMap.Context<Operation,Logical>>,
    ~Representation.lifted(~PiecesMap.Context<Operation,Logical>,~PiecesMap.value(~F32,~Operation,~Logical,~operation,~original)),after_of(~Operation,~Logical,scope),first,count,
    input,step(~Operation,~transform,context_of(~Operation,~Logical,scope),first,count,input)):
  Scope{+context,+logical,+layout} = scope
  Buffer.Content{+tree,+length} = input
  +rebuilt = {Scope{context,logical,layout} : Scope<Operation,Logical>}
  +before = {Representation.Lifted{logical,PaddedCells.layout_geometry(layout)} : Representation.Lifted<Logical>}
  +content = {Buffer.Content{tree,length} : Buffer.Content<Tile.Tile8>}
  PaddedCells.padded_piece(~Scope<Operation,Logical>,~PiecesMap.Context<Operation,Representation.Lifted<Logical>>,
    ~PiecesMap.value(~Tile.Tile8,~Operation,~Representation.Lifted<Logical>,~transform,~Representation.lifted(~Logical,~original)),
    ~Representation.Lifted<PiecesMap.Context<Operation,Logical>>,~Representation.lifted(~PiecesMap.Context<Operation,Logical>,~PiecesMap.value(~F32,~Operation,~Logical,~operation,~original)),
    ~layout_of(~Operation,~Logical),~mapped_of(~Operation,~Logical),~after_of(~Operation,~Logical),~mapped_agree(~Operation,~Logical,~operation,~transform,~lane_law,~original),
    rebuilt,first,count,PiecesMap.map_step(~Tile.Tile8,~Operation,~transform,context,U32.to_nat(count),content),
    PiecesMap.map_ready(~Tile.Tile8,~Operation,~transform,context,U32.to_nat(count),content,ready),length_exact,inside,
    PiecesMap.map_values(~Tile.Tile8,~Operation,~Representation.Lifted<Logical>,~transform,~Representation.lifted(~Logical,~original),
      context,before,first,count,content,ready,length_exact,values))

# ---- The map of a tiled tensor ----

# The map of the pieces of a scope keeps the contract with the mapped reader.
def scoped(~Operation: Data,~Logical: Data,~operation: Operation -> F32 -> F32,~transform: Operation -> Tile.Tile8 -> Tile.Tile8,~lane_law: LaneLaw(~Operation,~transform,~operation),~original: Logical -> U32 -> F32,+scope: Scope<Operation,Logical>,+parallel: Bool,
  +tasks: Pieces.Ranges,-pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,
  +total_exact: {total_of(~Operation,~Logical,scope) == ShapeLaws.elements(shape) : Nat},observation: Model.Observation<Tile.Tile8,pieces>,
  contract: Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),before_of(~Operation,~Logical,scope),
    Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers})) ->
  Global.Contract(~Tile.Tile8,~Representation.Lifted<PiecesMap.Context<Operation,Logical>>,
    ~Representation.lifted(~PiecesMap.Context<Operation,Logical>,~PiecesMap.value(~F32,~Operation,~Logical,~operation,~original)),after_of(~Operation,~Logical,scope),
    Tensors.mapped_decided(~Tile.Tile8,~TiledMap.Mapping<Operation>,~TiledMap.map_leaf(~Operation,~transform),context_of(~Operation,~Logical,scope),
      Tasks.Decision{parallel,tasks},pieces,shape,fill,workers)):
  Model.Observation{+model,+equation} = observation
  %equation : Global.Contract(~Tile.Tile8,~Representation.Lifted<PiecesMap.Context<Operation,Logical>>,
    ~Representation.lifted(~PiecesMap.Context<Operation,Logical>,~PiecesMap.value(~F32,~Operation,~Logical,~operation,~original)),after_of(~Operation,~Logical,scope),
    Tensors.mapped_decided(~Tile.Tile8,~TiledMap.Mapping<Operation>,~TiledMap.map_leaf(~Operation,~transform),context_of(~Operation,~Logical,scope),
      Tasks.Decision{parallel,tasks},_,shape,fill,workers))
  PiecesMap.decided(~Tile.Tile8,~TiledMap.Mapping<Operation>,~TiledMap.map_leaf(~Operation,~transform),~step(~Operation,~transform),
    ~rule(~Operation,~transform),~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
    ~Representation.Lifted<PiecesMap.Context<Operation,Logical>>,~Representation.lifted(~PiecesMap.Context<Operation,Logical>,~PiecesMap.value(~F32,~Operation,~Logical,~operation,~original)),
    ~Scope<Operation,Logical>,~context_of(~Operation,~Logical),~before_of(~Operation,~Logical),~after_of(~Operation,~Logical),~total_of(~Operation,~Logical),
    ~piece_rule(~Operation,~Logical,~operation,~transform,~lane_law,~original),scope,Tasks.Decision{parallel,tasks},model,shape,fill,workers,total_exact,
    %Equal.sym(Pieces.Pieces(~Tile.Tile8),Model.pack(~Tile.Tile8,model),pieces,equation) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),before_of(~Operation,~Logical,scope),
        Tensors.Tensor{_,shape,fill,Tensors.Usable{},workers}) contract)

# The map of the cells of values in their layout.
def placed_map(~Operation: Data,~Logical: Data,~operation: Operation -> F32 -> F32,~transform: Operation -> Tile.Tile8 -> Tile.Tile8,~lane_law: LaneLaw(~Operation,~transform,~operation),~original: Logical -> U32 -> F32,+within: Bool,+context: Operation,+logical: Logical,
  +values: List<&2,U32>,+axis: Nat,+lanes: U32,+position: Grid.Position,+parallel: Bool,+tasks: Pieces.Ranges,pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,
  +fill: Tile.Tile8,+workers: U32,
  global: Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,axis,position)},
    Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}),
  +placed: PaddedCells.Placed<shape,values,axis,lanes,position,within>) ->
  Global.Contract(~Tile.Tile8,~Representation.Lifted<PiecesMap.Context<Operation,Logical>>,
    ~Representation.lifted(~PiecesMap.Context<Operation,Logical>,~PiecesMap.value(~F32,~Operation,~Logical,~operation,~original)),
    Representation.Lifted{PiecesMap.Context{context,logical},Public.geometry_of(values,axis,position)},
    Tensors.mapped_decided(~Tile.Tile8,~TiledMap.Mapping<Operation>,~TiledMap.map_leaf(~Operation,~transform),
      TiledMap.Mapping{context,TiledMap.padding_when(within,shape,lanes,position)},Tasks.Decision{parallel,tasks},pieces,shape,fill,workers)):
  PaddedCells.Placed{+layout,+padding_exact,+geometry_exact,+total_exact} = placed
  +geometry = PaddedCells.layout_geometry(layout)
  %padding_exact : Global.Contract(~Tile.Tile8,~Representation.Lifted<PiecesMap.Context<Operation,Logical>>,
    ~Representation.lifted(~PiecesMap.Context<Operation,Logical>,~PiecesMap.value(~F32,~Operation,~Logical,~operation,~original)),
    Representation.Lifted{PiecesMap.Context{context,logical},Public.geometry_of(values,axis,position)},
    Tensors.mapped_decided(~Tile.Tile8,~TiledMap.Mapping<Operation>,~TiledMap.map_leaf(~Operation,~transform),
      TiledMap.Mapping{context,_},Tasks.Decision{parallel,tasks},pieces,shape,fill,workers))
  %geometry_exact : Global.Contract(~Tile.Tile8,~Representation.Lifted<PiecesMap.Context<Operation,Logical>>,
    ~Representation.lifted(~PiecesMap.Context<Operation,Logical>,~PiecesMap.value(~F32,~Operation,~Logical,~operation,~original)),
    Representation.Lifted{PiecesMap.Context{context,logical},_},
    Tensors.mapped_decided(~Tile.Tile8,~TiledMap.Mapping<Operation>,~TiledMap.map_leaf(~Operation,~transform),
      TiledMap.Mapping{context,PaddedCells.layout_padding(layout)},Tasks.Decision{parallel,tasks},pieces,shape,fill,workers))
  scoped(~Operation,~Logical,~operation,~transform,~lane_law,~original,Scope{context,logical,layout},parallel,tasks,pieces,shape,fill,workers,total_exact,Model.observe(~Tile.Tile8,pieces),
    %Equal.sym(Cells.Cells,geometry,Public.geometry_of(values,axis,position),geometry_exact) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,_},
        Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}) global)

def decided_map(~Operation: Data,~Logical: Data,~operation: Operation -> F32 -> F32,~transform: Operation -> Tile.Tile8 -> Tile.Tile8,~lane_law: LaneLaw(~Operation,~transform,~operation),~original: Logical -> U32 -> F32,+context: Operation,+logical: Logical,+values: List<&2,U32>,
  +axis: Nat,+lanes: U32,+position: Grid.Position,decision: Tasks.Decision,pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,
  global: Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,axis,position)},
    Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}),
  +laid: Public.Laid<shape,axis,lanes,position,values>) ->
  Public.Within(~Representation.Lifted<PiecesMap.Context<Operation,Logical>>,
    ~Representation.lifted(~PiecesMap.Context<Operation,Logical>,~PiecesMap.value(~F32,~Operation,~Logical,~operation,~original)),~PiecesMap.Context<Operation,Logical>,
    ~Public.lifted_at(~PiecesMap.Context<Operation,Logical>),PiecesMap.Context{context,logical},values,axis,lanes,position,
    Tensors.mapped_decided(~Tile.Tile8,~TiledMap.Mapping<Operation>,~TiledMap.map_leaf(~Operation,~transform),
      TiledMap.Mapping{context,TiledMap.padding_of(shape,axis,lanes,position)},decision,pieces,shape,fill,workers)):
  match decision:
    case Tasks.Decision{+parallel,+tasks}:
      +within = Nat.is_lt(axis,List.length(&2,U32,shape))
      KernelGemm.Both{placed_map(~Operation,~Logical,~operation,~transform,~lane_law,~original,within,context,logical,values,axis,lanes,position,parallel,tasks,pieces,shape,fill,workers,global,
        PaddedCells.placed(within,values,axis,lanes,position,shape,{==},laid)),laid}

# A map whose tile function is the scalar operation at every lane keeps the
# contract of a tiled tensor with the mapped reader.
def map(~Operation: Data,~Logical: Data,~operation: Operation -> F32 -> F32,~transform: Operation -> Tile.Tile8 -> Tile.Tile8,~lane_law: LaneLaw(~Operation,~transform,~operation),~original: Logical -> U32 -> F32,+context: Operation,+operations: F32,+logical: Logical,
  +values: List<&2,U32>,tiled: Tiled.Tiled,contract: Public.Contract(~Logical,~original,logical,values,tiled)) ->
  Public.Contract(~PiecesMap.Context<Operation,Logical>,~PiecesMap.value(~F32,~Operation,~Logical,~operation,~original),PiecesMap.Context{context,logical},values,
    TiledMap.map(~Operation,~transform,context,operations,tiled)):
  match tiled:
    case Tiled.Tiled{cells,+axis,+lanes,+position}:
      match cells:
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Usable{},+workers}:
          KernelGemm.Both{global,+laid} = contract
          decided_map(~Operation,~Logical,~operation,~transform,~lane_law,~original,context,logical,values,axis,lanes,position,
            Tasks.decide(Tensors.grid_of(shape),workers,Tensors.pass_rows(1.0,operations),Tensors.pass_steps(),Tasks.no_extra()),pieces,shape,fill,workers,global,laid)
        case Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers}:
          Empty.absurd(Public.Contract(~PiecesMap.Context<Operation,Logical>,~PiecesMap.value(~F32,~Operation,~Logical,~operation,~original),PiecesMap.Context{context,logical},values,
            TiledMap.map(~Operation,~transform,context,operations,
              Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers},axis,lanes,position})),contract)

# ---- Combinations of tiled tensors ----

# A combined lane is the combined value below the count, and +0 like it past
# the count.
def tail_combined(~operation: F32 -> F32 -> F32,~left: F32,~right: F32,valid: Bool) ->
  {Bool.pick(F32,valid,operation(Bool.pick(F32,valid,left,0.0),Bool.pick(F32,valid,right,0.0)),0.0) == Bool.pick(F32,valid,operation(left,right),0.0) : F32}:
  match valid:
    case True{}: {==}
    case False{}: {==}

def combined_lane(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,
  +geometry: Cells.Cells,+outer: Nat,+group: Nat,+lane: Nat) ->
  {Bool.pick(F32,Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),Representation.lane_count(geometry)),
      operation(Representation.lane(~Left,~left_value,left,geometry,outer,group,lane),Representation.lane(~Right,~right_value,right,geometry,outer,group,lane)),0.0) ==
    Representation.lane(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),CombineLeaf.Context{left,right},geometry,outer,group,lane) : F32}:
  +channel = Nat.add(lane,Nat.mul(group,8n))
  +index = U32.from_nat(Representation.logical_index(geometry,outer,channel))
  tail_combined(~operation,~left_value(left,index),~right_value(right,index),Nat.is_lt(channel,Representation.lane_count(geometry)))

# The lanes of a combined cell agree with the cell of the combined reader.
def combined_valid(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,
  +geometry: Cells.Cells,+outer: Nat,+group: Nat) ->
  PaddedCells.Valid<Representation.lane_count(geometry),group,
    TiledMap.pairs(~operation,Representation.cell(~Left,~left_value,left,geometry,outer,group),Representation.cell(~Right,~right_value,right,geometry,outer,group)),
    Representation.cell(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),CombineLeaf.Context{left,right},geometry,outer,group)>:
  PaddedCells.Valid{combined_lane(~Left,~left_value,~Right,~right_value,~operation,left,right,geometry,outer,group,0n),
    combined_lane(~Left,~left_value,~Right,~right_value,~operation,left,right,geometry,outer,group,1n),
    combined_lane(~Left,~left_value,~Right,~right_value,~operation,left,right,geometry,outer,group,2n),
    combined_lane(~Left,~left_value,~Right,~right_value,~operation,left,right,geometry,outer,group,3n),
    combined_lane(~Left,~left_value,~Right,~right_value,~operation,left,right,geometry,outer,group,4n),
    combined_lane(~Left,~left_value,~Right,~right_value,~operation,left,right,geometry,outer,group,5n),
    combined_lane(~Left,~left_value,~Right,~right_value,~operation,left,right,geometry,outer,group,6n),
    combined_lane(~Left,~left_value,~Right,~right_value,~operation,left,right,geometry,outer,group,7n)}

# What every piece of a combination reads: the two logical readers and the
# layout of the cells.
type Operands<-Left: Data,-Right: Data> is Data:
  Operands{left: Left,right: Right,layout: PaddedCells.Layout}

def pair_layout(~Left: Data,~Right: Data,+pair: Operands<Left,Right>) -> PaddedCells.Layout:
  Operands{+left,+right,+layout} = pair
  layout

def pair_padding(~Left: Data,~Right: Data,+pair: Operands<Left,Right>) -> TiledMap.Padding:
  Operands{+left,+right,+layout} = pair
  PaddedCells.layout_padding(layout)

def pair_left(~Left: Data,~Right: Data,+pair: Operands<Left,Right>) -> Representation.Lifted<Left>:
  Operands{+left,+right,+layout} = pair
  Representation.Lifted{left,PaddedCells.layout_geometry(layout)}

def pair_right(~Left: Data,~Right: Data,+pair: Operands<Left,Right>) -> Representation.Lifted<Right>:
  Operands{+left,+right,+layout} = pair
  Representation.Lifted{right,PaddedCells.layout_geometry(layout)}

def pair_after(~Left: Data,~Right: Data,+pair: Operands<Left,Right>) -> Representation.Lifted<CombineLeaf.Context<Left,Right>>:
  Operands{+left,+right,+layout} = pair
  Representation.Lifted{CombineLeaf.Context{left,right},PaddedCells.layout_geometry(layout)}

def pair_total(~Left: Data,~Right: Data,+pair: Operands<Left,Right>) -> Nat:
  Operands{+left,+right,+layout} = pair
  PaddedCells.layout_total(layout)

def paired_of(~Left: Data,~Right: Data,+pair: Operands<Left,Right>) -> CombineLeaf.Context<Representation.Lifted<Left>,Representation.Lifted<Right>>:
  PiecesCombine.combined_of(~Representation.Lifted<Left>,~Representation.Lifted<Right>,~Operands<Left,Right>,~pair_left(~Left,~Right),~pair_right(~Left,~Right),pair)

def paired_agree(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+pair: Operands<Left,Right>,+word: U32,
  +inside: {Nat.is_lt(U32.to_nat(word),PaddedCells.layout_total(pair_layout(~Left,~Right,pair))) == True{} : Bool}) ->
  PaddedCells.Valid<Representation.lane_count(PaddedCells.layout_geometry(pair_layout(~Left,~Right,pair))),
    Representation.cell_group(PaddedCells.layout_geometry(pair_layout(~Left,~Right,pair)),U32.to_nat(word)),
    CombineLeaf.value(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.Lifted<Right>,~TiledMap.pairs(~operation),~Representation.lifted(~Left,~left_value),
      ~Representation.lifted(~Right,~right_value),paired_of(~Left,~Right,pair),word),
    Representation.lifted(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),pair_after(~Left,~Right,pair),word)>:
  Operands{+left,+right,+layout} = pair
  +geometry = PaddedCells.layout_geometry(layout)
  +number = U32.to_nat(word)
  combined_valid(~Left,~left_value,~Right,~right_value,~operation,left,right,geometry,Representation.cell_outer(geometry,number),Representation.cell_group(geometry,number))

# The padding after a combination keeps a piece of the combined tiles to the
# cells of the combined reader.
def paired_rule(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+pair: Operands<Left,Right>,
  +first: U32,+count: U32,+input: Buffer.Content<Tile.Tile8>,+ready: Buffer.ContentReady<Tile.Tile8,input>,
  +length_exact: {Buffer.content_length(~Tile.Tile8,input) == U32.to_nat(count) : Nat},
  +inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),pair_total(~Left,~Right,pair)) == True{} : Bool},
  +values: Content.piece_values(~Tile.Tile8,~CombineLeaf.Context<Representation.Lifted<Left>,Representation.Lifted<Right>>,
    ~CombineLeaf.value(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.Lifted<Right>,~TiledMap.pairs(~operation),~Representation.lifted(~Left,~left_value),
      ~Representation.lifted(~Right,~right_value)),
    PiecesCombine.combined_of(~Representation.Lifted<Left>,~Representation.Lifted<Right>,~Operands<Left,Right>,~pair_left(~Left,~Right),~pair_right(~Left,~Right),pair),first,count,input)) ->
  PiecesMap.Stepped(~Tile.Tile8,~Representation.Lifted<CombineLeaf.Context<Left,Right>>,
    ~Representation.lifted(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value)),pair_after(~Left,~Right,pair),first,count,input,
    Padding.step(pair_padding(~Left,~Right,pair),first,count,input)):
  Operands{+left,+right,+layout} = pair
  PaddedCells.padded_piece(~Operands<Left,Right>,~CombineLeaf.Context<Representation.Lifted<Left>,Representation.Lifted<Right>>,
    ~CombineLeaf.value(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.Lifted<Right>,~TiledMap.pairs(~operation),~Representation.lifted(~Left,~left_value),
      ~Representation.lifted(~Right,~right_value)),
    ~Representation.Lifted<CombineLeaf.Context<Left,Right>>,~Representation.lifted(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value)),
    ~pair_layout(~Left,~Right),~paired_of(~Left,~Right),~pair_after(~Left,~Right),~paired_agree(~Left,~left_value,~Right,~right_value,~operation),
    Operands{left,right,layout},first,count,input,ready,length_exact,inside,values)

# The combination of the pieces of a pair keeps the contract with the
# combined reader.
def paired(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+pair: Operands<Left,Right>,+parallel: Bool,
  +tasks: Pieces.Ranges,-left_pieces: Pieces.Pieces(~Tile.Tile8),-right_pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,
  +right_fill: Tile.Tile8,+right_workers: U32,+total_exact: {pair_total(~Left,~Right,pair) == ShapeLaws.elements(shape) : Nat},
  left_observation: Model.Observation<Tile.Tile8,left_pieces>,right_observation: Model.Observation<Tile.Tile8,right_pieces>,
  left_contract: Global.Contract(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),pair_left(~Left,~Right,pair),
    Tensors.Tensor{left_pieces,shape,fill,Tensors.Usable{},workers}),
  right_contract: Global.Contract(~Tile.Tile8,~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),pair_right(~Left,~Right,pair),
    Tensors.Tensor{right_pieces,shape,right_fill,Tensors.Usable{},right_workers})) ->
  Global.Contract(~Tile.Tile8,~Representation.Lifted<CombineLeaf.Context<Left,Right>>,
    ~Representation.lifted(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value)),pair_after(~Left,~Right,pair),
    Tensors.zipped_decided(~Tile.Tile8,~TiledMap.Padding,~Operations.combine_kernel(~Tile.Tile8,~TiledMap.pairs(~operation)),~TiledMap.padded,pair_padding(~Left,~Right,pair),
      Tasks.Decision{parallel,tasks},shape,fill,workers,Pieces.zip(~Storage.Buffer<Tile.Tile8>,~Storage.Buffer<Tile.Tile8>,left_pieces,right_pieces))):
  Model.Observation{+left_model,+left_equation} = left_observation
  Model.Observation{+right_model,+right_equation} = right_observation
  +after = pair_after(~Left,~Right,pair)
  +padding = pair_padding(~Left,~Right,pair)
  %left_equation : Global.Contract(~Tile.Tile8,~Representation.Lifted<CombineLeaf.Context<Left,Right>>,
    ~Representation.lifted(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value)),after,
    Tensors.zipped_decided(~Tile.Tile8,~TiledMap.Padding,~Operations.combine_kernel(~Tile.Tile8,~TiledMap.pairs(~operation)),~TiledMap.padded,padding,
      Tasks.Decision{parallel,tasks},shape,fill,workers,Pieces.zip(~Storage.Buffer<Tile.Tile8>,~Storage.Buffer<Tile.Tile8>,_,right_pieces)))
  %right_equation : Global.Contract(~Tile.Tile8,~Representation.Lifted<CombineLeaf.Context<Left,Right>>,
    ~Representation.lifted(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value)),after,
    Tensors.zipped_decided(~Tile.Tile8,~TiledMap.Padding,~Operations.combine_kernel(~Tile.Tile8,~TiledMap.pairs(~operation)),~TiledMap.padded,padding,
      Tasks.Decision{parallel,tasks},shape,fill,workers,Pieces.zip(~Storage.Buffer<Tile.Tile8>,~Storage.Buffer<Tile.Tile8>,Model.pack(~Tile.Tile8,left_model),_)))
  PiecesCombine.decided(~Tile.Tile8,~TiledMap.Padding,~TiledMap.pairs(~operation),~Operations.combine_kernel(~Tile.Tile8,~TiledMap.pairs(~operation)),
    ~OperationLaws.scalar_kernel(~Tile.Tile8,~TiledMap.pairs(~operation)),~TiledMap.padded,~Padding.step,~Padding.rule,
    ~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),
    ~Representation.Lifted<CombineLeaf.Context<Left,Right>>,~Representation.lifted(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value)),
    ~Operands<Left,Right>,~pair_padding(~Left,~Right),~pair_left(~Left,~Right),~pair_right(~Left,~Right),~pair_after(~Left,~Right),~pair_total(~Left,~Right),
    ~paired_rule(~Left,~left_value,~Right,~right_value,~operation),pair,Tasks.Decision{parallel,tasks},left_model,right_model,shape,fill,workers,right_fill,right_workers,total_exact,
    %Equal.sym(Pieces.Pieces(~Tile.Tile8),Model.pack(~Tile.Tile8,left_model),left_pieces,left_equation) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),pair_left(~Left,~Right,pair),Tensors.Tensor{_,shape,fill,Tensors.Usable{},workers})
      left_contract,
    %Equal.sym(Pieces.Pieces(~Tile.Tile8),Model.pack(~Tile.Tile8,right_model),right_pieces,right_equation) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),pair_right(~Left,~Right,pair),
        Tensors.Tensor{_,shape,right_fill,Tensors.Usable{},right_workers}) right_contract)

# The combination of the cells of two tensors of one layout.
def placed_combine(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+within: Bool,+left: Left,+right: Right,
  +values: List<&2,U32>,+axis: Nat,+lanes: U32,+position: Grid.Position,+parallel: Bool,+tasks: Pieces.Ranges,left_pieces: Pieces.Pieces(~Tile.Tile8),
  right_pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+right_fill: Tile.Tile8,+right_workers: U32,
  left_contract: Global.Contract(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),Representation.Lifted{left,Public.geometry_of(values,axis,position)},
    Tensors.Tensor{left_pieces,shape,fill,Tensors.Usable{},workers}),
  right_contract: Global.Contract(~Tile.Tile8,~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),Representation.Lifted{right,Public.geometry_of(values,axis,position)},
    Tensors.Tensor{right_pieces,shape,right_fill,Tensors.Usable{},right_workers}),
  +placed: PaddedCells.Placed<shape,values,axis,lanes,position,within>) ->
  Global.Contract(~Tile.Tile8,~Representation.Lifted<CombineLeaf.Context<Left,Right>>,
    ~Representation.lifted(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value)),
    Representation.Lifted{CombineLeaf.Context{left,right},Public.geometry_of(values,axis,position)},
    Tensors.zipped_decided(~Tile.Tile8,~TiledMap.Padding,~Operations.combine_kernel(~Tile.Tile8,~TiledMap.pairs(~operation)),~TiledMap.padded,
      TiledMap.padding_when(within,shape,lanes,position),Tasks.Decision{parallel,tasks},shape,fill,workers,
      Pieces.zip(~Storage.Buffer<Tile.Tile8>,~Storage.Buffer<Tile.Tile8>,left_pieces,right_pieces))):
  PaddedCells.Placed{+layout,+padding_exact,+geometry_exact,+total_exact} = placed
  +geometry = PaddedCells.layout_geometry(layout)
  +back = Equal.sym(Cells.Cells,geometry,Public.geometry_of(values,axis,position),geometry_exact)
  %padding_exact : Global.Contract(~Tile.Tile8,~Representation.Lifted<CombineLeaf.Context<Left,Right>>,
    ~Representation.lifted(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value)),
    Representation.Lifted{CombineLeaf.Context{left,right},Public.geometry_of(values,axis,position)},
    Tensors.zipped_decided(~Tile.Tile8,~TiledMap.Padding,~Operations.combine_kernel(~Tile.Tile8,~TiledMap.pairs(~operation)),~TiledMap.padded,_,
      Tasks.Decision{parallel,tasks},shape,fill,workers,Pieces.zip(~Storage.Buffer<Tile.Tile8>,~Storage.Buffer<Tile.Tile8>,left_pieces,right_pieces)))
  %geometry_exact : Global.Contract(~Tile.Tile8,~Representation.Lifted<CombineLeaf.Context<Left,Right>>,
    ~Representation.lifted(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value)),
    Representation.Lifted{CombineLeaf.Context{left,right},_},
    Tensors.zipped_decided(~Tile.Tile8,~TiledMap.Padding,~Operations.combine_kernel(~Tile.Tile8,~TiledMap.pairs(~operation)),~TiledMap.padded,PaddedCells.layout_padding(layout),
      Tasks.Decision{parallel,tasks},shape,fill,workers,Pieces.zip(~Storage.Buffer<Tile.Tile8>,~Storage.Buffer<Tile.Tile8>,left_pieces,right_pieces)))
  paired(~Left,~left_value,~Right,~right_value,~operation,Operands{left,right,layout},parallel,tasks,left_pieces,right_pieces,shape,fill,workers,right_fill,right_workers,total_exact,
    Model.observe(~Tile.Tile8,left_pieces),Model.observe(~Tile.Tile8,right_pieces),
    %back : Global.Contract(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),Representation.Lifted{left,_},
      Tensors.Tensor{left_pieces,shape,fill,Tensors.Usable{},workers}) left_contract,
    %back : Global.Contract(~Tile.Tile8,~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),Representation.Lifted{right,_},
      Tensors.Tensor{right_pieces,shape,right_fill,Tensors.Usable{},right_workers}) right_contract)

def decided_combine(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,
  +values: List<&2,U32>,+axis: Nat,+lanes: U32,+position: Grid.Position,decision: Tasks.Decision,left_pieces: Pieces.Pieces(~Tile.Tile8),right_pieces: Pieces.Pieces(~Tile.Tile8),
  +shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+right_fill: Tile.Tile8,+right_workers: U32,
  left_contract: Global.Contract(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),Representation.Lifted{left,Public.geometry_of(values,axis,position)},
    Tensors.Tensor{left_pieces,shape,fill,Tensors.Usable{},workers}),
  right_contract: Global.Contract(~Tile.Tile8,~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),Representation.Lifted{right,Public.geometry_of(values,axis,position)},
    Tensors.Tensor{right_pieces,shape,right_fill,Tensors.Usable{},right_workers}),
  +laid: Public.Laid<shape,axis,lanes,position,values>) ->
  Public.Within(~Representation.Lifted<CombineLeaf.Context<Left,Right>>,
    ~Representation.lifted(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value)),~CombineLeaf.Context<Left,Right>,
    ~Public.lifted_at(~CombineLeaf.Context<Left,Right>),CombineLeaf.Context{left,right},values,axis,lanes,position,
    Tensors.zipped_decided(~Tile.Tile8,~TiledMap.Padding,~Operations.combine_kernel(~Tile.Tile8,~TiledMap.pairs(~operation)),~TiledMap.padded,
      TiledMap.padding_of(shape,axis,lanes,position),decision,shape,fill,workers,Pieces.zip(~Storage.Buffer<Tile.Tile8>,~Storage.Buffer<Tile.Tile8>,left_pieces,right_pieces))):
  match decision:
    case Tasks.Decision{+parallel,+tasks}:
      +within = Nat.is_lt(axis,List.length(&2,U32,shape))
      KernelGemm.Both{placed_combine(~Left,~left_value,~Right,~right_value,~operation,within,left,right,values,axis,lanes,position,parallel,tasks,left_pieces,right_pieces,
        shape,fill,workers,right_fill,right_workers,left_contract,right_contract,PaddedCells.placed(within,values,axis,lanes,position,shape,{==},laid)),laid}

def laid_shape(+cells: List<&2,U32>,+axis: Nat,+lanes: U32,+position: Grid.Position,+values: List<&2,U32>,+laid: Public.Laid<cells,axis,lanes,position,values>) ->
  {Tiled.cell_shape(values,axis,position) == cells : List<&2,U32>}:
  Public.Laid{+inside,+shaped,+restored,+counted,+pinned} = laid
  shaped

# The combination of two usable tensors of one shape and one lane axis.
def usable_combine(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,
  +values: List<&2,U32>,+name: String,+operations: F32,+axis: Nat,+lanes: U32,+right_lanes: U32,+position: Grid.Position,
  left_pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,
  right_pieces: Pieces.Pieces(~Tile.Tile8),+right_shape: List<&2,U32>,+right_fill: Tile.Tile8,+right_workers: U32,
  left_contract: Public.Contract(~Left,~left_value,left,values,Tiled.Tiled{Tensors.Tensor{left_pieces,shape,fill,Tensors.Usable{},workers},axis,lanes,position}),
  right_contract: Public.Contract(~Right,~right_value,right,values,
    Tiled.Tiled{Tensors.Tensor{right_pieces,right_shape,right_fill,Tensors.Usable{},right_workers},axis,right_lanes,position})) ->
  Public.Contract(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),CombineLeaf.Context{left,right},values,
    TiledMap.combine(~TiledMap.pairs(~operation),name,operations,Tiled.Tiled{Tensors.Tensor{left_pieces,shape,fill,Tensors.Usable{},workers},axis,lanes,position},
      Tiled.Tiled{Tensors.Tensor{right_pieces,right_shape,right_fill,Tensors.Usable{},right_workers},axis,right_lanes,position})):
  KernelGemm.Both{+left_global,+left_laid} = left_contract
  KernelGemm.Both{+right_global,+right_laid} = right_contract
  +same = {Equal.trans(List<&2,U32>,right_shape,Tiled.cell_shape(values,axis,position),shape,
      Equal.sym(List<&2,U32>,Tiled.cell_shape(values,axis,position),right_shape,laid_shape(right_shape,axis,right_lanes,position,values,right_laid)),
      laid_shape(shape,axis,lanes,position,values,left_laid)) : {right_shape == shape : List<&2,U32>}}
  %Equal.sym(List<&2,U32>,right_shape,shape,same) :
    Public.Contract(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),CombineLeaf.Context{left,right},values,
      TiledMap.combine(~TiledMap.pairs(~operation),name,operations,Tiled.Tiled{Tensors.Tensor{left_pieces,shape,fill,Tensors.Usable{},workers},axis,lanes,position},
        Tiled.Tiled{Tensors.Tensor{right_pieces,_,right_fill,Tensors.Usable{},right_workers},axis,right_lanes,position}))
  %Equal.sym(Bool,Shapes.same_extents(shape,shape),True{},ShapeLaws.same_extents_reflexive(shape)) :
    Public.Contract(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),CombineLeaf.Context{left,right},values,
      Tiled.Tiled{Tensors.combine_shapes(~Tile.Tile8,~TiledMap.Padding,~Operations.combine_kernel(~Tile.Tile8,~TiledMap.pairs(~operation)),~TiledMap.padded,
        TiledMap.padding_of(shape,axis,lanes,position),name,operations,_,left_pieces,shape,fill,Tensors.Usable{},workers,
        right_pieces,shape,right_fill,Tensors.Usable{},right_workers),axis,lanes,position})
  decided_combine(~Left,~left_value,~Right,~right_value,~operation,left,right,values,axis,lanes,position,
    Tasks.decide(Tensors.grid_of(shape),U32.max(workers,right_workers),Tensors.pass_rows(2.0,operations),Tensors.pass_steps(),Tasks.no_extra()),left_pieces,right_pieces,shape,fill,U32.max(workers,right_workers),
    right_fill,right_workers,left_global,
    %same : Global.Contract(~Tile.Tile8,~Representation.Lifted<Right>,~Representation.lifted(~Right,~right_value),Representation.Lifted{right,Public.geometry_of(values,axis,position)},
      Tensors.Tensor{right_pieces,_,right_fill,Tensors.Usable{},right_workers}) right_global,
    left_laid)

# The operation at every pair of lanes of two tiled tensors of one shape whose
# lanes lie on one axis at one position keeps the contract with the combined
# reader.
def combine(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,
  +values: List<&2,U32>,+name: String,+operations: F32,+axis: Nat,+lanes: U32,+right_lanes: U32,+position: Grid.Position,
  left_cells: Tensors.Tensor<Tile.Tile8>,right_cells: Tensors.Tensor<Tile.Tile8>,
  left_contract: Public.Contract(~Left,~left_value,left,values,Tiled.Tiled{left_cells,axis,lanes,position}),
  right_contract: Public.Contract(~Right,~right_value,right,values,Tiled.Tiled{right_cells,axis,right_lanes,position})) ->
  Public.Contract(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),CombineLeaf.Context{left,right},values,
    TiledMap.combine(~TiledMap.pairs(~operation),name,operations,Tiled.Tiled{left_cells,axis,lanes,position},Tiled.Tiled{right_cells,axis,right_lanes,position})):
  match left_cells:
    case Tensors.Tensor{left_pieces,+shape,+fill,Tensors.Usable{},+workers}:
      match right_cells:
        case Tensors.Tensor{right_pieces,+right_shape,+right_fill,Tensors.Usable{},+right_workers}:
          usable_combine(~Left,~left_value,~Right,~right_value,~operation,left,right,values,name,operations,axis,lanes,right_lanes,position,
            left_pieces,shape,fill,workers,right_pieces,right_shape,right_fill,right_workers,left_contract,right_contract)
        case Tensors.Tensor{right_pieces,+right_shape,+right_fill,Tensors.Failed{+problem},+right_workers}:
          Empty.absurd(Public.Contract(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),CombineLeaf.Context{left,right},values,
            TiledMap.combine(~TiledMap.pairs(~operation),name,operations,Tiled.Tiled{Tensors.Tensor{left_pieces,shape,fill,Tensors.Usable{},workers},axis,lanes,position},
              Tiled.Tiled{Tensors.Tensor{right_pieces,right_shape,right_fill,Tensors.Failed{problem},right_workers},axis,right_lanes,position})),right_contract)
    case Tensors.Tensor{left_pieces,+shape,+fill,Tensors.Failed{+problem},+workers}:
      Empty.absurd(Public.Contract(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),CombineLeaf.Context{left,right},values,
        TiledMap.combine(~TiledMap.pairs(~operation),name,operations,Tiled.Tiled{Tensors.Tensor{left_pieces,shape,fill,Tensors.Failed{problem},workers},axis,lanes,position},
          Tiled.Tiled{right_cells,axis,right_lanes,position})),left_contract)

# ---- Cells combined with cells read through a view ----

# A lane of a left cell and a lane that agrees with the right reader below the
# count combine into the lane of the combined reader.
def paired_lane(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,valid: Bool,
  +left: Left,+right: Right,+index: U32,+tile_lane: F32,+same: {Bool.pick(F32,valid,tile_lane,0.0) == Bool.pick(F32,valid,right_value(right,index),0.0) : F32}) ->
  {Bool.pick(F32,valid,operation(Bool.pick(F32,valid,left_value(left,index),0.0),tile_lane),0.0) ==
    Bool.pick(F32,valid,CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value,CombineLeaf.Context{left,right},index),0.0) : F32}:
  match valid:
    case True{}:
      %Equal.sym(F32,tile_lane,right_value(right,index),same) : {operation(left_value(left,index),_) == operation(left_value(left,index),right_value(right,index)) : F32}
      {==}
    case False{}: {==}

# A left cell paired with a tile that agrees with the right reader at the
# lanes of the cell agrees with the combined reader.
def paired_valid(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,+left: Left,+right: Right,
  +geometry: Cells.Cells,+word: U32,tile: Tile.Tile8,
  +valid: PaddedCells.Valid<Representation.lane_count(geometry),Representation.cell_group(geometry,U32.to_nat(word)),tile,
    Representation.lifted(~Right,~right_value,Representation.Lifted{right,geometry},word)>) ->
  PaddedCells.Valid<Representation.lane_count(geometry),Representation.cell_group(geometry,U32.to_nat(word)),
    TiledMap.pairs(~operation,Representation.lifted(~Left,~left_value,Representation.Lifted{left,geometry},word),tile),
    Representation.lifted(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),
      Representation.Lifted{CombineLeaf.Context{left,right},geometry},word)>:
  match tile:
    case Tile.Tile{+a,+b,+c,+d,+e,+f,+g,+h}:
      PaddedCells.Valid{+lane0,+lane1,+lane2,+lane3,+lane4,+lane5,+lane6,+lane7} = valid
      +outer = Representation.cell_outer(geometry,U32.to_nat(word))
      +group = Representation.cell_group(geometry,U32.to_nat(word))
      +lanes = Representation.lane_count(geometry)
      PaddedCells.Valid{
        paired_lane(~Left,~left_value,~Right,~right_value,~operation,Nat.is_lt(Nat.add(0n,Nat.mul(group,8n)),lanes),left,right,
          U32.from_nat(Representation.logical_index(geometry,outer,Nat.add(0n,Nat.mul(group,8n)))),a,lane0),
        paired_lane(~Left,~left_value,~Right,~right_value,~operation,Nat.is_lt(Nat.add(1n,Nat.mul(group,8n)),lanes),left,right,
          U32.from_nat(Representation.logical_index(geometry,outer,Nat.add(1n,Nat.mul(group,8n)))),b,lane1),
        paired_lane(~Left,~left_value,~Right,~right_value,~operation,Nat.is_lt(Nat.add(2n,Nat.mul(group,8n)),lanes),left,right,
          U32.from_nat(Representation.logical_index(geometry,outer,Nat.add(2n,Nat.mul(group,8n)))),c,lane2),
        paired_lane(~Left,~left_value,~Right,~right_value,~operation,Nat.is_lt(Nat.add(3n,Nat.mul(group,8n)),lanes),left,right,
          U32.from_nat(Representation.logical_index(geometry,outer,Nat.add(3n,Nat.mul(group,8n)))),d,lane3),
        paired_lane(~Left,~left_value,~Right,~right_value,~operation,Nat.is_lt(Nat.add(4n,Nat.mul(group,8n)),lanes),left,right,
          U32.from_nat(Representation.logical_index(geometry,outer,Nat.add(4n,Nat.mul(group,8n)))),e,lane4),
        paired_lane(~Left,~left_value,~Right,~right_value,~operation,Nat.is_lt(Nat.add(5n,Nat.mul(group,8n)),lanes),left,right,
          U32.from_nat(Representation.logical_index(geometry,outer,Nat.add(5n,Nat.mul(group,8n)))),f,lane5),
        paired_lane(~Left,~left_value,~Right,~right_value,~operation,Nat.is_lt(Nat.add(6n,Nat.mul(group,8n)),lanes),left,right,
          U32.from_nat(Representation.logical_index(geometry,outer,Nat.add(6n,Nat.mul(group,8n)))),g,lane6),
        paired_lane(~Left,~left_value,~Right,~right_value,~operation,Nat.is_lt(Nat.add(7n,Nat.mul(group,8n)),lanes),left,right,
          U32.from_nat(Representation.logical_index(geometry,outer,Nat.add(7n,Nat.mul(group,8n)))),h,lane7)}

# A tile kernel of the left cells and the cells of the right that a view of
# them with the cell extents of the left reads: when the kernel of every left
# cell and the tile the view reads agrees with a target cell, the result
# holds the target values.
def viewed_placed(~Left: Data,~left_value: Left -> U32 -> F32,~Cells: Data,~cell_value: Cells -> U32 -> Tile.Tile8,~Target: Data,~target_value: Target -> U32 -> F32,
  ~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+left: Left,+cells: Cells,+target: Target,+values: List<&2,U32>,+operations: F32,+axis: Nat,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+lm: Model.Model<Tile.Tile8>,+lf: Tile.Tile8,+lw: U32,
  +rm: Model.Model<Tile.Tile8>,+right_shape: List<&2,U32>,+rf: Tile.Tile8,+rw: U32,+right_axis: Nat,+right_lanes: U32,+right_position: Grid.Position,
  +axes: List<&2,Views.Axis>,+offset: U32,+extents: {Views.axis_extents(axes) == Tiled.cell_shape(values,axis,Grid.Innermost{}) : List<&2,U32>},
  +fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(right_shape)) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,axis,Grid.Innermost{})),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,axis,Grid.Innermost{}))) == True{} : Bool},
  +cells_rank: {Nat.is_le(List.length(&2,U32,Tiled.cell_shape(values,axis,Grid.Innermost{})),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +left_laid: Public.Laid<Tiled.cell_shape(values,axis,Grid.Innermost{}),axis,Grid.lanes_at(values,axis),Grid.Innermost{},values>,
  +left_global: Global.Contract(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),
    Representation.Lifted{left,Public.geometry_of(values,axis,Grid.Innermost{})},Tensors.Tensor{Model.pack(~Tile.Tile8,lm),Tiled.cell_shape(values,axis,Grid.Innermost{}),lf,Tensors.Usable{},lw}),
  +right_global: Global.Contract(~Tile.Tile8,~Cells,~cell_value,cells,Tensors.Tensor{Model.pack(~Tile.Tile8,rm),right_shape,rf,Tensors.Usable{},rw}),
  reads: @+word: U32 -> @+inside: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(values,axis,Grid.Innermost{}))) == True{} : Bool} ->
    PaddedCells.Valid<Representation.lane_count(Public.geometry_of(values,axis,Grid.Innermost{})),
      Representation.cell_group(Public.geometry_of(values,axis,Grid.Innermost{}),U32.to_nat(word)),
      operation(Representation.lifted(~Left,~left_value,Representation.Lifted{left,Public.geometry_of(values,axis,Grid.Innermost{})},word),
        ViewGather.viewed(~Tile.Tile8,~Cells,~cell_value,ViewGather.Viewing{cells,axes,offset,[]},word)),
      Representation.lifted(~Target,~target_value,Representation.Lifted{target,Public.geometry_of(values,axis,Grid.Innermost{})},word)>,
  +placed: PaddedCells.Placed<Tiled.cell_shape(values,axis,Grid.Innermost{}),values,axis,Grid.lanes_at(values,axis),Grid.Innermost{},
    Nat.is_lt(axis,List.length(&2,U32,Tiled.cell_shape(values,axis,Grid.Innermost{})))>) ->
  Public.Contract(~Target,~target_value,target,values,TiledMap.combine_viewed(~operation,operations,
      Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,lm),Tiled.cell_shape(values,axis,Grid.Innermost{}),lf,Tensors.Usable{},lw},axis,Grid.lanes_at(values,axis),Grid.Innermost{}},
      Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,rm),right_shape,rf,Tensors.Usable{},rw},right_axis,right_lanes,right_position},axes,offset)):
  PaddedCells.Placed{+layout,+padding_exact,+geometry_exact,+total_exact} = placed
  +cells_shape = Tiled.cell_shape(values,axis,Grid.Innermost{})
  +geometry = Public.geometry_of(values,axis,Grid.Innermost{})
  +padding = TiledMap.padding_of(cells_shape,axis,Grid.lanes_at(values,axis),Grid.Innermost{})
  +positive_total = {%Equal.sym(Nat,PaddedCells.layout_total(layout),ShapeLaws.elements(cells_shape),total_exact) : {Nat.is_lt(0n,_) == True{} : Bool} cells_positive :
    {Nat.is_lt(0n,PaddedCells.layout_total(layout)) == True{} : Bool}}
  +cap = Machine.half_word(PaddedPieces.layout_limits(layout))
  +fit = {%padding_exact : {PaddedPieces.groups_fit(cap,_) == True{} : Bool} PaddedPieces.layout_fit(layout,positive_total) : {PaddedPieces.groups_fit(cap,padding) == True{} : Bool}}
  +inner = {CombineLeaf.Context{Representation.Lifted{left,geometry},ViewGather.Viewing{cells,axes,offset,[]}} : CombineLeaf.Context<Representation.Lifted<Left>,ViewGather.Viewing<Cells>>}
  +scope = {PaddedPieces.PadScope{Representation.Lifted{left,geometry},ViewGather.Viewing{cells,axes,offset,[]},padding,cap,Limits.half_word_fits(PaddedPieces.layout_limits(layout)),fit} :
    PaddedPieces.PadScope<Representation.Lifted<Left>,ViewGather.Viewing<Cells>>}
  +scope_total = Equal.trans(Nat,PaddedPieces.padded_cells(padding),PaddedCells.layout_total(layout),ShapeLaws.elements(cells_shape),
    {%padding_exact : {PaddedPieces.padded_cells(_) == PaddedCells.layout_total(layout) : Nat} PaddedPieces.layout_cells(layout) :
      {PaddedPieces.padded_cells(padding) == PaddedCells.layout_total(layout) : Nat}},total_exact)
  +cells_shaped = PiecesSpread.viewed_combined(~Tile.Tile8,~TiledMap.Padding,~operation,~Padding.step,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),
    ~Cells,~cell_value,~PaddedPieces.Padded<CombineLeaf.Context<Representation.Lifted<Left>,ViewGather.Viewing<Cells>>>,
    ~PaddedPieces.padded_value(~CombineLeaf.Context<Representation.Lifted<Left>,ViewGather.Viewing<Cells>>,~CombineLeaf.value(~Tile.Tile8,~Representation.Lifted<Left>,~ViewGather.Viewing<Cells>,
      ~operation,~Representation.lifted(~Left,~left_value),~ViewGather.viewed(~Tile.Tile8,~Cells,~cell_value))),
    ~PaddedPieces.PadScope<Representation.Lifted<Left>,ViewGather.Viewing<Cells>>,~PaddedPieces.pad_padding(~Representation.Lifted<Left>,~ViewGather.Viewing<Cells>),
    ~PaddedPieces.pad_left(~Representation.Lifted<Left>,~ViewGather.Viewing<Cells>),~PaddedPieces.pad_right(~Representation.Lifted<Left>,~ViewGather.Viewing<Cells>),
    ~PaddedPieces.pad_after(~Representation.Lifted<Left>,~ViewGather.Viewing<Cells>),~PaddedPieces.pad_total(~Representation.Lifted<Left>,~ViewGather.Viewing<Cells>),
    ~PaddedPieces.padded_rule(~operation,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),~ViewGather.Viewing<Cells>,~ViewGather.viewed(~Tile.Tile8,~Cells,~cell_value)),
    ~Operations.combine_kernel(~Tile.Tile8,~operation),~OperationLaws.scalar_kernel(~Tile.Tile8,~operation),~TiledMap.padded,~Padding.rule,
    scope,operations,lm,lf,lw,cells_shape,limits,default_capacity,axes,offset,cells,{==},rm,right_shape,rf,rw,extents,scope_total,cells_room,cells_positive,cells_rank,
    fits,right_global,left_global)
  +restated = PaddedPieces.padded_restated(~CombineLeaf.Context<Representation.Lifted<Left>,ViewGather.Viewing<Cells>>,
    ~CombineLeaf.value(~Tile.Tile8,~Representation.Lifted<Left>,~ViewGather.Viewing<Cells>,~operation,~Representation.lifted(~Left,~left_value),~ViewGather.viewed(~Tile.Tile8,~Cells,~cell_value)),
    ~Representation.Lifted<Target>,~Representation.lifted(~Target,~target_value),inner,Representation.Lifted{target,geometry},layout,cells_shape,geometry,geometry_exact,total_exact,
    word => inside => reads(word,inside),
    Machine.capacity(limits),cells_room,
    Tensors.combine_viewed(~Tile.Tile8,~TiledMap.Padding,~operation,~TiledMap.padded,padding,operations,
      Tensors.Tensor{Model.pack(~Tile.Tile8,lm),cells_shape,lf,Tensors.Usable{},lw},Tensors.Tensor{Model.pack(~Tile.Tile8,rm),right_shape,rf,Tensors.Usable{},rw},axes,offset),
    {%Equal.sym(TiledMap.Padding,PaddedCells.layout_padding(layout),padding,padding_exact) :
      Global.Shaped(~Tile.Tile8,~PaddedPieces.Padded<CombineLeaf.Context<Representation.Lifted<Left>,ViewGather.Viewing<Cells>>>,
        ~PaddedPieces.padded_value(~CombineLeaf.Context<Representation.Lifted<Left>,ViewGather.Viewing<Cells>>,~CombineLeaf.value(~Tile.Tile8,~Representation.Lifted<Left>,~ViewGather.Viewing<Cells>,
          ~operation,~Representation.lifted(~Left,~left_value),~ViewGather.viewed(~Tile.Tile8,~Cells,~cell_value))),PaddedPieces.Padded{_,inner},cells_shape,
        Tensors.combine_viewed(~Tile.Tile8,~TiledMap.Padding,~operation,~TiledMap.padded,padding,operations,
          Tensors.Tensor{Model.pack(~Tile.Tile8,lm),cells_shape,lf,Tensors.Usable{},lw},Tensors.Tensor{Model.pack(~Tile.Tile8,rm),right_shape,rf,Tensors.Usable{},rw},axes,offset))
      cells_shaped :
      Global.Shaped(~Tile.Tile8,~PaddedPieces.Padded<CombineLeaf.Context<Representation.Lifted<Left>,ViewGather.Viewing<Cells>>>,
        ~PaddedPieces.padded_value(~CombineLeaf.Context<Representation.Lifted<Left>,ViewGather.Viewing<Cells>>,~CombineLeaf.value(~Tile.Tile8,~Representation.Lifted<Left>,~ViewGather.Viewing<Cells>,
          ~operation,~Representation.lifted(~Left,~left_value),~ViewGather.viewed(~Tile.Tile8,~Cells,~cell_value))),PaddedPieces.Padded{PaddedCells.layout_padding(layout),inner},
        cells_shape,Tensors.combine_viewed(~Tile.Tile8,~TiledMap.Padding,~operation,~TiledMap.padded,padding,operations,
          Tensors.Tensor{Model.pack(~Tile.Tile8,lm),cells_shape,lf,Tensors.Usable{},lw},Tensors.Tensor{Model.pack(~Tile.Tile8,rm),right_shape,rf,Tensors.Usable{},rw},axes,offset))})
  Public.laid_contract(~Target,~target_value,target,values,
    Tensors.combine_viewed(~Tile.Tile8,~TiledMap.Padding,~operation,~TiledMap.padded,padding,operations,
      Tensors.Tensor{Model.pack(~Tile.Tile8,lm),cells_shape,lf,Tensors.Usable{},lw},Tensors.Tensor{Model.pack(~Tile.Tile8,rm),right_shape,rf,Tensors.Usable{},rw},axes,offset),
    axis,Grid.lanes_at(values,axis),Grid.Innermost{},KernelGemm.Both{restated,left_laid})

# The same of left cells of laid values without a zero extent.
def viewed_models(~Left: Data,~left_value: Left -> U32 -> F32,~Cells: Data,~cell_value: Cells -> U32 -> Tile.Tile8,~Target: Data,~target_value: Target -> U32 -> F32,
  ~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+left: Left,+cells: Cells,+target: Target,+values: List<&2,U32>,+operations: F32,+axis: Nat,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+lm: Model.Model<Tile.Tile8>,+lf: Tile.Tile8,+lw: U32,
  +rm: Model.Model<Tile.Tile8>,+right_shape: List<&2,U32>,+rf: Tile.Tile8,+rw: U32,+right_axis: Nat,+right_lanes: U32,+right_position: Grid.Position,
  +axes: List<&2,Views.Axis>,+offset: U32,+extents: {Views.axis_extents(axes) == Tiled.cell_shape(values,axis,Grid.Innermost{}) : List<&2,U32>},
  +fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(right_shape)) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +left_laid: Public.Laid<Tiled.cell_shape(values,axis,Grid.Innermost{}),axis,Grid.lanes_at(values,axis),Grid.Innermost{},values>,
  +left_global: Global.Contract(~Tile.Tile8,~Representation.Lifted<Left>,~Representation.lifted(~Left,~left_value),
    Representation.Lifted{left,Public.geometry_of(values,axis,Grid.Innermost{})},Tensors.Tensor{Model.pack(~Tile.Tile8,lm),Tiled.cell_shape(values,axis,Grid.Innermost{}),lf,Tensors.Usable{},lw}),
  +right_global: Global.Contract(~Tile.Tile8,~Cells,~cell_value,cells,Tensors.Tensor{Model.pack(~Tile.Tile8,rm),right_shape,rf,Tensors.Usable{},rw}),
  reads: @+word: U32 -> @+inside: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(values,axis,Grid.Innermost{}))) == True{} : Bool} ->
    PaddedCells.Valid<Representation.lane_count(Public.geometry_of(values,axis,Grid.Innermost{})),
      Representation.cell_group(Public.geometry_of(values,axis,Grid.Innermost{}),U32.to_nat(word)),
      operation(Representation.lifted(~Left,~left_value,Representation.Lifted{left,Public.geometry_of(values,axis,Grid.Innermost{})},word),
        ViewGather.viewed(~Tile.Tile8,~Cells,~cell_value,ViewGather.Viewing{cells,axes,offset,[]},word)),
      Representation.lifted(~Target,~target_value,Representation.Lifted{target,Public.geometry_of(values,axis,Grid.Innermost{})},word)>) ->
  Public.Contract(~Target,~target_value,target,values,TiledMap.combine_viewed(~operation,operations,
      Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,lm),Tiled.cell_shape(values,axis,Grid.Innermost{}),lf,Tensors.Usable{},lw},axis,Grid.lanes_at(values,axis),Grid.Innermost{}},
      Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,rm),right_shape,rf,Tensors.Usable{},rw},right_axis,right_lanes,right_position},axes,offset)):
  Public.Laid{+inside,+shaped,+restored,+counted,+pinned} = left_laid
  +cells_shape = Tiled.cell_shape(values,axis,Grid.Innermost{})
  +room = StorageCount.counted_room(limits,values,counted,default_capacity)
  +laid = {Public.Laid{inside,shaped,restored,counted,pinned} : Public.Laid<cells_shape,axis,Grid.lanes_at(values,axis),Grid.Innermost{},values>}
  viewed_placed(~Left,~left_value,~Cells,~cell_value,~Target,~target_value,~operation,left,cells,target,values,operations,axis,limits,default_capacity,lm,lf,lw,
    rm,right_shape,rf,rw,right_axis,right_lanes,right_position,axes,offset,extents,fits,
    Rooms.cells_room(limits,values,axis,Grid.Innermost{},nonempty,room),Rooms.cells_positive(limits,values,axis,Grid.Innermost{},nonempty,room),
    {%Equal.sym(Nat,List.length(&2,U32,cells_shape),List.length(&2,U32,values),Rooms.cell_rank(values,axis,Grid.Innermost{})) :
      {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} rank_room : {Nat.is_le(List.length(&2,U32,cells_shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}},
    laid,left_global,right_global,reads,
    PaddedCells.placed(Nat.is_lt(axis,List.length(&2,U32,cells_shape)),values,axis,Grid.lanes_at(values,axis),Grid.Innermost{},cells_shape,{==},laid))
