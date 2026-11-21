# Pieces after padding: a cell of a reader is, after padding, the cell padding
# leaves of it, a guard keeps the pieces the padding fits, and padded cells
# of a reader that agrees with a target cell hold the target cells.
import Base
import ../kernel_tile.bend as Tile
import ../kernel_gemm.bend as KernelGemm
import ../tensor_tiled_map.bend as TiledMap
import ./tensor_tiled_padding.bend as Padding
import ./tensor_tiled_padded_cells.bend as PaddedCells
import ./tensor_placed.bend as Placed
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_pieces_content.bend as Content
import ./tensor_pieces_map.bend as PiecesMap
import ./tensor_pieces_combine_leaf.bend as CombineLeaf
import ./tensor_task_runs.bend as Runs
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges
import ./boolean_logic.bend as Boolean
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_pieces.bend as Pieces
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits
import ./tensor_move_readback.bend as Held
import ./tensor_grid_ranges.bend as GridRanges
import ../tensor_view.bend as Views
import ../tensor_tasks.bend as Tasks
import ../default_machine_limits.bend as Default
import ./tensor_view_gather.bend as ViewGather
import ./tensor_view_moved.bend as ViewMoved
import ./tensor_moved_sources.bend as Sources
import ./tensor_storage_count.bend as StorageCount
import ./tensor_global_shape_count.bend as Global
import ./tensor_shape_refinement.bend as ShapeLaws
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_cells.bend as Cells
import ./tensor_tiled_public.bend as Public
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_values.bend as TiledValues
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./tensor_joined.bend as Joined
import ../storage_buffer.bend as Storage
import ./tensor_whole_values.bend as WholeValues
import ./tensor_contract.bend as Contract

# ---- A piece after padding ----

# What padding leaves in the cell of a reader.
type Padded<-Inner: Data> is Data:
  Padded{padding: TiledMap.Padding,inner: Inner}

def padded_value(~Inner: Data,~read: Inner -> U32 -> Tile.Tile8,padded: Padded<Inner>,index: U32) -> Tile.Tile8:
  Padded{+padding,+inner} = padded
  +number = U32.to_nat(index)
  Padding.cell(padding,number,read(inner,U32.from_nat(number)))

# What every cell of a padded piece reads: the padding, the reader, the piece
# and the bounds the padding needs.
type Scope<-Inner: Data> is Data:
  Scope{padding: TiledMap.Padding,inner: Inner,cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},first: U32,count: U32,tree: Tree.Tree<Tile.Tile8>,length: Nat,
    ready: Buffer.ContentReady<Tile.Tile8,Buffer.Content{tree,length}>,length_exact: {length == U32.to_nat(count) : Nat},fits: Padding.padding_fits(cap,padding,first,count),
    span: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool},
    same: {Buffer.content_length(~Tile.Tile8,Padding.step(padding,first,count,Buffer.Content{tree,length})) == length : Nat}}

def Before(~Inner: Data,~read: Inner -> U32 -> Tile.Tile8,+scope: Scope<Inner>,+index: Nat) -> Data:
  Scope{+padding,+inner,cap,cap_ok,+first,count,+tree,+length,ready,length_exact,fits,span,same} = scope
  Content.At(~Tile.Tile8,~Inner,~read,inner,first,tree,length,index)

def After(~Inner: Data,~read: Inner -> U32 -> Tile.Tile8,+scope: Scope<Inner>,+index: Nat) -> Data:
  Scope{+padding,+inner,cap,cap_ok,+first,+count,+tree,+length,ready,length_exact,fits,span,same} = scope
  +output = Padding.step(padding,first,count,Buffer.Content{tree,length})
  Content.At(~Tile.Tile8,~Padded<Inner>,~padded_value(~Inner,~read),Padded{padding,inner},first,Buffer.content_tree(~Tile.Tile8,output),Buffer.content_length(~Tile.Tile8,output),index)

# A cell of the reader is, after padding, the cell padding leaves of it.
def cell_at(~Inner: Data,~read: Inner -> U32 -> Tile.Tile8,+scope: Scope<Inner>,+index: Nat,+before: Before(~Inner,~read,scope,index)) -> After(~Inner,~read,scope,index):
  Scope{+padding,+inner,+cap,+cap_ok,+first,+count,+tree,+length,+ready,+length_exact,+fits,+span,+same} = scope
  KernelGemm.Both{+below,+equal} = before
  +output = Padding.step(padding,first,count,Buffer.Content{tree,length})
  +number = Nat.add(U32.to_nat(first),index)
  +count_inside = {%length_exact : {Nat.is_lt(index,_) == True{} : Bool} below : {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}}
  +number_room = Order.transitive(number,Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap),
    Bounds.strict_le(number,Nat.add(U32.to_nat(first),U32.to_nat(count)),Order.offset_lt(U32.to_nat(first),index,U32.to_nat(count),count_inside)),span)
  +bounded = {%Equal.sym(Nat,length,U32.to_nat(count),length_exact) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
    Order.transitive(U32.to_nat(count),Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap),Order.right_le(U32.to_nat(first),U32.to_nat(count)),span) :
    {Nat.is_le(length,U32.to_nat(cap)) == True{} : Bool}}
  +room = {%Equal.sym(Nat,length,U32.to_nat(count),length_exact) : {Nat.is_le(U32.to_nat(count),_) == True{} : Bool} Order.reflexive(U32.to_nat(count)) :
    {Nat.is_le(U32.to_nat(count),length) == True{} : Bool}}
  +back = Bounds.roundtrip(number,cap,cap_ok,number_room)
  KernelGemm.Both{%Equal.sym(Nat,Buffer.content_length(~Tile.Tile8,output),length,same) : {Nat.is_lt(index,_) == True{} : Bool} below,
    Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,output),U32.from_nat(index)),Padding.cell(padding,number,Tree.read(Tile.Tile8,tree,U32.from_nat(index))),
      Padding.cell(padding,U32.to_nat(U32.from_nat(number)),read(inner,U32.from_nat(U32.to_nat(U32.from_nat(number))))),
      Padding.read(cap,cap_ok,padding,first,count,tree,length,ready,fits,bounded,room,index,count_inside),
      %Equal.sym(Nat,U32.to_nat(U32.from_nat(number)),number,back) :
        {Padding.cell(padding,number,Tree.read(Tile.Tile8,tree,U32.from_nat(index))) == Padding.cell(padding,_,read(inner,U32.from_nat(U32.to_nat(U32.from_nat(number))))) : Tile.Tile8}
      %Equal.sym(Nat,U32.to_nat(U32.from_nat(number)),number,back) :
        {Padding.cell(padding,number,Tree.read(Tile.Tile8,tree,U32.from_nat(index))) == Padding.cell(padding,number,read(inner,U32.from_nat(_))) : Tile.Tile8}
      Equal.cong(Tile.Tile8,Tile.Tile8,tile => Padding.cell(padding,number,tile),Tree.read(Tile.Tile8,tree,U32.from_nat(index)),read(inner,U32.from_nat(number)),equal))}

def stepped_piece(~Inner: Data,~read: Inner -> U32 -> Tile.Tile8,+padding: TiledMap.Padding,+inner: Inner,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +first: U32,+count: U32,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,+ready: Buffer.ContentReady<Tile.Tile8,Buffer.Content{tree,length}>,
  +length_exact: {length == U32.to_nat(count) : Nat},+fits: Padding.padding_fits(cap,padding,first,count),
  +span: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool},
  +values: Content.piece_values(~Tile.Tile8,~Inner,~read,inner,first,count,Buffer.Content{tree,length}),
  +kept: KernelGemm.Both<Buffer.ContentReady<Tile.Tile8,Padding.step(padding,first,count,Buffer.Content{tree,length})>,
    {Buffer.content_length(~Tile.Tile8,Padding.step(padding,first,count,Buffer.Content{tree,length})) == length : Nat}>) ->
  Placed.PieceHolds(~Tile.Tile8,~Padded<Inner>,~padded_value(~Inner,~read),Padded{padding,inner},first,count,Padding.step(padding,first,count,Buffer.Content{tree,length})):
  KernelGemm.Both{+out_ready,+same} = kept
  +output = Padding.step(padding,first,count,Buffer.Content{tree,length})
  KernelGemm.Both{%Equal.sym(Buffer.Content<Tile.Tile8>,output,Buffer.Content{Buffer.content_tree(~Tile.Tile8,output),Buffer.content_length(~Tile.Tile8,output)},Buffer.content_eta(~Tile.Tile8,output)) :
      Content.piece_values(~Tile.Tile8,~Padded<Inner>,~padded_value(~Inner,~read),Padded{padding,inner},first,count,_)
    Members.transform_all(~Nat,~Scope<Inner>,~Before(~Inner,~read),~After(~Inner,~read),~cell_at(~Inner,~read),Ranges.range(U32.to_nat(count),0n),
      Scope{padding,inner,cap,cap_ok,first,count,tree,length,ready,length_exact,fits,span,same},values),
    KernelGemm.Both{Equal.trans(Nat,Buffer.content_length(~Tile.Tile8,output),length,U32.to_nat(count),same,length_exact),out_ready}}

# A piece holding the cells of a reader holds, after padding, the cells
# padding leaves of them.
def padded_piece(~Inner: Data,~read: Inner -> U32 -> Tile.Tile8,+padding: TiledMap.Padding,+inner: Inner,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +first: U32,+count: U32,input: Buffer.Content<Tile.Tile8>,+fits: Padding.padding_fits(cap,padding,first,count),
  +span: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool},
  +holds: Placed.PieceHolds(~Tile.Tile8,~Inner,~read,inner,first,count,input)) ->
  Placed.PieceHolds(~Tile.Tile8,~Padded<Inner>,~padded_value(~Inner,~read),Padded{padding,inner},first,count,Padding.step(padding,first,count,input)):
  match input:
    case Buffer.Content{+tree,+length}:
      KernelGemm.Both{+values,KernelGemm.Both{+sized,+ready}} = holds
      stepped_piece(~Inner,~read,padding,inner,cap,cap_ok,first,count,tree,length,ready,sized,fits,span,values,Padding.kept_ready(padding,first,count,tree,length,ready))

# ---- Pieces the padding fits ----

# The groups and the outer cells of a padding within a bound, and the cells
# it pads.
def groups_fit(+cap: U32,+padding: TiledMap.Padding) -> Bool:
  TiledMap.Padding{lanes,+groups,+outer,position} = padding
  Nat.is_lt(0n,U32.to_nat(groups)) && (Nat.is_le(Nat.double(U32.to_nat(groups)),U32.to_nat(cap)) && Nat.is_le(Nat.mul(U32.to_nat(groups),U32.to_nat(outer)),U32.to_nat(cap)))

def padded_cells(+padding: TiledMap.Padding) -> Nat:
  TiledMap.Padding{lanes,+groups,+outer,position} = padding
  Nat.mul(U32.to_nat(groups),U32.to_nat(outer))

def piece_inside(+cells: Nat,+first: U32,+rows: U32) -> Bool:
  Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),cells)

def padding_guard(+cap: U32,+padding: TiledMap.Padding,+first: U32,+rows: U32) -> Bool:
  groups_fit(cap,padding) && piece_inside(padded_cells(padding),first,rows)

# A piece the padding fits holds after padding what it leaves of the reader.
def PadHolds(~Inner: Data,~read: Inner -> U32 -> Tile.Tile8,guard: Bool,+padding: TiledMap.Padding,+inner: Inner,+first: U32,+rows: U32,+buffer: Buffer.Content<Tile.Tile8>) -> Data:
  match guard:
    case True{}: Placed.PieceHolds(~Tile.Tile8,~Padded<Inner>,~padded_value(~Inner,~read),Padded{padding,inner},first,rows,buffer)
    case False{}: Unit

def Guarded(~Inner: Data,~read: Inner -> U32 -> Tile.Tile8,+cap: U32,+padding: TiledMap.Padding,+inner: Inner,+first: U32,+rows: U32,+buffer: Buffer.Content<Tile.Tile8>) -> Data:
  PadHolds(~Inner,~read,padding_guard(cap,padding,first,rows),padding,inner,first,rows,buffer)

def span_of(+cap: U32,+groups: U32,+outer: U32,+first: U32,+rows: U32,
  +fit: {Nat.is_lt(0n,U32.to_nat(groups)) && (Nat.is_le(Nat.double(U32.to_nat(groups)),U32.to_nat(cap)) && Nat.is_le(Nat.mul(U32.to_nat(groups),U32.to_nat(outer)),U32.to_nat(cap))) == True{} : Bool},
  +inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(groups),U32.to_nat(outer))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(cap)) == True{} : Bool}:
  +positive = Nat.is_lt(0n,U32.to_nat(groups))
  +double = Nat.is_le(Nat.double(U32.to_nat(groups)),U32.to_nat(cap))
  +product = Nat.is_le(Nat.mul(U32.to_nat(groups),U32.to_nat(outer)),U32.to_nat(cap))
  Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(groups),U32.to_nat(outer)),U32.to_nat(cap),inside,
    Boolean.right(double,product,Boolean.right(positive,double && product,fit)))

def fits_of(+cap: U32,+lanes: U32,+groups: U32,+outer: U32,+position: Grid.Position,+first: U32,+rows: U32,
  +fit: {Nat.is_lt(0n,U32.to_nat(groups)) && (Nat.is_le(Nat.double(U32.to_nat(groups)),U32.to_nat(cap)) && Nat.is_le(Nat.mul(U32.to_nat(groups),U32.to_nat(outer)),U32.to_nat(cap))) == True{} : Bool},
  +inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(groups),U32.to_nat(outer))) == True{} : Bool}) ->
  Padding.padding_fits(cap,TiledMap.Padding{lanes,groups,outer,position},first,rows):
  +positive = Nat.is_lt(0n,U32.to_nat(groups))
  +double = Nat.is_le(Nat.double(U32.to_nat(groups)),U32.to_nat(cap))
  +product = Nat.is_le(Nat.mul(U32.to_nat(groups),U32.to_nat(outer)),U32.to_nat(cap))
  PaddedCells.fits_when(U32.is_zero((lanes % 8 : U32)),cap,U32.to_nat(groups),U32.to_nat(outer),U32.to_nat(first),U32.to_nat(rows),
    KernelGemm.Both{Boolean.left(positive,double && product,fit),KernelGemm.Both{Boolean.left(double,product,Boolean.right(positive,double && product,fit)),
      KernelGemm.Both{Boolean.right(double,product,Boolean.right(positive,double && product,fit)),span_of(cap,groups,outer,first,rows,fit,inside)}}})

# A piece the padding fits holds after padding what it leaves of the reader;
# any other piece holds nothing.
def pad_lift(~Inner: Data,~read: Inner -> U32 -> Tile.Tile8,guard: Bool,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},padding: TiledMap.Padding,+inner: Inner,
  +first: U32,+rows: U32,+input: Buffer.Content<Tile.Tile8>,+classified: {padding_guard(cap,padding,first,rows) == guard : Bool},
  +holds: Placed.PieceHolds(~Tile.Tile8,~Inner,~read,inner,first,rows,input)) ->
  PadHolds(~Inner,~read,guard,padding,inner,first,rows,Padding.step(padding,first,rows,input)):
  match guard padding:
    case True{} TiledMap.Padding{+lanes,+groups,+outer,+position}:
      +whole = {TiledMap.Padding{lanes,groups,outer,position} : TiledMap.Padding}
      +fit = Boolean.left(groups_fit(cap,whole),piece_inside(padded_cells(whole),first,rows),classified)
      +inside = Boolean.right(groups_fit(cap,whole),piece_inside(padded_cells(whole),first,rows),classified)
      padded_piece(~Inner,~read,whole,inner,cap,cap_ok,first,rows,input,fits_of(cap,lanes,groups,outer,position,first,rows,fit,inside),span_of(cap,groups,outer,first,rows,fit,inside),holds)
    case False{} padding: Unit{}

def guarded_lift(~Inner: Data,~read: Inner -> U32 -> Tile.Tile8,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+padding: TiledMap.Padding,+inner: Inner,
  +first: U32,+rows: U32,+input: Buffer.Content<Tile.Tile8>,+holds: Placed.PieceHolds(~Tile.Tile8,~Inner,~read,inner,first,rows,input)) ->
  Guarded(~Inner,~read,cap,padding,inner,first,rows,Padding.step(padding,first,rows,input)):
  pad_lift(~Inner,~read,padding_guard(cap,padding,first,rows),cap,cap_ok,padding,inner,first,rows,input,{==},holds)

# The pieces of a run lie below its bound.
def Inside(~Element: Data,+bound: Nat,+first: U32,+rows: U32,+buffer: Buffer.Content<Element>) -> Data:
  {piece_inside(bound,first,rows) == True{} : Bool}

def content_inside(~Element: Data,content: PiecesModel.Content<Element>,+unit: Nat,+bound: Nat,+run: Runs.Run(PiecesModel.content_layout(~Element,content),unit,bound)) ->
  PiecesModel.ContentAll(~Element,~Inside(~Element,bound),content):
  match content:
    case PiecesModel.PieceContent{+buffer,+first,+rows}: Runs.fits_inside(unit,U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(rows)),bound,run)
    case PiecesModel.Joined{+left,+right}:
      Held.Both{+left_run,+run_rest} = run
      Held.Both{+right_run,+glue} = run_rest
      KernelGemm.Both{content_inside(~Element,left,unit,bound,left_run),content_inside(~Element,right,unit,bound,right_run)}

# Pieces of padded cells hold what padding leaves of the reader when the
# padding fits them all.
def unguard(~Inner: Data,~read: Inner -> U32 -> Tile.Tile8,content: PiecesModel.Content<Tile.Tile8>,+cap: U32,+padding: TiledMap.Padding,+inner: Inner,
  +fit: {groups_fit(cap,padding) == True{} : Bool},+holds: PiecesModel.ContentAll(~Tile.Tile8,~Guarded(~Inner,~read,cap,padding,inner),content),
  +inside: PiecesModel.ContentAll(~Tile.Tile8,~Inside(~Tile.Tile8,padded_cells(padding)),content)) ->
  PiecesModel.ContentAll(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Padded<Inner>,~padded_value(~Inner,~read),Padded{padding,inner}),content):
  match content:
    case PiecesModel.PieceContent{+buffer,+first,+rows}:
      %inside : PadHolds(~Inner,~read,_,padding,inner,first,rows,buffer)
      %fit : PadHolds(~Inner,~read,_ && piece_inside(padded_cells(padding),first,rows),padding,inner,first,rows,buffer)
      holds
    case PiecesModel.Joined{+left,+right}:
      KernelGemm.Both{+left_holds,+right_holds} = holds
      KernelGemm.Both{+left_inside,+right_inside} = inside
      KernelGemm.Both{unguard(~Inner,~read,left,cap,padding,inner,fit,left_holds,left_inside),unguard(~Inner,~read,right,cap,padding,inner,fit,right_holds,right_inside)}

def layout_run(+cells: List<&2,U32>,+target: U32,+unit: Nat,+layout: GridRanges.GridLayout<cells,target,unit>) ->
  Runs.Run(Pieces.ranges_of_grid(Grid.grid(cells,target)),unit,ShapeLaws.elements(cells)):
  GridRanges.GridLayout{+run,+low,+high} = layout
  run

# Pieces laid as a run up to the cells the padding fits hold what padding
# leaves of the reader.
def unguarded_model(~Inner: Data,~read: Inner -> U32 -> Tile.Tile8,+cap: U32,+padding: TiledMap.Padding,+inner: Inner,+layout: Pieces.Ranges,-pieces: Pieces.Pieces(~Tile.Tile8),
  +bound: Nat,+unit: Nat,+run: Runs.Run(layout,unit,bound),+fit: {groups_fit(cap,padding) == True{} : Bool},+total: {padded_cells(padding) == bound : Nat},
  +placed: Placed.PlacedModel<Tile.Tile8,Guarded(~Inner,~read,cap,padding,inner),layout,pieces>) ->
  Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Padded<Inner>,~padded_value(~Inner,~read),Padded{padding,inner}),layout,pieces>:
  Placed.PlacedModel{+model,+exact,+laid,+holds} = placed
  +same = Equal.trans(Pieces.Ranges,layout,PiecesModel.layout(~Tile.Tile8,model),PiecesModel.content_layout(~Tile.Tile8,PiecesModel.content(~Tile.Tile8,model)),
    Equal.sym(Pieces.Ranges,PiecesModel.layout(~Tile.Tile8,model),layout,laid),PiecesModel.layout_matches_content(~Tile.Tile8,model))
  +content_run = {%same : Runs.Run(_,unit,bound) run : Runs.Run(PiecesModel.content_layout(~Tile.Tile8,PiecesModel.content(~Tile.Tile8,model)),unit,bound)}
  +inside = {%Equal.sym(Nat,padded_cells(padding),bound,total) : PiecesModel.ContentAll(~Tile.Tile8,~Inside(~Tile.Tile8,_),PiecesModel.content(~Tile.Tile8,model))
    content_inside(~Tile.Tile8,PiecesModel.content(~Tile.Tile8,model),unit,bound,content_run) :
    PiecesModel.ContentAll(~Tile.Tile8,~Inside(~Tile.Tile8,padded_cells(padding)),PiecesModel.content(~Tile.Tile8,model))}
  Placed.PlacedModel{model,exact,laid,unguard(~Inner,~read,PiecesModel.content(~Tile.Tile8,model),cap,padding,inner,fit,holds,inside)}

# ---- What padding leaves in a cell ----

# Padding leaves in a cell of a layout the target cell that agrees with the
# cell lane by lane below the lane count and holds +0 past it.
def layout_cell(layout: PaddedCells.Layout,+word: U32,+inside: {Nat.is_lt(U32.to_nat(word),PaddedCells.layout_total(layout)) == True{} : Bool},+tile: Tile.Tile8,+target: Tile.Tile8,
  +valid: PaddedCells.Valid<Representation.lane_count(PaddedCells.layout_geometry(layout)),Representation.cell_group(PaddedCells.layout_geometry(layout),U32.to_nat(word)),tile,target>) ->
  {Padding.cell(PaddedCells.layout_padding(layout),U32.to_nat(word),tile) == target : Tile.Tile8}:
  match layout:
    case PaddedCells.Layout{+limits,+lanes,+padded_outer,+padded_position,+outer,+after,+position,+total,+cover}:
      Public.Cover{+counted,+cells_count,+padded_cells,+placement} = cover
      +number = U32.to_nat(word)
      +groups = U32.to_nat(Grid.groups(lanes))
      +outer_count = U32.to_nat(outer)
      +lane_count = U32.to_nat(lanes)
      +capacity = U32.to_nat(Machine.capacity(limits))
      +half = U32.to_nat(Machine.half_word(limits))
      +product_positive = {%Equal.sym(Nat,Nat.mul(groups,outer_count),total,cells_count) : {Nat.is_lt(0n,_) == True{} : Bool} Order.positive_bound(number,total,inside) :
          {Nat.is_lt(0n,Nat.mul(groups,outer_count)) == True{} : Bool}}
      +outer_positive = DivisionOrder.right_factor_positive(groups,outer_count,product_positive)
      +lanes_bound = Order.transitive(lane_count,Nat.mul(lane_count,outer_count),capacity,RowMajor.product_covers(lane_count,outer_count,outer_positive),counted)
      +facts = TiledValues.grouped(limits,lanes,lanes_bound)
      +total_bound = {%cells_count : {Nat.is_le(_,capacity) == True{} : Bool}
          Order.transitive(Nat.mul(groups,outer_count),Nat.mul(lane_count,outer_count),capacity,RowMajor.mul_le(groups,lane_count,outer_count,TiledValues.fewer_of(limits,lanes,facts)),counted) :
          {Nat.is_le(total,capacity) == True{} : Bool}}
      +total_half = Order.transitive(total,capacity,half,total_bound,Bounds.as_nat_le(Machine.capacity(limits),Machine.half_word(limits),Limits.capacity_within_half(limits)))
      PaddedCells.padded_index(limits,lanes,padded_outer,padded_position,outer,after,position,total,number,tile,target,facts,cells_count,placement,total_half,inside,
        %Equal.sym(U32,U32.from_nat(number),word,Arithmetic.native_roundtrip(word)) :
          PaddedCells.Valid<lane_count,Representation.cell_group(Cells.Cells{outer,after,lanes,Tiled.groups(lanes),position},U32.to_nat(_)),tile,target>
        valid)

# A padded cell of a reader that agrees with a target cell holds it.
def padded_agree(~Inner: Data,~read: Inner -> U32 -> Tile.Tile8,~Target: Data,~target_read: Target -> U32 -> Tile.Tile8,+inner: Inner,+target: Target,+layout: PaddedCells.Layout,
  +cells: List<&2,U32>,+geometry: Cells.Cells,+geometry_exact: {PaddedCells.layout_geometry(layout) == geometry : Cells.Cells},
  +total_exact: {PaddedCells.layout_total(layout) == ShapeLaws.elements(cells) : Nat},+word: U32,+inside: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(cells)) == True{} : Bool},
  +valid: PaddedCells.Valid<Representation.lane_count(geometry),Representation.cell_group(geometry,U32.to_nat(word)),read(inner,word),target_read(target,word)>) ->
  {padded_value(~Inner,~read,Padded{PaddedCells.layout_padding(layout),inner},word) == target_read(target,word) : Tile.Tile8}:
  +geometry_back = Equal.sym(Cells.Cells,PaddedCells.layout_geometry(layout),geometry,geometry_exact)
  %Equal.sym(U32,U32.from_nat(U32.to_nat(word)),word,Arithmetic.native_roundtrip(word)) :
    {Padding.cell(PaddedCells.layout_padding(layout),U32.to_nat(word),read(inner,_)) == target_read(target,word) : Tile.Tile8}
  layout_cell(layout,word,
    {%Equal.sym(Nat,PaddedCells.layout_total(layout),ShapeLaws.elements(cells),total_exact) : {Nat.is_lt(U32.to_nat(word),_) == True{} : Bool} inside :
      {Nat.is_lt(U32.to_nat(word),PaddedCells.layout_total(layout)) == True{} : Bool}},
    read(inner,word),target_read(target,word),
    {%geometry_back : PaddedCells.Valid<Representation.lane_count(_),Representation.cell_group(PaddedCells.layout_geometry(layout),U32.to_nat(word)),read(inner,word),target_read(target,word)>
      %geometry_back : PaddedCells.Valid<Representation.lane_count(geometry),Representation.cell_group(_,U32.to_nat(word)),read(inner,word),target_read(target,word)>
      valid :
      PaddedCells.Valid<Representation.lane_count(PaddedCells.layout_geometry(layout)),Representation.cell_group(PaddedCells.layout_geometry(layout),U32.to_nat(word)),read(inner,word),target_read(target,word)>})

# Padded cells of a reader whose every cell agrees with a target cell hold
# the target cells.
def padded_restated(~Inner: Data,~read: Inner -> U32 -> Tile.Tile8,~Target: Data,~target_read: Target -> U32 -> Tile.Tile8,+inner: Inner,+target: Target,+layout: PaddedCells.Layout,
  +cells: List<&2,U32>,+geometry: Cells.Cells,+geometry_exact: {PaddedCells.layout_geometry(layout) == geometry : Cells.Cells},
  +total_exact: {PaddedCells.layout_total(layout) == ShapeLaws.elements(cells) : Nat},
  agree: @+word: U32 -> @+inside: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(cells)) == True{} : Bool} ->
    PaddedCells.Valid<Representation.lane_count(geometry),Representation.cell_group(geometry,U32.to_nat(word)),read(inner,word),target_read(target,word)>,
  +capacity: U32,+fits: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(capacity)) == True{} : Bool},-tensor: Tensors.Tensor<Tile.Tile8>,
  +shaped: Global.Shaped(~Tile.Tile8,~Padded<Inner>,~padded_value(~Inner,~read),Padded{PaddedCells.layout_padding(layout),inner},cells,tensor)) ->
  Global.Shaped(~Tile.Tile8,~Target,~target_read,target,cells,tensor):
  Global.restated_below(~Tile.Tile8,~Padded<Inner>,~Target,~padded_value(~Inner,~read),~target_read,~ShapeLaws.elements(cells),Padded{PaddedCells.layout_padding(layout),inner},target,
    word => inside => padded_agree(~Inner,~read,~Target,~target_read,inner,target,layout,cells,geometry,geometry_exact,total_exact,word,inside,agree(word,inside)),
    cells,tensor,shaped,Order.reflexive(ShapeLaws.elements(cells)),capacity,fits)

def layout_limits(+layout: PaddedCells.Layout) -> Machine.Limits:
  PaddedCells.Layout{+limits,+lanes,+padded_outer,+padded_position,+outer,+after,+position,+total,+cover} = layout
  limits

# The padding of a layout covers its cells.
def layout_cells(+layout: PaddedCells.Layout) -> {padded_cells(PaddedCells.layout_padding(layout)) == PaddedCells.layout_total(layout) : Nat}:
  match layout:
    case PaddedCells.Layout{+limits,+lanes,+padded_outer,+padded_position,+outer,+after,+position,+total,+cover}:
      Public.Cover{+counted,+cells_count,+padded_count,+placement} = cover
      padded_count

# The padding of a layout of some cells fits half a word of its limits.
def layout_fit(+layout: PaddedCells.Layout,+positive: {Nat.is_lt(0n,PaddedCells.layout_total(layout)) == True{} : Bool}) ->
  {groups_fit(Machine.half_word(layout_limits(layout)),PaddedCells.layout_padding(layout)) == True{} : Bool}:
  match layout:
    case PaddedCells.Layout{+limits,+lanes,+padded_outer,+padded_position,+outer,+after,+position,+total,+cover}:
      Public.Cover{+counted,+cells_count,+padded_count,+placement} = cover
      +groups = U32.to_nat(Grid.groups(lanes))
      +outer_count = U32.to_nat(outer)
      +lane_count = U32.to_nat(lanes)
      +capacity = U32.to_nat(Machine.capacity(limits))
      +half = U32.to_nat(Machine.half_word(limits))
      +padded_count_nat = Nat.mul(groups,U32.to_nat(padded_outer))
      +product_positive = {%Equal.sym(Nat,Nat.mul(groups,outer_count),total,cells_count) : {Nat.is_lt(0n,_) == True{} : Bool} positive :
          {Nat.is_lt(0n,Nat.mul(groups,outer_count)) == True{} : Bool}}
      +outer_positive = DivisionOrder.right_factor_positive(groups,outer_count,product_positive)
      +groups_positive = RowMajor.left_factor_positive(groups,outer_count,product_positive)
      +lanes_bound = Order.transitive(lane_count,Nat.mul(lane_count,outer_count),capacity,RowMajor.product_covers(lane_count,outer_count,outer_positive),counted)
      +facts = TiledValues.grouped(limits,lanes,lanes_bound)
      +total_bound = {%cells_count : {Nat.is_le(_,capacity) == True{} : Bool}
          Order.transitive(Nat.mul(groups,outer_count),Nat.mul(lane_count,outer_count),capacity,RowMajor.mul_le(groups,lane_count,outer_count,TiledValues.fewer_of(limits,lanes,facts)),counted) :
          {Nat.is_le(total,capacity) == True{} : Bool}}
      +total_half = Order.transitive(total,capacity,half,total_bound,Bounds.as_nat_le(Machine.capacity(limits),Machine.half_word(limits),Limits.capacity_within_half(limits)))
      +groups_bound = Order.transitive(groups,total,capacity,
        %cells_count : {Nat.is_le(groups,_) == True{} : Bool} RowMajor.product_covers(groups,outer_count,outer_positive),total_bound)
      +padded_half = {%Equal.sym(Nat,padded_count_nat,total,padded_count) : {Nat.is_le(_,half) == True{} : Bool} total_half : {Nat.is_le(padded_count_nat,half) == True{} : Bool}}
      %Equal.sym(Bool,Nat.is_lt(0n,groups),True{},groups_positive) :
        {_ && (Nat.is_le(Nat.double(groups),half) && Nat.is_le(padded_count_nat,half)) == True{} : Bool}
      %Equal.sym(Bool,Nat.is_le(Nat.double(groups),half),True{},Limits.double_bound(limits,groups,groups_bound)) :
        {True{} && (_ && Nat.is_le(padded_count_nat,half)) == True{} : Bool}
      %Equal.sym(Bool,Nat.is_le(padded_count_nat,half),True{},padded_half) : {True{} && (True{} && _) == True{} : Bool}
      {==}

# ---- A padded view ----

# A view of a tensor of the contract that the runtime accepts, with a target
# of nonzero cells that the padding covers and fits, gives a ready tensor of
# its extents holding what the padding leaves of the viewed values.
def padded_view(~Logical: Data,~original: Logical -> U32 -> Tile.Tile8,+logical: Logical,+limits: Machine.Limits,+name: String,+model: PiecesModel.Model<Tile.Tile8>,
  +source: List<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32,+flags: List<&2,Bool>,+fill: Tile.Tile8,+workers: U32,
  +contract: Global.Contract(~Tile.Tile8,~Logical,~original,logical,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),source,fill,Tensors.Usable{},workers}),
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +source_rank: {Nat.is_le(List.length(&2,U32,source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_rank: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +count: U32,+counted: {Tensors.count_of(Views.axis_extents(axes)) == Some{count} : Maybe<&2,U32>},+nonzero: {U32.is_zero(count) == False{} : Bool},
  +fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(source)) == True{} : Bool},
  +padding: TiledMap.Padding,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+fit: {groups_fit(cap,padding) == True{} : Bool},
  +total: {padded_cells(padding) == ShapeLaws.elements(Views.axis_extents(axes)) : Nat}) ->
  Global.Shaped(~Tile.Tile8,~Padded<ViewGather.Viewing<Logical>>,~padded_value(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~Tile.Tile8,~Logical,~original)),
    Padded{padding,ViewGather.Viewing{logical,axes,offset,flags}},Views.axis_extents(axes),
    Tensors.moved_then(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,padding,name,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),source,fill,Tensors.Usable{},workers},axes,offset,flags)):
  +extents = Views.axis_extents(axes)
  +viewing = {ViewGather.Viewing{logical,axes,offset,flags} : ViewGather.Viewing<Logical>}
  +padded = {Padded{padding,viewing} : Padded<ViewGather.Viewing<Logical>>}
  +source_counted = Global.contract_counted(~Tile.Tile8,~Logical,~original,logical,PiecesModel.pack(~Tile.Tile8,model),source,fill,workers,contract)
  +source_count = Global.count(source,source_counted)
  +facts = ViewMoved.accepted_facts(limits,source,axes,offset,flags,source_count,Global.accepted(source,source_counted),count,counted,nonzero,default_capacity,source_rank,target_rank,fits)
  +room = StorageCount.counted_room(limits,extents,Global.Counted{count,counted},default_capacity)
  +positive = ViewMoved.counted_positive(extents,count,counted,nonzero)
  +divides = {GridRanges.Multiple{GridRanges.last_extent(extents),Equal.sym(Nat,Nat.mul(GridRanges.last_extent(extents),1n),GridRanges.last_extent(extents),Order.times_one(GridRanges.last_extent(extents)))} :
    GridRanges.Multiple<GridRanges.last_extent(extents),1n>}
  %Equal.sym(Bool,Views.fits(Views.View{axes,offset},Tensors.count_or_zero(source)),True{},fits) :
    Global.Shaped(~Tile.Tile8,~Padded<ViewGather.Viewing<Logical>>,~padded_value(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~Tile.Tile8,~Logical,~original)),padded,extents,
      Tensors.moved_status(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,padding,name,Tensors.Usable{},_,Tensors.count_of(extents),PiecesModel.pack(~Tile.Tile8,model),source,extents,axes,offset,flags,fill,workers))
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(extents),Some{count},counted) :
    Global.Shaped(~Tile.Tile8,~Padded<ViewGather.Viewing<Logical>>,~padded_value(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~Tile.Tile8,~Logical,~original)),padded,extents,
      Tensors.moved_status(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,padding,name,Tensors.Usable{},True{},_,PiecesModel.pack(~Tile.Tile8,model),source,extents,axes,offset,flags,fill,workers))
  %Equal.sym(Bool,U32.is_zero(count),False{},nonzero) :
    Global.Shaped(~Tile.Tile8,~Padded<ViewGather.Viewing<Logical>>,~padded_value(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~Tile.Tile8,~Logical,~original)),padded,extents,
      Tensors.moved_counted(~Tile.Tile8,~TiledMap.Padding,~TiledMap.padded,padding,_,PiecesModel.pack(~Tile.Tile8,model),source,extents,axes,offset,flags,fill,workers))
  Global.shaped(~Tile.Tile8,~Padded<ViewGather.Viewing<Logical>>,~padded_value(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~Tile.Tile8,~Logical,~original)),padded,extents,
    Tensors.Tensor{Tensors.moved_with(~Tile.Tile8,~Tensors.Then<Tensors.Viewed,TiledMap.Padding>,~Tensors.then_gather(~Tile.Tile8,~Tensors.Viewed,~TiledMap.Padding,~Tensors.view_gather(~Tile.Tile8),~TiledMap.padded),
      Tasks.components(source,Tensors.grid_of(source),Tensors.grid_of(extents),Views.View{axes,offset},flags),
      Tensors.Then{Tensors.Viewed{source,Tensors.grid_of(extents),extents,axes,offset,flags},padding},Tensors.move_rows(),0.0,
      Tensors.Source{PiecesModel.pack(~Tile.Tile8,model),Tensors.Ended{}},[Tensors.origin(source,offset)],extents,fill,workers),extents,fill,Tensors.Usable{},workers},{==},{==},
    GridRanges.grid_contract(~Tile.Tile8,~Padded<ViewGather.Viewing<Logical>>,~padded_value(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~Tile.Tile8,~Logical,~original)),padded,limits,extents,fill,workers,
      Tensors.moved_with(~Tile.Tile8,~Tensors.Then<Tensors.Viewed,TiledMap.Padding>,~Tensors.then_gather(~Tile.Tile8,~Tensors.Viewed,~TiledMap.Padding,~Tensors.view_gather(~Tile.Tile8),~TiledMap.padded),
        Tasks.components(source,Tensors.grid_of(source),Tensors.grid_of(extents),Views.View{axes,offset},flags),
        Tensors.Then{Tensors.Viewed{source,Tensors.grid_of(extents),extents,axes,offset,flags},padding},Tensors.move_rows(),0.0,
        Tensors.Source{PiecesModel.pack(~Tile.Tile8,model),Tensors.Ended{}},[Tensors.origin(source,offset)],extents,fill,workers),
      room,positive,1n,divides,Global.Counted{count,counted},
      unguarded_model(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~Tile.Tile8,~Logical,~original),cap,padding,viewing,Pieces.ranges_of_grid(Tensors.grid_of(extents)),
        Tensors.moved_with(~Tile.Tile8,~Tensors.Then<Tensors.Viewed,TiledMap.Padding>,~Tensors.then_gather(~Tile.Tile8,~Tensors.Viewed,~TiledMap.Padding,~Tensors.view_gather(~Tile.Tile8),~TiledMap.padded),
          Tasks.components(source,Tensors.grid_of(source),Tensors.grid_of(extents),Views.View{axes,offset},flags),
          Tensors.Then{Tensors.Viewed{source,Tensors.grid_of(extents),extents,axes,offset,flags},padding},Tensors.move_rows(),0.0,
          Tensors.Source{PiecesModel.pack(~Tile.Tile8,model),Tensors.Ended{}},[Tensors.origin(source,offset)],extents,fill,workers),
        ShapeLaws.elements(extents),1n,layout_run(extents,Grid.piece_target(),1n,GridRanges.grid_layout(limits,extents,Grid.piece_target(),room,positive,1n,divides)),fit,total,
        ViewMoved.view_then(~Tile.Tile8,~Logical,~original,~TiledMap.Padding,~TiledMap.padded,~Padding.step,~Padding.rule,
          ~(later => viewing => first => rows => buffer => Guarded(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~Tile.Tile8,~Logical,~original),cap,later,viewing,first,rows,buffer)),
          ~(later => viewing => first => rows => inner => holds => guarded_lift(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~Tile.Tile8,~Logical,~original),cap,cap_ok,later,viewing,first,rows,inner,holds)),
          padding,logical,limits,model,source,axes,offset,flags,fill,workers,
          Sources.source_placed(~Tile.Tile8,~Logical,~original,logical,model,source,fill,workers,contract),facts))))

# ---- Combined cells, padded ----

# What every piece of a combination of cells padded after it reads: the two
# readers, the padding and a bound it fits.
type PadScope<-Left: Data,-Right: Data> is Data:
  PadScope{left: Left,right: Right,padding: TiledMap.Padding,cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},fit: {groups_fit(cap,padding) == True{} : Bool}}

def pad_padding(~Left: Data,~Right: Data,+scope: PadScope<Left,Right>) -> TiledMap.Padding:
  PadScope{+left,+right,+padding,+cap,+cap_ok,+fit} = scope
  padding

def pad_left(~Left: Data,~Right: Data,+scope: PadScope<Left,Right>) -> Left:
  PadScope{+left,+right,+padding,+cap,+cap_ok,+fit} = scope
  left

def pad_right(~Left: Data,~Right: Data,+scope: PadScope<Left,Right>) -> Right:
  PadScope{+left,+right,+padding,+cap,+cap_ok,+fit} = scope
  right

def pad_after(~Left: Data,~Right: Data,+scope: PadScope<Left,Right>) -> Padded<CombineLeaf.Context<Left,Right>>:
  PadScope{+left,+right,+padding,+cap,+cap_ok,+fit} = scope
  Padded{padding,CombineLeaf.Context{left,right}}

def pad_total(~Left: Data,~Right: Data,+scope: PadScope<Left,Right>) -> Nat:
  PadScope{+left,+right,+padding,+cap,+cap_ok,+fit} = scope
  padded_cells(padding)

# A piece that holds after padding steps into it.
def held_stepped(~Inner: Data,~read: Inner -> U32 -> Tile.Tile8,+padding: TiledMap.Padding,+inner: Inner,+first: U32,+count: U32,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,
  +length_exact: {length == U32.to_nat(count) : Nat},
  +holds: Placed.PieceHolds(~Tile.Tile8,~Padded<Inner>,~padded_value(~Inner,~read),Padded{padding,inner},first,count,Padding.step(padding,first,count,Buffer.Content{tree,length}))) ->
  PiecesMap.Stepped(~Tile.Tile8,~Padded<Inner>,~padded_value(~Inner,~read),Padded{padding,inner},first,count,Buffer.Content{tree,length},Padding.step(padding,first,count,Buffer.Content{tree,length})):
  KernelGemm.Both{+values,KernelGemm.Both{+sized,+ready}} = holds
  KernelGemm.Both{ready,KernelGemm.Both{Equal.trans(Nat,Buffer.content_length(~Tile.Tile8,Padding.step(padding,first,count,Buffer.Content{tree,length})),U32.to_nat(count),length,sized,
    Equal.sym(Nat,length,U32.to_nat(count),length_exact)),values}}

def padded_combined(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,~Left: Data,~left_value: Left -> U32 -> Tile.Tile8,~Right: Data,~right_value: Right -> U32 -> Tile.Tile8,+left: Left,+right: Right,padding: TiledMap.Padding,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +fit: {groups_fit(cap,padding) == True{} : Bool},+first: U32,+count: U32,input: Buffer.Content<Tile.Tile8>,+ready: Buffer.ContentReady<Tile.Tile8,input>,
  +length_exact: {Buffer.content_length(~Tile.Tile8,input) == U32.to_nat(count) : Nat},
  +inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),padded_cells(padding)) == True{} : Bool},
  +values: Content.piece_values(~Tile.Tile8,~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~Tile.Tile8,~Left,~Right,~operation,~left_value,~right_value),CombineLeaf.Context{left,right},first,count,input)) ->
  PiecesMap.Stepped(~Tile.Tile8,~Padded<CombineLeaf.Context<Left,Right>>,~padded_value(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~Tile.Tile8,~Left,~Right,~operation,~left_value,~right_value)),Padded{padding,CombineLeaf.Context{left,right}},first,count,input,
    Padding.step(padding,first,count,input)):
  match padding:
    case TiledMap.Padding{+lanes,+groups,+outer,+position}:
      match input:
        case Buffer.Content{+tree,+length}:
          +whole = {TiledMap.Padding{lanes,groups,outer,position} : TiledMap.Padding}
          held_stepped(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~Tile.Tile8,~Left,~Right,~operation,~left_value,~right_value),whole,CombineLeaf.Context{left,right},first,count,tree,length,length_exact,
            stepped_piece(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~Tile.Tile8,~Left,~Right,~operation,~left_value,~right_value),whole,CombineLeaf.Context{left,right},cap,cap_ok,first,count,tree,length,ready,length_exact,
              fits_of(cap,lanes,groups,outer,position,first,count,fit,inside),span_of(cap,groups,outer,first,count,fit,inside),values,
              Padding.kept_ready(whole,first,count,tree,length,ready)))

# Every piece of a combination of cells holds, after padding, what the padding
# leaves of the combined cells.
def padded_rule(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,~Left: Data,~left_value: Left -> U32 -> Tile.Tile8,~Right: Data,~right_value: Right -> U32 -> Tile.Tile8,+scope: PadScope<Left,Right>,+first: U32,+count: U32,+input: Buffer.Content<Tile.Tile8>,+ready: Buffer.ContentReady<Tile.Tile8,input>,
  +length_exact: {Buffer.content_length(~Tile.Tile8,input) == U32.to_nat(count) : Nat},
  +inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),pad_total(~Left,~Right,scope)) == True{} : Bool},
  +values: Content.piece_values(~Tile.Tile8,~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~Tile.Tile8,~Left,~Right,~operation,~left_value,~right_value),CombineLeaf.Context{pad_left(~Left,~Right,scope),pad_right(~Left,~Right,scope)},first,count,input)) ->
  PiecesMap.Stepped(~Tile.Tile8,~Padded<CombineLeaf.Context<Left,Right>>,~padded_value(~CombineLeaf.Context<Left,Right>,~CombineLeaf.value(~Tile.Tile8,~Left,~Right,~operation,~left_value,~right_value)),pad_after(~Left,~Right,scope),first,count,input,
    Padding.step(pad_padding(~Left,~Right,scope),first,count,input)):
  PadScope{+left,+right,+padding,+cap,+cap_ok,+fit} = scope
  padded_combined(~operation,~Left,~left_value,~Right,~right_value,left,right,padding,cap,cap_ok,fit,first,count,input,ready,length_exact,inside,values)

# ---- The whole tensor of padded cells ----

# The whole tensor of a padded buffer whose cells, from cell zero, hold a
# reader holds what the padding leaves of the reader.
def padded_whole(~Inner: Data,~read: Inner -> U32 -> Tile.Tile8,+inner: Inner,+limits: Machine.Limits,+padding: TiledMap.Padding,+shape: List<&2,U32>,+fill: Tile.Tile8,
  +workers: U32,+count: U32,+values: Buffer.Model<Tile.Tile8>,+counted: Global.Counted<shape>,+exact: {ShapeLaws.elements(shape) == U32.to_nat(count) : Nat},
  +room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},
  +fits: Padding.padding_fits(Machine.half_word(limits),padding,0,count),
  +holds: Placed.PieceHolds(~Tile.Tile8,~Inner,~read,inner,0,count,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,values)))) ->
  Global.Contract(~Tile.Tile8,~Padded<Inner>,~padded_value(~Inner,~read),Padded{padding,inner},
    Tensors.whole(~Tile.Tile8,TiledMap.padded(padding,0,count,Buffer.pack(~Tile.Tile8,values)),shape,fill,Tensors.Usable{},workers)):
  +content = Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,values))
  +reader = {Padded{padding,inner} : Padded<Inner>}
  +span = {%exact : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} room : {Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +half = Machine.half_word(limits)
  +span_half = Order.transitive(U32.to_nat(count),U32.to_nat(Machine.capacity(limits)),U32.to_nat(half),span,
    Bounds.as_nat_le(Machine.capacity(limits),half,Limits.capacity_within_half(limits)))
  +stepped = padded_piece(~Inner,~read,padding,inner,half,Limits.half_word_fits(limits),0,count,content,fits,span_half,holds)
  +moved = {%Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,TiledMap.padded(padding,0,count,Buffer.pack(~Tile.Tile8,values))),Padding.step(padding,0,count,content),
        Padding.rule(padding,0,count,Buffer.pack(~Tile.Tile8,values))) :
      Placed.PieceHolds(~Tile.Tile8,~Padded<Inner>,~padded_value(~Inner,~read),reader,0,count,_)
    stepped : Placed.PieceHolds(~Tile.Tile8,~Padded<Inner>,~padded_value(~Inner,~read),reader,0,count,
      Buffer.contents(~Tile.Tile8,TiledMap.padded(padding,0,count,Buffer.pack(~Tile.Tile8,values))))}
  +source = Buffer.remodel(~Tile.Tile8,TiledMap.padded(padding,0,count,Buffer.pack(~Tile.Tile8,values)))
  +facts = {%Equal.sym(Storage.Buffer<Tile.Tile8>,Buffer.pack(~Tile.Tile8,source),TiledMap.padded(padding,0,count,Buffer.pack(~Tile.Tile8,values)),
        Buffer.remodeled(~Tile.Tile8,TiledMap.padded(padding,0,count,Buffer.pack(~Tile.Tile8,values)))) :
      Contract.Facts(~Tile.Tile8,~Padded<Inner>,~padded_value(~Inner,~read),reader,shape,Buffer.contents(~Tile.Tile8,_))
    KernelGemm.Both{Joined.piece_formed(~Tile.Tile8,~Padded<Inner>,~padded_value(~Inner,~read),reader,count,shape,exact,
        Buffer.contents(~Tile.Tile8,TiledMap.padded(padding,0,count,Buffer.pack(~Tile.Tile8,values))),moved),
      Joined.piece_whole(~Tile.Tile8,~Padded<Inner>,~padded_value(~Inner,~read),reader,count,
        Buffer.contents(~Tile.Tile8,TiledMap.padded(padding,0,count,Buffer.pack(~Tile.Tile8,values))),moved)} :
    Contract.Facts(~Tile.Tile8,~Padded<Inner>,~padded_value(~Inner,~read),reader,shape,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,source)))}
  %Buffer.remodeled(~Tile.Tile8,TiledMap.padded(padding,0,count,Buffer.pack(~Tile.Tile8,values))) :
    Global.Contract(~Tile.Tile8,~Padded<Inner>,~padded_value(~Inner,~read),reader,Tensors.whole(~Tile.Tile8,_,shape,fill,Tensors.Usable{},workers))
  WholeValues.whole_contract(~Tile.Tile8,~Padded<Inner>,~padded_value(~Inner,~read),reader,limits,shape,fill,workers,source,counted,room,nonempty,facts)
