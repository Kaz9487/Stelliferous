# Retiling row-major values that hold a reader's values in a counted shape
# recovers the public tiled contract for both nonempty and empty shapes.
import Base
import ../kernel_tile.bend as Tile
import ../kernel_gemm.bend as KernelGemm
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ../storage_buffer.bend as Storage
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_pieces.bend as Pieces
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_cells.bend as Cells
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./tensor_contract.bend as Contract
import ./tensor_global_shape_count.bend as Global
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_pieces_content.bend as Content
import ./tensor_pieces_plan.bend as PiecesPlan
import ./tensor_pieces_each.bend as PiecesEach
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_count_refinement.bend as Count
import ./tensor_storage_count.bend as StorageCount
import ./tensor_cell_room.bend as Rooms
import ./tensor_grid_pieces.bend as GridPieces
import ./tensor_tiled_public.bend as Public
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_values.bend as TiledValues
import ./tensor_tiled_entry.bend as Entry
import ./tensor_tiled_padded_cells.bend as PaddedCells
import ./traversal_indexed_values.bend as IndexedValues
import ./kernel_tile_refinement.bend as TileLaws
import ./nat_order.bend as Order
import ./row_major_index.bend as RowMajor
import ./nat_division_order.bend as DivisionOrder
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_comparison.bend as Comparison
import ./word_encoding.bend as WordEncoding

# ---- Nonempty shapes ----

# Stored row-major values and a logical reader that agree on a shape, both
# lifted in the geometry of the values.
def SameValues(~Logical: Data,~original: Logical -> U32 -> F32,+values: List<&2,U32>,+geometry: Cells.Cells,
  before: Representation.Lifted<Buffer.Content<F32>>,after: Representation.Lifted<Logical>) -> Data:
  match before after:
    case Representation.Lifted{content,before_geometry} Representation.Lifted{logical,after_geometry}:
      KernelGemm.Both<Contract.Facts(~F32,~Logical,~original,logical,values,content),
        KernelGemm.Both<{before_geometry == geometry : Cells.Cells},{after_geometry == geometry : Cells.Cells}>>

def lane_stored_cases(~Logical: Data,~original: Logical -> U32 -> F32,valid: Bool,+values: List<&2,U32>,+geometry: Cells.Cells,+blocks: Nat,
  +outer: Nat,+channel: Nat,+content: Buffer.Content<F32>,+logical: Logical,
  +facts: Contract.Facts(~F32,~Logical,~original,logical,values,content),
  +outer_in_blocks: {Nat.is_lt(outer,Nat.mul(blocks,Representation.after_count(geometry))) == True{} : Bool},
  +elements_exact: {ShapeLaws.elements(values) == Nat.mul(Nat.mul(blocks,Representation.lane_count(geometry)),Representation.after_count(geometry)) : Nat},
  +classified: {Nat.is_lt(channel,Representation.lane_count(geometry)) == valid : Bool}) ->
  {Bool.pick(F32,valid,Contract.stored(~F32,content,U32.from_nat(Representation.logical_index(geometry,outer,channel))),0.0) ==
    Bool.pick(F32,valid,original(logical,U32.from_nat(Representation.logical_index(geometry,outer,channel))),0.0) : F32}:
  match valid:
    case False{}: {==}
    case True{}:
      KernelGemm.Both{Contract.WellFormed{+length_exact,ready},+values_ok} = facts
      +index = Representation.logical_index(geometry,outer,channel)
      +below = Representation.logical_below(geometry,outer,channel,blocks,outer_in_blocks,classified)
      +inside_values = {%Equal.sym(Nat,ShapeLaws.elements(values),Nat.mul(Nat.mul(blocks,Representation.lane_count(geometry)),Representation.after_count(geometry)),elements_exact) :
        {Nat.is_lt(index,_) == True{} : Bool} below : {Nat.is_lt(index,ShapeLaws.elements(values)) == True{} : Bool}}
      +inside_content = {%Equal.sym(Nat,Buffer.content_length(~F32,content),ShapeLaws.elements(values),length_exact) :
        {Nat.is_lt(index,_) == True{} : Bool} inside_values : {Nat.is_lt(index,Buffer.content_length(~F32,content)) == True{} : Bool}}
      Contract.read(~F32,~Logical,~original,logical,content,index,values_ok,inside_content)

def lane_stored(~Logical: Data,~original: Logical -> U32 -> F32,+values: List<&2,U32>,+geometry: Cells.Cells,+blocks: Nat,
  +outer: Nat,+group: Nat,+lane: Nat,+content: Buffer.Content<F32>,+logical: Logical,
  +facts: Contract.Facts(~F32,~Logical,~original,logical,values,content),
  +outer_in_blocks: {Nat.is_lt(outer,Nat.mul(blocks,Representation.after_count(geometry))) == True{} : Bool},
  +elements_exact: {ShapeLaws.elements(values) == Nat.mul(Nat.mul(blocks,Representation.lane_count(geometry)),Representation.after_count(geometry)) : Nat}) ->
  {Representation.lane(~Buffer.Content<F32>,~Contract.stored(~F32),content,geometry,outer,group,lane) ==
    Representation.lane(~Logical,~original,logical,geometry,outer,group,lane) : F32}:
  +channel = Nat.add(lane,Nat.mul(group,8n))
  lane_stored_cases(~Logical,~original,Nat.is_lt(channel,Representation.lane_count(geometry)),values,geometry,blocks,outer,channel,content,logical,facts,outer_in_blocks,elements_exact,{==})

def outer_below_blocks(+geometry: Cells.Cells,+blocks: Nat,+total: Nat,+number: Nat,+cells_count: {Nat.mul(Representation.group_count(geometry),Representation.outer_count(geometry)) == total : Nat},
  +outer_exact: {Representation.outer_count(geometry) == Nat.mul(blocks,Representation.after_count(geometry)) : Nat},
  +outer_positive: {Nat.is_lt(0n,Representation.outer_count(geometry)) == True{} : Bool},
  +groups_positive: {Nat.is_lt(0n,Representation.group_count(geometry)) == True{} : Bool},
  +inside: {Nat.is_lt(number,total) == True{} : Bool}) ->
  {Nat.is_lt(Representation.cell_outer(geometry,number),Nat.mul(blocks,Representation.after_count(geometry))) == True{} : Bool}:
  Cells.Cells{+outer_word,+after_word,+lanes_word,+groups_word,+position} = geometry
  +outer_in = Representation.outer_inside(position,U32.to_nat(outer_word),U32.to_nat(groups_word),number,
    %Equal.sym(Nat,Nat.mul(U32.to_nat(groups_word),U32.to_nat(outer_word)),total,cells_count) : {Nat.is_lt(number,_) == True{} : Bool} inside,outer_positive,groups_positive)
  %outer_exact :
    {Nat.is_lt(Representation.outer_at(position,U32.to_nat(outer_word),U32.to_nat(groups_word),number),_) == True{} : Bool}
  outer_in

def lifted_stored_agree(~Logical: Data,~original: Logical -> U32 -> F32,+values: List<&2,U32>,+geometry: Cells.Cells,+blocks: Nat,+total: Nat,
  +cells_count: {Nat.mul(Representation.group_count(geometry),Representation.outer_count(geometry)) == total : Nat},
  +outer_exact: {Representation.outer_count(geometry) == Nat.mul(blocks,Representation.after_count(geometry)) : Nat},
  +elements_exact: {ShapeLaws.elements(values) == Nat.mul(Nat.mul(blocks,Representation.lane_count(geometry)),Representation.after_count(geometry)) : Nat},
  +outer_positive: {Nat.is_lt(0n,Representation.outer_count(geometry)) == True{} : Bool},
  +groups_positive: {Nat.is_lt(0n,Representation.group_count(geometry)) == True{} : Bool},
  +before: Representation.Lifted<Buffer.Content<F32>>,+after: Representation.Lifted<Logical>,
  +related: SameValues(~Logical,~original,values,geometry,before,after),+word: U32,+inside: {Nat.is_lt(U32.to_nat(word),total) == True{} : Bool}) ->
  {Representation.lifted(~Buffer.Content<F32>,~Contract.stored(~F32),before,word) == Representation.lifted(~Logical,~original,after,word) : Tile.Tile8}:
  match before after:
    case Representation.Lifted{+content,+before_geometry} Representation.Lifted{+logical,+after_geometry}:
      KernelGemm.Both{+facts,KernelGemm.Both{+here,+there}} = related
      %Equal.sym(Cells.Cells,before_geometry,geometry,here) :
        {Representation.lifted(~Buffer.Content<F32>,~Contract.stored(~F32),Representation.Lifted{content,_},word) ==
          Representation.lifted(~Logical,~original,Representation.Lifted{logical,after_geometry},word) : Tile.Tile8}
      %Equal.sym(Cells.Cells,after_geometry,geometry,there) :
        {Representation.lifted(~Buffer.Content<F32>,~Contract.stored(~F32),Representation.Lifted{content,geometry},word) ==
          Representation.lifted(~Logical,~original,Representation.Lifted{logical,_},word) : Tile.Tile8}
      +number = U32.to_nat(word)
      +outer = Representation.cell_outer(geometry,number)
      +group = Representation.cell_group(geometry,number)
      +outer_in_blocks = outer_below_blocks(geometry,blocks,total,number,cells_count,outer_exact,outer_positive,groups_positive,inside)
      TileLaws.tiles_equal(Representation.cell(~Buffer.Content<F32>,~Contract.stored(~F32),content,geometry,outer,group),
        Representation.cell(~Logical,~original,logical,geometry,outer,group),
        lane_stored(~Logical,~original,values,geometry,blocks,outer,group,0n,content,logical,facts,outer_in_blocks,elements_exact),
        lane_stored(~Logical,~original,values,geometry,blocks,outer,group,1n,content,logical,facts,outer_in_blocks,elements_exact),
        lane_stored(~Logical,~original,values,geometry,blocks,outer,group,2n,content,logical,facts,outer_in_blocks,elements_exact),
        lane_stored(~Logical,~original,values,geometry,blocks,outer,group,3n,content,logical,facts,outer_in_blocks,elements_exact),
        lane_stored(~Logical,~original,values,geometry,blocks,outer,group,4n,content,logical,facts,outer_in_blocks,elements_exact),
        lane_stored(~Logical,~original,values,geometry,blocks,outer,group,5n,content,logical,facts,outer_in_blocks,elements_exact),
        lane_stored(~Logical,~original,values,geometry,blocks,outer,group,6n,content,logical,facts,outer_in_blocks,elements_exact),
        lane_stored(~Logical,~original,values,geometry,blocks,outer,group,7n,content,logical,facts,outer_in_blocks,elements_exact))

def cover_cells(-limits: Machine.Limits,-lanes: U32,-padded_outer: U32,-padded_position: Grid.Position,-outer: U32,-position: Grid.Position,-total: Nat,
  cover: Public.Cover<limits,lanes,padded_outer,padded_position,outer,position,total>) ->
  {Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer)) == total : Nat}:
  Public.Cover{counted,cells,padded_cells,placement} = cover
  cells

def cells_product_cases(within: Bool,+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+lanes: U32,+position: Grid.Position,+count: U32,
  +classified: {Nat.is_lt(axis,List.length(&2,U32,values)) == within : Bool},+exact: {ShapeLaws.elements(values) == U32.to_nat(count) : Nat},
  +bounded: {U32.is_le(count,Machine.capacity(limits)) == True{} : Bool},+pinned: {Grid.lanes_at(values,axis) == lanes : U32}) ->
  {Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))) ==
    ShapeLaws.elements(Tiled.cell_shape(values,axis,position)) : Nat}:
  match within:
    case True{}:
      cover_cells(limits,lanes,Grid.outer_count(Tiled.cell_shape(values,axis,position),position),position,
        (Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),position,
        ShapeLaws.elements(Tiled.cell_shape(values,axis,position)),
        Public.along_cover(limits,values,axis,lanes,position,count,exact,bounded,classified,pinned))
    case False{}:
      +one = Equal.trans(U32,lanes,Grid.lanes_at(values,axis),1,Equal.sym(U32,Grid.lanes_at(values,axis),lanes,pinned),Public.lanes_past(values,axis,classified))
      %Equal.sym(U32,lanes,1,one) :
        {Nat.mul(U32.to_nat(Grid.groups(_)),U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))) ==
          ShapeLaws.elements(Tiled.cell_shape(values,axis,position)) : Nat}
      cover_cells(limits,1,Grid.product(Tiled.cell_shape(values,axis,position)),Grid.Innermost{},
        (Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),position,
        ShapeLaws.elements(Tiled.cell_shape(values,axis,position)),
        Public.single_cover(limits,values,axis,position,count,exact,bounded,classified))

def relabel_split(~Logical: Data,~original: Logical -> U32 -> F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +logical: Logical,+values: List<&2,U32>,+content: Buffer.Content<F32>,+facts: Contract.Facts(~F32,~Logical,~original,logical,values,content),
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,
  +axis: Nat,+lanes: U32,+position: Grid.Position,
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+shaped: {Tiled.cell_shape(values,axis,position) == shape : List<&2,U32>},
  +restored: {Tiled.logical_shape(shape,axis,lanes,position) == values : List<&2,U32>},+counted: Global.Counted<values>,
  +pinned: {Grid.lanes_at(values,axis) == lanes : U32},
  +global: Global.Contract(~Tile.Tile8,~Representation.Lifted<Buffer.Content<F32>>,~Representation.lifted(~Buffer.Content<F32>,~Contract.stored(~F32)),
    Representation.Lifted{content,Public.geometry_of(values,axis,position)},Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}),
  +split: Public.Split<limits,values,axis,lanes>) ->
  Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},axis,lanes,position}):
  Public.Split{+blocks,+outer_exact,+elements_exact,lanes_within,after_positive} = split
  +val_count = Global.count(values,counted)
  +val_evidence = Global.evidence(values,counted)
  +val_exact = Count.count_exact(ShapeLaws.elements(values),val_count,val_evidence)
  +val_bounded = {%Equal.sym(U32,Machine.capacity(limits),Machine.capacity(Default.limits()),default_capacity) : {U32.is_le(val_count,_) == True{} : Bool}
    PaddedCells.count_bounded(ShapeLaws.elements(values),val_count,val_evidence) : {U32.is_le(val_count,Machine.capacity(limits)) == True{} : Bool}}
  +values_room = StorageCount.counted_room(limits,values,counted,default_capacity)
  +outer_word = (Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32)
  +after_word = Tiled.product(Public.lane_after(values,axis))
  +geometry = Public.geometry_of(values,axis,position)
  +explicit = {Cells.Cells{outer_word,after_word,lanes,Tiled.groups(lanes),position} : Cells.Cells}
  +geometry_same = Equal.trans(Cells.Cells,geometry,Cells.Cells{outer_word,after_word,Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),position},explicit,
    Public.geometry_exact(values,axis,position,inside),
    Equal.cong(U32,Cells.Cells,word => Cells.Cells{outer_word,after_word,word,Tiled.groups(word),position},Grid.lanes_at(values,axis),lanes,pinned))
  +geometry_back = Equal.sym(Cells.Cells,geometry,explicit,geometry_same)
  +total = ShapeLaws.elements(Tiled.cell_shape(values,axis,position))
  +cells_count = cells_product_cases(Nat.is_lt(axis,List.length(&2,U32,values)),limits,values,axis,lanes,position,val_count,{==},val_exact,val_bounded,pinned)
  +cells_pos = Rooms.cells_positive(limits,values,axis,position,nonempty,values_room)
  +product_pos = {%Equal.sym(Nat,Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer_word)),total,cells_count) : {Nat.is_lt(0n,_) == True{} : Bool} cells_pos :
    {Nat.is_lt(0n,Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer_word))) == True{} : Bool}}
  +groups_pos = RowMajor.left_factor_positive(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer_word),product_pos)
  +outer_pos = DivisionOrder.right_factor_positive(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer_word),product_pos)
  +geom_cells_count = {%geometry_back : {Nat.mul(Representation.group_count(_),Representation.outer_count(_)) == total : Nat} cells_count :
    {Nat.mul(Representation.group_count(geometry),Representation.outer_count(geometry)) == total : Nat}}
  +geom_outer_exact = {%geometry_back : {Representation.outer_count(_) == Nat.mul(blocks,Representation.after_count(_)) : Nat} outer_exact :
    {Representation.outer_count(geometry) == Nat.mul(blocks,Representation.after_count(geometry)) : Nat}}
  +geom_elements_exact = {%geometry_back : {ShapeLaws.elements(values) == Nat.mul(Nat.mul(blocks,Representation.lane_count(_)),Representation.after_count(_)) : Nat} elements_exact :
    {ShapeLaws.elements(values) == Nat.mul(Nat.mul(blocks,Representation.lane_count(geometry)),Representation.after_count(geometry)) : Nat}}
  +geom_outer_pos = {%geometry_back : {Nat.is_lt(0n,Representation.outer_count(_)) == True{} : Bool} outer_pos : {Nat.is_lt(0n,Representation.outer_count(geometry)) == True{} : Bool}}
  +geom_groups_pos = {%geometry_back : {Nat.is_lt(0n,Representation.group_count(_)) == True{} : Bool} groups_pos : {Nat.is_lt(0n,Representation.group_count(geometry)) == True{} : Bool}}
  +counted_cells = Global.contract_counted(~Tile.Tile8,~Representation.Lifted<Buffer.Content<F32>>,~Representation.lifted(~Buffer.Content<F32>,~Contract.stored(~F32)),
    Representation.Lifted{content,geometry},pieces,shape,fill,workers,global)
  +cell_count = Global.count(shape,counted_cells)
  +cell_exact = Count.count_exact(ShapeLaws.elements(shape),cell_count,Global.evidence(shape,counted_cells))
  +room_cells = {%Equal.sym(List<&2,U32>,Tiled.cell_shape(values,axis,position),shape,shaped) : {Nat.is_le(ShapeLaws.elements(shape),ShapeLaws.elements(_)) == True{} : Bool}
    Order.reflexive(ShapeLaws.elements(shape)) : {Nat.is_le(ShapeLaws.elements(shape),total) == True{} : Bool}}
  +fits_cells = {%Equal.sym(List<&2,U32>,Tiled.cell_shape(values,axis,position),shape,shaped) : {Nat.is_le(ShapeLaws.elements(_),U32.to_nat(cell_count)) == True{} : Bool}
    %cell_exact : {Nat.is_le(ShapeLaws.elements(shape),_) == True{} : Bool} Order.reflexive(ShapeLaws.elements(shape)) :
    {Nat.is_le(total,U32.to_nat(cell_count)) == True{} : Bool}}
  KernelGemm.Both{
    Global.relabeled_below(~Tile.Tile8,~Representation.Lifted<Buffer.Content<F32>>,~Representation.Lifted<Logical>,
      ~Representation.lifted(~Buffer.Content<F32>,~Contract.stored(~F32)),~Representation.lifted(~Logical,~original),
      ~SameValues(~Logical,~original,values,geometry),~total,
      ~lifted_stored_agree(~Logical,~original,values,geometry,blocks,total,geom_cells_count,geom_outer_exact,geom_elements_exact,geom_outer_pos,geom_groups_pos),
      Representation.Lifted{content,geometry},Representation.Lifted{logical,geometry},KernelGemm.Both{facts,KernelGemm.Both{{==},{==}}},
      shape,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},Global.usable(~Tile.Tile8,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},{==}),
      global,room_cells,cell_count,fits_cells),
    Public.Laid{inside,shaped,restored,counted,pinned}}

def relabel_stored(~Logical: Data,~original: Logical -> U32 -> F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +logical: Logical,+values: List<&2,U32>,+content: Buffer.Content<F32>,+facts: Contract.Facts(~F32,~Logical,~original,logical,values,content),
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},tiled: Tiled.Tiled,
  stored: Public.Contract(~Buffer.Content<F32>,~Contract.stored(~F32),content,values,tiled)) ->
  Public.Contract(~Logical,~original,logical,values,tiled):
  match tiled:
    case Tiled.Tiled{cells,+axis,+lanes,+position}:
      match cells:
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Usable{},+workers}:
          KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}} = stored
          +values_room = StorageCount.counted_room(limits,values,counted,default_capacity)
          relabel_split(~Logical,~original,limits,default_capacity,logical,values,content,facts,nonempty,pieces,shape,fill,workers,axis,lanes,position,
            inside,shaped,restored,counted,pinned,global,Public.values_split(limits,values,axis,lanes,pinned,values_room,nonempty))
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Failed{+problem},+workers}:
          Empty.absurd(Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers},axis,lanes,position}),stored)

def tiled_nonempty(~Logical: Data,~original: Logical -> U32 -> F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +logical: Logical,+values: List<&2,U32>,+workers: U32,+source: Buffer.Model<F32>,+counted: Global.Counted<values>,
  +facts: Contract.Facts(~F32,~Logical,~original,logical,values,Buffer.content(~F32,source)),+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool}) ->
  Public.Contract(~Logical,~original,logical,values,Tiled.tiled_shape(Tensors.Usable{},values,workers,Buffer.pack(~F32,source))):
  +packed_facts = {%Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,source)),Buffer.content(~F32,source),Buffer.packed_contents(~F32,source)) :
    Contract.Facts(~F32,~Logical,~original,logical,values,_) facts : Contract.Facts(~F32,~Logical,~original,logical,values,Buffer.contents(~F32,Buffer.pack(~F32,source)))}
  relabel_stored(~Logical,~original,limits,default_capacity,logical,values,Buffer.contents(~F32,Buffer.pack(~F32,source)),packed_facts,nonempty,
    Tiled.tiled_shape(Tensors.Usable{},values,workers,Buffer.pack(~F32,source)),Entry.tiled_stored(values,workers,source,counted,nonempty))

# ---- Empty shapes ----

def nat_zero_ge_false(+value: Nat,+positive: {Nat.is_lt(0n,value) == True{} : Bool}) -> {Nat.is_ge(0n,value) == False{} : Bool}:
  match value:
    case 0n: Empty.absurd({Nat.is_ge(0n,0n) == False{} : Bool},WordEncoding.false_true(positive))
    case 1n+rest: {==}

def zero_ge_false(+divisor: U32,+nonzero: {U32.is_zero(divisor) == False{} : Bool}) -> {U32.is_ge(0,divisor) == False{} : Bool}:
  %Equal.sym(Bool,U32.is_ge(0,divisor),Nat.is_ge(0n,U32.to_nat(divisor)),Comparison.ge(0,divisor)) : {_ == False{} : Bool}
  nat_zero_ge_false(U32.to_nat(divisor),Arithmetic.word_nonzero_positive(divisor,nonzero))

def zero_divmod(m: Nat,+divisor: U32,+not_ge: {U32.is_ge(0,divisor) == False{} : Bool}) ->
  {U32.divmod.go(m,Word.zero(m),divisor) == (Word.zero(m),0) : Word(m) & U32}:
  match m:
    case 0n: {==}
    case 1n+ +p:
      %Equal.sym(Word(p) & U32,U32.divmod.go(p,Word.zero(p),divisor),(Word.zero(p),0),zero_divmod(p,divisor,not_ge)) :
        {U32.divmod.go.rec(p,False{},divisor,_) == (WCon{False{},Word.zero(p)},0) : Word(1n+p) & U32}
      %Equal.sym(Bool,U32.is_ge(0,divisor),False{},not_ge) :
        {U32.divmod.go.fin(p,Word.zero(p),0,divisor,_) == (WCon{False{},Word.zero(p)},0) : Word(1n+p) & U32}
      {==}

def zero_div_when(zero: Bool,+divisor: U32,+classified: {U32.is_zero(divisor) == zero : Bool}) ->
  {U32.div.if(Word.zero(32n),divisor,zero) == 0 : U32}:
  match zero:
    case True{}: {==}
    case False{}:
      Equal.cong(Word(32n) & U32,U32,U32.div.fin,U32.divmod.go(32n,Word.zero(32n),divisor),(Word.zero(32n),0),
        zero_divmod(32n,divisor,zero_ge_false(divisor,classified)))

# Zero divided by any word is zero.
def zero_div(+divisor: U32) -> {(0 / divisor : U32) == 0 : U32}:
  U32{+bw} = divisor
  zero_div_when(U32.is_zero(U32{bw}),U32{bw},{==})

def right_zero(+a: Nat,+b: Nat,+positive: {Nat.is_lt(0n,a) == True{} : Bool},+product: {Nat.mul(a,b) == 0n : Nat}) -> {b == 0n : Nat}:
  match a b:
    case 0n b: Empty.absurd({b == 0n : Nat},WordEncoding.false_true(positive))
    case 1n+p 0n: {==}
    case 1n+ +p 1n+ +q:
      Empty.absurd({1n+q == 0n : Nat},WordEncoding.false_true(Equal.cong(Nat,Bool,n => Nat.is_eq(n,0n),Nat.mul(1n+p,1n+q),0n,product)))

def empty_cells_lane(zero: Bool,+lanes: U32,+others: Nat,+classified: {U32.is_zero(lanes) == zero : Bool},
  +product: {Nat.mul(U32.to_nat(lanes),others) == 0n : Nat}) -> {Nat.mul(U32.to_nat(Grid.groups(lanes)),others) == 0n : Nat}:
  match zero:
    case True{}:
      %Equal.sym(U32,lanes,0,Arithmetic.word_zero(lanes,classified)) : {Nat.mul(U32.to_nat(Grid.groups(_)),others) == 0n : Nat}
      {==}
    case False{}:
      %Equal.sym(Nat,others,0n,right_zero(U32.to_nat(lanes),others,Arithmetic.word_nonzero_positive(lanes,classified),product)) :
        {Nat.mul(U32.to_nat(Grid.groups(lanes)),_) == 0n : Nat}
      RowMajor.multiply_zero_right(U32.to_nat(Grid.groups(lanes)))

def empty_cells_when(within: Bool,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,
  +classified: {Nat.is_lt(axis,List.length(&2,U32,values)) == within : Bool},+empty: {ShapeLaws.elements(values) == 0n : Nat}) ->
  {ShapeLaws.elements(Tiled.cell_shape(values,axis,position)) == 0n : Nat}:
  match within:
    case False{}:
      %Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,values)),False{},classified) :
        {ShapeLaws.elements(Grid.placed(_,position,Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))) == 0n : Nat}
      %Equal.sym(List<&2,U32>,Grid.removed(values,axis),values,Public.removed_past(values,axis,classified)) : {ShapeLaws.elements(_) == 0n : Nat}
      empty
    case True{}:
      +lanes = Grid.lanes_at(values,axis)
      +others = ShapeLaws.elements(Grid.removed(values,axis))
      +product = Equal.trans(Nat,Nat.mul(U32.to_nat(lanes),others),ShapeLaws.elements(values),0n,
        Equal.sym(Nat,ShapeLaws.elements(values),Nat.mul(U32.to_nat(lanes),others),Public.elements_split(values,axis,classified)),empty)
      Equal.trans(Nat,ShapeLaws.elements(Tiled.cell_shape(values,axis,position)),Nat.mul(U32.to_nat(Grid.groups(lanes)),others),0n,
        Public.cells_split(values,axis,position,classified),empty_cells_lane(U32.is_zero(lanes),lanes,others,{==},product))

# The cells of an empty shape are empty.
def empty_cells(+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+empty: {ShapeLaws.elements(values) == 0n : Nat}) ->
  {ShapeLaws.elements(Tiled.cell_shape(values,axis,position)) == 0n : Nat}:
  empty_cells_when(Nat.is_lt(axis,List.length(&2,U32,values)),values,axis,position,{==},empty)

# The grid of an empty cell shape has one empty range.
def empty_grid_ranges(+cells: List<&2,U32>,+empty: {ShapeLaws.elements(cells) == 0n : Nat}) ->
  {Pieces.ranges_of_grid(Tensors.grid_of(cells)) == Pieces.Range{0,0} : Pieces.Ranges}:
  +product_zero = Arithmetic.to_nat_injective(Grid.product(cells),0,
    Equal.trans(Nat,U32.to_nat(Grid.product(cells)),ShapeLaws.elements(cells),0n,
      Public.product_exact(2,{==},cells,%Equal.sym(Nat,ShapeLaws.elements(cells),0n,empty) : {Nat.is_le(_,2n) == True{} : Bool} {==}),empty))
  %Equal.sym(Grid.Grid,Grid.grid(cells,Grid.piece_target()),
      Grid.grid_of_units(Grid.units_from(cells,Grid.strides(cells),U32.is_le(Grid.product(cells),Grid.piece_target()),0,1,1,Grid.piece_target()),Grid.piece_target()),
      GridPieces.grid_units(cells,Grid.piece_target())) :
    {Pieces.ranges_of_grid(_) == Pieces.Range{0,0} : Pieces.Ranges}
  %Equal.sym(U32,Grid.product(cells),0,product_zero) :
    {Pieces.ranges_of_grid(Grid.grid_of_units(Grid.units_from(cells,Grid.strides(cells),U32.is_le(_,Grid.piece_target()),0,1,1,Grid.piece_target()),Grid.piece_target())) ==
      Pieces.Range{0,0} : Pieces.Ranges}
  %Equal.sym(Grid.Units,Grid.units_from(cells,Grid.strides(cells),True{},0,1,1,Grid.piece_target()),Grid.Units{0,1,1,Grid.product(cells)},
      GridPieces.units_fit(cells,0,1,1,Grid.piece_target())) :
    {Pieces.ranges_of_grid(Grid.grid_of_units(_,Grid.piece_target())) == Pieces.Range{0,0} : Pieces.Ranges}
  %Equal.sym(U32,Grid.product(cells),0,product_zero) :
    {Pieces.ranges_of_grid(Grid.grid_of_units(Grid.Units{0,1,1,_},Grid.piece_target())) == Pieces.Range{0,0} : Pieces.Ranges}
  {==}

def empty_piece_content(+outer: U32,+after: U32,+lanes: U32,+groups: U32,+values: Buffer.Content<F32>) ->
  {Entry.piece(Tiled.Tiling{outer,after,lanes,groups,Grid.Innermost{}},values,0,0) == Buffer.Content{Entry.fresh_cells(0),0n} : Buffer.Content<Tile.Tile8>}:
  %Equal.sym(U32,(0 / U32.max(groups,1) : U32),0,zero_div(U32.max(groups,1))) :
    {Buffer.Content{IndexedValues.output_tree(~Tree.Tree<F32>,~Tile.Tile8,TiledValues.tiled_blocks(_,Cells.Tiling{after,lanes,groups,(0 / U32.max(groups,1) : U32)},
        Buffer.content_tree(~F32,values),Buffer.tree(~Tile.Tile8,Entry.cells_model(0)))),Buffer.length(~Tile.Tile8,Entry.cells_model(0))} ==
      Buffer.Content{Entry.fresh_cells(0),0n} : Buffer.Content<Tile.Tile8>}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,Entry.cells_model(0)),Entry.fresh_cells(0),Entry.cells_tree(0)) :
    {Buffer.Content{_,Buffer.length(~Tile.Tile8,Entry.cells_model(0))} == Buffer.Content{Entry.fresh_cells(0),0n} : Buffer.Content<Tile.Tile8>}
  %Equal.sym(Nat,Buffer.length(~Tile.Tile8,Entry.cells_model(0)),0n,Entry.cells_length(0)) :
    {Buffer.Content{Entry.fresh_cells(0),_} == Buffer.Content{Entry.fresh_cells(0),0n} : Buffer.Content<Tile.Tile8>}
  {==}

def built_snd(pair: Storage.Buffer<F32> & Pieces.Pieces(~Tile.Tile8)) ->
  {PiecesModel.contents(~Tile.Tile8,Pair.snd(Storage.Buffer<F32>,Pieces.Pieces(~Tile.Tile8),pair)) ==
    Pair.snd(Buffer.Content<F32>,PiecesModel.Content<Tile.Tile8>,PiecesEach.seen(~Tile.Tile8,~Storage.Buffer<F32>,~Buffer.Content<F32>,~Buffer.contents(~F32),pair)) :
    PiecesModel.Content<Tile.Tile8>}:
  (state,pieces) = pair
  {==}

def empty_built_contents(+outer: U32,+after: U32,+lanes: U32,+groups: U32,+layout: Pieces.Ranges,+source: Buffer.Model<F32>,
  +range_zero: {layout == Pieces.Range{0,0} : Pieces.Ranges}) ->
  {PiecesModel.contents(~Tile.Tile8,Pair.snd(Storage.Buffer<F32>,Pieces.Pieces(~Tile.Tile8),
      Pieces.build(~Storage.Buffer<Tile.Tile8>,~Storage.Buffer<F32>,~Tiled.Tiling,~Tiled.tiled_piece,layout,Tiled.Tiling{outer,after,lanes,groups,Grid.Innermost{}},
        Buffer.pack(~F32,source)))) ==
    PiecesModel.PieceContent{Buffer.Content{Entry.fresh_cells(0),0n},0,0} : PiecesModel.Content<Tile.Tile8>}:
  +tiling = {Tiled.Tiling{outer,after,lanes,groups,Grid.Innermost{}} : Tiled.Tiling}
  %Equal.sym(Pieces.Ranges,layout,Pieces.Range{0,0},range_zero) :
    {PiecesModel.contents(~Tile.Tile8,Pair.snd(Storage.Buffer<F32>,Pieces.Pieces(~Tile.Tile8),
        Pieces.build(~Storage.Buffer<Tile.Tile8>,~Storage.Buffer<F32>,~Tiled.Tiling,~Tiled.tiled_piece,_,tiling,Buffer.pack(~F32,source)))) ==
      PiecesModel.PieceContent{Buffer.Content{Entry.fresh_cells(0),0n},0,0} : PiecesModel.Content<Tile.Tile8>}
  +piece_content = {PiecesModel.PieceContent{Entry.piece(tiling,Buffer.contents(~F32,Buffer.pack(~F32,source)),0,0),0,0} : PiecesModel.Content<Tile.Tile8>}
  Equal.trans(PiecesModel.Content<Tile.Tile8>,
    PiecesModel.contents(~Tile.Tile8,Pair.snd(Storage.Buffer<F32>,Pieces.Pieces(~Tile.Tile8),Tiled.tiled_piece(tiling,0,0,Buffer.pack(~F32,source)))),
    piece_content,
    PiecesModel.PieceContent{Buffer.Content{Entry.fresh_cells(0),0n},0,0},
    Equal.trans(PiecesModel.Content<Tile.Tile8>,
      PiecesModel.contents(~Tile.Tile8,Pair.snd(Storage.Buffer<F32>,Pieces.Pieces(~Tile.Tile8),Tiled.tiled_piece(tiling,0,0,Buffer.pack(~F32,source)))),
      Pair.snd(Buffer.Content<F32>,PiecesModel.Content<Tile.Tile8>,
        PiecesEach.seen(~Tile.Tile8,~Storage.Buffer<F32>,~Buffer.Content<F32>,~Buffer.contents(~F32),Tiled.tiled_piece(tiling,0,0,Buffer.pack(~F32,source)))),
      piece_content,
      built_snd(Tiled.tiled_piece(tiling,0,0,Buffer.pack(~F32,source))),
      Equal.cong(Buffer.Content<F32> & PiecesModel.Content<Tile.Tile8>,PiecesModel.Content<Tile.Tile8>,
        pair => Pair.snd(Buffer.Content<F32>,PiecesModel.Content<Tile.Tile8>,pair),
        PiecesEach.seen(~Tile.Tile8,~Storage.Buffer<F32>,~Buffer.Content<F32>,~Buffer.contents(~F32),Tiled.tiled_piece(tiling,0,0,Buffer.pack(~F32,source))),
        (Buffer.contents(~F32,Buffer.pack(~F32,source)),piece_content),
        Entry.leaf_rule(tiling,0,0,source))),
    Equal.cong(Buffer.Content<Tile.Tile8>,PiecesModel.Content<Tile.Tile8>,content => PiecesModel.PieceContent{content,0,0},
      Entry.piece(tiling,Buffer.contents(~F32,Buffer.pack(~F32,source)),0,0),Buffer.Content{Entry.fresh_cells(0),0n},
      empty_piece_content(outer,after,lanes,groups,Buffer.contents(~F32,Buffer.pack(~F32,source)))))

def empty_pieces_contract(~Logical: Data,~original: Logical -> U32 -> F32,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +logical: Logical,+values: List<&2,U32>,+axis: Nat,+lanes: U32,+workers: U32,-pieces: Pieces.Pieces(~Tile.Tile8),
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+counted: Global.Counted<values>,
  +cells_empty: {ShapeLaws.elements(Tiled.cell_shape(values,axis,Grid.Innermost{})) == 0n : Nat},
  +range_zero: {Pieces.ranges_of_grid(Tensors.grid_of(Tiled.cell_shape(values,axis,Grid.Innermost{}))) == Pieces.Range{0,0} : Pieces.Ranges},
  +contents_eq: {PiecesModel.contents(~Tile.Tile8,pieces) == PiecesModel.PieceContent{Buffer.Content{Entry.fresh_cells(0),0n},0,0} : PiecesModel.Content<Tile.Tile8>},
  +pinned: {Grid.lanes_at(values,axis) == lanes : U32}) ->
  Public.Contract(~Logical,~original,logical,values,
    Tiled.Tiled{Tensors.Tensor{pieces,Tiled.cell_shape(values,axis,Grid.Innermost{}),Tiled.zero_tile(),Tensors.Usable{},workers},axis,lanes,Grid.Innermost{}}):
  +cells = Tiled.cell_shape(values,axis,Grid.Innermost{})
  +layout = Pieces.ranges_of_grid(Tensors.grid_of(cells))
  +cells_room = {%Equal.sym(Nat,ShapeLaws.elements(cells),0n,cells_empty) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    Order.zero_le(U32.to_nat(Machine.capacity(limits))) : {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +cells_counted = StorageCount.counted_within(limits,cells,cells_room,default_capacity)
  +fresh_ready = Buffer.content_ready(~Tile.Tile8,Buffer.fresh(~Tile.Tile8,Tensors.depth_for(0),Tiled.zero_tile(),0n),
    Buffer.fresh_ready(~Tile.Tile8,Tensors.depth_for(0),Tiled.zero_tile(),0n,{==},{==}))
  KernelGemm.Both{
    KernelGemm.Both{
      KernelGemm.Both{
        %Equal.sym(PiecesModel.Content<Tile.Tile8>,PiecesModel.contents(~Tile.Tile8,pieces),PiecesModel.PieceContent{Buffer.Content{Entry.fresh_cells(0),0n},0,0},contents_eq) :
          Content.WellFormed<Tile.Tile8,ShapeLaws.elements(cells),_>
        %Equal.sym(Nat,ShapeLaws.elements(cells),0n,cells_empty) : Content.WellFormed<Tile.Tile8,_,PiecesModel.PieceContent{Buffer.Content{Entry.fresh_cells(0),0n},0,0}>
        Content.WellFormed{Content.Fields{{==},{==}},fresh_ready,PiecesPlan.LeafCoverage{{==},{==}}},
        KernelGemm.Both{cells_counted,
          %Equal.sym(PiecesModel.Content<Tile.Tile8>,PiecesModel.contents(~Tile.Tile8,pieces),PiecesModel.PieceContent{Buffer.Content{Entry.fresh_cells(0),0n},0,0},contents_eq) :
            Global.Canonical(~Tile.Tile8,cells,_)
          Equal.sym(Pieces.Ranges,layout,Pieces.Range{0,0},range_zero)}},
      %Equal.sym(PiecesModel.Content<Tile.Tile8>,PiecesModel.contents(~Tile.Tile8,pieces),PiecesModel.PieceContent{Buffer.Content{Entry.fresh_cells(0),0n},0,0},contents_eq) :
        Content.Values(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,axis,Grid.Innermost{})},_)
      Unit{}},
    Public.Laid{inside,{==},
      %pinned : {Tiled.logical_shape(cells,axis,_,Grid.Innermost{}) == values : List<&2,U32>}
      Public.cells_restored(values,axis,Grid.Innermost{}),counted,pinned}}

def tiled_empty(~Logical: Data,~original: Logical -> U32 -> F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +logical: Logical,+values: List<&2,U32>,+workers: U32,+source: Buffer.Model<F32>,+counted: Global.Counted<values>,+empty: {ShapeLaws.elements(values) == 0n : Nat}) ->
  Public.Contract(~Logical,~original,logical,values,Tiled.tiled_shape(Tensors.Usable{},values,workers,Buffer.pack(~F32,source))):
  +axis = Tiled.lane_axis_for(values)
  +inside = Public.axis_for_within(values)
  +prefix = Public.lane_prefix(values,axis,1)
  +lanes = Grid.lanes_at(values,axis)
  +after_axes = Public.lane_after(values,axis)
  +after = Tiled.product(after_axes)
  +outer = (prefix * after : U32)
  +groups = Tiled.groups(lanes)
  +cells = Tiled.cell_shape(values,axis,Grid.Innermost{})
  +layout = Pieces.ranges_of_grid(Tensors.grid_of(cells))
  +cells_empty = empty_cells(values,axis,Grid.Innermost{},empty)
  +range_zero = empty_grid_ranges(cells,cells_empty)
  +contents_eq = empty_built_contents(outer,after,lanes,groups,layout,source,range_zero)
  %Equal.sym(Maybe<&2,Tiled.Around>,Tiled.around(values,axis,1),Some{Tiled.Around{prefix,lanes,after_axes}},Public.located(values,axis,1,inside)) :
    Public.Contract(~Logical,~original,logical,values,Tiled.tiled_around(_,Tensors.Usable{},values,axis,Grid.Innermost{},workers,Buffer.pack(~F32,source)))
  empty_pieces_contract(~Logical,~original,limits,default_capacity,logical,values,axis,lanes,workers,
    Pair.snd(Storage.Buffer<F32>,Pieces.Pieces(~Tile.Tile8),
      Pieces.build(~Storage.Buffer<Tile.Tile8>,~Storage.Buffer<F32>,~Tiled.Tiling,~Tiled.tiled_piece,layout,Tiled.Tiling{outer,after,lanes,groups,Grid.Innermost{}},
        Buffer.pack(~F32,source))),
    inside,counted,cells_empty,range_zero,contents_eq,{==})

# ---- Both cases ----

def tiled_cases(~Logical: Data,~original: Logical -> U32 -> F32,nonempty: Bool,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +logical: Logical,+values: List<&2,U32>,+workers: U32,+source: Buffer.Model<F32>,+counted: Global.Counted<values>,
  +facts: Contract.Facts(~F32,~Logical,~original,logical,values,Buffer.content(~F32,source)),
  +classified: {Nat.is_lt(0n,ShapeLaws.elements(values)) == nonempty : Bool}) ->
  Public.Contract(~Logical,~original,logical,values,Tiled.tiled_shape(Tensors.Usable{},values,workers,Buffer.pack(~F32,source))):
  match nonempty:
    case True{}: tiled_nonempty(~Logical,~original,limits,default_capacity,logical,values,workers,source,counted,facts,classified)
    case False{}:
      tiled_empty(~Logical,~original,limits,default_capacity,logical,values,workers,source,counted,
        Order.zero_below(ShapeLaws.elements(values),Order.not_less(0n,ShapeLaws.elements(values),classified)))

# The tiled form of row-major values that hold a reader's values in a counted
# shape holds the same contract.
def tiled(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+workers: U32,+source: Buffer.Model<F32>,
  +counted: Global.Counted<values>,+facts: Contract.Facts(~F32,~Logical,~original,logical,values,Buffer.content(~F32,source))) ->
  Public.Contract(~Logical,~original,logical,values,Tiled.tiled(Tensors.values(~F32,Buffer.pack(~F32,source),values,0.0,Tensors.Usable{},workers))):
  tiled_cases(~Logical,~original,Nat.is_lt(0n,ShapeLaws.elements(values)),Default.limits(),{==},logical,values,workers,source,counted,facts,{==})
