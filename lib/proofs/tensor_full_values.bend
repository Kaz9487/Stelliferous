# A full tensor holds its value at every index: a fresh buffer of the value
# in every piece of the grid of its shape. The tiled +0 tensor of a shape
# without a zero extent holds +0 at every index.
import Base
import ../kernel_tile.bend as Tile
import ../kernel_gemm.bend as KernelGemm
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ../storage_buffer.bend as Storage
import ../tensor.bend as Tensors
import ../tensor_pieces.bend as Pieces
import ../tensor_grid.bend as Grid
import ../tensor_shape.bend as Shapes
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_cells.bend as Cells
import ../tensor_tiled_map.bend as TiledMap
import ../traversal_partition.bend as Partition
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_pieces_content.bend as Content
import ./tensor_global_shape_count.bend as Global
import ./tensor_grid_ranges.bend as GridRanges
import ./tensor_placed.bend as Placed
import ./tensor_task_runs.bend as Runs
import ./tensor_move_readback.bend as Held
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_readback.bend as StorageReadback
import ./storage_allocation_refinement.bend as AllocationLaws
import ./storage_shape.bend as Shape
import ./tensor_allocation_depth.bend as Depth
import ./sequence_membership.bend as Members
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_storage_count.bend as StorageCount
import ./tensor_cell_room.bend as Rooms
import ./tensor_tiled_public.bend as Public
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_map.bend as TiledMaps
import ./tensor_pieces_map.bend as PiecesMap
import ./kernel_tile_refinement.bend as TileLaws
import ./nat_order.bend as Order
import ./u32_comparison.bend as Comparison
import ./machine_limit_bounds.bend as Limits

# Every element is the initial value.
def constant(~Element: Data,initial: Element,position: U32) -> Element:
  initial

# ---- A fresh piece ----

# What every cell of a fresh piece reads: the value, the first cell, the
# depth and the length of the piece.
type Fresh<-Element: Data> is Data:
  Fresh{fill: Element,first: U32,depth: Nat,length: Nat}

def fresh_length(~Element: Data,fresh: Fresh<Element>) -> Nat:
  Fresh{fill,first,depth,length} = fresh
  length

def FreshAt(~Element: Data,+fresh: Fresh<Element>,+index: Nat) -> Data:
  Fresh{+fill,+first,+depth,+length} = fresh
  Content.At(~Element,~Element,~constant(~Element),fill,first,Tree.fresh(Element,depth,fill),length,index)

def fresh_at(~Element: Data,+fresh: Fresh<Element>,+index: Nat,+inside: {Nat.is_lt(index,fresh_length(~Element,fresh)) == True{} : Bool}) -> FreshAt(~Element,fresh,index):
  Fresh{+fill,+first,+depth,+length} = fresh
  KernelGemm.Both{inside,StorageReadback.read_fresh(~Element,depth,fill,U32.from_nat(index))}

# A fresh buffer of count cells of the value holds the value in every cell,
# as many cells as its rows, ready.
def piece_holds(~Element: Data,+fill: Element,+first: U32,+count: U32,+within: {U32.is_le(count,1073741824) == True{} : Bool}) ->
  Placed.PieceHolds(~Element,~Element,~constant(~Element),fill,first,count,Buffer.content(~Element,Buffer.remodel(~Element,Tensors.buffer(~Element,fill,count)))):
  +depth = Tensors.depth_for(count)
  +length = U32.to_nat(count)
  +fresh = Tree.fresh(Element,depth,fill)
  +sized = Depth.bounded(count,within)
  +same = {Equal.trans(Buffer.Content<Element>,Buffer.content(~Element,Buffer.remodel(~Element,Tensors.buffer(~Element,fill,count))),
      Buffer.contents(~Element,Tensors.buffer(~Element,fill,count)),Buffer.Content{fresh,length},Buffer.remodel_content(~Element,Tensors.buffer(~Element,fill,count)),
      AllocationLaws.contents(~Element,depth,fill,length)) :
    {Buffer.content(~Element,Buffer.remodel(~Element,Tensors.buffer(~Element,fill,count))) == Buffer.Content{fresh,length} : Buffer.Content<Element>}}
  +room = {%Equal.sym(U32,Tree.capacity(Element,fresh),U32.shln(1,depth),Shape.capacity(Element,depth,fresh,Shape.fresh(Element,depth,fill))) :
      {Nat.is_le(length,U32.to_nat(_)) == True{} : Bool}
    Depth.allocation_room(count,sized) : {Nat.is_le(length,U32.to_nat(Tree.capacity(Element,fresh))) == True{} : Bool}}
  %Equal.sym(Buffer.Content<Element>,Buffer.content(~Element,Buffer.remodel(~Element,Tensors.buffer(~Element,fill,count))),Buffer.Content{fresh,length},same) :
    Placed.PieceHolds(~Element,~Element,~constant(~Element),fill,first,count,_)
  KernelGemm.Both{Members.range_all(~Fresh<Element>,~FreshAt(~Element),~fresh_length(~Element),~fresh_at(~Element),length,0n,Fresh{fill,first,depth,length},
      Order.reflexive(length)),
    KernelGemm.Both{{==},Buffer.ContentReady{depth,Shape.fresh(Element,depth,fill),Depth.allocation_depth(count,sized),room}}}

# ---- Filled pieces ----

# The pieces of a layout, each a fresh buffer of its rows filled with the
# value.
def filled_model(~Element: Data,layout: Pieces.Ranges,+fill: Element) -> PiecesModel.Model<Element>:
  match layout:
    case Pieces.Range{first,+count}:
      PiecesModel.Leaf{Buffer.remodel(~Element,Tensors.buffer(~Element,fill,count)),first,count}
    case Pieces.Split{left,right}:
      PiecesModel.Fork{filled_model(~Element,left,fill),filled_model(~Element,right,fill)}

def filled_layout(~Element: Data,layout: Pieces.Ranges,+fill: Element) -> {PiecesModel.layout(~Element,filled_model(~Element,layout,fill)) == layout : Pieces.Ranges}:
  match layout:
    case Pieces.Range{+first,+count}:
      {==}
    case Pieces.Split{+left,+right}:
      %Equal.sym(Pieces.Ranges,PiecesModel.layout(~Element,filled_model(~Element,left,fill)),left,filled_layout(~Element,left,fill)) :
        {Pieces.Split{_,PiecesModel.layout(~Element,filled_model(~Element,right,fill))} == Pieces.Split{left,right} : Pieces.Ranges}
      %Equal.sym(Pieces.Ranges,PiecesModel.layout(~Element,filled_model(~Element,right,fill)),right,filled_layout(~Element,right,fill)) :
        {Pieces.Split{left,_} == Pieces.Split{left,right} : Pieces.Ranges}
      {==}

# The pieces Tensors.filled builds are the filled pieces of its layout.
def filled_built(~Element: Data,layout: Pieces.Ranges,+fill: Element) ->
  {Pieces.build(~Storage.Buffer<Element>,~Unit,~Element,~Tensors.filled_piece(~Element),layout,fill,Unit{}) ==
    (Unit{},PiecesModel.pack(~Element,filled_model(~Element,layout,fill))) : Unit & Pieces.Pieces(~Element)}:
  match layout:
    case Pieces.Range{+first,+count}:
      %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,Buffer.remodel(~Element,Tensors.buffer(~Element,fill,count))),Tensors.buffer(~Element,fill,count),
          Buffer.remodeled(~Element,Tensors.buffer(~Element,fill,count))) :
        {Tensors.filled_piece(~Element,fill,first,count,Unit{}) == (Unit{},Partition.Leaf{_,first,count,Unit{}}) : Unit & Pieces.Pieces(~Element)}
      {==}
    case Pieces.Split{+left,+right}:
      %Equal.sym(Unit & Pieces.Pieces(~Element),Pieces.build(~Storage.Buffer<Element>,~Unit,~Element,~Tensors.filled_piece(~Element),left,fill,Unit{}),
          (Unit{},PiecesModel.pack(~Element,filled_model(~Element,left,fill))),filled_built(~Element,left,fill)) :
        {Storage.bind_pair(~Unit,~Pieces.Pieces(~Element),~(Unit & Pieces.Pieces(~Element)),_,state => left_tree =>
            Storage.bind_pair(~Unit,~Pieces.Pieces(~Element),~(Unit & Pieces.Pieces(~Element)),
              Pieces.build(~Storage.Buffer<Element>,~Unit,~Element,~Tensors.filled_piece(~Element),right,fill,state),state => right_tree => (state,Partition.Fork{left_tree,right_tree}))) ==
          (Unit{},PiecesModel.pack(~Element,filled_model(~Element,Pieces.Split{left,right},fill))) : Unit & Pieces.Pieces(~Element)}
      %Equal.sym(Unit & Pieces.Pieces(~Element),Pieces.build(~Storage.Buffer<Element>,~Unit,~Element,~Tensors.filled_piece(~Element),right,fill,Unit{}),
          (Unit{},PiecesModel.pack(~Element,filled_model(~Element,right,fill))),filled_built(~Element,right,fill)) :
        {Storage.bind_pair(~Unit,~Pieces.Pieces(~Element),~(Unit & Pieces.Pieces(~Element)),_,state => right_tree =>
            (state,Partition.Fork{PiecesModel.pack(~Element,filled_model(~Element,left,fill)),right_tree})) ==
          (Unit{},PiecesModel.pack(~Element,filled_model(~Element,Pieces.Split{left,right},fill))) : Unit & Pieces.Pieces(~Element)}
      {==}

# Every piece of a run of the layout holds the value.
def filled_holds(~Element: Data,+limits: Machine.Limits,layout: Pieces.Ranges,+fill: Element,+unit: Nat,+bound: Nat,+run: Runs.Run(layout,unit,bound),
  +room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  PiecesModel.ContentAll(~Element,~Placed.PieceHolds(~Element,~Element,~constant(~Element),fill),PiecesModel.content(~Element,filled_model(~Element,layout,fill))):
  match layout:
    case Pieces.Range{+first,+count}:
      +rows = U32.to_nat(count)
      +span = Nat.add(U32.to_nat(first),rows)
      +capacity = Machine.capacity(limits)
      +count_room = Order.transitive(rows,bound,U32.to_nat(capacity),
        Order.transitive(rows,span,bound,Order.right_le(U32.to_nat(first),rows),Runs.fits_inside(unit,U32.to_nat(first),span,bound,run)),room)
      +within = Order.word_le(count,capacity,1073741824,
        %Equal.sym(Bool,U32.is_le(count,capacity),Nat.is_le(rows,U32.to_nat(capacity)),Comparison.le(count,capacity)) : {_ == True{} : Bool} count_room,
        Limits.capacity_allocatable(limits))
      piece_holds(~Element,fill,first,count,within)
    case Pieces.Split{+left,+right}:
      Held.Both{+left_run,+rest} = run
      Held.Both{+right_run,+glue} = rest
      KernelGemm.Both{filled_holds(~Element,limits,left,fill,unit,bound,left_run,room),filled_holds(~Element,limits,right,fill,unit,bound,right_run,room)}

# The counted tensor of a nonempty shape full of a value holds the value at
# every index.
def full_counted(~Element: Data,+limits: Machine.Limits,+shape: List<&2,U32>,+value: Element,+counted: Global.Counted<shape>,
  +room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool}) ->
  Global.Contract(~Element,~Element,~constant(~Element),value,Tensors.Tensor{Tensors.filled(~Element,shape,value),shape,value,Tensors.Usable{},1}):
  +layout = Pieces.ranges_of_grid(Tensors.grid_of(shape))
  +model = filled_model(~Element,layout,value)
  +divides = {GridRanges.Multiple{GridRanges.last_extent(shape),Equal.sym(Nat,Nat.mul(GridRanges.last_extent(shape),1n),GridRanges.last_extent(shape),
      Order.times_one(GridRanges.last_extent(shape)))} : GridRanges.Multiple<GridRanges.last_extent(shape),1n>}
  +grid = GridRanges.grid_layout(limits,shape,Grid.piece_target(),room,nonempty,1n,divides)
  %Equal.sym(Unit & Pieces.Pieces(~Element),Pieces.build(~Storage.Buffer<Element>,~Unit,~Element,~Tensors.filled_piece(~Element),layout,value,Unit{}),
      (Unit{},PiecesModel.pack(~Element,model)),filled_built(~Element,layout,value)) :
    Global.Contract(~Element,~Element,~constant(~Element),value,Tensors.Tensor{Pair.snd(Unit,Pieces.Pieces(~Element),_),shape,value,Tensors.Usable{},1})
  GridRanges.grid_contract(~Element,~Element,~constant(~Element),value,limits,shape,value,1,PiecesModel.pack(~Element,model),room,nonempty,1n,divides,counted,
    Placed.PlacedModel{model,{==},filled_layout(~Element,layout,value),
      filled_holds(~Element,limits,layout,value,1n,ShapeLaws.elements(shape),GridRanges.layout_run(shape,Grid.piece_target(),1n,grid),room)})

# ---- The tiled +0 tensor ----

def pick_same(valid: Bool,+value: F32) -> {Bool.pick(F32,valid,value,value) == value : F32}:
  match valid:
    case True{}: {==}
    case False{}: {==}

# A zero tile and the lifted +0 of any geometry.
def ZeroRelated(tile: Tile.Tile8,lifted: Representation.Lifted<F32>) -> Data:
  KernelGemm.Both<{tile == Tiled.zero_tile() : Tile.Tile8},{Public.lifted_logical(~F32,lifted) == 0.0 : F32}>

def zero_lane(+geometry: Cells.Cells,+outer: Nat,+group: Nat,+lane: Nat) ->
  {0.0 == Representation.lane(~F32,~constant(~F32),0.0,geometry,outer,group,lane) : F32}:
  Equal.sym(F32,Bool.pick(F32,Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),Representation.lane_count(geometry)),0.0,0.0),0.0,
    pick_same(Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),Representation.lane_count(geometry)),0.0))

# Every cell of the lifted +0 is the zero tile.
def zero_agree(+tile: Tile.Tile8,+lifted: Representation.Lifted<F32>,+related: ZeroRelated(tile,lifted),+index: U32) ->
  {constant(~Tile.Tile8,tile,index) == Representation.lifted(~F32,~constant(~F32),lifted,index) : Tile.Tile8}:
  match lifted:
    case Representation.Lifted{+logical,+geometry}:
      KernelGemm.Both{+zero,+logical_zero} = related
      %Equal.sym(Tile.Tile8,tile,Tiled.zero_tile(),zero) : {_ == Representation.lifted(~F32,~constant(~F32),Representation.Lifted{logical,geometry},index) : Tile.Tile8}
      %Equal.sym(F32,logical,0.0,logical_zero) : {Tiled.zero_tile() == Representation.lifted(~F32,~constant(~F32),Representation.Lifted{_,geometry},index) : Tile.Tile8}
      +number = U32.to_nat(index)
      +outer = Representation.cell_outer(geometry,number)
      +group = Representation.cell_group(geometry,number)
      TileLaws.tiles_equal(Tiled.zero_tile(),Representation.cell(~F32,~constant(~F32),0.0,geometry,outer,group),
        zero_lane(geometry,outer,group,0n),zero_lane(geometry,outer,group,1n),zero_lane(geometry,outer,group,2n),zero_lane(geometry,outer,group,3n),
        zero_lane(geometry,outer,group,4n),zero_lane(geometry,outer,group,5n),zero_lane(geometry,outer,group,6n),zero_lane(geometry,outer,group,7n))

def zeros_cells(+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+count: U32,+counts: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},
  +nonempty: {Shapes.has_zero(shape) == False{} : Bool},+inside: {Nat.is_le(axis,List.length(&2,U32,shape)) == True{} : Bool},
  +cells_counted: Global.Counted<Tiled.cell_shape(shape,axis,Grid.Innermost{})>,
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(shape,axis,Grid.Innermost{})),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(shape,axis,Grid.Innermost{}))) == True{} : Bool}) ->
  Public.Contract(~F32,~constant(~F32),0.0,shape,Tiled.zeros_counted(Tensors.count_of(shape),Shapes.has_zero(shape),Tiled.around(shape,axis,1),shape,axis)):
  Global.Counted{+cell_count,+cell_accepted} = cells_counted
  +cells = Tiled.cell_shape(shape,axis,Grid.Innermost{})
  +lanes = Grid.lanes_at(shape,axis)
  +geometry = Public.geometry_of(shape,axis,Grid.Innermost{})
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(shape),Some{count},counts) :
    Public.Contract(~F32,~constant(~F32),0.0,shape,Tiled.zeros_counted(_,Shapes.has_zero(shape),Tiled.around(shape,axis,1),shape,axis))
  %Equal.sym(Bool,Shapes.has_zero(shape),False{},nonempty) :
    Public.Contract(~F32,~constant(~F32),0.0,shape,Tiled.zeros_counted(Some{count},_,Tiled.around(shape,axis,1),shape,axis))
  %Equal.sym(Maybe<&2,Tiled.Around>,Tiled.around(shape,axis,1),Some{Tiled.Around{Public.lane_prefix(shape,axis,1),lanes,Public.lane_after(shape,axis)}},
      Public.located(shape,axis,1,inside)) :
    Public.Contract(~F32,~constant(~F32),0.0,shape,Tiled.zeros_counted(Some{count},False{},_,shape,axis))
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(cells),Some{cell_count},cell_accepted) :
    Public.Contract(~F32,~constant(~F32),0.0,shape,Tiled.Tiled{Tensors.full_count(~Tile.Tile8,cells,_,Tiled.zero_tile()),axis,lanes,Grid.Innermost{}})
  KernelGemm.Both{
    Global.relabeled(~Tile.Tile8,~Tile.Tile8,~Representation.Lifted<F32>,~constant(~Tile.Tile8),~Representation.lifted(~F32,~constant(~F32)),~ZeroRelated,~zero_agree,
      Tiled.zero_tile(),Representation.Lifted{0.0,geometry},KernelGemm.Both{{==},{==}},
      Tensors.Tensor{Tensors.filled(~Tile.Tile8,cells,Tiled.zero_tile()),cells,Tiled.zero_tile(),Tensors.Usable{},1},
      full_counted(~Tile.Tile8,limits,cells,Tiled.zero_tile(),Global.Counted{cell_count,cell_accepted},cells_room,cells_positive)),
    Public.Laid{inside,{==},Public.cells_restored(shape,axis,Grid.Innermost{}),Global.Counted{count,counts},{==}}}

# The +0 tensor of a counted shape without a zero extent holds +0 at every
# index.
def zeros(+shape: List<&2,U32>,+count: U32,+counts: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},+nonempty: {Shapes.has_zero(shape) == False{} : Bool}) ->
  Public.Contract(~F32,~constant(~F32),0.0,shape,Tiled.zeros(shape)):
  +axis = Tiled.lane_axis_for(shape)
  +limits = Default.limits()
  +cells = Tiled.cell_shape(shape,axis,Grid.Innermost{})
  +counted = {Global.Counted{count,counts} : Global.Counted<shape>}
  +positive = CountCompletion.positive_elements(shape,nonempty)
  +room = StorageCount.counted_room(limits,shape,counted,{==})
  +cells_room = Rooms.cells_room(limits,shape,axis,Grid.Innermost{},positive,room)
  zeros_cells(limits,shape,axis,count,counts,nonempty,Public.axis_for_within(shape),StorageCount.counted_within(limits,cells,cells_room,{==}),cells_room,
    Rooms.cells_positive(limits,shape,axis,Grid.Innermost{},positive,room))

# The tensor of a value: the value at every index of a counted shape without
# a zero extent.
def full(+shape: List<&2,U32>,+value: F32,+count: U32,+counts: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},+nonempty: {Shapes.has_zero(shape) == False{} : Bool}) ->
  Public.Contract(~PiecesMap.Context<F32,F32>,~PiecesMap.value(~F32,~F32,~F32,~TiledMap.constant,~constant(~F32)),PiecesMap.Context{value,0.0},shape,TiledMap.full(shape,value)):
  TiledMaps.map(~F32,~F32,~TiledMap.constant,~TiledMap.lanes(~F32,~TiledMap.constant),~TiledMaps.lanes_law(~F32,~TiledMap.constant),~constant(~F32),value,1.0,0.0,shape,
    Tiled.zeros(shape),zeros(shape,count,counts,nonempty))
