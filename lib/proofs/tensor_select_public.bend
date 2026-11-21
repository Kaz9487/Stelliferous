# Select on three tiled tensors holds, at every flat index, the yes value
# where the mask is nonzero and the no value elsewhere: the three operands
# take the lane axis of yes with their groups innermost and select cell by
# cell, and every piece sets the lanes past the true extent to +0. Operands
# of other extents are first repeated to the common extents, each read
# through its view stretched to them.
import Base
import ../kernel_tile.bend as Tile
import ../kernel_gemm.bend as KernelGemm
import ../math_compare.bend as Compare
import ../tensor.bend as Tensors
import ../tensor_tasks.bend as Tasks
import ../tensor_pieces.bend as Pieces
import ../tensor_lanes.bend as Lanes
import ../tensor_operations.bend as Operations
import ../tensor_select.bend as Select
import ../tensor_tiled_cells.bend as Cells
import ../tensor_grid.bend as Grid
import ../tensor_shape.bend as Shapes
import ../tensor_view.bend as Views
import ../tensor_tiled.bend as Tiled
import ../tensor_f32.bend as F
import ../tensor_tiled_broadcast.bend as TiledBroadcast
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ../tensor_tiled_map.bend as TiledMap
import ../traversal_partition.bend as Partition
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./tensor_pieces_model.bend as Model
import ./tensor_pieces_content.bend as Content
import ./tensor_pieces_each.bend as Each
import ./tensor_pieces_walk.bend as Walk
import ./tensor_pieces_map.bend as PiecesMap
import ./tensor_pieces_combine.bend as PiecesCombine
import ./tensor_pieces_plan.bend as PiecesPlan
import ./tensor_select_kernel.bend as SelectKernel
import ./tensor_tiled_padding.bend as Padding
import ./tensor_tiled_padded_cells.bend as PaddedCells
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_public.bend as Public
import ./tensor_tiled_relane.bend as Relane
import ./tensor_tiled_map.bend as TiledMapLaws
import ./tensor_tiled_broadcast.bend as Broadcast
import ./tensor_view_gather.bend as ViewGather
import ./tensor_unit_broadcast.bend as UnitBroadcast
import ./tensor_stretched_digits.bend as Stretched
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_binary_values.bend as BinaryValues
import ./boolean_logic.bend as Boolean
import ./tensor_global_shape_count.bend as Global
import ./tensor_shape_refinement.bend as ShapeLaws
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges
import ./nat_order.bend as Order

# ---- The reader of a selection ----

# The readers of the mask, yes and no operands.
type Choice<-Mask: Data,-Yes: Data,-No: Data> is Data:
  Choice{mask: Mask,yes: Yes,no: No}

# The yes value where the mask is nonzero, the no value elsewhere.
def value(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,context: Choice<Mask,Yes,No>,index: U32) -> F32:
  Choice{mask,yes,no} = context
  +index = index
  Compare.select(mask_value(mask,index),yes_value(yes,index),no_value(no,index))

# The cells of the three operands in one geometry.
def Lifts(~Mask: Data,~Yes: Data,~No: Data) -> Data:
  Choice<Representation.Lifted<Mask>,Representation.Lifted<Yes>,Representation.Lifted<No>>

# The selection of the three cells at a cell number.
def cells_tile(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,context: Lifts(~Mask,~Yes,~No),word: U32) ->
  Tile.Tile8:
  Choice{mask,yes,no} = context
  +word = word
  Lanes.select_tile(Representation.lifted(~Mask,~mask_value,mask,word),Representation.lifted(~Yes,~yes_value,yes,word),Representation.lifted(~No,~no_value,no,word))

# ---- A selected cell ----

# A selected lane is the selected value below the count, and +0 like it past
# the count.
def tail_selected(~mask: F32,~yes: F32,~no: F32,valid: Bool) ->
  {Bool.pick(F32,valid,Compare.select(Bool.pick(F32,valid,mask,0.0),Bool.pick(F32,valid,yes,0.0),Bool.pick(F32,valid,no,0.0)),0.0) ==
    Bool.pick(F32,valid,Compare.select(mask,yes,no),0.0) : F32}:
  match valid:
    case True{}: {==}
    case False{}: {==}

def selected_lane(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,+mask: Mask,+yes: Yes,+no: No,
  +geometry: Cells.Cells,+outer: Nat,+group: Nat,+lane: Nat) ->
  {Bool.pick(F32,Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),Representation.lane_count(geometry)),
      Compare.select(Representation.lane(~Mask,~mask_value,mask,geometry,outer,group,lane),Representation.lane(~Yes,~yes_value,yes,geometry,outer,group,lane),
        Representation.lane(~No,~no_value,no,geometry,outer,group,lane)),0.0) ==
    Representation.lane(~Choice<Mask,Yes,No>,~value(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value),Choice{mask,yes,no},geometry,outer,group,lane) : F32}:
  +channel = Nat.add(lane,Nat.mul(group,8n))
  +index = U32.from_nat(Representation.logical_index(geometry,outer,channel))
  tail_selected(~mask_value(mask,index),~yes_value(yes,index),~no_value(no,index),Nat.is_lt(channel,Representation.lane_count(geometry)))

# The lanes of the selection of three cells agree with the cell of the
# selected reader.
def selected_valid(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,+mask: Mask,+yes: Yes,+no: No,
  +geometry: Cells.Cells,+outer: Nat,+group: Nat) ->
  PaddedCells.Valid<Representation.lane_count(geometry),group,
    Lanes.select_tile(Representation.cell(~Mask,~mask_value,mask,geometry,outer,group),Representation.cell(~Yes,~yes_value,yes,geometry,outer,group),
      Representation.cell(~No,~no_value,no,geometry,outer,group)),
    Representation.cell(~Choice<Mask,Yes,No>,~value(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value),Choice{mask,yes,no},geometry,outer,group)>:
  PaddedCells.Valid{selected_lane(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,geometry,outer,group,0n),
    selected_lane(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,geometry,outer,group,1n),
    selected_lane(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,geometry,outer,group,2n),
    selected_lane(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,geometry,outer,group,3n),
    selected_lane(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,geometry,outer,group,4n),
    selected_lane(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,geometry,outer,group,5n),
    selected_lane(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,geometry,outer,group,6n),
    selected_lane(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,geometry,outer,group,7n)}

# What every piece of a selection reads: the three readers and the layout of
# the cells.
type Selecting<-Mask: Data,-Yes: Data,-No: Data> is Data:
  Selecting{mask: Mask,yes: Yes,no: No,layout: PaddedCells.Layout}

def selecting_layout(~Mask: Data,~Yes: Data,~No: Data,+scope: Selecting<Mask,Yes,No>) -> PaddedCells.Layout:
  Selecting{+mask,+yes,+no,+layout} = scope
  layout

def selecting_padding(~Mask: Data,~Yes: Data,~No: Data,+scope: Selecting<Mask,Yes,No>) -> TiledMap.Padding:
  Selecting{+mask,+yes,+no,+layout} = scope
  PaddedCells.layout_padding(layout)

def selecting_total(~Mask: Data,~Yes: Data,~No: Data,+scope: Selecting<Mask,Yes,No>) -> Nat:
  Selecting{+mask,+yes,+no,+layout} = scope
  PaddedCells.layout_total(layout)

def selecting_mask(~Mask: Data,~Yes: Data,~No: Data,+scope: Selecting<Mask,Yes,No>) -> Representation.Lifted<Mask>:
  Selecting{+mask,+yes,+no,+layout} = scope
  Representation.Lifted{mask,PaddedCells.layout_geometry(layout)}

def selecting_yes(~Mask: Data,~Yes: Data,~No: Data,+scope: Selecting<Mask,Yes,No>) -> Representation.Lifted<Yes>:
  Selecting{+mask,+yes,+no,+layout} = scope
  Representation.Lifted{yes,PaddedCells.layout_geometry(layout)}

def selecting_no(~Mask: Data,~Yes: Data,~No: Data,+scope: Selecting<Mask,Yes,No>) -> Representation.Lifted<No>:
  Selecting{+mask,+yes,+no,+layout} = scope
  Representation.Lifted{no,PaddedCells.layout_geometry(layout)}

def selecting_before(~Mask: Data,~Yes: Data,~No: Data,+scope: Selecting<Mask,Yes,No>) -> Lifts(~Mask,~Yes,~No):
  Choice{selecting_mask(~Mask,~Yes,~No,scope),selecting_yes(~Mask,~Yes,~No,scope),selecting_no(~Mask,~Yes,~No,scope)}

def selecting_after(~Mask: Data,~Yes: Data,~No: Data,+scope: Selecting<Mask,Yes,No>) -> Representation.Lifted<Choice<Mask,Yes,No>>:
  Selecting{+mask,+yes,+no,+layout} = scope
  Representation.Lifted{Choice{mask,yes,no},PaddedCells.layout_geometry(layout)}

def selected_agree(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,+scope: Selecting<Mask,Yes,No>,+word: U32,
  +inside: {Nat.is_lt(U32.to_nat(word),PaddedCells.layout_total(selecting_layout(~Mask,~Yes,~No,scope))) == True{} : Bool}) ->
  PaddedCells.Valid<Representation.lane_count(PaddedCells.layout_geometry(selecting_layout(~Mask,~Yes,~No,scope))),
    Representation.cell_group(PaddedCells.layout_geometry(selecting_layout(~Mask,~Yes,~No,scope)),U32.to_nat(word)),
    cells_tile(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,selecting_before(~Mask,~Yes,~No,scope),word),
    Representation.lifted(~Choice<Mask,Yes,No>,~value(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value),selecting_after(~Mask,~Yes,~No,scope),word)>:
  Selecting{+mask,+yes,+no,+layout} = scope
  +geometry = PaddedCells.layout_geometry(layout)
  +number = U32.to_nat(word)
  selected_valid(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,geometry,Representation.cell_outer(geometry,number),Representation.cell_group(geometry,number))

# The padding after a selection keeps a piece of the selected tiles to the
# cells of the selected reader.
def selected_rule(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,+scope: Selecting<Mask,Yes,No>,
  +first: U32,+count: U32,+input: Buffer.Content<Tile.Tile8>,+ready: Buffer.ContentReady<Tile.Tile8,input>,
  +length_exact: {Buffer.content_length(~Tile.Tile8,input) == U32.to_nat(count) : Nat},
  +inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),selecting_total(~Mask,~Yes,~No,scope)) == True{} : Bool},
  +values: Content.piece_values(~Tile.Tile8,~Lifts(~Mask,~Yes,~No),~cells_tile(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value),selecting_before(~Mask,~Yes,~No,scope),first,count,input)) ->
  PiecesMap.Stepped(~Tile.Tile8,~Representation.Lifted<Choice<Mask,Yes,No>>,~Representation.lifted(~Choice<Mask,Yes,No>,~value(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value)),selecting_after(~Mask,~Yes,~No,scope),first,count,input,
    Padding.step(selecting_padding(~Mask,~Yes,~No,scope),first,count,input)):
  Selecting{+mask,+yes,+no,+layout} = scope
  PaddedCells.padded_piece(~Selecting<Mask,Yes,No>,~Lifts(~Mask,~Yes,~No),~cells_tile(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value),
    ~Representation.Lifted<Choice<Mask,Yes,No>>,~Representation.lifted(~Choice<Mask,Yes,No>,~value(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value)),
    ~selecting_layout(~Mask,~Yes,~No),~selecting_before(~Mask,~Yes,~No),~selecting_after(~Mask,~Yes,~No),~selected_agree(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value),
    Selecting{mask,yes,no,layout},first,count,input,ready,length_exact,inside,values)

# ---- The leaf of a selection ----

# The tiles the kernel leaves in the yes piece.
def kernel_tree(+count: U32,+yes: Buffer.Model<Tile.Tile8>,+mask: Buffer.Model<Tile.Tile8>,+no: Buffer.Model<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  SelectKernel.output(~Tile.Tile8,~Lanes.select_tile,U32.to_nat(count),Buffer.tree(~Tile.Tile8,mask),Buffer.tree(~Tile.Tile8,no),Buffer.tree(~Tile.Tile8,yes))

def kernel_content(+count: U32,+yes: Buffer.Model<Tile.Tile8>,+mask: Buffer.Model<Tile.Tile8>,+no: Buffer.Model<Tile.Tile8>) -> Buffer.Content<Tile.Tile8>:
  Buffer.Content{kernel_tree(count,yes,mask,no),Buffer.length(~Tile.Tile8,yes)}

def tiles_selected(+mask: Tile.Tile8,+yes: Tile.Tile8,+no: Tile.Tile8,+other_mask: Tile.Tile8,+other_yes: Tile.Tile8,+other_no: Tile.Tile8,
  +mask_equal: {mask == other_mask : Tile.Tile8},+yes_equal: {yes == other_yes : Tile.Tile8},+no_equal: {no == other_no : Tile.Tile8}) ->
  {Lanes.select_tile(mask,yes,no) == Lanes.select_tile(other_mask,other_yes,other_no) : Tile.Tile8}:
  %mask_equal : {Lanes.select_tile(mask,yes,no) == Lanes.select_tile(_,other_yes,other_no) : Tile.Tile8}
  %yes_equal : {Lanes.select_tile(mask,yes,no) == Lanes.select_tile(mask,_,other_no) : Tile.Tile8}
  %no_equal : {Lanes.select_tile(mask,yes,no) == Lanes.select_tile(mask,yes,_) : Tile.Tile8}
  {==}

# A selected cell stays at its index and selects the three cells there.
def selected_at(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,
  +mask: Representation.Lifted<Mask>,+yes: Representation.Lifted<Yes>,+no: Representation.Lifted<No>,+first: U32,+count: U32,
  +mask_model: Buffer.Model<Tile.Tile8>,+yes_model: Buffer.Model<Tile.Tile8>,+no_model: Buffer.Model<Tile.Tile8>,+sized: Buffer.Sized<Tile.Tile8,yes_model>,
  +counted: {U32.to_nat(count) == Buffer.length(~Tile.Tile8,yes_model) : Nat},+index: Nat,+below: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool},
  +mask_cell: Content.At(~Tile.Tile8,~Representation.Lifted<Mask>,~Representation.lifted(~Mask,~mask_value),mask,first,Buffer.tree(~Tile.Tile8,mask_model),
    Buffer.length(~Tile.Tile8,mask_model),index),
  +yes_cell: Content.At(~Tile.Tile8,~Representation.Lifted<Yes>,~Representation.lifted(~Yes,~yes_value),yes,first,Buffer.tree(~Tile.Tile8,yes_model),
    Buffer.length(~Tile.Tile8,yes_model),index),
  +no_cell: Content.At(~Tile.Tile8,~Representation.Lifted<No>,~Representation.lifted(~No,~no_value),no,first,Buffer.tree(~Tile.Tile8,no_model),
    Buffer.length(~Tile.Tile8,no_model),index)) ->
  Content.At(~Tile.Tile8,~Lifts(~Mask,~Yes,~No),~cells_tile(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value),Choice{mask,yes,no},first,
    kernel_tree(count,yes_model,mask_model,no_model),Buffer.length(~Tile.Tile8,yes_model),index):
  KernelGemm.Both{mask_inside,+mask_equal} = mask_cell
  KernelGemm.Both{+yes_inside,+yes_equal} = yes_cell
  KernelGemm.Both{no_inside,+no_equal} = no_cell
  +position = U32.from_nat(index)
  +pixel = U32.from_nat(Nat.add(U32.to_nat(first),index))
  KernelGemm.Both{yes_inside,
    Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,kernel_tree(count,yes_model,mask_model,no_model),position),
      Lanes.select_tile(Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,mask_model),position),Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,yes_model),position),
        Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,no_model),position)),
      Lanes.select_tile(Representation.lifted(~Mask,~mask_value,mask,pixel),Representation.lifted(~Yes,~yes_value,yes,pixel),Representation.lifted(~No,~no_value,no,pixel)),
      SelectKernel.value(~Tile.Tile8,~Lanes.select_tile,U32.to_nat(count),mask_model,yes_model,no_model,index,sized,below,counted),
      tiles_selected(Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,mask_model),position),Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,yes_model),position),
        Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,no_model),position),
        Representation.lifted(~Mask,~mask_value,mask,pixel),Representation.lifted(~Yes,~yes_value,yes,pixel),Representation.lifted(~No,~no_value,no,pixel),
        mask_equal,yes_equal,no_equal))}

# What the three pieces of one range supply to every cell.
type Chosen<-Mask: Data,-Yes: Data,-No: Data,-mask_value: Mask -> U32 -> F32,-yes_value: Yes -> U32 -> F32,-no_value: No -> U32 -> F32> is Data:
  Chosen{mask: Representation.Lifted<Mask>,yes: Representation.Lifted<Yes>,no: Representation.Lifted<No>,first: U32,count: U32,
    mask_model: Buffer.Model<Tile.Tile8>,yes_model: Buffer.Model<Tile.Tile8>,no_model: Buffer.Model<Tile.Tile8>,sized: Buffer.Sized<Tile.Tile8,yes_model>,
    counted: {U32.to_nat(count) == Buffer.length(~Tile.Tile8,yes_model) : Nat},
    mask_values: Content.Leaf(~Tile.Tile8,~Representation.Lifted<Mask>,~Representation.lifted(~Mask,~mask_value),mask,first,count,Buffer.tree(~Tile.Tile8,mask_model),
      Buffer.length(~Tile.Tile8,mask_model)),
    yes_values: Content.Leaf(~Tile.Tile8,~Representation.Lifted<Yes>,~Representation.lifted(~Yes,~yes_value),yes,first,count,Buffer.tree(~Tile.Tile8,yes_model),
      Buffer.length(~Tile.Tile8,yes_model)),
    no_values: Content.Leaf(~Tile.Tile8,~Representation.Lifted<No>,~Representation.lifted(~No,~no_value),no,first,count,Buffer.tree(~Tile.Tile8,no_model),
      Buffer.length(~Tile.Tile8,no_model))}

def chosen_count(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,
  scope: Chosen<Mask,Yes,No,mask_value,yes_value,no_value>) -> Nat:
  Chosen{mask,yes,no,first,count,mask_model,yes_model,no_model,sized,counted,mask_values,yes_values,no_values} = scope
  U32.to_nat(count)

def ChosenCell(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,
  +scope: Chosen<Mask,Yes,No,mask_value,yes_value,no_value>,+index: Nat) -> Data:
  Chosen{mask,yes,no,first,count,mask_model,+yes_model,no_model,sized,counted,mask_values,yes_values,no_values} = scope
  Content.At(~Tile.Tile8,~Lifts(~Mask,~Yes,~No),~cells_tile(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value),Choice{mask,yes,no},first,
    kernel_tree(count,yes_model,mask_model,no_model),Buffer.length(~Tile.Tile8,yes_model),index)

def chosen_cell(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,
  +scope: Chosen<Mask,Yes,No,mask_value,yes_value,no_value>,+index: Nat,
  +below: {Nat.is_lt(index,chosen_count(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,scope)) == True{} : Bool}) ->
  ChosenCell(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,scope,index):
  Chosen{+mask,+yes,+no,+first,+count,+mask_model,+yes_model,+no_model,+sized,+counted,+mask_values,+yes_values,+no_values} = scope
  selected_at(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,first,count,mask_model,yes_model,no_model,sized,counted,index,below,
    Content.leaf_cell(~Tile.Tile8,~Representation.Lifted<Mask>,~Representation.lifted(~Mask,~mask_value),mask,first,count,Buffer.tree(~Tile.Tile8,mask_model),
      Buffer.length(~Tile.Tile8,mask_model),index,mask_values,below),
    Content.leaf_cell(~Tile.Tile8,~Representation.Lifted<Yes>,~Representation.lifted(~Yes,~yes_value),yes,first,count,Buffer.tree(~Tile.Tile8,yes_model),
      Buffer.length(~Tile.Tile8,yes_model),index,yes_values,below),
    Content.leaf_cell(~Tile.Tile8,~Representation.Lifted<No>,~Representation.lifted(~No,~no_value),no,first,count,Buffer.tree(~Tile.Tile8,no_model),
      Buffer.length(~Tile.Tile8,no_model),index,no_values,below))

# The kernel on three pieces of one range holding the values of three readers
# holds the selection of the three.
def chosen_values(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,
  +scope: Chosen<Mask,Yes,No,mask_value,yes_value,no_value>) ->
  Members.All(Nat,{index => ChosenCell(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,scope,index) : Nat -> Data},
    Ranges.range(chosen_count(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,scope),0n)):
  Members.range_all(~Chosen<Mask,Yes,No,mask_value,yes_value,no_value>,~ChosenCell(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value),
    ~chosen_count(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value),~chosen_cell(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value),
    chosen_count(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,scope),0n,scope,Order.reflexive(chosen_count(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,scope)))

# The facts of the padded selection of one range of three pieces that hold
# the values of their readers.
def leaf_facts(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,+scope: Selecting<Mask,Yes,No>,
  +first: U32,+count: U32,+yes: Buffer.Model<Tile.Tile8>,+mask: Buffer.Model<Tile.Tile8>,+no: Buffer.Model<Tile.Tile8>,
  +yes_fields: Content.piece_fields(~Tile.Tile8,selecting_total(~Mask,~Yes,~No,scope),first,count,Buffer.content(~Tile.Tile8,yes)),
  +yes_ready: Buffer.ContentReady<Tile.Tile8,Buffer.content(~Tile.Tile8,yes)>,
  +yes_values: Content.piece_values(~Tile.Tile8,~Representation.Lifted<Yes>,~Representation.lifted(~Yes,~yes_value),selecting_yes(~Mask,~Yes,~No,scope),first,count,
    Buffer.content(~Tile.Tile8,yes)),
  +mask_values: Content.piece_values(~Tile.Tile8,~Representation.Lifted<Mask>,~Representation.lifted(~Mask,~mask_value),selecting_mask(~Mask,~Yes,~No,scope),first,count,
    Buffer.content(~Tile.Tile8,mask)),
  +no_values: Content.piece_values(~Tile.Tile8,~Representation.Lifted<No>,~Representation.lifted(~No,~no_value),selecting_no(~Mask,~Yes,~No,scope),first,count,
    Buffer.content(~Tile.Tile8,no))) ->
  PiecesMap.Facts(~Tile.Tile8,~Representation.Lifted<Choice<Mask,Yes,No>>,~Representation.lifted(~Choice<Mask,Yes,No>,~value(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value)),selecting_after(~Mask,~Yes,~No,scope),
    selecting_total(~Mask,~Yes,~No,scope),Model.PieceContent{Padding.step(selecting_padding(~Mask,~Yes,~No,scope),first,count,kernel_content(count,yes,mask,no)),first,count}):
  Buffer.Model{+yes_tree,+yes_length,+yes_certificate} = yes
  Buffer.Model{+mask_tree,+mask_length,+mask_certificate} = mask
  Buffer.Model{+no_tree,+no_length,+no_certificate} = no
  Content.Fields{+yes_exact,+yes_inside} = yes_fields
  +yes_model = {Buffer.Model{yes_tree,yes_length,yes_certificate} : Buffer.Model<Tile.Tile8>}
  +mask_model = {Buffer.Model{mask_tree,mask_length,mask_certificate} : Buffer.Model<Tile.Tile8>}
  +no_model = {Buffer.Model{no_tree,no_length,no_certificate} : Buffer.Model<Tile.Tile8>}
  +input = kernel_content(count,yes_model,mask_model,no_model)
  +sized = Buffer.ready_content(~Tile.Tile8,yes_model,yes_ready)
  +ready = SelectKernel.ready(~Tile.Tile8,~Lanes.select_tile,U32.to_nat(count),mask_model,yes_model,no_model,sized)
  +values = chosen_values(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,
    Chosen{selecting_mask(~Mask,~Yes,~No,scope),selecting_yes(~Mask,~Yes,~No,scope),selecting_no(~Mask,~Yes,~No,scope),first,count,mask_model,yes_model,no_model,sized,
      Equal.sym(Nat,yes_length,U32.to_nat(count),yes_exact),mask_values,yes_values,no_values})
  +made = selected_rule(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,scope,first,count,input,ready,yes_exact,yes_inside,values)
  PiecesMap.piece_facts(~Tile.Tile8,~Representation.Lifted<Choice<Mask,Yes,No>>,~Representation.lifted(~Choice<Mask,Yes,No>,~value(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value)),selecting_after(~Mask,~Yes,~No,scope),
    first,count,selecting_total(~Mask,~Yes,~No,scope),input,Padding.step(selecting_padding(~Mask,~Yes,~No,scope),first,count,input),yes_exact,yes_inside,made)

# ---- The pieces of three trees ----

# One piece of each of three trees of one shape.
type PieceTriple is Data:
  PieceTriple{yes: Buffer.Model<Tile.Tile8>,mask: Buffer.Model<Tile.Tile8>,no: Buffer.Model<Tile.Tile8>}

def triple_pack(part: PieceTriple) -> Select.Triple():
  PieceTriple{yes,mask,no} = part
  Pieces.Both{Buffer.pack(~Tile.Tile8,yes),Pieces.Both{Buffer.pack(~Tile.Tile8,mask),Buffer.pack(~Tile.Tile8,no)}}

def self_triples(model: Model.Model<Tile.Tile8>) -> Model.Shaped<PieceTriple>:
  match model:
    case Model.Leaf{+buffer,first,rows}: Model.Piece{PieceTriple{buffer,buffer,buffer},first,rows}
    case Model.Fork{left,right}: Model.Pieces{self_triples(left),self_triples(right)}

# The pieces of three trees at the rows of the yes tree; where the shapes
# differ, the yes tree stands for all three.
def triples(yes: Model.Model<Tile.Tile8>,mask: Model.Model<Tile.Tile8>,no: Model.Model<Tile.Tile8>) -> Model.Shaped<PieceTriple>:
  match yes mask no:
    case Model.Leaf{yes,first,rows} Model.Leaf{mask,mask_first,mask_rows} Model.Leaf{no,no_first,no_rows}: Model.Piece{PieceTriple{yes,mask,no},first,rows}
    case Model.Fork{yes_left,yes_right} Model.Fork{mask_left,mask_right} Model.Fork{no_left,no_right}:
      Model.Pieces{triples(yes_left,mask_left,no_left),triples(yes_right,mask_right,no_right)}
    case yes mask no: self_triples(yes)

def triple_tree(+yes: Model.Model<Tile.Tile8>,+mask: Model.Model<Tile.Tile8>,+no: Model.Model<Tile.Tile8>) -> Partition.Tree<&1,Select.Triple(),Unit>:
  Model.shaped_pack(~PieceTriple,~Select.Triple(),~triple_pack,triples(yes,mask,no))

def Zipped(+yes: Model.Model<Tile.Tile8>,+mask: Model.Model<Tile.Tile8>,+no: Model.Model<Tile.Tile8>) -> Data:
  {Pieces.zip(~Select.Buffer(),~Pieces.Paired<Select.Buffer(),Select.Buffer()>,Model.pack(~Tile.Tile8,yes),
      Pieces.zip(~Select.Buffer(),~Select.Buffer(),Model.pack(~Tile.Tile8,mask),Model.pack(~Tile.Tile8,no))) ==
    Model.shaped_pack(~PieceTriple,~Select.Triple(),~triple_pack,triples(yes,mask,no)) : Partition.Tree<&1,Select.Triple(),Unit>}

# Three trees of one shape zip into the triples of their pieces.
def triple_zip(+yes: Model.Model<Tile.Tile8>,+mask: Model.Model<Tile.Tile8>,+no: Model.Model<Tile.Tile8>,+mask_common: Each.Common(~Tile.Tile8,yes,mask),
  +no_common: Each.Common(~Tile.Tile8,yes,no)) -> Zipped(yes,mask,no):
  match yes mask no:
    case Model.Leaf{+yes,+first,+rows} Model.Leaf{+mask,mask_first,mask_rows} Model.Leaf{+no,no_first,no_rows}: {==}
    case Model.Fork{+yes_left,+yes_right} Model.Fork{+mask_left,+mask_right} Model.Fork{+no_left,+no_right}:
      KernelGemm.Both{+mask_left_common,+mask_right_common} = mask_common
      KernelGemm.Both{+no_left_common,+no_right_common} = no_common
      %Equal.sym(Partition.Tree<&1,Select.Triple(),Unit>,
          Pieces.zip(~Select.Buffer(),~Pieces.Paired<Select.Buffer(),Select.Buffer()>,Model.pack(~Tile.Tile8,yes_left),
            Pieces.zip(~Select.Buffer(),~Select.Buffer(),Model.pack(~Tile.Tile8,mask_left),Model.pack(~Tile.Tile8,no_left))),triple_tree(yes_left,mask_left,no_left),
          triple_zip(yes_left,mask_left,no_left,mask_left_common,no_left_common)) :
        {Partition.Fork{_,Pieces.zip(~Select.Buffer(),~Pieces.Paired<Select.Buffer(),Select.Buffer()>,Model.pack(~Tile.Tile8,yes_right),
            Pieces.zip(~Select.Buffer(),~Select.Buffer(),Model.pack(~Tile.Tile8,mask_right),Model.pack(~Tile.Tile8,no_right)))} ==
          Partition.Fork{triple_tree(yes_left,mask_left,no_left),triple_tree(yes_right,mask_right,no_right)} : Partition.Tree<&1,Select.Triple(),Unit>}
      %Equal.sym(Partition.Tree<&1,Select.Triple(),Unit>,
          Pieces.zip(~Select.Buffer(),~Pieces.Paired<Select.Buffer(),Select.Buffer()>,Model.pack(~Tile.Tile8,yes_right),
            Pieces.zip(~Select.Buffer(),~Select.Buffer(),Model.pack(~Tile.Tile8,mask_right),Model.pack(~Tile.Tile8,no_right))),triple_tree(yes_right,mask_right,no_right),
          triple_zip(yes_right,mask_right,no_right,mask_right_common,no_right_common)) :
        {Partition.Fork{triple_tree(yes_left,mask_left,no_left),_} == Partition.Fork{triple_tree(yes_left,mask_left,no_left),triple_tree(yes_right,mask_right,no_right)} : Partition.Tree<&1,Select.Triple(),Unit>}
      {==}
    case Model.Leaf{+yes,+first,+rows} Model.Leaf{+mask,+mask_first,+mask_rows} Model.Fork{+no_left,+no_right}:
      Empty.absurd(Zipped(Model.Leaf{yes,first,rows},Model.Leaf{mask,mask_first,mask_rows},Model.Fork{no_left,no_right}),no_common)
    case Model.Leaf{+yes,+first,+rows} Model.Fork{+mask_left,+mask_right} no:
      Empty.absurd(Zipped(Model.Leaf{yes,first,rows},Model.Fork{mask_left,mask_right},no),mask_common)
    case Model.Fork{+yes_left,+yes_right} Model.Leaf{+mask,+mask_first,+mask_rows} no:
      Empty.absurd(Zipped(Model.Fork{yes_left,yes_right},Model.Leaf{mask,mask_first,mask_rows},no),mask_common)
    case Model.Fork{+yes_left,+yes_right} Model.Fork{+mask_left,+mask_right} Model.Leaf{+no,+no_first,+no_rows}:
      Empty.absurd(Zipped(Model.Fork{yes_left,yes_right},Model.Fork{mask_left,mask_right},Model.Leaf{no,no_first,no_rows}),no_common)

# What the padded selection of the pieces of three trees holds.
def piece_content(+padding: TiledMap.Padding,+first: U32,+rows: U32,+part: PieceTriple) -> Model.Content<Tile.Tile8>:
  PieceTriple{+yes,+mask,+no} = part
  Model.PieceContent{Padding.step(padding,first,rows,kernel_content(rows,yes,mask,no)),first,rows}

def shaped_content(+padding: TiledMap.Padding,tree: Model.Shaped<PieceTriple>) -> Model.Content<Tile.Tile8>:
  match tree:
    case Model.Piece{+part,+first,+rows}: piece_content(padding,first,rows,part)
    case Model.Pieces{left,right}: Model.Joined{shaped_content(padding,left),shaped_content(padding,right)}

# The buffer the kernel leaves.
def kernel_buffer(+count: U32,+yes: Buffer.Model<Tile.Tile8>,+mask: Buffer.Model<Tile.Tile8>,+no: Buffer.Model<Tile.Tile8>) -> Select.Buffer():
  Operations.choose_kernel(~Tile.Tile8,~Lanes.select_tile,U32.to_nat(count),Buffer.pack(~Tile.Tile8,mask),Buffer.pack(~Tile.Tile8,yes),Buffer.pack(~Tile.Tile8,no))

def piece_contents(+padding: TiledMap.Padding,+first: U32,+rows: U32,+part: PieceTriple) ->
  {Model.contents(~Tile.Tile8,Select.selected_piece(padding,first,rows,triple_pack(part),Unit{})) == piece_content(padding,first,rows,part) : Model.Content<Tile.Tile8>}:
  PieceTriple{+yes,+mask,+no} = part
  Equal.cong(Buffer.Content<Tile.Tile8>,Model.Content<Tile.Tile8>,content => Model.PieceContent{content,first,rows},
    Buffer.contents(~Tile.Tile8,TiledMap.padded(padding,first,rows,kernel_buffer(rows,yes,mask,no))),Padding.step(padding,first,rows,kernel_content(rows,yes,mask,no)),
    Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,TiledMap.padded(padding,first,rows,kernel_buffer(rows,yes,mask,no))),
      Padding.step(padding,first,rows,Buffer.contents(~Tile.Tile8,kernel_buffer(rows,yes,mask,no))),Padding.step(padding,first,rows,kernel_content(rows,yes,mask,no)),
      Padding.rule(padding,first,rows,kernel_buffer(rows,yes,mask,no)),
      Equal.cong(Buffer.Content<Tile.Tile8>,Buffer.Content<Tile.Tile8>,content => Padding.step(padding,first,rows,content),Buffer.contents(~Tile.Tile8,kernel_buffer(rows,yes,mask,no)),
        kernel_content(rows,yes,mask,no),SelectKernel.kernel(~Tile.Tile8,~Lanes.select_tile,U32.to_nat(rows),mask,yes,no))))

# The leaf of the selection on every triple of pieces holds the padded
# selections.
def leaf_contents(+padding: TiledMap.Padding,+tree: Model.Shaped<PieceTriple>) ->
  {Model.contents(~Tile.Tile8,Each.shaped_leaf_map(~PieceTriple,~Select.Triple(),~triple_pack,~Select.Buffer(),~TiledMap.Padding,~Select.selected_piece,padding,tree)) ==
    shaped_content(padding,tree) : Model.Content<Tile.Tile8>}:
  match tree:
    case Model.Piece{+part,+first,+rows}: piece_contents(padding,first,rows,part)
    case Model.Pieces{+left,+right}:
      +left_made = Model.contents(~Tile.Tile8,Each.shaped_leaf_map(~PieceTriple,~Select.Triple(),~triple_pack,~Select.Buffer(),~TiledMap.Padding,~Select.selected_piece,padding,left))
      +right_made = Model.contents(~Tile.Tile8,Each.shaped_leaf_map(~PieceTriple,~Select.Triple(),~triple_pack,~Select.Buffer(),~TiledMap.Padding,~Select.selected_piece,padding,right))
      %Equal.sym(Model.Content<Tile.Tile8>,left_made,shaped_content(padding,left),leaf_contents(padding,left)) :
        {Model.Joined{_,right_made} == Model.Joined{shaped_content(padding,left),shaped_content(padding,right)} : Model.Content<Tile.Tile8>}
      %Equal.sym(Model.Content<Tile.Tile8>,right_made,shaped_content(padding,right),leaf_contents(padding,right)) :
        {Model.Joined{shaped_content(padding,left),_} == Model.Joined{shaped_content(padding,left),shaped_content(padding,right)} : Model.Content<Tile.Tile8>}
      {==}

def OutputLayout(+padding: TiledMap.Padding,+yes: Model.Model<Tile.Tile8>,+mask: Model.Model<Tile.Tile8>,+no: Model.Model<Tile.Tile8>) -> Data:
  {Model.content_layout(~Tile.Tile8,shaped_content(padding,triples(yes,mask,no))) == Model.layout(~Tile.Tile8,yes) : Pieces.Ranges}

# The selection keeps the layout of the yes tree.
def output_layout(+padding: TiledMap.Padding,+yes: Model.Model<Tile.Tile8>,+mask: Model.Model<Tile.Tile8>,+no: Model.Model<Tile.Tile8>,
  +mask_common: Each.Common(~Tile.Tile8,yes,mask),+no_common: Each.Common(~Tile.Tile8,yes,no)) -> OutputLayout(padding,yes,mask,no):
  match yes mask no:
    case Model.Leaf{+yes,+first,+rows} Model.Leaf{+mask,mask_first,mask_rows} Model.Leaf{+no,no_first,no_rows}: {==}
    case Model.Fork{+yes_left,+yes_right} Model.Fork{+mask_left,+mask_right} Model.Fork{+no_left,+no_right}:
      KernelGemm.Both{+mask_left_common,+mask_right_common} = mask_common
      KernelGemm.Both{+no_left_common,+no_right_common} = no_common
      %Equal.sym(Pieces.Ranges,Model.content_layout(~Tile.Tile8,shaped_content(padding,triples(yes_left,mask_left,no_left))),Model.layout(~Tile.Tile8,yes_left),
          output_layout(padding,yes_left,mask_left,no_left,mask_left_common,no_left_common)) :
        {Pieces.Split{_,Model.content_layout(~Tile.Tile8,shaped_content(padding,triples(yes_right,mask_right,no_right)))} ==
          Pieces.Split{Model.layout(~Tile.Tile8,yes_left),Model.layout(~Tile.Tile8,yes_right)} : Pieces.Ranges}
      %Equal.sym(Pieces.Ranges,Model.content_layout(~Tile.Tile8,shaped_content(padding,triples(yes_right,mask_right,no_right))),Model.layout(~Tile.Tile8,yes_right),
          output_layout(padding,yes_right,mask_right,no_right,mask_right_common,no_right_common)) :
        {Pieces.Split{Model.layout(~Tile.Tile8,yes_left),_} == Pieces.Split{Model.layout(~Tile.Tile8,yes_left),Model.layout(~Tile.Tile8,yes_right)} : Pieces.Ranges}
      {==}
    case Model.Leaf{+yes,+first,+rows} Model.Leaf{+mask,+mask_first,+mask_rows} Model.Fork{+no_left,+no_right}:
      Empty.absurd(OutputLayout(padding,Model.Leaf{yes,first,rows},Model.Leaf{mask,mask_first,mask_rows},Model.Fork{no_left,no_right}),no_common)
    case Model.Leaf{+yes,+first,+rows} Model.Fork{+mask_left,+mask_right} no:
      Empty.absurd(OutputLayout(padding,Model.Leaf{yes,first,rows},Model.Fork{mask_left,mask_right},no),mask_common)
    case Model.Fork{+yes_left,+yes_right} Model.Leaf{+mask,+mask_first,+mask_rows} no:
      Empty.absurd(OutputLayout(padding,Model.Fork{yes_left,yes_right},Model.Leaf{mask,mask_first,mask_rows},no),mask_common)
    case Model.Fork{+yes_left,+yes_right} Model.Fork{+mask_left,+mask_right} Model.Leaf{+no,+no_first,+no_rows}:
      Empty.absurd(OutputLayout(padding,Model.Fork{yes_left,yes_right},Model.Fork{mask_left,mask_right},Model.Leaf{no,no_first,no_rows}),no_common)

# ---- The facts of a selection ----

# What a triple of pieces holds: the yes piece fits, owns its storage and
# holds its values, and the mask and no pieces hold theirs.
def TripleFacts(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,+scope: Selecting<Mask,Yes,No>,
  +part: PieceTriple,+first: U32,+rows: U32) -> Data:
  PieceTriple{+yes,+mask,+no} = part
  KernelGemm.Both<Content.piece_fields(~Tile.Tile8,selecting_total(~Mask,~Yes,~No,scope),first,rows,Buffer.content(~Tile.Tile8,yes)),
    KernelGemm.Both<Buffer.ContentReady<Tile.Tile8,Buffer.content(~Tile.Tile8,yes)>,
      KernelGemm.Both<Content.piece_values(~Tile.Tile8,~Representation.Lifted<Yes>,~Representation.lifted(~Yes,~yes_value),selecting_yes(~Mask,~Yes,~No,scope),first,rows,
          Buffer.content(~Tile.Tile8,yes)),
        KernelGemm.Both<Content.piece_values(~Tile.Tile8,~Representation.Lifted<Mask>,~Representation.lifted(~Mask,~mask_value),selecting_mask(~Mask,~Yes,~No,scope),first,rows,
            Buffer.content(~Tile.Tile8,mask)),
          Content.piece_values(~Tile.Tile8,~Representation.Lifted<No>,~Representation.lifted(~No,~no_value),selecting_no(~Mask,~Yes,~No,scope),first,rows,
            Buffer.content(~Tile.Tile8,no))>>>>

def part_facts(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,+scope: Selecting<Mask,Yes,No>,
  +part: PieceTriple,+first: U32,+rows: U32,+facts: TripleFacts(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,scope,part,first,rows)) ->
  PiecesMap.Facts(~Tile.Tile8,~Representation.Lifted<Choice<Mask,Yes,No>>,~Representation.lifted(~Choice<Mask,Yes,No>,~value(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value)),selecting_after(~Mask,~Yes,~No,scope),
    selecting_total(~Mask,~Yes,~No,scope),piece_content(selecting_padding(~Mask,~Yes,~No,scope),first,rows,part)):
  PieceTriple{+yes,+mask,+no} = part
  KernelGemm.Both{+yes_fields,KernelGemm.Both{+yes_ready,KernelGemm.Both{+yes_values,KernelGemm.Both{+mask_values,+no_values}}}} = facts
  leaf_facts(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,scope,first,rows,yes,mask,no,yes_fields,yes_ready,yes_values,mask_values,no_values)

# Triples of pieces that hold their facts select into pieces of the selected
# reader.
def shaped_facts(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,+scope: Selecting<Mask,Yes,No>,
  +tree: Model.Shaped<PieceTriple>,
  +facts: Model.ShapedAll(~PieceTriple,~Selecting<Mask,Yes,No>,~TripleFacts(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value),scope,tree)) ->
  PiecesMap.Facts(~Tile.Tile8,~Representation.Lifted<Choice<Mask,Yes,No>>,~Representation.lifted(~Choice<Mask,Yes,No>,~value(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value)),selecting_after(~Mask,~Yes,~No,scope),
    selecting_total(~Mask,~Yes,~No,scope),shaped_content(selecting_padding(~Mask,~Yes,~No,scope),tree)):
  match tree:
    case Model.Piece{+part,+first,+rows}: part_facts(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,scope,part,first,rows,facts)
    case Model.Pieces{+left,+right}:
      KernelGemm.Both{+left_facts,+right_facts} = facts
      +padding = selecting_padding(~Mask,~Yes,~No,scope)
      PiecesMap.joined_facts(~Tile.Tile8,~Representation.Lifted<Choice<Mask,Yes,No>>,~Representation.lifted(~Choice<Mask,Yes,No>,~value(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value)),
        selecting_after(~Mask,~Yes,~No,scope),selecting_total(~Mask,~Yes,~No,scope),shaped_content(padding,left),shaped_content(padding,right),
        shaped_facts(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,scope,left,left_facts),shaped_facts(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,scope,right,right_facts))

def AllFacts(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,+scope: Selecting<Mask,Yes,No>,
  +yes: Model.Model<Tile.Tile8>,+mask: Model.Model<Tile.Tile8>,+no: Model.Model<Tile.Tile8>) -> Data:
  Model.ShapedAll(~PieceTriple,~Selecting<Mask,Yes,No>,~TripleFacts(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value),scope,triples(yes,mask,no))

# What holds of the pieces of each of three trees of one shape holds of their
# triples.
def triple_all(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,+scope: Selecting<Mask,Yes,No>,
  +yes: Model.Model<Tile.Tile8>,+mask: Model.Model<Tile.Tile8>,+no: Model.Model<Tile.Tile8>,+mask_common: Each.Common(~Tile.Tile8,yes,mask),
  +no_common: Each.Common(~Tile.Tile8,yes,no),+yes_fields: Content.Contents(~Tile.Tile8,selecting_total(~Mask,~Yes,~No,scope),Model.content(~Tile.Tile8,yes)),
  +yes_ready: Model.ContentReady(~Tile.Tile8,Model.content(~Tile.Tile8,yes)),
  +yes_values: Content.Values(~Tile.Tile8,~Representation.Lifted<Yes>,~Representation.lifted(~Yes,~yes_value),selecting_yes(~Mask,~Yes,~No,scope),Model.content(~Tile.Tile8,yes)),
  +mask_values: Content.Values(~Tile.Tile8,~Representation.Lifted<Mask>,~Representation.lifted(~Mask,~mask_value),selecting_mask(~Mask,~Yes,~No,scope),
    Model.content(~Tile.Tile8,mask)),
  +no_values: Content.Values(~Tile.Tile8,~Representation.Lifted<No>,~Representation.lifted(~No,~no_value),selecting_no(~Mask,~Yes,~No,scope),Model.content(~Tile.Tile8,no))) ->
  AllFacts(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,scope,yes,mask,no):
  match yes mask no:
    case Model.Leaf{+yes,+first,+rows} Model.Leaf{+mask,+mask_first,+mask_rows} Model.Leaf{+no,+no_first,+no_rows}:
      KernelGemm.Both{+same_first,+same_rows} = mask_common
      KernelGemm.Both{+no_same_first,+no_same_rows} = no_common
      KernelGemm.Both{yes_fields,KernelGemm.Both{yes_ready,KernelGemm.Both{yes_values,KernelGemm.Both{
        %Equal.sym(U32,first,mask_first,same_first) :
          Content.piece_values(~Tile.Tile8,~Representation.Lifted<Mask>,~Representation.lifted(~Mask,~mask_value),selecting_mask(~Mask,~Yes,~No,scope),_,rows,
            Buffer.content(~Tile.Tile8,mask))
        %Equal.sym(U32,rows,mask_rows,same_rows) :
          Content.piece_values(~Tile.Tile8,~Representation.Lifted<Mask>,~Representation.lifted(~Mask,~mask_value),selecting_mask(~Mask,~Yes,~No,scope),mask_first,_,
            Buffer.content(~Tile.Tile8,mask))
        mask_values,
        %Equal.sym(U32,first,no_first,no_same_first) :
          Content.piece_values(~Tile.Tile8,~Representation.Lifted<No>,~Representation.lifted(~No,~no_value),selecting_no(~Mask,~Yes,~No,scope),_,rows,
            Buffer.content(~Tile.Tile8,no))
        %Equal.sym(U32,rows,no_rows,no_same_rows) :
          Content.piece_values(~Tile.Tile8,~Representation.Lifted<No>,~Representation.lifted(~No,~no_value),selecting_no(~Mask,~Yes,~No,scope),no_first,_,
            Buffer.content(~Tile.Tile8,no))
        no_values}}}}
    case Model.Fork{+yes_left,+yes_right} Model.Fork{+mask_left,+mask_right} Model.Fork{+no_left,+no_right}:
      KernelGemm.Both{+mask_left_common,+mask_right_common} = mask_common
      KernelGemm.Both{+no_left_common,+no_right_common} = no_common
      KernelGemm.Both{+yes_fields_left,+yes_fields_right} = yes_fields
      KernelGemm.Both{+yes_ready_left,+yes_ready_right} = yes_ready
      KernelGemm.Both{+yes_values_left,+yes_values_right} = yes_values
      KernelGemm.Both{+mask_values_left,+mask_values_right} = mask_values
      KernelGemm.Both{+no_values_left,+no_values_right} = no_values
      KernelGemm.Both{
        triple_all(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,scope,yes_left,mask_left,no_left,mask_left_common,no_left_common,yes_fields_left,yes_ready_left,
          yes_values_left,mask_values_left,no_values_left),
        triple_all(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,scope,yes_right,mask_right,no_right,mask_right_common,no_right_common,yes_fields_right,yes_ready_right,
          yes_values_right,mask_values_right,no_values_right)}
    case Model.Leaf{+yes,+first,+rows} Model.Leaf{+mask,+mask_first,+mask_rows} Model.Fork{+no_left,+no_right}:
      Empty.absurd(AllFacts(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,scope,Model.Leaf{yes,first,rows},Model.Leaf{mask,mask_first,mask_rows},Model.Fork{no_left,no_right}),
        no_common)
    case Model.Leaf{+yes,+first,+rows} Model.Fork{+mask_left,+mask_right} no:
      Empty.absurd(AllFacts(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,scope,Model.Leaf{yes,first,rows},Model.Fork{mask_left,mask_right},no),mask_common)
    case Model.Fork{+yes_left,+yes_right} Model.Leaf{+mask,+mask_first,+mask_rows} no:
      Empty.absurd(AllFacts(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,scope,Model.Fork{yes_left,yes_right},Model.Leaf{mask,mask_first,mask_rows},no),mask_common)
    case Model.Fork{+yes_left,+yes_right} Model.Fork{+mask_left,+mask_right} Model.Leaf{+no,+no_first,+no_rows}:
      Empty.absurd(AllFacts(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,scope,Model.Fork{yes_left,yes_right},Model.Fork{mask_left,mask_right},Model.Leaf{no,no_first,no_rows}),
        no_common)

# ---- The selection of three trees ----

# The grouped leaf on the triples of the pieces of three trees holds the
# padded selections, whichever halves run in parallel.
def grouped_contents(+padding: TiledMap.Padding,+tasks: Pieces.Ranges,+parallel: Bool,+tree: Model.Shaped<PieceTriple>) ->
  {Model.contents(~Tile.Tile8,Pieces.grouped(~Select.Triple(),~Select.Buffer(),~TiledMap.Padding,~Select.selected_piece,tasks,parallel,padding,
      Model.shaped_pack(~PieceTriple,~Select.Triple(),~triple_pack,tree))) == shaped_content(padding,tree) : Model.Content<Tile.Tile8>}:
  %Equal.sym(Partition.Tree<&1,Select.Buffer(),Unit>,
      Pieces.grouped(~Select.Triple(),~Select.Buffer(),~TiledMap.Padding,~Select.selected_piece,tasks,parallel,padding,Model.shaped_pack(~PieceTriple,~Select.Triple(),~triple_pack,tree)),
      Pieces.group_run(~Select.Triple(),~Select.Buffer(),~TiledMap.Padding,~Select.selected_piece,padding,Model.shaped_pack(~PieceTriple,~Select.Triple(),~triple_pack,tree)),
      Walk.grouped(~PieceTriple,~Select.Triple(),~triple_pack,~Select.Buffer(),~TiledMap.Padding,~Select.selected_piece,tasks,parallel,padding,tree)) :
    {Model.contents(~Tile.Tile8,_) == shaped_content(padding,tree) : Model.Content<Tile.Tile8>}
  %Equal.sym(Partition.Tree<&1,Select.Buffer(),Unit>,
      Pieces.group_run(~Select.Triple(),~Select.Buffer(),~TiledMap.Padding,~Select.selected_piece,padding,Model.shaped_pack(~PieceTriple,~Select.Triple(),~triple_pack,tree)),
      Each.shaped_leaf_map(~PieceTriple,~Select.Triple(),~triple_pack,~Select.Buffer(),~TiledMap.Padding,~Select.selected_piece,padding,tree),
      Each.shaped_group_serial(~PieceTriple,~Select.Triple(),~triple_pack,~Select.Buffer(),~TiledMap.Padding,~Select.selected_piece,padding,tree)) :
    {Model.contents(~Tile.Tile8,_) == shaped_content(padding,tree) : Model.Content<Tile.Tile8>}
  leaf_contents(padding,tree)

# Three tensors of one shape holding the values of three readers select on
# the tasks of any decision into a tensor of the selected reader.
def decided(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,+scope: Selecting<Mask,Yes,No>,
  decision: Tasks.Decision,+yes: Model.Model<Tile.Tile8>,+mask: Model.Model<Tile.Tile8>,+no: Model.Model<Tile.Tile8>,+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,
  +total_exact: {selecting_total(~Mask,~Yes,~No,scope) == ShapeLaws.elements(shape) : Nat},
  +yes_contract: Global.Contract(~Tile.Tile8,~Representation.Lifted<Yes>,~Representation.lifted(~Yes,~yes_value),selecting_yes(~Mask,~Yes,~No,scope),
    Tensors.Tensor{Model.pack(~Tile.Tile8,yes),shape,fill,Tensors.Usable{},workers}),
  +mask_contract: Global.Contract(~Tile.Tile8,~Representation.Lifted<Mask>,~Representation.lifted(~Mask,~mask_value),selecting_mask(~Mask,~Yes,~No,scope),
    Tensors.Tensor{Model.pack(~Tile.Tile8,mask),shape,fill,Tensors.Usable{},workers}),
  +no_contract: Global.Contract(~Tile.Tile8,~Representation.Lifted<No>,~Representation.lifted(~No,~no_value),selecting_no(~Mask,~Yes,~No,scope),
    Tensors.Tensor{Model.pack(~Tile.Tile8,no),shape,fill,Tensors.Usable{},workers})) ->
  Global.Contract(~Tile.Tile8,~Representation.Lifted<Choice<Mask,Yes,No>>,~Representation.lifted(~Choice<Mask,Yes,No>,~value(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value)),selecting_after(~Mask,~Yes,~No,scope),
    Select.selected_decided(decision,selecting_padding(~Mask,~Yes,~No,scope),shape,fill,workers,
      Pieces.zip(~Select.Buffer(),~Pieces.Paired<Select.Buffer(),Select.Buffer()>,Model.pack(~Tile.Tile8,yes),
        Pieces.zip(~Select.Buffer(),~Select.Buffer(),Model.pack(~Tile.Tile8,mask),Model.pack(~Tile.Tile8,no))))):
  match decision:
    case Tasks.Decision{+parallel,+tasks}:
      KernelGemm.Both{KernelGemm.Both{Content.WellFormed{+yes_fields,+yes_ready,+coverage},KernelGemm.Both{+counted,+yes_canonical}},+yes_values} = yes_contract
      KernelGemm.Both{KernelGemm.Both{Content.WellFormed{mask_fields,mask_ready,mask_coverage},KernelGemm.Both{mask_counted,+mask_canonical}},+mask_values} = mask_contract
      KernelGemm.Both{KernelGemm.Both{Content.WellFormed{no_fields,no_ready,no_coverage},KernelGemm.Both{no_counted,+no_canonical}},+no_values} = no_contract
      +padding = selecting_padding(~Mask,~Yes,~No,scope)
      +total = ShapeLaws.elements(shape)
      +total_back = Equal.sym(Nat,selecting_total(~Mask,~Yes,~No,scope),total,total_exact)
      +yes_packed = Model.packed_contents(~Tile.Tile8,yes)
      +mask_packed = Model.packed_contents(~Tile.Tile8,mask)
      +no_packed = Model.packed_contents(~Tile.Tile8,no)
      +mask_common = Each.matched(~Tile.Tile8,yes,mask,PiecesCombine.same_layout(~Tile.Tile8,shape,yes,mask,yes_canonical,mask_canonical))
      +no_common = Each.matched(~Tile.Tile8,yes,no,PiecesCombine.same_layout(~Tile.Tile8,shape,yes,no,yes_canonical,no_canonical))
      +parts = triples(yes,mask,no)
      +output = shaped_content(padding,parts)
      +laid_out = {Equal.trans(Pieces.Ranges,Model.content_layout(~Tile.Tile8,output),Model.layout(~Tile.Tile8,yes),
          Model.content_layout(~Tile.Tile8,Model.contents(~Tile.Tile8,Model.pack(~Tile.Tile8,yes))),output_layout(padding,yes,mask,no,mask_common,no_common),
          PiecesCombine.packed_layout(~Tile.Tile8,yes)) :
        {Model.content_layout(~Tile.Tile8,output) == Model.content_layout(~Tile.Tile8,Model.contents(~Tile.Tile8,Model.pack(~Tile.Tile8,yes))) : Pieces.Ranges}}
      %Equal.sym(Partition.Tree<&1,Select.Triple(),Unit>,
          Pieces.zip(~Select.Buffer(),~Pieces.Paired<Select.Buffer(),Select.Buffer()>,Model.pack(~Tile.Tile8,yes),
            Pieces.zip(~Select.Buffer(),~Select.Buffer(),Model.pack(~Tile.Tile8,mask),Model.pack(~Tile.Tile8,no))),
          triple_tree(yes,mask,no),triple_zip(yes,mask,no,mask_common,no_common)) :
        Global.Contract(~Tile.Tile8,~Representation.Lifted<Choice<Mask,Yes,No>>,~Representation.lifted(~Choice<Mask,Yes,No>,~value(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value)),selecting_after(~Mask,~Yes,~No,scope),
          Tensors.Tensor{Pieces.grouped(~Select.Triple(),~Select.Buffer(),~TiledMap.Padding,~Select.selected_piece,tasks,parallel,padding,_),shape,fill,Tensors.Usable{},workers})
      %Equal.sym(Model.Content<Tile.Tile8>,
          Model.contents(~Tile.Tile8,Pieces.grouped(~Select.Triple(),~Select.Buffer(),~TiledMap.Padding,~Select.selected_piece,tasks,parallel,padding,triple_tree(yes,mask,no))),
          output,grouped_contents(padding,tasks,parallel,parts)) :
        PiecesMap.Holds(~Tile.Tile8,~Representation.Lifted<Choice<Mask,Yes,No>>,~Representation.lifted(~Choice<Mask,Yes,No>,~value(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value)),selecting_after(~Mask,~Yes,~No,scope),
          shape,_)
      PiecesMap.holds(~Tile.Tile8,~Representation.Lifted<Choice<Mask,Yes,No>>,~Representation.lifted(~Choice<Mask,Yes,No>,~value(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value)),selecting_after(~Mask,~Yes,~No,scope),
        shape,output,
        %total_exact : PiecesMap.Facts(~Tile.Tile8,~Representation.Lifted<Choice<Mask,Yes,No>>,~Representation.lifted(~Choice<Mask,Yes,No>,~value(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value)),
          selecting_after(~Mask,~Yes,~No,scope),_,output)
          shaped_facts(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,scope,parts,
            triple_all(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,scope,yes,mask,no,mask_common,no_common,
              %total_back : Content.Contents(~Tile.Tile8,_,Model.content(~Tile.Tile8,yes)) %yes_packed : Content.Contents(~Tile.Tile8,total,_) yes_fields,
              %yes_packed : Model.ContentReady(~Tile.Tile8,_) yes_ready,
              %yes_packed : Content.Values(~Tile.Tile8,~Representation.Lifted<Yes>,~Representation.lifted(~Yes,~yes_value),selecting_yes(~Mask,~Yes,~No,scope),_) yes_values,
              %mask_packed : Content.Values(~Tile.Tile8,~Representation.Lifted<Mask>,~Representation.lifted(~Mask,~mask_value),selecting_mask(~Mask,~Yes,~No,scope),_) mask_values,
              %no_packed : Content.Values(~Tile.Tile8,~Representation.Lifted<No>,~Representation.lifted(~No,~no_value),selecting_no(~Mask,~Yes,~No,scope),_) no_values)),
        %Equal.sym(Pieces.Ranges,Model.content_layout(~Tile.Tile8,output),Model.content_layout(~Tile.Tile8,Model.contents(~Tile.Tile8,Model.pack(~Tile.Tile8,yes))),laid_out) :
          PiecesPlan.Coverage(_,0n,total) coverage,
        counted,
        %Equal.sym(Pieces.Ranges,Model.content_layout(~Tile.Tile8,output),Model.content_layout(~Tile.Tile8,Model.contents(~Tile.Tile8,Model.pack(~Tile.Tile8,yes))),laid_out) :
          {_ == Pieces.ranges_of_grid(Tensors.grid_of(shape)) : Pieces.Ranges} yes_canonical)

# ---- The tiled operands ----

# The selection of the cells of three tensors of one layout.
def placed_select(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,+within: Bool,
  +mask: Mask,+yes: Yes,+no: No,+values: List<&2,U32>,+axis: Nat,+lanes: U32,+position: Grid.Position,decision: Tasks.Decision,
  +yes_model: Model.Model<Tile.Tile8>,+mask_model: Model.Model<Tile.Tile8>,+no_model: Model.Model<Tile.Tile8>,+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,
  +yes_contract: Global.Contract(~Tile.Tile8,~Representation.Lifted<Yes>,~Representation.lifted(~Yes,~yes_value),Representation.Lifted{yes,Public.geometry_of(values,axis,position)},
    Tensors.Tensor{Model.pack(~Tile.Tile8,yes_model),shape,fill,Tensors.Usable{},workers}),
  +mask_contract: Global.Contract(~Tile.Tile8,~Representation.Lifted<Mask>,~Representation.lifted(~Mask,~mask_value),Representation.Lifted{mask,Public.geometry_of(values,axis,position)},
    Tensors.Tensor{Model.pack(~Tile.Tile8,mask_model),shape,fill,Tensors.Usable{},workers}),
  +no_contract: Global.Contract(~Tile.Tile8,~Representation.Lifted<No>,~Representation.lifted(~No,~no_value),Representation.Lifted{no,Public.geometry_of(values,axis,position)},
    Tensors.Tensor{Model.pack(~Tile.Tile8,no_model),shape,fill,Tensors.Usable{},workers}),
  +placed: PaddedCells.Placed<shape,values,axis,lanes,position,within>) ->
  Global.Contract(~Tile.Tile8,~Representation.Lifted<Choice<Mask,Yes,No>>,~Representation.lifted(~Choice<Mask,Yes,No>,~value(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value)),
    Representation.Lifted{Choice{mask,yes,no},Public.geometry_of(values,axis,position)},
    Select.selected_decided(decision,TiledMap.padding_when(within,shape,lanes,position),shape,fill,workers,
      Pieces.zip(~Select.Buffer(),~Pieces.Paired<Select.Buffer(),Select.Buffer()>,Model.pack(~Tile.Tile8,yes_model),
        Pieces.zip(~Select.Buffer(),~Select.Buffer(),Model.pack(~Tile.Tile8,mask_model),Model.pack(~Tile.Tile8,no_model))))):
  PaddedCells.Placed{+layout,+padding_exact,+geometry_exact,+total_exact} = placed
  +back = Equal.sym(Cells.Cells,PaddedCells.layout_geometry(layout),Public.geometry_of(values,axis,position),geometry_exact)
  %padding_exact : Global.Contract(~Tile.Tile8,~Representation.Lifted<Choice<Mask,Yes,No>>,
    ~Representation.lifted(~Choice<Mask,Yes,No>,~value(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value)),
    Representation.Lifted{Choice{mask,yes,no},Public.geometry_of(values,axis,position)},
    Select.selected_decided(decision,_,shape,fill,workers,
      Pieces.zip(~Select.Buffer(),~Pieces.Paired<Select.Buffer(),Select.Buffer()>,Model.pack(~Tile.Tile8,yes_model),
        Pieces.zip(~Select.Buffer(),~Select.Buffer(),Model.pack(~Tile.Tile8,mask_model),Model.pack(~Tile.Tile8,no_model)))))
  %geometry_exact : Global.Contract(~Tile.Tile8,~Representation.Lifted<Choice<Mask,Yes,No>>,
    ~Representation.lifted(~Choice<Mask,Yes,No>,~value(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value)),Representation.Lifted{Choice{mask,yes,no},_},
    Select.selected_decided(decision,PaddedCells.layout_padding(layout),shape,fill,workers,
      Pieces.zip(~Select.Buffer(),~Pieces.Paired<Select.Buffer(),Select.Buffer()>,Model.pack(~Tile.Tile8,yes_model),
        Pieces.zip(~Select.Buffer(),~Select.Buffer(),Model.pack(~Tile.Tile8,mask_model),Model.pack(~Tile.Tile8,no_model)))))
  decided(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,Selecting{mask,yes,no,layout},decision,yes_model,mask_model,no_model,shape,fill,workers,total_exact,
    %back : Global.Contract(~Tile.Tile8,~Representation.Lifted<Yes>,~Representation.lifted(~Yes,~yes_value),Representation.Lifted{yes,_},
      Tensors.Tensor{Model.pack(~Tile.Tile8,yes_model),shape,fill,Tensors.Usable{},workers}) yes_contract,
    %back : Global.Contract(~Tile.Tile8,~Representation.Lifted<Mask>,~Representation.lifted(~Mask,~mask_value),Representation.Lifted{mask,_},
      Tensors.Tensor{Model.pack(~Tile.Tile8,mask_model),shape,fill,Tensors.Usable{},workers}) mask_contract,
    %back : Global.Contract(~Tile.Tile8,~Representation.Lifted<No>,~Representation.lifted(~No,~no_value),Representation.Lifted{no,_},
      Tensors.Tensor{Model.pack(~Tile.Tile8,no_model),shape,fill,Tensors.Usable{},workers}) no_contract)

# The selection of the cells of three tensors of one layout, their pieces
# observed as models.
def observed_select(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,+within: Bool,
  +mask: Mask,+yes: Yes,+no: No,+values: List<&2,U32>,+axis: Nat,+lanes: U32,+position: Grid.Position,decision: Tasks.Decision,
  -yes_pieces: Pieces.Pieces(~Tile.Tile8),-mask_pieces: Pieces.Pieces(~Tile.Tile8),-no_pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,
  yes_observation: Model.Observation<Tile.Tile8,yes_pieces>,mask_observation: Model.Observation<Tile.Tile8,mask_pieces>,no_observation: Model.Observation<Tile.Tile8,no_pieces>,
  +yes_contract: Global.Contract(~Tile.Tile8,~Representation.Lifted<Yes>,~Representation.lifted(~Yes,~yes_value),Representation.Lifted{yes,Public.geometry_of(values,axis,position)},
    Tensors.Tensor{yes_pieces,shape,fill,Tensors.Usable{},workers}),
  +mask_contract: Global.Contract(~Tile.Tile8,~Representation.Lifted<Mask>,~Representation.lifted(~Mask,~mask_value),Representation.Lifted{mask,Public.geometry_of(values,axis,position)},
    Tensors.Tensor{mask_pieces,shape,fill,Tensors.Usable{},workers}),
  +no_contract: Global.Contract(~Tile.Tile8,~Representation.Lifted<No>,~Representation.lifted(~No,~no_value),Representation.Lifted{no,Public.geometry_of(values,axis,position)},
    Tensors.Tensor{no_pieces,shape,fill,Tensors.Usable{},workers}),
  +placed: PaddedCells.Placed<shape,values,axis,lanes,position,within>) ->
  Global.Contract(~Tile.Tile8,~Representation.Lifted<Choice<Mask,Yes,No>>,~Representation.lifted(~Choice<Mask,Yes,No>,~value(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value)),
    Representation.Lifted{Choice{mask,yes,no},Public.geometry_of(values,axis,position)},
    Select.selected_decided(decision,TiledMap.padding_when(within,shape,lanes,position),shape,fill,workers,
      Pieces.zip(~Select.Buffer(),~Pieces.Paired<Select.Buffer(),Select.Buffer()>,yes_pieces,Pieces.zip(~Select.Buffer(),~Select.Buffer(),mask_pieces,no_pieces)))):
  Model.Observation{+yes_model,+yes_equation} = yes_observation
  Model.Observation{+mask_model,+mask_equation} = mask_observation
  Model.Observation{+no_model,+no_equation} = no_observation
  +after = {Representation.Lifted{Choice{mask,yes,no},Public.geometry_of(values,axis,position)} : Representation.Lifted<Choice<Mask,Yes,No>>}
  +padding = TiledMap.padding_when(within,shape,lanes,position)
  %yes_equation : Global.Contract(~Tile.Tile8,~Representation.Lifted<Choice<Mask,Yes,No>>,
    ~Representation.lifted(~Choice<Mask,Yes,No>,~value(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value)),after,
    Select.selected_decided(decision,padding,shape,fill,workers,
      Pieces.zip(~Select.Buffer(),~Pieces.Paired<Select.Buffer(),Select.Buffer()>,_,Pieces.zip(~Select.Buffer(),~Select.Buffer(),mask_pieces,no_pieces))))
  %mask_equation : Global.Contract(~Tile.Tile8,~Representation.Lifted<Choice<Mask,Yes,No>>,
    ~Representation.lifted(~Choice<Mask,Yes,No>,~value(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value)),after,
    Select.selected_decided(decision,padding,shape,fill,workers,
      Pieces.zip(~Select.Buffer(),~Pieces.Paired<Select.Buffer(),Select.Buffer()>,Model.pack(~Tile.Tile8,yes_model),
        Pieces.zip(~Select.Buffer(),~Select.Buffer(),_,no_pieces))))
  %no_equation : Global.Contract(~Tile.Tile8,~Representation.Lifted<Choice<Mask,Yes,No>>,
    ~Representation.lifted(~Choice<Mask,Yes,No>,~value(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value)),after,
    Select.selected_decided(decision,padding,shape,fill,workers,
      Pieces.zip(~Select.Buffer(),~Pieces.Paired<Select.Buffer(),Select.Buffer()>,Model.pack(~Tile.Tile8,yes_model),
        Pieces.zip(~Select.Buffer(),~Select.Buffer(),Model.pack(~Tile.Tile8,mask_model),_))))
  placed_select(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,within,mask,yes,no,values,axis,lanes,position,decision,yes_model,mask_model,no_model,shape,fill,workers,
    %Equal.sym(Pieces.Pieces(~Tile.Tile8),Model.pack(~Tile.Tile8,yes_model),yes_pieces,yes_equation) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Yes>,~Representation.lifted(~Yes,~yes_value),Representation.Lifted{yes,Public.geometry_of(values,axis,position)},
        Tensors.Tensor{_,shape,fill,Tensors.Usable{},workers}) yes_contract,
    %Equal.sym(Pieces.Pieces(~Tile.Tile8),Model.pack(~Tile.Tile8,mask_model),mask_pieces,mask_equation) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Mask>,~Representation.lifted(~Mask,~mask_value),Representation.Lifted{mask,Public.geometry_of(values,axis,position)},
        Tensors.Tensor{_,shape,fill,Tensors.Usable{},workers}) mask_contract,
    %Equal.sym(Pieces.Pieces(~Tile.Tile8),Model.pack(~Tile.Tile8,no_model),no_pieces,no_equation) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<No>,~Representation.lifted(~No,~no_value),Representation.Lifted{no,Public.geometry_of(values,axis,position)},
        Tensors.Tensor{_,shape,fill,Tensors.Usable{},workers}) no_contract,
    placed)

# The selection on the tasks of a decision holds the selected reader in the
# layout of the yes cells.
def decided_within(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,
  +mask: Mask,+yes: Yes,+no: No,+values: List<&2,U32>,+axis: Nat,+lanes: U32,decision: Tasks.Decision,yes_pieces: Pieces.Pieces(~Tile.Tile8),
  mask_pieces: Pieces.Pieces(~Tile.Tile8),no_pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,
  +yes_contract: Global.Contract(~Tile.Tile8,~Representation.Lifted<Yes>,~Representation.lifted(~Yes,~yes_value),
    Representation.Lifted{yes,Public.geometry_of(values,axis,Grid.Innermost{})},Tensors.Tensor{yes_pieces,shape,fill,Tensors.Usable{},workers}),
  +mask_contract: Global.Contract(~Tile.Tile8,~Representation.Lifted<Mask>,~Representation.lifted(~Mask,~mask_value),
    Representation.Lifted{mask,Public.geometry_of(values,axis,Grid.Innermost{})},Tensors.Tensor{mask_pieces,shape,fill,Tensors.Usable{},workers}),
  +no_contract: Global.Contract(~Tile.Tile8,~Representation.Lifted<No>,~Representation.lifted(~No,~no_value),
    Representation.Lifted{no,Public.geometry_of(values,axis,Grid.Innermost{})},Tensors.Tensor{no_pieces,shape,fill,Tensors.Usable{},workers}),
  +laid: Public.Laid<shape,axis,lanes,Grid.Innermost{},values>) ->
  Public.Within(~Representation.Lifted<Choice<Mask,Yes,No>>,~Representation.lifted(~Choice<Mask,Yes,No>,~value(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value)),
    ~Choice<Mask,Yes,No>,~Public.lifted_at(~Choice<Mask,Yes,No>),Choice{mask,yes,no},values,axis,lanes,Grid.Innermost{},
    Select.selected_decided(decision,TiledMap.padding_of(shape,axis,lanes,Grid.Innermost{}),shape,fill,workers,
      Pieces.zip(~Select.Buffer(),~Pieces.Paired<Select.Buffer(),Select.Buffer()>,yes_pieces,Pieces.zip(~Select.Buffer(),~Select.Buffer(),mask_pieces,no_pieces)))):
  match decision:
    case Tasks.Decision{+parallel,+tasks}:
      +within = Nat.is_lt(axis,List.length(&2,U32,shape))
      KernelGemm.Both{observed_select(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,within,mask,yes,no,values,axis,lanes,Grid.Innermost{},Tasks.Decision{parallel,tasks},
          yes_pieces,mask_pieces,no_pieces,shape,fill,workers,Model.observe(~Tile.Tile8,yes_pieces),Model.observe(~Tile.Tile8,mask_pieces),Model.observe(~Tile.Tile8,no_pieces),
          yes_contract,mask_contract,no_contract,PaddedCells.placed(within,values,axis,lanes,Grid.Innermost{},shape,{==},laid)),
        laid}

def Selected(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,+mask: Mask,+yes: Yes,+no: No,
  +values: List<&2,U32>,+workers: U32,tiled: Tiled.Tiled) -> Data:
  Public.Contract(~Choice<Mask,Yes,No>,~value(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value),Choice{mask,yes,no},values,tiled)

def tiles_of(+workers: U32,+axis: Nat,+lanes: U32,mask: Tensors.Tensor<Tile.Tile8>,yes: Tensors.Tensor<Tile.Tile8>,no: Tensors.Tensor<Tile.Tile8>) -> Tiled.Tiled:
  Select.tiles(workers,Tiled.Tiled{mask,axis,lanes,Grid.Innermost{}},TiledMap.padding_cells(Tiled.Tiled{yes,axis,lanes,Grid.Innermost{}}),Tiled.Tiled{no,axis,lanes,Grid.Innermost{}})

# Three usable tensors of the contract with their lanes on one axis and their
# groups innermost select into the selected reader.
def usable_select(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,
  +mask: Mask,+yes: Yes,+no: No,+values: List<&2,U32>,+axis: Nat,+lanes: U32,+workers: U32,
  mask_pieces: Pieces.Pieces(~Tile.Tile8),+mask_shape: List<&2,U32>,+mask_fill: Tile.Tile8,+mask_workers: U32,
  yes_pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+yes_workers: U32,
  no_pieces: Pieces.Pieces(~Tile.Tile8),+no_shape: List<&2,U32>,+no_fill: Tile.Tile8,+no_workers: U32,
  +mask_contract: Public.Contract(~Mask,~mask_value,mask,values,Tiled.Tiled{Tensors.Tensor{mask_pieces,mask_shape,mask_fill,Tensors.Usable{},mask_workers},axis,lanes,Grid.Innermost{}}),
  +yes_contract: Public.Contract(~Yes,~yes_value,yes,values,Tiled.Tiled{Tensors.Tensor{yes_pieces,shape,fill,Tensors.Usable{},yes_workers},axis,lanes,Grid.Innermost{}}),
  +no_contract: Public.Contract(~No,~no_value,no,values,Tiled.Tiled{Tensors.Tensor{no_pieces,no_shape,no_fill,Tensors.Usable{},no_workers},axis,lanes,Grid.Innermost{}})) ->
  Selected(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,workers,
    tiles_of(workers,axis,lanes,Tensors.Tensor{mask_pieces,mask_shape,mask_fill,Tensors.Usable{},mask_workers},Tensors.Tensor{yes_pieces,shape,fill,Tensors.Usable{},yes_workers},
      Tensors.Tensor{no_pieces,no_shape,no_fill,Tensors.Usable{},no_workers})):
  KernelGemm.Both{+mask_global,+mask_laid} = mask_contract
  KernelGemm.Both{+yes_global,+yes_laid} = yes_contract
  KernelGemm.Both{+no_global,+no_laid} = no_contract
  +cells = Tiled.cell_shape(values,axis,Grid.Innermost{})
  +yes_shaped = TiledMapLaws.laid_shape(shape,axis,lanes,Grid.Innermost{},values,yes_laid)
  +mask_same = Equal.trans(List<&2,U32>,mask_shape,cells,shape,
    Equal.sym(List<&2,U32>,cells,mask_shape,TiledMapLaws.laid_shape(mask_shape,axis,lanes,Grid.Innermost{},values,mask_laid)),yes_shaped)
  +no_same = Equal.trans(List<&2,U32>,no_shape,cells,shape,
    Equal.sym(List<&2,U32>,cells,no_shape,TiledMapLaws.laid_shape(no_shape,axis,lanes,Grid.Innermost{},values,no_laid)),yes_shaped)
  decided_within(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,axis,lanes,Tasks.decide(Tensors.grid_of(shape),workers,4.0,7.0,Tasks.no_extra()),
    yes_pieces,mask_pieces,no_pieces,shape,fill,workers,yes_global,
    %mask_same : Global.Contract(~Tile.Tile8,~Representation.Lifted<Mask>,~Representation.lifted(~Mask,~mask_value),
      Representation.Lifted{mask,Public.geometry_of(values,axis,Grid.Innermost{})},Tensors.Tensor{mask_pieces,_,mask_fill,Tensors.Usable{},mask_workers}) mask_global,
    %no_same : Global.Contract(~Tile.Tile8,~Representation.Lifted<No>,~Representation.lifted(~No,~no_value),
      Representation.Lifted{no,Public.geometry_of(values,axis,Grid.Innermost{})},Tensors.Tensor{no_pieces,_,no_fill,Tensors.Usable{},no_workers}) no_global,
    yes_laid)

# Three tensors of the contract with their lanes on one axis and their groups
# innermost select into the selected reader; a failed operand states nothing.
def cells_select(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,
  +mask: Mask,+yes: Yes,+no: No,+values: List<&2,U32>,+axis: Nat,+lanes: U32,+workers: U32,
  mask_cells: Tensors.Tensor<Tile.Tile8>,yes_cells: Tensors.Tensor<Tile.Tile8>,no_cells: Tensors.Tensor<Tile.Tile8>,
  +mask_contract: Public.Contract(~Mask,~mask_value,mask,values,Tiled.Tiled{mask_cells,axis,lanes,Grid.Innermost{}}),
  +yes_contract: Public.Contract(~Yes,~yes_value,yes,values,Tiled.Tiled{yes_cells,axis,lanes,Grid.Innermost{}}),
  +no_contract: Public.Contract(~No,~no_value,no,values,Tiled.Tiled{no_cells,axis,lanes,Grid.Innermost{}})) ->
  Selected(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,workers,tiles_of(workers,axis,lanes,mask_cells,yes_cells,no_cells)):
  match mask_cells:
    case Tensors.Tensor{mask_pieces,+mask_shape,+mask_fill,Tensors.Usable{},+mask_workers}:
      match yes_cells:
        case Tensors.Tensor{yes_pieces,+shape,+fill,Tensors.Usable{},+yes_workers}:
          match no_cells:
            case Tensors.Tensor{no_pieces,+no_shape,+no_fill,Tensors.Usable{},+no_workers}:
              usable_select(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,axis,lanes,workers,mask_pieces,mask_shape,mask_fill,mask_workers,
                yes_pieces,shape,fill,yes_workers,no_pieces,no_shape,no_fill,no_workers,mask_contract,yes_contract,no_contract)
            case Tensors.Tensor{no_pieces,+no_shape,+no_fill,Tensors.Failed{+problem},+no_workers}:
              Empty.absurd(Selected(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,workers,
                tiles_of(workers,axis,lanes,Tensors.Tensor{mask_pieces,mask_shape,mask_fill,Tensors.Usable{},mask_workers},Tensors.Tensor{yes_pieces,shape,fill,Tensors.Usable{},yes_workers},
                  Tensors.Tensor{no_pieces,no_shape,no_fill,Tensors.Failed{problem},no_workers})),no_contract)
        case Tensors.Tensor{yes_pieces,+shape,+fill,Tensors.Failed{+problem},+yes_workers}:
          Empty.absurd(Selected(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,workers,
            tiles_of(workers,axis,lanes,Tensors.Tensor{mask_pieces,mask_shape,mask_fill,Tensors.Usable{},mask_workers},Tensors.Tensor{yes_pieces,shape,fill,Tensors.Failed{problem},yes_workers},
              no_cells)),yes_contract)
    case Tensors.Tensor{mask_pieces,+mask_shape,+mask_fill,Tensors.Failed{+problem},+mask_workers}:
      Empty.absurd(Selected(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,workers,
        tiles_of(workers,axis,lanes,Tensors.Tensor{mask_pieces,mask_shape,mask_fill,Tensors.Failed{problem},mask_workers},yes_cells,no_cells)),mask_contract)

# Three tensors of the contract arranged with their lanes on one axis and
# their groups innermost select into the selected reader.
def arranged_select(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,
  +mask: Mask,+yes: Yes,+no: No,+values: List<&2,U32>,+workers: U32,mask_tiled: Tiled.Tiled,yes_tiled: Tiled.Tiled,no_tiled: Tiled.Tiled,+axis: Nat,
  +mask_held: Relane.Arranged(~Mask,~mask_value,mask,values,mask_tiled,axis,Grid.Innermost{}),
  +yes_held: Relane.Arranged(~Yes,~yes_value,yes,values,yes_tiled,axis,Grid.Innermost{}),
  +no_held: Relane.Arranged(~No,~no_value,no,values,no_tiled,axis,Grid.Innermost{})) ->
  Selected(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,workers,Select.chosen_along(workers,mask_tiled,(yes_tiled,axis),no_tiled)):
  KernelGemm.Both{+mask_form,+mask_contract} = mask_held
  KernelGemm.Both{+yes_form,+yes_contract} = yes_held
  KernelGemm.Both{+no_form,+no_contract} = no_held
  +lanes = Grid.lanes_at(values,axis)
  %Equal.sym(Tiled.Tiled,Tiled.arranged(mask_tiled,axis,Grid.Innermost{}),
      Tiled.Tiled{Tiled.cells_of(Tiled.arranged(mask_tiled,axis,Grid.Innermost{})),axis,lanes,Grid.Innermost{}},mask_form) :
    Selected(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,workers,
      Select.tiles(workers,_,TiledMap.padding_cells(Tiled.arranged(yes_tiled,axis,Grid.Innermost{})),Tiled.arranged(no_tiled,axis,Grid.Innermost{})))
  %Equal.sym(Tiled.Tiled,Tiled.arranged(yes_tiled,axis,Grid.Innermost{}),
      Tiled.Tiled{Tiled.cells_of(Tiled.arranged(yes_tiled,axis,Grid.Innermost{})),axis,lanes,Grid.Innermost{}},yes_form) :
    Selected(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,workers,
      Select.tiles(workers,Tiled.Tiled{Tiled.cells_of(Tiled.arranged(mask_tiled,axis,Grid.Innermost{})),axis,lanes,Grid.Innermost{}},TiledMap.padding_cells(_),
        Tiled.arranged(no_tiled,axis,Grid.Innermost{})))
  %Equal.sym(Tiled.Tiled,Tiled.arranged(no_tiled,axis,Grid.Innermost{}),
      Tiled.Tiled{Tiled.cells_of(Tiled.arranged(no_tiled,axis,Grid.Innermost{})),axis,lanes,Grid.Innermost{}},no_form) :
    Selected(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,workers,
      Select.tiles(workers,Tiled.Tiled{Tiled.cells_of(Tiled.arranged(mask_tiled,axis,Grid.Innermost{})),axis,lanes,Grid.Innermost{}},
        TiledMap.padding_cells(Tiled.Tiled{Tiled.cells_of(Tiled.arranged(yes_tiled,axis,Grid.Innermost{})),axis,lanes,Grid.Innermost{}}),_))
  cells_select(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,axis,lanes,workers,Tiled.cells_of(Tiled.arranged(mask_tiled,axis,Grid.Innermost{})),
    Tiled.cells_of(Tiled.arranged(yes_tiled,axis,Grid.Innermost{})),Tiled.cells_of(Tiled.arranged(no_tiled,axis,Grid.Innermost{})),mask_contract,yes_contract,no_contract)

# Three packed tensors of the contract of one shape select on the lane axis
# of yes.
def chosen_packed(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,
  +mask: Mask,+yes: Yes,+no: No,+values: List<&2,U32>,+workers: U32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +mm: Model.Model<Tile.Tile8>,+mc: List<&2,U32>,+mf: Tile.Tile8,+mw: U32,+mx: Nat,+ml: U32,+mp: Grid.Position,
  +ym: Model.Model<Tile.Tile8>,+yc: List<&2,U32>,+yf: Tile.Tile8,+yw: U32,+yx: Nat,+yl: U32,+yp: Grid.Position,
  +nm: Model.Model<Tile.Tile8>,+nc: List<&2,U32>,+nf: Tile.Tile8,+nw: U32,+nx: Nat,+nl: U32,+np: Grid.Position,
  +mask_contract: Public.Contract(~Mask,~mask_value,mask,values,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp}),
  +yes_contract: Public.Contract(~Yes,~yes_value,yes,values,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp}),
  +no_contract: Public.Contract(~No,~no_value,no,values,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,nm),nc,nf,Tensors.Usable{},nw},nx,nl,np})) ->
  Selected(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,workers,Select.chosen(workers,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp},Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp},Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,nm),nc,nf,Tensors.Usable{},nw},nx,nl,np})):
  +inside = Broadcast.packed_inside(~Yes,~yes_value,yes,values,ym,yc,yf,yw,yx,yl,yp,yes_contract)
  arranged_select(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,workers,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp},Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp},Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,nm),nc,nf,Tensors.Usable{},nw},nx,nl,np},yx,
    Relane.arranged_held(~Mask,~mask_value,mask,values,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp},yx,Grid.Innermost{},limits,default_capacity,inside,nonempty,rank_room,mask_contract),
    Relane.arranged_held(~Yes,~yes_value,yes,values,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp},yx,Grid.Innermost{},limits,default_capacity,inside,nonempty,rank_room,yes_contract),
    Relane.arranged_held(~No,~no_value,no,values,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,nm),nc,nf,Tensors.Usable{},nw},nx,nl,np},yx,Grid.Innermost{},limits,default_capacity,inside,nonempty,rank_room,no_contract))

def chosen_described(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,
  +mask: Mask,+yes: Yes,+no: No,+values: List<&2,U32>,+workers: U32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  -mask_tiled: Tiled.Tiled,+mask_packed: Public.Packed<mask_tiled>,-yes_tiled: Tiled.Tiled,+yes_packed: Public.Packed<yes_tiled>,
  -no_tiled: Tiled.Tiled,+no_packed: Public.Packed<no_tiled>,+mask_contract: Public.Contract(~Mask,~mask_value,mask,values,mask_tiled),
  +yes_contract: Public.Contract(~Yes,~yes_value,yes,values,yes_tiled),+no_contract: Public.Contract(~No,~no_value,no,values,no_tiled)) ->
  Selected(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,workers,Select.chosen(workers,mask_tiled,yes_tiled,no_tiled)):
  Public.Packed{+mm,+mc,+mf,+mw,+mx,+ml,+mp,+mask_equation} = mask_packed
  Public.Packed{+ym,+yc,+yf,+yw,+yx,+yl,+yp,+yes_equation} = yes_packed
  Public.Packed{+nm,+nc,+nf,+nw,+nx,+nl,+np,+no_equation} = no_packed
  %mask_equation : Selected(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,workers,Select.chosen(workers,_,yes_tiled,no_tiled))
  %yes_equation : Selected(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,workers,Select.chosen(workers,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp},_,no_tiled))
  %no_equation : Selected(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,workers,Select.chosen(workers,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp},Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp},_))
  chosen_packed(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,workers,limits,default_capacity,nonempty,rank_room,mm,mc,mf,mw,mx,ml,mp,ym,yc,yf,yw,yx,yl,yp,nm,nc,nf,nw,nx,nl,np,
    {%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp},mask_tiled,mask_equation) : Public.Contract(~Mask,~mask_value,mask,values,_) mask_contract :
      Public.Contract(~Mask,~mask_value,mask,values,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp})},
    {%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp},yes_tiled,yes_equation) : Public.Contract(~Yes,~yes_value,yes,values,_) yes_contract :
      Public.Contract(~Yes,~yes_value,yes,values,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp})},
    {%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,nm),nc,nf,Tensors.Usable{},nw},nx,nl,np},no_tiled,no_equation) : Public.Contract(~No,~no_value,no,values,_) no_contract :
      Public.Contract(~No,~no_value,no,values,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,nm),nc,nf,Tensors.Usable{},nw},nx,nl,np})})

# Three tensors of the contract of one shape select on the lane axis of yes
# into the selected reader.
def chosen_contract(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,
  +mask: Mask,+yes: Yes,+no: No,+values: List<&2,U32>,+workers: U32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  mask_tiled: Tiled.Tiled,+mask_contract: Public.Contract(~Mask,~mask_value,mask,values,mask_tiled),
  yes_tiled: Tiled.Tiled,+yes_contract: Public.Contract(~Yes,~yes_value,yes,values,yes_tiled),
  no_tiled: Tiled.Tiled,+no_contract: Public.Contract(~No,~no_value,no,values,no_tiled)) ->
  Selected(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,workers,Select.chosen(workers,mask_tiled,yes_tiled,no_tiled)):
  chosen_described(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,workers,limits,default_capacity,nonempty,rank_room,
    mask_tiled,Public.packed(~Mask,~mask_value,mask,values,mask_tiled,mask_contract),yes_tiled,Public.packed(~Yes,~yes_value,yes,values,yes_tiled,yes_contract),
    no_tiled,Public.packed(~No,~no_value,no,values,no_tiled,no_contract),mask_contract,yes_contract,no_contract)

# ---- Operands of one shape ----

def same_packed(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,+mask: Mask,+yes: Yes,+no: No,+values: List<&2,U32>,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +mm: Model.Model<Tile.Tile8>,+mc: List<&2,U32>,+mf: Tile.Tile8,+mw: U32,+mx: Nat,+ml: U32,+mp: Grid.Position,
  +ym: Model.Model<Tile.Tile8>,+yc: List<&2,U32>,+yf: Tile.Tile8,+yw: U32,+yx: Nat,+yl: U32,+yp: Grid.Position,
  +nm: Model.Model<Tile.Tile8>,+nc: List<&2,U32>,+nf: Tile.Tile8,+nw: U32,+nx: Nat,+nl: U32,+np: Grid.Position,
  +mask_contract: Public.Contract(~Mask,~mask_value,mask,values,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp}),
  +yes_contract: Public.Contract(~Yes,~yes_value,yes,values,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp}),
  +no_contract: Public.Contract(~No,~no_value,no,values,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,nm),nc,nf,Tensors.Usable{},nw},nx,nl,np})) ->
  Selected(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,U32.max(mw,U32.max(yw,nw)),F.select(Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp},Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp},Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,nm),nc,nf,Tensors.Usable{},nw},nx,nl,np})):
  KernelGemm.Both{mask_global,Public.Laid{mask_inside,mask_shaped,+mask_restored,mask_counted,mask_pinned}} = mask_contract
  KernelGemm.Both{yes_global,Public.Laid{yes_inside,yes_shaped,+yes_restored,+yes_counted,yes_pinned}} = yes_contract
  KernelGemm.Both{no_global,Public.Laid{no_inside,no_shaped,+no_restored,no_counted,no_pinned}} = no_contract
  +workers = U32.max(mw,U32.max(yw,nw))
  +self = Equal.sym(Maybe<&2,List<&2,U32>>,Shapes.broadcast(values,values),Some{values},UnitBroadcast.self_broadcast(values))
  +same = Equal.sym(Bool,Shapes.same_extents(values,values),True{},ShapeLaws.same_extents_reflexive(values))
  %Equal.sym(List<&2,U32>,Tiled.logical_shape(mc,mx,ml,mp),values,mask_restored) :
    Selected(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,workers,Select.values_shape(Shapes.broadcast(Tiled.logical_shape(yc,yx,yl,yp),Tiled.logical_shape(nc,nx,nl,np)),_,Tiled.logical_shape(yc,yx,yl,yp),Tiled.logical_shape(nc,nx,nl,np),workers,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp},Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp},Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,nm),nc,nf,Tensors.Usable{},nw},nx,nl,np}))
  %Equal.sym(List<&2,U32>,Tiled.logical_shape(yc,yx,yl,yp),values,yes_restored) :
    Selected(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,workers,Select.values_shape(Shapes.broadcast(_,Tiled.logical_shape(nc,nx,nl,np)),values,_,Tiled.logical_shape(nc,nx,nl,np),workers,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp},Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp},Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,nm),nc,nf,Tensors.Usable{},nw},nx,nl,np}))
  %Equal.sym(List<&2,U32>,Tiled.logical_shape(nc,nx,nl,np),values,no_restored) :
    Selected(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,workers,Select.values_shape(Shapes.broadcast(values,_),values,values,_,workers,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp},Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp},Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,nm),nc,nf,Tensors.Usable{},nw},nx,nl,np}))
  %self : Selected(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,workers,Select.values_shape(_,values,values,values,workers,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp},Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp},Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,nm),nc,nf,Tensors.Usable{},nw},nx,nl,np}))
  %self : Selected(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,workers,Select.masked_shape(_,values,values,values,values,workers,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp},Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp},Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,nm),nc,nf,Tensors.Usable{},nw},nx,nl,np}))
  %same : Selected(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,workers,Select.shapes(_ && Shapes.same_extents(values,values),values,Tensors.count_of(values),values,values,values,workers,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp},Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp},Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,nm),nc,nf,Tensors.Usable{},nw},nx,nl,np}))
  %same : Selected(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,workers,Select.shapes(_,values,Tensors.count_of(values),values,values,values,workers,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp},Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp},Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,nm),nc,nf,Tensors.Usable{},nw},nx,nl,np}))
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(values),Some{Global.count(values,yes_counted)},Global.accepted(values,yes_counted)) :
    Selected(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,workers,Select.shapes(True{},values,_,values,values,values,workers,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp},Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp},Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,nm),nc,nf,Tensors.Usable{},nw},nx,nl,np}))
  chosen_contract(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,workers,limits,default_capacity,nonempty,rank_room,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp},mask_contract,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp},yes_contract,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,nm),nc,nf,Tensors.Usable{},nw},nx,nl,np},no_contract)

def same_described(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,+mask: Mask,+yes: Yes,+no: No,+values: List<&2,U32>,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  -mask_tiled: Tiled.Tiled,+mask_packed: Public.Packed<mask_tiled>,-yes_tiled: Tiled.Tiled,+yes_packed: Public.Packed<yes_tiled>,
  -no_tiled: Tiled.Tiled,+no_packed: Public.Packed<no_tiled>,+mask_contract: Public.Contract(~Mask,~mask_value,mask,values,mask_tiled),
  +yes_contract: Public.Contract(~Yes,~yes_value,yes,values,yes_tiled),+no_contract: Public.Contract(~No,~no_value,no,values,no_tiled)) ->
  Public.Contract(~Choice<Mask,Yes,No>,~value(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value),Choice{mask,yes,no},values,F.select(mask_tiled,yes_tiled,no_tiled)):
  Public.Packed{+mm,+mc,+mf,+mw,+mx,+ml,+mp,+mask_equation} = mask_packed
  Public.Packed{+ym,+yc,+yf,+yw,+yx,+yl,+yp,+yes_equation} = yes_packed
  Public.Packed{+nm,+nc,+nf,+nw,+nx,+nl,+np,+no_equation} = no_packed
  %mask_equation : Public.Contract(~Choice<Mask,Yes,No>,~value(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value),Choice{mask,yes,no},values,F.select(_,yes_tiled,no_tiled))
  %yes_equation : Public.Contract(~Choice<Mask,Yes,No>,~value(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value),Choice{mask,yes,no},values,F.select(Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp},_,no_tiled))
  %no_equation : Public.Contract(~Choice<Mask,Yes,No>,~value(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value),Choice{mask,yes,no},values,F.select(Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp},Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp},_))
  same_packed(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,limits,default_capacity,nonempty,rank_room,mm,mc,mf,mw,mx,ml,mp,ym,yc,yf,yw,yx,yl,yp,nm,nc,nf,nw,nx,nl,np,
    {%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp},mask_tiled,mask_equation) : Public.Contract(~Mask,~mask_value,mask,values,_) mask_contract :
      Public.Contract(~Mask,~mask_value,mask,values,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp})},
    {%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp},yes_tiled,yes_equation) : Public.Contract(~Yes,~yes_value,yes,values,_) yes_contract :
      Public.Contract(~Yes,~yes_value,yes,values,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp})},
    {%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,nm),nc,nf,Tensors.Usable{},nw},nx,nl,np},no_tiled,no_equation) : Public.Contract(~No,~no_value,no,values,_) no_contract :
      Public.Contract(~No,~no_value,no,values,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,nm),nc,nf,Tensors.Usable{},nw},nx,nl,np})})

# Three tiled operands of the contract and of one shape without a zero extent
# select into a tensor of the shape that holds, at every flat index, the yes
# value where the mask is nonzero and the no value elsewhere.
def same(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,+mask: Mask,+yes: Yes,+no: No,+values: List<&2,U32>,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  mask_tiled: Tiled.Tiled,+mask_contract: Public.Contract(~Mask,~mask_value,mask,values,mask_tiled),
  yes_tiled: Tiled.Tiled,+yes_contract: Public.Contract(~Yes,~yes_value,yes,values,yes_tiled),
  no_tiled: Tiled.Tiled,+no_contract: Public.Contract(~No,~no_value,no,values,no_tiled)) ->
  Public.Contract(~Choice<Mask,Yes,No>,~value(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value),Choice{mask,yes,no},values,F.select(mask_tiled,yes_tiled,no_tiled)):
  same_described(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,values,limits,default_capacity,nonempty,rank_room,
    mask_tiled,Public.packed(~Mask,~mask_value,mask,values,mask_tiled,mask_contract),yes_tiled,Public.packed(~Yes,~yes_value,yes,values,yes_tiled,yes_contract),
    no_tiled,Public.packed(~No,~no_value,no,values,no_tiled,no_contract),mask_contract,yes_contract,no_contract)

# ---- Operands that broadcast ----

# A view stretched from counted extents to extents without a zero fits the
# values of the source.
def view_fits(+source: List<&2,U32>,+shape: List<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32,+counted: Global.Counted<source>,
  +stretched: {Tensors.stretched(Tensors.view_of(source),shape) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>},+nonzero: {Shapes.has_zero(shape) == False{} : Bool}) ->
  {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(source)) == True{} : Bool}:
  Global.Counted{+count,+accepted} = counted
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(source),Some{count},accepted) : {Views.fits(Views.View{axes,offset},Tensors.known_count(_)) == True{} : Bool}
  Stretched.stretched_fits(source,shape,count,Global.whole_counted(source,count,accepted),axes,offset,stretched,nonzero)

# Three packed operands of the contract, each repeated to the common extents,
# select into the selection of the values each reads through its view
# stretched to those extents.
def repeated_select(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,+mask: Mask,+yes: Yes,+no: No,+ms: List<&2,U32>,+ys: List<&2,U32>,+ns: List<&2,U32>,+shape: List<&2,U32>,+workers: U32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +mask_axes: List<&2,Views.Axis>,+mask_offset: U32,+yes_axes: List<&2,Views.Axis>,+yes_offset: U32,+no_axes: List<&2,Views.Axis>,+no_offset: U32,
  +mask_stretched: {Tensors.stretched(Tensors.view_of(ms),shape) == Some{Views.View{mask_axes,mask_offset}} : Maybe<&2,Views.View>},+yes_stretched: {Tensors.stretched(Tensors.view_of(ys),shape) == Some{Views.View{yes_axes,yes_offset}} : Maybe<&2,Views.View>},
  +no_stretched: {Tensors.stretched(Tensors.view_of(ns),shape) == Some{Views.View{no_axes,no_offset}} : Maybe<&2,Views.View>},
  +mask_nonzero: {Shapes.has_zero(ms) == False{} : Bool},+yes_nonzero: {Shapes.has_zero(ys) == False{} : Bool},+no_nonzero: {Shapes.has_zero(ns) == False{} : Bool},
  +shape_nonzero: {Shapes.has_zero(shape) == False{} : Bool},+mask_rank: {Nat.is_le(List.length(&2,U32,ms),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+yes_rank: {Nat.is_le(List.length(&2,U32,ys),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+no_rank: {Nat.is_le(List.length(&2,U32,ns),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+shape_rank: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +counted: Global.Counted<shape>,
  +mm: Model.Model<Tile.Tile8>,+mc: List<&2,U32>,+mf: Tile.Tile8,+mw: U32,+mx: Nat,+ml: U32,+mp: Grid.Position,
  +ym: Model.Model<Tile.Tile8>,+yc: List<&2,U32>,+yf: Tile.Tile8,+yw: U32,+yx: Nat,+yl: U32,+yp: Grid.Position,
  +nm: Model.Model<Tile.Tile8>,+nc: List<&2,U32>,+nf: Tile.Tile8,+nw: U32,+nx: Nat,+nl: U32,+np: Grid.Position,
  +mask_contract: Public.Contract(~Mask,~mask_value,mask,ms,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp}),
  +yes_contract: Public.Contract(~Yes,~yes_value,yes,ys,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp}),
  +no_contract: Public.Contract(~No,~no_value,no,ns,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,nm),nc,nf,Tensors.Usable{},nw},nx,nl,np})) ->
  Selected(~ViewGather.Viewing<Mask>,~ViewGather.Viewing<Yes>,~ViewGather.Viewing<No>,~ViewGather.viewed(~F32,~Mask,~mask_value),~ViewGather.viewed(~F32,~Yes,~yes_value),~ViewGather.viewed(~F32,~No,~no_value),ViewGather.Viewing{mask,mask_axes,mask_offset,[]},ViewGather.Viewing{yes,yes_axes,yes_offset,[]},ViewGather.Viewing{no,no_axes,no_offset,[]},shape,workers,Select.chosen(workers,TiledBroadcast.repeated(Shapes.same_extents(ms,shape),shape,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp}),TiledBroadcast.repeated(Shapes.same_extents(ys,shape),shape,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp}),TiledBroadcast.repeated(Shapes.same_extents(ns,shape),shape,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,nm),nc,nf,Tensors.Usable{},nw},nx,nl,np}))):
  KernelGemm.Both{mask_global,Public.Laid{mask_inside,mask_shaped,mask_restored,+mask_counted,mask_pinned}} = mask_contract
  KernelGemm.Both{yes_global,Public.Laid{yes_inside,yes_shaped,yes_restored,+yes_counted,yes_pinned}} = yes_contract
  KernelGemm.Both{no_global,Public.Laid{no_inside,no_shaped,no_restored,+no_counted,no_pinned}} = no_contract
  +count = Global.count(shape,counted)
  +accepted = Global.accepted(shape,counted)
  chosen_contract(~ViewGather.Viewing<Mask>,~ViewGather.Viewing<Yes>,~ViewGather.Viewing<No>,~ViewGather.viewed(~F32,~Mask,~mask_value),~ViewGather.viewed(~F32,~Yes,~yes_value),~ViewGather.viewed(~F32,~No,~no_value),ViewGather.Viewing{mask,mask_axes,mask_offset,[]},ViewGather.Viewing{yes,yes_axes,yes_offset,[]},ViewGather.Viewing{no,no_axes,no_offset,[]},shape,workers,limits,default_capacity,CountCompletion.positive_elements(shape,shape_nonzero),shape_rank,
    TiledBroadcast.repeated(Shapes.same_extents(ms,shape),shape,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp}),
    Broadcast.repeated_contract(~Mask,~mask_value,Shapes.same_extents(ms,shape),mask,ms,shape,mask_axes,mask_offset,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp},{==},limits,default_capacity,
      mask_stretched,mask_nonzero,shape_nonzero,mask_rank,shape_rank,count,accepted,view_fits(ms,shape,mask_axes,mask_offset,mask_counted,mask_stretched,shape_nonzero),
      mask_contract),
    TiledBroadcast.repeated(Shapes.same_extents(ys,shape),shape,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp}),
    Broadcast.repeated_contract(~Yes,~yes_value,Shapes.same_extents(ys,shape),yes,ys,shape,yes_axes,yes_offset,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp},{==},limits,default_capacity,
      yes_stretched,yes_nonzero,shape_nonzero,yes_rank,shape_rank,count,accepted,view_fits(ys,shape,yes_axes,yes_offset,yes_counted,yes_stretched,shape_nonzero),
      yes_contract),
    TiledBroadcast.repeated(Shapes.same_extents(ns,shape),shape,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,nm),nc,nf,Tensors.Usable{},nw},nx,nl,np}),
    Broadcast.repeated_contract(~No,~no_value,Shapes.same_extents(ns,shape),no,ns,shape,no_axes,no_offset,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,nm),nc,nf,Tensors.Usable{},nw},nx,nl,np},{==},limits,default_capacity,
      no_stretched,no_nonzero,shape_nonzero,no_rank,shape_rank,count,accepted,view_fits(ns,shape,no_axes,no_offset,no_counted,no_stretched,shape_nonzero),
      no_contract))

# Equal extents are the same extents.
def extents_same(+source: List<&2,U32>,+shape: List<&2,U32>,+equal: {source == shape : List<&2,U32>}) -> {Shapes.same_extents(source,shape) == True{} : Bool}:
  %Equal.sym(List<&2,U32>,source,shape,equal) : {Shapes.same_extents(_,shape) == True{} : Bool}
  ShapeLaws.same_extents_reflexive(shape)

# Extents broadcast to themselves only.
def broadcast_own(+source: List<&2,U32>,+shape: List<&2,U32>,+common: {Shapes.broadcast(source,source) == Some{shape} : Maybe<&2,List<&2,U32>>}) -> {source == shape : List<&2,U32>}:
  BinaryValues.found_same(source,shape,Equal.trans(Maybe<&2,List<&2,U32>>,Some{source},Shapes.broadcast(source,source),Some{shape},
    Equal.sym(Maybe<&2,List<&2,U32>>,Shapes.broadcast(source,source),Some{source},UnitBroadcast.self_broadcast(source)),common))

# Operands of one shape keep their cells; operands of other extents are
# repeated to the common extents.
def broadcast_cases(~Mask: Data,~Yes: Data,~No: Data,~mask_value: Mask -> U32 -> F32,~yes_value: Yes -> U32 -> F32,~no_value: No -> U32 -> F32,same: Bool,+mask: Mask,+yes: Yes,+no: No,+ms: List<&2,U32>,+ys: List<&2,U32>,+ns: List<&2,U32>,+values: List<&2,U32>,+shape: List<&2,U32>,+workers: U32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +classified: {Shapes.same_extents(ms,ys) && Shapes.same_extents(ys,ns) == same : Bool},
  +values_common: {Shapes.broadcast(ys,ns) == Some{values} : Maybe<&2,List<&2,U32>>},+common: {Shapes.broadcast(ms,values) == Some{shape} : Maybe<&2,List<&2,U32>>},
  +mask_axes: List<&2,Views.Axis>,+mask_offset: U32,+yes_axes: List<&2,Views.Axis>,+yes_offset: U32,+no_axes: List<&2,Views.Axis>,+no_offset: U32,
  +mask_stretched: {Tensors.stretched(Tensors.view_of(ms),shape) == Some{Views.View{mask_axes,mask_offset}} : Maybe<&2,Views.View>},+yes_stretched: {Tensors.stretched(Tensors.view_of(ys),shape) == Some{Views.View{yes_axes,yes_offset}} : Maybe<&2,Views.View>},
  +no_stretched: {Tensors.stretched(Tensors.view_of(ns),shape) == Some{Views.View{no_axes,no_offset}} : Maybe<&2,Views.View>},
  +mask_nonzero: {Shapes.has_zero(ms) == False{} : Bool},+yes_nonzero: {Shapes.has_zero(ys) == False{} : Bool},+no_nonzero: {Shapes.has_zero(ns) == False{} : Bool},
  +shape_nonzero: {Shapes.has_zero(shape) == False{} : Bool},+mask_rank: {Nat.is_le(List.length(&2,U32,ms),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+yes_rank: {Nat.is_le(List.length(&2,U32,ys),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+no_rank: {Nat.is_le(List.length(&2,U32,ns),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+shape_rank: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +counted: Global.Counted<shape>,
  +mm: Model.Model<Tile.Tile8>,+mc: List<&2,U32>,+mf: Tile.Tile8,+mw: U32,+mx: Nat,+ml: U32,+mp: Grid.Position,
  +ym: Model.Model<Tile.Tile8>,+yc: List<&2,U32>,+yf: Tile.Tile8,+yw: U32,+yx: Nat,+yl: U32,+yp: Grid.Position,
  +nm: Model.Model<Tile.Tile8>,+nc: List<&2,U32>,+nf: Tile.Tile8,+nw: U32,+nx: Nat,+nl: U32,+np: Grid.Position,
  +mask_contract: Public.Contract(~Mask,~mask_value,mask,ms,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp}),
  +yes_contract: Public.Contract(~Yes,~yes_value,yes,ys,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp}),
  +no_contract: Public.Contract(~No,~no_value,no,ns,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,nm),nc,nf,Tensors.Usable{},nw},nx,nl,np})) ->
  Selected(~ViewGather.Viewing<Mask>,~ViewGather.Viewing<Yes>,~ViewGather.Viewing<No>,~ViewGather.viewed(~F32,~Mask,~mask_value),~ViewGather.viewed(~F32,~Yes,~yes_value),~ViewGather.viewed(~F32,~No,~no_value),ViewGather.Viewing{mask,mask_axes,mask_offset,[]},ViewGather.Viewing{yes,yes_axes,yes_offset,[]},ViewGather.Viewing{no,no_axes,no_offset,[]},shape,workers,Select.shapes(same,shape,Some{Global.count(shape,counted)},ms,ys,ns,workers,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp},Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp},Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,nm),nc,nf,Tensors.Usable{},nw},nx,nl,np})):
  match same:
    case False{}:
      repeated_select(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,ms,ys,ns,shape,workers,limits,default_capacity,mask_axes,mask_offset,yes_axes,yes_offset,no_axes,no_offset,mask_stretched,yes_stretched,no_stretched,mask_nonzero,yes_nonzero,no_nonzero,shape_nonzero,mask_rank,yes_rank,no_rank,shape_rank,counted,
        mm,mc,mf,mw,mx,ml,mp,ym,yc,yf,yw,yx,yl,yp,nm,nc,nf,nw,nx,nl,np,mask_contract,yes_contract,no_contract)
    case True{}:
      +mask_yes = ShapeLaws.same_extents(ms,ys,Boolean.and_true_left(Shapes.same_extents(ms,ys),Shapes.same_extents(ys,ns),classified))
      +yes_no = ShapeLaws.same_extents(ys,ns,Boolean.and_true_right(Shapes.same_extents(ms,ys),Shapes.same_extents(ys,ns),classified))
      +yes_values = broadcast_own(ys,values,
        %Equal.sym(List<&2,U32>,ys,ns,yes_no) : {Shapes.broadcast(ys,_) == Some{values} : Maybe<&2,List<&2,U32>>} values_common)
      +mask_values = Equal.trans(List<&2,U32>,ms,ys,values,mask_yes,yes_values)
      +values_shape = broadcast_own(values,shape,%mask_values : {Shapes.broadcast(_,values) == Some{shape} : Maybe<&2,List<&2,U32>>} common)
      +yes_shape = Equal.trans(List<&2,U32>,ys,values,shape,yes_values,values_shape)
      %extents_same(ms,shape,Equal.trans(List<&2,U32>,ms,ys,shape,mask_yes,yes_shape)) :
        Selected(~ViewGather.Viewing<Mask>,~ViewGather.Viewing<Yes>,~ViewGather.Viewing<No>,~ViewGather.viewed(~F32,~Mask,~mask_value),~ViewGather.viewed(~F32,~Yes,~yes_value),~ViewGather.viewed(~F32,~No,~no_value),ViewGather.Viewing{mask,mask_axes,mask_offset,[]},ViewGather.Viewing{yes,yes_axes,yes_offset,[]},ViewGather.Viewing{no,no_axes,no_offset,[]},shape,workers,Select.chosen(workers,TiledBroadcast.repeated(_,shape,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp}),Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp},Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,nm),nc,nf,Tensors.Usable{},nw},nx,nl,np}))
      %extents_same(ys,shape,yes_shape) :
        Selected(~ViewGather.Viewing<Mask>,~ViewGather.Viewing<Yes>,~ViewGather.Viewing<No>,~ViewGather.viewed(~F32,~Mask,~mask_value),~ViewGather.viewed(~F32,~Yes,~yes_value),~ViewGather.viewed(~F32,~No,~no_value),ViewGather.Viewing{mask,mask_axes,mask_offset,[]},ViewGather.Viewing{yes,yes_axes,yes_offset,[]},ViewGather.Viewing{no,no_axes,no_offset,[]},shape,workers,Select.chosen(workers,TiledBroadcast.repeated(Shapes.same_extents(ms,shape),shape,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp}),TiledBroadcast.repeated(_,shape,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp}),Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,nm),nc,nf,Tensors.Usable{},nw},nx,nl,np}))
      %extents_same(ns,shape,Equal.trans(List<&2,U32>,ns,ys,shape,Equal.sym(List<&2,U32>,ys,ns,yes_no),yes_shape)) :
        Selected(~ViewGather.Viewing<Mask>,~ViewGather.Viewing<Yes>,~ViewGather.Viewing<No>,~ViewGather.viewed(~F32,~Mask,~mask_value),~ViewGather.viewed(~F32,~Yes,~yes_value),~ViewGather.viewed(~F32,~No,~no_value),ViewGather.Viewing{mask,mask_axes,mask_offset,[]},ViewGather.Viewing{yes,yes_axes,yes_offset,[]},ViewGather.Viewing{no,no_axes,no_offset,[]},shape,workers,Select.chosen(workers,TiledBroadcast.repeated(Shapes.same_extents(ms,shape),shape,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,mm),mc,mf,Tensors.Usable{},mw},mx,ml,mp}),TiledBroadcast.repeated(Shapes.same_extents(ys,shape),shape,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,ym),yc,yf,Tensors.Usable{},yw},yx,yl,yp}),TiledBroadcast.repeated(_,shape,Tiled.Tiled{Tensors.Tensor{Model.pack(~Tile.Tile8,nm),nc,nf,Tensors.Usable{},nw},nx,nl,np})))
      repeated_select(~Mask,~Yes,~No,~mask_value,~yes_value,~no_value,mask,yes,no,ms,ys,ns,shape,workers,limits,default_capacity,mask_axes,mask_offset,yes_axes,yes_offset,no_axes,no_offset,mask_stretched,yes_stretched,no_stretched,mask_nonzero,yes_nonzero,no_nonzero,shape_nonzero,mask_rank,yes_rank,no_rank,shape_rank,counted,
        mm,mc,mf,mw,mx,ml,mp,ym,yc,yf,yw,yx,yl,yp,nm,nc,nf,nw,nx,nl,np,mask_contract,yes_contract,no_contract)
