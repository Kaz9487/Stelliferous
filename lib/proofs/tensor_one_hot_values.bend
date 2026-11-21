# Train.one_hot of class numbers [N] compares the labels as a column [N,1]
# with the places 0 to K - 1 as a row [1,K]: the result holds at every index
# of [N,K] the equality of the label and the place the two read through their
# views stretched to [N,K].
import Base
import ../tensor.bend as Tensors
import ../tensor_f32.bend as F
import ../kernel_tile.bend as Tile
import ../kernel_gemm.bend as KernelGemm
import ../tensor_loss.bend as Loss
import ../tensor_shape.bend as Shapes
import ../tensor_view.bend as Views
import ../tensor_grid.bend as Grid
import ../tensor_training.bend as Train
import ../tensor_tiled.bend as Tiled
import ../math_compare.bend as Compare
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./natural_addition.bend as Addition
import ./nat_order.bend as Order
import ./tensor_binary_values.bend as Binary
import ./tensor_pieces_map.bend as PiecesMap
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_global_shape_count.bend as Global
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_count_refinement.bend as Count
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_random_values.bend as RandomValues
import ./tensor_tiled_broadcast.bend as Broadcast
import ./tensor_tiled_public.bend as Public
import ./tensor_tiled_reshape.bend as Reshape
import ./tensor_tiled_padded_cells.bend as PaddedCells
import ./tensor_unit_broadcast.bend as UnitBroadcast

# The places 0 to K - 1.
def Places() -> Data:
  PiecesMap.Context<F32,F32>

def places() -> Places():
  PiecesMap.Context{0.0,0.0}

# The equality of the labels and the places, each read through its view
# stretched to [N,K].
def OneHot(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+images: U32,+classes: U32,+laxes: List<&2,Views.Axis>,+loffset: U32,
  +raxes: List<&2,Views.Axis>,+roffset: U32,result: Tiled.Tiled) -> Data:
  Broadcast.BroadcastContract(~Logical,~original,~Places(),~RandomValues.arange_value(),~Compare.equal_elements,logical,places(),[images,classes],laxes,loffset,raxes,roffset,result)

# ---- The extents ----

# The column of extent 1 meets the row of extent K at K.
def column(+classes: U32,+same: Bool,+equal: {U32.is_eq(1,classes) == same : Bool}) ->
  {Shapes.broadcast_extent(same,True{},U32.is_eq(classes,1),1,classes) == Some{classes} : Maybe<&2,U32>}:
  match same:
    case True{}: Equal.cong(U32,Maybe<&2,U32>,extent => Some{extent},1,classes,Arithmetic.equal_of_is_eq(1,classes,equal))
    case False{}: {==}

# [N,1] and [1,K] broadcast to [N,K].
def outer(+images: U32,+classes: U32) -> {Shapes.broadcast([images,1],[1,classes]) == Some{[images,classes]} : Maybe<&2,List<&2,U32>>}:
  %Equal.sym(Maybe<&2,U32>,Shapes.broadcast_extent(U32.is_eq(images,1),U32.is_eq(images,1),True{},images,1),Some{images},
    UnitBroadcast.singleton(images,U32.is_eq(images,1),U32.is_eq(images,1),{==},{==})) :
    {Shapes.prepend_extent(_,Shapes.prepend_extent(Shapes.broadcast_extent(U32.is_eq(1,classes),True{},U32.is_eq(classes,1),1,classes),Some{[]})) == Some{[images,classes]} : Maybe<&2,List<&2,U32>>}
  %Equal.sym(Maybe<&2,U32>,Shapes.broadcast_extent(U32.is_eq(1,classes),True{},U32.is_eq(classes,1),1,classes),Some{classes},column(classes,U32.is_eq(1,classes),{==})) :
    {Shapes.prepend_extent(Some{images},Shapes.prepend_extent(_,Some{[]})) == Some{[images,classes]} : Maybe<&2,List<&2,U32>>}
  {==}

# Nonzero extents give extents without a zero.
def single_nonzero(+extent: U32,+nonzero: {U32.is_zero(extent) == False{} : Bool}) -> {Shapes.has_zero([extent]) == False{} : Bool}:
  %Equal.sym(Bool,U32.is_zero(extent),False{},nonzero) : {_ || Shapes.has_zero([]) == False{} : Bool}
  {==}

def column_nonzero(+images: U32,+nonzero: {U32.is_zero(images) == False{} : Bool}) -> {Shapes.has_zero([images,1]) == False{} : Bool}:
  %Equal.sym(Bool,U32.is_zero(images),False{},nonzero) : {_ || Shapes.has_zero([1]) == False{} : Bool}
  {==}

def row_nonzero(+classes: U32,+nonzero: {U32.is_zero(classes) == False{} : Bool}) -> {Shapes.has_zero([1,classes]) == False{} : Bool}:
  %Equal.sym(Bool,U32.is_zero(classes),False{},nonzero) : {U32.is_zero(1) || (_ || Shapes.has_zero([])) == False{} : Bool}
  {==}

def grid_nonzero(+images: U32,+classes: U32,+images_nonzero: {U32.is_zero(images) == False{} : Bool},+classes_nonzero: {U32.is_zero(classes) == False{} : Bool}) ->
  {Shapes.has_zero([images,classes]) == False{} : Bool}:
  %Equal.sym(Bool,U32.is_zero(images),False{},images_nonzero) : {_ || (U32.is_zero(classes) || Shapes.has_zero([])) == False{} : Bool}
  %Equal.sym(Bool,U32.is_zero(classes),False{},classes_nonzero) : {False{} || (_ || Shapes.has_zero([])) == False{} : Bool}
  {==}

# A counted extent counts the same as a column or a row of it.
def column_count(+images: U32,+counted: Global.Counted<[images]>) -> {Tensors.count_of([images,1]) == Tensors.count_of([images]) : Maybe<&2,U32>}:
  Global.Counted{+count,+accepted} = counted
  Equal.trans(Maybe<&2,U32>,Tensors.count_of([images,1]),Some{count},Tensors.count_of([images]),
    CountCompletion.count_of([images,1],count,Global.whole_counted([images],count,accepted),
      PaddedCells.count_bounded(ShapeLaws.elements([images]),count,Count.count_of([images],count,accepted))),
    Equal.sym(Maybe<&2,U32>,Tensors.count_of([images]),Some{count},accepted))

def row_count(+classes: U32,+counted: Global.Counted<[classes]>) -> {Tensors.count_of([1,classes]) == Tensors.count_of([classes]) : Maybe<&2,U32>}:
  Global.Counted{+count,+accepted} = counted
  +elements = ShapeLaws.elements([classes])
  Equal.trans(Maybe<&2,U32>,Tensors.count_of([1,classes]),Some{count},Tensors.count_of([classes]),
    CountCompletion.count_of([1,classes],count,
      Equal.trans(Nat,Nat.add(elements,0n),elements,U32.to_nat(count),Addition.zero_right(elements),Global.whole_counted([classes],count,accepted)),
      PaddedCells.count_bounded(elements,count,Count.count_of([classes],count,accepted))),
    Equal.sym(Maybe<&2,U32>,Tensors.count_of([classes]),Some{count},accepted))

# ---- The comparison ----

# Packed values of the contract give their values back.
def restored_of(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+model: PiecesModel.Model<Tile.Tile8>,+cells: List<&2,U32>,
  +fill: Tile.Tile8,+workers: U32,+axis: Nat,+lanes: U32,+position: Grid.Position,
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,fill,Tensors.Usable{},workers},axis,lanes,position})) ->
  {Tiled.logical_shape(cells,axis,lanes,position) == values : List<&2,U32>}:
  KernelGemm.Both{global,Public.Laid{inside,shaped,+restored,counted,pinned}} = contract
  restored

# The packed labels as a column compared with the places as a row.
def compared(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+images: U32,+classes: U32,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,
  +left_stretched: {Tensors.stretched(Tensors.view_of([images,1]),[images,classes]) == Some{Views.View{laxes,loffset}} : Maybe<&2,Views.View>},
  +right_stretched: {Tensors.stretched(Tensors.view_of([1,classes]),[images,classes]) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},
  +images_nonzero: {U32.is_zero(images) == False{} : Bool},+classes_nonzero: {U32.is_zero(classes) == False{} : Bool},
  +rank_room: {Nat.is_le(2n,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+counted: Global.Counted<[images,classes]>,+places_counted: Global.Counted<[classes]>,
  +model: PiecesModel.Model<Tile.Tile8>,+cells: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+axis: Nat,+lanes: U32,+position: Grid.Position,
  +contract: Public.Contract(~Logical,~original,logical,[images],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,fill,Tensors.Usable{},workers},axis,lanes,position})) ->
  OneHot(~Logical,~original,logical,images,classes,laxes,loffset,raxes,roffset,
    F.equal(F.reshape(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,fill,Tensors.Usable{},workers},axis,lanes,position},[images,1]),
      F.reshape(F.arange(0.0,classes),[1,classes]))):
  Global.Counted{+places_count,+places_accepted} = places_counted
  KernelGemm.Both{global,Public.Laid{inside,shaped,restored,+labels_counted,pinned}} = contract
  +single = Order.transitive(1n,2n,U32.to_nat(Machine.capacity(limits)),{==},rank_room)
  Binary.broadcast(~Logical,~original,~Places(),~RandomValues.arange_value(),~Compare.equal_elements,logical,places(),"equal",1.375,limits,default_capacity,
    [images,1],[1,classes],[images,classes],laxes,loffset,raxes,roffset,outer(images,classes),left_stretched,right_stretched,
    column_nonzero(images,images_nonzero),row_nonzero(classes,classes_nonzero),grid_nonzero(images,classes,images_nonzero,classes_nonzero),rank_room,rank_room,rank_room,counted,
    F.reshape(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,fill,Tensors.Usable{},workers},axis,lanes,position},[images,1]),
    Reshape.reshaped(~Logical,~original,logical,limits,[images],[images,1],
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,fill,Tensors.Usable{},workers},axis,lanes,position},
      single_nonzero(images,images_nonzero),column_count(images,labels_counted),single,rank_room,default_capacity,contract),
    F.reshape(F.arange(0.0,classes),[1,classes]),
    Reshape.reshaped(~Places(),~RandomValues.arange_value(),places(),limits,[classes],[1,classes],F.arange(0.0,classes),
      single_nonzero(classes,classes_nonzero),row_count(classes,Global.Counted{places_count,places_accepted}),single,rank_room,default_capacity,
      RandomValues.arange(0.0,classes,places_count,places_accepted,single_nonzero(classes,classes_nonzero))))

def described(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+images: U32,+classes: U32,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,
  +left_stretched: {Tensors.stretched(Tensors.view_of([images,1]),[images,classes]) == Some{Views.View{laxes,loffset}} : Maybe<&2,Views.View>},
  +right_stretched: {Tensors.stretched(Tensors.view_of([1,classes]),[images,classes]) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},
  +images_nonzero: {U32.is_zero(images) == False{} : Bool},+classes_nonzero: {U32.is_zero(classes) == False{} : Bool},
  +rank_room: {Nat.is_le(2n,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+counted: Global.Counted<[images,classes]>,+places_counted: Global.Counted<[classes]>,
  -labels: Tiled.Tiled,+packed: Public.Packed<labels>,+contract: Public.Contract(~Logical,~original,logical,[images],labels)) ->
  OneHot(~Logical,~original,logical,images,classes,laxes,loffset,raxes,roffset,Train.one_hot(labels,classes)):
  Public.Packed{+model,+cells,+fill,+workers,+axis,+lanes,+position,+equation} = packed
  +moved = {%Equal.sym(Tiled.Tiled,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,fill,Tensors.Usable{},workers},axis,lanes,position},labels,equation) :
      Public.Contract(~Logical,~original,logical,[images],_) contract :
    Public.Contract(~Logical,~original,logical,[images],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,fill,Tensors.Usable{},workers},axis,lanes,position})}
  %equation : OneHot(~Logical,~original,logical,images,classes,laxes,loffset,raxes,roffset,Train.one_hot(_,classes))
  %Equal.sym(List<&2,U32>,Tiled.logical_shape(cells,axis,lanes,position),[images],restored_of(~Logical,~original,logical,[images],model,cells,fill,workers,axis,lanes,position,moved)) :
    OneHot(~Logical,~original,logical,images,classes,laxes,loffset,raxes,roffset,
      Loss.one_hot_rows(_,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,fill,Tensors.Usable{},workers},axis,lanes,position},classes))
  compared(~Logical,~original,logical,images,classes,limits,default_capacity,laxes,loffset,raxes,roffset,left_stretched,right_stretched,images_nonzero,classes_nonzero,
    rank_room,counted,places_counted,model,cells,fill,workers,axis,lanes,position,moved)

# one_hot of labels [N] of the contract holds at every index of [N,K] the
# comparison equal_elements of the label and the place U32.to_f32(j) + 0.0,
# each read through its view stretched to [N,K].
def one_hot(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+images: U32,+classes: U32,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,
  +left_stretched: {Tensors.stretched(Tensors.view_of([images,1]),[images,classes]) == Some{Views.View{laxes,loffset}} : Maybe<&2,Views.View>},
  +right_stretched: {Tensors.stretched(Tensors.view_of([1,classes]),[images,classes]) == Some{Views.View{raxes,roffset}} : Maybe<&2,Views.View>},
  +images_nonzero: {U32.is_zero(images) == False{} : Bool},+classes_nonzero: {U32.is_zero(classes) == False{} : Bool},
  +rank_room: {Nat.is_le(2n,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+counted: Global.Counted<[images,classes]>,+places_counted: Global.Counted<[classes]>,
  labels: Tiled.Tiled,+contract: Public.Contract(~Logical,~original,logical,[images],labels)) ->
  OneHot(~Logical,~original,logical,images,classes,laxes,loffset,raxes,roffset,Train.one_hot(labels,classes)):
  described(~Logical,~original,logical,images,classes,limits,default_capacity,laxes,loffset,raxes,roffset,left_stretched,right_stretched,images_nonzero,classes_nonzero,
    rank_room,counted,places_counted,labels,Public.packed(~Logical,~original,logical,[images],labels,contract),contract)
