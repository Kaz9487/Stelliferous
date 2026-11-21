# Three-input selection on the tiled forms: the three operands take one
# layout, the lanes on the axis of yes and the groups innermost, so their
# pieces zip leaf by leaf, and every piece selects tile by tile and sets the
# lanes past the true extent to +0.
import Base
import ./tensor.bend as Tensors
import ./tensor_grid.bend as Grid
import ./tensor_pieces.bend as Pieces
import ./tensor_shape.bend as Shapes
import ./tensor_lanes.bend as Lanes
import ./tensor_tasks.bend as Tasks
import ./tensor_operations.bend as Operations
import ./tensor_tiled.bend as Tiled
import ./tensor_tiled_map.bend as TiledMap
import ./tensor_tiled_broadcast.bend as TiledBroadcast
import ./kernel_tile.bend as Tile
import ./storage_buffer.bend as Storage
import ./traversal_partition.bend as Partition

def Buffer() -> Type:
  Storage.Buffer<Tile.Tile8>

def Triple() -> Type:
  Pieces.Paired<Buffer(),Pieces.Paired<Buffer(),Buffer()>>

def selected_piece(padding: TiledMap.Padding,first: U32,count: U32,input: Triple(),resource: Unit) -> Pieces.Pieces(~Tile.Tile8):
  match input:
    case Pieces.Both{yes,Pieces.Both{mask,no}}:
      +first = first
      +count = count
      +padding = padding
      Partition.Leaf{TiledMap.padded(padding,first,count,Operations.choose_kernel(~Tile.Tile8,~Lanes.select_tile,U32.to_nat(count),mask,yes,no)),first,count,Unit{}}
    case Pieces.Both{yes,Pieces.Only{mask,own_first,own_count}}: Partition.Leaf{yes,first,count,Unit{}}
    case Pieces.Only{yes,own_first,own_count}: Partition.Leaf{yes,own_first,own_count,Unit{}}

# Priced per cell: three tiles read and one written, and the word operations
# of eight lanes.
def selected_decided(decision: Tasks.Decision,+padding: TiledMap.Padding,+shape: List<&2,U32>,fill: Tile.Tile8,+workers: U32,triples: Partition.Tree<&1,Triple(),Unit>) ->
  Tensors.Tensor<Tile.Tile8>:
  Tasks.Decision{+parallel,+tasks} = decision
  Tensors.Tensor{Pieces.grouped(~Triple(),~Buffer(),~TiledMap.Padding,~selected_piece,tasks,parallel,padding,triples),shape,fill,Tensors.Usable{},workers}

# A failed operand fails the selection, which keeps the extents of yes.
def cells_status(status: Tensors.Status,+padding: TiledMap.Padding,+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,
  mask: Pieces.Pieces(~Tile.Tile8),yes: Pieces.Pieces(~Tile.Tile8),no: Pieces.Pieces(~Tile.Tile8)) -> Tensors.Tensor<Tile.Tile8>:
  match status:
    case Tensors.Usable{}:
      selected_decided(Tasks.decide(Tensors.grid_of(shape),workers,4.0,7.0,Tasks.no_extra()),padding,shape,fill,workers,
        Pieces.zip(~Buffer(),~Pieces.Paired<Buffer(),Buffer()>,yes,Pieces.zip(~Buffer(),~Buffer(),mask,no)))
    case Tensors.Failed{problem}: Tensors.Tensor{yes,shape,fill,Tensors.Failed{problem},workers}

def cells(+workers: U32,+padding: TiledMap.Padding,mask: Tensors.Tensor<Tile.Tile8>,yes: Tensors.Tensor<Tile.Tile8>,no: Tensors.Tensor<Tile.Tile8>) -> Tensors.Tensor<Tile.Tile8>:
  Tensors.Tensor{mask,mask_shape,mask_fill,mask_status,mask_workers} = mask
  Tensors.Tensor{yes,+shape,+fill,yes_status,yes_workers} = yes
  Tensors.Tensor{no,no_shape,no_fill,no_status,no_workers} = no
  cells_status(Tensors.first_failure(mask_status,Tensors.first_failure(yes_status,no_status)),padding,shape,fill,workers,mask,yes,no)

def tiles(+workers: U32,mask: Tiled.Tiled,yes: Tiled.Tiled & TiledMap.Padding,no: Tiled.Tiled) -> Tiled.Tiled:
  Tiled.Tiled{mask,mask_axis,mask_lanes,mask_position} = mask
  (Tiled.Tiled{yes,+axis,+lanes,position},+padding) = yes
  Tiled.Tiled{no,no_axis,no_lanes,no_position} = no
  Tiled.Tiled{cells(workers,padding,mask,yes,no),axis,lanes,Grid.Innermost{}}

def innermost(tiled: Tiled.Tiled,+axis: Nat) -> Tiled.Tiled:
  Tiled.arranged(tiled,axis,Grid.Innermost{})

# The operands of one shape select with their lanes on the axis of yes.
def chosen_along(+workers: U32,mask: Tiled.Tiled,yes: Tiled.Tiled & Nat,no: Tiled.Tiled) -> Tiled.Tiled:
  (yes,+axis) = yes
  tiles(workers,innermost(mask,axis),TiledMap.padding_cells(innermost(yes,axis)),innermost(no,axis))

def chosen(+workers: U32,mask: Tiled.Tiled,yes: Tiled.Tiled,no: Tiled.Tiled) -> Tiled.Tiled:
  chosen_along(workers,mask,Tiled.lane_axis_of(yes),no)

def shapes(same: Bool,+shape: List<&2,U32>,count: Maybe<&2,U32>,mask_shape: List<&2,U32>,yes_shape: List<&2,U32>,no_shape: List<&2,U32>,workers: U32,
  mask: Tiled.Tiled,yes: Tiled.Tiled,no: Tiled.Tiled) -> Tiled.Tiled:
  match same count:
    case True{} Some{count}: chosen(workers,mask,yes,no)
    case False{} Some{count}:
      chosen(workers,TiledBroadcast.repeated(Shapes.same_extents(mask_shape,shape),shape,mask),
        TiledBroadcast.repeated(Shapes.same_extents(yes_shape,shape),shape,yes),TiledBroadcast.repeated(Shapes.same_extents(no_shape,shape),shape,no))
    case same None{}: Tiled.failed("select",Tensors.InvalidShape{shape},shape,workers)

def masked_shape(common: Maybe<&2,List<&2,U32>>,+mask_shape: List<&2,U32>,+yes_shape: List<&2,U32>,+no_shape: List<&2,U32>,values_shape: List<&2,U32>,workers: U32,
  mask: Tiled.Tiled,yes: Tiled.Tiled,no: Tiled.Tiled) -> Tiled.Tiled:
  match common:
    case Some{+shape}:
      shapes(Shapes.same_extents(mask_shape,yes_shape) && Shapes.same_extents(yes_shape,no_shape),shape,Tensors.count_of(shape),mask_shape,yes_shape,no_shape,workers,mask,yes,no)
    case None{}: Tiled.failed("select",Tensors.ShapeMismatch{mask_shape,values_shape},mask_shape,workers)

def values_shape(common: Maybe<&2,List<&2,U32>>,+mask_shape: List<&2,U32>,+yes_shape: List<&2,U32>,no_shape: List<&2,U32>,workers: U32,
  mask: Tiled.Tiled,yes: Tiled.Tiled,no: Tiled.Tiled) -> Tiled.Tiled:
  match common:
    case Some{+shape}: masked_shape(Shapes.broadcast(mask_shape,shape),mask_shape,yes_shape,no_shape,shape,workers,mask,yes,no)
    case None{}: Tiled.failed("select",Tensors.ShapeMismatch{yes_shape,no_shape},yes_shape,workers)

def status(status: Tensors.Status,mask_shape: List<&2,U32>,+yes_shape: List<&2,U32>,+no_shape: List<&2,U32>,workers: U32,
  mask: Tiled.Tiled,yes: Tiled.Tiled,no: Tiled.Tiled) -> Tiled.Tiled:
  match status:
    case Tensors.Failed{problem}: Tiled.with_status(Tensors.Failed{problem},mask)
    case Tensors.Usable{}: values_shape(Shapes.broadcast(yes_shape,no_shape),mask_shape,yes_shape,no_shape,workers,mask,yes,no)

def select(mask: Tiled.Tiled,yes: Tiled.Tiled,no: Tiled.Tiled) -> Tiled.Tiled:
  Tiled.with_metadata(~Tiled.Tiled,mask,mask => mask_shape => mt => mw =>
    Tiled.with_metadata(~Tiled.Tiled,yes,yes => yes_shape => yt => yw =>
      Tiled.with_metadata(~Tiled.Tiled,no,no => no_shape => nt => nw =>
        status(Tensors.first_failure(mt,Tensors.first_failure(yt,nt)),mask_shape,yes_shape,no_shape,U32.max(mw,U32.max(yw,nw)),mask,yes,no))))
