# A tiled tensor keeps eight values of one axis of its logical shape, its lane
# axis, in one Tile8, lanes past the true extent holding +0. Its cells are a
# tensor of tiles over the cell axes of the logical shape (Grid.cell_shape):
# the other axes in order with the groups of the lane axis innermost, or
# outermost for an operand a kernel streams by group. A lane axis past the
# end of the shape is the unit axis: one value a cell, the cells in the
# logical shape.
import Base
import ./kernel_tile.bend as Tile
import ./math_choice.bend as Choice
import ./storage_buffer.bend as Storage
import ./tensor.bend as Tensors
import ./tensor_grid.bend as Grid
import ./tensor_pieces.bend as Pieces
import ./tensor_tasks.bend as Tasks
import ./tensor_move.bend as Move
import ./traversal_loop.bend as Loop
import ./tensor_view.bend as Views
import ./tensor_tiled_cells.bend as Cells
import ./traversal_partition.bend as Partition
import ./tensor_shape.bend as Shapes
import ./proofs/storage_certificate.bend as Certificate
import ./proofs/traversal_certificate.bend as TraversalCertificate
import ./proofs/tensor_tiled_certificate.bend as TiledCertificate

type Tiled is Type:
  Tiled{cells: Tensors.Tensor<Tile.Tile8>,axis: Nat,lanes: U32,position: Grid.Position}

# The axes of a shape around one axis: the product of those before it, its
# extent and the axes after it. The axis just past the last is a unit axis,
# so a tensor of rank 0 is one lane of one cell.
type Around is Data:
  Around{outer: U32,extent: U32,after: List<&2,U32>}

def around(shape: List<&2,U32>,axis: Nat,+outer: U32) -> Maybe<&2,Around>:
  match shape axis:
    case Nil{} 0n: Some{Around{outer,1,Nil{}}}
    case Nil{} 1n+rest: None{}
    case Con{extent,after} 0n: Some{Around{outer,extent,after}}
    case Con{extent,after} 1n+rest: around(after,rest,(outer * extent : U32))

def product(extents: List<&2,U32>) -> U32:
  Grid.product(extents)

def groups(lanes: U32) -> U32:
  Grid.groups(lanes)

# The lane axis of a tensor: the channels of [N,]C,H,W, otherwise the last
# axis, and the unit axis of rank 0.
def lane_axis(rank: Nat) -> Nat:
  match rank:
    case 0n: 0n
    case 1n: 0n
    case 2n: 1n
    case 3n+rest: rest

# The lane axis of a shape: that of its rank, or the last axis when that one
# holds a single element.
def lane_axis_for(+shape: List<&2,U32>) -> Nat:
  +rank = List.length(&2,U32,shape)
  +axis = lane_axis(rank)
  Bool.pick(Nat,U32.is_eq(Tensors.known_count(List.get(&2,U32,shape,axis)),1) && Nat.is_lt(1n,rank),Nat.sub(rank,1n),axis)

def zero_tile() -> Tile.Tile8:
  Tile.filled(0.0)

# The cell axes of a logical shape with its lanes on axis, the groups at
# position.
def cell_shape(+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position) -> List<&2,U32>:
  Grid.cell_shape(shape,Grid.Layout{axis,position})

def logical_shape(+cells: List<&2,U32>,+axis: Nat,+lanes: U32,+position: Grid.Position) -> List<&2,U32>:
  Grid.logical_shape(cells,Grid.Layout{axis,position},lanes)

def cell_axis_at(position: Grid.Position,+axis: Nat,+lane: Nat,+rank: Nat) -> Nat:
  match position:
    case Grid.Innermost{}: Bool.pick(Nat,Nat.is_eq(axis,lane),Nat.sub(rank,1n),Bool.pick(Nat,Nat.is_lt(axis,lane),axis,Nat.sub(axis,1n)))
    case Grid.Outermost{}: Bool.pick(Nat,Nat.is_eq(axis,lane),0n,Bool.pick(Nat,Nat.is_lt(axis,lane),1n+axis,axis))

# The cell axis of an axis of a logical shape of rank rank with its lanes on
# lane, the groups at position: the groups for the lane axis itself. Lanes on
# the unit axis leave the axes as they are.
def cell_axis(position: Grid.Position,+axis: Nat,+lane: Nat,+rank: Nat) -> Nat:
  Bool.pick(Nat,Nat.is_lt(lane,rank),cell_axis_at(position,axis,lane,rank),axis)

# Values per axis of a logical shape as values per cell axis: the lane axis
# drops out and the groups take zero.
def cell_values(+values: List<&2,U32>,+lane: Nat,+position: Grid.Position) -> List<&2,U32>:
  Grid.placed(Nat.is_lt(lane,List.length(&2,U32,values)),position,Grid.removed(values,lane),0)

# ---- Values in and out ----

# What every piece of a conversion reads: outer indices of the axes but the
# lane axis, after of them after it, the lanes and groups of the lane axis,
# and where the groups lie.
type Tiling is Data:
  Tiling{outer: U32,after: U32,lanes: U32,groups: U32,position: Grid.Position}

def tiled_pair(input_length: Nat,output_length: Nat,result: Array<F32> & Array<Tile.Tile8>,
  certificates: TraversalCertificate.Certificates<Array<F32>,Tile.Tile8,Certificate.family(~F32),result>) -> Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>:
  (input,output) = result
  TraversalCertificate.Certificates{input_certificate,output_certificate} = certificates
  (Storage.Buffer{input,input_length,input_certificate},Storage.Buffer{output,output_length,output_certificate})

def tiled_inner(+first: U32,+count: U32,+after: U32,+lanes: U32,+groups: U32,input: Storage.Buffer<F32>,output: Storage.Buffer<Tile.Tile8>) ->
  Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>:
  Storage.Buffer{input,+input_length,input_certificate} = input
  Storage.Buffer{output,+output_length,output_certificate} = output
  +blocks = (count / U32.max(groups,1) : U32)
  +run = {Cells.Tiling{after,lanes,groups,(first / U32.max(groups,1) : U32)} : Cells.Tiling}
  tiled_pair(input_length,output_length,Cells.tiled(blocks,run,input,output),TiledCertificate.tiled_certificates(blocks,run,input,output,input_certificate,output_certificate))

def tiled_outer(+first: U32,+count: U32,+outer: U32,+after: U32,+lanes: U32,input: Storage.Buffer<F32>,output: Storage.Buffer<Tile.Tile8>) ->
  Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>:
  Storage.Buffer{input,+input_length,input_certificate} = input
  Storage.Buffer{output,+output_length,output_certificate} = output
  +run = {Cells.Outer{outer,after,lanes,first} : Cells.Outer}
  tiled_pair(input_length,output_length,Cells.tiled_outermost(count,run,input,output),TiledCertificate.outermost_certificates(count,run,input,output,input_certificate,output_certificate))

def tiled_into(position: Grid.Position,+first: U32,+count: U32,+outer: U32,+after: U32,+lanes: U32,+groups: U32,input: Storage.Buffer<F32>,output: Storage.Buffer<Tile.Tile8>) ->
  Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>:
  match position:
    case Grid.Innermost{}: tiled_inner(first,count,after,lanes,groups,input,output)
    case Grid.Outermost{}: tiled_outer(first,count,outer,after,lanes,input,output)

def piece_leaf(+first: U32,+count: U32,pair: Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<F32> & Pieces.Pieces(~Tile.Tile8):
  (input,output) = pair
  (input,Partition.Leaf{output,first,count,Unit{}})

def tiled_piece(context: Tiling,first: U32,count: U32,input: Storage.Buffer<F32>) -> Storage.Buffer<F32> & Pieces.Pieces(~Tile.Tile8):
  Tiling{+outer,+after,+lanes,+groups,position} = context
  +first = first
  +count = count
  piece_leaf(first,count,tiled_into(position,first,count,outer,after,lanes,groups,input,Tensors.buffer(~Tile.Tile8,zero_tile(),count)))

# The cells of a logical shape from its values in row-major order, each piece
# from the one buffer of the values.
def tiled_from(+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+workers: U32,axes: Around,input: Storage.Buffer<F32>) -> Tiled:
  Around{+before,+lanes,after} = axes
  +cells = cell_shape(shape,axis,position)
  +inner = product(after)
  Tiled{Tensors.Tensor{Pair.snd(Storage.Buffer<F32>,Pieces.Pieces(~Tile.Tile8),
      Pieces.build(~Storage.Buffer<Tile.Tile8>,~Storage.Buffer<F32>,~Tiling,~tiled_piece,Pieces.ranges_of_grid(Tensors.grid_of(cells)),
        Tiling{(before * inner : U32),inner,lanes,groups(lanes),position},input)),
    cells,zero_tile(),Tensors.Usable{},workers},axis,lanes,position}

# A failed tensor keeps its problem in the cells it would have had.
def failed_cells(problem: Tensors.Problem,+shape: List<&2,U32>,+axis: Nat,+lanes: U32,+position: Grid.Position,+workers: U32) -> Tiled:
  Tiled{Tensors.Tensor{Partition.Leaf{Tensors.buffer(~Tile.Tile8,zero_tile(),0),0,0,Unit{}},cell_shape(shape,axis,position),zero_tile(),Tensors.Failed{problem},workers},axis,lanes,position}

def tiled_around(axes: Maybe<&2,Around>,status: Tensors.Status,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+workers: U32,input: Storage.Buffer<F32>) -> Tiled:
  match axes status:
    case Some{axes} Tensors.Usable{}: tiled_from(shape,axis,position,workers,axes,input)
    case Some{Around{before,extent,after}} Tensors.Failed{problem}: failed_cells(problem,shape,axis,extent,position,workers)
    case None{} status: Tiled{Tensors.failed(~Tile.Tile8,"tiled",Tensors.InvalidShape{shape},shape,zero_tile(),workers),axis,0,position}

# The tiled form of row-major values of a logical shape: the lanes on axis,
# the groups at position. The values must fill the shape.
def from_buffer(+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,status: Tensors.Status,+workers: U32,input: Storage.Buffer<F32>) -> Tiled:
  tiled_around(around(shape,axis,1),status,shape,axis,position,workers,input)

def tiled_shape(status: Tensors.Status,+shape: List<&2,U32>,workers: U32,input: Storage.Buffer<F32>) -> Tiled:
  from_buffer(shape,lane_axis_for(shape),Grid.Innermost{},status,workers,input)

# The tiled form of a tensor with its lanes on axis, the groups at position.
def tiled_at(tensor: Tensors.Tensor<F32>,+axis: Nat,+position: Grid.Position) -> Tiled:
  Tensors.with_whole(~F32,~Tiled,tensor,storage => shape => fill => status => workers =>
    +shape = shape
    from_buffer(shape,axis,position,status,workers,storage))

# The tiled form of a tensor along its lane axis.
def tiled(tensor: Tensors.Tensor<F32>) -> Tiled:
  Tensors.with_whole(~F32,~Tiled,tensor,storage => shape => fill => status => workers =>
    +shape = shape
    tiled_shape(status,shape,workers,storage))

# A value of a cell written when its lane holds one.
def lane_written(wanted: Bool,+index: U32,value: F32,output: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  match wanted:
    case True{}: Storage.write(~F32,index,value,output)
    case False{}: output

# The lanes of a cell below valid written from base on, step apart.
def cell_untiled(+base: U32,+step: U32,+valid: U32,tile: Tile.Tile8,output: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  Tile.Tile{a,b,c,d,e,f,g,h} = tile
  lane_written(U32.is_lt(7,valid),(base + 7 * step : U32),h,lane_written(U32.is_lt(6,valid),(base + 6 * step : U32),g,
    lane_written(U32.is_lt(5,valid),(base + 5 * step : U32),f,lane_written(U32.is_lt(4,valid),(base + 4 * step : U32),e,
      lane_written(U32.is_lt(3,valid),(base + 3 * step : U32),d,lane_written(U32.is_lt(2,valid),(base + 2 * step : U32),c,
        lane_written(U32.is_lt(1,valid),(base + step : U32),b,lane_written(U32.is_lt(0,valid),base,a,output))))))))

# The first value and the values of cell index: outer index o and group g
# hold values (o / after * lanes + 8 * g + j) * after + o % after.
def written_cell(+context: Cells.Cells,+cell: U32,tile: Tile.Tile8,output: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  Cells.Cells{+outer,+after,+lanes,+groups,position} = context
  +group = Bool.pick(U32,Grid.is_outermost(position),(cell / U32.max(outer,1) : U32),(cell % U32.max(groups,1) : U32))
  +index = Bool.pick(U32,Grid.is_outermost(position),(cell % U32.max(outer,1) : U32),(cell / U32.max(groups,1) : U32))
  +inner = U32.max(after,1)
  cell_untiled(((index / inner * lanes + group * 8) * inner + index % inner : U32),inner,(lanes - group * 8 : U32),tile,output)

# Untiling a piece: the cells read from index on, their values written.
type Untiling is Type:
  Untiling{input: Storage.Buffer<Tile.Tile8>,output: Storage.Buffer<F32>,index: U32,first: U32,context: Cells.Cells}

def untiling_read(+index: U32,+first: U32,+context: Cells.Cells,read: Storage.Buffer<Tile.Tile8> & Tile.Tile8,output: Storage.Buffer<F32>) -> Untiling:
  (input,tile) = read
  Untiling{input,written_cell(context,(first + index : U32),tile,output),(index + 1 : U32),first,context}

# The cell at index written, then the next.
def untiling_step(state: Untiling) -> Untiling:
  Untiling{input,output,+index,+first,+context} = state
  untiling_read(index,first,context,Storage.read(~Tile.Tile8,input,index),output)

def untiled_output(state: Untiling) -> Storage.Buffer<F32>:
  Untiling{input,output,index,first,context} = state
  output

# The values of the cells of a piece, written in place.
def untiled_piece(context: Cells.Cells,first: U32,count: U32,input: Storage.Buffer<Tile.Tile8>,output: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  untiled_output(Loop.run(~Untiling,~untiling_step,U32.to_nat(count),Untiling{input,output,0,first,context}))

def values_around(axes: Maybe<&2,Around>,+lanes: U32,+position: Grid.Position,pieces: Pieces.Pieces(~Tile.Tile8)) -> Storage.Buffer<F32>:
  match axes:
    case Some{Around{+before,extent,after}}:
      +inner = product(after)
      Pieces.fold(~Storage.Buffer<Tile.Tile8>,~Storage.Buffer<F32>,~Cells.Cells,~untiled_piece,Cells.Cells{(before * inner : U32),inner,lanes,groups(lanes),position},pieces,
        Tensors.buffer(~F32,0.0,(before * lanes * inner : U32)))
    case None{}: Tensors.buffer(~F32,0.0,0)

def values_status(status: Tensors.Status,+shape: List<&2,U32>,+axis: Nat,+lanes: U32,+position: Grid.Position,pieces: Pieces.Pieces(~Tile.Tile8)) -> Storage.Buffer<F32>:
  match status:
    case Tensors.Usable{}: values_around(around(shape,axis,1),lanes,position,pieces)
    case Tensors.Failed{problem}: Tensors.buffer(~F32,0.0,0)

# The values of a tiled tensor in row-major order of its logical shape, one
# buffer, with the shape, status and worker count.
def with_values(~Result: Type,tiled: Tiled,body: Storage.Buffer<F32> -> List<&2,U32> -> Tensors.Status -> U32 -> Result) -> Result:
  Tiled{Tensors.Tensor{pieces,+cells,fill,+status,+workers},+axis,+lanes,+position} = tiled
  +shape = logical_shape(cells,axis,lanes,position)
  body(values_status(status,shape,axis,lanes,position,pieces),shape,status,workers)

# The values whose tiled form the cells are.
def untiled(tiled: Tiled) -> Tensors.Tensor<F32>:
  with_values(~Tensors.Tensor<F32>,tiled,values => shape => status => workers =>
    +shape = shape
    Tensors.values(~F32,values,shape,0.0,status,workers))

# ---- The public tensor ----

def shape(tiled: Tiled) -> Tiled & List<&2,U32>:
  Tiled{Tensors.Tensor{pieces,+cells,fill,status,workers},+axis,+lanes,+position} = tiled
  (Tiled{Tensors.Tensor{pieces,cells,fill,status,workers},axis,lanes,position},logical_shape(cells,axis,lanes,position))

def checked(+axis: Nat,+lanes: U32,+position: Grid.Position,result: Result<&1,&1,Tensors.Tensor<Tile.Tile8> & Tensors.Problem,Tensors.Tensor<Tile.Tile8>>) ->
  Result<&1,&1,Tiled & Tensors.Problem,Tiled>:
  match result:
    case Done{cells}: Done{Tiled{cells,axis,lanes,position}}
    case Fail{(cells,problem)}: Fail{(Tiled{cells,axis,lanes,position},problem)}

def check(tiled: Tiled) -> Result<&1,&1,Tiled & Tensors.Problem,Tiled>:
  Tiled{cells,+axis,+lanes,+position} = tiled
  checked(axis,lanes,position,Tensors.check(~Tile.Tile8,cells))

def counted(+axis: Nat,+lanes: U32,+position: Grid.Position,pair: Tensors.Tensor<Tile.Tile8> & U32) -> Tiled & U32:
  (cells,workers) = pair
  (Tiled{cells,axis,lanes,position},workers)

def workers(tiled: Tiled) -> Tiled & U32:
  Tiled{cells,+axis,+lanes,+position} = tiled
  counted(axis,lanes,position,Tensors.workers(~Tile.Tile8,cells))

def with_workers(tiled: Tiled,+count: U32) -> Tiled:
  Tiled{cells,axis,lanes,position} = tiled
  Tiled{Tensors.with_workers(~Tile.Tile8,cells,count),axis,lanes,position}

# A usable tensor fails with an invalid argument of the operation; a failed
# one keeps its problem.
def rejected(name: String,tiled: Tiled) -> Tiled:
  Tiled{cells,+axis,+lanes,+position} = tiled
  match cells:
    case Tensors.Tensor{pieces,+cells_shape,fill,Tensors.Usable{},workers}:
      Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Failed{Tensors.Problem{name,Tensors.InvalidArgument{logical_shape(cells_shape,axis,lanes,position)}}},workers},axis,lanes,position}
    case Tensors.Tensor{pieces,cells_shape,fill,Tensors.Failed{problem},workers}: Tiled{Tensors.Tensor{pieces,cells_shape,fill,Tensors.Failed{problem},workers},axis,lanes,position}

# The cells of a shape with a zero extent: no storage, the lanes on the lane
# axis of its rank.
def empty_around(axes: Maybe<&2,Around>,+shape: List<&2,U32>,+axis: Nat,+workers: U32) -> Tiled:
  match axes:
    case Some{Around{before,+extent,after}}:
      Tiled{Tensors.Tensor{Partition.Leaf{Tensors.buffer(~Tile.Tile8,zero_tile(),0),0,0,Unit{}},cell_shape(shape,axis,Grid.Innermost{}),zero_tile(),Tensors.Usable{},workers},axis,extent,Grid.Innermost{}}
    case None{}: Tiled{Tensors.failed(~Tile.Tile8,"empty",Tensors.InvalidShape{shape},shape,zero_tile(),workers),axis,0,Grid.Innermost{}}

def empty(+shape: List<&2,U32>,+workers: U32) -> Tiled:
  +axis = lane_axis(List.length(&2,U32,shape))
  empty_around(around(shape,axis,1),shape,axis,workers)

# A tensor of +0: zero tiles in the cells of the shape, the lanes on the lane
# axis of the shape. The count of the shape must fit the machine.
def zeros_counted(count: Maybe<&2,U32>,vacant: Bool,axes: Maybe<&2,Around>,+shape: List<&2,U32>,+axis: Nat) -> Tiled:
  match count vacant axes:
    case Some{count} True{} axes: empty(shape,1)
    case Some{count} False{} Some{Around{before,+extent,after}}:
      Tiled{Tensors.full(~Tile.Tile8,cell_shape(shape,axis,Grid.Innermost{}),zero_tile()),axis,extent,Grid.Innermost{}}
    case count vacant axes: Tiled{Tensors.failed(~Tile.Tile8,"full",Tensors.InvalidShape{shape},shape,zero_tile(),1),axis,0,Grid.Innermost{}}

def zeros(+shape: List<&2,U32>) -> Tiled:
  +axis = lane_axis_for(shape)
  zeros_counted(Tensors.count_of(shape),Shapes.has_zero(shape),around(shape,axis,1),shape,axis)

# A tensor of a shape that failed in an operation: failed cells of the shape,
# one lane a cell on the unit axis.
def failed(operation: String,reason: Tensors.Reason,+shape: List<&2,U32>,workers: U32) -> Tiled:
  Tiled{Tensors.failed(~Tile.Tile8,operation,reason,shape,zero_tile(),workers),List.length(&2,U32,shape),1,Grid.Innermost{}}

# Failed cells of a shape with the problem of another tensor.
def failed_as(problem: Tensors.Problem,+shape: List<&2,U32>,+workers: U32) -> Tiled:
  Tensors.Problem{operation,reason} = problem
  failed(operation,reason,shape,workers)

def copied(+axis: Nat,+lanes: U32,+position: Grid.Position,pair: Tensors.Tensor<Tile.Tile8> & Tensors.Tensor<Tile.Tile8>) -> Tiled & Tiled:
  (first,second) = pair
  (Tiled{first,axis,lanes,position},Tiled{second,axis,lanes,position})

# The only operation that keeps its argument: (original, copy).
def copy(tiled: Tiled) -> Tiled & Tiled:
  Tiled{cells,+axis,+lanes,+position} = tiled
  copied(axis,lanes,position,Tensors.copied(~Tile.Tile8,32.0,cells))

# A relabel of the cells that reached its result, or the tensor it starts
# from when it cannot.
type Attempt<-Start: Type> is Type:
  Reached{result: Tiled}
  Untried{start: Start}

# ---- The lanes on another axis ----

# A move of the lanes from axis source to axis target of a logical shape:
# whether each lies inside the shape, where the groups lie on each side, the
# lanes of the target axis and the target cell axes.
type Relane is Data:
  Relane{source: Nat,source_within: Bool,source_position: Grid.Position,target: Nat,target_within: Bool,target_position: Grid.Position,lanes: U32,cells: List<&2,U32>}

# The source cell axis of a logical axis other than the source lane axis.
def source_axis(+source: Nat,+source_within: Bool,+source_position: Grid.Position,+axis: Nat) -> U32:
  +inner = Bool.pick(Nat,source_within && Nat.is_lt(source,axis),Nat.sub(axis,1n),axis)
  U32.from_nat(Bool.pick(Nat,source_within && Grid.is_outermost(source_position),1n+inner,inner))

# A relane moves blocks: eight indices 8k + i of the source lane axis and
# eight indices 8g + j of the target lane axis, where target cell i holds in
# lane j what source cell j holds in lane i. What a target cell axis is to
# the blocks: an axis both sides share, the target groups, or the source lane
# axis in blocks of eight. source is its source cell stride (per index of the
# target lane axis for the groups, per block for the lane axis) and lanes the
# extent of the lane axis it carries.
type Role is Data:
  Shared{}
  Groups{}
  Lanes{}

type Blocked is Data:
  Blocked{role: Role,source: U32,lanes: U32}

# What a target cell axis is: the cells of a logical axis, or the groups of
# the lane axis. A type of its own: lists of it are shared, and a list of
# Maybe would share every Maybe, and with it every View in a Maybe.
type CellAxis is Data:
  OfAxis{axis: Nat}
  OfGroups{}

def blocked_axis(axis: CellAxis,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,+strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32) -> Blocked:
  match axis:
    case OfGroups{}: Blocked{Groups{},Tasks.nth(strides,source_axis(source,source_within,source_position,target)),Grid.lanes_at(shape,target)}
    case OfAxis{+logical}:
      Bool.pick(Blocked,source_within && Nat.is_eq(logical,source),Blocked{Lanes{},Tasks.nth(strides,groups_axis),Grid.lanes_at(shape,source)},
        Blocked{Shared{},Tasks.nth(strides,source_axis(source,source_within,source_position,logical)),1})

def blocked_list(axes: List<&2,CellAxis>,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,+strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32) -> List<&2,Blocked>:
  match axes:
    case Nil{}: []
    case Con{axis,rest}:
      blocked_axis(axis,source,source_within,source_position,target,strides,shape,groups_axis) <> blocked_list(rest,source,source_within,source_position,target,strides,shape,groups_axis)

def blocked_axes(+walk: Relane,+strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32,axes: List<&2,CellAxis>) -> List<&2,Blocked>:
  Relane{+source,+source_within,+source_position,+target,target_within,target_position,lanes,cells} = walk
  blocked_list(axes,source,source_within,source_position,target,strides,shape,groups_axis)

# A relane and what every target cell axis is to its blocks.
type Relaning is Data:
  Relaning{walk: Relane,axes: List<&2,Blocked>}

# One axis of the blocks of a target piece: the first index of the piece on
# it, the indices it holds, the blocks it counts and the blocks of the axes
# after it, and its stride in the piece.
type Ranged is Data:
  Ranged{blocked: Blocked,low: U32,extent: U32,count: U32,inner: U32,target: U32}

def block_count(role: Role,+low: U32,+extent: U32) -> U32:
  match role:
    case Lanes{}: ((low + extent - 1) / 8 - low / 8 + 1 : U32)
    case role: extent

def blocked_role(blocked: Blocked) -> Role:
  Blocked{role,source,lanes} = blocked
  role

def ranged_count(ranged: Ranged) -> U32:
  Ranged{blocked,low,extent,+count,+inner,target} = ranged
  (count * inner : U32)

def ranged_head_count(rest: List<&2,Ranged>) -> U32:
  match rest:
    case Nil{}: 1
    case Con{+head,tail}: ranged_count(head)

def ranged_prepended(+blocked: Blocked,+low: U32,+extent: U32,+target: U32,+rest: List<&2,Ranged>) -> List<&2,Ranged>:
  Ranged{blocked,low,extent,block_count(blocked_role(blocked),low,extent),ranged_head_count(rest),target} <> rest

def ranged_axes(axes: List<&2,Blocked>,lows: List<&2,U32>,extents: List<&2,U32>,targets: List<&2,U32>) -> List<&2,Ranged>:
  match axes lows extents targets:
    case Con{+blocked,axes} Con{+low,lows} Con{+extent,extents} Con{+target,targets}: ranged_prepended(blocked,low,extent,target,ranged_axes(axes,lows,extents,targets))
    case axes lows extents targets: []

# One block of a target piece: its first source cell, the first target cell
# it writes in the piece and the step between its target cells, the cells of
# the block it writes from low to high, and the source cells it reads, step
# apart.
type Block is Data:
  Block{source: U32,target: U32,stride: U32,low: U32,high: U32,reads: U32,step: U32}

def block_axis(role: Role,block: Block,+index: U32,+low: U32,+extent: U32,+source: U32,+lanes: U32,+target: U32) -> Block:
  match role block:
    case Shared{} Block{+block_source,+block_target,+stride,+block_low,+block_high,+reads,+step}:
      Block{(block_source + (low + index) * source : U32),(block_target + index * target : U32),stride,block_low,block_high,reads,step}
    case Groups{} Block{+block_source,+block_target,+stride,+block_low,+block_high,+reads,+step}:
      +group = (low + index : U32)
      Block{(block_source + group * 8 * source : U32),(block_target + index * target : U32),stride,block_low,block_high,U32.min(8,(lanes - group * 8 : U32)),source}
    case Lanes{} Block{+block_source,+block_target,+stride,+block_low,+block_high,+reads,+step}:
      +lane_group = (low / 8 + index : U32)
      +first = (lane_group * 8 : U32)
      +start = U32.max(low,first)
      Block{(block_source + lane_group * source : U32),(block_target + (start - low) * target : U32),target,(start - first : U32),
        U32.min((low + extent - first : U32),8),reads,step}

def block_at(axes: List<&2,Ranged>,+index: U32,block: Block) -> Block:
  match axes:
    case Nil{}: block
    case Con{Ranged{Blocked{role,+source,+lanes},+low,+extent,+count,+inner,+target},rest}:
      block_at(rest,index,block_axis(role,block,(index / U32.max(inner,1) % U32.max(count,1) : U32),low,extent,source,lanes,target))

type Eight is Data:
  Eight{c0: Tile.Tile8,c1: Tile.Tile8,c2: Tile.Tile8,c3: Tile.Tile8,c4: Tile.Tile8,c5: Tile.Tile8,c6: Tile.Tile8,c7: Tile.Tile8}

# Cell i of the result holds lane i of every cell j in lane j.
def transposed(eight: Eight) -> Eight:
  Eight{Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7},Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7},Tile.Tile{c0,c1,c2,c3,c4,c5,c6,c7},Tile.Tile{d0,d1,d2,d3,d4,d5,d6,d7},
    Tile.Tile{e0,e1,e2,e3,e4,e5,e6,e7},Tile.Tile{f0,f1,f2,f3,f4,f5,f6,f7},Tile.Tile{g0,g1,g2,g3,g4,g5,g6,g7},Tile.Tile{h0,h1,h2,h3,h4,h5,h6,h7}} = eight
  Eight{Tile.Tile{a0,b0,c0,d0,e0,f0,g0,h0},Tile.Tile{a1,b1,c1,d1,e1,f1,g1,h1},Tile.Tile{a2,b2,c2,d2,e2,f2,g2,h2},Tile.Tile{a3,b3,c3,d3,e3,f3,g3,h3},
    Tile.Tile{a4,b4,c4,d4,e4,f4,g4,h4},Tile.Tile{a5,b5,c5,d5,e5,f5,g5,h5},Tile.Tile{a6,b6,c6,d6,e6,f6,g6,h6},Tile.Tile{a7,b7,c7,d7,e7,f7,g7,h7}}

# Cell j of a block read, or +0 past the cells it reads.
def block_read(~Reader: Type,~read: Reader -> U32 -> Reader & Tile.Tile8,wanted: Bool,+source: U32,+step: U32,+index: U32,reader: Reader) -> Reader & Tile.Tile8:
  match wanted:
    case True{}: read(reader,(source + index * step : U32))
    case False{}: (reader,zero_tile())

def read_next(~Reader: Type,~read: Reader -> U32 -> Reader & Tile.Tile8,+reads: U32,+source: U32,+step: U32,+index: U32,reader: Reader) -> Reader & Tile.Tile8:
  block_read(~Reader,~read,U32.is_lt(index,reads),source,step,index,reader)

# The eight cells of a block, cell j read from source + j * step: each read
# hands the reader to the next.
def after_eighth(~Reader: Type,+c0: Tile.Tile8,+c1: Tile.Tile8,+c2: Tile.Tile8,+c3: Tile.Tile8,+c4: Tile.Tile8,+c5: Tile.Tile8,+c6: Tile.Tile8,pair: Reader & Tile.Tile8) -> Reader & Eight:
  (reader,c7) = pair
  (reader,Eight{c0,c1,c2,c3,c4,c5,c6,c7})

def after_seventh(~Reader: Type,~read: Reader -> U32 -> Reader & Tile.Tile8,+reads: U32,+source: U32,+step: U32,+c0: Tile.Tile8,+c1: Tile.Tile8,+c2: Tile.Tile8,+c3: Tile.Tile8,+c4: Tile.Tile8,+c5: Tile.Tile8,
  pair: Reader & Tile.Tile8) -> Reader & Eight:
  (reader,c6) = pair
  after_eighth(~Reader,c0,c1,c2,c3,c4,c5,c6,read_next(~Reader,~read,reads,source,step,7,reader))

def after_sixth(~Reader: Type,~read: Reader -> U32 -> Reader & Tile.Tile8,+reads: U32,+source: U32,+step: U32,+c0: Tile.Tile8,+c1: Tile.Tile8,+c2: Tile.Tile8,+c3: Tile.Tile8,+c4: Tile.Tile8,pair: Reader & Tile.Tile8) -> Reader & Eight:
  (reader,c5) = pair
  after_seventh(~Reader,~read,reads,source,step,c0,c1,c2,c3,c4,c5,read_next(~Reader,~read,reads,source,step,6,reader))

def after_fifth(~Reader: Type,~read: Reader -> U32 -> Reader & Tile.Tile8,+reads: U32,+source: U32,+step: U32,+c0: Tile.Tile8,+c1: Tile.Tile8,+c2: Tile.Tile8,+c3: Tile.Tile8,pair: Reader & Tile.Tile8) -> Reader & Eight:
  (reader,c4) = pair
  after_sixth(~Reader,~read,reads,source,step,c0,c1,c2,c3,c4,read_next(~Reader,~read,reads,source,step,5,reader))

def after_fourth(~Reader: Type,~read: Reader -> U32 -> Reader & Tile.Tile8,+reads: U32,+source: U32,+step: U32,+c0: Tile.Tile8,+c1: Tile.Tile8,+c2: Tile.Tile8,pair: Reader & Tile.Tile8) -> Reader & Eight:
  (reader,c3) = pair
  after_fifth(~Reader,~read,reads,source,step,c0,c1,c2,c3,read_next(~Reader,~read,reads,source,step,4,reader))

def after_third(~Reader: Type,~read: Reader -> U32 -> Reader & Tile.Tile8,+reads: U32,+source: U32,+step: U32,+c0: Tile.Tile8,+c1: Tile.Tile8,pair: Reader & Tile.Tile8) -> Reader & Eight:
  (reader,c2) = pair
  after_fourth(~Reader,~read,reads,source,step,c0,c1,c2,read_next(~Reader,~read,reads,source,step,3,reader))

def after_second(~Reader: Type,~read: Reader -> U32 -> Reader & Tile.Tile8,+reads: U32,+source: U32,+step: U32,+c0: Tile.Tile8,pair: Reader & Tile.Tile8) -> Reader & Eight:
  (reader,c1) = pair
  after_third(~Reader,~read,reads,source,step,c0,c1,read_next(~Reader,~read,reads,source,step,2,reader))

def after_first(~Reader: Type,~read: Reader -> U32 -> Reader & Tile.Tile8,+reads: U32,+source: U32,+step: U32,pair: Reader & Tile.Tile8) -> Reader & Eight:
  (reader,c0) = pair
  after_second(~Reader,~read,reads,source,step,c0,read_next(~Reader,~read,reads,source,step,1,reader))

def eight_read(~Reader: Type,~read: Reader -> U32 -> Reader & Tile.Tile8,+reads: U32,+source: U32,+step: U32,reader: Reader) -> Reader & Eight:
  after_first(~Reader,~read,reads,source,step,read_next(~Reader,~read,reads,source,step,0,reader))

# A cell of the source tensor through the holding, +0 where no held piece
# holds it, and a cell of one piece.
def held_cell(holding: Move.Holding<Tile.Tile8>,index: U32) -> Move.Holding<Tile.Tile8> & Tile.Tile8:
  Move.read(~Tile.Tile8,holding,index,zero_tile())

def piece_cell(buffer: Storage.Buffer<Tile.Tile8>,index: U32) -> Storage.Buffer<Tile.Tile8> & Tile.Tile8:
  Storage.read(~Tile.Tile8,buffer,index)

def piece_returned(behind: Move.Stack<Tile.Tile8>,+first: U32,+count: U32,below: Move.Stack<Tile.Tile8>,read: Storage.Buffer<Tile.Tile8> & Eight) ->
  Move.Holding<Tile.Tile8> & Eight:
  (buffer,eight) = read
  (Move.Holding{behind,Move.Held{buffer,first,count,below}},eight)

# A block whose cells one step of a run would take from the current piece
# reads them from it.
def piece_block(inside: Bool,+reads: U32,+source: U32,+step: U32,behind: Move.Stack<Tile.Tile8>,buffer: Storage.Buffer<Tile.Tile8>,+first: U32,+count: U32,
  below: Move.Stack<Tile.Tile8>) -> Move.Holding<Tile.Tile8> & Eight:
  match inside:
    case True{}:
      piece_returned(behind,first,count,below,eight_read(~Storage.Buffer<Tile.Tile8>,~piece_cell,reads,(source - first : U32),step,buffer))
    case False{}: eight_read(~Move.Holding<Tile.Tile8>,~held_cell,reads,source,step,Move.Holding{behind,Move.Held{buffer,first,count,below}})

def current_block(+reads: U32,+source: U32,+step: U32,holding: Move.Holding<Tile.Tile8>) -> Move.Holding<Tile.Tile8> & Eight:
  match holding:
    case Move.Holding{behind,Move.Held{buffer,+first,+count,below}}:
      piece_block(U32.is_le(reads,Move.inside(reads,source,step,False{},first,count)),reads,source,step,behind,buffer,first,count,below)
    case Move.Holding{behind,Move.Bottom{}}: eight_read(~Move.Holding<Tile.Tile8>,~held_cell,reads,source,step,Move.Holding{behind,Move.Bottom{}})

# The eight cells of a block from the piece of its first cell.
def block_reads(+reads: U32,+source: U32,+step: U32,holding: Move.Holding<Tile.Tile8>) -> Move.Holding<Tile.Tile8> & Eight:
  current_block(reads,source,step,Move.resolve(~Tile.Tile8,holding,source))

def cell_written(wanted: Bool,+target: U32,cell: Tile.Tile8,output: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  match wanted:
    case True{}: Storage.write(~Tile.Tile8,target,cell,output)
    case False{}: output

# Cells low to high of a block written, cell i at target + (i - low) * stride.
def block_writes(+low: U32,+high: U32,+target: U32,+stride: U32,eight: Eight,output: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  Eight{c0,c1,c2,c3,c4,c5,c6,c7} = eight
  +start = (target - low * stride : U32)
  cell_written(U32.is_le(low,7) && U32.is_lt(7,high),(start + 7 * stride : U32),c7,
    cell_written(U32.is_le(low,6) && U32.is_lt(6,high),(start + 6 * stride : U32),c6,
      cell_written(U32.is_le(low,5) && U32.is_lt(5,high),(start + 5 * stride : U32),c5,
        cell_written(U32.is_le(low,4) && U32.is_lt(4,high),(start + 4 * stride : U32),c4,
          cell_written(U32.is_le(low,3) && U32.is_lt(3,high),(start + 3 * stride : U32),c3,
            cell_written(U32.is_le(low,2) && U32.is_lt(2,high),(start + 2 * stride : U32),c2,
              cell_written(U32.is_le(low,1) && U32.is_lt(1,high),(start + stride : U32),c1,
                cell_written(U32.is_eq(low,0) && U32.is_lt(0,high),start,c0,output))))))))

def block_moved(+block: Block,read: Move.Holding<Tile.Tile8> & Eight,output: Storage.Buffer<Tile.Tile8>) -> Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  Block{source,+target,+stride,+low,+high,reads,step} = block
  (holding,eight) = read
  (holding,block_writes(low,high,target,stride,transposed(eight),output))

def block_written(+block: Block,holding: Move.Holding<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>) -> Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  Block{+source,target,stride,low,high,+reads,+step} = block
  block_moved(block,block_reads(reads,source,step,holding),output)

# Blocks from index on of a target piece.
def blocks(fuel: Nat,+index: U32,+axes: List<&2,Ranged>,pair: Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) -> Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  match fuel:
    case 0n: pair
    case 1n+rest:
      (holding,output) = pair
      blocks(rest,(index + 1 : U32),axes,block_written(block_at(axes,index,Block{0,0,0,0,1,1,0}),holding,output))

def relaned_piece(+axes: List<&2,Ranged>,+count: U32,holding: Move.Holding<Tile.Tile8>) -> Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  blocks(U32.to_nat(ranged_head_count(axes)),0,axes,(holding,Tensors.buffer(~Tile.Tile8,zero_tile(),count)))

def relane_written(rest: Tensors.Holdings<Tile.Tile8>,pair: Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) -> Tensors.Holdings<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  (holding,output) = pair
  (Tensors.Holds{holding,rest},output)

def relane_held(+axes: List<&2,Blocked>,+cells: List<&2,U32>,+first: U32,+count: U32,+fill: Tile.Tile8,input: Tensors.Holdings<Tile.Tile8>) ->
  Tensors.Holdings<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  match input:
    case Tensors.Holds{holding,rest}:
      +extents = Grid.box_of(Tensors.grid_of(cells),cells,count)
      relane_written(rest,relaned_piece(ranged_axes(axes,Grid.coordinates(cells,first),extents,Tasks.strides_of(extents)),count,holding))
    case Tensors.Unheld{}: (Tensors.Unheld{},Tensors.buffer(~Tile.Tile8,fill,count))

# A target piece written block by block from the source cells its task holds.
def relane_gather(+context: Relaning,+components: Tasks.Components,+first: U32,+count: U32,+fill: Tile.Tile8,input: Tensors.Holdings<Tile.Tile8>) ->
  Tensors.Holdings<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  Relaning{walk,+axes} = context
  Relane{source,source_within,source_position,target,target_within,target_position,lanes,+cells} = walk
  relane_held(axes,cells,first,count,fill,input)

def read_at(+grid: Grid.Grid,+axis: U32,+extent: U32,+step: U32,+width: U32,+divisor: U32) -> Tasks.Source:
  Tasks.Read{axis,Grid.block(grid,axis,extent),step,0,False{},width,divisor,0,extent,False{}}

# How a target cell axis of logical axis axis reads the source cells: the
# source lane axis reads the source groups, every other axis its own axis.
def logical_reading(+source: Nat,+source_within: Bool,+source_position: Grid.Position,+source_cells: List<&2,U32>,+source_grid: Grid.Grid,+axis: Nat,+extent: U32) -> Tasks.Source:
  +groups_axis = Bool.pick(U32,Grid.is_outermost(source_position),0,(U32.from_nat(List.length(&2,U32,source_cells)) - 1 : U32))
  Bool.pick(Tasks.Source,source_within && Nat.is_eq(axis,source),
    read_at(source_grid,groups_axis,Tasks.nth(source_cells,groups_axis),1,1,8),
    read_at(source_grid,source_axis(source,source_within,source_position,axis),extent,1,1,1))

# The axes of a shape from index on but skipped.
def axes_from(remaining: Nat,+index: Nat,+skipped: Nat) -> List<&2,CellAxis>:
  match remaining:
    case 0n: []
    case 1n+rest:
      +later = axes_from(rest,1n+index,skipped)
      Bool.pick(List<&2,CellAxis>,Nat.is_eq(index,skipped),later,OfAxis{index} <> later)

# The logical axis of every target cell axis; the target groups have none.
def target_axes(within: Bool,position: Grid.Position,+axis: Nat,+rank: Nat) -> List<&2,CellAxis>:
  match within position:
    case False{} position: axes_from(rank,0n,rank)
    case True{} Grid.Innermost{}: List.append(&2,CellAxis,axes_from(rank,0n,axis),[OfGroups{}])
    case True{} Grid.Outermost{}: OfGroups{} <> axes_from(rank,0n,axis)

def relane_readings(+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,+source_cells: List<&2,U32>,+source_grid: Grid.Grid,+target_grid: Grid.Grid,
  +shape: List<&2,U32>,axes: List<&2,CellAxis>,extents: List<&2,U32>,+index: U32) -> List<&2,Tasks.Reading>:
  match axes extents:
    case Con{OfGroups{},axes} Con{+extent,extents}:
      Tasks.Reading{extent,Grid.block(target_grid,index,extent),
        read_at(source_grid,source_axis(source,source_within,source_position,target),Grid.lanes_at(shape,target),8,8,1)} <>
        relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,axes,extents,(index + 1 : U32))
    case Con{OfAxis{+axis},axes} Con{+extent,extents}:
      Tasks.Reading{extent,Grid.block(target_grid,index,extent),logical_reading(source,source_within,source_position,source_cells,source_grid,axis,Grid.lanes_at(shape,axis))} <>
        relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,axes,extents,(index + 1 : U32))
    case axes extents: []

def relane_cells(+walk: Relane,+shape: List<&2,U32>,+source_cells: List<&2,U32>,+workers: U32,pieces: Pieces.Pieces(~Tile.Tile8)) -> Pieces.Pieces(~Tile.Tile8):
  Relane{+source,+source_within,+source_position,+target,+target_within,+target_position,lanes,+cells} = walk
  +source_grid = Tensors.grid_of(source_cells)
  +axes = target_axes(target_within,target_position,target,List.length(&2,U32,shape))
  +groups_axis = Bool.pick(U32,Grid.is_outermost(source_position),0,(U32.from_nat(List.length(&2,U32,source_cells)) - 1 : U32))
  +walk = {Relane{source,source_within,source_position,target,target_within,target_position,lanes,cells} : Relane}
  Tensors.moved_with(~Tile.Tile8,~Relaning,~relane_gather,
    Tasks.components_of([relane_readings(source,source_within,source_position,target,source_cells,source_grid,Tensors.grid_of(cells),shape,axes,cells,0)]),
    Relaning{walk,blocked_axes(walk,Tasks.strides_of(source_cells),shape,groups_axis,axes)},9.0,16.0,Tensors.Source{pieces,Tensors.Ended{}},
    [Tensors.origin(source_cells,0)],cells,zero_tile(),workers)

def walk_cells(walk: Relane) -> List<&2,U32>:
  Relane{source,source_within,source_position,target,target_within,target_position,lanes,cells} = walk
  cells

def relane_status(status: Tensors.Status,+walk: Relane,+shape: List<&2,U32>,+source_cells: List<&2,U32>,+workers: U32,pieces: Pieces.Pieces(~Tile.Tile8)) -> Tensors.Tensor<Tile.Tile8>:
  match status:
    case Tensors.Usable{}: Tensors.Tensor{relane_cells(walk,shape,source_cells,workers,pieces),walk_cells(walk),zero_tile(),Tensors.Usable{},workers}
    case Tensors.Failed{problem}: Tensors.Tensor{Partition.Leaf{Tensors.buffer(~Tile.Tile8,zero_tile(),0),0,0,Unit{}},walk_cells(walk),zero_tile(),Tensors.Failed{problem},workers}

# The tiled form with its lanes moved from their axis to another, the groups
# keeping their position. The cell at index x of the old lane axis and group
# g of the new one holds, in lane l, lane x % 8 of the cell of group x / 8 of
# the old axis and index 8 * g + l of the new one; lanes past the extent of
# the new axis are +0.
def relane(tiled: Tiled,+axis: Nat,+target_position: Grid.Position) -> Tiled:
  Tiled{Tensors.Tensor{pieces,+source_cells,fill,status,+workers},+old,+lanes,+position} = tiled
  +shape = logical_shape(source_cells,old,lanes,position)
  +rank = List.length(&2,U32,shape)
  +within = Nat.is_lt(axis,rank)
  +new_lanes = Grid.lanes_at(shape,axis)
  Tiled{relane_status(status,Relane{old,Nat.is_lt(old,rank),position,axis,within,target_position,new_lanes,cell_shape(shape,axis,target_position)},shape,source_cells,workers,pieces),
    axis,new_lanes,target_position}

# ---- Lane splits ----

# Lanes shift to shift + 7 of a cell followed by the next.
def joined_1(low: Tile.Tile8,high: Tile.Tile8) -> Tile.Tile8:
  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
  Tile.Tile{a1,a2,a3,a4,a5,a6,a7,b0}

def joined_2(low: Tile.Tile8,high: Tile.Tile8) -> Tile.Tile8:
  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
  Tile.Tile{a2,a3,a4,a5,a6,a7,b0,b1}

def joined_3(low: Tile.Tile8,high: Tile.Tile8) -> Tile.Tile8:
  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
  Tile.Tile{a3,a4,a5,a6,a7,b0,b1,b2}

def joined_4(low: Tile.Tile8,high: Tile.Tile8) -> Tile.Tile8:
  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
  Tile.Tile{a4,a5,a6,a7,b0,b1,b2,b3}

def joined_5(low: Tile.Tile8,high: Tile.Tile8) -> Tile.Tile8:
  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
  Tile.Tile{a5,a6,a7,b0,b1,b2,b3,b4}

def joined_6(low: Tile.Tile8,high: Tile.Tile8) -> Tile.Tile8:
  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
  Tile.Tile{a6,a7,b0,b1,b2,b3,b4,b5}

def joined_7(low: Tile.Tile8,high: Tile.Tile8) -> Tile.Tile8:
  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = low
  Tile.Tile{b0,b1,b2,b3,b4,b5,b6,b7} = high
  Tile.Tile{a7,b0,b1,b2,b3,b4,b5,b6}

def joined(+shift: U32,low: Tile.Tile8,high: Tile.Tile8) -> Tile.Tile8:
  match shift:
    case 0: low
    case 1: joined_1(low,high)
    case 2: joined_2(low,high)
    case 3: joined_3(low,high)
    case 4: joined_4(low,high)
    case 5: joined_5(low,high)
    case 6: joined_6(low,high)
    case _: joined_7(low,high)

# A lane axis of extent anchors * width split into an axis of anchors and the
# next of width, which takes the lanes, the groups keeping their position.
# The cells, in order, are the outer elements before the lane axis, the
# anchors, the after elements after it and the groups of the new lanes: cell
# (o, a, r, j) holds in lane l the element of index a * width + 8 * j + l of
# the old lane axis, +0 past width.
type Split is Data:
  Split{outer: U32,anchors: U32,width: U32,after: U32,groups: U32,source_groups: U32,position: Grid.Position}

# A target cell: its anchor, its group, the source cell of group 0 at its
# outer and after indices and the step between source groups.
type Spot is Data:
  Spot{anchor: U32,group: U32,base: U32,step: U32}

def split_spot(position: Grid.Position,+outer: U32,+anchors: U32,+after: U32,+groups: U32,+source_groups: U32,+index: U32) -> Spot:
  match position:
    case Grid.Innermost{}:
      +rest = (index / U32.max(groups,1) : U32)
      +inner = (rest / U32.max(after,1) : U32)
      Spot{(inner % U32.max(anchors,1) : U32),(index % U32.max(groups,1) : U32),((inner / U32.max(anchors,1) * after + rest % U32.max(after,1)) * source_groups : U32),1}
    case Grid.Outermost{}:
      +rest = (index / U32.max(after,1) : U32)
      +inner = (rest / U32.max(anchors,1) : U32)
      Spot{(rest % U32.max(anchors,1) : U32),(inner / U32.max(outer,1) : U32),(inner % U32.max(outer,1) * after + index % U32.max(after,1) : U32),(outer * after : U32)}

def split_high(+shift: U32,+first: U32,+width: U32,low: Tile.Tile8,pair: Move.Holding<Tile.Tile8> & Tile.Tile8) -> Move.Holding<Tile.Tile8> & Tile.Tile8:
  (holding,high) = pair
  (holding,Cells.tail(Cells.Tail{first,width},joined(shift,low,high)))

def split_low(+shift: U32,+first: U32,+width: U32,+source: U32,+step: U32,further: Bool,pair: Move.Holding<Tile.Tile8> & Tile.Tile8) -> Move.Holding<Tile.Tile8> & Tile.Tile8:
  (holding,low) = pair
  split_high(shift,first,width,low,block_read(~Move.Holding<Tile.Tile8>,~held_cell,further,source,step,1,holding))

# A target cell from the source cell of its first lane and, when its lanes
# start inside that cell and another follows, the next one.
def split_spotted(+width: U32,+source_groups: U32,spot: Spot,holding: Move.Holding<Tile.Tile8>) -> Move.Holding<Tile.Tile8> & Tile.Tile8:
  Spot{+anchor,+group,+base,+step} = spot
  +first = (anchor * width + group * 8 : U32)
  +cell = (first / 8 : U32)
  +shift = (first % 8 : U32)
  +source = (base + cell * step : U32)
  split_low(shift,(group * 8 : U32),width,source,step,U32.is_lt(0,shift) && U32.is_lt((cell + 1 : U32),source_groups),held_cell(holding,source))

def split_cell(+split: Split,+index: U32,holding: Move.Holding<Tile.Tile8>) -> Move.Holding<Tile.Tile8> & Tile.Tile8:
  Split{+outer,+anchors,+width,+after,+groups,+source_groups,+position} = split
  split_spotted(width,source_groups,split_spot(position,outer,anchors,after,groups,source_groups,index),holding)

def split_written(+index: U32,output: Storage.Buffer<Tile.Tile8>,read: Move.Holding<Tile.Tile8> & Tile.Tile8) -> Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  (holding,cell) = read
  (holding,Storage.write(~Tile.Tile8,index,cell,output))

def split_cells(fuel: Nat,+split: Split,+first: U32,+index: U32,pair: Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) -> Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  match fuel:
    case 0n: pair
    case 1n+rest:
      (holding,output) = pair
      split_cells(rest,split,first,(index + 1 : U32),split_written(index,output,split_cell(split,(first + index : U32),holding)))

# With the groups innermost the cells of one outer, anchor and after index are
# a run: consecutive target cells that read consecutive source cells, every
# one shifted by the same lanes. A run starts at target cell index and stops
# at the end of its groups or at end; its source cells start at source, the
# cell of group group of the old lanes.
type Run is Data:
  Run{source: U32,shift: U32,group: U32,count: U32,cell: U32}

def run_spotted(+width: U32,+groups: U32,+index: U32,+end: U32,spot: Spot) -> Run:
  Spot{+anchor,+group,+base,step} = spot
  +first = (anchor * width + group * 8 : U32)
  +cell = (first / 8 : U32)
  Run{(base + cell : U32),(first % 8 : U32),group,U32.min((groups - group : U32),(end - index : U32)),cell}

def split_run(+split: Split,+index: U32,+end: U32) -> Run:
  Split{+outer,+anchors,+width,+after,+groups,+source_groups,+position} = split
  run_spotted(width,groups,index,end,split_spot(position,outer,anchors,after,groups,source_groups,index))

# The lanes of a target cell past width are +0.
def run_masked(partial: Bool,+first: U32,+width: U32,cell: Tile.Tile8) -> Tile.Tile8:
  match partial:
    case True{}: Cells.tail(Cells.Tail{first,width},cell)
    case False{}: cell

def run_high(+shift: U32,+first: U32,+width: U32,low: Tile.Tile8,read: Storage.Buffer<Tile.Tile8> & Tile.Tile8) -> Storage.Buffer<Tile.Tile8> & Tile.Tile8:
  (input,high) = read
  (input,run_masked(U32.is_lt(width,(first + 8 : U32)),first,width,joined(shift,low,high)))

def run_low(+shift: U32,+first: U32,+width: U32,+local: U32,further: Bool,read: Storage.Buffer<Tile.Tile8> & Tile.Tile8) -> Storage.Buffer<Tile.Tile8> & Tile.Tile8:
  (input,low) = read
  run_high(shift,first,width,low,block_read(~Storage.Buffer<Tile.Tile8>,~piece_cell,further,(local + 1 : U32),0,0,input))

def run_written(+target: U32,output: Storage.Buffer<Tile.Tile8>,read: Storage.Buffer<Tile.Tile8> & Tile.Tile8) -> Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  (input,cell) = read
  (input,Storage.write(~Tile.Tile8,target,cell,output))

# The cells of a run whose source cells one piece holds, local from its first:
# the low cell and, when the lanes shift and another source cell follows, the
# next.
def run_cells(fuel: Nat,+width: U32,+source_groups: U32,+shift: U32,+local: U32,+cell: U32,+group: U32,+target: U32,
  pair: Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  match fuel:
    case 0n: pair
    case 1n+rest:
      (input,output) = pair
      run_cells(rest,width,source_groups,shift,(local + 1 : U32),(cell + 1 : U32),(group + 1 : U32),(target + 1 : U32),
        run_written(target,output,run_low(shift,(group * 8 : U32),width,local,U32.is_lt(0,shift) && U32.is_lt((cell + 1 : U32),source_groups),piece_cell(input,local))))

def run_held(+split: Split,+run: Run,inside: Bool,+first: U32,+local: U32,behind: Move.Stack<Tile.Tile8>,buffer: Storage.Buffer<Tile.Tile8>,+start: U32,+span: U32,
  below: Move.Stack<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>) -> Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  Split{outer,anchors,+width,after,groups,+source_groups,position} = split
  Run{+source,+shift,+group,+count,+cell} = run
  match inside:
    case True{}: Move.piece_returned(~Tile.Tile8,behind,start,span,below,run_cells(U32.to_nat(count),width,source_groups,shift,(source - start : U32),cell,group,local,(buffer,output)))
    case False{}: split_cells(U32.to_nat(count),split,first,local,(Move.Holding{behind,Move.Held{buffer,start,span,below}},output))

# A run from the piece that holds its first source cell when it holds them
# all, otherwise cell by cell.
def run_resolved(+split: Split,+run: Run,+first: U32,+local: U32,holding: Move.Holding<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>) ->
  Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  Split{outer,anchors,width,after,groups,+source_groups,position} = split
  Run{+source,+shift,group,+count,+cell} = run
  match holding:
    case Move.Holding{behind,Move.Held{buffer,+start,+span,below}}:
      +reads = (count + Bool.pick(U32,U32.is_lt(0,shift) && U32.is_lt((cell + count : U32),source_groups),1,0) : U32)
      run_held(split,run,U32.is_le(start,source) && U32.is_le((source + reads : U32),(start + span : U32)),first,local,behind,buffer,start,span,below,output)
    case Move.Holding{behind,Move.Bottom{}}: split_cells(U32.to_nat(count),split,first,local,(Move.Holding{behind,Move.Bottom{}},output))

def run_applied(+split: Split,+run: Run,+first: U32,+local: U32,pair: Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) ->
  Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  Run{+source,shift,group,count,cell} = run
  (holding,output) = pair
  run_resolved(split,run,first,local,Move.resolve(~Tile.Tile8,holding,source),output)

def run_count(run: Run) -> U32:
  Run{source,shift,group,count,cell} = run
  count

# The runs of a target piece from local cell local on, while more.
def split_runs(fuel: Nat,more: Bool,+split: Split,+first: U32,+local: U32,+count: U32,pair: Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) ->
  Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  match fuel more:
    case 0n more: pair
    case 1n+rest False{}: pair
    case 1n+rest True{}:
      +run = split_run(split,(first + local : U32),(first + count : U32))
      +next = (local + U32.max(run_count(run),1) : U32)
      split_runs(rest,U32.is_lt(next,count),split,first,next,count,run_applied(split,run,first,local,pair))

# The cells of a target piece: by runs with the groups innermost, otherwise
# cell by cell.
def split_written_piece(innermost: Bool,+split: Split,+first: U32,+count: U32,pair: Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) ->
  Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  match innermost:
    case True{}: split_runs(U32.to_nat(count),U32.is_lt(0,count),split,first,0,count,pair)
    case False{}: split_cells(U32.to_nat(count),split,first,0,pair)

def split_position(split: Split) -> Grid.Position:
  Split{outer,anchors,width,after,groups,source_groups,position} = split
  position

# A target piece written from the source cells its task holds.
def split_gather(+split: Split,+components: Tasks.Components,+first: U32,+count: U32,+fill: Tile.Tile8,input: Tensors.Holdings<Tile.Tile8>) ->
  Tensors.Holdings<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  match input:
    case Tensors.Holds{holding,rest}:
      relane_written(rest,split_written_piece(Bool.not(Grid.is_outermost(split_position(split))),split,first,count,(holding,Tensors.buffer(~Tile.Tile8,zero_tile(),count))))
    case Tensors.Unheld{}: (Tensors.Unheld{},Tensors.buffer(~Tile.Tile8,fill,count))

# What an axis of the new extents reads: a source axis of its elements, the
# source groups at its anchors, nothing for one index, or, for the axis of
# the new lanes, no cell axis.
type Link is Data:
  Linked{axis: Nat}
  Anchor{}
  Alone{}
  Lane{}

def prepended(link: Link,rest: Maybe<&2,List<&2,Link>>) -> Maybe<&2,List<&2,Link>>:
  match rest:
    case Some{links}: Some{link <> links}
    case None{}: None{}

# The next step of a pairing: both ends reached, a target axis of one index,
# a source axis of one index, two axes of one extent, or none of these.
type Step is Data:
  BothEnded{}
  TargetUnit{}
  SourceUnit{}
  SameExtent{}
  Unpaired{}

def step_of(target: List<&2,U32>,source: List<&2,U32>) -> Step:
  match target source:
    case Nil{} Nil{}: BothEnded{}
    case Nil{} Con{+other,more}: Bool.pick(Step,U32.is_eq(other,1),SourceUnit{},Unpaired{})
    case Con{+extent,rest} Nil{}: Bool.pick(Step,U32.is_eq(extent,1),TargetUnit{},Unpaired{})
    case Con{+extent,rest} Con{+other,more}:
      Bool.pick(Step,U32.is_eq(extent,1),TargetUnit{},Bool.pick(Step,U32.is_eq(other,1),SourceUnit{},Bool.pick(Step,U32.is_eq(extent,other),SameExtent{},Unpaired{})))

# The target axes paired in order with the source axes from source_index on:
# an axis of one index reads nothing, every other the next source axis of
# more than one index, which has its extent. None when they do not pair.
# Every step drops an axis, within the fuel of both lengths.
def paired(fuel: Nat,step: Step,+target: List<&2,U32>,+source: List<&2,U32>,+source_index: Nat) -> Maybe<&2,List<&2,Link>>:
  match fuel step:
    case fuel BothEnded{}: Some{[]}
    case 1n+left TargetUnit{}:
      +rest = List.drop(&2,U32,target,1n)
      prepended(Alone{},paired(left,step_of(rest,source),rest,source,source_index))
    case 1n+left SourceUnit{}:
      +more = List.drop(&2,U32,source,1n)
      paired(left,step_of(target,more),target,more,Nat.add(source_index,1n))
    case 1n+left SameExtent{}:
      +rest = List.drop(&2,U32,target,1n)
      +more = List.drop(&2,U32,source,1n)
      prepended(Linked{source_index},paired(left,step_of(rest,more),rest,more,Nat.add(source_index,1n)))
    case fuel step: None{}

def pairing(+target: List<&2,U32>,+source: List<&2,U32>,+source_index: Nat) -> Maybe<&2,List<&2,Link>>:
  paired(Nat.add(List.length(&2,U32,target),List.length(&2,U32,source)),step_of(target,source),target,source,source_index)

def joined_links(before: Maybe<&2,List<&2,Link>>,after: Maybe<&2,List<&2,Link>>) -> Maybe<&2,List<&2,Link>>:
  match before after:
    case Some{before} Some{after}: Some{List.append(&2,Link,before,Anchor{} <> Lane{} <> after)}
    case before after: None{}

# Every axis of the new extents: those before the split pair with the source
# axes before the lane axis, those after the new lanes with those after it.
def split_links(+source: List<&2,U32>,+source_lane: Nat,+shape: List<&2,U32>,+axis: Nat) -> Maybe<&2,List<&2,Link>>:
  joined_links(pairing(List.take(&2,U32,shape,axis),List.take(&2,U32,source,source_lane),0n),
    pairing(List.drop(&2,U32,shape,Nat.add(axis,2n)),List.drop(&2,U32,source,Nat.add(source_lane,1n)),Nat.add(source_lane,1n)))

# The grids and the source lanes the readings of a split use.
type Sides is Data:
  Sides{source_grid: Grid.Grid,target_grid: Grid.Grid,lane: Nat,position: Grid.Position,width: U32,source_groups: U32,groups_axis: U32}

def sides_target(sides: Sides) -> Grid.Grid:
  Sides{source_grid,target_grid,lane,position,width,source_groups,groups_axis} = sides
  target_grid

def linked_source(+sides: Sides,+axis: Nat,+extent: U32) -> Tasks.Source:
  Sides{+source_grid,target_grid,+lane,+position,width,source_groups,groups_axis} = sides
  read_at(source_grid,source_axis(lane,True{},position,axis),extent,1,1,1)

def anchor_source(+sides: Sides) -> Tasks.Source:
  Sides{+source_grid,target_grid,lane,position,+width,+source_groups,+groups_axis} = sides
  read_at(source_grid,groups_axis,source_groups,width,width,8)

def link_readings(links: List<&2,Link>,shape: List<&2,U32>,+sides: Sides,+index: U32) -> List<&2,Tasks.Reading>:
  match links shape:
    case Con{Lane{},links} Con{extent,shape}: link_readings(links,shape,sides,index)
    case Con{Linked{+axis},links} Con{+extent,shape}:
      Tasks.Reading{extent,Grid.block(sides_target(sides),index,extent),linked_source(sides,axis,extent)} <> link_readings(links,shape,sides,(index + 1 : U32))
    case Con{Anchor{},links} Con{+extent,shape}:
      Tasks.Reading{extent,Grid.block(sides_target(sides),index,extent),anchor_source(sides)} <> link_readings(links,shape,sides,(index + 1 : U32))
    case Con{Alone{},links} Con{+extent,shape}:
      Tasks.Reading{extent,Grid.block(sides_target(sides),index,extent),Tasks.Unread{}} <> link_readings(links,shape,sides,(index + 1 : U32))
    case links shape: []

def crossing_readings(+target_grid: Grid.Grid,cells: List<&2,U32>,+index: U32) -> List<&2,Tasks.Reading>:
  match cells:
    case Nil{}: []
    case Con{+extent,rest}: Tasks.Reading{extent,Grid.block(target_grid,index,extent),Tasks.Crossing{}} <> crossing_readings(target_grid,rest,(index + 1 : U32))

def groups_reading(+sides: Sides,+groups: U32,+index: U32) -> Tasks.Reading:
  Tasks.Reading{groups,Grid.block(sides_target(sides),index,groups),Tasks.Unread{}}

# How every target cell axis reads the source cells: the groups of the new
# lanes repeat the cells of their anchor. Axes that do not pair read across
# the source, which a task then holds whole.
def split_readings(found: Maybe<&2,List<&2,Link>>,+shape: List<&2,U32>,+cells: List<&2,U32>,+sides: Sides,+outermost: Bool,+groups: U32) -> List<&2,Tasks.Reading>:
  match found:
    case Some{+links}:
      Bool.pick(List<&2,Tasks.Reading>,outermost,groups_reading(sides,groups,0) <> link_readings(links,shape,sides,1),
        List.append(&2,Tasks.Reading,link_readings(links,shape,sides,0),[groups_reading(sides,groups,(U32.from_nat(List.length(&2,U32,cells)) - 1 : U32))]))
    case None{}: crossing_readings(sides_target(sides),cells,0)

def split_cells_of(+split: Split,+links: Maybe<&2,List<&2,Link>>,+shape: List<&2,U32>,+cells: List<&2,U32>,+source_cells: List<&2,U32>,+lane: Nat,+workers: U32,
  pieces: Pieces.Pieces(~Tile.Tile8)) -> Pieces.Pieces(~Tile.Tile8):
  Split{+outer,+anchors,+width,+after,+groups,+source_groups,+position} = split
  +outermost = Grid.is_outermost(position)
  +groups_axis = Bool.pick(U32,outermost,0,(U32.from_nat(List.length(&2,U32,source_cells)) - 1 : U32))
  +sides = {Sides{Tensors.grid_of(source_cells),Tensors.grid_of(cells),lane,position,width,source_groups,groups_axis} : Sides}
  Tensors.moved_with(~Tile.Tile8,~Split,~split_gather,Tasks.components_of([split_readings(links,shape,cells,sides,outermost,groups)]),
    Split{outer,anchors,width,after,groups,source_groups,position},3.0,16.0,Tensors.Source{pieces,Tensors.Ended{}},[Tensors.origin(source_cells,0)],cells,zero_tile(),workers)

def split_status(status: Tensors.Status,+split: Split,+links: Maybe<&2,List<&2,Link>>,+shape: List<&2,U32>,+cells: List<&2,U32>,+source_cells: List<&2,U32>,+lane: Nat,+workers: U32,
  pieces: Pieces.Pieces(~Tile.Tile8)) -> Tensors.Tensor<Tile.Tile8>:
  match status:
    case Tensors.Usable{}: Tensors.Tensor{split_cells_of(split,links,shape,cells,source_cells,lane,workers,pieces),cells,zero_tile(),Tensors.Usable{},workers}
    case Tensors.Failed{problem}: Tensors.Tensor{Partition.Leaf{Tensors.buffer(~Tile.Tile8,zero_tile(),0),0,0,Unit{}},cells,zero_tile(),Tensors.Failed{problem},workers}

# The tiled form reshaped to extents shape that split its lane axis into
# axis, of anchors indices, and the next, of width, with outer elements
# before them and after elements after: the lanes move to the next axis, the
# groups keeping their position.
def split_lanes(tiled: Tiled,+shape: List<&2,U32>,+axis: Nat,+outer: U32,+anchors: U32,+width: U32,+after: U32) -> Tiled:
  Tiled{Tensors.Tensor{pieces,+source_cells,fill,status,+workers},+old,+lanes,+position} = tiled
  +lane = Nat.add(axis,1n)
  +cells = cell_shape(shape,lane,position)
  Tiled{split_status(status,Split{outer,anchors,width,after,groups(width),groups(lanes),position},split_links(logical_shape(source_cells,old,lanes,position),old,shape,axis),shape,
    cells,source_cells,old,workers,pieces),lane,width,position}

# The tiled form with its lanes on an axis: itself when they lie there,
# otherwise its cells relaned.
def along_when(same: Bool,tiled: Tiled,+axis: Nat,+position: Grid.Position) -> Tiled:
  match same:
    case True{}: tiled
    case False{}: relane(tiled,axis,position)

def along(tiled: Tiled,+axis: Nat) -> Tiled:
  Tiled{cells,+own,lanes,+position} = tiled
  along_when(Nat.is_eq(own,axis),Tiled{cells,own,lanes,position},axis,position)


# ---- The groups at another position ----

# The view of cells with the groups at one end that reads them with the
# groups at the other.
def scaled_axes(extents: List<&2,U32>,strides: List<&2,U32>,+stride: U32) -> List<&2,Views.Axis>:
  match extents strides:
    case Con{+extent,rest} Con{+own,more}: Views.Axis{extent,(own * stride : U32)} <> scaled_axes(rest,more,stride)
    case extents strides: []

def inner_axes(+extents: List<&2,U32>,+stride: U32) -> List<&2,Views.Axis>:
  scaled_axes(extents,Grid.strides(extents),stride)

def repositioned_view(position: Grid.Position,+cells: List<&2,U32>) -> List<&2,Views.Axis>:
  match position:
    case Grid.Innermost{}:
      +others = Grid.others_of(Grid.Outermost{},cells)
      List.append(&2,Views.Axis,inner_axes(others,1),[Views.Axis{Grid.groups_of(Grid.Outermost{},cells),Grid.product(others)}])
    case Grid.Outermost{}:
      +others = Grid.others_of(Grid.Innermost{},cells)
      +groups = Grid.groups_of(Grid.Innermost{},cells)
      Views.Axis{groups,1} <> inner_axes(others,groups)

def repositioned_when(same: Bool,tiled: Tiled,+position: Grid.Position) -> Tiled:
  match same:
    case True{}:
      Tiled{cells,axis,lanes,old} = tiled
      Tiled{cells,axis,lanes,position}
    case False{}:
      Tiled{Tensors.Tensor{pieces,+cells,fill,status,workers},axis,lanes,old} = tiled
      Tiled{Tensors.moved(~Tile.Tile8,"repositioned",Tensors.Tensor{pieces,cells,fill,status,workers},repositioned_view(position,cells),0,[]),axis,lanes,position}

# The tiled form with its groups at position. Lanes on the unit axis have no
# groups to move: the cells stay and read the same at either position.
def repositioned(tiled: Tiled,+position: Grid.Position) -> Tiled:
  Tiled{Tensors.Tensor{pieces,+cells,fill,status,workers},+axis,lanes,+own} = tiled
  repositioned_when(Grid.same_position(own,position) || Bool.not(Nat.is_lt(axis,List.length(&2,U32,cells))),
    Tiled{Tensors.Tensor{pieces,cells,fill,status,workers},axis,lanes,own},position)

# The tiled form with its lanes on an axis and its groups at position, in
# one pass.
def arranged_when(same: Bool,tiled: Tiled,+axis: Nat,+position: Grid.Position) -> Tiled:
  match same:
    case True{}: repositioned(tiled,position)
    case False{}: relane(tiled,axis,position)

def arranged(tiled: Tiled,+axis: Nat,+position: Grid.Position) -> Tiled:
  Tiled{cells,+own,lanes,own_position} = tiled
  arranged_when(Nat.is_eq(own,axis),Tiled{cells,own,lanes,own_position},axis,position)

# ---- Writes along the lane axis ----

# The lanes of inside whose channel, base plus the lane, lies from start
# below end, the others of outside.
def lanes_between(+start: U32,+end: U32,+base: U32,inside: Tile.Tile8,outside: Tile.Tile8) -> Tile.Tile8:
  Tile.Tile{a0,a1,a2,a3,a4,a5,a6,a7} = inside
  Tile.Tile{k0,k1,k2,k3,k4,k5,k6,k7} = outside
  Tile.Tile{Choice.chosen(~F32,U32.is_le(start,(base + 0 : U32)) && U32.is_lt((base + 0 : U32),end),a0,k0),
    Choice.chosen(~F32,U32.is_le(start,(base + 1 : U32)) && U32.is_lt((base + 1 : U32),end),a1,k1),
    Choice.chosen(~F32,U32.is_le(start,(base + 2 : U32)) && U32.is_lt((base + 2 : U32),end),a2,k2),
    Choice.chosen(~F32,U32.is_le(start,(base + 3 : U32)) && U32.is_lt((base + 3 : U32),end),a3,k3),
    Choice.chosen(~F32,U32.is_le(start,(base + 4 : U32)) && U32.is_lt((base + 4 : U32),end),a4,k4),
    Choice.chosen(~F32,U32.is_le(start,(base + 5 : U32)) && U32.is_lt((base + 5 : U32),end),a5,k5),
    Choice.chosen(~F32,U32.is_le(start,(base + 6 : U32)) && U32.is_lt((base + 6 : U32),end),a6,k6),
    Choice.chosen(~F32,U32.is_le(start,(base + 7 : U32)) && U32.is_lt((base + 7 : U32),end),a7,k7)}

# A part written into a target along the lane axis from index start, width
# indices wide: target cell (o, k), of other index o and group k, takes in
# lane l, when start <= 8k + l < start + width, element 8k + l - start of the
# part's lane axis at o. groups and part_groups count the groups, others the
# cells of one group.
type Write is Data:
  Write{start: U32,width: U32,groups: U32,part_groups: U32,others: U32,position: Grid.Position}

# The cell of group group at other index other.
def cell_of(position: Grid.Position,+other: U32,+group: U32,+groups: U32,+others: U32) -> U32:
  match position:
    case Grid.Innermost{}: (other * groups + group : U32)
    case Grid.Outermost{}: (group * others + other : U32)

# The other indices of a group's cells a target piece may hold, from low
# below high.
type Others is Data:
  Others{low: U32,high: U32}

def others_of(position: Grid.Position,+groups: U32,+others: U32,+first: U32,+count: U32,+group: U32) -> Others:
  match position:
    case Grid.Innermost{}: Others{(first / U32.max(groups,1) : U32),((first + count - 1) / U32.max(groups,1) + 1 : U32)}
    case Grid.Outermost{}:
      +base = (group * others : U32)
      Others{(U32.min(U32.max(first,base),(base + others : U32)) - base : U32),(U32.min(U32.max((first + count : U32),base),(base + others : U32)) - base : U32)}

# The holdings of the target and the part and the target piece.
type Writing is Type:
  Writing{target: Move.Holding<Tile.Tile8>,part: Move.Holding<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>}

def touched_written(+start: U32,+end: U32,+base: U32,+local: U32,value: Tile.Tile8,target: Move.Holding<Tile.Tile8>,part: Move.Holding<Tile.Tile8>,
  read: Storage.Buffer<Tile.Tile8> & Tile.Tile8) -> Writing:
  (output,kept) = read
  Writing{target,part,Storage.write(~Tile.Tile8,local,lanes_between(start,end,base,value,kept),output)}

def touched_high(+shift: U32,+start: U32,+end: U32,+base: U32,+local: U32,below: Tile.Tile8,target: Move.Holding<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>,
  pair: Move.Holding<Tile.Tile8> & Tile.Tile8) -> Writing:
  (part,above) = pair
  touched_written(start,end,base,local,joined(shift,below,above),target,part,Storage.read(~Tile.Tile8,output,local))

def touched_low(+shift: U32,+start: U32,+end: U32,+base: U32,+local: U32,further: Bool,+next: U32,target: Move.Holding<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>,
  pair: Move.Holding<Tile.Tile8> & Tile.Tile8) -> Writing:
  (part,below) = pair
  touched_high(shift,start,end,base,local,below,target,output,block_read(~Move.Holding<Tile.Tile8>,~held_cell,further,next,0,0,part))

# A target cell the part reaches: the part cells of its first lane and the
# next joined, their lanes the part reaches put over its own.
def touched_cell(+write: Write,+first: U32,+group: U32,+other: U32,writing: Writing) -> Writing:
  Write{+start,+width,+groups,+part_groups,+others,+position} = write
  Writing{target,part,output} = writing
  +reach = (8 * group + 8 - start : U32)
  +above = (reach / 8 : U32)
  +shift = (reach % 8 : U32)
  +base = (8 * group : U32)
  touched_low(shift,start,(start + width : U32),base,(cell_of(position,other,group,groups,others) - first : U32),
    U32.is_lt(0,shift) && U32.is_lt(above,part_groups),cell_of(position,other,above,part_groups,others),target,output,
    block_read(~Move.Holding<Tile.Tile8>,~held_cell,U32.is_le(8,reach),cell_of(position,other,(above - 1 : U32),part_groups,others),0,0,part))

def touched_when(inside: Bool,+write: Write,+first: U32,+group: U32,+other: U32,writing: Writing) -> Writing:
  match inside:
    case True{}: touched_cell(write,first,group,other,writing)
    case False{}: writing

def touched_others(fuel: Nat,+write: Write,+first: U32,+count: U32,+group: U32,+other: U32,writing: Writing) -> Writing:
  match fuel:
    case 0n: writing
    case 1n+rest:
      Write{+start,+width,+groups,+part_groups,+others,+position} = write
      +index = cell_of(position,other,group,groups,others)
      +write = {Write{start,width,groups,part_groups,others,position} : Write}
      touched_others(rest,write,first,count,group,(other + 1 : U32),
        touched_when(U32.is_le(first,index) && U32.is_lt(index,(first + count : U32)),write,first,group,other,writing))

def touched_range(found: Others,+write: Write,+first: U32,+count: U32,+group: U32,writing: Writing) -> Writing:
  Others{+low,+high} = found
  touched_others(U32.to_nat((U32.max(high,low) - low : U32)),write,first,count,group,low,writing)

def touched_group(+write: Write,+first: U32,+count: U32,+group: U32,writing: Writing) -> Writing:
  Write{+start,+width,+groups,+part_groups,+others,+position} = write
  touched_range(others_of(position,groups,others,first,count,group),Write{start,width,groups,part_groups,others,position},first,count,group,writing)

def touched_groups(fuel: Nat,+write: Write,+first: U32,+count: U32,+group: U32,writing: Writing) -> Writing:
  match fuel:
    case 0n: writing
    case 1n+rest: touched_groups(rest,write,first,count,(group + 1 : U32),touched_group(write,first,count,group,writing))

def write_done(rest: Tensors.Holdings<Tile.Tile8>,writing: Writing) -> Tensors.Holdings<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  Writing{target,part,output} = writing
  (Tensors.Holds{target,Tensors.Holds{part,rest}},output)

# The groups the part reaches, from the group of its first index to that of
# its last.
def write_part(rest: Tensors.Holdings<Tile.Tile8>,+write: Write,+first: U32,+count: U32,copied: Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) ->
  Tensors.Holdings<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  match rest:
    case Tensors.Holds{part,later}:
      Write{+start,+width,+groups,+part_groups,+others,+position} = write
      (target,output) = copied
      +reached = Bool.pick(U32,U32.is_zero(width),0,((start + width - 1) / 8 - start / 8 + 1 : U32))
      write_done(later,touched_groups(U32.to_nat(reached),Write{start,width,groups,part_groups,others,position},first,count,(start / 8 : U32),Writing{target,part,output}))
    case Tensors.Unheld{}:
      (target,output) = copied
      (Tensors.Holds{target,Tensors.Unheld{}},output)

# The target piece that holds exactly the cells of a target piece is taken
# out of the target and written in place, and the pieces before it are let
# go: the later target pieces of a task start after it. Otherwise its cells
# are copied.
def write_hole(exact: Bool,rest: Tensors.Holdings<Tile.Tile8>,+write: Write,+first: U32,+count: U32,behind: Move.Stack<Tile.Tile8>,below: Move.Stack<Tile.Tile8>,
  +start: U32,+span: U32,buffer: Storage.Buffer<Tile.Tile8>) -> Tensors.Holdings<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  match exact:
    case True{}: write_part(rest,write,first,count,(Move.Holding{Move.Bottom{},below},buffer))
    case False{}:
      write_part(rest,write,first,count,
        Move.copied_into(~Tile.Tile8,Move.Holding{behind,Move.Held{buffer,start,span,below}},first,count,Tensors.buffer(~Tile.Tile8,zero_tile(),count)))

def write_lent(rest: Tensors.Holdings<Tile.Tile8>,+write: Write,+first: U32,+count: U32,band: Move.Band<Tile.Tile8>) -> Tensors.Holdings<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  Move.Band{lent,buffer,offset} = band
  match lent:
    case Move.Hole{behind,below,+start,+span,spare}: write_hole(U32.is_eq(start,first) && U32.is_eq(span,count),rest,write,first,count,behind,below,start,span,buffer)
    case Move.Assembled{holding,capacity}: write_part(rest,write,first,count,(holding,buffer))

# A target piece: the target's own piece, then every cell the part reaches
# written.
def write_gather(+write: Write,+components: Tasks.Components,+first: U32,+count: U32,+fill: Tile.Tile8,input: Tensors.Holdings<Tile.Tile8>) ->
  Tensors.Holdings<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  match input:
    case Tensors.Holds{target,rest}: write_lent(rest,write,first,count,Move.band(~Tile.Tile8,target,first,count,zero_tile(),Move.spare(~Tile.Tile8,zero_tile())))
    case Tensors.Unheld{}: (Tensors.Unheld{},Tensors.buffer(~Tile.Tile8,fill,count))

# Every target cell axis reads the same index of the target.
def own_readings(extents: List<&2,U32>,+grid: Grid.Grid,+index: U32) -> List<&2,Tasks.Reading>:
  match extents:
    case Nil{}: []
    case Con{+extent,rest}: Tasks.Reading{extent,Grid.block(grid,index,extent),read_at(grid,index,extent,1,1,1)} <> own_readings(rest,grid,(index + 1 : U32))

# Every target cell axis reads the same index of the part, but the groups:
# group k reads the part groups from (8k - start) / 8 to (8k + 7 - start) / 8,
# none below the first.
def part_readings(extents: List<&2,U32>,+part_grid: Grid.Grid,+grid: Grid.Grid,+groups_axis: U32,+start: U32,+part_groups: U32,+index: U32) -> List<&2,Tasks.Reading>:
  match extents:
    case Nil{}: []
    case Con{+extent,rest}:
      Tasks.Reading{extent,Grid.block(grid,index,extent),
        Bool.pick(Tasks.Source,U32.is_eq(index,groups_axis),
          Tasks.Read{index,Grid.block(part_grid,index,part_groups),8,(8 - start % 8 : U32),False{},8,8,(start / 8 + 1 : U32),part_groups,False{}},
          read_at(part_grid,index,extent,1,1,1))} <> part_readings(rest,part_grid,grid,groups_axis,start,part_groups,(index + 1 : U32))

def lanes_write(empty: Bool,+write: Write,pieces: Pieces.Pieces(~Tile.Tile8),part_pieces: Pieces.Pieces(~Tile.Tile8),+cells: List<&2,U32>,+part_cells: List<&2,U32>,
  +fill: Tile.Tile8,status: Tensors.Status,+workers: U32) -> Tensors.Tensor<Tile.Tile8>:
  match empty:
    case True{}: Tensors.Tensor{pieces,cells,fill,status,workers}
    case False{}:
      Write{+start,+width,+groups,+part_groups,+others,+position} = write
      +grid = Tensors.grid_of(cells)
      Tensors.Tensor{Tensors.moved_with(~Tile.Tile8,~Write,~write_gather,
        Tasks.components_of([own_readings(cells,grid,0),
          part_readings(cells,Tensors.grid_of(part_cells),grid,Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,cells)) - 1 : U32)),start,part_groups,0)]),
        Write{start,width,groups,part_groups,others,position},Tensors.move_rows(),0.0,Tensors.Source{pieces,Tensors.Source{part_pieces,Tensors.Ended{}}},
        [Tensors.origin(cells,0),Tensors.origin(part_cells,0)],cells,fill,workers),cells,fill,status,workers}

def lanes_parted(+start: U32,+lane: Nat,+lanes: U32,+position: Grid.Position,pieces: Pieces.Pieces(~Tile.Tile8),+cells: List<&2,U32>,+fill: Tile.Tile8,status: Tensors.Status,
  +workers: U32,part: Tiled) -> Tiled:
  Tiled{Tensors.Tensor{part_pieces,+part_cells,part_fill,part_status,part_workers},part_lane,+width,part_position} = part
  +lane_groups = groups(lanes)
  +count = Tensors.count_or_zero(cells)
  Tiled{lanes_write(U32.is_zero(width) || U32.is_zero(count),Write{start,width,lane_groups,groups(width),(count / U32.max(lane_groups,1) : U32),position},pieces,part_pieces,cells,part_cells,
    fill,status,workers),lane,lanes,position}

# The part, of the extents of a usable target but along its lane axis, written
# into the target from index start of the lane axis: the target keeps its
# lanes, every cell the part reaches takes the lanes the part writes.
def lanes_written(target: Tiled,+start: U32,part: Tiled) -> Tiled:
  Tiled{Tensors.Tensor{pieces,+cells,+fill,status,+workers},+lane,+lanes,+position} = target
  lanes_parted(start,lane,lanes,position,pieces,cells,fill,status,workers,arranged(part,lane,position))

# ---- Metadata ----

# The lane axis of a tiled tensor.
def lane_axis_of(tiled: Tiled) -> Tiled & Nat:
  Tiled{cells,+axis,lanes,position} = tiled
  (Tiled{cells,axis,lanes,position},axis)

def position_of(tiled: Tiled) -> Tiled & Grid.Position:
  Tiled{cells,axis,lanes,+position} = tiled
  (Tiled{cells,axis,lanes,position},position)

# The metadata of a tiled tensor: its extents, status and worker count.
def with_metadata(~Result: Type,tiled: Tiled,body: Tiled -> List<&2,U32> -> Tensors.Status -> U32 -> Result) -> Result:
  Tiled{Tensors.Tensor{pieces,+cells,fill,+status,+workers},+axis,+lanes,+position} = tiled
  body(Tiled{Tensors.Tensor{pieces,cells,fill,status,workers},axis,lanes,position},logical_shape(cells,axis,lanes,position),status,workers)

# The axis of the largest extent but skip, the later among equal extents, or
# the unit axis past the last when no other axis has one.
def longest_other(shape: List<&2,U32>,+skip: Nat,+index: Nat,+best: Nat,+extent: U32) -> Nat:
  match shape:
    case []: best
    case axis_extent <> rest:
      +axis_extent = axis_extent
      +longer = U32.is_le(extent,axis_extent) && Bool.not(Nat.is_eq(index,skip))
      longest_other(rest,skip,1n+index,Bool.pick(Nat,longer,index,best),Bool.pick(U32,longer,axis_extent,extent))

def with_status(status: Tensors.Status,tiled: Tiled) -> Tiled:
  Tiled{cells,axis,lanes,position} = tiled
  Tiled{Tensors.with_status(~Tile.Tile8,status,cells),axis,lanes,position}

# The cells of a tiled tensor.
def cells_of(tiled: Tiled) -> Tensors.Tensor<Tile.Tile8>:
  Tiled{cells,axis,lanes,position} = tiled
  cells

# The cells of every group of the first axis in order, the groups outermost:
# the tiles of a tensor read as [C].
def listed(tiled: Tiled) -> Result<&2,&2,Tensors.Problem,List<&2,Tile.Tile8>>:
  Tensors.to_list(~Tile.Tile8,cells_of(arranged(tiled,0n,Grid.Outermost{})))

# ---- The unit axis ----

# A cell of the unit axis: the value in its first lane, +0 in the others.
def unit_tile(value: F32) -> Tile.Tile8:
  Tile.Tile{value,0.0,0.0,0.0,0.0,0.0,0.0,0.0}

# The cells of a tiled tensor with its lanes on the unit axis of its shape:
# one value a cell, in the shape of the tensor.
def unit_cells(tiled: Tiled) -> Tensors.Tensor<Tile.Tile8>:
  with_metadata(~Tensors.Tensor<Tile.Tile8>,tiled,tiled => shape => status => workers => cells_of(along(tiled,List.length(&2,U32,shape))))

# Cells of one value each, the lanes on the unit axis of their shape.
def unit(cells: Tensors.Tensor<Tile.Tile8>) -> Tiled:
  Tensors.with_metadata(~Tile.Tile8,~Tiled,cells,cells => shape => fill => status => workers => Tiled{cells,List.length(&2,U32,shape),1,Grid.Innermost{}})
