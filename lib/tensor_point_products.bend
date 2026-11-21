# A one-cell product over owned ranges of the operands. Jobs share the
# pointwise kernel and deposit fragments bounded by the canonical grid.
import Base
import ./default_kernel_cost.bend as DefaultCost
import ./kernel_cost.bend as Cost
import ./kernel_gemm.bend as Gemm
import ./kernel_gemm_buffer.bend as GemmBuffer
import ./proofs/kernel_gemm_certificate.bend as GemmCertificate
import ./kernel_tile.bend as Tile
import ./storage_buffer.bend as Storage
import ./storage_clone.bend as Clone
import ./storage_copy.bend as Copy
import ./tensor.bend as Tensors
import ./tensor_grid.bend as Grid
import ./tensor_halo.bend as Halo
import ./tensor_move.bend as Move
import ./tensor_pieces.bend as Pieces
import ./tensor_tasks.bend as Tasks
import ./traversal_partition.bend as Partition

type Shape is Data:
  Shape{positions: U32,inner: U32,channels: U32,grid: Grid.Grid}

type Rectangle is Data:
  Rectangle{first: U32,positions: U32,inner: U32,channels: U32,group_major: Bool,grid: Grid.Grid}

def Cells() -> Type:
  Storage.Buffer<Tile.Tile8>

def Kernel() -> Type:
  @+matrix: Gemm.PointMatrix -> Cells() -> Storage.Buffer<Tile.Tile16> -> Cells() -> Storage.Buffer<Tile.Tile16> -> GemmBuffer.PointKept

def zero() -> Tile.Tile8:
  Tile.filled(0.0)

def emitted_piece(lefts: Pieces.Lefts<Cells()>,+first: U32,+count: U32,buffers: Storage.Buffer<Tile.Tile16> & Cells()) ->
  Pieces.Walked<Cells(),Storage.Buffer<Tile.Tile16>>:
  (staged,output) = buffers
  Pieces.piece_walked(~Tile.Tile8,~Storage.Buffer<Tile.Tile16>,lefts,first,count,(output,staged))

# Each fragment stays in the slot containing its first cell.
def emit_piece(+grid: Grid.Grid,+piece: U32,+first: U32,+end: U32,+groups: U32,+positions: U32,+group_major: Bool,
  ready: Pieces.Walking<Cells(),Storage.Buffer<Tile.Tile16>>) -> Pieces.Walked<Cells(),Storage.Buffer<Tile.Tile16>>:
  Pieces.Walking{lefts,staged} = ready
  +low = U32.max(first,Grid.start(grid,piece))
  +count = (U32.min(end,(Grid.start(grid,piece) + Grid.span(grid,piece) : U32)) - low : U32)
  emitted_piece(lefts,low,count,
    GemmBuffer.staged_slice(Gemm.StagedSlice{groups,positions,(low - first : U32),group_major},count,staged,Tensors.buffer(~Tile.Tile8,zero(),count)))

def emit(remaining: Nat,+grid: Grid.Grid,+piece: U32,+first: U32,+end: U32,+groups: U32,+positions: U32,+group_major: Bool,
  ready: Pieces.Walking<Cells(),Storage.Buffer<Tile.Tile16>>) -> Pieces.Walked<Cells(),Storage.Buffer<Tile.Tile16>>:
  match remaining:
    case 0n: emit_piece(grid,piece,first,end,groups,positions,group_major,ready)
    case 1n+rest:
      Pieces.joined(~Cells(),~Storage.Buffer<Tile.Tile16>,
        emit(rest,grid,(piece + 1 : U32),first,end,groups,positions,group_major,
          Pieces.pushed(~Cells(),~Storage.Buffer<Tile.Tile16>,emit_piece(grid,piece,first,end,groups,positions,group_major,ready))))

type Completed is Type:
  Completed{pieces: Pieces.Pieces(~Tile.Tile8),input: Cells(),bias: Cells()}

# The staged pairs are read in the target order before each piece is placed.
def emitted(+rectangle: Rectangle,kept: GemmBuffer.PointKept) -> Completed:
  Rectangle{+first,+positions,inner,+channels,+group_major,+grid} = rectangle
  GemmBuffer.PointKept{bias,input,pairs,staged} = kept
  +groups = Gemm.blocks(channels)
  +count = (positions * groups : U32)
  Completed{Pair.fst(Pieces.Pieces(~Tile.Tile8),Storage.Buffer<Tile.Tile16>,Pieces.walk_result(~Cells(),~Storage.Buffer<Tile.Tile16>,
    emit(Grid.later_pieces(grid,first,count),grid,Grid.piece_of(grid,first),first,(first + count : U32),groups,positions,group_major,Pieces.Walking{Pieces.NoLefts{},staged}))),input,bias}

# A position block keeps its input cells within the kernel's tile budget.
# Only the last block may end in a short row unit.
def block_positions(+inner: U32) -> U32:
  (4 * U32.max(1,(Gemm.block_tiles() / Gemm.blocks(inner) / 4 : U32)) : U32)

def rect_blocks(~kernel: Kernel(),remaining: Nat,+rectangle: Rectangle,+input_base: U32,+first: U32,+span: U32,state: GemmBuffer.PointKept) -> GemmBuffer.PointKept:
  match remaining:
    case 0n: state
    case 1n+rest:
      Rectangle{output_first,+positions,+inner,+channels,group_major,grid} = rectangle
      GemmBuffer.PointKept{bias,input,pairs,staged} = state
      +rows = U32.min(span,(positions - first : U32))
      +groups = Gemm.blocks(inner)
      +end = (first + rows : U32)
      rect_blocks(~kernel,rest,rectangle,input_base,(first + rows : U32),span,
        kernel(Gemm.PointMatrix{Gemm.TileMatrix{channels,rows,inner,first,end},(input_base + first * groups : U32),groups},input,pairs,bias,staged))

def computed(~kernel: Kernel(),+rectangle: Rectangle,+input_base: U32,input: Cells(),bias: Cells(),pair: Cells() & Storage.Buffer<Tile.Tile16>) -> Completed:
  Rectangle{first,+positions,+inner,+channels,group_major,grid} = rectangle
  (weights,pairs) = pair
  +span = block_positions(inner)
  emitted(rectangle,rect_blocks(~kernel,U32.to_nat(Grid.divided_up(positions,span)),rectangle,input_base,0,span,
    GemmBuffer.PointKept{bias,input,pairs,Tensors.buffer(~Tile.Tile16,Tile.filled_pair(0.0),(Gemm.pairs_of(channels) * positions : U32))}))

# A band back in the holding, with the pairs.
def band_returned(lent: Move.Lent<Tile.Tile8>,pair: Cells() & Storage.Buffer<Tile.Tile16>) -> (Move.Holding<Tile.Tile8> & Move.Spare<Tile.Tile8>) & Storage.Buffer<Tile.Tile16>:
  (buffer,pairs) = pair
  (Move.returned(~Tile.Tile8,lent,buffer),pairs)

# One half of the pairs from target on: the inner tiles of a band, each
# combined into its pair by half.
def band_half(~half: Tile.Tile16 -> Tile.Tile8 -> Tile.Tile16,+target: U32,+inner: U32,band: Move.Band<Tile.Tile8>,pairs: Storage.Buffer<Tile.Tile16>) ->
  (Move.Holding<Tile.Tile8> & Move.Spare<Tile.Tile8>) & Storage.Buffer<Tile.Tile16>:
  Move.Band{lent,buffer,+offset} = band
  band_returned(lent,GemmBuffer.updated_into(~Tile.Tile8,~Tile.Tile16,~Tile.Tile8,~Gemm.TilePlan,~Gemm.tile_read,~GemmCertificate.tile_read,~half,~Gemm.tile_placed,
    U32.to_nat(inner),Gemm.TilePlan{offset,target},buffer,pairs))

# The upper halves of pair pair: its odd group, or its even group again when
# the last group has no partner.
def upper_band(+pair: U32,+groups: U32,+inner: U32,+base: U32,state: (Move.Holding<Tile.Tile8> & Move.Spare<Tile.Tile8>) & Storage.Buffer<Tile.Tile16>) ->
  (Move.Holding<Tile.Tile8> & Move.Spare<Tile.Tile8>) & Storage.Buffer<Tile.Tile16>:
  ((holding,spare),pairs) = state
  band_half(~Gemm.upper_half,(pair * inner : U32),inner,
    Move.band(~Tile.Tile8,holding,(base + U32.min((pair * 2 + 1 : U32),(groups - 1 : U32)) * inner : U32),inner,zero(),spare),pairs)

# The pairs from pair pair on, each from the bands of its two groups in the
# held weights from base on.
def held_pairs(remaining: Nat,+pair: U32,+groups: U32,+inner: U32,+base: U32,state: (Move.Holding<Tile.Tile8> & Move.Spare<Tile.Tile8>) & Storage.Buffer<Tile.Tile16>) ->
  Storage.Buffer<Tile.Tile16>:
  match remaining:
    case 0n:
      (held,pairs) = state
      pairs
    case 1n+rest:
      ((holding,spare),pairs) = state
      held_pairs(rest,(pair + 1 : U32),groups,inner,base,upper_band(pair,groups,inner,base,
        band_half(~Gemm.lower_half,(pair * inner : U32),inner,Move.band(~Tile.Tile8,holding,(base + pair * 2 * inner : U32),inner,zero(),spare),pairs)))

def compute_rectangle(~kernel: Kernel(),+rectangle: Rectangle,+input_base: U32,+weight_base: U32,input: Cells(),bias: Cells(),weights: Move.Holding<Tile.Tile8>) -> Completed:
  Rectangle{first,positions,+inner,+channels,group_major,grid} = rectangle
  +pairs = Gemm.pairs_of(channels)
  computed(~kernel,rectangle,input_base,input,bias,(Tensors.buffer(~Tile.Tile8,zero(),0),
    held_pairs(U32.to_nat(pairs),0,Gemm.blocks(channels),inner,weight_base,((weights,Move.spare(~Tile.Tile8,zero())),
      Tensors.buffer(~Tile.Tile16,Tile.filled_pair(0.0),(pairs * inner : U32))))))

def shared_reach(+shape: Shape,+first: U32,+count: U32) -> Tasks.Span:
  Shape{+positions,+inner,channels,grid} = shape
  Tasks.Span{(first / positions * inner : U32),(count / positions * inner : U32)}

def shared_resources(resources: Cells() & Cells()) -> (Cells() & Cells()) & (Cells() & Cells()):
  (input,bias) = resources
  Storage.bind_pair(~Cells(),~Cells(),~((Cells() & Cells()) & (Cells() & Cells())),Clone.clone(~Tile.Tile8,input),left => right =>
    Storage.bind_pair(~Cells(),~Cells(),~((Cells() & Cells()) & (Cells() & Cells())),Clone.clone(~Tile.Tile8,bias),lower => upper => ((left,lower),(right,upper))))

def shared_done(+first: U32,+count: U32,bias: Cells(),finished: Completed) -> Partition.Tree<&1,Pieces.Pieces(~Tile.Tile8),Unit> & (Cells() & Cells()):
  Completed{pieces,input,local_bias} = finished
  (Partition.Leaf{pieces,first,count,Unit{}},(input,bias))

def shared_with_bias(+rectangle: Rectangle,+count: U32,+group: U32,input: Cells(),weights: Move.Holding<Tile.Tile8>,biases: Cells() & Cells()) ->
  Partition.Tree<&1,Pieces.Pieces(~Tile.Tile8),Unit> & (Cells() & Cells()):
  Rectangle{+first,positions,+inner,channels,group_major,grid} = rectangle
  (bias,local_bias) = biases
  shared_done(first,count,bias,compute_rectangle(~GemmBuffer.point_keeping,rectangle,0,(group * inner : U32),input,local_bias,weights))

def shared_leaf(work: Shape,first: U32,count: U32,weights: Move.Holding<Tile.Tile8>,resources: Cells() & Cells()) ->
  Partition.Tree<&1,Pieces.Pieces(~Tile.Tile8),Unit> & (Cells() & Cells()):
  Shape{+positions,+inner,+channels,+grid} = work
  (input,bias) = resources
  +first = first
  +count = count
  +group = (first / positions : U32)
  +groups = (count / positions : U32)
  +extent = U32.min((groups * 8 : U32),(channels - group * 8 : U32))
  shared_with_bias(Rectangle{first,positions,inner,extent,True{},grid},count,group,input,weights,
    Copy.range(~Tile.Tile8,groups,group,0,bias,Tensors.buffer(~Tile.Tile8,zero(),groups)))

def shared_tasks(+work: Shape,+weight_grid: Grid.Grid,decision: Tasks.Decision,input: Cells(),weights: Pieces.Pieces(~Tile.Tile8),bias: Cells()) -> Pieces.Pieces(~Tile.Tile8):
  Shape{positions,inner,channels,+grid} = work
  Tasks.Decision{+parallel,+tasks} = decision
  Tensors.placed(~Tile.Tile8,Pieces.ranges_of_grid(grid),zero(),
    Pieces.grouped_with(~Move.Holding<Tile.Tile8>,~Pieces.Pieces(~Tile.Tile8),~(Cells() & Cells()),~Shape,~shared_resources,~shared_leaf,tasks,parallel,work,
      Halo.windows(~Tile.Tile8,~Shape,~shared_reach,work,weight_grid,zero(),tasks,weights),(input,bias)))

def shared(+rows: U32,+inner: U32,+outputs: U32,+workers: U32,pieces: Pieces.Pieces(~Tile.Tile8),weights: Pieces.Pieces(~Tile.Tile8),bias: Cells()) -> Pieces.Pieces(~Tile.Tile8):
  +groups = Gemm.blocks(outputs)
  +grid = Tensors.grid_of([groups,1,rows])
  +work = {Shape{rows,inner,outputs,grid} : Shape}
  +input_bytes = U32.to_f32((rows * Gemm.blocks(inner) * 32 : U32))
  +cut_ns = Cost.copy_ns(32.0,DefaultCost.configuration())
  shared_tasks(work,Tensors.grid_of([groups,inner]),
    Tasks.decide_cut(grid,workers,rows,U32.to_f32(inner),0.0,cut_ns,Tasks.Extra{input_bytes,0.0,0.0}),
    Tensors.joined(~Tile.Tile8,(rows * Gemm.blocks(inner) : U32),zero(),pieces),weights,bias)

def panel_left(+shape: Shape,+first: U32,+count: U32) -> Tasks.Span:
  Shape{+positions,+inner,channels,grid} = shape
  +length = (positions * Gemm.blocks(inner) : U32)
  Tasks.Span{(first * length : U32),(count * length : U32)}

def panel_right(+shape: Shape,+first: U32,+count: U32) -> Tasks.Span:
  Shape{positions,+inner,+channels,grid} = shape
  +length = (Gemm.blocks(channels) * inner : U32)
  Tasks.Span{(first * length : U32),(count * length : U32)}

def panel_bands(~kernel: Kernel(),+rectangle: Rectangle,input_band: Move.Band<Tile.Tile8>,weight_band: Move.Band<Tile.Tile8>,bias: Cells()) -> Completed:
  Rectangle{first,positions,+inner,+channels,group_major,grid} = rectangle
  Move.Band{lent_input,input,base} = input_band
  Move.Band{lent_weights,weights,weight_base} = weight_band
  computed(~kernel,rectangle,base,input,bias,GemmBuffer.panel_pairs(channels,inner,weight_base,weights,
    Tensors.buffer(~Tile.Tile16,Tile.filled_pair(0.0),(Gemm.pairs_of(channels) * inner : U32))))

def panel_rectangle(~kernel: Kernel(),+shape: Shape,+first: U32,input: Move.Holding<Tile.Tile8>,weights: Move.Holding<Tile.Tile8>,bias: Cells()) -> Completed:
  Shape{+positions,+inner,+channels,+grid} = shape
  +in_cells = (positions * Gemm.blocks(inner) : U32)
  +weight_cells = (inner * Gemm.blocks(channels) : U32)
  panel_bands(~kernel,Rectangle{(first * positions * Gemm.blocks(channels) : U32),positions,inner,channels,False{},grid},
    Move.band(~Tile.Tile8,input,(first * in_cells : U32),in_cells,zero(),Move.spare(~Tile.Tile8,zero())),
    Move.band(~Tile.Tile8,weights,(first * weight_cells : U32),weight_cells,zero(),Move.spare(~Tile.Tile8,zero())),bias)

def panel_done(first: U32,count: U32,finished: Completed) -> Partition.Tree<&1,Pieces.Pieces(~Tile.Tile8),Unit> & Cells():
  Completed{pieces,input,bias} = finished
  (Partition.Leaf{pieces,first,count,Unit{}},bias)

def panel_leaf(~kernel: Kernel(),shape: Shape,first: U32,count: U32,operands: Pieces.Paired<Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>>,bias: Cells()) ->
  Partition.Tree<&1,Pieces.Pieces(~Tile.Tile8),Unit> & Cells():
  match operands:
    case Pieces.Both{input,weights}:
      +first = first
      panel_done(first,count,panel_rectangle(~kernel,shape,first,input,weights,bias))
    case Pieces.Only{input,own_first,own_count}:
      (Partition.Leaf{Partition.Leaf{Tensors.buffer(~Tile.Tile8,zero(),0),0,0,Unit{}},first,count,Unit{}},bias)

def panel_tasks(~kernel: Kernel(),+shape: Shape,+ranges: Pieces.Ranges,+input_grid: Grid.Grid,+weight_grid: Grid.Grid,decision: Tasks.Decision,
  input: Pieces.Pieces(~Tile.Tile8),weights: Pieces.Pieces(~Tile.Tile8),bias: Cells()) -> Pieces.Pieces(~Tile.Tile8):
  Shape{positions,inner,channels,+grid} = shape
  Tasks.Decision{+parallel,+tasks} = decision
  Tensors.placed(~Tile.Tile8,Pieces.ranges_of_grid(grid),zero(),
    Pieces.grouped_with(~Pieces.Paired<Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>>,~Pieces.Pieces(~Tile.Tile8),~Cells(),~Shape,
      ~Clone.clone(~Tile.Tile8),~panel_leaf(~kernel),tasks,parallel,shape,
      Pieces.zip(~Move.Holding<Tile.Tile8>,~Move.Holding<Tile.Tile8>,
        Halo.windows(~Tile.Tile8,~Shape,~panel_left,shape,input_grid,zero(),ranges,input),
        Halo.windows(~Tile.Tile8,~Shape,~panel_right,shape,weight_grid,zero(),ranges,weights)),bias))

def panels(~kernel: Kernel(),+count: U32,+rows: U32,+inner: U32,+outputs: U32,+workers: U32,pieces: Pieces.Pieces(~Tile.Tile8),weights: Pieces.Pieces(~Tile.Tile8),bias: Cells()) -> Pieces.Pieces(~Tile.Tile8):
  +groups = Gemm.blocks(outputs)
  +grid = Tensors.grid_of([count,rows,groups])
  +ranges = Tasks.component_ranges(count)
  panel_tasks(~kernel,Shape{rows,inner,outputs,grid},ranges,Tensors.grid_of([count,rows,Gemm.blocks(inner)]),Tensors.grid_of([count,inner,groups]),
    Tasks.decide_over(ranges,count,workers,1,U32.to_f32((rows * groups * inner : U32)),0.0),pieces,weights,bias)
