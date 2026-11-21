# Operations over the k x k windows of the last two axes of image cells
# [N,H,W,groups]. A task holds its input pieces and the halo before them
# (Halo.windows) and computes its output pieces one output row at a time: every
# kernel row reads one input row, lent from the piece that holds it, and
# updates the cells of the output row in place. Every output cell sees its
# samples in the order of the window. Pooling folds the k samples of every
# kernel row in increasing kx from pad and the output folds the k rows in
# increasing ky from pad; a sample outside the input reads pad, so pad should
# be the identity of the operation. Weighting adds every sample times its
# weight tile to the output from +0 in row-major order of the taps, then the
# bias tile of its group.
import Base
import ./kernel_tile.bend as Tile
import ./kernel_cost.bend as Cost
import ./default_kernel_cost.bend as DefaultCost
import ./storage_buffer.bend as Storage
import ./tensor_move.bend as Move
import ./tensor_access.bend as Access
import ./tensor_operations.bend as Operations
import ./tensor.bend as Tensors
import ./tensor_grid.bend as Grid
import ./tensor_pieces.bend as Pieces
import ./tensor_tasks.bend as Tasks
import ./tensor_halo.bend as Halo
import ./tensor_layout.bend as Layout
import ./tensor_tiled_map.bend as TiledMap
import ./tensor_view.bend as Views
import ./traversal_partition.bend as Partition

def out_extent(+extent: U32,+kernel: U32,+stride: U32,+padding: U32) -> U32:
  ((extent + 2 * padding - kernel : U32) / U32.max(stride,1) + 1 : U32)

def geometry(+shape: List<&2,U32>,+kernel: U32,+stride: U32,+padding: U32) -> Access.Pool:
  +rank = List.length(&2,U32,shape)
  +height = Layout.extent_at(shape,Nat.sub(rank,2n))
  +width = Layout.extent_at(shape,Nat.sub(rank,1n))
  Access.Pool{height,width,kernel,stride,padding,out_extent(height,kernel,stride,padding),out_extent(width,kernel,stride,padding)}

# The column-swept planes keep the rows of the input and have the columns of
# the output.
def middle_shape(+shape: List<&2,U32>,pool: Access.Pool) -> List<&2,U32>:
  Access.Pool{height,width,kernel,stride,padding,out_height,out_width} = pool
  List.append(&2,U32,List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),2n)),[height,out_width])

def output_shape(+shape: List<&2,U32>,pool: Access.Pool) -> List<&2,U32>:
  Access.Pool{height,width,kernel,stride,padding,out_height,out_width} = pool
  List.append(&2,U32,List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),2n)),[out_height,out_width])

# A window needs planes before the last two axes, positive extents and
# windows that fit them, and counts of the column-swept planes and of the
# output.
def valid(+shape: List<&2,U32>,+pool: Access.Pool) -> Bool:
  Access.Pool{+height,+width,kernel,stride,padding,out_height,out_width} = pool
  Operations.pool_geometry(pool) && U32.is_lt(0,height) && U32.is_lt(0,width) &&
    Views.has_count(Tensors.count_of(middle_shape(shape,pool))) && Views.has_count(Tensors.count_of(output_shape(shape,pool)))


# ---- Windows over image cells ----

# The window of one position: kernel x kernel taps over images of height
# rows of width positions, groups cells a position.
type Taps is Data:
  Taps{kernel: U32,stride: U32,padding: U32,height: U32,width: U32,groups: U32}

# What a window does with its samples: pooling folds every kernel row from
# pad and the output folds the rows from pad, weighting adds every sample
# times its weight tile to the output from +0 and then the bias.
type Sampling is Data:
  Pooling{pad: Tile.Tile8}
  Weighting{}

def initial(kind: Sampling) -> Tile.Tile8:
  match kind:
    case Pooling{+pad}: pad
    case Weighting{}: Tile.filled(0.0)

def sample_pad(kind: Sampling) -> Tile.Tile8:
  match kind:
    case Pooling{+pad}: pad
    case Weighting{}: Tile.filled(0.0)

def weighted(kind: Sampling) -> Bool:
  match kind:
    case Pooling{pad}: False{}
    case Weighting{}: True{}

# A window operation of a kind over image cells [N,H,W,groups] whose output
# cells are [N,out_height,out_width,groups]. The parameters of weighting are
# the weight tiles of every group, kernel by kernel, and then its bias tiles.
type Windowing is Data:
  Windowing{kind: Sampling,taps: Taps,lanes: U32,out_height: U32,out_width: U32,grid: Grid.Grid}

# The input cells the output cells [first, first + count) read: from the
# first stored row the first output row reads to the end of the last one.
def reach(+windowing: Windowing,+first: U32,+count: U32) -> Tasks.Span:
  Windowing{kind,Taps{+kernel,+stride,+padding,+height,+width,+groups},lanes,+out_height,+out_width,grid} = windowing
  +first_row = (first / U32.max(groups,1) / U32.max(out_width,1) : U32)
  +last_row = (((first + count) / U32.max(groups,1) - 1) / U32.max(out_width,1) : U32)
  +top = ((first_row % U32.max(out_height,1)) * stride : U32)
  +bottom = ((last_row % U32.max(out_height,1)) * stride + kernel : U32)
  +row = (width * groups : U32)
  +low = ((first_row / U32.max(out_height,1) * height + Bool.pick(U32,U32.is_le(padding,top),U32.min(height,(top - padding : U32)),0)) * row : U32)
  +high = ((last_row / U32.max(out_height,1) * height + Bool.pick(U32,U32.is_lt(padding,bottom),U32.min(height,(bottom - padding : U32)),0)) * row : U32)
  Tasks.Span{low,(U32.max(high,low) - low : U32)}

# One kernel row of the cells of one output row: the window of every output
# column starts left before padding, the row is stored or reads pad, its
# cells start at base of the row buffer, and its weights at weights.
type Line is Data:
  Line{taps: Taps,stored: Bool,base: U32,left: U32,weights: U32}

# The sample of kernel column column of the window of group group: pad
# outside the row.
def sample(+line: Line,+column: U32,+group: U32,+pad: Tile.Tile8,row: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8> & Tile.Tile8:
  Line{Taps{kernel,stride,+padding,height,+width,+groups},+stored,+base,+left,weights} = line
  +x = (left + column - padding : U32)
  Storage.read_masked(~Tile.Tile8,pad,stored && U32.is_lt(x,width),row,(base + x * groups + group : U32))

# A kernel row folded: the output so far and, for pooling, the row so far.
type Folded is Data:
  Folded{output: Tile.Tile8,line: Tile.Tile8}

def tapped(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,kind: Sampling,folded: Folded,value: Tile.Tile8,weight: Tile.Tile8) -> Folded:
  match kind folded:
    case Pooling{pad} Folded{output,line}: Folded{output,operation(line,value)}
    case Weighting{} Folded{output,line}: Folded{Tile.update_lanes(output,value,weight),line}

def row_closed(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,kind: Sampling,folded: Folded) -> Tile.Tile8:
  match kind folded:
    case Pooling{pad} Folded{output,line}: operation(output,line)
    case Weighting{} Folded{output,line}: output

type Tapping is Type:
  Tapping{row: Storage.Buffer<Tile.Tile8>,parameters: Storage.Buffer<Tile.Tile8>,folded: Folded}

def tap_read(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+kind: Sampling,folded: Folded,value: Tile.Tile8,read: Storage.Buffer<Tile.Tile8> & Tile.Tile8,row: Storage.Buffer<Tile.Tile8>) -> Tapping:
  (parameters,weight) = read
  Tapping{row,parameters,tapped(~operation,kind,folded,value,weight)}

def tap_sampled(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+kind: Sampling,+weight: U32,folded: Folded,parameters: Storage.Buffer<Tile.Tile8>,
  read: Storage.Buffer<Tile.Tile8> & Tile.Tile8) -> Tapping:
  (row,value) = read
  tap_read(~operation,kind,folded,value,Storage.read_masked(~Tile.Tile8,Tile.filled(0.0),weighted(kind),parameters,weight),row)

# The kernel columns of a kernel row from column on, for group group.
def taps(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,fuel: Nat,+column: U32,+kind: Sampling,+line: Line,+group: U32,state: Tapping) -> Tapping:
  match fuel:
    case 0n: state
    case 1n+rest:
      Line{Taps{+kernel,stride,padding,height,width,groups},stored,base,left,+weights} = line
      Tapping{row,parameters,folded} = state
      taps(~operation,rest,(column + 1 : U32),kind,line,group,
        tap_sampled(~operation,kind,(weights + group * kernel * kernel + column : U32),folded,parameters,sample(line,column,group,sample_pad(kind),row)))

type Swept is Type:
  Swept{row: Storage.Buffer<Tile.Tile8>,parameters: Storage.Buffer<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>}

# A task between kernel rows: the pieces it holds, the parameters and the
# output.
type Rowed is Type:
  Rowed{window: Move.Holding<Tile.Tile8>,parameters: Storage.Buffer<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>}

def cell_swept(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+kind: Sampling,+index: U32,output: Storage.Buffer<Tile.Tile8>,state: Tapping) -> Swept:
  Tapping{row,parameters,folded} = state
  Swept{row,parameters,Storage.write(~Tile.Tile8,index,row_closed(~operation,kind,folded),output)}

def cell_started(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+kind: Sampling,+line: Line,+group: U32,+index: U32,row: Storage.Buffer<Tile.Tile8>,parameters: Storage.Buffer<Tile.Tile8>,
  read: Storage.Buffer<Tile.Tile8> & Tile.Tile8) -> Swept:
  Line{Taps{+kernel,stride,padding,height,width,groups},stored,base,left,weights} = line
  (output,value) = read
  cell_swept(~operation,kind,index,output,taps(~operation,U32.to_nat(kernel),0,kind,line,group,Tapping{row,parameters,Folded{value,sample_pad(kind)}}))

# One kernel row applied to the output cells from cell on of an output row
# that starts at output index start and column column.
def cells(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,fuel: Nat,+cell: U32,+kind: Sampling,+line: Line,+start: U32,state: Swept) -> Swept:
  match fuel:
    case 0n: state
    case 1n+rest:
      Line{Taps{kernel,+stride,padding,height,width,+groups},+stored,+base,+left,+weights} = line
      Swept{row,parameters,output} = state
      +column = (cell / U32.max(groups,1) : U32)
      cells(~operation,rest,(cell + 1 : U32),kind,line,start,
        cell_started(~operation,kind,Line{Taps{kernel,stride,padding,height,width,groups},stored,base,(left + column * stride : U32),weights},(cell % U32.max(groups,1) : U32),
          (start + cell : U32),row,parameters,Storage.read(~Tile.Tile8,output,(start + cell : U32))))

# The cells of a row of positions of one output row: its first output cell,
# its cells, the input row of its image and the first column its windows
# start at before padding.
type Segment is Data:
  Segment{start: U32,count: U32,image: U32,top: U32,left: U32}

def swept_back(lent: Move.Lent<Tile.Tile8>,swept: Swept) -> Rowed:
  Swept{row,parameters,output} = swept
  Rowed{Move.returned_holding(~Tile.Tile8,lent,row),parameters,output}

# One kernel row over a segment, its input row lent from the window.
def row_banded(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+kind: Sampling,+taps: Taps,+segment: Segment,+stored: Bool,+weights: U32,banded: Move.Band<Tile.Tile8>,
  parameters: Storage.Buffer<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>) -> Rowed:
  Segment{+start,+count,image,top,+left} = segment
  Move.Band{lent,row,+base} = banded
  swept_back(lent,cells(~operation,U32.to_nat(count),0,kind,Line{taps,stored,base,left,weights},start,Swept{row,parameters,output}))

def rowed_unstored(window: Move.Holding<Tile.Tile8>,swept: Swept) -> Rowed:
  Swept{row,parameters,output} = swept
  Rowed{window,parameters,output}

def row_unstored(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+kind: Sampling,+taps: Taps,+segment: Segment,+weights: U32,window: Move.Holding<Tile.Tile8>,
  parameters: Storage.Buffer<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>) -> Rowed:
  Segment{+start,+count,image,top,+left} = segment
  rowed_unstored(window,cells(~operation,U32.to_nat(count),0,kind,Line{taps,False{},0,left,weights},start,Swept{Storage.allocate(~Tile.Tile8,0n,Tile.filled(0.0),0n),parameters,output}))

def row_applied(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,stored: Bool,+kind: Sampling,+taps: Taps,+segment: Segment,+row: U32,+weights: U32,window: Move.Holding<Tile.Tile8>,
  parameters: Storage.Buffer<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>) -> Rowed:
  match stored:
    case True{}:
      Taps{kernel,stride,padding,+height,+width,+groups} = taps
      Segment{start,count,+image,top,left} = segment
      row_banded(~operation,kind,taps,segment,True{},weights,Move.band(~Tile.Tile8,window,((image * height + row) * width * groups : U32),(width * groups : U32),sample_pad(kind),Move.spare(~Tile.Tile8,sample_pad(kind))),
        parameters,output)
    case False{}: row_unstored(~operation,kind,taps,segment,weights,window,parameters,output)

def row_applied_of(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,stored: Bool,+kind: Sampling,+taps: Taps,+segment: Segment,+row: U32,+weights: U32,rowed: Rowed) -> Rowed:
  Rowed{window,parameters,output} = rowed
  row_applied(~operation,stored,kind,taps,segment,row,weights,window,parameters,output)

# The kernel rows from kernel_row on over a segment, in order.
def kernel_rows(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,fuel: Nat,+kernel_row: U32,+kind: Sampling,+taps: Taps,+segment: Segment,rowed: Rowed) -> Rowed:
  match fuel:
    case 0n: rowed
    case 1n+rest:
      Taps{+kernel,stride,+padding,+height,width,groups} = taps
      Segment{start,count,image,+top,left} = segment
      +y = (top + kernel_row - padding : U32)
      kernel_rows(~operation,rest,(kernel_row + 1 : U32),kind,taps,segment,
        row_applied_of(~operation,U32.is_lt(y,height),kind,taps,segment,y,(kernel_row * kernel : U32),rowed))

# The output positions from position to end, one output row at a time, of an
# output piece whose first cell is first. more says position is before end.
def segments(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,fuel: Nat,more: Bool,+position: U32,+end: U32,+first: U32,+windowing: Windowing,rowed: Rowed) -> Rowed:
  match fuel more:
    case 0n more: rowed
    case 1n+rest False{}: rowed
    case 1n+rest True{}:
      Windowing{+kind,+taps,lanes,+out_height,+out_width,grid} = windowing
      Taps{+kernel,+stride,padding,height,width,+groups} = taps
      +row = (position / U32.max(out_width,1) : U32)
      +stop = U32.min(end,((row + 1) * out_width : U32))
      segments(~operation,rest,U32.is_lt(stop,end),stop,end,first,windowing,
        kernel_rows(~operation,U32.to_nat(kernel),0,kind,taps,
          Segment{(position * groups - first : U32),((stop - position) * groups : U32),(row / U32.max(out_height,1) : U32),((row % U32.max(out_height,1)) * stride : U32),
            ((position % U32.max(out_width,1)) * stride : U32)},rowed))

def sum_biased(+cell: U32,value: Tile.Tile8,read: Storage.Buffer<Tile.Tile8> & Tile.Tile8) -> Storage.Buffer<Tile.Tile8>:
  (output,sum) = read
  Storage.write(~Tile.Tile8,cell,Tile.plus(sum,value),output)

def bias_read(+cell: U32,read: Storage.Buffer<Tile.Tile8> & Tile.Tile8,output: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  (parameters,value) = read
  (parameters,sum_biased(cell,value,Storage.read(~Tile.Tile8,output,cell)))

def cell_biased(+cell: U32,+index: U32,pair: Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  (parameters,output) = pair
  bias_read(cell,Storage.read(~Tile.Tile8,parameters,index),output)

# Weighting adds the bias tile of its group to every output cell from cell on.
def biased(fuel: Nat,+cell: U32,+groups: U32,+bias: U32,pair: Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) ->
  Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  match fuel:
    case 0n: pair
    case 1n+rest: biased(rest,(cell + 1 : U32),groups,bias,cell_biased(cell,(bias + cell % U32.max(groups,1) : U32),pair))

def finish_when(weighting: Bool,+count: U32,+groups: U32,+bias: U32,parameters: Storage.Buffer<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>) ->
  Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  match weighting:
    case True{}: biased(U32.to_nat(count),0,groups,bias,(parameters,output))
    case False{}: (parameters,output)

type Computed is Type:
  Computed{piece: Pieces.Pieces(~Tile.Tile8),window: Move.Holding<Tile.Tile8>,parameters: Storage.Buffer<Tile.Tile8>}

def finished_piece(+windowing: Windowing,+first: U32,+count: U32,window: Move.Holding<Tile.Tile8>,pair: Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) -> Computed:
  Windowing{kind,Taps{kernel,stride,padding,height,width,+groups},+lanes,out_height,out_width,grid} = windowing
  (parameters,output) = pair
  Computed{Partition.Leaf{TiledMap.padded(TiledMap.Padding{lanes,groups,0,Grid.Innermost{}},first,count,output),first,count,Unit{}},window,parameters}

def swept_piece(+windowing: Windowing,+first: U32,+count: U32,rowed: Rowed) -> Computed:
  Windowing{+kind,Taps{+kernel,stride,padding,height,width,+groups},lanes,out_height,out_width,grid} = windowing
  Rowed{window,parameters,output} = rowed
  finished_piece(windowing,first,count,window,finish_when(weighted(kind),count,groups,(groups * kernel * kernel : U32),parameters,output))

# The output piece of first cell first and count cells; the lanes past the
# extent are +0.
def computed(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+windowing: Windowing,+first: U32,+count: U32,window: Move.Holding<Tile.Tile8>,parameters: Storage.Buffer<Tile.Tile8>) ->
  Computed:
  Windowing{+kind,Taps{kernel,stride,padding,height,width,+groups},lanes,out_height,out_width,grid} = windowing
  +position = (first / U32.max(groups,1) : U32)
  +end = ((first + count) / U32.max(groups,1) : U32)
  swept_piece(windowing,first,count,
    segments(~operation,U32.to_nat((end - position : U32)),U32.is_lt(position,end),position,end,first,windowing,Rowed{window,parameters,Tensors.buffer(~Tile.Tile8,initial(kind),count)}))

def Held() -> Type:
  Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>

def computed_walked(lefts: Pieces.Lefts<Storage.Buffer<Tile.Tile8>>,result: Computed) -> Pieces.Walked<Storage.Buffer<Tile.Tile8>,Held()>:
  Computed{piece,window,parameters} = result
  Pieces.Walked{piece,lefts,(window,parameters)}

# The output pieces from piece on, remaining more after it. Each piece waits
# on the left for the pieces after it, so the task calls no closure.
def task_tree(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,remaining: Nat,+windowing: Windowing,+grid: Grid.Grid,+piece: U32,
  ready: Pieces.Walking<Storage.Buffer<Tile.Tile8>,Held()>) -> Pieces.Walked<Storage.Buffer<Tile.Tile8>,Held()>:
  match remaining:
    case 0n:
      Pieces.Walking{lefts,held} = ready
      (window,parameters) = held
      computed_walked(lefts,computed(~operation,windowing,Grid.start(grid,piece),Grid.span(grid,piece),window,parameters))
    case 1n+rest:
      Pieces.Walking{lefts,held} = ready
      (window,parameters) = held
      Pieces.joined(~Storage.Buffer<Tile.Tile8>,~Held(),task_tree(~operation,rest,windowing,grid,(piece + 1 : U32),
        Pieces.pushed(~Storage.Buffer<Tile.Tile8>,~Held(),computed_walked(lefts,computed(~operation,windowing,Grid.start(grid,piece),Grid.span(grid,piece),window,parameters)))))

def task_finished(+first: U32,+count: U32,walked: Pieces.Walked<Storage.Buffer<Tile.Tile8>,Held()>) -> Partition.Tree<&1,Pieces.Pieces(~Tile.Tile8),Unit> & Storage.Buffer<Tile.Tile8>:
  Pieces.Walked{pieces,lefts,held} = walked
  (window,parameters) = held
  (Partition.Leaf{pieces,first,count,Unit{}},parameters)

def grid_of(windowing: Windowing) -> Grid.Grid:
  Windowing{kind,taps,lanes,out_height,out_width,grid} = windowing
  grid

# A task: its output pieces from the input pieces it holds.
def task_leaf(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,windowing: Windowing,first: U32,count: U32,window: Move.Holding<Tile.Tile8>,parameters: Storage.Buffer<Tile.Tile8>) ->
  Partition.Tree<&1,Pieces.Pieces(~Tile.Tile8),Unit> & Storage.Buffer<Tile.Tile8>:
  +windowing = windowing
  +first = first
  +count = count
  +grid = grid_of(windowing)
  +piece = Grid.piece_of(grid,first)
  task_finished(first,count,task_tree(~operation,Grid.later_pieces(grid,first,count),windowing,grid,piece,Pieces.Walking{Pieces.NoLefts{},(window,parameters)}))

def reach_of(+windowing: Windowing,+first: U32,+count: U32) -> Tasks.Span:
  reach(windowing,first,count)

def sample_pad_of(windowing: Windowing) -> Tile.Tile8:
  Windowing{kind,taps,lanes,out_height,out_width,grid} = windowing
  sample_pad(kind)

def windowed_tasks(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+windowing: Windowing,+input_grid: Grid.Grid,decision: Tasks.Decision,pieces: Pieces.Pieces(~Tile.Tile8),
  parameters: Storage.Buffer<Tile.Tile8>) -> Pieces.Pieces(~Tile.Tile8):
  Tasks.Decision{+parallel,+tasks} = decision
  Tensors.placed(~Tile.Tile8,Pieces.ranges_of_grid(grid_of(windowing)),Tile.filled(0.0),
    Pieces.grouped_with(~Move.Holding<Tile.Tile8>,~Pieces.Pieces(~Tile.Tile8),~Storage.Buffer<Tile.Tile8>,~Windowing,~Tensors.shared_buffer(~Tile.Tile8),~task_leaf(~operation),tasks,parallel,windowing,
      Halo.windows(~Tile.Tile8,~Windowing,~reach_of,windowing,input_grid,sample_pad_of(windowing),tasks,pieces),parameters))

def windowed_when(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,empty: Bool,+windowing: Windowing,+input_grid: Grid.Grid,decision: Tasks.Decision,pieces: Pieces.Pieces(~Tile.Tile8),
  parameters: Storage.Buffer<Tile.Tile8>) -> Pieces.Pieces(~Tile.Tile8):
  match empty:
    case True{}: Tensors.filled_grid(~Tile.Tile8,grid_of(windowing),Tile.filled(0.0))
    case False{}: windowed_tasks(~operation,windowing,input_grid,decision,pieces,parameters)

# An empty output grid has no task.
def windowed_decided(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+windowing: Windowing,+input_grid: Grid.Grid,decision: Tasks.Decision,pieces: Pieces.Pieces(~Tile.Tile8),
  parameters: Storage.Buffer<Tile.Tile8>) -> Pieces.Pieces(~Tile.Tile8):
  windowed_when(~operation,U32.is_zero(Grid.count_of(grid_of(windowing))),windowing,input_grid,decision,pieces,parameters)

# What a task beyond the first adds: its own parameters and a halo of the rows
# a window reaches past the step.
def extra_of(+windowing: Windowing,+parameters: U32) -> Tasks.Extra:
  Windowing{kind,Taps{+kernel,+stride,padding,height,+width,+groups},lanes,out_height,out_width,grid} = windowing
  +halo = Bool.pick(U32,U32.is_le(kernel,stride),0,((kernel - stride) * width * groups : U32))
  Tasks.Extra{(U32.to_f32(parameters) * 32.0 : F32),(U32.to_f32(halo) * Cost.copy_ns(64.0,DefaultCost.configuration()) : F32),0.0}

# The output pieces of a window operation over the input pieces of input_grid:
# every task reads the pieces it holds. rows and steps count one output
# cell, parameters counts the parameter tiles.
def windowed(~operation: Tile.Tile8 -> Tile.Tile8 -> Tile.Tile8,+windowing: Windowing,+input_grid: Grid.Grid,+rows: F32,+steps: F32,+workers: U32,+count: U32,
  pieces: Pieces.Pieces(~Tile.Tile8),parameters: Storage.Buffer<Tile.Tile8>) -> Pieces.Pieces(~Tile.Tile8):
  windowed_decided(~operation,windowing,input_grid,Tasks.decide(grid_of(windowing),workers,rows,steps,extra_of(windowing,count)),pieces,parameters)
