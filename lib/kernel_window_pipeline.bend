# A convolution over the cells of a batch of images. Every output piece is
# computed by calls of the window kernel, one call for the positions of one
# image, each packing the input rows its positions read. A task holds its
# input pieces and the halo before them (Halo.windows); a call reads its rows
# from the held piece that holds them all, or from a buffer of them assembled
# from the pieces that hold its parts.
import Base
import ./storage_buffer.bend as Storage
import ./storage_clone.bend as Clone
import ./tensor_move.bend as Move
import ./storage_copy.bend as Copy
import ./kernel_tile.bend as Tile
import ./kernel_gemm.bend as Gemm
import ./kernel_gemm_buffer.bend as GemmBuffer
import ./kernel_window.bend as KernelWindow
import ./kernel_pipeline.bend as Pipeline
import ./kernel_spatial_packing.bend as Packing
import ./math_activation.bend as Activations
import ./tensor.bend as Tensors
import ./tensor_grid.bend as Grid
import ./tensor_pieces.bend as Pieces
import ./tensor_tasks.bend as Tasks
import ./tensor_halo.bend as Halo
import ./traversal_partition.bend as Partition

# The image extents of the input and the output, the window, the output
# channels, the inner extent of the product, the cells of a position in the
# input and the output, and the activation.
type Convolution is Data:
  Convolution{height: U32,width: U32,kernel: U32,stride: U32,padding: U32,output_height: U32,output_width: U32,channels: U32,inner: U32,input_groups: U32,
    output_groups: U32,activation: Activations.Activation}

def plane_of(convolution: Convolution) -> U32:
  Convolution{h,w,k,s,p,+oh,+ow,channels,inner,gi,go,activation} = convolution
  (oh * ow : U32)

def output_groups_of(convolution: Convolution) -> U32:
  Convolution{h,w,k,s,p,oh,ow,channels,inner,gi,go,activation} = convolution
  go

def image_cells(convolution: Convolution) -> U32:
  Convolution{+h,+w,k,s,p,oh,ow,channels,inner,+gi,go,activation} = convolution
  (h * w * gi : U32)

def output_image_cells(convolution: Convolution) -> U32:
  Convolution{h,w,k,s,p,+oh,+ow,channels,inner,gi,+go,activation} = convolution
  (oh * ow * go : U32)

def activation_of(convolution: Convolution) -> Activations.Activation:
  Convolution{h,w,k,s,p,oh,ow,channels,inner,gi,go,activation} = convolution
  activation

def channels_of(convolution: Convolution) -> U32:
  Convolution{h,w,k,s,p,oh,ow,channels,inner,gi,go,activation} = convolution
  channels

def inner_of(convolution: Convolution) -> U32:
  Convolution{h,w,k,s,p,oh,ow,channels,inner,gi,go,activation} = convolution
  inner

# The rows the positions [first, first + count) of an image read.
def band_of(+convolution: Convolution,+first: U32,+count: U32) -> KernelWindow.Band:
  Convolution{+h,w,+k,+s,+p,oh,+ow,channels,inner,gi,go,activation} = convolution
  KernelWindow.band(0,h,h,k,s,p,ow,first,count)

# ---- The task skeleton ----

# One call of a kernel: the positions [first, first + count) of an image,
# whose rows are the band of input rows held from cell base on, written from
# output position offset on.
type Call is Data:
  Call{band: KernelWindow.Band,image: U32,first: U32,offset: U32,count: U32,base: U32}

# A task between calls: the pieces it holds, the spare its bands are
# assembled in and the work of the kernel.
type Calling<-Work: Type> is Type:
  Calling{window: Move.Holding<Tile.Tile8>,spare: Move.Spare<Tile.Tile8>,work: Work}

def called_back(~Work: Type,returned: Move.Holding<Tile.Tile8> & Move.Spare<Tile.Tile8>,work: Work) -> Calling<Work>:
  (window,spare) = returned
  Calling{window,spare,work}

def called(~Work: Type,lent: Move.Lent<Tile.Tile8>,computed: Storage.Buffer<Tile.Tile8> & Work) -> Calling<Work>:
  (input,work) = computed
  called_back(~Work,Move.returned(~Tile.Tile8,lent,input),work)

def call_banded(~Work: Type,~compute: @+convolution: Convolution -> @+call: Call -> Storage.Buffer<Tile.Tile8> -> Work -> Storage.Buffer<Tile.Tile8> & Work,+convolution: Convolution,
  +band: KernelWindow.Band,+image: U32,+first: U32,+offset: U32,+count: U32,banded: Move.Band<Tile.Tile8>,work: Work) -> Calling<Work>:
  Move.Band{lent,input,+base} = banded
  called(~Work,lent,compute(convolution,Call{band,image,first,offset,count,base},input,work))

def call_band(~Work: Type,~compute: @+convolution: Convolution -> @+call: Call -> Storage.Buffer<Tile.Tile8> -> Work -> Storage.Buffer<Tile.Tile8> & Work,+convolution: Convolution,
  +band: KernelWindow.Band,+image: U32,+first: U32,+count: U32,+offset: U32,state: Calling<Work>) -> Calling<Work>:
  Convolution{+h,+w,k,s,p,oh,ow,channels,inner,+gi,go,activation} = convolution
  KernelWindow.Band{+row_first,+rows} = band
  Calling{window,spare,work} = state
  call_banded(~Work,~compute,convolution,band,image,first,offset,count,Move.band(~Tile.Tile8,window,((image * h + row_first) * w * gi : U32),(rows * w * gi : U32),Tile.filled(0.0),spare),
    work)

# One call: positions [first, first + count) of an image, written from output
# position offset on.
def call(~Work: Type,~compute: @+convolution: Convolution -> @+call: Call -> Storage.Buffer<Tile.Tile8> -> Work -> Storage.Buffer<Tile.Tile8> & Work,+convolution: Convolution,
  +image: U32,+first: U32,+count: U32,+offset: U32,state: Calling<Work>) -> Calling<Work>:
  call_band(~Work,~compute,convolution,band_of(convolution,first,count),image,first,count,offset,state)

# The calls of output positions [position, end), one an image; start is the
# first position of the output.
def calls(~Work: Type,~compute: @+convolution: Convolution -> @+call: Call -> Storage.Buffer<Tile.Tile8> -> Work -> Storage.Buffer<Tile.Tile8> & Work,remaining: Nat,
  +convolution: Convolution,+position: U32,+end: U32,+start: U32,state: Calling<Work>) -> Calling<Work>:
  match remaining:
    case 0n: state
    case 1n+rest:
      +plane = U32.max(plane_of(convolution),1)
      +local = (position % plane : U32)
      +count = U32.min((plane - local : U32),(end - position : U32))
      calls(~Work,~compute,rest,convolution,(position + count : U32),end,start,call(~Work,~compute,convolution,(position / plane : U32),local,count,(position - start : U32),state))

def images_between(+plane: U32,+position: U32,+end: U32) -> U32:
  Bool.pick(U32,U32.is_lt(position,end),((end - 1) / plane - position / plane + 1 : U32),0)

# What a task keeps between its output pieces: the kernel's read-only
# operands and workspace, and the spare its bands are assembled in.
type Parameters<-Kept: Type> is Type:
  Parameters{kept: Kept,spare: Move.Spare<Tile.Tile8>}

def Operands(~Kept: Type) -> Type:
  Move.Holding<Tile.Tile8> & Parameters<Kept>

def owned(~Kept: Type,window: Move.Holding<Tile.Tile8>,spare: Move.Spare<Tile.Tile8>,finished: Storage.Buffer<Tile.Tile8> & Kept) -> Storage.Buffer<Tile.Tile8> & Operands(~Kept):
  (output,kept) = finished
  (output,(window,Parameters{kept,spare}))

def ended(~Kept: Type,~Work: Type,~finish: @+convolution: Convolution -> @+positions: U32 -> Work -> Storage.Buffer<Tile.Tile8> & Kept,+convolution: Convolution,+positions: U32,
  state: Calling<Work>) -> Storage.Buffer<Tile.Tile8> & Operands(~Kept):
  Calling{window,spare,work} = state
  owned(~Kept,window,spare,finish(convolution,positions,work))

# The output piece of first cell first and count cells: begin gives the work
# of its positions, the calls compute them and finish gives its cells.
def computed(~Kept: Type,~Work: Type,~begin: @+convolution: Convolution -> @+positions: U32 -> @+count: U32 -> Kept -> Work,
  ~compute: @+convolution: Convolution -> @+call: Call -> Storage.Buffer<Tile.Tile8> -> Work -> Storage.Buffer<Tile.Tile8> & Work,
  ~finish: @+convolution: Convolution -> @+positions: U32 -> Work -> Storage.Buffer<Tile.Tile8> & Kept,+convolution: Convolution,+first: U32,+count: U32,operands: Operands(~Kept)) ->
  Storage.Buffer<Tile.Tile8> & Operands(~Kept):
  (window,parameters) = operands
  Parameters{kept,spare} = parameters
  +groups = U32.max(output_groups_of(convolution),1)
  +position = (first / groups : U32)
  +end = ((first + count) / groups : U32)
  ended(~Kept,~Work,~finish,convolution,(end - position : U32),
    calls(~Work,~compute,U32.to_nat(images_between(U32.max(plane_of(convolution),1),position,end)),convolution,position,end,position,
      Calling{window,spare,begin(convolution,(end - position : U32),count,kept)}))

# The tasks after a split: the new half gets a spare of its own.
def split_parameters(~Kept: Type,spare: Move.Spare<Tile.Tile8>,kept: Kept & Kept) -> Parameters<Kept> & Parameters<Kept>:
  (left,right) = kept
  (Parameters{left,spare},Parameters{right,Move.spare(~Tile.Tile8,Tile.filled(0.0))})

def shared(~Kept: Type,~share: Kept -> Kept & Kept,parameters: Parameters<Kept>) -> Parameters<Kept> & Parameters<Kept>:
  Parameters{kept,spare} = parameters
  split_parameters(~Kept,spare,share(kept))

def height_of(convolution: Convolution) -> U32:
  Convolution{h,w,k,s,p,oh,ow,channels,inner,gi,go,activation} = convolution
  h

def reach_bands(+convolution: Convolution,first_band: KernelWindow.Band,end_band: KernelWindow.Band,+position: U32,+last: U32) -> Tasks.Span:
  KernelWindow.Band{+first_row,first_rows} = first_band
  KernelWindow.Band{+end_first,+end_rows} = end_band
  +plane = U32.max(plane_of(convolution),1)
  +height = height_of(convolution)
  +row = (image_cells(convolution) / U32.max(height,1) : U32)
  +low = ((position / plane * height + first_row) * row : U32)
  Tasks.Span{low,((last / plane * height + end_first + end_rows) * row - low : U32)}

# The input cells the output cells [first, first + count) read: from the
# first row the first position reads to the last row the last one reads.
def reach(+convolution: Convolution,+first: U32,+count: U32) -> Tasks.Span:
  +groups = U32.max(output_groups_of(convolution),1)
  +plane = U32.max(plane_of(convolution),1)
  +position = (first / groups : U32)
  +last = ((first + count) / groups - 1 : U32)
  +same = U32.is_eq((position / plane : U32),(last / plane : U32))
  +local = (position % plane : U32)
  reach_bands(convolution,band_of(convolution,local,Bool.pick(U32,same,(last - position + 1 : U32),(plane - local : U32))),
    band_of(convolution,Bool.pick(U32,same,local,0),Bool.pick(U32,same,(last - position + 1 : U32),(last % plane + 1 : U32))),position,last)

# What a task reads: the convolution and the output grid.
type Windowed is Data:
  Windowed{convolution: Convolution,grid: Grid.Grid}

# The cells of a piece that the task's cells [low, high) cover.
def clipped_first(+grid: Grid.Grid,+piece: U32,+low: U32) -> U32:
  U32.max(Grid.start(grid,piece),low)

def clipped_count(+grid: Grid.Grid,+piece: U32,+low: U32,+high: U32) -> U32:
  (U32.min((Grid.start(grid,piece) + Grid.span(grid,piece) : U32),high) - clipped_first(grid,piece,low) : U32)

# The cells [first, first + count) of the output piece piece and of the
# pieces after it that the task's cells [low, high) cover, remaining more.
# Each piece waits on the left for the pieces after it, so the task calls no
# closure.
def task_tree(~Kept: Type,~Work: Type,~begin: @+convolution: Convolution -> @+positions: U32 -> @+count: U32 -> Kept -> Work,
  ~compute: @+convolution: Convolution -> @+call: Call -> Storage.Buffer<Tile.Tile8> -> Work -> Storage.Buffer<Tile.Tile8> & Work,
  ~finish: @+convolution: Convolution -> @+positions: U32 -> Work -> Storage.Buffer<Tile.Tile8> & Kept,remaining: Nat,+convolution: Convolution,+grid: Grid.Grid,+piece: U32,
  +low: U32,+high: U32,+first: U32,+count: U32,ready: Pieces.Walking<Storage.Buffer<Tile.Tile8>,Operands(~Kept)>) -> Pieces.Walked<Storage.Buffer<Tile.Tile8>,Operands(~Kept)>:
  match remaining:
    case 0n:
      Pieces.Walking{lefts,operands} = ready
      Pieces.piece_walked(~Tile.Tile8,~Operands(~Kept),lefts,first,count,computed(~Kept,~Work,~begin,~compute,~finish,convolution,first,count,operands))
    case 1n+rest:
      Pieces.Walking{lefts,operands} = ready
      Pieces.joined(~Storage.Buffer<Tile.Tile8>,~Operands(~Kept),task_tree(~Kept,~Work,~begin,~compute,~finish,rest,convolution,grid,(piece + 1 : U32),low,high,
        clipped_first(grid,(piece + 1 : U32),low),clipped_count(grid,(piece + 1 : U32),low,high),
        Pieces.pushed(~Storage.Buffer<Tile.Tile8>,~Operands(~Kept),Pieces.piece_walked(~Tile.Tile8,~Operands(~Kept),lefts,first,count,
          computed(~Kept,~Work,~begin,~compute,~finish,convolution,first,count,operands)))))

def task_finished(~Kept: Type,+first: U32,+count: U32,walked: Pieces.Walked<Storage.Buffer<Tile.Tile8>,Operands(~Kept)>) ->
  Partition.Tree<&1,Pieces.Pieces(~Tile.Tile8),Unit> & Parameters<Kept>:
  Pieces.Walked{pieces,lefts,operands} = walked
  (window,parameters) = operands
  (Partition.Leaf{pieces,first,count,Unit{}},parameters)

# The output positions of a task's cells [first, first + count).
def task_positions(+convolution: Convolution,+first: U32,+count: U32) -> U32:
  +groups = U32.max(output_groups_of(convolution),1)
  ((first + count) / groups - first / groups : U32)

def prepared(~Kept: Type,~prepare: @+convolution: Convolution -> @+positions: U32 -> Kept -> Kept,+convolution: Convolution,+positions: U32,parameters: Parameters<Kept>) ->
  Parameters<Kept>:
  Parameters{kept,spare} = parameters
  Parameters{prepare(convolution,positions,kept),spare}

# A task: its output cells, whole pieces or a part of one, from the input
# pieces it holds.
def task_leaf(~Kept: Type,~Work: Type,~begin: @+convolution: Convolution -> @+positions: U32 -> @+count: U32 -> Kept -> Work,
  ~compute: @+convolution: Convolution -> @+call: Call -> Storage.Buffer<Tile.Tile8> -> Work -> Storage.Buffer<Tile.Tile8> & Work,
  ~finish: @+convolution: Convolution -> @+positions: U32 -> Work -> Storage.Buffer<Tile.Tile8> & Kept,~prepare: @+convolution: Convolution -> @+positions: U32 -> Kept -> Kept,context: Windowed,
  first: U32,count: U32,window: Move.Holding<Tile.Tile8>,parameters: Parameters<Kept>) -> Partition.Tree<&1,Pieces.Pieces(~Tile.Tile8),Unit> & Parameters<Kept>:
  Windowed{+convolution,+grid} = context
  +first = first
  +count = count
  +piece = Grid.piece_of(grid,first)
  +high = (first + count : U32)
  task_finished(~Kept,first,count,task_tree(~Kept,~Work,~begin,~compute,~finish,Grid.later_pieces(grid,first,count),convolution,grid,piece,first,high,
    clipped_first(grid,piece,first),clipped_count(grid,piece,first,high),Pieces.Walking{Pieces.NoLefts{},(window,prepared(~Kept,~prepare,convolution,task_positions(convolution,first,count),parameters))}))

def reach_of(+convolution: Convolution,+first: U32,+count: U32) -> Tasks.Span:
  reach(convolution,first,count)

# Every task reads the input pieces it holds and keeps its copy of the
# kernel's operands; the output pieces go to their places of the output grid.
def windowed_tasks(~Kept: Type,~Work: Type,~begin: @+convolution: Convolution -> @+positions: U32 -> @+count: U32 -> Kept -> Work,
  ~compute: @+convolution: Convolution -> @+call: Call -> Storage.Buffer<Tile.Tile8> -> Work -> Storage.Buffer<Tile.Tile8> & Work,
  ~finish: @+convolution: Convolution -> @+positions: U32 -> Work -> Storage.Buffer<Tile.Tile8> & Kept,~prepare: @+convolution: Convolution -> @+positions: U32 -> Kept -> Kept,
  ~share: Kept -> Kept & Kept,+convolution: Convolution,+input_grid: Grid.Grid,+grid: Grid.Grid,decision: Tasks.Decision,pieces: Pieces.Pieces(~Tile.Tile8),kept: Kept) ->
  Pieces.Pieces(~Tile.Tile8):
  Tasks.Decision{+parallel,+tasks} = decision
  Tensors.placed(~Tile.Tile8,Pieces.ranges_of_grid(grid),Tile.filled(0.0),
    Pieces.grouped_with(~Move.Holding<Tile.Tile8>,~Pieces.Pieces(~Tile.Tile8),~Parameters<Kept>,~Windowed,~shared(~Kept,~share),
      ~task_leaf(~Kept,~Work,~begin,~compute,~finish,~prepare),tasks,parallel,Windowed{convolution,grid},
      Halo.windows(~Tile.Tile8,~Convolution,~reach_of,convolution,input_grid,Tile.filled(0.0),tasks,pieces),Parameters{kept,Move.spare(~Tile.Tile8,Tile.filled(0.0))}))

# ---- Packed: the rows of the product are packed from the band ----

# The weights and bias of a packed task and its two packing workspaces: the
# planes of a call and the tiles of a block.
type PackedKept is Type:
  PackedKept{weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<Tile.Tile8>,planes: Storage.Buffer<Tile.Tile16>,tiles: Storage.Buffer<Tile.Tile8>}

# A packed output piece: the kept operands and its output cells.
type PackedWork is Type:
  PackedWork{weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<Tile.Tile8>,planes: Storage.Buffer<Tile.Tile16>,tiles: Storage.Buffer<Tile.Tile8>,
    output: Storage.Buffer<Tile.Tile8>}

def packed_returned(kept: Pipeline.Kept) -> Storage.Buffer<Tile.Tile8> & PackedWork:
  Pipeline.Kept{input,weights,bias,planes,tiles,output} = kept
  (input,PackedWork{weights,bias,planes,tiles,output})

def stencil_of(+convolution: Convolution,+base: U32,+rows: U32) -> Packing.Stencil:
  Convolution{h,+w,+k,+s,+p,oh,+ow,channels,+inner,+gi,go,activation} = convolution
  Packing.Stencil{base,rows,w,k,k,s,(0 - p : U32),(0 - p : U32),ow,inner,gi}

# A packed call packs the rows its positions read from the band and
# multiplies them by the weights.
def packed_call(+convolution: Convolution,+call: Call,input: Storage.Buffer<Tile.Tile8>,work: PackedWork) -> Storage.Buffer<Tile.Tile8> & PackedWork:
  Convolution{h,w,k,+s,p,oh,+ow,+channels,inner,gi,go,activation} = convolution
  Call{band,image,+first,+offset,+count,+base} = call
  KernelWindow.Band{+row_first,+rows} = band
  PackedWork{weights,bias,planes,tiles,output} = work
  packed_returned(Pipeline.blocked_into(Pipeline.WindowWork{channels,(offset + count : U32),stencil_of(convolution,base,rows)},
    KernelWindow.start(first,s,ow,row_first),offset,input,weights,bias,planes,tiles,output))

def packed_begin(+convolution: Convolution,+positions: U32,+count: U32,kept: PackedKept) -> PackedWork:
  PackedKept{weights,bias,planes,tiles} = kept
  PackedWork{weights,bias,planes,tiles,Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count)}

def packed_finish(~custom: F32 -> F32,+convolution: Convolution,+positions: U32,work: PackedWork) -> Storage.Buffer<Tile.Tile8> & PackedKept:
  PackedWork{weights,bias,planes,tiles,output} = work
  (Pipeline.finished(~custom,activation_of(convolution),channels_of(convolution),positions,output),PackedKept{weights,bias,planes,tiles})

# The workspaces of the largest call of a task of positions positions: a call
# holds the positions of one image.
def call_positions(+convolution: Convolution,+positions: U32) -> U32:
  Convolution{h,w,k,s,p,+oh,+ow,channels,inner,gi,go,activation} = convolution
  U32.min(positions,(oh * ow : U32))

def plane_size(+convolution: Convolution,+positions: U32) -> U32:
  Packing.plane_bound(stencil_of(convolution,0,0),KernelWindow.padded(call_positions(convolution,positions)))

def workspace_size(+convolution: Convolution,+positions: U32) -> U32:
  Convolution{h,w,k,s,p,oh,ow,channels,+inner,gi,go,activation} = convolution
  Pipeline.workspace_tiles(Pipeline.block_tiles(),inner,call_positions(convolution,positions))

# The task's own workspaces in place of the empty ones it was given, so the
# tasks allocate in parallel.
def packed_prepare(+convolution: Convolution,+positions: U32,kept: PackedKept) -> PackedKept:
  PackedKept{weights,bias,planes,tiles} = kept
  PackedKept{weights,bias,Tensors.buffer(~Tile.Tile16,Tile.filled_pair(0.0),plane_size(convolution,positions)),
    Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),workspace_size(convolution,positions))}

def packed_shared(weights: Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>,planes: Storage.Buffer<Tile.Tile16>,
  tiles: Storage.Buffer<Tile.Tile8>) -> PackedKept & PackedKept:
  (left_weights,right_weights) = weights
  (left_bias,right_bias) = bias
  (PackedKept{left_weights,left_bias,planes,tiles},
    PackedKept{right_weights,right_bias,Tensors.buffer(~Tile.Tile16,Tile.filled_pair(0.0),0),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),0)})

# Each task owns its copy of the read-only weights and bias.
def packed_share(kept: PackedKept) -> PackedKept & PackedKept:
  PackedKept{weights,bias,planes,tiles} = kept
  packed_shared(Clone.clone(~Tile.Tile8,weights),Clone.clone(~Tile.Tile8,bias),planes,tiles)

def packed_tasks(~custom: F32 -> F32,+convolution: Convolution,+input_grid: Grid.Grid,+grid: Grid.Grid,decision: Tasks.Decision,pieces: Pieces.Pieces(~Tile.Tile8),
  weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<Tile.Tile8>) -> Pieces.Pieces(~Tile.Tile8):
  windowed_tasks(~PackedKept,~PackedWork,~packed_begin,~packed_call,~packed_finish(~custom),~packed_prepare,~packed_share,convolution,input_grid,grid,decision,pieces,
    PackedKept{weights,bias,Tensors.buffer(~Tile.Tile16,Tile.filled_pair(0.0),0),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),0)})

# ---- Pointwise: the rows of the product are the input cells ----

# A 1 x 1 window of stride 1 without padding over an inner extent of whole
# pairs of groups reads its rows from the input cells, sixteen output
# channels a block, without packing.
def pointwise(convolution: Convolution) -> Bool:
  Convolution{h,w,+k,+s,+p,oh,ow,channels,+inner,gi,go,activation} = convolution
  U32.is_zero(U32.max(U32.max((k - 1 : U32),(s - 1 : U32)),U32.max(p,(inner % 16 : U32))))

# The pairs of the weights and the bias of a pointwise task.
type PointKept is Type:
  PointKept{pairs: Storage.Buffer<Tile.Tile16>,bias: Storage.Buffer<Tile.Tile8>}

# A pointwise output piece: the kept operands and its products staged as
# pairs.
type PointWork is Type:
  PointWork{pairs: Storage.Buffer<Tile.Tile16>,bias: Storage.Buffer<Tile.Tile8>,staged: Storage.Buffer<Tile.Tile16>}

def point_returned(kept: GemmBuffer.PointKept) -> Storage.Buffer<Tile.Tile8> & PointWork:
  GemmBuffer.PointKept{bias,input,pairs,staged} = kept
  (input,PointWork{pairs,bias,staged})

# Position q of a pointwise call reads its cells from base + (q - first row)
# * groups.
def point_call(+convolution: Convolution,+call: Call,input: Storage.Buffer<Tile.Tile8>,work: PointWork) -> Storage.Buffer<Tile.Tile8> & PointWork:
  Convolution{h,w,k,s,p,oh,+ow,+channels,+inner,+gi,go,activation} = convolution
  Call{band,image,+first,+offset,+count,+base} = call
  KernelWindow.Band{+row_first,rows} = band
  PointWork{pairs,bias,staged} = work
  point_returned(GemmBuffer.point_keeping(
    Gemm.PointMatrix{Gemm.TileMatrix{channels,count,inner,offset,(offset + count : U32)},(base + KernelWindow.start(first,1,ow,row_first) * gi : U32),gi},
    input,pairs,bias,staged))

def point_begin(+convolution: Convolution,+positions: U32,+count: U32,kept: PointKept) -> PointWork:
  PointKept{pairs,bias} = kept
  PointWork{pairs,bias,Tensors.buffer(~Tile.Tile16,Tile.filled_pair(0.0),(positions * Gemm.pairs_of(channels_of(convolution)) : U32))}

def point_finish(~custom: F32 -> F32,+convolution: Convolution,+positions: U32,work: PointWork) -> Storage.Buffer<Tile.Tile8> & PointKept:
  PointWork{pairs,bias,staged} = work
  (Pipeline.point_finished(~custom,activation_of(convolution),channels_of(convolution),positions,staged),PointKept{pairs,bias})

def point_prepare(+convolution: Convolution,+positions: U32,kept: PointKept) -> PointKept:
  kept

def point_shared(pairs: Storage.Buffer<Tile.Tile16> & Storage.Buffer<Tile.Tile16>,bias: Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) -> PointKept & PointKept:
  (left_pairs,right_pairs) = pairs
  (left_bias,right_bias) = bias
  (PointKept{left_pairs,left_bias},PointKept{right_pairs,right_bias})

def point_share(kept: PointKept) -> PointKept & PointKept:
  PointKept{pairs,bias} = kept
  point_shared(Clone.clone(~Tile.Tile16,pairs),Clone.clone(~Tile.Tile8,bias))

def pairs_built(built: Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile16>) -> Storage.Buffer<Tile.Tile16>:
  (weights,pairs) = built
  pairs

# The pairs of the weights are built once and every task reads its copy.
def point_tasks(~custom: F32 -> F32,+convolution: Convolution,+input_grid: Grid.Grid,+grid: Grid.Grid,decision: Tasks.Decision,pieces: Pieces.Pieces(~Tile.Tile8),
  weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<Tile.Tile8>) -> Pieces.Pieces(~Tile.Tile8):
  +channels = channels_of(convolution)
  +inner = inner_of(convolution)
  windowed_tasks(~PointKept,~PointWork,~point_begin,~point_call,~point_finish(~custom),~point_prepare,~point_share,convolution,input_grid,grid,decision,pieces,
    PointKept{pairs_built(GemmBuffer.point_pairs(channels,inner,weights,Tensors.buffer(~Tile.Tile16,Tile.filled_pair(0.0),(Gemm.pairs_of(channels) * inner : U32)))),bias})

def windowed_route(~custom: F32 -> F32,direct: Bool,+convolution: Convolution,+input_grid: Grid.Grid,+grid: Grid.Grid,decision: Tasks.Decision,pieces: Pieces.Pieces(~Tile.Tile8),
  weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<Tile.Tile8>) -> Pieces.Pieces(~Tile.Tile8):
  match direct:
    case True{}: point_tasks(~custom,convolution,input_grid,grid,decision,pieces,weights,bias)
    case False{}: packed_tasks(~custom,convolution,input_grid,grid,decision,pieces,weights,bias)

def windowed_when(~custom: F32 -> F32,empty: Bool,+convolution: Convolution,+input_grid: Grid.Grid,+grid: Grid.Grid,decision: Tasks.Decision,pieces: Pieces.Pieces(~Tile.Tile8),
  weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<Tile.Tile8>) -> Pieces.Pieces(~Tile.Tile8):
  match empty:
    case True{}: Tensors.filled_grid(~Tile.Tile8,grid,Tile.filled(0.0))
    case False{}: windowed_route(~custom,pointwise(convolution),convolution,input_grid,grid,decision,pieces,weights,bias)

# An empty output grid has no task.
def windowed(~custom: F32 -> F32,+convolution: Convolution,+input_grid: Grid.Grid,+grid: Grid.Grid,decision: Tasks.Decision,pieces: Pieces.Pieces(~Tile.Tile8),
  weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<Tile.Tile8>) -> Pieces.Pieces(~Tile.Tile8):
  windowed_when(~custom,U32.is_zero(Grid.count_of(grid)),convolution,input_grid,grid,decision,pieces,weights,bias)
