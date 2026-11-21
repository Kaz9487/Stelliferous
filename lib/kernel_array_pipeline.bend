# The Array executor of the array API and of the unconditional root. Its
# cells are those of a tensor: the groups of eight channels of a position side
# by side, position after position. Its parts run through the recursion that
# pieces use.
import Base
import ./kernel_pipeline.bend as Pipeline
import ./kernel_gemm.bend as Gemm
import ./kernel_weights.bend as Weights
import ./kernel_spatial_packing.bend as Packing
import ./kernel_gather.bend as Gather
import ./kernel_tile.bend as Tile
import ./storage_allocation.bend as Allocation
import ./tensor.bend as Tensors
import ./tensor_grid.bend as Grid
import ./traversal_partition.bend as Partition
import ./traversal_array.bend as Traversal
import ./tensor_pieces.bend as Pieces
import ./kernel_window.bend as KernelWindow
import ./tensor_tiled_cells.bend as TiledCells

def allocated(+rows: U32,+span: U32) -> Array<F32>:
  Allocation.array(~F32,Tensors.depth_for((rows * span : U32)),0.0)

def allocated_tiles(+rows: U32,+span: U32) -> Array<Tile.Tile8>:
  Allocation.array(~Tile.Tile8,Tensors.depth_for(Pipeline.output_tiles(rows,span)),Tile.filled(0.0))

def values(owners: Array<Tile.Tile8> & Array<F32>) -> Array<F32>:
  (tiles,values) = owners
  values

def cells(owners: Array<F32> & Array<Tile.Tile8>) -> Array<Tile.Tile8>:
  (input,tiles) = owners
  tiles

# The rows of a block are eight positions of the packed input, its lanes eight
# channels of the packed weights; the leaf returns the cells of its positions.
def computed(+rows: U32,+span: U32,inner: U32,weights: Array<Tile.Tile8>,bias: Array<Tile.Tile8>,packed: Array<Tile.Tile16> & Array<Tile.Tile8>) -> Array<Tile.Tile8>:
  (planes,tiles) = packed
  +matrix = {Gemm.TileMatrix{rows,span,inner,0,span} : Gemm.TileMatrix}
  Gemm.cell_output(Gemm.cell_run(U32.to_nat(Gemm.tile_jobs(matrix)),matrix,
    Gemm.CellOperands{bias,Gemm.Operands{tiles,weights,Unit{},allocated_tiles(rows,span)}}))

# The tiles of count entries from start, packed from the planes of the input.
def planes_packed(+window: Packing.Stencil,+planes: Packing.Planes,count: Nat,start: U32,tiles: Array<Tile.Tile8>,
  transposed: Array<Tile.Tile8> & Array<Tile.Tile16>) -> Array<Tile.Tile16> & Array<Tile.Tile8>:
  (input,pairs) = transposed
  Packing.pack(window,planes,count,start,pairs,tiles)

# The cells of rows [first, first + rows) of the image.
type InputWindow is Type:
  InputWindow{owner: Array<Tile.Tile8>,first: U32,rows: U32}

def windows(+first: U32,+rows: U32,+left_first: U32,+left_rows: U32,owners: Array<Tile.Tile8> & Array<Tile.Tile8>) -> InputWindow & InputWindow:
  (source,target) = owners
  (InputWindow{target,left_first,left_rows},InputWindow{source,first,rows})

def cloned(+first: U32,+rows: U32,owners: Array<Tile.Tile8> & Array<Tile.Tile8>) -> InputWindow & InputWindow:
  (left,right) = owners
  (InputWindow{left,first,rows},InputWindow{right,first,rows})

# The rows of a band are one run of cells.
def copied(+groups: U32,+width: U32,+first: U32,+rows: U32,band: KernelWindow.Band,owner: Array<Tile.Tile8>) -> InputWindow & InputWindow:
  KernelWindow.Band{+left_first,+left_rows} = band
  +count = (left_rows * width * groups : U32)
  windows(first,rows,left_first,left_rows,Traversal.copy(~Tile.Tile8,U32.to_nat(count),((left_first - first) * width * groups : U32),0,
    owner,Allocation.array(~Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0))))

def split_when(copy: Bool,+groups: U32,+width: U32,+first: U32,+rows: U32,+band: KernelWindow.Band,owner: Array<Tile.Tile8>) -> InputWindow & InputWindow:
  match copy:
    case False{}: cloned(first,rows,Array.clone(Tile.Tile8,owner))
    case True{}: copied(groups,width,first,rows,band,owner)

# The right child keeps the parent's rows; the left child copies only its band.
# Unaligned rows would break whole eight-position blocks, so they are cloned;
# so is a band that fails its coverage check.
def split(work: Pipeline.Work,offset: U32,half: U32,window: InputWindow) -> InputWindow & InputWindow:
  Pipeline.Work{channels_out,+height,+width,+kernel,+stride,+padding,+output_width,positions,+inner} = work
  InputWindow{owner,+first,+rows} = window
  +offset = offset
  +half = half
  +band = KernelWindow.band(first,rows,height,kernel,stride,padding,output_width,offset,half)
  split_when(KernelWindow.copies_band(first,rows,height,kernel,stride,padding,output_width,offset,half,band),
    Gemm.blocks((inner / (kernel * kernel) : U32)),width,first,rows,band,owner)

def leaf(work: Pipeline.Work,offset: U32,span: U32,window: InputWindow,weights: Array<Tile.Tile8>,bias: Array<Tile.Tile8>) -> Partition.Tree<&1,Array<Tile.Tile8>,U32>:
  Pipeline.Work{+rows,h,w,+k,+s,p,+ow,positions,+inner} = work
  InputWindow{input,+first,height} = window
  +offset = offset
  +span = span
  +count = (span / 8 * inner : U32)
  +stencil = Packing.square(height,w,k,s,p,ow,inner,Gemm.blocks((inner / (k * k) : U32)))
  +start = KernelWindow.start(offset,s,ow,first)
  +planes = Packing.call_planes(stencil,start,KernelWindow.padded(span))
  Partition.Leaf{computed(rows,span,inner,weights,bias,
    planes_packed(stencil,planes,U32.to_nat(count),start,Allocation.array(~Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0)),
      Packing.transpose(stencil,planes,input,
        Allocation.array(~Tile.Tile16,Tensors.depth_for(Packing.plane_pairs(stencil,planes)),Tile.filled_pair(0.0))))),offset,span,rows}

# What one part owns: the input rows it reads, its weights and its bias.
type Owners is Type:
  Owners{window: InputWindow,weights: Array<Tile.Tile8>,bias: Array<Tile.Tile8>}

def divided(windows: InputWindow & InputWindow,weights: Array<Tile.Tile8> & Array<Tile.Tile8>,bias: Array<Tile.Tile8> & Array<Tile.Tile8>) -> Pieces.Division<Owners>:
  (left_window,right_window) = windows
  (left_weights,right_weights) = weights
  (left_bias,right_bias) = bias
  Pieces.Divided{Owners{left_window,left_weights,left_bias},Owners{right_window,right_weights,right_bias}}

# The left half reads the positions of its ranges; each half owns its weights and bias.
def divide(work: Pipeline.Work,left: Pieces.Ranges,right: Pieces.Ranges,owners: Owners) -> Pieces.Division<Owners>:
  Owners{window,weights,bias} = owners
  +offset = Pieces.first(left)
  divided(split(work,offset,(Pieces.first(right) - offset : U32),window),Array.clone(Tile.Tile8,weights),Array.clone(Tile.Tile8,bias))

def part(work: Pipeline.Work,offset: U32,span: U32,owners: Owners) -> Partition.Tree<&1,Array<Tile.Tile8>,U32>:
  Owners{window,weights,bias} = owners
  leaf(work,offset,span,window,weights,bias)

# A split always divides the owners; kept whole, they would compute the whole
# ranges as one part.
def whole(work: Pipeline.Work,layout: Pieces.Ranges,owners: Owners) -> Partition.Tree<&1,Array<Tile.Tile8>,U32>:
  +layout = layout
  part(work,Pieces.first(layout),(Pieces.last(layout) - Pieces.first(layout) : U32),owners)

# Positions [offset, offset + span) halved depth times at whole blocks of eight.
def partition(depth: Nat,work: Pipeline.Work,offset: U32,span: U32,input: InputWindow,weights: Array<Tile.Tile8>,bias: Array<Tile.Tile8>) -> Partition.Tree<&1,Array<Tile.Tile8>,U32>:
  +offset = offset
  +span = span
  Pieces.each_part(~Owners,~Array<Tile.Tile8>,~U32,~Pipeline.Work,~Unit,~Pieces.always,~divide,~part,~whole,
    Pieces.unit_ranges(depth,8,offset,span),True{},Unit{},work,Owners{input,weights,bias})

# The cells of a part are one run of the output cells.
def gathered(rows: U32,offset: U32,positions: U32,span: U32,input: Array<Tile.Tile8>,output: Array<Tile.Tile8>) -> Array<Tile.Tile8>:
  +groups = Gemm.blocks(rows)
  Gather.gather_block(~Tile.Tile8,1,(offset * groups : U32),(positions * groups : U32),(span * groups : U32),input,output)

def finish_leaf(identity: Bool,input: Array<Tile.Tile8>,+rows: U32,offset: U32,span: U32,+positions: U32) -> Array<Tile.Tile8>:
  match identity:
    case True{}: input
    case False{}: gathered(rows,offset,positions,span,input,allocated_tiles(rows,positions))

def finish(partitions: Partition.Tree<&1,Array<Tile.Tile8>,U32>,+rows: U32,+positions: U32) -> Array<Tile.Tile8>:
  match partitions:
    case Partition.Leaf{input,+offset,+span,channels}:
      finish_leaf(U32.is_zero(offset) && U32.is_eq(span,positions),input,channels,offset,span,positions)
    case Partition.Fork{left,right}:
      Partition.gather(~&1,~Array<Tile.Tile8>,~gathered,Partition.Fork{left,right},allocated_tiles(rows,positions),positions)

def run_cells(depth: U32,work: Pipeline.Work,input: Array<Tile.Tile8>,weights: Array<Tile.Tile8>,bias: Array<Tile.Tile8>) -> Array<Tile.Tile8>:
  Pipeline.Work{+rows,+h,w,k,s,p,ow,+positions,inner} = work
  +padded = KernelWindow.padded(positions)
  +limit = Pipeline.depth_limit(padded)
  finish(partition(U32.to_nat(U32.min(depth,limit)),work,0,padded,InputWindow{input,0,h},weights,bias),rows,positions)

# The cells of the channels of every pixel, the groups of a pixel side by side.
def input_cells(+channels: U32,+pixels: U32,input: Array<F32>) -> Array<Tile.Tile8>:
  +groups = Gemm.blocks(channels)
  cells(TiledCells.tiled(pixels,TiledCells.Tiling{pixels,channels,groups,0},input,
    Allocation.array(~Tile.Tile8,Tensors.depth_for((pixels * groups : U32)),Tile.filled(0.0))))

def run(depth: U32,work: Pipeline.Work,input: Array<F32>,weights: Array<F32>,bias: Array<F32>) -> Array<F32>:
  Pipeline.Work{+rows,+h,+w,+k,s,p,ow,+positions,+inner} = work
  values(TiledCells.untiled(rows,TiledCells.Cells{positions,positions,rows,Gemm.blocks(rows),Grid.Innermost{}},
    run_cells(depth,work,input_cells((inner / (k * k) : U32),(h * w : U32),input),Weights.run(rows,inner,weights),Weights.run(rows,1,bias)),allocated(rows,positions)))
