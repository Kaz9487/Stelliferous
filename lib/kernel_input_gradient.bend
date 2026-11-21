# The input gradient of a convolution by phases: output positions congruent
# modulo the stride read the same taps. Every row of every output channel of
# the output gradient is cut into eight-lane tiles once for each column shift
# the phases read, zero outside; a phase entry is then a tile, and a job is a
# GEMM block that stores its tiles at the final strided positions of its
# piece. The cells of both gradients are those of a tensor: the groups of a
# position side by side, position after position. Every task copies the rows
# of the output gradient its pieces read and computes its pieces from them.
import Base
import ./storage_buffer.bend as Storage
import ./kernel_tile.bend as Tile
import ./kernel_gemm.bend as Gemm
import ./kernel_gemm_buffer.bend as GemmBuffer
import ./proofs/kernel_gemm_certificate.bend as GemmCertificate
import ./storage_copy.bend as StorageCopy
import ./storage_clone.bend as StorageClone
import ./kernel_input_gradient_fill.bend as Fill
import ./proofs/storage_certificate.bend as Certificate
import ./proofs/traversal_certificate.bend as TraversalCertificate
import ./proofs/kernel_input_gradient_fill_certificate.bend as FillCertificate
import ./tensor_view.bend as Views
import ./kernel_pipeline.bend as Pipeline
import ./traversal_array.bend as Traversal
import ./traversal_partition.bend as Partition
import ./tensor.bend as Tensors
import ./tensor_tiled.bend as Tiled
import ./tensor_tiled_map.bend as TiledMap
import ./tensor_grid.bend as Grid
import ./tensor_pieces.bend as Pieces
import ./tensor_tasks.bend as Tasks
import ./tensor_halo.bend as Halo
import ./tensor_move.bend as Move
import ./kernel_cost.bend as Cost
import ./default_kernel_cost.bend as DefaultCost

type Geometry is Data:
  Geometry{channels: U32,outputs: U32,height: U32,width: U32,kernel: U32,stride: U32,padding: U32,rows: U32,columns: U32}

type Axis is Data:
  Axis{first: U32,count: U32,tap: U32,taps: U32,source: U32}

def axis(+first: U32,+size: U32,+kernel: U32,+stride: U32,+padding: U32) -> Axis:
  +residue = ((first + padding) % stride : U32)
  +taps = Bool.pick(U32,U32.is_lt(residue,kernel),((kernel - 1 - residue) / stride + 1 : U32),0)
  +tap = (residue + (taps - 1) * stride : U32)
  Axis{first,((size - 1 - first) / stride + 1 : U32),tap,taps,((first + padding) / stride - tap / stride : U32)}

type Phase is Data:
  Phase{geometry: Geometry,y: Axis,x: Axis,inner: U32}

def phase_axes(geometry: Geometry,a: Axis,b: Axis) -> Phase:
  Geometry{c,+o,h,w,k,s,p,gh,gw} = geometry
  Axis{af,ac,at,+ar,asrc} = a
  Axis{bf,bc,bt,+br,bsrc} = b
  Phase{geometry,a,b,(o * ar * br : U32)}

def phase(+geometry: Geometry,+y: U32,+x: U32) -> Phase:
  Geometry{c,o,+h,+w,+k,+s,+p,gh,gw} = geometry
  phase_axes(geometry,axis(y,h,k,s,p),axis(x,w,k,s,p))

def window_jobs(geometry: Geometry) -> Nat:
  Geometry{c,o,+h,+w,k,+s,p,gh,gw} = geometry
  Nat.mul(U32.to_nat(U32.min(s,h)),U32.to_nat(U32.min(s,w)))

def phase_of(geometry: Geometry,index: U32) -> Phase:
  Geometry{c,o,h,+w,k,+s,p,gh,gw} = geometry
  +index = index
  +columns = U32.min(s,w)
  phase(geometry,(index / columns : U32),(index % columns : U32))

def phase_inner(context: Phase) -> U32:
  Phase{g,y,x,inner} = context
  inner

# The tiling cuts every row of every output channel of the output gradient
# into eight-lane tiles, once for each column shift the phases read, zero
# outside. It is built in chunks of at most chunk_rows output rows and
# chunk_groups groups, about one pipeline block of tiles; one output row of one
# group is the smallest chunk, and fits bounds it by the capacity. A chunk holds
# tile rows * chunk_groups blocks of outputs * shifts tiles (output channel
# outer, shift inner) and one unused tile, so that the rows a step reads do
# not share cache sets. Tile row t is gradient row t - kernel; shift i
# starts at gradient column 8 * group + i + column_low - kernel.
type Tiling is Data:
  Tiling{groups: U32,shifts: U32,outputs: U32,column_low: U32,kernel: U32,height: U32,width: U32,chunk_groups: U32,chunk_rows: U32}

# The outputs [first, last) of an axis of the given size. Over the phases that
# read a tap for one of them: what 0 the lowest tile (lowest output + source +
# kernel), what 1 one past the highest tile (highest output + source + kernel
# + taps), what 2 the most outputs of a phase.
type Band is Data:
  Band{size: U32,kernel: U32,stride: U32,padding: U32,first: U32,last: U32,what: U32}

def band_pick(used: Bool,+low: U32,+high: U32,+count: U32,+value: U32,what: U32) -> U32:
  match what:
    case 0: Bool.pick(U32,used,U32.min(value,low),value)
    case what: Bool.pick(U32,used,U32.max(value,Bool.pick(U32,U32.is_eq(what,1),high,count)),value)

def band_measure(+context: Axis,+kernel: U32,+stride: U32,+first: U32,+last: U32,+value: U32,what: U32) -> U32:
  Axis{+start,+count,tap,+taps,+source} = context
  +low = Bool.pick(U32,U32.is_lt(start,first),((first - start + stride - 1) / stride : U32),0)
  +high = U32.min(count,Bool.pick(U32,U32.is_lt(start,last),((last - start + stride - 1) / stride : U32),0))
  band_pick(U32.is_lt(0,taps) && U32.is_lt(low,high),(low + source + kernel : U32),(high - 1 + source + kernel + taps : U32),count,value,what)

def band_phase(index: U32,context: Band,value: U32) -> U32:
  Band{+size,+kernel,+stride,+padding,+first,+last,what} = context
  band_measure(axis(index,size,kernel,stride,padding),kernel,stride,first,last,value,what)

def band_fold(+context: Band,value: U32) -> U32:
  Band{+size,kernel,+stride,padding,first,last,what} = context
  Traversal.indexed_fold(~U32,~Band,~band_phase,U32.to_nat(U32.min(stride,size)),0,context,value)

# Output rows [first, first + rows) read at most rows - 1 + taps tile rows,
# taps = ceil(kernel / stride), and shifts is at most taps + 1.
def taps(+kernel: U32,+stride: U32) -> U32:
  ((kernel + stride - 1) / stride : U32)

def fits(+geometry: Geometry,+capacity: U32) -> Bool:
  Geometry{c,+o,h,w,+k,+s,p,gh,gw} = geometry
  +taps = Nat.div(Nat.add(U32.to_nat(k),U32.to_nat((s - 1 : U32))),U32.to_nat(s))
  Nat.is_le(Nat.mul(taps,Nat.add(Nat.mul(U32.to_nat(o),Nat.add(taps,1n)),1n)),U32.to_nat(capacity))

def chunked(+groups: U32,+shifts: U32,+outputs: U32,+column_low: U32,+kernel: U32,+stride: U32,+height: U32,+width: U32,+limit: U32) -> Tiling:
  +block = (outputs * shifts + 1 : U32)
  +chunk_groups = U32.max(1,U32.min(groups,(limit / (taps(kernel,stride) * block) : U32)))
  +fitting = (limit / (chunk_groups * block) + 1 : U32)
  Tiling{groups,shifts,outputs,column_low,kernel,height,width,chunk_groups,
    Bool.pick(U32,U32.is_lt(taps(kernel,stride),fitting),(fitting - taps(kernel,stride) : U32),1)}

# Column shifts come from the first output of every column phase; groups from
# the most outputs of a column phase.
def tiling(geometry: Geometry) -> Tiling:
  Geometry{c,+o,h,+w,+k,+s,+p,+gh,+gw} = geometry
  +columns = U32.min(s,w)
  +column_low = band_fold(Band{w,k,s,p,0,columns,0},4294967295)
  chunked(((band_fold(Band{w,k,s,p,0,w,2},0) + 7) / 8 : U32),(band_fold(Band{w,k,s,p,0,columns,1},0) - column_low : U32),
    o,column_low,k,s,gh,gw,Pipeline.block_tiles())

# Rows [first, first + rows) of the output gradient of one image, from cell
# base of data on.
type Source is Type:
  Source{data: Storage.Buffer<Tile.Tile8>,base: U32,first: U32,rows: U32}

def fill_buffers(source_length: Nat,length: Nat,result: Array<Tile.Tile8> & Array<Tile.Tile8>,
  certificates: TraversalCertificate.Certificates<Array<Tile.Tile8>,Tile.Tile8,Certificate.family(~Tile.Tile8),result>) -> Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  (source,tiles) = result
  TraversalCertificate.Certificates{source_certificate,certificate} = certificates
  (Storage.Buffer{source,source_length,source_certificate},Storage.Buffer{tiles,length,certificate})

# The tiles of a chunk and the source they are read from, both certified.
def filled_tiles(+call: Fill.Call,owners: Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  (source,tiles) = owners
  Storage.Buffer{source,source_length,source_certificate} = source
  Storage.Buffer{tiles,length,certificate} = tiles
  fill_buffers(source_length,length,Fill.run(call,source,tiles),FillCertificate.certificates(call,source,tiles,source_certificate,certificate))

type Jobs is Type:
  Jobs{tiles: Storage.Buffer<Tile.Tile8>,weights: Storage.Buffer<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>}

# The gradient tile of phase entry (output o, tap row a, tap column b) is the
# first tile + a * row_step + o * shifts + b: a grid of tap_columns by
# tap_rows entries for each output. The weights are consecutive.
def Cursor() -> Data:
  Gemm.Both<U32,Gemm.Entries>

def entries(cursor: Cursor()) -> Gemm.Step<Cursor()>:
  Gemm.both(~U32,~Gemm.Entries,~Gemm.direct,~Gemm.entry,cursor)

# One job is eight consecutive outputs of one phase row, its rows a stride
# apart, at the eight input channels of one group, its lanes: the cell of
# group g at local position p is at p * cells + g. Its steps run over the
# phase entries, the weights from weight_base.
type Plan is Data:
  Plan{units: U32,groups: U32,inner: U32,source_base: U32,row_step: U32,group_step: U32,
    target_base: U32,target_row: U32,step: U32,cells: U32,channels: U32,columns: U32,weight_base: U32,
    tap_columns: U32,tap_rows: U32,row_jump: U32,output_jump: U32}

def job(plan: Plan,index: U32) -> Gemm.Job<Cursor()>:
  Plan{+units,+groups,+inner,+source_base,+row_step,+group_step,+target_base,+target_row,+step,+cells,+channels,+columns,+weight_base,
    +tap_columns,+tap_rows,+row_jump,+output_jump} = plan
  +index = index
  +unit = (index % units : U32)
  +group = (index / units % groups : U32)
  +line = (index / units / groups : U32)
  +row = (unit * 8 : U32)
  Gemm.Job{row,U32.to_nat(inner),
    Gemm.Both{(weight_base + unit * inner : U32),Gemm.Entries{(source_base + line * row_step + group * group_step : U32),0,0,tap_columns,tap_rows,row_jump,output_jump}},
    Gemm.Store{(target_base + unit + (line * target_row + group * 8 * step) * cells : U32),(step * cells : U32),1,U32.min(8,(columns - group * 8 : U32)),U32.min(8,(channels - row : U32))}}

def jobs_kept(kept: GemmBuffer.Kept<Tile.Tile8>) -> Jobs:
  GemmBuffer.Kept{tiles,weights,bias,output} = kept
  Jobs{tiles,weights,output}

# Each cell is its ordered fold, the gradient times the weight, then + 0.0.
# The gradient tiles are the rows, the weights the lanes.
def jobs(+count: Nat,+plan: Plan,state: Jobs) -> Jobs:
  Jobs{tiles,weights,output} = state
  jobs_kept(GemmBuffer.run(~Plan,~Cursor(),~Tile.Tile8,~job,~entries,~Gemm.row_times,~Gemm.write_tiles(~Gemm.unbiased,~Gemm.plus),
    ~GemmCertificate.write_tiles(~Gemm.unbiased,~GemmCertificate.unbiased_read,~Gemm.plus),count,plan,
    tiles,weights,Tensors.buffer(~F32,0.0,1),output))

# Every phase weights, side by side in phase order.
type Packed is Type:
  Packed{source: Storage.Buffer<Tile.Tile8>,weights: Storage.Buffer<Tile.Tile8>,weight_base: U32}

def packed_copied(+weight_base: U32,+count: U32,copies: Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) -> Packed:
  (source,weights) = copies
  Packed{source,weights,(weight_base + count : U32)}

# Cells [C/8,O,K,K], the groups outermost, in phase order [C/8,O,tap row,tap column].
def weight_view(context: Phase) -> Views.View:
  Phase{Geometry{c,+o,h,w,+k,+s,p,gh,gw},Axis{yf,yc,+yt,+yr,ys},Axis{xf,xc,+xt,+xr,xs},inner} = context
  Views.View{[Views.Axis{Tiled.groups(c),(o * k * k : U32)},Views.Axis{o,(k * k : U32)},
    Views.Axis{yr,(0 - s * k : U32)},Views.Axis{xr,(0 - s : U32)}],(yt * k + xt : U32)}

def packed_phase(+context: Phase,+weight_base: U32,source: Storage.Buffer<Tile.Tile8>,weights: Storage.Buffer<Tile.Tile8>) -> Packed:
  Phase{Geometry{+c,o,h,w,k,s,p,gh,gw},y,x,+inner} = context
  +count = (Tiled.groups(c) * inner : U32)
  # the view built for each use: a View used twice shares every View
  packed_copied(weight_base,count,StorageCopy.view(~Tile.Tile8,Views.is_contiguous(weight_view(context)),count,weight_view(context),weight_base,source,weights))

def pack_selected(empty: Bool,+context: Phase,state: Packed) -> Packed:
  match empty:
    case True{}: state
    case False{}:
      Phase{Geometry{+c,o,h,w,+k,s,p,gh,gw},y,x,+inner} = context
      Packed{source,weights,weight_base} = state
      packed_phase(context,weight_base,source,weights)

def pack_phase(index: U32,geometry: Geometry,state: Packed) -> Packed:
  +selected = phase_of(geometry,index)
  pack_selected(U32.is_eq(phase_inner(selected),0),selected,state)

def weight_total(index: U32,geometry: Geometry,total: U32) -> U32:
  Geometry{+c,o,h,w,k,s,p,gh,gw} = geometry
  (total + (c + 7) / 8 * phase_inner(phase_of(geometry,index)) : U32)

# The tile rows that output rows [first, last) read.
def band_low(+geometry: Geometry,+first: U32,+last: U32) -> U32:
  Geometry{c,o,+h,w,+k,+s,+p,gh,gw} = geometry
  band_fold(Band{h,k,s,p,first,last,0},4294967295)

def band_high(+geometry: Geometry,+first: U32,+last: U32) -> U32:
  Geometry{c,o,+h,w,+k,+s,+p,gh,gw} = geometry
  band_fold(Band{h,k,s,p,first,last,1},0)

def band_rows(+geometry: Geometry,+first: U32,+count: U32) -> U32:
  +low = band_low(geometry,first,(first + count : U32))
  +high = band_high(geometry,first,(first + count : U32))
  Bool.pick(U32,U32.is_lt(low,high),(high - low : U32),0)

type Task is Data:
  Task{geometry: Geometry,layout: Tiling}

type Leaf is Type:
  Leaf{tiles: Storage.Buffer<Tile.Tile8>,weights: Storage.Buffer<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>,weight_base: U32}

# Output rows [first, first + count) go to the cells from target on: the
# cell of group g at row and column is target + ((row - first) * width +
# column) * groups + g.
type LeafContext is Data:
  LeafContext{task: Task,first: U32,count: U32,target: U32}

# Output rows [row, row + rows) and groups [group, group + chunk_groups) of a
# leaf, over a chunk of the tiling that starts at tile row low.
type Chunk is Data:
  Chunk{leaf: LeafContext,row: U32,rows: U32,low: U32,group: U32}

# A tile row of the chunk is one gradient row, so a phase line and a tap row
# both move by row_step; the first entry reads shift xs + k - column_low.
def leaf_jobs(+context: Phase,+chunk: Chunk,+weight_base: U32,state: Jobs) -> Jobs:
  Phase{Geometry{+c,o,+h,+w,+k,+s,p,gh,gw},Axis{+yf,+yc,yt,+yr,+ys},Axis{+xf,+xc,xt,+xr,+xs},+inner} = context
  Chunk{LeafContext{Task{geometry,+layout},+first,count,+target},+row,+rows,+low,+group} = chunk
  Tiling{groups,+shifts,+outputs,+column_low,kernel,height,width,+chunk_groups,chunk_rows} = layout
  +start = Bool.pick(U32,U32.is_lt(yf,row),((row - yf + s - 1) / s : U32),0)
  +end = U32.min(yc,Bool.pick(U32,U32.is_lt(yf,(row + rows : U32)),((row + rows - yf + s - 1) / s : U32),0))
  +units = ((c + 7) / 8 : U32)
  +phase_groups = ((xc + 7) / 8 : U32)
  +span = Bool.pick(U32,U32.is_lt(group,phase_groups),U32.min(chunk_groups,(phase_groups - group : U32)),0)
  +group_step = (outputs * shifts + 1 : U32)
  +row_step = (chunk_groups * group_step : U32)
  jobs(U32.to_nat(Bool.pick(U32,U32.is_lt(start,end),((end - start) * span * units : U32),0)),
    Plan{units,span,inner,((start + ys + k - low) * row_step + xs + k - column_low : U32),row_step,group_step,
      (target + ((yf + start * s - first) * w + xf + group * 8 * s) * units : U32),(s * w : U32),s,units,c,(xc - group * 8 : U32),weight_base,
      xr,yr,(row_step - (xr - 1) : U32),(shifts - (yr - 1) * row_step - (xr - 1) : U32)},state)

def leaf_finished(+weight_base: U32,state: Jobs) -> Leaf:
  Jobs{tiles,weights,output} = state
  Leaf{tiles,weights,output,weight_base}

def leaf_selected(empty: Bool,+context: Phase,+chunk: Chunk,state: Leaf) -> Leaf:
  match empty:
    case True{}: state
    case False{}:
      Phase{Geometry{+c,o,h,w,k,s,p,gh,gw},y,x,+inner} = context
      Leaf{tiles,weights,output,+weight_base} = state
      leaf_finished((weight_base + (c + 7) / 8 * inner : U32),leaf_jobs(context,chunk,weight_base,Jobs{tiles,weights,output}))

def leaf_phase(index: U32,chunk: Chunk,state: Leaf) -> Leaf:
  Chunk{LeafContext{Task{+geometry,layout},first,count,target},row,rows,low,group} = chunk
  +selected = phase_of(geometry,index)
  leaf_selected(U32.is_eq(phase_inner(selected),0),selected,chunk,state)

# The gradient a leaf read, the weights and the output it wrote.
type Part is Type:
  Part{gradient: Storage.Buffer<Tile.Tile8>,weights: Storage.Buffer<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>}

# A leaf runs its chunks in turn with the source and the weights.
type Sweep is Type:
  Sweep{source: Source,weights: Storage.Buffer<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>}

def chunk_leaf(+base: U32,+first: U32,+rows: U32,data: Storage.Buffer<Tile.Tile8>,state: Leaf) -> Sweep:
  Leaf{tiles,weights,output,weight_base} = state
  Sweep{Source{data,base,first,rows},weights,output}

def chunk_finished(+base: U32,+first: U32,+rows: U32,pair: Storage.Buffer<Tile.Tile8> & Leaf) -> Sweep:
  (data,state) = pair
  chunk_leaf(base,first,rows,data,state)

def chunk_run(+chunk: Chunk,weights: Storage.Buffer<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>,owners: Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8> & Leaf:
  Chunk{LeafContext{Task{+geometry,layout},first,count,target},row,rows,low,group} = chunk
  (data,tiles) = owners
  (data,Traversal.indexed_fold(~Leaf,~Chunk,~leaf_phase,window_jobs(geometry),0,chunk,Leaf{tiles,weights,output,0}))

# Chunk index: row chunk outer, group chunk inner.
def chunk_step(index: U32,context: LeafContext,state: Sweep) -> Sweep:
  LeafContext{Task{+geometry,+layout},+first,+count,target} = context
  Geometry{c,+o,h,w,k,s,p,gh,gw} = geometry
  Tiling{+groups,+shifts,+outputs,+column_low,+kernel,height,+width,+chunk_groups,+chunk_rows} = layout
  Sweep{Source{data,+base,+source_first,+source_rows},weights,output} = state
  +index = index
  +group_chunks = ((groups + chunk_groups - 1) / chunk_groups : U32)
  +row = (first + index / group_chunks * chunk_rows : U32)
  +rows = U32.min(chunk_rows,(first + count - row : U32))
  +group = (index % group_chunks * chunk_groups : U32)
  +span = U32.min(chunk_groups,(groups - group : U32))
  +low = band_low(geometry,row,(row + rows : U32))
  +tile_rows = band_rows(geometry,row,rows)
  chunk_finished(base,source_first,source_rows,
    chunk_run(Chunk{context,row,rows,low,group},weights,output,
      filled_tiles(Fill.Call{U32.to_nat((tile_rows * span : U32)),Fill.Filling{shifts,outputs,column_low,kernel,width,chunk_groups,low,base,Tiled.groups(o),source_first,source_rows,group,span}},
        (data,Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),(tile_rows * chunk_groups * (outputs * shifts + 1) + 1 : U32))))))

def swept(state: Sweep) -> Part:
  Sweep{Source{data,base,first,rows},weights,output} = state
  Part{data,weights,output}

# Output rows [first, first + count) from a source holding the gradient rows they read.
def leaf_band(+context: LeafContext,source: Source,weights: Storage.Buffer<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>) -> Part:
  LeafContext{Task{+geometry,+layout},first,+count,target} = context
  Tiling{+groups,shifts,outputs,column_low,kernel,height,width,+chunk_groups,+chunk_rows} = layout
  swept(Traversal.indexed_fold(~Sweep,~LeafContext,~chunk_step,
    U32.to_nat(((count + chunk_rows - 1) / chunk_rows * ((groups + chunk_groups - 1) / chunk_groups) : U32)),0,context,Sweep{source,weights,output}))

# The gradient rows [start, start + count) that output rows [first, first + rows) read.
def copy_start(+geometry: Geometry,+first: U32,+rows: U32) -> U32:
  Geometry{c,o,h,w,+k,s,p,gh,gw} = geometry
  +low = band_low(geometry,first,(first + rows : U32))
  Bool.pick(U32,U32.is_lt(k,low),(low - k : U32),0)

def copy_stop(+geometry: Geometry,+first: U32,+rows: U32) -> U32:
  Geometry{c,o,h,w,+k,s,p,+gh,gw} = geometry
  +high = band_high(geometry,first,(first + rows : U32))
  Bool.pick(U32,U32.is_lt(k,high),U32.min((high - k : U32),gh),0)

def copy_count(+geometry: Geometry,+first: U32,+rows: U32) -> U32:
  +start = copy_start(geometry,first,rows)
  +stop = copy_stop(geometry,first,rows)
  Bool.pick(U32,U32.is_lt(start,stop),(stop - start : U32),0)

# ---- Tasks over the pieces of the input gradient ----

def row_cells(+geometry: Geometry) -> U32:
  Geometry{+c,o,h,+w,k,s,p,gh,gw} = geometry
  (w * Tiled.groups(c) : U32)

def gradient_row_cells(+geometry: Geometry) -> U32:
  Geometry{c,+o,h,w,k,s,p,gh,+gw} = geometry
  (gw * Tiled.groups(o) : U32)

def image_height(+geometry: Geometry) -> U32:
  Geometry{c,o,+h,w,k,s,p,gh,gw} = geometry
  U32.max(h,1)

def gradient_height(+geometry: Geometry) -> U32:
  Geometry{c,o,h,w,k,s,p,+gh,gw} = geometry
  gh

# A task between images: the gradient pieces it holds, the spare its bands
# are assembled in, the weights and the output.
type Rows is Type:
  Rows{gradient: Move.Holding<Tile.Tile8>,spare: Move.Spare<Tile.Tile8>,weights: Storage.Buffer<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>}

def rows_back(returned: Move.Holding<Tile.Tile8> & Move.Spare<Tile.Tile8>,weights: Storage.Buffer<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>) -> Rows:
  (gradient,spare) = returned
  Rows{gradient,spare,weights,output}

def image_returned(lent: Move.Lent<Tile.Tile8>,part: Part) -> Rows:
  Part{gradient,weights,output} = part
  rows_back(Move.returned(~Tile.Tile8,lent,gradient),weights,output)

def image_banded(+task: Task,+first: U32,+count: U32,+target: U32,banded: Move.Band<Tile.Tile8>,weights: Storage.Buffer<Tile.Tile8>,output: Storage.Buffer<Tile.Tile8>) -> Rows:
  Task{+geometry,layout} = task
  Move.Band{lent,data,+base} = banded
  image_returned(lent,leaf_band(LeafContext{task,first,count,target},Source{data,base,copy_start(geometry,first,count),copy_count(geometry,first,count)},weights,output))

# Rows [first, first + count) of one image, written from cell target of the
# output, from the gradient rows they read, lent from the pieces that hold
# them.
def image_rows(+task: Task,+image: U32,+first: U32,+count: U32,+target: U32,rows: Rows) -> Rows:
  Task{+geometry,layout} = task
  Rows{gradient,spare,weights,output} = rows
  +size = gradient_row_cells(geometry)
  image_banded(task,first,count,target,
    Move.band(~Tile.Tile8,gradient,((image * gradient_height(geometry) + copy_start(geometry,first,count)) * size : U32),(copy_count(geometry,first,count) * size : U32),Tile.filled(0.0),spare),
    weights,output)

# Rows [row, end) of the batch, image by image, from cell target on.
def batch_rows(fuel: Nat,+task: Task,+row: U32,+end: U32,+target: U32,rows: Rows) -> Rows:
  match fuel:
    case 0n: rows
    case 1n+rest:
      Task{+geometry,layout} = task
      +height = image_height(geometry)
      +image = (row / height : U32)
      +count = (U32.min(end,((image + 1) * height : U32)) - row : U32)
      batch_rows(rest,task,(row + count : U32),end,(target + count * row_cells(geometry) : U32),image_rows(task,image,(row % height : U32),count,target,rows))

def images_between(+geometry: Geometry,+row: U32,+end: U32) -> Nat:
  +height = image_height(geometry)
  U32.to_nat(Bool.pick(U32,U32.is_lt(row,end),((end - 1) / height - row / height + 1 : U32),0))

# The rows of the batch that hold the cells [first, first + count).
def first_row(+geometry: Geometry,+first: U32) -> U32:
  (first / U32.max(row_cells(geometry),1) : U32)

def end_row(+geometry: Geometry,+first: U32,+count: U32) -> U32:
  +size = U32.max(row_cells(geometry),1)
  ((first + count + size - 1) / size : U32)

def Operands() -> Type:
  Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>

def rows_whole(rows: Rows) -> Storage.Buffer<Tile.Tile8> & Operands():
  Rows{gradient,spare,weights,output} = rows
  (output,(gradient,weights))

def rows_cut(+offset: U32,+count: U32,rows: Rows) -> Storage.Buffer<Tile.Tile8> & Operands():
  Rows{gradient,spare,weights,output} = rows
  (Pair.snd(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,
    StorageCopy.view(~Tile.Tile8,True{},count,Views.View{[Views.Axis{count,1}],offset},0,output,Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),count))),(gradient,weights))

def piece_kept(whole: Bool,+offset: U32,+count: U32,rows: Rows) -> Storage.Buffer<Tile.Tile8> & Operands():
  match whole:
    case True{}: rows_whole(rows)
    case False{}: rows_cut(offset,count,rows)

# The cells of a piece with +0 in the lanes past the channels.
def piece_padded(+geometry: Geometry,+first: U32,+count: U32,pair: Storage.Buffer<Tile.Tile8> & Operands()) -> Storage.Buffer<Tile.Tile8> & Operands():
  Geometry{+c,o,h,w,k,s,p,gh,gw} = geometry
  (output,operands) = pair
  (TiledMap.padded(TiledMap.Padding{c,Tiled.groups(c),0,Grid.Innermost{}},first,count,output),operands)

# The cells [first, first + count) of the input gradient: the rows that hold
# them, computed whole, and the cells cut from them when the piece holds part
# of a row.
def piece_cells(+task: Task,+first: U32,+count: U32,operands: Operands()) -> Storage.Buffer<Tile.Tile8> & Operands():
  Task{+geometry,layout} = task
  (window,weights) = operands
  +row = first_row(geometry,first)
  +end = end_row(geometry,first,count)
  +size = row_cells(geometry)
  piece_padded(geometry,first,count,piece_kept(U32.is_zero((first % U32.max(size,1) : U32)) && U32.is_zero((count % U32.max(size,1) : U32)),(first - row * size : U32),count,
    batch_rows(images_between(geometry,row,end),task,row,end,0,
      Rows{window,Move.spare(~Tile.Tile8,Tile.filled(0.0)),weights,Tensors.buffer(~Tile.Tile8,Tile.filled((0.0 + 0.0 : F32)),((end - row) * size : U32))})))

# The output gradient cells that the cells [first, first + count) read: from
# the first row the first image reads to the last row the last image reads.
# Rows of an image that read no gradient row start at the end of its rows
# and stop at its first row.
def reach(+task: Task,+first: U32,+count: U32) -> Tasks.Span:
  Task{+geometry,layout} = task
  +height = image_height(geometry)
  +row = first_row(geometry,first)
  +end = end_row(geometry,first,count)
  +last = (end - 1 : U32)
  +head = (U32.min(end,((row / height + 1) * height : U32)) - row : U32)
  +tail_first = Bool.pick(U32,U32.is_eq((row / height : U32),(last / height : U32)),(row % height : U32),0)
  +tail = (last % height + 1 - tail_first : U32)
  +gradient_rows = gradient_height(geometry)
  +size = gradient_row_cells(geometry)
  +low = ((row / height * gradient_rows + U32.min(copy_start(geometry,(row % height : U32),head),gradient_rows)) * size : U32)
  +high = ((last / height * gradient_rows + copy_stop(geometry,tail_first,tail)) * size : U32)
  Tasks.Span{low,(U32.max(high,low) - low : U32)}

# What every task reads: the work and the grid of the input gradient.
type Work is Data:
  Work{task: Task,grid: Grid.Grid}

# The pieces from piece on, remaining more after it. Each piece waits on the
# left for the pieces after it, so the task calls no closure.
def task_tree(remaining: Nat,+task: Task,+grid: Grid.Grid,+piece: U32,ready: Pieces.Walking<Storage.Buffer<Tile.Tile8>,Operands()>) ->
  Pieces.Walked<Storage.Buffer<Tile.Tile8>,Operands()>:
  match remaining:
    case 0n:
      Pieces.Walking{lefts,operands} = ready
      +first = Grid.start(grid,piece)
      +count = Grid.span(grid,piece)
      Pieces.piece_walked(~Tile.Tile8,~Operands(),lefts,first,count,piece_cells(task,first,count,operands))
    case 1n+rest:
      Pieces.Walking{lefts,operands} = ready
      +first = Grid.start(grid,piece)
      +count = Grid.span(grid,piece)
      Pieces.joined(~Storage.Buffer<Tile.Tile8>,~Operands(),task_tree(rest,task,grid,(piece + 1 : U32),
        Pieces.pushed(~Storage.Buffer<Tile.Tile8>,~Operands(),Pieces.piece_walked(~Tile.Tile8,~Operands(),lefts,first,count,piece_cells(task,first,count,operands)))))

def task_finished(+first: U32,+count: U32,walked: Pieces.Walked<Storage.Buffer<Tile.Tile8>,Operands()>) -> Partition.Tree<&1,Pieces.Pieces(~Tile.Tile8),Unit> & Storage.Buffer<Tile.Tile8>:
  Pieces.Walked{pieces,lefts,operands} = walked
  (window,weights) = operands
  (Partition.Leaf{pieces,first,count,Unit{}},weights)

# A task: its pieces from the gradient pieces it holds and its copy of the
# weights.
def task_leaf(work: Work,first: U32,count: U32,window: Move.Holding<Tile.Tile8>,weights: Storage.Buffer<Tile.Tile8>) ->
  Partition.Tree<&1,Pieces.Pieces(~Tile.Tile8),Unit> & Storage.Buffer<Tile.Tile8>:
  Work{+task,+grid} = work
  +first = first
  +count = count
  +piece = Grid.piece_of(grid,first)
  task_finished(first,count,
    task_tree(Grid.later_pieces(grid,first,count),task,grid,piece,Pieces.Walking{Pieces.NoLefts{},(window,weights)}))

def reach_of(+task: Task,+first: U32,+count: U32) -> Tasks.Span:
  reach(task,first,count)

# Each task owns its copy of the packed weights.
def share(weights: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  StorageClone.clone(~Tile.Tile8,weights)

def windowed_tasks(+work: Work,+gradient_grid: Grid.Grid,decision: Tasks.Decision,gradient: Pieces.Pieces(~Tile.Tile8),weights: Storage.Buffer<Tile.Tile8>) -> Pieces.Pieces(~Tile.Tile8):
  Work{+task,+grid} = work
  Tasks.Decision{+parallel,+tasks} = decision
  Tensors.placed(~Tile.Tile8,Pieces.ranges_of_grid(grid),Tile.filled(0.0),
    Pieces.grouped_with(~Move.Holding<Tile.Tile8>,~Pieces.Pieces(~Tile.Tile8),~Storage.Buffer<Tile.Tile8>,~Work,~share,~task_leaf,tasks,parallel,Work{task,grid},
      Halo.windows(~Tile.Tile8,~Task,~reach_of,task,gradient_grid,Tile.filled(0.0),tasks,gradient),weights))

def windowed_when(empty: Bool,+task: Task,+grid: Grid.Grid,+gradient_grid: Grid.Grid,decision: Tasks.Decision,gradient: Pieces.Pieces(~Tile.Tile8),weights: Storage.Buffer<Tile.Tile8>) ->
  Pieces.Pieces(~Tile.Tile8):
  match empty:
    case True{}: Tensors.filled_grid(~Tile.Tile8,grid,Tile.filled(0.0))
    case False{}: windowed_tasks(Work{task,grid},gradient_grid,decision,gradient,weights)

# An empty input gradient has no task.
def windowed(+work: Work,+gradient_grid: Grid.Grid,decision: Tasks.Decision,gradient: Pieces.Pieces(~Tile.Tile8),weights: Storage.Buffer<Tile.Tile8>) -> Pieces.Pieces(~Tile.Tile8):
  Work{+task,+grid} = work
  windowed_when(U32.is_zero(Grid.count_of(grid)),task,grid,gradient_grid,decision,gradient,weights)

type Outcome is Type:
  Outcome{weights: Storage.Buffer<Tile.Tile8>,output: Pieces.Pieces(~Tile.Tile8)}

# One cell of the input gradient: a multiply-add row of its eight channels for
# every output channel of every tap of its phase, its store, and its share of
# the packed gradient tiles over its groups.
def cell_inner(+geometry: Geometry) -> F32:
  Geometry{c,+o,h,w,+k,+s,p,gh,gw} = geometry
  +taps = taps(k,s)
  U32.to_f32((o * taps * taps : U32))

def packed_share(+geometry: Geometry) -> F32:
  Geometry{+c,o,h,w,k,s,p,gh,gw} = geometry
  (cell_inner(geometry) / 8.0 / U32.to_f32(Tiled.groups(c)) : F32)

def cell_rows(+geometry: Geometry) -> F32:
  (cell_inner(geometry) + 1.0 + packed_share(geometry) * Pipeline.packed_tile_rows() : F32)

def cell_steps(+geometry: Geometry) -> F32:
  (4.0 + packed_share(geometry) * Pipeline.packed_tile_steps() : F32)

# What a task beyond the first adds: its own weights, and a halo of the
# gradient rows that two tasks of one image both read.
def extra_of(+geometry: Geometry,+whole: U32,+weight_tiles: U32) -> Tasks.Extra:
  Geometry{c,o,h,w,+k,+s,p,gh,gw} = geometry
  +halo = Bool.pick(U32,U32.is_lt(1,whole) || U32.is_le(taps(k,s),1),0,((taps(k,s) - 1) * gradient_row_cells(geometry) : U32))
  Tasks.Extra{(U32.to_f32(weight_tiles) * 32.0 : F32),(U32.to_f32(halo) * Cost.copy_ns(64.0,DefaultCost.configuration()) : F32),0.0}

def packed_run(+geometry: Geometry,+grid: Grid.Grid,+gradient_grid: Grid.Grid,+whole: U32,+workers: U32,gradient: Pieces.Pieces(~Tile.Tile8),packed: Packed) -> Outcome:
  Packed{source,weights,+weight_base} = packed
  Outcome{source,windowed(Work{Task{geometry,tiling(geometry)},grid},gradient_grid,
    Tasks.decide_whole(grid,workers,whole,cell_rows(geometry),cell_steps(geometry),extra_of(geometry,whole,weight_base)),gradient,weights)}

def tiled(+geometry: Geometry,+grid: Grid.Grid,+gradient_grid: Grid.Grid,+whole: U32,+workers: U32,gradient: Pieces.Pieces(~Tile.Tile8),weights: Storage.Buffer<Tile.Tile8>) -> Outcome:
  +phases = window_jobs(geometry)
  +weight_tiles = Traversal.indexed_fold(~U32,~Geometry,~weight_total,phases,0,geometry,0)
  packed_run(geometry,grid,gradient_grid,whole,workers,gradient,
    Traversal.indexed_fold(~Packed,~Geometry,~pack_phase,phases,0,geometry,
      Packed{weights,Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),(weight_tiles + 1 : U32)),0}))

def run_selected(empty: Bool,+geometry: Geometry,+cells: List<&2,U32>,+gradient_grid: Grid.Grid,+whole: U32,+workers: U32,gradient: Pieces.Pieces(~Tile.Tile8),
  weights: Storage.Buffer<Tile.Tile8>) -> Outcome:
  match empty:
    case True{}: Outcome{weights,Tensors.filled(~Tile.Tile8,cells,Tiled.zero_tile())}
    case False{}: tiled(geometry,Tensors.grid_of(cells),gradient_grid,whole,workers,gradient,weights)

# The output gradient cells [N][Ho][Wo][O / 8] on gradient_grid and the
# weight cells [C / 8][O][k][k] give the input gradient cells of the cell
# shape cells, [N][H][W][C / 8]. A task holds the cells of whole images in
# runs of whole.
def run(+geometry: Geometry,+cells: List<&2,U32>,+gradient_grid: Grid.Grid,+whole: U32,+workers: U32,gradient: Pieces.Pieces(~Tile.Tile8),weights: Storage.Buffer<Tile.Tile8>) -> Outcome:
  run_selected(U32.is_zero(Grid.product(cells)),geometry,cells,gradient_grid,whole,workers,gradient,weights)
