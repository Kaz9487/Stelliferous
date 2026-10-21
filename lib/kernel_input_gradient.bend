# The input gradient of a convolution by phases: output positions congruent
# modulo the stride read the same taps. Every row of every output channel of
# the output gradient is cut into eight-lane tiles once for each column shift
# the phases read, zero outside; a phase entry is then a tile index, and a job
# stores at the final strided addresses.
import Base
import ./storage_buffer.bend as Storage
import ./storage_allocation.bend as Allocation
import ./kernel_tile.bend as Tile
import ./kernel_rows.bend as Rows
import ./kernel_strided_store.bend as Stores
import ./proofs/kernel_strided_store_certificate.bend as StoreCertificate
import ./kernel_gemm_buffer.bend as Gemm
import ./kernel_spatial_packing.bend as Spatial
import ./kernel_window.bend as Window
import ./kernel_pipeline.bend as Pipeline
import ./traversal_loop.bend as Iteration
import ./traversal_array.bend as Traversal
import ./traversal_partition.bend as Partition
import ./tensor_band.bend as Bands
import ./tensor.bend as Tensors
import ./kernel_cost.bend as Cost
import ./default_kernel_cost.bend as DefaultCost
import ./kernel_planning.bend as KernelPlanning
import ./kernel_band_planning.bend as BandPlanning

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

# Tap coordinates and source strides for mapped weight reorder.
type WeightWindow is Data:
  WeightWindow{kernel: U32,stride: U32,y_tap: U32,y_rows: U32,x_tap: U32,x_columns: U32,tap_area: U32,output_step: U32}

def weight_window(context: Phase) -> WeightWindow:
  Phase{Geometry{c,o,h,w,+k,s,p,gh,gw},Axis{yf,yc,yt,+yr,ys},Axis{xf,xc,xt,+xr,xs},inner} = context
  WeightWindow{k,s,yt,yr,xt,xr,(yr * xr : U32),(c * k * k : U32)}

def weight_window_offset(window: WeightWindow,index: U32) -> U32:
  WeightWindow{+kernel,+stride,y_tap,y_rows,x_tap,+x_columns,tap_area,output_step} = window
  +index = index
  +row = (index / x_columns : U32)
  (index / tap_area * output_step + (y_tap - row % y_rows * stride) * kernel + x_tap - index % x_columns * stride : U32)

def weight_offset(context: Phase,index: U32) -> U32:
  weight_window_offset(weight_window(context),index)

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

type Fill is Type:
  Fill{input: Array<F32>,tiles: Array<Tile.Tile8>,index: U32,row: U32,column: U32,first: U32,base: U32,plane: U32,height: U32,width: U32,shifts: Nat}

def filled(tiles: Array<Tile.Tile8>,+index: U32,row: U32,+column: U32,first: U32,base: U32,plane: U32,height: U32,width: U32,shifts: Nat,read: Array<F32> & Tile.Tile8) -> Fill:
  (input,tile) = read
  Fill{input,Array.set(Tile.Tile8,tiles,index,tile),(index + 1 : U32),row,(column + 1 : U32),first,base,plane,height,width,shifts}

def fill_shift(state: Fill) -> Fill:
  Fill{input,tiles,index,+row,+column,first,+base,plane,+height,+width,shifts} = state
  filled(tiles,index,row,column,first,base,plane,height,width,shifts,Spatial.block(input,row,column,base,height,width,1,0,0,0,0,0))

def next_channel(state: Fill) -> Fill:
  Fill{input,tiles,index,row,column,+first,+base,+plane,height,width,shifts} = state
  Fill{input,tiles,index,row,first,first,(base + plane : U32),plane,height,width,shifts}

def fill_channel(state: Fill) -> Fill:
  Fill{input,tiles,index,row,column,first,base,plane,height,width,+shifts} = state
  next_channel(Iteration.run(~Fill,~fill_shift,shifts,Fill{input,tiles,index,row,column,first,base,plane,height,width,shifts}))

def fill_finished(state: Fill) -> Array<F32> & Array<Tile.Tile8>:
  Fill{input,tiles,index,row,column,first,base,plane,height,width,shifts} = state
  (input,tiles)

# Rows [first, first + rows) of every output channel of one image.
type Source is Type:
  Source{data: Array<F32>,base: U32,plane: U32,first: U32,rows: U32}

# Tile rows from tile row low and groups [group, group + span) of the tiling,
# read from a source.
type Filling is Data:
  Filling{context: Tiling,low: U32,base: U32,plane: U32,first: U32,rows: U32,group: U32,span: U32}

# One tile row and group: every output channel, every shift.
def fill_group(index: U32,context: Filling,owners: Array<F32> & Array<Tile.Tile8>) -> Array<F32> & Array<Tile.Tile8>:
  Filling{Tiling{groups,+shifts,+outputs,+column_low,+kernel,height,+width,+chunk_groups,chunk_rows},+low,+base,+plane,+first,+source_rows,+group,+span} = context
  (input,tiles) = owners
  +index = index
  +column = ((index % span + group) * 8 + column_low - kernel : U32)
  fill_finished(Iteration.run(~Fill,~fill_channel,U32.to_nat(outputs),
    Fill{input,tiles,((index / span * chunk_groups + index % span) * (outputs * shifts + 1) : U32),(index / span + low - kernel - first : U32),column,column,
      base,plane,source_rows,width,U32.to_nat(shifts)}))

type Jobs is Type:
  Jobs{tiles: Array<Tile.Tile8>,weights: Array<Tile.Tile8>,offsets: Array<U32>,output: Storage.Buffer<F32>}

type Context is Data:
  Context{source: U32,weight_base: U32}

type Loop is Type:
  Loop{tiles: Array<Tile.Tile8>,weights: Array<Tile.Tile8>,offsets: Array<U32>,index: U32,context: Context,accumulators: Rows.Rows<Tile.Tile8>}

def with_tile(weights: Array<Tile.Tile8>,offsets: Array<U32>,+index: U32,context: Context,
  accumulators: Rows.Rows<Tile.Tile8>,samples: Rows.Rows<F32>,read: Array<Tile.Tile8> & Tile.Tile8) -> Loop:
  (tiles,tile) = read
  Loop{tiles,weights,offsets,(index + 1 : U32),context,Rows.map(~Tile.Tile8,~Tile.Tile8,~F32,~Tile.update,accumulators,tile,samples)}

def with_offset(tiles: Array<Tile.Tile8>,weights: Array<Tile.Tile8>,+index: U32,context: Context,+source: U32,
  accumulators: Rows.Rows<Tile.Tile8>,samples: Rows.Rows<F32>,read: Array<U32> & U32) -> Loop:
  (offsets,offset) = read
  with_tile(weights,offsets,index,context,accumulators,samples,Array.get(Tile.Tile8,tiles,(source + offset : U32)))

def with_weights(~upper: Bool,tiles: Array<Tile.Tile8>,offsets: Array<U32>,+index: U32,context: Context,+source: U32,
  accumulators: Rows.Rows<Tile.Tile8>,read: Array<Tile.Tile8> & Tile.Tile8) -> Loop:
  (weights,tile) = read
  with_offset(tiles,weights,index,context,source,accumulators,Rows.half(upper,tile),Array.get(U32,offsets,index))

# One ordered step: the weight tile, the tile of the entry through the table,
# then the products of four rows at eight outputs.
def step(~upper: Bool,state: Loop) -> Loop:
  Loop{tiles,weights,offsets,+index,context,accumulators} = state
  Context{+source,+weight_base} = context
  with_weights(~upper,tiles,offsets,index,Context{source,weight_base},source,accumulators,
    Array.get(Tile.Tile8,weights,(weight_base + index : U32)))

def execute(upper: Bool,count: Nat,state: Loop) -> Loop:
  match upper:
    case False{}: Iteration.run(~Loop,~step(~False{}),count,state)
    case True{}: Iteration.run(~Loop,~step(~True{}),count,state)

# Raw stores within a job, one certificate update after them.
def stored(state: Loop,+store: Stores.Store,output: Storage.Buffer<F32>) -> Jobs:
  Loop{tiles,weights,offsets,index,context,+accumulators} = state
  Storage.Buffer{storage,length,certificate} = output
  Jobs{tiles,weights,offsets,Storage.Buffer{Stores.block(store,accumulators,storage),length,StoreCertificate.certificate(store,accumulators,storage,certificate)}}

# One job is four input channels at eight consecutive outputs of one phase
# row. The loop index runs over the phase table from offset_base; weight_base
# is the phase weights less offset_base.
type Plan is Data:
  Plan{units: U32,groups: U32,inner: U32,source_base: U32,row_step: U32,group_step: U32,
    target_base: U32,target_row: U32,step: U32,plane: U32,channels: U32,columns: U32,weight_base: U32,offset_base: U32}

def job(index: U32,plan: Plan,state: Jobs) -> Jobs:
  Plan{+units,+groups,+inner,+source_base,+row_step,+group_step,+target_base,+target_row,+step,+plane,+channels,+columns,+weight_base,+offset_base} = plan
  Jobs{tiles,weights,offsets,output} = state
  +index = index
  +unit = (index % units : U32)
  +group = (index / units % groups : U32)
  +line = (index / units / groups : U32)
  stored(execute(U32.is_eq((unit % 2 : U32),1),U32.to_nat(inner),
    Loop{tiles,weights,offsets,offset_base,Context{(source_base + line * row_step + group * group_step : U32),(weight_base + unit / 2 * inner : U32)},
      Rows.filled(~Tile.Tile8,Tile.filled(0.0))}),
    Stores.Store{(target_base + unit * 4 * plane + line * target_row + group * 8 * step : U32),step,plane,U32.min(4,(channels - unit * 4 : U32)),U32.min(8,(columns - group * 8 : U32))},output)

# The tile of entry (output o, tap row a, tap column b) of a phase, relative to
# the tile of the job row and group, at position base + entry of the table.
type Table is Data:
  Table{tap_rows: U32,tap_columns: U32,row_step: U32,shifts: U32,first_shift: U32,base: U32}

def offset(index: U32,context: Table,offsets: Array<U32>) -> Array<U32>:
  Table{+tap_rows,+tap_columns,+row_step,+shifts,+first_shift,+base} = context
  +index = index
  Array.set(U32,offsets,(base + index : U32),(index / tap_columns % tap_rows * row_step + index / (tap_rows * tap_columns) * shifts + index % tap_columns + first_shift : U32))

type TileCopy is Type:
  TileCopy{source: Array<Tile.Tile8>,target: Array<Tile.Tile8>,from: U32,to: U32}

def tile_copied(target: Array<Tile.Tile8>,+from: U32,+to: U32,read: Array<Tile.Tile8> & Tile.Tile8) -> TileCopy:
  (source,tile) = read
  TileCopy{source,Array.set(Tile.Tile8,target,to,tile),(from + 1 : U32),(to + 1 : U32)}

def tile_step(state: TileCopy) -> TileCopy:
  TileCopy{source,target,+from,+to} = state
  tile_copied(target,from,to,Array.get(Tile.Tile8,source,from))

def tile_target(state: TileCopy) -> Array<Tile.Tile8>:
  TileCopy{source,target,from,to} = state
  target

# Every phase weights and table, side by side in phase order.
type Packed is Type:
  Packed{source: Storage.Buffer<F32>,weights: Array<Tile.Tile8>,offsets: Array<U32>,weight_base: U32,offset_base: U32}

def packed_copied(+context: Phase,+layout: Tiling,source: Storage.Buffer<F32>,offsets: Array<U32>,+weight_base: U32,+offset_base: U32,+count: U32,weights: Array<Tile.Tile8>) -> Packed:
  Phase{Geometry{c,o,h,w,+k,s,p,gh,gw},Axis{yf,yc,yt,+yr,ys},Axis{xf,xc,xt,+xr,+xs},+inner} = context
  Tiling{groups,+shifts,+outputs,+column_low,kernel,height,width,+chunk_groups,chunk_rows} = layout
  Packed{source,weights,Traversal.indexed_fold(~Array<U32>,~Table,~offset,U32.to_nat(inner),0,
    Table{yr,xr,(chunk_groups * (outputs * shifts + 1) : U32),shifts,(xs + k - column_low : U32),offset_base},offsets),(weight_base + count : U32),(offset_base + inner : U32)}

def packed_phase(+context: Phase,+layout: Tiling,weights: Array<Tile.Tile8>,offsets: Array<U32>,+weight_base: U32,+offset_base: U32,read: Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>) -> Packed:
  Phase{Geometry{+c,o,h,w,k,s,p,gh,gw},y,x,+inner} = context
  (source,phase_weights) = read
  Storage.Buffer{phase_tiles,phase_length,phase_certificate} = phase_weights
  +count = ((c + 7) / 8 * inner : U32)
  packed_copied(context,layout,source,offsets,weight_base,offset_base,count,
    tile_target(Iteration.run(~TileCopy,~tile_step,U32.to_nat(count),TileCopy{phase_tiles,weights,0,weight_base})))

def pack_selected(empty: Bool,+context: Phase,+layout: Tiling,state: Packed) -> Packed:
  match empty:
    case True{}: state
    case False{}:
      Phase{Geometry{+c,o,h,w,+k,s,p,gh,gw},y,x,+inner} = context
      Packed{source,weights,offsets,weight_base,offset_base} = state
      packed_phase(context,layout,weights,offsets,weight_base,offset_base,
        Gemm.reorder_mapped(~WeightWindow,~weight_window_offset,c,inner,(k * k : U32),weight_window(context),source))

type PackContext is Data:
  PackContext{geometry: Geometry,layout: Tiling}

def pack_phase(index: U32,context: PackContext,state: Packed) -> Packed:
  PackContext{+geometry,+layout} = context
  +selected = phase_of(geometry,index)
  pack_selected(U32.is_eq(phase_inner(selected),0),selected,layout,state)

def inner_total(index: U32,geometry: Geometry,total: U32) -> U32:
  (total + phase_inner(phase_of(geometry,index)) : U32)

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
  Leaf{tiles: Array<Tile.Tile8>,weights: Array<Tile.Tile8>,offsets: Array<U32>,output: Storage.Buffer<F32>,weight_base: U32,offset_base: U32}

# Output rows [first, first + count) go to target + channel * plane +
# (row - first) * width + column.
type LeafContext is Data:
  LeafContext{task: Task,first: U32,count: U32,target: U32,plane: U32}

# Output rows [row, row + rows) and groups [group, group + chunk_groups) of a
# leaf, over a chunk of the tiling that starts at tile row low.
type Chunk is Data:
  Chunk{leaf: LeafContext,row: U32,rows: U32,low: U32,group: U32}

def leaf_jobs(+context: Phase,+chunk: Chunk,+weight_base: U32,+offset_base: U32,state: Jobs) -> Jobs:
  Phase{Geometry{+c,o,+h,+w,+k,+s,p,gh,gw},Axis{+yf,+yc,yt,yr,+ys},Axis{+xf,+xc,xt,xr,xs},+inner} = context
  Chunk{LeafContext{Task{geometry,+layout},+first,count,+target,+plane},+row,+rows,+low,+group} = chunk
  Tiling{groups,+shifts,+outputs,column_low,kernel,height,width,+chunk_groups,chunk_rows} = layout
  +start = Bool.pick(U32,U32.is_lt(yf,row),((row - yf + s - 1) / s : U32),0)
  +end = U32.min(yc,Bool.pick(U32,U32.is_lt(yf,(row + rows : U32)),((row + rows - yf + s - 1) / s : U32),0))
  +units = ((c + 3) / 4 : U32)
  +phase_groups = ((xc + 7) / 8 : U32)
  +span = Bool.pick(U32,U32.is_lt(group,phase_groups),U32.min(chunk_groups,(phase_groups - group : U32)),0)
  +group_step = (outputs * shifts + 1 : U32)
  Traversal.indexed_fold(~Jobs,~Plan,~job,U32.to_nat(Bool.pick(U32,U32.is_lt(start,end),((end - start) * span * units : U32),0)),0,
    Plan{units,span,inner,((start + ys + k - low) * chunk_groups * group_step : U32),(chunk_groups * group_step : U32),group_step,
      (target + (yf + start * s - first) * w + xf + group * 8 * s : U32),(s * w : U32),s,plane,c,(xc - group * 8 : U32),(weight_base - offset_base : U32),offset_base},state)

def leaf_finished(+weight_base: U32,+offset_base: U32,state: Jobs) -> Leaf:
  Jobs{tiles,weights,offsets,output} = state
  Leaf{tiles,weights,offsets,output,weight_base,offset_base}

def leaf_selected(empty: Bool,+context: Phase,+chunk: Chunk,state: Leaf) -> Leaf:
  match empty:
    case True{}: state
    case False{}:
      Phase{Geometry{+c,o,h,w,k,s,p,gh,gw},y,x,+inner} = context
      Leaf{tiles,weights,offsets,output,+weight_base,+offset_base} = state
      leaf_finished((weight_base + (c + 7) / 8 * inner : U32),(offset_base + inner : U32),
        leaf_jobs(context,chunk,weight_base,offset_base,Jobs{tiles,weights,offsets,output}))

def leaf_phase(index: U32,chunk: Chunk,state: Leaf) -> Leaf:
  Chunk{LeafContext{Task{+geometry,layout},first,count,target,plane},row,rows,low,group} = chunk
  +selected = phase_of(geometry,index)
  leaf_selected(U32.is_eq(phase_inner(selected),0),selected,chunk,state)

# The gradient a leaf read and the output it wrote.
type Part is Type:
  Part{gradient: Array<F32>,output: Storage.Buffer<F32>}

# A leaf runs its chunks in turn with the source, the weights and the table.
type Sweep is Type:
  Sweep{source: Source,weights: Array<Tile.Tile8>,offsets: Array<U32>,output: Storage.Buffer<F32>}

def chunk_leaf(+base: U32,+plane: U32,+first: U32,+rows: U32,data: Array<F32>,state: Leaf) -> Sweep:
  Leaf{tiles,weights,offsets,output,weight_base,offset_base} = state
  Sweep{Source{data,base,plane,first,rows},weights,offsets,output}

def chunk_finished(+base: U32,+plane: U32,+first: U32,+rows: U32,pair: Array<F32> & Leaf) -> Sweep:
  (data,state) = pair
  chunk_leaf(base,plane,first,rows,data,state)

def chunk_run(+chunk: Chunk,weights: Array<Tile.Tile8>,offsets: Array<U32>,output: Storage.Buffer<F32>,owners: Array<F32> & Array<Tile.Tile8>) -> Array<F32> & Leaf:
  Chunk{LeafContext{Task{+geometry,layout},first,count,target,plane},row,rows,low,group} = chunk
  (data,tiles) = owners
  (data,Traversal.indexed_fold(~Leaf,~Chunk,~leaf_phase,window_jobs(geometry),0,chunk,Leaf{tiles,weights,offsets,output,0,0}))

# Chunk index: row chunk outer, group chunk inner.
def chunk_step(index: U32,context: LeafContext,state: Sweep) -> Sweep:
  LeafContext{Task{+geometry,+layout},+first,+count,target,plane} = context
  Tiling{+groups,+shifts,+outputs,column_low,kernel,height,width,+chunk_groups,+chunk_rows} = layout
  Sweep{Source{data,+base,+source_plane,+source_first,+source_rows},weights,offsets,output} = state
  +index = index
  +group_chunks = ((groups + chunk_groups - 1) / chunk_groups : U32)
  +row = (first + index / group_chunks * chunk_rows : U32)
  +rows = U32.min(chunk_rows,(first + count - row : U32))
  +group = (index % group_chunks * chunk_groups : U32)
  +span = U32.min(chunk_groups,(groups - group : U32))
  +low = band_low(geometry,row,(row + rows : U32))
  +tile_rows = band_rows(geometry,row,rows)
  chunk_finished(base,source_plane,source_first,source_rows,
    chunk_run(Chunk{context,row,rows,low,group},weights,offsets,output,
      Traversal.indexed_fold(~(Array<F32> & Array<Tile.Tile8>),~Filling,~fill_group,U32.to_nat((tile_rows * span : U32)),0,
        Filling{layout,low,base,source_plane,source_first,source_rows,group,span},
        (data,Allocation.array(~Tile.Tile8,Tensors.depth_for((tile_rows * chunk_groups * (outputs * shifts + 1) + 1 : U32)),Tile.filled(0.0))))))

def swept(state: Sweep) -> Part:
  Sweep{Source{data,base,plane,first,rows},weights,offsets,output} = state
  Part{data,output}

# Output rows [first, first + count) from a source holding the gradient rows they read.
def leaf_band(+context: LeafContext,source: Source,weights: Array<Tile.Tile8>,offsets: Array<U32>,output: Storage.Buffer<F32>) -> Part:
  LeafContext{Task{geometry,+layout},first,+count,target,plane} = context
  Tiling{+groups,shifts,outputs,column_low,kernel,height,width,+chunk_groups,+chunk_rows} = layout
  swept(Traversal.indexed_fold(~Sweep,~LeafContext,~chunk_step,
    U32.to_nat(((count + chunk_rows - 1) / chunk_rows * ((groups + chunk_groups - 1) / chunk_groups) : U32)),0,context,Sweep{source,weights,offsets,output}))

# What one part owns: the gradient rows it reads, the weights and the table.
type Owners is Type:
  Owners{source: Source,weights: Array<Tile.Tile8>,offsets: Array<U32>}

# A leaf writes output rows [first, first + count), stored [C][count][W].
def leaf(task: Task,first: U32,count: U32,owners: Owners) -> Partition.Tree<&1,Part,Unit>:
  Task{+geometry,+layout} = task
  Geometry{+c,o,h,+w,k,s,p,gh,gw} = geometry
  Owners{source,weights,offsets} = owners
  +first = first
  +count = count
  Partition.Leaf{leaf_band(LeafContext{Task{geometry,layout},first,count,0,(count * w : U32)},
    source,weights,offsets,Tensors.buffer(~F32,(0.0 + 0.0 : F32),(c * count * w : U32))),first,count,Unit{}}

def copied_pair(+start: U32,+count: U32,+width: U32,+base: U32,+plane: U32,+first: U32,+rows: U32,owners: Array<F32> & Array<F32>) -> Source & Source:
  (data,copy) = owners
  (Source{copy,0,(count * width : U32),start,count},Source{data,base,plane,first,rows})

# The gradient rows [start, start + count) that output rows [first, first + rows) read.
def copy_start(+geometry: Geometry,+first: U32,+rows: U32) -> U32:
  Geometry{c,o,h,w,+k,s,p,gh,gw} = geometry
  +low = band_low(geometry,first,(first + rows : U32))
  Bool.pick(U32,U32.is_lt(k,low),(low - k : U32),0)

def copy_count(+geometry: Geometry,+first: U32,+rows: U32) -> U32:
  Geometry{c,o,h,w,+k,s,p,+gh,gw} = geometry
  +start = copy_start(geometry,first,rows)
  +high = band_high(geometry,first,(first + rows : U32))
  +stop = Bool.pick(U32,U32.is_lt(k,high),U32.min((high - k : U32),gh),0)
  Bool.pick(U32,U32.is_lt(start,stop),(stop - start : U32),0)

# The left child, output rows [row, row + half), copies the gradient rows it
# reads; the right child keeps the source of its parent.
def split(task: Task,row: U32,half: U32,source: Source) -> Source & Source:
  Task{+geometry,layout} = task
  Geometry{c,+o,h,w,k,s,p,gh,+gw} = geometry
  Source{data,+base,+plane,+first,+rows} = source
  +row = row
  +half = half
  +start = copy_start(geometry,row,half)
  +count = copy_count(geometry,row,half)
  copied_pair(start,count,gw,base,plane,first,rows,
    Window.copy_planes(U32.to_nat(o),0,(count * gw : U32),plane,(base + (start - first) * gw : U32),(count * gw : U32),
      (data,Allocation.array(~F32,Tensors.depth_for((o * count * gw + 1 : U32)),0.0))))

def divided(sources: Source & Source,weights: Array<Tile.Tile8> & Array<Tile.Tile8>,offsets: Array<U32> & Array<U32>) -> Bands.Division<Owners>:
  (left_source,right_source) = sources
  (left_weights,right_weights) = weights
  (left_offsets,right_offsets) = offsets
  Bands.Divided{Owners{left_source,left_weights,left_offsets},Owners{right_source,right_weights,right_offsets}}

# The left half reads the rows of its plan; each half owns the weights and the table.
def divide(task: Task,left: Bands.Plan,right: Bands.Plan,owners: Owners) -> Bands.Division<Owners>:
  Owners{source,weights,offsets} = owners
  +row = Bands.first(left)
  divided(split(task,row,(Bands.first(right) - row : U32),source),Array.clone(Tile.Tile8,weights),Array.clone(U32,offsets))

# A split always divides the owners; kept whole, they would compute the whole
# plan as one part.
def whole(task: Task,layout: Bands.Plan,owners: Owners) -> Partition.Tree<&1,Part,Unit>:
  +layout = layout
  leaf(task,Bands.first(layout),(Bands.last(layout) - Bands.first(layout) : U32),owners)

# A leaf output [C][count][W] is the band of its rows of the output, counted
# over the batch from start. Its gradient replaces the one carried so far, so
# the gradient of the last leaf, the original, comes back.
def placed(start: U32,first: U32,count: U32,part: Part,gradient: Array<F32>) -> Bands.Bands(~F32) & Array<F32>:
  Part{data,local} = part
  (Partition.Leaf{local,(start + first : U32),count,Unit{}},data)

# The split depth, priced with the shared cost configuration. A leaf builds
# the tiling of the gradient rows it reads, runs its jobs and stores its
# rows. A split copies the left band and clones the weights and the table.
# A split image is one parallel region with two wake rounds, so the batch
# does not change the choice.
type Pricing is Data:
  Pricing{geometry: Geometry,layout: Tiling,lanes: F32,clone: F32,config: Cost.Configuration}

type Movement is Data:
  Movement{work: F32,span: F32,leaves: U32}

type Choice is Data:
  Choice{depth: U32,time: F32}

# Lanes are computed in eight-wide groups of each phase row.
def phase_lanes(context: Phase,total: F32) -> F32:
  Phase{g,Axis{yf,+yc,yt,yr,ys},Axis{xf,+xc,xt,xr,xs},+inner} = context
  (total + U32.to_f32(yc) * U32.to_f32(((xc + 7) / 8 * 8 : U32)) * U32.to_f32(inner) : F32)

def phase_cost(index: U32,geometry: Geometry,total: F32) -> F32:
  phase_lanes(phase_of(geometry,index),total)

def leaf_ns(+context: Pricing,+first: U32,+count: U32) -> F32:
  Pricing{+geometry,+layout,+lanes,clone,+config} = context
  Geometry{+c,o,+h,+w,k,s,p,gh,gw} = geometry
  Tiling{+groups,+shifts,+outputs,column_low,kernel,height,width,+chunk_groups,+chunk_rows} = layout
  Cost.Configuration{+mac,+byte_ns,operation_ns,grain,wake} = config
  # Each row chunk fills the tile rows its output rows read, halo included.
  +rows = band_rows(geometry,first,U32.min(count,chunk_rows))
  +row_chunks = ((count + chunk_rows - 1) / chunk_rows : U32)
  +chunk = (rows * chunk_groups * (outputs * shifts + 1) : U32)
  +chunks = U32.to_f32((row_chunks * ((groups + chunk_groups - 1) / chunk_groups) : U32))
  ((U32.to_f32(row_chunks) * U32.to_f32(rows) * U32.to_f32(groups) * U32.to_f32((outputs * shifts + 1 : U32)) * 32.0 +
    chunks * KernelPlanning.payload_float(Nat.mul(32n,KernelPlanning.capacity(chunk)))) * byte_ns +
    U32.to_f32(((c + 3) / 4 * 4 : U32)) * lanes * U32.to_f32(count) / U32.to_f32(U32.max(h,1)) * mac +
    U32.to_f32((c * count * w : U32)) * 4.0 * byte_ns : F32)

def fork_ns(+context: Pricing,+first: U32,+rows: U32) -> F32:
  Pricing{+geometry,layout,lanes,+clone,+config} = context
  Geometry{c,+o,h,w,k,s,p,gh,+gw} = geometry
  Cost.Configuration{mac,+byte_ns,operation_ns,grain,wake} = config
  (BandPlanning.transfer((o * copy_count(geometry,first,rows) * gw : U32),byte_ns) + clone : F32)

def merged(+fork: F32,left: Movement,right: Movement) -> Movement:
  Movement{left_work,+left_span,left_leaves} = left
  Movement{right_work,+right_span,right_leaves} = right
  Movement{(fork + left_work + right_work : F32),(fork + F32.max(left_span,right_span) : F32),(left_leaves + right_leaves : U32)}

def leaf_movement(context: Pricing,first: U32,count: U32) -> Movement:
  +time = leaf_ns(context,first,count)
  Movement{time,time,1}

def fork_movement(context: Pricing,left: Bands.Plan,right: Bands.Plan,first: Movement,second: Movement) -> Movement:
  +row = Bands.first(left)
  merged(fork_ns(context,row,(Bands.first(right) - row : U32)),first,second)

# The plan the partition runs: the rows of one image halved depth times.
def image_rows(depth: Nat,+context: Pricing) -> Bands.Plan:
  Pricing{+geometry,layout,lanes,clone,config} = context
  Geometry{c,o,+h,w,k,s,p,gh,gw} = geometry
  Bands.row_plan(depth,U32.is_lt(1,h),0,h)

def movement(depth: Nat,+context: Pricing) -> Movement:
  Bands.plan_fold(~Movement,~Pricing,~leaf_movement,~fork_movement,image_rows(depth,context),context)

def estimate(+depth: U32,+workers: U32,+wake: F32,cost: Movement) -> F32:
  Movement{+work,+span,+leaves} = cost
  Bool.pick(F32,U32.is_zero(depth),work,(Cost.region_ns(workers,leaves,span) + 2.0 * wake : F32))

def movement_work(cost: Movement) -> F32:
  Movement{work,span,leaves} = cost
  work

def chosen(allowed: Bool,+depth: U32,+time: F32,best: Choice) -> Choice:
  Choice{old_depth,+old_time} = best
  Bool.pick(Choice,allowed && F32.is_lt(time,old_time),Choice{depth,time},Choice{old_depth,old_time})

def candidate(+depth: U32,+workers: U32,+context: Pricing,best: Choice) -> Choice:
  Pricing{geometry,layout,lanes,clone,+config} = context
  Cost.Configuration{mac,byte_ns,operation_ns,+grain,+wake} = config
  +cost = movement(U32.to_nat(depth),context)
  chosen(U32.is_zero(depth) || F32.is_le(grain,movement_work(cost)),depth,
    estimate(depth,workers,wake,cost),best)

# As the other planners: up to five leaves for each worker, at most one leaf a
# row, one worker never splits.
def depths(remaining: Nat,allowed: Bool,+depth: U32,+tasks: U32,+ceiling: U32,+workers: U32,+context: Pricing,best: Choice) -> Choice:
  match remaining allowed:
    case 0n allowed: best
    case 1n+rest False{}: best
    case 1n+rest True{}:
      Pricing{+geometry,layout,lanes,clone,config} = context
      Geometry{c,o,+h,w,k,s,p,gh,gw} = geometry
      +next = (tasks * 2 : U32)
      depths(rest,U32.is_le(next,ceiling) && U32.is_le(next,h),(depth + 1 : U32),next,ceiling,workers,context,candidate(depth,workers,context,best))

# One worker, no split: the work of the whole image.
def baseline(+context: Pricing) -> Choice:
  Choice{0,movement_work(movement(0n,context))}

def chosen_depth(best: Choice) -> Nat:
  Choice{depth,time} = best
  U32.to_nat(depth)

def byte_cost(config: Cost.Configuration) -> F32:
  Cost.Configuration{mac,byte_ns,operation_ns,grain,wake} = config
  byte_ns

def select(+geometry: Geometry,+layout: Tiling,+workers: U32,+weight_tiles: U32,+entries: U32) -> Nat:
  Geometry{c,o,+h,w,k,s,p,gh,gw} = geometry
  +config = DefaultCost.configuration()
  +workers = U32.max(1,workers)
  +context = {Pricing{geometry,layout,Traversal.indexed_fold(~F32,~Geometry,~phase_cost,window_jobs(geometry),0,geometry,0.0),
    (KernelPlanning.payload_float(Nat.add(Nat.mul(32n,KernelPlanning.capacity(weight_tiles)),Nat.mul(4n,KernelPlanning.capacity(entries)))) * byte_cost(config) : F32),config} : Pricing}
  chosen_depth(depths(32n,U32.is_lt(1,workers) && U32.is_le(2,h),1,2,(workers * 5 : U32),workers,context,baseline(context)))

type Images is Data:
  Images{geometry: Geometry,layout: Tiling,depth: Nat}

type ImageState is Type:
  ImageState{gradient: Array<F32>,weights: Array<Tile.Tile8>,offsets: Array<U32>}

def image_bands(weights: Array<Tile.Tile8>,offsets: Array<U32>,result: Bands.Bands(~F32) & Array<F32>) -> ImageState & Bands.Bands(~F32):
  (bands,gradient) = result
  (ImageState{gradient,weights,offsets},bands)

def image_offsets(+first: U32,+context: Images,gradient: Array<F32>,weights: Array<Tile.Tile8>,kept_weights: Array<Tile.Tile8>,pair: Array<U32> & Array<U32>) -> ImageState & Bands.Bands(~F32):
  Images{+geometry,+layout,+depth} = context
  Geometry{+c,+o,+h,+w,k,s,p,+gh,+gw} = geometry
  (offsets,kept_offsets) = pair
  image_bands(kept_weights,kept_offsets,Bands.walk(~Part,~Storage.Buffer<F32>,~Array<F32>,~U32,~placed,first,
    Bands.each_part(~Owners,~Part,~Unit,~Task,~Unit,~Bands.always,~divide,~leaf,~whole,Bands.row_plan(depth,U32.is_lt(1,h),0,h),True{},Unit{},Task{geometry,layout},
      Owners{Source{gradient,(first / h * o * gh * gw : U32),(gh * gw : U32),0,gh},weights,offsets}),
    Array.new(F32,0n,0.0)))

def image_weights(+first: U32,+context: Images,gradient: Array<F32>,offsets: Array<U32>,pair: Array<Tile.Tile8> & Array<Tile.Tile8>) -> ImageState & Bands.Bands(~F32):
  (weights,kept_weights) = pair
  image_offsets(first,context,gradient,weights,kept_weights,Array.clone(U32,offsets))

# The image whose rows start at first runs its plan, a single leaf when the
# depth is 0. Its leaves are its bands of the output.
def image(context: Images,first: U32,rows: U32,state: ImageState) -> ImageState & Bands.Bands(~F32):
  Images{+geometry,+layout,+depth} = context
  ImageState{gradient,weights,offsets} = state
  image_weights(first,Images{geometry,layout,depth},gradient,offsets,Array.clone(Tile.Tile8,weights))

type Outcome is Type:
  Outcome{weights: Storage.Buffer<F32>,output: Bands.Bands(~F32)}

def finished(source: Storage.Buffer<F32>,result: ImageState & Bands.Bands(~F32)) -> Outcome:
  (state,bands) = result
  Outcome{source,bands}

# The images in order, each one band of rows of the batch.
def packed(+geometry: Geometry,+layout: Tiling,+batch: Nat,+depth: Nat,gradient: Array<F32>,state: Packed) -> Outcome:
  Geometry{c,o,+h,w,k,s,p,gh,gw} = geometry
  Packed{source,weights,offsets,weight_base,offset_base} = state
  finished(source,Bands.build(~Storage.Buffer<F32>,~ImageState,~Images,~image,Bands.image_plan(batch,0,0n,h),Images{geometry,layout,depth},ImageState{gradient,weights,offsets}))

def tiled(+geometry: Geometry,batch: Nat,+workers: U32,gradient: Array<F32>,weights: Storage.Buffer<F32>) -> Outcome:
  +layout = tiling(geometry)
  +phases = window_jobs(geometry)
  +weight_tiles = Traversal.indexed_fold(~U32,~Geometry,~weight_total,phases,0,geometry,0)
  +entries = Traversal.indexed_fold(~U32,~Geometry,~inner_total,phases,0,geometry,0)
  packed(geometry,layout,batch,select(geometry,layout,workers,weight_tiles,entries),gradient,
    Traversal.indexed_fold(~Packed,~PackContext,~pack_phase,phases,0,PackContext{geometry,layout},
      Packed{weights,Allocation.array(~Tile.Tile8,Tensors.depth_for((weight_tiles + 1 : U32)),Tile.filled(0.0)),
        Allocation.array(~U32,Tensors.depth_for((entries + 1 : U32)),0),0,0}))

def run_selected(empty: Bool,+geometry: Geometry,batch: Nat,+workers: U32,gradient: Storage.Buffer<F32>,weights: Storage.Buffer<F32>) -> Outcome:
  match empty:
    case True{}: Outcome{weights,Partition.Leaf{Tensors.buffer(~F32,0.0,0),0,0,Unit{}}}
    case False{}:
      Storage.Buffer{data,length,certificate} = gradient
      tiled(geometry,batch,workers,data,weights)

# d [N][O][Ho][Wo] and weights [O][C][k][k] give the input gradient
# [N][C][H][W] as bands of rows of the batch; count is N * C * H * W.
def run(+geometry: Geometry,count: Nat,+workers: U32,gradient: Storage.Buffer<F32>,weights: Storage.Buffer<F32>) -> Outcome:
  Geometry{+c,o,+h,+w,k,s,p,gh,gw} = geometry
  +batch = Nat.div(count,U32.to_nat((c * h * w : U32)))
  run_selected(Nat.is_eq(batch,0n),geometry,batch,workers,gradient,weights)
