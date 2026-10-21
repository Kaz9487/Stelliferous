# Gradient tiles are the GEMM input, so multiplication keeps gradient first.
# GEMM rows are kernel taps. The final read changes [tap,output] to [output,tap].
import Base
import ./kernel_parameter_gradient.bend as Parameter
import ./kernel_gemm.bend as Gemm
import ./kernel_tile.bend as Tile
import ./storage_buffer.bend as Storage
import ./traversal_array.bend as Traversal
import ./traversal_loop.bend as Loop
import ./traversal_grouped.bend as Grouped
import ./traversal_lanes.bend as Lanes
import ./tensor_lanes.bend as TensorLanes
import ./storage_clone.bend as StorageClone
import ./tensor.bend as Tensors
import ./tensor_band.bend as Bands
import ./tensor_band_schedule.bend as BandSchedule
import ./traversal_partition.bend as Partition
import ./kernel_cost.bend as Cost
import ./default_kernel_cost.bend as DefaultCost

type Geometry is Data:
  Geometry{channels: U32,outputs: U32,height: U32,width: U32,kernel: U32,stride: U32,padding: U32,rows: U32,columns: U32,taps: U32}

def geometry(work: Parameter.Work) -> Geometry:
  Parameter.Work{n,+c,o,h,w,+k,s,p,rows,columns,g} = work
  Geometry{U32.from_nat(c),U32.from_nat(o),U32.from_nat(h),U32.from_nat(w),U32.from_nat(k),U32.from_nat(s),U32.from_nat(p),U32.from_nat(rows),U32.from_nat(columns),U32.from_nat(Nat.mul(c,Nat.mul(k,k)))}

def array(~Element: Data,+count: U32,value: Element) -> Array<Element>:
  Array.new(Element,Storage.depth_for(count),value)

type Lane is Data:
  Lane{base: U32,row: U32,column: U32,valid: Bool,row_first: U32,row_end: U32,column_first: U32,column_end: U32}

def axis_first(+offset: U32,+stride: U32,padding: U32) -> U32:
  U32.from_nat(Nat.div(Nat.add(Nat.sub(U32.to_nat(padding),U32.to_nat(offset)),Nat.sub(U32.to_nat(stride),1n)),U32.to_nat(stride)))

def axis_end(+offset: U32,+stride: U32,padding: U32,size: U32,positions: U32) -> U32:
  U32.from_nat(Nat.min(U32.to_nat(positions),Nat.div(Nat.add(Nat.sub(Nat.add(U32.to_nat(padding),U32.to_nat(size)),U32.to_nat(offset)),Nat.sub(U32.to_nat(stride),1n)),U32.to_nat(stride))))

def lane(+tap: U32,geometry: Geometry,image: U32) -> Lane:
  Geometry{c,o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  +kr = (tap / k % k : U32)
  +kc = (tap % k : U32)
  Lane{((image * c + tap / (k * k)) * h * w : U32),kr,kc,U32.is_lt(tap,taps),
    axis_first(kr,s,p),axis_end(kr,s,p,h,rows),axis_first(kc,s,p),axis_end(kc,s,p,w,columns)}

type Patch is Data:
  Patch{a: Lane,b: Lane,c: Lane,d: Lane,e: Lane,f: Lane,g: Lane,h: Lane}

def patch(+tap: U32,+geometry: Geometry,+image: U32) -> Patch:
  Patch{lane(tap,geometry,image),lane((tap + 1 : U32),geometry,image),lane((tap + 2 : U32),geometry,image),lane((tap + 3 : U32),geometry,image),
    lane((tap + 4 : U32),geometry,image),lane((tap + 5 : U32),geometry,image),lane((tap + 6 : U32),geometry,image),lane((tap + 7 : U32),geometry,image)}

def selected(patch: Patch,index: Nat) -> Lane:
  Patch{a,b,c,d,e,f,g,h} = patch
  match index:
    case 0n: a
    case 1n: b
    case 2n: c
    case 3n: d
    case 4n: e
    case 5n: f
    case 6n: g
    case index: h

type Pixel is Data:
  Pixel{patch: Patch,row: U32,column: U32,height: U32,width: U32,stride: U32,padding: U32}

def pixel(input: Storage.Buffer<F32>,+row: U32,+column: U32,+height: U32,+width: U32,+stride: U32,+padding: U32,lane: Lane) -> Storage.Buffer<F32> & F32:
  Lane{base,kr,kc,valid,rf,re,cf,ce} = lane
  +y = (row * stride + kr - padding : U32)
  +x = (column * stride + kc - padding : U32)
  Storage.read_masked(~F32,0.0,valid && U32.is_le(rf,row) && U32.is_lt(row,re) && U32.is_le(cf,column) && U32.is_lt(column,ce),input,(base + y * width + x : U32))

def read_pixel(input: Storage.Buffer<F32>,context: Pixel,index: Nat) -> Storage.Buffer<F32> & F32:
  Pixel{patch,row,column,height,width,stride,padding} = context
  pixel(input,row,column,height,width,stride,padding,selected(patch,index))

type Packing is Type:
  Packing{input: Storage.Buffer<F32>,output: Array<Tile.Tile8>,index: U32,row: U32,column: U32,columns: U32,patch: Patch,height: U32,width: U32,stride: U32,padding: U32}

def advanced(wrap: Bool,input: Storage.Buffer<F32>,output: Array<Tile.Tile8>,index: U32,+row: U32,column: U32,columns: U32,patch: Patch,height: U32,width: U32,stride: U32,padding: U32) -> Packing:
  match wrap:
    case False{}: Packing{input,output,index,row,column,columns,patch,height,width,stride,padding}
    case True{}: Packing{input,output,index,(row + 1 : U32),0,columns,patch,height,width,stride,padding}

def packed(output: Array<Tile.Tile8>,+index: U32,row: U32,+column: U32,+columns: U32,patch: Patch,height: U32,width: U32,stride: U32,padding: U32,result: Storage.Buffer<F32> & Tile.Tile8) -> Packing:
  (input,value) = result
  advanced(U32.is_eq((column + 1 : U32),columns),input,Array.set(Tile.Tile8,output,index,value),(index + 1 : U32),row,(column + 1 : U32),columns,patch,height,width,stride,padding)

def pack_step(state: Packing) -> Packing:
  Packing{input,output,index,+row,+column,columns,+patch,+height,+width,+stride,+padding} = state
  +context = {Pixel{patch,row,column,height,width,stride,padding} : Pixel}
  packed(output,index,row,column,columns,patch,height,width,stride,padding,
    Tile.read_tail(~Storage.Buffer<F32>,~Pixel,~read_pixel,read_pixel(input,context,0n),context))

type Gradient is Data:
  Gradient{outputs: U32,spatial: U32,start: U32,row: U32,destination: U32}

def gradient_pixel(input: Storage.Buffer<F32>,context: Gradient,lane: Nat,position: U32) -> Storage.Buffer<F32> & F32:
  Gradient{outputs,+spatial,+start,row,destination} = context
  +row = (row + U32.from_nat(lane) : U32)
  Storage.read_masked(~F32,0.0,U32.is_lt(row,outputs),input,(row * spatial + start + position : U32))

type GradientRead is Data:
  GradientRead{context: Gradient,position: U32}

def gradient_lane(input: Storage.Buffer<F32>,context: GradientRead,lane: Nat) -> Storage.Buffer<F32> & F32:
  GradientRead{context,position} = context
  gradient_pixel(input,context,lane,position)

def gradient_tile(input: Storage.Buffer<F32>,context: Gradient,position: U32) -> Storage.Buffer<F32> & Tile.Tile8:
  +context = context
  +position = position
  Tile.read_tail(~Storage.Buffer<F32>,~GradientRead,~gradient_lane,gradient_pixel(input,context,0n,position),GradientRead{context,position})

def gradient_destination(context: Gradient,position: U32) -> U32:
  position

type WeightWindow is Data:
  WeightWindow{geometry: Geometry,image: U32,tap: U32,output: U32,row: U32,column: U32,first: U32}

def window_input(window: WeightWindow,+count: U32,input: Storage.Buffer<F32>) -> Packing:
  WeightWindow{+geometry,image,tap,output,row,column,first} = window
  Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  Loop.run(~Packing,~pack_step,U32.to_nat(count),
    Packing{input,array(~Tile.Tile8,count,Tile.filled(0.0)),0,row,column,columns,patch(tap,geometry,image),h,w,s,p})

def window_gradient(window: WeightWindow,+count: U32,input: Storage.Buffer<F32>) -> Storage.Buffer<F32> & Array<Tile.Tile8>:
  WeightWindow{geometry,image,tap,output,row,column,first} = window
  Geometry{c,+o,h,w,k,s,p,+rows,+columns,taps} = geometry
  Traversal.finish(~Storage.Buffer<F32>,~Tile.Tile8,~Gradient,
    Traversal.run(~Storage.Buffer<F32>,~Tile.Tile8,~Gradient,~gradient_tile,~gradient_destination,U32.to_nat(count),
      Traversal.State{input,array(~Tile.Tile8,count,Tile.filled(0.0)),0,
        Gradient{o,(rows * columns : U32),(image * o * rows * columns + first : U32),output,0}}))

type ChunkState is Type:
  ChunkState{input: Storage.Buffer<F32>,gradient: Storage.Buffer<F32>,values: Array<F32>,remaining: U32,window: WeightWindow}

def window_gemm(continued: Bool,count: U32,input: Array<Tile.Tile8>,weights: Array<Tile.Tile8>,output: Array<F32>) -> Array<F32>:
  match continued:
    case False{}: Gemm.result(Gemm.run_keeping(~False{},8,input,weights,array(~F32,1,0.0),output,8,count,0,8))
    case True{}: Gemm.result(Gemm.continue_keeping(8,input,weights,array(~F32,1,0.0),output,8,count))

def chunk_result(~continued: Bool,+count: U32,+remaining: U32,+window: WeightWindow,values: Array<F32>,packed: Packing,read: Storage.Buffer<F32> & Array<Tile.Tile8>) -> ChunkState:
  WeightWindow{geometry,image,tap,output,old_row,old_column,first} = window
  Packing{input,weights,index,row,column,columns,patch,h,w,s,p} = packed
  (gradient,tiles) = read
  ChunkState{input,gradient,window_gemm(continued,count,tiles,weights,values),(remaining - count : U32),
    WeightWindow{geometry,image,tap,output,row,column,(first + count : U32)}}

def chunk(~continued: Bool,state: ChunkState) -> ChunkState:
  ChunkState{input,gradient,values,+remaining,+window} = state
  +count = U32.min(256,remaining)
  chunk_result(~continued,count,remaining,window,values,window_input(window,count,input),window_gradient(window,count,gradient))

def chunk_values(state: ChunkState) -> (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>:
  ChunkState{input,gradient,values,remaining,window} = state
  ((input,gradient),values)

def window_product(+count: U32,+window: WeightWindow,input: Storage.Buffer<F32>,gradient: Storage.Buffer<F32>) ->
  (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>:
  +turns = (((count - 1) / 256) + 1 : U32)
  chunk_values(Loop.run(~ChunkState,~chunk(~True{}),U32.to_nat((turns - 1 : U32)),
    chunk(~False{},ChunkState{input,gradient,array(~F32,64,0.0),count,window})))

def copied(valid: Bool,+address: U32,output: Array<F32>,read: Array<F32> & F32) -> Array<F32> & Array<F32>:
  (input,value) = read
  (input,Gemm.store_lane(valid,output,address,value))

type BlockCopy is Data:
  BlockCopy{tap: U32,output: U32,taps: U32,outputs: U32}

def copy_cell(index: U32,context: BlockCopy,owners: Array<F32> & Array<F32>) -> Array<F32> & Array<F32>:
  BlockCopy{+tap,+output,taps,+outputs} = context
  (input,destination) = owners
  +index = index
  +row = (tap + index / 8 : U32)
  +column = (output + index % 8 : U32)
  copied(U32.is_lt(row,taps) && U32.is_lt(column,outputs),(row * outputs + column : U32),destination,Array.get(F32,input,index))

def block_result(+context: BlockCopy,output: Array<F32>,result: (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>) ->
  (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>:
  (sources,input) = result
  (sources,Pair.snd(Array<F32>,Array<F32>,Traversal.indexed_fold(~(Array<F32> & Array<F32>),~BlockCopy,~copy_cell,64n,0,context,(input,output))))

type GroupBlocks is Data:
  GroupBlocks{window: WeightWindow,count: U32,output_blocks: U32}

def block(index: U32,context: GroupBlocks,owners: (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>) ->
  (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>:
  GroupBlocks{WeightWindow{+geometry,+image,tap,output,+row,+column,+first},count,+output_blocks} = context
  Geometry{c,+o,h,w,k,s,p,rows,columns,+taps} = geometry
  ((input,gradient),values) = owners
  +index = index
  +tap = (index / output_blocks * 8 : U32)
  +output = (index % output_blocks * 8 : U32)
  block_result(BlockCopy{tap,output,taps,o},values,window_product(count,WeightWindow{geometry,image,tap,output,row,column,first},input,gradient))

def group(+geometry: Geometry,+image: U32,range: Grouped.Range,input: Storage.Buffer<F32>,gradient: Storage.Buffer<F32>) ->
  (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>:
  Geometry{c,+o,h,w,k,s,p,rows,+columns,+taps} = geometry
  Grouped.Range{+first,count} = range
  +output_blocks = ((o + 7) / 8 : U32)
  +context = {GroupBlocks{WeightWindow{geometry,image,0,0,U32.from_nat(Nat.div(first,U32.to_nat(columns))),U32.from_nat(Nat.mod(first,U32.to_nat(columns))),U32.from_nat(first)},U32.from_nat(count),output_blocks} : GroupBlocks}
  Traversal.indexed_fold(~((Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>),~GroupBlocks,~block,
    U32.to_nat((((taps + 7) / 8) * output_blocks : U32)),0,context,
    ((input,gradient),array(~F32,(((taps * o + 7) / 8) * 8 : U32),0.0)))

# Array sums use the same eight-lane gather, addition and scatter as tensors.
def sum_result(+position: U32,left: Array<F32> & Tile.Tile8,right: Array<F32> & Tile.Tile8) -> Array<F32> & Array<F32>:
  (left,a) = left
  (right,b) = right
  (Lanes.scatter(left,position,TensorLanes.combine_tile(~F32.add,a,b)),right)

def sum_step(position: U32,context: Unit,owners: Array<F32> & Array<F32>) -> Array<F32> & Array<F32>:
  (left,right) = owners
  +position = (position * 8 : U32)
  sum_result(position,Lanes.gather(~Unit,~TensorLanes.identity,left,position,Unit{}),Lanes.gather(~Unit,~TensorLanes.identity,right,position,Unit{}))

def sum(+count: U32,left: Array<F32>,right: Array<F32>) -> Array<F32>:
  Pair.fst(Array<F32>,Array<F32>,Traversal.indexed_fold(~(Array<F32> & Array<F32>),~Unit,~sum_step,U32.to_nat(((count + 7) / 8 : U32)),0,Unit{},(left,right)))

type GroupContext is Data:
  GroupContext{geometry: Geometry,image: U32}

def group_read(owners: Storage.Buffer<F32> & Storage.Buffer<F32>,context: GroupContext,range: Grouped.Range) ->
  (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>:
  (input,gradient) = owners
  GroupContext{geometry,image} = context
  group(geometry,image,range,input,gradient)

def groups(ranges: List<&2,Grouped.Range>,geometry: Geometry,image: U32,input: Storage.Buffer<F32>,gradient: Storage.Buffer<F32>) ->
  (Storage.Buffer<F32> & Storage.Buffer<F32>) & List<&1,Array<F32>>:
  Traversal.collect_list_loop(~&1,~(Storage.Buffer<F32> & Storage.Buffer<F32>),~Array<F32>,~Grouped.Range,~GroupContext,~group_read,ranges,(input,gradient),GroupContext{geometry,image})

def reduced(fuel: Nat,count: U32,result: (Storage.Buffer<F32> & Storage.Buffer<F32>) & List<&1,Array<F32>>) ->
  (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>:
  (owners,values) = result
  (owners,Grouped.levels(~&1,~Array<F32>,~U32,~(count => array(~F32,(((count + 7) / 8) * 8 : U32),0.0)),~(count => left => right => sum(count,left,right)),fuel,count,values))

def image(+geometry: Geometry,index: U32,+ranges: List<&2,Grouped.Range>,input: Storage.Buffer<F32>,gradient: Storage.Buffer<F32>) ->
  (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>:
  Geometry{c,+o,h,w,k,s,p,rows,columns,+taps} = geometry
  reduced(List.length(&2,Grouped.Range,ranges),(o * taps : U32),groups(ranges,geometry,index,input,gradient))

# Every image builds its own group ranges: a list kept for the next image
# would be used twice and make the kernel's constructors shared.
type ImageContext is Data:
  ImageContext{geometry: Geometry,image: U32,rows: Nat,columns: Nat,group: Nat}

def image_collected(context: ImageContext,result: (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>) ->
  (Storage.Buffer<F32> & Storage.Buffer<F32>) & Traversal.Collected<&1,ImageContext,Array<F32>>:
  (owners,value) = result
  (owners,Traversal.Collected{context,value})

def image_read(owners: Storage.Buffer<F32> & Storage.Buffer<F32>,context: ImageContext) ->
  (Storage.Buffer<F32> & Storage.Buffer<F32>) & Traversal.Collected<&1,ImageContext,Array<F32>>:
  (input,gradient) = owners
  ImageContext{+geometry,+index,+rows,+columns,+group} = context
  image_collected(ImageContext{geometry,(index + 1 : U32),rows,columns,group},image(geometry,index,Grouped.ranges(rows,columns,group),input,gradient))

def images(remaining: Nat,index: U32,geometry: Geometry,rows: Nat,columns: Nat,group: Nat,input: Storage.Buffer<F32>,gradient: Storage.Buffer<F32>) ->
  (Storage.Buffer<F32> & Storage.Buffer<F32>) & List<&1,Array<F32>>:
  Traversal.collect_loop(~&1,~(Storage.Buffer<F32> & Storage.Buffer<F32>),~Array<F32>,~ImageContext,~image_read,remaining,(input,gradient),ImageContext{geometry,index,rows,columns,group})

# The values are [tap][output]; the result is [output][tap]. Output row of
# the values is read at stride rows and written in place at its row of the result.
type RowCopy is Type:
  RowCopy{values: Array<F32>,result: Storage.Buffer<F32>,address: U32,stride: U32,destination: U32}

def tap_copied(+address: U32,+stride: U32,+destination: U32,result: Storage.Buffer<F32>,read: Array<F32> & F32) -> RowCopy:
  (values,value) = read
  RowCopy{values,Storage.write(~F32,destination,value,result),(address + stride : U32),stride,(destination + 1 : U32)}

def tap_copy(state: RowCopy) -> RowCopy:
  RowCopy{values,result,+address,+stride,+destination} = state
  tap_copied(address,stride,destination,result,Array.get(F32,values,address))

type RowWrite is Type:
  RowWrite{values: Array<F32>,result: Storage.Buffer<F32>,row: U32,rows: U32,taps: U32,first: U32}

def row_copied(+row: U32,+rows: U32,+taps: U32,+first: U32,state: RowCopy) -> RowWrite:
  RowCopy{values,result,address,stride,destination} = state
  RowWrite{values,result,(row + 1 : U32),rows,taps,first}

def row_step(state: RowWrite) -> RowWrite:
  RowWrite{values,result,+row,+rows,+taps,+first} = state
  row_copied(row,rows,taps,first,Loop.run(~RowCopy,~tap_copy,U32.to_nat(taps),RowCopy{values,result,row,rows,((first + row) * taps : U32)}))

def rows_result(state: RowWrite) -> Storage.Buffer<F32>:
  RowWrite{values,result,row,rows,taps,first} = state
  result

# Rows [first, first + rows) of the result from values of rows outputs.
def rows_written(+rows: U32,+taps: U32,+first: U32,values: Array<F32>,result: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  rows_result(Loop.run(~RowWrite,~row_step,U32.to_nat(rows),RowWrite{values,result,0,rows,taps,first}))

def values_selected(+n: Nat,rows: Nat,columns: Nat,g: Nat,+geometry: Geometry,input: Storage.Buffer<F32>,gradient: Storage.Buffer<F32>) ->
  (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>:
  Geometry{gc,+go,gh,gw,gk,gs,gp,gr,gcols,+taps} = geometry
  reduced(n,(go * taps : U32),images(n,0,geometry,rows,columns,g,input,gradient))

# The values of every tap and output, [tap][output].
def weight_values(+work: Parameter.Work,input: Storage.Buffer<F32>,gradient: Storage.Buffer<F32>) ->
  (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>:
  Parameter.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  values_selected(n,rows,columns,g,geometry(work),input,gradient)

def written(+outputs: U32,+taps: U32,result: (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>) ->
  (Storage.Buffer<F32> & Storage.Buffer<F32>) & Storage.Buffer<F32>:
  (owners,values) = result
  (owners,rows_written(outputs,taps,0,values,Tensors.buffer(~F32,0.0,(outputs * taps : U32))))

def taps(work: Parameter.Work) -> U32:
  Parameter.Work{n,+c,o,h,w,+k,s,p,rows,columns,g} = work
  U32.from_nat(Nat.mul(c,Nat.mul(k,k)))

def outputs(work: Parameter.Work) -> U32:
  Parameter.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  U32.from_nat(o)

# The weight gradient of every output, [output][tap].
def weight(+work: Parameter.Work,input: Storage.Buffer<F32>,gradient: Storage.Buffer<F32>) ->
  (Storage.Buffer<F32> & Storage.Buffer<F32>) & Storage.Buffer<F32>:
  written(outputs(work),taps(work),weight_values(work,input,gradient))

# ---- Parts of the outputs ----

# The outputs split at whole blocks of eight; each part computes its own
# outputs with the serial kernel, so every value keeps its order of sums. A
# part owns a copy of the input and the output gradient of its outputs, laid
# out as [images, outputs, rows, columns]: the gradient seen as images of
# output rows, cut at the rows of the part. The copies are made before the
# parts run.
type Owners is Type:
  Owners{input: Storage.Buffer<F32>,gradient: Storage.Buffer<F32>}

def part_work(+work: Parameter.Work,+outputs: U32) -> Parameter.Work:
  Parameter.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  Parameter.Work{n,c,U32.to_nat(outputs),h,w,k,s,p,rows,columns,g}

def owned(+first: U32,+rows: U32,inputs: Storage.Buffer<F32> & Storage.Buffer<F32>,gradients: Storage.Buffer<F32> & Storage.Buffer<F32>) ->
  Owners & Partition.Tree<&1,Owners,Unit>:
  (input,copy) = inputs
  (gradient,channels) = gradients
  (Owners{input,gradient},Partition.Leaf{Owners{copy,channels},first,rows,Unit{}})

def own(work: Parameter.Work,first: U32,rows: U32,sources: Owners) -> Owners & Partition.Tree<&1,Owners,Unit>:
  Parameter.Work{n,c,o,h,w,k,s,p,height,width,g} = work
  Owners{input,gradient} = sources
  +first = first
  +rows = rows
  owned(first,rows,StorageClone.clone(~F32,input),
    Bands.whole_rows(~F32,Bands.Geometry{1,U32.from_nat(n),U32.from_nat(o),U32.from_nat(Nat.mul(height,width))},0.0,first,rows,gradient))

def part_values(result: (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>) -> Array<F32>:
  (owners,values) = result
  values

def part(work: Parameter.Work,first: U32,rows: U32,owners: Owners,resource: Unit) -> Partition.Tree<&1,Array<F32>,Unit>:
  Owners{input,gradient} = owners
  +rows = rows
  Partition.Leaf{part_values(weight_values(part_work(work,rows),input,gradient)),first,rows,Unit{}}

# The owners are made along the plan, so a part never meets another shape.
def unmatched(tree: Partition.Tree<&1,Owners,Unit>) -> Partition.Tree<&1,Array<F32>,Unit>:
  Partition.Leaf{Array.new(F32,0n,0.0),0,0,Unit{}}

def part_rows(taps: U32,first: U32,rows: U32,values: Array<F32>,result: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  rows_written(rows,taps,first,values,result)

def owners_of(sources: Owners) -> Storage.Buffer<F32> & Storage.Buffer<F32>:
  Owners{input,gradient} = sources
  (input,gradient)

def parts_run(+depth: Nat,+work: Parameter.Work,made: Owners & Partition.Tree<&1,Owners,Unit>) ->
  (Storage.Buffer<F32> & Storage.Buffer<F32>) & Storage.Buffer<F32>:
  (sources,tree) = made
  +outputs = outputs(work)
  +taps = taps(work)
  (owners_of(sources),Bands.fold(~Array<F32>,~Storage.Buffer<F32>,~U32,~part_rows,taps,
    Bands.each(~Owners,~Array<F32>,~Unit,~Parameter.Work,~Bands.unshared,~part,~unmatched,Bands.unit_plan(depth,8,0,outputs),True{},work,tree,Unit{}),
    Tensors.buffer(~F32,0.0,(outputs * taps : U32))))

# The outputs [0, outputs) halved depth times at whole blocks of eight. The
# rows of the parts are written in output order.
def parts(+depth: Nat,+work: Parameter.Work,input: Storage.Buffer<F32>,gradient: Storage.Buffer<F32>) ->
  (Storage.Buffer<F32> & Storage.Buffer<F32>) & Storage.Buffer<F32>:
  parts_run(depth,work,Bands.build(~Owners,~Owners,~Parameter.Work,~own,Bands.unit_plan(depth,8,0,outputs(work)),work,Owners{input,gradient}))

# ---- How many parts ----

# One output sums its positions in every image: the MACs, and the input and
# gradient lanes the kernel packs, counted from source as 13 and 6
# operations.
def output_ns(+work: Parameter.Work,config: Cost.Configuration) -> F32:
  Parameter.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  Cost.Configuration{mac_ns,byte_ns,operation_ns,grain,wake} = config
  +positions = U32.to_f32(U32.from_nat(Nat.mul(n,Nat.mul(rows,columns))))
  +taps = U32.to_f32(U32.from_nat(Nat.mul(c,Nat.mul(k,k))))
  (positions * taps * mac_ns + positions * (13.0 + 6.0) * operation_ns : F32)

# A split's time: the parts in waves of the workers, one copy of the input
# for every part and one of the gradient, and the two wake rounds.
def split_ns(+work: Parameter.Work,+workers: U32,+config: Cost.Configuration,size: BandSchedule.Size) -> F32:
  Parameter.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  Cost.Configuration{mac_ns,+byte_ns,operation_ns,grain,+wake} = config
  BandSchedule.Size{+leaves,minimum,+maximum} = size
  +input = U32.to_f32(U32.from_nat(Nat.mul(n,Nat.mul(c,Nat.mul(h,w)))))
  +gradient = U32.to_f32(U32.from_nat(Nat.mul(n,Nat.mul(o,Nat.mul(rows,columns)))))
  (Cost.region_ns(workers,leaves,(U32.to_f32(maximum) * output_ns(work,config) : F32)) + (U32.to_f32(leaves) * input + gradient) * 4.0 * byte_ns + 2.0 * wake : F32)

def depth_ns(depth: Nat,+work: Parameter.Work,+workers: U32,+config: Cost.Configuration) -> F32:
  match depth:
    case 0n: (U32.to_f32(outputs(work)) * output_ns(work,config) : F32)
    case 1n+ +rest: split_ns(work,workers,config,BandSchedule.size(Bands.unit_plan(1n+rest,8,0,outputs(work))))

def faster(+depth: Nat,+time: F32,best: Nat & F32) -> Nat & F32:
  (old_depth,+old_time) = best
  Bool.pick((Nat & F32),F32.is_lt(time,old_time),(depth,time),(old_depth,old_time))

def best_depth(remaining: Nat,+depth: Nat,+work: Parameter.Work,+workers: U32,+config: Cost.Configuration,best: Nat & F32) -> Nat & F32:
  match remaining:
    case 0n: best
    case 1n+rest: best_depth(rest,1n+depth,work,workers,config,faster(depth,depth_ns(depth,work,workers,config),best))

def log2(+value: U32) -> Nat:
  Nat.sub(Tensors.significant_bits(32n,value,0n,U32.is_zero(value)),1n)

# The fastest depth that leaves every part at least eight outputs and no more
# parts than workers.
def depth(+work: Parameter.Work,+workers: U32) -> Nat:
  +config = DefaultCost.configuration()
  +limit = Nat.min(log2(U32.max(workers,1)),log2(U32.max((outputs(work) / 8 : U32),1)))
  Pair.fst(Nat,F32,best_depth(limit,1n,work,workers,config,(0n,depth_ns(0n,work,workers,config))))

def routed(depth: Nat,+work: Parameter.Work,input: Storage.Buffer<F32>,gradient: Storage.Buffer<F32>) ->
  (Storage.Buffer<F32> & Storage.Buffer<F32>) & Storage.Buffer<F32>:
  match depth:
    case 0n: weight(work,input,gradient)
    case 1n+rest: parts(1n+rest,work,input,gradient)

# The weight gradient on the workers of the call.
def weight_parts(+work: Parameter.Work,+workers: U32,input: Storage.Buffer<F32>,gradient: Storage.Buffer<F32>) ->
  (Storage.Buffer<F32> & Storage.Buffer<F32>) & Storage.Buffer<F32>:
  routed(depth(work,workers),work,input,gradient)
