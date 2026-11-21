# Gradient tiles are the GEMM input, so multiplication keeps gradient first.
# GEMM rows are input channels at one tap, read as cells of eight channels at
# one pixel; GEMM lanes are cells of eight outputs at one position. The cells
# of the input and the gradient are those of a tensor: the groups of a pixel
# side by side, pixel after pixel. Each row is stored as one tile: the result
# is the cells [output block][tap], lane l of a cell the output 8 * block + l,
# +0 past the outputs.
import Base
import ./kernel_parameter_gradient.bend as Parameter
import ./kernel_gemm.bend as Gemm
import ./kernel_tile.bend as Tile
import ./storage_buffer.bend as Storage
import ./traversal_array.bend as Traversal
import ./traversal_loop.bend as Loop
import ./traversal_grouped.bend as Grouped
import ./tensor_move.bend as Move
import ./tensor.bend as Tensors
import ./tensor_pieces.bend as Pieces
import ./tensor_tasks.bend as Tasks
import ./traversal_partition.bend as Partition
import ./kernel_cost.bend as Cost
import ./default_kernel_cost.bend as DefaultCost
import ./traversal_blocks.bend as TraversalBlocks

type Geometry is Data:
  Geometry{channels: U32,outputs: U32,height: U32,width: U32,kernel: U32,stride: U32,padding: U32,rows: U32,columns: U32,taps: U32}

def geometry(work: Parameter.Work) -> Geometry:
  Parameter.Work{n,+c,o,h,w,+k,s,p,rows,columns,g} = work
  Geometry{U32.from_nat(c),U32.from_nat(o),U32.from_nat(h),U32.from_nat(w),U32.from_nat(k),U32.from_nat(s),U32.from_nat(p),U32.from_nat(rows),U32.from_nat(columns),U32.from_nat(Nat.mul(c,Nat.mul(k,k)))}

def array(~Element: Data,+count: U32,value: Element) -> Array<Element>:
  Array.new(Element,Storage.depth_for(count),value)

# The pieces of the input and of the gradient a kernel holds.
def Held() -> Type:
  Move.Holding<Tile.Tile8> & Move.Holding<Tile.Tile8>

# The first output position whose input coordinate offset + position * stride
# - padding is at least 0, and the end of those below size.
def axis_first(+offset: U32,+stride: U32,padding: U32) -> U32:
  U32.from_nat(Nat.div(Nat.add(Nat.sub(U32.to_nat(padding),U32.to_nat(offset)),Nat.sub(U32.to_nat(stride),1n)),U32.to_nat(stride)))

def axis_end(+offset: U32,+stride: U32,padding: U32,size: U32,positions: U32) -> U32:
  U32.from_nat(Nat.min(U32.to_nat(positions),Nat.div(Nat.add(Nat.sub(Nat.add(U32.to_nat(padding),U32.to_nat(size)),U32.to_nat(offset)),Nat.sub(U32.to_nat(stride),1n)),U32.to_nat(stride))))

# ---- Pixel cells ----

# The cells of one plane of band_columns columns from start: position start
# + t holds, for tile row R = first_row + t / band_columns and column X = t
# % band_columns, the cell at base of input row R * stride + row and column
# X * stride + column, a pixel groups cells after the one before it, read
# when R is in [row_first, row_end) and X in [column_first, column_end), zero
# otherwise.
type Plane is Data:
  Plane{base: U32,row: U32,column: U32,stride: U32,width: U32,groups: U32,row_first: U32,row_end: U32,column_first: U32,column_end: U32,first_row: U32,band_columns: U32,
    start: U32}

def plane_read(input: Storage.Buffer<Tile.Tile8>,plane: Plane,position: U32) -> Storage.Buffer<Tile.Tile8> & Tile.Tile8:
  Plane{+base,+row,+column,+stride,+width,+groups,+row_first,+row_end,+column_first,+column_end,+first_row,+band_columns,+start} = plane
  +within = (position - start : U32)
  +tile_row = (first_row + within / band_columns : U32)
  +tile_column = (within % band_columns : U32)
  Storage.read_masked(~Tile.Tile8,Tile.filled(0.0),
    U32.is_le(row_first,tile_row) && U32.is_lt(tile_row,row_end) && U32.is_le(column_first,tile_column) && U32.is_lt(tile_column,column_end),
    input,(base + ((tile_row * stride + row) * width + tile_column * stride + column) * groups : U32))

# The input cells a group reads, in phase planes: for every channel block and
# phase (row, column) below phases, a plane of band_rows tile rows from
# first_row by band_columns columns, tile row R and column X holding the cell
# of the block at input row R * stride + row - padding and column X * stride
# + column - padding, zero outside the image. Tap (a, b) of output (y, x) is
# the pixel of row y + a / phases and column x + b / phases in phase
# (a % phases, b % phases), phases = min(stride, kernel).
type Pixels is Data:
  Pixels{geometry: Geometry,base: U32,phases: U32,first_row: U32,band_rows: U32,band_columns: U32}

def plane_of(+plane: U32,+pixels: Pixels) -> Plane:
  Pixels{+geometry,+base,+phases,+first_row,+band_rows,+band_columns} = pixels
  Geometry{+c,o,+h,+w,k,+s,+p,rows,columns,taps} = geometry
  +row = (plane / phases % phases : U32)
  +column = (plane % phases : U32)
  Plane{(base + plane / phases / phases : U32),(row - p : U32),(column - p : U32),s,w,((c + 7) / 8 : U32),
    axis_first(row,s,p),axis_end(row,s,p,h,(first_row + band_rows : U32)),axis_first(column,s,p),axis_end(column,s,p,w,band_columns),
    first_row,band_columns,(plane * band_rows * band_columns : U32)}

def pixel_plane(plane: U32,pixels: Pixels,owners: Storage.Buffer<Tile.Tile8> & Array<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8> & Array<Tile.Tile8>:
  Pixels{+geometry,+base,+phases,+first_row,+band_rows,+band_columns} = pixels
  (input,tiles) = owners
  +plane = plane
  Traversal.finish(~Storage.Buffer<Tile.Tile8>,~Tile.Tile8,~Plane,
    Traversal.run(~Storage.Buffer<Tile.Tile8>,~Tile.Tile8,~Plane,~plane_read,~TraversalBlocks.in_place(~Plane),U32.to_nat((band_rows * band_columns : U32)),
      Traversal.State{input,tiles,(plane * band_rows * band_columns : U32),plane_of(plane,Pixels{geometry,base,phases,first_row,band_rows,band_columns})}))

# ---- Gradient cells ----

# Tile b * count + t is the gradient cell of output block b at position t
# from the position of base, groups cells a position.
type Positions is Data:
  Positions{base: U32,groups: U32,count: U32}

def gradient_read(gradient: Storage.Buffer<Tile.Tile8>,context: Positions,position: U32) -> Storage.Buffer<Tile.Tile8> & Tile.Tile8:
  Positions{+base,+groups,+count} = context
  +position = position
  Storage.read(~Tile.Tile8,gradient,(base + position % count * groups + position / count : U32))

# ---- Jobs ----

# Job (block, tap, output block) of a group: rows are the eight input
# channels of the block at one tap, lanes the eight outputs of the output
# block, steps the count positions from column of the first row. Its row of
# channel is the tile values [output block * taps + channel * square + tap].
type Plan is Data:
  Plan{blocks: U32,square: U32,kernel: U32,phases: U32,band_rows: U32,band_columns: U32,columns: U32,column: U32,count: U32,channels: U32,outputs: U32}

def Cursor() -> Data:
  Gemm.Both<U32,Gemm.Entries>

# Gradient tiles consecutive, pixel tiles along the rows of the band.
def advance(cursor: Cursor()) -> Gemm.Step<Cursor()>:
  Gemm.both(~U32,~Gemm.Entries,~Gemm.direct,~Gemm.entry,cursor)

def job(plan: Plan,index: U32) -> Gemm.Job<Cursor()>:
  Plan{+blocks,+square,+k,+phases,+band_rows,+band_columns,+columns,+column,+count,+c,+o} = plan
  +index = index
  +output = (index % blocks * 8 : U32)
  +tap = (index / blocks % square : U32)
  +row = (index / blocks / square * 8 : U32)
  +a = (tap / k : U32)
  +b = (tap % k : U32)
  Gemm.Job{row,U32.to_nat(count),
    Gemm.Both{(output / 8 * count : U32),
      Gemm.Entries{((((row / 8 * phases + a % phases) * phases + b % phases) * band_rows + a / phases) * band_columns + column + b / phases : U32),column,0,columns,band_rows,
        (band_columns - columns + 1 : U32),0}},
    Gemm.Store{((output / 8 * c + row) * square + tap : U32),square,1,U32.min(8,(c - row : U32)),U32.min(8,(o - output : U32))}}

def product(+plan: Plan,+jobs: U32,pixels: Storage.Buffer<Tile.Tile8> & Array<Tile.Tile8>,gradients: Storage.Buffer<Tile.Tile8> & Array<Tile.Tile8>,values: Array<Tile.Tile8>) ->
  (Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & Array<Tile.Tile8>:
  (input,pixel_tiles) = pixels
  (gradient,gradient_tiles) = gradients
  ((input,gradient),Gemm.output_of(~Tile.Tile8,~Array<F32>,Gemm.run(~Plan,~Cursor(),~Tile.Tile8,~job,~advance,~Gemm.lane_times,~Gemm.write_tiles(~Gemm.unbiased,~Gemm.kept),U32.to_nat(jobs),plan,
    Gemm.Operands{pixel_tiles,gradient_tiles,array(~F32,1,0.0),values})))

# The pixel planes of every output row and the gradient tiles of every
# output position fit the capacity.
def fits(+channels: Nat,+outputs: Nat,+kernel: Nat,+stride: Nat,+rows: Nat,+columns: Nat,+capacity: U32) -> Bool:
  +phases = Nat.min(stride,kernel)
  +reach = Nat.div(Nat.sub(kernel,1n),phases)
  Nat.is_le(Nat.mul(Nat.mul(Nat.mul(Nat.mul(Nat.div(Nat.add(channels,7n),8n),phases),phases),Nat.add(rows,reach)),Nat.add(columns,reach)),U32.to_nat(capacity)) &&
    Nat.is_le(Nat.mul(Nat.mul(Nat.div(Nat.add(outputs,7n),8n),rows),columns),U32.to_nat(capacity))

# The pixel cells and the gradient cells of every output block are copied
# once for the group, from the image cells at x_base of the input and d_base
# of the gradient; every cell is then its ordered fold from +0 over the
# positions.
def group(+geometry: Geometry,+x_base: U32,+d_base: U32,range: Grouped.Range,input: Storage.Buffer<Tile.Tile8>,gradient: Storage.Buffer<Tile.Tile8>) ->
  (Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & Array<Tile.Tile8>:
  Geometry{+c,+o,h,w,+k,+s,p,+rows,+columns,+taps} = geometry
  Grouped.Range{+first,+count} = range
  +positions = U32.from_nat(count)
  +start = U32.from_nat(first)
  +first_row = (start / columns : U32)
  +phases = U32.min(s,k)
  +reach = ((k - 1) / phases : U32)
  +band_rows = ((start + positions - 1) / columns - first_row + 1 + reach : U32)
  +band_columns = (columns + reach : U32)
  +blocks = ((c + 7) / 8 : U32)
  +output_blocks = ((o + 7) / 8 : U32)
  +planes = (blocks * phases * phases : U32)
  product(Plan{output_blocks,(k * k : U32),k,phases,band_rows,band_columns,columns,(start % columns : U32),positions,c,o},(blocks * k * k * output_blocks : U32),
    Traversal.indexed_fold(~(Storage.Buffer<Tile.Tile8> & Array<Tile.Tile8>),~Pixels,~pixel_plane,U32.to_nat(planes),0,Pixels{geometry,x_base,phases,first_row,band_rows,band_columns},
      (input,array(~Tile.Tile8,(planes * band_rows * band_columns : U32),Tile.filled(0.0)))),
    Traversal.finish(~Storage.Buffer<Tile.Tile8>,~Tile.Tile8,~Positions,
      Traversal.run(~Storage.Buffer<Tile.Tile8>,~Tile.Tile8,~Positions,~gradient_read,~TraversalBlocks.in_place(~Positions),U32.to_nat((output_blocks * positions : U32)),
        Traversal.State{gradient,array(~Tile.Tile8,(output_blocks * positions : U32),Tile.filled(0.0)),0,
          Positions{(d_base + start * output_blocks : U32),output_blocks,positions}})),
    array(~Tile.Tile8,(output_blocks * taps : U32),Tile.filled(0.0)))

# Array sums add the count tiles lane by lane.
def sum_result(+position: U32,left: Array<Tile.Tile8> & Tile.Tile8,right: Array<Tile.Tile8> & Tile.Tile8) -> Array<Tile.Tile8> & Array<Tile.Tile8>:
  (left,a) = left
  (right,b) = right
  (Array.set(Tile.Tile8,left,position,Tile.plus(a,b)),right)

def sum_step(position: U32,context: Unit,owners: Array<Tile.Tile8> & Array<Tile.Tile8>) -> Array<Tile.Tile8> & Array<Tile.Tile8>:
  (left,right) = owners
  +position = position
  sum_result(position,Array.get(Tile.Tile8,left,position),Array.get(Tile.Tile8,right,position))

def sum(+count: U32,left: Array<Tile.Tile8>,right: Array<Tile.Tile8>) -> Array<Tile.Tile8>:
  Pair.fst(Array<Tile.Tile8>,Array<Tile.Tile8>,Traversal.indexed_fold(~(Array<Tile.Tile8> & Array<Tile.Tile8>),~Unit,~sum_step,U32.to_nat(count),0,Unit{},(left,right)))

type GroupContext is Data:
  GroupContext{geometry: Geometry,x_base: U32,d_base: U32}

def group_read(owners: Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>,context: GroupContext,range: Grouped.Range) ->
  (Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & Array<Tile.Tile8>:
  (input,gradient) = owners
  GroupContext{geometry,x_base,d_base} = context
  group(geometry,x_base,d_base,range,input,gradient)

def groups(ranges: List<&2,Grouped.Range>,geometry: Geometry,x_base: U32,d_base: U32,input: Storage.Buffer<Tile.Tile8>,gradient: Storage.Buffer<Tile.Tile8>) ->
  (Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & List<&1,Array<Tile.Tile8>>:
  Traversal.collect_list_loop(~&1,~(Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>),~Array<Tile.Tile8>,~Grouped.Range,~GroupContext,~group_read,ranges,(input,gradient),GroupContext{geometry,x_base,d_base})

def reduced(fuel: Nat,count: U32,result: Held() & List<&1,Array<Tile.Tile8>>) -> Held() & Array<Tile.Tile8>:
  (owners,values) = result
  (owners,Grouped.levels(~&1,~Array<Tile.Tile8>,~U32,~(count => array(~Tile.Tile8,count,Tile.filled(0.0))),~(count => left => right => sum(count,left,right)),fuel,count,values))

# The cells of a result: the output blocks at every tap.
def cells(+geometry: Geometry) -> U32:
  Geometry{c,+o,h,w,k,s,p,rows,columns,+taps} = geometry
  ((o + 7) / 8 * taps : U32)

def image_returned(x_lent: Move.Lent<Tile.Tile8>,d_lent: Move.Lent<Tile.Tile8>,result: (Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & List<&1,Array<Tile.Tile8>>) ->
  Held() & List<&1,Array<Tile.Tile8>>:
  ((x,d),values) = result
  ((Move.returned_holding(~Tile.Tile8,x_lent,x),Move.returned_holding(~Tile.Tile8,d_lent,d)),values)

def image_banded(+geometry: Geometry,+ranges: List<&2,Grouped.Range>,x: Move.Band<Tile.Tile8>,d: Move.Band<Tile.Tile8>) -> Held() & Array<Tile.Tile8>:
  Move.Band{x_lent,x_band,+x_base} = x
  Move.Band{d_lent,d_band,+d_base} = d
  reduced(List.length(&2,Grouped.Range,ranges),cells(geometry),image_returned(x_lent,d_lent,groups(ranges,geometry,x_base,d_base,x_band,d_band)))

# An image reads its input and gradient cells as two bands of the held pieces.
def image(+geometry: Geometry,index: U32,+ranges: List<&2,Grouped.Range>,input: Move.Holding<Tile.Tile8>,gradient: Move.Holding<Tile.Tile8>) -> Held() & Array<Tile.Tile8>:
  Geometry{+c,+o,+h,+w,k,s,p,+rows,+columns,taps} = geometry
  +index = index
  +x_cells = ((c + 7) / 8 * h * w : U32)
  +d_cells = ((o + 7) / 8 * rows * columns : U32)
  image_banded(geometry,ranges,Move.band(~Tile.Tile8,input,(index * x_cells : U32),x_cells,Tile.filled(0.0),Move.spare(~Tile.Tile8,Tile.filled(0.0))),
    Move.band(~Tile.Tile8,gradient,(index * d_cells : U32),d_cells,Tile.filled(0.0),Move.spare(~Tile.Tile8,Tile.filled(0.0))))

# Every image builds its own group ranges: a list kept for the next image
# would be used twice and make the kernel's constructors shared.
type ImageContext is Data:
  ImageContext{geometry: Geometry,image: U32,rows: Nat,columns: Nat,group: Nat}

def image_collected(context: ImageContext,result: Held() & Array<Tile.Tile8>) -> Held() & Traversal.Collected<&1,ImageContext,Array<Tile.Tile8>>:
  (owners,value) = result
  (owners,Traversal.Collected{context,value})

def image_read(owners: Held(),context: ImageContext) -> Held() & Traversal.Collected<&1,ImageContext,Array<Tile.Tile8>>:
  (input,gradient) = owners
  ImageContext{+geometry,+index,+rows,+columns,+group} = context
  image_collected(ImageContext{geometry,(index + 1 : U32),rows,columns,group},image(geometry,index,Grouped.ranges(rows,columns,group),input,gradient))

def images(remaining: Nat,index: U32,geometry: Geometry,rows: Nat,columns: Nat,group: Nat,input: Move.Holding<Tile.Tile8>,gradient: Move.Holding<Tile.Tile8>) ->
  Held() & List<&1,Array<Tile.Tile8>>:
  Traversal.collect_loop(~&1,~Held(),~Array<Tile.Tile8>,~ImageContext,~image_read,remaining,(input,gradient),ImageContext{geometry,index,rows,columns,group})

# Tiles [first, first + count) of the result from the count tiles of values.
type TileCopy is Type:
  TileCopy{values: Array<Tile.Tile8>,result: Storage.Buffer<Tile.Tile8>,index: U32,first: U32}

def tile_copied(+index: U32,+first: U32,result: Storage.Buffer<Tile.Tile8>,read: Array<Tile.Tile8> & Tile.Tile8) -> TileCopy:
  (values,value) = read
  TileCopy{values,Storage.write(~Tile.Tile8,(first + index : U32),value,result),(index + 1 : U32),first}

def tile_copy(state: TileCopy) -> TileCopy:
  TileCopy{values,result,+index,+first} = state
  tile_copied(index,first,result,Array.get(Tile.Tile8,values,index))

def tiles_result(state: TileCopy) -> Storage.Buffer<Tile.Tile8>:
  TileCopy{values,result,index,first} = state
  result

def tiles_written(count: Nat,+first: U32,values: Array<Tile.Tile8>,result: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  tiles_result(Loop.run(~TileCopy,~tile_copy,count,TileCopy{values,result,0,first}))

def values_selected(+n: Nat,rows: Nat,columns: Nat,g: Nat,+geometry: Geometry,input: Move.Holding<Tile.Tile8>,gradient: Move.Holding<Tile.Tile8>) ->
  Held() & Array<Tile.Tile8>:
  reduced(n,cells(geometry),images(n,0,geometry,rows,columns,g,input,gradient))

# The cells of every output block and tap, [output block][tap].
def weight_values(+work: Parameter.Work,input: Move.Holding<Tile.Tile8>,gradient: Move.Holding<Tile.Tile8>) -> Held() & Array<Tile.Tile8>:
  Parameter.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  values_selected(n,rows,columns,g,geometry(work),input,gradient)

def written(+count: U32,result: Held() & Array<Tile.Tile8>) -> Held() & Storage.Buffer<Tile.Tile8>:
  (owners,values) = result
  (owners,tiles_written(U32.to_nat(count),0,values,Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count)))

def taps(work: Parameter.Work) -> U32:
  Parameter.Work{n,+c,o,h,w,+k,s,p,rows,columns,g} = work
  U32.from_nat(Nat.mul(c,Nat.mul(k,k)))

def outputs(work: Parameter.Work) -> U32:
  Parameter.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  U32.from_nat(o)

def blocks(+work: Parameter.Work) -> U32:
  ((outputs(work) + 7) / 8 : U32)

# The weight gradient as cells [output block][tap].
def weight(+work: Parameter.Work,input: Move.Holding<Tile.Tile8>,gradient: Move.Holding<Tile.Tile8>) -> Held() & Storage.Buffer<Tile.Tile8>:
  written(cells(geometry(work)),weight_values(work,input,gradient))

# ---- Parts of the outputs ----

# The output blocks split into parts; each part computes the cells of its
# blocks with the serial kernel, so every value keeps its order of sums, and
# the cells of a part follow those of the parts before it. A part holds a copy
# of the input cells and the gradient cells of its blocks, laid out as
# [images, rows, columns, output blocks]: the gradient cells of the blocks of
# the part at every position. The copies are made before the parts run.
type Owners is Type:
  Owners{input: Move.Holding<Tile.Tile8>,gradient: Move.Holding<Tile.Tile8>}

def part_work(+work: Parameter.Work,+outputs: U32) -> Parameter.Work:
  Parameter.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  Parameter.Work{n,c,U32.to_nat(outputs),h,w,k,s,p,rows,columns,g}

# One buffer held as the only piece.
def alone(+count: U32,buffer: Storage.Buffer<Tile.Tile8>) -> Move.Holding<Tile.Tile8>:
  Move.held(~Tile.Tile8,Partition.Leaf{buffer,0,count,Unit{}})

def owned(+first: U32,+rows: U32,+input_cells: U32,+gradient_cells: U32,inputs: Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>,
  gradients: Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) -> Owners & Partition.Tree<&1,Owners,Unit>:
  (input,copy) = inputs
  (gradient,channels) = gradients
  (Owners{input,gradient},Partition.Leaf{Owners{alone(input_cells,copy),alone(gradient_cells,channels)},first,rows,Unit{}})

# The cells of blocks [first, first + count) at every position of a gradient
# of groups cells a position.
def blocks_of(+positions: U32,+groups: U32,+first: U32,+count: U32,gradient: Move.Holding<Tile.Tile8>) -> Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  Move.part(~Tile.Tile8,Move.Part{[Move.Axis{positions,groups,False{},count},Move.Axis{count,1,False{},1}],first,0},gradient,
    Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),(positions * count : U32)))

def own(work: Parameter.Work,first: U32,rows: U32,sources: Owners) -> Owners & Partition.Tree<&1,Owners,Unit>:
  Parameter.Work{+n,+c,+o,+h,+w,k,s,p,+height,+width,g} = work
  Owners{input,gradient} = sources
  +first = first
  +rows = rows
  +positions = U32.from_nat(Nat.mul(n,Nat.mul(height,width)))
  +cells = U32.from_nat(Nat.mul(n,Nat.mul(Nat.div(Nat.add(c,7n),8n),Nat.mul(h,w))))
  owned(first,rows,cells,(positions * rows : U32),Move.copied(~Tile.Tile8,input,0,cells,Tile.filled(0.0)),
    blocks_of(positions,U32.from_nat(Nat.div(Nat.add(o,7n),8n)),first,rows,gradient))

def part_values(result: Held() & Array<Tile.Tile8>) -> Array<Tile.Tile8>:
  (owners,values) = result
  values

# The outputs of the blocks [first, first + rows).
def part_outputs(+work: Parameter.Work,+first: U32,+rows: U32) -> U32:
  U32.min((rows * 8 : U32),(outputs(work) - first * 8 : U32))

def part(work: Parameter.Work,first: U32,rows: U32,owners: Owners,resource: Unit) -> Partition.Tree<&1,Array<Tile.Tile8>,Unit>:
  Owners{input,gradient} = owners
  +work = work
  +first = first
  +rows = rows
  Partition.Leaf{part_values(weight_values(part_work(work,part_outputs(work,first,rows)),input,gradient)),first,rows,Unit{}}

# The owners are made along the plan, so a part never meets another shape.
def unmatched(tree: Partition.Tree<&1,Owners,Unit>) -> Partition.Tree<&1,Array<Tile.Tile8>,Unit>:
  Partition.Leaf{Array.new(Tile.Tile8,0n,Tile.filled(0.0)),0,0,Unit{}}

# The blocks [first, first + rows) own the cells from first * taps.
def part_cells(taps: U32,first: U32,rows: U32,values: Array<Tile.Tile8>,result: Storage.Buffer<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  +taps = taps
  tiles_written(Nat.mul(U32.to_nat(rows),U32.to_nat(taps)),(first * taps : U32),values,result)

def owners_of(sources: Owners) -> Held():
  Owners{input,gradient} = sources
  (input,gradient)

def parts_run(+depth: Nat,+work: Parameter.Work,made: Owners & Partition.Tree<&1,Owners,Unit>) -> Held() & Storage.Buffer<Tile.Tile8>:
  (sources,tree) = made
  +count = cells(geometry(work))
  (owners_of(sources),Pieces.fold(~Array<Tile.Tile8>,~Storage.Buffer<Tile.Tile8>,~U32,~part_cells,taps(work),
    Pieces.each(~Owners,~Array<Tile.Tile8>,~Unit,~Parameter.Work,~Pieces.unshared,~part,~unmatched,Pieces.unit_ranges(depth,1,0,blocks(work)),True{},work,tree,Unit{}),
    Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count)))

# The output blocks [0, blocks) halved depth times. The cells of the parts are
# written in output order.
def parts(+depth: Nat,+work: Parameter.Work,input: Move.Holding<Tile.Tile8>,gradient: Move.Holding<Tile.Tile8>) -> Held() & Storage.Buffer<Tile.Tile8>:
  parts_run(depth,work,Pieces.build(~Owners,~Owners,~Parameter.Work,~own,Pieces.unit_ranges(depth,1,0,blocks(work)),work,Owners{input,gradient}))

# ---- Parts of the images ----

# Each part holds a copy of its consecutive images. Its per-image results are
# concatenated before the original global reduction, preserving its sum tree.
def images_of(+size: U32,+first: U32,+count: U32,cells: Move.Holding<Tile.Tile8>) -> Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  Move.copied(~Tile.Tile8,cells,(size * first : U32),(size * count : U32),Tile.filled(0.0))

def input_image(work: Parameter.Work) -> U32:
  Parameter.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  U32.from_nat(Nat.mul(Nat.div(Nat.add(c,7n),8n),Nat.mul(h,w)))

def gradient_image(work: Parameter.Work) -> U32:
  Parameter.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  U32.from_nat(Nat.mul(Nat.div(Nat.add(o,7n),8n),Nat.mul(rows,columns)))

def batch_own(work: Parameter.Work,first: U32,rows: U32,sources: Owners) -> Owners & Partition.Tree<&1,Owners,Unit>:
  Owners{input,gradient} = sources
  +work = work
  +first = first
  +rows = rows
  owned(first,rows,(input_image(work) * rows : U32),(gradient_image(work) * rows : U32),images_of(input_image(work),first,rows,input),
    images_of(gradient_image(work),first,rows,gradient))

type BatchContext is Data:
  BatchContext{geometry: Geometry,rows: Nat,columns: Nat,group: Nat}

def batch_context(+work: Parameter.Work) -> BatchContext:
  Parameter.Work{n,c,o,h,w,k,s,p,+rows,+columns,+g} = work
  BatchContext{geometry(work),rows,columns,g}

def batch_values(result: Held() & List<&1,Array<Tile.Tile8>>) -> List<&1,Array<Tile.Tile8>>:
  (owners,values) = result
  values

def batch_part(context: BatchContext,first: U32,rows: U32,owners: Owners,resource: Unit) -> Partition.Tree<&1,List<&1,Array<Tile.Tile8>>,Unit>:
  BatchContext{geometry,height,width,group} = context
  Owners{input,gradient} = owners
  +rows = rows
  Partition.Leaf{batch_values(images(U32.to_nat(rows),0,geometry,height,width,group,input,gradient)),first,rows,Unit{}}

def batch_unmatched(tree: Partition.Tree<&1,Owners,Unit>) -> Partition.Tree<&1,List<&1,Array<Tile.Tile8>>,Unit>:
  Partition.Leaf{[],0,0,Unit{}}

def batch_collected(context: Unit,first: U32,rows: U32,values: List<&1,Array<Tile.Tile8>>) -> Partition.Tree<&1,Unit,Unit> & List<&1,Array<Tile.Tile8>>:
  (Partition.Leaf{Unit{},first,rows,Unit{}},values)

def batch_arrays(tree: Partition.Tree<&1,List<&1,Array<Tile.Tile8>>,Unit>) -> List<&1,Array<Tile.Tile8>>:
  Pair.snd(Partition.Tree<&1,Unit,Unit>,List<&1,Array<Tile.Tile8>>,
    Pieces.reduce(~List<&1,Array<Tile.Tile8>>,~Unit,~List<&1,Array<Tile.Tile8>>,~Unit,~batch_collected,~List.append(&1,Array<Tile.Tile8>),Unit{},tree))

def batch_plan(depth: Nat,work: Parameter.Work) -> Pieces.Ranges:
  Parameter.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  Pieces.unit_ranges(depth,1,0,U32.from_nat(n))

def batch_run(+depth: Nat,+work: Parameter.Work,made: Owners & Partition.Tree<&1,Owners,Unit>) -> Held() & Storage.Buffer<Tile.Tile8>:
  Parameter.Work{+n,c,o,h,w,k,s,p,rows,columns,g} = work
  (sources,tree) = made
  +count = cells(geometry(work))
  written(count,reduced(n,count,
    (owners_of(sources),batch_arrays(Pieces.each(~Owners,~List<&1,Array<Tile.Tile8>>,~Unit,~BatchContext,~Pieces.unshared,~batch_part,~batch_unmatched,
      batch_plan(depth,work),True{},batch_context(work),tree,Unit{})))))

def batch_parts(+depth: Nat,+work: Parameter.Work,input: Move.Holding<Tile.Tile8>,gradient: Move.Holding<Tile.Tile8>) -> Held() & Storage.Buffer<Tile.Tile8>:
  batch_run(depth,work,Pieces.build(~Owners,~Owners,~Parameter.Work,~batch_own,batch_plan(depth,work),work,Owners{input,gradient}))

# ---- How many parts ----

# The outputs of a part sum their positions in every image: the MACs, eight
# a row, the gradient cell of each output block and the pixel cells of every
# channel block and phase, which the part makes once for all its outputs,
# counted from source as 8 and 20 steps.
def part_ns(+work: Parameter.Work,+outputs: U32,config: Cost.Configuration) -> F32:
  Parameter.Work{n,+c,o,h,w,+k,+s,p,rows,columns,g} = work
  Cost.Configuration{+row_ns,byte_ns,+step_ns,move,grain,wake,dispatch} = config
  +positions = U32.to_f32(U32.from_nat(Nat.mul(n,Nat.mul(rows,columns))))
  +taps = U32.to_f32(U32.from_nat(Nat.mul(c,Nat.mul(k,k))))
  +phases = Nat.min(s,k)
  +cells = U32.to_f32(U32.from_nat(Nat.mul(Nat.div(Nat.add(c,7n),8n),Nat.mul(phases,phases))))
  +blocks = U32.to_f32(((outputs + 7) / 8 : U32))
  (positions * (U32.to_f32(outputs) * taps / 8.0 * row_ns + blocks * 8.0 * step_ns + cells * 20.0 * step_ns) : F32)

# A split's time: the parts in waves of the workers, one copy of the input
# cells for every part and one of the gradient cells, and the two wake rounds.
def split_ns(+work: Parameter.Work,+workers: U32,+config: Cost.Configuration,size: Tasks.Size) -> F32:
  Parameter.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  Cost.Configuration{row_ns,+byte_ns,step_ns,move,grain,+wake,dispatch} = config
  Tasks.Size{+leaves,+maximum} = size
  +input = U32.to_f32(U32.from_nat(Nat.mul(n,Nat.mul(Nat.div(Nat.add(c,7n),8n),Nat.mul(h,w)))))
  +gradient = U32.to_f32(U32.from_nat(Nat.mul(n,Nat.mul(Nat.div(Nat.add(o,7n),8n),Nat.mul(rows,columns)))))
  (Cost.region_ns(workers,leaves,part_ns(work,part_outputs(work,0,maximum),config)) + (U32.to_f32(leaves) * input + gradient) * 32.0 * byte_ns + 2.0 * wake : F32)

def depth_ns(depth: Nat,+work: Parameter.Work,+workers: U32,+config: Cost.Configuration) -> F32:
  match depth:
    case 0n: part_ns(work,outputs(work),config)
    case 1n+ +rest: split_ns(work,workers,config,Tasks.size(Pieces.unit_ranges(1n+rest,1,0,blocks(work))))

def faster(+depth: Nat,+time: F32,best: Nat & F32) -> Nat & F32:
  (old_depth,+old_time) = best
  Bool.pick((Nat & F32),F32.is_lt(time,old_time),(depth,time),(old_depth,old_time))

def best_depth(remaining: Nat,+depth: Nat,+work: Parameter.Work,+workers: U32,+config: Cost.Configuration,best: Nat & F32) -> Nat & F32:
  match remaining:
    case 0n: best
    case 1n+rest: best_depth(rest,1n+depth,work,workers,config,faster(depth,depth_ns(depth,work,workers,config),best))

def log2(+value: U32) -> Nat:
  Nat.sub(Tensors.significant_bits(32n,value,0n,U32.is_zero(value)),1n)

# The fastest depth that leaves every part at least one output block and no more
# parts than workers.
def depth(+work: Parameter.Work,+workers: U32) -> Nat:
  +config = DefaultCost.configuration()
  +limit = Nat.min(log2(U32.max(workers,1)),log2(blocks(work)))
  Pair.fst(Nat,F32,best_depth(limit,1n,work,workers,config,(0n,depth_ns(0n,work,workers,config))))

def routed(depth: Nat,+work: Parameter.Work,input: Move.Holding<Tile.Tile8>,gradient: Move.Holding<Tile.Tile8>) -> Held() & Storage.Buffer<Tile.Tile8>:
  match depth:
    case 0n: weight(work,input,gradient)
    case 1n+rest: parts(1n+rest,work,input,gradient)

# Output and image partitions use the same cost configuration. Image parts
# copy the input and gradient once in total, retaining all output channels.
type Route is Data:
  OutputParts{depth: Nat}
  BatchParts{depth: Nat}

type Choice is Data:
  Choice{route: Route,time: F32}

type BatchPlanning is Data:
  BatchPlanning{work: Parameter.Work,workers: U32,config: Cost.Configuration}

def batch_work(work: Parameter.Work,images: U32) -> Parameter.Work:
  Parameter.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  Parameter.Work{U32.to_nat(images),c,o,h,w,k,s,p,rows,columns,g}

def batch_split_ns(+work: Parameter.Work,+workers: U32,+config: Cost.Configuration,size: Tasks.Size) -> F32:
  Parameter.Work{+n,+c,+o,+h,+w,k,s,p,+rows,+columns,g} = work
  Cost.Configuration{row_ns,+byte_ns,step_ns,move,grain,+wake,dispatch} = config
  Tasks.Size{+leaves,+maximum} = size
  +input = U32.to_f32(U32.from_nat(Nat.mul(n,Nat.mul(Nat.div(Nat.add(c,7n),8n),Nat.mul(h,w)))))
  +gradient = U32.to_f32(U32.from_nat(Nat.mul(n,Nat.mul(Nat.div(Nat.add(o,7n),8n),Nat.mul(rows,columns)))))
  (Cost.region_ns(workers,leaves,part_ns(batch_work(work,maximum),outputs(work),config)) + (input + gradient) * 32.0 * byte_ns + 2.0 * wake : F32)

def batch_ns(+depth: Nat,+work: Parameter.Work,+workers: U32,+config: Cost.Configuration) -> F32:
  batch_split_ns(work,workers,config,Tasks.size(batch_plan(depth,work)))

def batch_choice(index: U32,context: BatchPlanning,best: Choice) -> Choice:
  BatchPlanning{+work,+workers,+config} = context
  Choice{old_route,+old_time} = best
  +depth = U32.to_nat(index)
  +time = batch_ns(depth,work,workers,config)
  Bool.pick(Choice,F32.is_lt(time,old_time),Choice{BatchParts{depth},time},Choice{old_route,old_time})

def chosen(choice: Choice) -> Route:
  Choice{route,time} = choice
  route

def route(+work: Parameter.Work,+workers: U32) -> Route:
  Parameter.Work{+n,c,o,h,w,k,s,p,rows,columns,g} = work
  +config = DefaultCost.configuration()
  +output_depth = depth(work,workers)
  +limit = Nat.min(log2(U32.max(workers,1)),log2(U32.from_nat(n)))
  chosen(Traversal.indexed_fold(~Choice,~BatchPlanning,~batch_choice,limit,1,BatchPlanning{work,workers,config},Choice{OutputParts{output_depth},depth_ns(output_depth,work,workers,config)}))

def route_run(route: Route,+work: Parameter.Work,input: Move.Holding<Tile.Tile8>,gradient: Move.Holding<Tile.Tile8>) -> Held() & Storage.Buffer<Tile.Tile8>:
  match route:
    case OutputParts{depth}: routed(depth,work,input,gradient)
    case BatchParts{depth}: batch_parts(depth,work,input,gradient)

# The weight gradient on the workers of the call, from the held pieces of the
# input and the gradient.
def weight_parts(+work: Parameter.Work,+workers: U32,input: Move.Holding<Tile.Tile8>,gradient: Move.Holding<Tile.Tile8>) -> Held() & Storage.Buffer<Tile.Tile8>:
  route_run(route(work,workers),work,input,gradient)
