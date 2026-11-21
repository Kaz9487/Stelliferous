# Batch normalization on cells [N,H,W,G] whose lanes are the channels, the
# groups innermost. The statistics read every cell once a sum in the order of
# Partials and hand the cells back. Every other pass walks the pieces in
# tasks, and in a piece one loop for each group reads its cells with the
# parameters of the group.
import Base
import ./kernel_batch_norm.bend as Kernel
import ./kernel_tile.bend as Tile
import ./math_sum.bend as Sum
import ./storage_buffer.bend as Storage
import ./tensor.bend as Tensors
import ./tensor_operations.bend as Operations
import ./tensor_partials.bend as Partials
import ./tensor_pieces.bend as Pieces
import ./tensor_tasks.bend as Tasks
import ./tensor_tiled_map.bend as TiledMap
import ./tensor_tiled_reduce.bend as TiledReduce
import ./traversal_array.bend as Traversal
import ./traversal_partition.bend as Partition
import ./proofs/storage_certificate.bend as Certificate
import ./proofs/storage_witness.bend as Witness
import ./proofs/traversal_certificate.bend as TraversalCertificate
import ./proofs/traversal_witness.bend as TraversalWitness
import ./proofs/kernel_batch_norm_certificate.bend as KernelCertificate

def Cells() -> Type:
  Storage.Buffer<Tile.Tile8>

def CellPair() -> Type:
  Pieces.Paired<Cells(),Cells()>

def Pairs() -> Type:
  Partition.Tree<&1,CellPair(),Unit>

# The parameters of every group, the number of groups and the padding that
# zeroes the lanes of the last group past the channels.
type Mapping<-Group: Data> is Data:
  Mapping{groups: List<&2,Group>,cells: U32,padding: TiledMap.Padding}

def zero_cells(+count: U32) -> Cells():
  Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count)

# The loop of group group in the piece of count cells from cell first.
def grouped(~Group: Data,+parameters: Group,+first: U32,+count: U32,+cells: U32,+group: U32) -> Kernel.Grouped<Group>:
  +start = ((group + cells - first % cells) % cells : U32)
  Kernel.Grouped{parameters,start,cells,Bool.pick(U32,U32.is_lt(start,count),((count - start + cells - 1) / cells : U32),0)}

def buffers_of(+input_length: Nat,+output_length: Nat,result: Array<Tile.Tile8> & Array<Tile.Tile8>,
  certificates: TraversalCertificate.Certificates<Array<Tile.Tile8>,Tile.Tile8,Certificate.family(~Tile.Tile8),result>) -> Cells() & Cells():
  (input,output) = result
  TraversalCertificate.Certificates{input_certificate,output_certificate} = certificates
  (Storage.Buffer{input,input_length,input_certificate},Storage.Buffer{output,output_length,output_certificate})

def mapped_buffers(~Group: Data,~read: Array<Tile.Tile8> -> Kernel.Grouped<Group> -> U32 -> Array<Tile.Tile8> & Tile.Tile8,
  ~read_preserves: @-input: Array<Tile.Tile8> -> @context: Kernel.Grouped<Group> -> @index: U32 -> Witness.ArrayWitness<Tile.Tile8,input> -> KernelCertificate.Read(input,read(input,context,index)),
  context: Kernel.Grouped<Group>,input: Cells(),output: Cells()) -> Cells() & Cells():
  Storage.Buffer{input,+input_length,input_certificate} = input
  Storage.Buffer{output,+output_length,output_certificate} = output
  +context = context
  buffers_of(input_length,output_length,Kernel.mapped(~Array<Tile.Tile8>,~Group,~read,context,input,output),
    KernelCertificate.array_certificates(~Group,~read,~read_preserves,context,input,output,input_certificate,output_certificate))

# Every group of a piece from group on, each a step on the two buffers.
def each_group(~Group: Data,~step: Kernel.Grouped<Group> -> Cells() -> Cells() -> Cells() & Cells(),groups: List<&2,Group>,+first: U32,+count: U32,+cells: U32,+group: U32,
  left: Cells(),right: Cells()) -> Cells() & Cells():
  match groups:
    case Nil{}: (left,right)
    case Con{+parameters,rest}:
      Storage.bind_pair(~Cells(),~Cells(),~(Cells() & Cells()),step(grouped(~Group,parameters,first,count,cells,group),left,right),left => right =>
        each_group(~Group,~step,rest,first,count,cells,(group + 1 : U32),left,right))

# The output of a piece with the lanes past the channels zero.
def padded_output(+padding: TiledMap.Padding,+first: U32,+count: U32,pair: Cells() & Cells()) -> Cells() & Cells():
  (input,output) = pair
  (input,TiledMap.padded(padding,first,count,output))

def padded_both(+padding: TiledMap.Padding,+first: U32,+count: U32,pair: Cells() & Cells()) -> Cells() & Cells():
  (left,right) = pair
  (TiledMap.padded(padding,first,count,left),TiledMap.padded(padding,first,count,right))

# A piece and the cells read from it.
def map_leaf(~Group: Data,~read: Array<Tile.Tile8> -> Kernel.Grouped<Group> -> U32 -> Array<Tile.Tile8> & Tile.Tile8,
  ~read_preserves: @-input: Array<Tile.Tile8> -> @context: Kernel.Grouped<Group> -> @index: U32 -> Witness.ArrayWitness<Tile.Tile8,input> -> KernelCertificate.Read(input,read(input,context,index)),
  context: Mapping<Group>,first: U32,count: U32,input: Cells(),resource: Unit) -> Partition.Tree<&1,Cells() & Cells(),Unit>:
  Mapping{+groups,+cells,+padding} = context
  +first = first
  +count = count
  Partition.Leaf{padded_output(padding,first,count,each_group(~Group,~mapped_buffers(~Group,~read,~read_preserves),groups,first,count,cells,0,input,zero_cells(count))),
    first,count,Unit{}}

def output_of(pair: Cells() & Cells()) -> Cells():
  (input,output) = pair
  output

def forward_output(+groups: List<&2,Kernel.Normal>,+first: U32,+count: U32,+cells: U32,normalized: Cells() & Cells()) -> Cells() & Cells():
  (input,xhat) = normalized
  (xhat,output_of(each_group(~Kernel.Normal,~mapped_buffers(~Kernel.Normal,~Kernel.read_output,~KernelCertificate.read_output),groups,first,count,cells,0,input,zero_cells(count))))

# A piece normalized and its output; the piece itself is dropped.
def forward_leaf(context: Mapping<Kernel.Normal>,first: U32,count: U32,input: Cells(),resource: Unit) -> Partition.Tree<&1,Cells() & Cells(),Unit>:
  Mapping{+groups,+cells,+padding} = context
  +first = first
  +count = count
  Partition.Leaf{padded_both(padding,first,count,forward_output(groups,first,count,cells,
    each_group(~Kernel.Normal,~mapped_buffers(~Kernel.Normal,~Kernel.read_normalized,~KernelCertificate.read_normalized),groups,first,count,cells,0,input,zero_cells(count)))),
    first,count,Unit{}}

# ---- Two pieces ----

def leaves_buffers(+left_length: Nat,+right_length: Nat,leaves: Kernel.Leaves(),certificates: KernelCertificate.LeavesCertificates<leaves>) -> Cells() & Cells():
  (left,right) = leaves
  KernelCertificate.LeavesCertificates{left_certificate,right_certificate} = certificates
  (Storage.Buffer{left,left_length,left_certificate},Storage.Buffer{right,right_length,right_certificate})

# dy of a group in place, xhat kept.
def back_buffers(context: Kernel.Grouped<Kernel.Back>,xhat: Cells(),dy: Cells()) -> Cells() & Cells():
  Storage.Buffer{xhat,+xhat_length,xhat_certificate} = xhat
  Storage.Buffer{dy,+dy_length,dy_certificate} = dy
  +context = context
  buffers_of(xhat_length,dy_length,Kernel.updated(context,xhat,dy),KernelCertificate.updated_certificates(context,xhat,dy,xhat_certificate,dy_certificate))

def back_pair(+groups: List<&2,Kernel.Back>,+cells: U32,+padding: TiledMap.Padding,+first: U32,+count: U32,pair: CellPair()) -> Pieces.Pieces(~Tile.Tile8):
  match pair:
    case Pieces.Both{dy,xhat}:
      Partition.Leaf{TiledMap.padded(padding,first,count,output_of(each_group(~Kernel.Back,~back_buffers,groups,first,count,cells,0,xhat,dy))),first,count,Unit{}}
    case Pieces.Only{left,own_first,own_count}: Partition.Leaf{left,own_first,own_count,Unit{}}

# dy of a piece in place, xhat dropped.
def back_leaf(context: Mapping<Kernel.Back>,first: U32,count: U32,pair: CellPair(),resource: Unit) -> Pieces.Pieces(~Tile.Tile8):
  Mapping{+groups,+cells,+padding} = context
  back_pair(groups,cells,padding,first,count,pair)

# ---- Passes ----

def tasked(~Block: Type,~Output: Type,~Context: Data,~leaf: Context -> U32 -> U32 -> Block -> Unit -> Partition.Tree<&1,Output,Unit>,decision: Tasks.Decision,context: Context,
  tree: Partition.Tree<&1,Block,Unit>) -> Partition.Tree<&1,Output,Unit>:
  Tasks.Decision{+parallel,+tasks} = decision
  Pieces.grouped(~Block,~Output,~Context,~leaf,tasks,parallel,context,tree)

# The leaf on every piece of cells of a shape, in tasks. rows and steps count
# one cell.
def pass(~Block: Type,~Output: Type,~Context: Data,~leaf: Context -> U32 -> U32 -> Block -> Unit -> Partition.Tree<&1,Output,Unit>,context: Context,+shape: List<&2,U32>,
  +workers: U32,+rows: F32,+steps: F32,tree: Partition.Tree<&1,Block,Unit>) -> Partition.Tree<&1,Output,Unit>:
  tasked(~Block,~Output,~Context,~leaf,Tasks.decide(Tensors.grid_of(shape),workers,rows,steps,Tasks.no_extra()),context,tree)

# xhat and the output of the input, which is dropped. A cell is read twice and
# written twice, with six operations a lane.
def forward(groups: List<&2,Kernel.Normal>,+cells: U32,+padding: TiledMap.Padding,+shape: List<&2,U32>,+workers: U32,pieces: Pieces.Pieces(~Tile.Tile8)) ->
  Pieces.Pieces(~Tile.Tile8) & Pieces.Pieces(~Tile.Tile8):
  Tensors.unzipped(~Tile.Tile8,pass(~Cells(),~(Cells() & Cells()),~Mapping<Kernel.Normal>,~forward_leaf,Mapping{groups,cells,padding},shape,workers,10.0,2.0,pieces))

def outputs(pair: Pieces.Pieces(~Tile.Tile8) & Pieces.Pieces(~Tile.Tile8)) -> Pieces.Pieces(~Tile.Tile8):
  (input,output) = pair
  output

# The output of the input, which is dropped. A cell is read once and written
# once, with four operations a lane.
def evaluated(groups: List<&2,Kernel.Normal>,+cells: U32,+padding: TiledMap.Padding,+shape: List<&2,U32>,+workers: U32,pieces: Pieces.Pieces(~Tile.Tile8)) -> Pieces.Pieces(~Tile.Tile8):
  outputs(Tensors.unzipped(~Tile.Tile8,pass(~Cells(),~(Cells() & Cells()),~Mapping<Kernel.Normal>,~map_leaf(~Kernel.Normal,~Kernel.read_output,~KernelCertificate.read_output),
    Mapping{groups,cells,padding},shape,workers,6.0,2.0,pieces)))

# dy in place with xhat dropped. A cell reads both and writes one, with five
# operations a lane.
def backward(groups: List<&2,Kernel.Back>,+cells: U32,+padding: TiledMap.Padding,+shape: List<&2,U32>,+workers: U32,pairs: Pairs()) -> Pieces.Pieces(~Tile.Tile8):
  pass(~CellPair(),~Cells(),~Mapping<Kernel.Back>,~back_leaf,Mapping{groups,cells,padding},shape,workers,8.0,2.0,pairs)

# ---- Statistics ----

def sum_finished(parts: Sum.TileParts) -> Tile.Tile8:
  Sum.tile_rounded_total(parts)

def sums_of(result: Pieces.Pieces(~Tile.Tile8) & List<&2,Sum.TileParts>) -> Pieces.Pieces(~Tile.Tile8) & List<&2,Tile.Tile8>:
  (pieces,partials) = result
  (pieces,Partials.finished(~Tile.Tile8,~Sum.TileParts,~sum_finished,partials))

# Every axis of [N,H,W,G] but the groups is summed. A row is the H x W plane
# of an image, and the rows of the images merge in the tree.
def summed_flags() -> List<&2,Bool>:
  [True{},True{},True{},False{}]

# The sums of every group in the order of sum_axes, and the cells. A cell
# adds to the slot of its place in the row.
def sums(+shape: List<&2,U32>,+workers: U32,pieces: Pieces.Pieces(~Tile.Tile8)) -> Pieces.Pieces(~Tile.Tile8) & List<&2,Tile.Tile8>:
  sums_of(Partials.read(~Cells(),~Tile.Tile8,~Unit,~Operations.TileSlots<Unit>,~Sum.TileParts,~Operations.slots_begun(~Unit),
    ~Operations.cells_slots(~Unit,~Traversal.spaced(~Tile.Tile8,~Unit),~TraversalWitness.spaced_read(~Tile.Tile8,~Unit)),~Operations.slots_closed(~Unit),
    ~Sum.tile_merge,~Pieces.vacant_buffer(~Tile.Tile8),[Unit{}],7.0,2.0,shape,summed_flags(),1n,2n,Tile.filled(0.0),workers,pieces))

# The sums of (x - mean) * (x - mean) of every group with its mean, in the
# order of sum_axes, and the cells.
def centered_sums(means: List<&2,Tile.Tile8>,+shape: List<&2,U32>,+workers: U32,pieces: Pieces.Pieces(~Tile.Tile8)) -> Pieces.Pieces(~Tile.Tile8) & List<&2,Tile.Tile8>:
  sums_of(Partials.read(~Cells(),~Tile.Tile8,~Tile.Tile8,~Operations.TileSlots<Tile.Tile8>,~Sum.TileParts,~Operations.slots_begun(~Tile.Tile8),
    ~Operations.cells_slots(~Tile.Tile8,~Kernel.read_centered,~KernelCertificate.read_centered),~Operations.slots_closed(~Tile.Tile8),~Sum.tile_merge,~Pieces.vacant_buffer(~Tile.Tile8),means,9.0,2.0,shape,summed_flags(),1n,2n,Tile.filled(0.0),workers,pieces))

def both_of(pair: Cells() & Cells()) -> CellPair():
  (left,right) = pair
  Pieces.Both{left,right}

def product_slotted(+left_length: Nat,+right_length: Nat,+slots_length: Nat,+phase: U32,result: Kernel.Leaves() & Array<Sum.TileParts>,
  certificates: TraversalCertificate.Certificates<Kernel.Leaves(),Sum.TileParts,KernelCertificate.leaves_owned,result>) -> CellPair() & Operations.TileSlots<Unit>:
  (leaves,slots) = result
  TraversalCertificate.Certificates{owned,slots_certificate} = certificates
  (both_of(leaves_buffers(left_length,right_length,leaves,KernelCertificate.owned_certificates(leaves,owned))),
    Operations.TileSlots{Storage.Buffer{slots,slots_length,slots_certificate},phase,Unit{}})

def product_buffers(+count: Nat,+position: U32,+step: U32,initial: Operations.TileSlots<Unit>,left: Cells(),right: Cells()) -> CellPair() & Operations.TileSlots<Unit>:
  Operations.TileSlots{parts,+phase,extra} = initial
  Storage.Buffer{slots,+slots_length,slots_certificate} = parts
  Storage.Buffer{left,+left_length,left_certificate} = left
  Storage.Buffer{right,+right_length,right_certificate} = right
  +run = {Traversal.Steps{count,phase,Traversal.Spaced{(position - phase * step : U32),step,Unit{}}} : Traversal.Steps<Traversal.Spaced<Unit>>}
  product_slotted(left_length,right_length,slots_length,(phase + U32.from_nat(count) : U32),
    Traversal.updated(~Kernel.Leaves(),~Sum.TileParts,~Tile.Tile8,~Traversal.Spaced<Unit>,~Kernel.read_product,~Sum.tile_slot_step,~Operations.slot_of(~Unit),run,(left,right),slots),
    KernelCertificate.product_updates(run,left,right,slots,left_certificate,right_certificate,slots_certificate))

# The products of count pairs of cells from position on, step apart, added to
# the slots of their places in the row, both pieces handed back.
def product_slots(+count: Nat,+position: U32,+step: U32,initial: Operations.TileSlots<Unit>,pair: CellPair()) -> CellPair() & Operations.TileSlots<Unit>:
  match pair:
    case Pieces.Both{left,right}: product_buffers(count,position,step,initial,left,right)
    case Pieces.Only{left,+first,+own}: (Pieces.Only{left,first,own},initial)

def vacant_pair(fill: Tile.Tile8) -> CellPair():
  +tile = fill
  Pieces.Both{Pieces.vacant_buffer(~Tile.Tile8,tile),Pieces.vacant_buffer(~Tile.Tile8,tile)}

def scales_of(shifts: List<&2,Tile.Tile8>,result: Pairs() & List<&2,Sum.TileParts>) -> Pairs() & (List<&2,Tile.Tile8> & List<&2,Tile.Tile8>):
  (pairs,partials) = result
  (pairs,(shifts,Partials.finished(~Tile.Tile8,~Sum.TileParts,~sum_finished,partials)))

def scales_after(+shape: List<&2,U32>,+workers: U32,xhat: Pieces.Pieces(~Tile.Tile8),result: Pieces.Pieces(~Tile.Tile8) & List<&2,Tile.Tile8>) ->
  Pairs() & (List<&2,Tile.Tile8> & List<&2,Tile.Tile8>):
  (dy,shifts) = result
  scales_of(shifts,Partials.read(~CellPair(),~Tile.Tile8,~Unit,~Operations.TileSlots<Unit>,~Sum.TileParts,~Operations.slots_begun(~Unit),~product_slots,
    ~Operations.slots_closed(~Unit),~Sum.tile_merge,
    ~vacant_pair,[Unit{}],9.0,2.0,shape,summed_flags(),1n,2n,Tile.filled(0.0),workers,Pieces.zip(~Cells(),~Cells(),dy,xhat)))

# The sums of dy and of dy * xhat of every group in the order of sum_axes,
# and the pairs: the sums of dy as the sums of the statistics, then the pairs
# fold dy * xhat.
def gradient_sums(+shape: List<&2,U32>,+workers: U32,dy: Pieces.Pieces(~Tile.Tile8),xhat: Pieces.Pieces(~Tile.Tile8)) -> Pairs() & (List<&2,Tile.Tile8> & List<&2,Tile.Tile8>):
  scales_after(shape,workers,xhat,sums(shape,workers,dy))

