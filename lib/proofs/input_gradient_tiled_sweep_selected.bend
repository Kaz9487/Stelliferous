# The leaf a query selects: its native context, its run on the packed tables
# and the values of its output.
import Base
import ../storage_buffer.bend as Storage
import ./input_gradient_tiled_chunks.bend as Partition
import ../kernel_input_gradient.bend as Kernel
import ../kernel_tile.bend as Tile
import ./input_gradient_tiled_chunk_reads.bend as Chunks
import ./input_gradient_tiled_chunk_geometry.bend as Geometry
import ./input_gradient_tiled_leaf_execution.bend as Execution
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./input_gradient_phase_domain.bend as Domain

# The packed weight tiles and offset table a leaf reads.
type Packed is Data:
  Packed{weights: Tree.Tree<Tile.Tile8>,offsets: Tree.Tree<U32>}

def native_context(+query: Chunks.Query) -> Kernel.LeafContext:
  +grid = Chunks.grid(query)
  +g = Geometry.grid_geometry(grid)
  Kernel.LeafContext{Kernel.Task{g,Kernel.tiling(g)},Chunks.first(query),Geometry.grid_count(grid),Chunks.target(query),Chunks.plane(query)}
def index(+query: Chunks.Query) -> U32:
  Geometry.native_producer(Chunks.grid(query),Chunks.row(query),Chunks.group(query))
def count(+query: Chunks.Query) -> {Execution.chunk_count(native_context(query)) == Partition.chunk_count(Geometry.query(Chunks.grid(query),0n,0n)) : Nat}:
  Chunks.Query{+grid,first,target,plane,source,row,group,wb,ob} = query
  Geometry.Grid{+call,+e,+rows,+range,+rp,+rr} = grid
  Domain.Call{+g,+b,+l,facts} = call
  Geometry.leaf_count_exact(call,e,rows,first,target,plane,range,rp,rr)
def part_output(part: Execution.Part) -> Buffer.Model<F32>:
  Execution.Part{gradient,output} = part
  output
def run(+query: Chunks.Query,packed: Packed,output: Buffer.Model<F32>) -> Execution.Part:
  Packed{weights,offsets} = packed
  Execution.leaf_band(native_context(query),Chunks.source(query),weights,offsets,output)
def native_run(+query: Chunks.Query,packed: Packed,output: Buffer.Model<F32>) -> Kernel.Part:
  Packed{weights,offsets} = packed
  Kernel.leaf_band(native_context(query),Execution.represent_source(Chunks.source(query)),Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),Buffer.pack(~F32,output))
def run_matches(+query: Chunks.Query,+packed: Packed,+output: Buffer.Model<F32>) -> {native_run(query,packed,output) == Execution.represent_part(run(query,packed,output)) : Kernel.Part}:
  Packed{weights,offsets} = packed
  Execution.leaf_matches(native_context(query),Chunks.source(query),weights,offsets,output)
def native_output(part: Kernel.Part) -> Storage.Buffer<F32>:
  Kernel.Part{gradient,output} = part
  output
def native_value(part: Kernel.Part,index: U32) -> F32:
  Buffer.value(~F32,index,Buffer.contents(~F32,native_output(part)))
def represented_value(+part: Execution.Part,+index: U32) -> {native_value(Execution.represent_part(part),index) == Tree.read(F32,Buffer.tree(~F32,part_output(part)),index) : F32}:
  Execution.Part{gradient,+output} = part
  Buffer.Model{tree,length,certificate} = output
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,output)),Buffer.content(~F32,output),Buffer.packed_contents(~F32,output)) : {Buffer.value(~F32,index,_) == Tree.read(F32,Buffer.tree(~F32,output),index) : F32}
  {==}
