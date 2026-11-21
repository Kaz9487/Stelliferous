# The leaf a query selects: its native context, its run on the packed
# weights and the values of its output.
import Base
import ./input_gradient_tiled_chunks.bend as Partition
import ../kernel_input_gradient.bend as Kernel
import ../kernel_tile.bend as Tile
import ./input_gradient_tiled_chunk_reads.bend as Chunks
import ./input_gradient_tiled_chunk_geometry.bend as Geometry
import ./input_gradient_tiled_leaf_execution.bend as Execution
import ./storage_buffer_model.bend as Buffer
import ./input_gradient_phase_domain.bend as Domain

def native_context(+query: Chunks.Query) -> Kernel.LeafContext:
  +grid = Chunks.grid(query)
  +g = Geometry.grid_geometry(grid)
  Kernel.LeafContext{Kernel.Task{g,Kernel.tiling(g)},Chunks.first(query),Geometry.grid_count(grid),Chunks.target(query)}
def count(+query: Chunks.Query) -> {Execution.chunk_count(native_context(query)) == Partition.chunk_count(Geometry.query(Chunks.grid(query),0n,0n)) : Nat}:
  Chunks.Query{+grid,first,target,source,row,group,wb} = query
  Geometry.Grid{+call,+e,+rows,+range,+rp,+rr} = grid
  Domain.Call{+g,+b,+l,facts} = call
  Geometry.leaf_count_exact(call,e,rows,first,target,range,rp,rr)
def part_output(part: Execution.Part) -> Buffer.Model<Tile.Tile8>:
  Execution.part_output(part)
