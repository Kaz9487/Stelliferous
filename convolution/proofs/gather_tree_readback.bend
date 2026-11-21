# The cells gather tree writes in bounds, covers each real position, and preserves its lane value.
import Base
import ../specification.bend as ConvSpec
import ../inference.bend as NativeConv
import ../../lib/machine_limits.bend as Machine
import ../../lib/kernel_tile.bend as Tile
import ../../lib/traversal_partition.bend as Partition
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/write_program.bend as Writes
import ../../lib/proofs/write_program_readback.bend as Readback
import ../../lib/proofs/write_program_observation.bend as Observation
import ../../lib/proofs/traversal_partition_refinement.bend as PartitionLaws
import ../../lib/proofs/sequence_membership.bend as Members
import ./input_storage.bend as Inputs
import ./convolution_guard_facts.bend as Guards
import ./supported_padded_coordinates.bend as Padded
import ./reduction_coordinate_bounds.bend as Reduction
import ./array_leaf_cells.bend as ArrayLeaf
import ./tile_layout.bend as TileLayout
import ./tile_partition_coverage.bend as TileCoverage
import ./microkernel_storage_execution.bend as Execution
import ./cell_gather_leaf.bend as Leaf
import ./cell_gather_coordinates.bend as Coordinates
import ./cell_gather_values.bend as CellValues
import ./output_values.bend as Values
import ../../lib/kernel_gemm.bend as KernelGemm

def result(+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,layout: TileLayout.Layout()) -> Execution.CellResult():
  ArrayLeaf.layout_cells(problem,storage,layout)

def writes(+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,layout: TileLayout.Layout()) -> List<&2,Writes.Assignment<Tile.Tile8>>:
  PartitionLaws.tree_writes(~Tile.Tile8,~Execution.cell_block_writes,result(problem,storage,layout),Padded.spatial(problem))

def observed(+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,layout: TileLayout.Layout(),channel: Nat) -> List<&2,Writes.Assignment<F32>>:
  Observation.projected(~Tile.Tile8,~F32,~Nat,~CellValues.lane,writes(problem,storage,layout),Nat.mod(channel,8n))

def inside(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+layout: TileLayout.Layout(),+channel: Nat,+row: Nat,+column: Nat,+capacity: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},certificates: ArrayLeaf.AllGeometry(problem,layout),
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool},
  +slots: {Nat.is_le(Nat.mul(U32.to_nat(Coordinates.groups(problem)),U32.to_nat(Padded.spatial(problem))),U32.to_nat(capacity)) == True{} : Bool}) ->
  Readback.Inside(~Tile.Tile8,capacity,writes(problem,storage,layout)):
  match layout:
    case Partition.Leaf{Unit{},+offset,+span,Unit{}}:
      Leaf.inside(Leaf.Frame{limits,problem,storage,offset,span,channel,row,column,capacity,valid,supported,certificates,channel_inside,row_inside,column_inside,slots})
    case Partition.Fork{+left,+right}:
      KernelGemm.Both{left_certificates,right_certificates} = certificates
      Readback.inside_append(~Tile.Tile8,capacity,writes(problem,storage,left),writes(problem,storage,right),
        inside(limits,problem,storage,left,channel,row,column,capacity,valid,supported,left_certificates,channel_inside,row_inside,column_inside,slots),
        inside(limits,problem,storage,right,channel,row,column,capacity,valid,supported,right_certificates,channel_inside,row_inside,column_inside,slots))

def matching(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+layout: TileLayout.Layout(),+channel: Nat,+row: Nat,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},certificates: ArrayLeaf.AllGeometry(problem,layout),
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  Readback.Matching(~F32,U32.from_nat(Coordinates.query_index(problem,channel,row,column)),Values.matrix_value(problem,channel,row,column),observed(problem,storage,layout,channel)):
  match layout:
    case Partition.Leaf{Unit{},+offset,+span,Unit{}}:
      Leaf.matching(Leaf.Frame{limits,problem,storage,offset,span,channel,row,column,Machine.capacity(limits),valid,supported,certificates,channel_inside,row_inside,column_inside,Coordinates.slots_bound(limits,problem,valid,supported)})
    case Partition.Fork{+left,+right}:
      KernelGemm.Both{left_certificates,right_certificates} = certificates
      %Equal.sym(List<&2,Writes.Assignment<F32>>,Observation.projected(~Tile.Tile8,~F32,~Nat,~CellValues.lane,List.append(&2,Writes.Assignment<Tile.Tile8>,writes(problem,storage,left),writes(problem,storage,right)),Nat.mod(channel,8n)),
        List.append(&2,Writes.Assignment<F32>,observed(problem,storage,left,channel),observed(problem,storage,right,channel)),
        Observation.appended(~Tile.Tile8,~F32,~Nat,~CellValues.lane,Nat.mod(channel,8n),writes(problem,storage,left),writes(problem,storage,right))) :
        Readback.Matching(~F32,U32.from_nat(Coordinates.query_index(problem,channel,row,column)),Values.matrix_value(problem,channel,row,column),_)
      Readback.matching_append(~F32,U32.from_nat(Coordinates.query_index(problem,channel,row,column)),Values.matrix_value(problem,channel,row,column),observed(problem,storage,left,channel),observed(problem,storage,right,channel),
        matching(limits,problem,storage,left,channel,row,column,valid,supported,left_certificates,channel_inside,row_inside,column_inside),
        matching(limits,problem,storage,right,channel,row,column,valid,supported,right_certificates,channel_inside,row_inside,column_inside))

def leaf_coverage(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+offset: U32,+span: U32,+channel: Nat,+row: Nat,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+certificate: ArrayLeaf.Geometry<problem,offset,span>,
  witness: Members.RangeIndex<U32.to_nat(span),U32.to_nat(offset),Reduction.position(problem,row,column)>,
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {Readback.contains(~Tile.Tile8,U32.from_nat(Coordinates.query_index(problem,channel,row,column)),writes(problem,storage,Partition.Leaf{Unit{},offset,span,Unit{}})) == True{} : Bool}:
  Members.RangeIndex{+local,+inside,+exact} = witness
  Leaf.covered(Leaf.Frame{limits,problem,storage,offset,span,channel,row,column,Machine.capacity(limits),valid,supported,certificate,channel_inside,row_inside,column_inside,Coordinates.slots_bound(limits,problem,valid,supported)},local,inside,exact)

def fork_coverage(+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+left: TileLayout.Layout(),+right: TileLayout.Layout(),+channel: Nat,+row: Nat,+column: Nat,
  choice: Either<&2,&2,Members.Member(Nat,Reduction.position(problem,row,column),TileCoverage.positions(left)),Members.Member(Nat,Reduction.position(problem,row,column),TileCoverage.positions(right))>,
  left_proof: Members.Member(Nat,Reduction.position(problem,row,column),TileCoverage.positions(left)) ->
    {Readback.contains(~Tile.Tile8,U32.from_nat(Coordinates.query_index(problem,channel,row,column)),writes(problem,storage,left)) == True{} : Bool},
  right_proof: Members.Member(Nat,Reduction.position(problem,row,column),TileCoverage.positions(right)) ->
    {Readback.contains(~Tile.Tile8,U32.from_nat(Coordinates.query_index(problem,channel,row,column)),writes(problem,storage,right)) == True{} : Bool}) ->
  {Readback.contains(~Tile.Tile8,U32.from_nat(Coordinates.query_index(problem,channel,row,column)),writes(problem,storage,Partition.Fork{left,right})) == True{} : Bool}:
  match choice:
    case Inl{member}: Readback.covered_left(~Tile.Tile8,U32.from_nat(Coordinates.query_index(problem,channel,row,column)),writes(problem,storage,left),writes(problem,storage,right),left_proof(member))
    case Inr{member}: Readback.covered_right(~Tile.Tile8,U32.from_nat(Coordinates.query_index(problem,channel,row,column)),writes(problem,storage,left),writes(problem,storage,right),right_proof(member))

def tree_coverage(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+layout: TileLayout.Layout(),+channel: Nat,+row: Nat,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},certificates: ArrayLeaf.AllGeometry(problem,layout),
  member: Members.Member(Nat,Reduction.position(problem,row,column),TileCoverage.positions(layout)),
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {Readback.contains(~Tile.Tile8,U32.from_nat(Coordinates.query_index(problem,channel,row,column)),writes(problem,storage,layout)) == True{} : Bool}:
  match layout:
    case Partition.Leaf{Unit{},+offset,+span,Unit{}}:
      leaf_coverage(limits,problem,storage,offset,span,channel,row,column,valid,supported,certificates,
        Members.range_index(U32.to_nat(span),U32.to_nat(offset),Reduction.position(problem,row,column),member),channel_inside,row_inside,column_inside)
    case Partition.Fork{+left,+right}:
      KernelGemm.Both{left_certificates,right_certificates} = certificates
      fork_coverage(problem,storage,left,right,channel,row,column,
        Members.split_concatenated(Nat,Reduction.position(problem,row,column),TileCoverage.positions(left),TileCoverage.positions(right),member),
        left_member => tree_coverage(limits,problem,storage,left,channel,row,column,valid,supported,left_certificates,left_member,channel_inside,row_inside,column_inside),
        right_member => tree_coverage(limits,problem,storage,right,channel,row,column,valid,supported,right_certificates,right_member,channel_inside,row_inside,column_inside))

def readback(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+layout: TileLayout.Layout(),+output: Tree.Tree<Tile.Tile8>,+depth: Nat,+channel: Nat,+row: Nat,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+certificates: ArrayLeaf.AllGeometry(problem,layout),
  +member: Members.Member(Nat,Reduction.position(problem,row,column),TileCoverage.positions(layout)),
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool},
  +balanced: Shape.Balanced(Tile.Tile8,depth,output),+depth_bound: {Nat.is_le(depth,30n) == True{} : Bool},
  +query_inside: {U32.is_lt(U32.from_nat(Coordinates.query_index(problem,channel,row,column)),Tree.capacity(Tile.Tile8,output)) == True{} : Bool},
  +slots: {Nat.is_le(Nat.mul(U32.to_nat(Coordinates.groups(problem)),U32.to_nat(Padded.spatial(problem))),U32.to_nat(Tree.capacity(Tile.Tile8,output))) == True{} : Bool}) ->
  {CellValues.lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,Execution.gather(result(problem,storage,layout),output,Padded.spatial(problem)),U32.from_nat(Coordinates.query_index(problem,channel,row,column)))) == Values.matrix_value(problem,channel,row,column) : F32}:
  +query = U32.from_nat(Coordinates.query_index(problem,channel,row,column))
  %Equal.sym(Tree.Tree<Tile.Tile8>,Execution.gather(result(problem,storage,layout),output,Padded.spatial(problem)),Writes.pure(Tile.Tile8,writes(problem,storage,layout),output),
    PartitionLaws.gather_written(~Tile.Tile8,~Execution.cell_block_writes,result(problem,storage,layout),output,Padded.spatial(problem))) :
    {CellValues.lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,_,query)) == Values.matrix_value(problem,channel,row,column) : F32}
  Observation.covered(~Tile.Tile8,~F32,~Nat,~CellValues.lane,Nat.mod(channel,8n),writes(problem,storage,layout),depth,output,query,Values.matrix_value(problem,channel,row,column),balanced,depth_bound,query_inside,
    inside(limits,problem,storage,layout,channel,row,column,Tree.capacity(Tile.Tile8,output),valid,supported,certificates,channel_inside,row_inside,column_inside,slots),
    matching(limits,problem,storage,layout,channel,row,column,valid,supported,certificates,channel_inside,row_inside,column_inside),
    tree_coverage(limits,problem,storage,layout,channel,row,column,valid,supported,certificates,member,channel_inside,row_inside,column_inside))
