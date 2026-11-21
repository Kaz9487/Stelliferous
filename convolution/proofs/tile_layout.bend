# The parts of the plan the Array executor runs, with their data erased.
import Base
import ../../lib/traversal_partition.bend as Partition
import ../../lib/tensor_pieces.bend as Pieces
import ../../lib/proofs/tensor_pieces_walk.bend as Walk

def Layout() -> Data:
  Partition.Tree<&2,Unit,Unit>

def divide(depth: Nat,+offset: U32,+span: U32) -> Layout():
  Walk.shape(Pieces.unit_ranges(depth,8,offset,span))
