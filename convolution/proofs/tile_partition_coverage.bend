# Full prepare-tree coverage: concatenating its leaf ranges yields precisely
# one consecutive sequence. Includes every recursive split and empty leaf.
import Base
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ./sequence_congruence.bend as Sequences
import ./tile_layout.bend as TileLayout
import ../../lib/proofs/kernel_partition_bounds.bend as TilePartitionBounds
import ../../lib/ordered_matrix.bend as OrderedMatrix
import ../../lib/proofs/tensor_band_plan.bend as TensorBandPlan
import ../../lib/proofs/traversal_partition_refinement.bend as PartitionLaws

def part_positions(+context: Unit,+offset: U32,+span: U32,+details: Unit,block: Unit) -> List<&2,Nat>:
  Ranges.range(U32.to_nat(span),U32.to_nat(offset))

def joined(left: List<&2,Nat>,right: List<&2,Nat>) -> List<&2,Nat>:
  OrderedMatrix.cat(Nat,left,right)

def positions(t: TileLayout.Layout()) -> List<&2,Nat>:
  PartitionLaws.reduce(~Unit,~Unit,~List<&2,Nat>,~Unit,~part_positions,~joined,Unit{},t)

law split_tiles_cover_range:
  for +depth: Nat
  for +offset: U32
  for +span: U32
  for +cap: U32
  for +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool}
  for +room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),U32.to_nat(cap)) == True{} : Bool}
  {positions(TileLayout.divide(depth,offset,span)) == Ranges.range(U32.to_nat(span),U32.to_nat(offset)) : List<&2,Nat>}
def split_tiles_cover_range(depth,offset,span,cap,cap_ok,room):
  match depth:
    case 0n:
      {==}
    case 1n+ +rest:
      +half = TilePartitionBounds.half(span)
      +end = (offset + half : U32)
      +rest_span = (span - half : U32)
      +left = TileLayout.divide(rest,offset,half)
      +right = TileLayout.divide(rest,end,rest_span)
      +left_range = Ranges.range(U32.to_nat(half),U32.to_nat(offset))
      +right_range = Ranges.range(U32.to_nat(rest_span),U32.to_nat(end))
      Equal.trans(List<&2,Nat>,
        OrderedMatrix.cat(Nat,positions(left),positions(right)),
        OrderedMatrix.cat(Nat,left_range,right_range),
        Ranges.range(U32.to_nat(span),U32.to_nat(offset)),
        Sequences.cat(Nat,positions(left),positions(right),left_range,right_range,
          split_tiles_cover_range(rest,offset,half,cap,cap_ok,TilePartitionBounds.left_room(offset,span,cap,cap_ok,room)),
          split_tiles_cover_range(rest,end,rest_span,cap,cap_ok,TilePartitionBounds.right_room(offset,span,cap,cap_ok,room))),
        Ranges.join(U32.to_nat(offset),U32.to_nat(half),U32.to_nat(rest_span),U32.to_nat(end),U32.to_nat(span),
          TilePartitionBounds.offset_exact(offset,span,cap,cap_ok,room),
          TilePartitionBounds.sum(span,cap,cap_ok,TensorBandPlan.rows_bound(offset,span,cap,room))))
