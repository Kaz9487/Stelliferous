# Exact writes of the native channel/block gather; padding contributes no assignment.
import Base
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/write_program.bend as Writes
import ../../lib/proofs/kernel_gather_blocks.bend as Blocks
import ./gather_storage_model.bend as GatherModel
import ../../lib/proofs/write_program_readback.bend as WriteReadback
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/traversal_partition_refinement.bend as PartitionLaws

def gather_writes(result: GatherModel.ConvResult(),+spatial: U32) -> List<&2,Writes.Assignment<F32>>:
  PartitionLaws.tree_writes(~F32,~Blocks.block_writes,result,spatial)
def gather_matches_writes(+result: GatherModel.ConvResult(),+output: Tree.Tree<F32>,+spatial: U32) ->
  {GatherModel.gather(result,output,spatial) == Writes.pure(F32,gather_writes(result,spatial),output) : Tree.Tree<F32>}:
  PartitionLaws.gather_written(~F32,~Blocks.block_writes,result,output,spatial)
def prepend_keeps_coverage(+keep: Bool,+query: Nat,+index: U32,+value: F32,+rest: List<&2,Writes.Assignment<F32>>,
  covered: {WriteReadback.contains_native(query,rest) == True{} : Bool}) -> {WriteReadback.contains_native(query,Writes.prepend(~F32,keep,index,value,rest)) == True{} : Bool}:
  match keep:
    case False{}: covered
    case True{}:
      %Equal.sym(Bool,WriteReadback.contains_native(query,rest),True{},covered) : {Nat.is_eq(query,U32.to_nat(index)) || _ == True{} : Bool}
      Boolean.or_true_right(Nat.is_eq(query,U32.to_nat(index)))
