# The Array entry packs its owners, partitions cells, merges cells, then untile once.
import Base
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../specification.bend as ConvSpec
import ../shape.bend as ConvShape
import ../inference.bend as NativeConv
import ./input_storage.bend as Inputs
import ./microkernel_storage_execution.bend as Execution
import ./window_cell_values.bend as CellValues

import ./array_cell_execution.bend as CellExecution

def native_run_matches(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+depth: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {NativeConv.run(ConvShape.from_problem(problem),depth,Inputs.input_owner(problem,storage),Inputs.weight_owner(problem,storage),Inputs.bias_owner(problem,storage)) ==
    Tree.pack(F32,CellExecution.output(problem,CellValues.root_contract(limits,problem,storage,valid,supported),depth)) : Array<F32>}:
  Equal.trans(Array<F32>,NativeConv.run(ConvShape.from_problem(problem),depth,Inputs.input_owner(problem,storage),Inputs.weight_owner(problem,storage),Inputs.bias_owner(problem,storage)),
    Tree.pack(F32,Execution.run(depth,CellExecution.work(problem),Inputs.input_tree(problem,storage),Inputs.weight_tree(problem,storage),Inputs.bias_tree(problem,storage))),
    Tree.pack(F32,CellExecution.output(problem,CellValues.root_contract(limits,problem,storage,valid,supported),depth)),
    Execution.run_matches(depth,CellExecution.work(problem),Inputs.input_tree(problem,storage),Inputs.weight_tree(problem,storage),Inputs.bias_tree(problem,storage)),
    Equal.cong(Tree.Tree<F32>,Array<F32>,tree => Tree.pack(F32,tree),Execution.run(depth,CellExecution.work(problem),Inputs.input_tree(problem,storage),Inputs.weight_tree(problem,storage),Inputs.bias_tree(problem,storage)),
      CellExecution.output(problem,CellValues.root_contract(limits,problem,storage,valid,supported),depth),CellExecution.run_output(limits,problem,storage,depth,valid,supported)))
