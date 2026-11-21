# The Array entry packs its owners, partitions cells, merges cells, then untile once.
import Base
import ../../lib/tensor.bend as Tensors
import ../../lib/storage_buffer.bend as Storage
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/tensor_allocation_depth.bend as Depth
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/tensor_tiled_values.bend as TiledValues
import ../../lib/proofs/kernel_weights_refinement.bend as Weights
import ../../lib/kernel_tile.bend as Tile
import ../specification.bend as ConvSpec
import ../shape.bend as ConvShape
import ../inference.bend as NativeConv
import ./input_storage.bend as Inputs
import ./convolution_guard_facts.bend as Guards
import ./supported_padded_coordinates.bend as Padded
import ./reduction_coordinate_bounds.bend as Coordinates
import ./tile_layout.bend as TileLayout
import ./microkernel_storage_execution.bend as Execution
import ./array_leaf_cells.bend as ArrayLeaf
import ./window_partition.bend as Windows
import ./window_cell_values.bend as CellValues
import ./window_cell_contents.bend as CellContents
import ./microkernel_leaf_readback.bend as MicroLeaf
import ./window_leaf_readback.bend as LeafReadback
import ./concrete_support_guard.bend as Support

def work(+problem: ConvSpec.Problem) -> Pipeline.Work:
  NativeConv.work(ConvShape.from_problem(problem))

def selected_depth(+problem: ConvSpec.Problem,depth: U32) -> Nat:
  U32.to_nat(U32.min(depth,(U32.from_nat(Storage.bit_length(32n,(Padded.padded(problem) / 8 : U32))) - 1 : U32)))

def cells_output(+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,depth: U32) -> Tree.Tree<Tile.Tile8>:
  Execution.finish(ArrayLeaf.layout_cells(problem,storage,TileLayout.divide(selected_depth(problem,depth),0,Padded.padded(problem))),MicroLeaf.rows(problem),Padded.spatial(problem))

def output(+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,depth: U32) -> Tree.Tree<F32>:
  TiledValues.image_values(MicroLeaf.rows(problem),Padded.spatial(problem),cells_output(problem,storage,depth))

def run_output_with(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+reference: Inputs.Contract<problem>,+depth: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +weights_equal: {Weights.run(MicroLeaf.rows(problem),Coordinates.native_inner(problem),Inputs.weight_tree(problem,storage)) == Inputs.weight_cells(problem,reference) : Tree.Tree<Tile.Tile8>},
  +bias_equal: {Weights.run(MicroLeaf.rows(problem),1,Inputs.bias_tree(problem,storage)) == Inputs.bias_cells(problem,reference) : Tree.Tree<Tile.Tile8>},
  +values: {LeafReadback.input_values(problem,CellContents.input_cells(problem,storage)) == Inputs.input_tree(problem,reference) : Tree.Tree<F32>}) ->
  {Execution.run(depth,work(problem),Inputs.input_tree(problem,storage),Inputs.weight_tree(problem,storage),Inputs.bias_tree(problem,storage)) ==
    output(problem,reference,depth) : Tree.Tree<F32>}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  Inputs.Contract{+input_evidence,+weight_evidence,+bias_evidence,+parameters} = storage
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +storage = {Inputs.Contract{input_evidence,weight_evidence,bias_evidence,parameters} : Inputs.Contract<problem>}
  +inner = Coordinates.native_inner(problem)
  +channels = (inner / (U32.from_nat(kernel) * U32.from_nat(kernel)) : U32)
  +channel_equal = Encoded.encoding(channels,ci,Coordinates.inner_channels(limits,problem,Guards.valid(problem,valid),Padded.scalar(limits,problem,Support.fast(limits,problem,supported))))
  +blocks = (Padded.padded(problem) / 8 : U32)
  +split = selected_depth(problem,depth)
  +parts = Execution.partition(split,work(problem),0,Padded.padded(problem),Windows.root_window(problem,storage),Inputs.weight_cells(problem,reference),Inputs.bias_cells(problem,reference))
  +layout = ArrayLeaf.layout_cells(problem,reference,TileLayout.divide(split,0,Padded.padded(problem)))
  %Equal.sym(Tree.Tree<Tile.Tile8>,Weights.run(MicroLeaf.rows(problem),inner,Inputs.weight_tree(problem,storage)),Inputs.weight_cells(problem,reference),weights_equal) :
    {TiledValues.image_values(MicroLeaf.rows(problem),Padded.spatial(problem),Execution.run_cells(depth,work(problem),Execution.input_cells(channels,(U32.from_nat(height) * U32.from_nat(width) : U32),Inputs.input_tree(problem,storage)),_,Weights.run(MicroLeaf.rows(problem),1,Inputs.bias_tree(problem,storage)))) == output(problem,reference,depth) : Tree.Tree<F32>}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Weights.run(MicroLeaf.rows(problem),1,Inputs.bias_tree(problem,storage)),Inputs.bias_cells(problem,reference),bias_equal) :
    {TiledValues.image_values(MicroLeaf.rows(problem),Padded.spatial(problem),Execution.run_cells(depth,work(problem),Execution.input_cells(channels,(U32.from_nat(height) * U32.from_nat(width) : U32),Inputs.input_tree(problem,storage)),Inputs.weight_cells(problem,reference),_)) == output(problem,reference,depth) : Tree.Tree<F32>}
  %Equal.sym(U32,channels,U32.from_nat(ci),channel_equal) :
    {TiledValues.image_values(MicroLeaf.rows(problem),Padded.spatial(problem),Execution.run_cells(depth,work(problem),
      Execution.input_cells(_,(U32.from_nat(height) * U32.from_nat(width) : U32),Inputs.input_tree(problem,storage)),Inputs.weight_cells(problem,reference),Inputs.bias_cells(problem,reference))) == output(problem,reference,depth) : Tree.Tree<F32>}
  %Equal.sym(Nat,Tensors.significant_bits(32n,blocks,0n,U32.is_zero(blocks)),Storage.bit_length(32n,blocks),Depth.countdown(32n,blocks,0n)) :
    {TiledValues.image_values(MicroLeaf.rows(problem),Padded.spatial(problem),Execution.finish(Execution.partition(U32.to_nat(U32.min(depth,(U32.from_nat(_) - 1 : U32))),work(problem),0,Padded.padded(problem),
      Windows.root_window(problem,storage),Inputs.weight_cells(problem,reference),Inputs.bias_cells(problem,reference)),MicroLeaf.rows(problem),Padded.spatial(problem))) == output(problem,reference,depth) : Tree.Tree<F32>}
  Equal.cong(Execution.CellResult(),Tree.Tree<F32>,tree => TiledValues.image_values(MicroLeaf.rows(problem),Padded.spatial(problem),Execution.finish(tree,MicroLeaf.rows(problem),Padded.spatial(problem))),
    parts,layout,Windows.partition_root(limits,problem,storage,reference,values,valid,supported,split))


def run_output(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+depth: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {Execution.run(depth,work(problem),Inputs.input_tree(problem,storage),Inputs.weight_tree(problem,storage),Inputs.bias_tree(problem,storage)) ==
    output(problem,CellValues.root_contract(limits,problem,storage,valid,supported),depth) : Tree.Tree<F32>}:
  run_output_with(limits,problem,storage,CellValues.root_contract(limits,problem,storage,valid,supported),depth,valid,supported,
    CellValues.root_weights(limits,problem,storage,valid,supported),CellValues.root_bias(limits,problem,storage,valid,supported),CellValues.root_values(limits,problem,storage,valid,supported))
