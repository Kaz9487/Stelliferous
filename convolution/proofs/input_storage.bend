# Input contents and capacity, shared by every convolution execution strategy.
import Base
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/machine_limits.bend as Machine
import ../specification.bend as ConvSpec
import ../scalar_kernel.bend as ScalarConv
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./list_array_roundtrip.bend as Roundtrip
import ./native_weight_refinement.bend as NativeWeightRefinement
import ../../lib/proofs/list_storage_readback.bend as ListRead
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/tensor_tiled_values.bend as TiledValues
import ../../lib/proofs/kernel_weights_refinement.bend as Weights

def parameter_inner(+problem: ConvSpec.Problem) -> Nat:
  Nat.mul(Guards.input_channel_count(problem),Nat.mul(Guards.kernel(problem),Guards.kernel(problem)))

def parameter_values(+count: Nat,+channels: Nat,+plane: Nat,cells: Tree.Tree<Tile.Tile8>) -> List<&2,F32>:
  ListRead.enumerate(~F32,~TiledValues.CellSource,~TiledValues.cell_value,count,0n,TiledValues.CellSource{cells,channels,plane})

# Array callers pack at their boundary; tensor callers supply their cell owners.
type ParameterSource<-channels: Nat,-inner: Nat,-weight_list: List<&2,F32>,-bias_list: List<&2,F32>> is Data:
  ArrayParameters{}
  CellParameters{weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<Tile.Tile8>,
    weight_values: {parameter_values(ConvSpec.length(weight_list),channels,inner,weights) == weight_list : List<&2,F32>},
    bias_values: {parameter_values(ConvSpec.length(bias_list),channels,1n,bias) == bias_list : List<&2,F32>}}

def Parameters(+problem: ConvSpec.Problem) -> Data:
  ParameterSource<Guards.output_channel_count(problem),parameter_inner(problem),Guards.weights(problem),Guards.bias(problem)>

def length_matches_list(+values: List<&2,F32>) ->
  {ConvSpec.length(values) == List.length(&2,F32,values) : Nat}:
  match values:
    case Nil{}: {==}
    case head <> +tail:
      %length_matches_list(tail) : {1n+ConvSpec.length(tail) == 1n+_ : Nat}
      {==}

def observed_length(+length: Nat,+tree: Tree.Tree<F32>) ->
  {ConvSpec.length(Tree.read_sequence(~F32,length,0,tree)) == length : Nat}:
  Equal.trans(Nat,ConvSpec.length(Tree.read_sequence(~F32,length,0,tree)),
    List.length(&2,F32,Tree.read_sequence(~F32,length,0,tree)),length,
    length_matches_list(Tree.read_sequence(~F32,length,0,tree)),StorageLaws.sequence_length(~F32,length,0,tree))

def lookup_matches_list(+index: Nat,+values: List<&2,F32>) ->
  {ConvSpec.at(index,values) == Maybe.default(&2,F32,List.get(&2,F32,values,index),0.0) : F32}:
  match index values:
    case 0n Nil{}: {==}
    case 1n+previous Nil{}: {==}
    case 0n head <> tail: {==}
    case 1n+previous head <> tail: lookup_matches_list(previous,tail)

def lookup_matches_readback(+index: Nat,+values: List<&2,F32>) ->
  {ConvSpec.at(index,values) == ListRead.at(~F32,0.0,index,values) : F32}:
  match index values:
    case 0n Nil{}: {==}
    case 1n+previous Nil{}: {==}
    case 0n head <> tail: {==}
    case 1n+previous head <> tail: lookup_matches_readback(previous,tail)

def prefix_evidence(+length: Nat,+storage: Tree.Tree<F32>,
  room: {Nat.is_le(length,U32.to_nat(Tree.capacity(F32,storage))) == True{} : Bool}) ->
  StorageLaws.SequenceEvidence<F32,ConvSpec.length(Tree.read_sequence(~F32,length,0,storage)),Tree.read_sequence(~F32,length,0,storage)>:
  StorageLaws.SequenceEvidence{storage,
    %Equal.sym(Nat,ConvSpec.length(Tree.read_sequence(~F32,length,0,storage)),length,observed_length(length,storage)) :
      {Tree.read_sequence(~F32,_,0,storage) == Tree.read_sequence(~F32,length,0,storage) : List<&2,F32>}
    {==},
    %Equal.sym(Nat,ConvSpec.length(Tree.read_sequence(~F32,length,0,storage)),length,observed_length(length,storage)) :
      {Nat.is_le(_,U32.to_nat(Tree.capacity(F32,storage))) == True{} : Bool}
    room}

# Evidence is attached to the supplied storage, not to an allocator identity.
type Contract<-problem: ConvSpec.Problem> is Data:
  Contract{
    input: StorageLaws.SequenceEvidence<F32,ConvSpec.length(Guards.input(problem)),Guards.input(problem)>,
    weights: StorageLaws.SequenceEvidence<F32,ConvSpec.length(Guards.weights(problem)),Guards.weights(problem)>,
    bias: StorageLaws.SequenceEvidence<F32,ConvSpec.length(Guards.bias(problem)),Guards.bias(problem)>,parameters: Parameters(problem)}

def input_sequence(-problem: ConvSpec.Problem,inputs: Contract<problem>) ->
  StorageLaws.SequenceEvidence<F32,ConvSpec.length(Guards.input(problem)),Guards.input(problem)>:
  Contract{input,weights,bias,parameters} = inputs
  input

def weight_sequence(-problem: ConvSpec.Problem,inputs: Contract<problem>) ->
  StorageLaws.SequenceEvidence<F32,ConvSpec.length(Guards.weights(problem)),Guards.weights(problem)>:
  Contract{input,weights,bias,parameters} = inputs
  weights

def bias_sequence(-problem: ConvSpec.Problem,inputs: Contract<problem>) ->
  StorageLaws.SequenceEvidence<F32,ConvSpec.length(Guards.bias(problem)),Guards.bias(problem)>:
  Contract{input,weights,bias,parameters} = inputs
  bias

def input_tree(-problem: ConvSpec.Problem,inputs: Contract<problem>) -> Tree.Tree<F32>:
  StorageLaws.sequence_storage(F32,ConvSpec.length(Guards.input(problem)),Guards.input(problem),input_sequence(problem,inputs))

def weight_tree(-problem: ConvSpec.Problem,inputs: Contract<problem>) -> Tree.Tree<F32>:
  Contract{input,weights,bias,parameters} = inputs
  StorageLaws.sequence_storage(F32,ConvSpec.length(Guards.weights(problem)),Guards.weights(problem),weights)

def bias_tree(-problem: ConvSpec.Problem,inputs: Contract<problem>) -> Tree.Tree<F32>:
  Contract{input,weights,bias,parameters} = inputs
  StorageLaws.sequence_storage(F32,ConvSpec.length(Guards.bias(problem)),Guards.bias(problem),bias)

def input_owner(-problem: ConvSpec.Problem,inputs: Contract<problem>) -> Array<F32>:
  Tree.pack(F32,input_tree(problem,inputs))

def weight_owner(-problem: ConvSpec.Problem,inputs: Contract<problem>) -> Array<F32>:
  Tree.pack(F32,weight_tree(problem,inputs))

def bias_owner(-problem: ConvSpec.Problem,inputs: Contract<problem>) -> Array<F32>:
  Tree.pack(F32,bias_tree(problem,inputs))

def stored_value(+values: List<&2,F32>,+evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(values),values>,+index: Nat,
  inside: {Nat.is_lt(index,ConvSpec.length(values)) == True{} : Bool}) ->
  {Tree.read(F32,StorageLaws.sequence_storage(F32,ConvSpec.length(values),values,evidence),U32.from_nat(index)) == ConvSpec.at(index,values) : F32}:
  %Equal.sym(F32,ConvSpec.at(index,values),Maybe.default(&2,F32,List.get(&2,F32,values,index),0.0),lookup_matches_list(index,values)) :
    {Tree.read(F32,StorageLaws.sequence_storage(F32,ConvSpec.length(values),values,evidence),U32.from_nat(index)) == _ : F32}
  StorageLaws.stored_sequence_element(~F32,0.0,ConvSpec.length(values),values,evidence,index,inside)

# Coordinates establish logical bounds; the common prefix theorem establishes
# the read. Neither theorem below depends on how these owners were allocated.
def weight_matches_spec(+problem: ConvSpec.Problem,+inputs: Contract<problem>,
  +output_channel: Nat,+input_channel: Nat,+kernel_row: Nat,+kernel_column: Nat,
  valid: Guards.Valid<problem>,output_inside: {Nat.is_lt(output_channel,Guards.output_channel_count(problem)) == True{} : Bool},
  input_inside: {Nat.is_lt(input_channel,Guards.input_channel_count(problem)) == True{} : Bool},
  row_inside: {Nat.is_lt(kernel_row,Guards.kernel(problem)) == True{} : Bool},
  column_inside: {Nat.is_lt(kernel_column,Guards.kernel(problem)) == True{} : Bool}) ->
  {Tree.read(F32,weight_tree(problem,inputs),U32.from_nat(NativeWeightRefinement.weight_index(output_channel,Guards.input_channel_count(problem),input_channel,
      Guards.kernel(problem),kernel_row,kernel_column))) == ConvSpec.weight(problem,output_channel,input_channel,kernel_row,kernel_column) : F32}:
  ConvSpec.Problem{shape,input_values,weight_values,bias_values} = problem
  ConvSpec.Shape{input_channel_count,output_channel_count,height,width,kernel,stride,padding} = shape
  Contract{input,+weights,bias,parameters} = inputs
  stored_value(weight_values,weights,
    NativeWeightRefinement.weight_index(output_channel,input_channel_count,input_channel,kernel,kernel_row,kernel_column),
    NativeWeightRefinement.index_bound(problem,output_channel,input_channel,kernel_row,kernel_column,valid,output_inside,input_inside,row_inside,column_inside))

def bias_matches_spec(+problem: ConvSpec.Problem,+inputs: Contract<problem>,+channel: Nat,
  valid: Guards.Valid<problem>,inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool}) ->
  {Tree.read(F32,bias_tree(problem,inputs),U32.from_nat(channel)) == ConvSpec.bias_at(problem,channel) : F32}:
  ConvSpec.Problem{shape,input_values,weight_values,bias_values} = problem
  Contract{input,weights,+bias,parameters} = inputs
  stored_value(bias_values,bias,channel,
    NativeWeightRefinement.bias_index(problem,channel,valid,inside))

def from_lists(+limits: Machine.Limits,+problem: ConvSpec.Problem,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {ScalarConv.supported_with_limits(limits,problem) == True{} : Bool}) -> Contract<problem>:
  +facts = Guards.valid(problem,valid)
  +bounds = Support.scalar(limits,problem,supported)
  Contract{
    Roundtrip.sequence_evidence(Guards.input(problem),Support.input_ready_facts(limits,problem,facts,bounds)),
    Roundtrip.sequence_evidence(Guards.weights(problem),Support.weights_ready_facts(limits,problem,facts,bounds)),
    Roundtrip.sequence_evidence(Guards.bias(problem),Support.bias_ready_facts(limits,problem,facts,bounds)),ArrayParameters{}}

def weights_from(+channels: Nat,+outputs: Nat,+kernel: Nat,+weight_list: List<&2,F32>,+bias_list: List<&2,F32>,
  weights: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weight_list),weight_list>,parameters: ParameterSource<outputs,Nat.mul(channels,Nat.mul(kernel,kernel)),weight_list,bias_list>) -> Tree.Tree<Tile.Tile8>:
  match parameters:
    case ArrayParameters{}:
      Weights.run(U32.from_nat(outputs),(U32.from_nat(channels) * U32.from_nat(kernel) * U32.from_nat(kernel) : U32),
        StorageLaws.sequence_storage(F32,ConvSpec.length(weight_list),weight_list,weights))
    case CellParameters{weights,bias,weight_values,bias_values}: weights

def bias_from(+outputs: Nat,+inner: Nat,+weight_list: List<&2,F32>,+bias_list: List<&2,F32>,
  bias: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias_list),bias_list>,parameters: ParameterSource<outputs,inner,weight_list,bias_list>) -> Tree.Tree<Tile.Tile8>:
  match parameters:
    case ArrayParameters{}:
      Weights.run(U32.from_nat(outputs),1,StorageLaws.sequence_storage(F32,ConvSpec.length(bias_list),bias_list,bias))
    case CellParameters{weights,bias,weight_values,bias_values}: bias

def parameter_weights(+problem: ConvSpec.Problem,weights: StorageLaws.SequenceEvidence<F32,ConvSpec.length(Guards.weights(problem)),Guards.weights(problem)>,parameters: Parameters(problem)) -> Tree.Tree<Tile.Tile8>:
  weights_from(Guards.input_channel_count(problem),Guards.output_channel_count(problem),Guards.kernel(problem),Guards.weights(problem),Guards.bias(problem),weights,parameters)

def parameter_bias(+problem: ConvSpec.Problem,bias: StorageLaws.SequenceEvidence<F32,ConvSpec.length(Guards.bias(problem)),Guards.bias(problem)>,parameters: Parameters(problem)) -> Tree.Tree<Tile.Tile8>:
  bias_from(Guards.output_channel_count(problem),parameter_inner(problem),Guards.weights(problem),Guards.bias(problem),bias,parameters)

def weight_cells(+problem: ConvSpec.Problem,inputs: Contract<problem>) -> Tree.Tree<Tile.Tile8>:
  Contract{input,weights,bias,parameters} = inputs
  parameter_weights(problem,weights,parameters)

def bias_cells(+problem: ConvSpec.Problem,inputs: Contract<problem>) -> Tree.Tree<Tile.Tile8>:
  Contract{input,weights,bias,parameters} = inputs
  parameter_bias(problem,bias,parameters)

def parameter_at(+count: Nat,+channels: Nat,+plane: Nat,+cells: Tree.Tree<Tile.Tile8>,+index: Nat,
  +inside: {Nat.is_lt(index,count) == True{} : Bool}) ->
  {TiledValues.cell_value(TiledValues.CellSource{cells,channels,plane},index) == ConvSpec.at(index,parameter_values(count,channels,plane,cells)) : F32}:
  Equal.sym(F32,ConvSpec.at(index,parameter_values(count,channels,plane,cells)),TiledValues.cell_value(TiledValues.CellSource{cells,channels,plane},index),
    Equal.trans(F32,ConvSpec.at(index,parameter_values(count,channels,plane,cells)),ListRead.at(~F32,0.0,index,parameter_values(count,channels,plane,cells)),
      TiledValues.cell_value(TiledValues.CellSource{cells,channels,plane},index),lookup_matches_readback(index,parameter_values(count,channels,plane,cells)),
      ListRead.enumerate_at(~F32,~TiledValues.CellSource,~TiledValues.cell_value,0.0,count,index,0n,TiledValues.CellSource{cells,channels,plane},inside)))
