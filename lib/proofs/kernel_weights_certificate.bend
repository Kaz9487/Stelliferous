import Base
import ../kernel_weights.bend as Weights
import ../kernel_tile.bend as Tile
import ../traversal_array.bend as Traversal
import ../storage_allocation.bend as Allocation
import ../tensor.bend as Tensors
import ./kernel_weights_witness.bend as WeightsWitness
import ./traversal_witness.bend as TraversalWitness
import ./storage_witness.bend as Witness
import ./storage_certificate.bend as Certificate
import ./storage_allocation_refinement.bend as AllocationLaws
import ./storage_tree_model.bend as Tree

def family(-input: Array<F32>) -> Data:
  Witness.ArrayWitness<F32,input>

def block_mapped_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Weights.Bases,+depth: Nat,+block: U32,+layout: Weights.MappedLayout<Context>,-input: Array<F32>,-owners: Array<F32> & Array<Tile.Tile8>,
  input_evidence: Witness.ArrayWitness<F32,input>,evidence: TraversalWitness.ResultEvidence<Array<F32>,Tile.Tile8,input,depth,owners>) ->
  TraversalWitness.ResultEvidence<Array<F32>,Tile.Tile8,input,depth,Weights.copy_mapped_with(~Context,~offset,~row_bases,block,layout,owners)>:
  Weights.MappedLayout{+rows,+inner,+row_stride,+context} = layout
  TraversalWitness.ResultEvidence{output,equation,output_evidence} = evidence
  %Equal.sym(Array<F32> & Array<Tile.Tile8>,owners,(input,output),equation) :
    TraversalWitness.ResultEvidence<Array<F32>,Tile.Tile8,input,depth,Weights.copy_mapped_with(~Context,~offset,~row_bases,block,Weights.MappedLayout{rows,inner,row_stride,context},_)>
  +mapped = {Weights.MappedRows{rows,(block * 8 : U32),(block * inner : U32),context,row_bases(context,row_stride,(block * 8 : U32))} : Weights.MappedRows<Context>}
  -state = {Traversal.State{input,output,0,mapped} : Traversal.State<&1,&1,Array<F32>,Array<Tile.Tile8>,Weights.MappedRows<Context>>}
  TraversalWitness.finish(~Array<F32>,~Tile.Tile8,~Weights.MappedRows<Context>,input,depth,
    Traversal.run(~Array<F32>,~Tile.Tile8,~Weights.MappedRows<Context>,~Weights.read_mapped_rows(~Context,~offset),~Weights.destination_rows(~Context),U32.to_nat(inner),state),
    TraversalWitness.run(~Array<F32>,~Tile.Tile8,~Weights.MappedRows<Context>,~family,~Weights.read_mapped_rows(~Context,~offset),~Weights.destination_rows(~Context),~WeightsWitness.read_mapped_rows(~Context,~offset),
      depth,U32.to_nat(inner),input,state,input_evidence,TraversalWitness.StateEvidence{output,0,mapped,{==},output_evidence}))

def traverse_mapped_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Weights.Bases,+depth: Nat,count: Nat,+index: U32,+layout: Weights.MappedLayout<Context>,-input: Array<F32>,-owners: Array<F32> & Array<Tile.Tile8>,
  +input_evidence: Witness.ArrayWitness<F32,input>,evidence: TraversalWitness.ResultEvidence<Array<F32>,Tile.Tile8,input,depth,owners>) ->
  TraversalWitness.ResultEvidence<Array<F32>,Tile.Tile8,input,depth,
    Traversal.indexed_fold(~(Array<F32> & Array<Tile.Tile8>),~Weights.MappedLayout<Context>,~Weights.copy_mapped_with(~Context,~offset,~row_bases),count,index,layout,owners)>:
  match count:
    case 0n: evidence
    case 1n+rest:
      traverse_mapped_with(~Context,~offset,~row_bases,depth,rest,(index + 1 : U32),layout,input,Weights.copy_mapped_with(~Context,~offset,~row_bases,index,layout,owners),input_evidence,
        block_mapped_with(~Context,~offset,~row_bases,depth,index,layout,input,owners,input_evidence,evidence))

def finish(-depth: Nat,-input: Array<F32>,-owners: Array<F32> & Array<Tile.Tile8>,
  evidence: TraversalWitness.ResultEvidence<Array<F32>,Tile.Tile8,input,depth,owners>) -> Witness.Witness(Tile.Tile8,depth,Weights.reordered(owners)):
  TraversalWitness.ResultEvidence{output,equation,output_evidence} = evidence
  %Equal.sym(Array<F32> & Array<Tile.Tile8>,owners,(input,output),equation) : Witness.Witness(Tile.Tile8,depth,Weights.reordered(_))
  output_evidence

def allocated(+depth: Nat) -> Witness.Witness(Tile.Tile8,depth,Allocation.array(~Tile.Tile8,depth,Tile.filled(0.0))):
  %Equal.sym(Array<Tile.Tile8>,Allocation.array(~Tile.Tile8,depth,Tile.filled(0.0)),Array.new(Tile.Tile8,depth,Tile.filled(0.0)),
    AllocationLaws.array_same(Tile.Tile8,depth,Tile.filled(0.0))) : Witness.Witness(Tile.Tile8,depth,_)
  Witness.fresh(Tile.Tile8,depth,Tile.filled(0.0))

def run_mapped_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Weights.Bases,+rows: U32,+inner: U32,+row_stride: U32,+context: Context,-input: Array<F32>,evidence: Witness.ArrayWitness<F32,input>) ->
  TraversalWitness.ResultEvidence<Array<F32>,Tile.Tile8,input,Tensors.depth_for((((rows + 7) / 8) * inner : U32)),Weights.run_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,input)>:
  +blocks = ((rows + 7) / 8 : U32)
  +depth = Tensors.depth_for((blocks * inner : U32))
  +layout = {Weights.MappedLayout{rows,inner,row_stride,context} : Weights.MappedLayout<Context>}
  -output = Allocation.array(~Tile.Tile8,depth,Tile.filled(0.0))
  traverse_mapped_with(~Context,~offset,~row_bases,depth,U32.to_nat(blocks),0,layout,input,(input,output),evidence,TraversalWitness.ResultEvidence{output,{==},allocated(depth)})

def run_mapped(~Context: Data,~offset: Context -> U32 -> U32,+rows: U32,+inner: U32,+row_stride: U32,+context: Context,-input: Array<F32>,evidence: Witness.ArrayWitness<F32,input>) ->
  TraversalWitness.ResultEvidence<Array<F32>,Tile.Tile8,input,Tensors.depth_for((((rows + 7) / 8) * inner : U32)),Weights.run_mapped(~Context,~offset,rows,inner,row_stride,context,input)>:
  run_mapped_with(~Context,~offset,~Weights.linear_bases(~Context),rows,inner,row_stride,context,input,evidence)

def run(+rows: U32,+inner: U32,-input: Array<F32>,evidence: Witness.ArrayWitness<F32,input>) ->
  Witness.Witness(Tile.Tile8,Tensors.depth_for((((rows + 7) / 8) * inner : U32)),Weights.run(rows,inner,input)):
  finish(Tensors.depth_for((((rows + 7) / 8) * inner : U32)),input,Weights.run_mapped(~Unit,~Weights.identity,rows,inner,inner,Unit{},input),
    run_mapped(~Unit,~Weights.identity,rows,inner,inner,Unit{},input,evidence))

def equation(+rows: U32,+inner: U32,-input: Array<F32>,certificate: Certificate.Certificate<F32,input>) ->
  {Weights.run(rows,inner,input) == Tree.pack(Tile.Tile8,Certificate.normalized(Tile.Tile8,Tensors.depth_for((((rows + 7) / 8) * inner : U32)),
    Tree.reflect(Tile.Tile8,Weights.run(rows,inner,input)))) : Array<Tile.Tile8>}:
  %Certificate.restored(Tile.Tile8,Tensors.depth_for((((rows + 7) / 8) * inner : U32)),Weights.run(rows,inner,input),
    run(rows,inner,input,Certificate.witness(F32,input,certificate))) :
    {_ == Tree.pack(Tile.Tile8,Certificate.normalized(Tile.Tile8,Tensors.depth_for((((rows + 7) / 8) * inner : U32)),
      Tree.reflect(Tile.Tile8,Weights.run(rows,inner,input)))) : Array<Tile.Tile8>}
  {==}

def certificate(+rows: U32,+inner: U32,-input: Array<F32>,certificate: Certificate.Certificate<F32,input>) ->
  Certificate.Certificate<Tile.Tile8,Weights.run(rows,inner,input)>:
  Certificate.Certificate{Tensors.depth_for((((rows + 7) / 8) * inner : U32)),equation(rows,inner,input,certificate)}

# The witness traversal occurs only beneath erased owner equations.
def source_at(+depth: Nat,-input: Array<F32>,-result: Array<F32> & Array<Tile.Tile8>,evidence: TraversalWitness.ResultEvidence<Array<F32>,Tile.Tile8,input,depth,result>) ->
  {Weights.source(result) == input : Array<F32>}:
  TraversalWitness.ResultEvidence{output,equation,output_evidence} = evidence
  Equal.cong(Array<F32> & Array<Tile.Tile8>,Array<F32>,pair => Weights.source(pair),result,(input,output),equation)

def input_equation_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Weights.Bases,+rows: U32,+inner: U32,+row_stride: U32,+context: Context,-input: Array<F32>,+depth: Nat,
  equation: {input == Tree.pack(F32,Certificate.normalized(F32,depth,Tree.reflect(F32,input))) : Array<F32>}) ->
  {input == Weights.source(Weights.run_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,input)) : Array<F32>}:
  %Equal.sym(Array<F32>,Weights.source(Weights.run_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,input)),input,
    source_at(Tensors.depth_for((((rows + 7) / 8) * inner : U32)),input,Weights.run_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,input),
      run_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,input,Witness.ArrayWitness{depth,Certificate.witness_at(F32,depth,input,equation)}))) : {input == _ : Array<F32>}
  {==}

def retained(-input: Array<F32>,-output: Array<F32>,+depth: Nat,
  equation: {input == Tree.pack(F32,Certificate.normalized(F32,depth,Tree.reflect(F32,input))) : Array<F32>},same: {input == output : Array<F32>}) -> Certificate.Certificate<F32,output>:
  %same : Certificate.Certificate<F32,_>
  Certificate.Certificate{depth,equation}

def equation_mapped_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Weights.Bases,+rows: U32,+inner: U32,+row_stride: U32,+context: Context,-input: Array<F32>,+depth: Nat,
  equation: {input == Tree.pack(F32,Certificate.normalized(F32,depth,Tree.reflect(F32,input))) : Array<F32>}) ->
  {Weights.reordered(Weights.run_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,input)) == Tree.pack(Tile.Tile8,Certificate.normalized(Tile.Tile8,Tensors.depth_for((((rows + 7) / 8) * inner : U32)),
    Tree.reflect(Tile.Tile8,Weights.reordered(Weights.run_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,input))))) : Array<Tile.Tile8>}:
  %Certificate.restored(Tile.Tile8,Tensors.depth_for((((rows + 7) / 8) * inner : U32)),Weights.reordered(Weights.run_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,input)),
    finish(Tensors.depth_for((((rows + 7) / 8) * inner : U32)),input,Weights.run_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,input),
      run_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,input,Witness.ArrayWitness{depth,Certificate.witness_at(F32,depth,input,equation)}))) :
    {_ == Tree.pack(Tile.Tile8,Certificate.normalized(Tile.Tile8,Tensors.depth_for((((rows + 7) / 8) * inner : U32)),
      Tree.reflect(Tile.Tile8,Weights.reordered(Weights.run_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,input))))) : Array<Tile.Tile8>}
  {==}

type MappedCertificates<-result: Array<F32> & Array<Tile.Tile8>> is Data:
  MappedCertificates{source: Certificate.Certificate<F32,Weights.source(result)>,output: Certificate.Certificate<Tile.Tile8,Weights.reordered(result)>}

def certificates_mapped_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Weights.Bases,+rows: U32,+inner: U32,+row_stride: U32,+context: Context,-input: Array<F32>,certificate: Certificate.Certificate<F32,input>) ->
  MappedCertificates<Weights.run_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,input)>:
  Certificate.Certificate{+depth,+equation} = certificate
  MappedCertificates{retained(input,Weights.source(Weights.run_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,input)),depth,equation,
      input_equation_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,input,depth,equation)),
    Certificate.Certificate{Tensors.depth_for((((rows + 7) / 8) * inner : U32)),equation_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,input,depth,equation)}}

def certificates_mapped(~Context: Data,~offset: Context -> U32 -> U32,+rows: U32,+inner: U32,+row_stride: U32,+context: Context,-input: Array<F32>,certificate: Certificate.Certificate<F32,input>) ->
  MappedCertificates<Weights.run_mapped(~Context,~offset,rows,inner,row_stride,context,input)>:
  certificates_mapped_with(~Context,~offset,~Weights.linear_bases(~Context),rows,inner,row_stride,context,input,certificate)
