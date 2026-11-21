# The indexed map of cells reads its source and writes only its output: the
# source comes back unchanged and the output keeps its certificate.
import Base
import ../kernel_tile.bend as Tile
import ../tensor_tiled_indexed.bend as TiledIndexed
import ../traversal_array.bend as Traversal
import ./storage_tree_model.bend as Tree
import ./storage_certificate.bend as Certificate
import ./storage_witness.bend as Witness
import ./traversal_witness.bend as TraversalWitness
import ./traversal_certificate.bend as TraversalCertificate
import ../traversal_blocks.bend as TraversalBlocks

def sample(~operation: U32 -> F32 -> F32,-input: Array<Tile.Tile8>,+context: TiledIndexed.Indexing,+position: U32,-result: Array<Tile.Tile8> & Tile.Tile8,
  evidence: TraversalWitness.Sample<Array<Tile.Tile8>,Tile.Tile8,input,result>) ->
  TraversalWitness.Sample<Array<Tile.Tile8>,Tile.Tile8,input,TiledIndexed.cell(~operation,context,position,result)>:
  match evidence:
    case TraversalWitness.Sample{value,equation}:
      %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,result,(input,value),equation) :
        TraversalWitness.Sample<Array<Tile.Tile8>,Tile.Tile8,input,TiledIndexed.cell(~operation,context,position,_)>
      TraversalWitness.Sample{TiledIndexed.lanes(~operation,TiledIndexed.lane_index(context,position),TiledIndexed.lane_step(context),value),{==}}

def read(~operation: U32 -> F32 -> F32,-input: Array<Tile.Tile8>,context: TiledIndexed.Indexing,position: U32,evidence: Witness.ArrayWitness<Tile.Tile8,input>) ->
  TraversalWitness.Sample<Array<Tile.Tile8>,Tile.Tile8,input,TiledIndexed.read(~operation,input,context,position)>:
  +position = position
  sample(~operation,input,context,position,Array.get(Tile.Tile8,input,position),
    TraversalWitness.array_sample(~Tile.Tile8,input,Array.get(Tile.Tile8,input,position),Witness.observed(Tile.Tile8,input,position,evidence)))

def preserves(~operation: U32 -> F32 -> F32,+depth: Nat,call: TiledIndexed.Call,-input: Array<Tile.Tile8>,-output: Array<Tile.Tile8>,input_certificate: Certificate.Certificate<Tile.Tile8,input>,
  equation: {output == Tree.pack(Tile.Tile8,Certificate.normalized(Tile.Tile8,depth,Tree.reflect(Tile.Tile8,output))) : Array<Tile.Tile8>}) ->
  TraversalWitness.ResultEvidence<Array<Tile.Tile8>,Tile.Tile8,input,depth,TiledIndexed.run(~operation,call,input,output)>:
  TiledIndexed.Call{count,context} = call
  TraversalCertificate.completed(~Array<Tile.Tile8>,~Tile.Tile8,~TiledIndexed.Indexing,~TraversalWitness.array_evidence(~Tile.Tile8),
    ~Traversal.run(~Array<Tile.Tile8>,~Tile.Tile8,~TiledIndexed.Indexing,~TiledIndexed.read(~operation),~TraversalBlocks.in_place(~TiledIndexed.Indexing)),
    ~TraversalWitness.run(~Array<Tile.Tile8>,~Tile.Tile8,~TiledIndexed.Indexing,~TraversalWitness.array_evidence(~Tile.Tile8),~TiledIndexed.read(~operation),
      ~TraversalBlocks.in_place(~TiledIndexed.Indexing),~read(~operation)),
    count,0,context,input,output,Certificate.witness(Tile.Tile8,input,input_certificate),depth,equation)

def certificates(~operation: U32 -> F32 -> F32,call: TiledIndexed.Call,-input: Array<Tile.Tile8>,-output: Array<Tile.Tile8>,input_certificate: Certificate.Certificate<Tile.Tile8,input>,
  output_certificate: Certificate.Certificate<Tile.Tile8,output>) ->
  TraversalCertificate.Certificates<Array<Tile.Tile8>,Tile.Tile8,Certificate.family(~Tile.Tile8),TiledIndexed.run(~operation,call,input,output)>:
  Certificate.Certificate{+depth,+equation} = input_certificate
  TraversalCertificate.operation(~Array<Tile.Tile8>,~Tile.Tile8,~TiledIndexed.Call,~Certificate.family(~Tile.Tile8),~TiledIndexed.run(~operation),~preserves(~operation),
    call,input,output,Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},output_certificate)
