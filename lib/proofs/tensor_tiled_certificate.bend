# Certificates of the cell conversions: every read hands the source array
# back unchanged, so the output keeps the depth it was allocated with.
import Base
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ../traversal_blocks.bend as Blocks
import ../tensor_tiled_cells.bend as Cells
import ./storage_tree_model.bend as Tree
import ./storage_certificate.bend as Certificate
import ./storage_witness.bend as Witness
import ./traversal_witness.bend as TraversalWitness
import ./traversal_certificate.bend as TraversalCertificate
import ./traversal_blocks_certificate.bend as BlocksCertificate
import ./kernel_tile_witness.bend as TileWitness
import ../traversal_array.bend as Traversal

def when_sample(-input: Array<F32>,valid: Bool,+index: U32,evidence: Witness.ArrayWitness<F32,input>) ->
  TraversalWitness.Sample<Array<F32>,F32,input,Storage.read_when(~F32,0.0,valid,input,index)>:
  match valid:
    case False{}: TraversalWitness.Sample{0.0,{==}}
    case True{}: TraversalWitness.array_sample(~F32,input,Array.get(F32,input,index),Witness.observed(F32,input,index,evidence))

def spot_sample(-input: Array<F32>,spot: Cells.Spot,evidence: Witness.ArrayWitness<F32,input>) ->
  TraversalWitness.Sample<Array<F32>,F32,input,Cells.read_spot(input,spot)>:
  Cells.Spot{+channel,+index,plane,+lanes} = spot
  when_sample(input,U32.is_lt(channel,lanes),index,evidence)

def lanes_sample(-input: Array<F32>,+spot: Cells.Spot,
  r0: TraversalWitness.Sample<Array<F32>,F32,input,Cells.read_spot(input,spot)>,
  r1: TraversalWitness.Sample<Array<F32>,F32,input,Cells.read_spot(input,Cells.next(spot))>,
  r2: TraversalWitness.Sample<Array<F32>,F32,input,Cells.read_spot(input,Cells.next(Cells.next(spot)))>,
  r3: TraversalWitness.Sample<Array<F32>,F32,input,Cells.read_spot(input,Cells.next(Cells.next(Cells.next(spot))))>,
  r4: TraversalWitness.Sample<Array<F32>,F32,input,Cells.read_spot(input,Cells.next(Cells.next(Cells.next(Cells.next(spot)))))>,
  r5: TraversalWitness.Sample<Array<F32>,F32,input,Cells.read_spot(input,Cells.next(Cells.next(Cells.next(Cells.next(Cells.next(spot))))))>,
  r6: TraversalWitness.Sample<Array<F32>,F32,input,Cells.read_spot(input,Cells.next(Cells.next(Cells.next(Cells.next(Cells.next(Cells.next(spot)))))))>,
  r7: TraversalWitness.Sample<Array<F32>,F32,input,Cells.read_spot(input,Cells.next(Cells.next(Cells.next(Cells.next(Cells.next(Cells.next(Cells.next(spot))))))))>) ->
  TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Cells.lane1(spot,Cells.read_spot(input,spot))>:
  match r0 r1 r2 r3 r4 r5 r6 r7:
    case TraversalWitness.Sample{a,ea} TraversalWitness.Sample{b,eb} TraversalWitness.Sample{c,ec} TraversalWitness.Sample{d,ed}
      TraversalWitness.Sample{e,ee} TraversalWitness.Sample{f,ef} TraversalWitness.Sample{g,eg} TraversalWitness.Sample{h,eh}:
      +s1 = Cells.next(spot)
      +s2 = Cells.next(s1)
      +s3 = Cells.next(s2)
      +s4 = Cells.next(s3)
      +s5 = Cells.next(s4)
      +s6 = Cells.next(s5)
      +s7 = Cells.next(s6)
      %Equal.sym(Array<F32> & F32,Cells.read_spot(input,spot),(input,a),ea) : TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Cells.lane1(spot,_)>
      %Equal.sym(Array<F32> & F32,Cells.read_spot(input,s1),(input,b),eb) : TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Cells.lane2(s1,a,_)>
      %Equal.sym(Array<F32> & F32,Cells.read_spot(input,s2),(input,c),ec) : TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Cells.lane3(s2,a,b,_)>
      %Equal.sym(Array<F32> & F32,Cells.read_spot(input,s3),(input,d),ed) : TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Cells.lane4(s3,a,b,c,_)>
      %Equal.sym(Array<F32> & F32,Cells.read_spot(input,s4),(input,e),ee) : TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Cells.lane5(s4,a,b,c,d,_)>
      %Equal.sym(Array<F32> & F32,Cells.read_spot(input,s5),(input,f),ef) : TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Cells.lane6(s5,a,b,c,d,e,_)>
      %Equal.sym(Array<F32> & F32,Cells.read_spot(input,s6),(input,g),eg) : TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Cells.lane7(s6,a,b,c,d,e,f,_)>
      %Equal.sym(Array<F32> & F32,Cells.read_spot(input,s7),(input,h),eh) : TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Cells.lane8(a,b,c,d,e,f,g,_)>
      TraversalWitness.Sample{Tile.Tile{a,b,c,d,e,f,g,h},{==}}

def cell_sample(-input: Array<F32>,run: Cells.Run,position: U32,witness: Witness.ArrayWitness<F32,input>) ->
  TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Cells.cell_read(input,run,position)>:
  +evidence = witness
  +spot = Cells.run_spot(run,position)
  +s1 = Cells.next(spot)
  +s2 = Cells.next(s1)
  +s3 = Cells.next(s2)
  +s4 = Cells.next(s3)
  +s5 = Cells.next(s4)
  +s6 = Cells.next(s5)
  lanes_sample(input,spot,spot_sample(input,spot,evidence),spot_sample(input,s1,evidence),spot_sample(input,s2,evidence),spot_sample(input,s3,evidence),
    spot_sample(input,s4,evidence),spot_sample(input,s5,evidence),spot_sample(input,s6,evidence),spot_sample(input,Cells.next(s6),evidence))

def outer_sample(-input: Array<F32>,context: Cells.Outer,position: U32,witness: Witness.ArrayWitness<F32,input>) ->
  TraversalWitness.Sample<Array<F32>,Tile.Tile8,input,Cells.outer_read(input,context,position)>:
  +evidence = witness
  +spot = Cells.outer_spot(context,position)
  +s1 = Cells.next(spot)
  +s2 = Cells.next(s1)
  +s3 = Cells.next(s2)
  +s4 = Cells.next(s3)
  +s5 = Cells.next(s4)
  +s6 = Cells.next(s5)
  lanes_sample(input,spot,spot_sample(input,spot,evidence),spot_sample(input,s1,evidence),spot_sample(input,s2,evidence),spot_sample(input,s3,evidence),
    spot_sample(input,s4,evidence),spot_sample(input,s5,evidence),spot_sample(input,s6,evidence),spot_sample(input,Cells.next(s6),evidence))

type Leaf is Data:
  Leaf{blocks: U32,context: Cells.Tiling}

def tiled_cells(leaf: Leaf,input: Array<F32>,output: Array<Tile.Tile8>) -> Array<F32> & Array<Tile.Tile8>:
  Leaf{+blocks,context} = leaf
  Cells.tiled(blocks,context,input,output)

def tiled_preserves(+depth: Nat,leaf: Leaf,-input: Array<F32>,-output: Array<Tile.Tile8>,input_certificate: Certificate.Certificate<F32,input>,
  equation: {output == Tree.pack(Tile.Tile8,Certificate.normalized(Tile.Tile8,depth,Tree.reflect(Tile.Tile8,output))) : Array<Tile.Tile8>}) ->
  TraversalWitness.ResultEvidence<Array<F32>,Tile.Tile8,input,depth,tiled_cells(leaf,input,output)>:
  Leaf{+blocks,context} = leaf
  Cells.Tiling{+after,+lanes,+groups,+first} = context
  BlocksCertificate.run(~Array<F32>,~Tile.Tile8,~Cells.Tiling,~Cells.Run,~TraversalWitness.array_evidence(~F32),~Cells.group_run,~Cells.cell_read,~Blocks.in_place(~Cells.Run),
    ~cell_sample,U32.to_nat(blocks),groups,Cells.Tiling{after,lanes,groups,first},input,output,Certificate.witness(F32,input,input_certificate),depth,equation)

# The cells of one piece with the groups outermost.
type Outer is Data:
  Outer{count: U32,context: Cells.Outer}

def outer_cells(leaf: Outer,input: Array<F32>,output: Array<Tile.Tile8>) -> Array<F32> & Array<Tile.Tile8>:
  Outer{+count,context} = leaf
  Cells.tiled_outermost(count,context,input,output)

def outer_preserves(+depth: Nat,leaf: Outer,-input: Array<F32>,-output: Array<Tile.Tile8>,input_certificate: Certificate.Certificate<F32,input>,
  equation: {output == Tree.pack(Tile.Tile8,Certificate.normalized(Tile.Tile8,depth,Tree.reflect(Tile.Tile8,output))) : Array<Tile.Tile8>}) ->
  TraversalWitness.ResultEvidence<Array<F32>,Tile.Tile8,input,depth,outer_cells(leaf,input,output)>:
  Outer{+count,context} = leaf
  TraversalCertificate.completed(~Array<F32>,~Tile.Tile8,~Cells.Outer,~TraversalWitness.array_evidence(~F32),
    ~Traversal.run(~Array<F32>,~Tile.Tile8,~Cells.Outer,~Cells.outer_read,~Blocks.in_place(~Cells.Outer)),
    ~TraversalWitness.run(~Array<F32>,~Tile.Tile8,~Cells.Outer,~TraversalWitness.array_evidence(~F32),~Cells.outer_read,~Blocks.in_place(~Cells.Outer),~outer_sample),
    U32.to_nat(count),0,context,input,output,Certificate.witness(F32,input,input_certificate),depth,equation)

def tiled_certificates(+blocks: U32,+context: Cells.Tiling,-input: Array<F32>,-output: Array<Tile.Tile8>,input_certificate: Certificate.Certificate<F32,input>,
  output_certificate: Certificate.Certificate<Tile.Tile8,output>) ->
  TraversalCertificate.Certificates<Array<F32>,Tile.Tile8,Certificate.family(~F32),Cells.tiled(blocks,context,input,output)>:
  Certificate.Certificate{+depth,+equation} = input_certificate
  TraversalCertificate.operation(~Array<F32>,~Tile.Tile8,~Leaf,~Certificate.family(~F32),~tiled_cells,~tiled_preserves,
    Leaf{blocks,context},input,output,Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},output_certificate)

def outermost_certificates(+count: U32,+context: Cells.Outer,-input: Array<F32>,-output: Array<Tile.Tile8>,input_certificate: Certificate.Certificate<F32,input>,
  output_certificate: Certificate.Certificate<Tile.Tile8,output>) ->
  TraversalCertificate.Certificates<Array<F32>,Tile.Tile8,Certificate.family(~F32),Cells.tiled_outermost(count,context,input,output)>:
  Certificate.Certificate{+depth,+equation} = input_certificate
  TraversalCertificate.operation(~Array<F32>,~Tile.Tile8,~Outer,~Certificate.family(~F32),~outer_cells,~outer_preserves,
    Outer{count,context},input,output,Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},output_certificate)
