import Base
import ../kernel_spatial_packing.bend as Packing
import ../kernel_tile.bend as Tile
import ./kernel_packing_witness.bend as PackingWitness
import ./storage_tree_model.bend as Tree
import ./storage_certificate.bend as Certificate
import ./traversal_witness.bend as TraversalWitness
import ./traversal_certificate.bend as TraversalCertificate

# The planes of a call and the packed positions of a block.
type PlanesCall is Data:
  PlanesCall{window: Packing.Stencil,planes: Packing.Planes}

type PackCall is Data:
  PackCall{window: Packing.Stencil,planes: Packing.Planes,count: Nat,start: U32}

def transpose_operation(call: PlanesCall,input: Array<Tile.Tile8>,pairs: Array<Tile.Tile16>) -> Array<Tile.Tile8> & Array<Tile.Tile16>:
  PlanesCall{window,planes} = call
  Packing.transpose(window,planes,input,pairs)

def transpose_preserves(+depth: Nat,call: PlanesCall,-input: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,input_certificate: Certificate.Certificate<Tile.Tile8,input>,
  equation: {pairs == Tree.pack(Tile.Tile16,Certificate.normalized(Tile.Tile16,depth,Tree.reflect(Tile.Tile16,pairs))) : Array<Tile.Tile16>}) ->
  TraversalWitness.ResultEvidence<Array<Tile.Tile8>,Tile.Tile16,input,depth,transpose_operation(call,input,pairs)>:
  PlanesCall{window,planes} = call
  PackingWitness.transpose(depth,window,planes,input,pairs,Certificate.witness(Tile.Tile8,input,input_certificate),Certificate.witness_at(Tile.Tile16,depth,pairs,equation))

def transpose_certificates(window: Packing.Stencil,planes: Packing.Planes,-input: Array<Tile.Tile8>,-pairs: Array<Tile.Tile16>,
  input_certificate: Certificate.Certificate<Tile.Tile8,input>,pairs_certificate: Certificate.Certificate<Tile.Tile16,pairs>) ->
  TraversalCertificate.Certificates<Array<Tile.Tile8>,Tile.Tile16,Certificate.family(~Tile.Tile8),Packing.transpose(window,planes,input,pairs)>:
  Certificate.Certificate{+depth,+equation} = input_certificate
  TraversalCertificate.operation(~Array<Tile.Tile8>,~Tile.Tile16,~PlanesCall,~Certificate.family(~Tile.Tile8),~transpose_operation,~transpose_preserves,
    PlanesCall{window,planes},input,pairs,Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},pairs_certificate)

def pack_operation(call: PackCall,pairs: Array<Tile.Tile16>,tiles: Array<Tile.Tile8>) -> Array<Tile.Tile16> & Array<Tile.Tile8>:
  PackCall{window,planes,count,start} = call
  Packing.pack(window,planes,count,start,pairs,tiles)

def pack_preserves(+depth: Nat,call: PackCall,-pairs: Array<Tile.Tile16>,-tiles: Array<Tile.Tile8>,pairs_certificate: Certificate.Certificate<Tile.Tile16,pairs>,
  equation: {tiles == Tree.pack(Tile.Tile8,Certificate.normalized(Tile.Tile8,depth,Tree.reflect(Tile.Tile8,tiles))) : Array<Tile.Tile8>}) ->
  TraversalWitness.ResultEvidence<Array<Tile.Tile16>,Tile.Tile8,pairs,depth,pack_operation(call,pairs,tiles)>:
  PackCall{window,planes,count,start} = call
  PackingWitness.pack(depth,window,planes,count,start,pairs,tiles,Certificate.witness(Tile.Tile16,pairs,pairs_certificate),Certificate.witness_at(Tile.Tile8,depth,tiles,equation))

def pack_certificates(window: Packing.Stencil,planes: Packing.Planes,count: Nat,start: U32,-pairs: Array<Tile.Tile16>,-tiles: Array<Tile.Tile8>,
  pairs_certificate: Certificate.Certificate<Tile.Tile16,pairs>,tiles_certificate: Certificate.Certificate<Tile.Tile8,tiles>) ->
  TraversalCertificate.Certificates<Array<Tile.Tile16>,Tile.Tile8,Certificate.family(~Tile.Tile16),Packing.pack(window,planes,count,start,pairs,tiles)>:
  Certificate.Certificate{+depth,+equation} = pairs_certificate
  TraversalCertificate.operation(~Array<Tile.Tile16>,~Tile.Tile8,~PackCall,~Certificate.family(~Tile.Tile16),~pack_operation,~pack_preserves,
    PackCall{window,planes,count,start},pairs,tiles,Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},tiles_certificate)
