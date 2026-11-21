import Base
import ./kernel_spatial_packing.bend as Packing
import ./kernel_tile.bend as Tile
import ./storage_buffer.bend as Storage
import ./proofs/kernel_packing_certificate.bend as PackingCertificate
import ./proofs/storage_certificate.bend as Certificate
import ./proofs/traversal_certificate.bend as TraversalCertificate

def buffers(~Source: Data,~Element: Data,source_length: Nat,output_length: Nat,result: Array<Source> & Array<Element>,
  certificates: TraversalCertificate.Certificates<Array<Source>,Element,Certificate.family(~Source),result>) ->
  Storage.Buffer<Source> & Storage.Buffer<Element>:
  (source,output) = result
  TraversalCertificate.Certificates{source_certificate,output_certificate} = certificates
  (Storage.Buffer{source,source_length,source_certificate},Storage.Buffer{output,output_length,output_certificate})

# The planes of a call from its input cells; the input comes back.
def transpose(+window: Packing.Stencil,+planes: Packing.Planes,input: Storage.Buffer<Tile.Tile8>,pairs: Storage.Buffer<Tile.Tile16>) ->
  Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile16>:
  Storage.Buffer{input,input_length,input_certificate} = input
  Storage.Buffer{pairs,pairs_length,pairs_certificate} = pairs
  buffers(~Tile.Tile8,~Tile.Tile16,input_length,pairs_length,Packing.transpose(window,planes,input,pairs),
    PackingCertificate.transpose_certificates(window,planes,input,pairs,input_certificate,pairs_certificate))

# count entries of the positions from start packed from the planes; the
# planes come back.
def pack(+window: Packing.Stencil,+planes: Packing.Planes,+count: Nat,+start: U32,pairs: Storage.Buffer<Tile.Tile16>,tiles: Storage.Buffer<Tile.Tile8>) ->
  Storage.Buffer<Tile.Tile16> & Storage.Buffer<Tile.Tile8>:
  Storage.Buffer{pairs,pairs_length,pairs_certificate} = pairs
  Storage.Buffer{tiles,tiles_length,tiles_certificate} = tiles
  buffers(~Tile.Tile16,~Tile.Tile8,pairs_length,tiles_length,Packing.pack(window,planes,count,start,pairs,tiles),
    PackingCertificate.pack_certificates(window,planes,count,start,pairs,tiles,pairs_certificate,tiles_certificate))
