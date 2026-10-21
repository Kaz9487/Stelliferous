# The compensated GELU mapper, with one scalar tail and two eight-lane stages.
import Base
import ./math_gelu_lanes.bend as Gelu
import ./traversal_lanes.bend as Lanes
import ./storage_buffer.bend as Storage
import ./proofs/storage_certificate.bend as Certificate
import ./proofs/math_gelu_lanes_certificate.bend as GeluCertificate

def mapped(+count: Nat,+position: U32,buffer: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  Storage.Buffer{values,length,Certificate.Certificate{+depth,equation}} = buffer
  Storage.Buffer{Gelu.map_when(Lanes.covered(depth,position,count),count,position,values),length,
    GeluCertificate.map_when(Lanes.covered(depth,position,count),count,position,values,Certificate.Certificate{depth,equation})}
