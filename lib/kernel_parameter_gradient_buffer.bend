# Rebuild source owners once after the complete read-only native traversal.
import Base
import ./kernel_parameter_gradient.bend as Kernel
import ./storage_buffer.bend as Storage
import ./proofs/parameter_gradient_certificate.bend as Certificate
import ./proofs/storage_certificate.bend as StorageCertificate

def bias_kept(length: Nat,result: Array<F32> & Storage.Buffer<F32>,certificate: StorageCertificate.Certificate<F32,Certificate.source(~Array<F32>,result)>) ->
  Storage.Buffer<F32> & Storage.Buffer<F32>:
  (d,values) = result
  (Storage.Buffer{d,length,certificate},values)

# d is read; every output is written into result.
def bias(+work: Kernel.Work,d: Storage.Buffer<F32>,result: Storage.Buffer<F32>) -> Storage.Buffer<F32> & Storage.Buffer<F32>:
  Storage.Buffer{d,length,certificate} = d
  bias_kept(length,Kernel.bias(work,d,result),Certificate.bias(work,d,result,certificate))
