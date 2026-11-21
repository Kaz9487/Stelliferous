# Internal contiguous ranges. Callers establish bounds; the shared copy proof
# and storage certificates cover the actual loop without a per-element check.
import Base
import ./storage_buffer.bend as Storage
import ./tensor_view.bend as Views
import ./tensor_access.bend as Access
import ./tensor_operations.bend as Operations
import ./proofs/tensor_access_certificate.bend as AccessCertificate

# count elements of a view, in order, to destination onward. fast copies a
# contiguous view as one run.
def view(~Element: Data,+fast: Bool,+count: U32,+source: Views.View,+destination: U32,input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) ->
  Storage.Buffer<Element> & Storage.Buffer<Element>:
  Storage.Buffer{input,input_length,input_certificate} = input
  Storage.Buffer{output,output_length,output_certificate} = output
  Operations.pair_buffers(~Element,input_length,output_length,
    Access.copy_run(~Element,fast,U32.to_nat(count),source,destination,input,output),
    AccessCertificate.copy(~Element,fast,U32.to_nat(count),source,destination,input,output,input_certificate,output_certificate))

def range(~Element: Data,+count: U32,+source: U32,+destination: U32,input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) ->
  Storage.Buffer<Element> & Storage.Buffer<Element>:
  view(~Element,True{},count,Views.View{[],source},destination,input,output)
