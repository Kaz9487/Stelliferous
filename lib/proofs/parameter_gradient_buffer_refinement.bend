# Rebuilding erased certificates changes no written output.
import Base
import ../kernel_parameter_gradient.bend as Kernel
import ../kernel_parameter_gradient_buffer.bend as Buffers
import ../storage_buffer.bend as Storage
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./parameter_gradient_refinement.bend as Model
import ./parameter_gradient_certificate.bend as Certificate
import ./storage_certificate.bend as Flat

def written(result: Storage.Buffer<F32> & Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  (d,values) = result
  values

def kernel_written(result: Array<F32> & Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  (d,values) = result
  values

def bias_kept(+length: Nat,result: Array<F32> & Storage.Buffer<F32>,certificate: Flat.Certificate<F32,Certificate.source(~Array<F32>,result)>) ->
  {written(Buffers.bias_kept(length,result,certificate)) == kernel_written(result) : Storage.Buffer<F32>}:
  (d,values) = result
  {==}

def bias(+work: Kernel.Work,+d: Buffer.Model<F32>,+result: Buffer.Model<F32>) ->
  {Buffer.contents(~F32,written(Buffers.bias(work,Buffer.pack(~F32,d),Buffer.pack(~F32,result)))) == Buffer.content(~F32,Model.bias_written(work,Buffer.tree(~F32,d),result)) :
    Buffer.Content<F32>}:
  Buffer.Model{+dt,+dl,+dc} = d
  +values = Model.bias_written(work,dt,result)
  %Equal.sym(Storage.Buffer<F32>,written(Buffers.bias_kept(dl,Kernel.bias(work,Tree.pack(F32,dt),Buffer.pack(~F32,result)),
      Certificate.bias(work,Tree.pack(F32,dt),Buffer.pack(~F32,result),dc))),
    kernel_written(Kernel.bias(work,Tree.pack(F32,dt),Buffer.pack(~F32,result))),
    bias_kept(dl,Kernel.bias(work,Tree.pack(F32,dt),Buffer.pack(~F32,result)),Certificate.bias(work,Tree.pack(F32,dt),Buffer.pack(~F32,result),dc))) :
    {Buffer.contents(~F32,_) == Buffer.content(~F32,values) : Buffer.Content<F32>}
  %Equal.sym(Array<F32> & Storage.Buffer<F32>,Kernel.bias(work,Tree.pack(F32,dt),Buffer.pack(~F32,result)),(Tree.pack(F32,dt),Buffer.pack(~F32,values)),
    Model.bias(work,dt,result)) :
    {Buffer.contents(~F32,kernel_written(_)) == Buffer.content(~F32,values) : Buffer.Content<F32>}
  Buffer.packed_contents(~F32,values)
