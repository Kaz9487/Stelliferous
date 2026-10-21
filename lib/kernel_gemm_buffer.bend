import Base
import ./kernel_tile.bend as Tile
import ./kernel_gemm.bend as Gemm
import ./storage_buffer.bend as Storage
import ./proofs/kernel_gemm_certificate.bend as GemmCertificate
import ./kernel_weights.bend as Weights
import ./proofs/kernel_weights_certificate.bend as WeightCertificate

# The live certificate is updated once, outside every output and MAC loop.
def run(+rows: U32,input: Storage.Buffer<Tile.Tile8>,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,
  output: Storage.Buffer<F32>,+columns: U32,+inner: U32) -> Storage.Buffer<F32>:
  Storage.Buffer{input,input_length,input_certificate} = input
  Storage.Buffer{weights,weight_length,weight_certificate} = weights
  Storage.Buffer{bias,bias_length,bias_certificate} = bias
  Storage.Buffer{output,length,certificate} = output
  Storage.Buffer{Gemm.run(~True{},rows,input,weights,bias,output,columns,inner),length,
    GemmCertificate.certificate(rows,input,weights,bias,output,columns,inner,0,columns,input_certificate,weight_certificate,bias_certificate,certificate)}

# The tiles, the weights and the bias come back with the output, for a caller
# that uses them again.
type Kept is Type:
  Kept{input: Storage.Buffer<Tile.Tile8>,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,output: Storage.Buffer<F32>}

def kept(state: Gemm.Buffers,input_length: Nat,weight_length: Nat,bias_length: Nat,length: Nat,certificates: GemmCertificate.Kept<state>) -> Kept:
  Gemm.Buffers{input,weights,bias,output} = state
  GemmCertificate.Kept{input_certificate,weight_certificate,bias_certificate,certificate} = certificates
  Kept{Storage.Buffer{input,input_length,input_certificate},Storage.Buffer{weights,weight_length,weight_certificate},
    Storage.Buffer{bias,bias_length,bias_certificate},Storage.Buffer{output,length,certificate}}

def run_keeping(+rows: U32,input: Storage.Buffer<Tile.Tile8>,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,
  output: Storage.Buffer<F32>,+columns: U32,+inner: U32,+start: U32,+output_columns: U32) -> Kept:
  Storage.Buffer{input,input_length,input_certificate} = input
  Storage.Buffer{weights,weight_length,weight_certificate} = weights
  Storage.Buffer{bias,bias_length,bias_certificate} = bias
  Storage.Buffer{output,length,certificate} = output
  kept(Gemm.run_keeping(~True{},rows,input,weights,bias,output,columns,inner,start,output_columns),input_length,weight_length,bias_length,length,
    GemmCertificate.kept(rows,input,weights,bias,output,columns,inner,start,output_columns,input_certificate,weight_certificate,bias_certificate,certificate))

def reorder(+rows: U32,+inner: U32,weights: Storage.Buffer<F32>) -> Storage.Buffer<Tile.Tile8>:
  Storage.Buffer{weights,length,certificate} = weights
  Storage.Buffer{Weights.run(rows,inner,weights),U32.to_nat((((rows + 7) / 8) * inner : U32)),
    WeightCertificate.certificate(rows,inner,weights,certificate)}

# Mapping runs once for each packed tile; both affine owners return.
def mapped_buffers(+input_length: Nat,+output_length: Nat,result: Array<F32> & Array<Tile.Tile8>,
  certificates: WeightCertificate.MappedCertificates<result>) -> Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>:
  (input,output) = result
  WeightCertificate.MappedCertificates{input_certificate,output_certificate} = certificates
  (Storage.Buffer{input,input_length,input_certificate},Storage.Buffer{output,output_length,output_certificate})

def reorder_mapped(~Context: Data,~offset: Context -> U32 -> U32,+rows: U32,+inner: U32,+row_stride: U32,+context: Context,weights: Storage.Buffer<F32>) -> Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>:
  Storage.Buffer{weights,+length,certificate} = weights
  mapped_buffers(length,U32.to_nat((((rows + 7) / 8) * inner : U32)),Weights.run_mapped(~Context,~offset,rows,inner,row_stride,context,weights),
    WeightCertificate.certificates_mapped(~Context,~offset,rows,inner,row_stride,context,weights,certificate))


def reorder_mapped_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Weights.Bases,+rows: U32,+inner: U32,+row_stride: U32,+context: Context,weights: Storage.Buffer<F32>) -> Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>:
  Storage.Buffer{weights,+length,certificate} = weights
  mapped_buffers(length,U32.to_nat((((rows + 7) / 8) * inner : U32)),Weights.run_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,weights),
    WeightCertificate.certificates_mapped_with(~Context,~offset,~row_bases,rows,inner,row_stride,context,weights,certificate))
