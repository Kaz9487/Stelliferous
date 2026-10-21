# The read-only traversal of the kernel preserves its native source.
# Certificates are transported once at this boundary; every model traversal
# is erased.
import Base
import ../kernel_parameter_gradient.bend as Kernel
import ../traversal_grouped.bend as Grouped
import ../storage_buffer.bend as Storage
import ./storage_certificate.bend as Certificate
import ./parameter_gradient_source_witness.bend as SourceWitness
import ./traversal_witness.bend as Witness

def source(~Reader: Type,result: Reader & Storage.Buffer<F32>) -> Reader:
  (reader,values) = result
  reader

def stored_source(~Reader: Type,~read: Reader -> Kernel.Context -> Nat -> Reader & F32,rest: Nat,+index: Nat,+position: U32,+work: Kernel.Work,+ranges: List<&2,Grouped.Range>,
  -reader: Reader,-result: Storage.Buffer<F32>,-pair: Reader & F32,sample: Witness.Sample<Reader,F32,reader,pair>,
  tail: @-value: F32 -> {source(~Reader,Kernel.written(~Reader,~read,rest,1n+index,(position + 1 : U32),work,ranges,reader,Storage.write(~F32,position,value,result))) == reader : Reader}) ->
  {source(~Reader,Kernel.stored(~Reader,position,result,pair,
    reader => result => Kernel.written(~Reader,~read,rest,1n+index,(position + 1 : U32),work,ranges,reader,result))) == reader : Reader}:
  Witness.Sample{value,equation} = sample
  %Equal.sym(Reader & F32,pair,(reader,value),equation) :
    {source(~Reader,Kernel.stored(~Reader,position,result,_,
      reader => result => Kernel.written(~Reader,~read,rest,1n+index,(position + 1 : U32),work,ranges,reader,result))) == reader : Reader}
  tail(value)

def written_source(~Reader: Type,~Evidence: @-reader: Reader -> Data,~read: Reader -> Kernel.Context -> Nat -> Reader & F32,
  ~preserves: @-reader: Reader -> @context: Kernel.Context -> @position: Nat -> Evidence(reader) -> Witness.Sample<Reader,F32,reader,read(reader,context,position)>,
  remaining: Nat,+index: Nat,+position: U32,+work: Kernel.Work,+ranges: List<&2,Grouped.Range>,-reader: Reader,-result: Storage.Buffer<F32>,+evidence: Evidence(reader)) ->
  {source(~Reader,Kernel.written(~Reader,~read,remaining,index,position,work,ranges,reader,result)) == reader : Reader}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      stored_source(~Reader,~read,rest,index,position,work,ranges,reader,result,Kernel.element(~Reader,~read,work,index,ranges,reader),
        SourceWitness.element(~Reader,~Evidence,~read,~preserves,work,index,ranges,reader,evidence),
        value => written_source(~Reader,~Evidence,~read,~preserves,rest,1n+index,(position + 1 : U32),work,ranges,reader,Storage.write(~F32,position,value,result),evidence))

def bias_owner(+work: Kernel.Work,-d: Array<F32>,-result: Storage.Buffer<F32>,certificate: Certificate.Certificate<F32,d>) ->
  {source(~Array<F32>,Kernel.bias(work,d,result)) == d : Array<F32>}:
  Kernel.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  written_source(~Array<F32>,~SourceWitness.bias_evidence,~Kernel.bias_at,~SourceWitness.bias_cell,o,0n,0,work,Grouped.ranges(rows,columns,g),d,result,
    Certificate.witness(F32,d,certificate))

def bias(+work: Kernel.Work,-d: Array<F32>,-result: Storage.Buffer<F32>,+certificate: Certificate.Certificate<F32,d>) ->
  Certificate.Certificate<F32,source(~Array<F32>,Kernel.bias(work,d,result))>:
  %Equal.sym(Array<F32>,source(~Array<F32>,Kernel.bias(work,d,result)),d,bias_owner(work,d,result,certificate)) : Certificate.Certificate<F32,_>
  certificate
