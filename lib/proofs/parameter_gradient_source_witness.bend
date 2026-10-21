# The bias reader instantiates the shared grouped source witness: each
# reduced output leaves the reader as it was.
import Base
import ../kernel_parameter_gradient.bend as Kernel
import ../traversal_grouped.bend as Grouped
import ./storage_witness.bend as ArrayWitness
import ./traversal_witness.bend as Witness
import ./grouped_source_witness.bend as GroupWitness

def bias_evidence(-reader: Array<F32>) -> Data:
  ArrayWitness.ArrayWitness<F32,reader>

def bias_cell(-reader: Array<F32>,context: Kernel.Context,position: Nat,source: bias_evidence(reader)) ->
  Witness.Sample<Array<F32>,F32,reader,Kernel.bias_at(reader,context,position)>:
  Witness.array_sample(~F32,reader,Kernel.bias_at(reader,context,position),ArrayWitness.observed(F32,reader,Kernel.bias_address(context,position),source))

def image_tail(~Reader: Type,~read: Reader -> Kernel.Context -> Nat -> Reader & F32,-reader: Reader,-result: Reader & F32,
  +rest: Nat,+image: Nat,+element: Nat,+work: Kernel.Work,+ranges: List<&2,Grouped.Range>,
  head: Witness.Sample<Reader,F32,reader,result>,tail: Witness.Sample<Reader,List<&2,F32>,reader,Kernel.image_values(~Reader,~read,rest,image,element,work,ranges,reader)>) ->
  Witness.Sample<Reader,List<&2,F32>,reader,Kernel.image_tail(~Reader,result,reader => Kernel.image_values(~Reader,~read,rest,image,element,work,ranges,reader))>:
  Witness.Sample{value,equation} = head
  %Equal.sym(Reader & F32,result,(reader,value),equation) : Witness.Sample<Reader,List<&2,F32>,reader,Kernel.image_tail(~Reader,_,reader => Kernel.image_values(~Reader,~read,rest,image,element,work,ranges,reader))>
  GroupWitness.prepended(~Reader,reader,value,Kernel.image_values(~Reader,~read,rest,image,element,work,ranges,reader),tail)

def images(~Reader: Type,~Evidence: @-reader: Reader -> Data,~read: Reader -> Kernel.Context -> Nat -> Reader & F32,
  ~preserves: @-reader: Reader -> @context: Kernel.Context -> @position: Nat -> Evidence(reader) -> Witness.Sample<Reader,F32,reader,read(reader,context,position)>,
  remaining: Nat,+image: Nat,+element: Nat,+work: Kernel.Work,+ranges: List<&2,Grouped.Range>,-reader: Reader,+source: Evidence(reader)) ->
  Witness.Sample<Reader,List<&2,F32>,reader,Kernel.image_values(~Reader,~read,remaining,image,element,work,ranges,reader)>:
  match remaining:
    case 0n: Witness.Sample{[],{==}}
    case 1n+ +rest:
      image_tail(~Reader,~read,reader,Grouped.run(~Reader,~Kernel.Context,~read,ranges,reader,Kernel.Context{work,image,element}),rest,1n+image,element,work,ranges,
        GroupWitness.run(~Reader,~Kernel.Context,~Evidence,~read,~preserves,ranges,reader,Kernel.Context{work,image,element},source),
        images(~Reader,~Evidence,~read,~preserves,rest,1n+image,element,work,ranges,reader,source))

def element(~Reader: Type,~Evidence: @-reader: Reader -> Data,~read: Reader -> Kernel.Context -> Nat -> Reader & F32,
  ~preserves: @-reader: Reader -> @context: Kernel.Context -> @position: Nat -> Evidence(reader) -> Witness.Sample<Reader,F32,reader,read(reader,context,position)>,
  +work: Kernel.Work,+index: Nat,+ranges: List<&2,Grouped.Range>,-reader: Reader,source: Evidence(reader)) ->
  Witness.Sample<Reader,F32,reader,Kernel.element(~Reader,~read,work,index,ranges,reader)>:
  Kernel.Work{+count,c,o,h,w,k,s,p,rows,columns,g} = work
  GroupWitness.reduced(~Reader,reader,Kernel.image_values(~Reader,~read,count,0n,index,work,ranges,reader),images(~Reader,~Evidence,~read,~preserves,count,0n,index,work,ranges,reader,source))
