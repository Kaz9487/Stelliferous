# Bias gradient traversal over ordered spatial groups.
import Base
import ./traversal_grouped.bend as Grouped
import ./storage_buffer.bend as Storage

type Work is Data:
  Work{images: Nat,channels: Nat,outputs: Nat,height: Nat,width: Nat,kernel: Nat,stride: Nat,padding: Nat,rows: Nat,columns: Nat,group: Nat}

type Context is Data:
  Context{work: Work,image: Nat,element: Nat}

def bias_address(context: Context,position: Nat) -> U32:
  Context{Work{n,c,+o,h,w,k,s,p,+rows,+columns,g},image,element} = context
  U32.from_nat(Nat.add(Nat.mul(Nat.mul(Nat.add(Nat.mul(image,o),element),rows),columns),position))

def bias_at(reader: Array<F32>,context: Context,position: Nat) -> Array<F32> & F32:
  Array.get(F32,reader,bias_address(context,position))

def image_tail(~Reader: Type,pair: Reader & F32,tail: Reader -> Reader & List<&2,F32>) -> Reader & List<&2,F32>:
  (reader,value) = pair
  Grouped.prepend(~Reader,value,tail(reader))

def image_values(~Reader: Type,~read: Reader -> Context -> Nat -> Reader & F32,remaining: Nat,+image: Nat,+element: Nat,+work: Work,
  +ranges: List<&2,Grouped.Range>,reader: Reader) -> Reader & List<&2,F32>:
  match remaining:
    case 0n: (reader,[])
    case 1n+rest:
      image_tail(~Reader,Grouped.run(~Reader,~Context,~read,ranges,reader,Context{work,image,element}),
        reader => image_values(~Reader,~read,rest,1n+image,element,work,ranges,reader))

def element(~Reader: Type,~read: Reader -> Context -> Nat -> Reader & F32,+work: Work,index: Nat,+ranges: List<&2,Grouped.Range>,reader: Reader) -> Reader & F32:
  Work{images,c,o,h,w,k,s,p,rows,columns,g} = work
  Grouped.reduced(~Reader,image_values(~Reader,~read,images,0n,index,work,ranges,reader))

# One reduced output written at its position, then the rest of the outputs.
def stored(~Reader: Type,+position: U32,result: Storage.Buffer<F32>,pair: Reader & F32,
  tail: Reader -> Storage.Buffer<F32> -> Reader & Storage.Buffer<F32>) -> Reader & Storage.Buffer<F32>:
  (reader,value) = pair
  tail(reader,Storage.write(~F32,position,value,result))

def written(~Reader: Type,~read: Reader -> Context -> Nat -> Reader & F32,remaining: Nat,+index: Nat,+position: U32,+work: Work,+ranges: List<&2,Grouped.Range>,
  reader: Reader,result: Storage.Buffer<F32>) -> Reader & Storage.Buffer<F32>:
  match remaining:
    case 0n: (reader,result)
    case 1n+rest:
      stored(~Reader,position,result,element(~Reader,~read,work,index,ranges,reader),
        reader => result => written(~Reader,~read,rest,1n+index,(position + 1 : U32),work,ranges,reader,result))

def bias(+work: Work,d: Array<F32>,result: Storage.Buffer<F32>) -> Array<F32> & Storage.Buffer<F32>:
  Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  written(~Array<F32>,~bias_at,o,0n,0,work,Grouped.ranges(rows,columns,g),d,result)
