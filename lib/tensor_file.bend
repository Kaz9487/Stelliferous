# Raw little-endian FP32 files and NumPy .npy files through the official File
# effects, and the reader that other formats share: a byte range of an open
# file, decoded element by element, written straight into a tensor's storage.
import Base
import ./storage_bytes.bend as Bytes
import ./storage_element_format.bend as Formats
import ./math_fp32.bend as Fp32
import ./storage_buffer.bend as Buffers
import ./tensor.bend as Tensors
import ./tensor_layout.bend as Layout

def Tensor() -> Type:
  Tensors.Tensor<F32>

def Read() -> Type:
  File & Result<&1,&1,U32 & String,List<&2,U32>>

def Written() -> Type:
  File & Result<&1,&1,U32 & String,Unit>

def Sized() -> Type:
  File & Result<&1,&1,U32 & String,U32>

# FP32 values per File effect; a read takes four times as many bytes.
def chunk() -> U32:
  16384

def chunk_bytes() -> U32:
  (chunk() * 4 : U32)

# Reading: bytes gather into elements, least significant first. The values of
# one read are written into the buffer before the next read, so no list
# outgrows a read.

# An element in progress: its first four bytes, the bytes above them, the
# weight of the next byte within its word, and how many bytes arrived.
type Partial is Data:
  Partial{low: U32,high: U32,weight: U32,filled: U32}

type Words is Data:
  Words{values: List<&2,F32>,count: U32,partial: Partial}

# The buffer of the tensor, the values found so far, and an element in progress.
type Target is Type:
  Target{buffer: Buffers.Buffer<F32>,found: U32,partial: Partial}

type Reading is Type:
  More{file: File,position: U32,remaining: U32,target: Target}
  End{file: File,target: Target,complete: Bool}

type Outcome is Type:
  Outcome{file: File,target: Target,complete: Bool}

def fresh_partial() -> Partial:
  Partial{0,0,1,0}

def gathered(lower: Bool,done: Bool,wrap: Bool,+byte: U32,format: Formats.Format,values: List<&2,F32>,+count: U32,+low: U32,+high: U32,+weight: U32,+filled: U32) -> Words:
  match lower done wrap:
    case True{} True{} wrap: Words{Formats.widen(format,(low + byte * weight : U32),high) <> values,(count + 1 : U32),fresh_partial()}
    case False{} True{} wrap: Words{Formats.widen(format,low,(high + byte * weight : U32)) <> values,(count + 1 : U32),fresh_partial()}
    case True{} False{} True{}: Words{values,count,Partial{(low + byte * weight : U32),high,1,filled}}
    case True{} False{} False{}: Words{values,count,Partial{(low + byte * weight : U32),high,(weight * 256 : U32),filled}}
    case False{} False{} wrap: Words{values,count,Partial{low,(high + byte * weight : U32),(weight * 256 : U32),filled}}

def push(+byte: U32,+format: Formats.Format,words: Words) -> Words:
  Words{values,+count,partial} = words
  Partial{+low,+high,+weight,+filled} = partial
  gathered(U32.is_lt(filled,4),U32.is_eq((filled + 1 : U32),Formats.width(format)),U32.is_eq(weight,16777216),byte,format,values,count,low,high,weight,(filled + 1 : U32))

def trickle(bytes: List<&2,U32>,+format: Formats.Format,words: Words) -> Words:
  match bytes:
    case []: words
    case byte <> rest: trickle(rest,format,push(byte,format,words))

# Whole elements at a time while the bytes allow; the values are newest first.
def pairs(bytes: List<&2,U32>,+format: Formats.Format,values: List<&2,F32>,+count: U32) -> Words:
  match bytes:
    case Con{first,Con{second,rest}}:
      pairs(rest,format,Formats.widen(format,(first + second * 256 : U32),0) <> values,(count + 1 : U32))
    case rest: trickle(rest,format,Words{values,count,fresh_partial()})

def quads(bytes: List<&2,U32>,+format: Formats.Format,values: List<&2,F32>,+count: U32) -> Words:
  match bytes:
    case Con{first,Con{second,Con{third,Con{fourth,rest}}}}:
      quads(rest,format,Formats.widen(format,(first + second * 256 + third * 65536 + fourth * 16777216 : U32),0) <> values,(count + 1 : U32))
    case rest: trickle(rest,format,Words{values,count,fresh_partial()})

def octets(bytes: List<&2,U32>,+format: Formats.Format,values: List<&2,F32>,+count: U32) -> Words:
  match bytes:
    case Con{first,Con{second,Con{third,Con{fourth,Con{fifth,Con{sixth,Con{seventh,Con{eighth,rest}}}}}}}}:
      octets(rest,format,Formats.widen(format,(first + second * 256 + third * 65536 + fourth * 16777216 : U32),
        (fifth + sixth * 256 + seventh * 65536 + eighth * 16777216 : U32)) <> values,(count + 1 : U32))
    case rest: trickle(rest,format,Words{values,count,fresh_partial()})

def by_width(+width: U32,bytes: List<&2,U32>,+format: Formats.Format) -> Words:
  match width:
    case 2: pairs(bytes,format,[],0)
    case 4: quads(bytes,format,[],0)
    case 8: octets(bytes,format,[],0)
    case _: trickle(bytes,format,Words{[],0,fresh_partial()})

def words(whole: Bool,bytes: List<&2,U32>,+format: Formats.Format,partial: Partial) -> Words:
  match whole:
    case True{}: by_width(Formats.width(format),bytes,format)
    case False{}: trickle(bytes,format,Words{[],0,partial})

# Values beyond the count of the shape are counted and not written.
def placed(read: Words,buffer: Buffers.Buffer<F32>,+found: U32,+count: U32) -> Target:
  Words{values,+added,partial} = read
  Target{Buffers.write_at(~F32,List.take(&2,F32,List.reverse(&2,F32,values),U32.to_nat((count - U32.min(found,count) : U32))),U32.min(found,count),buffer),
    (found + added : U32),partial}

def fill(bytes: List<&2,U32>,target: Target,+format: Formats.Format,+count: U32) -> Target:
  Target{buffer,+found,partial} = target
  Partial{low,high,weight,+filled} = partial
  placed(words(U32.is_zero(filled),bytes,format,Partial{low,high,weight,filled}),buffer,found,count)

def ranged(exhausted: Bool,file: File,+position: U32,+remaining: U32,target: Target) -> Reading:
  match exhausted:
    case True{}: End{file,target,True{}}
    case False{}: More{file,position,remaining,target}

# A read that ends early leaves the count short, which the count check reports.
def absorbed(bytes: List<&2,U32>,file: File,+position: U32,+remaining: U32,target: Target,+format: Formats.Format,+count: U32) -> Reading:
  match bytes:
    case []: End{file,target,True{}}
    case Con{+byte,+rest}:
      +length = U32.min(U32.from_nat(List.length(&2,U32,byte <> rest)),remaining)
      ranged(U32.is_eq(length,remaining),file,(position + length : U32),(remaining - length : U32),fill(byte <> rest,target,format,count))

def received(result: Result<&1,&1,U32 & String,List<&2,U32>>,file: File,+position: U32,+remaining: U32,target: Target,+format: Formats.Format,+count: U32) -> Reading:
  match result:
    case Fail{error}: End{file,target,False{}}
    case Done{bytes}: absorbed(bytes,file,position,remaining,target,format,count)

def taken(read: Read(),+position: U32,+remaining: U32,target: Target,+format: Formats.Format,+count: U32) -> Reading:
  (file,result) = read
  received(result,file,position,remaining,target,format,count)

def read_all(fuel: Nat,reading: Reading,+format: Formats.Format,+count: U32) -> IO(Outcome):
  match fuel reading:
    case 0n More{file,position,remaining,target}: IO.pure(Outcome,Outcome{file,target,False{}})
    case 0n End{file,target,complete}: IO.pure(Outcome,Outcome{file,target,complete})
    case 1n+rest End{file,target,complete}: IO.pure(Outcome,Outcome{file,target,complete})
    case 1n+rest More{file,+position,+remaining,target}:
      do IO<Outcome>:
        read : Read() <- File.read_at(file,position,U32.min(remaining,chunk_bytes()))
        read_all(rest,taken(read,position,remaining,target,format,count),format,count)

def fresh(+count: U32) -> Target:
  Target{Tensors.buffer(~F32,0.0,count),0,fresh_partial()}

def fuel(+length: U32) -> Nat:
  U32.to_nat((length / chunk_bytes() + 2 : U32))

# The elements stored from the position on, as many bytes as the count of the
# format takes; the file stays open.
def read_range(file: File,+position: U32,+length: U32,+format: Formats.Format,+count: U32) -> IO(Outcome):
  read_all(fuel(length),ranged(U32.is_zero(length),file,position,length,fresh(count)),format,count)

def fault(operation: String,path: String,detail: String,shape: List<&2,U32>) -> Tensor():
  Tensors.failed(~F32,operation,Tensors.FileFault{path,detail},shape,0.0,1)

def sized(matches: Bool,operation: String,path: String,+shape: List<&2,U32>,buffer: Buffers.Buffer<F32>) -> Tensor():
  match matches:
    case True{}: Tensors.from_storage(~F32,shape,buffer,0.0)
    case False{}: fault(operation,path,"does not hold the values of the shape",shape)

def whole(aligned: Bool,operation: String,path: String,+shape: List<&2,U32>,buffer: Buffers.Buffer<F32>,+found: U32,+count: U32) -> Tensor():
  match aligned:
    case False{}: fault(operation,path,"holds a partial element",shape)
    case True{}: sized(U32.is_eq(found,count),operation,path,shape,buffer)

def aligned(operation: String,path: String,+shape: List<&2,U32>,target: Target,+count: U32) -> Tensor():
  Target{buffer,+found,partial} = target
  Partial{low,high,weight,+filled} = partial
  whole(U32.is_zero(filled),operation,path,shape,buffer,found,count)

def checked(complete: Bool,operation: String,path: String,+shape: List<&2,U32>,target: Target,+count: U32) -> Tensor():
  match complete:
    case False{}: fault(operation,path,"cannot be read",shape)
    case True{}: aligned(operation,path,shape,target,count)

def decoded(operation: String,path: String,+shape: List<&2,U32>,outcome: Outcome,+count: U32) -> File & Tensor():
  Outcome{file,target,complete} = outcome
  (file,checked(complete,operation,path,shape,target,count))

def ranged_count(fits: Bool,file: File,operation: String,path: String,+shape: List<&2,U32>,+position: U32,+length: U32,+format: Formats.Format,+count: U32) -> IO(File & Tensor()):
  match fits:
    case False{}: IO.pure(File & Tensor(),(file,fault(operation,path,"holds a tensor beyond the machine capacity",shape)))
    case True{}:
      do IO<File & Tensor()>:
        outcome : Outcome <- read_range(file,position,length,format,count)
        IO.pure(File & Tensor(),decoded(operation,path,shape,outcome,count))

def ranged_shape(count: Maybe<&2,U32>,file: File,operation: String,path: String,+shape: List<&2,U32>,+position: U32,+format: Formats.Format) -> IO(File & Tensor()):
  match count:
    case None{}: IO.pure(File & Tensor(),(file,Tensors.failed(~F32,operation,Tensors.InvalidShape{shape},shape,0.0,1)))
    case Some{+count}:
      +length = (count * Formats.width(format) : U32)
      ranged_count(U32.is_eq((length / Formats.width(format) : U32),count),file,operation,path,shape,position,length,format,count)

# The tensor of the shape whose elements of the format start at the position
# of an open file, which stays open.
def load_range(file: File,operation: String,+path: String,+shape: List<&2,U32>,+position: U32,+format: Formats.Format) -> IO(File & Tensor()):
  ranged_shape(Tensors.count_of(shape),file,operation,path,shape,position,format)

def closed(pair: File & Tensor()) -> IO(Tensor()):
  (file,tensor) = pair
  do IO<Tensor()>:
    File.close(file)
    IO.pure(Tensor(),tensor)

def refused(file: File,tensor: Tensor()) -> IO(Tensor()):
  do IO<Tensor()>:
    File.close(file)
    IO.pure(Tensor(),tensor)

# A raw file holds exactly the bytes of its shape: a size that is not a whole
# number of values, or another number of them, is refused before reading.
def measured(aligned: Bool,matches: Bool,file: File,operation: String,+path: String,+shape: List<&2,U32>,+position: U32,+count: U32) -> IO(Tensor()):
  match aligned matches:
    case False{} matches: refused(file,fault(operation,path,"is not a whole number of FP32 values",shape))
    case True{} False{}: refused(file,fault(operation,path,"does not hold the values of the shape",shape))
    case True{} True{}:
      do IO<Tensor()>:
        pair : File & Tensor() <- load_range(file,operation,path,shape,position,Formats.Fp32{})
        closed(pair)

def with_size(result: Result<&1,&1,U32 & String,U32>,file: File,operation: String,+path: String,+shape: List<&2,U32>,+position: U32,+count: U32) -> IO(Tensor()):
  match result:
    case Fail{error}: refused(file,fault(operation,path,"cannot be measured",shape))
    case Done{+size}:
      measured(U32.is_zero(((size - U32.min(size,position)) % 4 : U32)),U32.is_le(position,size) && U32.is_eq(((size - U32.min(size,position)) / 4 : U32),count),file,operation,path,shape,position,count)

def sized_file(sized: Sized(),operation: String,+path: String,+shape: List<&2,U32>,+position: U32,+count: U32) -> IO(Tensor()):
  (file,result) = sized
  with_size(result,file,operation,path,shape,position,count)

def fp32_values(file: File,operation: String,+path: String,+shape: List<&2,U32>,+position: U32,+count: U32) -> IO(Tensor()):
  do IO<Tensor()>:
    sized : Sized() <- File.size(file)
    sized_file(sized,operation,path,shape,position,count)

def opened(result: Result<&1,&1,U32 & String,File>,+path: String,+shape: List<&2,U32>,+count: U32) -> IO(Tensor()):
  match result:
    case Fail{error}: IO.pure(Tensor(),fault("load",path,"cannot be opened",shape))
    case Done{file}: fp32_values(file,"load",path,shape,0,count)

def counted(count: Maybe<&2,U32>,+path: String,+shape: List<&2,U32>) -> IO(Tensor()):
  match count:
    case None{}: IO.pure(Tensor(),Tensors.failed(~F32,"load",Tensors.InvalidShape{shape},shape,0.0,1))
    case Some{+count}:
      do IO<Tensor()>:
        result : Result<&1,&1,U32 & String,File> <- File.open(path,"r")
        opened(result,path,shape,count)

# The file holds the values of the shape in row-major order and nothing else.
def load(+path: String,+shape: List<&2,U32>) -> IO(Tensor()):
  counted(Tensors.count_of(shape),path,shape)

# The NumPy .npy header: magic, version, header length, then a text such as
# {'descr': '<f4', 'fortran_order': False, 'shape': (16, 3, 6, 6), }

# The shape and the position of the first value.
type Header is Data:
  Header{shape: List<&2,U32>,data: U32}

def magic() -> List<&2,U32>:
  [147,78,85,77,80,89]

def contains(text: List<&2,U32>,+pattern: List<&2,U32>) -> Bool:
  match text:
    case Nil{}: False{}
    case Con{+byte,+rest}: Bytes.starts(pattern,byte <> rest) || contains(rest,pattern)

# The text after the first occurrence of the pattern.
def after(text: List<&2,U32>,+pattern: List<&2,U32>) -> Maybe<&2,List<&2,U32>>:
  match text:
    case Nil{}: None{}
    case Con{+byte,+rest}:
      Bool.pick(Maybe<&2,List<&2,U32>>,Bytes.starts(pattern,byte <> rest),Some{List.drop(&2,U32,byte <> rest,List.length(&2,U32,pattern))},after(rest,pattern))

def pushed(digits: Bool,current: U32,shape: List<&2,U32>) -> List<&2,U32>:
  match digits:
    case True{}: current <> shape
    case False{}: shape

def closed_shape(valid: Bool,shape: List<&2,U32>) -> Maybe<&2,List<&2,U32>>:
  match valid:
    case True{}: Some{List.reverse(&2,U32,shape)}
    case False{}: None{}

# Extents up to the closing parenthesis: digits, commas and spaces.
def extents(text: List<&2,U32>,+current: U32,digits: Bool,valid: Bool,shape: List<&2,U32>) -> Maybe<&2,List<&2,U32>>:
  match text:
    case Nil{}: None{}
    case Con{+byte,rest}:
      match byte:
        case 41: closed_shape(valid,pushed(digits,current,shape))
        case 44: extents(rest,0,False{},valid,pushed(digits,current,shape))
        case 32: extents(rest,current,digits,valid,shape)
        case _: extents(rest,(current * 10 + byte - 48 : U32),True{},valid && U32.is_le(48,byte) && U32.is_le(byte,57),shape)

def shaped(shape: Maybe<&2,List<&2,U32>>,+data: U32) -> Maybe<&2,Header>:
  match shape:
    case None{}: None{}
    case Some{shape}: Some{Header{shape,data}}

def typed(supported: Bool,text: Maybe<&2,List<&2,U32>>,+data: U32) -> Maybe<&2,Header>:
  match supported text:
    case False{} text: None{}
    case True{} None{}: None{}
    case True{} Some{text}: shaped(extents(text,0,False{},True{},[]),data)

def described(+text: List<&2,U32>,+length: Nat,+data: U32) -> Maybe<&2,Header>:
  typed(Nat.is_eq(List.length(&2,U32,text),length) && contains(text,Bytes.codes("'descr': '<f4'")) && contains(text,Bytes.codes("'fortran_order': False")),
    after(text,Bytes.codes("'shape': (")),data)

def region(+bytes: List<&2,U32>,+prefix: U32,+length: U32) -> Maybe<&2,Header>:
  described(List.take(&2,U32,List.drop(&2,U32,bytes,U32.to_nat(prefix)),U32.to_nat(length)),U32.to_nat(length),(prefix + length : U32))

# Version 1 has a two-byte header length; later versions have four bytes.
def versioned(major: U32,+bytes: List<&2,U32>) -> Maybe<&2,Header>:
  match major:
    case 1: region(bytes,10,Bytes.u16_at(bytes,8n))
    case _: region(bytes,12,Bytes.u32_at(bytes,8n))

def marked(valid: Bool,+bytes: List<&2,U32>) -> Maybe<&2,Header>:
  match valid:
    case False{}: None{}
    case True{}: versioned(Bytes.at(bytes,6n),bytes)

def header(+bytes: List<&2,U32>) -> Maybe<&2,Header>:
  marked(Bytes.starts(magic(),bytes),bytes)

def npy_counted(count: Maybe<&2,U32>,+shape: List<&2,U32>,+data: U32,file: File,+path: String) -> IO(Tensor()):
  match count:
    case None{}: refused(file,Tensors.failed(~F32,"load_npy",Tensors.InvalidShape{shape},shape,0.0,1))
    case Some{+count}: fp32_values(file,"load_npy",path,shape,data,count)

def npy_parsed(parsed: Maybe<&2,Header>,file: File,+path: String) -> IO(Tensor()):
  match parsed:
    case None{}: refused(file,fault("load_npy",path,"is not a little-endian FP32 .npy array in C order",[]))
    case Some{Header{+shape,data}}: npy_counted(Tensors.count_of(shape),shape,data,file,path)

def npy_received(result: Result<&1,&1,U32 & String,List<&2,U32>>,file: File,+path: String) -> IO(Tensor()):
  match result:
    case Fail{error}: refused(file,fault("load_npy",path,"cannot be read",[]))
    case Done{bytes}: npy_parsed(header(bytes),file,path)

def npy_headed(read: Read(),+path: String) -> IO(Tensor()):
  (file,result) = read
  npy_received(result,file,path)

def npy_opened(result: Result<&1,&1,U32 & String,File>,+path: String) -> IO(Tensor()):
  match result:
    case Fail{error}: IO.pure(Tensor(),fault("load_npy",path,"cannot be opened",[]))
    case Done{file}:
      do IO<Tensor()>:
        read : Read() <- File.read_at(file,0,chunk_bytes())
        npy_headed(read,path)

# An FP32 array saved by NumPy, with its shape.
def load_npy(+path: String) -> IO(Tensor()):
  do IO<Tensor()>:
    result : Result<&1,&1,U32 & String,File> <- File.open(path,"r")
    npy_opened(result,path)

# Writing

type Writing is Type:
  Pending{file: File,storage: Array<F32>,offset: U32,remaining: Nat}
  Finished{file: File,complete: Bool}

def bytes(+word: U32,rest: List<&2,U32>) -> List<&2,U32>:
  (word % 256 : U32) <> (word / 256 % 256 : U32) <> (word / 65536 % 256 : U32) <> (word / 16777216 : U32) <> rest

def encode(count: Nat,values: List<&2,F32>) -> List<&2,U32>:
  match count values:
    case 0n values: []
    case 1n+rest Nil{}: []
    case 1n+rest Con{value,more}: bytes(Fp32.to_bits(value),encode(rest,more))

# The cursor retains the array; only the current chunk becomes a list.
def read_chunk(count: Nat,+offset: U32,storage: Array<F32>,values: List<&2,F32>) -> Array<F32> & List<&2,F32>:
  match count:
    case 0n: (storage,List.reverse(&2,F32,values))
    case 1n+rest:
      Buffers.bind_pair(Array<F32>,F32,Array<F32> & List<&2,F32>,Array.get(F32,storage,offset),storage => value =>
        read_chunk(rest,(offset + 1 : U32),storage,value <> values))

def pending(remaining: Nat,file: File,storage: Array<F32>,offset: U32) -> Writing:
  match remaining:
    case 0n: Finished{file,True{}}
    case 1n+rest: Pending{file,storage,offset,1n+rest}

def continued(result: Result<&1,&1,U32 & String,Unit>,file: File,storage: Array<F32>,remaining: Nat,offset: U32) -> Writing:
  match result:
    case Fail{error}: Finished{file,False{}}
    case Done{unit}: pending(remaining,file,storage,offset)

def after_write(written: Written(),storage: Array<F32>,remaining: Nat,offset: U32) -> Writing:
  (file,result) = written
  continued(result,file,storage,remaining,offset)

def write_chunk(file: File,read: Array<F32> & List<&2,F32>,+count: Nat,remaining: Nat,offset: U32) -> IO(Writing):
  (storage,values) = read
  do IO<Writing>:
    written : Written() <- File.write_bytes(file,encode(count,values))
    IO.pure(Writing,after_write(written,storage,Nat.sub(remaining,count),(offset + U32.from_nat(count) : U32)))

def finished(file: File,complete: Bool) -> IO(Bool):
  do IO<Bool>:
    File.close(file)
    IO.pure(Bool,complete)

def write_all(fuel: Nat,writing: Writing) -> IO(Bool):
  match fuel writing:
    case 0n Pending{file,storage,offset,remaining}: finished(file,False{})
    case 0n Finished{file,complete}: finished(file,complete)
    case 1n+rest Finished{file,complete}: finished(file,complete)
    case 1n+rest Pending{file,storage,+offset,+remaining}:
      +count = Nat.min(U32.to_nat(chunk()),remaining)
      do IO<Bool>:
        next : Writing <- write_chunk(file,read_chunk(count,offset,storage,[]),count,remaining,offset)
        write_all(rest,next)

def unsaved(pair: Tensor() & List<&2,U32>,operation: String,path: String,detail: String) -> Tensor():
  (tensor,shape) = pair
  fault(operation,path,detail,shape)

def written(complete: Bool,operation: String,path: String,tensor: Tensor()) -> Tensor():
  match complete:
    case True{}: tensor
    case False{}: unsaved(Tensors.shape(~F32,tensor),operation,path,"cannot be written")

# The prefix is written first; a raw file has none.
def created(result: Result<&1,&1,U32 & String,File>,operation: String,+path: String,tensor: Tensor(),prefix: List<&2,U32>,storage: Array<F32>,+length: Nat) -> IO(Tensor()):
  match result:
    case Fail{error}: IO.pure(Tensor(),unsaved(Tensors.shape(~F32,tensor),operation,path,"cannot be created"))
    case Done{file}:
      do IO<Tensor()>:
        head : Written() <- File.write_bytes(file,prefix)
        complete : Bool <- write_all(Nat.add(length,2n),after_write(head,storage,length,0))
        IO.pure(Tensor(),written(complete,operation,path,tensor))

def saving(operation: String,+path: String,tensor: Tensor(),prefix: List<&2,U32>,buffer: Result<&2,&1,Tensors.Problem,Buffers.Buffer<F32>>) -> IO(Tensor()):
  match buffer:
    case Fail{problem}: IO.pure(Tensor(),tensor)
    case Done{Buffers.Buffer{storage,length,certificate}}:
      do IO<Tensor()>:
        result : Result<&1,&1,U32 & String,File> <- File.open(path,"w")
        created(result,operation,path,tensor,prefix,storage,length)

# Writes the values in row-major order and returns the tensor. A failed tensor
# writes nothing and keeps its problem.
def save(+path: String,tensor: Tensor()) -> IO(Tensor()):
  Tensors.then(~Tensor(),~Tensor(),~IO(Tensor()),Layout.copy(~F32,tensor),kept => copy => saving("save",path,kept,[],Tensors.into_storage(~F32,copy)))

def joined(shape: List<&2,U32>) -> String:
  match shape:
    case []: ""
    case [extent]: U32.show(extent)
    case extent <> rest: U32.show(extent) ++ ", " ++ joined(rest)

# NumPy writes (), (3,) and (2, 3).
def tuple(shape: List<&2,U32>) -> String:
  match shape:
    case []: "()"
    case [extent]: "(" ++ U32.show(extent) ++ ",)"
    case extent <> rest: "(" ++ joined(extent <> rest) ++ ")"

def spaces(count: Nat) -> List<&2,U32>:
  match count:
    case 0n: [10]
    case 1n+rest: 32 <> spaces(rest)

# Version 1: the header ends in a newline and the data starts at a multiple of 64 bytes.
def framed(+text: List<&2,U32>,+length: U32) -> List<&2,U32>:
  +padding = ((64 - (11 + length) % 64) % 64 : U32)
  +total = (length + padding + 1 : U32)
  List.append(&2,U32,magic(),1 <> 0 <> (total % 256 : U32) <> (total / 256 : U32) <> List.append(&2,U32,text,spaces(U32.to_nat(padding))))

def npy_header(shape: List<&2,U32>) -> List<&2,U32>:
  +text = Bytes.codes("{'descr': '<f4', 'fortran_order': False, 'shape': " ++ tuple(shape) ++ ", }")
  framed(text,U32.from_nat(List.length(&2,U32,text)))

def npy_saving(+path: String,pair: Tensor() & Tensor(),+shape: List<&2,U32>) -> IO(Tensor()):
  (kept,copy) = pair
  saving("save_npy",path,kept,npy_header(shape),Tensors.into_storage(~F32,copy))

# Writes a .npy file that NumPy loads with the tensor's shape.
def save_npy(+path: String,tensor: Tensor()) -> IO(Tensor()):
  Tensors.then(~Tensor(),~List<&2,U32>,~IO(Tensor()),Tensors.shape(~F32,tensor),tensor => shape => npy_saving(path,Layout.copy(~F32,tensor),shape))
