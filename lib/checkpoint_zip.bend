# A ZIP archive read through the official File effects. The end record locates
# the central directory, which names every entry with its size and the offset of
# its local header; the entry's bytes follow that header. Only stored entries
# are read, the form torch.save writes.
import Base
import ./storage_bytes.bend as Bytes

# An entry: its name as bytes, the offset of its local header, its length and
# its compression method, 0 for stored.
type Record is Data:
  Record{name: List<&2,U32>,header: U32,length: U32,method: U32}

type Listing is Data:
  Records{records: List<&2,Record>}
  Unreadable{detail: String}

type Fetched is Type:
  Fetched{file: File,bytes: Maybe<&2,List<&2,U32>>}

def Read() -> Type:
  File & Result<&1,&1,U32 & String,List<&2,U32>>

def Sized() -> Type:
  File & Result<&1,&1,U32 & String,U32>

# Signatures, least significant byte first.
def central_signature() -> U32:
  33639248

def local_signature() -> U32:
  67324752

def end_signature() -> List<&2,U32>:
  [80,75,5,6]

# A central directory entry is 46 bytes, then its name, extra field and comment.
def records(fuel: Nat,valid: Bool,+bytes: List<&2,U32>,found: List<&2,Record>) -> Listing:
  match fuel valid:
    case 0n valid: Records{List.reverse(&2,Record,found)}
    case 1n+rest False{}: Unreadable{"has a damaged central directory"}
    case 1n+rest True{}:
      +name_length = Bytes.u16_at(bytes,28n)
      +skipped = U32.to_nat((46 + name_length + Bytes.u16_at(bytes,30n) + Bytes.u16_at(bytes,32n) : U32))
      +next = List.drop(&2,U32,bytes,skipped)
      records(rest,U32.is_eq(Bytes.u32_at(next,0n),central_signature()),next,
        Record{List.take(&2,U32,List.drop(&2,U32,bytes,46n),U32.to_nat(name_length)),Bytes.u32_at(bytes,42n),Bytes.u32_at(bytes,24n),Bytes.u16_at(bytes,10n)} <> found)

def directory_read(result: Result<&1,&1,U32 & String,List<&2,U32>>,+count: U32) -> Listing:
  match result:
    case Fail{error}: Unreadable{"cannot be read"}
    case Done{+bytes}: records(U32.to_nat(count),U32.is_eq(Bytes.u32_at(bytes,0n),central_signature()),bytes,[])

def directory_fetched(read: Read(),+count: U32) -> File & Listing:
  (file,result) = read
  (file,directory_read(result,count))

def ending(wide: Bool,file: File,+count: U32,+size: U32,+offset: U32) -> IO(File & Listing):
  match wide:
    case True{}: IO.pure(File & Listing,(file,Unreadable{"is a ZIP64 archive"}))
    case False{}:
      do IO<File & Listing>:
        read : Read() <- File.read_at(file,offset,size)
        IO.pure(File & Listing,directory_fetched(read,count))

# The end record: entry count at byte 10, directory size at 12 and offset at 16.
# A count or an offset of all ones means ZIP64, which a 32-bit file never needs.
def end_found(found: Maybe<&2,U32>,+tail: List<&2,U32>,file: File) -> IO(File & Listing):
  match found:
    case None{}: IO.pure(File & Listing,(file,Unreadable{"is not a ZIP archive"}))
    case Some{+start}:
      +count = Bytes.u16_at(tail,U32.to_nat((start + 10 : U32)))
      +size = Bytes.u32_at(tail,U32.to_nat((start + 12 : U32)))
      +offset = Bytes.u32_at(tail,U32.to_nat((start + 16 : U32)))
      ending(U32.is_eq(count,65535) || U32.is_eq(offset,4294967295),file,count,size,offset)

def tail_read(result: Result<&1,&1,U32 & String,List<&2,U32>>,file: File) -> IO(File & Listing):
  match result:
    case Fail{error}: IO.pure(File & Listing,(file,Unreadable{"cannot be read"}))
    case Done{+tail}: end_found(Bytes.last_start(tail,end_signature(),0,None{}),tail,file)

def tail_fetched(read: Read()) -> IO(File & Listing):
  (file,result) = read
  tail_read(result,file)

# The end record is 22 bytes plus a comment of at most 65535.
def measured(result: Result<&1,&1,U32 & String,U32>,file: File) -> IO(File & Listing):
  match result:
    case Fail{error}: IO.pure(File & Listing,(file,Unreadable{"cannot be measured"}))
    case Done{+size}:
      +tail = U32.min(size,65557)
      do IO<File & Listing>:
        read : Read() <- File.read_at(file,(size - tail : U32),tail)
        tail_fetched(read)

def sized(sized: Sized()) -> IO(File & Listing):
  (file,result) = sized
  measured(result,file)

# The entries of an open file, in directory order.
def directory(file: File) -> IO(File & Listing):
  do IO<File & Listing>:
    size : Sized() <- File.size(file)
    sized(size)

def find(records: List<&2,Record>,+name: List<&2,U32>) -> Maybe<&2,Record>:
  match records:
    case Nil{}: None{}
    case Con{Record{+entry,+header,+length,+method},rest}:
      Bool.pick(Maybe<&2,Record>,Bytes.same(entry,name),Some{Record{entry,header,length,method}},find(rest,name))

def first_matching(records: List<&2,Record>,+suffix: List<&2,U32>) -> Maybe<&2,Record>:
  match records:
    case Nil{}: None{}
    case Con{Record{+entry,+header,+length,+method},rest}:
      Bool.pick(Maybe<&2,Record>,Bytes.ends(entry,suffix),Some{Record{entry,header,length,method}},first_matching(rest,suffix))

# The local header is 30 bytes, then the name and the extra field; the data follows.
def located(valid: Bool,+bytes: List<&2,U32>,+header: U32) -> Maybe<&2,U32>:
  match valid:
    case False{}: None{}
    case True{}: Some{(header + 30 + Bytes.u16_at(bytes,26n) + Bytes.u16_at(bytes,28n) : U32)}

def header_read(result: Result<&1,&1,U32 & String,List<&2,U32>>,+header: U32) -> Maybe<&2,U32>:
  match result:
    case Fail{error}: None{}
    case Done{+bytes}: located(U32.is_eq(Bytes.u32_at(bytes,0n),local_signature()),bytes,header)

def header_fetched(read: Read(),+header: U32) -> File & Maybe<&2,U32>:
  (file,result) = read
  (file,header_read(result,header))

# The offset of the first byte of a stored entry. A compressed entry has none.
def stored_offset(stored: Bool,file: File,+header: U32) -> IO(File & Maybe<&2,U32>):
  match stored:
    case True{}:
      do IO<File & Maybe<&2,U32>>:
        read : Read() <- File.read_at(file,header,30)
        IO.pure(File & Maybe<&2,U32>,header_fetched(read,header))
    case False{}: IO.pure(File & Maybe<&2,U32>,(file,None{}))

def data_offset(file: File,+record: Record) -> IO(File & Maybe<&2,U32>):
  Record{name,+header,length,+method} = record
  stored_offset(U32.is_eq(method,0),file,header)

def entry_read(result: Result<&1,&1,U32 & String,List<&2,U32>>,+length: U32) -> Maybe<&2,List<&2,U32>>:
  match result:
    case Fail{error}: None{}
    case Done{+bytes}: Bool.pick(Maybe<&2,List<&2,U32>>,Nat.is_eq(List.length(&2,U32,bytes),U32.to_nat(length)),Some{bytes},None{})

def entry_fetched(read: Read(),+length: U32) -> Fetched:
  (file,result) = read
  Fetched{file,entry_read(result,length)}

def entry_at(offset: Maybe<&2,U32>,file: File,+length: U32) -> IO(Fetched):
  match offset:
    case None{}: IO.pure(Fetched,Fetched{file,None{}})
    case Some{offset}:
      do IO<Fetched>:
        read : Read() <- File.read_at(file,offset,length)
        IO.pure(Fetched,entry_fetched(read,length))

def entry_located(located: File & Maybe<&2,U32>,+length: U32) -> IO(Fetched):
  (file,offset) = located
  entry_at(offset,file,length)

# Every byte of a stored entry in one read, for small entries.
def entry(file: File,+record: Record) -> IO(Fetched):
  Record{name,header,+length,method} = record
  do IO<Fetched>:
    located : File & Maybe<&2,U32> <- data_offset(file,Record{name,header,length,method})
    entry_located(located,length)
