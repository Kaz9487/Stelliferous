import Base
import ./storage_buffer.bend as Storage
import ./storage_allocation.bend as Allocation
import ./traversal_array.bend as Traversal
import ./kernel_tile.bend as Tile
import ./tensor.bend as Tensors

# Each ordered K coordinate owns eight adjacent weights, one per output row of
# a block; rows past the last output read zero. One native tile read replaces
# eight separately masked scalar reads.
type MappedLayout<-Context: Data> is Data:
  MappedLayout{rows: U32,inner: U32,row_stride: U32,context: Context}

def identity(context: Unit,index: U32) -> U32:
  index

type Bases is Data:
  Bases{a: U32,b: U32,c: U32,d: U32,e: U32,f: U32,g: U32,h: U32}

def linear_bases(~Context: Data,context: Context,+row_stride: U32,+row: U32) -> Bases:
  Bases{(row * row_stride : U32),((row + 1) * row_stride : U32),((row + 2) * row_stride : U32),((row + 3) * row_stride : U32),
    ((row + 4) * row_stride : U32),((row + 5) * row_stride : U32),((row + 6) * row_stride : U32),((row + 7) * row_stride : U32)}

def base0(bases: Bases) -> U32:
  Bases{a,b,c,d,e,f,g,h} = bases
  a

def base1(bases: Bases) -> U32:
  Bases{a,b,c,d,e,f,g,h} = bases
  b

def base2(bases: Bases) -> U32:
  Bases{a,b,c,d,e,f,g,h} = bases
  c

def base3(bases: Bases) -> U32:
  Bases{a,b,c,d,e,f,g,h} = bases
  d

def base4(bases: Bases) -> U32:
  Bases{a,b,c,d,e,f,g,h} = bases
  e

def base5(bases: Bases) -> U32:
  Bases{a,b,c,d,e,f,g,h} = bases
  f

def base6(bases: Bases) -> U32:
  Bases{a,b,c,d,e,f,g,h} = bases
  g

def base7(bases: Bases) -> U32:
  Bases{a,b,c,d,e,f,g,h} = bases
  h

def eighth(a: F32,b: F32,c: F32,d: F32,e: F32,f: F32,g: F32,read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (input,h) = read
  (input,Tile.Tile{a,b,c,d,e,f,g,h})

def seventh_rows(+rows: U32,+row: U32,+index: U32,+bases: Bases,a: F32,b: F32,c: F32,d: F32,e: F32,f: F32,read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (input,g) = read
  eighth(a,b,c,d,e,f,g,Storage.read_when(~F32,0.0,U32.is_lt((row + 7 : U32),rows),input,(base7(bases) + index : U32)))

def sixth_rows(+rows: U32,+row: U32,+index: U32,+bases: Bases,a: F32,b: F32,c: F32,d: F32,e: F32,read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (input,f) = read
  seventh_rows(rows,row,index,bases,a,b,c,d,e,f,Storage.read_when(~F32,0.0,U32.is_lt((row + 6 : U32),rows),input,(base6(bases) + index : U32)))

def fifth_rows(+rows: U32,+row: U32,+index: U32,+bases: Bases,a: F32,b: F32,c: F32,d: F32,read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (input,e) = read
  sixth_rows(rows,row,index,bases,a,b,c,d,e,Storage.read_when(~F32,0.0,U32.is_lt((row + 5 : U32),rows),input,(base5(bases) + index : U32)))

def fourth_rows(+rows: U32,+row: U32,+index: U32,+bases: Bases,a: F32,b: F32,c: F32,read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (input,d) = read
  fifth_rows(rows,row,index,bases,a,b,c,d,Storage.read_when(~F32,0.0,U32.is_lt((row + 4 : U32),rows),input,(base4(bases) + index : U32)))

def third_rows(+rows: U32,+row: U32,+index: U32,+bases: Bases,a: F32,b: F32,read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (input,c) = read
  fourth_rows(rows,row,index,bases,a,b,c,Storage.read_when(~F32,0.0,U32.is_lt((row + 3 : U32),rows),input,(base3(bases) + index : U32)))

def second_rows(+rows: U32,+row: U32,+index: U32,+bases: Bases,a: F32,read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (input,b) = read
  third_rows(rows,row,index,bases,a,b,Storage.read_when(~F32,0.0,U32.is_lt((row + 2 : U32),rows),input,(base2(bases) + index : U32)))

def first_rows(+rows: U32,+row: U32,+index: U32,+bases: Bases,read: Array<F32> & F32) -> Array<F32> & Tile.Tile8:
  (input,a) = read
  second_rows(rows,row,index,bases,a,Storage.read_when(~F32,0.0,U32.is_lt((row + 1 : U32),rows),input,(base1(bases) + index : U32)))

def read_rows(input: Array<F32>,+rows: U32,+row: U32,+index: U32,+bases: Bases) -> Array<F32> & Tile.Tile8:
  first_rows(rows,row,index,bases,Storage.read_when(~F32,0.0,U32.is_lt(row,rows),input,(base0(bases) + index : U32)))

type MappedRows<-Context: Data> is Data:
  MappedRows{rows: U32,row: U32,destination: U32,context: Context,bases: Bases}

def read_mapped_rows(~Context: Data,~offset: Context -> U32 -> U32,input: Array<F32>,context: MappedRows<Context>,index: U32) -> Array<F32> & Tile.Tile8:
  MappedRows{rows,row,destination,context,bases} = context
  read_rows(input,rows,row,offset(context,index),bases)

def destination_rows(~Context: Data,context: MappedRows<Context>,index: U32) -> U32:
  MappedRows{rows,row,destination,context,bases} = context
  (destination + index : U32)

def copy_mapped_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Bases,block: U32,layout: MappedLayout<Context>,owners: Array<F32> & Array<Tile.Tile8>) -> Array<F32> & Array<Tile.Tile8>:
  MappedLayout{rows,+inner,row_stride,+context} = layout
  (input,output) = owners
  +block = block
  +row = (block * 8 : U32)
  +bases = row_bases(context,row_stride,row)
  Traversal.finish(~Array<F32>,~Tile.Tile8,~MappedRows<Context>,
    Traversal.run(~Array<F32>,~Tile.Tile8,~MappedRows<Context>,~read_mapped_rows(~Context,~offset),~destination_rows(~Context),U32.to_nat(inner),
      Traversal.State{input,output,0,MappedRows{rows,row,(block * inner : U32),context,bases}}))

def source(pair: Array<F32> & Array<Tile.Tile8>) -> Array<F32>:
  (input,output) = pair
  input

def reordered(pair: Array<F32> & Array<Tile.Tile8>) -> Array<Tile.Tile8>:
  (input,output) = pair
  output

def run_mapped_with(~Context: Data,~offset: Context -> U32 -> U32,~row_bases: Context -> @+row_stride: U32 -> @+row: U32 -> Bases,+rows: U32,+inner: U32,row_stride: U32,context: Context,input: Array<F32>) -> Array<F32> & Array<Tile.Tile8>:
  +blocks = ((rows + 7) / 8 : U32)
  Traversal.indexed_fold(~(Array<F32> & Array<Tile.Tile8>),~MappedLayout<Context>,~copy_mapped_with(~Context,~offset,~row_bases),U32.to_nat(blocks),0,
    MappedLayout{rows,inner,row_stride,context},(input,Allocation.array(~Tile.Tile8,Tensors.depth_for((blocks * inner : U32)),Tile.filled(0.0))))

def run_mapped(~Context: Data,~offset: Context -> U32 -> U32,+rows: U32,+inner: U32,row_stride: U32,context: Context,input: Array<F32>) -> Array<F32> & Array<Tile.Tile8>:
  run_mapped_with(~Context,~offset,~linear_bases(~Context),rows,inner,row_stride,context,input)

def run(+rows: U32,+inner: U32,input: Array<F32>) -> Array<Tile.Tile8>:
  reordered(run_mapped(~Unit,~identity,rows,inner,inner,Unit{},input))
