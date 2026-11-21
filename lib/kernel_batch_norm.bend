# The array loops of batch normalization on cells whose lanes are the
# channels with the groups innermost, so cell i holds group i % G. A loop
# walks one group of a piece: count cells step apart from start, all with the
# parameters of the group. The statistics read two arrays at once.
import Base
import ./kernel_tile.bend as Tile
import ./traversal_array.bend as Traversal

# One group of a piece: its parameters, its first cell, the cells between two
# of its cells and its number of cells.
type Grouped<-Group: Data> is Data:
  Grouped{group: Group,start: U32,step: U32,count: U32}

def place(~Group: Data,context: Grouped<Group>,position: U32) -> U32:
  Grouped{group,+start,+step,count} = context
  (start + position * step : U32)

def count_of(~Group: Data,context: Grouped<Group>) -> U32:
  Grouped{group,start,step,count} = context
  count

# The cells of the group read into a new array at their own places.
def mapped(~Reader: Type,~Group: Data,~read: Reader -> Grouped<Group> -> U32 -> Reader & Tile.Tile8,context: Grouped<Group>,reader: Reader,target: Array<Tile.Tile8>) ->
  Reader & Array<Tile.Tile8>:
  +context = context
  Traversal.finish(~Reader,~Tile.Tile8,~Grouped<Group>,
    Traversal.run(~Reader,~Tile.Tile8,~Grouped<Group>,~read,~place(~Group),U32.to_nat(count_of(~Group,context)),Traversal.State{reader,target,0,context}))

def squared(+tile: Tile.Tile8) -> Tile.Tile8:
  Tile.times(tile,tile)

def centered_cell(mean: Tile.Tile8,read: Array<Tile.Tile8> & Tile.Tile8) -> Array<Tile.Tile8> & Tile.Tile8:
  (input,tile) = read
  (input,squared(Tile.minus(tile,mean)))

# (x - mean) * (x - mean) of the cell at base + position * stride, the mean
# the extra of the context.
def read_centered(input: Array<Tile.Tile8>,context: Traversal.Spaced<Tile.Tile8>,position: U32) -> Array<Tile.Tile8> & Tile.Tile8:
  Traversal.Spaced{+base,+stride,mean} = context
  centered_cell(mean,Array.get(Tile.Tile8,input,(base + position * stride : U32)))

# Two arrays of one layout, read together.
def Leaves() -> Type:
  Array<Tile.Tile8> & Array<Tile.Tile8>

def product_right(left: Array<Tile.Tile8>,a: Tile.Tile8,right: Array<Tile.Tile8> & Tile.Tile8) -> Leaves() & Tile.Tile8:
  (right,b) = right
  ((left,right),Tile.times(a,b))

def product_left(right: Array<Tile.Tile8>,+index: U32,left: Array<Tile.Tile8> & Tile.Tile8) -> Leaves() & Tile.Tile8:
  (left,a) = left
  product_right(left,a,Array.get(Tile.Tile8,right,index))

# The product of the cells of the two arrays at base + position * stride,
# left times right.
def read_product(leaves: Leaves(),context: Traversal.Spaced<Unit>,position: U32) -> Leaves() & Tile.Tile8:
  (left,right) = leaves
  Traversal.Spaced{+base,+stride,extra} = context
  +index = (base + position * stride : U32)
  product_left(right,index,Array.get(Tile.Tile8,left,index))

# ---- Forward ----

# What forward reads of one group.
type Normal is Data:
  Normal{mean: Tile.Tile8,deviation: Tile.Tile8,gamma: Tile.Tile8,beta: Tile.Tile8}

# (x - mean) * deviation.
def normalized(group: Normal,tile: Tile.Tile8) -> Tile.Tile8:
  Normal{mean,deviation,gamma,beta} = group
  Tile.times(Tile.minus(tile,mean),deviation)

# ((x - mean) * deviation) * gamma + beta.
def output(+group: Normal,tile: Tile.Tile8) -> Tile.Tile8:
  Normal{mean,deviation,gamma,beta} = group
  Tile.plus(Tile.times(normalized(group,tile),gamma),beta)

def normal_cell(+group: Normal,read: Array<Tile.Tile8> & Tile.Tile8) -> Array<Tile.Tile8> & Tile.Tile8:
  (input,tile) = read
  (input,normalized(group,tile))

def output_cell(+group: Normal,read: Array<Tile.Tile8> & Tile.Tile8) -> Array<Tile.Tile8> & Tile.Tile8:
  (input,tile) = read
  (input,output(group,tile))

def read_normalized(input: Array<Tile.Tile8>,context: Grouped<Normal>,position: U32) -> Array<Tile.Tile8> & Tile.Tile8:
  Grouped{+group,+start,+step,count} = context
  normal_cell(group,Array.get(Tile.Tile8,input,(start + position * step : U32)))

def read_output(input: Array<Tile.Tile8>,context: Grouped<Normal>,position: U32) -> Array<Tile.Tile8> & Tile.Tile8:
  Grouped{+group,+start,+step,count} = context
  output_cell(group,Array.get(Tile.Tile8,input,(start + position * step : U32)))

# ---- Backward ----

# What backward reads of one group; total is the number of values of a channel.
type Back is Data:
  Back{beta: Tile.Tile8,gamma: Tile.Tile8,scale: Tile.Tile8,total: F32}

type Sampled is Data:
  Sampled{shift: Tile.Tile8,group: Back}

def back_sample(+group: Back,read: Array<Tile.Tile8> & Tile.Tile8) -> Array<Tile.Tile8> & Sampled:
  Back{beta,+gamma,scale,total} = group
  (reader,tile) = read
  (reader,Sampled{Tile.times(tile,gamma),group})

def back_read(reader: Array<Tile.Tile8>,context: Grouped<Back>,position: U32) -> Array<Tile.Tile8> & Sampled:
  Grouped{+group,+start,+step,count} = context
  back_sample(group,Array.get(Tile.Tile8,reader,(start + position * step : U32)))

# ((dy * total - beta) - xhat * gamma) * scale.
def back_combine(current: Tile.Tile8,sample: Sampled) -> Tile.Tile8:
  Sampled{shift,Back{beta,gamma,scale,+total}} = sample
  Tile.times(Tile.minus(Tile.minus(Tile.scaled(current,total),beta),shift),scale)

# dy of the group in place, xhat kept.
def updated(context: Grouped<Back>,xhat: Array<Tile.Tile8>,dy: Array<Tile.Tile8>) -> Array<Tile.Tile8> & Array<Tile.Tile8>:
  +context = context
  Traversal.finish(~Array<Tile.Tile8>,~Tile.Tile8,~Grouped<Back>,
    Traversal.update(~Array<Tile.Tile8>,~Tile.Tile8,~Sampled,~Grouped<Back>,~back_read,~back_combine,~place(~Back),U32.to_nat(count_of(~Back,context)),
      Traversal.State{xhat,dy,0,context}))
