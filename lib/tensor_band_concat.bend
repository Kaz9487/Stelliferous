# Concatenation: the parts meet in one plan of the rows of their joining and
# arrive in it. Every band of the output is the leaf concat of the bands of
# the parts there, along the middle axis of [outer, extent, inner]: one
# checked call per band.
import Base
import ./tensor.bend as Tensors
import ./tensor_band.bend as Bands
import ./storage_buffer.bend as Storage
import ./traversal_partition.bend as Partition
import ./tensor_band_schedule.bend as BandSchedule

# How the parts lie in bands. Along the channels of images a band holds rows
# of one image. Along any other axis the tensor is [lines, extent, inner] and
# a band holds whole lines.
type Joining is Data:
  Channels{axis: U32,batch: U32,height: U32,width: U32}
  Lines{axis: U32,lines: U32,inner: U32}

def lined(+shape: List<&2,U32>,+axis: U32) -> Joining:
  Lines{axis,Tensors.count_or_zero(List.take(&2,U32,shape,U32.to_nat(axis))),Tensors.count_or_zero(List.drop(&2,U32,shape,1n+U32.to_nat(axis)))}

# The channels of an image shape, when the axis is its channel axis.
def imaged(image: Bool,+channels: Joining,+shape: List<&2,U32>,+axis: U32) -> Joining:
  match image:
    case True{}: channels
    case False{}: lined(shape,axis)

def joining(shape: List<&2,U32>,+axis: U32) -> Joining:
  match shape:
    case [+channels,+height,+width]: imaged(U32.is_eq(axis,0),Channels{axis,1,height,width},[channels,height,width],axis)
    case [+batch,+channels,+height,+width]: imaged(U32.is_eq(axis,1),Channels{axis,batch,height,width},[batch,channels,height,width],axis)
    case +shape: lined(shape,axis)

def axis(joining: Joining) -> U32:
  match joining:
    case Channels{axis,batch,height,width}: axis
    case Lines{axis,lines,inner}: axis

# The extent of a part along the axis.
def extent(joining: Joining,shape: List<&2,U32>) -> U32:
  Tensors.known_count(List.get(&2,U32,shape,U32.to_nat(axis(joining))))

# The geometry of a tensor of an extent along the axis.
def geometry(joining: Joining,+extent: U32) -> Bands.Geometry:
  match joining:
    case Channels{axis,batch,height,width}: Bands.Geometry{batch,extent,height,width}
    case Lines{axis,lines,inner}: Bands.Geometry{1,1,lines,(extent * inner : U32)}

# A band of rows rows as [outer, extent, inner]: the rows of one image in
# every channel, or whole lines.
def band_outer(joining: Joining,+rows: U32) -> U32:
  match joining:
    case Channels{axis,batch,height,width}: 1
    case Lines{axis,lines,inner}: rows

def band_inner(joining: Joining,+rows: U32) -> U32:
  match joining:
    case Channels{axis,batch,height,width}: (rows * width : U32)
    case Lines{axis,lines,inner}: inner

def band_shape(+joining: Joining,+extent: U32,+rows: U32) -> List<&2,U32>:
  [band_outer(joining,rows),extent,band_inner(joining,rows)]

# A part offers its plan when its bands are in the geometry of the joining,
# and one band of every image otherwise.
def part_target(rows: Bool,+geometry: Bands.Geometry,+layout: Bands.Plan) -> BandSchedule.Target:
  match rows:
    case True{}: BandSchedule.Target{geometry,layout,BandSchedule.splits(layout)}
    case False{}: BandSchedule.Target{geometry,Bands.unsplit(geometry),False{}}

def part_offered(~Element: Data,+target: Bands.Geometry,+geometry: Bands.Geometry,+shape: List<&2,U32>,fill: Element,status: Tensors.Status,workers: U32,
  owners: Bands.Bands(~Element) & Bands.Plan) -> Tensors.Tensor<Element> & BandSchedule.Target:
  (bands,layout) = owners
  (Tensors.Tensor{bands,geometry,shape,fill,status,workers},part_target(Bands.same_geometry(geometry,target),target,layout))

def part_offer(~Element: Data,+joining: Joining,tensor: Tensors.Tensor<Element>) -> Tensors.Tensor<Element> & BandSchedule.Target:
  Tensors.Tensor{bands,+current,+shape,fill,status,workers} = tensor
  part_offered(~Element,geometry(joining,extent(joining,shape)),current,shape,fill,status,workers,Bands.plan(~Element,bands))

def offered_single(~Element: Data,offered: Tensors.Tensor<Element> & BandSchedule.Target) -> Tensors.Parts<Element> & BandSchedule.Target:
  (tensor,target) = offered
  (Tensors.Single{tensor},target)

def offered_cons(~Element: Data,offered: Tensors.Tensor<Element> & BandSchedule.Target,rest: Tensors.Parts<Element> & BandSchedule.Target) ->
  Tensors.Parts<Element> & BandSchedule.Target:
  (tensor,target) = offered
  (others,other_target) = rest
  (Tensors.Cons{tensor,others},BandSchedule.meet(target,other_target))

# The parts in order beside the layout they meet in.
def offers(~Element: Data,+joining: Joining,parts: Tensors.Parts<Element>) -> Tensors.Parts<Element> & BandSchedule.Target:
  match parts:
    case Tensors.Single{tensor}: offered_single(~Element,part_offer(~Element,joining,tensor))
    case Tensors.Cons{tensor,rest}: offered_cons(~Element,part_offer(~Element,joining,tensor),offers(~Element,joining,rest))

# A part in the geometry of the joining, cut by the plan the parts met in.
def arrived(~Element: Data,+joining: Joining,+layout: Bands.Plan,tensor: Tensors.Tensor<Element>) -> Tensors.Tensor<Element>:
  Tensors.Tensor{bands,+current,+shape,+fill,+status,+workers} = tensor
  +target = geometry(joining,extent(joining,shape))
  Tensors.Tensor{Tensors.arrive(~Element,target,layout,Tensors.Tensor{bands,current,shape,fill,status,workers}),target,shape,fill,status,workers}

def arrived_parts(~Element: Data,+joining: Joining,+layout: Bands.Plan,parts: Tensors.Parts<Element>) -> Tensors.Parts<Element>:
  match parts:
    case Tensors.Single{tensor}: Tensors.Single{arrived(~Element,joining,layout,tensor)}
    case Tensors.Cons{tensor,rest}: Tensors.Cons{arrived(~Element,joining,layout,tensor),arrived_parts(~Element,joining,layout,rest)}

# ---- The bands of the parts, band by band ----

# The bands of every part at one band of the output, as whole tensors of their
# band shapes in the order of the parts, and whether every part has a band of
# the same rows there.
type Gathered<-Element: Data> is Type:
  Gathered{parts: Tensors.Parts<Element>,fill: Element,valid: Bool}

def Tree(~Element: Data) -> Type:
  Partition.Tree<&1,Gathered<Element>,Unit>

def band_tensor(~Element: Data,+joining: Joining,+extent: U32,+fill: Element,+rows: U32,buffer: Storage.Buffer<Element>) -> Tensors.Tensor<Element>:
  Tensors.whole(~Element,buffer,band_shape(joining,extent,rows),fill,Tensors.Usable{},1)

def last_bands(~Element: Data,+joining: Joining,+extent: U32,+fill: Element,bands: Bands.Bands(~Element)) -> Tree(~Element):
  match bands:
    case Partition.Leaf{buffer,+first,+rows,details}:
      Partition.Leaf{Gathered{Tensors.Single{band_tensor(~Element,joining,extent,fill,rows,buffer)},fill,True{}},first,rows,Unit{}}
    case Partition.Fork{left,right}:
      Partition.Fork{last_bands(~Element,joining,extent,fill,left),last_bands(~Element,joining,extent,fill,right)}

def invalid_band(~Element: Data,gathered: Gathered<Element>,first: U32,rows: U32) -> Tree(~Element):
  Gathered{parts,fill,valid} = gathered
  Partition.Leaf{Gathered{parts,fill,False{}},first,rows,Unit{}}

def invalid(~Element: Data,gathered: Tree(~Element)) -> Tree(~Element):
  match gathered:
    case Partition.Leaf{leaf,first,rows,details}: invalid_band(~Element,leaf,first,rows)
    case Partition.Fork{left,right}: Partition.Fork{invalid(~Element,left),invalid(~Element,right)}

def joined_band(~Element: Data,+joining: Joining,+extent: U32,+fill: Element,buffer: Storage.Buffer<Element>,+first: U32,+rows: U32,
  gathered: Gathered<Element>,+other_first: U32,+other_rows: U32) -> Tree(~Element):
  Gathered{parts,other_fill,valid} = gathered
  Partition.Leaf{Gathered{Tensors.Cons{band_tensor(~Element,joining,extent,fill,rows,buffer),parts},other_fill,
    valid && U32.is_eq(first,other_first) && U32.is_eq(rows,other_rows)},first,rows,Unit{}}

# The bands of one more part in front: two trees of different forms, or two
# bands of different rows, leave the band invalid.
def joined_bands(~Element: Data,+joining: Joining,+extent: U32,+fill: Element,bands: Bands.Bands(~Element),gathered: Tree(~Element)) -> Tree(~Element):
  match bands gathered:
    case Partition.Leaf{buffer,first,rows,details} Partition.Leaf{other,other_first,other_rows,other_details}:
      joined_band(~Element,joining,extent,fill,buffer,first,rows,other,other_first,other_rows)
    case Partition.Fork{left,right} Partition.Fork{other_left,other_right}:
      Partition.Fork{joined_bands(~Element,joining,extent,fill,left,other_left),joined_bands(~Element,joining,extent,fill,right,other_right)}
    case bands gathered: invalid(~Element,gathered)

def last_part(~Element: Data,+joining: Joining,tensor: Tensors.Tensor<Element>) -> Tree(~Element):
  Tensors.Tensor{bands,geometry,+shape,+fill,status,workers} = tensor
  last_bands(~Element,joining,extent(joining,shape),fill,bands)

def joined_part(~Element: Data,+joining: Joining,tensor: Tensors.Tensor<Element>,gathered: Tree(~Element)) -> Tree(~Element):
  Tensors.Tensor{bands,geometry,+shape,+fill,status,workers} = tensor
  joined_bands(~Element,joining,extent(joining,shape),fill,bands,gathered)

# One tree whose bands hold the bands of every part, without copying values.
def gather(~Element: Data,+joining: Joining,parts: Tensors.Parts<Element>) -> Tree(~Element):
  match parts:
    case Tensors.Single{tensor}: last_part(~Element,joining,tensor)
    case Tensors.Cons{tensor,rest}: joined_part(~Element,joining,tensor,gather(~Element,joining,rest))

# ---- One checked leaf concat per band ----

type Context is Data:
  Context{joining: Joining,extent: U32}

def checked(~Element: Data,valid: Bool,+first: U32,+rows: U32,tensor: Tensors.Tensor<Element>) -> Bands.Transformed(~Element):
  Tensors.with_whole(~Element,~Bands.Transformed(~Element),tensor,storage => shape => fill => status => workers =>
    Partition.Leaf{Bands.Checked{storage,valid && Tensors.usable(status)},first,rows,Unit{}})

def band(~Element: Data,~joined: Tensors.Parts<Element> -> List<&2,U32> -> Element -> Tensors.Tensor<Element>,
  context: Context,first: U32,rows: U32,gathered: Gathered<Element>,resource: Unit) -> Bands.Transformed(~Element):
  Context{joining,extent} = context
  Gathered{parts,fill,valid} = gathered
  +rows = rows
  checked(~Element,valid,first,rows,joined(parts,band_shape(joining,extent,rows),fill))

def unjoined_band(~Element: Data,gathered: Gathered<Element>,first: U32,rows: U32) -> Bands.Transformed(~Element):
  Gathered{parts,+fill,valid} = gathered
  Partition.Leaf{Bands.Checked{Tensors.buffer(~Element,fill,0),False{}},first,rows,Unit{}}

def unjoined(~Element: Data,gathered: Tree(~Element)) -> Bands.Transformed(~Element):
  match gathered:
    case Partition.Leaf{leaf,first,rows,details}: unjoined_band(~Element,leaf,first,rows)
    case Partition.Fork{left,right}: Partition.Fork{unjoined(~Element,left),unjoined(~Element,right)}

# Every band of the output through the leaf concat joined of the bands of
# the parts, beside whether every call was usable.
def execute(~Element: Data,~joined: Tensors.Parts<Element> -> List<&2,U32> -> Element -> Tensors.Tensor<Element>,
  +layout: Bands.Plan,+joining: Joining,+extent: U32,+workers: U32,parts: Tensors.Parts<Element>) -> Bands.Bands(~Element) & Bool:
  +output = geometry(joining,extent)
  +parallel = BandSchedule.parallel(output,workers,8.0,0.0,layout)
  Bands.reduce(~Bands.Checked<Element>,~Storage.Buffer<Element>,~Bool,~Unit,~Bands.checked_band(~Element),~Bool.and,Unit{},
    Bands.each(~Gathered<Element>,~Bands.Checked<Element>,~Unit,~Context,~Bands.unshared,~band(~Element,~joined),~unjoined(~Element),
      layout,parallel,Context{joining,extent},gather(~Element,joining,parts),Unit{}))

# The bands of the output when every band call was usable and the bands
# cover the output; otherwise concat fails with the output shape.
def join(~Element: Data,~joined: Tensors.Parts<Element> -> List<&2,U32> -> Element -> Tensors.Tensor<Element>,
  +layout: Bands.Plan,+joining: Joining,+extent: U32,parts: Tensors.Parts<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  Tensors.transformed(~Element,"concat",Tensors.InvalidArgument{shape},geometry(joining,extent),shape,fill,workers,execute(~Element,~joined,layout,joining,extent,workers,parts))

# Every part arrives in the plan the parts met in, in the geometry of the
# joining, and the band concat joins them.
def met(~Element: Data,~joined: Tensors.Parts<Element> -> List<&2,U32> -> Element -> Tensors.Tensor<Element>,
  +joining: Joining,+shape: List<&2,U32>,+fill: Element,+workers: U32,offered: Tensors.Parts<Element> & BandSchedule.Target) -> Tensors.Tensor<Element>:
  (parts,target) = offered
  BandSchedule.Target{other,+layout,split} = target
  join(~Element,~joined,layout,joining,extent(joining,shape),arrived_parts(~Element,joining,layout,parts),shape,fill,workers)
