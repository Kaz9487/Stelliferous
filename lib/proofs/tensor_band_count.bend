# Local element counts of band owners and the whole-tensor machine envelope.
import Base
import ../tensor_band.bend as Bands
import ./tensor_band_gather.bend as BandGather
import ../machine_limits.bend as Machine
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./machine_limit_bounds.bend as Limits
import ./row_major_index.bend as RowMajor
import ./storage_buffer_model.bend as Buffer
import ./tensor_shape_refinement.bend as ShapeLaws
import ./u32_comparison.bend as Comparison
import ../default_machine_limits.bend as Default
import ./boolean_logic.bend as Boolean
import ./encoded_coordinates.bend as Encoded
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./storage_allocation_refinement.bend as AllocationLaws
import ./storage_refinement.bend as StorageLaws
import ./storage_shape.bend as Shape
import ./storage_tree_model.bend as Tree
import ./tensor_allocation_depth.bend as Depth
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_count_refinement.bend as Count
import ./word_encoding.bend as WordEncoding
import ../storage_allocation.bend as Allocation
import ../storage_buffer.bend as Storage
import ../tensor.bend as Tensors
import ../tensor_window.bend as Window
import ../tensor_layout.bend as Layout
import ./tensor_count_completion.bend as CountCompletion

# The actual allocator supplies the depth and logical-length bounds consumed
# by band traversal and gather semantics, including the empty-buffer case.
def fresh_capacity(~Element: Data,+depth: Nat,+fill: Element) ->
  {Tree.capacity(Element,Tree.reflect(Element,Array.new(Element,depth,fill))) == U32.shln(1,depth) : U32}:
  %Equal.sym(Array<Element>,Array.new(Element,depth,fill),Tree.pack(Element,Tree.fresh(Element,depth,fill)),StorageLaws.fresh(Element,depth,fill)) :
    {Tree.capacity(Element,Tree.reflect(Element,_)) == U32.shln(1,depth) : U32}
  %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,Tree.fresh(Element,depth,fill))),Tree.fresh(Element,depth,fill),StorageLaws.reflect_packed(Element,Tree.fresh(Element,depth,fill))) :
    {Tree.capacity(Element,_) == U32.shln(1,depth) : U32}
  Shape.capacity(Element,depth,Tree.fresh(Element,depth,fill),Shape.fresh(Element,depth,fill))

def allocated_ready(~Element: Data,+count: U32,+fill: Element,ready: Depth.Sized<count>) ->
  Buffer.NativeReady(~Element,Bands.allocate(~Element,fill,count)):
  Depth.Sized{depth_ok,room} = ready
  %Equal.sym(Storage.Buffer<Element>,Allocation.allocate(~Element,Storage.depth_for(count),fill,U32.to_nat(count)),Storage.allocate(~Element,Storage.depth_for(count),fill,U32.to_nat(count)),
    AllocationLaws.dispatch_same(Element,Allocation.allocated(~Element),Storage.depth_for(count),fill,U32.to_nat(count))) : Buffer.NativeReady(~Element,_)
  Shape.Both{depth_ok,
    %Equal.sym(U32,Tree.capacity(Element,Tree.reflect(Element,Array.new(Element,Storage.depth_for(count),fill))),U32.shln(1,Storage.depth_for(count)),fresh_capacity(~Element,Storage.depth_for(count),fill)) :
      {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool} room}

def model(~Element: Data,+count: U32,fill: Element) -> Buffer.Model<Element>:
  Buffer.fresh(~Element,Storage.depth_for(count),fill,U32.to_nat(count))

def model_ready(~Element: Data,+count: U32,+fill: Element,ready: Depth.Sized<count>) -> Buffer.Sized<Element,model(~Element,count,fill)>:
  Depth.Sized{depth_ok,room} = ready
  Buffer.fresh_ready(~Element,Storage.depth_for(count),fill,U32.to_nat(count),depth_ok,room)

def bounded_model(~Element: Data,+count: U32,+fill: Element,within: {U32.is_le(count,1073741824) == True{} : Bool}) -> Buffer.Sized<Element,model(~Element,count,fill)>:
  model_ready(~Element,count,fill,Depth.bounded(count,within))

# Public shape counting discharges the whole-tensor machine envelope used by
# band semantics. Per-owner readiness remains a storage representation fact.
def extents(geometry: Bands.Geometry) -> List<&2,U32>:
  Bands.Geometry{images,channels,height,width} = geometry
  [images,channels,height,width]

def domain_elements(+geometry: Bands.Geometry) -> {ShapeLaws.elements(extents(geometry)) == BandGather.elements(geometry) : Nat}:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  Equal.cong(Nat,Nat,last => Nat.mul(U32.to_nat(images),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),last))),
    Nat.mul(U32.to_nat(width),1n),U32.to_nat(width),Order.times_one(U32.to_nat(width)))

def natural_room(+expected: Nat,+count: U32,+limit: U32,exact: {expected == U32.to_nat(count) : Nat},bounded: {U32.is_le(count,limit) == True{} : Bool}) ->
  {Nat.is_le(expected,U32.to_nat(limit)) == True{} : Bool}:
  %Equal.sym(Nat,expected,U32.to_nat(count),exact) : {Nat.is_le(_,U32.to_nat(limit)) == True{} : Bool}
  %Comparison.le(count,limit) : {_ == True{} : Bool}
  bounded

def bounded(+geometry: Bands.Geometry,+limits: Machine.Limits,+count: U32,
  channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool},height_positive: {BandGather.height_positive(geometry) == True{} : Bool},width_positive: {BandGather.width_positive(geometry) == True{} : Bool},
  exact: {ShapeLaws.elements(extents(geometry)) == U32.to_nat(count) : Nat},bounded: {U32.is_le(count,Machine.capacity(limits)) == True{} : Bool}) ->
  BandGather.Domain<geometry,limits,Machine.capacity(limits)>:
  +room = {
    %domain_elements(geometry) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    natural_room(ShapeLaws.elements(extents(geometry)),count,Machine.capacity(limits),exact,bounded) : {Nat.is_le(BandGather.elements(geometry),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  BandGather.Domain{Limits.capacity_fits(limits),channels_positive,height_positive,width_positive,room,room}

def evidence(+geometry: Bands.Geometry,+count: U32,
  channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool},height_positive: {BandGather.height_positive(geometry) == True{} : Bool},width_positive: {BandGather.width_positive(geometry) == True{} : Bool},
  domain_counted: Count.Count<ShapeLaws.elements(extents(geometry)),count>) ->
  BandGather.Domain<geometry,Default.limits(),Machine.capacity(Default.limits())>:
  Count.Count{exact,within} = domain_counted
  bounded(geometry,Default.limits(),count,channels_positive,height_positive,width_positive,exact,within)

def domain_counted(+geometry: Bands.Geometry,+count: U32,
  channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool},height_positive: {BandGather.height_positive(geometry) == True{} : Bool},width_positive: {BandGather.width_positive(geometry) == True{} : Bool},
  accepted: {Tensors.count_of(extents(geometry)) == Some{count} : Maybe<&2,U32>}) ->
  BandGather.Domain<geometry,Default.limits(),Machine.capacity(Default.limits())>:
  evidence(geometry,count,channels_positive,height_positive,width_positive,Count.count_of(extents(geometry),count,accepted))

# The whole-tensor envelope makes the actual native allocation count exact.
# Image count may be zero; positive spatial factors bound partial products.
def grouped(+images: U32,+channels: U32,+height: U32,+width: U32) ->
  {Nat.mul(Nat.mul(Nat.mul(U32.to_nat(images),U32.to_nat(channels)),U32.to_nat(height)),U32.to_nat(width)) ==
    BandGather.elements(Bands.Geometry{images,channels,height,width}) : Nat}:
  Equal.trans(Nat,Nat.mul(Nat.mul(Nat.mul(U32.to_nat(images),U32.to_nat(channels)),U32.to_nat(height)),U32.to_nat(width)),
    Nat.mul(Nat.mul(U32.to_nat(images),U32.to_nat(channels)),Nat.mul(U32.to_nat(height),U32.to_nat(width))),BandGather.elements(Bands.Geometry{images,channels,height,width}),
    RowMajor.assoc(Nat.mul(U32.to_nat(images),U32.to_nat(channels)),U32.to_nat(height),U32.to_nat(width)),
    RowMajor.assoc(U32.to_nat(images),U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width))))

def exact(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,domain: BandGather.Domain<geometry,limits,capacity>) ->
  {U32.to_nat(Bands.elements(geometry)) == BandGather.elements(geometry) : Nat}:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  BandGather.Domain{capacity_ok,channels_positive,+height_positive,+width_positive,+machine_room,allocation_room} = domain
  +prefix = Nat.mul(Nat.mul(U32.to_nat(images),U32.to_nat(channels)),U32.to_nat(height))
  +total = Nat.mul(prefix,U32.to_nat(width))
  +whole_room = {
    %Equal.sym(Nat,total,BandGather.elements(Bands.Geometry{images,channels,height,width}),grouped(images,channels,height,width)) :
      {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    machine_room : {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +prefix_room = Order.transitive(prefix,total,U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(prefix,U32.to_nat(width),width_positive),whole_room)
  +prefix_exact = Encoded.multiply_three(images,channels,height,Machine.capacity(limits),Limits.capacity_fits(limits),height_positive,prefix_room)
  Equal.trans(Nat,U32.to_nat(Bands.elements(Bands.Geometry{images,channels,height,width})),total,BandGather.elements(Bands.Geometry{images,channels,height,width}),
    %prefix_exact : {U32.to_nat((images * channels * height * width : U32)) == Nat.mul(_,U32.to_nat(width)) : Nat}
    Arithmetic.multiply((images * channels * height : U32),width,Machine.capacity(limits),Limits.capacity_fits(limits),
      %Equal.sym(Nat,U32.to_nat((images * channels * height : U32)),prefix,prefix_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(width)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
      whole_room),
    grouped(images,channels,height,width))

def encoded(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,domain: BandGather.Domain<geometry,limits,capacity>) ->
  {Bands.elements(geometry) == U32.from_nat(BandGather.elements(geometry)) : U32}:
  Encoded.encoding(Bands.elements(geometry),BandGather.elements(geometry),exact(geometry,limits,capacity,domain))

def room(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,domain: BandGather.Domain<geometry,limits,capacity>) ->
  {Nat.is_le(BandGather.elements(geometry),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  BandGather.Domain{capacity_ok,channels_positive,height_positive,width_positive,machine_room,allocation_room} = domain
  machine_room

def within(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,+domain: BandGather.Domain<geometry,limits,capacity>) ->
  {U32.is_le(Bands.elements(geometry),Machine.capacity(limits)) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_le(Bands.elements(geometry),Machine.capacity(limits)),Nat.is_le(U32.to_nat(Bands.elements(geometry)),U32.to_nat(Machine.capacity(limits))),Comparison.le(Bands.elements(geometry),Machine.capacity(limits))) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(Bands.elements(geometry)),BandGather.elements(geometry),exact(geometry,limits,capacity,domain)) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  room(geometry,limits,capacity,domain)

def element_count_counted(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,+count: U32,domain: BandGather.Domain<geometry,limits,capacity>,
  count_exact: {BandGather.elements(geometry) == U32.to_nat(count) : Nat}) -> {Bands.elements(geometry) == count : U32}:
  Equal.trans(U32,Bands.elements(geometry),U32.from_nat(BandGather.elements(geometry)),count,encoded(geometry,limits,capacity,domain),
    Equal.trans(U32,U32.from_nat(BandGather.elements(geometry)),U32.from_nat(U32.to_nat(count)),count,
      Equal.cong(Nat,U32,U32.from_nat,BandGather.elements(geometry),U32.to_nat(count),count_exact),Arithmetic.native_roundtrip(count)))

# The actual public band-shape guard retains every CHW/NCHW axis. Equal
# element products alone cannot establish that two band geometries agree.
def decode(shape: List<&2,U32>) -> Maybe<&2,Bands.Geometry>:
  match shape:
    case [channels,height,width]: Some{Bands.Geometry{1,channels,height,width}}
    case [images,channels,height,width]: Some{Bands.Geometry{images,channels,height,width}}
    case shape: None{}

def decoded(+shape: List<&2,U32>,+geometry: Bands.Geometry,+matched: {Bands.image_rows(shape,geometry) == True{} : Bool}) ->
  {decode(shape) == Some{geometry} : Maybe<&2,Bands.Geometry>}:
  match shape geometry:
    case Con{+c,Con{+h,Con{+w,Nil{}}}} Bands.Geometry{+images,+channels,+height,+width}:
      +prefix = Boolean.left((U32.is_eq(images,1) && U32.is_eq(c,channels) && U32.is_eq(h,height) : Bool),U32.is_eq(w,width),matched)
      +pair = Boolean.left((U32.is_eq(images,1) && U32.is_eq(c,channels) : Bool),U32.is_eq(h,height),prefix)
      %Equal.sym(U32,images,1,Arithmetic.equal_of_is_eq(images,1,Boolean.left(U32.is_eq(images,1),U32.is_eq(c,channels),pair))) :
        {Some{Bands.Geometry{1,c,h,w}} == Some{Bands.Geometry{_,channels,height,width}} : Maybe<&2,Bands.Geometry>}
      %Equal.sym(U32,c,channels,Arithmetic.equal_of_is_eq(c,channels,Boolean.right(U32.is_eq(images,1),U32.is_eq(c,channels),pair))) :
        {Some{Bands.Geometry{1,_,h,w}} == Some{Bands.Geometry{1,channels,height,width}} : Maybe<&2,Bands.Geometry>}
      %Equal.sym(U32,h,height,Arithmetic.equal_of_is_eq(h,height,Boolean.right((U32.is_eq(images,1) && U32.is_eq(c,channels) : Bool),U32.is_eq(h,height),prefix))) :
        {Some{Bands.Geometry{1,channels,_,w}} == Some{Bands.Geometry{1,channels,height,width}} : Maybe<&2,Bands.Geometry>}
      %Equal.sym(U32,w,width,Arithmetic.equal_of_is_eq(w,width,Boolean.right((U32.is_eq(images,1) && U32.is_eq(c,channels) && U32.is_eq(h,height) : Bool),U32.is_eq(w,width),matched))) :
        {Some{Bands.Geometry{1,channels,height,_}} == Some{Bands.Geometry{1,channels,height,width}} : Maybe<&2,Bands.Geometry>}
      {==}
    case Con{+n,Con{+c,Con{+h,Con{+w,Nil{}}}}} Bands.Geometry{+images,+channels,+height,+width}:
      +prefix = Boolean.left((U32.is_eq(n,images) && U32.is_eq(c,channels) && U32.is_eq(h,height) : Bool),U32.is_eq(w,width),matched)
      +pair = Boolean.left((U32.is_eq(n,images) && U32.is_eq(c,channels) : Bool),U32.is_eq(h,height),prefix)
      %Equal.sym(U32,n,images,Arithmetic.equal_of_is_eq(n,images,Boolean.left(U32.is_eq(n,images),U32.is_eq(c,channels),pair))) :
        {Some{Bands.Geometry{_,c,h,w}} == Some{geometry} : Maybe<&2,Bands.Geometry>}
      %Equal.sym(U32,c,channels,Arithmetic.equal_of_is_eq(c,channels,Boolean.right(U32.is_eq(n,images),U32.is_eq(c,channels),pair))) :
        {Some{Bands.Geometry{images,_,h,w}} == Some{geometry} : Maybe<&2,Bands.Geometry>}
      %Equal.sym(U32,h,height,Arithmetic.equal_of_is_eq(h,height,Boolean.right((U32.is_eq(n,images) && U32.is_eq(c,channels) : Bool),U32.is_eq(h,height),prefix))) :
        {Some{Bands.Geometry{images,channels,_,w}} == Some{geometry} : Maybe<&2,Bands.Geometry>}
      %Equal.sym(U32,w,width,Arithmetic.equal_of_is_eq(w,width,Boolean.right((U32.is_eq(n,images) && U32.is_eq(c,channels) && U32.is_eq(h,height) : Bool),U32.is_eq(w,width),matched))) :
        {Some{Bands.Geometry{images,channels,height,_}} == Some{geometry} : Maybe<&2,Bands.Geometry>}
      {==}
    case Nil{} Bands.Geometry{n,c,h,w}: Empty.absurd({decode(shape) == Some{geometry} : Maybe<&2,Bands.Geometry>},WordEncoding.false_true(matched))
    case Con{a,Nil{}} Bands.Geometry{n,c,h,w}: Empty.absurd({decode(shape) == Some{geometry} : Maybe<&2,Bands.Geometry>},WordEncoding.false_true(matched))
    case Con{a,Con{b,Nil{}}} Bands.Geometry{n,c,h,w}: Empty.absurd({decode(shape) == Some{geometry} : Maybe<&2,Bands.Geometry>},WordEncoding.false_true(matched))
    case Con{a,Con{b,Con{c,Con{d,Con{e,rest}}}}} Bands.Geometry{n,channels,h,w}: Empty.absurd({decode(shape) == Some{geometry} : Maybe<&2,Bands.Geometry>},WordEncoding.false_true(matched))

def ElementProperty(+shape: List<&2,U32>,result: Maybe<&2,Bands.Geometry>) -> Data:
  match result:
    case None{}: Unit
    case Some{geometry}: {ShapeLaws.elements(shape) == BandGather.elements(geometry) : Nat}

def decoded_elements(+shape: List<&2,U32>) -> ElementProperty(shape,decode(shape)):
  match shape:
    case Con{+c,Con{+h,Con{+w,Nil{}}}}:
      +shape_coherence_elements = Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(h),U32.to_nat(w)))
      Equal.trans(Nat,ShapeLaws.elements([c,h,w]),shape_coherence_elements,BandGather.elements(Bands.Geometry{1,c,h,w}),
        Equal.cong(Nat,Nat,last => Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(h),last)),Nat.mul(U32.to_nat(w),1n),U32.to_nat(w),Order.times_one(U32.to_nat(w))),
        Equal.sym(Nat,BandGather.elements(Bands.Geometry{1,c,h,w}),shape_coherence_elements,Addition.zero_right(shape_coherence_elements)))
    case Con{+n,Con{+c,Con{+h,Con{+w,Nil{}}}}}: domain_elements(Bands.Geometry{n,c,h,w})
    case Nil{}: Unit{}
    case Con{a,Nil{}}: Unit{}
    case Con{a,Con{b,Nil{}}}: Unit{}
    case Con{a,Con{b,Con{c,Con{d,Con{e,rest}}}}}: Unit{}

def shape_coherence_elements(+shape: List<&2,U32>,+geometry: Bands.Geometry,matched: {Bands.image_rows(shape,geometry) == True{} : Bool}) ->
  {ShapeLaws.elements(shape) == BandGather.elements(geometry) : Nat}:
  %decoded(shape,geometry,matched) : ElementProperty(shape,_)
  decoded_elements(shape)

def value(result: Maybe<&2,Bands.Geometry>) -> Bands.Geometry:
  match result:
    case None{}: Bands.Geometry{0,0,0,0}
    case Some{geometry}: geometry

def unique(+shape: List<&2,U32>,+left: Bands.Geometry,+right: Bands.Geometry,
  left_matched: {Bands.image_rows(shape,left) == True{} : Bool},right_matched: {Bands.image_rows(shape,right) == True{} : Bool}) -> {left == right : Bands.Geometry}:
  Equal.cong(Maybe<&2,Bands.Geometry>,Bands.Geometry,value,Some{left},Some{right},
    Equal.trans(Maybe<&2,Bands.Geometry>,Some{left},decode(shape),Some{right},Equal.sym(Maybe<&2,Bands.Geometry>,decode(shape),Some{left},decoded(shape,left,left_matched)),decoded(shape,right,right_matched)))

# ---- The planes of a shape ----

# A shape of rank two or more whose last two axes are the rows and columns of
# the geometry and whose axes before them hold its images times its channels.
def planes_rows(+shape: List<&2,U32>,+geometry: Bands.Geometry) -> Bool:
  Bands.Geometry{images,channels,+height,+width} = geometry
  +rank = List.length(&2,U32,shape)
  Nat.is_le(2n,rank) && U32.is_eq(Layout.extent_at(shape,Nat.sub(rank,2n)),height) && U32.is_eq(Layout.extent_at(shape,Nat.sub(rank,1n)),width) &&
    Nat.is_eq(ShapeLaws.elements(List.take(&2,U32,shape,Nat.sub(rank,2n))),Nat.mul(U32.to_nat(images),U32.to_nat(channels)))

type Planes<-shape: List<&2,U32>,-images: U32,-channels: U32,-height: U32,-width: U32> is Data:
  Planes{ranked: {Nat.is_le(2n,List.length(&2,U32,shape)) == True{} : Bool},
    rows: {Layout.extent_at(shape,Nat.sub(List.length(&2,U32,shape),2n)) == height : U32},
    columns: {Layout.extent_at(shape,Nat.sub(List.length(&2,U32,shape),1n)) == width : U32},
    leading: {ShapeLaws.elements(List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),2n))) == Nat.mul(U32.to_nat(images),U32.to_nat(channels)) : Nat}}

def decoded_planes(+shape: List<&2,U32>,+images: U32,+channels: U32,+height: U32,+width: U32,
  +matched: {planes_rows(shape,Bands.Geometry{images,channels,height,width}) == True{} : Bool}) -> Planes<shape,images,channels,height,width>:
  +rank = List.length(&2,U32,shape)
  +ranked = Nat.is_le(2n,rank)
  +rows = U32.is_eq(Layout.extent_at(shape,Nat.sub(rank,2n)),height)
  +columns = U32.is_eq(Layout.extent_at(shape,Nat.sub(rank,1n)),width)
  +leading = Nat.is_eq(ShapeLaws.elements(List.take(&2,U32,shape,Nat.sub(rank,2n))),Nat.mul(U32.to_nat(images),U32.to_nat(channels)))
  +prefix = Boolean.left((ranked && rows && columns : Bool),leading,matched)
  +pair = Boolean.left((ranked && rows : Bool),columns,prefix)
  Planes{Boolean.left(ranked,rows,pair),
    Arithmetic.equal_of_is_eq(Layout.extent_at(shape,Nat.sub(rank,2n)),height,Boolean.right(ranked,rows,pair)),
    Arithmetic.equal_of_is_eq(Layout.extent_at(shape,Nat.sub(rank,1n)),width,Boolean.right((ranked && rows : Bool),columns,prefix)),
    Order.equal_of_is_eq(ShapeLaws.elements(List.take(&2,U32,shape,Nat.sub(rank,2n))),Nat.mul(U32.to_nat(images),U32.to_nat(channels)),
      Boolean.right((ranked && rows && columns : Bool),leading,matched))}

# A shape of rank two or more ends with its last two axes.
def last_two(+shape: List<&2,U32>,ranked: {Nat.is_le(2n,List.length(&2,U32,shape)) == True{} : Bool}) ->
  {List.drop(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),2n)) ==
    [Layout.extent_at(shape,Nat.sub(List.length(&2,U32,shape),2n)),Layout.extent_at(shape,Nat.sub(List.length(&2,U32,shape),1n))] : List<&2,U32>}:
  match shape:
    case []: Empty.absurd({List.drop(&2,U32,[],0n) == [0,0] : List<&2,U32>},WordEncoding.false_true(ranked))
    case [+a]: Empty.absurd({List.drop(&2,U32,[a],0n) == [a,a] : List<&2,U32>},WordEncoding.false_true(ranked))
    case [+a,+b]: {==}
    case a <> +b <> +c <> +rest:
      +remaining = List.length(&2,U32,rest)
      %Order.subtract_zero(remaining) :
        {List.drop(&2,U32,b <> c <> rest,_) == [Layout.extent_at(b <> c <> rest,_),Layout.extent_at(b <> c <> rest,1n+remaining)] : List<&2,U32>}
      last_two(b <> c <> rest,Order.zero_le(remaining))

# The cells of the shape with its last two axes replaced: the images and
# channels of the geometry and the new rows and columns.
def planes_elements(+shape: List<&2,U32>,+images: U32,+channels: U32,+height: U32,+width: U32,+rows: U32,+columns: U32,
  planes: Planes<shape,images,channels,height,width>) ->
  {ShapeLaws.elements(List.append(&2,U32,List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),2n)),[rows,columns])) ==
    BandGather.elements(Bands.Geometry{images,channels,rows,columns}) : Nat}:
  Planes{ranked,height_equal,width_equal,+leading} = planes
  +leading_axes = List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),2n))
  +plane = ShapeLaws.elements([rows,columns])
  Equal.trans(Nat,ShapeLaws.elements(List.append(&2,U32,leading_axes,[rows,columns])),Nat.mul(ShapeLaws.elements(leading_axes),plane),
    BandGather.elements(Bands.Geometry{images,channels,rows,columns}),
    ShapeLaws.elements_appended(leading_axes,[rows,columns]),
    %Equal.sym(Nat,ShapeLaws.elements(leading_axes),Nat.mul(U32.to_nat(images),U32.to_nat(channels)),leading) :
      {Nat.mul(_,plane) == BandGather.elements(Bands.Geometry{images,channels,rows,columns}) : Nat}
    Equal.trans(Nat,Nat.mul(Nat.mul(U32.to_nat(images),U32.to_nat(channels)),plane),Nat.mul(U32.to_nat(images),Nat.mul(U32.to_nat(channels),plane)),
      BandGather.elements(Bands.Geometry{images,channels,rows,columns}),RowMajor.assoc(U32.to_nat(images),U32.to_nat(channels),plane),
      Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(images),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),value))),Nat.mul(U32.to_nat(columns),1n),U32.to_nat(columns),
        Order.times_one(U32.to_nat(columns)))))

# A shape of planes has the cells of the geometry.
def planes_coherent(+shape: List<&2,U32>,+images: U32,+channels: U32,+height: U32,+width: U32,planes: Planes<shape,images,channels,height,width>) ->
  {ShapeLaws.elements(shape) == BandGather.elements(Bands.Geometry{images,channels,height,width}) : Nat}:
  Planes{+ranked,+height_equal,+width_equal,+leading} = planes
  +axis = Nat.sub(List.length(&2,U32,shape),2n)
  +leading_axes = List.take(&2,U32,shape,axis)
  +rows = Layout.extent_at(shape,axis)
  +columns = Layout.extent_at(shape,Nat.sub(List.length(&2,U32,shape),1n))
  +ends = Equal.trans(List<&2,U32>,List.drop(&2,U32,shape,axis),[rows,columns],[height,width],last_two(shape,ranked),
    %Equal.sym(U32,rows,height,height_equal) : {[_,columns] == [height,width] : List<&2,U32>}
    %Equal.sym(U32,columns,width,width_equal) : {[height,_] == [height,width] : List<&2,U32>}
    {==})
  Equal.trans(Nat,ShapeLaws.elements(shape),Nat.mul(ShapeLaws.elements(leading_axes),ShapeLaws.elements(List.drop(&2,U32,shape,axis))),
    BandGather.elements(Bands.Geometry{images,channels,height,width}),CountCompletion.elements_split(shape,axis),
    Equal.trans(Nat,Nat.mul(ShapeLaws.elements(leading_axes),ShapeLaws.elements(List.drop(&2,U32,shape,axis))),Nat.mul(ShapeLaws.elements(leading_axes),ShapeLaws.elements([height,width])),
      BandGather.elements(Bands.Geometry{images,channels,height,width}),
      Equal.cong(List<&2,U32>,Nat,value => Nat.mul(ShapeLaws.elements(leading_axes),ShapeLaws.elements(value)),List.drop(&2,U32,shape,axis),[height,width],ends),
      Equal.trans(Nat,Nat.mul(ShapeLaws.elements(leading_axes),ShapeLaws.elements([height,width])),ShapeLaws.elements(List.append(&2,U32,leading_axes,[height,width])),
        BandGather.elements(Bands.Geometry{images,channels,height,width}),
        Equal.sym(Nat,ShapeLaws.elements(List.append(&2,U32,leading_axes,[height,width])),Nat.mul(ShapeLaws.elements(leading_axes),ShapeLaws.elements([height,width])),
          ShapeLaws.elements_appended(leading_axes,[height,width])),
        planes_elements(shape,images,channels,height,width,height,width,Planes{ranked,height_equal,width_equal,leading}))))

# Bands of the image rows of a CHW or NCHW shape hold its planes.
def image_planes(+shape: List<&2,U32>,+images: U32,+channels: U32,+height: U32,+width: U32,
  +matched: {Bands.image_rows(shape,Bands.Geometry{images,channels,height,width}) == True{} : Bool}) ->
  {planes_rows(shape,Bands.Geometry{images,channels,height,width}) == True{} : Bool}:
  match shape:
    case [+c,+h,+w]:
      +prefix = Boolean.left((U32.is_eq(images,1) && U32.is_eq(c,channels) && U32.is_eq(h,height) : Bool),U32.is_eq(w,width),matched)
      +pair = Boolean.left((U32.is_eq(images,1) && U32.is_eq(c,channels) : Bool),U32.is_eq(h,height),prefix)
      +images_equal = Arithmetic.equal_of_is_eq(images,1,Boolean.left(U32.is_eq(images,1),U32.is_eq(c,channels),pair))
      +channels_equal = Arithmetic.equal_of_is_eq(c,channels,Boolean.right(U32.is_eq(images,1),U32.is_eq(c,channels),pair))
      +cells = U32.to_nat(channels)
      +leading = {
        %Equal.sym(U32,images,1,images_equal) : {Nat.is_eq(Nat.mul(U32.to_nat(c),1n),Nat.mul(U32.to_nat(_),cells)) == True{} : Bool}
        %Equal.sym(U32,c,channels,channels_equal) : {Nat.is_eq(Nat.mul(U32.to_nat(_),1n),Nat.mul(1n,cells)) == True{} : Bool}
        %Equal.sym(Nat,Nat.mul(cells,1n),cells,Order.times_one(cells)) : {Nat.is_eq(_,Nat.mul(1n,cells)) == True{} : Bool}
        %Equal.sym(Nat,Nat.mul(1n,cells),cells,Addition.zero_right(cells)) : {Nat.is_eq(cells,_) == True{} : Bool}
        Order.equal_reflexive(cells) :
        {Nat.is_eq(Nat.mul(U32.to_nat(c),1n),Nat.mul(U32.to_nat(images),cells)) == True{} : Bool}}
      Boolean.and_true((True{} && U32.is_eq(h,height) && U32.is_eq(w,width) : Bool),Nat.is_eq(Nat.mul(U32.to_nat(c),1n),Nat.mul(U32.to_nat(images),cells)),
        Boolean.and_true((True{} && U32.is_eq(h,height) : Bool),U32.is_eq(w,width),
          Boolean.and_true(True{},U32.is_eq(h,height),{==},Boolean.right((U32.is_eq(images,1) && U32.is_eq(c,channels) : Bool),U32.is_eq(h,height),prefix)),
          Boolean.right((U32.is_eq(images,1) && U32.is_eq(c,channels) && U32.is_eq(h,height) : Bool),U32.is_eq(w,width),matched)),
        leading)
    case [+n,+c,+h,+w]:
      +prefix = Boolean.left((U32.is_eq(n,images) && U32.is_eq(c,channels) && U32.is_eq(h,height) : Bool),U32.is_eq(w,width),matched)
      +pair = Boolean.left((U32.is_eq(n,images) && U32.is_eq(c,channels) : Bool),U32.is_eq(h,height),prefix)
      +images_equal = Arithmetic.equal_of_is_eq(n,images,Boolean.left(U32.is_eq(n,images),U32.is_eq(c,channels),pair))
      +channels_equal = Arithmetic.equal_of_is_eq(c,channels,Boolean.right(U32.is_eq(n,images),U32.is_eq(c,channels),pair))
      +cells = Nat.mul(U32.to_nat(images),U32.to_nat(channels))
      +leading = {
        %Equal.sym(U32,n,images,images_equal) : {Nat.is_eq(Nat.mul(U32.to_nat(_),Nat.mul(U32.to_nat(c),1n)),cells) == True{} : Bool}
        %Equal.sym(U32,c,channels,channels_equal) : {Nat.is_eq(Nat.mul(U32.to_nat(images),Nat.mul(U32.to_nat(_),1n)),cells) == True{} : Bool}
        %Equal.sym(Nat,Nat.mul(U32.to_nat(channels),1n),U32.to_nat(channels),Order.times_one(U32.to_nat(channels))) :
          {Nat.is_eq(Nat.mul(U32.to_nat(images),_),cells) == True{} : Bool}
        Order.equal_reflexive(cells) :
        {Nat.is_eq(Nat.mul(U32.to_nat(n),Nat.mul(U32.to_nat(c),1n)),cells) == True{} : Bool}}
      Boolean.and_true((True{} && U32.is_eq(h,height) && U32.is_eq(w,width) : Bool),Nat.is_eq(Nat.mul(U32.to_nat(n),Nat.mul(U32.to_nat(c),1n)),cells),
        Boolean.and_true((True{} && U32.is_eq(h,height) : Bool),U32.is_eq(w,width),
          Boolean.and_true(True{},U32.is_eq(h,height),{==},Boolean.right((U32.is_eq(n,images) && U32.is_eq(c,channels) : Bool),U32.is_eq(h,height),prefix)),
          Boolean.right((U32.is_eq(n,images) && U32.is_eq(c,channels) && U32.is_eq(h,height) : Bool),U32.is_eq(w,width),matched)),
        leading)
    case Nil{}: Empty.absurd({planes_rows(shape,Bands.Geometry{images,channels,height,width}) == True{} : Bool},WordEncoding.false_true(matched))
    case Con{a,Nil{}}: Empty.absurd({planes_rows(shape,Bands.Geometry{images,channels,height,width}) == True{} : Bool},WordEncoding.false_true(matched))
    case Con{a,Con{b,Nil{}}}: Empty.absurd({planes_rows(shape,Bands.Geometry{images,channels,height,width}) == True{} : Bool},WordEncoding.false_true(matched))
    case Con{a,Con{b,Con{c,Con{d,Con{e,rest}}}}}:
      Empty.absurd({planes_rows(shape,Bands.Geometry{images,channels,height,width}) == True{} : Bool},WordEncoding.false_true(matched))

def ranked_planes(ranked: Bool,+shape: List<&2,U32>,+planes: U32,+compared: {Nat.is_le(2n,List.length(&2,U32,shape)) == ranked : Bool},
  counted: {Layout.leading(ranked,shape) == Some{planes} : Maybe<&2,U32>}) ->
  {planes_rows(shape,Bands.Geometry{1,planes,Layout.extent_at(shape,Nat.sub(List.length(&2,U32,shape),2n)),Layout.extent_at(shape,Nat.sub(List.length(&2,U32,shape),1n))}) == True{} : Bool}:
  match ranked:
    case False{}:
      Empty.absurd({planes_rows(shape,Bands.Geometry{1,planes,Layout.extent_at(shape,Nat.sub(List.length(&2,U32,shape),2n)),
        Layout.extent_at(shape,Nat.sub(List.length(&2,U32,shape),1n))}) == True{} : Bool},
        WordEncoding.false_true(Equal.cong(Maybe<&2,U32>,Bool,result => Maybe.is_some(&2,U32,result),None{},Some{planes},counted)))
    case True{}:
      +rows = Layout.extent_at(shape,Nat.sub(List.length(&2,U32,shape),2n))
      +columns = Layout.extent_at(shape,Nat.sub(List.length(&2,U32,shape),1n))
      +leading_axes = List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),2n))
      +cells = U32.to_nat(planes)
      +exact = {%counted : ShapeLaws.Count(_,ShapeLaws.elements(leading_axes)) ShapeLaws.nonempty_count(leading_axes) : {ShapeLaws.elements(leading_axes) == cells : Nat}}
      +leading = {
        %Equal.sym(Nat,ShapeLaws.elements(leading_axes),cells,exact) : {Nat.is_eq(_,Nat.mul(1n,cells)) == True{} : Bool}
        %Equal.sym(Nat,Nat.mul(1n,cells),cells,Addition.zero_right(cells)) : {Nat.is_eq(cells,_) == True{} : Bool}
        Order.equal_reflexive(cells) :
        {Nat.is_eq(ShapeLaws.elements(leading_axes),Nat.mul(1n,cells)) == True{} : Bool}}
      Boolean.and_true((Nat.is_le(2n,List.length(&2,U32,shape)) && U32.is_eq(rows,rows) && U32.is_eq(columns,columns) : Bool),Nat.is_eq(ShapeLaws.elements(leading_axes),Nat.mul(1n,cells)),
        Boolean.and_true((Nat.is_le(2n,List.length(&2,U32,shape)) && U32.is_eq(rows,rows) : Bool),U32.is_eq(columns,columns),
          Boolean.and_true(Nat.is_le(2n,List.length(&2,U32,shape)),U32.is_eq(rows,rows),compared,Arithmetic.equal_reflexive(rows)),Arithmetic.equal_reflexive(columns)),
        leading)

# The planes of a shape of rank two or more are the channels of one image.
def plane_planes(+shape: List<&2,U32>,+planes: U32,counted: {Layout.planes(shape) == Some{planes} : Maybe<&2,U32>}) ->
  {planes_rows(shape,Bands.Geometry{1,planes,Layout.extent_at(shape,Nat.sub(List.length(&2,U32,shape),2n)),Layout.extent_at(shape,Nat.sub(List.length(&2,U32,shape),1n))}) == True{} : Bool}:
  ranked_planes(Nat.is_le(2n,List.length(&2,U32,shape)),shape,planes,{==},counted)

def same_geometry_self(+geometry: Bands.Geometry) -> {Bands.same_geometry(geometry,geometry) == True{} : Bool}:
  match geometry:
    case Bands.Geometry{+images,+channels,+height,+width}:
      %Equal.sym(Bool,U32.is_eq(images,images),True{},Arithmetic.equal_reflexive(images)) :
        {_ && U32.is_eq(channels,channels) && U32.is_eq(height,height) && U32.is_eq(width,width) == True{} : Bool}
      %Equal.sym(Bool,U32.is_eq(channels,channels),True{},Arithmetic.equal_reflexive(channels)) :
        {True{} && _ && U32.is_eq(height,height) && U32.is_eq(width,width) == True{} : Bool}
      %Equal.sym(Bool,U32.is_eq(height,height),True{},Arithmetic.equal_reflexive(height)) :
        {True{} && True{} && _ && U32.is_eq(width,width) == True{} : Bool}
      %Equal.sym(Bool,U32.is_eq(width,width),True{},Arithmetic.equal_reflexive(width)) :
        {True{} && True{} && True{} && _ == True{} : Bool}
      {==}

def same_geometry_equal(+left: Bands.Geometry,+right: Bands.Geometry,+same: {Bands.same_geometry(left,right) == True{} : Bool}) -> {left == right : Bands.Geometry}:
  match left right:
    case Bands.Geometry{+images,+channels,+height,+width} Bands.Geometry{+other_images,+other_channels,+other_height,+other_width}:
      +prefix = Boolean.left((U32.is_eq(images,other_images) && U32.is_eq(channels,other_channels) && U32.is_eq(height,other_height) : Bool),U32.is_eq(width,other_width),same)
      +pair = Boolean.left((U32.is_eq(images,other_images) && U32.is_eq(channels,other_channels) : Bool),U32.is_eq(height,other_height),prefix)
      %Arithmetic.equal_of_is_eq(images,other_images,Boolean.left(U32.is_eq(images,other_images),U32.is_eq(channels,other_channels),pair)) :
        {Bands.Geometry{images,channels,height,width} == Bands.Geometry{_,other_channels,other_height,other_width} : Bands.Geometry}
      %Arithmetic.equal_of_is_eq(channels,other_channels,Boolean.right(U32.is_eq(images,other_images),U32.is_eq(channels,other_channels),pair)) :
        {Bands.Geometry{images,channels,height,width} == Bands.Geometry{images,_,other_height,other_width} : Bands.Geometry}
      %Arithmetic.equal_of_is_eq(height,other_height,Boolean.right((U32.is_eq(images,other_images) && U32.is_eq(channels,other_channels) : Bool),U32.is_eq(height,other_height),prefix)) :
        {Bands.Geometry{images,channels,height,width} == Bands.Geometry{images,channels,_,other_width} : Bands.Geometry}
      %Arithmetic.equal_of_is_eq(width,other_width,Boolean.right((U32.is_eq(images,other_images) && U32.is_eq(channels,other_channels) && U32.is_eq(height,other_height) : Bool),U32.is_eq(width,other_width),same)) :
        {Bands.Geometry{images,channels,height,width} == Bands.Geometry{images,channels,height,_} : Bands.Geometry}
      {==}

def width_native(+images: U32,+channels: U32,+height: U32,+rows: U32) ->
  {Bands.local_count(Bands.Geometry{images,channels,height,0},rows) == 0 : U32}:
  Arithmetic.multiply_zero_right((channels * rows : U32))

def channels_native(+images: U32,+height: U32,+width: U32,+rows: U32) ->
  {Bands.local_count(Bands.Geometry{images,0,height,width},rows) == 0 : U32}:
  %Equal.sym(U32,(0 * rows : U32),0,Arithmetic.multiply_zero_left(rows)) : {(_ * width : U32) == 0 : U32}
  Arithmetic.multiply_zero_left(width)

def from_zero(+geometry: Bands.Geometry,+rows: U32,+length: Nat,
  +native_zero: {Bands.local_count(geometry,rows) == 0 : U32},length_zero: {length == 0n : Nat}) -> BandShape.Count<geometry,rows,length>:
  BandShape.Count{
    Equal.trans(Nat,U32.to_nat(Bands.local_count(geometry,rows)),0n,length,
      Equal.cong(U32,Nat,U32.to_nat,Bands.local_count(geometry,rows),0,native_zero),Equal.sym(Nat,length,0n,length_zero)),
    %Equal.sym(U32,Bands.local_count(geometry,rows),0,native_zero) : {U32.is_le(_,1073741824) == True{} : Bool}
    {==}}

def width(~Element: Data,+images: U32,+channels: U32,+height: U32,+first: U32,+rows: U32,+input: Buffer.Model<Element>,
  metadata: BandShape.Leaf(Bands.Geometry{images,channels,height,0},first,rows,Buffer.length(~Element,input))) ->
  BandShape.Count<Bands.Geometry{images,channels,height,0},rows,Buffer.length(~Element,input)>:
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  from_zero(Bands.Geometry{images,channels,height,0},rows,Buffer.length(~Element,input),width_native(images,channels,height,rows),
    Equal.trans(Nat,Buffer.length(~Element,input),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),0n)),0n,length_exact,
      Equal.trans(Nat,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),0n)),Nat.mul(U32.to_nat(channels),0n),0n,
        Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(channels),value),Nat.mul(U32.to_nat(rows),0n),0n,RowMajor.multiply_zero_right(U32.to_nat(rows))),
        RowMajor.multiply_zero_right(U32.to_nat(channels)))))

def channels(~Element: Data,+images: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+input: Buffer.Model<Element>,
  metadata: BandShape.Leaf(Bands.Geometry{images,0,height,width},first,rows,Buffer.length(~Element,input))) ->
  BandShape.Count<Bands.Geometry{images,0,height,width},rows,Buffer.length(~Element,input)>:
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  from_zero(Bands.Geometry{images,0,height,width},rows,Buffer.length(~Element,input),channels_native(images,height,width,rows),length_exact)
