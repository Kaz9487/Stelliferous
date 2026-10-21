# A view of the storage of a geometry, seen from its bands. A view that keeps
# the images and the rows of the geometry in place has axes before the rows,
# repeats of each row and axes after the rows, each moving by whole channels
# and columns; every band reads its cells through the same axes over its own
# rows.
import Base
import ./tensor_view.bend as Views
import ./tensor_shape.bend as Shapes
import ./tensor_band.bend as Bands

# The axes before the rows, the repeats of each row, the axes after the rows,
# and the offset of the view, as the view gives them.
type Banded is Data:
  Banded{before: List<&2,Views.Axis>,repeats: List<&2,U32>,after: List<&2,Views.Axis>,offset: U32}

# ---- The view of a band ----

# The channel of plane words a word of the storage of a geometry lies in, or
# its column within that channel.
def component(+plane: U32,column: Bool,+word: U32) -> U32:
  match column:
    case False{}: (word / plane : U32)
    case True{}: (word % plane : U32)

# A word of the storage of the geometry, a channel of plane words and a column
# within it, as the same channel and column in a band of band words per
# channel.
def localized(+plane: U32,+band: U32,+word: U32) -> U32:
  (component(plane,False{},word) * band + component(plane,True{},word) : U32)

def localized_axes(+plane: U32,+band: U32,axes: List<&2,Views.Axis>) -> List<&2,Views.Axis>:
  match axes:
    case []: []
    case Views.Axis{extent,stride} <> rest: Views.Axis{extent,localized(plane,band,stride)} <> localized_axes(plane,band,rest)

def repeated(repeats: List<&2,U32>) -> List<&2,Views.Axis>:
  match repeats:
    case []: []
    case extent <> rest: Views.Axis{extent,0} <> repeated(rest)

# The rows axis, which a view of a geometry of one row per image leaves out,
# and the repeats of each row.
def rows_axes(rowed: Bool,+rows: U32,+width: U32,repeats: List<&2,U32>) -> List<&2,Views.Axis>:
  match rowed:
    case True{}: Views.Axis{rows,width} <> repeated(repeats)
    case False{}: repeated(repeats)

# The axes of the view over rows rows of each channel. A band of a geometry
# of one row per image has at most one row.
def axes(+banded: Banded,+geometry: Bands.Geometry,+rows: U32) -> List<&2,Views.Axis>:
  Banded{before,repeats,after,offset} = banded
  Bands.Geometry{batch,channels,+height,+width} = geometry
  +plane = (height * width : U32)
  +band = (rows * width : U32)
  List.append(&2,Views.Axis,localized_axes(plane,band,before),List.append(&2,Views.Axis,rows_axes(True{},rows,width,repeats),localized_axes(plane,band,after)))

# The view of the local storage of a band of rows rows.
def local(+banded: Banded,+geometry: Bands.Geometry,+rows: U32) -> Views.View:
  Banded{before,repeats,after,+offset} = banded
  Bands.Geometry{batch,channels,+height,+width} = geometry
  Views.View{axes(banded,geometry,rows),localized((height * width : U32),(rows * width : U32),offset)}

# The extents of the view over rows rows.
def extents(+banded: Banded,+rows: U32) -> List<&2,U32>:
  Banded{before,repeats,after,offset} = banded
  List.append(&2,U32,Views.axis_extents(before),List.append(&2,U32,rows <> repeats,Views.axis_extents(after)))

def known(count: Maybe<&2,U32>) -> U32:
  match count:
    case Some{count}: count
    case None{}: 0

def product_or_zero(extents: List<&2,U32>) -> U32:
  known(Shapes.product(extents,Some{1}))

# The rows of the output for each row of the input.
def scale(+banded: Banded) -> U32:
  Banded{before,repeats,after,offset} = banded
  product_or_zero(repeats)

# The geometry of the output: the same images, the axes before the rows as
# channels, each row repeated, the axes after the rows as columns.
def output(+banded: Banded,+geometry: Bands.Geometry) -> Bands.Geometry:
  Banded{before,repeats,after,offset} = banded
  Bands.Geometry{batch,channels,height,width} = geometry
  Bands.Geometry{batch,product_or_zero(Views.axis_extents(before)),(height * product_or_zero(repeats) : U32),product_or_zero(Views.axis_extents(after))}

# ---- Directions ----

def count_when(present: Bool) -> Nat:
  match present:
    case True{}: 1n
    case False{}: 0n

def local_when(valid: Bool,flags: List<&2,Bool>) -> Maybe<&2,List<&2,Bool>>:
  match valid:
    case True{}: Some{flags}
    case False{}: None{}

# The flags of the axes of a local view: those of the axes before the rows,
# the rows axis and its repeats forward, those of the axes after the rows.
def local_directions(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+before_flags: List<&2,Bool>,+after_flags: List<&2,Bool>) -> List<&2,Bool>:
  List.append(&2,Bool,Views.fitted_flags(List.length(&2,Views.Axis,before),before_flags),
    List.append(&2,Bool,Views.fitted_flags(1n+List.length(&2,U32,repeats),[]),Views.fitted_flags(List.length(&2,Views.Axis,after),after_flags)))

# The flags of the local view of a band, when the view reads its images axis,
# its rows axis and the repeats of each row forward. An axis without a flag is
# read forward.
def local_flags(+banded: Banded,+geometry: Bands.Geometry,+flags: List<&2,Bool>) -> Maybe<&2,List<&2,Bool>>:
  Banded{+before,+repeats,+after,offset} = banded
  Bands.Geometry{+batch,channels,+height,width} = geometry
  +images = count_when(U32.is_lt(1,batch))
  +rows = Nat.add(count_when(U32.is_lt(1,height)),List.length(&2,U32,repeats))
  +later = Views.drop_flags(Nat.add(images,List.length(&2,Views.Axis,before)),flags)
  local_when(Views.forward(images,flags) && Views.forward(rows,later),local_directions(before,repeats,after,Views.drop_flags(images,flags),Views.drop_flags(rows,later)))

# ---- Classification ----

# The axes after the images axis, which a view of more than one image starts
# with.
def images_kept(kept: Bool,rest: List<&2,Views.Axis>) -> Maybe<&2,List<&2,Views.Axis>>:
  match kept:
    case True{}: Some{rest}
    case False{}: None{}

def after_images(many: Bool,+batch: U32,+image: U32,axes: List<&2,Views.Axis>) -> Maybe<&2,List<&2,Views.Axis>>:
  match many axes:
    case False{} axes: Some{axes}
    case True{} Nil{}: None{}
    case True{} Views.Axis{+extent,+stride} <> rest: images_kept(U32.is_eq(extent,batch) && U32.is_eq(stride,image),rest)

# The axes before the rows axis and those after it.
type Split is Data:
  Split{before: List<&2,Views.Axis>,rest: List<&2,Views.Axis>}

def prepend_before(axis: Views.Axis,split: Maybe<&2,Split>) -> Maybe<&2,Split>:
  match split:
    case Some{Split{before,rest}}: Some{Split{axis <> before,rest}}
    case None{}: None{}

def split_from(rest: List<&2,Views.Axis>,found: Bool,+height: U32,+width: U32,+extent: U32,+stride: U32) -> Maybe<&2,Split>:
  match rest found:
    case rest True{}: Some{Split{[],rest}}
    case Nil{} False{}: None{}
    case Views.Axis{+next_extent,+next_stride} <> tail False{}:
      prepend_before(Views.Axis{extent,stride},split_from(tail,U32.is_eq(next_extent,height) && U32.is_eq(next_stride,width),height,width,next_extent,next_stride))

def split_rows(+height: U32,+width: U32,axes: List<&2,Views.Axis>) -> Maybe<&2,Split>:
  match axes:
    case []: None{}
    case Views.Axis{+extent,+stride} <> rest: split_from(rest,U32.is_eq(extent,height) && U32.is_eq(stride,width),height,width,extent,stride)

# The leading axes of stride zero repeat each row.
def prepend_repeat(extent: U32,split: List<&2,U32> & List<&2,Views.Axis>) -> List<&2,U32> & List<&2,Views.Axis>:
  (repeats,rest) = split
  (extent <> repeats,rest)

def repeats_from(rest: List<&2,Views.Axis>,zero: Bool,extent: U32,stride: U32) -> List<&2,U32> & List<&2,Views.Axis>:
  match rest zero:
    case rest False{}: ([],Views.Axis{extent,stride} <> rest)
    case Nil{} True{}: ([extent],[])
    case Views.Axis{next_extent,+next_stride} <> tail True{}: prepend_repeat(extent,repeats_from(tail,U32.is_zero(next_stride),next_extent,next_stride))

def split_repeats(axes: List<&2,Views.Axis>) -> List<&2,U32> & List<&2,Views.Axis>:
  match axes:
    case []: ([],[])
    case Views.Axis{extent,+stride} <> rest: repeats_from(rest,U32.is_zero(stride),extent,stride)

# The largest channel, or column, the axes reach from a start.
def reach(+plane: U32,+column: Bool,axes: List<&2,Views.Axis>,start: Maybe<&2,U32>) -> Maybe<&2,U32>:
  match axes start:
    case [] start: start
    case Views.Axis{+extent,stride} <> rest Some{+start}:
      reach(plane,column,rest,Shapes.checked_add_to(Shapes.checked_multiply((extent - 1 : U32),component(plane,column,stride)),start))
    case axes None{}: None{}

def below(reach: Maybe<&2,U32>,+limit: U32) -> Bool:
  match reach:
    case Some{reach}: U32.is_lt(reach,limit)
    case None{}: False{}

def counted(extents: List<&2,U32>) -> Bool:
  Views.has_count(Shapes.product(extents,Some{1}))

# A classification is kept when the geometry has images and rows, every axis
# stays inside the channels and the columns of a band and every group of
# extents has a count.
def kept_when(valid: Bool,banded: Banded) -> Maybe<&2,Banded>:
  match valid:
    case True{}: Some{banded}
    case False{}: None{}

def fits(+banded: Banded,+geometry: Bands.Geometry) -> Bool:
  Banded{+before,+repeats,+after,+offset} = banded
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  +plane = (height * width : U32)
  U32.is_lt(0,batch) && U32.is_lt(0,height) && below(reach(plane,False{},after,reach(plane,False{},before,Some{component(plane,False{},offset)})),channels) &&
    below(reach(plane,True{},after,reach(plane,True{},before,Some{component(plane,True{},offset)})),width) &&
    counted(Views.axis_extents(before)) && counted(Views.axis_extents(after)) && counted(repeats) && Views.has_count(Shapes.checked_multiply(height,product_or_zero(repeats)))

def kept(+banded: Banded,+geometry: Bands.Geometry) -> Maybe<&2,Banded>:
  kept_when(fits(banded,geometry),banded)

def split_kept(+offset: U32,+geometry: Bands.Geometry,before: List<&2,Views.Axis>,split: List<&2,U32> & List<&2,Views.Axis>) -> Maybe<&2,Banded>:
  (repeats,after) = split
  kept(Banded{before,repeats,after,offset},geometry)

def rowed_split(+offset: U32,+geometry: Bands.Geometry,split: Maybe<&2,Split>) -> Maybe<&2,Banded>:
  match split:
    case Some{Split{before,rest}}: split_kept(offset,geometry,before,split_repeats(rest))
    case None{}: None{}

def split_geometry(+geometry: Bands.Geometry,axes: List<&2,Views.Axis>) -> Maybe<&2,Split>:
  Bands.Geometry{batch,channels,+height,+width} = geometry
  split_rows(height,width,axes)

def classify_rows(rowed: Bool,axes: Maybe<&2,List<&2,Views.Axis>>,+offset: U32,+geometry: Bands.Geometry) -> Maybe<&2,Banded>:
  match rowed axes:
    case True{} Some{axes}: rowed_split(offset,geometry,split_geometry(geometry,axes))
    case False{} Some{axes}: kept(Banded{[],[],axes,offset},geometry)
    case rowed None{}: None{}

# The banded form of a view, given by its axes and offset, of the storage of a
# geometry, when the view starts with the images axis of a batch of more than
# one image, has the rows axis of a geometry of more than one row, and stays
# inside the channels and columns. A geometry of one image of one row, as a
# whole tensor has, takes every view whose addresses fit it.
def classify(axes: List<&2,Views.Axis>,+offset: U32,+geometry: Bands.Geometry) -> Maybe<&2,Banded>:
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  classify_rows(U32.is_lt(1,height),after_images(U32.is_lt(1,batch),batch,(channels * (height * width : U32) : U32),axes),offset,geometry)
