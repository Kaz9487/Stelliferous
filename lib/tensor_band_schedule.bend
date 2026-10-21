# Conservative memory-work estimates for elementwise and layout operations.
# Small operations retain their owners and run sequentially. Multi-image trees
# remain sequential here; window operations have explicit per-image dispatch.
import Base
import ./tensor_band.bend as Bands
import ./storage_buffer.bend as Storage
import ./kernel_cost.bend as Cost
import ./default_kernel_cost.bend as DefaultCost

type Size is Data:
  Size{leaves: U32,minimum_rows: U32,maximum_rows: U32}

def merge(left: Size,right: Size) -> Size:
  Size{ln,lmin,lmax} = left
  Size{rn,rmin,rmax} = right
  Size{(ln + rn : U32),U32.min(lmin,rmin),U32.max(lmax,rmax)}

def leaf_size(context: Unit,first: U32,rows: U32) -> Size:
  +rows = rows
  Size{1,rows,rows}

def size(layout: Bands.Plan) -> Size:
  Bands.plan_fold(~Size,~Unit,~leaf_size,~Bands.plain(~Size,~Unit,~merge),layout,Unit{})

# The whole operation, not each leaf, must reach the grain: small leaves only
# lengthen the runtime's rounds, which the region estimate already counts.
def sized(geometry: Bands.Geometry,+workers: U32,element_ns: F32,config: Cost.Configuration,size: Size) -> Bool:
  Bands.Geometry{+batch,channels,height,width} = geometry
  Cost.Configuration{mac,byte_ns,operation_ns,grain,wake} = config
  Size{+leaves,rows,maximum} = size
  +row_ns = (U32.to_f32(channels) * U32.to_f32(width) * element_ns : F32)
  +total = (U32.to_f32(batch) * U32.to_f32(height) * row_ns : F32)
  U32.is_lt(1,workers) && U32.is_lt(1,leaves) && Nat.is_le(U32.to_nat(leaves),Nat.mul(U32.to_nat(batch),U32.to_nat((U32.min(workers,858993459) * 5 : U32)))) &&
    F32.is_le(grain,total) &&
    F32.is_lt((Cost.region_ns(workers,leaves,(U32.to_f32(maximum) * row_ns : F32)) + 2.0 * wake : F32),total)

def parallel(geometry: Bands.Geometry,workers: U32,bytes_per_element: F32,operations: F32,layout: Bands.Plan) -> Bool:
  +config = DefaultCost.configuration()
  sized(geometry,workers,Cost.element_ns(bytes_per_element,operations,config),config,size(layout))

# For equal-shaped operands the logical read/write volume of repartition is
# identical. The target owners' rounded capacities decide the remaining cost.
def leaf_elements(geometry: Bands.Geometry,first: U32,rows: U32) -> Nat:
  U32.to_nat(U32.shln(1,Storage.depth_for(Bands.local_count(geometry,rows))))

def allocation_elements(layout: Bands.Plan,+geometry: Bands.Geometry) -> Nat:
  Bands.plan_fold(~Nat,~Bands.Geometry,~leaf_elements,~Bands.plain(~Nat,~Bands.Geometry,~Nat.add),layout,geometry)

def alignment(+geometry: Bands.Geometry,+left: Bands.Plan,+right: Bands.Plan) -> Bands.Plan:
  Bool.pick(Bands.Plan,Bands.same_plan(left,right) || Nat.is_le(allocation_elements(left,geometry),allocation_elements(right,geometry)),left,right)

# The layout an operation runs in: a geometry, a plan of its rows, and whether
# the plan splits them into more than one band.
type Target is Data:
  Target{geometry: Bands.Geometry,layout: Bands.Plan,split: Bool}

def splits(layout: Bands.Plan) -> Bool:
  match layout:
    case Bands.Rows{first,count}: False{}
    case Bands.Split{left,right}: True{}

def joined(same: Bool,+geometry: Bands.Geometry,+layout: Bands.Plan,other_layout: Bands.Plan) -> Target:
  match same:
    case True{}: Target{geometry,alignment(geometry,layout,other_layout),True{}}
    case False{}: Target{geometry,layout,True{}}

def met(split: Bool,other_split: Bool,+geometry: Bands.Geometry,layout: Bands.Plan,other: Bands.Geometry,other_layout: Bands.Plan) -> Target:
  match split other_split:
    case True{} True{}: joined(Bands.same_geometry(geometry,other),geometry,layout,other_layout)
    case True{} False{}: Target{geometry,layout,True{}}
    case False{} True{}: Target{other,other_layout,True{}}
    case False{} False{}: Target{geometry,layout,False{}}

# The layout of two operands: one split into bands names it, and of two split
# operands of one geometry the plan that allocates less. When neither is
# split, the first names it.
def meet(left: Target,right: Target) -> Target:
  Target{geometry,layout,split} = left
  Target{other,other_layout,other_split} = right
  met(split,other_split,geometry,layout,other,other_layout)

def plane_chosen(rows: Bool,+planes: U32,+height: U32,+width: U32,target: Target) -> Target:
  match rows:
    case True{}: target
    case False{}: Target{Bands.Geometry{1,planes,height,width},Bands.unsplit(Bands.Geometry{1,planes,height,width}),False{}}

# The layout of an operation on each plane of the last two axes: the tensor's
# own when its bands split the image rows of its shape, otherwise its planes
# as the channels of one image in one band, which a whole tensor already is.
def plane_target(+shape: List<&2,U32>,+planes: U32,+height: U32,+width: U32,target: Target) -> Target:
  Target{+geometry,layout,split} = target
  plane_chosen(Bands.image_rows(shape,geometry),planes,height,width,Target{geometry,layout,split})

def mapped_context(~Element: Data,~Context: Data,~mapper: @+count: Nat -> @+position: U32 -> @+context: Context -> Storage.Buffer<Element> -> Storage.Buffer<Element>,context: Context,+geometry: Bands.Geometry,workers: U32,operations: F32,owners: Bands.Bands(~Element) & Bands.Plan) -> Bands.Bands(~Element):
  (bands,+layout) = owners
  Bands.map_with(~Element,~Context,~Bands.map_whole_context(~Element,~Context,~mapper),layout,parallel(geometry,workers,8.0,operations,layout),context,bands)

def map_context(~Element: Data,~Context: Data,~mapper: @+count: Nat -> @+position: U32 -> @+context: Context -> Storage.Buffer<Element> -> Storage.Buffer<Element>,context: Context,geometry: Bands.Geometry,workers: U32,operations: F32,bands: Bands.Bands(~Element)) -> Bands.Bands(~Element):
  mapped_context(~Element,~Context,~mapper,context,geometry,workers,operations,Bands.plan(~Element,bands))

def scalar_planned(~Element: Data,~operation: Element -> Element -> Element,+geometry: Bands.Geometry,workers: U32,operations: F32,value: Element,owners: Bands.Bands(~Element) & Bands.Plan) -> Bands.Bands(~Element):
  (bands,+layout) = owners
  Bands.scalar(~Element,~operation,layout,parallel(geometry,workers,8.0,operations,layout),value,bands)

def scalar(~Element: Data,~operation: Element -> Element -> Element,geometry: Bands.Geometry,workers: U32,operations: F32,value: Element,bands: Bands.Bands(~Element)) -> Bands.Bands(~Element):
  scalar_planned(~Element,~operation,geometry,workers,operations,value,Bands.plan(~Element,bands))

def indexed_planned(~Element: Data,~operation: U32 -> Element -> Element,+geometry: Bands.Geometry,workers: U32,operations: F32,fill: Element,owners: Bands.Bands(~Element) & Bands.Plan) -> Bands.Bands(~Element):
  (bands,+layout) = owners
  Bands.apply_indexed(~Element,~operation,layout,parallel(geometry,workers,12.0,operations,layout),geometry,fill,bands)

def indexed(~Element: Data,~operation: U32 -> Element -> Element,geometry: Bands.Geometry,workers: U32,operations: F32,fill: Element,bands: Bands.Bands(~Element)) -> Bands.Bands(~Element):
  indexed_planned(~Element,~operation,geometry,workers,operations,fill,Bands.plan(~Element,bands))

def transformed(~Element: Data,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Bands.Transformed(~Element),
  context: Context,+geometry: Bands.Geometry,workers: U32,bytes_per_element: F32,operations: F32,owners: Bands.Bands(~Element) & Bands.Plan) -> Bands.Bands(~Element) & Bool:
  (bands,+layout) = owners
  Bands.transform(~Element,~Context,~leaf,layout,parallel(geometry,workers,bytes_per_element,operations,layout),context,bands)

# Every band runs the leaf in the plan it has. bytes_per_element and
# operations price one input element.
def transform(~Element: Data,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Bands.Transformed(~Element),
  context: Context,geometry: Bands.Geometry,workers: U32,bytes_per_element: F32,operations: F32,bands: Bands.Bands(~Element)) -> Bands.Bands(~Element) & Bool:
  transformed(~Element,~Context,~leaf,context,geometry,workers,bytes_per_element,operations,Bands.plan(~Element,bands))
