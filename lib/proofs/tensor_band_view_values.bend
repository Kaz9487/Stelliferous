# The gather of a view in its banded form, with some axes read in reverse.
# Every band runs one checked copy through its local view with the flags of
# its axes; when every call is usable and the bands cover the output
# geometry, the result holds at every cell the input at the address the
# whole view, read with its flags, gives to that cell.
import Base
import ../tensor.bend as Tensors
import ../tensor_view.bend as Views
import ../tensor_shape.bend as Shapes
import ../tensor_band.bend as Bands
import ../tensor_band_view.bend as BandViews
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ../tensor_band_schedule.bend as BandSchedule
import ../tensor_operations.bend as Operations
import ../storage_buffer.bend as Storage
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./tensor_band_model.bend as Model
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_values.bend as BandValues
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_inspection.bend as Inspection
import ./tensor_band_each.bend as Each
import ./tensor_global_shape_count.bend as Global
import ./tensor_band_arrive.bend as Arrive
import ./tensor_band_target.bend as Target
import ./tensor_whole_leaf.bend as WholeLeaf
import ./tensor_count_refinement.bend as Count
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./tensor_operations_refinement.bend as OperationsLaws
import ./tensor_contract.bend as Contract
import ./tensor_gather_values.bend as GatherValues
import ./tensor_copy_values.bend as CopyValues
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_band_view_classified.bend as Classified
import ./tensor_band_view_address.bend as Address
import ./checked_word_refinement.bend as Checked
import ./machine_limit_bounds.bend as Limits
import ./row_major_index.bend as RowMajor
import ./tensor_row_major_coordinates.bend as Coordinates
import ./nat_order.bend as Order
import ./nat_division_order.bend as DivisionOrder
import ./natural_addition.bend as Addition
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_full_range.bend as FullRange
import ./boolean_logic.bend as Boolean
import ./word_encoding.bend as WordEncoding
import ./tensor_line_values.bend as LineValues
import ./tensor_mirror_address.bend as MirrorAddress
import ./tensor_view_metadata_refinement.bend as Metadata

# ---- What a classification establishes ----

def known_count(result: Maybe<&2,U32>,+extents: List<&2,U32>,count: ShapeLaws.Count(result,ShapeLaws.elements(extents)),counted: {Views.has_count(result) == True{} : Bool}) ->
  {U32.to_nat(BandViews.known(result)) == ShapeLaws.elements(extents) : Nat}:
  match result:
    case Some{+value}: Equal.sym(Nat,ShapeLaws.elements(extents),U32.to_nat(value),count)
    case None{}: Empty.absurd({U32.to_nat(BandViews.known(None{})) == ShapeLaws.elements(extents) : Nat},WordEncoding.false_true(counted))

# Extents a classification counts have their elements as their product.
def counted_exact(+extents: List<&2,U32>,counted: {BandViews.counted(extents) == True{} : Bool}) ->
  {U32.to_nat(BandViews.product_or_zero(extents)) == ShapeLaws.elements(extents) : Nat}:
  known_count(Shapes.product(extents,Some{1}),extents,ShapeLaws.nonempty_count(extents),counted)

def product_word(result: Maybe<&2,U32>,+left: U32,+right: U32,+computed: {Shapes.checked_multiply(left,right) == result : Maybe<&2,U32>},
  counted: {Views.has_count(result) == True{} : Bool}) -> {U32.to_nat((left * right : U32)) == Nat.mul(U32.to_nat(left),U32.to_nat(right)) : Nat}:
  match result:
    case Some{+value}:
      FullRange.multiply(left,right,value,
        %Checked.multiply_sound(left,right,value,computed) : {Nat.is_le(Nat.mul(U32.to_nat(left),U32.to_nat(right)),_) == True{} : Bool}
        Order.reflexive(Nat.mul(U32.to_nat(left),U32.to_nat(right))))
    case None{}: Empty.absurd({U32.to_nat((left * right : U32)) == Nat.mul(U32.to_nat(left),U32.to_nat(right)) : Nat},WordEncoding.false_true(counted))

type Fitted<-before: List<&2,Views.Axis>,-repeats: List<&2,U32>,-after: List<&2,Views.Axis>,-offset: U32,-batch: U32,-channels: U32,-height: U32,-width: U32> is Data:
  Fitted{batch_positive: {Nat.is_lt(0n,U32.to_nat(batch)) == True{} : Bool},height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
    channel_reach: {Nat.is_lt(Address.reach_total((height * width : U32),False{},before,after,offset),U32.to_nat(channels)) == True{} : Bool},
    column_reach: {Nat.is_lt(Address.reach_total((height * width : U32),True{},before,after,offset),U32.to_nat(width)) == True{} : Bool},
    early: {U32.to_nat(BandViews.product_or_zero(Views.axis_extents(before))) == ShapeLaws.elements(Views.axis_extents(before)) : Nat},
    late: {U32.to_nat(BandViews.product_or_zero(Views.axis_extents(after))) == ShapeLaws.elements(Views.axis_extents(after)) : Nat},
    repeated: {U32.to_nat(BandViews.product_or_zero(repeats)) == ShapeLaws.elements(repeats) : Nat},
    rows_exact: {U32.to_nat((height * BandViews.product_or_zero(repeats) : U32)) == Nat.mul(U32.to_nat(height),U32.to_nat(BandViews.product_or_zero(repeats))) : Nat}}

# A classification that fits its geometry has images and rows, stays inside
# the channels and columns, and counts its extents.
def fitted(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+offset: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,
  +fits: {BandViews.fits(BandViews.Banded{before,repeats,after,offset},Bands.Geometry{batch,channels,height,width}) == True{} : Bool}) ->
  Fitted<before,repeats,after,offset,batch,channels,height,width>:
  +plane = (height * width : U32)
  +scale = BandViews.product_or_zero(repeats)
  +a1 = U32.is_lt(0,batch)
  +a2 = U32.is_lt(0,height)
  +a3 = BandViews.below(BandViews.reach(plane,False{},after,BandViews.reach(plane,False{},before,Some{BandViews.component(plane,False{},offset)})),channels)
  +a4 = BandViews.below(BandViews.reach(plane,True{},after,BandViews.reach(plane,True{},before,Some{BandViews.component(plane,True{},offset)})),width)
  +a5 = BandViews.counted(Views.axis_extents(before))
  +a6 = BandViews.counted(Views.axis_extents(after))
  +a7 = BandViews.counted(repeats)
  +a8 = Views.has_count(Shapes.checked_multiply(height,scale))
  +l2 = a1 && a2
  +l3 = a1 && a2 && a3
  +l4 = a1 && a2 && a3 && a4
  +l5 = a1 && a2 && a3 && a4 && a5
  +l6 = a1 && a2 && a3 && a4 && a5 && a6
  +l7 = a1 && a2 && a3 && a4 && a5 && a6 && a7
  +f7 = Boolean.left(l7,a8,fits)
  +f6 = Boolean.left(l6,a7,f7)
  +f5 = Boolean.left(l5,a6,f6)
  +f4 = Boolean.left(l4,a5,f5)
  +f3 = Boolean.left(l3,a4,f4)
  +f2 = Boolean.left(l2,a3,f3)
  Fitted{Bounds.as_nat_lt(0,batch,Boolean.left(a1,a2,f2)),Bounds.as_nat_lt(0,height,Boolean.right(a1,a2,f2)),
    Address.reach_within(plane,False{},before,after,BandViews.component(plane,False{},offset),channels,Boolean.right(l2,a3,f3)),
    Address.reach_within(plane,True{},before,after,BandViews.component(plane,True{},offset),width,Boolean.right(l3,a4,f4)),
    counted_exact(Views.axis_extents(before),Boolean.right(l4,a5,f5)),counted_exact(Views.axis_extents(after),Boolean.right(l5,a6,f6)),
    counted_exact(repeats,Boolean.right(l6,a7,f7)),product_word(Shapes.checked_multiply(height,scale),height,scale,{==},Boolean.right(l7,a8,fits))}

# The plane and the image of a geometry the machine holds, as words.
type Plane<-batch: U32,-channels: U32,-height: U32,-width: U32> is Data:
  Plane{divisor: Address.Divisor<(height * width : U32)>,
    plane_exact: {U32.to_nat((height * width : U32)) == Nat.mul(U32.to_nat(height),U32.to_nat(width)) : Nat},
    image_exact: {U32.to_nat((channels * (height * width : U32) : U32)) == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width))) : Nat}}

def plane_words(+batch: U32,+channels: U32,+height: U32,+width: U32,+limits: Machine.Limits,+batch_positive: {Nat.is_lt(0n,U32.to_nat(batch)) == True{} : Bool},
  domain: BandGather.Domain<Bands.Geometry{batch,channels,height,width},limits,Machine.capacity(limits)>) -> Plane<batch,channels,height,width>:
  BandGather.Domain{capacity_ok,+channels_positive,+height_positive,+width_positive,+machine_room,allocation_room} = domain
  +plane = (height * width : U32)
  +spatial = Nat.mul(U32.to_nat(height),U32.to_nat(width))
  +frame = Nat.mul(U32.to_nat(channels),spatial)
  +capacity = U32.to_nat(Machine.capacity(limits))
  +frame_room = Order.transitive(frame,Nat.mul(U32.to_nat(batch),frame),capacity,DivisionOrder.le_multiple(U32.to_nat(batch),frame,batch_positive),machine_room)
  +spatial_room = Order.transitive(spatial,frame,capacity,DivisionOrder.le_multiple(U32.to_nat(channels),spatial,channels_positive),frame_room)
  +plane_exact = FullRange.multiply(height,width,Machine.capacity(limits),spatial_room)
  +plane_room = {%Equal.sym(Nat,U32.to_nat(plane),spatial,plane_exact) : {Nat.is_le(_,capacity) == True{} : Bool} spatial_room : {Nat.is_le(U32.to_nat(plane),capacity) == True{} : Bool}}
  +image_word = FullRange.multiply(channels,plane,Machine.capacity(limits),
    %Equal.sym(Nat,U32.to_nat(plane),spatial,plane_exact) : {Nat.is_le(Nat.mul(U32.to_nat(channels),_),capacity) == True{} : Bool} frame_room)
  +positive = {%Equal.sym(Nat,U32.to_nat(plane),spatial,plane_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    Bounds.nat_trans(0n,U32.to_nat(width),spatial,width_positive,DivisionOrder.le_multiple(U32.to_nat(height),U32.to_nat(width),height_positive)) :
    {Nat.is_lt(0n,U32.to_nat(plane)) == True{} : Bool}}
  Plane{Address.Divisor{Machine.half_word(limits),Limits.half_word_fits(limits),Limits.double_bound(limits,U32.to_nat(plane),plane_room),positive},plane_exact,
    Equal.trans(Nat,U32.to_nat(U32.mul(channels,plane)),Nat.mul(U32.to_nat(channels),U32.to_nat(plane)),frame,image_word,
      Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(channels),value),U32.to_nat(plane),spatial,plane_exact))}

# ---- Rows of a band, repeated ----

def scaled_rows_fit(+batch: U32,+height: U32,+scale: U32,+first: U32,+rows: U32,
  rows_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(batch),U32.to_nat(height))) == True{} : Bool},
  height_exact: {U32.to_nat((height * scale : U32)) == Nat.mul(U32.to_nat(height),U32.to_nat(scale)) : Nat}) ->
  {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(scale)),Nat.mul(U32.to_nat(batch),U32.to_nat((height * scale : U32)))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat((height * scale : U32)),Nat.mul(U32.to_nat(height),U32.to_nat(scale)),height_exact) :
    {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(scale)),Nat.mul(U32.to_nat(batch),_)) == True{} : Bool}
  %RowMajor.assoc(U32.to_nat(batch),U32.to_nat(height),U32.to_nat(scale)) :
    {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(scale)),_) == True{} : Bool}
  RowMajor.mul_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(batch),U32.to_nat(height)),U32.to_nat(scale),rows_fit)

# A repeated row of a band is a row of the repeated batch.
def scaled_row_inside(+batch: U32,+height: U32,+scale: U32,+first: U32,+rows: U32,+row: Nat,
  row_inside: {Nat.is_lt(row,U32.to_nat((rows * scale : U32))) == True{} : Bool},
  first_exact: {U32.to_nat((first * scale : U32)) == Nat.mul(U32.to_nat(first),U32.to_nat(scale)) : Nat},
  rows_exact: {U32.to_nat((rows * scale : U32)) == Nat.mul(U32.to_nat(rows),U32.to_nat(scale)) : Nat},
  fit: {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(scale)),Nat.mul(U32.to_nat(batch),U32.to_nat((height * scale : U32)))) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(U32.to_nat((first * scale : U32)),row),Nat.mul(U32.to_nat(batch),U32.to_nat((height * scale : U32)))) == True{} : Bool}:
  +start = U32.to_nat((first * scale : U32))
  +end = Nat.add(start,U32.to_nat((rows * scale : U32)))
  +total = Nat.mul(U32.to_nat(batch),U32.to_nat((height * scale : U32)))
  +end_fit = {
    %Equal.sym(Nat,start,Nat.mul(U32.to_nat(first),U32.to_nat(scale)),first_exact) : {Nat.is_le(Nat.add(_,U32.to_nat((rows * scale : U32))),total) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat((rows * scale : U32)),Nat.mul(U32.to_nat(rows),U32.to_nat(scale)),rows_exact) :
      {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(first),U32.to_nat(scale)),_),total) == True{} : Bool}
    %RowMajor.distrib(U32.to_nat(first),U32.to_nat(rows),U32.to_nat(scale)) : {Nat.is_le(_,total) == True{} : Bool}
    fit : {Nat.is_le(end,total) == True{} : Bool}}
  Bounds.predecessor_strict(Nat.add(start,row),total,
    Order.transitive(1n+Nat.add(start,row),end,total,
      Order.strict_follow(Nat.add(start,row),end,Order.offset_lt(start,row,U32.to_nat((rows * scale : U32)),row_inside)),end_fit))

def scaled_first_exact(+first: U32,+rows: U32,+scale: U32,+capacity: U32,
  room: {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(scale)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((first * scale : U32)) == Nat.mul(U32.to_nat(first),U32.to_nat(scale)) : Nat}:
  FullRange.multiply(first,scale,capacity,Order.transitive(Nat.mul(U32.to_nat(first),U32.to_nat(scale)),Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(scale)),U32.to_nat(capacity),
    RowMajor.mul_le(U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(scale),Order.le_add(U32.to_nat(first),U32.to_nat(rows))),room))

def scaled_rows_exact(+first: U32,+rows: U32,+scale: U32,+capacity: U32,
  room: {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(scale)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((rows * scale : U32)) == Nat.mul(U32.to_nat(rows),U32.to_nat(scale)) : Nat}:
  FullRange.multiply(rows,scale,capacity,Order.transitive(Nat.mul(U32.to_nat(rows),U32.to_nat(scale)),Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(scale)),U32.to_nat(capacity),
    RowMajor.mul_le(U32.to_nat(rows),Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(scale),
      %Order.commutative(U32.to_nat(rows),U32.to_nat(first)) : {Nat.is_le(U32.to_nat(rows),_) == True{} : Bool}
      Order.le_add(U32.to_nat(rows),U32.to_nat(first))),room))

def content_room(~Element: Data,+content: Buffer.Content<Element>,ready: Buffer.ContentReady<Element,content>) ->
  {Nat.is_le(Buffer.content_length(~Element,content),U32.to_nat(Tree.capacity(Element,Buffer.content_tree(~Element,content)))) == True{} : Bool}:
  Buffer.ContentReady{depth,balanced,depth_ok,room} = ready
  room

# A repeated row of the output lies in the image of its input row, at that
# row's repeat.
def located_row(+first: Nat,+row: Nat,+repeat: Nat,+height: Nat,+scale: Nat,+positive: {Nat.is_lt(0n,height) == True{} : Bool}) ->
  {Nat.add(Nat.mul(Nat.div(Nat.add(first,row),height),Nat.mul(height,scale)),Nat.add(Nat.mul(Nat.mod(Nat.add(first,row),height),scale),repeat)) ==
    Nat.add(Nat.mul(first,scale),Nat.add(Nat.mul(row,scale),repeat)) : Nat}:
  +image = Nat.div(Nat.add(first,row),height)
  +inside = Nat.mod(Nat.add(first,row),height)
  Equal.trans(Nat,Nat.add(Nat.mul(image,Nat.mul(height,scale)),Nat.add(Nat.mul(inside,scale),repeat)),Nat.add(Nat.mul(Nat.add(Nat.mul(image,height),inside),scale),repeat),
    Nat.add(Nat.mul(first,scale),Nat.add(Nat.mul(row,scale),repeat)),
    Equal.sym(Nat,Nat.add(Nat.mul(Nat.add(Nat.mul(image,height),inside),scale),repeat),Nat.add(Nat.mul(image,Nat.mul(height,scale)),Nat.add(Nat.mul(inside,scale),repeat)),
      Coordinates.digit_join(image,height,inside,scale,repeat)),
    %DivisionOrder.quotient_exact(Nat.add(first,row),height,positive) :
      {Nat.add(Nat.mul(_,scale),repeat) == Nat.add(Nat.mul(first,scale),Nat.add(Nat.mul(row,scale),repeat)) : Nat}
    %Equal.sym(Nat,Nat.mul(Nat.add(first,row),scale),Nat.add(Nat.mul(first,scale),Nat.mul(row,scale)),RowMajor.distrib(first,row,scale)) : {Nat.add(_,repeat) == Nat.add(Nat.mul(first,scale),Nat.add(Nat.mul(row,scale),repeat)) : Nat}
    Addition.associative(Nat.mul(first,scale),Nat.mul(row,scale),repeat))

def band_rows_fit(+geometry: Bands.Geometry,+first: U32,+rows: U32,+length: Nat,fields: BandShape.Leaf(geometry,first,rows,length)) ->
  {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Inspection.total_rows(geometry)) == True{} : Bool}:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  BandShape.Fields{local,rows_fit,length_exact} = fields
  rows_fit

def domain_height(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,domain: BandGather.Domain<geometry,limits,capacity>) ->
  {BandGather.height_positive(geometry) == True{} : Bool}:
  BandGather.Domain{capacity_ok,channels_positive,height_positive,width_positive,machine_room,allocation_room} = domain
  height_positive

def domain_elements(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,domain: BandGather.Domain<geometry,limits,capacity>) ->
  {Nat.is_le(BandGather.elements(geometry),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  BandGather.Domain{capacity_ok,channels_positive,height_positive,width_positive,machine_room,allocation_room} = domain
  machine_room

# ---- One band ----

def Words(banded: BandViews.Banded,geometry: Bands.Geometry) -> Data:
  BandViews.Banded{before,repeats,after,offset} = banded
  Bands.Geometry{batch,channels,height,width} = geometry
  Shape.Both<Fitted<before,repeats,after,offset,batch,channels,height,width>,Plane<batch,channels,height,width>>

# What every band of one gather shares: the logical input, the view, the
# flags of its axes and its banded form over the input geometry, the flags of
# the local view, the call, and what its guard established.
type Scope<-Element: Data,-Logical: Data> is Data:
  Scope{logical: Logical,name: String,view_axes: List<&2,Views.Axis>,offset: U32,reversed: List<&2,Bool>,banded: BandViews.Banded,geometry: Bands.Geometry,local: List<&2,Bool>,
    fill: Element,limits: Machine.Limits,words: Words(banded,geometry),classified: Classified.Classified<view_axes,offset,geometry,banded>,
    directions: {BandViews.local_flags(banded,geometry,reversed) == Some{local} : Maybe<&2,List<&2,Bool>>},
    room: BandGather.Domain<BandViews.output(banded,geometry),limits,Machine.capacity(limits)>}

def scope_gathered(~Element: Data,~Logical: Data,+scope: Scope<Element,Logical>) -> Tensors.Gathered<Element>:
  Scope{logical,name,view_axes,offset,reversed,banded,geometry,local,fill,limits,words,classified,directions,room} = scope
  Tensors.Gathered{name,banded,geometry,local,fill}

def scaled_word(~Element: Data,~Logical: Data,+scope: Scope<Element,Logical>,+value: U32) -> U32:
  Scope{logical,name,view_axes,offset,reversed,+banded,geometry,local,fill,limits,words,classified,directions,room} = scope
  (value * BandViews.scale(banded) : U32)

def output_geometry(~Element: Data,~Logical: Data,+scope: Scope<Element,Logical>) -> Bands.Geometry:
  Scope{logical,name,view_axes,offset,reversed,+banded,+geometry,+local,fill,limits,words,classified,directions,room} = scope
  BandViews.output(banded,geometry)

# The facts of one input band: its fields, its buffer is ready, and its
# values.
def input_fields(~Element: Data,~Logical: Data,+scope: Scope<Element,Logical>) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  Scope{logical,name,view_axes,offset,reversed,banded,geometry,local,fill,limits,words,classified,directions,room} = scope
  BandShape.band_fields(~Element,geometry)

def input_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: Scope<Element,Logical>) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  Scope{logical,name,view_axes,offset,reversed,banded,geometry,local,fill,limits,words,classified,directions,room} = scope
  BandValues.band_values(~Element,~Logical,~original,logical,geometry)

def Input(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: Scope<Element,Logical>) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  Model.Both(~Element,~Scope<Element,Logical>,~Model.Both(~Element,~Scope<Element,Logical>,~input_fields(~Element,~Logical),~Model.always(~Element,~Scope<Element,Logical>,~Model.ready_band(~Element))),
    ~input_values(~Element,~Logical,~original),scope)

# The local view of a band, its flags and its extents.
def source(~Element: Data,~Logical: Data,+scope: Scope<Element,Logical>,+rows: U32) -> Views.View:
  Scope{logical,name,view_axes,offset,reversed,+banded,+geometry,+local,fill,limits,words,classified,directions,room} = scope
  BandViews.local(banded,geometry,rows)

def scope_local(~Element: Data,~Logical: Data,+scope: Scope<Element,Logical>) -> List<&2,Bool>:
  Scope{logical,name,view_axes,offset,reversed,banded,geometry,+local,fill,limits,words,classified,directions,room} = scope
  local

def band_extents(~Element: Data,~Logical: Data,+scope: Scope<Element,Logical>,+rows: U32) -> List<&2,U32>:
  Scope{logical,name,view_axes,offset,reversed,+banded,geometry,local,fill,limits,words,classified,directions,room} = scope
  BandViews.extents(banded,rows)

# What the copy of one band through its local view leaves.
def Viewed(~Element: Data,~Logical: Data,+scope: Scope<Element,Logical>,+rows: U32,+input: Buffer.Content<Element>,+output: Buffer.Content<Element>) -> Data:
  GatherValues.Facts(~Element,~GatherValues.Directed<Buffer.Content<Element>>,~GatherValues.directed(~Element,~Buffer.Content<Element>,~Contract.stored(~Element)),
    GatherValues.Directed{input,source(~Element,~Logical,scope,rows),scope_local(~Element,~Logical,scope)},band_extents(~Element,~Logical,scope,rows),output)

# An output band holds at each cell the input at the address the whole view,
# read with its flags, gives to that cell.
def OutputValues(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: Scope<Element,Logical>) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  Scope{logical,name,+view_axes,+offset,+reversed,+banded,+geometry,+local,fill,limits,words,classified,directions,room} = scope
  BandValues.band_values(~Element,~GatherValues.Directed<Logical>,~GatherValues.directed(~Element,~Logical,~original),GatherValues.Directed{logical,Views.View{view_axes,offset},reversed},
    BandViews.output(banded,geometry))

# ---- The cells of one band ----

# What one band hands to each of its cells.
type Cells<-Element: Data,-Logical: Data,-original: Logical -> U32 -> Element> is Data:
  Cells{scope: Scope<Element,Logical>,first: U32,rows: U32,input: Buffer.Model<Element>,model: Buffer.Model<Element>,
    input_fact: Input(~Element,~Logical,~original,scope,first,rows,Buffer.content(~Element,input)),
    viewed: Viewed(~Element,~Logical,scope,rows,Buffer.content(~Element,input),Buffer.content(~Element,model))}

def cells_geometry(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,evidence: Cells<Element,Logical,original>) -> Bands.Geometry:
  Cells{scope,first,rows,input,model,input_fact,viewed} = evidence
  output_geometry(~Element,~Logical,scope)

def cells_rows(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,evidence: Cells<Element,Logical,original>) -> U32:
  Cells{scope,first,+rows,input,model,input_fact,viewed} = evidence
  scaled_word(~Element,~Logical,scope,rows)

def CellAt(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+evidence: Cells<Element,Logical,original>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  Cells{+scope,+first,+rows,input,+model,input_fact,viewed} = evidence
  Scope{+logical,name,+view_axes,+offset,+reversed,+banded,+geometry,+local,fill,limits,words,classified,directions,room} = scope
  BandValues.At(~Element,~GatherValues.Directed<Logical>,~GatherValues.directed(~Element,~Logical,~original),GatherValues.Directed{logical,Views.View{view_axes,offset},reversed},
    BandViews.output(banded,geometry),(first * BandViews.scale(banded) : U32),(rows * BandViews.scale(banded) : U32),
    Buffer.content_tree(~Element,Buffer.content(~Element,model)),Buffer.content_length(~Element,Buffer.content(~Element,model)),channel,row,column)

# The image of an output row of a band, and the row within that image.
def row_image(+first: U32,+row: Nat,+height: U32,+repeats: List<&2,U32>) -> Nat:
  Nat.div(Nat.add(U32.to_nat(first),Nat.div(row,ShapeLaws.elements(repeats))),U32.to_nat(height))

def image_row(+first: U32,+row: Nat,+height: U32,+repeats: List<&2,U32>) -> Nat:
  +es = ShapeLaws.elements(repeats)
  Nat.add(Nat.mul(Nat.mod(Nat.add(U32.to_nat(first),Nat.div(row,es)),U32.to_nat(height)),es),Nat.mod(row,es))

def row_image_below(+batch: U32,+channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+source_length: Nat,+repeats: List<&2,U32>,+row: Nat,
  +fields: BandShape.Leaf(Bands.Geometry{batch,channels,height,width},first,rows,source_length),
  +height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +row_below: {Nat.is_lt(row,Nat.mul(U32.to_nat(rows),ShapeLaws.elements(repeats))) == True{} : Bool}) ->
  {Nat.is_lt(row_image(first,row,height,repeats),U32.to_nat(batch)) == True{} : Bool}:
  +es = ShapeLaws.elements(repeats)
  +rows_n = U32.to_nat(rows)
  +scale_positive = DivisionOrder.right_factor_positive(rows_n,es,Order.positive_bound(row,Nat.mul(rows_n,es),row_below))
  BandValues.image_below(U32.to_nat(first),rows_n,Nat.div(row,es),U32.to_nat(batch),U32.to_nat(height),height_positive,
    DivisionOrder.quotient_below(row,rows_n,es,scale_positive,row_below),band_rows_fit(Bands.Geometry{batch,channels,height,width},first,rows,source_length,fields))

def image_row_below(+first: U32,+rows: U32,+row: Nat,+height: U32,+repeats: List<&2,U32>,
  +height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +row_below: {Nat.is_lt(row,Nat.mul(U32.to_nat(rows),ShapeLaws.elements(repeats))) == True{} : Bool}) ->
  {Nat.is_lt(image_row(first,row,height,repeats),Nat.mul(U32.to_nat(height),ShapeLaws.elements(repeats))) == True{} : Bool}:
  +es = ShapeLaws.elements(repeats)
  +rows_n = U32.to_nat(rows)
  +source_global = Nat.add(U32.to_nat(first),Nat.div(row,es))
  +scale_positive = DivisionOrder.right_factor_positive(rows_n,es,Order.positive_bound(row,Nat.mul(rows_n,es),row_below))
  RowMajor.radix(Nat.mod(source_global,U32.to_nat(height)),U32.to_nat(height),Nat.mod(row,es),es,
    DivisionOrder.remainder_small(source_global,U32.to_nat(height),height_positive),DivisionOrder.remainder_small(row,es,scale_positive))

# The pixel of an output cell of a band, at the image of its row and the row
# within that image.
def pixel_split(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,+limits: Machine.Limits,
  +words: Words(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width}),
  +rows_room: {Nat.is_le(Nat.mul(U32.to_nat(batch),U32.to_nat((height * BandViews.product_or_zero(repeats) : U32))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +first: U32,+rows: U32,+source_length: Nat,+fields: BandShape.Leaf(Bands.Geometry{batch,channels,height,width},first,rows,source_length),
  +channel: Nat,+row: Nat,+column: Nat,
  +row_below: {Nat.is_lt(row,Nat.mul(U32.to_nat(rows),ShapeLaws.elements(repeats))) == True{} : Bool}) ->
  {BandValues.pixel(BandViews.output(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width}),channel,
      Nat.add(U32.to_nat((first * BandViews.product_or_zero(repeats) : U32)),row),column) ==
    RowMajor.index(Nat.add(Nat.mul(row_image(first,row,height,repeats),ShapeLaws.elements(Views.axis_extents(before))),channel),
      Nat.mul(U32.to_nat(height),ShapeLaws.elements(repeats)),ShapeLaws.elements(Views.axis_extents(after)),image_row(first,row,height,repeats),column) : Nat}:
  Shape.Both{Fitted{batch_positive,+height_positive,channel_reach,column_reach,+early,+late,+repeated,+rows_exact},Plane{divisor,plane_exact,image_exact}} = words
  +scale = BandViews.product_or_zero(repeats)
  +before_count = BandViews.product_or_zero(Views.axis_extents(before))
  +after_count = BandViews.product_or_zero(Views.axis_extents(after))
  +output = {Bands.Geometry{batch,before_count,(height * scale : U32),after_count} : Bands.Geometry}
  +capacity = Machine.capacity(limits)
  +es = ShapeLaws.elements(repeats)
  +eb = ShapeLaws.elements(Views.axis_extents(before))
  +ea = ShapeLaws.elements(Views.axis_extents(after))
  +h = U32.to_nat(height)
  +first_n = U32.to_nat(first)
  +rows_n = U32.to_nat(rows)
  +rows_fit = band_rows_fit(Bands.Geometry{batch,channels,height,width},first,rows,source_length,fields)
  +fit = scaled_rows_fit(batch,height,scale,first,rows,rows_fit,rows_exact)
  +fit_room = Order.transitive(Nat.mul(Nat.add(first_n,rows_n),U32.to_nat(scale)),Nat.mul(U32.to_nat(batch),U32.to_nat((height * scale : U32))),U32.to_nat(capacity),fit,rows_room)
  +first_exact = scaled_first_exact(first,rows,scale,capacity,fit_room)
  +first_es = {%repeated : {U32.to_nat((first * scale : U32)) == Nat.mul(first_n,_) : Nat} first_exact : {U32.to_nat((first * scale : U32)) == Nat.mul(first_n,es) : Nat}}
  +height_es = {%repeated : {U32.to_nat((height * scale : U32)) == Nat.mul(h,_) : Nat} rows_exact : {U32.to_nat((height * scale : U32)) == Nat.mul(h,es) : Nat}}
  +scale_positive = DivisionOrder.right_factor_positive(rows_n,es,Order.positive_bound(row,Nat.mul(rows_n,es),row_below))
  +source_row = Nat.div(row,es)
  +repeat = Nat.mod(row,es)
  +row_exact = DivisionOrder.quotient_exact(row,es,scale_positive)
  +global = Nat.add(U32.to_nat((first * scale : U32)),row)
  +image = row_image(first,row,height,repeats)
  +inside_row = image_row(first,row,height,repeats)
  +inside_row_below = {%Equal.sym(Nat,U32.to_nat((height * scale : U32)),Nat.mul(h,es),height_es) : {Nat.is_lt(inside_row,_) == True{} : Bool}
    image_row_below(first,rows,row,height,repeats,height_positive,row_below) : {Nat.is_lt(inside_row,U32.to_nat((height * scale : U32))) == True{} : Bool}}
  +located = {
    %Equal.sym(Nat,U32.to_nat((height * scale : U32)),Nat.mul(h,es),height_es) : {Nat.add(Nat.mul(image,_),inside_row) == global : Nat}
    %Equal.sym(Nat,U32.to_nat((first * scale : U32)),Nat.mul(first_n,es),first_es) : {Nat.add(Nat.mul(image,Nat.mul(h,es)),inside_row) == Nat.add(_,row) : Nat}
    %Equal.sym(Nat,row,Nat.add(Nat.mul(source_row,es),repeat),row_exact) : {Nat.add(Nat.mul(image,Nat.mul(h,es)),inside_row) == Nat.add(Nat.mul(first_n,es),_) : Nat}
    located_row(first_n,source_row,repeat,h,es,height_positive) : {BandValues.global_row(output,image,inside_row) == global : Nat}}
  Equal.trans(Nat,BandValues.pixel(output,channel,global,column),BandValues.point(output,image,channel,inside_row,column),
    RowMajor.index(Nat.add(Nat.mul(image,eb),channel),Nat.mul(h,es),ea,inside_row,column),
    BandValues.located_pixel(output,image,channel,inside_row,column,global,inside_row_below,located),
    %early : {RowMajor.index(Nat.add(Nat.mul(image,U32.to_nat(before_count)),channel),U32.to_nat((height * scale : U32)),U32.to_nat(after_count),inside_row,column) ==
      RowMajor.index(Nat.add(Nat.mul(image,_),channel),Nat.mul(h,es),ea,inside_row,column) : Nat}
    %height_es : {RowMajor.index(Nat.add(Nat.mul(image,U32.to_nat(before_count)),channel),U32.to_nat((height * scale : U32)),U32.to_nat(after_count),inside_row,column) ==
      RowMajor.index(Nat.add(Nat.mul(image,U32.to_nat(before_count)),channel),_,ea,inside_row,column) : Nat}
    %late : {RowMajor.index(Nat.add(Nat.mul(image,U32.to_nat(before_count)),channel),U32.to_nat((height * scale : U32)),U32.to_nat(after_count),inside_row,column) ==
      RowMajor.index(Nat.add(Nat.mul(image,U32.to_nat(before_count)),channel),U32.to_nat((height * scale : U32)),_,inside_row,column) : Nat}
    {==})

# A cell of the banded form of the view over a band of the input, read
# through the local view of the band, is the input at the address the whole
# view gives to the pixel of the cell. Its row splits into the input row and
# the repeat.
def local_read(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+view_axes: List<&2,Views.Axis>,+offset: U32,
  +before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,+limits: Machine.Limits,
  +words: Words(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width}),
  +classified: Classified.Classified<view_axes,offset,Bands.Geometry{batch,channels,height,width},BandViews.Banded{before,repeats,after,start}>,
  +rows_room: {Nat.is_le(Nat.mul(U32.to_nat(batch),U32.to_nat((height * BandViews.product_or_zero(repeats) : U32))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +first: U32,+rows: U32,+source_tree: Tree.Tree<Element>,+source_length: Nat,
  +fields: BandShape.Leaf(Bands.Geometry{batch,channels,height,width},first,rows,source_length),
  +source_ready: Buffer.ContentReady<Element,Buffer.Content{source_tree,source_length}>,
  +source_values: BandValues.Leaf(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},first,rows,source_tree,source_length),
  +channel: Nat,+row: Nat,+column: Nat,
  +channel_below: {Nat.is_lt(channel,ShapeLaws.elements(Views.axis_extents(before))) == True{} : Bool},
  +row_below: {Nat.is_lt(row,Nat.mul(U32.to_nat(rows),ShapeLaws.elements(repeats))) == True{} : Bool},
  +column_below: {Nat.is_lt(column,ShapeLaws.elements(Views.axis_extents(after))) == True{} : Bool}) ->
  {Tree.read(Element,source_tree,CopyValues.address(BandViews.local(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},rows),
      RowMajor.index(channel,Nat.mul(U32.to_nat(rows),ShapeLaws.elements(repeats)),ShapeLaws.elements(Views.axis_extents(after)),row,column))) ==
    original(logical,CopyValues.address(Views.View{view_axes,offset},
      BandValues.pixel(BandViews.output(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width}),channel,
        Nat.add(U32.to_nat((first * BandViews.product_or_zero(repeats) : U32)),row),column))) : Element}:
  Shape.Both{Fitted{batch_positive,+height_positive,+channel_reach,+column_reach,early,late,repeated,rows_exact},Plane{+divisor,+plane_exact,+image_exact}} = words
  Classified.Classified{+axes_exact,+offset_exact,fits} = classified
  +plane = (height * width : U32)
  +scale = BandViews.product_or_zero(repeats)
  +before_count = BandViews.product_or_zero(Views.axis_extents(before))
  +after_count = BandViews.product_or_zero(Views.axis_extents(after))
  +output = {Bands.Geometry{batch,before_count,(height * scale : U32),after_count} : Bands.Geometry}
  +es = ShapeLaws.elements(repeats)
  +eb = ShapeLaws.elements(Views.axis_extents(before))
  +ea = ShapeLaws.elements(Views.axis_extents(after))
  +h = U32.to_nat(height)
  +first_n = U32.to_nat(first)
  +rows_n = U32.to_nat(rows)
  +scale_positive = DivisionOrder.right_factor_positive(rows_n,es,Order.positive_bound(row,Nat.mul(rows_n,es),row_below))
  +source_row = Nat.div(row,es)
  +repeat = Nat.mod(row,es)
  +source_row_inside = DivisionOrder.quotient_below(row,rows_n,es,scale_positive,row_below)
  +repeat_inside = DivisionOrder.remainder_small(row,es,scale_positive)
  +row_exact = DivisionOrder.quotient_exact(row,es,scale_positive)
  +local = BandViews.local(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},rows)
  +channel_at = Address.at(plane,False{},before,after,start,channel,column)
  +column_at = Address.at(plane,True{},before,after,start,channel,column)
  +channel_at_inside = Order.le_lt_trans(channel_at,Address.reach_total(plane,False{},before,after,start),U32.to_nat(channels),
    Address.at_within(plane,False{},before,after,start,channel,column,channel_below,column_below),channel_reach)
  +column_at_inside = Order.le_lt_trans(column_at,Address.reach_total(plane,True{},before,after,start),U32.to_nat(width),
    Address.at_within(plane,True{},before,after,start,channel,column,channel_below,column_below),column_reach)
  +source_capacity = Tree.capacity(Element,source_tree)
  +cell_count = Nat.mul(U32.to_nat(channels),Nat.mul(rows_n,U32.to_nat(width)))
  +cells_room = {%BandValues.leaf_length(Bands.Geometry{batch,channels,height,width},first,rows,source_length,fields) : {Nat.is_le(_,U32.to_nat(source_capacity)) == True{} : Bool}
    content_room(~Element,Buffer.Content{source_tree,source_length},source_ready) : {Nat.is_le(cell_count,U32.to_nat(source_capacity)) == True{} : Bool}}
  +channels_positive = Order.positive_bound(Address.reach_total(plane,False{},before,after,start),U32.to_nat(channels),channel_reach)
  +band_exact = FullRange.multiply(rows,width,source_capacity,
    Order.transitive(Nat.mul(rows_n,U32.to_nat(width)),cell_count,U32.to_nat(source_capacity),DivisionOrder.le_multiple(U32.to_nat(channels),Nat.mul(rows_n,U32.to_nat(width)),channels_positive),cells_room))
  +band_room = {%Equal.sym(Nat,U32.to_nat((rows * width : U32)),Nat.mul(rows_n,U32.to_nat(width)),band_exact) : {Nat.is_le(Nat.mul(U32.to_nat(channels),_),U32.to_nat(source_capacity)) == True{} : Bool}
    cells_room : {Nat.is_le(Nat.mul(U32.to_nat(channels),U32.to_nat((rows * width : U32))),U32.to_nat(source_capacity)) == True{} : Bool}}
  +source_index = RowMajor.index(channel_at,rows_n,U32.to_nat(width),source_row,column_at)
  +global = Nat.add(U32.to_nat((first * scale : U32)),row)
  +source_global = Nat.add(first_n,source_row)
  +image = Nat.div(source_global,h)
  +inside = Nat.mod(source_global,h)
  +inside_row = Nat.add(Nat.mul(inside,es),repeat)
  +inside_below = DivisionOrder.remainder_small(source_global,h,height_positive)
  +image_inside = row_image_below(batch,channels,height,width,first,rows,source_length,repeats,row,fields,height_positive,row_below)
  +whole = Classified.whole_axes(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width})
  +index = RowMajor.index(channel,Nat.mul(rows_n,es),ea,Nat.add(Nat.mul(source_row,es),repeat),column)
  +band_read = Equal.cong(U32,Element,word => Tree.read(Element,source_tree,word),CopyValues.address(local,index),U32.from_nat(source_index),
    Address.band_address(before,repeats,after,start,batch,channels,height,width,rows,source_capacity,channel,source_row,repeat,column,
      channel_below,source_row_inside,repeat_inside,column_below,channel_reach,column_reach,band_exact,band_room))
  %Equal.sym(Nat,row,Nat.add(Nat.mul(source_row,es),repeat),row_exact) :
    {Tree.read(Element,source_tree,CopyValues.address(local,RowMajor.index(channel,Nat.mul(rows_n,es),ea,_,column))) ==
      original(logical,CopyValues.address(Views.View{view_axes,offset},BandValues.pixel(output,channel,global,column))) : Element}
  %Equal.sym(Nat,BandValues.pixel(output,channel,global,column),RowMajor.index(Nat.add(Nat.mul(image,eb),channel),Nat.mul(h,es),ea,inside_row,column),
      pixel_split(before,repeats,after,start,batch,channels,height,width,limits,words,rows_room,first,rows,source_length,fields,channel,row,column,row_below)) :
    {Tree.read(Element,source_tree,CopyValues.address(local,index)) == original(logical,CopyValues.address(Views.View{view_axes,offset},_)) : Element}
  %Equal.sym(List<&2,Views.Axis>,view_axes,whole,axes_exact) :
    {Tree.read(Element,source_tree,CopyValues.address(local,index)) ==
      original(logical,CopyValues.address(Views.View{_,offset},RowMajor.index(Nat.add(Nat.mul(image,eb),channel),Nat.mul(h,es),ea,inside_row,column))) : Element}
  %offset_exact :
    {Tree.read(Element,source_tree,CopyValues.address(local,index)) ==
      original(logical,CopyValues.address(Views.View{whole,_},RowMajor.index(Nat.add(Nat.mul(image,eb),channel),Nat.mul(h,es),ea,inside_row,column))) : Element}
  %Equal.sym(U32,CopyValues.address(Views.View{whole,start},RowMajor.index(Nat.add(Nat.mul(image,eb),channel),Nat.mul(h,es),ea,inside_row,column)),
      U32.from_nat(RowMajor.index(Nat.add(Nat.mul(image,U32.to_nat(channels)),channel_at),h,U32.to_nat(width),inside,column_at)),
      Address.whole_address(before,repeats,after,start,batch,channels,height,width,image,channel,inside,repeat,column,divisor,plane_exact,image_exact,
        image_inside,channel_below,inside_below,repeat_inside,column_below)) :
    {Tree.read(Element,source_tree,CopyValues.address(local,index)) == original(logical,_) : Element}
  Equal.trans(Element,Tree.read(Element,source_tree,CopyValues.address(local,index)),Tree.read(Element,source_tree,U32.from_nat(source_index)),
    original(logical,U32.from_nat(BandValues.pixel(Bands.Geometry{batch,channels,height,width},channel_at,source_global,column_at))),
    band_read,
    BandValues.leaf_read(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},first,rows,source_tree,source_length,channel_at,source_row,column_at,source_values,
      channel_at_inside,source_row_inside,column_at_inside))

# ---- The elements of a classified view ----

def images_elements(many: Bool,+batch: U32,+stride: U32,+computed: {U32.is_lt(1,batch) == many : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(batch)) == True{} : Bool}) ->
  {ShapeLaws.elements(Views.axis_extents(Classified.images_axes(many,batch,stride))) == U32.to_nat(batch) : Nat}:
  match many:
    case True{}: Order.times_one(U32.to_nat(batch))
    case False{}:
      Equal.sym(Nat,U32.to_nat(batch),1n,Shape.first({U32.to_nat(batch) == 1n : Nat},{0n == 0n : Nat},Address.presence(False{},batch,0n,computed,positive)))

def rows_elements(rowed: Bool,+height: U32,+width: U32,+repeats: List<&2,U32>,+computed: {U32.is_lt(1,height) == rowed : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool}) ->
  {ShapeLaws.elements(Views.axis_extents(BandViews.rows_axes(rowed,height,width,repeats))) == Nat.mul(U32.to_nat(height),ShapeLaws.elements(repeats)) : Nat}:
  match rowed:
    case True{}:
      Equal.cong(List<&2,U32>,Nat,extents => Nat.mul(U32.to_nat(height),ShapeLaws.elements(extents)),Views.axis_extents(BandViews.repeated(repeats)),repeats,Address.extents_repeated(repeats))
    case False{}:
      %Equal.sym(Nat,U32.to_nat(height),1n,Shape.first({U32.to_nat(height) == 1n : Nat},{0n == 0n : Nat},Address.presence(False{},height,0n,computed,positive))) : {ShapeLaws.elements(Views.axis_extents(BandViews.repeated(repeats))) == Nat.mul(_,ShapeLaws.elements(repeats)) : Nat}
      Equal.trans(Nat,ShapeLaws.elements(Views.axis_extents(BandViews.repeated(repeats))),ShapeLaws.elements(repeats),Nat.mul(1n,ShapeLaws.elements(repeats)),
        Equal.cong(List<&2,U32>,Nat,extents => ShapeLaws.elements(extents),Views.axis_extents(BandViews.repeated(repeats)),repeats,Address.extents_repeated(repeats)),
        Equal.sym(Nat,Nat.add(ShapeLaws.elements(repeats),0n),ShapeLaws.elements(repeats),Addition.zero_right(ShapeLaws.elements(repeats))))

# The axes a banded form describes hold as many elements as its output
# geometry.
def whole_elements(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,
  +fitted: Fitted<before,repeats,after,start,batch,channels,height,width>) ->
  {ShapeLaws.elements(Views.axis_extents(Classified.whole_axes(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width}))) ==
    BandGather.elements(BandViews.output(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width})) : Nat}:
  Fitted{+batch_positive,+height_positive,channel_reach,column_reach,+early,+late,+repeated,+rows_exact} = fitted
  +scale = BandViews.product_or_zero(repeats)
  +before_count = BandViews.product_or_zero(Views.axis_extents(before))
  +after_count = BandViews.product_or_zero(Views.axis_extents(after))
  +images = Classified.images_axes(U32.is_lt(1,batch),batch,(channels * (height * width : U32) : U32))
  +rowed = BandViews.rows_axes(U32.is_lt(1,height),height,width,repeats)
  +rest = List.append(&2,Views.Axis,rowed,after)
  +middle = List.append(&2,Views.Axis,before,rest)
  +all = List.append(&2,Views.Axis,images,middle)
  +b = U32.to_nat(batch)
  +h = U32.to_nat(height)
  +es = ShapeLaws.elements(repeats)
  +eb = ShapeLaws.elements(Views.axis_extents(before))
  +ea = ShapeLaws.elements(Views.axis_extents(after))
  +height_es = {%repeated : {U32.to_nat((height * scale : U32)) == Nat.mul(h,_) : Nat} rows_exact : {U32.to_nat((height * scale : U32)) == Nat.mul(h,es) : Nat}}
  +target = Nat.mul(b,Nat.mul(eb,Nat.mul(Nat.mul(h,es),ea)))
  %Equal.sym(Nat,U32.to_nat(before_count),eb,early) :
    {ShapeLaws.elements(Views.axis_extents(all)) == Nat.mul(b,Nat.mul(_,Nat.mul(U32.to_nat((height * scale : U32)),U32.to_nat(after_count)))) : Nat}
  %Equal.sym(Nat,U32.to_nat((height * scale : U32)),Nat.mul(h,es),height_es) :
    {ShapeLaws.elements(Views.axis_extents(all)) == Nat.mul(b,Nat.mul(eb,Nat.mul(_,U32.to_nat(after_count)))) : Nat}
  %Equal.sym(Nat,U32.to_nat(after_count),ea,late) :
    {ShapeLaws.elements(Views.axis_extents(all)) == Nat.mul(b,Nat.mul(eb,Nat.mul(Nat.mul(h,es),_))) : Nat}
  Equal.trans(Nat,ShapeLaws.elements(Views.axis_extents(all)),Nat.mul(ShapeLaws.elements(Views.axis_extents(images)),ShapeLaws.elements(Views.axis_extents(middle))),target,
    LineValues.axes_elements(images,middle),
    %Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(images)),b,images_elements(U32.is_lt(1,batch),batch,(channels * (height * width : U32) : U32),{==},batch_positive)) :
      {Nat.mul(_,ShapeLaws.elements(Views.axis_extents(middle))) == target : Nat}
    Equal.cong(Nat,Nat,value => Nat.mul(b,value),ShapeLaws.elements(Views.axis_extents(middle)),Nat.mul(eb,Nat.mul(Nat.mul(h,es),ea)),
      Equal.trans(Nat,ShapeLaws.elements(Views.axis_extents(middle)),Nat.mul(eb,ShapeLaws.elements(Views.axis_extents(rest))),Nat.mul(eb,Nat.mul(Nat.mul(h,es),ea)),
        LineValues.axes_elements(before,rest),
        Equal.cong(Nat,Nat,value => Nat.mul(eb,value),ShapeLaws.elements(Views.axis_extents(rest)),Nat.mul(Nat.mul(h,es),ea),
          %rows_elements(U32.is_lt(1,height),height,width,repeats,{==},height_positive) : {ShapeLaws.elements(Views.axis_extents(rest)) == Nat.mul(_,ea) : Nat}
          LineValues.axes_elements(rowed,after)))))

# ---- Cells read in reverse ----

# The flags of the axes before and after the rows that the local view of a
# band reads with.
def before_flags(+batch: U32,+reversed: List<&2,Bool>) -> List<&2,Bool>:
  Views.drop_flags(BandViews.count_when(U32.is_lt(1,batch)),reversed)

def lines_count(+height: U32,+repeats: List<&2,U32>) -> Nat:
  Nat.add(BandViews.count_when(U32.is_lt(1,height)),List.length(&2,U32,repeats))

def later_flags(+before: List<&2,Views.Axis>,+batch: U32,+reversed: List<&2,Bool>) -> List<&2,Bool>:
  Views.drop_flags(Nat.add(BandViews.count_when(U32.is_lt(1,batch)),List.length(&2,Views.Axis,before)),reversed)

def after_flags(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+batch: U32,+height: U32,+reversed: List<&2,Bool>) -> List<&2,Bool>:
  Views.drop_flags(lines_count(height,repeats),later_flags(before,batch,reversed))

# The flags read the images axis, the rows axis and the repeats of each row
# forward.
def Forward(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+batch: U32,+height: U32,+reversed: List<&2,Bool>) -> Data:
  Shape.Both<{Views.forward(BandViews.count_when(U32.is_lt(1,batch)),reversed) == True{} : Bool},
    {Views.forward(lines_count(height,repeats),later_flags(before,batch,reversed)) == True{} : Bool}>

def found_flags(found: Maybe<&2,List<&2,Bool>>) -> List<&2,Bool>:
  match found:
    case Some{flags}: flags
    case None{}: []

def is_found(found: Maybe<&2,List<&2,Bool>>) -> Bool:
  match found:
    case Some{flags}: True{}
    case None{}: False{}

# Flags a classification gives the local view are valid and are the flags it
# lists.
def directions_valid(valid: Bool,+flags: List<&2,Bool>,+local: List<&2,Bool>,same: {BandViews.local_when(valid,flags) == Some{local} : Maybe<&2,List<&2,Bool>>}) -> {valid == True{} : Bool}:
  match valid:
    case True{}: {==}
    case False{}:
      Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(Equal.cong(Maybe<&2,List<&2,Bool>>,Bool,is_found,None{},Some{local},same)))

def directions_exact(valid: Bool,+flags: List<&2,Bool>,+local: List<&2,Bool>,same: {BandViews.local_when(valid,flags) == Some{local} : Maybe<&2,List<&2,Bool>>}) -> {flags == local : List<&2,Bool>}:
  match valid:
    case True{}: Equal.cong(Maybe<&2,List<&2,Bool>>,List<&2,Bool>,found_flags,Some{flags},Some{local},same)
    case False{}:
      Empty.absurd({flags == local : List<&2,Bool>},WordEncoding.false_true(Equal.cong(Maybe<&2,List<&2,Bool>>,Bool,is_found,None{},Some{local},same)))

def repeated_length(+repeats: List<&2,U32>) -> {List.length(&2,Views.Axis,BandViews.repeated(repeats)) == List.length(&2,U32,repeats) : Nat}:
  match repeats:
    case []: {==}
    case +extent <> +tail:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Views.Axis,BandViews.repeated(tail)),List.length(&2,U32,tail),repeated_length(tail))

def images_count(many: Bool,+batch: U32,+stride: U32) -> {List.length(&2,Views.Axis,Classified.images_axes(many,batch,stride)) == BandViews.count_when(many) : Nat}:
  match many:
    case True{}: {==}
    case False{}: {==}

def rows_count(rowed: Bool,+rows: U32,+width: U32,+repeats: List<&2,U32>) ->
  {List.length(&2,Views.Axis,BandViews.rows_axes(rowed,rows,width,repeats)) == Nat.add(BandViews.count_when(rowed),List.length(&2,U32,repeats)) : Nat}:
  match rowed:
    case True{}: Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Views.Axis,BandViews.repeated(repeats)),List.length(&2,U32,repeats),repeated_length(repeats))
    case False{}: repeated_length(repeats)

# The local view of a band read with the flags of its axes reads, at a cell,
# what it reads forward at the cell of the reflected channel and column.
def local_reflected(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,+rows: U32,
  +early_flags: List<&2,Bool>,+late_flags: List<&2,Bool>,+channel: Nat,+row: Nat,+column: Nat,
  +channel_below: {Nat.is_lt(channel,ShapeLaws.elements(Views.axis_extents(before))) == True{} : Bool},
  +row_below: {Nat.is_lt(row,Nat.mul(U32.to_nat(rows),ShapeLaws.elements(repeats))) == True{} : Bool},
  +column_below: {Nat.is_lt(column,ShapeLaws.elements(Views.axis_extents(after))) == True{} : Bool}) ->
  {MirrorAddress.address(BandViews.local(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},rows),
      BandViews.local_directions(before,repeats,after,early_flags,late_flags),
      RowMajor.index(channel,Nat.mul(U32.to_nat(rows),ShapeLaws.elements(repeats)),ShapeLaws.elements(Views.axis_extents(after)),row,column)) ==
    CopyValues.address(BandViews.local(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},rows),
      RowMajor.index(Coordinates.reflected(Views.axis_extents(before),early_flags,channel),Nat.mul(U32.to_nat(rows),ShapeLaws.elements(repeats)),
        ShapeLaws.elements(Views.axis_extents(after)),row,Coordinates.reflected(Views.axis_extents(after),late_flags,column))) : U32}:
  +plane = (height * width : U32)
  +band = (rows * width : U32)
  +early = BandViews.localized_axes(plane,band,before)
  +middle = BandViews.rows_axes(True{},rows,width,repeats)
  +late = BandViews.localized_axes(plane,band,after)
  +offset = BandViews.localized(plane,band,start)
  +view = BandViews.local(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},rows)
  +late_fitted = Views.fitted_flags(List.length(&2,Views.Axis,after),late_flags)
  +middle_flags = List.append(&2,Bool,Views.fitted_flags(1n+List.length(&2,U32,repeats),[]),late_fitted)
  +flagged = BandViews.local_directions(before,repeats,after,early_flags,late_flags)
  +es = ShapeLaws.elements(repeats)
  +ea = ShapeLaws.elements(Views.axis_extents(after))
  +extent = Nat.mul(U32.to_nat(rows),es)
  +nc = ShapeLaws.elements(Views.axis_extents(middle))
  +nd = ShapeLaws.elements(Views.axis_extents(late))
  +turned_channel = Coordinates.reflected(Views.axis_extents(before),early_flags,channel)
  +turned_column = Coordinates.reflected(Views.axis_extents(after),late_flags,column)
  +early_extents = Address.extents_localized(plane,band,before)
  +late_extents = Address.extents_localized(plane,band,after)
  +middle_elements = Equal.cong(List<&2,U32>,Nat,extents => Nat.mul(U32.to_nat(rows),ShapeLaws.elements(extents)),Views.axis_extents(BandViews.repeated(repeats)),repeats,
    Address.extents_repeated(repeats))
  +late_elements = Equal.cong(List<&2,U32>,Nat,ShapeLaws.elements,Views.axis_extents(late),Views.axis_extents(after),late_extents)
  +middle_length = Equal.cong(List<&2,U32>,Nat,extents => 1n+List.length(&2,U32,extents),Views.axis_extents(BandViews.repeated(repeats)),repeats,Address.extents_repeated(repeats))
  +middle_axes = Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Views.Axis,BandViews.repeated(repeats)),List.length(&2,U32,repeats),repeated_length(repeats))
  +early_inside = {%Equal.sym(List<&2,U32>,Views.axis_extents(early),Views.axis_extents(before),early_extents) : {Nat.is_lt(channel,ShapeLaws.elements(_)) == True{} : Bool}
    channel_below : {Nat.is_lt(channel,ShapeLaws.elements(Views.axis_extents(early))) == True{} : Bool}}
  +middle_inside = {%Equal.sym(Nat,nc,extent,middle_elements) : {Nat.is_lt(row,_) == True{} : Bool} row_below : {Nat.is_lt(row,nc) == True{} : Bool}}
  +late_inside = {%Equal.sym(Nat,nd,ea,late_elements) : {Nat.is_lt(column,_) == True{} : Bool} column_below : {Nat.is_lt(column,nd) == True{} : Bool}}
  +early_dropped = Equal.trans(List<&2,Bool>,Views.drop_flags(List.length(&2,Views.Axis,early),flagged),Views.drop_flags(List.length(&2,Views.Axis,before),flagged),middle_flags,
    Equal.cong(Nat,List<&2,Bool>,count => Views.drop_flags(count,flagged),List.length(&2,Views.Axis,early),List.length(&2,Views.Axis,before),Address.localized_length(plane,band,before)),
    Coordinates.drop_fitted(List.length(&2,Views.Axis,before),early_flags,middle_flags))
  +late_dropped = Equal.trans(List<&2,Bool>,Views.drop_flags(List.length(&2,Views.Axis,middle),Views.drop_flags(List.length(&2,Views.Axis,early),flagged)),
    Views.drop_flags(List.length(&2,Views.Axis,middle),middle_flags),late_fitted,
    Equal.cong(List<&2,Bool>,List<&2,Bool>,flags => Views.drop_flags(List.length(&2,Views.Axis,middle),flags),Views.drop_flags(List.length(&2,Views.Axis,early),flagged),middle_flags,early_dropped),
    Equal.trans(List<&2,Bool>,Views.drop_flags(List.length(&2,Views.Axis,middle),middle_flags),Views.drop_flags(1n+List.length(&2,U32,repeats),middle_flags),late_fitted,
      Equal.cong(Nat,List<&2,Bool>,count => Views.drop_flags(count,middle_flags),List.length(&2,Views.Axis,middle),1n+List.length(&2,U32,repeats),middle_axes),
      Coordinates.drop_fitted(1n+List.length(&2,U32,repeats),[],late_fitted)))
  +channel_turned = Equal.trans(Nat,Coordinates.reflected(Views.axis_extents(early),flagged,channel),Coordinates.reflected(Views.axis_extents(before),flagged,channel),turned_channel,
    Equal.cong(List<&2,U32>,Nat,extents => Coordinates.reflected(extents,flagged,channel),Views.axis_extents(early),Views.axis_extents(before),early_extents),
    %Metadata.extents_length(before) :
      {Coordinates.reflected(Views.axis_extents(before),List.append(&2,Bool,Views.fitted_flags(_,early_flags),middle_flags),channel) == turned_channel : Nat}
    Coordinates.reflected_prefix(Views.axis_extents(before),early_flags,middle_flags,channel))
  +row_kept = Equal.trans(Nat,Coordinates.reflected(Views.axis_extents(middle),Views.drop_flags(List.length(&2,Views.Axis,early),flagged),row),
    Coordinates.reflected(Views.axis_extents(middle),middle_flags,row),row,
    Equal.cong(List<&2,Bool>,Nat,flags => Coordinates.reflected(Views.axis_extents(middle),flags,row),Views.drop_flags(List.length(&2,Views.Axis,early),flagged),middle_flags,early_dropped),
    Equal.trans(Nat,Coordinates.reflected(Views.axis_extents(middle),middle_flags,row),Coordinates.reflected(Views.axis_extents(middle),[],row),row,
      %middle_length : {Coordinates.reflected(Views.axis_extents(middle),List.append(&2,Bool,Views.fitted_flags(_,[]),late_fitted),row) ==
        Coordinates.reflected(Views.axis_extents(middle),[],row) : Nat}
      Coordinates.reflected_prefix(Views.axis_extents(middle),[],late_fitted,row),
      Coordinates.reflected_forward(Views.axis_extents(middle),[],row,Coordinates.forward_none(List.length(&2,U32,Views.axis_extents(middle))),middle_inside)))
  +column_turned = Equal.trans(Nat,
    Coordinates.reflected(Views.axis_extents(late),Views.drop_flags(List.length(&2,Views.Axis,middle),Views.drop_flags(List.length(&2,Views.Axis,early),flagged)),column),
    Coordinates.reflected(Views.axis_extents(after),late_fitted,column),turned_column,
    Equal.trans(Nat,Coordinates.reflected(Views.axis_extents(late),Views.drop_flags(List.length(&2,Views.Axis,middle),Views.drop_flags(List.length(&2,Views.Axis,early),flagged)),column),
      Coordinates.reflected(Views.axis_extents(late),late_fitted,column),Coordinates.reflected(Views.axis_extents(after),late_fitted,column),
      Equal.cong(List<&2,Bool>,Nat,flags => Coordinates.reflected(Views.axis_extents(late),flags,column),
        Views.drop_flags(List.length(&2,Views.Axis,middle),Views.drop_flags(List.length(&2,Views.Axis,early),flagged)),late_fitted,late_dropped),
      Equal.cong(List<&2,U32>,Nat,extents => Coordinates.reflected(extents,late_fitted,column),Views.axis_extents(late),Views.axis_extents(after),late_extents)),
    %Metadata.extents_length(after) : {Coordinates.reflected(Views.axis_extents(after),Views.fitted_flags(_,late_flags),column) == turned_column : Nat}
    Coordinates.reflected_fitted(Views.axis_extents(after),late_flags,column))
  +forward_position = Nat.add(Nat.mul(Nat.add(Nat.mul(channel,nc),row),nd),column)
  +turned_position = Nat.add(Nat.mul(Nat.add(Nat.mul(Coordinates.reflected(Views.axis_extents(early),flagged,channel),nc),
    Coordinates.reflected(Views.axis_extents(middle),Views.drop_flags(List.length(&2,Views.Axis,early),flagged),row)),nd),
    Coordinates.reflected(Views.axis_extents(late),Views.drop_flags(List.length(&2,Views.Axis,middle),Views.drop_flags(List.length(&2,Views.Axis,early),flagged)),column))
  +forward_index = RowMajor.index(channel,extent,ea,row,column)
  +turned_index = RowMajor.index(turned_channel,extent,ea,row,turned_column)
  +forward_exact = Equal.sym(Nat,forward_position,forward_index,
    Equal.trans(Nat,forward_position,Nat.add(Nat.mul(Nat.add(Nat.mul(channel,extent),row),ea),column),forward_index,
      Coordinates.digits_equal(channel,channel,nc,extent,row,row,nd,ea,column,column,{==},middle_elements,{==},late_elements,{==}),
      Coordinates.digit_join(channel,extent,row,ea,column)))
  +turned_exact = Equal.trans(Nat,turned_position,Nat.add(Nat.mul(Nat.add(Nat.mul(turned_channel,extent),row),ea),turned_column),turned_index,
    Coordinates.digits_equal(Coordinates.reflected(Views.axis_extents(early),flagged,channel),turned_channel,nc,extent,
      Coordinates.reflected(Views.axis_extents(middle),Views.drop_flags(List.length(&2,Views.Axis,early),flagged),row),row,nd,ea,
      Coordinates.reflected(Views.axis_extents(late),Views.drop_flags(List.length(&2,Views.Axis,middle),Views.drop_flags(List.length(&2,Views.Axis,early),flagged)),column),turned_column,
      channel_turned,middle_elements,row_kept,late_elements,column_turned),
    Coordinates.digit_join(turned_channel,extent,row,ea,turned_column))
  Equal.trans(U32,MirrorAddress.address(view,flagged,forward_index),MirrorAddress.address(view,flagged,forward_position),CopyValues.address(view,turned_index),
    Equal.cong(Nat,U32,position => MirrorAddress.address(view,flagged,position),forward_index,forward_position,forward_exact),
    Equal.trans(U32,MirrorAddress.address(view,flagged,forward_position),CopyValues.address(view,turned_position),CopyValues.address(view,turned_index),
      Coordinates.grouped([],early,middle,late,offset,flagged,0n,channel,row,column,{==},early_inside,middle_inside,late_inside),
      Equal.cong(Nat,U32,position => CopyValues.address(view,position),turned_position,turned_index,turned_exact)))

# The whole view of a banded form read with flags that read its images axis,
# its rows axis and the repeats of each row forward reads, at an output cell,
# what it reads forward at the cell of the reflected channel and column.
def whole_reflected(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,
  +reversed: List<&2,Bool>,+image: Nat,+channel: Nat,+row: Nat,+column: Nat,
  +batch_positive: {Nat.is_lt(0n,U32.to_nat(batch)) == True{} : Bool},+height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
  +forward: Forward(before,repeats,batch,height,reversed),
  +image_inside: {Nat.is_lt(image,U32.to_nat(batch)) == True{} : Bool},
  +channel_below: {Nat.is_lt(channel,ShapeLaws.elements(Views.axis_extents(before))) == True{} : Bool},
  +row_below: {Nat.is_lt(row,Nat.mul(U32.to_nat(height),ShapeLaws.elements(repeats))) == True{} : Bool},
  +column_below: {Nat.is_lt(column,ShapeLaws.elements(Views.axis_extents(after))) == True{} : Bool}) ->
  {MirrorAddress.address(Views.View{Classified.whole_axes(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width}),start},reversed,
      RowMajor.index(Nat.add(Nat.mul(image,ShapeLaws.elements(Views.axis_extents(before))),channel),Nat.mul(U32.to_nat(height),ShapeLaws.elements(repeats)),
        ShapeLaws.elements(Views.axis_extents(after)),row,column)) ==
    CopyValues.address(Views.View{Classified.whole_axes(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width}),start},
      RowMajor.index(Nat.add(Nat.mul(image,ShapeLaws.elements(Views.axis_extents(before))),Coordinates.reflected(Views.axis_extents(before),before_flags(batch,reversed),channel)),
        Nat.mul(U32.to_nat(height),ShapeLaws.elements(repeats)),ShapeLaws.elements(Views.axis_extents(after)),row,
        Coordinates.reflected(Views.axis_extents(after),after_flags(before,repeats,batch,height,reversed),column))) : U32}:
  Shape.Both{+images_forward,+rows_forward} = forward
  +many = U32.is_lt(1,batch)
  +rowed = U32.is_lt(1,height)
  +stride = (channels * (height * width : U32) : U32)
  +first = Classified.images_axes(many,batch,stride)
  +middle = BandViews.rows_axes(rowed,height,width,repeats)
  +images = BandViews.count_when(many)
  +lines = lines_count(height,repeats)
  +later = later_flags(before,batch,reversed)
  +view = {Views.View{Classified.whole_axes(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width}),start} : Views.View}
  +eb = ShapeLaws.elements(Views.axis_extents(before))
  +ea = ShapeLaws.elements(Views.axis_extents(after))
  +extent = Nat.mul(U32.to_nat(height),ShapeLaws.elements(repeats))
  +nc = ShapeLaws.elements(Views.axis_extents(middle))
  +turned_channel = Coordinates.reflected(Views.axis_extents(before),before_flags(batch,reversed),channel)
  +turned_column = Coordinates.reflected(Views.axis_extents(after),after_flags(before,repeats,batch,height,reversed),column)
  +first_elements = images_elements(many,batch,stride,{==},batch_positive)
  +middle_elements = rows_elements(rowed,height,width,repeats,{==},height_positive)
  +first_count = images_count(many,batch,stride)
  +middle_count = rows_count(rowed,height,width,repeats)
  +image_below = {%Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(first)),U32.to_nat(batch),first_elements) : {Nat.is_lt(image,_) == True{} : Bool}
    image_inside : {Nat.is_lt(image,ShapeLaws.elements(Views.axis_extents(first))) == True{} : Bool}}
  +row_inside = {%Equal.sym(Nat,nc,extent,middle_elements) : {Nat.is_lt(row,_) == True{} : Bool} row_below : {Nat.is_lt(row,nc) == True{} : Bool}}
  +first_dropped = Equal.cong(Nat,List<&2,Bool>,count => Views.drop_flags(count,reversed),List.length(&2,Views.Axis,first),images,first_count)
  +before_dropped = Equal.trans(List<&2,Bool>,Views.drop_flags(List.length(&2,Views.Axis,before),Views.drop_flags(List.length(&2,Views.Axis,first),reversed)),
    Views.drop_flags(List.length(&2,Views.Axis,before),Views.drop_flags(images,reversed)),later,
    Equal.cong(List<&2,Bool>,List<&2,Bool>,flags => Views.drop_flags(List.length(&2,Views.Axis,before),flags),Views.drop_flags(List.length(&2,Views.Axis,first),reversed),
      Views.drop_flags(images,reversed),first_dropped),
    Coordinates.drop_drop(images,List.length(&2,Views.Axis,before),reversed))
  +middle_dropped = Equal.trans(List<&2,Bool>,
    Views.drop_flags(List.length(&2,Views.Axis,middle),Views.drop_flags(List.length(&2,Views.Axis,before),Views.drop_flags(List.length(&2,Views.Axis,first),reversed))),
    Views.drop_flags(List.length(&2,Views.Axis,middle),later),Views.drop_flags(lines,later),
    Equal.cong(List<&2,Bool>,List<&2,Bool>,flags => Views.drop_flags(List.length(&2,Views.Axis,middle),flags),
      Views.drop_flags(List.length(&2,Views.Axis,before),Views.drop_flags(List.length(&2,Views.Axis,first),reversed)),later,before_dropped),
    Equal.cong(Nat,List<&2,Bool>,count => Views.drop_flags(count,later),List.length(&2,Views.Axis,middle),lines,middle_count))
  +image_kept = Coordinates.reflected_forward(Views.axis_extents(first),reversed,image,
    %Equal.sym(Nat,List.length(&2,U32,Views.axis_extents(first)),images,
        Equal.trans(Nat,List.length(&2,U32,Views.axis_extents(first)),List.length(&2,Views.Axis,first),images,Metadata.extents_length(first),first_count)) :
      {Views.forward(_,reversed) == True{} : Bool}
    images_forward,image_below)
  +channel_turned = Equal.cong(List<&2,Bool>,Nat,flags => Coordinates.reflected(Views.axis_extents(before),flags,channel),
    Views.drop_flags(List.length(&2,Views.Axis,first),reversed),Views.drop_flags(images,reversed),first_dropped)
  +row_kept = Equal.trans(Nat,
    Coordinates.reflected(Views.axis_extents(middle),Views.drop_flags(List.length(&2,Views.Axis,before),Views.drop_flags(List.length(&2,Views.Axis,first),reversed)),row),
    Coordinates.reflected(Views.axis_extents(middle),later,row),row,
    Equal.cong(List<&2,Bool>,Nat,flags => Coordinates.reflected(Views.axis_extents(middle),flags,row),
      Views.drop_flags(List.length(&2,Views.Axis,before),Views.drop_flags(List.length(&2,Views.Axis,first),reversed)),later,before_dropped),
    Coordinates.reflected_forward(Views.axis_extents(middle),later,row,
      %Equal.sym(Nat,List.length(&2,U32,Views.axis_extents(middle)),lines,
          Equal.trans(Nat,List.length(&2,U32,Views.axis_extents(middle)),List.length(&2,Views.Axis,middle),lines,Metadata.extents_length(middle),middle_count)) :
        {Views.forward(_,later) == True{} : Bool}
      rows_forward,row_inside))
  +column_turned = Equal.cong(List<&2,Bool>,Nat,flags => Coordinates.reflected(Views.axis_extents(after),flags,column),
    Views.drop_flags(List.length(&2,Views.Axis,middle),Views.drop_flags(List.length(&2,Views.Axis,before),Views.drop_flags(List.length(&2,Views.Axis,first),reversed))),
    Views.drop_flags(lines,later),middle_dropped)
  +lead = Nat.add(Nat.mul(image,eb),channel)
  +turned_lead = Nat.add(Nat.mul(image,eb),turned_channel)
  +image_reflected = Coordinates.reflected(Views.axis_extents(first),reversed,image)
  +channel_reflected = Coordinates.reflected(Views.axis_extents(before),Views.drop_flags(List.length(&2,Views.Axis,first),reversed),channel)
  +row_reflected = Coordinates.reflected(Views.axis_extents(middle),Views.drop_flags(List.length(&2,Views.Axis,before),Views.drop_flags(List.length(&2,Views.Axis,first),reversed)),row)
  +column_reflected = Coordinates.reflected(Views.axis_extents(after),
    Views.drop_flags(List.length(&2,Views.Axis,middle),Views.drop_flags(List.length(&2,Views.Axis,before),Views.drop_flags(List.length(&2,Views.Axis,first),reversed))),column)
  +forward_position = Nat.add(Nat.mul(Nat.add(Nat.mul(lead,nc),row),ea),column)
  +turned_position = Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(image_reflected,eb),channel_reflected),nc),row_reflected),ea),column_reflected)
  +forward_index = RowMajor.index(lead,extent,ea,row,column)
  +turned_index = RowMajor.index(turned_lead,extent,ea,row,turned_column)
  +lead_turned = Equal.trans(Nat,Nat.add(Nat.mul(image_reflected,eb),channel_reflected),Nat.add(Nat.mul(image,eb),channel_reflected),turned_lead,
    Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(value,eb),channel_reflected),image_reflected,image,image_kept),
    Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(image,eb),value),channel_reflected,turned_channel,channel_turned))
  +forward_exact = Equal.sym(Nat,forward_position,forward_index,
    Equal.trans(Nat,forward_position,Nat.add(Nat.mul(Nat.add(Nat.mul(lead,extent),row),ea),column),forward_index,
      Coordinates.digits_equal(lead,lead,nc,extent,row,row,ea,ea,column,column,{==},middle_elements,{==},{==},{==}),
      Coordinates.digit_join(lead,extent,row,ea,column)))
  +turned_exact = Equal.trans(Nat,turned_position,Nat.add(Nat.mul(Nat.add(Nat.mul(turned_lead,extent),row),ea),turned_column),turned_index,
    Coordinates.digits_equal(Nat.add(Nat.mul(image_reflected,eb),channel_reflected),turned_lead,nc,extent,row_reflected,row,ea,ea,column_reflected,turned_column,
      lead_turned,middle_elements,row_kept,{==},column_turned),
    Coordinates.digit_join(turned_lead,extent,row,ea,turned_column))
  Equal.trans(U32,MirrorAddress.address(view,reversed,forward_index),MirrorAddress.address(view,reversed,forward_position),CopyValues.address(view,turned_index),
    Equal.cong(Nat,U32,position => MirrorAddress.address(view,reversed,position),forward_index,forward_position,forward_exact),
    Equal.trans(U32,MirrorAddress.address(view,reversed,forward_position),CopyValues.address(view,turned_position),CopyValues.address(view,turned_index),
      Coordinates.grouped(first,before,middle,after,start,reversed,image,channel,row,column,image_below,channel_below,row_inside,column_below),
      Equal.cong(Nat,U32,position => CopyValues.address(view,position),turned_position,turned_index,turned_exact)))

# The view read with its flags at the pixel of an output cell of a band reads
# what it reads forward at the pixel of the reflected channel and column.
def view_reflected(+view_axes: List<&2,Views.Axis>,+offset: U32,
  +before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,+limits: Machine.Limits,
  +words: Words(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width}),
  +classified: Classified.Classified<view_axes,offset,Bands.Geometry{batch,channels,height,width},BandViews.Banded{before,repeats,after,start}>,
  +rows_room: {Nat.is_le(Nat.mul(U32.to_nat(batch),U32.to_nat((height * BandViews.product_or_zero(repeats) : U32))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +first: U32,+rows: U32,+source_length: Nat,+fields: BandShape.Leaf(Bands.Geometry{batch,channels,height,width},first,rows,source_length),
  +reversed: List<&2,Bool>,+forward: Forward(before,repeats,batch,height,reversed),+channel: Nat,+row: Nat,+column: Nat,
  +channel_below: {Nat.is_lt(channel,ShapeLaws.elements(Views.axis_extents(before))) == True{} : Bool},
  +row_below: {Nat.is_lt(row,Nat.mul(U32.to_nat(rows),ShapeLaws.elements(repeats))) == True{} : Bool},
  +column_below: {Nat.is_lt(column,ShapeLaws.elements(Views.axis_extents(after))) == True{} : Bool}) ->
  {MirrorAddress.address(Views.View{view_axes,offset},reversed,
      BandValues.pixel(BandViews.output(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width}),channel,
        Nat.add(U32.to_nat((first * BandViews.product_or_zero(repeats) : U32)),row),column)) ==
    CopyValues.address(Views.View{view_axes,offset},
      BandValues.pixel(BandViews.output(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width}),
        Coordinates.reflected(Views.axis_extents(before),before_flags(batch,reversed),channel),Nat.add(U32.to_nat((first * BandViews.product_or_zero(repeats) : U32)),row),
        Coordinates.reflected(Views.axis_extents(after),after_flags(before,repeats,batch,height,reversed),column))) : U32}:
  Shape.Both{Fitted{+batch_positive,+height_positive,channel_reach,column_reach,early,late,repeated,rows_exact},Plane{divisor,plane_exact,image_exact}} = words
  Classified.Classified{+axes_exact,+offset_exact,fits} = classified
  +output = BandViews.output(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width})
  +global = Nat.add(U32.to_nat((first * BandViews.product_or_zero(repeats) : U32)),row)
  +turned_channel = Coordinates.reflected(Views.axis_extents(before),before_flags(batch,reversed),channel)
  +turned_column = Coordinates.reflected(Views.axis_extents(after),after_flags(before,repeats,batch,height,reversed),column)
  +eb = ShapeLaws.elements(Views.axis_extents(before))
  +ea = ShapeLaws.elements(Views.axis_extents(after))
  +extent = Nat.mul(U32.to_nat(height),ShapeLaws.elements(repeats))
  +image = row_image(first,row,height,repeats)
  +inside_row = image_row(first,row,height,repeats)
  +whole = Classified.whole_axes(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width})
  +forward_index = RowMajor.index(Nat.add(Nat.mul(image,eb),channel),extent,ea,inside_row,column)
  +turned_index = RowMajor.index(Nat.add(Nat.mul(image,eb),turned_channel),extent,ea,inside_row,turned_column)
  %Equal.sym(Nat,BandValues.pixel(output,channel,global,column),forward_index,
      pixel_split(before,repeats,after,start,batch,channels,height,width,limits,words,rows_room,first,rows,source_length,fields,channel,row,column,row_below)) :
    {MirrorAddress.address(Views.View{view_axes,offset},reversed,_) == CopyValues.address(Views.View{view_axes,offset},BandValues.pixel(output,turned_channel,global,turned_column)) : U32}
  %Equal.sym(Nat,BandValues.pixel(output,turned_channel,global,turned_column),turned_index,
      pixel_split(before,repeats,after,start,batch,channels,height,width,limits,words,rows_room,first,rows,source_length,fields,turned_channel,row,turned_column,row_below)) :
    {MirrorAddress.address(Views.View{view_axes,offset},reversed,forward_index) == CopyValues.address(Views.View{view_axes,offset},_) : U32}
  %Equal.sym(List<&2,Views.Axis>,view_axes,whole,axes_exact) :
    {MirrorAddress.address(Views.View{_,offset},reversed,forward_index) == CopyValues.address(Views.View{_,offset},turned_index) : U32}
  %offset_exact : {MirrorAddress.address(Views.View{whole,_},reversed,forward_index) == CopyValues.address(Views.View{whole,_},turned_index) : U32}
  whole_reflected(before,repeats,after,start,batch,channels,height,width,reversed,image,channel,inside_row,column,batch_positive,height_positive,forward,
    row_image_below(batch,channels,height,width,first,rows,source_length,repeats,row,fields,height_positive,row_below),channel_below,
    image_row_below(first,rows,row,height,repeats,height_positive,row_below),column_below)

# One cell of an output band: the leaf gather of the band through its local
# view stores, at the position of the cell, what the local view reads there
# with its flags.
def cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+evidence: Cells<Element,Logical,original>,+channel: Nat,+row: Nat,+column: Nat,
  channel_inside: {Nat.is_lt(channel,BandValues.channels(cells_geometry(~Element,~Logical,~original,evidence))) == True{} : Bool},
  row_inside: {Nat.is_lt(row,U32.to_nat(cells_rows(~Element,~Logical,~original,evidence))) == True{} : Bool},
  column_inside: {Nat.is_lt(column,BandValues.width(cells_geometry(~Element,~Logical,~original,evidence))) == True{} : Bool}) ->
  CellAt(~Element,~Logical,~original,evidence,channel,row,column):
  Cells{+scope,+first,+rows,+input,+model,+input_fact,+viewed} = evidence
  Scope{+logical,name,+view_axes,+offset,+reversed,+banded,+geometry,+local,fill,+limits,+words,+classified,+directions,+room} = scope
  BandViews.Banded{+before,+repeats,+after,+start} = banded
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  Shape.Both{Fitted{batch_positive,height_positive,channel_reach,column_reach,+early,+late,+repeated,+rows_exact},Plane{divisor,plane_exact,image_exact}} = words
  Buffer.Model{+source_tree,+source_length,+source_certificate} = input
  Buffer.Model{+tree,+length,+certificate} = model
  Shape.Both{Shape.Both{+fields,+source_ready},+source_values} = input_fact
  Shape.Both{Contract.WellFormed{+length_exact,+ready},+values} = viewed
  +channel_inside = channel_inside
  +row_inside = row_inside
  +column_inside = column_inside
  +scale = BandViews.product_or_zero(repeats)
  +before_count = BandViews.product_or_zero(Views.axis_extents(before))
  +after_count = BandViews.product_or_zero(Views.axis_extents(after))
  +output = {Bands.Geometry{batch,before_count,(height * scale : U32),after_count} : Bands.Geometry}
  +capacity = Machine.capacity(limits)
  +es = ShapeLaws.elements(repeats)
  +eb = ShapeLaws.elements(Views.axis_extents(before))
  +ea = ShapeLaws.elements(Views.axis_extents(after))
  +first_n = U32.to_nat(first)
  +rows_n = U32.to_nat(rows)
  +rows_room = BandGather.rows_room(output,limits,capacity,room)
  +rows_fit = band_rows_fit(Bands.Geometry{batch,channels,height,width},first,rows,source_length,fields)
  +fit = scaled_rows_fit(batch,height,scale,first,rows,rows_fit,rows_exact)
  +fit_room = Order.transitive(Nat.mul(Nat.add(first_n,rows_n),U32.to_nat(scale)),Nat.mul(U32.to_nat(batch),U32.to_nat((height * scale : U32))),U32.to_nat(capacity),fit,rows_room)
  +first_exact = scaled_first_exact(first,rows,scale,capacity,fit_room)
  +band_rows_exact = scaled_rows_exact(first,rows,scale,capacity,fit_room)
  +rows_es = {%repeated : {U32.to_nat((rows * scale : U32)) == Nat.mul(rows_n,_) : Nat} band_rows_exact : {U32.to_nat((rows * scale : U32)) == Nat.mul(rows_n,es) : Nat}}
  +row_below = {%rows_es : {Nat.is_lt(row,_) == True{} : Bool} row_inside : {Nat.is_lt(row,Nat.mul(rows_n,es)) == True{} : Bool}}
  +channel_below = {%early : {Nat.is_lt(channel,_) == True{} : Bool} channel_inside : {Nat.is_lt(channel,eb) == True{} : Bool}}
  +column_below = {%late : {Nat.is_lt(column,_) == True{} : Bool} column_inside : {Nat.is_lt(column,ea) == True{} : Bool}}
  +index = RowMajor.index(channel,Nat.mul(rows_n,es),ea,row,column)
  +position = BandValues.position(output,(rows * scale : U32),channel,row,column)
  +position_exact = {
    %Equal.sym(Nat,U32.to_nat((rows * scale : U32)),Nat.mul(rows_n,es),rows_es) : {RowMajor.index(channel,_,U32.to_nat(after_count),row,column) == index : Nat}
    %Equal.sym(Nat,U32.to_nat(after_count),ea,late) : {RowMajor.index(channel,Nat.mul(rows_n,es),_,row,column) == index : Nat}
    {==} : {position == index : Nat}}
  +elements_exact = Equal.trans(Nat,length,ShapeLaws.elements(BandViews.extents(BandViews.Banded{before,repeats,after,start},rows)),Nat.mul(eb,Nat.mul(Nat.mul(rows_n,es),ea)),length_exact,
    Equal.trans(Nat,ShapeLaws.elements(List.append(&2,U32,Views.axis_extents(before),List.append(&2,U32,rows <> repeats,Views.axis_extents(after)))),
      Nat.mul(eb,ShapeLaws.elements(List.append(&2,U32,rows <> repeats,Views.axis_extents(after)))),Nat.mul(eb,Nat.mul(Nat.mul(rows_n,es),ea)),
      ShapeLaws.elements_appended(Views.axis_extents(before),List.append(&2,U32,rows <> repeats,Views.axis_extents(after))),
      Equal.cong(Nat,Nat,value => Nat.mul(eb,value),ShapeLaws.elements(List.append(&2,U32,rows <> repeats,Views.axis_extents(after))),Nat.mul(Nat.mul(rows_n,es),ea),
        ShapeLaws.elements_appended(rows <> repeats,Views.axis_extents(after)))))
  +index_below = {
    %Equal.sym(Nat,length,Nat.mul(eb,Nat.mul(Nat.mul(rows_n,es),ea)),elements_exact) : {Nat.is_lt(index,_) == True{} : Bool}
    RowMajor.in_bounds(channel,eb,Nat.mul(rows_n,es),ea,row,column,channel_below,row_below,column_below) : {Nat.is_lt(index,length) == True{} : Bool}}
  +bound = {%Equal.sym(Nat,position,index,position_exact) : {Nat.is_lt(_,length) == True{} : Bool} index_below : {Nat.is_lt(position,length) == True{} : Bool}}
  +index_word = FullRange.roundtrip(index,Tree.capacity(Element,tree),
    Order.transitive(index,length,U32.to_nat(Tree.capacity(Element,tree)),Bounds.strict_le(index,length,index_below),content_room(~Element,Buffer.Content{tree,length},ready)))
  +band_view = BandViews.local(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},rows)
  +images = BandViews.count_when(U32.is_lt(1,batch))
  +lines = lines_count(height,repeats)
  +later = later_flags(before,batch,reversed)
  +early_flags = before_flags(batch,reversed)
  +late_flags = after_flags(before,repeats,batch,height,reversed)
  +listed = BandViews.local_directions(before,repeats,after,early_flags,late_flags)
  +valid = directions_valid(Views.forward(images,reversed) && Views.forward(lines,later),listed,local,directions)
  +local_exact = directions_exact(Views.forward(images,reversed) && Views.forward(lines,later),listed,local,directions)
  +forward = {Shape.Both{Boolean.left(Views.forward(images,reversed),Views.forward(lines,later),valid),Boolean.right(Views.forward(images,reversed),Views.forward(lines,later),valid)} :
    Forward(before,repeats,batch,height,reversed)}
  +turned_channel = Coordinates.reflected(Views.axis_extents(before),early_flags,channel)
  +turned_column = Coordinates.reflected(Views.axis_extents(after),late_flags,column)
  +turned_index = RowMajor.index(turned_channel,Nat.mul(rows_n,es),ea,row,turned_column)
  +stored = {
    %index_word : {Tree.read(Element,tree,U32.from_nat(index)) == Tree.read(Element,source_tree,MirrorAddress.address(band_view,local,_)) : Element}
    Contract.read(~Element,~GatherValues.Directed<Buffer.Content<Element>>,~GatherValues.directed(~Element,~Buffer.Content<Element>,~Contract.stored(~Element)),
      GatherValues.Directed{Buffer.Content{source_tree,source_length},band_view,local},Buffer.Content{tree,length},index,values,index_below)
    : {Tree.read(Element,tree,U32.from_nat(index)) == Tree.read(Element,source_tree,MirrorAddress.address(band_view,local,index)) : Element}}
  +moved = {
    %local_exact : {MirrorAddress.address(band_view,_,index) == CopyValues.address(band_view,turned_index) : U32}
    local_reflected(before,repeats,after,start,batch,channels,height,width,rows,early_flags,late_flags,channel,row,column,channel_below,row_below,column_below)
    : {MirrorAddress.address(band_view,local,index) == CopyValues.address(band_view,turned_index) : U32}}
  +global = Nat.add(U32.to_nat((first * scale : U32)),row)
  +pixel = BandValues.pixel(output,channel,global,column)
  +turned_pixel = BandValues.pixel(output,turned_channel,global,turned_column)
  +global_inside = scaled_row_inside(batch,height,scale,first,rows,row,row_inside,first_exact,band_rows_exact,fit)
  +pixel_word = FullRange.roundtrip(pixel,capacity,
    Order.transitive(pixel,BandGather.elements(output),U32.to_nat(capacity),
      Bounds.strict_le(pixel,BandGather.elements(output),BandGather.pixel_below(output,channel,global,column,channel_inside,global_inside,column_inside,domain_height(output,limits,capacity,room))),
      domain_elements(output,limits,capacity,room)))
  +read = {
    %Equal.sym(Nat,position,index,position_exact) :
      {Tree.read(Element,tree,U32.from_nat(_)) == original(logical,MirrorAddress.address(Views.View{view_axes,offset},reversed,U32.to_nat(U32.from_nat(pixel)))) : Element}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(pixel)),pixel,pixel_word) :
      {Tree.read(Element,tree,U32.from_nat(index)) == original(logical,MirrorAddress.address(Views.View{view_axes,offset},reversed,_)) : Element}
    Equal.trans(Element,Tree.read(Element,tree,U32.from_nat(index)),Tree.read(Element,source_tree,MirrorAddress.address(band_view,local,index)),
      original(logical,MirrorAddress.address(Views.View{view_axes,offset},reversed,pixel)),stored,
      Equal.trans(Element,Tree.read(Element,source_tree,MirrorAddress.address(band_view,local,index)),Tree.read(Element,source_tree,CopyValues.address(band_view,turned_index)),
        original(logical,MirrorAddress.address(Views.View{view_axes,offset},reversed,pixel)),
        Equal.cong(U32,Element,word => Tree.read(Element,source_tree,word),MirrorAddress.address(band_view,local,index),CopyValues.address(band_view,turned_index),moved),
        %Equal.sym(U32,MirrorAddress.address(Views.View{view_axes,offset},reversed,pixel),CopyValues.address(Views.View{view_axes,offset},turned_pixel),
            view_reflected(view_axes,offset,before,repeats,after,start,batch,channels,height,width,limits,words,classified,rows_room,first,rows,source_length,fields,
              reversed,forward,channel,row,column,channel_below,row_below,column_below)) :
          {Tree.read(Element,source_tree,CopyValues.address(band_view,turned_index)) == original(logical,_) : Element}
        local_read(~Element,~Logical,~original,logical,view_axes,offset,before,repeats,after,start,batch,channels,height,width,limits,words,classified,rows_room,
          first,rows,source_tree,source_length,fields,source_ready,source_values,turned_channel,row,turned_column,
          Coordinates.reflected_below(before,early_flags,channel,channel_below),row_below,Coordinates.reflected_below(after,late_flags,column,column_below))))
    : {Tree.read(Element,tree,U32.from_nat(position)) ==
        GatherValues.directed(~Element,~Logical,~original,GatherValues.Directed{logical,Views.View{view_axes,offset},reversed},U32.from_nat(pixel)) : Element}}
  Shape.Both{bound,read}

# Every cell of an output band, from the leaf gather of its input band.
def values_output(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: Scope<Element,Logical>,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+model: Buffer.Model<Element>,
  +input_fact: Input(~Element,~Logical,~original,scope,first,rows,Buffer.content(~Element,input)),
  +viewed: Viewed(~Element,~Logical,scope,rows,Buffer.content(~Element,input),Buffer.content(~Element,model))) ->
  OutputValues(~Element,~Logical,~original,scope,scaled_word(~Element,~Logical,scope,first),scaled_word(~Element,~Logical,scope,rows),Buffer.content(~Element,model)):
  Scope{+logical,+name,+view_axes,+offset,+reversed,+banded,+geometry,+local,+fill,+limits,+words,+classified,+directions,+room} = scope
  Buffer.Model{+tree,+length,+certificate} = model
  +scope = {Scope{logical,name,view_axes,offset,reversed,banded,geometry,local,fill,limits,words,classified,directions,room} : Scope<Element,Logical>}
  +model = {Buffer.Model{tree,length,certificate} : Buffer.Model<Element>}
  BandValues.cells(~Cells<Element,Logical,original>,~CellAt(~Element,~Logical,~original),~cells_geometry(~Element,~Logical,~original),~cells_rows(~Element,~Logical,~original),~cell(~Element,~Logical,~original),
    Cells{scope,first,rows,input,model,input_fact,viewed})

# ---- One band through its leaf gather ----

def usable_ready(+status: Tensors.Status,accepted: {Tensors.usable(status) == True{} : Bool}) -> {Tensors.usable(status) == True{} : Bool}:
  match status:
    case Tensors.Usable{}: {==}
    case Tensors.Failed{problem}: accepted

# A checked band is usable only when its call returned a usable whole tensor.
def checked_ready(~Element: Data,+first: U32,+rows: U32,result: Tensors.Tensor<Element>,
  accepted: {Each.usable(~Element,Each.checked(~Element,Tensors.checked_band(~Element,first,rows,result))) == True{} : Bool}) ->
  {Contract.tensor_ready(~Element,result) == True{} : Bool}:
  match result:
    case Tensors.Tensor{bands,geometry,shape,fill,+status,workers}: usable_ready(status,accepted)

def viewed_ready(~Element: Data,~Logical: Data,+scope: Scope<Element,Logical>,+rows: U32,+input: Buffer.Content<Element>,+output: Buffer.Content<Element>,
  viewed: Viewed(~Element,~Logical,scope,rows,input,output)) -> Buffer.ContentReady<Element,output>:
  Shape.Both{Contract.WellFormed{length_exact,ready},values} = viewed
  ready

# An output band is ready and holds the values of the view.
def Output(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: Scope<Element,Logical>) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  Global.Made(~Element,~Scope<Element,Logical>,~OutputValues(~Element,~Logical,~original),scope)

def output(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: Scope<Element,Logical>,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+model: Buffer.Model<Element>,
  +input_fact: Input(~Element,~Logical,~original,scope,first,rows,Buffer.content(~Element,input)),
  +viewed: Viewed(~Element,~Logical,scope,rows,Buffer.content(~Element,input),Buffer.content(~Element,model))) ->
  Output(~Element,~Logical,~original,scope,scaled_word(~Element,~Logical,scope,first),scaled_word(~Element,~Logical,scope,rows),Buffer.content(~Element,model)):
  Shape.Both{viewed_ready(~Element,~Logical,scope,rows,Buffer.content(~Element,input),Buffer.content(~Element,model),viewed),
    values_output(~Element,~Logical,~original,scope,first,rows,input,model,input_fact,viewed)}

# The gathered band, named by its buffer model.
def stored_output(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,
  +scope: Scope<Element,Logical>,+first: U32,+rows: U32,+input: Buffer.Model<Element>,
  +input_fact: Input(~Element,~Logical,~original,scope,first,rows,Buffer.content(~Element,input)),-storage: Storage.Buffer<Element>,
  observation: Buffer.Observation<Element,storage>,
  viewed: Viewed(~Element,~Logical,scope,rows,Buffer.content(~Element,input),Buffer.contents(~Element,storage))) ->
  Output(~Element,~Logical,~original,scope,scaled_word(~Element,~Logical,scope,first),scaled_word(~Element,~Logical,scope,rows),Buffer.contents(~Element,storage)):
  Buffer.Observation{+model,+equation} = observation
  %equation : Output(~Element,~Logical,~original,scope,scaled_word(~Element,~Logical,scope,first),scaled_word(~Element,~Logical,scope,rows),Buffer.contents(~Element,_))
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,model)),Buffer.content(~Element,model),Buffer.packed_contents(~Element,model)) :
    Output(~Element,~Logical,~original,scope,scaled_word(~Element,~Logical,scope,first),scaled_word(~Element,~Logical,scope,rows),_)
  output(~Element,~Logical,~original,scope,first,rows,input,model,input_fact,
    %Buffer.packed_contents(~Element,model) : Viewed(~Element,~Logical,scope,rows,Buffer.content(~Element,input),_)
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,model),storage,equation) : Viewed(~Element,~Logical,scope,rows,Buffer.content(~Element,input),Buffer.contents(~Element,_))
    viewed)

def Made(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: Scope<Element,Logical>,+first: U32,+rows: U32,result: Tensors.Tensor<Element>) -> Data:
  Model.ContentAll(~Element,~Output(~Element,~Logical,~original,scope),Model.contents(~Element,Each.made(~Element,Each.checked(~Element,
    Tensors.checked_band(~Element,scaled_word(~Element,~Logical,scope,first),scaled_word(~Element,~Logical,scope,rows),result)))))

# An accepted copy leaves the band of its output.
def copied_output(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,outcome: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>,
  +scope: Scope<Element,Logical>,+first: U32,+rows: U32,+input: Buffer.Model<Element>,
  +input_fact: Input(~Element,~Logical,~original,scope,first,rows,Buffer.content(~Element,input)),+name: String,+source_shape: List<&2,U32>,+fill: Element,
  viewed: Viewed(~Element,~Logical,scope,rows,Buffer.content(~Element,input),CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,outcome))),
  accepted: {Each.usable(~Element,Each.checked(~Element,Tensors.checked_band(~Element,scaled_word(~Element,~Logical,scope,first),scaled_word(~Element,~Logical,scope,rows),
    Tensors.result_of(~Element,Tensors.copied(~Element,name,source_shape,band_extents(~Element,~Logical,scope,rows),fill,1,outcome))))) == True{} : Bool}) ->
  Made(~Element,~Logical,~original,scope,first,rows,Tensors.result_of(~Element,Tensors.copied(~Element,name,source_shape,band_extents(~Element,~Logical,scope,rows),fill,1,outcome))):
  match outcome:
    case Operations.Accepted{(kept,Storage.Buffer{storage,length,certificate})}:
      stored_output(~Element,~Logical,~original,scope,first,rows,input,input_fact,Storage.Buffer{storage,length,certificate},
        Buffer.observe(~Element,Storage.Buffer{storage,length,certificate}),viewed)
    case Operations.Rejected{(kept,output)}:
      Empty.absurd(Made(~Element,~Logical,~original,scope,first,rows,
        Tensors.result_of(~Element,Tensors.copied(~Element,name,source_shape,band_extents(~Element,~Logical,scope,rows),fill,1,Operations.Rejected{(kept,output)}))),
        WordEncoding.false_true(accepted))

# The call is one checked copy through the local view into fresh storage laid
# out as its extents.
def viewed_output(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,target: Maybe<&2,Views.View>,count: Maybe<&2,U32>,
  +scope: Scope<Element,Logical>,+first: U32,+rows: U32,+input: Buffer.Model<Element>,
  +input_fact: Input(~Element,~Logical,~original,scope,first,rows,Buffer.content(~Element,input)),+name: String,+source_shape: List<&2,U32>,+fill: Element,
  +input_ready: Buffer.Sized<Element,input>,
  +targets: {Tensors.view_of(band_extents(~Element,~Logical,scope,rows)) == target : Maybe<&2,Views.View>},
  +counts: {Tensors.count_of(band_extents(~Element,~Logical,scope,rows)) == count : Maybe<&2,U32>},
  +accepted: {Each.usable(~Element,Each.checked(~Element,Tensors.checked_band(~Element,scaled_word(~Element,~Logical,scope,first),scaled_word(~Element,~Logical,scope,rows),
    Tensors.result_of(~Element,Tensors.gather_views(~Element,name,Some{source(~Element,~Logical,scope,rows)},scope_local(~Element,~Logical,scope),target,count,Buffer.pack(~Element,input),
      source_shape,band_extents(~Element,~Logical,scope,rows),fill,1))))) == True{} : Bool}) ->
  Made(~Element,~Logical,~original,scope,first,rows,Tensors.result_of(~Element,Tensors.gather_views(~Element,name,Some{source(~Element,~Logical,scope,rows)},scope_local(~Element,~Logical,scope),target,count,Buffer.pack(~Element,input),
    source_shape,band_extents(~Element,~Logical,scope,rows),fill,1))):
  match target count:
    case Some{+target} Some{+count}:
      +view = source(~Element,~Logical,scope,rows)
      +shape = band_extents(~Element,~Logical,scope,rows)
      +flags = scope_local(~Element,~Logical,scope)
      +ready = checked_ready(~Element,scaled_word(~Element,~Logical,scope,first),scaled_word(~Element,~Logical,scope,rows),
        Tensors.result_of(~Element,Tensors.copied(~Element,name,source_shape,shape,fill,1,Operations.copy(~Element,Buffer.pack(~Element,input),view,flags,Tensors.buffer(~Element,fill,count),target))),accepted)
      copied_output(~Element,~Logical,~original,Operations.copy(~Element,Buffer.pack(~Element,input),view,flags,Tensors.buffer(~Element,fill,count),target),scope,first,rows,input,input_fact,name,source_shape,fill,
        GatherValues.targeted_facts(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),target,Buffer.content(~Element,input),view,flags,input,shape,shape,fill,count,
          input_ready,Contract.stored_values(~Element,Buffer.content(~Element,input)),targets,counts,count => counted => {==},
          Equal.trans(Bool,GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),view,flags,Tensors.buffer(~Element,fill,count),target))),
            Contract.tensor_ready(~Element,Tensors.result_of(~Element,Tensors.copied(~Element,name,source_shape,shape,fill,1,Operations.copy(~Element,Buffer.pack(~Element,input),view,flags,Tensors.buffer(~Element,fill,count),target)))),True{},
            Equal.sym(Bool,Contract.tensor_ready(~Element,Tensors.result_of(~Element,Tensors.copied(~Element,name,source_shape,shape,fill,1,Operations.copy(~Element,Buffer.pack(~Element,input),view,flags,Tensors.buffer(~Element,fill,count),target)))),
              GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),view,flags,Tensors.buffer(~Element,fill,count),target))),
              GatherValues.ready_accepted(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),view,flags,Tensors.buffer(~Element,fill,count),target),name,source_shape,shape,fill,1)),ready)),
        accepted)
    case Some{target} None{}:
      Empty.absurd(Made(~Element,~Logical,~original,scope,first,rows,Tensors.result_of(~Element,Tensors.gather_views(~Element,name,Some{source(~Element,~Logical,scope,rows)},scope_local(~Element,~Logical,scope),Some{target},None{},
        Buffer.pack(~Element,input),source_shape,band_extents(~Element,~Logical,scope,rows),fill,1))),WordEncoding.false_true(accepted))
    case None{} Some{count}:
      Empty.absurd(Made(~Element,~Logical,~original,scope,first,rows,Tensors.result_of(~Element,Tensors.gather_views(~Element,name,Some{source(~Element,~Logical,scope,rows)},scope_local(~Element,~Logical,scope),None{},Some{count},
        Buffer.pack(~Element,input),source_shape,band_extents(~Element,~Logical,scope,rows),fill,1))),WordEncoding.false_true(accepted))
    case None{} None{}:
      Empty.absurd(Made(~Element,~Logical,~original,scope,first,rows,Tensors.result_of(~Element,Tensors.gather_views(~Element,name,Some{source(~Element,~Logical,scope,rows)},scope_local(~Element,~Logical,scope),None{},None{},
        Buffer.pack(~Element,input),source_shape,band_extents(~Element,~Logical,scope,rows),fill,1))),WordEncoding.false_true(accepted))

# A band of no rows leaves an empty output band.
def empty_output(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: Scope<Element,Logical>,+first: U32,+rows: U32,+fill: Element,
  +extents: List<&2,U32>,+empty: {U32.is_zero(rows) == True{} : Bool}) ->
  Made(~Element,~Logical,~original,scope,first,rows,Tensors.whole(~Element,Tensors.buffer(~Element,fill,0),extents,fill,Tensors.Usable{},1)):
  Scope{+logical,name,+view_axes,+offset,+reversed,+banded,+geometry,+local,scope_fill,limits,words,classified,directions,room} = scope
  +output = BandViews.output(banded,geometry)
  +scale = BandViews.scale(banded)
  +fresh = {Buffer.Content{Tree.fresh(Element,Tensors.depth_for(0),fill),0n} : Buffer.Content<Element>}
  +no_cells = {
    %Equal.sym(U32,rows,0,Arithmetic.equal_of_is_eq(rows,0,empty)) : {BandValues.cell_count(output,(_ * scale : U32)) == 0n : Nat}
    %Equal.sym(U32,(0 * scale : U32),0,Arithmetic.multiply_zero_left(scale)) : {BandValues.cell_count(output,_) == 0n : Nat}
    RowMajor.multiply_zero_right(BandValues.channels(output)) : {BandValues.cell_count(output,(rows * scale : U32)) == 0n : Nat}}
  Shape.Both{Contract.empty_ready(~Element,fill),
    %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Tensors.buffer(~Element,fill,0)),fresh,Contract.empty_contents(~Element,fill)) :
      BandValues.band_values(~Element,~GatherValues.Directed<Logical>,~GatherValues.directed(~Element,~Logical,~original),GatherValues.Directed{logical,Views.View{view_axes,offset},reversed},
        output,(first * scale : U32),(rows * scale : U32),_)
    BandValues.empty_leaf(~Element,~GatherValues.Directed<Logical>,~GatherValues.directed(~Element,~Logical,~original),GatherValues.Directed{logical,Views.View{view_axes,offset},reversed},
      output,(first * scale : U32),(rows * scale : U32),Tree.fresh(Element,Tensors.depth_for(0),fill),0n,no_cells)}

def scope_of(~Element: Data,~Logical: Data,+logical: Logical,+name: String,+view_axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,+banded: BandViews.Banded,+geometry: Bands.Geometry,
  +local: List<&2,Bool>,+fill: Element,+limits: Machine.Limits,+words: Words(banded,geometry),+classified: Classified.Classified<view_axes,offset,geometry,banded>,
  +directions: {BandViews.local_flags(banded,geometry,reversed) == Some{local} : Maybe<&2,List<&2,Bool>>},
  +room: BandGather.Domain<BandViews.output(banded,geometry),limits,Machine.capacity(limits)>) -> Scope<Element,Logical>:
  Scope{logical,name,view_axes,offset,reversed,banded,geometry,local,fill,limits,words,classified,directions,room}

# A band of rows gathers through its local view; a band of none is empty.
def rows_output(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,empty: Bool,+logical: Logical,+name: String,+view_axes: List<&2,Views.Axis>,+offset: U32,
  +reversed: List<&2,Bool>,+banded: BandViews.Banded,+batch: U32,+channels: U32,+height: U32,+width: U32,+local: List<&2,Bool>,+fill: Element,+limits: Machine.Limits,
  +words: Words(banded,Bands.Geometry{batch,channels,height,width}),+classified: Classified.Classified<view_axes,offset,Bands.Geometry{batch,channels,height,width},banded>,
  +directions: {BandViews.local_flags(banded,Bands.Geometry{batch,channels,height,width},reversed) == Some{local} : Maybe<&2,List<&2,Bool>>},
  +room: BandGather.Domain<BandViews.output(banded,Bands.Geometry{batch,channels,height,width}),limits,Machine.capacity(limits)>,+first: U32,+rows: U32,+input: Buffer.Model<Element>,
  +input_fact: Input(~Element,~Logical,~original,scope_of(~Element,~Logical,logical,name,view_axes,offset,reversed,banded,Bands.Geometry{batch,channels,height,width},local,fill,limits,words,classified,directions,room),
    first,rows,Buffer.content(~Element,input)),
  +ready: Buffer.Sized<Element,input>,+compared: {U32.is_zero(rows) == empty : Bool},
  accepted: {Each.usable(~Element,Each.checked(~Element,Tensors.gathered_rows(~Element,empty,name,banded,batch,channels,height,width,local,fill,first,rows,Buffer.pack(~Element,input)))) == True{} : Bool}) ->
  Model.ContentAll(~Element,~Output(~Element,~Logical,~original,
      scope_of(~Element,~Logical,logical,name,view_axes,offset,reversed,banded,Bands.Geometry{batch,channels,height,width},local,fill,limits,words,classified,directions,room)),
    Model.contents(~Element,Each.made(~Element,Each.checked(~Element,Tensors.gathered_rows(~Element,empty,name,banded,batch,channels,height,width,local,fill,first,rows,Buffer.pack(~Element,input)))))):
  match empty:
    case True{}:
      empty_output(~Element,~Logical,~original,
        scope_of(~Element,~Logical,logical,name,view_axes,offset,reversed,banded,Bands.Geometry{batch,channels,height,width},local,fill,limits,words,classified,directions,room),
        first,rows,fill,BandViews.extents(banded,rows),compared)
    case False{}:
      +shape = BandViews.extents(banded,rows)
      viewed_output(~Element,~Logical,~original,Tensors.view_of(shape),Tensors.count_of(shape),
        scope_of(~Element,~Logical,logical,name,view_axes,offset,reversed,banded,Bands.Geometry{batch,channels,height,width},local,fill,limits,words,classified,directions,room),
        first,rows,input,input_fact,name,[channels,rows,width],fill,ready,{==},{==},accepted)

# One band through its leaf gather: a usable call leaves the output band.
def leaf_output(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,
  +scope: Scope<Element,Logical>,+first: U32,+rows: U32,+input: Buffer.Model<Element>,
  +input_fact: Input(~Element,~Logical,~original,scope,first,rows,Buffer.content(~Element,input)),
  accepted: {Each.usable(~Element,Each.checked(~Element,Bands.transform_band(~Element,~Tensors.Gathered<Element>,~Tensors.gathered_band(~Element),
    scope_gathered(~Element,~Logical,scope),first,rows,Buffer.pack(~Element,input),Unit{}))) == True{} : Bool}) ->
  Model.ContentAll(~Element,~Output(~Element,~Logical,~original,scope),Model.contents(~Element,Each.made(~Element,Each.checked(~Element,
    Bands.transform_band(~Element,~Tensors.Gathered<Element>,~Tensors.gathered_band(~Element),scope_gathered(~Element,~Logical,scope),first,rows,Buffer.pack(~Element,input),Unit{}))))):
  Scope{+logical,+name,+view_axes,+offset,+reversed,+banded,+geometry,+local,+fill,+limits,+words,+classified,+directions,+room} = scope
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  Buffer.Model{+tree,+length,+certificate} = input
  Shape.Both{Shape.Both{+fields,+ready},+values} = input_fact
  +input = {Buffer.Model{tree,length,certificate} : Buffer.Model<Element>}
  rows_output(~Element,~Logical,~original,U32.is_zero(rows),logical,name,view_axes,offset,reversed,banded,batch,channels,height,width,local,fill,limits,words,classified,directions,room,
    first,rows,input,
    Shape.Both{Shape.Both{fields,ready},values},Buffer.ready_content(~Element,input,ready),{==},accepted)

# ---- Every band ----

# A Normal result that holds at every index the input at the address the
# view, read with its flags, gives to that index.
def Gathered(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+view: Views.View,+reversed: List<&2,Bool>,-tensor: Tensors.Tensor<Element>) -> Data:
  Global.Contract(~Element,~GatherValues.Directed<Logical>,~GatherValues.directed(~Element,~Logical,~original),GatherValues.Directed{logical,view,reversed},tensor)

# A gather that reads every axis forward holds the values of its view.
def forward(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+view: Views.View,tensor: Tensors.Tensor<Element>,
  gathered: Gathered(~Element,~Logical,~original,logical,view,[],tensor)) ->
  Global.Contract(~Element,~GatherValues.Gather<Logical>,~GatherValues.value(~Element,~Logical,~original),GatherValues.Gather{logical,view},tensor):
  Global.relabeled(~Element,~GatherValues.Directed<Logical>,~GatherValues.Gather<Logical>,~GatherValues.directed(~Element,~Logical,~original),
    ~GatherValues.value(~Element,~Logical,~original),~GatherValues.Forward(~Logical),~GatherValues.forward(~Element,~Logical,~original),
    GatherValues.Directed{logical,view,[]},GatherValues.Gather{logical,view},{==},tensor,gathered)

def Selected(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: Scope<Element,Logical>,-tensor: Tensors.Tensor<Element>) -> Data:
  Scope{+logical,name,+view_axes,+offset,+reversed,banded,geometry,local,fill,limits,words,classified,directions,room} = scope
  Gathered(~Element,~Logical,~original,logical,Views.View{view_axes,offset},reversed,tensor)

# Every band through its leaf gather, on the plan of its own bands.
def scheduled_output(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+name: String,+view_axes: List<&2,Views.Axis>,+offset: U32,
  +reversed: List<&2,Bool>,+banded: BandViews.Banded,+geometry: Bands.Geometry,+local: List<&2,Bool>,+shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32,
  +limits: Machine.Limits,+words: Words(banded,geometry),+classified: Classified.Classified<view_axes,offset,geometry,banded>,
  +directions: {BandViews.local_flags(banded,geometry,reversed) == Some{local} : Maybe<&2,List<&2,Bool>>},
  +room: BandGather.Domain<BandViews.output(banded,geometry),limits,Machine.capacity(limits)>,
  +model: Model.Model<Element>,metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,model)),ready: Model.ContentReady(~Element,Model.content(~Element,model)),
  values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,model)),
  counted: Global.Counted<target>,coherent: Global.Coherent(target,BandViews.output(banded,geometry)),
  accepted: {Contract.tensor_ready(~Element,Tensors.gather_banded(~Element,name,banded,geometry,local,shape,target,fill,workers,Model.pack(~Element,model))) == True{} : Bool}) ->
  Gathered(~Element,~Logical,~original,logical,Views.View{view_axes,offset},reversed,Tensors.gather_banded(~Element,name,banded,geometry,local,shape,target,fill,workers,Model.pack(~Element,model))):
  +scope = {Scope{logical,name,view_axes,offset,reversed,banded,geometry,local,fill,limits,words,classified,directions,room} : Scope<Element,Logical>}
  +context = {Tensors.Gathered{name,banded,geometry,local,fill} : Tensors.Gathered<Element>}
  +output = BandViews.output(banded,geometry)
  +schedule = Each.scheduled(~Element,~Tensors.Gathered<Element>,~Tensors.gathered_band(~Element),context,geometry,workers,12.0,0.0,model)
  +first_fact = Model.content_both(~Element,~Scope<Element,Logical>,~Model.Both(~Element,~Scope<Element,Logical>,~input_fields(~Element,~Logical),~Model.always(~Element,~Scope<Element,Logical>,~Model.ready_band(~Element))),
    ~input_values(~Element,~Logical,~original),scope,Model.content(~Element,model),
    Model.content_both(~Element,~Scope<Element,Logical>,~input_fields(~Element,~Logical),~Model.always(~Element,~Scope<Element,Logical>,~Model.ready_band(~Element)),scope,Model.content(~Element,model),metadata,ready),values)
  %Equal.sym(Bands.Bands(~Element) & Bool,BandSchedule.transform(~Element,~Tensors.Gathered<Element>,~Tensors.gathered_band(~Element),context,geometry,workers,12.0,0.0,Model.pack(~Element,model)),
    Each.checked(~Element,Each.leaf_map(~Element,~Bands.Checked<Element>,~Tensors.Gathered<Element>,~Bands.transform_band(~Element,~Tensors.Gathered<Element>,~Tensors.gathered_band(~Element)),context,model)),schedule) :
    Gathered(~Element,~Logical,~original,logical,Views.View{view_axes,offset},reversed,Tensors.transformed(~Element,name,Tensors.InvalidArgument{shape},output,target,fill,workers,_))
  Global.transformed(~Element,~Scope<Element,Logical>,~OutputValues(~Element,~Logical,~original),~Selected(~Element,~Logical,~original),scope,
    name,Tensors.InvalidArgument{shape},output,target,fill,workers,limits,BandGather.rows_room(output,limits,Machine.capacity(limits),room),
    Each.checked(~Element,Each.leaf_map(~Element,~Bands.Checked<Element>,~Tensors.Gathered<Element>,~Bands.transform_band(~Element,~Tensors.Gathered<Element>,~Tensors.gathered_band(~Element)),context,model)),
    usable => Each.checked_all(~Element,~Tensors.Gathered<Element>,~Bands.transform_band(~Element,~Tensors.Gathered<Element>,~Tensors.gathered_band(~Element)),~Scope<Element,Logical>,~scope_gathered(~Element,~Logical),
      ~Input(~Element,~Logical,~original),~Output(~Element,~Logical,~original),~leaf_output(~Element,~Logical,~original),scope,model,first_fact,usable),
    counted,coherent,bands => normal => facts => Shape.Both{normal,facts},
    %schedule : {Contract.tensor_ready(~Element,Tensors.transformed(~Element,name,Tensors.InvalidArgument{shape},output,target,fill,workers,_)) == True{} : Bool} accepted)

# ---- The public gather ----

def count_elements(+target: List<&2,U32>,+count: U32,counted: Count.Count<ShapeLaws.elements(target),count>) -> {ShapeLaws.elements(target) == U32.to_nat(count) : Nat}:
  Count.Count{exact,bounded} = counted
  exact

# The output of a counted, nonempty target the machine holds has channels,
# rows and columns.
def output_domain(+target: List<&2,U32>,+count: U32,+counts: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},+nonzero: {U32.is_zero(count) == False{} : Bool},
  +batch: U32,+channels: U32,+height: U32,+width: U32,+coherent: Global.Coherent(target,Bands.Geometry{batch,channels,height,width})) ->
  BandGather.Domain<Bands.Geometry{batch,channels,height,width},Default.limits(),Machine.capacity(Default.limits())>:
  +exact = count_elements(target,count,Count.count_of(target,count,counts))
  +c = U32.to_nat(channels)
  +h = U32.to_nat(height)
  +w = U32.to_nat(width)
  +spatial = Nat.mul(h,w)
  +frame = Nat.mul(c,spatial)
  +total = BandGather.elements(Bands.Geometry{batch,channels,height,width})
  +positive = {%Equal.trans(Nat,U32.to_nat(count),ShapeLaws.elements(target),total,Equal.sym(Nat,ShapeLaws.elements(target),U32.to_nat(count),exact),coherent) :
    {Nat.is_lt(0n,_) == True{} : Bool} Arithmetic.word_nonzero_positive(count,nonzero) : {Nat.is_lt(0n,total) == True{} : Bool}}
  +frame_positive = DivisionOrder.right_factor_positive(U32.to_nat(batch),frame,positive)
  +spatial_positive = DivisionOrder.right_factor_positive(c,spatial,frame_positive)
  Global.domain(target,Bands.Geometry{batch,channels,height,width},Global.Counted{count,counts},coherent,
    RowMajor.left_factor_positive(c,spatial,frame_positive),RowMajor.left_factor_positive(h,w,spatial_positive),DivisionOrder.right_factor_positive(h,w,spatial_positive))

def fitted_batch(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,
  fitted: Fitted<before,repeats,after,start,batch,channels,height,width>) -> {Nat.is_lt(0n,U32.to_nat(batch)) == True{} : Bool}:
  Fitted{batch_positive,height_positive,channel_reach,column_reach,early,late,repeated,rows_exact} = fitted
  batch_positive

def fitted_height(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,
  fitted: Fitted<before,repeats,after,start,batch,channels,height,width>) -> {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool}:
  Fitted{batch_positive,height_positive,channel_reach,column_reach,early,late,repeated,rows_exact} = fitted
  height_positive

# A classification that fits a geometry of a counted shape leaves channels,
# rows and columns in it.
def fitted_domain(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,
  +shape: List<&2,U32>,fitted: Fitted<before,repeats,after,start,batch,channels,height,width>,counted: Global.Counted<shape>,
  coherent: Global.Coherent(shape,Bands.Geometry{batch,channels,height,width})) ->
  BandGather.Domain<Bands.Geometry{batch,channels,height,width},Default.limits(),Machine.capacity(Default.limits())>:
  Fitted{batch_positive,height_positive,channel_reach,column_reach,early,late,repeated,rows_exact} = fitted
  +plane = (height * width : U32)
  Global.domain(shape,Bands.Geometry{batch,channels,height,width},counted,coherent,
    Order.positive_bound(Address.reach_total(plane,False{},before,after,start),U32.to_nat(channels),channel_reach),height_positive,
    Order.positive_bound(Address.reach_total(plane,True{},before,after,start),U32.to_nat(width),column_reach))

# A classification that fits a geometry of a counted shape gives the words of
# every band.
def words(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,
  +shape: List<&2,U32>,+fitted: Fitted<before,repeats,after,start,batch,channels,height,width>,counted: Global.Counted<shape>,
  coherent: Global.Coherent(shape,Bands.Geometry{batch,channels,height,width})) ->
  Words(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width}):
  Shape.Both{fitted,plane_words(batch,channels,height,width,Default.limits(),fitted_batch(before,repeats,after,start,batch,channels,height,width,fitted),
    fitted_domain(before,repeats,after,start,batch,channels,height,width,shape,fitted,counted,coherent))}

# Bands whose images and rows a classified view keeps in place and reads
# forward gather their own cells.
def banded_gather(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+name: String,+view_axes: List<&2,Views.Axis>,+offset: U32,
  +reversed: List<&2,Bool>,+banded: BandViews.Banded,+geometry: Bands.Geometry,+local: List<&2,Bool>,+shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32,
  +model: Model.Model<Element>,+classified: Classified.Classified<view_axes,offset,geometry,banded>,
  +directions: {BandViews.local_flags(banded,geometry,reversed) == Some{local} : Maybe<&2,List<&2,Bool>>},
  well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,model)>,counted: Global.Counted<shape>,coherent: Global.Coherent(shape,geometry),
  values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,model)),
  +count: U32,+counts: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},+nonzero: {U32.is_zero(count) == False{} : Bool},
  +same_elements: {ShapeLaws.elements(target) == ShapeLaws.elements(Views.axis_extents(view_axes)) : Nat},
  accepted: {Contract.tensor_ready(~Element,Tensors.gather_banded(~Element,name,banded,geometry,local,shape,target,fill,workers,Model.pack(~Element,model))) == True{} : Bool}) ->
  Gathered(~Element,~Logical,~original,logical,Views.View{view_axes,offset},reversed,Tensors.gather_banded(~Element,name,banded,geometry,local,shape,target,fill,workers,Model.pack(~Element,model))):
  BandViews.Banded{+before,+repeats,+after,+start} = banded
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  Classified.Classified{+axes_exact,+offset_exact,+fits} = classified
  BandShape.WellFormed{metadata,ready,coverage} = well_formed
  +banded = {BandViews.Banded{before,repeats,after,start} : BandViews.Banded}
  +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
  +classified = {Classified.Classified{axes_exact,offset_exact,fits} : Classified.Classified<view_axes,offset,geometry,banded>}
  +fitted = fitted(before,repeats,after,start,batch,channels,height,width,fits)
  +whole = Classified.whole_axes(banded,geometry)
  +output_elements = BandGather.elements(BandViews.output(banded,geometry))
  +coherent_output = Equal.trans(Nat,ShapeLaws.elements(target),ShapeLaws.elements(Views.axis_extents(view_axes)),output_elements,same_elements,
    %Equal.sym(List<&2,Views.Axis>,view_axes,whole,axes_exact) : {ShapeLaws.elements(Views.axis_extents(_)) == output_elements : Nat}
    whole_elements(before,repeats,after,start,batch,channels,height,width,fitted))
  scheduled_output(~Element,~Logical,~original,logical,name,view_axes,offset,reversed,banded,geometry,local,shape,target,fill,workers,Default.limits(),
    words(before,repeats,after,start,batch,channels,height,width,shape,fitted,counted,coherent),classified,directions,
    output_domain(target,count,counts,nonzero,batch,BandViews.product_or_zero(Views.axis_extents(before)),(height * BandViews.product_or_zero(repeats) : U32),
      BandViews.product_or_zero(Views.axis_extents(after)),coherent_output),
    model,metadata,ready,values,Global.Counted{count,counts},coherent_output,accepted)

# Arrived bands are read through their model.
def arrived_gather(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+name: String,+view_axes: List<&2,Views.Axis>,+offset: U32,
  +reversed: List<&2,Bool>,+banded: BandViews.Banded,+whole: Bands.Geometry,+local: List<&2,Bool>,+shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32,
  -bands: Bands.Bands(~Element),
  observation: Model.Observation<Element,bands>,arrived: Arrive.Arrived(~Element,~Logical,~original,logical,whole,Bands.unsplit(whole),bands),
  +classified: Classified.Classified<view_axes,offset,whole,banded>,
  +directions: {BandViews.local_flags(banded,whole,reversed) == Some{local} : Maybe<&2,List<&2,Bool>>},
  counted: Global.Counted<shape>,coherent: Global.Coherent(shape,whole),
  +count: U32,+counts: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},+nonzero: {U32.is_zero(count) == False{} : Bool},
  +same_elements: {ShapeLaws.elements(target) == ShapeLaws.elements(Views.axis_extents(view_axes)) : Nat},
  accepted: {Contract.tensor_ready(~Element,Tensors.gather_banded(~Element,name,banded,whole,local,shape,target,fill,workers,bands)) == True{} : Bool}) ->
  Gathered(~Element,~Logical,~original,logical,Views.View{view_axes,offset},reversed,Tensors.gather_banded(~Element,name,banded,whole,local,shape,target,fill,workers,bands)):
  Model.Observation{+source,+equation} = observation
  Shape.Both{same,Shape.Both{well_formed,values}} = arrived
  +contents = Model.observed_contents(~Element,source,bands,equation)
  %equation : Gathered(~Element,~Logical,~original,logical,Views.View{view_axes,offset},reversed,Tensors.gather_banded(~Element,name,banded,whole,local,shape,target,fill,workers,_))
  banded_gather(~Element,~Logical,~original,logical,name,view_axes,offset,reversed,banded,whole,local,shape,target,fill,workers,source,classified,directions,
    %contents : BandShape.WellFormed<Element,whole,_> well_formed,counted,coherent,
    %contents : BandValues.Values(~Element,~Logical,~original,logical,whole,_) values,count,counts,nonzero,same_elements,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,source),bands,equation) :
      {Contract.tensor_ready(~Element,Tensors.gather_banded(~Element,name,banded,whole,local,shape,target,fill,workers,_)) == True{} : Bool}
      accepted)

# The bands of the whole geometry gather with the flags of their local view.
def whole_local_gather(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,found: Maybe<&2,List<&2,Bool>>,+logical: Logical,+name: String,
  +view_axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,+banded: BandViews.Banded,+model: Model.Model<Element>,+geometry: Bands.Geometry,
  +shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32,
  +classified: Classified.Classified<view_axes,offset,Bands.Geometry{1,1,1,Tensors.count_or_zero(shape)},banded>,
  +directions: {BandViews.local_flags(banded,Bands.Geometry{1,1,1,Tensors.count_or_zero(shape)},reversed) == found : Maybe<&2,List<&2,Bool>>},
  well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,model)>,+counted: Global.Counted<shape>,coherent: Global.Coherent(shape,geometry),
  values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,model)),
  +count: U32,+counts: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},+nonzero: {U32.is_zero(count) == False{} : Bool},
  +same_elements: {ShapeLaws.elements(target) == ShapeLaws.elements(Views.axis_extents(view_axes)) : Nat},
  accepted: {Contract.tensor_ready(~Element,Tensors.gather_whole_local(~Element,name,banded,found,Bands.Geometry{1,1,1,Tensors.count_or_zero(shape)},shape,target,fill,workers,
    Tensors.Tensor{Model.pack(~Element,model),geometry,shape,fill,Tensors.Usable{},workers})) == True{} : Bool}) ->
  Gathered(~Element,~Logical,~original,logical,Views.View{view_axes,offset},reversed,Tensors.gather_whole_local(~Element,name,banded,found,
    Bands.Geometry{1,1,1,Tensors.count_or_zero(shape)},shape,target,fill,workers,Tensors.Tensor{Model.pack(~Element,model),geometry,shape,fill,Tensors.Usable{},workers})):
  match found:
    case Some{+local}:
      +whole = {Bands.Geometry{1,1,1,Tensors.count_or_zero(shape)} : Bands.Geometry}
      +arrival = WholeLeaf.whole_arrival(shape,counted)
      arrived_gather(~Element,~Logical,~original,logical,name,view_axes,offset,reversed,banded,whole,local,shape,target,fill,workers,
        Tensors.arrive(~Element,whole,Bands.unsplit(whole),Tensors.Tensor{Model.pack(~Element,model),geometry,shape,fill,Tensors.Usable{},workers}),
        Model.observe(~Element,Tensors.arrive(~Element,whole,Bands.unsplit(whole),Tensors.Tensor{Model.pack(~Element,model),geometry,shape,fill,Tensors.Usable{},workers})),
        Arrive.arrived(~Element,~Logical,~original,logical,shape,whole,Bands.unsplit(whole),geometry,fill,Tensors.Usable{},workers,model,arrival,well_formed,coherent,values),
        classified,directions,counted,Target.arrival_coherent(shape,whole,Bands.unsplit(whole),arrival),count,counts,nonzero,same_elements,accepted)
    case None{}:
      Empty.absurd(Gathered(~Element,~Logical,~original,logical,Views.View{view_axes,offset},reversed,Tensors.gather_whole_local(~Element,name,banded,None{},
        Bands.Geometry{1,1,1,Tensors.count_or_zero(shape)},shape,target,fill,workers,Tensors.Tensor{Model.pack(~Element,model),geometry,shape,fill,Tensors.Usable{},workers})),
        WordEncoding.false_true(accepted))

# The view classified by the whole geometry.
def whole_gather(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,found: Maybe<&2,BandViews.Banded>,+logical: Logical,+name: String,
  +view_axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,+model: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,
  +fill: Element,+workers: U32,
  +classification: Classified.Classification(view_axes,offset,Bands.Geometry{1,1,1,Tensors.count_or_zero(shape)},found),
  well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,model)>,+counted: Global.Counted<shape>,coherent: Global.Coherent(shape,geometry),
  values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,model)),
  +count: U32,+counts: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},+nonzero: {U32.is_zero(count) == False{} : Bool},
  +same_elements: {ShapeLaws.elements(target) == ShapeLaws.elements(Views.axis_extents(view_axes)) : Nat},
  accepted: {Contract.tensor_ready(~Element,Tensors.gather_whole(~Element,name,found,reversed,Bands.Geometry{1,1,1,Tensors.count_or_zero(shape)},shape,target,fill,workers,
    Tensors.Tensor{Model.pack(~Element,model),geometry,shape,fill,Tensors.Usable{},workers})) == True{} : Bool}) ->
  Gathered(~Element,~Logical,~original,logical,Views.View{view_axes,offset},reversed,Tensors.gather_whole(~Element,name,found,reversed,
    Bands.Geometry{1,1,1,Tensors.count_or_zero(shape)},shape,target,fill,workers,Tensors.Tensor{Model.pack(~Element,model),geometry,shape,fill,Tensors.Usable{},workers})):
  match found:
    case Some{+banded}:
      +whole = {Bands.Geometry{1,1,1,Tensors.count_or_zero(shape)} : Bands.Geometry}
      whole_local_gather(~Element,~Logical,~original,BandViews.local_flags(banded,whole,reversed),logical,name,view_axes,offset,reversed,banded,model,geometry,shape,target,
        fill,workers,classification,{==},well_formed,counted,coherent,values,count,counts,nonzero,same_elements,accepted)
    case None{}:
      Empty.absurd(Gathered(~Element,~Logical,~original,logical,Views.View{view_axes,offset},reversed,Tensors.gather_whole(~Element,name,None{},reversed,
        Bands.Geometry{1,1,1,Tensors.count_or_zero(shape)},shape,target,fill,workers,Tensors.Tensor{Model.pack(~Element,model),geometry,shape,fill,Tensors.Usable{},workers})),
        WordEncoding.false_true(accepted))

# Other bands first arrive as the one band of the whole geometry, whose view
# is classified again.
def arrival_gather(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+name: String,
  +view_axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,+model: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,
  +fill: Element,+workers: U32,
  well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,model)>,+counted: Global.Counted<shape>,coherent: Global.Coherent(shape,geometry),
  values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,model)),
  +count: U32,+counts: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},+nonzero: {U32.is_zero(count) == False{} : Bool},
  +same_elements: {ShapeLaws.elements(target) == ShapeLaws.elements(Views.axis_extents(view_axes)) : Nat},
  accepted: {Contract.tensor_ready(~Element,Tensors.gather_arrived(~Element,name,reversed,view_axes,offset,Model.pack(~Element,model),geometry,shape,target,fill,workers)) == True{} : Bool}) ->
  Gathered(~Element,~Logical,~original,logical,Views.View{view_axes,offset},reversed,
    Tensors.gather_arrived(~Element,name,reversed,view_axes,offset,Model.pack(~Element,model),geometry,shape,target,fill,workers)):
  +whole = {Bands.Geometry{1,1,1,Tensors.count_or_zero(shape)} : Bands.Geometry}
  whole_gather(~Element,~Logical,~original,BandViews.classify(view_axes,offset,whole),logical,name,view_axes,offset,reversed,model,geometry,shape,target,fill,workers,
    Classified.classify(view_axes,offset,whole),well_formed,counted,coherent,values,count,counts,nonzero,same_elements,accepted)

# A classified view gathers band by band when its local view has flags.
def local_gather(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,found: Maybe<&2,List<&2,Bool>>,+logical: Logical,+name: String,
  +view_axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,+banded: BandViews.Banded,+model: Model.Model<Element>,+geometry: Bands.Geometry,
  +shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32,
  +classified: Classified.Classified<view_axes,offset,geometry,banded>,
  +directions: {BandViews.local_flags(banded,geometry,reversed) == found : Maybe<&2,List<&2,Bool>>},
  +well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,model)>,+counted: Global.Counted<shape>,+coherent: Global.Coherent(shape,geometry),
  +values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,model)),
  +count: U32,+counts: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},+nonzero: {U32.is_zero(count) == False{} : Bool},
  +same_elements: {ShapeLaws.elements(target) == ShapeLaws.elements(Views.axis_extents(view_axes)) : Nat},
  accepted: {Contract.tensor_ready(~Element,Tensors.gather_local(~Element,name,banded,found,reversed,view_axes,offset,Model.pack(~Element,model),geometry,shape,target,fill,workers)) == True{} : Bool}) ->
  Gathered(~Element,~Logical,~original,logical,Views.View{view_axes,offset},reversed,
    Tensors.gather_local(~Element,name,banded,found,reversed,view_axes,offset,Model.pack(~Element,model),geometry,shape,target,fill,workers)):
  match found:
    case Some{+local}:
      banded_gather(~Element,~Logical,~original,logical,name,view_axes,offset,reversed,banded,geometry,local,shape,target,fill,workers,model,classified,directions,
        well_formed,counted,coherent,values,count,counts,nonzero,same_elements,accepted)
    case None{}:
      arrival_gather(~Element,~Logical,~original,logical,name,view_axes,offset,reversed,model,geometry,shape,target,fill,workers,
        well_formed,counted,coherent,values,count,counts,nonzero,same_elements,accepted)

# The view is classified by the geometry of the input first.
def found_gather(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,found: Maybe<&2,BandViews.Banded>,+logical: Logical,+name: String,
  +view_axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,+model: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,
  +fill: Element,+workers: U32,
  +classification: Classified.Classification(view_axes,offset,geometry,found),
  +well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,model)>,+counted: Global.Counted<shape>,+coherent: Global.Coherent(shape,geometry),
  +values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,model)),
  +count: U32,+counts: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},+nonzero: {U32.is_zero(count) == False{} : Bool},
  +same_elements: {ShapeLaws.elements(target) == ShapeLaws.elements(Views.axis_extents(view_axes)) : Nat},
  accepted: {Contract.tensor_ready(~Element,Tensors.gather_found(~Element,name,found,reversed,view_axes,offset,Model.pack(~Element,model),geometry,shape,target,fill,workers)) == True{} : Bool}) ->
  Gathered(~Element,~Logical,~original,logical,Views.View{view_axes,offset},reversed,
    Tensors.gather_found(~Element,name,found,reversed,view_axes,offset,Model.pack(~Element,model),geometry,shape,target,fill,workers)):
  match found:
    case Some{+banded}:
      local_gather(~Element,~Logical,~original,BandViews.local_flags(banded,geometry,reversed),logical,name,view_axes,offset,reversed,banded,model,geometry,shape,target,
        fill,workers,classification,{==},well_formed,counted,coherent,values,count,counts,nonzero,same_elements,accepted)
    case None{}:
      arrival_gather(~Element,~Logical,~original,logical,name,view_axes,offset,reversed,model,geometry,shape,target,fill,workers,
        well_formed,counted,coherent,values,count,counts,nonzero,same_elements,accepted)

# A target with no elements gives an empty whole result.
def empty_gather(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+view: Views.View,+reversed: List<&2,Bool>,+target: List<&2,U32>,
  +fill: Element,+workers: U32,+count: U32,+counts: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},+empty: {U32.is_zero(count) == True{} : Bool}) ->
  Gathered(~Element,~Logical,~original,logical,view,reversed,Tensors.whole(~Element,Tensors.buffer(~Element,fill,0),target,fill,Tensors.Usable{},workers)):
  +exact = count_elements(target,count,Count.count_of(target,count,counts))
  +zero = {%Arithmetic.equal_of_is_eq(count,0,empty) : {ShapeLaws.elements(target) == U32.to_nat(_) : Nat} exact : {ShapeLaws.elements(target) == 0n : Nat}}
  +length_exact = {%Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Tensors.buffer(~Element,fill,0)),Buffer.Content{Tree.fresh(Element,Tensors.depth_for(0),fill),0n},Contract.empty_contents(~Element,fill)) :
    {Buffer.content_length(~Element,_) == ShapeLaws.elements(target) : Nat}
    Equal.sym(Nat,ShapeLaws.elements(target),0n,zero) : {Buffer.content_length(~Element,Buffer.contents(~Element,Tensors.buffer(~Element,fill,0))) == ShapeLaws.elements(target) : Nat}}
  +values = {%Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Tensors.buffer(~Element,fill,0)),Buffer.Content{Tree.fresh(Element,Tensors.depth_for(0),fill),0n},Contract.empty_contents(~Element,fill)) :
    Contract.Values(~Element,~GatherValues.Directed<Logical>,~GatherValues.directed(~Element,~Logical,~original),GatherValues.Directed{logical,view,reversed},_)
    Unit{} : Contract.Values(~Element,~GatherValues.Directed<Logical>,~GatherValues.directed(~Element,~Logical,~original),GatherValues.Directed{logical,view,reversed},
      Buffer.contents(~Element,Tensors.buffer(~Element,fill,0)))}
  WholeLeaf.stored(~Element,~GatherValues.Directed<Logical>,~GatherValues.directed(~Element,~Logical,~original),GatherValues.Directed{logical,view,reversed},
    Tensors.buffer(~Element,fill,0),target,fill,workers,
    Buffer.observe(~Element,Tensors.buffer(~Element,fill,0)),Shape.Both{Contract.WellFormed{length_exact,Contract.empty_ready(~Element,fill)},values})

def counted_gather(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,empty: Bool,+logical: Logical,+name: String,
  +view_axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,+model: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,
  +fill: Element,+workers: U32,
  +well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,model)>,+counted: Global.Counted<shape>,+coherent: Global.Coherent(shape,geometry),
  +values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,model)),
  +count: U32,+counts: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},+compared: {U32.is_zero(count) == empty : Bool},
  +same_elements: {ShapeLaws.elements(target) == ShapeLaws.elements(Views.axis_extents(view_axes)) : Nat},
  accepted: {Contract.tensor_ready(~Element,Tensors.gather_counted(~Element,name,empty,view_axes,offset,reversed,Model.pack(~Element,model),geometry,shape,target,fill,workers)) == True{} : Bool}) ->
  Gathered(~Element,~Logical,~original,logical,Views.View{view_axes,offset},reversed,
    Tensors.gather_counted(~Element,name,empty,view_axes,offset,reversed,Model.pack(~Element,model),geometry,shape,target,fill,workers)):
  match empty:
    case True{}: empty_gather(~Element,~Logical,~original,logical,Views.View{view_axes,offset},reversed,target,fill,workers,count,counts,compared)
    case False{}:
      found_gather(~Element,~Logical,~original,BandViews.classify(view_axes,offset,geometry),logical,name,view_axes,offset,reversed,model,geometry,shape,target,fill,workers,
        Classified.classify(view_axes,offset,geometry),well_formed,counted,coherent,values,count,counts,compared,same_elements,accepted)

def axes_gather(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,count: Maybe<&2,U32>,+logical: Logical,+name: String,
  +view_axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,+model: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,
  +fill: Element,+workers: U32,
  +well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,model)>,+counted: Global.Counted<shape>,+coherent: Global.Coherent(shape,geometry),
  +values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,model)),
  +counts: {Tensors.count_of(target) == count : Maybe<&2,U32>},
  +same_elements: {ShapeLaws.elements(target) == ShapeLaws.elements(Views.axis_extents(view_axes)) : Nat},
  accepted: {Contract.tensor_ready(~Element,Tensors.gather_axes(~Element,name,count,view_axes,offset,reversed,Model.pack(~Element,model),geometry,shape,target,fill,workers)) == True{} : Bool}) ->
  Gathered(~Element,~Logical,~original,logical,Views.View{view_axes,offset},reversed,
    Tensors.gather_axes(~Element,name,count,view_axes,offset,reversed,Model.pack(~Element,model),geometry,shape,target,fill,workers)):
  match count:
    case Some{+count}:
      counted_gather(~Element,~Logical,~original,U32.is_zero(count),logical,name,view_axes,offset,reversed,model,geometry,shape,target,fill,workers,
        well_formed,counted,coherent,values,count,counts,{==},same_elements,accepted)
    case None{}:
      Empty.absurd(Gathered(~Element,~Logical,~original,logical,Views.View{view_axes,offset},reversed,
        Tensors.gather_axes(~Element,name,None{},view_axes,offset,reversed,Model.pack(~Element,model),geometry,shape,target,fill,workers)),
        WordEncoding.false_true(accepted))

def model_gather(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+name: String,+view_axes: List<&2,Views.Axis>,+offset: U32,
  +reversed: List<&2,Bool>,+model: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,model),geometry,shape,fill,Tensors.Usable{},workers}),
  +same_elements: {ShapeLaws.elements(target) == ShapeLaws.elements(Views.axis_extents(view_axes)) : Nat},
  accepted: {Contract.tensor_ready(~Element,Tensors.gather(~Element,name,Tensors.Usable{},Some{Views.View{view_axes,offset}},reversed,Model.pack(~Element,model),geometry,shape,target,
    fill,workers)) == True{} : Bool}) ->
  Gathered(~Element,~Logical,~original,logical,Views.View{view_axes,offset},reversed,
    Tensors.gather(~Element,name,Tensors.Usable{},Some{Views.View{view_axes,offset}},reversed,Model.pack(~Element,model),geometry,shape,target,fill,workers)):
  Shape.Both{Shape.Both{well_formed,Shape.Both{+counted,+coherent}},values} = contract
  +contents = Model.packed_contents(~Element,model)
  axes_gather(~Element,~Logical,~original,Tensors.count_of(target),logical,name,view_axes,offset,reversed,model,geometry,shape,target,fill,workers,
    %contents : BandShape.WellFormed<Element,geometry,_> well_formed,counted,coherent,
    %contents : BandValues.Values(~Element,~Logical,~original,logical,geometry,_) values,{==},same_elements,accepted)

def observed_gather(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+name: String,+view_axes: List<&2,Views.Axis>,+offset: U32,
  +reversed: List<&2,Bool>,-bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32,
  observation: Model.Observation<Element,bands>,
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}),
  +same_elements: {ShapeLaws.elements(target) == ShapeLaws.elements(Views.axis_extents(view_axes)) : Nat},
  accepted: {Contract.tensor_ready(~Element,Tensors.gather(~Element,name,Tensors.Usable{},Some{Views.View{view_axes,offset}},reversed,bands,geometry,shape,target,fill,workers)) == True{} : Bool}) ->
  Gathered(~Element,~Logical,~original,logical,Views.View{view_axes,offset},reversed,
    Tensors.gather(~Element,name,Tensors.Usable{},Some{Views.View{view_axes,offset}},reversed,bands,geometry,shape,target,fill,workers)):
  Model.Observation{+model,+equation} = observation
  %equation : Gathered(~Element,~Logical,~original,logical,Views.View{view_axes,offset},reversed,
    Tensors.gather(~Element,name,Tensors.Usable{},Some{Views.View{view_axes,offset}},reversed,_,geometry,shape,target,fill,workers))
  model_gather(~Element,~Logical,~original,logical,name,view_axes,offset,reversed,model,geometry,shape,target,fill,workers,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,model),bands,equation) :
      Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{_,geometry,shape,fill,Tensors.Usable{},workers}) contract,
    same_elements,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,model),bands,equation) :
      {Contract.tensor_ready(~Element,Tensors.gather(~Element,name,Tensors.Usable{},Some{Views.View{view_axes,offset}},reversed,_,geometry,shape,target,fill,workers)) == True{} : Bool}
      accepted)

# The public gather of a Ready tensor with Normal state and complete values,
# whatever its layout, through a view whose elements are the target's, with
# the flagged axes read in reverse. When the call returns Ready, the result is
# Normal and holds at every index the input at the address the view, read with
# its flags, gives to that index.
def gather(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+name: String,+view_axes: List<&2,Views.Axis>,+offset: U32,
  +reversed: List<&2,Bool>,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}),
  +same_elements: {ShapeLaws.elements(target) == ShapeLaws.elements(Views.axis_extents(view_axes)) : Nat},
  accepted: {Contract.tensor_ready(~Element,Tensors.gather(~Element,name,Tensors.Usable{},Some{Views.View{view_axes,offset}},reversed,bands,geometry,shape,target,fill,workers)) == True{} : Bool}) ->
  Gathered(~Element,~Logical,~original,logical,Views.View{view_axes,offset},reversed,
    Tensors.gather(~Element,name,Tensors.Usable{},Some{Views.View{view_axes,offset}},reversed,bands,geometry,shape,target,fill,workers)):
  observed_gather(~Element,~Logical,~original,logical,name,view_axes,offset,reversed,bands,geometry,shape,target,fill,workers,Model.observe(~Element,bands),contract,same_elements,accepted)

# The public gather with no flag holds the values of its view.
def forward_gather(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+name: String,+view_axes: List<&2,Views.Axis>,+offset: U32,
  -bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32,observation: Model.Observation<Element,bands>,
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}),
  +same_elements: {ShapeLaws.elements(target) == ShapeLaws.elements(Views.axis_extents(view_axes)) : Nat},
  accepted: {Contract.tensor_ready(~Element,Tensors.gather(~Element,name,Tensors.Usable{},Some{Views.View{view_axes,offset}},[],bands,geometry,shape,target,fill,workers)) == True{} : Bool}) ->
  Global.Contract(~Element,~GatherValues.Gather<Logical>,~GatherValues.value(~Element,~Logical,~original),GatherValues.Gather{logical,Views.View{view_axes,offset}},
    Tensors.gather(~Element,name,Tensors.Usable{},Some{Views.View{view_axes,offset}},[],bands,geometry,shape,target,fill,workers)):
  Model.Observation{+model,+equation} = observation
  %equation : Global.Contract(~Element,~GatherValues.Gather<Logical>,~GatherValues.value(~Element,~Logical,~original),GatherValues.Gather{logical,Views.View{view_axes,offset}},
    Tensors.gather(~Element,name,Tensors.Usable{},Some{Views.View{view_axes,offset}},[],_,geometry,shape,target,fill,workers))
  forward(~Element,~Logical,~original,logical,Views.View{view_axes,offset},
    Tensors.gather(~Element,name,Tensors.Usable{},Some{Views.View{view_axes,offset}},[],Model.pack(~Element,model),geometry,shape,target,fill,workers),
    gather(~Element,~Logical,~original,logical,name,view_axes,offset,[],Model.pack(~Element,model),geometry,shape,target,fill,workers,
      %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,model),bands,equation) :
        Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{_,geometry,shape,fill,Tensors.Usable{},workers}) contract,
      same_elements,
      %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,model),bands,equation) :
        {Contract.tensor_ready(~Element,Tensors.gather(~Element,name,Tensors.Usable{},Some{Views.View{view_axes,offset}},[],_,geometry,shape,target,fill,workers)) == True{} : Bool}
        accepted))
