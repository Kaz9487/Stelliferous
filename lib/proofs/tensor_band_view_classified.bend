# What the classification of a view by a geometry shows: the axes of the view
# are the images axis when the geometry has many images, the axes before the
# rows, the rows axis when it has many rows, the repeats of each row and the
# axes after the rows; the classification keeps the offset of the view and
# fits the geometry.
import Base
import ../tensor_view.bend as Views
import ../tensor_band.bend as Bands
import ../tensor_band_view.bend as BandViews
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./boolean_logic.bend as Boolean

def images_axes(many: Bool,+batch: U32,+image: U32) -> List<&2,Views.Axis>:
  match many:
    case True{}: [Views.Axis{batch,image}]
    case False{}: []

# The axes a banded form describes over a geometry.
def whole_axes(+banded: BandViews.Banded,+geometry: Bands.Geometry) -> List<&2,Views.Axis>:
  BandViews.Banded{before,repeats,after,offset} = banded
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  List.append(&2,Views.Axis,images_axes(U32.is_lt(1,batch),batch,(channels * (height * width : U32) : U32)),
    List.append(&2,Views.Axis,before,List.append(&2,Views.Axis,BandViews.rows_axes(U32.is_lt(1,height),height,width,repeats),after)))

def offset_of(banded: BandViews.Banded) -> U32:
  BandViews.Banded{before,repeats,after,offset} = banded
  offset

type Classified<-view_axes: List<&2,Views.Axis>,-offset: U32,-geometry: Bands.Geometry,-banded: BandViews.Banded> is Data:
  Classified{axes_exact: {view_axes == whole_axes(banded,geometry) : List<&2,Views.Axis>},offset_exact: {offset_of(banded) == offset : U32},
    fits: {BandViews.fits(banded,geometry) == True{} : Bool}}

def Classification(view_axes: List<&2,Views.Axis>,offset: U32,geometry: Bands.Geometry,result: Maybe<&2,BandViews.Banded>) -> Data:
  match result:
    case Some{banded}: Classified<view_axes,offset,geometry,banded>
    case None{}: Unit

# ---- Each step of the classification ----

def kept_when(valid: Bool,+banded: BandViews.Banded,+geometry: Bands.Geometry,+view_axes: List<&2,Views.Axis>,+offset: U32,
  +computed: {BandViews.fits(banded,geometry) == valid : Bool},
  axes_exact: {view_axes == whole_axes(banded,geometry) : List<&2,Views.Axis>},offset_exact: {offset_of(banded) == offset : U32}) ->
  Classification(view_axes,offset,geometry,BandViews.kept_when(valid,banded)):
  match valid:
    case True{}: Classified{axes_exact,offset_exact,computed}
    case False{}: Unit{}

def kept(+banded: BandViews.Banded,+geometry: Bands.Geometry,+view_axes: List<&2,Views.Axis>,+offset: U32,
  axes_exact: {view_axes == whole_axes(banded,geometry) : List<&2,Views.Axis>},offset_exact: {offset_of(banded) == offset : U32}) ->
  Classification(view_axes,offset,geometry,BandViews.kept(banded,geometry)):
  kept_when(BandViews.fits(banded,geometry),banded,geometry,view_axes,offset,{==},axes_exact,offset_exact)

# The leading axes of stride zero, as repeats, then the rest.
def Repeats(rest: List<&2,Views.Axis>,split: List<&2,U32> & List<&2,Views.Axis>) -> Data:
  (repeats,after) = split
  {rest == List.append(&2,Views.Axis,BandViews.repeated(repeats),after) : List<&2,Views.Axis>}

def prepended(+extent: U32,+rest: List<&2,Views.Axis>,split: List<&2,U32> & List<&2,Views.Axis>,found: Repeats(rest,split)) ->
  Repeats(Views.Axis{extent,0} <> rest,BandViews.prepend_repeat(extent,split)):
  (+repeats,+after) = split
  Equal.cong(List<&2,Views.Axis>,List<&2,Views.Axis>,others => Views.Axis{extent,0} <> others,rest,List.append(&2,Views.Axis,BandViews.repeated(repeats),after),found)

def repeats_from(rest: List<&2,Views.Axis>,zero: Bool,+extent: U32,+stride: U32,+computed: {U32.is_zero(stride) == zero : Bool}) ->
  Repeats(Views.Axis{extent,stride} <> rest,BandViews.repeats_from(rest,zero,extent,stride)):
  match rest zero:
    case rest False{}: {==}
    case Nil{} True{}:
      Equal.cong(U32,List<&2,Views.Axis>,word => [Views.Axis{extent,word}],stride,0,Arithmetic.word_zero(stride,computed))
    case Views.Axis{+next_extent,+next_stride} <> +tail True{}:
      %Equal.sym(U32,stride,0,Arithmetic.word_zero(stride,computed)) :
        Repeats(Views.Axis{extent,_} <> Views.Axis{next_extent,next_stride} <> tail,BandViews.prepend_repeat(extent,BandViews.repeats_from(tail,U32.is_zero(next_stride),next_extent,next_stride)))
      prepended(extent,Views.Axis{next_extent,next_stride} <> tail,BandViews.repeats_from(tail,U32.is_zero(next_stride),next_extent,next_stride),
        repeats_from(tail,U32.is_zero(next_stride),next_extent,next_stride,{==}))

def split_repeats(+view_axes: List<&2,Views.Axis>) -> Repeats(view_axes,BandViews.split_repeats(view_axes)):
  match view_axes:
    case []: {==}
    case Views.Axis{+extent,+stride} <> +rest: repeats_from(rest,U32.is_zero(stride),extent,stride,{==})

# The axes before the rows axis, the rows axis, then the rest.
def Splits(+height: U32,+width: U32,view_axes: List<&2,Views.Axis>,split: Maybe<&2,BandViews.Split>) -> Data:
  match split:
    case Some{BandViews.Split{before,rest}}: {view_axes == List.append(&2,Views.Axis,before,Views.Axis{height,width} <> rest) : List<&2,Views.Axis>}
    case None{}: Unit

def prepend_before(+axis: Views.Axis,+height: U32,+width: U32,+view_axes: List<&2,Views.Axis>,split: Maybe<&2,BandViews.Split>,found: Splits(height,width,view_axes,split)) ->
  Splits(height,width,axis <> view_axes,BandViews.prepend_before(axis,split)):
  match split:
    case Some{BandViews.Split{+before,+rest}}:
      Equal.cong(List<&2,Views.Axis>,List<&2,Views.Axis>,others => axis <> others,view_axes,List.append(&2,Views.Axis,before,Views.Axis{height,width} <> rest),found)
    case None{}: Unit{}

# An axis equal, field by field, to the one it is compared with.
def same_axis(+extent: U32,+stride: U32,+expected_extent: U32,+expected_stride: U32,+rest: List<&2,Views.Axis>,
  +same: {U32.is_eq(extent,expected_extent) && U32.is_eq(stride,expected_stride) == True{} : Bool}) ->
  {Views.Axis{extent,stride} <> rest == Views.Axis{expected_extent,expected_stride} <> rest : List<&2,Views.Axis>}:
  +extents = Arithmetic.equal_of_is_eq(extent,expected_extent,Boolean.left(U32.is_eq(extent,expected_extent),U32.is_eq(stride,expected_stride),same))
  +strides = Arithmetic.equal_of_is_eq(stride,expected_stride,Boolean.right(U32.is_eq(extent,expected_extent),U32.is_eq(stride,expected_stride),same))
  %Equal.sym(U32,extent,expected_extent,extents) : {Views.Axis{_,stride} <> rest == Views.Axis{expected_extent,expected_stride} <> rest : List<&2,Views.Axis>}
  %Equal.sym(U32,stride,expected_stride,strides) : {Views.Axis{expected_extent,_} <> rest == Views.Axis{expected_extent,expected_stride} <> rest : List<&2,Views.Axis>}
  {==}

def split_from(rest: List<&2,Views.Axis>,found: Bool,+height: U32,+width: U32,+extent: U32,+stride: U32,
  +computed: {U32.is_eq(extent,height) && U32.is_eq(stride,width) == found : Bool}) ->
  Splits(height,width,Views.Axis{extent,stride} <> rest,BandViews.split_from(rest,found,height,width,extent,stride)):
  match rest found:
    case rest True{}: same_axis(extent,stride,height,width,rest,computed)
    case Nil{} False{}: Unit{}
    case Views.Axis{+next_extent,+next_stride} <> +tail False{}:
      prepend_before(Views.Axis{extent,stride},height,width,Views.Axis{next_extent,next_stride} <> tail,
        BandViews.split_from(tail,U32.is_eq(next_extent,height) && U32.is_eq(next_stride,width),height,width,next_extent,next_stride),
        split_from(tail,U32.is_eq(next_extent,height) && U32.is_eq(next_stride,width),height,width,next_extent,next_stride,{==}))

def split_rows(+height: U32,+width: U32,+view_axes: List<&2,Views.Axis>) -> Splits(height,width,view_axes,BandViews.split_rows(height,width,view_axes)):
  match view_axes:
    case []: Unit{}
    case Views.Axis{+extent,+stride} <> +rest: split_from(rest,U32.is_eq(extent,height) && U32.is_eq(stride,width),height,width,extent,stride,{==})

# The axes after the images axis of a batch of many images.
def Images(many: Bool,+batch: U32,+image: U32,view_axes: List<&2,Views.Axis>,result: Maybe<&2,List<&2,Views.Axis>>) -> Data:
  match result:
    case Some{rest}: {view_axes == List.append(&2,Views.Axis,images_axes(many,batch,image),rest) : List<&2,Views.Axis>}
    case None{}: Unit

def images_kept(kept: Bool,+batch: U32,+image: U32,+extent: U32,+stride: U32,+rest: List<&2,Views.Axis>,
  +computed: {U32.is_eq(extent,batch) && U32.is_eq(stride,image) == kept : Bool}) ->
  Images(True{},batch,image,Views.Axis{extent,stride} <> rest,BandViews.images_kept(kept,rest)):
  match kept:
    case True{}: same_axis(extent,stride,batch,image,rest,computed)
    case False{}: Unit{}

def after_images(many: Bool,+batch: U32,+image: U32,+view_axes: List<&2,Views.Axis>) -> Images(many,batch,image,view_axes,BandViews.after_images(many,batch,image,view_axes)):
  match many view_axes:
    case False{} view_axes: {==}
    case True{} Nil{}: Unit{}
    case True{} Views.Axis{+extent,+stride} <> +rest: images_kept(U32.is_eq(extent,batch) && U32.is_eq(stride,image),batch,image,extent,stride,rest,{==})

# ---- The classification ----

def split_kept(split: List<&2,U32> & List<&2,Views.Axis>,+offset: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,+view_axes: List<&2,Views.Axis>,
  +before: List<&2,Views.Axis>,+tail: List<&2,Views.Axis>,+rest: List<&2,Views.Axis>,+rowed: {U32.is_lt(1,height) == True{} : Bool},
  images: {view_axes == List.append(&2,Views.Axis,images_axes(U32.is_lt(1,batch),batch,(channels * (height * width : U32) : U32)),rest) : List<&2,Views.Axis>},
  found: {rest == List.append(&2,Views.Axis,before,Views.Axis{height,width} <> tail) : List<&2,Views.Axis>},repeats: Repeats(tail,split)) ->
  Classification(view_axes,offset,Bands.Geometry{batch,channels,height,width},BandViews.split_kept(offset,Bands.Geometry{batch,channels,height,width},before,split)):
  (+repeated,+after) = split
  +image = (channels * (height * width : U32) : U32)
  +leading = images_axes(U32.is_lt(1,batch),batch,image)
  +rows = List.append(&2,Views.Axis,BandViews.repeated(repeated),after)
  +described = Equal.trans(List<&2,Views.Axis>,view_axes,List.append(&2,Views.Axis,leading,rest),
    List.append(&2,Views.Axis,leading,List.append(&2,Views.Axis,before,Views.Axis{height,width} <> rows)),images,
    Equal.cong(List<&2,Views.Axis>,List<&2,Views.Axis>,others => List.append(&2,Views.Axis,leading,others),rest,List.append(&2,Views.Axis,before,Views.Axis{height,width} <> rows),
      Equal.trans(List<&2,Views.Axis>,rest,List.append(&2,Views.Axis,before,Views.Axis{height,width} <> tail),List.append(&2,Views.Axis,before,Views.Axis{height,width} <> rows),found,
        Equal.cong(List<&2,Views.Axis>,List<&2,Views.Axis>,others => List.append(&2,Views.Axis,before,Views.Axis{height,width} <> others),tail,rows,repeats))))
  kept(BandViews.Banded{before,repeated,after,offset},Bands.Geometry{batch,channels,height,width},view_axes,offset,
    %Equal.sym(Bool,U32.is_lt(1,height),True{},rowed) :
      {view_axes == List.append(&2,Views.Axis,leading,List.append(&2,Views.Axis,before,List.append(&2,Views.Axis,BandViews.rows_axes(_,height,width,repeated),after))) : List<&2,Views.Axis>}
    described,{==})

def rowed_split(split: Maybe<&2,BandViews.Split>,+offset: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,+view_axes: List<&2,Views.Axis>,+rest: List<&2,Views.Axis>,
  +rowed: {U32.is_lt(1,height) == True{} : Bool},
  images: {view_axes == List.append(&2,Views.Axis,images_axes(U32.is_lt(1,batch),batch,(channels * (height * width : U32) : U32)),rest) : List<&2,Views.Axis>},
  found: Splits(height,width,rest,split)) ->
  Classification(view_axes,offset,Bands.Geometry{batch,channels,height,width},BandViews.rowed_split(offset,Bands.Geometry{batch,channels,height,width},split)):
  match split:
    case Some{BandViews.Split{+before,+tail}}:
      split_kept(BandViews.split_repeats(tail),offset,batch,channels,height,width,view_axes,before,tail,rest,rowed,images,found,split_repeats(tail))
    case None{}: Unit{}

def classify_rows(rowed: Bool,result: Maybe<&2,List<&2,Views.Axis>>,+offset: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,+view_axes: List<&2,Views.Axis>,
  +computed: {U32.is_lt(1,height) == rowed : Bool},images: Images(U32.is_lt(1,batch),batch,(channels * (height * width : U32) : U32),view_axes,result)) ->
  Classification(view_axes,offset,Bands.Geometry{batch,channels,height,width},BandViews.classify_rows(rowed,result,offset,Bands.Geometry{batch,channels,height,width})):
  match rowed result:
    case True{} Some{+rest}: rowed_split(BandViews.split_rows(height,width,rest),offset,batch,channels,height,width,view_axes,rest,computed,images,split_rows(height,width,rest))
    case False{} Some{+rest}:
      kept(BandViews.Banded{[],[],rest,offset},Bands.Geometry{batch,channels,height,width},view_axes,offset,
        %Equal.sym(Bool,U32.is_lt(1,height),False{},computed) :
          {view_axes == List.append(&2,Views.Axis,images_axes(U32.is_lt(1,batch),batch,(channels * (height * width : U32) : U32)),
            List.append(&2,Views.Axis,[],List.append(&2,Views.Axis,BandViews.rows_axes(_,height,width,[]),rest))) : List<&2,Views.Axis>}
        images,{==})
    case rowed None{}: Unit{}

# Every classification a view receives is one of its banded forms.
def classify(+view_axes: List<&2,Views.Axis>,+offset: U32,+geometry: Bands.Geometry) ->
  Classification(view_axes,offset,geometry,BandViews.classify(view_axes,offset,geometry)):
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  +image = (channels * (height * width : U32) : U32)
  classify_rows(U32.is_lt(1,height),BandViews.after_images(U32.is_lt(1,batch),batch,image,view_axes),offset,batch,channels,height,width,view_axes,{==},
    after_images(U32.is_lt(1,batch),batch,image,view_axes))
