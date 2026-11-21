# The weight gradient of tensors of cells: the run holds the pieces of the
# input and the output gradient while it reads them, gives them back as
# tensors of their contract, and the cells of the gradient hold what the
# kernel model sums.

import Base
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_pieces.bend as Pieces
import ../tensor_move.bend as Move
import ../kernel_tile.bend as Tile
import ../machine_limits.bend as Machine
import ../kernel_shape.bend as KernelShape
import ../tensor_view.bend as TensorView
import ../kernel_gemm.bend as KernelGemm
import ../tensor_parameter_gradient.bend as ParameterGradient
import ../kernel_parameter_gradient.bend as Parameter
import ../kernel_weight_gradient.bend as Kernel
import ./kernel_weight_gradient_group.bend as Groups
import ./kernel_weight_gradient_packing.bend as Packing
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_count_refinement.bend as Count
import ./tensor_global_shape_count.bend as Global
import ./tensor_adjoint_index.bend as AdjointIndex
import ./boolean_logic.bend as Logic
import ./u32_comparison.bend as Comparison
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./u32_full_range.bend as FullRange
import ./nat_ceiling.bend as Ceiling
import ./word_encoding.bend as WordEncoding
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./encoded_coordinates.bend as Encoded
import ./kernel_weight_gradient_held.bend as Sources
import ./kernel_weight_gradient_images.bend as Images
import ./kernel_weight_gradient_routes.bend as Routes
import ./grouped_reduction_refinement.bend as GroupLaws
import ./tensor_move_readback.bend as Held
import ../default_machine_limits.bend as Default
import ./tensor_list_values.bend as Listed
import ./storage_buffer_model.bend as Buffer
import ./nat_to_u32_bounds.bend as Bounds
import ./tensor_halo_windows.bend as Windows
import ./tensor_released.bend as Released
import ./storage_tree_model.bend as Tree
import ../storage_buffer.bend as Storage
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_map.bend as TiledMap
import ./tensor_storage_count.bend as StorageCount
import ./tensor_tiled_public.bend as TiledPublic
import ./machine_limit_bounds.bend as LimitBounds
import ./tensor_tiled_padding.bend as Padding
import ./tensor_tiled_padded_cells.bend as PaddedCells
import ./kernel_tile_refinement.bend as TileLaws
import ./list_storage_readback.bend as ListRead
import ./tensor_allocation_depth.bend as Depth

# ---- The images a call names ----

def image_count(image: ParameterGradient.ImageGeometry) -> U32:
  ParameterGradient.ImageGeometry{n,c,h,w} = image
  n

def image_channels(image: ParameterGradient.ImageGeometry) -> U32:
  ParameterGradient.ImageGeometry{n,c,h,w} = image
  c

def image_height(image: ParameterGradient.ImageGeometry) -> U32:
  ParameterGradient.ImageGeometry{n,c,h,w} = image
  h

def image_width(image: ParameterGradient.ImageGeometry) -> U32:
  ParameterGradient.ImageGeometry{n,c,h,w} = image
  w

def image_or(named: Maybe<&2,ParameterGradient.ImageGeometry>) -> ParameterGradient.ImageGeometry:
  match named:
    case Some{image}: image
    case None{}: ParameterGradient.ImageGeometry{0,0,0,0}

def named_image(named: Maybe<&2,ParameterGradient.ImageGeometry>) -> Bool:
  match named:
    case Some{image}: True{}
    case None{}: False{}

def unnamed(+image: ParameterGradient.ImageGeometry,+named: {None{} == Some{image} : Maybe<&2,ParameterGradient.ImageGeometry>}) -> Empty:
  WordEncoding.false_true(Equal.cong(Maybe<&2,ParameterGradient.ImageGeometry>,Bool,named_image,None{},Some{image},named))

# An image of cells with matching groups is that image, its groups named.
type Grouped<-same: Bool,-image: ParameterGradient.ImageGeometry,-named: ParameterGradient.ImageGeometry> is Data:
  Grouped{same_true: {same == True{} : Bool},exact: {image == named : ParameterGradient.ImageGeometry}}

def grouped_named(+same: Bool,+image: ParameterGradient.ImageGeometry,+named: ParameterGradient.ImageGeometry,
  +found: {ParameterGradient.grouped(same,image) == Some{named} : Maybe<&2,ParameterGradient.ImageGeometry>}) -> Grouped<same,image,named>:
  match same:
    case True{}: Grouped{{==},Equal.cong(Maybe<&2,ParameterGradient.ImageGeometry>,ParameterGradient.ImageGeometry,image_or,Some{image},Some{named},found)}
    case False{}: Empty.absurd(Grouped<False{},image,named>,unnamed(named,found))

# Groups that match the groups of lanes count as many.
def groups_matched(+groups: U32,+lanes: U32,+same: {U32.is_eq(groups,((lanes + 7) / 8 : U32)) == True{} : Bool}) ->
  {U32.to_nat(groups) == U32.to_nat(Grid.groups(lanes)) : Nat}:
  Order.equal_of_is_eq(U32.to_nat(groups),U32.to_nat(Grid.groups(lanes)),%Comparison.eq(groups,Grid.groups(lanes)) : {_ == True{} : Bool} same)

# The cells of an image: its channels are the lanes, and the cells are the
# images, rows, columns and groups of lanes.
type ImageCells<-shape: List<&2,U32>,-lanes: U32,-image: ParameterGradient.ImageGeometry> is Data:
  ImageCells{channels: {image_channels(image) == lanes : U32},
    elements: {ShapeLaws.elements(shape) ==
      Nat.mul(U32.to_nat(image_count(image)),Nat.mul(U32.to_nat(image_height(image)),Nat.mul(U32.to_nat(image_width(image)),U32.to_nat(Grid.groups(lanes))))) : Nat}}

def planar_cells(+h: U32,+w: U32,+groups: U32,+lanes: U32,+same: {U32.is_eq(groups,((lanes + 7) / 8 : U32)) == True{} : Bool}) ->
  ImageCells<[h,w,groups],lanes,ParameterGradient.ImageGeometry{1,lanes,h,w}>:
  +inner = Nat.mul(U32.to_nat(h),Nat.mul(U32.to_nat(w),U32.to_nat(Grid.groups(lanes))))
  ImageCells{{==},
    Equal.trans(Nat,ShapeLaws.elements([h,w,groups]),inner,Nat.add(inner,0n),
      ShapeLaws.elements_three_exact(h,w,groups,U32.to_nat(h),U32.to_nat(w),U32.to_nat(Grid.groups(lanes)),{==},{==},groups_matched(groups,lanes,same)),
      Equal.sym(Nat,Nat.add(inner,0n),inner,Addition.zero_right(inner)))}

def batched_cells(+n: U32,+h: U32,+w: U32,+groups: U32,+lanes: U32,+same: {U32.is_eq(groups,((lanes + 7) / 8 : U32)) == True{} : Bool}) ->
  ImageCells<[n,h,w,groups],lanes,ParameterGradient.ImageGeometry{n,lanes,h,w}>:
  ImageCells{{==},
    ShapeLaws.elements_four_exact(n,h,w,groups,U32.to_nat(n),U32.to_nat(h),U32.to_nat(w),U32.to_nat(Grid.groups(lanes)),{==},{==},{==},groups_matched(groups,lanes,same))}

def planar_named(+h: U32,+w: U32,+groups: U32,+lanes: U32,+image: ParameterGradient.ImageGeometry,
  +grouped: Grouped<U32.is_eq(groups,((lanes + 7) / 8 : U32)),ParameterGradient.ImageGeometry{1,lanes,h,w},image>) ->
  ImageCells<[h,w,groups],lanes,image>:
  Grouped{+same,+exact} = grouped
  %exact : ImageCells<[h,w,groups],lanes,_>
  planar_cells(h,w,groups,lanes,same)

def batched_named(+n: U32,+h: U32,+w: U32,+groups: U32,+lanes: U32,+image: ParameterGradient.ImageGeometry,
  +grouped: Grouped<U32.is_eq(groups,((lanes + 7) / 8 : U32)),ParameterGradient.ImageGeometry{n,lanes,h,w},image>) ->
  ImageCells<[n,h,w,groups],lanes,image>:
  Grouped{+same,+exact} = grouped
  %exact : ImageCells<[n,h,w,groups],lanes,_>
  batched_cells(n,h,w,groups,lanes,same)

# Cells [n,h,w,groups] or [h,w,groups] that name an image are its cells.
def image_cells(+shape: List<&2,U32>,+lanes: U32,+image: ParameterGradient.ImageGeometry,
  +named: {ParameterGradient.cell_image(shape,lanes) == Some{image} : Maybe<&2,ParameterGradient.ImageGeometry>}) -> ImageCells<shape,lanes,image>:
  match shape:
    case Nil{}: Empty.absurd(ImageCells<[],lanes,image>,unnamed(image,named))
    case Con{+first,+after_first}:
      match after_first:
        case Nil{}: Empty.absurd(ImageCells<[first],lanes,image>,unnamed(image,named))
        case Con{+second,+after_second}:
          match after_second:
            case Nil{}: Empty.absurd(ImageCells<[first,second],lanes,image>,unnamed(image,named))
            case Con{+third,+after_third}:
              match after_third:
                case Nil{}:
                  planar_named(first,second,third,lanes,image,
                    grouped_named(U32.is_eq(third,((lanes + 7) / 8 : U32)),ParameterGradient.ImageGeometry{1,lanes,first,second},image,named))
                case Con{+fourth,+after_fourth}:
                  match after_fourth:
                    case Nil{}:
                      batched_named(first,second,third,fourth,lanes,image,
                        grouped_named(U32.is_eq(fourth,((lanes + 7) / 8 : U32)),ParameterGradient.ImageGeometry{first,lanes,second,third},image,named))
                    case Con{+fifth,+rest}:
                      Empty.absurd(ImageCells<first <> second <> third <> fourth <> fifth <> rest,lanes,image>,unnamed(image,named))

# ---- The conditions of a call ----

# The weights of a call [o,c,k,k].
def weight_shape(+xi: ParameterGradient.ImageGeometry,+di: ParameterGradient.ImageGeometry,+k: U32) -> List<&2,U32>:
  ParameterGradient.ImageGeometry{n,+c,h,w} = xi
  ParameterGradient.ImageGeometry{dn,+o,rows,columns} = di
  [o,c,k,k]

# The conditions of a valid call before the count of its weights.
def conditions(+xi: ParameterGradient.ImageGeometry,+di: ParameterGradient.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32) -> Bool:
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{dn,+o,+rows,+columns} = di
  ParameterGradient.positive(xi) && ParameterGradient.positive(di) && U32.is_eq(n,dn) && U32.is_lt(0,k) && U32.is_lt(0,s) && U32.is_lt(0,g) &&
    Nat.is_le(U32.to_nat(k),Nat.add(U32.to_nat(h),Nat.mul(2n,U32.to_nat(p)))) &&
    Nat.is_le(U32.to_nat(k),Nat.add(U32.to_nat(w),Nat.mul(2n,U32.to_nat(p)))) &&
    Nat.is_eq(U32.to_nat(rows),KernelShape.axis_size(U32.to_nat(h),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p))) &&
    Nat.is_eq(U32.to_nat(columns),KernelShape.axis_size(U32.to_nat(w),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p)))

# A valid call meets the conditions and counts its weights, and its packed
# sources fit the capacity.
def valid_counted(+xi: ParameterGradient.ImageGeometry,+di: ParameterGradient.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,+limits: Machine.Limits,
  +valid: {ParameterGradient.valid_with(xi,di,k,s,p,g,limits) == True{} : Bool}) ->
  {conditions(xi,di,k,s,p,g) && TensorView.has_count(Tensors.count_of(weight_shape(xi,di,k))) == True{} : Bool}:
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{+dn,+o,+rows,+columns} = di
  Logic.left(conditions(xi,di,k,s,p,g) && TensorView.has_count(Tensors.count_of([o,c,k,k])),ParameterGradient.packed_fits(xi,di,k,s,Machine.capacity(limits)),valid)

def valid_packed(+xi: ParameterGradient.ImageGeometry,+di: ParameterGradient.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,+limits: Machine.Limits,
  +valid: {ParameterGradient.valid_with(xi,di,k,s,p,g,limits) == True{} : Bool}) -> {ParameterGradient.packed_fits(xi,di,k,s,Machine.capacity(limits)) == True{} : Bool}:
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{+dn,+o,+rows,+columns} = di
  Logic.right(conditions(xi,di,k,s,p,g) && TensorView.has_count(Tensors.count_of([o,c,k,k])),ParameterGradient.packed_fits(xi,di,k,s,Machine.capacity(limits)),valid)

# The positive sizes of a valid call, its images the same and its packed
# sources within the capacity.
type Positive<-xi: ParameterGradient.ImageGeometry,-di: ParameterGradient.ImageGeometry,-k: U32,-s: U32,-g: U32,-capacity: U32> is Data:
  Positive{input: {ParameterGradient.positive(xi) == True{} : Bool},gradient: {ParameterGradient.positive(di) == True{} : Bool},
    kernel: {U32.is_lt(0,k) == True{} : Bool},stride: {U32.is_lt(0,s) == True{} : Bool},group: {U32.is_lt(0,g) == True{} : Bool},
    images: {U32.is_eq(image_count(xi),image_count(di)) == True{} : Bool},bands: {ParameterGradient.packed_fits(xi,di,k,s,capacity) == True{} : Bool}}

def valid_positive(+xi: ParameterGradient.ImageGeometry,+di: ParameterGradient.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,+limits: Machine.Limits,
  +valid: {ParameterGradient.valid_with(xi,di,k,s,p,g,limits) == True{} : Bool}) -> Positive<xi,di,k,s,g,Machine.capacity(limits)>:
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{+dn,+o,+rows,+columns} = di
  +ab = ParameterGradient.positive(xi) && ParameterGradient.positive(di)
  +abc = ab && U32.is_eq(n,dn)
  +ak = abc && U32.is_lt(0,k)
  +as = ak && U32.is_lt(0,s)
  +ag = as && U32.is_lt(0,g)
  +height = Nat.is_le(U32.to_nat(k),Nat.add(U32.to_nat(h),Nat.mul(2n,U32.to_nat(p))))
  +width = Nat.is_le(U32.to_nat(k),Nat.add(U32.to_nat(w),Nat.mul(2n,U32.to_nat(p))))
  +row = Nat.is_eq(U32.to_nat(rows),KernelShape.axis_size(U32.to_nat(h),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p)))
  +column = Nat.is_eq(U32.to_nat(columns),KernelShape.axis_size(U32.to_nat(w),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p)))
  +ah = ag && height
  +aw = ah && width
  +ar = aw && row
  +ac_true = Logic.left(conditions(xi,di,k,s,p,g),TensorView.has_count(Tensors.count_of([o,c,k,k])),valid_counted(xi,di,k,s,p,g,limits,valid))
  +ar_true = Logic.left(ar,column,ac_true)
  +aw_true = Logic.left(aw,row,ar_true)
  +ah_true = Logic.left(ah,width,aw_true)
  +ag_true = Logic.left(ag,height,ah_true)
  +as_true = Logic.left(as,U32.is_lt(0,g),ag_true)
  +ak_true = Logic.left(ak,U32.is_lt(0,s),as_true)
  +abc_true = Logic.left(abc,U32.is_lt(0,k),ak_true)
  +ab_true = Logic.left(ab,U32.is_eq(n,dn),abc_true)
  Positive{Logic.left(ParameterGradient.positive(xi),ParameterGradient.positive(di),ab_true),Logic.right(ParameterGradient.positive(xi),ParameterGradient.positive(di),ab_true),
    Logic.right(abc,U32.is_lt(0,k),ak_true),Logic.right(ak,U32.is_lt(0,s),as_true),Logic.right(as,U32.is_lt(0,g),ag_true),
    Logic.right(ab,U32.is_eq(n,dn),abc_true),valid_packed(xi,di,k,s,p,g,limits,valid)}

# A shape whose count is known is counted.
def counted_when(+shape: List<&2,U32>,found: Maybe<&2,U32>,+same: {Tensors.count_of(shape) == found : Maybe<&2,U32>},+known: {TensorView.has_count(found) == True{} : Bool}) ->
  Global.Counted<shape>:
  match found:
    case Some{+count}: Global.Counted{count,same}
    case None{}: Empty.absurd(Global.Counted<shape>,WordEncoding.false_true(known))

def valid_weights(+xi: ParameterGradient.ImageGeometry,+di: ParameterGradient.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,+limits: Machine.Limits,
  +valid: {ParameterGradient.valid_with(xi,di,k,s,p,g,limits) == True{} : Bool}) -> Global.Counted<weight_shape(xi,di,k)>:
  counted_when(weight_shape(xi,di,k),Tensors.count_of(weight_shape(xi,di,k)),{==},
    Logic.right(conditions(xi,di,k,s,p,g),TensorView.has_count(Tensors.count_of(weight_shape(xi,di,k))),valid_counted(xi,di,k,s,p,g,limits,valid)))

# ---- The geometry of a call ----

# The kernel geometry of a call: its taps are the channels at every kernel
# position.
def call_geometry(+xi: ParameterGradient.ImageGeometry,+di: ParameterGradient.ImageGeometry,+k: U32,+s: U32,+p: U32) -> Kernel.Geometry:
  ParameterGradient.ImageGeometry{n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{dn,+o,+rows,+columns} = di
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,U32.from_nat(Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))))}

def tap_total(+xi: ParameterGradient.ImageGeometry,+k: U32) -> Nat:
  ParameterGradient.ImageGeometry{n,+c,h,w} = xi
  Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))

# The geometry of a positive call whose weights and packed sources fit the
# capacity is ready for the kernel.
def geometry_ready(+limits: Machine.Limits,+aligned: {U32.is_zero((Machine.capacity(limits) % 8 : U32)) == True{} : Bool},
  +xi: ParameterGradient.ImageGeometry,+di: ParameterGradient.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,
  +positive: Positive<xi,di,k,s,g,Machine.capacity(limits)>,+count: U32,
  +exact: {ShapeLaws.elements(weight_shape(xi,di,k)) == U32.to_nat(count) : Nat},
  +bounded: {U32.is_le(count,Machine.capacity(limits)) == True{} : Bool}) -> Groups.GeometryReady<call_geometry(xi,di,k,s,p)>:
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{+dn,+o,+rows,+columns} = di
  Positive{+xp,+dp,+kp,+sp,gp,images,+bands} = positive
  +xc = Logic.left((U32.is_lt(0,n) && U32.is_lt(0,c)) && U32.is_lt(0,h),U32.is_lt(0,w),xp)
  +nc = Logic.left(U32.is_lt(0,n) && U32.is_lt(0,c),U32.is_lt(0,h),xc)
  +cp_word = Logic.right(U32.is_lt(0,n),U32.is_lt(0,c),nc)
  +dr = Logic.left((U32.is_lt(0,dn) && U32.is_lt(0,o)) && U32.is_lt(0,rows),U32.is_lt(0,columns),dp)
  +no = Logic.left(U32.is_lt(0,dn) && U32.is_lt(0,o),U32.is_lt(0,rows),dr)
  +op_word = Logic.right(U32.is_lt(0,dn),U32.is_lt(0,o),no)
  +columns_word = Logic.right((U32.is_lt(0,dn) && U32.is_lt(0,o)) && U32.is_lt(0,rows),U32.is_lt(0,columns),dp)
  +cp = {%Comparison.lt(0,c) : {_ == True{} : Bool} cp_word : {Nat.is_lt(0n,U32.to_nat(c)) == True{} : Bool}}
  +op = {%Comparison.lt(0,o) : {_ == True{} : Bool} op_word : {Nat.is_lt(0n,U32.to_nat(o)) == True{} : Bool}}
  +kernel_positive = {%Comparison.lt(0,k) : {_ == True{} : Bool} kp : {Nat.is_lt(0n,U32.to_nat(k)) == True{} : Bool}}
  +stride_positive = {%Comparison.lt(0,s) : {_ == True{} : Bool} sp : {Nat.is_lt(0n,U32.to_nat(s)) == True{} : Bool}}
  +columns_positive = {%Comparison.lt(0,columns) : {_ == True{} : Bool} columns_word : {Nat.is_lt(0n,U32.to_nat(columns)) == True{} : Bool}}
  +square = Nat.mul(U32.to_nat(k),U32.to_nat(k))
  +taps = tap_total(xi,k)
  +total = Nat.mul(U32.to_nat(o),taps)
  +capacity = Machine.capacity(limits)
  +cap = U32.to_nat(capacity)
  +square_positive = Order.product_positive(U32.to_nat(k),U32.to_nat(k),kernel_positive,kernel_positive)
  +taps_positive = Order.product_positive(U32.to_nat(c),square,cp,square_positive)
  +total_bound = {%AdjointIndex.elements_four(o,c,k) : {Nat.is_le(_,cap) == True{} : Bool}
    %Equal.sym(Nat,ShapeLaws.elements([o,c,k,k]),U32.to_nat(count),exact) : {Nat.is_le(_,cap) == True{} : Bool}
    %Comparison.le(count,capacity) : {_ == True{} : Bool} bounded : {Nat.is_le(total,cap) == True{} : Bool}}
  +taps_bound = Order.transitive(taps,total,cap,
    %RowMajor.multiply_commutative(taps,U32.to_nat(o)) : {Nat.is_le(taps,_) == True{} : Bool} RowMajor.product_covers(taps,U32.to_nat(o),op),total_bound)
  +square_bound = Order.transitive(square,taps,cap,
    %RowMajor.multiply_commutative(square,U32.to_nat(c)) : {Nat.is_le(square,_) == True{} : Bool} RowMajor.product_covers(square,U32.to_nat(c),cp),taps_bound)
  +taps_exact = FullRange.roundtrip(taps,capacity,taps_bound)
  +native_taps_positive = {%Equal.sym(Nat,U32.to_nat(U32.from_nat(taps)),taps,taps_exact) : {Nat.is_lt(0n,_) == True{} : Bool} taps_positive :
    {Nat.is_lt(0n,U32.to_nat(U32.from_nat(taps))) == True{} : Bool}}
  +native_count_bound = {%Equal.sym(Nat,U32.to_nat(U32.from_nat(taps)),taps,taps_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(o)),cap) == True{} : Bool}
    %Equal.sym(Nat,Nat.mul(taps,U32.to_nat(o)),total,RowMajor.multiply_commutative(taps,U32.to_nat(o))) : {Nat.is_le(_,cap) == True{} : Bool}
    total_bound : {Nat.is_le(Nat.mul(U32.to_nat(U32.from_nat(taps)),U32.to_nat(o)),cap) == True{} : Bool}}
  +phases = Nat.min(U32.to_nat(s),U32.to_nat(k))
  +reach = Nat.div(Nat.sub(U32.to_nat(k),1n),phases)
  +band_rows = Nat.add(U32.to_nat(rows),reach)
  +band_columns = Nat.add(U32.to_nat(columns),reach)
  +pixels = Nat.mul(Nat.mul(Nat.mul(Nat.mul(Nat.div(Nat.add(U32.to_nat(c),7n),8n),phases),phases),band_rows),band_columns)
  +gradients = Nat.mul(Nat.mul(Nat.div(Nat.add(U32.to_nat(o),7n),8n),U32.to_nat(rows)),U32.to_nat(columns))
  +pixels_bound = {%Ceiling.shifted_formula(U32.to_nat(c),8n,{==}) : {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(Nat.mul(_,phases),phases),band_rows),band_columns),cap) == True{} : Bool}
    Logic.left(Nat.is_le(pixels,cap),Nat.is_le(gradients,cap),bands) : {Nat.is_le(Groups.pixel_extent(call_geometry(xi,di,k,s,p)),cap) == True{} : Bool}}
  +gradients_bound = {%Ceiling.shifted_formula(U32.to_nat(o),8n,{==}) : {Nat.is_le(Nat.mul(Nat.mul(_,U32.to_nat(rows)),U32.to_nat(columns)),cap) == True{} : Bool}
    Logic.right(Nat.is_le(pixels,cap),Nat.is_le(gradients,cap),bands) : {Nat.is_le(Groups.gradient_extent(call_geometry(xi,di,k,s,p)),cap) == True{} : Bool}}
  Groups.GeometryReady{limits,columns_positive,kernel_positive,stride_positive,square_bound,native_taps_positive,op,native_count_bound,aligned,taps_exact,
    pixels_bound,gradients_bound}

# The limits of the geometry of a call are the limits it was made for.
def ready_limits(+limits: Machine.Limits,+aligned: {U32.is_zero((Machine.capacity(limits) % 8 : U32)) == True{} : Bool},
  +xi: ParameterGradient.ImageGeometry,+di: ParameterGradient.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,
  +positive: Positive<xi,di,k,s,g,Machine.capacity(limits)>,+count: U32,
  +exact: {ShapeLaws.elements(weight_shape(xi,di,k)) == U32.to_nat(count) : Nat},
  +bounded: {U32.is_le(count,Machine.capacity(limits)) == True{} : Bool}) ->
  {Groups.domain_limits(call_geometry(xi,di,k,s,p),geometry_ready(limits,aligned,xi,di,k,s,p,g,positive,count,exact,bounded)) == limits : Machine.Limits}:
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{+dn,+o,+rows,+columns} = di
  Positive{+xp,+dp,+kp,+sp,+gp,+images,+bands} = positive
  {==}

# ---- The batch of a call ----

# The images of a call read from the cells of x and d, the geometry ready.
def call_batch(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+x_context: X,+d_context: D,
  +xi: ParameterGradient.ImageGeometry,+di: ParameterGradient.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,
  +domain: Groups.GeometryReady<call_geometry(xi,di,k,s,p)>,
  +columns_positive: {Nat.is_lt(0n,Groups.column_count(call_geometry(xi,di,k,s,p))) == True{} : Bool},+group_positive: {Nat.is_lt(0n,U32.to_nat(g)) == True{} : Bool},
  +x_cells: List<&2,U32>,
  +x_room: {Nat.is_le(Nat.mul(U32.to_nat(image_count(xi)),Sources.input_cells(call_geometry(xi,di,k,s,p))),ShapeLaws.elements(x_cells)) == True{} : Bool},
  +x_bounded: {Nat.is_le(ShapeLaws.elements(x_cells),U32.to_nat(Machine.capacity(Groups.domain_limits(call_geometry(xi,di,k,s,p),domain)))) == True{} : Bool},
  +d_cells: List<&2,U32>,
  +d_room: {Nat.is_le(Nat.mul(U32.to_nat(image_count(xi)),Groups.gradient_extent(call_geometry(xi,di,k,s,p))),ShapeLaws.elements(d_cells)) == True{} : Bool},
  +d_bounded: {Nat.is_le(ShapeLaws.elements(d_cells),U32.to_nat(Machine.capacity(Groups.domain_limits(call_geometry(xi,di,k,s,p),domain)))) == True{} : Bool}) ->
  Images.Batch<X,xv,D,dv>:
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{+dn,+o,+rows,+columns} = di
  Images.Batch{x_context,d_context,call_geometry(ParameterGradient.ImageGeometry{n,c,h,w},ParameterGradient.ImageGeometry{dn,o,rows,columns},k,s,p),domain,
    U32.to_nat(n),U32.to_nat(rows),U32.to_nat(columns),U32.to_nat(g),g,
    GroupLaws.ranges_bounded(U32.to_nat(rows),U32.to_nat(columns),U32.to_nat(g),columns_positive,group_positive),
    ShapeLaws.elements(x_cells),Held.layout_spans(Pieces.ranges_of_grid(Tensors.grid_of(x_cells)),Held.Ended{}),x_room,x_bounded,
    ShapeLaws.elements(d_cells),Held.layout_spans(Pieces.ranges_of_grid(Tensors.grid_of(d_cells)),Held.Ended{}),d_room,d_bounded}

# The kernel geometry of the work of a call is the geometry of the call.
def work_geometry(+xi: ParameterGradient.ImageGeometry,+di: ParameterGradient.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32) ->
  {Kernel.geometry(ParameterGradient.work(xi,di,k,s,p,g)) == call_geometry(xi,di,k,s,p) : Kernel.Geometry}:
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{+dn,+o,+rows,+columns} = di
  +taps = U32.from_nat(Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))))
  %Equal.sym(U32,U32.from_nat(U32.to_nat(c)),c,Arithmetic.native_roundtrip(c)) :
    {Kernel.Geometry{_,U32.from_nat(U32.to_nat(o)),U32.from_nat(U32.to_nat(h)),U32.from_nat(U32.to_nat(w)),U32.from_nat(U32.to_nat(k)),U32.from_nat(U32.to_nat(s)),
      U32.from_nat(U32.to_nat(p)),U32.from_nat(U32.to_nat(rows)),U32.from_nat(U32.to_nat(columns)),taps} == Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} : Kernel.Geometry}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(o)),o,Arithmetic.native_roundtrip(o)) :
    {Kernel.Geometry{c,_,U32.from_nat(U32.to_nat(h)),U32.from_nat(U32.to_nat(w)),U32.from_nat(U32.to_nat(k)),U32.from_nat(U32.to_nat(s)),
      U32.from_nat(U32.to_nat(p)),U32.from_nat(U32.to_nat(rows)),U32.from_nat(U32.to_nat(columns)),taps} == Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} : Kernel.Geometry}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(h)),h,Arithmetic.native_roundtrip(h)) :
    {Kernel.Geometry{c,o,_,U32.from_nat(U32.to_nat(w)),U32.from_nat(U32.to_nat(k)),U32.from_nat(U32.to_nat(s)),
      U32.from_nat(U32.to_nat(p)),U32.from_nat(U32.to_nat(rows)),U32.from_nat(U32.to_nat(columns)),taps} == Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} : Kernel.Geometry}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(w)),w,Arithmetic.native_roundtrip(w)) :
    {Kernel.Geometry{c,o,h,_,U32.from_nat(U32.to_nat(k)),U32.from_nat(U32.to_nat(s)),
      U32.from_nat(U32.to_nat(p)),U32.from_nat(U32.to_nat(rows)),U32.from_nat(U32.to_nat(columns)),taps} == Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} : Kernel.Geometry}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(k)),k,Arithmetic.native_roundtrip(k)) :
    {Kernel.Geometry{c,o,h,w,_,U32.from_nat(U32.to_nat(s)),
      U32.from_nat(U32.to_nat(p)),U32.from_nat(U32.to_nat(rows)),U32.from_nat(U32.to_nat(columns)),taps} == Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} : Kernel.Geometry}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(s)),s,Arithmetic.native_roundtrip(s)) :
    {Kernel.Geometry{c,o,h,w,k,_,U32.from_nat(U32.to_nat(p)),U32.from_nat(U32.to_nat(rows)),U32.from_nat(U32.to_nat(columns)),taps} ==
      Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} : Kernel.Geometry}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(p)),p,Arithmetic.native_roundtrip(p)) :
    {Kernel.Geometry{c,o,h,w,k,s,_,U32.from_nat(U32.to_nat(rows)),U32.from_nat(U32.to_nat(columns)),taps} == Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} : Kernel.Geometry}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(rows)),rows,Arithmetic.native_roundtrip(rows)) :
    {Kernel.Geometry{c,o,h,w,k,s,p,_,U32.from_nat(U32.to_nat(columns)),taps} == Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} : Kernel.Geometry}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(columns)),columns,Arithmetic.native_roundtrip(columns)) :
    {Kernel.Geometry{c,o,h,w,k,s,p,rows,_,taps} == Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} : Kernel.Geometry}
  {==}

# Groups of eight of a count, written as the kernel writes them, are its
# ceiling.
def blocks_cells(+count: Nat,+rows: Nat,+columns: Nat) ->
  {Nat.mul(Nat.div(Nat.add(count,7n),8n),Nat.mul(rows,columns)) == Nat.mul(Nat.mul(Ceiling.ceiling(count,8n),rows),columns) : Nat}:
  Equal.trans(Nat,Nat.mul(Nat.div(Nat.add(count,7n),8n),Nat.mul(rows,columns)),Nat.mul(Ceiling.ceiling(count,8n),Nat.mul(rows,columns)),
    Nat.mul(Nat.mul(Ceiling.ceiling(count,8n),rows),columns),
    Equal.cong(Nat,Nat,blocks => Nat.mul(blocks,Nat.mul(rows,columns)),Nat.div(Nat.add(count,7n),8n),Ceiling.ceiling(count,8n),Ceiling.shifted_formula(count,8n,{==})),
    Equal.sym(Nat,Nat.mul(Nat.mul(Ceiling.ceiling(count,8n),rows),columns),Nat.mul(Ceiling.ceiling(count,8n),Nat.mul(rows,columns)),
      RowMajor.assoc(Ceiling.ceiling(count,8n),rows,columns)))

# The positions of the images of a call are no more than their gradient
# cells.
def positions_below(+images: Nat,+blocks: Nat,+rows: Nat,+columns: Nat,+positive: {Nat.is_lt(0n,blocks) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(images,Nat.mul(rows,columns)),Nat.mul(images,Nat.mul(Nat.mul(blocks,rows),columns))) == True{} : Bool}:
  RowMajor.mul_le_left(images,Nat.mul(rows,columns),Nat.mul(Nat.mul(blocks,rows),columns),
    %Equal.sym(Nat,Nat.mul(Nat.mul(blocks,rows),columns),Nat.mul(blocks,Nat.mul(rows,columns)),RowMajor.assoc(blocks,rows,columns)) :
      {Nat.is_le(Nat.mul(rows,columns),_) == True{} : Bool}
    %RowMajor.multiply_commutative(Nat.mul(rows,columns),blocks) : {Nat.is_le(Nat.mul(rows,columns),_) == True{} : Bool}
    RowMajor.product_covers(Nat.mul(rows,columns),blocks,positive))

# The work of a call reads the batch of the call.
def work_batch(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+x_context: X,+d_context: D,
  +xi: ParameterGradient.ImageGeometry,+di: ParameterGradient.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,
  +domain: Groups.GeometryReady<call_geometry(xi,di,k,s,p)>,
  +rows_positive: {Nat.is_lt(0n,Groups.row_count(call_geometry(xi,di,k,s,p))) == True{} : Bool},
  +columns_positive: {Nat.is_lt(0n,Groups.column_count(call_geometry(xi,di,k,s,p))) == True{} : Bool},+group_positive: {Nat.is_lt(0n,U32.to_nat(g)) == True{} : Bool},
  +x_cells: List<&2,U32>,
  +x_room: {Nat.is_le(Nat.mul(U32.to_nat(image_count(xi)),Sources.input_cells(call_geometry(xi,di,k,s,p))),ShapeLaws.elements(x_cells)) == True{} : Bool},
  +x_bounded: {Nat.is_le(ShapeLaws.elements(x_cells),U32.to_nat(Machine.capacity(Groups.domain_limits(call_geometry(xi,di,k,s,p),domain)))) == True{} : Bool},
  +d_cells: List<&2,U32>,
  +d_room: {Nat.is_le(Nat.mul(U32.to_nat(image_count(xi)),Groups.gradient_extent(call_geometry(xi,di,k,s,p))),ShapeLaws.elements(d_cells)) == True{} : Bool},
  +d_bounded: {Nat.is_le(ShapeLaws.elements(d_cells),U32.to_nat(Machine.capacity(Groups.domain_limits(call_geometry(xi,di,k,s,p),domain)))) == True{} : Bool}) ->
  Routes.WorkBatch<X,xv,D,dv,
    call_batch(~X,~xv,~D,~dv,x_context,d_context,xi,di,k,s,p,g,domain,columns_positive,group_positive,x_cells,x_room,x_bounded,d_cells,d_room,d_bounded),
    ParameterGradient.work(xi,di,k,s,p,g)>:
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{+dn,+o,+rows,+columns} = di
  +x_image = {ParameterGradient.ImageGeometry{n,c,h,w} : ParameterGradient.ImageGeometry}
  +d_image = {ParameterGradient.ImageGeometry{dn,o,rows,columns} : ParameterGradient.ImageGeometry}
  +geometry = call_geometry(x_image,d_image,k,s,p)
  +same = work_geometry(x_image,d_image,k,s,p,g)
  +cap = U32.to_nat(Machine.capacity(Groups.domain_limits(geometry,domain)))
  +images = U32.to_nat(n)
  +positions = Nat.mul(images,Nat.mul(U32.to_nat(rows),U32.to_nat(columns)))
  +input_total = Nat.mul(images,Sources.input_cells(geometry))
  +gradient_total = Nat.mul(images,Groups.gradient_extent(geometry))
  +blocks = Ceiling.ceiling(U32.to_nat(o),8n)
  +gradient_room = Order.transitive(gradient_total,ShapeLaws.elements(d_cells),cap,d_room,d_bounded)
  +positions_room = Order.transitive(positions,gradient_total,cap,
    positions_below(images,blocks,U32.to_nat(rows),U32.to_nat(columns),Groups.output_blocks_positive(geometry,domain)),gradient_room)
  +input_room = Order.transitive(input_total,ShapeLaws.elements(x_cells),cap,x_room,x_bounded)
  +plane = Nat.mul(U32.to_nat(rows),U32.to_nat(columns))
  +plane_positive = Order.product_positive(U32.to_nat(rows),U32.to_nat(columns),rows_positive,columns_positive)
  +work_cells = Nat.mul(images,Nat.mul(Nat.div(Nat.add(U32.to_nat(c),7n),8n),Nat.mul(U32.to_nat(h),U32.to_nat(w))))
  +cells_exact = Equal.cong(Nat,Nat,cells => Nat.mul(images,cells),Nat.mul(Nat.div(Nat.add(U32.to_nat(c),7n),8n),Nat.mul(U32.to_nat(h),U32.to_nat(w))),
    Sources.input_cells(geometry),blocks_cells(U32.to_nat(c),U32.to_nat(h),U32.to_nat(w)))
  +groups_exact = Ceiling.shifted_formula(U32.to_nat(o),8n,{==})
  Routes.WorkBatch{
    Equal.cong(Kernel.Geometry,Kernel.BatchContext,geometry => Kernel.BatchContext{geometry,U32.to_nat(rows),U32.to_nat(columns),U32.to_nat(g)},
      Kernel.geometry(ParameterGradient.work(x_image,d_image,k,s,p,g)),geometry,same),
    {==},
    Equal.cong(Kernel.Geometry,U32,geometry => Kernel.cells(geometry),Kernel.geometry(ParameterGradient.work(x_image,d_image,k,s,p,g)),geometry,same),
    Equal.cong(Nat,U32,U32.from_nat,Nat.mul(Nat.div(Nat.add(U32.to_nat(c),7n),8n),Nat.mul(U32.to_nat(h),U32.to_nat(w))),Sources.input_cells(geometry),
      blocks_cells(U32.to_nat(c),U32.to_nat(h),U32.to_nat(w))),
    Equal.cong(Nat,U32,U32.from_nat,Nat.mul(Nat.div(Nat.add(U32.to_nat(o),7n),8n),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))),Groups.gradient_extent(geometry),
      blocks_cells(U32.to_nat(o),U32.to_nat(rows),U32.to_nat(columns))),
    Order.transitive(images,positions,cap,RowMajor.product_covers(images,plane,plane_positive),positions_room),
    FullRange.roundtrip(positions,Machine.capacity(Groups.domain_limits(geometry,domain)),positions_room),
    Equal.trans(Nat,U32.to_nat(U32.from_nat(Nat.div(Nat.add(U32.to_nat(o),7n),8n))),Nat.div(Nat.add(U32.to_nat(o),7n),8n),blocks,
      FullRange.roundtrip(Nat.div(Nat.add(U32.to_nat(o),7n),8n),Machine.capacity(Groups.domain_limits(geometry,domain)),
        %Equal.sym(Nat,Nat.div(Nat.add(U32.to_nat(o),7n),8n),blocks,groups_exact) : {Nat.is_le(_,cap) == True{} : Bool} Groups.output_blocks_room(geometry,domain)),
      groups_exact),
    Equal.trans(Nat,U32.to_nat(U32.from_nat(work_cells)),work_cells,input_total,
      FullRange.roundtrip(work_cells,Machine.capacity(Groups.domain_limits(geometry,domain)),
        %Equal.sym(Nat,work_cells,input_total,cells_exact) : {Nat.is_le(_,cap) == True{} : Bool} input_room),
      cells_exact),
    %Equal.sym(U32,U32.from_nat(U32.to_nat(o)),o,Arithmetic.native_roundtrip(o)) : {U32.to_nat(((_ + 7) / 8 : U32)) == blocks : Nat}
    Encoded.eight_blocks_exact(Groups.domain_limits(geometry,domain),o,Groups.outputs_room(geometry,domain))}

# ---- The sizes of a call ----

# The sizes of a positive image.
type Sizes<-image: ParameterGradient.ImageGeometry> is Data:
  Sizes{count: {Nat.is_lt(0n,U32.to_nat(image_count(image))) == True{} : Bool},channels: {Nat.is_lt(0n,U32.to_nat(image_channels(image))) == True{} : Bool},
    height: {Nat.is_lt(0n,U32.to_nat(image_height(image))) == True{} : Bool},width: {Nat.is_lt(0n,U32.to_nat(image_width(image))) == True{} : Bool}}

def image_sizes(+image: ParameterGradient.ImageGeometry,+positive: {ParameterGradient.positive(image) == True{} : Bool}) -> Sizes<image>:
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = image
  +nch = Logic.left((U32.is_lt(0,n) && U32.is_lt(0,c)) && U32.is_lt(0,h),U32.is_lt(0,w),positive)
  +nc = Logic.left(U32.is_lt(0,n) && U32.is_lt(0,c),U32.is_lt(0,h),nch)
  Sizes{%Comparison.lt(0,n) : {_ == True{} : Bool} Logic.left(U32.is_lt(0,n),U32.is_lt(0,c),nc),
    %Comparison.lt(0,c) : {_ == True{} : Bool} Logic.right(U32.is_lt(0,n),U32.is_lt(0,c),nc),
    %Comparison.lt(0,h) : {_ == True{} : Bool} Logic.right(U32.is_lt(0,n) && U32.is_lt(0,c),U32.is_lt(0,h),nch),
    %Comparison.lt(0,w) : {_ == True{} : Bool} Logic.right((U32.is_lt(0,n) && U32.is_lt(0,c)) && U32.is_lt(0,h),U32.is_lt(0,w),positive)}

def positive_input(-xi: ParameterGradient.ImageGeometry,-di: ParameterGradient.ImageGeometry,-k: U32,-s: U32,-g: U32,-capacity: U32,+positive: Positive<xi,di,k,s,g,capacity>) ->
  {ParameterGradient.positive(xi) == True{} : Bool}:
  Positive{+input,gradient,kernel,stride,group,images,bands} = positive
  input

def positive_gradient(-xi: ParameterGradient.ImageGeometry,-di: ParameterGradient.ImageGeometry,-k: U32,-s: U32,-g: U32,-capacity: U32,+positive: Positive<xi,di,k,s,g,capacity>) ->
  {ParameterGradient.positive(di) == True{} : Bool}:
  Positive{input,+gradient,kernel,stride,group,images,bands} = positive
  gradient

def positive_group(-xi: ParameterGradient.ImageGeometry,-di: ParameterGradient.ImageGeometry,-k: U32,-s: U32,+g: U32,-capacity: U32,+positive: Positive<xi,di,k,s,g,capacity>) ->
  {Nat.is_lt(0n,U32.to_nat(g)) == True{} : Bool}:
  Positive{input,gradient,kernel,stride,+group,images,bands} = positive
  %Comparison.lt(0,g) : {_ == True{} : Bool} group

def positive_images(+xi: ParameterGradient.ImageGeometry,+di: ParameterGradient.ImageGeometry,-k: U32,-s: U32,-g: U32,-capacity: U32,+positive: Positive<xi,di,k,s,g,capacity>) ->
  {U32.to_nat(image_count(di)) == U32.to_nat(image_count(xi)) : Nat}:
  Positive{input,gradient,kernel,stride,group,+images,bands} = positive
  Equal.sym(Nat,U32.to_nat(image_count(xi)),U32.to_nat(image_count(di)),
    Order.equal_of_is_eq(U32.to_nat(image_count(xi)),U32.to_nat(image_count(di)),
      %Comparison.eq(image_count(xi),image_count(di)) : {_ == True{} : Bool} images))

# ---- The rooms of a call ----

# Cells of images of rows, columns and groups hold the groups at every row
# and column of every image.
def image_room(+count: Nat,+height: Nat,+width: Nat,+blocks: Nat,+total: Nat,+total_exact: {total == Nat.mul(count,Nat.mul(height,Nat.mul(width,blocks))) : Nat}) ->
  {Nat.is_le(Nat.mul(count,Nat.mul(Nat.mul(blocks,height),width)),total) == True{} : Bool}:
  +ordered = Nat.mul(count,Nat.mul(Nat.mul(blocks,height),width))
  +stored = Nat.mul(count,Nat.mul(height,Nat.mul(width,blocks)))
  +inner = Equal.trans(Nat,Nat.mul(Nat.mul(blocks,height),width),Nat.mul(blocks,Nat.mul(height,width)),Nat.mul(height,Nat.mul(width,blocks)),
    RowMajor.assoc(blocks,height,width),
    Equal.trans(Nat,Nat.mul(blocks,Nat.mul(height,width)),Nat.mul(height,Nat.mul(blocks,width)),Nat.mul(height,Nat.mul(width,blocks)),
      RowMajor.swap_factors(blocks,height,width),
      Equal.cong(Nat,Nat,value => Nat.mul(height,value),Nat.mul(blocks,width),Nat.mul(width,blocks),RowMajor.multiply_commutative(blocks,width))))
  +same = Equal.cong(Nat,Nat,value => Nat.mul(count,value),Nat.mul(Nat.mul(blocks,height),width),Nat.mul(height,Nat.mul(width,blocks)),inner)
  %Equal.sym(Nat,ordered,stored,same) : {Nat.is_le(_,total) == True{} : Bool}
  %Equal.sym(Nat,total,stored,total_exact) : {Nat.is_le(stored,_) == True{} : Bool}
  Order.reflexive(stored)

# The capacity of limits whose capacity is the default capacity is a multiple
# of eight.
def capacity_aligned(+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32}) ->
  {U32.is_zero((Machine.capacity(limits) % 8 : U32)) == True{} : Bool}:
  %Equal.sym(U32,Machine.capacity(limits),Machine.capacity(Default.limits()),default_capacity) : {U32.is_zero((_ % 8 : U32)) == True{} : Bool}
  {==}

# The bound of the count of the weights of a valid call.
def weights_bounded(+xi: ParameterGradient.ImageGeometry,+di: ParameterGradient.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+valid: {ParameterGradient.valid_with(xi,di,k,s,p,g,limits) == True{} : Bool}) ->
  {U32.is_le(Global.count(weight_shape(xi,di,k),valid_weights(xi,di,k,s,p,g,limits,valid)),Machine.capacity(limits)) == True{} : Bool}:
  +shape = weight_shape(xi,di,k)
  +counted = valid_weights(xi,di,k,s,p,g,limits,valid)
  %Equal.sym(U32,Machine.capacity(limits),Machine.capacity(Default.limits()),default_capacity) : {U32.is_le(Global.count(shape,counted),_) == True{} : Bool}
  Listed.count_bounded(ShapeLaws.elements(shape),Global.count(shape,counted),Global.evidence(shape,counted))

# The geometry of a valid call, ready for the kernel.
def call_domain(+xi: ParameterGradient.ImageGeometry,+di: ParameterGradient.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+valid: {ParameterGradient.valid_with(xi,di,k,s,p,g,limits) == True{} : Bool}) ->
  Groups.GeometryReady<call_geometry(xi,di,k,s,p)>:
  +shape = weight_shape(xi,di,k)
  +counted = valid_weights(xi,di,k,s,p,g,limits,valid)
  geometry_ready(limits,capacity_aligned(limits,default_capacity),xi,di,k,s,p,g,valid_positive(xi,di,k,s,p,g,limits,valid),Global.count(shape,counted),
    Count.count_exact(ShapeLaws.elements(shape),Global.count(shape,counted),Global.evidence(shape,counted)),weights_bounded(xi,di,k,s,p,g,limits,default_capacity,valid))

def limits_kept(+xi: ParameterGradient.ImageGeometry,+di: ParameterGradient.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+valid: {ParameterGradient.valid_with(xi,di,k,s,p,g,limits) == True{} : Bool}) ->
  {Groups.domain_limits(call_geometry(xi,di,k,s,p),call_domain(xi,di,k,s,p,g,limits,default_capacity,valid)) == limits : Machine.Limits}:
  +shape = weight_shape(xi,di,k)
  +counted = valid_weights(xi,di,k,s,p,g,limits,valid)
  ready_limits(limits,capacity_aligned(limits,default_capacity),xi,di,k,s,p,g,valid_positive(xi,di,k,s,p,g,limits,valid),Global.count(shape,counted),
    Count.count_exact(ShapeLaws.elements(shape),Global.count(shape,counted),Global.evidence(shape,counted)),weights_bounded(xi,di,k,s,p,g,limits,default_capacity,valid))

# A bound within the limits of a call is a bound within the limits of its
# geometry.
def domain_room(+xi: ParameterGradient.ImageGeometry,+di: ParameterGradient.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+valid: {ParameterGradient.valid_with(xi,di,k,s,p,g,limits) == True{} : Bool},
  +value: Nat,+room: {Nat.is_le(value,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(value,U32.to_nat(Machine.capacity(Groups.domain_limits(call_geometry(xi,di,k,s,p),call_domain(xi,di,k,s,p,g,limits,default_capacity,valid))))) == True{} : Bool}:
  %Equal.sym(Machine.Limits,Groups.domain_limits(call_geometry(xi,di,k,s,p),call_domain(xi,di,k,s,p,g,limits,default_capacity,valid)),limits,
      limits_kept(xi,di,k,s,p,g,limits,default_capacity,valid)) :
    {Nat.is_le(value,U32.to_nat(Machine.capacity(_))) == True{} : Bool}
  room

def sizes_count(-image: ParameterGradient.ImageGeometry,+sizes: Sizes<image>) -> {Nat.is_lt(0n,U32.to_nat(image_count(image))) == True{} : Bool}:
  Sizes{+count,channels,height,width} = sizes
  count

def sizes_height(-image: ParameterGradient.ImageGeometry,+sizes: Sizes<image>) -> {Nat.is_lt(0n,U32.to_nat(image_height(image))) == True{} : Bool}:
  Sizes{count,channels,+height,width} = sizes
  height

def sizes_width(-image: ParameterGradient.ImageGeometry,+sizes: Sizes<image>) -> {Nat.is_lt(0n,U32.to_nat(image_width(image))) == True{} : Bool}:
  Sizes{count,channels,height,+width} = sizes
  width

def rows_positive(+xi: ParameterGradient.ImageGeometry,+di: ParameterGradient.ImageGeometry,+k: U32,+s: U32,+p: U32,+sizes: Sizes<di>) ->
  {Nat.is_lt(0n,Groups.row_count(call_geometry(xi,di,k,s,p))) == True{} : Bool}:
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{+dn,+o,+rows,+columns} = di
  sizes_height(ParameterGradient.ImageGeometry{dn,o,rows,columns},sizes)

def columns_positive(+xi: ParameterGradient.ImageGeometry,+di: ParameterGradient.ImageGeometry,+k: U32,+s: U32,+p: U32,+sizes: Sizes<di>) ->
  {Nat.is_lt(0n,Groups.column_count(call_geometry(xi,di,k,s,p))) == True{} : Bool}:
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{+dn,+o,+rows,+columns} = di
  sizes_width(ParameterGradient.ImageGeometry{dn,o,rows,columns},sizes)

# ---- The cells a call reads ----

# The cells of an image whose channels fit the capacity hold the groups of
# its channels at every row and column of every image.
def cells_room(+shape: List<&2,U32>,+lanes: U32,+image: ParameterGradient.ImageGeometry,+cells: ImageCells<shape,lanes,image>,+limits: Machine.Limits,
  +bound: {Nat.is_le(U32.to_nat(image_channels(image)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(U32.to_nat(image_count(image)),Nat.mul(Nat.mul(Ceiling.ceiling(U32.to_nat(image_channels(image)),8n),U32.to_nat(image_height(image))),
    U32.to_nat(image_width(image)))),ShapeLaws.elements(shape)) == True{} : Bool}:
  ImageCells{+channels,+elements} = cells
  +n = U32.to_nat(image_count(image))
  +h = U32.to_nat(image_height(image))
  +w = U32.to_nat(image_width(image))
  +blocks = Ceiling.ceiling(U32.to_nat(lanes),8n)
  +lanes_bound = {%channels : {Nat.is_le(U32.to_nat(_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} bound :
    {Nat.is_le(U32.to_nat(lanes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +total_exact = Equal.trans(Nat,ShapeLaws.elements(shape),Nat.mul(n,Nat.mul(h,Nat.mul(w,U32.to_nat(Grid.groups(lanes))))),Nat.mul(n,Nat.mul(h,Nat.mul(w,blocks))),elements,
    Equal.cong(Nat,Nat,value => Nat.mul(n,Nat.mul(h,Nat.mul(w,value))),U32.to_nat(Grid.groups(lanes)),blocks,Encoded.eight_blocks_exact(limits,lanes,lanes_bound)))
  %Equal.sym(U32,image_channels(image),lanes,channels) : {Nat.is_le(Nat.mul(n,Nat.mul(Nat.mul(Ceiling.ceiling(U32.to_nat(_),8n),h),w)),ShapeLaws.elements(shape)) == True{} : Bool}
  image_room(n,h,w,blocks,ShapeLaws.elements(shape),total_exact)

# The cells of x hold the input of every image of a call.
def input_room(+xi: ParameterGradient.ImageGeometry,+di: ParameterGradient.ImageGeometry,+k: U32,+s: U32,+p: U32,+domain: Groups.GeometryReady<call_geometry(xi,di,k,s,p)>,
  +x_cells: List<&2,U32>,+x_lanes: U32,+cells: ImageCells<x_cells,x_lanes,xi>) ->
  {Nat.is_le(Nat.mul(U32.to_nat(image_count(xi)),Sources.input_cells(call_geometry(xi,di,k,s,p))),ShapeLaws.elements(x_cells)) == True{} : Bool}:
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{+dn,+o,+rows,+columns} = di
  +geometry = call_geometry(ParameterGradient.ImageGeometry{n,c,h,w},ParameterGradient.ImageGeometry{dn,o,rows,columns},k,s,p)
  cells_room(x_cells,x_lanes,ParameterGradient.ImageGeometry{n,c,h,w},cells,Groups.domain_limits(geometry,domain),Groups.channels_room(geometry,domain))

# The cells of d hold the output gradient of every image of a call.
def gradient_room(+xi: ParameterGradient.ImageGeometry,+di: ParameterGradient.ImageGeometry,+k: U32,+s: U32,+p: U32,+domain: Groups.GeometryReady<call_geometry(xi,di,k,s,p)>,
  +d_cells: List<&2,U32>,+d_lanes: U32,+cells: ImageCells<d_cells,d_lanes,di>,+images: {U32.to_nat(image_count(di)) == U32.to_nat(image_count(xi)) : Nat}) ->
  {Nat.is_le(Nat.mul(U32.to_nat(image_count(xi)),Groups.gradient_extent(call_geometry(xi,di,k,s,p))),ShapeLaws.elements(d_cells)) == True{} : Bool}:
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{+dn,+o,+rows,+columns} = di
  +geometry = call_geometry(ParameterGradient.ImageGeometry{n,c,h,w},ParameterGradient.ImageGeometry{dn,o,rows,columns},k,s,p)
  %images : {Nat.is_le(Nat.mul(_,Groups.gradient_extent(geometry)),ShapeLaws.elements(d_cells)) == True{} : Bool}
  cells_room(d_cells,d_lanes,ParameterGradient.ImageGeometry{dn,o,rows,columns},cells,Groups.domain_limits(geometry,domain),Groups.outputs_room(geometry,domain))

# The cells of x of a call are not empty.
def input_nonempty(+xi: ParameterGradient.ImageGeometry,+di: ParameterGradient.ImageGeometry,+k: U32,+s: U32,+p: U32,+domain: Groups.GeometryReady<call_geometry(xi,di,k,s,p)>,
  +x_cells: List<&2,U32>,+x_lanes: U32,+cells: ImageCells<x_cells,x_lanes,xi>,+sizes: Sizes<xi>) -> {Nat.is_lt(0n,ShapeLaws.elements(x_cells)) == True{} : Bool}:
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{+dn,+o,+rows,+columns} = di
  +x_image = {ParameterGradient.ImageGeometry{n,c,h,w} : ParameterGradient.ImageGeometry}
  +geometry = call_geometry(x_image,ParameterGradient.ImageGeometry{dn,o,rows,columns},k,s,p)
  +blocks = Ceiling.ceiling(U32.to_nat(c),8n)
  +plane = Order.product_positive(blocks,U32.to_nat(h),Groups.channel_blocks_positive(geometry,domain),sizes_height(x_image,sizes))
  +input = Order.product_positive(Nat.mul(blocks,U32.to_nat(h)),U32.to_nat(w),plane,sizes_width(x_image,sizes))
  +total = Order.product_positive(U32.to_nat(n),Sources.input_cells(geometry),sizes_count(x_image,sizes),input)
  Bounds.nat_trans(0n,Nat.mul(U32.to_nat(n),Sources.input_cells(geometry)),ShapeLaws.elements(x_cells),total,
    input_room(x_image,ParameterGradient.ImageGeometry{dn,o,rows,columns},k,s,p,domain,x_cells,x_lanes,cells))

# The cells of d of a call are not empty.
def gradient_nonempty(+xi: ParameterGradient.ImageGeometry,+di: ParameterGradient.ImageGeometry,+k: U32,+s: U32,+p: U32,+domain: Groups.GeometryReady<call_geometry(xi,di,k,s,p)>,
  +d_cells: List<&2,U32>,+d_lanes: U32,+cells: ImageCells<d_cells,d_lanes,di>,+images: {U32.to_nat(image_count(di)) == U32.to_nat(image_count(xi)) : Nat},
  +x_sizes: Sizes<xi>,+d_sizes: Sizes<di>) -> {Nat.is_lt(0n,ShapeLaws.elements(d_cells)) == True{} : Bool}:
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{+dn,+o,+rows,+columns} = di
  +x_image = {ParameterGradient.ImageGeometry{n,c,h,w} : ParameterGradient.ImageGeometry}
  +d_image = {ParameterGradient.ImageGeometry{dn,o,rows,columns} : ParameterGradient.ImageGeometry}
  +geometry = call_geometry(x_image,d_image,k,s,p)
  +blocks = Ceiling.ceiling(U32.to_nat(o),8n)
  +plane = Order.product_positive(blocks,U32.to_nat(rows),Groups.output_blocks_positive(geometry,domain),sizes_height(d_image,d_sizes))
  +extent = Order.product_positive(Nat.mul(blocks,U32.to_nat(rows)),U32.to_nat(columns),plane,sizes_width(d_image,d_sizes))
  +total = Order.product_positive(U32.to_nat(n),Groups.gradient_extent(geometry),sizes_count(x_image,x_sizes),extent)
  Bounds.nat_trans(0n,Nat.mul(U32.to_nat(n),Groups.gradient_extent(geometry)),ShapeLaws.elements(d_cells),total,
    gradient_room(x_image,d_image,k,s,p,domain,d_cells,d_lanes,cells,images))

# ---- A valid call on cells ----

# A call on cells x and d that name its images, valid within limits whose
# capacity is the default capacity.
type Call is Data:
  Call{limits: Machine.Limits,default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
    x_cells: List<&2,U32>,x_lanes: U32,xi: ParameterGradient.ImageGeometry,
    x_named: {ParameterGradient.cell_image(x_cells,x_lanes) == Some{xi} : Maybe<&2,ParameterGradient.ImageGeometry>},x_counted: Global.Counted<x_cells>,
    d_cells: List<&2,U32>,d_lanes: U32,di: ParameterGradient.ImageGeometry,
    d_named: {ParameterGradient.cell_image(d_cells,d_lanes) == Some{di} : Maybe<&2,ParameterGradient.ImageGeometry>},d_counted: Global.Counted<d_cells>,
    k: U32,s: U32,p: U32,g: U32,valid: {ParameterGradient.valid_with(xi,di,k,s,p,g,limits) == True{} : Bool}}

def limits_of(call: Call) -> Machine.Limits:
  Call{limits,default_capacity,x_cells,x_lanes,xi,x_named,x_counted,d_cells,d_lanes,di,d_named,d_counted,k,s,p,g,valid} = call
  limits

def x_cells_of(call: Call) -> List<&2,U32>:
  Call{limits,default_capacity,x_cells,x_lanes,xi,x_named,x_counted,d_cells,d_lanes,di,d_named,d_counted,k,s,p,g,valid} = call
  x_cells

def x_lanes_of(call: Call) -> U32:
  Call{limits,default_capacity,x_cells,x_lanes,xi,x_named,x_counted,d_cells,d_lanes,di,d_named,d_counted,k,s,p,g,valid} = call
  x_lanes

def x_image_of(call: Call) -> ParameterGradient.ImageGeometry:
  Call{limits,default_capacity,x_cells,x_lanes,xi,x_named,x_counted,d_cells,d_lanes,di,d_named,d_counted,k,s,p,g,valid} = call
  xi

def d_cells_of(call: Call) -> List<&2,U32>:
  Call{limits,default_capacity,x_cells,x_lanes,xi,x_named,x_counted,d_cells,d_lanes,di,d_named,d_counted,k,s,p,g,valid} = call
  d_cells

def d_lanes_of(call: Call) -> U32:
  Call{limits,default_capacity,x_cells,x_lanes,xi,x_named,x_counted,d_cells,d_lanes,di,d_named,d_counted,k,s,p,g,valid} = call
  d_lanes

def d_image_of(call: Call) -> ParameterGradient.ImageGeometry:
  Call{limits,default_capacity,x_cells,x_lanes,xi,x_named,x_counted,d_cells,d_lanes,di,d_named,d_counted,k,s,p,g,valid} = call
  di

def kernel_of(call: Call) -> U32:
  Call{limits,default_capacity,x_cells,x_lanes,xi,x_named,x_counted,d_cells,d_lanes,di,d_named,d_counted,k,s,p,g,valid} = call
  k

def stride_of(call: Call) -> U32:
  Call{limits,default_capacity,x_cells,x_lanes,xi,x_named,x_counted,d_cells,d_lanes,di,d_named,d_counted,k,s,p,g,valid} = call
  s

def padding_of(call: Call) -> U32:
  Call{limits,default_capacity,x_cells,x_lanes,xi,x_named,x_counted,d_cells,d_lanes,di,d_named,d_counted,k,s,p,g,valid} = call
  p

def group_of(call: Call) -> U32:
  Call{limits,default_capacity,x_cells,x_lanes,xi,x_named,x_counted,d_cells,d_lanes,di,d_named,d_counted,k,s,p,g,valid} = call
  g

# The work of a call.
def work_of(+call: Call) -> Parameter.Work:
  Call{limits,default_capacity,x_cells,x_lanes,+xi,x_named,x_counted,d_cells,d_lanes,+di,d_named,d_counted,+k,+s,+p,+g,valid} = call
  ParameterGradient.work(xi,di,k,s,p,g)

# The images a call reads from the cells of x and d.
def batch_of(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+x_context: X,+d_context: D,+call: Call) -> Images.Batch<X,xv,D,dv>:
  Call{+limits,+default_capacity,+x_cells,+x_lanes,+xi,+x_named,+x_counted,+d_cells,+d_lanes,+di,+d_named,+d_counted,+k,+s,+p,+g,+valid} = call
  +capacity = Machine.capacity(limits)
  +positive = valid_positive(xi,di,k,s,p,g,limits,valid)
  +domain = call_domain(xi,di,k,s,p,g,limits,default_capacity,valid)
  call_batch(~X,~xv,~D,~dv,x_context,d_context,xi,di,k,s,p,g,domain,
    columns_positive(xi,di,k,s,p,image_sizes(di,positive_gradient(xi,di,k,s,g,capacity,positive))),positive_group(xi,di,k,s,g,capacity,positive),
    x_cells,input_room(xi,di,k,s,p,domain,x_cells,x_lanes,image_cells(x_cells,x_lanes,xi,x_named)),
    domain_room(xi,di,k,s,p,g,limits,default_capacity,valid,ShapeLaws.elements(x_cells),StorageCount.counted_room(limits,x_cells,x_counted,default_capacity)),
    d_cells,gradient_room(xi,di,k,s,p,domain,d_cells,d_lanes,image_cells(d_cells,d_lanes,di,d_named),positive_images(xi,di,k,s,g,capacity,positive)),
    domain_room(xi,di,k,s,p,g,limits,default_capacity,valid,ShapeLaws.elements(d_cells),StorageCount.counted_room(limits,d_cells,d_counted,default_capacity)))

# The work of a call reads its batch.
def work_batch_of(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+x_context: X,+d_context: D,+call: Call) ->
  Routes.WorkBatch<X,xv,D,dv,batch_of(~X,~xv,~D,~dv,x_context,d_context,call),work_of(call)>:
  Call{+limits,+default_capacity,+x_cells,+x_lanes,+xi,+x_named,+x_counted,+d_cells,+d_lanes,+di,+d_named,+d_counted,+k,+s,+p,+g,+valid} = call
  +capacity = Machine.capacity(limits)
  +positive = valid_positive(xi,di,k,s,p,g,limits,valid)
  +domain = call_domain(xi,di,k,s,p,g,limits,default_capacity,valid)
  +sizes = image_sizes(di,positive_gradient(xi,di,k,s,g,capacity,positive))
  work_batch(~X,~xv,~D,~dv,x_context,d_context,xi,di,k,s,p,g,domain,rows_positive(xi,di,k,s,p,sizes),columns_positive(xi,di,k,s,p,sizes),
    positive_group(xi,di,k,s,g,capacity,positive),
    x_cells,input_room(xi,di,k,s,p,domain,x_cells,x_lanes,image_cells(x_cells,x_lanes,xi,x_named)),
    domain_room(xi,di,k,s,p,g,limits,default_capacity,valid,ShapeLaws.elements(x_cells),StorageCount.counted_room(limits,x_cells,x_counted,default_capacity)),
    d_cells,gradient_room(xi,di,k,s,p,domain,d_cells,d_lanes,image_cells(d_cells,d_lanes,di,d_named),positive_images(xi,di,k,s,g,capacity,positive)),
    domain_room(xi,di,k,s,p,g,limits,default_capacity,valid,ShapeLaws.elements(d_cells),StorageCount.counted_room(limits,d_cells,d_counted,default_capacity)))

# ---- The run of a call ----

# The pieces of x of a call cover the input of its batch.
def x_covered_of(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+x_context: X,+d_context: D,+call: Call,-xp: Pieces.Pieces(~Tile.Tile8),
  +xf: Tile.Tile8,+xw: U32,+shaped: Global.Shaped(~Tile.Tile8,~X,~xv,x_context,x_cells_of(call),Tensors.Tensor{xp,x_cells_of(call),xf,Tensors.Usable{},xw})) ->
  Held.Covered<Tile.Tile8,X,xv,Images.batch_input(~X,~xv,~D,~dv,batch_of(~X,~xv,~D,~dv,x_context,d_context,call)),Move.held(~Tile.Tile8,xp),0n,
    Images.batch_x_high(~X,~xv,~D,~dv,batch_of(~X,~xv,~D,~dv,x_context,d_context,call)),Images.batch_x_layout(~X,~xv,~D,~dv,batch_of(~X,~xv,~D,~dv,x_context,d_context,call))>:
  Call{+limits,+default_capacity,+x_cells,+x_lanes,+xi,+x_named,+x_counted,+d_cells,+d_lanes,+di,+d_named,+d_counted,+k,+s,+p,+g,+valid} = call
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{+dn,+o,+rows,+columns} = di
  Windows.shaped_covered(~Tile.Tile8,~X,~xv,x_context,x_cells,xp,xf,Tensors.Usable{},xw,shaped)

# The pieces of d of a call cover the output gradient of its batch.
def d_covered_of(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+x_context: X,+d_context: D,+call: Call,-dp: Pieces.Pieces(~Tile.Tile8),
  +df: Tile.Tile8,+dw: U32,+shaped: Global.Shaped(~Tile.Tile8,~D,~dv,d_context,d_cells_of(call),Tensors.Tensor{dp,d_cells_of(call),df,Tensors.Usable{},dw})) ->
  Held.Covered<Tile.Tile8,D,dv,Images.batch_gradient(~X,~xv,~D,~dv,batch_of(~X,~xv,~D,~dv,x_context,d_context,call)),Move.held(~Tile.Tile8,dp),0n,
    Images.batch_d_high(~X,~xv,~D,~dv,batch_of(~X,~xv,~D,~dv,x_context,d_context,call)),Images.batch_d_layout(~X,~xv,~D,~dv,batch_of(~X,~xv,~D,~dv,x_context,d_context,call))>:
  Call{+limits,+default_capacity,+x_cells,+x_lanes,+xi,+x_named,+x_counted,+d_cells,+d_lanes,+di,+d_named,+d_counted,+k,+s,+p,+g,+valid} = call
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{+dn,+o,+rows,+columns} = di
  Windows.shaped_covered(~Tile.Tile8,~D,~dv,d_context,d_cells,dp,df,Tensors.Usable{},dw,shaped)

# The kernel run of a call on the pieces of x and d gives back holdings that
# cover them and the cells of the sums of the images.
def weight_cells(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+x_context: X,+d_context: D,+call: Call,+workers: U32,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(Images.batch_geometry(~X,~xv,~D,~dv,batch_of(~X,~xv,~D,~dv,x_context,d_context,call)))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(Images.batch_geometry(~X,~xv,~D,~dv,batch_of(~X,~xv,~D,~dv,x_context,d_context,call)))) == True{} : Bool},
  -xp: Pieces.Pieces(~Tile.Tile8),+xf: Tile.Tile8,+xw: U32,
  +x_shaped: Global.Shaped(~Tile.Tile8,~X,~xv,x_context,x_cells_of(call),Tensors.Tensor{xp,x_cells_of(call),xf,Tensors.Usable{},xw}),
  -dp: Pieces.Pieces(~Tile.Tile8),+df: Tile.Tile8,+dw: U32,
  +d_shaped: Global.Shaped(~Tile.Tile8,~D,~dv,d_context,d_cells_of(call),Tensors.Tensor{dp,d_cells_of(call),df,Tensors.Usable{},dw})) ->
  Routes.WeightCells<X,xv,D,dv,batch_of(~X,~xv,~D,~dv,x_context,d_context,call),tap,output,
    Kernel.weight_parts(work_of(call),workers,Move.held(~Tile.Tile8,xp),Move.held(~Tile.Tile8,dp))>:
  Routes.weight_parts(~X,~xv,~D,~dv,work_of(call),workers,batch_of(~X,~xv,~D,~dv,x_context,d_context,call),work_batch_of(~X,~xv,~D,~dv,x_context,d_context,call),
    tap,output,tap_inside,output_inside,Move.held(~Tile.Tile8,xp),Move.held(~Tile.Tile8,dp),
    x_covered_of(~X,~xv,~D,~dv,x_context,d_context,call,xp,xf,xw,x_shaped),d_covered_of(~X,~xv,~D,~dv,x_context,d_context,call,dp,df,dw,d_shaped))

# ---- What a run gives back ----

# The input a run gives back, placed in its grid, is a tensor of the contract
# of x.
def x_back(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+x_context: X,+d_context: D,+call: Call,+tap: U32,+output: U32,
  -result: Kernel.Held() & Storage.Buffer<Tile.Tile8>,+cells: Routes.WeightCells<X,xv,D,dv,batch_of(~X,~xv,~D,~dv,x_context,d_context,call),tap,output,result>,
  +xf: Tile.Tile8,+xw: U32) ->
  Global.Contract(~Tile.Tile8,~X,~xv,x_context,ParameterGradient.back(ParameterGradient.Read{x_cells_of(call),xf,Tensors.Usable{},xw},
    Pair.fst(Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>,Pair.fst(Kernel.Held(),Storage.Buffer<Tile.Tile8>,result)))):
  Call{+limits,+default_capacity,+x_cells,+x_lanes,+xi,+x_named,+x_counted,+d_cells,+d_lanes,+di,+d_named,+d_counted,+k,+s,+p,+g,+valid} = call
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{+dn,+o,+rows,+columns} = di
  Routes.WeightCells{list,contents,length,+x_covered,d_covered,lane} = cells
  +x_image = {ParameterGradient.ImageGeometry{n,c,h,w} : ParameterGradient.ImageGeometry}
  +d_image = {ParameterGradient.ImageGeometry{dn,o,rows,columns} : ParameterGradient.ImageGeometry}
  +capacity = Machine.capacity(limits)
  +positive = valid_positive(x_image,d_image,k,s,p,g,limits,valid)
  +domain = call_domain(x_image,d_image,k,s,p,g,limits,default_capacity,valid)
  Released.released_counted(~Tile.Tile8,~X,~xv,x_context,limits,x_cells,xf,xw,
    Pair.fst(Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>,Pair.fst(Kernel.Held(),Storage.Buffer<Tile.Tile8>,result)),x_counted,
    StorageCount.counted_room(limits,x_cells,x_counted,default_capacity),
    input_nonempty(x_image,d_image,k,s,p,domain,x_cells,x_lanes,image_cells(x_cells,x_lanes,x_image,x_named),
      image_sizes(x_image,positive_input(x_image,d_image,k,s,g,capacity,positive))),
    x_covered)

# The output gradient a run gives back, placed in its grid, is a tensor of the
# contract of d.
def d_back(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+x_context: X,+d_context: D,+call: Call,+tap: U32,+output: U32,
  -result: Kernel.Held() & Storage.Buffer<Tile.Tile8>,+cells: Routes.WeightCells<X,xv,D,dv,batch_of(~X,~xv,~D,~dv,x_context,d_context,call),tap,output,result>,
  +df: Tile.Tile8,+dw: U32) ->
  Global.Contract(~Tile.Tile8,~D,~dv,d_context,ParameterGradient.back(ParameterGradient.Read{d_cells_of(call),df,Tensors.Usable{},dw},
    Pair.snd(Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>,Pair.fst(Kernel.Held(),Storage.Buffer<Tile.Tile8>,result)))):
  Call{+limits,+default_capacity,+x_cells,+x_lanes,+xi,+x_named,+x_counted,+d_cells,+d_lanes,+di,+d_named,+d_counted,+k,+s,+p,+g,+valid} = call
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{+dn,+o,+rows,+columns} = di
  Routes.WeightCells{list,contents,length,x_covered,+d_covered,lane} = cells
  +x_image = {ParameterGradient.ImageGeometry{n,c,h,w} : ParameterGradient.ImageGeometry}
  +d_image = {ParameterGradient.ImageGeometry{dn,o,rows,columns} : ParameterGradient.ImageGeometry}
  +capacity = Machine.capacity(limits)
  +positive = valid_positive(x_image,d_image,k,s,p,g,limits,valid)
  +domain = call_domain(x_image,d_image,k,s,p,g,limits,default_capacity,valid)
  Released.released_counted(~Tile.Tile8,~D,~dv,d_context,limits,d_cells,df,dw,
    Pair.snd(Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>,Pair.fst(Kernel.Held(),Storage.Buffer<Tile.Tile8>,result)),d_counted,
    StorageCount.counted_room(limits,d_cells,d_counted,default_capacity),
    gradient_nonempty(x_image,d_image,k,s,p,domain,d_cells,d_lanes,image_cells(d_cells,d_lanes,d_image,d_named),
      positive_images(x_image,d_image,k,s,g,capacity,positive),image_sizes(x_image,positive_input(x_image,d_image,k,s,g,capacity,positive)),
      image_sizes(d_image,positive_gradient(x_image,d_image,k,s,g,capacity,positive))),
    d_covered)

# The cells of the sums of a run are ready.
def values_ready(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+x_context: X,+d_context: D,+call: Call,+tap: U32,+output: U32,
  -result: Kernel.Held() & Storage.Buffer<Tile.Tile8>,+cells: Routes.WeightCells<X,xv,D,dv,batch_of(~X,~xv,~D,~dv,x_context,d_context,call),tap,output,result>) ->
  Buffer.ContentReady<Tile.Tile8,Buffer.contents(~Tile.Tile8,Pair.snd(Kernel.Held(),Storage.Buffer<Tile.Tile8>,result))>:
  Call{+limits,+default_capacity,+x_cells,+x_lanes,+xi,+x_named,+x_counted,+d_cells,+d_lanes,+di,+d_named,+d_counted,+k,+s,+p,+g,+valid} = call
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{+dn,+o,+rows,+columns} = di
  Routes.WeightCells{+list,+contents,length,x_covered,d_covered,lane} = cells
  +x_image = {ParameterGradient.ImageGeometry{n,c,h,w} : ParameterGradient.ImageGeometry}
  +d_image = {ParameterGradient.ImageGeometry{dn,o,rows,columns} : ParameterGradient.ImageGeometry}
  +geometry = call_geometry(x_image,d_image,k,s,p)
  +domain = call_domain(x_image,d_image,k,s,p,g,limits,default_capacity,valid)
  +count = Kernel.cells(geometry)
  +room = {%Equal.sym(Nat,U32.to_nat(count),Nat.mul(Groups.output_blocks(geometry),Groups.tap_count(geometry)),Routes.cells_count(geometry,domain)) :
      {Nat.is_le(_,U32.to_nat(Machine.capacity(Groups.domain_limits(geometry,domain)))) == True{} : Bool}
    Groups.cells_room(geometry,domain) : {Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(Groups.domain_limits(geometry,domain)))) == True{} : Bool}}
  +within = {%default_capacity : {U32.is_le(count,_) == True{} : Bool}
    %Equal.sym(Bool,U32.is_le(count,Machine.capacity(limits)),Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(limits))),Comparison.le(count,Machine.capacity(limits))) :
      {_ == True{} : Bool}
    %limits_kept(x_image,d_image,k,s,p,g,limits,default_capacity,valid) : {Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(_))) == True{} : Bool}
    room : {U32.is_le(count,Machine.capacity(Default.limits())) == True{} : Bool}}
  %Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,Pair.snd(Kernel.Held(),Storage.Buffer<Tile.Tile8>,result)),
      Buffer.Content{Tree.write_sequence(~Tile.Tile8,list,0,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0))),U32.to_nat(count)},contents) :
    Buffer.ContentReady<Tile.Tile8,_>
  Listed.sequence_ready(~Tile.Tile8,list,count,Tile.filled(0.0),within)

# The cells of the sums of a run are the cells of the geometry of the call.
def values_length(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+x_context: X,+d_context: D,+call: Call,+tap: U32,+output: U32,
  -result: Kernel.Held() & Storage.Buffer<Tile.Tile8>,+cells: Routes.WeightCells<X,xv,D,dv,batch_of(~X,~xv,~D,~dv,x_context,d_context,call),tap,output,result>) ->
  {Buffer.content_length(~Tile.Tile8,Buffer.contents(~Tile.Tile8,Pair.snd(Kernel.Held(),Storage.Buffer<Tile.Tile8>,result))) ==
    U32.to_nat(Kernel.cells(call_geometry(x_image_of(call),d_image_of(call),kernel_of(call),stride_of(call),padding_of(call)))) : Nat}:
  Call{+limits,+default_capacity,+x_cells,+x_lanes,+xi,+x_named,+x_counted,+d_cells,+d_lanes,+di,+d_named,+d_counted,+k,+s,+p,+g,+valid} = call
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{+dn,+o,+rows,+columns} = di
  Routes.WeightCells{+list,+contents,length,x_covered,d_covered,lane} = cells
  +count = Kernel.cells(call_geometry(ParameterGradient.ImageGeometry{n,c,h,w},ParameterGradient.ImageGeometry{dn,o,rows,columns},k,s,p))
  %Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,Pair.snd(Kernel.Held(),Storage.Buffer<Tile.Tile8>,result)),
      Buffer.Content{Tree.write_sequence(~Tile.Tile8,list,0,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0))),U32.to_nat(count)},contents) :
    {Buffer.content_length(~Tile.Tile8,_) == U32.to_nat(count) : Nat}
  {==}

# ---- The runtime of a call ----

# A call valid within limits is valid within limits of the same capacity.
def valid_capacity(+xi: ParameterGradient.ImageGeometry,+di: ParameterGradient.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,+limits: Machine.Limits,+other: Machine.Limits,
  +same: {Machine.capacity(limits) == Machine.capacity(other) : U32}) ->
  {ParameterGradient.valid_with(xi,di,k,s,p,g,limits) == ParameterGradient.valid_with(xi,di,k,s,p,g,other) : Bool}:
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{+dn,+o,+rows,+columns} = di
  +x_image = {ParameterGradient.ImageGeometry{n,c,h,w} : ParameterGradient.ImageGeometry}
  +d_image = {ParameterGradient.ImageGeometry{dn,o,rows,columns} : ParameterGradient.ImageGeometry}
  %same : {ParameterGradient.valid_with(x_image,d_image,k,s,p,g,limits) ==
    conditions(x_image,d_image,k,s,p,g) && TensorView.has_count(Tensors.count_of([o,c,k,k])) && ParameterGradient.packed_fits(x_image,d_image,k,s,_) : Bool}
  {==}

# A call valid within limits of the default capacity is valid.
def valid_default(+xi: ParameterGradient.ImageGeometry,+di: ParameterGradient.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+valid: {ParameterGradient.valid_with(xi,di,k,s,p,g,limits) == True{} : Bool}) ->
  {ParameterGradient.valid(xi,di,k,s,p,g) == True{} : Bool}:
  Equal.trans(Bool,ParameterGradient.valid_with(xi,di,k,s,p,g,Default.limits()),ParameterGradient.valid_with(xi,di,k,s,p,g,limits),True{},
    Equal.sym(Bool,ParameterGradient.valid_with(xi,di,k,s,p,g,limits),ParameterGradient.valid_with(xi,di,k,s,p,g,Default.limits()),
      valid_capacity(xi,di,k,s,p,g,limits,Default.limits(),default_capacity)),
    valid)

# An accepted call finishes the run of its work.
def selected_exact(+xi: ParameterGradient.ImageGeometry,+di: ParameterGradient.ImageGeometry,+k: U32,+s: U32,+p: U32,+g: U32,xr: ParameterGradient.Read,
  -x: Move.Holding<Tile.Tile8>,dr: ParameterGradient.Read,-d: Move.Holding<Tile.Tile8>,+workers: U32,+xs: List<&2,U32>) ->
  {ParameterGradient.weight_selected(True{},xi,di,k,s,p,g,xr,x,dr,d,workers,xs) ==
    ParameterGradient.weight_finished(xr,dr,workers,image_channels(di),(image_channels(xi) * k * k : U32),[Tiled.groups(image_channels(di)),image_channels(xi),k,k],
      Kernel.weight_parts(ParameterGradient.work(xi,di,k,s,p,g),workers,x,d)) : Tiled.Tiled & (Tensors.Tensor<Tile.Tile8> & Tensors.Tensor<Tile.Tile8>)}:
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{+dn,+o,+rows,+columns} = di
  {==}

# The weight gradient of x and d of a valid call finishes the run of its work
# on the pieces of x and d.
def run_exact(+call: Call,-xp: Pieces.Pieces(~Tile.Tile8),+xf: Tile.Tile8,+xw: U32,-dp: Pieces.Pieces(~Tile.Tile8),+df: Tile.Tile8,+dw: U32) ->
  {ParameterGradient.weight(Tensors.Tensor{xp,x_cells_of(call),xf,Tensors.Usable{},xw},x_lanes_of(call),Tensors.Tensor{dp,d_cells_of(call),df,Tensors.Usable{},dw},d_lanes_of(call),
      kernel_of(call),stride_of(call),padding_of(call),group_of(call)) ==
    ParameterGradient.weight_finished(ParameterGradient.Read{x_cells_of(call),xf,Tensors.Usable{},xw},ParameterGradient.Read{d_cells_of(call),df,Tensors.Usable{},dw},
      U32.max(xw,dw),image_channels(d_image_of(call)),(image_channels(x_image_of(call)) * kernel_of(call) * kernel_of(call) : U32),
      [Tiled.groups(image_channels(d_image_of(call))),image_channels(x_image_of(call)),kernel_of(call),kernel_of(call)],
      Kernel.weight_parts(work_of(call),U32.max(xw,dw),Move.held(~Tile.Tile8,xp),Move.held(~Tile.Tile8,dp))) :
    Tiled.Tiled & (Tensors.Tensor<Tile.Tile8> & Tensors.Tensor<Tile.Tile8>)}:
  Call{+limits,+default_capacity,+x_cells,+x_lanes,+xi,+x_named,+x_counted,+d_cells,+d_lanes,+di,+d_named,+d_counted,+k,+s,+p,+g,+valid} = call
  %Equal.sym(Maybe<&2,ParameterGradient.ImageGeometry>,ParameterGradient.cell_image(x_cells,x_lanes),Some{xi},x_named) :
    {ParameterGradient.weight_shaped(_,ParameterGradient.cell_image(d_cells,d_lanes),True{},k,s,p,g,ParameterGradient.Read{x_cells,xf,Tensors.Usable{},xw},Move.held(~Tile.Tile8,xp),
        ParameterGradient.Read{d_cells,df,Tensors.Usable{},dw},Move.held(~Tile.Tile8,dp),U32.max(xw,dw),x_cells) ==
      ParameterGradient.weight_finished(ParameterGradient.Read{x_cells,xf,Tensors.Usable{},xw},ParameterGradient.Read{d_cells,df,Tensors.Usable{},dw},
        U32.max(xw,dw),image_channels(di),(image_channels(xi) * k * k : U32),[Tiled.groups(image_channels(di)),image_channels(xi),k,k],
        Kernel.weight_parts(ParameterGradient.work(xi,di,k,s,p,g),U32.max(xw,dw),Move.held(~Tile.Tile8,xp),Move.held(~Tile.Tile8,dp))) :
      Tiled.Tiled & (Tensors.Tensor<Tile.Tile8> & Tensors.Tensor<Tile.Tile8>)}
  %Equal.sym(Maybe<&2,ParameterGradient.ImageGeometry>,ParameterGradient.cell_image(d_cells,d_lanes),Some{di},d_named) :
    {ParameterGradient.weight_shaped(Some{xi},_,True{},k,s,p,g,ParameterGradient.Read{x_cells,xf,Tensors.Usable{},xw},Move.held(~Tile.Tile8,xp),
        ParameterGradient.Read{d_cells,df,Tensors.Usable{},dw},Move.held(~Tile.Tile8,dp),U32.max(xw,dw),x_cells) ==
      ParameterGradient.weight_finished(ParameterGradient.Read{x_cells,xf,Tensors.Usable{},xw},ParameterGradient.Read{d_cells,df,Tensors.Usable{},dw},
        U32.max(xw,dw),image_channels(di),(image_channels(xi) * k * k : U32),[Tiled.groups(image_channels(di)),image_channels(xi),k,k],
        Kernel.weight_parts(ParameterGradient.work(xi,di,k,s,p,g),U32.max(xw,dw),Move.held(~Tile.Tile8,xp),Move.held(~Tile.Tile8,dp))) :
      Tiled.Tiled & (Tensors.Tensor<Tile.Tile8> & Tensors.Tensor<Tile.Tile8>)}
  %Equal.sym(Bool,ParameterGradient.valid(xi,di,k,s,p,g),True{},valid_default(xi,di,k,s,p,g,limits,default_capacity,valid)) :
    {ParameterGradient.weight_selected(_,xi,di,k,s,p,g,ParameterGradient.Read{x_cells,xf,Tensors.Usable{},xw},Move.held(~Tile.Tile8,xp),
        ParameterGradient.Read{d_cells,df,Tensors.Usable{},dw},Move.held(~Tile.Tile8,dp),U32.max(xw,dw),x_cells) ==
      ParameterGradient.weight_finished(ParameterGradient.Read{x_cells,xf,Tensors.Usable{},xw},ParameterGradient.Read{d_cells,df,Tensors.Usable{},dw},
        U32.max(xw,dw),image_channels(di),(image_channels(xi) * k * k : U32),[Tiled.groups(image_channels(di)),image_channels(xi),k,k],
        Kernel.weight_parts(ParameterGradient.work(xi,di,k,s,p,g),U32.max(xw,dw),Move.held(~Tile.Tile8,xp),Move.held(~Tile.Tile8,dp))) :
      Tiled.Tiled & (Tensors.Tensor<Tile.Tile8> & Tensors.Tensor<Tile.Tile8>)}
  selected_exact(xi,di,k,s,p,g,ParameterGradient.Read{x_cells,xf,Tensors.Usable{},xw},Move.held(~Tile.Tile8,xp),ParameterGradient.Read{d_cells,df,Tensors.Usable{},dw},
    Move.held(~Tile.Tile8,dp),U32.max(xw,dw),x_cells)

# A finished run is the padded cells of its sums and the tensors it gives back.
def finished_exact(xr: ParameterGradient.Read,dr: ParameterGradient.Read,+workers: U32,+o: U32,+taps: U32,+cells: List<&2,U32>,result: Kernel.Held() & Storage.Buffer<Tile.Tile8>) ->
  {ParameterGradient.weight_finished(xr,dr,workers,o,taps,cells,result) ==
    (Tiled.Tiled{Tensors.whole(~Tile.Tile8,TiledMap.padded(TiledMap.Padding{o,Tiled.groups(o),taps,Grid.Outermost{}},0,Grid.product(cells),
        Pair.snd(Kernel.Held(),Storage.Buffer<Tile.Tile8>,result)),cells,Tiled.zero_tile(),Tensors.Usable{},workers),0n,o,Grid.Outermost{}},
      (ParameterGradient.back(xr,Pair.fst(Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>,Pair.fst(Kernel.Held(),Storage.Buffer<Tile.Tile8>,result))),
        ParameterGradient.back(dr,Pair.snd(Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>,Pair.fst(Kernel.Held(),Storage.Buffer<Tile.Tile8>,result))))) :
    Tiled.Tiled & (Tensors.Tensor<Tile.Tile8> & Tensors.Tensor<Tile.Tile8>)}:
  (held,values) = result
  (x,d) = held
  {==}

# ---- The weights a call writes ----

def default_capacity_of(+call: Call) -> {Machine.capacity(limits_of(call)) == Machine.capacity(Default.limits()) : U32}:
  Call{limits,+default_capacity,x_cells,x_lanes,xi,x_named,x_counted,d_cells,d_lanes,di,d_named,d_counted,k,s,p,g,valid} = call
  default_capacity

# The cells of the weights of a call: the groups of its outputs, its channels
# and its kernel rows and columns.
def result_shape(+call: Call) -> List<&2,U32>:
  [Tiled.groups(image_channels(d_image_of(call))),image_channels(x_image_of(call)),kernel_of(call),kernel_of(call)]

def result_taps(+call: Call) -> U32:
  (image_channels(x_image_of(call)) * kernel_of(call) * kernel_of(call) : U32)

# The outputs of a call in the lanes of its cells, the groups outermost.
def result_padding(+call: Call) -> TiledMap.Padding:
  TiledMap.Padding{image_channels(d_image_of(call)),Tiled.groups(image_channels(d_image_of(call))),result_taps(call),Grid.Outermost{}}

def call_cells(+call: Call) -> U32:
  Kernel.cells(call_geometry(x_image_of(call),d_image_of(call),kernel_of(call),stride_of(call),padding_of(call)))

# The cells of the weights are the cells the kernel writes.
def result_cells(+call: Call) -> {ShapeLaws.elements(result_shape(call)) == U32.to_nat(call_cells(call)) : Nat}:
  Call{+limits,+default_capacity,+x_cells,+x_lanes,+xi,+x_named,+x_counted,+d_cells,+d_lanes,+di,+d_named,+d_counted,+k,+s,+p,+g,+valid} = call
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{+dn,+o,+rows,+columns} = di
  +x_image = {ParameterGradient.ImageGeometry{n,c,h,w} : ParameterGradient.ImageGeometry}
  +d_image = {ParameterGradient.ImageGeometry{dn,o,rows,columns} : ParameterGradient.ImageGeometry}
  +geometry = call_geometry(x_image,d_image,k,s,p)
  +domain = call_domain(x_image,d_image,k,s,p,g,limits,default_capacity,valid)
  +taps = Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))
  +blocks = Ceiling.ceiling(U32.to_nat(o),8n)
  Equal.trans(Nat,ShapeLaws.elements([Tiled.groups(o),c,k,k]),Nat.mul(blocks,taps),U32.to_nat(Kernel.cells(geometry)),
    ShapeLaws.elements_four_exact(Tiled.groups(o),c,k,k,blocks,U32.to_nat(c),U32.to_nat(k),U32.to_nat(k),
      Encoded.eight_blocks_exact(Groups.domain_limits(geometry,domain),o,Groups.outputs_room(geometry,domain)),{==},{==},{==}),
    Equal.sym(Nat,U32.to_nat(Kernel.cells(geometry)),Nat.mul(blocks,taps),
      Equal.trans(Nat,U32.to_nat(Kernel.cells(geometry)),Nat.mul(Groups.output_blocks(geometry),Groups.tap_count(geometry)),Nat.mul(blocks,taps),
        Routes.cells_count(geometry,domain),
        Equal.cong(Nat,Nat,value => Nat.mul(blocks,value),Groups.tap_count(geometry),taps,Groups.taps_exact_of(geometry,domain)))))

# The cells the kernel writes fit the capacity.
def cells_bound(+call: Call) -> {Nat.is_le(U32.to_nat(call_cells(call)),U32.to_nat(Machine.capacity(limits_of(call)))) == True{} : Bool}:
  Call{+limits,+default_capacity,+x_cells,+x_lanes,+xi,+x_named,+x_counted,+d_cells,+d_lanes,+di,+d_named,+d_counted,+k,+s,+p,+g,+valid} = call
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{+dn,+o,+rows,+columns} = di
  +x_image = {ParameterGradient.ImageGeometry{n,c,h,w} : ParameterGradient.ImageGeometry}
  +d_image = {ParameterGradient.ImageGeometry{dn,o,rows,columns} : ParameterGradient.ImageGeometry}
  +geometry = call_geometry(x_image,d_image,k,s,p)
  +domain = call_domain(x_image,d_image,k,s,p,g,limits,default_capacity,valid)
  %limits_kept(x_image,d_image,k,s,p,g,limits,default_capacity,valid) : {Nat.is_le(U32.to_nat(Kernel.cells(geometry)),U32.to_nat(Machine.capacity(_))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(Kernel.cells(geometry)),Nat.mul(Groups.output_blocks(geometry),Groups.tap_count(geometry)),Routes.cells_count(geometry,domain)) :
    {Nat.is_le(_,U32.to_nat(Machine.capacity(Groups.domain_limits(geometry,domain)))) == True{} : Bool}
  Groups.cells_room(geometry,domain)

# The kernel writes some cells.
def cells_positive(+call: Call) -> {Nat.is_lt(0n,U32.to_nat(call_cells(call))) == True{} : Bool}:
  Call{+limits,+default_capacity,+x_cells,+x_lanes,+xi,+x_named,+x_counted,+d_cells,+d_lanes,+di,+d_named,+d_counted,+k,+s,+p,+g,+valid} = call
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{+dn,+o,+rows,+columns} = di
  +x_image = {ParameterGradient.ImageGeometry{n,c,h,w} : ParameterGradient.ImageGeometry}
  +d_image = {ParameterGradient.ImageGeometry{dn,o,rows,columns} : ParameterGradient.ImageGeometry}
  +geometry = call_geometry(x_image,d_image,k,s,p)
  +domain = call_domain(x_image,d_image,k,s,p,g,limits,default_capacity,valid)
  %Equal.sym(Nat,U32.to_nat(Kernel.cells(geometry)),Nat.mul(Groups.output_blocks(geometry),Groups.tap_count(geometry)),Routes.cells_count(geometry,domain)) :
    {Nat.is_lt(0n,_) == True{} : Bool}
  Order.product_positive(Groups.output_blocks(geometry),Groups.tap_count(geometry),Groups.output_blocks_positive(geometry,domain),Groups.taps_positive(geometry,domain))

# The groups of the outputs of a call are some groups and fit the capacity.
def blocks_positive(+call: Call) -> {Nat.is_lt(0n,U32.to_nat(Tiled.groups(image_channels(d_image_of(call))))) == True{} : Bool}:
  Call{+limits,+default_capacity,+x_cells,+x_lanes,+xi,+x_named,+x_counted,+d_cells,+d_lanes,+di,+d_named,+d_counted,+k,+s,+p,+g,+valid} = call
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{+dn,+o,+rows,+columns} = di
  +geometry = call_geometry(ParameterGradient.ImageGeometry{n,c,h,w},ParameterGradient.ImageGeometry{dn,o,rows,columns},k,s,p)
  +domain = call_domain(ParameterGradient.ImageGeometry{n,c,h,w},ParameterGradient.ImageGeometry{dn,o,rows,columns},k,s,p,g,limits,default_capacity,valid)
  %Equal.sym(Nat,U32.to_nat(Tiled.groups(o)),Ceiling.ceiling(U32.to_nat(o),8n),
      Encoded.eight_blocks_exact(Groups.domain_limits(geometry,domain),o,Groups.outputs_room(geometry,domain))) : {Nat.is_lt(0n,_) == True{} : Bool}
  Groups.output_blocks_positive(geometry,domain)

def blocks_room(+call: Call) -> {Nat.is_le(U32.to_nat(Tiled.groups(image_channels(d_image_of(call)))),U32.to_nat(Machine.capacity(limits_of(call)))) == True{} : Bool}:
  Call{+limits,+default_capacity,+x_cells,+x_lanes,+xi,+x_named,+x_counted,+d_cells,+d_lanes,+di,+d_named,+d_counted,+k,+s,+p,+g,+valid} = call
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{+dn,+o,+rows,+columns} = di
  +x_image = {ParameterGradient.ImageGeometry{n,c,h,w} : ParameterGradient.ImageGeometry}
  +d_image = {ParameterGradient.ImageGeometry{dn,o,rows,columns} : ParameterGradient.ImageGeometry}
  +geometry = call_geometry(x_image,d_image,k,s,p)
  +domain = call_domain(x_image,d_image,k,s,p,g,limits,default_capacity,valid)
  %limits_kept(x_image,d_image,k,s,p,g,limits,default_capacity,valid) : {Nat.is_le(U32.to_nat(Tiled.groups(o)),U32.to_nat(Machine.capacity(_))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(Tiled.groups(o)),Ceiling.ceiling(U32.to_nat(o),8n),
      Encoded.eight_blocks_exact(Groups.domain_limits(geometry,domain),o,Groups.outputs_room(geometry,domain))) :
    {Nat.is_le(_,U32.to_nat(Machine.capacity(Groups.domain_limits(geometry,domain)))) == True{} : Bool}
  Groups.output_blocks_room(geometry,domain)

# The word of channels times the square of a kernel within a capacity is that
# product.
def taps_word(+c: U32,+k: U32,+capacity: U32,+k_positive: {Nat.is_lt(0n,U32.to_nat(k)) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((c * k * k : U32)) == Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))) : Nat}:
  +cn = U32.to_nat(c)
  +kn = U32.to_nat(k)
  +cap = U32.to_nat(capacity)
  +pair_room = Order.transitive(Nat.mul(cn,kn),Nat.mul(cn,Nat.mul(kn,kn)),cap,RowMajor.mul_le_left(cn,kn,Nat.mul(kn,kn),RowMajor.product_covers(kn,kn,k_positive)),room)
  +pair = Routes.word_product(c,k,capacity,pair_room)
  +triple_room = {%Equal.sym(Nat,U32.to_nat((c * k : U32)),Nat.mul(cn,kn),pair) : {Nat.is_le(Nat.mul(_,kn),cap) == True{} : Bool}
    %Equal.sym(Nat,Nat.mul(Nat.mul(cn,kn),kn),Nat.mul(cn,Nat.mul(kn,kn)),RowMajor.assoc(cn,kn,kn)) : {Nat.is_le(_,cap) == True{} : Bool} room : {Nat.is_le(Nat.mul(U32.to_nat((c * k : U32)),kn),cap) == True{} : Bool}}
  Equal.trans(Nat,U32.to_nat((c * k * k : U32)),Nat.mul(U32.to_nat((c * k : U32)),kn),Nat.mul(cn,Nat.mul(kn,kn)),
    Routes.word_product((c * k : U32),k,capacity,triple_room),
    Equal.trans(Nat,Nat.mul(U32.to_nat((c * k : U32)),kn),Nat.mul(Nat.mul(cn,kn),kn),Nat.mul(cn,Nat.mul(kn,kn)),
      Equal.cong(Nat,Nat,value => Nat.mul(value,kn),U32.to_nat((c * k : U32)),Nat.mul(cn,kn),pair),RowMajor.assoc(cn,kn,kn)))

def taps_exact(+call: Call) -> {U32.to_nat(result_taps(call)) ==
  Nat.mul(U32.to_nat(image_channels(x_image_of(call))),Nat.mul(U32.to_nat(kernel_of(call)),U32.to_nat(kernel_of(call)))) : Nat}:
  Call{+limits,+default_capacity,+x_cells,+x_lanes,+xi,+x_named,+x_counted,+d_cells,+d_lanes,+di,+d_named,+d_counted,+k,+s,+p,+g,+valid} = call
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{+dn,+o,+rows,+columns} = di
  +geometry = call_geometry(ParameterGradient.ImageGeometry{n,c,h,w},ParameterGradient.ImageGeometry{dn,o,rows,columns},k,s,p)
  +domain = call_domain(ParameterGradient.ImageGeometry{n,c,h,w},ParameterGradient.ImageGeometry{dn,o,rows,columns},k,s,p,g,limits,default_capacity,valid)
  taps_word(c,k,Machine.capacity(Groups.domain_limits(geometry,domain)),Groups.kernel_positive(geometry,domain),
    %Groups.taps_exact_of(geometry,domain) : {Nat.is_le(_,U32.to_nat(Machine.capacity(Groups.domain_limits(geometry,domain)))) == True{} : Bool}
    Groups.taps_room(geometry,domain))

def result_room(+call: Call) -> {Nat.is_le(ShapeLaws.elements(result_shape(call)),U32.to_nat(Machine.capacity(limits_of(call)))) == True{} : Bool}:
  %Equal.sym(Nat,ShapeLaws.elements(result_shape(call)),U32.to_nat(call_cells(call)),result_cells(call)) :
    {Nat.is_le(_,U32.to_nat(Machine.capacity(limits_of(call)))) == True{} : Bool}
  cells_bound(call)

def result_nonempty(+call: Call) -> {Nat.is_lt(0n,ShapeLaws.elements(result_shape(call))) == True{} : Bool}:
  %Equal.sym(Nat,ShapeLaws.elements(result_shape(call)),U32.to_nat(call_cells(call)),result_cells(call)) : {Nat.is_lt(0n,_) == True{} : Bool}
  cells_positive(call)

def result_counted(+call: Call) -> Global.Counted<result_shape(call)>:
  StorageCount.counted_within(limits_of(call),result_shape(call),result_room(call),default_capacity_of(call))

# The word product of the cells of the weights counts them.
def result_product(+call: Call) -> {ShapeLaws.elements(result_shape(call)) == U32.to_nat(Grid.product(result_shape(call))) : Nat}:
  Equal.sym(Nat,U32.to_nat(Grid.product(result_shape(call))),ShapeLaws.elements(result_shape(call)),
    TiledPublic.product_exact(Machine.capacity(limits_of(call)),LimitBounds.capacity_fits(limits_of(call)),result_shape(call),result_room(call)))

# The groups of outputs of every tap fit the half word, and so does every
# cell of them.
def fits_from(+limits: Machine.Limits,+o: U32,+c: U32,+k: U32,+count: U32,+positive: {Nat.is_lt(0n,U32.to_nat(Tiled.groups(o))) == True{} : Bool},
  +blocks_room: {Nat.is_le(U32.to_nat(Tiled.groups(o)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +taps_exact: {U32.to_nat((c * k * k : U32)) == Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))) : Nat},
  +room: {Nat.is_le(ShapeLaws.elements([Tiled.groups(o),c,k,k]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +count_exact: {U32.to_nat(count) == ShapeLaws.elements([Tiled.groups(o),c,k,k]) : Nat}) ->
  Padding.padding_fits(Machine.half_word(limits),TiledMap.Padding{o,Tiled.groups(o),(c * k * k : U32),Grid.Outermost{}},0,count):
  +half = Machine.half_word(limits)
  +blocks = U32.to_nat(Tiled.groups(o))
  +total = ShapeLaws.elements([Tiled.groups(o),c,k,k])
  +within = Bounds.as_nat_le(Machine.capacity(limits),half,LimitBounds.capacity_within_half(limits))
  +total_half = Order.transitive(total,U32.to_nat(Machine.capacity(limits)),U32.to_nat(half),room,within)
  +total_exact = ShapeLaws.elements_four_exact(Tiled.groups(o),c,k,k,blocks,U32.to_nat(c),U32.to_nat(k),U32.to_nat(k),{==},{==},{==},{==})
  +product_half = {%Equal.sym(Nat,U32.to_nat((c * k * k : U32)),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))),taps_exact) :
      {Nat.is_le(Nat.mul(blocks,_),U32.to_nat(half)) == True{} : Bool}
    %total_exact : {Nat.is_le(_,U32.to_nat(half)) == True{} : Bool} total_half :
    {Nat.is_le(Nat.mul(blocks,U32.to_nat((c * k * k : U32))),U32.to_nat(half)) == True{} : Bool}}
  +span = {%Equal.sym(Nat,U32.to_nat(count),total,count_exact) : {Nat.is_le(_,U32.to_nat(half)) == True{} : Bool} total_half :
    {Nat.is_le(U32.to_nat(count),U32.to_nat(half)) == True{} : Bool}}
  PaddedCells.fits_when(U32.is_zero((o % 8 : U32)),half,blocks,U32.to_nat((c * k * k : U32)),0n,U32.to_nat(count),
    KernelGemm.Both{positive,KernelGemm.Both{LimitBounds.double_bound(limits,blocks,blocks_room),KernelGemm.Both{product_half,span}}})

def result_fits(+call: Call) -> Padding.padding_fits(Machine.half_word(limits_of(call)),result_padding(call),0,Grid.product(result_shape(call))):
  fits_from(limits_of(call),image_channels(d_image_of(call)),image_channels(x_image_of(call)),kernel_of(call),Grid.product(result_shape(call)),
    blocks_positive(call),blocks_room(call),taps_exact(call),result_room(call),
    Equal.sym(Nat,ShapeLaws.elements(result_shape(call)),U32.to_nat(Grid.product(result_shape(call))),result_product(call)))

# The cells of the sums of a run are as many as the cells of the weights.
def values_count(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+x_context: X,+d_context: D,+call: Call,+tap: U32,+output: U32,
  -result: Kernel.Held() & Storage.Buffer<Tile.Tile8>,+cells: Routes.WeightCells<X,xv,D,dv,batch_of(~X,~xv,~D,~dv,x_context,d_context,call),tap,output,result>) ->
  {Buffer.content_length(~Tile.Tile8,Buffer.contents(~Tile.Tile8,Pair.snd(Kernel.Held(),Storage.Buffer<Tile.Tile8>,result))) == U32.to_nat(Grid.product(result_shape(call))) : Nat}:
  Equal.trans(Nat,Buffer.content_length(~Tile.Tile8,Buffer.contents(~Tile.Tile8,Pair.snd(Kernel.Held(),Storage.Buffer<Tile.Tile8>,result))),U32.to_nat(call_cells(call)),
    U32.to_nat(Grid.product(result_shape(call))),values_length(~X,~xv,~D,~dv,x_context,d_context,call,tap,output,result,cells),
    Equal.trans(Nat,U32.to_nat(call_cells(call)),ShapeLaws.elements(result_shape(call)),U32.to_nat(Grid.product(result_shape(call))),
      Equal.sym(Nat,ShapeLaws.elements(result_shape(call)),U32.to_nat(call_cells(call)),result_cells(call)),result_product(call)))

# A cell of the sums of a run holds, in the lane of an output, the levels of
# what the images give that output at that tap.
def value_lane(~X: Data,~xv: X -> U32 -> Tile.Tile8,~D: Data,~dv: D -> U32 -> Tile.Tile8,+x_context: X,+d_context: D,+call: Call,+tap: U32,+output: U32,
  -result: Kernel.Held() & Storage.Buffer<Tile.Tile8>,+cells: Routes.WeightCells<X,xv,D,dv,batch_of(~X,~xv,~D,~dv,x_context,d_context,call),tap,output,result>,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(Images.batch_geometry(~X,~xv,~D,~dv,batch_of(~X,~xv,~D,~dv,x_context,d_context,call)))) == True{} : Bool},
  +output_inside: {U32.is_lt(output,Groups.geometry_outputs(Images.batch_geometry(~X,~xv,~D,~dv,batch_of(~X,~xv,~D,~dv,x_context,d_context,call)))) == True{} : Bool}) ->
  {TileLaws.select_lane(Nat.mod(U32.to_nat(output),8n),Buffer.value(~Tile.Tile8,
      U32.from_nat(Routes.cell_index(Images.batch_geometry(~X,~xv,~D,~dv,batch_of(~X,~xv,~D,~dv,x_context,d_context,call)),tap,output)),
      Buffer.contents(~Tile.Tile8,Pair.snd(Kernel.Held(),Storage.Buffer<Tile.Tile8>,result)))) ==
    Images.lane_levels(Images.batch_images(~X,~xv,~D,~dv,batch_of(~X,~xv,~D,~dv,x_context,d_context,call)),
      Images.held_images(~X,~xv,~D,~dv,Images.batch_images(~X,~xv,~D,~dv,batch_of(~X,~xv,~D,~dv,x_context,d_context,call)),0n,
        batch_of(~X,~xv,~D,~dv,x_context,d_context,call),tap,output)) : F32}:
  Call{+limits,+default_capacity,+x_cells,+x_lanes,+xi,+x_named,+x_counted,+d_cells,+d_lanes,+di,+d_named,+d_counted,+k,+s,+p,+g,+valid} = call
  ParameterGradient.ImageGeometry{+n,+c,+h,+w} = xi
  ParameterGradient.ImageGeometry{+dn,+o,+rows,+columns} = di
  Routes.WeightCells{+list,+contents,+length,x_covered,d_covered,+lane} = cells
  +x_image = {ParameterGradient.ImageGeometry{n,c,h,w} : ParameterGradient.ImageGeometry}
  +d_image = {ParameterGradient.ImageGeometry{dn,o,rows,columns} : ParameterGradient.ImageGeometry}
  +rebuilt = {Call{limits,default_capacity,x_cells,x_lanes,x_image,x_named,x_counted,d_cells,d_lanes,d_image,d_named,d_counted,k,s,p,g,valid} : Call}
  +batch = batch_of(~X,~xv,~D,~dv,x_context,d_context,rebuilt)
  +geometry = call_geometry(x_image,d_image,k,s,p)
  +domain = call_domain(x_image,d_image,k,s,p,g,limits,default_capacity,valid)
  +count = Kernel.cells(geometry)
  +index = Routes.cell_index(geometry,tap,output)
  +blocks_taps = Nat.mul(Groups.output_blocks(geometry),Groups.tap_count(geometry))
  +fill = Tile.filled(0.0)
  +sequence = {Buffer.Content{Tree.write_sequence(~Tile.Tile8,list,0,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),fill)),U32.to_nat(count)} : Buffer.Content<Tile.Tile8>}
  +counted = Routes.cells_count(geometry,domain)
  +equal = Equal.trans(Nat,List.length(&2,Tile.Tile8,list),blocks_taps,U32.to_nat(count),length,Equal.sym(Nat,U32.to_nat(count),blocks_taps,counted))
  +inside = {%Equal.sym(Nat,U32.to_nat(count),blocks_taps,counted) : {Nat.is_lt(index,_) == True{} : Bool}
    Routes.cell_inside(~X,~xv,~D,~dv,batch,tap,output,tap_inside,output_inside) : {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}}
  +room = {%Equal.sym(Nat,U32.to_nat(count),blocks_taps,counted) : {Nat.is_le(_,U32.to_nat(Machine.capacity(Groups.domain_limits(geometry,domain)))) == True{} : Bool}
    Groups.cells_room(geometry,domain) : {Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(Groups.domain_limits(geometry,domain)))) == True{} : Bool}}
  +within = {%default_capacity : {U32.is_le(count,_) == True{} : Bool}
    %Equal.sym(Bool,U32.is_le(count,Machine.capacity(limits)),Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(limits))),Comparison.le(count,Machine.capacity(limits))) :
      {_ == True{} : Bool}
    %limits_kept(x_image,d_image,k,s,p,g,limits,default_capacity,valid) : {Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(_))) == True{} : Bool}
    room : {U32.is_le(count,Machine.capacity(Default.limits())) == True{} : Bool}}
  +read = {%Listed.contents(~Tile.Tile8,list,Tensors.depth_for(count),fill,U32.to_nat(count)) :
      {Buffer.value(~Tile.Tile8,U32.from_nat(index),_) == ListRead.at(~Tile.Tile8,fill,index,list) : Tile.Tile8}
    Listed.written(~Tile.Tile8,list,count,fill,index,equal,inside,Depth.bounded(count,within)) :
    {Buffer.value(~Tile.Tile8,U32.from_nat(index),sequence) == ListRead.at(~Tile.Tile8,fill,index,list) : Tile.Tile8}}
  +levels = Images.lane_levels(Images.batch_images(~X,~xv,~D,~dv,batch),Images.held_images(~X,~xv,~D,~dv,Images.batch_images(~X,~xv,~D,~dv,batch),0n,batch,tap,output))
  %Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,Pair.snd(Kernel.Held(),Storage.Buffer<Tile.Tile8>,result)),sequence,contents) :
    {TileLaws.select_lane(Nat.mod(U32.to_nat(output),8n),Buffer.value(~Tile.Tile8,U32.from_nat(index),_)) == levels : F32}
  %Equal.sym(Tile.Tile8,Buffer.value(~Tile.Tile8,U32.from_nat(index),sequence),ListRead.at(~Tile.Tile8,fill,index,list),read) :
    {TileLaws.select_lane(Nat.mod(U32.to_nat(output),8n),_) == levels : F32}
  lane

# ---- The cells of an image ----

def sizes_channels(-image: ParameterGradient.ImageGeometry,+sizes: Sizes<image>) -> {Nat.is_lt(0n,U32.to_nat(image_channels(image))) == True{} : Bool}:
  Sizes{count,+channels,height,width} = sizes
  channels

def cells_channels(-shape: List<&2,U32>,-lanes: U32,-image: ParameterGradient.ImageGeometry,+cells: ImageCells<shape,lanes,image>) -> {image_channels(image) == lanes : U32}:
  ImageCells{+channels,elements} = cells
  channels

# The cells of a positive image of lanes within the capacity are not empty.
def cells_nonempty(+shape: List<&2,U32>,+lanes: U32,+image: ParameterGradient.ImageGeometry,+cells: ImageCells<shape,lanes,image>,+sizes: Sizes<image>,+limits: Machine.Limits,
  +lanes_room: {Nat.is_le(U32.to_nat(lanes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool}:
  ImageCells{+channels,+elements} = cells
  +blocks = Ceiling.ceiling(U32.to_nat(lanes),8n)
  +groups = U32.to_nat(Grid.groups(lanes))
  +w = U32.to_nat(image_width(image))
  +h = U32.to_nat(image_height(image))
  +lanes_positive = {%channels : {Nat.is_lt(0n,U32.to_nat(_)) == True{} : Bool} sizes_channels(image,sizes) : {Nat.is_lt(0n,U32.to_nat(lanes)) == True{} : Bool}}
  +blocks_positive = RowMajor.left_factor_positive(blocks,8n,
    Bounds.nat_trans(0n,U32.to_nat(lanes),Nat.mul(blocks,8n),lanes_positive,Ceiling.covers(U32.to_nat(lanes),8n,{==})))
  +groups_positive = {%Equal.sym(Nat,groups,blocks,Encoded.eight_blocks_exact(limits,lanes,lanes_room)) : {Nat.is_lt(0n,_) == True{} : Bool} blocks_positive :
    {Nat.is_lt(0n,groups) == True{} : Bool}}
  +inner = Order.product_positive(w,groups,sizes_width(image,sizes),groups_positive)
  +plane = Order.product_positive(h,Nat.mul(w,groups),sizes_height(image,sizes),inner)
  +total = Order.product_positive(U32.to_nat(image_count(image)),Nat.mul(h,Nat.mul(w,groups)),sizes_count(image,sizes),plane)
  %Equal.sym(Nat,ShapeLaws.elements(shape),Nat.mul(U32.to_nat(image_count(image)),Nat.mul(h,Nat.mul(w,groups))),elements) : {Nat.is_lt(0n,_) == True{} : Bool}
  total
