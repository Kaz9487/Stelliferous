# What one call of the pointwise kernel stores: at each output position of
# the call and each pair of output channels, the values of those channels.
import Base
import ../specification.bend as ConvSpec
import ../inference.bend as NativeConv
import ../../lib/machine_limits.bend as Machine
import ../../lib/kernel_tile.bend as Tile
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/kernel_window.bend as KernelWindow
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/nat_division.bend as Division
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/u32_division.bend as U32Division
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ./convolution_guard_facts.bend as Guards
import ./window_cell_contents.bend as Cells
import ./window_cell_values.bend as CellValues
import ./window_packed_values.bend as Calls
import ./reduction_coordinate_bounds.bend as Coordinates
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./window_positions.bend as Positions
import ../../lib/proofs/word_arithmetic.bend as WordArithmetic
import ../../lib/proofs/kernel_gemm_point_keys.bend as Keys
import ./window_contents.bend as Contents
import ./window_point_values.bend as PointValues
import ../../lib/proofs/kernel_gemm_point_spec.bend as PointSpec
import ../../lib/proofs/kernel_gemm_point_readback.bend as PointReadback
import ../../lib/proofs/kernel_gemm_point_jobs.bend as PointJobs
import ../../lib/proofs/kernel_gemm_point_buffer.bend as PointBuffer
import ../../lib/proofs/write_program.bend as Writes
import ../../lib/proofs/u32_shift.bend as U32Shift
import ./output_values.bend as Values
import ./input_storage.bend as Inputs
import ./window_packed_band.bend as Band
import ./window_skeleton.bend as Skeleton
import ./window_band_cover.bend as BandCover
import ./microkernel_leaf_readback.bend as MicroLeaf
import ../../lib/proofs/tensor_tiled_values.bend as TiledValues
import ./supported_coordinate_bounds.bend as SupportedCoordinates
import ../../lib/proofs/nat_ceiling.bend as Ceiling
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ./native_weight_refinement.bend as NativeWeightRefinement
import ../../lib/tensor.bend as Tensors
import ../../lib/proofs/tensor_allocation_depth.bend as Depth
import ../../lib/proofs/kernel_window_pointwise.bend as KernelPointwise
import ../../lib/kernel_window_pipeline.bend as WindowPipeline
import ./window_start.bend as Start

# ---- The geometry of a pointwise call ----

# An axis of a 1 x 1 kernel of stride 1 without padding keeps its size.
def unit_axis(+size: Nat,+positive: {Nat.is_lt(0n,size) == True{} : Bool}) -> {ConvSpec.axis_size(size,1n,1n,0n) == size : Nat}:
  match size:
    case 0n: Empty.absurd({ConvSpec.axis_size(0n,1n,1n,0n) == 0n : Nat},Order.below_zero(0n,positive))
    case 1n+ +rest:
      %Equal.sym(Nat,Nat.sub(Nat.add(rest,0n),0n),Nat.add(rest,0n),Order.subtract_zero(Nat.add(rest,0n))) : {1n+Nat.div(_,1n) == 1n+rest : Nat}
      %Equal.sym(Nat,Nat.add(rest,0n),rest,Addition.zero_right(rest)) : {1n+Nat.div(_,1n) == 1n+rest : Nat}
      %Order.times_one(rest) : {1n+Nat.div(_,1n) == 1n+rest : Nat}
      Equal.cong(Nat,Nat,value => 1n+value,Nat.div(Nat.mul(rest,1n),1n),rest,Division.exact_quotient(rest,1n,{==}))

# The output of a pointwise problem has the height and width of its input.
def point_width(+problem: ConvSpec.Problem,+kernel_one: {Guards.kernel(problem) == 1n : Nat},+stride_one: {Guards.stride(problem) == 1n : Nat},
  +pad_zero: {Guards.pad(problem) == 0n : Nat},+positive: {Nat.is_lt(0n,Guards.width(problem)) == True{} : Bool}) ->
  {ConvSpec.out_width(problem) == Guards.width(problem) : Nat}:
  ConvSpec.Problem{+shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  %Equal.sym(Nat,kernel,1n,kernel_one) : {ConvSpec.axis_size(width,_,stride,padding) == width : Nat}
  %Equal.sym(Nat,stride,1n,stride_one) : {ConvSpec.axis_size(width,1n,_,padding) == width : Nat}
  %Equal.sym(Nat,padding,0n,pad_zero) : {ConvSpec.axis_size(width,1n,1n,_) == width : Nat}
  unit_axis(width,positive)

def point_height(+problem: ConvSpec.Problem,+kernel_one: {Guards.kernel(problem) == 1n : Nat},+stride_one: {Guards.stride(problem) == 1n : Nat},
  +pad_zero: {Guards.pad(problem) == 0n : Nat},+positive: {Nat.is_lt(0n,Guards.height(problem)) == True{} : Bool}) ->
  {ConvSpec.out_height(problem) == Guards.height(problem) : Nat}:
  ConvSpec.Problem{+shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  %Equal.sym(Nat,kernel,1n,kernel_one) : {ConvSpec.axis_size(height,_,stride,padding) == height : Nat}
  %Equal.sym(Nat,stride,1n,stride_one) : {ConvSpec.axis_size(height,1n,_,padding) == height : Nat}
  %Equal.sym(Nat,padding,0n,pad_zero) : {ConvSpec.axis_size(height,1n,1n,_) == height : Nat}
  unit_axis(height,positive)

# The first position of a call of stride 1 lies its band's first row times
# the width after the first cell of the band.
def chosen_start(zero: Bool,+local: U32,+row_first: U32,+width: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +classified: {U32.is_zero(row_first) == zero : Bool},+reach: {Nat.is_le(Nat.mul(U32.to_nat(row_first),U32.to_nat(width)),U32.to_nat(local)) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(local),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat(Bool.pick(U32,zero,local,(local - row_first / 1 * width : U32))) == Nat.sub(U32.to_nat(local),Nat.mul(U32.to_nat(row_first),U32.to_nat(width))) : Nat}:
  match zero:
    case True{}:
      %Equal.sym(U32,row_first,0,Arithmetic.word_zero(row_first,classified)) :
        {U32.to_nat(local) == Nat.sub(U32.to_nat(local),Nat.mul(U32.to_nat(_),U32.to_nat(width))) : Nat}
      Equal.sym(Nat,Nat.sub(U32.to_nat(local),0n),U32.to_nat(local),Order.subtract_zero(U32.to_nat(local)))
    case False{}:
      +product = Nat.mul(U32.to_nat(row_first),U32.to_nat(width))
      +scaled = Arithmetic.product_exact(cap,cap_ok,(row_first / 1 : U32),width,U32.to_nat(row_first),U32.to_nat(width),U32Division.divided_by_one(row_first),{==},
        Order.transitive(product,U32.to_nat(local),U32.to_nat(cap),reach,room))
      Arithmetic.difference_exact(cap,cap_ok,local,(row_first / 1 * width : U32),U32.to_nat(local),product,{==},scaled,reach,room)

def start_word(+local: U32,+row_first: U32,+width: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +reach: {Nat.is_le(Nat.mul(U32.to_nat(row_first),U32.to_nat(width)),U32.to_nat(local)) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(local),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat(KernelWindow.start(local,1,width,row_first)) == Nat.sub(U32.to_nat(local),Nat.mul(U32.to_nat(row_first),U32.to_nat(width))) : Nat}:
  chosen_start(U32.is_zero(row_first),local,row_first,width,cap,cap_ok,{==},reach,room)

# ---- Where a position of a call lies in its band ----

def cover_first(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+offset: Nat,+span: Nat,+cover: Positions.Cover<problem,first,rows,offset,span>) ->
  {first == Nat.mul(Positions.block(problem,first,rows,offset,span,cover),Guards.stride(problem)) : Nat}:
  Positions.Cover{+block,+first_exact,+fits,+rows_positive,+kernel_fits,+top,+bottom} = cover
  first_exact

def cover_fits(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+offset: Nat,+span: Nat,+cover: Positions.Cover<problem,first,rows,offset,span>) ->
  {Nat.is_le(Nat.add(first,rows),Guards.height(problem)) == True{} : Bool}:
  Positions.Cover{+block,+first_exact,+fits,+rows_positive,+kernel_fits,+top,+bottom} = cover
  fits

def cover_rows(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+offset: Nat,+span: Nat,+cover: Positions.Cover<problem,first,rows,offset,span>) ->
  {Nat.is_lt(0n,rows) == True{} : Bool}:
  Positions.Cover{+block,+first_exact,+fits,+rows_positive,+kernel_fits,+top,+bottom} = cover
  rows_positive

# A pointwise axis of the problem keeps its size.
def problem_axis(+problem: ConvSpec.Problem,+size: Nat,+kernel_one: {Guards.kernel(problem) == 1n : Nat},+stride_one: {Guards.stride(problem) == 1n : Nat},
  +pad_zero: {Guards.pad(problem) == 0n : Nat},+positive: {Nat.is_lt(0n,size) == True{} : Bool}) ->
  {ConvSpec.axis_size(size,Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem)) == size : Nat}:
  %Equal.sym(Nat,Guards.kernel(problem),1n,kernel_one) : {ConvSpec.axis_size(size,_,Guards.stride(problem),Guards.pad(problem)) == size : Nat}
  %Equal.sym(Nat,Guards.stride(problem),1n,stride_one) : {ConvSpec.axis_size(size,1n,_,Guards.pad(problem)) == size : Nat}
  %Equal.sym(Nat,Guards.pad(problem),0n,pad_zero) : {ConvSpec.axis_size(size,1n,1n,_) == size : Nat}
  unit_axis(size,positive)

# Position local + index of a call whose band starts at row first: its cell in
# the band, its row and its column.
type BandPlace<-first: Nat,-rows: Nat,-local: Nat,-index: Nat,-width: Nat,-height: Nat> is Data:
  BandPlace{reach: {Nat.is_le(Nat.mul(first,width),local) == True{} : Bool},
    inside: {Nat.is_lt(Nat.add(Nat.sub(local,Nat.mul(first,width)),index),Nat.mul(rows,width)) == True{} : Bool},
    pixel: {Nat.add(Nat.mul(first,width),Nat.add(Nat.sub(local,Nat.mul(first,width)),index)) ==
      Nat.add(Nat.mul(Nat.div(Nat.add(local,index),width),width),Nat.mod(Nat.add(local,index),width)) : Nat},
    row_inside: {Nat.is_lt(Nat.div(Nat.add(local,index),width),height) == True{} : Bool},
    column_inside: {Nat.is_lt(Nat.mod(Nat.add(local,index),width),width) == True{} : Bool}}

def place(+first: Nat,+rows: Nat,+local: Nat,+index: Nat,+width: Nat,+height: Nat,+positive: {Nat.is_lt(0n,width) == True{} : Bool},
  +start_row: {Nat.is_le(first,Nat.div(local,width)) == True{} : Bool},+ordered: {Nat.is_le(first,Nat.div(Nat.add(local,index),width)) == True{} : Bool},
  +virtual: {Nat.is_lt(Nat.sub(Nat.div(Nat.add(local,index),width),first),rows) == True{} : Bool},
  +grid: {Nat.is_lt(Nat.add(local,index),Nat.mul(height,width)) == True{} : Bool}) -> BandPlace<first,rows,local,index,width,height>:
  +position = Nat.add(local,index)
  +skipped = Nat.mul(first,width)
  +reach = {Order.transitive(skipped,Nat.mul(Nat.div(local,width),width),local,RowMajor.mul_le(first,Nat.div(local,width),width,start_row),
      DivisionOrder.product_below(local,width,positive)) : {Nat.is_le(skipped,local) == True{} : Bool}}
  +window = Positions.window_position(local,index,width,first,positive,reach,ordered)
  BandPlace{reach,
    %Equal.sym(Nat,Nat.add(Nat.sub(local,skipped),index),Nat.add(Nat.mul(Nat.sub(Nat.div(position,width),first),width),Nat.mod(position,width)),window) :
      {Nat.is_lt(_,Nat.mul(rows,width)) == True{} : Bool}
    RowMajor.radix(Nat.sub(Nat.div(position,width),first),rows,Nat.mod(position,width),width,virtual,DivisionOrder.remainder_small(position,width,positive)),
    Equal.trans(Nat,Nat.add(skipped,Nat.add(Nat.sub(local,skipped),index)),Nat.add(Nat.add(skipped,Nat.sub(local,skipped)),index),
      Nat.add(Nat.mul(Nat.div(position,width),width),Nat.mod(position,width)),
      Equal.sym(Nat,Nat.add(Nat.add(skipped,Nat.sub(local,skipped)),index),Nat.add(skipped,Nat.add(Nat.sub(local,skipped),index)),
        Addition.associative(skipped,Nat.sub(local,skipped),index)),
      Equal.trans(Nat,Nat.add(Nat.add(skipped,Nat.sub(local,skipped)),index),position,Nat.add(Nat.mul(Nat.div(position,width),width),Nat.mod(position,width)),
        Equal.cong(Nat,Nat,value => Nat.add(value,index),Nat.add(skipped,Nat.sub(local,skipped)),local,Order.restore_shift(skipped,local,reach)),
        DivisionOrder.quotient_exact(position,width,positive))),
    DivisionOrder.quotient_below(position,height,width,positive,grid),
    DivisionOrder.remainder_small(position,width,positive)}

# Where position local + index of a pointwise call lies in the band of the
# call.
def band_place(+problem: ConvSpec.Problem,+checked: Guards.Valid<problem>,+kernel_one: {Guards.kernel(problem) == 1n : Nat},+stride_one: {Guards.stride(problem) == 1n : Nat},
  +pad_zero: {Guards.pad(problem) == 0n : Nat},+first: Nat,+rows: Nat,+local: Nat,+count: Nat,+cover: Positions.Cover<problem,first,rows,local,count>,
  +index: Nat,+inside: {Nat.is_lt(index,count) == True{} : Bool},
  +grid: {Nat.is_le(Nat.add(local,count),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool}) ->
  BandPlace<first,rows,local,index,Guards.width(problem),Guards.height(problem)>:
  Guards.Valid{+v0,+v1,+v2,+v3,+v4,+v5,+v6,+v7,+v8,+v9,+v10} = checked
  +outer = ConvSpec.out_width(problem)
  +height = Guards.height(problem)
  +width_same = point_width(problem,kernel_one,stride_one,pad_zero,v3)
  +height_same = point_height(problem,kernel_one,stride_one,pad_zero,v2)
  +positive = {%Equal.sym(Nat,outer,Guards.width(problem),width_same) : {Nat.is_lt(0n,_) == True{} : Bool} v3 : {Nat.is_lt(0n,outer) == True{} : Bool}}
  +position = Nat.add(local,index)
  +after = Order.le_add(local,index)
  +position_inside = Order.offset_lt(local,index,count,inside)
  +block = Positions.block(problem,first,rows,local,count,cover)
  +first_block = {Equal.trans(Nat,first,Nat.mul(block,Guards.stride(problem)),block,cover_first(problem,first,rows,local,count,cover),
      %Equal.sym(Nat,Guards.stride(problem),1n,stride_one) : {Nat.mul(block,_) == block : Nat} Order.times_one(block)) : {first == block : Nat}}
  +position_grid = {Order.lt_le(position,Nat.add(local,count),Nat.mul(height,outer),position_inside,
      %height_same : {Nat.is_le(Nat.add(local,count),Nat.mul(_,outer)) == True{} : Bool} grid) : {Nat.is_lt(position,Nat.mul(height,outer)) == True{} : Bool}}
  +row_valid = {%Equal.sym(Nat,ConvSpec.axis_size(height,Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem)),height,problem_axis(problem,height,kernel_one,stride_one,pad_zero,v2)) :
      {Nat.is_lt(Nat.div(position,outer),_) == True{} : Bool}
    DivisionOrder.quotient_below(position,height,outer,positive,position_grid) :
    {Nat.is_lt(Nat.div(position,outer),ConvSpec.axis_size(height,Guards.kernel(problem),Guards.stride(problem),Guards.pad(problem))) == True{} : Bool}}
  +ordered = Positions.block_le_row(problem,first,rows,local,count,cover,position,v5,positive,after)
  +start_row = Positions.block_le_row(problem,first,rows,local,count,cover,local,v5,positive,Order.reflexive(local))
  +virtual = {%problem_axis(problem,rows,kernel_one,stride_one,pad_zero,cover_rows(problem,first,rows,local,count,cover)) :
      {Nat.is_lt(Nat.sub(Nat.div(position,outer),block),_) == True{} : Bool}
    Positions.virtual_row_inside(problem,first,rows,local,count,cover,position,v5,positive,after,position_inside,row_valid) :
    {Nat.is_lt(Nat.sub(Nat.div(position,outer),block),rows) == True{} : Bool}}
  %width_same : BandPlace<first,rows,local,index,_,height>
  place(first,rows,local,index,outer,height,positive,
    %Equal.sym(Nat,first,block,first_block) : {Nat.is_le(_,Nat.div(local,outer)) == True{} : Bool} start_row,
    %Equal.sym(Nat,first,block,first_block) : {Nat.is_le(_,Nat.div(position,outer)) == True{} : Bool} ordered,
    %Equal.sym(Nat,first,block,first_block) : {Nat.is_lt(Nat.sub(Nat.div(position,outer),_),rows) == True{} : Bool} virtual,
    position_grid)

def place_reach(-first: Nat,-rows: Nat,-local: Nat,-index: Nat,-width: Nat,-height: Nat,placed: BandPlace<first,rows,local,index,width,height>) ->
  {Nat.is_le(Nat.mul(first,width),local) == True{} : Bool}:
  BandPlace{reach,inside,pixel,row_inside,column_inside} = placed
  reach

def place_inside(-first: Nat,-rows: Nat,-local: Nat,-index: Nat,-width: Nat,-height: Nat,placed: BandPlace<first,rows,local,index,width,height>) ->
  {Nat.is_lt(Nat.add(Nat.sub(local,Nat.mul(first,width)),index),Nat.mul(rows,width)) == True{} : Bool}:
  BandPlace{reach,inside,pixel,row_inside,column_inside} = placed
  inside

def place_pixel(-first: Nat,-rows: Nat,-local: Nat,-index: Nat,-width: Nat,-height: Nat,placed: BandPlace<first,rows,local,index,width,height>) ->
  {Nat.add(Nat.mul(first,width),Nat.add(Nat.sub(local,Nat.mul(first,width)),index)) ==
    Nat.add(Nat.mul(Nat.div(Nat.add(local,index),width),width),Nat.mod(Nat.add(local,index),width)) : Nat}:
  BandPlace{reach,inside,pixel,row_inside,column_inside} = placed
  pixel

def place_row(-first: Nat,-rows: Nat,-local: Nat,-index: Nat,-width: Nat,-height: Nat,placed: BandPlace<first,rows,local,index,width,height>) ->
  {Nat.is_lt(Nat.div(Nat.add(local,index),width),height) == True{} : Bool}:
  BandPlace{reach,inside,pixel,row_inside,column_inside} = placed
  row_inside

def place_column(-first: Nat,-rows: Nat,-local: Nat,-index: Nat,-width: Nat,-height: Nat,placed: BandPlace<first,rows,local,index,width,height>) ->
  {Nat.is_lt(Nat.mod(Nat.add(local,index),width),width) == True{} : Bool}:
  BandPlace{reach,inside,pixel,row_inside,column_inside} = placed
  column_inside

# ---- The cell a job of a call reads first ----

# The first cell of a job from a base and a shift is, as a word, the base plus
# the shift plus its units.
def cell_first(+base: U32,+shift: U32,+shift_value: Nat,+shift_exact: {U32.to_nat(shift) == shift_value : Nat},+unit: Nat,+groups: U32,+group_count: Nat,
  +groups_exact: {U32.to_nat(groups) == group_count : Nat}) ->
  {U32.from_nat(Keys.job_first((base + shift : U32),unit,groups)) == U32.from_nat(Nat.add(U32.to_nat(base),Nat.add(shift_value,Nat.mul(Nat.mul(unit,4n),group_count)))) : U32}:
  +units = Nat.mul(Nat.mul(unit,4n),group_count)
  +shifted = (base + shift : U32)
  +target = U32.from_nat(Nat.add(U32.to_nat(base),Nat.add(shift_value,units)))
  %Equal.sym(Nat,U32.to_nat(groups),group_count,groups_exact) :
    {U32.from_nat(Nat.add(U32.to_nat(shifted),Nat.mul(Nat.mul(unit,4n),_))) == target : U32}
  Equal.trans(U32,U32.from_nat(Nat.add(U32.to_nat(shifted),units)),(shifted + U32.from_nat(units) : U32),target,
    %Arithmetic.native_roundtrip(shifted) : {U32.from_nat(Nat.add(U32.to_nat(shifted),units)) == (_ + U32.from_nat(units) : U32) : U32}
    Equal.sym(U32,(U32.from_nat(U32.to_nat(shifted)) + U32.from_nat(units) : U32),U32.from_nat(Nat.add(U32.to_nat(shifted),units)),WordArithmetic.native_add(U32.to_nat(shifted),units)),
    Equal.trans(U32,(shifted + U32.from_nat(units) : U32),(base + (shift + U32.from_nat(units) : U32) : U32),target,
      Arithmetic.add_associative(base,shift,U32.from_nat(units)),
      %Arithmetic.from_nat_exact(shift,shift_value,shift_exact) : {(base + (_ + U32.from_nat(units) : U32) : U32) == target : U32}
      %Equal.sym(U32,(U32.from_nat(shift_value) + U32.from_nat(units) : U32),U32.from_nat(Nat.add(shift_value,units)),WordArithmetic.native_add(shift_value,units)) :
        {(base + _ : U32) == target : U32}
      %Arithmetic.native_roundtrip(base) : {(_ + U32.from_nat(Nat.add(shift_value,units)) : U32) == target : U32}
      WordArithmetic.native_add(U32.to_nat(base),Nat.add(shift_value,units))))

# The first cell of a job and its row add up to the cell of its position.
def first_algebra(+base: Nat,+shift: Nat,+unit: Nat,+row: Nat,+groups: Nat) ->
  {Nat.add(Nat.add(base,Nat.add(Nat.mul(shift,groups),Nat.mul(Nat.mul(unit,4n),groups))),Nat.mul(row,groups)) ==
    Nat.add(base,Nat.mul(Nat.add(shift,Nat.add(Nat.mul(unit,4n),row)),groups)) : Nat}:
  +left = Nat.mul(shift,groups)
  +middle = Nat.mul(Nat.mul(unit,4n),groups)
  +right = Nat.mul(row,groups)
  Equal.trans(Nat,Nat.add(Nat.add(base,Nat.add(left,middle)),right),Nat.add(base,Nat.add(Nat.add(left,middle),right)),
    Nat.add(base,Nat.mul(Nat.add(shift,Nat.add(Nat.mul(unit,4n),row)),groups)),
    Addition.associative(base,Nat.add(left,middle),right),
    Equal.cong(Nat,Nat,value => Nat.add(base,value),Nat.add(Nat.add(left,middle),right),Nat.mul(Nat.add(shift,Nat.add(Nat.mul(unit,4n),row)),groups),
      Equal.trans(Nat,Nat.add(Nat.add(left,middle),right),Nat.add(left,Nat.add(middle,right)),Nat.mul(Nat.add(shift,Nat.add(Nat.mul(unit,4n),row)),groups),
        Addition.associative(left,middle,right),
        %RowMajor.distrib(Nat.mul(unit,4n),row,groups) : {Nat.add(left,_) == Nat.mul(Nat.add(shift,Nat.add(Nat.mul(unit,4n),row)),groups) : Nat}
        Equal.sym(Nat,Nat.mul(Nat.add(shift,Nat.add(Nat.mul(unit,4n),row)),groups),Nat.add(left,Nat.mul(Nat.add(Nat.mul(unit,4n),row),groups)),
          RowMajor.distrib(shift,Nat.add(Nat.mul(unit,4n),row),groups)))))

# ---- The value a call stores ----

# The width word of a pointwise problem is its width.
def width_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+checked: Guards.Valid<problem>,
  +support: Guards.Supported<problem,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>,
  +kernel_one: {Guards.kernel(problem) == 1n : Nat},+stride_one: {Guards.stride(problem) == 1n : Nat},+pad_zero: {Guards.pad(problem) == 0n : Nat}) ->
  {U32.to_nat(BandCover.width_word(problem)) == Guards.width(problem) : Nat}:
  Guards.Valid{+v0,+v1,+v2,+v3,+v4,+v5,+v6,+v7,+v8,+v9,+v10} = checked
  +same = point_width(problem,kernel_one,stride_one,pad_zero,v3)
  Equal.trans(Nat,U32.to_nat(U32.from_nat(ConvSpec.out_width(problem))),ConvSpec.out_width(problem),Guards.width(problem),
    Bounds.roundtrip(ConvSpec.out_width(problem),Machine.capacity(limits),Limits.capacity_fits(limits),
      %Equal.sym(Nat,ConvSpec.out_width(problem),Guards.width(problem),same) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
      Limits.dimension_capacity(limits,Guards.width(problem),SupportedCoordinates.width_bound(limits,problem,support))),
    same)

# Lane lane of what a pointwise call stores at position index of the call and
# pair pair is the matrix value of its channel at its position in the image.
def call_lane(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +kernel_one: {Guards.kernel(problem) == 1n : Nat},+stride_one: {Guards.stride(problem) == 1n : Nat},+pad_zero: {Guards.pad(problem) == 0n : Nat},
  +channels_exact: {Guards.input_channel_count(problem) == Nat.mul(U32.to_nat((Coordinates.native_inner(problem) / 16 : U32)),16n) : Nat},
  +groups_exact: {U32.to_nat(Gemm.blocks(MicroLeaf.rows(problem))) == TiledValues.groups_of(Guards.output_channel_count(problem)) : Nat},
  +inner_exact: {U32.to_nat(Coordinates.native_inner(problem)) == Coordinates.inner(problem) : Nat},
  +count_exact: {PointBuffer.pair_count(MicroLeaf.rows(problem),Coordinates.native_inner(problem)) ==
    Nat.mul(U32.to_nat(Gemm.pairs_of(MicroLeaf.rows(problem))),Coordinates.inner(problem)) : Nat},
  +output: Tree.Tree<Tile.Tile16>,+depth: Nat,+balanced: Shape.Balanced(Tile.Tile16,depth,output),+limit: {Nat.is_le(depth,31n) == True{} : Bool},
  +capacity_ok: {U32.is_le(Tree.capacity(Tile.Tile16,output),2147483648) == True{} : Bool},
  +room: {Nat.is_le(PointBuffer.pair_count(MicroLeaf.rows(problem),Coordinates.native_inner(problem)),U32.to_nat(Tree.capacity(Tile.Tile16,output))) == True{} : Bool},
  +image: U32,+local: U32,+count: U32,+offset: U32,+base: U32,+band: Buffer.Model<Tile.Tile8>,+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
  +grid: {Nat.is_le(Nat.add(U32.to_nat(local),U32.to_nat(count)),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool},
  +image_room: {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(image),Cells.count(problem,Guards.height(problem))),Cells.count(problem,Guards.height(problem))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +start: Nat,+start_exact: {start == Nat.mul(U32.to_nat(image),Cells.count(problem,Guards.height(problem))) : Nat},+held: Band.HeldImage(~Holder,~value,holder,problem,storage,start),
  +reads: Skeleton.BandHolds(~Holder,~value,holder,band,base,Calls.band_first(problem,image,local),Calls.band_cells(problem,local,count)),
  +index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool},+pair: Nat,+pair_inside: {Nat.is_lt(pair,U32.to_nat(Gemm.pairs_of(MicroLeaf.rows(problem)))) == True{} : Bool},
  +lane: Nat,+lane_small: {Nat.is_lt(lane,16n) == True{} : Bool},+channel_inside: {Nat.is_lt(PointValues.pair_channel(lane,pair),Guards.output_channel_count(problem)) == True{} : Bool}) ->
  {PointSpec.pair_lane(lane,PointReadback.stored_value(PointReadback.Target{Gemm.PointMatrix{Gemm.TileMatrix{MicroLeaf.rows(problem),count,Coordinates.native_inner(problem),offset,(offset + count : U32)},
      (base + KernelWindow.start(local,1,BandCover.width_word(problem),BandCover.first_word(problem,local)) * Calls.groups_in(problem) : U32),Calls.groups_in(problem)},
      PointJobs.Readers{Inputs.bias_cells(problem,storage),Buffer.tree(~Tile.Tile8,band),
        PointBuffer.pair_loop(MicroLeaf.rows(problem),Coordinates.native_inner(problem),Inputs.weight_cells(problem,storage),output)}},
      U32.from_nat(pair),U32.from_nat(Nat.div(index,4n)),Nat.mod(index,4n))) ==
    Values.matrix_value(problem,PointValues.pair_channel(lane,pair),Nat.div(Nat.add(U32.to_nat(local),index),Guards.width(problem)),
      Nat.mod(Nat.add(U32.to_nat(local),index),Guards.width(problem))) : F32}:
  +checked = Guards.valid(problem,valid)
  +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  +cap = Machine.capacity(limits)
  +cap_ok = Limits.capacity_fits(limits)
  +width = Guards.width(problem)
  +height = Guards.height(problem)
  +groups = Cells.group_count(problem)
  +local_value = U32.to_nat(local)
  +row_first = U32.to_nat(BandCover.first_word(problem,local))
  +rows = U32.to_nat(BandCover.rows_word(problem,local,count))
  +cover = BandCover.band_cover(limits,problem,valid,supported,local,count,positive,grid)
  +words = Calls.band_words(limits,problem,valid,supported,image,local,count,cover,image_room)
  +from_exact = {%Equal.sym(Nat,start,Nat.mul(U32.to_nat(image),Cells.count(problem,height)),start_exact) :
      {U32.to_nat(Calls.band_first(problem,image,local)) == Nat.add(_,Cells.count(problem,row_first)) : Nat}
    Calls.band_first_exact(problem,image,local,count,words) : {U32.to_nat(Calls.band_first(problem,image,local)) == Nat.add(start,Cells.count(problem,row_first)) : Nat}}
  +cells_fit = {%Calls.band_cells_exact(problem,image,local,count,words) : {Nat.is_le(_,U32.to_nat(Calls.band_cells(problem,local,count))) == True{} : Bool}
    Order.reflexive(U32.to_nat(Calls.band_cells(problem,local,count))) : {Nat.is_le(Cells.count(problem,rows),U32.to_nat(Calls.band_cells(problem,local,count))) == True{} : Bool}}
  +fits = cover_fits(problem,row_first,rows,local_value,U32.to_nat(count),cover)
  +contents = Band.band_contents(~Holder,~value,holder,problem,storage,band,base,Calls.band_first(problem,image,local),Calls.band_cells(problem,local,count),reads,
    start,row_first,rows,from_exact,fits,cells_fit,held)
  +placed = band_place(problem,checked,kernel_one,stride_one,pad_zero,row_first,rows,local_value,U32.to_nat(count),cover,index,inside,grid)
  +shifted = Nat.sub(local_value,Nat.mul(row_first,width))
  +unit = Nat.div(index,4n)
  +row = Nat.mod(index,4n)
  +position = Nat.add(local_value,index)
  +groups_word = Calls.groups_in(problem)
  +groups_positive = Calls.group_positive(problem,valid)
  +gi_exact = Calls.groups_word_exact(limits,problem,valid,supported)
  +width_word_exact = width_exact(limits,problem,checked,support,kernel_one,stride_one,pad_zero)
  +image_cells = Cells.count(problem,height)
  +cells_cap = {Order.transitive(image_cells,Nat.add(Nat.mul(U32.to_nat(image),image_cells),image_cells),U32.to_nat(cap),
      Order.right_le(Nat.mul(U32.to_nat(image),image_cells),image_cells),image_room) : {Nat.is_le(image_cells,U32.to_nat(cap)) == True{} : Bool}}
  +local_cap = {Order.transitive(local_value,Nat.mul(Nat.mul(height,width),groups),U32.to_nat(cap),
      Order.transitive(local_value,Nat.mul(height,width),Nat.mul(Nat.mul(height,width),groups),
        Order.transitive(local_value,Nat.add(local_value,U32.to_nat(count)),Nat.mul(height,width),Order.le_add(local_value,U32.to_nat(count)),
          %point_height(problem,kernel_one,stride_one,pad_zero,Contents.valid_condition(problem,valid,8n)) :
            {Nat.is_le(Nat.add(local_value,U32.to_nat(count)),Nat.mul(_,width)) == True{} : Bool}
          %point_width(problem,kernel_one,stride_one,pad_zero,Contents.valid_condition(problem,valid,7n)) :
            {Nat.is_le(Nat.add(local_value,U32.to_nat(count)),Nat.mul(ConvSpec.out_height(problem),_)) == True{} : Bool}
          grid),
        RowMajor.product_covers(Nat.mul(height,width),groups,groups_positive)),
      cells_cap) : {Nat.is_le(local_value,U32.to_nat(cap)) == True{} : Bool}}
  +start_word_exact = {%width_word_exact : {U32.to_nat(KernelWindow.start(local,1,BandCover.width_word(problem),BandCover.first_word(problem,local))) ==
        Nat.sub(local_value,Nat.mul(row_first,_)) : Nat}
    start_word(local,BandCover.first_word(problem,local),BandCover.width_word(problem),cap,cap_ok,
      %Equal.sym(Nat,U32.to_nat(BandCover.width_word(problem)),width,width_word_exact) : {Nat.is_le(Nat.mul(row_first,_),local_value) == True{} : Bool}
      place_reach(row_first,rows,local_value,index,width,height,placed),local_cap) :
    {U32.to_nat(KernelWindow.start(local,1,BandCover.width_word(problem),BandCover.first_word(problem,local))) == shifted : Nat}}
  +rows_height = {Order.transitive(rows,Nat.add(row_first,rows),height,Order.right_le(row_first,rows),fits) : {Nat.is_le(rows,height) == True{} : Bool}}
  +shift_cap = {Order.transitive(Nat.mul(shifted,groups),Nat.mul(Nat.mul(rows,width),groups),U32.to_nat(cap),
      RowMajor.mul_le(shifted,Nat.mul(rows,width),groups,
        Bounds.strict_le(shifted,Nat.mul(rows,width),Order.le_lt_trans(shifted,Nat.add(shifted,index),Nat.mul(rows,width),Order.le_add(shifted,index),
          place_inside(row_first,rows,local_value,index,width,height,placed)))),
      Order.transitive(Nat.mul(Nat.mul(rows,width),groups),image_cells,U32.to_nat(cap),
        RowMajor.mul_le(Nat.mul(rows,width),Nat.mul(height,width),groups,RowMajor.mul_le(rows,height,width,rows_height)),cells_cap)) :
    {Nat.is_le(Nat.mul(shifted,groups),U32.to_nat(cap)) == True{} : Bool}}
  +start_cell = KernelWindow.start(local,1,BandCover.width_word(problem),BandCover.first_word(problem,local))
  +shift_exact = Arithmetic.product_exact(cap,cap_ok,start_cell,groups_word,shifted,groups,start_word_exact,gi_exact,shift_cap)
  +first = Nat.add(U32.to_nat(base),Nat.add(Nat.mul(shifted,groups),Nat.mul(Nat.mul(unit,4n),groups)))
  +first_exact = {%Equal.sym(Nat,U32.to_nat(groups_word),groups,gi_exact) :
      {Nat.add(first,Nat.mul(row,_)) == Nat.add(U32.to_nat(base),Nat.mul(Nat.add(shifted,index),groups)) : Nat}
    %Equal.sym(Nat,index,Nat.add(Nat.mul(unit,4n),row),DivisionOrder.quotient_exact(index,4n,{==})) :
      {Nat.add(first,Nat.mul(row,groups)) == Nat.add(U32.to_nat(base),Nat.mul(Nat.add(shifted,_),groups)) : Nat}
    first_algebra(U32.to_nat(base),shifted,unit,row,groups) :
    {Nat.add(first,Nat.mul(row,U32.to_nat(groups_word))) == Nat.add(U32.to_nat(base),Nat.mul(Nat.add(shifted,index),groups)) : Nat}}
  PointValues.spot_value(limits,problem,storage,valid,supported,kernel_one,stride_one,pad_zero,MicroLeaf.rows(problem),count,Coordinates.native_inner(problem),offset,(offset + count : U32),
    (base + start_cell * groups_word : U32),groups_word,Inputs.bias_cells(problem,storage),Buffer.tree(~Tile.Tile8,band),
    PointBuffer.pair_loop(MicroLeaf.rows(problem),Coordinates.native_inner(problem),Inputs.weight_cells(problem,storage),output),pair,unit,row,
    Division.lt_succ_le(row,3n,DivisionOrder.remainder_small(index,4n,{==})),channels_exact,U32.to_nat(base),row_first,rows,contents,fits,
    Nat.add(shifted,index),place_inside(row_first,rows,local_value,index,width,height,placed),first,
    cell_first(base,(start_cell * groups_word : U32),Nat.mul(shifted,groups),shift_exact,unit,groups_word,groups,gi_exact),first_exact,
    Nat.div(position,width),Nat.mod(position,width),place_pixel(row_first,rows,local_value,index,width,height,placed),
    place_row(row_first,rows,local_value,index,width,height,placed),place_column(row_first,rows,local_value,index,width,height,placed),lane,channel_inside,
    PointValues.pairs_read(limits,problem,storage,valid,supported,MicroLeaf.rows(problem),Coordinates.native_inner(problem),groups_exact,inner_exact,count_exact,
      output,depth,balanced,limit,capacity_ok,room,pair,pair_inside,lane,lane_small,channel_inside),
    PointValues.bias_read(limits,problem,storage,valid,supported,MicroLeaf.rows(problem),groups_exact,pair,lane,lane_small,channel_inside))

# ---- The staged pairs after a call ----

# The run of a pointwise call: its matrix and what its jobs read.
def call_target(+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+count: U32,+offset: U32,+base: U32,+local: U32,+band: Buffer.Model<Tile.Tile8>,
  +output: Tree.Tree<Tile.Tile16>) -> PointReadback.Target:
  PointReadback.Target{Gemm.PointMatrix{Gemm.TileMatrix{MicroLeaf.rows(problem),count,Coordinates.native_inner(problem),offset,(offset + count : U32)},(base + KernelWindow.start(local,1,BandCover.width_word(problem),BandCover.first_word(problem,local)) * Calls.groups_in(problem) : U32),Calls.groups_in(problem)},
    PointJobs.Readers{Inputs.bias_cells(problem,storage),Buffer.tree(~Tile.Tile8,band),PointBuffer.pair_loop(MicroLeaf.rows(problem),Coordinates.native_inner(problem),Inputs.weight_cells(problem,storage),output)}}

# A machine has room for four positions.
def four_fits(+limits: Machine.Limits) -> {Nat.is_le(4n,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Order.transitive(4n,7n,U32.to_nat(Machine.capacity(limits)),{==},Bounds.as_nat_le(7,Machine.capacity(limits),Limits.seven_within_capacity(limits)))

# The run of a call checked against twice the capacity.
def call_checked(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+count: U32,+offset: U32,+base: U32,+local: U32,+band: Buffer.Model<Tile.Tile8>,
  +output: Tree.Tree<Tile.Tile16>,
  +groups_exact: {U32.to_nat(Gemm.blocks(MicroLeaf.rows(problem))) == TiledValues.groups_of(Guards.output_channel_count(problem)) : Nat},
  +groups_positive: {Nat.is_lt(0n,TiledValues.groups_of(Guards.output_channel_count(problem))) == True{} : Bool},+group_room: {Nat.is_le(Nat.add(Nat.mul(TiledValues.groups_of(Guards.output_channel_count(problem)),2n),2n),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},+end_exact: {U32.to_nat((offset + count : U32)) == Nat.add(U32.to_nat(offset),U32.to_nat(count)) : Nat},
  +span: {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(offset),U32.to_nat(count)),TiledValues.groups_of(Guards.output_channel_count(problem))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> PointReadback.Checked:
  PointReadback.Checked{call_target(problem,storage,count,offset,base,local,band,output),Machine.half_word(limits),
    PointReadback.call_run(MicroLeaf.rows(problem),count,Coordinates.native_inner(problem),offset,(offset + count : U32),(base + KernelWindow.start(local,1,BandCover.width_word(problem),BandCover.first_word(problem,local)) * Calls.groups_in(problem) : U32),Calls.groups_in(problem),
      PointJobs.Readers{Inputs.bias_cells(problem,storage),Buffer.tree(~Tile.Tile8,band),PointBuffer.pair_loop(MicroLeaf.rows(problem),Coordinates.native_inner(problem),Inputs.weight_cells(problem,storage),output)},
      Machine.half_word(limits),Limits.half_word_fits(limits),U32.to_nat(Machine.capacity(limits)),Limits.double_bound(limits,U32.to_nat(Machine.capacity(limits)),Order.reflexive(U32.to_nat(Machine.capacity(limits)))),
      TiledValues.groups_of(Guards.output_channel_count(problem)),groups_exact,groups_positive,group_room,four_fits(limits),positive,end_exact,span)}

# The cells before the first position of a call keep their pairs.
def call_kept(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+count: U32,+offset: U32,+base: U32,+local: U32,+band: Buffer.Model<Tile.Tile8>,
  +output: Tree.Tree<Tile.Tile16>,
  +groups_exact: {U32.to_nat(Gemm.blocks(MicroLeaf.rows(problem))) == TiledValues.groups_of(Guards.output_channel_count(problem)) : Nat},
  +groups_positive: {Nat.is_lt(0n,TiledValues.groups_of(Guards.output_channel_count(problem))) == True{} : Bool},+group_room: {Nat.is_le(Nat.add(Nat.mul(TiledValues.groups_of(Guards.output_channel_count(problem)),2n),2n),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},+end_exact: {U32.to_nat((offset + count : U32)) == Nat.add(U32.to_nat(offset),U32.to_nat(count)) : Nat},
  +span: {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(offset),U32.to_nat(count)),TiledValues.groups_of(Guards.output_channel_count(problem))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +staged: Tree.Tree<Tile.Tile16>,+staged_depth: Nat,+staged_balanced: Shape.Balanced(Tile.Tile16,staged_depth,staged),+staged_depth_ok: {Nat.is_le(staged_depth,30n) == True{} : Bool},
  +staged_room: {Nat.is_le(Nat.mul(U32.to_nat((offset + count : U32)),U32.to_nat(Gemm.pairs_of(MicroLeaf.rows(problem)))),U32.to_nat(Tree.capacity(Tile.Tile16,staged))) == True{} : Bool},
  +query: Nat,+before: {Nat.is_lt(query,Nat.mul(U32.to_nat(offset),U32.to_nat(Gemm.pairs_of(MicroLeaf.rows(problem))))) == True{} : Bool},+inside: {U32.is_lt(U32.from_nat(query),Tree.capacity(Tile.Tile16,staged)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile16,Writes.pure(Tile.Tile16,PointJobs.all_writes(U32.to_nat(Gemm.point_count(Gemm.PointMatrix{Gemm.TileMatrix{MicroLeaf.rows(problem),count,Coordinates.native_inner(problem),offset,(offset + count : U32)},(base + KernelWindow.start(local,1,BandCover.width_word(problem),BandCover.first_word(problem,local)) * Calls.groups_in(problem) : U32),Calls.groups_in(problem)})),0,Gemm.PointMatrix{Gemm.TileMatrix{MicroLeaf.rows(problem),count,Coordinates.native_inner(problem),offset,(offset + count : U32)},(base + KernelWindow.start(local,1,BandCover.width_word(problem),BandCover.first_word(problem,local)) * Calls.groups_in(problem) : U32),Calls.groups_in(problem)},PointJobs.Readers{Inputs.bias_cells(problem,storage),Buffer.tree(~Tile.Tile8,band),PointBuffer.pair_loop(MicroLeaf.rows(problem),Coordinates.native_inner(problem),Inputs.weight_cells(problem,storage),output)}),staged),U32.from_nat(query)) == Tree.read(Tile.Tile16,staged,U32.from_nat(query)) : Tile.Tile16}:
  +checked = call_checked(limits,problem,storage,count,offset,base,local,band,output,groups_exact,groups_positive,group_room,positive,end_exact,span)
  %Equal.sym(List<&2,Writes.Assignment<Tile.Tile16>>,PointJobs.all_writes(U32.to_nat(Gemm.point_count(PointReadback.target_matrix(PointReadback.checked_target(checked)))),0,PointReadback.target_matrix(PointReadback.checked_target(checked)),PointReadback.target_readers(PointReadback.checked_target(checked))),PointReadback.checked_writes(checked),PointReadback.checked_stores(checked)) :
    {Tree.read(Tile.Tile16,Writes.pure(Tile.Tile16,_,staged),U32.from_nat(query)) == Tree.read(Tile.Tile16,staged,U32.from_nat(query)) : Tile.Tile16}
  PointReadback.spot_keeps(checked,query,before,staged_depth,staged,staged_balanced,staged_depth_ok,staged_room,inside)

# Lane lane of the pair a call leaves at step step of its positions and pair
# pair is the matrix value of its channel at its position in the image.
def call_written(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +kernel_one: {Guards.kernel(problem) == 1n : Nat},+stride_one: {Guards.stride(problem) == 1n : Nat},+pad_zero: {Guards.pad(problem) == 0n : Nat},
  +channels_exact: {Guards.input_channel_count(problem) == Nat.mul(U32.to_nat((Coordinates.native_inner(problem) / 16 : U32)),16n) : Nat},
  +groups_exact: {U32.to_nat(Gemm.blocks(MicroLeaf.rows(problem))) == TiledValues.groups_of(Guards.output_channel_count(problem)) : Nat},
  +inner_exact: {U32.to_nat(Coordinates.native_inner(problem)) == Coordinates.inner(problem) : Nat},
  +count_exact: {PointBuffer.pair_count(MicroLeaf.rows(problem),Coordinates.native_inner(problem)) ==
    Nat.mul(U32.to_nat(Gemm.pairs_of(MicroLeaf.rows(problem))),Coordinates.inner(problem)) : Nat},
  +output: Tree.Tree<Tile.Tile16>,+depth: Nat,+balanced: Shape.Balanced(Tile.Tile16,depth,output),+limit: {Nat.is_le(depth,31n) == True{} : Bool},
  +capacity_ok: {U32.is_le(Tree.capacity(Tile.Tile16,output),2147483648) == True{} : Bool},
  +room: {Nat.is_le(PointBuffer.pair_count(MicroLeaf.rows(problem),Coordinates.native_inner(problem)),U32.to_nat(Tree.capacity(Tile.Tile16,output))) == True{} : Bool},
  +image: U32,+local: U32,+count: U32,+offset: U32,+base: U32,+band: Buffer.Model<Tile.Tile8>,+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
  +grid: {Nat.is_le(Nat.add(U32.to_nat(local),U32.to_nat(count)),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool},
  +image_room: {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(image),Cells.count(problem,Guards.height(problem))),Cells.count(problem,Guards.height(problem))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +start: Nat,+start_exact: {start == Nat.mul(U32.to_nat(image),Cells.count(problem,Guards.height(problem))) : Nat},+held: Band.HeldImage(~Holder,~value,holder,problem,storage,start),
  +reads: Skeleton.BandHolds(~Holder,~value,holder,band,base,Calls.band_first(problem,image,local),Calls.band_cells(problem,local,count)),
  +groups_positive: {Nat.is_lt(0n,TiledValues.groups_of(Guards.output_channel_count(problem))) == True{} : Bool},+group_room: {Nat.is_le(Nat.add(Nat.mul(TiledValues.groups_of(Guards.output_channel_count(problem)),2n),2n),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +end_exact: {U32.to_nat((offset + count : U32)) == Nat.add(U32.to_nat(offset),U32.to_nat(count)) : Nat},
  +span: {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(offset),U32.to_nat(count)),TiledValues.groups_of(Guards.output_channel_count(problem))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +staged: Tree.Tree<Tile.Tile16>,+staged_depth: Nat,+staged_balanced: Shape.Balanced(Tile.Tile16,staged_depth,staged),+staged_depth_ok: {Nat.is_le(staged_depth,30n) == True{} : Bool},
  +staged_room: {Nat.is_le(Nat.mul(U32.to_nat((offset + count : U32)),U32.to_nat(Gemm.pairs_of(MicroLeaf.rows(problem)))),U32.to_nat(Tree.capacity(Tile.Tile16,staged))) == True{} : Bool},
  +step: Nat,+step_inside: {Nat.is_lt(step,U32.to_nat(count)) == True{} : Bool},+pair: Nat,+pair_inside: {Nat.is_lt(pair,U32.to_nat(Gemm.pairs_of(MicroLeaf.rows(problem)))) == True{} : Bool},
  +lane: Nat,+lane_small: {Nat.is_lt(lane,16n) == True{} : Bool},+channel_inside: {Nat.is_lt(PointValues.pair_channel(lane,pair),Guards.output_channel_count(problem)) == True{} : Bool}) ->
  {PointSpec.pair_lane(lane,Tree.read(Tile.Tile16,Writes.pure(Tile.Tile16,PointJobs.all_writes(U32.to_nat(Gemm.point_count(Gemm.PointMatrix{Gemm.TileMatrix{MicroLeaf.rows(problem),count,Coordinates.native_inner(problem),offset,(offset + count : U32)},(base + KernelWindow.start(local,1,BandCover.width_word(problem),BandCover.first_word(problem,local)) * Calls.groups_in(problem) : U32),Calls.groups_in(problem)})),0,Gemm.PointMatrix{Gemm.TileMatrix{MicroLeaf.rows(problem),count,Coordinates.native_inner(problem),offset,(offset + count : U32)},(base + KernelWindow.start(local,1,BandCover.width_word(problem),BandCover.first_word(problem,local)) * Calls.groups_in(problem) : U32),Calls.groups_in(problem)},PointJobs.Readers{Inputs.bias_cells(problem,storage),Buffer.tree(~Tile.Tile8,band),PointBuffer.pair_loop(MicroLeaf.rows(problem),Coordinates.native_inner(problem),Inputs.weight_cells(problem,storage),output)}),staged),U32.from_nat(Nat.add(Nat.mul(Nat.add(U32.to_nat(offset),step),U32.to_nat(Gemm.pairs_of(MicroLeaf.rows(problem)))),pair)))) ==
    Values.matrix_value(problem,PointValues.pair_channel(lane,pair),Nat.div(Nat.add(U32.to_nat(local),step),Guards.width(problem)),
      Nat.mod(Nat.add(U32.to_nat(local),step),Guards.width(problem))) : F32}:
  +checked = call_checked(limits,problem,storage,count,offset,base,local,band,output,groups_exact,groups_positive,group_room,positive,end_exact,span)
  +word = U32.from_nat(Nat.add(Nat.mul(Nat.add(U32.to_nat(offset),step),U32.to_nat(Gemm.pairs_of(MicroLeaf.rows(problem)))),pair))
  +wanted = Values.matrix_value(problem,PointValues.pair_channel(lane,pair),Nat.div(Nat.add(U32.to_nat(local),step),Guards.width(problem)),
    Nat.mod(Nat.add(U32.to_nat(local),step),Guards.width(problem)))
  +before_end = {%Equal.sym(Nat,U32.to_nat((offset + count : U32)),Nat.add(U32.to_nat(offset),U32.to_nat(count)),end_exact) :
      {Nat.is_lt(Nat.add(U32.to_nat(offset),step),_) == True{} : Bool}
    Order.offset_lt(U32.to_nat(offset),step,U32.to_nat(count),step_inside) : {Nat.is_lt(Nat.add(U32.to_nat(offset),step),U32.to_nat((offset + count : U32))) == True{} : Bool}}
  +stored = PointReadback.stored_at(checked,step,pair,pair_inside,before_end,staged_depth,staged,staged_balanced,staged_depth_ok,staged_room)
  %Equal.sym(List<&2,Writes.Assignment<Tile.Tile16>>,PointJobs.all_writes(U32.to_nat(Gemm.point_count(PointReadback.target_matrix(PointReadback.checked_target(checked)))),0,PointReadback.target_matrix(PointReadback.checked_target(checked)),PointReadback.target_readers(PointReadback.checked_target(checked))),PointReadback.checked_writes(checked),PointReadback.checked_stores(checked)) :
    {PointSpec.pair_lane(lane,Tree.read(Tile.Tile16,Writes.pure(Tile.Tile16,_,staged),word)) == wanted : F32}
  %Equal.sym(Tile.Tile16,Tree.read(Tile.Tile16,Writes.pure(Tile.Tile16,PointReadback.checked_writes(checked),staged),word),
      PointReadback.stored_value(call_target(problem,storage,count,offset,base,local,band,output),U32.from_nat(pair),U32.from_nat(Nat.div(step,4n)),Nat.mod(step,4n)),stored) :
    {PointSpec.pair_lane(lane,_) == wanted : F32}
  call_lane(~Holder,~value,holder,limits,problem,storage,valid,supported,kernel_one,stride_one,pad_zero,channels_exact,groups_exact,inner_exact,count_exact,output,depth,balanced,limit,capacity_ok,room,
    image,local,count,offset,base,band,positive,grid,image_room,start,start_exact,held,reads,step,step_inside,pair,pair_inside,lane,lane_small,channel_inside)

# ---- The record of a pointwise call ----

# A call on positions [local, local + count) of an image, stored from offset
# on, with what it reads and the staged pairs it leaves.
type PointCall<-Holder: Data,-value: Holder -> U32 -> Tile.Tile8,-holder: Holder,-limits: Machine.Limits,-problem: ConvSpec.Problem,-storage: Inputs.Contract<problem>,-local: U32,-count: U32,-offset: U32,-staged: Tree.Tree<Tile.Tile16>,-result: Tree.Tree<Tile.Tile16>> is Data:
  PointCall{valid: {ConvSpec.valid(problem) == True{} : Bool},supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},kernel_one: {Guards.kernel(problem) == 1n : Nat},stride_one: {Guards.stride(problem) == 1n : Nat},pad_zero: {Guards.pad(problem) == 0n : Nat},channels_exact: {Guards.input_channel_count(problem) == Nat.mul(U32.to_nat((Coordinates.native_inner(problem) / 16 : U32)),16n) : Nat},groups_exact: {U32.to_nat(Gemm.blocks(MicroLeaf.rows(problem))) == TiledValues.groups_of(Guards.output_channel_count(problem)) : Nat},inner_exact: {U32.to_nat(Coordinates.native_inner(problem)) == Coordinates.inner(problem) : Nat},count_exact: {PointBuffer.pair_count(MicroLeaf.rows(problem),Coordinates.native_inner(problem)) ==
    Nat.mul(U32.to_nat(Gemm.pairs_of(MicroLeaf.rows(problem))),Coordinates.inner(problem)) : Nat},output: Tree.Tree<Tile.Tile16>,depth: Nat,balanced: Shape.Balanced(Tile.Tile16,depth,output),limit: {Nat.is_le(depth,31n) == True{} : Bool},capacity_ok: {U32.is_le(Tree.capacity(Tile.Tile16,output),2147483648) == True{} : Bool},room: {Nat.is_le(PointBuffer.pair_count(MicroLeaf.rows(problem),Coordinates.native_inner(problem)),U32.to_nat(Tree.capacity(Tile.Tile16,output))) == True{} : Bool},image: U32,base: U32,band: Buffer.Model<Tile.Tile8>,positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},grid: {Nat.is_le(Nat.add(U32.to_nat(local),U32.to_nat(count)),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool},image_room: {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(image),Cells.count(problem,Guards.height(problem))),Cells.count(problem,Guards.height(problem))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},start: Nat,start_exact: {start == Nat.mul(U32.to_nat(image),Cells.count(problem,Guards.height(problem))) : Nat},held: Band.HeldImage(~Holder,~value,holder,problem,storage,start),reads: Skeleton.BandHolds(~Holder,~value,holder,band,base,Calls.band_first(problem,image,local),Calls.band_cells(problem,local,count)),groups_positive: {Nat.is_lt(0n,TiledValues.groups_of(Guards.output_channel_count(problem))) == True{} : Bool},group_room: {Nat.is_le(Nat.add(Nat.mul(TiledValues.groups_of(Guards.output_channel_count(problem)),2n),2n),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},end_exact: {U32.to_nat((offset + count : U32)) == Nat.add(U32.to_nat(offset),U32.to_nat(count)) : Nat},span: {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(offset),U32.to_nat(count)),TiledValues.groups_of(Guards.output_channel_count(problem))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},staged_depth: Nat,staged_balanced: Shape.Balanced(Tile.Tile16,staged_depth,staged),staged_depth_ok: {Nat.is_le(staged_depth,30n) == True{} : Bool},staged_room: {Nat.is_le(Nat.mul(U32.to_nat((offset + count : U32)),U32.to_nat(Gemm.pairs_of(MicroLeaf.rows(problem)))),U32.to_nat(Tree.capacity(Tile.Tile16,staged))) == True{} : Bool},result_exact: {result == Writes.pure(Tile.Tile16,PointJobs.all_writes(U32.to_nat(Gemm.point_count(Gemm.PointMatrix{Gemm.TileMatrix{MicroLeaf.rows(problem),count,Coordinates.native_inner(problem),offset,(offset + count : U32)},(base + KernelWindow.start(local,1,BandCover.width_word(problem),BandCover.first_word(problem,local)) * Calls.groups_in(problem) : U32),Calls.groups_in(problem)})),0,Gemm.PointMatrix{Gemm.TileMatrix{MicroLeaf.rows(problem),count,Coordinates.native_inner(problem),offset,(offset + count : U32)},(base + KernelWindow.start(local,1,BandCover.width_word(problem),BandCover.first_word(problem,local)) * Calls.groups_in(problem) : U32),Calls.groups_in(problem)},PointJobs.Readers{Inputs.bias_cells(problem,storage),Buffer.tree(~Tile.Tile8,band),PointBuffer.pair_loop(MicroLeaf.rows(problem),Coordinates.native_inner(problem),Inputs.weight_cells(problem,storage),output)}),staged) : Tree.Tree<Tile.Tile16>}}

# Lane lane of the pair a call leaves at step step and pair pair is the matrix
# value of its channel at its position.
def point_written(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+local: U32,+count: U32,+offset: U32,+staged: Tree.Tree<Tile.Tile16>,+result: Tree.Tree<Tile.Tile16>,
  +call: PointCall<Holder,value,holder,limits,problem,storage,local,count,offset,staged,result>,
  +step: Nat,+step_inside: {Nat.is_lt(step,U32.to_nat(count)) == True{} : Bool},+pair: Nat,+pair_inside: {Nat.is_lt(pair,U32.to_nat(Gemm.pairs_of(MicroLeaf.rows(problem)))) == True{} : Bool},
  +lane: Nat,+lane_small: {Nat.is_lt(lane,16n) == True{} : Bool},+channel_inside: {Nat.is_lt(PointValues.pair_channel(lane,pair),Guards.output_channel_count(problem)) == True{} : Bool}) ->
  {PointSpec.pair_lane(lane,Tree.read(Tile.Tile16,result,U32.from_nat(Nat.add(Nat.mul(Nat.add(U32.to_nat(offset),step),U32.to_nat(Gemm.pairs_of(MicroLeaf.rows(problem)))),pair)))) ==
    Values.matrix_value(problem,PointValues.pair_channel(lane,pair),Nat.div(Nat.add(U32.to_nat(local),step),ConvSpec.out_width(problem)),
      Nat.mod(Nat.add(U32.to_nat(local),step),ConvSpec.out_width(problem))) : F32}:
  PointCall{+valid,+supported,+kernel_one,+stride_one,+pad_zero,+channels_exact,+groups_exact,+inner_exact,+count_exact,+output,+depth,+balanced,+limit,+capacity_ok,+room,+image,+base,+band,+positive,+grid,+image_room,+start,+start_exact,+held,+reads,+groups_positive,+group_room,+end_exact,+span,+staged_depth,+staged_balanced,+staged_depth_ok,+staged_room,+result_exact} = call
  +word = U32.from_nat(Nat.add(Nat.mul(Nat.add(U32.to_nat(offset),step),U32.to_nat(Gemm.pairs_of(MicroLeaf.rows(problem)))),pair))
  %Equal.sym(Nat,ConvSpec.out_width(problem),Guards.width(problem),point_width(problem,kernel_one,stride_one,pad_zero,Contents.valid_condition(problem,valid,7n))) :
    {PointSpec.pair_lane(lane,Tree.read(Tile.Tile16,result,word)) ==
      Values.matrix_value(problem,PointValues.pair_channel(lane,pair),Nat.div(Nat.add(U32.to_nat(local),step),_),Nat.mod(Nat.add(U32.to_nat(local),step),_)) : F32}
  %Equal.sym(Tree.Tree<Tile.Tile16>,result,Writes.pure(Tile.Tile16,PointJobs.all_writes(U32.to_nat(Gemm.point_count(Gemm.PointMatrix{Gemm.TileMatrix{MicroLeaf.rows(problem),count,Coordinates.native_inner(problem),offset,(offset + count : U32)},(base + KernelWindow.start(local,1,BandCover.width_word(problem),BandCover.first_word(problem,local)) * Calls.groups_in(problem) : U32),Calls.groups_in(problem)})),0,Gemm.PointMatrix{Gemm.TileMatrix{MicroLeaf.rows(problem),count,Coordinates.native_inner(problem),offset,(offset + count : U32)},(base + KernelWindow.start(local,1,BandCover.width_word(problem),BandCover.first_word(problem,local)) * Calls.groups_in(problem) : U32),Calls.groups_in(problem)},PointJobs.Readers{Inputs.bias_cells(problem,storage),Buffer.tree(~Tile.Tile8,band),PointBuffer.pair_loop(MicroLeaf.rows(problem),Coordinates.native_inner(problem),Inputs.weight_cells(problem,storage),output)}),staged),result_exact) :
    {PointSpec.pair_lane(lane,Tree.read(Tile.Tile16,_,word)) ==
      Values.matrix_value(problem,PointValues.pair_channel(lane,pair),Nat.div(Nat.add(U32.to_nat(local),step),Guards.width(problem)),Nat.mod(Nat.add(U32.to_nat(local),step),Guards.width(problem))) : F32}
  call_written(Holder,value,holder,limits,problem,storage,valid,supported,kernel_one,stride_one,pad_zero,channels_exact,groups_exact,inner_exact,count_exact,output,depth,balanced,limit,capacity_ok,room,image,local,count,offset,base,band,positive,grid,image_room,start,start_exact,held,reads,groups_positive,group_room,end_exact,span,staged,staged_depth,staged_balanced,staged_depth_ok,staged_room,step,step_inside,pair,pair_inside,lane,lane_small,channel_inside)

# A call keeps the staged pairs before its offset.
def point_kept(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+local: U32,+count: U32,+offset: U32,+staged: Tree.Tree<Tile.Tile16>,+result: Tree.Tree<Tile.Tile16>,
  +call: PointCall<Holder,value,holder,limits,problem,storage,local,count,offset,staged,result>,
  +query: Nat,+before: {Nat.is_lt(query,Nat.mul(U32.to_nat(offset),U32.to_nat(Gemm.pairs_of(MicroLeaf.rows(problem))))) == True{} : Bool}) ->
  {Tree.read(Tile.Tile16,result,U32.from_nat(query)) == Tree.read(Tile.Tile16,staged,U32.from_nat(query)) : Tile.Tile16}:
  PointCall{+valid,+supported,+kernel_one,+stride_one,+pad_zero,+channels_exact,+groups_exact,+inner_exact,+count_exact,+output,+depth,+balanced,+limit,+capacity_ok,+room,+image,+base,+band,+positive,+grid,+image_room,+start,+start_exact,+held,+reads,+groups_positive,+group_room,+end_exact,+span,+staged_depth,+staged_balanced,+staged_depth_ok,+staged_room,+result_exact} = call
  +capacity = Tree.capacity(Tile.Tile16,staged)
  +fits = {%Equal.sym(U32,capacity,U32.shln(1,staged_depth),Shape.capacity(Tile.Tile16,staged_depth,staged,staged_balanced)) : {U32.is_le(_,2147483648) == True{} : Bool}
    U32Shift.power_fits(staged_depth,staged_depth_ok) : {U32.is_le(capacity,2147483648) == True{} : Bool}}
  +offset_le = {%Equal.sym(Nat,U32.to_nat((offset + count : U32)),Nat.add(U32.to_nat(offset),U32.to_nat(count)),end_exact) : {Nat.is_le(U32.to_nat(offset),_) == True{} : Bool}
    Order.le_add(U32.to_nat(offset),U32.to_nat(count)) : {Nat.is_le(U32.to_nat(offset),U32.to_nat((offset + count : U32))) == True{} : Bool}}
  +below = Bounds.nat_trans(query,Nat.mul(U32.to_nat(offset),U32.to_nat(Gemm.pairs_of(MicroLeaf.rows(problem)))),U32.to_nat(capacity),before,
    Order.transitive(Nat.mul(U32.to_nat(offset),U32.to_nat(Gemm.pairs_of(MicroLeaf.rows(problem)))),Nat.mul(U32.to_nat((offset + count : U32)),U32.to_nat(Gemm.pairs_of(MicroLeaf.rows(problem)))),U32.to_nat(capacity),
      RowMajor.mul_le(U32.to_nat(offset),U32.to_nat((offset + count : U32)),U32.to_nat(Gemm.pairs_of(MicroLeaf.rows(problem))),offset_le),staged_room))
  +inside = Bounds.native_bound(query,capacity,Bounds.roundtrip(query,capacity,fits,Bounds.strict_le(query,U32.to_nat(capacity),below)),below)
  %Equal.sym(Tree.Tree<Tile.Tile16>,result,Writes.pure(Tile.Tile16,PointJobs.all_writes(U32.to_nat(Gemm.point_count(Gemm.PointMatrix{Gemm.TileMatrix{MicroLeaf.rows(problem),count,Coordinates.native_inner(problem),offset,(offset + count : U32)},(base + KernelWindow.start(local,1,BandCover.width_word(problem),BandCover.first_word(problem,local)) * Calls.groups_in(problem) : U32),Calls.groups_in(problem)})),0,Gemm.PointMatrix{Gemm.TileMatrix{MicroLeaf.rows(problem),count,Coordinates.native_inner(problem),offset,(offset + count : U32)},(base + KernelWindow.start(local,1,BandCover.width_word(problem),BandCover.first_word(problem,local)) * Calls.groups_in(problem) : U32),Calls.groups_in(problem)},PointJobs.Readers{Inputs.bias_cells(problem,storage),Buffer.tree(~Tile.Tile8,band),PointBuffer.pair_loop(MicroLeaf.rows(problem),Coordinates.native_inner(problem),Inputs.weight_cells(problem,storage),output)}),staged),result_exact) :
    {Tree.read(Tile.Tile16,_,U32.from_nat(query)) == Tree.read(Tile.Tile16,staged,U32.from_nat(query)) : Tile.Tile16}
  call_kept(limits,problem,storage,count,offset,base,local,band,output,groups_exact,groups_positive,group_room,positive,end_exact,span,
    staged,staged_depth,staged_balanced,staged_depth_ok,staged_room,query,before,inside)

# ---- What a pointwise problem gives every call ----

# Twice the groups of eight channels and two more fit in a capacity above
# the channels and seven.
def doubled_groups(+groups: Nat,+capacity: Nat,+positive: {Nat.is_lt(0n,groups) == True{} : Bool},
  +below: {Nat.is_lt(Nat.mul(Nat.sub(groups,1n),8n),capacity) == True{} : Bool},+seven: {Nat.is_le(7n,capacity) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.mul(groups,2n),2n),capacity) == True{} : Bool}:
  match groups:
    case 0n:
      Empty.absurd({Nat.is_le(Nat.add(Nat.mul(0n,2n),2n),capacity) == True{} : Bool},Order.below_zero(0n,positive))
    case 1n+ +rest:
      match rest:
        case 0n:
          Order.transitive(4n,7n,capacity,{==},seven)
        case 1n+ +more:
          +twice = Nat.mul(more,2n)
          +eight = Nat.mul(more,8n)
          +ordered = Order.transitive(Nat.add(twice,2n),Nat.add(eight,2n),Nat.add(eight,4n),Order.add_le(twice,eight,2n,RowMajor.mul_le_left(more,2n,8n,{==})),
            Order.offset_le(eight,2n,4n,{==}))
          +swapped = {%Order.commutative(eight,4n) : {Nat.is_le(Nat.add(twice,2n),_) == True{} : Bool} ordered : {Nat.is_le(Nat.add(twice,2n),Nat.add(4n,eight)) == True{} : Bool}}
          Order.transitive(Nat.add(Nat.mul(2n+more,2n),2n),Nat.mul(1n+more,8n),capacity,
            {swapped : {Nat.is_le(Nat.add(Nat.mul(2n+more,2n),2n),Nat.mul(1n+more,8n)) == True{} : Bool}},
            Bounds.strict_le(Nat.mul(1n+more,8n),capacity,below))

# The value of a word read back from the value it encodes.
def word_value(+value: Nat,+word: U32,+exact: {U32.to_nat(U32.from_nat(value)) == value : Nat},+same: {U32.from_nat(value) == word : U32}) ->
  {value == U32.to_nat(word) : Nat}:
  Equal.trans(Nat,value,U32.to_nat(U32.from_nat(value)),U32.to_nat(word),Equal.sym(Nat,U32.to_nat(U32.from_nat(value)),value,exact),
    Equal.cong(U32,Nat,encoded => U32.to_nat(encoded),U32.from_nat(value),word,same))

# The shape of a pointwise problem: a 1 x 1 kernel of stride 1 without
# padding over whole groups of sixteen input channels.
type PointShape<-problem: ConvSpec.Problem> is Data:
  PointShape{kernel_one: {Guards.kernel(problem) == 1n : Nat},stride_one: {Guards.stride(problem) == 1n : Nat},pad_zero: {Guards.pad(problem) == 0n : Nat},
    channels_exact: {Guards.input_channel_count(problem) == Nat.mul(U32.to_nat((Coordinates.native_inner(problem) / 16 : U32)),16n) : Nat}}

def point_shape(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +facts: KernelPointwise.PointFacts<BandCover.kernel_word(problem),BandCover.stride_word(problem),BandCover.padding_word(problem),Coordinates.native_inner(problem)>) ->
  PointShape<problem>:
  KernelPointwise.PointFacts{+kernel_one,+stride_one,+padding_zero,+whole} = facts
  +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  +kernel = word_value(Guards.kernel(problem),1,SupportedCoordinates.kernel_exact(limits,problem,support),kernel_one)
  +stride = word_value(Guards.stride(problem),1,Start.stride_word(limits,problem,support),stride_one)
  +pad = word_value(Guards.pad(problem),0,SupportedCoordinates.pad_exact(limits,problem,support),padding_zero)
  +channels = Guards.input_channel_count(problem)
  +inner = Coordinates.native_inner(problem)
  +inner_value = Coordinates.inner_exact(limits,problem,Guards.valid(problem,valid),support)
  +inner_channels = Equal.trans(Nat,U32.to_nat(inner),Nat.mul(channels,Nat.mul(1n,1n)),channels,
    %kernel : {U32.to_nat(inner) == Nat.mul(channels,Nat.mul(_,_)) : Nat} inner_value,Order.times_one(channels))
  +quotient = U32Division.quotient_exact(2147483648,{==},inner,16,channels,16n,inner_channels,{==},{==},{==})
  +remainder = U32Division.remainder_exact(2147483648,{==},inner,16,channels,16n,inner_channels,{==},{==},{==})
  +remainder_zero = Equal.trans(Nat,Nat.mod(channels,16n),U32.to_nat((inner % 16 : U32)),0n,Equal.sym(Nat,U32.to_nat((inner % 16 : U32)),Nat.mod(channels,16n),remainder),
    Equal.cong(U32,Nat,encoded => U32.to_nat(encoded),(inner % 16 : U32),0,whole))
  +whole_channels = {%remainder_zero : {channels == Nat.add(Nat.mul(Nat.div(channels,16n),16n),_) : Nat}
      DivisionOrder.quotient_exact(channels,16n,{==}) : {channels == Nat.add(Nat.mul(Nat.div(channels,16n),16n),0n) : Nat}}
  +whole_product = {%Addition.zero_right(Nat.mul(Nat.div(channels,16n),16n)) : {channels == _ : Nat} whole_channels : {channels == Nat.mul(Nat.div(channels,16n),16n) : Nat}}
  +channels_exact = {%Equal.sym(Nat,U32.to_nat((inner / 16 : U32)),Nat.div(channels,16n),quotient) : {channels == Nat.mul(_,16n) : Nat}
      whole_product : {channels == Nat.mul(U32.to_nat((inner / 16 : U32)),16n) : Nat}}
  PointShape{kernel,stride,pad,channels_exact}

# The weights of a valid problem: its output channels times its inner extent.
def weights_count(+problem: ConvSpec.Problem,+valid: Guards.Valid<problem>) ->
  {ConvSpec.length(Guards.weights(problem)) == Nat.mul(Guards.output_channel_count(problem),Coordinates.inner(problem)) : Nat}:
  Guards.Valid{v0,v1,v2,v3,v4,v5,v6,v7,v8,+v9,v10} = valid
  Order.equal_of_is_eq(ConvSpec.length(Guards.weights(problem)),Nat.mul(Guards.output_channel_count(problem),Coordinates.inner(problem)),v9)

# The words of the pairs of the weights and the tree they are built in.
def pair_words(+problem: ConvSpec.Problem) -> U32:
  (Gemm.pairs_of(MicroLeaf.rows(problem)) * Coordinates.native_inner(problem) : U32)

def pair_source(+problem: ConvSpec.Problem) -> Tree.Tree<Tile.Tile16>:
  Tree.fresh(Tile.Tile16,Tensors.depth_for(pair_words(problem)),Tile.filled_pair(0.0))

# What every call of a pointwise problem relies on: its shape, its groups,
# its inner extent and the pairs of its weights.
type PointReady<-limits: Machine.Limits,-problem: ConvSpec.Problem> is Data:
  PointReady{shape: PointShape<problem>,
    groups_exact: {U32.to_nat(Gemm.blocks(MicroLeaf.rows(problem))) == TiledValues.groups_of(Guards.output_channel_count(problem)) : Nat},
    inner_exact: {U32.to_nat(Coordinates.native_inner(problem)) == Coordinates.inner(problem) : Nat},
    count_exact: {PointBuffer.pair_count(MicroLeaf.rows(problem),Coordinates.native_inner(problem)) ==
      Nat.mul(U32.to_nat(Gemm.pairs_of(MicroLeaf.rows(problem))),Coordinates.inner(problem)) : Nat},
    balanced: Shape.Balanced(Tile.Tile16,Tensors.depth_for(pair_words(problem)),pair_source(problem)),
    limit: {Nat.is_le(Tensors.depth_for(pair_words(problem)),31n) == True{} : Bool},
    capacity_ok: {U32.is_le(Tree.capacity(Tile.Tile16,pair_source(problem)),2147483648) == True{} : Bool},
    room: {Nat.is_le(PointBuffer.pair_count(MicroLeaf.rows(problem),Coordinates.native_inner(problem)),U32.to_nat(Tree.capacity(Tile.Tile16,pair_source(problem)))) == True{} : Bool},
    groups_positive: {Nat.is_lt(0n,TiledValues.groups_of(Guards.output_channel_count(problem))) == True{} : Bool},
    group_room: {Nat.is_le(Nat.add(Nat.mul(TiledValues.groups_of(Guards.output_channel_count(problem)),2n),2n),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    pairs_fewer: {Nat.is_le(U32.to_nat(Gemm.pairs_of(MicroLeaf.rows(problem))),U32.to_nat(Gemm.blocks(MicroLeaf.rows(problem)))) == True{} : Bool}}

def point_ready(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +facts: KernelPointwise.PointFacts<BandCover.kernel_word(problem),BandCover.stride_word(problem),BandCover.padding_word(problem),Coordinates.native_inner(problem)>) ->
  PointReady<limits,problem>:
  +checked = Guards.valid(problem,valid)
  +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  +capacity = U32.to_nat(Machine.capacity(limits))
  +half = U32.to_nat(Machine.half_word(limits))
  +outputs = Guards.output_channel_count(problem)
  +groups = TiledValues.groups_of(outputs)
  +inner = Coordinates.inner(problem)
  +rows = MicroLeaf.rows(problem)
  +outputs_exact = SupportedCoordinates.co_exact(limits,problem,support)
  +outputs_room = Limits.dimension_capacity(limits,outputs,SupportedCoordinates.co_bound(limits,problem,support))
  +outputs_positive = Contents.valid_condition(problem,valid,9n)
  +seven = Bounds.as_nat_le(7,Machine.capacity(limits),Limits.seven_within_capacity(limits))
  +groups_exact = {%outputs_exact : {U32.to_nat(Gemm.blocks(rows)) == Ceiling.ceiling(_,8n) : Nat}
    Encoded.eight_blocks_exact(limits,rows,
      %Equal.sym(Nat,U32.to_nat(rows),outputs,outputs_exact) : {Nat.is_le(_,capacity) == True{} : Bool} outputs_room) :
    {U32.to_nat(Gemm.blocks(rows)) == groups : Nat}}
  +groups_positive = RowMajor.left_factor_positive(groups,8n,Bounds.nat_trans(0n,outputs,Nat.mul(groups,8n),outputs_positive,Ceiling.covers(outputs,8n,{==})))
  +groups_outputs = Ceiling.bounded(outputs,8n,{==})
  +groups_capacity = Order.transitive(groups,outputs,capacity,groups_outputs,outputs_room)
  +doubled = {%Equal.sym(Nat,Nat.add(capacity,capacity),Nat.double(capacity),Addition.double(capacity)) : {Nat.is_le(_,half) == True{} : Bool}
    Limits.double_bound(limits,capacity,Order.reflexive(capacity)) : {Nat.is_le(Nat.add(capacity,capacity),half) == True{} : Bool}}
  +sum_room = Order.transitive(Nat.add(groups,1n),Nat.add(capacity,capacity),half,
    Order.transitive(Nat.add(groups,1n),Nat.add(capacity,1n),Nat.add(capacity,capacity),Order.add_le(groups,capacity,1n,groups_capacity),
      Order.offset_le(capacity,1n,capacity,Order.transitive(1n,7n,capacity,{==},seven))),doubled)
  +pairs = Nat.div(Nat.add(groups,1n),2n)
  +pairs_exact = {U32Division.quotient_exact(Machine.half_word(limits),Limits.half_word_fits(limits),(Gemm.blocks(rows) + 1 : U32),2,Nat.add(groups,1n),2n,
      Arithmetic.sum_exact(Machine.half_word(limits),Limits.half_word_fits(limits),Gemm.blocks(rows),1,groups,1n,groups_exact,{==},sum_room),{==},
      Limits.double_bound(limits,2n,Order.transitive(2n,7n,capacity,{==},seven)),{==}) :
    {U32.to_nat(Gemm.pairs_of(rows)) == pairs : Nat}}
  +weights_size = {%weights_count(problem,checked) : {Nat.is_le(_,capacity) == True{} : Bool}
    NativeWeightRefinement.weights_bound(limits,problem,support) : {Nat.is_le(Nat.mul(outputs,inner),capacity) == True{} : Bool}}
  +pairs_room = {%Equal.sym(Nat,U32.to_nat(Gemm.pairs_of(rows)),pairs,pairs_exact) : {Nat.is_le(Nat.mul(_,inner),capacity) == True{} : Bool}
    Order.transitive(Nat.mul(pairs,inner),Nat.mul(outputs,inner),capacity,
      RowMajor.mul_le(pairs,outputs,inner,Order.transitive(pairs,groups,outputs,PointReadback.pairs_below(groups),groups_outputs)),weights_size) :
    {Nat.is_le(Nat.mul(U32.to_nat(Gemm.pairs_of(rows)),inner),capacity) == True{} : Bool}}
  +inner_exact = Coordinates.inner_exact(limits,problem,checked,support)
  +count_exact = Arithmetic.product_exact(Machine.capacity(limits),Limits.capacity_fits(limits),Gemm.pairs_of(rows),Coordinates.native_inner(problem),
    U32.to_nat(Gemm.pairs_of(rows)),inner,{==},inner_exact,pairs_room)
  +words = pair_words(problem)
  +words_room = {%Equal.sym(Nat,U32.to_nat(words),Nat.mul(U32.to_nat(Gemm.pairs_of(rows)),inner),count_exact) : {Nat.is_le(_,capacity) == True{} : Bool}
    pairs_room : {Nat.is_le(U32.to_nat(words),capacity) == True{} : Bool}}
  +sized = Depth.bounded(words,Order.word_le(words,Machine.capacity(limits),1073741824,U32Division.nat_le_native(words,Machine.capacity(limits),words_room),
    Limits.capacity_allocatable(limits)))
  +depth = Tensors.depth_for(words)
  +depth_ok = Depth.allocation_depth(words,sized)
  +balanced = Shape.fresh(Tile.Tile16,depth,Tile.filled_pair(0.0))
  +capacity_same = Shape.capacity(Tile.Tile16,depth,pair_source(problem),balanced)
  +capacity_ok = {%Equal.sym(U32,Tree.capacity(Tile.Tile16,pair_source(problem)),U32.shln(1,depth),capacity_same) : {U32.is_le(_,2147483648) == True{} : Bool}
    U32Shift.power_fits(depth,depth_ok) : {U32.is_le(Tree.capacity(Tile.Tile16,pair_source(problem)),2147483648) == True{} : Bool}}
  +room = {%Equal.sym(U32,Tree.capacity(Tile.Tile16,pair_source(problem)),U32.shln(1,depth),capacity_same) : {Nat.is_le(U32.to_nat(words),U32.to_nat(_)) == True{} : Bool}
    Depth.allocation_room(words,sized) : {Nat.is_le(U32.to_nat(words),U32.to_nat(Tree.capacity(Tile.Tile16,pair_source(problem)))) == True{} : Bool}}
  +below = Bounds.nat_trans(Nat.mul(Nat.sub(groups,1n),8n),outputs,capacity,Ceiling.previous_below(outputs,8n,outputs_positive,{==}),outputs_room)
  +pairs_groups = {%Equal.sym(Nat,U32.to_nat(Gemm.blocks(rows)),groups,groups_exact) : {Nat.is_le(pairs,_) == True{} : Bool}
    PointReadback.pairs_below(groups) : {Nat.is_le(pairs,U32.to_nat(Gemm.blocks(rows))) == True{} : Bool}}
  +pairs_fewer = {%Equal.sym(Nat,U32.to_nat(Gemm.pairs_of(rows)),pairs,pairs_exact) : {Nat.is_le(_,U32.to_nat(Gemm.blocks(rows))) == True{} : Bool}
    pairs_groups : {Nat.is_le(U32.to_nat(Gemm.pairs_of(rows)),U32.to_nat(Gemm.blocks(rows))) == True{} : Bool}}
  PointReady{point_shape(limits,problem,valid,supported,facts),groups_exact,inner_exact,count_exact,balanced,
    Order.transitive(depth,30n,31n,depth_ok,{==}),capacity_ok,room,groups_positive,doubled_groups(groups,capacity,groups_positive,below,seven),pairs_fewer}

# The record of a call of a pointwise problem, from what the problem gives
# every call.
def call_of(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +local: U32,+count: U32,+offset: U32,+staged: Tree.Tree<Tile.Tile16>,+result: Tree.Tree<Tile.Tile16>,+ready: PointReady<limits,problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+image: U32,+base: U32,+band: Buffer.Model<Tile.Tile8>,+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},+grid: {Nat.is_le(Nat.add(U32.to_nat(local),U32.to_nat(count)),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool},+image_room: {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(image),Cells.count(problem,Guards.height(problem))),Cells.count(problem,Guards.height(problem))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+start: Nat,+start_exact: {start == Nat.mul(U32.to_nat(image),Cells.count(problem,Guards.height(problem))) : Nat},+held: Band.HeldImage(~Holder,~value,holder,problem,storage,start),+reads: Skeleton.BandHolds(~Holder,~value,holder,band,base,Calls.band_first(problem,image,local),Calls.band_cells(problem,local,count)),+end_exact: {U32.to_nat((offset + count : U32)) == Nat.add(U32.to_nat(offset),U32.to_nat(count)) : Nat},+span: {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(offset),U32.to_nat(count)),TiledValues.groups_of(Guards.output_channel_count(problem))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+staged_depth: Nat,+staged_balanced: Shape.Balanced(Tile.Tile16,staged_depth,staged),+staged_depth_ok: {Nat.is_le(staged_depth,30n) == True{} : Bool},+staged_room: {Nat.is_le(Nat.mul(U32.to_nat((offset + count : U32)),U32.to_nat(Gemm.pairs_of(MicroLeaf.rows(problem)))),U32.to_nat(Tree.capacity(Tile.Tile16,staged))) == True{} : Bool},+result_exact: {result == Writes.pure(Tile.Tile16,PointJobs.all_writes(U32.to_nat(Gemm.point_count(Gemm.PointMatrix{Gemm.TileMatrix{MicroLeaf.rows(problem),count,Coordinates.native_inner(problem),offset,(offset + count : U32)},(base + KernelWindow.start(local,1,BandCover.width_word(problem),BandCover.first_word(problem,local)) * Calls.groups_in(problem) : U32),Calls.groups_in(problem)})),0,Gemm.PointMatrix{Gemm.TileMatrix{MicroLeaf.rows(problem),count,Coordinates.native_inner(problem),offset,(offset + count : U32)},(base + KernelWindow.start(local,1,BandCover.width_word(problem),BandCover.first_word(problem,local)) * Calls.groups_in(problem) : U32),Calls.groups_in(problem)},PointJobs.Readers{Inputs.bias_cells(problem,storage),Buffer.tree(~Tile.Tile8,band),PointBuffer.pair_loop(MicroLeaf.rows(problem),Coordinates.native_inner(problem),Inputs.weight_cells(problem,storage),pair_source(problem))}),staged) : Tree.Tree<Tile.Tile16>}) ->
  PointCall<Holder,value,holder,limits,problem,storage,local,count,offset,staged,result>:
  PointReady{+shape,+groups_exact,+inner_exact,+count_exact,+balanced,+limit,+capacity_ok,+room,+groups_positive,+group_room,pairs_fewer} = ready
  PointShape{+kernel_one,+stride_one,+pad_zero,+channels_exact} = shape
  PointCall{valid,supported,kernel_one,stride_one,pad_zero,channels_exact,groups_exact,inner_exact,count_exact,pair_source(problem),Tensors.depth_for(pair_words(problem)),balanced,limit,capacity_ok,room,image,base,band,positive,grid,image_room,start,start_exact,held,reads,groups_positive,group_room,end_exact,span,staged_depth,staged_balanced,staged_depth_ok,staged_room,result_exact}
