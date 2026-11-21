# A call of the packed window kernel on a band is ready: the planes of the
# call have room by the packed guard, its blocks are scheduled from zero and
# its two workspaces are balanced and deep enough.
import Base
import ../specification.bend as ConvSpec
import ../inference.bend as NativeConv
import ../../lib/machine_limits.bend as Machine
import ../../lib/kernel_tile.bend as Tile
import ../../lib/kernel_window.bend as Window
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/kernel_spatial_packing.bend as Packing
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/kernel_pipeline_model.bend as Model
import ../../lib/proofs/kernel_pipeline_bounds.bend as BlockBounds
import ../../lib/proofs/kernel_spatial_packing_call.bend as Call
import ../../lib/proofs/kernel_spatial_packing_planes.bend as Planes
import ../../lib/proofs/kernel_spatial_packing_pixels.bend as Pixels
import ../../lib/proofs/kernel_spatial_packing_passes.bend as Passes
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/nat_division.bend as Division
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/nat_extrema.bend as NatExtrema
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/u32_extrema.bend as Extrema
import ../../lib/proofs/u32_shift.bend as Shift
import ../../lib/proofs/u32_division.bend as U32Division
import ../../lib/proofs/tensor_allocation_depth.bend as Depth
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./supported_coordinate_bounds.bend as SupportedCoordinates
import ./reduction_coordinate_bounds.bend as Coordinates
import ./packing_tile_bounds.bend as PackingBounds
import ./window_contents.bend as Contents
import ./window_band_cover.bend as BandCover
import ./window_cell_values.bend as CellValues
import ./window_problem.bend as WindowProblem
import ./window_positions.bend as Positions
import ./window_call_evidence.bend as Evidence
import ./blocked_convolution_readback.bend as Blocked
import ./microkernel_leaf_readback.bend as MicroLeaf

# The stencil of a call on rows rows of the image of a problem held from
# cell base on.
def band_window(+problem: ConvSpec.Problem,+base: U32,+rows: U32) -> Packing.Stencil:
  Packing.Stencil{base,rows,U32.from_nat(Guards.width(problem)),U32.from_nat(Guards.kernel(problem)),U32.from_nat(Guards.kernel(problem)),U32.from_nat(Guards.stride(problem)),
    (0 - U32.from_nat(Guards.pad(problem)) : U32),(0 - U32.from_nat(Guards.pad(problem)) : U32),U32.from_nat(ConvSpec.out_width(problem)),Coordinates.native_inner(problem),
    Gemm.blocks((Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32))}

# The work of a call on positions [local, local + count) of an image written
# from output position offset on.
def band_work(+problem: ConvSpec.Problem,+base: U32,+local: U32,+count: U32,+offset: U32) -> Pipeline.WindowWork:
  Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + count : U32),band_window(problem,base,BandCover.rows_word(problem,local,count))}

# ---- Words below the loop cap ----

def allocation_room(+limits: Machine.Limits,+allocation_cap: U32,+cap: U32,+configuration: BlockBounds.Configuration<limits,8192,allocation_cap,cap>,+value: Nat,+room: {Nat.is_le(value,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(value,U32.to_nat(allocation_cap)) == True{} : Bool}:
  BlockBounds.Configuration{target_positive,target_room,target_span_room,allocation_fits,allocation_bound,+capacity_room,loop_fits,doubled} = configuration
  Order.transitive(value,U32.to_nat(Machine.capacity(limits)),U32.to_nat(allocation_cap),room,Bounds.as_nat_le(Machine.capacity(limits),allocation_cap,capacity_room))

def sum_room(+limits: Machine.Limits,+allocation_cap: U32,+cap: U32,+configuration: BlockBounds.Configuration<limits,8192,allocation_cap,cap>,+left: Nat,+right: Nat,+left_room: {Nat.is_le(left,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +right_room: {Nat.is_le(right,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {Nat.is_le(Nat.add(left,right),U32.to_nat(cap)) == True{} : Bool}:
  BlockBounds.Configuration{target_positive,target_room,target_span_room,allocation_fits,allocation_bound,capacity_room,loop_fits,+doubled} = configuration
  BlockBounds.sum_room(left,right,allocation_cap,cap,allocation_room(limits,allocation_cap,cap,configuration,left,left_room),
    allocation_room(limits,allocation_cap,cap,configuration,right,right_room),doubled)

def word_room(+limits: Machine.Limits,+allocation_cap: U32,+cap: U32,+configuration: BlockBounds.Configuration<limits,8192,allocation_cap,cap>,+value: Nat,+room: {Nat.is_le(value,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(value,U32.to_nat(cap)) == True{} : Bool}:
  BlockBounds.Configuration{target_positive,target_room,target_span_room,allocation_fits,allocation_bound,capacity_room,loop_fits,+doubled} = configuration
  %Equal.sym(Nat,U32.to_nat(cap),Nat.double(U32.to_nat(allocation_cap)),doubled) : {Nat.is_le(value,_) == True{} : Bool}
  Order.transitive(value,U32.to_nat(allocation_cap),Nat.double(U32.to_nat(allocation_cap)),allocation_room(limits,allocation_cap,cap,configuration,value,room),
    Order.le_double(U32.to_nat(allocation_cap)))

def double_room(+limits: Machine.Limits,+allocation_cap: U32,+cap: U32,+configuration: BlockBounds.Configuration<limits,8192,allocation_cap,cap>,+value: Nat,+room: {Nat.is_le(value,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(Nat.double(value),U32.to_nat(cap)) == True{} : Bool}:
  %Addition.double(value) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
  sum_room(limits,allocation_cap,cap,configuration,value,value,room,room)

def swap_factors(+first: Nat,+second: Nat,+third: Nat) -> {Nat.mul(Nat.mul(first,second),third) == Nat.mul(Nat.mul(first,third),second) : Nat}:
  Equal.trans(Nat,Nat.mul(Nat.mul(first,second),third),Nat.mul(first,Nat.mul(second,third)),Nat.mul(Nat.mul(first,third),second),
    RowMajor.assoc(first,second,third),
    %RowMajor.multiply_commutative(third,second) : {Nat.mul(first,_) == Nat.mul(Nat.mul(first,third),second) : Nat}
    Equal.sym(Nat,Nat.mul(Nat.mul(first,third),second),Nat.mul(first,Nat.mul(third,second)),RowMajor.assoc(first,third,second)))

def stride_positive(+problem: ConvSpec.Problem,+facts: Guards.Valid<problem>) -> {Nat.is_lt(0n,Guards.stride(problem)) == True{} : Bool}:
  Guards.Valid{channels,outputs,height,width,kernel,+stride,kernel_height,kernel_width,input,weights,bias} = facts
  stride

def planes_guard(+limits: Machine.Limits,+problem: ConvSpec.Problem,+fast: Support.FastSupported<limits,problem>) -> {Support.planes_bound(limits,problem) == True{} : Bool}:
  Support.FastSupported{scalar,packing,output,+planes} = fast
  planes

# ---- The stencil of a call ----

def width_word(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) -> {U32.to_nat(U32.from_nat(ConvSpec.out_width(problem))) == ConvSpec.out_width(problem) : Nat}:
  Coordinates.width_exact(limits,problem,Guards.valid(problem,valid),Padded.scalar(limits,problem,Support.fast(limits,problem,supported)))

def kernel_word(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) -> {U32.to_nat(U32.from_nat(Guards.kernel(problem))) == Guards.kernel(problem) : Nat}:
  SupportedCoordinates.kernel_exact(limits,problem,Padded.scalar(limits,problem,Support.fast(limits,problem,supported)))

def stride_word(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) -> {U32.to_nat(U32.from_nat(Guards.stride(problem))) == Guards.stride(problem) : Nat}:
  Bounds.roundtrip(Guards.stride(problem),Machine.dimension(limits),Limits.dimension_fits(limits),
    SupportedCoordinates.stride_bound(limits,problem,Padded.scalar(limits,problem,Support.fast(limits,problem,supported))))

def channels_word(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) -> {U32.to_nat((Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32)) == Guards.input_channel_count(problem) : Nat}:
  Coordinates.inner_channels(limits,problem,Guards.valid(problem,valid),Padded.scalar(limits,problem,Support.fast(limits,problem,supported)))

# What the stencil of every call of a problem has room for.
type StencilRoom<-limits: Machine.Limits,-problem: ConvSpec.Problem,-cap: U32> is Data:
  StencilRoom{sixteen: {Nat.is_le(16n,U32.to_nat(cap)) == True{} : Bool},seven: {Nat.is_le(7n,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},positive: {Nat.is_lt(0n,U32.to_nat(U32.from_nat(ConvSpec.out_width(problem)))) == True{} : Bool},
    width_bound: {Nat.is_le(Nat.double(U32.to_nat(U32.from_nat(ConvSpec.out_width(problem)))),U32.to_nat(cap)) == True{} : Bool},width_room: {Nat.is_le(U32.to_nat(U32.from_nat(ConvSpec.out_width(problem))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},lanes_room: {Nat.is_le(Nat.add(U32.to_nat(U32.from_nat(ConvSpec.out_width(problem))),8n),U32.to_nat(cap)) == True{} : Bool},
    stride_positive: {Nat.is_lt(0n,U32.to_nat(U32.from_nat(Guards.stride(problem)))) == True{} : Bool},stride_bound: {Nat.is_le(Nat.double(U32.to_nat(U32.from_nat(Guards.stride(problem)))),U32.to_nat(cap)) == True{} : Bool},taps_positive: {Nat.is_lt(0n,U32.to_nat(U32.from_nat(Guards.kernel(problem)))) == True{} : Bool},
    taps_room: {Nat.is_le(U32.to_nat(U32.from_nat(Guards.kernel(problem))),U32.to_nat(cap)) == True{} : Bool},channels_positive: {Nat.is_lt(0n,U32.to_nat((Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32))) == True{} : Bool},divisor_bound: {Nat.is_le(Nat.double(U32.to_nat((Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32))),U32.to_nat(cap)) == True{} : Bool}}

def stencil_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+allocation_cap: U32,+cap: U32,+configuration: BlockBounds.Configuration<limits,8192,allocation_cap,cap>) -> StencilRoom<limits,problem,cap>:
  +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  +capacity = Machine.capacity(limits)
  +ow = ConvSpec.out_width(problem)
  +k = Guards.kernel(problem)
  +s = Guards.stride(problem)
  +c = Guards.input_channel_count(problem)
  +ow_exact = width_word(limits,problem,valid,supported)
  +k_exact = kernel_word(limits,problem,valid,supported)
  +s_exact = stride_word(limits,problem,valid,supported)
  +c_exact = channels_word(limits,problem,valid,supported)
  +ow_cap = Coordinates.width_bound(limits,problem,Guards.valid(problem,valid),support)
  +k_cap = Limits.dimension_capacity(limits,k,SupportedCoordinates.kernel_bound(limits,problem,support))
  +s_cap = Limits.dimension_capacity(limits,s,SupportedCoordinates.stride_bound(limits,problem,support))
  +c_cap = Limits.dimension_capacity(limits,c,SupportedCoordinates.ci_bound(limits,problem,support))
  +sixteen_cap = Bounds.as_nat_le(16,capacity,Limits.sixteen_within_capacity(limits))
  StencilRoom{word_room(limits,allocation_cap,cap,configuration,16n,sixteen_cap),Bounds.as_nat_le(7,capacity,Limits.seven_within_capacity(limits)),
    {%Equal.sym(Nat,U32.to_nat(U32.from_nat(ConvSpec.out_width(problem))),ow,ow_exact) : {Nat.is_lt(0n,_) == True{} : Bool} Coordinates.width_positive(problem) : {Nat.is_lt(0n,U32.to_nat(U32.from_nat(ConvSpec.out_width(problem)))) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat(U32.from_nat(ConvSpec.out_width(problem))),ow,ow_exact) : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool} double_room(limits,allocation_cap,cap,configuration,ow,ow_cap) : {Nat.is_le(Nat.double(U32.to_nat(U32.from_nat(ConvSpec.out_width(problem)))),U32.to_nat(cap)) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat(U32.from_nat(ConvSpec.out_width(problem))),ow,ow_exact) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} ow_cap : {Nat.is_le(U32.to_nat(U32.from_nat(ConvSpec.out_width(problem))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat(U32.from_nat(ConvSpec.out_width(problem))),ow,ow_exact) : {Nat.is_le(Nat.add(_,8n),U32.to_nat(cap)) == True{} : Bool}
      sum_room(limits,allocation_cap,cap,configuration,ow,8n,ow_cap,Order.transitive(8n,16n,U32.to_nat(Machine.capacity(limits)),{==},sixteen_cap)) : {Nat.is_le(Nat.add(U32.to_nat(U32.from_nat(ConvSpec.out_width(problem))),8n),U32.to_nat(cap)) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.stride(problem))),s,s_exact) : {Nat.is_lt(0n,_) == True{} : Bool} stride_positive(problem,Guards.valid(problem,valid)) : {Nat.is_lt(0n,U32.to_nat(U32.from_nat(Guards.stride(problem)))) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.stride(problem))),s,s_exact) : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool} double_room(limits,allocation_cap,cap,configuration,s,s_cap) : {Nat.is_le(Nat.double(U32.to_nat(U32.from_nat(Guards.stride(problem)))),U32.to_nat(cap)) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.kernel(problem))),k,k_exact) : {Nat.is_lt(0n,_) == True{} : Bool} Coordinates.kernel_positive(problem,Guards.valid(problem,valid)) : {Nat.is_lt(0n,U32.to_nat(U32.from_nat(Guards.kernel(problem)))) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.kernel(problem))),k,k_exact) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} word_room(limits,allocation_cap,cap,configuration,k,k_cap) : {Nat.is_le(U32.to_nat(U32.from_nat(Guards.kernel(problem))),U32.to_nat(cap)) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat((Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32)),c,c_exact) : {Nat.is_lt(0n,_) == True{} : Bool} Contents.valid_condition(problem,valid,10n) : {Nat.is_lt(0n,U32.to_nat((Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32))) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat((Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32)),c,c_exact) : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool} double_room(limits,allocation_cap,cap,configuration,c,c_cap) : {Nat.is_le(Nat.double(U32.to_nat((Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32))),U32.to_nat(cap)) == True{} : Bool}}}

# The planes guard of a problem, for the planes of plane positions, read on
# the words of its stencil.
type PlaneFit<-limits: Machine.Limits,-problem: ConvSpec.Problem,-plane: U32> is Data:
  PlaneFit{fit: {KernelShape.planes_fit(U32.to_nat(Window.padded(plane)),U32.to_nat((Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32)),U32.to_nat(U32.from_nat(ConvSpec.out_width(problem))),U32.to_nat(U32.from_nat(Guards.kernel(problem))),U32.to_nat(U32.from_nat(Guards.stride(problem))),Machine.capacity(limits)) == True{} : Bool},
    plane_room: {Nat.is_le(U32.to_nat(plane),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},bound_room: {Nat.is_le(U32.to_nat(Window.padded(plane)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

# Planes of plane positions, padded within the padded grid.
def plane_fit_within(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+plane: U32,
  +plane_room: {Nat.is_le(U32.to_nat(plane),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +bound_le: {Nat.is_le(U32.to_nat(Window.padded(plane)),Support.padded(problem)) == True{} : Bool}) -> PlaneFit<limits,problem,plane>:
  ConvSpec.Problem{shape,+input_values,+weight_values,+bias_values} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input_values,weight_values,bias_values} : ConvSpec.Problem}
  +facts = Guards.valid(problem,valid)
  +fast = Support.fast(limits,problem,supported)
  +grid_padded = Support.padded(problem)
  +fewer = WindowProblem.planes_fewer(U32.to_nat(Window.padded(plane)),grid_padded,Guards.input_channel_count(problem),ConvSpec.out_width(problem),Guards.kernel(problem),
    Guards.stride(problem),Machine.capacity(limits),Coordinates.width_positive(problem),bound_le,planes_guard(limits,problem,fast))
  PlaneFit{
    %Equal.sym(Nat,U32.to_nat((Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32)),Guards.input_channel_count(problem),channels_word(limits,problem,valid,supported)) :
      {KernelShape.planes_fit(U32.to_nat(Window.padded(plane)),_,U32.to_nat(U32.from_nat(ConvSpec.out_width(problem))),U32.to_nat(U32.from_nat(Guards.kernel(problem))),U32.to_nat(U32.from_nat(Guards.stride(problem))),Machine.capacity(limits)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(ConvSpec.out_width(problem))),ConvSpec.out_width(problem),width_word(limits,problem,valid,supported)) :
      {KernelShape.planes_fit(U32.to_nat(Window.padded(plane)),Guards.input_channel_count(problem),_,U32.to_nat(U32.from_nat(Guards.kernel(problem))),U32.to_nat(U32.from_nat(Guards.stride(problem))),Machine.capacity(limits)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.kernel(problem))),Guards.kernel(problem),kernel_word(limits,problem,valid,supported)) :
      {KernelShape.planes_fit(U32.to_nat(Window.padded(plane)),Guards.input_channel_count(problem),ConvSpec.out_width(problem),_,U32.to_nat(U32.from_nat(Guards.stride(problem))),Machine.capacity(limits)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.stride(problem))),Guards.stride(problem),stride_word(limits,problem,valid,supported)) :
      {KernelShape.planes_fit(U32.to_nat(Window.padded(plane)),Guards.input_channel_count(problem),ConvSpec.out_width(problem),Guards.kernel(problem),_,Machine.capacity(limits)) == True{} : Bool}
    fewer,plane_room,Order.transitive(U32.to_nat(Window.padded(plane)),grid_padded,U32.to_nat(Machine.capacity(limits)),bound_le,Padded.padded_bound(limits,problem,facts,fast))}

def plane_fit(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+plane: U32,
  +plane_grid: {Nat.is_le(U32.to_nat(plane),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool}) -> PlaneFit<limits,problem,plane>:
  +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  +area = Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))
  +plane_room = Order.transitive(U32.to_nat(plane),area,U32.to_nat(Machine.capacity(limits)),plane_grid,Padded.spatial_bound_facts(limits,problem,Guards.valid(problem,valid),support))
  +bound_exact = Encoded.padded_exact(limits,plane,plane_room)
  plane_fit_within(limits,problem,valid,supported,plane,plane_room,
    %Equal.sym(Nat,U32.to_nat(Window.padded(plane)),Encoded.ceiling_eight(U32.to_nat(plane)),bound_exact) : {Nat.is_le(_,Support.padded(problem)) == True{} : Bool}
    Encoded.padded_le(U32.to_nat(plane),area,plane_grid))

# What the bound of the planes of plane positions gives a call: room for
# the rows and tiles plane_bound allots and for all planes.
type BoundRoom<-window: Packing.Stencil,-bound: U32,-limits: Machine.Limits,-cap: U32> is Data:
  BoundRoom{area: {Nat.is_le(Nat.mul(Call.bound_rows(window,bound),Nat.mul(Call.call_tiles(window),8n)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    pairs: {Nat.is_le(Nat.mul(Nat.mul(Planes.phases_of(window),Planes.channel_count(window)),Nat.mul(Call.bound_rows(window,bound),Call.call_tiles(window))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    tiles_room: {Nat.is_le(Nat.add(Nat.mul(Call.call_tiles(window),8n),7n),U32.to_nat(cap)) == True{} : Bool},
    planes_bound: {Nat.is_le(Nat.mul(Planes.phases_of(window),Planes.channel_count(window)),U32.to_nat(cap)) == True{} : Bool},
    rows_positive: {Nat.is_lt(0n,Call.bound_rows(window,bound)) == True{} : Bool},tiles_positive: {Nat.is_lt(0n,Call.call_tiles(window)) == True{} : Bool}}

def bound_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+allocation_cap: U32,+cap: U32,+configuration: BlockBounds.Configuration<limits,8192,allocation_cap,cap>,+base: U32,+rows: U32,+plane: U32,+words: StencilRoom<limits,problem,cap>,+fit: PlaneFit<limits,problem,plane>) ->
  BoundRoom<band_window(problem,base,rows),Window.padded(plane),limits,cap>:
  StencilRoom{sixteen,+seven,+positive,width_bound,width_room,lanes_room,stride_positive,stride_bound,+taps_positive,taps_room,channels_positive,divisor_bound} = words
  PlaneFit{+planes,plane_room,bound_room} = fit
  +window = band_window(problem,base,rows)
  +tiles = Call.call_tiles(window)
  +width = Nat.mul(tiles,8n)
  +rows_count = Call.bound_rows(window,Window.padded(plane))
  +area_fact = Nat.is_le(Nat.mul(rows_count,width),U32.to_nat(Machine.capacity(limits)))
  +pairs_fact = Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(U32.from_nat(Guards.stride(problem))),U32.to_nat((Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32))),Nat.mul(rows_count,tiles)),U32.to_nat(Machine.capacity(limits)))
  +area = {Boolean.left(area_fact,pairs_fact,planes) : {area_fact == True{} : Bool}}
  +pairs = {Boolean.right(area_fact,pairs_fact,planes) : {pairs_fact == True{} : Bool}}
  +tiles_positive = RowMajor.cancel_multiply_lt(0n,tiles,8n,Call.width_positive(window,positive))
  +rows_positive = Bounds.nat_trans(0n,U32.to_nat(U32.from_nat(Guards.kernel(problem))),rows_count,taps_positive,
    Order.right_le(Nat.mul(Nat.div(Nat.sub(Nat.add(U32.to_nat(Window.padded(plane)),U32.to_nat(U32.from_nat(ConvSpec.out_width(problem)))),1n),U32.to_nat(U32.from_nat(ConvSpec.out_width(problem)))),U32.to_nat(U32.from_nat(Guards.stride(problem)))),U32.to_nat(U32.from_nat(Guards.kernel(problem)))))
  +width_cap = Order.transitive(width,Nat.mul(rows_count,width),U32.to_nat(Machine.capacity(limits)),
    {%RowMajor.multiply_commutative(width,rows_count) : {Nat.is_le(width,_) == True{} : Bool}
      RowMajor.product_covers(width,rows_count,rows_positive) : {Nat.is_le(width,Nat.mul(rows_count,width)) == True{} : Bool}},area)
  +planes_count = Nat.mul(U32.to_nat(U32.from_nat(Guards.stride(problem))),U32.to_nat((Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32)))
  BoundRoom{area,pairs,sum_room(limits,allocation_cap,cap,configuration,width,7n,width_cap,seven),
    word_room(limits,allocation_cap,cap,configuration,planes_count,Order.transitive(planes_count,Nat.mul(planes_count,Nat.mul(rows_count,tiles)),U32.to_nat(Machine.capacity(limits)),
      RowMajor.product_covers(planes_count,Nat.mul(rows_count,tiles),Order.product_positive(rows_count,tiles,rows_positive,tiles_positive)),pairs)),
    rows_positive,tiles_positive}

# The pairs of the planes of calls of at most plane positions.
def plane_value(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+allocation_cap: U32,+cap: U32,+configuration: BlockBounds.Configuration<limits,8192,allocation_cap,cap>,
  +base: U32,+rows: U32,+plane: U32,+words: StencilRoom<limits,problem,cap>,+fit: PlaneFit<limits,problem,plane>,+bounds: BoundRoom<band_window(problem,base,rows),Window.padded(plane),limits,cap>) ->
  {U32.to_nat(Packing.plane_bound(band_window(problem,base,rows),Window.padded(plane))) ==
    Nat.mul(Nat.mul(Planes.phases_of(band_window(problem,base,rows)),Planes.channel_count(band_window(problem,base,rows))),
      Nat.mul(Call.bound_rows(band_window(problem,base,rows),Window.padded(plane)),Call.call_tiles(band_window(problem,base,rows)))) : Nat}:
  BlockBounds.Configuration{target_positive,target_room,target_span_room,allocation_fits,allocation_bound,capacity_room,+loop_fits,doubled} = configuration
  StencilRoom{+sixteen,seven,+positive_width,+width_bound,+width_room,lanes_room,+stride_positive,+stride_bound,+taps_positive,+taps_room,channels_positive,
    divisor_bound} = words
  PlaneFit{planes,plane_room,+bound_cap} = fit
  BoundRoom{+area,+pairs,+tiles_room,+planes_bound,rows_positive,tiles_positive} = bounds
  +window = band_window(problem,base,rows)
  +tiles = Call.call_tiles(window)
  +rows_count = Call.bound_rows(window,Window.padded(plane))
  +planes_count = Nat.mul(U32.to_nat(U32.from_nat(Guards.stride(problem))),U32.to_nat((Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32)))
  Call.bound_exact(window,Window.padded(plane),cap,loop_fits,sixteen,positive_width,width_bound,stride_positive,stride_bound,taps_positive,taps_room,tiles_room,
    sum_room(limits,allocation_cap,cap,configuration,U32.to_nat(Window.padded(plane)),U32.to_nat(U32.from_nat(ConvSpec.out_width(problem))),bound_cap,width_room),word_room(limits,allocation_cap,cap,configuration,Nat.mul(rows_count,Nat.mul(tiles,8n)),area),planes_bound,
    word_room(limits,allocation_cap,cap,configuration,Nat.mul(planes_count,Nat.mul(rows_count,tiles)),pairs))

# The planes of calls of at most plane positions fit an allocation.
def plane_capacity(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+base: U32,+rows: U32,+plane: U32,+words: StencilRoom<limits,problem,2147483648>,
  +fit: PlaneFit<limits,problem,plane>,+bounds: BoundRoom<band_window(problem,base,rows),Window.padded(plane),limits,2147483648>) ->
  {Nat.is_le(U32.to_nat(Packing.plane_bound(band_window(problem,base,rows),Window.padded(plane))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  BoundRoom{area,+pairs,tiles_room,planes_bound,rows_positive,tiles_positive} = bounds
  +window = band_window(problem,base,rows)
  %Equal.sym(Nat,U32.to_nat(Packing.plane_bound(window,Window.padded(plane))),Nat.mul(Nat.mul(Planes.phases_of(window),Planes.channel_count(window)),Nat.mul(Call.bound_rows(window,Window.padded(plane)),Call.call_tiles(window))),
    plane_value(limits,problem,valid,supported,1073741824,2147483648,BlockBounds.configuration(limits),base,rows,plane,words,fit,bounds)) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  pairs

def plane_sized_within(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+base: U32,+rows: U32,+plane: U32,+fit: PlaneFit<limits,problem,plane>) ->
  Depth.Sized<Packing.plane_bound(band_window(problem,base,rows),Window.padded(plane))>:
  +configuration = BlockBounds.configuration(limits)
  +words = stencil_room(limits,problem,valid,supported,1073741824,2147483648,configuration)
  +value = Packing.plane_bound(band_window(problem,base,rows),Window.padded(plane))
  +room = plane_capacity(limits,problem,valid,supported,base,rows,plane,words,fit,bound_room(limits,problem,valid,supported,1073741824,2147483648,configuration,base,rows,plane,words,fit))
  Depth.bounded(value,Order.word_le(value,Machine.capacity(limits),1073741824,U32Division.nat_le_native(value,Machine.capacity(limits),room),Limits.capacity_allocatable(limits)))

def plane_sized(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+base: U32,+rows: U32,+plane: U32,
  +plane_grid: {Nat.is_le(U32.to_nat(plane),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool}) ->
  Depth.Sized<Packing.plane_bound(band_window(problem,base,rows),Window.padded(plane))>:
  plane_sized_within(limits,problem,valid,supported,base,rows,plane,plane_fit(limits,problem,valid,supported,plane,plane_grid))

# The workspace of a call on any positions fits its allocation.
def workspace_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+positions: U32,+target: U32,+allocation_cap: U32,+loop_cap: U32,
  +configuration: BlockBounds.Configuration<limits,target,allocation_cap,loop_cap>) ->
  {Nat.is_le(U32.to_nat(Pipeline.workspace_tiles(target,Coordinates.native_inner(problem),positions)),U32.to_nat(allocation_cap)) == True{} : Bool}:
  BlockBounds.Configuration{+target_positive,+target_room,+target_span_room,+allocation_fits,+allocation_bound,+capacity_room,+loop_fits,+doubled} = configuration
  +inner_bound = Evidence.inner_room(limits,problem,Guards.valid(problem,valid),Support.fast(limits,problem,supported))
  +inner_positive = Evidence.inner_positive(limits,problem,Guards.valid(problem,valid),Support.fast(limits,problem,supported))
  +span_exact = BlockBounds.span_exact(target,target_positive,limits,Coordinates.native_inner(problem),allocation_cap,allocation_fits,target_span_room,inner_positive,inner_bound)
  +prefix_room = BlockBounds.tile_prefix_room(target,limits,Coordinates.native_inner(problem),inner_bound,inner_positive)
  +max_room = BlockBounds.tile_count_room(target,Coordinates.native_inner(problem),allocation_cap,target_room,
    Order.transitive(U32.to_nat(Coordinates.native_inner(problem)),U32.to_nat(Machine.capacity(limits)),U32.to_nat(allocation_cap),inner_bound,Bounds.as_nat_le(Machine.capacity(limits),allocation_cap,capacity_room)))
  +tiles_exact = BlockBounds.workspace_count_exact(target,Coordinates.native_inner(problem),positions,allocation_cap,allocation_fits,span_exact,prefix_room,max_room)
  BlockBounds.workspace_count_room(target,Coordinates.native_inner(problem),positions,allocation_cap,tiles_exact,span_exact,prefix_room,max_room)

def workspace_sized(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+positions: U32) -> Depth.Sized<Pipeline.workspace_tiles(8192,Coordinates.native_inner(problem),positions)>:
  Depth.bounded(Pipeline.workspace_tiles(8192,Coordinates.native_inner(problem),positions),
    U32Division.nat_le_native(Pipeline.workspace_tiles(8192,Coordinates.native_inner(problem),positions),1073741824,
      workspace_room(limits,problem,valid,supported,positions,8192,1073741824,2147483648,BlockBounds.configuration(limits))))

# ---- A call ----

# The planes of a call of count positions from start of a stencil on rows
# rows of the image have room by the planes guard, as planes of room for
# plane positions.
# The room of the planes of a call on count positions from start whose pairs
# have depth pair_depth, from the words of its pairs.
def call_room_with(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+allocation_cap: U32,+cap: U32,+configuration: BlockBounds.Configuration<limits,8192,allocation_cap,cap>,+base: U32,+rows: U32,+start: U32,+count: U32,+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
  +pack_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(Window.padded(count))),U32.to_nat(cap)) == True{} : Bool},+plane: U32,+count_le: {Nat.is_le(U32.to_nat(count),U32.to_nat(plane)) == True{} : Bool},
  +pair_depth: Nat,+pair_depth_ok: {Nat.is_le(pair_depth,30n) == True{} : Bool},+pair_power: {U32.is_le(U32.shln(1,pair_depth),cap) == True{} : Bool},
  +fits: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(U32.from_nat(Guards.stride(problem))),Nat.mul(Call.call_rows(band_window(problem,base,rows),start,Window.padded(count)),Call.call_tiles(band_window(problem,base,rows)))),U32.to_nat((Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32))),U32.to_nat(U32.shln(1,pair_depth))) == True{} : Bool},
  +words: StencilRoom<limits,problem,cap>,+fit: PlaneFit<limits,problem,plane>,+bounds: BoundRoom<band_window(problem,base,rows),Window.padded(plane),limits,cap>) ->
  Call.CallRoom<band_window(problem,base,rows),start,Window.padded(count),pair_depth,cap>:
  BlockBounds.Configuration{target_positive,target_room,target_span_room,allocation_fits,allocation_bound,capacity_room,+loop_fits,doubled} = configuration
  StencilRoom{+sixteen,seven,+positive_width,+width_bound,+width_room,+lanes_room,+stride_positive,+stride_bound,+taps_positive,+taps_room,+channels_positive,
    +divisor_bound} = words
  PlaneFit{planes,+plane_room,+bound_cap} = fit
  BoundRoom{+area,+pairs,+tiles_room,+planes_bound,rows_positive,tiles_positive} = bounds
  +window = band_window(problem,base,rows)
  +padded = Window.padded(count)
  +n = U32.to_nat(count)
  +tiles = Call.call_tiles(window)
  +rows_count = Call.bound_rows(window,Window.padded(plane))
  +padded_exact = Encoded.padded_exact(limits,count,Order.transitive(n,U32.to_nat(plane),U32.to_nat(Machine.capacity(limits)),count_le,plane_room))
  +padded_positive = {%Equal.sym(Nat,U32.to_nat(padded),Encoded.ceiling_eight(n),padded_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    Bounds.nat_trans(0n,n,Encoded.ceiling_eight(n),positive,Encoded.within_ceiling(n)) : {Nat.is_lt(0n,U32.to_nat(padded)) == True{} : Bool}}
  +ordered = {%Equal.sym(Nat,U32.to_nat(padded),Encoded.ceiling_eight(n),padded_exact) : {Nat.is_le(_,U32.to_nat(Window.padded(plane))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(Window.padded(plane)),Encoded.ceiling_eight(U32.to_nat(plane)),Encoded.padded_exact(limits,plane,plane_room)) : {Nat.is_le(Encoded.ceiling_eight(n),_) == True{} : Bool}
    Encoded.padded_le(n,U32.to_nat(plane),count_le) : {Nat.is_le(U32.to_nat(padded),U32.to_nat(Window.padded(plane))) == True{} : Bool}}
  +call_rows = Call.call_rows(window,start,padded)
  +rows_le = Call.rows_within(window,start,padded,Window.padded(plane),positive_width,padded_positive,ordered)
  Call.CallRoom{loop_fits,sixteen,positive_width,width_bound,stride_positive,stride_bound,taps_positive,taps_positive,lanes_room,taps_room,tiles_room,channels_positive,
    divisor_bound,planes_bound,padded_positive,pack_room,
    word_room(limits,allocation_cap,cap,configuration,Nat.mul(call_rows,Nat.mul(tiles,8n)),
      Order.transitive(Nat.mul(call_rows,Nat.mul(tiles,8n)),Nat.mul(rows_count,Nat.mul(tiles,8n)),U32.to_nat(Machine.capacity(limits)),RowMajor.mul_le(call_rows,rows_count,Nat.mul(tiles,8n),rows_le),area)),
    Order.transitive(pair_depth,30n,31n,pair_depth_ok,{==}),Bounds.as_nat_le(U32.shln(1,pair_depth),cap,pair_power),fits}

# The room of a call whose pairs hold the planes of any plane positions.
def call_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+allocation_cap: U32,+cap: U32,+configuration: BlockBounds.Configuration<limits,8192,allocation_cap,cap>,+base: U32,+rows: U32,+start: U32,+count: U32,+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
  +pack_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(Window.padded(count))),U32.to_nat(cap)) == True{} : Bool},+plane: U32,+count_le: {Nat.is_le(U32.to_nat(count),U32.to_nat(plane)) == True{} : Bool},
  +pair_depth: Nat,+pair_depth_ok: {Nat.is_le(pair_depth,30n) == True{} : Bool},+pair_power: {U32.is_le(U32.shln(1,pair_depth),cap) == True{} : Bool},
  +pair_need: {Nat.is_le(U32.to_nat(Packing.plane_bound(band_window(problem,base,rows),Window.padded(plane))),U32.to_nat(U32.shln(1,pair_depth))) == True{} : Bool},
  +words: StencilRoom<limits,problem,cap>,+fit: PlaneFit<limits,problem,plane>,+bounds: BoundRoom<band_window(problem,base,rows),Window.padded(plane),limits,cap>) ->
  Call.CallRoom<band_window(problem,base,rows),start,Window.padded(count),pair_depth,cap>:
  StencilRoom{sixteen,seven,+positive_width,width_bound,width_room,lanes_room,stride_positive,stride_bound,taps_positive,taps_room,channels_positive,divisor_bound} = words
  PlaneFit{planes,+plane_room,bound_cap} = fit
  +window = band_window(problem,base,rows)
  +n = U32.to_nat(count)
  +tiles = Call.call_tiles(window)
  +rows_count = Call.bound_rows(window,Window.padded(plane))
  +padded_exact = Encoded.padded_exact(limits,count,Order.transitive(n,U32.to_nat(plane),U32.to_nat(Machine.capacity(limits)),count_le,plane_room))
  +padded_positive = {%Equal.sym(Nat,U32.to_nat(Window.padded(count)),Encoded.ceiling_eight(n),padded_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    Bounds.nat_trans(0n,n,Encoded.ceiling_eight(n),positive,Encoded.within_ceiling(n)) : {Nat.is_lt(0n,U32.to_nat(Window.padded(count))) == True{} : Bool}}
  +ordered = {%Equal.sym(Nat,U32.to_nat(Window.padded(count)),Encoded.ceiling_eight(n),padded_exact) : {Nat.is_le(_,U32.to_nat(Window.padded(plane))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(Window.padded(plane)),Encoded.ceiling_eight(U32.to_nat(plane)),Encoded.padded_exact(limits,plane,plane_room)) : {Nat.is_le(Encoded.ceiling_eight(n),_) == True{} : Bool}
    Encoded.padded_le(n,U32.to_nat(plane),count_le) : {Nat.is_le(U32.to_nat(Window.padded(count)),U32.to_nat(Window.padded(plane))) == True{} : Bool}}
  +call_rows = Call.call_rows(window,start,Window.padded(count))
  +rows_le = Call.rows_within(window,start,Window.padded(count),Window.padded(plane),positive_width,padded_positive,ordered)
  +planes_count = Nat.mul(U32.to_nat(U32.from_nat(Guards.stride(problem))),U32.to_nat((Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32)))
  +bound_value = plane_value(limits,problem,valid,supported,allocation_cap,cap,configuration,base,rows,plane,words,fit,bounds)
  +power = U32.to_nat(U32.shln(1,pair_depth))
  +fits = {
    %Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(U32.from_nat(Guards.stride(problem))),Nat.mul(call_rows,tiles)),U32.to_nat((Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32))),Nat.mul(planes_count,Nat.mul(call_rows,tiles)),swap_factors(U32.to_nat(U32.from_nat(Guards.stride(problem))),Nat.mul(call_rows,tiles),U32.to_nat((Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32)))) :
      {Nat.is_le(_,power) == True{} : Bool}
    Order.transitive(Nat.mul(planes_count,Nat.mul(call_rows,tiles)),Nat.mul(planes_count,Nat.mul(rows_count,tiles)),power,
      DivisionOrder.multiply_left_le(planes_count,Nat.mul(call_rows,tiles),Nat.mul(rows_count,tiles),RowMajor.mul_le(call_rows,rows_count,tiles,rows_le)),
      %bound_value : {Nat.is_le(_,power) == True{} : Bool}
      pair_need) : {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(U32.from_nat(Guards.stride(problem))),Nat.mul(call_rows,tiles)),U32.to_nat((Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32))),power) == True{} : Bool}}
  call_room_with(limits,problem,valid,supported,allocation_cap,cap,configuration,base,rows,start,count,positive,pack_room,plane,count_le,pair_depth,pair_depth_ok,pair_power,fits,
    words,fit,bounds)

# The pairs of the planes of a call on count positions from start: their
# words exactly, and the allocation that holds them.
type PairsRoom<-limits: Machine.Limits,-problem: ConvSpec.Problem,-base: U32,-rows: U32,-start: U32,-count: U32> is Data:
  PairsRoom{exact: {U32.to_nat(Packing.plane_pairs(band_window(problem,base,rows),Packing.planes_of(band_window(problem,base,rows),start,Window.padded(count)))) == Nat.mul(Nat.mul(Planes.phases_of(band_window(problem,base,rows)),Planes.channel_count(band_window(problem,base,rows))),Nat.mul(Call.call_rows(band_window(problem,base,rows),start,Window.padded(count)),Call.call_tiles(band_window(problem,base,rows)))) : Nat},
    sized: Depth.Sized<Packing.plane_pairs(band_window(problem,base,rows),Packing.planes_of(band_window(problem,base,rows),start,Window.padded(count)))>}

def call_pairs(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+allocation_cap: U32,+cap: U32,+configuration: BlockBounds.Configuration<limits,8192,allocation_cap,cap>,+base: U32,+rows: U32,+start: U32,+count: U32,+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
  +pack_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(Window.padded(count))),U32.to_nat(cap)) == True{} : Bool},+plane: U32,+count_le: {Nat.is_le(U32.to_nat(count),U32.to_nat(plane)) == True{} : Bool},
  +words: StencilRoom<limits,problem,cap>,+fit: PlaneFit<limits,problem,plane>,+bounds: BoundRoom<band_window(problem,base,rows),Window.padded(plane),limits,cap>) ->
  PairsRoom<limits,problem,base,rows,start,count>:
  BlockBounds.Configuration{target_positive,target_room,target_span_room,allocation_fits,allocation_bound,capacity_room,+loop_fits,doubled} = configuration
  StencilRoom{+sixteen,seven,+positive_width,+width_bound,width_room,lanes_room,+stride_positive,+stride_bound,+taps_positive,+taps_room,channels_positive,divisor_bound} = words
  PlaneFit{planes,+plane_room,bound_cap} = fit
  BoundRoom{+area,+pairs,+tiles_room,+planes_bound,rows_positive,tiles_positive} = bounds
  +window = band_window(problem,base,rows)
  +padded = Window.padded(count)
  +n = U32.to_nat(count)
  +tiles = Call.call_tiles(window)
  +rows_count = Call.bound_rows(window,Window.padded(plane))
  +padded_exact = Encoded.padded_exact(limits,count,Order.transitive(n,U32.to_nat(plane),U32.to_nat(Machine.capacity(limits)),count_le,plane_room))
  +padded_positive = {%Equal.sym(Nat,U32.to_nat(padded),Encoded.ceiling_eight(n),padded_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    Bounds.nat_trans(0n,n,Encoded.ceiling_eight(n),positive,Encoded.within_ceiling(n)) : {Nat.is_lt(0n,U32.to_nat(padded)) == True{} : Bool}}
  +ordered = {%Equal.sym(Nat,U32.to_nat(padded),Encoded.ceiling_eight(n),padded_exact) : {Nat.is_le(_,U32.to_nat(Window.padded(plane))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(Window.padded(plane)),Encoded.ceiling_eight(U32.to_nat(plane)),Encoded.padded_exact(limits,plane,plane_room)) : {Nat.is_le(Encoded.ceiling_eight(n),_) == True{} : Bool}
    Encoded.padded_le(n,U32.to_nat(plane),count_le) : {Nat.is_le(U32.to_nat(padded),U32.to_nat(Window.padded(plane))) == True{} : Bool}}
  +call_rows = Call.call_rows(window,start,padded)
  +rows_le = Call.rows_within(window,start,padded,Window.padded(plane),positive_width,padded_positive,ordered)
  +planes_count = Nat.mul(Planes.phases_of(window),Planes.channel_count(window))
  +area_room = word_room(limits,allocation_cap,cap,configuration,Nat.mul(call_rows,Nat.mul(tiles,8n)),
    Order.transitive(Nat.mul(call_rows,Nat.mul(tiles,8n)),Nat.mul(rows_count,Nat.mul(tiles,8n)),U32.to_nat(Machine.capacity(limits)),RowMajor.mul_le(call_rows,rows_count,Nat.mul(tiles,8n),rows_le),area))
  +pairs_capacity = Order.transitive(Nat.mul(planes_count,Nat.mul(call_rows,tiles)),Nat.mul(planes_count,Nat.mul(rows_count,tiles)),U32.to_nat(Machine.capacity(limits)),
    DivisionOrder.multiply_left_le(planes_count,Nat.mul(call_rows,tiles),Nat.mul(rows_count,tiles),RowMajor.mul_le(call_rows,rows_count,tiles,rows_le)),pairs)
  +exact = Call.pairs_exact(window,start,padded,cap,loop_fits,sixteen,positive_width,width_bound,stride_positive,stride_bound,taps_positive,taps_room,tiles_room,padded_positive,
    pack_room,area_room,planes_bound,word_room(limits,allocation_cap,cap,configuration,Nat.mul(planes_count,Nat.mul(call_rows,tiles)),pairs_capacity))
  +value = Packing.plane_pairs(band_window(problem,base,rows),Packing.planes_of(band_window(problem,base,rows),start,Window.padded(count)))
  +room = {%Equal.sym(Nat,U32.to_nat(value),Nat.mul(planes_count,Nat.mul(call_rows,tiles)),exact) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    pairs_capacity : {Nat.is_le(U32.to_nat(value),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  PairsRoom{exact,Depth.bounded(value,Order.word_le(value,Machine.capacity(limits),1073741824,U32Division.nat_le_native(value,Machine.capacity(limits),room),
    Limits.capacity_allocatable(limits)))}

def pairs_exact(-limits: Machine.Limits,-problem: ConvSpec.Problem,-base: U32,-rows: U32,-start: U32,-count: U32,room: PairsRoom<limits,problem,base,rows,start,count>) ->
  {U32.to_nat(Packing.plane_pairs(band_window(problem,base,rows),Packing.planes_of(band_window(problem,base,rows),start,Window.padded(count)))) == Nat.mul(Nat.mul(Planes.phases_of(band_window(problem,base,rows)),Planes.channel_count(band_window(problem,base,rows))),Nat.mul(Call.call_rows(band_window(problem,base,rows),start,Window.padded(count)),Call.call_tiles(band_window(problem,base,rows)))) : Nat}:
  PairsRoom{exact,sized} = room
  exact

def pairs_sized(-limits: Machine.Limits,-problem: ConvSpec.Problem,-base: U32,-rows: U32,-start: U32,-count: U32,room: PairsRoom<limits,problem,base,rows,start,count>) ->
  Depth.Sized<Packing.plane_pairs(band_window(problem,base,rows),Packing.planes_of(band_window(problem,base,rows),start,Window.padded(count)))>:
  PairsRoom{exact,sized} = room
  sized

# ---- The tiles of a call ----

# The entries of a pass: the channels times the kernel rows.
def entries_value(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+allocation_cap: U32,+cap: U32,+configuration: BlockBounds.Configuration<limits,8192,allocation_cap,cap>) -> {U32.to_nat(((Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32) * U32.from_nat(Guards.kernel(problem)) : U32)) == Nat.mul(Guards.input_channel_count(problem),Guards.kernel(problem)) : Nat}:
  BlockBounds.Configuration{target_positive,target_room,target_span_room,allocation_fits,allocation_bound,capacity_room,+loop_fits,doubled} = configuration
  +facts = Guards.valid(problem,valid)
  +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  +c = Guards.input_channel_count(problem)
  +k = Guards.kernel(problem)
  Arithmetic.product_exact(cap,loop_fits,(Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32),U32.from_nat(k),c,k,channels_word(limits,problem,valid,supported),kernel_word(limits,problem,valid,supported),
    word_room(limits,allocation_cap,cap,configuration,Nat.mul(c,k),Order.transitive(Nat.mul(c,k),Coordinates.inner(problem),U32.to_nat(Machine.capacity(limits)),
      DivisionOrder.multiply_left_le(c,k,Nat.mul(k,k),RowMajor.product_covers(k,k,Coordinates.kernel_positive(problem,facts))),Coordinates.inner_bound(limits,problem,facts,support))))

# The reduction extent is the entries of a pass times the kernel columns.
def inner_entries(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+allocation_cap: U32,+cap: U32,+configuration: BlockBounds.Configuration<limits,8192,allocation_cap,cap>) ->
  {U32.to_nat(Coordinates.native_inner(problem)) == Nat.mul(U32.to_nat(((Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32) * U32.from_nat(Guards.kernel(problem)) : U32)),U32.to_nat(U32.from_nat(Guards.kernel(problem)))) : Nat}:
  +c = Guards.input_channel_count(problem)
  +k = Guards.kernel(problem)
  +entries = U32.to_nat(((Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32) * U32.from_nat(k) : U32))
  +value = entries_value(limits,problem,valid,supported,allocation_cap,cap,configuration)
  Equal.trans(Nat,U32.to_nat(Coordinates.native_inner(problem)),Coordinates.inner(problem),Nat.mul(entries,U32.to_nat(U32.from_nat(Guards.kernel(problem)))),
    Coordinates.inner_exact(limits,problem,Guards.valid(problem,valid),Padded.scalar(limits,problem,Support.fast(limits,problem,supported))),
    Equal.trans(Nat,Coordinates.inner(problem),Nat.mul(Nat.mul(c,k),k),Nat.mul(entries,U32.to_nat(U32.from_nat(Guards.kernel(problem)))),Equal.sym(Nat,Nat.mul(Nat.mul(c,k),k),Coordinates.inner(problem),RowMajor.assoc(c,k,k)),
      Equal.sym(Nat,Nat.mul(entries,U32.to_nat(U32.from_nat(Guards.kernel(problem)))),Nat.mul(Nat.mul(c,k),k),{
        %Equal.sym(Nat,entries,Nat.mul(c,k),value) : {Nat.mul(_,U32.to_nat(U32.from_nat(Guards.kernel(problem)))) == Nat.mul(Nat.mul(c,k),k) : Nat}
        %Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.kernel(problem))),k,kernel_word(limits,problem,valid,supported)) : {Nat.mul(Nat.mul(c,k),_) == Nat.mul(Nat.mul(c,k),k) : Nat}
        {==} : {Nat.mul(entries,U32.to_nat(U32.from_nat(Guards.kernel(problem)))) == Nat.mul(Nat.mul(c,k),k) : Nat}})))

# The blocks of a call fit the tiles of a workspace of room for plane
# positions: the smaller of a span and the units of the call.
def fewest_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,+count: U32,+plane_room: {Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(Nat.min(Nat.div(U32.to_nat(Evidence.span(8192,problem)),8n),Nat.div(Nat.add(U32.to_nat(count),7n),8n)),PackingBounds.groups(U32.min(Evidence.span(8192,problem),Window.padded(count)))) == True{} : Bool}:
  +n = U32.to_nat(count)
  +units = Nat.div(Nat.add(U32.to_nat(count),7n),8n)
  +span_word = Evidence.span(8192,problem)
  +span_n = U32.to_nat(span_word)
  +padded_n = U32.to_nat(Window.padded(count))
  +fewest = Nat.min(Nat.div(span_n,8n),units)
  +within_span = Order.transitive(Nat.mul(fewest,8n),Nat.mul(Nat.div(span_n,8n),8n),span_n,RowMajor.mul_le(fewest,Nat.div(span_n,8n),8n,NatExtrema.minimum_left(Nat.div(span_n,8n),units)),
    DivisionOrder.product_below(span_n,8n,{==}))
  +within_padded = Order.transitive(Nat.mul(fewest,8n),Nat.mul(units,8n),padded_n,RowMajor.mul_le(fewest,units,8n,NatExtrema.minimum_right(Nat.div(span_n,8n),units)),
    {%Equal.sym(Nat,padded_n,Encoded.ceiling_eight(n),Encoded.padded_exact(limits,count,plane_room)) : {Nat.is_le(Nat.mul(units,8n),_) == True{} : Bool}
      Order.reflexive(Nat.mul(units,8n)) : {Nat.is_le(Nat.mul(units,8n),padded_n) == True{} : Bool}})
  %Equal.sym(Nat,U32.to_nat(U32.min(span_word,Window.padded(count))),Nat.min(span_n,padded_n),Extrema.minimum(span_word,Window.padded(count))) :
    {Nat.is_le(fewest,Nat.div(_,8n)) == True{} : Bool}
  %Division.exact_quotient(fewest,8n,{==}) : {Nat.is_le(_,Nat.div(Nat.min(span_n,padded_n),8n)) == True{} : Bool}
  DivisionOrder.quotient_le(Nat.mul(fewest,8n),Nat.min(span_n,padded_n),8n,{==},NatExtrema.minimum_lower(span_n,padded_n,Nat.mul(fewest,8n),within_span,within_padded))

# The tiles of units eight-position blocks of a stencil on rows rows of the
# image have room for their packing.
def stencil_packable(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+allocation_cap: U32,+cap: U32,+configuration: BlockBounds.Configuration<limits,8192,allocation_cap,cap>,
  +base: U32,+rows: U32,+units: Nat,+tile_depth: Nat,+tile_depth_ok: {Nat.is_le(tile_depth,30n) == True{} : Bool},+tile_power: {U32.is_le(U32.shln(1,tile_depth),cap) == True{} : Bool},
  +room_fits: {Nat.is_le(Nat.mul(units,Coordinates.inner(problem)),U32.to_nat(U32.shln(1,tile_depth))) == True{} : Bool},
  +rows_height: {Nat.is_le(U32.to_nat(rows),Guards.height(problem)) == True{} : Bool},+words: StencilRoom<limits,problem,cap>) ->
  Model.Packable<band_window(problem,base,rows),units,tile_depth,cap>:
  BlockBounds.Configuration{target_positive,target_room,target_span_room,allocation_fits,allocation_bound,capacity_room,+loop_fits,doubled} = configuration
  StencilRoom{sixteen,seven,positive,width_bound,width_room,lanes_room,stride_positive,stride_bound,+taps_positive,+taps_room,channels_positive,divisor_bound} = words
  +facts = Guards.valid(problem,valid)
  +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  +c = Guards.input_channel_count(problem)
  +k = Guards.kernel(problem)
  +entries = U32.to_nat(((Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32) * U32.from_nat(k) : U32))
  +value = entries_value(limits,problem,valid,supported,allocation_cap,cap,configuration)
  +inner = U32.to_nat(Coordinates.native_inner(problem))
  +inner_exact = Coordinates.inner_exact(limits,problem,facts,support)
  +power = U32.to_nat(U32.shln(1,tile_depth))
  +fits = {
    %inner_entries(limits,problem,valid,supported,allocation_cap,cap,configuration) : {Nat.is_le(Nat.mul(units,_),power) == True{} : Bool}
    %Equal.sym(Nat,inner,Coordinates.inner(problem),inner_exact) : {Nat.is_le(Nat.mul(units,_),power) == True{} : Bool}
    room_fits : {Nat.is_le(Nat.mul(units,Nat.mul(entries,U32.to_nat(U32.from_nat(Guards.kernel(problem))))),power) == True{} : Bool}}
  Model.Packable{
    Passes.PackRoom{cap,loop_fits,Order.transitive(tile_depth,30n,31n,tile_depth_ok,{==}),Bounds.as_nat_le(U32.shln(1,tile_depth),cap,tile_power),taps_room,taps_positive,
      taps_positive,
      {%Equal.sym(Nat,entries,Nat.mul(c,k),value) : {Nat.is_lt(0n,_) == True{} : Bool}
        Order.product_positive(c,k,Contents.valid_condition(problem,valid,10n),Coordinates.kernel_positive(problem,facts)) : {Nat.is_lt(0n,entries) == True{} : Bool}},
      inner_entries(limits,problem,valid,supported,allocation_cap,cap,configuration),fits},
    {%Equal.sym(Nat,U32.to_nat((Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32)),c,channels_word(limits,problem,valid,supported)) : {entries == Nat.mul(_,U32.to_nat(U32.from_nat(Guards.kernel(problem)))) : Nat}
      %Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.kernel(problem))),k,kernel_word(limits,problem,valid,supported)) : {entries == Nat.mul(c,_) : Nat}
      value : {entries == Nat.mul(U32.to_nat((Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32)),U32.to_nat(U32.from_nat(Guards.kernel(problem)))) : Nat}},
    Pixels.ValuesRoom{CellValues.geometry(limits,problem,(Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32),rows,U32.from_nat(Guards.width(problem)),valid,supported,channels_word(limits,problem,valid,supported),
        SupportedCoordinates.width_exact(limits,problem,support),rows_height),
      {%Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.kernel(problem))),k,kernel_word(limits,problem,valid,supported)) : {Nat.is_le(Nat.double(Nat.mul(_,_)),U32.to_nat(cap)) == True{} : Bool}
        double_room(limits,allocation_cap,cap,configuration,Nat.mul(k,k),Coordinates.square_bound(limits,problem,facts,support)) : {Nat.is_le(Nat.double(Nat.mul(U32.to_nat(U32.from_nat(Guards.kernel(problem))),U32.to_nat(U32.from_nat(Guards.kernel(problem))))),U32.to_nat(cap)) == True{} : Bool}},
      {%Equal.sym(Nat,inner,Coordinates.inner(problem),inner_exact) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
        word_room(limits,allocation_cap,cap,configuration,Coordinates.inner(problem),Coordinates.inner_bound(limits,problem,facts,support)) : {Nat.is_le(inner,U32.to_nat(cap)) == True{} : Bool}}},
    tile_depth_ok}

# The tiles of a call have room for its blocks.
def call_packable(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+allocation_cap: U32,+cap: U32,+configuration: BlockBounds.Configuration<limits,8192,allocation_cap,cap>,+base: U32,+local: U32,+count: U32,
  +cover: Positions.Cover<problem,U32.to_nat(BandCover.first_word(problem,local)),U32.to_nat(BandCover.rows_word(problem,local,count)),U32.to_nat(local),U32.to_nat(count)>,
  +plane: U32,+count_le: {Nat.is_le(U32.to_nat(count),U32.to_nat(plane)) == True{} : Bool},
  +plane_grid: {Nat.is_le(U32.to_nat(plane),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool},+tile_depth: Nat,+tile_depth_ok: {Nat.is_le(tile_depth,30n) == True{} : Bool},+tile_power: {U32.is_le(U32.shln(1,tile_depth),cap) == True{} : Bool},
  +tile_need: {Nat.is_le(U32.to_nat(Pipeline.workspace_tiles(8192,Coordinates.native_inner(problem),plane)),U32.to_nat(U32.shln(1,tile_depth))) == True{} : Bool},
  +words: StencilRoom<limits,problem,cap>,+fit: PlaneFit<limits,problem,plane>) ->
  Model.Packable<band_window(problem,base,BandCover.rows_word(problem,local,count)),Nat.min(Nat.div(U32.to_nat(Evidence.span(8192,problem)),8n),Nat.div(Nat.add(U32.to_nat(count),7n),8n)),tile_depth,cap>:
  PlaneFit{planes,+plane_room,bound_cap} = fit
  +rows = BandCover.rows_word(problem,local,count)
  +fewest = Nat.min(Nat.div(U32.to_nat(Evidence.span(8192,problem)),8n),Nat.div(Nat.add(U32.to_nat(count),7n),8n))
  +groups = PackingBounds.groups(U32.min(Evidence.span(8192,problem),Window.padded(count)))
  +count_room = Order.transitive(U32.to_nat(count),U32.to_nat(plane),U32.to_nat(Machine.capacity(limits)),count_le,plane_room)
  +room_fits = Order.transitive(Nat.mul(fewest,Coordinates.inner(problem)),Nat.mul(groups,Coordinates.inner(problem)),U32.to_nat(U32.shln(1,tile_depth)),
    RowMajor.mul_le(fewest,groups,Coordinates.inner(problem),fewest_room(limits,problem,count,count_room)),
    Evidence.tile_room(limits,problem,valid,supported,count,plane,count_le,plane_room,8192,allocation_cap,cap,configuration,U32.shln(1,tile_depth),tile_need))
  +first = U32.to_nat(BandCover.first_word(problem,local))
  +rows_height = Order.transitive(U32.to_nat(rows),Nat.add(first,U32.to_nat(rows)),Guards.height(problem),Order.right_le(first,U32.to_nat(rows)),
    Evidence.cover_fits(problem,first,U32.to_nat(rows),U32.to_nat(local),U32.to_nat(count),cover))
  stencil_packable(limits,problem,valid,supported,allocation_cap,cap,configuration,base,rows,fewest,tile_depth,tile_depth_ok,tile_power,room_fits,rows_height,words)

# ---- A ready call ----

# A call is ready: its planes and tiles have room and its blocks are
# scheduled.
def call_ready(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+allocation_cap: U32,+cap: U32,+configuration: BlockBounds.Configuration<limits,8192,allocation_cap,cap>,+base: U32,+local: U32,+count: U32,+offset: U32,+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
  +cover: Positions.Cover<problem,U32.to_nat(BandCover.first_word(problem,local)),U32.to_nat(BandCover.rows_word(problem,local,count)),U32.to_nat(local),U32.to_nat(count)>,
  +geometry: Blocked.Geometry<limits,Evidence.window(problem,local,count),Evidence.pack_start(problem,local),offset,count,Window.padded(count),
    Evidence.span(8192,Evidence.window(problem,local,count)),cap>,
  +schedule: BlockBounds.Schedule<U32.to_nat(((Window.padded(count) + Evidence.span(8192,Evidence.window(problem,local,count)) - 1) / Evidence.span(8192,Evidence.window(problem,local,count)) : U32)),
    0n,U32.to_nat(Window.padded(count)),U32.to_nat(Evidence.span(8192,Evidence.window(problem,local,count)))>,
  +plane: U32,+count_le: {Nat.is_le(U32.to_nat(count),U32.to_nat(plane)) == True{} : Bool},
  +plane_grid: {Nat.is_le(U32.to_nat(plane),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool},+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+tile_depth: Nat,+pair_depth: Nat,+tiles_balanced: Shape.Balanced(Tile.Tile8,tile_depth,tiles),
  +pairs_balanced: Shape.Balanced(Tile.Tile16,pair_depth,pairs),+tile_depth_ok: {Nat.is_le(tile_depth,30n) == True{} : Bool},+pair_depth_ok: {Nat.is_le(pair_depth,30n) == True{} : Bool},
  +tile_power: {U32.is_le(U32.shln(1,tile_depth),cap) == True{} : Bool},+pair_power: {U32.is_le(U32.shln(1,pair_depth),cap) == True{} : Bool},
  +tile_need: {Nat.is_le(U32.to_nat(Pipeline.workspace_tiles(8192,Coordinates.native_inner(problem),plane)),U32.to_nat(U32.shln(1,tile_depth))) == True{} : Bool},
  +pair_need: {Nat.is_le(U32.to_nat(Packing.plane_bound(band_window(problem,base,BandCover.rows_word(problem,local,count)),Window.padded(plane))),U32.to_nat(U32.shln(1,pair_depth))) == True{} : Bool}) ->
  Model.CallReady<band_work(problem,base,local,count,offset),Evidence.pack_start(problem,local),offset,pairs,tiles,tile_depth,pair_depth,Nat.div(Nat.add(U32.to_nat(count),7n),8n),cap>:
  ConvSpec.Problem{shape,+input_values,+weight_values,+bias_values} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  Blocked.Geometry{+pack_room,geometry_valid,geometry_supported,geometry_cap,padded_bound,padded_end,outputs_room,end_exact,+offset_room,geometry_sum,span_positive,
    padded_aligned,span_aligned,padded_value} = geometry
  BlockBounds.Schedule{covers,+last} = schedule
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input_values,weight_values,bias_values} : ConvSpec.Problem}
  +words = stencil_room(limits,problem,valid,supported,allocation_cap,cap,configuration)
  +fit = plane_fit(limits,problem,valid,supported,plane,plane_grid)
  +rows = BandCover.rows_word(problem,local,count)
  +bounds = bound_room(limits,problem,valid,supported,allocation_cap,cap,configuration,base,rows,plane,words,fit)
  +count_room = Order.transitive(U32.to_nat(count),U32.to_nat(plane),U32.to_nat(Machine.capacity(limits)),count_le,
    Order.transitive(U32.to_nat(plane),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem)),U32.to_nat(Machine.capacity(limits)),plane_grid,
      Padded.spatial_bound_facts(limits,problem,Guards.valid(problem,valid),Padded.scalar(limits,problem,Support.fast(limits,problem,supported)))))
  Model.ready(MicroLeaf.rows(problem),count,band_window(problem,base,BandCover.rows_word(problem,local,count)),Evidence.pack_start(problem,local),offset,pairs,tiles,tile_depth,pair_depth,Nat.div(Nat.add(U32.to_nat(count),7n),8n),cap,
    call_packable(limits,problem,valid,supported,allocation_cap,cap,configuration,base,local,count,cover,plane,count_le,plane_grid,tile_depth,tile_depth_ok,tile_power,tile_need,words,fit),
    call_room(limits,problem,valid,supported,allocation_cap,cap,configuration,base,rows,Evidence.pack_start(problem,local),count,positive,pack_room,plane,count_le,pair_depth,pair_depth_ok,pair_power,pair_need,words,fit,bounds),
    Encoded.padded_exact(limits,count,count_room),tiles_balanced,pairs_balanced,last,offset_room)
