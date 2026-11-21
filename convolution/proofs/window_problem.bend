# A window of a valid, supported problem is itself valid and supported when it
# keeps at least one row and the kernel still fits; its contract is the window
# tree for input and the original owners for weights and bias.
import Base
import ../specification.bend as ConvSpec
import ../inference.bend as NativeConv
import ../scalar_kernel.bend as ScalarConv
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/row_major_index.bend as RowMajor
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./input_storage.bend as Inputs
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ./window_band_values.bend as WindowBandValues
import ./window_samples.bend as Samples
import ../../lib/proofs/encoded_coordinates.bend as Encoding
import ../../lib/kernel_gemm.bend as KernelGemm
import ../../lib/kernel_shape.bend as KernelShape

def band(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat) -> WindowBandValues.Band:
  WindowBandValues.Band{Guards.input(problem),Guards.height(problem),Guards.width(problem),first,rows}

def count(+problem: ConvSpec.Problem,+rows: Nat) -> Nat:
  Nat.mul(Guards.input_channel_count(problem),Nat.mul(rows,Guards.width(problem)))

def valid_from(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+facts: Guards.Valid<problem>,
  +rows_positive: {Nat.is_lt(0n,rows) == True{} : Bool},
  +kernel_fits: {Nat.is_le(Guards.kernel(problem),Nat.add(rows,Nat.mul(2n,Guards.pad(problem)))) == True{} : Bool}) ->
  {ConvSpec.valid(Samples.window(problem,first,rows)) == True{} : Bool}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  Guards.Valid{+ci_positive,+co_positive,+height_positive,+width_positive,+kernel_positive,+stride_positive,+kernel_height,+kernel_width,
    +input_length,+weight_length,+bias_length} = facts
  +total = Nat.mul(ci,Nat.mul(rows,width))
  +windowed = {ConvSpec.Problem{ConvSpec.Shape{ci,co,rows,width,kernel,stride,padding},
    WindowBandValues.values(total,0n,WindowBandValues.Band{input,height,width,first,rows}),weights,bias} : ConvSpec.Problem}
  +window_length = {%Equal.sym(Nat,ConvSpec.length(WindowBandValues.values(total,0n,WindowBandValues.Band{input,height,width,first,rows})),total,
        WindowBandValues.length(total,0n,WindowBandValues.Band{input,height,width,first,rows})) : {Nat.is_eq(_,total) == True{} : Bool}
      Order.equal_reflexive(total) : {Guards.input_length_matches_shape(windowed) == True{} : Bool}}
  Boolean.all_hold(Guards.valid_conditions(windowed),
    KernelGemm.Both{bias_length,KernelGemm.Both{weight_length,KernelGemm.Both{window_length,KernelGemm.Both{kernel_width,KernelGemm.Both{kernel_fits,KernelGemm.Both{stride_positive,
      KernelGemm.Both{kernel_positive,KernelGemm.Both{width_positive,KernelGemm.Both{rows_positive,KernelGemm.Both{co_positive,KernelGemm.Both{ci_positive,Unit{}}}}}}}}}}}})

def valid(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +rows_positive: {Nat.is_lt(0n,rows) == True{} : Bool},
  +kernel_fits: {Nat.is_le(Guards.kernel(problem),Nat.add(rows,Nat.mul(2n,Guards.pad(problem)))) == True{} : Bool}) ->
  {ConvSpec.valid(Samples.window(problem,first,rows)) == True{} : Bool}:
  valid_from(problem,first,rows,Guards.valid(problem,valid),rows_positive,kernel_fits)

def axis_le(+smaller: Nat,+larger: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+stride_positive: {Nat.is_lt(0n,stride) == True{} : Bool},
  +ordered: {Nat.is_le(smaller,larger) == True{} : Bool}) ->
  {Nat.is_le(ConvSpec.axis_size(smaller,kernel,stride,padding),ConvSpec.axis_size(larger,kernel,stride,padding)) == True{} : Bool}:
  DivisionOrder.quotient_le(Nat.sub(Nat.add(smaller,Nat.mul(2n,padding)),kernel),Nat.sub(Nat.add(larger,Nat.mul(2n,padding)),kernel),stride,stride_positive,
    DivisionOrder.subtract_le(Nat.add(smaller,Nat.mul(2n,padding)),Nat.add(larger,Nat.mul(2n,padding)),kernel,
      Order.add_le(smaller,larger,Nat.mul(2n,padding),ordered)))

def scalar_from_formula(+limits: Machine.Limits,+problem: ConvSpec.Problem,
  +formula: {Guards.supported_formula(problem,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)) == True{} : Bool}) ->
  {ScalarConv.supported_with_limits(limits,problem) == True{} : Bool}:
  Machine.Limits{dimension,capacity,coordinate,half_word,facts} = limits
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  formula

# The planes of fewer positions fit where those of more do.
def planes_fewer(+fewer: Nat,+count: Nat,+ci: Nat,+ow: Nat,+kernel: Nat,+stride: Nat,+capacity: U32,+positive: {Nat.is_lt(0n,ow) == True{} : Bool},
  +ordered: {Nat.is_le(fewer,count) == True{} : Bool},+fit: {KernelShape.planes_fit(count,ci,ow,kernel,stride,capacity) == True{} : Bool}) ->
  {KernelShape.planes_fit(fewer,ci,ow,kernel,stride,capacity) == True{} : Bool}:
  +tiles = KernelShape.plane_tiles(ow,kernel,stride)
  +rows = KernelShape.plane_rows(fewer,ow,kernel,stride)
  +more = KernelShape.plane_rows(count,ow,kernel,stride)
  +area = Nat.is_le(Nat.mul(more,Nat.mul(tiles,8n)),U32.to_nat(capacity))
  +pairs = Nat.is_le(Nat.mul(Nat.mul(stride,ci),Nat.mul(more,tiles)),U32.to_nat(capacity))
  +last = Nat.sub(Nat.add(fewer,ow),1n)
  +more_last = Nat.sub(Nat.add(count,ow),1n)
  +rows_le = Order.add_le(Nat.mul(Nat.div(last,ow),stride),Nat.mul(Nat.div(more_last,ow),stride),kernel,
    RowMajor.mul_le(Nat.div(last,ow),Nat.div(more_last,ow),stride,
      DivisionOrder.quotient_le(last,more_last,ow,positive,DivisionOrder.subtract_le(Nat.add(fewer,ow),Nat.add(count,ow),1n,Order.add_le(fewer,count,ow,ordered)))))
  +area_fewer = Order.transitive(Nat.mul(rows,Nat.mul(tiles,8n)),Nat.mul(more,Nat.mul(tiles,8n)),U32.to_nat(capacity),RowMajor.mul_le(rows,more,Nat.mul(tiles,8n),rows_le),
    Boolean.left(area,pairs,fit))
  +pairs_fewer = Order.transitive(Nat.mul(Nat.mul(stride,ci),Nat.mul(rows,tiles)),Nat.mul(Nat.mul(stride,ci),Nat.mul(more,tiles)),U32.to_nat(capacity),
    DivisionOrder.multiply_left_le(Nat.mul(stride,ci),Nat.mul(rows,tiles),Nat.mul(more,tiles),RowMajor.mul_le(rows,more,tiles,rows_le)),Boolean.right(area,pairs,fit))
  Boolean.all_hold([Nat.is_le(Nat.mul(Nat.mul(stride,ci),Nat.mul(rows,tiles)),U32.to_nat(capacity)),Nat.is_le(Nat.mul(rows,Nat.mul(tiles,8n)),U32.to_nat(capacity))],
    KernelGemm.Both{pairs_fewer,KernelGemm.Both{area_fewer,Unit{}}})

def supported_from(+limits: Machine.Limits,+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+facts: Guards.Valid<problem>,
  +fast: Support.FastSupported<limits,problem>,+fits: {Nat.is_le(Nat.add(first,rows),Guards.height(problem)) == True{} : Bool}) ->
  {NativeConv.supports_packed_with_limits(limits,Samples.window(problem,first,rows)) == True{} : Bool}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  Guards.Valid{ci_positive,co_positive,height_positive,width_positive,kernel_positive,+stride_positive,kernel_height,kernel_width,
    +input_length,weight_length,bias_length} = facts
  Support.FastSupported{+scalar,+packing,+output,+planes} = fast
  Guards.Supported{+ci_bound,+co_bound,+height_bound,+width_bound,+kernel_bound,+stride_bound,+padding_bound,+input_bound,+weight_bound,
    +output_bound,+vertical_bound,+horizontal_bound} = scalar
  +dimension = Machine.dimension(limits)
  +capacity = Machine.capacity(limits)
  +coordinate = Machine.coordinate(limits)
  +total = Nat.mul(ci,Nat.mul(rows,width))
  +values = WindowBandValues.values(total,0n,WindowBandValues.Band{input,height,width,first,rows})
  +windowed = {ConvSpec.Problem{ConvSpec.Shape{ci,co,rows,width,kernel,stride,padding},values,weights,bias} : ConvSpec.Problem}
  +rows_le = Order.transitive(rows,Nat.add(first,rows),height,Order.right_le(first,rows),fits)
  +out_height = ConvSpec.axis_size(height,kernel,stride,padding)
  +out_rows = ConvSpec.axis_size(rows,kernel,stride,padding)
  +out_width = ConvSpec.axis_size(width,kernel,stride,padding)
  +axis = axis_le(rows,height,kernel,stride,padding,stride_positive,rows_le)
  +area = RowMajor.mul_le(out_rows,out_height,out_width,axis)
  +padded = Encoding.padded_le(Nat.mul(out_rows,out_width),Nat.mul(out_height,out_width),area)
  +input_fits = {%Equal.sym(Nat,ConvSpec.length(values),total,WindowBandValues.length(total,0n,WindowBandValues.Band{input,height,width,first,rows})) :
        {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
      Order.transitive(total,Nat.mul(ci,Nat.mul(height,width)),U32.to_nat(capacity),
        DivisionOrder.multiply_left_le(ci,Nat.mul(rows,width),Nat.mul(height,width),RowMajor.mul_le(rows,height,width,rows_le)),
        %Order.equal_of_is_eq(ConvSpec.length(input),Nat.mul(ci,Nat.mul(height,width)),input_length) :
          {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
        input_bound) : {Guards.input_buffer_supported(windowed,dimension,capacity,coordinate) == True{} : Bool}}
  +formula = Boolean.all_hold(Guards.supported_conditions(windowed,dimension,capacity,coordinate),
    KernelGemm.Both{horizontal_bound,
    KernelGemm.Both{Order.transitive(Nat.add(Nat.mul(out_rows,stride),kernel),Nat.add(Nat.mul(out_height,stride),kernel),U32.to_nat(coordinate),
        Order.add_le(Nat.mul(out_rows,stride),Nat.mul(out_height,stride),kernel,RowMajor.mul_le(out_rows,out_height,stride,axis)),vertical_bound),
    KernelGemm.Both{Order.transitive(Nat.mul(co,Nat.mul(out_rows,out_width)),Nat.mul(co,Nat.mul(out_height,out_width)),U32.to_nat(capacity),
        DivisionOrder.multiply_left_le(co,Nat.mul(out_rows,out_width),Nat.mul(out_height,out_width),area),output_bound),
    KernelGemm.Both{weight_bound,
    KernelGemm.Both{input_fits,
    KernelGemm.Both{padding_bound,KernelGemm.Both{stride_bound,KernelGemm.Both{kernel_bound,KernelGemm.Both{width_bound,
    KernelGemm.Both{Order.transitive(rows,height,U32.to_nat(dimension),rows_le,height_bound),
    KernelGemm.Both{co_bound,KernelGemm.Both{ci_bound,Unit{}}}}}}}}}}}}})
  +window_scalar = scalar_from_formula(limits,windowed,formula)
  +inner = Nat.mul(ci,Nat.mul(kernel,kernel))
  +window_packing = {Order.transitive(Nat.mul(Nat.mul(Nat.div(Nat.add(Nat.mul(out_rows,out_width),7n),8n),8n),inner),
        Nat.mul(Nat.mul(Nat.div(Nat.add(Nat.mul(out_height,out_width),7n),8n),8n),inner),U32.to_nat(capacity),
        RowMajor.mul_le(Nat.mul(Nat.div(Nat.add(Nat.mul(out_rows,out_width),7n),8n),8n),Nat.mul(Nat.div(Nat.add(Nat.mul(out_height,out_width),7n),8n),8n),inner,padded),
        packing) : {Support.packed_bound(limits,windowed) == True{} : Bool}}
  +window_output = {Order.transitive(Nat.mul(Nat.mul(Nat.div(Nat.add(Nat.mul(out_rows,out_width),7n),8n),8n),co),
        Nat.mul(Nat.mul(Nat.div(Nat.add(Nat.mul(out_height,out_width),7n),8n),8n),co),U32.to_nat(capacity),
        RowMajor.mul_le(Nat.mul(Nat.div(Nat.add(Nat.mul(out_rows,out_width),7n),8n),8n),Nat.mul(Nat.div(Nat.add(Nat.mul(out_height,out_width),7n),8n),8n),co,padded),
        output) : {Support.output_bound(limits,windowed) == True{} : Bool}}
  +window_planes = {planes_fewer(Nat.mul(Nat.div(Nat.add(Nat.mul(out_rows,out_width),7n),8n),8n),Nat.mul(Nat.div(Nat.add(Nat.mul(out_height,out_width),7n),8n),8n),
        ci,out_width,kernel,stride,capacity,{==},padded,planes) : {Support.planes_bound(limits,windowed) == True{} : Bool}}
  %Equal.sym(Bool,NativeConv.supports_packed_with_limits(limits,windowed),
    ScalarConv.supported_with_limits(limits,windowed) && Support.packed_bound(limits,windowed) && Support.output_bound(limits,windowed) && Support.planes_bound(limits,windowed),
    Support.packed_guard_matches_original_conditions(limits,windowed)) : {_ == True{} : Bool}
  Boolean.all_hold([Support.planes_bound(limits,windowed),Support.output_bound(limits,windowed),Support.packed_bound(limits,windowed),ScalarConv.supported_with_limits(limits,windowed)],
    KernelGemm.Both{window_planes,KernelGemm.Both{window_output,KernelGemm.Both{window_packing,KernelGemm.Both{window_scalar,Unit{}}}}})

def supported(+limits: Machine.Limits,+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(first,rows),Guards.height(problem)) == True{} : Bool}) ->
  {NativeConv.supports_packed_with_limits(limits,Samples.window(problem,first,rows)) == True{} : Bool}:
  supported_from(limits,problem,first,rows,Guards.valid(problem,valid),Support.fast(limits,problem,supported),fits)

# The window tree carries the band; weights and bias keep their original owners.
def contract(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+first: Nat,+rows: Nat,+tree: Tree.Tree<F32>,
  +contents: WindowBandValues.Contents(count(problem,rows),band(problem,first,rows),tree),
  +room: {Nat.is_le(count(problem,rows),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool}) ->
  Inputs.Contract<Samples.window(problem,first,rows)>:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  Inputs.Contract{input_evidence,+weight_evidence,+bias_evidence,+parameters} = storage_contract
  Inputs.Contract{WindowBandValues.evidence(Nat.mul(ci,Nat.mul(rows,width)),WindowBandValues.Band{input,height,width,first,rows},tree,contents,room),weight_evidence,bias_evidence,parameters}
