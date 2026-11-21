# The output cells of a position are its channel groups side by side; their
# addresses are bounded by the original guards.
import Base
import ../specification.bend as ConvSpec
import ../inference.bend as NativeConv
import ../../lib/machine_limits.bend as Machine
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/nat_ceiling.bend as Ceiling
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/word_multiplication.bend as WordMultiplication
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./supported_coordinate_bounds.bend as Coordinates
import ./packed_output_bounds.bend as Output
import ./gather_output_index.bend as Destination
import ./reduction_coordinate_bounds.bend as Reduction

def groups(+problem: ConvSpec.Problem) -> U32:
  Gemm.blocks(U32.from_nat(Guards.output_channel_count(problem)))

def groups_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {U32.to_nat(groups(problem)) == Ceiling.ceiling(Guards.output_channel_count(problem),8n) : Nat}:
  +fast = Support.fast(limits,problem,supported)
  +rows = U32.from_nat(Guards.output_channel_count(problem))
  +exact = Coordinates.co_exact(limits,problem,Padded.scalar(limits,problem,fast))
  %exact : {U32.to_nat(groups(problem)) == Ceiling.ceiling(_,8n) : Nat}
  Encoded.eight_blocks_exact(limits,rows,
    %Equal.sym(Nat,U32.to_nat(rows),Guards.output_channel_count(problem),exact) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    Output.channel_capacity_bound(limits,problem,fast))

def groups_le(+limits: Machine.Limits,+problem: ConvSpec.Problem,+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(groups(problem)),Guards.output_channel_count(problem)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(groups(problem)),Ceiling.ceiling(Guards.output_channel_count(problem),8n),groups_exact(limits,problem,supported)) :
    {Nat.is_le(_,Guards.output_channel_count(problem)) == True{} : Bool}
  Ceiling.bounded(Guards.output_channel_count(problem),8n,{==})

def channel_group(+limits: Machine.Limits,+problem: ConvSpec.Problem,+channel: Nat,
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.div(channel,8n),U32.to_nat(groups(problem))) == True{} : Bool}:
  +rows = Guards.output_channel_count(problem)
  +group = Nat.div(channel,8n)
  %Equal.sym(Nat,U32.to_nat(groups(problem)),Ceiling.ceiling(rows,8n),groups_exact(limits,problem,supported)) : {Nat.is_lt(group,_) == True{} : Bool}
  %Equal.sym(Bool,Nat.is_lt(group,Ceiling.ceiling(rows,8n)),Nat.is_lt(Nat.mul(group,8n),rows),Ceiling.upper_comparison(rows,8n,group,{==})) : {_ == True{} : Bool}
  Order.le_lt_trans(Nat.mul(group,8n),channel,rows,DivisionOrder.product_below(channel,8n,{==}),inside)

def spatial_bound(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Padded.spatial(problem)),Support.padded(problem)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(Padded.spatial(problem)),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem)),
    Padded.spatial_exact(limits,problem,Guards.valid(problem,valid),Support.fast(limits,problem,supported))) : {Nat.is_le(_,Support.padded(problem)) == True{} : Bool}
  Padded.spatial_within_padded(limits,problem,Guards.valid(problem,valid),Support.fast(limits,problem,supported))

def slots_bound(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(U32.to_nat(groups(problem)),U32.to_nat(Padded.spatial(problem))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Order.transitive(Nat.mul(U32.to_nat(groups(problem)),U32.to_nat(Padded.spatial(problem))),Nat.mul(Guards.output_channel_count(problem),U32.to_nat(Padded.spatial(problem))),U32.to_nat(Machine.capacity(limits)),
    RowMajor.mul_le(U32.to_nat(groups(problem)),Guards.output_channel_count(problem),U32.to_nat(Padded.spatial(problem)),groups_le(limits,problem,supported)),
    Output.elements_bound(limits,problem,Padded.spatial(problem),Support.fast(limits,problem,supported),spatial_bound(limits,problem,valid,supported)))

# A number of positions within the padded output, times the groups.
def scaled(+limits: Machine.Limits,+problem: ConvSpec.Problem,+count: U32,
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +bound: {Nat.is_le(U32.to_nat(count),Support.padded(problem)) == True{} : Bool}) ->
  {U32.to_nat((count * groups(problem) : U32)) == Nat.mul(U32.to_nat(count),U32.to_nat(groups(problem))) : Nat}:
  +n = U32.to_nat(count)
  +channels = Guards.output_channel_count(problem)
  WordMultiplication.bounded_multiply(count,groups(problem),Machine.capacity(limits),Limits.capacity_fits(limits),
    Order.transitive(Nat.mul(n,U32.to_nat(groups(problem))),Nat.mul(n,channels),U32.to_nat(Machine.capacity(limits)),
      DivisionOrder.multiply_left_le(n,U32.to_nat(groups(problem)),channels,groups_le(limits,problem,supported)),
      %RowMajor.multiply_commutative(channels,n) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
      Output.elements_bound(limits,problem,count,Support.fast(limits,problem,supported),bound)))

# The cells of every output position fit the machine.
def cells_bound(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(U32.to_nat(Padded.spatial(problem)),U32.to_nat(groups(problem))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  %RowMajor.multiply_commutative(U32.to_nat(groups(problem)),U32.to_nat(Padded.spatial(problem))) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  slots_bound(limits,problem,valid,supported)

# A gather of a part from offset writes cell `column` of its run at that cell
# past the cells of the positions before offset.
def destination(+limits: Machine.Limits,+problem: ConvSpec.Problem,+offset: U32,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +offset_bound: {Nat.is_le(U32.to_nat(offset),Support.padded(problem)) == True{} : Bool},
  +point_inside: {Nat.is_lt(Nat.add(Nat.mul(U32.to_nat(offset),U32.to_nat(groups(problem))),column),
    Nat.mul(U32.to_nat(Padded.spatial(problem)),U32.to_nat(groups(problem)))) == True{} : Bool}) ->
  {U32.to_nat((U32.from_nat(0n) * (Padded.spatial(problem) * groups(problem) : U32) + (offset * groups(problem) : U32) + U32.from_nat(column) : U32)) ==
    Nat.add(Nat.mul(U32.to_nat(offset),U32.to_nat(groups(problem))),column) : Nat}:
  +space = Nat.mul(U32.to_nat(Padded.spatial(problem)),U32.to_nat(groups(problem)))
  +start = Nat.mul(U32.to_nat(offset),U32.to_nat(groups(problem)))
  +capacity = Machine.capacity(limits)
  +cap = U32.to_nat(capacity)
  +fits = Limits.capacity_fits(limits)
  +reach = Bounds.nat_trans(Nat.add(start,column),space,cap,point_inside,cells_bound(limits,problem,valid,supported))
  +column_bound = Order.transitive(column,Nat.add(start,column),cap,
    %Order.commutative(column,start) : {Nat.is_le(column,_) == True{} : Bool} Order.le_add(column,start),Bounds.strict_le(Nat.add(start,column),cap,reach))
  Destination.combine(U32.from_nat(0n),U32.from_nat(column),(Padded.spatial(problem) * groups(problem) : U32),(offset * groups(problem) : U32),0n,column,space,start,capacity,fits,
    {==},Bounds.roundtrip(column,capacity,fits,column_bound),
    scaled(limits,problem,Padded.spatial(problem),supported,spatial_bound(limits,problem,valid,supported)),scaled(limits,problem,offset,supported,offset_bound),
    Bounds.strict_le(Nat.add(start,column),cap,reach))

# The cell of a channel at an output position.
def query_index(+problem: ConvSpec.Problem,+channel: Nat,+row: Nat,+column: Nat) -> Nat:
  Nat.add(Nat.mul(Reduction.position(problem,row,column),U32.to_nat(groups(problem))),Nat.div(channel,8n))

def position_inside(+limits: Machine.Limits,+problem: ConvSpec.Problem,+row: Nat,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {Nat.is_lt(Reduction.position(problem,row,column),U32.to_nat(Padded.spatial(problem))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(Padded.spatial(problem)),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem)),Padded.spatial_exact(limits,problem,Guards.valid(problem,valid),Support.fast(limits,problem,supported))) :
    {Nat.is_lt(Reduction.position(problem,row,column),_) == True{} : Bool}
  RowMajor.radix(row,ConvSpec.out_height(problem),column,ConvSpec.out_width(problem),row_inside,column_inside)

def query_inside(+limits: Machine.Limits,+problem: ConvSpec.Problem,+channel: Nat,+row: Nat,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {Nat.is_lt(query_index(problem,channel,row,column),Nat.mul(U32.to_nat(Padded.spatial(problem)),U32.to_nat(groups(problem)))) == True{} : Bool}:
  RowMajor.radix(Reduction.position(problem,row,column),U32.to_nat(Padded.spatial(problem)),Nat.div(channel,8n),U32.to_nat(groups(problem)),
    position_inside(limits,problem,row,column,valid,supported,row_inside,column_inside),channel_group(limits,problem,channel,supported,channel_inside))

def query_bound(+limits: Machine.Limits,+problem: ConvSpec.Problem,+channel: Nat,+row: Nat,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {Nat.is_le(query_index(problem,channel,row,column),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Bounds.strict_le(query_index(problem,channel,row,column),U32.to_nat(Machine.capacity(limits)),
    Bounds.nat_trans(query_index(problem,channel,row,column),Nat.mul(U32.to_nat(Padded.spatial(problem)),U32.to_nat(groups(problem))),U32.to_nat(Machine.capacity(limits)),
      query_inside(limits,problem,channel,row,column,valid,supported,channel_inside,row_inside,column_inside),cells_bound(limits,problem,valid,supported)))

def query_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+channel: Nat,+row: Nat,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(query_index(problem,channel,row,column))) == query_index(problem,channel,row,column) : Nat}:
  Bounds.roundtrip(query_index(problem,channel,row,column),Machine.capacity(limits),Limits.capacity_fits(limits),query_bound(limits,problem,channel,row,column,valid,supported,channel_inside,row_inside,column_inside))
