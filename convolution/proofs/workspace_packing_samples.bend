# The tiles a block fills in its workspace hold the samples of the problem.
import Base
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/kernel_spatial_packing.bend as Packing
import ../../lib/proofs/kernel_spatial_packing_samples.bend as SpatialSamples
import ./tile_array_projection.bend as ArrayProjection
import ./input_storage.bend as Inputs
import ../../lib/math_sequence.bend as Sequence
import ../specification.bend as ConvSpec
import ../../lib/machine_limits.bend as Machine
import ../../lib/kernel_tile.bend as Tile
import ../inference.bend as NativeConv
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./packing_tile_bounds.bend as PackingBounds
import ./reduction_coordinate_bounds.bend as Coordinates
import ./packed_reduction_samples.bend as ReductionSamples
import ./packing_readback.bend as PackingReadback
import ./packing_sample_refinement.bend as PackingSampleRefinement
import ./supported_padded_coordinates.bend as Padded
import ./packing_lane_reads.bend as LaneReads
import ./packing_tabulation.bend as PackingTabulation
import ../../lib/proofs/row_major_index.bend as RowMajor

def original_input(+p: ConvSpec.Problem,storage_contract: Inputs.Contract<p>) -> Tree.Tree<F32>:
  Inputs.input_tree(p,storage_contract)

# The stencil of a problem over its values.
def window(+p: ConvSpec.Problem) -> Packing.Stencil:
  LaneReads.window(U32.from_nat(Guards.height(p)),U32.from_nat(Guards.width(p)),U32.from_nat(Guards.kernel(p)),U32.from_nat(Guards.stride(p)),
    U32.from_nat(Guards.pad(p)),U32.from_nat(ConvSpec.out_width(p)),Coordinates.native_inner(p))

# A block fills its workspace with the value vectors of its entries.
def workspace_filled(+p: ConvSpec.Problem,+storage_contract: Inputs.Contract<p>,+offset: U32,+span: U32,workspace: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  SpatialSamples.fill(U32.to_nat(PackingBounds.count(span,p)),original_input(p,storage_contract),window(p),offset,workspace)

def LeafSamples(+p: ConvSpec.Problem,+offset: U32,+span: U32,+data: Tree.Tree<Tile.Tile8>) -> Type:
  @+lane: Sequence.Index(8n) ->
  @+group: Nat -> @+input_channel_index: Nat -> @+output_row_index: Nat -> @+output_column_index: Nat -> @+kernel_row_index: Nat -> @+kernel_column_index: Nat ->
  @group_inside: {Nat.is_lt(group,PackingBounds.groups(span)) == True{} : Bool} ->
  @position: {PackingReadback.logical_position(offset,group,lane) == Coordinates.position(p,output_row_index,output_column_index) : Nat} ->
  @input_channel_in_bounds: {Nat.is_lt(input_channel_index,Guards.input_channel_count(p)) == True{} : Bool} ->
  @output_row_in_bounds: {Nat.is_lt(output_row_index,ConvSpec.out_height(p)) == True{} : Bool} ->
  @output_column_in_bounds: {Nat.is_lt(output_column_index,ConvSpec.out_width(p)) == True{} : Bool} ->
  @kernel_row_in_bounds: {Nat.is_lt(kernel_row_index,Guards.kernel(p)) == True{} : Bool} ->
  @kernel_column_in_bounds: {Nat.is_lt(kernel_column_index,Guards.kernel(p)) == True{} : Bool} ->
  {ReductionSamples.select(lane,Tree.read(Tile.Tile8,data,U32.from_nat(PackingSampleRefinement.packing_index(p,group,input_channel_index,kernel_row_index,kernel_column_index)))) == ConvSpec.sample(p,input_channel_index,output_row_index,output_column_index,
      kernel_row_index,kernel_column_index) : F32}

def workspace_tabulated(+p: ConvSpec.Problem,+storage_contract: Inputs.Contract<p>,+offset: U32,+span: U32,workspace: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  PackingTabulation.fill(U32.to_nat(PackingBounds.count(span,p)),0,offset,original_input(p,storage_contract),workspace,
    U32.from_nat(Guards.height(p)),U32.from_nat(Guards.width(p)),U32.from_nat(Guards.kernel(p)),U32.from_nat(Guards.stride(p)),
    U32.from_nat(Guards.pad(p)),U32.from_nat(ConvSpec.out_width(p)),Coordinates.native_inner(p))

# The filled workspace of a block is the flat tabulation of its entries.
def workspace_contents(+limits: Machine.Limits,+p: ConvSpec.Problem,+storage_contract: Inputs.Contract<p>,+offset: U32,+span: U32,+workspace: Tree.Tree<Tile.Tile8>,
  +valid: Guards.Valid<p>,+support: Support.FastSupported<limits,p>,+span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(p)) == True{} : Bool}) ->
  {workspace_filled(p,storage_contract,offset,span,workspace) == workspace_tabulated(p,storage_contract,offset,span,workspace) : Tree.Tree<Tile.Tile8>}:
  +half = Machine.half_word(limits)
  +capacity = U32.to_nat(Machine.capacity(limits))
  +inner_exact = Coordinates.inner_exact(limits,p,valid,Padded.scalar(limits,p,support))
  +capacity_room = Order.transitive(capacity,Nat.double(capacity),U32.to_nat(half),Order.le_double(capacity),Limits.double_bound(limits,capacity,Order.reflexive(capacity)))
  +divisor_bound = {
    %Equal.sym(Nat,U32.to_nat(Coordinates.native_inner(p)),Coordinates.inner(p),inner_exact) : {Nat.is_le(Nat.double(_),U32.to_nat(half)) == True{} : Bool}
    Limits.double_bound(limits,Coordinates.inner(p),Coordinates.inner_bound(limits,p,valid,Padded.scalar(limits,p,support))) :
      {Nat.is_le(Nat.double(U32.to_nat(Coordinates.native_inner(p))),U32.to_nat(half)) == True{} : Bool}}
  +positive = {
    %Equal.sym(Nat,U32.to_nat(Coordinates.native_inner(p)),Coordinates.inner(p),inner_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    Coordinates.inner_positive(p,valid) : {Nat.is_lt(0n,U32.to_nat(Coordinates.native_inner(p))) == True{} : Bool}}
  PackingTabulation.filled(U32.to_nat(PackingBounds.count(span,p)),offset,original_input(p,storage_contract),workspace,U32.from_nat(Guards.height(p)),
    U32.from_nat(Guards.width(p)),U32.from_nat(Guards.kernel(p)),U32.from_nat(Guards.stride(p)),U32.from_nat(Guards.pad(p)),U32.from_nat(ConvSpec.out_width(p)),
    Coordinates.native_inner(p),half,Limits.half_word_fits(limits),
    Order.transitive(U32.to_nat(PackingBounds.count(span,p)),capacity,U32.to_nat(half),PackingBounds.count_bound(limits,p,span,valid,support,span_bound),capacity_room),
    divisor_bound,positive)

def tabulated(+p: ConvSpec.Problem,+storage_contract: Inputs.Contract<p>,+offset: U32,+span: U32) -> Tree.Tree<Tile.Tile8>:
  workspace_tabulated(p,storage_contract,offset,span,PackingReadback.allocated(PackingBounds.count(span,p)))

def sample_inside(+limits: Machine.Limits,+p: ConvSpec.Problem,+span: U32,+valid: Guards.Valid<p>,+support: Support.FastSupported<limits,p>,
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(p)) == True{} : Bool},+group: Nat,+input_channel: Nat,+kernel_row: Nat,+kernel_column: Nat,
  group_inside: {Nat.is_lt(group,PackingBounds.groups(span)) == True{} : Bool},channel_inside: {Nat.is_lt(input_channel,Guards.input_channel_count(p)) == True{} : Bool},
  row_inside: {Nat.is_lt(kernel_row,Guards.kernel(p)) == True{} : Bool},column_inside: {Nat.is_lt(kernel_column,Guards.kernel(p)) == True{} : Bool}) ->
  {Nat.is_lt(PackingSampleRefinement.packing_index(p,group,input_channel,kernel_row,kernel_column),U32.to_nat(PackingBounds.count(span,p))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(PackingBounds.count(span,p)),Nat.mul(PackingBounds.groups(span),Coordinates.inner(p)),PackingBounds.count_exact(limits,p,span,valid,support,span_bound)) :
    {Nat.is_lt(PackingSampleRefinement.packing_index(p,group,input_channel,kernel_row,kernel_column),_) == True{} : Bool}
  RowMajor.radix(group,PackingBounds.groups(span),Coordinates.reduction(input_channel,kernel_row,kernel_column,Guards.kernel(p)),Coordinates.inner(p),group_inside,
    Coordinates.reduction_inside(p,input_channel,kernel_row,kernel_column,channel_inside,row_inside,column_inside))

def leaf_lane(+limits: Machine.Limits,+p: ConvSpec.Problem,+storage_contract: Inputs.Contract<p>,+offset: U32,+span: U32,
  +valid: {ConvSpec.valid(p) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,p) == True{} : Bool},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(p)) == True{} : Bool},+lane: Sequence.Index(8n),
  +group: Nat,+input_channel_index: Nat,+output_row_index: Nat,+output_column_index: Nat,+kernel_row_index: Nat,+kernel_column_index: Nat,
  +group_inside: {Nat.is_lt(group,PackingBounds.groups(span)) == True{} : Bool},
  +position: {PackingReadback.logical_position(offset,group,lane) == Coordinates.position(p,output_row_index,output_column_index) : Nat},
  +input_channel_in_bounds: {Nat.is_lt(input_channel_index,Guards.input_channel_count(p)) == True{} : Bool},
  +output_row_in_bounds: {Nat.is_lt(output_row_index,ConvSpec.out_height(p)) == True{} : Bool},
  +output_column_in_bounds: {Nat.is_lt(output_column_index,ConvSpec.out_width(p)) == True{} : Bool},
  +kernel_row_in_bounds: {Nat.is_lt(kernel_row_index,Guards.kernel(p)) == True{} : Bool},
  +kernel_column_in_bounds: {Nat.is_lt(kernel_column_index,Guards.kernel(p)) == True{} : Bool}) ->
  {ReductionSamples.select(lane,Tree.read(Tile.Tile8,tabulated(p,storage_contract,offset,span),U32.from_nat(PackingSampleRefinement.packing_index(p,group,input_channel_index,kernel_row_index,kernel_column_index)))) ==
    ConvSpec.sample(p,input_channel_index,output_row_index,output_column_index,kernel_row_index,kernel_column_index) : F32}:
  +valid_facts = Guards.valid(p,valid)
  +support_facts = Support.fast(limits,p,supported)
  +count = PackingBounds.count(span,p)
  PackingSampleRefinement.logical_packing_lane_matches_sample(limits,p,storage_contract,count,offset,lane,group,input_channel_index,output_row_index,output_column_index,kernel_row_index,kernel_column_index,
    valid,Support.scalar_guard(limits,p,supported),PackingBounds.count_bound(limits,p,span,valid_facts,support_facts,span_bound),
    sample_inside(limits,p,span,valid_facts,support_facts,span_bound,group,input_channel_index,kernel_row_index,kernel_column_index,group_inside,input_channel_in_bounds,kernel_row_in_bounds,kernel_column_in_bounds),
    position,input_channel_in_bounds,output_row_in_bounds,output_column_in_bounds,kernel_row_in_bounds,kernel_column_in_bounds)

def workspace_lane(+limits: Machine.Limits,+p: ConvSpec.Problem,+storage_contract: Inputs.Contract<p>,+offset: U32,+span: U32,
  +workspace: Tree.Tree<Tile.Tile8>,+depth: Nat,+balanced: Shape.Balanced(Tile.Tile8,depth,workspace),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(PackingBounds.count(span,p)),U32.to_nat(Tree.capacity(Tile.Tile8,workspace))) == True{} : Bool},
  +valid: {ConvSpec.valid(p) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,p) == True{} : Bool},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(p)) == True{} : Bool},+lane: Sequence.Index(8n),
  +group: Nat,+input_channel: Nat,+output_row: Nat,+output_column: Nat,+kernel_row: Nat,+kernel_column: Nat,
  +group_inside: {Nat.is_lt(group,PackingBounds.groups(span)) == True{} : Bool},
  +position: {PackingReadback.logical_position(offset,group,lane) == Coordinates.position(p,output_row,output_column) : Nat},
  +channel_inside: {Nat.is_lt(input_channel,Guards.input_channel_count(p)) == True{} : Bool},
  +row_inside: {Nat.is_lt(output_row,ConvSpec.out_height(p)) == True{} : Bool},+column_inside: {Nat.is_lt(output_column,ConvSpec.out_width(p)) == True{} : Bool},
  +kernel_row_inside: {Nat.is_lt(kernel_row,Guards.kernel(p)) == True{} : Bool},+kernel_column_inside: {Nat.is_lt(kernel_column,Guards.kernel(p)) == True{} : Bool}) ->
  {ReductionSamples.select(lane,Tree.read(Tile.Tile8,workspace_tabulated(p,storage_contract,offset,span,workspace),U32.from_nat(PackingSampleRefinement.packing_index(p,group,input_channel,kernel_row,kernel_column)))) ==
    ConvSpec.sample(p,input_channel,output_row,output_column,kernel_row,kernel_column) : F32}:
  +count = PackingBounds.count(span,p)
  +index = PackingSampleRefinement.packing_index(p,group,input_channel,kernel_row,kernel_column)
  +query = U32.from_nat(index)
  +bounded = PackingBounds.count_bound(limits,p,span,Guards.valid(p,valid),Support.fast(limits,p,supported),span_bound)
  +inside = sample_inside(limits,p,span,Guards.valid(p,valid),Support.fast(limits,p,supported),span_bound,group,input_channel,kernel_row,kernel_column,group_inside,channel_inside,kernel_row_inside,kernel_column_inside)
  +query_exact = PackingSampleRefinement.packing_index_exact(limits,p,count,group,input_channel,kernel_row,kernel_column,bounded,inside)
  Equal.trans(F32,ReductionSamples.select(lane,Tree.read(Tile.Tile8,workspace_tabulated(p,storage_contract,offset,span,workspace),query)),
    ReductionSamples.select(lane,Tree.read(Tile.Tile8,tabulated(p,storage_contract,offset,span),query)),ConvSpec.sample(p,input_channel,output_row,output_column,kernel_row,kernel_column),
    PackingReadback.workspace_readback(limits,count,offset,original_input(p,storage_contract),workspace,U32.from_nat(Guards.height(p)),U32.from_nat(Guards.width(p)),U32.from_nat(Guards.kernel(p)),U32.from_nat(Guards.stride(p)),U32.from_nat(Guards.pad(p)),U32.from_nat(ConvSpec.out_width(p)),Coordinates.native_inner(p),lane,query,depth,
      ArrayProjection.balanced(depth,workspace,lane,balanced),depth_ok,
      %Equal.sym(U32,Tree.capacity(F32,ArrayProjection.project(workspace,lane)),Tree.capacity(Tile.Tile8,workspace),ArrayProjection.capacity(workspace,lane)) : {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool} room,
      Order.le_lt_trans(0n,index,U32.to_nat(count),Order.zero_le(index),inside),bounded,
      %Equal.sym(Nat,U32.to_nat(query),index,query_exact) : {Nat.is_lt(_,U32.to_nat(count)) == True{} : Bool} inside),
    leaf_lane(limits,p,storage_contract,offset,span,valid,supported,span_bound,lane,group,input_channel,output_row,output_column,kernel_row,kernel_column,group_inside,position,channel_inside,row_inside,column_inside,kernel_row_inside,kernel_column_inside))

def workspace_samples(+limits: Machine.Limits,+p: ConvSpec.Problem,+storage_contract: Inputs.Contract<p>,+offset: U32,+span: U32,
  +workspace: Tree.Tree<Tile.Tile8>,+depth: Nat,+balanced: Shape.Balanced(Tile.Tile8,depth,workspace),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(PackingBounds.count(span,p)),U32.to_nat(Tree.capacity(Tile.Tile8,workspace))) == True{} : Bool},
  +valid: {ConvSpec.valid(p) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,p) == True{} : Bool},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(p)) == True{} : Bool}) ->
  LeafSamples(p,offset,span,workspace_filled(p,storage_contract,offset,span,workspace)):
  %Equal.sym(Tree.Tree<Tile.Tile8>,workspace_filled(p,storage_contract,offset,span,workspace),workspace_tabulated(p,storage_contract,offset,span,workspace),
    workspace_contents(limits,p,storage_contract,offset,span,workspace,Guards.valid(p,valid),Support.fast(limits,p,supported),span_bound)) : LeafSamples(p,offset,span,_)
  lane => group => input_channel => output_row => output_column => kernel_row => kernel_column => group_inside => position => channel_inside => row_inside => column_inside => kernel_row_inside => kernel_column_inside =>
    workspace_lane(limits,p,storage_contract,offset,span,workspace,depth,balanced,depth_ok,room,valid,supported,span_bound,lane,group,input_channel,output_row,output_column,kernel_row,kernel_column,group_inside,position,channel_inside,row_inside,column_inside,kernel_row_inside,kernel_column_inside)
