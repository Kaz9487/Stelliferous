# Logical readback of a cell window for the frozen convolution specification.
import Base
import ../specification.bend as ConvSpec
import ../inference.bend as NativeConv
import ../../lib/kernel_tile.bend as Tile
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/tensor_allocation_depth.bend as Depth
import ../../lib/proofs/kernel_channel_lanes.bend as Agreement
import ../../lib/proofs/kernel_packing_values_at.bend as Values
import ../../lib/proofs/traversal_indexed_values.bend as IndexedValues
import ../../lib/traversal_blocks.bend as Blocks
import ../../lib/proofs/tensor_tiled_values.bend as TiledValues
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/storage_buffer.bend as Storage
import ./convolution_guard_facts.bend as Guards
import ./input_storage.bend as Inputs
import ./window_cell_contents.bend as Cells
import ./window_contents.bend as WindowBounds
import ./window_problem.bend as WindowProblem
import ./window_band_values.bend as BandValues
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./reduction_coordinate_bounds.bend as Coordinates
import ./supported_coordinate_bounds.bend as SupportedCoordinates
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/tensor_tiled_cells.bend as TiledCells
import ../../lib/tensor_grid.bend as Grid
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/proofs/kernel_weights_refinement.bend as Weights
import ../../lib/proofs/word_multiplication.bend as WordMultiplication
import ../../lib/proofs/u32_division.bend as WordDivision

type Context is Data:
  Context{limits: Machine.Limits,problem: ConvSpec.Problem,storage: Inputs.Contract<problem>,tree: Tree.Tree<Tile.Tile8>,base: U32,first: U32,rows: U32,channels: U32,width: U32,
    valid: {ConvSpec.valid(problem) == True{} : Bool},supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
    channels_exact: {U32.to_nat(channels) == Guards.input_channel_count(problem) : Nat},width_exact: {U32.to_nat(width) == Guards.width(problem) : Nat},
    rows_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Guards.height(problem)) == True{} : Bool},
    rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
    contents: Cells.Contents(problem,Cells.input_cells(problem,storage),tree,U32.to_nat(base),U32.to_nat(first),U32.to_nat(rows))}

def values(context: Context) -> Tree.Tree<F32>:
  Context{limits,problem,storage,+tree,+base,first,+rows,+channels,+width,valid,supported,channels_exact,width_exact,rows_fit,rows_positive,contents} = context
  Values.values_at(base,channels,rows,width,tree)

def band(context: Context) -> BandValues.Band:
  Context{limits,+problem,storage,tree,base,+first,+rows,channels,width,valid,supported,channels_exact,width_exact,rows_fit,rows_positive,contents} = context
  WindowProblem.band(problem,U32.to_nat(first),U32.to_nat(rows))

def count(context: Context) -> Nat:
  Context{limits,+problem,storage,tree,base,first,+rows,channels,width,valid,supported,channels_exact,width_exact,rows_fit,rows_positive,contents} = context
  WindowProblem.count(problem,U32.to_nat(rows))

def channels_of(+context: Context) -> U32:
  Context{limits,problem,storage,tree,base,first,rows,+channels,width,valid,supported,channels_exact,width_exact,rows_fit,rows_positive,contents} = context
  channels

def rows_of(+context: Context) -> U32:
  Context{limits,problem,storage,tree,base,first,+rows,channels,width,valid,supported,channels_exact,width_exact,rows_fit,rows_positive,contents} = context
  rows

def width_of(+context: Context) -> U32:
  Context{limits,problem,storage,tree,base,first,rows,channels,+width,valid,supported,channels_exact,width_exact,rows_fit,rows_positive,contents} = context
  width

def geometry(+limits: Machine.Limits,+problem: ConvSpec.Problem,+channels: U32,+rows: U32,+width: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +channels_exact: {U32.to_nat(channels) == Guards.input_channel_count(problem) : Nat},+width_exact: {U32.to_nat(width) == Guards.width(problem) : Nat},
  +rows_le: {Nat.is_le(U32.to_nat(rows),Guards.height(problem)) == True{} : Bool}) -> Agreement.Fits<channels,rows,width,Grid.groups(channels)>:
  +capacity = Machine.capacity(limits)
  +cap = U32.to_nat(capacity)
  +c = U32.to_nat(channels)
  +r = U32.to_nat(rows)
  +w = U32.to_nat(width)
  +plane = (rows * width : U32)
  +area = Nat.mul(Guards.height(problem),Guards.width(problem))
  +plane_room = {%Equal.sym(Nat,w,Guards.width(problem),width_exact) : {Nat.is_le(Nat.mul(r,_),cap) == True{} : Bool}
    Order.transitive(Nat.mul(r,Guards.width(problem)),area,cap,RowMajor.mul_le(r,Guards.height(problem),Guards.width(problem),rows_le),WindowBounds.area_room(limits,problem,valid,supported)) :
    {Nat.is_le(Nat.mul(r,w),cap) == True{} : Bool}}
  +plane_exact = WordMultiplication.bounded_multiply(rows,width,capacity,Limits.capacity_fits(limits),plane_room)
  +total_room = {%Equal.sym(Nat,c,Guards.input_channel_count(problem),channels_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(plane)),cap) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(plane),Nat.mul(r,w),plane_exact) : {Nat.is_le(Nat.mul(Guards.input_channel_count(problem),_),cap) == True{} : Bool}
    %Equal.sym(Nat,w,Guards.width(problem),width_exact) : {Nat.is_le(Nat.mul(Guards.input_channel_count(problem),Nat.mul(r,_)),cap) == True{} : Bool}
    Order.transitive(Nat.mul(Guards.input_channel_count(problem),Nat.mul(r,Guards.width(problem))),Nat.mul(Guards.input_channel_count(problem),area),cap,
      DivisionOrder.multiply_left_le(Guards.input_channel_count(problem),Nat.mul(r,Guards.width(problem)),area,RowMajor.mul_le(r,Guards.height(problem),Guards.width(problem),rows_le)),
      WindowBounds.input_room(limits,problem,Padded.scalar(limits,problem,Support.fast(limits,problem,supported)),valid)) : {Nat.is_le(Nat.mul(c,U32.to_nat(plane)),cap) == True{} : Bool}}
  +total = (channels * plane : U32)
  +total_exact = WordMultiplication.bounded_multiply(channels,plane,capacity,Limits.capacity_fits(limits),total_room)
  Agreement.Fits{limits,{==},Equal.sym(Nat,U32.to_nat(plane),Nat.mul(r,w),plane_exact),Equal.sym(Nat,U32.to_nat(total),Nat.mul(c,U32.to_nat(plane)),total_exact),
    %Equal.sym(Bool,U32.is_le(total,capacity),Nat.is_le(U32.to_nat(total),cap),Comparison.le(total,capacity)) : {_ == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(total),Nat.mul(c,U32.to_nat(plane)),total_exact) : {Nat.is_le(_,cap) == True{} : Bool} total_room}

# The rows of a supported image.
def image_rows(+limits: Machine.Limits,+problem: ConvSpec.Problem,+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(U32.from_nat(Guards.height(problem))),Guards.height(problem)) == True{} : Bool}:
  +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.height(problem))),Guards.height(problem),SupportedCoordinates.height_exact(limits,problem,support)) :
    {Nat.is_le(_,Guards.height(problem)) == True{} : Bool}
  Order.reflexive(Guards.height(problem))

def fits(+context: Context) -> Agreement.Fits<channels_of(context),rows_of(context),width_of(context),Grid.groups(channels_of(context))>:
  Context{+limits,+problem,storage,tree,base,+first,+rows,+channels,+width,+valid,+supported,+channels_exact,+width_exact,+rows_fit,rows_positive,contents} = context
  geometry(limits,problem,channels,rows,width,valid,supported,channels_exact,width_exact,
    Order.transitive(U32.to_nat(rows),Nat.add(U32.to_nat(first),U32.to_nat(rows)),Guards.height(problem),Order.right_le(U32.to_nat(first),U32.to_nat(rows)),rows_fit))

# The groups of eight of the channels of a supported problem.
def group_count_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+channels: U32,
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +channels_exact: {U32.to_nat(channels) == Guards.input_channel_count(problem) : Nat}) -> {U32.to_nat(Gemm.blocks(channels)) == Cells.group_count(problem) : Nat}:
  +ci = Guards.input_channel_count(problem)
  +scalar = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  +ci_bound = Limits.dimension_capacity(limits,ci,SupportedCoordinates.ci_bound(limits,problem,scalar))
  %channels_exact : {U32.to_nat(Gemm.blocks(channels)) == TiledValues.groups_of(_) : Nat}
  Encoded.eight_blocks_exact(limits,channels,%Equal.sym(Nat,U32.to_nat(channels),ci,channels_exact) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} ci_bound)

def read_with(+context: Context,+geometry: Agreement.Fits<channels_of(context),rows_of(context),width_of(context),Grid.groups(channels_of(context))>,+index: Nat,+inside: {Nat.is_lt(index,count(context)) == True{} : Bool}) ->
  {Tree.read(F32,values(context),U32.from_nat(index)) == BandValues.value(band(context),index) : F32}:
  Context{+limits,+problem,+storage,+tree,+base,+first,+rows,+channels,+width,+valid,+supported,+channels_exact,+width_exact,+rows_fit,+rows_positive,+contents} = context
  Agreement.Fits{+machine,+grouped,+plane_forward,+total_exact,+bounded} = geometry
  +plane = (rows * width : U32)
  +p = Nat.mul(U32.to_nat(rows),Guards.width(problem))
  +r = U32.to_nat(rows)
  +ci = Guards.input_channel_count(problem)
  +cap = U32.to_nat(Machine.capacity(limits))
  +scalar = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  +groups_exact = group_count_exact(limits,problem,channels,supported,channels_exact)
  +plane_exact = {%width_exact : {U32.to_nat(plane) == Nat.mul(r,_) : Nat}
    Equal.sym(Nat,Nat.mul(r,U32.to_nat(width)),U32.to_nat(plane),plane_forward) : {U32.to_nat(plane) == p : Nat}}
  +positive = Order.product_positive(r,Guards.width(problem),rows_positive,WindowBounds.valid_condition(problem,valid,7n))
  +channel = Nat.div(index,p)
  +offset = Nat.mod(index,p)
  +channel_inside = DivisionOrder.quotient_below(index,ci,p,positive,inside)
  +offset_inside = DivisionOrder.remainder_small(index,p,positive)
  +coordinate = DivisionOrder.quotient_exact(index,p,positive)
  +selected = TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(U32.to_nat(base),Nat.add(Nat.mul(offset,Cells.group_count(problem)),Nat.div(channel,8n))))))
  %Equal.sym(Nat,index,Nat.add(Nat.mul(channel,p),offset),coordinate) : {Tree.read(F32,Values.values_at(base,channels,rows,width,tree),U32.from_nat(_)) == BandValues.value(WindowProblem.band(problem,U32.to_nat(first),r),_) : F32}
  Equal.trans(F32,Tree.read(F32,Values.values_at(base,channels,rows,width,tree),U32.from_nat(Nat.add(Nat.mul(channel,p),offset))),selected,
    BandValues.value(WindowProblem.band(problem,U32.to_nat(first),r),Nat.add(Nat.mul(channel,p),offset)),
    %plane_exact : {Tree.read(F32,Values.values_at(base,channels,rows,width,tree),U32.from_nat(Nat.add(Nat.mul(channel,_),offset))) == selected : F32}
    %groups_exact :
      {Tree.read(F32,Values.values_at(base,channels,rows,width,tree),U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(plane)),offset))) ==
        TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(U32.to_nat(base),Nat.add(Nat.mul(offset,_),Nat.div(channel,8n)))))) : F32}
    Values.values_cell(base,channels,rows,width,Agreement.Fits{machine,grouped,plane_forward,total_exact,bounded},tree,channel,offset,
      %Equal.sym(Nat,U32.to_nat(channels),ci,channels_exact) : {Nat.is_lt(channel,_) == True{} : Bool} channel_inside,
      %Equal.sym(Nat,U32.to_nat(plane),p,plane_exact) : {Nat.is_lt(offset,_) == True{} : Bool} offset_inside),
    Cells.window_lane(limits,problem,storage,tree,U32.to_nat(base),U32.to_nat(first),r,channel,offset,contents,valid,supported,rows_fit,channel_inside,offset_inside))

def read(+context: Context,+index: Nat,+inside: {Nat.is_lt(index,count(context)) == True{} : Bool}) ->
  {Tree.read(F32,values(context),U32.from_nat(index)) == BandValues.value(band(context),index) : F32}:
  read_with(context,fits(context),index,inside)

def listed(+context: Context) -> BandValues.Contents(count(context),band(context),values(context)):
  BandValues.listed(~Context,~values,~band,~count,~read,count(context),0n,context,Order.reflexive(count(context)))

def room_with(+context: Context,+geometry: Agreement.Fits<channels_of(context),rows_of(context),width_of(context),Grid.groups(channels_of(context))>) -> {Nat.is_le(count(context),U32.to_nat(Tree.capacity(F32,values(context)))) == True{} : Bool}:
  Context{+limits,+problem,+storage,+tree,+base,+first,+rows,+channels,+width,+valid,+supported,+channels_exact,+width_exact,+rows_fit,+rows_positive,+contents} = context
  Agreement.Fits{+machine,+grouped,+plane_exact,+total_exact,+bounded} = geometry
  +r = U32.to_nat(rows)
  +rows_le = Order.transitive(r,Nat.add(U32.to_nat(first),r),Guards.height(problem),Order.right_le(U32.to_nat(first),r),rows_fit)
  +plane = (rows * width : U32)
  +total = (channels * plane : U32)
  +depth = Storage.depth_for(total)
  +logical = Nat.mul(Guards.input_channel_count(problem),Nat.mul(r,Guards.width(problem)))
  +exact = {%channels_exact : {Nat.mul(_,Nat.mul(r,Guards.width(problem))) == U32.to_nat(total) : Nat}
    %width_exact : {Nat.mul(U32.to_nat(channels),Nat.mul(r,_)) == U32.to_nat(total) : Nat}
    %Equal.sym(Nat,Nat.mul(r,U32.to_nat(width)),U32.to_nat(plane),plane_exact) : {Nat.mul(U32.to_nat(channels),_) == U32.to_nat(total) : Nat}
    total_exact : {logical == U32.to_nat(total) : Nat}}
  +fresh = Tree.fresh(F32,depth,0.0)
  %Equal.sym(U32,Tree.capacity(F32,Values.values_at(base,channels,rows,width,tree)),Tree.capacity(F32,fresh),
    IndexedValues.capacity(~Tree.Tree<Tile.Tile8>,~F32,~Values.At,~Values.at_sample,~Blocks.in_place(~Values.At),U32.to_nat(total),0,Values.At{base,plane,Grid.groups(channels)},tree,fresh)) :
    {Nat.is_le(logical,U32.to_nat(_)) == True{} : Bool}
  %Equal.sym(U32,Tree.capacity(F32,fresh),U32.shln(1,depth),Shape.capacity(F32,depth,fresh,Shape.fresh(F32,depth,0.0))) :
    {Nat.is_le(logical,U32.to_nat(_)) == True{} : Bool}
  %Equal.sym(Nat,logical,U32.to_nat(total),exact) : {Nat.is_le(_,U32.to_nat(U32.shln(1,depth))) == True{} : Bool}
  Depth.allocation_room(total,Depth.bounded(total,Encoded.native_le_transitive(total,Machine.capacity(machine),1073741824,bounded,Limits.capacity_allocatable(machine))))

def room(+context: Context) -> {Nat.is_le(count(context),U32.to_nat(Tree.capacity(F32,values(context)))) == True{} : Bool}:
  room_with(context,fits(context))

# The root reference retains the caller's logical values and parameter owners.
def root_context(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) -> Context:
  +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  +height = Guards.height(problem)
  +rows = U32.from_nat(height)
  +height_exact = SupportedCoordinates.height_exact(limits,problem,support)
  +width = U32.from_nat(Guards.width(problem))
  +kernel = U32.from_nat(Guards.kernel(problem))
  +channels = (Coordinates.native_inner(problem) / (kernel * kernel) : U32)
  +tree = Cells.input_cells(problem,storage)
  Context{limits,problem,storage,tree,0,0,rows,channels,width,valid,supported,Coordinates.inner_channels(limits,problem,Guards.valid(problem,valid),support),SupportedCoordinates.width_exact(limits,problem,support),
    %Equal.sym(Nat,U32.to_nat(rows),height,height_exact) : {Nat.is_le(_,height) == True{} : Bool} Order.reflexive(height),
    %Equal.sym(Nat,U32.to_nat(rows),height,height_exact) : {Nat.is_lt(0n,_) == True{} : Bool} WindowBounds.valid_condition(problem,valid,8n),
    %Equal.sym(Nat,U32.to_nat(rows),height,height_exact) : Cells.Contents(problem,tree,tree,0n,0n,_)
    Cells.root(problem,tree)}

def root_band_values(+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+valid: {ConvSpec.valid(problem) == True{} : Bool}) ->
  {BandValues.values(WindowProblem.count(problem,Guards.height(problem)),0n,WindowProblem.band(problem,0n,Guards.height(problem))) == Guards.input(problem) : List<&2,F32>}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  Inputs.Contract{+input_evidence,+weight_evidence,+bias_evidence,+parameters} = storage
  StorageLaws.SequenceEvidence{+tree,+stored,+room} = input_evidence
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +storage = {Inputs.Contract{StorageLaws.SequenceEvidence{tree,stored,room},weight_evidence,bias_evidence,parameters} : Inputs.Contract<problem>}
  +total = Nat.mul(ci,Nat.mul(height,width))
  Equal.trans(List<&2,F32>,BandValues.values(total,0n,WindowProblem.band(problem,0n,height)),Tree.read_sequence(~F32,total,0,tree),input,
    Equal.sym(List<&2,F32>,Tree.read_sequence(~F32,total,0,tree),BandValues.values(total,0n,WindowProblem.band(problem,0n,height)),WindowBounds.root_contents(problem,storage,valid)),
    %Order.equal_of_is_eq(ConvSpec.length(input),total,WindowBounds.valid_condition(problem,valid,2n)) : {Tree.read_sequence(~F32,_,0,tree) == input : List<&2,F32>} stored)

def root_stored(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {Tree.read_sequence(~F32,ConvSpec.length(Guards.input(problem)),0,values(root_context(limits,problem,storage,valid,supported))) == Guards.input(problem) : List<&2,F32>}:
  +context = root_context(limits,problem,storage,valid,supported)
  +height = Guards.height(problem)
  +rows = U32.from_nat(height)
  +height_exact = SupportedCoordinates.height_exact(limits,problem,Padded.scalar(limits,problem,Support.fast(limits,problem,supported)))
  +total = WindowProblem.count(problem,height)
  +length = Order.equal_of_is_eq(ConvSpec.length(Guards.input(problem)),total,WindowBounds.valid_condition(problem,valid,2n))
  %Equal.sym(Nat,ConvSpec.length(Guards.input(problem)),total,length) : {Tree.read_sequence(~F32,_,0,values(context)) == Guards.input(problem) : List<&2,F32>}
  Equal.trans(List<&2,F32>,Tree.read_sequence(~F32,total,0,values(context)),BandValues.values(total,0n,WindowProblem.band(problem,0n,height)),Guards.input(problem),
    %height_exact : {Tree.read_sequence(~F32,WindowProblem.count(problem,_),0,values(context)) == BandValues.values(WindowProblem.count(problem,_),0n,WindowProblem.band(problem,0n,_)) : List<&2,F32>}
    listed(context),root_band_values(problem,storage,valid))

def root_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {Nat.is_le(ConvSpec.length(Guards.input(problem)),U32.to_nat(Tree.capacity(F32,values(root_context(limits,problem,storage,valid,supported))))) == True{} : Bool}:
  +context = root_context(limits,problem,storage,valid,supported)
  +height = Guards.height(problem)
  +height_exact = SupportedCoordinates.height_exact(limits,problem,Padded.scalar(limits,problem,Support.fast(limits,problem,supported)))
  %Equal.sym(Nat,ConvSpec.length(Guards.input(problem)),WindowProblem.count(problem,height),
    Order.equal_of_is_eq(ConvSpec.length(Guards.input(problem)),WindowProblem.count(problem,height),WindowBounds.valid_condition(problem,valid,2n))) :
    {Nat.is_le(_,U32.to_nat(Tree.capacity(F32,values(context)))) == True{} : Bool}
  %height_exact : {Nat.is_le(WindowProblem.count(problem,_),U32.to_nat(Tree.capacity(F32,values(context)))) == True{} : Bool}
  room(context)

def root_contract(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) -> Inputs.Contract<problem>:
  Inputs.Contract{StorageLaws.SequenceEvidence{values(root_context(limits,problem,storage,valid,supported)),root_stored(limits,problem,storage,valid,supported),root_room(limits,problem,storage,valid,supported)},
    Inputs.weight_sequence(problem,storage),Inputs.bias_sequence(problem,storage),Inputs.ArrayParameters{}}

def root_values(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {Values.values_at(0,(Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem)) : U32) : U32),
      U32.from_nat(Guards.height(problem)),U32.from_nat(Guards.width(problem)),Cells.input_cells(problem,storage)) ==
    Inputs.input_tree(problem,root_contract(limits,problem,storage,valid,supported)) : Tree.Tree<F32>}:
  {==}

def root_weights(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {Weights.run(U32.from_nat(Guards.output_channel_count(problem)),Coordinates.native_inner(problem),Inputs.weight_tree(problem,storage)) ==
    Inputs.weight_cells(problem,root_contract(limits,problem,storage,valid,supported)) : Tree.Tree<Tile.Tile8>}:
  ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input,weights,bias} = problem
  Inputs.Contract{input_evidence,weight_evidence,bias_evidence,parameters} = storage
  {==}

def root_bias(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {Weights.run(U32.from_nat(Guards.output_channel_count(problem)),1,Inputs.bias_tree(problem,storage)) ==
    Inputs.bias_cells(problem,root_contract(limits,problem,storage,valid,supported)) : Tree.Tree<Tile.Tile8>}:
  ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input,weights,bias} = problem
  Inputs.Contract{input_evidence,weight_evidence,bias_evidence,parameters} = storage
  {==}
