# One packed leaf of the GEMM reads back the frozen matrix value at every
# channel and column.
import Base
import ../../lib/tensor.bend as Tensors
import ../../lib/storage_buffer.bend as Storage
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/kernel_tile.bend as Tile
import ../../lib/math_sequence.bend as Sequence
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/kernel_gemm_refinement.bend as GemmLaws
import ../../lib/proofs/kernel_gemm_spec.bend as GemmSpec
import ../../lib/proofs/kernel_gemm_readback.bend as GemmReadback
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/proofs/tensor_allocation_depth.bend as Depth
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/binary_radix.bend as Radix
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../specification.bend as ConvSpec
import ../matrix_specification.bend as MatrixSpec
import ../inference.bend as NativeConv
import ./input_storage.bend as Inputs
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ./reduction_coordinate_bounds.bend as Coordinates
import ./convolution_leaf_contract.bend as LeafContract
import ./packing_tile_bounds.bend as PackingBounds
import ./packing_readback.bend as PackingReadback
import ./packed_output_bounds.bend as Output
import ./packed_output_allocation.bend as PackedOutputAllocation
import ./output_values.bend as Values
import ../../lib/proofs/kernel_partition_bounds.bend as TilePartitionBounds
import ./weight_reorder_bounds.bend as WeightBounds
import ./gemm_dot_matches_matrix.bend as GemmDot
import ../../lib/proofs/encoded_coordinates.bend as Encoding
import ../../lib/proofs/traversal_partition_refinement.bend as PartitionLaws
import ../../lib/proofs/tensor_tiled_values.bend as TiledValues
import ../../lib/tensor_grid.bend as Grid
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/traversal_partition.bend as Partition
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/proofs/word_multiplication.bend as WordMultiplication

def rows(+problem: ConvSpec.Problem) -> U32:
  U32.from_nat(Guards.output_channel_count(problem))

# A leaf is the tile product of its span into an output of columns positions:
# its rows are the packed positions, its lanes the reordered weights.
def leaf_matrix(+problem: ConvSpec.Problem,+span: U32,+start: U32,+columns: U32) -> Gemm.TileMatrix:
  Gemm.TileMatrix{rows(problem),span,Coordinates.native_inner(problem),start,columns}

def leaf_tree(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+tiles: Tree.Tree<Tile.Tile8>,+span: U32,+start: U32,+columns: U32,+buffer: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  GemmReadback.cell_tree(leaf_matrix(problem,span,start,columns),tiles,Inputs.weight_cells(problem,storage_contract),
    Inputs.bias_cells(problem,storage_contract),buffer)

def lane_position(+offset: U32,+group: Nat,+lane: Nat,+position: Nat,
  +lane_small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +exact: {Nat.add(Nat.add(U32.to_nat(offset),Nat.mul(group,8n)),lane) == position : Nat}) ->
  {PackingReadback.logical_position(offset,group,TileLaws.lane_of(lane)) == position : Nat}:
  %Equal.sym(Nat,Sequence.index_value(8n,TileLaws.lane_of(lane)),lane,TileLaws.lane_value_of(lane,lane_small)) :
    {Nat.add(Nat.add(U32.to_nat(offset),Nat.mul(group,8n)),_) == position : Nat}
  exact

# The product of a leaf with the bounds of its readback.
def leaf_product(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+tiles: Tree.Tree<Tile.Tile8>,
  +span: U32,+blocks: Nat,+start: U32,+columns: U32,+facts: Guards.Valid<problem>,+fast: Support.FastSupported<limits,problem>,
  +span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},
  +end_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(span)),Nat.add(U32.to_nat(columns),7n)) == True{} : Bool},
  +outputs_room: {Nat.is_le(Nat.mul(Guards.output_channel_count(problem),U32.to_nat(columns)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> GemmReadback.Product:
  +channels = Guards.output_channel_count(problem)
  +rows_exact = SupportedCoordinateBounds.co_exact(limits,problem,Padded.scalar(limits,problem,fast))
  +half = Machine.half_word(limits)
  +area = Order.transitive(Nat.mul(channels,U32.to_nat(columns)),U32.to_nat(Machine.capacity(limits)),U32.to_nat(half),
    outputs_room,Bounds.as_nat_le(Machine.capacity(limits),half,Limits.capacity_within_half(limits)))
  GemmReadback.Product{rows(problem),span,Coordinates.native_inner(problem),start,columns,
    tiles,Inputs.weight_cells(problem,storage_contract),Tree.fresh(F32,0n,0.0),half,blocks,Limits.half_word_fits(limits),span_exact,
    {%Equal.sym(Nat,U32.to_nat(rows(problem)),channels,rows_exact) : {Nat.is_le(Nat.add(_,7n),U32.to_nat(half)) == True{} : Bool}
      WeightBounds.padded_rows_bound(limits,problem,fast) : {Nat.is_le(Nat.add(U32.to_nat(rows(problem)),7n),U32.to_nat(half)) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat(rows(problem)),channels,rows_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(columns)),U32.to_nat(half)) == True{} : Bool}
      area : {Nat.is_le(Nat.mul(U32.to_nat(rows(problem)),U32.to_nat(columns)),U32.to_nat(half)) == True{} : Bool}},
    end_room,
    Order.transitive(U32.to_nat(columns),Nat.mul(channels,U32.to_nat(columns)),U32.to_nat(half),
      DivisionOrder.le_multiple(channels,U32.to_nat(columns),PackedOutputAllocation.channel_positive(problem,facts)),area)}

# The ordered dot a leaf computes for one channel and column: the packed
# sample of the column's row by the weight of the channel's lane.
def leaf_dot(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+tiles: Tree.Tree<Tile.Tile8>,+channel: Nat,+column: Nat) -> F32:
  +native = Coordinates.native_inner(problem)
  GemmSpec.dot(GemmLaws.RowFirst{},GemmSpec.taps(~Gemm.Direct(),~Gemm.directs,U32.to_nat(native),Gemm.Both{(U32.from_nat(Nat.div(channel,8n)) * native : U32),(U32.from_nat(Nat.div(column,8n)) * native : U32)}),
    Nat.mod(column,8n),Nat.mod(channel,8n),GemmSpec.Source{tiles,Inputs.weight_cells(problem,storage_contract)},0.0)

# A word index times the inner width is the natural address while it fits.
def scaled_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: Guards.Valid<problem>,+fast: Support.FastSupported<limits,problem>,+index: Nat,
  +room: {Nat.is_le(Nat.mul(index,Coordinates.inner(problem)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {(U32.from_nat(index) * Coordinates.native_inner(problem) : U32) == U32.from_nat(Nat.mul(index,Coordinates.inner(problem))) : U32}:
  +inner = Coordinates.inner(problem)
  +native = Coordinates.native_inner(problem)
  +cap = Machine.capacity(limits)
  +word = U32.from_nat(index)
  +inner_exact = Coordinates.inner_exact(limits,problem,valid,Padded.scalar(limits,problem,fast))
  +index_exact = Bounds.roundtrip(index,cap,Limits.capacity_fits(limits),
    Order.transitive(index,Nat.mul(index,inner),U32.to_nat(cap),RowMajor.product_covers(index,inner,Coordinates.inner_positive(problem,valid)),room))
  Encoding.encoding((word * native : U32),Nat.mul(index,inner),
    %index_exact : {U32.to_nat((word * native : U32)) == Nat.mul(_,inner) : Nat}
    %inner_exact : {U32.to_nat((word * native : U32)) == Nat.mul(U32.to_nat(word),_) : Nat}
    WordMultiplication.bounded_multiply(word,native,cap,Limits.capacity_fits(limits),
      %Equal.sym(Nat,U32.to_nat(word),index,index_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(native)),U32.to_nat(cap)) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(native),inner,inner_exact) : {Nat.is_le(Nat.mul(index,_),U32.to_nat(cap)) == True{} : Bool}
      room))

def groups_exact(+span: U32,+blocks: Nat,+span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat}) -> {PackingBounds.groups(span) == blocks : Nat}:
  %Equal.sym(Nat,U32.to_nat(span),Nat.mul(blocks,8n),span_exact) : {Nat.div(_,8n) == blocks : Nat}
  %Addition.zero_right(Nat.mul(blocks,8n)) : {Nat.div(_,8n) == blocks : Nat}
  Radix.quotient(blocks,0n,8n,{==},{==})

def eight_position(+offset: U32,+column: Nat,+group: Nat,+lane: Nat,+position: Nat,
  +column_eight: {column == Nat.add(Nat.mul(group,8n),lane) : Nat},+exact: {Nat.add(U32.to_nat(offset),column) == position : Nat}) ->
  {Nat.add(Nat.add(U32.to_nat(offset),Nat.mul(group,8n)),lane) == position : Nat}:
  %Equal.sym(Nat,Nat.add(Nat.add(U32.to_nat(offset),Nat.mul(group,8n)),lane),Nat.add(U32.to_nat(offset),Nat.add(Nat.mul(group,8n),lane)),
    Addition.associative(U32.to_nat(offset),Nat.mul(group,8n),lane)) : {_ == position : Nat}
  %column_eight : {Nat.add(U32.to_nat(offset),_) == position : Nat}
  exact

def group_inside(+span: U32,+blocks: Nat,+column: Nat,+group: Nat,+lane: Nat,
  +span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},+column_eight: {column == Nat.add(Nat.mul(group,8n),lane) : Nat},
  +column_inside: {Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool}) -> {Nat.is_lt(group,PackingBounds.groups(span)) == True{} : Bool}:
  %Equal.sym(Nat,PackingBounds.groups(span),blocks,groups_exact(span,blocks,span_exact)) : {Nat.is_lt(group,_) == True{} : Bool}
  RowMajor.row_inside(group,blocks,8n,lane,
    %column_eight : {Nat.is_lt(_,Nat.mul(blocks,8n)) == True{} : Bool}
    %span_exact : {Nat.is_lt(column,_) == True{} : Bool}
    column_inside)

# The leaf computes the frozen ordered matrix dot of the channel and column.
def selection_matches(~Evidence: LeafContract.Evidence(),~samples: LeafContract.Sampling(~Evidence),+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +tiles: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,+blocks: Nat,+channel: Nat,+column: Nat,+output_row: Nat,+output_column: Nat,
  +certificate: Evidence(problem,offset,span,tiles,output_row,output_column),
  +span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},+span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool},
  +position: {Nat.add(U32.to_nat(offset),column) == Coordinates.position(problem,output_row,output_column) : Nat},
  +row_inside: {Nat.is_lt(output_row,ConvSpec.out_height(problem)) == True{} : Bool},+column_in_width: {Nat.is_lt(output_column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {leaf_dot(problem,storage_contract,tiles,channel,column) ==
    Values.matrix_dot(MatrixSpec.kchannels(Guards.input_channel_count(problem),0n,Guards.kernel(problem)),problem,channel,output_row,output_column,0.0) : F32}:
  +fast = Support.fast(limits,problem,supported)
  +facts = Guards.valid(problem,valid)
  +inner = Coordinates.inner(problem)
  +native = Coordinates.native_inner(problem)
  +cap = Machine.capacity(limits)
  +inner_exact = Coordinates.inner_exact(limits,problem,facts,Padded.scalar(limits,problem,fast))
  +group = Nat.div(column,8n)
  +lane = Nat.mod(column,8n)
  +block = Nat.div(channel,8n)
  +row = Nat.mod(channel,8n)
  +column_split = DivisionOrder.quotient_exact(column,8n,{==})
  +channel_split = DivisionOrder.quotient_exact(channel,8n,{==})
  +packing_group = group_inside(span,blocks,column,group,lane,span_exact,column_split,column_inside)
  +group_room = Order.transitive(Nat.mul(group,inner),Nat.mul(PackingBounds.groups(span),inner),U32.to_nat(cap),
    RowMajor.mul_le(group,PackingBounds.groups(span),inner,Bounds.strict_le(group,PackingBounds.groups(span),packing_group)),
    PackingBounds.elements_bound(limits,problem,span,fast,span_bound))
  +block_channel = Order.transitive(block,Nat.mul(block,8n),channel,RowMajor.product_covers(block,8n,{==}),DivisionOrder.product_below(channel,8n,{==}))
  +block_room = Order.transitive(Nat.mul(block,inner),Nat.mul(Guards.output_channel_count(problem),inner),U32.to_nat(cap),
    RowMajor.mul_le(block,Guards.output_channel_count(problem),inner,
      Bounds.strict_le(block,Guards.output_channel_count(problem),Order.le_lt_trans(block,channel,Guards.output_channel_count(problem),block_channel,channel_inside))),
    GemmDot.weights_capacity(limits,problem,facts,fast))
  +source = {GemmSpec.Source{tiles,Inputs.weight_cells(problem,storage_contract)} : GemmSpec.Source}
  +matrix = Values.matrix_dot(MatrixSpec.kchannels(Guards.input_channel_count(problem),0n,Guards.kernel(problem)),problem,channel,output_row,output_column,0.0)
  %Equal.sym(Nat,U32.to_nat(native),inner,inner_exact) :
    {GemmSpec.dot(GemmLaws.RowFirst{},GemmSpec.taps(~Gemm.Direct(),~Gemm.directs,_,Gemm.Both{(U32.from_nat(block) * native : U32),(U32.from_nat(group) * native : U32)}),lane,row,source,0.0) == matrix : F32}
  %Equal.sym(U32,(U32.from_nat(block) * native : U32),U32.from_nat(Nat.mul(block,inner)),scaled_exact(limits,problem,facts,fast,block,block_room)) :
    {GemmSpec.dot(GemmLaws.RowFirst{},GemmSpec.taps(~Gemm.Direct(),~Gemm.directs,inner,Gemm.Both{_,(U32.from_nat(group) * native : U32)}),lane,row,source,0.0) == matrix : F32}
  %Equal.sym(U32,(U32.from_nat(group) * native : U32),U32.from_nat(Nat.mul(group,inner)),scaled_exact(limits,problem,facts,fast,group,group_room)) :
    {GemmSpec.dot(GemmLaws.RowFirst{},GemmSpec.taps(~Gemm.Direct(),~Gemm.directs,inner,Gemm.Both{U32.from_nat(Nat.mul(block,inner)),_}),lane,row,source,0.0) == matrix : F32}
  GemmDot.job_dot(~Evidence,~samples,limits,problem,storage_contract,valid,supported,tiles,offset,span,group,block,row,lane,channel,output_row,output_column,
    certificate,packing_group,
    lane_position(offset,group,lane,Coordinates.position(problem,output_row,output_column),DivisionOrder.remainder_small(column,8n,{==}),
      eight_position(offset,column,group,lane,Coordinates.position(problem,output_row,output_column),column_split,position)),
    row_inside,column_in_width,DivisionOrder.remainder_small(channel,8n,{==}),channel_split,channel_inside)

def wanted_matches(~Evidence: LeafContract.Evidence(),~samples: LeafContract.Sampling(~Evidence),+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +tiles: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,+blocks: Nat,+start: U32,+columns: U32,+channel: Nat,+column: Nat,+output_row: Nat,+output_column: Nat,
  +certificate: Evidence(problem,offset,span,tiles,output_row,output_column),
  +span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},+span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(span)),Nat.add(U32.to_nat(columns),7n)) == True{} : Bool},
  +outputs_room: {Nat.is_le(Nat.mul(Guards.output_channel_count(problem),U32.to_nat(columns)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool},
  +in_output: {Nat.is_lt(Nat.add(U32.to_nat(start),column),U32.to_nat(columns)) == True{} : Bool},
  +position: {Nat.add(U32.to_nat(offset),column) == Coordinates.position(problem,output_row,output_column) : Nat},
  +row_inside: {Nat.is_lt(output_row,ConvSpec.out_height(problem)) == True{} : Bool},+column_in_width: {Nat.is_lt(output_column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {GemmReadback.cell_lane_wanted(GemmReadback.CellProduct{leaf_product(limits,problem,storage_contract,tiles,span,blocks,start,columns,
      Guards.valid(problem,valid),Support.fast(limits,problem,supported),span_exact,end_room,outputs_room),Inputs.bias_cells(problem,storage_contract)},channel,column) ==
    Values.matrix_value(problem,channel,output_row,output_column) : F32}:
  +selection = leaf_dot(problem,storage_contract,tiles,channel,column)
  +matrix = Values.matrix_dot(MatrixSpec.kchannels(Guards.input_channel_count(problem),0n,Guards.kernel(problem)),problem,channel,output_row,output_column,0.0)
  +bias_read = TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,Inputs.bias_cells(problem,storage_contract),U32.from_nat(Nat.div(channel,8n))))
  %Equal.sym(F32,selection,matrix,selection_matches(~Evidence,~samples,limits,problem,storage_contract,valid,supported,tiles,offset,span,blocks,channel,column,output_row,output_column,
    certificate,span_exact,span_bound,channel_inside,column_inside,position,row_inside,column_in_width)) :
    {(_ + bias_read : F32) == Values.matrix_value(problem,channel,output_row,output_column) : F32}
  %Equal.sym(F32,bias_read,ConvSpec.bias_at(problem,channel),GemmDot.bias_value(limits,problem,storage_contract,channel,valid,supported,channel_inside)) :
    {(matrix + _ : F32) == Values.matrix_value(problem,channel,output_row,output_column) : F32}
  {==}

# A leaf written at output_start of an output of columns positions: the lane
# of a channel in the tile of its group.
def relocated_readback(~Evidence: LeafContract.Evidence(),~samples: LeafContract.Sampling(~Evidence),+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +tiles: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,+blocks: Nat,+channel: Nat,+column: Nat,+output_row: Nat,+output_column: Nat,
  +certificate: Evidence(problem,offset,span,tiles,output_row,output_column),
  +span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},+span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool},
  +position: {Nat.add(U32.to_nat(offset),column) == Coordinates.position(problem,output_row,output_column) : Nat},
  +row_inside: {Nat.is_lt(output_row,ConvSpec.out_height(problem)) == True{} : Bool},+column_in_width: {Nat.is_lt(output_column,ConvSpec.out_width(problem)) == True{} : Bool},
  +output_start: U32,+columns: U32,+buffer: Tree.Tree<Tile.Tile8>,+depth: Nat,+balanced: Shape.Balanced(Tile.Tile8,depth,buffer),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +output_room: {Nat.is_le(Nat.mul(U32.to_nat(columns),U32.to_nat(Gemm.blocks(rows(problem)))),U32.to_nat(Tree.capacity(Tile.Tile8,buffer))) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(U32.to_nat(output_start),U32.to_nat(span)),Nat.add(U32.to_nat(columns),7n)) == True{} : Bool},
  +outputs_room: {Nat.is_le(Nat.mul(Guards.output_channel_count(problem),U32.to_nat(columns)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +in_output: {Nat.is_lt(Nat.add(U32.to_nat(output_start),column),U32.to_nat(columns)) == True{} : Bool}) ->
  {TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,leaf_tree(problem,storage_contract,tiles,span,output_start,columns,buffer),
      U32.from_nat(Nat.add(Nat.mul(Nat.add(U32.to_nat(output_start),column),U32.to_nat(Gemm.blocks(rows(problem)))),Nat.div(channel,8n))))) ==
    Values.matrix_value(problem,channel,output_row,output_column) : F32}:
  +fast = Support.fast(limits,problem,supported)
  +rows_exact = SupportedCoordinateBounds.co_exact(limits,problem,Padded.scalar(limits,problem,fast))
  +product = leaf_product(limits,problem,storage_contract,tiles,span,blocks,output_start,columns,Guards.valid(problem,valid),fast,span_exact,end_room,outputs_room)
  +channel_rows = {%Equal.sym(Nat,U32.to_nat(rows(problem)),Guards.output_channel_count(problem),rows_exact) : {Nat.is_lt(channel,_) == True{} : Bool}
    channel_inside : {Nat.is_lt(channel,U32.to_nat(rows(problem))) == True{} : Bool}}
  Equal.trans(F32,TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,leaf_tree(problem,storage_contract,tiles,span,output_start,columns,buffer),
      U32.from_nat(Nat.add(Nat.mul(Nat.add(U32.to_nat(output_start),column),U32.to_nat(Gemm.blocks(rows(problem)))),Nat.div(channel,8n))))),
    GemmReadback.cell_lane_wanted(GemmReadback.CellProduct{product,Inputs.bias_cells(problem,storage_contract)},channel,column),Values.matrix_value(problem,channel,output_row,output_column),
    GemmReadback.cell_lane_readback(GemmReadback.CellProduct{product,Inputs.bias_cells(problem,storage_contract)},channel,column,channel_rows,in_output,column_inside,depth,buffer,balanced,depth_ok,output_room),
    wanted_matches(~Evidence,~samples,limits,problem,storage_contract,valid,supported,tiles,offset,span,blocks,output_start,columns,channel,column,output_row,output_column,
      certificate,span_exact,span_bound,end_room,outputs_room,channel_inside,column_inside,in_output,position,row_inside,column_in_width))
