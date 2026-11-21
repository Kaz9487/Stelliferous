# A GEMM job computes the frozen ordered matrix dot for every valid row and
# lane: packed input samples and reordered weights are the spec values.
import Base
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/proofs/kernel_gemm_refinement.bend as GemmLaws
import ../../lib/proofs/kernel_gemm_spec.bend as GemmSpec
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/kernel_tile.bend as Tile
import ../../lib/math_sequence.bend as Sequence
import ../../lib/ordered_matrix.bend as OrderedMatrix
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/traversal_reduction_refinement.bend as ReductionLaws
import ../../lib/proofs/ordered_reduction_split.bend as Ordered
import ../matrix_specification.bend as MatrixSpec
import ./matrix_reduction_order.bend as MatrixOrder
import ../specification.bend as ConvSpec
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/kernel_weights_refinement.bend as WeightsLaws
import ../../lib/proofs/kernel_weights_readback.bend as WeightsReadback
import ../../lib/proofs/kernel_block_geometry.bend as Geometry
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/proofs/natural_addition.bend as Addition
import ./input_storage.bend as Inputs
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ./native_weight_refinement.bend as NativeWeightRefinement
import ./reduction_coordinate_bounds.bend as Coordinates
import ./weight_reorder_bounds.bend as WeightBounds
import ./weight_reorder_readback.bend as WeightReadback
import ./packed_matrix_accessors.bend as Accessors
import ./convolution_leaf_contract.bend as LeafContract
import ./packing_tile_bounds.bend as PackingBounds
import ./packing_readback.bend as PackingReadback
import ./matrix_reduction_bounds.bend as ReductionBounds
import ./output_values.bend as Values
import ../inference.bend as NativeConv
import ../../lib/proofs/encoded_coordinates.bend as Encoding
import ../../lib/proofs/tensor_tiled_values.bend as TiledValues
import ../../lib/proofs/tensor_allocation_depth.bend as Depth
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/kernel_weights.bend as Weights
import ../../lib/tensor.bend as Tensors
import ../../lib/proofs/word_multiplication.bend as WordMultiplication

# Step k of a job is the frozen reduction index k: input tile group*inner + k
# and weight tile block*inner + k.
def Mapped() -> Data:
  Gemm.Both<MatrixOrder.TapLayout,GemmSpec.Source>

def mapped_input() -> Mapped() -> Nat -> MatrixSpec.Reduction -> F32:
  Ordered.mapped_sample(~MatrixSpec.Reduction,~ReductionLaws.AddressPair,~Nat,~GemmSpec.Source,~MatrixOrder.TapLayout,~MatrixOrder.mapped_tap,~GemmSpec.lane_element)

def mapped_weight() -> Mapped() -> Nat -> MatrixSpec.Reduction -> F32:
  Ordered.mapped_sample(~MatrixSpec.Reduction,~ReductionLaws.AddressPair,~Nat,~GemmSpec.Source,~MatrixOrder.TapLayout,~MatrixOrder.mapped_tap,~GemmSpec.row_element)

# The packed input sample of a lane times the weight of a row, in the frozen
# reduction order.
def matrix_keyed_dot(+row: Nat,+lane: Nat,+channels: Nat,+kernel: Nat,+input_base: Nat,+weight_base: Nat,+source: GemmSpec.Source) -> F32:
  OrderedMatrix.dot(~Nat,~MatrixSpec.Reduction,~Nat,~Mapped(),~mapped_weight(),~mapped_input(),
    MatrixSpec.kchannels(channels,0n,kernel),row,lane,Gemm.Both{MatrixOrder.TapLayout{kernel,input_base,weight_base},source},0.0)

# The job reads consecutive K addresses; they are exactly the frozen K axis.
def reindexed(+row: Nat,+lane: Nat,+source: GemmSpec.Source,+channels: Nat,+kernel: Nat,+input_base: Nat,+weight_base: Nat) ->
  {GemmSpec.dot(GemmLaws.RowFirst{},GemmSpec.taps(~Gemm.Direct(),~Gemm.directs,Nat.mul(channels,Nat.mul(kernel,kernel)),Gemm.Both{U32.from_nat(input_base),U32.from_nat(weight_base)}),lane,row,source,0.0) ==
    matrix_keyed_dot(row,lane,channels,kernel,input_base,weight_base,source) : F32}:
  +count = Nat.mul(channels,Nat.mul(kernel,kernel))
  +wanted = matrix_keyed_dot(row,lane,channels,kernel,input_base,weight_base,source)
  %Equal.sym(List<&2,ReductionLaws.AddressPair>,GemmSpec.taps(~Gemm.Direct(),~Gemm.directs,count,Gemm.Both{U32.from_nat(input_base),U32.from_nat(weight_base)}),
    ReductionLaws.addresses(count,U32.from_nat(input_base),U32.from_nat(weight_base)),GemmSpec.direct_taps(count,U32.from_nat(input_base),U32.from_nat(weight_base))) :
    {GemmSpec.dot(GemmLaws.RowFirst{},_,lane,row,source,0.0) == wanted : F32}
  %Equal.sym(List<&2,ReductionLaws.AddressPair>,ReductionLaws.addresses(count,U32.from_nat(input_base),U32.from_nat(weight_base)),
    MatrixOrder.matrix_taps(MatrixSpec.kchannels(channels,0n,kernel),kernel,input_base,weight_base),
    MatrixOrder.kernel_taps_follow_matrix_reduction(channels,kernel,input_base,weight_base)) :
    {GemmSpec.dot(GemmLaws.RowFirst{},_,lane,row,source,0.0) == wanted : F32}
  Ordered.dot_reindexed(~MatrixSpec.Reduction,~ReductionLaws.AddressPair,~Nat,~Nat,~GemmSpec.Source,~MatrixOrder.TapLayout,~MatrixOrder.mapped_tap,
    ~GemmSpec.row_element,~GemmSpec.lane_element,MatrixSpec.kchannels(channels,0n,kernel),MatrixOrder.TapLayout{kernel,input_base,weight_base},row,lane,source,0.0)

def reordered(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>) -> Tree.Tree<Tile.Tile8>:
  WeightsLaws.run(U32.from_nat(Guards.output_channel_count(problem)),Coordinates.native_inner(problem),Inputs.weight_tree(problem,storage_contract))

def weights_capacity(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: Guards.Valid<problem>,+fast: Support.FastSupported<limits,problem>) ->
  {Nat.is_le(Nat.mul(Guards.output_channel_count(problem),Coordinates.inner(problem)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Guards.Valid{v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10} = valid
  %Order.equal_of_is_eq(ConvSpec.length(Guards.weights(problem)),Nat.mul(Guards.output_channel_count(problem),Coordinates.inner(problem)),v9) :
    {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  NativeWeightRefinement.weights_bound(limits,problem,Padded.scalar(limits,problem,fast))

def half_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,+fast: Support.FastSupported<limits,problem>,+value: Nat,
  inside: {Nat.is_lt(value,Guards.output_channel_count(problem)) == True{} : Bool}) -> {Nat.is_le(value,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  Order.transitive(value,Nat.add(Guards.output_channel_count(problem),7n),U32.to_nat(Machine.half_word(limits)),
    Order.transitive(value,Guards.output_channel_count(problem),Nat.add(Guards.output_channel_count(problem),7n),Bounds.strict_le(value,Guards.output_channel_count(problem),inside),
      Order.le_add(Guards.output_channel_count(problem),7n)),
    WeightBounds.padded_rows_bound(limits,problem,fast))

def offset_row(+limits: Machine.Limits,+first: U32,+base: Nat,+offset: Nat,+channel: Nat,
  +first_exact: {U32.to_nat(first) == base : Nat},+channel_exact: {channel == Nat.add(base,offset) : Nat},
  +channel_small: {Nat.is_le(channel,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  {U32.to_nat((first + U32.from_nat(offset) : U32)) == channel : Nat}:
  +half = Machine.half_word(limits)
  +sum_small = {%channel_exact : {Nat.is_le(_,U32.to_nat(half)) == True{} : Bool}
    channel_small : {Nat.is_le(Nat.add(base,offset),U32.to_nat(half)) == True{} : Bool}}
  +offset_exact = Bounds.roundtrip(offset,half,Limits.half_word_fits(limits),
    Order.transitive(offset,Nat.add(base,offset),U32.to_nat(half),
      %Order.commutative(offset,base) : {Nat.is_le(offset,_) == True{} : Bool}
      Order.le_add(offset,base),sum_small))
  %Equal.sym(Nat,channel,Nat.add(base,offset),channel_exact) : {U32.to_nat((first + U32.from_nat(offset) : U32)) == _ : Nat}
  %first_exact : {U32.to_nat((first + U32.from_nat(offset) : U32)) == Nat.add(_,offset) : Nat}
  %offset_exact : {U32.to_nat((first + U32.from_nat(offset) : U32)) == Nat.add(U32.to_nat(first),_) : Nat}
  Arithmetic.add(first,U32.from_nat(offset),half,Limits.half_word_fits(limits),
    %Equal.sym(Nat,U32.to_nat(first),base,first_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(U32.from_nat(offset))),U32.to_nat(half)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(offset)),offset,offset_exact) : {Nat.is_le(Nat.add(base,_),U32.to_nat(half)) == True{} : Bool}
    sum_small)

def first_word_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+fast: Support.FastSupported<limits,problem>,+block: Nat,+lane: Sequence.Index(8n),+channel: Nat,
  +channel_exact: {channel == Nat.add(Nat.mul(block,8n),Sequence.index_value(8n,lane)) : Nat},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool}) ->
  {U32.to_nat((U32.from_nat(block) * 8 : U32)) == Nat.mul(block,8n) : Nat}:
  +half = Machine.half_word(limits)
  +fits = Limits.half_word_fits(limits)
  +channel_small = half_room(limits,problem,fast,channel,channel_inside)
  +first_small = Order.transitive(Nat.mul(block,8n),channel,U32.to_nat(half),
    %Equal.sym(Nat,channel,Nat.add(Nat.mul(block,8n),Sequence.index_value(8n,lane)),channel_exact) : {Nat.is_le(Nat.mul(block,8n),_) == True{} : Bool}
    Order.le_add(Nat.mul(block,8n),Sequence.index_value(8n,lane)),channel_small)
  +block_exact = Bounds.roundtrip(block,half,fits,Order.transitive(block,Nat.mul(block,8n),U32.to_nat(half),RowMajor.product_covers(block,8n,{==}),first_small))
  %block_exact : {U32.to_nat((U32.from_nat(block) * 8 : U32)) == Nat.mul(_,8n) : Nat}
  WordMultiplication.bounded_multiply(U32.from_nat(block),8,half,fits,
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(block)),block,block_exact) : {Nat.is_le(Nat.mul(_,8n),U32.to_nat(half)) == True{} : Bool}
    first_small)

def lane_word_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+fast: Support.FastSupported<limits,problem>,+block: Nat,+lane: Sequence.Index(8n),+channel: Nat,
  +channel_exact: {channel == Nat.add(Nat.mul(block,8n),Sequence.index_value(8n,lane)) : Nat},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool}) ->
  {U32.to_nat(WeightsReadback.lane_word((U32.from_nat(block) * 8 : U32),lane)) == channel : Nat}:
  match lane:
    case Sequence.First{}:
      %Equal.sym(Nat,channel,Nat.add(Nat.mul(block,8n),0n),channel_exact) : {U32.to_nat((U32.from_nat(block) * 8 : U32)) == _ : Nat}
      %Equal.sym(Nat,Nat.add(Nat.mul(block,8n),0n),Nat.mul(block,8n),Addition.zero_right(Nat.mul(block,8n))) : {U32.to_nat((U32.from_nat(block) * 8 : U32)) == _ : Nat}
      first_word_exact(limits,problem,fast,block,Sequence.First{},channel,channel_exact,channel_inside)
    case Sequence.Next{Sequence.First{}}:
      offset_row(limits,(U32.from_nat(block) * 8 : U32),Nat.mul(block,8n),1n,channel,first_word_exact(limits,problem,fast,block,Sequence.Next{Sequence.First{}},channel,channel_exact,channel_inside),
        channel_exact,half_room(limits,problem,fast,channel,channel_inside))
    case Sequence.Next{Sequence.Next{Sequence.First{}}}:
      offset_row(limits,(U32.from_nat(block) * 8 : U32),Nat.mul(block,8n),2n,channel,first_word_exact(limits,problem,fast,block,Sequence.Next{Sequence.Next{Sequence.First{}}},channel,channel_exact,channel_inside),
        channel_exact,half_room(limits,problem,fast,channel,channel_inside))
    case Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}:
      offset_row(limits,(U32.from_nat(block) * 8 : U32),Nat.mul(block,8n),3n,channel,first_word_exact(limits,problem,fast,block,Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}},channel,channel_exact,channel_inside),
        channel_exact,half_room(limits,problem,fast,channel,channel_inside))
    case Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}}:
      offset_row(limits,(U32.from_nat(block) * 8 : U32),Nat.mul(block,8n),4n,channel,first_word_exact(limits,problem,fast,block,Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}},channel,channel_exact,channel_inside),
        channel_exact,half_room(limits,problem,fast,channel,channel_inside))
    case Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}}}:
      offset_row(limits,(U32.from_nat(block) * 8 : U32),Nat.mul(block,8n),5n,channel,first_word_exact(limits,problem,fast,block,Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}}},channel,channel_exact,channel_inside),
        channel_exact,half_room(limits,problem,fast,channel,channel_inside))
    case Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}}}}:
      offset_row(limits,(U32.from_nat(block) * 8 : U32),Nat.mul(block,8n),6n,channel,first_word_exact(limits,problem,fast,block,Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}}}},channel,channel_exact,channel_inside),
        channel_exact,half_room(limits,problem,fast,channel,channel_inside))
    case Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}}}}}:
      offset_row(limits,(U32.from_nat(block) * 8 : U32),Nat.mul(block,8n),7n,channel,first_word_exact(limits,problem,fast,block,Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}}}}},channel,channel_exact,channel_inside),
        channel_exact,half_room(limits,problem,fast,channel,channel_inside))
    case Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{empty}}}}}}}}:
      match empty:

def value_read(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: Guards.Valid<problem>,+fast: Support.FastSupported<limits,problem>,
  +storage_contract: Inputs.Contract<problem>,+channel: Nat,+tap: Nat,+word: U32,
  +word_exact: {U32.to_nat(word) == channel : Nat},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +tap_inside: {Nat.is_lt(tap,Coordinates.inner(problem)) == True{} : Bool}) ->
  {WeightsReadback.value(Inputs.weight_tree(problem,storage_contract),U32.from_nat(Guards.output_channel_count(problem)),Coordinates.native_inner(problem),word,U32.from_nat(tap)) ==
    Tree.read(F32,Inputs.weight_tree(problem,storage_contract),U32.from_nat(Nat.add(Nat.mul(channel,Coordinates.inner(problem)),tap))) : F32}:
  +inner = Coordinates.inner(problem)
  +native = Coordinates.native_inner(problem)
  +rows = U32.from_nat(Guards.output_channel_count(problem))
  +cap = Machine.capacity(limits)
  +fits = Limits.capacity_fits(limits)
  +target = Nat.add(Nat.mul(channel,inner),tap)
  +inner_exact = Coordinates.inner_exact(limits,problem,valid,Padded.scalar(limits,problem,fast))
  +small = Order.transitive(target,Nat.mul(Guards.output_channel_count(problem),inner),U32.to_nat(cap),
    Bounds.strict_le(target,Nat.mul(Guards.output_channel_count(problem),inner),RowMajor.radix(channel,Guards.output_channel_count(problem),tap,inner,channel_inside,tap_inside)),
    weights_capacity(limits,problem,valid,fast))
  +product_small = Order.transitive(Nat.mul(channel,inner),target,U32.to_nat(cap),Order.le_add(Nat.mul(channel,inner),tap),small)
  +tap_exact = Bounds.roundtrip(tap,cap,fits,Order.transitive(tap,target,U32.to_nat(cap),
    %Order.commutative(tap,Nat.mul(channel,inner)) : {Nat.is_le(tap,_) == True{} : Bool}
    Order.le_add(tap,Nat.mul(channel,inner)),small))
  +product = (word * native : U32)
  +product_exact = {%word_exact : {U32.to_nat(product) == Nat.mul(_,inner) : Nat}
    %inner_exact : {U32.to_nat(product) == Nat.mul(U32.to_nat(word),_) : Nat}
    WordMultiplication.bounded_multiply(word,native,cap,fits,
      %Equal.sym(Nat,U32.to_nat(word),channel,word_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(native)),U32.to_nat(cap)) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(native),inner,inner_exact) : {Nat.is_le(Nat.mul(channel,_),U32.to_nat(cap)) == True{} : Bool}
      product_small) : {U32.to_nat(product) == Nat.mul(channel,inner) : Nat}}
  +address = Encoding.encoding((product + U32.from_nat(tap) : U32),target,
    %product_exact : {U32.to_nat((product + U32.from_nat(tap) : U32)) == Nat.add(_,tap) : Nat}
    %tap_exact : {U32.to_nat((product + U32.from_nat(tap) : U32)) == Nat.add(U32.to_nat(product),_) : Nat}
    Arithmetic.add(product,U32.from_nat(tap),cap,fits,
      %Equal.sym(Nat,U32.to_nat(product),Nat.mul(channel,inner),product_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(U32.from_nat(tap))),U32.to_nat(cap)) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(U32.from_nat(tap)),tap,tap_exact) : {Nat.is_le(Nat.add(Nat.mul(channel,inner),_),U32.to_nat(cap)) == True{} : Bool}
      small))
  +accepted = {%Equal.sym(Bool,U32.is_lt(word,rows),Nat.is_lt(U32.to_nat(word),U32.to_nat(rows)),Comparison.lt(word,rows)) : {_ == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(word),channel,word_exact) : {Nat.is_lt(_,U32.to_nat(rows)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(rows),Guards.output_channel_count(problem),SupportedCoordinateBounds.co_exact(limits,problem,Padded.scalar(limits,problem,fast))) :
      {Nat.is_lt(channel,_) == True{} : Bool}
    channel_inside : {U32.is_lt(word,rows) == True{} : Bool}}
  %Equal.sym(Bool,U32.is_lt(word,rows),True{},accepted) :
    {Bool.pick(F32,_,Tree.read(F32,Inputs.weight_tree(problem,storage_contract),(word * native + U32.from_nat(tap) : U32)),0.0) ==
      Tree.read(F32,Inputs.weight_tree(problem,storage_contract),U32.from_nat(target)) : F32}
  %Equal.sym(U32,(product + U32.from_nat(tap) : U32),U32.from_nat(target),address) :
    {Tree.read(F32,Inputs.weight_tree(problem,storage_contract),_) == Tree.read(F32,Inputs.weight_tree(problem,storage_contract),U32.from_nat(target)) : F32}
  {==}

def block_le_channel(+block: Nat,+lane: Sequence.Index(8n),+channel: Nat,+channel_exact: {channel == Nat.add(Nat.mul(block,8n),Sequence.index_value(8n,lane)) : Nat}) ->
  {Nat.is_le(Nat.mul(block,8n),channel) == True{} : Bool}:
  %Equal.sym(Nat,channel,Nat.add(Nat.mul(block,8n),Sequence.index_value(8n,lane)),channel_exact) : {Nat.is_le(Nat.mul(block,8n),_) == True{} : Bool}
  Order.le_add(Nat.mul(block,8n),Sequence.index_value(8n,lane))

# The rows of a job read the packed input tiles, its lanes the weights.
def mapped_source(+problem: ConvSpec.Problem,+weights: Tree.Tree<Tile.Tile8>,+tiles: Tree.Tree<Tile.Tile8>,+group: Nat,+block: Nat) -> Mapped():
  Gemm.Both{MatrixOrder.TapLayout{Guards.kernel(problem),Nat.mul(block,Coordinates.inner(problem)),Nat.mul(group,Coordinates.inner(problem))},
    GemmSpec.Source{tiles,weights}}

def input_agrees(~Evidence: LeafContract.Evidence(),~samples: LeafContract.Sampling(~Evidence),+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +tiles: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,+group: Nat,+block: Nat,+lane: Nat,+output_row: Nat,+output_column: Nat,+index: MatrixSpec.Reduction,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +certificate: Evidence(problem,offset,span,tiles,output_row,output_column),
  +group_inside: {Nat.is_lt(group,PackingBounds.groups(span)) == True{} : Bool},
  +position: {PackingReadback.logical_position(offset,group,TileLaws.lane_of(lane)) == Coordinates.position(problem,output_row,output_column) : Nat},
  +row_inside: {Nat.is_lt(output_row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(output_column,ConvSpec.out_width(problem)) == True{} : Bool},
  +bounds: ReductionBounds.IndexBounds(problem,index)) ->
  {mapped_weight()(mapped_source(problem,Inputs.weight_cells(problem,storage_contract),tiles,group,block),lane,index) ==
    MatrixSpec.patch(problem,MatrixSpec.Position{output_row,output_column},index) : F32}:
  MatrixSpec.KIndex{+channel,+kernel_row,+kernel_column} = index
  Gemm.Both{+channel_inside,+spatial} = bounds
  Gemm.Both{+kernel_row_inside,+kernel_column_inside} = spatial
  +tile = Tree.read(Tile.Tile8,tiles,U32.from_nat(Nat.add(Nat.mul(group,Coordinates.inner(problem)),Coordinates.reduction(channel,kernel_row,kernel_column,Guards.kernel(problem)))))
  Equal.trans(F32,TileLaws.select_lane(lane,tile),Tile.element_at(TileLaws.lane_of(lane),tile),
    MatrixSpec.patch(problem,MatrixSpec.Position{output_row,output_column},MatrixSpec.KIndex{channel,kernel_row,kernel_column}),TileLaws.selected_lane(lane,tile),
    samples(limits,problem,offset,span,tiles,output_row,output_column,valid,supported,certificate)(TileLaws.lane_of(lane),group,channel,kernel_row,kernel_column,
      group_inside,position,channel_inside,row_inside,column_inside,kernel_row_inside,kernel_column_inside))

def array_weight_agrees(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +tiles: Tree.Tree<Tile.Tile8>,+group: Nat,+block: Nat,+row: Nat,+channel: Nat,+index: MatrixSpec.Reduction,
  +row_small: {Nat.is_lt(row,8n) == True{} : Bool},+channel_exact: {channel == Nat.add(Nat.mul(block,8n),row) : Nat},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +bounds: ReductionBounds.IndexBounds(problem,index)) ->
  {mapped_input()(mapped_source(problem,reordered(problem,storage_contract),tiles,group,block),row,index) == MatrixSpec.weight(problem,channel,index) : F32}:
  MatrixSpec.KIndex{+input_channel,+kernel_row,+kernel_column} = index
  Gemm.Both{+input_inside,+spatial} = bounds
  Gemm.Both{+kernel_row_inside,+kernel_column_inside} = spatial
  +facts = Guards.valid(problem,valid)
  +fast = Support.fast(limits,problem,supported)
  +tap = Coordinates.reduction(input_channel,kernel_row,kernel_column,Guards.kernel(problem))
  +tap_inside = Coordinates.reduction_inside(problem,input_channel,kernel_row,kernel_column,input_inside,kernel_row_inside,kernel_column_inside)
  +weights = Inputs.weight_tree(problem,storage_contract)
  +rows = U32.from_nat(Guards.output_channel_count(problem))
  +native = Coordinates.native_inner(problem)
  +target = U32.from_nat(Nat.add(Nat.mul(block,Coordinates.inner(problem)),tap))
  +first = (U32.from_nat(block) * 8 : U32)
  +lane8 = TileLaws.lane_of(row)
  +lane_exact = {%Equal.sym(Nat,Sequence.index_value(8n,lane8),row,TileLaws.lane_value_of(row,row_small)) : {channel == Nat.add(Nat.mul(block,8n),_) : Nat}
    channel_exact : {channel == Nat.add(Nat.mul(block,8n),Sequence.index_value(8n,lane8)) : Nat}}
  +eight_le = block_le_channel(block,lane8,channel,lane_exact)
  +block_inside = Geometry.contains_row(Guards.output_channel_count(problem),block,Order.le_lt_trans(Nat.mul(block,8n),channel,Guards.output_channel_count(problem),eight_le,channel_inside))
  +read = Tree.read(Tile.Tile8,reordered(problem,storage_contract),target)
  %Equal.sym(F32,TileLaws.select_lane(row,read),Tile.element_at(lane8,read),TileLaws.selected_lane(row,read)) :
    {_ == MatrixSpec.weight(problem,channel,MatrixSpec.KIndex{input_channel,kernel_row,kernel_column}) : F32}
  %Equal.sym(F32,Tile.element_at(lane8,read),
    Tile.element_at(lane8,WeightsReadback.sample_linear(weights,rows,native,first,U32.from_nat(tap))),
    Equal.cong(Tile.Tile8,F32,tile => Tile.element_at(lane8,tile),read,
      WeightsReadback.sample_linear(weights,rows,native,first,U32.from_nat(tap)),WeightReadback.run_readback(limits,problem,facts,fast,weights,block,tap,block_inside,tap_inside))) :
    {_ == MatrixSpec.weight(problem,channel,MatrixSpec.KIndex{input_channel,kernel_row,kernel_column}) : F32}
  %Equal.sym(F32,Tile.element_at(lane8,WeightsReadback.sample_linear(weights,rows,native,first,U32.from_nat(tap))),
    WeightsReadback.value(weights,rows,native,WeightsReadback.lane_word(first,lane8),U32.from_nat(tap)),WeightsReadback.sample_lane(weights,rows,native,first,U32.from_nat(tap),lane8)) :
    {_ == MatrixSpec.weight(problem,channel,MatrixSpec.KIndex{input_channel,kernel_row,kernel_column}) : F32}
  %Equal.sym(F32,WeightsReadback.value(weights,rows,native,WeightsReadback.lane_word(first,lane8),U32.from_nat(tap)),
    Tree.read(F32,weights,U32.from_nat(Nat.add(Nat.mul(channel,Coordinates.inner(problem)),tap))),
    value_read(limits,problem,facts,fast,storage_contract,channel,tap,WeightsReadback.lane_word(first,lane8),
      lane_word_exact(limits,problem,fast,block,lane8,channel,lane_exact,channel_inside),channel_inside,tap_inside)) :
    {_ == MatrixSpec.weight(problem,channel,MatrixSpec.KIndex{input_channel,kernel_row,kernel_column}) : F32}
  Accessors.first_weight_matches_matrix(problem,storage_contract,tiles,0n,channel,input_channel,kernel_row,kernel_column,valid,
    channel_inside,input_inside,kernel_row_inside,kernel_column_inside)

def weight_agrees(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +tiles: Tree.Tree<Tile.Tile8>,+group: Nat,+block: Nat,+row: Nat,+channel: Nat,+index: MatrixSpec.Reduction,
  +row_small: {Nat.is_lt(row,8n) == True{} : Bool},+channel_exact: {channel == Nat.add(Nat.mul(block,8n),row) : Nat},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +bounds: ReductionBounds.IndexBounds(problem,index)) ->
  {mapped_input()(mapped_source(problem,Inputs.weight_cells(problem,storage_contract),tiles,group,block),row,index) == MatrixSpec.weight(problem,channel,index) : F32}:
  ConvSpec.Problem{shape,+xs,+ws,+bs} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  Inputs.Contract{+input,+weights,+bias,parameters} = storage_contract
  match parameters:
    case Inputs.ArrayParameters{}:
      array_weight_agrees(limits,problem,Inputs.Contract{input,weights,bias,Inputs.ArrayParameters{}},valid,supported,tiles,group,block,row,channel,index,row_small,channel_exact,channel_inside,bounds)
    case Inputs.CellParameters{+weight_cells,+bias_cells,+weight_values,+bias_values}:
      MatrixSpec.KIndex{+input_channel,+kernel_row,+kernel_column} = index
      Gemm.Both{+input_inside,Gemm.Both{+kernel_row_inside,+kernel_column_inside}} = bounds
      +channels = Guards.output_channel_count(problem)
      +inner = Coordinates.inner(problem)
      +tap = Coordinates.reduction(input_channel,kernel_row,kernel_column,Guards.kernel(problem))
      +position = Nat.add(Nat.mul(channel,inner),tap)
      +source = {TiledValues.CellSource{weight_cells,channels,inner} : TiledValues.CellSource}
      +flat = NativeWeightRefinement.weight_index(channel,Guards.input_channel_count(problem),input_channel,Guards.kernel(problem),kernel_row,kernel_column)
      +index_exact = Coordinates.weight_index(channel,Guards.input_channel_count(problem),input_channel,Guards.kernel(problem),kernel_row,kernel_column)
      +inside = {%index_exact : {Nat.is_lt(_,ConvSpec.length(Guards.weights(problem))) == True{} : Bool}
        NativeWeightRefinement.index_bound(problem,channel,input_channel,kernel_row,kernel_column,Guards.valid(problem,valid),channel_inside,input_inside,kernel_row_inside,kernel_column_inside) :
        {Nat.is_lt(position,ConvSpec.length(Guards.weights(problem))) == True{} : Bool}}
      Equal.trans(F32,TileLaws.select_lane(row,Tree.read(Tile.Tile8,weight_cells,U32.from_nat(Nat.add(Nat.mul(block,inner),tap)))),TiledValues.cell_value(source,position),
        MatrixSpec.weight(problem,channel,MatrixSpec.KIndex{input_channel,kernel_row,kernel_column}),
        %Equal.sym(Nat,channel,Nat.add(Nat.mul(block,8n),row),channel_exact) :
          {TileLaws.select_lane(row,Tree.read(Tile.Tile8,weight_cells,U32.from_nat(Nat.add(Nat.mul(block,inner),tap)))) == TiledValues.cell_value(source,Nat.add(Nat.mul(_,inner),tap)) : F32}
        TiledValues.cell_lane_at(weight_cells,channels,inner,0n,block,row,tap,row_small,
          %channel_exact : {Nat.is_lt(_,channels) == True{} : Bool} channel_inside,
          Coordinates.reduction_inside(problem,input_channel,kernel_row,kernel_column,input_inside,kernel_row_inside,kernel_column_inside)),
        Equal.trans(F32,TiledValues.cell_value(source,position),ConvSpec.at(position,Guards.weights(problem)),MatrixSpec.weight(problem,channel,MatrixSpec.KIndex{input_channel,kernel_row,kernel_column}),
          %weight_values : {TiledValues.cell_value(source,position) == ConvSpec.at(position,_) : F32}
          Inputs.parameter_at(ConvSpec.length(Guards.weights(problem)),channels,inner,weight_cells,position,inside),
          Equal.cong(Nat,F32,value => ConvSpec.at(value,Guards.weights(problem)),position,flat,Equal.sym(Nat,flat,position,index_exact))))

def array_bias_tile(+limits: Machine.Limits,+problem: ConvSpec.Problem,+tree: Tree.Tree<F32>,+channel: Nat,
  +fast: Support.FastSupported<limits,problem>,+inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,WeightsLaws.run(U32.from_nat(Guards.output_channel_count(problem)),1,tree),U32.from_nat(Nat.div(channel,8n))) ==
    WeightsReadback.sample_linear(tree,U32.from_nat(Guards.output_channel_count(problem)),1,(U32.from_nat(Nat.div(channel,8n)) * 8 : U32),0) : Tile.Tile8}:
  +rows = U32.from_nat(Guards.output_channel_count(problem))
  +blocks = WeightBounds.blocks(problem)
  +count = (blocks * 1 : U32)
  +count_exact = Equal.trans(Nat,U32.to_nat(count),U32.to_nat(blocks),Geometry.blocks(Guards.output_channel_count(problem)),
    Equal.cong(U32,Nat,U32.to_nat,count,blocks,Arithmetic.multiply_one(blocks)),WeightBounds.blocks_exact(limits,problem,fast))
  +capacity = Machine.capacity(limits)
  +bound = Order.transitive(Geometry.blocks(Guards.output_channel_count(problem)),Guards.output_channel_count(problem),U32.to_nat(capacity),Geometry.count_bound(Guards.output_channel_count(problem)),
    Limits.dimension_capacity(limits,Guards.output_channel_count(problem),SupportedCoordinateBounds.co_bound(limits,problem,Padded.scalar(limits,problem,fast))))
  +ready = Depth.bounded_with(count,capacity,Limits.capacity_fits(limits),Limits.capacity_allocatable(limits),
    %Equal.sym(Bool,U32.is_le(1,capacity),Nat.is_le(1n,U32.to_nat(capacity)),Comparison.le(1,capacity)) : {_ == True{} : Bool}
      Order.transitive(1n,7n,U32.to_nat(capacity),{==},Bounds.as_nat_le(7,capacity,Limits.seven_within_capacity(limits))),
    %Equal.sym(Bool,U32.is_le(count,capacity),Nat.is_le(U32.to_nat(count),U32.to_nat(capacity)),Comparison.le(count,capacity)) : {_ == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(count),Geometry.blocks(Guards.output_channel_count(problem)),count_exact) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} bound)
  +group = Nat.div(channel,8n)
  +query = {WeightsLaws.Query{Weights.MappedLayout{rows,1,1,Unit{}},tree,group,0n,Depth.allocation_depth(count,ready),
    %Equal.sym(Nat,Nat.mul(U32.to_nat(blocks),1n),U32.to_nat(blocks),Order.times_one(U32.to_nat(blocks))) :
      {Nat.is_le(_,U32.to_nat(U32.shln(1,Tensors.depth_for(count)))) == True{} : Bool}
    %Arithmetic.multiply_one(blocks) : {Nat.is_le(U32.to_nat(_),U32.to_nat(U32.shln(1,Tensors.depth_for(count)))) == True{} : Bool}
    Depth.allocation_room(count,ready),
    %Equal.sym(Nat,U32.to_nat(blocks),Geometry.blocks(Guards.output_channel_count(problem)),WeightBounds.blocks_exact(limits,problem,fast)) : {Nat.is_lt(group,_) == True{} : Bool}
    Geometry.contains_row(Guards.output_channel_count(problem),group,
      Order.le_lt_trans(Nat.mul(group,8n),channel,Guards.output_channel_count(problem),DivisionOrder.product_below(channel,8n,{==}),inside)),{==}} : WeightsLaws.Query<Unit>}
  %Order.times_one(group) :
    {Tree.read(Tile.Tile8,WeightsLaws.run(rows,1,tree),U32.from_nat(_)) == WeightsReadback.sample_linear(tree,rows,1,(U32.from_nat(group) * 8 : U32),0) : Tile.Tile8}
  %Addition.zero_right(Nat.mul(group,1n)) :
    {Tree.read(Tile.Tile8,WeightsLaws.run(rows,1,tree),U32.from_nat(_)) == WeightsReadback.sample_linear(tree,rows,1,(U32.from_nat(group) * 8 : U32),0) : Tile.Tile8}
  WeightsLaws.mapped_lookup(~Unit,~Weights.identity,query)

def array_bias_value(+limits: Machine.Limits,+problem: ConvSpec.Problem,+tree: Tree.Tree<F32>,+channel: Nat,
  +fast: Support.FastSupported<limits,problem>,+inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool}) ->
  {TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,WeightsLaws.run(U32.from_nat(Guards.output_channel_count(problem)),1,tree),U32.from_nat(Nat.div(channel,8n)))) ==
    Tree.read(F32,tree,U32.from_nat(channel)) : F32}:
  +rows = U32.from_nat(Guards.output_channel_count(problem))
  +group = Nat.div(channel,8n)
  +lane = Nat.mod(channel,8n)
  +lane8 = TileLaws.lane_of(lane)
  +first = (U32.from_nat(group) * 8 : U32)
  +word = WeightsReadback.lane_word(first,lane8)
  +split = DivisionOrder.quotient_exact(channel,8n,{==})
  +lane_exact = {%Equal.sym(Nat,Sequence.index_value(8n,lane8),lane,TileLaws.lane_value_of(lane,DivisionOrder.remainder_small(channel,8n,{==}))) :
      {channel == Nat.add(Nat.mul(group,8n),_) : Nat} split : {channel == Nat.add(Nat.mul(group,8n),Sequence.index_value(8n,lane8)) : Nat}}
  +word_exact = lane_word_exact(limits,problem,fast,group,lane8,channel,lane_exact,inside)
  +read = Tree.read(Tile.Tile8,WeightsLaws.run(rows,1,tree),U32.from_nat(group))
  %Equal.sym(F32,TileLaws.select_lane(lane,read),Tile.element_at(lane8,read),TileLaws.selected_lane(lane,read)) : {_ == Tree.read(F32,tree,U32.from_nat(channel)) : F32}
  %Equal.sym(Tile.Tile8,read,WeightsReadback.sample_linear(tree,rows,1,first,0),array_bias_tile(limits,problem,tree,channel,fast,inside)) :
    {Tile.element_at(lane8,_) == Tree.read(F32,tree,U32.from_nat(channel)) : F32}
  %Equal.sym(F32,Tile.element_at(lane8,WeightsReadback.sample_linear(tree,rows,1,first,0)),WeightsReadback.value(tree,rows,1,word,0),WeightsReadback.sample_lane(tree,rows,1,first,0,lane8)) :
    {_ == Tree.read(F32,tree,U32.from_nat(channel)) : F32}
  +accepted = {%Equal.sym(Bool,U32.is_lt(word,rows),Nat.is_lt(U32.to_nat(word),U32.to_nat(rows)),Comparison.lt(word,rows)) : {_ == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(word),channel,word_exact) : {Nat.is_lt(_,U32.to_nat(rows)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(rows),Guards.output_channel_count(problem),SupportedCoordinateBounds.co_exact(limits,problem,Padded.scalar(limits,problem,fast))) : {Nat.is_lt(channel,_) == True{} : Bool}
    inside : {U32.is_lt(word,rows) == True{} : Bool}}
  %Equal.sym(Bool,U32.is_lt(word,rows),True{},accepted) : {Bool.pick(F32,_,Tree.read(F32,tree,(word * 1 + 0 : U32)),0.0) == Tree.read(F32,tree,U32.from_nat(channel)) : F32}
  Equal.cong(U32,F32,Tree.read(F32,tree),(word * 1 + 0 : U32),U32.from_nat(channel),
    Equal.trans(U32,(word * 1 + 0 : U32),(word * 1 : U32),U32.from_nat(channel),Arithmetic.add_zero((word * 1 : U32)),
      Equal.trans(U32,(word * 1 : U32),word,U32.from_nat(channel),Arithmetic.multiply_one(word),Encoding.encoding(word,channel,word_exact))))

def bias_value(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+channel: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool}) ->
  {TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,Inputs.bias_cells(problem,storage),U32.from_nat(Nat.div(channel,8n)))) == ConvSpec.bias_at(problem,channel) : F32}:
  ConvSpec.Problem{shape,+xs,+ws,+bs} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  Inputs.Contract{+input,+weights,+bias,parameters} = storage
  match parameters:
    case Inputs.ArrayParameters{}:
      +storage = {Inputs.Contract{input,weights,bias,Inputs.ArrayParameters{}} : Inputs.Contract<problem>}
      +tree = Inputs.bias_tree(problem,storage)
      Equal.trans(F32,TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,WeightsLaws.run(U32.from_nat(co),1,tree),U32.from_nat(Nat.div(channel,8n)))),
        Tree.read(F32,tree,U32.from_nat(channel)),ConvSpec.bias_at(problem,channel),array_bias_value(limits,problem,tree,channel,Support.fast(limits,problem,supported),inside),
        Inputs.bias_matches_spec(problem,storage,channel,Guards.valid(problem,valid),inside))
    case Inputs.CellParameters{weight_cells,+bias_cells,weight_values,+bias_values}:
      +source = {TiledValues.CellSource{bias_cells,co,1n} : TiledValues.CellSource}
      +group = Nat.div(channel,8n)
      +lane = Nat.mod(channel,8n)
      +read = TileLaws.select_lane(lane,Tree.read(Tile.Tile8,bias_cells,U32.from_nat(group)))
      +address = Equal.trans(Nat,Nat.add(Nat.mul(group,1n),0n),Nat.mul(group,1n),group,Addition.zero_right(Nat.mul(group,1n)),Order.times_one(group))
      +position = Equal.trans(Nat,Nat.add(Nat.mul(channel,1n),0n),Nat.mul(channel,1n),channel,Addition.zero_right(Nat.mul(channel,1n)),Order.times_one(channel))
      Equal.trans(F32,read,TiledValues.cell_value(source,channel),ConvSpec.bias_at(problem,channel),
        %address : {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,bias_cells,U32.from_nat(_))) == TiledValues.cell_value(source,channel) : F32}
        %position : {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,bias_cells,U32.from_nat(Nat.add(Nat.mul(group,1n),0n)))) == TiledValues.cell_value(source,_) : F32}
        TiledValues.cell_channel_at(bias_cells,co,1n,0n,channel,0n,inside,{==}),
        %bias_values : {TiledValues.cell_value(source,channel) == ConvSpec.at(channel,_) : F32}
        Inputs.parameter_at(ConvSpec.length(bs),co,1n,bias_cells,channel,NativeWeightRefinement.bias_index(problem,channel,Guards.valid(problem,valid),inside)))

def Agreements(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+tiles: Tree.Tree<Tile.Tile8>,+group: Nat,+block: Nat,
  +row: Nat,+lane: Nat,+channel: Nat,+output_row: Nat,+output_column: Nat,+indices: List<&2,MatrixSpec.Reduction>) -> Data:
  Ordered.MatchingSamples(~MatrixSpec.Reduction,~Nat,~Nat,~Mapped(),~Nat,~MatrixSpec.Pixel,~ConvSpec.Problem,
    ~mapped_weight(),~mapped_input(),~MatrixSpec.patch,~MatrixSpec.weight,
    indices,row,lane,mapped_source(problem,Inputs.weight_cells(problem,storage_contract),tiles,group,block),channel,MatrixSpec.Position{output_row,output_column},problem)

def agreements(~Evidence: LeafContract.Evidence(),~samples: LeafContract.Sampling(~Evidence),+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +tiles: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,+group: Nat,+block: Nat,+row: Nat,+lane: Nat,
  +channel: Nat,+output_row: Nat,+output_column: Nat,+indices: List<&2,MatrixSpec.Reduction>,
  +certificate: Evidence(problem,offset,span,tiles,output_row,output_column),
  +group_inside: {Nat.is_lt(group,PackingBounds.groups(span)) == True{} : Bool},
  +position: {PackingReadback.logical_position(offset,group,TileLaws.lane_of(lane)) == Coordinates.position(problem,output_row,output_column) : Nat},
  +row_inside: {Nat.is_lt(output_row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(output_column,ConvSpec.out_width(problem)) == True{} : Bool},
  +row_small: {Nat.is_lt(row,8n) == True{} : Bool},+channel_exact: {channel == Nat.add(Nat.mul(block,8n),row) : Nat},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  bounds: ReductionBounds.AllBounds(problem,indices)) ->
  Agreements(problem,storage_contract,tiles,group,block,row,lane,channel,output_row,output_column,indices):
  match indices:
    case Nil{}: Unit{}
    case +index <> +rest:
      Gemm.Both{+head,tail} = bounds
      Gemm.Both{
        Gemm.Both{
          input_agrees(~Evidence,~samples,limits,problem,storage_contract,tiles,offset,span,group,block,lane,output_row,output_column,index,valid,supported,certificate,group_inside,position,row_inside,column_inside,head),
          weight_agrees(limits,problem,storage_contract,valid,supported,tiles,group,block,row,channel,index,row_small,channel_exact,channel_inside,head)},
        agreements(~Evidence,~samples,limits,problem,storage_contract,valid,supported,tiles,offset,span,group,block,row,lane,channel,output_row,output_column,rest,
          certificate,group_inside,position,row_inside,column_inside,row_small,channel_exact,channel_inside,tail)}

# One job row and lane equal the frozen ordered matrix dot for their coordinate.
def job_dot(~Evidence: LeafContract.Evidence(),~samples: LeafContract.Sampling(~Evidence),+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +tiles: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,+group: Nat,+block: Nat,+row: Nat,+lane: Nat,
  +channel: Nat,+output_row: Nat,+output_column: Nat,
  +certificate: Evidence(problem,offset,span,tiles,output_row,output_column),
  +group_inside: {Nat.is_lt(group,PackingBounds.groups(span)) == True{} : Bool},
  +position: {PackingReadback.logical_position(offset,group,TileLaws.lane_of(lane)) == Coordinates.position(problem,output_row,output_column) : Nat},
  +row_inside: {Nat.is_lt(output_row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(output_column,ConvSpec.out_width(problem)) == True{} : Bool},
  +row_small: {Nat.is_lt(row,8n) == True{} : Bool},+channel_exact: {channel == Nat.add(Nat.mul(block,8n),row) : Nat},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool}) ->
  {GemmSpec.dot(GemmLaws.RowFirst{},GemmSpec.taps(~Gemm.Direct(),~Gemm.directs,Coordinates.inner(problem),Gemm.Both{U32.from_nat(Nat.mul(block,Coordinates.inner(problem))),U32.from_nat(Nat.mul(group,Coordinates.inner(problem)))}),
      lane,row,GemmSpec.Source{tiles,Inputs.weight_cells(problem,storage_contract)},0.0) ==
    Values.matrix_dot(MatrixSpec.kchannels(Guards.input_channel_count(problem),0n,Guards.kernel(problem)),problem,channel,output_row,output_column,0.0) : F32}:
  +channels = Guards.input_channel_count(problem)
  +kernel = Guards.kernel(problem)
  +input_base = Nat.mul(block,Coordinates.inner(problem))
  +weight_base = Nat.mul(group,Coordinates.inner(problem))
  +source = {GemmSpec.Source{tiles,Inputs.weight_cells(problem,storage_contract)} : GemmSpec.Source}
  Equal.trans(F32,
    GemmSpec.dot(GemmLaws.RowFirst{},GemmSpec.taps(~Gemm.Direct(),~Gemm.directs,Coordinates.inner(problem),Gemm.Both{U32.from_nat(input_base),U32.from_nat(weight_base)}),lane,row,source,0.0),
    matrix_keyed_dot(row,lane,channels,kernel,input_base,weight_base,source),
    Values.matrix_dot(MatrixSpec.kchannels(channels,0n,kernel),problem,channel,output_row,output_column,0.0),
    reindexed(row,lane,source,channels,kernel,input_base,weight_base),
    Ordered.dot_congruence(~MatrixSpec.Reduction,~Nat,~Nat,~Mapped(),~Nat,~MatrixSpec.Pixel,~ConvSpec.Problem,
      ~mapped_weight(),~mapped_input(),~MatrixSpec.patch,~MatrixSpec.weight,
      MatrixSpec.kchannels(channels,0n,kernel),row,lane,mapped_source(problem,Inputs.weight_cells(problem,storage_contract),tiles,group,block),channel,MatrixSpec.Position{output_row,output_column},problem,0.0,
      agreements(~Evidence,~samples,limits,problem,storage_contract,valid,supported,tiles,offset,span,group,block,row,lane,channel,output_row,output_column,
        MatrixSpec.kchannels(channels,0n,kernel),certificate,group_inside,position,row_inside,column_inside,row_small,channel_exact,channel_inside,
        ReductionBounds.matrix_reduction_in_bounds(problem))))
