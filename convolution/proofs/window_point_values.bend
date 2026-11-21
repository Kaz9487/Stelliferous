# The output of a pointwise call: lane by lane, the pair a job stores is the
# matrix value of its position, the input channels in the frozen order.
import Base
import ../specification.bend as ConvSpec
import ../matrix_specification.bend as MatrixSpec
import ../../lib/ordered_matrix.bend as OrderedMatrix
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/ordered_reduction_split.bend as Ordered
import ../../lib/proofs/kernel_gemm_point_spec.bend as PointSpec
import ../../lib/proofs/kernel_gemm_point_keys.bend as Keys
import ../../lib/proofs/kernel_gemm_point_readback.bend as PointReadback
import ../../lib/proofs/kernel_gemm_point_jobs.bend as PointJobs
import ./output_values.bend as Values
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/nat_ceiling.bend as Ceiling
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/machine_limits.bend as Machine
import ../inference.bend as NativeConv
import ./convolution_guard_facts.bend as Guards
import ./input_storage.bend as Inputs
import ./window_cell_contents.bend as Cells
import ./window_packed_values.bend as PackedValues
import ./gemm_dot_matches_matrix.bend as GemmMatrix
import ./reduction_coordinate_bounds.bend as Coordinates
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/nat_division.bend as Division
import ../../lib/proofs/nat_extrema.bend as NatExtrema
import ../../lib/proofs/tensor_tiled_values.bend as TiledValues
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ../../lib/proofs/kernel_gemm_point_buffer.bend as PointBuffer
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./native_weight_refinement.bend as NativeWeightRefinement

# ---- The frozen reduction of a pointwise problem by key ----

# Key lane of cell group cell is input channel lane + 8 cell of the one kernel
# tap.
def key_reduction(context: Unit,key: Keys.Key) -> MatrixSpec.Reduction:
  MatrixSpec.KIndex{Keys.key_channel(key),0n,0n}

def channels_keyed(count: Nat,+cell: Nat) ->
  {Ordered.map_keys(~Keys.Key,~MatrixSpec.Reduction,~Unit,~key_reduction,Keys.keys(count,cell),Unit{}) == MatrixSpec.kchannels(Nat.mul(count,16n),Nat.mul(cell,8n),1n) :
    List<&2,MatrixSpec.Reduction>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +prefix = Ordered.map_keys(~Keys.Key,~MatrixSpec.Reduction,~Unit,~key_reduction,OrderedMatrix.cat(Keys.Key,Keys.lane_keys(8n,0n,cell),Keys.lane_keys(8n,0n,1n+cell)),Unit{})
      Equal.cong(List<&2,MatrixSpec.Reduction>,List<&2,MatrixSpec.Reduction>,tail => OrderedMatrix.cat(MatrixSpec.Reduction,prefix,tail),
        Ordered.map_keys(~Keys.Key,~MatrixSpec.Reduction,~Unit,~key_reduction,Keys.keys(rest,1n+1n+cell),Unit{}),MatrixSpec.kchannels(Nat.mul(rest,16n),Nat.mul(1n+1n+cell,8n),1n),
        channels_keyed(rest,1n+1n+cell))

def spec_input() -> Gemm.Both<Unit,ConvSpec.Problem> -> MatrixSpec.Pixel -> Keys.Key -> F32:
  Ordered.mapped_sample(~Keys.Key,~MatrixSpec.Reduction,~MatrixSpec.Pixel,~ConvSpec.Problem,~Unit,~key_reduction,~MatrixSpec.patch)

def spec_weight() -> Gemm.Both<Unit,ConvSpec.Problem> -> Nat -> Keys.Key -> F32:
  Ordered.mapped_sample(~Keys.Key,~MatrixSpec.Reduction,~Nat,~ConvSpec.Problem,~Unit,~key_reduction,~MatrixSpec.weight)

def matrix_keyed(+groups: Nat,+problem: ConvSpec.Problem,+channel: Nat,+row: Nat,+column: Nat) ->
  {Values.matrix_dot(MatrixSpec.kchannels(Nat.mul(groups,16n),0n,1n),problem,channel,row,column,0.0) ==
    OrderedMatrix.dot(~Nat,~Keys.Key,~MatrixSpec.Pixel,~Gemm.Both<Unit,ConvSpec.Problem>,~spec_input(),~spec_weight(),Keys.keys(groups,0n),channel,MatrixSpec.Position{row,column},
      Gemm.Both{Unit{},problem},0.0) : F32}:
  %channels_keyed(groups,0n) :
    {Values.matrix_dot(_,problem,channel,row,column,0.0) ==
      OrderedMatrix.dot(~Nat,~Keys.Key,~MatrixSpec.Pixel,~Gemm.Both<Unit,ConvSpec.Problem>,~spec_input(),~spec_weight(),Keys.keys(groups,0n),channel,MatrixSpec.Position{row,column},
        Gemm.Both{Unit{},problem},0.0) : F32}
  Ordered.dot_reindexed(~Keys.Key,~MatrixSpec.Reduction,~Nat,~MatrixSpec.Pixel,~ConvSpec.Problem,~Unit,~key_reduction,~MatrixSpec.patch,~MatrixSpec.weight,
    Keys.keys(groups,0n),Unit{},channel,MatrixSpec.Position{row,column},problem,0.0)

# ---- A job's dot is the matrix dot ----

# Key by key, a job reads the input and the weight the matrix reads.
def Agreements(+groups: Nat,+address: Nat,+first: Nat,+cells: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+stride: U32,+row: Nat,+lane: Nat,
  +problem: ConvSpec.Problem,+channel: Nat,+output_row: Nat,+output_column: Nat) -> Data:
  Ordered.MatchingSamples(~Keys.Key,~Nat,~Nat,~Gemm.Both<Keys.Origin,PointSpec.Sampled>,~Nat,~MatrixSpec.Pixel,~Gemm.Both<Unit,ConvSpec.Problem>,
    ~Keys.key_input(),~Keys.key_weight(),~spec_input(),~spec_weight(),Keys.keys(groups,0n),lane,row,Gemm.Both{Keys.TapOrigin{address,first},PointSpec.Sampled{cells,pairs,stride}},
    channel,MatrixSpec.Position{output_row,output_column},Gemm.Both{Unit{},problem})

def keyed_matrix(+groups: Nat,+address: Nat,+first: Nat,+cells: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+stride: U32,+row: Nat,+lane: Nat,
  +problem: ConvSpec.Problem,+channel: Nat,+output_row: Nat,+output_column: Nat,
  +agreements: Agreements(groups,address,first,cells,pairs,stride,row,lane,problem,channel,output_row,output_column)) ->
  {OrderedMatrix.dot(~Nat,~Keys.Key,~Nat,~Gemm.Both<Keys.Origin,PointSpec.Sampled>,~Keys.key_input(),~Keys.key_weight(),Keys.keys(groups,0n),lane,row,
      Gemm.Both{Keys.TapOrigin{address,first},PointSpec.Sampled{cells,pairs,stride}},0.0) ==
    Values.matrix_dot(MatrixSpec.kchannels(Nat.mul(groups,16n),0n,1n),problem,channel,output_row,output_column,0.0) : F32}:
  +left = {Gemm.Both{Keys.TapOrigin{address,first},PointSpec.Sampled{cells,pairs,stride}} : Gemm.Both<Keys.Origin,PointSpec.Sampled>}
  +right = {Gemm.Both{Unit{},problem} : Gemm.Both<Unit,ConvSpec.Problem>}
  +pixel = {MatrixSpec.Position{output_row,output_column} : MatrixSpec.Pixel}
  +keyed = OrderedMatrix.dot(~Nat,~Keys.Key,~MatrixSpec.Pixel,~Gemm.Both<Unit,ConvSpec.Problem>,~spec_input(),~spec_weight(),Keys.keys(groups,0n),channel,pixel,right,0.0)
  Equal.trans(F32,OrderedMatrix.dot(~Nat,~Keys.Key,~Nat,~Gemm.Both<Keys.Origin,PointSpec.Sampled>,~Keys.key_input(),~Keys.key_weight(),Keys.keys(groups,0n),lane,row,left,0.0),
    keyed,Values.matrix_dot(MatrixSpec.kchannels(Nat.mul(groups,16n),0n,1n),problem,channel,output_row,output_column,0.0),
    Ordered.dot_congruence(~Keys.Key,~Nat,~Nat,~Gemm.Both<Keys.Origin,PointSpec.Sampled>,~Nat,~MatrixSpec.Pixel,~Gemm.Both<Unit,ConvSpec.Problem>,
      ~Keys.key_input(),~Keys.key_weight(),~spec_input(),~spec_weight(),Keys.keys(groups,0n),lane,row,left,channel,pixel,right,0.0,agreements),
    Equal.sym(F32,Values.matrix_dot(MatrixSpec.kchannels(Nat.mul(groups,16n),0n,1n),problem,channel,output_row,output_column,0.0),keyed,
      matrix_keyed(groups,problem,channel,output_row,output_column)))

# ---- The input a key reads ----

# Inside the image a sample at stride one without padding is the input value
# of its channel at its pixel.
def inside_sample(+height: Nat,+width: Nat,+input: List<&2,F32>,+channel: Nat,+row: Nat,+column: Nat,
  +row_inside: {Nat.is_lt(row,height) == True{} : Bool},+column_inside: {Nat.is_lt(column,width) == True{} : Bool}) ->
  {ConvSpec.sample_if(Nat.is_le(0n,row) && Nat.is_lt(row,Nat.add(0n,height)) && Nat.is_le(0n,column) && Nat.is_lt(column,Nat.add(0n,width)),
      Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(Nat.sub(row,0n),width),Nat.sub(column,0n))),input) ==
    ConvSpec.at(Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(row,width),column)),input) : F32}:
  -wanted = ConvSpec.at(Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(row,width),column)),input)
  %Equal.sym(Bool,Nat.is_le(0n,row),True{},Order.zero_le(row)) :
    {ConvSpec.sample_if(_ && Nat.is_lt(row,height) && Nat.is_le(0n,column) && Nat.is_lt(column,width),
      Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(Nat.sub(row,0n),width),Nat.sub(column,0n))),input) == wanted : F32}
  %Equal.sym(Bool,Nat.is_lt(row,height),True{},row_inside) :
    {ConvSpec.sample_if(True{} && _ && Nat.is_le(0n,column) && Nat.is_lt(column,width),
      Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(Nat.sub(row,0n),width),Nat.sub(column,0n))),input) == wanted : F32}
  %Equal.sym(Bool,Nat.is_le(0n,column),True{},Order.zero_le(column)) :
    {ConvSpec.sample_if(True{} && True{} && _ && Nat.is_lt(column,width),
      Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(Nat.sub(row,0n),width),Nat.sub(column,0n))),input) == wanted : F32}
  %Equal.sym(Bool,Nat.is_lt(column,width),True{},column_inside) :
    {ConvSpec.sample_if(True{} && True{} && True{} && _,
      Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(Nat.sub(row,0n),width),Nat.sub(column,0n))),input) == wanted : F32}
  %Equal.sym(Nat,Nat.sub(row,0n),row,Order.subtract_zero(row)) :
    {ConvSpec.at(Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(_,width),Nat.sub(column,0n))),input) == wanted : F32}
  %Equal.sym(Nat,Nat.sub(column,0n),column,Order.subtract_zero(column)) :
    {ConvSpec.at(Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(row,width),_)),input) == wanted : F32}
  {==}

# The row of an output position at stride one without padding.
def unit_step(+position: Nat) -> {Nat.add(Nat.mul(position,1n),0n) == position : Nat}:
  Equal.trans(Nat,Nat.add(Nat.mul(position,1n),0n),Nat.mul(position,1n),position,Addition.zero_right(Nat.mul(position,1n)),Order.times_one(position))

def point_sample(+ci: Nat,+co: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+pad: Nat,+input: List<&2,F32>,+weights: List<&2,F32>,+bias: List<&2,F32>,
  +channel: Nat,+output_row: Nat,+output_column: Nat,+stride_one: {stride == 1n : Nat},+pad_zero: {pad == 0n : Nat},
  +row_inside: {Nat.is_lt(output_row,height) == True{} : Bool},+column_inside: {Nat.is_lt(output_column,width) == True{} : Bool}) ->
  {ConvSpec.sample(ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,pad},input,weights,bias},channel,output_row,output_column,0n,0n) ==
    ConvSpec.at(Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(output_row,width),output_column)),input) : F32}:
  +row = Nat.add(Nat.mul(output_row,1n),0n)
  +column = Nat.add(Nat.mul(output_column,1n),0n)
  +row_same = unit_step(output_row)
  +column_same = unit_step(output_column)
  +wanted = ConvSpec.at(Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(output_row,width),output_column)),input)
  %Equal.sym(Nat,stride,1n,stride_one) :
    {ConvSpec.sample(ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,_,pad},input,weights,bias},channel,output_row,output_column,0n,0n) == wanted : F32}
  %Equal.sym(Nat,pad,0n,pad_zero) :
    {ConvSpec.sample(ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,1n,_},input,weights,bias},channel,output_row,output_column,0n,0n) == wanted : F32}
  Equal.trans(F32,ConvSpec.sample(ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,1n,0n},input,weights,bias},channel,output_row,output_column,0n,0n),
    ConvSpec.at(Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(row,width),column)),input),wanted,
    inside_sample(height,width,input,channel,row,column,
      %Equal.sym(Nat,row,output_row,row_same) : {Nat.is_lt(_,height) == True{} : Bool} row_inside,
      %Equal.sym(Nat,column,output_column,column_same) : {Nat.is_lt(_,width) == True{} : Bool} column_inside),
    %row_same : {ConvSpec.at(Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(row,width),column)),input) ==
      ConvSpec.at(Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(_,width),output_column)),input) : F32}
    %column_same : {ConvSpec.at(Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(row,width),column)),input) ==
      ConvSpec.at(Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(row,width),_)),input) : F32}
    {==})

def point_patch(+problem: ConvSpec.Problem,+channel: Nat,+output_row: Nat,+output_column: Nat,
  +stride_one: {Guards.stride(problem) == 1n : Nat},+pad_zero: {Guards.pad(problem) == 0n : Nat},
  +row_inside: {Nat.is_lt(output_row,Guards.height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(output_column,Guards.width(problem)) == True{} : Bool}) ->
  {MatrixSpec.patch(problem,MatrixSpec.Position{output_row,output_column},MatrixSpec.KIndex{channel,0n,0n}) ==
    ConvSpec.at(Nat.add(Nat.mul(channel,Nat.mul(Guards.height(problem),Guards.width(problem))),Nat.add(Nat.mul(output_row,Guards.width(problem)),output_column)),Guards.input(problem)) : F32}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+pad} = shape
  point_sample(ci,co,height,width,kernel,stride,pad,input,weights,bias,channel,output_row,output_column,stride_one,pad_zero,row_inside,column_inside)

# A lane of a band cell of an image's rows from row_first is the input value
# of its channel at its pixel.
def key_input_read(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+cells: Tree.Tree<Tile.Tile8>,+base: Nat,+row_first: Nat,+rows: Nat,
  +contents: Cells.Contents(problem,Cells.input_cells(problem,storage),cells,base,row_first,rows),
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +rows_fit: {Nat.is_le(Nat.add(row_first,rows),Guards.height(problem)) == True{} : Bool},
  +cell: Nat,+lane: Nat,+lane_small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +channel_inside: {Nat.is_lt(Nat.add(lane,Nat.mul(cell,8n)),Guards.input_channel_count(problem)) == True{} : Bool},
  +offset: Nat,+offset_inside: {Nat.is_lt(offset,Nat.mul(rows,Guards.width(problem))) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,cells,U32.from_nat(Nat.add(base,Nat.add(Nat.mul(offset,Cells.group_count(problem)),cell))))) ==
    ConvSpec.at(Nat.add(Nat.mul(Nat.add(lane,Nat.mul(cell,8n)),Nat.mul(Guards.height(problem),Guards.width(problem))),Nat.add(Nat.mul(row_first,Guards.width(problem)),offset)),
      Guards.input(problem)) : F32}:
  +ci = Guards.input_channel_count(problem)
  +width = Guards.width(problem)
  +area = Nat.mul(Guards.height(problem),width)
  +plane = Nat.mul(rows,width)
  +groups = Cells.group_count(problem)
  +channel = Nat.add(lane,Nat.mul(cell,8n))
  +index = Nat.add(Nat.mul(offset,groups),cell)
  +pixel = Nat.add(Nat.mul(row_first,width),offset)
  +root = Cells.input_cells(problem,storage)
  +wanted = ConvSpec.at(Nat.add(Nat.mul(channel,area),pixel),Guards.input(problem))
  +eight_le = {%Order.commutative(Nat.mul(cell,8n),lane) : {Nat.is_le(Nat.mul(cell,8n),_) == True{} : Bool}
    Order.le_add(Nat.mul(cell,8n),lane) : {Nat.is_le(Nat.mul(cell,8n),channel) == True{} : Bool}}
  +cell_inside = {%Equal.sym(Bool,Nat.is_lt(cell,groups),Nat.is_lt(Nat.mul(cell,8n),ci),Ceiling.upper_comparison(ci,8n,cell,{==})) : {_ == True{} : Bool}
    Order.le_lt_trans(Nat.mul(cell,8n),channel,ci,eight_le,channel_inside) : {Nat.is_lt(cell,groups) == True{} : Bool}}
  +pixel_inside = Bounds.nat_trans(pixel,Nat.add(Nat.mul(row_first,width),plane),area,Order.offset_lt(Nat.mul(row_first,width),offset,plane,offset_inside),
    %RowMajor.distrib(row_first,rows,width) : {Nat.is_le(_,area) == True{} : Bool} RowMajor.mul_le(Nat.add(row_first,rows),Guards.height(problem),width,rows_fit))
  +root_cell = Tree.read(Tile.Tile8,root,U32.from_nat(Nat.add(Nat.mul(pixel,groups),cell)))
  +cell_equal = Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,cells,U32.from_nat(Nat.add(base,index))),Tree.read(Tile.Tile8,root,U32.from_nat(Cells.source(problem,row_first,index))),root_cell,
    Cells.read(problem,root,cells,base,row_first,rows,contents,index,RowMajor.radix(offset,plane,cell,groups,offset_inside,cell_inside)),
    Equal.cong(Nat,Tile.Tile8,position => Tree.read(Tile.Tile8,root,U32.from_nat(position)),Cells.source(problem,row_first,index),Nat.add(Nat.mul(pixel,groups),cell),
      Cells.source_pixel(problem,row_first,offset,cell)))
  Equal.trans(F32,TileLaws.select_lane(lane,Tree.read(Tile.Tile8,cells,U32.from_nat(Nat.add(base,index)))),TileLaws.select_lane(lane,root_cell),wanted,
    Equal.cong(Tile.Tile8,F32,tile => TileLaws.select_lane(lane,tile),Tree.read(Tile.Tile8,cells,U32.from_nat(Nat.add(base,index))),root_cell,cell_equal),
    %PackedValues.lane_place(lane,cell,lane_small) :
      {TileLaws.select_lane(_,root_cell) == wanted : F32}
    %PackedValues.lane_group(lane,cell,lane_small) :
      {TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,root,U32.from_nat(Nat.add(Nat.mul(pixel,groups),_)))) == wanted : F32}
    Cells.input_lane(limits,problem,storage,channel,pixel,valid,supported,channel_inside,pixel_inside))

# Where the row of a job reads in the band: past its first cell, the cell of
# its position.
def band_index(+cell: Nat,+first: Nat,+step: Nat,+base: Nat,+cells: Nat,+first_exact: {Nat.add(first,step) == Nat.add(base,cells) : Nat}) ->
  {Nat.add(Nat.add(cell,first),step) == Nat.add(base,Nat.add(cells,cell)) : Nat}:
  Equal.trans(Nat,Nat.add(Nat.add(cell,first),step),Nat.add(cell,Nat.add(first,step)),Nat.add(base,Nat.add(cells,cell)),Addition.associative(cell,first,step),
    Equal.trans(Nat,Nat.add(cell,Nat.add(first,step)),Nat.add(cell,Nat.add(base,cells)),Nat.add(base,Nat.add(cells,cell)),
      Equal.cong(Nat,Nat,sum => Nat.add(cell,sum),Nat.add(first,step),Nat.add(base,cells),first_exact),
      Equal.trans(Nat,Nat.add(cell,Nat.add(base,cells)),Nat.add(base,Nat.add(cell,cells)),Nat.add(base,Nat.add(cells,cell)),RowMajor.add_swap(cell,base,cells),
        Equal.cong(Nat,Nat,sum => Nat.add(base,sum),Nat.add(cell,cells),Nat.add(cells,cell),Order.commutative(cell,cells)))))

# A key of a job reads the input the matrix reads at the position of its row.
def key_input_agrees(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+cells: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,
  +stride: U32,+address: Nat,+first: Nat,+row: Nat,+row_small: {Nat.is_le(row,3n) == True{} : Bool},+base: Nat,+row_first: Nat,+rows: Nat,
  +contents: Cells.Contents(problem,Cells.input_cells(problem,storage),cells,base,row_first,rows),
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +rows_fit: {Nat.is_le(Nat.add(row_first,rows),Guards.height(problem)) == True{} : Bool},
  +offset: Nat,+offset_inside: {Nat.is_lt(offset,Nat.mul(rows,Guards.width(problem))) == True{} : Bool},
  +first_exact: {Nat.add(first,Nat.mul(row,U32.to_nat(stride))) == Nat.add(base,Nat.mul(offset,Cells.group_count(problem))) : Nat},
  +output_row: Nat,+output_column: Nat,
  +pixel_exact: {Nat.add(Nat.mul(row_first,Guards.width(problem)),offset) == Nat.add(Nat.mul(output_row,Guards.width(problem)),output_column) : Nat},
  +row_inside: {Nat.is_lt(output_row,Guards.height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(output_column,Guards.width(problem)) == True{} : Bool},
  +stride_one: {Guards.stride(problem) == 1n : Nat},+pad_zero: {Guards.pad(problem) == 0n : Nat},
  +cell: Nat,+lane: Nat,+lane_small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +channel_inside: {Nat.is_lt(Nat.add(lane,Nat.mul(cell,8n)),Guards.input_channel_count(problem)) == True{} : Bool}) ->
  {Keys.key_input()(Gemm.Both{Keys.TapOrigin{address,first},PointSpec.Sampled{cells,pairs,stride}},row,Keys.LaneKey{cell,lane}) ==
    spec_input()(Gemm.Both{Unit{},problem},MatrixSpec.Position{output_row,output_column},Keys.LaneKey{cell,lane}) : F32}:
  +channel = Nat.add(lane,Nat.mul(cell,8n))
  +width = Guards.width(problem)
  +area = Nat.mul(Guards.height(problem),width)
  +step = Nat.mul(row,U32.to_nat(stride))
  +cells_at = Nat.mul(offset,Cells.group_count(problem))
  +read = TileLaws.select_lane(lane,Tree.read(Tile.Tile8,cells,U32.from_nat(Nat.add(base,Nat.add(cells_at,cell)))))
  +patch = MatrixSpec.patch(problem,MatrixSpec.Position{output_row,output_column},MatrixSpec.KIndex{channel,0n,0n})
  Equal.trans(F32,Keys.key_input()(Gemm.Both{Keys.TapOrigin{address,first},PointSpec.Sampled{cells,pairs,stride}},row,Keys.LaneKey{cell,lane}),read,patch,
    %band_index(cell,first,step,base,cells_at,first_exact) :
      {Keys.key_input()(Gemm.Both{Keys.TapOrigin{address,first},PointSpec.Sampled{cells,pairs,stride}},row,Keys.LaneKey{cell,lane}) ==
        TileLaws.select_lane(lane,Tree.read(Tile.Tile8,cells,U32.from_nat(_))) : F32}
    Keys.key_input_value(address,first,cells,pairs,stride,row,cell,lane,row_small),
    Equal.trans(F32,read,ConvSpec.at(Nat.add(Nat.mul(channel,area),Nat.add(Nat.mul(row_first,width),offset)),Guards.input(problem)),patch,
      key_input_read(limits,problem,storage,cells,base,row_first,rows,contents,valid,supported,rows_fit,cell,lane,lane_small,channel_inside,offset,offset_inside),
      %Equal.sym(Nat,Nat.add(Nat.mul(row_first,width),offset),Nat.add(Nat.mul(output_row,width),output_column),pixel_exact) :
        {ConvSpec.at(Nat.add(Nat.mul(channel,area),_),Guards.input(problem)) == patch : F32}
      Equal.sym(F32,patch,ConvSpec.at(Nat.add(Nat.mul(channel,area),Nat.add(Nat.mul(output_row,width),output_column)),Guards.input(problem)),
        point_patch(problem,channel,output_row,output_column,stride_one,pad_zero,row_inside,column_inside))))

# ---- The weight a key reads ----

# The one tap of a kernel of one is its input channel.
def tap_one(+channel: Nat) -> {Coordinates.reduction(channel,0n,0n,1n) == channel : Nat}:
  Equal.trans(Nat,Coordinates.reduction(channel,0n,0n,1n),Nat.add(Nat.mul(channel,1n),0n),channel,unit_step(Nat.add(Nat.mul(channel,1n),0n)),unit_step(channel))

# Where a key reads in the pairs: its channel past the pair's weights.
def pair_index(+lane: Nat,+cell: Nat,+address: Nat) -> {Nat.add(lane,Nat.add(Nat.mul(cell,8n),address)) == Nat.add(address,Nat.add(lane,Nat.mul(cell,8n))) : Nat}:
  Equal.trans(Nat,Nat.add(lane,Nat.add(Nat.mul(cell,8n),address)),Nat.add(Nat.add(lane,Nat.mul(cell,8n)),address),Nat.add(address,Nat.add(lane,Nat.mul(cell,8n))),
    Equal.sym(Nat,Nat.add(Nat.add(lane,Nat.mul(cell,8n)),address),Nat.add(lane,Nat.add(Nat.mul(cell,8n),address)),Addition.associative(lane,Nat.mul(cell,8n),address)),
    Order.commutative(Nat.add(lane,Nat.mul(cell,8n)),address))

# A key of a job reads the weight the matrix reads for the channel of its
# lane, when lane `lane` of the pair at a channel holds that channel's lane
# of its weight cell.
def key_weight_agrees(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +kernel_one: {Guards.kernel(problem) == 1n : Nat},+cells: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+stride: U32,+address: Nat,+first: Nat,
  +lane: Nat,+channel: Nat,+channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +cell: Nat,+key_lane: Nat,+input_inside: {Nat.is_lt(Nat.add(key_lane,Nat.mul(cell,8n)),Guards.input_channel_count(problem)) == True{} : Bool},
  +pair_read: {PointSpec.pair_lane(lane,Tree.read(Tile.Tile16,pairs,U32.from_nat(Nat.add(address,Nat.add(key_lane,Nat.mul(cell,8n)))))) ==
    TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,Inputs.weight_cells(problem,storage),
      U32.from_nat(Nat.add(Nat.mul(Nat.div(channel,8n),Coordinates.inner(problem)),Nat.add(key_lane,Nat.mul(cell,8n)))))) : F32}) ->
  {Keys.key_weight()(Gemm.Both{Keys.TapOrigin{address,first},PointSpec.Sampled{cells,pairs,stride}},lane,Keys.LaneKey{cell,key_lane}) ==
    spec_weight()(Gemm.Both{Unit{},problem},channel,Keys.LaneKey{cell,key_lane}) : F32}:
  +input = Nat.add(key_lane,Nat.mul(cell,8n))
  +block = Nat.div(channel,8n)
  +row = Nat.mod(channel,8n)
  +inner = Coordinates.inner(problem)
  +weights = Inputs.weight_cells(problem,storage)
  +positive = {{==} : {Nat.is_lt(0n,8n) == True{} : Bool}}
  +kernel_positive = {%Equal.sym(Nat,Guards.kernel(problem),1n,kernel_one) : {Nat.is_lt(0n,_) == True{} : Bool} {==} : {Nat.is_lt(0n,Guards.kernel(problem)) == True{} : Bool}}
  +agreed = {GemmMatrix.weight_agrees(limits,problem,storage,valid,supported,weights,0n,block,row,channel,MatrixSpec.KIndex{input,0n,0n},
      DivisionOrder.remainder_small(channel,8n,positive),DivisionOrder.quotient_exact(channel,8n,positive),channel_inside,
      Gemm.Both{input_inside,Gemm.Both{kernel_positive,kernel_positive}}) :
    {TileLaws.select_lane(row,Tree.read(Tile.Tile8,weights,U32.from_nat(Nat.add(Nat.mul(block,inner),Coordinates.reduction(input,0n,0n,Guards.kernel(problem)))))) ==
      MatrixSpec.weight(problem,channel,MatrixSpec.KIndex{input,0n,0n}) : F32}}
  %Equal.sym(Nat,Nat.add(key_lane,Nat.add(Nat.mul(cell,8n),address)),Nat.add(address,input),pair_index(key_lane,cell,address)) :
    {PointSpec.pair_lane(lane,Tree.read(Tile.Tile16,pairs,U32.from_nat(_))) == MatrixSpec.weight(problem,channel,MatrixSpec.KIndex{input,0n,0n}) : F32}
  %Equal.sym(F32,PointSpec.pair_lane(lane,Tree.read(Tile.Tile16,pairs,U32.from_nat(Nat.add(address,input)))),
    TileLaws.select_lane(row,Tree.read(Tile.Tile8,weights,U32.from_nat(Nat.add(Nat.mul(block,inner),input)))),pair_read) :
    {_ == MatrixSpec.weight(problem,channel,MatrixSpec.KIndex{input,0n,0n}) : F32}
  %tap_one(input) :
    {TileLaws.select_lane(row,Tree.read(Tile.Tile8,weights,U32.from_nat(Nat.add(Nat.mul(block,inner),_)))) == MatrixSpec.weight(problem,channel,MatrixSpec.KIndex{input,0n,0n}) : F32}
  %kernel_one :
    {TileLaws.select_lane(row,Tree.read(Tile.Tile8,weights,U32.from_nat(Nat.add(Nat.mul(block,inner),Coordinates.reduction(input,0n,0n,_))))) ==
      MatrixSpec.weight(problem,channel,MatrixSpec.KIndex{input,0n,0n}) : F32}
  agreed

# ---- Every key of a job ----

# Lane `lane` of the pair at each input channel holds that channel's lane of
# the weight cell of channel `channel`.
def PairRead(+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+pairs: Tree.Tree<Tile.Tile16>,+address: Nat,+lane: Nat,+channel: Nat,+input: Nat) -> Data:
  {PointSpec.pair_lane(lane,Tree.read(Tile.Tile16,pairs,U32.from_nat(Nat.add(address,input)))) ==
    TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,Inputs.weight_cells(problem,storage),
      U32.from_nat(Nat.add(Nat.mul(Nat.div(channel,8n),Coordinates.inner(problem)),input)))) : F32}

def PairsRead(+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+pairs: Tree.Tree<Tile.Tile16>,+address: Nat,+lane: Nat,+channel: Nat) -> Data:
  Members.All(Nat,{input => PairRead(problem,storage,pairs,address,lane,channel,input) : Nat -> Data},Ranges.range(Guards.input_channel_count(problem),0n))

# What every key of the job at a row and a lane reads: the band cells of the
# row's position and the pairs of the lane's channel.
type KeyFacts is Data:
  KeyFacts{limits: Machine.Limits,problem: ConvSpec.Problem,storage: Inputs.Contract<problem>,valid: {ConvSpec.valid(problem) == True{} : Bool},supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},kernel_one: {Guards.kernel(problem) == 1n : Nat},stride_one: {Guards.stride(problem) == 1n : Nat},pad_zero: {Guards.pad(problem) == 0n : Nat},groups: Nat,channels_exact: {Guards.input_channel_count(problem) == Nat.mul(groups,16n) : Nat},cells: Tree.Tree<Tile.Tile8>,pairs: Tree.Tree<Tile.Tile16>,stride: U32,address: Nat,first: Nat,row: Nat,row_small: {Nat.is_le(row,3n) == True{} : Bool},base: Nat,row_first: Nat,rows: Nat,contents: Cells.Contents(problem,Cells.input_cells(problem,storage),cells,base,row_first,rows),rows_fit: {Nat.is_le(Nat.add(row_first,rows),Guards.height(problem)) == True{} : Bool},offset: Nat,offset_inside: {Nat.is_lt(offset,Nat.mul(rows,Guards.width(problem))) == True{} : Bool},first_exact: {Nat.add(first,Nat.mul(row,U32.to_nat(stride))) == Nat.add(base,Nat.mul(offset,Cells.group_count(problem))) : Nat},output_row: Nat,output_column: Nat,pixel_exact: {Nat.add(Nat.mul(row_first,Guards.width(problem)),offset) == Nat.add(Nat.mul(output_row,Guards.width(problem)),output_column) : Nat},row_inside: {Nat.is_lt(output_row,Guards.height(problem)) == True{} : Bool},column_inside: {Nat.is_lt(output_column,Guards.width(problem)) == True{} : Bool},lane: Nat,channel: Nat,channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},pairs_read: PairsRead(problem,storage,pairs,address,lane,channel)}

def key_cells(context: KeyFacts) -> Nat:
  KeyFacts{limits,problem,storage,valid,supported,kernel_one,stride_one,pad_zero,groups,channels_exact,cells,pairs,stride,address,first,row,row_small,base,row_first,rows,contents,rows_fit,offset,offset_inside,first_exact,output_row,output_column,pixel_exact,row_inside,column_inside,lane,channel,channel_inside,pairs_read} = context
  Nat.mul(groups,2n)

def KeyAgrees(+context: KeyFacts,+key: Keys.Key) -> Data:
  KeyFacts{+limits,+problem,+storage,+valid,+supported,+kernel_one,+stride_one,+pad_zero,+groups,+channels_exact,+cells,+pairs,+stride,+address,+first,+row,+row_small,+base,+row_first,+rows,+contents,+rows_fit,+offset,+offset_inside,+first_exact,+output_row,+output_column,+pixel_exact,+row_inside,+column_inside,+lane,+channel,+channel_inside,+pairs_read} = context
  Gemm.Both<{Keys.key_input()(Gemm.Both{Keys.TapOrigin{address,first},PointSpec.Sampled{cells,pairs,stride}},row,key) == spec_input()(Gemm.Both{Unit{},problem},MatrixSpec.Position{output_row,output_column},key) : F32},
    {Keys.key_weight()(Gemm.Both{Keys.TapOrigin{address,first},PointSpec.Sampled{cells,pairs,stride}},lane,key) == spec_weight()(Gemm.Both{Unit{},problem},channel,key) : F32}>

# The input channel of a key of groups of sixteen is one of them.
def key_inside(+cell: Nat,+lane: Nat,+groups: Nat,+cell_inside: {Nat.is_lt(cell,Nat.mul(groups,2n)) == True{} : Bool},+lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(lane,Nat.mul(cell,8n)),Nat.mul(groups,16n)) == True{} : Bool}:
  %RowMajor.assoc(groups,2n,8n) : {Nat.is_lt(Nat.add(lane,Nat.mul(cell,8n)),_) == True{} : Bool}
  %Order.commutative(Nat.mul(cell,8n),lane) : {Nat.is_lt(_,Nat.mul(Nat.mul(groups,2n),8n)) == True{} : Bool}
  RowMajor.radix(cell,Nat.mul(groups,2n),lane,8n,cell_inside,lane_inside)

def key_agrees(+context: KeyFacts,+cell: Nat,+key_lane: Nat,+cell_inside: {Nat.is_lt(cell,key_cells(context)) == True{} : Bool},+lane_inside: {Nat.is_lt(key_lane,8n) == True{} : Bool}) ->
  KeyAgrees(context,Keys.LaneKey{cell,key_lane}):
  KeyFacts{+limits,+problem,+storage,+valid,+supported,+kernel_one,+stride_one,+pad_zero,+groups,+channels_exact,+cells,+pairs,+stride,+address,+first,+row,+row_small,+base,+row_first,+rows,+contents,+rows_fit,+offset,+offset_inside,+first_exact,+output_row,+output_column,+pixel_exact,+row_inside,+column_inside,+lane,+channel,+channel_inside,+pairs_read} = context
  +input = Nat.add(key_lane,Nat.mul(cell,8n))
  +input_inside = {%Equal.sym(Nat,Guards.input_channel_count(problem),Nat.mul(groups,16n),channels_exact) : {Nat.is_lt(input,_) == True{} : Bool}
    key_inside(cell,key_lane,groups,cell_inside,lane_inside) : {Nat.is_lt(input,Guards.input_channel_count(problem)) == True{} : Bool}}
  Gemm.Both{key_input_agrees(limits,problem,storage,cells,pairs,stride,address,first,row,row_small,base,row_first,rows,contents,valid,supported,rows_fit,offset,offset_inside,
      first_exact,output_row,output_column,pixel_exact,row_inside,column_inside,stride_one,pad_zero,cell,key_lane,lane_inside,input_inside),
    key_weight_agrees(limits,problem,storage,valid,supported,kernel_one,cells,pairs,stride,address,first,lane,channel,channel_inside,cell,key_lane,input_inside,
      Members.all_member(Nat,{value => PairRead(problem,storage,pairs,address,lane,channel,value) : Nat -> Data},input,Ranges.range(Guards.input_channel_count(problem),0n),pairs_read,
        Members.range_member(Guards.input_channel_count(problem),0n,input,input_inside)))}

# Every key of the job at a row and a lane reads what the matrix reads.
def job_agreements(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+kernel_one: {Guards.kernel(problem) == 1n : Nat},+stride_one: {Guards.stride(problem) == 1n : Nat},+pad_zero: {Guards.pad(problem) == 0n : Nat},+groups: Nat,+channels_exact: {Guards.input_channel_count(problem) == Nat.mul(groups,16n) : Nat},+cells: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+stride: U32,+address: Nat,+first: Nat,+row: Nat,+row_small: {Nat.is_le(row,3n) == True{} : Bool},+base: Nat,+row_first: Nat,+rows: Nat,+contents: Cells.Contents(problem,Cells.input_cells(problem,storage),cells,base,row_first,rows),+rows_fit: {Nat.is_le(Nat.add(row_first,rows),Guards.height(problem)) == True{} : Bool},+offset: Nat,+offset_inside: {Nat.is_lt(offset,Nat.mul(rows,Guards.width(problem))) == True{} : Bool},+first_exact: {Nat.add(first,Nat.mul(row,U32.to_nat(stride))) == Nat.add(base,Nat.mul(offset,Cells.group_count(problem))) : Nat},+output_row: Nat,+output_column: Nat,+pixel_exact: {Nat.add(Nat.mul(row_first,Guards.width(problem)),offset) == Nat.add(Nat.mul(output_row,Guards.width(problem)),output_column) : Nat},+row_inside: {Nat.is_lt(output_row,Guards.height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(output_column,Guards.width(problem)) == True{} : Bool},+lane: Nat,+channel: Nat,+channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+pairs_read: PairsRead(problem,storage,pairs,address,lane,channel)) ->
  Agreements(groups,address,first,cells,pairs,stride,row,lane,problem,channel,output_row,output_column):
  Keys.keys_all(~KeyFacts,~KeyAgrees,~key_cells,~key_agrees,groups,0n,KeyFacts{limits,problem,storage,valid,supported,kernel_one,stride_one,pad_zero,groups,channels_exact,cells,pairs,stride,address,first,row,row_small,base,row_first,rows,contents,rows_fit,offset,offset_inside,first_exact,output_row,output_column,pixel_exact,row_inside,column_inside,lane,channel,channel_inside,pairs_read},Order.reflexive(Nat.mul(groups,2n)))

# ---- The lanes of a pair ----

# A group below a count of groups is at most the last.
def last_group(+group: Nat,+groups: Nat,+inside: {Nat.is_lt(group,groups) == True{} : Bool}) -> {Nat.is_le(group,Nat.sub(groups,1n)) == True{} : Bool}:
  match groups:
    case 0n: Empty.absurd({Nat.is_le(group,Nat.sub(0n,1n)) == True{} : Bool},Order.below_zero(group,inside))
    case 1n+ +rest:
      %Equal.sym(Nat,Nat.sub(rest,0n),rest,Order.subtract_zero(rest)) : {Nat.is_le(group,_) == True{} : Bool}
      Division.lt_succ_le(group,rest,inside)

# Sixteen channels a pair: channel lane + 16 pair.
def pair_channel(+lane: Nat,+pair: Nat) -> Nat:
  Nat.add(lane,Nat.mul(pair,16n))

def pair_groups(+pair: Nat) -> {Nat.mul(Nat.mul(pair,2n),8n) == Nat.mul(pair,16n) : Nat}:
  RowMajor.assoc(pair,2n,8n)

# The lower group of a pair holds its first eight lanes.
def low_lane(~Context: Data,~cell: Context -> Nat -> Tile.Tile8,+context: Context,+pair: Nat,+lane: Nat,+small: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,cell(context,Nat.mul(pair,2n))) ==
    TileLaws.select_lane(Nat.mod(pair_channel(lane,pair),8n),cell(context,Nat.div(pair_channel(lane,pair),8n))) : F32}:
  +group = Nat.mul(pair,2n)
  %pair_groups(pair) :
    {TileLaws.select_lane(lane,cell(context,group)) == TileLaws.select_lane(Nat.mod(Nat.add(lane,_),8n),cell(context,Nat.div(Nat.add(lane,_),8n))) : F32}
  %Equal.sym(Nat,Nat.mod(Nat.add(lane,Nat.mul(group,8n)),8n),lane,PackedValues.lane_place(lane,group,small)) :
    {TileLaws.select_lane(lane,cell(context,group)) == TileLaws.select_lane(_,cell(context,Nat.div(Nat.add(lane,Nat.mul(group,8n)),8n))) : F32}
  %Equal.sym(Nat,Nat.div(Nat.add(lane,Nat.mul(group,8n)),8n),group,PackedValues.lane_group(lane,group,small)) :
    {TileLaws.select_lane(lane,cell(context,group)) == TileLaws.select_lane(lane,cell(context,_)) : F32}
  {==}

# The channel of lane 8 + rest of a pair is lane rest of its upper group.
def high_channel(+rest: Nat,+pair: Nat) -> {Nat.add(rest,Nat.mul(1n+Nat.mul(pair,2n),8n)) == pair_channel(8n+rest,pair) : Nat}:
  %pair_groups(pair) : {Nat.add(rest,Nat.mul(1n+Nat.mul(pair,2n),8n)) == Nat.add(8n,Nat.add(rest,_)) : Nat}
  RowMajor.add_swap(rest,8n,Nat.mul(Nat.mul(pair,2n),8n))

# The upper group of a pair holds its last eight lanes.
def high_select(~Context: Data,~cell: Context -> Nat -> Tile.Tile8,+context: Context,+pair: Nat,+rest: Nat,+outputs: Nat,+small: {Nat.is_lt(rest,8n) == True{} : Bool},
  +inside: {Nat.is_lt(pair_channel(8n+rest,pair),outputs) == True{} : Bool}) ->
  {TileLaws.select_lane(Nat.sub(rest,0n),cell(context,Nat.min(1n+Nat.mul(pair,2n),Nat.sub(TiledValues.groups_of(outputs),1n)))) ==
    TileLaws.select_lane(Nat.mod(pair_channel(8n+rest,pair),8n),cell(context,Nat.div(pair_channel(8n+rest,pair),8n))) : F32}:
  +group = {1n+Nat.mul(pair,2n) : Nat}
  +start = Nat.mul(group,8n)
  +channel = Nat.add(rest,start)
  +before = {%Order.commutative(start,rest) : {Nat.is_le(start,_) == True{} : Bool} Order.le_add(start,rest) : {Nat.is_le(start,channel) == True{} : Bool}}
  +below = {%Equal.sym(Nat,channel,pair_channel(8n+rest,pair),high_channel(rest,pair)) : {Nat.is_lt(_,outputs) == True{} : Bool} inside : {Nat.is_lt(channel,outputs) == True{} : Bool}}
  +group_inside = {%Equal.sym(Bool,Nat.is_lt(group,TiledValues.groups_of(outputs)),Nat.is_lt(start,outputs),Ceiling.upper_comparison(outputs,8n,group,{==})) : {_ == True{} : Bool}
    Order.le_lt_trans(start,channel,outputs,before,below) : {Nat.is_lt(group,TiledValues.groups_of(outputs)) == True{} : Bool}}
  %Equal.sym(Nat,Nat.sub(rest,0n),rest,Order.subtract_zero(rest)) :
    {TileLaws.select_lane(_,cell(context,Nat.min(group,Nat.sub(TiledValues.groups_of(outputs),1n)))) ==
      TileLaws.select_lane(Nat.mod(pair_channel(8n+rest,pair),8n),cell(context,Nat.div(pair_channel(8n+rest,pair),8n))) : F32}
  %Equal.sym(Nat,Nat.min(group,Nat.sub(TiledValues.groups_of(outputs),1n)),group,
    NatExtrema.minimum_when_ordered(group,Nat.sub(TiledValues.groups_of(outputs),1n),last_group(group,TiledValues.groups_of(outputs),group_inside))) :
    {TileLaws.select_lane(rest,cell(context,_)) ==
      TileLaws.select_lane(Nat.mod(pair_channel(8n+rest,pair),8n),cell(context,Nat.div(pair_channel(8n+rest,pair),8n))) : F32}
  %high_channel(rest,pair) :
    {TileLaws.select_lane(rest,cell(context,group)) == TileLaws.select_lane(Nat.mod(_,8n),cell(context,Nat.div(_,8n))) : F32}
  %Equal.sym(Nat,Nat.mod(Nat.add(rest,Nat.mul(group,8n)),8n),rest,PackedValues.lane_place(rest,group,small)) :
    {TileLaws.select_lane(rest,cell(context,group)) == TileLaws.select_lane(_,cell(context,Nat.div(channel,8n))) : F32}
  %Equal.sym(Nat,Nat.div(Nat.add(rest,Nat.mul(group,8n)),8n),group,PackedValues.lane_group(rest,group,small)) :
    {TileLaws.select_lane(rest,cell(context,group)) == TileLaws.select_lane(rest,cell(context,_)) : F32}
  {==}

def high_lane(~Context: Data,~cell: Context -> Nat -> Tile.Tile8,+context: Context,+pair: Nat,+rest: Nat,+outputs: Nat,+small: {Nat.is_lt(rest,8n) == True{} : Bool},
  +inside: {Nat.is_lt(pair_channel(8n+rest,pair),outputs) == True{} : Bool}) ->
  {PointSpec.pair_lane(8n+rest,Tile.Tiles{cell(context,Nat.mul(pair,2n)),cell(context,Nat.min(1n+Nat.mul(pair,2n),Nat.sub(TiledValues.groups_of(outputs),1n)))}) == TileLaws.select_lane(Nat.mod(pair_channel(8n+rest,pair),8n),cell(context,Nat.div(pair_channel(8n+rest,pair),8n))) : F32}:
  %Equal.sym(Bool,Nat.is_lt(rest,0n),False{},Division.lt_zero(rest)) :
    {PointSpec.half_lane(_,8n+rest,cell(context,Nat.mul(pair,2n)),cell(context,Nat.min(1n+Nat.mul(pair,2n),Nat.sub(TiledValues.groups_of(outputs),1n)))) == TileLaws.select_lane(Nat.mod(pair_channel(8n+rest,pair),8n),cell(context,Nat.div(pair_channel(8n+rest,pair),8n))) : F32}
  high_select(~Context,~cell,context,pair,rest,outputs,small,inside)

# Lane lane of the pair of the groups 2 pair and 2 pair + 1, or the last group
# for a missing upper one, is the lane of its channel in that channel's group.
def pair_lane_group(~Context: Data,~cell: Context -> Nat -> Tile.Tile8,+context: Context,+pair: Nat,+lane: Nat,+outputs: Nat,
  +small: {Nat.is_lt(lane,16n) == True{} : Bool},+inside: {Nat.is_lt(pair_channel(lane,pair),outputs) == True{} : Bool}) ->
  {PointSpec.pair_lane(lane,Tile.Tiles{cell(context,Nat.mul(pair,2n)),cell(context,Nat.min(1n+Nat.mul(pair,2n),Nat.sub(TiledValues.groups_of(outputs),1n)))}) ==
    TileLaws.select_lane(Nat.mod(pair_channel(lane,pair),8n),cell(context,Nat.div(pair_channel(lane,pair),8n))) : F32}:
  match lane:
    case 0n: low_lane(~Context,~cell,context,pair,0n,{==})
    case 1n+ +a:
      match a:
        case 0n: low_lane(~Context,~cell,context,pair,1n,{==})
        case 1n+ +b:
          match b:
            case 0n: low_lane(~Context,~cell,context,pair,2n,{==})
            case 1n+ +c:
              match c:
                case 0n: low_lane(~Context,~cell,context,pair,3n,{==})
                case 1n+ +d:
                  match d:
                    case 0n: low_lane(~Context,~cell,context,pair,4n,{==})
                    case 1n+ +e:
                      match e:
                        case 0n: low_lane(~Context,~cell,context,pair,5n,{==})
                        case 1n+ +f:
                          match f:
                            case 0n: low_lane(~Context,~cell,context,pair,6n,{==})
                            case 1n+ +g:
                              match g:
                                case 0n: low_lane(~Context,~cell,context,pair,7n,{==})
                                case 1n+ +rest: high_lane(~Context,~cell,context,pair,rest,outputs,small,inside)

# ---- The value a job stores ----

# Lane lane of what the job of a pair and a unit stores at a row is the
# matrix value of its output channel at the position of the row.
def spot_value(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +kernel_one: {Guards.kernel(problem) == 1n : Nat},+stride_one: {Guards.stride(problem) == 1n : Nat},+pad_zero: {Guards.pad(problem) == 0n : Nat},
  +channels: U32,+positions: U32,+inner: U32,+start: U32,+output_positions: U32,+cell: U32,+stride: U32,
  +bias: Tree.Tree<Tile.Tile8>,+cells: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+pair: Nat,+unit: Nat,+row: Nat,+row_small: {Nat.is_le(row,3n) == True{} : Bool},
  +channels_exact: {Guards.input_channel_count(problem) == Nat.mul(U32.to_nat((inner / 16 : U32)),16n) : Nat},
  +base: Nat,+row_first: Nat,+rows: Nat,+contents: Cells.Contents(problem,Cells.input_cells(problem,storage),cells,base,row_first,rows),
  +rows_fit: {Nat.is_le(Nat.add(row_first,rows),Guards.height(problem)) == True{} : Bool},
  +offset: Nat,+offset_inside: {Nat.is_lt(offset,Nat.mul(rows,Guards.width(problem))) == True{} : Bool},
  +first: Nat,+first_same: {U32.from_nat(Keys.job_first(cell,unit,stride)) == U32.from_nat(first) : U32},
  +first_exact: {Nat.add(first,Nat.mul(row,U32.to_nat(stride))) == Nat.add(base,Nat.mul(offset,Cells.group_count(problem))) : Nat},
  +output_row: Nat,+output_column: Nat,
  +pixel_exact: {Nat.add(Nat.mul(row_first,Guards.width(problem)),offset) == Nat.add(Nat.mul(output_row,Guards.width(problem)),output_column) : Nat},
  +row_inside: {Nat.is_lt(output_row,Guards.height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(output_column,Guards.width(problem)) == True{} : Bool},
  +lane: Nat,+channel_inside: {Nat.is_lt(pair_channel(lane,pair),Guards.output_channel_count(problem)) == True{} : Bool},
  +pairs_read: PairsRead(problem,storage,pairs,Nat.mul(pair,U32.to_nat(inner)),lane,pair_channel(lane,pair)),
  +bias_read: {PointSpec.pair_lane(lane,PointReadback.biases(bias,PointReadback.target_groups(PointReadback.Target{Gemm.PointMatrix{Gemm.TileMatrix{channels,positions,inner,start,output_positions},cell,stride},
      PointJobs.Readers{bias,cells,pairs}}),U32.from_nat(pair))) ==
    TileLaws.select_lane(Nat.mod(pair_channel(lane,pair),8n),Tree.read(Tile.Tile8,Inputs.bias_cells(problem,storage),U32.from_nat(Nat.div(pair_channel(lane,pair),8n)))) : F32}) ->
  {PointSpec.pair_lane(lane,PointReadback.stored_value(PointReadback.Target{Gemm.PointMatrix{Gemm.TileMatrix{channels,positions,inner,start,output_positions},cell,stride},
      PointJobs.Readers{bias,cells,pairs}},U32.from_nat(pair),U32.from_nat(unit),row)) ==
    Values.matrix_value(problem,pair_channel(lane,pair),output_row,output_column) : F32}:
  +groups = U32.to_nat((inner / 16 : U32))
  +channel = pair_channel(lane,pair)
  +address = Nat.mul(pair,U32.to_nat(inner))
  +target = {PointReadback.Target{Gemm.PointMatrix{Gemm.TileMatrix{channels,positions,inner,start,output_positions},cell,stride},PointJobs.Readers{bias,cells,pairs}} : PointReadback.Target}
  +dotted = OrderedMatrix.dot(~Nat,~Keys.Key,~Nat,~Gemm.Both<Keys.Origin,PointSpec.Sampled>,~Keys.key_input(),~Keys.key_weight(),Keys.keys(groups,0n),lane,row,
    Gemm.Both{Keys.TapOrigin{address,first},PointSpec.Sampled{cells,pairs,stride}},0.0)
  +biased = PointSpec.pair_lane(lane,PointReadback.biases(bias,PointReadback.target_groups(target),U32.from_nat(pair)))
  +matrix = Values.matrix_dot(MatrixSpec.kchannels(Nat.mul(groups,16n),0n,1n),problem,channel,output_row,output_column,0.0)
  +agreed = job_agreements(limits,problem,storage,valid,supported,kernel_one,stride_one,pad_zero,groups,channels_exact,cells,pairs,stride,address,first,row,row_small,
    base,row_first,rows,contents,rows_fit,offset,offset_inside,first_exact,output_row,output_column,pixel_exact,row_inside,column_inside,lane,channel,channel_inside,pairs_read)
  %Equal.sym(Nat,Guards.input_channel_count(problem),Nat.mul(groups,16n),channels_exact) :
    {PointSpec.pair_lane(lane,PointReadback.stored_value(target,U32.from_nat(pair),U32.from_nat(unit),row)) ==
      (Values.matrix_dot(MatrixSpec.kchannels(_,0n,Guards.kernel(problem)),problem,channel,output_row,output_column,0.0) + ConvSpec.bias_at(problem,channel) : F32) : F32}
  %Equal.sym(Nat,Guards.kernel(problem),1n,kernel_one) :
    {PointSpec.pair_lane(lane,PointReadback.stored_value(target,U32.from_nat(pair),U32.from_nat(unit),row)) ==
      (Values.matrix_dot(MatrixSpec.kchannels(Nat.mul(groups,16n),0n,_),problem,channel,output_row,output_column,0.0) + ConvSpec.bias_at(problem,channel) : F32) : F32}
  Equal.trans(F32,PointSpec.pair_lane(lane,PointReadback.stored_value(target,U32.from_nat(pair),U32.from_nat(unit),row)),(dotted + biased : F32),
    (matrix + ConvSpec.bias_at(problem,channel) : F32),
    Equal.trans(F32,PointSpec.pair_lane(lane,PointReadback.stored_value(target,U32.from_nat(pair),U32.from_nat(unit),row)),
      (OrderedMatrix.dot(~Nat,~Keys.Key,~Nat,~Gemm.Both<Keys.Origin,PointSpec.Sampled>,~Keys.key_input(),~Keys.key_weight(),Keys.keys(groups,0n),lane,row,
        Gemm.Both{Keys.TapOrigin{address,Keys.job_first(cell,unit,stride)},PointSpec.Sampled{cells,pairs,stride}},0.0) + biased : F32),(dotted + biased : F32),
      Keys.stored_lane(channels,positions,inner,start,output_positions,cell,stride,bias,cells,pairs,pair,unit,row,lane),
      Equal.cong(F32,F32,value => (value + biased : F32),
        OrderedMatrix.dot(~Nat,~Keys.Key,~Nat,~Gemm.Both<Keys.Origin,PointSpec.Sampled>,~Keys.key_input(),~Keys.key_weight(),Keys.keys(groups,0n),lane,row,
          Gemm.Both{Keys.TapOrigin{address,Keys.job_first(cell,unit,stride)},PointSpec.Sampled{cells,pairs,stride}},0.0),dotted,
        Keys.origin_dot(Keys.keys(groups,0n),address,Keys.job_first(cell,unit,stride),first,first_same,PointSpec.Sampled{cells,pairs,stride},lane,row,0.0))),
    Equal.trans(F32,(dotted + biased : F32),(matrix + biased : F32),(matrix + ConvSpec.bias_at(problem,channel) : F32),
      Equal.cong(F32,F32,value => (value + biased : F32),dotted,matrix,
        keyed_matrix(groups,address,first,cells,pairs,stride,row,lane,problem,channel,output_row,output_column,agreed)),
      Equal.cong(F32,F32,value => (matrix + value : F32),biased,ConvSpec.bias_at(problem,channel),
        Equal.trans(F32,biased,TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,Inputs.bias_cells(problem,storage),U32.from_nat(Nat.div(channel,8n)))),
          ConvSpec.bias_at(problem,channel),bias_read,GemmMatrix.bias_value(limits,problem,storage,channel,valid,supported,channel_inside)))))

# ---- The pairs the kernel builds ----

# The weight cells of a supported problem fit in half a word.
def weights_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: Guards.Valid<problem>,
  +support: Guards.Supported<problem,Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits)>) ->
  {Nat.is_le(Nat.mul(TiledValues.groups_of(Guards.output_channel_count(problem)),Coordinates.inner(problem)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  Guards.Valid{v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10} = valid
  +outputs = Guards.output_channel_count(problem)
  +inner = Coordinates.inner(problem)
  +capacity = U32.to_nat(Machine.capacity(limits))
  Order.transitive(Nat.mul(TiledValues.groups_of(outputs),inner),capacity,U32.to_nat(Machine.half_word(limits)),
    Order.transitive(Nat.mul(TiledValues.groups_of(outputs),inner),Nat.mul(outputs,inner),capacity,
      RowMajor.mul_le(TiledValues.groups_of(outputs),outputs,inner,Ceiling.minimal(outputs,8n,outputs,{==},RowMajor.product_covers(outputs,8n,{==}))),
      %Order.equal_of_is_eq(ConvSpec.length(Guards.weights(problem)),Nat.mul(outputs,inner),v9) : {Nat.is_le(_,capacity) == True{} : Bool}
      NativeWeightRefinement.weights_bound(limits,problem,support)),
    Bounds.as_nat_le(Machine.capacity(limits),Machine.half_word(limits),Limits.capacity_within_half(limits)))

# A pair below the output channels is followed by a group of them.
def pair_follows(+pair: Nat,+lane: Nat,+outputs: Nat,+inside: {Nat.is_lt(pair_channel(lane,pair),outputs) == True{} : Bool}) ->
  {Nat.is_le(1n+Nat.mul(pair,2n),TiledValues.groups_of(outputs)) == True{} : Bool}:
  +group = Nat.mul(pair,2n)
  +start = {%Equal.sym(Nat,Nat.mul(group,8n),Nat.mul(pair,16n),pair_groups(pair)) : {Nat.is_lt(_,outputs) == True{} : Bool}
    Order.le_lt_trans(Nat.mul(pair,16n),pair_channel(lane,pair),outputs,Order.right_le(lane,Nat.mul(pair,16n)),inside) : {Nat.is_lt(Nat.mul(group,8n),outputs) == True{} : Bool}}
  Order.strict_follow(group,TiledValues.groups_of(outputs),
    %Equal.sym(Bool,Nat.is_lt(group,TiledValues.groups_of(outputs)),Nat.is_lt(Nat.mul(group,8n),outputs),Ceiling.upper_comparison(outputs,8n,group,{==})) : {_ == True{} : Bool}
    start)

# What the pairs of one lane of one pair are read from.
type PairFacts is Data:
  PairFacts{limits: Machine.Limits,problem: ConvSpec.Problem,storage: Inputs.Contract<problem>,valid: {ConvSpec.valid(problem) == True{} : Bool},
    supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},channels: U32,inner: U32,
    groups_exact: {U32.to_nat(Gemm.blocks(channels)) == TiledValues.groups_of(Guards.output_channel_count(problem)) : Nat},
    inner_exact: {U32.to_nat(inner) == Coordinates.inner(problem) : Nat},
    count_exact: {PointBuffer.pair_count(channels,inner) == Nat.mul(U32.to_nat(Gemm.pairs_of(channels)),Coordinates.inner(problem)) : Nat},
    output: Tree.Tree<Tile.Tile16>,depth: Nat,balanced: Shape.Balanced(Tile.Tile16,depth,output),limit: {Nat.is_le(depth,31n) == True{} : Bool},
    capacity_ok: {U32.is_le(Tree.capacity(Tile.Tile16,output),2147483648) == True{} : Bool},
    room: {Nat.is_le(PointBuffer.pair_count(channels,inner),U32.to_nat(Tree.capacity(Tile.Tile16,output))) == True{} : Bool},
    pair: Nat,pair_inside: {Nat.is_lt(pair,U32.to_nat(Gemm.pairs_of(channels))) == True{} : Bool},lane: Nat,lane_small: {Nat.is_lt(lane,16n) == True{} : Bool},
    channel_inside: {Nat.is_lt(pair_channel(lane,pair),Guards.output_channel_count(problem)) == True{} : Bool}}

def pair_inputs(context: PairFacts) -> Nat:
  PairFacts{limits,problem,storage,valid,supported,channels,inner,groups_exact,inner_exact,count_exact,output,depth,balanced,limit,capacity_ok,room,pair,pair_inside,lane,lane_small,channel_inside} = context
  Guards.input_channel_count(problem)

def PairHolds(+context: PairFacts,+input: Nat) -> Data:
  PairFacts{+limits,+problem,+storage,+valid,+supported,+channels,+inner,+groups_exact,+inner_exact,+count_exact,+output,+depth,+balanced,+limit,+capacity_ok,+room,+pair,+pair_inside,+lane,+lane_small,+channel_inside} = context
  PairRead(problem,storage,PointBuffer.pair_loop(channels,inner,Inputs.weight_cells(problem,storage),output),Nat.mul(pair,U32.to_nat(inner)),lane,pair_channel(lane,pair),input)

# The weight tile of a group at an input channel.
def weight_tile(context: Gemm.Both<Tree.Tree<Tile.Tile8>,Gemm.Both<Nat,Nat>>,group: Nat) -> Tile.Tile8:
  Gemm.Both{weights,Gemm.Both{inner,input}} = context
  Tree.read(Tile.Tile8,weights,U32.from_nat(Nat.add(Nat.mul(group,inner),input)))

# Lane lane of the pair the kernel builds at an input channel holds that
# channel's lane of the weight cell of its output channel.
def pair_holds(+context: PairFacts,+input: Nat,+inside: {Nat.is_lt(input,pair_inputs(context)) == True{} : Bool}) -> PairHolds(context,input):
  PairFacts{+limits,+problem,+storage,+valid,+supported,+channels,+inner,+groups_exact,+inner_exact,+count_exact,+output,+depth,+balanced,+limit,+capacity_ok,+room,+pair,+pair_inside,+lane,+lane_small,+channel_inside} = context
  +checked = Guards.valid(problem,valid)
  +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  +inner_count = Coordinates.inner(problem)
  +outputs = Guards.output_channel_count(problem)
  +channel = pair_channel(lane,pair)
  +weights = Inputs.weight_cells(problem,storage)
  +pairs = PointBuffer.pair_loop(channels,inner,weights,output)
  +index = Nat.add(Nat.mul(pair,inner_count),input)
  +kernel_positive = Coordinates.kernel_positive(problem,checked)
  +tap_inside = {Order.lt_le(input,Guards.input_channel_count(problem),inner_count,inside,
      RowMajor.product_covers(Guards.input_channel_count(problem),Nat.mul(Guards.kernel(problem),Guards.kernel(problem)),
        Order.product_positive(Guards.kernel(problem),Guards.kernel(problem),kernel_positive,kernel_positive))) :
    {Nat.is_lt(input,inner_count) == True{} : Bool}}
  +index_inside = {%Equal.sym(Nat,PointBuffer.pair_count(channels,inner),Nat.mul(U32.to_nat(Gemm.pairs_of(channels)),inner_count),count_exact) :
      {Nat.is_lt(index,_) == True{} : Bool}
    RowMajor.radix(pair,U32.to_nat(Gemm.pairs_of(channels)),input,inner_count,pair_inside,tap_inside) :
    {Nat.is_lt(index,PointBuffer.pair_count(channels,inner)) == True{} : Bool}}
  +lower = Tree.read(Tile.Tile8,weights,U32.from_nat(Nat.add(Nat.mul(Nat.mul(pair,2n),inner_count),input)))
  +upper = Tree.read(Tile.Tile8,weights,U32.from_nat(Nat.add(Nat.mul(Nat.min(1n+Nat.mul(pair,2n),Nat.sub(TiledValues.groups_of(outputs),1n)),inner_count),input)))
  +tiles_exact = Equal.trans(Tile.Tile16,Tree.read(Tile.Tile16,pairs,U32.from_nat(index)),
    PointBuffer.pair_tiles(weights,Gemm.PairPlan{Gemm.blocks(channels),inner},U32.from_nat(index)),Tile.Tiles{lower,upper},
    PointBuffer.pair_at(channels,inner,weights,output,depth,balanced,limit,capacity_ok,room,index,index_inside),
    PointBuffer.pair_address(weights,Gemm.blocks(channels),inner,TiledValues.groups_of(outputs),inner_count,groups_exact,inner_exact,
      Machine.half_word(limits),Limits.half_word_fits(limits),weights_room(limits,problem,checked,support),
      Limits.double_bound(limits,inner_count,Coordinates.inner_bound(limits,problem,checked,support)),pair,input,tap_inside,
      pair_follows(pair,lane,outputs,channel_inside)))
  %Equal.sym(Nat,U32.to_nat(inner),inner_count,inner_exact) : PairRead(problem,storage,pairs,Nat.mul(pair,_),lane,channel,input)
  Equal.trans(F32,PointSpec.pair_lane(lane,Tree.read(Tile.Tile16,pairs,U32.from_nat(index))),PointSpec.pair_lane(lane,Tile.Tiles{lower,upper}),
    TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,weights,U32.from_nat(Nat.add(Nat.mul(Nat.div(channel,8n),inner_count),input)))),
    Equal.cong(Tile.Tile16,F32,tile => PointSpec.pair_lane(lane,tile),Tree.read(Tile.Tile16,pairs,U32.from_nat(index)),Tile.Tiles{lower,upper},tiles_exact),
    pair_lane_group(~Gemm.Both<Tree.Tree<Tile.Tile8>,Gemm.Both<Nat,Nat>>,~weight_tile,Gemm.Both{weights,Gemm.Both{inner_count,input}},pair,lane,outputs,lane_small,channel_inside))

# Lane lane of the pairs the kernel builds holds, at every input channel, that
# channel's lane of the weight cell of its output channel.
def pairs_read(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+channels: U32,+inner: U32,
  +groups_exact: {U32.to_nat(Gemm.blocks(channels)) == TiledValues.groups_of(Guards.output_channel_count(problem)) : Nat},
  +inner_exact: {U32.to_nat(inner) == Coordinates.inner(problem) : Nat},
  +count_exact: {PointBuffer.pair_count(channels,inner) == Nat.mul(U32.to_nat(Gemm.pairs_of(channels)),Coordinates.inner(problem)) : Nat},
  +output: Tree.Tree<Tile.Tile16>,+depth: Nat,+balanced: Shape.Balanced(Tile.Tile16,depth,output),+limit: {Nat.is_le(depth,31n) == True{} : Bool},
  +capacity_ok: {U32.is_le(Tree.capacity(Tile.Tile16,output),2147483648) == True{} : Bool},
  +room: {Nat.is_le(PointBuffer.pair_count(channels,inner),U32.to_nat(Tree.capacity(Tile.Tile16,output))) == True{} : Bool},
  +pair: Nat,+pair_inside: {Nat.is_lt(pair,U32.to_nat(Gemm.pairs_of(channels))) == True{} : Bool},+lane: Nat,+lane_small: {Nat.is_lt(lane,16n) == True{} : Bool},
  +channel_inside: {Nat.is_lt(pair_channel(lane,pair),Guards.output_channel_count(problem)) == True{} : Bool}) ->
  PairsRead(problem,storage,PointBuffer.pair_loop(channels,inner,Inputs.weight_cells(problem,storage),output),Nat.mul(pair,U32.to_nat(inner)),lane,pair_channel(lane,pair)):
  Members.range_all(~PairFacts,~PairHolds,~pair_inputs,~pair_holds,Guards.input_channel_count(problem),0n,
    PairFacts{limits,problem,storage,valid,supported,channels,inner,groups_exact,inner_exact,count_exact,output,depth,balanced,limit,capacity_ok,room,pair,pair_inside,lane,lane_small,channel_inside},
    Order.reflexive(Guards.input_channel_count(problem)))

# ---- The biases the kernel adds ----

# The bias tile of a group.
def bias_tile(bias: Tree.Tree<Tile.Tile8>,group: Nat) -> Tile.Tile8:
  Tree.read(Tile.Tile8,bias,U32.from_nat(group))

# Lane lane of the biases of a pair is the bias of its output channel.
def bias_read(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+channels: U32,
  +groups_exact: {U32.to_nat(Gemm.blocks(channels)) == TiledValues.groups_of(Guards.output_channel_count(problem)) : Nat},
  +pair: Nat,+lane: Nat,+lane_small: {Nat.is_lt(lane,16n) == True{} : Bool},
  +channel_inside: {Nat.is_lt(pair_channel(lane,pair),Guards.output_channel_count(problem)) == True{} : Bool}) ->
  {PointSpec.pair_lane(lane,PointReadback.biases(Inputs.bias_cells(problem,storage),Gemm.blocks(channels),U32.from_nat(pair))) ==
    TileLaws.select_lane(Nat.mod(pair_channel(lane,pair),8n),Tree.read(Tile.Tile8,Inputs.bias_cells(problem,storage),U32.from_nat(Nat.div(pair_channel(lane,pair),8n)))) : F32}:
  +checked = Guards.valid(problem,valid)
  +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  +outputs = Guards.output_channel_count(problem)
  +groups = TiledValues.groups_of(outputs)
  +bias = Inputs.bias_cells(problem,storage)
  +room = {Order.transitive(groups,Nat.mul(groups,Coordinates.inner(problem)),U32.to_nat(Machine.half_word(limits)),
      RowMajor.product_covers(groups,Coordinates.inner(problem),Coordinates.inner_positive(problem,checked)),weights_room(limits,problem,checked,support)) :
    {Nat.is_le(groups,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}}
  +lower = Tree.read(Tile.Tile8,bias,U32.from_nat(Nat.mul(pair,2n)))
  +upper = Tree.read(Tile.Tile8,bias,U32.from_nat(Nat.min(1n+Nat.mul(pair,2n),Nat.sub(groups,1n))))
  Equal.trans(F32,PointSpec.pair_lane(lane,PointReadback.biases(bias,Gemm.blocks(channels),U32.from_nat(pair))),PointSpec.pair_lane(lane,Tile.Tiles{lower,upper}),
    TileLaws.select_lane(Nat.mod(pair_channel(lane,pair),8n),Tree.read(Tile.Tile8,bias,U32.from_nat(Nat.div(pair_channel(lane,pair),8n)))),
    Equal.cong(Tile.Tile16,F32,tile => PointSpec.pair_lane(lane,tile),PointReadback.biases(bias,Gemm.blocks(channels),U32.from_nat(pair)),Tile.Tiles{lower,upper},
      PointReadback.biases_address(bias,Gemm.blocks(channels),groups,groups_exact,Machine.half_word(limits),Limits.half_word_fits(limits),room,pair,
        pair_follows(pair,lane,outputs,channel_inside))),
    pair_lane_group(~Tree.Tree<Tile.Tile8>,~bias_tile,bias,pair,lane,outputs,lane_small,channel_inside))
