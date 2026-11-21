# The cells of the input tensor hold the input cells of every image: cell
# pixel * groups + group of image n is the input cell of that pixel and group.
import Base
import ../specification.bend as ConvSpec
import ../../lib/kernel_tile.bend as Tile
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/machine_limits.bend as Machine
import ../../lib/tensor_grid.bend as Grid
import ../../lib/tensor_tiled_cells.bend as TiledCells
import ../../lib/proofs/tensor_tiled_representation.bend as Representation
import ../../lib/proofs/tensor_tiled_values.bend as TiledValues
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/binary_radix.bend as Radix
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/word_multiplication.bend as WordMultiplication
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/kernel_gemm.bend as KernelGemm
import ../inference.bend as NativeConv
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./supported_coordinate_bounds.bend as Coordinates
import ./window_contents.bend as WindowBounds
import ./input_storage.bend as Inputs
import ./window_cell_contents.bend as InputCells
import ./image_problems.bend as ImageProblems
import ./window_packed_band.bend as Band
import ./window_packed_values.bend as Calls

# ---- Words of the input cells ----

# The plane of the input as a word.
def input_plane(+problem: ConvSpec.Problem) -> U32:
  (U32.from_nat(Guards.height(problem)) * U32.from_nat(Guards.width(problem)) : U32)

# The first lane of the input cell of a pixel and group.
def input_spot(+problem: ConvSpec.Problem,+group: Nat,+pixel: Nat) -> TiledCells.Spot:
  TiledCells.Spot{U32.from_nat(Nat.mul(group,8n)),U32.from_nat(Nat.add(Nat.mul(Nat.mul(group,8n),U32.to_nat(input_plane(problem))),pixel)),
    input_plane(problem),U32.from_nat(Guards.input_channel_count(problem))}

def input_plane_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {U32.to_nat(input_plane(problem)) == Nat.mul(Guards.height(problem),Guards.width(problem)) : Nat}:
  +height = Guards.height(problem)
  +width = Guards.width(problem)
  +native_height = U32.from_nat(height)
  +native_width = U32.from_nat(width)
  +plane = (native_height * native_width : U32)
  +scalar = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  +height_exact = Coordinates.height_exact(limits,problem,scalar)
  +width_exact = Coordinates.width_exact(limits,problem,scalar)
  +capacity = Machine.capacity(limits)
  +cap = U32.to_nat(capacity)
  %height_exact : {U32.to_nat(plane) == Nat.mul(_,width) : Nat}
  %width_exact : {U32.to_nat(plane) == Nat.mul(U32.to_nat(native_height),_) : Nat}
  WordMultiplication.bounded_multiply(native_height,native_width,capacity,Limits.capacity_fits(limits),
    %Equal.sym(Nat,U32.to_nat(native_height),height,height_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(native_width)),cap) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(native_width),width,width_exact) : {Nat.is_le(Nat.mul(height,_),cap) == True{} : Bool}
    WindowBounds.area_room(limits,problem,valid,supported))

def input_channels_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(Guards.input_channel_count(problem))) == Guards.input_channel_count(problem) : Nat}:
  Coordinates.ci_exact(limits,problem,Padded.scalar(limits,problem,Support.fast(limits,problem,supported)))

def input_channels_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(U32.from_nat(Guards.input_channel_count(problem))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  +ci = Guards.input_channel_count(problem)
  +scalar = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(ci)),ci,Coordinates.ci_exact(limits,problem,scalar)) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  Limits.dimension_capacity(limits,ci,Coordinates.ci_bound(limits,problem,scalar))

def input_groups_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {U32.to_nat(Gemm.blocks(U32.from_nat(Guards.input_channel_count(problem)))) == InputCells.group_count(problem) : Nat}:
  +ci = Guards.input_channel_count(problem)
  %input_channels_exact(limits,problem,supported) : {U32.to_nat(Gemm.blocks(U32.from_nat(ci))) == TiledValues.groups_of(_) : Nat}
  Encoded.eight_blocks_exact(limits,U32.from_nat(ci),input_channels_room(limits,problem,supported))

# The input cell of a pixel and group holds the eight channels from group * 8
# of that pixel, +0 past the channels.
def input_cell(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+pixel: Nat,+group: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +pixel_inside: {Nat.is_lt(pixel,Nat.mul(Guards.height(problem),Guards.width(problem))) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,InputCells.group_count(problem)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,InputCells.input_cells(problem,storage),U32.from_nat(Nat.add(Nat.mul(pixel,InputCells.group_count(problem)),group))) ==
    TiledValues.spot_cell(Inputs.input_tree(problem,storage),input_spot(problem,group,pixel)) : Tile.Tile8}:
  +ci = Guards.input_channel_count(problem)
  +height = Guards.height(problem)
  +width = Guards.width(problem)
  +area = Nat.mul(height,width)
  +native_channels = U32.from_nat(ci)
  +plane = (U32.from_nat(height) * U32.from_nat(width) : U32)
  +native_groups = Gemm.blocks(native_channels)
  +cells = (plane * native_groups : U32)
  +groups = InputCells.group_count(problem)
  +capacity = Machine.capacity(limits)
  +cap = U32.to_nat(capacity)
  +plane_exact = input_plane_exact(limits,problem,valid,supported)
  +groups_exact = input_groups_exact(limits,problem,supported)
  +cells_bound = {%Equal.sym(Nat,U32.to_nat(plane),area,plane_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(native_groups)),cap) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(native_groups),groups,groups_exact) : {Nat.is_le(Nat.mul(area,_),cap) == True{} : Bool}
    InputCells.cells_room(limits,problem,valid,supported,height,Order.reflexive(height)) :
      {Nat.is_le(Nat.mul(U32.to_nat(plane),U32.to_nat(native_groups)),cap) == True{} : Bool}}
  +cells_word = WordMultiplication.bounded_multiply(plane,native_groups,capacity,Limits.capacity_fits(limits),cells_bound)
  +cells_bounded = {%Equal.sym(Bool,U32.is_le(cells,capacity),Nat.is_le(U32.to_nat(cells),cap),Comparison.le(cells,capacity)) : {_ == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(cells),Nat.mul(U32.to_nat(plane),U32.to_nat(native_groups)),cells_word) : {Nat.is_le(_,cap) == True{} : Bool} cells_bound : {U32.is_le(cells,capacity) == True{} : Bool}}
  %groups_exact :
    {Tree.read(Tile.Tile8,InputCells.input_cells(problem,storage),U32.from_nat(Nat.add(Nat.mul(pixel,_),group))) ==
      TiledValues.spot_cell(Inputs.input_tree(problem,storage),input_spot(problem,group,pixel)) : Tile.Tile8}
  TiledValues.tiled_read(limits,cells,plane,native_channels,native_groups,Tile.filled(0.0),Inputs.input_tree(problem,storage),pixel,group,
    Equal.sym(Nat,U32.to_nat(cells),Nat.mul(U32.to_nat(plane),U32.to_nat(native_groups)),cells_word),cells_bounded,
    %Equal.sym(Nat,U32.to_nat(plane),area,plane_exact) : {Nat.is_lt(pixel,_) == True{} : Bool} pixel_inside,
    %Equal.sym(Nat,U32.to_nat(native_groups),groups,groups_exact) : {Nat.is_lt(group,_) == True{} : Bool} group_inside)

# ---- The input of an image ----

# The logical index of lane channel of the cell of pixel of image image is
# value image * area + channel * plane + pixel of the input.
def input_index(+image: Nat,+channels: Nat,+plane: Nat,+pixel: Nat,+channel: Nat,+positive: {Nat.is_lt(0n,plane) == True{} : Bool},
  +pixel_inside: {Nat.is_lt(pixel,plane) == True{} : Bool}) ->
  {Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(Nat.add(Nat.mul(image,plane),pixel),plane),channels),channel),plane),Nat.mod(Nat.add(Nat.mul(image,plane),pixel),plane)) ==
    Nat.add(Nat.mul(image,Nat.mul(channels,plane)),Nat.add(Nat.mul(channel,plane),pixel)) : Nat}:
  %Equal.sym(Nat,Nat.div(Nat.add(Nat.mul(image,plane),pixel),plane),image,Radix.quotient(image,pixel,plane,positive,pixel_inside)) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(_,channels),channel),plane),Nat.mod(Nat.add(Nat.mul(image,plane),pixel),plane)) ==
      Nat.add(Nat.mul(image,Nat.mul(channels,plane)),Nat.add(Nat.mul(channel,plane),pixel)) : Nat}
  %Equal.sym(Nat,Nat.mod(Nat.add(Nat.mul(image,plane),pixel),plane),pixel,Radix.remainder(image,pixel,plane,positive,pixel_inside)) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(image,channels),channel),plane),_) == Nat.add(Nat.mul(image,Nat.mul(channels,plane)),Nat.add(Nat.mul(channel,plane),pixel)) : Nat}
  %Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(image,channels),channel),plane),Nat.add(Nat.mul(Nat.mul(image,channels),plane),Nat.mul(channel,plane)),RowMajor.distrib(Nat.mul(image,channels),channel,plane)) :
    {Nat.add(_,pixel) == Nat.add(Nat.mul(image,Nat.mul(channels,plane)),Nat.add(Nat.mul(channel,plane),pixel)) : Nat}
  %Equal.sym(Nat,Nat.mul(Nat.mul(image,channels),plane),Nat.mul(image,Nat.mul(channels,plane)),RowMajor.assoc(image,channels,plane)) :
    {Nat.add(Nat.add(_,Nat.mul(channel,plane)),pixel) == Nat.add(Nat.mul(image,Nat.mul(channels,plane)),Nat.add(Nat.mul(channel,plane),pixel)) : Nat}
  Addition.associative(Nat.mul(image,Nat.mul(channels,plane)),Nat.mul(channel,plane),pixel)

# Lane lane of group group past lane planes: the plane of the channel.
def lane_plane(+lane: Nat,+group: Nat,+plane: Nat,+pixel: Nat) ->
  {Nat.add(Nat.mul(lane,plane),Nat.add(Nat.mul(Nat.mul(group,8n),plane),pixel)) == Nat.add(Nat.mul(Nat.add(lane,Nat.mul(group,8n)),plane),pixel) : Nat}:
  %Equal.sym(Nat,Nat.mul(Nat.add(lane,Nat.mul(group,8n)),plane),Nat.add(Nat.mul(lane,plane),Nat.mul(Nat.mul(group,8n),plane)),RowMajor.distrib(lane,Nat.mul(group,8n),plane)) : {Nat.add(Nat.mul(lane,plane),Nat.add(Nat.mul(Nat.mul(group,8n),plane),pixel)) == Nat.add(_,pixel) : Nat}
  Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(lane,plane),Nat.mul(Nat.mul(group,8n),plane)),pixel),Nat.add(Nat.mul(lane,plane),Nat.add(Nat.mul(Nat.mul(group,8n),plane),pixel)),
    Addition.associative(Nat.mul(lane,plane),Nat.mul(Nat.mul(group,8n),plane),pixel))

# A lane of the input tensor cell of a pixel of an image is the lane of the
# input cell of that pixel: the input value of its channel, +0 past the
# channels.
def held_lane(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weights: List<&2,F32>,+bias: List<&2,F32>,+image: Nat,+upper: U32,+upper_ok: {U32.is_le(upper,2147483648) == True{} : Bool},+upper_bound: {U32.is_le(upper,1073741824) == True{} : Bool},+area_positive: {Nat.is_lt(0n,Nat.mul(channels,Nat.mul(height,width))) == True{} : Bool},+area_room: {Nat.is_le(Nat.mul(channels,Nat.mul(height,width)),U32.to_nat(upper)) == True{} : Bool},+weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,+parameters: Inputs.Parameters(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,0n)),+limits: Machine.Limits,+valid: {ConvSpec.valid(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image)) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image)) == True{} : Bool},+outer_word: U32,+after_word: U32,+lanes_word: U32,+groups_word: U32,+after_exact: {U32.to_nat(after_word) == Nat.mul(height,width) : Nat},+lanes_exact: {U32.to_nat(lanes_word) == channels : Nat},+groups_exact: {U32.to_nat(groups_word) == InputCells.group_count(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image)) : Nat},+pixel: Nat,+group: Nat,+lane: Nat,
  +pixel_inside: {Nat.is_lt(pixel,Nat.mul(height,width)) == True{} : Bool},+group_inside: {Nat.is_lt(group,InputCells.group_count(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {Representation.lane(~Logical,~original,logical,TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}},Nat.add(Nat.mul(image,Nat.mul(height,width)),pixel),group,lane) ==
    TiledValues.spot_value(Inputs.input_tree(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),Calls.image_contract(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters)),TiledValues.steps(input_spot(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),group,pixel),lane)) : F32}:
  +plane = Nat.mul(height,width)
  +area = Nat.mul(channels,plane)
  +channel = Nat.add(lane,Nat.mul(group,8n))
  +native = input_plane(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image))
  +tree = Inputs.input_tree(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),Calls.image_contract(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters))
  +base = Nat.add(Nat.mul(Nat.mul(group,8n),U32.to_nat(native)),pixel)
  +position = Nat.add(Nat.mul(channel,plane),pixel)
  +outer = Nat.add(Nat.mul(image,plane),pixel)
  +plane_exact = input_plane_exact(limits,ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),valid,supported)
  +channels_exact = input_channels_exact(limits,ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),supported)
  +plane_positive = DivisionOrder.right_factor_positive(channels,plane,area_positive)
  +after_one = {%Equal.sym(Nat,U32.to_nat(after_word),plane,after_exact) :{Representation.least_one(_) == plane : Nat}
    Representation.least_positive(plane,plane_positive) : {Representation.least_one(U32.to_nat(after_word)) == plane : Nat}}
  +bound = TiledValues.channel_bound(limits,U32.from_nat(channels),group,lane,TiledValues.grouped(limits,U32.from_nat(channels),input_channels_room(limits,ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),supported)),
    %Equal.sym(Nat,U32.to_nat(Gemm.blocks(U32.from_nat(channels))),InputCells.group_count(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image)),input_groups_exact(limits,ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),supported)) :
      {Nat.is_lt(group,_) == True{} : Bool} group_inside,lane_inside)
  +test = {%channels_exact : {U32.is_lt(U32.from_nat(channel),U32.from_nat(channels)) == Nat.is_lt(channel,_) : Bool}
    TiledValues.channel_compare(limits,channel,U32.from_nat(channels),bound) : {U32.is_lt(U32.from_nat(channel),U32.from_nat(channels)) == Nat.is_lt(channel,channels) : Bool}}
  +index = {%Equal.sym(Nat,Representation.least_one(U32.to_nat(after_word)),plane,after_one) :{Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(outer,_),U32.to_nat(lanes_word)),channel),_),Nat.mod(outer,_)) ==
      Nat.add(Nat.mul(image,area),position) : Nat}
    %Equal.sym(Nat,U32.to_nat(lanes_word),channels,lanes_exact) : {Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(outer,plane),_),channel),plane),Nat.mod(outer,plane)) == Nat.add(Nat.mul(image,area),position) : Nat}
    input_index(image,channels,plane,pixel,channel,plane_positive,pixel_inside) :
    {Representation.logical_index(TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}},outer,channel) == Nat.add(Nat.mul(image,area),position) : Nat}}
  +read = {%Equal.sym(Nat,U32.to_nat(native),plane,plane_exact) :{Nat.add(Nat.mul(lane,_),Nat.add(Nat.mul(Nat.mul(group,8n),_),pixel)) == position : Nat}
    lane_plane(lane,group,plane,pixel) : {Nat.add(Nat.mul(lane,U32.to_nat(native)),base) == position : Nat}}
  +length = ImageProblems.image_length(~Logical,~original,area,Nat.mul(image,area),logical)
  %Equal.sym(TiledCells.Spot,TiledValues.steps(input_spot(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),group,pixel),lane),
    TiledCells.Spot{U32.from_nat(channel),U32.from_nat(Nat.add(Nat.mul(lane,U32.to_nat(native)),base)),native,U32.from_nat(channels)},
    TiledValues.stepped(Nat.mul(group,8n),base,native,U32.from_nat(channels),lane)) :
    {Representation.lane(~Logical,~original,logical,TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}},outer,group,lane) == TiledValues.spot_value(tree,_) : F32}
  %Equal.sym(Bool,U32.is_lt(U32.from_nat(channel),U32.from_nat(channels)),Nat.is_lt(channel,channels),test) :
    {Representation.lane(~Logical,~original,logical,TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}},outer,group,lane) ==
      Bool.pick(F32,_,Tree.read(F32,tree,U32.from_nat(Nat.add(Nat.mul(lane,U32.to_nat(native)),base))),0.0) : F32}
  %Equal.sym(Nat,U32.to_nat(lanes_word),channels,lanes_exact) : {Bool.pick(F32,Nat.is_lt(channel,_),original(logical,U32.from_nat(Representation.logical_index(TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}},outer,channel))),0.0) ==
      Bool.pick(F32,Nat.is_lt(channel,channels),Tree.read(F32,tree,U32.from_nat(Nat.add(Nat.mul(lane,U32.to_nat(native)),base))),0.0) : F32}
  TiledValues.picked(Nat.is_lt(channel,channels),original(logical,U32.from_nat(Representation.logical_index(TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}},outer,channel))),
    Tree.read(F32,tree,U32.from_nat(Nat.add(Nat.mul(lane,U32.to_nat(native)),base))),
    +inside => {%Equal.sym(Nat,Nat.add(Nat.mul(lane,U32.to_nat(native)),base),position,read) :
        {original(logical,U32.from_nat(Representation.logical_index(TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}},outer,channel))) == Tree.read(F32,tree,U32.from_nat(_)) : F32}
      %Equal.sym(Nat,Representation.logical_index(TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}},outer,channel),Nat.add(Nat.mul(image,area),position),index) :
        {original(logical,U32.from_nat(_)) == Tree.read(F32,tree,U32.from_nat(position)) : F32}
      Equal.trans(F32,original(logical,U32.from_nat(Nat.add(Nat.mul(image,area),position))),ConvSpec.at(position,Guards.input(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image))),Tree.read(F32,tree,U32.from_nat(position)),
        Equal.sym(F32,ConvSpec.at(position,Guards.input(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image))),original(logical,U32.from_nat(Nat.add(Nat.mul(image,area),position))),
          ImageProblems.image_at(~Logical,~original,area,Nat.mul(image,area),logical,position,RowMajor.radix(channel,channels,pixel,plane,inside,pixel_inside))),
        Equal.sym(F32,Tree.read(F32,tree,U32.from_nat(position)),ConvSpec.at(position,Guards.input(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image))),
          Inputs.stored_value(Guards.input(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image)),Inputs.input_sequence(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),Calls.image_contract(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters)),position,
            %Equal.sym(Nat,ConvSpec.length(Guards.input(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image))),area,length) : {Nat.is_lt(position,_) == True{} : Bool}
            RowMajor.radix(channel,channels,pixel,plane,inside,pixel_inside)))) :
      {original(logical,U32.from_nat(Representation.logical_index(TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}},outer,channel))) == Tree.read(F32,tree,U32.from_nat(Nat.add(Nat.mul(lane,U32.to_nat(native)),base))) : F32}})

# The input tensor cell of image image at cell is the input cell of the image
# at cell.
def held_cell(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weights: List<&2,F32>,+bias: List<&2,F32>,+image: Nat,+upper: U32,+upper_ok: {U32.is_le(upper,2147483648) == True{} : Bool},+upper_bound: {U32.is_le(upper,1073741824) == True{} : Bool},+area_positive: {Nat.is_lt(0n,Nat.mul(channels,Nat.mul(height,width))) == True{} : Bool},+area_room: {Nat.is_le(Nat.mul(channels,Nat.mul(height,width)),U32.to_nat(upper)) == True{} : Bool},+weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,+parameters: Inputs.Parameters(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,0n)),+limits: Machine.Limits,+valid: {ConvSpec.valid(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image)) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image)) == True{} : Bool},+outer_word: U32,+after_word: U32,+lanes_word: U32,+groups_word: U32,+after_exact: {U32.to_nat(after_word) == Nat.mul(height,width) : Nat},+lanes_exact: {U32.to_nat(lanes_word) == channels : Nat},+groups_exact: {U32.to_nat(groups_word) == InputCells.group_count(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image)) : Nat},+cell: Nat,+inside: {Nat.is_lt(cell,InputCells.count(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),height)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(Nat.mul(image,InputCells.count(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),height)),InputCells.count(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),height)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Representation.lifted(~Logical,~original,Representation.Lifted{logical,TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}}},U32.from_nat(Nat.add(Nat.mul(image,InputCells.count(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),height)),cell))) ==
    Tree.read(Tile.Tile8,InputCells.input_cells(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),Calls.image_contract(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters)),U32.from_nat(cell)) : Tile.Tile8}:
  +plane = Nat.mul(height,width)
  +groups = InputCells.group_count(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image))
  +count = InputCells.count(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),height)
  +start = Nat.mul(image,count)
  +number = Nat.add(start,cell)
  +pixel = Nat.div(cell,groups)
  +group = Nat.mod(cell,groups)
  +outer = Nat.add(Nat.mul(image,plane),pixel)
  +product_positive = Order.le_lt_trans(0n,cell,count,Order.zero_le(cell),inside)
  +groups_positive = DivisionOrder.right_factor_positive(plane,groups,product_positive)
  +pixel_inside = DivisionOrder.quotient_below(cell,plane,groups,groups_positive,inside)
  +group_inside = DivisionOrder.remainder_small(cell,groups,groups_positive)
  +split = DivisionOrder.quotient_exact(cell,groups,groups_positive)
  +number_room = Bounds.strict_le(number,U32.to_nat(Machine.capacity(limits)),
    Bounds.nat_trans(number,Nat.add(start,count),U32.to_nat(Machine.capacity(limits)),Order.offset_lt(start,cell,count,inside),room))
  +number_exact = Bounds.roundtrip(number,Machine.capacity(limits),Limits.capacity_fits(limits),number_room)
  +groups_one = {%Equal.sym(Nat,U32.to_nat(groups_word),groups,groups_exact) :{Representation.least_one(_) == groups : Nat}
    Representation.least_positive(groups,groups_positive) : {Representation.least_one(U32.to_nat(groups_word)) == groups : Nat}}
  +whole = Nat.mul(Nat.mul(image,plane),groups)
  +part = Nat.mul(pixel,groups)
  +placed = Equal.trans(Nat,number,Nat.add(Nat.mul(image,Nat.mul(plane,groups)),Nat.add(part,group)),Nat.add(Nat.mul(outer,groups),group),
    Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(image,Nat.mul(plane,groups)),value),cell,Nat.add(part,group),split),
    Equal.trans(Nat,Nat.add(Nat.mul(image,Nat.mul(plane,groups)),Nat.add(part,group)),Nat.add(whole,Nat.add(part,group)),Nat.add(Nat.mul(outer,groups),group),
      Equal.cong(Nat,Nat,value => Nat.add(value,Nat.add(part,group)),Nat.mul(image,Nat.mul(plane,groups)),whole,
        Equal.sym(Nat,whole,Nat.mul(image,Nat.mul(plane,groups)),RowMajor.assoc(image,plane,groups))),
      Equal.trans(Nat,Nat.add(whole,Nat.add(part,group)),Nat.add(Nat.add(whole,part),group),Nat.add(Nat.mul(outer,groups),group),
        Equal.sym(Nat,Nat.add(Nat.add(whole,part),group),Nat.add(whole,Nat.add(part,group)),Addition.associative(whole,part,group)),
        Equal.cong(Nat,Nat,value => Nat.add(value,group),Nat.add(whole,part),Nat.mul(outer,groups),
          Equal.sym(Nat,Nat.mul(outer,groups),Nat.add(whole,part),RowMajor.distrib(Nat.mul(image,plane),pixel,groups))))))
  +outer_exact = {%Equal.sym(Nat,number,Nat.add(Nat.mul(outer,groups),group),placed) : {Nat.div(_,groups) == outer : Nat}
    Radix.quotient(outer,group,groups,groups_positive,group_inside) : {Nat.div(number,groups) == outer : Nat}}
  +group_exact = {%Equal.sym(Nat,number,Nat.add(Nat.mul(outer,groups),group),placed) : {Nat.mod(_,groups) == group : Nat}
    Radix.remainder(outer,group,groups,groups_positive,group_inside) : {Nat.mod(number,groups) == group : Nat}}
  +tiles = TiledValues.tiles_equal(Representation.lane(~Logical,~original,logical,TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}},outer,group,0n),Representation.lane(~Logical,~original,logical,TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}},outer,group,1n),Representation.lane(~Logical,~original,logical,TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}},outer,group,2n),Representation.lane(~Logical,~original,logical,TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}},outer,group,3n),Representation.lane(~Logical,~original,logical,TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}},outer,group,4n),Representation.lane(~Logical,~original,logical,TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}},outer,group,5n),Representation.lane(~Logical,~original,logical,TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}},outer,group,6n),Representation.lane(~Logical,~original,logical,TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}},outer,group,7n),TiledValues.spot_value(Inputs.input_tree(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),Calls.image_contract(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters)),TiledValues.steps(input_spot(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),group,pixel),0n)),TiledValues.spot_value(Inputs.input_tree(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),Calls.image_contract(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters)),TiledValues.steps(input_spot(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),group,pixel),1n)),TiledValues.spot_value(Inputs.input_tree(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),Calls.image_contract(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters)),TiledValues.steps(input_spot(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),group,pixel),2n)),TiledValues.spot_value(Inputs.input_tree(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),Calls.image_contract(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters)),TiledValues.steps(input_spot(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),group,pixel),3n)),TiledValues.spot_value(Inputs.input_tree(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),Calls.image_contract(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters)),TiledValues.steps(input_spot(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),group,pixel),4n)),TiledValues.spot_value(Inputs.input_tree(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),Calls.image_contract(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters)),TiledValues.steps(input_spot(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),group,pixel),5n)),TiledValues.spot_value(Inputs.input_tree(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),Calls.image_contract(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters)),TiledValues.steps(input_spot(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),group,pixel),6n)),TiledValues.spot_value(Inputs.input_tree(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),Calls.image_contract(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters)),TiledValues.steps(input_spot(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),group,pixel),7n)),held_lane(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,limits,valid,supported,outer_word,after_word,lanes_word,groups_word,after_exact,lanes_exact,groups_exact,pixel,group,0n,pixel_inside,group_inside,{==}),held_lane(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,limits,valid,supported,outer_word,after_word,lanes_word,groups_word,after_exact,lanes_exact,groups_exact,pixel,group,1n,pixel_inside,group_inside,{==}),held_lane(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,limits,valid,supported,outer_word,after_word,lanes_word,groups_word,after_exact,lanes_exact,groups_exact,pixel,group,2n,pixel_inside,group_inside,{==}),held_lane(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,limits,valid,supported,outer_word,after_word,lanes_word,groups_word,after_exact,lanes_exact,groups_exact,pixel,group,3n,pixel_inside,group_inside,{==}),held_lane(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,limits,valid,supported,outer_word,after_word,lanes_word,groups_word,after_exact,lanes_exact,groups_exact,pixel,group,4n,pixel_inside,group_inside,{==}),held_lane(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,limits,valid,supported,outer_word,after_word,lanes_word,groups_word,after_exact,lanes_exact,groups_exact,pixel,group,5n,pixel_inside,group_inside,{==}),held_lane(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,limits,valid,supported,outer_word,after_word,lanes_word,groups_word,after_exact,lanes_exact,groups_exact,pixel,group,6n,pixel_inside,group_inside,{==}),held_lane(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,limits,valid,supported,outer_word,after_word,lanes_word,groups_word,after_exact,lanes_exact,groups_exact,pixel,group,7n,pixel_inside,group_inside,{==}))
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(number)),number,number_exact) :
    {Representation.cell(~Logical,~original,logical,TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}},Nat.div(_,Representation.least_one(U32.to_nat(groups_word))),Nat.mod(_,Representation.least_one(U32.to_nat(groups_word)))) ==
      Tree.read(Tile.Tile8,InputCells.input_cells(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),Calls.image_contract(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters)),U32.from_nat(cell)) : Tile.Tile8}
  %Equal.sym(Nat,Representation.least_one(U32.to_nat(groups_word)),groups,groups_one) : {Representation.cell(~Logical,~original,logical,TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}},Nat.div(number,_),Nat.mod(number,_)) == Tree.read(Tile.Tile8,InputCells.input_cells(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),Calls.image_contract(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters)),U32.from_nat(cell)) : Tile.Tile8}
  %Equal.sym(Nat,Nat.div(number,groups),outer,outer_exact) : {Representation.cell(~Logical,~original,logical,TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}},_,Nat.mod(number,groups)) == Tree.read(Tile.Tile8,InputCells.input_cells(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),Calls.image_contract(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters)),U32.from_nat(cell)) : Tile.Tile8}
  %Equal.sym(Nat,Nat.mod(number,groups),group,group_exact) : {Representation.cell(~Logical,~original,logical,TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}},outer,_) == Tree.read(Tile.Tile8,InputCells.input_cells(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),Calls.image_contract(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters)),U32.from_nat(cell)) : Tile.Tile8}
  %Equal.sym(Nat,cell,Nat.add(Nat.mul(pixel,groups),group),split) : {Representation.cell(~Logical,~original,logical,TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}},outer,group) == Tree.read(Tile.Tile8,InputCells.input_cells(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),Calls.image_contract(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters)),U32.from_nat(_)) : Tile.Tile8}
  Equal.trans(Tile.Tile8,Representation.cell(~Logical,~original,logical,TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}},outer,group),TiledValues.spot_cell(Inputs.input_tree(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),Calls.image_contract(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters)),input_spot(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),group,pixel)),
    Tree.read(Tile.Tile8,InputCells.input_cells(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),Calls.image_contract(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters)),U32.from_nat(Nat.add(Nat.mul(pixel,groups),group))),tiles,
    Equal.sym(Tile.Tile8,Tree.read(Tile.Tile8,InputCells.input_cells(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),Calls.image_contract(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters)),U32.from_nat(Nat.add(Nat.mul(pixel,groups),group))),TiledValues.spot_cell(Inputs.input_tree(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),Calls.image_contract(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters)),input_spot(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),group,pixel)),
      input_cell(limits,ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),Calls.image_contract(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters),pixel,group,valid,supported,pixel_inside,group_inside)))

# The input tensor cells of image image from position on, remaining of them.
def held_from(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weights: List<&2,F32>,+bias: List<&2,F32>,+image: Nat,+upper: U32,+upper_ok: {U32.is_le(upper,2147483648) == True{} : Bool},+upper_bound: {U32.is_le(upper,1073741824) == True{} : Bool},+area_positive: {Nat.is_lt(0n,Nat.mul(channels,Nat.mul(height,width))) == True{} : Bool},+area_room: {Nat.is_le(Nat.mul(channels,Nat.mul(height,width)),U32.to_nat(upper)) == True{} : Bool},+weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,+parameters: Inputs.Parameters(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,0n)),+limits: Machine.Limits,+valid: {ConvSpec.valid(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image)) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image)) == True{} : Bool},+outer_word: U32,+after_word: U32,+lanes_word: U32,+groups_word: U32,+after_exact: {U32.to_nat(after_word) == Nat.mul(height,width) : Nat},+lanes_exact: {U32.to_nat(lanes_word) == channels : Nat},+groups_exact: {U32.to_nat(groups_word) == InputCells.group_count(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image)) : Nat},remaining: Nat,+position: Nat,
  +room: {Nat.is_le(Nat.add(Nat.mul(image,InputCells.count(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),height)),InputCells.count(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),height)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(position,remaining),InputCells.count(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),height)) == True{} : Bool}) ->
  Members.All(Nat,{cell => {Representation.lifted(~Logical,~original,Representation.Lifted{logical,TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}}},U32.from_nat(Nat.add(Nat.mul(image,InputCells.count(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),height)),cell))) ==
      Tree.read(Tile.Tile8,InputCells.input_cells(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),Calls.image_contract(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters)),U32.from_nat(cell)) : Tile.Tile8} : Nat -> Data},Ranges.range(remaining,position)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      KernelGemm.Both{held_cell(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,limits,valid,supported,outer_word,after_word,lanes_word,groups_word,after_exact,lanes_exact,groups_exact,position,Order.room_index(position,rest,InputCells.count(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),height),fits),room),
        held_from(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,limits,valid,supported,outer_word,after_word,lanes_word,groups_word,after_exact,lanes_exact,groups_exact,rest,1n+position,room,Order.advance_room(position,rest,InputCells.count(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),height),fits))}

# Every input tensor cell of image image is the input cell of the image.
def held_image(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weights: List<&2,F32>,+bias: List<&2,F32>,+image: Nat,+upper: U32,+upper_ok: {U32.is_le(upper,2147483648) == True{} : Bool},+upper_bound: {U32.is_le(upper,1073741824) == True{} : Bool},+area_positive: {Nat.is_lt(0n,Nat.mul(channels,Nat.mul(height,width))) == True{} : Bool},+area_room: {Nat.is_le(Nat.mul(channels,Nat.mul(height,width)),U32.to_nat(upper)) == True{} : Bool},+weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,+parameters: Inputs.Parameters(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,0n)),+limits: Machine.Limits,+valid: {ConvSpec.valid(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image)) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image)) == True{} : Bool},+outer_word: U32,+after_word: U32,+lanes_word: U32,+groups_word: U32,+after_exact: {U32.to_nat(after_word) == Nat.mul(height,width) : Nat},+lanes_exact: {U32.to_nat(lanes_word) == channels : Nat},+groups_exact: {U32.to_nat(groups_word) == InputCells.group_count(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image)) : Nat},
  +room: {Nat.is_le(Nat.add(Nat.mul(image,InputCells.count(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),height)),InputCells.count(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),height)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Band.HeldImage(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,TiledCells.Cells{outer_word,after_word,lanes_word,groups_word,Grid.Innermost{}}},ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),Calls.image_contract(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters),Nat.mul(image,InputCells.count(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),Guards.height(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image))))):
  held_from(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,area_positive,area_room,weight_evidence,bias_evidence,parameters,limits,valid,supported,outer_word,after_word,lanes_word,groups_word,after_exact,lanes_exact,groups_exact,InputCells.count(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),height),0n,room,Order.reflexive(InputCells.count(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),height)))
