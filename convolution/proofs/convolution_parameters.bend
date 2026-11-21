# The weights and bias of a convolution as the call reads them: a tensor of
# the contract with its lanes on the output channels, its cells joined into
# one buffer, is a cell source of the outputs by the values after them that
# lists the values of the tensor.
import Base
import ../../lib/kernel_tile.bend as Tile
import ../../lib/kernel_gemm.bend as KernelGemm
import ../../lib/machine_limits.bend as Machine
import ../../lib/default_machine_limits.bend as Default
import ../../lib/storage_buffer.bend as Storage
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_grid.bend as Grid
import ../../lib/tensor_tiled.bend as Tiled
import ../specification.bend as ConvSpec
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/tensor_convolution.bend as TensorConvolution
import ./image_problems.bend as ImageProblems
import ../../lib/proofs/boolean_logic.bend as Boolean
import ./input_storage.bend as Inputs
import ../../lib/tensor_tiled_cells.bend as Cells
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_certificate.bend as Certificate
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/nat_ceiling.bend as Ceiling
import ../../lib/proofs/tensor_tiled_values.bend as TiledValues
import ../../lib/proofs/tensor_contract.bend as Contract
import ../../lib/proofs/tensor_pieces_model.bend as PiecesModel
import ../../lib/proofs/tensor_tiled_public.bend as Public
import ../../lib/proofs/tensor_tiled_representation.bend as Representation
import ../../lib/proofs/tensor_global_shape_count.bend as Global
import ../../lib/proofs/tensor_storage_count.bend as StorageCount
import ../../lib/proofs/tensor_joined.bend as Joined
import ../../lib/proofs/tensor_move_readback.bend as Held
import ../../lib/proofs/tensor_cell_source.bend as CellSource
import ../../lib/proofs/tensor_box_digits.bend as Box
import ../../lib/proofs/tensor_shape_refinement.bend as ShapeLaws
import ../../lib/proofs/machine_limit_bounds.bend as LimitBounds
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/nat_division_order.bend as DivisionOrder

# A list of the values of a reader has their count.
def listed_length(~Context: Data,~value: Context -> U32 -> F32,count: Nat,+offset: Nat,+context: Context) ->
  {ConvSpec.length(Contract.listed(~F32,~Context,~value,count,offset,context)) == count : Nat}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      Equal.cong(Nat,Nat,length => 1n+length,ConvSpec.length(Contract.listed(~F32,~Context,~value,rest,1n+offset,context)),rest,
        listed_length(~Context,~value,rest,1n+offset,context))

# The capacity of the machine fits half the words.
def capacity_word(+limits: Machine.Limits) -> {U32.is_le(Machine.capacity(limits),2147483648) == True{} : Bool}:
  Order.word_le(Machine.capacity(limits),1073741824,2147483648,LimitBounds.capacity_allocatable(limits),{==})

# The product of extents of a count within the machine is that count.
def product_exact(+limits: Machine.Limits,+extents: List<&2,U32>,+room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat(Grid.product(extents)) == ShapeLaws.elements(extents) : Nat}:
  %Box.product_from(extents) : {U32.to_nat(_) == ShapeLaws.elements(extents) : Nat}
  Bounds.roundtrip(ShapeLaws.elements(extents),Machine.capacity(limits),capacity_word(limits),room)

# The cells of lanes before extents after them, the groups at a position,
# count the groups by the values after the lanes.
def cells_count(position: Grid.Position,+lanes: U32,+after: List<&2,U32>) ->
  {ShapeLaws.elements(Tiled.cell_shape(lanes <> after,0n,position)) == Nat.mul(U32.to_nat(Grid.groups(lanes)),ShapeLaws.elements(after)) : Nat}:
  match position:
    case Grid.Outermost{}: {==}
    case Grid.Innermost{}:
      +groups = U32.to_nat(Grid.groups(lanes))
      +count = ShapeLaws.elements(after)
      Equal.trans(Nat,ShapeLaws.elements(List.append(&2,U32,after,[Grid.groups(lanes)])),Nat.mul(count,Nat.mul(groups,1n)),Nat.mul(groups,count),
        ShapeLaws.elements_appended(after,[Grid.groups(lanes)]),
        %Equal.sym(Nat,Nat.mul(groups,1n),groups,Order.times_one(groups)) : {Nat.mul(count,_) == Nat.mul(groups,count) : Nat} RowMajor.multiply_commutative(count,groups))

# The groups of the lanes by the values after them are at most the values.
def groups_fewer(+limits: Machine.Limits,+lanes: U32,+after: List<&2,U32>,+plane: Nat,+plane_exact: {ShapeLaws.elements(after) == plane : Nat},
  +groups: TiledValues.Groups<limits,lanes>) -> {Nat.is_le(Nat.mul(U32.to_nat(Grid.groups(lanes)),plane),ShapeLaws.elements(lanes <> after)) == True{} : Bool}:
  TiledValues.Groups{+exact,+fewer,+half} = groups
  %Equal.sym(Nat,ShapeLaws.elements(after),plane,plane_exact) : {Nat.is_le(Nat.mul(U32.to_nat(Grid.groups(lanes)),plane),Nat.mul(U32.to_nat(lanes),_)) == True{} : Bool}
  RowMajor.mul_le(U32.to_nat(Grid.groups(lanes)),U32.to_nat(lanes),plane,fewer)

# What the call reads of the joined cells of a parameter: the values of the
# reader they list, their length and its room in the allocation.
def Parameter(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+lanes: U32,+after: List<&2,U32>,+plane: Nat,+kept: Buffer.Model<Tile.Tile8>) -> Data:
  Held.Both<{Inputs.parameter_values(ConvSpec.length(Contract.listed(~F32,~Logical,~original,ShapeLaws.elements(lanes <> after),0n,logical)),U32.to_nat(lanes),plane,
        Buffer.tree(~Tile.Tile8,kept)) ==
      Contract.listed(~F32,~Logical,~original,ShapeLaws.elements(lanes <> after),0n,logical) : List<&2,F32>},
    Held.Both<{Buffer.length(~Tile.Tile8,kept) == Nat.mul(Nat.div(Nat.add(U32.to_nat(lanes),7n),8n),plane) : Nat},
      {Nat.is_le(Buffer.length(~Tile.Tile8,kept),U32.to_nat(U32.shln(1,Buffer.depth(~Tile.Tile8,kept)))) == True{} : Bool}>>

def parameter_listed(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+lanes: U32,+after: List<&2,U32>,+plane: Nat,+kept: Buffer.Model<Tile.Tile8>,
  +parameter: Parameter(~Logical,~original,logical,lanes,after,plane,kept)) ->
  {Inputs.parameter_values(ConvSpec.length(Contract.listed(~F32,~Logical,~original,ShapeLaws.elements(lanes <> after),0n,logical)),U32.to_nat(lanes),plane,
      Buffer.tree(~Tile.Tile8,kept)) ==
    Contract.listed(~F32,~Logical,~original,ShapeLaws.elements(lanes <> after),0n,logical) : List<&2,F32>}:
  Held.Both{+listed,+sizes} = parameter
  listed

def parameter_length(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+lanes: U32,+after: List<&2,U32>,+plane: Nat,+kept: Buffer.Model<Tile.Tile8>,
  +parameter: Parameter(~Logical,~original,logical,lanes,after,plane,kept)) ->
  {Buffer.length(~Tile.Tile8,kept) == Nat.mul(Nat.div(Nat.add(U32.to_nat(lanes),7n),8n),plane) : Nat}:
  Held.Both{+listed,Held.Both{+length_exact,+room}} = parameter
  length_exact

def parameter_room(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+lanes: U32,+after: List<&2,U32>,+plane: Nat,+kept: Buffer.Model<Tile.Tile8>,
  +parameter: Parameter(~Logical,~original,logical,lanes,after,plane,kept)) ->
  {Nat.is_le(Buffer.length(~Tile.Tile8,kept),U32.to_nat(U32.shln(1,Buffer.depth(~Tile.Tile8,kept)))) == True{} : Bool}:
  Held.Both{+listed,Held.Both{+length_exact,+room}} = parameter
  room

# The length of a buffer within the capacity of its tree is within the
# allocation of its depth.
def sized_room(+model: Buffer.Model<Tile.Tile8>,+sized: Buffer.Sized<Tile.Tile8,model>) ->
  {Nat.is_le(Buffer.length(~Tile.Tile8,model),U32.to_nat(U32.shln(1,Buffer.depth(~Tile.Tile8,model)))) == True{} : Bool}:
  Buffer.Sized{+depth_ok,+room} = sized
  %Shape.capacity(Tile.Tile8,Buffer.depth(~Tile.Tile8,model),Buffer.tree(~Tile.Tile8,model),Buffer.balanced(~Tile.Tile8,model)) :
    {Nat.is_le(Buffer.length(~Tile.Tile8,model),U32.to_nat(_)) == True{} : Bool}
  room

# The cells of lanes before extents after them, joined: whole storage of the
# shape of the cells, one cell for each group by the values after the lanes.
def shape_count(+lanes: U32,+after: List<&2,U32>,+position: Grid.Position,+plane: Nat,+shape: List<&2,U32>,
  +shaped: {Tiled.cell_shape(lanes <> after,0n,position) == shape : List<&2,U32>},+plane_exact: {ShapeLaws.elements(after) == plane : Nat}) ->
  {ShapeLaws.elements(shape) == Nat.mul(U32.to_nat(Grid.groups(lanes)),plane) : Nat}:
  Equal.trans(Nat,ShapeLaws.elements(shape),ShapeLaws.elements(Tiled.cell_shape(lanes <> after,0n,position)),Nat.mul(U32.to_nat(Grid.groups(lanes)),plane),
    %Equal.sym(List<&2,U32>,Tiled.cell_shape(lanes <> after,0n,position),shape,shaped) : {ShapeLaws.elements(shape) == ShapeLaws.elements(_) : Nat} {==},
    %plane_exact : {ShapeLaws.elements(Tiled.cell_shape(lanes <> after,0n,position)) == Nat.mul(U32.to_nat(Grid.groups(lanes)),_) : Nat} cells_count(position,lanes,after))

# The joined cells hold the groups of the lanes by the plane, within their
# allocation.
def kept_sized(+limits: Machine.Limits,+lanes: U32,+plane: Nat,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,+certificate: Certificate.Certificate<Tile.Tile8,Tree.pack(Tile.Tile8,tree)>,
  +shape: List<&2,U32>,+length_exact: {length == ShapeLaws.elements(shape) : Nat},+ready: Buffer.ContentReady<Tile.Tile8,Buffer.Content{tree,length}>,
  +shape_exact: {ShapeLaws.elements(shape) == Nat.mul(U32.to_nat(Grid.groups(lanes)),plane) : Nat},+groups: TiledValues.Groups<limits,lanes>) ->
  Held.Both<{Buffer.length(~Tile.Tile8,Buffer.Model{tree,length,certificate}) == Nat.mul(Nat.div(Nat.add(U32.to_nat(lanes),7n),8n),plane) : Nat},
    {Nat.is_le(Buffer.length(~Tile.Tile8,Buffer.Model{tree,length,certificate}),U32.to_nat(U32.shln(1,Buffer.depth(~Tile.Tile8,Buffer.Model{tree,length,certificate})))) == True{} : Bool}>:
  TiledValues.Groups{+exact,+fewer,+half} = groups
  +lanes_count = U32.to_nat(lanes)
  +model = {Buffer.Model{tree,length,certificate} : Buffer.Model<Tile.Tile8>}
  Held.Both{
    {%Equal.sym(Nat,Nat.div(Nat.add(lanes_count,7n),8n),Ceiling.ceiling(lanes_count,8n),Ceiling.shifted_formula(lanes_count,8n,{==})) : {length == Nat.mul(_,plane) : Nat}
      %exact : {length == Nat.mul(_,plane) : Nat}
      Equal.trans(Nat,length,ShapeLaws.elements(shape),Nat.mul(U32.to_nat(Grid.groups(lanes)),plane),length_exact,shape_exact) :
      {length == Nat.mul(Nat.div(Nat.add(lanes_count,7n),8n),plane) : Nat}},
    sized_room(model,Buffer.ready_content(~Tile.Tile8,model,ready))}

# The joined cells, a model of their tree, list the values of the reader.
def kept_parameter(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+lanes: U32,+after: List<&2,U32>,+position: Grid.Position,
  +plane: Nat,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,+certificate: Certificate.Certificate<Tile.Tile8,Tree.pack(Tile.Tile8,tree)>,+shape: List<&2,U32>,
  +formed: Contract.WellFormed<Tile.Tile8,shape,Buffer.Content{tree,length}>,
  +values: Contract.Values(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
    Representation.Lifted{logical,Public.geometry_of(lanes <> after,0n,position)},Buffer.Content{tree,length}),
  +shaped: {Tiled.cell_shape(lanes <> after,0n,position) == shape : List<&2,U32>},+counted: Global.Counted<lanes <> after>,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(lanes <> after)) == True{} : Bool},
  +plane_exact: {ShapeLaws.elements(after) == plane : Nat},+outer: CellSource.Outer(position,plane)) ->
  Parameter(~Logical,~original,logical,lanes,after,plane,Buffer.Model{tree,length,certificate}):
  Contract.WellFormed{+length_exact,+ready} = formed
  +count = ShapeLaws.elements(lanes <> after)
  +lanes_count = U32.to_nat(lanes)
  +room = StorageCount.counted_room(limits,lanes <> after,counted,default_capacity)
  +lanes_positive = RowMajor.left_factor_positive(lanes_count,ShapeLaws.elements(after),nonempty)
  +after_positive = DivisionOrder.right_factor_positive(lanes_count,ShapeLaws.elements(after),nonempty)
  +after_room = Order.transitive(ShapeLaws.elements(after),count,U32.to_nat(Machine.capacity(limits)),
    %RowMajor.multiply_commutative(ShapeLaws.elements(after),lanes_count) : {Nat.is_le(ShapeLaws.elements(after),_) == True{} : Bool}
      RowMajor.product_covers(ShapeLaws.elements(after),lanes_count,lanes_positive),room)
  +lanes_room = Order.transitive(lanes_count,count,U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(lanes_count,ShapeLaws.elements(after),after_positive),room)
  +word_exact = Equal.trans(Nat,U32.to_nat(Grid.product(after)),ShapeLaws.elements(after),plane,product_exact(limits,after,after_room),plane_exact)
  +shape_count = shape_count(lanes,after,position,plane,shape,shaped,plane_exact)
  +groups = TiledValues.grouped(limits,lanes,lanes_room)
  +cells_room = {%Equal.sym(Nat,length,Nat.mul(U32.to_nat(Grid.groups(lanes)),plane),Equal.trans(Nat,length,ShapeLaws.elements(shape),Nat.mul(U32.to_nat(Grid.groups(lanes)),plane),length_exact,shape_count)) :
    {Nat.is_le(Nat.mul(U32.to_nat(Grid.groups(lanes)),plane),_) == True{} : Bool} Order.reflexive(Nat.mul(U32.to_nat(Grid.groups(lanes)),plane)) :
    {Nat.is_le(Nat.mul(U32.to_nat(Grid.groups(lanes)),plane),length) == True{} : Bool}}
  +length_room = Equal.sym(Nat,length,ShapeLaws.elements(shape),length_exact)
  +planes = {%Public.one_times(Grid.product(after)) : Contract.Values(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
        Representation.Lifted{logical,Cells.Cells{_,Grid.product(after),lanes,Grid.groups(lanes),position}},Buffer.Content{tree,length}) values :
    Contract.Values(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,CellSource.Planes(Grid.product(after),lanes,position)},
      Buffer.Content{tree,length})}
  +listed = {%Equal.sym(Nat,ConvSpec.length(Contract.listed(~F32,~Logical,~original,count,0n,logical)),count,listed_length(~Logical,~original,count,0n,logical)) :
      {Inputs.parameter_values(_,lanes_count,plane,tree) == Contract.listed(~F32,~Logical,~original,count,0n,logical) : List<&2,F32>}
    CellSource.source_listed(~Logical,~original,logical,limits,Grid.product(after),lanes,position,plane,tree,length,planes,word_exact,outer,lanes_room,cells_room,
      Machine.capacity(limits),capacity_word(limits),
      %length_room : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
        Order.transitive(ShapeLaws.elements(shape),Nat.mul(U32.to_nat(Grid.groups(lanes)),plane),U32.to_nat(Machine.capacity(limits)),
          %Equal.sym(Nat,ShapeLaws.elements(shape),Nat.mul(U32.to_nat(Grid.groups(lanes)),plane),shape_count) : {Nat.is_le(_,Nat.mul(U32.to_nat(Grid.groups(lanes)),plane)) == True{} : Bool} Order.reflexive(Nat.mul(U32.to_nat(Grid.groups(lanes)),plane)),
          Order.transitive(Nat.mul(U32.to_nat(Grid.groups(lanes)),plane),count,U32.to_nat(Machine.capacity(limits)),groups_fewer(limits,lanes,after,plane,plane_exact,groups),room)),
      count,0n,%plane_exact : {Nat.is_le(count,Nat.mul(lanes_count,_)) == True{} : Bool} Order.reflexive(count)) :
    {Inputs.parameter_values(ConvSpec.length(Contract.listed(~F32,~Logical,~original,count,0n,logical)),lanes_count,plane,tree) ==
      Contract.listed(~F32,~Logical,~original,count,0n,logical) : List<&2,F32>}}
  Held.Both{listed,kept_sized(limits,lanes,plane,tree,length,certificate,shape,length_exact,ready,shape_count,groups)}

# The cells of a tensor of the contract with its lanes on the first axis,
# joined into one buffer, are a parameter that lists its values.
def joined_parameter(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+lanes: U32,+after: List<&2,U32>,+position: Grid.Position,
  +plane: Nat,+model: PiecesModel.Model<Tile.Tile8>,+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+cell_lanes: U32,+kept: Buffer.Model<Tile.Tile8>,
  +contract: Public.Contract(~Logical,~original,logical,lanes <> after,
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},0n,cell_lanes,position}),
  +equation: {Buffer.pack(~Tile.Tile8,kept) == Tensors.joined(~Tile.Tile8,Tensors.count_or_zero(shape),fill,PiecesModel.pack(~Tile.Tile8,model)) : Storage.Buffer<Tile.Tile8>},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(lanes <> after)) == True{} : Bool},
  +plane_exact: {ShapeLaws.elements(after) == plane : Nat},+outer: CellSource.Outer(position,plane)) ->
  Parameter(~Logical,~original,logical,lanes,after,plane,kept):
  match kept:
    case Buffer.Model{+tree,+length,+certificate}:
      KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}} = contract
      +reader = {Representation.Lifted{logical,Public.geometry_of(lanes <> after,0n,position)} : Representation.Lifted<Logical>}
      +facts = Joined.joined_facts(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),reader,model,shape,fill,workers,global,
        Buffer.Model{tree,length,certificate},equation)
      +formed = Held.first_of(Contract.WellFormed<Tile.Tile8,shape,Buffer.Content{tree,length}>,
        Contract.Values(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),reader,Buffer.Content{tree,length}),facts)
      +values = Held.second_of(Contract.WellFormed<Tile.Tile8,shape,Buffer.Content{tree,length}>,
        Contract.Values(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),reader,Buffer.Content{tree,length}),facts)
      kept_parameter(~Logical,~original,logical,limits,lanes,after,position,plane,tree,length,certificate,shape,formed,values,shaped,counted,default_capacity,nonempty,plane_exact,outer)

# The call accepts weights and bias of the lengths of their cells within
# their allocations for a valid problem.
def parameters_accepted(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,
  +padding: Nat,+weight_list: List<&2,F32>,+bias_list: List<&2,F32>,
  +valid: {ConvSpec.valid(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n)) == True{} : Bool},
  +weight_length: Nat,+weight_depth: Nat,+bias_length: Nat,+bias_depth: Nat,
  +weight_exact: {weight_length == Nat.mul(Nat.div(Nat.add(outputs,7n),8n),Nat.mul(channels,Nat.mul(kernel,kernel))) : Nat},
  +bias_exact: {bias_length == Nat.div(Nat.add(outputs,7n),8n) : Nat},
  +weight_room: {Nat.is_le(weight_length,U32.to_nat(U32.shln(1,weight_depth))) == True{} : Bool},
  +bias_room: {Nat.is_le(bias_length,U32.to_nat(U32.shln(1,bias_depth))) == True{} : Bool}) ->
  {TensorConvolution.parameters_valid(KernelShape.Dimensions{channels,outputs,height,width,kernel,stride,padding},weight_length,weight_depth,bias_length,bias_depth) == True{} : Bool}:
  +inner = Nat.mul(channels,Nat.mul(kernel,kernel))
  +weight_size = Nat.mul(outputs,inner)
  +area = Nat.mul(channels,Nat.mul(height,width))
  +groups = Nat.div(Nat.add(outputs,7n),8n)
  +prefix = Boolean.all([Nat.is_le(kernel,Nat.add(width,Nat.mul(2n,padding))),Nat.is_le(kernel,Nat.add(height,Nat.mul(2n,padding))),
    Nat.is_lt(0n,stride),Nat.is_lt(0n,kernel),Nat.is_lt(0n,width),Nat.is_lt(0n,height),Nat.is_lt(0n,outputs),Nat.is_lt(0n,channels)])
  +input_eq = Nat.is_eq(ConvSpec.length(ImageProblems.image_values(~Logical,~original,area,Nat.mul(0n,area),logical)),area)
  +weight_eq = Nat.is_eq(ConvSpec.length(weight_list),weight_size)
  +bias_eq = Nat.is_eq(ConvSpec.length(bias_list),outputs)
  +base = Boolean.left(prefix,input_eq,Boolean.left(prefix && input_eq,weight_eq,Boolean.left((prefix && input_eq) && weight_eq,bias_eq,valid)))
  +same_area = Nat.is_eq(area,area)
  +same_size = Nat.is_eq(weight_size,weight_size)
  +shape_valid = Boolean.and_true((prefix && same_area) && same_size,Nat.is_eq(outputs,outputs),
    Boolean.and_true(prefix && same_area,same_size,Boolean.and_true(prefix,same_area,base,Order.equal_reflexive(area)),Order.equal_reflexive(weight_size)),
    Order.equal_reflexive(outputs))
  +shape_condition = KernelShape.valid(KernelShape.Dimensions{channels,outputs,height,width,kernel,stride,padding},area,weight_size,outputs)
  +weights_condition = Nat.is_eq(weight_length,Nat.mul(groups,inner))
  +bias_condition = Nat.is_eq(bias_length,groups)
  +weights_fit = Nat.is_le(weight_length,U32.to_nat(U32.shln(1,weight_depth)))
  Boolean.and_true(((shape_condition && weights_condition) && bias_condition) && weights_fit,Nat.is_le(bias_length,U32.to_nat(U32.shln(1,bias_depth))),
    Boolean.and_true((shape_condition && weights_condition) && bias_condition,weights_fit,
      Boolean.and_true(shape_condition && weights_condition,bias_condition,
        Boolean.and_true(shape_condition,weights_condition,shape_valid,
          %Equal.sym(Nat,weight_length,Nat.mul(groups,inner),weight_exact) : {Nat.is_eq(_,Nat.mul(groups,inner)) == True{} : Bool} Order.equal_reflexive(Nat.mul(groups,inner))),
        %Equal.sym(Nat,bias_length,groups,bias_exact) : {Nat.is_eq(_,groups) == True{} : Bool} Order.equal_reflexive(groups)),
      weight_room),
    bias_room)
