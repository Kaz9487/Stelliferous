# Canonical scalar observations of the shared stencil packing and GEMM path.
import Base
import ./binary_radix.bend as Radix
import ./storage_readback.bend as TreeRead
import ./kernel_gemm_readback.bend as GemmReadback
import ./nat_division_order.bend as DivisionOrder
import ./u32_division.bend as U32Division
import ../kernel_rows.bend as Rows
import ../ordered_matrix.bend as OrderedMatrix
import ./kernel_pipeline_model.bend as Model
import ./kernel_accumulator.bend as Accumulator
import ./kernel_coordinate_refinement.bend as CoordinateLaws
import ./kernel_tile_refinement.bend as TileLaws
import ./kernel_reduction_model.bend as ReductionModel
import ./traversal_reduction_refinement.bend as ReductionLaws
import ./ordered_reduction_split.bend as Ordered
import ./ordered_ranges.bend as Ranges
import ./sequence_membership.bend as Members
import ./word_arithmetic.bend as WordArithmetic
import ../kernel_spatial_packing.bend as Packing
import ../kernel_tile.bend as Tile
import ../math_sequence.bend as Sequence
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./kernel_spatial_packing_refinement.bend as Packed
import ./kernel_spatial_packing_samples.bend as Samples
import ./kernel_packing_window_coordinates.bend as Coordinates
import ./kernel_packing_window_readback.bend as PackingReadback
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./row_major_index.bend as RowMajor
import ./natural_addition.bend as Addition

# A lane is the same scalar reader with its position advanced before decoding.
def shifted_sample(+input: Tree.Tree<F32>,+window: Packing.Stencil,+position: U32,+q: U32,+lane: Nat) ->
  {Samples.window_sample(input,Samples.window_context(window,position,q),lane) ==
    Samples.window_value(input,window,(position + U32.from_nat(lane) : U32),q) : F32}:
  Packing.Stencil{+base,+height,+width,+rows,+columns,+stride,+ro,+co,+ow,inner} = window
  %Equal.sym(F32,Samples.window_value(input,window,(position + U32.from_nat(lane) : U32),q),Samples.window_scalar(input,window,(position + U32.from_nat(lane) : U32),q),Samples.window_scalar_matches(input,window,(position + U32.from_nat(lane) : U32),q)) :
    {Samples.window_sample(input,Samples.window_context(window,position,q),lane) == _ : F32}
  {==}

def domain_room(+domain: PackingReadback.Domain) ->
  {Nat.is_le(PackingReadback.count(domain),U32.to_nat(Coordinates.capacity(PackingReadback.geometry(domain)))) == True{} : Bool}:
  PackingReadback.Domain{geometry,groups,start,packed_room,position_room} = domain
  packed_room

# Read the actual filled prefix, with no sample equality supplied by its caller.
def packed_scalar(+domain: PackingReadback.Domain,+input: Tree.Tree<F32>,+workspace: Tree.Tree<Tile.Tile8>,+depth: Nat,+capacity: U32,+group: Nat,+q: Nat,+lane: Sequence.Index(8n),
  +capacity_equal: {Tree.capacity(Tile.Tile8,workspace) == capacity : U32},+balanced: Shape.Balanced(Tile.Tile8,depth,workspace),
  +depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(PackingReadback.count(domain),U32.to_nat(capacity)) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,PackingReadback.groups(domain)) == True{} : Bool},+q_inside: {Nat.is_lt(q,Coordinates.inner(PackingReadback.geometry(domain))) == True{} : Bool}) ->
  {Tile.element_at(lane,Tree.read(Tile.Tile8,Packed.packed_output(Packed.packed_window(Coordinates.window(PackingReadback.geometry(domain)),PackingReadback.count(domain),PackingReadback.start(domain),input,workspace)),U32.from_nat(Nat.add(Nat.mul(group,Coordinates.inner(PackingReadback.geometry(domain))),q)))) ==
    Samples.window_value(input,Coordinates.window(PackingReadback.geometry(domain)),(U32.from_nat(Nat.add(U32.to_nat(PackingReadback.start(domain)),Nat.mul(group,8n))) + U32.from_nat(Sequence.index_value(8n,lane)) : U32),U32.from_nat(q)) : F32}:
  +geometry = PackingReadback.geometry(domain)
  +window = Coordinates.window(geometry)
  +index = Nat.add(Nat.mul(group,Coordinates.inner(geometry)),q)
  +position = Nat.add(U32.to_nat(PackingReadback.start(domain)),Nat.mul(group,8n))
  +selected = RowMajor.radix(group,PackingReadback.groups(domain),q,Coordinates.inner(geometry),group_inside,q_inside)
  +index_room = Order.transitive(index,PackingReadback.count(domain),U32.to_nat(Coordinates.capacity(geometry)),Bounds.strict_le(index,PackingReadback.count(domain),selected),domain_room(domain))
  %Equal.sym(Tile.Tile8,Tree.read(Tile.Tile8,Packed.packed_output(Packed.packed_window(window,PackingReadback.count(domain),PackingReadback.start(domain),input,workspace)),U32.from_nat(index)),PackingReadback.entry(PackingReadback.Tabulation{input,window,PackingReadback.start(domain)},U32.from_nat(index)),PackingReadback.packed_readback(domain,input,workspace,depth,capacity,index,capacity_equal,balanced,depth_ok,capacity_ok,room,selected)) :
    {Tile.element_at(lane,_) == Samples.window_value(input,window,(U32.from_nat(position) + U32.from_nat(Sequence.index_value(8n,lane)) : U32),U32.from_nat(q)) : F32}
  %Equal.sym(Tile.Tile8,PackingReadback.entry(PackingReadback.Tabulation{input,window,PackingReadback.start(domain)},U32.from_nat(index)),Samples.window_vector(input,window,U32.from_nat(position),U32.from_nat(q)),PackingReadback.entry_at(geometry,PackingReadback.start(domain),input,group,q,index,position,{==},{==},index_room,q_inside)) :
    {Tile.element_at(lane,_) == Samples.window_value(input,window,(U32.from_nat(position) + U32.from_nat(Sequence.index_value(8n,lane)) : U32),U32.from_nat(q)) : F32}
  Equal.trans(F32,Tile.element_at(lane,Samples.window_vector(input,window,U32.from_nat(position),U32.from_nat(q))),Samples.window_sample(input,Samples.window_context(window,U32.from_nat(position),U32.from_nat(q)),Sequence.index_value(8n,lane)),
    Samples.window_value(input,window,(U32.from_nat(position) + U32.from_nat(Sequence.index_value(8n,lane)) : U32),U32.from_nat(q)),PackingReadback.window_lane(input,window,U32.from_nat(position),U32.from_nat(q),lane),shifted_sample(input,window,U32.from_nat(position),U32.from_nat(q),Sequence.index_value(8n,lane)))

# All premises describe concrete geometry and storage. Sample equalities below
# are proved from the actual packing operation, never supplied as premises.
type Prepared<-upper: Bool> is Data:
  Prepared{domain: PackingReadback.Domain,input: Tree.Tree<F32>,workspace: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,
    depth: Nat,capacity: U32,capacity_equal: {Tree.capacity(Tile.Tile8,workspace) == capacity : U32},balanced: Shape.Balanced(Tile.Tile8,depth,workspace),
    depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},room: {Nat.is_le(PackingReadback.count(domain),U32.to_nat(capacity)) == True{} : Bool},
    group: Nat,channel: Nat,row: Accumulator.Row,lane: Sequence.Index(8n),group_inside: {Nat.is_lt(group,PackingReadback.groups(domain)) == True{} : Bool},
    weight_lane: {CoordinateLaws.weight_lane(upper,row) == TileLaws.lane_of(Nat.mod(channel,8n)) : Sequence.Index(8n)}}

def prepared_context(upper: Bool,prepared: Prepared<upper>) -> Model.WindowContext:
  Prepared{domain,input,workspace,weights,depth,capacity,capacity_equal,balanced,depth_ok,capacity_ok,room,group,channel,row,lane,group_inside,weight_lane} = prepared
  Model.WindowContext{Coordinates.window(PackingReadback.geometry(domain)),input,weights}

def prepared_position(upper: Bool,+prepared: Prepared<upper>) -> U32:
  Prepared{domain,input,workspace,weights,depth,capacity,capacity_equal,balanced,depth_ok,capacity_ok,room,group,channel,row,lane,group_inside,weight_lane} = prepared
  (U32.from_nat(Nat.add(U32.to_nat(PackingReadback.start(domain)),Nat.mul(group,8n))) + U32.from_nat(Sequence.index_value(8n,lane)) : U32)

def prepared_channel(upper: Bool,prepared: Prepared<upper>) -> Nat:
  Prepared{domain,input,workspace,weights,depth,capacity,capacity_equal,balanced,depth_ok,capacity_ok,room,group,channel,row,lane,group_inside,weight_lane} = prepared
  channel

def prepared_size(upper: Bool,prepared: Prepared<upper>) -> Nat:
  Prepared{domain,input,workspace,weights,depth,capacity,capacity_equal,balanced,depth_ok,capacity_ok,room,group,channel,row,lane,group_inside,weight_lane} = prepared
  Coordinates.inner(PackingReadback.geometry(domain))

def prepared_key(upper: Bool,prepared: Prepared<upper>) -> ReductionModel.RowLane:
  Prepared{domain,input,workspace,weights,depth,capacity,capacity_equal,balanced,depth_ok,capacity_ok,room,group,channel,row,lane,group_inside,weight_lane} = prepared
  ReductionModel.RowLane{row,lane}

def prepared_source(upper: Bool,+prepared: Prepared<upper>) -> Shape.Both<ReductionLaws.AddressLayout,ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>>:
  Prepared{+domain,input,workspace,weights,depth,capacity,capacity_equal,balanced,depth_ok,capacity_ok,room,group,+channel,row,lane,group_inside,weight_lane} = prepared
  +inner = Coordinates.inner(PackingReadback.geometry(domain))
  Shape.Both{ReductionLaws.AddressLayout{Nat.mul(group,inner),0n},
    ReductionLaws.SampleSource{Packed.packed_output(Packed.packed_window(Coordinates.window(PackingReadback.geometry(domain)),PackingReadback.count(domain),PackingReadback.start(domain),input,workspace)),weights,U32.from_nat(Nat.mul(Nat.div(channel,8n),inner))}}

def input_agrees(~upper: Bool,+prepared: Prepared<upper>,+q: Nat,inside: {Nat.is_lt(q,prepared_size(upper,prepared)) == True{} : Bool}) ->
  {ReductionModel.natural_input()(prepared_source(upper,prepared),prepared_key(upper,prepared),q) == Model.window_sample(prepared_context(upper,prepared),prepared_position(upper,prepared),q) : F32}:
  Prepared{+domain,+input,+workspace,+weights,+depth,+capacity,+capacity_equal,+balanced,+depth_ok,+capacity_ok,+room,+group,+channel,+row,+lane,+group_inside,+weight_lane} = prepared
  packed_scalar(domain,input,workspace,depth,capacity,group,q,lane,capacity_equal,balanced,depth_ok,capacity_ok,room,group_inside,inside)

def weight_agrees(~upper: Bool,+prepared: Prepared<upper>,+q: Nat) ->
  {ReductionModel.natural_weight(~upper)(prepared_source(upper,prepared),prepared_key(upper,prepared),q) == Model.window_weight(prepared_context(upper,prepared),prepared_channel(upper,prepared),q) : F32}:
  Prepared{+domain,+input,+workspace,+weights,depth,capacity,capacity_equal,balanced,depth_ok,capacity_ok,room,group,+channel,+row,lane,group_inside,+weight_lane} = prepared
  +inner = Coordinates.inner(PackingReadback.geometry(domain))
  +base = Nat.mul(Nat.div(channel,8n),inner)
  +address = (U32.from_nat(base) + U32.from_nat(q) : U32)
  %weight_lane :
    {ReductionModel.natural_weight(~upper)(prepared_source(upper,prepared),prepared_key(upper,prepared),q) == Tile.element_at(_,Tree.read(Tile.Tile8,weights,U32.from_nat(Nat.add(base,q)))) : F32}
  %WordArithmetic.native_add(base,q) :
    {ReductionModel.natural_weight(~upper)(prepared_source(upper,prepared),prepared_key(upper,prepared),q) == Tile.element_at(CoordinateLaws.weight_lane(upper,row),Tree.read(Tile.Tile8,weights,_)) : F32}
  CoordinateLaws.weight_sample(upper,row,Tree.read(Tile.Tile8,weights,address))

def Agreement(~upper: Bool,+prepared: Prepared<upper>,+q: Nat) -> Data:
  Shape.Both<
    {ReductionModel.natural_input()(prepared_source(upper,prepared),prepared_key(upper,prepared),q) == Model.window_sample(prepared_context(upper,prepared),prepared_position(upper,prepared),q) : F32},
    {ReductionModel.natural_weight(~upper)(prepared_source(upper,prepared),prepared_key(upper,prepared),q) == Model.window_weight(prepared_context(upper,prepared),prepared_channel(upper,prepared),q) : F32}>

def agreement(~upper: Bool,+prepared: Prepared<upper>,+q: Nat,+inside: {Nat.is_lt(q,prepared_size(upper,prepared)) == True{} : Bool}) -> Agreement(~upper,prepared,q):
  Shape.Both{input_agrees(~upper,prepared,q,inside),weight_agrees(~upper,prepared,q)}

def canonical_dot(~upper: Bool,+prepared: Prepared<upper>) ->
  {(OrderedMatrix.dot(~ReductionModel.RowLane,~Nat,~ReductionModel.RowLane,~Shape.Both<ReductionLaws.AddressLayout,ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>>,
      ~ReductionModel.natural_input(),~ReductionModel.natural_weight(~upper),Ranges.range(prepared_size(upper,prepared),0n),prepared_key(upper,prepared),prepared_key(upper,prepared),prepared_source(upper,prepared),0.0) + 0.0 : F32) ==
    Model.window_dot(prepared_context(upper,prepared),prepared_position(upper,prepared),prepared_channel(upper,prepared)) : F32}:
  Prepared{+domain,+input,+workspace,+weights,+depth,+capacity,+capacity_equal,+balanced,+depth_ok,+capacity_ok,+room,+group,+channel,+row,+lane,+group_inside,+weight_lane} = prepared
  Equal.cong(F32,F32,value => (value + 0.0 : F32),
    OrderedMatrix.dot(~ReductionModel.RowLane,~Nat,~ReductionModel.RowLane,~Shape.Both<ReductionLaws.AddressLayout,ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>>,
      ~ReductionModel.natural_input(),~ReductionModel.natural_weight(~upper),Ranges.range(prepared_size(upper,prepared),0n),prepared_key(upper,prepared),prepared_key(upper,prepared),prepared_source(upper,prepared),0.0),
    OrderedMatrix.dot(~Nat,~Nat,~U32,~Model.WindowContext,~Model.window_sample,~Model.window_weight,Ranges.range(prepared_size(upper,prepared),0n),prepared_channel(upper,prepared),prepared_position(upper,prepared),prepared_context(upper,prepared),0.0),
    Ordered.dot_congruence(~Nat,~ReductionModel.RowLane,~ReductionModel.RowLane,~Shape.Both<ReductionLaws.AddressLayout,ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>>,~Nat,~U32,~Model.WindowContext,
      ~ReductionModel.natural_input(),~ReductionModel.natural_weight(~upper),~Model.window_sample,~Model.window_weight,Ranges.range(prepared_size(upper,prepared),0n),prepared_key(upper,prepared),prepared_key(upper,prepared),prepared_source(upper,prepared),prepared_channel(upper,prepared),prepared_position(upper,prepared),prepared_context(upper,prepared),0.0,
      Members.range_all(~Prepared<upper>,~Agreement(~upper),~prepared_size(upper),~agreement(~upper),prepared_size(upper,prepared),0n,prepared,Order.reflexive(prepared_size(upper,prepared)))))

# The actual ordered accumulator uses these packed owners and cursor bases.
def ordered_value(~upper: Bool,+prepared: Prepared<upper>) -> F32:
  Prepared{+domain,input,workspace,weights,depth,capacity,capacity_equal,balanced,depth_ok,capacity_ok,room,group,+channel,row,lane,group_inside,weight_lane} = prepared
  +inner = Coordinates.inner(PackingReadback.geometry(domain))
  (ReductionModel.ordered_dot(~upper,ReductionModel.RowLane{row,lane},inner,
    ReductionModel.initial(Packed.packed_output(Packed.packed_window(Coordinates.window(PackingReadback.geometry(domain)),PackingReadback.count(domain),PackingReadback.start(domain),input,workspace)),weights,U32.from_nat(Nat.mul(group,inner)),0,U32.from_nat(Nat.mul(Nat.div(channel,8n),inner)))) + 0.0 : F32)

def ordered_value_matches(~upper: Bool,+prepared: Prepared<upper>) ->
  {ordered_value(~upper,prepared) == Model.window_dot(prepared_context(upper,prepared),prepared_position(upper,prepared),prepared_channel(upper,prepared)) : F32}:
  Prepared{+domain,+input,+workspace,+weights,+depth,+capacity,+capacity_equal,+balanced,+depth_ok,+capacity_ok,+room,+group,+channel,+row,+lane,+group_inside,+weight_lane} = prepared
  +inner = Coordinates.inner(PackingReadback.geometry(domain))
  +packed = Packed.packed_output(Packed.packed_window(Coordinates.window(PackingReadback.geometry(domain)),PackingReadback.count(domain),PackingReadback.start(domain),input,workspace))
  +key = prepared_key(upper,prepared)
  +initial = Rows.filled(~Tile.Tile8,Tile.filled(0.0))
  +natural = ReductionModel.encoded_dot(~upper,key,inner,packed,weights,Nat.mul(group,inner),0n,U32.from_nat(Nat.mul(Nat.div(channel,8n),inner)),initial)
  Equal.trans(F32,ordered_value(~upper,prepared),(natural + 0.0 : F32),Model.window_dot(prepared_context(upper,prepared),prepared_position(upper,prepared),channel),
    Equal.cong(F32,F32,value => (value + 0.0 : F32),
      ReductionModel.ordered_dot(~upper,key,inner,ReductionModel.initial(packed,weights,U32.from_nat(Nat.mul(group,inner)),0,U32.from_nat(Nat.mul(Nat.div(channel,8n),inner)))),natural,
      ReductionModel.encoded_as_natural(~upper,key,inner,packed,weights,Nat.mul(group,inner),0n,U32.from_nat(Nat.mul(Nat.div(channel,8n),inner)),initial)),
    %Equal.sym(F32,ReductionModel.accumulator_at(key,initial),0.0,ReductionModel.initial_zero(key)) :
      {(OrderedMatrix.dot(~ReductionModel.RowLane,~Nat,~ReductionModel.RowLane,~Shape.Both<ReductionLaws.AddressLayout,ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>>,
        ~ReductionModel.natural_input(),~ReductionModel.natural_weight(~upper),Ranges.range(inner,0n),key,key,prepared_source(upper,prepared),_) + 0.0 : F32) == Model.window_dot(prepared_context(upper,prepared),prepared_position(upper,prepared),channel) : F32}
    canonical_dot(~upper,prepared))

# Bounded native products denote the same shared natural address coordinates.
def multiply_pointer(+word: U32,+inner: U32,+index: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +exact: {U32.to_nat(word) == index : Nat},room: {Nat.is_le(Nat.mul(index,U32.to_nat(inner)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {(word * inner : U32) == U32.from_nat(Nat.mul(index,U32.to_nat(inner))) : U32}:
  +product_exact = {
    %exact : {U32.to_nat((word * inner : U32)) == Nat.mul(_,U32.to_nat(inner)) : Nat}
    Arithmetic.multiply(word,inner,capacity,capacity_ok,
      %Equal.sym(Nat,U32.to_nat(word),index,exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(inner)),U32.to_nat(capacity)) == True{} : Bool}
      room) : {U32.to_nat((word * inner : U32)) == Nat.mul(index,U32.to_nat(inner)) : Nat}}
  Equal.trans(U32,(word * inner : U32),U32.from_nat(U32.to_nat((word * inner : U32))),U32.from_nat(Nat.mul(index,U32.to_nat(inner))),
    Equal.sym(U32,U32.from_nat(U32.to_nat((word * inner : U32))),(word * inner : U32),Arithmetic.native_roundtrip((word * inner : U32))),
    Equal.cong(Nat,U32,U32.from_nat,U32.to_nat((word * inner : U32)),Nat.mul(index,U32.to_nat(inner)),product_exact))

def query_input_pointer(+query: GemmReadback.Query,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  room: {Nat.is_le(Nat.mul(GemmReadback.groups(query),U32.to_nat(GemmReadback.inner(query))),U32.to_nat(capacity)) == True{} : Bool}) ->
  {(GemmReadback.group_word(query,Nat.div(GemmReadback.channel(query),4n),Nat.div(GemmReadback.column(query),8n)) * GemmReadback.inner(query) : U32) == U32.from_nat(Nat.mul(Nat.div(GemmReadback.column(query),8n),U32.to_nat(GemmReadback.inner(query)))) : U32}:
  multiply_pointer(GemmReadback.group_word(query,Nat.div(GemmReadback.channel(query),4n),Nat.div(GemmReadback.column(query),8n)),GemmReadback.inner(query),Nat.div(GemmReadback.column(query),8n),capacity,capacity_ok,GemmReadback.selected_group_exact(query),
    Order.transitive(Nat.mul(Nat.div(GemmReadback.column(query),8n),U32.to_nat(GemmReadback.inner(query))),Nat.mul(GemmReadback.groups(query),U32.to_nat(GemmReadback.inner(query))),U32.to_nat(capacity),
      RowMajor.mul_le(Nat.div(GemmReadback.column(query),8n),GemmReadback.groups(query),U32.to_nat(GemmReadback.inner(query)),Bounds.strict_le(Nat.div(GemmReadback.column(query),8n),GemmReadback.groups(query),GemmReadback.selected_group_inside(query))),room))

def selected_unit_rows(+query: GemmReadback.Query) -> {Nat.is_le(Nat.div(GemmReadback.channel(query),4n),U32.to_nat(GemmReadback.rows(query))) == True{} : Bool}:
  Order.transitive(Nat.div(GemmReadback.channel(query),4n),GemmReadback.channel(query),U32.to_nat(GemmReadback.rows(query)),
    Order.transitive(Nat.div(GemmReadback.channel(query),4n),Nat.mul(Nat.div(GemmReadback.channel(query),4n),4n),GemmReadback.channel(query),RowMajor.product_covers(Nat.div(GemmReadback.channel(query),4n),4n,{==}),
      %Equal.sym(Nat,GemmReadback.channel(query),GemmReadback.row_number(Nat.div(GemmReadback.channel(query),4n),Nat.mod(GemmReadback.channel(query),4n)),GemmReadback.selected_channel_split(query)) : {Nat.is_le(Nat.mul(Nat.div(GemmReadback.channel(query),4n),4n),_) == True{} : Bool}
      Order.le_add(Nat.mul(Nat.div(GemmReadback.channel(query),4n),4n),Nat.mod(GemmReadback.channel(query),4n))),
    Bounds.strict_le(GemmReadback.channel(query),U32.to_nat(GemmReadback.rows(query)),GemmReadback.channel_inside(query)))

def query_weight_pointer(+query: GemmReadback.Query,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  room: {Nat.is_le(Nat.mul(U32.to_nat(GemmReadback.rows(query)),U32.to_nat(GemmReadback.inner(query))),U32.to_nat(capacity)) == True{} : Bool}) ->
  {(GemmReadback.unit_word(query,Nat.div(GemmReadback.channel(query),4n),Nat.div(GemmReadback.column(query),8n)) / 2 * GemmReadback.inner(query) : U32) == U32.from_nat(Nat.mul(Nat.div(GemmReadback.channel(query),8n),U32.to_nat(GemmReadback.inner(query)))) : U32}:
  +word = GemmReadback.unit_word(query,Nat.div(GemmReadback.channel(query),4n),Nat.div(GemmReadback.column(query),8n))
  +unit = Nat.div(GemmReadback.channel(query),4n)
  +block = Nat.div(unit,2n)
  +half_exact = {
    %GemmReadback.selected_unit_exact(query) : {U32.to_nat((word / 2 : U32)) == Nat.div(_,2n) : Nat}
    U32Division.divide(word,2,2147483648,{==},{==},{==}) : {U32.to_nat((word / 2 : U32)) == block : Nat}}
  %Equal.sym(Nat,Nat.div(GemmReadback.channel(query),8n),block,GemmReadback.selected_block_exact(query)) :
    {(word / 2 * GemmReadback.inner(query) : U32) == U32.from_nat(Nat.mul(_,U32.to_nat(GemmReadback.inner(query)))) : U32}
  multiply_pointer((word / 2 : U32),GemmReadback.inner(query),block,capacity,capacity_ok,half_exact,
    Order.transitive(Nat.mul(block,U32.to_nat(GemmReadback.inner(query))),Nat.mul(U32.to_nat(GemmReadback.rows(query)),U32.to_nat(GemmReadback.inner(query))),U32.to_nat(capacity),
      RowMajor.mul_le(block,U32.to_nat(GemmReadback.rows(query)),U32.to_nat(GemmReadback.inner(query)),
        Order.transitive(block,unit,U32.to_nat(GemmReadback.rows(query)),Order.transitive(block,Nat.mul(block,2n),unit,RowMajor.product_covers(block,2n,{==}),DivisionOrder.product_below(unit,2n,{==})),selected_unit_rows(query))),room))

# The query is built from the actual packed prefix. Its bounds are geometry only.
type JobGeometry is Data:
  JobGeometry{domain: PackingReadback.Domain,rows: U32,span: U32,channel: Nat,column: Nat,
    span_exact: {U32.to_nat(span) == Nat.mul(PackingReadback.groups(domain),8n) : Nat},
    rows_room: {Nat.is_le(Nat.add(U32.to_nat(rows),3n),U32.to_nat(Coordinates.capacity(PackingReadback.geometry(domain)))) == True{} : Bool},
    area_room: {Nat.is_le(Nat.mul(U32.to_nat(rows),U32.to_nat(span)),U32.to_nat(Coordinates.capacity(PackingReadback.geometry(domain)))) == True{} : Bool},
    weights_room: {Nat.is_le(Nat.mul(U32.to_nat(rows),Coordinates.inner(PackingReadback.geometry(domain))),U32.to_nat(Coordinates.capacity(PackingReadback.geometry(domain)))) == True{} : Bool},
    channel_inside: {Nat.is_lt(channel,U32.to_nat(rows)) == True{} : Bool},column_inside: {Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool}}

def coordinates_ok(+geometry: Coordinates.Geometry) -> {U32.is_le(Coordinates.capacity(geometry),2147483648) == True{} : Bool}:
  Coordinates.Geometry{window,channels,capacity,word_ok,rp,cp,ip,wp,ie,ce,rr,cr,ar,ir,wr} = geometry
  word_ok

def job_domain(geometry: JobGeometry) -> PackingReadback.Domain:
  JobGeometry{domain,rows,span,channel,column,span_exact,rows_room,area_room,weights_room,channel_inside,column_inside} = geometry
  domain

def job_channel(geometry: JobGeometry) -> Nat:
  JobGeometry{domain,rows,span,channel,column,span_exact,rows_room,area_room,weights_room,channel_inside,column_inside} = geometry
  channel

def job_column(geometry: JobGeometry) -> Nat:
  JobGeometry{domain,rows,span,channel,column,span_exact,rows_room,area_room,weights_room,channel_inside,column_inside} = geometry
  column

def job_rows(geometry: JobGeometry) -> U32:
  JobGeometry{domain,rows,span,channel,column,span_exact,rows_room,area_room,weights_room,channel_inside,column_inside} = geometry
  rows

def job_span(geometry: JobGeometry) -> U32:
  JobGeometry{domain,rows,span,channel,column,span_exact,rows_room,area_room,weights_room,channel_inside,column_inside} = geometry
  span

def job_span_exact(+geometry: JobGeometry) -> {U32.to_nat(job_span(geometry)) == Nat.mul(PackingReadback.groups(job_domain(geometry)),8n) : Nat}:
  JobGeometry{domain,rows,span,channel,column,span_exact,rows_room,area_room,weights_room,channel_inside,column_inside} = geometry
  span_exact

def job_rows_room(+geometry: JobGeometry) -> {Nat.is_le(Nat.add(U32.to_nat(job_rows(geometry)),3n),U32.to_nat(Coordinates.capacity(PackingReadback.geometry(job_domain(geometry))))) == True{} : Bool}:
  JobGeometry{domain,rows,span,channel,column,span_exact,rows_room,area_room,weights_room,channel_inside,column_inside} = geometry
  rows_room

def job_area_room(+geometry: JobGeometry) -> {Nat.is_le(Nat.mul(U32.to_nat(job_rows(geometry)),U32.to_nat(job_span(geometry))),U32.to_nat(Coordinates.capacity(PackingReadback.geometry(job_domain(geometry))))) == True{} : Bool}:
  JobGeometry{domain,rows,span,channel,column,span_exact,rows_room,area_room,weights_room,channel_inside,column_inside} = geometry
  area_room

def job_channel_inside(+geometry: JobGeometry) -> {Nat.is_lt(job_channel(geometry),U32.to_nat(job_rows(geometry))) == True{} : Bool}:
  JobGeometry{domain,rows,span,channel,column,span_exact,rows_room,area_room,weights_room,channel_inside,column_inside} = geometry
  channel_inside

def job_column_inside(+geometry: JobGeometry) -> {Nat.is_lt(job_column(geometry),U32.to_nat(job_span(geometry))) == True{} : Bool}:
  JobGeometry{domain,rows,span,channel,column,span_exact,rows_room,area_room,weights_room,channel_inside,column_inside} = geometry
  column_inside

def query(+geometry: JobGeometry,input: Tree.Tree<F32>,workspace: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,bias_depth: Nat) -> GemmReadback.Query:
  JobGeometry{+domain,rows,span,channel,column,span_exact,rows_room,area_room,weights_room,channel_inside,column_inside} = geometry
  +coordinates = PackingReadback.geometry(domain)
  GemmReadback.Query{rows,span,Coordinates.inner_word(coordinates),PackingReadback.groups(domain),Coordinates.capacity(coordinates),
    Packed.packed_output(Packed.packed_window(Coordinates.window(coordinates),PackingReadback.count(domain),PackingReadback.start(domain),input,workspace)),weights,Tree.fresh(F32,bias_depth,0.0),channel,column,
    coordinates_ok(coordinates),span_exact,rows_room,area_room,channel_inside,column_inside}

def selected_upper(+query: GemmReadback.Query) -> Bool:
  U32.is_eq((GemmReadback.unit_word(query,Nat.div(GemmReadback.channel(query),4n),Nat.div(GemmReadback.column(query),8n)) % 2 : U32),1)

def prepared(+geometry: JobGeometry,+input: Tree.Tree<F32>,+workspace: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias_depth: Nat,+depth: Nat,+capacity: U32,
  capacity_equal: {Tree.capacity(Tile.Tile8,workspace) == capacity : U32},balanced: Shape.Balanced(Tile.Tile8,depth,workspace),depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  room: {Nat.is_le(PackingReadback.count(job_domain(geometry)),U32.to_nat(capacity)) == True{} : Bool}) -> Prepared<selected_upper(query(geometry,input,workspace,weights,bias_depth))>:
  JobGeometry{+domain,rows,span,+channel,+column,span_exact,rows_room,area_room,weights_room,channel_inside,column_inside} = geometry
  +q = query(geometry,input,workspace,weights,bias_depth)
  Prepared{domain,input,workspace,weights,depth,capacity,capacity_equal,balanced,depth_ok,capacity_ok,room,Nat.div(column,8n),channel,Accumulator.row_of(Nat.mod(channel,4n)),TileLaws.lane_of(Nat.mod(column,8n)),GemmReadback.selected_group_inside(q),GemmReadback.selected_weight_lane(q)}

def selected_value(upper: Bool,prepared: Prepared<upper>) -> F32:
  match upper:
    case False{}: ordered_value(~False{},prepared)
    case True{}: ordered_value(~True{},prepared)

def selected_value_matches(+upper: Bool,+prepared: Prepared<upper>) ->
  {selected_value(upper,prepared) == Model.window_dot(prepared_context(upper,prepared),prepared_position(upper,prepared),prepared_channel(upper,prepared)) : F32}:
  match upper:
    case False{}: ordered_value_matches(~False{},prepared)
    case True{}: ordered_value_matches(~True{},prepared)

# Column decoding changes only word addresses, preserving every FP32 sample.
def logical_position(+origin: U32,+column: Nat,+lane_small: {Nat.is_lt(Nat.mod(column,8n),8n) == True{} : Bool}) ->
  {(U32.from_nat(Nat.add(U32.to_nat(origin),Nat.mul(Nat.div(column,8n),8n))) + U32.from_nat(Sequence.index_value(8n,TileLaws.lane_of(Nat.mod(column,8n)))) : U32) == (origin + U32.from_nat(column) : U32) : U32}:
  %Equal.sym(Nat,Sequence.index_value(8n,TileLaws.lane_of(Nat.mod(column,8n))),Nat.mod(column,8n),GemmReadback.lane_value_of(Nat.mod(column,8n),lane_small)) :
    {(U32.from_nat(Nat.add(U32.to_nat(origin),Nat.mul(Nat.div(column,8n),8n))) + U32.from_nat(_) : U32) == (origin + U32.from_nat(column) : U32) : U32}
  %Equal.sym(U32,(U32.from_nat(Nat.add(U32.to_nat(origin),Nat.mul(Nat.div(column,8n),8n))) + U32.from_nat(Nat.mod(column,8n)) : U32),U32.from_nat(Nat.add(Nat.add(U32.to_nat(origin),Nat.mul(Nat.div(column,8n),8n)),Nat.mod(column,8n))),WordArithmetic.native_add(Nat.add(U32.to_nat(origin),Nat.mul(Nat.div(column,8n),8n)),Nat.mod(column,8n))) :
    {_ == (origin + U32.from_nat(column) : U32) : U32}
  %Equal.sym(Nat,Nat.add(Nat.add(U32.to_nat(origin),Nat.mul(Nat.div(column,8n),8n)),Nat.mod(column,8n)),Nat.add(U32.to_nat(origin),Nat.add(Nat.mul(Nat.div(column,8n),8n),Nat.mod(column,8n))),Addition.associative(U32.to_nat(origin),Nat.mul(Nat.div(column,8n),8n),Nat.mod(column,8n))) :
    {U32.from_nat(_) == (origin + U32.from_nat(column) : U32) : U32}
  %DivisionOrder.quotient_exact(column,8n,{==}) : {U32.from_nat(Nat.add(U32.to_nat(origin),_)) == (origin + U32.from_nat(column) : U32) : U32}
  %Arithmetic.native_roundtrip(origin) : {U32.from_nat(Nat.add(U32.to_nat(origin),column)) == (_ + U32.from_nat(column) : U32) : U32}
  Equal.sym(U32,(U32.from_nat(U32.to_nat(origin)) + U32.from_nat(column) : U32),U32.from_nat(Nat.add(U32.to_nat(origin),column)),WordArithmetic.native_add(U32.to_nat(origin),column))

def prepared_state(upper: Bool,+prepared: Prepared<upper>) -> ReductionModel.State():
  Prepared{+domain,input,workspace,weights,depth,capacity,capacity_equal,balanced,depth_ok,capacity_ok,room,group,+channel,row,lane,group_inside,weight_lane} = prepared
  +inner = Coordinates.inner(PackingReadback.geometry(domain))
  ReductionModel.initial(Packed.packed_output(Packed.packed_window(Coordinates.window(PackingReadback.geometry(domain)),PackingReadback.count(domain),PackingReadback.start(domain),input,workspace)),weights,U32.from_nat(Nat.mul(group,inner)),0,U32.from_nat(Nat.mul(Nat.div(channel,8n),inner)))

def selected_value_unfold(+upper: Bool,+prepared: Prepared<upper>) ->
  {selected_value(upper,prepared) == (ReductionModel.ordered_selected(upper,prepared_key(upper,prepared),prepared_size(upper,prepared),prepared_state(upper,prepared)) + 0.0 : F32) : F32}:
  match upper prepared:
    case False{} Prepared{domain,input,workspace,weights,depth,capacity,capacity_equal,balanced,depth_ok,capacity_ok,room,group,channel,row,lane,group_inside,weight_lane}: {==}
    case True{} Prepared{domain,input,workspace,weights,depth,capacity,capacity_equal,balanced,depth_ok,capacity_ok,room,group,channel,row,lane,group_inside,weight_lane}: {==}

def query_selected_value(+geometry: JobGeometry,+input: Tree.Tree<F32>,+workspace: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias_depth: Nat,+depth: Nat,+capacity: U32,
  +capacity_equal: {Tree.capacity(Tile.Tile8,workspace) == capacity : U32},+balanced: Shape.Balanced(Tile.Tile8,depth,workspace),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(PackingReadback.count(job_domain(geometry)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {GemmReadback.wanted_ordered(~True{},query(geometry,input,workspace,weights,bias_depth)) == selected_value(selected_upper(query(geometry,input,workspace,weights,bias_depth)),prepared(geometry,input,workspace,weights,bias_depth,depth,capacity,capacity_equal,balanced,depth_ok,capacity_ok,room)) : F32}:
  JobGeometry{+domain,+rows,+span,+channel,+column,+span_exact,+rows_room,+area_room,+weights_room,+channel_inside,+column_inside} = geometry
  +coordinates = PackingReadback.geometry(domain)
  +inner = Coordinates.inner(coordinates)
  +q = query(geometry,input,workspace,weights,bias_depth)
  +p = prepared(geometry,input,workspace,weights,bias_depth,depth,capacity,capacity_equal,balanced,depth_ok,capacity_ok,room)
  +upper = selected_upper(q)
  +key = prepared_key(upper,p)
  +packed = Packed.packed_output(Packed.packed_window(Coordinates.window(coordinates),PackingReadback.count(domain),PackingReadback.start(domain),input,workspace))
  +input_word = (GemmReadback.group_word(q,Nat.div(channel,4n),Nat.div(column,8n)) * Coordinates.inner_word(coordinates) : U32)
  +weight_word = (GemmReadback.unit_word(q,Nat.div(channel,4n),Nat.div(column,8n)) / 2 * Coordinates.inner_word(coordinates) : U32)
  +input_exact = query_input_pointer(q,Coordinates.capacity(coordinates),coordinates_ok(coordinates),domain_room(domain))
  +weight_exact = query_weight_pointer(q,Coordinates.capacity(coordinates),coordinates_ok(coordinates),weights_room)
  %Equal.sym(F32,selected_value(upper,p),(ReductionModel.ordered_selected(upper,key,inner,prepared_state(upper,p)) + 0.0 : F32),selected_value_unfold(upper,p)) :
    {GemmReadback.wanted_ordered(~True{},q) == _ : F32}
  %Equal.sym(U32,input_word,U32.from_nat(Nat.mul(Nat.div(column,8n),inner)),input_exact) :
    {(ReductionModel.ordered_selected(upper,key,inner,ReductionModel.initial(packed,weights,_,0,weight_word)) + Tree.read(F32,Tree.fresh(F32,bias_depth,0.0),U32.from_nat(channel)) : F32) ==
      (ReductionModel.ordered_selected(upper,key,inner,prepared_state(upper,p)) + 0.0 : F32) : F32}
  %Equal.sym(U32,weight_word,U32.from_nat(Nat.mul(Nat.div(channel,8n),inner)),weight_exact) :
    {(ReductionModel.ordered_selected(upper,key,inner,ReductionModel.initial(packed,weights,U32.from_nat(Nat.mul(Nat.div(column,8n),inner)),0,_)) + Tree.read(F32,Tree.fresh(F32,bias_depth,0.0),U32.from_nat(channel)) : F32) ==
      (ReductionModel.ordered_selected(upper,key,inner,prepared_state(upper,p)) + 0.0 : F32) : F32}
  %Equal.sym(F32,Tree.read(F32,Tree.fresh(F32,bias_depth,0.0),U32.from_nat(channel)),0.0,TreeRead.read_fresh(~F32,bias_depth,0.0,U32.from_nat(channel))) :
    {(ReductionModel.ordered_selected(upper,key,inner,prepared_state(upper,p)) + _ : F32) == (ReductionModel.ordered_selected(upper,key,inner,prepared_state(upper,p)) + 0.0 : F32) : F32}
  {==}

def packed_count_exact(+domain: PackingReadback.Domain,+span: U32,+span_exact: {U32.to_nat(span) == Nat.mul(PackingReadback.groups(domain),8n) : Nat}) ->
  {U32.to_nat((span / 8 * Coordinates.inner_word(PackingReadback.geometry(domain)) : U32)) == PackingReadback.count(domain) : Nat}:
  +geometry = PackingReadback.geometry(domain)
  +groups = PackingReadback.groups(domain)
  +groups_exact = {Equal.trans(Nat,U32.to_nat((span / 8 : U32)),Nat.div(U32.to_nat(span),8n),groups,
    U32Division.divide(span,8,2147483648,{==},{==},{==}),
    %Equal.sym(Nat,U32.to_nat(span),Nat.mul(groups,8n),span_exact) : {Nat.div(_,8n) == groups : Nat}
    %Addition.zero_right(Nat.mul(groups,8n)) : {Nat.div(_,8n) == groups : Nat}
    Radix.quotient(groups,0n,8n,{==},{==})) : {U32.to_nat((span / 8 : U32)) == groups : Nat}}
  %groups_exact : {U32.to_nat((span / 8 * Coordinates.inner_word(geometry) : U32)) == Nat.mul(_,Coordinates.inner(geometry)) : Nat}
  Arithmetic.multiply((span / 8 : U32),Coordinates.inner_word(geometry),Coordinates.capacity(geometry),coordinates_ok(geometry),
    %Equal.sym(Nat,U32.to_nat((span / 8 : U32)),groups,groups_exact) : {Nat.is_le(Nat.mul(_,Coordinates.inner(geometry)),U32.to_nat(Coordinates.capacity(geometry))) == True{} : Bool}
    domain_room(domain))

def query_matches_native(+geometry: JobGeometry,+input: Tree.Tree<F32>,+workspace: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias_depth: Nat) ->
  {query(geometry,input,workspace,weights,bias_depth) ==
    Model.step_gemm_query(job_rows(geometry),job_span(geometry),Coordinates.window(PackingReadback.geometry(job_domain(geometry))),PackingReadback.start(job_domain(geometry)),input,workspace,weights,Tree.fresh(F32,bias_depth,0.0),job_channel(geometry),job_column(geometry),PackingReadback.groups(job_domain(geometry)),Coordinates.capacity(PackingReadback.geometry(job_domain(geometry))),coordinates_ok(PackingReadback.geometry(job_domain(geometry))),job_span_exact(geometry),job_rows_room(geometry),job_area_room(geometry),job_channel_inside(geometry),job_column_inside(geometry)) : GemmReadback.Query}:
  JobGeometry{+domain,+rows,+span,+channel,+column,+span_exact,+rows_room,+area_room,+weights_room,+channel_inside,+column_inside} = geometry
  PackingReadback.Domain{+coordinates,+groups,+start,+packed_room,+position_room} = domain
  Coordinates.Geometry{+window,+channels,+word_capacity,+word_ok,+rp,+cp,+ip,+wp,+ie,+ce,+rr,+cr,+ar,+ir,+wr} = coordinates
  Packing.Stencil{+base,+height,+width,+tap_rows,+tap_columns,+stride,+row_origin,+column_origin,+output_width,+inner} = window
  %Equal.sym(Nat,U32.to_nat((span / 8 * inner : U32)),PackingReadback.count(domain),packed_count_exact(domain,span,span_exact)) :
    {query(geometry,input,workspace,weights,bias_depth) == GemmReadback.Query{rows,span,inner,groups,word_capacity,Packed.packed_output(Packed.packed_window(window,_,start,input,workspace)),weights,Tree.fresh(F32,bias_depth,0.0),channel,column,word_ok,span_exact,rows_room,area_room,channel_inside,column_inside} : GemmReadback.Query}
  {==}

def wanted_matches(+geometry: JobGeometry,+input: Tree.Tree<F32>,+workspace: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias_depth: Nat,+depth: Nat,+capacity: U32,
  +capacity_equal: {Tree.capacity(Tile.Tile8,workspace) == capacity : U32},+balanced: Shape.Balanced(Tile.Tile8,depth,workspace),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(PackingReadback.count(job_domain(geometry)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {GemmReadback.wanted(~True{},query(geometry,input,workspace,weights,bias_depth)) ==
    Model.window_dot(Model.WindowContext{Coordinates.window(PackingReadback.geometry(job_domain(geometry))),input,weights},(PackingReadback.start(job_domain(geometry)) + U32.from_nat(job_column(geometry)) : U32),job_channel(geometry)) : F32}:
  JobGeometry{+domain,+rows,+span,+channel,+column,+span_exact,+rows_room,+area_room,+weights_room,+channel_inside,+column_inside} = geometry
  +q = query(geometry,input,workspace,weights,bias_depth)
  +p = prepared(geometry,input,workspace,weights,bias_depth,depth,capacity,capacity_equal,balanced,depth_ok,capacity_ok,room)
  +upper = selected_upper(q)
  +target = Model.window_dot(Model.WindowContext{Coordinates.window(PackingReadback.geometry(domain)),input,weights},(PackingReadback.start(domain) + U32.from_nat(column) : U32),channel)
  Equal.trans(F32,GemmReadback.wanted(~True{},q),GemmReadback.wanted_ordered(~True{},q),target,GemmReadback.wanted_as_ordered(~True{},q),
    Equal.trans(F32,GemmReadback.wanted_ordered(~True{},q),selected_value(upper,p),target,query_selected_value(geometry,input,workspace,weights,bias_depth,depth,capacity,capacity_equal,balanced,depth_ok,capacity_ok,room),
      %logical_position(PackingReadback.start(domain),column,GemmReadback.selected_lane_small(q)) :
        {selected_value(upper,p) == Model.window_dot(Model.WindowContext{Coordinates.window(PackingReadback.geometry(domain)),input,weights},_,channel) : F32}
      selected_value_matches(upper,p)))
