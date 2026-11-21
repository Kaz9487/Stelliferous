# A copied output cell preserves each valid channel's matrix value.
import Base
import ../specification.bend as ConvSpec
import ../inference.bend as NativeConv
import ../../lib/machine_limits.bend as Machine
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/proofs/kernel_window_readback.bend as Copy
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/binary_radix.bend as Radix
import ../../lib/proofs/natural_addition.bend as Addition
import ./input_storage.bend as Inputs
import ./convolution_guard_facts.bend as Guards
import ./supported_padded_coordinates.bend as Padded
import ./reduction_coordinate_bounds.bend as Reduction
import ./array_leaf_cells.bend as ArrayLeaf
import ./microkernel_leaf_readback.bend as MicroLeaf
import ./output_values.bend as Values
import ./cell_gather_coordinates.bend as Coordinates
import ./concrete_support_guard.bend as Support
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/nat_division_order.bend as DivisionOrder

def lane(lane: Nat,cell: Tile.Tile8) -> F32:
  TileLaws.select_lane(lane,cell)

def source_address(+group: Nat,+span: U32,+column: Nat) ->
  {(U32.from_nat(group) * span + 0 + U32.from_nat(column) : U32) == U32.from_nat(Nat.add(Nat.mul(group,U32.to_nat(span)),column)) : U32}:
  Equal.trans(U32,(U32.from_nat(group) * span + 0 + U32.from_nat(column) : U32),U32.from_nat(Nat.add(Nat.add(Nat.mul(group,U32.to_nat(span)),0n),column)),
    U32.from_nat(Nat.add(Nat.mul(group,U32.to_nat(span)),column)),Copy.plane_address(group,span,0,column),
    Equal.cong(Nat,U32,base => U32.from_nat(Nat.add(base,column)),Nat.add(Nat.mul(group,U32.to_nat(span)),0n),Nat.mul(group,U32.to_nat(span)),Addition.zero_right(Nat.mul(group,U32.to_nat(span)))))

# Cell `cell` of the run of a part, past the cells before its offset, lies among
# the cells of the output.
def reach(+limits: Machine.Limits,+problem: ConvSpec.Problem,+offset: U32,+cell: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +offset_bound: {Nat.is_le(U32.to_nat(offset),Support.padded(problem)) == True{} : Bool},
  +point_inside: {Nat.is_lt(Nat.add(U32.to_nat((offset * Coordinates.groups(problem) : U32)),cell),U32.to_nat((Padded.spatial(problem) * Coordinates.groups(problem) : U32))) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(Nat.mul(U32.to_nat(offset),U32.to_nat(Coordinates.groups(problem))),cell),
    Nat.mul(U32.to_nat(Padded.spatial(problem)),U32.to_nat(Coordinates.groups(problem)))) == True{} : Bool}:
  +g = U32.to_nat(Coordinates.groups(problem))
  %Coordinates.scaled(limits,problem,offset,supported,offset_bound) : {Nat.is_lt(Nat.add(_,cell),Nat.mul(U32.to_nat(Padded.spatial(problem)),g)) == True{} : Bool}
  %Coordinates.scaled(limits,problem,Padded.spatial(problem),supported,Coordinates.spatial_bound(limits,problem,valid,supported)) :
    {Nat.is_lt(Nat.add(U32.to_nat((offset * Coordinates.groups(problem) : U32)),cell),_) == True{} : Bool}
  point_inside

# The cells of the positions before a local position, then a group.
def regrouped(+offset: Nat,+local: Nat,+groups: Nat,+group: Nat) ->
  {Nat.add(Nat.mul(Nat.add(offset,local),groups),group) == Nat.add(Nat.mul(offset,groups),Nat.add(Nat.mul(local,groups),group)) : Nat}:
  Equal.trans(Nat,Nat.add(Nat.mul(Nat.add(offset,local),groups),group),Nat.add(Nat.add(Nat.mul(offset,groups),Nat.mul(local,groups)),group),
    Nat.add(Nat.mul(offset,groups),Nat.add(Nat.mul(local,groups),group)),
    Equal.cong(Nat,Nat,value => Nat.add(value,group),Nat.mul(Nat.add(offset,local),groups),Nat.add(Nat.mul(offset,groups),Nat.mul(local,groups)),RowMajor.distrib(offset,local,groups)),
    Addition.associative(Nat.mul(offset,groups),Nat.mul(local,groups),group))

# A cell a gather copies to the query: its position is the query position and
# its group the group of the channel, so it holds the matrix value.
def copied_value(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+offset: U32,+span: U32,
  +channel: Nat,+row: Nat,+column: Nat,+cell: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +certificate: ArrayLeaf.Geometry<problem,offset,span>,
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool},
  +in_span: {Nat.is_lt(cell,U32.to_nat((span * Coordinates.groups(problem) : U32))) == True{} : Bool},
  +point_inside: {Nat.is_lt(Nat.add(U32.to_nat((offset * Coordinates.groups(problem) : U32)),cell),U32.to_nat((Padded.spatial(problem) * Coordinates.groups(problem) : U32))) == True{} : Bool},
  +same: {U32.is_eq((U32.from_nat(0n) * (Padded.spatial(problem) * Coordinates.groups(problem) : U32) + (offset * Coordinates.groups(problem) : U32) + U32.from_nat(cell) : U32),
    U32.from_nat(Coordinates.query_index(problem,channel,row,column))) == True{} : Bool}) ->
  {lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,ArrayLeaf.leaf_cells(problem,storage,offset,span),(U32.from_nat(0n) * (span * Coordinates.groups(problem) : U32) + 0 + U32.from_nat(cell) : U32))) ==
    Values.matrix_value(problem,channel,row,column) : F32}:
  ArrayLeaf.Geometry{+span_bound,+offset_alignment,+span_alignment,+range_bound} = certificate
  +certificate = {ArrayLeaf.Geometry{span_bound,offset_alignment,span_alignment,range_bound} : ArrayLeaf.Geometry<problem,offset,span>}
  +groups = Coordinates.groups(problem)
  +g = U32.to_nat(groups)
  +o = U32.to_nat(offset)
  +n = U32.to_nat(span)
  +s = U32.to_nat(Padded.spatial(problem))
  +start = Nat.mul(o,g)
  +point = Reduction.position(problem,row,column)
  +wanted_group = Nat.div(channel,8n)
  +query = Coordinates.query_index(problem,channel,row,column)
  +destination = (U32.from_nat(0n) * (Padded.spatial(problem) * groups : U32) + (offset * groups : U32) + U32.from_nat(cell) : U32)
  +group_inside = Coordinates.channel_group(limits,problem,channel,supported,channel_inside)
  +positive = Order.le_lt_trans(0n,wanted_group,g,Order.zero_le(wanted_group),group_inside)
  +bound = ArrayLeaf.offset_bound(problem,offset,span,certificate)
  +indices = Equal.trans(Nat,Nat.add(start,cell),U32.to_nat(destination),query,
    Equal.sym(Nat,U32.to_nat(destination),Nat.add(start,cell),Coordinates.destination(limits,problem,offset,cell,valid,supported,bound,reach(limits,problem,offset,cell,valid,supported,bound,point_inside))),
    Equal.trans(Nat,U32.to_nat(destination),U32.to_nat(U32.from_nat(query)),query,
      Equal.cong(U32,Nat,value => U32.to_nat(value),destination,U32.from_nat(query),Arithmetic.equal_of_is_eq(destination,U32.from_nat(query),same)),
      Coordinates.query_exact(limits,problem,channel,row,column,valid,supported,channel_inside,row_inside,column_inside)))
  +local = Nat.div(cell,g)
  +lane_group = Nat.mod(cell,g)
  +split = DivisionOrder.quotient_exact(cell,g,positive)
  +grouped = Equal.trans(Nat,Nat.add(Nat.mul(Nat.add(o,local),g),lane_group),Nat.add(start,Nat.add(Nat.mul(local,g),lane_group)),query,
    regrouped(o,local,g,lane_group),
    %split : {Nat.add(start,_) == query : Nat}
    indices)
  +same_position = Radix.row_unique(Nat.add(o,local),point,g,lane_group,wanted_group,positive,DivisionOrder.remainder_small(cell,g,positive),group_inside,grouped)
  +same_group = Radix.col_unique(Nat.add(o,local),point,g,lane_group,wanted_group,positive,DivisionOrder.remainder_small(cell,g,positive),group_inside,grouped)
  +local_inside = DivisionOrder.quotient_below(cell,n,g,positive,
    %Coordinates.scaled(limits,problem,span,supported,span_bound) : {Nat.is_lt(cell,_) == True{} : Bool} in_span)
  +cells = ArrayLeaf.leaf_cells(problem,storage,offset,span)
  +source = (U32.from_nat(0n) * (span * groups : U32) + 0 + U32.from_nat(cell) : U32)
  Equal.trans(F32,lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,cells,source)),lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,cells,U32.from_nat(cell))),Values.matrix_value(problem,channel,row,column),
    Equal.cong(U32,F32,address => lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,cells,address)),source,U32.from_nat(cell),source_address(0n,(span * groups : U32),cell)),
    %Equal.sym(Nat,cell,Nat.add(Nat.mul(local,g),lane_group),split) :
      {lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,cells,U32.from_nat(_))) == Values.matrix_value(problem,channel,row,column) : F32}
    %Equal.sym(Nat,lane_group,wanted_group,same_group) :
      {lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,cells,U32.from_nat(Nat.add(Nat.mul(local,g),_)))) == Values.matrix_value(problem,channel,row,column) : F32}
    ArrayLeaf.leaf_cells_every_position(limits,problem,storage,offset,span,channel,local,row,column,valid,supported,certificate,channel_inside,local_inside,same_position,row_inside,column_inside))
