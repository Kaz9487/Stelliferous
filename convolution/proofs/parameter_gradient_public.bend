# The public weight and bias gradients of tiled tensors of the contract: the
# results hold the contract of the readers of the input and of the output
# gradient, and the tensors the calls read come back holding their values.

import Base
import ../parameter_gradient_specification.bend as Spec
import ./parameter_gradient_terms.bend as Terms
import ../../lib/tensor_parameter_gradient.bend as Parameter
import ../../lib/kernel_parameter_gradient.bend as Kernel
import ../../lib/kernel_tile.bend as Tile
import ../../lib/tensor_grid.bend as Grid
import ../../lib/tensor_tiled.bend as Tiled
import ../../lib/tensor_tiled_cells.bend as Cells
import ../../lib/proofs/kernel_bias_gradient.bend as Bias
import ../../lib/proofs/tensor_parameter_gradient_bias.bend as BiasCells
import ../../lib/proofs/tensor_tiled_representation.bend as Representation
import ../../lib/proofs/tensor_tiled_public.bend as TiledPublic
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/tensor_contract.bend as Contract
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/proofs/u32_full_range.bend as FullRange
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/kernel_gemm.bend as KernelGemm
import ../../lib/traversal_grouped.bend as Grouped
import ../../lib/machine_limits.bend as Machine
import ../../lib/default_machine_limits.bend as Default
import ../../lib/proofs/tensor_global_shape_count.bend as Global
import ../../lib/tensor.bend as Tensors
import ../../lib/proofs/tensor_whole_values.bend as WholeValues
import ../../lib/proofs/tensor_shape_refinement.bend as ShapeLaws
import ../../lib/proofs/tensor_parameter_gradient_cells.bend as ParameterCells
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/nat_ceiling.bend as Ceiling
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/boolean_logic.bend as Logic
import ../../lib/proofs/tensor_storage_count.bend as StorageCount
import ../../lib/tensor_convolution_gradient.bend as Gradient
import ../../lib/tensor_move.bend as Move
import ../../lib/storage_buffer.bend as Storage
import ../../lib/proofs/tensor_pieces_model.bend as PiecesModel
import ../../lib/kernel_input_gradient.bend as InputKernel
import ../../lib/tensor_convolution.bend as Conv
import ../../lib/proofs/tensor_tiled_relane.bend as Relane
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/word_multiplication.bend as WordMultiplication
import ../../lib/proofs/input_gradient_phase_domain.bend as Domain
import ../../lib/kernel_weight_gradient.bend as WeightKernel
import ../../lib/proofs/kernel_weight_gradient_packing.bend as Packing
import ../../lib/proofs/kernel_weight_gradient_group.bend as Groups
import ../../lib/proofs/kernel_weight_gradient_routes.bend as Routes
import ../../lib/proofs/kernel_weight_gradient_images.bend as Images
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/tensor_tiled_padded_cells.bend as PaddedCells
import ../../lib/proofs/tensor_padded_pieces.bend as PaddedPieces
import ../../lib/tensor_tiled_map.bend as TiledMap
import ../../lib/proofs/tensor_tiled_values.bend as TiledValues
import ../../lib/proofs/tensor_joined.bend as Joined
import ./input_gradient_call.bend as Call

# ---- The cells of the bias ----

# The bias a block of outputs reads at a lane inside the outputs.
def lane_case(~DL: Data,~d_value: DL -> U32 -> F32,inside: Bool,+work: Kernel.Work,+index: Nat,+ranges: List<&2,Grouped.Range>,+holder: Representation.Lifted<DL>,
  +right: F32,
  same: @inside_true: {inside == True{} : Bool} ->
    {Bias.element(~Representation.Lifted<DL>,~Bias.held_sample(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value)),work,index,ranges,holder) == right : F32}) ->
  {Bias.lane_value(~Representation.Lifted<DL>,~Bias.held_sample(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value)),inside,work,index,ranges,holder) ==
    Bool.pick(F32,inside,right,0.0) : F32}:
  match inside:
    case True{}: same({==})
    case False{}: {==}

# A lane of a block inside the outputs of the bias model over the lifted
# output gradient is the bias the reader of the bias reads there.
def bias_value_lane(~DL: Data,~d_value: DL -> U32 -> F32,+logical: DL,+outer_word: U32,+after: U32,+lanes: U32,+groups: U32,
  +nw: U32,+ow: U32,+rw: U32,+cw: U32,+gw: U32,+capacity: U32,
  +after_exact: {U32.to_nat(after) == Nat.mul(U32.to_nat(rw),U32.to_nat(cw)) : Nat},+lanes_exact: {U32.to_nat(lanes) == U32.to_nat(ow) : Nat},
  +groups_exact: {U32.to_nat(groups) == Kernel.groups(U32.to_nat(ow)) : Nat},
  +outer_exact: {U32.to_nat(outer_word) == Nat.mul(U32.to_nat(nw),Nat.mul(U32.to_nat(rw),U32.to_nat(cw))) : Nat},
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(nw),Nat.mul(U32.to_nat(rw),U32.to_nat(cw))),Kernel.groups(U32.to_nat(ow))),U32.to_nat(capacity)) == True{} : Bool},
  +outputs_room: {Nat.is_le(U32.to_nat(ow),U32.to_nat(capacity)) == True{} : Bool},
  +columns_positive: {Nat.is_lt(0n,U32.to_nat(cw)) == True{} : Bool},+group_positive: {Nat.is_lt(0n,U32.to_nat(gw)) == True{} : Bool},
  +block: Nat,+lane: Nat,+small: {Nat.is_le(lane,7n) == True{} : Bool},+below: {Nat.is_lt(lane,8n) == True{} : Bool},
  +block_inside: {Nat.is_lt(block,U32.to_nat(groups)) == True{} : Bool},
  +inside_true: {Nat.is_lt(Nat.add(Nat.mul(block,8n),lane),U32.to_nat(ow)) == True{} : Bool}) ->
  {Bias.element(~Representation.Lifted<DL>,~Bias.held_sample(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value)),
      Parameter.bias_work(Parameter.ImageGeometry{nw,ow,rw,cw},gw),Nat.add(Nat.mul(block,8n),lane),Bias.work_ranges(Parameter.bias_work(Parameter.ImageGeometry{nw,ow,rw,cw},gw)),
      Representation.Lifted{logical,Cells.Cells{outer_word,after,lanes,groups,Grid.Innermost{}}}) ==
    Terms.bias_at(~DL,~d_value,Terms.BiasRead{logical,Some{Parameter.ImageGeometry{nw,ow,rw,cw}},gw},U32.from_nat(Nat.add(Nat.mul(block,8n),lane))) : F32}:
  +index = Nat.add(Nat.mul(block,8n),lane)
  +work = Parameter.bias_work(Parameter.ImageGeometry{nw,ow,rw,cw},gw)
  +holder = {Representation.Lifted{logical,Cells.Cells{outer_word,after,lanes,groups,Grid.Innermost{}}} : Representation.Lifted<DL>}
  +problem = {Spec.Problem{Terms.shape(work),logical,logical} : Spec.Problem<DL>}
  +sample = Terms.bias_lane(~DL,~d_value,logical,outer_word,after,lanes,groups,U32.to_nat(nw),0n,U32.to_nat(ow),0n,0n,1n,1n,0n,U32.to_nat(rw),U32.to_nat(cw),U32.to_nat(gw),
    block,lane,small,below,capacity,after_exact,lanes_exact,groups_exact,outer_exact,total_room,block_inside,inside_true,columns_positive,group_positive)
  +element = {%inside_true :
      {Bias.lane_value(~Representation.Lifted<DL>,~Bias.held_sample(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value)),_,work,index,Bias.work_ranges(work),holder) ==
        Spec.bias(~DL,~Terms.gradient_at(~DL,~d_value),problem,index) : F32}
    sample :
    {Bias.element(~Representation.Lifted<DL>,~Bias.held_sample(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value)),work,index,Bias.work_ranges(work),holder) ==
      Spec.bias(~DL,~Terms.gradient_at(~DL,~d_value),problem,index) : F32}}
  +room = Order.transitive(index,U32.to_nat(ow),U32.to_nat(capacity),Bounds.strict_le(index,U32.to_nat(ow),inside_true),outputs_room)
  Equal.trans(F32,Bias.element(~Representation.Lifted<DL>,~Bias.held_sample(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value)),work,index,Bias.work_ranges(work),holder),
    Spec.bias(~DL,~Terms.gradient_at(~DL,~d_value),problem,index),Spec.bias(~DL,~Terms.gradient_at(~DL,~d_value),problem,U32.to_nat(U32.from_nat(index))),element,
    Equal.cong(Nat,F32,value => Spec.bias(~DL,~Terms.gradient_at(~DL,~d_value),problem,value),index,U32.to_nat(U32.from_nat(index)),
      Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,FullRange.roundtrip(index,capacity,room))))

# A lane of a block of the bias model is the lane of the cell of the lifted
# bias reader.
def bias_cell_lane(~DL: Data,~d_value: DL -> U32 -> F32,+logical: DL,+outer_word: U32,+after: U32,+lanes: U32,+groups: U32,
  +nw: U32,+ow: U32,+rw: U32,+cw: U32,+gw: U32,+capacity: U32,
  +after_exact: {U32.to_nat(after) == Nat.mul(U32.to_nat(rw),U32.to_nat(cw)) : Nat},+lanes_exact: {U32.to_nat(lanes) == U32.to_nat(ow) : Nat},
  +groups_exact: {U32.to_nat(groups) == Kernel.groups(U32.to_nat(ow)) : Nat},
  +outer_exact: {U32.to_nat(outer_word) == Nat.mul(U32.to_nat(nw),Nat.mul(U32.to_nat(rw),U32.to_nat(cw))) : Nat},
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(nw),Nat.mul(U32.to_nat(rw),U32.to_nat(cw))),Kernel.groups(U32.to_nat(ow))),U32.to_nat(capacity)) == True{} : Bool},
  +outputs_room: {Nat.is_le(U32.to_nat(ow),U32.to_nat(capacity)) == True{} : Bool},
  +columns_positive: {Nat.is_lt(0n,U32.to_nat(cw)) == True{} : Bool},+group_positive: {Nat.is_lt(0n,U32.to_nat(gw)) == True{} : Bool},
  +block: Nat,+lane: Nat,+small: {Nat.is_le(lane,7n) == True{} : Bool},+below: {Nat.is_lt(lane,8n) == True{} : Bool},
  +block_inside: {Nat.is_lt(block,U32.to_nat(groups)) == True{} : Bool}) ->
  {Bias.block_lane(~Representation.Lifted<DL>,~Bias.held_sample(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value)),
      Representation.Lifted{logical,Cells.Cells{outer_word,after,lanes,groups,Grid.Innermost{}}},
      Kernel.Block{Parameter.bias_work(Parameter.ImageGeometry{nw,ow,rw,cw},gw),block,Bias.work_ranges(Parameter.bias_work(Parameter.ImageGeometry{nw,ow,rw,cw},gw))},lane) ==
    Representation.lane(~Terms.BiasRead<DL>,~Terms.bias_at(~DL,~d_value),Terms.BiasRead{logical,Some{Parameter.ImageGeometry{nw,ow,rw,cw}},gw},
      TiledPublic.geometry_of([ow],0n,Grid.Innermost{}),0n,block,lane) : F32}:
  +index = Nat.add(Nat.mul(block,8n),lane)
  +work = Parameter.bias_work(Parameter.ImageGeometry{nw,ow,rw,cw},gw)
  +holder = {Representation.Lifted{logical,Cells.Cells{outer_word,after,lanes,groups,Grid.Innermost{}}} : Representation.Lifted<DL>}
  +read = {Terms.BiasRead{logical,Some{Parameter.ImageGeometry{nw,ow,rw,cw}},gw} : Terms.BiasRead<DL>}
  +lhs = Bias.block_lane(~Representation.Lifted<DL>,~Bias.held_sample(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value)),holder,
    Kernel.Block{work,block,Bias.work_ranges(work)},lane)
  %Order.commutative(Nat.mul(block,8n),lane) :
    {lhs == Bool.pick(F32,Nat.is_lt(_,U32.to_nat(ow)),Terms.bias_at(~DL,~d_value,read,U32.from_nat(Nat.add(Nat.mul(_,1n),0n))),0.0) : F32}
  %Equal.sym(Nat,Nat.add(Nat.mul(index,1n),0n),index,Equal.trans(Nat,Nat.add(Nat.mul(index,1n),0n),Nat.mul(index,1n),index,Addition.zero_right(Nat.mul(index,1n)),Order.times_one(index))) :
    {lhs == Bool.pick(F32,Nat.is_lt(index,U32.to_nat(ow)),Terms.bias_at(~DL,~d_value,read,U32.from_nat(_)),0.0) : F32}
  lane_case(~DL,~d_value,Nat.is_lt(index,U32.to_nat(ow)),work,index,Bias.work_ranges(work),holder,Terms.bias_at(~DL,~d_value,read,U32.from_nat(index)),
    inside_true => bias_value_lane(~DL,~d_value,logical,outer_word,after,lanes,groups,nw,ow,rw,cw,gw,capacity,after_exact,lanes_exact,groups_exact,outer_exact,total_room,
      outputs_room,columns_positive,group_positive,block,lane,small,below,block_inside,inside_true))

# A block of the bias model is the cell of the lifted bias reader.
def bias_cell(~DL: Data,~d_value: DL -> U32 -> F32,+logical: DL,+outer_word: U32,+after: U32,+lanes: U32,+groups: U32,
  +nw: U32,+ow: U32,+rw: U32,+cw: U32,+gw: U32,+capacity: U32,
  +after_exact: {U32.to_nat(after) == Nat.mul(U32.to_nat(rw),U32.to_nat(cw)) : Nat},+lanes_exact: {U32.to_nat(lanes) == U32.to_nat(ow) : Nat},
  +groups_exact: {U32.to_nat(groups) == Kernel.groups(U32.to_nat(ow)) : Nat},
  +outer_exact: {U32.to_nat(outer_word) == Nat.mul(U32.to_nat(nw),Nat.mul(U32.to_nat(rw),U32.to_nat(cw))) : Nat},
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(nw),Nat.mul(U32.to_nat(rw),U32.to_nat(cw))),Kernel.groups(U32.to_nat(ow))),U32.to_nat(capacity)) == True{} : Bool},
  +outputs_room: {Nat.is_le(U32.to_nat(ow),U32.to_nat(capacity)) == True{} : Bool},
  +columns_positive: {Nat.is_lt(0n,U32.to_nat(cw)) == True{} : Bool},+group_positive: {Nat.is_lt(0n,U32.to_nat(gw)) == True{} : Bool},
  +block: Nat,+block_inside: {Nat.is_lt(block,U32.to_nat(groups)) == True{} : Bool}) ->
  {Bias.cell_values(~Representation.Lifted<DL>,~Bias.held_sample(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value)),
      Representation.Lifted{logical,Cells.Cells{outer_word,after,lanes,groups,Grid.Innermost{}}},
      Kernel.Block{Parameter.bias_work(Parameter.ImageGeometry{nw,ow,rw,cw},gw),block,Bias.work_ranges(Parameter.bias_work(Parameter.ImageGeometry{nw,ow,rw,cw},gw))}) ==
    Representation.cell(~Terms.BiasRead<DL>,~Terms.bias_at(~DL,~d_value),Terms.BiasRead{logical,Some{Parameter.ImageGeometry{nw,ow,rw,cw}},gw},
      TiledPublic.geometry_of([ow],0n,Grid.Innermost{}),0n,block) : Tile.Tile8}:
  +work = Parameter.bias_work(Parameter.ImageGeometry{nw,ow,rw,cw},gw)
  +holder = {Representation.Lifted{logical,Cells.Cells{outer_word,after,lanes,groups,Grid.Innermost{}}} : Representation.Lifted<DL>}
  +read = {Terms.BiasRead{logical,Some{Parameter.ImageGeometry{nw,ow,rw,cw}},gw} : Terms.BiasRead<DL>}
  TileLaws.tiles_equal(
    Bias.cell_values(~Representation.Lifted<DL>,~Bias.held_sample(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value)),holder,
      Kernel.Block{work,block,Bias.work_ranges(work)}),
    Representation.cell(~Terms.BiasRead<DL>,~Terms.bias_at(~DL,~d_value),read,TiledPublic.geometry_of([ow],0n,Grid.Innermost{}),0n,block),
    bias_cell_lane(~DL,~d_value,logical,outer_word,after,lanes,groups,nw,ow,rw,cw,gw,capacity,after_exact,lanes_exact,groups_exact,outer_exact,total_room,outputs_room,
      columns_positive,group_positive,block,0n,{==},{==},block_inside),
    bias_cell_lane(~DL,~d_value,logical,outer_word,after,lanes,groups,nw,ow,rw,cw,gw,capacity,after_exact,lanes_exact,groups_exact,outer_exact,total_room,outputs_room,
      columns_positive,group_positive,block,1n,{==},{==},block_inside),
    bias_cell_lane(~DL,~d_value,logical,outer_word,after,lanes,groups,nw,ow,rw,cw,gw,capacity,after_exact,lanes_exact,groups_exact,outer_exact,total_room,outputs_room,
      columns_positive,group_positive,block,2n,{==},{==},block_inside),
    bias_cell_lane(~DL,~d_value,logical,outer_word,after,lanes,groups,nw,ow,rw,cw,gw,capacity,after_exact,lanes_exact,groups_exact,outer_exact,total_room,outputs_room,
      columns_positive,group_positive,block,3n,{==},{==},block_inside),
    bias_cell_lane(~DL,~d_value,logical,outer_word,after,lanes,groups,nw,ow,rw,cw,gw,capacity,after_exact,lanes_exact,groups_exact,outer_exact,total_room,outputs_room,
      columns_positive,group_positive,block,4n,{==},{==},block_inside),
    bias_cell_lane(~DL,~d_value,logical,outer_word,after,lanes,groups,nw,ow,rw,cw,gw,capacity,after_exact,lanes_exact,groups_exact,outer_exact,total_room,outputs_room,
      columns_positive,group_positive,block,5n,{==},{==},block_inside),
    bias_cell_lane(~DL,~d_value,logical,outer_word,after,lanes,groups,nw,ow,rw,cw,gw,capacity,after_exact,lanes_exact,groups_exact,outer_exact,total_room,outputs_room,
      columns_positive,group_positive,block,6n,{==},{==},block_inside),
    bias_cell_lane(~DL,~d_value,logical,outer_word,after,lanes,groups,nw,ow,rw,cw,gw,capacity,after_exact,lanes_exact,groups_exact,outer_exact,total_room,outputs_room,
      columns_positive,group_positive,block,7n,{==},{==},block_inside))

# The blocks a bias call writes from position on hold the lifted bias reader.
def bias_values(~DL: Data,~d_value: DL -> U32 -> F32,+logical: DL,+outer_word: U32,+after: U32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+d_cells: List<&2,U32>,+d_lanes: U32,
  +nw: U32,+ow: U32,+rw: U32,+cw: U32,+gw: U32,
  +d_named: {Parameter.cell_image(d_cells,d_lanes) == Some{Parameter.ImageGeometry{nw,ow,rw,cw}} : Maybe<&2,Parameter.ImageGeometry>},+d_counted: Global.Counted<d_cells>,
  +lanes_room: {Nat.is_le(U32.to_nat(d_lanes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +accepted: {Parameter.positive(Parameter.ImageGeometry{nw,ow,rw,cw}) && U32.is_lt(0,gw) == True{} : Bool},
  +after_exact: {U32.to_nat(after) == Nat.mul(U32.to_nat(rw),U32.to_nat(cw)) : Nat},
  +outer_exact: {U32.to_nat(outer_word) == Nat.mul(U32.to_nat(nw),Nat.mul(U32.to_nat(rw),U32.to_nat(cw))) : Nat},
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(nw),Nat.mul(U32.to_nat(rw),U32.to_nat(cw))),Kernel.groups(U32.to_nat(ow))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +columns_positive: {Nat.is_lt(0n,U32.to_nat(cw)) == True{} : Bool},+group_positive: {Nat.is_lt(0n,U32.to_nat(gw)) == True{} : Bool},
  remaining: Nat,+position: Nat,+room: {Nat.is_le(Nat.add(position,remaining),U32.to_nat(Tiled.groups(ow))) == True{} : Bool}) ->
  Members.All(Nat,{index => Contract.At(~Tile.Tile8,~Representation.Lifted<Terms.BiasRead<DL>>,~Representation.lifted(~Terms.BiasRead<DL>,~Terms.bias_at(~DL,~d_value)),
      Representation.Lifted{Terms.BiasRead{logical,Some{Parameter.ImageGeometry{nw,ow,rw,cw}},gw},TiledPublic.geometry_of([ow],0n,Grid.Innermost{})},
      Buffer.content(~Tile.Tile8,BiasCells.written_of(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),
        Representation.Lifted{logical,Cells.Cells{outer_word,after,ow,Tiled.groups(ow),Grid.Innermost{}}},
        BiasCells.BiasCall{limits,default_capacity,d_cells,d_lanes,Parameter.ImageGeometry{nw,ow,rw,cw},d_named,d_counted,lanes_room,gw,accepted})),index) : Nat -> Data},
    Ranges.range(remaining,position)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      +image = {Parameter.ImageGeometry{nw,ow,rw,cw} : Parameter.ImageGeometry}
      +call = {BiasCells.BiasCall{limits,default_capacity,d_cells,d_lanes,image,d_named,d_counted,lanes_room,gw,accepted} : BiasCells.BiasCall}
      +holder = {Representation.Lifted{logical,Cells.Cells{outer_word,after,ow,Tiled.groups(ow),Grid.Innermost{}}} : Representation.Lifted<DL>}
      +read = {Terms.BiasRead{logical,Some{image},gw} : Terms.BiasRead<DL>}
      +geometry = TiledPublic.geometry_of([ow],0n,Grid.Innermost{})
      +capacity = Machine.capacity(limits)
      +groups_exact = Equal.sym(Nat,Kernel.groups(U32.to_nat(ow)),U32.to_nat(Tiled.groups(ow)),BiasCells.groups_exact(call))
      +inside = Order.room_index(position,rest,U32.to_nat(Tiled.groups(ow)),room)
      +stored = BiasCells.value_at(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),holder,call,position,inside)
      +cell = bias_cell(~DL,~d_value,logical,outer_word,after,ow,Tiled.groups(ow),nw,ow,rw,cw,gw,capacity,after_exact,{==},groups_exact,outer_exact,total_room,
        BiasCells.outputs_room(call),columns_positive,group_positive,position,inside)
      +word = U32.from_nat(position)
      +exact = FullRange.roundtrip(position,capacity,
        Order.transitive(position,U32.to_nat(Tiled.groups(ow)),U32.to_nat(capacity),Bounds.strict_le(position,U32.to_nat(Tiled.groups(ow)),inside),BiasCells.groups_room(call)))
      +decoded = Representation.decoded(~Terms.BiasRead<DL>,~Terms.bias_at(~DL,~d_value),read,geometry,0n,position,word,exact,{==},inside)
      KernelGemm.Both{
        Equal.trans(Tile.Tile8,Buffer.value(~Tile.Tile8,word,Buffer.content(~Tile.Tile8,BiasCells.written_of(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),holder,call))),
          Representation.cell(~Terms.BiasRead<DL>,~Terms.bias_at(~DL,~d_value),read,geometry,0n,position),
          Representation.lifted(~Terms.BiasRead<DL>,~Terms.bias_at(~DL,~d_value),Representation.Lifted{read,geometry},word),
          Equal.trans(Tile.Tile8,Buffer.value(~Tile.Tile8,word,Buffer.content(~Tile.Tile8,BiasCells.written_of(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),holder,call))),
            Bias.cell_values(~Representation.Lifted<DL>,~Bias.held_sample(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value)),holder,
              Kernel.Block{Parameter.bias_work(image,gw),position,Bias.work_ranges(Parameter.bias_work(image,gw))}),
            Representation.cell(~Terms.BiasRead<DL>,~Terms.bias_at(~DL,~d_value),read,geometry,0n,position),stored,cell),
          Equal.sym(Tile.Tile8,Representation.lifted(~Terms.BiasRead<DL>,~Terms.bias_at(~DL,~d_value),Representation.Lifted{read,geometry},word),
            Representation.cell(~Terms.BiasRead<DL>,~Terms.bias_at(~DL,~d_value),read,geometry,0n,position),decoded)),
        bias_values(~DL,~d_value,logical,outer_word,after,limits,default_capacity,d_cells,d_lanes,nw,ow,rw,cw,gw,d_named,d_counted,lanes_room,accepted,after_exact,outer_exact,
          total_room,columns_positive,group_positive,rest,1n+position,Order.advance_room(position,rest,U32.to_nat(Tiled.groups(ow)),room))}

# Positive outputs make at least one group.
def groups_positive(+limits: Machine.Limits,+outputs: U32,+room: {Nat.is_le(U32.to_nat(outputs),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(outputs)) == True{} : Bool}) -> {Nat.is_lt(0n,U32.to_nat(Tiled.groups(outputs))) == True{} : Bool}:
  +count = U32.to_nat(outputs)
  +blocks = Ceiling.ceiling(count,8n)
  %Equal.sym(Nat,U32.to_nat(Tiled.groups(outputs)),blocks,Encoded.eight_blocks_exact(limits,outputs,room)) : {Nat.is_lt(0n,_) == True{} : Bool}
  RowMajor.left_factor_positive(blocks,8n,Bounds.nat_trans(0n,count,Nat.mul(blocks,8n),positive,Ceiling.covers(count,8n,{==})))

# The cells a bias call writes are a whole tensor of the lifted bias reader.
def bias_whole(~DL: Data,~d_value: DL -> U32 -> F32,+logical: DL,+outer_word: U32,+after: U32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+d_cells: List<&2,U32>,+d_lanes: U32,
  +nw: U32,+ow: U32,+rw: U32,+cw: U32,+gw: U32,
  +d_named: {Parameter.cell_image(d_cells,d_lanes) == Some{Parameter.ImageGeometry{nw,ow,rw,cw}} : Maybe<&2,Parameter.ImageGeometry>},+d_counted: Global.Counted<d_cells>,
  +lanes_room: {Nat.is_le(U32.to_nat(d_lanes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +accepted: {Parameter.positive(Parameter.ImageGeometry{nw,ow,rw,cw}) && U32.is_lt(0,gw) == True{} : Bool},
  +after_exact: {U32.to_nat(after) == Nat.mul(U32.to_nat(rw),U32.to_nat(cw)) : Nat},
  +outer_exact: {U32.to_nat(outer_word) == Nat.mul(U32.to_nat(nw),Nat.mul(U32.to_nat(rw),U32.to_nat(cw))) : Nat},
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(nw),Nat.mul(U32.to_nat(rw),U32.to_nat(cw))),Kernel.groups(U32.to_nat(ow))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +columns_positive: {Nat.is_lt(0n,U32.to_nat(cw)) == True{} : Bool},+group_positive: {Nat.is_lt(0n,U32.to_nat(gw)) == True{} : Bool},+workers: U32) ->
  Global.Contract(~Tile.Tile8,~Representation.Lifted<Terms.BiasRead<DL>>,~Representation.lifted(~Terms.BiasRead<DL>,~Terms.bias_at(~DL,~d_value)),
    Representation.Lifted{Terms.BiasRead{logical,Some{Parameter.ImageGeometry{nw,ow,rw,cw}},gw},TiledPublic.geometry_of([ow],0n,Grid.Innermost{})},
    Tensors.whole(~Tile.Tile8,Buffer.pack(~Tile.Tile8,BiasCells.written_of(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),
        Representation.Lifted{logical,Cells.Cells{outer_word,after,ow,Tiled.groups(ow),Grid.Innermost{}}},
        BiasCells.BiasCall{limits,default_capacity,d_cells,d_lanes,Parameter.ImageGeometry{nw,ow,rw,cw},d_named,d_counted,lanes_room,gw,accepted})),
      [Tiled.groups(ow)],Tiled.zero_tile(),Tensors.Usable{},workers)):
  +image = {Parameter.ImageGeometry{nw,ow,rw,cw} : Parameter.ImageGeometry}
  +call = {BiasCells.BiasCall{limits,default_capacity,d_cells,d_lanes,image,d_named,d_counted,lanes_room,gw,accepted} : BiasCells.BiasCall}
  +holder = {Representation.Lifted{logical,Cells.Cells{outer_word,after,ow,Tiled.groups(ow),Grid.Innermost{}}} : Representation.Lifted<DL>}
  +reader = {Representation.Lifted{Terms.BiasRead{logical,Some{image},gw},TiledPublic.geometry_of([ow],0n,Grid.Innermost{})} : Representation.Lifted<Terms.BiasRead<DL>>}
  +written = BiasCells.written_of(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),holder,call)
  +content = Buffer.content(~Tile.Tile8,written)
  +count = U32.to_nat(Tiled.groups(ow))
  +shape = {[Tiled.groups(ow)] : List<&2,U32>}
  +length = BiasCells.values_length(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),holder,call)
  +values = {%Equal.sym(Nat,Buffer.content_length(~Tile.Tile8,content),count,length) :
      Members.All(Nat,{index => Contract.At(~Tile.Tile8,~Representation.Lifted<Terms.BiasRead<DL>>,~Representation.lifted(~Terms.BiasRead<DL>,~Terms.bias_at(~DL,~d_value)),reader,
        content,index) : Nat -> Data},Ranges.range(_,0n))
    bias_values(~DL,~d_value,logical,outer_word,after,limits,default_capacity,d_cells,d_lanes,nw,ow,rw,cw,gw,d_named,d_counted,lanes_room,accepted,after_exact,outer_exact,
      total_room,columns_positive,group_positive,count,0n,Order.reflexive(count)) :
    Contract.Values(~Tile.Tile8,~Representation.Lifted<Terms.BiasRead<DL>>,~Representation.lifted(~Terms.BiasRead<DL>,~Terms.bias_at(~DL,~d_value)),reader,content)}
  +formed = {Contract.WellFormed{Equal.trans(Nat,Buffer.content_length(~Tile.Tile8,content),count,ShapeLaws.elements(shape),length,
      Equal.sym(Nat,Nat.mul(count,1n),count,Order.times_one(count))),BiasCells.values_ready(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),holder,call)} :
    Contract.WellFormed<Tile.Tile8,shape,content>}
  +facts = {%Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,written)),content,Buffer.packed_contents(~Tile.Tile8,written)) :
      Contract.Facts(~Tile.Tile8,~Representation.Lifted<Terms.BiasRead<DL>>,~Representation.lifted(~Terms.BiasRead<DL>,~Terms.bias_at(~DL,~d_value)),reader,shape,_)
    KernelGemm.Both{formed,values} :
    Contract.Facts(~Tile.Tile8,~Representation.Lifted<Terms.BiasRead<DL>>,~Representation.lifted(~Terms.BiasRead<DL>,~Terms.bias_at(~DL,~d_value)),reader,shape,
      Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,written)))}
  +positive = ParameterCells.sizes_channels(image,ParameterCells.image_sizes(image,Logic.left(Parameter.positive(image),U32.is_lt(0,gw),accepted)))
  +nonempty = {%Equal.sym(Nat,Nat.mul(count,1n),count,Order.times_one(count)) : {Nat.is_lt(0n,_) == True{} : Bool}
    groups_positive(limits,ow,BiasCells.outputs_room(call),positive) : {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool}}
  WholeValues.whole_contract(~Tile.Tile8,~Representation.Lifted<Terms.BiasRead<DL>>,~Representation.lifted(~Terms.BiasRead<DL>,~Terms.bias_at(~DL,~d_value)),reader,limits,shape,
    Tiled.zero_tile(),workers,written,BiasCells.result_counted(call),BiasCells.result_room(call),nonempty,facts)

# The cells a bias call writes, laid out with the outputs in their lanes,
# hold the lifted bias reader.
def bias_contract(~DL: Data,~d_value: DL -> U32 -> F32,+logical: DL,+outer_word: U32,+after: U32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+d_cells: List<&2,U32>,+d_lanes: U32,
  +nw: U32,+ow: U32,+rw: U32,+cw: U32,+gw: U32,
  +d_named: {Parameter.cell_image(d_cells,d_lanes) == Some{Parameter.ImageGeometry{nw,ow,rw,cw}} : Maybe<&2,Parameter.ImageGeometry>},+d_counted: Global.Counted<d_cells>,
  +lanes_room: {Nat.is_le(U32.to_nat(d_lanes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +accepted: {Parameter.positive(Parameter.ImageGeometry{nw,ow,rw,cw}) && U32.is_lt(0,gw) == True{} : Bool},
  +after_exact: {U32.to_nat(after) == Nat.mul(U32.to_nat(rw),U32.to_nat(cw)) : Nat},
  +outer_exact: {U32.to_nat(outer_word) == Nat.mul(U32.to_nat(nw),Nat.mul(U32.to_nat(rw),U32.to_nat(cw))) : Nat},
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(nw),Nat.mul(U32.to_nat(rw),U32.to_nat(cw))),Kernel.groups(U32.to_nat(ow))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +columns_positive: {Nat.is_lt(0n,U32.to_nat(cw)) == True{} : Bool},+group_positive: {Nat.is_lt(0n,U32.to_nat(gw)) == True{} : Bool},+workers: U32) ->
  TiledPublic.Contract(~Terms.BiasRead<DL>,~Terms.bias_at(~DL,~d_value),Terms.BiasRead{logical,Some{Parameter.ImageGeometry{nw,ow,rw,cw}},gw},[ow],
    Tiled.Tiled{Tensors.whole(~Tile.Tile8,Buffer.pack(~Tile.Tile8,BiasCells.written_of(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),
        Representation.Lifted{logical,Cells.Cells{outer_word,after,ow,Tiled.groups(ow),Grid.Innermost{}}},
        BiasCells.BiasCall{limits,default_capacity,d_cells,d_lanes,Parameter.ImageGeometry{nw,ow,rw,cw},d_named,d_counted,lanes_room,gw,accepted})),
      [Tiled.groups(ow)],Tiled.zero_tile(),Tensors.Usable{},workers),0n,ow,Grid.Innermost{}}):
  +call = {BiasCells.BiasCall{limits,default_capacity,d_cells,d_lanes,Parameter.ImageGeometry{nw,ow,rw,cw},d_named,d_counted,lanes_room,gw,accepted} : BiasCells.BiasCall}
  +outputs = U32.to_nat(ow)
  +room = {%Equal.sym(Nat,Nat.mul(outputs,1n),outputs,Order.times_one(outputs)) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    BiasCells.outputs_room(call) : {Nat.is_le(ShapeLaws.elements([ow]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  KernelGemm.Both{
    bias_whole(~DL,~d_value,logical,outer_word,after,limits,default_capacity,d_cells,d_lanes,nw,ow,rw,cw,gw,d_named,d_counted,lanes_room,accepted,after_exact,outer_exact,
      total_room,columns_positive,group_positive,workers),
    TiledPublic.Laid{{==},{==},{==},StorageCount.counted_within(limits,[ow],room,default_capacity),{==}}}

# A bias call on the cells of d runs the bias kernel, writes its cells whole
# and gives the pieces of d back in their grid.
def bias_run(~DL: Data,~d_value: DL -> U32 -> F32,+logical: DL,+outer_word: U32,+after: U32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+d_cells: List<&2,U32>,
  +nw: U32,+ow: U32,+rw: U32,+cw: U32,+gw: U32,
  +d_named: {Parameter.cell_image(d_cells,ow) == Some{Parameter.ImageGeometry{nw,ow,rw,cw}} : Maybe<&2,Parameter.ImageGeometry>},+d_counted: Global.Counted<d_cells>,
  +lanes_room: {Nat.is_le(U32.to_nat(ow),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +accepted: {Parameter.positive(Parameter.ImageGeometry{nw,ow,rw,cw}) && U32.is_lt(0,gw) == True{} : Bool},
  +dm: PiecesModel.Model<Tile.Tile8>,+df: Tile.Tile8,+dw: U32,+axis: Nat,
  +shaped: Global.Shaped(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),
    Representation.Lifted{logical,Cells.Cells{outer_word,after,ow,Tiled.groups(ow),Grid.Innermost{}}},d_cells,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),d_cells,df,Tensors.Usable{},dw})) ->
  {Gradient.bias_cells(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),d_cells,df,Tensors.Usable{},dw},axis,ow,Grid.Innermost{}},gw) ==
    (Tiled.Tiled{Tensors.whole(~Tile.Tile8,Buffer.pack(~Tile.Tile8,BiasCells.written_of(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),
          Representation.Lifted{logical,Cells.Cells{outer_word,after,ow,Tiled.groups(ow),Grid.Innermost{}}},
          BiasCells.BiasCall{limits,default_capacity,d_cells,ow,Parameter.ImageGeometry{nw,ow,rw,cw},d_named,d_counted,lanes_room,gw,accepted})),
        [Tiled.groups(ow)],Tiled.zero_tile(),Tensors.Usable{},dw),0n,ow,Grid.Innermost{}},
      Tiled.Tiled{Parameter.back(Parameter.Read{d_cells,df,Tensors.Usable{},dw},Pair.fst(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,
        Kernel.bias(Parameter.bias_work(Parameter.ImageGeometry{nw,ow,rw,cw},gw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)),Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),Tiled.groups(ow))))),
        axis,ow,Grid.Innermost{}}) : Tiled.Tiled & Tiled.Tiled}:
  +image = {Parameter.ImageGeometry{nw,ow,rw,cw} : Parameter.ImageGeometry}
  +call = {BiasCells.BiasCall{limits,default_capacity,d_cells,ow,image,d_named,d_counted,lanes_room,gw,accepted} : BiasCells.BiasCall}
  +holder = {Representation.Lifted{logical,Cells.Cells{outer_word,after,ow,Tiled.groups(ow),Grid.Innermost{}}} : Representation.Lifted<DL>}
  %Equal.sym(Tiled.Tiled & Tensors.Tensor<Tile.Tile8>,Parameter.bias(Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),d_cells,df,Tensors.Usable{},dw},ow,gw),
      Parameter.bias_finished(Parameter.Read{d_cells,df,Tensors.Usable{},dw},dw,ow,Kernel.bias(Parameter.bias_work(image,gw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)),Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),Tiled.groups(ow)))),BiasCells.run_exact(call,PiecesModel.pack(~Tile.Tile8,dm),df,dw)) :
    {Gradient.bias_result(axis,ow,_) ==
      (Tiled.Tiled{Tensors.whole(~Tile.Tile8,Buffer.pack(~Tile.Tile8,BiasCells.written_of(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),holder,call)),[Tiled.groups(ow)],Tiled.zero_tile(),Tensors.Usable{},dw),0n,ow,Grid.Innermost{}},
      Tiled.Tiled{Parameter.back(Parameter.Read{d_cells,df,Tensors.Usable{},dw},Pair.fst(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,Kernel.bias(Parameter.bias_work(image,gw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)),Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),Tiled.groups(ow))))),axis,ow,Grid.Innermost{}}) : Tiled.Tiled & Tiled.Tiled}
  %Equal.sym(Tiled.Tiled & Tensors.Tensor<Tile.Tile8>,Parameter.bias_finished(Parameter.Read{d_cells,df,Tensors.Usable{},dw},dw,ow,Kernel.bias(Parameter.bias_work(image,gw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)),Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),Tiled.groups(ow)))),
      (Tiled.Tiled{Tensors.whole(~Tile.Tile8,Pair.snd(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,Kernel.bias(Parameter.bias_work(image,gw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)),Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),Tiled.groups(ow)))),[Tiled.groups(ow)],Tiled.zero_tile(),Tensors.Usable{},dw),0n,ow,Grid.Innermost{}},
        Parameter.back(Parameter.Read{d_cells,df,Tensors.Usable{},dw},Pair.fst(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,Kernel.bias(Parameter.bias_work(image,gw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)),Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),Tiled.groups(ow)))))),
      BiasCells.finished_exact(Parameter.Read{d_cells,df,Tensors.Usable{},dw},dw,ow,Kernel.bias(Parameter.bias_work(image,gw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)),Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),Tiled.groups(ow))))) :
    {Gradient.bias_result(axis,ow,_) ==
      (Tiled.Tiled{Tensors.whole(~Tile.Tile8,Buffer.pack(~Tile.Tile8,BiasCells.written_of(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),holder,call)),[Tiled.groups(ow)],Tiled.zero_tile(),Tensors.Usable{},dw),0n,ow,Grid.Innermost{}},
      Tiled.Tiled{Parameter.back(Parameter.Read{d_cells,df,Tensors.Usable{},dw},Pair.fst(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,Kernel.bias(Parameter.bias_work(image,gw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)),Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),Tiled.groups(ow))))),axis,ow,Grid.Innermost{}}) : Tiled.Tiled & Tiled.Tiled}
  %Equal.sym(Storage.Buffer<Tile.Tile8>,Pair.snd(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,Kernel.bias(Parameter.bias_work(image,gw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)),Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),Tiled.groups(ow)))),Buffer.pack(~Tile.Tile8,BiasCells.written_of(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),holder,call)),
      BiasCells.values_exact(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),holder,call,PiecesModel.pack(~Tile.Tile8,dm),df,dw,shaped)) :
    {Gradient.bias_result(axis,ow,(Tiled.Tiled{Tensors.whole(~Tile.Tile8,_,[Tiled.groups(ow)],Tiled.zero_tile(),Tensors.Usable{},dw),0n,ow,Grid.Innermost{}},
        Parameter.back(Parameter.Read{d_cells,df,Tensors.Usable{},dw},Pair.fst(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,Kernel.bias(Parameter.bias_work(image,gw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)),Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),Tiled.groups(ow))))))) ==
      (Tiled.Tiled{Tensors.whole(~Tile.Tile8,Buffer.pack(~Tile.Tile8,BiasCells.written_of(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),holder,call)),[Tiled.groups(ow)],Tiled.zero_tile(),Tensors.Usable{},dw),0n,ow,Grid.Innermost{}},
      Tiled.Tiled{Parameter.back(Parameter.Read{d_cells,df,Tensors.Usable{},dw},Pair.fst(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,Kernel.bias(Parameter.bias_work(image,gw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)),Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),Tiled.groups(ow))))),axis,ow,Grid.Innermost{}}) : Tiled.Tiled & Tiled.Tiled}
  {==}

# ---- The bias gradient of a call ----

# The bias gradient and the output gradient a bias call returns hold the bias
# reader and the reader of the output gradient.
def BiasHeld(~DL: Data,~d_value: DL -> U32 -> F32,+read: Terms.BiasRead<DL>,+outputs: U32,+logical: DL,+values: List<&2,U32>,result: Tiled.Tiled & Tiled.Tiled) -> Data:
  (bias,d) = result
  KernelGemm.Both<TiledPublic.Contract(~Terms.BiasRead<DL>,~Terms.bias_at(~DL,~d_value),read,[outputs],bias),TiledPublic.Contract(~DL,~d_value,logical,values,d)>

# A bias call on the cells of an output gradient laid out with the outputs in
# their lanes and the groups innermost.
def bias_call(~DL: Data,~d_value: DL -> U32 -> F32,+logical: DL,+values: List<&2,U32>,+axis: Nat,+outer_word: U32,+after: U32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +nw: U32,+ow: U32,+rw: U32,+cw: U32,
  +geometry_exact: {TiledPublic.geometry_of(values,axis,Grid.Innermost{}) == Cells.Cells{outer_word,after,ow,Tiled.groups(ow),Grid.Innermost{}} : Cells.Cells},
  +d_named: {Parameter.cell_image(Tiled.cell_shape(values,axis,Grid.Innermost{}),ow) == Some{Parameter.ImageGeometry{nw,ow,rw,cw}} : Maybe<&2,Parameter.ImageGeometry>},
  +lanes_room: {Nat.is_le(U32.to_nat(ow),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Parameter.positive(Parameter.ImageGeometry{nw,ow,rw,cw}) == True{} : Bool},
  +after_exact: {U32.to_nat(after) == Nat.mul(U32.to_nat(rw),U32.to_nat(cw)) : Nat},
  +outer_exact: {U32.to_nat(outer_word) == Nat.mul(U32.to_nat(nw),Nat.mul(U32.to_nat(rw),U32.to_nat(cw))) : Nat},
  +total_room: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(nw),Nat.mul(U32.to_nat(rw),U32.to_nat(cw))),Kernel.groups(U32.to_nat(ow))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +columns_positive: {Nat.is_lt(0n,U32.to_nat(cw)) == True{} : Bool},
  +dm: PiecesModel.Model<Tile.Tile8>,+df: Tile.Tile8,+dw: U32,
  +d_contract: TiledPublic.Contract(~DL,~d_value,logical,values,
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),Tiled.cell_shape(values,axis,Grid.Innermost{}),df,Tensors.Usable{},dw},axis,ow,Grid.Innermost{}})) ->
  BiasHeld(~DL,~d_value,Terms.BiasRead{logical,Some{Parameter.ImageGeometry{nw,ow,rw,cw}},Gradient.group_length()},ow,logical,values,
    Gradient.bias_cells(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),Tiled.cell_shape(values,axis,Grid.Innermost{}),df,Tensors.Usable{},dw},axis,ow,Grid.Innermost{}},
      Gradient.group_length())):
  KernelGemm.Both{+global,TiledPublic.Laid{+inside,+laid_shaped,+restored,+counted,+pinned}} = d_contract
  +image = {Parameter.ImageGeometry{nw,ow,rw,cw} : Parameter.ImageGeometry}
  +own = {KernelGemm.Both{global,TiledPublic.Laid{inside,laid_shaped,restored,counted,pinned}} :
    TiledPublic.Contract(~DL,~d_value,logical,values,
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),Tiled.cell_shape(values,axis,Grid.Innermost{}),df,Tensors.Usable{},dw},axis,ow,Grid.Innermost{}})}
  +group = Gradient.group_length()
  +dcells = Tiled.cell_shape(values,axis,Grid.Innermost{})
  +holder = {Representation.Lifted{logical,Cells.Cells{outer_word,after,ow,Tiled.groups(ow),Grid.Innermost{}}} : Representation.Lifted<DL>}
  +shaped = {%geometry_exact : Global.Shaped(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),Representation.Lifted{logical,_},dcells,
        Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),dcells,df,Tensors.Usable{},dw})
      TiledPublic.cells_shaped(~DL,~d_value,logical,values,axis,ow,Grid.Innermost{},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),dcells,df,Tensors.Usable{},dw},own) :
    Global.Shaped(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),holder,dcells,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),dcells,df,Tensors.Usable{},dw})}
  +d_counted = Global.shaped_counted(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),holder,dcells,
    Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),dcells,df,Tensors.Usable{},dw},shaped)
  +accepted = {%Equal.sym(Bool,Parameter.positive(image),True{},positive) : {_ && U32.is_lt(0,group) == True{} : Bool}
      {==} :
    {Parameter.positive(image) && U32.is_lt(0,group) == True{} : Bool}}
  +call = {BiasCells.BiasCall{limits,default_capacity,dcells,ow,image,d_named,d_counted,lanes_room,group,accepted} : BiasCells.BiasCall}
  +back_global = {%Equal.sym(Cells.Cells,TiledPublic.geometry_of(values,axis,Grid.Innermost{}),Cells.Cells{outer_word,after,ow,Tiled.groups(ow),Grid.Innermost{}},geometry_exact) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),Representation.Lifted{logical,_},
        Parameter.back(Parameter.Read{dcells,df,Tensors.Usable{},dw},Pair.fst(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,
          Kernel.bias(Parameter.bias_work(image,group),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)),Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),Tiled.groups(ow))))))
      BiasCells.d_back(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),holder,call,PiecesModel.pack(~Tile.Tile8,dm),df,dw,shaped) :
    Global.Contract(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),Representation.Lifted{logical,TiledPublic.geometry_of(values,axis,Grid.Innermost{})},
      Parameter.back(Parameter.Read{dcells,df,Tensors.Usable{},dw},Pair.fst(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,
        Kernel.bias(Parameter.bias_work(image,group),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)),Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),Tiled.groups(ow))))))}
  %Equal.sym(Tiled.Tiled & Tiled.Tiled,
      Gradient.bias_cells(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),dcells,df,Tensors.Usable{},dw},axis,ow,Grid.Innermost{}},group),
      (Tiled.Tiled{Tensors.whole(~Tile.Tile8,Buffer.pack(~Tile.Tile8,BiasCells.written_of(~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),holder,call)),
          [Tiled.groups(ow)],Tiled.zero_tile(),Tensors.Usable{},dw),0n,ow,Grid.Innermost{}},
        Tiled.Tiled{Parameter.back(Parameter.Read{dcells,df,Tensors.Usable{},dw},Pair.fst(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,
          Kernel.bias(Parameter.bias_work(image,group),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)),Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),Tiled.groups(ow))))),
          axis,ow,Grid.Innermost{}}),
      bias_run(~DL,~d_value,logical,outer_word,after,limits,default_capacity,dcells,nw,ow,rw,cw,group,d_named,d_counted,lanes_room,accepted,dm,df,dw,axis,shaped)) :
    BiasHeld(~DL,~d_value,Terms.BiasRead{logical,Some{image},group},ow,logical,values,_)
  KernelGemm.Both{
    bias_contract(~DL,~d_value,logical,outer_word,after,limits,default_capacity,dcells,ow,nw,ow,rw,cw,group,d_named,d_counted,lanes_room,accepted,after_exact,outer_exact,
      total_room,columns_positive,{==},dw),
    KernelGemm.Both{back_global,TiledPublic.Laid{inside,laid_shaped,restored,counted,pinned}}}

# ---- The bias gradient of the output gradient of a call ----

# The output gradient of a call as an image geometry.
def gradient_image(+mode: Call.Mode,+geometry: InputKernel.Geometry) -> Parameter.ImageGeometry:
  InputKernel.Geometry{c,+o,h,w,k,s,p,+gh,+gw} = geometry
  Parameter.ImageGeometry{Call.images(mode),o,gh,gw}

# The bias reader of a call over the reader of its output gradient.
def bias_read(~DL: Data,+d_context: DL,+mode: Call.Mode,+geometry: InputKernel.Geometry) -> Terms.BiasRead<DL>:
  Terms.BiasRead{d_context,Some{gradient_image(mode,geometry)},Gradient.group_length()}

# A bias gradient call: the bias gradient holds the bias reader and the output
# gradient comes back holding its reader.
def TiledBiasCall(~DL: Data,~d_value: DL -> U32 -> F32,+d_context: DL,+mode: Call.Mode,+geometry: InputKernel.Geometry,result: Tiled.Tiled & Tiled.Tiled) -> Data:
  BiasHeld(~DL,~d_value,bias_read(~DL,d_context,mode,geometry),Call.outputs_word(geometry),d_context,Call.input_shape(mode,geometry),result)

# The cells of the output gradient name its image geometry.
def gradient_named(+mode: Call.Mode,+geometry: InputKernel.Geometry) ->
  {Parameter.cell_image(Call.gradient_cell_shape(mode,geometry),Call.outputs_word(geometry)) == Some{gradient_image(mode,geometry)} : Maybe<&2,Parameter.ImageGeometry>}:
  match mode:
    case Call.Single{}:
      InputKernel.Geometry{c,+o,h,w,k,s,p,+gh,+gw} = geometry
      +image = {Parameter.ImageGeometry{1,o,gh,gw} : Parameter.ImageGeometry}
      %Equal.sym(Bool,U32.is_eq(Tiled.groups(o),Tiled.groups(o)),True{},Arithmetic.equal_reflexive(Tiled.groups(o))) :
        {Parameter.grouped(_,image) == Some{image} : Maybe<&2,Parameter.ImageGeometry>}
      {==}
    case Call.Batched{+batch}:
      InputKernel.Geometry{c,+o,h,w,k,s,p,+gh,+gw} = geometry
      +image = {Parameter.ImageGeometry{batch,o,gh,gw} : Parameter.ImageGeometry}
      %Equal.sym(Bool,U32.is_eq(Tiled.groups(o),Tiled.groups(o)),True{},Arithmetic.equal_reflexive(Tiled.groups(o))) :
        {Parameter.grouped(_,image) == Some{image} : Maybe<&2,Parameter.ImageGeometry>}
      {==}

# The output gradient of a call whose image sizes are positive holds values.
def gradient_nonempty(+mode: Call.Mode,+geometry: InputKernel.Geometry,+positive: {Parameter.positive(gradient_image(mode,geometry)) == True{} : Bool}) ->
  {Nat.is_lt(0n,ShapeLaws.elements(Call.input_shape(mode,geometry))) == True{} : Bool}:
  InputKernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +g = {InputKernel.Geometry{c,o,h,w,k,s,p,gh,gw} : InputKernel.Geometry}
  +image = gradient_image(mode,g)
  +sizes = ParameterCells.image_sizes(image,positive)
  +rows = U32.to_nat(gh)
  +columns = U32.to_nat(gw)
  %Call.gradient_count(mode,g) : {Nat.is_lt(0n,_) == True{} : Bool}
  Order.product_positive(U32.to_nat(Call.images(mode)),Nat.mul(U32.to_nat(o),Nat.mul(rows,columns)),ParameterCells.sizes_count(image,sizes),
    Order.product_positive(U32.to_nat(o),Nat.mul(rows,columns),ParameterCells.sizes_channels(image,sizes),
      Order.product_positive(rows,columns,ParameterCells.sizes_height(image,sizes),ParameterCells.sizes_width(image,sizes))))

# What the cells of an image operand of a call fit: its lanes, a plane, its
# images and its groups.
type ImageRooms<-limits: Machine.Limits,-images: Nat,-channels: U32,-rows: Nat,-columns: Nat> is Data:
  ImageRooms{lanes_room: {Nat.is_le(U32.to_nat(channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    plane_room: {Nat.is_le(Nat.mul(rows,columns),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    outer_room: {Nat.is_le(Nat.mul(images,Nat.mul(rows,columns)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    groups_exact: {U32.to_nat(Tiled.groups(channels)) == Kernel.groups(U32.to_nat(channels)) : Nat},
    total_room: {Nat.is_le(Nat.mul(Nat.mul(images,Nat.mul(rows,columns)),Kernel.groups(U32.to_nat(channels))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

# Values and cells of positive images within the capacity give the rooms of
# the images.
def image_rooms(+limits: Machine.Limits,+images: Nat,+channels: U32,+rows: Nat,+columns: Nat,
  +images_positive: {Nat.is_lt(0n,images) == True{} : Bool},+channels_positive: {Nat.is_lt(0n,U32.to_nat(channels)) == True{} : Bool},
  +rows_positive: {Nat.is_lt(0n,rows) == True{} : Bool},+columns_positive: {Nat.is_lt(0n,columns) == True{} : Bool},
  +values_room: {Nat.is_le(Nat.mul(images,Nat.mul(U32.to_nat(channels),Nat.mul(rows,columns))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(images,rows),columns),U32.to_nat(Tiled.groups(channels))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  ImageRooms<limits,images,channels,rows,columns>:
  +outputs = U32.to_nat(channels)
  +plane = Nat.mul(rows,columns)
  +product = Nat.mul(outputs,plane)
  +cap = U32.to_nat(Machine.capacity(limits))
  +plane_positive = Order.product_positive(rows,columns,rows_positive,columns_positive)
  +product_room = Order.transitive(product,Nat.mul(images,product),cap,
    {%RowMajor.multiply_commutative(product,images) : {Nat.is_le(product,_) == True{} : Bool} RowMajor.product_covers(product,images,images_positive) :
      {Nat.is_le(product,Nat.mul(images,product)) == True{} : Bool}},values_room)
  +lanes_room = Order.transitive(outputs,product,cap,RowMajor.product_covers(outputs,plane,plane_positive),product_room)
  +plane_below = {%RowMajor.multiply_commutative(plane,outputs) : {Nat.is_le(plane,_) == True{} : Bool} RowMajor.product_covers(plane,outputs,channels_positive) :
    {Nat.is_le(plane,product) == True{} : Bool}}
  +outer_room = Order.transitive(Nat.mul(images,plane),Nat.mul(images,product),cap,RowMajor.mul_le_left(images,plane,product,plane_below),values_room)
  +plane_room = Order.transitive(plane,Nat.mul(images,plane),cap,
    {%RowMajor.multiply_commutative(plane,images) : {Nat.is_le(plane,_) == True{} : Bool} RowMajor.product_covers(plane,images,images_positive) :
      {Nat.is_le(plane,Nat.mul(images,plane)) == True{} : Bool}},outer_room)
  +groups = Tiled.groups(channels)
  +groups_exact = {Equal.sym(Nat,Bias.work_groups(Parameter.bias_work(Parameter.ImageGeometry{1,channels,1,1},Gradient.group_length())),U32.to_nat(groups),
      BiasCells.groups_from(Parameter.ImageGeometry{1,channels,1,1},Gradient.group_length(),limits,lanes_room)) :
    {U32.to_nat(groups) == Kernel.groups(outputs) : Nat}}
  +grouped_room = {%RowMajor.assoc(images,rows,columns) : {Nat.is_le(Nat.mul(_,U32.to_nat(groups)),cap) == True{} : Bool} cells_room :
    {Nat.is_le(Nat.mul(Nat.mul(images,plane),U32.to_nat(groups)),cap) == True{} : Bool}}
  ImageRooms{lanes_room,plane_room,outer_room,groups_exact,
    %groups_exact : {Nat.is_le(Nat.mul(Nat.mul(images,plane),_),cap) == True{} : Bool} grouped_room}

def rooms_lanes(+limits: Machine.Limits,-images: Nat,+channels: U32,-rows: Nat,-columns: Nat,+rooms: ImageRooms<limits,images,channels,rows,columns>) ->
  {Nat.is_le(U32.to_nat(channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  ImageRooms{+lanes_room,plane_room,outer_room,groups_exact,total_room} = rooms
  lanes_room

def rooms_plane(+limits: Machine.Limits,-images: Nat,-channels: U32,+rows: Nat,+columns: Nat,+rooms: ImageRooms<limits,images,channels,rows,columns>) ->
  {Nat.is_le(Nat.mul(rows,columns),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  ImageRooms{lanes_room,+plane_room,outer_room,groups_exact,total_room} = rooms
  plane_room

def rooms_outer(+limits: Machine.Limits,+images: Nat,-channels: U32,+rows: Nat,+columns: Nat,+rooms: ImageRooms<limits,images,channels,rows,columns>) ->
  {Nat.is_le(Nat.mul(images,Nat.mul(rows,columns)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  ImageRooms{lanes_room,plane_room,+outer_room,groups_exact,total_room} = rooms
  outer_room

def rooms_groups(-limits: Machine.Limits,-images: Nat,+channels: U32,-rows: Nat,-columns: Nat,+rooms: ImageRooms<limits,images,channels,rows,columns>) ->
  {U32.to_nat(Tiled.groups(channels)) == Kernel.groups(U32.to_nat(channels)) : Nat}:
  ImageRooms{lanes_room,plane_room,outer_room,+groups_exact,total_room} = rooms
  groups_exact

def rooms_total(+limits: Machine.Limits,+images: Nat,+channels: U32,+rows: Nat,+columns: Nat,+rooms: ImageRooms<limits,images,channels,rows,columns>) ->
  {Nat.is_le(Nat.mul(Nat.mul(images,Nat.mul(rows,columns)),Kernel.groups(U32.to_nat(channels))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  ImageRooms{lanes_room,plane_room,outer_room,groups_exact,+total_room} = rooms
  total_room

# The cells of the output gradient of a call, laid out with the outputs in the
# lanes, give the rooms of its images.
def gradient_rooms(~DL: Data,~d_value: DL -> U32 -> F32,+d_context: DL,+mode: Call.Mode,+geometry: InputKernel.Geometry,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +positive: {Parameter.positive(gradient_image(mode,geometry)) == True{} : Bool},
  +dm: PiecesModel.Model<Tile.Tile8>,+df: Tile.Tile8,+dw: U32,
  +d_contract: TiledPublic.Contract(~DL,~d_value,d_context,Call.input_shape(mode,geometry),
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),Call.gradient_cell_shape(mode,geometry),df,Tensors.Usable{},dw},Conv.lane_of(Call.batched(mode)),
      Grid.lanes_at(Call.input_shape(mode,geometry),Conv.lane_of(Call.batched(mode))),Grid.Innermost{}})) ->
  ImageRooms<limits,U32.to_nat(Call.images(mode)),Call.outputs_word(geometry),Domain.gradient_height(geometry),Domain.gradient_width(geometry)>:
  InputKernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  KernelGemm.Both{+global,TiledPublic.Laid{+inside,+laid_shaped,+restored,+counted,+pinned}} = d_contract
  +g = {InputKernel.Geometry{c,o,h,w,k,s,p,gh,gw} : InputKernel.Geometry}
  +values = Call.input_shape(mode,g)
  +axis = Conv.lane_of(Call.batched(mode))
  +dcells = Call.gradient_cell_shape(mode,g)
  +d_own = {KernelGemm.Both{global,TiledPublic.Laid{inside,laid_shaped,restored,counted,pinned}} :
    TiledPublic.Contract(~DL,~d_value,d_context,values,
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),dcells,df,Tensors.Usable{},dw},axis,Grid.lanes_at(values,axis),Grid.Innermost{}})}
  +image = gradient_image(mode,g)
  +sizes = ParameterCells.image_sizes(image,positive)
  +images = U32.to_nat(Call.images(mode))
  +rows = U32.to_nat(gh)
  +columns = U32.to_nat(gw)
  +cap = U32.to_nat(Machine.capacity(limits))
  +values_room = {%Equal.sym(Nat,Nat.mul(images,Domain.gradient_size(g)),ShapeLaws.elements(values),Call.gradient_count(mode,g)) : {Nat.is_le(_,cap) == True{} : Bool}
      StorageCount.counted_room(limits,values,counted,default_capacity) :
    {Nat.is_le(Nat.mul(images,Nat.mul(U32.to_nat(o),Nat.mul(rows,columns))),cap) == True{} : Bool}}
  +cells_counted = Global.shaped_counted(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),
    Representation.Lifted{d_context,TiledPublic.geometry_of(values,axis,Grid.Innermost{})},dcells,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),dcells,df,Tensors.Usable{},dw},
    TiledPublic.cells_shaped(~DL,~d_value,d_context,values,axis,Grid.lanes_at(values,axis),Grid.Innermost{},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),dcells,df,Tensors.Usable{},dw},d_own))
  +cells_room = {%Call.gradient_cells_count(mode,g) : {Nat.is_le(_,cap) == True{} : Bool}
      StorageCount.counted_room(limits,dcells,cells_counted,default_capacity) :
    {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(images,rows),columns),U32.to_nat(Tiled.groups(o))),cap) == True{} : Bool}}
  image_rooms(limits,images,o,rows,columns,ParameterCells.sizes_count(image,sizes),ParameterCells.sizes_channels(image,sizes),ParameterCells.sizes_height(image,sizes),
    ParameterCells.sizes_width(image,sizes),values_room,cells_room)

# The bias gradient of the cells of an output gradient laid out with its
# outputs in the lanes, as the cells of a model.
def bias_modeled(~DL: Data,~d_value: DL -> U32 -> F32,+d_context: DL,+mode: Call.Mode,+geometry: InputKernel.Geometry,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +positive: {Parameter.positive(gradient_image(mode,geometry)) == True{} : Bool},
  +dm: PiecesModel.Model<Tile.Tile8>,+df: Tile.Tile8,+dw: U32,
  +d_contract: TiledPublic.Contract(~DL,~d_value,d_context,Call.input_shape(mode,geometry),
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),Call.gradient_cell_shape(mode,geometry),df,Tensors.Usable{},dw},Conv.lane_of(Call.batched(mode)),
      Grid.lanes_at(Call.input_shape(mode,geometry),Conv.lane_of(Call.batched(mode))),Grid.Innermost{}})) ->
  TiledBiasCall(~DL,~d_value,d_context,mode,geometry,
    Gradient.bias_cells(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),Call.gradient_cell_shape(mode,geometry),df,Tensors.Usable{},dw},Conv.lane_of(Call.batched(mode)),
      Grid.lanes_at(Call.input_shape(mode,geometry),Conv.lane_of(Call.batched(mode))),Grid.Innermost{}},Gradient.group_length())):
  InputKernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +g = {InputKernel.Geometry{c,o,h,w,k,s,p,gh,gw} : InputKernel.Geometry}
  +values = Call.input_shape(mode,g)
  +axis = Conv.lane_of(Call.batched(mode))
  +dcells = Call.gradient_cell_shape(mode,g)
  +images = U32.to_nat(Call.images(mode))
  +rows = U32.to_nat(gh)
  +columns = U32.to_nat(gw)
  +image = gradient_image(mode,g)
  +rooms = gradient_rooms(~DL,~d_value,d_context,mode,g,limits,default_capacity,positive,dm,df,dw,d_contract)
  +plane_room = rooms_plane(limits,images,o,rows,columns,rooms)
  +own = {%Call.gradient_lanes(mode,g) :
      TiledPublic.Contract(~DL,~d_value,d_context,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),dcells,df,Tensors.Usable{},dw},axis,_,Grid.Innermost{}})
      d_contract :
    TiledPublic.Contract(~DL,~d_value,d_context,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),dcells,df,Tensors.Usable{},dw},axis,o,Grid.Innermost{}})}
  %Equal.sym(U32,Grid.lanes_at(values,axis),o,Call.gradient_lanes(mode,g)) :
    TiledBiasCall(~DL,~d_value,d_context,mode,g,
      Gradient.bias_cells(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),dcells,df,Tensors.Usable{},dw},axis,_,Grid.Innermost{}},Gradient.group_length()))
  bias_call(~DL,~d_value,d_context,values,axis,Call.gradient_outer(mode,g),Call.gradient_after(g),limits,default_capacity,Call.images(mode),o,gh,gw,
    Call.gradient_geometry(mode,g),gradient_named(mode,g),rooms_lanes(limits,images,o,rows,columns,rooms),positive,Call.gradient_after_word(limits,g,plane_room),
    Call.gradient_outer_exact(limits,mode,g,plane_room,rooms_outer(limits,images,o,rows,columns,rooms)),rooms_total(limits,images,o,rows,columns,rooms),
    ParameterCells.sizes_width(image,ParameterCells.image_sizes(image,positive)),dm,df,dw,own)

# The output gradient arranged with its outputs in the lanes and the groups
# innermost, as the cells of a model.
def bias_arranged(~DL: Data,~d_value: DL -> U32 -> F32,+d_context: DL,+mode: Call.Mode,+geometry: InputKernel.Geometry,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +positive: {Parameter.positive(gradient_image(mode,geometry)) == True{} : Bool},
  gradient: Tiled.Tiled,+held: Relane.Arranged(~DL,~d_value,d_context,Call.input_shape(mode,geometry),gradient,Conv.lane_of(Call.batched(mode)),Grid.Innermost{})) ->
  TiledBiasCall(~DL,~d_value,d_context,mode,geometry,Gradient.bias_cells(Tiled.arranged(gradient,Conv.lane_of(Call.batched(mode)),Grid.Innermost{}),Gradient.group_length())):
  KernelGemm.Both{+form,+arranged} = held
  +dshape = Call.input_shape(mode,geometry)
  +axis = Conv.lane_of(Call.batched(mode))
  %Equal.sym(Tiled.Tiled,Tiled.arranged(gradient,axis,Grid.Innermost{}),
      Tiled.Tiled{Tiled.cells_of(Tiled.arranged(gradient,axis,Grid.Innermost{})),axis,Grid.lanes_at(dshape,axis),Grid.Innermost{}},form) :
    TiledBiasCall(~DL,~d_value,d_context,mode,geometry,Gradient.bias_cells(_,Gradient.group_length()))
  Relane.modeled(~DL,~d_value,(tiled => TiledBiasCall(~DL,~d_value,d_context,mode,geometry,Gradient.bias_cells(tiled,Gradient.group_length()))),
    d_context,dshape,Tiled.cells_of(Tiled.arranged(gradient,axis,Grid.Innermost{})),axis,Grid.lanes_at(dshape,axis),Grid.Innermost{},arranged,
    dm => df => dw => d_contract => bias_modeled(~DL,~d_value,d_context,mode,geometry,limits,default_capacity,positive,dm,df,dw,d_contract))

# The output gradient as the cells of a model, moved onto its outputs.
def bias_packed(~DL: Data,~d_value: DL -> U32 -> F32,+d_context: DL,+mode: Call.Mode,+geometry: InputKernel.Geometry,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +positive: {Parameter.positive(gradient_image(mode,geometry)) == True{} : Bool},
  +xm: PiecesModel.Model<Tile.Tile8>,+xs: List<&2,U32>,+xf: Tile.Tile8,+xw: U32,+xa: Nat,+xl: U32,+xp: Grid.Position,
  +packed: TiledPublic.Contract(~DL,~d_value,d_context,Call.input_shape(mode,geometry),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp})) ->
  TiledBiasCall(~DL,~d_value,d_context,mode,geometry,Gradient.bias(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp})):
  KernelGemm.Both{+global,TiledPublic.Laid{+inside,+shaped,+restored,+counted,+pinned}} = packed
  +dshape = Call.input_shape(mode,geometry)
  +axis = Conv.lane_of(Call.batched(mode))
  +own = {KernelGemm.Both{global,TiledPublic.Laid{inside,shaped,restored,counted,pinned}} : TiledPublic.Contract(~DL,~d_value,d_context,dshape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp})}
  +rank_room = Order.transitive(List.length(&2,U32,dshape),4n,U32.to_nat(Machine.capacity(limits)),Call.gradient_rank(mode,geometry),Call.four_room(limits))
  %Equal.sym(List<&2,U32>,Tiled.logical_shape(xs,xa,xl,xp),dshape,restored) :
    TiledBiasCall(~DL,~d_value,d_context,mode,geometry,
      Gradient.bias_cells(Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},Tiled.lane_axis(List.length(&2,U32,_)),Grid.Innermost{}),Gradient.group_length()))
  %Equal.sym(Nat,Tiled.lane_axis(List.length(&2,U32,dshape)),axis,Call.shape_axis(mode,geometry)) :
    TiledBiasCall(~DL,~d_value,d_context,mode,geometry,Gradient.bias_cells(Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},_,Grid.Innermost{}),Gradient.group_length()))
  bias_arranged(~DL,~d_value,d_context,mode,geometry,limits,default_capacity,positive,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},
    Relane.arranged_held(~DL,~d_value,d_context,dshape,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},axis,Grid.Innermost{},limits,default_capacity,Call.gradient_axis_inside(mode,geometry),
      gradient_nonempty(mode,geometry,positive),rank_room,own))

# The public bias gradient: an output gradient of the contract of the shape of
# a call whose image sizes are positive gives the bias gradient of its reader
# with the outputs in its lanes, and comes back holding its reader.
def tiled_bias(~DL: Data,~d_value: DL -> U32 -> F32,+d_context: DL,+mode: Call.Mode,+geometry: InputKernel.Geometry,
  gradient: Tiled.Tiled,+d_contract: TiledPublic.Contract(~DL,~d_value,d_context,Call.input_shape(mode,geometry),gradient),
  +positive: {Parameter.positive(gradient_image(mode,geometry)) == True{} : Bool}) ->
  TiledBiasCall(~DL,~d_value,d_context,mode,geometry,Gradient.bias(gradient)):
  TiledPublic.by_packed(~DL,~d_value,(tiled => TiledBiasCall(~DL,~d_value,d_context,mode,geometry,Gradient.bias(tiled))),d_context,Call.input_shape(mode,geometry),
    gradient,d_contract,
    xm => xs => xf => xw => xa => xl => xp => packed => bias_packed(~DL,~d_value,d_context,mode,geometry,Default.limits(),{==},positive,xm,xs,xf,xw,xa,xl,xp,packed))

# ---- The weight gradient of the cells of x and d ----

# The facts the weight bridge reads about the words of a call.
type WeightRooms<-limits: Machine.Limits,-n: U32,-c: U32,-h: U32,-w: U32,-o: U32,-rows: U32,-columns: U32,-k: U32,-g: U32> is Data:
  WeightRooms{x_groups_exact: {U32.to_nat(Tiled.groups(c)) == Kernel.groups(U32.to_nat(c)) : Nat},
    x_room: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(n),Nat.mul(U32.to_nat(h),U32.to_nat(w))),Kernel.groups(U32.to_nat(c))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    blocks_exact: {U32.to_nat(((c + 7) / 8 : U32)) == Ceiling.ceiling(U32.to_nat(c),8n) : Nat},
    d_groups_exact: {U32.to_nat(Tiled.groups(o)) == Kernel.groups(U32.to_nat(o)) : Nat},
    d_room: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(n),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))),Kernel.groups(U32.to_nat(o))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    channels_positive: {Nat.is_lt(0n,U32.to_nat(c)) == True{} : Bool},kernel_positive: {Nat.is_lt(0n,U32.to_nat(k)) == True{} : Bool},
    columns_positive: {Nat.is_lt(0n,U32.to_nat(columns)) == True{} : Bool},group_positive: {Nat.is_lt(0n,U32.to_nat(g)) == True{} : Bool},
    outputs_positive: {Nat.is_lt(0n,U32.to_nat(o)) == True{} : Bool},
    weights_room: {Nat.is_le(Nat.mul(U32.to_nat(o),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

# The taps of a call are positive and fit the capacity, and so do its
# outputs.
def rooms_taps_positive(-limits: Machine.Limits,-n: U32,+c: U32,-h: U32,-w: U32,-o: U32,-rows: U32,-columns: U32,+k: U32,-g: U32,
  +rooms: WeightRooms<limits,n,c,h,w,o,rows,columns,k,g>) -> {Nat.is_lt(0n,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))) == True{} : Bool}:
  WeightRooms{x_groups_exact,x_room,blocks_exact,d_groups_exact,d_room,+channels_positive,+kernel_positive,columns_positive,group_positive,outputs_positive,weights_room} = rooms
  Order.product_positive(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)),channels_positive,Order.product_positive(U32.to_nat(k),U32.to_nat(k),kernel_positive,kernel_positive))

def rooms_taps(+limits: Machine.Limits,-n: U32,+c: U32,-h: U32,-w: U32,+o: U32,-rows: U32,-columns: U32,+k: U32,-g: U32,
  +rooms: WeightRooms<limits,n,c,h,w,o,rows,columns,k,g>) ->
  {Nat.is_le(Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  WeightRooms{x_groups_exact,x_room,blocks_exact,d_groups_exact,d_room,channels_positive,kernel_positive,columns_positive,group_positive,+outputs_positive,+weights_room} = rooms
  +taps = Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))
  +outputs = U32.to_nat(o)
  Order.transitive(taps,Nat.mul(outputs,taps),U32.to_nat(Machine.capacity(limits)),
    {%RowMajor.multiply_commutative(taps,outputs) : {Nat.is_le(taps,_) == True{} : Bool} RowMajor.product_covers(taps,outputs,outputs_positive) :
      {Nat.is_le(taps,Nat.mul(outputs,taps)) == True{} : Bool}},weights_room)

def rooms_outputs(+limits: Machine.Limits,-n: U32,+c: U32,-h: U32,-w: U32,+o: U32,-rows: U32,-columns: U32,+k: U32,-g: U32,
  +rooms: WeightRooms<limits,n,c,h,w,o,rows,columns,k,g>) -> {Nat.is_le(U32.to_nat(o),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  WeightRooms{x_groups_exact,x_room,blocks_exact,d_groups_exact,d_room,+channels_positive,+kernel_positive,columns_positive,group_positive,outputs_positive,+weights_room} = rooms
  +taps = Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))
  Order.transitive(U32.to_nat(o),Nat.mul(U32.to_nat(o),taps),U32.to_nat(Machine.capacity(limits)),
    RowMajor.product_covers(U32.to_nat(o),taps,
      Order.product_positive(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)),channels_positive,Order.product_positive(U32.to_nat(k),U32.to_nat(k),kernel_positive,kernel_positive))),
    weights_room)

# A tap and an output below their counts name a tap and an output of the
# geometry of a call.
def tap_word_inside(+n: U32,+c: U32,+h: U32,+w: U32,+o: U32,+rows: U32,+columns: U32,+k: U32,+s: U32,+p: U32,+tap: Nat,+capacity: U32,
  +taps_room: {Nat.is_le(Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))),U32.to_nat(capacity)) == True{} : Bool},
  +below: {Nat.is_lt(tap,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))) == True{} : Bool}) ->
  {U32.is_lt(U32.from_nat(tap),Packing.geometry_taps(ParameterCells.call_geometry(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p))) ==
    True{} : Bool}:
  +taps = Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))
  +room = Order.transitive(tap,taps,U32.to_nat(capacity),Bounds.strict_le(tap,taps,below),taps_room)
  %Equal.sym(Bool,U32.is_lt(U32.from_nat(tap),U32.from_nat(taps)),Nat.is_lt(U32.to_nat(U32.from_nat(tap)),U32.to_nat(U32.from_nat(taps))),
      Comparison.lt(U32.from_nat(tap),U32.from_nat(taps))) :
    {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(tap)),tap,FullRange.roundtrip(tap,capacity,room)) : {Nat.is_lt(_,U32.to_nat(U32.from_nat(taps))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(taps)),taps,FullRange.roundtrip(taps,capacity,taps_room)) : {Nat.is_lt(tap,_) == True{} : Bool}
  below

def output_word_inside(+n: U32,+c: U32,+h: U32,+w: U32,+o: U32,+rows: U32,+columns: U32,+k: U32,+s: U32,+p: U32,+channel: Nat,+capacity: U32,
  +outputs_room: {Nat.is_le(U32.to_nat(o),U32.to_nat(capacity)) == True{} : Bool},+below: {Nat.is_lt(channel,U32.to_nat(o)) == True{} : Bool}) ->
  {U32.is_lt(U32.from_nat(channel),Groups.geometry_outputs(ParameterCells.call_geometry(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p))) ==
    True{} : Bool}:
  +room = Order.transitive(channel,U32.to_nat(o),U32.to_nat(capacity),Bounds.strict_le(channel,U32.to_nat(o),below),outputs_room)
  %Equal.sym(Bool,U32.is_lt(U32.from_nat(channel),o),Nat.is_lt(U32.to_nat(U32.from_nat(channel)),U32.to_nat(o)),Comparison.lt(U32.from_nat(channel),o)) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(channel)),channel,FullRange.roundtrip(channel,capacity,room)) : {Nat.is_lt(_,U32.to_nat(o)) == True{} : Bool}
  below

# A lane of a cell the weight kernel writes for the lifted x and d, below the
# outputs, is the weight the weight reader reads at its output and tap.
def weight_value_lane(~XL: Data,~x_value: XL -> U32 -> F32,~DL: Data,~d_value: DL -> U32 -> F32,+x_logical: XL,+x_outer: U32,+x_after: U32,+d_logical: DL,+d_outer: U32,
  +d_after: U32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +n: U32,+c: U32,+h: U32,+w: U32,+o: U32,+rows: U32,+columns: U32,+k: U32,+s: U32,+p: U32,+g: U32,
  +x_cells: List<&2,U32>,+x_named: {Parameter.cell_image(x_cells,c) == Some{Parameter.ImageGeometry{n,c,h,w}} : Maybe<&2,Parameter.ImageGeometry>},+x_counted: Global.Counted<x_cells>,
  +d_cells: List<&2,U32>,+d_named: {Parameter.cell_image(d_cells,o) == Some{Parameter.ImageGeometry{n,o,rows,columns}} : Maybe<&2,Parameter.ImageGeometry>},+d_counted: Global.Counted<d_cells>,
  +valid: {Parameter.valid_with(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,g,limits) == True{} : Bool},
  +rooms: WeightRooms<limits,n,c,h,w,o,rows,columns,k,g>,
  +x_after_exact: {U32.to_nat(x_after) == Nat.mul(U32.to_nat(h),U32.to_nat(w)) : Nat},
  +x_outer_exact: {U32.to_nat(x_outer) == Nat.mul(U32.to_nat(n),Nat.mul(U32.to_nat(h),U32.to_nat(w))) : Nat},
  +d_after_exact: {U32.to_nat(d_after) == Nat.mul(U32.to_nat(rows),U32.to_nat(columns)) : Nat},
  +d_outer_exact: {U32.to_nat(d_outer) == Nat.mul(U32.to_nat(n),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) : Nat},
  +xm: PiecesModel.Model<Tile.Tile8>,+xf: Tile.Tile8,+xw: U32,
  +x_shaped: Global.Shaped(~Tile.Tile8,~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),Representation.Lifted{x_logical,Cells.Cells{x_outer,x_after,c,Tiled.groups(c),Grid.Innermost{}}},
    x_cells,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),x_cells,xf,Tensors.Usable{},xw}),
  +dm: PiecesModel.Model<Tile.Tile8>,+df: Tile.Tile8,+dw: U32,
  +d_shaped: Global.Shaped(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),Representation.Lifted{d_logical,Cells.Cells{d_outer,d_after,o,Tiled.groups(o),Grid.Innermost{}}},
    d_cells,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),d_cells,df,Tensors.Usable{},dw}),
  +cell: Nat,+lane: Nat,+below: {Nat.is_lt(lane,8n) == True{} : Bool},
  +inside: {Nat.is_lt(Nat.add(lane,Nat.mul(Nat.div(cell,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),8n)),U32.to_nat(o)) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Buffer.value(~Tile.Tile8,U32.from_nat(cell),Buffer.contents(~Tile.Tile8,Pair.snd(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,
      WeightKernel.weight_parts(Parameter.work(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,g),U32.max(xw,dw),
        Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm))))))) ==
    Terms.weight_at(~XL,~x_value,~DL,~d_value,Terms.WeightRead{x_logical,d_logical,Some{Parameter.ImageGeometry{n,c,h,w}},Some{Parameter.ImageGeometry{n,o,rows,columns}},k,s,p,g},
      U32.from_nat(Nat.add(Nat.mul(Nat.add(lane,Nat.mul(Nat.div(cell,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),8n)),
        Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),Nat.mod(cell,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))))))) : F32}:
  WeightRooms{+x_groups_exact,+x_room,+blocks_exact,+d_groups_exact,+d_room,+channels_positive,+kernel_positive,+columns_positive,+group_positive,+outputs_positive,+weights_room} = rooms
  +xi = {Parameter.ImageGeometry{n,c,h,w} : Parameter.ImageGeometry}
  +di = {Parameter.ImageGeometry{n,o,rows,columns} : Parameter.ImageGeometry}
  +call = {ParameterCells.Call{limits,default_capacity,x_cells,c,xi,x_named,x_counted,d_cells,o,di,d_named,d_counted,k,s,p,g,valid} : ParameterCells.Call}
  +x_holder = {Representation.Lifted{x_logical,Cells.Cells{x_outer,x_after,c,Tiled.groups(c),Grid.Innermost{}}} : Representation.Lifted<XL>}
  +d_holder = {Representation.Lifted{d_logical,Cells.Cells{d_outer,d_after,o,Tiled.groups(o),Grid.Innermost{}}} : Representation.Lifted<DL>}
  +batch = ParameterCells.batch_of(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),x_holder,d_holder,call)
  +geometry = ParameterCells.call_geometry(xi,di,k,s,p)
  +capacity = Machine.capacity(limits)
  +cap = U32.to_nat(capacity)
  +taps = Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))
  +taps_positive = rooms_taps_positive(limits,n,c,h,w,o,rows,columns,k,g,rooms)
  +taps_room = rooms_taps(limits,n,c,h,w,o,rows,columns,k,g,rooms)
  +outputs_room = rooms_outputs(limits,n,c,h,w,o,rows,columns,k,g,rooms)
  +block = Nat.div(cell,taps)
  +tap_nat = Nat.mod(cell,taps)
  +channel = Nat.add(lane,Nat.mul(block,8n))
  +tap_below = DivisionOrder.remainder_small(cell,taps,taps_positive)
  +tap = U32.from_nat(tap_nat)
  +output = U32.from_nat(channel)
  +tap_inside = tap_word_inside(n,c,h,w,o,rows,columns,k,s,p,tap_nat,capacity,taps_room,tap_below)
  +output_inside = output_word_inside(n,c,h,w,o,rows,columns,k,s,p,channel,capacity,outputs_room,inside)
  +tap_back = FullRange.roundtrip(tap_nat,capacity,Order.transitive(tap_nat,taps,cap,Bounds.strict_le(tap_nat,taps,tap_below),taps_room))
  +channel_back = FullRange.roundtrip(channel,capacity,Order.transitive(channel,U32.to_nat(o),cap,Bounds.strict_le(channel,U32.to_nat(o),inside),outputs_room))
  +taps_back = Groups.taps_exact_of(geometry,ParameterCells.call_domain(xi,di,k,s,p,g,limits,default_capacity,valid))
  +index = Nat.add(Nat.mul(channel,taps),tap_nat)
  +index_room = Order.transitive(index,Nat.mul(U32.to_nat(o),taps),cap,
    Bounds.strict_le(index,Nat.mul(U32.to_nat(o),taps),Representation.index_inside(Grid.Outermost{},taps,U32.to_nat(o),tap_nat,channel,tap_below,inside)),weights_room)
  +index_back = FullRange.roundtrip(index,capacity,index_room)
  +levels = Images.lane_levels(Images.batch_images(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),batch),
    Images.held_images(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),
      Images.batch_images(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),batch),0n,batch,tap,output))
  +contents = Buffer.contents(~Tile.Tile8,Pair.snd(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,
    WeightKernel.weight_parts(Parameter.work(xi,di,k,s,p,g),U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)))))
  +cells = ParameterCells.weight_cells(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),
    x_holder,d_holder,call,U32.max(xw,dw),tap,output,tap_inside,output_inside,PiecesModel.pack(~Tile.Tile8,xm),xf,xw,x_shaped,PiecesModel.pack(~Tile.Tile8,dm),df,dw,d_shaped)
  +lane_eq = ParameterCells.value_lane(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),
    x_holder,d_holder,call,tap,output,
    WeightKernel.weight_parts(Parameter.work(xi,di,k,s,p,g),U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm))),
    cells,tap_inside,output_inside)
  +mod_eq = Equal.trans(Nat,Nat.mod(U32.to_nat(output),8n),Nat.mod(channel,8n),lane,Equal.cong(Nat,Nat,value => Nat.mod(value,8n),U32.to_nat(output),channel,channel_back),
    DivisionOrder.shifted_small_remainder(lane,8n,block,{==},below))
  +quotient_eq = DivisionOrder.shifted_small_quotient(lane,8n,block,{==},below)
  +cell_eq = {%Equal.sym(Nat,U32.to_nat(tap),tap_nat,tap_back) : {Nat.add(Nat.mul(Nat.div(U32.to_nat(output),8n),Groups.tap_count(geometry)),_) == cell : Nat}
      %Equal.sym(Nat,Groups.tap_count(geometry),taps,taps_back) : {Nat.add(Nat.mul(Nat.div(U32.to_nat(output),8n),_),tap_nat) == cell : Nat}
      %Equal.sym(Nat,U32.to_nat(output),channel,channel_back) : {Nat.add(Nat.mul(Nat.div(_,8n),taps),tap_nat) == cell : Nat}
      %Equal.sym(Nat,Nat.div(channel,8n),block,quotient_eq) : {Nat.add(Nat.mul(_,taps),tap_nat) == cell : Nat}
      Equal.sym(Nat,cell,Nat.add(Nat.mul(block,taps),tap_nat),DivisionOrder.quotient_exact(cell,taps,taps_positive)) :
    {Routes.cell_index(geometry,tap,output) == cell : Nat}}
  +left = {%cell_eq : {TileLaws.select_lane(lane,Buffer.value(~Tile.Tile8,U32.from_nat(_),contents)) == levels : F32}
      %mod_eq : {TileLaws.select_lane(_,Buffer.value(~Tile.Tile8,U32.from_nat(Routes.cell_index(geometry,tap,output)),contents)) == levels : F32}
      lane_eq :
    {TileLaws.select_lane(lane,Buffer.value(~Tile.Tile8,U32.from_nat(cell),contents)) == levels : F32}}
  +problem_geometry = {WeightKernel.Geometry{c,o,h,w,k,s,p,rows,columns,U32.from_nat(taps)} : WeightKernel.Geometry}
  +level_eq = Terms.weight_lane(~XL,~x_value,~DL,~d_value,x_logical,x_outer,x_after,c,Tiled.groups(c),d_logical,d_outer,d_after,o,Tiled.groups(o),c,o,h,w,k,s,p,rows,columns,
    U32.from_nat(taps),U32.to_nat(n),U32.to_nat(g),tap,output,capacity,x_after_exact,{==},x_groups_exact,x_outer_exact,x_room,blocks_exact,d_after_exact,{==},d_groups_exact,
    d_outer_exact,d_room,channels_positive,kernel_positive,columns_positive,group_positive,
    %Equal.sym(Nat,U32.to_nat(tap),tap_nat,tap_back) : {Nat.is_lt(_,taps) == True{} : Bool} tap_below,
    %Equal.sym(Nat,U32.to_nat(output),channel,channel_back) : {Nat.is_lt(_,U32.to_nat(o)) == True{} : Bool} inside,
    batch,{==},{==},{==},{==},{==})
  +right = {%Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,index_back) :
        {levels == Spec.weight(~Terms.Operand<XL,DL>,~Terms.operand_at(~XL,~x_value,~DL,~d_value),
          Terms.weight_problem(~XL,~DL,x_logical,d_logical,problem_geometry,U32.to_nat(n),U32.to_nat(g)),_) : F32}
      %tap_back :
        {levels == Spec.weight(~Terms.Operand<XL,DL>,~Terms.operand_at(~XL,~x_value,~DL,~d_value),
          Terms.weight_problem(~XL,~DL,x_logical,d_logical,problem_geometry,U32.to_nat(n),U32.to_nat(g)),Nat.add(Nat.mul(channel,taps),_)) : F32}
      %channel_back :
        {levels == Spec.weight(~Terms.Operand<XL,DL>,~Terms.operand_at(~XL,~x_value,~DL,~d_value),
          Terms.weight_problem(~XL,~DL,x_logical,d_logical,problem_geometry,U32.to_nat(n),U32.to_nat(g)),Nat.add(Nat.mul(_,taps),U32.to_nat(tap))) : F32}
      level_eq :
    {levels == Terms.weight_at(~XL,~x_value,~DL,~d_value,Terms.WeightRead{x_logical,d_logical,Some{xi},Some{di},k,s,p,g},U32.from_nat(index)) : F32}}
  Equal.trans(F32,TileLaws.select_lane(lane,Buffer.value(~Tile.Tile8,U32.from_nat(cell),contents)),levels,
    Terms.weight_at(~XL,~x_value,~DL,~d_value,Terms.WeightRead{x_logical,d_logical,Some{xi},Some{di},k,s,p,g},U32.from_nat(index)),left,right)

# The logical index of a lane of a cell of the weights, the groups outermost
# over the taps: the output by the taps and the tap.
def logical_at(+o: U32,+w_outer: U32,+w_after: U32,+cell: Nat,+taps: Nat,+channel: Nat,
  +after_exact: {Representation.after_count(Cells.Cells{w_outer,w_after,o,Tiled.groups(o),Grid.Outermost{}}) == taps : Nat},
  +tap_below: {Nat.is_lt(Nat.mod(cell,taps),taps) == True{} : Bool}) ->
  {Representation.logical_index(Cells.Cells{w_outer,w_after,o,Tiled.groups(o),Grid.Outermost{}},Nat.mod(cell,taps),channel) == Nat.add(Nat.mul(channel,taps),Nat.mod(cell,taps)) : Nat}:
  +geometry = {Cells.Cells{w_outer,w_after,o,Tiled.groups(o),Grid.Outermost{}} : Cells.Cells}
  +tap = Nat.mod(cell,taps)
  +lanes = U32.to_nat(o)
  %Equal.sym(Nat,Representation.after_count(geometry),taps,after_exact) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(tap,_),lanes),channel),_),Nat.mod(tap,_)) == Nat.add(Nat.mul(channel,taps),tap) : Nat}
  %Equal.sym(Nat,Nat.div(tap,taps),0n,DivisionOrder.small_quotient(tap,taps,tap_below)) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(_,lanes),channel),taps),Nat.mod(tap,taps)) == Nat.add(Nat.mul(channel,taps),tap) : Nat}
  %Equal.sym(Nat,Nat.mod(tap,taps),tap,DivisionOrder.small_remainder(tap,taps,tap_below)) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(0n,lanes),channel),taps),_) == Nat.add(Nat.mul(channel,taps),tap) : Nat}
  {==}

# A lane of a cell the weight kernel writes, kept below the outputs, is the
# lane of the cell of the lifted weight reader.
def weight_kept(~XL: Data,~x_value: XL -> U32 -> F32,~DL: Data,~d_value: DL -> U32 -> F32,+x_logical: XL,+x_outer: U32,+x_after: U32,+d_logical: DL,+d_outer: U32,
  +d_after: U32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +n: U32,+c: U32,+h: U32,+w: U32,+o: U32,+rows: U32,+columns: U32,+k: U32,+s: U32,+p: U32,+g: U32,
  +x_cells: List<&2,U32>,+x_named: {Parameter.cell_image(x_cells,c) == Some{Parameter.ImageGeometry{n,c,h,w}} : Maybe<&2,Parameter.ImageGeometry>},+x_counted: Global.Counted<x_cells>,
  +d_cells: List<&2,U32>,+d_named: {Parameter.cell_image(d_cells,o) == Some{Parameter.ImageGeometry{n,o,rows,columns}} : Maybe<&2,Parameter.ImageGeometry>},+d_counted: Global.Counted<d_cells>,
  +valid: {Parameter.valid_with(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,g,limits) == True{} : Bool},
  +rooms: WeightRooms<limits,n,c,h,w,o,rows,columns,k,g>,
  +x_after_exact: {U32.to_nat(x_after) == Nat.mul(U32.to_nat(h),U32.to_nat(w)) : Nat},
  +x_outer_exact: {U32.to_nat(x_outer) == Nat.mul(U32.to_nat(n),Nat.mul(U32.to_nat(h),U32.to_nat(w))) : Nat},
  +d_after_exact: {U32.to_nat(d_after) == Nat.mul(U32.to_nat(rows),U32.to_nat(columns)) : Nat},
  +d_outer_exact: {U32.to_nat(d_outer) == Nat.mul(U32.to_nat(n),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) : Nat},
  +xm: PiecesModel.Model<Tile.Tile8>,+xf: Tile.Tile8,+xw: U32,
  +x_shaped: Global.Shaped(~Tile.Tile8,~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),Representation.Lifted{x_logical,Cells.Cells{x_outer,x_after,c,Tiled.groups(c),Grid.Innermost{}}},
    x_cells,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),x_cells,xf,Tensors.Usable{},xw}),
  +dm: PiecesModel.Model<Tile.Tile8>,+df: Tile.Tile8,+dw: U32,
  +d_shaped: Global.Shaped(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),Representation.Lifted{d_logical,Cells.Cells{d_outer,d_after,o,Tiled.groups(o),Grid.Innermost{}}},
    d_cells,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),d_cells,df,Tensors.Usable{},dw}),
  +cell: Nat,+w_outer: U32,+w_after: U32,
  +after_exact: {Representation.after_count(Cells.Cells{w_outer,w_after,o,Tiled.groups(o),Grid.Outermost{}}) == Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))) : Nat},
  +lane: Nat,+below: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {PaddedCells.kept(U32.to_nat(o),Nat.div(cell,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),lane,
      TileLaws.select_lane(lane,Buffer.value(~Tile.Tile8,U32.from_nat(cell),Buffer.contents(~Tile.Tile8,Pair.snd(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,
        WeightKernel.weight_parts(Parameter.work(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,g),U32.max(xw,dw),
          Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)))))))) ==
    Representation.lane(~Terms.WeightRead<XL,DL>,~Terms.weight_at(~XL,~x_value,~DL,~d_value),
      Terms.WeightRead{x_logical,d_logical,Some{Parameter.ImageGeometry{n,c,h,w}},Some{Parameter.ImageGeometry{n,o,rows,columns}},k,s,p,g},
      Cells.Cells{w_outer,w_after,o,Tiled.groups(o),Grid.Outermost{}},Nat.mod(cell,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),
      Nat.div(cell,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),lane) : F32}:
  +taps = Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))
  +block = Nat.div(cell,taps)
  +tap = Nat.mod(cell,taps)
  +channel = Nat.add(lane,Nat.mul(block,8n))
  +read = {Terms.WeightRead{x_logical,d_logical,Some{Parameter.ImageGeometry{n,c,h,w}},Some{Parameter.ImageGeometry{n,o,rows,columns}},k,s,p,g} : Terms.WeightRead<XL,DL>}
  +tile = Buffer.value(~Tile.Tile8,U32.from_nat(cell),Buffer.contents(~Tile.Tile8,Pair.snd(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,
    WeightKernel.weight_parts(Parameter.work(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,g),U32.max(xw,dw),
      Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm))))))
  +tap_below = DivisionOrder.remainder_small(cell,taps,rooms_taps_positive(limits,n,c,h,w,o,rows,columns,k,g,rooms))
  +logical_eq = logical_at(o,w_outer,w_after,cell,taps,channel,after_exact,tap_below)
  %Equal.sym(Nat,Representation.logical_index(Cells.Cells{w_outer,w_after,o,Tiled.groups(o),Grid.Outermost{}},tap,channel),Nat.add(Nat.mul(channel,taps),tap),logical_eq) :
    {PaddedCells.kept(U32.to_nat(o),block,lane,TileLaws.select_lane(lane,tile)) ==
      Bool.pick(F32,Nat.is_lt(channel,U32.to_nat(o)),Terms.weight_at(~XL,~x_value,~DL,~d_value,read,U32.from_nat(_)),0.0) : F32}
  PaddedCells.kept_agree(U32.to_nat(o),block,lane,TileLaws.select_lane(lane,tile),Terms.weight_at(~XL,~x_value,~DL,~d_value,read,U32.from_nat(Nat.add(Nat.mul(channel,taps),tap))),
    Nat.is_lt(channel,U32.to_nat(o)),{==},
    inside => weight_value_lane(~XL,~x_value,~DL,~d_value,x_logical,x_outer,x_after,d_logical,d_outer,d_after,limits,default_capacity,n,c,h,w,o,rows,columns,k,s,p,g,
      x_cells,x_named,x_counted,d_cells,d_named,d_counted,valid,rooms,x_after_exact,x_outer_exact,d_after_exact,d_outer_exact,xm,xf,xw,x_shaped,dm,df,dw,d_shaped,cell,lane,below,inside))

# Every lane of a cell the weight kernel writes, kept below the outputs, is
# the lane of the cell of the lifted weight reader.
def weight_valid(~XL: Data,~x_value: XL -> U32 -> F32,~DL: Data,~d_value: DL -> U32 -> F32,+x_logical: XL,+x_outer: U32,+x_after: U32,+d_logical: DL,+d_outer: U32,
  +d_after: U32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +n: U32,+c: U32,+h: U32,+w: U32,+o: U32,+rows: U32,+columns: U32,+k: U32,+s: U32,+p: U32,+g: U32,
  +x_cells: List<&2,U32>,+x_named: {Parameter.cell_image(x_cells,c) == Some{Parameter.ImageGeometry{n,c,h,w}} : Maybe<&2,Parameter.ImageGeometry>},+x_counted: Global.Counted<x_cells>,
  +d_cells: List<&2,U32>,+d_named: {Parameter.cell_image(d_cells,o) == Some{Parameter.ImageGeometry{n,o,rows,columns}} : Maybe<&2,Parameter.ImageGeometry>},+d_counted: Global.Counted<d_cells>,
  +valid: {Parameter.valid_with(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,g,limits) == True{} : Bool},
  +rooms: WeightRooms<limits,n,c,h,w,o,rows,columns,k,g>,
  +x_after_exact: {U32.to_nat(x_after) == Nat.mul(U32.to_nat(h),U32.to_nat(w)) : Nat},
  +x_outer_exact: {U32.to_nat(x_outer) == Nat.mul(U32.to_nat(n),Nat.mul(U32.to_nat(h),U32.to_nat(w))) : Nat},
  +d_after_exact: {U32.to_nat(d_after) == Nat.mul(U32.to_nat(rows),U32.to_nat(columns)) : Nat},
  +d_outer_exact: {U32.to_nat(d_outer) == Nat.mul(U32.to_nat(n),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) : Nat},
  +xm: PiecesModel.Model<Tile.Tile8>,+xf: Tile.Tile8,+xw: U32,
  +x_shaped: Global.Shaped(~Tile.Tile8,~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),Representation.Lifted{x_logical,Cells.Cells{x_outer,x_after,c,Tiled.groups(c),Grid.Innermost{}}},
    x_cells,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),x_cells,xf,Tensors.Usable{},xw}),
  +dm: PiecesModel.Model<Tile.Tile8>,+df: Tile.Tile8,+dw: U32,
  +d_shaped: Global.Shaped(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),Representation.Lifted{d_logical,Cells.Cells{d_outer,d_after,o,Tiled.groups(o),Grid.Innermost{}}},
    d_cells,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),d_cells,df,Tensors.Usable{},dw}),
  +cell: Nat,+w_outer: U32,+w_after: U32,
  +after_exact: {Representation.after_count(Cells.Cells{w_outer,w_after,o,Tiled.groups(o),Grid.Outermost{}}) == Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))) : Nat}) ->
  PaddedCells.Valid<U32.to_nat(o),Nat.div(cell,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),
    Buffer.value(~Tile.Tile8,U32.from_nat(cell),Buffer.contents(~Tile.Tile8,Pair.snd(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,
      WeightKernel.weight_parts(Parameter.work(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,g),U32.max(xw,dw),
        Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)))))),
    Representation.cell(~Terms.WeightRead<XL,DL>,~Terms.weight_at(~XL,~x_value,~DL,~d_value),
      Terms.WeightRead{x_logical,d_logical,Some{Parameter.ImageGeometry{n,c,h,w}},Some{Parameter.ImageGeometry{n,o,rows,columns}},k,s,p,g},
      Cells.Cells{w_outer,w_after,o,Tiled.groups(o),Grid.Outermost{}},Nat.mod(cell,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),
      Nat.div(cell,Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))))>:
  PaddedCells.Valid{
    weight_kept(~XL,~x_value,~DL,~d_value,x_logical,x_outer,x_after,d_logical,d_outer,d_after,limits,default_capacity,n,c,h,w,o,rows,columns,k,s,p,g,
      x_cells,x_named,x_counted,d_cells,d_named,d_counted,valid,rooms,x_after_exact,x_outer_exact,d_after_exact,d_outer_exact,xm,xf,xw,x_shaped,dm,df,dw,d_shaped,
      cell,w_outer,w_after,after_exact,0n,{==}),
    weight_kept(~XL,~x_value,~DL,~d_value,x_logical,x_outer,x_after,d_logical,d_outer,d_after,limits,default_capacity,n,c,h,w,o,rows,columns,k,s,p,g,
      x_cells,x_named,x_counted,d_cells,d_named,d_counted,valid,rooms,x_after_exact,x_outer_exact,d_after_exact,d_outer_exact,xm,xf,xw,x_shaped,dm,df,dw,d_shaped,
      cell,w_outer,w_after,after_exact,1n,{==}),
    weight_kept(~XL,~x_value,~DL,~d_value,x_logical,x_outer,x_after,d_logical,d_outer,d_after,limits,default_capacity,n,c,h,w,o,rows,columns,k,s,p,g,
      x_cells,x_named,x_counted,d_cells,d_named,d_counted,valid,rooms,x_after_exact,x_outer_exact,d_after_exact,d_outer_exact,xm,xf,xw,x_shaped,dm,df,dw,d_shaped,
      cell,w_outer,w_after,after_exact,2n,{==}),
    weight_kept(~XL,~x_value,~DL,~d_value,x_logical,x_outer,x_after,d_logical,d_outer,d_after,limits,default_capacity,n,c,h,w,o,rows,columns,k,s,p,g,
      x_cells,x_named,x_counted,d_cells,d_named,d_counted,valid,rooms,x_after_exact,x_outer_exact,d_after_exact,d_outer_exact,xm,xf,xw,x_shaped,dm,df,dw,d_shaped,
      cell,w_outer,w_after,after_exact,3n,{==}),
    weight_kept(~XL,~x_value,~DL,~d_value,x_logical,x_outer,x_after,d_logical,d_outer,d_after,limits,default_capacity,n,c,h,w,o,rows,columns,k,s,p,g,
      x_cells,x_named,x_counted,d_cells,d_named,d_counted,valid,rooms,x_after_exact,x_outer_exact,d_after_exact,d_outer_exact,xm,xf,xw,x_shaped,dm,df,dw,d_shaped,
      cell,w_outer,w_after,after_exact,4n,{==}),
    weight_kept(~XL,~x_value,~DL,~d_value,x_logical,x_outer,x_after,d_logical,d_outer,d_after,limits,default_capacity,n,c,h,w,o,rows,columns,k,s,p,g,
      x_cells,x_named,x_counted,d_cells,d_named,d_counted,valid,rooms,x_after_exact,x_outer_exact,d_after_exact,d_outer_exact,xm,xf,xw,x_shaped,dm,df,dw,d_shaped,
      cell,w_outer,w_after,after_exact,5n,{==}),
    weight_kept(~XL,~x_value,~DL,~d_value,x_logical,x_outer,x_after,d_logical,d_outer,d_after,limits,default_capacity,n,c,h,w,o,rows,columns,k,s,p,g,
      x_cells,x_named,x_counted,d_cells,d_named,d_counted,valid,rooms,x_after_exact,x_outer_exact,d_after_exact,d_outer_exact,xm,xf,xw,x_shaped,dm,df,dw,d_shaped,
      cell,w_outer,w_after,after_exact,6n,{==}),
    weight_kept(~XL,~x_value,~DL,~d_value,x_logical,x_outer,x_after,d_logical,d_outer,d_after,limits,default_capacity,n,c,h,w,o,rows,columns,k,s,p,g,
      x_cells,x_named,x_counted,d_cells,d_named,d_counted,valid,rooms,x_after_exact,x_outer_exact,d_after_exact,d_outer_exact,xm,xf,xw,x_shaped,dm,df,dw,d_shaped,
      cell,w_outer,w_after,after_exact,7n,{==})}

# The weights [o, c, k, k] have their outputs in the lanes and the groups
# outermost over the taps.
def weight_geometry(+o: U32,+c: U32,+k: U32) ->
  {TiledPublic.geometry_of([o,c,k,k],0n,Grid.Outermost{}) == Cells.Cells{(1 * Grid.product([c,k,k]) : U32),Grid.product([c,k,k]),o,Tiled.groups(o),Grid.Outermost{}} : Cells.Cells}:
  {==}

# A cell of the padded weights the kernel writes is the cell of the lifted
# weight reader.
def weight_fixed(~XL: Data,~x_value: XL -> U32 -> F32,~DL: Data,~d_value: DL -> U32 -> F32,+x_logical: XL,+x_outer: U32,+x_after: U32,+d_logical: DL,+d_outer: U32,
  +d_after: U32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +n: U32,+c: U32,+h: U32,+w: U32,+o: U32,+rows: U32,+columns: U32,+k: U32,+s: U32,+p: U32,+g: U32,
  +x_cells: List<&2,U32>,+x_named: {Parameter.cell_image(x_cells,c) == Some{Parameter.ImageGeometry{n,c,h,w}} : Maybe<&2,Parameter.ImageGeometry>},+x_counted: Global.Counted<x_cells>,
  +d_cells: List<&2,U32>,+d_named: {Parameter.cell_image(d_cells,o) == Some{Parameter.ImageGeometry{n,o,rows,columns}} : Maybe<&2,Parameter.ImageGeometry>},+d_counted: Global.Counted<d_cells>,
  +valid: {Parameter.valid_with(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,g,limits) == True{} : Bool},
  +rooms: WeightRooms<limits,n,c,h,w,o,rows,columns,k,g>,
  +x_after_exact: {U32.to_nat(x_after) == Nat.mul(U32.to_nat(h),U32.to_nat(w)) : Nat},
  +x_outer_exact: {U32.to_nat(x_outer) == Nat.mul(U32.to_nat(n),Nat.mul(U32.to_nat(h),U32.to_nat(w))) : Nat},
  +d_after_exact: {U32.to_nat(d_after) == Nat.mul(U32.to_nat(rows),U32.to_nat(columns)) : Nat},
  +d_outer_exact: {U32.to_nat(d_outer) == Nat.mul(U32.to_nat(n),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) : Nat},
  +xm: PiecesModel.Model<Tile.Tile8>,+xf: Tile.Tile8,+xw: U32,
  +x_shaped: Global.Shaped(~Tile.Tile8,~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),Representation.Lifted{x_logical,Cells.Cells{x_outer,x_after,c,Tiled.groups(c),Grid.Innermost{}}},
    x_cells,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),x_cells,xf,Tensors.Usable{},xw}),
  +dm: PiecesModel.Model<Tile.Tile8>,+df: Tile.Tile8,+dw: U32,
  +d_shaped: Global.Shaped(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),Representation.Lifted{d_logical,Cells.Cells{d_outer,d_after,o,Tiled.groups(o),Grid.Innermost{}}},
    d_cells,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),d_cells,df,Tensors.Usable{},dw}),
  +model: Buffer.Model<Tile.Tile8>,
  +equation: {Buffer.pack(~Tile.Tile8,model) == Pair.snd(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,
    WeightKernel.weight_parts(Parameter.work(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,g),U32.max(xw,dw),
      Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)))) : Storage.Buffer<Tile.Tile8>},
  +w_outer: U32,+w_after: U32,
  +after_exact: {Representation.after_count(Cells.Cells{w_outer,w_after,o,Tiled.groups(o),Grid.Outermost{}}) == Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))) : Nat},
  +outer_exact: {Representation.least_one(U32.to_nat(w_outer)) == Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))) : Nat},
  +outer_same: {(c * k * k : U32) == w_outer : U32},
  +count: U32,+cells_exact: {Nat.mul(U32.to_nat(Grid.groups(o)),U32.to_nat(w_outer)) == U32.to_nat(count) : Nat},
  +count_half: {Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +word: U32,+inside: {Nat.is_lt(U32.to_nat(word),U32.to_nat(count)) == True{} : Bool}) ->
  {PaddedPieces.padded_value(~Buffer.Content<Tile.Tile8>,~Contract.stored(~Tile.Tile8),
      PaddedPieces.Padded{TiledMap.Padding{o,Tiled.groups(o),(c * k * k : U32),Grid.Outermost{}},Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,model))},word) ==
    Representation.lifted(~Terms.WeightRead<XL,DL>,~Terms.weight_at(~XL,~x_value,~DL,~d_value),
      Representation.Lifted{Terms.WeightRead{x_logical,d_logical,Some{Parameter.ImageGeometry{n,c,h,w}},Some{Parameter.ImageGeometry{n,o,rows,columns}},k,s,p,g},
        Cells.Cells{w_outer,w_after,o,Tiled.groups(o),Grid.Outermost{}}},word) : Tile.Tile8}:
  +taps = Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))
  +number = U32.to_nat(word)
  +geometry = {Cells.Cells{w_outer,w_after,o,Tiled.groups(o),Grid.Outermost{}} : Cells.Cells}
  +read = {Terms.WeightRead{x_logical,d_logical,Some{Parameter.ImageGeometry{n,c,h,w}},Some{Parameter.ImageGeometry{n,o,rows,columns}},k,s,p,g} : Terms.WeightRead<XL,DL>}
  +tile = Buffer.value(~Tile.Tile8,U32.from_nat(number),Buffer.contents(~Tile.Tile8,Pair.snd(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,WeightKernel.weight_parts(Parameter.work(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,g),U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm))))))
  +target = Representation.lifted(~Terms.WeightRead<XL,DL>,~Terms.weight_at(~XL,~x_value,~DL,~d_value),Representation.Lifted{read,geometry},word)
  +padding = {TiledMap.Padding{o,Tiled.groups(o),(c * k * k : U32),Grid.Outermost{}} : TiledMap.Padding}
  +valid_cell = weight_valid(~XL,~x_value,~DL,~d_value,x_logical,x_outer,x_after,d_logical,d_outer,d_after,limits,default_capacity,n,c,h,w,o,rows,columns,k,s,p,g,
    x_cells,x_named,x_counted,d_cells,d_named,d_counted,valid,rooms,x_after_exact,x_outer_exact,d_after_exact,d_outer_exact,xm,xf,xw,x_shaped,dm,df,dw,d_shaped,
    number,w_outer,w_after,after_exact)
  +valid_word = {%Equal.sym(Nat,Representation.least_one(U32.to_nat(w_outer)),taps,outer_exact) :
      PaddedCells.Valid<U32.to_nat(o),Nat.div(number,_),tile,
        Representation.cell(~Terms.WeightRead<XL,DL>,~Terms.weight_at(~XL,~x_value,~DL,~d_value),read,geometry,Nat.mod(number,_),Nat.div(number,_))>
      valid_cell :
    PaddedCells.Valid<U32.to_nat(o),Representation.cell_group(geometry,number),tile,target>}
  +valid_index = {%Equal.sym(U32,U32.from_nat(number),word,Arithmetic.native_roundtrip(word)) :
      PaddedCells.Valid<U32.to_nat(o),Representation.cell_group(geometry,U32.to_nat(_)),tile,target>
      valid_word :
    PaddedCells.Valid<U32.to_nat(o),Representation.cell_group(geometry,U32.to_nat(U32.from_nat(number))),tile,target>}
  +padded = PaddedCells.padded_index(limits,o,(c * k * k : U32),Grid.Outermost{},w_outer,w_after,Grid.Outermost{},U32.to_nat(count),number,tile,target,
    TiledValues.grouped(limits,o,rooms_outputs(limits,n,c,h,w,o,rows,columns,k,g,rooms)),cells_exact,TiledPublic.Along{outer_same,{==}},count_half,inside,valid_index)
  %Equal.sym(Storage.Buffer<Tile.Tile8>,Buffer.pack(~Tile.Tile8,model),Pair.snd(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,WeightKernel.weight_parts(Parameter.work(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,g),U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)))),equation) :
    {PaddedPieces.padded_value(~Buffer.Content<Tile.Tile8>,~Contract.stored(~Tile.Tile8),PaddedPieces.Padded{padding,Buffer.contents(~Tile.Tile8,_)},word) == target : Tile.Tile8}
  padded

# The padded weights the kernel writes are a whole tensor of the lifted
# weight reader.
def weight_whole(~XL: Data,~x_value: XL -> U32 -> F32,~DL: Data,~d_value: DL -> U32 -> F32,+x_logical: XL,+x_outer: U32,+x_after: U32,+d_logical: DL,+d_outer: U32,
  +d_after: U32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +n: U32,+c: U32,+h: U32,+w: U32,+o: U32,+rows: U32,+columns: U32,+k: U32,+s: U32,+p: U32,+g: U32,
  +x_cells: List<&2,U32>,+x_named: {Parameter.cell_image(x_cells,c) == Some{Parameter.ImageGeometry{n,c,h,w}} : Maybe<&2,Parameter.ImageGeometry>},+x_counted: Global.Counted<x_cells>,
  +d_cells: List<&2,U32>,+d_named: {Parameter.cell_image(d_cells,o) == Some{Parameter.ImageGeometry{n,o,rows,columns}} : Maybe<&2,Parameter.ImageGeometry>},+d_counted: Global.Counted<d_cells>,
  +valid: {Parameter.valid_with(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,g,limits) == True{} : Bool},
  +rooms: WeightRooms<limits,n,c,h,w,o,rows,columns,k,g>,
  +x_after_exact: {U32.to_nat(x_after) == Nat.mul(U32.to_nat(h),U32.to_nat(w)) : Nat},
  +x_outer_exact: {U32.to_nat(x_outer) == Nat.mul(U32.to_nat(n),Nat.mul(U32.to_nat(h),U32.to_nat(w))) : Nat},
  +d_after_exact: {U32.to_nat(d_after) == Nat.mul(U32.to_nat(rows),U32.to_nat(columns)) : Nat},
  +d_outer_exact: {U32.to_nat(d_outer) == Nat.mul(U32.to_nat(n),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) : Nat},
  +xm: PiecesModel.Model<Tile.Tile8>,+xf: Tile.Tile8,+xw: U32,
  +x_shaped: Global.Shaped(~Tile.Tile8,~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),Representation.Lifted{x_logical,Cells.Cells{x_outer,x_after,c,Tiled.groups(c),Grid.Innermost{}}},
    x_cells,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),x_cells,xf,Tensors.Usable{},xw}),
  +dm: PiecesModel.Model<Tile.Tile8>,+df: Tile.Tile8,+dw: U32,
  +d_shaped: Global.Shaped(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),Representation.Lifted{d_logical,Cells.Cells{d_outer,d_after,o,Tiled.groups(o),Grid.Innermost{}}},
    d_cells,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),d_cells,df,Tensors.Usable{},dw}),
  +observation: Buffer.Observation<Tile.Tile8,Pair.snd(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,
    WeightKernel.weight_parts(Parameter.work(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,g),U32.max(xw,dw),
      Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm))))>) ->
  Global.Contract(~Tile.Tile8,~Representation.Lifted<Terms.WeightRead<XL,DL>>,~Representation.lifted(~Terms.WeightRead<XL,DL>,~Terms.weight_at(~XL,~x_value,~DL,~d_value)),
    Representation.Lifted{Terms.WeightRead{x_logical,d_logical,Some{Parameter.ImageGeometry{n,c,h,w}},Some{Parameter.ImageGeometry{n,o,rows,columns}},k,s,p,g},
      TiledPublic.geometry_of([o,c,k,k],0n,Grid.Outermost{})},
    Tensors.whole(~Tile.Tile8,TiledMap.padded(TiledMap.Padding{o,Tiled.groups(o),(c * k * k : U32),Grid.Outermost{}},0,Grid.product([Tiled.groups(o),c,k,k]),
      Pair.snd(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,
        WeightKernel.weight_parts(Parameter.work(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,g),U32.max(xw,dw),
          Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm))))),
      [Tiled.groups(o),c,k,k],Tiled.zero_tile(),Tensors.Usable{},U32.max(xw,dw))):
  WeightRooms{+x_groups_exact,+x_room,+blocks_exact,+d_groups_exact,+d_room,+channels_positive,+kernel_positive,+columns_positive,+group_positive,+outputs_positive,+weights_room} = rooms
  Buffer.Observation{+model,+equation} = observation
  +xi = {Parameter.ImageGeometry{n,c,h,w} : Parameter.ImageGeometry}
  +di = {Parameter.ImageGeometry{n,o,rows,columns} : Parameter.ImageGeometry}
  +call = {ParameterCells.Call{limits,default_capacity,x_cells,c,xi,x_named,x_counted,d_cells,o,di,d_named,d_counted,k,s,p,g,valid} : ParameterCells.Call}
  +x_holder = {Representation.Lifted{x_logical,Cells.Cells{x_outer,x_after,c,Tiled.groups(c),Grid.Innermost{}}} : Representation.Lifted<XL>}
  +d_holder = {Representation.Lifted{d_logical,Cells.Cells{d_outer,d_after,o,Tiled.groups(o),Grid.Innermost{}}} : Representation.Lifted<DL>}
  +read = {Terms.WeightRead{x_logical,d_logical,Some{xi},Some{di},k,s,p,g} : Terms.WeightRead<XL,DL>}
  +capacity = Machine.capacity(limits)
  +cap = U32.to_nat(capacity)
  +cn = U32.to_nat(c)
  +kn = U32.to_nat(k)
  +taps = Nat.mul(cn,Nat.mul(kn,kn))
  +taps_positive = rooms_taps_positive(limits,n,c,h,w,o,rows,columns,k,g,rooms)
  +taps_room = rooms_taps(limits,n,c,h,w,o,rows,columns,k,g,rooms)
  +outputs_room = rooms_outputs(limits,n,c,h,w,o,rows,columns,k,g,rooms)
  +after = Grid.product([c,k,k])
  +w_outer = (1 * after : U32)
  +geometry = {Cells.Cells{w_outer,after,o,Tiled.groups(o),Grid.Outermost{}} : Cells.Cells}
  +shape = {[Tiled.groups(o),c,k,k] : List<&2,U32>}
  +count = Grid.product(shape)
  +padding = {TiledMap.Padding{o,Tiled.groups(o),(c * k * k : U32),Grid.Outermost{}} : TiledMap.Padding}
  +elements_room = {%Equal.sym(Nat,Nat.mul(kn,1n),kn,Order.times_one(kn)) : {Nat.is_le(Nat.mul(cn,Nat.mul(kn,_)),cap) == True{} : Bool} taps_room :
    {Nat.is_le(ShapeLaws.elements([c,k,k]),cap) == True{} : Bool}}
  +after_elements = TiledPublic.product_exact(capacity,Limits.capacity_fits(limits),[c,k,k],elements_room)
  +after_taps = Equal.trans(Nat,U32.to_nat(after),ShapeLaws.elements([c,k,k]),taps,after_elements,
    Equal.cong(Nat,Nat,value => Nat.mul(cn,Nat.mul(kn,value)),Nat.mul(kn,1n),kn,Order.times_one(kn)))
  +outer_elements = {%Equal.sym(U32,(1 * after : U32),after,TiledPublic.one_times(after)) : {U32.to_nat(_) == ShapeLaws.elements([c,k,k]) : Nat} after_elements :
    {U32.to_nat(w_outer) == ShapeLaws.elements([c,k,k]) : Nat}}
  +outer_taps = {%Equal.sym(U32,(1 * after : U32),after,TiledPublic.one_times(after)) : {U32.to_nat(_) == taps : Nat} after_taps : {U32.to_nat(w_outer) == taps : Nat}}
  +after_exact = Equal.trans(Nat,Representation.least_one(U32.to_nat(after)),Representation.least_one(taps),taps,
    Equal.cong(Nat,Nat,value => Representation.least_one(value),U32.to_nat(after),taps,after_taps),Representation.least_positive(taps,taps_positive))
  +outer_exact = Equal.trans(Nat,Representation.least_one(U32.to_nat(w_outer)),Representation.least_one(taps),taps,
    Equal.cong(Nat,Nat,value => Representation.least_one(value),U32.to_nat(w_outer),taps,outer_taps),Representation.least_positive(taps,taps_positive))
  +outer_same = Arithmetic.to_nat_injective((c * k * k : U32),w_outer,
    Equal.trans(Nat,U32.to_nat((c * k * k : U32)),taps,U32.to_nat(w_outer),ParameterCells.taps_word(c,k,capacity,kernel_positive,taps_room),Equal.sym(Nat,U32.to_nat(w_outer),taps,outer_taps)))
  +cells_exact = Equal.trans(Nat,Nat.mul(U32.to_nat(Grid.groups(o)),U32.to_nat(w_outer)),ShapeLaws.elements(shape),U32.to_nat(count),
    Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(Grid.groups(o)),value),U32.to_nat(w_outer),ShapeLaws.elements([c,k,k]),outer_elements),ParameterCells.result_product(call))
  +count_room = {%ParameterCells.result_product(call) : {Nat.is_le(_,cap) == True{} : Bool} ParameterCells.result_room(call) : {Nat.is_le(U32.to_nat(count),cap) == True{} : Bool}}
  +count_half = Order.transitive(U32.to_nat(count),cap,U32.to_nat(Machine.half_word(limits)),count_room,
    Bounds.as_nat_le(capacity,Machine.half_word(limits),Limits.capacity_within_half(limits)))
  +bound_room = {%ParameterCells.result_product(call) : {Nat.is_le(ShapeLaws.elements(shape),_) == True{} : Bool} Order.reflexive(ShapeLaws.elements(shape)) :
    {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(count)) == True{} : Bool}}
  +tap_zero = tap_word_inside(n,c,h,w,o,rows,columns,k,s,p,0n,capacity,taps_room,taps_positive)
  +output_zero = output_word_inside(n,c,h,w,o,rows,columns,k,s,p,0n,capacity,outputs_room,outputs_positive)
  +cells = ParameterCells.weight_cells(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),
    x_holder,d_holder,call,U32.max(xw,dw),U32.from_nat(0n),U32.from_nat(0n),tap_zero,output_zero,PiecesModel.pack(~Tile.Tile8,xm),xf,xw,x_shaped,PiecesModel.pack(~Tile.Tile8,dm),df,dw,d_shaped)
  +length = {%Equal.sym(Storage.Buffer<Tile.Tile8>,Buffer.pack(~Tile.Tile8,model),Pair.snd(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,WeightKernel.weight_parts(Parameter.work(xi,di,k,s,p,g),U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)))),equation) :
      {Buffer.content_length(~Tile.Tile8,Buffer.contents(~Tile.Tile8,_)) == U32.to_nat(count) : Nat}
      ParameterCells.values_count(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),
        x_holder,d_holder,call,U32.from_nat(0n),U32.from_nat(0n),
        WeightKernel.weight_parts(Parameter.work(xi,di,k,s,p,g),U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm))),
        cells) :
    {Buffer.content_length(~Tile.Tile8,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,model))) == U32.to_nat(count) : Nat}}
  +ready = {%Equal.sym(Storage.Buffer<Tile.Tile8>,Buffer.pack(~Tile.Tile8,model),Pair.snd(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,WeightKernel.weight_parts(Parameter.work(xi,di,k,s,p,g),U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)))),equation) : Buffer.ContentReady<Tile.Tile8,Buffer.contents(~Tile.Tile8,_)>
      ParameterCells.values_ready(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),
        x_holder,d_holder,call,U32.from_nat(0n),U32.from_nat(0n),
        WeightKernel.weight_parts(Parameter.work(xi,di,k,s,p,g),U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm))),
        cells) :
    Buffer.ContentReady<Tile.Tile8,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,model))>}
  +content = Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,model))
  +before = {PaddedPieces.Padded{padding,content} : PaddedPieces.Padded<Buffer.Content<Tile.Tile8>>}
  +after_reader = {Representation.Lifted{read,geometry} : Representation.Lifted<Terms.WeightRead<XL,DL>>}
  +whole = PaddedPieces.padded_whole(~Buffer.Content<Tile.Tile8>,~Contract.stored(~Tile.Tile8),content,limits,padding,shape,Tiled.zero_tile(),U32.max(xw,dw),count,model,
    ParameterCells.result_counted(call),ParameterCells.result_product(call),ParameterCells.result_room(call),ParameterCells.result_nonempty(call),ParameterCells.result_fits(call),
    Joined.stored_holds(~Tile.Tile8,count,content,length,ready))
  +relabeled = Global.relabeled_below(~Tile.Tile8,~PaddedPieces.Padded<Buffer.Content<Tile.Tile8>>,~Representation.Lifted<Terms.WeightRead<XL,DL>>,
    ~PaddedPieces.padded_value(~Buffer.Content<Tile.Tile8>,~Contract.stored(~Tile.Tile8)),~Representation.lifted(~Terms.WeightRead<XL,DL>,~Terms.weight_at(~XL,~x_value,~DL,~d_value)),
    ~(left => right => Global.Fixed(~PaddedPieces.Padded<Buffer.Content<Tile.Tile8>>,~Representation.Lifted<Terms.WeightRead<XL,DL>>,before,after_reader,left,right)),~U32.to_nat(count),
    ~(left => right => related => index => inside => Global.fixed_below(~Tile.Tile8,~PaddedPieces.Padded<Buffer.Content<Tile.Tile8>>,~Representation.Lifted<Terms.WeightRead<XL,DL>>,
      ~PaddedPieces.padded_value(~Buffer.Content<Tile.Tile8>,~Contract.stored(~Tile.Tile8)),~Representation.lifted(~Terms.WeightRead<XL,DL>,~Terms.weight_at(~XL,~x_value,~DL,~d_value)),
      ~U32.to_nat(count),before,after_reader,
      word => below => weight_fixed(~XL,~x_value,~DL,~d_value,x_logical,x_outer,x_after,d_logical,d_outer,d_after,limits,default_capacity,n,c,h,w,o,rows,columns,k,s,p,g,
        x_cells,x_named,x_counted,d_cells,d_named,d_counted,valid,rooms,x_after_exact,x_outer_exact,d_after_exact,d_outer_exact,xm,xf,xw,x_shaped,dm,df,dw,d_shaped,
        model,equation,w_outer,after,after_exact,outer_exact,outer_same,count,cells_exact,count_half,word,below),
      left,right,related,index,inside)),
    before,after_reader,KernelGemm.Both{{==},{==}},shape,
    Tensors.whole(~Tile.Tile8,TiledMap.padded(padding,0,count,Buffer.pack(~Tile.Tile8,model)),shape,Tiled.zero_tile(),Tensors.Usable{},U32.max(xw,dw)),
    Global.usable(~Tile.Tile8,Tensors.whole(~Tile.Tile8,TiledMap.padded(padding,0,count,Buffer.pack(~Tile.Tile8,model)),shape,Tiled.zero_tile(),Tensors.Usable{},U32.max(xw,dw)),{==}),
    whole,bound_room,capacity,count_room)
  %Equal.sym(Cells.Cells,TiledPublic.geometry_of([o,c,k,k],0n,Grid.Outermost{}),geometry,weight_geometry(o,c,k)) :
    Global.Contract(~Tile.Tile8,~Representation.Lifted<Terms.WeightRead<XL,DL>>,~Representation.lifted(~Terms.WeightRead<XL,DL>,~Terms.weight_at(~XL,~x_value,~DL,~d_value)),
      Representation.Lifted{read,_},Tensors.whole(~Tile.Tile8,TiledMap.padded(padding,0,count,Pair.snd(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,WeightKernel.weight_parts(Parameter.work(xi,di,k,s,p,g),U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm))))),shape,Tiled.zero_tile(),Tensors.Usable{},U32.max(xw,dw)))
  %equation :
    Global.Contract(~Tile.Tile8,~Representation.Lifted<Terms.WeightRead<XL,DL>>,~Representation.lifted(~Terms.WeightRead<XL,DL>,~Terms.weight_at(~XL,~x_value,~DL,~d_value)),
      after_reader,Tensors.whole(~Tile.Tile8,TiledMap.padded(padding,0,count,_),shape,Tiled.zero_tile(),Tensors.Usable{},U32.max(xw,dw)))
  relabeled

# A weight call on the cells of x and d runs the weight kernel, pads its cells
# whole with the groups outermost and gives x and d back in their grids.
def weight_run(+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +n: U32,+c: U32,+h: U32,+w: U32,+o: U32,+rows: U32,+columns: U32,+k: U32,+s: U32,+p: U32,+g: U32,
  +x_cells: List<&2,U32>,+x_named: {Parameter.cell_image(x_cells,c) == Some{Parameter.ImageGeometry{n,c,h,w}} : Maybe<&2,Parameter.ImageGeometry>},+x_counted: Global.Counted<x_cells>,
  +d_cells: List<&2,U32>,+d_named: {Parameter.cell_image(d_cells,o) == Some{Parameter.ImageGeometry{n,o,rows,columns}} : Maybe<&2,Parameter.ImageGeometry>},+d_counted: Global.Counted<d_cells>,
  +valid: {Parameter.valid_with(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,g,limits) == True{} : Bool},
  +xm: PiecesModel.Model<Tile.Tile8>,+xf: Tile.Tile8,+xw: U32,+x_axis: Nat,+dm: PiecesModel.Model<Tile.Tile8>,+df: Tile.Tile8,+dw: U32,+d_axis: Nat) ->
  {Gradient.weight_cells(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),x_cells,xf,Tensors.Usable{},xw},x_axis,c,Grid.Innermost{}},
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),d_cells,df,Tensors.Usable{},dw},d_axis,o,Grid.Innermost{}},k,s,p,g) ==
    (Tiled.Tiled{Tensors.whole(~Tile.Tile8,TiledMap.padded(TiledMap.Padding{o,Tiled.groups(o),(c * k * k : U32),Grid.Outermost{}},0,Grid.product([Tiled.groups(o),c,k,k]),
        Pair.snd(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,WeightKernel.weight_parts(Parameter.work(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,g),
          U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm))))),
        [Tiled.groups(o),c,k,k],Tiled.zero_tile(),Tensors.Usable{},U32.max(xw,dw)),0n,o,Grid.Outermost{}},
      (Tiled.Tiled{Parameter.back(Parameter.Read{x_cells,xf,Tensors.Usable{},xw},Pair.fst(Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>,
          Pair.fst(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,WeightKernel.weight_parts(Parameter.work(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,g),
            U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)))))),x_axis,c,Grid.Innermost{}},
        Tiled.Tiled{Parameter.back(Parameter.Read{d_cells,df,Tensors.Usable{},dw},Pair.snd(Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>,
          Pair.fst(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,WeightKernel.weight_parts(Parameter.work(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,g),
            U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)))))),d_axis,o,Grid.Innermost{}})) :
    Tiled.Tiled & (Tiled.Tiled & Tiled.Tiled)}:
  +xi = {Parameter.ImageGeometry{n,c,h,w} : Parameter.ImageGeometry}
  +di = {Parameter.ImageGeometry{n,o,rows,columns} : Parameter.ImageGeometry}
  +call = {ParameterCells.Call{limits,default_capacity,x_cells,c,xi,x_named,x_counted,d_cells,o,di,d_named,d_counted,k,s,p,g,valid} : ParameterCells.Call}
  %Equal.sym(Tiled.Tiled & (Tensors.Tensor<Tile.Tile8> & Tensors.Tensor<Tile.Tile8>),
      Parameter.weight(Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),x_cells,xf,Tensors.Usable{},xw},c,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),d_cells,df,Tensors.Usable{},dw},o,k,s,p,g),
      Parameter.weight_finished(Parameter.Read{x_cells,xf,Tensors.Usable{},xw},Parameter.Read{d_cells,df,Tensors.Usable{},dw},U32.max(xw,dw),o,(c * k * k : U32),[Tiled.groups(o),c,k,k],
        WeightKernel.weight_parts(Parameter.work(xi,di,k,s,p,g),U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)))),
      ParameterCells.run_exact(call,PiecesModel.pack(~Tile.Tile8,xm),xf,xw,PiecesModel.pack(~Tile.Tile8,dm),df,dw)) :
    {Gradient.weight_result(x_axis,c,d_axis,o,_) == (Tiled.Tiled{Tensors.whole(~Tile.Tile8,TiledMap.padded(TiledMap.Padding{o,Tiled.groups(o),(c * k * k : U32),Grid.Outermost{}},0,Grid.product([Tiled.groups(o),c,k,k]),
        Pair.snd(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,WeightKernel.weight_parts(Parameter.work(xi,di,k,s,p,g),
          U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm))))),
        [Tiled.groups(o),c,k,k],Tiled.zero_tile(),Tensors.Usable{},U32.max(xw,dw)),0n,o,Grid.Outermost{}},
      (Tiled.Tiled{Parameter.back(Parameter.Read{x_cells,xf,Tensors.Usable{},xw},Pair.fst(Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>,
          Pair.fst(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,WeightKernel.weight_parts(Parameter.work(xi,di,k,s,p,g),
            U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)))))),x_axis,c,Grid.Innermost{}},
        Tiled.Tiled{Parameter.back(Parameter.Read{d_cells,df,Tensors.Usable{},dw},Pair.snd(Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>,
          Pair.fst(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,WeightKernel.weight_parts(Parameter.work(xi,di,k,s,p,g),
            U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)))))),d_axis,o,Grid.Innermost{}})) : Tiled.Tiled & (Tiled.Tiled & Tiled.Tiled)}
  %Equal.sym(Tiled.Tiled & (Tensors.Tensor<Tile.Tile8> & Tensors.Tensor<Tile.Tile8>),
      Parameter.weight_finished(Parameter.Read{x_cells,xf,Tensors.Usable{},xw},Parameter.Read{d_cells,df,Tensors.Usable{},dw},U32.max(xw,dw),o,(c * k * k : U32),[Tiled.groups(o),c,k,k],
        WeightKernel.weight_parts(Parameter.work(xi,di,k,s,p,g),U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)))),
      (Tiled.Tiled{Tensors.whole(~Tile.Tile8,TiledMap.padded(TiledMap.Padding{o,Tiled.groups(o),(c * k * k : U32),Grid.Outermost{}},0,Grid.product([Tiled.groups(o),c,k,k]),
          Pair.snd(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,WeightKernel.weight_parts(Parameter.work(xi,di,k,s,p,g),
            U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm))))),
          [Tiled.groups(o),c,k,k],Tiled.zero_tile(),Tensors.Usable{},U32.max(xw,dw)),0n,o,Grid.Outermost{}},
        (Parameter.back(Parameter.Read{x_cells,xf,Tensors.Usable{},xw},Pair.fst(Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>,
            Pair.fst(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,WeightKernel.weight_parts(Parameter.work(xi,di,k,s,p,g),
              U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)))))),
          Parameter.back(Parameter.Read{d_cells,df,Tensors.Usable{},dw},Pair.snd(Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>,
            Pair.fst(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,WeightKernel.weight_parts(Parameter.work(xi,di,k,s,p,g),
              U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)))))))),
      ParameterCells.finished_exact(Parameter.Read{x_cells,xf,Tensors.Usable{},xw},Parameter.Read{d_cells,df,Tensors.Usable{},dw},U32.max(xw,dw),o,(c * k * k : U32),[Tiled.groups(o),c,k,k],
        WeightKernel.weight_parts(Parameter.work(xi,di,k,s,p,g),U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm))))) :
    {Gradient.weight_result(x_axis,c,d_axis,o,_) == (Tiled.Tiled{Tensors.whole(~Tile.Tile8,TiledMap.padded(TiledMap.Padding{o,Tiled.groups(o),(c * k * k : U32),Grid.Outermost{}},0,Grid.product([Tiled.groups(o),c,k,k]),
        Pair.snd(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,WeightKernel.weight_parts(Parameter.work(xi,di,k,s,p,g),
          U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm))))),
        [Tiled.groups(o),c,k,k],Tiled.zero_tile(),Tensors.Usable{},U32.max(xw,dw)),0n,o,Grid.Outermost{}},
      (Tiled.Tiled{Parameter.back(Parameter.Read{x_cells,xf,Tensors.Usable{},xw},Pair.fst(Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>,
          Pair.fst(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,WeightKernel.weight_parts(Parameter.work(xi,di,k,s,p,g),
            U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)))))),x_axis,c,Grid.Innermost{}},
        Tiled.Tiled{Parameter.back(Parameter.Read{d_cells,df,Tensors.Usable{},dw},Pair.snd(Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>,
          Pair.fst(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,WeightKernel.weight_parts(Parameter.work(xi,di,k,s,p,g),
            U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)))))),d_axis,o,Grid.Innermost{}})) : Tiled.Tiled & (Tiled.Tiled & Tiled.Tiled)}
  {==}

# ---- The weight gradient of a call ----

# The weight gradient, x and d a weight call returns hold the weight reader
# and the readers of x and d.
def WeightHeld(~XL: Data,~x_value: XL -> U32 -> F32,~DL: Data,~d_value: DL -> U32 -> F32,+read: Terms.WeightRead<XL,DL>,+weights: List<&2,U32>,+x_logical: XL,
  +x_values: List<&2,U32>,+d_logical: DL,+d_values: List<&2,U32>,result: Tiled.Tiled & (Tiled.Tiled & Tiled.Tiled)) -> Data:
  (weight,cells) = result
  (x,d) = cells
  KernelGemm.Both<TiledPublic.Contract(~Terms.WeightRead<XL,DL>,~Terms.weight_at(~XL,~x_value,~DL,~d_value),read,weights,weight),
    KernelGemm.Both<TiledPublic.Contract(~XL,~x_value,x_logical,x_values,x),TiledPublic.Contract(~DL,~d_value,d_logical,d_values,d)>>

# A weight call on the cells of x and d laid out with their channels in the
# lanes and the groups innermost.
def weight_call(~XL: Data,~x_value: XL -> U32 -> F32,~DL: Data,~d_value: DL -> U32 -> F32,+x_logical: XL,+x_values: List<&2,U32>,+x_axis: Nat,+x_outer: U32,+x_after: U32,
  +d_logical: DL,+d_values: List<&2,U32>,+d_axis: Nat,+d_outer: U32,+d_after: U32,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +n: U32,+c: U32,+h: U32,+w: U32,+o: U32,+rows: U32,+columns: U32,+k: U32,+s: U32,+p: U32,+g: U32,
  +x_geometry_exact: {TiledPublic.geometry_of(x_values,x_axis,Grid.Innermost{}) == Cells.Cells{x_outer,x_after,c,Tiled.groups(c),Grid.Innermost{}} : Cells.Cells},
  +x_named: {Parameter.cell_image(Tiled.cell_shape(x_values,x_axis,Grid.Innermost{}),c) == Some{Parameter.ImageGeometry{n,c,h,w}} : Maybe<&2,Parameter.ImageGeometry>},
  +d_geometry_exact: {TiledPublic.geometry_of(d_values,d_axis,Grid.Innermost{}) == Cells.Cells{d_outer,d_after,o,Tiled.groups(o),Grid.Innermost{}} : Cells.Cells},
  +d_named: {Parameter.cell_image(Tiled.cell_shape(d_values,d_axis,Grid.Innermost{}),o) == Some{Parameter.ImageGeometry{n,o,rows,columns}} : Maybe<&2,Parameter.ImageGeometry>},
  +valid: {Parameter.valid_with(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,g,limits) == True{} : Bool},
  +rooms: WeightRooms<limits,n,c,h,w,o,rows,columns,k,g>,
  +x_after_exact: {U32.to_nat(x_after) == Nat.mul(U32.to_nat(h),U32.to_nat(w)) : Nat},
  +x_outer_exact: {U32.to_nat(x_outer) == Nat.mul(U32.to_nat(n),Nat.mul(U32.to_nat(h),U32.to_nat(w))) : Nat},
  +d_after_exact: {U32.to_nat(d_after) == Nat.mul(U32.to_nat(rows),U32.to_nat(columns)) : Nat},
  +d_outer_exact: {U32.to_nat(d_outer) == Nat.mul(U32.to_nat(n),Nat.mul(U32.to_nat(rows),U32.to_nat(columns))) : Nat},
  +xm: PiecesModel.Model<Tile.Tile8>,+xf: Tile.Tile8,+xw: U32,
  +x_contract: TiledPublic.Contract(~XL,~x_value,x_logical,x_values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),Tiled.cell_shape(x_values,x_axis,Grid.Innermost{}),xf,Tensors.Usable{},xw},x_axis,c,Grid.Innermost{}}),
  +dm: PiecesModel.Model<Tile.Tile8>,+df: Tile.Tile8,+dw: U32,
  +d_contract: TiledPublic.Contract(~DL,~d_value,d_logical,d_values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),Tiled.cell_shape(d_values,d_axis,Grid.Innermost{}),df,Tensors.Usable{},dw},d_axis,o,Grid.Innermost{}})) ->
  WeightHeld(~XL,~x_value,~DL,~d_value,Terms.WeightRead{x_logical,d_logical,Some{Parameter.ImageGeometry{n,c,h,w}},Some{Parameter.ImageGeometry{n,o,rows,columns}},k,s,p,g},[o,c,k,k],
    x_logical,x_values,d_logical,d_values,Gradient.weight_cells(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),Tiled.cell_shape(x_values,x_axis,Grid.Innermost{}),xf,Tensors.Usable{},xw},x_axis,c,Grid.Innermost{}},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),Tiled.cell_shape(d_values,d_axis,Grid.Innermost{}),df,Tensors.Usable{},dw},d_axis,o,Grid.Innermost{}},k,s,p,g)):
  WeightRooms{+x_groups_exact,+x_room,+blocks_exact,+d_groups_exact,+d_room,+channels_positive,+kernel_positive,+columns_positive,+group_positive,+outputs_positive,+weights_room} = rooms
  KernelGemm.Both{+x_global,TiledPublic.Laid{+x_inside,+x_laid_shaped,+x_restored,+x_values_counted,+x_pinned}} = x_contract
  KernelGemm.Both{+d_global,TiledPublic.Laid{+d_inside,+d_laid_shaped,+d_restored,+d_values_counted,+d_pinned}} = d_contract
  +x_cells = Tiled.cell_shape(x_values,x_axis,Grid.Innermost{})
  +d_cells = Tiled.cell_shape(d_values,d_axis,Grid.Innermost{})
  +x_own = {KernelGemm.Both{x_global,TiledPublic.Laid{x_inside,x_laid_shaped,x_restored,x_values_counted,x_pinned}} :
    TiledPublic.Contract(~XL,~x_value,x_logical,x_values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),Tiled.cell_shape(x_values,x_axis,Grid.Innermost{}),xf,Tensors.Usable{},xw},x_axis,c,Grid.Innermost{}})}
  +d_own = {KernelGemm.Both{d_global,TiledPublic.Laid{d_inside,d_laid_shaped,d_restored,d_values_counted,d_pinned}} :
    TiledPublic.Contract(~DL,~d_value,d_logical,d_values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),Tiled.cell_shape(d_values,d_axis,Grid.Innermost{}),df,Tensors.Usable{},dw},d_axis,o,Grid.Innermost{}})}
  +x_holder = {Representation.Lifted{x_logical,Cells.Cells{x_outer,x_after,c,Tiled.groups(c),Grid.Innermost{}}} : Representation.Lifted<XL>}
  +d_holder = {Representation.Lifted{d_logical,Cells.Cells{d_outer,d_after,o,Tiled.groups(o),Grid.Innermost{}}} : Representation.Lifted<DL>}
  +x_shaped = {%x_geometry_exact :
      Global.Shaped(~Tile.Tile8,~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),Representation.Lifted{x_logical,_},x_cells,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),Tiled.cell_shape(x_values,x_axis,Grid.Innermost{}),xf,Tensors.Usable{},xw})
      TiledPublic.cells_shaped(~XL,~x_value,x_logical,x_values,x_axis,c,Grid.Innermost{},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),Tiled.cell_shape(x_values,x_axis,Grid.Innermost{}),xf,Tensors.Usable{},xw},x_own) :
    Global.Shaped(~Tile.Tile8,~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),x_holder,x_cells,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),Tiled.cell_shape(x_values,x_axis,Grid.Innermost{}),xf,Tensors.Usable{},xw})}
  +d_shaped = {%d_geometry_exact :
      Global.Shaped(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),Representation.Lifted{d_logical,_},d_cells,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),Tiled.cell_shape(d_values,d_axis,Grid.Innermost{}),df,Tensors.Usable{},dw})
      TiledPublic.cells_shaped(~DL,~d_value,d_logical,d_values,d_axis,o,Grid.Innermost{},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),Tiled.cell_shape(d_values,d_axis,Grid.Innermost{}),df,Tensors.Usable{},dw},d_own) :
    Global.Shaped(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),d_holder,d_cells,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),Tiled.cell_shape(d_values,d_axis,Grid.Innermost{}),df,Tensors.Usable{},dw})}
  +x_counted = Global.shaped_counted(~Tile.Tile8,~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),x_holder,x_cells,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),Tiled.cell_shape(x_values,x_axis,Grid.Innermost{}),xf,Tensors.Usable{},xw},x_shaped)
  +d_counted = Global.shaped_counted(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),d_holder,d_cells,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),Tiled.cell_shape(d_values,d_axis,Grid.Innermost{}),df,Tensors.Usable{},dw},d_shaped)
  +xi = {Parameter.ImageGeometry{n,c,h,w} : Parameter.ImageGeometry}
  +di = {Parameter.ImageGeometry{n,o,rows,columns} : Parameter.ImageGeometry}
  +call = {ParameterCells.Call{limits,default_capacity,x_cells,c,xi,x_named,x_counted,d_cells,o,di,d_named,d_counted,k,s,p,g,valid} : ParameterCells.Call}
  +capacity = Machine.capacity(limits)
  +tap_zero = tap_word_inside(n,c,h,w,o,rows,columns,k,s,p,0n,capacity,rooms_taps(limits,n,c,h,w,o,rows,columns,k,g,rooms),
    rooms_taps_positive(limits,n,c,h,w,o,rows,columns,k,g,rooms))
  +output_zero = output_word_inside(n,c,h,w,o,rows,columns,k,s,p,0n,capacity,rooms_outputs(limits,n,c,h,w,o,rows,columns,k,g,rooms),outputs_positive)
  +cells = ParameterCells.weight_cells(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),
    x_holder,d_holder,call,U32.max(xw,dw),U32.from_nat(0n),U32.from_nat(0n),tap_zero,output_zero,PiecesModel.pack(~Tile.Tile8,xm),xf,xw,x_shaped,PiecesModel.pack(~Tile.Tile8,dm),df,dw,d_shaped)
  +x_back = {%Equal.sym(Cells.Cells,TiledPublic.geometry_of(x_values,x_axis,Grid.Innermost{}),Cells.Cells{x_outer,x_after,c,Tiled.groups(c),Grid.Innermost{}},x_geometry_exact) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),Representation.Lifted{x_logical,_},Parameter.back(Parameter.Read{Tiled.cell_shape(x_values,x_axis,Grid.Innermost{}),xf,Tensors.Usable{},xw},Pair.fst(Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>,Pair.fst(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,WeightKernel.weight_parts(Parameter.work(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,g),U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)))))))
      ParameterCells.x_back(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),
        x_holder,d_holder,call,U32.from_nat(0n),U32.from_nat(0n),WeightKernel.weight_parts(Parameter.work(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,g),U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm))),cells,xf,xw) :
    Global.Contract(~Tile.Tile8,~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),
      Representation.Lifted{x_logical,TiledPublic.geometry_of(x_values,x_axis,Grid.Innermost{})},Parameter.back(Parameter.Read{Tiled.cell_shape(x_values,x_axis,Grid.Innermost{}),xf,Tensors.Usable{},xw},Pair.fst(Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>,Pair.fst(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,WeightKernel.weight_parts(Parameter.work(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,g),U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)))))))}
  +d_back = {%Equal.sym(Cells.Cells,TiledPublic.geometry_of(d_values,d_axis,Grid.Innermost{}),Cells.Cells{d_outer,d_after,o,Tiled.groups(o),Grid.Innermost{}},d_geometry_exact) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),Representation.Lifted{d_logical,_},Parameter.back(Parameter.Read{Tiled.cell_shape(d_values,d_axis,Grid.Innermost{}),df,Tensors.Usable{},dw},Pair.snd(Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>,Pair.fst(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,WeightKernel.weight_parts(Parameter.work(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,g),U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)))))))
      ParameterCells.d_back(~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),
        x_holder,d_holder,call,U32.from_nat(0n),U32.from_nat(0n),WeightKernel.weight_parts(Parameter.work(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,g),U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm))),cells,df,dw) :
    Global.Contract(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.lifted(~DL,~d_value),
      Representation.Lifted{d_logical,TiledPublic.geometry_of(d_values,d_axis,Grid.Innermost{})},Parameter.back(Parameter.Read{Tiled.cell_shape(d_values,d_axis,Grid.Innermost{}),df,Tensors.Usable{},dw},Pair.snd(Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>,Pair.fst(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,WeightKernel.weight_parts(Parameter.work(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,g),U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)))))))}
  %Equal.sym(Tiled.Tiled & (Tiled.Tiled & Tiled.Tiled),Gradient.weight_cells(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),Tiled.cell_shape(x_values,x_axis,Grid.Innermost{}),xf,Tensors.Usable{},xw},x_axis,c,Grid.Innermost{}},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),Tiled.cell_shape(d_values,d_axis,Grid.Innermost{}),df,Tensors.Usable{},dw},d_axis,o,Grid.Innermost{}},k,s,p,g),(Tiled.Tiled{Tensors.whole(~Tile.Tile8,TiledMap.padded(TiledMap.Padding{o,Tiled.groups(o),(c * k * k : U32),Grid.Outermost{}},0,Grid.product([Tiled.groups(o),c,k,k]),Pair.snd(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,WeightKernel.weight_parts(Parameter.work(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,g),U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm))))),[Tiled.groups(o),c,k,k],Tiled.zero_tile(),Tensors.Usable{},U32.max(xw,dw)),0n,o,Grid.Outermost{}},(Tiled.Tiled{Parameter.back(Parameter.Read{Tiled.cell_shape(x_values,x_axis,Grid.Innermost{}),xf,Tensors.Usable{},xw},Pair.fst(Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>,Pair.fst(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,WeightKernel.weight_parts(Parameter.work(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,g),U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)))))),x_axis,c,Grid.Innermost{}},Tiled.Tiled{Parameter.back(Parameter.Read{Tiled.cell_shape(d_values,d_axis,Grid.Innermost{}),df,Tensors.Usable{},dw},Pair.snd(Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>,Pair.fst(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,WeightKernel.weight_parts(Parameter.work(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,g),U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)))))),d_axis,o,Grid.Innermost{}})),
      weight_run(limits,default_capacity,n,c,h,w,o,rows,columns,k,s,p,g,x_cells,x_named,x_counted,d_cells,d_named,d_counted,valid,xm,xf,xw,x_axis,dm,df,dw,d_axis)) :
    WeightHeld(~XL,~x_value,~DL,~d_value,Terms.WeightRead{x_logical,d_logical,Some{xi},Some{di},k,s,p,g},[o,c,k,k],x_logical,x_values,d_logical,d_values,_)
  KernelGemm.Both{
    KernelGemm.Both{
      weight_whole(~XL,~x_value,~DL,~d_value,x_logical,x_outer,x_after,d_logical,d_outer,d_after,limits,default_capacity,n,c,h,w,o,rows,columns,k,s,p,g,
        x_cells,x_named,x_counted,d_cells,d_named,d_counted,valid,rooms,x_after_exact,x_outer_exact,d_after_exact,d_outer_exact,xm,xf,xw,x_shaped,dm,df,dw,d_shaped,
        Buffer.observe(~Tile.Tile8,Pair.snd(WeightKernel.Held(),Storage.Buffer<Tile.Tile8>,WeightKernel.weight_parts(Parameter.work(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,g),U32.max(xw,dw),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,xm)),Move.held(~Tile.Tile8,PiecesModel.pack(~Tile.Tile8,dm)))))),
      TiledPublic.Laid{{==},{==},{==},ParameterCells.valid_weights(xi,di,k,s,p,g,limits,valid),{==}}},
    KernelGemm.Both{KernelGemm.Both{x_back,TiledPublic.Laid{x_inside,x_laid_shaped,x_restored,x_values_counted,x_pinned}},
      KernelGemm.Both{d_back,TiledPublic.Laid{d_inside,d_laid_shaped,d_restored,d_values_counted,d_pinned}}}}

# The input of a call as an image geometry.
def input_image(+mode: Call.Mode,+geometry: InputKernel.Geometry) -> Parameter.ImageGeometry:
  InputKernel.Geometry{+c,o,+h,+w,k,s,p,gh,gw} = geometry
  Parameter.ImageGeometry{Call.images(mode),c,h,w}

# The cells of the input name its image geometry.
def input_named(+mode: Call.Mode,+geometry: InputKernel.Geometry) ->
  {Parameter.cell_image(Tiled.cell_shape(Gradient.input_shape(Call.arguments(mode,geometry)),Conv.lane_of(Call.batched(mode)),Grid.Innermost{}),Domain.channel_word(geometry)) ==
    Some{input_image(mode,geometry)} : Maybe<&2,Parameter.ImageGeometry>}:
  match mode:
    case Call.Single{}:
      InputKernel.Geometry{+c,o,+h,+w,k,s,p,gh,gw} = geometry
      +image = {Parameter.ImageGeometry{1,c,h,w} : Parameter.ImageGeometry}
      %Equal.sym(Bool,U32.is_eq(Tiled.groups(c),Tiled.groups(c)),True{},Arithmetic.equal_reflexive(Tiled.groups(c))) :
        {Parameter.grouped(_,image) == Some{image} : Maybe<&2,Parameter.ImageGeometry>}
      {==}
    case Call.Batched{+batch}:
      InputKernel.Geometry{+c,o,+h,+w,k,s,p,gh,gw} = geometry
      +image = {Parameter.ImageGeometry{batch,c,h,w} : Parameter.ImageGeometry}
      %Equal.sym(Bool,U32.is_eq(Tiled.groups(c),Tiled.groups(c)),True{},Arithmetic.equal_reflexive(Tiled.groups(c))) :
        {Parameter.grouped(_,image) == Some{image} : Maybe<&2,Parameter.ImageGeometry>}
      {==}

# The input of a call whose image sizes are positive holds values.
def input_nonempty(+mode: Call.Mode,+geometry: InputKernel.Geometry,+positive: {Parameter.positive(input_image(mode,geometry)) == True{} : Bool}) ->
  {Nat.is_lt(0n,ShapeLaws.elements(Gradient.input_shape(Call.arguments(mode,geometry)))) == True{} : Bool}:
  InputKernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +g = {InputKernel.Geometry{c,o,h,w,k,s,p,gh,gw} : InputKernel.Geometry}
  +image = input_image(mode,g)
  +sizes = ParameterCells.image_sizes(image,positive)
  +heights = U32.to_nat(h)
  +widths = U32.to_nat(w)
  %Call.image_count(mode,g) : {Nat.is_lt(0n,_) == True{} : Bool}
  Order.product_positive(U32.to_nat(Call.images(mode)),Nat.mul(U32.to_nat(c),Nat.mul(heights,widths)),ParameterCells.sizes_count(image,sizes),
    Order.product_positive(U32.to_nat(c),Nat.mul(heights,widths),ParameterCells.sizes_channels(image,sizes),
      Order.product_positive(heights,widths,ParameterCells.sizes_height(image,sizes),ParameterCells.sizes_width(image,sizes))))

# The cells of the input of a call, laid out with the channels in the lanes,
# give the rooms of its images.
def input_rooms(~XL: Data,~x_value: XL -> U32 -> F32,+x_context: XL,+mode: Call.Mode,+geometry: InputKernel.Geometry,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +positive: {Parameter.positive(input_image(mode,geometry)) == True{} : Bool},
  +xm: PiecesModel.Model<Tile.Tile8>,+xf: Tile.Tile8,+xw: U32,
  +x_contract: TiledPublic.Contract(~XL,~x_value,x_context,Gradient.input_shape(Call.arguments(mode,geometry)),
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),Tiled.cell_shape(Gradient.input_shape(Call.arguments(mode,geometry)),Conv.lane_of(Call.batched(mode)),Grid.Innermost{}),
      xf,Tensors.Usable{},xw},Conv.lane_of(Call.batched(mode)),Grid.lanes_at(Gradient.input_shape(Call.arguments(mode,geometry)),Conv.lane_of(Call.batched(mode))),Grid.Innermost{}})) ->
  ImageRooms<limits,U32.to_nat(Call.images(mode)),Domain.channel_word(geometry),Domain.height(geometry),Domain.width(geometry)>:
  InputKernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  KernelGemm.Both{+global,TiledPublic.Laid{+inside,+laid_shaped,+restored,+counted,+pinned}} = x_contract
  +g = {InputKernel.Geometry{c,o,h,w,k,s,p,gh,gw} : InputKernel.Geometry}
  +values = Gradient.input_shape(Call.arguments(mode,g))
  +axis = Conv.lane_of(Call.batched(mode))
  +xcells = Tiled.cell_shape(values,axis,Grid.Innermost{})
  +x_own = {KernelGemm.Both{global,TiledPublic.Laid{inside,laid_shaped,restored,counted,pinned}} :
    TiledPublic.Contract(~XL,~x_value,x_context,values,
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xcells,xf,Tensors.Usable{},xw},axis,Grid.lanes_at(values,axis),Grid.Innermost{}})}
  +image = input_image(mode,g)
  +sizes = ParameterCells.image_sizes(image,positive)
  +images = U32.to_nat(Call.images(mode))
  +heights = U32.to_nat(h)
  +widths = U32.to_nat(w)
  +cap = U32.to_nat(Machine.capacity(limits))
  +groups = Tiled.groups(c)
  +values_room = {%Equal.sym(Nat,Nat.mul(images,Domain.image_size(g)),ShapeLaws.elements(values),Call.image_count(mode,g)) : {Nat.is_le(_,cap) == True{} : Bool}
      StorageCount.counted_room(limits,values,counted,default_capacity) :
    {Nat.is_le(Nat.mul(images,Nat.mul(U32.to_nat(c),Nat.mul(heights,widths))),cap) == True{} : Bool}}
  +cells_counted = Global.shaped_counted(~Tile.Tile8,~Representation.Lifted<XL>,~Representation.lifted(~XL,~x_value),
    Representation.Lifted{x_context,TiledPublic.geometry_of(values,axis,Grid.Innermost{})},xcells,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xcells,xf,Tensors.Usable{},xw},
    TiledPublic.cells_shaped(~XL,~x_value,x_context,values,axis,Grid.lanes_at(values,axis),Grid.Innermost{},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xcells,xf,Tensors.Usable{},xw},x_own))
  +cells_room = {%Equal.sym(Nat,Nat.mul(Nat.mul(Nat.mul(images,heights),widths),U32.to_nat(groups)),Nat.mul(Nat.mul(images,heights),Nat.mul(widths,U32.to_nat(groups))),
        RowMajor.assoc(Nat.mul(images,heights),widths,U32.to_nat(groups))) :
      {Nat.is_le(_,cap) == True{} : Bool}
      %Call.image_cells_count(mode,g) : {Nat.is_le(_,cap) == True{} : Bool}
      %Call.image_cell_shape(mode,g) : {Nat.is_le(ShapeLaws.elements(_),cap) == True{} : Bool}
      StorageCount.counted_room(limits,xcells,cells_counted,default_capacity) :
    {Nat.is_le(Nat.mul(Nat.mul(Nat.mul(images,heights),widths),U32.to_nat(groups)),cap) == True{} : Bool}}
  image_rooms(limits,images,c,heights,widths,ParameterCells.sizes_count(image,sizes),ParameterCells.sizes_channels(image,sizes),ParameterCells.sizes_height(image,sizes),
    ParameterCells.sizes_width(image,sizes),values_room,cells_room)

# The rooms of the input and of the output gradient of a valid call give the
# facts the weight bridge reads.
def weight_rooms_of(+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +n: U32,+c: U32,+h: U32,+w: U32,+o: U32,+rows: U32,+columns: U32,+k: U32,+s: U32,+p: U32,
  +x_rooms: ImageRooms<limits,U32.to_nat(n),c,U32.to_nat(h),U32.to_nat(w)>,+d_rooms: ImageRooms<limits,U32.to_nat(n),o,U32.to_nat(rows),U32.to_nat(columns)>,
  +valid: {Parameter.valid_with(Parameter.ImageGeometry{n,c,h,w},Parameter.ImageGeometry{n,o,rows,columns},k,s,p,Gradient.group_length(),limits) == True{} : Bool}) ->
  WeightRooms<limits,n,c,h,w,o,rows,columns,k,Gradient.group_length()>:
  +xi = {Parameter.ImageGeometry{n,c,h,w} : Parameter.ImageGeometry}
  +di = {Parameter.ImageGeometry{n,o,rows,columns} : Parameter.ImageGeometry}
  +group = Gradient.group_length()
  +capacity = Machine.capacity(limits)
  +cap = U32.to_nat(capacity)
  +images = U32.to_nat(n)
  +positive = ParameterCells.valid_positive(xi,di,k,s,p,group,limits,valid)
  +x_sizes = ParameterCells.image_sizes(xi,ParameterCells.positive_input(xi,di,k,s,group,capacity,positive))
  +d_sizes = ParameterCells.image_sizes(di,ParameterCells.positive_gradient(xi,di,k,s,group,capacity,positive))
  +domain = ParameterCells.call_domain(xi,di,k,s,p,group,limits,default_capacity,valid)
  +on = U32.to_nat(o)
  +cn = U32.to_nat(c)
  +kn = U32.to_nat(k)
  +weights = {%Order.times_one(kn) : {Nat.is_le(Nat.mul(on,Nat.mul(cn,Nat.mul(kn,_))),cap) == True{} : Bool}
      StorageCount.counted_room(limits,[o,c,k,k],ParameterCells.valid_weights(xi,di,k,s,p,group,limits,valid),default_capacity) :
    {Nat.is_le(Nat.mul(on,Nat.mul(cn,Nat.mul(kn,kn))),cap) == True{} : Bool}}
  WeightRooms{rooms_groups(limits,images,c,U32.to_nat(h),U32.to_nat(w),x_rooms),rooms_total(limits,images,c,U32.to_nat(h),U32.to_nat(w),x_rooms),
    Encoded.eight_blocks_exact(limits,c,rooms_lanes(limits,images,c,U32.to_nat(h),U32.to_nat(w),x_rooms)),
    rooms_groups(limits,images,o,U32.to_nat(rows),U32.to_nat(columns),d_rooms),rooms_total(limits,images,o,U32.to_nat(rows),U32.to_nat(columns),d_rooms),
    ParameterCells.sizes_channels(xi,x_sizes),Groups.kernel_positive(ParameterCells.call_geometry(xi,di,k,s,p),domain),ParameterCells.sizes_width(di,d_sizes),
    ParameterCells.positive_group(xi,di,k,s,group,capacity,positive),ParameterCells.sizes_channels(di,d_sizes),weights}

# The weight reader of a call over the readers of its input and its output
# gradient.
def weight_read(~XL: Data,~DL: Data,+x_context: XL,+d_context: DL,+mode: Call.Mode,+geometry: InputKernel.Geometry) -> Terms.WeightRead<XL,DL>:
  Terms.WeightRead{x_context,d_context,Some{input_image(mode,geometry)},Some{gradient_image(mode,geometry)},Domain.kernel_word(geometry),Call.stride(geometry),
    Call.padding(geometry),Gradient.group_length()}

# A weight gradient call: the weight gradient holds the weight reader, and the
# input and the output gradient come back holding their readers.
def TiledWeightCall(~XL: Data,~x_value: XL -> U32 -> F32,~DL: Data,~d_value: DL -> U32 -> F32,+x_context: XL,+d_context: DL,+mode: Call.Mode,+geometry: InputKernel.Geometry,
  result: Tiled.Tiled & (Tiled.Tiled & Tiled.Tiled)) -> Data:
  WeightHeld(~XL,~x_value,~DL,~d_value,weight_read(~XL,~DL,x_context,d_context,mode,geometry),Call.weight_shape(geometry),x_context,Gradient.input_shape(Call.arguments(mode,geometry)),
    d_context,Call.input_shape(mode,geometry),result)

# The weight gradient of the cells of an input and an output gradient laid
# out with their channels in the lanes, as the cells of models.
def weight_modeled(~XL: Data,~x_value: XL -> U32 -> F32,~DL: Data,~d_value: DL -> U32 -> F32,+x_context: XL,+d_context: DL,+mode: Call.Mode,+geometry: InputKernel.Geometry,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +valid: {Parameter.valid_with(input_image(mode,geometry),gradient_image(mode,geometry),Domain.kernel_word(geometry),Call.stride(geometry),Call.padding(geometry),
    Gradient.group_length(),limits) == True{} : Bool},
  +xm: PiecesModel.Model<Tile.Tile8>,+xf: Tile.Tile8,+xw: U32,
  +x_contract: TiledPublic.Contract(~XL,~x_value,x_context,Gradient.input_shape(Call.arguments(mode,geometry)),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),Tiled.cell_shape(Gradient.input_shape(Call.arguments(mode,geometry)),Conv.lane_of(Call.batched(mode)),Grid.Innermost{}),xf,Tensors.Usable{},xw},Conv.lane_of(Call.batched(mode)),Grid.lanes_at(Gradient.input_shape(Call.arguments(mode,geometry)),Conv.lane_of(Call.batched(mode))),Grid.Innermost{}}),
  +dm: PiecesModel.Model<Tile.Tile8>,+df: Tile.Tile8,+dw: U32,
  +d_contract: TiledPublic.Contract(~DL,~d_value,d_context,Call.input_shape(mode,geometry),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),Tiled.cell_shape(Call.input_shape(mode,geometry),Conv.lane_of(Call.batched(mode)),Grid.Innermost{}),df,Tensors.Usable{},dw},Conv.lane_of(Call.batched(mode)),Grid.lanes_at(Call.input_shape(mode,geometry),Conv.lane_of(Call.batched(mode))),Grid.Innermost{}})) ->
  TiledWeightCall(~XL,~x_value,~DL,~d_value,x_context,d_context,mode,geometry,
    Gradient.weight_cells(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),Tiled.cell_shape(Gradient.input_shape(Call.arguments(mode,geometry)),Conv.lane_of(Call.batched(mode)),Grid.Innermost{}),xf,Tensors.Usable{},xw},Conv.lane_of(Call.batched(mode)),Grid.lanes_at(Gradient.input_shape(Call.arguments(mode,geometry)),Conv.lane_of(Call.batched(mode))),Grid.Innermost{}},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),Tiled.cell_shape(Call.input_shape(mode,geometry),Conv.lane_of(Call.batched(mode)),Grid.Innermost{}),df,Tensors.Usable{},dw},Conv.lane_of(Call.batched(mode)),Grid.lanes_at(Call.input_shape(mode,geometry),Conv.lane_of(Call.batched(mode))),Grid.Innermost{}},Domain.kernel_word(geometry),Call.stride(geometry),Call.padding(geometry),Gradient.group_length())):
  InputKernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +g = {InputKernel.Geometry{c,o,h,w,k,s,p,gh,gw} : InputKernel.Geometry}
  +xv = Gradient.input_shape(Call.arguments(mode,g))
  +dv = Call.input_shape(mode,g)
  +axis = Conv.lane_of(Call.batched(mode))
  +xcells = Tiled.cell_shape(xv,axis,Grid.Innermost{})
  +dcells = Tiled.cell_shape(dv,axis,Grid.Innermost{})
  +n = Call.images(mode)
  +images = U32.to_nat(n)
  +group = Gradient.group_length()
  +xi = input_image(mode,g)
  +di = gradient_image(mode,g)
  +capacity = Machine.capacity(limits)
  +positive = ParameterCells.valid_positive(xi,di,k,s,p,group,limits,valid)
  +x_rooms = input_rooms(~XL,~x_value,x_context,mode,g,limits,default_capacity,ParameterCells.positive_input(xi,di,k,s,group,capacity,positive),xm,xf,xw,x_contract)
  +d_rooms = gradient_rooms(~DL,~d_value,d_context,mode,g,limits,default_capacity,ParameterCells.positive_gradient(xi,di,k,s,group,capacity,positive),dm,df,dw,d_contract)
  +heights = U32.to_nat(h)
  +widths = U32.to_nat(w)
  +rows = U32.to_nat(gh)
  +columns = U32.to_nat(gw)
  +x_plane = rooms_plane(limits,images,c,heights,widths,x_rooms)
  +d_plane = rooms_plane(limits,images,o,rows,columns,d_rooms)
  +x_own = {%Call.image_lanes(mode,g) :
      TiledPublic.Contract(~XL,~x_value,x_context,xv,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xcells,xf,Tensors.Usable{},xw},axis,_,Grid.Innermost{}})
      x_contract :
    TiledPublic.Contract(~XL,~x_value,x_context,xv,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xcells,xf,Tensors.Usable{},xw},axis,c,Grid.Innermost{}})}
  +d_own = {%Call.gradient_lanes(mode,g) :
      TiledPublic.Contract(~DL,~d_value,d_context,dv,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),dcells,df,Tensors.Usable{},dw},axis,_,Grid.Innermost{}})
      d_contract :
    TiledPublic.Contract(~DL,~d_value,d_context,dv,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),dcells,df,Tensors.Usable{},dw},axis,o,Grid.Innermost{}})}
  %Equal.sym(U32,Grid.lanes_at(xv,axis),c,Call.image_lanes(mode,g)) :
    TiledWeightCall(~XL,~x_value,~DL,~d_value,x_context,d_context,mode,g,Gradient.weight_cells(
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xcells,xf,Tensors.Usable{},xw},axis,_,Grid.Innermost{}},
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),dcells,df,Tensors.Usable{},dw},axis,Grid.lanes_at(dv,axis),Grid.Innermost{}},k,s,p,group))
  %Equal.sym(U32,Grid.lanes_at(dv,axis),o,Call.gradient_lanes(mode,g)) :
    TiledWeightCall(~XL,~x_value,~DL,~d_value,x_context,d_context,mode,g,Gradient.weight_cells(
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xcells,xf,Tensors.Usable{},xw},axis,c,Grid.Innermost{}},
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),dcells,df,Tensors.Usable{},dw},axis,_,Grid.Innermost{}},k,s,p,group))
  weight_call(~XL,~x_value,~DL,~d_value,x_context,xv,axis,Call.image_outer(mode,g),Call.image_after(g),d_context,dv,axis,Call.gradient_outer(mode,g),Call.gradient_after(g),
    limits,default_capacity,n,c,h,w,o,gh,gw,k,s,p,group,Call.image_cells_geometry(mode,g),input_named(mode,g),Call.gradient_geometry(mode,g),gradient_named(mode,g),valid,
    weight_rooms_of(limits,default_capacity,n,c,h,w,o,gh,gw,k,s,p,x_rooms,d_rooms,valid),
    Call.image_after_word(limits,g,x_plane),Call.image_outer_exact(limits,mode,g,x_plane,rooms_outer(limits,images,c,heights,widths,x_rooms)),
    Call.gradient_after_word(limits,g,d_plane),Call.gradient_outer_exact(limits,mode,g,d_plane,rooms_outer(limits,images,o,rows,columns,d_rooms)),
    xm,xf,xw,x_own,dm,df,dw,d_own)

# The input and the output gradient arranged with their channels in the lanes
# and the groups innermost, as the cells of models.
def weight_arranged(~XL: Data,~x_value: XL -> U32 -> F32,~DL: Data,~d_value: DL -> U32 -> F32,+x_context: XL,+d_context: DL,+mode: Call.Mode,+geometry: InputKernel.Geometry,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +valid: {Parameter.valid_with(input_image(mode,geometry),gradient_image(mode,geometry),Domain.kernel_word(geometry),Call.stride(geometry),Call.padding(geometry),
    Gradient.group_length(),limits) == True{} : Bool},
  x: Tiled.Tiled,+x_held: Relane.Arranged(~XL,~x_value,x_context,Gradient.input_shape(Call.arguments(mode,geometry)),x,Conv.lane_of(Call.batched(mode)),Grid.Innermost{}),
  d: Tiled.Tiled,+d_held: Relane.Arranged(~DL,~d_value,d_context,Call.input_shape(mode,geometry),d,Conv.lane_of(Call.batched(mode)),Grid.Innermost{})) ->
  TiledWeightCall(~XL,~x_value,~DL,~d_value,x_context,d_context,mode,geometry,
    Gradient.weight_cells(Tiled.arranged(x,Conv.lane_of(Call.batched(mode)),Grid.Innermost{}),Tiled.arranged(d,Conv.lane_of(Call.batched(mode)),Grid.Innermost{}),
      Domain.kernel_word(geometry),Call.stride(geometry),Call.padding(geometry),Gradient.group_length())):
  KernelGemm.Both{+x_form,+x_arranged} = x_held
  KernelGemm.Both{+d_form,+d_arranged} = d_held
  +xv = Gradient.input_shape(Call.arguments(mode,geometry))
  +dv = Call.input_shape(mode,geometry)
  +axis = Conv.lane_of(Call.batched(mode))
  +k = Domain.kernel_word(geometry)
  +s = Call.stride(geometry)
  +p = Call.padding(geometry)
  +group = Gradient.group_length()
  %Equal.sym(Tiled.Tiled,Tiled.arranged(x,axis,Grid.Innermost{}),
      Tiled.Tiled{Tiled.cells_of(Tiled.arranged(x,axis,Grid.Innermost{})),axis,Grid.lanes_at(xv,axis),Grid.Innermost{}},x_form) :
    TiledWeightCall(~XL,~x_value,~DL,~d_value,x_context,d_context,mode,geometry,Gradient.weight_cells(_,Tiled.arranged(d,axis,Grid.Innermost{}),k,s,p,group))
  %Equal.sym(Tiled.Tiled,Tiled.arranged(d,axis,Grid.Innermost{}),
      Tiled.Tiled{Tiled.cells_of(Tiled.arranged(d,axis,Grid.Innermost{})),axis,Grid.lanes_at(dv,axis),Grid.Innermost{}},d_form) :
    TiledWeightCall(~XL,~x_value,~DL,~d_value,x_context,d_context,mode,geometry,Gradient.weight_cells(
      Tiled.Tiled{Tiled.cells_of(Tiled.arranged(x,axis,Grid.Innermost{})),axis,Grid.lanes_at(xv,axis),Grid.Innermost{}},_,k,s,p,group))
  Relane.modeled(~XL,~x_value,(tiled => TiledWeightCall(~XL,~x_value,~DL,~d_value,x_context,d_context,mode,geometry,Gradient.weight_cells(tiled,
      Tiled.Tiled{Tiled.cells_of(Tiled.arranged(d,axis,Grid.Innermost{})),axis,Grid.lanes_at(dv,axis),Grid.Innermost{}},k,s,p,group))),
    x_context,xv,Tiled.cells_of(Tiled.arranged(x,axis,Grid.Innermost{})),axis,Grid.lanes_at(xv,axis),Grid.Innermost{},x_arranged,
    xm => xf => xw => x_contract => Relane.modeled(~DL,~d_value,(tiled => TiledWeightCall(~XL,~x_value,~DL,~d_value,x_context,d_context,mode,geometry,Gradient.weight_cells(
        Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),Tiled.cell_shape(xv,axis,Grid.Innermost{}),xf,Tensors.Usable{},xw},axis,Grid.lanes_at(xv,axis),Grid.Innermost{}},
        tiled,k,s,p,group))),
      d_context,dv,Tiled.cells_of(Tiled.arranged(d,axis,Grid.Innermost{})),axis,Grid.lanes_at(dv,axis),Grid.Innermost{},d_arranged,
      dm => df => dw => d_contract => weight_modeled(~XL,~x_value,~DL,~d_value,x_context,d_context,mode,geometry,limits,default_capacity,valid,xm,xf,xw,x_contract,dm,df,dw,d_contract)))

# The input and the output gradient as the cells of models, moved onto their
# channels.
def weight_packed(~XL: Data,~x_value: XL -> U32 -> F32,~DL: Data,~d_value: DL -> U32 -> F32,+x_context: XL,+d_context: DL,+mode: Call.Mode,+geometry: InputKernel.Geometry,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +valid: {Parameter.valid_with(input_image(mode,geometry),gradient_image(mode,geometry),Domain.kernel_word(geometry),Call.stride(geometry),Call.padding(geometry),
    Gradient.group_length(),limits) == True{} : Bool},
  +xm: PiecesModel.Model<Tile.Tile8>,+xs: List<&2,U32>,+xf: Tile.Tile8,+xw: U32,+xa: Nat,+xl: U32,+xp: Grid.Position,
  +x_packed: TiledPublic.Contract(~XL,~x_value,x_context,Gradient.input_shape(Call.arguments(mode,geometry)),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp}),
  +dm: PiecesModel.Model<Tile.Tile8>,+ds: List<&2,U32>,+df: Tile.Tile8,+dw: U32,+da: Nat,+dl: U32,+dp: Grid.Position,
  +d_packed: TiledPublic.Contract(~DL,~d_value,d_context,Call.input_shape(mode,geometry),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,Tensors.Usable{},dw},da,dl,dp})) ->
  TiledWeightCall(~XL,~x_value,~DL,~d_value,x_context,d_context,mode,geometry,
    Gradient.weight(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,Tensors.Usable{},dw},da,dl,dp},Domain.kernel_word(geometry),Call.stride(geometry),Call.padding(geometry))):
  KernelGemm.Both{+x_global,TiledPublic.Laid{+x_inside,+x_shaped,+x_restored,+x_counted,+x_pinned}} = x_packed
  KernelGemm.Both{+d_global,TiledPublic.Laid{+d_inside,+d_shaped,+d_restored,+d_counted,+d_pinned}} = d_packed
  +xv = Gradient.input_shape(Call.arguments(mode,geometry))
  +dv = Call.input_shape(mode,geometry)
  +axis = Conv.lane_of(Call.batched(mode))
  +k = Domain.kernel_word(geometry)
  +s = Call.stride(geometry)
  +p = Call.padding(geometry)
  +group = Gradient.group_length()
  +positive = ParameterCells.valid_positive(input_image(mode,geometry),gradient_image(mode,geometry),k,s,p,group,limits,valid)
  +capacity = Machine.capacity(limits)
  +x_own = {KernelGemm.Both{x_global,TiledPublic.Laid{x_inside,x_shaped,x_restored,x_counted,x_pinned}} : TiledPublic.Contract(~XL,~x_value,x_context,xv,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp})}
  +d_own = {KernelGemm.Both{d_global,TiledPublic.Laid{d_inside,d_shaped,d_restored,d_counted,d_pinned}} : TiledPublic.Contract(~DL,~d_value,d_context,dv,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,Tensors.Usable{},dw},da,dl,dp})}
  +x_rank = Order.transitive(List.length(&2,U32,xv),4n,U32.to_nat(capacity),Call.image_rank(mode,geometry),Call.four_room(limits))
  +d_rank = Order.transitive(List.length(&2,U32,dv),4n,U32.to_nat(capacity),Call.gradient_rank(mode,geometry),Call.four_room(limits))
  %Equal.sym(List<&2,U32>,Tiled.logical_shape(xs,xa,xl,xp),xv,x_restored) :
    TiledWeightCall(~XL,~x_value,~DL,~d_value,x_context,d_context,mode,geometry,Gradient.weight_cells(Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},Tiled.lane_axis(List.length(&2,U32,_)),Grid.Innermost{}),
      Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,Tensors.Usable{},dw},da,dl,dp},Tiled.lane_axis(List.length(&2,U32,Tiled.logical_shape(ds,da,dl,dp))),Grid.Innermost{}),k,s,p,group))
  %Equal.sym(Nat,Tiled.lane_axis(List.length(&2,U32,xv)),axis,Call.image_shape_axis(mode,geometry)) :
    TiledWeightCall(~XL,~x_value,~DL,~d_value,x_context,d_context,mode,geometry,Gradient.weight_cells(Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},_,Grid.Innermost{}),
      Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,Tensors.Usable{},dw},da,dl,dp},Tiled.lane_axis(List.length(&2,U32,Tiled.logical_shape(ds,da,dl,dp))),Grid.Innermost{}),k,s,p,group))
  %Equal.sym(List<&2,U32>,Tiled.logical_shape(ds,da,dl,dp),dv,d_restored) :
    TiledWeightCall(~XL,~x_value,~DL,~d_value,x_context,d_context,mode,geometry,Gradient.weight_cells(Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},axis,Grid.Innermost{}),
      Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,Tensors.Usable{},dw},da,dl,dp},Tiled.lane_axis(List.length(&2,U32,_)),Grid.Innermost{}),k,s,p,group))
  %Equal.sym(Nat,Tiled.lane_axis(List.length(&2,U32,dv)),axis,Call.shape_axis(mode,geometry)) :
    TiledWeightCall(~XL,~x_value,~DL,~d_value,x_context,d_context,mode,geometry,Gradient.weight_cells(Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},axis,Grid.Innermost{}),
      Tiled.arranged(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,Tensors.Usable{},dw},da,dl,dp},_,Grid.Innermost{}),k,s,p,group))
  weight_arranged(~XL,~x_value,~DL,~d_value,x_context,d_context,mode,geometry,limits,default_capacity,valid,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},
    Relane.arranged_held(~XL,~x_value,x_context,xv,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},axis,Grid.Innermost{},limits,default_capacity,Call.image_axis_inside(mode,geometry),
      input_nonempty(mode,geometry,ParameterCells.positive_input(input_image(mode,geometry),gradient_image(mode,geometry),k,s,group,capacity,positive)),x_rank,x_own),
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,Tensors.Usable{},dw},da,dl,dp},
    Relane.arranged_held(~DL,~d_value,d_context,dv,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,Tensors.Usable{},dw},da,dl,dp},axis,Grid.Innermost{},limits,default_capacity,Call.gradient_axis_inside(mode,geometry),
      gradient_nonempty(mode,geometry,ParameterCells.positive_gradient(input_image(mode,geometry),gradient_image(mode,geometry),k,s,group,capacity,positive)),d_rank,d_own))

# The public weight gradient: an input and an output gradient of the contract
# of the shapes of a valid call give the weight gradient of their readers with
# the outputs in its lanes and the groups outermost, and come back holding
# their readers.
def tiled_weight(~XL: Data,~x_value: XL -> U32 -> F32,~DL: Data,~d_value: DL -> U32 -> F32,+x_context: XL,+d_context: DL,+mode: Call.Mode,+geometry: InputKernel.Geometry,
  x: Tiled.Tiled,+x_contract: TiledPublic.Contract(~XL,~x_value,x_context,Gradient.input_shape(Call.arguments(mode,geometry)),x),
  d: Tiled.Tiled,+d_contract: TiledPublic.Contract(~DL,~d_value,d_context,Call.input_shape(mode,geometry),d),
  +valid: {Parameter.valid(input_image(mode,geometry),gradient_image(mode,geometry),Domain.kernel_word(geometry),Call.stride(geometry),Call.padding(geometry),
    Gradient.group_length()) == True{} : Bool}) ->
  TiledWeightCall(~XL,~x_value,~DL,~d_value,x_context,d_context,mode,geometry,
    Gradient.weight(x,d,Domain.kernel_word(geometry),Call.stride(geometry),Call.padding(geometry))):
  TiledPublic.by_packed(~XL,~x_value,(tiled => TiledWeightCall(~XL,~x_value,~DL,~d_value,x_context,d_context,mode,geometry,
      Gradient.weight(tiled,d,Domain.kernel_word(geometry),Call.stride(geometry),Call.padding(geometry)))),
    x_context,Gradient.input_shape(Call.arguments(mode,geometry)),x,x_contract,
    xm => xs => xf => xw => xa => xl => xp => x_packed => TiledPublic.by_packed(~DL,~d_value,(tiled => TiledWeightCall(~XL,~x_value,~DL,~d_value,x_context,d_context,mode,geometry,
        Gradient.weight(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},tiled,Domain.kernel_word(geometry),Call.stride(geometry),Call.padding(geometry)))),
      d_context,Call.input_shape(mode,geometry),d,d_contract,
      dm => ds => df => dw => da => dl => dp => d_packed => weight_packed(~XL,~x_value,~DL,~d_value,x_context,d_context,mode,geometry,Default.limits(),{==},valid,
        xm,xs,xf,xw,xa,xl,xp,x_packed,dm,ds,df,dw,da,dl,dp,d_packed)))

# The output gradient of a valid call has positive image sizes.
def gradient_positive(+mode: Call.Mode,+geometry: InputKernel.Geometry,
  +valid: {Parameter.valid(input_image(mode,geometry),gradient_image(mode,geometry),Domain.kernel_word(geometry),Call.stride(geometry),Call.padding(geometry),
    Gradient.group_length()) == True{} : Bool}) -> {Parameter.positive(gradient_image(mode,geometry)) == True{} : Bool}:
  +k = Domain.kernel_word(geometry)
  +s = Call.stride(geometry)
  +group = Gradient.group_length()
  ParameterCells.positive_gradient(input_image(mode,geometry),gradient_image(mode,geometry),k,s,group,Machine.capacity(Default.limits()),
    ParameterCells.valid_positive(input_image(mode,geometry),gradient_image(mode,geometry),k,s,Call.padding(geometry),group,Default.limits(),valid))
