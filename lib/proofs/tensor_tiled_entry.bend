# The tiled form of row-major values: every piece of cells built from one
# buffer of values holds, at each cell, the lanes of those values the cell
# names, and +0 past the lane count.
import Base
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ../tensor_grid.bend as Grid
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_cells.bend as Cells
import ../machine_limits.bend as Machine
import ./storage_tree_model.bend as Tree
import ./storage_buffer_model.bend as Buffer
import ./tensor_contract.bend as Contract
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_values.bend as TiledValues
import ./tensor_tiled_indexed_values.bend as TiledIndexedValues
import ../tensor_tiled_indexed.bend as TiledIndexed
import ./traversal_indexed_values.bend as IndexedValues
import ./traversal_array_refinement.bend as ArrayLaws
import ./tensor_access_refinement.bend as AccessLaws
import ./kernel_tile_refinement.bend as TileLaws
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ../traversal_blocks.bend as Blocks
import ./tensor_allocation_depth.bend as Depth
import ./u32_full_range.bend as FullRange
import ./u32_division.bend as U32Division
import ./word_multiplication.bend as WordMultiplication
import ./word_arithmetic.bend as WordArithmetic
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./natural_addition.bend as Addition
import ./nat_order.bend as Order
import ./row_major_index.bend as RowMajor
import ./tensor_placed.bend as Placed
import ./storage_allocation_refinement.bend as AllocationLaws
import ./nat_division_order.bend as DivisionOrder
import ./nat_division.bend as Division
import ./tensor_pieces_content.bend as Content
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges
import ../kernel_gemm.bend as KernelGemm
import ./nat_to_u32_bounds.bend as Bounds
import ./machine_limit_bounds.bend as Limits
import ../tensor.bend as Tensors
import ../tensor_pieces.bend as Pieces
import ./storage_certificate.bend as Certificate
import ./storage_refinement.bend as StorageLaws
import ./traversal_certificate.bend as TraversalCertificate
import ./tensor_tiled_certificate.bend as TiledCertificate
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_pieces_each.bend as PiecesEach
import ./traversal_blocks_refinement.bend as BlocksLaws
import ./tensor_grid_ranges.bend as GridRanges
import ./tensor_storage_count.bend as StorageCount
import ./tensor_cell_room.bend as Rooms
import ./tensor_tiled_padded_cells.bend as PaddedCells
import ./tensor_tiled_public.bend as Public
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_count_refinement.bend as Count
import ./tensor_global_shape_count.bend as Global
import ../default_machine_limits.bend as Default
import ./tensor_move_readback.bend as Held
import ./tensor_task_runs.bend as Runs
import ./u32_comparison.bend as Comparison

# ---- The index a run reads ----

# The index of lane 0 of a group as a run reads it: the index of the outer
# index, then eight lanes of after values for every group before.
def run_form(+quotient: U32,+lanes: U32,+group: U32,+after: U32,+rest: U32) -> U32:
  ((quotient * lanes * after + rest : U32) + group * 8 * after : U32)

def run_natural(+quotient: Nat,+lanes: Nat,+group: Nat,+after: Nat,+rest: Nat) ->
  {Nat.add(Nat.add(Nat.mul(Nat.mul(quotient,lanes),after),rest),Nat.mul(Nat.mul(group,8n),after)) ==
    TiledIndexedValues.index_natural(quotient,lanes,group,after,rest) : Nat}:
  +front = Nat.mul(Nat.mul(quotient,lanes),after)
  +shift = Nat.mul(Nat.mul(group,8n),after)
  +target = TiledIndexedValues.index_natural(quotient,lanes,group,after,rest)
  Equal.trans(Nat,Nat.add(Nat.add(front,rest),shift),Nat.add(front,Nat.add(rest,shift)),target,Addition.associative(front,rest,shift),
    Equal.trans(Nat,Nat.add(front,Nat.add(rest,shift)),Nat.add(front,Nat.add(shift,rest)),target,
      Equal.cong(Nat,Nat,value => Nat.add(front,value),Nat.add(rest,shift),Nat.add(shift,rest),Order.commutative(rest,shift)),
      Equal.trans(Nat,Nat.add(front,Nat.add(shift,rest)),Nat.add(Nat.add(front,shift),rest),target,
        Equal.sym(Nat,Nat.add(Nat.add(front,shift),rest),Nat.add(front,Nat.add(shift,rest)),Addition.associative(front,shift,rest)),
        Equal.cong(Nat,Nat,value => Nat.add(value,rest),Nat.add(front,shift),Nat.mul(Nat.add(Nat.mul(quotient,lanes),Nat.mul(group,8n)),after),
          Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(quotient,lanes),Nat.mul(group,8n)),after),Nat.add(front,shift),
            RowMajor.distrib(Nat.mul(quotient,lanes),Nat.mul(group,8n),after))))))

def run_encoded(+quotient: Nat,+lanes: Nat,+group: Nat,+after: Nat,+rest: Nat) ->
  {run_form(U32.from_nat(quotient),U32.from_nat(lanes),U32.from_nat(group),U32.from_nat(after),U32.from_nat(rest)) ==
    U32.from_nat(TiledIndexedValues.index_natural(quotient,lanes,group,after,rest)) : U32}:
  +product = Nat.mul(quotient,lanes)
  +scaled = Nat.mul(product,after)
  +front = Nat.add(scaled,rest)
  +channel = Nat.mul(group,8n)
  +shift = Nat.mul(channel,after)
  +target = U32.from_nat(TiledIndexedValues.index_natural(quotient,lanes,group,after,rest))
  %Equal.sym(U32,(U32.from_nat(quotient) * U32.from_nat(lanes) : U32),U32.from_nat(product),WordMultiplication.native_encoded_multiply(quotient,lanes)) :
    {((_ * U32.from_nat(after) + U32.from_nat(rest) : U32) + U32.from_nat(group) * 8 * U32.from_nat(after) : U32) == target : U32}
  %Equal.sym(U32,(U32.from_nat(product) * U32.from_nat(after) : U32),U32.from_nat(scaled),WordMultiplication.native_encoded_multiply(product,after)) :
    {((_ + U32.from_nat(rest) : U32) + U32.from_nat(group) * 8 * U32.from_nat(after) : U32) == target : U32}
  %Equal.sym(U32,(U32.from_nat(scaled) + U32.from_nat(rest) : U32),U32.from_nat(front),WordArithmetic.native_add(scaled,rest)) :
    {(_ + U32.from_nat(group) * 8 * U32.from_nat(after) : U32) == target : U32}
  %Equal.sym(U32,(U32.from_nat(group) * 8 : U32),U32.from_nat(channel),WordMultiplication.native_encoded_multiply(group,8n)) :
    {(U32.from_nat(front) + _ * U32.from_nat(after) : U32) == target : U32}
  %Equal.sym(U32,(U32.from_nat(channel) * U32.from_nat(after) : U32),U32.from_nat(shift),WordMultiplication.native_encoded_multiply(channel,after)) :
    {(U32.from_nat(front) + _ : U32) == target : U32}
  %Equal.sym(U32,(U32.from_nat(front) + U32.from_nat(shift) : U32),U32.from_nat(Nat.add(front,shift)),WordArithmetic.native_add(front,shift)) :
    {_ == target : U32}
  Equal.cong(Nat,U32,U32.from_nat,Nat.add(front,shift),TiledIndexedValues.index_natural(quotient,lanes,group,after,rest),run_natural(quotient,lanes,group,after,rest))

def run_decoded(+quotient: U32,+lanes: U32,+group: U32,+after: U32,+rest: U32,+quotient_value: Nat,+group_value: Nat,+rest_value: Nat,
  +quotient_exact: {U32.to_nat(quotient) == quotient_value : Nat},+group_exact: {U32.to_nat(group) == group_value : Nat},
  +rest_exact: {U32.to_nat(rest) == rest_value : Nat}) ->
  {run_form(quotient,lanes,group,after,rest) ==
    U32.from_nat(TiledIndexedValues.index_natural(quotient_value,U32.to_nat(lanes),group_value,U32.to_nat(after),rest_value)) : U32}:
  %quotient_exact :
    {run_form(quotient,lanes,group,after,rest) == U32.from_nat(TiledIndexedValues.index_natural(_,U32.to_nat(lanes),group_value,U32.to_nat(after),rest_value)) : U32}
  %group_exact :
    {run_form(quotient,lanes,group,after,rest) == U32.from_nat(TiledIndexedValues.index_natural(U32.to_nat(quotient),U32.to_nat(lanes),_,U32.to_nat(after),rest_value)) : U32}
  %rest_exact :
    {run_form(quotient,lanes,group,after,rest) == U32.from_nat(TiledIndexedValues.index_natural(U32.to_nat(quotient),U32.to_nat(lanes),U32.to_nat(group),U32.to_nat(after),_)) : U32}
  +target = U32.from_nat(TiledIndexedValues.index_natural(U32.to_nat(quotient),U32.to_nat(lanes),U32.to_nat(group),U32.to_nat(after),U32.to_nat(rest)))
  %Arithmetic.native_roundtrip(quotient) : {run_form(_,lanes,group,after,rest) == target : U32}
  %Arithmetic.native_roundtrip(lanes) : {run_form(U32.from_nat(U32.to_nat(quotient)),_,group,after,rest) == target : U32}
  %Arithmetic.native_roundtrip(group) : {run_form(U32.from_nat(U32.to_nat(quotient)),U32.from_nat(U32.to_nat(lanes)),_,after,rest) == target : U32}
  %Arithmetic.native_roundtrip(after) :
    {run_form(U32.from_nat(U32.to_nat(quotient)),U32.from_nat(U32.to_nat(lanes)),U32.from_nat(U32.to_nat(group)),_,rest) == target : U32}
  %Arithmetic.native_roundtrip(rest) :
    {run_form(U32.from_nat(U32.to_nat(quotient)),U32.from_nat(U32.to_nat(lanes)),U32.from_nat(U32.to_nat(group)),U32.from_nat(U32.to_nat(after)),_) == target : U32}
  run_encoded(U32.to_nat(quotient),U32.to_nat(lanes),U32.to_nat(group),U32.to_nat(after),U32.to_nat(rest))

# ---- The lanes of a cell ----

# The index of lane 0 of a group of an outer index.
def cell_base(+outer: Nat,+after: U32,+lanes: U32,+group: Nat) -> Nat:
  +count = U32.to_nat(after)
  TiledIndexedValues.index_natural(Nat.div(outer,count),U32.to_nat(lanes),group,count,Nat.mod(outer,count))

def cell_spot(+outer: Nat,+after: U32,+lanes: U32,+group: Nat) -> Cells.Spot:
  Cells.Spot{U32.from_nat(Nat.mul(group,8n)),U32.from_nat(cell_base(outer,after,lanes,group)),after,lanes}

# Lane lane of the cell read at the spot of a group is the lane of the cell
# of the stored values there.
def entry_lane(+limits: Machine.Limits,+tree: Tree.Tree<F32>,+length: Nat,+outer_count: U32,+after: U32,+lanes: U32,+position: Grid.Position,+outer: Nat,+group: Nat,
  +lane: Nat,+facts: TiledValues.Groups<limits,lanes>,+group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(lanes))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(after)) == True{} : Bool}) ->
  {TiledValues.spot_value(tree,TiledValues.steps(cell_spot(outer,after,lanes,group),lane)) ==
    Representation.lane(~Buffer.Content<F32>,~Contract.stored(~F32),Buffer.Content{tree,length},Cells.Cells{outer_count,after,lanes,Grid.groups(lanes),position},outer,group,lane) : F32}:
  +count = U32.to_nat(after)
  +lane_total = U32.to_nat(lanes)
  +quotient = Nat.div(outer,count)
  +rest = Nat.mod(outer,count)
  +base = cell_base(outer,after,lanes,group)
  +channel = Nat.add(lane,Nat.mul(group,8n))
  +valid = Nat.is_lt(channel,lane_total)
  +target = Nat.add(Nat.mul(Nat.add(Nat.mul(quotient,lane_total),channel),count),rest)
  %Equal.sym(Cells.Spot,TiledValues.steps(cell_spot(outer,after,lanes,group),lane),
      Cells.Spot{U32.from_nat(channel),U32.from_nat(Nat.add(Nat.mul(lane,count),base)),after,lanes},TiledValues.stepped(Nat.mul(group,8n),base,after,lanes,lane)) :
    {TiledValues.spot_value(tree,_) ==
      Representation.lane(~Buffer.Content<F32>,~Contract.stored(~F32),Buffer.Content{tree,length},Cells.Cells{outer_count,after,lanes,Grid.groups(lanes),position},outer,group,lane) : F32}
  %Equal.sym(Nat,Representation.least_one(count),count,Representation.least_positive(count,positive)) :
    {Bool.pick(F32,U32.is_lt(U32.from_nat(channel),lanes),Tree.read(F32,tree,U32.from_nat(Nat.add(Nat.mul(lane,count),base))),0.0) ==
      Bool.pick(F32,valid,Tree.read(F32,tree,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(outer,_),lane_total),channel),_),Nat.mod(outer,_)))),0.0) : F32}
  %Equal.sym(Bool,U32.is_lt(U32.from_nat(channel),lanes),valid,
      TiledValues.channel_compare(limits,channel,lanes,TiledValues.channel_bound(limits,lanes,group,lane,facts,group_inside,lane_inside))) :
    {Bool.pick(F32,_,Tree.read(F32,tree,U32.from_nat(Nat.add(Nat.mul(lane,count),base))),0.0) == Bool.pick(F32,valid,Tree.read(F32,tree,U32.from_nat(target)),0.0) : F32}
  Equal.cong(Nat,F32,index => Bool.pick(F32,valid,Tree.read(F32,tree,U32.from_nat(index)),0.0),Nat.add(Nat.mul(lane,count),base),target,
    Equal.trans(Nat,Nat.add(Nat.mul(lane,count),base),Nat.add(base,Nat.mul(lane,count)),target,Order.commutative(Nat.mul(lane,count),base),
      TiledIndexedValues.lane_shift(quotient,lane_total,group,count,rest,lane)))

# The cell read at the spot of a group is the cell of the stored values.
def entry_cell(+limits: Machine.Limits,+tree: Tree.Tree<F32>,+length: Nat,+outer_count: U32,+after: U32,+lanes: U32,+position: Grid.Position,+outer: Nat,+group: Nat,
  +facts: TiledValues.Groups<limits,lanes>,+group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(lanes))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(after)) == True{} : Bool}) ->
  {TiledValues.spot_cell(tree,cell_spot(outer,after,lanes,group)) ==
    Representation.cell(~Buffer.Content<F32>,~Contract.stored(~F32),Buffer.Content{tree,length},Cells.Cells{outer_count,after,lanes,Grid.groups(lanes),position},outer,group) : Tile.Tile8}:
  TileLaws.tiles_equal(TiledValues.spot_cell(tree,cell_spot(outer,after,lanes,group)),
    Representation.cell(~Buffer.Content<F32>,~Contract.stored(~F32),Buffer.Content{tree,length},Cells.Cells{outer_count,after,lanes,Grid.groups(lanes),position},outer,group),
    entry_lane(limits,tree,length,outer_count,after,lanes,position,outer,group,0n,facts,group_inside,{==},positive),
    entry_lane(limits,tree,length,outer_count,after,lanes,position,outer,group,1n,facts,group_inside,{==},positive),
    entry_lane(limits,tree,length,outer_count,after,lanes,position,outer,group,2n,facts,group_inside,{==},positive),
    entry_lane(limits,tree,length,outer_count,after,lanes,position,outer,group,3n,facts,group_inside,{==},positive),
    entry_lane(limits,tree,length,outer_count,after,lanes,position,outer,group,4n,facts,group_inside,{==},positive),
    entry_lane(limits,tree,length,outer_count,after,lanes,position,outer,group,5n,facts,group_inside,{==},positive),
    entry_lane(limits,tree,length,outer_count,after,lanes,position,outer,group,6n,facts,group_inside,{==},positive),
    entry_lane(limits,tree,length,outer_count,after,lanes,position,outer,group,7n,facts,group_inside,{==},positive))

# ---- The cells of a block ----

# The run of block block of a piece whose first outer index is first reads,
# at group group of the block, the spot of that group of outer index
# first + block.
def block_cell(+tree: Tree.Tree<F32>,+after: U32,+lanes: U32,+groups: U32,+first: U32,+block: Nat,+group: Nat,+capacity: U32,
  +spreads: TiledIndexedValues.Spreads<after>,+outer_small: {Nat.is_le(Nat.add(U32.to_nat(first),block),U32.to_nat(capacity)) == True{} : Bool},
  +group_small: {Nat.is_le(group,U32.to_nat(capacity)) == True{} : Bool},+position: U32,+start: U32,+local: {(position - start : U32) == U32.from_nat(group) : U32}) ->
  {TiledValues.run_value(tree,Cells.group_run(Cells.Tiling{after,lanes,groups,first},U32.from_nat(block),start),position) ==
    TiledValues.spot_cell(tree,cell_spot(Nat.add(U32.to_nat(first),block),after,lanes,group)) : Tile.Tile8}:
  TiledIndexedValues.Spreads{+cap,+cap_ok,+positive,+bounded} = spreads
  +outer = Nat.add(U32.to_nat(first),block)
  +count = U32.to_nat(after)
  +word = (first + U32.from_nat(block) : U32)
  +target = TiledValues.spot_cell(tree,cell_spot(outer,after,lanes,group))
  +outer_exact = FullRange.roundtrip(outer,capacity,outer_small)
  +quotient_exact = {Equal.trans(Nat,U32.to_nat((U32.from_nat(outer) / after : U32)),Nat.div(U32.to_nat(U32.from_nat(outer)),count),Nat.div(outer,count),
      U32Division.divide(U32.from_nat(outer),after,cap,cap_ok,bounded,positive),
      Equal.cong(Nat,Nat,value => Nat.div(value,count),U32.to_nat(U32.from_nat(outer)),outer,outer_exact)) :
    {U32.to_nat((U32.from_nat(outer) / after : U32)) == Nat.div(outer,count) : Nat}}
  +rest_exact = {Equal.trans(Nat,U32.to_nat((U32.from_nat(outer) % after : U32)),Nat.mod(U32.to_nat(U32.from_nat(outer)),count),Nat.mod(outer,count),
      U32Division.modulo(U32.from_nat(outer),after,cap,cap_ok,bounded,positive),
      Equal.cong(Nat,Nat,value => Nat.mod(value,count),U32.to_nat(U32.from_nat(outer)),outer,outer_exact)) :
    {U32.to_nat((U32.from_nat(outer) % after : U32)) == Nat.mod(outer,count) : Nat}}
  +channel_exact = {Equal.trans(U32,(0 + U32.from_nat(group) * 8 : U32),(U32.from_nat(group) * 8 : U32),U32.from_nat(Nat.mul(group,8n)),
      Arithmetic.add_zero_left((U32.from_nat(group) * 8 : U32)),WordMultiplication.native_encoded_multiply(group,8n)) :
    {(0 + U32.from_nat(group) * 8 : U32) == U32.from_nat(Nat.mul(group,8n)) : U32}}
  +index_exact = run_decoded((U32.from_nat(outer) / after : U32),lanes,U32.from_nat(group),after,(U32.from_nat(outer) % after : U32),Nat.div(outer,count),group,
    Nat.mod(outer,count),quotient_exact,FullRange.roundtrip(group,capacity,group_small),rest_exact)
  %Equal.sym(U32,(position - start : U32),U32.from_nat(group),local) :
    {TiledValues.spot_cell(tree,Cells.Spot{(0 + _ * 8 : U32),run_form((word / U32.max(after,1) : U32),lanes,_,after,(word % U32.max(after,1) : U32)),after,lanes}) == target : Tile.Tile8}
  %Equal.sym(U32,U32.max(after,1),after,Depth.positive_max(after,positive)) :
    {TiledValues.spot_cell(tree,Cells.Spot{(0 + U32.from_nat(group) * 8 : U32),run_form((word / _ : U32),lanes,U32.from_nat(group),after,(word % _ : U32)),after,lanes}) ==
      target : Tile.Tile8}
  %Equal.sym(U32,word,U32.from_nat(outer),TiledIndexedValues.position_word(first,block)) :
    {TiledValues.spot_cell(tree,Cells.Spot{(0 + U32.from_nat(group) * 8 : U32),run_form((_ / after : U32),lanes,U32.from_nat(group),after,(_ % after : U32)),after,lanes}) ==
      target : Tile.Tile8}
  %Equal.sym(U32,(0 + U32.from_nat(group) * 8 : U32),U32.from_nat(Nat.mul(group,8n)),channel_exact) :
    {TiledValues.spot_cell(tree,Cells.Spot{_,run_form((U32.from_nat(outer) / after : U32),lanes,U32.from_nat(group),after,(U32.from_nat(outer) % after : U32)),after,lanes}) ==
      target : Tile.Tile8}
  %Equal.sym(U32,run_form((U32.from_nat(outer) / after : U32),lanes,U32.from_nat(group),after,(U32.from_nat(outer) % after : U32)),
      U32.from_nat(cell_base(outer,after,lanes,group)),index_exact) :
    {TiledValues.spot_cell(tree,Cells.Spot{U32.from_nat(Nat.mul(group,8n)),_,after,lanes}) == target : Tile.Tile8}
  {==}

# ---- The cells of the groups outermost ----

# The cell at a position of a piece with the groups outermost.
def outer_cell(tree: Tree.Tree<F32>,context: Cells.Outer,position: U32) -> Tile.Tile8:
  TiledValues.spot_cell(tree,Cells.outer_spot(context,position))

def outer_matches(+tree: Tree.Tree<F32>,+context: Cells.Outer,+position: U32) ->
  {Cells.outer_read(ArrayLaws.pack_tree(~F32,tree),context,position) == (ArrayLaws.pack_tree(~F32,tree),outer_cell(tree,context,position)) : Array<F32> & Tile.Tile8}:
  TiledValues.spot_lanes(tree,Cells.outer_spot(context,position))

def outer_blocks(+count: U32,context: Cells.Outer,input: Tree.Tree<F32>,output: Tree.Tree<Tile.Tile8>) -> AccessLaws.ModelResult<Tree.Tree<F32>,Tile.Tile8>:
  AccessLaws.finish_model(~Tree.Tree<F32>,~Tile.Tile8,~Cells.Outer,
    Loop.run(~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Cells.Outer>,
      ~ArrayLaws.model_step(~Tree.Tree<F32>,~Tile.Tile8,~Cells.Outer,~outer_cell,~Blocks.in_place(~Cells.Outer)),U32.to_nat(count),Traversal.State{input,output,0,context}))

def outer_run(+count: U32,+context: Cells.Outer,+input: Tree.Tree<F32>,+output: Tree.Tree<Tile.Tile8>) ->
  {Cells.tiled_outermost(count,context,Tree.pack(F32,input),Tree.pack(Tile.Tile8,output)) ==
    AccessLaws.pack_result(~Array<F32>,~Tree.Tree<F32>,~Tile.Tile8,~ArrayLaws.pack_tree(~F32),outer_blocks(count,context,input,output)) : Array<F32> & Array<Tile.Tile8>}:
  +start = {Traversal.State{input,output,0,context} : Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Cells.Outer>}
  +done = Loop.run(~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Cells.Outer>,
    ~ArrayLaws.model_step(~Tree.Tree<F32>,~Tile.Tile8,~Cells.Outer,~outer_cell,~Blocks.in_place(~Cells.Outer)),U32.to_nat(count),start)
  %Equal.sym(Traversal.State<&1,&1,Array<F32>,Array<Tile.Tile8>,Cells.Outer>,
      Traversal.run(~Array<F32>,~Tile.Tile8,~Cells.Outer,~Cells.outer_read,~Blocks.in_place(~Cells.Outer),U32.to_nat(count),
        ArrayLaws.represent(~Array<F32>,~Tree.Tree<F32>,~Tile.Tile8,~Cells.Outer,~ArrayLaws.pack_tree(~F32),start)),
      ArrayLaws.represent(~Array<F32>,~Tree.Tree<F32>,~Tile.Tile8,~Cells.Outer,~ArrayLaws.pack_tree(~F32),done),
      ArrayLaws.traversal_refines_model(~Array<F32>,~Tree.Tree<F32>,~Tile.Tile8,~Cells.Outer,~ArrayLaws.pack_tree(~F32),~Cells.outer_read,~outer_cell,
        ~Blocks.in_place(~Cells.Outer),~outer_matches,U32.to_nat(count),start)) :
    {Traversal.finish(~Array<F32>,~Tile.Tile8,~Cells.Outer,_) ==
      AccessLaws.pack_result(~Array<F32>,~Tree.Tree<F32>,~Tile.Tile8,~ArrayLaws.pack_tree(~F32),AccessLaws.finish_model(~Tree.Tree<F32>,~Tile.Tile8,~Cells.Outer,done)) :
      Array<F32> & Array<Tile.Tile8>}
  AccessLaws.indexed_finished(~Array<F32>,~Tree.Tree<F32>,~Tile.Tile8,~Cells.Outer,~ArrayLaws.pack_tree(~F32),done)

# ---- The cells a piece builds ----

# The cells a piece of a tiling reads from a tree of values into a tree of
# cells.
def tiled_result(context: Tiled.Tiling,+first: U32,+count: U32,input: Tree.Tree<F32>,output: Tree.Tree<Tile.Tile8>) -> AccessLaws.ModelResult<Tree.Tree<F32>,Tile.Tile8>:
  Tiled.Tiling{+outer,+after,+lanes,+groups,position} = context
  match position:
    case Grid.Innermost{}:
      TiledValues.tiled_blocks((count / U32.max(groups,1) : U32),Cells.Tiling{after,lanes,groups,(first / U32.max(groups,1) : U32)},input,output)
    case Grid.Outermost{}: outer_blocks(count,Cells.Outer{outer,after,lanes,first},input,output)

# The buffer a piece of count cells writes into.
def cells_model(+count: U32) -> Buffer.Model<Tile.Tile8>:
  Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),count))

# The contents of the cells a piece builds from values.
def piece(context: Tiled.Tiling,values: Buffer.Content<F32>,+first: U32,+count: U32) -> Buffer.Content<Tile.Tile8>:
  +model = cells_model(count)
  Buffer.Content{IndexedValues.output_tree(~Tree.Tree<F32>,~Tile.Tile8,tiled_result(context,first,count,Buffer.content_tree(~F32,values),Buffer.tree(~Tile.Tile8,model))),
    Buffer.length(~Tile.Tile8,model)}

# What a piece leaf leaves: the contents of the values and of the piece.
def result_contents(result: Array<F32> & Array<Tile.Tile8>,+input_length: Nat,+output_length: Nat,+first: U32,+count: U32) ->
  Buffer.Content<F32> & PiecesModel.Content<Tile.Tile8>:
  (input,output) = result
  (Buffer.Content{Tree.reflect(F32,input),input_length},PiecesModel.PieceContent{Buffer.Content{Tree.reflect(Tile.Tile8,output),output_length},first,count})

def pair_seen(+input_length: Nat,+output_length: Nat,result: Array<F32> & Array<Tile.Tile8>,
  certificates: TraversalCertificate.Certificates<Array<F32>,Tile.Tile8,Certificate.family(~F32),result>,+first: U32,+count: U32) ->
  {PiecesEach.seen(~Tile.Tile8,~Storage.Buffer<F32>,~Buffer.Content<F32>,~Buffer.contents(~F32),Tiled.piece_leaf(first,count,Tiled.tiled_pair(input_length,output_length,result,certificates))) ==
    result_contents(result,input_length,output_length,first,count) : Buffer.Content<F32> & PiecesModel.Content<Tile.Tile8>}:
  (input,output) = result
  TraversalCertificate.Certificates{input_certificate,output_certificate} = certificates
  {==}

# The contents a model result leaves: its reader and its output.
def finished_contents(result: AccessLaws.ModelResult<Tree.Tree<F32>,Tile.Tile8>,+input_length: Nat,+output_length: Nat,+first: U32,+count: U32) ->
  Buffer.Content<F32> & PiecesModel.Content<Tile.Tile8>:
  AccessLaws.ModelResult{reader,output} = result
  (Buffer.Content{reader,input_length},PiecesModel.PieceContent{Buffer.Content{output,output_length},first,count})

def packed_seen(+result: AccessLaws.ModelResult<Tree.Tree<F32>,Tile.Tile8>,+input_length: Nat,+output_length: Nat,+first: U32,+count: U32) ->
  {result_contents(AccessLaws.pack_result(~Array<F32>,~Tree.Tree<F32>,~Tile.Tile8,~ArrayLaws.pack_tree(~F32),result),input_length,output_length,first,count) ==
    finished_contents(result,input_length,output_length,first,count) : Buffer.Content<F32> & PiecesModel.Content<Tile.Tile8>}:
  AccessLaws.ModelResult{+reader,+output} = result
  %StorageLaws.reflect_packed(F32,reader) :
    {result_contents(AccessLaws.pack_result(~Array<F32>,~Tree.Tree<F32>,~Tile.Tile8,~ArrayLaws.pack_tree(~F32),AccessLaws.ModelResult{reader,output}),input_length,output_length,first,count) ==
      (Buffer.Content{_,input_length},PiecesModel.PieceContent{Buffer.Content{output,output_length},first,count}) : Buffer.Content<F32> & PiecesModel.Content<Tile.Tile8>}
  %StorageLaws.reflect_packed(Tile.Tile8,output) :
    {result_contents(AccessLaws.pack_result(~Array<F32>,~Tree.Tree<F32>,~Tile.Tile8,~ArrayLaws.pack_tree(~F32),AccessLaws.ModelResult{reader,output}),input_length,output_length,first,count) ==
      (Buffer.Content{Tree.reflect(F32,Tree.pack(F32,reader)),input_length},PiecesModel.PieceContent{Buffer.Content{_,output_length},first,count}) :
      Buffer.Content<F32> & PiecesModel.Content<Tile.Tile8>}
  {==}

# A finished state leaves its reader and its output.
def state_contents(~Context: Data,+state: Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Context>,+input_length: Nat,+output_length: Nat,+first: U32,+count: U32) ->
  {finished_contents(AccessLaws.finish_model(~Tree.Tree<F32>,~Tile.Tile8,~Context,state),input_length,output_length,first,count) ==
    (Buffer.Content{ArrayLaws.model_reader(~Tree.Tree<F32>,~Tile.Tile8,~Context,state),input_length},
      PiecesModel.PieceContent{Buffer.Content{IndexedValues.output_tree(~Tree.Tree<F32>,~Tile.Tile8,AccessLaws.finish_model(~Tree.Tree<F32>,~Tile.Tile8,~Context,state)),output_length},
        first,count}) : Buffer.Content<F32> & PiecesModel.Content<Tile.Tile8>}:
  Traversal.State{reader,output,position,context} = state
  {==}

# The values kept and the piece of cells built.
def seen_piece(+tree: Tree.Tree<F32>,+length: Nat,+output: Tree.Tree<Tile.Tile8>,+output_length: Nat,+first: U32,+count: U32) ->
  Buffer.Content<F32> & PiecesModel.Content<Tile.Tile8>:
  (Buffer.Content{tree,length},PiecesModel.PieceContent{Buffer.Content{output,output_length},first,count})

# The values a piece leaf keeps and the cells it builds.
def into_seen(position: Grid.Position,+first: U32,+count: U32,+outer: U32,+after: U32,+lanes: U32,+groups: U32,+tree: Tree.Tree<F32>,+length: Nat,
  +certificate: Certificate.Certificate<F32,Tree.pack(F32,tree)>,+model: Buffer.Model<Tile.Tile8>) ->
  {PiecesEach.seen(~Tile.Tile8,~Storage.Buffer<F32>,~Buffer.Content<F32>,~Buffer.contents(~F32),
      Tiled.piece_leaf(first,count,Tiled.tiled_into(position,first,count,outer,after,lanes,groups,Storage.Buffer{Tree.pack(F32,tree),length,certificate},Buffer.pack(~Tile.Tile8,model)))) ==
    (Buffer.Content{tree,length},PiecesModel.PieceContent{Buffer.Content{IndexedValues.output_tree(~Tree.Tree<F32>,~Tile.Tile8,
      tiled_result(Tiled.Tiling{outer,after,lanes,groups,position},first,count,tree,Buffer.tree(~Tile.Tile8,model))),Buffer.length(~Tile.Tile8,model)},first,count}) :
    Buffer.Content<F32> & PiecesModel.Content<Tile.Tile8>}:
  match position:
    case Grid.Innermost{}:
      Buffer.Model{+cells,+cells_length,+cells_certificate} = model
      +blocks = (count / U32.max(groups,1) : U32)
      +run = {Cells.Tiling{after,lanes,groups,(first / U32.max(groups,1) : U32)} : Cells.Tiling}
      +result = TiledValues.tiled_blocks(blocks,run,tree,cells)
      +start = {Traversal.State{tree,cells,0,Blocks.Blocks{0,groups,run}} : Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Blocks.Blocks<Cells.Tiling>>}
      +state = Loop.run(~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Blocks.Blocks<Cells.Tiling>>,
        ~BlocksLaws.model_block_step(~Tree.Tree<F32>,~Tile.Tile8,~Cells.Tiling,~Cells.Run,~Cells.group_run,~TiledValues.run_value,~Blocks.in_place(~Cells.Run)),U32.to_nat(blocks),start)
      +output = IndexedValues.output_tree(~Tree.Tree<F32>,~Tile.Tile8,result)
      Equal.trans(Buffer.Content<F32> & PiecesModel.Content<Tile.Tile8>,
        PiecesEach.seen(~Tile.Tile8,~Storage.Buffer<F32>,~Buffer.Content<F32>,~Buffer.contents(~F32),
          Tiled.piece_leaf(first,count,Tiled.tiled_pair(length,cells_length,Cells.tiled(blocks,run,Tree.pack(F32,tree),Tree.pack(Tile.Tile8,cells)),
            TiledCertificate.tiled_certificates(blocks,run,Tree.pack(F32,tree),Tree.pack(Tile.Tile8,cells),certificate,cells_certificate)))),
        result_contents(Cells.tiled(blocks,run,Tree.pack(F32,tree),Tree.pack(Tile.Tile8,cells)),length,cells_length,first,count),seen_piece(tree,length,output,cells_length,first,count),
        pair_seen(length,cells_length,Cells.tiled(blocks,run,Tree.pack(F32,tree),Tree.pack(Tile.Tile8,cells)),
          TiledCertificate.tiled_certificates(blocks,run,Tree.pack(F32,tree),Tree.pack(Tile.Tile8,cells),certificate,cells_certificate),first,count),
        %Equal.sym(Array<F32> & Array<Tile.Tile8>,Cells.tiled(blocks,run,Tree.pack(F32,tree),Tree.pack(Tile.Tile8,cells)),
            AccessLaws.pack_result(~Array<F32>,~Tree.Tree<F32>,~Tile.Tile8,~ArrayLaws.pack_tree(~F32),result),TiledValues.tiled_run(blocks,run,tree,cells)) :
          {result_contents(_,length,cells_length,first,count) == seen_piece(tree,length,output,cells_length,first,count) : Buffer.Content<F32> & PiecesModel.Content<Tile.Tile8>}
        Equal.trans(Buffer.Content<F32> & PiecesModel.Content<Tile.Tile8>,
          result_contents(AccessLaws.pack_result(~Array<F32>,~Tree.Tree<F32>,~Tile.Tile8,~ArrayLaws.pack_tree(~F32),result),length,cells_length,first,count),
          finished_contents(result,length,cells_length,first,count),seen_piece(tree,length,output,cells_length,first,count),packed_seen(result,length,cells_length,first,count),
          Equal.trans(Buffer.Content<F32> & PiecesModel.Content<Tile.Tile8>,finished_contents(result,length,cells_length,first,count),
            (Buffer.Content{ArrayLaws.model_reader(~Tree.Tree<F32>,~Tile.Tile8,~Blocks.Blocks<Cells.Tiling>,state),length},
              PiecesModel.PieceContent{Buffer.Content{output,cells_length},first,count}),seen_piece(tree,length,output,cells_length,first,count),
            state_contents(~Blocks.Blocks<Cells.Tiling>,state,length,cells_length,first,count),
            Equal.cong(Tree.Tree<F32>,Buffer.Content<F32> & PiecesModel.Content<Tile.Tile8>,
              reader => (Buffer.Content{reader,length},PiecesModel.PieceContent{Buffer.Content{output,cells_length},first,count}),
              ArrayLaws.model_reader(~Tree.Tree<F32>,~Tile.Tile8,~Blocks.Blocks<Cells.Tiling>,state),tree,
              BlocksLaws.blocks_reader_kept(~Tree.Tree<F32>,~Tile.Tile8,~Cells.Tiling,~Cells.Run,~Cells.group_run,~TiledValues.run_value,~Blocks.in_place(~Cells.Run),
                U32.to_nat(blocks),start)))))
    case Grid.Outermost{}:
      Buffer.Model{+cells,+cells_length,+cells_certificate} = model
      +run = {Cells.Outer{outer,after,lanes,first} : Cells.Outer}
      +result = outer_blocks(count,run,tree,cells)
      +start = {Traversal.State{tree,cells,0,run} : Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Cells.Outer>}
      +state = Loop.run(~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Cells.Outer>,
        ~ArrayLaws.model_step(~Tree.Tree<F32>,~Tile.Tile8,~Cells.Outer,~outer_cell,~Blocks.in_place(~Cells.Outer)),U32.to_nat(count),start)
      +output = IndexedValues.output_tree(~Tree.Tree<F32>,~Tile.Tile8,result)
      Equal.trans(Buffer.Content<F32> & PiecesModel.Content<Tile.Tile8>,
        PiecesEach.seen(~Tile.Tile8,~Storage.Buffer<F32>,~Buffer.Content<F32>,~Buffer.contents(~F32),
          Tiled.piece_leaf(first,count,Tiled.tiled_pair(length,cells_length,Cells.tiled_outermost(count,run,Tree.pack(F32,tree),Tree.pack(Tile.Tile8,cells)),
            TiledCertificate.outermost_certificates(count,run,Tree.pack(F32,tree),Tree.pack(Tile.Tile8,cells),certificate,cells_certificate)))),
        result_contents(Cells.tiled_outermost(count,run,Tree.pack(F32,tree),Tree.pack(Tile.Tile8,cells)),length,cells_length,first,count),seen_piece(tree,length,output,cells_length,first,count),
        pair_seen(length,cells_length,Cells.tiled_outermost(count,run,Tree.pack(F32,tree),Tree.pack(Tile.Tile8,cells)),
          TiledCertificate.outermost_certificates(count,run,Tree.pack(F32,tree),Tree.pack(Tile.Tile8,cells),certificate,cells_certificate),first,count),
        %Equal.sym(Array<F32> & Array<Tile.Tile8>,Cells.tiled_outermost(count,run,Tree.pack(F32,tree),Tree.pack(Tile.Tile8,cells)),
            AccessLaws.pack_result(~Array<F32>,~Tree.Tree<F32>,~Tile.Tile8,~ArrayLaws.pack_tree(~F32),result),outer_run(count,run,tree,cells)) :
          {result_contents(_,length,cells_length,first,count) == seen_piece(tree,length,output,cells_length,first,count) : Buffer.Content<F32> & PiecesModel.Content<Tile.Tile8>}
        Equal.trans(Buffer.Content<F32> & PiecesModel.Content<Tile.Tile8>,
          result_contents(AccessLaws.pack_result(~Array<F32>,~Tree.Tree<F32>,~Tile.Tile8,~ArrayLaws.pack_tree(~F32),result),length,cells_length,first,count),
          finished_contents(result,length,cells_length,first,count),seen_piece(tree,length,output,cells_length,first,count),packed_seen(result,length,cells_length,first,count),
          Equal.trans(Buffer.Content<F32> & PiecesModel.Content<Tile.Tile8>,finished_contents(result,length,cells_length,first,count),
            (Buffer.Content{ArrayLaws.model_reader(~Tree.Tree<F32>,~Tile.Tile8,~Cells.Outer,state),length},
              PiecesModel.PieceContent{Buffer.Content{output,cells_length},first,count}),seen_piece(tree,length,output,cells_length,first,count),
            state_contents(~Cells.Outer,state,length,cells_length,first,count),
            Equal.cong(Tree.Tree<F32>,Buffer.Content<F32> & PiecesModel.Content<Tile.Tile8>,
              reader => (Buffer.Content{reader,length},PiecesModel.PieceContent{Buffer.Content{output,cells_length},first,count}),
              ArrayLaws.model_reader(~Tree.Tree<F32>,~Tile.Tile8,~Cells.Outer,state),tree,
              ArrayLaws.model_reader_kept(~Tree.Tree<F32>,~Tile.Tile8,~Cells.Outer,~outer_cell,~Blocks.in_place(~Cells.Outer),U32.to_nat(count),start)))))

# A piece leaf keeps its values and builds the piece of its tiling.
def leaf_seen(+context: Tiled.Tiling,+first: U32,+count: U32,+tree: Tree.Tree<F32>,+length: Nat,+certificate: Certificate.Certificate<F32,Tree.pack(F32,tree)>) ->
  {PiecesEach.seen(~Tile.Tile8,~Storage.Buffer<F32>,~Buffer.Content<F32>,~Buffer.contents(~F32),Tiled.tiled_piece(context,first,count,Storage.Buffer{Tree.pack(F32,tree),length,certificate})) ==
    (Buffer.Content{tree,length},PiecesModel.PieceContent{piece(context,Buffer.Content{tree,length},first,count),first,count}) : Buffer.Content<F32> & PiecesModel.Content<Tile.Tile8>}:
  Tiled.Tiling{+outer,+after,+lanes,+groups,+position} = context
  %Buffer.remodeled(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),count)) :
    {PiecesEach.seen(~Tile.Tile8,~Storage.Buffer<F32>,~Buffer.Content<F32>,~Buffer.contents(~F32),
        Tiled.piece_leaf(first,count,Tiled.tiled_into(position,first,count,outer,after,lanes,groups,Storage.Buffer{Tree.pack(F32,tree),length,certificate},_))) ==
      (Buffer.Content{tree,length},PiecesModel.PieceContent{piece(Tiled.Tiling{outer,after,lanes,groups,position},Buffer.Content{tree,length},first,count),first,count}) :
      Buffer.Content<F32> & PiecesModel.Content<Tile.Tile8>}
  into_seen(position,first,count,outer,after,lanes,groups,tree,length,certificate,cells_model(count))

def leaf_rule(+context: Tiled.Tiling,+first: U32,+count: U32,+kept: Buffer.Model<F32>) ->
  {PiecesEach.seen(~Tile.Tile8,~Storage.Buffer<F32>,~Buffer.Content<F32>,~Buffer.contents(~F32),Tiled.tiled_piece(context,first,count,Buffer.pack(~F32,kept))) ==
    (Buffer.contents(~F32,Buffer.pack(~F32,kept)),
      PiecesEach.row_part(~Tile.Tile8,~Tiled.Tiling,~Buffer.Content<F32>,~piece,context,Buffer.contents(~F32,Buffer.pack(~F32,kept)),first,count)) :
    Buffer.Content<F32> & PiecesModel.Content<Tile.Tile8>}:
  Buffer.Model{+tree,+length,+certificate} = kept
  %Equal.sym(Tree.Tree<F32>,Tree.reflect(F32,Tree.pack(F32,tree)),tree,StorageLaws.reflect_packed(F32,tree)) :
    {PiecesEach.seen(~Tile.Tile8,~Storage.Buffer<F32>,~Buffer.Content<F32>,~Buffer.contents(~F32),Tiled.tiled_piece(context,first,count,Storage.Buffer{Tree.pack(F32,tree),length,certificate})) ==
      (Buffer.Content{_,length},PiecesModel.PieceContent{piece(context,Buffer.Content{_,length},first,count),first,count}) : Buffer.Content<F32> & PiecesModel.Content<Tile.Tile8>}
  leaf_seen(context,first,count,tree,length,certificate)

# The pieces built from a buffer of values keep it and hold the piece of
# every range.
def built_cells(+context: Tiled.Tiling,+layout: Pieces.Ranges,+source: Buffer.Model<F32>) ->
  {PiecesEach.seen(~Tile.Tile8,~Storage.Buffer<F32>,~Buffer.Content<F32>,~Buffer.contents(~F32),
      Pieces.build(~Storage.Buffer<Tile.Tile8>,~Storage.Buffer<F32>,~Tiled.Tiling,~Tiled.tiled_piece,layout,context,Buffer.pack(~F32,source))) ==
    (Buffer.contents(~F32,Buffer.pack(~F32,source)),PiecesModel.build(~Tile.Tile8,~piece(context,Buffer.contents(~F32,Buffer.pack(~F32,source))),layout)) :
    Buffer.Content<F32> & PiecesModel.Content<Tile.Tile8>}:
  PiecesEach.built(~Tile.Tile8,~Storage.Buffer<F32>,~Tiled.Tiling,~Buffer.Model<F32>,~Buffer.pack(~F32),~PiecesEach.buffer_held(~F32),~Buffer.Content<F32>,~Buffer.contents(~F32),
    ~Tiled.tiled_piece,~piece,~leaf_rule,layout,context,source)

# ---- The cells of a piece of whole blocks ----

# Within the machine the cell at group group of block block of a run of
# blocks from outer index first is the spot of that group of outer index
# first + block.
def block_read(+limits: Machine.Limits,+total: U32,+blocks: U32,+after: U32,+lanes: U32,+groups: U32,+first: U32,+fill: Tile.Tile8,+source: Tree.Tree<F32>,
  +block: Nat,+group: Nat,+spreads: TiledIndexedValues.Spreads<after>,
  +exact: {Nat.mul(U32.to_nat(blocks),U32.to_nat(groups)) == U32.to_nat(total) : Nat},+bounded: {U32.is_le(total,Machine.capacity(limits)) == True{} : Bool},
  +block_inside: {Nat.is_lt(block,U32.to_nat(blocks)) == True{} : Bool},+group_inside: {Nat.is_lt(group,U32.to_nat(groups)) == True{} : Bool},
  +outer_small: {Nat.is_le(Nat.add(U32.to_nat(first),block),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,IndexedValues.output_tree(~Tree.Tree<F32>,~Tile.Tile8,TiledValues.tiled_blocks(blocks,Cells.Tiling{after,lanes,groups,first},source,
      Tree.fresh(Tile.Tile8,Storage.depth_for(total),fill))),U32.from_nat(Nat.add(Nat.mul(block,U32.to_nat(groups)),group))) ==
    TiledValues.spot_cell(source,cell_spot(Nat.add(U32.to_nat(first),block),after,lanes,group)) : Tile.Tile8}:
  +context = {Cells.Tiling{after,lanes,groups,first} : Cells.Tiling}
  +size = U32.to_nat(groups)
  +position = Nat.add(Nat.mul(block,size),group)
  +start = U32.from_nat(Nat.mul(block,size))
  +fresh = Tree.fresh(Tile.Tile8,Storage.depth_for(total),fill)
  +capacity = Machine.capacity(limits)
  +target = TiledValues.spot_cell(source,cell_spot(Nat.add(U32.to_nat(first),block),after,lanes,group))
  +flat = TiledValues.tiled_flat(limits,total,blocks,context,source,fresh,exact,bounded)
  +selected = TiledValues.inside_rows(block,U32.to_nat(blocks),group,size,U32.to_nat(total),exact,block_inside,group_inside)
  +room = TiledValues.row_room(block,U32.to_nat(blocks),size,U32.to_nat(total),exact,block_inside)
  +local_facts = TiledValues.decoded(limits,total,groups,block,group,group_inside,room,bounded)
  +group_small = Order.transitive(group,position,U32.to_nat(capacity),Order.right_le(Nat.mul(block,size),group),
    Order.transitive(position,U32.to_nat(total),U32.to_nat(capacity),Bounds.strict_le(position,U32.to_nat(total),selected),Bounds.as_nat_le(total,capacity,bounded)))
  %Equal.sym(AccessLaws.ModelResult<Tree.Tree<F32>,Tile.Tile8>,TiledValues.tiled_blocks(blocks,context,source,fresh),
    AccessLaws.finish_model(~Tree.Tree<F32>,~Tile.Tile8,~Cells.Tiling,
      Loop.run(~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Cells.Tiling>,
        ~ArrayLaws.model_step(~Tree.Tree<F32>,~Tile.Tile8,~Cells.Tiling,~TiledValues.tiled_cell,~Blocks.in_place(~Cells.Tiling)),
        U32.to_nat(total),Traversal.State{source,fresh,0,context})),flat) :
    {Tree.read(Tile.Tile8,IndexedValues.output_tree(~Tree.Tree<F32>,~Tile.Tile8,_),U32.from_nat(position)) == target : Tile.Tile8}
  Equal.trans(Tile.Tile8,
    Tree.read(Tile.Tile8,IndexedValues.run(~Tree.Tree<F32>,~Tile.Tile8,~Cells.Tiling,~TiledValues.tiled_cell,~Blocks.in_place(~Cells.Tiling),U32.to_nat(total),0,context,source,fresh),
      U32.from_nat(position)),
    TiledValues.tiled_cell(source,context,U32.from_nat(position)),target,
    TiledValues.fresh_stored(~Tree.Tree<F32>,~Tile.Tile8,~Cells.Tiling,~TiledValues.tiled_cell,total,fill,context,source,position,TiledValues.allocatable(limits,total,bounded),
      selected),
    Equal.trans(Tile.Tile8,TiledValues.tiled_cell(source,context,U32.from_nat(position)),
      TiledValues.run_value(source,Cells.group_run(context,U32.from_nat(block),start),U32.from_nat(position)),target,
      Equal.sym(Tile.Tile8,TiledValues.run_value(source,Cells.group_run(context,U32.from_nat(block),start),U32.from_nat(position)),
        TiledValues.tiled_cell(source,context,U32.from_nat(position)),TiledValues.run_decoded(source,after,lanes,groups,first,block,group,local_facts)),
      block_cell(source,after,lanes,groups,first,block,group,capacity,spreads,outer_small,group_small,U32.from_nat(position),start,
        TiledValues.local_of(groups,block,group,local_facts))))

# ---- The buffer a piece writes ----

def model_tree(~Element: Data,+model: Buffer.Model<Element>) -> {Buffer.content_tree(~Element,Buffer.content(~Element,model)) == Buffer.tree(~Element,model) : Tree.Tree<Element>}:
  Buffer.Model{tree,length,certificate} = model
  {==}

def model_length(~Element: Data,+model: Buffer.Model<Element>) -> {Buffer.content_length(~Element,Buffer.content(~Element,model)) == Buffer.length(~Element,model) : Nat}:
  Buffer.Model{tree,length,certificate} = model
  {==}

def fresh_cells(+count: U32) -> Tree.Tree<Tile.Tile8>:
  Tree.fresh(Tile.Tile8,Storage.depth_for(count),Tiled.zero_tile())

def cells_content(+count: U32) -> {Buffer.content(~Tile.Tile8,cells_model(count)) == Buffer.Content{fresh_cells(count),U32.to_nat(count)} : Buffer.Content<Tile.Tile8>}:
  Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),count))),
    Buffer.contents(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),count)),
    Buffer.Content{fresh_cells(count),U32.to_nat(count)},Buffer.remodel_content(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),count)),
    AllocationLaws.contents(~Tile.Tile8,Tensors.depth_for(count),Tiled.zero_tile(),U32.to_nat(count)))

def cells_tree(+count: U32) -> {Buffer.tree(~Tile.Tile8,cells_model(count)) == fresh_cells(count) : Tree.Tree<Tile.Tile8>}:
  +content = Buffer.content(~Tile.Tile8,cells_model(count))
  Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,cells_model(count)),Buffer.content_tree(~Tile.Tile8,content),fresh_cells(count),
    Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.content_tree(~Tile.Tile8,content),Buffer.tree(~Tile.Tile8,cells_model(count)),model_tree(~Tile.Tile8,cells_model(count))),
    Equal.cong(Buffer.Content<Tile.Tile8>,Tree.Tree<Tile.Tile8>,value => Buffer.content_tree(~Tile.Tile8,value),content,Buffer.Content{fresh_cells(count),U32.to_nat(count)},
      cells_content(count)))

def cells_length(+count: U32) -> {Buffer.length(~Tile.Tile8,cells_model(count)) == U32.to_nat(count) : Nat}:
  +content = Buffer.content(~Tile.Tile8,cells_model(count))
  Equal.trans(Nat,Buffer.length(~Tile.Tile8,cells_model(count)),Buffer.content_length(~Tile.Tile8,content),U32.to_nat(count),
    Equal.sym(Nat,Buffer.content_length(~Tile.Tile8,content),Buffer.length(~Tile.Tile8,cells_model(count)),model_length(~Tile.Tile8,cells_model(count))),
    Equal.cong(Buffer.Content<Tile.Tile8>,Nat,value => Buffer.content_length(~Tile.Tile8,value),content,Buffer.Content{fresh_cells(count),U32.to_nat(count)},
      cells_content(count)))

# ---- A piece of whole blocks with the groups innermost ----

# The tiling and the geometry of the cells of row-major values with the
# groups at a position.
def entry_context(position: Grid.Position,+outer: U32,+after: U32,+lanes: U32) -> Tiled.Tiling:
  Tiled.Tiling{outer,after,lanes,Grid.groups(lanes),position}

def entry_geometry(position: Grid.Position,+outer: U32,+after: U32,+lanes: U32) -> Cells.Cells:
  Cells.Cells{outer,after,lanes,Grid.groups(lanes),position}

def stored_lifted(+tree: Tree.Tree<F32>,+length: Nat,+geometry: Cells.Cells) -> Representation.Lifted<Buffer.Content<F32>>:
  Representation.Lifted{Buffer.Content{tree,length},geometry}

# What a piece of whole blocks needs: its outer indices lie in the outer
# extent, its counts within the machine, and the divisors in the half word.
type Inner is Data:
  Inner{limits: Machine.Limits,tree: Tree.Tree<F32>,length: Nat,outer: U32,after: U32,lanes: U32,first: U32,count: U32,first_units: Nat,count_units: Nat,
    after_spreads: TiledIndexedValues.Spreads<after>,group_spreads: TiledIndexedValues.Spreads<Grid.groups(lanes)>,group_facts: TiledValues.Groups<limits,lanes>,
    first_exact: {U32.to_nat(first) == Nat.mul(first_units,U32.to_nat(Grid.groups(lanes))) : Nat},
    count_exact: {U32.to_nat(count) == Nat.mul(count_units,U32.to_nat(Grid.groups(lanes))) : Nat},
    units_inside: {Nat.is_le(Nat.add(first_units,count_units),U32.to_nat(outer)) == True{} : Bool},
    bounded: {U32.is_le(count,Machine.capacity(limits)) == True{} : Bool},
    outer_room: {Nat.is_le(U32.to_nat(outer),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    span_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

def inner_size(inner: Inner) -> Nat:
  Inner{+limits,+tree,+length,+outer,+after,+lanes,+first,+count,+first_units,+count_units,+after_spreads,+group_spreads,+group_facts,+first_exact,+count_exact,
    +units_inside,+bounded,+outer_room,+span_room} = inner
  U32.to_nat(count)

def InnerAt(+inner: Inner,+index: Nat) -> Data:
  Inner{+limits,+tree,+length,+outer,+after,+lanes,+first,+count,+first_units,+count_units,+after_spreads,+group_spreads,+group_facts,+first_exact,+count_exact,
    +units_inside,+bounded,+outer_room,+span_room} = inner
  +built = piece(entry_context(Grid.Innermost{},outer,after,lanes),Buffer.Content{tree,length},first,count)
  Content.At(~Tile.Tile8,~Representation.Lifted<Buffer.Content<F32>>,~Representation.lifted(~Buffer.Content<F32>,~Contract.stored(~F32)),
    stored_lifted(tree,length,entry_geometry(Grid.Innermost{},outer,after,lanes)),first,Buffer.content_tree(~Tile.Tile8,built),Buffer.content_length(~Tile.Tile8,built),index)

# A word divided by a positive divisor within the half word, for a multiple.
def units_of(+word: U32,+divisor: U32,+units: Nat,+spreads: TiledIndexedValues.Spreads<divisor>,
  +exact: {U32.to_nat(word) == Nat.mul(units,U32.to_nat(divisor)) : Nat}) -> {U32.to_nat((word / U32.max(divisor,1) : U32)) == units : Nat}:
  TiledIndexedValues.Spreads{+cap,+cap_ok,+positive,+bounded} = spreads
  %Equal.sym(U32,U32.max(divisor,1),divisor,Depth.positive_max(divisor,positive)) : {U32.to_nat((word / _ : U32)) == units : Nat}
  Equal.trans(Nat,U32.to_nat((word / divisor : U32)),Nat.div(U32.to_nat(word),U32.to_nat(divisor)),units,U32Division.divide(word,divisor,cap,cap_ok,bounded,positive),
    %Equal.sym(Nat,U32.to_nat(word),Nat.mul(units,U32.to_nat(divisor)),exact) : {Nat.div(_,U32.to_nat(divisor)) == units : Nat}
    Division.exact_quotient(units,U32.to_nat(divisor),positive))

def left_le(+front: Nat,+low: Nat,+high: Nat,+below: {Nat.is_le(low,high) == True{} : Bool}) -> {Nat.is_le(Nat.add(front,low),Nat.add(front,high)) == True{} : Bool}:
  %Order.commutative(low,front) : {Nat.is_le(_,Nat.add(front,high)) == True{} : Bool}
  %Order.commutative(high,front) : {Nat.is_le(Nat.add(low,front),_) == True{} : Bool}
  Order.add_le(low,high,front,below)

def left_lt(+front: Nat,+low: Nat,+high: Nat,+below: {Nat.is_lt(low,high) == True{} : Bool}) -> {Nat.is_lt(Nat.add(front,low),Nat.add(front,high)) == True{} : Bool}:
  %Order.commutative(low,front) : {Nat.is_lt(_,Nat.add(front,high)) == True{} : Bool}
  %Order.commutative(high,front) : {Nat.is_lt(Nat.add(low,front),_) == True{} : Bool}
  Order.add_lt(low,high,front,below)

# Cell first + index of the piece is cell group of outer index first + block.
def inner_word(+first: Nat,+units: Nat,+size: Nat,+block: Nat,+group: Nat,+first_exact: {first == Nat.mul(units,size) : Nat}) ->
  {Nat.add(first,Nat.add(Nat.mul(block,size),group)) == Nat.add(Nat.mul(Nat.add(units,block),size),group) : Nat}:
  %Equal.sym(Nat,first,Nat.mul(units,size),first_exact) : {Nat.add(_,Nat.add(Nat.mul(block,size),group)) == Nat.add(Nat.mul(Nat.add(units,block),size),group) : Nat}
  Equal.trans(Nat,Nat.add(Nat.mul(units,size),Nat.add(Nat.mul(block,size),group)),Nat.add(Nat.add(Nat.mul(units,size),Nat.mul(block,size)),group),
    Nat.add(Nat.mul(Nat.add(units,block),size),group),
    Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(units,size),Nat.mul(block,size)),group),Nat.add(Nat.mul(units,size),Nat.add(Nat.mul(block,size),group)),
      Addition.associative(Nat.mul(units,size),Nat.mul(block,size),group)),
    Equal.cong(Nat,Nat,value => Nat.add(value,group),Nat.add(Nat.mul(units,size),Nat.mul(block,size)),Nat.mul(Nat.add(units,block),size),
      Equal.sym(Nat,Nat.mul(Nat.add(units,block),size),Nat.add(Nat.mul(units,size),Nat.mul(block,size)),RowMajor.distrib(units,block,size))))

# Every cell of a piece of whole blocks holds the lifted cell of the stored
# values.
def inner_cell(+inner: Inner,+index: Nat,+inside: {Nat.is_lt(index,inner_size(inner)) == True{} : Bool}) -> InnerAt(inner,index):
  Inner{+limits,+tree,+length,+outer,+after,+lanes,+first,+count,+first_units,+count_units,+after_spreads,+group_spreads,+group_facts,+first_exact,+count_exact,
    +units_inside,+bounded,+outer_room,+span_room} = inner
  TiledIndexedValues.Spreads{+after_cap,+after_cap_ok,+after_positive,+after_bounded} = after_spreads
  TiledIndexedValues.Spreads{+group_cap,+group_cap_ok,+group_positive,+group_bounded} = group_spreads
  +groups = Grid.groups(lanes)
  +size = U32.to_nat(groups)
  +capacity = Machine.capacity(limits)
  +block = Nat.div(index,size)
  +group = Nat.mod(index,size)
  +position = Nat.add(Nat.mul(block,size),group)
  +split = DivisionOrder.quotient_exact(index,size,group_positive)
  +below = {%count_exact : {Nat.is_lt(index,_) == True{} : Bool} inside : {Nat.is_lt(index,Nat.mul(count_units,size)) == True{} : Bool}}
  +block_inside = DivisionOrder.quotient_below(index,count_units,size,group_positive,below)
  +group_inside = DivisionOrder.remainder_small(index,size,group_positive)
  +blocks = (count / U32.max(groups,1) : U32)
  +start = (first / U32.max(groups,1) : U32)
  +blocks_exact = units_of(count,groups,count_units,TiledIndexedValues.Spreads{group_cap,group_cap_ok,group_positive,group_bounded},count_exact)
  +start_exact = units_of(first,groups,first_units,TiledIndexedValues.Spreads{group_cap,group_cap_ok,group_positive,group_bounded},first_exact)
  +exact = {%Equal.sym(Nat,U32.to_nat(blocks),count_units,blocks_exact) : {Nat.mul(_,size) == U32.to_nat(count) : Nat}
      Equal.sym(Nat,U32.to_nat(count),Nat.mul(count_units,size),count_exact) : {Nat.mul(U32.to_nat(blocks),size) == U32.to_nat(count) : Nat}}
  +block_below = {%Equal.sym(Nat,U32.to_nat(blocks),count_units,blocks_exact) : {Nat.is_lt(block,_) == True{} : Bool} block_inside :
      {Nat.is_lt(block,U32.to_nat(blocks)) == True{} : Bool}}
  +outer_index = Nat.add(first_units,block)
  +outer_inside = Bounds.nat_trans(outer_index,Nat.add(first_units,count_units),U32.to_nat(outer),left_lt(first_units,block,count_units,block_inside),units_inside)
  +start_small = {%Equal.sym(Nat,U32.to_nat(start),first_units,start_exact) : {Nat.is_le(Nat.add(_,block),U32.to_nat(capacity)) == True{} : Bool}
      Order.transitive(outer_index,U32.to_nat(outer),U32.to_nat(capacity),Bounds.strict_le(outer_index,U32.to_nat(outer),outer_inside),outer_room) :
      {Nat.is_le(Nat.add(U32.to_nat(start),block),U32.to_nat(capacity)) == True{} : Bool}}
  +geometry = entry_geometry(Grid.Innermost{},outer,after,lanes)
  +word = U32.from_nat(Nat.add(U32.to_nat(first),index))
  +word_room = Order.transitive(Nat.add(U32.to_nat(first),index),Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(capacity),
    left_le(U32.to_nat(first),index,U32.to_nat(count),Bounds.strict_le(index,U32.to_nat(count),inside)),span_room)
  +word_exact = {Equal.trans(Nat,U32.to_nat(word),Nat.add(U32.to_nat(first),index),Nat.add(Nat.mul(outer_index,size),group),
      FullRange.roundtrip(Nat.add(U32.to_nat(first),index),capacity,word_room),
      %Equal.sym(Nat,index,position,split) : {Nat.add(U32.to_nat(first),_) == Nat.add(Nat.mul(outer_index,size),group) : Nat}
      inner_word(U32.to_nat(first),first_units,size,block,group,first_exact)) :
    {U32.to_nat(word) == Nat.add(Nat.mul(outer_index,size),group) : Nat}}
  +lifted = Representation.decoded(~Buffer.Content<F32>,~Contract.stored(~F32),Buffer.Content{tree,length},geometry,outer_index,group,word,word_exact,outer_inside,group_inside)
  +cell = entry_cell(limits,tree,length,outer,after,lanes,Grid.Innermost{},outer_index,group,group_facts,group_inside,after_positive)
  +read = block_read(limits,count,blocks,after,lanes,groups,start,Tiled.zero_tile(),tree,block,group,
    TiledIndexedValues.Spreads{after_cap,after_cap_ok,after_positive,after_bounded},exact,bounded,block_below,group_inside,start_small)
  +target = Representation.lifted(~Buffer.Content<F32>,~Contract.stored(~F32),stored_lifted(tree,length,geometry),word)
  KernelGemm.Both{
    %Equal.sym(Nat,Buffer.length(~Tile.Tile8,cells_model(count)),U32.to_nat(count),cells_length(count)) : {Nat.is_lt(index,_) == True{} : Bool} inside,
    %Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,cells_model(count)),fresh_cells(count),cells_tree(count)) :
      {Tree.read(Tile.Tile8,IndexedValues.output_tree(~Tree.Tree<F32>,~Tile.Tile8,TiledValues.tiled_blocks(blocks,Cells.Tiling{after,lanes,groups,start},tree,_)),U32.from_nat(index)) ==
        target : Tile.Tile8}
    %Equal.sym(Nat,index,position,split) :
      {Tree.read(Tile.Tile8,IndexedValues.output_tree(~Tree.Tree<F32>,~Tile.Tile8,TiledValues.tiled_blocks(blocks,Cells.Tiling{after,lanes,groups,start},tree,fresh_cells(count))),
        U32.from_nat(_)) == target : Tile.Tile8}
    Equal.trans(Tile.Tile8,
      Tree.read(Tile.Tile8,IndexedValues.output_tree(~Tree.Tree<F32>,~Tile.Tile8,TiledValues.tiled_blocks(blocks,Cells.Tiling{after,lanes,groups,start},tree,fresh_cells(count))),
        U32.from_nat(position)),
      TiledValues.spot_cell(tree,cell_spot(Nat.add(U32.to_nat(start),block),after,lanes,group)),target,read,
      Equal.trans(Tile.Tile8,TiledValues.spot_cell(tree,cell_spot(Nat.add(U32.to_nat(start),block),after,lanes,group)),
        TiledValues.spot_cell(tree,cell_spot(outer_index,after,lanes,group)),target,
        Equal.cong(Nat,Tile.Tile8,value => TiledValues.spot_cell(tree,cell_spot(Nat.add(value,block),after,lanes,group)),U32.to_nat(start),first_units,start_exact),
        Equal.trans(Tile.Tile8,TiledValues.spot_cell(tree,cell_spot(outer_index,after,lanes,group)),
          Representation.cell(~Buffer.Content<F32>,~Contract.stored(~F32),Buffer.Content{tree,length},geometry,outer_index,group),target,cell,
          Equal.sym(Tile.Tile8,target,Representation.cell(~Buffer.Content<F32>,~Contract.stored(~F32),Buffer.Content{tree,length},geometry,outer_index,group),lifted))))}

def inner_values(+inner: Inner) -> Members.All(Nat,{index => InnerAt(inner,index) : Nat -> Data},Ranges.range(inner_size(inner),0n)):
  Members.range_all(~Inner,~InnerAt,~inner_size,~inner_cell,inner_size(inner),0n,inner,Order.reflexive(inner_size(inner)))

# Whole blocks of a multiple of the divisor cover the word.
def whole_blocks(+word: U32,+divisor: U32,+units: Nat,+spreads: TiledIndexedValues.Spreads<divisor>,
  +exact: {U32.to_nat(word) == Nat.mul(units,U32.to_nat(divisor)) : Nat}) ->
  {Nat.mul(U32.to_nat((word / U32.max(divisor,1) : U32)),U32.to_nat(divisor)) == U32.to_nat(word) : Nat}:
  %Equal.sym(Nat,U32.to_nat((word / U32.max(divisor,1) : U32)),units,units_of(word,divisor,units,spreads,exact)) : {Nat.mul(_,U32.to_nat(divisor)) == U32.to_nat(word) : Nat}
  Equal.sym(Nat,U32.to_nat(word),Nat.mul(units,U32.to_nat(divisor)),exact)

# A fresh buffer of cells within the machine is ready.
def fresh_ready(+limits: Machine.Limits,+count: U32,+bounded: {U32.is_le(count,Machine.capacity(limits)) == True{} : Bool}) ->
  Buffer.ContentReady<Tile.Tile8,Buffer.Content{fresh_cells(count),U32.to_nat(count)}>:
  Buffer.content_ready(~Tile.Tile8,Buffer.fresh(~Tile.Tile8,Storage.depth_for(count),Tiled.zero_tile(),U32.to_nat(count)),
    TiledValues.fresh_sized(~Tile.Tile8,count,Tiled.zero_tile(),Depth.bounded(count,TiledValues.allocatable(limits,count,bounded))))

# A piece of whole blocks within the machine is ready.
def inner_ready(+limits: Machine.Limits,+tree: Tree.Tree<F32>,+after: U32,+lanes: U32,+first: U32,+count: U32,+count_units: Nat,
  +group_spreads: TiledIndexedValues.Spreads<Grid.groups(lanes)>,+count_exact: {U32.to_nat(count) == Nat.mul(count_units,U32.to_nat(Grid.groups(lanes))) : Nat},
  +bounded: {U32.is_le(count,Machine.capacity(limits)) == True{} : Bool}) ->
  Buffer.ContentReady<Tile.Tile8,Buffer.Content{IndexedValues.output_tree(~Tree.Tree<F32>,~Tile.Tile8,
    TiledValues.tiled_blocks((count / U32.max(Grid.groups(lanes),1) : U32),Cells.Tiling{after,lanes,Grid.groups(lanes),(first / U32.max(Grid.groups(lanes),1) : U32)},tree,
      Buffer.tree(~Tile.Tile8,cells_model(count)))),Buffer.length(~Tile.Tile8,cells_model(count))}>:
  +groups = Grid.groups(lanes)
  +blocks = (count / U32.max(groups,1) : U32)
  +context = {Cells.Tiling{after,lanes,groups,(first / U32.max(groups,1) : U32)} : Cells.Tiling}
  +fresh = fresh_cells(count)
  +exact = whole_blocks(count,groups,count_units,group_spreads,count_exact)
  +flat = TiledValues.tiled_flat(limits,count,blocks,context,tree,fresh,exact,bounded)
  %Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,cells_model(count)),fresh,cells_tree(count)) :
    Buffer.ContentReady<Tile.Tile8,Buffer.Content{IndexedValues.output_tree(~Tree.Tree<F32>,~Tile.Tile8,TiledValues.tiled_blocks(blocks,context,tree,_)),
      Buffer.length(~Tile.Tile8,cells_model(count))}>
  %Equal.sym(Nat,Buffer.length(~Tile.Tile8,cells_model(count)),U32.to_nat(count),cells_length(count)) :
    Buffer.ContentReady<Tile.Tile8,Buffer.Content{IndexedValues.output_tree(~Tree.Tree<F32>,~Tile.Tile8,TiledValues.tiled_blocks(blocks,context,tree,fresh)),_}>
  %Equal.sym(AccessLaws.ModelResult<Tree.Tree<F32>,Tile.Tile8>,TiledValues.tiled_blocks(blocks,context,tree,fresh),
      AccessLaws.finish_model(~Tree.Tree<F32>,~Tile.Tile8,~Cells.Tiling,
        Loop.run(~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,Cells.Tiling>,
          ~ArrayLaws.model_step(~Tree.Tree<F32>,~Tile.Tile8,~Cells.Tiling,~TiledValues.tiled_cell,~Blocks.in_place(~Cells.Tiling)),
          U32.to_nat(count),Traversal.State{tree,fresh,0,context})),flat) :
    Buffer.ContentReady<Tile.Tile8,Buffer.Content{IndexedValues.output_tree(~Tree.Tree<F32>,~Tile.Tile8,_),U32.to_nat(count)}>
  IndexedValues.ready(~Tree.Tree<F32>,~Tile.Tile8,~Cells.Tiling,~TiledValues.tiled_cell,~Blocks.in_place(~Cells.Tiling),U32.to_nat(count),context,tree,fresh,U32.to_nat(count),
    fresh_ready(limits,count,bounded))

# A piece of whole blocks with the groups innermost holds the lifted values of
# the stored values.
def inner_holds(+limits: Machine.Limits,+tree: Tree.Tree<F32>,+length: Nat,+outer: U32,+after: U32,+lanes: U32,+first: U32,+count: U32,+first_units: Nat,+count_units: Nat,
  +after_spreads: TiledIndexedValues.Spreads<after>,+group_spreads: TiledIndexedValues.Spreads<Grid.groups(lanes)>,+group_facts: TiledValues.Groups<limits,lanes>,
  +first_exact: {U32.to_nat(first) == Nat.mul(first_units,U32.to_nat(Grid.groups(lanes))) : Nat},
  +count_exact: {U32.to_nat(count) == Nat.mul(count_units,U32.to_nat(Grid.groups(lanes))) : Nat},
  +units_inside: {Nat.is_le(Nat.add(first_units,count_units),U32.to_nat(outer)) == True{} : Bool},
  +bounded: {U32.is_le(count,Machine.capacity(limits)) == True{} : Bool},
  +outer_room: {Nat.is_le(U32.to_nat(outer),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +span_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Buffer.Content<F32>>,~Representation.lifted(~Buffer.Content<F32>,~Contract.stored(~F32)),
    stored_lifted(tree,length,entry_geometry(Grid.Innermost{},outer,after,lanes)),first,count,piece(entry_context(Grid.Innermost{},outer,after,lanes),Buffer.Content{tree,length},first,count)):
  KernelGemm.Both{
    inner_values(Inner{limits,tree,length,outer,after,lanes,first,count,first_units,count_units,after_spreads,group_spreads,group_facts,first_exact,count_exact,
      units_inside,bounded,outer_room,span_room}),
    KernelGemm.Both{cells_length(count),inner_ready(limits,tree,after,lanes,first,count,count_units,group_spreads,count_exact,bounded)}}

# ---- A piece with the groups outermost ----

# The channel of lane 0 of the group a word decodes to.
def group_channel(+group_word: U32,+geometry: Cells.Cells,+number: Nat,+outer_word: U32,
  +decoded: TiledIndexedValues.Decoded<outer_word,group_word,geometry,number>) ->
  {(group_word * 8 : U32) == U32.from_nat(Nat.mul(Representation.cell_group(geometry,number),8n)) : U32}:
  TiledIndexedValues.Decoded{+outer_exact,+group_exact} = decoded
  %group_exact : {(group_word * 8 : U32) == U32.from_nat(Nat.mul(_,8n)) : U32}
  %Arithmetic.native_roundtrip(group_word) : {(_ * 8 : U32) == U32.from_nat(Nat.mul(U32.to_nat(group_word),8n)) : U32}
  WordMultiplication.native_encoded_multiply(U32.to_nat(group_word),8n)

# The spot of cell index of a piece with the groups outermost from cell first
# is the spot of the group and the outer index the geometry names for cell
# first + index.
def outer_spot_exact(+limits: Machine.Limits,+outer: U32,+after: U32,+lanes: U32,+first: U32,+index: Nat,+divides: TiledIndexedValues.Divides<limits,outer,lanes>,
  +after_spreads: TiledIndexedValues.Spreads<after>) ->
  {Cells.outer_spot(Cells.Outer{outer,after,lanes,first},U32.from_nat(index)) ==
    cell_spot(Representation.cell_outer(entry_geometry(Grid.Outermost{},outer,after,lanes),U32.to_nat(U32.from_nat(Nat.add(U32.to_nat(first),index)))),after,lanes,
      Representation.cell_group(entry_geometry(Grid.Outermost{},outer,after,lanes),U32.to_nat(U32.from_nat(Nat.add(U32.to_nat(first),index))))) : Cells.Spot}:
  TiledIndexedValues.Divides{+outer_positive,+groups_positive,+outer_bound,+groups_bound} = divides
  TiledIndexedValues.Spreads{+after_cap,+after_cap_ok,+after_positive,+after_bounded} = after_spreads
  +groups = Grid.groups(lanes)
  +geometry = entry_geometry(Grid.Outermost{},outer,after,lanes)
  +word = U32.from_nat(Nat.add(U32.to_nat(first),index))
  +number = U32.to_nat(word)
  +at = Representation.cell_outer(geometry,number)
  +group = Representation.cell_group(geometry,number)
  +count = U32.to_nat(after)
  +group_word = TiledIndexedValues.group_word(Grid.Outermost{},word,outer,groups)
  +index_word = TiledIndexed.index_at(Grid.Outermost{},word,outer,after,lanes,groups)
  +decoded = TiledIndexedValues.decode_along(Grid.Outermost{},Machine.half_word(limits),Limits.half_word_fits(limits),word,outer,after,lanes,groups,outer_positive,
    groups_positive,outer_bound,groups_bound)
  +base = {%Representation.least_positive(count,after_positive) :
      {index_word == U32.from_nat(TiledIndexedValues.index_natural(Nat.div(at,_),U32.to_nat(lanes),group,_,Nat.mod(at,_))) : U32}
    TiledIndexedValues.index_exact(Grid.Outermost{},word,outer,groups,outer,after,lanes,Grid.Outermost{},
      TiledIndexedValues.Spreads{after_cap,after_cap_ok,after_positive,after_bounded},decoded) :
    {index_word == U32.from_nat(cell_base(at,after,lanes,group)) : U32}}
  +channel = group_channel(group_word,geometry,number,TiledIndexedValues.outer_word(Grid.Outermost{},word,outer,groups),decoded)
  %Equal.sym(U32,(first + U32.from_nat(index) : U32),word,TiledIndexedValues.position_word(first,index)) :
    {Cells.Spot{((_ / U32.max(outer,1) : U32) * 8 : U32),TiledIndexed.index_at(Grid.Outermost{},_,outer,after,lanes,groups),after,lanes} == cell_spot(at,after,lanes,group) : Cells.Spot}
  %Equal.sym(U32,(group_word * 8 : U32),U32.from_nat(Nat.mul(group,8n)),channel) : {Cells.Spot{_,index_word,after,lanes} == cell_spot(at,after,lanes,group) : Cells.Spot}
  %Equal.sym(U32,index_word,U32.from_nat(cell_base(at,after,lanes,group)),base) : {Cells.Spot{U32.from_nat(Nat.mul(group,8n)),_,after,lanes} == cell_spot(at,after,lanes,group) : Cells.Spot}
  {==}

# What a piece with the groups outermost needs: its cells lie in the cells of
# the geometry and within the machine, and the divisors in the half word.
type Outer is Data:
  Outer{limits: Machine.Limits,tree: Tree.Tree<F32>,length: Nat,outer: U32,after: U32,lanes: U32,first: U32,count: U32,
    divides: TiledIndexedValues.Divides<limits,outer,lanes>,after_spreads: TiledIndexedValues.Spreads<after>,group_facts: TiledValues.Groups<limits,lanes>,
    cells_inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer))) == True{} : Bool},
    bounded: {U32.is_le(count,Machine.capacity(limits)) == True{} : Bool},
    span_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

def outer_size(facts: Outer) -> Nat:
  Outer{+limits,+tree,+length,+outer,+after,+lanes,+first,+count,+divides,+after_spreads,+group_facts,+cells_inside,+bounded,+span_room} = facts
  U32.to_nat(count)

def OuterAt(+facts: Outer,+index: Nat) -> Data:
  Outer{+limits,+tree,+length,+outer,+after,+lanes,+first,+count,+divides,+after_spreads,+group_facts,+cells_inside,+bounded,+span_room} = facts
  +built = piece(entry_context(Grid.Outermost{},outer,after,lanes),Buffer.Content{tree,length},first,count)
  Content.At(~Tile.Tile8,~Representation.Lifted<Buffer.Content<F32>>,~Representation.lifted(~Buffer.Content<F32>,~Contract.stored(~F32)),
    stored_lifted(tree,length,entry_geometry(Grid.Outermost{},outer,after,lanes)),first,Buffer.content_tree(~Tile.Tile8,built),Buffer.content_length(~Tile.Tile8,built),index)

# Every cell of a piece with the groups outermost holds the lifted cell of the
# stored values.
def outer_piece_cell(+facts: Outer,+index: Nat,+inside: {Nat.is_lt(index,outer_size(facts)) == True{} : Bool}) -> OuterAt(facts,index):
  Outer{+limits,+tree,+length,+outer,+after,+lanes,+first,+count,+divides,+after_spreads,+group_facts,+cells_inside,+bounded,+span_room} = facts
  TiledIndexedValues.Divides{+outer_positive,+groups_positive,+outer_bound,+groups_bound} = divides
  TiledIndexedValues.Spreads{+after_cap,+after_cap_ok,+after_positive,+after_bounded} = after_spreads
  +groups = Grid.groups(lanes)
  +capacity = Machine.capacity(limits)
  +context = {Cells.Outer{outer,after,lanes,first} : Cells.Outer}
  +geometry = entry_geometry(Grid.Outermost{},outer,after,lanes)
  +cell = Nat.add(U32.to_nat(first),index)
  +word = U32.from_nat(cell)
  +number = U32.to_nat(word)
  +at = Representation.cell_outer(geometry,number)
  +group = Representation.cell_group(geometry,number)
  +cell_room = Order.transitive(cell,Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(capacity),
    left_le(U32.to_nat(first),index,U32.to_nat(count),Bounds.strict_le(index,U32.to_nat(count),inside)),span_room)
  +cell_below = Bounds.nat_trans(cell,Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(U32.to_nat(groups),U32.to_nat(outer)),
    left_lt(U32.to_nat(first),index,U32.to_nat(count),inside),cells_inside)
  +number_below = {%Equal.sym(Nat,U32.to_nat(word),cell,FullRange.roundtrip(cell,capacity,cell_room)) : {Nat.is_lt(_,Nat.mul(U32.to_nat(groups),U32.to_nat(outer))) == True{} : Bool}
      cell_below : {Nat.is_lt(number,Nat.mul(U32.to_nat(groups),U32.to_nat(outer))) == True{} : Bool}}
  +group_inside = {%Equal.sym(Nat,Representation.least_one(U32.to_nat(outer)),U32.to_nat(outer),Representation.least_positive(U32.to_nat(outer),outer_positive)) : {Nat.is_lt(Nat.div(number,_),U32.to_nat(groups)) == True{} : Bool}
      DivisionOrder.quotient_below(number,U32.to_nat(groups),U32.to_nat(outer),outer_positive,number_below) : {Nat.is_lt(group,U32.to_nat(groups)) == True{} : Bool}}
  +fresh = fresh_cells(count)
  +target = Representation.lifted(~Buffer.Content<F32>,~Contract.stored(~F32),stored_lifted(tree,length,geometry),word)
  +stored = TiledValues.fresh_stored(~Tree.Tree<F32>,~Tile.Tile8,~Cells.Outer,~outer_cell,count,Tiled.zero_tile(),context,tree,index,
    TiledValues.allocatable(limits,count,bounded),inside)
  +spot = Equal.cong(Cells.Spot,Tile.Tile8,spot => TiledValues.spot_cell(tree,spot),Cells.outer_spot(context,U32.from_nat(index)),cell_spot(at,after,lanes,group),
    outer_spot_exact(limits,outer,after,lanes,first,index,TiledIndexedValues.Divides{outer_positive,groups_positive,outer_bound,groups_bound},
      TiledIndexedValues.Spreads{after_cap,after_cap_ok,after_positive,after_bounded}))
  +lifted = entry_cell(limits,tree,length,outer,after,lanes,Grid.Outermost{},at,group,group_facts,group_inside,after_positive)
  KernelGemm.Both{
    %Equal.sym(Nat,Buffer.length(~Tile.Tile8,cells_model(count)),U32.to_nat(count),cells_length(count)) : {Nat.is_lt(index,_) == True{} : Bool} inside,
    %Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,cells_model(count)),fresh,cells_tree(count)) :
      {Tree.read(Tile.Tile8,IndexedValues.output_tree(~Tree.Tree<F32>,~Tile.Tile8,outer_blocks(count,context,tree,_)),U32.from_nat(index)) == target : Tile.Tile8}
    Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,IndexedValues.output_tree(~Tree.Tree<F32>,~Tile.Tile8,outer_blocks(count,context,tree,fresh)),U32.from_nat(index)),
      outer_cell(tree,context,U32.from_nat(index)),target,stored,
      Equal.trans(Tile.Tile8,outer_cell(tree,context,U32.from_nat(index)),TiledValues.spot_cell(tree,cell_spot(at,after,lanes,group)),target,spot,lifted))}

def outer_values(+facts: Outer) -> Members.All(Nat,{index => OuterAt(facts,index) : Nat -> Data},Ranges.range(outer_size(facts),0n)):
  Members.range_all(~Outer,~OuterAt,~outer_size,~outer_piece_cell,outer_size(facts),0n,facts,Order.reflexive(outer_size(facts)))

# A piece with the groups outermost within the machine is ready.
def outer_ready(+limits: Machine.Limits,+tree: Tree.Tree<F32>,+context: Cells.Outer,+count: U32,+bounded: {U32.is_le(count,Machine.capacity(limits)) == True{} : Bool}) ->
  Buffer.ContentReady<Tile.Tile8,Buffer.Content{IndexedValues.output_tree(~Tree.Tree<F32>,~Tile.Tile8,outer_blocks(count,context,tree,Buffer.tree(~Tile.Tile8,cells_model(count)))),
    Buffer.length(~Tile.Tile8,cells_model(count))}>:
  +fresh = fresh_cells(count)
  %Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,cells_model(count)),fresh,cells_tree(count)) :
    Buffer.ContentReady<Tile.Tile8,Buffer.Content{IndexedValues.output_tree(~Tree.Tree<F32>,~Tile.Tile8,outer_blocks(count,context,tree,_)),Buffer.length(~Tile.Tile8,cells_model(count))}>
  %Equal.sym(Nat,Buffer.length(~Tile.Tile8,cells_model(count)),U32.to_nat(count),cells_length(count)) :
    Buffer.ContentReady<Tile.Tile8,Buffer.Content{IndexedValues.output_tree(~Tree.Tree<F32>,~Tile.Tile8,outer_blocks(count,context,tree,fresh)),_}>
  IndexedValues.ready(~Tree.Tree<F32>,~Tile.Tile8,~Cells.Outer,~outer_cell,~Blocks.in_place(~Cells.Outer),U32.to_nat(count),context,tree,fresh,U32.to_nat(count),
    fresh_ready(limits,count,bounded))

# A piece with the groups outermost holds the lifted values of the stored
# values.
def outer_holds(+limits: Machine.Limits,+tree: Tree.Tree<F32>,+length: Nat,+outer: U32,+after: U32,+lanes: U32,+first: U32,+count: U32,
  +divides: TiledIndexedValues.Divides<limits,outer,lanes>,+after_spreads: TiledIndexedValues.Spreads<after>,+group_facts: TiledValues.Groups<limits,lanes>,
  +cells_inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer))) == True{} : Bool},
  +bounded: {U32.is_le(count,Machine.capacity(limits)) == True{} : Bool},
  +span_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Buffer.Content<F32>>,~Representation.lifted(~Buffer.Content<F32>,~Contract.stored(~F32)),
    stored_lifted(tree,length,entry_geometry(Grid.Outermost{},outer,after,lanes)),first,count,piece(entry_context(Grid.Outermost{},outer,after,lanes),Buffer.Content{tree,length},first,count)):
  KernelGemm.Both{
    outer_values(Outer{limits,tree,length,outer,after,lanes,first,count,divides,after_spreads,group_facts,cells_inside,bounded,span_room}),
    KernelGemm.Both{cells_length(count),outer_ready(limits,tree,Cells.Outer{outer,after,lanes,first},count,bounded)}}

# ---- Every piece of a layout ----

# The unit of the pieces of a position: the groups of an outer index with the
# groups innermost, one cell with the groups outermost.
def unit_of(position: Grid.Position,+lanes: U32) -> Nat:
  match position:
    case Grid.Innermost{}: U32.to_nat(Grid.groups(lanes))
    case Grid.Outermost{}: 1n

# What every piece of the cells of row-major values needs: the cells of the
# geometry within the machine, the divisors in the half word and the groups
# of the lanes.
type Entry<-limits: Machine.Limits,-outer: U32,-after: U32,-lanes: U32,-total: Nat> is Data:
  Entry{cells: {Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer)) == total : Nat},room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    divides: TiledIndexedValues.Divides<limits,outer,lanes>,after_spreads: TiledIndexedValues.Spreads<after>,group_facts: TiledValues.Groups<limits,lanes>}

# The cells up to the end of a piece within the machine.
def span_within(+limits: Machine.Limits,+first: U32,+count: U32,+total: Nat,+inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),total) == True{} : Bool},
  +room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(count)),total,U32.to_nat(Machine.capacity(limits)),inside,room)

def count_within(+limits: Machine.Limits,+first: U32,+count: U32,
  +span_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.is_le(count,Machine.capacity(limits)) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_le(count,Machine.capacity(limits)),Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(limits))),Comparison.le(count,Machine.capacity(limits))) :
    {_ == True{} : Bool}
  Order.transitive(U32.to_nat(count),Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(Machine.capacity(limits)),Order.right_le(U32.to_nat(first),U32.to_nat(count)),span_room)

# An outer extent lies within the cells of its groups.
def outer_within(+outer: Nat,+groups: Nat,+positive: {Nat.is_lt(0n,groups) == True{} : Bool}) -> {Nat.is_le(outer,Nat.mul(groups,outer)) == True{} : Bool}:
  %Order.times_one(outer) : {Nat.is_le(_,Nat.mul(groups,outer)) == True{} : Bool}
  %RowMajor.multiply_commutative(outer,groups) : {Nat.is_le(Nat.mul(outer,1n),_) == True{} : Bool}
  DivisionOrder.multiply_left_le(outer,1n,groups,Order.positive_one(groups,positive))

# A piece of a run of whole outer indices with the groups innermost holds the
# lifted values of the stored values.
def inner_leaf(+limits: Machine.Limits,+tree: Tree.Tree<F32>,+length: Nat,+outer: U32,+after: U32,+lanes: U32,+total: Nat,+entry: Entry<limits,outer,after,lanes,total>,
  +first: U32,+count: U32,+fits: Runs.Fits<U32.to_nat(Grid.groups(lanes)),U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),total>) ->
  Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Buffer.Content<F32>>,~Representation.lifted(~Buffer.Content<F32>,~Contract.stored(~F32)),
    stored_lifted(tree,length,entry_geometry(Grid.Innermost{},outer,after,lanes)),first,count,piece(entry_context(Grid.Innermost{},outer,after,lanes),Buffer.Content{tree,length},first,count)):
  Entry{+cells,+room,+divides,+after_spreads,+group_facts} = entry
  TiledIndexedValues.Divides{+outer_positive,+groups_positive,+outer_bound,+groups_bound} = divides
  Runs.Fits{+low_units,+high_units,+low_exact,+high_exact,+less,+inside} = fits
  +size = U32.to_nat(Grid.groups(lanes))
  +start = U32.to_nat(first)
  +rows = U32.to_nat(count)
  +extent = U32.to_nat(outer)
  +capacity = U32.to_nat(Machine.capacity(limits))
  +count_units = Nat.sub(high_units,low_units)
  +count_exact = {%Order.subtract_shift(start,rows) : {_ == Nat.mul(count_units,size) : Nat}
      %Equal.sym(Nat,Nat.add(start,rows),Nat.mul(high_units,size),high_exact) : {Nat.sub(_,start) == Nat.mul(count_units,size) : Nat}
      %Equal.sym(Nat,start,Nat.mul(low_units,size),low_exact) : {Nat.sub(Nat.mul(high_units,size),_) == Nat.mul(count_units,size) : Nat}
      RowMajor.subtract_products(high_units,low_units,size) : {rows == Nat.mul(count_units,size) : Nat}}
  +units_less = RowMajor.cancel_multiply_lt(low_units,high_units,size,
    %low_exact : {Nat.is_lt(_,Nat.mul(high_units,size)) == True{} : Bool} %high_exact : {Nat.is_lt(start,_) == True{} : Bool} less)
  +cells_total = {%Equal.sym(Nat,Nat.mul(extent,size),Nat.mul(size,extent),RowMajor.multiply_commutative(extent,size)) : {_ == total : Nat} cells :
    {Nat.mul(extent,size) == total : Nat}}
  +high_within = RowMajor.cancel_multiply_le(high_units,extent,size,groups_positive,
    %high_exact : {Nat.is_le(_,Nat.mul(extent,size)) == True{} : Bool} %Equal.sym(Nat,Nat.mul(extent,size),total,cells_total) : {Nat.is_le(Nat.add(start,rows),_) == True{} : Bool} inside)
  +units_inside = {%Equal.sym(Nat,Nat.add(low_units,count_units),high_units,Order.restore_shift(low_units,high_units,Bounds.strict_le(low_units,high_units,units_less))) :
      {Nat.is_le(_,extent) == True{} : Bool} high_within : {Nat.is_le(Nat.add(low_units,count_units),extent) == True{} : Bool}}
  +outer_room = Order.transitive(extent,Nat.mul(size,extent),capacity,outer_within(extent,size,groups_positive),
    %Equal.sym(Nat,Nat.mul(size,extent),total,cells) : {Nat.is_le(_,capacity) == True{} : Bool} room)
  +span_room = span_within(limits,first,count,total,inside,room)
  inner_holds(limits,tree,length,outer,after,lanes,first,count,low_units,count_units,after_spreads,
    TiledIndexedValues.Spreads{Machine.half_word(limits),Limits.half_word_fits(limits),groups_positive,groups_bound},group_facts,low_exact,count_exact,units_inside,
    count_within(limits,first,count,span_room),outer_room,span_room)

# A piece of the cells with the groups outermost holds the lifted values of
# the stored values.
def outer_leaf(+limits: Machine.Limits,+tree: Tree.Tree<F32>,+length: Nat,+outer: U32,+after: U32,+lanes: U32,+total: Nat,+entry: Entry<limits,outer,after,lanes,total>,
  +first: U32,+count: U32,+fits: Runs.Fits<1n,U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),total>) ->
  Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Buffer.Content<F32>>,~Representation.lifted(~Buffer.Content<F32>,~Contract.stored(~F32)),
    stored_lifted(tree,length,entry_geometry(Grid.Outermost{},outer,after,lanes)),first,count,piece(entry_context(Grid.Outermost{},outer,after,lanes),Buffer.Content{tree,length},first,count)):
  Entry{+cells,+room,+divides,+after_spreads,+group_facts} = entry
  +span_room = span_within(limits,first,count,total,Runs.fits_inside(1n,U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),total,fits),room)
  outer_holds(limits,tree,length,outer,after,lanes,first,count,divides,after_spreads,group_facts,
    %Equal.sym(Nat,Nat.mul(U32.to_nat(Grid.groups(lanes)),U32.to_nat(outer)),total,cells) : {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),_) == True{} : Bool}
    Runs.fits_inside(1n,U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),total,fits),
    count_within(limits,first,count,span_room),span_room)

def leaf_holds(position: Grid.Position,+limits: Machine.Limits,+tree: Tree.Tree<F32>,+length: Nat,+outer: U32,+after: U32,+lanes: U32,+total: Nat,
  +entry: Entry<limits,outer,after,lanes,total>,+first: U32,+count: U32,
  +fits: Runs.Fits<unit_of(position,lanes),U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),total>) ->
  Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Buffer.Content<F32>>,~Representation.lifted(~Buffer.Content<F32>,~Contract.stored(~F32)),
    stored_lifted(tree,length,entry_geometry(position,outer,after,lanes)),first,count,piece(entry_context(position,outer,after,lanes),Buffer.Content{tree,length},first,count)):
  match position:
    case Grid.Innermost{}: inner_leaf(limits,tree,length,outer,after,lanes,total,entry,first,count,fits)
    case Grid.Outermost{}: outer_leaf(limits,tree,length,outer,after,lanes,total,entry,first,count,fits)

# Every piece of a run of a layout holds the lifted values of the stored
# values.
def built_holds(+position: Grid.Position,+limits: Machine.Limits,+outer: U32,+after: U32,+lanes: U32,+total: Nat,+entry: Entry<limits,outer,after,lanes,total>,
  +values: Buffer.Content<F32>,layout: Pieces.Ranges,+run: Runs.Run(layout,unit_of(position,lanes),total)) ->
  PiecesModel.ContentAll(~Tile.Tile8,~Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Buffer.Content<F32>>,~Representation.lifted(~Buffer.Content<F32>,~Contract.stored(~F32)),
      Representation.Lifted{values,entry_geometry(position,outer,after,lanes)}),
    PiecesModel.build(~Tile.Tile8,~piece(entry_context(position,outer,after,lanes),values),layout)):
  Buffer.Content{+tree,+length} = values
  match layout:
    case Pieces.Range{+first,+count}: leaf_holds(position,limits,tree,length,outer,after,lanes,total,entry,first,count,run)
    case Pieces.Split{+left,+right}:
      Held.Both{+left_run,+rest} = run
      Held.Both{+right_run,+glue} = rest
      KernelGemm.Both{built_holds(position,limits,outer,after,lanes,total,entry,Buffer.Content{tree,length},left,left_run),
        built_holds(position,limits,outer,after,lanes,total,entry,Buffer.Content{tree,length},right,right_run)}

# The pieces built from values have the layout they were built on.
def built_layout(+context: Tiled.Tiling,+values: Buffer.Content<F32>,layout: Pieces.Ranges) ->
  {PiecesModel.content_layout(~Tile.Tile8,PiecesModel.build(~Tile.Tile8,~piece(context,values),layout)) == layout : Pieces.Ranges}:
  match layout:
    case Pieces.Range{+first,+count}: {==}
    case Pieces.Split{+left,+right}:
      %Equal.sym(Pieces.Ranges,PiecesModel.content_layout(~Tile.Tile8,PiecesModel.build(~Tile.Tile8,~piece(context,values),left)),left,built_layout(context,values,left)) :
        {Pieces.Split{_,PiecesModel.content_layout(~Tile.Tile8,PiecesModel.build(~Tile.Tile8,~piece(context,values),right))} == Pieces.Split{left,right} : Pieces.Ranges}
      %Equal.sym(Pieces.Ranges,PiecesModel.content_layout(~Tile.Tile8,PiecesModel.build(~Tile.Tile8,~piece(context,values),right)),right,built_layout(context,values,right)) :
        {Pieces.Split{left,_} == Pieces.Split{left,right} : Pieces.Ranges}
      {==}

# Pieces of the contents of a build, observed as a model, are placed by the
# model and hold what every piece of the build holds.
def observed_placed(~Context: Data,~Holds: @+context: Context -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Tile.Tile8> -> Data,+context: Context,
  +layout: Pieces.Ranges,-built: PiecesModel.Content<Tile.Tile8>,-pieces: Pieces.Pieces(~Tile.Tile8),observation: PiecesModel.Observation<Tile.Tile8,pieces>,
  +laid: {PiecesModel.content_layout(~Tile.Tile8,built) == layout : Pieces.Ranges},
  +contents: {PiecesModel.contents(~Tile.Tile8,pieces) == built : PiecesModel.Content<Tile.Tile8>},
  +holds: PiecesModel.ContentAll(~Tile.Tile8,~Holds(context),built)) ->
  Placed.PlacedModel<Tile.Tile8,Holds(context),layout,pieces>:
  PiecesModel.Observation{+model,+equation} = observation
  +content = PiecesModel.content(~Tile.Tile8,model)
  +same = Equal.trans(PiecesModel.Content<Tile.Tile8>,content,PiecesModel.contents(~Tile.Tile8,pieces),built,
    Equal.sym(PiecesModel.Content<Tile.Tile8>,PiecesModel.contents(~Tile.Tile8,pieces),content,PiecesModel.observed_contents(~Tile.Tile8,model,pieces,equation)),contents)
  Placed.PlacedModel{model,Equal.sym(Pieces.Pieces(~Tile.Tile8),PiecesModel.pack(~Tile.Tile8,model),pieces,equation),
    Equal.trans(Pieces.Ranges,PiecesModel.layout(~Tile.Tile8,model),PiecesModel.content_layout(~Tile.Tile8,content),layout,PiecesModel.layout_matches_content(~Tile.Tile8,model),
      Equal.trans(Pieces.Ranges,PiecesModel.content_layout(~Tile.Tile8,content),PiecesModel.content_layout(~Tile.Tile8,built),layout,
        Equal.cong(PiecesModel.Content<Tile.Tile8>,Pieces.Ranges,content => PiecesModel.content_layout(~Tile.Tile8,content),content,built,same),laid)),
    %Equal.sym(PiecesModel.Content<Tile.Tile8>,content,built,same) : PiecesModel.ContentAll(~Tile.Tile8,~Holds(context),_) holds}

# What a build of pieces from values sees, observed as a model, places the
# pieces by the model, each holding the lifted values of the stored values.
def placed_made(+position: Grid.Position,+limits: Machine.Limits,+outer: U32,+after: U32,+lanes: U32,+total: Nat,+entry: Entry<limits,outer,after,lanes,total>,
  +layout: Pieces.Ranges,+run: Runs.Run(layout,unit_of(position,lanes),total),+values: Buffer.Content<F32>,made: Storage.Buffer<F32> & Pieces.Pieces(~Tile.Tile8),
  +seen_exact: {PiecesEach.seen(~Tile.Tile8,~Storage.Buffer<F32>,~Buffer.Content<F32>,~Buffer.contents(~F32),made) ==
    (values,PiecesModel.build(~Tile.Tile8,~piece(entry_context(position,outer,after,lanes),values),layout)) : Buffer.Content<F32> & PiecesModel.Content<Tile.Tile8>}) ->
  Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Buffer.Content<F32>>,~Representation.lifted(~Buffer.Content<F32>,~Contract.stored(~F32)),
      Representation.Lifted{values,entry_geometry(position,outer,after,lanes)}),layout,Pair.snd(Storage.Buffer<F32>,Pieces.Pieces(~Tile.Tile8),made)>:
  (state,pieces) = made
  observed_placed(~Representation.Lifted<Buffer.Content<F32>>,
    ~Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Buffer.Content<F32>>,~Representation.lifted(~Buffer.Content<F32>,~Contract.stored(~F32))),
    Representation.Lifted{values,entry_geometry(position,outer,after,lanes)},layout,
    PiecesModel.build(~Tile.Tile8,~piece(entry_context(position,outer,after,lanes),values),layout),pieces,PiecesModel.observe(~Tile.Tile8,pieces),
    built_layout(entry_context(position,outer,after,lanes),values,layout),
    Equal.cong(Buffer.Content<F32> & PiecesModel.Content<Tile.Tile8>,PiecesModel.Content<Tile.Tile8>,pair => Pair.snd(Buffer.Content<F32>,PiecesModel.Content<Tile.Tile8>,pair),
      (Buffer.contents(~F32,state),PiecesModel.contents(~Tile.Tile8,pieces)),(values,PiecesModel.build(~Tile.Tile8,~piece(entry_context(position,outer,after,lanes),values),layout)),
      seen_exact),
    built_holds(position,limits,outer,after,lanes,total,entry,values,layout,run))

# The pieces built from a buffer of values hold, piece by piece, the lifted
# values of the stored values.
def entry_placed(+position: Grid.Position,+limits: Machine.Limits,+outer: U32,+after: U32,+lanes: U32,+total: Nat,+entry: Entry<limits,outer,after,lanes,total>,
  +layout: Pieces.Ranges,+run: Runs.Run(layout,unit_of(position,lanes),total),+source: Buffer.Model<F32>) ->
  Placed.PlacedModel<Tile.Tile8,Placed.PieceHolds(~Tile.Tile8,~Representation.Lifted<Buffer.Content<F32>>,~Representation.lifted(~Buffer.Content<F32>,~Contract.stored(~F32)),
      Representation.Lifted{Buffer.contents(~F32,Buffer.pack(~F32,source)),entry_geometry(position,outer,after,lanes)}),layout,
    Pair.snd(Storage.Buffer<F32>,Pieces.Pieces(~Tile.Tile8),Pieces.build(~Storage.Buffer<Tile.Tile8>,~Storage.Buffer<F32>,~Tiled.Tiling,~Tiled.tiled_piece,layout,
      entry_context(position,outer,after,lanes),Buffer.pack(~F32,source)))>:
  placed_made(position,limits,outer,after,lanes,total,entry,layout,run,Buffer.contents(~F32,Buffer.pack(~F32,source)),
    Pieces.build(~Storage.Buffer<Tile.Tile8>,~Storage.Buffer<F32>,~Tiled.Tiling,~Tiled.tiled_piece,layout,entry_context(position,outer,after,lanes),Buffer.pack(~F32,source)),
    built_cells(entry_context(position,outer,after,lanes),layout,source))

# ---- The tiled form of stored values ----

# The cover of the cells of values gives every piece what it needs.
def cover_entry(+limits: Machine.Limits,+lanes: U32,+padded_outer: U32,+padded_position: Grid.Position,+outer: U32,+after: U32,+position: Grid.Position,+total: Nat,
  +cover: Public.Cover<limits,lanes,padded_outer,padded_position,outer,position,total>,+positive: {Nat.is_lt(0n,total) == True{} : Bool},
  +after_spreads: TiledIndexedValues.Spreads<after>) -> Entry<limits,outer,after,lanes,total>:
  Public.Cover{+counted,+cells,+padded_cells,+placement} = cover
  +size = U32.to_nat(Grid.groups(lanes))
  +extent = U32.to_nat(outer)
  +lane_count = U32.to_nat(lanes)
  +capacity = U32.to_nat(Machine.capacity(limits))
  +product_positive = {%Equal.sym(Nat,Nat.mul(size,extent),total,cells) : {Nat.is_lt(0n,_) == True{} : Bool} positive : {Nat.is_lt(0n,Nat.mul(size,extent)) == True{} : Bool}}
  +outer_positive = DivisionOrder.right_factor_positive(size,extent,product_positive)
  +groups_positive = RowMajor.left_factor_positive(size,extent,product_positive)
  +group_facts = TiledValues.grouped(limits,lanes,Order.transitive(lane_count,Nat.mul(lane_count,extent),capacity,RowMajor.product_covers(lane_count,extent,outer_positive),counted))
  +cells_room = Order.transitive(Nat.mul(size,extent),Nat.mul(lane_count,extent),capacity,RowMajor.mul_le(size,lane_count,extent,TiledValues.fewer_of(limits,lanes,group_facts)),counted)
  Entry{cells,%cells : {Nat.is_le(_,capacity) == True{} : Bool} cells_room,
    TiledIndexedValues.Divides{outer_positive,groups_positive,
      Limits.double_bound(limits,extent,Order.transitive(extent,Nat.mul(size,extent),capacity,outer_within(extent,size,groups_positive),cells_room)),
      Limits.double_bound(limits,size,Order.transitive(size,Nat.mul(size,extent),capacity,RowMajor.product_covers(size,extent,outer_positive),cells_room))},
    after_spreads,group_facts}

# What every piece of the cells of counted values needs, with the lane axis
# inside them or past them.
def entry_cases(within: Bool,+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+count: U32,
  +classified: {Nat.is_lt(axis,List.length(&2,U32,values)) == within : Bool},+exact: {ShapeLaws.elements(values) == U32.to_nat(count) : Nat},
  +bounded: {U32.is_le(count,Machine.capacity(limits)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,axis,position))) == True{} : Bool},
  +after_spreads: TiledIndexedValues.Spreads<Tiled.product(Public.lane_after(values,axis))>) ->
  Entry<limits,(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),
    Grid.lanes_at(values,axis),ShapeLaws.elements(Tiled.cell_shape(values,axis,position))>:
  match within:
    case True{}:
      +cells = Tiled.cell_shape(values,axis,position)
      +outer = (Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32)
      cover_entry(limits,Grid.lanes_at(values,axis),Grid.outer_count(cells,position),position,outer,Tiled.product(Public.lane_after(values,axis)),position,
        ShapeLaws.elements(cells),Public.along_cover(limits,values,axis,Grid.lanes_at(values,axis),position,count,exact,bounded,classified,{==}),positive,after_spreads)
    case False{}:
      +cells = Tiled.cell_shape(values,axis,position)
      +outer = (Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32)
      +after = Tiled.product(Public.lane_after(values,axis))
      %Equal.sym(U32,Grid.lanes_at(values,axis),1,Public.lanes_past(values,axis,classified)) : Entry<limits,outer,after,_,ShapeLaws.elements(cells)>
      cover_entry(limits,1,Grid.product(cells),Grid.Innermost{},outer,after,position,ShapeLaws.elements(cells),
        Public.single_cover(limits,values,axis,position,count,exact,bounded,classified),positive,after_spreads)

# The last extent of extents with one appended is the one appended.
def last_appended(+others: List<&2,U32>,+last: U32) -> {GridRanges.last_extent(List.append(&2,U32,others,[last])) == U32.to_nat(last) : Nat}:
  match others:
    case Nil{}: {==}
    case Con{+head,+tail}:
      match tail:
        case Nil{}: {==}
        case Con{+next,+rest}: last_appended(next <> rest,last)

def whole_multiple(+value: Nat) -> GridRanges.Multiple<value,1n>:
  GridRanges.Multiple{value,Equal.sym(Nat,Nat.mul(value,1n),value,Order.times_one(value))}

# The last cell axis is a whole number of units of the pieces: the groups
# with the groups innermost along a lane axis, one cell otherwise.
def unit_multiple(position: Grid.Position,within: Bool,+values: List<&2,U32>,+axis: Nat,+classified: {Nat.is_lt(axis,List.length(&2,U32,values)) == within : Bool}) ->
  GridRanges.Multiple<GridRanges.last_extent(Tiled.cell_shape(values,axis,position)),unit_of(position,Grid.lanes_at(values,axis))>:
  match position within:
    case Grid.Innermost{} True{}:
      +others = Grid.removed(values,axis)
      +groups = Grid.groups(Grid.lanes_at(values,axis))
      %Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,values)),True{},classified) :
        GridRanges.Multiple<GridRanges.last_extent(Grid.placed(_,Grid.Innermost{},others,groups)),U32.to_nat(groups)>
      %Equal.sym(Nat,GridRanges.last_extent(List.append(&2,U32,others,[groups])),U32.to_nat(groups),last_appended(others,groups)) :
        GridRanges.Multiple<_,U32.to_nat(groups)>
      GridRanges.Multiple{1n,Equal.sym(Nat,Nat.mul(1n,U32.to_nat(groups)),U32.to_nat(groups),Addition.zero_right(U32.to_nat(groups)))}
    case Grid.Innermost{} False{}:
      +last = GridRanges.last_extent(Tiled.cell_shape(values,axis,Grid.Innermost{}))
      %Equal.sym(U32,Grid.lanes_at(values,axis),1,Public.lanes_past(values,axis,classified)) : GridRanges.Multiple<last,U32.to_nat(Grid.groups(_))>
      whole_multiple(last)
    case Grid.Outermost{} within: whole_multiple(GridRanges.last_extent(Tiled.cell_shape(values,axis,Grid.Outermost{})))

# Counted values lie within the machine.
def values_room(+limits: Machine.Limits,+values: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(values) == U32.to_nat(count) : Nat},
  +bounded: {U32.is_le(count,Machine.capacity(limits)) == True{} : Bool}) ->
  {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  %Equal.sym(Nat,ShapeLaws.elements(values),U32.to_nat(count),exact) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  Bounds.as_nat_le(count,Machine.capacity(limits),bounded)

# The tiled form of nonempty counted values stored in a buffer, with the
# lanes on an axis inside them or just past them and the groups at a
# position, holds the lifted stored values.
def from_stored(+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+workers: U32,+source: Buffer.Model<F32>,
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+counted: Global.Counted<values>,
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool}) ->
  Public.Contract(~Buffer.Content<F32>,~Contract.stored(~F32),Buffer.contents(~F32,Buffer.pack(~F32,source)),values,
    Tiled.from_buffer(values,axis,position,Tensors.Usable{},workers,Buffer.pack(~F32,source))):
  Global.Counted{+count,+accepted} = counted
  +limits = Default.limits()
  +logical = Buffer.contents(~F32,Buffer.pack(~F32,source))
  +prefix = Public.lane_prefix(values,axis,1)
  +lanes = Grid.lanes_at(values,axis)
  +after_axes = Public.lane_after(values,axis)
  +after = Tiled.product(after_axes)
  +outer = (prefix * after : U32)
  +cells = Tiled.cell_shape(values,axis,position)
  +total = ShapeLaws.elements(cells)
  +layout = Pieces.ranges_of_grid(Tensors.grid_of(cells))
  +exact = Count.count_exact(ShapeLaws.elements(values),count,Count.count_of(values,count,accepted))
  +bounded = PaddedCells.count_bounded(ShapeLaws.elements(values),count,Count.count_of(values,count,accepted))
  +room = Rooms.cells_room(limits,values,axis,position,nonempty,values_room(limits,values,count,exact,bounded))
  +positive = Rooms.cells_positive(limits,values,axis,position,nonempty,values_room(limits,values,count,exact,bounded))
  +laid = {Public.Laid{inside,{==},Public.cells_restored(values,axis,position),Global.Counted{count,accepted},{==}} : Public.Laid<cells,axis,lanes,position,values>}
  +after_spreads = TiledIndexedValues.spread_of(total,after,0n,positive,TiledIndexedValues.after_facts(values,axis,lanes,position,cells,laid))
  +entry = entry_cases(Nat.is_lt(axis,List.length(&2,U32,values)),limits,values,axis,position,count,{==},exact,bounded,positive,after_spreads)
  +unit = unit_of(position,lanes)
  +divides = unit_multiple(position,Nat.is_lt(axis,List.length(&2,U32,values)),values,axis,{==})
  +run = GridRanges.layout_run(cells,Grid.piece_target(),unit,GridRanges.grid_layout(limits,cells,Grid.piece_target(),room,positive,unit,divides))
  %Equal.sym(Maybe<&2,Tiled.Around>,Tiled.around(values,axis,1),Some{Tiled.Around{prefix,lanes,after_axes}},Public.located(values,axis,1,inside)) :
    Public.Contract(~Buffer.Content<F32>,~Contract.stored(~F32),logical,values,Tiled.tiled_around(_,Tensors.Usable{},values,axis,position,workers,Buffer.pack(~F32,source)))
  KernelGemm.Both{
    %Equal.sym(Cells.Cells,Public.geometry_of(values,axis,position),entry_geometry(position,outer,after,lanes),Public.geometry_exact(values,axis,position,inside)) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Buffer.Content<F32>>,~Representation.lifted(~Buffer.Content<F32>,~Contract.stored(~F32)),Representation.Lifted{logical,_},
        Tensors.Tensor{Pair.snd(Storage.Buffer<F32>,Pieces.Pieces(~Tile.Tile8),Pieces.build(~Storage.Buffer<Tile.Tile8>,~Storage.Buffer<F32>,~Tiled.Tiling,~Tiled.tiled_piece,layout,
          entry_context(position,outer,after,lanes),Buffer.pack(~F32,source))),cells,Tiled.zero_tile(),Tensors.Usable{},workers})
    GridRanges.grid_contract(~Tile.Tile8,~Representation.Lifted<Buffer.Content<F32>>,~Representation.lifted(~Buffer.Content<F32>,~Contract.stored(~F32)),
      Representation.Lifted{logical,entry_geometry(position,outer,after,lanes)},limits,cells,Tiled.zero_tile(),workers,
      Pair.snd(Storage.Buffer<F32>,Pieces.Pieces(~Tile.Tile8),Pieces.build(~Storage.Buffer<Tile.Tile8>,~Storage.Buffer<F32>,~Tiled.Tiling,~Tiled.tiled_piece,layout,
        entry_context(position,outer,after,lanes),Buffer.pack(~F32,source))),
      room,positive,unit,divides,StorageCount.counted_within(limits,cells,room,{==}),
      entry_placed(position,limits,outer,after,lanes,total,entry,layout,run,source)),
    laid}

# The tiled form of nonempty counted values stored in a buffer, along the
# lane axis of their shape, holds the lifted stored values.
def tiled_stored(+values: List<&2,U32>,+workers: U32,+source: Buffer.Model<F32>,+counted: Global.Counted<values>,
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool}) ->
  Public.Contract(~Buffer.Content<F32>,~Contract.stored(~F32),Buffer.contents(~F32,Buffer.pack(~F32,source)),values,
    Tiled.tiled_shape(Tensors.Usable{},values,workers,Buffer.pack(~F32,source))):
  from_stored(values,Tiled.lane_axis_for(values),Grid.Innermost{},workers,source,Public.axis_for_within(values),counted,nonempty)
