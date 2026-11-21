# Cells of channels by a plane with their lanes on the channels and the
# groups outermost, or of a plane of one cell: the cell of group g at offset
# o is cell g * plane + o. Read as a cell source, value c * plane + o of cells
# that hold the lifted values is lane c % 8 of the cell of group c / 8 at o,
# the value of the reader at c * plane + o.
import Base
import ../kernel_tile.bend as Tile
import ../tensor_grid.bend as Grid
import ../tensor_tiled_cells.bend as Cells
import ../machine_limits.bend as Machine
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./tensor_contract.bend as Contract
import ./tensor_tiled_representation.bend as Representation
import ./kernel_tile_refinement.bend as TileLaws
import ./tensor_tiled_values.bend as TiledValues
import ./tensor_tiled_exit.bend as Exit
import ./list_storage_readback.bend as ListRead
import ./nat_order.bend as Order
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition

# The geometry of such cells.
def Planes(+plane: U32,+channels: U32,+position: Grid.Position) -> Cells.Cells:
  Cells.Cells{plane,plane,channels,Grid.groups(channels),position}

# Where the groups lie: outermost, or anywhere for a plane of one cell.
def Outer(+position: Grid.Position,+plane: Nat) -> Data:
  {Grid.same_position(position,Grid.Outermost{}) || Nat.is_eq(plane,1n) == True{} : Bool}

def first_offset(offset: Nat,+group: Nat,+groups: Nat,+inside: {Nat.is_lt(offset,1n) == True{} : Bool}) ->
  {Nat.add(Nat.mul(group,1n),offset) == Nat.add(Nat.mul(offset,groups),group) : Nat}:
  match offset:
    case 0n:
      %Equal.sym(Nat,Nat.add(Nat.mul(group,1n),0n),Nat.mul(group,1n),Addition.zero_right(Nat.mul(group,1n))) : {_ == group : Nat}
      Order.times_one(group)
    case 1n+rest: Empty.absurd({Nat.add(Nat.mul(group,1n),1n+rest) == Nat.add(Nat.mul(1n+rest,groups),group) : Nat},Order.below_zero(rest,inside))

# Cell g * plane + o is the cell of group g at offset o.
def located(position: Grid.Position,+plane_word: U32,+groups: Nat,+plane: Nat,+group: Nat,+offset: Nat,+plane_exact: {U32.to_nat(plane_word) == plane : Nat},
  +offset_inside: {Nat.is_lt(offset,plane) == True{} : Bool},+outer: Outer(position,plane)) ->
  {Nat.add(Nat.mul(group,plane),offset) == Representation.index_at(position,U32.to_nat(plane_word),groups,offset,group) : Nat}:
  match position:
    case Grid.Outermost{}:
      %Equal.sym(Nat,U32.to_nat(plane_word),plane,plane_exact) : {Nat.add(Nat.mul(group,plane),offset) == Nat.add(Nat.mul(group,_),offset) : Nat}
      {==}
    case Grid.Innermost{}:
      +single = Order.equal_of_is_eq(plane,1n,outer)
      %Equal.sym(Nat,plane,1n,single) : {Nat.add(Nat.mul(group,_),offset) == Nat.add(Nat.mul(offset,groups),group) : Nat}
      first_offset(offset,group,groups,%single : {Nat.is_lt(offset,_) == True{} : Bool} offset_inside)

# A value below the divisor is its own remainder, with quotient zero.
def below_split(+value: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool},+below: {Nat.is_lt(value,divisor) == True{} : Bool}) ->
  {Nat.divmod(value,divisor) == (0n,value) : Nat & Nat}:
  Division.unique(value,divisor,0n,value,{==},positive,below)

def below_quotient(+value: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool},+below: {Nat.is_lt(value,divisor) == True{} : Bool}) ->
  {Nat.div(value,divisor) == 0n : Nat}:
  Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(value,divisor),(0n,value),below_split(value,divisor,positive,below))

def below_remainder(+value: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool},+below: {Nat.is_lt(value,divisor) == True{} : Bool}) ->
  {Nat.mod(value,divisor) == value : Nat}:
  Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(value,divisor),(0n,value),below_split(value,divisor,positive,below))

# The logical index of a channel at an offset of the plane.
def offset_index(+plane_word: U32,+channels: U32,+position: Grid.Position,+plane: Nat,+channel: Nat,+offset: Nat,+plane_exact: {U32.to_nat(plane_word) == plane : Nat},
  +plane_positive: {Nat.is_lt(0n,plane) == True{} : Bool},+offset_inside: {Nat.is_lt(offset,plane) == True{} : Bool}) ->
  {Representation.logical_index(Planes(plane_word,channels,position),offset,channel) == Nat.add(Nat.mul(channel,plane),offset) : Nat}:
  +count = U32.to_nat(channels)
  +after = Representation.least_one(U32.to_nat(plane_word))
  +after_exact = Equal.trans(Nat,after,U32.to_nat(plane_word),plane,
    Representation.least_positive(U32.to_nat(plane_word),%Equal.sym(Nat,U32.to_nat(plane_word),plane,plane_exact) : {Nat.is_lt(0n,_) == True{} : Bool} plane_positive),plane_exact)
  %Equal.sym(Nat,Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(offset,after),count),channel),after),Nat.mod(offset,after)),Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(offset,plane),count),channel),plane),Nat.mod(offset,plane)),Equal.cong(Nat,Nat,step => Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(offset,step),count),channel),step),Nat.mod(offset,step)),after,plane,after_exact)) : {_ == Nat.add(Nat.mul(channel,plane),offset) : Nat}
  %Equal.sym(Nat,Nat.div(offset,plane),0n,below_quotient(offset,plane,plane_positive,offset_inside)) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(_,count),channel),plane),Nat.mod(offset,plane)) == Nat.add(Nat.mul(channel,plane),offset) : Nat}
  %Equal.sym(Nat,Nat.mod(offset,plane),offset,below_remainder(offset,plane,plane_positive,offset_inside)) :
    {Nat.add(Nat.mul(channel,plane),_) == Nat.add(Nat.mul(channel,plane),offset) : Nat}
  {==}

# Lane c % 8 of the cell of group c / 8 at an offset holds the value of
# channel c there.
def lane_value(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+plane_word: U32,+channels: U32,+position: Grid.Position,+plane: Nat,+channel: Nat,+offset: Nat,
  +plane_exact: {U32.to_nat(plane_word) == plane : Nat},+plane_positive: {Nat.is_lt(0n,plane) == True{} : Bool},
  +offset_inside: {Nat.is_lt(offset,plane) == True{} : Bool},+channel_inside: {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool}) ->
  {Representation.lane(~Logical,~original,logical,Planes(plane_word,channels,position),offset,Nat.div(channel,8n),Nat.mod(channel,8n)) ==
    original(logical,U32.from_nat(Nat.add(Nat.mul(channel,plane),offset))) : F32}:
  +geometry = Planes(plane_word,channels,position)
  %Equal.sym(F32,Bool.pick(F32,Nat.is_lt(Nat.add(Nat.mod(channel,8n),Nat.mul(Nat.div(channel,8n),8n)),U32.to_nat(channels)),original(logical,U32.from_nat(Representation.logical_index(geometry,offset,Nat.add(Nat.mod(channel,8n),Nat.mul(Nat.div(channel,8n),8n))))),0.0),Bool.pick(F32,Nat.is_lt(channel,U32.to_nat(channels)),original(logical,U32.from_nat(Representation.logical_index(geometry,offset,channel))),0.0),
    Equal.cong(Nat,F32,lane => Bool.pick(F32,Nat.is_lt(lane,U32.to_nat(channels)),original(logical,U32.from_nat(Representation.logical_index(geometry,offset,lane))),0.0),Nat.add(Nat.mod(channel,8n),Nat.mul(Nat.div(channel,8n),8n)),channel,TiledValues.channel_split(channel))) :
    {_ == original(logical,U32.from_nat(Nat.add(Nat.mul(channel,plane),offset))) : F32}
  %Equal.sym(Bool,Nat.is_lt(channel,U32.to_nat(channels)),True{},channel_inside) :
    {Bool.pick(F32,_,original(logical,U32.from_nat(Representation.logical_index(geometry,offset,channel))),0.0) == original(logical,U32.from_nat(Nat.add(Nat.mul(channel,plane),offset))) : F32}
  %Equal.sym(Nat,Representation.logical_index(geometry,offset,channel),Nat.add(Nat.mul(channel,plane),offset),
      offset_index(plane_word,channels,position,plane,channel,offset,plane_exact,plane_positive,offset_inside)) :
    {original(logical,U32.from_nat(_)) == original(logical,U32.from_nat(Nat.add(Nat.mul(channel,plane),offset))) : F32}
  {==}

# The cell of group c / 8 at an offset holds the lifted cell there.
def cell_read(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+plane_word: U32,+channels: U32,+position: Grid.Position,+plane: Nat,
  +tree: Tree.Tree<Tile.Tile8>,+length: Nat,
  +values: Contract.Values(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Planes(plane_word,channels,position)},Buffer.Content{tree,length}),
  +plane_exact: {U32.to_nat(plane_word) == plane : Nat},+outer: Outer(position,plane),
  +bound: {Nat.is_le(U32.to_nat(channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(Nat.mul(U32.to_nat(Grid.groups(channels)),plane),length) == True{} : Bool},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+length_room: {Nat.is_le(length,U32.to_nat(cap)) == True{} : Bool},
  +channel: Nat,+offset: Nat,+offset_inside: {Nat.is_lt(offset,plane) == True{} : Bool},+channel_inside: {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(Nat.mul(Nat.div(channel,8n),plane),offset))) ==
    Representation.cell(~Logical,~original,logical,Planes(plane_word,channels,position),offset,Nat.div(channel,8n)) : Tile.Tile8}:
  +geometry = Planes(plane_word,channels,position)
  +group = Nat.div(channel,8n)
  +number = Nat.add(Nat.mul(group,plane),offset)
  +group_inside = Exit.channel_group(limits,channels,channel,bound,channel_inside)
  +number_inside = Bounds.nat_trans(number,Nat.mul(U32.to_nat(Grid.groups(channels)),plane),length,
    RowMajor.radix(group,U32.to_nat(Grid.groups(channels)),offset,plane,group_inside,offset_inside),cells_room)
  +number_exact = Bounds.roundtrip(number,cap,cap_ok,Order.transitive(number,length,U32.to_nat(cap),Bounds.strict_le(number,length,number_inside),length_room))
  +cell_exact = Equal.trans(Nat,U32.to_nat(U32.from_nat(number)),number,Representation.cell_index(geometry,offset,group),number_exact,
    located(position,plane_word,U32.to_nat(Grid.groups(channels)),plane,group,offset,plane_exact,offset_inside,outer))
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,tree,U32.from_nat(number)),Representation.lifted(~Logical,~original,Representation.Lifted{logical,geometry},U32.from_nat(number)),
    Representation.cell(~Logical,~original,logical,geometry,offset,group),
    Contract.read(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,geometry},Buffer.Content{tree,length},number,values,number_inside),
    Representation.decoded(~Logical,~original,logical,geometry,offset,group,U32.from_nat(number),cell_exact,
      %Equal.sym(Nat,U32.to_nat(plane_word),plane,plane_exact) : {Nat.is_lt(offset,_) == True{} : Bool} offset_inside,group_inside))

# Value c * plane + o of the cell source of such cells is the value of the
# reader there.
def source_value(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+plane_word: U32,+channels: U32,+position: Grid.Position,+plane: Nat,
  +tree: Tree.Tree<Tile.Tile8>,+length: Nat,
  +values: Contract.Values(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Planes(plane_word,channels,position)},Buffer.Content{tree,length}),
  +plane_exact: {U32.to_nat(plane_word) == plane : Nat},+outer: Outer(position,plane),
  +bound: {Nat.is_le(U32.to_nat(channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(Nat.mul(U32.to_nat(Grid.groups(channels)),plane),length) == True{} : Bool},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+length_room: {Nat.is_le(length,U32.to_nat(cap)) == True{} : Bool},
  +index: Nat,+inside: {Nat.is_lt(index,Nat.mul(U32.to_nat(channels),plane)) == True{} : Bool}) ->
  {TiledValues.cell_value(TiledValues.CellSource{tree,U32.to_nat(channels),plane},index) == original(logical,U32.from_nat(index)) : F32}:
  +count = U32.to_nat(channels)
  +positive = Representation.below_positive(index,Nat.mul(count,plane),inside)
  +plane_positive = DivisionOrder.right_factor_positive(count,plane,positive)
  +count_positive = RowMajor.left_factor_positive(count,plane,positive)
  +block = Nat.div(index,plane)
  +offset = Nat.mod(index,plane)
  +block_inside = DivisionOrder.quotient_below(index,count,plane,plane_positive,inside)
  +offset_inside = DivisionOrder.remainder_small(index,plane,plane_positive)
  +geometry = Planes(plane_word,channels,position)
  %Equal.sym(F32,TileLaws.select_lane(Nat.mod(Nat.mod(block,count),8n),Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(block,count),TiledValues.groups_of(count)),Nat.div(Nat.mod(block,count),8n)),plane),offset)))),TileLaws.select_lane(Nat.mod(block,8n),Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(0n,TiledValues.groups_of(count)),Nat.div(block,8n)),plane),offset)))),Equal.cong(Nat & Nat,F32,pair => TileLaws.select_lane(Nat.mod(Pair.snd(Nat,Nat,pair),8n),Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(Pair.fst(Nat,Nat,pair),TiledValues.groups_of(count)),Nat.div(Pair.snd(Nat,Nat,pair),8n)),plane),offset)))),Nat.divmod(block,count),(0n,block),
      below_split(block,count,count_positive,block_inside))) :
    {_ == original(logical,U32.from_nat(index)) : F32}
  %Equal.sym(Tile.Tile8,Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(Nat.mul(Nat.div(block,8n),plane),offset))),
      Representation.cell(~Logical,~original,logical,geometry,offset,Nat.div(block,8n)),
      cell_read(~Logical,~original,logical,limits,plane_word,channels,position,plane,tree,length,values,plane_exact,outer,bound,cells_room,cap,cap_ok,length_room,block,offset,offset_inside,block_inside)) :
    {TileLaws.select_lane(Nat.mod(block,8n),_) == original(logical,U32.from_nat(index)) : F32}
  %Equal.sym(F32,TileLaws.select_lane(Nat.mod(block,8n),Representation.cell(~Logical,~original,logical,geometry,offset,Nat.div(block,8n))),
      Representation.lane(~Logical,~original,logical,geometry,offset,Nat.div(block,8n),Nat.mod(block,8n)),
      Representation.cell_lane(~Logical,~original,logical,geometry,offset,Nat.div(block,8n),Nat.mod(block,8n),Division.lt_succ_le(Nat.mod(block,8n),7n,DivisionOrder.remainder_small(block,8n,{==})))) :
    {_ == original(logical,U32.from_nat(index)) : F32}
  %Equal.sym(Nat,index,Nat.add(Nat.mul(block,plane),offset),DivisionOrder.quotient_exact(index,plane,plane_positive)) : {Representation.lane(~Logical,~original,logical,geometry,offset,Nat.div(block,8n),Nat.mod(block,8n)) == original(logical,U32.from_nat(_)) : F32}
  lane_value(~Logical,~original,logical,plane_word,channels,position,plane,block,offset,plane_exact,plane_positive,offset_inside,block_inside)

# The cell source lists the values of the reader.
def source_listed(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+plane_word: U32,+channels: U32,+position: Grid.Position,+plane: Nat,
  +tree: Tree.Tree<Tile.Tile8>,+length: Nat,
  +values: Contract.Values(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Planes(plane_word,channels,position)},Buffer.Content{tree,length}),
  +plane_exact: {U32.to_nat(plane_word) == plane : Nat},+outer: Outer(position,plane),
  +bound: {Nat.is_le(U32.to_nat(channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(Nat.mul(U32.to_nat(Grid.groups(channels)),plane),length) == True{} : Bool},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+length_room: {Nat.is_le(length,U32.to_nat(cap)) == True{} : Bool},
  count: Nat,+index: Nat,+room: {Nat.is_le(Nat.add(index,count),Nat.mul(U32.to_nat(channels),plane)) == True{} : Bool}) ->
  {ListRead.enumerate(~F32,~TiledValues.CellSource,~TiledValues.cell_value,count,index,TiledValues.CellSource{tree,U32.to_nat(channels),plane}) ==
    Contract.listed(~F32,~Logical,~original,count,index,logical) : List<&2,F32>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +source = {TiledValues.CellSource{tree,U32.to_nat(channels),plane} : TiledValues.CellSource}
      +total = Nat.mul(U32.to_nat(channels),plane)
      %Equal.sym(F32,TiledValues.cell_value(source,index),original(logical,U32.from_nat(index)),
          source_value(~Logical,~original,logical,limits,plane_word,channels,position,plane,tree,length,values,plane_exact,outer,bound,cells_room,cap,cap_ok,length_room,index,
            Order.room_index(index,rest,total,room))) :
        {_ <> ListRead.enumerate(~F32,~TiledValues.CellSource,~TiledValues.cell_value,rest,1n+index,source) == Contract.listed(~F32,~Logical,~original,1n+rest,index,logical) : List<&2,F32>}
      Equal.cong(List<&2,F32>,List<&2,F32>,tail => original(logical,U32.from_nat(index)) <> tail,
        ListRead.enumerate(~F32,~TiledValues.CellSource,~TiledValues.cell_value,rest,1n+index,source),Contract.listed(~F32,~Logical,~original,rest,1n+index,logical),
        source_listed(~Logical,~original,logical,limits,plane_word,channels,position,plane,tree,length,values,plane_exact,outer,bound,cells_room,cap,cap_ok,length_room,rest,1n+index,
          Order.advance_room(index,rest,total,room)))
