# The stored cells of the operand tensors read the logical samples of a dot.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ../machine_limits.bend as Machine
import ../tensor_grid.bend as Grid
import ../tensor_tiled_cells.bend as Cells
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./kernel_tile_refinement.bend as Lanes
import ./machine_limit_bounds.bend as Limits
import ./nat_division_order.bend as Division
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./row_major_index.bend as Index
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./tensor_contract.bend as Contract
import ./tensor_halo_windows.bend as Windows
import ./tensor_move_readback.bend as Held
import ./tensor_point_product_windows.bend as Operands
import ./tensor_point_products.bend as Point
import ./tensor_tiled_public.bend as Public
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_values.bend as TiledValues

def source_sample(~Source: Data,~value: Source -> U32 -> F32,+source: Source,+geometry: Cells.Cells,+tile: Tile.Tile8,+limits: Machine.Limits,
  +outer: Nat,+group: Nat,+selected: Nat,+index: Nat,
  +room: {Nat.is_le(index,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +indexed: {index == Representation.cell_index(geometry,outer,group) : Nat},
  +outer_inside: {Nat.is_lt(outer,Representation.outer_count(geometry)) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,Representation.group_count(geometry)) == True{} : Bool},
  +lane_inside: {Nat.is_lt(selected,8n) == True{} : Bool},
  +lane_active: {Nat.is_lt(Nat.add(selected,Nat.mul(group,8n)),Representation.lane_count(geometry)) == True{} : Bool},
  +read: {tile == Representation.lifted(~Source,~value,Representation.Lifted{source,geometry},U32.from_nat(index)) : Tile.Tile8}) ->
  {Lanes.select_lane(selected,tile) == value(source,U32.from_nat(Representation.logical_index(geometry,outer,Nat.add(selected,Nat.mul(group,8n))))) : F32}:
  +word = U32.from_nat(index)
  +roundtrip = Equal.trans(Nat,U32.to_nat(word),index,Representation.cell_index(geometry,outer,group),
    Bounds.roundtrip(index,Machine.capacity(limits),Limits.capacity_fits(limits),room),indexed)
  +lifted = Representation.lifted(~Source,~value,Representation.Lifted{source,geometry},word)
  +coordinate = Representation.logical_index(geometry,outer,Nat.add(selected,Nat.mul(group,8n)))
  Equal.trans(F32,Lanes.select_lane(selected,tile),Lanes.select_lane(selected,lifted),value(source,U32.from_nat(coordinate)),
    Equal.cong(Tile.Tile8,F32,cell => Lanes.select_lane(selected,cell),tile,lifted,read),
    Representation.decoded_lane(~Source,~value,source,geometry,outer,group,selected,word,roundtrip,outer_inside,group_inside,Order.strict_follow(selected,8n,lane_inside),lane_active))

def buffer_tile(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+buffer: Buffer.Content<Tile.Tile8>,
  +index: Nat,+inside: {Nat.is_lt(index,Buffer.content_length(~Tile.Tile8,buffer)) == True{} : Bool},
  +values: Contract.Values(~Tile.Tile8,~Source,~value,source,buffer)) ->
  {Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,buffer),U32.from_nat(index)) == value(source,U32.from_nat(index)) : Tile.Tile8}:
  Buffer.Content{+tree,+length} = buffer
  Contract.read(~Tile.Tile8,~Source,~value,source,buffer,index,values,inside)

def joined_rows(+first: Nat,+row: Nat,+width: Nat,+column: Nat) ->
  {Nat.add(Nat.mul(first,width),Nat.add(Nat.mul(row,width),column)) == Nat.add(Nat.mul(Nat.add(first,row),width),column) : Nat}:
  +offset = Nat.mul(first,width)
  +next = Nat.mul(row,width)
  +base = Nat.mul(Nat.add(first,row),width)
  Equal.trans(Nat,Nat.add(offset,Nat.add(next,column)),Nat.add(Nat.add(offset,next),column),Nat.add(base,column),
    Equal.sym(Nat,Nat.add(Nat.add(offset,next),column),Nat.add(offset,Nat.add(next,column)),Addition.associative(offset,next,column)),
    Equal.cong(Nat,Nat,at => Nat.add(at,column),Nat.add(offset,next),base,
      Equal.sym(Nat,base,Nat.add(offset,next),Index.distrib(first,row,width))))

def panel_index(+image: Nat,+rows: Nat,+width: Nat,+row: Nat,+column: Nat) ->
  {Nat.add(Nat.mul(image,Nat.mul(rows,width)),Nat.add(Nat.mul(row,width),column)) ==
    Nat.add(Nat.mul(Nat.add(Nat.mul(image,rows),row),width),column) : Nat}:
  %Index.assoc(image,rows,width) :
    {Nat.add(_,Nat.add(Nat.mul(row,width),column)) == Nat.add(Nat.mul(Nat.add(Nat.mul(image,rows),row),width),column) : Nat}
  joined_rows(Nat.mul(image,rows),row,width,column)

def panel_offset(+base: Nat,+row: Nat,+width: Nat,+column: Nat) ->
  {Nat.add(Nat.add(base,Nat.mul(row,width)),column) == Nat.add(base,Nat.add(Nat.mul(row,width),column)) : Nat}:
  Addition.associative(base,Nat.mul(row,width),column)

def offset_channel(+lower: Nat,+group: Nat,+lane: Nat) ->
  {Nat.add(Nat.mul(lower,8n),Nat.add(lane,Nat.mul(group,8n))) == Nat.add(lane,Nat.mul(Nat.add(lower,group),8n)) : Nat}:
  +start = Nat.mul(lower,8n)
  +width = Nat.mul(group,8n)
  +total = Nat.mul(Nat.add(lower,group),8n)
  Equal.trans(Nat,Nat.add(start,Nat.add(lane,width)),Nat.add(start,Nat.add(width,lane)),Nat.add(lane,total),
    Equal.cong(Nat,Nat,at => Nat.add(start,at),Nat.add(lane,width),Nat.add(width,lane),Order.commutative(lane,width)),
    Equal.trans(Nat,Nat.add(start,Nat.add(width,lane)),Nat.add(total,lane),Nat.add(lane,total),
      joined_rows(lower,group,8n,lane),Order.commutative(total,lane)))

def row_geometry(+outer: U32,+channels: U32) -> Cells.Cells:
  Cells.Cells{outer,1,channels,Gemm.blocks(channels),Grid.Innermost{}}

def one_times(+value: U32) -> {(1 * value : U32) == value : U32}:
  Equal.trans(U32,(1 * value : U32),(value * 1 : U32),value,Arithmetic.multiply_commutative(1,value),Arithmetic.multiply_one(value))

def matrix_geometry_of(+images: U32,+rows: U32,+channels: U32,+position: Grid.Position) ->
  {Public.geometry_of([images,rows,channels],2n,position) == Cells.Cells{(images * rows : U32),1,channels,Gemm.blocks(channels),position} : Cells.Cells}:
  +outer = (1 * images * rows : U32)
  +groups = Gemm.blocks(channels)
  %Equal.sym(U32,(outer * 1 : U32),outer,Arithmetic.multiply_one(outer)) :
    {Cells.Cells{_,1,channels,groups,position} == Cells.Cells{(images * rows : U32),1,channels,Gemm.blocks(channels),position} : Cells.Cells}
  %Equal.sym(U32,(1 * images : U32),images,one_times(images)) :
    {Cells.Cells{(_ * rows : U32),1,channels,groups,position} == Cells.Cells{(images * rows : U32),1,channels,Gemm.blocks(channels),position} : Cells.Cells}
  {==}

def bias_geometry_of(+channels: U32) ->
  {Public.geometry_of([channels],0n,Grid.Innermost{}) == row_geometry(1,channels) : Cells.Cells}:
  {==}

def row_sample(~Source: Data,~value: Source -> U32 -> F32,+source: Source,+limits: Machine.Limits,+rows: U32,+channels: U32,
  +row: Nat,+group: Nat,+lane: Nat,+tile: Tile.Tile8,
  +room: {Nat.is_le(Nat.mul(U32.to_nat(rows),U32.to_nat(Gemm.blocks(channels))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,U32.to_nat(Gemm.blocks(channels))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool},
  +active: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(channels)) == True{} : Bool},
  +read: {tile == Representation.lifted(~Source,~value,Representation.Lifted{source,row_geometry(rows,channels)},
    U32.from_nat(Nat.add(Nat.mul(row,U32.to_nat(Gemm.blocks(channels))),group))) : Tile.Tile8}) ->
  {Lanes.select_lane(lane,tile) == value(source,U32.from_nat(Nat.add(Nat.mul(row,U32.to_nat(channels)),Nat.add(lane,Nat.mul(group,8n))))) : F32}:
  +groups = U32.to_nat(Gemm.blocks(channels))
  +rows_extent = U32.to_nat(rows)
  +geometry = row_geometry(rows,channels)
  +channel = Nat.add(lane,Nat.mul(group,8n))
  +index = Nat.add(Nat.mul(row,groups),group)
  +inside = Index.radix(row,rows_extent,group,groups,row_inside,group_inside)
  %Representation.last_axis_index(geometry,row,channel,{==}) :
    {Lanes.select_lane(lane,tile) == value(source,U32.from_nat(_)) : F32}
  source_sample(~Source,~value,source,geometry,tile,limits,row,group,lane,index,
    Bounds.strict_le(index,U32.to_nat(Machine.capacity(limits)),Order.lt_le(index,Nat.mul(rows_extent,groups),U32.to_nat(Machine.capacity(limits)),inside,room)),
    {==},row_inside,group_inside,lane_inside,active,read)

def lane_of_tap(~Source: Data,~value: Source -> U32 -> F32,+source: Source,+limits: Machine.Limits,+rows: U32,+channels: U32,+row: Nat,+tap: Nat,+tile: Tile.Tile8,
  +room: {Nat.is_le(Nat.mul(U32.to_nat(rows),U32.to_nat(Gemm.blocks(channels))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +groups: TiledValues.Groups<limits,channels>,
  +row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},
  +tap_inside: {Nat.is_lt(tap,U32.to_nat(channels)) == True{} : Bool},
  +read: {tile == Representation.lifted(~Source,~value,Representation.Lifted{source,row_geometry(rows,channels)},
    U32.from_nat(Nat.add(Nat.mul(row,U32.to_nat(Gemm.blocks(channels))),Nat.div(tap,8n)))) : Tile.Tile8}) ->
  {Lanes.select_lane(Nat.mod(tap,8n),tile) == value(source,U32.from_nat(Nat.add(Nat.mul(row,U32.to_nat(channels)),tap))) : F32}:
  %TiledValues.channel_split(tap) : {Lanes.select_lane(Nat.mod(tap,8n),tile) == value(source,U32.from_nat(Nat.add(Nat.mul(row,U32.to_nat(channels)),_))) : F32}
  row_sample(~Source,~value,source,limits,rows,channels,row,Nat.div(tap,8n),Nat.mod(tap,8n),tile,room,row_inside,
    TiledValues.group_inside(limits,channels,tap,groups,tap_inside),Division.remainder_small(tap,8n,{==}),
    %Equal.sym(Nat,Nat.add(Nat.mod(tap,8n),Nat.mul(Nat.div(tap,8n),8n)),tap,TiledValues.channel_split(tap)) : {Nat.is_lt(_,U32.to_nat(channels)) == True{} : Bool}
    tap_inside,read)

def shared_geometry(+inner: U32,+channels: U32) -> Cells.Cells:
  Cells.Cells{inner,inner,channels,Gemm.blocks(channels),Grid.Outermost{}}

def shared_geometry_of(+inner: U32,+channels: U32) ->
  {Public.geometry_of([channels,inner],0n,Grid.Outermost{}) == shared_geometry(inner,channels) : Cells.Cells}:
  +width = (inner * 1 : U32)
  +groups = Gemm.blocks(channels)
  %Equal.sym(U32,(1 * width : U32),width,one_times(width)) :
    {Cells.Cells{_,width,channels,groups,Grid.Outermost{}} == shared_geometry(inner,channels) : Cells.Cells}
  %Equal.sym(U32,width,inner,Arithmetic.multiply_one(inner)) :
    {Cells.Cells{_,_,channels,groups,Grid.Outermost{}} == shared_geometry(inner,channels) : Cells.Cells}
  {==}

def shared_index(+inner: U32,+channels: U32,+tap: Nat,+output: Nat,+inside: {Nat.is_lt(tap,U32.to_nat(inner)) == True{} : Bool}) ->
  {Representation.logical_index(shared_geometry(inner,channels),tap,output) == Nat.add(Nat.mul(output,U32.to_nat(inner)),tap) : Nat}:
  +geometry = shared_geometry(inner,channels)
  +width = U32.to_nat(inner)
  +after = Representation.after_count(geometry)
  +positive = Order.positive_bound(tap,width,inside)
  +exact = Representation.least_positive(width,positive)
  +from = Nat.add(Nat.mul(0n,after),tap)
  %Equal.sym(Nat,tap,from,{==}) :
    {Representation.logical_index(geometry,_,output) == Nat.add(Nat.mul(output,width),tap) : Nat}
  Equal.trans(Nat,Representation.logical_index(geometry,from,output),Nat.add(Nat.mul(output,after),tap),Nat.add(Nat.mul(output,width),tap),
    Representation.logical_radix(geometry,0n,tap,output,
      %Equal.sym(Nat,after,width,exact) : {Nat.is_lt(tap,_) == True{} : Bool} inside),
    Equal.cong(Nat,Nat,size => Nat.add(Nat.mul(output,size),tap),after,width,exact))

def shared_sample(~Source: Data,~value: Source -> U32 -> F32,+source: Source,+limits: Machine.Limits,+inner: U32,+channels: U32,
  +tap: Nat,+group: Nat,+lane: Nat,+tile: Tile.Tile8,
  +room: {Nat.is_le(Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +tap_inside: {Nat.is_lt(tap,U32.to_nat(inner)) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,U32.to_nat(Gemm.blocks(channels))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool},
  +active: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(channels)) == True{} : Bool},
  +read: {tile == Representation.lifted(~Source,~value,Representation.Lifted{source,shared_geometry(inner,channels)},
    U32.from_nat(Nat.add(Nat.mul(group,U32.to_nat(inner)),tap))) : Tile.Tile8}) ->
  {Lanes.select_lane(lane,tile) == value(source,U32.from_nat(Nat.add(Nat.mul(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(inner)),tap))) : F32}:
  +geometry = shared_geometry(inner,channels)
  +width = U32.to_nat(inner)
  +groups = U32.to_nat(Gemm.blocks(channels))
  +index = Nat.add(Nat.mul(group,width),tap)
  +inside = Index.radix(group,groups,tap,width,group_inside,tap_inside)
  +channel = Nat.add(lane,Nat.mul(group,8n))
  %shared_index(inner,channels,tap,channel,tap_inside) :
    {Lanes.select_lane(lane,tile) == value(source,U32.from_nat(_)) : F32}
  source_sample(~Source,~value,source,geometry,tile,limits,tap,group,lane,index,
    Bounds.strict_le(index,U32.to_nat(Machine.capacity(limits)),Order.lt_le(index,Nat.mul(groups,width),U32.to_nat(Machine.capacity(limits)),inside,room)),
    {==},tap_inside,group_inside,lane_inside,active,read)

def copied_weight_sample(~Source: Data,~value: Source -> U32 -> F32,+source: Source,+limits: Machine.Limits,+inner: U32,+channels: U32,
  +holding: Held.Holding<Tile.Tile8>,+first: U32,+count: U32,+first_group: Nat,+group_count: Nat,
  +first_exact: {U32.to_nat(first) == Nat.mul(first_group,U32.to_nat(inner)) : Nat},
  +count_exact: {U32.to_nat(count) == Nat.mul(group_count,U32.to_nat(inner)) : Nat},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +contains: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner))) == True{} : Bool},
  +window: Windows.TaskWindow<Tile.Tile8,Representation.Lifted<Source>,Representation.lifted(~Source,~value),
    Representation.Lifted{source,shared_geometry(inner,channels)},holding,first,(first + count : U32),Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(inner))>,
  +tap: Nat,+group: Nat,+lane: Nat,
  +tap_inside: {Nat.is_lt(tap,U32.to_nat(inner)) == True{} : Bool},
  +local_group: {Nat.is_lt(group,group_count) == True{} : Bool},
  +global_group: {Nat.is_lt(Nat.add(first_group,group),U32.to_nat(Gemm.blocks(channels))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool},
  +active: {Nat.is_lt(Nat.add(lane,Nat.mul(Nat.add(first_group,group),8n)),U32.to_nat(channels)) == True{} : Bool}) ->
  {Lanes.select_lane(lane,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,Operands.copy_model(holding,first,count)),U32.from_nat(Nat.add(Nat.mul(group,U32.to_nat(inner)),tap)))) ==
    value(source,U32.from_nat(Nat.add(Nat.mul(Nat.add(lane,Nat.mul(Nat.add(first_group,group),8n)),U32.to_nat(inner)),tap))) : F32}:
  +width = U32.to_nat(inner)
  +local = Nat.add(Nat.mul(group,width),tap)
  +global = Nat.add(Nat.mul(Nat.add(first_group,group),width),tap)
  +total = Nat.mul(U32.to_nat(Gemm.blocks(channels)),width)
  +geometry = shared_geometry(inner,channels)
  +expected = {Representation.Lifted{source,geometry} : Representation.Lifted<Source>}
  +tile = Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,Operands.copy_model(holding,first,count)),U32.from_nat(local))
  +inside = {%Equal.sym(Nat,U32.to_nat(count),Nat.mul(group_count,width),count_exact) : {Nat.is_lt(local,_) == True{} : Bool} Index.radix(group,group_count,tap,width,local_group,tap_inside) :
    {Nat.is_lt(local,U32.to_nat(count)) == True{} : Bool}}
  +exact = {%joined_rows(first_group,group,width,tap) :
    {tile == Representation.lifted(~Source,~value,expected,U32.from_nat(_)) : Tile.Tile8}
    %first_exact :
      {tile == Representation.lifted(~Source,~value,expected,U32.from_nat(Nat.add(_,local))) : Tile.Tile8}
    Operands.copied_read(~Representation.Lifted<Source>,~Representation.lifted(~Source,~value),expected,limits,holding,first,count,total,room,contains,window,local,inside) :
    {tile == Representation.lifted(~Source,~value,expected,U32.from_nat(global)) : Tile.Tile8}}
  shared_sample(~Source,~value,source,limits,inner,channels,tap,Nat.add(first_group,group),lane,tile,room,tap_inside,global_group,lane_inside,active,exact)

def band_row_sample(~Source: Data,~value: Source -> U32 -> F32,+source: Source,+limits: Machine.Limits,+rows: U32,+channels: U32,
  +holding: Held.Holding<Tile.Tile8>,+first: U32,+count: U32,+first_row: Nat,+row_count: Nat,
  +first_exact: {U32.to_nat(first) == Nat.mul(first_row,U32.to_nat(Gemm.blocks(channels))) : Nat},
  +count_exact: {U32.to_nat(count) == Nat.mul(row_count,U32.to_nat(Gemm.blocks(channels))) : Nat},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(rows),U32.to_nat(Gemm.blocks(channels))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +contains: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(U32.to_nat(rows),U32.to_nat(Gemm.blocks(channels)))) == True{} : Bool},
  +window: Windows.TaskWindow<Tile.Tile8,Representation.Lifted<Source>,Representation.lifted(~Source,~value),
    Representation.Lifted{source,row_geometry(rows,channels)},holding,first,(first + count : U32),Nat.mul(U32.to_nat(rows),U32.to_nat(Gemm.blocks(channels)))>,
  +row: Nat,+group: Nat,+lane: Nat,
  +local_row: {Nat.is_lt(row,row_count) == True{} : Bool},
  +global_row: {Nat.is_lt(Nat.add(first_row,row),U32.to_nat(rows)) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,U32.to_nat(Gemm.blocks(channels))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool},
  +active: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(channels)) == True{} : Bool}) ->
  {Lanes.select_lane(lane,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,Operands.band_model(holding,first,count)),
      U32.from_nat(Nat.add(U32.to_nat(Operands.band_base(holding,first,count)),Nat.add(Nat.mul(row,U32.to_nat(Gemm.blocks(channels))),group))))) ==
    value(source,U32.from_nat(Nat.add(Nat.mul(Nat.add(first_row,row),U32.to_nat(channels)),Nat.add(lane,Nat.mul(group,8n))))) : F32}:
  +width = U32.to_nat(Gemm.blocks(channels))
  +local = Nat.add(Nat.mul(row,width),group)
  +global = Nat.add(Nat.mul(Nat.add(first_row,row),width),group)
  +total = Nat.mul(U32.to_nat(rows),width)
  +geometry = row_geometry(rows,channels)
  +expected = {Representation.Lifted{source,geometry} : Representation.Lifted<Source>}
  +tile = Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,Operands.band_model(holding,first,count)),U32.from_nat(Nat.add(U32.to_nat(Operands.band_base(holding,first,count)),local)))
  +inside = {%Equal.sym(Nat,U32.to_nat(count),Nat.mul(row_count,width),count_exact) : {Nat.is_lt(local,_) == True{} : Bool} Index.radix(row,row_count,group,width,local_row,group_inside) :
    {Nat.is_lt(local,U32.to_nat(count)) == True{} : Bool}}
  +exact = {%joined_rows(first_row,row,width,group) :
    {tile == Representation.lifted(~Source,~value,expected,U32.from_nat(_)) : Tile.Tile8}
    %first_exact :
      {tile == Representation.lifted(~Source,~value,expected,U32.from_nat(Nat.add(_,local))) : Tile.Tile8}
    Operands.band_read(~Representation.Lifted<Source>,~Representation.lifted(~Source,~value),expected,limits,holding,first,count,total,room,contains,window,local,inside) :
    {tile == Representation.lifted(~Source,~value,expected,U32.from_nat(global)) : Tile.Tile8}}
  row_sample(~Source,~value,source,limits,rows,channels,Nat.add(first_row,row),group,lane,tile,room,global_row,group_inside,lane_inside,active,exact)

def joined_input_sample(~Source: Data,~value: Source -> U32 -> F32,+source: Source,+limits: Machine.Limits,+rows: U32,+channels: U32,
  +buffer: Buffer.Content<Tile.Tile8>,+row: Nat,+tap: Nat,
  +room: {Nat.is_le(Nat.mul(U32.to_nat(rows),U32.to_nat(Gemm.blocks(channels))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +length: {Buffer.content_length(~Tile.Tile8,buffer) == Nat.mul(U32.to_nat(rows),U32.to_nat(Gemm.blocks(channels))) : Nat},
  +values: Contract.Values(~Tile.Tile8,~Representation.Lifted<Source>,~Representation.lifted(~Source,~value),Representation.Lifted{source,row_geometry(rows,channels)},buffer),
  +groups: TiledValues.Groups<limits,channels>,
  +row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},+tap_inside: {Nat.is_lt(tap,U32.to_nat(channels)) == True{} : Bool}) ->
  {Lanes.select_lane(Nat.mod(tap,8n),Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,buffer),U32.from_nat(Nat.add(Nat.mul(row,U32.to_nat(Gemm.blocks(channels))),Nat.div(tap,8n))))) ==
    value(source,U32.from_nat(Nat.add(Nat.mul(row,U32.to_nat(channels)),tap))) : F32}:
  +width = U32.to_nat(Gemm.blocks(channels))
  +group = Nat.div(tap,8n)
  +index = Nat.add(Nat.mul(row,width),group)
  +tile = Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,buffer),U32.from_nat(index))
  +inside = Index.radix(row,U32.to_nat(rows),group,width,row_inside,TiledValues.group_inside(limits,channels,tap,groups,tap_inside))
  lane_of_tap(~Source,~value,source,limits,rows,channels,row,tap,tile,room,groups,row_inside,tap_inside,
    buffer_tile(~Representation.Lifted<Source>,~Representation.lifted(~Source,~value),Representation.Lifted{source,row_geometry(rows,channels)},buffer,index,
      %Equal.sym(Nat,Buffer.content_length(~Tile.Tile8,buffer),Nat.mul(U32.to_nat(rows),width),length) : {Nat.is_lt(index,_) == True{} : Bool}
      inside,values))

def copied_bias_sample(~Source: Data,~value: Source -> U32 -> F32,+source: Source,+limits: Machine.Limits,+channels: U32,
  +bias: Buffer.Model<Tile.Tile8>,+first: U32,+count: U32,+group: Nat,+lane: Nat,
  +room: {Nat.is_le(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +length: {Buffer.length(~Tile.Tile8,bias) == U32.to_nat(Gemm.blocks(channels)) : Nat},
  +values: Contract.Values(~Tile.Tile8,~Representation.Lifted<Source>,~Representation.lifted(~Source,~value),Representation.Lifted{source,row_geometry(1,channels)},Buffer.content(~Tile.Tile8,bias)),
  +allocation: {U32.is_le(count,1073741824) == True{} : Bool},+local_group: {Nat.is_lt(group,U32.to_nat(count)) == True{} : Bool},
  +global_group: {Nat.is_lt(Nat.add(U32.to_nat(first),group),U32.to_nat(Gemm.blocks(channels))) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool},
  +active: {Nat.is_lt(Nat.add(lane,Nat.mul(Nat.add(U32.to_nat(first),group),8n)),U32.to_nat(channels)) == True{} : Bool}) ->
  {Lanes.select_lane(lane,Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,Point.shared_bias_buffer(Operands.bias_copy(bias,first,count)))),U32.from_nat(group))) ==
    value(source,U32.from_nat(Nat.add(lane,Nat.mul(Nat.add(U32.to_nat(first),group),8n)))) : F32}:
  Buffer.Model{+tree,+length_word,+certificate} = bias
  +global = Nat.add(U32.to_nat(first),group)
  +logical = {Representation.Lifted{source,row_geometry(1,channels)} : Representation.Lifted<Source>}
  +local = Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,Point.shared_bias_buffer(Operands.bias_copy(bias,first,count)))),U32.from_nat(group))
  +actual = Tree.read(Tile.Tile8,tree,U32.from_nat(global))
  +target = Representation.lifted(~Source,~value,logical,U32.from_nat(global))
  +holds = Equal.trans(Tile.Tile8,local,actual,target,Operands.bias_copy_read(bias,first,count,allocation,group,local_group),
    buffer_tile(~Representation.Lifted<Source>,~Representation.lifted(~Source,~value),logical,Buffer.content(~Tile.Tile8,bias),global,
      %Equal.sym(Nat,length_word,U32.to_nat(Gemm.blocks(channels)),length) : {Nat.is_lt(global,_) == True{} : Bool} global_group,values))
  row_sample(~Source,~value,source,limits,1,channels,0n,global,lane,local,
    %Equal.sym(Nat,Nat.add(U32.to_nat(Gemm.blocks(channels)),0n),U32.to_nat(Gemm.blocks(channels)),
      Addition.zero_right(U32.to_nat(Gemm.blocks(channels)))) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    room,{==},global_group,lane_inside,active,holds)
