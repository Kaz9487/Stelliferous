# The operand windows of a point-product task read their original tensors.
import Base
import ../kernel_tile.bend as Tile
import ../machine_limits.bend as Machine
import ../storage_copy.bend as Copy
import ../storage_buffer.bend as Storage
import ../tensor.bend as Tensors
import ../tensor_move.bend as Move
import ../tensor_point_products.bend as Products
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./contiguous_copy.bend as CopyWrites
import ./contiguous_copy_readback.bend as CopyReadback
import ./machine_limit_bounds.bend as Limits
import ./nat_order.bend as Order
import ./storage_allocation_refinement.bend as Allocation
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./storage_tree_model.bend as Tree
import ./tensor_allocation_depth.bend as Depth
import ./tensor_halo_windows.bend as Windows
import ./tensor_move_readback.bend as Held
import ./tensor_point_products.bend as Point
import ./tensor_placed_parts.bend as Placed
import ./word_arithmetic.bend as Words
import ./word_power_of_two.bend as Power
import ./write_program.bend as Writes

def available(+limits: Machine.Limits,+start: U32,+count: U32,+total: Nat,
  +room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +contains: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(count)),total) == True{} : Bool}) ->
  {U32.to_nat((start + count : U32)) == Nat.add(U32.to_nat(start),U32.to_nat(count)) : Nat}:
  +extent = Nat.add(U32.to_nat(start),U32.to_nat(count))
  Arithmetic.sum_exact(Machine.capacity(limits),Limits.capacity_fits(limits),start,count,U32.to_nat(start),U32.to_nat(count),{==},{==},
    Order.transitive(extent,total,U32.to_nat(Machine.capacity(limits)),contains,room))

def window_count(+limits: Machine.Limits,+start: U32,+count: U32,+total: Nat,
  +room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +contains: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(count)),total) == True{} : Bool}) ->
  {U32.is_le(count,1073741824) == True{} : Bool}:
  +extent = Nat.add(U32.to_nat(start),U32.to_nat(count))
  +size = U32.to_nat(count)
  +capacity = U32.to_nat(Machine.capacity(limits))
  Limits.count_allocatable(limits,count,Order.transitive(size,extent,capacity,Order.right_le(U32.to_nat(start),size),
    Order.transitive(extent,total,capacity,contains,room)))

def copy_model(+holding: Held.Holding<Tile.Tile8>,+start: U32,+count: U32) -> Buffer.Model<Tile.Tile8>:
  Point.copied_weight_model(Move.copied(~Tile.Tile8,Held.pack(~Tile.Tile8,holding),start,count,Products.zero()))

def copied_buffer(+holding: Held.Holding<Tile.Tile8>,+start: U32,+count: U32) -> Storage.Buffer<Tile.Tile8>:
  Pair.snd(Move.Holding<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,Move.copied(~Tile.Tile8,Held.pack(~Tile.Tile8,holding),start,count,Products.zero()))
def copied_read(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+limits: Machine.Limits,+holding: Held.Holding<Tile.Tile8>,
  +start: U32,+count: U32,+total: Nat,+room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +contains: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(count)),total) == True{} : Bool},
  window: Windows.TaskWindow<Tile.Tile8,Source,value,source,holding,start,(start + count : U32),total>,
  +index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,copy_model(holding,start,count)),U32.from_nat(index)) ==
    value(source,U32.from_nat(Nat.add(U32.to_nat(start),index))) : Tile.Tile8}:
  Windows.TaskWindow{+high,+reaches,+within,+tiled,+values} = window
  +cap = Machine.capacity(limits)
  +from = U32.to_nat(start)
  +model = copy_model(holding,start,count)
  +tree = Buffer.tree(~Tile.Tile8,model)
  +contents = Buffer.content(~Tile.Tile8,model)
  +actual_tree = Buffer.content_tree(~Tile.Tile8,Buffer.contents(~Tile.Tile8,copied_buffer(holding,start,count)))
  +result = value(source,U32.from_nat(Nat.add(from,index)))
  +projection = Point.copied_weight_projection(Move.copied(~Tile.Tile8,Held.pack(~Tile.Tile8,holding),start,count,Products.zero()))
  +same = Equal.trans(Tree.Tree<Tile.Tile8>,actual_tree,Buffer.content_tree(~Tile.Tile8,contents),tree,
    Equal.cong(Buffer.Content<Tile.Tile8>,Tree.Tree<Tile.Tile8>,Buffer.content_tree(~Tile.Tile8),Buffer.contents(~Tile.Tile8,copied_buffer(holding,start,count)),contents,
      %Equal.sym(Buffer.Model<Tile.Tile8>,model,Buffer.remodel(~Tile.Tile8,copied_buffer(holding,start,count)),projection) :
        {Buffer.contents(~Tile.Tile8,copied_buffer(holding,start,count)) == Buffer.content(~Tile.Tile8,_) : Buffer.Content<Tile.Tile8>}
      Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,copied_buffer(holding,start,count))),Buffer.contents(~Tile.Tile8,copied_buffer(holding,start,count)),Buffer.remodel_content(~Tile.Tile8,copied_buffer(holding,start,count)))),
    Buffer.content_tree_of(~Tile.Tile8,model))
  %same : {Tree.read(Tile.Tile8,_,U32.from_nat(index)) == result : Tile.Tile8}
  Held.copied_cell(~Tile.Tile8,~Source,~value,source,holding,start,count,Products.zero(),from,high,cap,Limits.capacity_fits(limits),
    Order.transitive(high,total,U32.to_nat(cap),within,room),tiled,values,Order.reflexive(from),
    %available(limits,start,count,total,room,contains) : {Nat.is_le(_,high) == True{} : Bool} reaches,
    window_count(limits,start,count,total,room,contains),index,inside)

def band(+holding: Held.Holding<Tile.Tile8>,+start: U32,+count: U32) -> Move.Band<Tile.Tile8>:
  Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,holding),start,count,Products.zero(),Move.spare(~Tile.Tile8,Products.zero()))

def band_model(+holding: Held.Holding<Tile.Tile8>,+start: U32,+count: U32) -> Buffer.Model<Tile.Tile8>:
  Buffer.remodel(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,band(holding,start,count)))

def band_base(+holding: Held.Holding<Tile.Tile8>,+start: U32,+count: U32) -> U32:
  Held.band_offset(~Tile.Tile8,band(holding,start,count))

def band_read(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+limits: Machine.Limits,+holding: Held.Holding<Tile.Tile8>,
  +start: U32,+count: U32,+total: Nat,+room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +contains: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(count)),total) == True{} : Bool},
  window: Windows.TaskWindow<Tile.Tile8,Source,value,source,holding,start,(start + count : U32),total>,
  +index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,band_model(holding,start,count)),U32.from_nat(Nat.add(U32.to_nat(band_base(holding,start,count)),index))) ==
    value(source,U32.from_nat(Nat.add(U32.to_nat(start),index))) : Tile.Tile8}:
  Windows.TaskWindow{+high,+reaches,+within,+tiled,+values} = window
  +cap = Machine.capacity(limits)
  +from = U32.to_nat(start)
  +fill = Products.zero()
  +spare = Held.fresh_spare(~Tile.Tile8,fill)
  %Equal.sym(Move.Spare<Tile.Tile8>,Move.spare(~Tile.Tile8,fill),Held.pack_spare(~Tile.Tile8,spare),Held.spare_packed(~Tile.Tile8,fill)) :
    {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Held.band_buffer(~Tile.Tile8,
      Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,holding),start,count,fill,_)))),
      U32.from_nat(Nat.add(U32.to_nat(Held.band_offset(~Tile.Tile8,Move.band(~Tile.Tile8,Held.pack(~Tile.Tile8,holding),start,count,fill,_))),index))) ==
      value(source,U32.from_nat(Nat.add(from,index))) : Tile.Tile8}
  Held.band_model_value(~Tile.Tile8,~Source,~value,source,holding,start,count,fill,spare,Held.fresh_spare_room(~Tile.Tile8,fill),
    from,high,cap,Limits.capacity_fits(limits),Order.transitive(high,total,U32.to_nat(cap),within,room),tiled,values,Order.reflexive(from),
    %available(limits,start,count,total,room,contains) : {Nat.is_le(_,high) == True{} : Bool} reaches,
    window_count(limits,start,count,total,room,contains),index,inside)

def bias_copy(+source: Buffer.Model<Tile.Tile8>,+first: U32,+count: U32) -> Point.Resources():
  Copy.range(~Tile.Tile8,count,first,0,Buffer.pack(~Tile.Tile8,source),Tensors.buffer(~Tile.Tile8,Products.zero(),count))

def bias_copy_content(+source: Buffer.Model<Tile.Tile8>,+first: U32,+count: U32) ->
  {Buffer.content(~Tile.Tile8,Point.shared_bias_buffer(bias_copy(source,first,count))) ==
    Buffer.Content{Writes.pure(Tile.Tile8,CopyWrites.writes(~Tile.Tile8,U32.to_nat(count),Buffer.tree(~Tile.Tile8,source),0,first,0),
      Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Products.zero())),U32.to_nat(count)} : Buffer.Content<Tile.Tile8>}:
  +fill = Products.zero()
  +tree = Buffer.tree(~Tile.Tile8,source)
  +fresh = Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count))
  +writes = CopyWrites.writes(~Tile.Tile8,U32.to_nat(count),tree,0,first,0)
  +model = Point.shared_bias_buffer(bias_copy(source,first,count))
  -actual = Tensors.buffer(~Tile.Tile8,fill,count)
  -packed = Buffer.pack(~Tile.Tile8,fresh)
  %Equal.sym(Buffer.Model<Tile.Tile8>,model,Buffer.remodel(~Tile.Tile8,Pair.snd(Products.Cells(),Products.Cells(),bias_copy(source,first,count))),
    Point.shared_bias_projection(bias_copy(source,first,count))) :
    {Buffer.content(~Tile.Tile8,_) == Buffer.Content{Writes.pure(Tile.Tile8,writes,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),fill)),U32.to_nat(count)} : Buffer.Content<Tile.Tile8>}
  %Buffer.remodeled(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,fill,count)) :
    {Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Pair.snd(Products.Cells(),Products.Cells(),Copy.range(~Tile.Tile8,count,first,0,Buffer.pack(~Tile.Tile8,source),_)))) ==
      Buffer.Content{Writes.pure(Tile.Tile8,writes,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),fill)),U32.to_nat(count)} : Buffer.Content<Tile.Tile8>}
  %Allocation.buffer_tree(~Tile.Tile8,fill,count) :
    {Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Pair.snd(Products.Cells(),Products.Cells(),Copy.range(~Tile.Tile8,count,first,0,Buffer.pack(~Tile.Tile8,source),packed)))) ==
      Buffer.Content{Writes.pure(Tile.Tile8,writes,_),U32.to_nat(count)} : Buffer.Content<Tile.Tile8>}
  %Allocation.buffer_length(~Tile.Tile8,fill,count) :
    {Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Pair.snd(Products.Cells(),Products.Cells(),Copy.range(~Tile.Tile8,count,first,0,Buffer.pack(~Tile.Tile8,source),packed)))) ==
      Buffer.Content{Writes.pure(Tile.Tile8,writes,Buffer.tree(~Tile.Tile8,fresh)),_} : Buffer.Content<Tile.Tile8>}
  Placed.copied_from_content(~Tile.Tile8,count,first,0,source,fresh)

def bias_copy_read(+source: Buffer.Model<Tile.Tile8>,+first: U32,+count: U32,
  +allocatable: {U32.is_le(count,1073741824) == True{} : Bool},+index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,Point.shared_bias_buffer(bias_copy(source,first,count)))),U32.from_nat(index)) ==
    Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,source),U32.from_nat(Nat.add(U32.to_nat(first),index))) : Tile.Tile8}:
  +size = U32.to_nat(count)
  +depth = Tensors.depth_for(count)
  +fill = Products.zero()
  +tree = Buffer.tree(~Tile.Tile8,source)
  +target = Tree.fresh(Tile.Tile8,depth,fill)
  +depth_fits = Depth.bounded(count,allocatable)
  +depth_ok = Depth.allocation_depth(count,depth_fits)
  +capacity = Tree.capacity(Tile.Tile8,target)
  +power = U32.shln(1,depth)
  +capacity_exact = Shape.capacity(Tile.Tile8,depth,target,Shape.fresh(Tile.Tile8,depth,fill))
  +written = Writes.pure(Tile.Tile8,CopyWrites.writes(~Tile.Tile8,size,tree,0,first,0),target)
  %Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,Point.shared_bias_buffer(bias_copy(source,first,count))),Buffer.Content{written,size},
    bias_copy_content(source,first,count)) :
    {Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,_),U32.from_nat(index)) == Tree.read(Tile.Tile8,tree,U32.from_nat(Nat.add(U32.to_nat(first),index))) : Tile.Tile8}
  %Words.native_add(U32.to_nat(first),index) :
    {Tree.read(Tile.Tile8,written,U32.from_nat(index)) == Tree.read(Tile.Tile8,tree,_) : Tile.Tile8}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(first)),first,Arithmetic.native_roundtrip(first)) : {Tree.read(Tile.Tile8,written,U32.from_nat(index)) == Tree.read(Tile.Tile8,tree,(_ + U32.from_nat(index) : U32)) : Tile.Tile8}
  CopyReadback.fresh_read(~Tile.Tile8,size,index,tree,first,depth,fill,depth_ok,
    %Equal.sym(U32,capacity,power,capacity_exact) : {U32.is_le(_,2147483648) == True{} : Bool}
    Power.native_power_bound(depth,Order.successor(depth,30n,depth_ok)),
    %Equal.sym(U32,capacity,power,capacity_exact) : {Nat.is_le(size,U32.to_nat(_)) == True{} : Bool}
    Depth.allocation_room(count,depth_fits),inside)
