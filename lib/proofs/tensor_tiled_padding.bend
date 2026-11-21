# Padding: the cells of a piece with +0 in every lane past the true lane
# count. The model of the runtime on contents, and what it leaves in each
# cell: the cells of the last group pass through the tail, the others stay.
import Base
import ../tensor_tiled_map.bend as TiledMap
import ../tensor_tiled_cells.bend as Cells
import ../tensor_grid.bend as Grid
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ../traversal_array.bend as Traversal
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./traversal_map_readback.bend as MapReadback
import ./storage_shape.bend as Shape
import ./word_power_of_two.bend as Power
import ./word_encoding.bend as WordEncoding
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./nat_extrema.bend as Extrema
import ./row_major_index.bend as RowMajor
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_interval.bend as Interval
import ./u32_full_range.bend as FullRange
import ./u32_comparison.bend as Comparison
import ./u32_division.bend as U32Division
import ./u32_extrema.bend as NativeExtrema
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./traversal_array_refinement.bend as ArrayLaws
import ../kernel_gemm.bend as KernelGemm

# ---- The model on contents ----

def at_step(+start: U32,+count: U32,+tail: Cells.Tail,content: Buffer.Content<Tile.Tile8>) -> Buffer.Content<Tile.Tile8>:
  Buffer.Content{tree,+length} = content
  Buffer.Content{MapReadback.run(~Tile.Tile8,~Cells.Tail,~Cells.tail,U32.to_nat(count),start,tail,tree),length}

def rows_step(remaining: Nat,+position: U32,+groups: U32,+tail: Cells.Tail,content: Buffer.Content<Tile.Tile8>) -> Buffer.Content<Tile.Tile8>:
  match remaining:
    case 0n: content
    case 1n+rest: rows_step(rest,(position + groups : U32),groups,tail,at_step(position,1,tail,content))

def inner_step(single: Bool,+first: U32,+count: U32,+groups: U32,+tail: Cells.Tail,content: Buffer.Content<Tile.Tile8>) -> Buffer.Content<Tile.Tile8>:
  match single:
    case True{}: at_step(0,count,tail,content)
    case False{}:
      +start = ((groups - 1) - first % groups : U32)
      rows_step(U32.to_nat(Bool.pick(U32,U32.is_lt(start,count),((count - start - 1) / groups + 1 : U32),0)),start,groups,tail,content)

def range_step(meets: Bool,+start: U32,+count: U32,+tail: Cells.Tail,content: Buffer.Content<Tile.Tile8>) -> Buffer.Content<Tile.Tile8>:
  match meets:
    case True{}: at_step(start,count,tail,content)
    case False{}: content

def placed_step(position: Grid.Position,+first: U32,+count: U32,+groups: U32,+outer: U32,+tail: Cells.Tail,content: Buffer.Content<Tile.Tile8>) -> Buffer.Content<Tile.Tile8>:
  match position:
    case Grid.Innermost{}: inner_step(U32.is_le(groups,1),first,count,groups,tail,content)
    case Grid.Outermost{}:
      +low = U32.max(((groups - 1) * outer : U32),first)
      +high = U32.min((groups * outer : U32),(first + count : U32))
      range_step(U32.is_lt(low,high),(low - first : U32),(high - low : U32),tail,content)

def when_step(aligned: Bool,position: Grid.Position,+first: U32,+count: U32,+lanes: U32,+groups: U32,+outer: U32,content: Buffer.Content<Tile.Tile8>) -> Buffer.Content<Tile.Tile8>:
  match aligned:
    case True{}: content
    case False{}: placed_step(position,first,count,groups,outer,Cells.Tail{((groups - 1) * 8 : U32),lanes},content)

# What padding makes of the contents of a piece of first cell first and count
# cells.
def step(+padding: TiledMap.Padding,+first: U32,+count: U32,content: Buffer.Content<Tile.Tile8>) -> Buffer.Content<Tile.Tile8>:
  TiledMap.Padding{+lanes,+groups,+outer,position} = padding
  when_step(U32.is_zero((lanes % 8 : U32)),position,first,count,lanes,groups,outer,content)

# ---- The rule of the runtime ----

def at_model(+start: U32,+count: U32,+tail: Cells.Tail,+input: Buffer.Model<Tile.Tile8>) ->
  {Buffer.contents(~Tile.Tile8,TiledMap.padded_at(start,count,tail,Buffer.pack(~Tile.Tile8,input))) ==
    at_step(start,count,tail,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,input))) : Buffer.Content<Tile.Tile8>}:
  Buffer.Model{+tree,+length,+certificate} = input
  +run = MapReadback.run(~Tile.Tile8,~Cells.Tail,~Cells.tail,U32.to_nat(count),start,tail,tree)
  %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,tree)),tree,StorageLaws.reflect_packed(Tile.Tile8,tree)) :
    {Buffer.contents(~Tile.Tile8,TiledMap.padded_at(start,count,tail,Storage.Buffer{Tree.pack(Tile.Tile8,tree),length,certificate})) ==
      at_step(start,count,tail,Buffer.Content{_,length}) : Buffer.Content<Tile.Tile8>}
  Equal.cong(Tree.Tree<Tile.Tile8>,Buffer.Content<Tile.Tile8>,tree => Buffer.Content{tree,length},
    Tree.reflect(Tile.Tile8,Traversal.map_in_place(~Tile.Tile8,~Cells.Tail,~Cells.tail,U32.to_nat(count),start,tail,Tree.pack(Tile.Tile8,tree))),run,
    Equal.trans(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Traversal.map_in_place(~Tile.Tile8,~Cells.Tail,~Cells.tail,U32.to_nat(count),start,tail,Tree.pack(Tile.Tile8,tree))),
      Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,run)),run,
      Equal.cong(Array<Tile.Tile8>,Tree.Tree<Tile.Tile8>,storage => Tree.reflect(Tile.Tile8,storage),
        Traversal.map_in_place(~Tile.Tile8,~Cells.Tail,~Cells.tail,U32.to_nat(count),start,tail,Tree.pack(Tile.Tile8,tree)),Tree.pack(Tile.Tile8,run),
        MapReadback.packed(~Tile.Tile8,~Cells.Tail,~Cells.tail,U32.to_nat(count),start,tail,tree)),
      StorageLaws.reflect_packed(Tile.Tile8,run)))

def at_observed(+start: U32,+count: U32,+tail: Cells.Tail,-buffer: Storage.Buffer<Tile.Tile8>,observation: Buffer.Observation<Tile.Tile8,buffer>) ->
  {Buffer.contents(~Tile.Tile8,TiledMap.padded_at(start,count,tail,buffer)) == at_step(start,count,tail,Buffer.contents(~Tile.Tile8,buffer)) : Buffer.Content<Tile.Tile8>}:
  Buffer.Observation{+model,+equation} = observation
  %equation : {Buffer.contents(~Tile.Tile8,TiledMap.padded_at(start,count,tail,_)) == at_step(start,count,tail,Buffer.contents(~Tile.Tile8,buffer)) : Buffer.Content<Tile.Tile8>}
  %equation : {Buffer.contents(~Tile.Tile8,TiledMap.padded_at(start,count,tail,Buffer.pack(~Tile.Tile8,model))) == at_step(start,count,tail,Buffer.contents(~Tile.Tile8,_)) : Buffer.Content<Tile.Tile8>}
  at_model(start,count,tail,model)

def rows_observed(remaining: Nat,+position: U32,+groups: U32,+tail: Cells.Tail,-buffer: Storage.Buffer<Tile.Tile8>,observation: Buffer.Observation<Tile.Tile8,buffer>) ->
  {Buffer.contents(~Tile.Tile8,TiledMap.padded_rows(remaining,position,groups,tail,buffer)) == rows_step(remaining,position,groups,tail,Buffer.contents(~Tile.Tile8,buffer)) : Buffer.Content<Tile.Tile8>}:
  match remaining:
    case 0n: {==}
    case 1n+rest:
      Buffer.Observation{+model,+equation} = observation
      %equation : {Buffer.contents(~Tile.Tile8,TiledMap.padded_rows(1n+rest,position,groups,tail,_)) == rows_step(1n+rest,position,groups,tail,Buffer.contents(~Tile.Tile8,buffer)) : Buffer.Content<Tile.Tile8>}
      %equation : {Buffer.contents(~Tile.Tile8,TiledMap.padded_rows(1n+rest,position,groups,tail,Buffer.pack(~Tile.Tile8,model))) == rows_step(1n+rest,position,groups,tail,Buffer.contents(~Tile.Tile8,_)) : Buffer.Content<Tile.Tile8>}
      Equal.trans(Buffer.Content<Tile.Tile8>,
        Buffer.contents(~Tile.Tile8,TiledMap.padded_rows(rest,(position + groups : U32),groups,tail,TiledMap.padded_at(position,1,tail,Buffer.pack(~Tile.Tile8,model)))),
        rows_step(rest,(position + groups : U32),groups,tail,Buffer.contents(~Tile.Tile8,TiledMap.padded_at(position,1,tail,Buffer.pack(~Tile.Tile8,model)))),
        rows_step(rest,(position + groups : U32),groups,tail,at_step(position,1,tail,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,model)))),
        rows_observed(rest,(position + groups : U32),groups,tail,TiledMap.padded_at(position,1,tail,Buffer.pack(~Tile.Tile8,model)),Buffer.observe(~Tile.Tile8,TiledMap.padded_at(position,1,tail,Buffer.pack(~Tile.Tile8,model)))),
        Equal.cong(Buffer.Content<Tile.Tile8>,Buffer.Content<Tile.Tile8>,content => rows_step(rest,(position + groups : U32),groups,tail,content),
          Buffer.contents(~Tile.Tile8,TiledMap.padded_at(position,1,tail,Buffer.pack(~Tile.Tile8,model))),at_step(position,1,tail,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,model))),at_model(position,1,tail,model)))

def inner_observed(single: Bool,+first: U32,+count: U32,+groups: U32,+tail: Cells.Tail,-buffer: Storage.Buffer<Tile.Tile8>,observation: Buffer.Observation<Tile.Tile8,buffer>) ->
  {Buffer.contents(~Tile.Tile8,TiledMap.padded_inner(single,first,count,groups,tail,buffer)) == inner_step(single,first,count,groups,tail,Buffer.contents(~Tile.Tile8,buffer)) : Buffer.Content<Tile.Tile8>}:
  match single:
    case True{}: at_observed(0,count,tail,buffer,observation)
    case False{}:
      +start = ((groups - 1) - first % groups : U32)
      rows_observed(U32.to_nat(Bool.pick(U32,U32.is_lt(start,count),((count - start - 1) / groups + 1 : U32),0)),start,groups,tail,buffer,observation)

def range_observed(meets: Bool,+start: U32,+count: U32,+tail: Cells.Tail,-buffer: Storage.Buffer<Tile.Tile8>,observation: Buffer.Observation<Tile.Tile8,buffer>) ->
  {Buffer.contents(~Tile.Tile8,TiledMap.padded_range(meets,start,count,tail,buffer)) == range_step(meets,start,count,tail,Buffer.contents(~Tile.Tile8,buffer)) : Buffer.Content<Tile.Tile8>}:
  match meets:
    case True{}: at_observed(start,count,tail,buffer,observation)
    case False{}: {==}

def placed_observed(position: Grid.Position,+first: U32,+count: U32,+groups: U32,+outer: U32,+tail: Cells.Tail,-buffer: Storage.Buffer<Tile.Tile8>,
  observation: Buffer.Observation<Tile.Tile8,buffer>) ->
  {Buffer.contents(~Tile.Tile8,TiledMap.padded_placed(position,first,count,groups,outer,tail,buffer)) ==
    placed_step(position,first,count,groups,outer,tail,Buffer.contents(~Tile.Tile8,buffer)) : Buffer.Content<Tile.Tile8>}:
  match position:
    case Grid.Innermost{}: inner_observed(U32.is_le(groups,1),first,count,groups,tail,buffer,observation)
    case Grid.Outermost{}:
      +low = U32.max(((groups - 1) * outer : U32),first)
      +high = U32.min((groups * outer : U32),(first + count : U32))
      range_observed(U32.is_lt(low,high),(low - first : U32),(high - low : U32),tail,buffer,observation)

def when_observed(aligned: Bool,position: Grid.Position,+first: U32,+count: U32,+lanes: U32,+groups: U32,+outer: U32,-buffer: Storage.Buffer<Tile.Tile8>,
  observation: Buffer.Observation<Tile.Tile8,buffer>) ->
  {Buffer.contents(~Tile.Tile8,TiledMap.padded_when(aligned,position,first,count,lanes,groups,outer,buffer)) ==
    when_step(aligned,position,first,count,lanes,groups,outer,Buffer.contents(~Tile.Tile8,buffer)) : Buffer.Content<Tile.Tile8>}:
  match aligned:
    case True{}: {==}
    case False{}: placed_observed(position,first,count,groups,outer,Cells.Tail{((groups - 1) * 8 : U32),lanes},buffer,observation)

def observed(+padding: TiledMap.Padding,+first: U32,+count: U32,-buffer: Storage.Buffer<Tile.Tile8>,observation: Buffer.Observation<Tile.Tile8,buffer>) ->
  {Buffer.contents(~Tile.Tile8,TiledMap.padded(padding,first,count,buffer)) == step(padding,first,count,Buffer.contents(~Tile.Tile8,buffer)) : Buffer.Content<Tile.Tile8>}:
  TiledMap.Padding{+lanes,+groups,+outer,position} = padding
  when_observed(U32.is_zero((lanes % 8 : U32)),position,first,count,lanes,groups,outer,buffer,observation)

# Padding is a leaf of every piece: its contents are the step of the contents
# it was given.
def rule(+padding: TiledMap.Padding,+first: U32,+count: U32,buffer: Storage.Buffer<Tile.Tile8>) ->
  {Buffer.contents(~Tile.Tile8,TiledMap.padded(padding,first,count,buffer)) == step(padding,first,count,Buffer.contents(~Tile.Tile8,buffer)) : Buffer.Content<Tile.Tile8>}:
  observed(padding,first,count,buffer,Buffer.observe(~Tile.Tile8,buffer))

# ---- What padding leaves in a cell ----

# Whether a cell lies in the last group of its lane axis.
def last(position: Grid.Position,+groups: Nat,+outer: Nat,+cell: Nat) -> Bool:
  match position:
    case Grid.Innermost{}: Nat.is_eq(Nat.mod(cell,groups),Nat.sub(groups,1n))
    case Grid.Outermost{}: Nat.is_le(Nat.mul(Nat.sub(groups,1n),outer),cell) && Nat.is_lt(cell,Nat.mul(groups,outer))

def kept(+chosen: Bool,+tail: Cells.Tail,+tile: Tile.Tile8) -> Tile.Tile8:
  Bool.pick(Tile.Tile8,chosen,Cells.tail(tail,tile),tile)

# What padding leaves in the cell of a tensor at a cell index.
def cell(+padding: TiledMap.Padding,+index: Nat,+tile: Tile.Tile8) -> Tile.Tile8:
  TiledMap.Padding{+lanes,+groups,+outer,position} = padding
  Bool.pick(Tile.Tile8,U32.is_zero((lanes % 8 : U32)),tile,
    kept(last(position,U32.to_nat(groups),U32.to_nat(outer),index),Cells.Tail{((groups - 1) * 8 : U32),lanes},tile))

# ---- Reading a tailed range ----

def above(+lower: Nat,+value: Nat,order: {Nat.is_le(lower,value) == False{} : Bool}) -> {Nat.is_lt(value,lower) == True{} : Bool}:
  match lower value:
    case 0n 0n: Empty.absurd({Nat.is_lt(0n,0n) == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},order)))
    case 0n 1n+value: Empty.absurd({Nat.is_lt(1n+value,0n) == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},order)))
    case 1n+lower 0n: {==}
    case 1n+lower 1n+value: above(lower,value,order)

def below_not_le(+value: Nat,+lower: Nat,order: {Nat.is_lt(value,lower) == True{} : Bool}) -> {Nat.is_le(lower,value) == False{} : Bool}:
  match value lower:
    case 0n 0n: Empty.absurd({Nat.is_le(0n,0n) == False{} : Bool},WordEncoding.false_true(order))
    case 1n+value 0n: Empty.absurd({Nat.is_le(0n,1n+value) == False{} : Bool},WordEncoding.false_true(order))
    case 0n 1n+lower: {==}
    case 1n+value 1n+lower: below_not_le(value,lower,order)

def capacity_ok(+tree: Tree.Tree<Tile.Tile8>,+depth: Nat,+balanced: Shape.Balanced(Tile.Tile8,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool}) ->
  {U32.is_le(Tree.capacity(Tile.Tile8,tree),2147483648) == True{} : Bool}:
  %Equal.sym(U32,Tree.capacity(Tile.Tile8,tree),U32.shln(1,depth),Shape.capacity(Tile.Tile8,depth,tree,balanced)) : {U32.is_le(_,2147483648) == True{} : Bool}
  Power.native_power_bound(depth,Order.successor(depth,30n,depth_ok))

def query_inside(+index: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+inside: {Nat.is_lt(index,U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.is_lt(U32.from_nat(index),capacity) == True{} : Bool}:
  +exact = FullRange.roundtrip(index,capacity,Bounds.strict_le(index,U32.to_nat(capacity),inside))
  %Equal.sym(Bool,U32.is_lt(U32.from_nat(index),capacity),Nat.is_lt(U32.to_nat(U32.from_nat(index)),U32.to_nat(capacity)),Comparison.lt(U32.from_nat(index),capacity)) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,exact) : {Nat.is_lt(_,U32.to_nat(capacity)) == True{} : Bool}
  inside

def outside(+start: Nat,+count: Nat,+tail: Cells.Tail,+tree: Tree.Tree<Tile.Tile8>,+depth: Nat,+balanced: Shape.Balanced(Tile.Tile8,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(start,count),U32.to_nat(Tree.capacity(Tile.Tile8,tree))) == True{} : Bool},+index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(Tree.capacity(Tile.Tile8,tree))) == True{} : Bool},
  separate: Interval.Outside<start,count,index>) ->
  {Tree.read(Tile.Tile8,MapReadback.run(~Tile.Tile8,~Cells.Tail,~Cells.tail,count,U32.from_nat(start),tail,tree),U32.from_nat(index)) == Tree.read(Tile.Tile8,tree,U32.from_nat(index)) : Tile.Tile8}:
  +capacity = Tree.capacity(Tile.Tile8,tree)
  +bounded = capacity_ok(tree,depth,balanced,depth_ok)
  +exact = FullRange.roundtrip(index,capacity,Bounds.strict_le(index,U32.to_nat(capacity),inside))
  MapReadback.outside_indexed(~Tile.Tile8,~Cells.Tail,~ArrayLaws.unindexed(~Tile.Tile8,~Cells.Tail,~Cells.tail),count,start,U32.from_nat(index),tail,tree,depth,capacity,{==},balanced,
    Order.successor(depth,30n,depth_ok),bounded,room,query_inside(index,capacity,bounded,inside),
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,exact) : Interval.Outside<start,count,_> separate)

def range_inside(+start: Nat,+count: Nat,+tail: Cells.Tail,+tree: Tree.Tree<Tile.Tile8>,+depth: Nat,+balanced: Shape.Balanced(Tile.Tile8,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(start,count),U32.to_nat(Tree.capacity(Tile.Tile8,tree))) == True{} : Bool},+index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(Tree.capacity(Tile.Tile8,tree))) == True{} : Bool},
  +lower: {Nat.is_le(start,index) == True{} : Bool},within: Bool,+classified: {Nat.is_lt(index,Nat.add(start,count)) == within : Bool}) ->
  {Tree.read(Tile.Tile8,MapReadback.run(~Tile.Tile8,~Cells.Tail,~Cells.tail,count,U32.from_nat(start),tail,tree),U32.from_nat(index)) ==
    kept(within,tail,Tree.read(Tile.Tile8,tree,U32.from_nat(index))) : Tile.Tile8}:
  match within:
    case True{}:
      +offset = Nat.sub(index,start)
      +restored = Order.restore_shift(start,index,lower)
      %restored :
        {Tree.read(Tile.Tile8,MapReadback.run(~Tile.Tile8,~Cells.Tail,~Cells.tail,count,U32.from_nat(start),tail,tree),U32.from_nat(_)) ==
          kept(True{},tail,Tree.read(Tile.Tile8,tree,U32.from_nat(_))) : Tile.Tile8}
      MapReadback.inside(~Tile.Tile8,~Cells.Tail,~Cells.tail,count,start,offset,tail,tree,depth,Tree.capacity(Tile.Tile8,tree),{==},balanced,depth_ok,
        capacity_ok(tree,depth,balanced,depth_ok),room,
        Order.cancel_offset_lt(start,offset,count,%Equal.sym(Nat,Nat.add(start,offset),index,restored) : {Nat.is_lt(_,Nat.add(start,count)) == True{} : Bool} classified))
    case False{}:
      outside(start,count,tail,tree,depth,balanced,depth_ok,room,index,inside,Interval.After{Order.not_less(index,Nat.add(start,count),classified)})

def range_cases(+start: Nat,+count: Nat,+tail: Cells.Tail,+tree: Tree.Tree<Tile.Tile8>,+depth: Nat,+balanced: Shape.Balanced(Tile.Tile8,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(start,count),U32.to_nat(Tree.capacity(Tile.Tile8,tree))) == True{} : Bool},+index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(Tree.capacity(Tile.Tile8,tree))) == True{} : Bool},
  below: Bool,+classified: {Nat.is_le(start,index) == below : Bool}) ->
  {Tree.read(Tile.Tile8,MapReadback.run(~Tile.Tile8,~Cells.Tail,~Cells.tail,count,U32.from_nat(start),tail,tree),U32.from_nat(index)) ==
    kept(below && Nat.is_lt(index,Nat.add(start,count)),tail,Tree.read(Tile.Tile8,tree,U32.from_nat(index))) : Tile.Tile8}:
  match below:
    case True{}: range_inside(start,count,tail,tree,depth,balanced,depth_ok,room,index,inside,classified,Nat.is_lt(index,Nat.add(start,count)),{==})
    case False{}: outside(start,count,tail,tree,depth,balanced,depth_ok,room,index,inside,Interval.Before{above(start,index,classified)})

# The cells of a range pass through the tail; the others stay.
def range_read(+start: Nat,+count: Nat,+tail: Cells.Tail,+tree: Tree.Tree<Tile.Tile8>,+depth: Nat,+balanced: Shape.Balanced(Tile.Tile8,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(start,count),U32.to_nat(Tree.capacity(Tile.Tile8,tree))) == True{} : Bool},+index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(Tree.capacity(Tile.Tile8,tree))) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,MapReadback.run(~Tile.Tile8,~Cells.Tail,~Cells.tail,count,U32.from_nat(start),tail,tree),U32.from_nat(index)) ==
    kept(Nat.is_le(start,index) && Nat.is_lt(index,Nat.add(start,count)),tail,Tree.read(Tile.Tile8,tree,U32.from_nat(index))) : Tile.Tile8}:
  range_cases(start,count,tail,tree,depth,balanced,depth_ok,room,index,inside,Nat.is_le(start,index),{==})

def at_ready(+start: U32,+count: U32,+tail: Cells.Tail,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,ready: Buffer.ContentReady<Tile.Tile8,Buffer.Content{tree,length}>) ->
  Buffer.ContentReady<Tile.Tile8,at_step(start,count,tail,Buffer.Content{tree,length})>:
  Buffer.ContentReady{+depth,+balanced,+depth_ok,+room} = ready
  +run = MapReadback.run(~Tile.Tile8,~Cells.Tail,~Cells.tail,U32.to_nat(count),start,tail,tree)
  Buffer.ContentReady{depth,MapReadback.balanced(~Tile.Tile8,~Cells.Tail,~Cells.tail,U32.to_nat(count),start,tail,tree,depth,balanced),depth_ok,
    %Equal.sym(U32,Tree.capacity(Tile.Tile8,run),Tree.capacity(Tile.Tile8,tree),MapReadback.capacity(~Tile.Tile8,~Cells.Tail,~Cells.tail,U32.to_nat(count),start,tail,tree)) :
      {Nat.is_le(length,U32.to_nat(_)) == True{} : Bool} room}

# A range of a piece through the tail, at a word start.
def at_read(+start: U32,+count: U32,+tail: Cells.Tail,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,+ready: Buffer.ContentReady<Tile.Tile8,Buffer.Content{tree,length}>,
  +room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(count)),length) == True{} : Bool},+index: Nat,+inside: {Nat.is_lt(index,length) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,at_step(start,count,tail,Buffer.Content{tree,length})),U32.from_nat(index)) ==
    kept(Nat.is_le(U32.to_nat(start),index) && Nat.is_lt(index,Nat.add(U32.to_nat(start),U32.to_nat(count))),tail,Tree.read(Tile.Tile8,tree,U32.from_nat(index))) : Tile.Tile8}:
  Buffer.ContentReady{+depth,+balanced,+depth_ok,+capacity_room} = ready
  %Arithmetic.native_roundtrip(start) :
    {Tree.read(Tile.Tile8,MapReadback.run(~Tile.Tile8,~Cells.Tail,~Cells.tail,U32.to_nat(count),_,tail,tree),U32.from_nat(index)) ==
      kept(Nat.is_le(U32.to_nat(start),index) && Nat.is_lt(index,Nat.add(U32.to_nat(start),U32.to_nat(count))),tail,Tree.read(Tile.Tile8,tree,U32.from_nat(index))) : Tile.Tile8}
  range_read(U32.to_nat(start),U32.to_nat(count),tail,tree,depth,balanced,depth_ok,
    Order.transitive(Nat.add(U32.to_nat(start),U32.to_nat(count)),length,U32.to_nat(Tree.capacity(Tile.Tile8,tree)),room,capacity_room),index,
    Bounds.nat_trans(index,length,U32.to_nat(Tree.capacity(Tile.Tile8,tree)),inside,capacity_room))

def succ_room(+position: Nat,+length: Nat,+below: {Nat.is_lt(position,length) == True{} : Bool}) -> {Nat.is_le(Nat.add(position,1n),length) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(position,1n),1n+position,Addition.successor_width(position)) : {Nat.is_le(_,length) == True{} : Bool}
  Order.strict_follow(position,length,below)

# ---- Reading a walk of one cell a run ----

# Whether a walk of remaining steps of one cell, groups cells apart from the
# position on, passes the cell at an index.
def hit(remaining: Nat,+position: Nat,+groups: Nat,+index: Nat) -> Bool:
  match remaining:
    case 0n: False{}
    case 1n+rest: (Nat.is_le(position,index) && Nat.is_lt(index,Nat.add(position,1n))) || hit(rest,Nat.add(position,groups),groups,index)

# Every cell of the walk lies below the length.
def Reach(remaining: Nat,+position: Nat,+groups: Nat,+length: Nat) -> Data:
  match remaining:
    case 0n: Unit
    case 1n+rest: KernelGemm.Both<{Nat.is_lt(position,length) == True{} : Bool},Reach(rest,Nat.add(position,groups),groups,length)>

def hit_below(remaining: Nat,+position: Nat,+groups: Nat,+index: Nat,+below: {Nat.is_lt(index,position) == True{} : Bool}) ->
  {hit(remaining,position,groups,index) == False{} : Bool}:
  match remaining:
    case 0n: {==}
    case 1n+rest:
      %Equal.sym(Bool,Nat.is_le(position,index),False{},below_not_le(index,position,below)) :
        {(_ && Nat.is_lt(index,Nat.add(position,1n))) || hit(rest,Nat.add(position,groups),groups,index) == False{} : Bool}
      hit_below(rest,Nat.add(position,groups),groups,index,Bounds.nat_trans(index,position,Nat.add(position,groups),below,Order.le_add(position,groups)))

def apart_high(+position: Nat,+groups: Nat,rest: Nat,+index: Nat,+positive: {Nat.is_lt(0n,groups) == True{} : Bool},
  high: Bool,+classified: {Nat.is_lt(index,Nat.add(position,1n)) == high : Bool}) ->
  {(True{} && high) && hit(rest,Nat.add(position,groups),groups,index) == False{} : Bool}:
  match high:
    case True{}:
      hit_below(rest,Nat.add(position,groups),groups,index,
        Bounds.nat_trans(index,Nat.add(position,1n),Nat.add(position,groups),classified,Order.offset_le(position,1n,groups,Order.positive_one(groups,positive))))
    case False{}: {==}

def apart_low(+position: Nat,+groups: Nat,rest: Nat,+index: Nat,+positive: {Nat.is_lt(0n,groups) == True{} : Bool},
  low: Bool,+classified: {Nat.is_le(position,index) == low : Bool}) ->
  {(low && Nat.is_lt(index,Nat.add(position,1n))) && hit(rest,Nat.add(position,groups),groups,index) == False{} : Bool}:
  match low:
    case True{}: apart_high(position,groups,rest,index,positive,Nat.is_lt(index,Nat.add(position,1n)),{==})
    case False{}: {==}

# The cell of one step is not passed again.
def apart(+position: Nat,+groups: Nat,rest: Nat,+index: Nat,+positive: {Nat.is_lt(0n,groups) == True{} : Bool}) ->
  {(Nat.is_le(position,index) && Nat.is_lt(index,Nat.add(position,1n))) && hit(rest,Nat.add(position,groups),groups,index) == False{} : Bool}:
  apart_low(position,groups,rest,index,positive,Nat.is_le(position,index),{==})

def merged(chosen: Bool,+later: Bool,+tail: Cells.Tail,+tile: Tile.Tile8,+separate: {chosen && later == False{} : Bool}) ->
  {kept(later,tail,kept(chosen,tail,tile)) == kept(chosen || later,tail,tile) : Tile.Tile8}:
  match chosen:
    case True{}:
      %Equal.sym(Bool,later,False{},separate) : {kept(_,tail,kept(True{},tail,tile)) == kept(True{} || _,tail,tile) : Tile.Tile8}
      {==}
    case False{}: {==}

def or_false(chosen: Bool) -> {chosen || False{} == chosen : Bool}:
  match chosen:
    case True{}: {==}
    case False{}: {==}

# One step of a walk at a word position, then the rest of the walk.
def rows_next(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},rest: Nat,+position: U32,+groups: U32,+tail: Cells.Tail,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,
  +ready: Buffer.ContentReady<Tile.Tile8,Buffer.Content{tree,length}>,+bounded: {Nat.is_le(length,U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(groups)) == True{} : Bool},
  +reach: Reach(1n+rest,U32.to_nat(position),U32.to_nat(groups),length),+index: Nat,+inside: {Nat.is_lt(index,length) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,rows_step(1n+rest,position,groups,tail,Buffer.Content{tree,length})),U32.from_nat(index)) ==
    kept(hit(1n+rest,U32.to_nat(position),U32.to_nat(groups),index),tail,Tree.read(Tile.Tile8,tree,U32.from_nat(index))) : Tile.Tile8}:
  match rest:
    case 0n:
      KernelGemm.Both{+here,+later} = reach
      +p = U32.to_nat(position)
      +head = Nat.is_le(p,index) && Nat.is_lt(index,Nat.add(p,1n))
      +original = Tree.read(Tile.Tile8,tree,U32.from_nat(index))
      Equal.trans(Tile.Tile8,
        Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,at_step(position,1,tail,Buffer.Content{tree,length})),U32.from_nat(index)),
        kept(head,tail,original),kept(head || False{},tail,original),
        at_read(position,1,tail,tree,length,ready,succ_room(p,length,here),index,inside),
        Equal.cong(Bool,Tile.Tile8,chosen => kept(chosen,tail,original),head,head || False{},Equal.sym(Bool,head || False{},head,or_false(head))))
    case 1n+ +more:
      KernelGemm.Both{+here,+later} = reach
      KernelGemm.Both{+next_here,+next_later} = later
      +p = U32.to_nat(position)
      +g = U32.to_nat(groups)
      +next = (position + groups : U32)
      +exact = Arithmetic.add(position,groups,cap,cap_ok,
        Order.transitive(Nat.add(p,g),length,U32.to_nat(cap),Bounds.strict_le(Nat.add(p,g),length,next_here),bounded))
      +stepped = MapReadback.run(~Tile.Tile8,~Cells.Tail,~Cells.tail,U32.to_nat(1),position,tail,tree)
      +head = Nat.is_le(p,index) && Nat.is_lt(index,Nat.add(p,1n))
      +rest_hit = hit(1n+more,Nat.add(p,g),g,index)
      +original = Tree.read(Tile.Tile8,tree,U32.from_nat(index))
      Equal.trans(Tile.Tile8,
        Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,rows_step(1n+more,next,groups,tail,Buffer.Content{stepped,length})),U32.from_nat(index)),
        kept(rest_hit,tail,Tree.read(Tile.Tile8,stepped,U32.from_nat(index))),
        kept(head || rest_hit,tail,original),
        %exact : {Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,rows_step(1n+more,next,groups,tail,Buffer.Content{stepped,length})),U32.from_nat(index)) ==
            kept(hit(1n+more,_,g,index),tail,Tree.read(Tile.Tile8,stepped,U32.from_nat(index))) : Tile.Tile8}
          rows_next(cap,cap_ok,more,next,groups,tail,stepped,length,at_ready(position,1,tail,tree,length,ready),bounded,positive,
            %Equal.sym(Nat,U32.to_nat(next),Nat.add(p,g),exact) : Reach(1n+more,_,g,length) later,index,inside),
        Equal.trans(Tile.Tile8,kept(rest_hit,tail,Tree.read(Tile.Tile8,stepped,U32.from_nat(index))),
          kept(rest_hit,tail,kept(head,tail,original)),
          kept(head || rest_hit,tail,original),
          Equal.cong(Tile.Tile8,Tile.Tile8,tile => kept(rest_hit,tail,tile),Tree.read(Tile.Tile8,stepped,U32.from_nat(index)),
            kept(head,tail,original),
            at_read(position,1,tail,tree,length,ready,succ_room(p,length,here),index,inside)),
          merged(head,rest_hit,tail,original,apart(p,g,1n+more,index,positive))))

# The cells passed by the walk pass through the tail; the others stay.
def rows_read(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},remaining: Nat,+position: U32,+groups: U32,+tail: Cells.Tail,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,
  +ready: Buffer.ContentReady<Tile.Tile8,Buffer.Content{tree,length}>,+bounded: {Nat.is_le(length,U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(groups)) == True{} : Bool},
  +reach: Reach(remaining,U32.to_nat(position),U32.to_nat(groups),length),+index: Nat,+inside: {Nat.is_lt(index,length) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,rows_step(remaining,position,groups,tail,Buffer.Content{tree,length})),U32.from_nat(index)) ==
    kept(hit(remaining,U32.to_nat(position),U32.to_nat(groups),index),tail,Tree.read(Tile.Tile8,tree,U32.from_nat(index))) : Tile.Tile8}:
  match remaining:
    case 0n: {==}
    case 1n+rest: rows_next(cap,cap_ok,rest,position,groups,tail,tree,length,ready,bounded,positive,reach,index,inside)

# ---- The walk of the innermost groups ----

def predecessor(+groups: Nat,+positive: {Nat.is_lt(0n,groups) == True{} : Bool}) -> {groups == 1n+Nat.sub(groups,1n) : Nat}:
  match groups:
    case 0n: Empty.absurd({0n == 1n+Nat.sub(0n,1n) : Nat},WordEncoding.false_true(positive))
    case 1n+rest:
      match rest:
        case 0n: {==}
        case 1n+more: {==}

def last_below(+groups: Nat,+positive: {Nat.is_lt(0n,groups) == True{} : Bool}) -> {Nat.is_lt(Nat.sub(groups,1n),groups) == True{} : Bool}:
  %Equal.sym(Nat,groups,1n+Nat.sub(groups,1n),predecessor(groups,positive)) : {Nat.is_lt(Nat.sub(groups,1n),_) == True{} : Bool}
  Bounds.predecessor_strict(Nat.sub(groups,1n),1n+Nat.sub(groups,1n),Order.reflexive(1n+Nat.sub(groups,1n)))

def below_last(+value: Nat,+groups: Nat,+positive: {Nat.is_lt(0n,groups) == True{} : Bool},+less: {Nat.is_lt(value,groups) == True{} : Bool}) ->
  {Nat.is_le(value,Nat.sub(groups,1n)) == True{} : Bool}:
  Division.lt_succ_le(value,Nat.sub(groups,1n),%predecessor(groups,positive) : {Nat.is_lt(value,_) == True{} : Bool} less)

def strict_last(+value: Nat,+groups: Nat,+positive: {Nat.is_lt(0n,groups) == True{} : Bool},+less: {Nat.is_lt(1n+value,groups) == True{} : Bool}) ->
  {Nat.is_lt(value,Nat.sub(groups,1n)) == True{} : Bool}:
  %predecessor(groups,positive) : {Nat.is_lt(1n+value,_) == True{} : Bool} less

def remainder_at(+value: Nat,+groups: Nat,+quotient: Nat,+remainder: Nat,+equation: {value == Nat.add(Nat.mul(quotient,groups),remainder) : Nat},
  +positive: {Nat.is_lt(0n,groups) == True{} : Bool},+small: {Nat.is_lt(remainder,groups) == True{} : Bool}) -> {Nat.mod(value,groups) == remainder : Nat}:
  Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(value,groups),(quotient,remainder),
    Division.unique(value,groups,quotient,remainder,equation,positive,small))

def regroup(+groups: Nat,+product: Nat,+start: Nat) -> {Nat.add(Nat.add(groups,product),start) == Nat.add(product,Nat.add(start,groups)) : Nat}:
  Equal.trans(Nat,Nat.add(Nat.add(groups,product),start),Nat.add(Nat.add(product,groups),start),Nat.add(product,Nat.add(start,groups)),
    Equal.cong(Nat,Nat,n => Nat.add(n,start),Nat.add(groups,product),Nat.add(product,groups),Order.commutative(groups,product)),
    Equal.trans(Nat,Nat.add(Nat.add(product,groups),start),Nat.add(product,Nat.add(groups,start)),Nat.add(product,Nat.add(start,groups)),
      Addition.associative(product,groups,start),
      Equal.cong(Nat,Nat,n => Nat.add(product,n),Nat.add(groups,start),Nat.add(start,groups),Order.commutative(groups,start))))

# A cell before the end of its run is not the last of its group.
def last_before(+first: Nat,+groups: Nat,+positive: {Nat.is_lt(0n,groups) == True{} : Bool},+index: Nat,
  +below: {Nat.is_lt(Nat.add(Nat.mod(first,groups),index),Nat.sub(groups,1n)) == True{} : Bool}) ->
  {Nat.is_eq(Nat.mod(Nat.add(first,index),groups),Nat.sub(groups,1n)) == False{} : Bool}:
  +quotient = Nat.div(first,groups)
  +product = Nat.mul(quotient,groups)
  +remainder = Nat.mod(first,groups)
  +within = Nat.add(remainder,index)
  +equation = Equal.trans(Nat,Nat.add(first,index),Nat.add(Nat.add(product,remainder),index),Nat.add(product,within),
    Equal.cong(Nat,Nat,n => Nat.add(n,index),first,Nat.add(product,remainder),DivisionOrder.quotient_exact(first,groups,positive)),
    Addition.associative(product,remainder,index))
  %Equal.sym(Nat,Nat.mod(Nat.add(first,index),groups),within,remainder_at(Nat.add(first,index),groups,quotient,within,equation,positive,
      Bounds.nat_trans(within,Nat.sub(groups,1n),groups,below,Bounds.strict_le(Nat.sub(groups,1n),groups,last_below(groups,positive))))) :
    {Nat.is_eq(_,Nat.sub(groups,1n)) == False{} : Bool}
  Order.less_not_equal(within,Nat.sub(groups,1n),below)

# The first position of a piece that ends a run.
def walk_start(+first: Nat,+groups: Nat) -> Nat:
  Nat.sub(Nat.sub(groups,1n),Nat.mod(first,groups))

# The number of runs that end inside a piece.
def walk_steps(+start: Nat,+groups: Nat,+count: Nat) -> Nat:
  Bool.pick(Nat,Nat.is_lt(start,count),1n+Nat.div(Nat.sub(Nat.sub(count,start),1n),groups),0n)

def aligned_start(+first: Nat,+groups: Nat,+positive: {Nat.is_lt(0n,groups) == True{} : Bool}) ->
  {Nat.mod(Nat.add(first,walk_start(first,groups)),groups) == Nat.sub(groups,1n) : Nat}:
  +quotient = Nat.div(first,groups)
  +product = Nat.mul(quotient,groups)
  +remainder = Nat.mod(first,groups)
  +last = Nat.sub(groups,1n)
  +rest = Nat.sub(last,remainder)
  +fits = below_last(remainder,groups,positive,DivisionOrder.remainder_small(first,groups,positive))
  +equation = Equal.trans(Nat,Nat.add(first,rest),Nat.add(Nat.add(product,remainder),rest),Nat.add(product,last),
    Equal.cong(Nat,Nat,n => Nat.add(n,rest),first,Nat.add(product,remainder),DivisionOrder.quotient_exact(first,groups,positive)),
    Equal.trans(Nat,Nat.add(Nat.add(product,remainder),rest),Nat.add(product,Nat.add(remainder,rest)),Nat.add(product,last),
      Addition.associative(product,remainder,rest),
      Equal.cong(Nat,Nat,n => Nat.add(product,n),Nat.add(remainder,rest),last,Order.restore_shift(remainder,last,fits))))
  remainder_at(Nat.add(first,rest),groups,quotient,last,equation,positive,last_below(groups,positive))

# The cell one past a last cell plus an offset, as whole runs and the offset.
def after_equation(+first: Nat,+start: Nat,+groups: Nat,+positive: {Nat.is_lt(0n,groups) == True{} : Bool},
  +aligned: {Nat.mod(Nat.add(first,start),groups) == Nat.sub(groups,1n) : Nat},+index: Nat,+above: {Nat.is_lt(start,index) == True{} : Bool}) ->
  {Nat.add(first,index) == Nat.add(Nat.mul(1n+Nat.div(Nat.add(first,start),groups),groups),Nat.sub(index,1n+start)) : Nat}:
  +base = Nat.add(first,start)
  +product = Nat.mul(Nat.div(base,groups),groups)
  +last = Nat.sub(groups,1n)
  +offset = Nat.sub(index,1n+start)
  +right = Nat.add(Nat.mul(1n+Nat.div(base,groups),groups),offset)
  %Order.restore_shift(1n+start,index,Order.strict_follow(start,index,above)) : {Nat.add(first,_) == right : Nat}
  %Equal.sym(Nat,Nat.add(first,1n+Nat.add(start,offset)),1n+Nat.add(first,Nat.add(start,offset)),Addition.successor_right(first,Nat.add(start,offset))) :
    {_ == right : Nat}
  %Addition.associative(first,start,offset) : {1n+_ == right : Nat}
  %Equal.sym(Nat,base,Nat.add(product,last),Equal.trans(Nat,base,Nat.add(product,Nat.mod(base,groups)),Nat.add(product,last),
      DivisionOrder.quotient_exact(base,groups,positive),Equal.cong(Nat,Nat,n => Nat.add(product,n),Nat.mod(base,groups),last,aligned))) :
    {1n+Nat.add(_,offset) == right : Nat}
  %Addition.successor_right(product,last) : {Nat.add(_,offset) == right : Nat}
  %predecessor(groups,positive) : {Nat.add(Nat.add(product,_),offset) == right : Nat}
  %Order.commutative(groups,product) : {Nat.add(_,offset) == right : Nat}
  {==}

# A cell after a last cell and before the next is not the last of its group.
def last_after(+first: Nat,+start: Nat,+groups: Nat,+positive: {Nat.is_lt(0n,groups) == True{} : Bool},
  +aligned: {Nat.mod(Nat.add(first,start),groups) == Nat.sub(groups,1n) : Nat},+index: Nat,+above: {Nat.is_lt(start,index) == True{} : Bool},
  +below: {Nat.is_lt(index,Nat.add(start,groups)) == True{} : Bool}) ->
  {Nat.is_eq(Nat.mod(Nat.add(first,index),groups),Nat.sub(groups,1n)) == False{} : Bool}:
  +last = Nat.sub(groups,1n)
  +offset = Nat.sub(index,1n+start)
  +restored = Order.restore_shift(1n+start,index,Order.strict_follow(start,index,above))
  +moved = Equal.trans(Nat,index,Nat.add(1n+start,offset),Nat.add(start,1n+offset),Equal.sym(Nat,Nat.add(1n+start,offset),index,restored),
    Equal.sym(Nat,Nat.add(start,1n+offset),1n+Nat.add(start,offset),Addition.successor_right(start,offset)))
  +small = strict_last(offset,groups,positive,
    Order.cancel_offset_lt(start,1n+offset,groups,%moved : {Nat.is_lt(_,Nat.add(start,groups)) == True{} : Bool} below))
  %Equal.sym(Nat,Nat.mod(Nat.add(first,index),groups),offset,
      remainder_at(Nat.add(first,index),groups,1n+Nat.div(Nat.add(first,start),groups),offset,after_equation(first,start,groups,positive,aligned,index,above),
        positive,Bounds.nat_trans(offset,last,groups,small,Bounds.strict_le(last,groups,last_below(groups,positive))))) :
    {Nat.is_eq(_,last) == False{} : Bool}
  Order.less_not_equal(offset,last,small)

def walk_far(rest: Nat,+start: Nat,+first: Nat,+groups: Nat,+positive: {Nat.is_lt(0n,groups) == True{} : Bool},
  +aligned: {Nat.mod(Nat.add(first,start),groups) == Nat.sub(groups,1n) : Nat},+index: Nat,+above: {Nat.is_lt(start,index) == True{} : Bool},
  later: {Nat.is_le(Nat.add(start,groups),index) == True{} : Bool} ->
    {hit(rest,Nat.add(start,groups),groups,index) == Nat.is_eq(Nat.mod(Nat.add(first,index),groups),Nat.sub(groups,1n)) : Bool},
  near: Bool,+classified: {Nat.is_lt(index,Nat.add(start,groups)) == near : Bool}) ->
  {hit(rest,Nat.add(start,groups),groups,index) == Nat.is_eq(Nat.mod(Nat.add(first,index),groups),Nat.sub(groups,1n)) : Bool}:
  match near:
    case True{}:
      Equal.trans(Bool,hit(rest,Nat.add(start,groups),groups,index),False{},Nat.is_eq(Nat.mod(Nat.add(first,index),groups),Nat.sub(groups,1n)),
        hit_below(rest,Nat.add(start,groups),groups,index,classified),
        Equal.sym(Bool,Nat.is_eq(Nat.mod(Nat.add(first,index),groups),Nat.sub(groups,1n)),False{},
          last_after(first,start,groups,positive,aligned,index,above,classified)))
    case False{}: later(Order.not_less(index,Nat.add(start,groups),classified))

def walk_cases(rest: Nat,+start: Nat,+first: Nat,+groups: Nat,+positive: {Nat.is_lt(0n,groups) == True{} : Bool},
  +aligned: {Nat.mod(Nat.add(first,start),groups) == Nat.sub(groups,1n) : Nat},+index: Nat,+lower: {Nat.is_le(start,index) == True{} : Bool},
  later: {Nat.is_le(Nat.add(start,groups),index) == True{} : Bool} ->
    {hit(rest,Nat.add(start,groups),groups,index) == Nat.is_eq(Nat.mod(Nat.add(first,index),groups),Nat.sub(groups,1n)) : Bool},
  head: Bool,+classified: {Nat.is_lt(index,Nat.add(start,1n)) == head : Bool}) ->
  {hit(1n+rest,start,groups,index) == Nat.is_eq(Nat.mod(Nat.add(first,index),groups),Nat.sub(groups,1n)) : Bool}:
  match head:
    case True{}:
      +last = Nat.sub(groups,1n)
      +same = Order.antisymmetric(index,start,Division.lt_succ_le(index,start,%Addition.successor_width(start) : {Nat.is_lt(index,_) == True{} : Bool} classified),lower)
      %Equal.sym(Bool,Nat.is_le(start,index),True{},lower) :
        {(_ && Nat.is_lt(index,Nat.add(start,1n))) || hit(rest,Nat.add(start,groups),groups,index) == Nat.is_eq(Nat.mod(Nat.add(first,index),groups),last) : Bool}
      %Equal.sym(Bool,Nat.is_lt(index,Nat.add(start,1n)),True{},classified) :
        {(True{} && _) || hit(rest,Nat.add(start,groups),groups,index) == Nat.is_eq(Nat.mod(Nat.add(first,index),groups),last) : Bool}
      %Equal.sym(Nat,index,start,same) : {True{} == Nat.is_eq(Nat.mod(Nat.add(first,_),groups),last) : Bool}
      %Equal.sym(Nat,Nat.mod(Nat.add(first,start),groups),last,aligned) : {True{} == Nat.is_eq(_,last) : Bool}
      Equal.sym(Bool,Nat.is_eq(last,last),True{},Order.equal_reflexive(last))
    case False{}:
      %Equal.sym(Bool,Nat.is_le(start,index),True{},lower) :
        {(_ && Nat.is_lt(index,Nat.add(start,1n))) || hit(rest,Nat.add(start,groups),groups,index) ==
          Nat.is_eq(Nat.mod(Nat.add(first,index),groups),Nat.sub(groups,1n)) : Bool}
      %Equal.sym(Bool,Nat.is_lt(index,Nat.add(start,1n)),False{},classified) :
        {(True{} && _) || hit(rest,Nat.add(start,groups),groups,index) == Nat.is_eq(Nat.mod(Nat.add(first,index),groups),Nat.sub(groups,1n)) : Bool}
      walk_far(rest,start,first,groups,positive,aligned,index,
        Bounds.predecessor_strict(start,index,%Addition.successor_width(start) : {Nat.is_le(_,index) == True{} : Bool}
          Order.not_less(index,Nat.add(start,1n),classified)),
        later,Nat.is_lt(index,Nat.add(start,groups)),{==})

# From a last cell on, a walk that covers the count passes exactly the last
# cells of their groups.
def walk(remaining: Nat,+start: Nat,+first: Nat,+groups: Nat,+count: Nat,+positive: {Nat.is_lt(0n,groups) == True{} : Bool},
  +aligned: {Nat.mod(Nat.add(first,start),groups) == Nat.sub(groups,1n) : Nat},
  +covers: {Nat.is_le(count,Nat.add(Nat.mul(remaining,groups),start)) == True{} : Bool},
  +index: Nat,+lower: {Nat.is_le(start,index) == True{} : Bool},+inside: {Nat.is_lt(index,count) == True{} : Bool}) ->
  {hit(remaining,start,groups,index) == Nat.is_eq(Nat.mod(Nat.add(first,index),groups),Nat.sub(groups,1n)) : Bool}:
  match remaining:
    case 0n:
      Empty.absurd({hit(0n,start,groups,index) == Nat.is_eq(Nat.mod(Nat.add(first,index),groups),Nat.sub(groups,1n)) : Bool},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_le(start,index),True{},
          Equal.sym(Bool,Nat.is_le(start,index),False{},below_not_le(index,start,Bounds.nat_trans(index,count,start,inside,covers))),lower)))
    case 1n+ +rest:
      +next = Nat.add(start,groups)
      +base = Nat.add(first,start)
      +shifted = Equal.trans(Nat,Nat.add(first,next),Nat.add(base,groups),Nat.add(Nat.mul(1n,groups),base),
        Equal.sym(Nat,Nat.add(base,groups),Nat.add(first,next),Addition.associative(first,start,groups)),
        Equal.trans(Nat,Nat.add(base,groups),Nat.add(groups,base),Nat.add(Nat.mul(1n,groups),base),Order.commutative(base,groups),
          Equal.cong(Nat,Nat,n => Nat.add(n,base),groups,Nat.mul(1n,groups),Equal.sym(Nat,Nat.add(groups,0n),groups,Addition.zero_right(groups)))))
      +aligned_next = Equal.trans(Nat,Nat.mod(Nat.add(first,next),groups),Nat.mod(Nat.add(Nat.mul(1n,groups),base),groups),Nat.sub(groups,1n),
        Equal.cong(Nat,Nat,n => Nat.mod(n,groups),Nat.add(first,next),Nat.add(Nat.mul(1n,groups),base),shifted),
        Equal.trans(Nat,Nat.mod(Nat.add(Nat.mul(1n,groups),base),groups),Nat.mod(base,groups),Nat.sub(groups,1n),
          DivisionOrder.remainder_shift(base,groups,1n,positive),aligned))
      +covers_next = {%regroup(groups,Nat.mul(rest,groups),start) : {Nat.is_le(count,_) == True{} : Bool} covers :
        {Nat.is_le(count,Nat.add(Nat.mul(rest,groups),next)) == True{} : Bool}}
      walk_cases(rest,start,first,groups,positive,aligned,index,lower,
        later => walk(rest,next,first,groups,count,positive,aligned_next,covers_next,index,later,inside),
        Nat.is_lt(index,Nat.add(start,1n)),{==})

def inner_cases(+first: Nat,+groups: Nat,+count: Nat,+positive: {Nat.is_lt(0n,groups) == True{} : Bool},+index: Nat,
  +inside: {Nat.is_lt(index,count) == True{} : Bool},lower: Bool,+classified: {Nat.is_le(walk_start(first,groups),index) == lower : Bool}) ->
  {hit(walk_steps(walk_start(first,groups),groups,count),walk_start(first,groups),groups,index) ==
    Nat.is_eq(Nat.mod(Nat.add(first,index),groups),Nat.sub(groups,1n)) : Bool}:
  match lower:
    case False{}:
      +start = walk_start(first,groups)
      +remainder = Nat.mod(first,groups)
      +last = Nat.sub(groups,1n)
      +below = above(start,index,classified)
      +fits = below_last(remainder,groups,positive,DivisionOrder.remainder_small(first,groups,positive))
      Equal.trans(Bool,hit(walk_steps(start,groups,count),start,groups,index),False{},Nat.is_eq(Nat.mod(Nat.add(first,index),groups),last),
        hit_below(walk_steps(start,groups,count),start,groups,index,below),
        Equal.sym(Bool,Nat.is_eq(Nat.mod(Nat.add(first,index),groups),last),False{},
          last_before(first,groups,positive,index,
            Equal.trans(Bool,Nat.is_lt(Nat.add(remainder,index),last),Nat.is_lt(index,start),True{},
              Equal.sym(Bool,Nat.is_lt(index,start),Nat.is_lt(Nat.add(remainder,index),last),Order.below_difference(remainder,index,last,fits)),below))))
    case True{}:
      +start = walk_start(first,groups)
      +difference = Nat.sub(count,start)
      +before = Nat.sub(difference,1n)
      +quotient = Nat.div(before,groups)
      +meets = Order.le_lt_trans(start,index,count,classified,inside)
      +filled = Equal.trans(Nat,Nat.add(start,1n+before),Nat.add(start,difference),count,
        Equal.cong(Nat,Nat,n => Nat.add(start,n),1n+before,difference,
          Order.restore_shift(1n,difference,Order.positive_one(difference,Order.positive_difference(count,start,meets)))),
        Order.restore_shift(start,count,Bounds.strict_le(start,count,meets)))
      +reached = Order.strict_follow(before,Nat.mul(1n+quotient,groups),DivisionOrder.next_above(before,groups,positive))
      +covers = {%filled : {Nat.is_le(_,Nat.add(Nat.mul(1n+quotient,groups),start)) == True{} : Bool}
        %Order.commutative(start,Nat.mul(1n+quotient,groups)) : {Nat.is_le(Nat.add(start,1n+before),_) == True{} : Bool}
          Order.offset_le(start,1n+before,Nat.mul(1n+quotient,groups),reached) :
        {Nat.is_le(count,Nat.add(Nat.mul(1n+quotient,groups),start)) == True{} : Bool}}
      %Equal.sym(Bool,Nat.is_lt(start,count),True{},meets) :
        {hit(Bool.pick(Nat,_,1n+quotient,0n),start,groups,index) == Nat.is_eq(Nat.mod(Nat.add(first,index),groups),Nat.sub(groups,1n)) : Bool}
      walk(1n+quotient,start,first,groups,count,positive,aligned_start(first,groups,positive),covers,index,classified,inside)

# The walk of a piece passes exactly the last cells of their groups.
def inner_hit(+first: Nat,+groups: Nat,+count: Nat,+positive: {Nat.is_lt(0n,groups) == True{} : Bool},+index: Nat,
  +inside: {Nat.is_lt(index,count) == True{} : Bool}) ->
  {hit(walk_steps(walk_start(first,groups),groups,count),walk_start(first,groups),groups,index) ==
    Nat.is_eq(Nat.mod(Nat.add(first,index),groups),Nat.sub(groups,1n)) : Bool}:
  inner_cases(first,groups,count,positive,index,inside,Nat.is_le(walk_start(first,groups),index),{==})

def reach_from(quotient: Nat,+start: Nat,+groups: Nat,+length: Nat,+below: {Nat.is_lt(Nat.add(Nat.mul(quotient,groups),start),length) == True{} : Bool}) ->
  Reach(1n+quotient,start,groups,length):
  match quotient:
    case 0n: KernelGemm.Both{below,Unit{}}
    case 1n+ +more:
      +product = Nat.mul(more,groups)
      +moved = {%regroup(groups,product,start) : {Nat.is_lt(_,length) == True{} : Bool} below :
        {Nat.is_lt(Nat.add(product,Nat.add(start,groups)),length) == True{} : Bool}}
      +here = Order.le_lt_trans(start,Nat.add(Nat.add(groups,product),start),length,
        %Order.commutative(start,Nat.add(groups,product)) : {Nat.is_le(start,_) == True{} : Bool} Order.le_add(start,Nat.add(groups,product)),below)
      KernelGemm.Both{here,reach_from(more,Nat.add(start,groups),groups,length,moved)}

def reach_cases(+first: Nat,+groups: Nat,+count: Nat,+length: Nat,+positive: {Nat.is_lt(0n,groups) == True{} : Bool},
  +room: {Nat.is_le(count,length) == True{} : Bool},meets: Bool,+classified: {Nat.is_lt(walk_start(first,groups),count) == meets : Bool}) ->
  Reach(walk_steps(walk_start(first,groups),groups,count),walk_start(first,groups),groups,length):
  match meets:
    case False{}:
      %Equal.sym(Bool,Nat.is_lt(walk_start(first,groups),count),False{},classified) :
        Reach(Bool.pick(Nat,_,1n+Nat.div(Nat.sub(Nat.sub(count,walk_start(first,groups)),1n),groups),0n),walk_start(first,groups),groups,length)
      Unit{}
    case True{}:
      +start = walk_start(first,groups)
      +difference = Nat.sub(count,start)
      +before = Nat.sub(difference,1n)
      +quotient = Nat.div(before,groups)
      +product = Nat.mul(quotient,groups)
      +filled = Equal.trans(Nat,Nat.add(start,1n+before),Nat.add(start,difference),count,
        Equal.cong(Nat,Nat,n => Nat.add(start,n),1n+before,difference,
          Order.restore_shift(1n,difference,Order.positive_one(difference,Order.positive_difference(count,start,classified)))),
        Order.restore_shift(start,count,Bounds.strict_le(start,count,classified)))
      +ending = {%filled : {Nat.is_lt(Nat.add(start,before),_) == True{} : Bool}
        %Equal.sym(Nat,Nat.add(start,1n+before),1n+Nat.add(start,before),Addition.successor_right(start,before)) : {Nat.is_lt(Nat.add(start,before),_) == True{} : Bool}
          Bounds.predecessor_strict(Nat.add(start,before),1n+Nat.add(start,before),Order.reflexive(1n+Nat.add(start,before))) :
        {Nat.is_lt(Nat.add(start,before),count) == True{} : Bool}}
      +bounded = {%Order.commutative(start,product) : {Nat.is_le(_,Nat.add(start,before)) == True{} : Bool}
        Order.offset_le(start,product,before,DivisionOrder.product_below(before,groups,positive)) :
        {Nat.is_le(Nat.add(product,start),Nat.add(start,before)) == True{} : Bool}}
      %Equal.sym(Bool,Nat.is_lt(start,count),True{},classified) :
        Reach(Bool.pick(Nat,_,1n+quotient,0n),start,groups,length)
      reach_from(quotient,start,groups,length,
        Bounds.nat_trans(Nat.add(product,start),count,length,Order.le_lt_trans(Nat.add(product,start),Nat.add(start,before),count,bounded,ending),room))

# Every position of the walk of a piece lies inside it.
def inner_reach(+first: Nat,+groups: Nat,+count: Nat,+length: Nat,+positive: {Nat.is_lt(0n,groups) == True{} : Bool},
  +room: {Nat.is_le(count,length) == True{} : Bool}) -> Reach(walk_steps(walk_start(first,groups),groups,count),walk_start(first,groups),groups,length):
  reach_cases(first,groups,count,length,positive,room,Nat.is_lt(walk_start(first,groups),count),{==})

# ---- Words of the walk and the range ----

def groups_bounded(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+groups: Nat,+double: {Nat.is_le(Nat.double(groups),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(groups,U32.to_nat(cap)) == True{} : Bool}:
  Order.transitive(groups,Nat.double(groups),U32.to_nat(cap),
    %Addition.double(groups) : {Nat.is_le(groups,_) == True{} : Bool} Order.le_add(groups,groups),double)

# The difference of two words in order is the difference of their values.
# The first walk position of a piece as a word.
def start_exact(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+first: U32,+groups: U32,+positive: {Nat.is_lt(0n,U32.to_nat(groups)) == True{} : Bool},
  +double: {Nat.is_le(Nat.double(U32.to_nat(groups)),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat(((groups - 1) - first % groups : U32)) == walk_start(U32.to_nat(first),U32.to_nat(groups)) : Nat}:
  +g = U32.to_nat(groups)
  +bounded = groups_bounded(cap,cap_ok,g,double)
  +last_exact = Arithmetic.difference_exact(cap,cap_ok,groups,1,g,1n,{==},{==},Order.positive_one(g,positive),bounded)
  Arithmetic.difference_exact(cap,cap_ok,(groups - 1 : U32),(first % groups : U32),Nat.sub(g,1n),Nat.mod(U32.to_nat(first),g),last_exact,
    U32Division.modulo(first,groups,cap,cap_ok,double,positive),
    below_last(Nat.mod(U32.to_nat(first),g),g,positive,DivisionOrder.remainder_small(U32.to_nat(first),g,positive)),
    Order.transitive(Nat.sub(g,1n),g,U32.to_nat(cap),Order.sub_le(g,1n),bounded))

# One past a word below the cap is one past its value.
def successor_exact(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+word: U32,+value: Nat,+exact: {U32.to_nat(word) == value : Nat},+below: {Nat.is_lt(value,U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat((word + 1 : U32)) == 1n+value : Nat}:
  +w = U32.to_nat(word)
  %exact : {U32.to_nat((word + 1 : U32)) == 1n+_ : Nat}
  Equal.trans(Nat,U32.to_nat((word + 1 : U32)),Nat.add(w,U32.to_nat(1)),1n+w,
    Arithmetic.add(word,1,cap,cap_ok,
      %Equal.sym(Nat,Nat.add(w,U32.to_nat(1)),1n+w,Addition.successor_width(w)) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
        Order.strict_follow(w,U32.to_nat(cap),%Equal.sym(Nat,w,value,exact) : {Nat.is_lt(_,U32.to_nat(cap)) == True{} : Bool} below)),
    Addition.successor_width(w))

# The number of runs that end inside a piece, as the word the runtime computes.
def next_exact(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+start: U32,+count: U32,+groups: U32,+start_value: Nat,+exact: {U32.to_nat(start) == start_value : Nat},
  +positive: {Nat.is_lt(0n,U32.to_nat(groups)) == True{} : Bool},+double: {Nat.is_le(Nat.double(U32.to_nat(groups)),U32.to_nat(cap)) == True{} : Bool},
  +span: {Nat.is_le(U32.to_nat(count),U32.to_nat(cap)) == True{} : Bool},+meets: {Nat.is_lt(start_value,U32.to_nat(count)) == True{} : Bool}) ->
  {U32.to_nat(((count - start - 1) / groups + 1 : U32)) == 1n+Nat.div(Nat.sub(Nat.sub(U32.to_nat(count),start_value),1n),U32.to_nat(groups)) : Nat}:
  +c = U32.to_nat(count)
  +g = U32.to_nat(groups)
  +difference = Nat.sub(c,start_value)
  +before = Nat.sub(difference,1n)
  +quotient = Nat.div(before,g)
  +d_exact = Arithmetic.difference_exact(cap,cap_ok,count,start,c,start_value,{==},exact,Bounds.strict_le(start_value,c,meets),span)
  +reaches = Order.positive_one(difference,Order.positive_difference(c,start_value,meets))
  +d_bounded = Order.transitive(difference,c,U32.to_nat(cap),Order.sub_le(c,start_value),span)
  +b_exact = Arithmetic.difference_exact(cap,cap_ok,(count - start : U32),1,difference,1n,d_exact,{==},reaches,d_bounded)
  +q_exact = Equal.trans(Nat,U32.to_nat(((count - start - 1) / groups : U32)),Nat.div(U32.to_nat((count - start - 1 : U32)),g),quotient,
    U32Division.divide((count - start - 1 : U32),groups,cap,cap_ok,double,positive),
    Equal.cong(Nat,Nat,n => Nat.div(n,g),U32.to_nat((count - start - 1 : U32)),before,b_exact))
  +restored = Order.restore_shift(1n,difference,reaches)
  +before_below = {%restored : {Nat.is_lt(before,_) == True{} : Bool} Bounds.predecessor_strict(before,1n+before,Order.reflexive(1n+before)) :
    {Nat.is_lt(before,difference) == True{} : Bool}}
  +quotient_below = Order.le_lt_trans(quotient,before,U32.to_nat(cap),DivisionOrder.quotient_at_most(before,g,positive),
    Bounds.nat_trans(before,difference,U32.to_nat(cap),before_below,d_bounded))
  successor_exact(cap,cap_ok,((count - start - 1) / groups : U32),quotient,q_exact,quotient_below)

def steps_cases(+start: U32,+count: U32,+start_value: Nat,+groups: Nat,+exact: {U32.to_nat(start) == start_value : Nat},+next: U32,
  next_value: {Nat.is_lt(start_value,U32.to_nat(count)) == True{} : Bool} ->
    {U32.to_nat(next) == 1n+Nat.div(Nat.sub(Nat.sub(U32.to_nat(count),start_value),1n),groups) : Nat},
  meets: Bool,+classified: {U32.is_lt(start,count) == meets : Bool}) ->
  {U32.to_nat(Bool.pick(U32,meets,next,0)) == walk_steps(start_value,groups,U32.to_nat(count)) : Nat}:
  match meets:
    case False{}:
      +apart = Equal.trans(Bool,Nat.is_lt(start_value,U32.to_nat(count)),U32.is_lt(start,count),False{},
        %exact : {Nat.is_lt(_,U32.to_nat(count)) == U32.is_lt(start,count) : Bool}
          Equal.sym(Bool,U32.is_lt(start,count),Nat.is_lt(U32.to_nat(start),U32.to_nat(count)),Comparison.lt(start,count)),classified)
      %Equal.sym(Bool,Nat.is_lt(start_value,U32.to_nat(count)),False{},apart) :
        {U32.to_nat(0) == Bool.pick(Nat,_,1n+Nat.div(Nat.sub(Nat.sub(U32.to_nat(count),start_value),1n),groups),0n) : Nat}
      {==}
    case True{}:
      +meets_value = Equal.trans(Bool,Nat.is_lt(start_value,U32.to_nat(count)),U32.is_lt(start,count),True{},
        %exact : {Nat.is_lt(_,U32.to_nat(count)) == U32.is_lt(start,count) : Bool}
          Equal.sym(Bool,U32.is_lt(start,count),Nat.is_lt(U32.to_nat(start),U32.to_nat(count)),Comparison.lt(start,count)),classified)
      %Equal.sym(Bool,Nat.is_lt(start_value,U32.to_nat(count)),True{},meets_value) :
        {U32.to_nat(next) == Bool.pick(Nat,_,1n+Nat.div(Nat.sub(Nat.sub(U32.to_nat(count),start_value),1n),groups),0n) : Nat}
      next_value(meets_value)

# ---- Reading the innermost groups ----

def below_one(value: Nat,+less: {Nat.is_lt(value,1n) == True{} : Bool}) -> {Nat.is_eq(value,0n) == True{} : Bool}:
  match value:
    case 0n: {==}
    case 1n+rest:
      match rest:
        case 0n: Empty.absurd({Nat.is_eq(1n,0n) == True{} : Bool},WordEncoding.false_true(less))
        case 1n+more: Empty.absurd({Nat.is_eq(2n+more,0n) == True{} : Bool},WordEncoding.false_true(less))

# With one group every cell is the last of its group.
def single_last(+first: Nat,+groups: Nat,+positive: {Nat.is_lt(0n,groups) == True{} : Bool},+small: {Nat.is_le(groups,1n) == True{} : Bool},+index: Nat) ->
  {Nat.is_eq(Nat.mod(Nat.add(first,index),groups),Nat.sub(groups,1n)) == True{} : Bool}:
  +one = Order.antisymmetric(groups,1n,small,Order.positive_one(groups,positive))
  %Equal.sym(Nat,groups,1n,one) : {Nat.is_eq(Nat.mod(Nat.add(first,index),_),Nat.sub(groups,1n)) == True{} : Bool}
  %Equal.sym(Nat,groups,1n,one) : {Nat.is_eq(Nat.mod(Nat.add(first,index),1n),Nat.sub(_,1n)) == True{} : Bool}
  below_one(Nat.mod(Nat.add(first,index),1n),DivisionOrder.remainder_small(Nat.add(first,index),1n,{==}))

def inner_read(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},single: Bool,+first: U32,+count: U32,+groups: U32,+tail: Cells.Tail,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,
  +ready: Buffer.ContentReady<Tile.Tile8,Buffer.Content{tree,length}>,+classified: {U32.is_le(groups,1) == single : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(groups)) == True{} : Bool},+double: {Nat.is_le(Nat.double(U32.to_nat(groups)),U32.to_nat(cap)) == True{} : Bool},
  +bounded: {Nat.is_le(length,U32.to_nat(cap)) == True{} : Bool},+room: {Nat.is_le(U32.to_nat(count),length) == True{} : Bool},+index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,inner_step(single,first,count,groups,tail,Buffer.Content{tree,length})),U32.from_nat(index)) ==
    kept(Nat.is_eq(Nat.mod(Nat.add(U32.to_nat(first),index),U32.to_nat(groups)),Nat.sub(U32.to_nat(groups),1n)),tail,Tree.read(Tile.Tile8,tree,U32.from_nat(index))) : Tile.Tile8}:
  match single:
    case True{}:
      +f = U32.to_nat(first)
      +c = U32.to_nat(count)
      +g = U32.to_nat(groups)
      +original = Tree.read(Tile.Tile8,tree,U32.from_nat(index))
      +last = Nat.is_eq(Nat.mod(Nat.add(f,index),g),Nat.sub(g,1n))
      +small = Equal.trans(Bool,Nat.is_le(g,1n),U32.is_le(groups,1),True{},Equal.sym(Bool,U32.is_le(groups,1),Nat.is_le(g,U32.to_nat(1)),Comparison.le(groups,1)),classified)
      +covered = {%Equal.sym(Bool,Nat.is_le(0n,index),True{},Order.zero_le(index)) : {_ && Nat.is_lt(index,c) == True{} : Bool} inside :
        {Nat.is_le(0n,index) && Nat.is_lt(index,c) == True{} : Bool}}
      Equal.trans(Tile.Tile8,
        Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,at_step(0,count,tail,Buffer.Content{tree,length})),U32.from_nat(index)),
        kept(Nat.is_le(0n,index) && Nat.is_lt(index,c),tail,original),kept(last,tail,original),
        at_read(0,count,tail,tree,length,ready,room,index,Bounds.nat_trans(index,c,length,inside,room)),
        Equal.cong(Bool,Tile.Tile8,chosen => kept(chosen,tail,original),Nat.is_le(0n,index) && Nat.is_lt(index,c),last,
          Equal.trans(Bool,Nat.is_le(0n,index) && Nat.is_lt(index,c),True{},last,covered,Equal.sym(Bool,last,True{},single_last(f,g,positive,small,index)))))
    case False{}:
      +f = U32.to_nat(first)
      +c = U32.to_nat(count)
      +g = U32.to_nat(groups)
      +original = Tree.read(Tile.Tile8,tree,U32.from_nat(index))
      +last = Nat.is_eq(Nat.mod(Nat.add(f,index),g),Nat.sub(g,1n))
      +start = ((groups - 1) - first % groups : U32)
      +steps = Bool.pick(U32,U32.is_lt(start,count),((count - start - 1) / groups + 1 : U32),0)
      +origin = walk_start(f,g)
      +origin_exact = start_exact(cap,cap_ok,first,groups,positive,double)
      +steps_exact = steps_cases(start,count,origin,g,origin_exact,((count - start - 1) / groups + 1 : U32),
        meets => next_exact(cap,cap_ok,start,count,groups,origin,origin_exact,positive,double,Order.transitive(c,length,U32.to_nat(cap),room,bounded),meets),
        U32.is_lt(start,count),{==})
      +walked = walk_steps(origin,g,c)
      +reach = {%Equal.sym(Nat,U32.to_nat(start),origin,origin_exact) : Reach(U32.to_nat(steps),_,g,length)
        %Equal.sym(Nat,U32.to_nat(steps),walked,steps_exact) : Reach(_,origin,g,length) inner_reach(f,g,c,length,positive,room) :
        Reach(U32.to_nat(steps),U32.to_nat(start),g,length)}
      +same = {%origin_exact : {hit(U32.to_nat(steps),U32.to_nat(start),g,index) == hit(walked,_,g,index) : Bool}
        %steps_exact : {hit(U32.to_nat(steps),U32.to_nat(start),g,index) == hit(_,U32.to_nat(start),g,index) : Bool} {==} :
        {hit(U32.to_nat(steps),U32.to_nat(start),g,index) == hit(walked,origin,g,index) : Bool}}
      Equal.trans(Tile.Tile8,
        Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,rows_step(U32.to_nat(steps),start,groups,tail,Buffer.Content{tree,length})),U32.from_nat(index)),
        kept(hit(U32.to_nat(steps),U32.to_nat(start),g,index),tail,original),kept(last,tail,original),
        rows_read(cap,cap_ok,U32.to_nat(steps),start,groups,tail,tree,length,ready,bounded,positive,reach,index,Bounds.nat_trans(index,c,length,inside,room)),
        Equal.cong(Bool,Tile.Tile8,chosen => kept(chosen,tail,original),hit(U32.to_nat(steps),U32.to_nat(start),g,index),last,
          Equal.trans(Bool,hit(U32.to_nat(steps),U32.to_nat(start),g,index),hit(walked,origin,g,index),last,same,inner_hit(f,g,c,positive,index,inside))))

# ---- Reading the outermost groups ----

def le_shift(+offset: Nat,+value: Nat,+index: Nat,+ordered: {Nat.is_le(offset,value) == True{} : Bool}) ->
  {Nat.is_le(Nat.sub(value,offset),index) == Nat.is_le(value,Nat.add(offset,index)) : Bool}:
  match offset value:
    case 0n 0n: {==}
    case 0n 1n+value: {==}
    case 1n+offset 0n:
      Empty.absurd({Nat.is_le(Nat.sub(0n,1n+offset),index) == Nat.is_le(0n,Nat.add(1n+offset,index)) : Bool},WordEncoding.false_true(ordered))
    case 1n+offset 1n+value: le_shift(offset,value,index,ordered)

def max_cases(+left: Nat,+right: Nat,+value: Nat,+ordered: {Nat.is_le(right,value) == True{} : Bool},chosen: Bool,+classified: {Nat.is_lt(left,right) == chosen : Bool}) ->
  {Nat.is_le(Nat.max(left,right),value) == Nat.is_le(left,value) : Bool}:
  match chosen:
    case True{}:
      %Equal.sym(Nat,Nat.max(left,right),Bool.pick(Nat,Nat.is_lt(left,right),right,left),Extrema.maximum_choice(left,right)) :
        {Nat.is_le(_,value) == Nat.is_le(left,value) : Bool}
      %Equal.sym(Bool,Nat.is_lt(left,right),True{},classified) : {Nat.is_le(Bool.pick(Nat,_,right,left),value) == Nat.is_le(left,value) : Bool}
      %Equal.sym(Bool,Nat.is_le(right,value),True{},ordered) : {_ == Nat.is_le(left,value) : Bool}
      Equal.sym(Bool,Nat.is_le(left,value),True{},Order.transitive(left,right,value,Bounds.strict_le(left,right,classified),ordered))
    case False{}:
      %Equal.sym(Nat,Nat.max(left,right),Bool.pick(Nat,Nat.is_lt(left,right),right,left),Extrema.maximum_choice(left,right)) :
        {Nat.is_le(_,value) == Nat.is_le(left,value) : Bool}
      %Equal.sym(Bool,Nat.is_lt(left,right),False{},classified) : {Nat.is_le(Bool.pick(Nat,_,right,left),value) == Nat.is_le(left,value) : Bool}
      {==}

def min_cases(+left: Nat,+right: Nat,+value: Nat,+below: {Nat.is_lt(value,right) == True{} : Bool},chosen: Bool,+classified: {Nat.is_lt(left,right) == chosen : Bool}) ->
  {Nat.is_lt(value,Nat.min(left,right)) == Nat.is_lt(value,left) : Bool}:
  match chosen:
    case True{}:
      %Equal.sym(Nat,Nat.min(left,right),Bool.pick(Nat,Nat.is_lt(left,right),left,right),Extrema.minimum_choice(left,right)) :
        {Nat.is_lt(value,_) == Nat.is_lt(value,left) : Bool}
      %Equal.sym(Bool,Nat.is_lt(left,right),True{},classified) : {Nat.is_lt(value,Bool.pick(Nat,_,left,right)) == Nat.is_lt(value,left) : Bool}
      {==}
    case False{}:
      %Equal.sym(Nat,Nat.min(left,right),Bool.pick(Nat,Nat.is_lt(left,right),left,right),Extrema.minimum_choice(left,right)) :
        {Nat.is_lt(value,_) == Nat.is_lt(value,left) : Bool}
      %Equal.sym(Bool,Nat.is_lt(left,right),False{},classified) : {Nat.is_lt(value,Bool.pick(Nat,_,left,right)) == Nat.is_lt(value,left) : Bool}
      %Equal.sym(Bool,Nat.is_lt(value,right),True{},below) : {_ == Nat.is_lt(value,left) : Bool}
      Equal.sym(Bool,Nat.is_lt(value,left),True{},Bounds.nat_trans(value,right,left,below,Order.not_less(left,right,classified)))

def sum_exact(+first: Nat,+low: Nat,+high: Nat,+lower: {Nat.is_le(first,low) == True{} : Bool},+ordered: {Nat.is_le(low,high) == True{} : Bool}) ->
  {Nat.add(Nat.sub(low,first),Nat.sub(high,low)) == Nat.sub(high,first) : Nat}:
  +left = Nat.add(Nat.sub(low,first),Nat.sub(high,low))
  Order.cancel_sum(first,left,Nat.sub(high,first),
    Equal.trans(Nat,Nat.add(first,left),Nat.add(Nat.add(first,Nat.sub(low,first)),Nat.sub(high,low)),Nat.add(first,Nat.sub(high,first)),
      Equal.sym(Nat,Nat.add(Nat.add(first,Nat.sub(low,first)),Nat.sub(high,low)),Nat.add(first,left),Addition.associative(first,Nat.sub(low,first),Nat.sub(high,low))),
      Equal.trans(Nat,Nat.add(Nat.add(first,Nat.sub(low,first)),Nat.sub(high,low)),Nat.add(low,Nat.sub(high,low)),Nat.add(first,Nat.sub(high,first)),
        Equal.cong(Nat,Nat,n => Nat.add(n,Nat.sub(high,low)),Nat.add(first,Nat.sub(low,first)),low,Order.restore_shift(first,low,lower)),
        Equal.trans(Nat,Nat.add(low,Nat.sub(high,low)),high,Nat.add(first,Nat.sub(high,first)),Order.restore_shift(low,high,ordered),
          Equal.sym(Nat,Nat.add(first,Nat.sub(high,first)),high,Order.restore_shift(first,high,Order.transitive(first,low,high,lower,ordered)))))))

# The cells of the last group read the same through the clipped bounds.
def range_spec(+lower: Nat,+upper: Nat,+first: Nat,+count: Nat,+index: Nat,+inside: {Nat.is_lt(index,count) == True{} : Bool}) ->
  {Nat.is_le(lower,Nat.add(first,index)) && Nat.is_lt(Nat.add(first,index),upper) ==
    Nat.is_le(Nat.max(lower,first),Nat.add(first,index)) && Nat.is_lt(Nat.add(first,index),Nat.min(upper,Nat.add(first,count))) : Bool}:
  +cell = Nat.add(first,index)
  +before = {%Equal.sym(Nat,Nat.sub(Nat.add(first,count),first),count,Order.subtract_shift(first,count)) : {Nat.is_lt(index,_) == True{} : Bool} inside : {Nat.is_lt(index,Nat.sub(Nat.add(first,count),first)) == True{} : Bool}}
  +below = Equal.trans(Bool,Nat.is_lt(cell,Nat.add(first,count)),Nat.is_lt(index,Nat.sub(Nat.add(first,count),first)),True{},
    Equal.sym(Bool,Nat.is_lt(index,Nat.sub(Nat.add(first,count),first)),Nat.is_lt(cell,Nat.add(first,count)),
      Order.below_difference(first,index,Nat.add(first,count),Order.le_add(first,count))),before)
  %Equal.sym(Bool,Nat.is_le(Nat.max(lower,first),cell),Nat.is_le(lower,cell),max_cases(lower,first,cell,Order.le_add(first,index),Nat.is_lt(lower,first),{==})) :
    {Nat.is_le(lower,cell) && Nat.is_lt(cell,upper) == _ && Nat.is_lt(cell,Nat.min(upper,Nat.add(first,count))) : Bool}
  %Equal.sym(Bool,Nat.is_lt(cell,Nat.min(upper,Nat.add(first,count))),Nat.is_lt(cell,upper),
      min_cases(upper,Nat.add(first,count),cell,below,Nat.is_lt(upper,Nat.add(first,count)),{==})) :
    {Nat.is_le(lower,cell) && Nat.is_lt(cell,upper) == Nat.is_le(lower,cell) && _ : Bool}
  {==}

# A range read from its clipped bounds.
def range_bool(+lower: Nat,+upper: Nat,+first: Nat,+count: Nat,+index: Nat,+inside: {Nat.is_lt(index,count) == True{} : Bool},
  +ordered: {Nat.is_le(Nat.max(lower,first),Nat.min(upper,Nat.add(first,count))) == True{} : Bool}) ->
  {Nat.is_le(Nat.sub(Nat.max(lower,first),first),index) &&
    Nat.is_lt(index,Nat.add(Nat.sub(Nat.max(lower,first),first),Nat.sub(Nat.min(upper,Nat.add(first,count)),Nat.max(lower,first)))) ==
    Nat.is_le(lower,Nat.add(first,index)) && Nat.is_lt(Nat.add(first,index),upper) : Bool}:
  +low = Nat.max(lower,first)
  +high = Nat.min(upper,Nat.add(first,count))
  +cell = Nat.add(first,index)
  +spec = Nat.is_le(lower,cell) && Nat.is_lt(cell,upper)
  +from_first = Extrema.maximum_right(lower,first)
  %Equal.sym(Nat,Nat.add(Nat.sub(low,first),Nat.sub(high,low)),Nat.sub(high,first),sum_exact(first,low,high,from_first,ordered)) :
    {Nat.is_le(Nat.sub(low,first),index) && Nat.is_lt(index,_) == spec : Bool}
  %Equal.sym(Bool,Nat.is_lt(index,Nat.sub(high,first)),Nat.is_lt(cell,high),
      Order.below_difference(first,index,high,Order.transitive(first,low,high,from_first,ordered))) :
    {Nat.is_le(Nat.sub(low,first),index) && _ == spec : Bool}
  %Equal.sym(Bool,Nat.is_le(Nat.sub(low,first),index),Nat.is_le(low,cell),le_shift(first,low,index,from_first)) :
    {_ && Nat.is_lt(cell,high) == spec : Bool}
  Equal.sym(Bool,spec,Nat.is_le(low,cell) && Nat.is_lt(cell,high),range_spec(lower,upper,first,count,index,inside))

def empty_high(+low: Nat,+high: Nat,+value: Nat,+apart: {Nat.is_lt(low,high) == False{} : Bool},+lower: {Nat.is_le(low,value) == True{} : Bool},
  upper: Bool,+classified: {Nat.is_lt(value,high) == upper : Bool}) -> {Nat.is_le(low,value) && Nat.is_lt(value,high) == False{} : Bool}:
  match upper:
    case True{}:
      Empty.absurd({Nat.is_le(low,value) && Nat.is_lt(value,high) == False{} : Bool},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(low,high),True{},Equal.sym(Bool,Nat.is_lt(low,high),False{},apart),
          Order.le_lt_trans(low,value,high,lower,classified))))
    case False{}:
      %Equal.sym(Bool,Nat.is_le(low,value),True{},lower) : {_ && Nat.is_lt(value,high) == False{} : Bool}
      %Equal.sym(Bool,Nat.is_lt(value,high),False{},classified) : {True{} && _ == False{} : Bool}
      {==}

# No cell lies in an empty range.
def empty_range(+low: Nat,+high: Nat,+value: Nat,+apart: {Nat.is_lt(low,high) == False{} : Bool},lower: Bool,+classified: {Nat.is_le(low,value) == lower : Bool}) ->
  {Nat.is_le(low,value) && Nat.is_lt(value,high) == False{} : Bool}:
  match lower:
    case True{}: empty_high(low,high,value,apart,classified,Nat.is_lt(value,high),{==})
    case False{}:
      %Equal.sym(Bool,Nat.is_le(low,value),False{},classified) : {_ && Nat.is_lt(value,high) == False{} : Bool}
      {==}

def outer_cases(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},meets: Bool,+low: U32,+high: U32,+first: U32,+count: U32,+lower: Nat,+upper: Nat,+tail: Cells.Tail,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,
  +ready: Buffer.ContentReady<Tile.Tile8,Buffer.Content{tree,length}>,
  +low_exact: {U32.to_nat(low) == Nat.max(lower,U32.to_nat(first)) : Nat},
  +high_exact: {U32.to_nat(high) == Nat.min(upper,Nat.add(U32.to_nat(first),U32.to_nat(count))) : Nat},
  +low_bound: {Nat.is_le(Nat.max(lower,U32.to_nat(first)),U32.to_nat(cap)) == True{} : Bool},
  +high_bound: {Nat.is_le(Nat.min(upper,Nat.add(U32.to_nat(first),U32.to_nat(count))),U32.to_nat(cap)) == True{} : Bool},
  +classified: {U32.is_lt(low,high) == meets : Bool},
  +room: {Nat.is_le(U32.to_nat(count),length) == True{} : Bool},+index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,range_step(meets,(low - first : U32),(high - low : U32),tail,Buffer.Content{tree,length})),U32.from_nat(index)) ==
    kept(Nat.is_le(lower,Nat.add(U32.to_nat(first),index)) && Nat.is_lt(Nat.add(U32.to_nat(first),index),upper),tail,Tree.read(Tile.Tile8,tree,U32.from_nat(index))) : Tile.Tile8}:
  match meets:
    case False{}:
      +f = U32.to_nat(first)
      +c = U32.to_nat(count)
      +l = Nat.max(lower,f)
      +h = Nat.min(upper,Nat.add(f,c))
      +cell = Nat.add(f,index)
      +spec = Nat.is_le(lower,cell) && Nat.is_lt(cell,upper)
      +original = Tree.read(Tile.Tile8,tree,U32.from_nat(index))
      +apart = Equal.trans(Bool,Nat.is_lt(l,h),U32.is_lt(low,high),False{},
        %low_exact : {Nat.is_lt(_,h) == U32.is_lt(low,high) : Bool}
          %high_exact : {Nat.is_lt(U32.to_nat(low),_) == U32.is_lt(low,high) : Bool}
            Equal.sym(Bool,U32.is_lt(low,high),Nat.is_lt(U32.to_nat(low),U32.to_nat(high)),Comparison.lt(low,high)),classified)
      Equal.cong(Bool,Tile.Tile8,chosen => kept(chosen,tail,original),False{},spec,
        Equal.sym(Bool,spec,False{},Equal.trans(Bool,spec,Nat.is_le(l,cell) && Nat.is_lt(cell,h),False{},range_spec(lower,upper,f,c,index,inside),
          empty_range(l,h,cell,apart,Nat.is_le(l,cell),{==}))))
    case True{}:
      +f = U32.to_nat(first)
      +c = U32.to_nat(count)
      +l = Nat.max(lower,f)
      +h = Nat.min(upper,Nat.add(f,c))
      +cell = Nat.add(f,index)
      +spec = Nat.is_le(lower,cell) && Nat.is_lt(cell,upper)
      +original = Tree.read(Tile.Tile8,tree,U32.from_nat(index))
      +meets_value = Equal.trans(Bool,Nat.is_lt(l,h),U32.is_lt(low,high),True{},
        %low_exact : {Nat.is_lt(_,h) == U32.is_lt(low,high) : Bool}
          %high_exact : {Nat.is_lt(U32.to_nat(low),_) == U32.is_lt(low,high) : Bool}
            Equal.sym(Bool,U32.is_lt(low,high),Nat.is_lt(U32.to_nat(low),U32.to_nat(high)),Comparison.lt(low,high)),classified)
      +ordered = Bounds.strict_le(l,h,meets_value)
      +from_first = Extrema.maximum_right(lower,f)
      +start = (low - first : U32)
      +size = (high - low : U32)
      +start_exact = Arithmetic.difference_exact(cap,cap_ok,low,first,l,f,low_exact,{==},from_first,low_bound)
      +size_exact = Arithmetic.difference_exact(cap,cap_ok,high,low,h,l,high_exact,low_exact,ordered,high_bound)
      +span = {%Order.subtract_shift(f,c) : {Nat.is_le(Nat.sub(h,f),_) == True{} : Bool}
        DivisionOrder.subtract_le(h,Nat.add(f,c),f,Extrema.minimum_right(upper,Nat.add(f,c))) : {Nat.is_le(Nat.sub(h,f),c) == True{} : Bool}}
      +room_value = {%Equal.sym(Nat,Nat.add(Nat.sub(l,f),Nat.sub(h,l)),Nat.sub(h,f),sum_exact(f,l,h,from_first,ordered)) : {Nat.is_le(_,length) == True{} : Bool}
        Order.transitive(Nat.sub(h,f),c,length,span,room) : {Nat.is_le(Nat.add(Nat.sub(l,f),Nat.sub(h,l)),length) == True{} : Bool}}
      +room_word = {%Equal.sym(Nat,U32.to_nat(start),Nat.sub(l,f),start_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(size)),length) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(size),Nat.sub(h,l),size_exact) : {Nat.is_le(Nat.add(Nat.sub(l,f),_),length) == True{} : Bool} room_value :
        {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(size)),length) == True{} : Bool}}
      +ranged = Nat.is_le(U32.to_nat(start),index) && Nat.is_lt(index,Nat.add(U32.to_nat(start),U32.to_nat(size)))
      +same = {%Equal.sym(Nat,U32.to_nat(start),Nat.sub(l,f),start_exact) : {Nat.is_le(_,index) && Nat.is_lt(index,Nat.add(U32.to_nat(start),U32.to_nat(size))) == spec : Bool}
        %Equal.sym(Nat,U32.to_nat(start),Nat.sub(l,f),start_exact) : {Nat.is_le(Nat.sub(l,f),index) && Nat.is_lt(index,Nat.add(_,U32.to_nat(size))) == spec : Bool}
          %Equal.sym(Nat,U32.to_nat(size),Nat.sub(h,l),size_exact) : {Nat.is_le(Nat.sub(l,f),index) && Nat.is_lt(index,Nat.add(Nat.sub(l,f),_)) == spec : Bool}
            range_bool(lower,upper,f,c,index,inside,ordered) : {ranged == spec : Bool}}
      Equal.trans(Tile.Tile8,
        Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,at_step(start,size,tail,Buffer.Content{tree,length})),U32.from_nat(index)),
        kept(ranged,tail,original),kept(spec,tail,original),
        at_read(start,size,tail,tree,length,ready,room_word,index,Bounds.nat_trans(index,c,length,inside,room)),
        Equal.cong(Bool,Tile.Tile8,chosen => kept(chosen,tail,original),ranged,spec,same))

def outer_read(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+first: U32,+count: U32,+groups: U32,+outer: U32,+tail: Cells.Tail,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,
  +ready: Buffer.ContentReady<Tile.Tile8,Buffer.Content{tree,length}>,+positive: {Nat.is_lt(0n,U32.to_nat(groups)) == True{} : Bool},
  +double: {Nat.is_le(Nat.double(U32.to_nat(groups)),U32.to_nat(cap)) == True{} : Bool},
  +product: {Nat.is_le(Nat.mul(U32.to_nat(groups),U32.to_nat(outer)),U32.to_nat(cap)) == True{} : Bool},
  +span: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(count),length) == True{} : Bool},+index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,placed_step(Grid.Outermost{},first,count,groups,outer,tail,Buffer.Content{tree,length})),U32.from_nat(index)) ==
    kept(last(Grid.Outermost{},U32.to_nat(groups),U32.to_nat(outer),Nat.add(U32.to_nat(first),index)),tail,Tree.read(Tile.Tile8,tree,U32.from_nat(index))) : Tile.Tile8}:
  +f = U32.to_nat(first)
  +c = U32.to_nat(count)
  +g = U32.to_nat(groups)
  +o = U32.to_nat(outer)
  +lower = Nat.mul(Nat.sub(g,1n),o)
  +upper = Nat.mul(g,o)
  +end = Nat.add(f,c)
  +bounded = groups_bounded(cap,cap_ok,g,double)
  +before = (groups - 1 : U32)
  +before_exact = Arithmetic.difference_exact(cap,cap_ok,groups,1,g,1n,{==},{==},Order.positive_one(g,positive),bounded)
  +lower_fits = Order.transitive(lower,upper,U32.to_nat(cap),RowMajor.mul_le(Nat.sub(g,1n),g,o,Order.sub_le(g,1n)),product)
  +lower_word = (before * outer : U32)
  +lower_exact = Equal.trans(Nat,U32.to_nat(lower_word),Nat.mul(U32.to_nat(before),o),lower,
    FullRange.multiply(before,outer,cap,%Equal.sym(Nat,U32.to_nat(before),Nat.sub(g,1n),before_exact) : {Nat.is_le(Nat.mul(_,o),U32.to_nat(cap)) == True{} : Bool} lower_fits),
    Equal.cong(Nat,Nat,n => Nat.mul(n,o),U32.to_nat(before),Nat.sub(g,1n),before_exact))
  +upper_word = (groups * outer : U32)
  +upper_exact = FullRange.multiply(groups,outer,cap,product)
  +end_word = (first + count : U32)
  +end_exact = Arithmetic.add(first,count,cap,cap_ok,span)
  +low = U32.max(lower_word,first)
  +high = U32.min(upper_word,end_word)
  +low_exact = Equal.trans(Nat,U32.to_nat(low),Nat.max(U32.to_nat(lower_word),f),Nat.max(lower,f),NativeExtrema.maximum(lower_word,first),
    Equal.cong(Nat,Nat,n => Nat.max(n,f),U32.to_nat(lower_word),lower,lower_exact))
  +high_exact = Equal.trans(Nat,U32.to_nat(high),Nat.min(U32.to_nat(upper_word),U32.to_nat(end_word)),Nat.min(upper,end),NativeExtrema.minimum(upper_word,end_word),
    Equal.trans(Nat,Nat.min(U32.to_nat(upper_word),U32.to_nat(end_word)),Nat.min(upper,U32.to_nat(end_word)),Nat.min(upper,end),
      Equal.cong(Nat,Nat,n => Nat.min(n,U32.to_nat(end_word)),U32.to_nat(upper_word),upper,upper_exact),
      Equal.cong(Nat,Nat,n => Nat.min(upper,n),U32.to_nat(end_word),end,end_exact)))
  +low_bound = Extrema.maximum_upper(lower,f,U32.to_nat(cap),lower_fits,Order.transitive(f,end,U32.to_nat(cap),Order.le_add(f,c),span))
  +high_bound = Order.transitive(Nat.min(upper,end),upper,U32.to_nat(cap),Extrema.minimum_left(upper,end),product)
  outer_cases(cap,cap_ok,U32.is_lt(low,high),low,high,first,count,lower,upper,tail,tree,length,ready,low_exact,high_exact,low_bound,high_bound,{==},room,index,inside)

# ---- Reading a padded piece ----

# What the padding of the last group of a piece needs of its words.
def Unaligned(+cap: U32,+groups: Nat,+outer: Nat,+first: Nat,+count: Nat) -> Data:
  KernelGemm.Both<{Nat.is_lt(0n,groups) == True{} : Bool},
    KernelGemm.Both<{Nat.is_le(Nat.double(groups),U32.to_nat(cap)) == True{} : Bool},
      KernelGemm.Both<{Nat.is_le(Nat.mul(groups,outer),U32.to_nat(cap)) == True{} : Bool},
        {Nat.is_le(Nat.add(first,count),U32.to_nat(cap)) == True{} : Bool}>>>

def Fits(+cap: U32,aligned: Bool,+groups: Nat,+outer: Nat,+first: Nat,+count: Nat) -> Data:
  match aligned:
    case True{}: Unit
    case False{}: Unaligned(cap,groups,outer,first,count)

def padding_fits(+cap: U32,+padding: TiledMap.Padding,+first: U32,+count: U32) -> Data:
  TiledMap.Padding{+lanes,+groups,+outer,position} = padding
  Fits(cap,U32.is_zero((lanes % 8 : U32)),U32.to_nat(groups),U32.to_nat(outer),U32.to_nat(first),U32.to_nat(count))

def placed_read(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},position: Grid.Position,+first: U32,+count: U32,+groups: U32,+outer: U32,+tail: Cells.Tail,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,
  +ready: Buffer.ContentReady<Tile.Tile8,Buffer.Content{tree,length}>,+fits: Fits(cap,False{},U32.to_nat(groups),U32.to_nat(outer),U32.to_nat(first),U32.to_nat(count)),
  +bounded: {Nat.is_le(length,U32.to_nat(cap)) == True{} : Bool},+room: {Nat.is_le(U32.to_nat(count),length) == True{} : Bool},+index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,placed_step(position,first,count,groups,outer,tail,Buffer.Content{tree,length})),U32.from_nat(index)) ==
    kept(last(position,U32.to_nat(groups),U32.to_nat(outer),Nat.add(U32.to_nat(first),index)),tail,Tree.read(Tile.Tile8,tree,U32.from_nat(index))) : Tile.Tile8}:
  match position:
    case Grid.Innermost{}:
      KernelGemm.Both{+positive,KernelGemm.Both{+double,+rest}} = fits
      inner_read(cap,cap_ok,U32.is_le(groups,1),first,count,groups,tail,tree,length,ready,{==},positive,double,bounded,room,index,inside)
    case Grid.Outermost{}:
      KernelGemm.Both{+positive,KernelGemm.Both{+double,KernelGemm.Both{+product,+span}}} = fits
      outer_read(cap,cap_ok,first,count,groups,outer,tail,tree,length,ready,positive,double,product,span,room,index,inside)

def when_read(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},aligned: Bool,+position: Grid.Position,+first: U32,+count: U32,+lanes: U32,+groups: U32,+outer: U32,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,
  +ready: Buffer.ContentReady<Tile.Tile8,Buffer.Content{tree,length}>,+fits: Fits(cap,aligned,U32.to_nat(groups),U32.to_nat(outer),U32.to_nat(first),U32.to_nat(count)),
  +bounded: {Nat.is_le(length,U32.to_nat(cap)) == True{} : Bool},+room: {Nat.is_le(U32.to_nat(count),length) == True{} : Bool},+index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,when_step(aligned,position,first,count,lanes,groups,outer,Buffer.Content{tree,length})),U32.from_nat(index)) ==
    Bool.pick(Tile.Tile8,aligned,Tree.read(Tile.Tile8,tree,U32.from_nat(index)),
      kept(last(position,U32.to_nat(groups),U32.to_nat(outer),Nat.add(U32.to_nat(first),index)),Cells.Tail{((groups - 1) * 8 : U32),lanes},
        Tree.read(Tile.Tile8,tree,U32.from_nat(index)))) : Tile.Tile8}:
  match aligned:
    case True{}: {==}
    case False{}: placed_read(cap,cap_ok,position,first,count,groups,outer,Cells.Tail{((groups - 1) * 8 : U32),lanes},tree,length,ready,fits,bounded,room,index,inside)

# Padding leaves in every cell of a piece what it leaves in that cell of the
# tensor.
def read(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+padding: TiledMap.Padding,+first: U32,+count: U32,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,
  +ready: Buffer.ContentReady<Tile.Tile8,Buffer.Content{tree,length}>,+fits: padding_fits(cap,padding,first,count),
  +bounded: {Nat.is_le(length,U32.to_nat(cap)) == True{} : Bool},+room: {Nat.is_le(U32.to_nat(count),length) == True{} : Bool},+index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,Buffer.content_tree(~Tile.Tile8,step(padding,first,count,Buffer.Content{tree,length})),U32.from_nat(index)) ==
    cell(padding,Nat.add(U32.to_nat(first),index),Tree.read(Tile.Tile8,tree,U32.from_nat(index))) : Tile.Tile8}:
  TiledMap.Padding{+lanes,+groups,+outer,+position} = padding
  when_read(cap,cap_ok,U32.is_zero((lanes % 8 : U32)),position,first,count,lanes,groups,outer,tree,length,ready,fits,bounded,room,index,inside)

# ---- Padding keeps a piece ready and its length ----

def rows_ready(remaining: Nat,+position: U32,+groups: U32,+tail: Cells.Tail,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,
  +ready: Buffer.ContentReady<Tile.Tile8,Buffer.Content{tree,length}>) ->
  KernelGemm.Both<Buffer.ContentReady<Tile.Tile8,rows_step(remaining,position,groups,tail,Buffer.Content{tree,length})>,
    {Buffer.content_length(~Tile.Tile8,rows_step(remaining,position,groups,tail,Buffer.Content{tree,length})) == length : Nat}>:
  match remaining:
    case 0n: KernelGemm.Both{ready,{==}}
    case 1n+rest:
      rows_ready(rest,(position + groups : U32),groups,tail,MapReadback.run(~Tile.Tile8,~Cells.Tail,~Cells.tail,U32.to_nat(1),position,tail,tree),length,
        at_ready(position,1,tail,tree,length,ready))

def inner_ready(single: Bool,+first: U32,+count: U32,+groups: U32,+tail: Cells.Tail,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,
  +ready: Buffer.ContentReady<Tile.Tile8,Buffer.Content{tree,length}>) ->
  KernelGemm.Both<Buffer.ContentReady<Tile.Tile8,inner_step(single,first,count,groups,tail,Buffer.Content{tree,length})>,
    {Buffer.content_length(~Tile.Tile8,inner_step(single,first,count,groups,tail,Buffer.Content{tree,length})) == length : Nat}>:
  match single:
    case True{}: KernelGemm.Both{at_ready(0,count,tail,tree,length,ready),{==}}
    case False{}:
      +start = ((groups - 1) - first % groups : U32)
      rows_ready(U32.to_nat(Bool.pick(U32,U32.is_lt(start,count),((count - start - 1) / groups + 1 : U32),0)),start,groups,tail,tree,length,ready)

def range_ready(meets: Bool,+start: U32,+count: U32,+tail: Cells.Tail,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,
  +ready: Buffer.ContentReady<Tile.Tile8,Buffer.Content{tree,length}>) ->
  KernelGemm.Both<Buffer.ContentReady<Tile.Tile8,range_step(meets,start,count,tail,Buffer.Content{tree,length})>,
    {Buffer.content_length(~Tile.Tile8,range_step(meets,start,count,tail,Buffer.Content{tree,length})) == length : Nat}>:
  match meets:
    case True{}: KernelGemm.Both{at_ready(start,count,tail,tree,length,ready),{==}}
    case False{}: KernelGemm.Both{ready,{==}}

def placed_ready(position: Grid.Position,+first: U32,+count: U32,+groups: U32,+outer: U32,+tail: Cells.Tail,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,
  +ready: Buffer.ContentReady<Tile.Tile8,Buffer.Content{tree,length}>) ->
  KernelGemm.Both<Buffer.ContentReady<Tile.Tile8,placed_step(position,first,count,groups,outer,tail,Buffer.Content{tree,length})>,
    {Buffer.content_length(~Tile.Tile8,placed_step(position,first,count,groups,outer,tail,Buffer.Content{tree,length})) == length : Nat}>:
  match position:
    case Grid.Innermost{}: inner_ready(U32.is_le(groups,1),first,count,groups,tail,tree,length,ready)
    case Grid.Outermost{}:
      +low = U32.max(((groups - 1) * outer : U32),first)
      +high = U32.min((groups * outer : U32),(first + count : U32))
      range_ready(U32.is_lt(low,high),(low - first : U32),(high - low : U32),tail,tree,length,ready)

def when_ready(aligned: Bool,+position: Grid.Position,+first: U32,+count: U32,+lanes: U32,+groups: U32,+outer: U32,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,
  +ready: Buffer.ContentReady<Tile.Tile8,Buffer.Content{tree,length}>) ->
  KernelGemm.Both<Buffer.ContentReady<Tile.Tile8,when_step(aligned,position,first,count,lanes,groups,outer,Buffer.Content{tree,length})>,
    {Buffer.content_length(~Tile.Tile8,when_step(aligned,position,first,count,lanes,groups,outer,Buffer.Content{tree,length})) == length : Nat}>:
  match aligned:
    case True{}: KernelGemm.Both{ready,{==}}
    case False{}: placed_ready(position,first,count,groups,outer,Cells.Tail{((groups - 1) * 8 : U32),lanes},tree,length,ready)

# Padding leaves a ready piece ready, of the same length.
def kept_ready(+padding: TiledMap.Padding,+first: U32,+count: U32,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,
  +ready: Buffer.ContentReady<Tile.Tile8,Buffer.Content{tree,length}>) ->
  KernelGemm.Both<Buffer.ContentReady<Tile.Tile8,step(padding,first,count,Buffer.Content{tree,length})>,
    {Buffer.content_length(~Tile.Tile8,step(padding,first,count,Buffer.Content{tree,length})) == length : Nat}>:
  TiledMap.Padding{+lanes,+groups,+outer,+position} = padding
  when_ready(U32.is_zero((lanes % 8 : U32)),position,first,count,lanes,groups,outer,tree,length,ready)
