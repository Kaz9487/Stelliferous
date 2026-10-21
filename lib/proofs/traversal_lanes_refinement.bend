# Lanes.map is Traversal.map_in_place: on a balanced array that holds its
# count positions, the rounds through tiles and the ordered scalar map leave
# the same array.
import Base
import ../traversal_lanes.bend as Lanes
import ../traversal_cells.bend as Cells
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ../kernel_tile.bend as Tile
import ./traversal_cells_refinement.bend as CellLaws
import ./traversal_batch_refinement.bend as Batch
import ./traversal_array_refinement.bend as ArrayLaws
import ./traversal_map_readback.bend as MapReadback
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./storage_shape.bend as Shape
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./nat_division.bend as Division
import ./word_encoding.bend as WordEncoding
import ./word_arithmetic.bend as WordArithmetic
import ./word_power_of_two.bend as Power
import ./kernel_tile_refinement.bend as TileLaws
import ./loop_projection.bend as Projection
import ./bounded_u32_arithmetic.bend as Arithmetic

def tile_values(values: Tile.Tile8) -> List<&2,F32>:
  Tile.Tile{a,b,c,d,e,f,g,h} = values
  [a,b,c,d,e,f,g,h]

def cell(~Context: Data,~map: Context -> F32 -> F32,+values: Tree.Tree<F32>,+position: U32,+context: Context) -> Tile.Tile8:
  +p1 = (position + 1 : U32)
  +p2 = (p1 + 1 : U32)
  +p3 = (p2 + 1 : U32)
  +p4 = (p3 + 1 : U32)
  +p5 = (p4 + 1 : U32)
  +p6 = (p5 + 1 : U32)
  +p7 = (p6 + 1 : U32)
  Tile.Tile{map(context,Tree.read(F32,values,position)),map(context,Tree.read(F32,values,p1)),map(context,Tree.read(F32,values,p2)),map(context,Tree.read(F32,values,p3)),
    map(context,Tree.read(F32,values,p4)),map(context,Tree.read(F32,values,p5)),map(context,Tree.read(F32,values,p6)),map(context,Tree.read(F32,values,p7))}

def position_after(count: Nat,position: U32) -> U32:
  match count:
    case 0n: position
    case 1n+rest: position_after(rest,(position + 1 : U32))

def position_after_matches(+count: Nat,+position: U32) -> {position_after(count,position) == (position + U32.from_nat(count) : U32) : U32}:
  match count:
    case 0n: Equal.sym(U32,(position + 0 : U32),position,Arithmetic.add_zero(position))
    case 1n+ +rest:
      Equal.trans(U32,position_after(rest,(position + 1 : U32)),((position + 1 : U32) + U32.from_nat(rest) : U32),(position + U32.from_nat(1n+rest) : U32),position_after_matches(rest,(position + 1 : U32)),Equal.sym(U32,(position + U32.from_nat(1n+rest) : U32),((position + 1 : U32) + U32.from_nat(rest) : U32),Projection.indexed_successor(position,rest)))

def identity_cell_at(+lane: Nat,+values: Tree.Tree<F32>,+position: U32,within: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,cell(~Unit,~(context => value => value),values,position,Unit{})) == Tree.read(F32,values,position_after(lane,position)) : F32}:
  match lane:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n+ +rest:
      Empty.absurd({TileLaws.select_lane(lane,cell(~Unit,~(context => value => value),values,position,Unit{})) == Tree.read(F32,values,position_after(lane,position)) : F32},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(rest,0n),True{},Equal.sym(Bool,Nat.is_lt(rest,0n),False{},Division.lt_zero(rest)),within)))

def identity_cell_indexed(+lane: Nat,+values: Tree.Tree<F32>,+position: U32,within: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,cell(~Unit,~(context => value => value),values,position,Unit{})) == Tree.read(F32,values,(position + U32.from_nat(lane) : U32)) : F32}:
  %position_after_matches(lane,position) : {TileLaws.select_lane(lane,cell(~Unit,~(context => value => value),values,position,Unit{})) == Tree.read(F32,values,_) : F32}
  identity_cell_at(lane,values,position,within)

def identity_cell_equal(+left: Tree.Tree<F32>,+right: Tree.Tree<F32>,+position: U32,
  e0: {Tree.read(F32,left,position_after(0n,position)) == Tree.read(F32,right,position_after(0n,position)) : F32},
  e1: {Tree.read(F32,left,position_after(1n,position)) == Tree.read(F32,right,position_after(1n,position)) : F32},
  e2: {Tree.read(F32,left,position_after(2n,position)) == Tree.read(F32,right,position_after(2n,position)) : F32},
  e3: {Tree.read(F32,left,position_after(3n,position)) == Tree.read(F32,right,position_after(3n,position)) : F32},
  e4: {Tree.read(F32,left,position_after(4n,position)) == Tree.read(F32,right,position_after(4n,position)) : F32},
  e5: {Tree.read(F32,left,position_after(5n,position)) == Tree.read(F32,right,position_after(5n,position)) : F32},
  e6: {Tree.read(F32,left,position_after(6n,position)) == Tree.read(F32,right,position_after(6n,position)) : F32},
  e7: {Tree.read(F32,left,position_after(7n,position)) == Tree.read(F32,right,position_after(7n,position)) : F32}) ->
  {cell(~Unit,~(context => value => value),left,position,Unit{}) == cell(~Unit,~(context => value => value),right,position,Unit{}) : Tile.Tile8}:
  TileLaws.tiles_equal(cell(~Unit,~(context => value => value),left,position,Unit{}),cell(~Unit,~(context => value => value),right,position,Unit{}),e0,e1,e2,e3,e4,e5,e6,e7)

def gathered(~Context: Data,~map: Context -> F32 -> F32,+values: Tree.Tree<F32>,+position: U32,+context: Context) ->
  {Lanes.gather(~Context,~map,Tree.pack(F32,values),position,context) == (Tree.pack(F32,values),cell(~Context,~map,values,position,context)) : Array<F32> & Tile.Tile8}:
  -owner = Tree.pack(F32,values)
  +p1 = (position + 1 : U32)
  +p2 = (p1 + 1 : U32)
  +p3 = (p2 + 1 : U32)
  +p4 = (p3 + 1 : U32)
  +p5 = (p4 + 1 : U32)
  +p6 = (p5 + 1 : U32)
  +p7 = (p6 + 1 : U32)
  +a = Tree.read(F32,values,position)
  +b = Tree.read(F32,values,p1)
  +c = Tree.read(F32,values,p2)
  +d = Tree.read(F32,values,p3)
  +e = Tree.read(F32,values,p4)
  +f = Tree.read(F32,values,p5)
  +g = Tree.read(F32,values,p6)
  +h = Tree.read(F32,values,p7)
  %Equal.sym(Array<F32> & F32,Array.get(F32,owner,position),(owner,a),StorageLaws.read(F32,values,position)) :
    {Lanes.gather1(~Context,~map,position,context,_) == (owner,cell(~Context,~map,values,position,context)) : Array<F32> & Tile.Tile8}
  %Equal.sym(Array<F32> & F32,Array.get(F32,owner,p1),(owner,b),StorageLaws.read(F32,values,p1)) :
    {Lanes.gather2(~Context,~map,p1,context,a,_) == (owner,cell(~Context,~map,values,position,context)) : Array<F32> & Tile.Tile8}
  %Equal.sym(Array<F32> & F32,Array.get(F32,owner,p2),(owner,c),StorageLaws.read(F32,values,p2)) :
    {Lanes.gather3(~Context,~map,p2,context,a,b,_) == (owner,cell(~Context,~map,values,position,context)) : Array<F32> & Tile.Tile8}
  %Equal.sym(Array<F32> & F32,Array.get(F32,owner,p3),(owner,d),StorageLaws.read(F32,values,p3)) :
    {Lanes.gather4(~Context,~map,p3,context,a,b,c,_) == (owner,cell(~Context,~map,values,position,context)) : Array<F32> & Tile.Tile8}
  %Equal.sym(Array<F32> & F32,Array.get(F32,owner,p4),(owner,e),StorageLaws.read(F32,values,p4)) :
    {Lanes.gather5(~Context,~map,p4,context,a,b,c,d,_) == (owner,cell(~Context,~map,values,position,context)) : Array<F32> & Tile.Tile8}
  %Equal.sym(Array<F32> & F32,Array.get(F32,owner,p5),(owner,f),StorageLaws.read(F32,values,p5)) :
    {Lanes.gather6(~Context,~map,p5,context,a,b,c,d,e,_) == (owner,cell(~Context,~map,values,position,context)) : Array<F32> & Tile.Tile8}
  %Equal.sym(Array<F32> & F32,Array.get(F32,owner,p6),(owner,g),StorageLaws.read(F32,values,p6)) :
    {Lanes.gather7(~Context,~map,p6,context,a,b,c,d,e,f,_) == (owner,cell(~Context,~map,values,position,context)) : Array<F32> & Tile.Tile8}
  %Equal.sym(Array<F32> & F32,Array.get(F32,owner,p7),(owner,h),StorageLaws.read(F32,values,p7)) :
    {Lanes.gathered(~Context,~map,context,a,b,c,d,e,f,g,_) == (owner,cell(~Context,~map,values,position,context)) : Array<F32> & Tile.Tile8}
  {==}

# Eight ordered writes from position.
def written(+values: Tree.Tree<F32>,+position: U32,a: F32,b: F32,c: F32,d: F32,e: F32,f: F32,g: F32,h: F32) -> Tree.Tree<F32>:
  +p1 = (position + 1 : U32)
  +p2 = (p1 + 1 : U32)
  +p3 = (p2 + 1 : U32)
  +p4 = (p3 + 1 : U32)
  +p5 = (p4 + 1 : U32)
  +p6 = (p5 + 1 : U32)
  +p7 = (p6 + 1 : U32)
  Tree.write(F32,Tree.write(F32,Tree.write(F32,Tree.write(F32,Tree.write(F32,Tree.write(F32,Tree.write(F32,Tree.write(F32,values,position,a),p1,b),p2,c),p3,d),p4,e),p5,f),p6,g),p7,h)

def scattered(+values: Tree.Tree<F32>,+position: U32,+a: F32,+b: F32,+c: F32,+d: F32,+e: F32,+f: F32,+g: F32,+h: F32) ->
  {Lanes.scatter(Tree.pack(F32,values),position,Tile.Tile{a,b,c,d,e,f,g,h}) == Tree.pack(F32,written(values,position,a,b,c,d,e,f,g,h)) : Array<F32>}:
  +p1 = (position + 1 : U32)
  +p2 = (p1 + 1 : U32)
  +p3 = (p2 + 1 : U32)
  +p4 = (p3 + 1 : U32)
  +p5 = (p4 + 1 : U32)
  +p6 = (p5 + 1 : U32)
  +p7 = (p6 + 1 : U32)
  +w1 = Tree.write(F32,values,position,a)
  +w2 = Tree.write(F32,w1,p1,b)
  +w3 = Tree.write(F32,w2,p2,c)
  +w4 = Tree.write(F32,w3,p3,d)
  +w5 = Tree.write(F32,w4,p4,e)
  +w6 = Tree.write(F32,w5,p5,f)
  +w7 = Tree.write(F32,w6,p6,g)
  +w8 = Tree.write(F32,w7,p7,h)
  %Equal.sym(Array<F32>,Array.set(F32,Tree.pack(F32,values),position,a),Tree.pack(F32,w1),StorageLaws.write(F32,values,position,a)) :
    {Array.set(F32,Array.set(F32,Array.set(F32,Array.set(F32,Array.set(F32,Array.set(F32,Array.set(F32,_,p1,b),p2,c),p3,d),p4,e),p5,f),p6,g),p7,h) == Tree.pack(F32,w8) : Array<F32>}
  %Equal.sym(Array<F32>,Array.set(F32,Tree.pack(F32,w1),p1,b),Tree.pack(F32,w2),StorageLaws.write(F32,w1,p1,b)) :
    {Array.set(F32,Array.set(F32,Array.set(F32,Array.set(F32,Array.set(F32,Array.set(F32,_,p2,c),p3,d),p4,e),p5,f),p6,g),p7,h) == Tree.pack(F32,w8) : Array<F32>}
  %Equal.sym(Array<F32>,Array.set(F32,Tree.pack(F32,w2),p2,c),Tree.pack(F32,w3),StorageLaws.write(F32,w2,p2,c)) :
    {Array.set(F32,Array.set(F32,Array.set(F32,Array.set(F32,Array.set(F32,_,p3,d),p4,e),p5,f),p6,g),p7,h) == Tree.pack(F32,w8) : Array<F32>}
  %Equal.sym(Array<F32>,Array.set(F32,Tree.pack(F32,w3),p3,d),Tree.pack(F32,w4),StorageLaws.write(F32,w3,p3,d)) :
    {Array.set(F32,Array.set(F32,Array.set(F32,Array.set(F32,_,p4,e),p5,f),p6,g),p7,h) == Tree.pack(F32,w8) : Array<F32>}
  %Equal.sym(Array<F32>,Array.set(F32,Tree.pack(F32,w4),p4,e),Tree.pack(F32,w5),StorageLaws.write(F32,w4,p4,e)) :
    {Array.set(F32,Array.set(F32,Array.set(F32,_,p5,f),p6,g),p7,h) == Tree.pack(F32,w8) : Array<F32>}
  %Equal.sym(Array<F32>,Array.set(F32,Tree.pack(F32,w5),p5,f),Tree.pack(F32,w6),StorageLaws.write(F32,w5,p5,f)) :
    {Array.set(F32,Array.set(F32,_,p6,g),p7,h) == Tree.pack(F32,w8) : Array<F32>}
  %Equal.sym(Array<F32>,Array.set(F32,Tree.pack(F32,w6),p6,g),Tree.pack(F32,w7),StorageLaws.write(F32,w6,p6,g)) :
    {Array.set(F32,_,p7,h) == Tree.pack(F32,w8) : Array<F32>}
  StorageLaws.write(F32,w7,p7,h)

def cell_scattered(~Context: Data,~map: Context -> F32 -> F32,+values: Tree.Tree<F32>,+position: U32,+context: Context) ->
  {Lanes.scatter(Tree.pack(F32,values),position,cell(~Context,~map,values,position,context)) ==
    Tree.pack(F32,Batch.batch(~F32,~Context,~ArrayLaws.unindexed(~F32,~Context,~map),8n,position,context,values,values)) : Array<F32>}:
  +p1 = (position + 1 : U32)
  +p2 = (p1 + 1 : U32)
  +p3 = (p2 + 1 : U32)
  +p4 = (p3 + 1 : U32)
  +p5 = (p4 + 1 : U32)
  +p6 = (p5 + 1 : U32)
  +p7 = (p6 + 1 : U32)
  scattered(values,position,map(context,Tree.read(F32,values,position)),map(context,Tree.read(F32,values,p1)),map(context,Tree.read(F32,values,p2)),map(context,Tree.read(F32,values,p3)),
    map(context,Tree.read(F32,values,p4)),map(context,Tree.read(F32,values,p5)),map(context,Tree.read(F32,values,p6)),map(context,Tree.read(F32,values,p7)))

# One round maps eight positions.
def round(~Context: Data,~map: Context -> F32 -> F32,+values: Tree.Tree<F32>,+scratch: Tree.Tree<Tile.Tile8>,+start: Nat,+slot: U32,+context: Context,
  +depth: Nat,+capacity: U32,+capacity_equal: {Tree.capacity(F32,values) == capacity : U32},+balanced: Shape.Balanced(F32,depth,values),
  +limit: {Nat.is_le(depth,30n) == True{} : Bool},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(start,8n),U32.to_nat(capacity)) == True{} : Bool}) ->
  {Cells.step(~Array<F32>,~Tile.Tile8,~Context,~Lanes.gather(~Context,~map),~Lanes.scatter,~8,
      Cells.State{Tree.pack(F32,values),Tree.pack(Tile.Tile8,scratch),U32.from_nat(start),slot,context}) ==
    Cells.State{Tree.pack(F32,MapReadback.run(~F32,~Context,~map,8n,U32.from_nat(start),context,values)),
      Tree.pack(Tile.Tile8,Tree.write(Tile.Tile8,scratch,slot,cell(~Context,~map,values,U32.from_nat(start),context))),(U32.from_nat(start) + 8 : U32),
      U32.and((slot + 1 : U32),63),context} : Cells.State<Array<F32>,Tile.Tile8,Context>}:
  +position = U32.from_nat(start)
  %Batch.mapped(~F32,~Context,~ArrayLaws.unindexed(~F32,~Context,~map),8n,start,context,values,depth,capacity,capacity_equal,balanced,Order.successor(depth,30n,limit),capacity_ok,room) :
    {Cells.step(~Array<F32>,~Tile.Tile8,~Context,~Lanes.gather(~Context,~map),~Lanes.scatter,~8,Cells.State{Tree.pack(F32,values),Tree.pack(Tile.Tile8,scratch),position,slot,context}) ==
      Cells.State{Tree.pack(F32,_),Tree.pack(Tile.Tile8,Tree.write(Tile.Tile8,scratch,slot,cell(~Context,~map,values,position,context))),(position + 8 : U32),
        U32.and((slot + 1 : U32),63),context} : Cells.State<Array<F32>,Tile.Tile8,Context>}
  CellLaws.round(~F32,~Tile.Tile8,~Context,~Lanes.gather(~Context,~map),~Lanes.scatter,~8,values,scratch,position,slot,context,
    cell(~Context,~map,values,position,context),Batch.batch(~F32,~Context,~ArrayLaws.unindexed(~F32,~Context,~map),8n,position,context,values,values),
    gathered(~Context,~map,values,position,context),cell_scattered(~Context,~map,values,position,context))

# The scalar map from any position, on a packed tree.
def output(~Context: Data,+state: Batch.ModelState(~F32,~Context)) ->
  {Traversal.mapped_output(~F32,~Context,ArrayLaws.represent_map(~F32,~Context,state)) == Tree.pack(F32,MapReadback.output(~F32,~Context,state)) : Array<F32>}:
  Traversal.State{Unit{},output,position,context} = state
  {==}

def scalar(~Context: Data,~map: Context -> F32 -> F32,+count: Nat,+position: U32,+context: Context,+values: Tree.Tree<F32>) ->
  {Traversal.map_in_place(~F32,~Context,~map,count,position,context,Tree.pack(F32,values)) ==
    Tree.pack(F32,MapReadback.run(~F32,~Context,~map,count,position,context,values)) : Array<F32>}:
  +initial = {Traversal.State{Unit{},values,position,context} : Batch.ModelState(~F32,~Context)}
  %Equal.sym(Traversal.State<&2,&1,Unit,Array<F32>,Context>,
    Loop.run(~Traversal.State<&2,&1,Unit,Array<F32>,Context>,~Traversal.map_step(~F32,~Context,~map),count,Traversal.State{Unit{},Tree.pack(F32,values),position,context}),
    ArrayLaws.represent_map(~F32,~Context,Loop.run(~Batch.ModelState(~F32,~Context),~ArrayLaws.model_map_step(~F32,~Context,~ArrayLaws.unindexed(~F32,~Context,~map)),count,initial)),
    ArrayLaws.map_loop_refines_model(~F32,~Context,~map,count,initial)) :
    {Traversal.mapped_output(~F32,~Context,_) == Tree.pack(F32,MapReadback.run(~F32,~Context,~map,count,position,context,values)) : Array<F32>}
  output(~Context,Loop.run(~Batch.ModelState(~F32,~Context),~ArrayLaws.model_map_step(~F32,~Context,~ArrayLaws.unindexed(~F32,~Context,~map)),count,initial))

def few(~Context: Data,~map: Context -> F32 -> F32,+count: Nat,+position: U32,+slot: U32,+context: Context,+values: Tree.Tree<F32>,+scratch: Tree.Tree<Tile.Tile8>) ->
  {Lanes.finish(~Context,~map,count,Cells.State{Tree.pack(F32,values),Tree.pack(Tile.Tile8,scratch),position,slot,context}) ==
    Tree.pack(F32,MapReadback.run(~F32,~Context,~map,count,position,context,values)) : Array<F32>}:
  scalar(~Context,~map,count,position,context,values)

# The position of the model map after k steps from start.
def stepped(steps: Nat,start: Nat) -> U32:
  match steps:
    case 0n: U32.from_nat(start)
    case 1n+rest: (stepped(rest,start) + 1 : U32)

def advanced(+steps: Nat,+start: Nat) -> {stepped(steps,start) == U32.from_nat(Nat.add(steps,start)) : U32}:
  match steps:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(U32,stepped(rest,start),U32.from_nat(Nat.add(rest,start)),advanced(rest,start)) : {(_ + 1 : U32) == U32.from_nat(1n+Nat.add(rest,start)) : U32}
      Bounds.add_one(U32.from_nat(Nat.add(rest,start)))

def after_round(+start: Nat) -> {(U32.from_nat(start) + 8 : U32) == U32.from_nat(8n+start) : U32}:
  %Order.commutative(start,8n) : {(U32.from_nat(start) + 8 : U32) == U32.from_nat(_) : U32}
  WordArithmetic.native_add(start,8n)

def round_room(+start: Nat,+rest: Nat,+capacity: Nat,room: {Nat.is_le(Nat.add(start,8n+rest),capacity) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(start,8n),capacity) == True{} : Bool}:
  Order.transitive(Nat.add(start,8n),Nat.add(start,8n+rest),capacity,Order.offset_le(start,8n,8n+rest,Order.le_add(8n,rest)),room)

def rest_room(+start: Nat,+rest: Nat,+capacity: Nat,room: {Nat.is_le(Nat.add(start,8n+rest),capacity) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(8n+start,rest),capacity) == True{} : Bool}:
  Order.advance_room(7n+start,rest,capacity,Order.advance_room(6n+start,1n+rest,capacity,Order.advance_room(5n+start,2n+rest,capacity,
    Order.advance_room(4n+start,3n+rest,capacity,Order.advance_room(3n+start,4n+rest,capacity,Order.advance_room(2n+start,5n+rest,capacity,
      Order.advance_room(1n+start,6n+rest,capacity,Order.advance_room(start,7n+rest,capacity,room))))))))

# The shared induction on eight-value rounds. The proposition may carry all
# owners and bounds, so binary and ternary instances need only local cases.
def finish_small(~Owner: Type,~Context: Data,~gather: Owner -> U32 -> Context -> Owner & Tile.Tile8,
  ~scatter: Owner -> U32 -> Tile.Tile8 -> Owner,~finish: Nat -> Cells.State<Owner,Tile.Tile8,Context> -> Owner,
  +count: Nat,-state: Cells.State<Owner,Tile.Tile8,Context>,small: {Nat.is_lt(count,8n) == True{} : Bool}) ->
  {Lanes.rounds_with(~Owner,~Context,~gather,~scatter,~finish,count,state) == finish(count,state) : Owner}:
  match count:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 1n+1n+1n+1n+1n+1n+1n+1n+ +rest:
      Empty.absurd({Lanes.rounds_with(~Owner,~Context,~gather,~scatter,~finish,8n+rest,state) == finish(8n+rest,state) : Owner},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(rest,0n),True{},Equal.sym(Bool,Nat.is_lt(rest,0n),False{},Division.lt_zero(rest)),small)))

def by_rounds(~Property: Nat -> Type,
  ~few: @+count: Nat -> {Nat.is_lt(count,8n) == True{} : Bool} -> Property(count),
  ~more: @+rest: Nat -> Property(rest) -> Property(8n+rest),count: Nat) -> Property(count):
  match count:
    case 0n: few(0n,{==})
    case 1n: few(1n,{==})
    case 2n: few(2n,{==})
    case 3n: few(3n,{==})
    case 4n: few(4n,{==})
    case 5n: few(5n,{==})
    case 6n: few(6n,{==})
    case 7n: few(7n,{==})
    case 1n+1n+1n+1n+1n+1n+1n+1n+ +rest:
      more(rest,by_rounds(~Property,~few,~more,rest))

# Every count: whole rounds, then the last positions one by one.
def rounds_property(~Context: Data,~map: Context -> F32 -> F32,count: Nat) -> Type:
  @+start: Nat -> @+slot: U32 -> @+context: Context -> @+values: Tree.Tree<F32> -> @+scratch: Tree.Tree<Tile.Tile8> ->
  @+depth: Nat -> @+capacity: U32 -> @+capacity_equal: {Tree.capacity(F32,values) == capacity : U32} -> @+balanced: Shape.Balanced(F32,depth,values) ->
  @+limit: {Nat.is_le(depth,30n) == True{} : Bool} -> @+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool} ->
  @+room: {Nat.is_le(Nat.add(start,count),U32.to_nat(capacity)) == True{} : Bool} ->
  {Lanes.rounds(~Context,~map,count,Cells.State{Tree.pack(F32,values),Tree.pack(Tile.Tile8,scratch),U32.from_nat(start),slot,context}) ==
    Tree.pack(F32,MapReadback.run(~F32,~Context,~map,count,U32.from_nat(start),context,values)) : Array<F32>}

def few_rounds(~Context: Data,~map: Context -> F32 -> F32,+count: Nat,small: {Nat.is_lt(count,8n) == True{} : Bool}) -> rounds_property(~Context,~map,count):
  match count:
    case 0n: start => slot => context => values => scratch => depth => capacity => capacity_equal => balanced => limit => capacity_ok => room =>
      few(~Context,~map,0n,U32.from_nat(start),slot,context,values,scratch)
    case 1n: start => slot => context => values => scratch => depth => capacity => capacity_equal => balanced => limit => capacity_ok => room =>
      few(~Context,~map,1n,U32.from_nat(start),slot,context,values,scratch)
    case 2n: start => slot => context => values => scratch => depth => capacity => capacity_equal => balanced => limit => capacity_ok => room =>
      few(~Context,~map,2n,U32.from_nat(start),slot,context,values,scratch)
    case 3n: start => slot => context => values => scratch => depth => capacity => capacity_equal => balanced => limit => capacity_ok => room =>
      few(~Context,~map,3n,U32.from_nat(start),slot,context,values,scratch)
    case 4n: start => slot => context => values => scratch => depth => capacity => capacity_equal => balanced => limit => capacity_ok => room =>
      few(~Context,~map,4n,U32.from_nat(start),slot,context,values,scratch)
    case 5n: start => slot => context => values => scratch => depth => capacity => capacity_equal => balanced => limit => capacity_ok => room =>
      few(~Context,~map,5n,U32.from_nat(start),slot,context,values,scratch)
    case 6n: start => slot => context => values => scratch => depth => capacity => capacity_equal => balanced => limit => capacity_ok => room =>
      few(~Context,~map,6n,U32.from_nat(start),slot,context,values,scratch)
    case 7n: start => slot => context => values => scratch => depth => capacity => capacity_equal => balanced => limit => capacity_ok => room =>
      few(~Context,~map,7n,U32.from_nat(start),slot,context,values,scratch)
    case 1n+1n+1n+1n+1n+1n+1n+1n+ +rest:
      start => slot => context => values => scratch => depth => capacity => capacity_equal => balanced => limit => capacity_ok => room =>
      Empty.absurd({Lanes.rounds(~Context,~map,8n+rest,Cells.State{Tree.pack(F32,values),Tree.pack(Tile.Tile8,scratch),U32.from_nat(start),slot,context}) ==
    Tree.pack(F32,MapReadback.run(~F32,~Context,~map,8n+rest,U32.from_nat(start),context,values)) : Array<F32>},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(rest,0n),True{},Equal.sym(Bool,Nat.is_lt(rest,0n),False{},Division.lt_zero(rest)),small)))

def more_rounds(~Context: Data,~map: Context -> F32 -> F32,+rest: Nat,previous: rounds_property(~Context,~map,rest)) -> rounds_property(~Context,~map,8n+rest):
  start => slot => context => values => scratch => depth => capacity => capacity_equal => balanced => limit => capacity_ok => room =>
  +position = U32.from_nat(start)
  +mapped = MapReadback.run(~F32,~Context,~map,8n,position,context,values)
  +next = Tree.write(Tile.Tile8,scratch,slot,cell(~Context,~map,values,position,context))
  +turn = U32.and((slot + 1 : U32),63)
  %Equal.sym(Cells.State<Array<F32>,Tile.Tile8,Context>,
    Cells.step(~Array<F32>,~Tile.Tile8,~Context,~Lanes.gather(~Context,~map),~Lanes.scatter,~8,Cells.State{Tree.pack(F32,values),Tree.pack(Tile.Tile8,scratch),position,slot,context}),
    Cells.State{Tree.pack(F32,mapped),Tree.pack(Tile.Tile8,next),(position + 8 : U32),turn,context},
    round(~Context,~map,values,scratch,start,slot,context,depth,capacity,capacity_equal,balanced,limit,capacity_ok,round_room(start,rest,U32.to_nat(capacity),room))) :
    {Lanes.rounds(~Context,~map,rest,_) == Tree.pack(F32,MapReadback.run(~F32,~Context,~map,8n+rest,position,context,values)) : Array<F32>}
  %Equal.sym(U32,(position + 8 : U32),U32.from_nat(8n+start),after_round(start)) :
    {Lanes.rounds(~Context,~map,rest,Cells.State{Tree.pack(F32,mapped),Tree.pack(Tile.Tile8,next),_,turn,context}) ==
      Tree.pack(F32,MapReadback.run(~F32,~Context,~map,8n+rest,position,context,values)) : Array<F32>}
  %Equal.sym(U32,stepped(8n,start),U32.from_nat(8n+start),advanced(8n,start)) :
    {Lanes.rounds(~Context,~map,rest,Cells.State{Tree.pack(F32,mapped),Tree.pack(Tile.Tile8,next),U32.from_nat(8n+start),turn,context}) ==
      Tree.pack(F32,MapReadback.run(~F32,~Context,~map,rest,_,context,mapped)) : Array<F32>}
  previous(8n+start,turn,context,mapped,next,depth,capacity,
    Equal.trans(U32,Tree.capacity(F32,mapped),Tree.capacity(F32,values),capacity,MapReadback.capacity(~F32,~Context,~map,8n,position,context,values),capacity_equal),
    MapReadback.balanced(~F32,~Context,~map,8n,position,context,values,depth,balanced),limit,capacity_ok,rest_room(start,rest,U32.to_nat(capacity),room))

def rounds(~Context: Data,~map: Context -> F32 -> F32,count: Nat,+start: Nat,+slot: U32,+context: Context,+values: Tree.Tree<F32>,+scratch: Tree.Tree<Tile.Tile8>,
  +depth: Nat,+capacity: U32,+capacity_equal: {Tree.capacity(F32,values) == capacity : U32},+balanced: Shape.Balanced(F32,depth,values),
  +limit: {Nat.is_le(depth,30n) == True{} : Bool},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(start,count),U32.to_nat(capacity)) == True{} : Bool}) ->
  {Lanes.rounds(~Context,~map,count,Cells.State{Tree.pack(F32,values),Tree.pack(Tile.Tile8,scratch),U32.from_nat(start),slot,context}) ==
    Tree.pack(F32,MapReadback.run(~F32,~Context,~map,count,U32.from_nat(start),context,values)) : Array<F32>}:
  by_rounds(~rounds_property(~Context,~map),~few_rounds(~Context,~map),~more_rounds(~Context,~map),count)(start,slot,context,values,scratch,depth,capacity,capacity_equal,balanced,limit,capacity_ok,room)

# The whole map from any position, with a fresh scratch array.
def map(~Context: Data,~map: Context -> F32 -> F32,+count: Nat,+start: Nat,+context: Context,+values: Tree.Tree<F32>,+depth: Nat,+balanced: Shape.Balanced(F32,depth,values),
  +limit: {Nat.is_le(depth,30n) == True{} : Bool},+room: {Nat.is_le(Nat.add(start,count),U32.to_nat(U32.shln(1,depth))) == True{} : Bool}) ->
  {Lanes.map(~Context,~map,count,U32.from_nat(start),context,Tree.pack(F32,values)) ==
    Traversal.map_in_place(~F32,~Context,~map,count,U32.from_nat(start),context,Tree.pack(F32,values)) : Array<F32>}:
  +position = U32.from_nat(start)
  %Equal.sym(Array<F32>,Traversal.map_in_place(~F32,~Context,~map,count,position,context,Tree.pack(F32,values)),Tree.pack(F32,MapReadback.run(~F32,~Context,~map,count,position,context,values)),
    scalar(~Context,~map,count,position,context,values)) :
    {Lanes.map(~Context,~map,count,position,context,Tree.pack(F32,values)) == _ : Array<F32>}
  %Equal.sym(Array<Tile.Tile8>,Array.new(Tile.Tile8,6n,Tile.filled(0.0)),Tree.pack(Tile.Tile8,Tree.fresh(Tile.Tile8,6n,Tile.filled(0.0))),StorageLaws.fresh(Tile.Tile8,6n,Tile.filled(0.0))) :
    {Lanes.rounds(~Context,~map,count,Cells.State{Tree.pack(F32,values),_,position,0,context}) == Tree.pack(F32,MapReadback.run(~F32,~Context,~map,count,position,context,values)) : Array<F32>}
  rounds(~Context,~map,count,start,0,context,values,Tree.fresh(Tile.Tile8,6n,Tile.filled(0.0)),depth,U32.shln(1,depth),Shape.capacity(F32,depth,values,balanced),balanced,limit,
    Power.native_power_bound(depth,Order.successor(depth,30n,limit)),room)

# A change after gathering is determined by one eight-value leaf equation.
def GatherRoundLaw(~Context: Data,~initial: Context -> F32 -> F32,~operation: Context -> F32 -> F32,~after: Context -> (Array<F32> & Tile.Tile8) -> Array<F32> & Tile.Tile8) -> Type:
  @+context: Context -> @+a: F32 -> @+b: F32 -> @+c: F32 -> @+d: F32 -> @+e: F32 -> @+f: F32 -> @+g: F32 -> @read: Array<F32> & F32 ->
  {after(context,Lanes.gathered(~Context,~initial,context,a,b,c,d,e,f,g,read)) == Lanes.gathered(~Context,~operation,context,a,b,c,d,e,f,g,read) : Array<F32> & Tile.Tile8}

def transform7(~Context: Data,~initial: Context -> F32 -> F32,~operation: Context -> F32 -> F32,~after: Context -> (Array<F32> & Tile.Tile8) -> Array<F32> & Tile.Tile8,~lawful: GatherRoundLaw(~Context,~initial,~operation,~after),+position: U32,+context: Context,+a: F32,+b: F32,+c: F32,+d: F32,+e: F32,+f: F32,read: Array<F32> & F32) ->
  {after(context,Lanes.gather7(~Context,~initial,position,context,a,b,c,d,e,f,read)) == Lanes.gather7(~Context,~operation,position,context,a,b,c,d,e,f,read) : Array<F32> & Tile.Tile8}:
  (values,g) = read
  lawful(context,a,b,c,d,e,f,g,Array.get(F32,values,(position + 1 : U32)))

def transform6(~Context: Data,~initial: Context -> F32 -> F32,~operation: Context -> F32 -> F32,~after: Context -> (Array<F32> & Tile.Tile8) -> Array<F32> & Tile.Tile8,~lawful: GatherRoundLaw(~Context,~initial,~operation,~after),+position: U32,+context: Context,+a: F32,+b: F32,+c: F32,+d: F32,+e: F32,read: Array<F32> & F32) ->
  {after(context,Lanes.gather6(~Context,~initial,position,context,a,b,c,d,e,read)) == Lanes.gather6(~Context,~operation,position,context,a,b,c,d,e,read) : Array<F32> & Tile.Tile8}:
  (values,f) = read
  transform7(~Context,~initial,~operation,~after,~lawful,(position + 1 : U32),context,a,b,c,d,e,f,Array.get(F32,values,(position + 1 : U32)))

def transform5(~Context: Data,~initial: Context -> F32 -> F32,~operation: Context -> F32 -> F32,~after: Context -> (Array<F32> & Tile.Tile8) -> Array<F32> & Tile.Tile8,~lawful: GatherRoundLaw(~Context,~initial,~operation,~after),+position: U32,+context: Context,+a: F32,+b: F32,+c: F32,+d: F32,read: Array<F32> & F32) ->
  {after(context,Lanes.gather5(~Context,~initial,position,context,a,b,c,d,read)) == Lanes.gather5(~Context,~operation,position,context,a,b,c,d,read) : Array<F32> & Tile.Tile8}:
  (values,e) = read
  transform6(~Context,~initial,~operation,~after,~lawful,(position + 1 : U32),context,a,b,c,d,e,Array.get(F32,values,(position + 1 : U32)))

def transform4(~Context: Data,~initial: Context -> F32 -> F32,~operation: Context -> F32 -> F32,~after: Context -> (Array<F32> & Tile.Tile8) -> Array<F32> & Tile.Tile8,~lawful: GatherRoundLaw(~Context,~initial,~operation,~after),+position: U32,+context: Context,+a: F32,+b: F32,+c: F32,read: Array<F32> & F32) ->
  {after(context,Lanes.gather4(~Context,~initial,position,context,a,b,c,read)) == Lanes.gather4(~Context,~operation,position,context,a,b,c,read) : Array<F32> & Tile.Tile8}:
  (values,d) = read
  transform5(~Context,~initial,~operation,~after,~lawful,(position + 1 : U32),context,a,b,c,d,Array.get(F32,values,(position + 1 : U32)))

def transform3(~Context: Data,~initial: Context -> F32 -> F32,~operation: Context -> F32 -> F32,~after: Context -> (Array<F32> & Tile.Tile8) -> Array<F32> & Tile.Tile8,~lawful: GatherRoundLaw(~Context,~initial,~operation,~after),+position: U32,+context: Context,+a: F32,+b: F32,read: Array<F32> & F32) ->
  {after(context,Lanes.gather3(~Context,~initial,position,context,a,b,read)) == Lanes.gather3(~Context,~operation,position,context,a,b,read) : Array<F32> & Tile.Tile8}:
  (values,c) = read
  transform4(~Context,~initial,~operation,~after,~lawful,(position + 1 : U32),context,a,b,c,Array.get(F32,values,(position + 1 : U32)))

def transform2(~Context: Data,~initial: Context -> F32 -> F32,~operation: Context -> F32 -> F32,~after: Context -> (Array<F32> & Tile.Tile8) -> Array<F32> & Tile.Tile8,~lawful: GatherRoundLaw(~Context,~initial,~operation,~after),+position: U32,+context: Context,+a: F32,read: Array<F32> & F32) ->
  {after(context,Lanes.gather2(~Context,~initial,position,context,a,read)) == Lanes.gather2(~Context,~operation,position,context,a,read) : Array<F32> & Tile.Tile8}:
  (values,b) = read
  transform3(~Context,~initial,~operation,~after,~lawful,(position + 1 : U32),context,a,b,Array.get(F32,values,(position + 1 : U32)))

def transform1(~Context: Data,~initial: Context -> F32 -> F32,~operation: Context -> F32 -> F32,~after: Context -> (Array<F32> & Tile.Tile8) -> Array<F32> & Tile.Tile8,~lawful: GatherRoundLaw(~Context,~initial,~operation,~after),+position: U32,+context: Context,read: Array<F32> & F32) ->
  {after(context,Lanes.gather1(~Context,~initial,position,context,read)) == Lanes.gather1(~Context,~operation,position,context,read) : Array<F32> & Tile.Tile8}:
  (values,a) = read
  transform2(~Context,~initial,~operation,~after,~lawful,(position + 1 : U32),context,a,Array.get(F32,values,(position + 1 : U32)))

def gather_transform(~Context: Data,~initial: Context -> F32 -> F32,~operation: Context -> F32 -> F32,~after: Context -> (Array<F32> & Tile.Tile8) -> Array<F32> & Tile.Tile8,~lawful: GatherRoundLaw(~Context,~initial,~operation,~after),values: Array<F32>,+position: U32,+context: Context) ->
  {after(context,Lanes.gather(~Context,~initial,values,position,context)) == Lanes.gather(~Context,~operation,values,position,context) : Array<F32> & Tile.Tile8}:
  transform1(~Context,~initial,~operation,~after,~lawful,position,context,Array.get(F32,values,position))

# A gather equal to the scalar gather may use any internal arithmetic stages.
def GatherLaw(~Context: Data,~operation: Context -> F32 -> F32,~collect: Array<F32> -> U32 -> Context -> Array<F32> & Tile.Tile8) -> Type:
  @values: Array<F32> -> @+position: U32 -> @+context: Context ->
  {collect(values,position,context) == Lanes.gather(~Context,~operation,values,position,context) : Array<F32> & Tile.Tile8}

def collected_values(~Context: Data,~operation: Context -> F32 -> F32,+values: Tree.Tree<F32>,+position: U32,+context: Context) -> Tree.Tree<F32>:
  Batch.batch(~F32,~Context,~ArrayLaws.unindexed(~F32,~Context,~operation),8n,position,context,values,values)

def collected_state(~Context: Data,~operation: Context -> F32 -> F32,+values: Tree.Tree<F32>,+scratch: Tree.Tree<Tile.Tile8>,+position: U32,+slot: U32,+context: Context) -> Cells.State<Array<F32>,Tile.Tile8,Context>:
  Cells.State{Tree.pack(F32,collected_values(~Context,~operation,values,position,context)),
    Tree.pack(Tile.Tile8,Tree.write(Tile.Tile8,scratch,slot,cell(~Context,~operation,values,position,context))),
    (position + 8 : U32),U32.and((slot + 1 : U32),63),context}

def collected_step(~Context: Data,~operation: Context -> F32 -> F32,~collect: Array<F32> -> U32 -> Context -> Array<F32> & Tile.Tile8,+values: Tree.Tree<F32>,+scratch: Tree.Tree<Tile.Tile8>,+position: U32,+slot: U32,+context: Context,
  gathered: {collect(Tree.pack(F32,values),position,context) == (Tree.pack(F32,values),cell(~Context,~operation,values,position,context)) : Array<F32> & Tile.Tile8}) ->
  {Cells.step(~Array<F32>,~Tile.Tile8,~Context,~collect,~Lanes.scatter,~8,Cells.State{Tree.pack(F32,values),Tree.pack(Tile.Tile8,scratch),position,slot,context}) == collected_state(~Context,~operation,values,scratch,position,slot,context) : Cells.State<Array<F32>,Tile.Tile8,Context>}:
  CellLaws.round(~F32,~Tile.Tile8,~Context,~collect,~Lanes.scatter,~8,values,scratch,position,slot,context,
    cell(~Context,~operation,values,position,context),collected_values(~Context,~operation,values,position,context),gathered,cell_scattered(~Context,~operation,values,position,context))

def WithProperty(~Context: Data,~operation: Context -> F32 -> F32,~collect: Array<F32> -> U32 -> Context -> Array<F32> & Tile.Tile8,count: Nat) -> Type:
  @+values: Tree.Tree<F32> -> @+scratch: Tree.Tree<Tile.Tile8> -> @+position: U32 -> @+slot: U32 -> @+context: Context ->
  {Lanes.rounds_with(~Array<F32>,~Context,~collect,~Lanes.scatter,~Lanes.finish(~Context,~operation),count,Cells.State{Tree.pack(F32,values),Tree.pack(Tile.Tile8,scratch),position,slot,context}) ==
    Lanes.rounds(~Context,~operation,count,Cells.State{Tree.pack(F32,values),Tree.pack(Tile.Tile8,scratch),position,slot,context}) : Array<F32>}

def with_few(~Context: Data,~operation: Context -> F32 -> F32,~collect: Array<F32> -> U32 -> Context -> Array<F32> & Tile.Tile8,+count: Nat,small: {Nat.is_lt(count,8n) == True{} : Bool}) -> WithProperty(~Context,~operation,~collect,count):
  +small = small
  values => scratch => position => slot => context =>
    Equal.trans(Array<F32>,
      Lanes.rounds_with(~Array<F32>,~Context,~collect,~Lanes.scatter,~Lanes.finish(~Context,~operation),count,Cells.State{Tree.pack(F32,values),Tree.pack(Tile.Tile8,scratch),position,slot,context}),
      Lanes.finish(~Context,~operation,count,Cells.State{Tree.pack(F32,values),Tree.pack(Tile.Tile8,scratch),position,slot,context}),
      Lanes.rounds(~Context,~operation,count,Cells.State{Tree.pack(F32,values),Tree.pack(Tile.Tile8,scratch),position,slot,context}),
      finish_small(~Array<F32>,~Context,~collect,~Lanes.scatter,~Lanes.finish(~Context,~operation),count,Cells.State{Tree.pack(F32,values),Tree.pack(Tile.Tile8,scratch),position,slot,context},small),
      Equal.sym(Array<F32>,Lanes.rounds(~Context,~operation,count,Cells.State{Tree.pack(F32,values),Tree.pack(Tile.Tile8,scratch),position,slot,context}),
        Lanes.finish(~Context,~operation,count,Cells.State{Tree.pack(F32,values),Tree.pack(Tile.Tile8,scratch),position,slot,context}),
        finish_small(~Array<F32>,~Context,~Lanes.gather(~Context,~operation),~Lanes.scatter,~Lanes.finish(~Context,~operation),count,Cells.State{Tree.pack(F32,values),Tree.pack(Tile.Tile8,scratch),position,slot,context},small)))

def with_more(~Context: Data,~operation: Context -> F32 -> F32,~collect: Array<F32> -> U32 -> Context -> Array<F32> & Tile.Tile8,~lawful: GatherLaw(~Context,~operation,~collect),+rest: Nat,remaining: WithProperty(~Context,~operation,~collect,rest)) -> WithProperty(~Context,~operation,~collect,8n+rest):
  values => scratch => position => slot => context =>
    %Equal.sym(Cells.State<Array<F32>,Tile.Tile8,Context>,
      Cells.step(~Array<F32>,~Tile.Tile8,~Context,~collect,~Lanes.scatter,~8,Cells.State{Tree.pack(F32,values),Tree.pack(Tile.Tile8,scratch),position,slot,context}),collected_state(~Context,~operation,values,scratch,position,slot,context),
      collected_step(~Context,~operation,~collect,values,scratch,position,slot,context,
        Equal.trans(Array<F32> & Tile.Tile8,collect(Tree.pack(F32,values),position,context),Lanes.gather(~Context,~operation,Tree.pack(F32,values),position,context),
          (Tree.pack(F32,values),cell(~Context,~operation,values,position,context)),lawful(Tree.pack(F32,values),position,context),gathered(~Context,~operation,values,position,context)))) :
      {Lanes.rounds_with(~Array<F32>,~Context,~collect,~Lanes.scatter,~Lanes.finish(~Context,~operation),rest,_) ==
        Lanes.rounds(~Context,~operation,rest,Cells.step(~Array<F32>,~Tile.Tile8,~Context,~Lanes.gather(~Context,~operation),~Lanes.scatter,~8,Cells.State{Tree.pack(F32,values),Tree.pack(Tile.Tile8,scratch),position,slot,context})) : Array<F32>}
    %Equal.sym(Cells.State<Array<F32>,Tile.Tile8,Context>,
      Cells.step(~Array<F32>,~Tile.Tile8,~Context,~Lanes.gather(~Context,~operation),~Lanes.scatter,~8,Cells.State{Tree.pack(F32,values),Tree.pack(Tile.Tile8,scratch),position,slot,context}),collected_state(~Context,~operation,values,scratch,position,slot,context),
      collected_step(~Context,~operation,~Lanes.gather(~Context,~operation),values,scratch,position,slot,context,gathered(~Context,~operation,values,position,context))) :
      {Lanes.rounds_with(~Array<F32>,~Context,~collect,~Lanes.scatter,~Lanes.finish(~Context,~operation),rest,collected_state(~Context,~operation,values,scratch,position,slot,context)) == Lanes.rounds(~Context,~operation,rest,_) : Array<F32>}
    remaining(collected_values(~Context,~operation,values,position,context),Tree.write(Tile.Tile8,scratch,slot,cell(~Context,~operation,values,position,context)),(position + 8 : U32),U32.and((slot + 1 : U32),63),context)

def with_rounds(~Context: Data,~operation: Context -> F32 -> F32,~collect: Array<F32> -> U32 -> Context -> Array<F32> & Tile.Tile8,~lawful: GatherLaw(~Context,~operation,~collect),count: Nat) -> WithProperty(~Context,~operation,~collect,count):
  by_rounds(~WithProperty(~Context,~operation,~collect),~with_few(~Context,~operation,~collect),~with_more(~Context,~operation,~collect,~lawful),count)

def with_packed(~Context: Data,~operation: Context -> F32 -> F32,~collect: Array<F32> -> U32 -> Context -> Array<F32> & Tile.Tile8,~lawful: GatherLaw(~Context,~operation,~collect),+count: Nat,+position: U32,+context: Context,+values: Tree.Tree<F32>) ->
  {Lanes.map_with(~Context,~operation,~collect,count,position,context,Tree.pack(F32,values)) == Lanes.map(~Context,~operation,count,position,context,Tree.pack(F32,values)) : Array<F32>}:
  %StorageLaws.fresh(Tile.Tile8,6n,Tile.filled(0.0)) :
    {Lanes.rounds_with(~Array<F32>,~Context,~collect,~Lanes.scatter,~Lanes.finish(~Context,~operation),count,Cells.State{Tree.pack(F32,values),_,position,0,context}) ==
      Lanes.rounds(~Context,~operation,count,Cells.State{Tree.pack(F32,values),_,position,0,context}) : Array<F32>}
  with_rounds(~Context,~operation,~collect,~lawful,count)(values,Tree.fresh(Tile.Tile8,6n,Tile.filled(0.0)),position,0,context)

def with_observed(~Context: Data,~operation: Context -> F32 -> F32,~collect: Array<F32> -> U32 -> Context -> Array<F32> & Tile.Tile8,~lawful: GatherLaw(~Context,~operation,~collect),+count: Nat,+position: U32,+context: Context,-values: Array<F32>,observation: StorageLaws.Observation<F32,values>) ->
  {Lanes.map_with(~Context,~operation,~collect,count,position,context,values) == Lanes.map(~Context,~operation,count,position,context,values) : Array<F32>}:
  StorageLaws.Observation{+tree,packed,reflected} = observation
  %packed : {Lanes.map_with(~Context,~operation,~collect,count,position,context,_) == Lanes.map(~Context,~operation,count,position,context,_) : Array<F32>}
  with_packed(~Context,~operation,~collect,~lawful,count,position,context,tree)

def map_with(~Context: Data,~operation: Context -> F32 -> F32,~collect: Array<F32> -> U32 -> Context -> Array<F32> & Tile.Tile8,~lawful: GatherLaw(~Context,~operation,~collect),+count: Nat,+position: U32,+context: Context,values: Array<F32>) ->
  {Lanes.map_with(~Context,~operation,~collect,count,position,context,values) == Lanes.map(~Context,~operation,count,position,context,values) : Array<F32>}:
  with_observed(~Context,~operation,~collect,~lawful,count,position,context,values,StorageLaws.observe(F32,values))

def map_with_when(~Context: Data,~operation: Context -> F32 -> F32,~collect: Array<F32> -> U32 -> Context -> Array<F32> & Tile.Tile8,~lawful: GatherLaw(~Context,~operation,~collect),+covered: Bool,+count: Nat,+position: U32,+context: Context,values: Array<F32>) ->
  {Lanes.map_with_when(~Context,~operation,~collect,covered,count,position,context,values) == Lanes.map_when(~Context,~operation,covered,count,position,context,values) : Array<F32>}:
  match covered:
    case False{}: {==}
    case True{}: map_with(~Context,~operation,~collect,~lawful,count,position,context,values)

def TileLaw(~Context: Data,~operation: Context -> F32 -> F32,~transform: Context -> Tile.Tile8 -> Tile.Tile8) -> Type:
  @+context: Context -> @+a: F32 -> @+b: F32 -> @+c: F32 -> @+d: F32 -> @+e: F32 -> @+f: F32 -> @+g: F32 -> @+h: F32 ->
  {transform(context,Tile.Tile{a,b,c,d,e,f,g,h}) == Tile.Tile{operation(context,a),operation(context,b),operation(context,c),operation(context,d),operation(context,e),operation(context,f),operation(context,g),operation(context,h)} : Tile.Tile8}

def transformed_round(~Context: Data,~operation: Context -> F32 -> F32,~transform: Context -> Tile.Tile8 -> Tile.Tile8,~lawful: TileLaw(~Context,~operation,~transform),+context: Context,+a: F32,+b: F32,+c: F32,+d: F32,+e: F32,+f: F32,+g: F32,read: Array<F32> & F32) ->
  {Lanes.transformed(~Context,~transform,context,Lanes.gathered(~Context,~Lanes.identity(~Context),context,a,b,c,d,e,f,g,read)) == Lanes.gathered(~Context,~operation,context,a,b,c,d,e,f,g,read) : Array<F32> & Tile.Tile8}:
  (values,h) = read
  Equal.cong(Tile.Tile8,Array<F32> & Tile.Tile8,tile => (values,tile),transform(context,Tile.Tile{a,b,c,d,e,f,g,h}),
    Tile.Tile{operation(context,a),operation(context,b),operation(context,c),operation(context,d),operation(context,e),operation(context,f),operation(context,g),operation(context,h)},lawful(context,a,b,c,d,e,f,g,h))

def gather_tiles(~Context: Data,~operation: Context -> F32 -> F32,~transform: Context -> Tile.Tile8 -> Tile.Tile8,~lawful: TileLaw(~Context,~operation,~transform),values: Array<F32>,+position: U32,+context: Context) ->
  {Lanes.gather_transformed(~Context,~transform,values,position,context) == Lanes.gather(~Context,~operation,values,position,context) : Array<F32> & Tile.Tile8}:
  gather_transform(~Context,~Lanes.identity(~Context),~operation,~Lanes.transformed(~Context,~transform),~transformed_round(~Context,~operation,~transform,~lawful),values,position,context)
