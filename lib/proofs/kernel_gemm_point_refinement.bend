# The k loop of a pointwise block over packed trees is a model: every step
# reads its pair of lane tiles and the four row cells from the trees and
# updates the accumulators as the native step does.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ../traversal_loop.bend as Loop
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws

# The operands of a pointwise product as trees.
type Sources is Data:
  Sources{cells: Tree.Tree<Tile.Tile8>,pairs: Tree.Tree<Tile.Tile16>}

# The k loop as the model sees it: the cursor and the accumulators.
type Reduced is Data:
  Reduced{cursor: Gemm.Points,accumulators: Gemm.Quad}

def represent(sources: Sources,state: Reduced) -> Gemm.PointLoop:
  Sources{cells,pairs} = sources
  Reduced{cursor,accumulators} = state
  Gemm.PointLoop{Tree.pack(Tile.Tile8,cells),Tree.pack(Tile.Tile16,pairs),cursor,accumulators}

# Lane `lane` of the four row cells from cell on, stride apart.
def rows_read(~lane: Tile.Tile8 -> F32,+cells: Tree.Tree<Tile.Tile8>,+cell: U32,+stride: U32) -> Gemm.Row4:
  Gemm.Row4{lane(Tree.read(Tile.Tile8,cells,cell)),lane(Tree.read(Tile.Tile8,cells,(cell + 1 * stride : U32))),lane(Tree.read(Tile.Tile8,cells,(cell + 2 * stride : U32))),
    lane(Tree.read(Tile.Tile8,cells,(cell + 3 * stride : U32)))}

# One step of the model: the pair at the cursor's address, the rows from the
# cell at picks, then the products.
def stepped(~at: U32 -> U32,~lane: Tile.Tile8 -> F32,~times: F32 -> F32 -> F32,+sources: Sources,state: Reduced) -> Reduced:
  Sources{+cells,+pairs} = sources
  Reduced{cursor,accumulators} = state
  Gemm.Points{+address,+cell,+stride} = cursor
  Reduced{Gemm.Points{(address + 1 : U32),cell,stride},Gemm.point_updated(~times,accumulators,Tree.read(Tile.Tile16,pairs,address),rows_read(~lane,cells,at(cell),stride))}

def step_matches(~at: U32 -> U32,~lane: Tile.Tile8 -> F32,~times: F32 -> F32 -> F32,+sources: Sources,+state: Reduced) ->
  {Gemm.point_step(~at,~lane,~times,represent(sources,state)) == represent(sources,stepped(~at,~lane,~times,sources,state)) : Gemm.PointLoop}:
  Sources{+cells,+pairs} = sources
  Reduced{+cursor,+accumulators} = state
  Gemm.Points{+address,+cell,+stride} = cursor
  -cell_array = Tree.pack(Tile.Tile8,cells)
  -pair_array = Tree.pack(Tile.Tile16,pairs)
  +first = at(cell)
  -next = {Gemm.Points{(address + 1 : U32),cell,stride} : Gemm.Points}
  -weight = Tree.read(Tile.Tile16,pairs,address)
  -x0 = lane(Tree.read(Tile.Tile8,cells,first))
  -x1 = lane(Tree.read(Tile.Tile8,cells,(first + 1 * stride : U32)))
  -x2 = lane(Tree.read(Tile.Tile8,cells,(first + 2 * stride : U32)))
  -result = {Gemm.PointLoop{cell_array,pair_array,next,Gemm.point_updated(~times,accumulators,weight,rows_read(~lane,cells,first,stride))} : Gemm.PointLoop}
  %Equal.sym(Array<Tile.Tile16> & Tile.Tile16,Array.get(Tile.Tile16,pair_array,address),(pair_array,weight),StorageLaws.read(Tile.Tile16,pairs,address)) :
    {Gemm.point_weight(~lane,~times,cell_array,first,stride,next,accumulators,_) == result : Gemm.PointLoop}
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,cell_array,first),(cell_array,Tree.read(Tile.Tile8,cells,first)),StorageLaws.read(Tile.Tile8,cells,first)) :
    {Gemm.point_products(~times,pair_array,next,accumulators,weight,Gemm.point_row_0(~lane,first,stride,_)) == result : Gemm.PointLoop}
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,cell_array,(first + 1 * stride : U32)),(cell_array,Tree.read(Tile.Tile8,cells,(first + 1 * stride : U32))),
    StorageLaws.read(Tile.Tile8,cells,(first + 1 * stride : U32))) :
    {Gemm.point_products(~times,pair_array,next,accumulators,weight,Gemm.point_row_1(~lane,first,stride,x0,_)) == result : Gemm.PointLoop}
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,cell_array,(first + 2 * stride : U32)),(cell_array,Tree.read(Tile.Tile8,cells,(first + 2 * stride : U32))),
    StorageLaws.read(Tile.Tile8,cells,(first + 2 * stride : U32))) :
    {Gemm.point_products(~times,pair_array,next,accumulators,weight,Gemm.point_row_2(~lane,first,stride,x0,x1,_)) == result : Gemm.PointLoop}
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,cell_array,(first + 3 * stride : U32)),(cell_array,Tree.read(Tile.Tile8,cells,(first + 3 * stride : U32))),
    StorageLaws.read(Tile.Tile8,cells,(first + 3 * stride : U32))) :
    {Gemm.point_products(~times,pair_array,next,accumulators,weight,Gemm.point_row_3(~lane,x0,x1,x2,_)) == result : Gemm.PointLoop}
  {==}

# A native state that represents a model state stays so after a step.
def step_matched(~at: U32 -> U32,~lane: Tile.Tile8 -> F32,~times: F32 -> F32 -> F32,+sources: Sources,-state: Gemm.PointLoop,+model: Reduced,
  equation: {state == represent(sources,model) : Gemm.PointLoop}) ->
  {Gemm.point_step(~at,~lane,~times,state) == represent(sources,stepped(~at,~lane,~times,sources,model)) : Gemm.PointLoop}:
  %Equal.sym(Gemm.PointLoop,state,represent(sources,model),equation) :
    {Gemm.point_step(~at,~lane,~times,_) == represent(sources,stepped(~at,~lane,~times,sources,model)) : Gemm.PointLoop}
  step_matches(~at,~lane,~times,sources,model)

def next_cells(state: Reduced) -> Reduced:
  Reduced{cursor,accumulators} = state
  Gemm.Points{address,cell,stride} = cursor
  Reduced{Gemm.Points{address,(cell + 2 : U32),stride},accumulators}

def next_matches(+sources: Sources,+model: Reduced) -> {Gemm.next_cells(represent(sources,model)) == represent(sources,next_cells(model)) : Gemm.PointLoop}:
  Sources{+cells,+pairs} = sources
  Reduced{+cursor,+accumulators} = model
  Gemm.Points{+address,+cell,+stride} = cursor
  {==}

def next_matched(+sources: Sources,-state: Gemm.PointLoop,+model: Reduced,equation: {state == represent(sources,model) : Gemm.PointLoop}) ->
  {Gemm.next_cells(state) == represent(sources,next_cells(model)) : Gemm.PointLoop}:
  %Equal.sym(Gemm.PointLoop,state,represent(sources,model),equation) : {Gemm.next_cells(_) == represent(sources,next_cells(model)) : Gemm.PointLoop}
  next_matches(sources,model)

# Sixteen steps over two cells of each row, then the next two cells.
def grouped(~times: F32 -> F32 -> F32,+sources: Sources,state: Reduced) -> Reduced:
  s = stepped(~Gemm.first_cell,~Tile.lane0,~times,sources,state)
  s = stepped(~Gemm.first_cell,~Tile.lane1,~times,sources,s)
  s = stepped(~Gemm.first_cell,~Tile.lane2,~times,sources,s)
  s = stepped(~Gemm.first_cell,~Tile.lane3,~times,sources,s)
  s = stepped(~Gemm.first_cell,~Tile.lane4,~times,sources,s)
  s = stepped(~Gemm.first_cell,~Tile.lane5,~times,sources,s)
  s = stepped(~Gemm.first_cell,~Tile.lane6,~times,sources,s)
  s = stepped(~Gemm.first_cell,~Tile.lane7,~times,sources,s)
  s = stepped(~Gemm.second_cell,~Tile.lane0,~times,sources,s)
  s = stepped(~Gemm.second_cell,~Tile.lane1,~times,sources,s)
  s = stepped(~Gemm.second_cell,~Tile.lane2,~times,sources,s)
  s = stepped(~Gemm.second_cell,~Tile.lane3,~times,sources,s)
  s = stepped(~Gemm.second_cell,~Tile.lane4,~times,sources,s)
  s = stepped(~Gemm.second_cell,~Tile.lane5,~times,sources,s)
  s = stepped(~Gemm.second_cell,~Tile.lane6,~times,sources,s)
  next_cells(stepped(~Gemm.second_cell,~Tile.lane7,~times,sources,s))

def group_matched(~times: F32 -> F32 -> F32,+sources: Sources,-state: Gemm.PointLoop,+model: Reduced,equation: {state == represent(sources,model) : Gemm.PointLoop}) ->
  {Gemm.point_group(~times,state) == represent(sources,grouped(~times,sources,model)) : Gemm.PointLoop}:
  -n0 = Gemm.point_step(~Gemm.first_cell,~Tile.lane0,~times,state)
  +m0 = stepped(~Gemm.first_cell,~Tile.lane0,~times,sources,model)
  equation = step_matched(~Gemm.first_cell,~Tile.lane0,~times,sources,state,model,equation)
  -n1 = Gemm.point_step(~Gemm.first_cell,~Tile.lane1,~times,n0)
  +m1 = stepped(~Gemm.first_cell,~Tile.lane1,~times,sources,m0)
  equation = step_matched(~Gemm.first_cell,~Tile.lane1,~times,sources,n0,m0,equation)
  -n2 = Gemm.point_step(~Gemm.first_cell,~Tile.lane2,~times,n1)
  +m2 = stepped(~Gemm.first_cell,~Tile.lane2,~times,sources,m1)
  equation = step_matched(~Gemm.first_cell,~Tile.lane2,~times,sources,n1,m1,equation)
  -n3 = Gemm.point_step(~Gemm.first_cell,~Tile.lane3,~times,n2)
  +m3 = stepped(~Gemm.first_cell,~Tile.lane3,~times,sources,m2)
  equation = step_matched(~Gemm.first_cell,~Tile.lane3,~times,sources,n2,m2,equation)
  -n4 = Gemm.point_step(~Gemm.first_cell,~Tile.lane4,~times,n3)
  +m4 = stepped(~Gemm.first_cell,~Tile.lane4,~times,sources,m3)
  equation = step_matched(~Gemm.first_cell,~Tile.lane4,~times,sources,n3,m3,equation)
  -n5 = Gemm.point_step(~Gemm.first_cell,~Tile.lane5,~times,n4)
  +m5 = stepped(~Gemm.first_cell,~Tile.lane5,~times,sources,m4)
  equation = step_matched(~Gemm.first_cell,~Tile.lane5,~times,sources,n4,m4,equation)
  -n6 = Gemm.point_step(~Gemm.first_cell,~Tile.lane6,~times,n5)
  +m6 = stepped(~Gemm.first_cell,~Tile.lane6,~times,sources,m5)
  equation = step_matched(~Gemm.first_cell,~Tile.lane6,~times,sources,n5,m5,equation)
  -n7 = Gemm.point_step(~Gemm.first_cell,~Tile.lane7,~times,n6)
  +m7 = stepped(~Gemm.first_cell,~Tile.lane7,~times,sources,m6)
  equation = step_matched(~Gemm.first_cell,~Tile.lane7,~times,sources,n6,m6,equation)
  -n8 = Gemm.point_step(~Gemm.second_cell,~Tile.lane0,~times,n7)
  +m8 = stepped(~Gemm.second_cell,~Tile.lane0,~times,sources,m7)
  equation = step_matched(~Gemm.second_cell,~Tile.lane0,~times,sources,n7,m7,equation)
  -n9 = Gemm.point_step(~Gemm.second_cell,~Tile.lane1,~times,n8)
  +m9 = stepped(~Gemm.second_cell,~Tile.lane1,~times,sources,m8)
  equation = step_matched(~Gemm.second_cell,~Tile.lane1,~times,sources,n8,m8,equation)
  -n10 = Gemm.point_step(~Gemm.second_cell,~Tile.lane2,~times,n9)
  +m10 = stepped(~Gemm.second_cell,~Tile.lane2,~times,sources,m9)
  equation = step_matched(~Gemm.second_cell,~Tile.lane2,~times,sources,n9,m9,equation)
  -n11 = Gemm.point_step(~Gemm.second_cell,~Tile.lane3,~times,n10)
  +m11 = stepped(~Gemm.second_cell,~Tile.lane3,~times,sources,m10)
  equation = step_matched(~Gemm.second_cell,~Tile.lane3,~times,sources,n10,m10,equation)
  -n12 = Gemm.point_step(~Gemm.second_cell,~Tile.lane4,~times,n11)
  +m12 = stepped(~Gemm.second_cell,~Tile.lane4,~times,sources,m11)
  equation = step_matched(~Gemm.second_cell,~Tile.lane4,~times,sources,n11,m11,equation)
  -n13 = Gemm.point_step(~Gemm.second_cell,~Tile.lane5,~times,n12)
  +m13 = stepped(~Gemm.second_cell,~Tile.lane5,~times,sources,m12)
  equation = step_matched(~Gemm.second_cell,~Tile.lane5,~times,sources,n12,m12,equation)
  -n14 = Gemm.point_step(~Gemm.second_cell,~Tile.lane6,~times,n13)
  +m14 = stepped(~Gemm.second_cell,~Tile.lane6,~times,sources,m13)
  equation = step_matched(~Gemm.second_cell,~Tile.lane6,~times,sources,n13,m13,equation)
  -n15 = Gemm.point_step(~Gemm.second_cell,~Tile.lane7,~times,n14)
  +m15 = stepped(~Gemm.second_cell,~Tile.lane7,~times,sources,m14)
  equation = step_matched(~Gemm.second_cell,~Tile.lane7,~times,sources,n14,m14,equation)
  next_matched(sources,n15,m15,equation)

# The model of count groups.
def groups(~times: F32 -> F32 -> F32,count: Nat,+sources: Sources,state: Reduced) -> Reduced:
  match count:
    case 0n: state
    case 1n+rest: groups(~times,rest,sources,grouped(~times,sources,state))

def groups_matched(~times: F32 -> F32 -> F32,count: Nat,+sources: Sources,-state: Gemm.PointLoop,+model: Reduced,equation: {state == represent(sources,model) : Gemm.PointLoop}) ->
  {Loop.run(~Gemm.PointLoop,~Gemm.point_group(~times),count,state) == represent(sources,groups(~times,count,sources,model)) : Gemm.PointLoop}:
  match count:
    case 0n: equation
    case 1n+ +rest:
      groups_matched(~times,rest,sources,Gemm.point_group(~times,state),grouped(~times,sources,model),group_matched(~times,sources,state,model,equation))
def tail_step(~at: U32 -> U32,lane: Nat,+sources: Sources,state: Reduced) -> Reduced:
  match lane:
    case 0n: stepped(~at,~Tile.lane0,~Gemm.tail_times,sources,state)
    case 1n: stepped(~at,~Tile.lane1,~Gemm.tail_times,sources,state)
    case 2n: stepped(~at,~Tile.lane2,~Gemm.tail_times,sources,state)
    case 3n: stepped(~at,~Tile.lane3,~Gemm.tail_times,sources,state)
    case 4n: stepped(~at,~Tile.lane4,~Gemm.tail_times,sources,state)
    case 5n: stepped(~at,~Tile.lane5,~Gemm.tail_times,sources,state)
    case 6n: stepped(~at,~Tile.lane6,~Gemm.tail_times,sources,state)
    case 7n+rest: stepped(~at,~Tile.lane7,~Gemm.tail_times,sources,state)

def tail_step_matches(~at: U32 -> U32,lane: Nat,+sources: Sources,+state: Reduced) ->
  {Gemm.tail_step(~at,lane,represent(sources,state)) == represent(sources,tail_step(~at,lane,sources,state)) : Gemm.PointLoop}:
  match lane:
    case 0n: step_matches(~at,~Tile.lane0,~Gemm.tail_times,sources,state)
    case 1n: step_matches(~at,~Tile.lane1,~Gemm.tail_times,sources,state)
    case 2n: step_matches(~at,~Tile.lane2,~Gemm.tail_times,sources,state)
    case 3n: step_matches(~at,~Tile.lane3,~Gemm.tail_times,sources,state)
    case 4n: step_matches(~at,~Tile.lane4,~Gemm.tail_times,sources,state)
    case 5n: step_matches(~at,~Tile.lane5,~Gemm.tail_times,sources,state)
    case 6n: step_matches(~at,~Tile.lane6,~Gemm.tail_times,sources,state)
    case 7n+rest: step_matches(~at,~Tile.lane7,~Gemm.tail_times,sources,state)

def tail_cell(~at: U32 -> U32,count: Nat,+lane: U32,+sources: Sources,state: Reduced) -> Reduced:
  match count:
    case 0n: state
    case 1n+rest: tail_cell(~at,rest,(lane + 1 : U32),sources,tail_step(~at,U32.to_nat(lane),sources,state))

def tail_cell_matches(~at: U32 -> U32,count: Nat,+lane: U32,+sources: Sources,+state: Reduced) ->
  {Gemm.tail_cell(~at,count,lane,represent(sources,state)) == represent(sources,tail_cell(~at,count,lane,sources,state)) : Gemm.PointLoop}:
  match count:
    case 0n: {==}
    case 1n+rest:
      +next = (lane + 1 : U32)
      +model = tail_step(~at,U32.to_nat(lane),sources,state)
      %Equal.sym(Gemm.PointLoop,Gemm.tail_step(~at,U32.to_nat(lane),represent(sources,state)),represent(sources,model),tail_step_matches(~at,U32.to_nat(lane),sources,state)) :
        {Gemm.tail_cell(~at,rest,next,_) == represent(sources,tail_cell(~at,rest,next,sources,model)) : Gemm.PointLoop}
      tail_cell_matches(~at,rest,next,sources,model)

def tail_reduced(+matrix: Gemm.PointMatrix,+sources: Sources,state: Reduced) -> Reduced:
  +tail = Gemm.point_remainder(matrix)
  +lower = U32.min(tail,8)
  tail_cell(~Gemm.second_cell,U32.to_nat((tail - lower : U32)),0,sources,tail_cell(~Gemm.first_cell,U32.to_nat(lower),0,sources,state))

def tail_matches(+matrix: Gemm.PointMatrix,+sources: Sources,+state: Reduced) ->
  {Gemm.finish_inner(matrix,represent(sources,state)) == represent(sources,tail_reduced(matrix,sources,state)) : Gemm.PointLoop}:
  +tail = Gemm.point_remainder(matrix)
  +lower = U32.to_nat(U32.min(tail,8))
  +upper = U32.to_nat((tail - U32.min(tail,8) : U32))
  +model = tail_cell(~Gemm.first_cell,lower,0,sources,state)
  %Equal.sym(Gemm.PointLoop,Gemm.tail_cell(~Gemm.first_cell,lower,0,represent(sources,state)),represent(sources,model),tail_cell_matches(~Gemm.first_cell,lower,0,sources,state)) :
    {Gemm.tail_cell(~Gemm.second_cell,upper,0,_) == represent(sources,tail_cell(~Gemm.second_cell,upper,0,sources,model)) : Gemm.PointLoop}
  tail_cell_matches(~Gemm.second_cell,upper,0,sources,model)
