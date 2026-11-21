# Each cell of a pointwise block is the ordered dot of its taps: step k reads
# the pair of lane tiles at its address and lane `lane` of the row cells from
# its cell on, and multiplies the row value by the lane value.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ../ordered_matrix.bend as OrderedMatrix
import ../traversal_loop.bend as Loop
import ./storage_tree_model.bend as Tree
import ./kernel_tile_refinement.bend as TileLaws
import ./kernel_gemm_point_refinement.bend as PointLaws
import ./ordered_reduction_split.bend as Ordered

# ---- The cells of the accumulators ----

def row_of(row: Nat,accumulators: Gemm.Quad) -> Tile.Tile16:
  match row:
    case 0n: Gemm.quad0(accumulators)
    case 1n+a:
      match a:
        case 0n: Gemm.quad1(accumulators)
        case 1n+b:
          match b:
            case 0n: Gemm.quad2(accumulators)
            case 1n+c: Gemm.quad3(accumulators)

# Lane `lane` of a pair: of the lower tile below eight, of the upper one from
# eight on.
def half_lane(low: Bool,lane: Nat,lower: Tile.Tile8,upper: Tile.Tile8) -> F32:
  match low:
    case True{}: TileLaws.select_lane(lane,lower)
    case False{}: TileLaws.select_lane(Nat.sub(lane,8n),upper)

def pair_lane(+lane: Nat,values: Tile.Tile16) -> F32:
  Tile.Tiles{lower,upper} = values
  half_lane(Nat.is_lt(lane,8n),lane,lower,upper)

def cell_value(row: Nat,lane: Nat,accumulators: Gemm.Quad) -> F32:
  pair_lane(lane,row_of(row,accumulators))

def row_value(values: Gemm.Row4,row: Nat) -> F32:
  Gemm.Row4{x0,x1,x2,x3} = values
  match row:
    case 0n: x0
    case 1n+a:
      match a:
        case 0n: x1
        case 1n+b:
          match b:
            case 0n: x2
            case 1n+c: x3

def half_updated(+low: Bool,+lane: Nat,+lower: Tile.Tile8,+upper: Tile.Tile8,+weight_lower: Tile.Tile8,+weight_upper: Tile.Tile8,+value: F32) ->
  {half_lane(low,lane,Tile.update(~Gemm.row_times,lower,weight_lower,value),Tile.update(~Gemm.row_times,upper,weight_upper,value)) ==
    (half_lane(low,lane,lower,upper) + (value * half_lane(low,lane,weight_lower,weight_upper) : F32) : F32) : F32}:
  match low:
    case True{}: TileLaws.updated_by_value(~Gemm.row_times,lower,weight_lower,value,lane)
    case False{}: TileLaws.updated_by_value(~Gemm.row_times,upper,weight_upper,value,Nat.sub(lane,8n))

def pair_updated(+accumulator: Tile.Tile16,+weight: Tile.Tile16,+value: F32,+lane: Nat) ->
  {pair_lane(lane,Tile.update_pair(~Gemm.row_times,accumulator,weight,value)) == (pair_lane(lane,accumulator) + (value * pair_lane(lane,weight) : F32) : F32) : F32}:
  Tile.Tiles{+lower,+upper} = accumulator
  Tile.Tiles{+low,+high} = weight
  half_updated(Nat.is_lt(lane,8n),lane,lower,upper,low,high,value)

# A cell of a step is the cell before plus the row value times the lane value.
def updated_cell(+accumulators: Gemm.Quad,+weight: Tile.Tile16,+values: Gemm.Row4,+row: Nat,+lane: Nat) ->
  {cell_value(row,lane,Gemm.point_updated(~Gemm.row_times,accumulators,weight,values)) == (cell_value(row,lane,accumulators) + (row_value(values,row) * pair_lane(lane,weight) : F32) : F32) : F32}:
  Gemm.Quad{+a0,+a1,+a2,+a3} = accumulators
  Gemm.Row4{+x0,+x1,+x2,+x3} = values
  match row:
    case 0n: pair_updated(a0,weight,x0,lane)
    case 1n+ +a:
      match a:
        case 0n: pair_updated(a1,weight,x1,lane)
        case 1n+ +b:
          match b:
            case 0n: pair_updated(a2,weight,x2,lane)
            case 1n+c: pair_updated(a3,weight,x3,lane)

def zero_half(+low: Bool,+lane: Nat) -> {half_lane(low,lane,Tile.filled(0.0),Tile.filled(0.0)) == 0.0 : F32}:
  match low:
    case True{}: TileLaws.filled_lane(lane,0.0)
    case False{}: TileLaws.filled_lane(Nat.sub(lane,8n),0.0)

# The zero accumulators hold +0 in every cell.
def zero_cell(+row: Nat,+lane: Nat) -> {cell_value(row,lane,Gemm.quad_zero()) == 0.0 : F32}:
  match row:
    case 0n: zero_half(Nat.is_lt(lane,8n),lane)
    case 1n+ +a:
      match a:
        case 0n: zero_half(Nat.is_lt(lane,8n),lane)
        case 1n+ +b:
          match b:
            case 0n: zero_half(Nat.is_lt(lane,8n),lane)
            case 1n+c: zero_half(Nat.is_lt(lane,8n),lane)

# ---- The taps ----

# A step reads the pair at address and lane `lane` of the row cells from cell
# on.
type Tap is Data:
  Tap{address: U32,cell: U32,lane: Nat}

# The trees and the cells between two rows.
type Sampled is Data:
  Sampled{cells: Tree.Tree<Tile.Tile8>,pairs: Tree.Tree<Tile.Tile16>,stride: U32}

def rows_selected(+lane: Nat,+cells: Tree.Tree<Tile.Tile8>,+cell: U32,+stride: U32) -> Gemm.Row4:
  Gemm.Row4{TileLaws.select_lane(lane,Tree.read(Tile.Tile8,cells,cell)),TileLaws.select_lane(lane,Tree.read(Tile.Tile8,cells,(cell + 1 * stride : U32))),
    TileLaws.select_lane(lane,Tree.read(Tile.Tile8,cells,(cell + 2 * stride : U32))),TileLaws.select_lane(lane,Tree.read(Tile.Tile8,cells,(cell + 3 * stride : U32)))}

def input_sample(sampled: Sampled,row: Nat,tap: Tap) -> F32:
  Sampled{+cells,pairs,+stride} = sampled
  Tap{address,+cell,+lane} = tap
  row_value(rows_selected(lane,cells,cell,stride),row)

def weight_sample(sampled: Sampled,lane: Nat,tap: Tap) -> F32:
  Sampled{cells,+pairs,stride} = sampled
  Tap{+address,cell,tap_lane} = tap
  pair_lane(lane,Tree.read(Tile.Tile16,pairs,address))

# The ordered dot of a cell: the row value of row times the lane value of
# lane, over the taps.
def dot(taps: List<&2,Tap>,+row: Nat,+lane: Nat,+sampled: Sampled,initial: F32) -> F32:
  OrderedMatrix.dot(~Nat,~Tap,~Nat,~Sampled,~input_sample,~weight_sample,taps,lane,row,sampled,initial)

def accumulators_of(state: PointLaws.Reduced) -> Gemm.Quad:
  PointLaws.Reduced{cursor,accumulators} = state
  accumulators

def cursor_of(state: PointLaws.Reduced) -> Gemm.Points:
  PointLaws.Reduced{cursor,accumulators} = state
  cursor

def stride_of(state: PointLaws.Reduced) -> U32:
  PointLaws.Reduced{cursor,accumulators} = state
  Gemm.Points{address,cell,stride} = cursor
  stride

def sampled_of(sources: PointLaws.Sources,state: PointLaws.Reduced) -> Sampled:
  PointLaws.Sources{cells,pairs} = sources
  Sampled{cells,pairs,stride_of(state)}

# The tap of a step from a cursor.
def tap_at(~at: U32 -> U32,lane: Nat,cursor: Gemm.Points) -> Tap:
  Gemm.Points{+address,+cell,stride} = cursor
  Tap{address,at(cell),lane}

def advanced(cursor: Gemm.Points) -> Gemm.Points:
  Gemm.Points{+address,cell,stride} = cursor
  Gemm.Points{(address + 1 : U32),cell,stride}

def cells_after(cursor: Gemm.Points) -> Gemm.Points:
  Gemm.Points{address,cell,stride} = cursor
  Gemm.Points{address,(cell + 2 : U32),stride}

# The sixteen taps of a group and the cursor after it.
def group_taps(cursor: Gemm.Points) -> List<&2,Tap>:
  +c0 = cursor
  +c1 = advanced(c0)
  +c2 = advanced(c1)
  +c3 = advanced(c2)
  +c4 = advanced(c3)
  +c5 = advanced(c4)
  +c6 = advanced(c5)
  +c7 = advanced(c6)
  +c8 = advanced(c7)
  +c9 = advanced(c8)
  +c10 = advanced(c9)
  +c11 = advanced(c10)
  +c12 = advanced(c11)
  +c13 = advanced(c12)
  +c14 = advanced(c13)
  +c15 = advanced(c14)
  [tap_at(~Gemm.first_cell,0n,c0),tap_at(~Gemm.first_cell,1n,c1),tap_at(~Gemm.first_cell,2n,c2),tap_at(~Gemm.first_cell,3n,c3),
    tap_at(~Gemm.first_cell,4n,c4),tap_at(~Gemm.first_cell,5n,c5),tap_at(~Gemm.first_cell,6n,c6),tap_at(~Gemm.first_cell,7n,c7),
    tap_at(~Gemm.second_cell,0n,c8),tap_at(~Gemm.second_cell,1n,c9),tap_at(~Gemm.second_cell,2n,c10),tap_at(~Gemm.second_cell,3n,c11),
    tap_at(~Gemm.second_cell,4n,c12),tap_at(~Gemm.second_cell,5n,c13),tap_at(~Gemm.second_cell,6n,c14),tap_at(~Gemm.second_cell,7n,c15)]

def group_after(cursor: Gemm.Points) -> Gemm.Points:
  c = advanced(cursor)
  c = advanced(c)
  c = advanced(c)
  c = advanced(c)
  c = advanced(c)
  c = advanced(c)
  c = advanced(c)
  c = advanced(c)
  c = advanced(c)
  c = advanced(c)
  c = advanced(c)
  c = advanced(c)
  c = advanced(c)
  c = advanced(c)
  c = advanced(c)
  c = advanced(c)
  cells_after(c)

# The taps of count groups from a cursor.
def taps(count: Nat,+cursor: Gemm.Points) -> List<&2,Tap>:
  match count:
    case 0n: []
    case 1n+rest: OrderedMatrix.cat(Tap,group_taps(cursor),taps(rest,group_after(cursor)))

# ---- The k loop ----

# A step adds the product of its tap to every cell.
def step_cell(~at: U32 -> U32,~lane: Tile.Tile8 -> F32,+index: Nat,+cells: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+address: U32,+cell: U32,+stride: U32,
  +accumulators: Gemm.Quad,+row: Nat,+channel: Nat,same: {PointLaws.rows_read(~lane,cells,at(cell),stride) == rows_selected(index,cells,at(cell),stride) : Gemm.Row4}) ->
  {cell_value(row,channel,accumulators_of(PointLaws.stepped(~at,~lane,~Gemm.row_times,PointLaws.Sources{cells,pairs},PointLaws.Reduced{Gemm.Points{address,cell,stride},accumulators}))) ==
    (cell_value(row,channel,accumulators) + (input_sample(Sampled{cells,pairs,stride},row,Tap{address,at(cell),index}) * weight_sample(Sampled{cells,pairs,stride},channel,Tap{address,at(cell),index}) : F32) : F32) : F32}:
  +weight = Tree.read(Tile.Tile16,pairs,address)
  %same : {cell_value(row,channel,Gemm.point_updated(~Gemm.row_times,accumulators,weight,PointLaws.rows_read(~lane,cells,at(cell),stride))) ==
    (cell_value(row,channel,accumulators) + (row_value(_,row) * pair_lane(channel,weight) : F32) : F32) : F32}
  updated_cell(accumulators,weight,PointLaws.rows_read(~lane,cells,at(cell),stride),row,channel)

# A group adds the products of its sixteen taps in order.
def group_cell(+cells: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+address: U32,+cell: U32,+stride: U32,+accumulators: Gemm.Quad,+row: Nat,+channel: Nat) ->
  {cell_value(row,channel,accumulators_of(PointLaws.grouped(~Gemm.row_times,PointLaws.Sources{cells,pairs},PointLaws.Reduced{Gemm.Points{address,cell,stride},accumulators}))) ==
    dot(group_taps(Gemm.Points{address,cell,stride}),row,channel,Sampled{cells,pairs,stride},cell_value(row,channel,accumulators)) : F32}:
  +sources = {PointLaws.Sources{cells,pairs} : PointLaws.Sources}
  +sampled = {Sampled{cells,pairs,stride} : Sampled}
  +low = Gemm.first_cell(cell)
  +high = Gemm.second_cell(cell)
  +m0 = {PointLaws.Reduced{Gemm.Points{address,cell,stride},accumulators} : PointLaws.Reduced}
  +m1 = PointLaws.stepped(~Gemm.first_cell,~Tile.lane0,~Gemm.row_times,sources,m0)
  +m2 = PointLaws.stepped(~Gemm.first_cell,~Tile.lane1,~Gemm.row_times,sources,m1)
  +m3 = PointLaws.stepped(~Gemm.first_cell,~Tile.lane2,~Gemm.row_times,sources,m2)
  +m4 = PointLaws.stepped(~Gemm.first_cell,~Tile.lane3,~Gemm.row_times,sources,m3)
  +m5 = PointLaws.stepped(~Gemm.first_cell,~Tile.lane4,~Gemm.row_times,sources,m4)
  +m6 = PointLaws.stepped(~Gemm.first_cell,~Tile.lane5,~Gemm.row_times,sources,m5)
  +m7 = PointLaws.stepped(~Gemm.first_cell,~Tile.lane6,~Gemm.row_times,sources,m6)
  +m8 = PointLaws.stepped(~Gemm.first_cell,~Tile.lane7,~Gemm.row_times,sources,m7)
  +m9 = PointLaws.stepped(~Gemm.second_cell,~Tile.lane0,~Gemm.row_times,sources,m8)
  +m10 = PointLaws.stepped(~Gemm.second_cell,~Tile.lane1,~Gemm.row_times,sources,m9)
  +m11 = PointLaws.stepped(~Gemm.second_cell,~Tile.lane2,~Gemm.row_times,sources,m10)
  +m12 = PointLaws.stepped(~Gemm.second_cell,~Tile.lane3,~Gemm.row_times,sources,m11)
  +m13 = PointLaws.stepped(~Gemm.second_cell,~Tile.lane4,~Gemm.row_times,sources,m12)
  +m14 = PointLaws.stepped(~Gemm.second_cell,~Tile.lane5,~Gemm.row_times,sources,m13)
  +m15 = PointLaws.stepped(~Gemm.second_cell,~Tile.lane6,~Gemm.row_times,sources,m14)
  +m16 = PointLaws.stepped(~Gemm.second_cell,~Tile.lane7,~Gemm.row_times,sources,m15)
  +a1 = (address + 1 : U32)
  +a2 = (a1 + 1 : U32)
  +a3 = (a2 + 1 : U32)
  +a4 = (a3 + 1 : U32)
  +a5 = (a4 + 1 : U32)
  +a6 = (a5 + 1 : U32)
  +a7 = (a6 + 1 : U32)
  +a8 = (a7 + 1 : U32)
  +a9 = (a8 + 1 : U32)
  +a10 = (a9 + 1 : U32)
  +a11 = (a10 + 1 : U32)
  +a12 = (a11 + 1 : U32)
  +a13 = (a12 + 1 : U32)
  +a14 = (a13 + 1 : U32)
  +a15 = (a14 + 1 : U32)
  -v0 = cell_value(row,channel,accumulators)
  -v1 = cell_value(row,channel,accumulators_of(m1))
  -v2 = cell_value(row,channel,accumulators_of(m2))
  -v3 = cell_value(row,channel,accumulators_of(m3))
  -v4 = cell_value(row,channel,accumulators_of(m4))
  -v5 = cell_value(row,channel,accumulators_of(m5))
  -v6 = cell_value(row,channel,accumulators_of(m6))
  -v7 = cell_value(row,channel,accumulators_of(m7))
  -v8 = cell_value(row,channel,accumulators_of(m8))
  -v9 = cell_value(row,channel,accumulators_of(m9))
  -v10 = cell_value(row,channel,accumulators_of(m10))
  -v11 = cell_value(row,channel,accumulators_of(m11))
  -v12 = cell_value(row,channel,accumulators_of(m12))
  -v13 = cell_value(row,channel,accumulators_of(m13))
  -v14 = cell_value(row,channel,accumulators_of(m14))
  -v15 = cell_value(row,channel,accumulators_of(m15))
  -v16 = cell_value(row,channel,accumulators_of(m16))
  -p0 = (input_sample(sampled,row,Tap{address,low,0n}) * weight_sample(sampled,channel,Tap{address,low,0n}) : F32)
  -p1 = (input_sample(sampled,row,Tap{a1,low,1n}) * weight_sample(sampled,channel,Tap{a1,low,1n}) : F32)
  -p2 = (input_sample(sampled,row,Tap{a2,low,2n}) * weight_sample(sampled,channel,Tap{a2,low,2n}) : F32)
  -p3 = (input_sample(sampled,row,Tap{a3,low,3n}) * weight_sample(sampled,channel,Tap{a3,low,3n}) : F32)
  -p4 = (input_sample(sampled,row,Tap{a4,low,4n}) * weight_sample(sampled,channel,Tap{a4,low,4n}) : F32)
  -p5 = (input_sample(sampled,row,Tap{a5,low,5n}) * weight_sample(sampled,channel,Tap{a5,low,5n}) : F32)
  -p6 = (input_sample(sampled,row,Tap{a6,low,6n}) * weight_sample(sampled,channel,Tap{a6,low,6n}) : F32)
  -p7 = (input_sample(sampled,row,Tap{a7,low,7n}) * weight_sample(sampled,channel,Tap{a7,low,7n}) : F32)
  -p8 = (input_sample(sampled,row,Tap{a8,high,0n}) * weight_sample(sampled,channel,Tap{a8,high,0n}) : F32)
  -p9 = (input_sample(sampled,row,Tap{a9,high,1n}) * weight_sample(sampled,channel,Tap{a9,high,1n}) : F32)
  -p10 = (input_sample(sampled,row,Tap{a10,high,2n}) * weight_sample(sampled,channel,Tap{a10,high,2n}) : F32)
  -p11 = (input_sample(sampled,row,Tap{a11,high,3n}) * weight_sample(sampled,channel,Tap{a11,high,3n}) : F32)
  -p12 = (input_sample(sampled,row,Tap{a12,high,4n}) * weight_sample(sampled,channel,Tap{a12,high,4n}) : F32)
  -p13 = (input_sample(sampled,row,Tap{a13,high,5n}) * weight_sample(sampled,channel,Tap{a13,high,5n}) : F32)
  -p14 = (input_sample(sampled,row,Tap{a14,high,6n}) * weight_sample(sampled,channel,Tap{a14,high,6n}) : F32)
  -p15 = (input_sample(sampled,row,Tap{a15,high,7n}) * weight_sample(sampled,channel,Tap{a15,high,7n}) : F32)
  -wanted = dot(group_taps(Gemm.Points{address,cell,stride}),row,channel,sampled,v0)
  e0 = step_cell(~Gemm.first_cell,~Tile.lane0,0n,cells,pairs,address,cell,stride,accumulators,row,channel,{==})
  e1 = step_cell(~Gemm.first_cell,~Tile.lane1,1n,cells,pairs,a1,cell,stride,accumulators_of(m1),row,channel,{==})
  e2 = step_cell(~Gemm.first_cell,~Tile.lane2,2n,cells,pairs,a2,cell,stride,accumulators_of(m2),row,channel,{==})
  e3 = step_cell(~Gemm.first_cell,~Tile.lane3,3n,cells,pairs,a3,cell,stride,accumulators_of(m3),row,channel,{==})
  e4 = step_cell(~Gemm.first_cell,~Tile.lane4,4n,cells,pairs,a4,cell,stride,accumulators_of(m4),row,channel,{==})
  e5 = step_cell(~Gemm.first_cell,~Tile.lane5,5n,cells,pairs,a5,cell,stride,accumulators_of(m5),row,channel,{==})
  e6 = step_cell(~Gemm.first_cell,~Tile.lane6,6n,cells,pairs,a6,cell,stride,accumulators_of(m6),row,channel,{==})
  e7 = step_cell(~Gemm.first_cell,~Tile.lane7,7n,cells,pairs,a7,cell,stride,accumulators_of(m7),row,channel,{==})
  e8 = step_cell(~Gemm.second_cell,~Tile.lane0,0n,cells,pairs,a8,cell,stride,accumulators_of(m8),row,channel,{==})
  e9 = step_cell(~Gemm.second_cell,~Tile.lane1,1n,cells,pairs,a9,cell,stride,accumulators_of(m9),row,channel,{==})
  e10 = step_cell(~Gemm.second_cell,~Tile.lane2,2n,cells,pairs,a10,cell,stride,accumulators_of(m10),row,channel,{==})
  e11 = step_cell(~Gemm.second_cell,~Tile.lane3,3n,cells,pairs,a11,cell,stride,accumulators_of(m11),row,channel,{==})
  e12 = step_cell(~Gemm.second_cell,~Tile.lane4,4n,cells,pairs,a12,cell,stride,accumulators_of(m12),row,channel,{==})
  e13 = step_cell(~Gemm.second_cell,~Tile.lane5,5n,cells,pairs,a13,cell,stride,accumulators_of(m13),row,channel,{==})
  e14 = step_cell(~Gemm.second_cell,~Tile.lane6,6n,cells,pairs,a14,cell,stride,accumulators_of(m14),row,channel,{==})
  e15 = step_cell(~Gemm.second_cell,~Tile.lane7,7n,cells,pairs,a15,cell,stride,accumulators_of(m15),row,channel,{==})
  %Equal.sym(F32,v16,(v15 + p15 : F32),e15) : {_ == wanted : F32}
  %Equal.sym(F32,v15,(v14 + p14 : F32),e14) : {(_ + p15 : F32) == wanted : F32}
  %Equal.sym(F32,v14,(v13 + p13 : F32),e13) : {((_ + p14 : F32) + p15 : F32) == wanted : F32}
  %Equal.sym(F32,v13,(v12 + p12 : F32),e12) : {(((_ + p13 : F32) + p14 : F32) + p15 : F32) == wanted : F32}
  %Equal.sym(F32,v12,(v11 + p11 : F32),e11) : {((((_ + p12 : F32) + p13 : F32) + p14 : F32) + p15 : F32) == wanted : F32}
  %Equal.sym(F32,v11,(v10 + p10 : F32),e10) : {(((((_ + p11 : F32) + p12 : F32) + p13 : F32) + p14 : F32) + p15 : F32) == wanted : F32}
  %Equal.sym(F32,v10,(v9 + p9 : F32),e9) : {((((((_ + p10 : F32) + p11 : F32) + p12 : F32) + p13 : F32) + p14 : F32) + p15 : F32) == wanted : F32}
  %Equal.sym(F32,v9,(v8 + p8 : F32),e8) : {(((((((_ + p9 : F32) + p10 : F32) + p11 : F32) + p12 : F32) + p13 : F32) + p14 : F32) + p15 : F32) == wanted : F32}
  %Equal.sym(F32,v8,(v7 + p7 : F32),e7) : {((((((((_ + p8 : F32) + p9 : F32) + p10 : F32) + p11 : F32) + p12 : F32) + p13 : F32) + p14 : F32) + p15 : F32) == wanted : F32}
  %Equal.sym(F32,v7,(v6 + p6 : F32),e6) :
    {(((((((((_ + p7 : F32) + p8 : F32) + p9 : F32) + p10 : F32) + p11 : F32) + p12 : F32) + p13 : F32) + p14 : F32) + p15 : F32) == wanted : F32}
  %Equal.sym(F32,v6,(v5 + p5 : F32),e5) :
    {((((((((((_ + p6 : F32) + p7 : F32) + p8 : F32) + p9 : F32) + p10 : F32) + p11 : F32) + p12 : F32) + p13 : F32) + p14 : F32) + p15 : F32) == wanted : F32}
  %Equal.sym(F32,v5,(v4 + p4 : F32),e4) :
    {(((((((((((_ + p5 : F32) + p6 : F32) + p7 : F32) + p8 : F32) + p9 : F32) + p10 : F32) + p11 : F32) + p12 : F32) + p13 : F32) + p14 : F32) + p15 : F32) == wanted : F32}
  %Equal.sym(F32,v4,(v3 + p3 : F32),e3) :
    {((((((((((((_ + p4 : F32) + p5 : F32) + p6 : F32) + p7 : F32) + p8 : F32) + p9 : F32) + p10 : F32) + p11 : F32) + p12 : F32) + p13 : F32) + p14 : F32) + p15 : F32) == wanted : F32}
  %Equal.sym(F32,v3,(v2 + p2 : F32),e2) :
    {(((((((((((((_ + p3 : F32) + p4 : F32) + p5 : F32) + p6 : F32) + p7 : F32) + p8 : F32) + p9 : F32) + p10 : F32) + p11 : F32) + p12 : F32) + p13 : F32) + p14 : F32) + p15 : F32) == wanted : F32}
  %Equal.sym(F32,v2,(v1 + p1 : F32),e1) :
    {((((((((((((((_ + p2 : F32) + p3 : F32) + p4 : F32) + p5 : F32) + p6 : F32) + p7 : F32) + p8 : F32) + p9 : F32) + p10 : F32) + p11 : F32) + p12 : F32) + p13 : F32) + p14 : F32) + p15 : F32) == wanted : F32}
  %Equal.sym(F32,v1,(v0 + p0 : F32),e0) :
    {(((((((((((((((_ + p1 : F32) + p2 : F32) + p3 : F32) + p4 : F32) + p5 : F32) + p6 : F32) + p7 : F32) + p8 : F32) + p9 : F32) + p10 : F32) + p11 : F32) + p12 : F32) + p13 : F32) + p14 : F32) + p15 : F32) == wanted : F32}
  {==}

def address_of(cursor: Gemm.Points) -> U32:
  Gemm.Points{address,cell,stride} = cursor
  address

def cell_of(cursor: Gemm.Points) -> U32:
  Gemm.Points{address,cell,stride} = cursor
  cell

# count groups add the products of their taps in order.
def groups_cell(count: Nat,+cells: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+address: U32,+cell: U32,+stride: U32,+accumulators: Gemm.Quad,+row: Nat,+channel: Nat) ->
  {cell_value(row,channel,accumulators_of(PointLaws.groups(~Gemm.row_times,count,PointLaws.Sources{cells,pairs},PointLaws.Reduced{Gemm.Points{address,cell,stride},accumulators}))) ==
    dot(taps(count,Gemm.Points{address,cell,stride}),row,channel,Sampled{cells,pairs,stride},cell_value(row,channel,accumulators)) : F32}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +sampled = {Sampled{cells,pairs,stride} : Sampled}
      +cursor = {Gemm.Points{address,cell,stride} : Gemm.Points}
      +next = PointLaws.grouped(~Gemm.row_times,PointLaws.Sources{cells,pairs},PointLaws.Reduced{cursor,accumulators})
      +after = group_after(cursor)
      +group = group_taps(cursor)
      +later = taps(rest,after)
      +start = cell_value(row,channel,accumulators)
      +reached = cell_value(row,channel,accumulators_of(next))
      Equal.trans(F32,cell_value(row,channel,accumulators_of(PointLaws.groups(~Gemm.row_times,rest,PointLaws.Sources{cells,pairs},next))),dot(later,row,channel,sampled,reached),
        dot(OrderedMatrix.cat(Tap,group,later),row,channel,sampled,start),
        groups_cell(rest,cells,pairs,address_of(after),cell_of(after),stride,accumulators_of(next),row,channel),
        Equal.trans(F32,dot(later,row,channel,sampled,reached),dot(later,row,channel,sampled,dot(group,row,channel,sampled,start)),
          dot(OrderedMatrix.cat(Tap,group,later),row,channel,sampled,start),
          Equal.cong(F32,F32,value => dot(later,row,channel,sampled,value),reached,dot(group,row,channel,sampled,start),
            group_cell(cells,pairs,address,cell,stride,accumulators,row,channel)),
          Equal.sym(F32,dot(OrderedMatrix.cat(Tap,group,later),row,channel,sampled,start),dot(later,row,channel,sampled,dot(group,row,channel,sampled,start)),
            Ordered.reduction_split(~Nat,~Tap,~Nat,~Sampled,~input_sample,~weight_sample,group,later,channel,row,sampled,start))))

def loop_accumulators(state: Gemm.PointLoop) -> Gemm.Quad:
  Gemm.PointLoop{cells,pairs,cursor,accumulators} = state
  accumulators

def represented_accumulators(+sources: PointLaws.Sources,+state: PointLaws.Reduced) -> {loop_accumulators(PointLaws.represent(sources,state)) == accumulators_of(state) : Gemm.Quad}:
  PointLaws.Sources{+cells,+pairs} = sources
  PointLaws.Reduced{+cursor,+accumulators} = state
  {==}

# The first tap of a job: the pairs of its pair from k = 0, the cells of its
# first position.
def job_start(+matrix: Gemm.PointMatrix,+place: Gemm.Place) -> Gemm.Points:
  Gemm.PointMatrix{Gemm.TileMatrix{channels,positions,+inner,start,output_positions},+cell,+stride} = matrix
  Gemm.Place{+pair,+unit} = place
  Gemm.Points{(pair * inner : U32),(cell + unit * 4 * stride : U32),stride}

def matrix_stride(matrix: Gemm.PointMatrix) -> U32:
  Gemm.PointMatrix{product,cell,stride} = matrix
  stride

# The k loop of a job leaves in every cell the ordered dot of the job's taps
# from +0.
def job_cell(+matrix: Gemm.PointMatrix,+place: Gemm.Place,+cells: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+row: Nat,+channel: Nat) ->
  {cell_value(row,channel,loop_accumulators(Loop.run(~Gemm.PointLoop,~Gemm.point_group(~Gemm.row_times),Gemm.point_groups_of(matrix),
      Gemm.point_start(Tree.pack(Tile.Tile8,cells),Tree.pack(Tile.Tile16,pairs),matrix,place)))) ==
    dot(taps(Gemm.point_groups_of(matrix),job_start(matrix,place)),row,channel,Sampled{cells,pairs,matrix_stride(matrix)},0.0) : F32}:
  Gemm.PointMatrix{Gemm.TileMatrix{+channels,+positions,+inner,+start,+output_positions},+cell,+stride} = matrix
  Gemm.Place{+pair,+unit} = place
  +sources = {PointLaws.Sources{cells,pairs} : PointLaws.Sources}
  +first = {Gemm.Points{(pair * inner : U32),(cell + unit * 4 * stride : U32),stride} : Gemm.Points}
  +model = {PointLaws.Reduced{first,Gemm.quad_zero()} : PointLaws.Reduced}
  +count = U32.to_nat((inner / 16 : U32))
  +ended = PointLaws.groups(~Gemm.row_times,count,sources,model)
  +wanted = dot(taps(count,first),row,channel,Sampled{cells,pairs,stride},0.0)
  %Equal.sym(Gemm.PointLoop,Loop.run(~Gemm.PointLoop,~Gemm.point_group(~Gemm.row_times),count,PointLaws.represent(sources,model)),PointLaws.represent(sources,ended),
    PointLaws.groups_matched(~Gemm.row_times,count,sources,PointLaws.represent(sources,model),model,{==})) :
    {cell_value(row,channel,loop_accumulators(_)) == wanted : F32}
  %Equal.sym(Gemm.Quad,loop_accumulators(PointLaws.represent(sources,ended)),accumulators_of(ended),represented_accumulators(sources,ended)) :
    {cell_value(row,channel,_) == wanted : F32}
  %zero_cell(row,channel) :
    {cell_value(row,channel,accumulators_of(ended)) == dot(taps(count,first),row,channel,Sampled{cells,pairs,stride},_) : F32}
  groups_cell(count,cells,pairs,(pair * inner : U32),(cell + unit * 4 * stride : U32),stride,Gemm.quad_zero(),row,channel)

# The suffix stays within one input cell and consumes consecutive pairs.
def suffix_taps(~at: U32 -> U32,count: Nat,+lane: U32,+cursor: Gemm.Points) -> List<&2,Tap>:
  match count:
    case 0n: []
    case 1n+rest: tap_at(~at,U32.to_nat(lane),cursor) <> suffix_taps(~at,rest,(lane + 1 : U32),advanced(cursor))

def suffix_step_cell(~at: U32 -> U32,lane: Nat,+cells: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+address: U32,+cell: U32,+stride: U32,
  +accumulators: Gemm.Quad,+row: Nat,+channel: Nat) ->
  {cell_value(row,channel,accumulators_of(PointLaws.tail_step(~at,lane,PointLaws.Sources{cells,pairs},PointLaws.Reduced{Gemm.Points{address,cell,stride},accumulators}))) ==
    dot([Tap{address,at(cell),lane}],row,channel,Sampled{cells,pairs,stride},cell_value(row,channel,accumulators)) : F32}:
  match lane:
    case 0n: step_cell(~at,~Tile.lane0,0n,cells,pairs,address,cell,stride,accumulators,row,channel,{==})
    case 1n: step_cell(~at,~Tile.lane1,1n,cells,pairs,address,cell,stride,accumulators,row,channel,{==})
    case 2n: step_cell(~at,~Tile.lane2,2n,cells,pairs,address,cell,stride,accumulators,row,channel,{==})
    case 3n: step_cell(~at,~Tile.lane3,3n,cells,pairs,address,cell,stride,accumulators,row,channel,{==})
    case 4n: step_cell(~at,~Tile.lane4,4n,cells,pairs,address,cell,stride,accumulators,row,channel,{==})
    case 5n: step_cell(~at,~Tile.lane5,5n,cells,pairs,address,cell,stride,accumulators,row,channel,{==})
    case 6n: step_cell(~at,~Tile.lane6,6n,cells,pairs,address,cell,stride,accumulators,row,channel,{==})
    case 7n+ +rest: step_cell(~at,~Tile.lane7,7n+rest,cells,pairs,address,cell,stride,accumulators,row,channel,{==})

def suffix_step_state(~at: U32 -> U32,lane: Nat,+cells: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+cursor: Gemm.Points,+accumulators: Gemm.Quad) ->
  {PointLaws.tail_step(~at,lane,PointLaws.Sources{cells,pairs},PointLaws.Reduced{cursor,accumulators}) ==
    PointLaws.Reduced{advanced(cursor),accumulators_of(PointLaws.tail_step(~at,lane,PointLaws.Sources{cells,pairs},PointLaws.Reduced{cursor,accumulators}))} : PointLaws.Reduced}:
  match lane cursor:
    case 0n Gemm.Points{address,cell,stride}: {==}
    case 1n Gemm.Points{address,cell,stride}: {==}
    case 2n Gemm.Points{address,cell,stride}: {==}
    case 3n Gemm.Points{address,cell,stride}: {==}
    case 4n Gemm.Points{address,cell,stride}: {==}
    case 5n Gemm.Points{address,cell,stride}: {==}
    case 6n Gemm.Points{address,cell,stride}: {==}
    case 7n+rest Gemm.Points{address,cell,stride}: {==}

def suffix_cell(~at: U32 -> U32,count: Nat,+lane: U32,+cells: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+address: U32,+cell: U32,+stride: U32,
  +accumulators: Gemm.Quad,+row: Nat,+channel: Nat) ->
  {cell_value(row,channel,accumulators_of(PointLaws.tail_cell(~at,count,lane,PointLaws.Sources{cells,pairs},PointLaws.Reduced{Gemm.Points{address,cell,stride},accumulators}))) ==
    dot(suffix_taps(~at,count,lane,Gemm.Points{address,cell,stride}),row,channel,Sampled{cells,pairs,stride},cell_value(row,channel,accumulators)) : F32}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +sources = {PointLaws.Sources{cells,pairs} : PointLaws.Sources}
      +cursor = {Gemm.Points{address,cell,stride} : Gemm.Points}
      +sampled = {Sampled{cells,pairs,stride} : Sampled}
      +index = U32.to_nat(lane)
      +next_lane = (lane + 1 : U32)
      +next = PointLaws.tail_step(~at,index,sources,PointLaws.Reduced{cursor,accumulators})
      +quad = accumulators_of(next)
      +taps = suffix_taps(~at,rest,next_lane,advanced(cursor))
      +wanted = dot(suffix_taps(~at,1n+rest,lane,cursor),row,channel,sampled,cell_value(row,channel,accumulators))
      %Equal.sym(PointLaws.Reduced,next,PointLaws.Reduced{advanced(cursor),quad},suffix_step_state(~at,index,cells,pairs,cursor,accumulators)) :
        {cell_value(row,channel,accumulators_of(PointLaws.tail_cell(~at,rest,next_lane,sources,_))) == wanted : F32}
      %suffix_step_cell(~at,index,cells,pairs,address,cell,stride,accumulators,row,channel) :
        {cell_value(row,channel,accumulators_of(PointLaws.tail_cell(~at,rest,next_lane,sources,PointLaws.Reduced{advanced(cursor),quad}))) ==
          dot(taps,row,channel,sampled,_) : F32}
      suffix_cell(~at,rest,next_lane,cells,pairs,(address + 1 : U32),cell,stride,quad,row,channel)

# The second input cell starts at the next weight pair after the first suffix.
def suffix_address(count: Nat,+address: U32) -> U32:
  match count:
    case 0n: address
    case 1n+rest: suffix_address(rest,(address + 1 : U32))

def suffix_state(~at: U32 -> U32,count: Nat,+lane: U32,+cells: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+address: U32,+cell: U32,+stride: U32,+accumulators: Gemm.Quad) ->
  {cursor_of(PointLaws.tail_cell(~at,count,lane,PointLaws.Sources{cells,pairs},PointLaws.Reduced{Gemm.Points{address,cell,stride},accumulators})) ==
    Gemm.Points{suffix_address(count,address),cell,stride} : Gemm.Points}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +sources = {PointLaws.Sources{cells,pairs} : PointLaws.Sources}
      +index = U32.to_nat(lane)
      +cursor = {Gemm.Points{address,cell,stride} : Gemm.Points}
      +next = PointLaws.tail_step(~at,index,sources,PointLaws.Reduced{cursor,accumulators})
      +quad = accumulators_of(next)
      %Equal.sym(PointLaws.Reduced,next,PointLaws.Reduced{advanced(cursor),quad},suffix_step_state(~at,index,cells,pairs,cursor,accumulators)) :
        {cursor_of(PointLaws.tail_cell(~at,rest,(lane + 1 : U32),sources,_)) == Gemm.Points{suffix_address(rest,(address + 1 : U32)),cell,stride} : Gemm.Points}
      suffix_state(~at,rest,(lane + 1 : U32),cells,pairs,(address + 1 : U32),cell,stride,quad)

def remainder_taps(+remainder: U32,+cursor: Gemm.Points) -> List<&2,Tap>:
  Gemm.Points{+address,+cell,+stride} = cursor
  +lower = U32.to_nat(U32.min(remainder,8))
  +upper = U32.to_nat((remainder - U32.min(remainder,8) : U32))
  OrderedMatrix.cat(Tap,suffix_taps(~Gemm.first_cell,lower,0,cursor),suffix_taps(~Gemm.second_cell,upper,0,Gemm.Points{suffix_address(lower,address),cell,stride}))

def reduced_fields(+state: PointLaws.Reduced) -> {state == PointLaws.Reduced{cursor_of(state),accumulators_of(state)} : PointLaws.Reduced}:
  PointLaws.Reduced{cursor,accumulators} = state
  {==}

def remainder_model_cell(+remainder: U32,+cells: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+address: U32,+cell: U32,+stride: U32,+accumulators: Gemm.Quad,+row: Nat,+channel: Nat) ->
  {cell_value(row,channel,accumulators_of(PointLaws.tail_cell(~Gemm.second_cell,U32.to_nat((remainder - U32.min(remainder,8) : U32)),0,PointLaws.Sources{cells,pairs},
    PointLaws.tail_cell(~Gemm.first_cell,U32.to_nat(U32.min(remainder,8)),0,PointLaws.Sources{cells,pairs},PointLaws.Reduced{Gemm.Points{address,cell,stride},accumulators})))) ==
    dot(remainder_taps(remainder,Gemm.Points{address,cell,stride}),row,channel,Sampled{cells,pairs,stride},cell_value(row,channel,accumulators)) : F32}:
  +sources = {PointLaws.Sources{cells,pairs} : PointLaws.Sources}
  +cursor = {Gemm.Points{address,cell,stride} : Gemm.Points}
  +sampled = {Sampled{cells,pairs,stride} : Sampled}
  +lower = U32.to_nat(U32.min(remainder,8))
  +upper = U32.to_nat((remainder - U32.min(remainder,8) : U32))
  +reached = PointLaws.tail_cell(~Gemm.first_cell,lower,0,sources,PointLaws.Reduced{cursor,accumulators})
  +first = suffix_taps(~Gemm.first_cell,lower,0,cursor)
  +next_address = suffix_address(lower,address)
  +following = {Gemm.Points{next_address,cell,stride} : Gemm.Points}
  +last = suffix_taps(~Gemm.second_cell,upper,0,following)
  +begun = cell_value(row,channel,accumulators)
  +quad = accumulators_of(reached)
  +next = cursor_of(reached)
  +folded = cell_value(row,channel,quad)
  +wanted = dot(remainder_taps(remainder,cursor),row,channel,sampled,begun)
  %Equal.sym(PointLaws.Reduced,reached,PointLaws.Reduced{next,quad},reduced_fields(reached)) :
    {cell_value(row,channel,accumulators_of(PointLaws.tail_cell(~Gemm.second_cell,upper,0,sources,_))) == wanted : F32}
  %Equal.sym(Gemm.Points,next,following,suffix_state(~Gemm.first_cell,lower,0,cells,pairs,address,cell,stride,accumulators)) :
    {cell_value(row,channel,accumulators_of(PointLaws.tail_cell(~Gemm.second_cell,upper,0,sources,PointLaws.Reduced{_,quad}))) == wanted : F32}
  Equal.trans(F32,cell_value(row,channel,accumulators_of(PointLaws.tail_cell(~Gemm.second_cell,upper,0,sources,PointLaws.Reduced{following,quad}))),
    dot(last,row,channel,sampled,folded),wanted,suffix_cell(~Gemm.second_cell,upper,0,cells,pairs,next_address,cell,stride,quad,row,channel),
    Equal.trans(F32,dot(last,row,channel,sampled,folded),dot(last,row,channel,sampled,dot(first,row,channel,sampled,begun)),wanted,
      Equal.cong(F32,F32,value => dot(last,row,channel,sampled,value),folded,dot(first,row,channel,sampled,begun),
        suffix_cell(~Gemm.first_cell,lower,0,cells,pairs,address,cell,stride,accumulators,row,channel)),
      Equal.sym(F32,wanted,dot(last,row,channel,sampled,dot(first,row,channel,sampled,begun)),
        Ordered.reduction_split(~Nat,~Tap,~Nat,~Sampled,~input_sample,~weight_sample,first,last,channel,row,sampled,begun))))

def native_remainder_cell(+matrix: Gemm.PointMatrix,+cells: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+address: U32,+cell: U32,+stride: U32,
  +accumulators: Gemm.Quad,+row: Nat,+channel: Nat) ->
  {cell_value(row,channel,loop_accumulators(Gemm.finish_inner(matrix,PointLaws.represent(PointLaws.Sources{cells,pairs},PointLaws.Reduced{Gemm.Points{address,cell,stride},accumulators})))) ==
    dot(remainder_taps(Gemm.point_remainder(matrix),Gemm.Points{address,cell,stride}),row,channel,Sampled{cells,pairs,stride},cell_value(row,channel,accumulators)) : F32}:
  +sources = {PointLaws.Sources{cells,pairs} : PointLaws.Sources}
  +cursor = {Gemm.Points{address,cell,stride} : Gemm.Points}
  +state = {PointLaws.Reduced{cursor,accumulators} : PointLaws.Reduced}
  +remainder = Gemm.point_remainder(matrix)
  +lower = U32.to_nat(U32.min(remainder,8))
  +upper = U32.to_nat((remainder - U32.min(remainder,8) : U32))
  +model = PointLaws.tail_cell(~Gemm.second_cell,upper,0,sources,PointLaws.tail_cell(~Gemm.first_cell,lower,0,sources,state))
  +wanted = dot(remainder_taps(remainder,cursor),row,channel,Sampled{cells,pairs,stride},cell_value(row,channel,accumulators))
  %Equal.sym(Gemm.PointLoop,Gemm.finish_inner(matrix,PointLaws.represent(sources,state)),PointLaws.represent(sources,model),PointLaws.tail_matches(matrix,sources,state)) :
    {cell_value(row,channel,loop_accumulators(_)) == wanted : F32}
  %Equal.sym(Gemm.Quad,loop_accumulators(PointLaws.represent(sources,model)),accumulators_of(model),represented_accumulators(sources,model)) :
    {cell_value(row,channel,_) == wanted : F32}
  remainder_model_cell(remainder,cells,pairs,address,cell,stride,accumulators,row,channel)
