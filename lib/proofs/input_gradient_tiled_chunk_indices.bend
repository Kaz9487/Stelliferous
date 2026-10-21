# Every actual chunk ordinal denotes a bounded clipped row/group chunk.
import Base
import ./input_gradient_tiled_chunk_geometry.bend as Geometry
import ./input_gradient_tiled_chunks.bend as Partition
import ./nat_division_order.bend as Division
import ./nat_division.bend as Quotient
import ./nat_ceiling.bend as Ceiling
import ./nat_order.bend as Order
import ./row_major_index.bend as RowMajor
import ./input_gradient_phase_domain.bend as Domain

def query(+grid: Geometry.Grid) -> Partition.Query:
  Geometry.query(grid,0n,0n)
def count(+grid: Geometry.Grid) -> Nat:
  Partition.chunk_count(query(grid))
def width(+grid: Geometry.Grid) -> Nat:
  Partition.group_chunks(query(grid))
def row(+grid: Geometry.Grid,+index: Nat) -> Nat:
  Nat.mul(Nat.div(index,width(grid)),Partition.chunk_rows(query(grid)))
def group(+grid: Geometry.Grid,+index: Nat) -> Nat:
  Nat.mul(Nat.mod(index,width(grid)),Partition.chunk_groups(query(grid)))
def width_positive(+grid: Geometry.Grid,+index: Nat,+inside: {Nat.is_lt(index,count(grid)) == True{} : Bool}) -> {Nat.is_lt(0n,width(grid)) == True{} : Bool}:
  +height = Partition.row_chunks(query(grid))
  RowMajor.left_factor_positive(width(grid),height,
    %RowMajor.multiply_commutative(height,width(grid)) : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.le_lt_trans(0n,index,count(grid),Order.zero_le(index),inside))
def inside(+grid: Geometry.Grid,+index: Nat,+bounded: {Nat.is_lt(index,count(grid)) == True{} : Bool}) -> Partition.Inside<Geometry.query(grid,row(grid,index),group(grid,index))>:
  Geometry.Grid{+call,e,c,range,+rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  +q = query(grid)
  +positive = width_positive(grid,index,bounded)
  +gp = Geometry.actual_groups_positive(g)
  Partition.Inside{rp,gp,
    %Ceiling.upper_comparison(Partition.rows(q),Partition.chunk_rows(q),Nat.div(index,width(grid)),rp) : {_ == True{} : Bool}
    Division.quotient_below(index,Partition.row_chunks(q),width(grid),positive,bounded),
    %Ceiling.upper_comparison(Partition.groups(q),Partition.chunk_groups(q),Nat.mod(index,width(grid)),gp) : {_ == True{} : Bool}
    Division.remainder_small(index,width(grid),positive)}
def exact(+grid: Geometry.Grid,+index: Nat,+bounded: {Nat.is_lt(index,count(grid)) == True{} : Bool}) -> {Partition.producing_chunk(Geometry.query(grid,row(grid,index),group(grid,index))) == index : Nat}:
  Geometry.Grid{+call,e,c,range,+rp,rr} = grid
  Domain.Call{+g,b,l,facts} = call
  +q = query(grid)
  +width = width(grid)
  +r = Nat.div(index,width)
  +c = Nat.mod(index,width)
  +gp = Geometry.actual_groups_positive(g)
  +row_exact = Quotient.exact_quotient(r,Partition.chunk_rows(q),rp)
  +group_exact = Quotient.exact_quotient(c,Partition.chunk_groups(q),gp)
  Equal.trans(Nat,Partition.producing_chunk(Geometry.query(grid,row(grid,index),group(grid,index))),Nat.add(Nat.mul(r,width),c),index,
    Equal.trans(Nat,Nat.add(Nat.mul(Nat.div(Nat.mul(r,Partition.chunk_rows(q)),Partition.chunk_rows(q)),width),Nat.div(Nat.mul(c,Partition.chunk_groups(q)),Partition.chunk_groups(q))),Nat.add(Nat.mul(r,width),Nat.div(Nat.mul(c,Partition.chunk_groups(q)),Partition.chunk_groups(q))),Nat.add(Nat.mul(r,width),c),
      Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(value,width),Nat.div(Nat.mul(c,Partition.chunk_groups(q)),Partition.chunk_groups(q))),Nat.div(Nat.mul(r,Partition.chunk_rows(q)),Partition.chunk_rows(q)),r,row_exact),
      Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(r,width),value),Nat.div(Nat.mul(c,Partition.chunk_groups(q)),Partition.chunk_groups(q)),c,group_exact)),
    Equal.sym(Nat,index,Nat.add(Nat.mul(r,width),c),Division.quotient_exact(index,width,width_positive(grid,index,bounded))))
def native_exact(+grid: Geometry.Grid,+index: Nat,+bounded: {Nat.is_lt(index,count(grid)) == True{} : Bool}) -> {Geometry.native_producer(grid,row(grid,index),group(grid,index)) == U32.from_nat(index) : U32}:
  Equal.cong(Nat,U32,U32.from_nat,Partition.producing_chunk(Geometry.query(grid,row(grid,index),group(grid,index))),index,exact(grid,index,bounded))
