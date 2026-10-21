# Clipped row and group chunks cover each leaf position once.
import Base
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_division_order.bend as Division
import ./nat_ceiling.bend as Ceiling
import ./nat_extrema.bend as Extrema
import ./binary_radix.bend as Radix
import ./row_major_index.bend as RowMajor

def chunks_inside(+position: Nat,+count: Nat,+span: Nat,+positive: {Nat.is_lt(0n,span) == True{} : Bool},
  inside: {Nat.is_lt(position,count) == True{} : Bool}) ->
  {Nat.is_lt(Nat.div(position,span),Ceiling.ceiling(count,span)) == True{} : Bool}:
  Division.quotient_below(position,Ceiling.ceiling(count,span),span,positive,
    Bounds.nat_trans(position,count,Nat.mul(Ceiling.ceiling(count,span),span),inside,Ceiling.covers(count,span,positive)))

def clipped_local(+position: Nat,+count: Nat,+span: Nat,+positive: {Nat.is_lt(0n,span) == True{} : Bool},
  inside: {Nat.is_lt(position,count) == True{} : Bool}) ->
  {Nat.is_lt(Nat.mod(position,span),Nat.min(span,Nat.sub(count,Nat.mul(Nat.div(position,span),span)))) == True{} : Bool}:
  +first = Nat.mul(Nat.div(position,span),span)
  +local = Nat.mod(position,span)
  +exact = Division.quotient_exact(position,span,positive)
  Extrema.minimum_strict_lower(span,Nat.sub(count,first),local,Division.remainder_small(position,span,positive),
    %Equal.sym(Bool,Nat.is_lt(local,Nat.sub(count,first)),Nat.is_lt(Nat.add(first,local),count),Order.subtraction_lt(count,first,local)) : {_ == True{} : Bool}
    %exact : {Nat.is_lt(_,count) == True{} : Bool} inside)

type Query is Data:
  Query{rows: Nat,groups: Nat,chunk_rows: Nat,chunk_groups: Nat,row: Nat,group: Nat}

def rows(query: Query) -> Nat:
  Query{rows,groups,chunk_rows,chunk_groups,row,group} = query
  rows
def groups(query: Query) -> Nat:
  Query{rows,groups,chunk_rows,chunk_groups,row,group} = query
  groups
def chunk_rows(query: Query) -> Nat:
  Query{rows,groups,chunk_rows,chunk_groups,row,group} = query
  chunk_rows
def chunk_groups(query: Query) -> Nat:
  Query{rows,groups,chunk_rows,chunk_groups,row,group} = query
  chunk_groups
def row(query: Query) -> Nat:
  Query{rows,groups,chunk_rows,chunk_groups,row,group} = query
  row
def group(query: Query) -> Nat:
  Query{rows,groups,chunk_rows,chunk_groups,row,group} = query
  group
def group_chunks(+query: Query) -> Nat:
  Ceiling.ceiling(groups(query),chunk_groups(query))
def row_chunks(+query: Query) -> Nat:
  Ceiling.ceiling(rows(query),chunk_rows(query))
def row_chunk(+query: Query) -> Nat:
  Nat.div(row(query),chunk_rows(query))
def group_chunk(+query: Query) -> Nat:
  Nat.div(group(query),chunk_groups(query))
def producing_chunk(+query: Query) -> Nat:
  Nat.add(Nat.mul(row_chunk(query),group_chunks(query)),group_chunk(query))
def chunk_count(+query: Query) -> Nat:
  Nat.mul(row_chunks(query),group_chunks(query))

type Inside<-query: Query> is Data:
  Inside{rows_positive: {Nat.is_lt(0n,chunk_rows(query)) == True{} : Bool},groups_positive: {Nat.is_lt(0n,chunk_groups(query)) == True{} : Bool},
    row_inside: {Nat.is_lt(row(query),rows(query)) == True{} : Bool},group_inside: {Nat.is_lt(group(query),groups(query)) == True{} : Bool}}

def producer_inside(+query: Query,+inside: Inside<query>) -> {Nat.is_lt(producing_chunk(query),chunk_count(query)) == True{} : Bool}:
  Inside{+rp,+gp,+ri,+gi} = inside
  RowMajor.radix(row_chunk(query),row_chunks(query),group_chunk(query),group_chunks(query),
    chunks_inside(row(query),rows(query),chunk_rows(query),rp,ri),chunks_inside(group(query),groups(query),chunk_groups(query),gp,gi))

def producer_row(+query: Query,+inside: Inside<query>) -> {Nat.div(producing_chunk(query),group_chunks(query)) == row_chunk(query) : Nat}:
  Inside{rp,+gp,ri,+gi} = inside
  +group_inside = chunks_inside(group(query),groups(query),chunk_groups(query),gp,gi)
  Radix.quotient(row_chunk(query),group_chunk(query),group_chunks(query),Order.positive_bound(group_chunk(query),group_chunks(query),group_inside),group_inside)

def producer_group(+query: Query,+inside: Inside<query>) -> {Nat.mod(producing_chunk(query),group_chunks(query)) == group_chunk(query) : Nat}:
  Inside{rp,+gp,ri,+gi} = inside
  +group_inside = chunks_inside(group(query),groups(query),chunk_groups(query),gp,gi)
  Radix.remainder(row_chunk(query),group_chunk(query),group_chunks(query),Order.positive_bound(group_chunk(query),group_chunks(query),group_inside),group_inside)

