# Every in-range row and column occurs in the gather's native write list.
import Base
import ./kernel_gather_blocks.bend as Blocks
import ./contiguous_copy.bend as Contiguous
import ./storage_tree_model.bend as Tree
import ./write_program_readback.bend as Readback
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./loop_projection.bend as Projection
import ./nat_order.bend as Order
import ./boolean_logic.bend as Boolean

def columns(~Element: Data,prefix: Nat,+suffix: Nat,+input: Tree.Tree<Element>,+position: U32,+source: U32,+destination: U32,+target: U32,
  +reached: {target == (position + U32.from_nat(prefix) : U32) : U32}) ->
  {Readback.contains(~Element,(destination + target : U32),Contiguous.writes(~Element,Nat.add(prefix,1n+suffix),input,position,source,destination)) == True{} : Bool}:
  match prefix:
    case 0n:
      +same = Equal.trans(U32,target,(position + 0 : U32),position,reached,Arithmetic.add_zero(position))
      %Equal.sym(U32,target,position,same) :
        {Readback.contains(~Element,(destination + _ : U32),Contiguous.writes(~Element,1n+suffix,input,position,source,destination)) == True{} : Bool}
      %Equal.sym(Bool,U32.is_eq((destination + position : U32),(destination + position : U32)),True{},Arithmetic.equal_reflexive((destination + position : U32))) :
        {_ || Readback.contains(~Element,(destination + position : U32),Contiguous.writes(~Element,suffix,input,(position + 1 : U32),source,destination)) == True{} : Bool}
      {==}
    case 1n+ +rest:
      +later = Contiguous.writes(~Element,Nat.add(rest,1n+suffix),input,(position + 1 : U32),source,destination)
      %Equal.sym(Bool,Readback.contains(~Element,(destination + target : U32),later),True{},
        columns(~Element,rest,suffix,input,(position + 1 : U32),source,destination,target,
          Equal.trans(U32,target,(position + U32.from_nat(1n+rest) : U32),((position + 1 : U32) + U32.from_nat(rest) : U32),reached,Projection.indexed_successor(position,rest)))) :
        {U32.is_eq((destination + position : U32),(destination + target : U32)) || _ == True{} : Bool}
      Boolean.or_true_right(U32.is_eq((destination + position : U32),(destination + target : U32)))

def row_columns(~Element: Data,+input: Tree.Tree<Element>,+row: U32,+width: U32,+offset: U32,+spatial: U32,+span: U32,+source_start: U32,+column: Nat,
  +inside: {Nat.is_lt(column,U32.to_nat(width)) == True{} : Bool}) ->
  {Readback.contains(~Element,(row * spatial + offset + U32.from_nat(column) : U32),
    Contiguous.writes(~Element,U32.to_nat(width),input,0,(row * span + source_start : U32),(row * spatial + offset : U32))) == True{} : Bool}:
  +tail = Nat.sub(U32.to_nat(width),1n+column)
  %Order.split_at_element(U32.to_nat(width),column,inside) :
    {Readback.contains(~Element,(row * spatial + offset + U32.from_nat(column) : U32),
      Contiguous.writes(~Element,_,input,0,(row * span + source_start : U32),(row * spatial + offset : U32))) == True{} : Bool}
  columns(~Element,column,tail,input,0,(row * span + source_start : U32),(row * spatial + offset : U32),U32.from_nat(column),
    Equal.sym(U32,(0 + U32.from_nat(column) : U32),U32.from_nat(column),Arithmetic.add_zero_left(U32.from_nat(column))))

def channels(~Element: Data,prefix: Nat,+suffix: Nat,+input: Tree.Tree<Element>,+row: U32,+width: U32,+offset: U32,+spatial: U32,+span: U32,+source_start: U32,+target: U32,+column: Nat,
  +reached: {target == (row + U32.from_nat(prefix) : U32) : U32},+inside: {Nat.is_lt(column,U32.to_nat(width)) == True{} : Bool}) ->
  {Readback.contains(~Element,(target * spatial + offset + U32.from_nat(column) : U32),Blocks.writes(~Element,Nat.add(prefix,1n+suffix),row,width,offset,spatial,span,source_start,input)) == True{} : Bool}:
  match prefix:
    case 0n:
      +same = Equal.trans(U32,target,(row + 0 : U32),row,reached,Arithmetic.add_zero(row))
      %Equal.sym(U32,target,row,same) :
        {Readback.contains(~Element,(_ * spatial + offset + U32.from_nat(column) : U32),Blocks.writes(~Element,1n+suffix,row,width,offset,spatial,span,source_start,input)) == True{} : Bool}
      Readback.covered_left(~Element,(row * spatial + offset + U32.from_nat(column) : U32),
        Contiguous.writes(~Element,U32.to_nat(width),input,0,(row * span + source_start : U32),(row * spatial + offset : U32)),
        Blocks.writes(~Element,suffix,(row + 1 : U32),width,offset,spatial,span,source_start,input),row_columns(~Element,input,row,width,offset,spatial,span,source_start,column,inside))
    case 1n+ +rest:
      Readback.covered_right(~Element,(target * spatial + offset + U32.from_nat(column) : U32),
        Contiguous.writes(~Element,U32.to_nat(width),input,0,(row * span + source_start : U32),(row * spatial + offset : U32)),
        Blocks.writes(~Element,Nat.add(rest,1n+suffix),(row + 1 : U32),width,offset,spatial,span,source_start,input),
        channels(~Element,rest,suffix,input,(row + 1 : U32),width,offset,spatial,span,source_start,target,column,
          Equal.trans(U32,target,(row + U32.from_nat(1n+rest) : U32),((row + 1 : U32) + U32.from_nat(rest) : U32),reached,Projection.indexed_successor(row,rest)),inside))

def copied_cell(~Element: Data,+rows: U32,+width: U32,+offset: U32,+spatial: U32,+span: U32,+source_start: U32,+input: Tree.Tree<Element>,+row: Nat,+column: Nat,
  +row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(width)) == True{} : Bool}) ->
  {Readback.contains(~Element,(U32.from_nat(row) * spatial + offset + U32.from_nat(column) : U32),Blocks.writes(~Element,U32.to_nat(rows),0,width,offset,spatial,span,source_start,input)) == True{} : Bool}:
  +tail = Nat.sub(U32.to_nat(rows),1n+row)
  %Order.split_at_element(U32.to_nat(rows),row,row_inside) :
    {Readback.contains(~Element,(U32.from_nat(row) * spatial + offset + U32.from_nat(column) : U32),Blocks.writes(~Element,_,0,width,offset,spatial,span,source_start,input)) == True{} : Bool}
  channels(~Element,row,tail,input,0,width,offset,spatial,span,source_start,U32.from_nat(row),column,
    Equal.sym(U32,(0 + U32.from_nat(row) : U32),U32.from_nat(row),Arithmetic.add_zero_left(U32.from_nat(row))),column_inside)
