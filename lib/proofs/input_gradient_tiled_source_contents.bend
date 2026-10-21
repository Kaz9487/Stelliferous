# Finite contents of retained gradient rows, independent of tiles and reductions.
import Base
import ./input_gradient_tiled_source_rows.bend as Rows
import ./input_gradient_tiled_leaf_execution.bend as Execution
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges
import ./nat_order.bend as Order
import ./u32_comparison.bend as Comparison
import ./bounded_u32_arithmetic.bend as Arithmetic

def global_row(+image: Rows.Slice,+row: Nat) -> U32:
  U32.from_nat(Nat.add(U32.to_nat(Execution.source_first(Rows.source(image))),row))
def Cell(+image: Rows.Slice,+output: Nat,+row: Nat,+column: Nat) -> Data:
  {Rows.local_read(image,U32.from_nat(output),global_row(image,row),U32.from_nat(column)) == Rows.global_read(image,U32.from_nat(output),global_row(image,row),U32.from_nat(column)) : F32}
def Content(+image: Rows.Slice,+outputs: U32) -> Data:
  Members.All(Nat,{output => Members.All(Nat,{row => Members.All(Nat,{column => Cell(image,output,row,column) : Nat -> Data},Ranges.range(U32.to_nat(Rows.width(image)),0n)) : Nat -> Data},Ranges.range(U32.to_nat(Execution.source_rows(Rows.source(image))),0n)) : Nat -> Data},Ranges.range(U32.to_nat(outputs),0n))
def cell(+image: Rows.Slice,+outputs: U32,+output: Nat,+row: Nat,+column: Nat,content: Content(image,outputs),
  +output_inside: {Nat.is_lt(output,U32.to_nat(outputs)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(Execution.source_rows(Rows.source(image)))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(Rows.width(image))) == True{} : Bool}) -> Cell(image,output,row,column):
  Members.all_member(Nat,{column => Cell(image,output,row,column) : Nat -> Data},column,Ranges.range(U32.to_nat(Rows.width(image)),0n),
    Members.all_member(Nat,{row => Members.All(Nat,{column => Cell(image,output,row,column) : Nat -> Data},Ranges.range(U32.to_nat(Rows.width(image)),0n)) : Nat -> Data},row,Ranges.range(U32.to_nat(Execution.source_rows(Rows.source(image))),0n),
      Members.all_member(Nat,{output => Members.All(Nat,{row => Members.All(Nat,{column => Cell(image,output,row,column) : Nat -> Data},Ranges.range(U32.to_nat(Rows.width(image)),0n)) : Nat -> Data},Ranges.range(U32.to_nat(Execution.source_rows(Rows.source(image))),0n)) : Nat -> Data},output,Ranges.range(U32.to_nat(outputs),0n),content,Members.range_member(U32.to_nat(outputs),0n,output,output_inside)),
      Members.range_member(U32.to_nat(Execution.source_rows(Rows.source(image))),0n,row,row_inside)),Members.range_member(U32.to_nat(Rows.width(image)),0n,column,column_inside))
def stored_cell(+image: Rows.Slice,+outputs: U32,+content: Content(image,outputs),+output: U32,+row: U32,+column: U32,
  +output_inside: {U32.is_lt(output,outputs) == True{} : Bool},+row_inside: {U32.is_lt(row,Rows.height(image)) == True{} : Bool},
  +column_inside: {U32.is_lt(column,Rows.width(image)) == True{} : Bool},
  +lower: {Nat.is_le(U32.to_nat(Execution.source_first(Rows.source(image))),U32.to_nat(row)) == True{} : Bool},
  +upper: {Nat.is_lt(U32.to_nat(row),Nat.add(U32.to_nat(Execution.source_first(Rows.source(image))),U32.to_nat(Execution.source_rows(Rows.source(image))))) == True{} : Bool}) ->
  {Rows.local_read(image,output,row,column) == Rows.global_read(image,output,row,column) : F32}:
  +first = U32.to_nat(Execution.source_first(Rows.source(image)))
  +local = Nat.sub(U32.to_nat(row),first)
  +encoded_row = Equal.trans(U32,global_row(image,local),U32.from_nat(U32.to_nat(row)),row,
    Equal.cong(Nat,U32,U32.from_nat,Nat.add(first,local),U32.to_nat(row),Order.restore_shift(first,U32.to_nat(row),lower)),Arithmetic.native_roundtrip(row))
  %Arithmetic.native_roundtrip(output) : {Rows.local_read(image,_,row,column) == Rows.global_read(image,_,row,column) : F32}
  %encoded_row : {Rows.local_read(image,U32.from_nat(U32.to_nat(output)),_,column) == Rows.global_read(image,U32.from_nat(U32.to_nat(output)),_,column) : F32}
  %Arithmetic.native_roundtrip(column) : {Rows.local_read(image,U32.from_nat(U32.to_nat(output)),global_row(image,local),_) == Rows.global_read(image,U32.from_nat(U32.to_nat(output)),global_row(image,local),_) : F32}
  cell(image,outputs,U32.to_nat(output),local,U32.to_nat(column),content,
    %Comparison.lt(output,outputs) : {_ == True{} : Bool} output_inside,
    Order.sub_lt(U32.to_nat(row),first,U32.to_nat(Execution.source_rows(Rows.source(image))),lower,upper),
    %Comparison.lt(column,Rows.width(image)) : {_ == True{} : Bool} column_inside)
def stored(+image: Rows.Slice,+outputs: U32,+content: Content(image,outputs)) -> Rows.Stored<image,outputs>:
  Rows.Stored{output => row => column => output_inside => row_inside => column_inside => lower => upper => stored_cell(image,outputs,content,output,row,column,output_inside,row_inside,column_inside,lower,upper)}
