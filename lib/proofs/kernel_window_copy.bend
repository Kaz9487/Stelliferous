# The actual window copy is one contiguous copy per channel plane, in order.
import Base
import ../kernel_window.bend as KernelWindow
import ../traversal_array.bend as Traversal
import ./storage_tree_model.bend as Tree
import ./contiguous_copy.bend as Contiguous
import ./write_program.bend as Writes
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./bounded_u32_arithmetic.bend as Arithmetic

def plane_writes(+channel: U32,+count: U32,+source_stride: U32,+source_start: U32,+target_stride: U32,+input: Tree.Tree<F32>) ->
  List<&2,Writes.Assignment<F32>>:
  Contiguous.writes(~F32,U32.to_nat(count),input,0,(channel * source_stride + source_start : U32),(channel * target_stride : U32))

def writes(remaining: Nat,+channel: U32,+count: U32,+source_stride: U32,+source_start: U32,+target_stride: U32,+input: Tree.Tree<F32>) ->
  List<&2,Writes.Assignment<F32>>:
  match remaining:
    case 0n: []
    case 1n+rest:
      List.append(&2,Writes.Assignment<F32>,plane_writes(channel,count,source_stride,source_start,target_stride,input),
        writes(rest,(channel + 1 : U32),count,source_stride,source_start,target_stride,input))

def planes(+remaining: Nat,+channel: U32,+count: U32,+source_stride: U32,+source_start: U32,+target_stride: U32,
  +input: Tree.Tree<F32>,+output: Tree.Tree<F32>) ->
  {KernelWindow.copy_planes(remaining,channel,count,source_stride,source_start,target_stride,(Tree.pack(F32,input),Tree.pack(F32,output))) ==
    (Tree.pack(F32,input),Tree.pack(F32,Writes.pure(F32,writes(remaining,channel,count,source_stride,source_start,target_stride,input),output))) : Array<F32> & Array<F32>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +here = plane_writes(channel,count,source_stride,source_start,target_stride,input)
      +later = writes(rest,(channel + 1 : U32),count,source_stride,source_start,target_stride,input)
      %Equal.sym(Array<F32> & Array<F32>,Traversal.copy(~F32,U32.to_nat(count),(channel * source_stride + source_start : U32),(channel * target_stride : U32),
          Tree.pack(F32,input),Tree.pack(F32,output)),
        (Tree.pack(F32,input),Tree.pack(F32,Writes.pure(F32,here,output))),
        Contiguous.copy(~F32,U32.to_nat(count),input,output,(channel * source_stride + source_start : U32),(channel * target_stride : U32))) :
        {KernelWindow.copy_planes(rest,(channel + 1 : U32),count,source_stride,source_start,target_stride,_) ==
          (Tree.pack(F32,input),Tree.pack(F32,Writes.pure(F32,List.append(&2,Writes.Assignment<F32>,here,later),output))) : Array<F32> & Array<F32>}
      %Equal.sym(Tree.Tree<F32>,Writes.pure(F32,List.append(&2,Writes.Assignment<F32>,here,later),output),
        Writes.pure(F32,later,Writes.pure(F32,here,output)),Writes.concatenated_writes(F32,here,later,output)) :
        {KernelWindow.copy_planes(rest,(channel + 1 : U32),count,source_stride,source_start,target_stride,
          (Tree.pack(F32,input),Tree.pack(F32,Writes.pure(F32,here,output)))) ==
          (Tree.pack(F32,input),Tree.pack(F32,_)) : Array<F32> & Array<F32>}
      planes(rest,(channel + 1 : U32),count,source_stride,source_start,target_stride,input,Writes.pure(F32,here,output))

def channel_successor(+channel: U32,+rest: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  channels_room: {Nat.is_le(Nat.add(U32.to_nat(channel),1n+rest),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((channel + 1 : U32)) == 1n+U32.to_nat(channel) : Nat}:
  %Addition.successor_width(U32.to_nat(channel)) : {U32.to_nat((channel + 1 : U32)) == _ : Nat}
  Arithmetic.add(channel,1,capacity,capacity_ok,
    %Equal.sym(Nat,Nat.add(U32.to_nat(channel),1n),1n+U32.to_nat(channel),Addition.successor_width(U32.to_nat(channel))) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    Order.strict_follow(U32.to_nat(channel),U32.to_nat(capacity),Order.room_index(U32.to_nat(channel),rest,U32.to_nat(capacity),channels_room)))

