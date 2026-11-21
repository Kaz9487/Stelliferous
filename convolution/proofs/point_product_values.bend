# The scalar readers of a point-product rectangle and the public dot.
import Base
import ../../lib/ordered_matrix.bend as Matrix
import ../../lib/proofs/kernel_gemm_point_dot.bend as Dot
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ./product_values.bend as Products

type Readers<-Left: Data,-Right: Data> is Data:
  Readers{left: Left,right: Right,width: Nat}

def input(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,readers: Readers<Left,Right>,position: Nat,tap: Nat) -> F32:
  Readers{left,right,width} = readers
  left_value(left,U32.from_nat(Nat.add(Nat.mul(position,width),tap)))

def weight(~Left: Data,~Right: Data,~right_value: Right -> U32 -> F32,readers: Readers<Left,Right>,channel: Nat,tap: Nat) -> F32:
  Readers{left,right,width} = readers
  right_value(right,U32.from_nat(Nat.add(Nat.mul(channel,width),tap)))

def OrderedFrom(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  +remaining: Nat,+first: Nat,readers: Readers<Left,Right>,+position: Nat,+channel: Nat,+accumulator: F32) -> Data:
  Readers{+left,+right,+width} = readers
  {Matrix.dot(~Nat,~Nat,~Nat,~Readers<Left,Right>,~input(~Left,~left_value,~Right),~weight(~Left,~Right,~right_value),Ranges.range(remaining,first),channel,position,readers,accumulator) ==
    Products.ordered_sum(~Left,~left_value,~Right,~right_value,remaining,first,left,Nat.mul(position,width),right,Nat.mul(channel,width),accumulator) : F32}

def ordered(~Left: Data,~left_value: Left -> U32 -> F32,~Right: Data,~right_value: Right -> U32 -> F32,
  remaining: Nat,+first: Nat,+readers: Readers<Left,Right>,+position: Nat,+channel: Nat,+accumulator: F32) ->
  OrderedFrom(~Left,~left_value,~Right,~right_value,remaining,first,readers,position,channel,accumulator):
  match remaining readers:
    case 0n Readers{left,right,width}: {==}
    case 1n+ +rest Readers{+left,+right,+width}:
      ordered(~Left,~left_value,~Right,~right_value,rest,1n+first,Readers{left,right,width},position,channel,
        (accumulator + input(~Left,~left_value,~Right,readers,position,first) * weight(~Left,~Right,~right_value,readers,channel,first) : F32))
