# Where a block of a relane writes: its write at a lane lands at the first
# target cell plus the sum over the target cell axes of the index of the
# block on the axis times the axis stride, the source lane axis at the lane.

import Base
import ../tensor_tiled.bend as Tiled
import ../tensor_tasks.bend as Tasks
import ../tensor_grid.bend as Grid
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./tensor_relane_geometry.bend as Geometry
import ./tensor_box_digits.bend as BoxDigits
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_tile_coordinates.bend as Coordinates
import ./tensor_move_readback.bend as Held
import ../kernel_gemm.bend as KernelGemm
import ./nat_order.bend as Order
import ./word_arithmetic.bend as WordArithmetic
import ./word_multiplication.bend as Multiplication
import ./word_subtraction.bend as Subtraction
import ./word_encoding.bend as WordEncoding
import ./natural_addition.bend as Addition
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ./u32_division.bend as U32Division
import ./u32_extrema.bend as Extrema
import ./nat_extrema.bend as NatExtrema
import ./tensor_grid_pieces.bend as GridLaws

# ---- The fields of a block that place its writes ----

def writes_of(+block: Tiled.Block) -> U32 & (U32 & U32):
  Tiled.Block{source,target,+stride,+low,+high,reads,step} = block
  (stride,(low,high))

def target_of(+block: Tiled.Block) -> U32:
  Tiled.Block{source,+target,stride,low,high,reads,step} = block
  target

def stride_of(+block: Tiled.Block) -> U32:
  Pair.fst(U32,U32 & U32,writes_of(block))

def low_of(+block: Tiled.Block) -> U32:
  Pair.fst(U32,U32,Pair.snd(U32,U32 & U32,writes_of(block)))

def high_of(+block: Tiled.Block) -> U32:
  Pair.snd(U32,U32,Pair.snd(U32,U32 & U32,writes_of(block)))

# What an axis of each role adds to the target cell of a write at lane lane:
# the source lane axis its lane in the block past the first index of the
# piece.
def write_term(role: Tiled.Role,+low: U32,+index: U32,+lane: U32,+target: U32) -> U32:
  match role:
    case Tiled.Shared{}: (index * target : U32)
    case Tiled.Groups{}: (index * target : U32)
    case Tiled.Lanes{}: (((((low / 8 + index : U32) * 8 : U32) + lane : U32) - low : U32) * target : U32)

def write_sum(axes: List<&2,Tiled.Ranged>,+index: U32,+lane: U32) -> U32:
  match axes:
    case Nil{}: 0
    case Con{Tiled.Ranged{Tiled.Blocked{role,source,lanes},+low,extent,+count,+inner,+target},rest}:
      (write_term(role,low,(index / U32.max(inner,1) % U32.max(count,1) : U32),lane,target) + write_sum(rest,index,lane) : U32)

# ---- Axes without the source lane axis ----

# They add their terms to the target cell and keep the fields that place
# the writes.
def kept_target(axes: List<&2,Tiled.Ranged>,block: Tiled.Block,+index: U32,+lane: U32,+none: Geometry.NoLanes(Geometry.blocked_of(axes))) ->
  {target_of(Tiled.block_at(axes,index,block)) == (target_of(block) + write_sum(axes,index,lane) : U32) : U32}:
  match axes block:
    case Nil{} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}: Equal.sym(U32,(start + 0 : U32),start,Arithmetic.add_zero(start))
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Lanes{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}:
      Empty.absurd({target_of(Tiled.block_at(Con{Tiled.Ranged{Tiled.Blocked{Tiled.Lanes{},source,lanes},low,extent,count,inner,target},rest},index,
        Tiled.Block{first,start,stride,block_low,block_high,reads,step})) ==
        (start + write_sum(Con{Tiled.Ranged{Tiled.Blocked{Tiled.Lanes{},source,lanes},low,extent,count,inner,target},rest},index,lane) : U32) : U32},none)
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Shared{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}:
      +at = (index / U32.max(inner,1) % U32.max(count,1) : U32)
      +next = Tiled.block_axis(Tiled.Shared{},Tiled.Block{first,start,stride,block_low,block_high,reads,step},at,low,extent,source,lanes,target)
      Equal.trans(U32,target_of(Tiled.block_at(rest,index,next)),((start + (at * target : U32) : U32) + write_sum(rest,index,lane) : U32),
        (start + ((at * target : U32) + write_sum(rest,index,lane) : U32) : U32),kept_target(rest,next,index,lane,none),
        Arithmetic.add_associative(start,(at * target : U32),write_sum(rest,index,lane)))
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Groups{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}:
      +at = (index / U32.max(inner,1) % U32.max(count,1) : U32)
      +next = Tiled.block_axis(Tiled.Groups{},Tiled.Block{first,start,stride,block_low,block_high,reads,step},at,low,extent,source,lanes,target)
      Equal.trans(U32,target_of(Tiled.block_at(rest,index,next)),((start + (at * target : U32) : U32) + write_sum(rest,index,lane) : U32),
        (start + ((at * target : U32) + write_sum(rest,index,lane) : U32) : U32),kept_target(rest,next,index,lane,none),
        Arithmetic.add_associative(start,(at * target : U32),write_sum(rest,index,lane)))

def kept_writes(axes: List<&2,Tiled.Ranged>,block: Tiled.Block,+index: U32,+none: Geometry.NoLanes(Geometry.blocked_of(axes))) ->
  {writes_of(Tiled.block_at(axes,index,block)) == writes_of(block) : U32 & (U32 & U32)}:
  match axes block:
    case Nil{} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}: {==}
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Lanes{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}:
      Empty.absurd({writes_of(Tiled.block_at(Con{Tiled.Ranged{Tiled.Blocked{Tiled.Lanes{},source,lanes},low,extent,count,inner,target},rest},index,
        Tiled.Block{first,start,stride,block_low,block_high,reads,step})) == (stride,(block_low,block_high)) : U32 & (U32 & U32)},none)
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Shared{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}:
      kept_writes(rest,Tiled.block_axis(Tiled.Shared{},Tiled.Block{first,start,stride,block_low,block_high,reads,step},
        (index / U32.max(inner,1) % U32.max(count,1) : U32),low,extent,source,lanes,target),index,none)
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Groups{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}:
      kept_writes(rest,Tiled.block_axis(Tiled.Groups{},Tiled.Block{first,start,stride,block_low,block_high,reads,step},
        (index / U32.max(inner,1) % U32.max(count,1) : U32),low,extent,source,lanes,target),index,none)

# ---- The cell of a write ----

# The first written index of the lane block and the lane after it add to the
# lane past the first index of the piece.
def lane_offset(+start: U32,+low: U32,+first: U32,+lane: U32) -> {((start - low : U32) + (lane - (start - first : U32) : U32) : U32) == ((first + lane : U32) - low : U32) : U32}:
  +inner = (lane - (start - first : U32) : U32)
  +moved = Equal.trans(U32,(start + inner : U32),((start + lane : U32) - (start - first : U32) : U32),(first + lane : U32),
    Equal.sym(U32,((start + lane : U32) - (start - first : U32) : U32),(start + inner : U32),Arithmetic.subtract_from_sum(start,lane,(start - first : U32))),
    Equal.trans(U32,((start + lane : U32) - (start - first : U32) : U32),(((start + lane : U32) - start : U32) + first : U32),(first + lane : U32),
      Equal.sym(U32,(((start + lane : U32) - start : U32) + first : U32),((start + lane : U32) - (start - first : U32) : U32),Arithmetic.subtract_difference((start + lane : U32),start,first)),
      Equal.trans(U32,(((start + lane : U32) - start : U32) + first : U32),(lane + first : U32),(first + lane : U32),
        Equal.cong(U32,U32,value => (value + first : U32),((start + lane : U32) - start : U32),lane,Arithmetic.subtract_first(start,lane)),
        Arithmetic.add_commutative(lane,first))))
  Equal.trans(U32,((start - low : U32) + inner : U32),((start + inner : U32) - low : U32),((first + lane : U32) - low : U32),
    Arithmetic.add_after_subtract(start,low,inner),Equal.cong(U32,U32,value => (value - low : U32),(start + inner : U32),(first + lane : U32),moved))

def regrouped(+base: U32,+head: U32,+rest: U32,+last: U32) -> {(((base + head : U32) + rest : U32) + last : U32) == (base + ((head + last : U32) + rest : U32) : U32) : U32}:
  Equal.trans(U32,(((base + head : U32) + rest : U32) + last : U32),((base + head : U32) + (rest + last : U32) : U32),(base + ((head + last : U32) + rest : U32) : U32),
    Arithmetic.add_associative((base + head : U32),rest,last),
    Equal.trans(U32,((base + head : U32) + (rest + last : U32) : U32),((base + head : U32) + (last + rest : U32) : U32),(base + ((head + last : U32) + rest : U32) : U32),
      Equal.cong(U32,U32,value => ((base + head : U32) + value : U32),(rest + last : U32),(last + rest : U32),Arithmetic.add_commutative(rest,last)),
      Equal.trans(U32,((base + head : U32) + (last + rest : U32) : U32),(((base + head : U32) + last : U32) + rest : U32),(base + ((head + last : U32) + rest : U32) : U32),
        Equal.sym(U32,(((base + head : U32) + last : U32) + rest : U32),((base + head : U32) + (last + rest : U32) : U32),Arithmetic.add_associative((base + head : U32),last,rest)),
        Equal.trans(U32,(((base + head : U32) + last : U32) + rest : U32),((base + (head + last : U32) : U32) + rest : U32),(base + ((head + last : U32) + rest : U32) : U32),
          Equal.cong(U32,U32,value => (value + rest : U32),((base + head : U32) + last : U32),(base + (head + last : U32) : U32),Arithmetic.add_associative(base,head,last)),
          Arithmetic.add_associative(base,(head + last : U32),rest)))))

# The cell of the write of a block at a lane, past the first target cell: the
# sum over the axes of their terms, when the source lane axis appears at most
# once and the first block writes in place.
def write_cell(axes: List<&2,Tiled.Ranged>,block: Tiled.Block,+index: U32,+lane: U32,+most: Geometry.LanesAtMost(Geometry.blocked_of(axes)),
  +still: {stride_of(block) == 0 : U32}) ->
  {(target_of(Tiled.block_at(axes,index,block)) + ((lane - low_of(Tiled.block_at(axes,index,block)) : U32) * stride_of(Tiled.block_at(axes,index,block)) : U32) : U32) ==
    (target_of(block) + write_sum(axes,index,lane) : U32) : U32}:
  match axes block:
    case Nil{} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}:
      %Equal.sym(U32,stride,0,still) : {(start + ((lane - block_low : U32) * _ : U32) : U32) == (start + 0 : U32) : U32}
      %Equal.sym(U32,((lane - block_low : U32) * 0 : U32),0,Arithmetic.multiply_zero_right((lane - block_low : U32))) : {(start + _ : U32) == (start + 0 : U32) : U32}
      {==}
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Shared{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}:
      +at = (index / U32.max(inner,1) % U32.max(count,1) : U32)
      +next = Tiled.block_axis(Tiled.Shared{},Tiled.Block{first,start,stride,block_low,block_high,reads,step},at,low,extent,source,lanes,target)
      +final = Tiled.block_at(rest,index,next)
      Equal.trans(U32,(target_of(final) + ((lane - low_of(final) : U32) * stride_of(final) : U32) : U32),((start + (at * target : U32) : U32) + write_sum(rest,index,lane) : U32),
        (start + ((at * target : U32) + write_sum(rest,index,lane) : U32) : U32),write_cell(rest,next,index,lane,most,still),
        Arithmetic.add_associative(start,(at * target : U32),write_sum(rest,index,lane)))
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Groups{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}:
      +at = (index / U32.max(inner,1) % U32.max(count,1) : U32)
      +next = Tiled.block_axis(Tiled.Groups{},Tiled.Block{first,start,stride,block_low,block_high,reads,step},at,low,extent,source,lanes,target)
      +final = Tiled.block_at(rest,index,next)
      Equal.trans(U32,(target_of(final) + ((lane - low_of(final) : U32) * stride_of(final) : U32) : U32),((start + (at * target : U32) : U32) + write_sum(rest,index,lane) : U32),
        (start + ((at * target : U32) + write_sum(rest,index,lane) : U32) : U32),write_cell(rest,next,index,lane,most,still),
        Arithmetic.add_associative(start,(at * target : U32),write_sum(rest,index,lane)))
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Lanes{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}:
      +at = (index / U32.max(inner,1) % U32.max(count,1) : U32)
      +group = (low / 8 + at : U32)
      +lead = (group * 8 : U32)
      +begin = U32.max(low,lead)
      +next = Tiled.block_axis(Tiled.Lanes{},Tiled.Block{first,start,stride,block_low,block_high,reads,step},at,low,extent,source,lanes,target)
      +final = Tiled.block_at(rest,index,next)
      +later = write_sum(rest,index,lane)
      +ahead = ((begin - low : U32) * target : U32)
      +after = ((lane - (begin - lead : U32) : U32) * target : U32)
      +kept = kept_writes(rest,next,index,most)
      +low_kept = Equal.cong(U32 & (U32 & U32),U32,pair => Pair.fst(U32,U32,Pair.snd(U32,U32 & U32,pair)),writes_of(final),writes_of(next),kept)
      +stride_kept = Equal.cong(U32 & (U32 & U32),U32,pair => Pair.fst(U32,U32 & U32,pair),writes_of(final),writes_of(next),kept)
      +placed = {%Equal.sym(U32,target_of(final),((start + ahead : U32) + later : U32),kept_target(rest,next,index,lane,most)) :
          {(_ + ((lane - low_of(final) : U32) * stride_of(final) : U32) : U32) == (((start + ahead : U32) + later : U32) + after : U32) : U32}
        %Equal.sym(U32,low_of(final),(begin - lead : U32),low_kept) :
          {(((start + ahead : U32) + later : U32) + ((lane - _ : U32) * stride_of(final) : U32) : U32) == (((start + ahead : U32) + later : U32) + after : U32) : U32}
        %Equal.sym(U32,stride_of(final),target,stride_kept) :
          {(((start + ahead : U32) + later : U32) + ((lane - (begin - lead : U32) : U32) * _ : U32) : U32) == (((start + ahead : U32) + later : U32) + after : U32) : U32}
        {==} : {(target_of(final) + ((lane - low_of(final) : U32) * stride_of(final) : U32) : U32) == (((start + ahead : U32) + later : U32) + after : U32) : U32}}
      +term = Equal.trans(U32,(ahead + after : U32),(((begin - low : U32) + (lane - (begin - lead : U32) : U32) : U32) * target : U32),(((lead + lane : U32) - low : U32) * target : U32),
        Equal.sym(U32,(((begin - low : U32) + (lane - (begin - lead : U32) : U32) : U32) * target : U32),(ahead + after : U32),
          Arithmetic.multiply_additive((begin - low : U32),(lane - (begin - lead : U32) : U32),target)),
        Equal.cong(U32,U32,value => (value * target : U32),((begin - low : U32) + (lane - (begin - lead : U32) : U32) : U32),((lead + lane : U32) - low : U32),
          lane_offset(begin,low,lead,lane)))
      Equal.trans(U32,(target_of(final) + ((lane - low_of(final) : U32) * stride_of(final) : U32) : U32),(((start + ahead : U32) + later : U32) + after : U32),
        (start + ((((lead + lane : U32) - low : U32) * target : U32) + later : U32) : U32),placed,
        Equal.trans(U32,(((start + ahead : U32) + later : U32) + after : U32),(start + ((ahead + after : U32) + later : U32) : U32),
          (start + ((((lead + lane : U32) - low : U32) * target : U32) + later : U32) : U32),regrouped(start,ahead,later,after),
          Equal.cong(U32,U32,value => (start + (value + later : U32) : U32),(ahead + after : U32),(((lead + lane : U32) - low : U32) * target : U32),term)))

# ---- The cell of a write as digits ----

# The digits of block index at lane lane in the box of the piece.
def box_digits(axes: List<&2,Tiled.Ranged>,+index: U32,+lane: Nat) -> List<&2,Nat>:
  match axes:
    case Nil{}: []
    case Con{Tiled.Ranged{Tiled.Blocked{role,source,lanes},+low,extent,+count,+inner,target},rest}:
      Nat.sub(Geometry.role_digit(role,low,(index / U32.max(inner,1) % U32.max(count,1) : U32),lane),U32.to_nat(low)) <> box_digits(rest,index,lane)

def targets_of(axes: List<&2,Tiled.Ranged>) -> List<&2,U32>:
  match axes:
    case Nil{}: []
    case Con{Tiled.Ranged{blocked,low,extent,count,inner,+target},rest}: target <> targets_of(rest)

# The lane of the source lane axis lies at or past the first index of the
# piece.
def LaneFrom(role: Tiled.Role,+low: U32,+index: U32,+lane: Nat) -> Data:
  match role:
    case Tiled.Lanes{}: {Nat.is_le(U32.to_nat(low),Geometry.role_digit(Tiled.Lanes{},low,index,lane)) == True{} : Bool}
    case Tiled.Shared{}: Unit
    case Tiled.Groups{}: Unit

def LanesFrom(axes: List<&2,Tiled.Ranged>,+index: U32,+lane: Nat) -> Data:
  match axes:
    case Nil{}: Unit
    case Con{Tiled.Ranged{Tiled.Blocked{role,source,lanes},+low,extent,+count,+inner,target},rest}:
      KernelGemm.Both<LaneFrom(role,low,(index / U32.max(inner,1) % U32.max(count,1) : U32),lane),LanesFrom(rest,index,lane)>

def term_digit(role: Tiled.Role,+low: U32,+index: U32,+lane: U32,+target: U32,+from: LaneFrom(role,low,index,U32.to_nat(lane))) ->
  {write_term(role,low,index,lane,target) == (U32.from_nat(Nat.sub(Geometry.role_digit(role,low,index,U32.to_nat(lane)),U32.to_nat(low))) * target : U32) : U32}:
  match role:
    case Tiled.Shared{}:
      %Equal.sym(Nat,Nat.sub(Nat.add(U32.to_nat(low),U32.to_nat(index)),U32.to_nat(low)),U32.to_nat(index),Order.subtract_shift(U32.to_nat(low),U32.to_nat(index))) :
        {(index * target : U32) == (U32.from_nat(_) * target : U32) : U32}
      %Equal.sym(U32,U32.from_nat(U32.to_nat(index)),index,Arithmetic.native_roundtrip(index)) : {(index * target : U32) == (_ * target : U32) : U32}
      {==}
    case Tiled.Groups{}:
      %Equal.sym(Nat,Nat.sub(Nat.add(U32.to_nat(low),U32.to_nat(index)),U32.to_nat(low)),U32.to_nat(index),Order.subtract_shift(U32.to_nat(low),U32.to_nat(index))) :
        {(index * target : U32) == (U32.from_nat(_) * target : U32) : U32}
      %Equal.sym(U32,U32.from_nat(U32.to_nat(index)),index,Arithmetic.native_roundtrip(index)) : {(index * target : U32) == (_ * target : U32) : U32}
      {==}
    case Tiled.Lanes{}:
      +group = Nat.add(U32.to_nat((low / 8 : U32)),U32.to_nat(index))
      +digit = Nat.add(Nat.mul(group,8n),U32.to_nat(lane))
      +written = (((((low / 8 + index : U32) * 8 : U32) + lane : U32) - low : U32) * target : U32)
      %Subtraction.encoded_subtract(digit,U32.to_nat(low),from) : {written == (_ * target : U32) : U32}
      %Equal.sym(U32,U32.from_nat(U32.to_nat(low)),low,Arithmetic.native_roundtrip(low)) : {written == ((U32.from_nat(digit) - _ : U32) * target : U32) : U32}
      %WordArithmetic.native_add(Nat.mul(group,8n),U32.to_nat(lane)) : {written == ((_ - low : U32) * target : U32) : U32}
      %Multiplication.native_encoded_multiply(group,8n) : {written == (((_ + U32.from_nat(U32.to_nat(lane)) : U32) - low : U32) * target : U32) : U32}
      %Arithmetic.native_add((low / 8 : U32),index) : {written == ((((_ * 8 : U32) + U32.from_nat(U32.to_nat(lane)) : U32) - low : U32) * target : U32) : U32}
      %Equal.sym(U32,U32.from_nat(U32.to_nat(lane)),lane,Arithmetic.native_roundtrip(lane)) :
        {written == (((((low / 8 + index : U32) * 8 : U32) + _ : U32) - low : U32) * target : U32) : U32}
      {==}

# The cell of a write past the first target cell is the box digits of the
# block at the lane times their strides.
def write_digits(axes: List<&2,Tiled.Ranged>,+index: U32,+lane: U32,+from: LanesFrom(axes,index,U32.to_nat(lane))) ->
  {write_sum(axes,index,lane) == BoxDigits.stepped_digits(box_digits(axes,index,U32.to_nat(lane)),targets_of(axes)) : U32}:
  match axes:
    case Nil{}: {==}
    case Con{Tiled.Ranged{Tiled.Blocked{+role,+source,+lanes},+low,+extent,+count,+inner,+target},+rest}:
      KernelGemm.Both{+here,+later} = from
      +at = (index / U32.max(inner,1) % U32.max(count,1) : U32)
      %Equal.sym(U32,write_term(role,low,at,lane,target),(U32.from_nat(Nat.sub(Geometry.role_digit(role,low,at,U32.to_nat(lane)),U32.to_nat(low))) * target : U32),
          term_digit(role,low,at,lane,target,here)) :
        {(_ + write_sum(rest,index,lane) : U32) == BoxDigits.stepped_digits(box_digits(Con{Tiled.Ranged{Tiled.Blocked{role,source,lanes},low,extent,count,inner,target},rest},index,U32.to_nat(lane)),
          targets_of(Con{Tiled.Ranged{Tiled.Blocked{role,source,lanes},low,extent,count,inner,target},rest})) : U32}
      %Equal.sym(U32,write_sum(rest,index,lane),BoxDigits.stepped_digits(box_digits(rest,index,U32.to_nat(lane)),targets_of(rest)),write_digits(rest,index,lane,later)) :
        {((U32.from_nat(Nat.sub(Geometry.role_digit(role,low,at,U32.to_nat(lane)),U32.to_nat(low))) * target : U32) + _ : U32) ==
          BoxDigits.stepped_digits(box_digits(Con{Tiled.Ranged{Tiled.Blocked{role,source,lanes},low,extent,count,inner,target},rest},index,U32.to_nat(lane)),
            targets_of(Con{Tiled.Ranged{Tiled.Blocked{role,source,lanes},low,extent,count,inner,target},rest})) : U32}
      {==}

# ---- The lanes a lane block writes ----

# The first lane of the lane block at index index, the first lane it writes
# and the lanes past it.
def lane_first(+low: U32,+index: U32) -> U32:
  ((low / 8 + index : U32) * 8 : U32)

def lane_low(+low: U32,+index: U32) -> U32:
  (U32.max(low,lane_first(low,index)) - lane_first(low,index) : U32)

def lane_high(+low: U32,+extent: U32,+index: U32) -> U32:
  U32.min((low + extent - lane_first(low,index) : U32),8)

# The numbers of the first cell, the first lane and the end of the lane block
# at an index of its axis, the first cell below the end of the axis.
def LaneWords(+low: U32,+extent: U32,+at: U32) -> Data:
  Held.Both<{U32.to_nat(lane_low(low,at)) == Nat.sub(Nat.max(U32.to_nat(low),Nat.mul(Nat.add(U32.to_nat((low / 8 : U32)),U32.to_nat(at)),8n)),
      Nat.mul(Nat.add(U32.to_nat((low / 8 : U32)),U32.to_nat(at)),8n)) : Nat},
    Held.Both<{U32.to_nat(lane_high(low,extent,at)) == Nat.min(Nat.sub(Nat.add(U32.to_nat(low),U32.to_nat(extent)),
        Nat.mul(Nat.add(U32.to_nat((low / 8 : U32)),U32.to_nat(at)),8n)),8n) : Nat},
      {Nat.is_lt(Nat.mul(Nat.add(U32.to_nat((low / 8 : U32)),U32.to_nat(at)),8n),Nat.add(U32.to_nat(low),U32.to_nat(extent))) == True{} : Bool}>>

def lane_words(+low: U32,+extent: U32,+at: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(cap)) == True{} : Bool},
  +at_le: {Nat.is_le(U32.to_nat(at),Nat.sub(Nat.div(Nat.sub(Nat.add(U32.to_nat(low),U32.to_nat(extent)),1n),8n),Nat.div(U32.to_nat(low),8n))) == True{} : Bool}) ->
  LaneWords(low,extent,at):
  +l = U32.to_nat(low)
  +e = U32.to_nat(extent)
  +sum = Nat.add(l,e)
  +limit = U32.to_nat(cap)
  +sum_le = Order.transitive(sum,Nat.double(sum),limit,Order.le_double(sum),room)
  +low_le = Order.transitive(l,sum,limit,Order.le_add(l,e),sum_le)
  +one_le = Order.transitive(1n,e,sum,Order.positive_one(e,positive),Order.right_le(l,e))
  +last = Nat.sub(sum,1n)
  +sum_word = Arithmetic.sum_exact(cap,cap_ok,low,extent,l,e,{==},{==},sum_le)
  +last_below = Order.last_below(sum,BoxDigits.follow_strict(0n,sum,one_le))
  +last_le = Order.transitive(last,sum,limit,Order.sub_le(sum,1n),sum_le)
  +below = Nat.div(l,8n)
  +above = Nat.div(last,8n)
  +below_word = U32Division.divide(low,8,2147483648,{==},{==},{==})
  +low_last = {%DivisionOrder.subtract_sum(l,1n) : {Nat.is_le(_,last) == True{} : Bool}
    DivisionOrder.subtract_le(Nat.add(l,1n),sum,1n,Order.offset_le(l,1n,e,Order.positive_one(e,positive))) : {Nat.is_le(l,last) == True{} : Bool}}
  +ordered = DivisionOrder.quotient_le(l,last,8n,{==},low_last)
  +above_last = Order.transitive(above,Nat.mul(above,8n),last,RowMajor.product_covers(above,8n,{==}),DivisionOrder.product_below(last,8n,{==}))
  +above_le = Order.transitive(above,last,limit,above_last,last_le)
  +difference = Nat.sub(above,below)
  +a = U32.to_nat(at)
  +group = Nat.add(U32.to_nat((low / 8 : U32)),a)
  +group_le = {%Equal.sym(Nat,U32.to_nat((low / 8 : U32)),below,below_word) : {Nat.is_le(Nat.add(_,a),above) == True{} : Bool}
    %Order.restore_shift(below,above,ordered) : {Nat.is_le(Nat.add(below,a),_) == True{} : Bool}
    Order.offset_le(below,a,difference,at_le) : {Nat.is_le(group,above) == True{} : Bool}}
  +f = Nat.mul(group,8n)
  +first_last = Order.transitive(f,Nat.mul(above,8n),last,RowMajor.mul_le(group,above,8n,group_le),DivisionOrder.product_below(last,8n,{==}))
  +first_below = Order.le_lt_trans(f,last,sum,first_last,last_below)
  +first_le = Order.transitive(f,last,limit,first_last,last_le)
  +group_word = Arithmetic.sum_exact(cap,cap_ok,(low / 8 : U32),at,U32.to_nat((low / 8 : U32)),a,{==},{==},
    Order.transitive(group,above,limit,group_le,above_le))
  +first_word = Arithmetic.product_exact(cap,cap_ok,(low / 8 + at : U32),8,group,8n,group_word,{==},first_le)
  +s = Nat.max(l,f)
  +start_word = {%first_word : {U32.to_nat(U32.max(low,lane_first(low,at))) == Nat.max(l,_) : Nat}
    Extrema.maximum(low,lane_first(low,at)) : {U32.to_nat(U32.max(low,lane_first(low,at))) == s : Nat}}
  +low_word = Arithmetic.difference_exact(cap,cap_ok,U32.max(low,lane_first(low,at)),lane_first(low,at),s,f,start_word,first_word,
    NatExtrema.maximum_right(l,f),NatExtrema.maximum_upper(l,f,limit,low_le,first_le))
  +reach_word = Arithmetic.difference_exact(cap,cap_ok,(low + extent : U32),lane_first(low,at),sum,f,sum_word,first_word,Bounds.strict_le(f,sum,first_below),sum_le)
  +high_word = {%reach_word : {U32.to_nat(lane_high(low,extent,at)) == Nat.min(_,8n) : Nat}
    Extrema.minimum((low + extent - lane_first(low,at) : U32),8) : {U32.to_nat(lane_high(low,extent,at)) == Nat.min(Nat.sub(sum,f),8n) : Nat}}
  Held.Both{low_word,Held.Both{high_word,first_below}}

# A lane between the first lane and the end of the lane block lies at or past
# the first index of the piece, its digit in the box below the extent, and
# below eight.
def lane_finish(+low: U32,+extent: U32,+at: U32,+lane: U32,
  +from_low: {Nat.is_le(U32.to_nat(lane_low(low,at)),U32.to_nat(lane)) == True{} : Bool},
  +to_high: {Nat.is_lt(U32.to_nat(lane),U32.to_nat(lane_high(low,extent,at))) == True{} : Bool},+words: LaneWords(low,extent,at)) ->
  Held.Both<LaneFrom(Tiled.Lanes{},low,at,U32.to_nat(lane)),
    Held.Both<{Nat.is_lt(Nat.sub(Geometry.role_digit(Tiled.Lanes{},low,at,U32.to_nat(lane)),U32.to_nat(low)),U32.to_nat(extent)) == True{} : Bool},
      {Nat.is_lt(U32.to_nat(lane),8n) == True{} : Bool}>>:
  Held.Both{+low_word,Held.Both{+high_word,+first_below}} = words
  +l = U32.to_nat(low)
  +e = U32.to_nat(extent)
  +sum = Nat.add(l,e)
  +f = Nat.mul(Nat.add(U32.to_nat((low / 8 : U32)),U32.to_nat(at)),8n)
  +s = Nat.max(l,f)
  +i = U32.to_nat(lane)
  +lane_from = {%low_word : {Nat.is_le(_,i) == True{} : Bool} from_low : {Nat.is_le(Nat.sub(s,f),i) == True{} : Bool}}
  +lane_to = {%high_word : {Nat.is_lt(i,_) == True{} : Bool} to_high : {Nat.is_lt(i,Nat.min(Nat.sub(sum,f),8n)) == True{} : Bool}}
  +lane_small = BoxDigits.follow_strict(i,8n,Order.transitive(1n+i,Nat.min(Nat.sub(sum,f),8n),8n,Order.strict_follow(i,Nat.min(Nat.sub(sum,f),8n),lane_to),
    NatExtrema.minimum_right(Nat.sub(sum,f),8n)))
  +lane_reach = BoxDigits.follow_strict(i,Nat.sub(sum,f),Order.transitive(1n+i,Nat.min(Nat.sub(sum,f),8n),Nat.sub(sum,f),
    Order.strict_follow(i,Nat.min(Nat.sub(sum,f),8n),lane_to),NatExtrema.minimum_left(Nat.sub(sum,f),8n)))
  +digit = Nat.add(f,i)
  +start_reach = {%Order.restore_shift(f,s,NatExtrema.maximum_right(l,f)) : {Nat.is_le(_,digit) == True{} : Bool}
    Order.offset_le(f,Nat.sub(s,f),i,lane_from) : {Nat.is_le(s,digit) == True{} : Bool}}
  +digit_from = Order.transitive(l,s,digit,NatExtrema.maximum_left(l,f),start_reach)
  +digit_below = {%Order.restore_shift(f,sum,Bounds.strict_le(f,sum,first_below)) : {Nat.is_lt(digit,_) == True{} : Bool}
    Order.offset_lt(f,i,Nat.sub(sum,f),lane_reach) : {Nat.is_lt(digit,sum) == True{} : Bool}}
  Held.Both{digit_from,Held.Both{Order.sub_lt(digit,l,e,digit_from,digit_below),lane_small}}

# The groups of eight an extent from a first index spans number at most the
# extent.
def lanes_within(+low: Nat,+extent: Nat,+positive: {Nat.is_lt(0n,extent) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.sub(Nat.div(Nat.sub(Nat.add(low,extent),1n),8n),Nat.div(low,8n)),1n),extent) == True{} : Bool}:
  match extent:
    case 0n: Empty.absurd({Nat.is_le(Nat.add(Nat.sub(Nat.div(Nat.sub(Nat.add(low,0n),1n),8n),Nat.div(low,8n)),1n),0n) == True{} : Bool},WordEncoding.false_true(positive))
    case 1n+ +rest:
      +start = Nat.div(low,8n)
      +lane = Nat.mod(low,8n)
      +tail = Nat.add(lane,rest)
      +groups = Nat.div(tail,8n)
      +shifted = Equal.trans(Nat,Nat.add(low,rest),Nat.add(Nat.add(Nat.mul(start,8n),lane),rest),Nat.add(Nat.mul(start,8n),tail),
        Equal.cong(Nat,Nat,value => Nat.add(value,rest),low,Nat.add(Nat.mul(start,8n),lane),DivisionOrder.quotient_exact(low,8n,{==})),
        Addition.associative(Nat.mul(start,8n),lane,rest))
      +last_group = Equal.trans(Nat,Nat.div(Nat.add(low,rest),8n),Nat.div(Nat.add(Nat.mul(start,8n),tail),8n),Nat.add(start,groups),
        Equal.cong(Nat,Nat,value => Nat.div(value,8n),Nat.add(low,rest),Nat.add(Nat.mul(start,8n),tail),shifted),DivisionOrder.quotient_shift(tail,8n,start,{==}))
      +inside = Order.le_lt_trans(tail,Nat.add(lane,rest),Nat.mul(1n+rest,8n),Order.reflexive(tail),
        Order.le_lt_trans(Nat.add(lane,rest),Nat.add(lane,Nat.mul(rest,8n)),Nat.add(8n,Nat.mul(rest,8n)),
          Order.offset_le(lane,rest,Nat.mul(rest,8n),RowMajor.product_covers(rest,8n,{==})),
          Order.add_lt(lane,8n,Nat.mul(rest,8n),DivisionOrder.remainder_small(low,8n,{==}))))
      +below = DivisionOrder.quotient_below(tail,1n+rest,8n,{==},inside)
      %Equal.sym(Nat,Nat.sub(Nat.add(low,1n+rest),1n),Nat.add(low,rest),Order.subtract_one_complement(low,rest)) : {Nat.is_le(Nat.add(Nat.sub(Nat.div(_,8n),start),1n),1n+rest) == True{} : Bool}
      %Equal.sym(Nat,Nat.div(Nat.add(low,rest),8n),Nat.add(start,groups),last_group) : {Nat.is_le(Nat.add(Nat.sub(_,start),1n),1n+rest) == True{} : Bool}
      %Equal.sym(Nat,Nat.sub(Nat.add(start,groups),start),groups,Order.subtract_shift(start,groups)) : {Nat.is_le(Nat.add(_,1n),1n+rest) == True{} : Bool}
      %Equal.sym(Nat,Nat.add(groups,1n),1n+groups,Addition.successor_width(groups)) : {Nat.is_le(_,1n+rest) == True{} : Bool}
      Order.strict_follow(groups,1n+rest,below)

# The blocks of the source lane axis: one past the difference of the groups
# of its last and first index, at most its extent.
def LaneCount(+low: U32,+extent: U32,+count: U32) -> Data:
  Held.Both<{U32.to_nat(count) == Nat.add(Nat.sub(Nat.div(Nat.sub(Nat.add(U32.to_nat(low),U32.to_nat(extent)),1n),8n),Nat.div(U32.to_nat(low),8n)),1n) : Nat},
    {Nat.is_le(Nat.add(Nat.sub(Nat.div(Nat.sub(Nat.add(U32.to_nat(low),U32.to_nat(extent)),1n),8n),Nat.div(U32.to_nat(low),8n)),1n),U32.to_nat(extent)) == True{} : Bool}>

def lane_count(+low: U32,+extent: U32,+count: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(cap)) == True{} : Bool},
  +counted: {count == Tiled.block_count(Tiled.Lanes{},low,extent) : U32}) -> LaneCount(low,extent,count):
  +l = U32.to_nat(low)
  +e = U32.to_nat(extent)
  +sum = Nat.add(l,e)
  +limit = U32.to_nat(cap)
  +sum_le = Order.transitive(sum,Nat.double(sum),limit,Order.le_double(sum),room)
  +one_le = Order.transitive(1n,e,sum,Order.positive_one(e,positive),Order.right_le(l,e))
  +last = Nat.sub(sum,1n)
  +sum_word = Arithmetic.sum_exact(cap,cap_ok,low,extent,l,e,{==},{==},sum_le)
  +last_le = Order.transitive(last,sum,limit,Order.sub_le(sum,1n),sum_le)
  +last_word = Arithmetic.difference_exact(cap,cap_ok,(low + extent : U32),1,sum,1n,sum_word,{==},one_le,sum_le)
  +below = Nat.div(l,8n)
  +above = Nat.div(last,8n)
  +below_word = U32Division.divide(low,8,2147483648,{==},{==},{==})
  +above_word = {%last_word : {U32.to_nat((((low + extent : U32) - 1 : U32) / 8 : U32)) == Nat.div(_,8n) : Nat}
    U32Division.divide(((low + extent : U32) - 1 : U32),8,2147483648,{==},{==},{==}) : {U32.to_nat((((low + extent : U32) - 1 : U32) / 8 : U32)) == above : Nat}}
  +low_last = {%DivisionOrder.subtract_sum(l,1n) : {Nat.is_le(_,last) == True{} : Bool}
    DivisionOrder.subtract_le(Nat.add(l,1n),sum,1n,Order.offset_le(l,1n,e,Order.positive_one(e,positive))) : {Nat.is_le(l,last) == True{} : Bool}}
  +ordered = DivisionOrder.quotient_le(l,last,8n,{==},low_last)
  +above_last = Order.transitive(above,Nat.mul(above,8n),last,RowMajor.product_covers(above,8n,{==}),DivisionOrder.product_below(last,8n,{==}))
  +above_le = Order.transitive(above,last,limit,above_last,last_le)
  +difference = Nat.sub(above,below)
  +counted_value = Nat.add(difference,1n)
  +difference_word = Arithmetic.difference_exact(cap,cap_ok,(((low + extent : U32) - 1 : U32) / 8 : U32),(low / 8 : U32),above,below,above_word,below_word,ordered,above_le)
  +count_le = lanes_within(l,e,positive)
  +count_word = {%Equal.sym(U32,count,Tiled.block_count(Tiled.Lanes{},low,extent),counted) : {U32.to_nat(_) == counted_value : Nat}
    Arithmetic.sum_exact(cap,cap_ok,((((low + extent : U32) - 1 : U32) / 8 : U32) - (low / 8 : U32) : U32),1,difference,1n,difference_word,{==},
      Order.transitive(counted_value,e,limit,count_le,Order.transitive(e,sum,limit,Order.right_le(l,e),sum_le))) : {U32.to_nat(count) == counted_value : Nat}}
  Held.Both{count_word,count_le}

# The lane block at an index of its axis lies inside the lanes of the axis.
def lane_block_words(+low: U32,+extent: U32,+index: U32,+inner: U32,+count: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(cap)) == True{} : Bool},+counts: LaneCount(low,extent,count)) ->
  LaneWords(low,extent,(index / U32.max(inner,1) % U32.max(count,1) : U32)):
  Held.Both{+count_word,+count_le} = counts
  +quotient = (index / U32.max(inner,1) : U32)
  +at = (quotient % U32.max(count,1) : U32)
  +l = U32.to_nat(low)
  +e = U32.to_nat(extent)
  +limit = U32.to_nat(cap)
  +difference = Nat.sub(Nat.div(Nat.sub(Nat.add(l,e),1n),8n),Nat.div(l,8n))
  +counted_value = Nat.add(difference,1n)
  +count_positive = {%Equal.sym(Nat,counted_value,1n+difference,Addition.successor_width(difference)) : {Nat.is_lt(0n,_) == True{} : Bool} {==} :
    {Nat.is_lt(0n,counted_value) == True{} : Bool}}
  +divisor_word = Extrema.max_one_exact(count,counted_value,count_word,count_positive)
  +divisor_back = Equal.sym(Nat,U32.to_nat(U32.max(count,1)),counted_value,divisor_word)
  +at_word = {%divisor_word : {U32.to_nat(at) == Nat.mod(U32.to_nat(quotient),_) : Nat}
    U32Division.modulo(quotient,U32.max(count,1),cap,cap_ok,
      {%divisor_back : {Nat.is_le(Nat.double(_),limit) == True{} : Bool}
        Order.transitive(Nat.double(counted_value),Nat.double(Nat.add(l,e)),limit,
          Order.double_le(counted_value,Nat.add(l,e),Order.transitive(counted_value,e,Nat.add(l,e),count_le,Order.right_le(l,e))),room) :
        {Nat.is_le(Nat.double(U32.to_nat(U32.max(count,1))),limit) == True{} : Bool}},
      {%divisor_back : {Nat.is_lt(0n,_) == True{} : Bool} count_positive : {Nat.is_lt(0n,U32.to_nat(U32.max(count,1))) == True{} : Bool}}) :
    {U32.to_nat(at) == Nat.mod(U32.to_nat(quotient),counted_value) : Nat}}
  +a = U32.to_nat(at)
  +at_below = {%Equal.sym(Nat,a,Nat.mod(U32.to_nat(quotient),counted_value),at_word) : {Nat.is_lt(_,counted_value) == True{} : Bool}
    DivisionOrder.remainder_small(U32.to_nat(quotient),counted_value,count_positive) : {Nat.is_lt(a,counted_value) == True{} : Bool}}
  +at_le = Division.lt_succ_le(a,difference,{%Addition.successor_width(difference) : {Nat.is_lt(a,_) == True{} : Bool} at_below : {Nat.is_lt(a,1n+difference) == True{} : Bool}})
  lane_words(low,extent,at,cap,cap_ok,positive,room,at_le)


# A lane the lane block writes lies at or past the first index of the piece,
# its digit in the box below the extent, and below eight.
def lane_facts_with(+low: U32,+extent: U32,+index: U32,+inner: U32,+count: U32,+lane: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(cap)) == True{} : Bool},
  +from_low: {Nat.is_le(U32.to_nat(lane_low(low,(index / U32.max(inner,1) % U32.max(count,1) : U32))),U32.to_nat(lane)) == True{} : Bool},
  +to_high: {Nat.is_lt(U32.to_nat(lane),U32.to_nat(lane_high(low,extent,(index / U32.max(inner,1) % U32.max(count,1) : U32)))) == True{} : Bool},
  +counts: LaneCount(low,extent,count)) ->
  Held.Both<LaneFrom(Tiled.Lanes{},low,(index / U32.max(inner,1) % U32.max(count,1) : U32),U32.to_nat(lane)),
    Held.Both<{Nat.is_lt(Nat.sub(Geometry.role_digit(Tiled.Lanes{},low,(index / U32.max(inner,1) % U32.max(count,1) : U32),U32.to_nat(lane)),U32.to_nat(low)),
      U32.to_nat(extent)) == True{} : Bool},{Nat.is_lt(U32.to_nat(lane),8n) == True{} : Bool}>>:
  lane_finish(low,extent,(index / U32.max(inner,1) % U32.max(count,1) : U32),lane,from_low,to_high,
    lane_block_words(low,extent,index,inner,count,cap,cap_ok,positive,room,counts))

def lane_facts(+low: U32,+extent: U32,+index: U32,+inner: U32,+count: U32,+lane: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(cap)) == True{} : Bool},
  +counted: {count == Tiled.block_count(Tiled.Lanes{},low,extent) : U32},
  +from_low: {Nat.is_le(U32.to_nat(lane_low(low,(index / U32.max(inner,1) % U32.max(count,1) : U32))),U32.to_nat(lane)) == True{} : Bool},
  +to_high: {Nat.is_lt(U32.to_nat(lane),U32.to_nat(lane_high(low,extent,(index / U32.max(inner,1) % U32.max(count,1) : U32)))) == True{} : Bool}) ->
  Held.Both<LaneFrom(Tiled.Lanes{},low,(index / U32.max(inner,1) % U32.max(count,1) : U32),U32.to_nat(lane)),
    Held.Both<{Nat.is_lt(Nat.sub(Geometry.role_digit(Tiled.Lanes{},low,(index / U32.max(inner,1) % U32.max(count,1) : U32),U32.to_nat(lane)),U32.to_nat(low)),
      U32.to_nat(extent)) == True{} : Bool},{Nat.is_lt(U32.to_nat(lane),8n) == True{} : Bool}>>:
  lane_facts_with(low,extent,index,inner,count,lane,cap,cap_ok,positive,room,from_low,to_high,lane_count(low,extent,count,cap,cap_ok,positive,room,counted))

# The parts of the facts of a lane.
def lane_from(+low: U32,+extent: U32,+index: U32,+inner: U32,+count: U32,+lane: U32,+facts: Held.Both<LaneFrom(Tiled.Lanes{},low,(index / U32.max(inner,1) % U32.max(count,1) : U32),U32.to_nat(lane)),Held.Both<{Nat.is_lt(Nat.sub(Geometry.role_digit(Tiled.Lanes{},low,(index / U32.max(inner,1) % U32.max(count,1) : U32),U32.to_nat(lane)),U32.to_nat(low)),U32.to_nat(extent)) == True{} : Bool},{Nat.is_lt(U32.to_nat(lane),8n) == True{} : Bool}>>) -> LaneFrom(Tiled.Lanes{},low,(index / U32.max(inner,1) % U32.max(count,1) : U32),U32.to_nat(lane)):
  Held.Both{+from,+rest} = facts
  from

def lane_rest(+low: U32,+extent: U32,+index: U32,+inner: U32,+count: U32,+lane: U32,+facts: Held.Both<LaneFrom(Tiled.Lanes{},low,(index / U32.max(inner,1) % U32.max(count,1) : U32),U32.to_nat(lane)),Held.Both<{Nat.is_lt(Nat.sub(Geometry.role_digit(Tiled.Lanes{},low,(index / U32.max(inner,1) % U32.max(count,1) : U32),U32.to_nat(lane)),U32.to_nat(low)),U32.to_nat(extent)) == True{} : Bool},{Nat.is_lt(U32.to_nat(lane),8n) == True{} : Bool}>>) -> Held.Both<{Nat.is_lt(Nat.sub(Geometry.role_digit(Tiled.Lanes{},low,(index / U32.max(inner,1) % U32.max(count,1) : U32),U32.to_nat(lane)),U32.to_nat(low)),U32.to_nat(extent)) == True{} : Bool},{Nat.is_lt(U32.to_nat(lane),8n) == True{} : Bool}>:
  Held.Both{+from,+rest} = facts
  rest

def lane_below(+low: U32,+extent: U32,+index: U32,+inner: U32,+count: U32,+lane: U32,+facts: Held.Both<{Nat.is_lt(Nat.sub(Geometry.role_digit(Tiled.Lanes{},low,(index / U32.max(inner,1) % U32.max(count,1) : U32),U32.to_nat(lane)),U32.to_nat(low)),U32.to_nat(extent)) == True{} : Bool},{Nat.is_lt(U32.to_nat(lane),8n) == True{} : Bool}>) -> {Nat.is_lt(Nat.sub(Geometry.role_digit(Tiled.Lanes{},low,(index / U32.max(inner,1) % U32.max(count,1) : U32),U32.to_nat(lane)),U32.to_nat(low)),U32.to_nat(extent)) == True{} : Bool}:
  Held.Both{+below,+small} = facts
  below

def lane_small(+low: U32,+extent: U32,+index: U32,+inner: U32,+count: U32,+lane: U32,+facts: Held.Both<{Nat.is_lt(Nat.sub(Geometry.role_digit(Tiled.Lanes{},low,(index / U32.max(inner,1) % U32.max(count,1) : U32),U32.to_nat(lane)),U32.to_nat(low)),U32.to_nat(extent)) == True{} : Bool},{Nat.is_lt(U32.to_nat(lane),8n) == True{} : Bool}>) -> {Nat.is_lt(U32.to_nat(lane),8n) == True{} : Bool}:
  Held.Both{+below,+small} = facts
  small

# ---- Every axis of a block ----

def extents_of(axes: List<&2,Tiled.Ranged>) -> List<&2,U32>:
  match axes:
    case Nil{}: []
    case Con{Tiled.Ranged{blocked,low,+extent,count,inner,target},rest}: extent <> extents_of(rest)

# Every axis of the blocks holds an index, fits half the capacity from its
# first index and counts its blocks.
def RangedOk(axes: List<&2,Tiled.Ranged>,+cap: U32) -> Data:
  match axes:
    case Nil{}: Unit
    case Con{Tiled.Ranged{Tiled.Blocked{role,source,lanes},+low,+extent,+count,inner,target},rest}:
      KernelGemm.Both<{Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
        KernelGemm.Both<{Nat.is_le(Nat.double(Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(cap)) == True{} : Bool},
          KernelGemm.Both<{count == Tiled.block_count(role,low,extent) : U32},RangedOk(rest,cap)>>>

# The digit of an axis of one block per index lies below its extent.
def shared_below(+low: U32,+extent: U32,+index: U32,+inner: U32,+count: U32,+lane: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(Nat.add(U32.to_nat(low),U32.to_nat(extent))),U32.to_nat(cap)) == True{} : Bool},+counted: {count == extent : U32}) ->
  {Nat.is_lt(Nat.sub(Nat.add(U32.to_nat(low),U32.to_nat((index / U32.max(inner,1) % U32.max(count,1) : U32))),U32.to_nat(low)),U32.to_nat(extent)) == True{} : Bool}:
  +quotient = (index / U32.max(inner,1) : U32)
  +at = (quotient % U32.max(count,1) : U32)
  +l = U32.to_nat(low)
  +e = U32.to_nat(extent)
  +a = U32.to_nat(at)
  +limit = U32.to_nat(cap)
  +divisor_word = Extrema.max_one_exact(count,e,{%Equal.sym(U32,count,extent,counted) : {U32.to_nat(_) == e : Nat} {==} : {U32.to_nat(count) == e : Nat}},positive)
  +divisor_back = Equal.sym(Nat,U32.to_nat(U32.max(count,1)),e,divisor_word)
  +extent_room = Order.transitive(Nat.double(e),Nat.double(Nat.add(l,e)),limit,Order.double_le(e,Nat.add(l,e),Order.right_le(l,e)),room)
  +at_word = {%divisor_word : {U32.to_nat(at) == Nat.mod(U32.to_nat(quotient),_) : Nat}
    U32Division.modulo(quotient,U32.max(count,1),cap,cap_ok,
      {%divisor_back : {Nat.is_le(Nat.double(_),limit) == True{} : Bool} extent_room : {Nat.is_le(Nat.double(U32.to_nat(U32.max(count,1))),limit) == True{} : Bool}},
      {%divisor_back : {Nat.is_lt(0n,_) == True{} : Bool} positive : {Nat.is_lt(0n,U32.to_nat(U32.max(count,1))) == True{} : Bool}}) :
    {U32.to_nat(at) == Nat.mod(U32.to_nat(quotient),e) : Nat}}
  %Equal.sym(Nat,Nat.sub(Nat.add(l,a),l),a,Order.subtract_shift(l,a)) : {Nat.is_lt(_,e) == True{} : Bool}
  %Equal.sym(Nat,a,Nat.mod(U32.to_nat(quotient),e),at_word) : {Nat.is_lt(_,e) == True{} : Bool}
  DivisionOrder.remainder_small(U32.to_nat(quotient),e,positive)

# Axes without the source lane axis: every lane lies past the first index of
# the piece and every digit below its extent.
def none_from(axes: List<&2,Tiled.Ranged>,+index: U32,+lane: Nat,+none: Geometry.NoLanes(Geometry.blocked_of(axes))) -> LanesFrom(axes,index,lane):
  match axes:
    case Nil{}: Unit{}
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Lanes{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest}:
      Empty.absurd(LanesFrom(Con{Tiled.Ranged{Tiled.Blocked{Tiled.Lanes{},source,lanes},low,extent,count,inner,target},rest},index,lane),none)
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Shared{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest}: KernelGemm.Both{Unit{},none_from(rest,index,lane,none)}
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Groups{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest}: KernelGemm.Both{Unit{},none_from(rest,index,lane,none)}

def none_below(axes: List<&2,Tiled.Ranged>,+index: U32,+lane: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +none: Geometry.NoLanes(Geometry.blocked_of(axes)),+ok: RangedOk(axes,cap)) -> Coordinates.DigitsBelow(extents_of(axes),box_digits(axes,index,lane)):
  match axes:
    case Nil{}: Unit{}
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Lanes{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest}:
      Empty.absurd(Coordinates.DigitsBelow(extents_of(Con{Tiled.Ranged{Tiled.Blocked{Tiled.Lanes{},source,lanes},low,extent,count,inner,target},rest}),
        box_digits(Con{Tiled.Ranged{Tiled.Blocked{Tiled.Lanes{},source,lanes},low,extent,count,inner,target},rest},index,lane)),none)
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Shared{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest}:
      KernelGemm.Both{+positive,+more} = ok
      KernelGemm.Both{+room,+tail} = more
      KernelGemm.Both{+counted,+later} = tail
      KernelGemm.Both{shared_below(low,extent,index,inner,count,lane,cap,cap_ok,positive,room,counted),none_below(rest,index,lane,cap,cap_ok,none,later)}
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Groups{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest}:
      KernelGemm.Both{+positive,+more} = ok
      KernelGemm.Both{+room,+tail} = more
      KernelGemm.Both{+counted,+later} = tail
      KernelGemm.Both{shared_below(low,extent,index,inner,count,lane,cap,cap_ok,positive,room,counted),none_below(rest,index,lane,cap,cap_ok,none,later)}

# ---- Every write of a block ----

# The lane of a write lies past the first index of the piece on the source
# lane axis.
def write_from(axes: List<&2,Tiled.Ranged>,block: Tiled.Block,+index: U32,+lane: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +most: Geometry.LanesAtMost(Geometry.blocked_of(axes)),+ok: RangedOk(axes,cap),
  +from_low: {Nat.is_le(U32.to_nat(low_of(Tiled.block_at(axes,index,block))),U32.to_nat(lane)) == True{} : Bool},
  +to_high: {Nat.is_lt(U32.to_nat(lane),U32.to_nat(high_of(Tiled.block_at(axes,index,block)))) == True{} : Bool}) ->
  LanesFrom(axes,index,U32.to_nat(lane)):
  match axes block:
    case Nil{} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}: Unit{}
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Shared{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}:
      KernelGemm.Both{+positive,+more} = ok
      KernelGemm.Both{+room,+tail} = more
      KernelGemm.Both{+counted,+later} = tail
      KernelGemm.Both{Unit{},write_from(rest,Tiled.block_axis(Tiled.Shared{},Tiled.Block{first,start,stride,block_low,block_high,reads,step},
        (index / U32.max(inner,1) % U32.max(count,1) : U32),low,extent,source,lanes,target),index,lane,cap,cap_ok,most,later,from_low,to_high)}
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Groups{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}:
      KernelGemm.Both{+positive,+more} = ok
      KernelGemm.Both{+room,+tail} = more
      KernelGemm.Both{+counted,+later} = tail
      KernelGemm.Both{Unit{},write_from(rest,Tiled.block_axis(Tiled.Groups{},Tiled.Block{first,start,stride,block_low,block_high,reads,step},
        (index / U32.max(inner,1) % U32.max(count,1) : U32),low,extent,source,lanes,target),index,lane,cap,cap_ok,most,later,from_low,to_high)}
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Lanes{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}:
      KernelGemm.Both{+positive,+more} = ok
      KernelGemm.Both{+room,+tail} = more
      KernelGemm.Both{+counted,+later} = tail
      +at = (index / U32.max(inner,1) % U32.max(count,1) : U32)
      +next = Tiled.block_axis(Tiled.Lanes{},Tiled.Block{first,start,stride,block_low,block_high,reads,step},at,low,extent,source,lanes,target)
      +final = Tiled.block_at(rest,index,next)
      +kept = kept_writes(rest,next,index,most)
      +low_kept = Equal.cong(U32 & (U32 & U32),U32,pair => Pair.fst(U32,U32,Pair.snd(U32,U32 & U32,pair)),writes_of(final),writes_of(next),kept)
      +high_kept = Equal.cong(U32 & (U32 & U32),U32,pair => Pair.snd(U32,U32,Pair.snd(U32,U32 & U32,pair)),writes_of(final),writes_of(next),kept)
      +facts = lane_facts(low,extent,index,inner,count,lane,cap,cap_ok,positive,room,counted,
        {%low_kept : {Nat.is_le(U32.to_nat(_),U32.to_nat(lane)) == True{} : Bool} from_low : {Nat.is_le(U32.to_nat(lane_low(low,at)),U32.to_nat(lane)) == True{} : Bool}},
        {%high_kept : {Nat.is_lt(U32.to_nat(lane),U32.to_nat(_)) == True{} : Bool} to_high : {Nat.is_lt(U32.to_nat(lane),U32.to_nat(lane_high(low,extent,at))) == True{} : Bool}})
      KernelGemm.Both{lane_from(low,extent,index,inner,count,lane,facts),none_from(rest,index,U32.to_nat(lane),most)}

# The digits of a write lie inside the box.
def write_below(axes: List<&2,Tiled.Ranged>,block: Tiled.Block,+index: U32,+lane: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +most: Geometry.LanesAtMost(Geometry.blocked_of(axes)),+ok: RangedOk(axes,cap),
  +from_low: {Nat.is_le(U32.to_nat(low_of(Tiled.block_at(axes,index,block))),U32.to_nat(lane)) == True{} : Bool},
  +to_high: {Nat.is_lt(U32.to_nat(lane),U32.to_nat(high_of(Tiled.block_at(axes,index,block)))) == True{} : Bool}) ->
  Coordinates.DigitsBelow(extents_of(axes),box_digits(axes,index,U32.to_nat(lane))):
  match axes block:
    case Nil{} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}: Unit{}
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Shared{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}:
      KernelGemm.Both{+positive,+more} = ok
      KernelGemm.Both{+room,+tail} = more
      KernelGemm.Both{+counted,+later} = tail
      KernelGemm.Both{shared_below(low,extent,index,inner,count,U32.to_nat(lane),cap,cap_ok,positive,room,counted),
        write_below(rest,Tiled.block_axis(Tiled.Shared{},Tiled.Block{first,start,stride,block_low,block_high,reads,step},
          (index / U32.max(inner,1) % U32.max(count,1) : U32),low,extent,source,lanes,target),index,lane,cap,cap_ok,most,later,from_low,to_high)}
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Groups{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}:
      KernelGemm.Both{+positive,+more} = ok
      KernelGemm.Both{+room,+tail} = more
      KernelGemm.Both{+counted,+later} = tail
      KernelGemm.Both{shared_below(low,extent,index,inner,count,U32.to_nat(lane),cap,cap_ok,positive,room,counted),
        write_below(rest,Tiled.block_axis(Tiled.Groups{},Tiled.Block{first,start,stride,block_low,block_high,reads,step},
          (index / U32.max(inner,1) % U32.max(count,1) : U32),low,extent,source,lanes,target),index,lane,cap,cap_ok,most,later,from_low,to_high)}
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Lanes{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}:
      KernelGemm.Both{+positive,+more} = ok
      KernelGemm.Both{+room,+tail} = more
      KernelGemm.Both{+counted,+later} = tail
      +at = (index / U32.max(inner,1) % U32.max(count,1) : U32)
      +next = Tiled.block_axis(Tiled.Lanes{},Tiled.Block{first,start,stride,block_low,block_high,reads,step},at,low,extent,source,lanes,target)
      +final = Tiled.block_at(rest,index,next)
      +kept = kept_writes(rest,next,index,most)
      +low_kept = Equal.cong(U32 & (U32 & U32),U32,pair => Pair.fst(U32,U32,Pair.snd(U32,U32 & U32,pair)),writes_of(final),writes_of(next),kept)
      +high_kept = Equal.cong(U32 & (U32 & U32),U32,pair => Pair.snd(U32,U32,Pair.snd(U32,U32 & U32,pair)),writes_of(final),writes_of(next),kept)
      +facts = lane_facts(low,extent,index,inner,count,lane,cap,cap_ok,positive,room,counted,
        {%low_kept : {Nat.is_le(U32.to_nat(_),U32.to_nat(lane)) == True{} : Bool} from_low : {Nat.is_le(U32.to_nat(lane_low(low,at)),U32.to_nat(lane)) == True{} : Bool}},
        {%high_kept : {Nat.is_lt(U32.to_nat(lane),U32.to_nat(_)) == True{} : Bool} to_high : {Nat.is_lt(U32.to_nat(lane),U32.to_nat(lane_high(low,extent,at))) == True{} : Bool}})
      KernelGemm.Both{lane_below(low,extent,index,inner,count,lane,lane_rest(low,extent,index,inner,count,lane,facts)),
        none_below(rest,index,U32.to_nat(lane),cap,cap_ok,most,later)}

# The lane of a write lies below eight.
def write_small(axes: List<&2,Tiled.Ranged>,block: Tiled.Block,+index: U32,+lane: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +most: Geometry.LanesAtMost(Geometry.blocked_of(axes)),+ok: RangedOk(axes,cap),+single: {high_of(block) == 1 : U32},
  +from_low: {Nat.is_le(U32.to_nat(low_of(Tiled.block_at(axes,index,block))),U32.to_nat(lane)) == True{} : Bool},
  +to_high: {Nat.is_lt(U32.to_nat(lane),U32.to_nat(high_of(Tiled.block_at(axes,index,block)))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(lane),8n) == True{} : Bool}:
  match axes block:
    case Nil{} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}:
      +below_one = {%single : {Nat.is_lt(U32.to_nat(lane),U32.to_nat(_)) == True{} : Bool} to_high : {Nat.is_lt(U32.to_nat(lane),1n) == True{} : Bool}}
      BoxDigits.follow_strict(U32.to_nat(lane),8n,Order.transitive(1n+U32.to_nat(lane),1n,8n,Order.strict_follow(U32.to_nat(lane),1n,below_one),{==}))
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Shared{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}:
      KernelGemm.Both{+positive,+more} = ok
      KernelGemm.Both{+room,+tail} = more
      KernelGemm.Both{+counted,+later} = tail
      write_small(rest,Tiled.block_axis(Tiled.Shared{},Tiled.Block{first,start,stride,block_low,block_high,reads,step},
        (index / U32.max(inner,1) % U32.max(count,1) : U32),low,extent,source,lanes,target),index,lane,cap,cap_ok,most,later,single,from_low,to_high)
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Groups{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}:
      KernelGemm.Both{+positive,+more} = ok
      KernelGemm.Both{+room,+tail} = more
      KernelGemm.Both{+counted,+later} = tail
      write_small(rest,Tiled.block_axis(Tiled.Groups{},Tiled.Block{first,start,stride,block_low,block_high,reads,step},
        (index / U32.max(inner,1) % U32.max(count,1) : U32),low,extent,source,lanes,target),index,lane,cap,cap_ok,most,later,single,from_low,to_high)
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Lanes{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}:
      KernelGemm.Both{+positive,+more} = ok
      KernelGemm.Both{+room,+tail} = more
      KernelGemm.Both{+counted,+later} = tail
      +at = (index / U32.max(inner,1) % U32.max(count,1) : U32)
      +next = Tiled.block_axis(Tiled.Lanes{},Tiled.Block{first,start,stride,block_low,block_high,reads,step},at,low,extent,source,lanes,target)
      +final = Tiled.block_at(rest,index,next)
      +kept = kept_writes(rest,next,index,most)
      +low_kept = Equal.cong(U32 & (U32 & U32),U32,pair => Pair.fst(U32,U32,Pair.snd(U32,U32 & U32,pair)),writes_of(final),writes_of(next),kept)
      +high_kept = Equal.cong(U32 & (U32 & U32),U32,pair => Pair.snd(U32,U32,Pair.snd(U32,U32 & U32,pair)),writes_of(final),writes_of(next),kept)
      +facts = lane_facts(low,extent,index,inner,count,lane,cap,cap_ok,positive,room,counted,
        {%low_kept : {Nat.is_le(U32.to_nat(_),U32.to_nat(lane)) == True{} : Bool} from_low : {Nat.is_le(U32.to_nat(lane_low(low,at)),U32.to_nat(lane)) == True{} : Bool}},
        {%high_kept : {Nat.is_lt(U32.to_nat(lane),U32.to_nat(_)) == True{} : Bool} to_high : {Nat.is_lt(U32.to_nat(lane),U32.to_nat(lane_high(low,extent,at))) == True{} : Bool}})
      lane_small(low,extent,index,inner,count,lane,lane_rest(low,extent,index,inner,count,lane,facts))

# ---- The target digits of a write ----

def lows_of(axes: List<&2,Tiled.Ranged>) -> List<&2,Nat>:
  match axes:
    case Nil{}: []
    case Con{Tiled.Ranged{blocked,+low,extent,count,inner,target},rest}: U32.to_nat(low) <> lows_of(rest)

def role_low(role: Tiled.Role,+low: U32,+index: U32,+lane: Nat,+from: LaneFrom(role,low,index,lane)) ->
  {Nat.is_le(U32.to_nat(low),Geometry.role_digit(role,low,index,lane)) == True{} : Bool}:
  match role:
    case Tiled.Shared{}: Order.le_add(U32.to_nat(low),U32.to_nat(index))
    case Tiled.Groups{}: Order.le_add(U32.to_nat(low),U32.to_nat(index))
    case Tiled.Lanes{}: from

# The target digits of a block at a lane are the first indices of the piece
# plus the box digits.
def digits_added(axes: List<&2,Tiled.Ranged>,+index: U32,+lane: Nat,+from: LanesFrom(axes,index,lane)) ->
  {Geometry.block_digits(axes,index,lane) == BoxDigits.added(lows_of(axes),box_digits(axes,index,lane)) : List<&2,Nat>}:
  match axes:
    case Nil{}: {==}
    case Con{Tiled.Ranged{Tiled.Blocked{+role,+source,+lanes},+low,+extent,+count,+inner,+target},+rest}:
      KernelGemm.Both{+here,+later} = from
      +at = (index / U32.max(inner,1) % U32.max(count,1) : U32)
      +digit = Geometry.role_digit(role,low,at,lane)
      %Equal.sym(Nat,Nat.add(U32.to_nat(low),Nat.sub(digit,U32.to_nat(low))),digit,Order.restore_shift(U32.to_nat(low),digit,role_low(role,low,at,lane,here))) :
        {Con{digit,Geometry.block_digits(rest,index,lane)} == Con{_,BoxDigits.added(lows_of(rest),box_digits(rest,index,lane))} : List<&2,Nat>}
      %digits_added(rest,index,lane,later) : {Con{digit,Geometry.block_digits(rest,index,lane)} == Con{digit,_} : List<&2,Nat>}
      {==}

# ---- The axes of the blocks of a piece ----

def ranged_blocked(blocked: List<&2,Tiled.Blocked>,lows: List<&2,U32>,extents: List<&2,U32>,
  +same_blocked: {List.length(&2,Tiled.Blocked,blocked) == List.length(&2,U32,extents) : Nat},+same_lows: {List.length(&2,U32,lows) == List.length(&2,U32,extents) : Nat}) ->
  {Geometry.blocked_of(Tiled.ranged_axes(blocked,lows,extents,Tasks.strides_of(extents))) == blocked : List<&2,Tiled.Blocked>}:
  match blocked lows extents:
    case Con{+head,+more} Con{+low,+later} Con{+extent,+rest}:
      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,rest}),Con{Grid.product(rest),Grid.strides(rest)},GridLaws.strides_con(extent,rest)) :
        {Geometry.blocked_of(Tiled.ranged_axes(Con{head,more},Con{low,later},Con{extent,rest},_)) == Con{head,more} : List<&2,Tiled.Blocked>}
      Equal.cong(List<&2,Tiled.Blocked>,List<&2,Tiled.Blocked>,tail => head <> tail,Geometry.blocked_of(Tiled.ranged_axes(more,later,rest,Tasks.strides_of(rest))),more,
        ranged_blocked(more,later,rest,Equal.cong(Nat,Nat,count => BoxDigits.before(count),1n+List.length(&2,Tiled.Blocked,more),1n+List.length(&2,U32,rest),same_blocked),
          Equal.cong(Nat,Nat,count => BoxDigits.before(count),1n+List.length(&2,U32,later),1n+List.length(&2,U32,rest),same_lows)))
    case Nil{} +lows Nil{}: {==}
    case Nil{} +lows Con{+extent,+rest}: Empty.absurd({Nil{} == Nil{} : List<&2,Tiled.Blocked>},BoxDigits.zero_successor(List.length(&2,U32,rest),same_blocked))
    case Con{+head,+more} +lows Nil{}:
      Empty.absurd({Geometry.blocked_of(Tiled.ranged_axes(Con{head,more},lows,Nil{},Nil{})) == Con{head,more} : List<&2,Tiled.Blocked>},
        BoxDigits.zero_successor(List.length(&2,Tiled.Blocked,more),Equal.sym(Nat,1n+List.length(&2,Tiled.Blocked,more),0n,same_blocked)))
    case Con{+head,+more} Nil{} Con{+extent,+rest}:
      Empty.absurd({Nil{} == Con{head,more} : List<&2,Tiled.Blocked>},BoxDigits.zero_successor(List.length(&2,U32,rest),same_lows))

def ranged_lows(blocked: List<&2,Tiled.Blocked>,lows: List<&2,U32>,extents: List<&2,U32>,
  +same_blocked: {List.length(&2,Tiled.Blocked,blocked) == List.length(&2,U32,extents) : Nat},+same_lows: {List.length(&2,U32,lows) == List.length(&2,U32,extents) : Nat}) ->
  {lows_of(Tiled.ranged_axes(blocked,lows,extents,Tasks.strides_of(extents))) == BoxDigits.words_nat(lows) : List<&2,Nat>}:
  match blocked lows extents:
    case Con{+head,+more} Con{+low,+later} Con{+extent,+rest}:
      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,rest}),Con{Grid.product(rest),Grid.strides(rest)},GridLaws.strides_con(extent,rest)) :
        {lows_of(Tiled.ranged_axes(Con{head,more},Con{low,later},Con{extent,rest},_)) == BoxDigits.words_nat(Con{low,later}) : List<&2,Nat>}
      Equal.cong(List<&2,Nat>,List<&2,Nat>,tail => U32.to_nat(low) <> tail,lows_of(Tiled.ranged_axes(more,later,rest,Tasks.strides_of(rest))),BoxDigits.words_nat(later),
        ranged_lows(more,later,rest,Equal.cong(Nat,Nat,count => BoxDigits.before(count),1n+List.length(&2,Tiled.Blocked,more),1n+List.length(&2,U32,rest),same_blocked),
          Equal.cong(Nat,Nat,count => BoxDigits.before(count),1n+List.length(&2,U32,later),1n+List.length(&2,U32,rest),same_lows)))
    case +blocked Nil{} Nil{}: {==}
    case +blocked Con{+low,+later} Nil{}:
      Empty.absurd({lows_of(Tiled.ranged_axes(blocked,Con{low,later},Nil{},Nil{})) == BoxDigits.words_nat(Con{low,later}) : List<&2,Nat>},
        BoxDigits.zero_successor(List.length(&2,U32,later),Equal.sym(Nat,1n+List.length(&2,U32,later),0n,same_lows)))
    case +blocked Nil{} Con{+extent,+rest}: Empty.absurd({lows_of(Tiled.ranged_axes(blocked,Nil{},Con{extent,rest},Tasks.strides_of(Con{extent,rest}))) == Nil{} : List<&2,Nat>},
      BoxDigits.zero_successor(List.length(&2,U32,rest),same_lows))
    case Nil{} Con{+low,+later} Con{+extent,+rest}:
      Empty.absurd({Nil{} == BoxDigits.words_nat(Con{low,later}) : List<&2,Nat>},BoxDigits.zero_successor(List.length(&2,U32,rest),same_blocked))

def ranged_extents(blocked: List<&2,Tiled.Blocked>,lows: List<&2,U32>,extents: List<&2,U32>,
  +same_blocked: {List.length(&2,Tiled.Blocked,blocked) == List.length(&2,U32,extents) : Nat},+same_lows: {List.length(&2,U32,lows) == List.length(&2,U32,extents) : Nat}) ->
  {extents_of(Tiled.ranged_axes(blocked,lows,extents,Tasks.strides_of(extents))) == extents : List<&2,U32>}:
  match blocked lows extents:
    case Con{+head,+more} Con{+low,+later} Con{+extent,+rest}:
      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,rest}),Con{Grid.product(rest),Grid.strides(rest)},GridLaws.strides_con(extent,rest)) :
        {extents_of(Tiled.ranged_axes(Con{head,more},Con{low,later},Con{extent,rest},_)) == Con{extent,rest} : List<&2,U32>}
      Equal.cong(List<&2,U32>,List<&2,U32>,tail => extent <> tail,extents_of(Tiled.ranged_axes(more,later,rest,Tasks.strides_of(rest))),rest,
        ranged_extents(more,later,rest,Equal.cong(Nat,Nat,count => BoxDigits.before(count),1n+List.length(&2,Tiled.Blocked,more),1n+List.length(&2,U32,rest),same_blocked),
          Equal.cong(Nat,Nat,count => BoxDigits.before(count),1n+List.length(&2,U32,later),1n+List.length(&2,U32,rest),same_lows)))
    case +blocked +lows Nil{}: {==}
    case Nil{} +lows Con{+extent,+rest}:
      Empty.absurd({extents_of(Tiled.ranged_axes(Nil{},lows,Con{extent,rest},Tasks.strides_of(Con{extent,rest}))) == Con{extent,rest} : List<&2,U32>},
        BoxDigits.zero_successor(List.length(&2,U32,rest),same_blocked))
    case Con{+head,+more} Nil{} Con{+extent,+rest}:
      Empty.absurd({extents_of(Tiled.ranged_axes(Con{head,more},Nil{},Con{extent,rest},Tasks.strides_of(Con{extent,rest}))) == Con{extent,rest} : List<&2,U32>},
        BoxDigits.zero_successor(List.length(&2,U32,rest),same_lows))

def ranged_targets(blocked: List<&2,Tiled.Blocked>,lows: List<&2,U32>,extents: List<&2,U32>,
  +same_blocked: {List.length(&2,Tiled.Blocked,blocked) == List.length(&2,U32,extents) : Nat},+same_lows: {List.length(&2,U32,lows) == List.length(&2,U32,extents) : Nat}) ->
  {targets_of(Tiled.ranged_axes(blocked,lows,extents,Tasks.strides_of(extents))) == Tasks.strides_of(extents) : List<&2,U32>}:
  match blocked lows extents:
    case Con{+head,+more} Con{+low,+later} Con{+extent,+rest}:
      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,rest}),Con{Grid.product(rest),Grid.strides(rest)},GridLaws.strides_con(extent,rest)) :
        {targets_of(Tiled.ranged_axes(Con{head,more},Con{low,later},Con{extent,rest},_)) == _ : List<&2,U32>}
      Equal.cong(List<&2,U32>,List<&2,U32>,tail => Grid.product(rest) <> tail,targets_of(Tiled.ranged_axes(more,later,rest,Tasks.strides_of(rest))),Tasks.strides_of(rest),
        ranged_targets(more,later,rest,Equal.cong(Nat,Nat,count => BoxDigits.before(count),1n+List.length(&2,Tiled.Blocked,more),1n+List.length(&2,U32,rest),same_blocked),
          Equal.cong(Nat,Nat,count => BoxDigits.before(count),1n+List.length(&2,U32,later),1n+List.length(&2,U32,rest),same_lows)))
    case +blocked +lows Nil{}: {==}
    case Nil{} +lows Con{+extent,+rest}:
      Empty.absurd({targets_of(Tiled.ranged_axes(Nil{},lows,Con{extent,rest},Tasks.strides_of(Con{extent,rest}))) == Tasks.strides_of(Con{extent,rest}) : List<&2,U32>},
        BoxDigits.zero_successor(List.length(&2,U32,rest),same_blocked))
    case Con{+head,+more} Nil{} Con{+extent,+rest}:
      Empty.absurd({targets_of(Tiled.ranged_axes(Con{head,more},Nil{},Con{extent,rest},Tasks.strides_of(Con{extent,rest}))) == Tasks.strides_of(Con{extent,rest}) : List<&2,U32>},
        BoxDigits.zero_successor(List.length(&2,U32,rest),same_lows))

# ---- The axes of the blocks of a piece fit ----

def ranged_ok(blocked: List<&2,Tiled.Blocked>,lows: List<&2,U32>,extents: List<&2,U32>,+cap: U32,+positive: BoxDigits.Positive(extents),
  +rooms: BoxDigits.Rooms(BoxDigits.words_nat(lows),extents,cap)) -> RangedOk(Tiled.ranged_axes(blocked,lows,extents,Tasks.strides_of(extents)),cap):
  match blocked lows extents:
    case Con{Tiled.Blocked{+role,+source,+lanes},+more} Con{+low,+later} Con{+extent,+rest}:
      KernelGemm.Both{+here,+after} = positive
      KernelGemm.Both{+room,+others} = rooms
      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,rest}),Con{Grid.product(rest),Grid.strides(rest)},GridLaws.strides_con(extent,rest)) :
        RangedOk(Tiled.ranged_axes(Con{Tiled.Blocked{role,source,lanes},more},Con{low,later},Con{extent,rest},_),cap)
      KernelGemm.Both{here,KernelGemm.Both{room,KernelGemm.Both{{==},ranged_ok(more,later,rest,cap,after,others)}}}
    case Nil{} +lows +extents: Unit{}
    case Con{+head,+more} Nil{} +extents: Unit{}
    case Con{+head,+more} Con{+low,+later} Nil{}: Unit{}

# ---- A write of a block of a piece ----

def first_block() -> Tiled.Block:
  Tiled.Block{0,0,0,0,1,1,0}

# The block axes of the piece with lows at its first cell and the extents and
# strides of its box.
def piece_axes(+blocked: List<&2,Tiled.Blocked>,+lows: List<&2,U32>,+boxes: List<&2,U32>) -> List<&2,Tiled.Ranged>:
  Tiled.ranged_axes(blocked,lows,boxes,Tasks.strides_of(boxes))

# The cell a write lands on.
def write_address(+axes: List<&2,Tiled.Ranged>,+index: U32,+lane: U32) -> U32:
  +block = Tiled.block_at(axes,index,first_block())
  (target_of(block) + ((lane - low_of(block) : U32) * stride_of(block) : U32) : U32)

# A write of a block of a piece at a lane past its first and before its last
# written lane: it lands on the number of its box digits, those lie inside the
# box, its target digits are the first indices plus its box digits, and the
# lane lies below eight.
def WriteFacts(+blocked: List<&2,Tiled.Blocked>,+lows: List<&2,U32>,+boxes: List<&2,U32>,+index: U32,+lane: U32) -> Data:
  Held.Both<{write_address(piece_axes(blocked,lows,boxes),index,lane) ==
      U32.from_nat(ShapeLaws.digits_value(boxes,box_digits(piece_axes(blocked,lows,boxes),index,U32.to_nat(lane)))) : U32},
    Held.Both<Coordinates.DigitsBelow(boxes,box_digits(piece_axes(blocked,lows,boxes),index,U32.to_nat(lane))),
      Held.Both<{Geometry.block_digits(piece_axes(blocked,lows,boxes),index,U32.to_nat(lane)) ==
          BoxDigits.added(BoxDigits.words_nat(lows),box_digits(piece_axes(blocked,lows,boxes),index,U32.to_nat(lane))) : List<&2,Nat>},
        {Nat.is_lt(U32.to_nat(lane),8n) == True{} : Bool}>>>

def piece_write(+blocked: List<&2,Tiled.Blocked>,+lows: List<&2,U32>,+boxes: List<&2,U32>,+index: U32,+lane: U32,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +same_blocked: {List.length(&2,Tiled.Blocked,blocked) == List.length(&2,U32,boxes) : Nat},+same_lows: {List.length(&2,U32,lows) == List.length(&2,U32,boxes) : Nat},
  +most: Geometry.LanesAtMost(blocked),+positive: BoxDigits.Positive(boxes),+rooms: BoxDigits.Rooms(BoxDigits.words_nat(lows),boxes,cap),
  +from_low: {Nat.is_le(U32.to_nat(low_of(Tiled.block_at(piece_axes(blocked,lows,boxes),index,first_block()))),U32.to_nat(lane)) == True{} : Bool},
  +to_high: {Nat.is_lt(U32.to_nat(lane),U32.to_nat(high_of(Tiled.block_at(piece_axes(blocked,lows,boxes),index,first_block())))) == True{} : Bool}) ->
  WriteFacts(blocked,lows,boxes,index,lane):
  +axes = piece_axes(blocked,lows,boxes)
  +digits = box_digits(axes,index,U32.to_nat(lane))
  +axes_most = {%Equal.sym(List<&2,Tiled.Blocked>,Geometry.blocked_of(axes),blocked,ranged_blocked(blocked,lows,boxes,same_blocked,same_lows)) :
      Geometry.LanesAtMost(_)
    most : Geometry.LanesAtMost(Geometry.blocked_of(axes))}
  +ok = ranged_ok(blocked,lows,boxes,cap,positive,rooms)
  +from = write_from(axes,first_block(),index,lane,cap,cap_ok,axes_most,ok,from_low,to_high)
  +below = {%ranged_extents(blocked,lows,boxes,same_blocked,same_lows) : Coordinates.DigitsBelow(_,digits)
    write_below(axes,first_block(),index,lane,cap,cap_ok,axes_most,ok,from_low,to_high) : Coordinates.DigitsBelow(boxes,digits)}
  +small = write_small(axes,first_block(),index,lane,cap,cap_ok,axes_most,ok,{==},from_low,to_high)
  +address = Equal.trans(U32,write_address(axes,index,lane),(0 + write_sum(axes,index,lane) : U32),U32.from_nat(ShapeLaws.digits_value(boxes,digits)),
    write_cell(axes,first_block(),index,lane,axes_most,{==}),
    Equal.trans(U32,(0 + write_sum(axes,index,lane) : U32),write_sum(axes,index,lane),U32.from_nat(ShapeLaws.digits_value(boxes,digits)),
      Arithmetic.add_zero_left(write_sum(axes,index,lane)),
      Equal.trans(U32,write_sum(axes,index,lane),BoxDigits.stepped_digits(digits,targets_of(axes)),U32.from_nat(ShapeLaws.digits_value(boxes,digits)),
        write_digits(axes,index,lane,from),
        {%Equal.sym(List<&2,U32>,targets_of(axes),Tasks.strides_of(boxes),ranged_targets(blocked,lows,boxes,same_blocked,same_lows)) : {BoxDigits.stepped_digits(digits,_) == U32.from_nat(ShapeLaws.digits_value(boxes,digits)) : U32}
          Equal.sym(U32,U32.from_nat(ShapeLaws.digits_value(boxes,digits)),BoxDigits.stepped_digits(digits,Tasks.strides_of(boxes)),BoxDigits.digits_stepped(boxes,digits)) :
          {BoxDigits.stepped_digits(digits,targets_of(axes)) == U32.from_nat(ShapeLaws.digits_value(boxes,digits)) : U32}})))
  +added = {%ranged_lows(blocked,lows,boxes,same_blocked,same_lows) : {Geometry.block_digits(axes,index,U32.to_nat(lane)) == BoxDigits.added(_,digits) : List<&2,Nat>}
    digits_added(axes,index,U32.to_nat(lane),from) : {Geometry.block_digits(axes,index,U32.to_nat(lane)) == BoxDigits.added(BoxDigits.words_nat(lows),digits) : List<&2,Nat>}}
  Held.Both{address,Held.Both{below,Held.Both{added,small}}}

# ---- Every block writes a lane ----

# A value below a positive size lies at or below the last index.
def below_last(+value: Nat,+size: Nat,+below: {Nat.is_lt(value,size) == True{} : Bool}) -> {Nat.is_le(value,Nat.sub(size,1n)) == True{} : Bool}:
  Division.lt_succ_le(value,Nat.sub(size,1n),
    {%Equal.sym(Nat,1n+Nat.sub(size,1n),size,Order.positive_has_last(size,Order.le_lt_trans(0n,value,size,Order.zero_le(value),below))) :
        {Nat.is_lt(value,_) == True{} : Bool}
      below : {Nat.is_lt(value,1n+Nat.sub(size,1n)) == True{} : Bool}})

def max_below(+left: Nat,+right: Nat,+limit: Nat,+left_below: {Nat.is_lt(left,limit) == True{} : Bool},+right_below: {Nat.is_lt(right,limit) == True{} : Bool}) ->
  {Nat.is_lt(Nat.max(left,right),limit) == True{} : Bool}:
  Order.le_lt_trans(Nat.max(left,right),Nat.sub(limit,1n),limit,
    NatExtrema.maximum_upper(left,right,Nat.sub(limit,1n),below_last(left,limit,left_below),below_last(right,limit,right_below)),
    Order.last_below(limit,Order.le_lt_trans(0n,left,limit,Order.zero_le(left),left_below)))

# A lane block inside the lanes of its axis writes a lane.
def lane_open(+low: U32,+extent: U32,+at: U32,+positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},+words: LaneWords(low,extent,at)) ->
  {Nat.is_lt(U32.to_nat(lane_low(low,at)),U32.to_nat(lane_high(low,extent,at))) == True{} : Bool}:
  Held.Both{+low_word,Held.Both{+high_word,+first_below}} = words
  +l = U32.to_nat(low)
  +e = U32.to_nat(extent)
  +a = U32.to_nat(at)
  +sum = Nat.add(l,e)
  +q = U32.to_nat((low / 8 : U32))
  +f = Nat.mul(Nat.add(q,a),8n)
  +s = Nat.max(l,f)
  +below_word = U32Division.divide(low,8,2147483648,{==},{==},{==})
  +low_sum = {%Addition.zero_right(l) : {Nat.is_lt(_,sum) == True{} : Bool} Order.offset_lt(l,0n,e,positive) : {Nat.is_lt(l,sum) == True{} : Bool}}
  +f_le = NatExtrema.maximum_right(l,f)
  +start_sum = max_below(l,f,sum,low_sum,first_below)
  +before_end = Equal.trans(Bool,Nat.is_lt(Nat.sub(s,f),Nat.sub(sum,f)),Nat.is_lt(Nat.add(f,Nat.sub(s,f)),sum),True{},Order.subtraction_lt(sum,f,Nat.sub(s,f)),
    {%Equal.sym(Nat,Nat.add(f,Nat.sub(s,f)),s,Order.restore_shift(f,s,f_le)) : {Nat.is_lt(_,sum) == True{} : Bool} start_sum :
      {Nat.is_lt(Nat.add(f,Nat.sub(s,f)),sum) == True{} : Bool}})
  +g = Nat.mul(Nat.div(l,8n),8n)
  +group_le = RowMajor.mul_le(Nat.div(l,8n),Nat.add(Nat.div(l,8n),a),8n,Order.le_add(Nat.div(l,8n),a))
  +low_next = {%Equal.sym(Nat,q,Nat.div(l,8n),below_word) : {Nat.is_lt(l,Nat.add(Nat.mul(Nat.add(_,a),8n),8n)) == True{} : Bool}
    Bounds.nat_trans(l,Nat.add(8n,g),Nat.add(Nat.mul(Nat.add(Nat.div(l,8n),a),8n),8n),DivisionOrder.next_above(l,8n,{==}),
      {%Order.commutative(8n,Nat.mul(Nat.add(Nat.div(l,8n),a),8n)) : {Nat.is_le(Nat.add(8n,g),_) == True{} : Bool}
        Order.offset_le(8n,g,Nat.mul(Nat.add(Nat.div(l,8n),a),8n),group_le) : {Nat.is_le(Nat.add(8n,g),Nat.add(Nat.mul(Nat.add(Nat.div(l,8n),a),8n),8n)) == True{} : Bool}}) :
    {Nat.is_lt(l,Nat.add(f,8n)) == True{} : Bool}}
  +f_next = {%Addition.zero_right(f) : {Nat.is_lt(_,Nat.add(f,8n)) == True{} : Bool} Order.offset_lt(f,0n,8n,{==}) : {Nat.is_lt(f,Nat.add(f,8n)) == True{} : Bool}}
  +inside_lanes = Order.sub_lt(s,f,8n,f_le,max_below(l,f,Nat.add(f,8n),low_next,f_next))
  %Equal.sym(Nat,U32.to_nat(lane_low(low,at)),Nat.sub(s,f),low_word) : {Nat.is_lt(_,U32.to_nat(lane_high(low,extent,at))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(lane_high(low,extent,at)),Nat.min(Nat.sub(sum,f),8n),high_word) : {Nat.is_lt(Nat.sub(s,f),_) == True{} : Bool}
  NatExtrema.minimum_strict_lower(Nat.sub(sum,f),8n,Nat.sub(s,f),before_end,inside_lanes)

# Every block writes a lane: the lanes of the lane block of the source lane
# axis, or the one lane every block starts with.
def block_lane(axes: List<&2,Tiled.Ranged>,block: Tiled.Block,+index: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +most: Geometry.LanesAtMost(Geometry.blocked_of(axes)),+ok: RangedOk(axes,cap),
  +opened: {Nat.is_lt(U32.to_nat(low_of(block)),U32.to_nat(high_of(block))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(low_of(Tiled.block_at(axes,index,block))),U32.to_nat(high_of(Tiled.block_at(axes,index,block)))) == True{} : Bool}:
  match axes block:
    case Nil{} Tiled.Block{+first,+begin,+stride,+block_low,+block_high,+reads,+step}: opened
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Shared{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Tiled.Block{+first,+begin,+stride,+block_low,+block_high,+reads,+step}:
      KernelGemm.Both{+positive,+more} = ok
      KernelGemm.Both{+room,+tail} = more
      KernelGemm.Both{+counted,+later} = tail
      block_lane(rest,Tiled.block_axis(Tiled.Shared{},Tiled.Block{first,begin,stride,block_low,block_high,reads,step},(index / U32.max(inner,1) % U32.max(count,1) : U32),
        low,extent,source,lanes,target),index,cap,cap_ok,most,later,opened)
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Groups{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Tiled.Block{+first,+begin,+stride,+block_low,+block_high,+reads,+step}:
      KernelGemm.Both{+positive,+more} = ok
      KernelGemm.Both{+room,+tail} = more
      KernelGemm.Both{+counted,+later} = tail
      block_lane(rest,Tiled.block_axis(Tiled.Groups{},Tiled.Block{first,begin,stride,block_low,block_high,reads,step},(index / U32.max(inner,1) % U32.max(count,1) : U32),
        low,extent,source,lanes,target),index,cap,cap_ok,most,later,opened)
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Lanes{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Tiled.Block{+first,+begin,+stride,+block_low,+block_high,+reads,+step}:
      KernelGemm.Both{+positive,+more} = ok
      KernelGemm.Both{+room,+tail} = more
      KernelGemm.Both{+counted,+later} = tail
      +at = (index / U32.max(inner,1) % U32.max(count,1) : U32)
      +next = Tiled.block_axis(Tiled.Lanes{},Tiled.Block{first,begin,stride,block_low,block_high,reads,step},at,low,extent,source,lanes,target)
      +final = Tiled.block_at(rest,index,next)
      +kept = kept_writes(rest,next,index,most)
      +low_kept = Equal.cong(U32 & (U32 & U32),U32,pair => Pair.fst(U32,U32,Pair.snd(U32,U32 & U32,pair)),writes_of(final),writes_of(next),kept)
      +high_kept = Equal.cong(U32 & (U32 & U32),U32,pair => Pair.snd(U32,U32,Pair.snd(U32,U32 & U32,pair)),writes_of(final),writes_of(next),kept)
      %Equal.sym(U32,low_of(final),lane_low(low,at),low_kept) : {Nat.is_lt(U32.to_nat(_),U32.to_nat(high_of(final))) == True{} : Bool}
      %Equal.sym(U32,high_of(final),lane_high(low,extent,at),high_kept) : {Nat.is_lt(U32.to_nat(lane_low(low,at)),U32.to_nat(_)) == True{} : Bool}
      lane_open(low,extent,at,positive,lane_block_words(low,extent,index,inner,count,cap,cap_ok,positive,room,lane_count(low,extent,count,cap,cap_ok,positive,room,counted)))
