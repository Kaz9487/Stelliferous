# The parts that place a source in a target piece read source cells the
# reads of the source keep: every box axis of a part reads a segment of its
# axis, so a cell of the part reads, axis by axis, the index the read keeps
# for its target coordinate.

import Base
import ../tensor_tasks.bend as Tasks
import ../tensor_grid.bend as Grid
import ../tensor_layout.bend as Layout
import ./tensor_move_readback.bend as Held
import ./tensor_box_parts.bend as BoxParts
import ./tensor_placements.bend as Placements
import ./tensor_placing_reads.bend as Reads
import ./tensor_task_reads.bend as TaskReads
import ./tensor_view_part.bend as ViewPart
import ./tensor_box_digits.bend as Digits
import ./tensor_tile_coordinates.bend as Coordinates
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_arithmetic.bend as WordArithmetic
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./nat_to_u32_bounds.bend as Bounds
import ../kernel_gemm.bend as KernelGemm
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ./word_encoding.bend as WordEncoding
import ./tensor_move_parts.bend as MoveParts
import ./boolean_logic.bend as Logic
import ../tensor_move.bend as Move
import ./tensor_view_reads.bend as ViewReads
import ./tensor_components.bend as Components
import ./tensor_placing_gather.bend as Gathered
import ./tensor_move_part.bend as Part
import ./tensor_view_gather.bend as ViewGather
import ./tensor_grid_box.bend as Box
import ./tensor_target_cells.bend as Targets
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_moved.bend as Moved
import ../tensor.bend as Tensors
import ./machine_limit_bounds.bend as Limits
import ../machine_limits.bend as Machine
import ./nat_extrema.bend as NatExtrema
import ./word_subtraction.bend as Subtraction
import ./tensor_grid_pieces.bend as GridLaws

# ---- A box axis reads a segment ----

# A box axis of a piece reads a segment of its axis: the segment holds the
# target indices of the box from where it starts and ends inside cap, and the
# cut steps and starts the source as the segment reads it.
type BoxRead<-count: U32,-source_stride: U32,-source_offset: U32,-start: Nat,-low: U32,-stride: U32,-shift: U32,-bound: U32,-clamped: Bool,-cap: U32> is Data:
  BoxRead{first: U32,length: U32,from: U32,step: U32,read: Reads.SegmentRead<first,length,from,step,shift,bound,clamped>,
    stepped: {source_stride == (step * stride : U32) : U32},
    offset: {source_offset == ((from + ((low + U32.from_nat(start) : U32) - first) * step : U32) * stride : U32) : U32},
    after: {Nat.is_le(U32.to_nat(first),Nat.add(U32.to_nat(low),start)) == True{} : Bool},
    before: {Nat.is_le(Nat.add(Nat.add(U32.to_nat(low),start),U32.to_nat(count)),Nat.add(U32.to_nat(first),U32.to_nat(length))) == True{} : Bool},
    ends: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(length)),U32.to_nat(cap)) == True{} : Bool}}

# A source index from a start moved by a digit, stepped and strided, is the
# strided start plus the digit times the step and the stride.
def shifted_word(+from: U32,+start: U32,+first: U32,+step: U32,+stride: U32,+digit: U32) ->
  {((from + (start - first) * step : U32) * stride + digit * (step * stride : U32) : U32) == ((from + ((start + digit : U32) - first) * step : U32) * stride : U32) : U32}:
  +a = (from + (start - first) * step : U32)
  +inner = Equal.trans(U32,(a + digit * step : U32),(from + ((start - first : U32) * step + digit * step : U32) : U32),(from + ((start + digit : U32) - first) * step : U32),
    Arithmetic.add_associative(from,((start - first : U32) * step : U32),(digit * step : U32)),
    Equal.cong(U32,U32,w => (from + w : U32),((start - first : U32) * step + digit * step : U32),(((start + digit : U32) - first : U32) * step : U32),
      Equal.trans(U32,((start - first : U32) * step + digit * step : U32),(((start - first : U32) + digit : U32) * step : U32),(((start + digit : U32) - first : U32) * step : U32),
        Equal.sym(U32,(((start - first : U32) + digit : U32) * step : U32),((start - first : U32) * step + digit * step : U32),Arithmetic.multiply_additive((start - first : U32),digit,step)),
        Equal.cong(U32,U32,w => (w * step : U32),((start - first : U32) + digit : U32),((start + digit : U32) - first : U32),Arithmetic.add_after_subtract(start,first,digit)))))
  Equal.trans(U32,(a * stride + digit * (step * stride : U32) : U32),(a * stride + (digit * step : U32) * stride : U32),((from + ((start + digit : U32) - first) * step : U32) * stride : U32),
    Equal.cong(U32,U32,w => (a * stride + w : U32),(digit * (step * stride : U32) : U32),((digit * step : U32) * stride : U32),
      Equal.sym(U32,((digit * step : U32) * stride : U32),(digit * (step * stride : U32) : U32),Arithmetic.multiply_associative(digit,step,stride))),
    Equal.trans(U32,(a * stride + (digit * step : U32) * stride : U32),((a + digit * step : U32) * stride : U32),((from + ((start + digit : U32) - first) * step : U32) * stride : U32),
      Equal.sym(U32,((a + digit * step : U32) * stride : U32),(a * stride + (digit * step : U32) * stride : U32),Arithmetic.multiply_additive(a,(digit * step : U32),stride)),
      Equal.cong(U32,U32,w => (w * stride : U32),(a + digit * step : U32),(from + ((start + digit : U32) - first) * step : U32),inner)))

def box_cell_kept(+count: U32,+source_stride: U32,+source_offset: U32,+start: Nat,+low: U32,+stride: U32,+shift: U32,+bound: U32,+clamped: Bool,+first: U32,+length: U32,+from: U32,+step: U32,+digit: Nat,
  +stepped: {source_stride == (step * stride : U32) : U32},
  +offset: {source_offset == ((from + ((low + U32.from_nat(start) : U32) - first) * step : U32) * stride : U32) : U32},
  +kept: Held.Both<Reads.KeptAt(clamped,shift,bound,(low + U32.from_nat(Nat.add(start,digit)) : U32)),{U32.to_nat((from + ((low + U32.from_nat(Nat.add(start,digit)) : U32) - first) * step : U32)) == Reads.kept_index(clamped,U32.to_nat(shift),U32.to_nat(bound),U32.to_nat((low + U32.from_nat(Nat.add(start,digit)) : U32))) : Nat}>) ->
  Held.Both<Reads.KeptAt(clamped,shift,bound,(low + U32.from_nat(Nat.add(start,digit)) : U32)),{(source_offset + U32.from_nat(digit) * source_stride : U32) == (U32.from_nat(Reads.kept_index(clamped,U32.to_nat(shift),U32.to_nat(bound),U32.to_nat((low + U32.from_nat(Nat.add(start,digit)) : U32)))) * stride : U32) : U32}>:
  Held.Both{+at,+word} = kept
  +u = (low + U32.from_nat(start) : U32)
  +d = U32.from_nat(digit)
  +index = (low + U32.from_nat(Nat.add(start,digit)) : U32)
  +w = (from + (index - first) * step : U32)
  +moved = Equal.trans(U32,(u + d : U32),(low + (U32.from_nat(start) + d : U32) : U32),index,Arithmetic.add_associative(low,U32.from_nat(start),d),
    Equal.cong(U32,U32,word => (low + word : U32),(U32.from_nat(start) + d : U32),U32.from_nat(Nat.add(start,digit)),WordArithmetic.native_add(start,digit)))
  +word_back = Equal.trans(U32,w,U32.from_nat(U32.to_nat(w)),U32.from_nat(Reads.kept_index(clamped,U32.to_nat(shift),U32.to_nat(bound),U32.to_nat((low + U32.from_nat(Nat.add(start,digit)) : U32)))),Equal.sym(U32,U32.from_nat(U32.to_nat(w)),w,Arithmetic.native_roundtrip(w)),
    Equal.cong(Nat,U32,U32.from_nat,U32.to_nat(w),Reads.kept_index(clamped,U32.to_nat(shift),U32.to_nat(bound),U32.to_nat((low + U32.from_nat(Nat.add(start,digit)) : U32))),word))
  +sum = Equal.trans(U32,((from + (u - first) * step : U32) * stride + d * (step * stride : U32) : U32),((from + ((u + d : U32) - first) * step : U32) * stride : U32),
    (U32.from_nat(Reads.kept_index(clamped,U32.to_nat(shift),U32.to_nat(bound),U32.to_nat((low + U32.from_nat(Nat.add(start,digit)) : U32)))) * stride : U32),shifted_word(from,u,first,step,stride,d),
    Equal.trans(U32,((from + ((u + d : U32) - first) * step : U32) * stride : U32),(w * stride : U32),(U32.from_nat(Reads.kept_index(clamped,U32.to_nat(shift),U32.to_nat(bound),U32.to_nat((low + U32.from_nat(Nat.add(start,digit)) : U32)))) * stride : U32),
      Equal.cong(U32,U32,value => ((from + (value - first) * step : U32) * stride : U32),(u + d : U32),index,moved),
      Equal.cong(U32,U32,value => (value * stride : U32),w,U32.from_nat(Reads.kept_index(clamped,U32.to_nat(shift),U32.to_nat(bound),U32.to_nat((low + U32.from_nat(Nat.add(start,digit)) : U32)))),word_back)))
  Held.Both{at,
    {%Equal.sym(U32,source_offset,((from + (u - first) * step : U32) * stride : U32),offset) : {(_ + d * source_stride : U32) == (U32.from_nat(Reads.kept_index(clamped,U32.to_nat(shift),U32.to_nat(bound),U32.to_nat((low + U32.from_nat(Nat.add(start,digit)) : U32)))) * stride : U32) : U32}
      %Equal.sym(U32,source_stride,(step * stride : U32),stepped) : {(((from + (u - first) * step : U32) * stride : U32) + d * _ : U32) == (U32.from_nat(Reads.kept_index(clamped,U32.to_nat(shift),U32.to_nat(bound),U32.to_nat((low + U32.from_nat(Nat.add(start,digit)) : U32)))) * stride : U32) : U32}
      sum : {(source_offset + U32.from_nat(digit) * source_stride : U32) == (U32.from_nat(Reads.kept_index(clamped,U32.to_nat(shift),U32.to_nat(bound),U32.to_nat((low + U32.from_nat(Nat.add(start,digit)) : U32)))) * stride : U32) : U32}}}

# A box axis that reads a segment keeps at every digit below its count the
# target coordinate of the digit and reads the index the read keeps for it.
def box_cell_read(+count: U32,+source_stride: U32,+source_offset: U32,+start: Nat,+low: U32,+stride: U32,+shift: U32,+bound: U32,+clamped: Bool,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+box: BoxRead<count,source_stride,source_offset,start,low,stride,shift,bound,clamped,cap>,+shift_room: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},+digit: Nat,
  +below: {Nat.is_lt(digit,U32.to_nat(count)) == True{} : Bool}) ->
  Held.Both<Reads.KeptAt(clamped,shift,bound,(low + U32.from_nat(Nat.add(start,digit)) : U32)),{(source_offset + U32.from_nat(digit) * source_stride : U32) == (U32.from_nat(Reads.kept_index(clamped,U32.to_nat(shift),U32.to_nat(bound),U32.to_nat((low + U32.from_nat(Nat.add(start,digit)) : U32)))) * stride : U32) : U32}>:
  BoxRead{+first,+length,+from,+step,+read,+stepped,+offset,+after,+before,+ends} = box
  +l = U32.to_nat(low)
  +f = U32.to_nat(first)
  +n = U32.to_nat(length)
  +position = Nat.add(start,digit)
  +index = (low + U32.from_nat(Nat.add(start,digit)) : U32)
  +near = Nat.add(Nat.add(l,start),digit)
  +assoc = Addition.associative(l,start,digit)
  +reach = Bounds.nat_trans(near,Nat.add(Nat.add(l,start),U32.to_nat(count)),Nat.add(f,n),Order.offset_lt(Nat.add(l,start),digit,U32.to_nat(count),below),before)
  +room = {%assoc : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} Order.transitive(near,Nat.add(f,n),U32.to_nat(cap),Bounds.strict_le(near,Nat.add(f,n),reach),ends) :
    {Nat.is_le(Nat.add(l,position),U32.to_nat(cap)) == True{} : Bool}}
  +position_room = Order.transitive(position,Nat.add(l,position),U32.to_nat(cap),Order.right_le(l,position),room)
  +exact = Arithmetic.sum_exact(cap,cap_ok,low,U32.from_nat(position),l,position,{==},Bounds.roundtrip(position,cap,cap_ok,position_room),room)
  +index_exact = Equal.trans(Nat,U32.to_nat(index),Nat.add(l,position),near,exact,Equal.sym(Nat,near,Nat.add(l,position),assoc))
  +after_index = {%Equal.sym(Nat,U32.to_nat(index),near,index_exact) : {Nat.is_le(f,_) == True{} : Bool}
    Order.transitive(f,Nat.add(l,start),near,after,Order.le_add(Nat.add(l,start),digit)) : {Nat.is_le(f,U32.to_nat(index)) == True{} : Bool}}
  +before_index = {%Equal.sym(Nat,U32.to_nat(index),near,index_exact) : {Nat.is_lt(_,Nat.add(f,n)) == True{} : Bool} reach : {Nat.is_lt(U32.to_nat(index),Nat.add(f,n)) == True{} : Bool}}
  box_cell_kept(count,source_stride,source_offset,start,low,stride,shift,bound,clamped,first,length,from,step,digit,stepped,offset,
    Reads.segment_kept(clamped,first,length,from,step,shift,bound,read,index,cap,cap_ok,shift_room,after_index,before_index))

# ---- The source cell a cell of a part reads ----

# Every stride is the stride of the source axis its position names.
def StridesAt(full: List<&2,U32>,strides: List<&2,U32>,+axis: U32) -> Data:
  match strides:
    case Nil{}: Unit
    case Con{+stride,strides}: Held.Both<{Tasks.nth(full,axis) == stride : U32},StridesAt(full,strides,(axis + 1 : U32))>

# Every box axis of a part reads a segment of its axis, whose read fits cap.
def BoxesRead(boxes: List<&2,BoxParts.BoxAxis>,target: List<&2,U32>,lows: List<&2,U32>,source: List<&2,U32>,shifts: List<&2,U32>,strides: List<&2,U32>,+clamped: Bool,+cap: U32) -> Data:
  match boxes target lows source shifts strides:
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,target_stride,target_offset},+start,extent},boxes} Con{+total,target} Con{+low,lows} Con{+bound,source}
      Con{+shift,shifts} Con{+stride,strides}:
      Held.Both<BoxRead<count,source_stride,source_offset,start,low,stride,shift,bound,clamped,cap>,Held.Both<{Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},BoxesRead(boxes,target,lows,source,shifts,strides,clamped,cap)>>
    case Nil{} Nil{} Nil{} Nil{} Nil{} Nil{}:
      Unit
    case boxes target lows source shifts strides:
      Empty

# The digit sums of a cell move the first source cell of a box by the read
# indices times the strides.
def cell_sum(+rest: U32,+sum: U32,+word: U32,+base: U32,+offset: U32,+moved: U32,+read: U32,+later: {(rest + sum : U32) == ((base + offset : U32) + word : U32) : U32},
  +here: {(offset + moved : U32) == read : U32}) -> {(rest + (moved + sum : U32) : U32) == (base + (read + word : U32) : U32) : U32}:
  Equal.trans(U32,(rest + (moved + sum : U32) : U32),((rest + sum : U32) + moved : U32),(base + (read + word : U32) : U32),
    Equal.trans(U32,(rest + (moved + sum : U32) : U32),((rest + moved : U32) + sum : U32),((rest + sum : U32) + moved : U32),
      Equal.sym(U32,((rest + moved : U32) + sum : U32),(rest + (moved + sum : U32) : U32),Arithmetic.add_associative(rest,moved,sum)),Arithmetic.add_swap(rest,moved,sum)),
    Equal.trans(U32,((rest + sum : U32) + moved : U32),(((base + offset : U32) + word : U32) + moved : U32),(base + (read + word : U32) : U32),
      Equal.cong(U32,U32,value => (value + moved : U32),(rest + sum : U32),((base + offset : U32) + word : U32),later),
      Equal.trans(U32,(((base + offset : U32) + word : U32) + moved : U32),(base + ((offset + word : U32) + moved : U32) : U32),(base + (read + word : U32) : U32),
        Equal.trans(U32,(((base + offset : U32) + word : U32) + moved : U32),((base + offset : U32) + (word + moved : U32) : U32),(base + ((offset + word : U32) + moved : U32) : U32),
          Arithmetic.add_associative((base + offset : U32),word,moved),
          Equal.trans(U32,((base + offset : U32) + (word + moved : U32) : U32),(base + (offset + (word + moved : U32) : U32) : U32),(base + ((offset + word : U32) + moved : U32) : U32),
            Arithmetic.add_associative(base,offset,(word + moved : U32)),
            Equal.cong(U32,U32,value => (base + value : U32),(offset + (word + moved : U32) : U32),((offset + word : U32) + moved : U32),
              Equal.sym(U32,((offset + word : U32) + moved : U32),(offset + (word + moved : U32) : U32),Arithmetic.add_associative(offset,word,moved))))),
        Equal.cong(U32,U32,value => (base + value : U32),((offset + word : U32) + moved : U32),(read + word : U32),
          Equal.trans(U32,((offset + word : U32) + moved : U32),((offset + moved : U32) + word : U32),(read + word : U32),Arithmetic.add_swap(offset,word,moved),
            Equal.cong(U32,U32,value => (value + word : U32),(offset + moved : U32),read,here))))))

def boxes_step(+boxes: List<&2,BoxParts.BoxAxis>,+target: List<&2,U32>,+lows: List<&2,U32>,+source: List<&2,U32>,+shifts: List<&2,U32>,+strides: List<&2,U32>,
  +digits: List<&2,Nat>,+full: List<&2,U32>,+target_grid: Grid.Grid,+source_grid: Grid.Grid,+clamped: Bool,+axis: U32,+base: U32,+offset: U32,+moved: U32,+stride: U32,
  +position: U32,+shift: U32,+bound: U32,+kept: Nat,+stride_at: {Tasks.nth(full,axis) == stride : U32},
  +here: Held.Both<Reads.KeptAt(clamped,shift,bound,position),{(offset + moved : U32) == (U32.from_nat(kept) * stride : U32) : U32}>,
  +later: Held.Both<Reads.KeptAll(Placements.positions(lows,Digits.added(BoxParts.box_starts(boxes),digits)),source,shifts,clamped),{(BoxParts.box_source(boxes,(base + offset : U32)) + Digits.stepped_digits(digits,ViewPart.move_steps(BoxParts.box_axes(boxes))) : U32) ==
    ((base + offset : U32) + TaskReads.placed_word(Layout.placed_readings(target,source,shifts,target_grid,source_grid,clamped,(axis + 1 : U32)),Reads.kept_indices(Placements.positions(lows,Digits.added(BoxParts.box_starts(boxes),digits)),source,shifts,clamped),full) : U32) : U32}>) ->
  Held.Both<Held.Both<Reads.KeptAt(clamped,shift,bound,position),Reads.KeptAll(Placements.positions(lows,Digits.added(BoxParts.box_starts(boxes),digits)),source,shifts,clamped)>,
    {(BoxParts.box_source(boxes,(base + offset : U32)) + (moved + Digits.stepped_digits(digits,ViewPart.move_steps(BoxParts.box_axes(boxes))) : U32) : U32) ==
      (base + (U32.from_nat(kept) * Tasks.nth(full,axis) + TaskReads.placed_word(Layout.placed_readings(target,source,shifts,target_grid,source_grid,clamped,(axis + 1 : U32)),Reads.kept_indices(Placements.positions(lows,Digits.added(BoxParts.box_starts(boxes),digits)),source,shifts,clamped),full) : U32) : U32) : U32}>:
  Held.Both{+at,+cell} = here
  Held.Both{+kept_rest,+sum} = later
  +word = TaskReads.placed_word(Layout.placed_readings(target,source,shifts,target_grid,source_grid,clamped,(axis + 1 : U32)),Reads.kept_indices(Placements.positions(lows,Digits.added(BoxParts.box_starts(boxes),digits)),source,shifts,clamped),full)
  Held.Both{Held.Both{at,kept_rest},
    {%Equal.sym(U32,Tasks.nth(full,axis),stride,stride_at) : {(BoxParts.box_source(boxes,(base + offset : U32)) + (moved + Digits.stepped_digits(digits,ViewPart.move_steps(BoxParts.box_axes(boxes))) : U32) : U32) ==
        (base + (U32.from_nat(kept) * _ + word : U32) : U32) : U32}
      cell_sum(BoxParts.box_source(boxes,(base + offset : U32)),Digits.stepped_digits(digits,ViewPart.move_steps(BoxParts.box_axes(boxes))),word,base,offset,moved,
        (U32.from_nat(kept) * stride : U32),sum,cell) :
      {(BoxParts.box_source(boxes,(base + offset : U32)) + (moved + Digits.stepped_digits(digits,ViewPart.move_steps(BoxParts.box_axes(boxes))) : U32) : U32) ==
        (base + (U32.from_nat(kept) * Tasks.nth(full,axis) + word : U32) : U32) : U32}}}

# Every cell of a box reads, axis by axis, the index the read of the axis
# keeps for its target coordinate, and the reads keep the coordinates.
def boxes_cell(boxes: List<&2,BoxParts.BoxAxis>,target: List<&2,U32>,lows: List<&2,U32>,source: List<&2,U32>,shifts: List<&2,U32>,strides: List<&2,U32>,
  digits: List<&2,Nat>,+full: List<&2,U32>,+target_grid: Grid.Grid,+source_grid: Grid.Grid,+clamped: Bool,+axis: U32,+base: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +reads: BoxesRead(boxes,target,lows,source,shifts,strides,clamped,cap),+at: StridesAt(full,strides,axis),
  +below: Coordinates.DigitsBelow(ViewPart.move_extents(BoxParts.box_axes(boxes)),digits)) ->
  Held.Both<Reads.KeptAll(Placements.positions(lows,Digits.added(BoxParts.box_starts(boxes),digits)),source,shifts,clamped),{(BoxParts.box_source(boxes,base) + Digits.stepped_digits(digits,ViewPart.move_steps(BoxParts.box_axes(boxes))) : U32) == (base + TaskReads.placed_word(Layout.placed_readings(target,source,shifts,target_grid,source_grid,clamped,axis),Reads.kept_indices(Placements.positions(lows,Digits.added(BoxParts.box_starts(boxes),digits)),source,shifts,clamped),full) : U32) : U32}>:
  match boxes target lows source shifts strides digits:
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+boxes} Con{+total,+target} Con{+low,+lows} Con{+bound,+source} Con{+shift,+shifts} Con{+stride,+strides} Con{+digit,+more}:
      Held.Both{+box,Held.Both{+shift_room,+reads_rest}} = reads
      Held.Both{+stride_at,+at_rest} = at
      KernelGemm.Both{+digit_below,+below_rest} = below
      +position = (low + U32.from_nat(Nat.add(start,digit)) : U32)
      +kept = Reads.kept_index(clamped,U32.to_nat(shift),U32.to_nat(bound),U32.to_nat(position))
      boxes_step(boxes,target,lows,source,shifts,strides,more,full,target_grid,source_grid,clamped,axis,base,source_offset,(U32.from_nat(digit) * source_stride : U32),stride,
        position,shift,bound,kept,stride_at,
        box_cell_read(count,source_stride,source_offset,start,low,stride,shift,bound,clamped,cap,cap_ok,box,shift_room,digit,digit_below),
        boxes_cell(boxes,target,lows,source,shifts,strides,more,full,target_grid,source_grid,clamped,(axis + 1 : U32),(base + source_offset : U32),cap,cap_ok,reads_rest,at_rest,
          below_rest))
    case Nil{} Nil{} Nil{} Nil{} Nil{} Nil{} Nil{}:
      Held.Both{Unit{},{==}}
    case Nil{} Con{+total,+target} lows source shifts strides digits:
      Empty.absurd(Held.Both<Reads.KeptAll(Placements.positions(lows,Digits.added(BoxParts.box_starts(Nil{}),digits)),source,shifts,clamped),{(BoxParts.box_source(Nil{},base) + Digits.stepped_digits(digits,ViewPart.move_steps(BoxParts.box_axes(Nil{}))) : U32) == (base + TaskReads.placed_word(Layout.placed_readings(Con{total,target},source,shifts,target_grid,source_grid,clamped,axis),Reads.kept_indices(Placements.positions(lows,Digits.added(BoxParts.box_starts(Nil{}),digits)),source,shifts,clamped),full) : U32) : U32}>,reads)
    case Nil{} Nil{} Con{+low,+lows} source shifts strides digits:
      Empty.absurd(Held.Both<Reads.KeptAll(Placements.positions(Con{low,lows},Digits.added(BoxParts.box_starts(Nil{}),digits)),source,shifts,clamped),{(BoxParts.box_source(Nil{},base) + Digits.stepped_digits(digits,ViewPart.move_steps(BoxParts.box_axes(Nil{}))) : U32) == (base + TaskReads.placed_word(Layout.placed_readings(Nil{},source,shifts,target_grid,source_grid,clamped,axis),Reads.kept_indices(Placements.positions(Con{low,lows},Digits.added(BoxParts.box_starts(Nil{}),digits)),source,shifts,clamped),full) : U32) : U32}>,reads)
    case Nil{} Nil{} Nil{} Con{+bound,+source} shifts strides digits:
      Empty.absurd(Held.Both<Reads.KeptAll(Placements.positions(Nil{},Digits.added(BoxParts.box_starts(Nil{}),digits)),Con{bound,source},shifts,clamped),{(BoxParts.box_source(Nil{},base) + Digits.stepped_digits(digits,ViewPart.move_steps(BoxParts.box_axes(Nil{}))) : U32) == (base + TaskReads.placed_word(Layout.placed_readings(Nil{},Con{bound,source},shifts,target_grid,source_grid,clamped,axis),Reads.kept_indices(Placements.positions(Nil{},Digits.added(BoxParts.box_starts(Nil{}),digits)),Con{bound,source},shifts,clamped),full) : U32) : U32}>,reads)
    case Nil{} Nil{} Nil{} Nil{} Con{+shift,+shifts} strides digits:
      Empty.absurd(Held.Both<Reads.KeptAll(Placements.positions(Nil{},Digits.added(BoxParts.box_starts(Nil{}),digits)),Nil{},Con{shift,shifts},clamped),{(BoxParts.box_source(Nil{},base) + Digits.stepped_digits(digits,ViewPart.move_steps(BoxParts.box_axes(Nil{}))) : U32) == (base + TaskReads.placed_word(Layout.placed_readings(Nil{},Nil{},Con{shift,shifts},target_grid,source_grid,clamped,axis),Reads.kept_indices(Placements.positions(Nil{},Digits.added(BoxParts.box_starts(Nil{}),digits)),Nil{},Con{shift,shifts},clamped),full) : U32) : U32}>,reads)
    case Nil{} Nil{} Nil{} Nil{} Nil{} Con{+stride,+strides} digits:
      Empty.absurd(Held.Both<Reads.KeptAll(Placements.positions(Nil{},Digits.added(BoxParts.box_starts(Nil{}),digits)),Nil{},Nil{},clamped),{(BoxParts.box_source(Nil{},base) + Digits.stepped_digits(digits,ViewPart.move_steps(BoxParts.box_axes(Nil{}))) : U32) == (base + TaskReads.placed_word(Layout.placed_readings(Nil{},Nil{},Nil{},target_grid,source_grid,clamped,axis),Reads.kept_indices(Placements.positions(Nil{},Digits.added(BoxParts.box_starts(Nil{}),digits)),Nil{},Nil{},clamped),full) : U32) : U32}>,reads)
    case Nil{} Nil{} Nil{} Nil{} Nil{} Nil{} Con{+digit,+more}:
      Empty.absurd(Held.Both<Reads.KeptAll(Placements.positions(Nil{},Digits.added(BoxParts.box_starts(Nil{}),Con{digit,more})),Nil{},Nil{},clamped),{(BoxParts.box_source(Nil{},base) + Digits.stepped_digits(Con{digit,more},ViewPart.move_steps(BoxParts.box_axes(Nil{}))) : U32) == (base + TaskReads.placed_word(Layout.placed_readings(Nil{},Nil{},Nil{},target_grid,source_grid,clamped,axis),Reads.kept_indices(Placements.positions(Nil{},Digits.added(BoxParts.box_starts(Nil{}),Con{digit,more})),Nil{},Nil{},clamped),full) : U32) : U32}>,below)
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+boxes} Nil{} lows source shifts strides digits:
      Empty.absurd(Held.Both<Reads.KeptAll(Placements.positions(lows,Digits.added(BoxParts.box_starts(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes}),digits)),source,shifts,clamped),{(BoxParts.box_source(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes},base) + Digits.stepped_digits(digits,ViewPart.move_steps(BoxParts.box_axes(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes}))) : U32) == (base + TaskReads.placed_word(Layout.placed_readings(Nil{},source,shifts,target_grid,source_grid,clamped,axis),Reads.kept_indices(Placements.positions(lows,Digits.added(BoxParts.box_starts(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes}),digits)),source,shifts,clamped),full) : U32) : U32}>,reads)
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+boxes} Con{+total,+target} Nil{} source shifts strides digits:
      Empty.absurd(Held.Both<Reads.KeptAll(Placements.positions(Nil{},Digits.added(BoxParts.box_starts(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes}),digits)),source,shifts,clamped),{(BoxParts.box_source(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes},base) + Digits.stepped_digits(digits,ViewPart.move_steps(BoxParts.box_axes(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes}))) : U32) == (base + TaskReads.placed_word(Layout.placed_readings(Con{total,target},source,shifts,target_grid,source_grid,clamped,axis),Reads.kept_indices(Placements.positions(Nil{},Digits.added(BoxParts.box_starts(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes}),digits)),source,shifts,clamped),full) : U32) : U32}>,reads)
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+boxes} Con{+total,+target} Con{+low,+lows} Nil{} shifts strides digits:
      Empty.absurd(Held.Both<Reads.KeptAll(Placements.positions(Con{low,lows},Digits.added(BoxParts.box_starts(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes}),digits)),Nil{},shifts,clamped),{(BoxParts.box_source(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes},base) + Digits.stepped_digits(digits,ViewPart.move_steps(BoxParts.box_axes(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes}))) : U32) == (base + TaskReads.placed_word(Layout.placed_readings(Con{total,target},Nil{},shifts,target_grid,source_grid,clamped,axis),Reads.kept_indices(Placements.positions(Con{low,lows},Digits.added(BoxParts.box_starts(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes}),digits)),Nil{},shifts,clamped),full) : U32) : U32}>,reads)
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+boxes} Con{+total,+target} Con{+low,+lows} Con{+bound,+source} Nil{} strides digits:
      Empty.absurd(Held.Both<Reads.KeptAll(Placements.positions(Con{low,lows},Digits.added(BoxParts.box_starts(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes}),digits)),Con{bound,source},Nil{},clamped),{(BoxParts.box_source(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes},base) + Digits.stepped_digits(digits,ViewPart.move_steps(BoxParts.box_axes(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes}))) : U32) == (base + TaskReads.placed_word(Layout.placed_readings(Con{total,target},Con{bound,source},Nil{},target_grid,source_grid,clamped,axis),Reads.kept_indices(Placements.positions(Con{low,lows},Digits.added(BoxParts.box_starts(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes}),digits)),Con{bound,source},Nil{},clamped),full) : U32) : U32}>,reads)
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+boxes} Con{+total,+target} Con{+low,+lows} Con{+bound,+source} Con{+shift,+shifts} Nil{} digits:
      Empty.absurd(Held.Both<Reads.KeptAll(Placements.positions(Con{low,lows},Digits.added(BoxParts.box_starts(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes}),digits)),Con{bound,source},Con{shift,shifts},clamped),{(BoxParts.box_source(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes},base) + Digits.stepped_digits(digits,ViewPart.move_steps(BoxParts.box_axes(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes}))) : U32) == (base + TaskReads.placed_word(Layout.placed_readings(Con{total,target},Con{bound,source},Con{shift,shifts},target_grid,source_grid,clamped,axis),Reads.kept_indices(Placements.positions(Con{low,lows},Digits.added(BoxParts.box_starts(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes}),digits)),Con{bound,source},Con{shift,shifts},clamped),full) : U32) : U32}>,reads)
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+boxes} Con{+total,+target} Con{+low,+lows} Con{+bound,+source} Con{+shift,+shifts} Con{+stride,+strides} Nil{}:
      Empty.absurd(Held.Both<Reads.KeptAll(Placements.positions(Con{low,lows},Digits.added(BoxParts.box_starts(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes}),Nil{})),Con{bound,source},Con{shift,shifts},clamped),{(BoxParts.box_source(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes},base) + Digits.stepped_digits(Nil{},ViewPart.move_steps(BoxParts.box_axes(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes}))) : U32) == (base + TaskReads.placed_word(Layout.placed_readings(Con{total,target},Con{bound,source},Con{shift,shifts},target_grid,source_grid,clamped,axis),Reads.kept_indices(Placements.positions(Con{low,lows},Digits.added(BoxParts.box_starts(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes}),Nil{})),Con{bound,source},Con{shift,shifts},clamped),full) : U32) : U32}>,below)

# ---- The box axes of an axis read its segments ----

# Every segment of an axis reads its source as the read of the axis keeps it.
def SegmentsReading(segments: List<&2,Tasks.Segment>,+shift: U32,+bound: U32,+clamped: Bool) -> Data:
  match segments:
    case Nil{}: Unit
    case Con{Tasks.Segment{+first,+count,+source,+step},rest}:
      Held.Both<Reads.SegmentRead<first,count,source,step,shift,bound,clamped>,SegmentsReading(rest,shift,bound,clamped)>

# Every box axis of an axis reads a segment of the axis.
def AxisReads(boxes: List<&2,BoxParts.BoxAxis>,+low: U32,+stride: U32,+shift: U32,+bound: U32,+clamped: Bool,+cap: U32) -> Data:
  match boxes:
    case Nil{}: Unit
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,target_stride,target_offset},+start,extent},more}:
      Held.Both<BoxRead<count,source_stride,source_offset,start,low,stride,shift,bound,clamped,cap>,AxisReads(more,low,stride,shift,bound,clamped,cap)>

# A start word is the low word plus the start of the box.
def start_word(+first: U32,+low: U32) -> {(low + U32.from_nat(U32.to_nat((U32.max(first,low) - low : U32))) : U32) == U32.max(first,low) : U32}:
  Equal.trans(U32,(low + U32.from_nat(U32.to_nat((U32.max(first,low) - low : U32))) : U32),(low + (U32.max(first,low) - low : U32) : U32),U32.max(first,low),
    Equal.cong(U32,U32,word => (low + word : U32),U32.from_nat(U32.to_nat((U32.max(first,low) - low : U32))),(U32.max(first,low) - low : U32),Arithmetic.native_roundtrip((U32.max(first,low) - low : U32))),
    Equal.trans(U32,(low + (U32.max(first,low) - low : U32) : U32),((U32.max(first,low) - low : U32) + low : U32),U32.max(first,low),Arithmetic.add_commutative(low,(U32.max(first,low) - low : U32)),Subtraction.add_subtract(U32.max(first,low),low)))

def picked_reads(picked: Bool,+source: U32,+step: U32,+source_stride: U32,+piece_stride: U32,+rest: List<&2,Tasks.Segment>,+first: U32,+count: U32,+low: U32,+extent: U32,
  +shift: U32,+bound: U32,+clamped: Bool,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+segment_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool},+piece_room: {Nat.is_le(Nat.add(U32.to_nat(low),U32.to_nat(extent)),U32.to_nat(cap)) == True{} : Bool},
  +by_apart: {U32.is_lt(U32.max(first,low),U32.min((first + count : U32),(low + extent : U32))) == picked : Bool},+read: Reads.SegmentRead<first,count,source,step,shift,bound,clamped>,
  +later: AxisReads(Placements.axis_boxes(rest,low,extent,source_stride,piece_stride),low,source_stride,shift,bound,clamped,cap)) ->
  AxisReads(Bool.pick(List<&2,BoxParts.BoxAxis>,picked,BoxParts.BoxAxis{Tasks.Cut{(U32.min((first + count : U32),(low + extent : U32)) - U32.max(first,low) : U32),(step * source_stride : U32),((source + (U32.max(first,low) - first : U32) * step : U32) * source_stride : U32),piece_stride,((U32.max(first,low) - low : U32) * piece_stride : U32)},U32.to_nat((U32.max(first,low) - low : U32)),extent} <> Placements.axis_boxes(rest,low,extent,source_stride,piece_stride),
    Placements.axis_boxes(rest,low,extent,source_stride,piece_stride)),low,source_stride,shift,bound,clamped,cap):
  match picked:
    case True{}:
      +apart = Equal.trans(Bool,Nat.is_lt(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent)))),U32.is_lt(U32.max(first,low),U32.min((first + count : U32),(low + extent : U32))),True{},Equal.sym(Bool,U32.is_lt(U32.max(first,low),U32.min((first + count : U32),(low + extent : U32))),Nat.is_lt(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent)))),Placements.apart_word(first,count,low,extent,cap,cap_ok,segment_room,piece_room)),by_apart)
      +low_max = NatExtrema.maximum_right(U32.to_nat(first),U32.to_nat(low))
      +max_min = Bounds.strict_le(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),apart)
      +start_exact = Equal.trans(Nat,Nat.add(U32.to_nat(low),U32.to_nat((U32.max(first,low) - low : U32))),Nat.add(U32.to_nat(low),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low))),Nat.max(U32.to_nat(first),U32.to_nat(low)),
        Equal.cong(Nat,Nat,value => Nat.add(U32.to_nat(low),value),U32.to_nat((U32.max(first,low) - low : U32)),Nat.sub(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat(low)),Placements.offset_word(first,count,low,extent,cap,cap_ok,segment_room,piece_room)),Order.restore_shift(U32.to_nat(low),Nat.max(U32.to_nat(first),U32.to_nat(low)),low_max))
      +ends_exact = Equal.trans(Nat,Nat.add(Nat.add(U32.to_nat(low),U32.to_nat((U32.max(first,low) - low : U32))),U32.to_nat((U32.min((first + count : U32),(low + extent : U32)) - U32.max(first,low) : U32))),Nat.add(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low)))),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),
        Equal.trans(Nat,Nat.add(Nat.add(U32.to_nat(low),U32.to_nat((U32.max(first,low) - low : U32))),U32.to_nat((U32.min((first + count : U32),(low + extent : U32)) - U32.max(first,low) : U32))),Nat.add(Nat.max(U32.to_nat(first),U32.to_nat(low)),U32.to_nat((U32.min((first + count : U32),(low + extent : U32)) - U32.max(first,low) : U32))),Nat.add(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low)))),
          Equal.cong(Nat,Nat,value => Nat.add(value,U32.to_nat((U32.min((first + count : U32),(low + extent : U32)) - U32.max(first,low) : U32))),Nat.add(U32.to_nat(low),U32.to_nat((U32.max(first,low) - low : U32))),Nat.max(U32.to_nat(first),U32.to_nat(low)),start_exact),
          Equal.cong(Nat,Nat,value => Nat.add(Nat.max(U32.to_nat(first),U32.to_nat(low)),value),U32.to_nat((U32.min((first + count : U32),(low + extent : U32)) - U32.max(first,low) : U32)),Nat.sub(Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),Nat.max(U32.to_nat(first),U32.to_nat(low))),Placements.count_word(first,count,low,extent,cap,cap_ok,segment_room,piece_room,apart))),
        Order.restore_shift(Nat.max(U32.to_nat(first),U32.to_nat(low)),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),max_min))
      +offset = Equal.sym(U32,((source + ((low + U32.from_nat(U32.to_nat((U32.max(first,low) - low : U32))) : U32) - first) * step : U32) * source_stride : U32),((source + (U32.max(first,low) - first : U32) * step : U32) * source_stride : U32),
        Equal.cong(U32,U32,word => ((source + (word - first) * step : U32) * source_stride : U32),(low + U32.from_nat(U32.to_nat((U32.max(first,low) - low : U32))) : U32),U32.max(first,low),start_word(first,low)))
      +after = {%Equal.sym(Nat,Nat.add(U32.to_nat(low),U32.to_nat((U32.max(first,low) - low : U32))),Nat.max(U32.to_nat(first),U32.to_nat(low)),start_exact) : {Nat.is_le(U32.to_nat(first),_) == True{} : Bool} NatExtrema.maximum_left(U32.to_nat(first),U32.to_nat(low)) : {Nat.is_le(U32.to_nat(first),Nat.add(U32.to_nat(low),U32.to_nat((U32.max(first,low) - low : U32)))) == True{} : Bool}}
      +before = {%Equal.sym(Nat,Nat.add(Nat.add(U32.to_nat(low),U32.to_nat((U32.max(first,low) - low : U32))),U32.to_nat((U32.min((first + count : U32),(low + extent : U32)) - U32.max(first,low) : U32))),Nat.min(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))),ends_exact) : {Nat.is_le(_,Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool}
        NatExtrema.minimum_left(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(low),U32.to_nat(extent))) :
        {Nat.is_le(Nat.add(Nat.add(U32.to_nat(low),U32.to_nat((U32.max(first,low) - low : U32))),U32.to_nat((U32.min((first + count : U32),(low + extent : U32)) - U32.max(first,low) : U32))),Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool}}
      Held.Both{BoxRead{first,count,source,step,read,{==},offset,after,before,segment_room},later}
    case False{}: later

# The box axes of an axis read its segments.
def axis_reads(segments: List<&2,Tasks.Segment>,+low: U32,+extent: U32,+source_stride: U32,+piece_stride: U32,+shift: U32,+bound: U32,+clamped: Bool,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+piece_room: {Nat.is_le(Nat.add(U32.to_nat(low),U32.to_nat(extent)),U32.to_nat(cap)) == True{} : Bool},+segments_in: Placements.SegmentsIn(segments,cap),+reading: SegmentsReading(segments,shift,bound,clamped)) ->
  AxisReads(Placements.axis_boxes(segments,low,extent,source_stride,piece_stride),low,source_stride,shift,bound,clamped,cap):
  match segments:
    case Nil{}: Unit{}
    case Con{Tasks.Segment{+first,+count,+source,+step},+rest}:
      Held.Both{+segment_room,+rest_in} = segments_in
      Held.Both{+read,+rest_reading} = reading
      picked_reads(U32.is_lt(U32.max(first,low),U32.min((first + count : U32),(low + extent : U32))),source,step,source_stride,piece_stride,rest,first,count,low,extent,shift,bound,clamped,cap,cap_ok,segment_room,piece_room,{==},read,
        axis_reads(rest,low,extent,source_stride,piece_stride,shift,bound,clamped,cap,cap_ok,piece_room,rest_in,rest_reading))

# ---- The prefixes read their segments ----

# The first source cell of a box reversed onto outer boxes.
def source_reversed(boxes: List<&2,BoxParts.BoxAxis>,+outer: List<&2,BoxParts.BoxAxis>,+first: U32) ->
  {BoxParts.box_source(Placements.reversed_boxes(boxes,outer),first) == BoxParts.box_source(outer,BoxParts.box_source(Placements.reversed_boxes(boxes,[]),first)) : U32}:
  match boxes:
    case Nil{}: {==}
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+own},+more}:
      +box = {BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,own} : BoxParts.BoxAxis}
      Equal.trans(U32,BoxParts.box_source(Placements.reversed_boxes(more,box <> outer),first),BoxParts.box_source(box <> outer,BoxParts.box_source(Placements.reversed_boxes(more,[]),first)),
        BoxParts.box_source(outer,BoxParts.box_source(Placements.reversed_boxes(more,[box]),first)),
        source_reversed(more,box <> outer,first),
        Equal.cong(U32,U32,value => BoxParts.box_source(outer,value),(BoxParts.box_source(Placements.reversed_boxes(more,[]),first) + source_offset : U32),
          BoxParts.box_source(Placements.reversed_boxes(more,[box]),first),
          Equal.sym(U32,BoxParts.box_source(Placements.reversed_boxes(more,[box]),first),BoxParts.box_source([box],BoxParts.box_source(Placements.reversed_boxes(more,[]),first)),
            source_reversed(more,[box],first))))

# Every prefix reads its segments, against the lists of the axes done,
# innermost first, and starts at the first source cell of its box.
def PrefixesRead(list: List<&2,Placements.Prefix>,+target: List<&2,U32>,+lows: List<&2,U32>,+source: List<&2,U32>,+shifts: List<&2,U32>,+strides: List<&2,U32>,+clamped: Bool,+cap: U32) ->
  Data:
  match list:
    case Nil{}: Unit
    case Con{Placements.Prefix{+boxes,+from,+to},more}:
      Held.Both<BoxesRead(boxes,target,lows,source,shifts,strides,clamped,cap),
        Held.Both<{from == BoxParts.box_source(Placements.reversed_boxes(boxes,[]),0) : U32},PrefixesRead(more,target,lows,source,shifts,strides,clamped,cap)>>

def prefix_extended_read(boxes: List<&2,BoxParts.BoxAxis>,+inner: List<&2,BoxParts.BoxAxis>,+from: U32,+to: U32,+later: List<&2,Placements.Prefix>,
  +total: U32,+low: U32,+bound: U32,+shift: U32,+stride: U32,+target: List<&2,U32>,+lows: List<&2,U32>,+source: List<&2,U32>,+shifts: List<&2,U32>,+strides: List<&2,U32>,
  +clamped: Bool,+cap: U32,+axis: AxisReads(boxes,low,stride,shift,bound,clamped,cap),+shift_room: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +held: BoxesRead(inner,target,lows,source,shifts,strides,clamped,cap),+placed: {from == BoxParts.box_source(Placements.reversed_boxes(inner,[]),0) : U32},
  +later_read: PrefixesRead(later,total <> target,low <> lows,bound <> source,shift <> shifts,stride <> strides,clamped,cap)) ->
  PrefixesRead(Placements.prefix_extended(boxes,inner,from,to,later),total <> target,low <> lows,bound <> source,shift <> shifts,stride <> strides,clamped,cap):
  match boxes:
    case Nil{}: later_read
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+own},+more}:
      Held.Both{+box_read,+axis_more} = axis
      +box = {BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,own} : BoxParts.BoxAxis}
      Held.Both{Held.Both{box_read,Held.Both{shift_room,held}},
        Held.Both{Equal.trans(U32,(from + source_offset : U32),(BoxParts.box_source(Placements.reversed_boxes(inner,[]),0) + source_offset : U32),
            BoxParts.box_source(Placements.reversed_boxes(inner,[box]),0),
            Equal.cong(U32,U32,value => (value + source_offset : U32),from,BoxParts.box_source(Placements.reversed_boxes(inner,[]),0),placed),
            Equal.sym(U32,BoxParts.box_source(Placements.reversed_boxes(inner,[box]),0),BoxParts.box_source([box],BoxParts.box_source(Placements.reversed_boxes(inner,[]),0)),
              source_reversed(inner,[box],0))),
          prefix_extended_read(more,inner,from,to,later,total,low,bound,shift,stride,target,lows,source,shifts,strides,clamped,cap,axis_more,shift_room,held,placed,later_read)}}

def prefixes_extended_read(prefixes: List<&2,Placements.Prefix>,+boxes: List<&2,BoxParts.BoxAxis>,+total: U32,+low: U32,+bound: U32,+shift: U32,+stride: U32,
  +target: List<&2,U32>,+lows: List<&2,U32>,+source: List<&2,U32>,+shifts: List<&2,U32>,+strides: List<&2,U32>,+clamped: Bool,+cap: U32,
  +axis: AxisReads(boxes,low,stride,shift,bound,clamped,cap),+shift_room: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},+held: PrefixesRead(prefixes,target,lows,source,shifts,strides,clamped,cap)) ->
  PrefixesRead(Placements.prefixes_extended(prefixes,boxes),total <> target,low <> lows,bound <> source,shift <> shifts,stride <> strides,clamped,cap):
  match prefixes:
    case Nil{}: Unit{}
    case Con{Placements.Prefix{+inner,+from,+to},+more}:
      Held.Both{+own,Held.Both{+placed,+held_more}} = held
      prefix_extended_read(boxes,inner,from,to,Placements.prefixes_extended(more,boxes),total,low,bound,shift,stride,target,lows,source,shifts,strides,clamped,cap,axis,shift_room,
        own,placed,prefixes_extended_read(more,boxes,total,low,bound,shift,stride,target,lows,source,shifts,strides,clamped,cap,axis,shift_room,held_more))

# Every axis of a placement fits cap with its segments, which read the
# source as the read of the axis keeps it, and all lists end together.
def AxesReading(segments: List<&2,List<&2,Tasks.Segment>>,target: List<&2,U32>,lows: List<&2,U32>,extents: List<&2,U32>,source: List<&2,U32>,shifts: List<&2,U32>,
  source_strides: List<&2,U32>,piece_strides: List<&2,U32>,+clamped: Bool,+cap: U32) -> Data:
  match segments target lows extents source shifts source_strides piece_strides:
    case Con{own,segments} Con{+total,target} Con{+low,lows} Con{+extent,extents} Con{+bound,source} Con{+shift,shifts} Con{+source_stride,source_strides}
      Con{+piece_stride,piece_strides}:
      Held.Both<{Nat.is_le(Nat.add(U32.to_nat(low),U32.to_nat(extent)),U32.to_nat(cap)) == True{} : Bool},Held.Both<Placements.SegmentsIn(own,cap),Held.Both<SegmentsReading(own,shift,bound,clamped),
        Held.Both<{Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},AxesReading(segments,target,lows,extents,source,shifts,source_strides,piece_strides,clamped,cap)>>>>
    case Nil{} Nil{} Nil{} Nil{} Nil{} Nil{} Nil{} Nil{}:
      Unit
    case segments target lows extents source shifts source_strides piece_strides:
      Empty

# The prefixes of a placement, built axis after axis, read their segments
# against the lists of every axis, innermost first.
def prefixes_read(segments: List<&2,List<&2,Tasks.Segment>>,target: List<&2,U32>,lows: List<&2,U32>,extents: List<&2,U32>,source: List<&2,U32>,shifts: List<&2,U32>,
  source_strides: List<&2,U32>,piece_strides: List<&2,U32>,+prefixes: List<&2,Placements.Prefix>,+target_done: List<&2,U32>,+lows_done: List<&2,U32>,+source_done: List<&2,U32>,
  +shifts_done: List<&2,U32>,+strides_done: List<&2,U32>,+clamped: Bool,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +axes: AxesReading(segments,target,lows,extents,source,shifts,source_strides,piece_strides,clamped,cap),
  +held: PrefixesRead(prefixes,target_done,lows_done,source_done,shifts_done,strides_done,clamped,cap)) ->
  PrefixesRead(Placements.placed_prefixes(segments,lows,extents,source_strides,piece_strides,prefixes),Placements.reversed_extents(target,target_done),Placements.reversed_extents(lows,lows_done),Placements.reversed_extents(source,source_done),Placements.reversed_extents(shifts,shifts_done),Placements.reversed_extents(source_strides,strides_done),clamped,cap):
  match segments target lows extents source shifts source_strides piece_strides:
    case Con{+own,+segments} Con{+total,+target} Con{+low,+lows} Con{+extent,+extents} Con{+bound,+source} Con{+shift,+shifts} Con{+source_stride,+source_strides} Con{+piece_stride,+piece_strides}:
      Held.Both{+piece_room,Held.Both{+segments_in,Held.Both{+reading,Held.Both{+shift_room,+axes_more}}}} = axes
      +boxes = Placements.axis_boxes(own,low,extent,source_stride,piece_stride)
      prefixes_read(segments,target,lows,extents,source,shifts,source_strides,piece_strides,Placements.prefixes_extended(prefixes,boxes),total <> target_done,low <> lows_done,
        bound <> source_done,shift <> shifts_done,source_stride <> strides_done,clamped,cap,cap_ok,axes_more,
        prefixes_extended_read(prefixes,boxes,total,low,bound,shift,source_stride,target_done,lows_done,source_done,shifts_done,strides_done,clamped,cap,
          axis_reads(own,low,extent,source_stride,piece_stride,shift,bound,clamped,cap,cap_ok,piece_room,segments_in,reading),shift_room,held))
    case Nil{} Nil{} Nil{} Nil{} Nil{} Nil{} Nil{} Nil{}:
      held
    case Nil{} Con{+total,+target} lows extents source shifts source_strides piece_strides:
      Empty.absurd(PrefixesRead(Placements.placed_prefixes(Nil{},lows,extents,source_strides,piece_strides,prefixes),Placements.reversed_extents(Con{total,target},target_done),Placements.reversed_extents(lows,lows_done),Placements.reversed_extents(source,source_done),Placements.reversed_extents(shifts,shifts_done),Placements.reversed_extents(source_strides,strides_done),clamped,cap),axes)
    case Nil{} Nil{} Con{+low,+lows} extents source shifts source_strides piece_strides:
      Empty.absurd(PrefixesRead(Placements.placed_prefixes(Nil{},Con{low,lows},extents,source_strides,piece_strides,prefixes),Placements.reversed_extents(Nil{},target_done),Placements.reversed_extents(Con{low,lows},lows_done),Placements.reversed_extents(source,source_done),Placements.reversed_extents(shifts,shifts_done),Placements.reversed_extents(source_strides,strides_done),clamped,cap),axes)
    case Nil{} Nil{} Nil{} Con{+extent,+extents} source shifts source_strides piece_strides:
      Empty.absurd(PrefixesRead(Placements.placed_prefixes(Nil{},Nil{},Con{extent,extents},source_strides,piece_strides,prefixes),Placements.reversed_extents(Nil{},target_done),Placements.reversed_extents(Nil{},lows_done),Placements.reversed_extents(source,source_done),Placements.reversed_extents(shifts,shifts_done),Placements.reversed_extents(source_strides,strides_done),clamped,cap),axes)
    case Nil{} Nil{} Nil{} Nil{} Con{+bound,+source} shifts source_strides piece_strides:
      Empty.absurd(PrefixesRead(Placements.placed_prefixes(Nil{},Nil{},Nil{},source_strides,piece_strides,prefixes),Placements.reversed_extents(Nil{},target_done),Placements.reversed_extents(Nil{},lows_done),Placements.reversed_extents(Con{bound,source},source_done),Placements.reversed_extents(shifts,shifts_done),Placements.reversed_extents(source_strides,strides_done),clamped,cap),axes)
    case Nil{} Nil{} Nil{} Nil{} Nil{} Con{+shift,+shifts} source_strides piece_strides:
      Empty.absurd(PrefixesRead(Placements.placed_prefixes(Nil{},Nil{},Nil{},source_strides,piece_strides,prefixes),Placements.reversed_extents(Nil{},target_done),Placements.reversed_extents(Nil{},lows_done),Placements.reversed_extents(Nil{},source_done),Placements.reversed_extents(Con{shift,shifts},shifts_done),Placements.reversed_extents(source_strides,strides_done),clamped,cap),axes)
    case Nil{} Nil{} Nil{} Nil{} Nil{} Nil{} Con{+source_stride,+source_strides} piece_strides:
      Empty.absurd(PrefixesRead(Placements.placed_prefixes(Nil{},Nil{},Nil{},Con{source_stride,source_strides},piece_strides,prefixes),Placements.reversed_extents(Nil{},target_done),Placements.reversed_extents(Nil{},lows_done),Placements.reversed_extents(Nil{},source_done),Placements.reversed_extents(Nil{},shifts_done),Placements.reversed_extents(Con{source_stride,source_strides},strides_done),clamped,cap),axes)
    case Nil{} Nil{} Nil{} Nil{} Nil{} Nil{} Nil{} Con{+piece_stride,+piece_strides}:
      Empty.absurd(PrefixesRead(Placements.placed_prefixes(Nil{},Nil{},Nil{},Nil{},Con{piece_stride,piece_strides},prefixes),Placements.reversed_extents(Nil{},target_done),Placements.reversed_extents(Nil{},lows_done),Placements.reversed_extents(Nil{},source_done),Placements.reversed_extents(Nil{},shifts_done),Placements.reversed_extents(Nil{},strides_done),clamped,cap),axes)
    case Con{+own,+segments} Nil{} lows extents source shifts source_strides piece_strides:
      Empty.absurd(PrefixesRead(Placements.placed_prefixes(Con{own,segments},lows,extents,source_strides,piece_strides,prefixes),Placements.reversed_extents(Nil{},target_done),Placements.reversed_extents(lows,lows_done),Placements.reversed_extents(source,source_done),Placements.reversed_extents(shifts,shifts_done),Placements.reversed_extents(source_strides,strides_done),clamped,cap),axes)
    case Con{+own,+segments} Con{+total,+target} Nil{} extents source shifts source_strides piece_strides:
      Empty.absurd(PrefixesRead(Placements.placed_prefixes(Con{own,segments},Nil{},extents,source_strides,piece_strides,prefixes),Placements.reversed_extents(Con{total,target},target_done),Placements.reversed_extents(Nil{},lows_done),Placements.reversed_extents(source,source_done),Placements.reversed_extents(shifts,shifts_done),Placements.reversed_extents(source_strides,strides_done),clamped,cap),axes)
    case Con{+own,+segments} Con{+total,+target} Con{+low,+lows} Nil{} source shifts source_strides piece_strides:
      Empty.absurd(PrefixesRead(Placements.placed_prefixes(Con{own,segments},Con{low,lows},Nil{},source_strides,piece_strides,prefixes),Placements.reversed_extents(Con{total,target},target_done),Placements.reversed_extents(Con{low,lows},lows_done),Placements.reversed_extents(source,source_done),Placements.reversed_extents(shifts,shifts_done),Placements.reversed_extents(source_strides,strides_done),clamped,cap),axes)
    case Con{+own,+segments} Con{+total,+target} Con{+low,+lows} Con{+extent,+extents} Nil{} shifts source_strides piece_strides:
      Empty.absurd(PrefixesRead(Placements.placed_prefixes(Con{own,segments},Con{low,lows},Con{extent,extents},source_strides,piece_strides,prefixes),Placements.reversed_extents(Con{total,target},target_done),Placements.reversed_extents(Con{low,lows},lows_done),Placements.reversed_extents(Nil{},source_done),Placements.reversed_extents(shifts,shifts_done),Placements.reversed_extents(source_strides,strides_done),clamped,cap),axes)
    case Con{+own,+segments} Con{+total,+target} Con{+low,+lows} Con{+extent,+extents} Con{+bound,+source} Nil{} source_strides piece_strides:
      Empty.absurd(PrefixesRead(Placements.placed_prefixes(Con{own,segments},Con{low,lows},Con{extent,extents},source_strides,piece_strides,prefixes),Placements.reversed_extents(Con{total,target},target_done),Placements.reversed_extents(Con{low,lows},lows_done),Placements.reversed_extents(Con{bound,source},source_done),Placements.reversed_extents(Nil{},shifts_done),Placements.reversed_extents(source_strides,strides_done),clamped,cap),axes)
    case Con{+own,+segments} Con{+total,+target} Con{+low,+lows} Con{+extent,+extents} Con{+bound,+source} Con{+shift,+shifts} Nil{} piece_strides:
      Empty.absurd(PrefixesRead(Placements.placed_prefixes(Con{own,segments},Con{low,lows},Con{extent,extents},Nil{},piece_strides,prefixes),Placements.reversed_extents(Con{total,target},target_done),Placements.reversed_extents(Con{low,lows},lows_done),Placements.reversed_extents(Con{bound,source},source_done),Placements.reversed_extents(Con{shift,shifts},shifts_done),Placements.reversed_extents(Nil{},strides_done),clamped,cap),axes)
    case Con{+own,+segments} Con{+total,+target} Con{+low,+lows} Con{+extent,+extents} Con{+bound,+source} Con{+shift,+shifts} Con{+source_stride,+source_strides} Nil{}:
      Empty.absurd(PrefixesRead(Placements.placed_prefixes(Con{own,segments},Con{low,lows},Con{extent,extents},Con{source_stride,source_strides},Nil{},prefixes),Placements.reversed_extents(Con{total,target},target_done),Placements.reversed_extents(Con{low,lows},lows_done),Placements.reversed_extents(Con{bound,source},source_done),Placements.reversed_extents(Con{shift,shifts},shifts_done),Placements.reversed_extents(Con{source_stride,source_strides},strides_done),clamped,cap),axes)

# ---- The parts of a placement read their segments ----

# The box axes of a prefix, innermost first, reversed onto boxes that read
# their segments, read their segments against the reversed lists.
def boxes_reversed(boxes: List<&2,BoxParts.BoxAxis>,target: List<&2,U32>,lows: List<&2,U32>,source: List<&2,U32>,shifts: List<&2,U32>,strides: List<&2,U32>,
  +acc: List<&2,BoxParts.BoxAxis>,+acc_target: List<&2,U32>,+acc_lows: List<&2,U32>,+acc_source: List<&2,U32>,+acc_shifts: List<&2,U32>,+acc_strides: List<&2,U32>,
  +clamped: Bool,+cap: U32,+reads: BoxesRead(boxes,target,lows,source,shifts,strides,clamped,cap),
  +held: BoxesRead(acc,acc_target,acc_lows,acc_source,acc_shifts,acc_strides,clamped,cap)) ->
  BoxesRead(Placements.reversed_boxes(boxes,acc),Placements.reversed_extents(target,acc_target),Placements.reversed_extents(lows,acc_lows),Placements.reversed_extents(source,acc_source),Placements.reversed_extents(shifts,acc_shifts),Placements.reversed_extents(strides,acc_strides),clamped,cap):
  match boxes target lows source shifts strides:
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+boxes} Con{+total,+target} Con{+low,+lows} Con{+bound,+source} Con{+shift,+shifts} Con{+stride,+strides}:
      Held.Both{+box_read,Held.Both{+shift_room,+rest}} = reads
      +box = {BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent} : BoxParts.BoxAxis}
      boxes_reversed(boxes,target,lows,source,shifts,strides,box <> acc,total <> acc_target,low <> acc_lows,bound <> acc_source,shift <> acc_shifts,stride <> acc_strides,
        clamped,cap,rest,Held.Both{box_read,Held.Both{shift_room,held}})
    case Nil{} Nil{} Nil{} Nil{} Nil{} Nil{}:
      held
    case Nil{} Con{+total,+target} lows source shifts strides:
      Empty.absurd(BoxesRead(Placements.reversed_boxes(Nil{},acc),Placements.reversed_extents(Con{total,target},acc_target),Placements.reversed_extents(lows,acc_lows),Placements.reversed_extents(source,acc_source),Placements.reversed_extents(shifts,acc_shifts),Placements.reversed_extents(strides,acc_strides),clamped,cap),reads)
    case Nil{} Nil{} Con{+low,+lows} source shifts strides:
      Empty.absurd(BoxesRead(Placements.reversed_boxes(Nil{},acc),Placements.reversed_extents(Nil{},acc_target),Placements.reversed_extents(Con{low,lows},acc_lows),Placements.reversed_extents(source,acc_source),Placements.reversed_extents(shifts,acc_shifts),Placements.reversed_extents(strides,acc_strides),clamped,cap),reads)
    case Nil{} Nil{} Nil{} Con{+bound,+source} shifts strides:
      Empty.absurd(BoxesRead(Placements.reversed_boxes(Nil{},acc),Placements.reversed_extents(Nil{},acc_target),Placements.reversed_extents(Nil{},acc_lows),Placements.reversed_extents(Con{bound,source},acc_source),Placements.reversed_extents(shifts,acc_shifts),Placements.reversed_extents(strides,acc_strides),clamped,cap),reads)
    case Nil{} Nil{} Nil{} Nil{} Con{+shift,+shifts} strides:
      Empty.absurd(BoxesRead(Placements.reversed_boxes(Nil{},acc),Placements.reversed_extents(Nil{},acc_target),Placements.reversed_extents(Nil{},acc_lows),Placements.reversed_extents(Nil{},acc_source),Placements.reversed_extents(Con{shift,shifts},acc_shifts),Placements.reversed_extents(strides,acc_strides),clamped,cap),reads)
    case Nil{} Nil{} Nil{} Nil{} Nil{} Con{+stride,+strides}:
      Empty.absurd(BoxesRead(Placements.reversed_boxes(Nil{},acc),Placements.reversed_extents(Nil{},acc_target),Placements.reversed_extents(Nil{},acc_lows),Placements.reversed_extents(Nil{},acc_source),Placements.reversed_extents(Nil{},acc_shifts),Placements.reversed_extents(Con{stride,strides},acc_strides),clamped,cap),reads)
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+boxes} Nil{} lows source shifts strides:
      Empty.absurd(BoxesRead(Placements.reversed_boxes(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes},acc),Placements.reversed_extents(Nil{},acc_target),Placements.reversed_extents(lows,acc_lows),Placements.reversed_extents(source,acc_source),Placements.reversed_extents(shifts,acc_shifts),Placements.reversed_extents(strides,acc_strides),clamped,cap),reads)
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+boxes} Con{+total,+target} Nil{} source shifts strides:
      Empty.absurd(BoxesRead(Placements.reversed_boxes(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes},acc),Placements.reversed_extents(Con{total,target},acc_target),Placements.reversed_extents(Nil{},acc_lows),Placements.reversed_extents(source,acc_source),Placements.reversed_extents(shifts,acc_shifts),Placements.reversed_extents(strides,acc_strides),clamped,cap),reads)
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+boxes} Con{+total,+target} Con{+low,+lows} Nil{} shifts strides:
      Empty.absurd(BoxesRead(Placements.reversed_boxes(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes},acc),Placements.reversed_extents(Con{total,target},acc_target),Placements.reversed_extents(Con{low,lows},acc_lows),Placements.reversed_extents(Nil{},acc_source),Placements.reversed_extents(shifts,acc_shifts),Placements.reversed_extents(strides,acc_strides),clamped,cap),reads)
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+boxes} Con{+total,+target} Con{+low,+lows} Con{+bound,+source} Nil{} strides:
      Empty.absurd(BoxesRead(Placements.reversed_boxes(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes},acc),Placements.reversed_extents(Con{total,target},acc_target),Placements.reversed_extents(Con{low,lows},acc_lows),Placements.reversed_extents(Con{bound,source},acc_source),Placements.reversed_extents(Nil{},acc_shifts),Placements.reversed_extents(strides,acc_strides),clamped,cap),reads)
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+boxes} Con{+total,+target} Con{+low,+lows} Con{+bound,+source} Con{+shift,+shifts} Nil{}:
      Empty.absurd(BoxesRead(Placements.reversed_boxes(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes},acc),Placements.reversed_extents(Con{total,target},acc_target),Placements.reversed_extents(Con{low,lows},acc_lows),Placements.reversed_extents(Con{bound,source},acc_source),Placements.reversed_extents(Con{shift,shifts},acc_shifts),Placements.reversed_extents(Nil{},acc_strides),clamped,cap),reads)

# A list of strides after one more stride names its strides one axis later.
def strides_next(+first: U32,+rest: List<&2,U32>,strides: List<&2,U32>,+axis: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(axis),List.length(&2,U32,strides)),U32.to_nat(cap)) == True{} : Bool},+at: StridesAt(rest,strides,axis)) ->
  StridesAt(first <> rest,strides,(axis + 1 : U32)):
  match strides:
    case Nil{}: Unit{}
    case Con{+stride,+more}:
      Held.Both{+here,+later} = at
      +a = U32.to_nat(axis)
      +m = List.length(&2,U32,more)
      +next = (axis + 1 : U32)
      +one_room = Order.transitive(Nat.add(a,1n),Nat.add(a,1n+m),U32.to_nat(cap),Order.offset_le(a,1n,1n+m,Order.le_add(1n,m)),room)
      +exact = Equal.trans(Nat,U32.to_nat(next),Nat.add(a,1n),1n+a,Arithmetic.sum_exact(cap,cap_ok,axis,1,a,1n,{==},{==},one_room),Addition.successor_width(a))
      +positive = {%Equal.sym(Nat,U32.to_nat(next),1n+a,exact) : {Nat.is_lt(0n,_) == True{} : Bool} Order.lt_succ(0n,a,Order.zero_le(a)) : {Nat.is_lt(0n,U32.to_nat(next)) == True{} : Bool}}
      +nonzero = Arithmetic.word_positive_nonzero(next,positive)
      +head = {%Equal.sym(Bool,U32.is_zero(next),False{},nonzero) : {Bool.pick(U32,_,first,Tasks.nth(rest,(next - 1 : U32))) == stride : U32}
        %Equal.sym(U32,(next - 1 : U32),axis,Subtraction.subtract_add(axis,1)) : {Tasks.nth(rest,_) == stride : U32}
        here : {Tasks.nth(first <> rest,next) == stride : U32}}
      +later_room = {%Equal.sym(Nat,U32.to_nat(next),1n+a,exact) : {Nat.is_le(Nat.add(_,m),U32.to_nat(cap)) == True{} : Bool} Order.advance_room(a,m,U32.to_nat(cap),room) :
        {Nat.is_le(Nat.add(U32.to_nat(next),m),U32.to_nat(cap)) == True{} : Bool}}
      Held.Both{head,strides_next(first,rest,more,next,cap,cap_ok,later_room,later)}

# Every stride of a list is the stride its position names.
def strides_at(list: List<&2,U32>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(List.length(&2,U32,list),U32.to_nat(cap)) == True{} : Bool}) -> StridesAt(list,list,0):
  match list:
    case Nil{}: Unit{}
    case Con{+first,+rest}:
      +m = List.length(&2,U32,rest)
      +rest_room = Order.transitive(m,1n+m,U32.to_nat(cap),Order.successor(m,m,Order.reflexive(m)),room)
      Held.Both{{==},strides_next(first,rest,rest,0,cap,cap_ok,rest_room,strides_at(rest,cap,cap_ok,rest_room))}

# A list of extents has a stride for every extent.
def strides_length(extents: List<&2,U32>) -> {List.length(&2,U32,Tasks.strides_of(extents)) == List.length(&2,U32,extents) : Nat}:
  match extents:
    case Nil{}: {==}
    case Con{+extent,+rest}:
      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,rest}),Con{Grid.product(rest),Grid.strides(rest)},GridLaws.strides_con(extent,rest)) :
        {List.length(&2,U32,_) == 1n+List.length(&2,U32,rest) : Nat}
      Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,U32,Tasks.strides_of(rest)),List.length(&2,U32,rest),strides_length(rest))

def part_cell_held(+limits: Machine.Limits,+target: List<&2,U32>,+source: List<&2,U32>,+shifts: List<&2,U32>,+clamped: Bool,+readings: List<&2,List<&2,Tasks.Reading>>,+component: U32,+piece: U32,
  +room: {Nat.is_le(ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+source_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(source)) == True{} : Bool},+source_rank: {Nat.is_le(List.length(&2,U32,source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Tensors.grid_of(target)))) == True{} : Bool},+separate: {Tasks.separable(readings) == True{} : Bool},
  +classes_ok: Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Tensors.grid_of(target),target,Machine.half_word(limits)),
  +member: Moved.Member(Tasks.target_pieces(Tasks.components_of(readings),Tensors.grid_of(target),target,component),piece),
  +row_ok: Components.RowOk(limits,Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tensors.grid_of(source),source),+ends: Components.RowEnds(Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component)),
  +blocks: ViewReads.BlocksIn(Tensors.grid_of(source),source,0,Machine.half_word(limits)),+apart: {Tasks.distinct(Tasks.read_axes(Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0))) == True{} : Bool},
  +boxes: List<&2,BoxParts.BoxAxis>,+from: U32,+to: U32,+from_at: {from == BoxParts.box_source(boxes,0) : U32},+box_in: BoxParts.BoxIn(boxes),
  +extents_at: {BoxParts.box_extents(boxes) == Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece)) : List<&2,U32>},+index: Nat,+index_in: {Nat.is_lt(index,Part.total(BoxParts.box_axes(boxes))) == True{} : Bool},
  +number: Nat,+number_in: {Nat.is_lt(number,U32.to_nat(Grid.span(Tensors.grid_of(target),piece))) == True{} : Bool},
  +cells_at: {Digits.words_nat(Placements.positions(Grid.coordinates(target,Grid.start(Tensors.grid_of(target),piece)),Digits.added(BoxParts.box_starts(boxes),Targets.cell_coordinates(ViewPart.move_extents(BoxParts.box_axes(boxes)),index)))) == Targets.cell_coordinates(target,Nat.add(U32.to_nat(Grid.start(Tensors.grid_of(target),piece)),number)) : List<&2,Nat>},
  +total_room: {Nat.is_le(Nat.double(Part.total(BoxParts.box_axes(boxes))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +cell: Held.Both<Reads.KeptAll(Placements.positions(Grid.coordinates(target,Grid.start(Tensors.grid_of(target),piece)),Digits.added(BoxParts.box_starts(boxes),Targets.cell_coordinates(ViewPart.move_extents(BoxParts.box_axes(boxes)),index))),source,shifts,clamped),
    {(BoxParts.box_source(boxes,0) + Digits.stepped_digits(Targets.cell_coordinates(ViewPart.move_extents(BoxParts.box_axes(boxes)),index),ViewPart.move_steps(BoxParts.box_axes(boxes))) : U32) == (0 + TaskReads.placed_word(Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Reads.kept_indices(Placements.positions(Grid.coordinates(target,Grid.start(Tensors.grid_of(target),piece)),Digits.added(BoxParts.box_starts(boxes),Targets.cell_coordinates(ViewPart.move_extents(BoxParts.box_axes(boxes)),index))),source,shifts,clamped),Tasks.strides_of(source)) : U32) : U32}>) ->
  Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Tensors.grid_of(source),source,Tasks.box_when(Tasks.separable(readings),Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Tensors.grid_of(source),source,0)),Tensors.grid_of(source)),U32.to_nat(Part.cell_source(BoxParts.box_axes(boxes),index,from,to))):
  Held.Both{+kept,+sum} = cell
  +axes = BoxParts.box_axes(boxes)
  +steps = ViewPart.move_steps(axes)
  +held = Reads.placing_held(limits,target,source,shifts,clamped,readings,component,piece,number,
    room,nonempty,rank_room,
    source_room,source_nonempty,source_rank,separate,classes_ok,member,number_in,Placements.positions(Grid.coordinates(target,Grid.start(Tensors.grid_of(target),piece)),Digits.added(BoxParts.box_starts(boxes),Targets.cell_coordinates(ViewPart.move_extents(BoxParts.box_axes(boxes)),index))),cells_at,kept,row_ok,ends,blocks,apart)
  +offsets = ViewPart.cell_offsets(axes,index,from,to,Machine.half_word(limits),Limits.half_word_fits(limits),total_room,index_in)
  +source_sum = Equal.cong(Move.Offsets,U32,Part.offsets_source,Move.row_offsets(axes,U32.from_nat(index),from,to),
    Move.Offsets{(from + Box.stepped_sum(ViewPart.move_extents(axes),steps,index) : U32),(to + Box.stepped_sum(ViewPart.move_extents(axes),ViewPart.move_targets(axes),index) : U32)},offsets)
  +word = Equal.trans(U32,Part.cell_source(axes,index,from,to),(from + Box.stepped_sum(ViewPart.move_extents(axes),steps,index) : U32),TaskReads.placed_word(Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Reads.kept_indices(Placements.positions(Grid.coordinates(target,Grid.start(Tensors.grid_of(target),piece)),Digits.added(BoxParts.box_starts(boxes),Targets.cell_coordinates(ViewPart.move_extents(BoxParts.box_axes(boxes)),index))),source,shifts,clamped),Tasks.strides_of(source)),source_sum,
    Equal.trans(U32,(from + Box.stepped_sum(ViewPart.move_extents(axes),steps,index) : U32),(BoxParts.box_source(boxes,0) + Digits.stepped_digits(Targets.cell_coordinates(ViewPart.move_extents(BoxParts.box_axes(boxes)),index),steps) : U32),TaskReads.placed_word(Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Reads.kept_indices(Placements.positions(Grid.coordinates(target,Grid.start(Tensors.grid_of(target),piece)),Digits.added(BoxParts.box_starts(boxes),Targets.cell_coordinates(ViewPart.move_extents(BoxParts.box_axes(boxes)),index))),source,shifts,clamped),Tasks.strides_of(source)),
      Equal.trans(U32,(from + Box.stepped_sum(ViewPart.move_extents(axes),steps,index) : U32),(from + Digits.stepped_digits(Targets.cell_coordinates(ViewPart.move_extents(BoxParts.box_axes(boxes)),index),steps) : U32),
        (BoxParts.box_source(boxes,0) + Digits.stepped_digits(Targets.cell_coordinates(ViewPart.move_extents(BoxParts.box_axes(boxes)),index),steps) : U32),
        Equal.cong(U32,U32,value => (from + value : U32),Box.stepped_sum(ViewPart.move_extents(axes),steps,index),Digits.stepped_digits(Targets.cell_coordinates(ViewPart.move_extents(BoxParts.box_axes(boxes)),index),steps),
          Digits.stepped_coordinates(ViewPart.move_extents(axes),steps,index)),
        Equal.cong(U32,U32,value => (value + Digits.stepped_digits(Targets.cell_coordinates(ViewPart.move_extents(BoxParts.box_axes(boxes)),index),steps) : U32),from,BoxParts.box_source(boxes,0),from_at)),
      Equal.trans(U32,(BoxParts.box_source(boxes,0) + Digits.stepped_digits(Targets.cell_coordinates(ViewPart.move_extents(BoxParts.box_axes(boxes)),index),steps) : U32),(0 + TaskReads.placed_word(Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Reads.kept_indices(Placements.positions(Grid.coordinates(target,Grid.start(Tensors.grid_of(target),piece)),Digits.added(BoxParts.box_starts(boxes),Targets.cell_coordinates(ViewPart.move_extents(BoxParts.box_axes(boxes)),index))),source,shifts,clamped),Tasks.strides_of(source)) : U32),TaskReads.placed_word(Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Reads.kept_indices(Placements.positions(Grid.coordinates(target,Grid.start(Tensors.grid_of(target),piece)),Digits.added(BoxParts.box_starts(boxes),Targets.cell_coordinates(ViewPart.move_extents(BoxParts.box_axes(boxes)),index))),source,shifts,clamped),Tasks.strides_of(source)),sum,Arithmetic.add_zero_left(TaskReads.placed_word(Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Reads.kept_indices(Placements.positions(Grid.coordinates(target,Grid.start(Tensors.grid_of(target),piece)),Digits.added(BoxParts.box_starts(boxes),Targets.cell_coordinates(ViewPart.move_extents(BoxParts.box_axes(boxes)),index))),source,shifts,clamped),Tasks.strides_of(source))))))
  %Equal.sym(U32,Part.cell_source(axes,index,from,to),TaskReads.placed_word(Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Reads.kept_indices(Placements.positions(Grid.coordinates(target,Grid.start(Tensors.grid_of(target),piece)),Digits.added(BoxParts.box_starts(boxes),Targets.cell_coordinates(ViewPart.move_extents(BoxParts.box_axes(boxes)),index))),source,shifts,clamped),Tasks.strides_of(source)),word) : Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Tensors.grid_of(source),source,Tasks.box_when(Tasks.separable(readings),Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Tensors.grid_of(source),source,0)),Tensors.grid_of(source)),U32.to_nat(_))
  held

# A cell of a part of a placement reads a source cell its component takes.
def part_cell(+limits: Machine.Limits,+target: List<&2,U32>,+source: List<&2,U32>,+shifts: List<&2,U32>,+clamped: Bool,+readings: List<&2,List<&2,Tasks.Reading>>,+component: U32,+piece: U32,
  +room: {Nat.is_le(ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+source_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(source)) == True{} : Bool},+source_rank: {Nat.is_le(List.length(&2,U32,source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Tensors.grid_of(target)))) == True{} : Bool},+separate: {Tasks.separable(readings) == True{} : Bool},
  +classes_ok: Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Tensors.grid_of(target),target,Machine.half_word(limits)),
  +member: Moved.Member(Tasks.target_pieces(Tasks.components_of(readings),Tensors.grid_of(target),target,component),piece),
  +row_ok: Components.RowOk(limits,Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tensors.grid_of(source),source),+ends: Components.RowEnds(Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component)),
  +blocks: ViewReads.BlocksIn(Tensors.grid_of(source),source,0,Machine.half_word(limits)),+apart: {Tasks.distinct(Tasks.read_axes(Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0))) == True{} : Bool},
  +boxes: List<&2,BoxParts.BoxAxis>,+from: U32,+to: U32,+from_at: {from == BoxParts.box_source(boxes,0) : U32},+box_in: BoxParts.BoxIn(boxes),
  +extents_at: {BoxParts.box_extents(boxes) == Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece)) : List<&2,U32>},+index: Nat,+index_in: {Nat.is_lt(index,Part.total(BoxParts.box_axes(boxes))) == True{} : Bool},
  +cells_exact: {ShapeLaws.elements(Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece))) == U32.to_nat(Grid.span(Tensors.grid_of(target),piece)) : Nat},+reads: BoxesRead(boxes,target,Grid.coordinates(target,Grid.start(Tensors.grid_of(target),piece)),source,shifts,Tasks.strides_of(source),clamped,Machine.half_word(limits))) ->
  Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Tensors.grid_of(source),source,Tasks.box_when(Tasks.separable(readings),Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Tensors.grid_of(source),source,0)),Tensors.grid_of(source)),U32.to_nat(Part.cell_source(BoxParts.box_axes(boxes),index,from,to))):
  +grid = Tensors.grid_of(target)
  +first = Grid.start(Tensors.grid_of(target),piece)
  +count = Grid.span(Tensors.grid_of(target),piece)
  +extents = Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece))
  +capacity = Machine.capacity(limits)
  +capacity_ok = Limits.capacity_fits(limits)
  +cap = Machine.half_word(limits)
  +cap_ok = Limits.half_word_fits(limits)
  +cap_half = Bounds.as_nat_le(capacity,cap,Limits.capacity_within_half(limits))
  +elements = ShapeLaws.elements(target)
  +single = room
  +axes = BoxParts.box_axes(boxes)
  +digits = Targets.cell_coordinates(ViewPart.move_extents(BoxParts.box_axes(boxes)),index)
  +placed = Digits.added(BoxParts.box_starts(boxes),digits)
  +below = BoxParts.index_below(boxes,index,index_in)
  +below_box = {%extents_at : Coordinates.DigitsBelow(_,placed) BoxParts.box_below(boxes,digits,box_in,below) : Coordinates.DigitsBelow(extents,placed)}
  +number = ShapeLaws.digits_value(extents,placed)
  +number_in = {%cells_exact : {Nat.is_lt(number,_) == True{} : Bool} Coordinates.digits_below(extents,placed,below_box) : {Nat.is_lt(number,U32.to_nat(count)) == True{} : Bool}}
  +steps = Tasks.strides_of(target)
  +boxed = Box.grid_piece_box(limits,target,steps,Grid.piece_target(),single,nonempty,rank_room,piece,inside,number,number_in)
  +within = ViewGather.box_within_of(target,steps,extents,U32.to_nat(first),U32.to_nat(count),number,boxed)
  +double_room = Limits.double_bound(limits,elements,single)
  +piece_room = Limits.double_bound(limits,ShapeLaws.elements(extents),Order.transitive(ShapeLaws.elements(extents),elements,U32.to_nat(capacity),within,single))
  +total_room = BoxParts.box_room(boxes,box_in,cap,cap_ok,
    {%Equal.sym(List<&2,U32>,BoxParts.box_extents(boxes),extents,extents_at) : {Nat.is_le(Nat.double(ShapeLaws.elements(_)),U32.to_nat(cap)) == True{} : Bool} piece_room :
      {Nat.is_le(Nat.double(ShapeLaws.elements(BoxParts.box_extents(boxes))),U32.to_nat(cap)) == True{} : Bool}})
  +ending = Box.box_end_of(target,steps,extents,U32.to_nat(first),U32.to_nat(count),number,boxed)
  +position = Nat.add(U32.to_nat(first),number)
  +back = Bounds.roundtrip(position,capacity,capacity_ok,Order.transitive(position,elements,U32.to_nat(capacity),Bounds.strict_le(position,elements,ending),single))
  +located = Gathered.piece_positions(target,extents,first,U32.to_nat(count),number,cap,cap_ok,double_room,Digits.box_of_length(grid,target,count),ending,
    Digits.grid_boxes(List.length(&2,U32,target),limits,target,Grid.piece_target(),single,nonempty,rank_room,piece,inside,number,number_in,0n))
  +corner = Grid.coordinates(target,first)
  +same = Equal.trans(List<&2,U32>,Placements.positions(corner,placed),Placements.positions(corner,Targets.cell_coordinates(extents,number)),
    Grid.coordinates(target,U32.from_nat(position)),
    Equal.cong(List<&2,Nat>,List<&2,U32>,numbers => Placements.positions(corner,numbers),placed,Targets.cell_coordinates(extents,number),
      Equal.sym(List<&2,Nat>,Targets.cell_coordinates(extents,number),placed,Digits.coordinates_value(extents,placed,below_box))),located)
  +cells_at = Equal.trans(List<&2,Nat>,Digits.words_nat(Placements.positions(corner,placed)),Digits.words_nat(Grid.coordinates(target,U32.from_nat(position))),
    Targets.cell_coordinates(target,position),Equal.cong(List<&2,U32>,List<&2,Nat>,words => Digits.words_nat(words),Placements.positions(corner,placed),
      Grid.coordinates(target,U32.from_nat(position)),same),
    Equal.trans(List<&2,Nat>,Digits.words_nat(Grid.coordinates(target,U32.from_nat(position))),Targets.cell_coordinates(target,U32.to_nat(U32.from_nat(position))),
      Targets.cell_coordinates(target,position),
      Digits.coordinates_exact(target,U32.from_nat(position),cap,cap_ok,double_room,
        {%Equal.sym(Nat,U32.to_nat(U32.from_nat(position)),position,back) : {Nat.is_lt(_,elements) == True{} : Bool} ending : {Nat.is_lt(U32.to_nat(U32.from_nat(position)),elements) == True{} : Bool}}),
      Equal.cong(Nat,List<&2,Nat>,value => Targets.cell_coordinates(target,value),U32.to_nat(U32.from_nat(position)),position,back)))
  +strides = Tasks.strides_of(source)
  +strides_room = {%Equal.sym(Nat,List.length(&2,U32,strides),List.length(&2,U32,source),strides_length(source)) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
    Order.transitive(List.length(&2,U32,source),U32.to_nat(capacity),U32.to_nat(cap),source_rank,cap_half) : {Nat.is_le(List.length(&2,U32,strides),U32.to_nat(cap)) == True{} : Bool}}
  part_cell_held(limits,target,source,shifts,clamped,readings,component,piece,room,nonempty,rank_room,source_room,source_nonempty,source_rank,inside,separate,classes_ok,member,row_ok,ends,blocks,apart,boxes,from,to,from_at,box_in,extents_at,index,index_in,number,number_in,cells_at,total_room,
    boxes_cell(boxes,target,corner,source,shifts,strides,digits,strides,grid,Tensors.grid_of(source),clamped,0,0,cap,cap_ok,reads,strides_at(strides,cap,cap_ok,strides_room),below))

# ---- The parts are ready ----

# The cells of a part from a box index on, count of them, read source cells
# the component takes.
def part_cells(count: Nat,+limits: Machine.Limits,+target: List<&2,U32>,+source: List<&2,U32>,+shifts: List<&2,U32>,+clamped: Bool,+readings: List<&2,List<&2,Tasks.Reading>>,+component: U32,+piece: U32,
  +room: {Nat.is_le(ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+source_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(source)) == True{} : Bool},+source_rank: {Nat.is_le(List.length(&2,U32,source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Tensors.grid_of(target)))) == True{} : Bool},+separate: {Tasks.separable(readings) == True{} : Bool},
  +classes_ok: Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Tensors.grid_of(target),target,Machine.half_word(limits)),
  +member: Moved.Member(Tasks.target_pieces(Tasks.components_of(readings),Tensors.grid_of(target),target,component),piece),
  +row_ok: Components.RowOk(limits,Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tensors.grid_of(source),source),+ends: Components.RowEnds(Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component)),
  +blocks: ViewReads.BlocksIn(Tensors.grid_of(source),source,0,Machine.half_word(limits)),+apart: {Tasks.distinct(Tasks.read_axes(Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0))) == True{} : Bool},
  +boxes: List<&2,BoxParts.BoxAxis>,+from: U32,+to: U32,+from_at: {from == BoxParts.box_source(boxes,0) : U32},+box_in: BoxParts.BoxIn(boxes),
  +extents_at: {BoxParts.box_extents(boxes) == Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece)) : List<&2,U32>},+cells_exact: {ShapeLaws.elements(Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece))) == U32.to_nat(Grid.span(Tensors.grid_of(target),piece)) : Nat},+reads: BoxesRead(boxes,target,Grid.coordinates(target,Grid.start(Tensors.grid_of(target),piece)),source,shifts,Tasks.strides_of(source),clamped,Machine.half_word(limits)),+first: Nat,+within: {Nat.is_le(Nat.add(first,count),Part.total(BoxParts.box_axes(boxes))) == True{} : Bool}) ->
  Part.CellsHeld(BoxParts.box_axes(boxes),from,to,Moved.taken_spans(Tasks.box_pieces(Tensors.grid_of(source),source,Tasks.box_when(Tasks.separable(readings),Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Tensors.grid_of(source),source,0)),Tensors.grid_of(source)),first,count):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      Held.Both{part_cell(limits,target,source,shifts,clamped,readings,component,piece,room,nonempty,rank_room,source_room,source_nonempty,source_rank,inside,separate,classes_ok,member,row_ok,ends,blocks,apart,boxes,from,to,from_at,box_in,extents_at,first,Order.room_index(first,rest,Part.total(BoxParts.box_axes(boxes)),within),cells_exact,reads),
        part_cells(rest,limits,target,source,shifts,clamped,readings,component,piece,room,nonempty,rank_room,source_room,source_nonempty,source_rank,inside,separate,classes_ok,member,row_ok,ends,blocks,apart,boxes,from,to,from_at,box_in,extents_at,cells_exact,reads,1n+first,Order.advance_room(first,rest,Part.total(BoxParts.box_axes(boxes)),within))}

# A box axis over more than one index steps by a source stride, or stays.
def box_stride(+count: U32,+source_stride: U32,+source_offset: U32,+start: Nat,+low: U32,+stride: U32,+shift: U32,+bound: U32,+clamped: Bool,+cap: U32,+word: U32,
  +box: BoxRead<count,source_stride,source_offset,start,low,stride,shift,bound,clamped,cap>,+bounded: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool},
  +wide: {U32.is_lt(1,count) == True{} : Bool}) -> {Nat.is_le(Nat.double(U32.to_nat(source_stride)),U32.to_nat(word)) == True{} : Bool}:
  BoxRead{+first,+length,+from,+step,+read,+stepped,+offset,+after,+before,+ends} = box
  match read:
    case Reads.Middle{+step_one,+placed,+inside}:
      %Equal.sym(U32,source_stride,(step * stride : U32),stepped) : {Nat.is_le(Nat.double(U32.to_nat(_)),U32.to_nat(word)) == True{} : Bool}
      %Equal.sym(U32,step,1,step_one) : {Nat.is_le(Nat.double(U32.to_nat((_ * stride : U32))),U32.to_nat(word)) == True{} : Bool}
      %Equal.sym(U32,(1 * stride : U32),stride,Equal.trans(U32,(1 * stride : U32),(stride * 1 : U32),stride,Arithmetic.multiply_commutative(1,stride),Arithmetic.multiply_one(stride))) :
        {Nat.is_le(Nat.double(U32.to_nat(_)),U32.to_nat(word)) == True{} : Bool}
      bounded
    case Reads.Before{+clamp,+step_zero,+source_zero,+positive,+ends_before}:
      %Equal.sym(U32,source_stride,(step * stride : U32),stepped) : {Nat.is_le(Nat.double(U32.to_nat(_)),U32.to_nat(word)) == True{} : Bool}
      %Equal.sym(U32,step,0,step_zero) : {Nat.is_le(Nat.double(U32.to_nat((_ * stride : U32))),U32.to_nat(word)) == True{} : Bool}
      %Equal.sym(U32,(0 * stride : U32),0,Arithmetic.multiply_zero_left(stride)) : {Nat.is_le(Nat.double(U32.to_nat(_)),U32.to_nat(word)) == True{} : Bool}
      Order.zero_le(U32.to_nat(word))
    case Reads.After{+clamp,+step_zero,+source_last,+starts}:
      %Equal.sym(U32,source_stride,(step * stride : U32),stepped) : {Nat.is_le(Nat.double(U32.to_nat(_)),U32.to_nat(word)) == True{} : Bool}
      %Equal.sym(U32,step,0,step_zero) : {Nat.is_le(Nat.double(U32.to_nat((_ * stride : U32))),U32.to_nat(word)) == True{} : Bool}
      %Equal.sym(U32,(0 * stride : U32),0,Arithmetic.multiply_zero_left(stride)) : {Nat.is_le(Nat.double(U32.to_nat(_)),U32.to_nat(word)) == True{} : Bool}
      Order.zero_le(U32.to_nat(word))
    case Reads.Unused{+none}:
      +near = Nat.add(U32.to_nat(low),start)
      +f = U32.to_nat(first)
      +c = U32.to_nat(count)
      +ended = {%none : {Nat.is_le(Nat.add(near,c),Nat.add(f,U32.to_nat(_))) == True{} : Bool} before : {Nat.is_le(Nat.add(near,c),Nat.add(f,0n)) == True{} : Bool}}
      +squeezed = Order.transitive(Nat.add(near,c),Nat.add(f,0n),near,ended,{%Equal.sym(Nat,Nat.add(f,0n),f,Addition.zero_right(f)) : {Nat.is_le(_,near) == True{} : Bool} after : {Nat.is_le(Nat.add(f,0n),near) == True{} : Bool}})
      +empty = Order.zero_below(c,Order.cancel_offset_le(near,c,0n,{%Equal.sym(Nat,Nat.add(near,0n),near,Addition.zero_right(near)) : {Nat.is_le(Nat.add(near,c),_) == True{} : Bool} squeezed :
        {Nat.is_le(Nat.add(near,c),Nat.add(near,0n)) == True{} : Bool}}))
      Empty.absurd({Nat.is_le(Nat.double(U32.to_nat(source_stride)),U32.to_nat(word)) == True{} : Bool},
        WordEncoding.false_true(Equal.trans(Bool,False{},U32.is_lt(1,count),True{},
          {%Equal.sym(U32,count,0,Arithmetic.to_nat_injective(count,0,empty)) : {False{} == U32.is_lt(1,_) : Bool} {==} : {False{} == U32.is_lt(1,count) : Bool}},wide)))

# The boxes of a part step at most half a word wherever they span more than
# one index.
def boxes_strides(boxes: List<&2,BoxParts.BoxAxis>,target: List<&2,U32>,lows: List<&2,U32>,source: List<&2,U32>,shifts: List<&2,U32>,strides: List<&2,U32>,
  +clamped: Bool,+cap: U32,+word: U32,+reads: BoxesRead(boxes,target,lows,source,shifts,strides,clamped,cap),+bounded: Digits.StridesIn(strides,word)) ->
  Part.StridesRoom(BoxParts.box_axes(boxes),word):
  match boxes target lows source shifts strides:
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+boxes} Con{+total,+target} Con{+low,+lows}
      Con{+bound,+source} Con{+shift,+shifts} Con{+stride,+strides}:
      Held.Both{+box,Held.Both{+shift_room,+reads_rest}} = reads
      Held.Both{+stride_bounded,+bounded_rest} = bounded
      Held.Both{Held.by_cases(Logic.When(U32.is_lt(1,count),{Nat.is_le(Nat.double(U32.to_nat(source_stride)),U32.to_nat(word)) == True{} : Bool}),U32.is_lt(1,count),
          +wide => {%Equal.sym(Bool,U32.is_lt(1,count),True{},wide) : Logic.When(_,{Nat.is_le(Nat.double(U32.to_nat(source_stride)),U32.to_nat(word)) == True{} : Bool})
            box_stride(count,source_stride,source_offset,start,low,stride,shift,bound,clamped,cap,word,box,stride_bounded,wide) :
            Logic.When(U32.is_lt(1,count),{Nat.is_le(Nat.double(U32.to_nat(source_stride)),U32.to_nat(word)) == True{} : Bool})},
          +narrow => {%Equal.sym(Bool,U32.is_lt(1,count),False{},narrow) : Logic.When(_,{Nat.is_le(Nat.double(U32.to_nat(source_stride)),U32.to_nat(word)) == True{} : Bool})
            Unit{} : Logic.When(U32.is_lt(1,count),{Nat.is_le(Nat.double(U32.to_nat(source_stride)),U32.to_nat(word)) == True{} : Bool})}),
        boxes_strides(boxes,target,lows,source,shifts,strides,clamped,cap,word,reads_rest,bounded_rest)}
    case Nil{} target lows source shifts strides:
      Unit{}
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+boxes} Nil{} lows source shifts strides:
      Empty.absurd(Part.StridesRoom(BoxParts.box_axes(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes}),word),reads)
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+boxes} Con{+total,+target} Nil{} source shifts strides:
      Empty.absurd(Part.StridesRoom(BoxParts.box_axes(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes}),word),reads)
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+boxes} Con{+total,+target} Con{+low,+lows} Nil{} shifts strides:
      Empty.absurd(Part.StridesRoom(BoxParts.box_axes(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes}),word),reads)
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+boxes} Con{+total,+target} Con{+low,+lows} Con{+bound,+source} Nil{} strides:
      Empty.absurd(Part.StridesRoom(BoxParts.box_axes(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes}),word),reads)
    case Con{BoxParts.BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+boxes} Con{+total,+target} Con{+low,+lows} Con{+bound,+source} Con{+shift,+shifts} Nil{}:
      Empty.absurd(Part.StridesRoom(BoxParts.box_axes(Con{BoxParts.BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},boxes}),word),reads)

# A part of a placement is ready: its cells fit a word, it is not empty, it
# steps at most half a word, its cells read held source cells and write
# target cells below capacity.
def part_ready(+limits: Machine.Limits,+target: List<&2,U32>,+source: List<&2,U32>,+shifts: List<&2,U32>,+clamped: Bool,+readings: List<&2,List<&2,Tasks.Reading>>,+component: U32,+piece: U32,
  +room: {Nat.is_le(ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+source_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(source)) == True{} : Bool},+source_rank: {Nat.is_le(List.length(&2,U32,source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Tensors.grid_of(target)))) == True{} : Bool},+separate: {Tasks.separable(readings) == True{} : Bool},
  +classes_ok: Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Tensors.grid_of(target),target,Machine.half_word(limits)),
  +member: Moved.Member(Tasks.target_pieces(Tasks.components_of(readings),Tensors.grid_of(target),target,component),piece),
  +row_ok: Components.RowOk(limits,Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tensors.grid_of(source),source),+ends: Components.RowEnds(Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component)),
  +blocks: ViewReads.BlocksIn(Tensors.grid_of(source),source,0,Machine.half_word(limits)),+apart: {Tasks.distinct(Tasks.read_axes(Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0))) == True{} : Bool},
  +boxes: List<&2,BoxParts.BoxAxis>,+from: U32,+to: U32,+from_at: {from == BoxParts.box_source(boxes,0) : U32},+box_in: BoxParts.BoxIn(boxes),
  +extents_at: {BoxParts.box_extents(boxes) == Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece)) : List<&2,U32>},+cells_exact: {ShapeLaws.elements(Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece))) == U32.to_nat(Grid.span(Tensors.grid_of(target),piece)) : Nat},+reads: BoxesRead(boxes,target,Grid.coordinates(target,Grid.start(Tensors.grid_of(target),piece)),source,shifts,Tasks.strides_of(source),clamped,Machine.half_word(limits)),+word: U32,+bounded: Digits.StridesIn(Tasks.strides_of(source),word),+capacity: U32,
  +holds: {Nat.is_le(ShapeLaws.elements(Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece))),U32.to_nat(capacity)) == True{} : Bool},+to_at: {to == BoxParts.box_target(boxes,0) : U32}) ->
  MoveParts.PartReady(BoxParts.box_axes(boxes),from,to,Machine.half_word(limits),word,Moved.taken_spans(Tasks.box_pieces(Tensors.grid_of(source),source,Tasks.box_when(Tasks.separable(readings),Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Tensors.grid_of(source),source,0)),Tensors.grid_of(source)),capacity):
  +cap = Machine.half_word(limits)
  +cap_ok = Limits.half_word_fits(limits)
  +extents = Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece))
  +elements = ShapeLaws.elements(target)
  +single = room
  +total = Part.total(BoxParts.box_axes(boxes))
  +positive = BoxParts.box_positive(boxes,box_in)
  +extents_positive = {%extents_at : {Nat.is_lt(0n,ShapeLaws.elements(_)) == True{} : Bool}
    Bounds.nat_trans(0n,total,ShapeLaws.elements(BoxParts.box_extents(boxes)),positive,BoxParts.box_total_le(boxes,box_in)) : {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}}
  +zero_in = {%cells_exact : {Nat.is_lt(0n,_) == True{} : Bool} extents_positive : {Nat.is_lt(0n,U32.to_nat(Grid.span(Tensors.grid_of(target),piece))) == True{} : Bool}}
  +boxed = Box.grid_piece_box(limits,target,Tasks.strides_of(target),Grid.piece_target(),single,nonempty,rank_room,piece,inside,0n,zero_in)
  +within = ViewGather.box_within_of(target,Tasks.strides_of(target),extents,U32.to_nat(Grid.start(Tensors.grid_of(target),piece)),U32.to_nat(Grid.span(Tensors.grid_of(target),piece)),0n,boxed)
  +piece_room = {%Equal.sym(List<&2,U32>,BoxParts.box_extents(boxes),extents,extents_at) : {Nat.is_le(Nat.double(ShapeLaws.elements(_)),U32.to_nat(cap)) == True{} : Bool}
    Limits.double_bound(limits,ShapeLaws.elements(extents),Order.transitive(ShapeLaws.elements(extents),elements,U32.to_nat(Machine.capacity(limits)),within,single)) :
    {Nat.is_le(Nat.double(ShapeLaws.elements(BoxParts.box_extents(boxes))),U32.to_nat(cap)) == True{} : Bool}}
  +holds_box = {%Equal.sym(List<&2,U32>,BoxParts.box_extents(boxes),extents,extents_at) : {Nat.is_le(ShapeLaws.elements(_),U32.to_nat(capacity)) == True{} : Bool} holds :
    {Nat.is_le(ShapeLaws.elements(BoxParts.box_extents(boxes)),U32.to_nat(capacity)) == True{} : Bool}}
  Held.Both{BoxParts.box_room(boxes,box_in,cap,cap_ok,piece_room),Held.Both{positive,
    Held.Both{boxes_strides(boxes,target,Grid.coordinates(target,Grid.start(Tensors.grid_of(target),piece)),source,shifts,Tasks.strides_of(source),clamped,cap,word,reads,bounded),
      Held.Both{part_cells(total,limits,target,source,shifts,clamped,readings,component,piece,room,nonempty,rank_room,source_room,source_nonempty,source_rank,inside,separate,classes_ok,member,row_ok,ends,blocks,apart,boxes,from,to,from_at,box_in,extents_at,cells_exact,reads,0n,Order.reflexive(total)),
        {%Equal.sym(U32,to,BoxParts.box_target(boxes,0),to_at) : Part.CellsBelow(BoxParts.box_axes(boxes),from,_,capacity,0n,total)
          BoxParts.box_cells_below(total,boxes,box_in,from,cap,cap_ok,piece_room,capacity,holds_box,0n,Order.reflexive(total)) :
          Part.CellsBelow(BoxParts.box_axes(boxes),from,to,capacity,0n,total)}}}}}

# Lists reversed twice are the lists.
def reads_twice(+boxes: List<&2,BoxParts.BoxAxis>,+target: List<&2,U32>,+lows: List<&2,U32>,+source: List<&2,U32>,+shifts: List<&2,U32>,+strides: List<&2,U32>,+clamped: Bool,+cap: U32,
  +reads: BoxesRead(boxes,Placements.reversed_extents(Placements.reversed_extents(target,[]),[]),Placements.reversed_extents(Placements.reversed_extents(lows,[]),[]),Placements.reversed_extents(Placements.reversed_extents(source,[]),[]),
    Placements.reversed_extents(Placements.reversed_extents(shifts,[]),[]),Placements.reversed_extents(Placements.reversed_extents(strides,[]),[]),clamped,cap)) -> BoxesRead(boxes,target,lows,source,shifts,strides,clamped,cap):
  %Placements.extents_twice(target,[]) : BoxesRead(boxes,_,lows,source,shifts,strides,clamped,cap)
  %Placements.extents_twice(lows,[]) : BoxesRead(boxes,Placements.reversed_extents(Placements.reversed_extents(target,[]),[]),_,source,shifts,strides,clamped,cap)
  %Placements.extents_twice(source,[]) : BoxesRead(boxes,Placements.reversed_extents(Placements.reversed_extents(target,[]),[]),Placements.reversed_extents(Placements.reversed_extents(lows,[]),[]),_,shifts,strides,clamped,cap)
  %Placements.extents_twice(shifts,[]) : BoxesRead(boxes,Placements.reversed_extents(Placements.reversed_extents(target,[]),[]),Placements.reversed_extents(Placements.reversed_extents(lows,[]),[]),
    Placements.reversed_extents(Placements.reversed_extents(source,[]),[]),_,strides,clamped,cap)
  %Placements.extents_twice(strides,[]) : BoxesRead(boxes,Placements.reversed_extents(Placements.reversed_extents(target,[]),[]),Placements.reversed_extents(Placements.reversed_extents(lows,[]),[]),
    Placements.reversed_extents(Placements.reversed_extents(source,[]),[]),Placements.reversed_extents(Placements.reversed_extents(shifts,[]),[]),_,clamped,cap)
  reads

# The parts of the finished prefixes of a placement are ready.
def parts_ready(list: List<&2,Placements.Prefix>,+limits: Machine.Limits,+target: List<&2,U32>,+source: List<&2,U32>,+shifts: List<&2,U32>,+clamped: Bool,+readings: List<&2,List<&2,Tasks.Reading>>,+component: U32,+piece: U32,
  +room: {Nat.is_le(ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+source_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(source)) == True{} : Bool},+source_rank: {Nat.is_le(List.length(&2,U32,source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Tensors.grid_of(target)))) == True{} : Bool},+separate: {Tasks.separable(readings) == True{} : Bool},
  +classes_ok: Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Tensors.grid_of(target),target,Machine.half_word(limits)),
  +member: Moved.Member(Tasks.target_pieces(Tasks.components_of(readings),Tensors.grid_of(target),target,component),piece),
  +row_ok: Components.RowOk(limits,Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tensors.grid_of(source),source),+ends: Components.RowEnds(Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component)),
  +blocks: ViewReads.BlocksIn(Tensors.grid_of(source),source,0,Machine.half_word(limits)),+apart: {Tasks.distinct(Tasks.read_axes(Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0))) == True{} : Bool},+word: U32,+bounded: Digits.StridesIn(Tasks.strides_of(source),word),+capacity: U32,+holds: {Nat.is_le(ShapeLaws.elements(Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece))),U32.to_nat(capacity)) == True{} : Bool},+cells_exact: {ShapeLaws.elements(Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece))) == U32.to_nat(Grid.span(Tensors.grid_of(target),piece)) : Nat},
  +reads: PrefixesRead(list,Placements.reversed_extents(target,[]),Placements.reversed_extents(Grid.coordinates(target,Grid.start(Tensors.grid_of(target),piece)),[]),Placements.reversed_extents(source,[]),Placements.reversed_extents(shifts,[]),Placements.reversed_extents(Tasks.strides_of(source),[]),clamped,Machine.half_word(limits)),
  +held: Placements.PrefixesIn(list,Placements.reversed_extents(Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece)),[]),[])) ->
  MoveParts.PartsReady(Placements.prefix_parts(list),Machine.half_word(limits),word,Moved.taken_spans(Tasks.box_pieces(Tensors.grid_of(source),source,Tasks.box_when(Tasks.separable(readings),Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Tensors.grid_of(source),source,0)),Tensors.grid_of(source)),capacity):
  match list:
    case Nil{}: Unit{}
    case Con{Placements.Prefix{+boxes,+from,+to},+more}:
      Held.Both{+box_reads,Held.Both{+from_at,+reads_more}} = reads
      Held.Both{+prefix_in,Held.Both{+to_at,+held_more}} = held
      +cap = Machine.half_word(limits)
      +box = Placements.reversed_boxes(boxes,[])
      +box_in = Placements.prefix_box_in(boxes,Placements.reversed_extents(Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece)),[]),[],prefix_in,Unit{})
      +extents_at = Equal.trans(List<&2,U32>,BoxParts.box_extents(box),Placements.reversed_extents(Placements.reversed_extents(Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece)),[]),[]),Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece)),
        Placements.prefix_box_extents(boxes,Placements.reversed_extents(Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece)),[]),[],prefix_in),Placements.extents_twice(Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece)),[]))
      +box_read = reads_twice(box,target,Grid.coordinates(target,Grid.start(Tensors.grid_of(target),piece)),source,shifts,Tasks.strides_of(source),clamped,cap,
        boxes_reversed(boxes,Placements.reversed_extents(target,[]),Placements.reversed_extents(Grid.coordinates(target,Grid.start(Tensors.grid_of(target),piece)),[]),Placements.reversed_extents(source,[]),Placements.reversed_extents(shifts,[]),Placements.reversed_extents(Tasks.strides_of(source),[]),[],[],[],[],[],[],clamped,cap,box_reads,Unit{}))
      Held.Both{part_ready(limits,target,source,shifts,clamped,readings,component,piece,room,nonempty,rank_room,source_room,source_nonempty,source_rank,inside,separate,classes_ok,member,row_ok,ends,blocks,apart,box,from,to,from_at,box_in,extents_at,cells_exact,box_read,word,bounded,capacity,holds,to_at),
        parts_ready(more,limits,target,source,shifts,clamped,readings,component,piece,room,nonempty,rank_room,source_room,source_nonempty,source_rank,inside,separate,classes_ok,member,row_ok,ends,blocks,apart,word,bounded,capacity,holds,cells_exact,reads_more,held_more)}

# The parts that place a source in a target piece are ready.
def placements_ready(+own: List<&2,List<&2,Tasks.Segment>>,+limits: Machine.Limits,+target: List<&2,U32>,+source: List<&2,U32>,+shifts: List<&2,U32>,+clamped: Bool,+readings: List<&2,List<&2,Tasks.Reading>>,+component: U32,+piece: U32,
  +room: {Nat.is_le(ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+source_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(source)) == True{} : Bool},+source_rank: {Nat.is_le(List.length(&2,U32,source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Tensors.grid_of(target)))) == True{} : Bool},+separate: {Tasks.separable(readings) == True{} : Bool},
  +classes_ok: Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Tensors.grid_of(target),target,Machine.half_word(limits)),
  +member: Moved.Member(Tasks.target_pieces(Tasks.components_of(readings),Tensors.grid_of(target),target,component),piece),
  +row_ok: Components.RowOk(limits,Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tensors.grid_of(source),source),+ends: Components.RowEnds(Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component)),
  +blocks: ViewReads.BlocksIn(Tensors.grid_of(source),source,0,Machine.half_word(limits)),+apart: {Tasks.distinct(Tasks.read_axes(Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0))) == True{} : Bool},+word: U32,+bounded: Digits.StridesIn(Tasks.strides_of(source),word),+capacity: U32,+holds: {Nat.is_le(ShapeLaws.elements(Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece))),U32.to_nat(capacity)) == True{} : Bool},+cells_exact: {ShapeLaws.elements(Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece))) == U32.to_nat(Grid.span(Tensors.grid_of(target),piece)) : Nat},
  +axes_in: Placements.AxesIn(own,Grid.coordinates(target,Grid.start(Tensors.grid_of(target),piece)),Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece)),Tasks.strides_of(source),Machine.half_word(limits)),
  +axes: AxesReading(own,target,Grid.coordinates(target,Grid.start(Tensors.grid_of(target),piece)),Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece)),source,shifts,Tasks.strides_of(source),Tasks.strides_of(Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece))),clamped,Machine.half_word(limits))) ->
  MoveParts.PartsReady(Tasks.placements(own,source,Tensors.grid_of(target),target,Grid.start(Tensors.grid_of(target),piece),Grid.span(Tensors.grid_of(target),piece)),Machine.half_word(limits),word,Moved.taken_spans(Tasks.box_pieces(Tensors.grid_of(source),source,Tasks.box_when(Tasks.separable(readings),Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Tensors.grid_of(source),source,0)),Tensors.grid_of(source)),capacity):
  +cap = Machine.half_word(limits)
  +cap_ok = Limits.half_word_fits(limits)
  +prefixes = Placements.placed_prefixes(own,Grid.coordinates(target,Grid.start(Tensors.grid_of(target),piece)),Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece)),Tasks.strides_of(source),Tasks.strides_of(Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece))),[Placements.Prefix{[],0,0}])
  %Equal.sym(List<&2,Move.Part>,Tasks.placements(own,source,Tensors.grid_of(target),target,Grid.start(Tensors.grid_of(target),piece),Grid.span(Tensors.grid_of(target),piece)),Placements.prefix_parts(prefixes),
    Placements.placements_prefixes(own,source,Tensors.grid_of(target),target,Grid.start(Tensors.grid_of(target),piece),Grid.span(Tensors.grid_of(target),piece))) : MoveParts.PartsReady(_,cap,word,Moved.taken_spans(Tasks.box_pieces(Tensors.grid_of(source),source,Tasks.box_when(Tasks.separable(readings),Layout.placed_readings(target,source,shifts,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),Tasks.chosen(Tasks.class_lists(Tasks.target_rank(readings),readings),component),Tensors.grid_of(source),source,0)),Tensors.grid_of(source)),capacity)
  parts_ready(prefixes,limits,target,source,shifts,clamped,readings,component,piece,room,nonempty,rank_room,source_room,source_nonempty,source_rank,inside,separate,classes_ok,member,row_ok,ends,blocks,apart,word,bounded,capacity,holds,cells_exact,
    prefixes_read(own,target,Grid.coordinates(target,Grid.start(Tensors.grid_of(target),piece)),Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece)),source,shifts,Tasks.strides_of(source),Tasks.strides_of(Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece))),[Placements.Prefix{[],0,0}],[],[],[],[],[],clamped,cap,cap_ok,axes,
      Held.Both{Unit{},Held.Both{{==},Unit{}}}),
    Placements.prefixes_in(own,Grid.coordinates(target,Grid.start(Tensors.grid_of(target),piece)),Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece)),Tasks.strides_of(source),Tasks.strides_of(Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece))),[Placements.Prefix{[],0,0}],[],Placements.axes_boxed(own,Grid.coordinates(target,Grid.start(Tensors.grid_of(target),piece)),Grid.box_of(Tensors.grid_of(target),target,Grid.span(Tensors.grid_of(target),piece)),Tasks.strides_of(source),cap,cap_ok,axes_in),
      Held.Both{Unit{},Held.Both{{==},Unit{}}}))

# ---- The axes of a source in a piece ----

# Every axis of a source placed in a target: its segments end inside cap and
# read the source as placed, the source from its shift ends inside cap, and
# the lists end together.
def SourceAxes(segments: List<&2,List<&2,Tasks.Segment>>,target: List<&2,U32>,source: List<&2,U32>,shifts: List<&2,U32>,+clamped: Bool,+cap: U32) -> Data:
  match segments target source shifts:
    case Con{own,segments} Con{+total,target} Con{+bound,source} Con{+shift,shifts}:
      Held.Both<Placements.SegmentsIn(own,cap),Held.Both<SegmentsReading(own,shift,bound,clamped),
        Held.Both<{Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},SourceAxes(segments,target,source,shifts,clamped,cap)>>>
    case Nil{} Nil{} Nil{} Nil{}:
      Unit
    case segments target source shifts:
      Empty

def axes_joined(+own: List<&2,Tasks.Segment>,+segments: List<&2,List<&2,Tasks.Segment>>,+target: List<&2,U32>,+lows: List<&2,U32>,+extents: List<&2,U32>,+source: List<&2,U32>,+shifts: List<&2,U32>,+source_strides: List<&2,U32>,+piece_strides: List<&2,U32>,+total: U32,+low: U32,+extent: U32,+bound: U32,+shift: U32,
  +source_stride: U32,+piece_stride: U32,+clamped: Bool,+cap: U32,+piece_room: {Nat.is_le(Nat.add(U32.to_nat(low),U32.to_nat(extent)),U32.to_nat(cap)) == True{} : Bool},
  +segments_in: Placements.SegmentsIn(own,cap),+reading: SegmentsReading(own,shift,bound,clamped),+shift_room: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(cap)) == True{} : Bool},
  +later: Held.Both<Placements.AxesIn(segments,lows,extents,source_strides,cap),AxesReading(segments,target,lows,extents,source,shifts,source_strides,piece_strides,clamped,cap)>) ->
  Held.Both<Placements.AxesIn(Con{own,segments},Con{low,lows},Con{extent,extents},Con{source_stride,source_strides},cap),AxesReading(Con{own,segments},Con{total,target},Con{low,lows},Con{extent,extents},Con{bound,source},Con{shift,shifts},Con{source_stride,source_strides},Con{piece_stride,piece_strides},clamped,cap)>:
  Held.Both{+in_rest,+reading_rest} = later
  Held.Both{Held.Both{piece_room,Held.Both{segments_in,in_rest}},Held.Both{piece_room,Held.Both{segments_in,Held.Both{reading,Held.Both{shift_room,reading_rest}}}}}

# A source whose axes are placed in a target reads every axis of a piece
# whose axes from its low corner fit cap.
def axes_placed(+segments: List<&2,List<&2,Tasks.Segment>>,+target: List<&2,U32>,+lows: List<&2,U32>,+extents: List<&2,U32>,+source: List<&2,U32>,+shifts: List<&2,U32>,+source_strides: List<&2,U32>,+piece_strides: List<&2,U32>,+clamped: Bool,+cap: U32,
  +rooms: Digits.Rooms(Digits.words_nat(lows),extents,cap),+axes: SourceAxes(segments,target,source,shifts,clamped,cap),+lows_length: {List.length(&2,U32,lows) == List.length(&2,U32,target) : Nat},+extents_length: {List.length(&2,U32,extents) == List.length(&2,U32,target) : Nat},+strides_length: {List.length(&2,U32,source_strides) == List.length(&2,U32,source) : Nat},+piece_length: {List.length(&2,U32,piece_strides) == List.length(&2,U32,extents) : Nat}) ->
  Held.Both<Placements.AxesIn(segments,lows,extents,source_strides,cap),AxesReading(segments,target,lows,extents,source,shifts,source_strides,piece_strides,clamped,cap)>:
  match segments target lows extents source shifts source_strides piece_strides:
    case Con{+own,+segments} Con{+total,+target} Con{+low,+lows} Con{+extent,+extents} Con{+bound,+source} Con{+shift,+shifts} Con{+source_stride,+source_strides} Con{+piece_stride,+piece_strides}:
      KernelGemm.Both{+room,+rooms_rest} = rooms
      Held.Both{+segments_in,Held.Both{+reading,Held.Both{+shift_room,+axes_rest}}} = axes
      +reach = Nat.add(U32.to_nat(low),U32.to_nat(extent))
      axes_joined(own,segments,target,lows,extents,source,shifts,source_strides,piece_strides,total,low,extent,bound,shift,source_stride,piece_stride,clamped,cap,
        Order.transitive(reach,Nat.double(reach),U32.to_nat(cap),Order.le_double(reach),room),segments_in,reading,shift_room,
        axes_placed(segments,target,lows,extents,source,shifts,source_strides,piece_strides,clamped,cap,rooms_rest,axes_rest,
          Equal.cong(Nat,Nat,count => Digits.before(count),1n+List.length(&2,U32,lows),1n+List.length(&2,U32,target),lows_length),Equal.cong(Nat,Nat,count => Digits.before(count),1n+List.length(&2,U32,extents),1n+List.length(&2,U32,target),extents_length),
          Equal.cong(Nat,Nat,count => Digits.before(count),1n+List.length(&2,U32,source_strides),1n+List.length(&2,U32,source),strides_length),Equal.cong(Nat,Nat,count => Digits.before(count),1n+List.length(&2,U32,piece_strides),1n+List.length(&2,U32,extents),piece_length)))
    case Nil{} Nil{} Nil{} Nil{} Nil{} Nil{} Nil{} Nil{}:
      Held.Both{{==},Unit{}}
    case Con{+own,+segments} Nil{} +lows +extents +source +shifts +source_strides +piece_strides:
      Empty.absurd(Held.Both<Placements.AxesIn(Con{own,segments},lows,extents,source_strides,cap),AxesReading(Con{own,segments},Nil{},lows,extents,source,shifts,source_strides,piece_strides,clamped,cap)>,axes)
    case Con{+own,+segments} Con{+total,+target} Nil{} +extents +source +shifts +source_strides +piece_strides:
      Empty.absurd(Held.Both<Placements.AxesIn(Con{own,segments},Nil{},extents,source_strides,cap),AxesReading(Con{own,segments},Con{total,target},Nil{},extents,source,shifts,source_strides,piece_strides,clamped,cap)>,Digits.zero_successor(List.length(&2,U32,target),lows_length))
    case Con{+own,+segments} Con{+total,+target} Con{+low,+lows} Nil{} +source +shifts +source_strides +piece_strides:
      Empty.absurd(Held.Both<Placements.AxesIn(Con{own,segments},Con{low,lows},Nil{},source_strides,cap),AxesReading(Con{own,segments},Con{total,target},Con{low,lows},Nil{},source,shifts,source_strides,piece_strides,clamped,cap)>,Digits.zero_successor(List.length(&2,U32,target),extents_length))
    case Con{+own,+segments} Con{+total,+target} Con{+low,+lows} Con{+extent,+extents} Nil{} +shifts +source_strides +piece_strides:
      Empty.absurd(Held.Both<Placements.AxesIn(Con{own,segments},Con{low,lows},Con{extent,extents},source_strides,cap),AxesReading(Con{own,segments},Con{total,target},Con{low,lows},Con{extent,extents},Nil{},shifts,source_strides,piece_strides,clamped,cap)>,axes)
    case Con{+own,+segments} Con{+total,+target} Con{+low,+lows} Con{+extent,+extents} Con{+bound,+source} Nil{} +source_strides +piece_strides:
      Empty.absurd(Held.Both<Placements.AxesIn(Con{own,segments},Con{low,lows},Con{extent,extents},source_strides,cap),AxesReading(Con{own,segments},Con{total,target},Con{low,lows},Con{extent,extents},Con{bound,source},Nil{},source_strides,piece_strides,clamped,cap)>,axes)
    case Con{+own,+segments} Con{+total,+target} Con{+low,+lows} Con{+extent,+extents} Con{+bound,+source} Con{+shift,+shifts} Nil{} +piece_strides:
      Empty.absurd(Held.Both<Placements.AxesIn(Con{own,segments},Con{low,lows},Con{extent,extents},Nil{},cap),AxesReading(Con{own,segments},Con{total,target},Con{low,lows},Con{extent,extents},Con{bound,source},Con{shift,shifts},Nil{},piece_strides,clamped,cap)>,Digits.zero_successor(List.length(&2,U32,source),strides_length))
    case Con{+own,+segments} Con{+total,+target} Con{+low,+lows} Con{+extent,+extents} Con{+bound,+source} Con{+shift,+shifts} Con{+source_stride,+source_strides} Nil{}:
      Empty.absurd(Held.Both<Placements.AxesIn(Con{own,segments},Con{low,lows},Con{extent,extents},Con{source_stride,source_strides},cap),AxesReading(Con{own,segments},Con{total,target},Con{low,lows},Con{extent,extents},Con{bound,source},Con{shift,shifts},Con{source_stride,source_strides},Nil{},clamped,cap)>,Digits.zero_successor(List.length(&2,U32,extents),piece_length))
    case Nil{} Con{+total,+target} +lows +extents +source +shifts +source_strides +piece_strides:
      Empty.absurd(Held.Both<Placements.AxesIn(Nil{},lows,extents,source_strides,cap),AxesReading(Nil{},Con{total,target},lows,extents,source,shifts,source_strides,piece_strides,clamped,cap)>,axes)
    case Nil{} Nil{} Con{+low,+lows} +extents +source +shifts +source_strides +piece_strides:
      Empty.absurd(Held.Both<Placements.AxesIn(Nil{},Con{low,lows},extents,source_strides,cap),AxesReading(Nil{},Nil{},Con{low,lows},extents,source,shifts,source_strides,piece_strides,clamped,cap)>,Digits.zero_successor(List.length(&2,U32,lows),Equal.sym(Nat,1n+List.length(&2,U32,lows),0n,lows_length)))
    case Nil{} Nil{} Nil{} Con{+extent,+extents} +source +shifts +source_strides +piece_strides:
      Empty.absurd(Held.Both<Placements.AxesIn(Nil{},Nil{},Con{extent,extents},source_strides,cap),AxesReading(Nil{},Nil{},Nil{},Con{extent,extents},source,shifts,source_strides,piece_strides,clamped,cap)>,Digits.zero_successor(List.length(&2,U32,extents),Equal.sym(Nat,1n+List.length(&2,U32,extents),0n,extents_length)))
    case Nil{} Nil{} Nil{} Nil{} Con{+bound,+source} +shifts +source_strides +piece_strides:
      Empty.absurd(Held.Both<Placements.AxesIn(Nil{},Nil{},Nil{},source_strides,cap),AxesReading(Nil{},Nil{},Nil{},Nil{},Con{bound,source},shifts,source_strides,piece_strides,clamped,cap)>,axes)
    case Nil{} Nil{} Nil{} Nil{} Nil{} Con{+shift,+shifts} +source_strides +piece_strides:
      Empty.absurd(Held.Both<Placements.AxesIn(Nil{},Nil{},Nil{},source_strides,cap),AxesReading(Nil{},Nil{},Nil{},Nil{},Nil{},Con{shift,shifts},source_strides,piece_strides,clamped,cap)>,axes)
    case Nil{} Nil{} Nil{} Nil{} Nil{} Nil{} Con{+source_stride,+source_strides} +piece_strides:
      Empty.absurd(Held.Both<Placements.AxesIn(Nil{},Nil{},Nil{},Con{source_stride,source_strides},cap),AxesReading(Nil{},Nil{},Nil{},Nil{},Nil{},Nil{},Con{source_stride,source_strides},piece_strides,clamped,cap)>,Digits.zero_successor(List.length(&2,U32,source_strides),Equal.sym(Nat,1n+List.length(&2,U32,source_strides),0n,strides_length)))
    case Nil{} Nil{} Nil{} Nil{} Nil{} Nil{} Nil{} Con{+piece_stride,+piece_strides}:
      Empty.absurd(Held.Both<Placements.AxesIn(Nil{},Nil{},Nil{},Nil{},cap),AxesReading(Nil{},Nil{},Nil{},Nil{},Nil{},Nil{},Nil{},Con{piece_stride,piece_strides},clamped,cap)>,Digits.zero_successor(List.length(&2,U32,piece_strides),Equal.sym(Nat,1n+List.length(&2,U32,piece_strides),0n,piece_length)))
