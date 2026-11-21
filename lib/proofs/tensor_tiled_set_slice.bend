# The tiled form of set_slice: the part, arranged on the lane axis of the
# target, written into the cells of the target holds the values of the part
# where it lies and those of the target elsewhere.

import Base
import ../tensor_grid.bend as Grid
import ../tensor_layout.bend as Layout
import ../tensor_tiled.bend as Tiled
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_comparison.bend as Comparison
import ./boolean_logic.bend as Boolean
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_encoding.bend as WordEncoding
import ./tensor_tiled_public.bend as Public
import ./tensor_cell_room.bend as Rooms
import ./tensor_tiled_placing.bend as Placing
import ./nat_division.bend as Division
import ../tensor_tiled_layout.bend as TiledLayout
import ../machine_limits.bend as Machine
import ./nat_ceiling.bend as Ceiling
import ./encoded_coordinates.bend as Encoded
import ./tensor_placing_public.bend as Written
import ./u32_full_range.bend as FullRange
import ./binary_radix.bend as Radix
import ../kernel_tile.bend as Tile
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ./machine_limit_bounds.bend as Limits
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_tiled_representation.bend as Representation
import ./tensor_placing_facts.bend as Facts
import ./tensor_box_digits.bend as Digits
import ./tensor_tile_coordinates.bend as Coordinates
import ./tensor_target_cells.bend as Targets
import ./tensor_components.bend as Components
import ./tensor_move_readback.bend as Held
import ../tensor_tiled_cells.bend as Cells
import ./tensor_placements.bend as Placements
import ../tensor.bend as Tensors
import ./tensor_global_shape_count.bend as Global
import ./tensor_placing_gather.bend as Gathered
import ../tensor_tasks.bend as Tasks
import ../tensor_shape.bend as Shapes
import ../default_machine_limits.bend as Default
import ./tensor_storage_count.bend as StorageCount
import ./tensor_tiled_relane.bend as Relane
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_view_moved.bend as ViewMoved
import ./tensor_contract.bend as Contract
import ../tensor_pieces.bend as Pieces
import ../kernel_gemm.bend as KernelGemm
import ./tensor_tiled_lanes_written.bend as LanesWritten
import ./tensor_lanes_write_law.bend as Law
import ./tensor_lanes_write_cell.bend as LanesCell
import ./tensor_relane_values.bend as RelaneValues
import ./kernel_tile_refinement.bend as TileLaws

# ---- Shifts as marks ----

def zeros(count: Nat) -> List<&2,U32>:
  match count:
    case 0n: []
    case 1n+rest: 0 <> zeros(rest)

# count words: the word at the mark and 0 elsewhere.
def marked(count: Nat,mark: Nat,+word: U32) -> List<&2,U32>:
  match count mark:
    case 0n mark: []
    case 1n+rest 0n: word <> zeros(rest)
    case 1n+rest 1n+next: 0 <> marked(rest,next,word)

def pick_same(choice: Bool,+value: U32) -> {Bool.pick(U32,choice,value,value) == value : U32}:
  match choice:
    case True{}: {==}
    case False{}: {==}

def zeros_length(count: Nat) -> {List.length(&2,U32,zeros(count)) == count : Nat}:
  match count:
    case 0n: {==}
    case 1n+ +rest: Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,U32,zeros(rest)),rest,zeros_length(rest))

def marked_length(count: Nat,mark: Nat,+word: U32) -> {List.length(&2,U32,marked(count,mark,word)) == count : Nat}:
  match count mark:
    case 0n +mark: {==}
    case 1n+ +rest 0n: Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,U32,zeros(rest)),rest,zeros_length(rest))
    case 1n+ +rest 1n+ +next: Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,U32,marked(rest,next,word)),rest,marked_length(rest,next,word))

def shifts_length(count: Nat,+axis: U32,+start: U32,+index: U32) -> {List.length(&2,U32,Layout.shifts_at(count,axis,start,index)) == count : Nat}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,U32,Layout.shifts_at(rest,axis,start,(index + 1 : U32))),rest,shifts_length(rest,axis,start,(index + 1 : U32)))

# The shifts of a whole target are zeros.
def shifts_zero(count: Nat,+index: U32) -> {Layout.shifts_at(count,0,0,index) == zeros(count) : List<&2,U32>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(U32,Bool.pick(U32,U32.is_eq(index,0),0,0),0,pick_same(U32.is_eq(index,0),0)) :
        {_ <> Layout.shifts_at(rest,0,0,(index + 1 : U32)) == zeros(1n+rest) : List<&2,U32>}
      Equal.cong(List<&2,U32>,List<&2,U32>,later => 0 <> later,Layout.shifts_at(rest,0,0,(index + 1 : U32)),zeros(rest),shifts_zero(rest,(index + 1 : U32)))

# An index within the bound compares with a word as a number.
def index_same(+index: Nat,+axis: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(index,U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.is_eq(U32.from_nat(index),axis) == Nat.is_eq(index,U32.to_nat(axis)) : Bool}:
  %Bounds.roundtrip(index,cap,cap_ok,room) : {U32.is_eq(U32.from_nat(index),axis) == Nat.is_eq(_,U32.to_nat(axis)) : Bool}
  Comparison.eq(U32.from_nat(index),axis)

# An index other than the axis does not compare equal with it.
def index_apart(+index: Nat,+axis: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(index,U32.to_nat(cap)) == True{} : Bool},
  +apart: {Nat.is_eq(index,U32.to_nat(axis)) == False{} : Bool}) -> {U32.is_eq(U32.from_nat(index),axis) == False{} : Bool}:
  Equal.trans(Bool,U32.is_eq(U32.from_nat(index),axis),Nat.is_eq(index,U32.to_nat(axis)),False{},index_same(index,axis,cap,cap_ok,room),apart)

def after_apart(+index: Nat,+axis: Nat,+after: {Nat.is_lt(axis,index) == True{} : Bool}) -> {Nat.is_eq(index,axis) == False{} : Bool}:
  Equal.trans(Bool,Nat.is_eq(index,axis),Nat.is_eq(axis,index),False{},Order.equal_symmetric(index,axis),Order.less_not_equal(axis,index,after))

def after_next(+index: Nat,+axis: Nat,+after: {Nat.is_lt(axis,index) == True{} : Bool}) -> {Nat.is_lt(axis,1n+index) == True{} : Bool}:
  Order.lt_succ(axis,index,Bounds.strict_le(axis,index,after))

# The shifts past the axis are zeros.
def shifts_after(count: Nat,+axis: U32,+start: U32,+index: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(index,count),U32.to_nat(cap)) == True{} : Bool},+after: {Nat.is_lt(U32.to_nat(axis),index) == True{} : Bool}) ->
  {Layout.shifts_at(count,axis,start,U32.from_nat(index)) == zeros(count) : List<&2,U32>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +index_room = Bounds.strict_le(index,U32.to_nat(cap),Order.room_index(index,rest,U32.to_nat(cap),room))
      %Equal.sym(Bool,U32.is_eq(U32.from_nat(index),axis),False{},index_apart(index,axis,cap,cap_ok,index_room,after_apart(index,U32.to_nat(axis),after))) :
        {Bool.pick(U32,_,start,0) <> Layout.shifts_at(rest,axis,start,(U32.from_nat(index) + 1 : U32)) == zeros(1n+rest) : List<&2,U32>}
      %Equal.sym(U32,(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),Bounds.increment_encoded(index)) :
        {0 <> Layout.shifts_at(rest,axis,start,_) == zeros(1n+rest) : List<&2,U32>}
      Equal.cong(List<&2,U32>,List<&2,U32>,later => 0 <> later,Layout.shifts_at(rest,axis,start,U32.from_nat(1n+index)),zeros(rest),
        shifts_after(rest,axis,start,1n+index,cap,cap_ok,Order.advance_room(index,rest,U32.to_nat(cap),room),after_next(index,U32.to_nat(axis),after)))

# The shifts of a part at an axis the mark past the first index are the
# start at the mark.
def shifts_marked(count: Nat,+axis: U32,+start: U32,+index: Nat,mark: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(index,count),U32.to_nat(cap)) == True{} : Bool},+at: {U32.to_nat(axis) == Nat.add(index,mark) : Nat}) ->
  {Layout.shifts_at(count,axis,start,U32.from_nat(index)) == marked(count,mark,start) : List<&2,U32>}:
  match count mark:
    case 0n +mark: {==}
    case 1n+ +rest 0n:
      +index_room = Bounds.strict_le(index,U32.to_nat(cap),Order.room_index(index,rest,U32.to_nat(cap),room))
      +hit = Equal.trans(Nat,U32.to_nat(axis),Nat.add(index,0n),index,at,Addition.zero_right(index))
      +same = Equal.trans(Bool,U32.is_eq(U32.from_nat(index),axis),Nat.is_eq(index,U32.to_nat(axis)),True{},index_same(index,axis,cap,cap_ok,index_room),
        {%Equal.sym(Nat,U32.to_nat(axis),index,hit) : {Nat.is_eq(index,_) == True{} : Bool} Order.equal_reflexive(index) : {Nat.is_eq(index,U32.to_nat(axis)) == True{} : Bool}})
      +past = {%Equal.sym(Nat,U32.to_nat(axis),index,hit) : {Nat.is_lt(_,1n+index) == True{} : Bool} Order.lt_succ(index,index,Order.reflexive(index)) :
        {Nat.is_lt(U32.to_nat(axis),1n+index) == True{} : Bool}}
      %Equal.sym(Bool,U32.is_eq(U32.from_nat(index),axis),True{},same) :
        {Bool.pick(U32,_,start,0) <> Layout.shifts_at(rest,axis,start,(U32.from_nat(index) + 1 : U32)) == marked(1n+rest,0n,start) : List<&2,U32>}
      %Equal.sym(U32,(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),Bounds.increment_encoded(index)) :
        {start <> Layout.shifts_at(rest,axis,start,_) == marked(1n+rest,0n,start) : List<&2,U32>}
      Equal.cong(List<&2,U32>,List<&2,U32>,later => start <> later,Layout.shifts_at(rest,axis,start,U32.from_nat(1n+index)),zeros(rest),
        shifts_after(rest,axis,start,1n+index,cap,cap_ok,Order.advance_room(index,rest,U32.to_nat(cap),room),past))
    case 1n+ +rest 1n+ +next:
      +index_room = Bounds.strict_le(index,U32.to_nat(cap),Order.room_index(index,rest,U32.to_nat(cap),room))
      +above = {%Equal.sym(Nat,U32.to_nat(axis),Nat.add(index,1n+next),at) : {Nat.is_lt(index,_) == True{} : Bool} Order.below_sum(index,next) :
        {Nat.is_lt(index,U32.to_nat(axis)) == True{} : Bool}}
      %Equal.sym(Bool,U32.is_eq(U32.from_nat(index),axis),False{},index_apart(index,axis,cap,cap_ok,index_room,Order.less_not_equal(index,U32.to_nat(axis),above))) :
        {Bool.pick(U32,_,start,0) <> Layout.shifts_at(rest,axis,start,(U32.from_nat(index) + 1 : U32)) == marked(1n+rest,1n+next,start) : List<&2,U32>}
      %Equal.sym(U32,(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),Bounds.increment_encoded(index)) :
        {0 <> Layout.shifts_at(rest,axis,start,_) == marked(1n+rest,1n+next,start) : List<&2,U32>}
      Equal.cong(List<&2,U32>,List<&2,U32>,later => 0 <> later,Layout.shifts_at(rest,axis,start,U32.from_nat(1n+index)),marked(rest,next,start),
        shifts_marked(rest,axis,start,1n+index,next,cap,cap_ok,Order.advance_room(index,rest,U32.to_nat(cap),room),
          Equal.trans(Nat,U32.to_nat(axis),Nat.add(index,1n+next),1n+Nat.add(index,next),at,Addition.successor_right(index,next))))

# ---- Marks at an axis ----

def zeros_lane(count: Nat,axis: Nat,+inside: {Nat.is_lt(axis,count) == True{} : Bool}) -> {Grid.lanes_at(zeros(count),axis) == 0 : U32}:
  match count axis:
    case 0n +axis: Empty.absurd({Grid.lanes_at(zeros(0n),axis) == 0 : U32},Order.below_zero(axis,inside))
    case 1n+ +rest 0n: {==}
    case 1n+ +rest 1n+ +next: zeros_lane(rest,next,inside)

def marked_lane(count: Nat,mark: Nat,+word: U32,+inside: {Nat.is_lt(mark,count) == True{} : Bool}) -> {Grid.lanes_at(marked(count,mark,word),mark) == word : U32}:
  match count mark:
    case 0n +mark: Empty.absurd({Grid.lanes_at(marked(0n,mark,word),mark) == word : U32},Order.below_zero(mark,inside))
    case 1n+ +rest 0n: {==}
    case 1n+ +rest 1n+ +next: marked_lane(rest,next,word,inside)

def marked_other(count: Nat,mark: Nat,axis: Nat,+word: U32,+inside: {Nat.is_lt(axis,count) == True{} : Bool},+apart: {Nat.is_eq(axis,mark) == False{} : Bool}) ->
  {Grid.lanes_at(marked(count,mark,word),axis) == 0 : U32}:
  match count mark axis:
    case 0n +mark +axis: Empty.absurd({Grid.lanes_at(marked(0n,mark,word),axis) == 0 : U32},Order.below_zero(axis,inside))
    case 1n+ +rest 0n 0n: Empty.absurd({Grid.lanes_at(marked(1n+rest,0n,word),0n) == 0 : U32},WordEncoding.false_true(Equal.sym(Bool,True{},False{},apart)))
    case 1n+ +rest 0n 1n+ +next: zeros_lane(rest,next,inside)
    case 1n+ +rest 1n+ +later 0n: {==}
    case 1n+ +rest 1n+ +later 1n+ +next: marked_other(rest,later,next,word,inside,apart)

def zeros_removed(count: Nat,axis: Nat,+inside: {Nat.is_lt(axis,1n+count) == True{} : Bool}) -> {Grid.removed(zeros(1n+count),axis) == zeros(count) : List<&2,U32>}:
  match count axis:
    case +count 0n: {==}
    case 0n 1n+ +next: Empty.absurd({Grid.removed(zeros(1n),1n+next) == zeros(0n) : List<&2,U32>},Order.below_zero(next,inside))
    case 1n+ +rest 1n+ +next:
      Equal.cong(List<&2,U32>,List<&2,U32>,later => 0 <> later,Grid.removed(zeros(1n+rest),next),zeros(rest),zeros_removed(rest,next,inside))

def marked_removed(count: Nat,mark: Nat,+word: U32,+inside: {Nat.is_lt(mark,1n+count) == True{} : Bool}) -> {Grid.removed(marked(1n+count,mark,word),mark) == zeros(count) : List<&2,U32>}:
  match count mark:
    case +count 0n: {==}
    case 0n 1n+ +next: Empty.absurd({Grid.removed(marked(1n,1n+next,word),1n+next) == zeros(0n) : List<&2,U32>},Order.below_zero(next,inside))
    case 1n+ +rest 1n+ +next:
      Equal.cong(List<&2,U32>,List<&2,U32>,later => 0 <> later,Grid.removed(marked(1n+rest,next,word),next),zeros(rest),marked_removed(rest,next,word,inside))

# The mark of a list with a zero word put first, after the axis the list
# keeps without the mark.
def kept_step(choice: Bool,mark: Nat,+rest: Nat,+word: U32,+next: Nat,+classified: {Nat.is_lt(mark,next) == choice : Bool},+apart: {Nat.is_eq(next,mark) == False{} : Bool}) ->
  {0 <> marked(rest,Bool.pick(Nat,choice,mark,Nat.sub(mark,1n)),word) == marked(1n+rest,Bool.pick(Nat,choice,1n+mark,mark),word) : List<&2,U32>}:
  match choice mark:
    case True{} +mark: {==}
    case False{} 1n+ +later:
      %Equal.sym(Nat,Nat.sub(later,0n),later,Order.subtract_zero(later)) : {0 <> marked(rest,_,word) == marked(1n+rest,1n+later,word) : List<&2,U32>}
      {==}
    case False{} 0n:
      +zero = Order.zero_below(next,Order.not_less(0n,next,classified))
      Empty.absurd({0 <> marked(rest,0n,word) == marked(1n+rest,0n,word) : List<&2,U32>},
        WordEncoding.false_true(Equal.sym(Bool,True{},False{},{%zero : {Nat.is_eq(_,0n) == False{} : Bool} apart : {True{} == False{} : Bool}})))

# A marked list without another axis: the mark before it stays, after it
# moves down.
def marked_kept(count: Nat,mark: Nat,axis: Nat,+word: U32,+inside: {Nat.is_lt(axis,1n+count) == True{} : Bool},+apart: {Nat.is_eq(axis,mark) == False{} : Bool}) ->
  {Grid.removed(marked(1n+count,mark,word),axis) == marked(count,Bool.pick(Nat,Nat.is_lt(mark,axis),mark,Nat.sub(mark,1n)),word) : List<&2,U32>}:
  match count mark axis:
    case +count 0n 0n:
      Empty.absurd({Grid.removed(marked(1n+count,0n,word),0n) == marked(count,Bool.pick(Nat,Nat.is_lt(0n,0n),0n,Nat.sub(0n,1n)),word) : List<&2,U32>},
        WordEncoding.false_true(Equal.sym(Bool,True{},False{},apart)))
    case +count 1n+ +later 0n:
      %Equal.sym(Nat,Nat.sub(later,0n),later,Order.subtract_zero(later)) : {Grid.removed(marked(1n+count,1n+later,word),0n) == marked(count,_,word) : List<&2,U32>}
      {==}
    case 0n 0n 1n+ +next:
      Empty.absurd({Grid.removed(marked(1n,0n,word),1n+next) == marked(0n,Bool.pick(Nat,Nat.is_lt(0n,1n+next),0n,Nat.sub(0n,1n)),word) : List<&2,U32>},Order.below_zero(next,inside))
    case 1n+ +rest 0n 1n+ +next:
      Equal.cong(List<&2,U32>,List<&2,U32>,later => word <> later,Grid.removed(zeros(1n+rest),next),zeros(rest),zeros_removed(rest,next,inside))
    case 0n 1n+ +later 1n+ +next:
      Empty.absurd({Grid.removed(marked(1n,1n+later,word),1n+next) == marked(0n,Bool.pick(Nat,Nat.is_lt(1n+later,1n+next),1n+later,Nat.sub(1n+later,1n)),word) : List<&2,U32>},
        Order.below_zero(next,inside))
    case 1n+ +rest 1n+ +later 1n+ +next:
      %Equal.sym(Nat,Nat.sub(later,0n),later,Order.subtract_zero(later)) :
        {0 <> Grid.removed(marked(1n+rest,later,word),next) == marked(1n+rest,Bool.pick(Nat,Nat.is_lt(later,next),1n+later,_),word) : List<&2,U32>}
      Equal.trans(List<&2,U32>,0 <> Grid.removed(marked(1n+rest,later,word),next),0 <> marked(rest,Bool.pick(Nat,Nat.is_lt(later,next),later,Nat.sub(later,1n)),word),
        marked(1n+rest,Bool.pick(Nat,Nat.is_lt(later,next),1n+later,later),word),
        Equal.cong(List<&2,U32>,List<&2,U32>,kept => 0 <> kept,Grid.removed(marked(1n+rest,later,word),next),marked(rest,Bool.pick(Nat,Nat.is_lt(later,next),later,Nat.sub(later,1n)),word),
          marked_kept(rest,later,next,word,inside,apart)),
        kept_step(Nat.is_lt(later,next),later,rest,word,next,{==},apart))

def zeros_grown(count: Nat) -> {List.append(&2,U32,zeros(count),[0]) == zeros(1n+count) : List<&2,U32>}:
  match count:
    case 0n: {==}
    case 1n+ +rest: Equal.cong(List<&2,U32>,List<&2,U32>,later => 0 <> later,List.append(&2,U32,zeros(rest),[0]),zeros(1n+rest),zeros_grown(rest))

def zeros_appended(count: Nat,+word: U32) -> {List.append(&2,U32,zeros(count),[word]) == marked(1n+count,count,word) : List<&2,U32>}:
  match count:
    case 0n: {==}
    case 1n+ +rest: Equal.cong(List<&2,U32>,List<&2,U32>,later => 0 <> later,List.append(&2,U32,zeros(rest),[word]),marked(1n+rest,rest,word),zeros_appended(rest,word))

def marked_appended(count: Nat,mark: Nat,+word: U32,+inside: {Nat.is_lt(mark,count) == True{} : Bool}) ->
  {List.append(&2,U32,marked(count,mark,word),[0]) == marked(1n+count,mark,word) : List<&2,U32>}:
  match count mark:
    case 0n +mark: Empty.absurd({List.append(&2,U32,marked(0n,mark,word),[0]) == marked(1n,mark,word) : List<&2,U32>},Order.below_zero(mark,inside))
    case 1n+ +rest 0n: Equal.cong(List<&2,U32>,List<&2,U32>,later => word <> later,List.append(&2,U32,zeros(rest),[0]),zeros(1n+rest),zeros_grown(rest))
    case 1n+ +rest 1n+ +next:
      Equal.cong(List<&2,U32>,List<&2,U32>,later => 0 <> later,List.append(&2,U32,marked(rest,next,word),[0]),marked(1n+rest,next,word),marked_appended(rest,next,word,inside))

# ---- The shifts of the cells ----

def placed_zeros(position: Grid.Position,+rest: Nat) -> {Grid.placed(True{},position,zeros(rest),0) == zeros(1n+rest) : List<&2,U32>}:
  match position:
    case Grid.Innermost{}: zeros_grown(rest)
    case Grid.Outermost{}: {==}

def zeros_cases(within: Bool,count: Nat,+axis: Nat,+position: Grid.Position,+classified: {Nat.is_lt(axis,count) == within : Bool}) ->
  {Grid.placed(within,position,Grid.removed(zeros(count),axis),(Grid.lanes_at(zeros(count),axis) / 8 : U32)) == zeros(count) : List<&2,U32>}:
  match within count:
    case False{} +count:
      Public.removed_past(zeros(count),axis,
        {%Equal.sym(Nat,List.length(&2,U32,zeros(count)),count,zeros_length(count)) : {Nat.is_lt(axis,_) == False{} : Bool} classified :
          {Nat.is_lt(axis,List.length(&2,U32,zeros(count))) == False{} : Bool}})
    case True{} 0n:
      Empty.absurd({Grid.placed(True{},position,Grid.removed(zeros(0n),axis),(Grid.lanes_at(zeros(0n),axis) / 8 : U32)) == zeros(0n) : List<&2,U32>},Order.below_zero(axis,classified))
    case True{} 1n+ +rest:
      %Equal.sym(U32,Grid.lanes_at(zeros(1n+rest),axis),0,zeros_lane(1n+rest,axis,classified)) :
        {Grid.placed(True{},position,Grid.removed(zeros(1n+rest),axis),(_ / 8 : U32)) == zeros(1n+rest) : List<&2,U32>}
      %Equal.sym(List<&2,U32>,Grid.removed(zeros(1n+rest),axis),zeros(rest),zeros_removed(rest,axis,classified)) :
        {Grid.placed(True{},position,_,(0 / 8 : U32)) == zeros(1n+rest) : List<&2,U32>}
      placed_zeros(position,rest)

# The cells of zero shifts are zero shifts.
def cell_zeros(+count: Nat,+axis: Nat,+position: Grid.Position) -> {Placing.cell_shifts(zeros(count),axis,position) == zeros(count) : List<&2,U32>}:
  zeros_cases(Nat.is_lt(axis,List.length(&2,U32,zeros(count))),count,axis,position,
    Equal.cong(Nat,Bool,length => Nat.is_lt(axis,length),count,List.length(&2,U32,zeros(count)),Equal.sym(Nat,List.length(&2,U32,zeros(count)),count,zeros_length(count))))

# The mark without another axis lies below the rest.
def skipped_below(choice: Bool,mark: Nat,+axis: Nat,+rest: Nat,+classified: {Nat.is_lt(mark,axis) == choice : Bool},+inside: {Nat.is_lt(mark,1n+rest) == True{} : Bool},
  +within: {Nat.is_lt(axis,1n+rest) == True{} : Bool},+apart: {Nat.is_eq(axis,mark) == False{} : Bool}) ->
  {Nat.is_lt(Bool.pick(Nat,choice,mark,Nat.sub(mark,1n)),rest) == True{} : Bool}:
  match choice mark:
    case True{} +mark: Bounds.nat_trans(mark,axis,rest,classified,Division.lt_succ_le(axis,rest,within))
    case False{} 1n+ +later:
      %Equal.sym(Nat,Nat.sub(later,0n),later,Order.subtract_zero(later)) : {Nat.is_lt(_,rest) == True{} : Bool}
      inside
    case False{} 0n:
      +zero = Order.zero_below(axis,Order.not_less(0n,axis,classified))
      Empty.absurd({Nat.is_lt(Bool.pick(Nat,False{},0n,Nat.sub(0n,1n)),rest) == True{} : Bool},
        WordEncoding.false_true(Equal.sym(Bool,True{},False{},{%zero : {Nat.is_eq(_,0n) == False{} : Bool} apart : {True{} == False{} : Bool}})))

# The cells of shifts marked at the lane axis: the group of the mark first or
# last, nothing else.
def hit_cells(position: Grid.Position,+rest: Nat,+mark: Nat,+start: U32,+inside: {Nat.is_lt(mark,1n+rest) == True{} : Bool}) ->
  {Grid.placed(True{},position,Grid.removed(marked(1n+rest,mark,start),mark),(Grid.lanes_at(marked(1n+rest,mark,start),mark) / 8 : U32)) ==
    marked(1n+rest,Tiled.cell_axis_at(position,mark,mark,1n+rest),Bool.pick(U32,Nat.is_eq(mark,mark),(start / 8 : U32),start)) : List<&2,U32>}:
  match position:
    case Grid.Innermost{}:
      +same = Equal.sym(Bool,Nat.is_eq(mark,mark),True{},Order.equal_reflexive(mark))
      +lane = Equal.sym(U32,Grid.lanes_at(marked(1n+rest,mark,start),mark),start,marked_lane(1n+rest,mark,start,inside))
      +removed = Equal.sym(List<&2,U32>,Grid.removed(marked(1n+rest,mark,start),mark),zeros(rest),marked_removed(rest,mark,start,inside))
      %same : {List.append(&2,U32,Grid.removed(marked(1n+rest,mark,start),mark),[(Grid.lanes_at(marked(1n+rest,mark,start),mark) / 8 : U32)]) ==
        marked(1n+rest,Bool.pick(Nat,_,Nat.sub(1n+rest,1n),Bool.pick(Nat,Nat.is_lt(mark,mark),mark,Nat.sub(mark,1n))),Bool.pick(U32,_,(start / 8 : U32),start)) : List<&2,U32>}
      %lane : {List.append(&2,U32,Grid.removed(marked(1n+rest,mark,start),mark),[(_ / 8 : U32)]) == marked(1n+rest,Nat.sub(1n+rest,1n),(start / 8 : U32)) : List<&2,U32>}
      %removed : {List.append(&2,U32,_,[(start / 8 : U32)]) == marked(1n+rest,Nat.sub(1n+rest,1n),(start / 8 : U32)) : List<&2,U32>}
      %Equal.sym(Nat,Nat.sub(rest,0n),rest,Order.subtract_zero(rest)) : {List.append(&2,U32,zeros(rest),[(start / 8 : U32)]) == marked(1n+rest,_,(start / 8 : U32)) : List<&2,U32>}
      zeros_appended(rest,(start / 8 : U32))
    case Grid.Outermost{}:
      +same = Equal.sym(Bool,Nat.is_eq(mark,mark),True{},Order.equal_reflexive(mark))
      +lane = Equal.sym(U32,Grid.lanes_at(marked(1n+rest,mark,start),mark),start,marked_lane(1n+rest,mark,start,inside))
      +removed = Equal.sym(List<&2,U32>,Grid.removed(marked(1n+rest,mark,start),mark),zeros(rest),marked_removed(rest,mark,start,inside))
      %same : {(Grid.lanes_at(marked(1n+rest,mark,start),mark) / 8 : U32) <> Grid.removed(marked(1n+rest,mark,start),mark) ==
        marked(1n+rest,Bool.pick(Nat,_,0n,Bool.pick(Nat,Nat.is_lt(mark,mark),1n+mark,mark)),Bool.pick(U32,_,(start / 8 : U32),start)) : List<&2,U32>}
      %lane : {(_ / 8 : U32) <> Grid.removed(marked(1n+rest,mark,start),mark) == marked(1n+rest,0n,(start / 8 : U32)) : List<&2,U32>}
      %removed : {(start / 8 : U32) <> _ == marked(1n+rest,0n,(start / 8 : U32)) : List<&2,U32>}
      {==}

# The cells of shifts marked at another axis than the lane axis: the mark
# moved with the other axes, the group zero.
def kept_cells(position: Grid.Position,+rest: Nat,+mark: Nat,+start: U32,+axis: Nat,+inside: {Nat.is_lt(mark,1n+rest) == True{} : Bool},
  +within: {Nat.is_lt(axis,1n+rest) == True{} : Bool},+apart: {Nat.is_eq(axis,mark) == False{} : Bool},+differs: {Nat.is_eq(mark,axis) == False{} : Bool}) ->
  {Grid.placed(True{},position,Grid.removed(marked(1n+rest,mark,start),axis),(Grid.lanes_at(marked(1n+rest,mark,start),axis) / 8 : U32)) ==
    marked(1n+rest,Tiled.cell_axis_at(position,mark,axis,1n+rest),Bool.pick(U32,Nat.is_eq(mark,axis),(start / 8 : U32),start)) : List<&2,U32>}:
  match position:
    case Grid.Innermost{}:
      +kept = Bool.pick(Nat,Nat.is_lt(mark,axis),mark,Nat.sub(mark,1n))
      +lane = Equal.sym(U32,Grid.lanes_at(marked(1n+rest,mark,start),axis),0,marked_other(1n+rest,mark,axis,start,within,apart))
      +removed = Equal.sym(List<&2,U32>,Grid.removed(marked(1n+rest,mark,start),axis),marked(rest,kept,start),marked_kept(rest,mark,axis,start,within,apart))
      +other = Equal.sym(Bool,Nat.is_eq(mark,axis),False{},differs)
      %other : {List.append(&2,U32,Grid.removed(marked(1n+rest,mark,start),axis),[(Grid.lanes_at(marked(1n+rest,mark,start),axis) / 8 : U32)]) ==
        marked(1n+rest,Bool.pick(Nat,_,Nat.sub(1n+rest,1n),kept),Bool.pick(U32,_,(start / 8 : U32),start)) : List<&2,U32>}
      %lane : {List.append(&2,U32,Grid.removed(marked(1n+rest,mark,start),axis),[(_ / 8 : U32)]) == marked(1n+rest,kept,start) : List<&2,U32>}
      %removed : {List.append(&2,U32,_,[(0 / 8 : U32)]) == marked(1n+rest,kept,start) : List<&2,U32>}
      marked_appended(rest,kept,start,skipped_below(Nat.is_lt(mark,axis),mark,axis,rest,{==},inside,within,apart))
    case Grid.Outermost{}:
      +kept = Bool.pick(Nat,Nat.is_lt(mark,axis),mark,Nat.sub(mark,1n))
      +lane = Equal.sym(U32,Grid.lanes_at(marked(1n+rest,mark,start),axis),0,marked_other(1n+rest,mark,axis,start,within,apart))
      +removed = Equal.sym(List<&2,U32>,Grid.removed(marked(1n+rest,mark,start),axis),marked(rest,kept,start),marked_kept(rest,mark,axis,start,within,apart))
      +other = Equal.sym(Bool,Nat.is_eq(mark,axis),False{},differs)
      %other : {(Grid.lanes_at(marked(1n+rest,mark,start),axis) / 8 : U32) <> Grid.removed(marked(1n+rest,mark,start),axis) ==
        marked(1n+rest,Bool.pick(Nat,_,0n,Bool.pick(Nat,Nat.is_lt(mark,axis),1n+mark,mark)),Bool.pick(U32,_,(start / 8 : U32),start)) : List<&2,U32>}
      %lane : {(_ / 8 : U32) <> Grid.removed(marked(1n+rest,mark,start),axis) == marked(1n+rest,Bool.pick(Nat,Nat.is_lt(mark,axis),1n+mark,mark),start) : List<&2,U32>}
      %removed : {(0 / 8 : U32) <> _ == marked(1n+rest,Bool.pick(Nat,Nat.is_lt(mark,axis),1n+mark,mark),start) : List<&2,U32>}
      kept_step(Nat.is_lt(mark,axis),mark,rest,start,axis,{==},apart)

def lane_cells(same: Bool,+rest: Nat,+mark: Nat,+start: U32,+axis: Nat,+position: Grid.Position,+classified: {Nat.is_eq(mark,axis) == same : Bool},
  +inside: {Nat.is_lt(mark,1n+rest) == True{} : Bool},+within: {Nat.is_lt(axis,1n+rest) == True{} : Bool}) ->
  {Grid.placed(True{},position,Grid.removed(marked(1n+rest,mark,start),axis),(Grid.lanes_at(marked(1n+rest,mark,start),axis) / 8 : U32)) ==
    marked(1n+rest,Tiled.cell_axis_at(position,mark,axis,1n+rest),Bool.pick(U32,Nat.is_eq(mark,axis),(start / 8 : U32),start)) : List<&2,U32>}:
  match same:
    case True{}:
      %Order.equal_of_is_eq(mark,axis,classified) :
        {Grid.placed(True{},position,Grid.removed(marked(1n+rest,mark,start),_),(Grid.lanes_at(marked(1n+rest,mark,start),_) / 8 : U32)) ==
          marked(1n+rest,Tiled.cell_axis_at(position,mark,_,1n+rest),Bool.pick(U32,Nat.is_eq(mark,_),(start / 8 : U32),start)) : List<&2,U32>}
      hit_cells(position,rest,mark,start,inside)
    case False{}:
      kept_cells(position,rest,mark,start,axis,inside,within,Equal.trans(Bool,Nat.is_eq(axis,mark),Nat.is_eq(mark,axis),False{},Order.equal_symmetric(axis,mark),classified),classified)

def marked_cases(within: Bool,count: Nat,+mark: Nat,+start: U32,+axis: Nat,+position: Grid.Position,+classified: {Nat.is_lt(axis,count) == within : Bool},
  +inside: {Nat.is_lt(mark,count) == True{} : Bool}) ->
  {Grid.placed(within,position,Grid.removed(marked(count,mark,start),axis),(Grid.lanes_at(marked(count,mark,start),axis) / 8 : U32)) ==
    marked(count,Tiled.cell_axis(position,mark,axis,count),Bool.pick(U32,Nat.is_eq(mark,axis),(start / 8 : U32),start)) : List<&2,U32>}:
  match within count:
    case False{} +count:
      +before = Bounds.nat_trans(mark,count,axis,inside,Order.not_less(axis,count,classified))
      %Equal.sym(Bool,Nat.is_lt(axis,count),False{},classified) :
        {Grid.removed(marked(count,mark,start),axis) == marked(count,Bool.pick(Nat,_,Tiled.cell_axis_at(position,mark,axis,count),mark),Bool.pick(U32,Nat.is_eq(mark,axis),(start / 8 : U32),start)) : List<&2,U32>}
      %Equal.sym(Bool,Nat.is_eq(mark,axis),False{},Order.less_not_equal(mark,axis,before)) :
        {Grid.removed(marked(count,mark,start),axis) == marked(count,mark,Bool.pick(U32,_,(start / 8 : U32),start)) : List<&2,U32>}
      Public.removed_past(marked(count,mark,start),axis,
        {%Equal.sym(Nat,List.length(&2,U32,marked(count,mark,start)),count,marked_length(count,mark,start)) : {Nat.is_lt(axis,_) == False{} : Bool} classified :
          {Nat.is_lt(axis,List.length(&2,U32,marked(count,mark,start))) == False{} : Bool}})
    case True{} 0n:
      Empty.absurd({Grid.placed(True{},position,Grid.removed(marked(0n,mark,start),axis),(Grid.lanes_at(marked(0n,mark,start),axis) / 8 : U32)) ==
        marked(0n,Tiled.cell_axis(position,mark,axis,0n),Bool.pick(U32,Nat.is_eq(mark,axis),(start / 8 : U32),start)) : List<&2,U32>},Order.below_zero(axis,classified))
    case True{} 1n+ +rest:
      %Equal.sym(Bool,Nat.is_lt(axis,1n+rest),True{},classified) :
        {Grid.placed(True{},position,Grid.removed(marked(1n+rest,mark,start),axis),(Grid.lanes_at(marked(1n+rest,mark,start),axis) / 8 : U32)) ==
          marked(1n+rest,Bool.pick(Nat,_,Tiled.cell_axis_at(position,mark,axis,1n+rest),mark),Bool.pick(U32,Nat.is_eq(mark,axis),(start / 8 : U32),start)) : List<&2,U32>}
      lane_cells(Nat.is_eq(mark,axis),rest,mark,start,axis,position,{==},inside,classified)

# The cells of shifts marked inside the rank: the mark at the cell axis, the
# group of the start along the lane axis.
def cell_marked(+count: Nat,+mark: Nat,+start: U32,+axis: Nat,+position: Grid.Position,+inside: {Nat.is_lt(mark,count) == True{} : Bool}) ->
  {Placing.cell_shifts(marked(count,mark,start),axis,position) == marked(count,Tiled.cell_axis(position,mark,axis,count),Bool.pick(U32,Nat.is_eq(mark,axis),(start / 8 : U32),start)) : List<&2,U32>}:
  %Equal.sym(Nat,List.length(&2,U32,marked(count,mark,start)),count,marked_length(count,mark,start)) :
    {Grid.placed(Nat.is_lt(axis,_),position,Grid.removed(marked(count,mark,start),axis),(Grid.lanes_at(marked(count,mark,start),axis) / 8 : U32)) ==
      marked(count,Tiled.cell_axis(position,mark,axis,count),Bool.pick(U32,Nat.is_eq(mark,axis),(start / 8 : U32),start)) : List<&2,U32>}
  marked_cases(Nat.is_lt(axis,count),count,mark,start,axis,position,{==},inside)

# ---- Fits as marks ----

def same_extents(target: List<&2,U32>,part: List<&2,U32>) -> Bool:
  match target part:
    case Con{+extent,target} Con{+width,part}: U32.is_eq(extent,width) && same_extents(target,part)
    case Nil{} Nil{}: True{}
    case target part: False{}

# Extents the same along every axis are the same list.
def same_listed(target: List<&2,U32>,part: List<&2,U32>,+same: {same_extents(target,part) == True{} : Bool}) -> {part == target : List<&2,U32>}:
  match target part:
    case Con{+extent,+rest} Con{+width,+part_rest}:
      %Arithmetic.equal_of_is_eq(extent,width,Boolean.left(U32.is_eq(extent,width),same_extents(rest,part_rest),same)) : {_ <> part_rest == extent <> rest : List<&2,U32>}
      %same_listed(rest,part_rest,Boolean.right(U32.is_eq(extent,width),same_extents(rest,part_rest),same)) : {extent <> part_rest == extent <> _ : List<&2,U32>}
      {==}
    case Nil{} Nil{}: {==}
    case Con{+extent,+rest} Nil{}: Empty.absurd({Nil{} == extent <> rest : List<&2,U32>},WordEncoding.false_true(same))
    case Nil{} Con{+width,+part_rest}: Empty.absurd({width <> part_rest == Nil{} : List<&2,U32>},WordEncoding.false_true(same))

# A part with the extents of the target but at the mark, where it fits from
# start on.
def fits_marked(target: List<&2,U32>,part: List<&2,U32>,mark: Nat,+start: U32) -> Bool:
  match target part mark:
    case Con{+extent,target} Con{+width,part} 0n: (U32.is_le(start,extent) && U32.is_le(width,(extent - start : U32))) && same_extents(target,part)
    case Con{+extent,target} Con{+width,part} 1n+next: U32.is_eq(extent,width) && fits_marked(target,part,next,start)
    case target part mark: False{}

def index_room(+index: Nat,+cap: U32,+room: {Nat.is_le(Nat.add(index,0n),U32.to_nat(cap)) == True{} : Bool}) -> {Nat.is_le(index,U32.to_nat(cap)) == True{} : Bool}:
  %Addition.zero_right(index) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
  room

# Past the axis a part fits when it has the extents of the target.
def fits_after(target: List<&2,U32>,part: List<&2,U32>,+axis: U32,+start: U32,+index: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(index,List.length(&2,U32,target)),U32.to_nat(cap)) == True{} : Bool},+after: {Nat.is_lt(U32.to_nat(axis),index) == True{} : Bool}) ->
  {Layout.slice_fits(target,part,axis,start,U32.from_nat(index)) == same_extents(target,part) : Bool}:
  match target part:
    case Con{+extent,+rest} Con{+width,+part_rest}:
      +inside = Bounds.strict_le(index,U32.to_nat(cap),Order.room_index(index,List.length(&2,U32,rest),U32.to_nat(cap),room))
      %Equal.sym(Bool,U32.is_eq(U32.from_nat(index),axis),False{},index_apart(index,axis,cap,cap_ok,inside,after_apart(index,U32.to_nat(axis),after))) :
        {Bool.pick(Bool,_,U32.is_le(start,extent) && U32.is_le(width,(extent - start : U32)),U32.is_eq(extent,width)) && Layout.slice_fits(rest,part_rest,axis,start,(U32.from_nat(index) + 1 : U32)) ==
          U32.is_eq(extent,width) && same_extents(rest,part_rest) : Bool}
      %Equal.sym(U32,(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),Bounds.increment_encoded(index)) :
        {U32.is_eq(extent,width) && Layout.slice_fits(rest,part_rest,axis,start,_) == U32.is_eq(extent,width) && same_extents(rest,part_rest) : Bool}
      Equal.cong(Bool,Bool,later => U32.is_eq(extent,width) && later,Layout.slice_fits(rest,part_rest,axis,start,U32.from_nat(1n+index)),same_extents(rest,part_rest),
        fits_after(rest,part_rest,axis,start,1n+index,cap,cap_ok,Order.advance_room(index,List.length(&2,U32,rest),U32.to_nat(cap),room),after_next(index,U32.to_nat(axis),after)))
    case Nil{} Nil{}:
      Equal.trans(Bool,U32.is_lt(axis,U32.from_nat(index)),Nat.is_lt(U32.to_nat(axis),U32.to_nat(U32.from_nat(index))),True{},Comparison.lt(axis,U32.from_nat(index)),
        {%Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,Bounds.roundtrip(index,cap,cap_ok,index_room(index,cap,room))) : {Nat.is_lt(U32.to_nat(axis),_) == True{} : Bool} after :
          {Nat.is_lt(U32.to_nat(axis),U32.to_nat(U32.from_nat(index))) == True{} : Bool}})
    case Con{+extent,+rest} Nil{}: {==}
    case Nil{} Con{+width,+part_rest}: {==}

# The check of a part at an axis the mark past the first index is the fit at
# the mark.
def fits_mark_eq(target: List<&2,U32>,part: List<&2,U32>,+axis: U32,+start: U32,+index: Nat,mark: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(index,List.length(&2,U32,target)),U32.to_nat(cap)) == True{} : Bool},+at: {U32.to_nat(axis) == Nat.add(index,mark) : Nat}) ->
  {Layout.slice_fits(target,part,axis,start,U32.from_nat(index)) == fits_marked(target,part,mark,start) : Bool}:
  match target part mark:
    case Con{+extent,+rest} Con{+width,+part_rest} 0n:
      +inside = Bounds.strict_le(index,U32.to_nat(cap),Order.room_index(index,List.length(&2,U32,rest),U32.to_nat(cap),room))
      +hit = Equal.trans(Nat,U32.to_nat(axis),Nat.add(index,0n),index,at,Addition.zero_right(index))
      +same = Equal.trans(Bool,U32.is_eq(U32.from_nat(index),axis),Nat.is_eq(index,U32.to_nat(axis)),True{},index_same(index,axis,cap,cap_ok,inside),
        {%Equal.sym(Nat,U32.to_nat(axis),index,hit) : {Nat.is_eq(index,_) == True{} : Bool} Order.equal_reflexive(index) : {Nat.is_eq(index,U32.to_nat(axis)) == True{} : Bool}})
      +past = {%Equal.sym(Nat,U32.to_nat(axis),index,hit) : {Nat.is_lt(_,1n+index) == True{} : Bool} Order.lt_succ(index,index,Order.reflexive(index)) :
        {Nat.is_lt(U32.to_nat(axis),1n+index) == True{} : Bool}}
      +own = U32.is_le(start,extent) && U32.is_le(width,(extent - start : U32))
      %Equal.sym(Bool,U32.is_eq(U32.from_nat(index),axis),True{},same) :
        {Bool.pick(Bool,_,own,U32.is_eq(extent,width)) && Layout.slice_fits(rest,part_rest,axis,start,(U32.from_nat(index) + 1 : U32)) == own && same_extents(rest,part_rest) : Bool}
      %Equal.sym(U32,(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),Bounds.increment_encoded(index)) :
        {own && Layout.slice_fits(rest,part_rest,axis,start,_) == own && same_extents(rest,part_rest) : Bool}
      Equal.cong(Bool,Bool,later => own && later,Layout.slice_fits(rest,part_rest,axis,start,U32.from_nat(1n+index)),same_extents(rest,part_rest),
        fits_after(rest,part_rest,axis,start,1n+index,cap,cap_ok,Order.advance_room(index,List.length(&2,U32,rest),U32.to_nat(cap),room),past))
    case Con{+extent,+rest} Con{+width,+part_rest} 1n+ +next:
      +inside = Bounds.strict_le(index,U32.to_nat(cap),Order.room_index(index,List.length(&2,U32,rest),U32.to_nat(cap),room))
      +above = {%Equal.sym(Nat,U32.to_nat(axis),Nat.add(index,1n+next),at) : {Nat.is_lt(index,_) == True{} : Bool} Order.below_sum(index,next) :
        {Nat.is_lt(index,U32.to_nat(axis)) == True{} : Bool}}
      %Equal.sym(Bool,U32.is_eq(U32.from_nat(index),axis),False{},index_apart(index,axis,cap,cap_ok,inside,Order.less_not_equal(index,U32.to_nat(axis),above))) :
        {Bool.pick(Bool,_,U32.is_le(start,extent) && U32.is_le(width,(extent - start : U32)),U32.is_eq(extent,width)) && Layout.slice_fits(rest,part_rest,axis,start,(U32.from_nat(index) + 1 : U32)) ==
          U32.is_eq(extent,width) && fits_marked(rest,part_rest,next,start) : Bool}
      %Equal.sym(U32,(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),Bounds.increment_encoded(index)) :
        {U32.is_eq(extent,width) && Layout.slice_fits(rest,part_rest,axis,start,_) == U32.is_eq(extent,width) && fits_marked(rest,part_rest,next,start) : Bool}
      Equal.cong(Bool,Bool,later => U32.is_eq(extent,width) && later,Layout.slice_fits(rest,part_rest,axis,start,U32.from_nat(1n+index)),fits_marked(rest,part_rest,next,start),
        fits_mark_eq(rest,part_rest,axis,start,1n+index,next,cap,cap_ok,Order.advance_room(index,List.length(&2,U32,rest),U32.to_nat(cap),room),
          Equal.trans(Nat,U32.to_nat(axis),Nat.add(index,1n+next),1n+Nat.add(index,next),at,Addition.successor_right(index,next))))
    case Nil{} Nil{} +mark:
      +below = {%Equal.sym(Nat,U32.to_nat(axis),Nat.add(index,mark),at) : {Nat.is_le(index,_) == True{} : Bool} Order.le_add(index,mark) : {Nat.is_le(index,U32.to_nat(axis)) == True{} : Bool}}
      Equal.trans(Bool,U32.is_lt(axis,U32.from_nat(index)),Nat.is_lt(U32.to_nat(axis),U32.to_nat(U32.from_nat(index))),False{},Comparison.lt(axis,U32.from_nat(index)),
        {%Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,Bounds.roundtrip(index,cap,cap_ok,index_room(index,cap,room))) : {Nat.is_lt(U32.to_nat(axis),_) == False{} : Bool}
          Order.not_below(index,U32.to_nat(axis),below) : {Nat.is_lt(U32.to_nat(axis),U32.to_nat(U32.from_nat(index))) == False{} : Bool}})
    case Con{+extent,+rest} Nil{} +mark: {==}
    case Nil{} Con{+width,+part_rest} +mark: {==}

# ---- What a fit at a mark gives ----

def fits_inside(target: List<&2,U32>,part: List<&2,U32>,mark: Nat,+start: U32,+fits: {fits_marked(target,part,mark,start) == True{} : Bool}) ->
  {Nat.is_lt(mark,List.length(&2,U32,target)) == True{} : Bool}:
  match target part mark:
    case Con{+extent,+rest} Con{+width,+part_rest} 0n: {==}
    case Con{+extent,+rest} Con{+width,+part_rest} 1n+ +next:
      fits_inside(rest,part_rest,next,start,Boolean.right(U32.is_eq(extent,width),fits_marked(rest,part_rest,next,start),fits))
    case Nil{} Nil{} +mark: Empty.absurd({Nat.is_lt(mark,0n) == True{} : Bool},WordEncoding.false_true(fits))
    case Con{+extent,+rest} Nil{} +mark: Empty.absurd({Nat.is_lt(mark,List.length(&2,U32,Con{extent,rest})) == True{} : Bool},WordEncoding.false_true(fits))
    case Nil{} Con{+width,+part_rest} +mark: Empty.absurd({Nat.is_lt(mark,0n) == True{} : Bool},WordEncoding.false_true(fits))

def same_length(target: List<&2,U32>,part: List<&2,U32>,+same: {same_extents(target,part) == True{} : Bool}) -> {List.length(&2,U32,part) == List.length(&2,U32,target) : Nat}:
  match target part:
    case Con{+extent,+rest} Con{+width,+part_rest}:
      Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,U32,part_rest),List.length(&2,U32,rest),
        same_length(rest,part_rest,Boolean.right(U32.is_eq(extent,width),same_extents(rest,part_rest),same)))
    case Nil{} Nil{}: {==}
    case Con{+extent,+rest} Nil{}: Empty.absurd({0n == List.length(&2,U32,Con{extent,rest}) : Nat},WordEncoding.false_true(same))
    case Nil{} Con{+width,+part_rest}: Empty.absurd({List.length(&2,U32,Con{width,part_rest}) == 0n : Nat},WordEncoding.false_true(same))

def fits_length(target: List<&2,U32>,part: List<&2,U32>,mark: Nat,+start: U32,+fits: {fits_marked(target,part,mark,start) == True{} : Bool}) ->
  {List.length(&2,U32,part) == List.length(&2,U32,target) : Nat}:
  match target part mark:
    case Con{+extent,+rest} Con{+width,+part_rest} 0n:
      Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,U32,part_rest),List.length(&2,U32,rest),
        same_length(rest,part_rest,Boolean.right(U32.is_le(start,extent) && U32.is_le(width,(extent - start : U32)),same_extents(rest,part_rest),fits)))
    case Con{+extent,+rest} Con{+width,+part_rest} 1n+ +next:
      Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,U32,part_rest),List.length(&2,U32,rest),
        fits_length(rest,part_rest,next,start,Boolean.right(U32.is_eq(extent,width),fits_marked(rest,part_rest,next,start),fits)))
    case Nil{} Nil{} +mark: {==}
    case Con{+extent,+rest} Nil{} +mark: Empty.absurd({0n == List.length(&2,U32,Con{extent,rest}) : Nat},WordEncoding.false_true(fits))
    case Nil{} Con{+width,+part_rest} +mark: Empty.absurd({List.length(&2,U32,Con{width,part_rest}) == 0n : Nat},WordEncoding.false_true(fits))

def same_lane(target: List<&2,U32>,part: List<&2,U32>,axis: Nat,+same: {same_extents(target,part) == True{} : Bool}) -> {Grid.lanes_at(target,axis) == Grid.lanes_at(part,axis) : U32}:
  match target part axis:
    case Con{+extent,+rest} Con{+width,+part_rest} 0n:
      Arithmetic.equal_of_is_eq(extent,width,Boolean.left(U32.is_eq(extent,width),same_extents(rest,part_rest),same))
    case Con{+extent,+rest} Con{+width,+part_rest} 1n+ +next:
      same_lane(rest,part_rest,next,Boolean.right(U32.is_eq(extent,width),same_extents(rest,part_rest),same))
    case Nil{} Nil{} +axis: {==}
    case Con{+extent,+rest} Nil{} +axis: Empty.absurd({Grid.lanes_at(Con{extent,rest},axis) == Grid.lanes_at(Nil{},axis) : U32},WordEncoding.false_true(same))
    case Nil{} Con{+width,+part_rest} +axis: Empty.absurd({Grid.lanes_at(Nil{},axis) == Grid.lanes_at(Con{width,part_rest},axis) : U32},WordEncoding.false_true(same))

# At the mark the part fits from start.
def fits_hit(target: List<&2,U32>,part: List<&2,U32>,mark: Nat,+start: U32,+fits: {fits_marked(target,part,mark,start) == True{} : Bool}) ->
  {U32.is_le(start,Grid.lanes_at(target,mark)) && U32.is_le(Grid.lanes_at(part,mark),(Grid.lanes_at(target,mark) - start : U32)) == True{} : Bool}:
  match target part mark:
    case Con{+extent,+rest} Con{+width,+part_rest} 0n:
      Boolean.left(U32.is_le(start,extent) && U32.is_le(width,(extent - start : U32)),same_extents(rest,part_rest),fits)
    case Con{+extent,+rest} Con{+width,+part_rest} 1n+ +next:
      fits_hit(rest,part_rest,next,start,Boolean.right(U32.is_eq(extent,width),fits_marked(rest,part_rest,next,start),fits))
    case Nil{} Nil{} +mark: Empty.absurd({U32.is_le(start,Grid.lanes_at(Nil{},mark)) && U32.is_le(Grid.lanes_at(Nil{},mark),(Grid.lanes_at(Nil{},mark) - start : U32)) == True{} : Bool},WordEncoding.false_true(fits))
    case Con{+extent,+rest} Nil{} +mark:
      Empty.absurd({U32.is_le(start,Grid.lanes_at(Con{extent,rest},mark)) && U32.is_le(Grid.lanes_at(Nil{},mark),(Grid.lanes_at(Con{extent,rest},mark) - start : U32)) == True{} : Bool},WordEncoding.false_true(fits))
    case Nil{} Con{+width,+part_rest} +mark:
      Empty.absurd({U32.is_le(start,Grid.lanes_at(Nil{},mark)) && U32.is_le(Grid.lanes_at(Con{width,part_rest},mark),(Grid.lanes_at(Nil{},mark) - start : U32)) == True{} : Bool},WordEncoding.false_true(fits))

# Off the mark the part has the extent of the target.
def fits_other(target: List<&2,U32>,part: List<&2,U32>,mark: Nat,axis: Nat,+start: U32,+fits: {fits_marked(target,part,mark,start) == True{} : Bool},
  +apart: {Nat.is_eq(axis,mark) == False{} : Bool}) -> {Grid.lanes_at(target,axis) == Grid.lanes_at(part,axis) : U32}:
  match target part mark axis:
    case Con{+extent,+rest} Con{+width,+part_rest} 0n 0n:
      Empty.absurd({extent == width : U32},WordEncoding.false_true(Equal.sym(Bool,True{},False{},apart)))
    case Con{+extent,+rest} Con{+width,+part_rest} 0n 1n+ +next:
      same_lane(rest,part_rest,next,Boolean.right(U32.is_le(start,extent) && U32.is_le(width,(extent - start : U32)),same_extents(rest,part_rest),fits))
    case Con{+extent,+rest} Con{+width,+part_rest} 1n+ +later 0n:
      Arithmetic.equal_of_is_eq(extent,width,Boolean.left(U32.is_eq(extent,width),fits_marked(rest,part_rest,later,start),fits))
    case Con{+extent,+rest} Con{+width,+part_rest} 1n+ +later 1n+ +next:
      fits_other(rest,part_rest,later,next,start,Boolean.right(U32.is_eq(extent,width),fits_marked(rest,part_rest,later,start),fits),apart)
    case Nil{} Nil{} +mark +axis: Empty.absurd({Grid.lanes_at(Nil{},axis) == Grid.lanes_at(Nil{},axis) : U32},WordEncoding.false_true(fits))
    case Con{+extent,+rest} Nil{} +mark +axis: Empty.absurd({Grid.lanes_at(Con{extent,rest},axis) == Grid.lanes_at(Nil{},axis) : U32},WordEncoding.false_true(fits))
    case Nil{} Con{+width,+part_rest} +mark +axis: Empty.absurd({Grid.lanes_at(Nil{},axis) == Grid.lanes_at(Con{width,part_rest},axis) : U32},WordEncoding.false_true(fits))

def same_removed(target: List<&2,U32>,part: List<&2,U32>,axis: Nat,+same: {same_extents(target,part) == True{} : Bool}) ->
  {same_extents(Grid.removed(target,axis),Grid.removed(part,axis)) == True{} : Bool}:
  match target part axis:
    case Con{+extent,+rest} Con{+width,+part_rest} 0n: Boolean.right(U32.is_eq(extent,width),same_extents(rest,part_rest),same)
    case Con{+extent,+rest} Con{+width,+part_rest} 1n+ +next:
      Boolean.and_true(U32.is_eq(extent,width),same_extents(Grid.removed(rest,next),Grid.removed(part_rest,next)),
        Boolean.left(U32.is_eq(extent,width),same_extents(rest,part_rest),same),
        same_removed(rest,part_rest,next,Boolean.right(U32.is_eq(extent,width),same_extents(rest,part_rest),same)))
    case Nil{} Nil{} +axis: {==}
    case Con{+extent,+rest} Nil{} +axis: Empty.absurd({same_extents(Grid.removed(Con{extent,rest},axis),Grid.removed(Nil{},axis)) == True{} : Bool},WordEncoding.false_true(same))
    case Nil{} Con{+width,+part_rest} +axis: Empty.absurd({same_extents(Grid.removed(Nil{},axis),Grid.removed(Con{width,part_rest},axis)) == True{} : Bool},WordEncoding.false_true(same))

# Without the mark the other extents agree.
def fits_removed_hit(target: List<&2,U32>,part: List<&2,U32>,mark: Nat,+start: U32,+fits: {fits_marked(target,part,mark,start) == True{} : Bool}) ->
  {same_extents(Grid.removed(target,mark),Grid.removed(part,mark)) == True{} : Bool}:
  match target part mark:
    case Con{+extent,+rest} Con{+width,+part_rest} 0n:
      Boolean.right(U32.is_le(start,extent) && U32.is_le(width,(extent - start : U32)),same_extents(rest,part_rest),fits)
    case Con{+extent,+rest} Con{+width,+part_rest} 1n+ +next:
      Boolean.and_true(U32.is_eq(extent,width),same_extents(Grid.removed(rest,next),Grid.removed(part_rest,next)),
        Boolean.left(U32.is_eq(extent,width),fits_marked(rest,part_rest,next,start),fits),
        fits_removed_hit(rest,part_rest,next,start,Boolean.right(U32.is_eq(extent,width),fits_marked(rest,part_rest,next,start),fits)))
    case Nil{} Nil{} +mark: Empty.absurd({same_extents(Grid.removed(Nil{},mark),Grid.removed(Nil{},mark)) == True{} : Bool},WordEncoding.false_true(fits))
    case Con{+extent,+rest} Nil{} +mark: Empty.absurd({same_extents(Grid.removed(Con{extent,rest},mark),Grid.removed(Nil{},mark)) == True{} : Bool},WordEncoding.false_true(fits))
    case Nil{} Con{+width,+part_rest} +mark: Empty.absurd({same_extents(Grid.removed(Nil{},mark),Grid.removed(Con{width,part_rest},mark)) == True{} : Bool},WordEncoding.false_true(fits))

# A fit at a mark with an agreeing extent put first fits one further, after
# the axis the list keeps.
def other_step(choice: Bool,mark: Nat,+axis: Nat,+extent: U32,+width: U32,+target: List<&2,U32>,+part: List<&2,U32>,+start: U32,
  +classified: {Nat.is_lt(mark,axis) == choice : Bool},+apart: {Nat.is_eq(axis,mark) == False{} : Bool},+head: {U32.is_eq(extent,width) == True{} : Bool},
  +rest: {fits_marked(target,part,Bool.pick(Nat,choice,mark,Nat.sub(mark,1n)),start) == True{} : Bool}) ->
  {fits_marked(extent <> target,width <> part,Bool.pick(Nat,choice,1n+mark,mark),start) == True{} : Bool}:
  match choice mark:
    case True{} +mark: Boolean.and_true(U32.is_eq(extent,width),fits_marked(target,part,mark,start),head,rest)
    case False{} 1n+ +later:
      Boolean.and_true(U32.is_eq(extent,width),fits_marked(target,part,later,start),head,
        {%Order.subtract_zero(later) : {fits_marked(target,part,_,start) == True{} : Bool} rest : {fits_marked(target,part,later,start) == True{} : Bool}})
    case False{} 0n:
      +zero = Order.zero_below(axis,Order.not_less(0n,axis,classified))
      Empty.absurd({fits_marked(extent <> target,width <> part,0n,start) == True{} : Bool},
        WordEncoding.false_true(Equal.sym(Bool,True{},False{},{%zero : {Nat.is_eq(_,0n) == False{} : Bool} apart : {True{} == False{} : Bool}})))

# Without another axis the part fits at the mark moved with the other axes.
def fits_removed_other(target: List<&2,U32>,part: List<&2,U32>,mark: Nat,axis: Nat,+start: U32,+fits: {fits_marked(target,part,mark,start) == True{} : Bool},
  +apart: {Nat.is_eq(axis,mark) == False{} : Bool}) ->
  {fits_marked(Grid.removed(target,axis),Grid.removed(part,axis),Bool.pick(Nat,Nat.is_lt(mark,axis),mark,Nat.sub(mark,1n)),start) == True{} : Bool}:
  match target part mark axis:
    case Con{+extent,+rest} Con{+width,+part_rest} 0n 0n:
      Empty.absurd({fits_marked(rest,part_rest,Bool.pick(Nat,Nat.is_lt(0n,0n),0n,Nat.sub(0n,1n)),start) == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},apart)))
    case Con{+extent,+rest} Con{+width,+part_rest} 0n 1n+ +next:
      +own = U32.is_le(start,extent) && U32.is_le(width,(extent - start : U32))
      Boolean.and_true(own,same_extents(Grid.removed(rest,next),Grid.removed(part_rest,next)),Boolean.left(own,same_extents(rest,part_rest),fits),
        same_removed(rest,part_rest,next,Boolean.right(own,same_extents(rest,part_rest),fits)))
    case Con{+extent,+rest} Con{+width,+part_rest} 1n+ +later 0n:
      %Equal.sym(Nat,Nat.sub(later,0n),later,Order.subtract_zero(later)) : {fits_marked(rest,part_rest,_,start) == True{} : Bool}
      Boolean.right(U32.is_eq(extent,width),fits_marked(rest,part_rest,later,start),fits)
    case Con{+extent,+rest} Con{+width,+part_rest} 1n+ +later 1n+ +next:
      %Equal.sym(Nat,Nat.sub(later,0n),later,Order.subtract_zero(later)) :
        {fits_marked(extent <> Grid.removed(rest,next),width <> Grid.removed(part_rest,next),Bool.pick(Nat,Nat.is_lt(later,next),1n+later,_),start) == True{} : Bool}
      other_step(Nat.is_lt(later,next),later,next,extent,width,Grid.removed(rest,next),Grid.removed(part_rest,next),start,{==},apart,
        Boolean.left(U32.is_eq(extent,width),fits_marked(rest,part_rest,later,start),fits),
        fits_removed_other(rest,part_rest,later,next,start,Boolean.right(U32.is_eq(extent,width),fits_marked(rest,part_rest,later,start),fits),apart))
    case Nil{} Nil{} +mark +axis:
      Empty.absurd({fits_marked(Grid.removed(Nil{},axis),Grid.removed(Nil{},axis),Bool.pick(Nat,Nat.is_lt(mark,axis),mark,Nat.sub(mark,1n)),start) == True{} : Bool},WordEncoding.false_true(fits))
    case Con{+extent,+rest} Nil{} +mark +axis:
      Empty.absurd({fits_marked(Grid.removed(Con{extent,rest},axis),Grid.removed(Nil{},axis),Bool.pick(Nat,Nat.is_lt(mark,axis),mark,Nat.sub(mark,1n)),start) == True{} : Bool},WordEncoding.false_true(fits))
    case Nil{} Con{+width,+part_rest} +mark +axis:
      Empty.absurd({fits_marked(Grid.removed(Nil{},axis),Grid.removed(Con{width,part_rest},axis),Bool.pick(Nat,Nat.is_lt(mark,axis),mark,Nat.sub(mark,1n)),start) == True{} : Bool},WordEncoding.false_true(fits))

def same_appended(target: List<&2,U32>,part: List<&2,U32>,+word: U32,+same: {same_extents(target,part) == True{} : Bool}) ->
  {same_extents(List.append(&2,U32,target,[word]),List.append(&2,U32,part,[word])) == True{} : Bool}:
  match target part:
    case Con{+extent,+rest} Con{+width,+part_rest}:
      Boolean.and_true(U32.is_eq(extent,width),same_extents(List.append(&2,U32,rest,[word]),List.append(&2,U32,part_rest,[word])),
        Boolean.left(U32.is_eq(extent,width),same_extents(rest,part_rest),same),
        same_appended(rest,part_rest,word,Boolean.right(U32.is_eq(extent,width),same_extents(rest,part_rest),same)))
    case Nil{} Nil{}: Boolean.and_true(U32.is_eq(word,word),True{},Arithmetic.equal_reflexive(word),{==})
    case Con{+extent,+rest} Nil{}:
      Empty.absurd({same_extents(List.append(&2,U32,Con{extent,rest},[word]),List.append(&2,U32,Nil{},[word])) == True{} : Bool},WordEncoding.false_true(same))
    case Nil{} Con{+width,+part_rest}:
      Empty.absurd({same_extents(List.append(&2,U32,Nil{},[word]),List.append(&2,U32,Con{width,part_rest},[word])) == True{} : Bool},WordEncoding.false_true(same))

def fits_appended_other(target: List<&2,U32>,part: List<&2,U32>,mark: Nat,+word: U32,+start: U32,+fits: {fits_marked(target,part,mark,start) == True{} : Bool}) ->
  {fits_marked(List.append(&2,U32,target,[word]),List.append(&2,U32,part,[word]),mark,start) == True{} : Bool}:
  match target part mark:
    case Con{+extent,+rest} Con{+width,+part_rest} 0n:
      +own = U32.is_le(start,extent) && U32.is_le(width,(extent - start : U32))
      Boolean.and_true(own,same_extents(List.append(&2,U32,rest,[word]),List.append(&2,U32,part_rest,[word])),Boolean.left(own,same_extents(rest,part_rest),fits),
        same_appended(rest,part_rest,word,Boolean.right(own,same_extents(rest,part_rest),fits)))
    case Con{+extent,+rest} Con{+width,+part_rest} 1n+ +next:
      Boolean.and_true(U32.is_eq(extent,width),fits_marked(List.append(&2,U32,rest,[word]),List.append(&2,U32,part_rest,[word]),next,start),
        Boolean.left(U32.is_eq(extent,width),fits_marked(rest,part_rest,next,start),fits),
        fits_appended_other(rest,part_rest,next,word,start,Boolean.right(U32.is_eq(extent,width),fits_marked(rest,part_rest,next,start),fits)))
    case Nil{} Nil{} +mark:
      Empty.absurd({fits_marked(List.append(&2,U32,Nil{},[word]),List.append(&2,U32,Nil{},[word]),mark,start) == True{} : Bool},WordEncoding.false_true(fits))
    case Con{+extent,+rest} Nil{} +mark:
      Empty.absurd({fits_marked(List.append(&2,U32,Con{extent,rest},[word]),List.append(&2,U32,Nil{},[word]),mark,start) == True{} : Bool},WordEncoding.false_true(fits))
    case Nil{} Con{+width,+part_rest} +mark:
      Empty.absurd({fits_marked(List.append(&2,U32,Nil{},[word]),List.append(&2,U32,Con{width,part_rest},[word]),mark,start) == True{} : Bool},WordEncoding.false_true(fits))

def fits_appended_last(target: List<&2,U32>,part: List<&2,U32>,+total: U32,+width: U32,+start: U32,+same: {same_extents(target,part) == True{} : Bool},
  +group: {U32.is_le(start,total) && U32.is_le(width,(total - start : U32)) == True{} : Bool}) ->
  {fits_marked(List.append(&2,U32,target,[total]),List.append(&2,U32,part,[width]),List.length(&2,U32,target),start) == True{} : Bool}:
  match target part:
    case Con{+extent,+rest} Con{+other,+part_rest}:
      Boolean.and_true(U32.is_eq(extent,other),fits_marked(List.append(&2,U32,rest,[total]),List.append(&2,U32,part_rest,[width]),List.length(&2,U32,rest),start),
        Boolean.left(U32.is_eq(extent,other),same_extents(rest,part_rest),same),
        fits_appended_last(rest,part_rest,total,width,start,Boolean.right(U32.is_eq(extent,other),same_extents(rest,part_rest),same),group))
    case Nil{} Nil{}: Boolean.and_true(U32.is_le(start,total) && U32.is_le(width,(total - start : U32)),True{},group,{==})
    case Con{+extent,+rest} Nil{}:
      Empty.absurd({fits_marked(List.append(&2,U32,Con{extent,rest},[total]),List.append(&2,U32,Nil{},[width]),List.length(&2,U32,Con{extent,rest}),start) == True{} : Bool},WordEncoding.false_true(same))
    case Nil{} Con{+other,+part_rest}:
      Empty.absurd({fits_marked(List.append(&2,U32,Nil{},[total]),List.append(&2,U32,Con{other,part_rest},[width]),0n,start) == True{} : Bool},WordEncoding.false_true(same))

# ---- The groups of a fit ----

# Whether a write keeps the cells of the target: along another axis than the
# lane axis, or along it from a group start when the part fills its groups or
# ends the axis.
def kept_lanes(+lane: Nat,+axis: Nat,+start: U32,+width: U32,+extent: U32) -> Bool:
  Bool.not(Nat.is_eq(lane,axis)) || (U32.is_zero((start % 8 : U32)) && (U32.is_zero((width % 8 : U32)) || U32.is_eq((start + width : U32),extent)))

# A part along the lane axis from a group start fits the groups of the target.
def groups_fit(+limits: Machine.Limits,+total: U32,+width: U32,+start: U32,
  +total_room: {Nat.is_le(U32.to_nat(total),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +aligned: {U32.is_zero((start % 8 : U32)) == True{} : Bool},+fit: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(width)),U32.to_nat(total)) == True{} : Bool}) ->
  {U32.is_le((start / 8 : U32),Grid.groups(total)) && U32.is_le(Grid.groups(width),(Grid.groups(total) - (start / 8 : U32) : U32)) == True{} : Bool}:
  +cap = U32.to_nat(Machine.capacity(limits))
  +first = U32.to_nat((start / 8 : U32))
  +all = U32.to_nat(Grid.groups(total))
  +own = U32.to_nat(Grid.groups(width))
  +width_room = Order.transitive(U32.to_nat(width),U32.to_nat(total),cap,
    Order.transitive(U32.to_nat(width),Nat.add(U32.to_nat(start),U32.to_nat(width)),U32.to_nat(total),Order.right_le(U32.to_nat(start),U32.to_nat(width)),fit),total_room)
  +exact = Radix.divide_eight_exact_multiple(start,aligned)
  +shifted = {%Equal.sym(Nat,Nat.mul(first,8n),U32.to_nat(start),exact) : {Nat.is_le(Nat.add(_,U32.to_nat(width)),U32.to_nat(total)) == True{} : Bool} fit :
    {Nat.is_le(Nat.add(Nat.mul(first,8n),U32.to_nat(width)),U32.to_nat(total)) == True{} : Bool}}
  +groups = {%Equal.sym(Nat,all,Ceiling.ceiling(U32.to_nat(total),8n),Encoded.eight_blocks_exact(limits,total,total_room)) :
      {Nat.is_le(Nat.add(first,own),_) == True{} : Bool}
    %Equal.sym(Nat,own,Ceiling.ceiling(U32.to_nat(width),8n),Encoded.eight_blocks_exact(limits,width,width_room)) :
      {Nat.is_le(Nat.add(first,_),Ceiling.ceiling(U32.to_nat(total),8n)) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(first,Ceiling.ceiling(U32.to_nat(width),8n)),Ceiling.ceiling(Nat.add(Nat.mul(first,8n),U32.to_nat(width)),8n),Ceiling.ceiling_after(first,U32.to_nat(width))) : {Nat.is_le(_,Ceiling.ceiling(U32.to_nat(total),8n)) == True{} : Bool}
    Ceiling.monotone(Nat.add(Nat.mul(first,8n),U32.to_nat(width)),U32.to_nat(total),8n,{==},shifted) :
    {Nat.is_le(Nat.add(first,own),all) == True{} : Bool}}
  +ordered = Order.transitive(first,Nat.add(first,own),all,Order.le_add(first,own),groups)
  +difference = FullRange.subtract(Grid.groups(total),(start / 8 : U32),Grid.groups(total),ordered,Order.sub_le(all,first))
  Boolean.and_true(U32.is_le((start / 8 : U32),Grid.groups(total)),U32.is_le(Grid.groups(width),(Grid.groups(total) - (start / 8 : U32) : U32)),
    {%Equal.sym(Bool,U32.is_le((start / 8 : U32),Grid.groups(total)),Nat.is_le(first,all),Comparison.le((start / 8 : U32),Grid.groups(total))) : {_ == True{} : Bool} ordered :
      {U32.is_le((start / 8 : U32),Grid.groups(total)) == True{} : Bool}},
    {%Equal.sym(Bool,U32.is_le(Grid.groups(width),(Grid.groups(total) - (start / 8 : U32) : U32)),Nat.is_le(own,U32.to_nat((Grid.groups(total) - (start / 8 : U32) : U32))),
        Comparison.le(Grid.groups(width),(Grid.groups(total) - (start / 8 : U32) : U32))) : {_ == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat((Grid.groups(total) - (start / 8 : U32) : U32)),Nat.sub(all,first),difference) : {Nat.is_le(own,_) == True{} : Bool}
      %Order.le_offset(first,own,Nat.sub(all,first)) : {_ == True{} : Bool}
      %Equal.sym(Nat,Nat.add(first,Nat.sub(all,first)),all,Order.restore_shift(first,all,ordered)) : {Nat.is_le(Nat.add(first,own),_) == True{} : Bool}
      groups : {U32.is_le(Grid.groups(width),(Grid.groups(total) - (start / 8 : U32) : U32)) == True{} : Bool}})

# ---- The fit of the cells ----

# The cells of a part along the lane axis from a group start fit the cells of
# the target at the groups of the start.
def hit_fits(position: Grid.Position,+limits: Machine.Limits,+values: List<&2,U32>,+part_values: List<&2,U32>,+mark: Nat,+start: U32,
  +inside: {Nat.is_lt(mark,List.length(&2,U32,values)) == True{} : Bool},+fits: {fits_marked(values,part_values,mark,start) == True{} : Bool},
  +aligned: {U32.is_zero((start % 8 : U32)) == True{} : Bool},+lane_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,mark)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {fits_marked(Grid.placed(True{},position,Grid.removed(values,mark),Grid.groups(Grid.lanes_at(values,mark))),Grid.placed(True{},position,Grid.removed(part_values,mark),Grid.groups(Grid.lanes_at(part_values,mark))),
    Tiled.cell_axis_at(position,mark,mark,List.length(&2,U32,values)),Bool.pick(U32,Nat.is_eq(mark,mark),(start / 8 : U32),start)) == True{} : Bool}:
  match position:
    case Grid.Innermost{}:
      +total = Grid.lanes_at(values,mark)
      +width = Grid.lanes_at(part_values,mark)
      +others = Grid.removed(values,mark)
      +group = groups_fit(limits,total,width,start,lane_room,aligned,Written.slice_axis(True{},total,width,start,fits_hit(values,part_values,mark,start,fits)))
      %Equal.sym(Bool,Nat.is_eq(mark,mark),True{},Order.equal_reflexive(mark)) :
        {fits_marked(List.append(&2,U32,others,[Grid.groups(total)]),List.append(&2,U32,Grid.removed(part_values,mark),[Grid.groups(width)]),
          Bool.pick(Nat,_,Nat.sub(List.length(&2,U32,values),1n),Bool.pick(Nat,Nat.is_lt(mark,mark),mark,Nat.sub(mark,1n))),Bool.pick(U32,_,(start / 8 : U32),start)) == True{} : Bool}
      %Public.removed_length(values,mark,inside) :
        {fits_marked(List.append(&2,U32,others,[Grid.groups(total)]),List.append(&2,U32,Grid.removed(part_values,mark),[Grid.groups(width)]),Nat.sub(_,1n),(start / 8 : U32)) == True{} : Bool}
      %Equal.sym(Nat,Nat.sub(List.length(&2,U32,others),0n),List.length(&2,U32,others),Order.subtract_zero(List.length(&2,U32,others))) :
        {fits_marked(List.append(&2,U32,others,[Grid.groups(total)]),List.append(&2,U32,Grid.removed(part_values,mark),[Grid.groups(width)]),_,(start / 8 : U32)) == True{} : Bool}
      fits_appended_last(others,Grid.removed(part_values,mark),Grid.groups(total),Grid.groups(width),(start / 8 : U32),fits_removed_hit(values,part_values,mark,start,fits),group)
    case Grid.Outermost{}:
      +total = Grid.lanes_at(values,mark)
      +width = Grid.lanes_at(part_values,mark)
      +group = groups_fit(limits,total,width,start,lane_room,aligned,Written.slice_axis(True{},total,width,start,fits_hit(values,part_values,mark,start,fits)))
      %Equal.sym(Bool,Nat.is_eq(mark,mark),True{},Order.equal_reflexive(mark)) :
        {fits_marked(Grid.groups(total) <> Grid.removed(values,mark),Grid.groups(width) <> Grid.removed(part_values,mark),
          Bool.pick(Nat,_,0n,Bool.pick(Nat,Nat.is_lt(mark,mark),1n+mark,mark)),Bool.pick(U32,_,(start / 8 : U32),start)) == True{} : Bool}
      Boolean.and_true(U32.is_le((start / 8 : U32),Grid.groups(total)) && U32.is_le(Grid.groups(width),(Grid.groups(total) - (start / 8 : U32) : U32)),
        same_extents(Grid.removed(values,mark),Grid.removed(part_values,mark)),group,fits_removed_hit(values,part_values,mark,start,fits))

# The cells of a part off the lane axis fit the cells of the target at the
# mark moved with the other axes.
def kept_fits(position: Grid.Position,+values: List<&2,U32>,+part_values: List<&2,U32>,+mark: Nat,+start: U32,+axis: Nat,
  +fits: {fits_marked(values,part_values,mark,start) == True{} : Bool},+apart: {Nat.is_eq(axis,mark) == False{} : Bool},+differs: {Nat.is_eq(mark,axis) == False{} : Bool}) ->
  {fits_marked(Grid.placed(True{},position,Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis))),Grid.placed(True{},position,Grid.removed(part_values,axis),Grid.groups(Grid.lanes_at(part_values,axis))),
    Tiled.cell_axis_at(position,mark,axis,List.length(&2,U32,values)),Bool.pick(U32,Nat.is_eq(mark,axis),(start / 8 : U32),start)) == True{} : Bool}:
  match position:
    case Grid.Innermost{}:
      +kept = Bool.pick(Nat,Nat.is_lt(mark,axis),mark,Nat.sub(mark,1n))
      %fits_other(values,part_values,mark,axis,start,fits,apart) :
        {fits_marked(List.append(&2,U32,Grid.removed(values,axis),[Grid.groups(Grid.lanes_at(values,axis))]),List.append(&2,U32,Grid.removed(part_values,axis),[Grid.groups(_)]),
          Bool.pick(Nat,Nat.is_eq(mark,axis),Nat.sub(List.length(&2,U32,values),1n),kept),Bool.pick(U32,Nat.is_eq(mark,axis),(start / 8 : U32),start)) == True{} : Bool}
      %Equal.sym(Bool,Nat.is_eq(mark,axis),False{},differs) :
        {fits_marked(List.append(&2,U32,Grid.removed(values,axis),[Grid.groups(Grid.lanes_at(values,axis))]),List.append(&2,U32,Grid.removed(part_values,axis),[Grid.groups(Grid.lanes_at(values,axis))]),
          Bool.pick(Nat,_,Nat.sub(List.length(&2,U32,values),1n),kept),Bool.pick(U32,_,(start / 8 : U32),start)) == True{} : Bool}
      fits_appended_other(Grid.removed(values,axis),Grid.removed(part_values,axis),kept,Grid.groups(Grid.lanes_at(values,axis)),start,fits_removed_other(values,part_values,mark,axis,start,fits,apart))
    case Grid.Outermost{}:
      +group = Grid.groups(Grid.lanes_at(values,axis))
      %fits_other(values,part_values,mark,axis,start,fits,apart) :
        {fits_marked(group <> Grid.removed(values,axis),Grid.groups(_) <> Grid.removed(part_values,axis),
          Bool.pick(Nat,Nat.is_eq(mark,axis),0n,Bool.pick(Nat,Nat.is_lt(mark,axis),1n+mark,mark)),Bool.pick(U32,Nat.is_eq(mark,axis),(start / 8 : U32),start)) == True{} : Bool}
      %Equal.sym(Bool,Nat.is_eq(mark,axis),False{},differs) :
        {fits_marked(group <> Grid.removed(values,axis),group <> Grid.removed(part_values,axis),Bool.pick(Nat,_,0n,Bool.pick(Nat,Nat.is_lt(mark,axis),1n+mark,mark)),Bool.pick(U32,_,(start / 8 : U32),start)) == True{} : Bool}
      other_step(Nat.is_lt(mark,axis),mark,axis,group,group,Grid.removed(values,axis),Grid.removed(part_values,axis),start,{==},apart,Arithmetic.equal_reflexive(group),
        fits_removed_other(values,part_values,mark,axis,start,fits,apart))

def lane_fits(same: Bool,+limits: Machine.Limits,+values: List<&2,U32>,+part_values: List<&2,U32>,+mark: Nat,+start: U32,+axis: Nat,+position: Grid.Position,+width: U32,+extent: U32,
  +classified: {Nat.is_eq(mark,axis) == same : Bool},+fits: {fits_marked(values,part_values,mark,start) == True{} : Bool},
  +keeps: {kept_lanes(axis,mark,start,width,extent) == True{} : Bool},
  +lane_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,mark)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {fits_marked(Grid.placed(True{},position,Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis))),Grid.placed(True{},position,Grid.removed(part_values,axis),Grid.groups(Grid.lanes_at(part_values,axis))),
    Tiled.cell_axis_at(position,mark,axis,List.length(&2,U32,values)),Bool.pick(U32,Nat.is_eq(mark,axis),(start / 8 : U32),start)) == True{} : Bool}:
  match same:
    case True{}:
      +equal = Order.equal_of_is_eq(mark,axis,classified)
      +hit = Equal.trans(Bool,Nat.is_eq(axis,mark),Nat.is_eq(mark,axis),True{},Order.equal_symmetric(axis,mark),classified)
      +rest = U32.is_zero((width % 8 : U32)) || U32.is_eq((start + width : U32),extent)
      +aligned = Boolean.left(U32.is_zero((start % 8 : U32)),rest,{%hit : {Bool.not(_) || (U32.is_zero((start % 8 : U32)) && rest) == True{} : Bool} keeps :
        {U32.is_zero((start % 8 : U32)) && rest == True{} : Bool}})
      %equal : {fits_marked(Grid.placed(True{},position,Grid.removed(values,_),Grid.groups(Grid.lanes_at(values,_))),Grid.placed(True{},position,Grid.removed(part_values,_),Grid.groups(Grid.lanes_at(part_values,_))),
        Tiled.cell_axis_at(position,mark,_,List.length(&2,U32,values)),Bool.pick(U32,Nat.is_eq(mark,_),(start / 8 : U32),start)) == True{} : Bool}
      hit_fits(position,limits,values,part_values,mark,start,fits_inside(values,part_values,mark,start,fits),fits,aligned,lane_room)
    case False{}:
      kept_fits(position,values,part_values,mark,start,axis,fits,Equal.trans(Bool,Nat.is_eq(axis,mark),Nat.is_eq(mark,axis),False{},Order.equal_symmetric(axis,mark),classified),classified)

def fits_cases(within: Bool,+limits: Machine.Limits,+values: List<&2,U32>,+part_values: List<&2,U32>,+mark: Nat,+start: U32,+axis: Nat,+position: Grid.Position,+width: U32,+extent: U32,
  +classified: {Nat.is_lt(axis,List.length(&2,U32,values)) == within : Bool},+fits: {fits_marked(values,part_values,mark,start) == True{} : Bool},
  +keeps: {kept_lanes(axis,mark,start,width,extent) == True{} : Bool},
  +lane_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,mark)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {fits_marked(Grid.placed(within,position,Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis))),Grid.placed(within,position,Grid.removed(part_values,axis),Grid.groups(Grid.lanes_at(part_values,axis))),
    Bool.pick(Nat,within,Tiled.cell_axis_at(position,mark,axis,List.length(&2,U32,values)),mark),Bool.pick(U32,Nat.is_eq(mark,axis),(start / 8 : U32),start)) == True{} : Bool}:
  match within:
    case False{}:
      +before = Bounds.nat_trans(mark,List.length(&2,U32,values),axis,fits_inside(values,part_values,mark,start,fits),Order.not_less(axis,List.length(&2,U32,values),classified))
      +part_past = {%Equal.sym(Nat,List.length(&2,U32,part_values),List.length(&2,U32,values),fits_length(values,part_values,mark,start,fits)) :
        {Nat.is_lt(axis,_) == False{} : Bool} classified : {Nat.is_lt(axis,List.length(&2,U32,part_values)) == False{} : Bool}}
      %Equal.sym(Bool,Nat.is_eq(mark,axis),False{},Order.less_not_equal(mark,axis,before)) :
        {fits_marked(Grid.removed(values,axis),Grid.removed(part_values,axis),mark,Bool.pick(U32,_,(start / 8 : U32),start)) == True{} : Bool}
      %Equal.sym(List<&2,U32>,Grid.removed(values,axis),values,Public.removed_past(values,axis,classified)) : {fits_marked(_,Grid.removed(part_values,axis),mark,start) == True{} : Bool}
      %Equal.sym(List<&2,U32>,Grid.removed(part_values,axis),part_values,Public.removed_past(part_values,axis,part_past)) : {fits_marked(values,_,mark,start) == True{} : Bool}
      fits
    case True{}:
      lane_fits(Nat.is_eq(mark,axis),limits,values,part_values,mark,start,axis,position,width,extent,{==},fits,keeps,lane_room)

# The cells of a part that fits the target, its lanes kept, fit the cells of
# the target at the cell axis, from the group of the start along the lane
# axis.
def cell_fits(+limits: Machine.Limits,+values: List<&2,U32>,+part_values: List<&2,U32>,+mark: Nat,+start: U32,+axis: Nat,+position: Grid.Position,+width: U32,+extent: U32,
  +fits: {fits_marked(values,part_values,mark,start) == True{} : Bool},+keeps: {kept_lanes(axis,mark,start,width,extent) == True{} : Bool},
  +lane_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,mark)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {fits_marked(Tiled.cell_shape(values,axis,position),Tiled.cell_shape(part_values,axis,position),Tiled.cell_axis(position,mark,axis,List.length(&2,U32,values)),
    Bool.pick(U32,Nat.is_eq(mark,axis),(start / 8 : U32),start)) == True{} : Bool}:
  +within = Nat.is_lt(axis,List.length(&2,U32,values))
  %Equal.sym(Nat,List.length(&2,U32,part_values),List.length(&2,U32,values),fits_length(values,part_values,mark,start,fits)) :
    {fits_marked(Grid.placed(within,position,Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis))),Grid.placed(Nat.is_lt(axis,_),position,Grid.removed(part_values,axis),Grid.groups(Grid.lanes_at(part_values,axis))),
      Bool.pick(Nat,within,Tiled.cell_axis_at(position,mark,axis,List.length(&2,U32,values)),mark),Bool.pick(U32,Nat.is_eq(mark,axis),(start / 8 : U32),start)) == True{} : Bool}
  fits_cases(within,limits,values,part_values,mark,start,axis,position,width,extent,{==},fits,keeps,lane_room)

# ---- What a placed source needs ----

# A source placed in a target of cells within the capacity has its shifts and
# extents within half a word.
def placed_room(target: List<&2,U32>,source: List<&2,U32>,shifts: List<&2,U32>,+limits: Machine.Limits,+within: Facts.PlacedIn(target,source,shifts),
  +room: {Nat.is_le(ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool}) ->
  Placing.Room(source,shifts,Machine.half_word(limits)):
  match target source shifts:
    case Con{+total,+target} Con{+bound,+source} Con{+shift,+shifts}:
      Held.Both{+fit,+later} = within
      +cap = U32.to_nat(Machine.capacity(limits))
      +size = U32.to_nat(total)
      +inner = ShapeLaws.elements(target)
      +whole = Nat.mul(size,inner)
      +size_positive = RowMajor.left_factor_positive(size,inner,positive)
      +inner_positive = DivisionOrder.right_factor_positive(size,inner,positive)
      +size_room = Order.transitive(size,whole,cap,RowMajor.product_covers(size,inner,inner_positive),room)
      +sum = Nat.add(U32.to_nat(shift),U32.to_nat(bound))
      KernelGemm.Both{Limits.within_half(limits,sum,Order.transitive(sum,size,cap,fit,size_room)),
        placed_room(target,source,shifts,limits,later,Order.transitive(inner,whole,cap,DivisionOrder.le_multiple(size,inner,size_positive),room),inner_positive)}
    case Nil{} Nil{} Nil{}: Unit{}
    case Con{+total,+target} Nil{} +shifts: Empty.absurd(Placing.Room(Nil{},shifts,Machine.half_word(limits)),within)
    case Con{+total,+target} Con{+bound,+source} Nil{}: Empty.absurd(Placing.Room(Con{bound,source},Nil{},Machine.half_word(limits)),within)
    case Nil{} Con{+bound,+source} +shifts: Empty.absurd(Placing.Room(Con{bound,source},shifts,Machine.half_word(limits)),within)
    case Nil{} Nil{} Con{+shift,+shifts}: Empty.absurd(Placing.Room(Nil{},Con{shift,shifts},Machine.half_word(limits)),within)

# A placed source fits the target along every axis inside it.
def placed_lane(target: List<&2,U32>,source: List<&2,U32>,shifts: List<&2,U32>,axis: Nat,+within: Facts.PlacedIn(target,source,shifts),
  +inside: {Nat.is_lt(axis,List.length(&2,U32,target)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(Grid.lanes_at(shifts,axis)),U32.to_nat(Grid.lanes_at(source,axis))),U32.to_nat(Grid.lanes_at(target,axis))) == True{} : Bool}:
  match target source shifts axis:
    case Con{+total,+target} Con{+bound,+source} Con{+shift,+shifts} 0n:
      Held.Both{+fit,+later} = within
      fit
    case Con{+total,+target} Con{+bound,+source} Con{+shift,+shifts} 1n+ +next:
      Held.Both{+fit,+later} = within
      placed_lane(target,source,shifts,next,later,inside)
    case Nil{} +source +shifts +axis:
      Empty.absurd({Nat.is_le(Nat.add(U32.to_nat(Grid.lanes_at(shifts,axis)),U32.to_nat(Grid.lanes_at(source,axis))),U32.to_nat(Grid.lanes_at(Nil{},axis))) == True{} : Bool},Order.below_zero(axis,inside))
    case Con{+total,+target} Nil{} +shifts +axis:
      Empty.absurd({Nat.is_le(Nat.add(U32.to_nat(Grid.lanes_at(shifts,axis)),U32.to_nat(Grid.lanes_at(Nil{},axis))),U32.to_nat(Grid.lanes_at(Con{total,target},axis))) == True{} : Bool},within)
    case Con{+total,+target} Con{+bound,+source} Nil{} +axis:
      Empty.absurd({Nat.is_le(Nat.add(U32.to_nat(Grid.lanes_at(Nil{},axis)),U32.to_nat(Grid.lanes_at(Con{bound,source},axis))),U32.to_nat(Grid.lanes_at(Con{total,target},axis))) == True{} : Bool},within)

# The extent of a shape at an axis inside it is its lanes there.
def extent_lanes(shape: List<&2,U32>,axis: Nat,+inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool}) ->
  {Layout.extent_at(shape,axis) == Grid.lanes_at(shape,axis) : U32}:
  match shape axis:
    case Con{+extent,+rest} 0n: {==}
    case Con{+extent,+rest} 1n+ +next: extent_lanes(rest,next,inside)
    case Nil{} +axis: Empty.absurd({Layout.extent_at(Nil{},axis) == Grid.lanes_at(Nil{},axis) : U32},Order.below_zero(axis,inside))

# Zero shifts hold digits below a source.
def inside_zeros(source: List<&2,U32>,digits: List<&2,Nat>,+below: Coordinates.DigitsBelow(source,digits)) ->
  {Placing.inside(source,zeros(List.length(&2,U32,source)),digits) == True{} : Bool}:
  match source digits:
    case Con{+extent,+rest} Con{+digit,+more}:
      KernelGemm.Both{+small,+later} = below
      Boolean.and_true(Placing.axis_inside(extent,0,digit),Placing.inside(rest,zeros(List.length(&2,U32,rest)),more),
        Boolean.and_true(Nat.is_le(0n,digit),Nat.is_lt(digit,U32.to_nat(extent)),Order.zero_le(digit),small),inside_zeros(rest,more,later))
    case Nil{} Nil{}: {==}
    case Con{+extent,+rest} Nil{}: {==}
    case Nil{} Con{+digit,+more}: {==}

# A source of cells placed from zero holds every cell of the target.
def covered(+values: List<&2,U32>,+part_values: List<&2,U32>,+shifts: List<&2,U32>,+part_shifts: List<&2,U32>,+lane: Nat,+position: Grid.Position,+limits: Machine.Limits,+index: U32,
  +zero: {Placing.cell_shifts(shifts,lane,position) == zeros(List.length(&2,U32,Tiled.cell_shape(values,lane,position))) : List<&2,U32>},
  +zero_room: Placing.Room(Tiled.cell_shape(values,lane,position),zeros(List.length(&2,U32,Tiled.cell_shape(values,lane,position))),Machine.half_word(limits)),
  +cells_room: {Nat.is_le(Nat.double(ShapeLaws.elements(Tiled.cell_shape(values,lane,position))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(Tiled.cell_shape(values,lane,position))) == True{} : Bool}) ->
  {Placing.covers([values,part_values],[shifts,part_shifts],Grid.coordinates(Tiled.cell_shape(values,lane,position),index),lane,position) == True{} : Bool}:
  +cells = Tiled.cell_shape(values,lane,position)
  +flat = zeros(List.length(&2,U32,cells))
  +words = Grid.coordinates(cells,index)
  +later = Placing.covers([part_values],[part_shifts],words,lane,position)
  %Equal.sym(List<&2,U32>,Placing.cell_shifts(shifts,lane,position),flat,zero) : {Placing.inside_words(cells,_,words) || later == True{} : Bool}
  %Equal.sym(Bool,Placing.inside_words(cells,flat,words),Placing.inside(cells,flat,Digits.words_nat(words)),Placing.inside_nat(cells,flat,words,Machine.half_word(limits),Limits.half_word_fits(limits),zero_room)) :
    {_ || later == True{} : Bool}
  %Equal.sym(List<&2,Nat>,Digits.words_nat(words),Targets.cell_coordinates(cells,U32.to_nat(index)),Digits.coordinates_exact(cells,index,Machine.half_word(limits),Limits.half_word_fits(limits),cells_room,inside)) :
    {Placing.inside(cells,flat,_) || later == True{} : Bool}
  %Equal.sym(Bool,Placing.inside(cells,flat,Targets.cell_coordinates(cells,U32.to_nat(index))),True{},inside_zeros(cells,Targets.cell_coordinates(cells,U32.to_nat(index)),Digits.coordinates_below(cells,U32.to_nat(index),inside))) :
    {_ || later == True{} : Bool}
  {==}

# ---- A cell of the written cells ----

# The written cells read the lifted values of each source as the lifted
# values of the written values.
def relabeled_result(~Base: Data,~base_value: Base -> U32 -> F32,~Part: Data,~part_value: Part -> U32 -> F32,+base: Base,+part: Part,+geometry: Cells.Cells,+part_geometry: Cells.Cells,
  first: Maybe<&2,U32>,second: Maybe<&2,U32>,+fill: Tile.Tile8) ->
  {Placements.placed_result(~Tile.Tile8,~Written.Written<Representation.Lifted<Base>,Representation.Lifted<Part>>,
      ~Written.written_value(~Tile.Tile8,~Representation.Lifted<Base>,~Representation.lifted(~Base,~base_value),~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value)),
      Written.Over{Representation.Lifted{part,part_geometry}},second,
      Placements.placed_result(~Tile.Tile8,~Written.Written<Representation.Lifted<Base>,Representation.Lifted<Part>>,
        ~Written.written_value(~Tile.Tile8,~Representation.Lifted<Base>,~Representation.lifted(~Base,~base_value),~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value)),
        Written.Before{Representation.Lifted{base,geometry}},first,fill)) ==
    Placements.placed_result(~Tile.Tile8,~Representation.Lifted<Written.Written<Base,Part>>,~Representation.lifted(~Written.Written<Base,Part>,~Written.written_value(~F32,~Base,~base_value,~Part,~part_value)),
      Representation.Lifted{Written.Over{part},part_geometry},second,
      Placements.placed_result(~Tile.Tile8,~Representation.Lifted<Written.Written<Base,Part>>,~Representation.lifted(~Written.Written<Base,Part>,~Written.written_value(~F32,~Base,~base_value,~Part,~part_value)),
        Representation.Lifted{Written.Before{base},geometry},first,fill)) : Tile.Tile8}:
  match first second:
    case +first Some{+cell}: {==}
    case Some{+cell} None{}: {==}
    case None{} None{}: {==}

# ---- The sources along the lane axis ----

# Within the capacity is within half a word.
def half_room(+limits: Machine.Limits,+value: Nat,+room: {Nat.is_le(value,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(value,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  Order.transitive(value,Nat.double(value),U32.to_nat(Machine.half_word(limits)),Order.le_double(value),Limits.double_bound(limits,value,room))

# The target placed from zero starts every group and ends the lane axis.
def target_lane(+values: List<&2,U32>,+axis: Nat,+within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool}) ->
  KernelGemm.Both<{U32.is_zero((Grid.lanes_at(Layout.shifts_at(List.length(&2,U32,values),0,0,0),axis) % 8 : U32)) == True{} : Bool},
    Placing.Ending<Grid.lanes_at(values,axis),Grid.lanes_at(Layout.shifts_at(List.length(&2,U32,values),0,0,0),axis),Grid.lanes_at(values,axis)>>:
  +rank = List.length(&2,U32,values)
  %Equal.sym(List<&2,U32>,Layout.shifts_at(rank,0,0,0),zeros(rank),shifts_zero(rank,0)) :
    KernelGemm.Both<{U32.is_zero((Grid.lanes_at(_,axis) % 8 : U32)) == True{} : Bool},Placing.Ending<Grid.lanes_at(values,axis),Grid.lanes_at(_,axis),Grid.lanes_at(values,axis)>>
  %Equal.sym(U32,Grid.lanes_at(zeros(rank),axis),0,zeros_lane(rank,axis,within)) :
    KernelGemm.Both<{U32.is_zero((_ % 8 : U32)) == True{} : Bool},Placing.Ending<Grid.lanes_at(values,axis),_,Grid.lanes_at(values,axis)>>
  KernelGemm.Both{{==},Placing.Last{{==}}}

# A part that ends with the target along the lane axis or fills its groups.
def rest_ending(choice: Bool,+width: U32,+extent: U32,+start: U32,+classified: {U32.is_zero((width % 8 : U32)) == choice : Bool},
  +rest: {choice || U32.is_eq((start + width : U32),extent) == True{} : Bool},
  +fit: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(width)),U32.to_nat(extent)) == True{} : Bool}) -> Placing.Ending<width,start,extent>:
  match choice:
    case True{}: Placing.Whole{classified}
    case False{}:
      Placing.Last{Equal.trans(Nat,Nat.add(U32.to_nat(start),U32.to_nat(width)),U32.to_nat((start + width : U32)),U32.to_nat(extent),
        Equal.sym(Nat,U32.to_nat((start + width : U32)),Nat.add(U32.to_nat(start),U32.to_nat(width)),FullRange.add(start,width,extent,fit)),
        Equal.cong(U32,Nat,word => U32.to_nat(word),(start + width : U32),extent,Arithmetic.equal_of_is_eq((start + width : U32),extent,rest)))}

# A part along the lane axis starts a group and ends with the target or fills
# its groups.
def hit_lane(+values: List<&2,U32>,+part_values: List<&2,U32>,+mark: Nat,+start: U32,+fits: {fits_marked(values,part_values,mark,start) == True{} : Bool},
  +kept: {U32.is_zero((start % 8 : U32)) && (U32.is_zero((Grid.lanes_at(part_values,mark) % 8 : U32)) || U32.is_eq((start + Grid.lanes_at(part_values,mark) : U32),Grid.lanes_at(values,mark))) == True{} : Bool}) ->
  KernelGemm.Both<{U32.is_zero((Grid.lanes_at(marked(List.length(&2,U32,values),mark,start),mark) % 8 : U32)) == True{} : Bool},
    Placing.Ending<Grid.lanes_at(part_values,mark),Grid.lanes_at(marked(List.length(&2,U32,values),mark,start),mark),Grid.lanes_at(values,mark)>>:
  +width = Grid.lanes_at(part_values,mark)
  +extent = Grid.lanes_at(values,mark)
  +rest = U32.is_zero((width % 8 : U32)) || U32.is_eq((start + width : U32),extent)
  %Equal.sym(U32,Grid.lanes_at(marked(List.length(&2,U32,values),mark,start),mark),start,marked_lane(List.length(&2,U32,values),mark,start,fits_inside(values,part_values,mark,start,fits))) :
    KernelGemm.Both<{U32.is_zero((_ % 8 : U32)) == True{} : Bool},Placing.Ending<width,_,extent>>
  KernelGemm.Both{Boolean.left(U32.is_zero((start % 8 : U32)),rest,kept),
    rest_ending(U32.is_zero((width % 8 : U32)),width,extent,start,{==},Boolean.right(U32.is_zero((start % 8 : U32)),rest,kept),
      Written.slice_axis(True{},extent,width,start,fits_hit(values,part_values,mark,start,fits)))}

# A part off the lane axis is placed from zero along it, with the extent of
# the target.
def other_lane(+values: List<&2,U32>,+part_values: List<&2,U32>,+mark: Nat,+start: U32,+axis: Nat,+fits: {fits_marked(values,part_values,mark,start) == True{} : Bool},
  +within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},+apart: {Nat.is_eq(axis,mark) == False{} : Bool}) ->
  KernelGemm.Both<{U32.is_zero((Grid.lanes_at(marked(List.length(&2,U32,values),mark,start),axis) % 8 : U32)) == True{} : Bool},
    Placing.Ending<Grid.lanes_at(part_values,axis),Grid.lanes_at(marked(List.length(&2,U32,values),mark,start),axis),Grid.lanes_at(values,axis)>>:
  +extent = Grid.lanes_at(values,axis)
  %Equal.sym(U32,Grid.lanes_at(marked(List.length(&2,U32,values),mark,start),axis),0,marked_other(List.length(&2,U32,values),mark,axis,start,within,apart)) :
    KernelGemm.Both<{U32.is_zero((_ % 8 : U32)) == True{} : Bool},Placing.Ending<Grid.lanes_at(part_values,axis),_,extent>>
  %fits_other(values,part_values,mark,axis,start,fits,apart) : KernelGemm.Both<{U32.is_zero((0 % 8 : U32)) == True{} : Bool},Placing.Ending<_,0,extent>>
  KernelGemm.Both{{==},Placing.Last{{==}}}

# The part along the lane axis of the target, its lanes kept.
def part_lane(same: Bool,+values: List<&2,U32>,+part_values: List<&2,U32>,+mark: Nat,+start: U32,+axis: Nat,+classified: {Nat.is_eq(axis,mark) == same : Bool},
  +fits: {fits_marked(values,part_values,mark,start) == True{} : Bool},+within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},
  +keeps: {kept_lanes(axis,mark,start,Grid.lanes_at(part_values,mark),Grid.lanes_at(values,mark)) == True{} : Bool}) ->
  KernelGemm.Both<{U32.is_zero((Grid.lanes_at(marked(List.length(&2,U32,values),mark,start),axis) % 8 : U32)) == True{} : Bool},
    Placing.Ending<Grid.lanes_at(part_values,axis),Grid.lanes_at(marked(List.length(&2,U32,values),mark,start),axis),Grid.lanes_at(values,axis)>>:
  match same:
    case True{}:
      +width = Grid.lanes_at(part_values,mark)
      +extent = Grid.lanes_at(values,mark)
      +rest = U32.is_zero((start % 8 : U32)) && (U32.is_zero((width % 8 : U32)) || U32.is_eq((start + width : U32),extent))
      %Equal.sym(Nat,axis,mark,Order.equal_of_is_eq(axis,mark,classified)) :
        KernelGemm.Both<{U32.is_zero((Grid.lanes_at(marked(List.length(&2,U32,values),mark,start),_) % 8 : U32)) == True{} : Bool},
          Placing.Ending<Grid.lanes_at(part_values,_),Grid.lanes_at(marked(List.length(&2,U32,values),mark,start),_),Grid.lanes_at(values,_)>>
      hit_lane(values,part_values,mark,start,fits,{%classified : {Bool.not(_) || rest == True{} : Bool} keeps : {rest == True{} : Bool}})
    case False{}:
      other_lane(values,part_values,mark,start,axis,fits,within,classified)

# ---- The facts of the two sources ----

# A source placed in the target with its cells placed in the cells of the
# target, along a lane axis inside it.
def source_lanes(+limits: Machine.Limits,+values: List<&2,U32>,+source: List<&2,U32>,+shifts: List<&2,U32>,+axis: Nat,+position: Grid.Position,
  +within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},+source_within: {Nat.is_lt(axis,List.length(&2,U32,source)) == True{} : Bool},
  +ranked: {List.length(&2,U32,source) == List.length(&2,U32,values) : Nat},+shifts_same: {List.length(&2,U32,shifts) == List.length(&2,U32,source) : Nat},
  +placed: Facts.PlacedIn(values,source,shifts),+cell_placed: Facts.PlacedIn(Tiled.cell_shape(values,axis,position),Tiled.cell_shape(source,axis,position),Placing.cell_shifts(shifts,axis,position)),
  +room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,axis,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,axis,position))) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+source_positive: {Nat.is_lt(0n,ShapeLaws.elements(source)) == True{} : Bool},
  +source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(source,axis,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lane: KernelGemm.Both<{U32.is_zero((Grid.lanes_at(shifts,axis) % 8 : U32)) == True{} : Bool},Placing.Ending<Grid.lanes_at(source,axis),Grid.lanes_at(shifts,axis),Grid.lanes_at(values,axis)>>) ->
  Placing.Lanes<limits,Machine.half_word(limits),Grid.lanes_at(values,axis),List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),source,shifts,axis,position>:
  KernelGemm.Both{+aligned,+ending} = lane
  Placing.Lanes{source_within,ranked,
    Equal.trans(Nat,List.length(&2,U32,Tiled.cell_shape(source,axis,position)),List.length(&2,U32,source),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),Rooms.cell_rank(source,axis,position),
      Equal.trans(Nat,List.length(&2,U32,source),List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),ranked,
        Equal.sym(Nat,List.length(&2,U32,Tiled.cell_shape(values,axis,position)),List.length(&2,U32,values),Rooms.cell_rank(values,axis,position)))),
    shifts_same,
    placed_room(Tiled.cell_shape(values,axis,position),Tiled.cell_shape(source,axis,position),Placing.cell_shifts(shifts,axis,position),limits,cell_placed,cells_room,cells_positive),
    placed_room(values,source,shifts,limits,placed,room,positive),
    half_room(limits,ShapeLaws.elements(Tiled.cell_shape(source,axis,position)),source_cells_room),
    half_room(limits,ShapeLaws.elements(source),source_room),
    Rooms.removed_room(limits,source,axis,source_within,source_positive,source_room),
    aligned,ending,
    Encoded.eight_blocks_exact(limits,Grid.lanes_at(source,axis),Rooms.lane_room(limits,source,axis,source_within,source_positive,source_room)),
    placed_lane(values,source,shifts,axis,placed,within)}

# A source placed in the target, along the unit axis past it.
def source_units(+limits: Machine.Limits,+values: List<&2,U32>,+source: List<&2,U32>,+shifts: List<&2,U32>,+axis: Nat,
  +past: {Nat.is_lt(axis,List.length(&2,U32,source)) == False{} : Bool},+inside: {Nat.is_le(axis,List.length(&2,U32,source)) == True{} : Bool},
  +ranked: {List.length(&2,U32,source) == List.length(&2,U32,values) : Nat},+shifts_same: {List.length(&2,U32,shifts) == List.length(&2,U32,source) : Nat},
  +placed: Facts.PlacedIn(values,source,shifts),
  +room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Placing.Units<limits,Machine.half_word(limits),List.length(&2,U32,values),source,shifts,axis>:
  Placing.Units{past,inside,ranked,shifts_same,placed_room(values,source,shifts,limits,placed,room,positive),half_room(limits,ShapeLaws.elements(source),source_room),
    {%Equal.sym(List<&2,U32>,Grid.removed(source,axis),source,Public.removed_past(source,axis,past)) : {Nat.is_le(ShapeLaws.elements(_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
      source_room : {Nat.is_le(ShapeLaws.elements(Grid.removed(source,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}}

# ---- The written values and cells ----

# What the written values read: the target placed in itself, then the part
# placed from start along axis.
def written_values(~Logical: Data,~Part: Data,+logical: Logical,+part: Part,+values: List<&2,U32>,+part_values: List<&2,U32>,+axis: U32,+start: U32) ->
  Gathered.Placings<Written.Written<Logical,Part>,F32>:
  Gathered.Placings{[Written.Before{logical},Written.Over{part}],
    [Layout.placed_segments(values,Layout.shifts_at(List.length(&2,U32,values),0,0,0)),Layout.placed_segments(part_values,Layout.shifts_at(List.length(&2,U32,values),axis,start,0))],
    [values,part_values],values,0.0}

# What the written cells read: the cells of the target and of the part, each
# lifted at its geometry, at the shifts of their cells.
def written_cells(~Logical: Data,~Part: Data,+logical: Logical,+part: Part,+values: List<&2,U32>,+part_values: List<&2,U32>,+shifts: List<&2,U32>,+part_shifts: List<&2,U32>,
  +axis: Nat,+position: Grid.Position,+fill: Tile.Tile8) -> Gathered.Placings<Written.Written<Representation.Lifted<Logical>,Representation.Lifted<Part>>,Tile.Tile8>:
  Gathered.Placings{[Written.Before{Representation.Lifted{logical,Public.geometry_of(values,axis,position)}},Written.Over{Representation.Lifted{part,Public.geometry_of(part_values,axis,position)}}],
    [Layout.placed_segments(Tiled.cell_shape(values,axis,position),Placing.cell_shifts(shifts,axis,position)),Layout.placed_segments(Tiled.cell_shape(part_values,axis,position),Placing.cell_shifts(part_shifts,axis,position))],
    [Tiled.cell_shape(values,axis,position),Tiled.cell_shape(part_values,axis,position)],Tiled.cell_shape(values,axis,position),fill}

# A cell of the written cells with the lanes inside the target holds the
# cell of the written values.
def lanes_agree(~Logical: Data,~original: Logical -> U32 -> F32,~Part: Data,~part_value: Part -> U32 -> F32,+logical: Logical,+part: Part,+values: List<&2,U32>,+part_values: List<&2,U32>,
  +shifts: List<&2,U32>,+part_shifts: List<&2,U32>,+axis: Nat,+position: Grid.Position,+limits: Machine.Limits,+fill: Tile.Tile8,
  +target: Placing.Target<limits,Machine.half_word(limits),values,axis,position>,+within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool},
  +facts: Placing.SourcesLanes(limits,Machine.half_word(limits),Grid.lanes_at(values,axis),List.length(&2,U32,values),List.length(&2,U32,Tiled.cell_shape(values,axis,position)),[values,part_values],[shifts,part_shifts],axis,position),
  +zero: {Placing.cell_shifts(shifts,axis,position) == zeros(List.length(&2,U32,Tiled.cell_shape(values,axis,position))) : List<&2,U32>},
  +zero_room: Placing.Room(Tiled.cell_shape(values,axis,position),zeros(List.length(&2,U32,Tiled.cell_shape(values,axis,position))),Machine.half_word(limits)),
  +cells_room: {Nat.is_le(Nat.double(ShapeLaws.elements(Tiled.cell_shape(values,axis,position))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +index: U32,+inside: {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(Tiled.cell_shape(values,axis,position))) == True{} : Bool}) ->
  {Gathered.placing_value(~Tile.Tile8,~Written.Written<Representation.Lifted<Logical>,Representation.Lifted<Part>>,~Written.written_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value)),written_cells(~Logical,~Part,logical,part,values,part_values,shifts,part_shifts,axis,position,fill),index) == Representation.lifted(~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),Representation.Lifted{Gathered.Placings{[Written.Before{logical},Written.Over{part}],[Layout.placed_segments(values,shifts),Layout.placed_segments(part_values,part_shifts)],[values,part_values],values,0.0},Public.geometry_of(values,axis,position)},index) : Tile.Tile8}:
  +logicals = {[Written.Before{logical},Written.Over{part}] : List<&2,Written.Written<Logical,Part>>}
  +sources = {[values,part_values] : List<&2,List<&2,U32>>}
  +shift_lists = {[shifts,part_shifts] : List<&2,List<&2,U32>>}
  +cells = Tiled.cell_shape(values,axis,position)
  +part_cells = Tiled.cell_shape(part_values,axis,position)
  +words = Grid.coordinates(cells,index)
  Equal.trans(Tile.Tile8,Gathered.placing_value(~Tile.Tile8,~Written.Written<Representation.Lifted<Logical>,Representation.Lifted<Part>>,~Written.written_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value)),written_cells(~Logical,~Part,logical,part,values,part_values,shifts,part_shifts,axis,position,fill),index),Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Written.Written<Logical,Part>>,~Representation.lifted(~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),Gathered.Placings{Placing.lifted_logicals(~Written.Written<Logical,Part>,logicals,sources,axis,position),Placing.cell_segments(sources,shift_lists,axis,position),Placing.cell_sources(sources,axis,position),cells,fill},index),Representation.lifted(~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),Representation.Lifted{Gathered.Placings{[Written.Before{logical},Written.Over{part}],[Layout.placed_segments(values,shifts),Layout.placed_segments(part_values,part_shifts)],[values,part_values],values,0.0},Public.geometry_of(values,axis,position)},index),
    relabeled_result(~Logical,~original,~Part,~part_value,logical,part,Public.geometry_of(values,axis,position),Public.geometry_of(part_values,axis,position),
      Placements.segments_cell(Layout.placed_segments(cells,Placing.cell_shifts(shifts,axis,position)),words,Tasks.strides_of(cells)),
      Placements.segments_cell(Layout.placed_segments(part_cells,Placing.cell_shifts(part_shifts,axis,position)),words,Tasks.strides_of(part_cells)),fill),
    Placing.cell_within(~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value),logicals,sources,shift_lists,limits,Machine.half_word(limits),values,axis,position,fill,0.0,index,target,within,{==},facts,inside,
      covered(values,part_values,shifts,part_shifts,axis,position,limits,index,zero,zero_room,cells_room,inside)))

# A cell of the written cells with the lanes on the unit axis holds the cell
# of the written values.
def units_agree(~Logical: Data,~original: Logical -> U32 -> F32,~Part: Data,~part_value: Part -> U32 -> F32,+logical: Logical,+part: Part,+values: List<&2,U32>,+part_values: List<&2,U32>,
  +shifts: List<&2,U32>,+part_shifts: List<&2,U32>,+axis: Nat,+position: Grid.Position,+limits: Machine.Limits,+fill: Tile.Tile8,
  +target: Placing.Target<limits,Machine.half_word(limits),values,axis,position>,+past: {Nat.is_lt(axis,List.length(&2,U32,values)) == False{} : Bool},
  +facts: Placing.SourcesUnits(limits,Machine.half_word(limits),List.length(&2,U32,values),[values,part_values],[shifts,part_shifts],axis),
  +zero: {Placing.cell_shifts(shifts,axis,position) == zeros(List.length(&2,U32,Tiled.cell_shape(values,axis,position))) : List<&2,U32>},
  +zero_room: Placing.Room(Tiled.cell_shape(values,axis,position),zeros(List.length(&2,U32,Tiled.cell_shape(values,axis,position))),Machine.half_word(limits)),
  +cells_room: {Nat.is_le(Nat.double(ShapeLaws.elements(Tiled.cell_shape(values,axis,position))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +index: U32,+inside: {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(Tiled.cell_shape(values,axis,position))) == True{} : Bool}) ->
  {Gathered.placing_value(~Tile.Tile8,~Written.Written<Representation.Lifted<Logical>,Representation.Lifted<Part>>,~Written.written_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value)),written_cells(~Logical,~Part,logical,part,values,part_values,shifts,part_shifts,axis,position,fill),index) == Representation.lifted(~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),Representation.Lifted{Gathered.Placings{[Written.Before{logical},Written.Over{part}],[Layout.placed_segments(values,shifts),Layout.placed_segments(part_values,part_shifts)],[values,part_values],values,0.0},Public.geometry_of(values,axis,position)},index) : Tile.Tile8}:
  +logicals = {[Written.Before{logical},Written.Over{part}] : List<&2,Written.Written<Logical,Part>>}
  +sources = {[values,part_values] : List<&2,List<&2,U32>>}
  +shift_lists = {[shifts,part_shifts] : List<&2,List<&2,U32>>}
  +cells = Tiled.cell_shape(values,axis,position)
  +part_cells = Tiled.cell_shape(part_values,axis,position)
  +words = Grid.coordinates(cells,index)
  Equal.trans(Tile.Tile8,Gathered.placing_value(~Tile.Tile8,~Written.Written<Representation.Lifted<Logical>,Representation.Lifted<Part>>,~Written.written_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value)),written_cells(~Logical,~Part,logical,part,values,part_values,shifts,part_shifts,axis,position,fill),index),Gathered.placing_value(~Tile.Tile8,~Representation.Lifted<Written.Written<Logical,Part>>,~Representation.lifted(~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),Gathered.Placings{Placing.lifted_logicals(~Written.Written<Logical,Part>,logicals,sources,axis,position),Placing.cell_segments(sources,shift_lists,axis,position),Placing.cell_sources(sources,axis,position),cells,fill},index),Representation.lifted(~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),Representation.Lifted{Gathered.Placings{[Written.Before{logical},Written.Over{part}],[Layout.placed_segments(values,shifts),Layout.placed_segments(part_values,part_shifts)],[values,part_values],values,0.0},Public.geometry_of(values,axis,position)},index),
    relabeled_result(~Logical,~original,~Part,~part_value,logical,part,Public.geometry_of(values,axis,position),Public.geometry_of(part_values,axis,position),
      Placements.segments_cell(Layout.placed_segments(cells,Placing.cell_shifts(shifts,axis,position)),words,Tasks.strides_of(cells)),
      Placements.segments_cell(Layout.placed_segments(part_cells,Placing.cell_shifts(part_shifts,axis,position)),words,Tasks.strides_of(part_cells)),fill),
    Placing.cell_unit(~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value),logicals,sources,shift_lists,limits,Machine.half_word(limits),values,axis,position,fill,0.0,index,target,past,{==},facts,inside,
      covered(values,part_values,shifts,part_shifts,axis,position,limits,index,zero,zero_room,cells_room,inside)))

# Written cells that hold the cells of the target and of the part hold the
# cells of the written values.
def cells_restated(~Logical: Data,~original: Logical -> U32 -> F32,~Part: Data,~part_value: Part -> U32 -> F32,within: Bool,+logical: Logical,+part: Part,+values: List<&2,U32>,+part_values: List<&2,U32>,
  +axis: U32,+start: U32,+lane: Nat,+position: Grid.Position,+limits: Machine.Limits,+fill: Tile.Tile8,
  +classified: {Nat.is_lt(lane,List.length(&2,U32,values)) == within : Bool},+inside: {Nat.is_le(lane,List.length(&2,U32,values)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,lane,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,lane,position))) == True{} : Bool},
  +part_room: {Nat.is_le(ShapeLaws.elements(part_values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+part_positive: {Nat.is_lt(0n,ShapeLaws.elements(part_values)) == True{} : Bool},
  +part_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(part_values,lane,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +fits: {Layout.slice_fits(values,part_values,axis,start,0) == True{} : Bool},+marked_fits: {fits_marked(values,part_values,U32.to_nat(axis),start) == True{} : Bool},
  +apart: {Nat.is_eq(lane,U32.to_nat(axis)) == False{} : Bool},
  +whole_cells: {Placing.cell_shifts(Layout.shifts_at(List.length(&2,U32,values),0,0,0),lane,position) == zeros(List.length(&2,U32,Tiled.cell_shape(values,lane,position))) : List<&2,U32>},
  +cells_in: Facts.PlacedIn(Tiled.cell_shape(values,lane,position),Tiled.cell_shape(part_values,lane,position),Placing.cell_shifts(Layout.shifts_at(List.length(&2,U32,values),axis,start,0),lane,position)),
  -tensor: Tensors.Tensor<Tile.Tile8>,
  shaped: Global.Shaped(~Tile.Tile8,~Gathered.Placings<Written.Written<Representation.Lifted<Logical>,Representation.Lifted<Part>>,Tile.Tile8>,~Gathered.placing_value(~Tile.Tile8,~Written.Written<Representation.Lifted<Logical>,Representation.Lifted<Part>>,~Written.written_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value))),written_cells(~Logical,~Part,logical,part,values,part_values,Layout.shifts_at(List.length(&2,U32,values),0,0,0),Layout.shifts_at(List.length(&2,U32,values),axis,start,0),lane,position,fill),Tiled.cell_shape(values,lane,position),tensor)) ->
  Global.Shaped(~Tile.Tile8,~Representation.Lifted<Gathered.Placings<Written.Written<Logical,Part>,F32>>,~Representation.lifted(~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value))),Representation.Lifted{written_values(~Logical,~Part,logical,part,values,part_values,axis,start),Public.geometry_of(values,lane,position)},
    Tiled.cell_shape(values,lane,position),tensor):
  match within:
    case True{}:
      +rank = List.length(&2,U32,values)
      +mark = U32.to_nat(axis)
      +whole = Layout.shifts_at(rank,0,0,0)
      +shifts = Layout.shifts_at(rank,axis,start,0)
      +cells = Tiled.cell_shape(values,lane,position)
      +ranked = fits_length(values,part_values,mark,start,marked_fits)
      +shifts_same = Equal.trans(Nat,List.length(&2,U32,shifts),rank,List.length(&2,U32,part_values),shifts_length(rank,axis,start,0),Equal.sym(Nat,List.length(&2,U32,part_values),rank,ranked))
      +marks = shifts_marked(rank,axis,start,0n,mark,Machine.capacity(limits),Limits.capacity_fits(limits),rank_room,{==})
      +cells_whole = Equal.trans(List<&2,U32>,Layout.shifts_at(List.length(&2,U32,cells),0,0,0),zeros(List.length(&2,U32,cells)),Placing.cell_shifts(whole,lane,position),
        shifts_zero(List.length(&2,U32,cells),0),Equal.sym(List<&2,U32>,Placing.cell_shifts(whole,lane,position),zeros(List.length(&2,U32,cells)),whole_cells))
      +whole_in_cells = {%cells_whole : Facts.PlacedIn(cells,cells,_) Written.whole_in(cells,0) : Facts.PlacedIn(cells,cells,Placing.cell_shifts(whole,lane,position))}
      +zero_room = {%shifts_zero(List.length(&2,U32,cells),0) : Placing.Room(cells,_,Machine.half_word(limits))
        placed_room(cells,cells,Layout.shifts_at(List.length(&2,U32,cells),0,0,0),limits,Written.whole_in(cells,0),cells_room,cells_positive) :
        Placing.Room(cells,zeros(List.length(&2,U32,cells)),Machine.half_word(limits))}
      +cells_half = Limits.double_bound(limits,ShapeLaws.elements(cells),cells_room)
      +lane_facts = {KernelGemm.Both{
        source_lanes(limits,values,values,whole,lane,position,classified,classified,{==},shifts_length(rank,0,0,0),Written.whole_in(values,0),whole_in_cells,
          room,positive,cells_room,cells_positive,room,positive,cells_room,target_lane(values,lane,classified)),
        KernelGemm.Both{
          source_lanes(limits,values,part_values,shifts,lane,position,classified,{%Equal.sym(Nat,List.length(&2,U32,part_values),rank,ranked) : {Nat.is_lt(lane,_) == True{} : Bool} classified :
              {Nat.is_lt(lane,List.length(&2,U32,part_values)) == True{} : Bool}},
            ranked,shifts_same,Written.slice_in(values,part_values,axis,start,0,fits),cells_in,room,positive,cells_room,cells_positive,part_room,part_positive,part_cells_room,
            {%Equal.sym(List<&2,U32>,shifts,marked(rank,mark,start),marks) :
                KernelGemm.Both<{U32.is_zero((Grid.lanes_at(_,lane) % 8 : U32)) == True{} : Bool},Placing.Ending<Grid.lanes_at(part_values,lane),Grid.lanes_at(_,lane),Grid.lanes_at(values,lane)>>
              other_lane(values,part_values,mark,start,lane,marked_fits,classified,apart) :
                KernelGemm.Both<{U32.is_zero((Grid.lanes_at(shifts,lane) % 8 : U32)) == True{} : Bool},
                  Placing.Ending<Grid.lanes_at(part_values,lane),Grid.lanes_at(shifts,lane),Grid.lanes_at(values,lane)>>}),
          Unit{}}} :
        Placing.SourcesLanes(limits,Machine.half_word(limits),Grid.lanes_at(values,lane),rank,List.length(&2,U32,cells),[values,part_values],[whole,shifts],lane,position)}
      +target = {Placing.Target{Limits.half_word_fits(limits),inside,cells_half,Limits.double_bound(limits,ShapeLaws.elements(values),room),Rooms.removed_room(limits,values,lane,classified,positive,room)} :
        Placing.Target<limits,Machine.half_word(limits),values,lane,position>}
      Global.restated_below(~Tile.Tile8,~Gathered.Placings<Written.Written<Representation.Lifted<Logical>,Representation.Lifted<Part>>,Tile.Tile8>,~Representation.Lifted<Gathered.Placings<Written.Written<Logical,Part>,F32>>,~Gathered.placing_value(~Tile.Tile8,~Written.Written<Representation.Lifted<Logical>,Representation.Lifted<Part>>,~Written.written_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value))),~Representation.lifted(~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value))),~ShapeLaws.elements(cells),
        written_cells(~Logical,~Part,logical,part,values,part_values,whole,shifts,lane,position,fill),
        Representation.Lifted{written_values(~Logical,~Part,logical,part,values,part_values,axis,start),Public.geometry_of(values,lane,position)},
        index => below => lanes_agree(~Logical,~original,~Part,~part_value,logical,part,values,part_values,whole,shifts,lane,position,limits,fill,target,classified,lane_facts,
          whole_cells,zero_room,cells_half,index,below),
        cells,tensor,shaped,Order.reflexive(ShapeLaws.elements(cells)),Machine.capacity(limits),cells_room)
    case False{}:
      +rank = List.length(&2,U32,values)
      +mark = U32.to_nat(axis)
      +whole = Layout.shifts_at(rank,0,0,0)
      +shifts = Layout.shifts_at(rank,axis,start,0)
      +cells = Tiled.cell_shape(values,lane,position)
      +ranked = fits_length(values,part_values,mark,start,marked_fits)
      +shifts_same = Equal.trans(Nat,List.length(&2,U32,shifts),rank,List.length(&2,U32,part_values),shifts_length(rank,axis,start,0),Equal.sym(Nat,List.length(&2,U32,part_values),rank,ranked))
      +marks = shifts_marked(rank,axis,start,0n,mark,Machine.capacity(limits),Limits.capacity_fits(limits),rank_room,{==})
      +cells_whole = Equal.trans(List<&2,U32>,Layout.shifts_at(List.length(&2,U32,cells),0,0,0),zeros(List.length(&2,U32,cells)),Placing.cell_shifts(whole,lane,position),
        shifts_zero(List.length(&2,U32,cells),0),Equal.sym(List<&2,U32>,Placing.cell_shifts(whole,lane,position),zeros(List.length(&2,U32,cells)),whole_cells))
      +whole_in_cells = {%cells_whole : Facts.PlacedIn(cells,cells,_) Written.whole_in(cells,0) : Facts.PlacedIn(cells,cells,Placing.cell_shifts(whole,lane,position))}
      +zero_room = {%shifts_zero(List.length(&2,U32,cells),0) : Placing.Room(cells,_,Machine.half_word(limits))
        placed_room(cells,cells,Layout.shifts_at(List.length(&2,U32,cells),0,0,0),limits,Written.whole_in(cells,0),cells_room,cells_positive) :
        Placing.Room(cells,zeros(List.length(&2,U32,cells)),Machine.half_word(limits))}
      +cells_half = Limits.double_bound(limits,ShapeLaws.elements(cells),cells_room)
      +part_past = {%Equal.sym(Nat,List.length(&2,U32,part_values),rank,ranked) : {Nat.is_lt(lane,_) == False{} : Bool} classified :
        {Nat.is_lt(lane,List.length(&2,U32,part_values)) == False{} : Bool}}
      +part_inside = {%Equal.sym(Nat,List.length(&2,U32,part_values),rank,ranked) : {Nat.is_le(lane,_) == True{} : Bool} inside :
        {Nat.is_le(lane,List.length(&2,U32,part_values)) == True{} : Bool}}
      +unit_facts = {KernelGemm.Both{source_units(limits,values,values,whole,lane,classified,inside,{==},shifts_length(rank,0,0,0),Written.whole_in(values,0),room,positive,room),
        KernelGemm.Both{source_units(limits,values,part_values,shifts,lane,part_past,part_inside,ranked,shifts_same,Written.slice_in(values,part_values,axis,start,0,fits),room,positive,part_room),Unit{}}} :
        Placing.SourcesUnits(limits,Machine.half_word(limits),rank,[values,part_values],[whole,shifts],lane)}
      +target = {Placing.Target{Limits.half_word_fits(limits),inside,cells_half,Limits.double_bound(limits,ShapeLaws.elements(values),room),
          {%Equal.sym(List<&2,U32>,Grid.removed(values,lane),values,Public.removed_past(values,lane,classified)) : {Nat.is_le(ShapeLaws.elements(_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
            room : {Nat.is_le(ShapeLaws.elements(Grid.removed(values,lane)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}} :
        Placing.Target<limits,Machine.half_word(limits),values,lane,position>}
      Global.restated_below(~Tile.Tile8,~Gathered.Placings<Written.Written<Representation.Lifted<Logical>,Representation.Lifted<Part>>,Tile.Tile8>,~Representation.Lifted<Gathered.Placings<Written.Written<Logical,Part>,F32>>,~Gathered.placing_value(~Tile.Tile8,~Written.Written<Representation.Lifted<Logical>,Representation.Lifted<Part>>,~Written.written_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value))),~Representation.lifted(~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value))),~ShapeLaws.elements(cells),
        written_cells(~Logical,~Part,logical,part,values,part_values,whole,shifts,lane,position,fill),
        Representation.Lifted{written_values(~Logical,~Part,logical,part,values,part_values,axis,start),Public.geometry_of(values,lane,position)},
        index => below => units_agree(~Logical,~original,~Part,~part_value,logical,part,values,part_values,whole,shifts,lane,position,limits,fill,target,classified,unit_facts,
          whole_cells,zero_room,cells_half,index,below),
        cells,tensor,shaped,Order.reflexive(ShapeLaws.elements(cells)),Machine.capacity(limits),cells_room)

# ---- The written values at a channel ----

# Shifts of zero leave the digits.
def shifted_zeros(count: Nat,digits: List<&2,Nat>,+same: {List.length(&2,Nat,digits) == count : Nat}) -> {Placing.shifted(zeros(count),digits) == digits : List<&2,Nat>}:
  match count digits:
    case 0n Nil{}: {==}
    case 1n+ +rest Con{+digit,+more}:
      %Equal.sym(Nat,Nat.sub(digit,0n),digit,Order.subtract_zero(digit)) : {_ <> Placing.shifted(zeros(rest),more) == digit <> more : List<&2,Nat>}
      %Equal.sym(List<&2,Nat>,Placing.shifted(zeros(rest),more),more,shifted_zeros(rest,more,Placing.pred_same(List.length(&2,Nat,more),rest,same))) :
        {digit <> _ == digit <> more : List<&2,Nat>}
      {==}
    case 1n+ +rest Nil{}:
      Empty.absurd({Placing.shifted(zeros(1n+rest),Nil{}) == Nil{} : List<&2,Nat>},Digits.zero_successor(rest,same))
    case 0n Con{+digit,+more}:
      Empty.absurd({Placing.shifted(zeros(0n),Con{digit,more}) == Con{digit,more} : List<&2,Nat>},
        Digits.zero_successor(List.length(&2,Nat,more),Equal.sym(Nat,1n+List.length(&2,Nat,more),0n,same)))

# ---- Writing the part into the cells ----

# A part that fits the target from start along axis fits it at the marked
# axis.
def marked_of(+limits: Machine.Limits,+values: List<&2,U32>,+part_values: List<&2,U32>,+axis: U32,+start: U32,
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: {Layout.slice_fits(values,part_values,axis,start,0) == True{} : Bool}) ->
  {fits_marked(values,part_values,U32.to_nat(axis),start) == True{} : Bool}:
  Equal.trans(Bool,fits_marked(values,part_values,U32.to_nat(axis),start),Layout.slice_fits(values,part_values,axis,start,0),True{},
    Equal.sym(Bool,Layout.slice_fits(values,part_values,axis,start,0),fits_marked(values,part_values,U32.to_nat(axis),start),
      fits_mark_eq(values,part_values,axis,start,0n,U32.to_nat(axis),Machine.capacity(limits),Limits.capacity_fits(limits),rank_room,{==})),fits)

# The cells of a part arranged at the lane axis and position of the target,
# written into the cells of the target, hold the written values.
def cells_written(~Logical: Data,~original: Logical -> U32 -> F32,~Part: Data,~part_value: Part -> U32 -> F32,+logical: Logical,+part: Part,+values: List<&2,U32>,+part_values: List<&2,U32>,
  +axis: U32,+start: U32,-pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+lane: Nat,+lanes: U32,+position: Grid.Position,-part_tiled: Tiled.Tiled,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: {Layout.slice_fits(values,part_values,axis,start,0) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(part_values)) == True{} : Bool},
  +apart: {Nat.is_eq(U32.to_nat(axis),lane) == False{} : Bool},
  +usable: Global.Usable<Tile.Tile8,shape,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers}>,
  +target_contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position}),
  +part_shaped: Public.ShapedLaid(~Part,~part_value,part,part_values,Tiled.Tiled{Tiled.cells_of(Tiled.arranged(part_tiled,lane,position)),lane,Grid.lanes_at(part_values,lane),position})) ->
  Public.Contract(~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),written_values(~Logical,~Part,logical,part,values,part_values,axis,start),values,Tiled.Tiled{Layout.set_slice(~Tile.Tile8,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},U32.from_nat(Tiled.cell_axis(position,U32.to_nat(axis),lane,List.length(&2,U32,shape))),
    start,Tiled.cells_of(Tiled.arranged(part_tiled,lane,position))),lane,lanes,position}):
  Global.Usable{+model,+model_fill,+model_workers,+equation} = usable
  KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}} = target_contract
  KernelGemm.Both{KernelGemm.Both{Global.Usable{+part_model,+part_fill,+part_workers,+part_equation},+part_global},Public.Laid{+part_inside,+part_cells_shape,+part_restored,+part_counted,+part_pinned}} = part_shaped
  +rank = List.length(&2,U32,values)
  +mark = U32.to_nat(axis)
  +cap = Machine.capacity(limits)
  +cells = Tiled.cell_shape(values,lane,position)
  +part_cells = Tiled.cell_shape(part_values,lane,position)
  +cell_axis = Tiled.cell_axis(position,mark,lane,rank)
  +rest = U32.is_zero((start % 8 : U32)) && (U32.is_zero((Layout.extent_at(part_values,mark) % 8 : U32)) || U32.is_eq((start + Layout.extent_at(part_values,mark) : U32),Layout.extent_at(values,mark)))
  +lane_apart = Equal.trans(Bool,Nat.is_eq(lane,mark),Nat.is_eq(mark,lane),False{},Order.equal_symmetric(lane,mark),apart)
  +keeps = {%Equal.sym(Bool,Nat.is_eq(lane,mark),False{},lane_apart) : {Bool.not(_) || rest == True{} : Bool} {==} :
    {kept_lanes(lane,mark,start,Layout.extent_at(part_values,mark),Layout.extent_at(values,mark)) == True{} : Bool}}
  +undivided = {%Equal.sym(Bool,Nat.is_eq(mark,lane),False{},apart) : {Bool.pick(U32,_,(start / 8 : U32),start) == start : U32} {==} :
    {Bool.pick(U32,Nat.is_eq(mark,lane),(start / 8 : U32),start) == start : U32}}
  +whole = Layout.shifts_at(rank,0,0,0)
  +shifts = Layout.shifts_at(rank,axis,start,0)
  +marked_fits = marked_of(limits,values,part_values,axis,start,rank_room,fits)
  +ranked = fits_length(values,part_values,mark,start,marked_fits)
  +mark_inside = fits_inside(values,part_values,mark,start,marked_fits)
  +part_mark_inside = {%Equal.sym(Nat,List.length(&2,U32,part_values),rank,ranked) : {Nat.is_lt(mark,_) == True{} : Bool} mark_inside :
    {Nat.is_lt(mark,List.length(&2,U32,part_values)) == True{} : Bool}}
  +room = StorageCount.counted_room(limits,values,counted,default_capacity)
  +positive = Facts.placed_positive(values,part_values,shifts,Written.slice_in(values,part_values,axis,start,0,fits),nonempty)
  +part_room = StorageCount.counted_room(limits,part_values,part_counted,default_capacity)
  +packed_global = {%Equal.sym(List<&2,U32>,cells,shape,shaped) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,lane,position)},
        Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),_,model_fill,Tensors.Usable{},model_workers})
    %Equal.sym(Tensors.Tensor<Tile.Tile8>,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,model_fill,Tensors.Usable{},model_workers},Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},equation) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,lane,position)},_)
    global : Global.Contract(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,lane,position)},
      Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,model_fill,Tensors.Usable{},model_workers})}
  +packed_part = {%Equal.sym(Tensors.Tensor<Tile.Tile8>,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,part_model),part_cells,part_fill,Tensors.Usable{},part_workers},Tiled.cells_of(Tiled.arranged(part_tiled,lane,position)),part_equation) :
      Global.Contract(~Tile.Tile8,~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value),Representation.Lifted{part,Public.geometry_of(part_values,lane,position)},_)
    part_global : Global.Contract(~Tile.Tile8,~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value),Representation.Lifted{part,Public.geometry_of(part_values,lane,position)},
      Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,part_model),part_cells,part_fill,Tensors.Usable{},part_workers})}
  +cells_room = StorageCount.counted_room(limits,cells,Global.contract_counted(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
    Representation.Lifted{logical,Public.geometry_of(values,lane,position)},PiecesModel.pack(~Tile.Tile8,model),cells,model_fill,model_workers,packed_global),default_capacity)
  +part_cells_room = StorageCount.counted_room(limits,part_cells,Global.contract_counted(~Tile.Tile8,~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value),
    Representation.Lifted{part,Public.geometry_of(part_values,lane,position)},PiecesModel.pack(~Tile.Tile8,part_model),part_cells,part_fill,part_workers,packed_part),default_capacity)
  +cells_filled = Rooms.cells_positive(limits,values,lane,position,positive,room)
  +part_cells_filled = Rooms.cells_positive(limits,part_values,lane,position,nonempty,part_room)
  +cells_marked = {%undivided : {fits_marked(cells,part_cells,cell_axis,_) == True{} : Bool}
    cell_fits(limits,values,part_values,mark,start,lane,position,Layout.extent_at(part_values,mark),Layout.extent_at(values,mark),marked_fits,keeps,
      Rooms.lane_room(limits,values,mark,mark_inside,positive,room)) : {fits_marked(cells,part_cells,cell_axis,start) == True{} : Bool}}
  +cell_ranked = Rooms.cell_rank(values,lane,position)
  +cells_rank_room = {%Equal.sym(Nat,List.length(&2,U32,cells),rank,cell_ranked) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} rank_room :
    {Nat.is_le(List.length(&2,U32,cells),U32.to_nat(cap)) == True{} : Bool}}
  +axis_room = Order.transitive(cell_axis,List.length(&2,U32,cells),U32.to_nat(cap),Bounds.strict_le(cell_axis,List.length(&2,U32,cells),fits_inside(cells,part_cells,cell_axis,start,cells_marked)),cells_rank_room)
  +axis_back = Bounds.roundtrip(cell_axis,cap,Limits.capacity_fits(limits),axis_room)
  +cells_fits = Equal.trans(Bool,Layout.slice_fits(cells,part_cells,U32.from_nat(cell_axis),start,0),fits_marked(cells,part_cells,cell_axis,start),True{},
    fits_mark_eq(cells,part_cells,U32.from_nat(cell_axis),start,0n,cell_axis,cap,Limits.capacity_fits(limits),cells_rank_room,axis_back),cells_marked)
  +cells_nonempty = ViewMoved.none_zero(part_cells,part_cells_filled)
  +whole_cells = Equal.trans(List<&2,U32>,Placing.cell_shifts(whole,lane,position),zeros(rank),zeros(List.length(&2,U32,cells)),
    Equal.trans(List<&2,U32>,Placing.cell_shifts(whole,lane,position),Placing.cell_shifts(zeros(rank),lane,position),zeros(rank),
      Equal.cong(List<&2,U32>,List<&2,U32>,moved => Placing.cell_shifts(moved,lane,position),whole,zeros(rank),shifts_zero(rank,0)),cell_zeros(rank,lane,position)),
    Equal.cong(Nat,List<&2,U32>,count => zeros(count),rank,List.length(&2,U32,cells),Equal.sym(Nat,List.length(&2,U32,cells),rank,cell_ranked)))
  +cells_whole = Equal.trans(List<&2,U32>,Layout.shifts_at(List.length(&2,U32,cells),0,0,0),zeros(List.length(&2,U32,cells)),Placing.cell_shifts(whole,lane,position),
    shifts_zero(List.length(&2,U32,cells),0),Equal.sym(List<&2,U32>,Placing.cell_shifts(whole,lane,position),zeros(List.length(&2,U32,cells)),whole_cells))
  +marks = shifts_marked(rank,axis,start,0n,mark,cap,Limits.capacity_fits(limits),rank_room,{==})
  +cells_shifts = Equal.trans(List<&2,U32>,Layout.shifts_at(List.length(&2,U32,cells),U32.from_nat(cell_axis),start,0),marked(List.length(&2,U32,cells),cell_axis,start),Placing.cell_shifts(shifts,lane,position),
    shifts_marked(List.length(&2,U32,cells),U32.from_nat(cell_axis),start,0n,cell_axis,cap,Limits.capacity_fits(limits),cells_rank_room,axis_back),
    Equal.trans(List<&2,U32>,marked(List.length(&2,U32,cells),cell_axis,start),marked(rank,cell_axis,start),Placing.cell_shifts(shifts,lane,position),
      Equal.cong(Nat,List<&2,U32>,count => marked(count,cell_axis,start),List.length(&2,U32,cells),rank,cell_ranked),
      Equal.trans(List<&2,U32>,marked(rank,cell_axis,start),Placing.cell_shifts(marked(rank,mark,start),lane,position),Placing.cell_shifts(shifts,lane,position),
        Equal.sym(List<&2,U32>,Placing.cell_shifts(marked(rank,mark,start),lane,position),marked(rank,cell_axis,start),{%undivided : {Placing.cell_shifts(marked(rank,mark,start),lane,position) == marked(rank,cell_axis,_) : List<&2,U32>}
          cell_marked(rank,mark,start,lane,position,mark_inside) : {Placing.cell_shifts(marked(rank,mark,start),lane,position) == marked(rank,cell_axis,start) : List<&2,U32>}}),
        Equal.cong(List<&2,U32>,List<&2,U32>,moved => Placing.cell_shifts(moved,lane,position),marked(rank,mark,start),shifts,Equal.sym(List<&2,U32>,shifts,marked(rank,mark,start),marks)))))
  +cells_in = {%cells_shifts : Facts.PlacedIn(cells,part_cells,_) Written.slice_in(cells,part_cells,U32.from_nat(cell_axis),start,0,cells_fits) :
    Facts.PlacedIn(cells,part_cells,Placing.cell_shifts(shifts,lane,position))}
  +written = {%cells_shifts : Global.Contract(~Tile.Tile8,~Gathered.Placings<Written.Written<Representation.Lifted<Logical>,Representation.Lifted<Part>>,Tile.Tile8>,~Gathered.placing_value(~Tile.Tile8,~Written.Written<Representation.Lifted<Logical>,Representation.Lifted<Part>>,~Written.written_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value))),Gathered.Placings{[Written.Before{Representation.Lifted{logical,Public.geometry_of(values,lane,position)}},Written.Over{Representation.Lifted{part,Public.geometry_of(part_values,lane,position)}}],[Layout.placed_segments(cells,Placing.cell_shifts(whole,lane,position)),Layout.placed_segments(part_cells,_)],[cells,part_cells],cells,model_fill},Layout.set_slice(~Tile.Tile8,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,model_fill,Tensors.Usable{},model_workers},U32.from_nat(cell_axis),start,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,part_model),part_cells,part_fill,Tensors.Usable{},part_workers}))
    %cells_whole : Global.Contract(~Tile.Tile8,~Gathered.Placings<Written.Written<Representation.Lifted<Logical>,Representation.Lifted<Part>>,Tile.Tile8>,~Gathered.placing_value(~Tile.Tile8,~Written.Written<Representation.Lifted<Logical>,Representation.Lifted<Part>>,~Written.written_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value))),Gathered.Placings{[Written.Before{Representation.Lifted{logical,Public.geometry_of(values,lane,position)}},Written.Over{Representation.Lifted{part,Public.geometry_of(part_values,lane,position)}}],[Layout.placed_segments(cells,_),Layout.placed_segments(part_cells,Layout.shifts_at(List.length(&2,U32,cells),U32.from_nat(cell_axis),start,0))],[cells,part_cells],cells,model_fill},Layout.set_slice(~Tile.Tile8,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,model_fill,Tensors.Usable{},model_workers},U32.from_nat(cell_axis),start,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,part_model),part_cells,part_fill,Tensors.Usable{},part_workers}))
    Written.set_slice(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value),
      Representation.Lifted{logical,Public.geometry_of(values,lane,position)},Representation.Lifted{part,Public.geometry_of(part_values,lane,position)},limits,
      model,part_model,cells,part_cells,U32.from_nat(cell_axis),start,model_fill,part_fill,model_workers,part_workers,packed_global,packed_part,default_capacity,cells_rank_room,cells_fits,cells_nonempty) :
    Global.Contract(~Tile.Tile8,~Gathered.Placings<Written.Written<Representation.Lifted<Logical>,Representation.Lifted<Part>>,Tile.Tile8>,~Gathered.placing_value(~Tile.Tile8,~Written.Written<Representation.Lifted<Logical>,Representation.Lifted<Part>>,~Written.written_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value))),written_cells(~Logical,~Part,logical,part,values,part_values,whole,shifts,lane,position,model_fill),Layout.set_slice(~Tile.Tile8,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,model_fill,Tensors.Usable{},model_workers},U32.from_nat(cell_axis),start,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,part_model),part_cells,part_fill,Tensors.Usable{},part_workers}))}
  +ready = {%Equal.sym(Bool,Layout.slice_fits(cells,part_cells,U32.from_nat(cell_axis),start,0),True{},cells_fits) :
      {Contract.tensor_ready(~Tile.Tile8,Layout.slice_status(~Tile.Tile8,Tensors.Usable{},Tensors.Usable{},_,PiecesModel.pack(~Tile.Tile8,model),PiecesModel.pack(~Tile.Tile8,part_model),cells,part_cells,U32.from_nat(cell_axis),start,model_fill,U32.max(model_workers,part_workers))) == True{} : Bool}
    %Equal.sym(Bool,Shapes.has_zero(part_cells),False{},cells_nonempty) :
      {Contract.tensor_ready(~Tile.Tile8,Layout.slice_placed(~Tile.Tile8,_,PiecesModel.pack(~Tile.Tile8,model),PiecesModel.pack(~Tile.Tile8,part_model),cells,part_cells,U32.from_nat(cell_axis),start,model_fill,U32.max(model_workers,part_workers))) == True{} : Bool}
    {==} : {Contract.tensor_ready(~Tile.Tile8,Layout.set_slice(~Tile.Tile8,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,model_fill,Tensors.Usable{},model_workers},U32.from_nat(cell_axis),start,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,part_model),part_cells,part_fill,Tensors.Usable{},part_workers})) == True{} : Bool}}
  +extents = {%Equal.sym(Bool,Layout.slice_fits(cells,part_cells,U32.from_nat(cell_axis),start,0),True{},cells_fits) :
      {Contract.extents(~Tile.Tile8,Layout.slice_status(~Tile.Tile8,Tensors.Usable{},Tensors.Usable{},_,PiecesModel.pack(~Tile.Tile8,model),PiecesModel.pack(~Tile.Tile8,part_model),cells,part_cells,U32.from_nat(cell_axis),start,model_fill,U32.max(model_workers,part_workers))) == cells : List<&2,U32>}
    %Equal.sym(Bool,Shapes.has_zero(part_cells),False{},cells_nonempty) :
      {Contract.extents(~Tile.Tile8,Layout.slice_placed(~Tile.Tile8,_,PiecesModel.pack(~Tile.Tile8,model),PiecesModel.pack(~Tile.Tile8,part_model),cells,part_cells,U32.from_nat(cell_axis),start,model_fill,U32.max(model_workers,part_workers))) == cells : List<&2,U32>}
    {==} : {Contract.extents(~Tile.Tile8,Layout.set_slice(~Tile.Tile8,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,model_fill,Tensors.Usable{},model_workers},U32.from_nat(cell_axis),start,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,part_model),part_cells,part_fill,Tensors.Usable{},part_workers})) == cells : List<&2,U32>}}
  +restated = cells_restated(~Logical,~original,~Part,~part_value,Nat.is_lt(lane,rank),logical,part,values,part_values,axis,start,lane,position,limits,model_fill,{==},inside,rank_room,
    room,positive,cells_room,cells_filled,part_room,nonempty,part_cells_room,fits,marked_fits,lane_apart,whole_cells,cells_in,Layout.set_slice(~Tile.Tile8,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,model_fill,Tensors.Usable{},model_workers},U32.from_nat(cell_axis),start,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,part_model),part_cells,part_fill,Tensors.Usable{},part_workers}),
    Global.shaped(~Tile.Tile8,~Gathered.Placings<Written.Written<Representation.Lifted<Logical>,Representation.Lifted<Part>>,Tile.Tile8>,~Gathered.placing_value(~Tile.Tile8,~Written.Written<Representation.Lifted<Logical>,Representation.Lifted<Part>>,~Written.written_value(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value))),written_cells(~Logical,~Part,logical,part,values,part_values,whole,shifts,lane,position,model_fill),cells,Layout.set_slice(~Tile.Tile8,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,model_fill,Tensors.Usable{},model_workers},U32.from_nat(cell_axis),start,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,part_model),part_cells,part_fill,Tensors.Usable{},part_workers}),ready,extents,written))
  +laid = {Public.Laid{inside,{==},{%pinned : {Tiled.logical_shape(cells,lane,_,position) == values : List<&2,U32>} Public.cells_restored(values,lane,position) :
      {Tiled.logical_shape(cells,lane,lanes,position) == values : List<&2,U32>}},counted,pinned} : Public.Laid<cells,lane,lanes,position,values>}
  %equation : Public.Contract(~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),written_values(~Logical,~Part,logical,part,values,part_values,axis,start),values,Tiled.Tiled{Layout.set_slice(~Tile.Tile8,_,U32.from_nat(Tiled.cell_axis(position,mark,lane,List.length(&2,U32,shape))),start,Tiled.cells_of(Tiled.arranged(part_tiled,lane,position))),lane,lanes,position})
  %part_equation : Public.Contract(~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),written_values(~Logical,~Part,logical,part,values,part_values,axis,start),values,Tiled.Tiled{Layout.set_slice(~Tile.Tile8,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,model_fill,Tensors.Usable{},model_workers},
    U32.from_nat(Tiled.cell_axis(position,mark,lane,List.length(&2,U32,shape))),start,_),lane,lanes,position})
  %shaped : Public.Contract(~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),written_values(~Logical,~Part,logical,part,values,part_values,axis,start),values,Tiled.Tiled{Layout.set_slice(~Tile.Tile8,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),_,model_fill,Tensors.Usable{},model_workers},
    U32.from_nat(Tiled.cell_axis(position,mark,lane,List.length(&2,U32,_))),start,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,part_model),part_cells,part_fill,Tensors.Usable{},part_workers}),lane,lanes,position})
  %Equal.sym(Nat,List.length(&2,U32,cells),rank,cell_ranked) : Public.Contract(~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),written_values(~Logical,~Part,logical,part,values,part_values,axis,start),values,Tiled.Tiled{Layout.set_slice(~Tile.Tile8,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,model_fill,Tensors.Usable{},model_workers},
    U32.from_nat(Tiled.cell_axis(position,mark,lane,_)),start,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,part_model),part_cells,part_fill,Tensors.Usable{},part_workers}),lane,lanes,position})
  Public.laid_contract(~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),written_values(~Logical,~Part,logical,part,values,part_values,axis,start),values,
    Layout.set_slice(~Tile.Tile8,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),cells,model_fill,Tensors.Usable{},model_workers},U32.from_nat(cell_axis),start,Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,part_model),part_cells,part_fill,Tensors.Usable{},part_workers}),lane,lanes,position,KernelGemm.Both{restated,laid})

# The part written into the cells of a target along another axis than its
# lane axis.
def written_kept(~Logical: Data,~original: Logical -> U32 -> F32,~Part: Data,~part_value: Part -> U32 -> F32,+logical: Logical,+part: Part,+values: List<&2,U32>,+part_values: List<&2,U32>,
  +axis: U32,+start: U32,pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+lane: Nat,+lanes: U32,+position: Grid.Position,part_tiled: Tiled.Tiled,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: {Layout.slice_fits(values,part_values,axis,start,0) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(part_values)) == True{} : Bool},
  +apart: {Nat.is_eq(U32.to_nat(axis),lane) == False{} : Bool},
  +target_contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position}),
  +part_contract: Public.Contract(~Part,~part_value,part,part_values,part_tiled)) ->
  Public.Contract(~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),written_values(~Logical,~Part,logical,part,values,part_values,axis,start),values,TiledLayout.written_cells(U32.to_nat(axis),start,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position},part_tiled)):
  KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}} = target_contract
  +ranked = fits_length(values,part_values,U32.to_nat(axis),start,marked_of(limits,values,part_values,axis,start,rank_room,fits))
  +part_inside = {%Equal.sym(Nat,List.length(&2,U32,part_values),List.length(&2,U32,values),ranked) : {Nat.is_le(lane,_) == True{} : Bool} inside :
    {Nat.is_le(lane,List.length(&2,U32,part_values)) == True{} : Bool}}
  +part_rank_room = {%Equal.sym(Nat,List.length(&2,U32,part_values),List.length(&2,U32,values),ranked) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} rank_room :
    {Nat.is_le(List.length(&2,U32,part_values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  cells_written(~Logical,~original,~Part,~part_value,logical,part,values,part_values,axis,start,pieces,shape,fill,workers,lane,lanes,position,part_tiled,
    limits,default_capacity,rank_room,fits,nonempty,apart,Global.usable(~Tile.Tile8,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},{==}),target_contract,
    Relane.arranged_cells(~Part,~part_value,part,part_values,part_tiled,lane,position,limits,default_capacity,part_inside,nonempty,part_rank_room,part_contract))

# The longest other axis lies within a bound past the axes from an index.
def longest_within(shape: List<&2,U32>,+skip: Nat,+index: Nat,+best: Nat,+extent: U32,+bound: Nat,
  +room: {Nat.is_le(Nat.add(index,List.length(&2,U32,shape)),bound) == True{} : Bool},+best_within: {Nat.is_le(best,bound) == True{} : Bool}) ->
  {Nat.is_le(Tiled.longest_other(shape,skip,index,best,extent),bound) == True{} : Bool}:
  match shape:
    case Nil{}: best_within
    case Con{+axis_extent,+rest}:
      +longer = U32.is_le(extent,axis_extent) && Bool.not(Nat.is_eq(index,skip))
      longest_within(rest,skip,1n+index,Bool.pick(Nat,longer,index,best),Bool.pick(U32,longer,axis_extent,extent),bound,
        Order.advance_room(index,List.length(&2,U32,rest),bound,room),
        Public.axis_picked(longer,index,best,bound,Order.transitive(index,Nat.add(index,1n+List.length(&2,U32,rest)),bound,Order.le_add(index,1n+List.length(&2,U32,rest)),room),best_within))

def picked_apart(wide: Bool,hit: Bool,+index: Nat,+best: Nat,+skip: Nat,+classified: {Nat.is_eq(index,skip) == hit : Bool},+apart: {Nat.is_eq(best,skip) == False{} : Bool}) ->
  {Nat.is_eq(Bool.pick(Nat,wide && Bool.not(hit),index,best),skip) == False{} : Bool}:
  match wide hit:
    case True{} False{}: classified
    case True{} True{}: apart
    case False{} +hit: apart

# The longest other axis is not the one it skips.
def longest_apart(shape: List<&2,U32>,+skip: Nat,+index: Nat,+best: Nat,+extent: U32,+apart: {Nat.is_eq(best,skip) == False{} : Bool}) ->
  {Nat.is_eq(Tiled.longest_other(shape,skip,index,best,extent),skip) == False{} : Bool}:
  match shape:
    case Nil{}: apart
    case Con{+axis_extent,+rest}:
      +longer = U32.is_le(extent,axis_extent) && Bool.not(Nat.is_eq(index,skip))
      longest_apart(rest,skip,1n+index,Bool.pick(Nat,longer,index,best),Bool.pick(U32,longer,axis_extent,extent),
        picked_apart(U32.is_le(extent,axis_extent),Nat.is_eq(index,skip),index,best,skip,{==},apart))

# ---- The part written along the lane axis ----

# A source placed in a target holds at most the values of the target.
def placed_elements(target: List<&2,U32>,source: List<&2,U32>,shifts: List<&2,U32>,+within: Facts.PlacedIn(target,source,shifts)) ->
  {Nat.is_le(ShapeLaws.elements(source),ShapeLaws.elements(target)) == True{} : Bool}:
  match target source shifts:
    case Con{+total,+target} Con{+bound,+source} Con{+shift,+shifts}:
      Held.Both{+fit,+later} = within
      +bound_le = Order.transitive(U32.to_nat(bound),Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(total),Order.right_le(U32.to_nat(shift),U32.to_nat(bound)),fit)
      Order.transitive(Nat.mul(U32.to_nat(bound),ShapeLaws.elements(source)),Nat.mul(U32.to_nat(total),ShapeLaws.elements(source)),Nat.mul(U32.to_nat(total),ShapeLaws.elements(target)),
        RowMajor.mul_le(U32.to_nat(bound),U32.to_nat(total),ShapeLaws.elements(source),bound_le),
        RowMajor.mul_le_left(U32.to_nat(total),ShapeLaws.elements(source),ShapeLaws.elements(target),placed_elements(target,source,shifts,later)))
    case Nil{} Nil{} Nil{}: {==}
    case Con{+total,+target} Nil{} +shifts: Empty.absurd({Nat.is_le(ShapeLaws.elements(Nil{}),ShapeLaws.elements(Con{total,target})) == True{} : Bool},within)
    case Con{+total,+target} Con{+bound,+source} Nil{}: Empty.absurd({Nat.is_le(ShapeLaws.elements(Con{bound,source}),ShapeLaws.elements(Con{total,target})) == True{} : Bool},within)
    case Nil{} Con{+bound,+source} +shifts: Empty.absurd({Nat.is_le(ShapeLaws.elements(Con{bound,source}),ShapeLaws.elements(Nil{})) == True{} : Bool},within)
    case Nil{} Nil{} Con{+shift,+shifts}: Empty.absurd({Nat.is_le(ShapeLaws.elements(Nil{}),ShapeLaws.elements(Nil{})) == True{} : Bool},within)

# The other digits of a spot lie inside a part of the other target extents
# placed at a mark, so the part holds the channels from start below its
# lanes.
def marked_inside(+part_values: List<&2,U32>,+values: List<&2,U32>,+mark: Nat,+start: U32,+spot: List<&2,Nat>,+channel: Nat,
  +same: {Grid.removed(part_values,mark) == Grid.removed(values,mark) : List<&2,U32>},+within: {Nat.is_lt(mark,List.length(&2,U32,values)) == True{} : Bool},
  +below: Coordinates.DigitsBelow(Grid.removed(values,mark),spot)) ->
  {Placing.inside(Grid.removed(part_values,mark),Grid.removed(marked(List.length(&2,U32,values),mark,start),mark),spot) &&
    Placing.axis_inside(Grid.lanes_at(part_values,mark),Grid.lanes_at(marked(List.length(&2,U32,values),mark,start),mark),channel) ==
    LanesCell.in_part(start,Grid.lanes_at(part_values,mark),channel) : Bool}:
  +others = List.length(&2,U32,Grid.removed(values,mark))
  +rank_eq = Public.removed_length(values,mark,within)
  +inside = {%Equal.sym(Nat,1n+others,List.length(&2,U32,values),rank_eq) : {Nat.is_lt(mark,_) == True{} : Bool} within : {Nat.is_lt(mark,1n+others) == True{} : Bool}}
  %rank_eq : {Placing.inside(Grid.removed(part_values,mark),Grid.removed(marked(_,mark,start),mark),spot) &&
    Placing.axis_inside(Grid.lanes_at(part_values,mark),Grid.lanes_at(marked(_,mark,start),mark),channel) == LanesCell.in_part(start,Grid.lanes_at(part_values,mark),channel) : Bool}
  %Equal.sym(List<&2,U32>,Grid.removed(marked(1n+others,mark,start),mark),zeros(others),marked_removed(others,mark,start,inside)) :
    {Placing.inside(Grid.removed(part_values,mark),_,spot) &&
      Placing.axis_inside(Grid.lanes_at(part_values,mark),Grid.lanes_at(marked(1n+others,mark,start),mark),channel) == LanesCell.in_part(start,Grid.lanes_at(part_values,mark),channel) : Bool}
  %Equal.sym(U32,Grid.lanes_at(marked(1n+others,mark,start),mark),start,marked_lane(1n+others,mark,start,inside)) :
    {Placing.inside(Grid.removed(part_values,mark),zeros(others),spot) &&
      Placing.axis_inside(Grid.lanes_at(part_values,mark),_,channel) == LanesCell.in_part(start,Grid.lanes_at(part_values,mark),channel) : Bool}
  %Equal.sym(List<&2,U32>,Grid.removed(part_values,mark),Grid.removed(values,mark),same) :
    {Placing.inside(_,zeros(others),spot) && Placing.axis_inside(Grid.lanes_at(part_values,mark),start,channel) == LanesCell.in_part(start,Grid.lanes_at(part_values,mark),channel) : Bool}
  %Equal.sym(Bool,Placing.inside(Grid.removed(values,mark),zeros(others),spot),True{},inside_zeros(Grid.removed(values,mark),spot,below)) :
    {_ && Placing.axis_inside(Grid.lanes_at(part_values,mark),start,channel) == LanesCell.in_part(start,Grid.lanes_at(part_values,mark),channel) : Bool}
  {==}

# The part value a spot reads at a channel through a part placed at a mark.
def marked_read(~Part: Data,~part_value: Part -> U32 -> F32,+part: Part,+part_values: List<&2,U32>,+values: List<&2,U32>,+mark: Nat,+start: U32,+spot: List<&2,Nat>,+channel: Nat,
  +within: {Nat.is_lt(mark,List.length(&2,U32,values)) == True{} : Bool},+below: Coordinates.DigitsBelow(Grid.removed(values,mark),spot)) ->
  {Placing.source_value(~Part,~part_value,part,part_values,marked(List.length(&2,U32,values),mark,start),mark,spot,channel) ==
    part_value(part,U32.from_nat(ShapeLaws.digits_value(part_values,Coordinates.inserted_digit(mark,spot,Nat.sub(channel,U32.to_nat(start)))))) : F32}:
  +others = List.length(&2,U32,Grid.removed(values,mark))
  +rank_eq = Public.removed_length(values,mark,within)
  +inside = {%Equal.sym(Nat,1n+others,List.length(&2,U32,values),rank_eq) : {Nat.is_lt(mark,_) == True{} : Bool} within : {Nat.is_lt(mark,1n+others) == True{} : Bool}}
  +read = part_value(part,U32.from_nat(ShapeLaws.digits_value(part_values,Coordinates.inserted_digit(mark,spot,Nat.sub(channel,U32.to_nat(start))))))
  %rank_eq : {Placing.source_value(~Part,~part_value,part,part_values,marked(_,mark,start),mark,spot,channel) == read : F32}
  %Equal.sym(List<&2,U32>,Grid.removed(marked(1n+others,mark,start),mark),zeros(others),marked_removed(others,mark,start,inside)) :
    {part_value(part,U32.from_nat(ShapeLaws.digits_value(part_values,Coordinates.inserted_digit(mark,Placing.shifted(_,spot),
      Nat.sub(channel,U32.to_nat(Grid.lanes_at(marked(1n+others,mark,start),mark))))))) == read : F32}
  %Equal.sym(U32,Grid.lanes_at(marked(1n+others,mark,start),mark),start,marked_lane(1n+others,mark,start,inside)) :
    {part_value(part,U32.from_nat(ShapeLaws.digits_value(part_values,Coordinates.inserted_digit(mark,Placing.shifted(zeros(others),spot),Nat.sub(channel,U32.to_nat(_)))))) == read : F32}
  %Equal.sym(List<&2,Nat>,Placing.shifted(zeros(others),spot),spot,shifted_zeros(others,spot,RelaneValues.below_length(Grid.removed(values,mark),spot,below))) :
    {part_value(part,U32.from_nat(ShapeLaws.digits_value(part_values,Coordinates.inserted_digit(mark,_,Nat.sub(channel,U32.to_nat(start)))))) == read : F32}
  {==}

# The target placed in itself from zero holds its value at the digits.
def zero_read(~Logical: Data,~original: Logical -> U32 -> F32,~Part: Data,~part_value: Part -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+digits: List<&2,Nat>,+limits: Machine.Limits,
  +room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+below: Coordinates.DigitsBelow(values,digits)) ->
  {Placements.placed_result(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value),Written.Before{logical},Placements.segments_cell(Layout.placed_segments(values,Layout.shifts_at(List.length(&2,U32,values),0,0,0)),
      Grid.coordinates(values,U32.from_nat(ShapeLaws.digits_value(values,digits))),Tasks.strides_of(values)),0.0) ==
    original(logical,U32.from_nat(ShapeLaws.digits_value(values,digits))) : F32}:
  +rank = List.length(&2,U32,values)
  +whole = Layout.shifts_at(rank,0,0,0)
  +number = ShapeLaws.digits_value(values,digits)
  +words = Grid.coordinates(values,U32.from_nat(number))
  +cap_ok = Limits.half_word_fits(limits)
  +value_room = placed_room(values,values,whole,limits,Written.whole_in(values,0),room,positive)
  +words_digits = Placing.digits_words(values,digits,Machine.half_word(limits),cap_ok,Limits.double_bound(limits,ShapeLaws.elements(values),room),below)
  +count = Digits.coordinates_count(values,U32.from_nat(number))
  +zero = shifts_zero(rank,0)
  +digits_count = {%words_digits : {List.length(&2,Nat,_) == rank : Nat}
    Equal.trans(Nat,List.length(&2,Nat,Digits.words_nat(words)),List.length(&2,U32,words),rank,Placing.words_length(words),count) : {List.length(&2,Nat,digits) == rank : Nat}}
  +inside_ok = {%Equal.sym(List<&2,U32>,whole,zeros(rank),zero) : {Placing.inside(values,_,Digits.words_nat(words)) == True{} : Bool}
    {%Equal.sym(List<&2,Nat>,Digits.words_nat(words),digits,words_digits) : {Placing.inside(values,zeros(rank),_) == True{} : Bool}
      inside_zeros(values,digits,below) : {Placing.inside(values,zeros(rank),Digits.words_nat(words)) == True{} : Bool}} :
    {Placing.inside(values,whole,Digits.words_nat(words)) == True{} : Bool}}
  +inside_words_ok = Equal.trans(Bool,Placing.inside_words(values,whole,words),Placing.inside(values,whole,Digits.words_nat(words)),True{},
    Placing.inside_nat(values,whole,words,Machine.half_word(limits),cap_ok,value_room),inside_ok)
  +offset = Placing.offset(values,whole,words)
  +offset_exact = Placing.offset_nat(values,whole,words,Machine.half_word(limits),cap_ok,value_room,count,half_room(limits,ShapeLaws.elements(values),room),inside_ok)
  +offset_number = {%shifted_zeros(rank,digits,digits_count) : {U32.to_nat(offset) == ShapeLaws.digits_value(values,_) : Nat}
    {%words_digits : {U32.to_nat(offset) == ShapeLaws.digits_value(values,Placing.shifted(zeros(rank),_)) : Nat}
      {%zero : {U32.to_nat(offset) == ShapeLaws.digits_value(values,Placing.shifted(_,Digits.words_nat(words))) : Nat}
        offset_exact : {U32.to_nat(offset) == ShapeLaws.digits_value(values,Placing.shifted(zeros(rank),Digits.words_nat(words))) : Nat}} :
      {U32.to_nat(offset) == ShapeLaws.digits_value(values,Placing.shifted(zeros(rank),digits)) : Nat}} :
    {U32.to_nat(offset) == number : Nat}}
  %Equal.sym(Maybe<&2,U32>,Placements.segments_cell(Layout.placed_segments(values,whole),words,Tasks.strides_of(values)),
      Bool.pick(Maybe<&2,U32>,Placing.inside_words(values,whole,words),Some{offset},None{}),Placing.segments_words(values,whole,words)) :
    {Placements.placed_result(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value),Written.Before{logical},_,0.0) == original(logical,U32.from_nat(number)) : F32}
  %Equal.sym(Bool,Placing.inside_words(values,whole,words),True{},inside_words_ok) :
    {Placements.placed_result(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value),Written.Before{logical},Bool.pick(Maybe<&2,U32>,_,Some{offset},None{}),0.0) == original(logical,U32.from_nat(number)) : F32}
  Equal.cong(U32,F32,at => original(logical,at),offset,U32.from_nat(number),
    Equal.trans(U32,offset,U32.from_nat(U32.to_nat(offset)),U32.from_nat(number),Equal.sym(U32,U32.from_nat(U32.to_nat(offset)),offset,Arithmetic.native_roundtrip(offset)),
      Equal.cong(Nat,U32,U32.from_nat,U32.to_nat(offset),number,offset_number)))

# A part placed where the choice holds gives the value it reads there and
# keeps the old value elsewhere.
def placed_over(~Logical: Data,~original: Logical -> U32 -> F32,~Part: Data,~part_value: Part -> U32 -> F32,+part: Part,inside: Bool,+offset: U32,+old: F32,+read: F32,
  value: @+hit: {inside == True{} : Bool} -> {part_value(part,offset) == read : F32}) ->
  {Placements.placed_result(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value),Written.Over{part},Bool.pick(Maybe<&2,U32>,inside,Some{offset},None{}),old) == Bool.pick(F32,inside,read,old) : F32}:
  match inside:
    case True{}: value({==})
    case False{}: {==}

# The part placed from start along the axis gives a spot inside the target
# its value at a channel in the part and keeps the old value elsewhere.
def part_read(~Logical: Data,~original: Logical -> U32 -> F32,~Part: Data,~part_value: Part -> U32 -> F32,+part: Part,+values: List<&2,U32>,+part_values: List<&2,U32>,+axis: U32,+start: U32,+limits: Machine.Limits,+room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+part_room: {Nat.is_le(ShapeLaws.elements(part_values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: {Layout.slice_fits(values,part_values,axis,start,0) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +spot: List<&2,Nat>,+spot_below: Coordinates.DigitsBelow(Grid.removed(values,U32.to_nat(axis)),spot),+group: Nat,+lane: Nat,+hit: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,U32.to_nat(axis)))) == True{} : Bool},+old: F32) ->
  {Placements.placed_result(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value),Written.Over{part},Placements.segments_cell(Layout.placed_segments(part_values,Layout.shifts_at(List.length(&2,U32,values),axis,start,0)),
      Grid.coordinates(values,U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(U32.to_nat(axis),spot,Nat.add(lane,Nat.mul(group,8n)))))),Tasks.strides_of(part_values)),old) ==
    Bool.pick(F32,LanesCell.in_part(start,Grid.lanes_at(part_values,U32.to_nat(axis)),Nat.add(lane,Nat.mul(group,8n))),part_value(part,U32.from_nat(ShapeLaws.digits_value(part_values,Coordinates.inserted_digit(U32.to_nat(axis),spot,Nat.sub(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(start)))))),old) : F32}:
  +mark = U32.to_nat(axis)
  +rank = List.length(&2,U32,values)
  +channel = Nat.add(lane,Nat.mul(group,8n))
  +digits = Coordinates.inserted_digit(U32.to_nat(axis),spot,channel)
  +words = Grid.coordinates(values,U32.from_nat(ShapeLaws.digits_value(values,digits)))
  +cap_ok = Limits.half_word_fits(limits)
  +marked_fits = marked_of(limits,values,part_values,axis,start,rank_room,fits)
  +ranked = fits_length(values,part_values,mark,start,marked_fits)
  +within = fits_inside(values,part_values,mark,start,marked_fits)
  +same = same_listed(Grid.removed(values,mark),Grid.removed(part_values,mark),fits_removed_hit(values,part_values,mark,start,marked_fits))
  +part_within = {%Equal.sym(Nat,List.length(&2,U32,part_values),rank,ranked) : {Nat.is_lt(mark,_) == True{} : Bool} within : {Nat.is_lt(mark,List.length(&2,U32,part_values)) == True{} : Bool}}
  +shifts = Layout.shifts_at(rank,axis,start,0)
  +moved = marked(rank,mark,start)
  +marks = shifts_marked(rank,axis,start,0n,mark,Machine.capacity(limits),Limits.capacity_fits(limits),rank_room,{==})
  +below = RelaneValues.inserted_below(values,mark,spot,channel,spot_below,hit,within)
  +words_digits = Placing.digits_words(values,digits,Machine.half_word(limits),cap_ok,Limits.double_bound(limits,ShapeLaws.elements(values),room),below)
  +read = {Placing.Read{words_digits,Digits.coordinates_count(values,U32.from_nat(ShapeLaws.digits_value(values,digits))),hit} :
    Placing.Read<Grid.lanes_at(values,U32.to_nat(axis)),rank,mark,spot,group,lane,words>}
  +others_same = Equal.trans(Nat,1n+List.length(&2,Nat,spot),1n+List.length(&2,U32,Grid.removed(values,U32.to_nat(axis))),rank,
    Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Nat,spot),List.length(&2,U32,Grid.removed(values,U32.to_nat(axis))),RelaneValues.below_length(Grid.removed(values,U32.to_nat(axis)),spot,spot_below)),
    Public.removed_length(values,mark,within))
  +moved_room = {%marks : Placing.Room(part_values,_,Machine.half_word(limits))
    placed_room(values,part_values,shifts,limits,Written.slice_in(values,part_values,axis,start,0,fits),room,positive) : Placing.Room(part_values,moved,Machine.half_word(limits))}
  +shifts_same = {%marks : {List.length(&2,U32,_) == List.length(&2,U32,part_values) : Nat}
    Equal.trans(Nat,List.length(&2,U32,shifts),rank,List.length(&2,U32,part_values),shifts_length(rank,axis,start,0),Equal.sym(Nat,List.length(&2,U32,part_values),rank,ranked)) :
    {List.length(&2,U32,moved) == List.length(&2,U32,part_values) : Nat}}
  +facts = {Placing.ValueFacts{cap_ok,part_within,ranked,shifts_same,moved_room,half_room(limits,ShapeLaws.elements(part_values),part_room),others_same} :
    Placing.ValueFacts<Machine.half_word(limits),rank,part_values,moved,mark,spot>}
  +kept = Placing.inside(Grid.removed(part_values,mark),Grid.removed(moved,mark),spot) && Placing.axis_inside(Grid.lanes_at(part_values,mark),Grid.lanes_at(moved,mark),channel)
  +along = marked_inside(part_values,values,mark,start,spot,channel,same,within,spot_below)
  +inside_eq = Equal.trans(Bool,Placing.inside_words(part_values,moved,words),kept,LanesCell.in_part(start,Grid.lanes_at(part_values,U32.to_nat(axis)),channel),
    Placing.value_inside(Machine.half_word(limits),Grid.lanes_at(values,U32.to_nat(axis)),rank,part_values,moved,mark,spot,group,lane,words,facts,read),along)
  +offset = Placing.offset(part_values,moved,words)
  %Equal.sym(List<&2,U32>,shifts,moved,marks) :
    {Placements.placed_result(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value),Written.Over{part},Placements.segments_cell(Layout.placed_segments(part_values,_),words,Tasks.strides_of(part_values)),old) ==
      Bool.pick(F32,LanesCell.in_part(start,Grid.lanes_at(part_values,U32.to_nat(axis)),channel),part_value(part,U32.from_nat(ShapeLaws.digits_value(part_values,Coordinates.inserted_digit(U32.to_nat(axis),spot,Nat.sub(channel,U32.to_nat(start)))))),old) : F32}
  %Equal.sym(Maybe<&2,U32>,Placements.segments_cell(Layout.placed_segments(part_values,moved),words,Tasks.strides_of(part_values)),
      Bool.pick(Maybe<&2,U32>,Placing.inside_words(part_values,moved,words),Some{offset},None{}),Placing.segments_words(part_values,moved,words)) :
    {Placements.placed_result(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value),Written.Over{part},_,old) == Bool.pick(F32,LanesCell.in_part(start,Grid.lanes_at(part_values,U32.to_nat(axis)),channel),part_value(part,U32.from_nat(ShapeLaws.digits_value(part_values,Coordinates.inserted_digit(U32.to_nat(axis),spot,Nat.sub(channel,U32.to_nat(start)))))),old) : F32}
  %Equal.sym(Bool,Placing.inside_words(part_values,moved,words),LanesCell.in_part(start,Grid.lanes_at(part_values,U32.to_nat(axis)),channel),inside_eq) :
    {Placements.placed_result(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value),Written.Over{part},Bool.pick(Maybe<&2,U32>,_,Some{offset},None{}),old) == Bool.pick(F32,LanesCell.in_part(start,Grid.lanes_at(part_values,U32.to_nat(axis)),channel),part_value(part,U32.from_nat(ShapeLaws.digits_value(part_values,Coordinates.inserted_digit(U32.to_nat(axis),spot,Nat.sub(channel,U32.to_nat(start)))))),old) : F32}
  placed_over(~Logical,~original,~Part,~part_value,part,LanesCell.in_part(start,Grid.lanes_at(part_values,U32.to_nat(axis)),channel),offset,old,part_value(part,U32.from_nat(ShapeLaws.digits_value(part_values,Coordinates.inserted_digit(U32.to_nat(axis),spot,Nat.sub(channel,U32.to_nat(start)))))),
    lies => Equal.trans(F32,part_value(part,offset),Placing.source_value(~Part,~part_value,part,part_values,moved,mark,spot,channel),part_value(part,U32.from_nat(ShapeLaws.digits_value(part_values,Coordinates.inserted_digit(U32.to_nat(axis),spot,Nat.sub(channel,U32.to_nat(start)))))),
      Placing.value_value(~Part,~part_value,part,Machine.half_word(limits),Grid.lanes_at(values,U32.to_nat(axis)),rank,part_values,moved,mark,spot,group,lane,words,facts,read,Equal.trans(Bool,kept,LanesCell.in_part(start,Grid.lanes_at(part_values,U32.to_nat(axis)),channel),True{},along,lies)),
      marked_read(~Part,~part_value,part,part_values,values,mark,start,spot,channel,within,spot_below)))

# The written values at the digits of a spot and a channel inside the target
# lanes: the part value where the channel lies in the part and the target
# value elsewhere.
def written_spec(~Logical: Data,~original: Logical -> U32 -> F32,~Part: Data,~part_value: Part -> U32 -> F32,+logical: Logical,+part: Part,+values: List<&2,U32>,+part_values: List<&2,U32>,+axis: U32,+start: U32,+limits: Machine.Limits,+room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+part_room: {Nat.is_le(ShapeLaws.elements(part_values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: {Layout.slice_fits(values,part_values,axis,start,0) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +spot: List<&2,Nat>,+spot_below: Coordinates.DigitsBelow(Grid.removed(values,U32.to_nat(axis)),spot),+group: Nat,+lane: Nat,+hit: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,U32.to_nat(axis)))) == True{} : Bool}) ->
  {Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value),written_values(~Logical,~Part,logical,part,values,part_values,axis,start),U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(U32.to_nat(axis),spot,Nat.add(lane,Nat.mul(group,8n)))))) == Bool.pick(F32,LanesCell.in_part(start,Grid.lanes_at(part_values,U32.to_nat(axis)),Nat.add(lane,Nat.mul(group,8n))),part_value(part,U32.from_nat(ShapeLaws.digits_value(part_values,Coordinates.inserted_digit(U32.to_nat(axis),spot,Nat.sub(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(start)))))),original(logical,U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(U32.to_nat(axis),spot,Nat.add(lane,Nat.mul(group,8n))))))) : F32}:
  +digits = Coordinates.inserted_digit(U32.to_nat(axis),spot,Nat.add(lane,Nat.mul(group,8n)))
  +words = Grid.coordinates(values,U32.from_nat(ShapeLaws.digits_value(values,digits)))
  +part_cell = Placements.segments_cell(Layout.placed_segments(part_values,Layout.shifts_at(List.length(&2,U32,values),axis,start,0)),words,Tasks.strides_of(part_values))
  +target_read = Placements.placed_result(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value),Written.Before{logical},Placements.segments_cell(Layout.placed_segments(values,Layout.shifts_at(List.length(&2,U32,values),0,0,0)),words,Tasks.strides_of(values)),0.0)
  +within = fits_inside(values,part_values,U32.to_nat(axis),start,marked_of(limits,values,part_values,axis,start,rank_room,fits))
  Equal.trans(F32,Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value),written_values(~Logical,~Part,logical,part,values,part_values,axis,start),U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(U32.to_nat(axis),spot,Nat.add(lane,Nat.mul(group,8n)))))),Placements.placed_result(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value),Written.Over{part},part_cell,original(logical,U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(U32.to_nat(axis),spot,Nat.add(lane,Nat.mul(group,8n))))))),Bool.pick(F32,LanesCell.in_part(start,Grid.lanes_at(part_values,U32.to_nat(axis)),Nat.add(lane,Nat.mul(group,8n))),part_value(part,U32.from_nat(ShapeLaws.digits_value(part_values,Coordinates.inserted_digit(U32.to_nat(axis),spot,Nat.sub(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(start)))))),original(logical,U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(U32.to_nat(axis),spot,Nat.add(lane,Nat.mul(group,8n))))))),
    Equal.cong(F32,F32,old => Placements.placed_result(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value),Written.Over{part},part_cell,old),target_read,original(logical,U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(U32.to_nat(axis),spot,Nat.add(lane,Nat.mul(group,8n)))))),
      zero_read(~Logical,~original,~Part,~part_value,logical,values,digits,limits,room,positive,RelaneValues.inserted_below(values,U32.to_nat(axis),spot,Nat.add(lane,Nat.mul(group,8n)),spot_below,hit,within))),
    part_read(~Logical,~original,~Part,~part_value,part,values,part_values,axis,start,limits,room,positive,part_room,fits,rank_room,spot,spot_below,group,lane,hit,original(logical,U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(U32.to_nat(axis),spot,Nat.add(lane,Nat.mul(group,8n))))))))

# A written cell at a spot and group holds the cell of the written values.
def cell_agrees(~Logical: Data,~original: Logical -> U32 -> F32,~Part: Data,~part_value: Part -> U32 -> F32,+logical: Logical,+part: Part,+values: List<&2,U32>,+part_values: List<&2,U32>,+axis: U32,+start: U32,+position: Grid.Position,
  +limits: Machine.Limits,+room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+part_room: {Nat.is_le(ShapeLaws.elements(part_values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: {Layout.slice_fits(values,part_values,axis,start,0) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+count: U32,+spot: List<&2,Nat>,+group: Nat,+word: U32,+spot_below: Coordinates.DigitsBelow(Grid.removed(values,U32.to_nat(axis)),spot),
  +group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(values,U32.to_nat(axis))))) == True{} : Bool},+exact: {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(values,U32.to_nat(axis),position),Coordinates.cell_digits(position,spot,group)) : Nat},
  +shape: LanesWritten.WriteShape<limits,values,part_values,U32.to_nat(axis),position,start,count>) -> {Law.written_value(~Logical,~original,~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value),Law.Written{Representation.Lifted{logical,Public.geometry_of(values,U32.to_nat(axis),position)},Representation.Lifted{part,Public.geometry_of(part_values,U32.to_nat(axis),position)},Tiled.Write{start,Grid.lanes_at(part_values,U32.to_nat(axis)),Grid.groups(Grid.lanes_at(values,U32.to_nat(axis))),Grid.groups(Grid.lanes_at(part_values,U32.to_nat(axis))),count,position}},word) == Representation.lifted(~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),Representation.Lifted{written_values(~Logical,~Part,logical,part,values,part_values,axis,start),Public.geometry_of(values,U32.to_nat(axis),position)},word) : Tile.Tile8}:
  TileLaws.tiles_equal(Law.written_value(~Logical,~original,~Representation.Lifted<Part>,~Representation.lifted(~Part,~part_value),Law.Written{Representation.Lifted{logical,Public.geometry_of(values,U32.to_nat(axis),position)},Representation.Lifted{part,Public.geometry_of(part_values,U32.to_nat(axis),position)},Tiled.Write{start,Grid.lanes_at(part_values,U32.to_nat(axis)),Grid.groups(Grid.lanes_at(values,U32.to_nat(axis))),Grid.groups(Grid.lanes_at(part_values,U32.to_nat(axis))),count,position}},word),
    Representation.lifted(~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),Representation.Lifted{written_values(~Logical,~Part,logical,part,values,part_values,axis,start),Public.geometry_of(values,U32.to_nat(axis),position)},word),
    LanesWritten.lane_agrees(~Logical,~original,~Part,~part_value,~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),logical,part,written_values(~Logical,~Part,logical,part,values,part_values,axis,start),limits,values,part_values,U32.to_nat(axis),position,start,count,shape,spot,group,word,spot_below,group_inside,exact,0n,{==},hit => written_spec(~Logical,~original,~Part,~part_value,logical,part,values,part_values,axis,start,limits,room,positive,part_room,fits,rank_room,spot,spot_below,group,0n,hit)),LanesWritten.lane_agrees(~Logical,~original,~Part,~part_value,~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),logical,part,written_values(~Logical,~Part,logical,part,values,part_values,axis,start),limits,values,part_values,U32.to_nat(axis),position,start,count,shape,spot,group,word,spot_below,group_inside,exact,1n,{==},hit => written_spec(~Logical,~original,~Part,~part_value,logical,part,values,part_values,axis,start,limits,room,positive,part_room,fits,rank_room,spot,spot_below,group,1n,hit)),LanesWritten.lane_agrees(~Logical,~original,~Part,~part_value,~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),logical,part,written_values(~Logical,~Part,logical,part,values,part_values,axis,start),limits,values,part_values,U32.to_nat(axis),position,start,count,shape,spot,group,word,spot_below,group_inside,exact,2n,{==},hit => written_spec(~Logical,~original,~Part,~part_value,logical,part,values,part_values,axis,start,limits,room,positive,part_room,fits,rank_room,spot,spot_below,group,2n,hit)),LanesWritten.lane_agrees(~Logical,~original,~Part,~part_value,~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),logical,part,written_values(~Logical,~Part,logical,part,values,part_values,axis,start),limits,values,part_values,U32.to_nat(axis),position,start,count,shape,spot,group,word,spot_below,group_inside,exact,3n,{==},hit => written_spec(~Logical,~original,~Part,~part_value,logical,part,values,part_values,axis,start,limits,room,positive,part_room,fits,rank_room,spot,spot_below,group,3n,hit)),LanesWritten.lane_agrees(~Logical,~original,~Part,~part_value,~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),logical,part,written_values(~Logical,~Part,logical,part,values,part_values,axis,start),limits,values,part_values,U32.to_nat(axis),position,start,count,shape,spot,group,word,spot_below,group_inside,exact,4n,{==},hit => written_spec(~Logical,~original,~Part,~part_value,logical,part,values,part_values,axis,start,limits,room,positive,part_room,fits,rank_room,spot,spot_below,group,4n,hit)),LanesWritten.lane_agrees(~Logical,~original,~Part,~part_value,~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),logical,part,written_values(~Logical,~Part,logical,part,values,part_values,axis,start),limits,values,part_values,U32.to_nat(axis),position,start,count,shape,spot,group,word,spot_below,group_inside,exact,5n,{==},hit => written_spec(~Logical,~original,~Part,~part_value,logical,part,values,part_values,axis,start,limits,room,positive,part_room,fits,rank_room,spot,spot_below,group,5n,hit)),LanesWritten.lane_agrees(~Logical,~original,~Part,~part_value,~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),logical,part,written_values(~Logical,~Part,logical,part,values,part_values,axis,start),limits,values,part_values,U32.to_nat(axis),position,start,count,shape,spot,group,word,spot_below,group_inside,exact,6n,{==},hit => written_spec(~Logical,~original,~Part,~part_value,logical,part,values,part_values,axis,start,limits,room,positive,part_room,fits,rank_room,spot,spot_below,group,6n,hit)),LanesWritten.lane_agrees(~Logical,~original,~Part,~part_value,~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),logical,part,written_values(~Logical,~Part,logical,part,values,part_values,axis,start),limits,values,part_values,U32.to_nat(axis),position,start,count,shape,spot,group,word,spot_below,group_inside,exact,7n,{==},hit => written_spec(~Logical,~original,~Part,~part_value,logical,part,values,part_values,axis,start,limits,room,positive,part_room,fits,rank_room,spot,spot_below,group,7n,hit)))

# The values of a target of the contract are at most the capacity.
def contract_room(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,-pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,
  +workers: U32,+lane: Nat,+lanes: U32,+position: Grid.Position,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position})) ->
  {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}} = contract
  StorageCount.counted_room(limits,values,counted,default_capacity)

# The part written into a target along its lane axis.
def written_on_lane(~Logical: Data,~original: Logical -> U32 -> F32,~Part: Data,~part_value: Part -> U32 -> F32,+logical: Logical,+part: Part,+values: List<&2,U32>,+part_values: List<&2,U32>,+axis: U32,+start: U32,
  pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+lanes: U32,+position: Grid.Position,part_tiled: Tiled.Tiled,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: {Layout.slice_fits(values,part_values,axis,start,0) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(part_values)) == True{} : Bool},
  +target_contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},U32.to_nat(axis),lanes,position}),
  +part_contract: Public.Contract(~Part,~part_value,part,part_values,part_tiled)) ->
  Public.Contract(~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),written_values(~Logical,~Part,logical,part,values,part_values,axis,start),values,Tiled.lanes_written(Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},U32.to_nat(axis),lanes,position},start,part_tiled)):
  +mark = U32.to_nat(axis)
  +shifts = Layout.shifts_at(List.length(&2,U32,values),axis,start,0)
  +marked_fits = marked_of(limits,values,part_values,axis,start,rank_room,fits)
  +ranked = fits_length(values,part_values,mark,start,marked_fits)
  +within = fits_inside(values,part_values,mark,start,marked_fits)
  +room = contract_room(~Logical,~original,logical,values,pieces,shape,fill,workers,mark,lanes,position,limits,default_capacity,target_contract)
  +placed = Written.slice_in(values,part_values,axis,start,0,fits)
  +positive = Facts.placed_positive(values,part_values,shifts,placed,nonempty)
  +part_room = Order.transitive(ShapeLaws.elements(part_values),ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits)),placed_elements(values,part_values,shifts,placed),room)
  +part_within = {%Equal.sym(Nat,List.length(&2,U32,part_values),List.length(&2,U32,values),ranked) : {Nat.is_lt(mark,_) == True{} : Bool} within : {Nat.is_lt(mark,List.length(&2,U32,part_values)) == True{} : Bool}}
  +removed_same = same_listed(Grid.removed(values,mark),Grid.removed(part_values,mark),fits_removed_hit(values,part_values,mark,start,marked_fits))
  +ends = Written.slice_axis(True{},Grid.lanes_at(values,U32.to_nat(axis)),Grid.lanes_at(part_values,U32.to_nat(axis)),start,fits_hit(values,part_values,mark,start,marked_fits))
  +part_rank_room = {%Equal.sym(Nat,List.length(&2,U32,part_values),List.length(&2,U32,values),ranked) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} rank_room :
    {Nat.is_le(List.length(&2,U32,part_values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  LanesWritten.lanes_written(~Logical,~original,~Part,~part_value,~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),logical,part,written_values(~Logical,~Part,logical,part,values,part_values,axis,start),values,part_values,mark,start,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lanes,position,
    part_tiled,limits,default_capacity,target_contract,part_contract,within,part_within,removed_same,ends,rank_room,part_rank_room,nonempty,
    spot => group => word => spot_below => group_inside => exact => written_shape =>
      cell_agrees(~Logical,~original,~Part,~part_value,logical,part,values,part_values,axis,start,position,limits,room,positive,part_room,fits,rank_room,
        (Tensors.count_or_zero(Tiled.cell_shape(values,U32.to_nat(axis),position)) / U32.max(Grid.groups(Grid.lanes_at(values,U32.to_nat(axis))),1) : U32),spot,group,word,spot_below,group_inside,exact,written_shape))

# The part written into a target along its lane axis or along another axis.
def written_lanes(~Logical: Data,~original: Logical -> U32 -> F32,~Part: Data,~part_value: Part -> U32 -> F32,on_lane: Bool,+logical: Logical,+part: Part,+values: List<&2,U32>,+part_values: List<&2,U32>,
  +axis: U32,+start: U32,pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+lane: Nat,+lanes: U32,+position: Grid.Position,part_tiled: Tiled.Tiled,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: {Layout.slice_fits(values,part_values,axis,start,0) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(part_values)) == True{} : Bool},
  +classified: {Nat.is_eq(U32.to_nat(axis),lane) == on_lane : Bool},
  +target_contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position}),
  +part_contract: Public.Contract(~Part,~part_value,part,part_values,part_tiled)) ->
  Public.Contract(~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),written_values(~Logical,~Part,logical,part,values,part_values,axis,start),values,TiledLayout.written_on(on_lane,U32.to_nat(axis),start,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position},part_tiled)):
  match on_lane:
    case True{}:
      +at = Order.equal_of_is_eq(U32.to_nat(axis),lane,classified)
      %at : Public.Contract(~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),
        written_values(~Logical,~Part,logical,part,values,part_values,axis,start),values,Tiled.lanes_written(Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},_,lanes,position},start,part_tiled))
      written_on_lane(~Logical,~original,~Part,~part_value,logical,part,values,part_values,axis,start,pieces,shape,fill,workers,lanes,position,part_tiled,
        limits,default_capacity,rank_room,fits,nonempty,
        {%Equal.sym(Nat,U32.to_nat(axis),lane,at) : Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},_,lanes,position})
          target_contract : Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},U32.to_nat(axis),lanes,position})},
        part_contract)
    case False{}:
      written_kept(~Logical,~original,~Part,~part_value,logical,part,values,part_values,axis,start,pieces,shape,fill,workers,lane,lanes,position,part_tiled,
        limits,default_capacity,rank_room,fits,nonempty,classified,target_contract,part_contract)

# ---- The written tiled values ----

# Usable cells of the target and of the part.
def both_usable(~Logical: Data,~original: Logical -> U32 -> F32,~Part: Data,~part_value: Part -> U32 -> F32,+logical: Logical,+part: Part,+values: List<&2,U32>,+part_values: List<&2,U32>,
  +axis: U32,+start: U32,pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+lane: Nat,+lanes: U32,+position: Grid.Position,
  part_pieces: Pieces.Pieces(~Tile.Tile8),+part_shape: List<&2,U32>,+part_fill: Tile.Tile8,+part_workers: U32,+part_lane: Nat,+part_lanes: U32,+part_position: Grid.Position,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: {Layout.slice_fits(values,part_values,axis,start,0) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(part_values)) == True{} : Bool},
  +target_contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position}),
  +part_contract: Public.Contract(~Part,~part_value,part,part_values,Tiled.Tiled{Tensors.Tensor{part_pieces,part_shape,part_fill,Tensors.Usable{},part_workers},part_lane,part_lanes,part_position})) ->
  Public.Contract(~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),written_values(~Logical,~Part,logical,part,values,part_values,axis,start),values,TiledLayout.set_slice(Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position},axis,start,Tiled.Tiled{Tensors.Tensor{part_pieces,part_shape,part_fill,Tensors.Usable{},part_workers},part_lane,part_lanes,part_position})):
  KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}} = target_contract
  KernelGemm.Both{+part_global,Public.Laid{+part_inside,+part_shaped,+part_restored,+part_counted,+part_pinned}} = part_contract
  +mark = U32.to_nat(axis)
  +count = U32.max(workers,part_workers)
  +target_values = Tiled.logical_shape(shape,lane,lanes,position)
  +part_logical = Tiled.logical_shape(part_shape,part_lane,part_lanes,part_position)
  %Equal.sym(List<&2,U32>,target_values,values,restored) :
    Public.Contract(~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),written_values(~Logical,~Part,logical,part,values,part_values,axis,start),values,TiledLayout.written(Layout.slice_fits(_,part_logical,axis,start,0),_,part_logical,axis,start,Tiled.with_workers(Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position},count),Tiled.with_workers(Tiled.Tiled{Tensors.Tensor{part_pieces,part_shape,part_fill,Tensors.Usable{},part_workers},part_lane,part_lanes,part_position},count)))
  %Equal.sym(List<&2,U32>,part_logical,part_values,part_restored) :
    Public.Contract(~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),written_values(~Logical,~Part,logical,part,values,part_values,axis,start),values,TiledLayout.written(Layout.slice_fits(values,_,axis,start,0),values,_,axis,start,Tiled.with_workers(Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position},count),Tiled.with_workers(Tiled.Tiled{Tensors.Tensor{part_pieces,part_shape,part_fill,Tensors.Usable{},part_workers},part_lane,part_lanes,part_position},count)))
  %Equal.sym(Bool,Layout.slice_fits(values,part_values,axis,start,0),True{},fits) :
    Public.Contract(~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),written_values(~Logical,~Part,logical,part,values,part_values,axis,start),values,TiledLayout.written(_,values,part_values,axis,start,Tiled.with_workers(Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position},count),Tiled.with_workers(Tiled.Tiled{Tensors.Tensor{part_pieces,part_shape,part_fill,Tensors.Usable{},part_workers},part_lane,part_lanes,part_position},count)))
  written_lanes(~Logical,~original,~Part,~part_value,Nat.is_eq(mark,lane),logical,part,values,part_values,axis,start,
    pieces,shape,fill,U32.max(1,count),lane,lanes,position,Tiled.with_workers(Tiled.Tiled{Tensors.Tensor{part_pieces,part_shape,part_fill,Tensors.Usable{},part_workers},part_lane,part_lanes,part_position},count),limits,default_capacity,rank_room,fits,nonempty,{==},
    target_contract,part_contract)

# Usable cells of the target.
def target_usable(~Logical: Data,~original: Logical -> U32 -> F32,~Part: Data,~part_value: Part -> U32 -> F32,+logical: Logical,+part: Part,+values: List<&2,U32>,+part_values: List<&2,U32>,
  +axis: U32,+start: U32,pieces: Pieces.Pieces(~Tile.Tile8),+shape: List<&2,U32>,+fill: Tile.Tile8,+workers: U32,+lane: Nat,+lanes: U32,+position: Grid.Position,part_tiled: Tiled.Tiled,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: {Layout.slice_fits(values,part_values,axis,start,0) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(part_values)) == True{} : Bool},
  +target_contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position}),
  +part_contract: Public.Contract(~Part,~part_value,part,part_values,part_tiled)) ->
  Public.Contract(~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),written_values(~Logical,~Part,logical,part,values,part_values,axis,start),values,TiledLayout.set_slice(Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position},axis,start,part_tiled)):
  match part_tiled:
    case Tiled.Tiled{part_cells,+part_lane,+part_lanes,+part_position}:
      match part_cells:
        case Tensors.Tensor{part_pieces,+part_shape,+part_fill,Tensors.Usable{},+part_workers}:
          both_usable(~Logical,~original,~Part,~part_value,logical,part,values,part_values,axis,start,pieces,shape,fill,workers,lane,lanes,position,
            part_pieces,part_shape,part_fill,part_workers,part_lane,part_lanes,part_position,limits,default_capacity,rank_room,fits,nonempty,target_contract,part_contract)
        case Tensors.Tensor{part_pieces,+part_shape,+part_fill,Tensors.Failed{+problem},+part_workers}:
          Empty.absurd(Public.Contract(~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),written_values(~Logical,~Part,logical,part,values,part_values,axis,start),values,TiledLayout.set_slice(Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},lane,lanes,position},axis,start,
            Tiled.Tiled{Tensors.Tensor{part_pieces,part_shape,part_fill,Tensors.Failed{problem},part_workers},part_lane,part_lanes,part_position})),part_contract)

# A part of the contract with at least one value that fits a target of the
# contract from start along axis, written into the target: the result holds
# the part where it lies and the target elsewhere.
def set_slice(~Logical: Data,~original: Logical -> U32 -> F32,~Part: Data,~part_value: Part -> U32 -> F32,+logical: Logical,+part: Part,+values: List<&2,U32>,+part_values: List<&2,U32>,
  target: Tiled.Tiled,+axis: U32,+start: U32,part_tiled: Tiled.Tiled,
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+fits: {Layout.slice_fits(values,part_values,axis,start,0) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(part_values)) == True{} : Bool},
  +target_contract: Public.Contract(~Logical,~original,logical,values,target),
  +part_contract: Public.Contract(~Part,~part_value,part,part_values,part_tiled)) ->
  Public.Contract(~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),written_values(~Logical,~Part,logical,part,values,part_values,axis,start),values,TiledLayout.set_slice(target,axis,start,part_tiled)):
  match target:
    case Tiled.Tiled{cells,+lane,+lanes,+position}:
      match cells:
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Usable{},+workers}:
          target_usable(~Logical,~original,~Part,~part_value,logical,part,values,part_values,axis,start,pieces,shape,fill,workers,lane,lanes,position,part_tiled,
            limits,default_capacity,rank_room,fits,nonempty,target_contract,part_contract)
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Failed{+problem},+workers}:
          Empty.absurd(Public.Contract(~Gathered.Placings<Written.Written<Logical,Part>,F32>,~Gathered.placing_value(~F32,~Written.Written<Logical,Part>,~Written.written_value(~F32,~Logical,~original,~Part,~part_value)),written_values(~Logical,~Part,logical,part,values,part_values,axis,start),values,TiledLayout.set_slice(Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers},lane,lanes,position},axis,start,part_tiled)),target_contract)
