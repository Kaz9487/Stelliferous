# The passes of batch normalization on cells whose lanes are the channels,
# the groups innermost: every piece is read group by group, each group a loop
# over its cells inner apart, and cell first + index of a piece ends with the
# parameters of group (first + index) % groups.
import Base
import ../kernel_batch_norm.bend as Kernel
import ../kernel_gemm.bend as KernelGemm
import ../kernel_tile.bend as Tile
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./u32_comparison.bend as Comparison
import ./word_encoding.bend as WordEncoding
import ./tensor_line_chain_values.bend as LineChainValues
import ./traversal_update_readback.bend as Updates
import ../tensor_batch_norm_cells.bend as BNCells
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./traversal_array_refinement.bend as ArrayLaws
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./tensor_placed.bend as Placed
import ./tensor_padded_pieces.bend as PaddedPieces
import ./tensor_tiled_padding.bend as Padding
import ../tensor_tiled_map.bend as TiledMap
import ./tensor_pieces_walk.bend as Walk
import ./storage_buffer_model.bend as Buffer
import ./storage_refinement.bend as StorageLaws
import ./storage_certificate.bend as Certificate
import ./storage_witness.bend as Witness
import ./traversal_certificate.bend as TraversalCertificate
import ./kernel_batch_norm_certificate.bend as KernelCertificate
import ../storage_buffer.bend as Storage
import ./tensor_pieces_content.bend as Content
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges
import ./u32_full_range.bend as FullRange
import ./tensor_pieces_model.bend as Model
import ./tensor_pieces_each.bend as Each
import ../tensor.bend as Tensors
import ../tensor_pieces.bend as Pieces
import ../traversal_partition.bend as Partition
import ./storage_allocation_refinement.bend as AllocationLaws
import ./tensor_allocation_depth.bend as Depth
import ./storage_readback.bend as StorageReadback
import ./tensor_pieces_plan.bend as PiecesPlan
import ../tensor_tasks.bend as Tasks

# ---- The cells of a group loop ----

# Cell index of a loop whose first cell is gap, inner cells apart.
def address(+gap: Nat,+inner: Nat,+index: Nat) -> Nat:
  Nat.add(gap,Nat.mul(index,inner))

def address_residue(+gap: Nat,+inner: Nat,+index: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool},+gap_below: {Nat.is_lt(gap,inner) == True{} : Bool}) ->
  {Nat.mod(address(gap,inner,index),inner) == gap : Nat}:
  %Order.commutative(Nat.mul(index,inner),gap) : {Nat.mod(_,inner) == gap : Nat}
  %Equal.sym(Nat,Nat.mod(Nat.add(Nat.mul(index,inner),gap),inner),Nat.mod(gap,inner),DivisionOrder.remainder_shift(gap,inner,index,positive)) : {_ == gap : Nat}
  LineChainValues.small_remainder(gap,inner,gap_below)

# A cell of residue gap is the cell of its quotient.
def address_of(+value: Nat,+inner: Nat,+gap: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool},+residue: {Nat.mod(value,inner) == gap : Nat}) ->
  {value == address(gap,inner,Nat.div(value,inner)) : Nat}:
  +quotient = Nat.div(value,inner)
  Equal.trans(Nat,value,Nat.add(Nat.mul(quotient,inner),gap),Nat.add(gap,Nat.mul(quotient,inner)),
    %residue : {value == Nat.add(Nat.mul(quotient,inner),_) : Nat} DivisionOrder.quotient_exact(value,inner,positive),
    Order.commutative(Nat.mul(quotient,inner),gap))

def address_less(+gap: Nat,+inner: Nat,+left: Nat,+right: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool},+ordered: {Nat.is_lt(left,right) == True{} : Bool}) ->
  {Nat.is_lt(address(gap,inner,left),address(gap,inner,right)) == True{} : Bool}:
  Order.offset_lt(gap,Nat.mul(left,inner),Nat.mul(right,inner),RowMajor.multiply_lt(left,right,inner,ordered,positive))

# The cells of a loop from gap on, inner apart, below count.
def cells_split(+gap: Nat,+inner: Nat,+count: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool},+below: {Nat.is_lt(gap,count) == True{} : Bool}) ->
  {LineChainValues.cells_from(gap,inner,count) == 1n+Nat.div(Nat.sub(count,1n+gap),inner) : Nat}:
  +rest = Nat.sub(count,1n+gap)
  %Equal.sym(Bool,Nat.is_lt(gap,count),True{},below) : {Bool.pick(Nat,_,Nat.div(Nat.sub(Nat.add(count,inner),1n+gap),inner),0n) == 1n+Nat.div(rest,inner) : Nat}
  %Order.commutative(inner,count) : {Nat.div(Nat.sub(_,1n+gap),inner) == 1n+Nat.div(rest,inner) : Nat}
  %Equal.sym(Nat,Nat.sub(Nat.add(inner,count),1n+gap),Nat.add(inner,rest),Order.subtract_from_sum(inner,count,1n+gap,Order.strict_follow(gap,count,below))) :
    {Nat.div(_,inner) == 1n+Nat.div(rest,inner) : Nat}
  %Addition.zero_right(inner) : {Nat.div(Nat.add(_,rest),inner) == 1n+Nat.div(rest,inner) : Nat}
  DivisionOrder.quotient_shift(rest,inner,1n,positive)

# Every cell of the loop lies below count.
def taken_inside(+gap: Nat,+inner: Nat,+count: Nat,+index: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool},+below: {Nat.is_lt(gap,count) == True{} : Bool},
  +inside: {Nat.is_lt(index,LineChainValues.cells_from(gap,inner,count)) == True{} : Bool}) -> {Nat.is_lt(address(gap,inner,index),count) == True{} : Bool}:
  +rest = Nat.sub(count,1n+gap)
  +quotient = Nat.div(rest,inner)
  +index_le = Division.lt_succ_le(index,quotient,%cells_split(gap,inner,count,positive,below) : {Nat.is_lt(index,_) == True{} : Bool} inside)
  +product_le = Order.transitive(Nat.mul(index,inner),Nat.mul(quotient,inner),rest,RowMajor.mul_le(index,quotient,inner,index_le),DivisionOrder.product_below(rest,inner,positive))
  %Order.restore_shift(1n+gap,count,Order.strict_follow(gap,count,below)) : {Nat.is_lt(address(gap,inner,index),_) == True{} : Bool}
  Order.le_lt_trans(address(gap,inner,index),Nat.add(gap,rest),Nat.add(1n+gap,rest),Order.offset_le(gap,Nat.mul(index,inner),rest,product_le),
    Order.lt_succ(Nat.add(gap,rest),Nat.add(gap,rest),Order.reflexive(Nat.add(gap,rest))))

# A cell below count of residue gap is a cell of the loop.
def taken_reached(+value: Nat,+gap: Nat,+inner: Nat,+count: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool},+residue: {Nat.mod(value,inner) == gap : Nat},
  +inside: {Nat.is_lt(value,count) == True{} : Bool}) -> {Nat.is_lt(Nat.div(value,inner),LineChainValues.cells_from(gap,inner,count)) == True{} : Bool}:
  +quotient = Nat.div(value,inner)
  +rest = Nat.sub(count,1n+gap)
  +gap_le = {%residue : {Nat.is_le(_,value) == True{} : Bool} DivisionOrder.remainder_le(value,inner,positive) : {Nat.is_le(gap,value) == True{} : Bool}}
  +below = Order.le_lt_trans(gap,value,count,gap_le,inside)
  +reach = {%address_of(value,inner,gap,positive,residue) : {Nat.is_le(1n+_,count) == True{} : Bool} Order.strict_follow(value,count,inside) :
    {Nat.is_le(1n+address(gap,inner,quotient),count) == True{} : Bool}}
  +product_le = Order.cancel_offset_le(1n+gap,Nat.mul(quotient,inner),rest,
    %Equal.sym(Nat,Nat.add(1n+gap,rest),count,Order.restore_shift(1n+gap,count,Order.strict_follow(gap,count,below))) :
      {Nat.is_le(Nat.add(1n+gap,Nat.mul(quotient,inner)),_) == True{} : Bool} reach)
  +bound = {%Division.exact_quotient(quotient,inner,positive) : {Nat.is_le(_,Nat.div(rest,inner)) == True{} : Bool}
    DivisionOrder.quotient_le(Nat.mul(quotient,inner),rest,inner,positive,product_le) : {Nat.is_le(quotient,Nat.div(rest,inner)) == True{} : Bool}}
  %Equal.sym(Nat,LineChainValues.cells_from(gap,inner,count),1n+Nat.div(rest,inner),cells_split(gap,inner,count,positive,below)) : {Nat.is_lt(quotient,_) == True{} : Bool}
  Order.lt_succ(quotient,Nat.div(rest,inner),bound)

def differs_by(+gap: Nat,+inner: Nat,+index: Nat,+value: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool},+gap_below: {Nat.is_lt(gap,inner) == True{} : Bool},
  +other: {Nat.is_eq(Nat.mod(value,inner),gap) == False{} : Bool},equal: Bool,+compared: {Nat.is_eq(address(gap,inner,index),value) == equal : Bool}) -> {equal == False{} : Bool}:
  match equal:
    case False{}: {==}
    case True{}:
      +residue = {%Order.equal_of_is_eq(address(gap,inner,index),value,compared) : {Nat.mod(_,inner) == gap : Nat} address_residue(gap,inner,index,positive,gap_below) :
        {Nat.mod(value,inner) == gap : Nat}}
      Empty.absurd({True{} == False{} : Bool},WordEncoding.false_true(
        Equal.trans(Bool,False{},Nat.is_eq(Nat.mod(value,inner),gap),True{},Equal.sym(Bool,Nat.is_eq(Nat.mod(value,inner),gap),False{},other),
          %Equal.sym(Nat,Nat.mod(value,inner),gap,residue) : {Nat.is_eq(_,gap) == True{} : Bool} Order.equal_reflexive(gap))))

# No cell of a loop has another residue.
def differs(+gap: Nat,+inner: Nat,+index: Nat,+value: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool},+gap_below: {Nat.is_lt(gap,inner) == True{} : Bool},
  +other: {Nat.is_eq(Nat.mod(value,inner),gap) == False{} : Bool}) -> {Nat.is_eq(address(gap,inner,index),value) == False{} : Bool}:
  differs_by(gap,inner,index,value,positive,gap_below,other,Nat.is_eq(address(gap,inner,index),value),{==})

def word_same(+left: U32,+right: U32,+left_value: Nat,+right_value: Nat,+left_exact: {U32.to_nat(left) == left_value : Nat},+right_exact: {U32.to_nat(right) == right_value : Nat}) ->
  {U32.is_eq(left,right) == Nat.is_eq(left_value,right_value) : Bool}:
  %left_exact : {U32.is_eq(left,right) == Nat.is_eq(_,right_value) : Bool}
  %right_exact : {U32.is_eq(left,right) == Nat.is_eq(U32.to_nat(left),_) : Bool}
  Comparison.eq(left,right)

# ---- The positions of a group loop ----

# A group loop of a piece of count cells: its first cell gap, inner cells
# apart, in words start and step.
type Strides<-start: U32,-step: U32,-count: Nat> is Data:
  Strides{gap: Nat,inner: Nat,cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},start_exact: {U32.to_nat(start) == gap : Nat},
    step_exact: {U32.to_nat(step) == inner : Nat},positive: {Nat.is_lt(0n,inner) == True{} : Bool},gap_below: {Nat.is_lt(gap,inner) == True{} : Bool},
    count_room: {Nat.is_le(count,U32.to_nat(cap)) == True{} : Bool}}

def strides_gap(-start: U32,-step: U32,-count: Nat,strides: Strides<start,step,count>) -> Nat:
  Strides{gap,inner,cap,cap_ok,start_exact,step_exact,positive,gap_below,count_room} = strides
  gap

def strides_inner(-start: U32,-step: U32,-count: Nat,strides: Strides<start,step,count>) -> Nat:
  Strides{gap,inner,cap,cap_ok,start_exact,step_exact,positive,gap_below,count_room} = strides
  inner

def strides_positive(-start: U32,-step: U32,-count: Nat,+strides: Strides<start,step,count>) -> {Nat.is_lt(0n,strides_inner(start,step,count,strides)) == True{} : Bool}:
  Strides{gap,inner,cap,cap_ok,start_exact,step_exact,positive,gap_below,count_room} = strides
  positive

def strides_below(-start: U32,-step: U32,-count: Nat,+strides: Strides<start,step,count>) ->
  {Nat.is_lt(strides_gap(start,step,count,strides),strides_inner(start,step,count,strides)) == True{} : Bool}:
  Strides{gap,inner,cap,cap_ok,start_exact,step_exact,positive,gap_below,count_room} = strides
  gap_below

# The positions of the loop.
def taken(-start: U32,-step: U32,+count: Nat,+strides: Strides<start,step,count>) -> Nat:
  LineChainValues.cells_from(strides_gap(start,step,count,strides),strides_inner(start,step,count,strides),count)

def cell_of(-start: U32,-step: U32,-count: Nat,+strides: Strides<start,step,count>,+index: Nat) -> Nat:
  address(strides_gap(start,step,count,strides),strides_inner(start,step,count,strides),index)

# The cell a position writes, in words.
def place_word(~Group: Data,+parameters: Group,+start: U32,+step: U32,+word: U32,+index: Nat,+gap: Nat,+inner: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +start_exact: {U32.to_nat(start) == gap : Nat},+step_exact: {U32.to_nat(step) == inner : Nat},+positive: {Nat.is_lt(0n,inner) == True{} : Bool},
  +room: {Nat.is_le(address(gap,inner,index),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat(Kernel.place(~Group,Kernel.Grouped{parameters,start,step,word},U32.from_nat(index))) == address(gap,inner,index) : Nat}:
  +product_room = Order.transitive(Nat.mul(index,inner),address(gap,inner,index),U32.to_nat(cap),Order.right_le(gap,Nat.mul(index,inner)),room)
  +index_room = Order.transitive(index,Nat.mul(index,inner),U32.to_nat(cap),RowMajor.product_covers(index,inner,positive),product_room)
  +product = Arithmetic.product_exact(cap,cap_ok,U32.from_nat(index),step,index,inner,Bounds.roundtrip(index,cap,cap_ok,index_room),step_exact,product_room)
  Arithmetic.sum_exact(cap,cap_ok,start,(U32.from_nat(index) * step : U32),gap,Nat.mul(index,inner),start_exact,product,room)

# What a position of the loop knows: the cell it writes, in words, and that
# the cell lies inside the piece.
def Position(~Group: Data,+parameters: Group,+start: U32,+step: U32,+word: U32,+count: Nat,+strides: Strides<start,step,count>,+index: Nat) -> Data:
  KernelGemm.Both<{U32.to_nat(Kernel.place(~Group,Kernel.Grouped{parameters,start,step,word},U32.from_nat(index))) == cell_of(start,step,count,strides,index) : Nat},
    {Nat.is_lt(cell_of(start,step,count,strides,index),count) == True{} : Bool}>

def position_by(~Group: Data,+parameters: Group,+start: U32,+step: U32,+word: U32,+count: Nat,+strides: Strides<start,step,count>,+index: Nat,
  +inside: {Nat.is_lt(index,taken(start,step,count,strides)) == True{} : Bool},which: Bool,
  +classified: {Nat.is_lt(strides_gap(start,step,count,strides),count) == which : Bool}) -> Position(~Group,parameters,start,step,word,count,strides,index):
  Strides{+gap,+inner,+cap,+cap_ok,+start_exact,+step_exact,+positive,+gap_below,+count_room} = strides
  match which:
    case True{}:
      +below = taken_inside(gap,inner,count,index,positive,classified,inside)
      KernelGemm.Both{place_word(~Group,parameters,start,step,word,index,gap,inner,cap,cap_ok,start_exact,step_exact,positive,
        Order.transitive(address(gap,inner,index),count,U32.to_nat(cap),Bounds.strict_le(address(gap,inner,index),count,below),count_room)),below}
    case False{}:
      +empty = {%classified : {Nat.is_lt(index,Bool.pick(Nat,_,Nat.div(Nat.sub(Nat.add(count,inner),1n+gap),inner),0n)) == True{} : Bool} inside :
        {Nat.is_lt(index,0n) == True{} : Bool}}
      Empty.absurd(KernelGemm.Both<{U32.to_nat(Kernel.place(~Group,Kernel.Grouped{parameters,start,step,word},U32.from_nat(index))) == address(gap,inner,index) : Nat},
          {Nat.is_lt(address(gap,inner,index),count) == True{} : Bool}>,
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(index,0n),True{},Equal.sym(Bool,Nat.is_lt(index,0n),False{},Division.lt_zero(index)),empty)))

def position(~Group: Data,+parameters: Group,+start: U32,+step: U32,+word: U32,+count: Nat,+strides: Strides<start,step,count>,+index: Nat,
  +inside: {Nat.is_lt(index,taken(start,step,count,strides)) == True{} : Bool}) -> Position(~Group,parameters,start,step,word,count,strides,index):
  position_by(~Group,parameters,start,step,word,count,strides,index,inside,Nat.is_lt(strides_gap(start,step,count,strides),count),{==})

def exact_of(~Group: Data,+parameters: Group,+start: U32,+step: U32,+word: U32,+count: Nat,+strides: Strides<start,step,count>,+index: Nat,
  fact: Position(~Group,parameters,start,step,word,count,strides,index)) ->
  {U32.to_nat(Kernel.place(~Group,Kernel.Grouped{parameters,start,step,word},U32.from_nat(index))) == cell_of(start,step,count,strides,index) : Nat}:
  KernelGemm.Both{exact,below} = fact
  exact

def below_of(~Group: Data,+parameters: Group,+start: U32,+step: U32,+word: U32,+count: Nat,+strides: Strides<start,step,count>,+index: Nat,
  fact: Position(~Group,parameters,start,step,word,count,strides,index)) -> {Nat.is_lt(cell_of(start,step,count,strides,index),count) == True{} : Bool}:
  KernelGemm.Both{exact,below} = fact
  below

def position_exact(~Group: Data,+parameters: Group,+start: U32,+step: U32,+word: U32,+count: Nat,+strides: Strides<start,step,count>,+index: Nat,
  +inside: {Nat.is_lt(index,taken(start,step,count,strides)) == True{} : Bool}) ->
  {U32.to_nat(Kernel.place(~Group,Kernel.Grouped{parameters,start,step,word},U32.from_nat(index))) == cell_of(start,step,count,strides,index) : Nat}:
  exact_of(~Group,parameters,start,step,word,count,strides,index,position(~Group,parameters,start,step,word,count,strides,index,inside))

def position_below(~Group: Data,+parameters: Group,+start: U32,+step: U32,+word: U32,+count: Nat,+strides: Strides<start,step,count>,+index: Nat,
  +inside: {Nat.is_lt(index,taken(start,step,count,strides)) == True{} : Bool}) -> {Nat.is_lt(cell_of(start,step,count,strides,index),count) == True{} : Bool}:
  below_of(~Group,parameters,start,step,word,count,strides,index,position(~Group,parameters,start,step,word,count,strides,index,inside))

def taken_next(+index: Nat,+rest: Nat,+limit: Nat,+room: {Nat.is_le(Nat.add(index,1n+rest),limit) == True{} : Bool}) -> {Nat.is_le(Nat.add(1n+index,rest),limit) == True{} : Bool}:
  %Addition.successor_right(index,rest) : {Nat.is_le(_,limit) == True{} : Bool}
  room

def taken_here(+index: Nat,+rest: Nat,+limit: Nat,+room: {Nat.is_le(Nat.add(index,1n+rest),limit) == True{} : Bool}) -> {Nat.is_lt(index,limit) == True{} : Bool}:
  Bounds.nat_trans(index,Nat.add(index,1n+rest),limit,Order.below_sum(index,rest),room)

# Every position from index on writes inside a capacity above the piece.
def strided_inside(~Group: Data,count: Nat,+index: Nat,+parameters: Group,+start: U32,+step: U32,+word: U32,+cells: Nat,+strides: Strides<start,step,cells>,
  +capacity: U32,+capacity_room: {Nat.is_le(cells,U32.to_nat(capacity)) == True{} : Bool},+room: {Nat.is_le(Nat.add(index,count),taken(start,step,cells,strides)) == True{} : Bool}) ->
  Updates.Destinations(~Kernel.Grouped<Group>,~Kernel.place(~Group),count,U32.from_nat(index),Kernel.Grouped{parameters,start,step,word},capacity):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      +exact = position_exact(~Group,parameters,start,step,word,cells,strides,index,taken_here(index,rest,taken(start,step,cells,strides),room))
      +below = position_below(~Group,parameters,start,step,word,cells,strides,index,taken_here(index,rest,taken(start,step,cells,strides),room))
      +place = Kernel.place(~Group,Kernel.Grouped{parameters,start,step,word},U32.from_nat(index))
      KernelGemm.Both{
        %Equal.sym(Bool,U32.is_lt(place,capacity),Nat.is_lt(U32.to_nat(place),U32.to_nat(capacity)),Comparison.lt(place,capacity)) : {_ == True{} : Bool}
          %Equal.sym(Nat,U32.to_nat(place),cell_of(start,step,cells,strides,index),exact) : {Nat.is_lt(_,U32.to_nat(capacity)) == True{} : Bool}
          Bounds.nat_trans(cell_of(start,step,cells,strides,index),cells,U32.to_nat(capacity),below,capacity_room),
        %Equal.sym(U32,(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),Bounds.increment_encoded(index)) :
          Updates.Destinations(~Kernel.Grouped<Group>,~Kernel.place(~Group),rest,_,Kernel.Grouped{parameters,start,step,word},capacity)
          strided_inside(~Group,rest,1n+index,parameters,start,step,word,cells,strides,capacity,capacity_room,taken_next(index,rest,taken(start,step,cells,strides),room))}

# A query of another residue is no cell of the loop.
def residue_misses(~Group: Data,count: Nat,+index: Nat,+parameters: Group,+start: U32,+step: U32,+word: U32,+cells: Nat,+strides: Strides<start,step,cells>,
  +query: U32,+value: Nat,+query_exact: {U32.to_nat(query) == value : Nat},
  +other: {Nat.is_eq(Nat.mod(value,strides_inner(start,step,cells,strides)),strides_gap(start,step,cells,strides)) == False{} : Bool},
  +room: {Nat.is_le(Nat.add(index,count),taken(start,step,cells,strides)) == True{} : Bool}) ->
  Updates.Misses(~Kernel.Grouped<Group>,~Kernel.place(~Group),count,U32.from_nat(index),Kernel.Grouped{parameters,start,step,word},query):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      +exact = position_exact(~Group,parameters,start,step,word,cells,strides,index,taken_here(index,rest,taken(start,step,cells,strides),room))
      +place = Kernel.place(~Group,Kernel.Grouped{parameters,start,step,word},U32.from_nat(index))
      KernelGemm.Both{
        %Equal.sym(Bool,U32.is_eq(place,query),Nat.is_eq(cell_of(start,step,cells,strides,index),value),word_same(place,query,cell_of(start,step,cells,strides,index),value,exact,query_exact)) :
          {_ == False{} : Bool}
          differs(strides_gap(start,step,cells,strides),strides_inner(start,step,cells,strides),index,value,strides_positive(start,step,cells,strides),strides_below(start,step,cells,strides),other),
        %Equal.sym(U32,(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),Bounds.increment_encoded(index)) :
          Updates.Misses(~Kernel.Grouped<Group>,~Kernel.place(~Group),rest,_,Kernel.Grouped{parameters,start,step,word},query)
          residue_misses(~Group,rest,1n+index,parameters,start,step,word,cells,strides,query,value,query_exact,other,taken_next(index,rest,taken(start,step,cells,strides),room))}

# The positions after the one that writes a query miss it.
def after_misses(~Group: Data,count: Nat,+index: Nat,+parameters: Group,+start: U32,+step: U32,+word: U32,+cells: Nat,+strides: Strides<start,step,cells>,
  +query: U32,+hit: Nat,+query_exact: {U32.to_nat(query) == cell_of(start,step,cells,strides,hit) : Nat},+later: {Nat.is_lt(hit,index) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(index,count),taken(start,step,cells,strides)) == True{} : Bool}) ->
  Updates.Misses(~Kernel.Grouped<Group>,~Kernel.place(~Group),count,U32.from_nat(index),Kernel.Grouped{parameters,start,step,word},query):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      +exact = position_exact(~Group,parameters,start,step,word,cells,strides,index,taken_here(index,rest,taken(start,step,cells,strides),room))
      +place = Kernel.place(~Group,Kernel.Grouped{parameters,start,step,word},U32.from_nat(index))
      +here = cell_of(start,step,cells,strides,index)
      +there = cell_of(start,step,cells,strides,hit)
      KernelGemm.Both{
        %Equal.sym(Bool,U32.is_eq(place,query),Nat.is_eq(here,there),word_same(place,query,here,there,exact,query_exact)) : {_ == False{} : Bool}
          %Order.equal_symmetric(there,here) : {_ == False{} : Bool}
          Order.less_not_equal(there,here,address_less(strides_gap(start,step,cells,strides),strides_inner(start,step,cells,strides),hit,index,strides_positive(start,step,cells,strides),later)),
        %Equal.sym(U32,(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),Bounds.increment_encoded(index)) :
          Updates.Misses(~Kernel.Grouped<Group>,~Kernel.place(~Group),rest,_,Kernel.Grouped{parameters,start,step,word},query)
          after_misses(~Group,rest,1n+index,parameters,start,step,word,cells,strides,query,hit,query_exact,
            Order.le_lt_trans(hit,index,1n+index,Bounds.strict_le(hit,index,later),Order.lt_succ(index,index,Order.reflexive(index))),
            taken_next(index,rest,taken(start,step,cells,strides),room))}

# The position hit writes the query, the others miss it.
def hits_from(~Group: Data,count: Nat,+index: Nat,before: Nat,+parameters: Group,+start: U32,+step: U32,+word: U32,+cells: Nat,+strides: Strides<start,step,cells>,
  +query: U32,+hit: Nat,+query_exact: {U32.to_nat(query) == cell_of(start,step,cells,strides,hit) : Nat},+reach: {Nat.add(index,before) == hit : Nat},
  +hit_before: {Nat.is_lt(hit,Nat.add(index,count)) == True{} : Bool},+room: {Nat.is_le(Nat.add(index,count),taken(start,step,cells,strides)) == True{} : Bool}) ->
  Updates.Hits(~Kernel.Grouped<Group>,~Kernel.place(~Group),count,U32.from_nat(index),Kernel.Grouped{parameters,start,step,word},query,U32.from_nat(hit),before):
  match count:
    case 0n:
      +ahead = {%Addition.zero_right(index) : {Nat.is_lt(hit,_) == True{} : Bool} hit_before : {Nat.is_lt(hit,index) == True{} : Bool}}
      +behind = {%reach : {Nat.is_le(index,_) == True{} : Bool} Order.le_add(index,before) : {Nat.is_le(index,hit) == True{} : Bool}}
      Empty.absurd(Empty,WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(hit,index),True{},Equal.sym(Bool,Nat.is_lt(hit,index),False{},Order.not_below(index,hit,behind)),ahead)))
    case 1n+ +rest:
      match before:
        case 0n:
          +exact = position_exact(~Group,parameters,start,step,word,cells,strides,index,taken_here(index,rest,taken(start,step,cells,strides),room))
          +place = Kernel.place(~Group,Kernel.Grouped{parameters,start,step,word},U32.from_nat(index))
          +here = cell_of(start,step,cells,strides,index)
          +there = cell_of(start,step,cells,strides,hit)
          +next_room = taken_next(index,rest,taken(start,step,cells,strides),room)
          +same = Equal.trans(Nat,index,Nat.add(index,0n),hit,Equal.sym(Nat,Nat.add(index,0n),index,Addition.zero_right(index)),reach)
          KernelGemm.Both{Equal.cong(Nat,U32,value => U32.from_nat(value),index,hit,same),
            KernelGemm.Both{
              %Equal.sym(Bool,U32.is_eq(place,query),Nat.is_eq(here,there),word_same(place,query,here,there,exact,query_exact)) : {_ == True{} : Bool}
                %same : {Nat.is_eq(here,cell_of(start,step,cells,strides,_)) == True{} : Bool}
                Order.equal_reflexive(here),
              %Equal.sym(U32,(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),Bounds.increment_encoded(index)) :
                Updates.Misses(~Kernel.Grouped<Group>,~Kernel.place(~Group),rest,_,Kernel.Grouped{parameters,start,step,word},query)
                after_misses(~Group,rest,1n+index,parameters,start,step,word,cells,strides,query,hit,query_exact,
                  %same : {Nat.is_lt(_,1n+index) == True{} : Bool} Order.lt_succ(index,index,Order.reflexive(index)),next_room)}}
        case 1n+ +earlier:
          +exact = position_exact(~Group,parameters,start,step,word,cells,strides,index,taken_here(index,rest,taken(start,step,cells,strides),room))
          +place = Kernel.place(~Group,Kernel.Grouped{parameters,start,step,word},U32.from_nat(index))
          +here = cell_of(start,step,cells,strides,index)
          +there = cell_of(start,step,cells,strides,hit)
          +next_room = taken_next(index,rest,taken(start,step,cells,strides),room)
          +ahead = Equal.trans(Nat,Nat.add(1n+index,earlier),Nat.add(index,1n+earlier),hit,Equal.sym(Nat,Nat.add(index,1n+earlier),1n+Nat.add(index,earlier),Addition.successor_right(index,earlier)),reach)
          +sooner = {%reach : {Nat.is_lt(index,_) == True{} : Bool} Order.below_sum(index,earlier) : {Nat.is_lt(index,hit) == True{} : Bool}}
          KernelGemm.Both{
            %Equal.sym(Bool,U32.is_eq(place,query),Nat.is_eq(here,there),word_same(place,query,here,there,exact,query_exact)) : {_ == False{} : Bool}
              Order.less_not_equal(here,there,address_less(strides_gap(start,step,cells,strides),strides_inner(start,step,cells,strides),index,hit,strides_positive(start,step,cells,strides),sooner)),
            %Equal.sym(U32,(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),Bounds.increment_encoded(index)) :
              Updates.Hits(~Kernel.Grouped<Group>,~Kernel.place(~Group),rest,_,Kernel.Grouped{parameters,start,step,word},query,U32.from_nat(hit),earlier)
              hits_from(~Group,rest,1n+index,earlier,parameters,start,step,word,cells,strides,query,hit,query_exact,ahead,
                %Addition.successor_right(index,rest) : {Nat.is_lt(hit,_) == True{} : Bool} hit_before,next_room)}

# ---- Residues of the groups ----

def gap_back_by(+residue: Nat,+inner: Nat,+line: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool},+residue_below: {Nat.is_lt(residue,inner) == True{} : Bool},
  +line_below: {Nat.is_lt(line,inner) == True{} : Bool},which: Bool,+classified: {Nat.is_le(residue,line) == which : Bool}) ->
  {Nat.mod(Nat.add(residue,LineChainValues.gap_at(line,inner,residue)),inner) == line : Nat}:
  match which:
    case True{}:
      %Equal.sym(Bool,Nat.is_le(residue,line),True{},classified) :
        {Nat.mod(Nat.add(residue,Bool.pick(Nat,_,Nat.sub(line,residue),Nat.sub(Nat.add(line,inner),residue))),inner) == line : Nat}
      %Equal.sym(Nat,Nat.add(residue,Nat.sub(line,residue)),line,Order.restore_shift(residue,line,classified)) : {Nat.mod(_,inner) == line : Nat}
      LineChainValues.small_remainder(line,inner,line_below)
    case False{}:
      +covered = Order.transitive(residue,inner,Nat.add(line,inner),Bounds.strict_le(residue,inner,residue_below),Order.right_le(line,inner))
      %Equal.sym(Bool,Nat.is_le(residue,line),False{},classified) :
        {Nat.mod(Nat.add(residue,Bool.pick(Nat,_,Nat.sub(line,residue),Nat.sub(Nat.add(line,inner),residue))),inner) == line : Nat}
      %Equal.sym(Nat,Nat.add(residue,Nat.sub(Nat.add(line,inner),residue)),Nat.add(line,inner),Order.restore_shift(residue,Nat.add(line,inner),covered)) : {Nat.mod(_,inner) == line : Nat}
      %Order.commutative(inner,line) : {Nat.mod(_,inner) == line : Nat}
      %Addition.zero_right(inner) : {Nat.mod(Nat.add(_,line),inner) == line : Nat}
      %Equal.sym(Nat,Nat.mod(Nat.add(Nat.mul(1n,inner),line),inner),Nat.mod(line,inner),DivisionOrder.remainder_shift(line,inner,1n,positive)) : {_ == line : Nat}
      LineChainValues.small_remainder(line,inner,line_below)

# The loop of group line starts at the cell that brings the piece to line.
def gap_back(+residue: Nat,+inner: Nat,+line: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool},+residue_below: {Nat.is_lt(residue,inner) == True{} : Bool},
  +line_below: {Nat.is_lt(line,inner) == True{} : Bool}) -> {Nat.mod(Nat.add(residue,LineChainValues.gap_at(line,inner,residue)),inner) == line : Nat}:
  gap_back_by(residue,inner,line,positive,residue_below,line_below,Nat.is_le(residue,line),{==})

def wrap_mod(+value: Nat,+inner: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool},+inner_le: {Nat.is_le(inner,value) == True{} : Bool},
  +below: {Nat.is_lt(Nat.sub(value,inner),inner) == True{} : Bool}) -> {Nat.mod(value,inner) == Nat.sub(value,inner) : Nat}:
  %Order.restore_shift(inner,value,inner_le) : {Nat.mod(_,inner) == Nat.sub(value,inner) : Nat}
  %Addition.zero_right(inner) : {Nat.mod(Nat.add(_,Nat.sub(value,inner)),inner) == Nat.sub(value,inner) : Nat}
  %Equal.sym(Nat,Nat.mod(Nat.add(Nat.mul(1n,inner),Nat.sub(value,inner)),inner),Nat.mod(Nat.sub(value,inner),inner),DivisionOrder.remainder_shift(Nat.sub(value,inner),inner,1n,positive)) :
    {_ == Nat.sub(value,inner) : Nat}
  LineChainValues.small_remainder(Nat.sub(value,inner),inner,below)

def wrap_sum_less(+residue: Nat,+offset: Nat,+inner: Nat,+offset_below: {Nat.is_lt(offset,inner) == True{} : Bool},+inner_le: {Nat.is_le(inner,Nat.add(residue,offset)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(inner,Nat.sub(Nat.add(residue,offset),inner)),Nat.add(inner,residue)) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(inner,Nat.sub(Nat.add(residue,offset),inner)),Nat.add(residue,offset),Order.restore_shift(inner,Nat.add(residue,offset),inner_le)) :
    {Nat.is_lt(_,Nat.add(inner,residue)) == True{} : Bool}
  %Order.commutative(residue,inner) : {Nat.is_lt(Nat.add(residue,offset),_) == True{} : Bool}
  Order.offset_lt(residue,offset,inner,offset_below)

def gap_sum_by(+residue: Nat,+offset: Nat,+inner: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool},+residue_below: {Nat.is_lt(residue,inner) == True{} : Bool},
  +offset_below: {Nat.is_lt(offset,inner) == True{} : Bool},which: Bool,+classified: {Nat.is_lt(Nat.add(residue,offset),inner) == which : Bool}) ->
  {LineChainValues.gap_at(Nat.mod(Nat.add(residue,offset),inner),inner,residue) == offset : Nat}:
  match which:
    case True{}:
      %Equal.sym(Nat,Nat.mod(Nat.add(residue,offset),inner),Nat.add(residue,offset),LineChainValues.small_remainder(Nat.add(residue,offset),inner,classified)) :
        {LineChainValues.gap_at(_,inner,residue) == offset : Nat}
      %Equal.sym(Bool,Nat.is_le(residue,Nat.add(residue,offset)),True{},Order.le_add(residue,offset)) :
        {Bool.pick(Nat,_,Nat.sub(Nat.add(residue,offset),residue),Nat.sub(Nat.add(Nat.add(residue,offset),inner),residue)) == offset : Nat}
      Order.subtract_shift(residue,offset)
    case False{}:
      +sum = Nat.add(residue,offset)
      +wrapped = Nat.sub(sum,inner)
      +inner_le = Order.not_less(sum,inner,classified)
      +double_bound = Order.le_lt_trans(sum,Nat.add(inner,offset),Nat.add(inner,inner),Bounds.strict_le(sum,Nat.add(inner,offset),Order.add_lt(residue,inner,offset,residue_below)),
        Order.offset_lt(inner,offset,inner,offset_below))
      +wrapped_below = Order.sub_lt(sum,inner,inner,inner_le,double_bound)
      +before = Order.cancel_offset_lt(inner,wrapped,residue,wrap_sum_less(residue,offset,inner,offset_below,inner_le))
      %Equal.sym(Nat,Nat.mod(sum,inner),wrapped,wrap_mod(sum,inner,positive,inner_le,wrapped_below)) : {LineChainValues.gap_at(_,inner,residue) == offset : Nat}
      %Equal.sym(Bool,Nat.is_le(residue,wrapped),False{},LineChainValues.lt_le_false(residue,wrapped,before)) :
        {Bool.pick(Nat,_,Nat.sub(wrapped,residue),Nat.sub(Nat.add(wrapped,inner),residue)) == offset : Nat}
      %Order.commutative(inner,wrapped) : {Nat.sub(_,residue) == offset : Nat}
      %Equal.sym(Nat,Nat.add(inner,wrapped),sum,Order.restore_shift(inner,sum,inner_le)) : {Nat.sub(_,residue) == offset : Nat}
      Order.subtract_shift(residue,offset)

# The loop of group (residue + offset) % inner starts at offset.
def gap_sum(+residue: Nat,+offset: Nat,+inner: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool},+residue_below: {Nat.is_lt(residue,inner) == True{} : Bool},
  +offset_below: {Nat.is_lt(offset,inner) == True{} : Bool}) -> {LineChainValues.gap_at(Nat.mod(Nat.add(residue,offset),inner),inner,residue) == offset : Nat}:
  gap_sum_by(residue,offset,inner,positive,residue_below,offset_below,Nat.is_lt(Nat.add(residue,offset),inner),{==})

def gap_other_by(+residue: Nat,+offset: Nat,+inner: Nat,+line: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool},+residue_below: {Nat.is_lt(residue,inner) == True{} : Bool},
  +line_below: {Nat.is_lt(line,inner) == True{} : Bool},+other: {Nat.is_eq(line,Nat.mod(Nat.add(residue,offset),inner)) == False{} : Bool},equal: Bool,
  +compared: {Nat.is_eq(offset,LineChainValues.gap_at(line,inner,residue)) == equal : Bool}) -> {equal == False{} : Bool}:
  match equal:
    case False{}: {==}
    case True{}:
      +same = Order.equal_of_is_eq(offset,LineChainValues.gap_at(line,inner,residue),compared)
      +back = {%Equal.sym(Nat,offset,LineChainValues.gap_at(line,inner,residue),same) : {Nat.mod(Nat.add(residue,_),inner) == line : Nat}
        gap_back(residue,inner,line,positive,residue_below,line_below) : {Nat.mod(Nat.add(residue,offset),inner) == line : Nat}}
      Empty.absurd({True{} == False{} : Bool},WordEncoding.false_true(
        Equal.trans(Bool,False{},Nat.is_eq(line,Nat.mod(Nat.add(residue,offset),inner)),True{},Equal.sym(Bool,Nat.is_eq(line,Nat.mod(Nat.add(residue,offset),inner)),False{},other),
          %Equal.sym(Nat,Nat.mod(Nat.add(residue,offset),inner),line,back) : {Nat.is_eq(line,_) == True{} : Bool} Order.equal_reflexive(line))))

# Another group starts elsewhere.
def gap_other(+residue: Nat,+offset: Nat,+inner: Nat,+line: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool},+residue_below: {Nat.is_lt(residue,inner) == True{} : Bool},
  +line_below: {Nat.is_lt(line,inner) == True{} : Bool},+other: {Nat.is_eq(line,Nat.mod(Nat.add(residue,offset),inner)) == False{} : Bool}) ->
  {Nat.is_eq(offset,LineChainValues.gap_at(line,inner,residue)) == False{} : Bool}:
  gap_other_by(residue,offset,inner,line,positive,residue_below,line_below,other,Nat.is_eq(offset,LineChainValues.gap_at(line,inner,residue)),{==})

def regroup(+first: Nat,+second: Nat,+third: Nat,+fourth: Nat) ->
  {Nat.add(Nat.add(first,second),Nat.add(third,fourth)) == Nat.add(Nat.add(first,third),Nat.add(second,fourth)) : Nat}:
  %Equal.sym(Nat,Nat.add(Nat.add(first,second),Nat.add(third,fourth)),Nat.add(first,Nat.add(second,Nat.add(third,fourth))),Addition.associative(first,second,Nat.add(third,fourth))) :
    {_ == Nat.add(Nat.add(first,third),Nat.add(second,fourth)) : Nat}
  %RowMajor.add_swap(third,second,fourth) : {Nat.add(first,_) == Nat.add(Nat.add(first,third),Nat.add(second,fourth)) : Nat}
  Equal.sym(Nat,Nat.add(Nat.add(first,third),Nat.add(second,fourth)),Nat.add(first,Nat.add(third,Nat.add(second,fourth))),Addition.associative(first,third,Nat.add(second,fourth)))

# The residue of a sum is the residue of the sum of the residues.
def mod_sum(+left: Nat,+right: Nat,+inner: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool}) ->
  {Nat.mod(Nat.add(left,right),inner) == Nat.mod(Nat.add(Nat.mod(left,inner),Nat.mod(right,inner)),inner) : Nat}:
  +left_rows = Nat.div(left,inner)
  +left_rest = Nat.mod(left,inner)
  +right_rows = Nat.div(right,inner)
  +right_rest = Nat.mod(right,inner)
  %Equal.sym(Nat,left,Nat.add(Nat.mul(left_rows,inner),left_rest),DivisionOrder.quotient_exact(left,inner,positive)) :
    {Nat.mod(Nat.add(_,right),inner) == Nat.mod(Nat.add(left_rest,right_rest),inner) : Nat}
  %Equal.sym(Nat,right,Nat.add(Nat.mul(right_rows,inner),right_rest),DivisionOrder.quotient_exact(right,inner,positive)) :
    {Nat.mod(Nat.add(Nat.add(Nat.mul(left_rows,inner),left_rest),_),inner) == Nat.mod(Nat.add(left_rest,right_rest),inner) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(left_rows,inner),left_rest),Nat.add(Nat.mul(right_rows,inner),right_rest)),
      Nat.add(Nat.add(Nat.mul(left_rows,inner),Nat.mul(right_rows,inner)),Nat.add(left_rest,right_rest)),regroup(Nat.mul(left_rows,inner),left_rest,Nat.mul(right_rows,inner),right_rest)) :
    {Nat.mod(_,inner) == Nat.mod(Nat.add(left_rest,right_rest),inner) : Nat}
  %RowMajor.distrib(left_rows,right_rows,inner) : {Nat.mod(Nat.add(_,Nat.add(left_rest,right_rest)),inner) == Nat.mod(Nat.add(left_rest,right_rest),inner) : Nat}
  DivisionOrder.remainder_shift(Nat.add(left_rest,right_rest),inner,Nat.add(left_rows,right_rows),positive)

# ---- The groups of a piece ----

# What every piece of a pass fits in: two strides of the groups, and the
# cells of the piece with one stride.
type Room<-count: U32,-cells: U32> is Data:
  Room{cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},double_room: {Nat.is_le(Nat.double(U32.to_nat(cells)),U32.to_nat(cap)) == True{} : Bool},
    positive: {Nat.is_lt(0n,U32.to_nat(cells)) == True{} : Bool},count_room: {Nat.is_le(Nat.add(U32.to_nat(count),U32.to_nat(cells)),U32.to_nat(cap)) == True{} : Bool}}

# The group of cell value of a piece from first.
def residue_group(+first: U32,+cells: U32,+value: Nat) -> Nat:
  Nat.mod(Nat.add(Nat.mod(U32.to_nat(first),U32.to_nat(cells)),Nat.mod(value,U32.to_nat(cells))),U32.to_nat(cells))

# The first cell of the loop of a group and how many it writes, as the pass
# computes them.
def group_start(+first: U32,+cells: U32,+group: Nat) -> U32:
  LineChainValues.segment_start(U32.from_nat(group),cells,first)

def group_gap(+first: U32,+cells: U32,+group: Nat) -> Nat:
  LineChainValues.gap_at(group,U32.to_nat(cells),Nat.mod(U32.to_nat(first),U32.to_nat(cells)))

def group_word(+first: U32,+count: U32,+cells: U32,+group: Nat) -> U32:
  LineChainValues.segment_cells(group_start(first,cells,group),cells,count)

def group_line(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+cells: U32,+double_room: {Nat.is_le(Nat.double(U32.to_nat(cells)),U32.to_nat(cap)) == True{} : Bool},
  +group: Nat,+below: {Nat.is_lt(group,U32.to_nat(cells)) == True{} : Bool}) -> {U32.to_nat(U32.from_nat(group)) == group : Nat}:
  Bounds.roundtrip(group,cap,cap_ok,Order.transitive(group,U32.to_nat(cells),U32.to_nat(cap),Bounds.strict_le(group,U32.to_nat(cells),below),
    Order.transitive(U32.to_nat(cells),Nat.double(U32.to_nat(cells)),U32.to_nat(cap),Order.le_double(U32.to_nat(cells)),double_room)))

def group_start_exact(+first: U32,+count: U32,+cells: U32,+room: Room<count,cells>,+group: Nat,+below: {Nat.is_lt(group,U32.to_nat(cells)) == True{} : Bool}) ->
  {U32.to_nat(group_start(first,cells,group)) == group_gap(first,cells,group) : Nat}:
  Room{+cap,+cap_ok,+double_room,+positive,+count_room} = room
  +inner = U32.to_nat(cells)
  +residue = Nat.mod(U32.to_nat(first),inner)
  +line_exact = group_line(cap,cap_ok,cells,double_room,group,below)
  Equal.trans(Nat,U32.to_nat(group_start(first,cells,group)),LineChainValues.gap_at(U32.to_nat(U32.from_nat(group)),inner,residue),group_gap(first,cells,group),
    LineChainValues.start_exact(U32.from_nat(group),cells,first,cap,cap_ok,double_room,positive,
      %Equal.sym(Nat,U32.to_nat(U32.from_nat(group)),group,line_exact) : {Nat.is_lt(_,inner) == True{} : Bool} below),
    Equal.cong(Nat,Nat,line => LineChainValues.gap_at(line,inner,residue),U32.to_nat(U32.from_nat(group)),group,line_exact))

def group_strides(+first: U32,+count: U32,+cells: U32,+room: Room<count,cells>,+group: Nat,+below: {Nat.is_lt(group,U32.to_nat(cells)) == True{} : Bool}) ->
  Strides<group_start(first,cells,group),cells,U32.to_nat(count)>:
  Room{+cap,+cap_ok,+double_room,+positive,+count_room} = room
  Strides{group_gap(first,cells,group),U32.to_nat(cells),cap,cap_ok,group_start_exact(first,count,cells,Room{cap,cap_ok,double_room,positive,count_room},group,below),{==},positive,
    LineChainValues.line_gap_below(group,U32.to_nat(cells),Nat.mod(U32.to_nat(first),U32.to_nat(cells)),below,DivisionOrder.remainder_small(U32.to_nat(first),U32.to_nat(cells),positive)),
    Order.transitive(U32.to_nat(count),Nat.add(U32.to_nat(count),U32.to_nat(cells)),U32.to_nat(cap),Order.le_add(U32.to_nat(count),U32.to_nat(cells)),count_room)}

def group_count(+first: U32,+count: U32,+cells: U32,+room: Room<count,cells>,+group: Nat,+below: {Nat.is_lt(group,U32.to_nat(cells)) == True{} : Bool}) ->
  {U32.to_nat(group_word(first,count,cells,group)) == taken(group_start(first,cells,group),cells,U32.to_nat(count),group_strides(first,count,cells,room,group,below)) : Nat}:
  Room{+cap,+cap_ok,+double_room,+positive,+count_room} = room
  LineChainValues.cells_exact(group_start(first,cells,group),cells,count,group_gap(first,cells,group),cap,cap_ok,double_room,positive,
    group_start_exact(first,count,cells,Room{cap,cap_ok,double_room,positive,count_room},group,below),count_room)

def group_below(~Group: Data,+group: Nat,+parameters: Group,+rest: List<&2,Group>,+cells: U32,+listed: {Nat.add(group,List.length(&2,Group,Con{parameters,rest})) == U32.to_nat(cells) : Nat}) ->
  {Nat.is_lt(group,U32.to_nat(cells)) == True{} : Bool}:
  %listed : {Nat.is_lt(group,_) == True{} : Bool}
  Order.below_sum(group,List.length(&2,Group,rest))

def group_listed(~Group: Data,+group: Nat,+parameters: Group,+rest: List<&2,Group>,+cells: U32,+listed: {Nat.add(group,List.length(&2,Group,Con{parameters,rest})) == U32.to_nat(cells) : Nat}) ->
  {Nat.add(1n+group,List.length(&2,Group,rest)) == U32.to_nat(cells) : Nat}:
  %Addition.successor_right(group,List.length(&2,Group,rest)) : {_ == U32.to_nat(cells) : Nat}
  listed

# ---- The loops of the groups ----

# A sample of the cell a position writes.
def placed(~Group: Data,~Sample: Data,~at: Group -> Tree.Tree<Tile.Tile8> -> U32 -> Sample,reader: Tree.Tree<Tile.Tile8>,context: Kernel.Grouped<Group>,position: U32) -> Sample:
  Kernel.Grouped{group,+start,+step,count} = context
  at(group,reader,(start + position * step : U32))

def group_loop(~Group: Data,~Sample: Data,~at: Group -> Tree.Tree<Tile.Tile8> -> U32 -> Sample,~combine: Tile.Tile8 -> Sample -> Tile.Tile8,+context: Kernel.Grouped<Group>,+reader: Tree.Tree<Tile.Tile8>,+tree: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  Updates.state_tree(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Group>,Loop.run(~Traversal.State<&2,&2,Tree.Tree<Tile.Tile8>,Tree.Tree<Tile.Tile8>,Kernel.Grouped<Group>>,
    ~ArrayLaws.model_update_step(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Sample,~Kernel.Grouped<Group>,~placed(~Group,~Sample,~at),~combine,~Kernel.place(~Group)),U32.to_nat(Kernel.count_of(~Group,context)),Traversal.State{reader,tree,0,context}))

# The loops of the groups of a piece from group on, one after another.
def groups_tree(~Group: Data,~Sample: Data,~at: Group -> Tree.Tree<Tile.Tile8> -> U32 -> Sample,~combine: Tile.Tile8 -> Sample -> Tile.Tile8,groups: List<&2,Group>,+first: U32,+count: U32,+cells: U32,+group: U32,+reader: Tree.Tree<Tile.Tile8>,+tree: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  match groups:
    case Nil{}: tree
    case Con{+parameters,rest}:
      groups_tree(~Group,~Sample,~at,~combine,rest,first,count,cells,(group + 1 : U32),reader,group_loop(~Group,~Sample,~at,~combine,BNCells.grouped(~Group,parameters,first,count,cells,group),reader,tree))

# The value they leave at query.
def groups_at(~Group: Data,~Sample: Data,~at: Group -> Tree.Tree<Tile.Tile8> -> U32 -> Sample,~combine: Tile.Tile8 -> Sample -> Tile.Tile8,groups: List<&2,Group>,+first: U32,+count: U32,+cells: U32,+group: U32,+reader: Tree.Tree<Tile.Tile8>,+query: U32,+current: Tile.Tile8) -> Tile.Tile8:
  match groups:
    case Nil{}: current
    case Con{+parameters,rest}:
      groups_at(~Group,~Sample,~at,~combine,rest,first,count,cells,(group + 1 : U32),reader,query,
        Updates.updated_at(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Sample,~Kernel.Grouped<Group>,~placed(~Group,~Sample,~at),~combine,~Kernel.place(~Group),U32.to_nat(Kernel.count_of(~Group,BNCells.grouped(~Group,parameters,first,count,cells,group))),0,reader,
          BNCells.grouped(~Group,parameters,first,count,cells,group),query,current))

def groups_read(~Group: Data,~Sample: Data,~at: Group -> Tree.Tree<Tile.Tile8> -> U32 -> Sample,~combine: Tile.Tile8 -> Sample -> Tile.Tile8,groups: List<&2,Group>,+group: Nat,+first: U32,+count: U32,+cells: U32,+room: Room<count,cells>,+reader: Tree.Tree<Tile.Tile8>,+tree: Tree.Tree<Tile.Tile8>,+depth: Nat,+query: U32,
  +balanced: Shape.Balanced(Tile.Tile8,depth,tree),+limit: {Nat.is_le(depth,31n) == True{} : Bool},+query_inside: {U32.is_lt(query,Tree.capacity(Tile.Tile8,tree)) == True{} : Bool},
  +capacity_room: {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Tile.Tile8,tree))) == True{} : Bool},+listed: {Nat.add(group,List.length(&2,Group,groups)) == U32.to_nat(cells) : Nat}) ->
  {Tree.read(Tile.Tile8,groups_tree(~Group,~Sample,~at,~combine,groups,first,count,cells,U32.from_nat(group),reader,tree),query) ==
    groups_at(~Group,~Sample,~at,~combine,groups,first,count,cells,U32.from_nat(group),reader,query,Tree.read(Tile.Tile8,tree,query)) : Tile.Tile8}:
  match groups:
    case Nil{}: {==}
    case Con{+parameters,+rest}:
      +below = group_below(~Group,group,parameters,rest,cells,listed)
      +strides = group_strides(first,count,cells,room,group,below)
      +start = group_start(first,cells,group)
      +word = group_word(first,count,cells,group)
      +loops = taken(start,cells,U32.to_nat(count),strides)
      +loop_tree = group_loop(~Group,~Sample,~at,~combine,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)),reader,tree)
      +inside = {%Equal.sym(Nat,U32.to_nat(word),loops,group_count(first,count,cells,room,group,below)) :
          Updates.Destinations(~Kernel.Grouped<Group>,~Kernel.place(~Group),_,U32.from_nat(0n),BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)),Tree.capacity(Tile.Tile8,tree))
          strided_inside(~Group,loops,0n,parameters,start,cells,word,U32.to_nat(count),strides,Tree.capacity(Tile.Tile8,tree),capacity_room,Order.reflexive(loops)) :
        Updates.Destinations(~Kernel.Grouped<Group>,~Kernel.place(~Group),U32.to_nat(Kernel.count_of(~Group,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)))),0,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)),Tree.capacity(Tile.Tile8,tree))}
      +read_loop = Updates.update_read(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Sample,~Kernel.Grouped<Group>,~placed(~Group,~Sample,~at),~combine,~Kernel.place(~Group),U32.to_nat(Kernel.count_of(~Group,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)))),0,reader,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)),tree,depth,query,balanced,limit,query_inside,inside)
      +capacity_same = Updates.update_capacity(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Sample,~Kernel.Grouped<Group>,~placed(~Group,~Sample,~at),~combine,~Kernel.place(~Group),U32.to_nat(Kernel.count_of(~Group,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)))),0,reader,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)),tree)
      %Equal.sym(U32,(U32.from_nat(group) + 1 : U32),U32.from_nat(1n+group),Bounds.increment_encoded(group)) :
        {Tree.read(Tile.Tile8,groups_tree(~Group,~Sample,~at,~combine,rest,first,count,cells,_,reader,loop_tree),query) ==
          groups_at(~Group,~Sample,~at,~combine,rest,first,count,cells,_,reader,query,Updates.updated_at(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Sample,~Kernel.Grouped<Group>,~placed(~Group,~Sample,~at),~combine,~Kernel.place(~Group),U32.to_nat(Kernel.count_of(~Group,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)))),0,reader,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)),query,Tree.read(Tile.Tile8,tree,query))) : Tile.Tile8}
      Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,groups_tree(~Group,~Sample,~at,~combine,rest,first,count,cells,U32.from_nat(1n+group),reader,loop_tree),query),
        groups_at(~Group,~Sample,~at,~combine,rest,first,count,cells,U32.from_nat(1n+group),reader,query,Tree.read(Tile.Tile8,loop_tree,query)),
        groups_at(~Group,~Sample,~at,~combine,rest,first,count,cells,U32.from_nat(1n+group),reader,query,Updates.updated_at(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Sample,~Kernel.Grouped<Group>,~placed(~Group,~Sample,~at),~combine,~Kernel.place(~Group),U32.to_nat(Kernel.count_of(~Group,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)))),0,reader,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)),query,Tree.read(Tile.Tile8,tree,query))),
        groups_read(~Group,~Sample,~at,~combine,rest,1n+group,first,count,cells,room,reader,loop_tree,depth,query,Updates.update_balanced(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Sample,~Kernel.Grouped<Group>,~placed(~Group,~Sample,~at),~combine,~Kernel.place(~Group),U32.to_nat(Kernel.count_of(~Group,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)))),0,reader,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)),tree,depth,balanced),limit,
          %Equal.sym(U32,Tree.capacity(Tile.Tile8,loop_tree),Tree.capacity(Tile.Tile8,tree),capacity_same) : {U32.is_lt(query,_) == True{} : Bool} query_inside,
          %Equal.sym(U32,Tree.capacity(Tile.Tile8,loop_tree),Tree.capacity(Tile.Tile8,tree),capacity_same) : {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool} capacity_room,
          group_listed(~Group,group,parameters,rest,cells,listed)),
        Equal.cong(Tile.Tile8,Tile.Tile8,current => groups_at(~Group,~Sample,~at,~combine,rest,first,count,cells,U32.from_nat(1n+group),reader,query,current),
          Tree.read(Tile.Tile8,loop_tree,query),Updates.updated_at(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Sample,~Kernel.Grouped<Group>,~placed(~Group,~Sample,~at),~combine,~Kernel.place(~Group),U32.to_nat(Kernel.count_of(~Group,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)))),0,reader,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)),query,Tree.read(Tile.Tile8,tree,query)),read_loop))

# ---- The one group that writes a cell ----

# The loop of another group leaves the cell.
def group_kept(~Group: Data,~Sample: Data,~at: Group -> Tree.Tree<Tile.Tile8> -> U32 -> Sample,~combine: Tile.Tile8 -> Sample -> Tile.Tile8,+parameters: Group,+first: U32,+count: U32,+cells: U32,+room: Room<count,cells>,+group: Nat,+below: {Nat.is_lt(group,U32.to_nat(cells)) == True{} : Bool},
  +reader: Tree.Tree<Tile.Tile8>,+query: U32,+value: Nat,+query_exact: {U32.to_nat(query) == value : Nat},+other: {Nat.is_eq(group,residue_group(first,cells,value)) == False{} : Bool},+current: Tile.Tile8) ->
  {Updates.updated_at(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Sample,~Kernel.Grouped<Group>,~placed(~Group,~Sample,~at),~combine,~Kernel.place(~Group),U32.to_nat(Kernel.count_of(~Group,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)))),0,reader,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)),query,current) == current : Tile.Tile8}:
  Room{+cap,+cap_ok,+double_room,+positive,+count_room} = room
  +whole = {Room{cap,cap_ok,double_room,positive,count_room} : Room<count,cells>}
  +inner = U32.to_nat(cells)
  +strides = group_strides(first,count,cells,whole,group,below)
  +start = group_start(first,cells,group)
  +word = group_word(first,count,cells,group)
  +loops = taken(start,cells,U32.to_nat(count),strides)
  +misses = {%Equal.sym(Nat,U32.to_nat(word),loops,group_count(first,count,cells,whole,group,below)) :
      Updates.Misses(~Kernel.Grouped<Group>,~Kernel.place(~Group),_,U32.from_nat(0n),BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)),query)
      residue_misses(~Group,loops,0n,parameters,start,cells,word,U32.to_nat(count),strides,query,value,query_exact,
        gap_other(Nat.mod(U32.to_nat(first),inner),Nat.mod(value,inner),inner,group,positive,DivisionOrder.remainder_small(U32.to_nat(first),inner,positive),below,other),
        Order.reflexive(loops)) :
    Updates.Misses(~Kernel.Grouped<Group>,~Kernel.place(~Group),U32.to_nat(Kernel.count_of(~Group,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)))),0,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)),query)}
  Updates.missed(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Sample,~Kernel.Grouped<Group>,~placed(~Group,~Sample,~at),~combine,~Kernel.place(~Group),U32.to_nat(Kernel.count_of(~Group,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)))),0,reader,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)),query,current,misses)

# The loop of the group of the cell writes it.
def group_hit(~Group: Data,~Sample: Data,~at: Group -> Tree.Tree<Tile.Tile8> -> U32 -> Sample,~combine: Tile.Tile8 -> Sample -> Tile.Tile8,+parameters: Group,+first: U32,+count: U32,+cells: U32,+room: Room<count,cells>,+group: Nat,+below: {Nat.is_lt(group,U32.to_nat(cells)) == True{} : Bool},
  +reader: Tree.Tree<Tile.Tile8>,+query: U32,+value: Nat,+query_exact: {U32.to_nat(query) == value : Nat},+inside: {Nat.is_lt(value,U32.to_nat(count)) == True{} : Bool},
  +same: {group == residue_group(first,cells,value) : Nat},+current: Tile.Tile8) ->
  {Updates.updated_at(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Sample,~Kernel.Grouped<Group>,~placed(~Group,~Sample,~at),~combine,~Kernel.place(~Group),U32.to_nat(Kernel.count_of(~Group,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)))),0,reader,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)),query,current) == combine(current,at(parameters,reader,query)) : Tile.Tile8}:
  Room{+cap,+cap_ok,+double_room,+positive,+count_room} = room
  +whole = {Room{cap,cap_ok,double_room,positive,count_room} : Room<count,cells>}
  +inner = U32.to_nat(cells)
  +residue = Nat.mod(U32.to_nat(first),inner)
  +offset = Nat.mod(value,inner)
  +strides = group_strides(first,count,cells,whole,group,below)
  +start = group_start(first,cells,group)
  +word = group_word(first,count,cells,group)
  +loops = taken(start,cells,U32.to_nat(count),strides)
  +hit = Nat.div(value,inner)
  +matched = {%Equal.sym(Nat,group,residue_group(first,cells,value),same) : {offset == LineChainValues.gap_at(_,inner,residue) : Nat}
      Equal.sym(Nat,LineChainValues.gap_at(residue_group(first,cells,value),inner,residue),offset,
        gap_sum(residue,offset,inner,positive,DivisionOrder.remainder_small(U32.to_nat(first),inner,positive),DivisionOrder.remainder_small(value,inner,positive))) :
    {Nat.mod(value,inner) == group_gap(first,cells,group) : Nat}}
  +located = address_of(value,inner,group_gap(first,cells,group),positive,matched)
  +hit_inside = taken_reached(value,group_gap(first,cells,group),inner,U32.to_nat(count),positive,matched,inside)
  +cell_exact = Equal.trans(Nat,U32.to_nat(query),value,cell_of(start,cells,U32.to_nat(count),strides,hit),query_exact,located)
  +hits = {%Equal.sym(Nat,U32.to_nat(word),loops,group_count(first,count,cells,whole,group,below)) :
      Updates.Hits(~Kernel.Grouped<Group>,~Kernel.place(~Group),_,U32.from_nat(0n),BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)),query,U32.from_nat(hit),hit)
      hits_from(~Group,loops,0n,hit,parameters,start,cells,word,U32.to_nat(count),strides,query,hit,cell_exact,{==},hit_inside,Order.reflexive(loops)) :
    Updates.Hits(~Kernel.Grouped<Group>,~Kernel.place(~Group),U32.to_nat(Kernel.count_of(~Group,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)))),0,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)),query,U32.from_nat(hit),hit)}
  +placed_exact = Arithmetic.to_nat_injective(Kernel.place(~Group,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)),U32.from_nat(hit)),query,
    Equal.trans(Nat,U32.to_nat(Kernel.place(~Group,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)),U32.from_nat(hit))),cell_of(start,cells,U32.to_nat(count),strides,hit),U32.to_nat(query),
      position_exact(~Group,parameters,start,cells,word,U32.to_nat(count),strides,hit,hit_inside),Equal.sym(Nat,U32.to_nat(query),cell_of(start,cells,U32.to_nat(count),strides,hit),cell_exact)))
  %placed_exact : {Updates.updated_at(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Sample,~Kernel.Grouped<Group>,~placed(~Group,~Sample,~at),~combine,~Kernel.place(~Group),U32.to_nat(Kernel.count_of(~Group,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)))),0,reader,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)),query,current) == combine(current,at(parameters,reader,_)) : Tile.Tile8}
  Updates.hit(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Sample,~Kernel.Grouped<Group>,~placed(~Group,~Sample,~at),~combine,~Kernel.place(~Group),U32.to_nat(Kernel.count_of(~Group,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)))),0,reader,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)),query,U32.from_nat(hit),hit,current,hits)

# The groups after the one of the cell leave it.
def groups_missed(~Group: Data,~Sample: Data,~at: Group -> Tree.Tree<Tile.Tile8> -> U32 -> Sample,~combine: Tile.Tile8 -> Sample -> Tile.Tile8,groups: List<&2,Group>,+group: Nat,+first: U32,+count: U32,+cells: U32,+room: Room<count,cells>,+reader: Tree.Tree<Tile.Tile8>,+query: U32,+value: Nat,
  +query_exact: {U32.to_nat(query) == value : Nat},+current: Tile.Tile8,+later: {Nat.is_lt(residue_group(first,cells,value),group) == True{} : Bool},
  +listed: {Nat.add(group,List.length(&2,Group,groups)) == U32.to_nat(cells) : Nat}) ->
  {groups_at(~Group,~Sample,~at,~combine,groups,first,count,cells,U32.from_nat(group),reader,query,current) == current : Tile.Tile8}:
  match groups:
    case Nil{}: {==}
    case Con{+parameters,+rest}:
      +below = group_below(~Group,group,parameters,rest,cells,listed)
      +other = {%Order.equal_symmetric(residue_group(first,cells,value),group) : {_ == False{} : Bool}
        Order.less_not_equal(residue_group(first,cells,value),group,later) : {Nat.is_eq(group,residue_group(first,cells,value)) == False{} : Bool}}
      %Equal.sym(U32,(U32.from_nat(group) + 1 : U32),U32.from_nat(1n+group),Bounds.increment_encoded(group)) :
        {groups_at(~Group,~Sample,~at,~combine,rest,first,count,cells,_,reader,query,Updates.updated_at(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Sample,~Kernel.Grouped<Group>,~placed(~Group,~Sample,~at),~combine,~Kernel.place(~Group),U32.to_nat(Kernel.count_of(~Group,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)))),0,reader,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)),query,current)) == current : Tile.Tile8}
      %Equal.sym(Tile.Tile8,Updates.updated_at(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Sample,~Kernel.Grouped<Group>,~placed(~Group,~Sample,~at),~combine,~Kernel.place(~Group),U32.to_nat(Kernel.count_of(~Group,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)))),0,reader,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)),query,current),current,group_kept(~Group,~Sample,~at,~combine,parameters,first,count,cells,room,group,below,reader,query,value,query_exact,other,current)) :
        {groups_at(~Group,~Sample,~at,~combine,rest,first,count,cells,U32.from_nat(1n+group),reader,query,_) == current : Tile.Tile8}
      groups_missed(~Group,~Sample,~at,~combine,rest,1n+group,first,count,cells,room,reader,query,value,query_exact,current,
        Order.le_lt_trans(residue_group(first,cells,value),group,1n+group,Bounds.strict_le(residue_group(first,cells,value),group,later),Order.lt_succ(group,group,Order.reflexive(group))),
        group_listed(~Group,group,parameters,rest,cells,listed))

# The value a group combines into the current one.
def combined(~Group: Data,~Sample: Data,~at: Group -> Tree.Tree<Tile.Tile8> -> U32 -> Sample,~combine: Tile.Tile8 -> Sample -> Tile.Tile8,+reader: Tree.Tree<Tile.Tile8>,+query: U32,+current: Tile.Tile8,parameters: Group) -> Tile.Tile8:
  combine(current,at(parameters,reader,query))

# What the group before groups on makes, or fallback past the groups.
def picked(~Group: Data,~Result: Data,~make: Group -> Result,groups: List<&2,Group>,before: Nat,+fallback: Result) -> Result:
  match groups:
    case Nil{}: fallback
    case Con{parameters,rest}:
      match before:
        case 0n: make(parameters)
        case 1n+earlier: picked(~Group,~Result,~make,rest,earlier,fallback)

# The groups from group on leave at a cell what its own group makes of it.
def groups_hit(~Group: Data,~Sample: Data,~at: Group -> Tree.Tree<Tile.Tile8> -> U32 -> Sample,~combine: Tile.Tile8 -> Sample -> Tile.Tile8,groups: List<&2,Group>,+group: Nat,before: Nat,+first: U32,+count: U32,+cells: U32,+room: Room<count,cells>,+reader: Tree.Tree<Tile.Tile8>,+query: U32,+value: Nat,
  +query_exact: {U32.to_nat(query) == value : Nat},+inside: {Nat.is_lt(value,U32.to_nat(count)) == True{} : Bool},+current: Tile.Tile8,
  +reach: {Nat.add(group,before) == residue_group(first,cells,value) : Nat},+listed: {Nat.add(group,List.length(&2,Group,groups)) == U32.to_nat(cells) : Nat}) ->
  {groups_at(~Group,~Sample,~at,~combine,groups,first,count,cells,U32.from_nat(group),reader,query,current) == picked(~Group,~Tile.Tile8,~combined(~Group,~Sample,~at,~combine,reader,query,current),groups,before,current) : Tile.Tile8}:
  match groups:
    case Nil{}: {==}
    case Con{+parameters,+rest}:
      match before:
        case 0n:
          +below = group_below(~Group,group,parameters,rest,cells,listed)
          +same = Equal.trans(Nat,group,Nat.add(group,0n),residue_group(first,cells,value),Equal.sym(Nat,Nat.add(group,0n),group,Addition.zero_right(group)),reach)
          +value_now = combine(current,at(parameters,reader,query))
          %Equal.sym(U32,(U32.from_nat(group) + 1 : U32),U32.from_nat(1n+group),Bounds.increment_encoded(group)) :
            {groups_at(~Group,~Sample,~at,~combine,rest,first,count,cells,_,reader,query,Updates.updated_at(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Sample,~Kernel.Grouped<Group>,~placed(~Group,~Sample,~at),~combine,~Kernel.place(~Group),U32.to_nat(Kernel.count_of(~Group,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)))),0,reader,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)),query,current)) == value_now : Tile.Tile8}
          %Equal.sym(Tile.Tile8,Updates.updated_at(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Sample,~Kernel.Grouped<Group>,~placed(~Group,~Sample,~at),~combine,~Kernel.place(~Group),U32.to_nat(Kernel.count_of(~Group,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)))),0,reader,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)),query,current),value_now,group_hit(~Group,~Sample,~at,~combine,parameters,first,count,cells,room,group,below,reader,query,value,query_exact,inside,same,current)) :
            {groups_at(~Group,~Sample,~at,~combine,rest,first,count,cells,U32.from_nat(1n+group),reader,query,_) == value_now : Tile.Tile8}
          groups_missed(~Group,~Sample,~at,~combine,rest,1n+group,first,count,cells,room,reader,query,value,query_exact,value_now,
            %same : {Nat.is_lt(_,1n+group) == True{} : Bool} Order.lt_succ(group,group,Order.reflexive(group)),group_listed(~Group,group,parameters,rest,cells,listed))
        case 1n+ +earlier:
          +below = group_below(~Group,group,parameters,rest,cells,listed)
          +ahead = Equal.trans(Nat,Nat.add(1n+group,earlier),Nat.add(group,1n+earlier),residue_group(first,cells,value),
            Equal.sym(Nat,Nat.add(group,1n+earlier),1n+Nat.add(group,earlier),Addition.successor_right(group,earlier)),reach)
          +sooner = {%reach : {Nat.is_lt(group,_) == True{} : Bool} Order.below_sum(group,earlier) : {Nat.is_lt(group,residue_group(first,cells,value)) == True{} : Bool}}
          %Equal.sym(U32,(U32.from_nat(group) + 1 : U32),U32.from_nat(1n+group),Bounds.increment_encoded(group)) :
            {groups_at(~Group,~Sample,~at,~combine,rest,first,count,cells,_,reader,query,Updates.updated_at(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Sample,~Kernel.Grouped<Group>,~placed(~Group,~Sample,~at),~combine,~Kernel.place(~Group),U32.to_nat(Kernel.count_of(~Group,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)))),0,reader,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)),query,current)) == picked(~Group,~Tile.Tile8,~combined(~Group,~Sample,~at,~combine,reader,query,current),rest,earlier,current) : Tile.Tile8}
          %Equal.sym(Tile.Tile8,Updates.updated_at(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Sample,~Kernel.Grouped<Group>,~placed(~Group,~Sample,~at),~combine,~Kernel.place(~Group),U32.to_nat(Kernel.count_of(~Group,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)))),0,reader,BNCells.grouped(~Group,parameters,first,count,cells,U32.from_nat(group)),query,current),current,group_kept(~Group,~Sample,~at,~combine,parameters,first,count,cells,room,group,below,reader,query,value,query_exact,
              Order.less_not_equal(group,residue_group(first,cells,value),sooner),current)) :
            {groups_at(~Group,~Sample,~at,~combine,rest,first,count,cells,U32.from_nat(1n+group),reader,query,_) == picked(~Group,~Tile.Tile8,~combined(~Group,~Sample,~at,~combine,reader,query,current),rest,earlier,current) : Tile.Tile8}
          groups_hit(~Group,~Sample,~at,~combine,rest,1n+group,earlier,first,count,cells,room,reader,query,value,query_exact,inside,current,ahead,group_listed(~Group,group,parameters,rest,cells,listed))

# A cell of the piece ends as its group makes it of its value.
def piece_read(~Group: Data,~Sample: Data,~at: Group -> Tree.Tree<Tile.Tile8> -> U32 -> Sample,~combine: Tile.Tile8 -> Sample -> Tile.Tile8,+groups: List<&2,Group>,+first: U32,+count: U32,+cells: U32,+room: Room<count,cells>,+reader: Tree.Tree<Tile.Tile8>,+tree: Tree.Tree<Tile.Tile8>,+depth: Nat,+query: U32,+value: Nat,
  +query_exact: {U32.to_nat(query) == value : Nat},+inside: {Nat.is_lt(value,U32.to_nat(count)) == True{} : Bool},
  +balanced: Shape.Balanced(Tile.Tile8,depth,tree),+limit: {Nat.is_le(depth,31n) == True{} : Bool},+query_inside: {U32.is_lt(query,Tree.capacity(Tile.Tile8,tree)) == True{} : Bool},
  +capacity_room: {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Tile.Tile8,tree))) == True{} : Bool},+listed: {List.length(&2,Group,groups) == U32.to_nat(cells) : Nat}) ->
  {Tree.read(Tile.Tile8,groups_tree(~Group,~Sample,~at,~combine,groups,first,count,cells,0,reader,tree),query) ==
    picked(~Group,~Tile.Tile8,~combined(~Group,~Sample,~at,~combine,reader,query,Tree.read(Tile.Tile8,tree,query)),groups,residue_group(first,cells,value),Tree.read(Tile.Tile8,tree,query)) : Tile.Tile8}:
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,groups_tree(~Group,~Sample,~at,~combine,groups,first,count,cells,U32.from_nat(0n),reader,tree),query),
    groups_at(~Group,~Sample,~at,~combine,groups,first,count,cells,U32.from_nat(0n),reader,query,Tree.read(Tile.Tile8,tree,query)),
    picked(~Group,~Tile.Tile8,~combined(~Group,~Sample,~at,~combine,reader,query,Tree.read(Tile.Tile8,tree,query)),groups,residue_group(first,cells,value),Tree.read(Tile.Tile8,tree,query)),
    groups_read(~Group,~Sample,~at,~combine,groups,0n,first,count,cells,room,reader,tree,depth,query,balanced,limit,query_inside,capacity_room,listed),
    groups_hit(~Group,~Sample,~at,~combine,groups,0n,residue_group(first,cells,value),first,count,cells,room,reader,query,value,query_exact,inside,Tree.read(Tile.Tile8,tree,query),{==},listed))

# ---- The buffers of a piece ----

def Contents() -> Data:
  Buffer.Content<Tile.Tile8>

def pair_contents(pair: BNCells.Cells() & BNCells.Cells()) -> Contents() & Contents():
  (left,right) = pair
  (Buffer.contents(~Tile.Tile8,left),Buffer.contents(~Tile.Tile8,right))

# The loop of a group on the cells of content, reading the cells of reader.
def loop_content(~Group: Data,~Sample: Data,~at: Group -> Tree.Tree<Tile.Tile8> -> U32 -> Sample,~combine: Tile.Tile8 -> Sample -> Tile.Tile8,+context: Kernel.Grouped<Group>,+reader: Contents(),content: Contents()) -> Contents():
  Buffer.Content{+tree,length} = content
  Buffer.Content{group_loop(~Group,~Sample,~at,~combine,context,Buffer.content_tree(~Tile.Tile8,reader),tree),length}

# The loops of the groups of a piece from group on.
def groups_content(~Group: Data,~Sample: Data,~at: Group -> Tree.Tree<Tile.Tile8> -> U32 -> Sample,~combine: Tile.Tile8 -> Sample -> Tile.Tile8,groups: List<&2,Group>,+first: U32,+count: U32,+cells: U32,+group: U32,+reader: Contents(),content: Contents()) -> Contents():
  match groups:
    case Nil{}: content
    case Con{+parameters,rest}:
      groups_content(~Group,~Sample,~at,~combine,rest,first,count,cells,(group + 1 : U32),reader,loop_content(~Group,~Sample,~at,~combine,BNCells.grouped(~Group,parameters,first,count,cells,group),reader,content))

# A step hands back its reader and runs the loop of its group on the other
# buffer.
def StepLaw(~Group: Data,~Sample: Data,~at: Group -> Tree.Tree<Tile.Tile8> -> U32 -> Sample,~combine: Tile.Tile8 -> Sample -> Tile.Tile8,~step: Kernel.Grouped<Group> -> BNCells.Cells() -> BNCells.Cells() -> BNCells.Cells() & BNCells.Cells()) -> Type:
  @+context: Kernel.Grouped<Group> -> @+left: Buffer.Model<Tile.Tile8> -> @+right: Buffer.Model<Tile.Tile8> ->
    {pair_contents(step(context,Buffer.pack(~Tile.Tile8,left),Buffer.pack(~Tile.Tile8,right))) == (Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,left)),loop_content(~Group,~Sample,~at,~combine,context,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,left)),Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,right)))) : Contents() & Contents()}

# The groups after a step go on from the buffers it hands back.
def bound_contents(~Group: Data,~Sample: Data,~at: Group -> Tree.Tree<Tile.Tile8> -> U32 -> Sample,~combine: Tile.Tile8 -> Sample -> Tile.Tile8,~step: Kernel.Grouped<Group> -> BNCells.Cells() -> BNCells.Cells() -> BNCells.Cells() & BNCells.Cells(),rest: List<&2,Group>,+first: U32,+count: U32,+cells: U32,+group: U32,
  hypothesis: @-left: BNCells.Cells() -> @-right: BNCells.Cells() -> @+left_seen: Buffer.Observation<Tile.Tile8,left> -> @+right_seen: Buffer.Observation<Tile.Tile8,right> ->
    {pair_contents(BNCells.each_group(~Group,~step,rest,first,count,cells,group,left,right)) ==
      (Buffer.contents(~Tile.Tile8,left),groups_content(~Group,~Sample,~at,~combine,rest,first,count,cells,group,Buffer.contents(~Tile.Tile8,left),Buffer.contents(~Tile.Tile8,right))) : Contents() & Contents()},
  pair: BNCells.Cells() & BNCells.Cells()) ->
  {pair_contents(Storage.bind_pair(~BNCells.Cells(),~BNCells.Cells(),~(BNCells.Cells() & BNCells.Cells()),pair,left => right => BNCells.each_group(~Group,~step,rest,first,count,cells,group,left,right))) ==
    (Pair.fst(Contents(),Contents(),pair_contents(pair)),
      groups_content(~Group,~Sample,~at,~combine,rest,first,count,cells,group,Pair.fst(Contents(),Contents(),pair_contents(pair)),Pair.snd(Contents(),Contents(),pair_contents(pair)))) : Contents() & Contents()}:
  (left,right) = pair
  hypothesis(left,right,Buffer.observe(~Tile.Tile8,left),Buffer.observe(~Tile.Tile8,right))

# The groups of a piece from group on leave the reader and run their loops on
# the other buffer.
def each_contents(~Group: Data,~Sample: Data,~at: Group -> Tree.Tree<Tile.Tile8> -> U32 -> Sample,~combine: Tile.Tile8 -> Sample -> Tile.Tile8,~step: Kernel.Grouped<Group> -> BNCells.Cells() -> BNCells.Cells() -> BNCells.Cells() & BNCells.Cells(),~stepped: StepLaw(~Group,~Sample,~at,~combine,~step),groups: List<&2,Group>,+first: U32,+count: U32,+cells: U32,+group: U32,-left: BNCells.Cells(),-right: BNCells.Cells(),
  +left_seen: Buffer.Observation<Tile.Tile8,left>,+right_seen: Buffer.Observation<Tile.Tile8,right>) ->
  {pair_contents(BNCells.each_group(~Group,~step,groups,first,count,cells,group,left,right)) ==
    (Buffer.contents(~Tile.Tile8,left),groups_content(~Group,~Sample,~at,~combine,groups,first,count,cells,group,Buffer.contents(~Tile.Tile8,left),Buffer.contents(~Tile.Tile8,right))) : Contents() & Contents()}:
  match groups:
    case Nil{}: {==}
    case Con{+parameters,+rest}:
      Buffer.Observation{+left_model,+left_equation} = left_seen
      Buffer.Observation{+right_model,+right_equation} = right_seen
      +context = BNCells.grouped(~Group,parameters,first,count,cells,group)
      %left_equation :
        {pair_contents(Storage.bind_pair(~BNCells.Cells(),~BNCells.Cells(),~(BNCells.Cells() & BNCells.Cells()),step(context,_,right),left => right => BNCells.each_group(~Group,~step,rest,first,count,cells,(group + 1 : U32),left,right))) ==
          (Buffer.contents(~Tile.Tile8,_),groups_content(~Group,~Sample,~at,~combine,rest,first,count,cells,(group + 1 : U32),Buffer.contents(~Tile.Tile8,_),loop_content(~Group,~Sample,~at,~combine,context,Buffer.contents(~Tile.Tile8,_),Buffer.contents(~Tile.Tile8,right)))) : Contents() & Contents()}
      %right_equation :
        {pair_contents(Storage.bind_pair(~BNCells.Cells(),~BNCells.Cells(),~(BNCells.Cells() & BNCells.Cells()),step(context,Buffer.pack(~Tile.Tile8,left_model),_),left => right => BNCells.each_group(~Group,~step,rest,first,count,cells,(group + 1 : U32),left,right))) ==
          (Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,left_model)),groups_content(~Group,~Sample,~at,~combine,rest,first,count,cells,(group + 1 : U32),Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,left_model)),loop_content(~Group,~Sample,~at,~combine,context,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,left_model)),Buffer.contents(~Tile.Tile8,_)))) : Contents() & Contents()}
      %stepped(context,left_model,right_model) :
        {pair_contents(Storage.bind_pair(~BNCells.Cells(),~BNCells.Cells(),~(BNCells.Cells() & BNCells.Cells()),step(context,Buffer.pack(~Tile.Tile8,left_model),Buffer.pack(~Tile.Tile8,right_model)),left => right => BNCells.each_group(~Group,~step,rest,first,count,cells,(group + 1 : U32),left,right))) ==
          (Pair.fst(Contents(),Contents(),_),groups_content(~Group,~Sample,~at,~combine,rest,first,count,cells,(group + 1 : U32),Pair.fst(Contents(),Contents(),_),Pair.snd(Contents(),Contents(),_))) : Contents() & Contents()}
      bound_contents(~Group,~Sample,~at,~combine,~step,rest,first,count,cells,(group + 1 : U32),
        left => right => left_seen => right_seen => each_contents(~Group,~Sample,~at,~combine,~step,~stepped,rest,first,count,cells,(group + 1 : U32),left,right,left_seen,right_seen),
        step(context,Buffer.pack(~Tile.Tile8,left_model),Buffer.pack(~Tile.Tile8,right_model)))

def loop_tree(~Group: Data,~Sample: Data,~at: Group -> Tree.Tree<Tile.Tile8> -> U32 -> Sample,~combine: Tile.Tile8 -> Sample -> Tile.Tile8,+context: Kernel.Grouped<Group>,+reader: Contents(),+content: Contents()) ->
  {Buffer.content_tree(~Tile.Tile8,loop_content(~Group,~Sample,~at,~combine,context,reader,content)) ==
    group_loop(~Group,~Sample,~at,~combine,context,Buffer.content_tree(~Tile.Tile8,reader),Buffer.content_tree(~Tile.Tile8,content)) : Tree.Tree<Tile.Tile8>}:
  Buffer.Content{+tree,+length} = content
  {==}

def loop_length(~Group: Data,~Sample: Data,~at: Group -> Tree.Tree<Tile.Tile8> -> U32 -> Sample,~combine: Tile.Tile8 -> Sample -> Tile.Tile8,+context: Kernel.Grouped<Group>,+reader: Contents(),+content: Contents()) ->
  {Buffer.content_length(~Tile.Tile8,loop_content(~Group,~Sample,~at,~combine,context,reader,content)) == Buffer.content_length(~Tile.Tile8,content) : Nat}:
  Buffer.Content{+tree,+length} = content
  {==}

# The cells the loops leave are the cells of the loops on trees.
def groups_content_tree(~Group: Data,~Sample: Data,~at: Group -> Tree.Tree<Tile.Tile8> -> U32 -> Sample,~combine: Tile.Tile8 -> Sample -> Tile.Tile8,groups: List<&2,Group>,+first: U32,+count: U32,+cells: U32,+group: U32,+reader: Contents(),+content: Contents()) ->
  {Buffer.content_tree(~Tile.Tile8,groups_content(~Group,~Sample,~at,~combine,groups,first,count,cells,group,reader,content)) ==
    groups_tree(~Group,~Sample,~at,~combine,groups,first,count,cells,group,Buffer.content_tree(~Tile.Tile8,reader),Buffer.content_tree(~Tile.Tile8,content)) : Tree.Tree<Tile.Tile8>}:
  match groups:
    case Nil{}: {==}
    case Con{+parameters,rest}:
      +context = BNCells.grouped(~Group,parameters,first,count,cells,group)
      %loop_tree(~Group,~Sample,~at,~combine,context,reader,content) :
        {Buffer.content_tree(~Tile.Tile8,groups_content(~Group,~Sample,~at,~combine,rest,first,count,cells,(group + 1 : U32),reader,loop_content(~Group,~Sample,~at,~combine,context,reader,content))) ==
          groups_tree(~Group,~Sample,~at,~combine,rest,first,count,cells,(group + 1 : U32),Buffer.content_tree(~Tile.Tile8,reader),_) : Tree.Tree<Tile.Tile8>}
      groups_content_tree(~Group,~Sample,~at,~combine,rest,first,count,cells,(group + 1 : U32),reader,loop_content(~Group,~Sample,~at,~combine,context,reader,content))

def groups_content_length(~Group: Data,~Sample: Data,~at: Group -> Tree.Tree<Tile.Tile8> -> U32 -> Sample,~combine: Tile.Tile8 -> Sample -> Tile.Tile8,groups: List<&2,Group>,+first: U32,+count: U32,+cells: U32,+group: U32,+reader: Contents(),+content: Contents()) ->
  {Buffer.content_length(~Tile.Tile8,groups_content(~Group,~Sample,~at,~combine,groups,first,count,cells,group,reader,content)) == Buffer.content_length(~Tile.Tile8,content) : Nat}:
  match groups:
    case Nil{}: {==}
    case Con{+parameters,rest}:
      +context = BNCells.grouped(~Group,parameters,first,count,cells,group)
      Equal.trans(Nat,Buffer.content_length(~Tile.Tile8,groups_content(~Group,~Sample,~at,~combine,rest,first,count,cells,(group + 1 : U32),reader,loop_content(~Group,~Sample,~at,~combine,context,reader,content))),
        Buffer.content_length(~Tile.Tile8,loop_content(~Group,~Sample,~at,~combine,context,reader,content)),Buffer.content_length(~Tile.Tile8,content),
        groups_content_length(~Group,~Sample,~at,~combine,rest,first,count,cells,(group + 1 : U32),reader,loop_content(~Group,~Sample,~at,~combine,context,reader,content)),
        loop_length(~Group,~Sample,~at,~combine,context,reader,content))

# ---- The two steps ----

def arrays_contents(+input_length: Nat,+output_length: Nat,result: Array<Tile.Tile8> & Array<Tile.Tile8>) -> Contents() & Contents():
  (input,output) = result
  (Buffer.Content{Tree.reflect(Tile.Tile8,input),input_length},Buffer.Content{Tree.reflect(Tile.Tile8,output),output_length})

# Two buffers made of two arrays have the contents of the arrays.
def buffers_law(+input_length: Nat,+output_length: Nat,result: Array<Tile.Tile8> & Array<Tile.Tile8>,
  certificates: TraversalCertificate.Certificates<Array<Tile.Tile8>,Tile.Tile8,Certificate.family(~Tile.Tile8),result>,-target: Contents() & Contents(),
  arrays: {arrays_contents(input_length,output_length,result) == target : Contents() & Contents()}) ->
  {pair_contents(BNCells.buffers_of(input_length,output_length,result,certificates)) == target : Contents() & Contents()}:
  (input,output) = result
  TraversalCertificate.Certificates{input_certificate,output_certificate} = certificates
  arrays

# The arrays of a model state are the trees of the state.
def represented_contents(~Context: Data,+input_length: Nat,+output_length: Nat,+state: Traversal.State<&2,&2,Tree.Tree<Tile.Tile8>,Tree.Tree<Tile.Tile8>,Context>) ->
  {arrays_contents(input_length,output_length,Traversal.finish(~Array<Tile.Tile8>,~Tile.Tile8,~Context,
      ArrayLaws.represent(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Context,~ArrayLaws.pack_tree(~Tile.Tile8),state))) ==
    (Buffer.Content{ArrayLaws.model_reader(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Context,state),input_length},Buffer.Content{Updates.state_tree(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Context,state),output_length}) : Contents() & Contents()}:
  Traversal.State{+reader,+output,position,context} = state
  %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,reader)),reader,StorageLaws.reflect_packed(Tile.Tile8,reader)) :
    {(Buffer.Content{_,input_length},Buffer.Content{Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,output)),output_length}) == (Buffer.Content{reader,input_length},Buffer.Content{output,output_length}) : Contents() & Contents()}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,output)),output,StorageLaws.reflect_packed(Tile.Tile8,output)) :
    {(Buffer.Content{reader,input_length},Buffer.Content{_,output_length}) == (Buffer.Content{reader,input_length},Buffer.Content{output,output_length}) : Contents() & Contents()}
  {==}

# The sample of the cell at an address, by the parameters of a group.
def sample_at(~Group: Data,~Sample: Data,~sample: Group -> Tile.Tile8 -> Sample,group: Group,tree: Tree.Tree<Tile.Tile8>,address: U32) -> Sample:
  sample(group,Tree.read(Tile.Tile8,tree,address))

def output_of(group: Kernel.Normal,tile: Tile.Tile8) -> Tile.Tile8:
  +group = group
  Kernel.output(group,tile)

def back_of(group: Kernel.Back,tile: Tile.Tile8) -> Kernel.Sampled:
  Kernel.Back{beta,+gamma,scale,total} = group
  Kernel.Sampled{Tile.times(tile,gamma),Kernel.Back{beta,gamma,scale,total}}

def normal_matches(+reader: Tree.Tree<Tile.Tile8>,+context: Kernel.Grouped<Kernel.Normal>,+position: U32) ->
  {Kernel.read_normalized(ArrayLaws.pack_tree(~Tile.Tile8,reader),context,position) ==
    (ArrayLaws.pack_tree(~Tile.Tile8,reader),placed(~Kernel.Normal,~Tile.Tile8,~sample_at(~Kernel.Normal,~Tile.Tile8,~Kernel.normalized),reader,context,position)) : Array<Tile.Tile8> & Tile.Tile8}:
  Kernel.Grouped{+group,+start,+step,count} = context
  +address = (start + position * step : U32)
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,Tree.pack(Tile.Tile8,reader),address),(Tree.pack(Tile.Tile8,reader),Tree.read(Tile.Tile8,reader,address)),StorageLaws.read(Tile.Tile8,reader,address)) :
    {Kernel.normal_cell(group,_) == (Tree.pack(Tile.Tile8,reader),Kernel.normalized(group,Tree.read(Tile.Tile8,reader,address))) : Array<Tile.Tile8> & Tile.Tile8}
  {==}

def output_matches(+reader: Tree.Tree<Tile.Tile8>,+context: Kernel.Grouped<Kernel.Normal>,+position: U32) ->
  {Kernel.read_output(ArrayLaws.pack_tree(~Tile.Tile8,reader),context,position) ==
    (ArrayLaws.pack_tree(~Tile.Tile8,reader),placed(~Kernel.Normal,~Tile.Tile8,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of),reader,context,position)) : Array<Tile.Tile8> & Tile.Tile8}:
  Kernel.Grouped{+group,+start,+step,count} = context
  +address = (start + position * step : U32)
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,Tree.pack(Tile.Tile8,reader),address),(Tree.pack(Tile.Tile8,reader),Tree.read(Tile.Tile8,reader,address)),StorageLaws.read(Tile.Tile8,reader,address)) :
    {Kernel.output_cell(group,_) == (Tree.pack(Tile.Tile8,reader),output_of(group,Tree.read(Tile.Tile8,reader,address))) : Array<Tile.Tile8> & Tile.Tile8}
  {==}

def back_matches(+reader: Tree.Tree<Tile.Tile8>,+context: Kernel.Grouped<Kernel.Back>,+position: U32) ->
  {Kernel.back_read(ArrayLaws.pack_tree(~Tile.Tile8,reader),context,position) ==
    (ArrayLaws.pack_tree(~Tile.Tile8,reader),placed(~Kernel.Back,~Kernel.Sampled,~sample_at(~Kernel.Back,~Kernel.Sampled,~back_of),reader,context,position)) : Array<Tile.Tile8> & Kernel.Sampled}:
  Kernel.Grouped{+group,+start,+step,count} = context
  Kernel.Back{+beta,+gamma,+scale,+total} = group
  +address = (start + position * step : U32)
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,Tree.pack(Tile.Tile8,reader),address),(Tree.pack(Tile.Tile8,reader),Tree.read(Tile.Tile8,reader,address)),StorageLaws.read(Tile.Tile8,reader,address)) :
    {Kernel.back_sample(Kernel.Back{beta,gamma,scale,total},_) == (Tree.pack(Tile.Tile8,reader),back_of(Kernel.Back{beta,gamma,scale,total},Tree.read(Tile.Tile8,reader,address))) : Array<Tile.Tile8> & Kernel.Sampled}
  {==}

# A loop that reads one array stores what the model of its group samples.
def mapped_arrays(~Group: Data,~read: Array<Tile.Tile8> -> Kernel.Grouped<Group> -> U32 -> Array<Tile.Tile8> & Tile.Tile8,~at: Group -> Tree.Tree<Tile.Tile8> -> U32 -> Tile.Tile8,~read_matches: @+reader: Tree.Tree<Tile.Tile8> -> @+context: Kernel.Grouped<Group> -> @+position: U32 -> {read(ArrayLaws.pack_tree(~Tile.Tile8,reader),context,position) == (ArrayLaws.pack_tree(~Tile.Tile8,reader),placed(~Group,~Tile.Tile8,~at,reader,context,position)) : Array<Tile.Tile8> & Tile.Tile8},+context: Kernel.Grouped<Group>,+input: Tree.Tree<Tile.Tile8>,+input_length: Nat,+output: Tree.Tree<Tile.Tile8>,+output_length: Nat) ->
  {arrays_contents(input_length,output_length,Kernel.mapped(~Array<Tile.Tile8>,~Group,~read,context,Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,output))) ==
    (Buffer.Content{Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,input)),input_length},loop_content(~Group,~Tile.Tile8,~at,~Updates.kept_sample(~Tile.Tile8),context,Buffer.Content{Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,input)),input_length},Buffer.Content{Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,output)),output_length})) : Contents() & Contents()}:
  +count = U32.to_nat(Kernel.count_of(~Group,context))
  +start = {Traversal.State{input,output,0,context} : Traversal.State<&2,&2,Tree.Tree<Tile.Tile8>,Tree.Tree<Tile.Tile8>,Kernel.Grouped<Group>>}
  +walked = Loop.run(~Traversal.State<&2,&2,Tree.Tree<Tile.Tile8>,Tree.Tree<Tile.Tile8>,Kernel.Grouped<Group>>,~ArrayLaws.model_step(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Group>,~placed(~Group,~Tile.Tile8,~at),~Kernel.place(~Group)),count,start)
  %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,input)),input,StorageLaws.reflect_packed(Tile.Tile8,input)) :
    {arrays_contents(input_length,output_length,Kernel.mapped(~Array<Tile.Tile8>,~Group,~read,context,Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,output))) ==
      (Buffer.Content{_,input_length},loop_content(~Group,~Tile.Tile8,~at,~Updates.kept_sample(~Tile.Tile8),context,Buffer.Content{_,input_length},Buffer.Content{Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,output)),output_length})) : Contents() & Contents()}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,output)),output,StorageLaws.reflect_packed(Tile.Tile8,output)) :
    {arrays_contents(input_length,output_length,Kernel.mapped(~Array<Tile.Tile8>,~Group,~read,context,Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,output))) ==
      (Buffer.Content{input,input_length},loop_content(~Group,~Tile.Tile8,~at,~Updates.kept_sample(~Tile.Tile8),context,Buffer.Content{input,input_length},Buffer.Content{_,output_length})) : Contents() & Contents()}
  %Equal.sym(Traversal.State<&1,&1,Array<Tile.Tile8>,Array<Tile.Tile8>,Kernel.Grouped<Group>>,Traversal.run(~Array<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Group>,~read,~Kernel.place(~Group),count,ArrayLaws.represent(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Group>,~ArrayLaws.pack_tree(~Tile.Tile8),start)),ArrayLaws.represent(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Group>,~ArrayLaws.pack_tree(~Tile.Tile8),walked),
      ArrayLaws.traversal_refines_model(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Group>,~ArrayLaws.pack_tree(~Tile.Tile8),~read,~placed(~Group,~Tile.Tile8,~at),~Kernel.place(~Group),
        ~read_matches,count,start)) :
    {arrays_contents(input_length,output_length,Traversal.finish(~Array<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Group>,_)) == (Buffer.Content{input,input_length},loop_content(~Group,~Tile.Tile8,~at,~Updates.kept_sample(~Tile.Tile8),context,Buffer.Content{input,input_length},Buffer.Content{output,output_length})) : Contents() & Contents()}
  Equal.trans(Contents() & Contents(),arrays_contents(input_length,output_length,Traversal.finish(~Array<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Group>,ArrayLaws.represent(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Group>,~ArrayLaws.pack_tree(~Tile.Tile8),walked))),
    (Buffer.Content{ArrayLaws.model_reader(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Group>,walked),input_length},Buffer.Content{Updates.state_tree(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Group>,walked),output_length}),
    (Buffer.Content{input,input_length},loop_content(~Group,~Tile.Tile8,~at,~Updates.kept_sample(~Tile.Tile8),context,Buffer.Content{input,input_length},Buffer.Content{output,output_length})),represented_contents(~Kernel.Grouped<Group>,input_length,output_length,walked),
    %Equal.sym(Tree.Tree<Tile.Tile8>,ArrayLaws.model_reader(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Group>,walked),input,
        ArrayLaws.model_reader_kept(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Group>,~placed(~Group,~Tile.Tile8,~at),~Kernel.place(~Group),count,start)) :
      {(Buffer.Content{_,input_length},Buffer.Content{Updates.state_tree(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Group>,walked),output_length}) == (Buffer.Content{input,input_length},loop_content(~Group,~Tile.Tile8,~at,~Updates.kept_sample(~Tile.Tile8),context,Buffer.Content{input,input_length},Buffer.Content{output,output_length})) : Contents() & Contents()}
    %Equal.sym(Traversal.State<&2,&2,Tree.Tree<Tile.Tile8>,Tree.Tree<Tile.Tile8>,Kernel.Grouped<Group>>,walked,Loop.run(~Traversal.State<&2,&2,Tree.Tree<Tile.Tile8>,Tree.Tree<Tile.Tile8>,Kernel.Grouped<Group>>,~ArrayLaws.model_update_step(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Tile.Tile8,~Kernel.Grouped<Group>,~placed(~Group,~Tile.Tile8,~at),~Updates.kept_sample(~Tile.Tile8),~Kernel.place(~Group)),count,start),
        Updates.run_as_update(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Group>,~placed(~Group,~Tile.Tile8,~at),~Kernel.place(~Group),count,start)) :
      {(Buffer.Content{input,input_length},Buffer.Content{Updates.state_tree(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Group>,_),output_length}) == (Buffer.Content{input,input_length},loop_content(~Group,~Tile.Tile8,~at,~Updates.kept_sample(~Tile.Tile8),context,Buffer.Content{input,input_length},Buffer.Content{output,output_length})) : Contents() & Contents()}
    {==})

# A loop that updates dy in place combines what the model of its group samples.
def back_arrays(+context: Kernel.Grouped<Kernel.Back>,+input: Tree.Tree<Tile.Tile8>,+input_length: Nat,+output: Tree.Tree<Tile.Tile8>,+output_length: Nat) ->
  {arrays_contents(input_length,output_length,Kernel.updated(context,Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,output))) ==
    (Buffer.Content{Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,input)),input_length},loop_content(~Kernel.Back,~Kernel.Sampled,~sample_at(~Kernel.Back,~Kernel.Sampled,~back_of),~Kernel.back_combine,context,Buffer.Content{Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,input)),input_length},Buffer.Content{Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,output)),output_length})) : Contents() & Contents()}:
  +count = U32.to_nat(Kernel.count_of(~Kernel.Back,context))
  +start = {Traversal.State{input,output,0,context} : Traversal.State<&2,&2,Tree.Tree<Tile.Tile8>,Tree.Tree<Tile.Tile8>,Kernel.Grouped<Kernel.Back>>}
  +walked = Loop.run(~Traversal.State<&2,&2,Tree.Tree<Tile.Tile8>,Tree.Tree<Tile.Tile8>,Kernel.Grouped<Kernel.Back>>,~ArrayLaws.model_update_step(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Kernel.Sampled,~Kernel.Grouped<Kernel.Back>,~placed(~Kernel.Back,~Kernel.Sampled,~sample_at(~Kernel.Back,~Kernel.Sampled,~back_of)),~Kernel.back_combine,~Kernel.place(~Kernel.Back)),count,start)
  %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,input)),input,StorageLaws.reflect_packed(Tile.Tile8,input)) :
    {arrays_contents(input_length,output_length,Kernel.updated(context,Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,output))) ==
      (Buffer.Content{_,input_length},loop_content(~Kernel.Back,~Kernel.Sampled,~sample_at(~Kernel.Back,~Kernel.Sampled,~back_of),~Kernel.back_combine,context,Buffer.Content{_,input_length},Buffer.Content{Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,output)),output_length})) : Contents() & Contents()}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,output)),output,StorageLaws.reflect_packed(Tile.Tile8,output)) :
    {arrays_contents(input_length,output_length,Kernel.updated(context,Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,output))) ==
      (Buffer.Content{input,input_length},loop_content(~Kernel.Back,~Kernel.Sampled,~sample_at(~Kernel.Back,~Kernel.Sampled,~back_of),~Kernel.back_combine,context,Buffer.Content{input,input_length},Buffer.Content{_,output_length})) : Contents() & Contents()}
  %Equal.sym(Traversal.State<&1,&1,Array<Tile.Tile8>,Array<Tile.Tile8>,Kernel.Grouped<Kernel.Back>>,Traversal.update(~Array<Tile.Tile8>,~Tile.Tile8,~Kernel.Sampled,~Kernel.Grouped<Kernel.Back>,~Kernel.back_read,~Kernel.back_combine,~Kernel.place(~Kernel.Back),count,ArrayLaws.represent(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Kernel.Back>,~ArrayLaws.pack_tree(~Tile.Tile8),start)),ArrayLaws.represent(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Kernel.Back>,~ArrayLaws.pack_tree(~Tile.Tile8),walked),
      ArrayLaws.update_refines_model(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Kernel.Sampled,~Kernel.Grouped<Kernel.Back>,~ArrayLaws.pack_tree(~Tile.Tile8),~Kernel.back_read,
        ~placed(~Kernel.Back,~Kernel.Sampled,~sample_at(~Kernel.Back,~Kernel.Sampled,~back_of)),~Kernel.back_combine,~Kernel.place(~Kernel.Back),~back_matches,count,start)) :
    {arrays_contents(input_length,output_length,Traversal.finish(~Array<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Kernel.Back>,_)) == (Buffer.Content{input,input_length},loop_content(~Kernel.Back,~Kernel.Sampled,~sample_at(~Kernel.Back,~Kernel.Sampled,~back_of),~Kernel.back_combine,context,Buffer.Content{input,input_length},Buffer.Content{output,output_length})) : Contents() & Contents()}
  Equal.trans(Contents() & Contents(),arrays_contents(input_length,output_length,Traversal.finish(~Array<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Kernel.Back>,ArrayLaws.represent(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Kernel.Back>,~ArrayLaws.pack_tree(~Tile.Tile8),walked))),
    (Buffer.Content{ArrayLaws.model_reader(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Kernel.Back>,walked),input_length},Buffer.Content{Updates.state_tree(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Kernel.Back>,walked),output_length}),
    (Buffer.Content{input,input_length},loop_content(~Kernel.Back,~Kernel.Sampled,~sample_at(~Kernel.Back,~Kernel.Sampled,~back_of),~Kernel.back_combine,context,Buffer.Content{input,input_length},Buffer.Content{output,output_length})),represented_contents(~Kernel.Grouped<Kernel.Back>,input_length,output_length,walked),
    %Equal.sym(Tree.Tree<Tile.Tile8>,ArrayLaws.model_reader(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Kernel.Back>,walked),input,
        Updates.update_reader(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Kernel.Sampled,~Kernel.Grouped<Kernel.Back>,~placed(~Kernel.Back,~Kernel.Sampled,~sample_at(~Kernel.Back,~Kernel.Sampled,~back_of)),~Kernel.back_combine,~Kernel.place(~Kernel.Back),count,0,input,context,output)) :
      {(Buffer.Content{_,input_length},Buffer.Content{Updates.state_tree(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Kernel.Grouped<Kernel.Back>,walked),output_length}) == (Buffer.Content{input,input_length},loop_content(~Kernel.Back,~Kernel.Sampled,~sample_at(~Kernel.Back,~Kernel.Sampled,~back_of),~Kernel.back_combine,context,Buffer.Content{input,input_length},Buffer.Content{output,output_length})) : Contents() & Contents()}
    {==})

# The step of the passes that read one buffer into the other.
def mapped_law(~Group: Data,~read: Array<Tile.Tile8> -> Kernel.Grouped<Group> -> U32 -> Array<Tile.Tile8> & Tile.Tile8,~read_preserves: @-input: Array<Tile.Tile8> -> @context: Kernel.Grouped<Group> -> @index: U32 -> Witness.ArrayWitness<Tile.Tile8,input> -> KernelCertificate.Read(input,read(input,context,index)),~at: Group -> Tree.Tree<Tile.Tile8> -> U32 -> Tile.Tile8,~read_matches: @+reader: Tree.Tree<Tile.Tile8> -> @+context: Kernel.Grouped<Group> -> @+position: U32 -> {read(ArrayLaws.pack_tree(~Tile.Tile8,reader),context,position) == (ArrayLaws.pack_tree(~Tile.Tile8,reader),placed(~Group,~Tile.Tile8,~at,reader,context,position)) : Array<Tile.Tile8> & Tile.Tile8},+context: Kernel.Grouped<Group>,+left: Buffer.Model<Tile.Tile8>,+right: Buffer.Model<Tile.Tile8>) ->
  {pair_contents(BNCells.mapped_buffers(~Group,~read,~read_preserves,context,Buffer.pack(~Tile.Tile8,left),Buffer.pack(~Tile.Tile8,right))) ==
    (Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,left)),loop_content(~Group,~Tile.Tile8,~at,~Updates.kept_sample(~Tile.Tile8),context,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,left)),Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,right)))) : Contents() & Contents()}:
  Buffer.Model{+input,+input_length,input_certificate} = left
  Buffer.Model{+output,+output_length,output_certificate} = right
  buffers_law(input_length,output_length,Kernel.mapped(~Array<Tile.Tile8>,~Group,~read,context,Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,output)),
    KernelCertificate.array_certificates(~Group,~read,~read_preserves,context,Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,output),input_certificate,output_certificate),
    (Buffer.Content{Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,input)),input_length},loop_content(~Group,~Tile.Tile8,~at,~Updates.kept_sample(~Tile.Tile8),context,Buffer.Content{Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,input)),input_length},Buffer.Content{Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,output)),output_length})),
    mapped_arrays(~Group,~read,~at,~read_matches,context,input,input_length,output,output_length))

# The step of the backward pass.
def back_law(+context: Kernel.Grouped<Kernel.Back>,+left: Buffer.Model<Tile.Tile8>,+right: Buffer.Model<Tile.Tile8>) ->
  {pair_contents(BNCells.back_buffers(context,Buffer.pack(~Tile.Tile8,left),Buffer.pack(~Tile.Tile8,right))) ==
    (Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,left)),loop_content(~Kernel.Back,~Kernel.Sampled,~sample_at(~Kernel.Back,~Kernel.Sampled,~back_of),~Kernel.back_combine,context,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,left)),Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,right)))) : Contents() & Contents()}:
  Buffer.Model{+input,+input_length,input_certificate} = left
  Buffer.Model{+output,+output_length,output_certificate} = right
  buffers_law(input_length,output_length,Kernel.updated(context,Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,output)),
    KernelCertificate.updated_certificates(context,Tree.pack(Tile.Tile8,input),Tree.pack(Tile.Tile8,output),input_certificate,output_certificate),
    (Buffer.Content{Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,input)),input_length},loop_content(~Kernel.Back,~Kernel.Sampled,~sample_at(~Kernel.Back,~Kernel.Sampled,~back_of),~Kernel.back_combine,context,Buffer.Content{Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,input)),input_length},Buffer.Content{Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,output)),output_length})),
    back_arrays(context,input,input_length,output,output_length))

# ---- The cells a pass leaves ----

# The loops keep the capacity and the balance of the tree.
def groups_capacity(~Group: Data,~Sample: Data,~at: Group -> Tree.Tree<Tile.Tile8> -> U32 -> Sample,~combine: Tile.Tile8 -> Sample -> Tile.Tile8,groups: List<&2,Group>,+first: U32,+count: U32,+cells: U32,+group: U32,+reader: Tree.Tree<Tile.Tile8>,+tree: Tree.Tree<Tile.Tile8>) ->
  {Tree.capacity(Tile.Tile8,groups_tree(~Group,~Sample,~at,~combine,groups,first,count,cells,group,reader,tree)) == Tree.capacity(Tile.Tile8,tree) : U32}:
  match groups:
    case Nil{}: {==}
    case Con{+parameters,rest}:
      +context = BNCells.grouped(~Group,parameters,first,count,cells,group)
      +looped = group_loop(~Group,~Sample,~at,~combine,context,reader,tree)
      Equal.trans(U32,Tree.capacity(Tile.Tile8,groups_tree(~Group,~Sample,~at,~combine,rest,first,count,cells,(group + 1 : U32),reader,looped)),Tree.capacity(Tile.Tile8,looped),Tree.capacity(Tile.Tile8,tree),
        groups_capacity(~Group,~Sample,~at,~combine,rest,first,count,cells,(group + 1 : U32),reader,looped),
        Updates.update_capacity(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Sample,~Kernel.Grouped<Group>,~placed(~Group,~Sample,~at),~combine,~Kernel.place(~Group),U32.to_nat(Kernel.count_of(~Group,context)),0,reader,context,tree))

def groups_balanced(~Group: Data,~Sample: Data,~at: Group -> Tree.Tree<Tile.Tile8> -> U32 -> Sample,~combine: Tile.Tile8 -> Sample -> Tile.Tile8,groups: List<&2,Group>,+first: U32,+count: U32,+cells: U32,+group: U32,+reader: Tree.Tree<Tile.Tile8>,+tree: Tree.Tree<Tile.Tile8>,+depth: Nat,
  balanced: Shape.Balanced(Tile.Tile8,depth,tree)) -> Shape.Balanced(Tile.Tile8,depth,groups_tree(~Group,~Sample,~at,~combine,groups,first,count,cells,group,reader,tree)):
  match groups:
    case Nil{}: balanced
    case Con{+parameters,rest}:
      +context = BNCells.grouped(~Group,parameters,first,count,cells,group)
      groups_balanced(~Group,~Sample,~at,~combine,rest,first,count,cells,(group + 1 : U32),reader,group_loop(~Group,~Sample,~at,~combine,context,reader,tree),depth,
        Updates.update_balanced(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Sample,~Kernel.Grouped<Group>,~placed(~Group,~Sample,~at),~combine,~Kernel.place(~Group),U32.to_nat(Kernel.count_of(~Group,context)),0,reader,context,tree,depth,balanced))

# The parameters of group before, or fallback past the groups.
def parameter(~Group: Data,groups: List<&2,Group>,before: Nat,+fallback: Group) -> Group:
  match groups:
    case Nil{}: fallback
    case Con{parameters,rest}:
      match before:
        case 0n: parameters
        case 1n+earlier: parameter(~Group,rest,earlier,fallback)

def picked_parameter(~Group: Data,~Sample: Data,~at: Group -> Tree.Tree<Tile.Tile8> -> U32 -> Sample,~combine: Tile.Tile8 -> Sample -> Tile.Tile8,groups: List<&2,Group>,+before: Nat,+reader: Tree.Tree<Tile.Tile8>,+query: U32,+current: Tile.Tile8,+fallback: Group,
  +below: {Nat.is_lt(before,List.length(&2,Group,groups)) == True{} : Bool}) ->
  {picked(~Group,~Tile.Tile8,~combined(~Group,~Sample,~at,~combine,reader,query,current),groups,before,current) == combined(~Group,~Sample,~at,~combine,reader,query,current,parameter(~Group,groups,before,fallback)) : Tile.Tile8}:
  match groups:
    case Nil{}:
      Empty.absurd({picked(~Group,~Tile.Tile8,~combined(~Group,~Sample,~at,~combine,reader,query,current),Nil{},before,current) == combined(~Group,~Sample,~at,~combine,reader,query,current,parameter(~Group,Nil{},before,fallback)) : Tile.Tile8},
        Order.below_zero(before,below))
    case Con{+parameters,rest}:
      match before:
        case 0n: {==}
        case 1n+ +earlier: picked_parameter(~Group,~Sample,~at,~combine,rest,earlier,reader,query,current,fallback,below)

def word_below(+index: Nat,+word: U32,+below: {Nat.is_lt(index,U32.to_nat(word)) == True{} : Bool}) -> {U32.is_lt(U32.from_nat(index),word) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_lt(U32.from_nat(index),word),Nat.is_lt(U32.to_nat(U32.from_nat(index)),U32.to_nat(word)),Comparison.lt(U32.from_nat(index),word)) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,FullRange.roundtrip(index,word,Bounds.strict_le(index,U32.to_nat(word),below))) : {Nat.is_lt(_,U32.to_nat(word)) == True{} : Bool}
  below

def room_positive(+count: U32,+cells: U32,+room: Room<count,cells>) -> {Nat.is_lt(0n,U32.to_nat(cells)) == True{} : Bool}:
  Room{cap,cap_ok,double_room,+positive,count_room} = room
  positive

# What a pass leaves at a cell: the parameters of its group combine the
# current cell with a sample of the cell it reads.
type Pass<-Current: Data,-Reader: Data,-Group: Data> is Data:
  Pass{current: Current,reader: Reader,groups: List<&2,Group>,cells: U32,fallback: Group}

def passed(~Group: Data,~Sample: Data,~sample: Group -> Tile.Tile8 -> Sample,~combine: Tile.Tile8 -> Sample -> Tile.Tile8,~Current: Data,~current_value: Current -> U32 -> Tile.Tile8,~Reader: Data,~reader_value: Reader -> U32 -> Tile.Tile8,context: Pass<Current,Reader,Group>,index: U32) -> Tile.Tile8:
  Pass{current,reader,groups,+cells,+fallback} = context
  +index = index
  combine(current_value(current,index),sample(parameter(~Group,groups,Nat.mod(U32.to_nat(index),U32.to_nat(cells)),fallback),reader_value(reader,index)))

# What a piece of a pass stands on: the room of its groups, the cells of the
# tensor, and the tree of its current cells.
type Piece<-first: U32,-count: U32,-cells: U32,-tree: Tree.Tree<Tile.Tile8>,-length: Nat> is Data:
  Piece{room: Room<count,cells>,limit: U32,inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(limit)) == True{} : Bool},
    depth: Nat,balanced: Shape.Balanced(Tile.Tile8,depth,tree),depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
    tree_room: {Nat.is_le(length,U32.to_nat(Tree.capacity(Tile.Tile8,tree))) == True{} : Bool},length_exact: {length == U32.to_nat(count) : Nat}}

def at_below(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,+first: U32,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,+index: Nat,
  at: Content.At(~Tile.Tile8,~Context,~value,context,first,tree,length,index)) -> {Nat.is_lt(index,length) == True{} : Bool}:
  KernelGemm.Both{below,equal} = at
  below

def at_value(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,+first: U32,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,+index: Nat,
  at: Content.At(~Tile.Tile8,~Context,~value,context,first,tree,length,index)) ->
  {Tree.read(Tile.Tile8,tree,U32.from_nat(index)) == value(context,U32.from_nat(Nat.add(U32.to_nat(first),index))) : Tile.Tile8}:
  KernelGemm.Both{below,equal} = at
  equal

# A cell of a piece ends as the parameters of its group in the tensor make
# it of its current and read values.
def cell_value(~Group: Data,~Sample: Data,~sample: Group -> Tile.Tile8 -> Sample,~combine: Tile.Tile8 -> Sample -> Tile.Tile8,~Current: Data,~current_value: Current -> U32 -> Tile.Tile8,~Reader: Data,~reader_value: Reader -> U32 -> Tile.Tile8,+groups: List<&2,Group>,+first: U32,+count: U32,+cells: U32,+fallback: Group,+current: Current,+reader: Reader,
  +reader_tree: Tree.Tree<Tile.Tile8>,+reader_length: Nat,+current_tree: Tree.Tree<Tile.Tile8>,+current_length: Nat,+piece: Piece<first,count,cells,current_tree,current_length>,
  +listed: {List.length(&2,Group,groups) == U32.to_nat(cells) : Nat},
  +reader_values: Content.Leaf(~Tile.Tile8,~Reader,~reader_value,reader,first,count,reader_tree,reader_length),
  +current_values: Content.Leaf(~Tile.Tile8,~Current,~current_value,current,first,count,current_tree,current_length),
  +index: Nat,+below: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  Content.At(~Tile.Tile8,~Pass<Current,Reader,Group>,~passed(~Group,~Sample,~sample,~combine,~Current,~current_value,~Reader,~reader_value),Pass{current,reader,groups,cells,fallback},first,groups_tree(~Group,~Sample,~sample_at(~Group,~Sample,~sample),~combine,groups,first,count,cells,0,reader_tree,current_tree),current_length,index):
  Piece{+room,+limit,+inside,+depth,+balanced,+depth_ok,+tree_room,+length_exact} = piece
  +capacity = Tree.capacity(Tile.Tile8,current_tree)
  +query = U32.from_nat(index)
  +global = U32.from_nat(Nat.add(U32.to_nat(first),index))
  +current_cell = Tree.read(Tile.Tile8,current_tree,query)
  +reader_cell = Tree.read(Tile.Tile8,reader_tree,query)
  +residue = Nat.mod(U32.to_nat(global),U32.to_nat(cells))
  +parameters = parameter(~Group,groups,residue,fallback)
  +target = combine(current_value(current,global),sample(parameters,reader_value(reader,global)))
  +positive = room_positive(count,cells,room)
  +count_cap = {%length_exact : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} tree_room : {Nat.is_le(U32.to_nat(count),U32.to_nat(capacity)) == True{} : Bool}}
  +index_cap = Bounds.nat_trans(index,U32.to_nat(count),U32.to_nat(capacity),below,count_cap)
  +query_exact = FullRange.roundtrip(index,capacity,Bounds.strict_le(index,U32.to_nat(capacity),index_cap))
  +global_room = Order.transitive(Nat.add(U32.to_nat(first),index),Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(limit),
    Order.offset_le(U32.to_nat(first),index,U32.to_nat(count),Bounds.strict_le(index,U32.to_nat(count),below)),inside)
  +global_exact = FullRange.roundtrip(Nat.add(U32.to_nat(first),index),limit,global_room)
  +residue_exact = Equal.trans(Nat,residue_group(first,cells,index),Nat.mod(Nat.add(U32.to_nat(first),index),U32.to_nat(cells)),residue,
    Equal.sym(Nat,Nat.mod(Nat.add(U32.to_nat(first),index),U32.to_nat(cells)),residue_group(first,cells,index),mod_sum(U32.to_nat(first),index,U32.to_nat(cells),positive)),
    Equal.cong(Nat,Nat,value => Nat.mod(value,U32.to_nat(cells)),Nat.add(U32.to_nat(first),index),U32.to_nat(global),Equal.sym(Nat,U32.to_nat(global),Nat.add(U32.to_nat(first),index),global_exact)))
  +group_below = {%Equal.sym(Nat,List.length(&2,Group,groups),U32.to_nat(cells),listed) : {Nat.is_lt(residue,_) == True{} : Bool}
    DivisionOrder.remainder_small(U32.to_nat(global),U32.to_nat(cells),positive) : {Nat.is_lt(residue,List.length(&2,Group,groups)) == True{} : Bool}}
  +current_at = Content.leaf_cell(~Tile.Tile8,~Current,~current_value,current,first,count,current_tree,current_length,index,current_values,below)
  +reader_at = Content.leaf_cell(~Tile.Tile8,~Reader,~reader_value,reader,first,count,reader_tree,reader_length,index,reader_values,below)
  KernelGemm.Both{at_below(~Current,~current_value,current,first,current_tree,current_length,index,current_at),
    Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,groups_tree(~Group,~Sample,~sample_at(~Group,~Sample,~sample),~combine,groups,first,count,cells,0,reader_tree,current_tree),query),
      picked(~Group,~Tile.Tile8,~combined(~Group,~Sample,~sample_at(~Group,~Sample,~sample),~combine,reader_tree,query,current_cell),groups,residue_group(first,cells,index),current_cell),target,
      piece_read(~Group,~Sample,~sample_at(~Group,~Sample,~sample),~combine,groups,first,count,cells,room,reader_tree,current_tree,depth,query,index,query_exact,below,balanced,Order.successor(depth,30n,depth_ok),
        word_below(index,capacity,index_cap),count_cap,listed),
      %Equal.sym(Nat,residue_group(first,cells,index),residue,residue_exact) : {picked(~Group,~Tile.Tile8,~combined(~Group,~Sample,~sample_at(~Group,~Sample,~sample),~combine,reader_tree,query,current_cell),groups,_,current_cell) == target : Tile.Tile8}
      Equal.trans(Tile.Tile8,picked(~Group,~Tile.Tile8,~combined(~Group,~Sample,~sample_at(~Group,~Sample,~sample),~combine,reader_tree,query,current_cell),groups,residue,current_cell),
        combine(current_cell,sample(parameters,reader_cell)),target,
        picked_parameter(~Group,~Sample,~sample_at(~Group,~Sample,~sample),~combine,groups,residue,reader_tree,query,current_cell,fallback,group_below),
        %Equal.sym(Tile.Tile8,reader_cell,reader_value(reader,global),at_value(~Reader,~reader_value,reader,first,reader_tree,reader_length,index,reader_at)) :
          {combine(current_cell,sample(parameters,_)) == target : Tile.Tile8}
        %Equal.sym(Tile.Tile8,current_cell,current_value(current,global),at_value(~Current,~current_value,current,first,current_tree,current_length,index,current_at)) :
          {combine(_,sample(parameters,reader_value(reader,global))) == target : Tile.Tile8}
        {==}))}

def piece_cells(~Group: Data,~Sample: Data,~sample: Group -> Tile.Tile8 -> Sample,~combine: Tile.Tile8 -> Sample -> Tile.Tile8,~Current: Data,~current_value: Current -> U32 -> Tile.Tile8,~Reader: Data,~reader_value: Reader -> U32 -> Tile.Tile8,+groups: List<&2,Group>,+first: U32,+count: U32,+cells: U32,+fallback: Group,+current: Current,+reader: Reader,
  +reader_tree: Tree.Tree<Tile.Tile8>,+reader_length: Nat,+current_tree: Tree.Tree<Tile.Tile8>,+current_length: Nat,+piece: Piece<first,count,cells,current_tree,current_length>,
  +listed: {List.length(&2,Group,groups) == U32.to_nat(cells) : Nat},
  +reader_values: Content.Leaf(~Tile.Tile8,~Reader,~reader_value,reader,first,count,reader_tree,reader_length),
  +current_values: Content.Leaf(~Tile.Tile8,~Current,~current_value,current,first,count,current_tree,current_length),
  remaining: Nat,+position: Nat,+ahead: {Nat.is_le(Nat.add(position,remaining),U32.to_nat(count)) == True{} : Bool}) ->
  Members.All(Nat,{index => Content.At(~Tile.Tile8,~Pass<Current,Reader,Group>,~passed(~Group,~Sample,~sample,~combine,~Current,~current_value,~Reader,~reader_value),Pass{current,reader,groups,cells,fallback},first,groups_tree(~Group,~Sample,~sample_at(~Group,~Sample,~sample),~combine,groups,first,count,cells,0,reader_tree,current_tree),current_length,index) : Nat -> Data},Ranges.range(remaining,position)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      KernelGemm.Both{cell_value(~Group,~Sample,~sample,~combine,~Current,~current_value,~Reader,~reader_value,groups,first,count,cells,fallback,current,reader,reader_tree,reader_length,current_tree,current_length,piece,listed,reader_values,current_values,position,Order.room_index(position,rest,U32.to_nat(count),ahead)),
        piece_cells(~Group,~Sample,~sample,~combine,~Current,~current_value,~Reader,~reader_value,groups,first,count,cells,fallback,current,reader,reader_tree,reader_length,current_tree,current_length,piece,listed,reader_values,current_values,rest,1n+position,Order.advance_room(position,rest,U32.to_nat(count),ahead))}

# Every cell of a piece ends as the parameters of its group make it.
def piece_values(~Group: Data,~Sample: Data,~sample: Group -> Tile.Tile8 -> Sample,~combine: Tile.Tile8 -> Sample -> Tile.Tile8,~Current: Data,~current_value: Current -> U32 -> Tile.Tile8,~Reader: Data,~reader_value: Reader -> U32 -> Tile.Tile8,+groups: List<&2,Group>,+first: U32,+count: U32,+cells: U32,+fallback: Group,+current: Current,+reader: Reader,
  +reader_tree: Tree.Tree<Tile.Tile8>,+reader_length: Nat,+current_tree: Tree.Tree<Tile.Tile8>,+current_length: Nat,+piece: Piece<first,count,cells,current_tree,current_length>,
  +listed: {List.length(&2,Group,groups) == U32.to_nat(cells) : Nat},
  +reader_values: Content.Leaf(~Tile.Tile8,~Reader,~reader_value,reader,first,count,reader_tree,reader_length),
  +current_values: Content.Leaf(~Tile.Tile8,~Current,~current_value,current,first,count,current_tree,current_length)) ->
  Content.Leaf(~Tile.Tile8,~Pass<Current,Reader,Group>,~passed(~Group,~Sample,~sample,~combine,~Current,~current_value,~Reader,~reader_value),Pass{current,reader,groups,cells,fallback},first,count,groups_tree(~Group,~Sample,~sample_at(~Group,~Sample,~sample),~combine,groups,first,count,cells,0,reader_tree,current_tree),current_length):
  piece_cells(~Group,~Sample,~sample,~combine,~Current,~current_value,~Reader,~reader_value,groups,first,count,cells,fallback,current,reader,reader_tree,reader_length,current_tree,current_length,piece,listed,reader_values,current_values,U32.to_nat(count),0n,Order.reflexive(U32.to_nat(count)))

# The tree a pass leaves keeps the shape of the current one.
def piece_ready(~Group: Data,~Sample: Data,~sample: Group -> Tile.Tile8 -> Sample,~combine: Tile.Tile8 -> Sample -> Tile.Tile8,+groups: List<&2,Group>,+first: U32,+count: U32,+cells: U32,
  +reader_tree: Tree.Tree<Tile.Tile8>,+current_tree: Tree.Tree<Tile.Tile8>,+current_length: Nat,+piece: Piece<first,count,cells,current_tree,current_length>) ->
  Buffer.ContentReady<Tile.Tile8,Buffer.Content{groups_tree(~Group,~Sample,~sample_at(~Group,~Sample,~sample),~combine,groups,first,count,cells,0,reader_tree,current_tree),current_length}>:
  Piece{room,limit,inside,+depth,+balanced,+depth_ok,+tree_room,length_exact} = piece
  Buffer.ContentReady{depth,groups_balanced(~Group,~Sample,~sample_at(~Group,~Sample,~sample),~combine,groups,first,count,cells,0,reader_tree,current_tree,depth,balanced),depth_ok,
    %Equal.sym(U32,Tree.capacity(Tile.Tile8,groups_tree(~Group,~Sample,~sample_at(~Group,~Sample,~sample),~combine,groups,first,count,cells,0,reader_tree,current_tree)),Tree.capacity(Tile.Tile8,current_tree),groups_capacity(~Group,~Sample,~sample_at(~Group,~Sample,~sample),~combine,groups,first,count,cells,0,reader_tree,current_tree)) :
      {Nat.is_le(current_length,U32.to_nat(_)) == True{} : Bool}
    tree_room}

# ---- The pieces of a pass ----

# The contents the groups of a piece leave are the trees of their loops.
def groups_content_exact(~Group: Data,~Sample: Data,~at: Group -> Tree.Tree<Tile.Tile8> -> U32 -> Sample,~combine: Tile.Tile8 -> Sample -> Tile.Tile8,+groups: List<&2,Group>,+first: U32,+count: U32,+cells: U32,+group: U32,+reader: Contents(),+content: Contents()) ->
  {groups_content(~Group,~Sample,~at,~combine,groups,first,count,cells,group,reader,content) ==
    Buffer.Content{groups_tree(~Group,~Sample,~at,~combine,groups,first,count,cells,group,Buffer.content_tree(~Tile.Tile8,reader),Buffer.content_tree(~Tile.Tile8,content)),Buffer.content_length(~Tile.Tile8,content)} : Contents()}:
  +made = groups_content(~Group,~Sample,~at,~combine,groups,first,count,cells,group,reader,content)
  %groups_content_tree(~Group,~Sample,~at,~combine,groups,first,count,cells,group,reader,content) : {made == Buffer.Content{_,Buffer.content_length(~Tile.Tile8,content)} : Contents()}
  %groups_content_length(~Group,~Sample,~at,~combine,groups,first,count,cells,group,reader,content) : {made == Buffer.Content{Buffer.content_tree(~Tile.Tile8,made),_} : Contents()}
  Buffer.content_eta(~Tile.Tile8,made)

def pieces_contents(pair: Pieces.Pieces(~Tile.Tile8) & Pieces.Pieces(~Tile.Tile8)) -> Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>:
  (left,right) = pair
  (Model.contents(~Tile.Tile8,left),Model.contents(~Tile.Tile8,right))

def piece_pair(+first: U32,+count: U32,pair: Contents() & Contents()) -> Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>:
  (left,right) = pair
  (Model.PieceContent{left,first,count},Model.PieceContent{right,first,count})

def joined_pair(left: Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>,right: Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>) -> Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>:
  (left_first,left_second) = left
  (right_first,right_second) = right
  (Model.Joined{left_first,right_first},Model.Joined{left_second,right_second})

def copy_pair_contents(+first: U32,+count: U32,pair: BNCells.Cells() & BNCells.Cells()) ->
  {pieces_contents(Tensors.copy_pair(~Tile.Tile8,first,count,pair)) == piece_pair(first,count,pair_contents(pair)) : Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>}:
  (left,right) = pair
  {==}

def copy_halves_contents(left: Pieces.Pieces(~Tile.Tile8) & Pieces.Pieces(~Tile.Tile8),right: Pieces.Pieces(~Tile.Tile8) & Pieces.Pieces(~Tile.Tile8)) ->
  {pieces_contents(Tensors.copy_halves(~Tile.Tile8,left,right)) == joined_pair(pieces_contents(left),pieces_contents(right)) : Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>}:
  (left_first,left_second) = left
  (right_first,right_second) = right
  {==}

def output_contents(pair: BNCells.Cells() & BNCells.Cells()) -> {Buffer.contents(~Tile.Tile8,BNCells.output_of(pair)) == Pair.snd(Contents(),Contents(),pair_contents(pair)) : Contents()}:
  (left,right) = pair
  {==}

# What the padding of a leaf leaves of the contents of its output.
def padded_output_contents(+padding: TiledMap.Padding,+first: U32,+count: U32,pair: BNCells.Cells() & BNCells.Cells()) ->
  {pair_contents(BNCells.padded_output(padding,first,count,pair)) ==
    (Pair.fst(Contents(),Contents(),pair_contents(pair)),Padding.step(padding,first,count,Pair.snd(Contents(),Contents(),pair_contents(pair)))) : Contents() & Contents()}:
  (input,output) = pair
  %Equal.sym(Contents(),Buffer.contents(~Tile.Tile8,TiledMap.padded(padding,first,count,output)),Padding.step(padding,first,count,Buffer.contents(~Tile.Tile8,output)),
      Padding.rule(padding,first,count,output)) :
    {(Buffer.contents(~Tile.Tile8,input),_) == (Buffer.contents(~Tile.Tile8,input),Padding.step(padding,first,count,Buffer.contents(~Tile.Tile8,output))) : Contents() & Contents()}
  {==}

def padded_both_contents(+padding: TiledMap.Padding,+first: U32,+count: U32,pair: BNCells.Cells() & BNCells.Cells()) ->
  {pair_contents(BNCells.padded_both(padding,first,count,pair)) ==
    (Padding.step(padding,first,count,Pair.fst(Contents(),Contents(),pair_contents(pair))),Padding.step(padding,first,count,Pair.snd(Contents(),Contents(),pair_contents(pair)))) : Contents() & Contents()}:
  (left,right) = pair
  %Equal.sym(Contents(),Buffer.contents(~Tile.Tile8,TiledMap.padded(padding,first,count,left)),Padding.step(padding,first,count,Buffer.contents(~Tile.Tile8,left)),
      Padding.rule(padding,first,count,left)) :
    {(_,Buffer.contents(~Tile.Tile8,TiledMap.padded(padding,first,count,right))) ==
      (Padding.step(padding,first,count,Buffer.contents(~Tile.Tile8,left)),Padding.step(padding,first,count,Buffer.contents(~Tile.Tile8,right))) : Contents() & Contents()}
  %Equal.sym(Contents(),Buffer.contents(~Tile.Tile8,TiledMap.padded(padding,first,count,right)),Padding.step(padding,first,count,Buffer.contents(~Tile.Tile8,right)),
      Padding.rule(padding,first,count,right)) :
    {(Padding.step(padding,first,count,Buffer.contents(~Tile.Tile8,left)),_) ==
      (Padding.step(padding,first,count,Buffer.contents(~Tile.Tile8,left)),Padding.step(padding,first,count,Buffer.contents(~Tile.Tile8,right))) : Contents() & Contents()}
  {==}

# A leaf that hands back two buffers: the contents of both trees of the pass.
def PairRule(~Context: Data,~leaf: Context -> U32 -> U32 -> BNCells.Cells() -> Unit -> Partition.Tree<&1,BNCells.Cells() & BNCells.Cells(),Unit>,
  ~first_model: Context -> U32 -> U32 -> Buffer.Model<Tile.Tile8> -> Model.Content<Tile.Tile8>,~second_model: Context -> U32 -> U32 -> Buffer.Model<Tile.Tile8> -> Model.Content<Tile.Tile8>) -> Type:
  @+context: Context -> @+first: U32 -> @+rows: U32 -> @+input: Buffer.Model<Tile.Tile8> ->
  {pieces_contents(Tensors.unzipped(~Tile.Tile8,leaf(context,first,rows,Buffer.pack(~Tile.Tile8,input),Unit{}))) ==
    (first_model(context,first,rows,input),second_model(context,first,rows,input)) : Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>}

def unzipped_contents(~Context: Data,~leaf: Context -> U32 -> U32 -> BNCells.Cells() -> Unit -> Partition.Tree<&1,BNCells.Cells() & BNCells.Cells(),Unit>,
  ~first_model: Context -> U32 -> U32 -> Buffer.Model<Tile.Tile8> -> Model.Content<Tile.Tile8>,~second_model: Context -> U32 -> U32 -> Buffer.Model<Tile.Tile8> -> Model.Content<Tile.Tile8>,
  ~rule: PairRule(~Context,~leaf,~first_model,~second_model),+context: Context,model: Model.Model<Tile.Tile8>) ->
  {pieces_contents(Tensors.unzipped(~Tile.Tile8,Each.leaf_map(~Tile.Tile8,~(BNCells.Cells() & BNCells.Cells()),~Context,~leaf,context,model))) ==
    (Each.content_map(~Tile.Tile8,~Tile.Tile8,~Context,~first_model,context,model),Each.content_map(~Tile.Tile8,~Tile.Tile8,~Context,~second_model,context,model)) : Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>}:
  match model:
    case Model.Leaf{+input,+first,+rows}: rule(context,first,rows,input)
    case Model.Fork{+left,+right}:
      target = {(Model.Joined{Each.content_map(~Tile.Tile8,~Tile.Tile8,~Context,~first_model,context,left),Each.content_map(~Tile.Tile8,~Tile.Tile8,~Context,~first_model,context,right)},
        Model.Joined{Each.content_map(~Tile.Tile8,~Tile.Tile8,~Context,~second_model,context,left),Each.content_map(~Tile.Tile8,~Tile.Tile8,~Context,~second_model,context,right)}) : Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>}
      Equal.trans(Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>,pieces_contents(Tensors.copy_halves(~Tile.Tile8,Tensors.unzipped(~Tile.Tile8,Each.leaf_map(~Tile.Tile8,~(BNCells.Cells() & BNCells.Cells()),~Context,~leaf,context,left)),Tensors.unzipped(~Tile.Tile8,Each.leaf_map(~Tile.Tile8,~(BNCells.Cells() & BNCells.Cells()),~Context,~leaf,context,right)))),joined_pair(pieces_contents(Tensors.unzipped(~Tile.Tile8,Each.leaf_map(~Tile.Tile8,~(BNCells.Cells() & BNCells.Cells()),~Context,~leaf,context,left))),pieces_contents(Tensors.unzipped(~Tile.Tile8,Each.leaf_map(~Tile.Tile8,~(BNCells.Cells() & BNCells.Cells()),~Context,~leaf,context,right)))),target,
        copy_halves_contents(Tensors.unzipped(~Tile.Tile8,Each.leaf_map(~Tile.Tile8,~(BNCells.Cells() & BNCells.Cells()),~Context,~leaf,context,left)),Tensors.unzipped(~Tile.Tile8,Each.leaf_map(~Tile.Tile8,~(BNCells.Cells() & BNCells.Cells()),~Context,~leaf,context,right))),
        %Equal.sym(Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>,pieces_contents(Tensors.unzipped(~Tile.Tile8,Each.leaf_map(~Tile.Tile8,~(BNCells.Cells() & BNCells.Cells()),~Context,~leaf,context,left))),(Each.content_map(~Tile.Tile8,~Tile.Tile8,~Context,~first_model,context,left),Each.content_map(~Tile.Tile8,~Tile.Tile8,~Context,~second_model,context,left)),
            unzipped_contents(~Context,~leaf,~first_model,~second_model,~rule,context,left)) :
          {joined_pair(_,pieces_contents(Tensors.unzipped(~Tile.Tile8,Each.leaf_map(~Tile.Tile8,~(BNCells.Cells() & BNCells.Cells()),~Context,~leaf,context,right)))) == target : Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>}
        %Equal.sym(Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>,pieces_contents(Tensors.unzipped(~Tile.Tile8,Each.leaf_map(~Tile.Tile8,~(BNCells.Cells() & BNCells.Cells()),~Context,~leaf,context,right))),(Each.content_map(~Tile.Tile8,~Tile.Tile8,~Context,~first_model,context,right),Each.content_map(~Tile.Tile8,~Tile.Tile8,~Context,~second_model,context,right)),
            unzipped_contents(~Context,~leaf,~first_model,~second_model,~rule,context,right)) :
          {joined_pair((Each.content_map(~Tile.Tile8,~Tile.Tile8,~Context,~first_model,context,left),Each.content_map(~Tile.Tile8,~Tile.Tile8,~Context,~second_model,context,left)),_) == target : Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>}
        {==})

# ---- The leaves of the forward passes ----

def zero_tree(+count: U32) -> Tree.Tree<Tile.Tile8>:
  Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0))

# The cells a leaf hands back unchanged.
def kept_model(~Context: Data,context: Context,first: U32,rows: U32,input: Buffer.Model<Tile.Tile8>) -> Model.Content<Tile.Tile8>:
  +first = first
  +rows = rows
  Model.PieceContent{Buffer.content(~Tile.Tile8,input),first,rows}

# The cells the groups of a leaf read into zero cells.
def mapped_model(~Group: Data,~at: Group -> Tree.Tree<Tile.Tile8> -> U32 -> Tile.Tile8,context: BNCells.Mapping<Group>,first: U32,rows: U32,input: Buffer.Model<Tile.Tile8>) -> Model.Content<Tile.Tile8>:
  BNCells.Mapping{groups,+cells,+padding} = context
  +first = first
  +rows = rows
  Model.PieceContent{Padding.step(padding,first,rows,Buffer.Content{groups_tree(~Group,~Tile.Tile8,~at,~Updates.kept_sample(~Tile.Tile8),groups,first,rows,cells,0,Buffer.tree(~Tile.Tile8,input),zero_tree(rows)),U32.to_nat(rows)}),first,rows}

def zero_contents(+count: U32) -> {Buffer.contents(~Tile.Tile8,BNCells.zero_cells(count)) == Buffer.Content{zero_tree(count),U32.to_nat(count)} : Contents()}:
  AllocationLaws.contents(~Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0),U32.to_nat(count))

def mapped_rule(~Group: Data,~read: Array<Tile.Tile8> -> Kernel.Grouped<Group> -> U32 -> Array<Tile.Tile8> & Tile.Tile8,~read_preserves: @-input: Array<Tile.Tile8> -> @context: Kernel.Grouped<Group> -> @index: U32 -> Witness.ArrayWitness<Tile.Tile8,input> -> KernelCertificate.Read(input,read(input,context,index)),~at: Group -> Tree.Tree<Tile.Tile8> -> U32 -> Tile.Tile8,~read_matches: @+reader: Tree.Tree<Tile.Tile8> -> @+context: Kernel.Grouped<Group> -> @+position: U32 -> {read(ArrayLaws.pack_tree(~Tile.Tile8,reader),context,position) == (ArrayLaws.pack_tree(~Tile.Tile8,reader),placed(~Group,~Tile.Tile8,~at,reader,context,position)) : Array<Tile.Tile8> & Tile.Tile8},+context: BNCells.Mapping<Group>,+first: U32,+rows: U32,+input: Buffer.Model<Tile.Tile8>) ->
  {pieces_contents(Tensors.unzipped(~Tile.Tile8,BNCells.map_leaf(~Group,~read,~read_preserves,context,first,rows,Buffer.pack(~Tile.Tile8,input),Unit{}))) ==
    (kept_model(~BNCells.Mapping<Group>,context,first,rows,input),mapped_model(~Group,~at,context,first,rows,input)) : Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>}:
  BNCells.Mapping{+groups,+cells,+padding} = context
  Buffer.Model{+tree,+length,+certificate} = input
  +packed = {Buffer.Model{tree,length,certificate} : Buffer.Model<Tile.Tile8>}
  target = {(Model.PieceContent{Buffer.Content{tree,length},first,rows},
    Model.PieceContent{Padding.step(padding,first,rows,Buffer.Content{groups_tree(~Group,~Tile.Tile8,~at,~Updates.kept_sample(~Tile.Tile8),groups,first,rows,cells,0,tree,zero_tree(rows)),U32.to_nat(rows)}),first,rows}) : Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>}
  Equal.trans(Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>,pieces_contents(Tensors.copy_pair(~Tile.Tile8,first,rows,BNCells.padded_output(padding,first,rows,BNCells.each_group(~Group,~BNCells.mapped_buffers(~Group,~read,~read_preserves),groups,first,rows,cells,0,Buffer.pack(~Tile.Tile8,packed),BNCells.zero_cells(rows))))),piece_pair(first,rows,pair_contents(BNCells.padded_output(padding,first,rows,BNCells.each_group(~Group,~BNCells.mapped_buffers(~Group,~read,~read_preserves),groups,first,rows,cells,0,Buffer.pack(~Tile.Tile8,packed),BNCells.zero_cells(rows))))),target,
    copy_pair_contents(first,rows,BNCells.padded_output(padding,first,rows,BNCells.each_group(~Group,~BNCells.mapped_buffers(~Group,~read,~read_preserves),groups,first,rows,cells,0,Buffer.pack(~Tile.Tile8,packed),BNCells.zero_cells(rows)))),
    %Equal.sym(Contents() & Contents(),pair_contents(BNCells.padded_output(padding,first,rows,BNCells.each_group(~Group,~BNCells.mapped_buffers(~Group,~read,~read_preserves),groups,first,rows,cells,0,Buffer.pack(~Tile.Tile8,packed),BNCells.zero_cells(rows)))),
        (Pair.fst(Contents(),Contents(),pair_contents(BNCells.each_group(~Group,~BNCells.mapped_buffers(~Group,~read,~read_preserves),groups,first,rows,cells,0,Buffer.pack(~Tile.Tile8,packed),BNCells.zero_cells(rows)))),Padding.step(padding,first,rows,Pair.snd(Contents(),Contents(),pair_contents(BNCells.each_group(~Group,~BNCells.mapped_buffers(~Group,~read,~read_preserves),groups,first,rows,cells,0,Buffer.pack(~Tile.Tile8,packed),BNCells.zero_cells(rows)))))),
        padded_output_contents(padding,first,rows,BNCells.each_group(~Group,~BNCells.mapped_buffers(~Group,~read,~read_preserves),groups,first,rows,cells,0,Buffer.pack(~Tile.Tile8,packed),BNCells.zero_cells(rows)))) :
      {piece_pair(first,rows,_) == target : Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>}
    %Equal.sym(Contents() & Contents(),pair_contents(BNCells.each_group(~Group,~BNCells.mapped_buffers(~Group,~read,~read_preserves),groups,first,rows,cells,0,Buffer.pack(~Tile.Tile8,packed),BNCells.zero_cells(rows))),
        (Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,packed)),groups_content(~Group,~Tile.Tile8,~at,~Updates.kept_sample(~Tile.Tile8),groups,first,rows,cells,0,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,packed)),Buffer.contents(~Tile.Tile8,BNCells.zero_cells(rows)))),
        each_contents(~Group,~Tile.Tile8,~at,~Updates.kept_sample(~Tile.Tile8),~BNCells.mapped_buffers(~Group,~read,~read_preserves),~mapped_law(~Group,~read,~read_preserves,~at,~read_matches),groups,first,rows,cells,0,
          Buffer.pack(~Tile.Tile8,packed),BNCells.zero_cells(rows),Buffer.Observation{packed,{==}},Buffer.observe(~Tile.Tile8,BNCells.zero_cells(rows)))) :
      {piece_pair(first,rows,(Pair.fst(Contents(),Contents(),_),Padding.step(padding,first,rows,Pair.snd(Contents(),Contents(),_)))) == target : Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>}
    %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,tree)),tree,StorageLaws.reflect_packed(Tile.Tile8,tree)) :
      {piece_pair(first,rows,(Buffer.Content{_,length},
        Padding.step(padding,first,rows,groups_content(~Group,~Tile.Tile8,~at,~Updates.kept_sample(~Tile.Tile8),groups,first,rows,cells,0,Buffer.Content{_,length},Buffer.contents(~Tile.Tile8,BNCells.zero_cells(rows)))))) == target : Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>}
    %Equal.sym(Contents(),Buffer.contents(~Tile.Tile8,BNCells.zero_cells(rows)),Buffer.Content{zero_tree(rows),U32.to_nat(rows)},zero_contents(rows)) :
      {piece_pair(first,rows,(Buffer.Content{tree,length},Padding.step(padding,first,rows,groups_content(~Group,~Tile.Tile8,~at,~Updates.kept_sample(~Tile.Tile8),groups,first,rows,cells,0,Buffer.Content{tree,length},_)))) == target : Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>}
    %Equal.sym(Contents(),groups_content(~Group,~Tile.Tile8,~at,~Updates.kept_sample(~Tile.Tile8),groups,first,rows,cells,0,Buffer.Content{tree,length},Buffer.Content{zero_tree(rows),U32.to_nat(rows)}),
        Buffer.Content{groups_tree(~Group,~Tile.Tile8,~at,~Updates.kept_sample(~Tile.Tile8),groups,first,rows,cells,0,tree,zero_tree(rows)),U32.to_nat(rows)},
        groups_content_exact(~Group,~Tile.Tile8,~at,~Updates.kept_sample(~Tile.Tile8),groups,first,rows,cells,0,Buffer.Content{tree,length},Buffer.Content{zero_tree(rows),U32.to_nat(rows)})) :
      {piece_pair(first,rows,(Buffer.Content{tree,length},Padding.step(padding,first,rows,_))) == target : Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>}
    {==})

# The output of a piece is read from the cells the normalizing loops hand
# back.
def forward_seen(+groups: List<&2,Kernel.Normal>,+first: U32,+count: U32,+cells: U32,-input: BNCells.Cells(),-xhat: BNCells.Cells(),+seen: Buffer.Observation<Tile.Tile8,input>) ->
  {pair_contents(BNCells.forward_output(groups,first,count,cells,(input,xhat))) == (Buffer.contents(~Tile.Tile8,xhat),groups_content(~Kernel.Normal,~Tile.Tile8,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of),~Updates.kept_sample(~Tile.Tile8),groups,first,count,cells,0,Buffer.contents(~Tile.Tile8,input),Buffer.contents(~Tile.Tile8,BNCells.zero_cells(count)))) : Contents() & Contents()}:
  Buffer.Observation{+model,+equation} = seen
  %equation : {pair_contents(BNCells.forward_output(groups,first,count,cells,(_,xhat))) ==
    (Buffer.contents(~Tile.Tile8,xhat),groups_content(~Kernel.Normal,~Tile.Tile8,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of),~Updates.kept_sample(~Tile.Tile8),groups,first,count,cells,0,Buffer.contents(~Tile.Tile8,_),Buffer.contents(~Tile.Tile8,BNCells.zero_cells(count)))) : Contents() & Contents()}
  %Equal.sym(Contents(),Buffer.contents(~Tile.Tile8,BNCells.output_of(BNCells.each_group(~Kernel.Normal,~BNCells.mapped_buffers(~Kernel.Normal,~Kernel.read_output,~KernelCertificate.read_output),groups,first,count,cells,0,Buffer.pack(~Tile.Tile8,model),BNCells.zero_cells(count)))),Pair.snd(Contents(),Contents(),pair_contents(BNCells.each_group(~Kernel.Normal,~BNCells.mapped_buffers(~Kernel.Normal,~Kernel.read_output,~KernelCertificate.read_output),groups,first,count,cells,0,Buffer.pack(~Tile.Tile8,model),BNCells.zero_cells(count)))),output_contents(BNCells.each_group(~Kernel.Normal,~BNCells.mapped_buffers(~Kernel.Normal,~Kernel.read_output,~KernelCertificate.read_output),groups,first,count,cells,0,Buffer.pack(~Tile.Tile8,model),BNCells.zero_cells(count)))) :
    {(Buffer.contents(~Tile.Tile8,xhat),_) == (Buffer.contents(~Tile.Tile8,xhat),groups_content(~Kernel.Normal,~Tile.Tile8,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of),~Updates.kept_sample(~Tile.Tile8),groups,first,count,cells,0,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,model)),Buffer.contents(~Tile.Tile8,BNCells.zero_cells(count)))) : Contents() & Contents()}
  %Equal.sym(Contents() & Contents(),pair_contents(BNCells.each_group(~Kernel.Normal,~BNCells.mapped_buffers(~Kernel.Normal,~Kernel.read_output,~KernelCertificate.read_output),groups,first,count,cells,0,Buffer.pack(~Tile.Tile8,model),BNCells.zero_cells(count))),
      (Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,model)),groups_content(~Kernel.Normal,~Tile.Tile8,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of),~Updates.kept_sample(~Tile.Tile8),groups,first,count,cells,0,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,model)),Buffer.contents(~Tile.Tile8,BNCells.zero_cells(count)))),
      each_contents(~Kernel.Normal,~Tile.Tile8,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of),~Updates.kept_sample(~Tile.Tile8),~BNCells.mapped_buffers(~Kernel.Normal,~Kernel.read_output,~KernelCertificate.read_output),
        ~mapped_law(~Kernel.Normal,~Kernel.read_output,~KernelCertificate.read_output,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of),~output_matches),groups,first,count,cells,0,
        Buffer.pack(~Tile.Tile8,model),BNCells.zero_cells(count),Buffer.Observation{model,{==}},Buffer.observe(~Tile.Tile8,BNCells.zero_cells(count)))) :
    {(Buffer.contents(~Tile.Tile8,xhat),Pair.snd(Contents(),Contents(),_)) == (Buffer.contents(~Tile.Tile8,xhat),groups_content(~Kernel.Normal,~Tile.Tile8,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of),~Updates.kept_sample(~Tile.Tile8),groups,first,count,cells,0,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,model)),Buffer.contents(~Tile.Tile8,BNCells.zero_cells(count)))) : Contents() & Contents()}
  {==}

def forward_contents(+groups: List<&2,Kernel.Normal>,+first: U32,+count: U32,+cells: U32,pair: BNCells.Cells() & BNCells.Cells()) ->
  {pair_contents(BNCells.forward_output(groups,first,count,cells,pair)) ==
    (Pair.snd(Contents(),Contents(),pair_contents(pair)),
      groups_content(~Kernel.Normal,~Tile.Tile8,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of),~Updates.kept_sample(~Tile.Tile8),groups,first,count,cells,0,Pair.fst(Contents(),Contents(),pair_contents(pair)),Buffer.contents(~Tile.Tile8,BNCells.zero_cells(count)))) : Contents() & Contents()}:
  (input,xhat) = pair
  forward_seen(groups,first,count,cells,input,xhat,Buffer.observe(~Tile.Tile8,input))

def forward_rule(+context: BNCells.Mapping<Kernel.Normal>,+first: U32,+rows: U32,+input: Buffer.Model<Tile.Tile8>) ->
  {pieces_contents(Tensors.unzipped(~Tile.Tile8,BNCells.forward_leaf(context,first,rows,Buffer.pack(~Tile.Tile8,input),Unit{}))) ==
    (mapped_model(~Kernel.Normal,~sample_at(~Kernel.Normal,~Tile.Tile8,~Kernel.normalized),context,first,rows,input),
      mapped_model(~Kernel.Normal,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of),context,first,rows,input)) : Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>}:
  BNCells.Mapping{+groups,+cells,+padding} = context
  Buffer.Model{+tree,+length,+certificate} = input
  +packed = {Buffer.Model{tree,length,certificate} : Buffer.Model<Tile.Tile8>}
  +zero = Buffer.contents(~Tile.Tile8,BNCells.zero_cells(rows))
  target = {(Model.PieceContent{Padding.step(padding,first,rows,Buffer.Content{groups_tree(~Kernel.Normal,~Tile.Tile8,~sample_at(~Kernel.Normal,~Tile.Tile8,~Kernel.normalized),~Updates.kept_sample(~Tile.Tile8),groups,first,rows,cells,0,tree,zero_tree(rows)),U32.to_nat(rows)}),first,rows},
    Model.PieceContent{Padding.step(padding,first,rows,Buffer.Content{groups_tree(~Kernel.Normal,~Tile.Tile8,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of),~Updates.kept_sample(~Tile.Tile8),groups,first,rows,cells,0,tree,zero_tree(rows)),U32.to_nat(rows)}),first,rows}) : Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>}
  Equal.trans(Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>,pieces_contents(Tensors.copy_pair(~Tile.Tile8,first,rows,BNCells.padded_both(padding,first,rows,BNCells.forward_output(groups,first,rows,cells,BNCells.each_group(~Kernel.Normal,~BNCells.mapped_buffers(~Kernel.Normal,~Kernel.read_normalized,~KernelCertificate.read_normalized),groups,first,rows,cells,0,Buffer.pack(~Tile.Tile8,packed),BNCells.zero_cells(rows)))))),piece_pair(first,rows,pair_contents(BNCells.padded_both(padding,first,rows,BNCells.forward_output(groups,first,rows,cells,BNCells.each_group(~Kernel.Normal,~BNCells.mapped_buffers(~Kernel.Normal,~Kernel.read_normalized,~KernelCertificate.read_normalized),groups,first,rows,cells,0,Buffer.pack(~Tile.Tile8,packed),BNCells.zero_cells(rows)))))),target,
    copy_pair_contents(first,rows,BNCells.padded_both(padding,first,rows,BNCells.forward_output(groups,first,rows,cells,BNCells.each_group(~Kernel.Normal,~BNCells.mapped_buffers(~Kernel.Normal,~Kernel.read_normalized,~KernelCertificate.read_normalized),groups,first,rows,cells,0,Buffer.pack(~Tile.Tile8,packed),BNCells.zero_cells(rows))))),
    %Equal.sym(Contents() & Contents(),pair_contents(BNCells.padded_both(padding,first,rows,BNCells.forward_output(groups,first,rows,cells,BNCells.each_group(~Kernel.Normal,~BNCells.mapped_buffers(~Kernel.Normal,~Kernel.read_normalized,~KernelCertificate.read_normalized),groups,first,rows,cells,0,Buffer.pack(~Tile.Tile8,packed),BNCells.zero_cells(rows))))),
        (Padding.step(padding,first,rows,Pair.fst(Contents(),Contents(),pair_contents(BNCells.forward_output(groups,first,rows,cells,BNCells.each_group(~Kernel.Normal,~BNCells.mapped_buffers(~Kernel.Normal,~Kernel.read_normalized,~KernelCertificate.read_normalized),groups,first,rows,cells,0,Buffer.pack(~Tile.Tile8,packed),BNCells.zero_cells(rows)))))),Padding.step(padding,first,rows,Pair.snd(Contents(),Contents(),pair_contents(BNCells.forward_output(groups,first,rows,cells,BNCells.each_group(~Kernel.Normal,~BNCells.mapped_buffers(~Kernel.Normal,~Kernel.read_normalized,~KernelCertificate.read_normalized),groups,first,rows,cells,0,Buffer.pack(~Tile.Tile8,packed),BNCells.zero_cells(rows))))))),
        padded_both_contents(padding,first,rows,BNCells.forward_output(groups,first,rows,cells,BNCells.each_group(~Kernel.Normal,~BNCells.mapped_buffers(~Kernel.Normal,~Kernel.read_normalized,~KernelCertificate.read_normalized),groups,first,rows,cells,0,Buffer.pack(~Tile.Tile8,packed),BNCells.zero_cells(rows))))) :
      {piece_pair(first,rows,_) == target : Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>}
    %Equal.sym(Contents() & Contents(),pair_contents(BNCells.forward_output(groups,first,rows,cells,BNCells.each_group(~Kernel.Normal,~BNCells.mapped_buffers(~Kernel.Normal,~Kernel.read_normalized,~KernelCertificate.read_normalized),groups,first,rows,cells,0,Buffer.pack(~Tile.Tile8,packed),BNCells.zero_cells(rows)))),
        (Pair.snd(Contents(),Contents(),pair_contents(BNCells.each_group(~Kernel.Normal,~BNCells.mapped_buffers(~Kernel.Normal,~Kernel.read_normalized,~KernelCertificate.read_normalized),groups,first,rows,cells,0,Buffer.pack(~Tile.Tile8,packed),BNCells.zero_cells(rows)))),groups_content(~Kernel.Normal,~Tile.Tile8,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of),~Updates.kept_sample(~Tile.Tile8),groups,first,rows,cells,0,Pair.fst(Contents(),Contents(),pair_contents(BNCells.each_group(~Kernel.Normal,~BNCells.mapped_buffers(~Kernel.Normal,~Kernel.read_normalized,~KernelCertificate.read_normalized),groups,first,rows,cells,0,Buffer.pack(~Tile.Tile8,packed),BNCells.zero_cells(rows)))),zero)),
        forward_contents(groups,first,rows,cells,BNCells.each_group(~Kernel.Normal,~BNCells.mapped_buffers(~Kernel.Normal,~Kernel.read_normalized,~KernelCertificate.read_normalized),groups,first,rows,cells,0,Buffer.pack(~Tile.Tile8,packed),BNCells.zero_cells(rows)))) :
      {piece_pair(first,rows,(Padding.step(padding,first,rows,Pair.fst(Contents(),Contents(),_)),Padding.step(padding,first,rows,Pair.snd(Contents(),Contents(),_)))) == target : Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>}
    %Equal.sym(Contents() & Contents(),pair_contents(BNCells.each_group(~Kernel.Normal,~BNCells.mapped_buffers(~Kernel.Normal,~Kernel.read_normalized,~KernelCertificate.read_normalized),groups,first,rows,cells,0,Buffer.pack(~Tile.Tile8,packed),BNCells.zero_cells(rows))),
        (Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,packed)),groups_content(~Kernel.Normal,~Tile.Tile8,~sample_at(~Kernel.Normal,~Tile.Tile8,~Kernel.normalized),~Updates.kept_sample(~Tile.Tile8),groups,first,rows,cells,0,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,packed)),zero)),
        each_contents(~Kernel.Normal,~Tile.Tile8,~sample_at(~Kernel.Normal,~Tile.Tile8,~Kernel.normalized),~Updates.kept_sample(~Tile.Tile8),~BNCells.mapped_buffers(~Kernel.Normal,~Kernel.read_normalized,~KernelCertificate.read_normalized),
          ~mapped_law(~Kernel.Normal,~Kernel.read_normalized,~KernelCertificate.read_normalized,~sample_at(~Kernel.Normal,~Tile.Tile8,~Kernel.normalized),~normal_matches),groups,first,rows,cells,0,
          Buffer.pack(~Tile.Tile8,packed),BNCells.zero_cells(rows),Buffer.Observation{packed,{==}},Buffer.observe(~Tile.Tile8,BNCells.zero_cells(rows)))) :
      {piece_pair(first,rows,(Padding.step(padding,first,rows,Pair.snd(Contents(),Contents(),_)),
        Padding.step(padding,first,rows,groups_content(~Kernel.Normal,~Tile.Tile8,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of),~Updates.kept_sample(~Tile.Tile8),groups,first,rows,cells,0,Pair.fst(Contents(),Contents(),_),zero)))) == target : Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>}
    %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,tree)),tree,StorageLaws.reflect_packed(Tile.Tile8,tree)) :
      {piece_pair(first,rows,(Padding.step(padding,first,rows,groups_content(~Kernel.Normal,~Tile.Tile8,~sample_at(~Kernel.Normal,~Tile.Tile8,~Kernel.normalized),~Updates.kept_sample(~Tile.Tile8),groups,first,rows,cells,0,Buffer.Content{_,length},zero)),
        Padding.step(padding,first,rows,groups_content(~Kernel.Normal,~Tile.Tile8,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of),~Updates.kept_sample(~Tile.Tile8),groups,first,rows,cells,0,Buffer.Content{_,length},zero)))) == target : Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>}
    %Equal.sym(Contents(),zero,Buffer.Content{zero_tree(rows),U32.to_nat(rows)},zero_contents(rows)) :
      {piece_pair(first,rows,(Padding.step(padding,first,rows,groups_content(~Kernel.Normal,~Tile.Tile8,~sample_at(~Kernel.Normal,~Tile.Tile8,~Kernel.normalized),~Updates.kept_sample(~Tile.Tile8),groups,first,rows,cells,0,Buffer.Content{tree,length},_)),
        Padding.step(padding,first,rows,groups_content(~Kernel.Normal,~Tile.Tile8,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of),~Updates.kept_sample(~Tile.Tile8),groups,first,rows,cells,0,Buffer.Content{tree,length},_)))) == target : Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>}
    %Equal.sym(Contents(),groups_content(~Kernel.Normal,~Tile.Tile8,~sample_at(~Kernel.Normal,~Tile.Tile8,~Kernel.normalized),~Updates.kept_sample(~Tile.Tile8),groups,first,rows,cells,0,Buffer.Content{tree,length},Buffer.Content{zero_tree(rows),U32.to_nat(rows)}),
        Buffer.Content{groups_tree(~Kernel.Normal,~Tile.Tile8,~sample_at(~Kernel.Normal,~Tile.Tile8,~Kernel.normalized),~Updates.kept_sample(~Tile.Tile8),groups,first,rows,cells,0,tree,zero_tree(rows)),U32.to_nat(rows)},
        groups_content_exact(~Kernel.Normal,~Tile.Tile8,~sample_at(~Kernel.Normal,~Tile.Tile8,~Kernel.normalized),~Updates.kept_sample(~Tile.Tile8),groups,first,rows,cells,0,Buffer.Content{tree,length},Buffer.Content{zero_tree(rows),U32.to_nat(rows)})) :
      {piece_pair(first,rows,(Padding.step(padding,first,rows,_),
        Padding.step(padding,first,rows,groups_content(~Kernel.Normal,~Tile.Tile8,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of),~Updates.kept_sample(~Tile.Tile8),groups,first,rows,cells,0,Buffer.Content{tree,length},Buffer.Content{zero_tree(rows),U32.to_nat(rows)})))) == target : Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>}
    %Equal.sym(Contents(),groups_content(~Kernel.Normal,~Tile.Tile8,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of),~Updates.kept_sample(~Tile.Tile8),groups,first,rows,cells,0,Buffer.Content{tree,length},Buffer.Content{zero_tree(rows),U32.to_nat(rows)}),
        Buffer.Content{groups_tree(~Kernel.Normal,~Tile.Tile8,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of),~Updates.kept_sample(~Tile.Tile8),groups,first,rows,cells,0,tree,zero_tree(rows)),U32.to_nat(rows)},
        groups_content_exact(~Kernel.Normal,~Tile.Tile8,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of),~Updates.kept_sample(~Tile.Tile8),groups,first,rows,cells,0,Buffer.Content{tree,length},Buffer.Content{zero_tree(rows),U32.to_nat(rows)})) :
      {piece_pair(first,rows,(Padding.step(padding,first,rows,Buffer.Content{groups_tree(~Kernel.Normal,~Tile.Tile8,~sample_at(~Kernel.Normal,~Tile.Tile8,~Kernel.normalized),~Updates.kept_sample(~Tile.Tile8),groups,first,rows,cells,0,tree,zero_tree(rows)),U32.to_nat(rows)}),
        Padding.step(padding,first,rows,_))) == target : Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>}
    {==})

def mapped_layout_rule(~Group: Data,~at: Group -> Tree.Tree<Tile.Tile8> -> U32 -> Tile.Tile8,+context: BNCells.Mapping<Group>,+first: U32,+rows: U32,+input: Buffer.Model<Tile.Tile8>) ->
  {Model.content_layout(~Tile.Tile8,mapped_model(~Group,~at,context,first,rows,input)) == Pieces.Range{first,rows} : Pieces.Ranges}:
  BNCells.Mapping{+groups,+cells,+padding} = context
  {==}

# ---- The facts of the pieces a forward pass leaves ----

def zero_value(context: Unit,index: U32) -> Tile.Tile8:
  Tile.filled(0.0)

def zero_values(+first: U32,+count: U32,remaining: Nat,+position: Nat,+ahead: {Nat.is_le(Nat.add(position,remaining),U32.to_nat(count)) == True{} : Bool}) ->
  Members.All(Nat,{index => Content.At(~Tile.Tile8,~Unit,~zero_value,Unit{},first,zero_tree(count),U32.to_nat(count),index) : Nat -> Data},Ranges.range(remaining,position)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      KernelGemm.Both{KernelGemm.Both{Order.room_index(position,rest,U32.to_nat(count),ahead),
          StorageReadback.read_fresh(~Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0),U32.from_nat(position))},
        zero_values(first,count,rest,1n+position,Order.advance_room(position,rest,U32.to_nat(count),ahead))}

# What the pieces of a pass fit in: the cells of the tensor, its allocation
# and two strides of the groups.
type Rooms<-limit: U32,-cells: U32,-padding: TiledMap.Padding> is Data:
  Rooms{cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},double_room: {Nat.is_le(Nat.double(U32.to_nat(cells)),U32.to_nat(cap)) == True{} : Bool},
    positive: {Nat.is_lt(0n,U32.to_nat(cells)) == True{} : Bool},total_room: {Nat.is_le(Nat.add(U32.to_nat(limit),U32.to_nat(cells)),U32.to_nat(cap)) == True{} : Bool},
    allocatable: {U32.is_le(limit,1073741824) == True{} : Bool},fit: {PaddedPieces.groups_fit(cap,padding) == True{} : Bool},
    whole: {Nat.is_le(U32.to_nat(limit),PaddedPieces.padded_cells(padding)) == True{} : Bool}}

# The facts of an input piece and of a piece a pass leaves.
def piece_in(~Logical: Data,~original: Logical -> U32 -> Tile.Tile8,+logical: Logical,+total: Nat,+first: U32,+rows: U32,+buffer: Contents()) -> Data:
  KernelGemm.Both<Content.piece_fields(~Tile.Tile8,total,first,rows,buffer),Content.piece_values(~Tile.Tile8,~Logical,~original,logical,first,rows,buffer)>

def piece_full(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,+total: Nat,+first: U32,+rows: U32,+buffer: Contents()) -> Data:
  KernelGemm.Both<Content.piece_fields(~Tile.Tile8,total,first,rows,buffer),KernelGemm.Both<Model.ready_piece(~Tile.Tile8,first,rows,buffer),
    Content.piece_values(~Tile.Tile8,~Context,~value,context,first,rows,buffer)>>

# A piece keeps its fields and readiness through the padding of its output,
# and holds what the padding leaves of its values.
def padded_of(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,+total: Nat,+padding: TiledMap.Padding,+first: U32,+rows: U32,+content: Contents(),
  +inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),total) == True{} : Bool},
  after: Placed.PieceHolds(~Tile.Tile8,~PaddedPieces.Padded<Context>,~PaddedPieces.padded_value(~Context,~value),PaddedPieces.Padded{padding,context},first,rows,
    Padding.step(padding,first,rows,content))) ->
  piece_full(~PaddedPieces.Padded<Context>,~PaddedPieces.padded_value(~Context,~value),PaddedPieces.Padded{padding,context},total,first,rows,Padding.step(padding,first,rows,content)):
  KernelGemm.Both{values,sized_ready} = after
  KernelGemm.Both{sized,ready} = sized_ready
  KernelGemm.Both{Content.Fields{sized,inside},KernelGemm.Both{ready,values}}

def padded_full(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,+cells: U32,+padding: TiledMap.Padding,+limit: U32,+rooms: Rooms<limit,cells,padding>,
  +first: U32,+rows: U32,+content: Contents(),+full: piece_full(~Context,~value,context,U32.to_nat(limit),first,rows,content)) ->
  piece_full(~PaddedPieces.Padded<Context>,~PaddedPieces.padded_value(~Context,~value),PaddedPieces.Padded{padding,context},U32.to_nat(limit),first,rows,
    Padding.step(padding,first,rows,content)):
  Rooms{+cap,+cap_ok,double_room,positive,total_room,allocatable,+fit,+whole} = rooms
  Buffer.Content{+tree,+length} = content
  KernelGemm.Both{fields,rest} = full
  Content.Fields{+length_exact,+inside} = fields
  KernelGemm.Both{+ready,+values} = rest
  +inside_cells = Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(limit),PaddedPieces.padded_cells(padding),inside,whole)
  +classified = {%Equal.sym(Bool,PaddedPieces.groups_fit(cap,padding),True{},fit) :
      {_ && PaddedPieces.piece_inside(PaddedPieces.padded_cells(padding),first,rows) == True{} : Bool}
    inside_cells : {PaddedPieces.padding_guard(cap,padding,first,rows) == True{} : Bool}}
  padded_of(~Context,~value,context,U32.to_nat(limit),padding,first,rows,Buffer.Content{tree,length},inside,
    PaddedPieces.pad_lift(~Context,~value,True{},cap,cap_ok,padding,context,first,rows,Buffer.Content{tree,length},classified,
      KernelGemm.Both{values,KernelGemm.Both{length_exact,ready}}))

# A piece the groups read into zero cells has the fields of its input, is
# ready, and holds what the parameters of every group make of its cells.
def mapped_piece(~Group: Data,~sample: Group -> Tile.Tile8 -> Tile.Tile8,~Logical: Data,~original: Logical -> U32 -> Tile.Tile8,+groups: List<&2,Group>,+cells: U32,+fallback: Group,
  +logical: Logical,+padding: TiledMap.Padding,+limit: U32,+rooms: Rooms<limit,cells,padding>,+listed: {List.length(&2,Group,groups) == U32.to_nat(cells) : Nat},+first: U32,+rows: U32,+tree: Tree.Tree<Tile.Tile8>,+length: Nat,
  +fields: Content.Fields<first,rows,length,U32.to_nat(limit)>,+values: Content.Leaf(~Tile.Tile8,~Logical,~original,logical,first,rows,tree,length)) ->
  piece_full(~Pass<Unit,Logical,Group>,~passed(~Group,~Tile.Tile8,~sample,~Updates.kept_sample(~Tile.Tile8),~Unit,~zero_value,~Logical,~original),Pass{Unit{},logical,groups,cells,fallback},U32.to_nat(limit),first,rows,Buffer.Content{groups_tree(~Group,~Tile.Tile8,~sample_at(~Group,~Tile.Tile8,~sample),~Updates.kept_sample(~Tile.Tile8),groups,first,rows,cells,0,tree,zero_tree(rows)),U32.to_nat(rows)}):
  Rooms{+cap,+cap_ok,+double_room,+positive,+total_room,+allocatable,fit,whole} = rooms
  Content.Fields{+length_exact,+inside} = fields
  +rows_limit = Order.transitive(U32.to_nat(rows),Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(limit),Order.right_le(U32.to_nat(first),U32.to_nat(rows)),inside)
  +room = {Room{cap,cap_ok,double_room,positive,
    Order.transitive(Nat.add(U32.to_nat(rows),U32.to_nat(cells)),Nat.add(U32.to_nat(limit),U32.to_nat(cells)),U32.to_nat(cap),
      Order.add_le(U32.to_nat(rows),U32.to_nat(limit),U32.to_nat(cells),rows_limit),total_room)} : Room<rows,cells>}
  +rows_word = {%Equal.sym(Bool,U32.is_le(rows,limit),Nat.is_le(U32.to_nat(rows),U32.to_nat(limit)),Comparison.le(rows,limit)) : {_ == True{} : Bool} rows_limit :
    {U32.is_le(rows,limit) == True{} : Bool}}
  +sized = Depth.bounded(rows,Order.word_le(rows,limit,1073741824,rows_word,allocatable))
  +depth = Tensors.depth_for(rows)
  +tree_room = {%Equal.sym(U32,Tree.capacity(Tile.Tile8,zero_tree(rows)),U32.shln(1,depth),Shape.capacity(Tile.Tile8,depth,zero_tree(rows),Shape.fresh(Tile.Tile8,depth,Tile.filled(0.0)))) :
      {Nat.is_le(U32.to_nat(rows),U32.to_nat(_)) == True{} : Bool}
    Depth.allocation_room(rows,sized) : {Nat.is_le(U32.to_nat(rows),U32.to_nat(Tree.capacity(Tile.Tile8,zero_tree(rows)))) == True{} : Bool}}
  +piece = {Piece{room,limit,inside,depth,Shape.fresh(Tile.Tile8,depth,Tile.filled(0.0)),Depth.allocation_depth(rows,sized),tree_room,{==}} :
    Piece<first,rows,cells,zero_tree(rows),U32.to_nat(rows)>}
  KernelGemm.Both{Content.Fields{{==},inside},
    KernelGemm.Both{piece_ready(~Group,~Tile.Tile8,~sample,~Updates.kept_sample(~Tile.Tile8),groups,first,rows,cells,tree,zero_tree(rows),U32.to_nat(rows),piece),
      piece_values(~Group,~Tile.Tile8,~sample,~Updates.kept_sample(~Tile.Tile8),~Unit,~zero_value,~Logical,~original,groups,first,rows,cells,fallback,Unit{},logical,tree,length,zero_tree(rows),U32.to_nat(rows),piece,listed,values,
        zero_values(first,rows,U32.to_nat(rows),0n,Order.reflexive(U32.to_nat(rows))))}}

# The fields, readiness and values of every piece, three properties of a
# tree of contents.
def Facts(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,+total: Nat,content: Model.Content<Tile.Tile8>) -> Data:
  KernelGemm.Both<Model.ContentAll(~Tile.Tile8,~Content.piece_fields(~Tile.Tile8,total),content),KernelGemm.Both<Model.ContentReady(~Tile.Tile8,content),
    Content.Values(~Tile.Tile8,~Context,~value,context,content)>>

def facts_joined(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,+total: Nat,-left: Model.Content<Tile.Tile8>,-right: Model.Content<Tile.Tile8>,
  left_facts: Facts(~Context,~value,context,total,left),right_facts: Facts(~Context,~value,context,total,right)) -> Facts(~Context,~value,context,total,Model.Joined{left,right}):
  KernelGemm.Both{left_fields,left_rest} = left_facts
  KernelGemm.Both{left_ready,left_values} = left_rest
  KernelGemm.Both{right_fields,right_rest} = right_facts
  KernelGemm.Both{right_ready,right_values} = right_rest
  KernelGemm.Both{KernelGemm.Both{left_fields,right_fields},KernelGemm.Both{KernelGemm.Both{left_ready,right_ready},KernelGemm.Both{left_values,right_values}}}

# The pieces a pass reads into zero cells.
def mapped_facts(~Group: Data,~sample: Group -> Tile.Tile8 -> Tile.Tile8,~Logical: Data,~original: Logical -> U32 -> Tile.Tile8,+groups: List<&2,Group>,+cells: U32,+fallback: Group,
  +logical: Logical,+padding: TiledMap.Padding,+limit: U32,+rooms: Rooms<limit,cells,padding>,+listed: {List.length(&2,Group,groups) == U32.to_nat(cells) : Nat},model: Model.Model<Tile.Tile8>,
  facts: Model.ContentAll(~Tile.Tile8,~piece_in(~Logical,~original,logical,U32.to_nat(limit)),Model.content(~Tile.Tile8,model))) ->
  Facts(~PaddedPieces.Padded<Pass<Unit,Logical,Group>>,~PaddedPieces.padded_value(~Pass<Unit,Logical,Group>,~passed(~Group,~Tile.Tile8,~sample,~Updates.kept_sample(~Tile.Tile8),~Unit,~zero_value,~Logical,~original)),PaddedPieces.Padded{padding,Pass{Unit{},logical,groups,cells,fallback}},U32.to_nat(limit),
    Each.content_map(~Tile.Tile8,~Tile.Tile8,~BNCells.Mapping<Group>,~mapped_model(~Group,~sample_at(~Group,~Tile.Tile8,~sample)),BNCells.Mapping{groups,cells,padding},model)):
  match model:
    case Model.Leaf{+input,+first,+rows}:
      Buffer.Model{+tree,+length,certificate} = input
      KernelGemm.Both{+fields,+values} = facts
      padded_full(~Pass<Unit,Logical,Group>,~passed(~Group,~Tile.Tile8,~sample,~Updates.kept_sample(~Tile.Tile8),~Unit,~zero_value,~Logical,~original),Pass{Unit{},logical,groups,cells,fallback},cells,padding,limit,rooms,first,rows,Buffer.Content{groups_tree(~Group,~Tile.Tile8,~sample_at(~Group,~Tile.Tile8,~sample),~Updates.kept_sample(~Tile.Tile8),groups,first,rows,cells,0,tree,zero_tree(rows)),U32.to_nat(rows)},
        mapped_piece(~Group,~sample,~Logical,~original,groups,cells,fallback,logical,padding,limit,rooms,listed,first,rows,tree,length,fields,values))
    case Model.Fork{+left,+right}:
      KernelGemm.Both{left_facts,right_facts} = facts
      facts_joined(~PaddedPieces.Padded<Pass<Unit,Logical,Group>>,~PaddedPieces.padded_value(~Pass<Unit,Logical,Group>,~passed(~Group,~Tile.Tile8,~sample,~Updates.kept_sample(~Tile.Tile8),~Unit,~zero_value,~Logical,~original)),PaddedPieces.Padded{padding,Pass{Unit{},logical,groups,cells,fallback}},U32.to_nat(limit),
        Each.content_map(~Tile.Tile8,~Tile.Tile8,~BNCells.Mapping<Group>,~mapped_model(~Group,~sample_at(~Group,~Tile.Tile8,~sample)),BNCells.Mapping{groups,cells,padding},left),
        Each.content_map(~Tile.Tile8,~Tile.Tile8,~BNCells.Mapping<Group>,~mapped_model(~Group,~sample_at(~Group,~Tile.Tile8,~sample)),BNCells.Mapping{groups,cells,padding},right),
        mapped_facts(~Group,~sample,~Logical,~original,groups,cells,fallback,logical,padding,limit,rooms,listed,left,left_facts),
        mapped_facts(~Group,~sample,~Logical,~original,groups,cells,fallback,logical,padding,limit,rooms,listed,right,right_facts))

# ---- The forward passes ----

# The fields and values of every input piece, one property of each.
def in_facts(~Logical: Data,~original: Logical -> U32 -> Tile.Tile8,+logical: Logical,+total: Nat,content: Model.Content<Tile.Tile8>,
  fields: Model.ContentAll(~Tile.Tile8,~Content.piece_fields(~Tile.Tile8,total),content),values: Content.Values(~Tile.Tile8,~Logical,~original,logical,content)) ->
  Model.ContentAll(~Tile.Tile8,~piece_in(~Logical,~original,logical,total),content):
  match content:
    case Model.PieceContent{+buffer,+first,+rows}: KernelGemm.Both{fields,values}
    case Model.Joined{+left,+right}:
      KernelGemm.Both{left_fields,right_fields} = fields
      KernelGemm.Both{left_values,right_values} = values
      KernelGemm.Both{in_facts(~Logical,~original,logical,total,left,left_fields,left_values),in_facts(~Logical,~original,logical,total,right,right_fields,right_values)}

def input_fields(~Element: Data,+total: Nat,-content: Model.Content<Element>,well: Content.WellFormed<Element,total,content>) -> Model.ContentAll(~Element,~Content.piece_fields(~Element,total),content):
  Content.WellFormed{fields,ready,coverage} = well
  fields

def input_coverage(~Element: Data,+total: Nat,-content: Model.Content<Element>,well: Content.WellFormed<Element,total,content>) ->
  PiecesPlan.Coverage(Model.content_layout(~Element,content),0n,total):
  Content.WellFormed{fields,ready,coverage} = well
  coverage

# What a pass that reads every piece into zero cells leaves: pieces of the
# layout of the input that hold what the parameters of every group make of
# its cells.
def Passed(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,+total: Nat,content: Model.Content<Tile.Tile8>) -> Data:
  KernelGemm.Both<Content.WellFormed<Tile.Tile8,total,content>,Content.Values(~Tile.Tile8,~Context,~value,context,content)>

def passed_of(~Context: Data,~value: Context -> U32 -> Tile.Tile8,+context: Context,+total: Nat,-content: Model.Content<Tile.Tile8>,facts: Facts(~Context,~value,context,total,content),
  coverage: PiecesPlan.Coverage(Model.content_layout(~Tile.Tile8,content),0n,total)) -> Passed(~Context,~value,context,total,content):
  KernelGemm.Both{fields,rest} = facts
  KernelGemm.Both{ready,values} = rest
  KernelGemm.Both{Content.WellFormed{fields,ready,coverage},values}

# The pieces a pass reads into zero cells are well formed and hold their
# values.
def mapped_passed(~Group: Data,~sample: Group -> Tile.Tile8 -> Tile.Tile8,~Logical: Data,~original: Logical -> U32 -> Tile.Tile8,+groups: List<&2,Group>,+cells: U32,+fallback: Group,
  +logical: Logical,+padding: TiledMap.Padding,+limit: U32,+rooms: Rooms<limit,cells,padding>,+listed: {List.length(&2,Group,groups) == U32.to_nat(cells) : Nat},+model: Model.Model<Tile.Tile8>,
  +well: Content.WellFormed<Tile.Tile8,U32.to_nat(limit),Model.content(~Tile.Tile8,model)>,+values: Content.Values(~Tile.Tile8,~Logical,~original,logical,Model.content(~Tile.Tile8,model))) ->
  Passed(~PaddedPieces.Padded<Pass<Unit,Logical,Group>>,~PaddedPieces.padded_value(~Pass<Unit,Logical,Group>,~passed(~Group,~Tile.Tile8,~sample,~Updates.kept_sample(~Tile.Tile8),~Unit,~zero_value,~Logical,~original)),PaddedPieces.Padded{padding,Pass{Unit{},logical,groups,cells,fallback}},U32.to_nat(limit),Each.content_map(~Tile.Tile8,~Tile.Tile8,~BNCells.Mapping<Group>,~mapped_model(~Group,~sample_at(~Group,~Tile.Tile8,~sample)),BNCells.Mapping{groups,cells,padding},model)):
  passed_of(~PaddedPieces.Padded<Pass<Unit,Logical,Group>>,~PaddedPieces.padded_value(~Pass<Unit,Logical,Group>,~passed(~Group,~Tile.Tile8,~sample,~Updates.kept_sample(~Tile.Tile8),~Unit,~zero_value,~Logical,~original)),PaddedPieces.Padded{padding,Pass{Unit{},logical,groups,cells,fallback}},U32.to_nat(limit),Each.content_map(~Tile.Tile8,~Tile.Tile8,~BNCells.Mapping<Group>,~mapped_model(~Group,~sample_at(~Group,~Tile.Tile8,~sample)),BNCells.Mapping{groups,cells,padding},model),
    mapped_facts(~Group,~sample,~Logical,~original,groups,cells,fallback,logical,padding,limit,rooms,listed,model,
      in_facts(~Logical,~original,logical,U32.to_nat(limit),Model.content(~Tile.Tile8,model),input_fields(~Tile.Tile8,U32.to_nat(limit),Model.content(~Tile.Tile8,model),well),values)),
    %Equal.sym(Pieces.Ranges,Model.content_layout(~Tile.Tile8,Each.content_map(~Tile.Tile8,~Tile.Tile8,~BNCells.Mapping<Group>,~mapped_model(~Group,~sample_at(~Group,~Tile.Tile8,~sample)),BNCells.Mapping{groups,cells,padding},model)),Model.layout(~Tile.Tile8,model),
        Each.mapped_layout(~Tile.Tile8,~Tile.Tile8,~BNCells.Mapping<Group>,~mapped_model(~Group,~sample_at(~Group,~Tile.Tile8,~sample)),
          ~mapped_layout_rule(~Group,~sample_at(~Group,~Tile.Tile8,~sample)),BNCells.Mapping{groups,cells,padding},model)) :
      PiecesPlan.Coverage(_,0n,U32.to_nat(limit))
    %Equal.sym(Pieces.Ranges,Model.layout(~Tile.Tile8,model),Model.content_layout(~Tile.Tile8,Model.content(~Tile.Tile8,model)),Model.layout_matches_content(~Tile.Tile8,model)) :
      PiecesPlan.Coverage(_,0n,U32.to_nat(limit))
    input_coverage(~Tile.Tile8,U32.to_nat(limit),Model.content(~Tile.Tile8,model),well))

# Whatever tasks run them, the leaves of a pass see every piece of a model.
def tasked_map(~Output: Type,~Context: Data,~leaf: Context -> U32 -> U32 -> BNCells.Cells() -> Unit -> Partition.Tree<&1,Output,Unit>,decision: Tasks.Decision,+context: Context,
  +model: Model.Model<Tile.Tile8>) ->
  {BNCells.tasked(~BNCells.Cells(),~Output,~Context,~leaf,decision,context,Model.pack(~Tile.Tile8,model)) == Each.leaf_map(~Tile.Tile8,~Output,~Context,~leaf,context,model) :
    Partition.Tree<&1,Output,Unit>}:
  match decision:
    case Tasks.Decision{+parallel,+tasks}: Each.grouped(~Tile.Tile8,~Output,~Context,~leaf,tasks,parallel,context,model)

def outputs_contents(pair: Pieces.Pieces(~Tile.Tile8) & Pieces.Pieces(~Tile.Tile8)) ->
  {Model.contents(~Tile.Tile8,BNCells.outputs(pair)) == Pair.snd(Model.Content<Tile.Tile8>,Model.Content<Tile.Tile8>,pieces_contents(pair)) : Model.Content<Tile.Tile8>}:
  (input,output) = pair
  {==}

# The output of the evaluation pass: every piece read into zero cells.
def evaluated_contents(+groups: List<&2,Kernel.Normal>,+cells: U32,+padding: TiledMap.Padding,+shape: List<&2,U32>,+workers: U32,+model: Model.Model<Tile.Tile8>) ->
  {Model.contents(~Tile.Tile8,BNCells.evaluated(groups,cells,padding,shape,workers,Model.pack(~Tile.Tile8,model))) == Each.content_map(~Tile.Tile8,~Tile.Tile8,~BNCells.Mapping<Kernel.Normal>,~mapped_model(~Kernel.Normal,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of)),BNCells.Mapping{groups,cells,padding},model) : Model.Content<Tile.Tile8>}:
  +context = {BNCells.Mapping{groups,cells,padding} : BNCells.Mapping<Kernel.Normal>}
  Equal.trans(Model.Content<Tile.Tile8>,Model.contents(~Tile.Tile8,BNCells.outputs(Tensors.unzipped(~Tile.Tile8,BNCells.tasked(~BNCells.Cells(),~(BNCells.Cells() & BNCells.Cells()),~BNCells.Mapping<Kernel.Normal>,~BNCells.map_leaf(~Kernel.Normal,~Kernel.read_output,~KernelCertificate.read_output),Tasks.decide(Tensors.grid_of(shape),workers,6.0,2.0,Tasks.no_extra()),context,Model.pack(~Tile.Tile8,model))))),Pair.snd(Model.Content<Tile.Tile8>,Model.Content<Tile.Tile8>,pieces_contents(Tensors.unzipped(~Tile.Tile8,BNCells.tasked(~BNCells.Cells(),~(BNCells.Cells() & BNCells.Cells()),~BNCells.Mapping<Kernel.Normal>,~BNCells.map_leaf(~Kernel.Normal,~Kernel.read_output,~KernelCertificate.read_output),Tasks.decide(Tensors.grid_of(shape),workers,6.0,2.0,Tasks.no_extra()),context,Model.pack(~Tile.Tile8,model))))),Each.content_map(~Tile.Tile8,~Tile.Tile8,~BNCells.Mapping<Kernel.Normal>,~mapped_model(~Kernel.Normal,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of)),BNCells.Mapping{groups,cells,padding},model),
    outputs_contents(Tensors.unzipped(~Tile.Tile8,BNCells.tasked(~BNCells.Cells(),~(BNCells.Cells() & BNCells.Cells()),~BNCells.Mapping<Kernel.Normal>,~BNCells.map_leaf(~Kernel.Normal,~Kernel.read_output,~KernelCertificate.read_output),Tasks.decide(Tensors.grid_of(shape),workers,6.0,2.0,Tasks.no_extra()),context,Model.pack(~Tile.Tile8,model)))),
    %Equal.sym(Partition.Tree<&1,BNCells.Cells() & BNCells.Cells(),Unit>,BNCells.tasked(~BNCells.Cells(),~(BNCells.Cells() & BNCells.Cells()),~BNCells.Mapping<Kernel.Normal>,~BNCells.map_leaf(~Kernel.Normal,~Kernel.read_output,~KernelCertificate.read_output),Tasks.decide(Tensors.grid_of(shape),workers,6.0,2.0,Tasks.no_extra()),context,Model.pack(~Tile.Tile8,model)),Each.leaf_map(~Tile.Tile8,~(BNCells.Cells() & BNCells.Cells()),~BNCells.Mapping<Kernel.Normal>,~BNCells.map_leaf(~Kernel.Normal,~Kernel.read_output,~KernelCertificate.read_output),context,model),
        tasked_map(~(BNCells.Cells() & BNCells.Cells()),~BNCells.Mapping<Kernel.Normal>,~BNCells.map_leaf(~Kernel.Normal,~Kernel.read_output,~KernelCertificate.read_output),Tasks.decide(Tensors.grid_of(shape),workers,6.0,2.0,Tasks.no_extra()),context,model)) :
      {Pair.snd(Model.Content<Tile.Tile8>,Model.Content<Tile.Tile8>,pieces_contents(Tensors.unzipped(~Tile.Tile8,_))) == Each.content_map(~Tile.Tile8,~Tile.Tile8,~BNCells.Mapping<Kernel.Normal>,~mapped_model(~Kernel.Normal,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of)),BNCells.Mapping{groups,cells,padding},model) : Model.Content<Tile.Tile8>}
    %Equal.sym(Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>,pieces_contents(Tensors.unzipped(~Tile.Tile8,Each.leaf_map(~Tile.Tile8,~(BNCells.Cells() & BNCells.Cells()),~BNCells.Mapping<Kernel.Normal>,~BNCells.map_leaf(~Kernel.Normal,~Kernel.read_output,~KernelCertificate.read_output),context,model))),
        (Each.content_map(~Tile.Tile8,~Tile.Tile8,~BNCells.Mapping<Kernel.Normal>,~kept_model(~BNCells.Mapping<Kernel.Normal>),context,model),Each.content_map(~Tile.Tile8,~Tile.Tile8,~BNCells.Mapping<Kernel.Normal>,~mapped_model(~Kernel.Normal,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of)),BNCells.Mapping{groups,cells,padding},model)),
        unzipped_contents(~BNCells.Mapping<Kernel.Normal>,~BNCells.map_leaf(~Kernel.Normal,~Kernel.read_output,~KernelCertificate.read_output),~kept_model(~BNCells.Mapping<Kernel.Normal>),~mapped_model(~Kernel.Normal,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of)),
          ~mapped_rule(~Kernel.Normal,~Kernel.read_output,~KernelCertificate.read_output,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of),~output_matches),context,model)) :
      {Pair.snd(Model.Content<Tile.Tile8>,Model.Content<Tile.Tile8>,_) == Each.content_map(~Tile.Tile8,~Tile.Tile8,~BNCells.Mapping<Kernel.Normal>,~mapped_model(~Kernel.Normal,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of)),BNCells.Mapping{groups,cells,padding},model) : Model.Content<Tile.Tile8>}
    {==})

# The forward pass: xhat and the output, every piece read into zero cells.
def forward_pieces(+groups: List<&2,Kernel.Normal>,+cells: U32,+padding: TiledMap.Padding,+shape: List<&2,U32>,+workers: U32,+model: Model.Model<Tile.Tile8>) ->
  {pieces_contents(BNCells.forward(groups,cells,padding,shape,workers,Model.pack(~Tile.Tile8,model))) == (Each.content_map(~Tile.Tile8,~Tile.Tile8,~BNCells.Mapping<Kernel.Normal>,~mapped_model(~Kernel.Normal,~sample_at(~Kernel.Normal,~Tile.Tile8,~Kernel.normalized)),BNCells.Mapping{groups,cells,padding},model),Each.content_map(~Tile.Tile8,~Tile.Tile8,~BNCells.Mapping<Kernel.Normal>,~mapped_model(~Kernel.Normal,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of)),BNCells.Mapping{groups,cells,padding},model)) : Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>}:
  +context = {BNCells.Mapping{groups,cells,padding} : BNCells.Mapping<Kernel.Normal>}
  %Equal.sym(Partition.Tree<&1,BNCells.Cells() & BNCells.Cells(),Unit>,BNCells.tasked(~BNCells.Cells(),~(BNCells.Cells() & BNCells.Cells()),~BNCells.Mapping<Kernel.Normal>,~BNCells.forward_leaf,Tasks.decide(Tensors.grid_of(shape),workers,10.0,2.0,Tasks.no_extra()),context,Model.pack(~Tile.Tile8,model)),Each.leaf_map(~Tile.Tile8,~(BNCells.Cells() & BNCells.Cells()),~BNCells.Mapping<Kernel.Normal>,~BNCells.forward_leaf,context,model),
      tasked_map(~(BNCells.Cells() & BNCells.Cells()),~BNCells.Mapping<Kernel.Normal>,~BNCells.forward_leaf,Tasks.decide(Tensors.grid_of(shape),workers,10.0,2.0,Tasks.no_extra()),context,model)) :
    {pieces_contents(Tensors.unzipped(~Tile.Tile8,_)) == (Each.content_map(~Tile.Tile8,~Tile.Tile8,~BNCells.Mapping<Kernel.Normal>,~mapped_model(~Kernel.Normal,~sample_at(~Kernel.Normal,~Tile.Tile8,~Kernel.normalized)),BNCells.Mapping{groups,cells,padding},model),Each.content_map(~Tile.Tile8,~Tile.Tile8,~BNCells.Mapping<Kernel.Normal>,~mapped_model(~Kernel.Normal,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of)),BNCells.Mapping{groups,cells,padding},model)) : Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>}
  unzipped_contents(~BNCells.Mapping<Kernel.Normal>,~BNCells.forward_leaf,~mapped_model(~Kernel.Normal,~sample_at(~Kernel.Normal,~Tile.Tile8,~Kernel.normalized)),
    ~mapped_model(~Kernel.Normal,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of)),~forward_rule,context,model)

# The evaluation pass leaves pieces of the layout of its input that hold
# the output of every group.
def evaluated(~Logical: Data,~original: Logical -> U32 -> Tile.Tile8,+groups: List<&2,Kernel.Normal>,+cells: U32,+fallback: Kernel.Normal,+logical: Logical,+padding: TiledMap.Padding,+limit: U32,
  +rooms: Rooms<limit,cells,padding>,+listed: {List.length(&2,Kernel.Normal,groups) == U32.to_nat(cells) : Nat},+shape: List<&2,U32>,+workers: U32,+model: Model.Model<Tile.Tile8>,
  +well: Content.WellFormed<Tile.Tile8,U32.to_nat(limit),Model.content(~Tile.Tile8,model)>,+values: Content.Values(~Tile.Tile8,~Logical,~original,logical,Model.content(~Tile.Tile8,model))) ->
  Passed(~PaddedPieces.Padded<Pass<Unit,Logical,Kernel.Normal>>,~PaddedPieces.padded_value(~Pass<Unit,Logical,Kernel.Normal>,~passed(~Kernel.Normal,~Tile.Tile8,~output_of,~Updates.kept_sample(~Tile.Tile8),~Unit,~zero_value,~Logical,~original)),PaddedPieces.Padded{padding,Pass{Unit{},logical,groups,cells,fallback}},U32.to_nat(limit),
    Model.contents(~Tile.Tile8,BNCells.evaluated(groups,cells,padding,shape,workers,Model.pack(~Tile.Tile8,model)))):
  %Equal.sym(Model.Content<Tile.Tile8>,Model.contents(~Tile.Tile8,BNCells.evaluated(groups,cells,padding,shape,workers,Model.pack(~Tile.Tile8,model))),Each.content_map(~Tile.Tile8,~Tile.Tile8,~BNCells.Mapping<Kernel.Normal>,~mapped_model(~Kernel.Normal,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of)),BNCells.Mapping{groups,cells,padding},model),evaluated_contents(groups,cells,padding,shape,workers,model)) :
    Passed(~PaddedPieces.Padded<Pass<Unit,Logical,Kernel.Normal>>,~PaddedPieces.padded_value(~Pass<Unit,Logical,Kernel.Normal>,~passed(~Kernel.Normal,~Tile.Tile8,~output_of,~Updates.kept_sample(~Tile.Tile8),~Unit,~zero_value,~Logical,~original)),PaddedPieces.Padded{padding,Pass{Unit{},logical,groups,cells,fallback}},U32.to_nat(limit),_)
  mapped_passed(~Kernel.Normal,~output_of,~Logical,~original,groups,cells,fallback,logical,padding,limit,rooms,listed,model,well,values)

def PairPassed(~Logical: Data,~original: Logical -> U32 -> Tile.Tile8,+groups: List<&2,Kernel.Normal>,+cells: U32,+fallback: Kernel.Normal,+logical: Logical,+padding: TiledMap.Padding,+total: Nat,
  pair: Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>) -> Data:
  (xhat,output) = pair
  KernelGemm.Both<Passed(~PaddedPieces.Padded<Pass<Unit,Logical,Kernel.Normal>>,~PaddedPieces.padded_value(~Pass<Unit,Logical,Kernel.Normal>,~passed(~Kernel.Normal,~Tile.Tile8,~Kernel.normalized,~Updates.kept_sample(~Tile.Tile8),~Unit,~zero_value,~Logical,~original)),PaddedPieces.Padded{padding,Pass{Unit{},logical,groups,cells,fallback}},total,xhat),
    Passed(~PaddedPieces.Padded<Pass<Unit,Logical,Kernel.Normal>>,~PaddedPieces.padded_value(~Pass<Unit,Logical,Kernel.Normal>,~passed(~Kernel.Normal,~Tile.Tile8,~output_of,~Updates.kept_sample(~Tile.Tile8),~Unit,~zero_value,~Logical,~original)),PaddedPieces.Padded{padding,Pass{Unit{},logical,groups,cells,fallback}},total,output)>

# The forward pass leaves xhat and the output, both pieces of the layout of
# its input.
def forward(~Logical: Data,~original: Logical -> U32 -> Tile.Tile8,+groups: List<&2,Kernel.Normal>,+cells: U32,+fallback: Kernel.Normal,+logical: Logical,+padding: TiledMap.Padding,+limit: U32,
  +rooms: Rooms<limit,cells,padding>,+listed: {List.length(&2,Kernel.Normal,groups) == U32.to_nat(cells) : Nat},+shape: List<&2,U32>,+workers: U32,+model: Model.Model<Tile.Tile8>,
  +well: Content.WellFormed<Tile.Tile8,U32.to_nat(limit),Model.content(~Tile.Tile8,model)>,+values: Content.Values(~Tile.Tile8,~Logical,~original,logical,Model.content(~Tile.Tile8,model))) ->
  PairPassed(~Logical,~original,groups,cells,fallback,logical,padding,U32.to_nat(limit),pieces_contents(BNCells.forward(groups,cells,padding,shape,workers,Model.pack(~Tile.Tile8,model)))):
  %Equal.sym(Model.Content<Tile.Tile8> & Model.Content<Tile.Tile8>,pieces_contents(BNCells.forward(groups,cells,padding,shape,workers,Model.pack(~Tile.Tile8,model))),(Each.content_map(~Tile.Tile8,~Tile.Tile8,~BNCells.Mapping<Kernel.Normal>,~mapped_model(~Kernel.Normal,~sample_at(~Kernel.Normal,~Tile.Tile8,~Kernel.normalized)),BNCells.Mapping{groups,cells,padding},model),Each.content_map(~Tile.Tile8,~Tile.Tile8,~BNCells.Mapping<Kernel.Normal>,~mapped_model(~Kernel.Normal,~sample_at(~Kernel.Normal,~Tile.Tile8,~output_of)),BNCells.Mapping{groups,cells,padding},model)),forward_pieces(groups,cells,padding,shape,workers,model)) :
    PairPassed(~Logical,~original,groups,cells,fallback,logical,padding,U32.to_nat(limit),_)
  KernelGemm.Both{mapped_passed(~Kernel.Normal,~Kernel.normalized,~Logical,~original,groups,cells,fallback,logical,padding,limit,rooms,listed,model,well,values),
    mapped_passed(~Kernel.Normal,~output_of,~Logical,~original,groups,cells,fallback,logical,padding,limit,rooms,listed,model,well,values)}

# ---- The backward pass ----

# The cells dy of a pair leaves, updated by its groups reading xhat.
def back_content(context: BNCells.Mapping<Kernel.Back>,pair: Each.PiecePair<Tile.Tile8>,+first: U32,+rows: U32) -> Contents():
  BNCells.Mapping{groups,+cells,+padding} = context
  Each.PiecePair{+dy,xhat} = pair
  Padding.step(padding,first,rows,Buffer.Content{groups_tree(~Kernel.Back,~Kernel.Sampled,~sample_at(~Kernel.Back,~Kernel.Sampled,~back_of),~Kernel.back_combine,groups,first,rows,cells,0,Buffer.tree(~Tile.Tile8,xhat),Buffer.tree(~Tile.Tile8,dy)),Buffer.length(~Tile.Tile8,dy)})

def back_rule(+context: BNCells.Mapping<Kernel.Back>,+first: U32,+rows: U32,+pair: Each.PiecePair<Tile.Tile8>) ->
  {Model.contents(~Tile.Tile8,BNCells.back_leaf(context,first,rows,Each.pair_pack(~Tile.Tile8,pair),Unit{})) ==
    Model.PieceContent{back_content(context,pair,first,rows),first,rows} : Model.Content<Tile.Tile8>}:
  BNCells.Mapping{+groups,+cells,+padding} = context
  Each.PiecePair{+dy,+xhat} = pair
  Buffer.Model{+dy_tree,+dy_length,+dy_certificate} = dy
  Buffer.Model{+xhat_tree,+xhat_length,+xhat_certificate} = xhat
  +dy_model = {Buffer.Model{dy_tree,dy_length,dy_certificate} : Buffer.Model<Tile.Tile8>}
  +xhat_model = {Buffer.Model{xhat_tree,xhat_length,xhat_certificate} : Buffer.Model<Tile.Tile8>}
  target = {Model.PieceContent{Padding.step(padding,first,rows,Buffer.Content{groups_tree(~Kernel.Back,~Kernel.Sampled,~sample_at(~Kernel.Back,~Kernel.Sampled,~back_of),~Kernel.back_combine,groups,first,rows,cells,0,xhat_tree,dy_tree),dy_length}),first,rows} : Model.Content<Tile.Tile8>}
  %Equal.sym(Contents(),Buffer.contents(~Tile.Tile8,TiledMap.padded(padding,first,rows,BNCells.output_of(BNCells.each_group(~Kernel.Back,~BNCells.back_buffers,groups,first,rows,cells,0,Buffer.pack(~Tile.Tile8,xhat_model),Buffer.pack(~Tile.Tile8,dy_model))))),
      Padding.step(padding,first,rows,Buffer.contents(~Tile.Tile8,BNCells.output_of(BNCells.each_group(~Kernel.Back,~BNCells.back_buffers,groups,first,rows,cells,0,Buffer.pack(~Tile.Tile8,xhat_model),Buffer.pack(~Tile.Tile8,dy_model))))),Padding.rule(padding,first,rows,BNCells.output_of(BNCells.each_group(~Kernel.Back,~BNCells.back_buffers,groups,first,rows,cells,0,Buffer.pack(~Tile.Tile8,xhat_model),Buffer.pack(~Tile.Tile8,dy_model))))) :
    {Model.PieceContent{_,first,rows} == target : Model.Content<Tile.Tile8>}
  %Equal.sym(Contents(),Buffer.contents(~Tile.Tile8,BNCells.output_of(BNCells.each_group(~Kernel.Back,~BNCells.back_buffers,groups,first,rows,cells,0,Buffer.pack(~Tile.Tile8,xhat_model),Buffer.pack(~Tile.Tile8,dy_model)))),Pair.snd(Contents(),Contents(),pair_contents(BNCells.each_group(~Kernel.Back,~BNCells.back_buffers,groups,first,rows,cells,0,Buffer.pack(~Tile.Tile8,xhat_model),Buffer.pack(~Tile.Tile8,dy_model)))),output_contents(BNCells.each_group(~Kernel.Back,~BNCells.back_buffers,groups,first,rows,cells,0,Buffer.pack(~Tile.Tile8,xhat_model),Buffer.pack(~Tile.Tile8,dy_model)))) :
    {Model.PieceContent{Padding.step(padding,first,rows,_),first,rows} == target : Model.Content<Tile.Tile8>}
  %Equal.sym(Contents() & Contents(),pair_contents(BNCells.each_group(~Kernel.Back,~BNCells.back_buffers,groups,first,rows,cells,0,Buffer.pack(~Tile.Tile8,xhat_model),Buffer.pack(~Tile.Tile8,dy_model))),
      (Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,xhat_model)),groups_content(~Kernel.Back,~Kernel.Sampled,~sample_at(~Kernel.Back,~Kernel.Sampled,~back_of),~Kernel.back_combine,groups,first,rows,cells,0,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,xhat_model)),Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,dy_model)))),
      each_contents(~Kernel.Back,~Kernel.Sampled,~sample_at(~Kernel.Back,~Kernel.Sampled,~back_of),~Kernel.back_combine,~BNCells.back_buffers,~back_law,groups,first,rows,cells,0,Buffer.pack(~Tile.Tile8,xhat_model),Buffer.pack(~Tile.Tile8,dy_model),Buffer.Observation{xhat_model,{==}},Buffer.Observation{dy_model,{==}})) :
    {Model.PieceContent{Padding.step(padding,first,rows,Pair.snd(Contents(),Contents(),_)),first,rows} == target : Model.Content<Tile.Tile8>}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,xhat_tree)),xhat_tree,StorageLaws.reflect_packed(Tile.Tile8,xhat_tree)) :
    {Model.PieceContent{Padding.step(padding,first,rows,groups_content(~Kernel.Back,~Kernel.Sampled,~sample_at(~Kernel.Back,~Kernel.Sampled,~back_of),~Kernel.back_combine,groups,first,rows,cells,0,Buffer.Content{_,xhat_length},Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,dy_model)))),first,rows} == target : Model.Content<Tile.Tile8>}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,dy_tree)),dy_tree,StorageLaws.reflect_packed(Tile.Tile8,dy_tree)) :
    {Model.PieceContent{Padding.step(padding,first,rows,groups_content(~Kernel.Back,~Kernel.Sampled,~sample_at(~Kernel.Back,~Kernel.Sampled,~back_of),~Kernel.back_combine,groups,first,rows,cells,0,Buffer.Content{xhat_tree,xhat_length},Buffer.Content{_,dy_length})),first,rows} == target : Model.Content<Tile.Tile8>}
  %Equal.sym(Contents(),groups_content(~Kernel.Back,~Kernel.Sampled,~sample_at(~Kernel.Back,~Kernel.Sampled,~back_of),~Kernel.back_combine,groups,first,rows,cells,0,Buffer.Content{xhat_tree,xhat_length},Buffer.Content{dy_tree,dy_length}),
      Buffer.Content{groups_tree(~Kernel.Back,~Kernel.Sampled,~sample_at(~Kernel.Back,~Kernel.Sampled,~back_of),~Kernel.back_combine,groups,first,rows,cells,0,xhat_tree,dy_tree),dy_length},
      groups_content_exact(~Kernel.Back,~Kernel.Sampled,~sample_at(~Kernel.Back,~Kernel.Sampled,~back_of),~Kernel.back_combine,groups,first,rows,cells,0,Buffer.Content{xhat_tree,xhat_length},Buffer.Content{dy_tree,dy_length})) :
    {Model.PieceContent{Padding.step(padding,first,rows,_),first,rows} == target : Model.Content<Tile.Tile8>}
  {==}

def back_contents(+context: BNCells.Mapping<Kernel.Back>,tree: Model.Shaped<Each.PiecePair<Tile.Tile8>>) ->
  {Model.contents(~Tile.Tile8,Each.shaped_leaf_map(~Each.PiecePair<Tile.Tile8>,~BNCells.CellPair(),~Each.pair_pack(~Tile.Tile8),~BNCells.Cells(),~BNCells.Mapping<Kernel.Back>,~BNCells.back_leaf,context,tree)) == Model.shaped_content(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~BNCells.Mapping<Kernel.Back>,~back_content,context,tree) : Model.Content<Tile.Tile8>}:
  match tree:
    case Model.Piece{+pair,+first,+rows}: back_rule(context,first,rows,pair)
    case Model.Pieces{+left,+right}:
      %Equal.sym(Model.Content<Tile.Tile8>,Model.contents(~Tile.Tile8,Each.shaped_leaf_map(~Each.PiecePair<Tile.Tile8>,~BNCells.CellPair(),~Each.pair_pack(~Tile.Tile8),~BNCells.Cells(),~BNCells.Mapping<Kernel.Back>,~BNCells.back_leaf,context,left)),Model.shaped_content(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~BNCells.Mapping<Kernel.Back>,~back_content,context,left),back_contents(context,left)) :
        {Model.Joined{_,Model.contents(~Tile.Tile8,Each.shaped_leaf_map(~Each.PiecePair<Tile.Tile8>,~BNCells.CellPair(),~Each.pair_pack(~Tile.Tile8),~BNCells.Cells(),~BNCells.Mapping<Kernel.Back>,~BNCells.back_leaf,context,right))} == Model.Joined{Model.shaped_content(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~BNCells.Mapping<Kernel.Back>,~back_content,context,left),Model.shaped_content(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~BNCells.Mapping<Kernel.Back>,~back_content,context,right)} : Model.Content<Tile.Tile8>}
      %Equal.sym(Model.Content<Tile.Tile8>,Model.contents(~Tile.Tile8,Each.shaped_leaf_map(~Each.PiecePair<Tile.Tile8>,~BNCells.CellPair(),~Each.pair_pack(~Tile.Tile8),~BNCells.Cells(),~BNCells.Mapping<Kernel.Back>,~BNCells.back_leaf,context,right)),Model.shaped_content(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~BNCells.Mapping<Kernel.Back>,~back_content,context,right),back_contents(context,right)) :
        {Model.Joined{Model.shaped_content(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~BNCells.Mapping<Kernel.Back>,~back_content,context,left),_} == Model.Joined{Model.shaped_content(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~BNCells.Mapping<Kernel.Back>,~back_content,context,left),Model.shaped_content(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~BNCells.Mapping<Kernel.Back>,~back_content,context,right)} : Model.Content<Tile.Tile8>}
      {==}

def back_layout(+context: BNCells.Mapping<Kernel.Back>,tree: Model.Shaped<Each.PiecePair<Tile.Tile8>>) ->
  {Model.content_layout(~Tile.Tile8,Model.shaped_content(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~BNCells.Mapping<Kernel.Back>,~back_content,context,tree)) == Model.shaped_plan(~Each.PiecePair<Tile.Tile8>,~Each.piece_rows(~Each.PiecePair<Tile.Tile8>),tree) : Pieces.Ranges}:
  match tree:
    case Model.Piece{+pair,+first,+rows}: {==}
    case Model.Pieces{+left,+right}:
      %Equal.sym(Pieces.Ranges,Model.content_layout(~Tile.Tile8,Model.shaped_content(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~BNCells.Mapping<Kernel.Back>,~back_content,context,left)),Model.shaped_plan(~Each.PiecePair<Tile.Tile8>,~Each.piece_rows(~Each.PiecePair<Tile.Tile8>),left),back_layout(context,left)) :
        {Pieces.Split{_,Model.content_layout(~Tile.Tile8,Model.shaped_content(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~BNCells.Mapping<Kernel.Back>,~back_content,context,right))} == Pieces.Split{Model.shaped_plan(~Each.PiecePair<Tile.Tile8>,~Each.piece_rows(~Each.PiecePair<Tile.Tile8>),left),Model.shaped_plan(~Each.PiecePair<Tile.Tile8>,~Each.piece_rows(~Each.PiecePair<Tile.Tile8>),right)} : Pieces.Ranges}
      %Equal.sym(Pieces.Ranges,Model.content_layout(~Tile.Tile8,Model.shaped_content(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~BNCells.Mapping<Kernel.Back>,~back_content,context,right)),Model.shaped_plan(~Each.PiecePair<Tile.Tile8>,~Each.piece_rows(~Each.PiecePair<Tile.Tile8>),right),back_layout(context,right)) :
        {Pieces.Split{Model.shaped_plan(~Each.PiecePair<Tile.Tile8>,~Each.piece_rows(~Each.PiecePair<Tile.Tile8>),left),_} == Pieces.Split{Model.shaped_plan(~Each.PiecePair<Tile.Tile8>,~Each.piece_rows(~Each.PiecePair<Tile.Tile8>),left),Model.shaped_plan(~Each.PiecePair<Tile.Tile8>,~Each.piece_rows(~Each.PiecePair<Tile.Tile8>),right)} : Pieces.Ranges}
      {==}

# Whatever tasks run them, the leaves of the backward pass see every pair.
def tasked_pairs(decision: Tasks.Decision,+context: BNCells.Mapping<Kernel.Back>,+tree: Model.Shaped<Each.PiecePair<Tile.Tile8>>) ->
  {BNCells.tasked(~BNCells.CellPair(),~BNCells.Cells(),~BNCells.Mapping<Kernel.Back>,~BNCells.back_leaf,decision,context,Model.shaped_pack(~Each.PiecePair<Tile.Tile8>,~BNCells.CellPair(),~Each.pair_pack(~Tile.Tile8),tree)) == Each.shaped_leaf_map(~Each.PiecePair<Tile.Tile8>,~BNCells.CellPair(),~Each.pair_pack(~Tile.Tile8),~BNCells.Cells(),~BNCells.Mapping<Kernel.Back>,~BNCells.back_leaf,context,tree) : Pieces.Pieces(~Tile.Tile8)}:
  match decision:
    case Tasks.Decision{+parallel,+tasks}:
      Equal.trans(Pieces.Pieces(~Tile.Tile8),Pieces.grouped(~BNCells.CellPair(),~BNCells.Cells(),~BNCells.Mapping<Kernel.Back>,~BNCells.back_leaf,tasks,parallel,context,Model.shaped_pack(~Each.PiecePair<Tile.Tile8>,~BNCells.CellPair(),~Each.pair_pack(~Tile.Tile8),tree)),Pieces.group_run(~BNCells.CellPair(),~BNCells.Cells(),~BNCells.Mapping<Kernel.Back>,~BNCells.back_leaf,context,Model.shaped_pack(~Each.PiecePair<Tile.Tile8>,~BNCells.CellPair(),~Each.pair_pack(~Tile.Tile8),tree)),Each.shaped_leaf_map(~Each.PiecePair<Tile.Tile8>,~BNCells.CellPair(),~Each.pair_pack(~Tile.Tile8),~BNCells.Cells(),~BNCells.Mapping<Kernel.Back>,~BNCells.back_leaf,context,tree),
        Walk.grouped(~Each.PiecePair<Tile.Tile8>,~BNCells.CellPair(),~Each.pair_pack(~Tile.Tile8),~BNCells.Cells(),~BNCells.Mapping<Kernel.Back>,~BNCells.back_leaf,tasks,parallel,context,tree),
        Each.shaped_group_serial(~Each.PiecePair<Tile.Tile8>,~BNCells.CellPair(),~Each.pair_pack(~Tile.Tile8),~BNCells.Cells(),~BNCells.Mapping<Kernel.Back>,~BNCells.back_leaf,context,tree))

def backward_contents(+groups: List<&2,Kernel.Back>,+cells: U32,+padding: TiledMap.Padding,+shape: List<&2,U32>,+workers: U32,+tree: Model.Shaped<Each.PiecePair<Tile.Tile8>>) ->
  {Model.contents(~Tile.Tile8,BNCells.backward(groups,cells,padding,shape,workers,Model.shaped_pack(~Each.PiecePair<Tile.Tile8>,~BNCells.CellPair(),~Each.pair_pack(~Tile.Tile8),tree))) == Model.shaped_content(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~BNCells.Mapping<Kernel.Back>,~back_content,BNCells.Mapping{groups,cells,padding},tree) : Model.Content<Tile.Tile8>}:
  +context = {BNCells.Mapping{groups,cells,padding} : BNCells.Mapping<Kernel.Back>}
  %Equal.sym(Pieces.Pieces(~Tile.Tile8),BNCells.tasked(~BNCells.CellPair(),~BNCells.Cells(),~BNCells.Mapping<Kernel.Back>,~BNCells.back_leaf,Tasks.decide(Tensors.grid_of(shape),workers,8.0,2.0,Tasks.no_extra()),context,Model.shaped_pack(~Each.PiecePair<Tile.Tile8>,~BNCells.CellPair(),~Each.pair_pack(~Tile.Tile8),tree)),Each.shaped_leaf_map(~Each.PiecePair<Tile.Tile8>,~BNCells.CellPair(),~Each.pair_pack(~Tile.Tile8),~BNCells.Cells(),~BNCells.Mapping<Kernel.Back>,~BNCells.back_leaf,context,tree),tasked_pairs(Tasks.decide(Tensors.grid_of(shape),workers,8.0,2.0,Tasks.no_extra()),context,tree)) :
    {Model.contents(~Tile.Tile8,_) == Model.shaped_content(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~BNCells.Mapping<Kernel.Back>,~back_content,context,tree) : Model.Content<Tile.Tile8>}
  back_contents(context,tree)

# What a pair stands on: the fields, readiness and values of its dy, and the
# values of its xhat.
type BackScope<-DL: Data,-XL: Data> is Data:
  BackScope{dy: DL,xhat: XL,total: Nat}

def dy_part(~DL: Data,~dv: DL -> U32 -> Tile.Tile8,~XL: Data,+scope: BackScope<DL,XL>,+first: U32,+rows: U32,+buffer: Contents()) -> Data:
  BackScope{+dy,xhat,+total} = scope
  piece_full(~DL,~dv,dy,total,first,rows,buffer)

def xhat_part(~DL: Data,~XL: Data,~xv: XL -> U32 -> Tile.Tile8,+scope: BackScope<DL,XL>,+first: U32,+rows: U32,+buffer: Contents()) -> Data:
  BackScope{dy,+xhat,total} = scope
  Content.piece_values(~Tile.Tile8,~XL,~xv,xhat,first,rows,buffer)

# A pair leaves dy with the fields and readiness it had, holding what the
# parameters of every group make of dy and xhat.
def back_piece(~DL: Data,~dv: DL -> U32 -> Tile.Tile8,~XL: Data,~xv: XL -> U32 -> Tile.Tile8,+groups: List<&2,Kernel.Back>,+cells: U32,+fallback: Kernel.Back,+dl: DL,+xl: XL,+padding: TiledMap.Padding,+limit: U32,+rooms: Rooms<limit,cells,padding>,+listed: {List.length(&2,Kernel.Back,groups) == U32.to_nat(cells) : Nat},+first: U32,+rows: U32,+dy: Buffer.Model<Tile.Tile8>,+xhat: Buffer.Model<Tile.Tile8>,
  facts: Each.PairAt(~Tile.Tile8,~BackScope<DL,XL>,~dy_part(~DL,~dv,~XL),~xhat_part(~DL,~XL,~xv),BackScope{dl,xl,U32.to_nat(limit)},Each.PiecePair{dy,xhat},first,rows)) ->
  piece_full(~Pass<DL,XL,Kernel.Back>,~passed(~Kernel.Back,~Kernel.Sampled,~back_of,~Kernel.back_combine,~DL,~dv,~XL,~xv),Pass{dl,xl,groups,cells,fallback},U32.to_nat(limit),first,rows,Buffer.Content{groups_tree(~Kernel.Back,~Kernel.Sampled,~sample_at(~Kernel.Back,~Kernel.Sampled,~back_of),~Kernel.back_combine,groups,first,rows,cells,0,Buffer.tree(~Tile.Tile8,xhat),Buffer.tree(~Tile.Tile8,dy)),Buffer.length(~Tile.Tile8,dy)}):
  Rooms{+cap,+cap_ok,+double_room,+positive,+total_room,+allocatable,fit,whole} = rooms
  Buffer.Model{+dy_tree,+dy_length,dy_certificate} = dy
  Buffer.Model{+xhat_tree,+xhat_length,xhat_certificate} = xhat
  KernelGemm.Both{dy_facts,+xhat_values} = facts
  KernelGemm.Both{fields,rest} = dy_facts
  Content.Fields{+length_exact,+inside} = fields
  KernelGemm.Both{ready,+dy_values} = rest
  Buffer.ContentReady{+depth,+balanced,+depth_ok,+tree_room} = ready
  +rows_limit = Order.transitive(U32.to_nat(rows),Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(limit),Order.right_le(U32.to_nat(first),U32.to_nat(rows)),inside)
  +room = {Room{cap,cap_ok,double_room,positive,
    Order.transitive(Nat.add(U32.to_nat(rows),U32.to_nat(cells)),Nat.add(U32.to_nat(limit),U32.to_nat(cells)),U32.to_nat(cap),
      Order.add_le(U32.to_nat(rows),U32.to_nat(limit),U32.to_nat(cells),rows_limit),total_room)} : Room<rows,cells>}
  +piece = {Piece{room,limit,inside,depth,balanced,depth_ok,tree_room,length_exact} : Piece<first,rows,cells,dy_tree,dy_length>}
  KernelGemm.Both{Content.Fields{length_exact,inside},
    KernelGemm.Both{piece_ready(~Kernel.Back,~Kernel.Sampled,~back_of,~Kernel.back_combine,groups,first,rows,cells,xhat_tree,dy_tree,dy_length,piece),
      piece_values(~Kernel.Back,~Kernel.Sampled,~back_of,~Kernel.back_combine,~DL,~dv,~XL,~xv,groups,first,rows,cells,fallback,dl,xl,xhat_tree,xhat_length,dy_tree,dy_length,piece,listed,xhat_values,dy_values)}}

def back_facts(~DL: Data,~dv: DL -> U32 -> Tile.Tile8,~XL: Data,~xv: XL -> U32 -> Tile.Tile8,+groups: List<&2,Kernel.Back>,+cells: U32,+fallback: Kernel.Back,+dl: DL,+xl: XL,+padding: TiledMap.Padding,+limit: U32,+rooms: Rooms<limit,cells,padding>,+listed: {List.length(&2,Kernel.Back,groups) == U32.to_nat(cells) : Nat},tree: Model.Shaped<Each.PiecePair<Tile.Tile8>>,facts: Model.ShapedAll(~Each.PiecePair<Tile.Tile8>,~BackScope<DL,XL>,~Each.PairAt(~Tile.Tile8,~BackScope<DL,XL>,~dy_part(~DL,~dv,~XL),~xhat_part(~DL,~XL,~xv)),BackScope{dl,xl,U32.to_nat(limit)},tree)) -> Facts(~PaddedPieces.Padded<Pass<DL,XL,Kernel.Back>>,~PaddedPieces.padded_value(~Pass<DL,XL,Kernel.Back>,~passed(~Kernel.Back,~Kernel.Sampled,~back_of,~Kernel.back_combine,~DL,~dv,~XL,~xv)),PaddedPieces.Padded{padding,Pass{dl,xl,groups,cells,fallback}},U32.to_nat(limit),Model.shaped_content(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~BNCells.Mapping<Kernel.Back>,~back_content,BNCells.Mapping{groups,cells,padding},tree)):
  match tree:
    case Model.Piece{+pair,+first,+rows}:
      Each.PiecePair{+dy,+xhat} = pair
      padded_full(~Pass<DL,XL,Kernel.Back>,~passed(~Kernel.Back,~Kernel.Sampled,~back_of,~Kernel.back_combine,~DL,~dv,~XL,~xv),Pass{dl,xl,groups,cells,fallback},cells,padding,limit,rooms,first,rows,Buffer.Content{groups_tree(~Kernel.Back,~Kernel.Sampled,~sample_at(~Kernel.Back,~Kernel.Sampled,~back_of),~Kernel.back_combine,groups,first,rows,cells,0,Buffer.tree(~Tile.Tile8,xhat),Buffer.tree(~Tile.Tile8,dy)),Buffer.length(~Tile.Tile8,dy)},back_piece(~DL,~dv,~XL,~xv,groups,cells,fallback,dl,xl,padding,limit,rooms,listed,first,rows,dy,xhat,facts))
    case Model.Pieces{+left,+right}:
      KernelGemm.Both{left_facts,right_facts} = facts
      facts_joined(~PaddedPieces.Padded<Pass<DL,XL,Kernel.Back>>,~PaddedPieces.padded_value(~Pass<DL,XL,Kernel.Back>,~passed(~Kernel.Back,~Kernel.Sampled,~back_of,~Kernel.back_combine,~DL,~dv,~XL,~xv)),PaddedPieces.Padded{padding,Pass{dl,xl,groups,cells,fallback}},U32.to_nat(limit),Model.shaped_content(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~BNCells.Mapping<Kernel.Back>,~back_content,BNCells.Mapping{groups,cells,padding},left),Model.shaped_content(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~BNCells.Mapping<Kernel.Back>,~back_content,BNCells.Mapping{groups,cells,padding},right),
        back_facts(~DL,~dv,~XL,~xv,groups,cells,fallback,dl,xl,padding,limit,rooms,listed,left,left_facts),back_facts(~DL,~dv,~XL,~xv,groups,cells,fallback,dl,xl,padding,limit,rooms,listed,right,right_facts))

# The backward pass leaves pieces of the layout of its pairs that hold what
# the parameters of every group make of dy and xhat.
def backward(~DL: Data,~dv: DL -> U32 -> Tile.Tile8,~XL: Data,~xv: XL -> U32 -> Tile.Tile8,+groups: List<&2,Kernel.Back>,+cells: U32,+fallback: Kernel.Back,+dl: DL,+xl: XL,+padding: TiledMap.Padding,+limit: U32,+rooms: Rooms<limit,cells,padding>,+listed: {List.length(&2,Kernel.Back,groups) == U32.to_nat(cells) : Nat},+shape: List<&2,U32>,+workers: U32,+tree: Model.Shaped<Each.PiecePair<Tile.Tile8>>,facts: Model.ShapedAll(~Each.PiecePair<Tile.Tile8>,~BackScope<DL,XL>,~Each.PairAt(~Tile.Tile8,~BackScope<DL,XL>,~dy_part(~DL,~dv,~XL),~xhat_part(~DL,~XL,~xv)),BackScope{dl,xl,U32.to_nat(limit)},tree),coverage: PiecesPlan.Coverage(Model.shaped_plan(~Each.PiecePair<Tile.Tile8>,~Each.piece_rows(~Each.PiecePair<Tile.Tile8>),tree),0n,U32.to_nat(limit))) ->
  Passed(~PaddedPieces.Padded<Pass<DL,XL,Kernel.Back>>,~PaddedPieces.padded_value(~Pass<DL,XL,Kernel.Back>,~passed(~Kernel.Back,~Kernel.Sampled,~back_of,~Kernel.back_combine,~DL,~dv,~XL,~xv)),PaddedPieces.Padded{padding,Pass{dl,xl,groups,cells,fallback}},U32.to_nat(limit),Model.contents(~Tile.Tile8,BNCells.backward(groups,cells,padding,shape,workers,Model.shaped_pack(~Each.PiecePair<Tile.Tile8>,~BNCells.CellPair(),~Each.pair_pack(~Tile.Tile8),tree)))):
  %Equal.sym(Model.Content<Tile.Tile8>,Model.contents(~Tile.Tile8,BNCells.backward(groups,cells,padding,shape,workers,Model.shaped_pack(~Each.PiecePair<Tile.Tile8>,~BNCells.CellPair(),~Each.pair_pack(~Tile.Tile8),tree))),Model.shaped_content(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~BNCells.Mapping<Kernel.Back>,~back_content,BNCells.Mapping{groups,cells,padding},tree),
      backward_contents(groups,cells,padding,shape,workers,tree)) :
    Passed(~PaddedPieces.Padded<Pass<DL,XL,Kernel.Back>>,~PaddedPieces.padded_value(~Pass<DL,XL,Kernel.Back>,~passed(~Kernel.Back,~Kernel.Sampled,~back_of,~Kernel.back_combine,~DL,~dv,~XL,~xv)),PaddedPieces.Padded{padding,Pass{dl,xl,groups,cells,fallback}},U32.to_nat(limit),_)
  passed_of(~PaddedPieces.Padded<Pass<DL,XL,Kernel.Back>>,~PaddedPieces.padded_value(~Pass<DL,XL,Kernel.Back>,~passed(~Kernel.Back,~Kernel.Sampled,~back_of,~Kernel.back_combine,~DL,~dv,~XL,~xv)),PaddedPieces.Padded{padding,Pass{dl,xl,groups,cells,fallback}},U32.to_nat(limit),Model.shaped_content(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~BNCells.Mapping<Kernel.Back>,~back_content,BNCells.Mapping{groups,cells,padding},tree),back_facts(~DL,~dv,~XL,~xv,groups,cells,fallback,dl,xl,padding,limit,rooms,listed,tree,facts),
    %Equal.sym(Pieces.Ranges,Model.content_layout(~Tile.Tile8,Model.shaped_content(~Each.PiecePair<Tile.Tile8>,~Tile.Tile8,~BNCells.Mapping<Kernel.Back>,~back_content,BNCells.Mapping{groups,cells,padding},tree)),Model.shaped_plan(~Each.PiecePair<Tile.Tile8>,~Each.piece_rows(~Each.PiecePair<Tile.Tile8>),tree),back_layout(BNCells.Mapping{groups,cells,padding},tree)) :
      PiecesPlan.Coverage(_,0n,U32.to_nat(limit))
    coverage)
