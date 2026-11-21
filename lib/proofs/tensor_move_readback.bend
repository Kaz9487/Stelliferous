# The pieces a task holds and what reading them gives: a holding of buffer
# models packs to the runtime holding, every runtime step on a packed holding
# is the packed step of the model, and on pieces that tile a range of cells
# with the values of a tensor a read gives the value of its cell.
import Base
import ../storage_buffer.bend as Storage
import ../tensor_move.bend as Move
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./nat_order.bend as Order
import ./word_encoding.bend as WordEncoding
import ./u32_comparison.bend as Comparison
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./tensor_pieces_content.bend as Content
import ../kernel_gemm.bend as KernelGemm
import ./natural_addition.bend as Addition
import ./boolean_logic.bend as Logic
import ../storage_copy.bend as Copy
import ../tensor_view.bend as Views
import ./storage_copy_refinement.bend as CopyLaws
import ./contiguous_copy.bend as Contiguous
import ./write_program.bend as Writes
import ./tensor_copy_values.bend as CopyValues
import ./storage_shape.bend as Shape
import ./storage_readback.bend as StorageReadback
import ./u32_extrema.bend as Extrema
import ./nat_extrema.bend as NatExtrema
import ./u32_division.bend as Division
import ./contiguous_copy_readback.bend as ContiguousReadback
import ./write_program_readback.bend as Readback
import ./nat_interval.bend as Interval
import ../storage_allocation.bend as Allocation
import ./storage_allocation_refinement.bend as AllocationLaws
import ../tensor_pieces.bend as Pieces
import ../traversal_partition.bend as Partition
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_pieces_plan.bend as PiecesPlan
import ./sequence_membership.bend as Members
import ./tensor_allocation_depth.bend as Depth
import ./word_power_of_two.bend as Power
import ./word_arithmetic.bend as WordArithmetic
import ./word_multiplication.bend as Multiplication
import ./tensor_cursor_model.bend as CursorModel
import ./tensor_cursor_order.bend as CursorOrder
import ./tensor_cursor_refinement.bend as CursorLaws
import ./tensor_visit_refinement.bend as TensorVisit
import ./consecutive_writes.bend as Consecutive
import ./nat_ceiling.bend as Ceiling
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ./nat_division.bend as NatDivision
import ./ordered_ranges.bend as Ranges

# ---- Models ----

type Stack<-Element: Data> is Data:
  Bottom{}
  Held{buffer: Buffer.Model<Element>,first: U32,count: U32,below: Stack<Element>}

type Holding<-Element: Data> is Data:
  Holding{behind: Stack<Element>,ahead: Stack<Element>}

def pack_stack(~Element: Data,stack: Stack<Element>) -> Move.Stack<Element>:
  match stack:
    case Bottom{}: Move.Bottom{}
    case Held{buffer,first,count,below}: Move.Held{Buffer.pack(~Element,buffer),first,count,pack_stack(~Element,below)}

def pack(~Element: Data,holding: Holding<Element>) -> Move.Holding<Element>:
  Holding{behind,ahead} = holding
  Move.Holding{pack_stack(~Element,behind),pack_stack(~Element,ahead)}

# ---- Resolve ----

def advanced(~Element: Data,below: Stack<Element>,step: Bool,+index: U32,buffer: Buffer.Model<Element>,+first: U32,+count: U32,behind: Stack<Element>) -> Holding<Element>:
  match below step:
    case Held{next,+next_first,+next_count,rest} True{}:
      advanced(~Element,rest,U32.is_le((next_first + next_count : U32),index),index,next,next_first,next_count,Held{buffer,first,count,behind})
    case Held{next,next_first,next_count,rest} False{}: Holding{behind,Held{buffer,first,count,Held{next,next_first,next_count,rest}}}
    case Bottom{} step: Holding{behind,Held{buffer,first,count,Bottom{}}}

def advanced_packed(~Element: Data,+below: Stack<Element>,+step: Bool,+index: U32,+buffer: Buffer.Model<Element>,+first: U32,+count: U32,+behind: Stack<Element>) ->
  {Move.advanced(~Element,pack_stack(~Element,below),step,index,Buffer.pack(~Element,buffer),first,count,pack_stack(~Element,behind)) ==
    pack(~Element,advanced(~Element,below,step,index,buffer,first,count,behind)) : Move.Holding<Element>}:
  match below step:
    case Held{+next,+next_first,+next_count,+rest} True{}:
      advanced_packed(~Element,rest,U32.is_le((next_first + next_count : U32),index),index,next,next_first,next_count,Held{buffer,first,count,behind})
    case Held{next,next_first,next_count,rest} False{}: {==}
    case Bottom{} True{}: {==}
    case Bottom{} False{}: {==}

def backed(~Element: Data,behind: Stack<Element>,step: Bool,+index: U32,ahead: Stack<Element>) -> Holding<Element>:
  match behind step:
    case Held{buffer,+first,+count,rest} True{}: backed(~Element,rest,U32.is_lt(index,first),index,Held{buffer,first,count,ahead})
    case Held{buffer,first,count,rest} False{}: Holding{Held{buffer,first,count,rest},ahead}
    case Bottom{} step: Holding{Bottom{},ahead}

def backed_packed(~Element: Data,+behind: Stack<Element>,+step: Bool,+index: U32,+ahead: Stack<Element>) ->
  {Move.backed(~Element,pack_stack(~Element,behind),step,index,pack_stack(~Element,ahead)) == pack(~Element,backed(~Element,behind,step,index,ahead)) : Move.Holding<Element>}:
  match behind step:
    case Held{+buffer,+first,+count,+rest} True{}: backed_packed(~Element,rest,U32.is_lt(index,first),index,Held{buffer,first,count,ahead})
    case Held{buffer,first,count,rest} False{}: {==}
    case Bottom{} True{}: {==}
    case Bottom{} False{}: {==}

def located(~Element: Data,before: Bool,+index: U32,behind: Stack<Element>,buffer: Buffer.Model<Element>,+first: U32,+count: U32,below: Stack<Element>) -> Holding<Element>:
  match before:
    case True{}: backed(~Element,behind,True{},index,Held{buffer,first,count,below})
    case False{}: advanced(~Element,below,U32.is_le((first + count : U32),index),index,buffer,first,count,behind)

def located_packed(~Element: Data,+before: Bool,+index: U32,+behind: Stack<Element>,+buffer: Buffer.Model<Element>,+first: U32,+count: U32,+below: Stack<Element>) ->
  {Move.located(~Element,before,index,pack_stack(~Element,behind),Buffer.pack(~Element,buffer),first,count,pack_stack(~Element,below)) ==
    pack(~Element,located(~Element,before,index,behind,buffer,first,count,below)) : Move.Holding<Element>}:
  match before:
    case True{}: backed_packed(~Element,behind,True{},index,Held{buffer,first,count,below})
    case False{}: advanced_packed(~Element,below,U32.is_le((first + count : U32),index),index,buffer,first,count,behind)

def resolve(~Element: Data,holding: Holding<Element>,+index: U32) -> Holding<Element>:
  match holding:
    case Holding{behind,Held{buffer,+first,+count,below}}: located(~Element,U32.is_lt(index,first),index,behind,buffer,first,count,below)
    case Holding{behind,Bottom{}}: backed(~Element,behind,True{},index,Bottom{})

def resolve_packed(~Element: Data,+holding: Holding<Element>,+index: U32) ->
  {Move.resolve(~Element,pack(~Element,holding),index) == pack(~Element,resolve(~Element,holding,index)) : Move.Holding<Element>}:
  match holding:
    case Holding{+behind,Held{+buffer,+first,+count,+below}}: located_packed(~Element,U32.is_lt(index,first),index,behind,buffer,first,count,below)
    case Holding{+behind,Bottom{}}: backed_packed(~Element,behind,True{},index,Bottom{})

# ---- Read ----

def Read(~Element: Data) -> Type:
  Holding<Element> & Element

def pack_read(~Element: Data,pair: Read(~Element)) -> Move.Holding<Element> & Element:
  (holding,value) = pair
  (pack(~Element,holding),value)

def read_held(~Element: Data,held: Bool,+index: U32,+fallback: Element,behind: Stack<Element>,+buffer: Buffer.Model<Element>,+first: U32,+span: U32,below: Stack<Element>) -> Read(~Element):
  match held:
    case True{}: (Holding{behind,Held{buffer,first,span,below}},Tree.read(Element,Buffer.tree(~Element,buffer),(index - first : U32)))
    case False{}: (Holding{behind,Held{buffer,first,span,below}},fallback)

def read_held_packed(~Element: Data,+held: Bool,+index: U32,+fallback: Element,+behind: Stack<Element>,+buffer: Buffer.Model<Element>,+first: U32,+span: U32,+below: Stack<Element>) ->
  {Move.read_held(~Element,held,index,fallback,pack_stack(~Element,behind),Buffer.pack(~Element,buffer),first,span,pack_stack(~Element,below)) ==
    pack_read(~Element,read_held(~Element,held,index,fallback,behind,buffer,first,span,below)) : Move.Holding<Element> & Element}:
  match held:
    case True{}:
      %Equal.sym(Storage.Buffer<Element> & Element,Storage.read(~Element,Buffer.pack(~Element,buffer),(index - first : U32)),
        (Buffer.pack(~Element,buffer),Tree.read(Element,Buffer.tree(~Element,buffer),(index - first : U32))),Buffer.read(~Element,buffer,(index - first : U32))) :
        {Move.held_read(~Element,pack_stack(~Element,behind),first,span,pack_stack(~Element,below),_) ==
          pack_read(~Element,read_held(~Element,True{},index,fallback,behind,buffer,first,span,below)) : Move.Holding<Element> & Element}
      {==}
    case False{}: {==}

def read_current(~Element: Data,holding: Holding<Element>,+index: U32,+fallback: Element) -> Read(~Element):
  match holding:
    case Holding{behind,Held{buffer,+first,+span,below}}:
      read_held(~Element,U32.is_le(first,index) && U32.is_lt(index,(first + span : U32)),index,fallback,behind,buffer,first,span,below)
    case Holding{behind,Bottom{}}: (Holding{behind,Bottom{}},fallback)

def read_current_packed(~Element: Data,+holding: Holding<Element>,+index: U32,+fallback: Element) ->
  {Move.read_current(~Element,pack(~Element,holding),index,fallback) == pack_read(~Element,read_current(~Element,holding,index,fallback)) : Move.Holding<Element> & Element}:
  match holding:
    case Holding{+behind,Held{+buffer,+first,+span,+below}}:
      read_held_packed(~Element,U32.is_le(first,index) && U32.is_lt(index,(first + span : U32)),index,fallback,behind,buffer,first,span,below)
    case Holding{behind,Bottom{}}: {==}

def read(~Element: Data,+holding: Holding<Element>,+index: U32,+fallback: Element) -> Read(~Element):
  read_current(~Element,resolve(~Element,holding,index),index,fallback)

# A read of a packed holding is the packed read of its model.
def read_packed(~Element: Data,+holding: Holding<Element>,+index: U32,+fallback: Element) ->
  {Move.read(~Element,pack(~Element,holding),index,fallback) == pack_read(~Element,read(~Element,holding,index,fallback)) : Move.Holding<Element> & Element}:
  %Equal.sym(Move.Holding<Element>,Move.resolve(~Element,pack(~Element,holding),index),pack(~Element,resolve(~Element,holding,index)),resolve_packed(~Element,holding,index)) :
    {Move.read_current(~Element,_,index,fallback) == pack_read(~Element,read(~Element,holding,index,fallback)) : Move.Holding<Element> & Element}
  read_current_packed(~Element,resolve(~Element,holding,index),index,fallback)

# ---- Order facts ----

def lt_false_le(+a: Nat,+b: Nat,h: {Nat.is_lt(a,b) == False{} : Bool}) -> {Nat.is_le(b,a) == True{} : Bool}:
  match a b:
    case 0n 0n: {==}
    case 0n 1n+b: Empty.absurd({Nat.is_le(1n+b,0n) == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},h)))
    case 1n+a 0n: {==}
    case 1n+a 1n+b: lt_false_le(a,b,h)

def le_false_lt(+a: Nat,+b: Nat,h: {Nat.is_le(a,b) == False{} : Bool}) -> {Nat.is_lt(b,a) == True{} : Bool}:
  match a b:
    case 0n 0n: Empty.absurd({Nat.is_lt(0n,0n) == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},h)))
    case 0n 1n+b: Empty.absurd({Nat.is_lt(1n+b,0n) == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},h)))
    case 1n+a 0n: {==}
    case 1n+a 1n+b: le_false_lt(a,b,h)

def lt_le_absurd(+a: Nat,+b: Nat,less: {Nat.is_lt(a,b) == True{} : Bool},order: {Nat.is_le(b,a) == True{} : Bool}) -> Empty:
  match a b:
    case 0n 0n: WordEncoding.false_true(less)
    case 0n 1n+b: WordEncoding.false_true(order)
    case 1n+a 0n: WordEncoding.false_true(less)
    case 1n+a 1n+b: lt_le_absurd(a,b,less,order)

# ---- Tiling ----

type Both<-First: Data,-Second: Data> is Data:
  Both{first: First,second: Second}

def first_of(-First: Data,-Second: Data,+both: Both<First,Second>) -> First:
  Both{first,second} = both
  first

def second_of(-First: Data,-Second: Data,+both: Both<First,Second>) -> Second:
  Both{first,second} = both
  second

# A piece that starts at low, its buffer ready and as long as its count.
type Starts<-Element: Data,-low: Nat,-first: U32,-count: U32,-buffer: Buffer.Model<Element>> is Data:
  Starts{start: {U32.to_nat(first) == low : Nat},sized: {Buffer.length(~Element,buffer) == U32.to_nat(count) : Nat},ready: Buffer.ContentReady<Element,Buffer.content(~Element,buffer)>}

# A piece that ends at high, its buffer ready and as long as its count.
type Ends<-Element: Data,-high: Nat,-first: U32,-count: U32,-buffer: Buffer.Model<Element>> is Data:
  Ends{end: {Nat.add(U32.to_nat(first),U32.to_nat(count)) == high : Nat},sized: {Buffer.length(~Element,buffer) == U32.to_nat(count) : Nat},ready: Buffer.ContentReady<Element,Buffer.content(~Element,buffer)>}

# The pieces of a stack, in order, cover [low, high) one after another.
def Tiles(~Element: Data,stack: Stack<Element>,+low: Nat,+high: Nat) -> Data:
  match stack:
    case Bottom{}: {low == high : Nat}
    case Held{buffer,+first,+count,below}: Both<Starts<Element,low,first,count,buffer>,Tiles(~Element,below,Nat.add(low,U32.to_nat(count)),high)>

# The pieces behind, nearest first, cover [low, mid) one before another.
def Behind(~Element: Data,stack: Stack<Element>,+low: Nat,+mid: Nat) -> Data:
  match stack:
    case Bottom{}: {low == mid : Nat}
    case Held{buffer,+first,+count,below}: Both<Ends<Element,mid,first,count,buffer>,Behind(~Element,below,low,U32.to_nat(first))>

# The pieces of a holding cover [low, high): those behind up to mid, the
# current one and those after it from mid on.
type Tiling<-Element: Data,-behind: Stack<Element>,-ahead: Stack<Element>,-low: Nat,-high: Nat> is Data:
  Tiling{mid: Nat,before: Behind(~Element,behind,low,mid),after: Tiles(~Element,ahead,mid,high)}

def Tiled(~Element: Data,holding: Holding<Element>,+low: Nat,+high: Nat) -> Data:
  Holding{behind,ahead} = holding
  Tiling<Element,behind,ahead,low,high>

def tiles_ordered(~Element: Data,+stack: Stack<Element>,+low: Nat,+high: Nat,+tiles: Tiles(~Element,stack,low,high)) -> {Nat.is_le(low,high) == True{} : Bool}:
  match stack:
    case Bottom{}:
      %tiles : {Nat.is_le(low,_) == True{} : Bool}
      Order.reflexive(low)
    case Held{buffer,+first,+count,+below}:
      Both{starts,+rest} = tiles
      Order.transitive(low,Nat.add(low,U32.to_nat(count)),high,Order.le_add(low,U32.to_nat(count)),tiles_ordered(~Element,below,Nat.add(low,U32.to_nat(count)),high,rest))

# The current piece holds cell index and ends within the capacity.
type Holds<-first: U32,-count: U32,-index: U32,-capacity: U32> is Data:
  Holds{after: {U32.is_le(first,index) == True{} : Bool},before: {U32.is_lt(index,(first + count : U32)) == True{} : Bool},
    room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(capacity)) == True{} : Bool}}

def Current(~Element: Data,holding: Holding<Element>,+index: U32,+capacity: U32) -> Data:
  Holding{behind,ahead} = holding
  match ahead:
    case Held{buffer,+first,+count,below}: Holds<first,count,index,capacity>
    case Bottom{}: Empty

def u32_le(+left: U32,+right: U32,+order: {Nat.is_le(U32.to_nat(left),U32.to_nat(right)) == True{} : Bool}) -> {U32.is_le(left,right) == True{} : Bool}:
  Equal.trans(Bool,U32.is_le(left,right),Nat.is_le(U32.to_nat(left),U32.to_nat(right)),True{},Comparison.le(left,right),order)

def u32_lt(+left: U32,+right: U32,+less: {Nat.is_lt(U32.to_nat(left),U32.to_nat(right)) == True{} : Bool}) -> {U32.is_lt(left,right) == True{} : Bool}:
  Equal.trans(Bool,U32.is_lt(left,right),Nat.is_lt(U32.to_nat(left),U32.to_nat(right)),True{},Comparison.lt(left,right),less)

def nat_le_false(+left: U32,+right: U32,+order: {U32.is_le(left,right) == False{} : Bool}) -> {Nat.is_le(U32.to_nat(left),U32.to_nat(right)) == False{} : Bool}:
  Equal.trans(Bool,Nat.is_le(U32.to_nat(left),U32.to_nat(right)),U32.is_le(left,right),False{},Equal.sym(Bool,U32.is_le(left,right),Nat.is_le(U32.to_nat(left),U32.to_nat(right)),Comparison.le(left,right)),order)

def nat_lt_false(+left: U32,+right: U32,+less: {U32.is_lt(left,right) == False{} : Bool}) -> {Nat.is_lt(U32.to_nat(left),U32.to_nat(right)) == False{} : Bool}:
  Equal.trans(Bool,Nat.is_lt(U32.to_nat(left),U32.to_nat(right)),U32.is_lt(left,right),False{},Equal.sym(Bool,U32.is_lt(left,right),Nat.is_lt(U32.to_nat(left),U32.to_nat(right)),Comparison.lt(left,right)),less)

# A piece holds index when it starts at or before it and does not end at or before it.
def holds_of(+first: U32,+count: U32,+index: U32,+capacity: U32,+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(capacity)) == True{} : Bool},
  +after: {Nat.is_le(U32.to_nat(first),U32.to_nat(index)) == True{} : Bool},+not_end: {U32.is_le((first + count : U32),index) == False{} : Bool}) -> Holds<first,count,index,capacity>:
  Holds{u32_le(first,index,after),u32_lt(index,(first + count : U32),le_false_lt(U32.to_nat((first + count : U32)),U32.to_nat(index),nat_le_false((first + count : U32),index,not_end))),room}

def tiles_moved(~Element: Data,+stack: Stack<Element>,+from: Nat,+to: Nat,+high: Nat,+equal: {from == to : Nat},+tiles: Tiles(~Element,stack,from,high)) -> Tiles(~Element,stack,to,high):
  %equal : Tiles(~Element,stack,_,high)
  tiles

def nat_le_moved(+from: Nat,+to: Nat,+bound: Nat,+equal: {from == to : Nat},+order: {Nat.is_le(to,bound) == True{} : Bool}) -> {Nat.is_le(from,bound) == True{} : Bool}:
  Equal.trans(Bool,Nat.is_le(from,bound),Nat.is_le(to,bound),True{},Equal.cong(Nat,Bool,value => Nat.is_le(value,bound),from,to,equal),order)

def nat_lt_moved(+value: Nat,+from: Nat,+to: Nat,+equal: {from == to : Nat},+less: {Nat.is_lt(value,to) == True{} : Bool}) -> {Nat.is_lt(value,from) == True{} : Bool}:
  Equal.trans(Bool,Nat.is_lt(value,from),Nat.is_lt(value,to),True{},Equal.cong(Nat,Bool,bound => Nat.is_lt(value,bound),from,to,equal),less)

# ---- Resolving keeps the cover ----

def behind_moved(~Element: Data,+stack: Stack<Element>,+low: Nat,+from: Nat,+to: Nat,+equal: {from == to : Nat},+behind: Behind(~Element,stack,low,from)) -> Behind(~Element,stack,low,to):
  %equal : Behind(~Element,stack,low,_)
  behind

def advanced_tiled(~Element: Data,+below: Stack<Element>,+step: Bool,+index: U32,+buffer: Buffer.Model<Element>,+first: U32,+count: U32,+behind: Stack<Element>,+low: Nat,+high: Nat,
  +before: Behind(~Element,behind,low,U32.to_nat(first)),+sized: {Buffer.length(~Element,buffer) == U32.to_nat(count) : Nat},+ready: Buffer.ContentReady<Element,Buffer.content(~Element,buffer)>,
  +after: Tiles(~Element,below,Nat.add(U32.to_nat(first),U32.to_nat(count)),high)) ->
  Tiled(~Element,advanced(~Element,below,step,index,buffer,first,count,behind),low,high):
  match below step:
    case Held{+next,+next_first,+next_count,+rest} True{}:
      Both{Starts{+start,+next_sized,+next_ready},+later} = after
      advanced_tiled(~Element,rest,U32.is_le((next_first + next_count : U32),index),index,next,next_first,next_count,Held{buffer,first,count,behind},low,high,
        Both{Ends{Equal.sym(Nat,U32.to_nat(next_first),Nat.add(U32.to_nat(first),U32.to_nat(count)),start),sized,ready},before},next_sized,next_ready,
        tiles_moved(~Element,rest,Nat.add(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(next_count)),Nat.add(U32.to_nat(next_first),U32.to_nat(next_count)),high,
          Equal.cong(Nat,Nat,value => Nat.add(value,U32.to_nat(next_count)),Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(next_first),
            Equal.sym(Nat,U32.to_nat(next_first),Nat.add(U32.to_nat(first),U32.to_nat(count)),start)),later))
    case Held{next,next_first,next_count,rest} False{}: Tiling{U32.to_nat(first),before,Both{Starts{{==},sized,ready},after}}
    case Bottom{} True{}: Tiling{U32.to_nat(first),before,Both{Starts{{==},sized,ready},after}}
    case Bottom{} False{}: Tiling{U32.to_nat(first),before,Both{Starts{{==},sized,ready},after}}

def backed_tiled(~Element: Data,+behind: Stack<Element>,+step: Bool,+index: U32,+ahead: Stack<Element>,+low: Nat,+mid: Nat,+high: Nat,
  +before: Behind(~Element,behind,low,mid),+after: Tiles(~Element,ahead,mid,high)) -> Tiled(~Element,backed(~Element,behind,step,index,ahead),low,high):
  match behind step:
    case Held{+buffer,+first,+count,+rest} True{}:
      Both{Ends{+end,+sized,+ready},+earlier} = before
      backed_tiled(~Element,rest,U32.is_lt(index,first),index,Held{buffer,first,count,ahead},low,U32.to_nat(first),high,earlier,
        Both{Starts{{==},sized,ready},tiles_moved(~Element,ahead,mid,Nat.add(U32.to_nat(first),U32.to_nat(count)),high,Equal.sym(Nat,Nat.add(U32.to_nat(first),U32.to_nat(count)),mid,end),after)})
    case Held{buffer,first,count,rest} False{}: Tiling{mid,before,after}
    case Bottom{} True{}: Tiling{mid,before,after}
    case Bottom{} False{}: Tiling{mid,before,after}

def located_tiled(~Element: Data,+before: Bool,+index: U32,+behind: Stack<Element>,+buffer: Buffer.Model<Element>,+first: U32,+count: U32,+below: Stack<Element>,
  +low: Nat,+mid: Nat,+high: Nat,+earlier: Behind(~Element,behind,low,mid),+start: {U32.to_nat(first) == mid : Nat},+sized: {Buffer.length(~Element,buffer) == U32.to_nat(count) : Nat},+ready: Buffer.ContentReady<Element,Buffer.content(~Element,buffer)>,
  +later: Tiles(~Element,below,Nat.add(mid,U32.to_nat(count)),high)) ->
  Tiled(~Element,located(~Element,before,index,behind,buffer,first,count,below),low,high):
  match before:
    case True{}: backed_tiled(~Element,behind,True{},index,Held{buffer,first,count,below},low,mid,high,earlier,Both{Starts{start,sized,ready},later})
    case False{}:
      advanced_tiled(~Element,below,U32.is_le((first + count : U32),index),index,buffer,first,count,behind,low,high,
        behind_moved(~Element,behind,low,mid,U32.to_nat(first),Equal.sym(Nat,U32.to_nat(first),mid,start),earlier),sized,ready,
        tiles_moved(~Element,below,Nat.add(mid,U32.to_nat(count)),Nat.add(U32.to_nat(first),U32.to_nat(count)),high,
          Equal.cong(Nat,Nat,value => Nat.add(value,U32.to_nat(count)),mid,U32.to_nat(first),Equal.sym(Nat,U32.to_nat(first),mid,start)),later))

# Resolving a cell keeps the pieces of a holding and the range they cover.
def resolve_tiled(~Element: Data,+holding: Holding<Element>,+index: U32,+low: Nat,+high: Nat,+tiled: Tiled(~Element,holding,low,high)) ->
  Tiled(~Element,resolve(~Element,holding,index),low,high):
  match holding:
    case Holding{+behind,Held{+buffer,+first,+count,+below}}:
      Tiling{+mid,+earlier,Both{Starts{+start,+sized,+ready},+later}} = tiled
      located_tiled(~Element,U32.is_lt(index,first),index,behind,buffer,first,count,below,low,mid,high,earlier,start,sized,ready,later)
    case Holding{+behind,Bottom{}}:
      Tiling{+mid,+earlier,+top} = tiled
      backed_tiled(~Element,behind,True{},index,Bottom{},low,mid,high,earlier,top)

# ---- Values ----

# Every held piece holds the values of its cells.
def StackValues(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,stack: Stack<Element>) -> Data:
  match stack:
    case Bottom{}: Unit
    case Held{+buffer,+first,+count,below}:
      Both<Content.Leaf(~Element,~Context,~value,context,first,count,Buffer.tree(~Element,buffer),Buffer.length(~Element,buffer)),StackValues(~Element,~Context,~value,context,below)>

def HoldingValues(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,holding: Holding<Element>) -> Data:
  Holding{behind,ahead} = holding
  Both<StackValues(~Element,~Context,~value,context,behind),StackValues(~Element,~Context,~value,context,ahead)>

# A reader through a projection: a holder reads the values of the source it
# projects to.
def projected(~Element: Data,~Source: Data,~value: Source -> U32 -> Element,~Holder: Data,~project: Holder -> Source,holder: Holder,index: U32) -> Element:
  value(project(holder),index)

def stack_projected(~Element: Data,~Source: Data,~value: Source -> U32 -> Element,~Holder: Data,~project: Holder -> Source,+holder: Holder,stack: Stack<Element>,values: StackValues(~Element,~Source,~value,project(holder),stack)) ->
  StackValues(~Element,~Holder,~projected(~Element,~Source,~value,~Holder,~project),holder,stack):
  match stack:
    case Bottom{}: values
    case Held{+buffer,+first,+count,below}:
      Both{+leaf,+rest} = values
      Both{leaf,stack_projected(~Element,~Source,~value,~Holder,~project,holder,below,rest)}

def holding_projected(~Element: Data,~Source: Data,~value: Source -> U32 -> Element,~Holder: Data,~project: Holder -> Source,+holder: Holder,holding: Holding<Element>,values: HoldingValues(~Element,~Source,~value,project(holder),holding)) ->
  HoldingValues(~Element,~Holder,~projected(~Element,~Source,~value,~Holder,~project),holder,holding):
  Holding{behind,ahead} = holding
  Both{+behind_values,+ahead_values} = values
  Both{stack_projected(~Element,~Source,~value,~Holder,~project,holder,behind,behind_values),stack_projected(~Element,~Source,~value,~Holder,~project,holder,ahead,ahead_values)}

def advanced_values(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+below: Stack<Element>,+step: Bool,+index: U32,+buffer: Buffer.Model<Element>,
  +first: U32,+count: U32,+behind: Stack<Element>,
  +piece: Content.Leaf(~Element,~Context,~value,context,first,count,Buffer.tree(~Element,buffer),Buffer.length(~Element,buffer)),
  +later: StackValues(~Element,~Context,~value,context,below),+earlier: StackValues(~Element,~Context,~value,context,behind)) ->
  HoldingValues(~Element,~Context,~value,context,advanced(~Element,below,step,index,buffer,first,count,behind)):
  match below step:
    case Held{+next,+next_first,+next_count,+rest} True{}:
      Both{+head,+tail} = later
      advanced_values(~Element,~Context,~value,context,rest,U32.is_le((next_first + next_count : U32),index),index,next,next_first,next_count,Held{buffer,first,count,behind},
        head,tail,Both{piece,earlier})
    case Held{next,next_first,next_count,rest} False{}: Both{earlier,Both{piece,later}}
    case Bottom{} True{}: Both{earlier,Both{piece,later}}
    case Bottom{} False{}: Both{earlier,Both{piece,later}}

def backed_values(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+behind: Stack<Element>,+step: Bool,+index: U32,+ahead: Stack<Element>,
  +earlier: StackValues(~Element,~Context,~value,context,behind),+later: StackValues(~Element,~Context,~value,context,ahead)) ->
  HoldingValues(~Element,~Context,~value,context,backed(~Element,behind,step,index,ahead)):
  match behind step:
    case Held{+buffer,+first,+count,+rest} True{}:
      Both{+head,+tail} = earlier
      backed_values(~Element,~Context,~value,context,rest,U32.is_lt(index,first),index,Held{buffer,first,count,ahead},tail,Both{head,later})
    case Held{buffer,first,count,rest} False{}: Both{earlier,later}
    case Bottom{} True{}: Both{earlier,later}
    case Bottom{} False{}: Both{earlier,later}

def located_values(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+before: Bool,+index: U32,+behind: Stack<Element>,
  +buffer: Buffer.Model<Element>,+first: U32,+count: U32,+below: Stack<Element>,
  +piece: Content.Leaf(~Element,~Context,~value,context,first,count,Buffer.tree(~Element,buffer),Buffer.length(~Element,buffer)),
  +earlier: StackValues(~Element,~Context,~value,context,behind),+later: StackValues(~Element,~Context,~value,context,below)) ->
  HoldingValues(~Element,~Context,~value,context,located(~Element,before,index,behind,buffer,first,count,below)):
  match before:
    case True{}: backed_values(~Element,~Context,~value,context,behind,True{},index,Held{buffer,first,count,below},earlier,Both{piece,later})
    case False{}: advanced_values(~Element,~Context,~value,context,below,U32.is_le((first + count : U32),index),index,buffer,first,count,behind,piece,later,earlier)

def resolve_values(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+holding: Holding<Element>,+index: U32,
  +values: HoldingValues(~Element,~Context,~value,context,holding)) -> HoldingValues(~Element,~Context,~value,context,resolve(~Element,holding,index)):
  match holding:
    case Holding{+behind,Held{+buffer,+first,+count,+below}}:
      Both{+earlier,Both{+piece,+later}} = values
      located_values(~Element,~Context,~value,context,U32.is_lt(index,first),index,behind,buffer,first,count,below,piece,earlier,later)
    case Holding{+behind,Bottom{}}:
      Both{+earlier,+later} = values
      backed_values(~Element,~Context,~value,context,behind,True{},index,Bottom{},earlier,later)

def by_cases(-Goal: Data,+test: Bool,below: {test == True{} : Bool} -> Goal,beyond: {test == False{} : Bool} -> Goal) -> Goal:
  match test:
    case True{}: below({==})
    case False{}: beyond({==})

# A number less than another is not at least it.
def lt_not_le(+left: Nat,+right: Nat,+less: {Nat.is_lt(left,right) == True{} : Bool}) -> {Nat.is_le(right,left) == False{} : Bool}:
  by_cases({Nat.is_le(right,left) == False{} : Bool},Nat.is_le(right,left),
    +reached => Empty.absurd({Nat.is_le(right,left) == False{} : Bool},lt_le_absurd(left,right,less,reached)),
    +other => other)

# ---- Order of the pieces ----

# The ranges of stacked pieces in order.
type Spans is Data:
  Ended{}
  Spanned{first: U32,count: U32,rest: Spans}

def spans(~Element: Data,stack: Stack<Element>) -> Spans:
  match stack:
    case Bottom{}: Ended{}
    case Held{buffer,first,count,below}: Spanned{first,count,spans(~Element,below)}

# The pieces behind, nearest last, onto those ahead.
def onto(~Element: Data,behind: Stack<Element>,ahead: Stack<Element>) -> Stack<Element>:
  match behind:
    case Bottom{}: ahead
    case Held{buffer,first,count,below}: onto(~Element,below,Held{buffer,first,count,ahead})

# The held pieces in ascending order.
def order(~Element: Data,holding: Holding<Element>) -> Stack<Element>:
  Holding{behind,ahead} = holding
  onto(~Element,behind,ahead)

def advanced_order(~Element: Data,+below: Stack<Element>,+step: Bool,+index: U32,+buffer: Buffer.Model<Element>,+first: U32,+count: U32,+behind: Stack<Element>) ->
  {order(~Element,advanced(~Element,below,step,index,buffer,first,count,behind)) == onto(~Element,behind,Held{buffer,first,count,below}) : Stack<Element>}:
  match below step:
    case Held{+next,+next_first,+next_count,+rest} True{}:
      advanced_order(~Element,rest,U32.is_le((next_first + next_count : U32),index),index,next,next_first,next_count,Held{buffer,first,count,behind})
    case Held{next,next_first,next_count,rest} False{}: {==}
    case Bottom{} True{}: {==}
    case Bottom{} False{}: {==}

def backed_order(~Element: Data,+behind: Stack<Element>,+step: Bool,+index: U32,+ahead: Stack<Element>) ->
  {order(~Element,backed(~Element,behind,step,index,ahead)) == onto(~Element,behind,ahead) : Stack<Element>}:
  match behind step:
    case Held{+buffer,+first,+count,+rest} True{}: backed_order(~Element,rest,U32.is_lt(index,first),index,Held{buffer,first,count,ahead})
    case Held{buffer,first,count,rest} False{}: {==}
    case Bottom{} True{}: {==}
    case Bottom{} False{}: {==}

def located_order(~Element: Data,+before: Bool,+index: U32,+behind: Stack<Element>,+buffer: Buffer.Model<Element>,+first: U32,+count: U32,+below: Stack<Element>) ->
  {order(~Element,located(~Element,before,index,behind,buffer,first,count,below)) == onto(~Element,behind,Held{buffer,first,count,below}) : Stack<Element>}:
  match before:
    case True{}: backed_order(~Element,behind,True{},index,Held{buffer,first,count,below})
    case False{}: advanced_order(~Element,below,U32.is_le((first + count : U32),index),index,buffer,first,count,behind)

# Resolving a cell keeps the pieces in their order.
def resolve_order(~Element: Data,+holding: Holding<Element>,+index: U32) -> {order(~Element,resolve(~Element,holding,index)) == order(~Element,holding) : Stack<Element>}:
  match holding:
    case Holding{+behind,Held{+buffer,+first,+count,+below}}: located_order(~Element,U32.is_lt(index,first),index,behind,buffer,first,count,below)
    case Holding{+behind,Bottom{}}: backed_order(~Element,behind,True{},index,Bottom{})

def onto_spans(~Element: Data,+behind: Stack<Element>,+left: Stack<Element>,+right: Stack<Element>,+same: {spans(~Element,left) == spans(~Element,right) : Spans}) ->
  {spans(~Element,onto(~Element,behind,left)) == spans(~Element,onto(~Element,behind,right)) : Spans}:
  match behind:
    case Bottom{}: same
    case Held{+buffer,+first,+count,+below}:
      onto_spans(~Element,below,Held{buffer,first,count,left},Held{buffer,first,count,right},
        Equal.cong(Spans,Spans,rest => Spanned{first,count,rest},spans(~Element,left),spans(~Element,right),same))

# The held pieces in order cover the range of the holding.
def onto_tiles(~Element: Data,+behind: Stack<Element>,+ahead: Stack<Element>,+low: Nat,+mid: Nat,+high: Nat,+before: Behind(~Element,behind,low,mid),
  +after: Tiles(~Element,ahead,mid,high)) -> Tiles(~Element,onto(~Element,behind,ahead),low,high):
  match behind:
    case Bottom{}: tiles_moved(~Element,ahead,mid,low,high,Equal.sym(Nat,low,mid,before),after)
    case Held{+buffer,+first,+count,+below}:
      Both{Ends{+end,+sized,+ready},+earlier} = before
      onto_tiles(~Element,below,Held{buffer,first,count,ahead},low,U32.to_nat(first),high,earlier,
        Both{Starts{{==},sized,ready},tiles_moved(~Element,ahead,mid,Nat.add(U32.to_nat(first),U32.to_nat(count)),high,
          Equal.sym(Nat,Nat.add(U32.to_nat(first),U32.to_nat(count)),mid,end),after)})

def onto_values(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+behind: Stack<Element>,+ahead: Stack<Element>,
  +before: StackValues(~Element,~Context,~value,context,behind),+after: StackValues(~Element,~Context,~value,context,ahead)) ->
  StackValues(~Element,~Context,~value,context,onto(~Element,behind,ahead)):
  match behind:
    case Bottom{}: after
    case Held{+buffer,+first,+count,+below}:
      Both{+piece,+earlier} = before
      onto_values(~Element,~Context,~value,context,below,Held{buffer,first,count,ahead},earlier,Both{piece,after})

def onto_moved(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+buffer: Buffer.Model<Element>,+first: U32,+count: U32,
  +below: Stack<Element>,+ahead: Stack<Element>,
  +pair: Both<StackValues(~Element,~Context,~value,context,below),StackValues(~Element,~Context,~value,context,Held{buffer,first,count,ahead})>) ->
  Both<StackValues(~Element,~Context,~value,context,Held{buffer,first,count,below}),StackValues(~Element,~Context,~value,context,ahead)>:
  Both{+earlier,+moved} = pair
  Both{+piece,+after} = moved
  Both{Both{piece,earlier},after}

# The values of pieces stacked onto pieces ahead are the values of each.
def onto_split(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+behind: Stack<Element>,+ahead: Stack<Element>,
  +values: StackValues(~Element,~Context,~value,context,onto(~Element,behind,ahead))) ->
  Both<StackValues(~Element,~Context,~value,context,behind),StackValues(~Element,~Context,~value,context,ahead)>:
  match behind:
    case Bottom{}: Both{Unit{},values}
    case Held{+buffer,+first,+count,+below}:
      onto_moved(~Element,~Context,~value,context,buffer,first,count,below,ahead,
        onto_split(~Element,~Context,~value,context,below,Held{buffer,first,count,ahead},values))

# A holding whose pieces in order hold their values holds them.
def holding_values(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+holding: Holding<Element>,
  +values: StackValues(~Element,~Context,~value,context,order(~Element,holding))) -> HoldingValues(~Element,~Context,~value,context,holding):
  match holding:
    case Holding{+behind,+ahead}:
      onto_split(~Element,~Context,~value,context,behind,ahead,values)

def order_tiles(~Element: Data,+holding: Holding<Element>,+low: Nat,+high: Nat,+tiled: Tiled(~Element,holding,low,high)) -> Tiles(~Element,order(~Element,holding),low,high):
  match holding:
    case Holding{+behind,+ahead}:
      Tiling{+mid,+before,+after} = tiled
      onto_tiles(~Element,behind,ahead,low,mid,high,before,after)

# Pieces in order that tile a range split, from any current piece, into those
# behind up to its start and those ahead from it on.
def tiling_held(~Element: Data,+buffer: Buffer.Model<Element>,+first: U32,+count: U32,+below: Stack<Element>,+ahead: Stack<Element>,+low: Nat,+high: Nat,
  +inner: Tiling<Element,below,Held{buffer,first,count,ahead},low,high>) -> Tiling<Element,Held{buffer,first,count,below},ahead,low,high>:
  Tiling{+mid,+before,Both{Starts{+start,+sized,+ready},+after}} = inner
  +moved = Equal.sym(Nat,U32.to_nat(first),mid,start)
  Tiling{Nat.add(U32.to_nat(first),U32.to_nat(count)),
    Both{Ends{{==},sized,ready},behind_moved(~Element,below,low,mid,U32.to_nat(first),moved,before)},
    tiles_moved(~Element,ahead,Nat.add(mid,U32.to_nat(count)),Nat.add(U32.to_nat(first),U32.to_nat(count)),high,
      Equal.cong(Nat,Nat,value => Nat.add(value,U32.to_nat(count)),mid,U32.to_nat(first),moved),after)}

def tiling_split(~Element: Data,+behind: Stack<Element>,+ahead: Stack<Element>,+low: Nat,+high: Nat,+tiles: Tiles(~Element,onto(~Element,behind,ahead),low,high)) ->
  Tiling<Element,behind,ahead,low,high>:
  match behind:
    case Bottom{}: Tiling{low,{==},tiles}
    case Held{+buffer,+first,+count,+below}:
      tiling_held(~Element,buffer,first,count,below,ahead,low,high,tiling_split(~Element,below,Held{buffer,first,count,ahead},low,high,tiles))

# A holding whose pieces in order tile a range tiles it.
def order_tiled(~Element: Data,+holding: Holding<Element>,+low: Nat,+high: Nat,+tiles: Tiles(~Element,order(~Element,holding),low,high)) -> Tiled(~Element,holding,low,high):
  match holding:
    case Holding{+behind,+ahead}: tiling_split(~Element,behind,ahead,low,high,tiles)

def order_values(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+holding: Holding<Element>,
  +values: HoldingValues(~Element,~Context,~value,context,holding)) -> StackValues(~Element,~Context,~value,context,order(~Element,holding)):
  match holding:
    case Holding{+behind,+ahead}:
      Both{+before,+after} = values
      onto_values(~Element,~Context,~value,context,behind,ahead,before,after)

# ---- Pieces in order, with gaps ----

# A piece that starts at or after low, its buffer as long as its count.
type From<-low: Nat,-first: U32,-count: U32,-length: Nat> is Data:
  From{start: {Nat.is_le(low,U32.to_nat(first)) == True{} : Bool},sized: {length == U32.to_nat(count) : Nat}}

# A piece that ends at or before high, its buffer as long as its count.
type Upto<-high: Nat,-first: U32,-count: U32,-length: Nat> is Data:
  Upto{end: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),high) == True{} : Bool},sized: {length == U32.to_nat(count) : Nat}}

# The pieces of a stack lie in order within [low, high), each after the one
# before it.
def Sorted(~Element: Data,stack: Stack<Element>,+low: Nat,+high: Nat) -> Data:
  match stack:
    case Bottom{}: {Nat.is_le(low,high) == True{} : Bool}
    case Held{buffer,+first,+count,below}:
      Both<From<low,first,count,Buffer.length(~Element,buffer)>,Sorted(~Element,below,Nat.add(U32.to_nat(first),U32.to_nat(count)),high)>

# The pieces behind, nearest first, lie in order within [low, mid).
def SortedBehind(~Element: Data,stack: Stack<Element>,+low: Nat,+mid: Nat) -> Data:
  match stack:
    case Bottom{}: {Nat.is_le(low,mid) == True{} : Bool}
    case Held{buffer,+first,+count,below}:
      Both<Upto<mid,first,count,Buffer.length(~Element,buffer)>,SortedBehind(~Element,below,low,U32.to_nat(first))>

# The pieces behind lie in order up to mid and those ahead from mid on.
type Sorting<-Element: Data,-behind: Stack<Element>,-ahead: Stack<Element>,-low: Nat,-high: Nat> is Data:
  Sorting{mid: Nat,before: SortedBehind(~Element,behind,low,mid),after: Sorted(~Element,ahead,mid,high)}

# Cell index lies in the piece from first of count cells.
type Within<-first: U32,-count: U32,-index: Nat> is Data:
  Within{after: {Nat.is_le(U32.to_nat(first),index) == True{} : Bool},before: {Nat.is_lt(index,Nat.add(U32.to_nat(first),U32.to_nat(count))) == True{} : Bool}}

# Cell index lies in one of the ranges.
def SpanHeld(ranges: Spans,+index: Nat) -> Data:
  match ranges:
    case Ended{}: Empty
    case Spanned{+first,+count,rest}: Either<&2,&2,Within<first,count,index>,SpanHeld(rest,index)>

def sorted_ordered(~Element: Data,+stack: Stack<Element>,+low: Nat,+high: Nat,+sorted: Sorted(~Element,stack,low,high)) -> {Nat.is_le(low,high) == True{} : Bool}:
  match stack:
    case Bottom{}: sorted
    case Held{buffer,+first,+count,+below}:
      Both{From{+start,sized},+rest} = sorted
      Order.transitive(low,U32.to_nat(first),high,start,
        Order.transitive(U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),high,Order.le_add(U32.to_nat(first),U32.to_nat(count)),
          sorted_ordered(~Element,below,Nat.add(U32.to_nat(first),U32.to_nat(count)),high,rest)))

# A cell held by pieces in order from low lies at or after low.
def held_start(~Element: Data,+stack: Stack<Element>,+low: Nat,+high: Nat,+index: Nat,+sorted: Sorted(~Element,stack,low,high),
  +held: SpanHeld(spans(~Element,stack),index)) -> {Nat.is_le(low,index) == True{} : Bool}:
  match stack:
    case Bottom{}: Empty.absurd({Nat.is_le(low,index) == True{} : Bool},held)
    case Held{buffer,+first,+count,+below}:
      Both{From{+start,sized},+rest} = sorted
      match held:
        case Inl{Within{+after,before}}: Order.transitive(low,U32.to_nat(first),index,start,after)
        case Inr{+later}:
          Order.transitive(low,U32.to_nat(first),index,start,
            Order.transitive(U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),index,Order.le_add(U32.to_nat(first),U32.to_nat(count)),
              held_start(~Element,below,Nat.add(U32.to_nat(first),U32.to_nat(count)),high,index,rest,later)))

# A cell held by a piece or those in order after it lies at or after its start.
def held_first(~Element: Data,+buffer: Buffer.Model<Element>,+first: U32,+count: U32,+below: Stack<Element>,+high: Nat,+index: Nat,
  +sorted: Sorted(~Element,below,Nat.add(U32.to_nat(first),U32.to_nat(count)),high),+held: SpanHeld(spans(~Element,Held{buffer,first,count,below}),index)) ->
  {Nat.is_le(U32.to_nat(first),index) == True{} : Bool}:
  match held:
    case Inl{Within{+after,before}}: after
    case Inr{+later}:
      Order.transitive(U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),index,Order.le_add(U32.to_nat(first),U32.to_nat(count)),
        held_start(~Element,below,Nat.add(U32.to_nat(first),U32.to_nat(count)),high,index,sorted,later))

# No piece behind, each ending at or before mid, holds a cell at or after mid.
def behind_free(~Element: Data,+stack: Stack<Element>,+low: Nat,+mid: Nat,+index: Nat,+sorted: SortedBehind(~Element,stack,low,mid),
  +past: {Nat.is_le(mid,index) == True{} : Bool},+held: SpanHeld(spans(~Element,stack),index)) -> Empty:
  match stack:
    case Bottom{}: held
    case Held{buffer,+first,+count,+below}:
      Both{Upto{+end,sized},+rest} = sorted
      match held:
        case Inl{Within{after,+before}}:
          lt_le_absurd(index,Nat.add(U32.to_nat(first),U32.to_nat(count)),before,Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(count)),mid,index,end,past))
        case Inr{+later}:
          behind_free(~Element,below,low,U32.to_nat(first),index,rest,
            Order.transitive(U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),index,Order.le_add(U32.to_nat(first),U32.to_nat(count)),
              Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(count)),mid,index,end,past)),later)

def regrouped(-First: Data,-Second: Data,-Third: Data,+either: Either<&2,&2,Second,Either<&2,&2,First,Third>>) -> Either<&2,&2,Either<&2,&2,First,Second>,Third>:
  match either:
    case Inl{+second}: Inl{Inr{second}}
    case Inr{+inner}:
      match inner:
        case Inl{+first}: Inl{Inl{first}}
        case Inr{+third}: Inr{third}

# A cell held by the pieces of a holding in order is held behind or ahead.
def onto_held(~Element: Data,+behind: Stack<Element>,+ahead: Stack<Element>,+index: Nat,+held: SpanHeld(spans(~Element,onto(~Element,behind,ahead)),index)) ->
  Either<&2,&2,SpanHeld(spans(~Element,behind),index),SpanHeld(spans(~Element,ahead),index)>:
  match behind:
    case Bottom{}: Inr{held}
    case Held{+buffer,+first,+count,+below}:
      regrouped(Within<first,count,index>,SpanHeld(spans(~Element,below),index),SpanHeld(spans(~Element,ahead),index),
        onto_held(~Element,below,Held{buffer,first,count,ahead},index,held))

def behind_raise(~Element: Data,+stack: Stack<Element>,+low: Nat,+from: Nat,+to: Nat,+raised: {Nat.is_le(from,to) == True{} : Bool},
  +sorted: SortedBehind(~Element,stack,low,from)) -> SortedBehind(~Element,stack,low,to):
  match stack:
    case Bottom{}: Order.transitive(low,from,to,sorted,raised)
    case Held{buffer,+first,+count,+below}:
      Both{Upto{+end,+sized},+rest} = sorted
      Both{Upto{Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(count)),from,to,end,raised),sized},rest}

def sorting_held(~Element: Data,+buffer: Buffer.Model<Element>,+first: U32,+count: U32,+below: Stack<Element>,+ahead: Stack<Element>,+low: Nat,+high: Nat,
  +inner: Sorting<Element,below,Held{buffer,first,count,ahead},low,high>) -> Sorting<Element,Held{buffer,first,count,below},ahead,low,high>:
  Sorting{+mid,+before,Both{From{+start,+sized},+after}} = inner
  Sorting{Nat.add(U32.to_nat(first),U32.to_nat(count)),
    Both{Upto{Order.reflexive(Nat.add(U32.to_nat(first),U32.to_nat(count))),sized},behind_raise(~Element,below,low,mid,U32.to_nat(first),start,before)},after}

# Pieces in order, seen from any current piece: those behind end before those
# ahead start.
def sorted_split(~Element: Data,+behind: Stack<Element>,+ahead: Stack<Element>,+low: Nat,+high: Nat,+sorted: Sorted(~Element,onto(~Element,behind,ahead),low,high)) ->
  Sorting<Element,behind,ahead,low,high>:
  match behind:
    case Bottom{}: Sorting{low,Order.reflexive(low),sorted}
    case Held{+buffer,+first,+count,+below}:
      sorting_held(~Element,buffer,first,count,below,ahead,low,high,sorted_split(~Element,below,Held{buffer,first,count,ahead},low,high,sorted))

# The current piece holds a cell that lies in it.
def within_holds(+first: U32,+count: U32,+index: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(capacity)) == True{} : Bool},+within: Within<first,count,U32.to_nat(index)>) ->
  Holds<first,count,index,capacity>:
  Within{+after,+before} = within
  Holds{u32_le(first,index,after),
    u32_lt(index,(first + count : U32),nat_lt_moved(U32.to_nat(index),U32.to_nat((first + count : U32)),Nat.add(U32.to_nat(first),U32.to_nat(count)),
      Arithmetic.add(first,count,capacity,capacity_ok,room),before)),room}

def advanced_current(~Element: Data,+below: Stack<Element>,+step: Bool,+index: U32,+buffer: Buffer.Model<Element>,+first: U32,+count: U32,+behind: Stack<Element>,+high: Nat,
  +capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +after: {Nat.is_le(U32.to_nat(first),U32.to_nat(index)) == True{} : Bool},+sorted: Sorted(~Element,below,Nat.add(U32.to_nat(first),U32.to_nat(count)),high),
  +held: SpanHeld(spans(~Element,Held{buffer,first,count,below}),U32.to_nat(index)),+stepped: {step == U32.is_le((first + count : U32),index) : Bool}) ->
  Current(~Element,advanced(~Element,below,step,index,buffer,first,count,behind),index,capacity):
  match below step:
    case Held{+next,+next_first,+next_count,+rest} True{}:
      Both{next_from,+later_sorted} = sorted
      match held:
        case Inl{Within{after_here,+before_here}}:
          +finish = Nat.add(U32.to_nat(first),U32.to_nat(count))
          +room = Order.transitive(finish,high,U32.to_nat(capacity),sorted_ordered(~Element,Held{next,next_first,next_count,rest},finish,high,sorted),bounded)
          +reached = nat_le_moved(finish,U32.to_nat((first + count : U32)),U32.to_nat(index),Equal.sym(Nat,U32.to_nat((first + count : U32)),finish,Arithmetic.add(first,count,capacity,capacity_ok,room)),
            Bounds.as_nat_le((first + count : U32),index,Equal.sym(Bool,True{},U32.is_le((first + count : U32),index),stepped)))
          Empty.absurd(Current(~Element,advanced(~Element,Held{next,next_first,next_count,rest},True{},index,buffer,first,count,behind),index,capacity),
            lt_le_absurd(U32.to_nat(index),finish,before_here,reached))
        case Inr{+later}:
          advanced_current(~Element,rest,U32.is_le((next_first + next_count : U32),index),index,next,next_first,next_count,Held{buffer,first,count,behind},high,capacity,capacity_ok,bounded,
            held_first(~Element,next,next_first,next_count,rest,high,U32.to_nat(index),later_sorted,later),later_sorted,later,{==})
    case Held{next,next_first,next_count,rest} False{}:
      +finish = Nat.add(U32.to_nat(first),U32.to_nat(count))
      holds_of(first,count,index,capacity,Order.transitive(finish,high,U32.to_nat(capacity),sorted_ordered(~Element,Held{next,next_first,next_count,rest},finish,high,sorted),bounded),
        after,Equal.sym(Bool,False{},U32.is_le((first + count : U32),index),stepped))
    case Bottom{} True{}:
      match held:
        case Inl{Within{after_here,+before_here}}:
          +finish = Nat.add(U32.to_nat(first),U32.to_nat(count))
          +room = Order.transitive(finish,high,U32.to_nat(capacity),sorted,bounded)
          +reached = nat_le_moved(finish,U32.to_nat((first + count : U32)),U32.to_nat(index),Equal.sym(Nat,U32.to_nat((first + count : U32)),finish,Arithmetic.add(first,count,capacity,capacity_ok,room)),
            Bounds.as_nat_le((first + count : U32),index,Equal.sym(Bool,True{},U32.is_le((first + count : U32),index),stepped)))
          Empty.absurd(Current(~Element,advanced(~Element,Bottom{},True{},index,buffer,first,count,behind),index,capacity),lt_le_absurd(U32.to_nat(index),finish,before_here,reached))
        case Inr{+none}:
          Empty.absurd(Current(~Element,advanced(~Element,Bottom{},True{},index,buffer,first,count,behind),index,capacity),none)
    case Bottom{} False{}:
      holds_of(first,count,index,capacity,Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(count)),high,U32.to_nat(capacity),sorted,bounded),
        after,Equal.sym(Bool,False{},U32.is_le((first + count : U32),index),stepped))

# Backing up from the piece popped last, which ends at or before mid: it
# holds a cell held behind unless the cell lies before it.
def backed_current(~Element: Data,+rest: Stack<Element>,+step: Bool,+index: U32,+buffer: Buffer.Model<Element>,+first: U32,+count: U32,+ahead: Stack<Element>,+low: Nat,+mid: Nat,
  +capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(mid,U32.to_nat(capacity)) == True{} : Bool},
  +earlier: SortedBehind(~Element,rest,low,U32.to_nat(first)),+held: SpanHeld(spans(~Element,Held{buffer,first,count,rest}),U32.to_nat(index)),
  +end: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),mid) == True{} : Bool},+stepped: {step == U32.is_lt(index,first) : Bool}) ->
  Current(~Element,backed(~Element,rest,step,index,Held{buffer,first,count,ahead}),index,capacity):
  match rest step:
    case Held{+earlier_buffer,+earlier_first,+earlier_count,+below} True{}:
      Both{Upto{+earlier_end,earlier_sized},+further} = earlier
      match held:
        case Inl{Within{+after_here,before_here}}:
          +less = Bounds.as_nat_lt(index,first,Equal.sym(Bool,True{},U32.is_lt(index,first),stepped))
          Empty.absurd(Current(~Element,backed(~Element,Held{earlier_buffer,earlier_first,earlier_count,below},True{},index,Held{buffer,first,count,ahead}),index,capacity),
            lt_le_absurd(U32.to_nat(index),U32.to_nat(first),less,after_here))
        case Inr{+later}:
          backed_current(~Element,below,U32.is_lt(index,earlier_first),index,earlier_buffer,earlier_first,earlier_count,Held{buffer,first,count,ahead},low,U32.to_nat(first),
            capacity,capacity_ok,Order.transitive(U32.to_nat(first),mid,U32.to_nat(capacity),
              Order.transitive(U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),mid,Order.le_add(U32.to_nat(first),U32.to_nat(count)),end),bounded),
            further,later,earlier_end,{==})
    case Bottom{} True{}:
      match held:
        case Inl{Within{+after_here,before_here}}:
          +less = Bounds.as_nat_lt(index,first,Equal.sym(Bool,True{},U32.is_lt(index,first),stepped))
          Empty.absurd(Current(~Element,backed(~Element,Bottom{},True{},index,Held{buffer,first,count,ahead}),index,capacity),
            lt_le_absurd(U32.to_nat(index),U32.to_nat(first),less,after_here))
        case Inr{+none}:
          Empty.absurd(Current(~Element,backed(~Element,Bottom{},True{},index,Held{buffer,first,count,ahead}),index,capacity),none)
    case Held{earlier_buffer,earlier_first,earlier_count,below} False{}:
      match held:
        case Inl{+within}:
          +room = Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(count)),mid,U32.to_nat(capacity),end,bounded)
          within_holds(first,count,index,capacity,capacity_ok,room,within)
        case Inr{+later}:
          +reached = lt_false_le(U32.to_nat(index),U32.to_nat(first),nat_lt_false(index,first,Equal.sym(Bool,False{},U32.is_lt(index,first),stepped)))
          Empty.absurd(Holds<first,count,index,capacity>,
            behind_free(~Element,Held{earlier_buffer,earlier_first,earlier_count,below},low,U32.to_nat(first),U32.to_nat(index),earlier,reached,later))
    case Bottom{} False{}:
      match held:
        case Inl{+within}:
          +room = Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(count)),mid,U32.to_nat(capacity),end,bounded)
          within_holds(first,count,index,capacity,capacity_ok,room,within)
        case Inr{+none}:
          Empty.absurd(Holds<first,count,index,capacity>,none)

def backed_start(~Element: Data,+behind: Stack<Element>,+index: U32,+ahead: Stack<Element>,+low: Nat,+mid: Nat,
  +capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(mid,U32.to_nat(capacity)) == True{} : Bool},
  +before: SortedBehind(~Element,behind,low,mid),+held: SpanHeld(spans(~Element,behind),U32.to_nat(index))) ->
  Current(~Element,backed(~Element,behind,True{},index,ahead),index,capacity):
  match behind:
    case Bottom{}: Empty.absurd(Current(~Element,backed(~Element,Bottom{},True{},index,ahead),index,capacity),held)
    case Held{+buffer,+first,+count,+rest}:
      Both{Upto{+end,sized},+earlier} = before
      backed_current(~Element,rest,U32.is_lt(index,first),index,buffer,first,count,ahead,low,mid,capacity,capacity_ok,bounded,earlier,held,end,{==})

def located_current(~Element: Data,+before: Bool,+index: U32,+behind: Stack<Element>,+buffer: Buffer.Model<Element>,+first: U32,+count: U32,+below: Stack<Element>,
  +low: Nat,+mid: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +earlier: SortedBehind(~Element,behind,low,mid),+start: {Nat.is_le(mid,U32.to_nat(first)) == True{} : Bool},
  +later: Sorted(~Element,below,Nat.add(U32.to_nat(first),U32.to_nat(count)),high),
  +held: Either<&2,&2,SpanHeld(spans(~Element,behind),U32.to_nat(index)),SpanHeld(spans(~Element,Held{buffer,first,count,below}),U32.to_nat(index))>,
  +flag: {before == U32.is_lt(index,first) : Bool}) ->
  Current(~Element,located(~Element,before,index,behind,buffer,first,count,below),index,capacity):
  match before:
    case True{}:
      match held:
        case Inl{+behind_held}:
          backed_start(~Element,behind,index,Held{buffer,first,count,below},low,mid,capacity,capacity_ok,
            Order.transitive(mid,U32.to_nat(first),U32.to_nat(capacity),start,
              Order.transitive(U32.to_nat(first),high,U32.to_nat(capacity),
                Order.transitive(U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),high,Order.le_add(U32.to_nat(first),U32.to_nat(count)),
                  sorted_ordered(~Element,below,Nat.add(U32.to_nat(first),U32.to_nat(count)),high,later)),bounded)),earlier,behind_held)
        case Inr{+ahead_held}:
          +less = Bounds.as_nat_lt(index,first,Equal.sym(Bool,True{},U32.is_lt(index,first),flag))
          Empty.absurd(Current(~Element,located(~Element,True{},index,behind,buffer,first,count,below),index,capacity),
            lt_le_absurd(U32.to_nat(index),U32.to_nat(first),less,held_first(~Element,buffer,first,count,below,high,U32.to_nat(index),later,ahead_held)))
    case False{}:
      match held:
        case Inl{+behind_held}:
          +reached = lt_false_le(U32.to_nat(index),U32.to_nat(first),nat_lt_false(index,first,Equal.sym(Bool,False{},U32.is_lt(index,first),flag)))
          Empty.absurd(Current(~Element,located(~Element,False{},index,behind,buffer,first,count,below),index,capacity),
            behind_free(~Element,behind,low,mid,U32.to_nat(index),earlier,Order.transitive(mid,U32.to_nat(first),U32.to_nat(index),start,reached),behind_held))
        case Inr{+ahead_held}:
          +reached = lt_false_le(U32.to_nat(index),U32.to_nat(first),nat_lt_false(index,first,Equal.sym(Bool,False{},U32.is_lt(index,first),flag)))
          advanced_current(~Element,below,U32.is_le((first + count : U32),index),index,buffer,first,count,behind,high,capacity,capacity_ok,bounded,reached,later,ahead_held,{==})

def resolve_located(~Element: Data,+behind: Stack<Element>,+buffer: Buffer.Model<Element>,+first: U32,+count: U32,+below: Stack<Element>,+index: U32,+low: Nat,+high: Nat,
  +capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +sorting: Sorting<Element,behind,Held{buffer,first,count,below},low,high>,
  +held: Either<&2,&2,SpanHeld(spans(~Element,behind),U32.to_nat(index)),SpanHeld(spans(~Element,Held{buffer,first,count,below}),U32.to_nat(index))>) ->
  Current(~Element,resolve(~Element,Holding{behind,Held{buffer,first,count,below}},index),index,capacity):
  Sorting{+mid,+earlier,Both{From{+start,sized},+later}} = sorting
  located_current(~Element,U32.is_lt(index,first),index,behind,buffer,first,count,below,low,mid,high,capacity,capacity_ok,bounded,earlier,start,later,held,{==})

def resolve_bottom(~Element: Data,+behind: Stack<Element>,+index: U32,+low: Nat,+high: Nat,
  +capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +sorting: Sorting<Element,behind,Bottom{},low,high>,+held: Either<&2,&2,SpanHeld(spans(~Element,behind),U32.to_nat(index)),SpanHeld(spans(~Element,Bottom{}),U32.to_nat(index))>) ->
  Current(~Element,resolve(~Element,Holding{behind,Bottom{}},index),index,capacity):
  Sorting{+mid,+earlier,+top} = sorting
  match held:
    case Inl{+behind_held}:
      backed_start(~Element,behind,index,Bottom{},low,mid,capacity,capacity_ok,Order.transitive(mid,high,U32.to_nat(capacity),top,bounded),earlier,behind_held)
    case Inr{+none}: Empty.absurd(Current(~Element,resolve(~Element,Holding{behind,Bottom{}},index),index,capacity),none)

# On pieces in order, resolving a cell one of them holds makes that piece
# current.
def resolve_current(~Element: Data,+holding: Holding<Element>,+index: U32,+low: Nat,+high: Nat,
  +capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +sorted: Sorted(~Element,order(~Element,holding),low,high),+held: SpanHeld(spans(~Element,order(~Element,holding)),U32.to_nat(index))) ->
  Current(~Element,resolve(~Element,holding,index),index,capacity):
  match holding:
    case Holding{+behind,Held{+buffer,+first,+count,+below}}:
      resolve_located(~Element,behind,buffer,first,count,below,index,low,high,capacity,capacity_ok,bounded,
        sorted_split(~Element,behind,Held{buffer,first,count,below},low,high,sorted),onto_held(~Element,behind,Held{buffer,first,count,below},U32.to_nat(index),held))
    case Holding{+behind,Bottom{}}:
      resolve_bottom(~Element,behind,index,low,high,capacity,capacity_ok,bounded,
        sorted_split(~Element,behind,Bottom{},low,high,sorted),onto_held(~Element,behind,Bottom{},U32.to_nat(index),held))

# Pieces that tile [low, high) lie in order within it.
def tiles_sorted(~Element: Data,+stack: Stack<Element>,+low: Nat,+high: Nat,+tiles: Tiles(~Element,stack,low,high)) -> Sorted(~Element,stack,low,high):
  match stack:
    case Bottom{}:
      %tiles : {Nat.is_le(low,_) == True{} : Bool}
      Order.reflexive(low)
    case Held{buffer,+first,+count,+below}:
      Both{Starts{+start,+sized,ready},+rest} = tiles
      Both{From{%start : {Nat.is_le(_,U32.to_nat(first)) == True{} : Bool} Order.reflexive(U32.to_nat(first)),sized},
        tiles_sorted(~Element,below,Nat.add(U32.to_nat(first),U32.to_nat(count)),high,
          tiles_moved(~Element,below,Nat.add(low,U32.to_nat(count)),Nat.add(U32.to_nat(first),U32.to_nat(count)),high,
            Equal.cong(Nat,Nat,value => Nat.add(value,U32.to_nat(count)),low,U32.to_nat(first),Equal.sym(Nat,U32.to_nat(first),low,start)),rest))}

# Pieces that tile [low, high) hold every cell of it.
def tiles_member(~Element: Data,+stack: Stack<Element>,+low: Nat,+high: Nat,+index: Nat,+tiles: Tiles(~Element,stack,low,high),
  +after: {Nat.is_le(low,index) == True{} : Bool},+inside: {Nat.is_lt(index,high) == True{} : Bool}) -> SpanHeld(spans(~Element,stack),index):
  match stack:
    case Bottom{}:
      Empty.absurd(SpanHeld(spans(~Element,Bottom{}),index),lt_le_absurd(index,high,inside,nat_le_moved(high,low,index,Equal.sym(Nat,low,high,tiles),after)))
    case Held{buffer,+first,+count,+below}:
      Both{Starts{+start,sized,ready},+rest} = tiles
      by_cases(SpanHeld(spans(~Element,Held{buffer,first,count,below}),index),Nat.is_lt(index,Nat.add(U32.to_nat(first),U32.to_nat(count))),
        +below_end => Inl{Within{%Equal.sym(Nat,U32.to_nat(first),low,start) : {Nat.is_le(_,index) == True{} : Bool} after,below_end}},
        +beyond => Inr{tiles_member(~Element,below,Nat.add(low,U32.to_nat(count)),high,index,rest,
          %start : {Nat.is_le(Nat.add(_,U32.to_nat(count)),index) == True{} : Bool}
            lt_false_le(index,Nat.add(U32.to_nat(first),U32.to_nat(count)),beyond),inside)})

# On pieces that cover [low, high), resolving a cell of the range makes the
# piece that holds it current.
def resolve_holds(~Element: Data,+holding: Holding<Element>,+index: U32,+low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},+tiled: Tiled(~Element,holding,low,high),
  +after: {Nat.is_le(low,U32.to_nat(index)) == True{} : Bool},+inside: {Nat.is_lt(U32.to_nat(index),high) == True{} : Bool}) ->
  Current(~Element,resolve(~Element,holding,index),index,capacity):
  resolve_current(~Element,holding,index,low,high,capacity,capacity_ok,bounded,
    tiles_sorted(~Element,order(~Element,holding),low,high,order_tiles(~Element,holding,low,high,tiled)),
    tiles_member(~Element,order(~Element,holding),low,high,U32.to_nat(index),order_tiles(~Element,holding,low,high,tiled),after,inside))

def at_value(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+first: U32,+tree: Tree.Tree<Element>,+length: Nat,+offset: Nat,
  +at: Content.At(~Element,~Context,~value,context,first,tree,length,offset)) ->
  {Tree.read(Element,tree,U32.from_nat(offset)) == value(context,U32.from_nat(Nat.add(U32.to_nat(first),offset))) : Element}:
  KernelGemm.Both{size,read} = at
  read

# The cell index of a piece that holds it is the value of the cell.
def cell_value(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+first: U32,+count: U32,+index: U32,+tree: Tree.Tree<Element>,+length: Nat,
  +capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+holds: Holds<first,count,index,capacity>,
  +piece: Content.Leaf(~Element,~Context,~value,context,first,count,tree,length)) ->
  {Tree.read(Element,tree,(index - first : U32)) == value(context,index) : Element}:
  Holds{+after,+before,+room} = holds
  +ordered = Bounds.as_nat_le(first,index,after)
  +exact_end = Arithmetic.add(first,count,capacity,capacity_ok,room)
  +below_end = nat_lt_moved(U32.to_nat(index),Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat((first + count : U32)),Equal.sym(Nat,U32.to_nat((first + count : U32)),
    Nat.add(U32.to_nat(first),U32.to_nat(count)),exact_end),Bounds.as_nat_lt(index,(first + count : U32),before))
  +offset = Nat.sub(U32.to_nat(index),U32.to_nat(first))
  +exact_offset = Arithmetic.subtract(index,first,capacity,capacity_ok,ordered,Order.transitive(offset,U32.to_nat(index),U32.to_nat(capacity),Order.sub_le(U32.to_nat(index),U32.to_nat(first)),
    Order.transitive(U32.to_nat(index),Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(capacity),
      Bounds.strict_le(U32.to_nat(index),Nat.add(U32.to_nat(first),U32.to_nat(count)),below_end),room)))
  +read = at_value(~Element,~Context,~value,context,first,tree,length,offset,
    Content.leaf_cell(~Element,~Context,~value,context,first,count,tree,length,offset,piece,Order.sub_lt(U32.to_nat(index),U32.to_nat(first),U32.to_nat(count),ordered,below_end)))
  +located = Equal.trans(U32,(index - first : U32),U32.from_nat(U32.to_nat((index - first : U32))),U32.from_nat(offset),
    Equal.sym(U32,U32.from_nat(U32.to_nat((index - first : U32))),(index - first : U32),Arithmetic.native_roundtrip((index - first : U32))),
    Equal.cong(Nat,U32,U32.from_nat,U32.to_nat((index - first : U32)),offset,exact_offset))
  +cell = Equal.trans(U32,U32.from_nat(Nat.add(U32.to_nat(first),offset)),U32.from_nat(U32.to_nat(index)),index,
    Equal.cong(Nat,U32,U32.from_nat,Nat.add(U32.to_nat(first),offset),U32.to_nat(index),Order.restore_shift(U32.to_nat(first),U32.to_nat(index),ordered)),
    Arithmetic.native_roundtrip(index))
  Equal.trans(Element,Tree.read(Element,tree,(index - first : U32)),Tree.read(Element,tree,U32.from_nat(offset)),value(context,index),
    Equal.cong(U32,Element,position => Tree.read(Element,tree,position),(index - first : U32),U32.from_nat(offset),located),
    Equal.trans(Element,Tree.read(Element,tree,U32.from_nat(offset)),value(context,U32.from_nat(Nat.add(U32.to_nat(first),offset))),value(context,index),read,
      Equal.cong(U32,Element,position => value(context,position),U32.from_nat(Nat.add(U32.to_nat(first),offset)),index,cell)))

def read_current_value(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+holding: Holding<Element>,+index: U32,+fallback: Element,
  +capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+current: Current(~Element,holding,index,capacity),
  +values: HoldingValues(~Element,~Context,~value,context,holding)) ->
  {Pair.snd(Holding<Element>,Element,read_current(~Element,holding,index,fallback)) == value(context,index) : Element}:
  match holding:
    case Holding{+behind,Held{+buffer,+first,+count,+below}}:
      Holds{+after,+before,+room} = current
      Both{earlier,Both{+piece,later}} = values
      %Equal.sym(Bool,U32.is_le(first,index),True{},after) :
        {Pair.snd(Holding<Element>,Element,read_held(~Element,_ && U32.is_lt(index,(first + count : U32)),index,fallback,behind,buffer,first,count,below)) == value(context,index) : Element}
      %Equal.sym(Bool,U32.is_lt(index,(first + count : U32)),True{},before) :
        {Pair.snd(Holding<Element>,Element,read_held(~Element,True{} && _,index,fallback,behind,buffer,first,count,below)) == value(context,index) : Element}
      cell_value(~Element,~Context,~value,context,first,count,index,Buffer.tree(~Element,buffer),Buffer.length(~Element,buffer),capacity,capacity_ok,Holds{after,before,room},piece)
    case Holding{behind,Bottom{}}:
      Empty.absurd({Pair.snd(Holding<Element>,Element,read_current(~Element,Holding{behind,Bottom{}},index,fallback)) == value(context,index) : Element},current)

def nat_le_right_moved(+value: Nat,+from: Nat,+to: Nat,+equal: {from == to : Nat},+order: {Nat.is_le(value,to) == True{} : Bool}) -> {Nat.is_le(value,from) == True{} : Bool}:
  Equal.trans(Bool,Nat.is_le(value,from),Nat.is_le(value,to),True{},Equal.cong(Nat,Bool,bound => Nat.is_le(value,bound),from,to,equal),order)

# A cell j of [first, first + count) lies in a piece from start of span cells
# that holds the range: first - start + j is below span.
def offset_inside(+first: Nat,+start: Nat,+count: Nat,+span: Nat,+index: Nat,+ordered: {Nat.is_le(start,first) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(first,count),Nat.add(start,span)) == True{} : Bool},+less: {Nat.is_lt(index,count) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(Nat.sub(first,start),index),span) == True{} : Bool}:
  +offset = Nat.sub(first,start)
  +shifted = Equal.trans(Nat,Nat.add(start,Nat.add(offset,count)),Nat.add(Nat.add(start,offset),count),Nat.add(first,count),
    Equal.sym(Nat,Nat.add(Nat.add(start,offset),count),Nat.add(start,Nat.add(offset,count)),Addition.associative(start,offset,count)),
    Equal.cong(Nat,Nat,value => Nat.add(value,count),Nat.add(start,offset),first,Order.restore_shift(start,first,ordered)))
  +room = Order.cancel_offset_le(start,Nat.add(offset,count),span,nat_le_moved(Nat.add(start,Nat.add(offset,count)),Nat.add(first,count),Nat.add(start,span),shifted,fits))
  Bounds.nat_trans(Nat.add(offset,index),Nat.add(offset,count),span,
    Equal.trans(Bool,Nat.is_lt(Nat.add(offset,index),Nat.add(offset,count)),Nat.is_lt(index,count),True{},Order.lt_offset(offset,index,count),less),room)

# Cell offset + j of the buffer of a piece from start that holds
# [first, first + count) is the value of cell first + j.
def band_cell(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+first: U32,+count: U32,+start: U32,+span: U32,+tree: Tree.Tree<Element>,+length: Nat,
  +capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+index: Nat,+after: {U32.is_le(start,first) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(span)),U32.to_nat(capacity)) == True{} : Bool},
  +within: {U32.is_le((first + count : U32),(start + span : U32)) == True{} : Bool},
  +fits_count: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(capacity)) == True{} : Bool},+less: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool},
  +piece: Content.Leaf(~Element,~Context,~value,context,start,span,tree,length)) ->
  {Tree.read(Element,tree,U32.from_nat(Nat.add(U32.to_nat((first - start : U32)),index))) == value(context,U32.from_nat(Nat.add(U32.to_nat(first),index))) : Element}:
  +ordered = Bounds.as_nat_le(start,first,after)
  +offset = Nat.sub(U32.to_nat(first),U32.to_nat(start))
  +exact_band = Arithmetic.add(first,count,capacity,capacity_ok,fits_count)
  +exact_piece = Arithmetic.add(start,span,capacity,capacity_ok,room)
  +fits = nat_le_right_moved(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.add(U32.to_nat(start),U32.to_nat(span)),U32.to_nat((start + span : U32)),
    Equal.sym(Nat,U32.to_nat((start + span : U32)),Nat.add(U32.to_nat(start),U32.to_nat(span)),exact_piece),
    nat_le_moved(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat((first + count : U32)),U32.to_nat((start + span : U32)),
      Equal.sym(Nat,U32.to_nat((first + count : U32)),Nat.add(U32.to_nat(first),U32.to_nat(count)),exact_band),
      Bounds.as_nat_le((first + count : U32),(start + span : U32),within)))
  +exact_offset = Arithmetic.subtract(first,start,capacity,capacity_ok,ordered,Order.transitive(offset,U32.to_nat(first),U32.to_nat(capacity),Order.sub_le(U32.to_nat(first),U32.to_nat(start)),
    Order.transitive(U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(capacity),Order.le_add(U32.to_nat(first),U32.to_nat(count)),fits_count)))
  +inside = offset_inside(U32.to_nat(first),U32.to_nat(start),U32.to_nat(count),U32.to_nat(span),index,ordered,fits,less)
  +read = at_value(~Element,~Context,~value,context,start,tree,length,Nat.add(offset,index),
    Content.leaf_cell(~Element,~Context,~value,context,start,span,tree,length,Nat.add(offset,index),piece,inside))
  +cell = Equal.trans(Nat,Nat.add(U32.to_nat(start),Nat.add(offset,index)),Nat.add(Nat.add(U32.to_nat(start),offset),index),Nat.add(U32.to_nat(first),index),
    Equal.sym(Nat,Nat.add(Nat.add(U32.to_nat(start),offset),index),Nat.add(U32.to_nat(start),Nat.add(offset,index)),Addition.associative(U32.to_nat(start),offset,index)),
    Equal.cong(Nat,Nat,value => Nat.add(value,index),Nat.add(U32.to_nat(start),offset),U32.to_nat(first),Order.restore_shift(U32.to_nat(start),U32.to_nat(first),ordered)))
  Equal.trans(Element,Tree.read(Element,tree,U32.from_nat(Nat.add(U32.to_nat((first - start : U32)),index))),Tree.read(Element,tree,U32.from_nat(Nat.add(offset,index))),
    value(context,U32.from_nat(Nat.add(U32.to_nat(first),index))),
    Equal.cong(Nat,Element,position => Tree.read(Element,tree,U32.from_nat(Nat.add(position,index))),U32.to_nat((first - start : U32)),offset,exact_offset),
    Equal.trans(Element,Tree.read(Element,tree,U32.from_nat(Nat.add(offset,index))),value(context,U32.from_nat(Nat.add(U32.to_nat(start),Nat.add(offset,index)))),
      value(context,U32.from_nat(Nat.add(U32.to_nat(first),index))),read,
      Equal.cong(Nat,Element,position => value(context,U32.from_nat(position)),Nat.add(U32.to_nat(start),Nat.add(offset,index)),Nat.add(U32.to_nat(first),index),cell)))

def owners_output(~Element: Data,owners: CopyLaws.Owners<Element>) -> Buffer.Content<Element>:
  CopyLaws.Owners{source,source_length,output,output_length} = owners
  Buffer.Content{output,output_length}

def output_contents(~Element: Data,pair: Storage.Buffer<Element> & Storage.Buffer<Element>) ->
  {Buffer.contents(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,pair)) == owners_output(~Element,CopyLaws.observe(~Element,pair)) : Buffer.Content<Element>}:
  (Storage.Buffer{input,input_length,input_certificate},Storage.Buffer{output,output_length,output_certificate}) = pair
  {==}

def content_parts(~Element: Data,+model: Buffer.Model<Element>) ->
  {Buffer.contents(~Element,Buffer.pack(~Element,model)) == Buffer.Content{Buffer.tree(~Element,model),Buffer.length(~Element,model)} : Buffer.Content<Element>}:
  Buffer.Model{tree,length,certificate} = model
  Buffer.packed_contents(~Element,Buffer.Model{tree,length,certificate})

def pair_eta(~First: Type,~Second: Type,pair: First & Second) -> {pair == (Pair.fst(First,Second,pair),Pair.snd(First,Second,pair)) : First & Second}:
  (first,second) = pair
  {==}

# The view a run copies through and its one-axis view copy the same writes.
def run_view_writes(~Element: Data,fast: Bool,+count: Nat,+cells: U32,+stride: U32,+back: Bool,+local: U32,+destination: U32,+input: Tree.Tree<Element>) ->
  {CopyValues.copy_writes(~Element,fast,count,Views.View{Move.run_axes(fast,cells,stride,back),local},destination,input) ==
    CopyValues.copy_writes(~Element,fast,count,Views.View{[Views.Axis{cells,Move.directed(back,stride)}],local},destination,input) : List<&2,Writes.Assignment<Element>>}:
  match fast:
    case True{}: {==}
    case False{}: {==}

# A copy of cells cells from local of a buffer, stride apart (backward when
# back), into a target from position writes the copy reader of its view.
def copy_step(~Element: Data,+cells: U32,+local: U32,+stride: U32,+back: Bool,+position: U32,+source: Buffer.Model<Element>,+target: Buffer.Model<Element>) ->
  {Buffer.contents(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,
      Move.copied_run(~Element,cells,local,stride,back,position,Buffer.pack(~Element,source),Buffer.pack(~Element,target)))) ==
    Buffer.Content{Writes.pure(Element,CopyValues.copy_writes(~Element,U32.is_eq(stride,1) && Bool.not(back),U32.to_nat(cells),
      Views.View{[Views.Axis{cells,Move.directed(back,stride)}],local},position,Buffer.tree(~Element,source)),Buffer.tree(~Element,target)),Buffer.length(~Element,target)} :
    Buffer.Content<Element>}:
  Buffer.Model{+tree,+length,+certificate} = source
  Buffer.Model{+target_tree,+target_length,+target_certificate} = target
  +fast = U32.is_eq(stride,1) && Bool.not(back)
  +view = {Views.View{Move.run_axes(fast,cells,stride,back),local} : Views.View}
  %run_view_writes(~Element,fast,U32.to_nat(cells),cells,stride,back,local,position,tree) :
    {Buffer.contents(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,
      Move.copied_run(~Element,cells,local,stride,back,position,Buffer.pack(~Element,Buffer.Model{tree,length,certificate}),Buffer.pack(~Element,Buffer.Model{target_tree,target_length,target_certificate})))) ==
      Buffer.Content{Writes.pure(Element,_,target_tree),target_length} : Buffer.Content<Element>}
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,Copy.view(~Element,fast,cells,view,position,Storage.Buffer{Tree.pack(Element,tree),length,certificate},Storage.Buffer{Tree.pack(Element,target_tree),target_length,target_certificate}))),owners_output(~Element,CopyLaws.observe(~Element,Copy.view(~Element,fast,cells,view,position,Storage.Buffer{Tree.pack(Element,tree),length,certificate},Storage.Buffer{Tree.pack(Element,target_tree),target_length,target_certificate}))),
    output_contents(~Element,Copy.view(~Element,fast,cells,view,position,Storage.Buffer{Tree.pack(Element,tree),length,certificate},Storage.Buffer{Tree.pack(Element,target_tree),target_length,target_certificate}))) :
    {_ == Buffer.Content{Writes.pure(Element,CopyValues.copy_writes(~Element,fast,U32.to_nat(cells),view,position,tree),target_tree),target_length} : Buffer.Content<Element>}
  %Equal.sym(CopyLaws.Owners<Element>,CopyLaws.observe(~Element,Copy.view(~Element,fast,cells,view,position,Storage.Buffer{Tree.pack(Element,tree),length,certificate},Storage.Buffer{Tree.pack(Element,target_tree),target_length,target_certificate})),
    CopyLaws.Owners{tree,length,Writes.pure(Element,CopyValues.copy_writes(~Element,fast,U32.to_nat(cells),view,position,tree),target_tree),target_length},
    CopyLaws.view_packed(~Element,fast,cells,view,position,tree,length,target_tree,target_length,certificate,target_certificate)) :
    {owners_output(~Element,_) == Buffer.Content{Writes.pure(Element,CopyValues.copy_writes(~Element,fast,U32.to_nat(cells),view,position,tree),target_tree),target_length} : Buffer.Content<Element>}
  {==}

# A copy returns its source buffer.
def copied_step_pair(~Element: Data,+cells: U32,+local: U32,+stride: U32,+back: Bool,+position: U32,+source: Buffer.Model<Element>,+target: Buffer.Model<Element>) ->
  {Move.copied_run(~Element,cells,local,stride,back,position,Buffer.pack(~Element,source),Buffer.pack(~Element,target)) ==
    (Buffer.pack(~Element,source),Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,Move.copied_run(~Element,cells,local,stride,back,position,Buffer.pack(~Element,source),Buffer.pack(~Element,target)))) :
    Storage.Buffer<Element> & Storage.Buffer<Element>}:
  Equal.trans(Storage.Buffer<Element> & Storage.Buffer<Element>,Move.copied_run(~Element,cells,local,stride,back,position,Buffer.pack(~Element,source),Buffer.pack(~Element,target)),
    (Pair.fst(Storage.Buffer<Element>,Storage.Buffer<Element>,Move.copied_run(~Element,cells,local,stride,back,position,Buffer.pack(~Element,source),Buffer.pack(~Element,target))),
      Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,Move.copied_run(~Element,cells,local,stride,back,position,Buffer.pack(~Element,source),Buffer.pack(~Element,target)))),
    (Buffer.pack(~Element,source),Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,Move.copied_run(~Element,cells,local,stride,back,position,Buffer.pack(~Element,source),Buffer.pack(~Element,target)))),
    pair_eta(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Move.copied_run(~Element,cells,local,stride,back,position,Buffer.pack(~Element,source),Buffer.pack(~Element,target))),
    Equal.cong(Storage.Buffer<Element>,Storage.Buffer<Element> & Storage.Buffer<Element>,
      first => (first,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,Move.copied_run(~Element,cells,local,stride,back,position,Buffer.pack(~Element,source),Buffer.pack(~Element,target)))),
      Pair.fst(Storage.Buffer<Element>,Storage.Buffer<Element>,Move.copied_run(~Element,cells,local,stride,back,position,Buffer.pack(~Element,source),Buffer.pack(~Element,target))),
      Buffer.pack(~Element,source),
      CopyLaws.source(~Element,U32.is_eq(stride,1) && Bool.not(back),cells,Views.View{Move.run_axes(U32.is_eq(stride,1) && Bool.not(back),cells,stride,back),local},position,Buffer.pack(~Element,source),Buffer.pack(~Element,target))))

# A negation that holds names a false flag.
def forward_of(back: Bool,+not_back: {Bool.not(back) == True{} : Bool}) -> {back == False{} : Bool}:
  match back:
    case True{}: Empty.absurd({True{} == False{} : Bool},WordEncoding.false_true(not_back))
    case False{}: {==}

# ---- The cells of a strided copy ----

# One coordinate of an axis stepped count times from coordinate, below its
# extent.
def iterate_single(+extent: Nat,+stride: Nat,count: Nat,+coordinate: Nat,+inside: {Nat.is_lt(Nat.add(coordinate,count),extent) == True{} : Bool}) ->
  {CursorOrder.iterate(count,[CursorModel.Coordinate{extent,coordinate,stride}]) == [CursorModel.Coordinate{extent,Nat.add(coordinate,count),stride}] : List<&2,CursorModel.Coordinate>}:
  match count:
    case 0n:
      %Equal.sym(Nat,Nat.add(coordinate,0n),coordinate,Addition.zero_right(coordinate)) :
        {[CursorModel.Coordinate{extent,coordinate,stride}] == [CursorModel.Coordinate{extent,_,stride}] : List<&2,CursorModel.Coordinate>}
      {==}
    case 1n+ +rest:
      +stepped = Order.le_lt_trans(1n+coordinate,Nat.add(coordinate,1n+rest),extent,
        %Equal.sym(Nat,Nat.add(coordinate,1n+rest),1n+Nat.add(coordinate,rest),Addition.successor_right(coordinate,rest)) : {Nat.is_le(1n+coordinate,_) == True{} : Bool}
          Order.le_add(coordinate,rest),inside)
      %Equal.sym(Bool,Nat.is_lt(1n+coordinate,extent),True{},stepped) :
        {CursorOrder.iterate(rest,Bool.pick(List<&2,CursorModel.Coordinate>,_,[CursorModel.Coordinate{extent,1n+coordinate,stride}],
          CursorModel.Coordinate{extent,0n,stride} <> CursorModel.advance([]))) ==
          [CursorModel.Coordinate{extent,Nat.add(coordinate,1n+rest),stride}] : List<&2,CursorModel.Coordinate>}
      %Equal.sym(Nat,Nat.add(coordinate,1n+rest),1n+Nat.add(coordinate,rest),Addition.successor_right(coordinate,rest)) :
        {CursorOrder.iterate(rest,[CursorModel.Coordinate{extent,1n+coordinate,stride}]) == [CursorModel.Coordinate{extent,_,stride}] : List<&2,CursorModel.Coordinate>}
      iterate_single(extent,stride,rest,1n+coordinate,%Addition.successor_right(coordinate,rest) : {Nat.is_lt(_,extent) == True{} : Bool} inside)

# A U32 address from a natural one: the natural offset plus index times stride.
def word_address(+local: U32,+step: U32,+index: Nat) ->
  {U32.from_nat(Nat.add(Nat.mul(index,U32.to_nat(step)),U32.to_nat(local))) == (local + U32.from_nat(index) * step : U32) : U32}:
  +product = Nat.mul(index,U32.to_nat(step))
  Equal.trans(U32,U32.from_nat(Nat.add(product,U32.to_nat(local))),(U32.from_nat(product) + U32.from_nat(U32.to_nat(local)) : U32),(local + U32.from_nat(index) * step : U32),
    Equal.sym(U32,(U32.from_nat(product) + U32.from_nat(U32.to_nat(local)) : U32),U32.from_nat(Nat.add(product,U32.to_nat(local))),WordArithmetic.native_add(product,U32.to_nat(local))),
    %Equal.sym(U32,U32.from_nat(U32.to_nat(local)),local,Arithmetic.native_roundtrip(local)) : {(U32.from_nat(product) + _ : U32) == (local + U32.from_nat(index) * step : U32) : U32}
    %Multiplication.native_encoded_multiply(index,U32.to_nat(step)) :
      {(_ + local : U32) == (local + U32.from_nat(index) * step : U32) : U32}
    %Equal.sym(U32,U32.from_nat(U32.to_nat(step)),step,Arithmetic.native_roundtrip(step)) : {((U32.from_nat(index) * _ : U32) + local : U32) == (local + U32.from_nat(index) * step : U32) : U32}
    Arithmetic.add_commutative((U32.from_nat(index) * step : U32),local))

def single_address_unit(+extent: U32,+step: U32,+local: U32,index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(extent)) == True{} : Bool},
  +few: {Nat.is_lt(1n,U32.to_nat(extent)) == False{} : Bool}) ->
  {CopyValues.address(Views.View{[Views.Axis{extent,step}],local},index) == (local + U32.from_nat(index) * step : U32) : U32}:
  match index:
    case 0n:
      %Equal.sym(Bool,Nat.is_lt(1n,U32.to_nat(extent)),False{},few) :
        {U32.from_nat(CursorModel.address(CursorOrder.iterate(0n,Bool.pick(List<&2,CursorModel.Coordinate>,_,[CursorModel.Coordinate{U32.to_nat(extent),0n,U32.to_nat(step)}],[])),U32.to_nat(local))) ==
          (local + U32.from_nat(0n) * step : U32) : U32}
      Equal.trans(U32,U32.from_nat(U32.to_nat(local)),local,(local + 0 * step : U32),Arithmetic.native_roundtrip(local),
        Equal.sym(U32,(local + 0 * step : U32),local,
          Equal.trans(U32,(local + 0 * step : U32),(local + 0 : U32),local,
            Equal.cong(U32,U32,product => (local + product : U32),(0 * step : U32),0,Arithmetic.multiply_zero_left(step)),Arithmetic.add_zero(local))))
    case 1n+ +rest:
      Empty.absurd({CopyValues.address(Views.View{[Views.Axis{extent,step}],local},1n+rest) == (local + U32.from_nat(1n+rest) * step : U32) : U32},
        Order.below_zero(rest,Bounds.nat_trans(1n+rest,U32.to_nat(extent),1n,inside,lt_false_le(1n,U32.to_nat(extent),few))))

# Index k below the extent of a view of one axis lies at its offset plus k
# times its stride.
def single_address(+extent: U32,+step: U32,+local: U32,+index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(extent)) == True{} : Bool}) ->
  {CopyValues.address(Views.View{[Views.Axis{extent,step}],local},index) == (local + U32.from_nat(index) * step : U32) : U32}:
  by_cases({CopyValues.address(Views.View{[Views.Axis{extent,step}],local},index) == (local + U32.from_nat(index) * step : U32) : U32},Nat.is_lt(1n,U32.to_nat(extent)),
    +many =>
      %Equal.sym(Bool,Nat.is_lt(1n,U32.to_nat(extent)),True{},many) :
        {U32.from_nat(CursorModel.address(CursorOrder.iterate(index,Bool.pick(List<&2,CursorModel.Coordinate>,_,[CursorModel.Coordinate{U32.to_nat(extent),0n,U32.to_nat(step)}],[])),U32.to_nat(local))) ==
          (local + U32.from_nat(index) * step : U32) : U32}
      %Equal.sym(List<&2,CursorModel.Coordinate>,CursorOrder.iterate(index,[CursorModel.Coordinate{U32.to_nat(extent),0n,U32.to_nat(step)}]),
        [CursorModel.Coordinate{U32.to_nat(extent),Nat.add(0n,index),U32.to_nat(step)}],iterate_single(U32.to_nat(extent),U32.to_nat(step),index,0n,inside)) :
        {U32.from_nat(CursorModel.address(_,U32.to_nat(local))) == (local + U32.from_nat(index) * step : U32) : U32}
      word_address(local,step,index),
    +few => single_address_unit(extent,step,local,index,inside,few))

# A unit-stride view of one axis lies at consecutive addresses from its offset.
def unit_line(+extent: U32,+local: U32,count: Nat,+position: Nat,+room: {Nat.is_le(Nat.add(position,count),U32.to_nat(extent)) == True{} : Bool}) ->
  CopyValues.Line(Views.View{[Views.Axis{extent,1}],local},U32.to_nat(local),count,position):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      KernelGemm.Both{
        Equal.trans(U32,CopyValues.address(Views.View{[Views.Axis{extent,1}],local},position),(local + U32.from_nat(position) * 1 : U32),U32.from_nat(Nat.add(U32.to_nat(local),position)),
          single_address(extent,1,local,position,Order.room_index(position,rest,U32.to_nat(extent),room)),
          Equal.trans(U32,(local + U32.from_nat(position) * 1 : U32),(local + U32.from_nat(position) : U32),U32.from_nat(Nat.add(U32.to_nat(local),position)),
            Equal.cong(U32,U32,word => (local + word : U32),(U32.from_nat(position) * 1 : U32),U32.from_nat(position),Arithmetic.multiply_one(U32.from_nat(position))),
            CopyValues.offset_address(local,position))),
        unit_line(extent,local,rest,1n+position,Order.advance_room(position,rest,U32.to_nat(extent),room))}

# The writes of a copy of a view of one axis are its reader at consecutive
# target addresses, for the fast and the strided path alike.
def step_line(~Element: Data,+cells: U32,+local: U32,+stride: U32,+back: Bool,fast: Bool,+destination: U32,+input: Tree.Tree<Element>,
  +capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(U32.to_nat(cells),U32.to_nat(capacity)) == True{} : Bool},
  +flag: {fast == (U32.is_eq(stride,1) && Bool.not(back)) : Bool}) ->
  {CopyValues.copy_writes(~Element,fast,U32.to_nat(cells),Views.View{[Views.Axis{cells,Move.directed(back,stride)}],local},destination,input) ==
    Consecutive.writes(~Element,~CopyValues.Sampled<Element>,~CopyValues.sampled(~Element),U32.to_nat(cells),U32.to_nat(destination),0n,
      CopyValues.Sampled{Views.View{[Views.Axis{cells,Move.directed(back,stride)}],local},input}) : List<&2,Writes.Assignment<Element>>}:
  match fast:
    case False{}:
      %Arithmetic.add_zero(destination) :
        {CopyValues.visit_writes(~Element,~CursorLaws.Coordinates,~TensorVisit.sample(~Element),~CursorLaws.next,U32.to_nat(cells),_,
          CursorLaws.initial(Views.View{[Views.Axis{cells,Move.directed(back,stride)}],local}),input) ==
          Consecutive.writes(~Element,~CopyValues.Sampled<Element>,~CopyValues.sampled(~Element),U32.to_nat(cells),U32.to_nat(destination),0n,
            CopyValues.Sampled{Views.View{[Views.Axis{cells,Move.directed(back,stride)}],local},input}) : List<&2,Writes.Assignment<Element>>}
      CopyValues.visit_line(~Element,~CursorLaws.Coordinates,~CopyValues.Sampled<Element>,~TensorVisit.sample(~Element),~CursorLaws.next,~CopyValues.sampled(~Element),
        ~CopyValues.sampled_start(~Element),~CopyValues.sampled_input(~Element),~CopyValues.sampled_agrees(~Element),
        CopyValues.Sampled{Views.View{[Views.Axis{cells,Move.directed(back,stride)}],local},input},U32.to_nat(cells),0n,destination)
    case True{}:
      +unit = Arithmetic.equal_of_is_eq(stride,1,Logic.left(U32.is_eq(stride,1),Bool.not(back),Equal.sym(Bool,True{},U32.is_eq(stride,1) && Bool.not(back),flag)))
      +forward = forward_of(back,Logic.right(U32.is_eq(stride,1),Bool.not(back),Equal.sym(Bool,True{},U32.is_eq(stride,1) && Bool.not(back),flag)))
      %Equal.sym(Bool,back,False{},forward) :
        {CopyValues.copy_writes(~Element,True{},U32.to_nat(cells),Views.View{[Views.Axis{cells,Move.directed(_,stride)}],local},destination,input) ==
          Consecutive.writes(~Element,~CopyValues.Sampled<Element>,~CopyValues.sampled(~Element),U32.to_nat(cells),U32.to_nat(destination),0n,
            CopyValues.Sampled{Views.View{[Views.Axis{cells,Move.directed(_,stride)}],local},input}) : List<&2,Writes.Assignment<Element>>}
      %Equal.sym(U32,stride,1,unit) :
        {CopyValues.copy_writes(~Element,True{},U32.to_nat(cells),Views.View{[Views.Axis{cells,Move.directed(False{},_)}],local},destination,input) ==
          Consecutive.writes(~Element,~CopyValues.Sampled<Element>,~CopyValues.sampled(~Element),U32.to_nat(cells),U32.to_nat(destination),0n,
            CopyValues.Sampled{Views.View{[Views.Axis{cells,Move.directed(False{},_)}],local},input}) : List<&2,Writes.Assignment<Element>>}
      CopyValues.fast_line(~Element,U32.to_nat(cells),0n,[Views.Axis{cells,1}],local,destination,input,capacity,capacity_ok,room,
        unit_line(cells,local,U32.to_nat(cells),0n,Order.reflexive(U32.to_nat(cells))))

# The cells a unit step copies from the current piece, the target tree after
# it and the target buffer after it.
def step_cells(~Element: Data,holding: Holding<Element>,+count: U32,+offset: U32,+stride: U32,+back: Bool) -> U32:
  Holding{behind,ahead} = holding
  match ahead:
    case Held{buffer,+first,+span,below}: Move.inside(count,offset,stride,back,first,span)
    case Bottom{}: 0

def step_tree(~Element: Data,holding: Holding<Element>,+count: U32,+offset: U32,+stride: U32,+back: Bool,+position: U32,tree: Tree.Tree<Element>) -> Tree.Tree<Element>:
  Holding{behind,ahead} = holding
  match ahead:
    case Held{+buffer,+first,+span,below}:
      Writes.pure(Element,CopyValues.copy_writes(~Element,U32.is_eq(stride,1) && Bool.not(back),U32.to_nat(Move.inside(count,offset,stride,back,first,span)),Views.View{[Views.Axis{Move.inside(count,offset,stride,back,first,span),Move.directed(back,stride)}],(offset - first : U32)},position,Buffer.tree(~Element,buffer)),tree)
    case Bottom{}: tree

def step_target(~Element: Data,holding: Holding<Element>,+count: U32,+offset: U32,+stride: U32,+back: Bool,+position: U32,target: Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  Holding{behind,ahead} = holding
  match ahead:
    case Held{+buffer,+first,+span,below}:
      Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,
        Move.copied_run(~Element,Move.inside(count,offset,stride,back,first,span),(offset - first : U32),stride,back,position,Buffer.pack(~Element,buffer),target))
    case Bottom{}: target

def step_shape(~Element: Data,+holding: Holding<Element>,+count: U32,+offset: U32,+stride: U32,+back: Bool,+position: U32,+target: Buffer.Model<Element>) ->
  {Move.step(~Element,pack(~Element,holding),count,offset,stride,back,position,Buffer.pack(~Element,target)) ==
    Move.stepped(~Element,pack(~Element,holding),step_target(~Element,holding,count,offset,stride,back,position,Buffer.pack(~Element,target)),count,offset,stride,back,position,
      step_cells(~Element,holding,count,offset,stride,back)) : Move.Stepped<Element>}:
  match holding:
    case Holding{+behind,Held{+buffer,+first,+span,+below}}:
      +cells = Move.inside(count,offset,stride,back,first,span)
      %Equal.sym(Storage.Buffer<Element> & Storage.Buffer<Element>,Move.copied_run(~Element,cells,(offset - first : U32),stride,back,position,Buffer.pack(~Element,buffer),Buffer.pack(~Element,target)),
        (Buffer.pack(~Element,buffer),Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,
          Move.copied_run(~Element,cells,(offset - first : U32),stride,back,position,Buffer.pack(~Element,buffer),Buffer.pack(~Element,target)))),
        copied_step_pair(~Element,cells,(offset - first : U32),stride,back,position,buffer,target)) :
        {Move.stepped_in(~Element,count,offset,stride,back,position,cells,pack_stack(~Element,behind),first,span,pack_stack(~Element,below),_) ==
          Move.stepped(~Element,pack(~Element,Holding{behind,Held{buffer,first,span,below}}),
            step_target(~Element,Holding{behind,Held{buffer,first,span,below}},count,offset,stride,back,position,Buffer.pack(~Element,target)),count,offset,stride,back,position,cells) :
          Move.Stepped<Element>}
      {==}
    case Holding{+behind,Bottom{}}: {==}

def step_contents(~Element: Data,+holding: Holding<Element>,+count: U32,+offset: U32,+stride: U32,+back: Bool,+position: U32,+target: Buffer.Model<Element>) ->
  {Buffer.contents(~Element,step_target(~Element,holding,count,offset,stride,back,position,Buffer.pack(~Element,target))) ==
    Buffer.Content{step_tree(~Element,holding,count,offset,stride,back,position,Buffer.tree(~Element,target)),Buffer.length(~Element,target)} : Buffer.Content<Element>}:
  match holding:
    case Holding{+behind,Held{+buffer,+first,+span,+below}}: copy_step(~Element,Move.inside(count,offset,stride,back,first,span),(offset - first : U32),stride,back,position,buffer,target)
    case Holding{+behind,Bottom{}}: content_parts(~Element,target)

# The model of a forward run of unit stride: the holding after it and the
# target tree.
type Ran<-Element: Data> is Data:
  Ran{holding: Holding<Element>,tree: Tree.Tree<Element>}

def ran_holding(~Element: Data,ran: Ran<Element>) -> Holding<Element>:
  Ran{holding,tree} = ran
  holding

def ran_tree(~Element: Data,ran: Ran<Element>) -> Tree.Tree<Element>:
  Ran{holding,tree} = ran
  tree

def model_run(~Element: Data,fuel: Nat,done: Bool,+count: U32,+offset: U32,+stride: U32,+back: Bool,+position: U32,+holding: Holding<Element>,+tree: Tree.Tree<Element>) -> Ran<Element>:
  match fuel done:
    case 0n done: Ran{holding,tree}
    case 1n+rest True{}: Ran{holding,tree}
    case 1n+rest False{}:
      +resolved = resolve(~Element,holding,offset)
      +cells = step_cells(~Element,resolved,count,offset,stride,back)
      model_run(~Element,rest,U32.is_zero((count - cells : U32)),(count - cells : U32),(offset + cells * Move.directed(back,stride) : U32),stride,back,(position + cells : U32),resolved,
        step_tree(~Element,resolved,count,offset,stride,back,position,tree))

type RunResult<-Element: Data,-result: Move.Holding<Element> & Storage.Buffer<Element>,-ran: Ran<Element>,-length: Nat> is Data:
  RunResult{holding: {Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,result) == pack(~Element,ran_holding(~Element,ran)) : Move.Holding<Element>},
    target: {Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,result)) == Buffer.Content{ran_tree(~Element,ran),length} : Buffer.Content<Element>}}

def RunFact(~Element: Data,result: Move.Holding<Element> & Storage.Buffer<Element>,ran: Ran<Element>,length: Nat) -> Data:
  RunResult<Element,result,ran,length>

def step_tree_of(~Element: Data,+holding: Holding<Element>,+count: U32,+offset: U32,+stride: U32,+back: Bool,+position: U32,+target: Buffer.Model<Element>) ->
  {Buffer.tree(~Element,Buffer.remodel(~Element,step_target(~Element,holding,count,offset,stride,back,position,Buffer.pack(~Element,target)))) ==
    step_tree(~Element,holding,count,offset,stride,back,position,Buffer.tree(~Element,target)) : Tree.Tree<Element>}:
  +next = Buffer.remodel(~Element,step_target(~Element,holding,count,offset,stride,back,position,Buffer.pack(~Element,target)))
  +content = Equal.trans(Buffer.Content<Element>,Buffer.content(~Element,next),Buffer.contents(~Element,step_target(~Element,holding,count,offset,stride,back,position,Buffer.pack(~Element,target))),
    Buffer.Content{step_tree(~Element,holding,count,offset,stride,back,position,Buffer.tree(~Element,target)),Buffer.length(~Element,target)},
    Buffer.remodel_content(~Element,step_target(~Element,holding,count,offset,stride,back,position,Buffer.pack(~Element,target))),step_contents(~Element,holding,count,offset,stride,back,position,target))
  Equal.trans(Tree.Tree<Element>,Buffer.tree(~Element,next),Buffer.content_tree(~Element,Buffer.content(~Element,next)),step_tree(~Element,holding,count,offset,stride,back,position,Buffer.tree(~Element,target)),
    Equal.sym(Tree.Tree<Element>,Buffer.content_tree(~Element,Buffer.content(~Element,next)),Buffer.tree(~Element,next),Buffer.content_tree_of(~Element,next)),
    Equal.cong(Buffer.Content<Element>,Tree.Tree<Element>,Buffer.content_tree(~Element),Buffer.content(~Element,next),
      Buffer.Content{step_tree(~Element,holding,count,offset,stride,back,position,Buffer.tree(~Element,target)),Buffer.length(~Element,target)},content))

def step_length_of(~Element: Data,+holding: Holding<Element>,+count: U32,+offset: U32,+stride: U32,+back: Bool,+position: U32,+target: Buffer.Model<Element>) ->
  {Buffer.length(~Element,Buffer.remodel(~Element,step_target(~Element,holding,count,offset,stride,back,position,Buffer.pack(~Element,target)))) == Buffer.length(~Element,target) : Nat}:
  +next = Buffer.remodel(~Element,step_target(~Element,holding,count,offset,stride,back,position,Buffer.pack(~Element,target)))
  +content = Equal.trans(Buffer.Content<Element>,Buffer.content(~Element,next),Buffer.contents(~Element,step_target(~Element,holding,count,offset,stride,back,position,Buffer.pack(~Element,target))),
    Buffer.Content{step_tree(~Element,holding,count,offset,stride,back,position,Buffer.tree(~Element,target)),Buffer.length(~Element,target)},
    Buffer.remodel_content(~Element,step_target(~Element,holding,count,offset,stride,back,position,Buffer.pack(~Element,target))),step_contents(~Element,holding,count,offset,stride,back,position,target))
  Equal.trans(Nat,Buffer.length(~Element,next),Buffer.content_length(~Element,Buffer.content(~Element,next)),Buffer.length(~Element,target),
    Equal.sym(Nat,Buffer.content_length(~Element,Buffer.content(~Element,next)),Buffer.length(~Element,next),Buffer.content_length_of(~Element,next)),
    Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),Buffer.content(~Element,next),
      Buffer.Content{step_tree(~Element,holding,count,offset,stride,back,position,Buffer.tree(~Element,target)),Buffer.length(~Element,target)},content))

# A runtime run of unit stride on a packed holding into a packed target is
# its model: the same holding and the model tree in the target.
def run_refines(~Element: Data,fuel: Nat,+done: Bool,+count: U32,+offset: U32,+stride: U32,+back: Bool,+position: U32,+holding: Holding<Element>,+target: Buffer.Model<Element>) ->
  RunFact(~Element,Move.run_on(~Element,fuel,stride,back,Move.Stepped{pack(~Element,holding),Buffer.pack(~Element,target),count,offset,position,done}),
    model_run(~Element,fuel,done,count,offset,stride,back,position,holding,Buffer.tree(~Element,target)),Buffer.length(~Element,target)):
  match fuel done:
    case 0n done: RunResult{{==},content_parts(~Element,target)}
    case 1n+rest True{}: RunResult{{==},content_parts(~Element,target)}
    case 1n+ +rest False{}:
      +resolved = resolve(~Element,holding,offset)
      +cells = step_cells(~Element,resolved,count,offset,stride,back)
      +left = (count - cells : U32)
      +moved = (offset + cells * Move.directed(back,stride) : U32)
      +after = (position + cells : U32)
      +next = Buffer.remodel(~Element,step_target(~Element,resolved,count,offset,stride,back,position,Buffer.pack(~Element,target)))
      %Equal.sym(Move.Holding<Element>,Move.resolve(~Element,pack(~Element,holding),offset),pack(~Element,resolved),resolve_packed(~Element,holding,offset)) :
        RunFact(~Element,Move.run_on(~Element,rest,stride,back,Move.step(~Element,_,count,offset,stride,back,position,Buffer.pack(~Element,target))),
          model_run(~Element,1n+rest,False{},count,offset,stride,back,position,holding,Buffer.tree(~Element,target)),Buffer.length(~Element,target))
      %Equal.sym(Move.Stepped<Element>,Move.step(~Element,pack(~Element,resolved),count,offset,stride,back,position,Buffer.pack(~Element,target)),
        Move.stepped(~Element,pack(~Element,resolved),step_target(~Element,resolved,count,offset,stride,back,position,Buffer.pack(~Element,target)),count,offset,stride,back,position,cells),
        step_shape(~Element,resolved,count,offset,stride,back,position,target)) :
        RunFact(~Element,Move.run_on(~Element,rest,stride,back,_),
          model_run(~Element,1n+rest,False{},count,offset,stride,back,position,holding,Buffer.tree(~Element,target)),Buffer.length(~Element,target))
      %Buffer.remodeled(~Element,step_target(~Element,resolved,count,offset,stride,back,position,Buffer.pack(~Element,target))) :
        RunFact(~Element,Move.run_on(~Element,rest,stride,back,Move.Stepped{pack(~Element,resolved),_,left,moved,after,U32.is_zero(left)}),
          model_run(~Element,1n+rest,False{},count,offset,stride,back,position,holding,Buffer.tree(~Element,target)),Buffer.length(~Element,target))
      %step_tree_of(~Element,resolved,count,offset,stride,back,position,target) :
        RunFact(~Element,Move.run_on(~Element,rest,stride,back,Move.Stepped{pack(~Element,resolved),Buffer.pack(~Element,next),left,moved,after,U32.is_zero(left)}),
          model_run(~Element,rest,U32.is_zero(left),left,moved,stride,back,after,resolved,_),Buffer.length(~Element,target))
      %step_length_of(~Element,resolved,count,offset,stride,back,position,target) :
        RunFact(~Element,Move.run_on(~Element,rest,stride,back,Move.Stepped{pack(~Element,resolved),Buffer.pack(~Element,next),left,moved,after,U32.is_zero(left)}),
          model_run(~Element,rest,U32.is_zero(left),left,moved,stride,back,after,resolved,Buffer.tree(~Element,next)),_)
      run_refines(~Element,rest,U32.is_zero(left),left,moved,stride,back,after,resolved,next)

# ---- What a unit run writes ----

def pure_balanced(~Element: Data,writes: List<&2,Writes.Assignment<Element>>,+depth: Nat,+tree: Tree.Tree<Element>,+balanced: Shape.Balanced(Element,depth,tree)) ->
  Shape.Balanced(Element,depth,Writes.pure(Element,writes,tree)):
  match writes:
    case []: balanced
    case Writes.Put{+index,+value} <> rest: pure_balanced(~Element,rest,depth,Tree.write(Element,tree,index,value),Shape.write(Element,depth,tree,index,value,balanced))

def pure_capacity(~Element: Data,writes: List<&2,Writes.Assignment<Element>>,+tree: Tree.Tree<Element>) ->
  {Tree.capacity(Element,Writes.pure(Element,writes,tree)) == Tree.capacity(Element,tree) : U32}:
  match writes:
    case []: {==}
    case Writes.Put{+index,+value} <> rest:
      Equal.trans(U32,Tree.capacity(Element,Writes.pure(Element,rest,Tree.write(Element,tree,index,value))),Tree.capacity(Element,Tree.write(Element,tree,index,value)),Tree.capacity(Element,tree),
        pure_capacity(~Element,rest,Tree.write(Element,tree,index,value)),StorageReadback.write_capacity(~Element,tree,index,value))

def step_balanced(~Element: Data,+holding: Holding<Element>,+count: U32,+offset: U32,+stride: U32,+back: Bool,+position: U32,+depth: Nat,+tree: Tree.Tree<Element>,+balanced: Shape.Balanced(Element,depth,tree)) ->
  Shape.Balanced(Element,depth,step_tree(~Element,holding,count,offset,stride,back,position,tree)):
  match holding:
    case Holding{+behind,Held{+buffer,+first,+span,+below}}:
      pure_balanced(~Element,CopyValues.copy_writes(~Element,U32.is_eq(stride,1) && Bool.not(back),U32.to_nat(Move.inside(count,offset,stride,back,first,span)),Views.View{[Views.Axis{Move.inside(count,offset,stride,back,first,span),Move.directed(back,stride)}],(offset - first : U32)},position,Buffer.tree(~Element,buffer)),depth,tree,balanced)
    case Holding{+behind,Bottom{}}: balanced

def step_capacity(~Element: Data,+holding: Holding<Element>,+count: U32,+offset: U32,+stride: U32,+back: Bool,+position: U32,+tree: Tree.Tree<Element>) ->
  {Tree.capacity(Element,step_tree(~Element,holding,count,offset,stride,back,position,tree)) == Tree.capacity(Element,tree) : U32}:
  match holding:
    case Holding{+behind,Held{+buffer,+first,+span,+below}}:
      pure_capacity(~Element,CopyValues.copy_writes(~Element,U32.is_eq(stride,1) && Bool.not(back),U32.to_nat(Move.inside(count,offset,stride,back,first,span)),Views.View{[Views.Axis{Move.inside(count,offset,stride,back,first,span),Move.directed(back,stride)}],(offset - first : U32)},position,Buffer.tree(~Element,buffer)),tree)
    case Holding{+behind,Bottom{}}: {==}

# The piece of offset ends at first + span within the capacity.
def piece_end(+first: U32,+span: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((first + span : U32)) == Nat.add(U32.to_nat(first),U32.to_nat(span)) : Nat}:
  Arithmetic.add(first,span,capacity,capacity_ok,room)

type Copies<-cells: U32,-count: U32> is Data:
  Copies{some: {Nat.is_lt(0n,U32.to_nat(cells)) == True{} : Bool},most: {Nat.is_le(U32.to_nat(cells),U32.to_nat(count)) == True{} : Bool}}

def nat_lt_left(+from: Nat,+to: Nat,+bound: Nat,+equal: {from == to : Nat},+less: {Nat.is_lt(to,bound) == True{} : Bool}) -> {Nat.is_lt(from,bound) == True{} : Bool}:
  Equal.trans(Bool,Nat.is_lt(from,bound),Nat.is_lt(to,bound),True{},Equal.cong(Nat,Bool,value => Nat.is_lt(value,bound),from,to,equal),less)

def le_shift(+offset: Nat,+left: Nat,+right: Nat,+order: {Nat.is_le(left,right) == True{} : Bool}) -> {Nat.is_le(Nat.add(offset,left),Nat.add(offset,right)) == True{} : Bool}:
  Equal.trans(Bool,Nat.is_le(Nat.add(offset,left),Nat.add(offset,right)),Nat.is_le(left,right),True{},Order.le_offset(offset,left,right),order)

def lt_shift(+offset: Nat,+left: Nat,+right: Nat,+less: {Nat.is_lt(left,right) == True{} : Bool}) -> {Nat.is_lt(Nat.add(offset,left),Nat.add(offset,right)) == True{} : Bool}:
  Equal.trans(Bool,Nat.is_lt(Nat.add(offset,left),Nat.add(offset,right)),Nat.is_lt(left,right),True{},Order.lt_offset(offset,left,right),less)

# ---- Steps of any stride ----

# The cells of the piece past offset in the direction of a step: before it
# when backward, after it when forward.
def step_reach(+back: Bool,+offset: U32,+first: U32,+span: U32) -> U32:
  Bool.pick(U32,back,(offset - first : U32),(((first + span : U32) - offset : U32) - 1 : U32))

# On a piece that holds offset, a step copies count cells at stride zero and
# otherwise one more than its reach over the stride, at most count.
def held_cells(+count: U32,+offset: U32,+stride: U32,+back: Bool,+first: U32,+span: U32,
  +after: {U32.is_le(first,offset) == True{} : Bool},+before: {U32.is_lt(offset,(first + span : U32)) == True{} : Bool}) ->
  {Move.inside(count,offset,stride,back,first,span) ==
    Bool.pick(U32,U32.is_zero(stride),count,U32.min(count,((step_reach(back,offset,first,span) / U32.max(stride,1) : U32) + 1 : U32))) : U32}:
  +inner = Bool.pick(U32,U32.is_zero(stride),count,U32.min(count,((step_reach(back,offset,first,span) / U32.max(stride,1) : U32) + 1 : U32)))
  %Equal.sym(Bool,U32.is_le(first,offset),True{},after) : {Bool.pick(U32,_ && U32.is_lt(offset,(first + span : U32)),inner,0) == inner : U32}
  %Equal.sym(Bool,U32.is_lt(offset,(first + span : U32)),True{},before) : {Bool.pick(U32,True{} && _,inner,0) == inner : U32}
  {==}

def directed_zero(+back: Bool) -> {Move.directed(back,0) == 0 : U32}:
  match back:
    case True{}: Arithmetic.subtract_zero(0)
    case False{}: {==}

def max_zero(value: Nat) -> {Nat.max(value,0n) == value : Nat}:
  match value:
    case 0n: {==}
    case 1n+rest: {==}

def max_one(+value: Nat,+positive: {Nat.is_lt(0n,value) == True{} : Bool}) -> {Nat.max(value,1n) == value : Nat}:
  match value:
    case 0n: Empty.absurd({Nat.max(0n,1n) == 0n : Nat},WordEncoding.false_true(positive))
    case 1n+ +rest: Equal.cong(Nat,Nat,number => 1n+number,Nat.max(rest,0n),rest,max_zero(rest))

# A stride that is not zero is the divisor of a step.
def stride_divisor(+stride: U32,+nonzero: {U32.is_zero(stride) == False{} : Bool}) -> {U32.to_nat(U32.max(stride,1)) == U32.to_nat(stride) : Nat}:
  Equal.trans(Nat,U32.to_nat(U32.max(stride,1)),Nat.max(U32.to_nat(stride),U32.to_nat(1)),U32.to_nat(stride),Extrema.maximum(stride,1),
    max_one(U32.to_nat(stride),Arithmetic.word_nonzero_positive(stride,nonzero)))

# A step of nonzero stride copies one more cell than its reach over the
# stride, before the bound by count.
def reach_whole(+reach: U32,+stride: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool},
  +nonzero: {U32.is_zero(stride) == False{} : Bool},+reach_room: {Nat.is_lt(U32.to_nat(reach),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat(((reach / U32.max(stride,1) : U32) + 1 : U32)) == 1n+Nat.div(U32.to_nat(reach),U32.to_nat(stride)) : Nat}:
  +positive = Arithmetic.word_nonzero_positive(stride,nonzero)
  +quotient = (reach / U32.max(stride,1) : U32)
  +whole = Nat.div(U32.to_nat(reach),U32.to_nat(stride))
  +quotient_exact = Division.quotient_exact(word,word_ok,reach,U32.max(stride,1),U32.to_nat(reach),U32.to_nat(stride),{==},stride_divisor(stride,nonzero),stride_room,positive)
  +quotient_small = nat_le_moved(U32.to_nat(quotient),whole,U32.to_nat(reach),quotient_exact,
    Order.transitive(whole,Nat.mul(whole,U32.to_nat(stride)),U32.to_nat(reach),RowMajor.product_covers(whole,U32.to_nat(stride),positive),
      DivisionOrder.product_below(U32.to_nat(reach),U32.to_nat(stride),positive)))
  +quotient_below = Order.le_lt_trans(U32.to_nat(quotient),U32.to_nat(reach),U32.to_nat(capacity),quotient_small,reach_room)
  Equal.trans(Nat,U32.to_nat((quotient + 1 : U32)),Nat.add(U32.to_nat(quotient),1n),1n+whole,
    Arithmetic.add(quotient,1,capacity,capacity_ok,
      nat_le_moved(Nat.add(U32.to_nat(quotient),1n),1n+U32.to_nat(quotient),U32.to_nat(capacity),Order.commutative(U32.to_nat(quotient),1n),
        Order.strict_follow(U32.to_nat(quotient),U32.to_nat(capacity),quotient_below))),
    Equal.trans(Nat,Nat.add(U32.to_nat(quotient),1n),1n+U32.to_nat(quotient),1n+whole,Order.commutative(U32.to_nat(quotient),1n),
      Equal.cong(Nat,Nat,number => 1n+number,U32.to_nat(quotient),whole,quotient_exact)))

# Cell index of a step of nonzero stride lies within its reach: index times the
# stride is at most the reach.
def reach_quotient(+count: U32,+reach: U32,+stride: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool},
  +nonzero: {U32.is_zero(stride) == False{} : Bool},+reach_room: {Nat.is_lt(U32.to_nat(reach),U32.to_nat(capacity)) == True{} : Bool},
  +index: Nat,+less: {Nat.is_lt(index,U32.to_nat(U32.min(count,((reach / U32.max(stride,1) : U32) + 1 : U32)))) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(index,U32.to_nat(stride)),U32.to_nat(reach)) == True{} : Bool}:
  +positive = Arithmetic.word_nonzero_positive(stride,nonzero)
  +whole = ((reach / U32.max(stride,1) : U32) + 1 : U32)
  +below_whole = Bounds.nat_trans(index,U32.to_nat(U32.min(count,whole)),U32.to_nat(whole),less,Bounds.as_nat_le(U32.min(count,whole),whole,Extrema.minimum_right(count,whole)))
  +below_next = nat_lt_moved(index,1n+Nat.div(U32.to_nat(reach),U32.to_nat(stride)),U32.to_nat(whole),
    Equal.sym(Nat,U32.to_nat(whole),1n+Nat.div(U32.to_nat(reach),U32.to_nat(stride)),reach_whole(reach,stride,capacity,capacity_ok,word,word_ok,stride_room,nonzero,reach_room)),
    below_whole)
  NatDivision.lt_succ_le(Nat.mul(index,U32.to_nat(stride)),U32.to_nat(reach),
    Equal.trans(Bool,Nat.is_lt(Nat.mul(index,U32.to_nat(stride)),1n+U32.to_nat(reach)),Nat.is_lt(index,Ceiling.ceiling(1n+U32.to_nat(reach),U32.to_nat(stride))),True{},
      Equal.sym(Bool,Nat.is_lt(index,Ceiling.ceiling(1n+U32.to_nat(reach),U32.to_nat(stride))),Nat.is_lt(Nat.mul(index,U32.to_nat(stride)),1n+U32.to_nat(reach)),
        Ceiling.upper_comparison(1n+U32.to_nat(reach),U32.to_nat(stride),index,positive)),
      nat_lt_moved(index,Ceiling.ceiling(1n+U32.to_nat(reach),U32.to_nat(stride)),1n+Nat.div(U32.to_nat(reach),U32.to_nat(stride)),
        Ceiling.successor_formula(U32.to_nat(reach),U32.to_nat(stride),positive),below_next)))

# A piece that holds offset: offset lies in it and it ends within the capacity.
def offset_after(+offset: U32,+first: U32,+span: U32,+capacity: U32,+holds: Holds<first,span,offset,capacity>) ->
  {Nat.is_le(U32.to_nat(first),U32.to_nat(offset)) == True{} : Bool}:
  Holds{+after,+before,+room} = holds
  Bounds.as_nat_le(first,offset,after)

def offset_below(+offset: U32,+first: U32,+span: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+holds: Holds<first,span,offset,capacity>) ->
  {Nat.is_lt(U32.to_nat(offset),Nat.add(U32.to_nat(first),U32.to_nat(span))) == True{} : Bool}:
  Holds{+after,+before,+room} = holds
  nat_lt_moved(U32.to_nat(offset),Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat((first + span : U32)),
    Equal.sym(Nat,U32.to_nat((first + span : U32)),Nat.add(U32.to_nat(first),U32.to_nat(span)),piece_end(first,span,capacity,capacity_ok,room)),
    Bounds.as_nat_lt(offset,(first + span : U32),before))

def held_room(+offset: U32,+first: U32,+span: U32,+capacity: U32,+holds: Holds<first,span,offset,capacity>) ->
  {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat(capacity)) == True{} : Bool}:
  Holds{+after,+before,+room} = holds
  room

# The cell of offset within its piece.
def local_exact(+offset: U32,+first: U32,+span: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+holds: Holds<first,span,offset,capacity>) ->
  {U32.to_nat((offset - first : U32)) == Nat.sub(U32.to_nat(offset),U32.to_nat(first)) : Nat}:
  Arithmetic.subtract(offset,first,capacity,capacity_ok,offset_after(offset,first,span,capacity,holds),
    Order.transitive(Nat.sub(U32.to_nat(offset),U32.to_nat(first)),U32.to_nat(offset),U32.to_nat(capacity),Order.sub_le(U32.to_nat(offset),U32.to_nat(first)),
      Order.transitive(U32.to_nat(offset),Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat(capacity),
        Bounds.strict_le(U32.to_nat(offset),Nat.add(U32.to_nat(first),U32.to_nat(span)),offset_below(offset,first,span,capacity,capacity_ok,holds)),
        held_room(offset,first,span,capacity,holds))))

def local_place(+offset: U32,+first: U32,+span: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+holds: Holds<first,span,offset,capacity>) ->
  {Nat.is_lt(U32.to_nat((offset - first : U32)),U32.to_nat(span)) == True{} : Bool}:
  nat_lt_left(U32.to_nat((offset - first : U32)),Nat.sub(U32.to_nat(offset),U32.to_nat(first)),U32.to_nat(span),local_exact(offset,first,span,capacity,capacity_ok,holds),
    Order.sub_lt(U32.to_nat(offset),U32.to_nat(first),U32.to_nat(span),offset_after(offset,first,span,capacity,holds),offset_below(offset,first,span,capacity,capacity_ok,holds)))

# Forward, the reach and the cell of offset make up the rest of the piece.
def forward_reach(+offset: U32,+first: U32,+span: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+holds: Holds<first,span,offset,capacity>) ->
  {1n+U32.to_nat((((first + span : U32) - offset : U32) - 1 : U32)) == Nat.sub(Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat(offset)) : Nat}:
  Holds{+after,+before,+room} = holds
  +end = (first + span : U32)
  +left = (end - offset : U32)
  +total = Nat.add(U32.to_nat(first),U32.to_nat(span))
  +rest = Nat.sub(total,U32.to_nat(offset))
  +exact_end = piece_end(first,span,capacity,capacity_ok,room)
  +rest_small = Order.transitive(rest,total,U32.to_nat(capacity),Order.sub_le(total,U32.to_nat(offset)),room)
  +end_rest = Equal.cong(Nat,Nat,number => Nat.sub(number,U32.to_nat(offset)),U32.to_nat(end),total,exact_end)
  +exact_left = Equal.trans(Nat,U32.to_nat(left),Nat.sub(U32.to_nat(end),U32.to_nat(offset)),rest,
    Arithmetic.subtract(end,offset,capacity,capacity_ok,Bounds.strict_le(U32.to_nat(offset),U32.to_nat(end),Bounds.as_nat_lt(offset,end,before)),
      nat_le_moved(Nat.sub(U32.to_nat(end),U32.to_nat(offset)),rest,U32.to_nat(capacity),end_rest,rest_small)),
    end_rest)
  +positive = Order.positive_difference(total,U32.to_nat(offset),offset_below(offset,first,span,capacity,capacity_ok,Holds{after,before,room}))
  +left_small = nat_le_moved(U32.to_nat(left),rest,U32.to_nat(capacity),exact_left,rest_small)
  +exact_reach = Arithmetic.subtract(left,1,capacity,capacity_ok,
    nat_le_right_moved(1n,U32.to_nat(left),rest,exact_left,Order.strict_follow(0n,rest,positive)),
    Order.transitive(Nat.sub(U32.to_nat(left),1n),U32.to_nat(left),U32.to_nat(capacity),Order.sub_le(U32.to_nat(left),1n),left_small))
  Equal.trans(Nat,1n+U32.to_nat((left - 1 : U32)),1n+Nat.sub(rest,1n),rest,
    Equal.cong(Nat,Nat,number => 1n+number,U32.to_nat((left - 1 : U32)),Nat.sub(rest,1n),
      Equal.trans(Nat,U32.to_nat((left - 1 : U32)),Nat.sub(U32.to_nat(left),1n),Nat.sub(rest,1n),exact_reach,
        Equal.cong(Nat,Nat,number => Nat.sub(number,1n),U32.to_nat(left),rest,exact_left))),
    Order.positive_has_last(rest,positive))

# The reach of a step lies below the capacity.
def reach_room(+back: Bool,+offset: U32,+first: U32,+span: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +holds: Holds<first,span,offset,capacity>) ->
  {Nat.is_lt(U32.to_nat(step_reach(back,offset,first,span)),U32.to_nat(capacity)) == True{} : Bool}:
  match back:
    case True{}:
      Bounds.nat_trans(U32.to_nat((offset - first : U32)),U32.to_nat(span),U32.to_nat(capacity),local_place(offset,first,span,capacity,capacity_ok,holds),
        Order.transitive(U32.to_nat(span),Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat(capacity),Order.right_le(U32.to_nat(first),U32.to_nat(span)),
          held_room(offset,first,span,capacity,holds)))
    case False{}:
      +reach = U32.to_nat((((first + span : U32) - offset : U32) - 1 : U32))
      +total = Nat.add(U32.to_nat(first),U32.to_nat(span))
      Bounds.nat_trans(reach,1n+reach,U32.to_nat(capacity),Order.lt_succ(reach,reach,Order.reflexive(reach)),
        nat_le_moved(1n+reach,Nat.sub(total,U32.to_nat(offset)),U32.to_nat(capacity),forward_reach(offset,first,span,capacity,capacity_ok,holds),
          Order.transitive(Nat.sub(total,U32.to_nat(offset)),total,U32.to_nat(capacity),Order.sub_le(total,U32.to_nat(offset)),held_room(offset,first,span,capacity,holds))))

# A product below the rest of the piece after offset reaches a cell of the
# piece.
def forward_place(+first: Nat,+span: Nat,+offset: Nat,+product: Nat,+ordered: {Nat.is_le(first,offset) == True{} : Bool},
  +below: {Nat.is_lt(product,Nat.sub(Nat.add(first,span),offset)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(product,Nat.sub(offset,first)),span) == True{} : Bool}:
  +local = Nat.sub(offset,first)
  %Order.commutative(local,product) : {Nat.is_lt(_,span) == True{} : Bool}
  Order.cancel_offset_lt(first,Nat.add(local,product),span,
    %Addition.associative(first,local,product) : {Nat.is_lt(_,Nat.add(first,span)) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(first,local),offset,Order.restore_shift(first,offset,ordered)) : {Nat.is_lt(Nat.add(_,product),Nat.add(first,span)) == True{} : Bool}
    Equal.trans(Bool,Nat.is_lt(Nat.add(offset,product),Nat.add(first,span)),Nat.is_lt(product,Nat.sub(Nat.add(first,span),offset)),True{},
      Equal.sym(Bool,Nat.is_lt(product,Nat.sub(Nat.add(first,span),offset)),Nat.is_lt(Nat.add(offset,product),Nat.add(first,span)),
        Order.subtraction_lt(Nat.add(first,span),offset,product)),
      below))

# At stride zero a step reads the cell of offset.
def zero_place(+offset: U32,+stride: U32,+back: Bool,+first: U32,+span: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +holds: Holds<first,span,offset,capacity>,+index: Nat,+zero: {U32.is_zero(stride) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(((offset - first : U32) + U32.from_nat(index) * Move.directed(back,stride) : U32)),U32.to_nat(span)) == True{} : Bool}:
  +local = (offset - first : U32)
  %Equal.sym(U32,stride,0,Arithmetic.word_zero(stride,zero)) :
    {Nat.is_lt(U32.to_nat((local + U32.from_nat(index) * Move.directed(back,_) : U32)),U32.to_nat(span)) == True{} : Bool}
  %Equal.sym(U32,Move.directed(back,0),0,directed_zero(back)) : {Nat.is_lt(U32.to_nat((local + U32.from_nat(index) * _ : U32)),U32.to_nat(span)) == True{} : Bool}
  %Equal.sym(U32,(U32.from_nat(index) * 0 : U32),0,Arithmetic.multiply_zero_right(U32.from_nat(index))) :
    {Nat.is_lt(U32.to_nat((local + _ : U32)),U32.to_nat(span)) == True{} : Bool}
  %Equal.sym(U32,(local + 0 : U32),local,Arithmetic.add_zero(local)) : {Nat.is_lt(U32.to_nat(_),U32.to_nat(span)) == True{} : Bool}
  local_place(offset,first,span,capacity,capacity_ok,holds)

# Backward, cell index of a step lies index times the stride before offset.
def backward_place(+count: U32,+offset: U32,+stride: U32,+first: U32,+span: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool},
  +holds: Holds<first,span,offset,capacity>,+nonzero: {U32.is_zero(stride) == False{} : Bool},+index: Nat,
  +less: {Nat.is_lt(index,U32.to_nat(U32.min(count,(((offset - first : U32) / U32.max(stride,1) : U32) + 1 : U32)))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(((offset - first : U32) + U32.from_nat(index) * Move.directed(True{},stride) : U32)),U32.to_nat(span)) == True{} : Bool}:
  +local = (offset - first : U32)
  +product = (U32.from_nat(index) * stride : U32)
  +scaled = Nat.mul(index,U32.to_nat(stride))
  +room = reach_room(True{},offset,first,span,capacity,capacity_ok,holds)
  +bound = reach_quotient(count,local,stride,capacity,capacity_ok,word,word_ok,stride_room,nonzero,room,index,less)
  +local_small = Bounds.strict_le(U32.to_nat(local),U32.to_nat(capacity),room)
  +product_exact = Equal.trans(Nat,U32.to_nat(product),U32.to_nat(U32.from_nat(scaled)),scaled,
    Equal.cong(U32,Nat,U32.to_nat,product,U32.from_nat(scaled),
      %Arithmetic.native_roundtrip(stride) : {(U32.from_nat(index) * _ : U32) == U32.from_nat(scaled) : U32}
      Multiplication.native_encoded_multiply(index,U32.to_nat(stride))),
    Bounds.roundtrip(scaled,capacity,capacity_ok,Order.transitive(scaled,U32.to_nat(local),U32.to_nat(capacity),bound,local_small)))
  +difference = Arithmetic.subtract(local,product,capacity,capacity_ok,nat_le_moved(U32.to_nat(product),scaled,U32.to_nat(local),product_exact,bound),
    Order.transitive(Nat.sub(U32.to_nat(local),U32.to_nat(product)),U32.to_nat(local),U32.to_nat(capacity),Order.sub_le(U32.to_nat(local),U32.to_nat(product)),local_small))
  %Equal.sym(U32,(local + U32.from_nat(index) * (0 - stride : U32) : U32),(local - product : U32),
    Equal.trans(U32,(local + U32.from_nat(index) * (0 - stride : U32) : U32),(U32.from_nat(index) * (0 - stride : U32) + local : U32),(local - product : U32),
      Arithmetic.add_commutative(local,(U32.from_nat(index) * (0 - stride : U32) : U32)),Arithmetic.negative_step(U32.from_nat(index),stride,local))) :
    {Nat.is_lt(U32.to_nat(_),U32.to_nat(span)) == True{} : Bool}
  Order.le_lt_trans(U32.to_nat((local - product : U32)),U32.to_nat(local),U32.to_nat(span),
    nat_le_moved(U32.to_nat((local - product : U32)),Nat.sub(U32.to_nat(local),U32.to_nat(product)),U32.to_nat(local),difference,Order.sub_le(U32.to_nat(local),U32.to_nat(product))),
    local_place(offset,first,span,capacity,capacity_ok,holds))

# Forward, cell index of a step lies index times the stride after offset.
def forward_step_place(+count: U32,+offset: U32,+stride: U32,+first: U32,+span: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool},
  +holds: Holds<first,span,offset,capacity>,+nonzero: {U32.is_zero(stride) == False{} : Bool},+index: Nat,
  +less: {Nat.is_lt(index,U32.to_nat(U32.min(count,(((((first + span : U32) - offset : U32) - 1 : U32) / U32.max(stride,1) : U32) + 1 : U32)))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(((offset - first : U32) + U32.from_nat(index) * Move.directed(False{},stride) : U32)),U32.to_nat(span)) == True{} : Bool}:
  +local = (offset - first : U32)
  +reach = (((first + span : U32) - offset : U32) - 1 : U32)
  +rest = Nat.sub(Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat(offset))
  +scaled = Nat.mul(index,U32.to_nat(stride))
  +bound = reach_quotient(count,reach,stride,capacity,capacity_ok,word,word_ok,stride_room,nonzero,reach_room(False{},offset,first,span,capacity,capacity_ok,holds),index,less)
  +scaled_below = nat_lt_moved(scaled,rest,1n+U32.to_nat(reach),Equal.sym(Nat,1n+U32.to_nat(reach),rest,forward_reach(offset,first,span,capacity,capacity_ok,holds)),
    Order.lt_succ(scaled,U32.to_nat(reach),bound))
  +place = forward_place(U32.to_nat(first),U32.to_nat(span),U32.to_nat(offset),scaled,offset_after(offset,first,span,capacity,holds),scaled_below)
  +sum = Nat.add(scaled,U32.to_nat(local))
  +sum_below = nat_lt_left(sum,Nat.add(scaled,Nat.sub(U32.to_nat(offset),U32.to_nat(first))),U32.to_nat(span),
    Equal.cong(Nat,Nat,number => Nat.add(scaled,number),U32.to_nat(local),Nat.sub(U32.to_nat(offset),U32.to_nat(first)),local_exact(offset,first,span,capacity,capacity_ok,holds)),
    place)
  +sum_small = Order.transitive(sum,U32.to_nat(span),U32.to_nat(capacity),Bounds.strict_le(sum,U32.to_nat(span),sum_below),
    Order.transitive(U32.to_nat(span),Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat(capacity),Order.right_le(U32.to_nat(first),U32.to_nat(span)),
      held_room(offset,first,span,capacity,holds)))
  %word_address(local,stride,index) : {Nat.is_lt(U32.to_nat(_),U32.to_nat(span)) == True{} : Bool}
  nat_lt_left(U32.to_nat(U32.from_nat(sum)),sum,U32.to_nat(span),Bounds.roundtrip(sum,capacity,capacity_ok,sum_small),sum_below)

def moving_place(+count: U32,+offset: U32,+stride: U32,+back: Bool,+first: U32,+span: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool},
  +holds: Holds<first,span,offset,capacity>,+nonzero: {U32.is_zero(stride) == False{} : Bool},+index: Nat,
  +less: {Nat.is_lt(index,U32.to_nat(U32.min(count,((step_reach(back,offset,first,span) / U32.max(stride,1) : U32) + 1 : U32)))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(((offset - first : U32) + U32.from_nat(index) * Move.directed(back,stride) : U32)),U32.to_nat(span)) == True{} : Bool}:
  match back:
    case True{}: backward_place(count,offset,stride,first,span,capacity,capacity_ok,word,word_ok,stride_room,holds,nonzero,index,less)
    case False{}: forward_step_place(count,offset,stride,first,span,capacity,capacity_ok,word,word_ok,stride_room,holds,nonzero,index,less)

# Every cell a step reads from a piece that holds offset lies in the piece.
def step_place(+count: U32,+offset: U32,+stride: U32,+back: Bool,+first: U32,+span: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool},
  +holds: Holds<first,span,offset,capacity>,+index: Nat,+less: {Nat.is_lt(index,U32.to_nat(Move.inside(count,offset,stride,back,first,span))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(((offset - first : U32) + U32.from_nat(index) * Move.directed(back,stride) : U32)),U32.to_nat(span)) == True{} : Bool}:
  Holds{+after,+before,+room} = holds
  +whole = ((step_reach(back,offset,first,span) / U32.max(stride,1) : U32) + 1 : U32)
  +inner = Bool.pick(U32,U32.is_zero(stride),count,U32.min(count,whole))
  +inner_less = nat_lt_moved(index,U32.to_nat(inner),U32.to_nat(Move.inside(count,offset,stride,back,first,span)),
    Equal.cong(U32,Nat,U32.to_nat,inner,Move.inside(count,offset,stride,back,first,span),
      Equal.sym(U32,Move.inside(count,offset,stride,back,first,span),inner,held_cells(count,offset,stride,back,first,span,after,before))),
    less)
  by_cases({Nat.is_lt(U32.to_nat(((offset - first : U32) + U32.from_nat(index) * Move.directed(back,stride) : U32)),U32.to_nat(span)) == True{} : Bool},U32.is_zero(stride),
    +zero => zero_place(offset,stride,back,first,span,capacity,capacity_ok,Holds{after,before,room},index,zero),
    +nonzero => moving_place(count,offset,stride,back,first,span,capacity,capacity_ok,word,word_ok,stride_room,Holds{after,before,room},nonzero,index,
      nat_lt_moved(index,U32.to_nat(U32.min(count,whole)),U32.to_nat(inner),
        Equal.sym(Nat,U32.to_nat(inner),U32.to_nat(U32.min(count,whole)),
          Equal.cong(Bool,Nat,test => U32.to_nat(Bool.pick(U32,test,count,U32.min(count,whole))),U32.is_zero(stride),False{},nonzero)),
        inner_less)))

# A step copies at most count cells, and at least one from a piece that holds
# offset when count is not zero.
def picked_most(choice: Bool,+count: U32,+inner: U32,+most: {Nat.is_le(U32.to_nat(inner),U32.to_nat(count)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Bool.pick(U32,choice,inner,0)),U32.to_nat(count)) == True{} : Bool}:
  match choice:
    case True{}: most
    case False{}: Order.zero_le(U32.to_nat(count))

def zero_most(choice: Bool,+count: U32,+whole: U32) -> {Nat.is_le(U32.to_nat(Bool.pick(U32,choice,count,U32.min(count,whole))),U32.to_nat(count)) == True{} : Bool}:
  match choice:
    case True{}: Order.reflexive(U32.to_nat(count))
    case False{}: Extrema.minimum_below(count,whole)

def cells_most(+count: U32,+offset: U32,+stride: U32,+back: Bool,+first: U32,+span: U32) ->
  {Nat.is_le(U32.to_nat(Move.inside(count,offset,stride,back,first,span)),U32.to_nat(count)) == True{} : Bool}:
  +whole = ((step_reach(back,offset,first,span) / U32.max(stride,1) : U32) + 1 : U32)
  picked_most(U32.is_le(first,offset) && U32.is_lt(offset,(first + span : U32)),count,Bool.pick(U32,U32.is_zero(stride),count,U32.min(count,whole)),
    zero_most(U32.is_zero(stride),count,whole))

def step_cells_most(~Element: Data,+holding: Holding<Element>,+count: U32,+offset: U32,+stride: U32,+back: Bool) ->
  {Nat.is_le(U32.to_nat(step_cells(~Element,holding,count,offset,stride,back)),U32.to_nat(count)) == True{} : Bool}:
  match holding:
    case Holding{+behind,Held{+buffer,+first,+span,+below}}: cells_most(count,offset,stride,back,first,span)
    case Holding{+behind,Bottom{}}: Order.zero_le(U32.to_nat(count))

def cells_some(+count: U32,+offset: U32,+stride: U32,+back: Bool,+first: U32,+span: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool},
  +holds: Holds<first,span,offset,capacity>,+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat(Move.inside(count,offset,stride,back,first,span))) == True{} : Bool}:
  Holds{+after,+before,+room} = holds
  +reach = step_reach(back,offset,first,span)
  +whole = ((reach / U32.max(stride,1) : U32) + 1 : U32)
  +inner = Bool.pick(U32,U32.is_zero(stride),count,U32.min(count,whole))
  nat_lt_moved(0n,U32.to_nat(Move.inside(count,offset,stride,back,first,span)),U32.to_nat(inner),
    Equal.cong(U32,Nat,U32.to_nat,Move.inside(count,offset,stride,back,first,span),inner,held_cells(count,offset,stride,back,first,span,after,before)),
    by_cases({Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},U32.is_zero(stride),
      +zero =>
        %Equal.sym(Bool,U32.is_zero(stride),True{},zero) : {Nat.is_lt(0n,U32.to_nat(Bool.pick(U32,_,count,U32.min(count,whole)))) == True{} : Bool}
        positive,
      +nonzero =>
        %Equal.sym(Bool,U32.is_zero(stride),False{},nonzero) : {Nat.is_lt(0n,U32.to_nat(Bool.pick(U32,_,count,U32.min(count,whole)))) == True{} : Bool}
        nat_lt_moved(0n,U32.to_nat(U32.min(count,whole)),Nat.min(U32.to_nat(count),U32.to_nat(whole)),Extrema.minimum(count,whole),
          NatExtrema.minimum_strict_lower(U32.to_nat(count),U32.to_nat(whole),0n,positive,
            nat_lt_moved(0n,U32.to_nat(whole),1n+Nat.div(U32.to_nat(reach),U32.to_nat(stride)),
              reach_whole(reach,stride,capacity,capacity_ok,word,word_ok,stride_room,nonzero,reach_room(back,offset,first,span,capacity,capacity_ok,Holds{after,before,room})),
              {==})))))

# A step from the piece that holds offset writes target cell position + k with
# the value of cell offset + k times its step, for every k below its cells.
def written_step(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+holding: Holding<Element>,+count: U32,+offset: U32,
  +stride: U32,+back: Bool,+position: U32,+depth: Nat,+tree: Tree.Tree<Element>,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool},
  +current: Current(~Element,holding,offset,capacity),+values: HoldingValues(~Element,~Context,~value,context,holding),+balanced: Shape.Balanced(Element,depth,tree),
  +depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},+target_ok: {U32.is_le(Tree.capacity(Element,tree),2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(position),U32.to_nat(count)),U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool},
  +index: Nat,+less: {Nat.is_lt(index,U32.to_nat(step_cells(~Element,holding,count,offset,stride,back))) == True{} : Bool}) ->
  {Tree.read(Element,step_tree(~Element,holding,count,offset,stride,back,position,tree),U32.from_nat(Nat.add(U32.to_nat(position),index))) ==
    value(context,(offset + U32.from_nat(index) * Move.directed(back,stride) : U32)) : Element}:
  match holding:
    case Holding{+behind,Held{+buffer,+first,+span,+below}}:
      Holds{+after,+before,+piece_room} = current
      Both{earlier,Both{+piece,later}} = values
      +cells = Move.inside(count,offset,stride,back,first,span)
      +local = (offset - first : U32)
      +direction = Move.directed(back,stride)
      +input = Buffer.tree(~Element,buffer)
      +target = Tree.capacity(Element,tree)
      +query = U32.from_nat(Nat.add(U32.to_nat(position),index))
      +address = (local + U32.from_nat(index) * direction : U32)
      +source = (offset + U32.from_nat(index) * direction : U32)
      +most = cells_most(count,offset,stride,back,first,span)
      +cells_room = Order.transitive(Nat.add(U32.to_nat(position),U32.to_nat(cells)),Nat.add(U32.to_nat(position),U32.to_nat(count)),U32.to_nat(target),
        le_shift(U32.to_nat(position),U32.to_nat(cells),U32.to_nat(count),most),room)
      +line = step_line(~Element,cells,local,stride,back,U32.is_eq(stride,1) && Bool.not(back),position,input,target,target_ok,
        Order.transitive(U32.to_nat(cells),Nat.add(U32.to_nat(position),U32.to_nat(cells)),U32.to_nat(target),Order.right_le(U32.to_nat(position),U32.to_nat(cells)),cells_room),{==})
      +read = Consecutive.read(~Element,~CopyValues.Sampled<Element>,~CopyValues.sampled(~Element),U32.to_nat(cells),U32.to_nat(position),
        CopyValues.Sampled{Views.View{[Views.Axis{cells,direction}],local},input},tree,depth,index,balanced,depth_ok,target_ok,cells_room,less)
      +place = step_place(count,offset,stride,back,first,span,capacity,capacity_ok,word,word_ok,stride_room,Holds{after,before,piece_room},index,less)
      +cell = at_value(~Element,~Context,~value,context,first,input,Buffer.length(~Element,buffer),U32.to_nat(address),
        Content.leaf_cell(~Element,~Context,~value,context,first,span,input,Buffer.length(~Element,buffer),U32.to_nat(address),piece,place))
      +start = Equal.trans(U32,(first + local : U32),((first + offset : U32) - first : U32),offset,
        Equal.sym(U32,((first + offset : U32) - first : U32),(first + local : U32),Arithmetic.subtract_from_sum(first,offset,first)),Arithmetic.subtract_first(first,offset))
      +moved = Equal.trans(U32,U32.from_nat(Nat.add(U32.to_nat(first),U32.to_nat(address))),(first + address : U32),source,
        Equal.sym(U32,(first + address : U32),U32.from_nat(Nat.add(U32.to_nat(first),U32.to_nat(address))),Arithmetic.native_add(first,address)),
        Equal.trans(U32,(first + address : U32),((first + local : U32) + U32.from_nat(index) * direction : U32),source,
          Equal.sym(U32,((first + local : U32) + U32.from_nat(index) * direction : U32),(first + address : U32),
            Arithmetic.add_associative(first,local,(U32.from_nat(index) * direction : U32))),
          Equal.cong(U32,U32,word => (word + U32.from_nat(index) * direction : U32),(first + local : U32),offset,start)))
      %Equal.sym(List<&2,Writes.Assignment<Element>>,
        CopyValues.copy_writes(~Element,U32.is_eq(stride,1) && Bool.not(back),U32.to_nat(cells),Views.View{[Views.Axis{cells,direction}],local},position,input),
        Consecutive.writes(~Element,~CopyValues.Sampled<Element>,~CopyValues.sampled(~Element),U32.to_nat(cells),U32.to_nat(position),0n,
          CopyValues.Sampled{Views.View{[Views.Axis{cells,direction}],local},input}),line) :
        {Tree.read(Element,Writes.pure(Element,_,tree),query) == value(context,source) : Element}
      Equal.trans(Element,
        Tree.read(Element,Writes.pure(Element,Consecutive.writes(~Element,~CopyValues.Sampled<Element>,~CopyValues.sampled(~Element),U32.to_nat(cells),U32.to_nat(position),0n,
          CopyValues.Sampled{Views.View{[Views.Axis{cells,direction}],local},input}),tree),query),
        Tree.read(Element,input,CopyValues.address(Views.View{[Views.Axis{cells,direction}],local},index)),value(context,source),read,
        Equal.trans(Element,Tree.read(Element,input,CopyValues.address(Views.View{[Views.Axis{cells,direction}],local},index)),Tree.read(Element,input,U32.from_nat(U32.to_nat(address))),
          value(context,source),
          Equal.cong(U32,Element,word => Tree.read(Element,input,word),CopyValues.address(Views.View{[Views.Axis{cells,direction}],local},index),U32.from_nat(U32.to_nat(address)),
            Equal.trans(U32,CopyValues.address(Views.View{[Views.Axis{cells,direction}],local},index),address,U32.from_nat(U32.to_nat(address)),
              single_address(cells,direction,local,index,less),Equal.sym(U32,U32.from_nat(U32.to_nat(address)),address,Arithmetic.native_roundtrip(address)))),
          Equal.trans(Element,Tree.read(Element,input,U32.from_nat(U32.to_nat(address))),value(context,U32.from_nat(Nat.add(U32.to_nat(first),U32.to_nat(address)))),
            value(context,source),cell,
            Equal.cong(U32,Element,word => value(context,word),U32.from_nat(Nat.add(U32.to_nat(first),U32.to_nat(address))),source,moved))))
    case Holding{behind,Bottom{}}:
      Empty.absurd({Tree.read(Element,step_tree(~Element,Holding{behind,Bottom{}},count,offset,stride,back,position,tree),U32.from_nat(Nat.add(U32.to_nat(position),index))) ==
        value(context,(offset + U32.from_nat(index) * Move.directed(back,stride) : U32)) : Element},current)

def bool_moved(~Value: Data,~test: Value -> Bool,+from: Value,+to: Value,+equal: {from == to : Value},+truth: {test(to) == True{} : Bool}) -> {test(from) == True{} : Bool}:
  Equal.trans(Bool,test(from),test(to),True{},Equal.cong(Value,Bool,test,from,to,equal),truth)

# Consecutive writes leave the cells outside them.
def kept_selected(~Element: Data,+cells: Nat,+position: Nat,+part: CopyValues.Sampled<Element>,+previous: Element,+query: U32,+target: U32,
  +room: {Nat.is_le(Nat.add(position,cells),U32.to_nat(target)) == True{} : Bool},+query_inside: {U32.is_lt(query,target) == True{} : Bool},
  +outside: Interval.Outside<position,cells,U32.to_nat(query)>) ->
  {Readback.selected(~Element,Consecutive.writes(~Element,~CopyValues.Sampled<Element>,~CopyValues.sampled(~Element),cells,position,0n,part),U32.from_nat(U32.to_nat(query)),previous) ==
    previous : Element}:
  match outside:
    case Interval.Before{+bound}:
      Consecutive.before(~Element,~CopyValues.Sampled<Element>,~CopyValues.sampled(~Element),cells,position,0n,part,previous,U32.to_nat(query),target,room,
        nat_lt_moved(U32.to_nat(query),Nat.add(position,0n),position,Addition.zero_right(position),bound))
    case Interval.After{+bound}:
      Consecutive.after(~Element,~CopyValues.Sampled<Element>,~CopyValues.sampled(~Element),cells,position,0n,part,previous,U32.to_nat(query),target,bound,
        Bounds.as_nat_lt(query,target,query_inside))

# A step leaves the target cells outside the cells it writes.
def kept_step(~Element: Data,+holding: Holding<Element>,+count: U32,+offset: U32,+stride: U32,+back: Bool,+position: U32,+depth: Nat,+tree: Tree.Tree<Element>,
  +balanced: Shape.Balanced(Element,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},+target_ok: {U32.is_le(Tree.capacity(Element,tree),2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(position),U32.to_nat(count)),U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool},
  +query: U32,+query_inside: {U32.is_lt(query,Tree.capacity(Element,tree)) == True{} : Bool},
  +outside: Interval.Outside<U32.to_nat(position),U32.to_nat(step_cells(~Element,holding,count,offset,stride,back)),U32.to_nat(query)>) ->
  {Tree.read(Element,step_tree(~Element,holding,count,offset,stride,back,position,tree),query) == Tree.read(Element,tree,query) : Element}:
  match holding:
    case Holding{+behind,Held{+buffer,+first,+span,+below}}:
      +cells = Move.inside(count,offset,stride,back,first,span)
      +local = (offset - first : U32)
      +direction = Move.directed(back,stride)
      +input = Buffer.tree(~Element,buffer)
      +target = Tree.capacity(Element,tree)
      +part = {CopyValues.Sampled{Views.View{[Views.Axis{cells,direction}],local},input} : CopyValues.Sampled<Element>}
      +writes = Consecutive.writes(~Element,~CopyValues.Sampled<Element>,~CopyValues.sampled(~Element),U32.to_nat(cells),U32.to_nat(position),0n,part)
      +most = cells_most(count,offset,stride,back,first,span)
      +cells_room = Order.transitive(Nat.add(U32.to_nat(position),U32.to_nat(cells)),Nat.add(U32.to_nat(position),U32.to_nat(count)),U32.to_nat(target),
        le_shift(U32.to_nat(position),U32.to_nat(cells),U32.to_nat(count),most),room)
      +line = step_line(~Element,cells,local,stride,back,U32.is_eq(stride,1) && Bool.not(back),position,input,target,target_ok,
        Order.transitive(U32.to_nat(cells),Nat.add(U32.to_nat(position),U32.to_nat(cells)),U32.to_nat(target),Order.right_le(U32.to_nat(position),U32.to_nat(cells)),cells_room),{==})
      +probe = U32.from_nat(U32.to_nat(query))
      %Equal.sym(List<&2,Writes.Assignment<Element>>,
        CopyValues.copy_writes(~Element,U32.is_eq(stride,1) && Bool.not(back),U32.to_nat(cells),Views.View{[Views.Axis{cells,direction}],local},position,input),writes,line) :
        {Tree.read(Element,Writes.pure(Element,_,tree),query) == Tree.read(Element,tree,query) : Element}
      %Arithmetic.native_roundtrip(query) : {Tree.read(Element,Writes.pure(Element,writes,tree),_) == Tree.read(Element,tree,_) : Element}
      Equal.trans(Element,Tree.read(Element,Writes.pure(Element,writes,tree),probe),Readback.selected(~Element,writes,probe,Tree.read(Element,tree,probe)),Tree.read(Element,tree,probe),
        Readback.program(~Element,writes,depth,tree,probe,balanced,depth_ok,
          bool_moved(~U32,~word => U32.is_lt(word,target),probe,query,Arithmetic.native_roundtrip(query),query_inside),
          Consecutive.inside(~Element,~CopyValues.Sampled<Element>,~CopyValues.sampled(~Element),U32.to_nat(cells),U32.to_nat(position),0n,part,target,target_ok,cells_room)),
        kept_selected(~Element,U32.to_nat(cells),U32.to_nat(position),part,Tree.read(Element,tree,probe),query,target,cells_room,query_inside,outside))
    case Holding{+behind,Bottom{}}: {==}

# ---- A whole run ----

def step_copies(~Element: Data,+holding: Holding<Element>,+count: U32,+offset: U32,+stride: U32,+back: Bool,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},
  +stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool},+current: Current(~Element,holding,offset,capacity),
  +positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}) -> Copies<step_cells(~Element,holding,count,offset,stride,back),count>:
  match holding:
    case Holding{+behind,Held{+buffer,+first,+span,+below}}:
      Copies{cells_some(count,offset,stride,back,first,span,capacity,capacity_ok,word,word_ok,stride_room,current,positive),cells_most(count,offset,stride,back,first,span)}
    case Holding{behind,Bottom{}}: Empty.absurd(Copies<step_cells(~Element,Holding{behind,Bottom{}},count,offset,stride,back),count>,current)

def copies_some(-cells: U32,-count: U32,+copies: Copies<cells,count>) -> {Nat.is_lt(0n,U32.to_nat(cells)) == True{} : Bool}:
  Copies{some,most} = copies
  some

def copies_most(-cells: U32,-count: U32,+copies: Copies<cells,count>) -> {Nat.is_le(U32.to_nat(cells),U32.to_nat(count)) == True{} : Bool}:
  Copies{some,most} = copies
  most

def left_count(+count: U32,+cells: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +most: {Nat.is_le(U32.to_nat(cells),U32.to_nat(count)) == True{} : Bool},+small: {Nat.is_le(U32.to_nat(count),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((count - cells : U32)) == Nat.sub(U32.to_nat(count),U32.to_nat(cells)) : Nat}:
  Arithmetic.subtract(count,cells,capacity,capacity_ok,most,Order.transitive(Nat.sub(U32.to_nat(count),U32.to_nat(cells)),U32.to_nat(count),U32.to_nat(capacity),
    Order.sub_le(U32.to_nat(count),U32.to_nat(cells)),small))

def rejoin(+base: Nat,+cells: Nat,+count: Nat,+most: {Nat.is_le(cells,count) == True{} : Bool}) -> {Nat.add(Nat.add(base,cells),Nat.sub(count,cells)) == Nat.add(base,count) : Nat}:
  Equal.trans(Nat,Nat.add(Nat.add(base,cells),Nat.sub(count,cells)),Nat.add(base,Nat.add(cells,Nat.sub(count,cells))),Nat.add(base,count),
    Addition.associative(base,cells,Nat.sub(count,cells)),
    Equal.cong(Nat,Nat,value => Nat.add(base,value),Nat.add(cells,Nat.sub(count,cells)),count,Order.restore_shift(cells,count,most)))

def sub_pred(+count: Nat,+cells: Nat,+rest: Nat,+some: {Nat.is_lt(0n,cells) == True{} : Bool},+bound: {Nat.is_le(count,1n+rest) == True{} : Bool}) ->
  {Nat.is_le(Nat.sub(count,cells),rest) == True{} : Bool}:
  match count cells:
    case 0n cells: Order.zero_le(rest)
    case 1n+count 0n: Empty.absurd({Nat.is_le(Nat.sub(1n+count,0n),rest) == True{} : Bool},WordEncoding.false_true(some))
    case 1n+count 1n+cells: Order.transitive(Nat.sub(count,cells),count,rest,Order.sub_le(count,cells),bound)

# After a step: the sums of the next run are the sums of this one.
def joined(+base: U32,+moved: U32,+left: U32,+cells: Nat,+count: Nat,+exact_moved: {U32.to_nat(moved) == Nat.add(U32.to_nat(base),cells) : Nat},
  +exact_left: {U32.to_nat(left) == Nat.sub(count,cells) : Nat},+most: {Nat.is_le(cells,count) == True{} : Bool}) ->
  {Nat.add(U32.to_nat(moved),U32.to_nat(left)) == Nat.add(U32.to_nat(base),count) : Nat}:
  Equal.trans(Nat,Nat.add(U32.to_nat(moved),U32.to_nat(left)),Nat.add(Nat.add(U32.to_nat(base),cells),U32.to_nat(left)),Nat.add(U32.to_nat(base),count),
    Equal.cong(Nat,Nat,value => Nat.add(value,U32.to_nat(left)),U32.to_nat(moved),Nat.add(U32.to_nat(base),cells),exact_moved),
    Equal.trans(Nat,Nat.add(Nat.add(U32.to_nat(base),cells),U32.to_nat(left)),Nat.add(Nat.add(U32.to_nat(base),cells),Nat.sub(count,cells)),Nat.add(U32.to_nat(base),count),
      Equal.cong(Nat,Nat,value => Nat.add(Nat.add(U32.to_nat(base),cells),value),U32.to_nat(left),Nat.sub(count,cells),exact_left),
      rejoin(U32.to_nat(base),cells,count,most)))

def step_outside(+position: Nat,+cells: Nat,+count: Nat,+query: Nat,+most: {Nat.is_le(cells,count) == True{} : Bool},+outside: Interval.Outside<position,count,query>) ->
  Interval.Outside<position,cells,query>:
  match outside:
    case Interval.Before{+bound}: Interval.Before{bound}
    case Interval.After{+bound}: Interval.After{Order.transitive(Nat.add(position,cells),Nat.add(position,count),query,le_shift(position,cells,count,most),bound)}

def next_outside(+position: U32,+moved: U32,+left: U32,+cells: Nat,+count: Nat,+query: Nat,+exact_moved: {U32.to_nat(moved) == Nat.add(U32.to_nat(position),cells) : Nat},
  +sum: {Nat.add(U32.to_nat(moved),U32.to_nat(left)) == Nat.add(U32.to_nat(position),count) : Nat},+outside: Interval.Outside<U32.to_nat(position),count,query>) ->
  Interval.Outside<U32.to_nat(moved),U32.to_nat(left),query>:
  match outside:
    case Interval.Before{+bound}:
      Interval.Before{nat_lt_moved(query,U32.to_nat(moved),Nat.add(U32.to_nat(position),cells),exact_moved,
        Bounds.nat_trans(query,U32.to_nat(position),Nat.add(U32.to_nat(position),cells),bound,Order.le_add(U32.to_nat(position),cells)))}
    case Interval.After{+bound}: Interval.After{nat_le_moved(Nat.add(U32.to_nat(moved),U32.to_nat(left)),Nat.add(U32.to_nat(position),count),query,sum,bound)}

# Cell zero of a run is its offset, and cell cells + later is cell later of the
# run after its first cells.
def run_start(+offset: U32,+step: U32) -> {(offset + U32.from_nat(0n) * step : U32) == offset : U32}:
  Equal.trans(U32,(offset + 0 * step : U32),(offset + 0 : U32),offset,
    Equal.cong(U32,U32,product => (offset + product : U32),(0 * step : U32),0,Arithmetic.multiply_zero_left(step)),Arithmetic.add_zero(offset))

def run_shift(+offset: U32,+step: U32,+cells: Nat,+later: Nat) ->
  {((offset + U32.from_nat(cells) * step : U32) + U32.from_nat(later) * step : U32) == (offset + U32.from_nat(Nat.add(cells,later)) * step : U32) : U32}:
  Equal.trans(U32,((offset + U32.from_nat(cells) * step : U32) + U32.from_nat(later) * step : U32),(offset + (U32.from_nat(cells) * step + U32.from_nat(later) * step : U32) : U32),
    (offset + U32.from_nat(Nat.add(cells,later)) * step : U32),
    Arithmetic.add_associative(offset,(U32.from_nat(cells) * step : U32),(U32.from_nat(later) * step : U32)),
    Equal.cong(U32,U32,word => (offset + word : U32),(U32.from_nat(cells) * step + U32.from_nat(later) * step : U32),(U32.from_nat(Nat.add(cells,later)) * step : U32),
      Equal.trans(U32,(U32.from_nat(cells) * step + U32.from_nat(later) * step : U32),((U32.from_nat(cells) + U32.from_nat(later) : U32) * step : U32),
        (U32.from_nat(Nat.add(cells,later)) * step : U32),
        Equal.sym(U32,((U32.from_nat(cells) + U32.from_nat(later) : U32) * step : U32),(U32.from_nat(cells) * step + U32.from_nat(later) * step : U32),
          Arithmetic.multiply_additive(U32.from_nat(cells),U32.from_nat(later),step)),
        Equal.cong(U32,U32,word => (word * step : U32),(U32.from_nat(cells) + U32.from_nat(later) : U32),U32.from_nat(Nat.add(cells,later)),WordArithmetic.native_add(cells,later)))))

def run_after(+offset: U32,+step: U32,+cells: U32,+later: Nat) ->
  {((offset + cells * step : U32) + U32.from_nat(later) * step : U32) == (offset + U32.from_nat(Nat.add(U32.to_nat(cells),later)) * step : U32) : U32}:
  Equal.trans(U32,((offset + cells * step : U32) + U32.from_nat(later) * step : U32),((offset + U32.from_nat(U32.to_nat(cells)) * step : U32) + U32.from_nat(later) * step : U32),
    (offset + U32.from_nat(Nat.add(U32.to_nat(cells),later)) * step : U32),
    Equal.cong(U32,U32,word => ((offset + word * step : U32) + U32.from_nat(later) * step : U32),cells,U32.from_nat(U32.to_nat(cells)),
      Equal.sym(U32,U32.from_nat(U32.to_nat(cells)),cells,Arithmetic.native_roundtrip(cells))),
    run_shift(offset,step,U32.to_nat(cells),later))

def run_next(+offset: U32,+step: U32,+rest: Nat) -> {((offset + step : U32) + U32.from_nat(rest) * step : U32) == (offset + U32.from_nat(1n+rest) * step : U32) : U32}:
  Equal.trans(U32,((offset + step : U32) + U32.from_nat(rest) * step : U32),((offset + 1 * step : U32) + U32.from_nat(rest) * step : U32),(offset + U32.from_nat(1n+rest) * step : U32),
    Equal.cong(U32,U32,word => ((offset + word : U32) + U32.from_nat(rest) * step : U32),step,(1 * step : U32),
      Equal.sym(U32,(1 * step : U32),step,Equal.trans(U32,(1 * step : U32),(step * 1 : U32),step,Arithmetic.multiply_commutative(1,step),Arithmetic.multiply_one(step)))),
    run_shift(offset,step,1n,rest))

# Every cell of a run of count cells from offset, step apart, lies in one of
# the ranges.
def RunHeld(ranges: Spans,+offset: U32,+step: U32,count: Nat) -> Data:
  match count:
    case 0n: Unit
    case 1n+rest: Both<SpanHeld(ranges,U32.to_nat(offset)),RunHeld(ranges,(offset + step : U32),step,rest)>

def run_head(ranges: Spans,+offset: U32,+step: U32,count: Nat,+positive: {Nat.is_lt(0n,count) == True{} : Bool},+held: RunHeld(ranges,offset,step,count)) ->
  SpanHeld(ranges,U32.to_nat(offset)):
  match count:
    case 0n: Empty.absurd(SpanHeld(ranges,U32.to_nat(offset)),WordEncoding.false_true(positive))
    case 1n+ +rest:
      Both{+head,+tail} = held
      head

# The cells of a run after its first cells are held when those of the run are.
def run_drop(ranges: Spans,count: Nat,cells: Nat,+offset: U32,+step: U32,+held: RunHeld(ranges,offset,step,count)) ->
  RunHeld(ranges,(offset + U32.from_nat(cells) * step : U32),step,Nat.sub(count,cells)):
  match count cells:
    case +count 0n:
      %Equal.sym(U32,(offset + U32.from_nat(0n) * step : U32),offset,run_start(offset,step)) : RunHeld(ranges,_,step,Nat.sub(count,0n))
      %Equal.sym(Nat,Nat.sub(count,0n),count,Order.subtract_zero(count)) : RunHeld(ranges,offset,step,_)
      held
    case 0n 1n+rest: Unit{}
    case 1n+ +more 1n+ +rest:
      Both{+head,+tail} = held
      %run_next(offset,step,rest) : RunHeld(ranges,_,step,Nat.sub(more,rest))
      run_drop(ranges,more,rest,(offset + step : U32),step,tail)

def run_moved(ranges: Spans,+from: U32,+to: U32,+step: U32,+from_count: Nat,+to_count: Nat,+offset_equal: {from == to : U32},+count_equal: {from_count == to_count : Nat},
  +held: RunHeld(ranges,from,step,from_count)) -> RunHeld(ranges,to,step,to_count):
  %offset_equal : RunHeld(ranges,_,step,to_count)
  %count_equal : RunHeld(ranges,from,step,_)
  held

def sorted_moved(~Element: Data,+from: Stack<Element>,+to: Stack<Element>,+low: Nat,+high: Nat,+equal: {from == to : Stack<Element>},+sorted: Sorted(~Element,from,low,high)) ->
  Sorted(~Element,to,low,high):
  %equal : Sorted(~Element,_,low,high)
  sorted

def stack_held(~Element: Data,+from: Stack<Element>,+to: Stack<Element>,+index: Nat,+equal: {from == to : Stack<Element>},+held: SpanHeld(spans(~Element,from),index)) ->
  SpanHeld(spans(~Element,to),index):
  %equal : SpanHeld(spans(~Element,_),index)
  held

# A run leaves the target cells outside [position, position + count).
def run_kept(~Element: Data,fuel: Nat,+done: Bool,+count: U32,+offset: U32,+stride: U32,+back: Bool,+position: U32,+holding: Holding<Element>,+tree: Tree.Tree<Element>,
  +depth: Nat,+balanced: Shape.Balanced(Element,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +target_ok: {U32.is_le(Tree.capacity(Element,tree),2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(position),U32.to_nat(count)),U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool},
  +query: U32,+query_inside: {U32.is_lt(query,Tree.capacity(Element,tree)) == True{} : Bool},
  +outside: Interval.Outside<U32.to_nat(position),U32.to_nat(count),U32.to_nat(query)>) ->
  {Tree.read(Element,ran_tree(~Element,model_run(~Element,fuel,done,count,offset,stride,back,position,holding,tree)),query) == Tree.read(Element,tree,query) : Element}:
  match fuel done:
    case 0n done: {==}
    case 1n+rest True{}: {==}
    case 1n+ +rest False{}:
      +resolved = resolve(~Element,holding,offset)
      +cells = step_cells(~Element,resolved,count,offset,stride,back)
      +most = step_cells_most(~Element,resolved,count,offset,stride,back)
      +target = Tree.capacity(Element,tree)
      +count_cap = Order.transitive(U32.to_nat(count),Nat.add(U32.to_nat(position),U32.to_nat(count)),U32.to_nat(target),Order.right_le(U32.to_nat(position),U32.to_nat(count)),room)
      +exact_count = left_count(count,cells,target,target_ok,most,count_cap)
      +exact_position = Arithmetic.add(position,cells,target,target_ok,Order.transitive(Nat.add(U32.to_nat(position),U32.to_nat(cells)),
        Nat.add(U32.to_nat(position),U32.to_nat(count)),U32.to_nat(target),le_shift(U32.to_nat(position),U32.to_nat(cells),U32.to_nat(count),most),room))
      +next_count = (count - cells : U32)
      +next_offset = (offset + cells * Move.directed(back,stride) : U32)
      +next_position = (position + cells : U32)
      +next_tree = step_tree(~Element,resolved,count,offset,stride,back,position,tree)
      +same_capacity = step_capacity(~Element,resolved,count,offset,stride,back,position,tree)
      +position_sum = joined(position,next_position,next_count,U32.to_nat(cells),U32.to_nat(count),exact_position,exact_count,most)
      Equal.trans(Element,Tree.read(Element,ran_tree(~Element,model_run(~Element,rest,U32.is_zero(next_count),next_count,next_offset,stride,back,next_position,resolved,next_tree)),query),
        Tree.read(Element,next_tree,query),Tree.read(Element,tree,query),
        run_kept(~Element,rest,U32.is_zero(next_count),next_count,next_offset,stride,back,next_position,resolved,next_tree,depth,
          step_balanced(~Element,resolved,count,offset,stride,back,position,depth,tree,balanced),depth_ok,
          bool_moved(~U32,~word => U32.is_le(word,2147483648),Tree.capacity(Element,next_tree),target,same_capacity,target_ok),
          nat_le_right_moved(Nat.add(U32.to_nat(next_position),U32.to_nat(next_count)),U32.to_nat(Tree.capacity(Element,next_tree)),U32.to_nat(target),
            Equal.cong(U32,Nat,U32.to_nat,Tree.capacity(Element,next_tree),target,same_capacity),
            nat_le_moved(Nat.add(U32.to_nat(next_position),U32.to_nat(next_count)),Nat.add(U32.to_nat(position),U32.to_nat(count)),U32.to_nat(target),position_sum,room)),
          query,bool_moved(~U32,~word => U32.is_lt(query,word),Tree.capacity(Element,next_tree),target,same_capacity,query_inside),
          next_outside(position,next_position,next_count,U32.to_nat(cells),U32.to_nat(count),U32.to_nat(query),exact_position,position_sum,outside)),
        kept_step(~Element,resolved,count,offset,stride,back,position,depth,tree,balanced,depth_ok,target_ok,room,query,query_inside,
          step_outside(U32.to_nat(position),U32.to_nat(cells),U32.to_nat(count),U32.to_nat(query),most,outside)))

# A run whose cells are held by pieces in order writes target cell position + k
# with the value of cell offset + k times its step, for every k below count.
def run_written(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,fuel: Nat,+done: Bool,+count: U32,+offset: U32,+stride: U32,+back: Bool,
  +position: U32,+holding: Holding<Element>,+tree: Tree.Tree<Element>,+stack: Stack<Element>,+ordered: {order(~Element,holding) == stack : Stack<Element>},
  +low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool},
  +sorted: Sorted(~Element,stack,low,high),+values: HoldingValues(~Element,~Context,~value,context,holding),
  +held: RunHeld(spans(~Element,stack),offset,Move.directed(back,stride),U32.to_nat(count)),
  +fuelled: {Nat.is_le(U32.to_nat(count),fuel) == True{} : Bool},+stopped: {done == U32.is_zero(count) : Bool},
  +depth: Nat,+balanced: Shape.Balanced(Element,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +target_ok: {U32.is_le(Tree.capacity(Element,tree),2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(position),U32.to_nat(count)),U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool},
  +index: Nat,+less: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Element,ran_tree(~Element,model_run(~Element,fuel,done,count,offset,stride,back,position,holding,tree)),U32.from_nat(Nat.add(U32.to_nat(position),index))) ==
    value(context,(offset + U32.from_nat(index) * Move.directed(back,stride) : U32)) : Element}:
  match fuel done:
    case 0n done:
      Empty.absurd({Tree.read(Element,ran_tree(~Element,model_run(~Element,0n,done,count,offset,stride,back,position,holding,tree)),U32.from_nat(Nat.add(U32.to_nat(position),index))) ==
        value(context,(offset + U32.from_nat(index) * Move.directed(back,stride) : U32)) : Element},Order.below_zero(index,Bounds.nat_trans(index,U32.to_nat(count),0n,less,fuelled)))
    case 1n+rest True{}:
      +zero = Arithmetic.word_zero(count,Equal.sym(Bool,True{},U32.is_zero(count),stopped))
      Empty.absurd({Tree.read(Element,ran_tree(~Element,model_run(~Element,1n+rest,True{},count,offset,stride,back,position,holding,tree)),U32.from_nat(Nat.add(U32.to_nat(position),index))) ==
        value(context,(offset + U32.from_nat(index) * Move.directed(back,stride) : U32)) : Element},
        Order.below_zero(index,nat_lt_moved(index,U32.to_nat(0),U32.to_nat(count),Equal.sym(Nat,U32.to_nat(count),U32.to_nat(0),Equal.cong(U32,Nat,U32.to_nat,count,0,zero)),less)))
    case 1n+ +rest False{}:
      +direction = Move.directed(back,stride)
      +positive = Arithmetic.word_nonzero_positive(count,Equal.sym(Bool,False{},U32.is_zero(count),stopped))
      +reorder = Equal.sym(Stack<Element>,order(~Element,holding),stack,ordered)
      +current = resolve_current(~Element,holding,offset,low,high,capacity,capacity_ok,bounded,
        sorted_moved(~Element,stack,order(~Element,holding),low,high,reorder,sorted),
        stack_held(~Element,stack,order(~Element,holding),U32.to_nat(offset),reorder,run_head(spans(~Element,stack),offset,direction,U32.to_nat(count),positive,held)))
      +resolved = resolve(~Element,holding,offset)
      +resolved_values = resolve_values(~Element,~Context,~value,context,holding,offset,values)
      +resolved_order = Equal.trans(Stack<Element>,order(~Element,resolved),order(~Element,holding),stack,resolve_order(~Element,holding,offset),ordered)
      +cells = step_cells(~Element,resolved,count,offset,stride,back)
      +copies = step_copies(~Element,resolved,count,offset,stride,back,capacity,capacity_ok,word,word_ok,stride_room,current,positive)
      +some = copies_some(cells,count,copies)
      +most = copies_most(cells,count,copies)
      +target = Tree.capacity(Element,tree)
      +count_cap = Order.transitive(U32.to_nat(count),Nat.add(U32.to_nat(position),U32.to_nat(count)),U32.to_nat(target),Order.right_le(U32.to_nat(position),U32.to_nat(count)),room)
      +exact_count = left_count(count,cells,target,target_ok,most,count_cap)
      +exact_position = Arithmetic.add(position,cells,target,target_ok,Order.transitive(Nat.add(U32.to_nat(position),U32.to_nat(cells)),
        Nat.add(U32.to_nat(position),U32.to_nat(count)),U32.to_nat(target),le_shift(U32.to_nat(position),U32.to_nat(cells),U32.to_nat(count),most),room))
      +next_count = (count - cells : U32)
      +next_offset = (offset + cells * direction : U32)
      +next_position = (position + cells : U32)
      +next_tree = step_tree(~Element,resolved,count,offset,stride,back,position,tree)
      +final = ran_tree(~Element,model_run(~Element,rest,U32.is_zero(next_count),next_count,next_offset,stride,back,next_position,resolved,next_tree))
      +same_capacity = step_capacity(~Element,resolved,count,offset,stride,back,position,tree)
      +position_sum = joined(position,next_position,next_count,U32.to_nat(cells),U32.to_nat(count),exact_position,exact_count,most)
      +next_target_ok = bool_moved(~U32,~word => U32.is_le(word,2147483648),Tree.capacity(Element,next_tree),target,same_capacity,target_ok)
      +next_room = nat_le_right_moved(Nat.add(U32.to_nat(next_position),U32.to_nat(next_count)),U32.to_nat(Tree.capacity(Element,next_tree)),U32.to_nat(target),
        Equal.cong(U32,Nat,U32.to_nat,Tree.capacity(Element,next_tree),target,same_capacity),
        nat_le_moved(Nat.add(U32.to_nat(next_position),U32.to_nat(next_count)),Nat.add(U32.to_nat(position),U32.to_nat(count)),U32.to_nat(target),position_sum,room))
      +reach = Nat.add(U32.to_nat(position),index)
      +reach_below = Bounds.nat_trans(reach,Nat.add(U32.to_nat(position),U32.to_nat(count)),U32.to_nat(target),lt_shift(U32.to_nat(position),index,U32.to_nat(count),less),room)
      +reach_exact = Bounds.roundtrip(reach,target,target_ok,Bounds.strict_le(reach,U32.to_nat(target),reach_below))
      +source = (offset + U32.from_nat(index) * direction : U32)
      by_cases({Tree.read(Element,final,U32.from_nat(reach)) == value(context,source) : Element},Nat.is_lt(index,U32.to_nat(cells)),
        +below => Equal.trans(Element,Tree.read(Element,final,U32.from_nat(reach)),Tree.read(Element,next_tree,U32.from_nat(reach)),value(context,source),
          run_kept(~Element,rest,U32.is_zero(next_count),next_count,next_offset,stride,back,next_position,resolved,next_tree,depth,
            step_balanced(~Element,resolved,count,offset,stride,back,position,depth,tree,balanced),depth_ok,next_target_ok,next_room,U32.from_nat(reach),
            bool_moved(~U32,~word => U32.is_lt(U32.from_nat(reach),word),Tree.capacity(Element,next_tree),target,same_capacity,
              u32_lt(U32.from_nat(reach),target,nat_lt_left(U32.to_nat(U32.from_nat(reach)),reach,U32.to_nat(target),reach_exact,reach_below))),
            Interval.Before{nat_lt_left(U32.to_nat(U32.from_nat(reach)),reach,U32.to_nat(next_position),reach_exact,
              nat_lt_moved(reach,U32.to_nat(next_position),Nat.add(U32.to_nat(position),U32.to_nat(cells)),exact_position,lt_shift(U32.to_nat(position),index,U32.to_nat(cells),below)))}),
          written_step(~Element,~Context,~value,context,resolved,count,offset,stride,back,position,depth,tree,capacity,capacity_ok,word,word_ok,stride_room,current,resolved_values,
            balanced,depth_ok,target_ok,room,index,below)),
        +beyond =>
          +ordered_index = lt_false_le(index,U32.to_nat(cells),beyond)
          +later = Nat.sub(index,U32.to_nat(cells))
          +position_index = Equal.trans(Nat,Nat.add(U32.to_nat(next_position),later),Nat.add(Nat.add(U32.to_nat(position),U32.to_nat(cells)),later),reach,
            Equal.cong(Nat,Nat,number => Nat.add(number,later),U32.to_nat(next_position),Nat.add(U32.to_nat(position),U32.to_nat(cells)),exact_position),
            rejoin(U32.to_nat(position),U32.to_nat(cells),index,ordered_index))
          +later_less = nat_lt_moved(later,U32.to_nat(next_count),Nat.sub(U32.to_nat(count),U32.to_nat(cells)),exact_count,
            Order.sub_lt(index,U32.to_nat(cells),Nat.sub(U32.to_nat(count),U32.to_nat(cells)),ordered_index,
              nat_lt_moved(index,Nat.add(U32.to_nat(cells),Nat.sub(U32.to_nat(count),U32.to_nat(cells))),U32.to_nat(count),Order.restore_shift(U32.to_nat(cells),U32.to_nat(count),most),less)))
          +shift = Equal.trans(U32,(next_offset + U32.from_nat(later) * direction : U32),(offset + U32.from_nat(Nat.add(U32.to_nat(cells),later)) * direction : U32),source,
            run_after(offset,direction,cells,later),
            Equal.cong(Nat,U32,number => (offset + U32.from_nat(number) * direction : U32),Nat.add(U32.to_nat(cells),later),index,Order.restore_shift(U32.to_nat(cells),index,ordered_index)))
          Equal.trans(Element,Tree.read(Element,final,U32.from_nat(reach)),Tree.read(Element,final,U32.from_nat(Nat.add(U32.to_nat(next_position),later))),value(context,source),
            Equal.cong(Nat,Element,number => Tree.read(Element,final,U32.from_nat(number)),reach,Nat.add(U32.to_nat(next_position),later),
              Equal.sym(Nat,Nat.add(U32.to_nat(next_position),later),reach,position_index)),
            Equal.trans(Element,Tree.read(Element,final,U32.from_nat(Nat.add(U32.to_nat(next_position),later))),value(context,(next_offset + U32.from_nat(later) * direction : U32)),
              value(context,source),
              run_written(~Element,~Context,~value,context,rest,U32.is_zero(next_count),next_count,next_offset,stride,back,next_position,resolved,next_tree,stack,resolved_order,
                low,high,capacity,capacity_ok,bounded,word,word_ok,stride_room,sorted,resolved_values,
                run_moved(spans(~Element,stack),(offset + U32.from_nat(U32.to_nat(cells)) * direction : U32),next_offset,direction,
                  Nat.sub(U32.to_nat(count),U32.to_nat(cells)),U32.to_nat(next_count),
                  Equal.cong(U32,U32,word => (offset + word * direction : U32),U32.from_nat(U32.to_nat(cells)),cells,Arithmetic.native_roundtrip(cells)),
                  Equal.sym(Nat,U32.to_nat(next_count),Nat.sub(U32.to_nat(count),U32.to_nat(cells)),exact_count),
                  run_drop(spans(~Element,stack),U32.to_nat(count),U32.to_nat(cells),offset,direction,held)),
                nat_le_moved(U32.to_nat(next_count),Nat.sub(U32.to_nat(count),U32.to_nat(cells)),rest,exact_count,sub_pred(U32.to_nat(count),U32.to_nat(cells),rest,some,fuelled)),
                {==},depth,step_balanced(~Element,resolved,count,offset,stride,back,position,depth,tree,balanced),depth_ok,next_target_ok,next_room,later,later_less),
              Equal.cong(U32,Element,word => value(context,word),(next_offset + U32.from_nat(later) * direction : U32),source,shift))))

def run_target(~Element: Data,-result: Move.Holding<Element> & Storage.Buffer<Element>,-ran: Ran<Element>,-length: Nat,+facts: RunResult<Element,result,ran,length>) ->
  {Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,result)) == Buffer.Content{ran_tree(~Element,ran),length} : Buffer.Content<Element>}:
  RunResult{holding,target} = facts
  target

def run_held(~Element: Data,-result: Move.Holding<Element> & Storage.Buffer<Element>,-ran: Ran<Element>,-length: Nat,+facts: RunResult<Element,result,ran,length>) ->
  {Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,result) == pack(~Element,ran_holding(~Element,ran)) : Move.Holding<Element>}:
  RunResult{holding,target} = facts
  holding

def band_buffer(~Element: Data,band: Move.Band<Element>) -> Storage.Buffer<Element>:
  Move.Band{lent,buffer,offset} = band
  buffer

# A fresh buffer is the pack of a model whose tree is the fresh tree.
def fresh_tree(~Element: Data,+depth: Nat,+fill: Element,+length: Nat) ->
  {Buffer.tree(~Element,Buffer.remodel(~Element,Allocation.allocate(~Element,depth,fill,length))) == Tree.fresh(Element,depth,fill) : Tree.Tree<Element>}:
  +model = Buffer.remodel(~Element,Allocation.allocate(~Element,depth,fill,length))
  Equal.trans(Tree.Tree<Element>,Buffer.tree(~Element,model),Buffer.content_tree(~Element,Buffer.content(~Element,model)),Tree.fresh(Element,depth,fill),
    Equal.sym(Tree.Tree<Element>,Buffer.content_tree(~Element,Buffer.content(~Element,model)),Buffer.tree(~Element,model),Buffer.content_tree_of(~Element,model)),
    Equal.cong(Buffer.Content<Element>,Tree.Tree<Element>,Buffer.content_tree(~Element),Buffer.content(~Element,model),Buffer.Content{Tree.fresh(Element,depth,fill),length},
      Equal.trans(Buffer.Content<Element>,Buffer.content(~Element,model),Buffer.contents(~Element,Allocation.allocate(~Element,depth,fill,length)),Buffer.Content{Tree.fresh(Element,depth,fill),length},
        Buffer.remodel_content(~Element,Allocation.allocate(~Element,depth,fill,length)),AllocationLaws.contents(~Element,depth,fill,length))))

# A unit run reads cell first + k.
def unit_source(+first: U32,+index: Nat) -> {(first + U32.from_nat(index) * Move.directed(False{},1) : U32) == U32.from_nat(Nat.add(U32.to_nat(first),index)) : U32}:
  Equal.trans(U32,(first + U32.from_nat(index) * 1 : U32),(U32.from_nat(U32.to_nat(first)) + U32.from_nat(index) : U32),U32.from_nat(Nat.add(U32.to_nat(first),index)),
    Equal.trans(U32,(first + U32.from_nat(index) * 1 : U32),(first + U32.from_nat(index) : U32),(U32.from_nat(U32.to_nat(first)) + U32.from_nat(index) : U32),
      Equal.cong(U32,U32,word => (first + word : U32),(U32.from_nat(index) * 1 : U32),U32.from_nat(index),Arithmetic.multiply_one(U32.from_nat(index))),
      Equal.cong(U32,U32,word => (word + U32.from_nat(index) : U32),first,U32.from_nat(U32.to_nat(first)),
        Equal.sym(U32,U32.from_nat(U32.to_nat(first)),first,Arithmetic.native_roundtrip(first)))),
    WordArithmetic.native_add(U32.to_nat(first),index))

# Pieces that tile a range hold every cell of a unit run inside it.
def unit_run(~Element: Data,+holding: Holding<Element>,count: Nat,+first: U32,+low: Nat,+high: Nat,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +tiled: Tiled(~Element,holding,low,high),+after: {Nat.is_le(low,U32.to_nat(first)) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(first),count),high) == True{} : Bool}) ->
  RunHeld(spans(~Element,order(~Element,holding)),first,Move.directed(False{},1),count):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      +start = U32.to_nat(first)
      +inside = Bounds.nat_trans(start,Nat.add(start,1n+rest),high,
        nat_lt_left(start,Nat.add(start,0n),Nat.add(start,1n+rest),Equal.sym(Nat,Nat.add(start,0n),start,Addition.zero_right(start)),lt_shift(start,0n,1n+rest,{==})),fits)
      +next_room = Order.transitive(Nat.add(start,1n),Nat.add(start,1n+rest),U32.to_nat(capacity),le_shift(start,1n,1n+rest,Order.zero_le(rest)),
        Order.transitive(Nat.add(start,1n+rest),high,U32.to_nat(capacity),fits,bounded))
      +exact_next = Arithmetic.add(first,1,capacity,capacity_ok,next_room)
      Both{tiles_member(~Element,order(~Element,holding),low,high,start,order_tiles(~Element,holding,low,high,tiled),after,inside),
        unit_run(~Element,holding,rest,(first + 1 : U32),low,high,capacity,capacity_ok,bounded,tiled,
          nat_le_right_moved(low,U32.to_nat((first + 1 : U32)),Nat.add(start,1n),exact_next,Order.transitive(low,start,Nat.add(start,1n),after,Order.le_add(start,1n))),
          nat_le_moved(Nat.add(U32.to_nat((first + 1 : U32)),rest),Nat.add(start,1n+rest),high,
            Equal.trans(Nat,Nat.add(U32.to_nat((first + 1 : U32)),rest),Nat.add(Nat.add(start,1n),rest),Nat.add(start,1n+rest),
              Equal.cong(Nat,Nat,number => Nat.add(number,rest),U32.to_nat((first + 1 : U32)),Nat.add(start,1n),exact_next),
              Addition.associative(start,1n,rest)),
            fits))}

# Room for a copy of count cells into a target: a balanced tree of depth at
# most 30 that holds count cells.
type CopyRoom<-Element: Data,-target: Buffer.Model<Element>,-count: U32> is Data:
  CopyRoom{depth: Nat,balanced: Shape.Balanced(Element,depth,Buffer.tree(~Element,target)),depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
    room: {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,target)))) == True{} : Bool}}

# A fresh buffer of count cells has room for them.
def fresh_room(~Element: Data,+count: U32,+fill: Element,+allocatable: {U32.is_le(count,1073741824) == True{} : Bool}) ->
  CopyRoom<Element,Buffer.remodel(~Element,Allocation.allocate(~Element,Storage.depth_for(count),fill,U32.to_nat(count))),count>:
  +depth = Storage.depth_for(count)
  +sized = Depth.bounded(count,allocatable)
  +model = Buffer.remodel(~Element,Allocation.allocate(~Element,depth,fill,U32.to_nat(count)))
  +same = fresh_tree(~Element,depth,fill,U32.to_nat(count))
  +capacity_exact = Shape.capacity(Element,depth,Tree.fresh(Element,depth,fill),Shape.fresh(Element,depth,fill))
  +balanced = {
    %Equal.sym(Tree.Tree<Element>,Buffer.tree(~Element,model),Tree.fresh(Element,depth,fill),same) : Shape.Balanced(Element,depth,_)
    Shape.fresh(Element,depth,fill) : Shape.Balanced(Element,depth,Buffer.tree(~Element,model))}
  +room = {
    %Equal.sym(Tree.Tree<Element>,Buffer.tree(~Element,model),Tree.fresh(Element,depth,fill),same) : {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Element,_))) == True{} : Bool}
    %Equal.sym(U32,Tree.capacity(Element,Tree.fresh(Element,depth,fill)),U32.shln(1,depth),capacity_exact) : {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool}
    Depth.allocation_room(count,sized) : {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,model)))) == True{} : Bool}}
  CopyRoom{depth,balanced,Depth.allocation_depth(count,sized),room}

# A copy of [first, first + count) from pieces that cover a range holding it,
# with the values of a tensor, into a target with room for it: cell k of the
# target is the value of cell first + k.
def copied_into_cell(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+holding: Holding<Element>,+first: U32,+count: U32,
  +target: Buffer.Model<Element>,+fits: CopyRoom<Element,target,count>,
  +low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +tiled: Tiled(~Element,holding,low,high),+values: HoldingValues(~Element,~Context,~value,context,holding),+after: {Nat.is_le(low,U32.to_nat(first)) == True{} : Bool},
  +inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),high) == True{} : Bool},
  +index: Nat,+less: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Element,Buffer.content_tree(~Element,Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,
      Move.copied_into(~Element,pack(~Element,holding),first,count,Buffer.pack(~Element,target))))),U32.from_nat(index)) ==
    value(context,U32.from_nat(Nat.add(U32.to_nat(first),index))) : Element}:
  CopyRoom{+depth,+balanced,+depth_ok,+room} = fits
  +tree = Buffer.tree(~Element,target)
  +capacity_exact = Shape.capacity(Element,depth,tree,balanced)
  +target_ok = {
    %Equal.sym(U32,Tree.capacity(Element,tree),U32.shln(1,depth),capacity_exact) : {U32.is_le(_,2147483648) == True{} : Bool}
    Power.native_power_bound(depth,Order.transitive(depth,30n,31n,depth_ok,{==})) : {U32.is_le(Tree.capacity(Element,tree),2147483648) == True{} : Bool}}
  +ran = model_run(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,tree)
  Equal.trans(Element,
    Tree.read(Element,Buffer.content_tree(~Element,Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.run(~Element,U32.to_nat(count),count,first,1,False{},0,pack(~Element,holding),Buffer.pack(~Element,target))))),U32.from_nat(index)),
    Tree.read(Element,ran_tree(~Element,ran),U32.from_nat(index)),value(context,U32.from_nat(Nat.add(U32.to_nat(first),index))),
    Equal.cong(Buffer.Content<Element>,Element,content => Tree.read(Element,Buffer.content_tree(~Element,content),U32.from_nat(index)),
      Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.run(~Element,U32.to_nat(count),count,first,1,False{},0,pack(~Element,holding),Buffer.pack(~Element,target)))),Buffer.Content{ran_tree(~Element,ran),Buffer.length(~Element,target)},
      run_target(~Element,Move.run(~Element,U32.to_nat(count),count,first,1,False{},0,pack(~Element,holding),Buffer.pack(~Element,target)),ran,Buffer.length(~Element,target),run_refines(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,target))),
    Equal.trans(Element,Tree.read(Element,ran_tree(~Element,ran),U32.from_nat(index)),
      value(context,(first + U32.from_nat(index) * Move.directed(False{},1) : U32)),value(context,U32.from_nat(Nat.add(U32.to_nat(first),index))),
      run_written(~Element,~Context,~value,context,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,tree,order(~Element,holding),{==},
        low,high,capacity,capacity_ok,bounded,2,{==},{==},tiles_sorted(~Element,order(~Element,holding),low,high,order_tiles(~Element,holding,low,high,tiled)),values,
        unit_run(~Element,holding,U32.to_nat(count),first,low,high,capacity,capacity_ok,bounded,tiled,after,inside),
        Order.reflexive(U32.to_nat(count)),{==},depth,balanced,depth_ok,target_ok,room,index,less),
      Equal.cong(U32,Element,word => value(context,word),(first + U32.from_nat(index) * Move.directed(False{},1) : U32),U32.from_nat(Nat.add(U32.to_nat(first),index)),
        unit_source(first,index))))

# A copy of [first, first + count) into a fresh buffer: cell k of the copy is
# the value of cell first + k.
def copied_cell(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+holding: Holding<Element>,+first: U32,+count: U32,+fill: Element,
  +low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +tiled: Tiled(~Element,holding,low,high),+values: HoldingValues(~Element,~Context,~value,context,holding),+after: {Nat.is_le(low,U32.to_nat(first)) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),high) == True{} : Bool},
  +allocatable: {U32.is_le(count,1073741824) == True{} : Bool},
  +index: Nat,+less: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Element,Buffer.content_tree(~Element,Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))),U32.from_nat(index)) ==
    value(context,U32.from_nat(Nat.add(U32.to_nat(first),index))) : Element}:
  %Buffer.remodeled(~Element,Allocation.allocate(~Element,Storage.depth_for(count),fill,U32.to_nat(count))) :
    {Tree.read(Element,Buffer.content_tree(~Element,Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,
      Move.copied_into(~Element,pack(~Element,holding),first,count,_)))),U32.from_nat(index)) == value(context,U32.from_nat(Nat.add(U32.to_nat(first),index))) : Element}
  copied_into_cell(~Element,~Context,~value,context,holding,first,count,Buffer.remodel(~Element,Allocation.allocate(~Element,Storage.depth_for(count),fill,U32.to_nat(count))),fresh_room(~Element,count,fill,allocatable),
    low,high,capacity,capacity_ok,bounded,tiled,values,after,fits,index,less)

# ---- Spares ----

# A spare: a buffer model with room for capacity cells.
type SpareModel<-Element: Data> is Data:
  SpareModel{buffer: Buffer.Model<Element>,capacity: U32}

def pack_spare(~Element: Data,spare: SpareModel<Element>) -> Move.Spare<Element>:
  SpareModel{buffer,capacity} = spare
  Move.Spare{Buffer.pack(~Element,buffer),capacity}

def SpareRoom(~Element: Data,+spare: SpareModel<Element>) -> Data:
  SpareModel{+buffer,+capacity} = spare
  CopyRoom<Element,buffer,capacity>

# A new spare: an empty buffer with room for none.
def fresh_spare(~Element: Data,+fill: Element) -> SpareModel<Element>:
  SpareModel{Buffer.remodel(~Element,Allocation.allocate(~Element,0n,fill,0n)),0}

def spare_packed(~Element: Data,+fill: Element) -> {Move.spare(~Element,fill) == pack_spare(~Element,fresh_spare(~Element,fill)) : Move.Spare<Element>}:
  Equal.cong(Storage.Buffer<Element>,Move.Spare<Element>,buffer => Move.Spare{buffer,0},Allocation.allocate(~Element,0n,fill,0n),Buffer.pack(~Element,Buffer.remodel(~Element,Allocation.allocate(~Element,0n,fill,0n))),
    Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,Buffer.remodel(~Element,Allocation.allocate(~Element,0n,fill,0n))),Allocation.allocate(~Element,0n,fill,0n),Buffer.remodeled(~Element,Allocation.allocate(~Element,0n,fill,0n))))

def fresh_spare_room(~Element: Data,+fill: Element) -> SpareRoom(~Element,fresh_spare(~Element,fill)):
  +model = Buffer.remodel(~Element,Allocation.allocate(~Element,0n,fill,0n))
  +balanced = {
    %Equal.sym(Tree.Tree<Element>,Buffer.tree(~Element,model),Tree.fresh(Element,0n,fill),fresh_tree(~Element,0n,fill,0n)) : Shape.Balanced(Element,0n,_)
    Shape.fresh(Element,0n,fill) : Shape.Balanced(Element,0n,Buffer.tree(~Element,model))}
  CopyRoom{0n,balanced,{==},Order.zero_le(U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,model))))}

# The buffer a band assembles into: the spare's when it has room, otherwise a
# fresh one.
def roomy_model(~Element: Data,enough: Bool,+count: U32,+fill: Element,+buffer: Buffer.Model<Element>) -> Buffer.Model<Element>:
  match enough:
    case True{}: buffer
    case False{}: Buffer.remodel(~Element,Allocation.allocate(~Element,Storage.depth_for(count),fill,U32.to_nat(count)))

def roomy_packed(~Element: Data,enough: Bool,+count: U32,+fill: Element,+buffer: Buffer.Model<Element>) ->
  {Move.roomy(~Element,enough,count,fill,Buffer.pack(~Element,buffer)) == Buffer.pack(~Element,roomy_model(~Element,enough,count,fill,buffer)) : Storage.Buffer<Element>}:
  match enough:
    case True{}: {==}
    case False{}:
      Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,Buffer.remodel(~Element,Allocation.allocate(~Element,Storage.depth_for(count),fill,U32.to_nat(count)))),Allocation.allocate(~Element,Storage.depth_for(count),fill,U32.to_nat(count)),Buffer.remodeled(~Element,Allocation.allocate(~Element,Storage.depth_for(count),fill,U32.to_nat(count))))

def roomy_room(~Element: Data,enough: Bool,+count: U32,+fill: Element,+buffer: Buffer.Model<Element>,+capacity: U32,
  +enough_exact: {U32.is_le(count,capacity) == enough : Bool},+spare_room: CopyRoom<Element,buffer,capacity>,+allocatable: {U32.is_le(count,1073741824) == True{} : Bool}) ->
  CopyRoom<Element,roomy_model(~Element,enough,count,fill,buffer),count>:
  match enough:
    case True{}:
      CopyRoom{+depth,+balanced,+depth_ok,+room} = spare_room
      CopyRoom{depth,balanced,depth_ok,
        Order.transitive(U32.to_nat(count),U32.to_nat(capacity),U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,buffer))),Bounds.as_nat_le(count,capacity,enough_exact),room)}
    case False{}: fresh_room(~Element,count,fill,allocatable)

def band_offset(~Element: Data,band: Move.Band<Element>) -> U32:
  Move.Band{lent,buffer,offset} = band
  offset

def assembled_band_offset(~Element: Data,+capacity: U32,pair: Move.Holding<Element> & Storage.Buffer<Element>) ->
  {band_offset(~Element,Move.assembled_band(~Element,capacity,pair)) == 0 : U32}:
  (held,buffer) = pair
  {==}

# An assembled band starts at cell zero of its buffer.
def assembled_offset(~Element: Data,+first: U32,+count: U32,+fill: Element,holding: Move.Holding<Element>,spare: Move.Spare<Element>) ->
  {band_offset(~Element,Move.assembled(~Element,first,count,fill,holding,spare)) == 0 : U32}:
  Move.Spare{buffer,+capacity} = spare
  assembled_band_offset(~Element,U32.max(capacity,count),Move.copied_into(~Element,holding,first,count,Move.roomy(~Element,U32.is_le(count,capacity),count,fill,buffer)))

# The band of [first, first + count) assembled from pieces that cover a range
# holding it, with the values of a tensor: cell k of its buffer is the value of
# cell first + k.
def band_assembled_cell(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+holding: Holding<Element>,+first: U32,+count: U32,+fill: Element,
  +spare: SpareModel<Element>,+spare_room: SpareRoom(~Element,spare),
  +low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +tiled: Tiled(~Element,holding,low,high),+values: HoldingValues(~Element,~Context,~value,context,holding),+after: {Nat.is_le(low,U32.to_nat(first)) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),high) == True{} : Bool},
  +allocatable: {U32.is_le(count,1073741824) == True{} : Bool},
  +index: Nat,+less: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Element,Buffer.content_tree(~Element,Buffer.contents(~Element,band_buffer(~Element,Move.assembled(~Element,first,count,fill,pack(~Element,holding),pack_spare(~Element,spare))))),
      U32.from_nat(index)) == value(context,U32.from_nat(Nat.add(U32.to_nat(first),index))) : Element}:
  SpareModel{+buffer,+room_cells} = spare
  +enough = U32.is_le(count,room_cells)
  +target = roomy_model(~Element,enough,count,fill,buffer)
  %Equal.sym(Storage.Buffer<Element>,Move.roomy(~Element,enough,count,fill,Buffer.pack(~Element,buffer)),Buffer.pack(~Element,target),roomy_packed(~Element,enough,count,fill,buffer)) :
    {Tree.read(Element,Buffer.content_tree(~Element,Buffer.contents(~Element,band_buffer(~Element,Move.assembled_band(~Element,U32.max(room_cells,count),
      Move.copied_into(~Element,pack(~Element,holding),first,count,_))))),U32.from_nat(index)) == value(context,U32.from_nat(Nat.add(U32.to_nat(first),index))) : Element}
  %Equal.sym(Move.Holding<Element> & Storage.Buffer<Element>,Move.copied_into(~Element,pack(~Element,holding),first,count,Buffer.pack(~Element,target)),(Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied_into(~Element,pack(~Element,holding),first,count,Buffer.pack(~Element,target))),Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied_into(~Element,pack(~Element,holding),first,count,Buffer.pack(~Element,target)))),
    pair_eta(~Move.Holding<Element>,~Storage.Buffer<Element>,Move.copied_into(~Element,pack(~Element,holding),first,count,Buffer.pack(~Element,target)))) :
    {Tree.read(Element,Buffer.content_tree(~Element,Buffer.contents(~Element,band_buffer(~Element,Move.assembled_band(~Element,U32.max(room_cells,count),_)))),U32.from_nat(index)) ==
      value(context,U32.from_nat(Nat.add(U32.to_nat(first),index))) : Element}
  copied_into_cell(~Element,~Context,~value,context,holding,first,count,target,roomy_room(~Element,enough,count,fill,buffer,room_cells,{==},spare_room,allocatable),
    low,high,capacity,capacity_ok,bounded,tiled,values,after,fits,index,less)

# Cell offset + k of the buffer of a band is the value of cell first + k.
def BandRead(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,band: Move.Band<Element>,+first: U32,+index: Nat) -> Type:
  {Tree.read(Element,Buffer.content_tree(~Element,Buffer.contents(~Element,band_buffer(~Element,band))),U32.from_nat(Nat.add(U32.to_nat(band_offset(~Element,band)),index))) ==
    value(context,U32.from_nat(Nat.add(U32.to_nat(first),index))) : Element}

def band_when_cell(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+within: Bool,+behind: Stack<Element>,+buffer: Buffer.Model<Element>,
  +start: U32,+span: U32,+below: Stack<Element>,+first: U32,+count: U32,+fill: Element,+spare: SpareModel<Element>,+spare_room: SpareRoom(~Element,spare),
  +low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +tiled: Tiled(~Element,Holding{behind,Held{buffer,start,span,below}},low,high),+values: HoldingValues(~Element,~Context,~value,context,Holding{behind,Held{buffer,start,span,below}}),
  +holds: Holds<start,span,first,capacity>,+after: {Nat.is_le(low,U32.to_nat(first)) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),high) == True{} : Bool},
  +allocatable: {U32.is_le(count,1073741824) == True{} : Bool},
  +flag: {within == U32.is_le(start,first) && U32.is_le((first + count : U32),(start + span : U32)) : Bool},
  +index: Nat,+less: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  BandRead(~Element,~Context,~value,context,
    Move.band_when(~Element,within,first,count,fill,pack_stack(~Element,behind),Buffer.pack(~Element,buffer),start,span,pack_stack(~Element,below),pack_spare(~Element,spare)),first,index):
  match within:
    case True{}:
      Both{earlier,Both{+piece,later}} = values
      Holds{+holds_after,holds_before,+piece_room} = holds
      Equal.trans(Element,
        Tree.read(Element,Buffer.content_tree(~Element,Buffer.contents(~Element,Buffer.pack(~Element,buffer))),U32.from_nat(Nat.add(U32.to_nat((first - start : U32)),index))),
        Tree.read(Element,Buffer.tree(~Element,buffer),U32.from_nat(Nat.add(U32.to_nat((first - start : U32)),index))),value(context,U32.from_nat(Nat.add(U32.to_nat(first),index))),
        Equal.cong(Buffer.Content<Element>,Element,content => Tree.read(Element,Buffer.content_tree(~Element,content),U32.from_nat(Nat.add(U32.to_nat((first - start : U32)),index))),
          Buffer.contents(~Element,Buffer.pack(~Element,buffer)),Buffer.Content{Buffer.tree(~Element,buffer),Buffer.length(~Element,buffer)},content_parts(~Element,buffer)),
        band_cell(~Element,~Context,~value,context,first,count,start,span,Buffer.tree(~Element,buffer),Buffer.length(~Element,buffer),capacity,capacity_ok,index,holds_after,piece_room,
          Logic.right(U32.is_le(start,first),U32.is_le((first + count : U32),(start + span : U32)),Equal.sym(Bool,True{},U32.is_le(start,first) && U32.is_le((first + count : U32),(start + span : U32)),flag)),
          Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(count)),high,U32.to_nat(capacity),fits,bounded),less,piece))
    case False{}:
      %Equal.sym(U32,band_offset(~Element,Move.assembled(~Element,first,count,fill,pack(~Element,Holding{behind,Held{buffer,start,span,below}}),pack_spare(~Element,spare))),0,assembled_offset(~Element,first,count,fill,pack(~Element,Holding{behind,Held{buffer,start,span,below}}),pack_spare(~Element,spare))) :
        {Tree.read(Element,Buffer.content_tree(~Element,Buffer.contents(~Element,band_buffer(~Element,Move.assembled(~Element,first,count,fill,pack(~Element,Holding{behind,Held{buffer,start,span,below}}),pack_spare(~Element,spare))))),
          U32.from_nat(Nat.add(U32.to_nat(_),index))) == value(context,U32.from_nat(Nat.add(U32.to_nat(first),index))) : Element}
      band_assembled_cell(~Element,~Context,~value,context,Holding{behind,Held{buffer,start,span,below}},first,count,fill,spare,spare_room,low,high,capacity,capacity_ok,bounded,tiled,values,
        after,fits,allocatable,index,less)

def band_current_cell(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+holding: Holding<Element>,+first: U32,+count: U32,+fill: Element,
  +spare: SpareModel<Element>,+spare_room: SpareRoom(~Element,spare),
  +low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +tiled: Tiled(~Element,holding,low,high),+values: HoldingValues(~Element,~Context,~value,context,holding),+current: Current(~Element,holding,first,capacity),
  +after: {Nat.is_le(low,U32.to_nat(first)) == True{} : Bool},+fits: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),high) == True{} : Bool},
  +allocatable: {U32.is_le(count,1073741824) == True{} : Bool},
  +index: Nat,+less: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  BandRead(~Element,~Context,~value,context,Move.band_current(~Element,first,count,fill,pack(~Element,holding),pack_spare(~Element,spare)),first,index):
  match holding:
    case Holding{+behind,Held{+buffer,+start,+span,+below}}:
      band_when_cell(~Element,~Context,~value,context,U32.is_le(start,first) && U32.is_le((first + count : U32),(start + span : U32)),behind,buffer,start,span,below,first,count,fill,
        spare,spare_room,low,high,capacity,capacity_ok,bounded,tiled,values,current,after,fits,allocatable,{==},index,less)
    case Holding{behind,Bottom{}}:
      Empty.absurd(BandRead(~Element,~Context,~value,context,Move.band_current(~Element,first,count,fill,pack(~Element,Holding{behind,Bottom{}}),pack_spare(~Element,spare)),first,index),current)

# The band of [first, first + count) from a holding that covers a range
# holding it with the values of a tensor: lent or assembled, cell offset + k
# of its buffer is the value of cell first + k.
def band_value(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+holding: Holding<Element>,+first: U32,+count: U32,+fill: Element,
  +spare: SpareModel<Element>,+spare_room: SpareRoom(~Element,spare),
  +low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +tiled: Tiled(~Element,holding,low,high),+values: HoldingValues(~Element,~Context,~value,context,holding),
  +after: {Nat.is_le(low,U32.to_nat(first)) == True{} : Bool},+inside: {Nat.is_lt(U32.to_nat(first),high) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),high) == True{} : Bool},
  +allocatable: {U32.is_le(count,1073741824) == True{} : Bool},
  +index: Nat,+less: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  BandRead(~Element,~Context,~value,context,Move.band(~Element,pack(~Element,holding),first,count,fill,pack_spare(~Element,spare)),first,index):
  %Equal.sym(Move.Holding<Element>,Move.resolve(~Element,pack(~Element,holding),first),pack(~Element,resolve(~Element,holding,first)),resolve_packed(~Element,holding,first)) :
    BandRead(~Element,~Context,~value,context,Move.band_current(~Element,first,count,fill,_,pack_spare(~Element,spare)),first,index)
  band_current_cell(~Element,~Context,~value,context,resolve(~Element,holding,first),first,count,fill,spare,spare_room,low,high,capacity,capacity_ok,bounded,
    resolve_tiled(~Element,holding,first,low,high,tiled),resolve_values(~Element,~Context,~value,context,holding,first,values),
    resolve_holds(~Element,holding,first,low,high,capacity,capacity_ok,bounded,tiled,after,inside),after,fits,allocatable,index,less)

# The same cell of the model of the buffer of the band.
def band_model_value(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+holding: Holding<Element>,+first: U32,+count: U32,+fill: Element,
  +spare: SpareModel<Element>,+spare_room: SpareRoom(~Element,spare),
  +low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +tiled: Tiled(~Element,holding,low,high),+values: HoldingValues(~Element,~Context,~value,context,holding),
  +after: {Nat.is_le(low,U32.to_nat(first)) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),high) == True{} : Bool},
  +allocatable: {U32.is_le(count,1073741824) == True{} : Bool},
  +index: Nat,+less: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Element,Buffer.tree(~Element,Buffer.remodel(~Element,band_buffer(~Element,Move.band(~Element,pack(~Element,holding),first,count,fill,pack_spare(~Element,spare))))),
      U32.from_nat(Nat.add(U32.to_nat(band_offset(~Element,Move.band(~Element,pack(~Element,holding),first,count,fill,pack_spare(~Element,spare)))),index))) ==
    value(context,U32.from_nat(Nat.add(U32.to_nat(first),index))) : Element}:
  +start = U32.to_nat(first)
  +inside = Bounds.nat_trans(start,Nat.add(start,U32.to_nat(count)),high,
    Order.le_lt_trans(start,Nat.add(start,index),Nat.add(start,U32.to_nat(count)),Order.le_add(start,index),Order.offset_lt(start,index,U32.to_nat(count),less)),fits)
  %Equal.trans(Tree.Tree<Element>,Buffer.content_tree(~Element,Buffer.contents(~Element,band_buffer(~Element,Move.band(~Element,pack(~Element,holding),first,count,fill,pack_spare(~Element,spare))))),
    Buffer.content_tree(~Element,Buffer.content(~Element,Buffer.remodel(~Element,band_buffer(~Element,Move.band(~Element,pack(~Element,holding),first,count,fill,pack_spare(~Element,spare)))))),
    Buffer.tree(~Element,Buffer.remodel(~Element,band_buffer(~Element,Move.band(~Element,pack(~Element,holding),first,count,fill,pack_spare(~Element,spare))))),
    Equal.cong(Buffer.Content<Element>,Tree.Tree<Element>,Buffer.content_tree(~Element),Buffer.contents(~Element,band_buffer(~Element,Move.band(~Element,pack(~Element,holding),first,count,fill,pack_spare(~Element,spare)))),
      Buffer.content(~Element,Buffer.remodel(~Element,band_buffer(~Element,Move.band(~Element,pack(~Element,holding),first,count,fill,pack_spare(~Element,spare))))),
      Equal.sym(Buffer.Content<Element>,Buffer.content(~Element,Buffer.remodel(~Element,band_buffer(~Element,Move.band(~Element,pack(~Element,holding),first,count,fill,pack_spare(~Element,spare))))),Buffer.contents(~Element,band_buffer(~Element,Move.band(~Element,pack(~Element,holding),first,count,fill,pack_spare(~Element,spare)))),
        Buffer.remodel_content(~Element,band_buffer(~Element,Move.band(~Element,pack(~Element,holding),first,count,fill,pack_spare(~Element,spare)))))),
    Buffer.content_tree_of(~Element,Buffer.remodel(~Element,band_buffer(~Element,Move.band(~Element,pack(~Element,holding),first,count,fill,pack_spare(~Element,spare)))))) :
    {Tree.read(Element,_,U32.from_nat(Nat.add(U32.to_nat(band_offset(~Element,Move.band(~Element,pack(~Element,holding),first,count,fill,pack_spare(~Element,spare)))),index))) == value(context,U32.from_nat(Nat.add(start,index))) : Element}
  band_value(~Element,~Context,~value,context,holding,first,count,fill,spare,spare_room,low,high,capacity,capacity_ok,bounded,tiled,values,after,inside,fits,allocatable,index,less)

# A unit run resolves only: its holding covers the same range with the same values.
def run_tiled(~Element: Data,fuel: Nat,+done: Bool,+count: U32,+offset: U32,+stride: U32,+back: Bool,+position: U32,+holding: Holding<Element>,+tree: Tree.Tree<Element>,+low: Nat,+high: Nat,
  +tiled: Tiled(~Element,holding,low,high)) -> Tiled(~Element,ran_holding(~Element,model_run(~Element,fuel,done,count,offset,stride,back,position,holding,tree)),low,high):
  match fuel done:
    case 0n done: tiled
    case 1n+rest True{}: tiled
    case 1n+ +rest False{}:
      +resolved = resolve(~Element,holding,offset)
      +cells = step_cells(~Element,resolved,count,offset,stride,back)
      run_tiled(~Element,rest,U32.is_zero((count - cells : U32)),(count - cells : U32),(offset + cells * Move.directed(back,stride) : U32),stride,back,(position + cells : U32),resolved,
        step_tree(~Element,resolved,count,offset,stride,back,position,tree),low,high,resolve_tiled(~Element,holding,offset,low,high,tiled))

def run_values(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,fuel: Nat,+done: Bool,+count: U32,+offset: U32,+stride: U32,+back: Bool,+position: U32,
  +holding: Holding<Element>,+tree: Tree.Tree<Element>,+values: HoldingValues(~Element,~Context,~value,context,holding)) ->
  HoldingValues(~Element,~Context,~value,context,ran_holding(~Element,model_run(~Element,fuel,done,count,offset,stride,back,position,holding,tree))):
  match fuel done:
    case 0n done: values
    case 1n+rest True{}: values
    case 1n+ +rest False{}:
      +resolved = resolve(~Element,holding,offset)
      +cells = step_cells(~Element,resolved,count,offset,stride,back)
      run_values(~Element,~Context,~value,context,rest,U32.is_zero((count - cells : U32)),(count - cells : U32),(offset + cells * Move.directed(back,stride) : U32),stride,back,(position + cells : U32),
        resolved,step_tree(~Element,resolved,count,offset,stride,back,position,tree),resolve_values(~Element,~Context,~value,context,holding,offset,values))

def run_order(~Element: Data,fuel: Nat,+done: Bool,+count: U32,+offset: U32,+stride: U32,+back: Bool,+position: U32,+holding: Holding<Element>,+tree: Tree.Tree<Element>) ->
  {order(~Element,ran_holding(~Element,model_run(~Element,fuel,done,count,offset,stride,back,position,holding,tree))) == order(~Element,holding) : Stack<Element>}:
  match fuel done:
    case 0n done: {==}
    case 1n+rest True{}: {==}
    case 1n+ +rest False{}:
      +resolved = resolve(~Element,holding,offset)
      +cells = step_cells(~Element,resolved,count,offset,stride,back)
      Equal.trans(Stack<Element>,
        order(~Element,ran_holding(~Element,model_run(~Element,rest,U32.is_zero((count - cells : U32)),(count - cells : U32),(offset + cells * Move.directed(back,stride) : U32),stride,back,
          (position + cells : U32),resolved,step_tree(~Element,resolved,count,offset,stride,back,position,tree)))),
        order(~Element,resolved),order(~Element,holding),
        run_order(~Element,rest,U32.is_zero((count - cells : U32)),(count - cells : U32),(offset + cells * Move.directed(back,stride) : U32),stride,back,(position + cells : U32),resolved,
          step_tree(~Element,resolved,count,offset,stride,back,position,tree)),
        resolve_order(~Element,holding,offset))

# Pieces of the same ranges onto the same pieces behind keep the same ranges.
# ---- Release ----

def onto_ahead_packed(~Element: Data,+behind: Stack<Element>,+ahead: Stack<Element>) ->
  {Move.onto_ahead(~Element,pack_stack(~Element,behind),pack_stack(~Element,ahead)) == pack_stack(~Element,onto(~Element,behind,ahead)) : Move.Stack<Element>}:
  match behind:
    case Bottom{}: {==}
    case Held{+buffer,+first,+count,+below}: onto_ahead_packed(~Element,below,Held{buffer,first,count,ahead})

def appended(~Element: Data,stack: Stack<Element>,rest: Stack<Element>) -> Stack<Element>:
  match stack:
    case Bottom{}: rest
    case Held{buffer,first,count,below}: Held{buffer,first,count,appended(~Element,below,rest)}

def appended_bottom(~Element: Data,+stack: Stack<Element>) -> {appended(~Element,stack,Bottom{}) == stack : Stack<Element>}:
  match stack:
    case Bottom{}: {==}
    case Held{+buffer,+first,+count,+below}:
      Equal.cong(Stack<Element>,Stack<Element>,rest => Held{buffer,first,count,rest},appended(~Element,below,Bottom{}),below,appended_bottom(~Element,below))

def stacked_unstacked(~Element: Data,+stack: Stack<Element>,tree: Pieces.Pieces(~Element),+rest: Stack<Element>) ->
  {Move.stacked(~Element,Move.unstacked(~Element,pack_stack(~Element,stack),tree),pack_stack(~Element,rest)) ==
    Move.stacked(~Element,tree,pack_stack(~Element,appended(~Element,stack,rest))) : Move.Stack<Element>}:
  match stack:
    case Bottom{}: {==}
    case Held{+buffer,+first,+count,+below}:
      stacked_unstacked(~Element,below,Partition.Fork{tree,Partition.Leaf{Buffer.pack(~Element,buffer),first,count,Unit{}}},rest)

def spanned(ranges: Spans) -> Bool:
  match ranges:
    case Ended{}: False{}
    case Spanned{first,count,rest}: True{}

# A stack of pieces released into a tree and stacked again.
def stack_released(~Element: Data,+stack: Stack<Element>,+fill: Element,+nonempty: {spanned(spans(~Element,stack)) == True{} : Bool}) ->
  {Move.stacked(~Element,Move.released_stack(~Element,Move.first_leaf(~Element,pack_stack(~Element,stack),fill)),Move.Bottom{}) == pack_stack(~Element,stack) : Move.Stack<Element>}:
  match stack:
    case Held{+buffer,+first,+count,+below}:
      Equal.trans(Move.Stack<Element>,
        Move.stacked(~Element,Move.unstacked(~Element,pack_stack(~Element,below),Partition.Leaf{Buffer.pack(~Element,buffer),first,count,Unit{}}),pack_stack(~Element,Bottom{})),
        pack_stack(~Element,Held{buffer,first,count,appended(~Element,below,Bottom{})}),pack_stack(~Element,Held{buffer,first,count,below}),
        stacked_unstacked(~Element,below,Partition.Leaf{Buffer.pack(~Element,buffer),first,count,Unit{}},Bottom{}),
        Equal.cong(Stack<Element>,Move.Stack<Element>,rest => pack_stack(~Element,Held{buffer,first,count,rest}),appended(~Element,below,Bottom{}),below,appended_bottom(~Element,below)))
    case Bottom{}:
      Empty.absurd({Move.stacked(~Element,Move.released_stack(~Element,Move.first_leaf(~Element,pack_stack(~Element,Bottom{}),fill)),Move.Bottom{}) == pack_stack(~Element,Bottom{}) : Move.Stack<Element>},
        WordEncoding.false_true(nonempty))

# The pieces of a holding released and stacked again are its pieces in order.
def released_stacked(~Element: Data,+holding: Holding<Element>,+fill: Element,+nonempty: {spanned(spans(~Element,order(~Element,holding))) == True{} : Bool}) ->
  {Move.stacked(~Element,Move.released(~Element,pack(~Element,holding),fill),Move.Bottom{}) == pack_stack(~Element,order(~Element,holding)) : Move.Stack<Element>}:
  match holding:
    case Holding{+behind,+ahead}:
      %Equal.sym(Move.Stack<Element>,Move.onto_ahead(~Element,pack_stack(~Element,behind),pack_stack(~Element,ahead)),pack_stack(~Element,onto(~Element,behind,ahead)),
        onto_ahead_packed(~Element,behind,ahead)) :
        {Move.stacked(~Element,Move.released_stack(~Element,Move.first_leaf(~Element,_,fill)),Move.Bottom{}) == pack_stack(~Element,onto(~Element,behind,ahead)) : Move.Stack<Element>}
      stack_released(~Element,onto(~Element,behind,ahead),fill,nonempty)

# The pieces of a holding released are those of its pieces in order.
def released_order(~Element: Data,+holding: Holding<Element>,+fill: Element) ->
  {Move.released(~Element,pack(~Element,holding),fill) == Move.released_stack(~Element,Move.first_leaf(~Element,pack_stack(~Element,order(~Element,holding)),fill)) : Pieces.Pieces(~Element)}:
  match holding:
    case Holding{+behind,+ahead}:
      Equal.cong(Move.Stack<Element>,Pieces.Pieces(~Element),stack => Move.released_stack(~Element,Move.first_leaf(~Element,stack,fill)),
        Move.onto_ahead(~Element,pack_stack(~Element,behind),pack_stack(~Element,ahead)),pack_stack(~Element,onto(~Element,behind,ahead)),onto_ahead_packed(~Element,behind,ahead))

# The pieces of a stack after a tree, in order, as Move.unstacked joins them.
def unstacked_model(~Element: Data,stack: Stack<Element>,tree: PiecesModel.Model<Element>) -> PiecesModel.Model<Element>:
  match stack:
    case Bottom{}: tree
    case Held{buffer,+first,+count,below}: unstacked_model(~Element,below,PiecesModel.Fork{tree,PiecesModel.Leaf{buffer,first,count}})

def unstacked_packed(~Element: Data,+stack: Stack<Element>,+tree: PiecesModel.Model<Element>) ->
  {Move.unstacked(~Element,pack_stack(~Element,stack),PiecesModel.pack(~Element,tree)) == PiecesModel.pack(~Element,unstacked_model(~Element,stack,tree)) : Pieces.Pieces(~Element)}:
  match stack:
    case Bottom{}: {==}
    case Held{+buffer,+first,+count,+below}: unstacked_packed(~Element,below,PiecesModel.Fork{tree,PiecesModel.Leaf{buffer,first,count}})

# ---- Covered holdings ----

# A runtime holding that is the pack of a model whose pieces cover [low, high)
# with the values of a tensor, the ranges of its pieces in order layout.
type Covered<-Element: Data,-Context: Data,-value: Context -> U32 -> Element,-context: Context,-holding: Move.Holding<Element>,-low: Nat,-high: Nat,-layout: Spans> is Data:
  Covered{model: Holding<Element>,packed: {holding == pack(~Element,model) : Move.Holding<Element>},tiled: Tiled(~Element,model,low,high),
    values: HoldingValues(~Element,~Context,~value,context,model),ordered: {spans(~Element,order(~Element,model)) == layout : Spans}}

def CoveredFact(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,holding: Move.Holding<Element>,+low: Nat,+high: Nat,+layout: Spans) -> Data:
  Covered<Element,Context,value,context,holding,low,high,layout>

# The model of the holding after a copy of [first, first + count) into a
# target.
def copied_into_holding(~Element: Data,+holding: Holding<Element>,+first: U32,+count: U32,+target: Buffer.Model<Element>) -> Holding<Element>:
  ran_holding(~Element,model_run(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,Buffer.tree(~Element,target)))

def copied_into_held(~Element: Data,+holding: Holding<Element>,+first: U32,+count: U32,+target: Buffer.Model<Element>) ->
  {Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied_into(~Element,pack(~Element,holding),first,count,Buffer.pack(~Element,target))) ==
    pack(~Element,copied_into_holding(~Element,holding,first,count,target)) : Move.Holding<Element>}:
  run_held(~Element,Move.run(~Element,U32.to_nat(count),count,first,1,False{},0,pack(~Element,holding),Buffer.pack(~Element,target)),
    model_run(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,Buffer.tree(~Element,target)),Buffer.length(~Element,target),
    run_refines(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,target))

# The holding after a copy covers the same range with the same values.
def copied_into_covered(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+holding: Holding<Element>,+first: U32,+count: U32,
  +target: Buffer.Model<Element>,+low: Nat,+high: Nat,+tiled: Tiled(~Element,holding,low,high),+values: HoldingValues(~Element,~Context,~value,context,holding),
  +layout: Spans,+ordered: {spans(~Element,order(~Element,holding)) == layout : Spans}) ->
  CoveredFact(~Element,~Context,~value,context,Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied_into(~Element,pack(~Element,holding),first,count,Buffer.pack(~Element,target))),
    low,high,layout):
  +tree = Buffer.tree(~Element,target)
  +after = copied_into_holding(~Element,holding,first,count,target)
  Covered{after,copied_into_held(~Element,holding,first,count,target),
    run_tiled(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,tree,low,high,tiled),
    run_values(~Element,~Context,~value,context,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,tree,values),
    Equal.trans(Spans,spans(~Element,order(~Element,after)),spans(~Element,order(~Element,holding)),layout,
      Equal.cong(Stack<Element>,Spans,spans(~Element),order(~Element,after),order(~Element,holding),
        run_order(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,tree)),ordered)}

# The model of the holding after a copy of [first, first + count).
def copied_holding(~Element: Data,+holding: Holding<Element>,+first: U32,+count: U32,+fill: Element) -> Holding<Element>:
  copied_into_holding(~Element,holding,first,count,Buffer.remodel(~Element,Allocation.allocate(~Element,Storage.depth_for(count),fill,U32.to_nat(count))))

def copied_held(~Element: Data,+holding: Holding<Element>,+first: U32,+count: U32,+fill: Element) ->
  {Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)) == pack(~Element,copied_holding(~Element,holding,first,count,fill)) : Move.Holding<Element>}:
  %Buffer.remodeled(~Element,Allocation.allocate(~Element,Storage.depth_for(count),fill,U32.to_nat(count))) :
    {Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied_into(~Element,pack(~Element,holding),first,count,_)) == pack(~Element,copied_holding(~Element,holding,first,count,fill)) :
      Move.Holding<Element>}
  copied_into_held(~Element,holding,first,count,Buffer.remodel(~Element,Allocation.allocate(~Element,Storage.depth_for(count),fill,U32.to_nat(count))))

# The holding after a copy covers the same range with the same values.
def copied_covered(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+holding: Holding<Element>,+first: U32,+count: U32,+fill: Element,
  +low: Nat,+high: Nat,+tiled: Tiled(~Element,holding,low,high),+values: HoldingValues(~Element,~Context,~value,context,holding),
  +layout: Spans,+ordered: {spans(~Element,order(~Element,holding)) == layout : Spans}) ->
  CoveredFact(~Element,~Context,~value,context,Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)),low,high,layout):
  %Buffer.remodeled(~Element,Allocation.allocate(~Element,Storage.depth_for(count),fill,U32.to_nat(count))) :
    CoveredFact(~Element,~Context,~value,context,Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied_into(~Element,pack(~Element,holding),first,count,_)),low,high,layout)
  copied_into_covered(~Element,~Context,~value,context,holding,first,count,
    Buffer.remodel(~Element,Allocation.allocate(~Element,Storage.depth_for(count),fill,U32.to_nat(count))),low,high,tiled,values,layout,ordered)

def band_loan(~Element: Data,band: Move.Band<Element>) -> Move.Lent<Element>:
  Move.Band{lent,buffer,offset} = band
  lent

# Two buffer models of the same contents hold the same tree and length.
def same_tree(~Element: Data,+left: Buffer.Model<Element>,+right: Buffer.Model<Element>,
  +same: {Buffer.content(~Element,left) == Buffer.contents(~Element,Buffer.pack(~Element,right)) : Buffer.Content<Element>}) ->
  {Buffer.tree(~Element,left) == Buffer.tree(~Element,right) : Tree.Tree<Element>}:
  Equal.trans(Tree.Tree<Element>,Buffer.tree(~Element,left),Buffer.content_tree(~Element,Buffer.content(~Element,left)),Buffer.tree(~Element,right),
    Equal.sym(Tree.Tree<Element>,Buffer.content_tree(~Element,Buffer.content(~Element,left)),Buffer.tree(~Element,left),Buffer.content_tree_of(~Element,left)),
    Equal.cong(Buffer.Content<Element>,Tree.Tree<Element>,Buffer.content_tree(~Element),Buffer.content(~Element,left),Buffer.Content{Buffer.tree(~Element,right),Buffer.length(~Element,right)},
      Equal.trans(Buffer.Content<Element>,Buffer.content(~Element,left),Buffer.contents(~Element,Buffer.pack(~Element,right)),
        Buffer.Content{Buffer.tree(~Element,right),Buffer.length(~Element,right)},same,content_parts(~Element,right))))

def same_length(~Element: Data,+left: Buffer.Model<Element>,+right: Buffer.Model<Element>,
  +same: {Buffer.content(~Element,left) == Buffer.contents(~Element,Buffer.pack(~Element,right)) : Buffer.Content<Element>}) ->
  {Buffer.length(~Element,left) == Buffer.length(~Element,right) : Nat}:
  Equal.trans(Nat,Buffer.length(~Element,left),Buffer.content_length(~Element,Buffer.content(~Element,left)),Buffer.length(~Element,right),
    Equal.sym(Nat,Buffer.content_length(~Element,Buffer.content(~Element,left)),Buffer.length(~Element,left),Buffer.content_length_of(~Element,left)),
    Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),Buffer.content(~Element,left),Buffer.Content{Buffer.tree(~Element,right),Buffer.length(~Element,right)},
      Equal.trans(Buffer.Content<Element>,Buffer.content(~Element,left),Buffer.contents(~Element,Buffer.pack(~Element,right)),
        Buffer.Content{Buffer.tree(~Element,right),Buffer.length(~Element,right)},same,content_parts(~Element,right))))

# A piece holds its values in any buffer of the same contents.
# A buffer with the contents of a ready buffer is ready.
def same_ready(~Element: Data,+left: Buffer.Model<Element>,+right: Buffer.Model<Element>,
  +same: {Buffer.content(~Element,left) == Buffer.contents(~Element,Buffer.pack(~Element,right)) : Buffer.Content<Element>},
  +ready: Buffer.ContentReady<Element,Buffer.content(~Element,right)>) -> Buffer.ContentReady<Element,Buffer.content(~Element,left)>:
  %Equal.sym(Buffer.Content<Element>,Buffer.content(~Element,left),Buffer.content(~Element,right),
    Equal.trans(Buffer.Content<Element>,Buffer.content(~Element,left),Buffer.contents(~Element,Buffer.pack(~Element,right)),Buffer.content(~Element,right),
      same,Buffer.packed_contents(~Element,right))) : Buffer.ContentReady<Element,_>
  ready

def leaf_moved(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+first: U32,+count: U32,+left: Buffer.Model<Element>,+right: Buffer.Model<Element>,
  +same: {Buffer.content(~Element,left) == Buffer.contents(~Element,Buffer.pack(~Element,right)) : Buffer.Content<Element>},
  +piece: Content.Leaf(~Element,~Context,~value,context,first,count,Buffer.tree(~Element,right),Buffer.length(~Element,right))) ->
  Content.Leaf(~Element,~Context,~value,context,first,count,Buffer.tree(~Element,left),Buffer.length(~Element,left)):
  %Equal.sym(Tree.Tree<Element>,Buffer.tree(~Element,left),Buffer.tree(~Element,right),same_tree(~Element,left,right,same)) :
    Content.Leaf(~Element,~Context,~value,context,first,count,_,Buffer.length(~Element,left))
  %Equal.sym(Nat,Buffer.length(~Element,left),Buffer.length(~Element,right),same_length(~Element,left,right,same)) :
    Content.Leaf(~Element,~Context,~value,context,first,count,Buffer.tree(~Element,right),_)
  piece

# After an assembled band the holding covers the same range with the same values.
def assembled_returned(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+holding: Holding<Element>,+first: U32,+count: U32,+fill: Element,
  +spare: SpareModel<Element>,+low: Nat,+high: Nat,+tiled: Tiled(~Element,holding,low,high),+values: HoldingValues(~Element,~Context,~value,context,holding),
  +layout: Spans,+ordered: {spans(~Element,order(~Element,holding)) == layout : Spans},buffer: Storage.Buffer<Element>) ->
  CoveredFact(~Element,~Context,~value,context,
    Move.returned_holding(~Element,band_loan(~Element,Move.assembled(~Element,first,count,fill,pack(~Element,holding),pack_spare(~Element,spare))),buffer),low,high,layout):
  SpareModel{+model,+cells} = spare
  +enough = U32.is_le(count,cells)
  +target = roomy_model(~Element,enough,count,fill,model)
  %Equal.sym(Storage.Buffer<Element>,Move.roomy(~Element,enough,count,fill,Buffer.pack(~Element,model)),Buffer.pack(~Element,target),roomy_packed(~Element,enough,count,fill,model)) :
    CoveredFact(~Element,~Context,~value,context,Move.returned_holding(~Element,band_loan(~Element,Move.assembled_band(~Element,U32.max(cells,count),
      Move.copied_into(~Element,pack(~Element,holding),first,count,_))),buffer),low,high,layout)
  %Equal.sym(Move.Holding<Element> & Storage.Buffer<Element>,Move.copied_into(~Element,pack(~Element,holding),first,count,Buffer.pack(~Element,target)),
    (Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied_into(~Element,pack(~Element,holding),first,count,Buffer.pack(~Element,target))),
      Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied_into(~Element,pack(~Element,holding),first,count,Buffer.pack(~Element,target)))),
    pair_eta(~Move.Holding<Element>,~Storage.Buffer<Element>,Move.copied_into(~Element,pack(~Element,holding),first,count,Buffer.pack(~Element,target)))) :
    CoveredFact(~Element,~Context,~value,context,Move.returned_holding(~Element,band_loan(~Element,Move.assembled_band(~Element,U32.max(cells,count),_)),buffer),low,high,layout)
  copied_into_covered(~Element,~Context,~value,context,holding,first,count,target,low,high,tiled,values,layout,ordered)

def band_when_returned(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+within: Bool,+behind: Stack<Element>,+buffer: Buffer.Model<Element>,
  +start: U32,+span: U32,+below: Stack<Element>,+first: U32,+count: U32,+fill: Element,+spare: SpareModel<Element>,+low: Nat,+high: Nat,
  +tiled: Tiled(~Element,Holding{behind,Held{buffer,start,span,below}},low,high),+values: HoldingValues(~Element,~Context,~value,context,Holding{behind,Held{buffer,start,span,below}}),
  +layout: Spans,+ordered: {spans(~Element,onto(~Element,behind,Held{buffer,start,span,below})) == layout : Spans},+returned: Buffer.Model<Element>,
  +same: {Buffer.content(~Element,returned) == Buffer.contents(~Element,band_buffer(~Element,
    Move.band_when(~Element,within,first,count,fill,pack_stack(~Element,behind),Buffer.pack(~Element,buffer),start,span,pack_stack(~Element,below),pack_spare(~Element,spare)))) :
    Buffer.Content<Element>}) ->
  CoveredFact(~Element,~Context,~value,context,Move.returned_holding(~Element,band_loan(~Element,
    Move.band_when(~Element,within,first,count,fill,pack_stack(~Element,behind),Buffer.pack(~Element,buffer),start,span,pack_stack(~Element,below),pack_spare(~Element,spare))),
    Buffer.pack(~Element,returned)),low,high,layout):
  match within:
    case True{}:
      Tiling{+mid,+earlier,Both{Starts{+start_at,+sized,+ready},+later}} = tiled
      Both{+earlier_values,Both{+piece,+later_values}} = values
      Covered{Holding{behind,Held{returned,start,span,below}},{==},
        Tiling{mid,earlier,Both{Starts{start_at,Equal.trans(Nat,Buffer.length(~Element,returned),Buffer.length(~Element,buffer),U32.to_nat(span),
          same_length(~Element,returned,buffer,same),sized),same_ready(~Element,returned,buffer,same,ready)},later}},
        Both{earlier_values,Both{leaf_moved(~Element,~Context,~value,context,start,span,returned,buffer,same,piece),later_values}},
        Equal.trans(Spans,spans(~Element,onto(~Element,behind,Held{returned,start,span,below})),spans(~Element,onto(~Element,behind,Held{buffer,start,span,below})),layout,
          onto_spans(~Element,behind,Held{returned,start,span,below},Held{buffer,start,span,below},{==}),ordered)}
    case False{}:
      assembled_returned(~Element,~Context,~value,context,Holding{behind,Held{buffer,start,span,below}},first,count,fill,spare,low,high,tiled,values,layout,ordered,
        Buffer.pack(~Element,returned))

def band_current_returned(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+holding: Holding<Element>,+first: U32,+count: U32,+fill: Element,
  +spare: SpareModel<Element>,+low: Nat,+high: Nat,+tiled: Tiled(~Element,holding,low,high),+values: HoldingValues(~Element,~Context,~value,context,holding),
  +layout: Spans,+ordered: {spans(~Element,order(~Element,holding)) == layout : Spans},+returned: Buffer.Model<Element>,
  +same: {Buffer.content(~Element,returned) == Buffer.contents(~Element,band_buffer(~Element,Move.band_current(~Element,first,count,fill,pack(~Element,holding),pack_spare(~Element,spare)))) :
    Buffer.Content<Element>}) ->
  CoveredFact(~Element,~Context,~value,context,Move.returned_holding(~Element,band_loan(~Element,Move.band_current(~Element,first,count,fill,pack(~Element,holding),pack_spare(~Element,spare))),
    Buffer.pack(~Element,returned)),low,high,layout):
  match holding:
    case Holding{+behind,Held{+buffer,+start,+span,+below}}:
      band_when_returned(~Element,~Context,~value,context,U32.is_le(start,first) && U32.is_le((first + count : U32),(start + span : U32)),behind,buffer,start,span,below,
        first,count,fill,spare,low,high,tiled,values,layout,ordered,returned,same)
    case Holding{+behind,Bottom{}}:
      assembled_returned(~Element,~Context,~value,context,Holding{behind,Bottom{}},first,count,fill,spare,low,high,tiled,values,layout,ordered,Buffer.pack(~Element,returned))

# After a band, with its buffer back unchanged in contents, the holding covers
# the same range with the same values.
def band_returned(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+holding: Holding<Element>,+first: U32,+count: U32,+fill: Element,
  +spare: SpareModel<Element>,+low: Nat,+high: Nat,+tiled: Tiled(~Element,holding,low,high),+values: HoldingValues(~Element,~Context,~value,context,holding),
  +layout: Spans,+ordered: {spans(~Element,order(~Element,holding)) == layout : Spans},+returned: Buffer.Model<Element>,
  +same: {Buffer.content(~Element,returned) == Buffer.contents(~Element,band_buffer(~Element,Move.band(~Element,pack(~Element,holding),first,count,fill,pack_spare(~Element,spare)))) :
    Buffer.Content<Element>}) ->
  CoveredFact(~Element,~Context,~value,context,Move.returned_holding(~Element,band_loan(~Element,Move.band(~Element,pack(~Element,holding),first,count,fill,pack_spare(~Element,spare))),
    Buffer.pack(~Element,returned)),low,high,layout):
  %Equal.sym(Move.Holding<Element>,Move.resolve(~Element,pack(~Element,holding),first),pack(~Element,resolve(~Element,holding,first)),resolve_packed(~Element,holding,first)) :
    CoveredFact(~Element,~Context,~value,context,Move.returned_holding(~Element,band_loan(~Element,Move.band_current(~Element,first,count,fill,_,pack_spare(~Element,spare))),
      Buffer.pack(~Element,returned)),low,high,layout)
  band_current_returned(~Element,~Context,~value,context,resolve(~Element,holding,first),first,count,fill,spare,low,high,
    resolve_tiled(~Element,holding,first,low,high,tiled),resolve_values(~Element,~Context,~value,context,holding,first,values),layout,
    Equal.trans(Spans,spans(~Element,order(~Element,resolve(~Element,holding,first))),spans(~Element,order(~Element,holding)),layout,
      Equal.cong(Stack<Element>,Spans,spans(~Element),order(~Element,resolve(~Element,holding,first)),order(~Element,holding),resolve_order(~Element,holding,first)),ordered),returned,
    Equal.trans(Buffer.Content<Element>,Buffer.content(~Element,returned),Buffer.contents(~Element,band_buffer(~Element,Move.band(~Element,pack(~Element,holding),first,count,fill,pack_spare(~Element,spare)))),
      Buffer.contents(~Element,band_buffer(~Element,Move.band_current(~Element,first,count,fill,pack(~Element,resolve(~Element,holding,first)),pack_spare(~Element,spare)))),same,
      Equal.cong(Move.Holding<Element>,Buffer.Content<Element>,resolved => Buffer.contents(~Element,band_buffer(~Element,Move.band_current(~Element,first,count,fill,resolved,pack_spare(~Element,spare)))),
        Move.resolve(~Element,pack(~Element,holding),first),pack(~Element,resolve(~Element,holding,first)),resolve_packed(~Element,holding,first))))

# ---- The pieces of a tensor held ----

# The pieces of a model tree onto a stack, in order.
def stacked_model(~Element: Data,model: PiecesModel.Model<Element>,rest: Stack<Element>) -> Stack<Element>:
  match model:
    case PiecesModel.Leaf{buffer,first,rows}: Held{buffer,first,rows,rest}
    case PiecesModel.Fork{left,right}: stacked_model(~Element,left,stacked_model(~Element,right,rest))

def stacked_packed(~Element: Data,+model: PiecesModel.Model<Element>,+rest: Stack<Element>) ->
  {Move.stacked(~Element,PiecesModel.pack(~Element,model),pack_stack(~Element,rest)) == pack_stack(~Element,stacked_model(~Element,model,rest)) : Move.Stack<Element>}:
  match model:
    case PiecesModel.Leaf{buffer,first,rows}: {==}
    case PiecesModel.Fork{+left,+right}:
      %Equal.sym(Move.Stack<Element>,Move.stacked(~Element,PiecesModel.pack(~Element,right),pack_stack(~Element,rest)),pack_stack(~Element,stacked_model(~Element,right,rest)),
        stacked_packed(~Element,right,rest)) :
        {Move.stacked(~Element,PiecesModel.pack(~Element,left),_) == pack_stack(~Element,stacked_model(~Element,left,stacked_model(~Element,right,rest))) : Move.Stack<Element>}
      stacked_packed(~Element,left,stacked_model(~Element,right,rest))

# Pieces that cover [low, low + rows) in order, onto a stack that covers the
# cells after them, cover the cells from low.
def stacked_tiles(~Element: Data,+model: PiecesModel.Model<Element>,+rest: Stack<Element>,+low: Nat,+rows: Nat,+high: Nat,+total: Nat,
  +coverage: PiecesPlan.Coverage(PiecesModel.content_layout(~Element,PiecesModel.content(~Element,model)),low,rows),
  +fields: Content.Contents(~Element,total,PiecesModel.content(~Element,model)),+ready: PiecesModel.ContentReady(~Element,PiecesModel.content(~Element,model)),
  +after: Tiles(~Element,rest,Nat.add(low,rows),high)) ->
  Tiles(~Element,stacked_model(~Element,model,rest),low,high):
  match model:
    case PiecesModel.Leaf{Buffer.Model{+tree,+length,+certificate},+first,+count}:
      PiecesPlan.LeafCoverage{+first_equal,+rows_equal} = coverage
      Content.Fields{+length_exact,+inside} = fields
      Both{Starts{first_equal,length_exact,ready},tiles_moved(~Element,rest,Nat.add(low,rows),Nat.add(low,U32.to_nat(count)),high,
        Equal.cong(Nat,Nat,value => Nat.add(low,value),rows,U32.to_nat(count),Equal.sym(Nat,U32.to_nat(count),rows,rows_equal)),after)}
    case PiecesModel.Fork{+left,+right}:
      PiecesPlan.ForkCoverage{+left_rows,+right_rows,+rows_equal,+left_coverage,+right_coverage} = coverage
      KernelGemm.Both{+left_fields,+right_fields} = fields
      KernelGemm.Both{+left_ready,+right_ready} = ready
      stacked_tiles(~Element,left,stacked_model(~Element,right,rest),low,left_rows,high,total,left_coverage,left_fields,left_ready,
        stacked_tiles(~Element,right,rest,Nat.add(low,left_rows),right_rows,high,total,right_coverage,right_fields,right_ready,
          tiles_moved(~Element,rest,Nat.add(low,rows),Nat.add(Nat.add(low,left_rows),right_rows),high,
            Equal.trans(Nat,Nat.add(low,rows),Nat.add(low,Nat.add(left_rows,right_rows)),Nat.add(Nat.add(low,left_rows),right_rows),
              Equal.cong(Nat,Nat,value => Nat.add(low,value),rows,Nat.add(left_rows,right_rows),rows_equal),
              Equal.sym(Nat,Nat.add(Nat.add(low,left_rows),right_rows),Nat.add(low,Nat.add(left_rows,right_rows)),Addition.associative(low,left_rows,right_rows))),
            after)))

def stacked_values(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+model: PiecesModel.Model<Element>,+rest: Stack<Element>,
  +values: Content.Values(~Element,~Context,~value,context,PiecesModel.content(~Element,model)),+after: StackValues(~Element,~Context,~value,context,rest)) ->
  StackValues(~Element,~Context,~value,context,stacked_model(~Element,model,rest)):
  match model:
    case PiecesModel.Leaf{Buffer.Model{+tree,+length,+certificate},+first,+count}: Both{values,after}
    case PiecesModel.Fork{+left,+right}:
      KernelGemm.Both{+left_values,+right_values} = values
      stacked_values(~Element,~Context,~value,context,left,stacked_model(~Element,right,rest),left_values,
        stacked_values(~Element,~Context,~value,context,right,rest,right_values,after))

# The ranges of a layout in order onto ranges after them.
def layout_spans(layout: Pieces.Ranges,rest: Spans) -> Spans:
  match layout:
    case Pieces.Range{first,count}: Spanned{first,count,rest}
    case Pieces.Split{left,right}: layout_spans(left,layout_spans(right,rest))

def stacked_spans(~Element: Data,+model: PiecesModel.Model<Element>,+rest: Stack<Element>) ->
  {spans(~Element,stacked_model(~Element,model,rest)) == layout_spans(PiecesModel.content_layout(~Element,PiecesModel.content(~Element,model)),spans(~Element,rest)) : Spans}:
  match model:
    case PiecesModel.Leaf{buffer,first,rows}: {==}
    case PiecesModel.Fork{+left,+right}:
      Equal.trans(Spans,spans(~Element,stacked_model(~Element,left,stacked_model(~Element,right,rest))),
        layout_spans(PiecesModel.content_layout(~Element,PiecesModel.content(~Element,left)),spans(~Element,stacked_model(~Element,right,rest))),
        layout_spans(PiecesModel.content_layout(~Element,PiecesModel.content(~Element,left)),
          layout_spans(PiecesModel.content_layout(~Element,PiecesModel.content(~Element,right)),spans(~Element,rest))),
        stacked_spans(~Element,left,stacked_model(~Element,right,rest)),
        Equal.cong(Spans,Spans,after => layout_spans(PiecesModel.content_layout(~Element,PiecesModel.content(~Element,left)),after),
          spans(~Element,stacked_model(~Element,right,rest)),layout_spans(PiecesModel.content_layout(~Element,PiecesModel.content(~Element,right)),spans(~Element,rest)),
          stacked_spans(~Element,right,rest)))

# The pieces of a tensor held, the first current: they cover all its cells
# with its values, in the order of its layout.
def held_covered(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+model: PiecesModel.Model<Element>,+total: Nat,
  +formed: Content.WellFormed<Element,total,PiecesModel.content(~Element,model)>,+values: Content.Values(~Element,~Context,~value,context,PiecesModel.content(~Element,model))) ->
  Covered<Element,Context,value,context,Move.held(~Element,PiecesModel.pack(~Element,model)),0n,total,
    layout_spans(PiecesModel.content_layout(~Element,PiecesModel.content(~Element,model)),Ended{})>:
  Content.WellFormed{+fields,+ready,+coverage} = formed
  Covered{Holding{Bottom{},stacked_model(~Element,model,Bottom{})},
    Equal.cong(Move.Stack<Element>,Move.Holding<Element>,stack => Move.Holding{Move.Bottom{},stack},Move.stacked(~Element,PiecesModel.pack(~Element,model),Move.Bottom{}),
      pack_stack(~Element,stacked_model(~Element,model,Bottom{})),stacked_packed(~Element,model,Bottom{})),
    Tiling{0n,{==},stacked_tiles(~Element,model,Bottom{},0n,total,total,total,coverage,fields,ready,{==})},
    Both{Unit{},stacked_values(~Element,~Context,~value,context,model,Bottom{},values,Unit{})},stacked_spans(~Element,model,Bottom{})}

# A layout has a piece.
def layout_spanned(+layout: Pieces.Ranges,+rest: Spans) -> {spanned(layout_spans(layout,rest)) == True{} : Bool}:
  match layout:
    case Pieces.Range{first,count}: {==}
    case Pieces.Split{+left,+right}: layout_spanned(left,layout_spans(right,rest))

# The ranges of the pieces of a stack after a tree are those of the tree and
# then those of the stack.
def unstacked_spans(~Element: Data,+stack: Stack<Element>,+tree: PiecesModel.Model<Element>) ->
  {layout_spans(PiecesModel.layout(~Element,unstacked_model(~Element,stack,tree)),Ended{}) == layout_spans(PiecesModel.layout(~Element,tree),spans(~Element,stack)) : Spans}:
  match stack:
    case Bottom{}: {==}
    case Held{+buffer,+first,+count,+below}: unstacked_spans(~Element,below,PiecesModel.Fork{tree,PiecesModel.Leaf{buffer,first,count}})

# ---- A copy as a piece ----

# The model of a fresh buffer is as long as its contents.
def allocated_length(~Element: Data,+depth: Nat,+fill: Element,+length: Nat) ->
  {Buffer.length(~Element,Buffer.remodel(~Element,Allocation.allocate(~Element,depth,fill,length))) == Buffer.content_length(~Element,Buffer.contents(~Element,Allocation.allocate(~Element,depth,fill,length))) : Nat}:
  Equal.trans(Nat,Buffer.length(~Element,Buffer.remodel(~Element,Allocation.allocate(~Element,depth,fill,length))),Buffer.content_length(~Element,Buffer.content(~Element,Buffer.remodel(~Element,Allocation.allocate(~Element,depth,fill,length)))),
    Buffer.content_length(~Element,Buffer.contents(~Element,Allocation.allocate(~Element,depth,fill,length))),
    Equal.sym(Nat,Buffer.content_length(~Element,Buffer.content(~Element,Buffer.remodel(~Element,Allocation.allocate(~Element,depth,fill,length)))),Buffer.length(~Element,Buffer.remodel(~Element,Allocation.allocate(~Element,depth,fill,length))),
      Buffer.content_length_of(~Element,Buffer.remodel(~Element,Allocation.allocate(~Element,depth,fill,length)))),
    Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),Buffer.content(~Element,Buffer.remodel(~Element,Allocation.allocate(~Element,depth,fill,length))),Buffer.contents(~Element,Allocation.allocate(~Element,depth,fill,length)),
      Buffer.remodel_content(~Element,Allocation.allocate(~Element,depth,fill,length))))

# The model of a copy holds the tree and the length of its contents.
def copy_tree(~Element: Data,+holding: Holding<Element>,+first: U32,+count: U32,+fill: Element) ->
  {Buffer.tree(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))) == Buffer.content_tree(~Element,Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))) : Tree.Tree<Element>}:
  Equal.trans(Tree.Tree<Element>,Buffer.tree(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))),Buffer.content_tree(~Element,Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill))))),
    Buffer.content_tree(~Element,Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))),
    Equal.sym(Tree.Tree<Element>,Buffer.content_tree(~Element,Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill))))),Buffer.tree(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))),
      Buffer.content_tree_of(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill))))),
    Equal.cong(Buffer.Content<Element>,Tree.Tree<Element>,Buffer.content_tree(~Element),Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))),Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill))),
      Buffer.remodel_content(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))))

def copy_length(~Element: Data,+holding: Holding<Element>,+first: U32,+count: U32,+fill: Element) ->
  {Buffer.length(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))) == Buffer.content_length(~Element,Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))) : Nat}:
  Equal.trans(Nat,Buffer.length(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))),Buffer.content_length(~Element,Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill))))),
    Buffer.content_length(~Element,Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))),
    Equal.sym(Nat,Buffer.content_length(~Element,Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill))))),Buffer.length(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))),
      Buffer.content_length_of(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill))))),
    Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))),Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill))),
      Buffer.remodel_content(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))))

# A copy of count cells is count cells long.
def copied_length(~Element: Data,+holding: Holding<Element>,+first: U32,+count: U32,+fill: Element) ->
  {Buffer.content_length(~Element,Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))) == U32.to_nat(count) : Nat}:
  +depth = Storage.depth_for(count)
  +model = Buffer.remodel(~Element,Allocation.allocate(~Element,depth,fill,U32.to_nat(count)))
  +ran = model_run(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,Buffer.tree(~Element,model))
  %Buffer.remodeled(~Element,Allocation.allocate(~Element,depth,fill,U32.to_nat(count))) :
    {Buffer.content_length(~Element,Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,
      Move.run(~Element,U32.to_nat(count),count,first,1,False{},0,pack(~Element,holding),_)))) == U32.to_nat(count) : Nat}
  Equal.trans(Nat,
    Buffer.content_length(~Element,Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,
      Move.run(~Element,U32.to_nat(count),count,first,1,False{},0,pack(~Element,holding),Buffer.pack(~Element,model))))),
    Buffer.length(~Element,model),U32.to_nat(count),
    Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),
      Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,
        Move.run(~Element,U32.to_nat(count),count,first,1,False{},0,pack(~Element,holding),Buffer.pack(~Element,model)))),
      Buffer.Content{ran_tree(~Element,ran),Buffer.length(~Element,model)},
      run_target(~Element,Move.run(~Element,U32.to_nat(count),count,first,1,False{},0,pack(~Element,holding),Buffer.pack(~Element,model)),ran,Buffer.length(~Element,model),
        run_refines(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,model))),
    Equal.trans(Nat,Buffer.length(~Element,model),Buffer.content_length(~Element,Buffer.contents(~Element,Allocation.allocate(~Element,depth,fill,U32.to_nat(count)))),U32.to_nat(count),
      allocated_length(~Element,depth,fill,U32.to_nat(count)),
      Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),Buffer.contents(~Element,Allocation.allocate(~Element,depth,fill,U32.to_nat(count))),
        Buffer.Content{Tree.fresh(Element,depth,fill),U32.to_nat(count)},AllocationLaws.contents(~Element,depth,fill,U32.to_nat(count)))))

# A unit run keeps the shape and the capacity of its target tree.
def run_balanced(~Element: Data,fuel: Nat,+done: Bool,+count: U32,+offset: U32,+stride: U32,+back: Bool,+position: U32,+holding: Holding<Element>,+tree: Tree.Tree<Element>,
  +depth: Nat,+balanced: Shape.Balanced(Element,depth,tree)) ->
  Shape.Balanced(Element,depth,ran_tree(~Element,model_run(~Element,fuel,done,count,offset,stride,back,position,holding,tree))):
  match fuel done:
    case 0n done: balanced
    case 1n+rest True{}: balanced
    case 1n+ +rest False{}:
      +resolved = resolve(~Element,holding,offset)
      +cells = step_cells(~Element,resolved,count,offset,stride,back)
      run_balanced(~Element,rest,U32.is_zero((count - cells : U32)),(count - cells : U32),(offset + cells * Move.directed(back,stride) : U32),stride,back,(position + cells : U32),resolved,
        step_tree(~Element,resolved,count,offset,stride,back,position,tree),depth,step_balanced(~Element,resolved,count,offset,stride,back,position,depth,tree,balanced))

def run_capacity(~Element: Data,fuel: Nat,+done: Bool,+count: U32,+offset: U32,+stride: U32,+back: Bool,+position: U32,+holding: Holding<Element>,+tree: Tree.Tree<Element>) ->
  {Tree.capacity(Element,ran_tree(~Element,model_run(~Element,fuel,done,count,offset,stride,back,position,holding,tree))) == Tree.capacity(Element,tree) : U32}:
  match fuel done:
    case 0n done: {==}
    case 1n+rest True{}: {==}
    case 1n+ +rest False{}:
      +resolved = resolve(~Element,holding,offset)
      +cells = step_cells(~Element,resolved,count,offset,stride,back)
      +stepped = step_tree(~Element,resolved,count,offset,stride,back,position,tree)
      Equal.trans(U32,
        Tree.capacity(Element,ran_tree(~Element,model_run(~Element,rest,U32.is_zero((count - cells : U32)),(count - cells : U32),(offset + cells * Move.directed(back,stride) : U32),stride,back,(position + cells : U32),resolved,stepped))),
        Tree.capacity(Element,stepped),Tree.capacity(Element,tree),
        run_capacity(~Element,rest,U32.is_zero((count - cells : U32)),(count - cells : U32),(offset + cells * Move.directed(back,stride) : U32),stride,back,(position + cells : U32),resolved,stepped),
        step_capacity(~Element,resolved,count,offset,stride,back,position,tree))

# The contents of a copy: the tree of the unit run into a fresh tree, count
# cells long.
def copied_contents(~Element: Data,+holding: Holding<Element>,+first: U32,+count: U32,+fill: Element) ->
  {Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill))) ==
    Buffer.Content{ran_tree(~Element,model_run(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,Tree.fresh(Element,Storage.depth_for(count),fill))),U32.to_nat(count)} :
    Buffer.Content<Element>}:
  +depth = Storage.depth_for(count)
  +model = Buffer.remodel(~Element,Allocation.allocate(~Element,depth,fill,U32.to_nat(count)))
  +ran = model_run(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,Buffer.tree(~Element,model))
  %Buffer.remodeled(~Element,Allocation.allocate(~Element,depth,fill,U32.to_nat(count))) :
    {Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.run(~Element,U32.to_nat(count),count,first,1,False{},0,pack(~Element,holding),_))) ==
      Buffer.Content{ran_tree(~Element,model_run(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,Tree.fresh(Element,depth,fill))),U32.to_nat(count)} : Buffer.Content<Element>}
  +length_exact = Equal.trans(Nat,Buffer.length(~Element,model),Buffer.content_length(~Element,Buffer.contents(~Element,Allocation.allocate(~Element,depth,fill,U32.to_nat(count)))),U32.to_nat(count),
    allocated_length(~Element,depth,fill,U32.to_nat(count)),
    Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),Buffer.contents(~Element,Allocation.allocate(~Element,depth,fill,U32.to_nat(count))),
      Buffer.Content{Tree.fresh(Element,depth,fill),U32.to_nat(count)},AllocationLaws.contents(~Element,depth,fill,U32.to_nat(count))))
  Equal.trans(Buffer.Content<Element>,
    Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,
      Move.run(~Element,U32.to_nat(count),count,first,1,False{},0,pack(~Element,holding),Buffer.pack(~Element,model)))),
    Buffer.Content{ran_tree(~Element,ran),Buffer.length(~Element,model)},
    Buffer.Content{ran_tree(~Element,model_run(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,Tree.fresh(Element,depth,fill))),U32.to_nat(count)},
    run_target(~Element,Move.run(~Element,U32.to_nat(count),count,first,1,False{},0,pack(~Element,holding),Buffer.pack(~Element,model)),ran,Buffer.length(~Element,model),
      run_refines(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,model)),
    Equal.trans(Buffer.Content<Element>,Buffer.Content{ran_tree(~Element,ran),Buffer.length(~Element,model)},
      Buffer.Content{ran_tree(~Element,model_run(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,Tree.fresh(Element,depth,fill))),Buffer.length(~Element,model)},
      Buffer.Content{ran_tree(~Element,model_run(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,Tree.fresh(Element,depth,fill))),U32.to_nat(count)},
      Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{ran_tree(~Element,model_run(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,tree)),Buffer.length(~Element,model)},
        Buffer.tree(~Element,model),Tree.fresh(Element,depth,fill),fresh_tree(~Element,depth,fill,U32.to_nat(count))),
      Equal.cong(Nat,Buffer.Content<Element>,length => Buffer.Content{ran_tree(~Element,model_run(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,Tree.fresh(Element,depth,fill))),length},
        Buffer.length(~Element,model),U32.to_nat(count),length_exact)))

# A copy of at most 2^30 cells owns a balanced tree with room for its cells.
def copied_ready(~Element: Data,+holding: Holding<Element>,+first: U32,+count: U32,+fill: Element,+allocatable: {U32.is_le(count,1073741824) == True{} : Bool}) ->
  Buffer.ContentReady<Element,Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))>:
  +depth = Storage.depth_for(count)
  +sized = Depth.bounded(count,allocatable)
  +fresh = Tree.fresh(Element,depth,fill)
  +ran = ran_tree(~Element,model_run(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,fresh))
  +capacity_exact = Shape.capacity(Element,depth,fresh,Shape.fresh(Element,depth,fill))
  +fresh_room = {%Equal.sym(U32,Tree.capacity(Element,fresh),U32.shln(1,depth),capacity_exact) : {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool}
    Depth.allocation_room(count,sized) : {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Element,fresh))) == True{} : Bool}}
  +room = {%Equal.sym(U32,Tree.capacity(Element,ran),Tree.capacity(Element,fresh),run_capacity(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,fresh)) :
      {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool}
    fresh_room : {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Element,ran))) == True{} : Bool}}
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill))),
    Buffer.Content{ran,U32.to_nat(count)},copied_contents(~Element,holding,first,count,fill)) : Buffer.ContentReady<Element,_>
  Buffer.ContentReady{depth,run_balanced(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,fresh,depth,Shape.fresh(Element,depth,fill)),
    Depth.allocation_depth(count,sized),room}

# What a copy of a covered range needs, for every cell of it.
type CopyCells<-Element: Data,-Context: Data,-value: Context -> U32 -> Element,-context: Context> is Data:
  CopyCells{holding: Holding<Element>,first: U32,count: U32,fill: Element,low: Nat,high: Nat,capacity: U32,
    capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
    tiled: Tiled(~Element,holding,low,high),values: HoldingValues(~Element,~Context,~value,context,holding),after: {Nat.is_le(low,U32.to_nat(first)) == True{} : Bool},
    fits: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),high) == True{} : Bool},allocatable: {U32.is_le(count,1073741824) == True{} : Bool}}

def copy_holding(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,~context: Context,cells: CopyCells<Element,Context,value,context>) -> Holding<Element>:
  CopyCells{holding,first,count,fill,low,high,capacity,capacity_ok,bounded,tiled,values,after,fits,allocatable} = cells
  holding

def copy_first(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,~context: Context,cells: CopyCells<Element,Context,value,context>) -> U32:
  CopyCells{holding,first,count,fill,low,high,capacity,capacity_ok,bounded,tiled,values,after,fits,allocatable} = cells
  first

def copy_count(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,~context: Context,cells: CopyCells<Element,Context,value,context>) -> U32:
  CopyCells{holding,first,count,fill,low,high,capacity,capacity_ok,bounded,tiled,values,after,fits,allocatable} = cells
  count

def copy_fill(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,~context: Context,cells: CopyCells<Element,Context,value,context>) -> Element:
  CopyCells{holding,first,count,fill,low,high,capacity,capacity_ok,bounded,tiled,values,after,fits,allocatable} = cells
  fill

def copy_size(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,~context: Context,cells: CopyCells<Element,Context,value,context>) -> Nat:
  U32.to_nat(copy_count(~Element,~Context,~value,~context,cells))

# The model of the buffer a copy writes.
def copy_model(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,~context: Context,+cells: CopyCells<Element,Context,value,context>) -> Buffer.Model<Element>:
  Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,copy_holding(~Element,~Context,~value,~context,cells)),
    copy_first(~Element,~Context,~value,~context,cells),copy_count(~Element,~Context,~value,~context,cells),copy_fill(~Element,~Context,~value,~context,cells))))

def CopyAt(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+cells: CopyCells<Element,Context,value,context>,+index: Nat) -> Data:
  +model = copy_model(~Element,~Context,~value,~context,cells)
  Content.At(~Element,~Context,~value,context,copy_first(~Element,~Context,~value,~context,cells),Buffer.tree(~Element,model),Buffer.length(~Element,model),index)

def copy_at(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+cells: CopyCells<Element,Context,value,context>,+index: Nat,
  +inside: {Nat.is_lt(index,copy_size(~Element,~Context,~value,~context,cells)) == True{} : Bool}) -> CopyAt(~Element,~Context,~value,context,cells,index):
  CopyCells{+holding,+first,+count,+fill,+low,+high,+capacity,+capacity_ok,+bounded,+tiled,+values,+after,+fits,+allocatable} = cells
  KernelGemm.Both{
    %Equal.trans(Nat,U32.to_nat(count),Buffer.content_length(~Element,Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))),Buffer.length(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))),
      Equal.sym(Nat,Buffer.content_length(~Element,Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))),U32.to_nat(count),copied_length(~Element,holding,first,count,fill)),
      Equal.sym(Nat,Buffer.length(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))),Buffer.content_length(~Element,Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))),copy_length(~Element,holding,first,count,fill))) :
      {Nat.is_lt(index,_) == True{} : Bool}
    inside,
    Equal.trans(Element,Tree.read(Element,Buffer.tree(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))),U32.from_nat(index)),
      Tree.read(Element,Buffer.content_tree(~Element,Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))),U32.from_nat(index)),value(context,U32.from_nat(Nat.add(U32.to_nat(first),index))),
      Equal.cong(Tree.Tree<Element>,Element,tree => Tree.read(Element,tree,U32.from_nat(index)),Buffer.tree(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))),
        Buffer.content_tree(~Element,Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))),copy_tree(~Element,holding,first,count,fill)),
      copied_cell(~Element,~Context,~value,context,holding,first,count,fill,low,high,capacity,capacity_ok,bounded,tiled,values,after,fits,allocatable,index,inside))}

# A copy of a range of a covering holding is a piece of that range with the
# values of the tensor.
def copied_leaf(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+holding: Holding<Element>,+first: U32,+count: U32,+fill: Element,
  +low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +tiled: Tiled(~Element,holding,low,high),+values: HoldingValues(~Element,~Context,~value,context,holding),+after: {Nat.is_le(low,U32.to_nat(first)) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),high) == True{} : Bool},+allocatable: {U32.is_le(count,1073741824) == True{} : Bool}) ->
  Content.Leaf(~Element,~Context,~value,context,first,count,Buffer.tree(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))),Buffer.length(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill))))):
  Members.range_all(~CopyCells<Element,Context,value,context>,~CopyAt(~Element,~Context,~value,context),~copy_size(~Element,~Context,~value,~context),
    ~copy_at(~Element,~Context,~value,context),U32.to_nat(count),0n,
    CopyCells{holding,first,count,fill,low,high,capacity,capacity_ok,bounded,tiled,values,after,fits,allocatable},Order.reflexive(U32.to_nat(count)))

# ---- A copy as a holding of its own ----

# A context whose values are those of another from a base on.
type Shifted<-Context: Data> is Data:
  Shifted{context: Context,base: U32}

def shifted_value(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,shifted: Shifted<Context>,index: U32) -> Element:
  Shifted{context,base} = shifted
  value(context,(base + index : U32))

type ShiftScope<-Element: Data,-Context: Data> is Data:
  ShiftScope{context: Context,first: U32,tree: Tree.Tree<Element>,length: Nat}

def shift_before(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+scope: ShiftScope<Element,Context>,+index: Nat) -> Data:
  ShiftScope{context,first,tree,length} = scope
  Content.At(~Element,~Context,~value,context,first,tree,length,index)

def shift_after(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+scope: ShiftScope<Element,Context>,+index: Nat) -> Data:
  ShiftScope{context,first,tree,length} = scope
  Content.At(~Element,~Shifted<Context>,~shifted_value(~Element,~Context,~value),Shifted{context,first},0,tree,length,index)

# Cell first + index of a context is cell index of the context shifted by first.
def shift_cell(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+scope: ShiftScope<Element,Context>,+index: Nat,
  +cell: shift_before(~Element,~Context,~value,scope,index)) -> shift_after(~Element,~Context,~value,scope,index):
  ShiftScope{+context,+first,+tree,+length} = scope
  KernelGemm.Both{inside,read} = cell
  KernelGemm.Both{inside,
    %{%Arithmetic.native_roundtrip(first) : {U32.from_nat(Nat.add(U32.to_nat(first),index)) == (_ + U32.from_nat(index) : U32) : U32}
      Equal.sym(U32,U32.add(U32.from_nat(U32.to_nat(first)),U32.from_nat(index)),U32.from_nat(Nat.add(U32.to_nat(first),index)),WordArithmetic.native_add(U32.to_nat(first),index)) :
      {U32.from_nat(Nat.add(U32.to_nat(first),index)) == (first + U32.from_nat(index) : U32) : U32}} :
      {Tree.read(Element,tree,U32.from_nat(index)) == value(context,_) : Element}
    read}

# A piece of cells from first on holds, from 0 on, the cells of the context
# shifted by first.
def shifted_leaf(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+first: U32,+count: U32,+tree: Tree.Tree<Element>,+length: Nat,
  leaf: Content.Leaf(~Element,~Context,~value,context,first,count,tree,length)) ->
  Content.Leaf(~Element,~Shifted<Context>,~shifted_value(~Element,~Context,~value),Shifted{context,first},0,count,tree,length):
  Members.transform_all(~Nat,~ShiftScope<Element,Context>,~shift_before(~Element,~Context,~value),~shift_after(~Element,~Context,~value),
    ~shift_cell(~Element,~Context,~value),Ranges.range(U32.to_nat(count),0n),ShiftScope{context,first,tree,length},leaf)

# One buffer of count cells as the only piece covers its cells with their
# values.
def alone_held(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+model: Buffer.Model<Element>,+count: U32,
  +length_exact: {Buffer.length(~Element,model) == U32.to_nat(count) : Nat},+sized: Buffer.Sized<Element,model>,
  +leaf: Content.Leaf(~Element,~Context,~value,context,0,count,Buffer.tree(~Element,model),Buffer.length(~Element,model))) ->
  Covered<Element,Context,value,context,Move.held(~Element,Partition.Leaf{Buffer.pack(~Element,model),0,count,Unit{}}),
    0n,U32.to_nat(count),layout_spans(Pieces.Range{0,count},Ended{})>:
  Buffer.Model{+tree,+length,+certificate} = model
  held_covered(~Element,~Context,~value,context,PiecesModel.Leaf{Buffer.Model{tree,length,certificate},0,count},U32.to_nat(count),
    Content.WellFormed{Content.Fields{length_exact,Order.reflexive(U32.to_nat(count))},Buffer.content_ready(~Element,Buffer.Model{tree,length,certificate},sized),
      PiecesPlan.LeafCoverage{{==},{==}}},
    leaf)

# A buffer whose contents are count cells of a tree, as the only piece, covers
# its cells with the values the tree holds.
def alone_content(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+model: Buffer.Model<Element>,+count: U32,+tree: Tree.Tree<Element>,
  +exact: {Buffer.content(~Element,model) == Buffer.Content{tree,U32.to_nat(count)} : Buffer.Content<Element>},
  +ready: Buffer.ContentReady<Element,Buffer.Content{tree,U32.to_nat(count)}>,
  +leaf: Content.Leaf(~Element,~Context,~value,context,0,count,tree,U32.to_nat(count))) ->
  Covered<Element,Context,value,context,Move.held(~Element,Partition.Leaf{Buffer.pack(~Element,model),0,count,Unit{}}),
    0n,U32.to_nat(count),layout_spans(Pieces.Range{0,count},Ended{})>:
  Buffer.Model{+cells,+length,+certificate} = model
  +back = Equal.sym(Buffer.Content<Element>,Buffer.Content{cells,length},Buffer.Content{tree,U32.to_nat(count)},exact)
  alone_held(~Element,~Context,~value,context,Buffer.Model{cells,length,certificate},count,
    Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),Buffer.Content{cells,length},Buffer.Content{tree,U32.to_nat(count)},exact),
    Buffer.ready_content(~Element,Buffer.Model{cells,length,certificate},%back : Buffer.ContentReady<Element,_> ready),
    %back : Content.piece_values(~Element,~Context,~value,context,0,count,_) leaf)

# One buffer of count cells as the only piece covers its cells with the
# values of a context shifted by base.
def alone_covered(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+base: U32,+model: Buffer.Model<Element>,+count: U32,
  +length_exact: {Buffer.length(~Element,model) == U32.to_nat(count) : Nat},+sized: Buffer.Sized<Element,model>,
  +leaf: Content.Leaf(~Element,~Context,~value,context,base,count,Buffer.tree(~Element,model),Buffer.length(~Element,model))) ->
  Covered<Element,Shifted<Context>,shifted_value(~Element,~Context,~value),Shifted{context,base},Move.held(~Element,Partition.Leaf{Buffer.pack(~Element,model),0,count,Unit{}}),
    0n,U32.to_nat(count),layout_spans(Pieces.Range{0,count},Ended{})>:
  alone_held(~Element,~Shifted<Context>,~shifted_value(~Element,~Context,~value),Shifted{context,base},model,count,length_exact,sized,
    shifted_leaf(~Element,~Context,~value,context,base,count,Buffer.tree(~Element,model),Buffer.length(~Element,model),leaf))

# A copy of a range of a covering holding, as a model held as the only piece,
# covers its cells with the values of the tensor from the first cell of the
# range on.
def copied_alone(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+holding: Holding<Element>,+first: U32,+count: U32,+fill: Element,
  +low: Nat,+high: Nat,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(high,U32.to_nat(capacity)) == True{} : Bool},
  +tiled: Tiled(~Element,holding,low,high),+values: HoldingValues(~Element,~Context,~value,context,holding),+after: {Nat.is_le(low,U32.to_nat(first)) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),high) == True{} : Bool},+allocatable: {U32.is_le(count,1073741824) == True{} : Bool}) ->
  Covered<Element,Shifted<Context>,shifted_value(~Element,~Context,~value),Shifted{context,first},
    Move.held(~Element,Partition.Leaf{Buffer.pack(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))),
      0,count,Unit{}}),0n,U32.to_nat(count),layout_spans(Pieces.Range{0,count},Ended{})>:
  alone_covered(~Element,~Context,~value,context,first,
    Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill))),count,
    Equal.trans(Nat,Buffer.length(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))),
      Buffer.content_length(~Element,Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))),U32.to_nat(count),
      copy_length(~Element,holding,first,count,fill),copied_length(~Element,holding,first,count,fill)),
    Buffer.ready_content(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill))),
      %Equal.sym(Buffer.Content<Element>,Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))),
          Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill))),
          Buffer.remodel_content(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,pack(~Element,holding),first,count,fill)))) :
        Buffer.ContentReady<Element,_>
      copied_ready(~Element,holding,first,count,fill,allocatable)),
    copied_leaf(~Element,~Context,~value,context,holding,first,count,fill,low,high,capacity,capacity_ok,bounded,tiled,values,after,fits,allocatable))

# ---- Slices of a stack ----

# The pieces of a stack after its first count.
def dropped(~Element: Data,count: Nat,stack: Stack<Element>) -> Stack<Element>:
  match count stack:
    case 1n+rest Held{buffer,first,pieces,below}: dropped(~Element,rest,below)
    case count stack: stack

# The first count pieces of a stack.
def taken(~Element: Data,count: Nat,stack: Stack<Element>) -> Stack<Element>:
  match count stack:
    case 1n+rest Held{buffer,first,pieces,below}: Held{buffer,first,pieces,taken(~Element,rest,below)}
    case count stack: Bottom{}

# Where the first count pieces of a stack from low end.
def boundary(~Element: Data,count: Nat,stack: Stack<Element>,+low: Nat) -> Nat:
  match count stack:
    case 1n+rest Held{buffer,first,+pieces,below}: boundary(~Element,rest,below,Nat.add(low,U32.to_nat(pieces)))
    case count stack: low

def split_held(~Element: Data,+rest: Nat,+buffer: Buffer.Model<Element>,+first: U32,+pieces: U32,+below: Stack<Element>,+low: Nat,+high: Nat,
  +starts: Starts<Element,low,first,pieces,buffer>,
  +split: Both<Tiles(~Element,taken(~Element,rest,below),Nat.add(low,U32.to_nat(pieces)),boundary(~Element,rest,below,Nat.add(low,U32.to_nat(pieces)))),
    Tiles(~Element,dropped(~Element,rest,below),boundary(~Element,rest,below,Nat.add(low,U32.to_nat(pieces))),high)>) ->
  Both<Tiles(~Element,Held{buffer,first,pieces,taken(~Element,rest,below)},low,boundary(~Element,rest,below,Nat.add(low,U32.to_nat(pieces)))),
    Tiles(~Element,dropped(~Element,rest,below),boundary(~Element,rest,below,Nat.add(low,U32.to_nat(pieces))),high)>:
  Both{+head,+tail} = split
  Both{Both{starts,head},tail}

# A stack that covers [low, high) splits after count pieces at their boundary.
def split_tiles(~Element: Data,+count: Nat,+stack: Stack<Element>,+low: Nat,+high: Nat,+tiles: Tiles(~Element,stack,low,high)) ->
  Both<Tiles(~Element,taken(~Element,count,stack),low,boundary(~Element,count,stack,low)),Tiles(~Element,dropped(~Element,count,stack),boundary(~Element,count,stack,low),high)>:
  match count stack:
    case 1n+ +rest Held{+buffer,+first,+pieces,+below}:
      Both{+starts,+later} = tiles
      split_held(~Element,rest,buffer,first,pieces,below,low,high,starts,split_tiles(~Element,rest,below,Nat.add(low,U32.to_nat(pieces)),high,later))
    case 0n stack: Both{{==},tiles}
    case 1n+rest Bottom{}: Both{{==},tiles}

def values_held(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+rest: Nat,+buffer: Buffer.Model<Element>,+first: U32,+pieces: U32,
  +below: Stack<Element>,+piece: Content.Leaf(~Element,~Context,~value,context,first,pieces,Buffer.tree(~Element,buffer),Buffer.length(~Element,buffer)),
  +split: Both<StackValues(~Element,~Context,~value,context,taken(~Element,rest,below)),StackValues(~Element,~Context,~value,context,dropped(~Element,rest,below))>) ->
  Both<StackValues(~Element,~Context,~value,context,Held{buffer,first,pieces,taken(~Element,rest,below)}),StackValues(~Element,~Context,~value,context,dropped(~Element,rest,below))>:
  Both{+head,+tail} = split
  Both{Both{piece,head},tail}

def split_values(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+count: Nat,+stack: Stack<Element>,
  +values: StackValues(~Element,~Context,~value,context,stack)) ->
  Both<StackValues(~Element,~Context,~value,context,taken(~Element,count,stack)),StackValues(~Element,~Context,~value,context,dropped(~Element,count,stack))>:
  match count stack:
    case 1n+ +rest Held{+buffer,+first,+pieces,+below}:
      Both{+piece,+later} = values
      values_held(~Element,~Context,~value,context,rest,buffer,first,pieces,below,piece,split_values(~Element,~Context,~value,context,rest,below,later))
    case 0n stack: Both{Unit{},values}
    case 1n+rest Bottom{}: Both{Unit{},values}

# Dropping in two steps drops the sum, and the boundaries add up.
def dropped_twice(~Element: Data,+first: Nat,+stack: Stack<Element>,+second: Nat) ->
  {dropped(~Element,second,dropped(~Element,first,stack)) == dropped(~Element,Nat.add(first,second),stack) : Stack<Element>}:
  match first stack:
    case 1n+ +rest Held{+buffer,+start,+pieces,+below}: dropped_twice(~Element,rest,below,second)
    case 0n stack: {==}
    case 1n+rest Bottom{}:
      match second:
        case 0n: {==}
        case 1n+more: {==}

def boundary_twice(~Element: Data,+first: Nat,+stack: Stack<Element>,+second: Nat,+low: Nat) ->
  {boundary(~Element,second,dropped(~Element,first,stack),boundary(~Element,first,stack,low)) == boundary(~Element,Nat.add(first,second),stack,low) : Nat}:
  match first stack:
    case 1n+ +rest Held{+buffer,+start,+pieces,+below}: boundary_twice(~Element,rest,below,second,Nat.add(low,U32.to_nat(pieces)))
    case 0n stack: {==}
    case 1n+rest Bottom{}:
      match second:
        case 0n: {==}
        case 1n+more: {==}

# ---- The spare after a band ----

def spare_cells(~Element: Data,spare: SpareModel<Element>) -> U32:
  SpareModel{buffer,capacity} = spare
  capacity

def spare_back(~Element: Data,lent: Move.Lent<Element>,buffer: Storage.Buffer<Element>) -> Move.Spare<Element>:
  Pair.snd(Move.Holding<Element>,Move.Spare<Element>,Move.returned(~Element,lent,buffer))

# The spare a band leaves: the same spare after a lent piece, otherwise the
# band's buffer with room for the band.
def spare_when(~Element: Data,within: Bool,+count: U32,+spare: SpareModel<Element>,+returned: Buffer.Model<Element>) -> SpareModel<Element>:
  match within:
    case True{}: spare
    case False{}: SpareModel{returned,U32.max(spare_cells(~Element,spare),count)}

def spare_current(~Element: Data,holding: Holding<Element>,+first: U32,+count: U32,+spare: SpareModel<Element>,+returned: Buffer.Model<Element>) -> SpareModel<Element>:
  match holding:
    case Holding{behind,Held{buffer,+start,+span,below}}: spare_when(~Element,U32.is_le(start,first) && U32.is_le((first + count : U32),(start + span : U32)),count,spare,returned)
    case Holding{behind,Bottom{}}: spare_when(~Element,False{},count,spare,returned)

def spare_after(~Element: Data,+holding: Holding<Element>,+first: U32,+count: U32,+spare: SpareModel<Element>,+returned: Buffer.Model<Element>) -> SpareModel<Element>:
  spare_current(~Element,resolve(~Element,holding,first),first,count,spare,returned)

# An assembled band leaves its buffer as the spare, with room for the band.
def assembled_spare(~Element: Data,+first: U32,+count: U32,+fill: Element,+holding: Holding<Element>,+spare: SpareModel<Element>,+returned: Buffer.Model<Element>) ->
  {spare_back(~Element,band_loan(~Element,Move.assembled(~Element,first,count,fill,pack(~Element,holding),pack_spare(~Element,spare))),Buffer.pack(~Element,returned)) ==
    pack_spare(~Element,SpareModel{returned,U32.max(spare_cells(~Element,spare),count)}) : Move.Spare<Element>}:
  SpareModel{+model,+cells} = spare
  %Equal.sym(Move.Holding<Element> & Storage.Buffer<Element>,
    Move.copied_into(~Element,pack(~Element,holding),first,count,Move.roomy(~Element,U32.is_le(count,cells),count,fill,Buffer.pack(~Element,model))),
    (Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied_into(~Element,pack(~Element,holding),first,count,Move.roomy(~Element,U32.is_le(count,cells),count,fill,Buffer.pack(~Element,model)))),
      Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied_into(~Element,pack(~Element,holding),first,count,Move.roomy(~Element,U32.is_le(count,cells),count,fill,Buffer.pack(~Element,model))))),
    pair_eta(~Move.Holding<Element>,~Storage.Buffer<Element>,
      Move.copied_into(~Element,pack(~Element,holding),first,count,Move.roomy(~Element,U32.is_le(count,cells),count,fill,Buffer.pack(~Element,model))))) :
    {spare_back(~Element,band_loan(~Element,Move.assembled_band(~Element,U32.max(cells,count),_)),Buffer.pack(~Element,returned)) ==
      pack_spare(~Element,SpareModel{returned,U32.max(cells,count)}) : Move.Spare<Element>}
  {==}

def band_when_spare(~Element: Data,+within: Bool,+behind: Stack<Element>,+buffer: Buffer.Model<Element>,+start: U32,+span: U32,+below: Stack<Element>,+first: U32,+count: U32,
  +fill: Element,+spare: SpareModel<Element>,+returned: Buffer.Model<Element>) ->
  {spare_back(~Element,band_loan(~Element,
      Move.band_when(~Element,within,first,count,fill,pack_stack(~Element,behind),Buffer.pack(~Element,buffer),start,span,pack_stack(~Element,below),pack_spare(~Element,spare))),
      Buffer.pack(~Element,returned)) == pack_spare(~Element,spare_when(~Element,within,count,spare,returned)) : Move.Spare<Element>}:
  match within:
    case True{}: {==}
    case False{}: assembled_spare(~Element,first,count,fill,Holding{behind,Held{buffer,start,span,below}},spare,returned)

def band_current_spare(~Element: Data,+holding: Holding<Element>,+first: U32,+count: U32,+fill: Element,+spare: SpareModel<Element>,+returned: Buffer.Model<Element>) ->
  {spare_back(~Element,band_loan(~Element,Move.band_current(~Element,first,count,fill,pack(~Element,holding),pack_spare(~Element,spare))),Buffer.pack(~Element,returned)) ==
    pack_spare(~Element,spare_current(~Element,holding,first,count,spare,returned)) : Move.Spare<Element>}:
  match holding:
    case Holding{+behind,Held{+buffer,+start,+span,+below}}:
      band_when_spare(~Element,U32.is_le(start,first) && U32.is_le((first + count : U32),(start + span : U32)),behind,buffer,start,span,below,first,count,fill,spare,returned)
    case Holding{+behind,Bottom{}}: assembled_spare(~Element,first,count,fill,Holding{behind,Bottom{}},spare,returned)

# The spare a band leaves is the model's.
def band_spare(~Element: Data,+holding: Holding<Element>,+first: U32,+count: U32,+fill: Element,+spare: SpareModel<Element>,+returned: Buffer.Model<Element>) ->
  {spare_back(~Element,band_loan(~Element,Move.band(~Element,pack(~Element,holding),first,count,fill,pack_spare(~Element,spare))),Buffer.pack(~Element,returned)) ==
    pack_spare(~Element,spare_after(~Element,holding,first,count,spare,returned)) : Move.Spare<Element>}:
  %Equal.sym(Move.Holding<Element>,Move.resolve(~Element,pack(~Element,holding),first),pack(~Element,resolve(~Element,holding,first)),resolve_packed(~Element,holding,first)) :
    {spare_back(~Element,band_loan(~Element,Move.band_current(~Element,first,count,fill,_,pack_spare(~Element,spare))),Buffer.pack(~Element,returned)) ==
      pack_spare(~Element,spare_after(~Element,holding,first,count,spare,returned)) : Move.Spare<Element>}
  band_current_spare(~Element,resolve(~Element,holding,first),first,count,fill,spare,returned)

# A buffer model of some contents holds their tree.
def content_tree_is(~Element: Data,+model: Buffer.Model<Element>,+content: Buffer.Content<Element>,+same: {Buffer.content(~Element,model) == content : Buffer.Content<Element>}) ->
  {Buffer.tree(~Element,model) == Buffer.content_tree(~Element,content) : Tree.Tree<Element>}:
  Equal.trans(Tree.Tree<Element>,Buffer.tree(~Element,model),Buffer.content_tree(~Element,Buffer.content(~Element,model)),Buffer.content_tree(~Element,content),
    Equal.sym(Tree.Tree<Element>,Buffer.content_tree(~Element,Buffer.content(~Element,model)),Buffer.tree(~Element,model),Buffer.content_tree_of(~Element,model)),
    Equal.cong(Buffer.Content<Element>,Tree.Tree<Element>,Buffer.content_tree(~Element),Buffer.content(~Element,model),content,same))

# The cells a copy room holds.
def copy_room_cells(~Element: Data,+target: Buffer.Model<Element>,+count: U32,+fits: CopyRoom<Element,target,count>) ->
  {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,target)))) == True{} : Bool}:
  CopyRoom{depth,balanced,depth_ok,+room} = fits
  room

# The spare's capacity fits the buffer a band assembles into.
def roomy_cells(~Element: Data,enough: Bool,+count: U32,+fill: Element,+buffer: Buffer.Model<Element>,+capacity: U32,
  +enough_exact: {U32.is_le(count,capacity) == enough : Bool},+spare_room: CopyRoom<Element,buffer,capacity>,+allocatable: {U32.is_le(count,1073741824) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(capacity),U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,roomy_model(~Element,enough,count,fill,buffer))))) == True{} : Bool}:
  match enough:
    case True{}: copy_room_cells(~Element,buffer,capacity,spare_room)
    case False{}:
      Order.transitive(U32.to_nat(capacity),U32.to_nat(count),U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,roomy_model(~Element,False{},count,fill,buffer)))),
        Bounds.strict_le(U32.to_nat(capacity),U32.to_nat(count),le_false_lt(U32.to_nat(count),U32.to_nat(capacity),nat_le_false(count,capacity,enough_exact))),
        copy_room_cells(~Element,roomy_model(~Element,False{},count,fill,buffer),count,fresh_room(~Element,count,fill,allocatable)))

# A buffer with the tree of a copy into a target with room has room for the
# spare's capacity and the band.
def copied_spare_room(~Element: Data,+holding: Holding<Element>,+first: U32,+count: U32,+target: Buffer.Model<Element>,+cells: U32,+returned: Buffer.Model<Element>,
  +fits: CopyRoom<Element,target,count>,+cells_room: {Nat.is_le(U32.to_nat(cells),U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,target)))) == True{} : Bool},
  +same: {Buffer.tree(~Element,returned) == ran_tree(~Element,model_run(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,Buffer.tree(~Element,target))) :
    Tree.Tree<Element>}) ->
  CopyRoom<Element,returned,U32.max(cells,count)>:
  CopyRoom{+depth,+balanced,+depth_ok,+room} = fits
  +tree = Buffer.tree(~Element,target)
  +ran = ran_tree(~Element,model_run(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,tree))
  +kept = {
    %Equal.sym(Tree.Tree<Element>,Buffer.tree(~Element,returned),ran,same) : Shape.Balanced(Element,depth,_)
    run_balanced(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,tree,depth,balanced) : Shape.Balanced(Element,depth,Buffer.tree(~Element,returned))}
  +wide = {
    %Equal.sym(Tree.Tree<Element>,Buffer.tree(~Element,returned),ran,same) : {Nat.is_le(U32.to_nat(U32.max(cells,count)),U32.to_nat(Tree.capacity(Element,_))) == True{} : Bool}
    %Equal.sym(U32,Tree.capacity(Element,ran),Tree.capacity(Element,tree),run_capacity(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,tree)) :
      {Nat.is_le(U32.to_nat(U32.max(cells,count)),U32.to_nat(_)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.max(cells,count)),Nat.max(U32.to_nat(cells),U32.to_nat(count)),Extrema.maximum(cells,count)) :
      {Nat.is_le(_,U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool}
    NatExtrema.maximum_upper(U32.to_nat(cells),U32.to_nat(count),U32.to_nat(Tree.capacity(Element,tree)),cells_room,room) :
      {Nat.is_le(U32.to_nat(U32.max(cells,count)),U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,returned)))) == True{} : Bool}}
  CopyRoom{depth,kept,depth_ok,wide}

# The buffer an assembled band returns, of the contents of the band, is a
# spare with room for the band.
def assembled_spare_room(~Element: Data,+first: U32,+count: U32,+fill: Element,+holding: Holding<Element>,+spare: SpareModel<Element>,+spare_room: SpareRoom(~Element,spare),
  +allocatable: {U32.is_le(count,1073741824) == True{} : Bool},+returned: Buffer.Model<Element>,
  +same: {Buffer.content(~Element,returned) == Buffer.contents(~Element,band_buffer(~Element,Move.assembled(~Element,first,count,fill,pack(~Element,holding),pack_spare(~Element,spare)))) :
    Buffer.Content<Element>}) ->
  CopyRoom<Element,returned,U32.max(spare_cells(~Element,spare),count)>:
  SpareModel{+model,+cells} = spare
  +enough = U32.is_le(count,cells)
  +target = roomy_model(~Element,enough,count,fill,model)
  +tree = Buffer.tree(~Element,target)
  +ran = model_run(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,tree)
  +copied = {
    %Equal.sym(Storage.Buffer<Element>,Move.roomy(~Element,enough,count,fill,Buffer.pack(~Element,model)),Buffer.pack(~Element,target),roomy_packed(~Element,enough,count,fill,model)) :
      {Buffer.contents(~Element,band_buffer(~Element,Move.assembled_band(~Element,U32.max(cells,count),Move.copied_into(~Element,pack(~Element,holding),first,count,_)))) ==
        Buffer.Content{ran_tree(~Element,ran),Buffer.length(~Element,target)} : Buffer.Content<Element>}
    %Equal.sym(Move.Holding<Element> & Storage.Buffer<Element>,Move.copied_into(~Element,pack(~Element,holding),first,count,Buffer.pack(~Element,target)),
      (Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied_into(~Element,pack(~Element,holding),first,count,Buffer.pack(~Element,target))),
        Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied_into(~Element,pack(~Element,holding),first,count,Buffer.pack(~Element,target)))),
      pair_eta(~Move.Holding<Element>,~Storage.Buffer<Element>,Move.copied_into(~Element,pack(~Element,holding),first,count,Buffer.pack(~Element,target)))) :
      {Buffer.contents(~Element,band_buffer(~Element,Move.assembled_band(~Element,U32.max(cells,count),_))) ==
        Buffer.Content{ran_tree(~Element,ran),Buffer.length(~Element,target)} : Buffer.Content<Element>}
    run_target(~Element,Move.run(~Element,U32.to_nat(count),count,first,1,False{},0,pack(~Element,holding),Buffer.pack(~Element,target)),ran,Buffer.length(~Element,target),
      run_refines(~Element,U32.to_nat(count),U32.is_zero(count),count,first,1,False{},0,holding,target)) :
      {Buffer.contents(~Element,band_buffer(~Element,Move.assembled(~Element,first,count,fill,pack(~Element,holding),pack_spare(~Element,SpareModel{model,cells})))) ==
        Buffer.Content{ran_tree(~Element,ran),Buffer.length(~Element,target)} : Buffer.Content<Element>}}
  copied_spare_room(~Element,holding,first,count,target,cells,returned,roomy_room(~Element,enough,count,fill,model,cells,{==},spare_room,allocatable),
    roomy_cells(~Element,enough,count,fill,model,cells,{==},spare_room,allocatable),
    content_tree_is(~Element,returned,Buffer.Content{ran_tree(~Element,ran),Buffer.length(~Element,target)},
      Equal.trans(Buffer.Content<Element>,Buffer.content(~Element,returned),
        Buffer.contents(~Element,band_buffer(~Element,Move.assembled(~Element,first,count,fill,pack(~Element,holding),pack_spare(~Element,SpareModel{model,cells})))),
        Buffer.Content{ran_tree(~Element,ran),Buffer.length(~Element,target)},same,copied)))

def band_when_spare_room(~Element: Data,+within: Bool,+behind: Stack<Element>,+buffer: Buffer.Model<Element>,+start: U32,+span: U32,+below: Stack<Element>,+first: U32,+count: U32,
  +fill: Element,+spare: SpareModel<Element>,+spare_room: SpareRoom(~Element,spare),+allocatable: {U32.is_le(count,1073741824) == True{} : Bool},+returned: Buffer.Model<Element>,
  +same: {Buffer.content(~Element,returned) == Buffer.contents(~Element,band_buffer(~Element,
    Move.band_when(~Element,within,first,count,fill,pack_stack(~Element,behind),Buffer.pack(~Element,buffer),start,span,pack_stack(~Element,below),pack_spare(~Element,spare)))) :
    Buffer.Content<Element>}) ->
  SpareRoom(~Element,spare_when(~Element,within,count,spare,returned)):
  match within:
    case True{}: spare_room
    case False{}: assembled_spare_room(~Element,first,count,fill,Holding{behind,Held{buffer,start,span,below}},spare,spare_room,allocatable,returned,same)

def band_current_spare_room(~Element: Data,+holding: Holding<Element>,+first: U32,+count: U32,+fill: Element,+spare: SpareModel<Element>,+spare_room: SpareRoom(~Element,spare),
  +allocatable: {U32.is_le(count,1073741824) == True{} : Bool},+returned: Buffer.Model<Element>,
  +same: {Buffer.content(~Element,returned) == Buffer.contents(~Element,band_buffer(~Element,Move.band_current(~Element,first,count,fill,pack(~Element,holding),pack_spare(~Element,spare)))) :
    Buffer.Content<Element>}) ->
  SpareRoom(~Element,spare_current(~Element,holding,first,count,spare,returned)):
  match holding:
    case Holding{+behind,Held{+buffer,+start,+span,+below}}:
      band_when_spare_room(~Element,U32.is_le(start,first) && U32.is_le((first + count : U32),(start + span : U32)),behind,buffer,start,span,below,first,count,fill,spare,spare_room,
        allocatable,returned,same)
    case Holding{+behind,Bottom{}}: assembled_spare_room(~Element,first,count,fill,Holding{behind,Bottom{}},spare,spare_room,allocatable,returned,same)

# After a band, with its buffer back unchanged in contents, the spare has room
# for what it says.
def band_spare_room(~Element: Data,+holding: Holding<Element>,+first: U32,+count: U32,+fill: Element,+spare: SpareModel<Element>,+spare_room: SpareRoom(~Element,spare),
  +allocatable: {U32.is_le(count,1073741824) == True{} : Bool},+returned: Buffer.Model<Element>,
  +same: {Buffer.content(~Element,returned) == Buffer.contents(~Element,band_buffer(~Element,Move.band(~Element,pack(~Element,holding),first,count,fill,pack_spare(~Element,spare)))) :
    Buffer.Content<Element>}) ->
  SpareRoom(~Element,spare_after(~Element,holding,first,count,spare,returned)):
  band_current_spare_room(~Element,resolve(~Element,holding,first),first,count,fill,spare,spare_room,allocatable,returned,
    Equal.trans(Buffer.Content<Element>,Buffer.content(~Element,returned),Buffer.contents(~Element,band_buffer(~Element,Move.band(~Element,pack(~Element,holding),first,count,fill,pack_spare(~Element,spare)))),
      Buffer.contents(~Element,band_buffer(~Element,Move.band_current(~Element,first,count,fill,pack(~Element,resolve(~Element,holding,first)),pack_spare(~Element,spare)))),same,
      Equal.cong(Move.Holding<Element>,Buffer.Content<Element>,resolved => Buffer.contents(~Element,band_buffer(~Element,Move.band_current(~Element,first,count,fill,resolved,pack_spare(~Element,spare)))),
        Move.resolve(~Element,pack(~Element,holding),first),pack(~Element,resolve(~Element,holding,first)),resolve_packed(~Element,holding,first))))
