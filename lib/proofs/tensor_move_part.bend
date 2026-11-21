# What a part of held source pieces writes. Every row of the axes before the
# run is one run, so cell T(j) of the target holds the value of source cell
# S(j) for every box index j, (S(j), T(j)) the offsets of j over the axes.
import Base
import ../storage_buffer.bend as Storage
import ../tensor_move.bend as Move
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./tensor_move_readback.bend as Held
import ./nat_order.bend as Order
import ./nat_interval.bend as Interval
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./word_encoding.bend as WordEncoding
import ./u32_comparison.bend as Comparison
import ./u32_extrema.bend as Extrema
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./binary_radix.bend as Radix
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ./nat_division.bend as Division
import ./word_arithmetic.bend as WordArithmetic
import ./word_multiplication.bend as Multiplication
import ./boolean_logic.bend as Logic
import ./write_program.bend as Writes
import ./tensor_copy_values.bend as CopyValues
import ../tensor_view.bend as Views
import ./nat_extrema.bend as NatExtrema
import ./u32_division.bend as U32Division

# ---- Rows ----

def offsets_source(offsets: Move.Offsets) -> U32:
  Move.Offsets{source,target} = offsets
  source

def offsets_target(offsets: Move.Offsets) -> U32:
  Move.Offsets{source,target} = offsets
  target

def offsets_eta(+offsets: Move.Offsets) -> {Move.Offsets{offsets_source(offsets),offsets_target(offsets)} == offsets : Move.Offsets}:
  match offsets:
    case Move.Offsets{+source,+target}: {==}

# The source and the target cell of row index row of the axes.
def row_source(+axes: List<&2,Move.Axis>,+row: Nat,+source: U32,+target: U32) -> U32:
  offsets_source(Move.row_offsets(axes,U32.from_nat(row),source,target))

def row_target(+axes: List<&2,Move.Axis>,+row: Nat,+source: U32,+target: U32) -> U32:
  offsets_target(Move.row_offsets(axes,U32.from_nat(row),source,target))

# The rows from row on, each a run of extent cells.
def model_rows(~Element: Data,fuel: Nat,+row: Nat,+outer: List<&2,Move.Axis>,+extent: U32,+stride: U32,+back: Bool,+source: U32,+target: U32,
  +holding: Held.Holding<Element>,+tree: Tree.Tree<Element>) -> Held.Ran<Element>:
  match fuel:
    case 0n: Held.Ran{holding,tree}
    case 1n+rest:
      +ran = Held.model_run(~Element,U32.to_nat(extent),U32.is_zero(extent),extent,row_source(outer,row,source,target),stride,back,row_target(outer,row,source,target),holding,tree)
      model_rows(~Element,rest,1n+row,outer,extent,stride,back,source,target,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran))

# A model holds the tree and the length of its content.
def model_tree(~Element: Data,+model: Buffer.Model<Element>,+tree: Tree.Tree<Element>,+length: Nat,
  +content: {Buffer.content(~Element,model) == Buffer.Content{tree,length} : Buffer.Content<Element>}) -> {Buffer.tree(~Element,model) == tree : Tree.Tree<Element>}:
  Equal.trans(Tree.Tree<Element>,Buffer.tree(~Element,model),Buffer.content_tree(~Element,Buffer.content(~Element,model)),tree,
    Equal.sym(Tree.Tree<Element>,Buffer.content_tree(~Element,Buffer.content(~Element,model)),Buffer.tree(~Element,model),Buffer.content_tree_of(~Element,model)),
    Equal.cong(Buffer.Content<Element>,Tree.Tree<Element>,Buffer.content_tree(~Element),Buffer.content(~Element,model),Buffer.Content{tree,length},content))

def model_length(~Element: Data,+model: Buffer.Model<Element>,+tree: Tree.Tree<Element>,+length: Nat,
  +content: {Buffer.content(~Element,model) == Buffer.Content{tree,length} : Buffer.Content<Element>}) -> {Buffer.length(~Element,model) == length : Nat}:
  Equal.trans(Nat,Buffer.length(~Element,model),Buffer.content_length(~Element,Buffer.content(~Element,model)),length,
    Equal.sym(Nat,Buffer.content_length(~Element,Buffer.content(~Element,model)),Buffer.length(~Element,model),Buffer.content_length_of(~Element,model)),
    Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),Buffer.content(~Element,model),Buffer.Content{tree,length},content))

# The buffer after a run is the model of its tree, as long as before.
def run_content(~Element: Data,+count: U32,+offset: U32,+stride: U32,+back: Bool,+position: U32,+holding: Held.Holding<Element>,+model: Buffer.Model<Element>) ->
  {Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,
      Move.run(~Element,U32.to_nat(count),count,offset,stride,back,position,Held.pack(~Element,holding),Buffer.pack(~Element,model))))) ==
    Buffer.Content{Held.ran_tree(~Element,Held.model_run(~Element,U32.to_nat(count),U32.is_zero(count),count,offset,stride,back,position,holding,Buffer.tree(~Element,model))),
      Buffer.length(~Element,model)} : Buffer.Content<Element>}:
  Equal.trans(Buffer.Content<Element>,
    Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,
      Move.run(~Element,U32.to_nat(count),count,offset,stride,back,position,Held.pack(~Element,holding),Buffer.pack(~Element,model))))),
    Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,
      Move.run(~Element,U32.to_nat(count),count,offset,stride,back,position,Held.pack(~Element,holding),Buffer.pack(~Element,model)))),
    Buffer.Content{Held.ran_tree(~Element,Held.model_run(~Element,U32.to_nat(count),U32.is_zero(count),count,offset,stride,back,position,holding,Buffer.tree(~Element,model))),
      Buffer.length(~Element,model)},
    Buffer.remodel_content(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,
      Move.run(~Element,U32.to_nat(count),count,offset,stride,back,position,Held.pack(~Element,holding),Buffer.pack(~Element,model)))),
    Held.run_target(~Element,Move.run(~Element,U32.to_nat(count),count,offset,stride,back,position,Held.pack(~Element,holding),Buffer.pack(~Element,model)),
      Held.model_run(~Element,U32.to_nat(count),U32.is_zero(count),count,offset,stride,back,position,holding,Buffer.tree(~Element,model)),Buffer.length(~Element,model),
      Held.run_refines(~Element,U32.to_nat(count),U32.is_zero(count),count,offset,stride,back,position,holding,model)))

# ---- Rows that write a cell or leave it ----

# Every row from row on, fuel of them, fits the target and leaves cell query
# alone.
def RowsMiss(fuel: Nat,+row: Nat,+outer: List<&2,Move.Axis>,+extent: U32,+source: U32,+target: U32,+capacity: U32,+query: Nat) -> Data:
  match fuel:
    case 0n: Unit
    case 1n+rest:
      Held.Both<{Nat.is_le(Nat.add(U32.to_nat(row_target(outer,row,source,target)),U32.to_nat(extent)),U32.to_nat(capacity)) == True{} : Bool},
        Held.Both<Interval.Outside<U32.to_nat(row_target(outer,row,source,target)),U32.to_nat(extent),query>,RowsMiss(rest,1n+row,outer,extent,source,target,capacity,query)>>

# The rows from row on keep every cell they leave alone.
def rows_missed(~Element: Data,fuel: Nat,+row: Nat,+outer: List<&2,Move.Axis>,+extent: U32,+stride: U32,+back: Bool,+source: U32,+target: U32,
  +holding: Held.Holding<Element>,+tree: Tree.Tree<Element>,+depth: Nat,+balanced: Shape.Balanced(Element,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity: U32,+same: {Tree.capacity(Element,tree) == capacity : U32},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +query: U32,+query_inside: {U32.is_lt(query,capacity) == True{} : Bool},+miss: RowsMiss(fuel,row,outer,extent,source,target,capacity,U32.to_nat(query))) ->
  {Tree.read(Element,Held.ran_tree(~Element,model_rows(~Element,fuel,row,outer,extent,stride,back,source,target,holding,tree)),query) == Tree.read(Element,tree,query) : Element}:
  match fuel:
    case 0n: {==}
    case 1n+ +rest:
      Held.Both{+fits,+more} = miss
      Held.Both{+outside,+later} = more
      +first = row_source(outer,row,source,target)
      +position = row_target(outer,row,source,target)
      +ran = Held.model_run(~Element,U32.to_nat(extent),U32.is_zero(extent),extent,first,stride,back,position,holding,tree)
      +same_next = Equal.trans(U32,Tree.capacity(Element,Held.ran_tree(~Element,ran)),Tree.capacity(Element,tree),capacity,
        Held.run_capacity(~Element,U32.to_nat(extent),U32.is_zero(extent),extent,first,stride,back,position,holding,tree),same)
      Equal.trans(Element,
        Tree.read(Element,Held.ran_tree(~Element,model_rows(~Element,rest,1n+row,outer,extent,stride,back,source,target,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran))),query),
        Tree.read(Element,Held.ran_tree(~Element,ran),query),Tree.read(Element,tree,query),
        rows_missed(~Element,rest,1n+row,outer,extent,stride,back,source,target,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran),
          depth,Held.run_balanced(~Element,U32.to_nat(extent),U32.is_zero(extent),extent,first,stride,back,position,holding,tree,depth,balanced),depth_ok,capacity,same_next,capacity_ok,query,query_inside,later),
        Held.run_kept(~Element,U32.to_nat(extent),U32.is_zero(extent),extent,first,stride,back,position,holding,tree,depth,balanced,depth_ok,
          {%Equal.sym(U32,Tree.capacity(Element,tree),capacity,same) : {U32.is_le(_,2147483648) == True{} : Bool} capacity_ok :
            {U32.is_le(Tree.capacity(Element,tree),2147483648) == True{} : Bool}},
          {%Equal.sym(U32,Tree.capacity(Element,tree),capacity,same) : {Nat.is_le(Nat.add(U32.to_nat(position),U32.to_nat(extent)),U32.to_nat(_)) == True{} : Bool} fits :
            {Nat.is_le(Nat.add(U32.to_nat(position),U32.to_nat(extent)),U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool}},
          query,
          {%Equal.sym(U32,Tree.capacity(Element,tree),capacity,same) : {U32.is_lt(query,_) == True{} : Bool} query_inside : {U32.is_lt(query,Tree.capacity(Element,tree)) == True{} : Bool}},
          outside))

# Cell index of row hit, steps rows from row, holds the value of its source
# cell when no later row reaches it.
def rows_written(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,steps: Nat,+later: Nat,+row: Nat,+hit: Nat,
  +gap: {Nat.add(steps,row) == hit : Nat},+outer: List<&2,Move.Axis>,+extent: U32,+stride: U32,+back: Bool,+source: U32,+target: U32,
  +holding: Held.Holding<Element>,+tree: Tree.Tree<Element>,+stack: Held.Stack<Element>,
  +ordered: {Held.order(~Element,holding) == stack : Held.Stack<Element>},+start: Nat,+end: Nat,+source_capacity: U32,
  +source_ok: {U32.is_le(source_capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(end,U32.to_nat(source_capacity)) == True{} : Bool},
  +word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool},
  +sorted: Held.Sorted(~Element,stack,start,end),+values: Held.HoldingValues(~Element,~Context,~value,context,holding),
  +run_held: Held.RunHeld(Held.spans(~Element,stack),row_source(outer,hit,source,target),Move.directed(back,stride),U32.to_nat(extent)),
  +depth: Nat,+balanced: Shape.Balanced(Element,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity: U32,+same: {Tree.capacity(Element,tree) == capacity : U32},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(U32.to_nat(row_target(outer,hit,source,target)),U32.to_nat(extent)),U32.to_nat(capacity)) == True{} : Bool},
  +index: Nat,+less: {Nat.is_lt(index,U32.to_nat(extent)) == True{} : Bool},
  +miss: RowsMiss(later,1n+hit,outer,extent,source,target,capacity,Nat.add(U32.to_nat(row_target(outer,hit,source,target)),index))) ->
  {Tree.read(Element,Held.ran_tree(~Element,model_rows(~Element,Nat.add(steps,1n+later),row,outer,extent,stride,back,source,target,holding,tree)),
      U32.from_nat(Nat.add(U32.to_nat(row_target(outer,hit,source,target)),index))) ==
    value(context,(row_source(outer,hit,source,target) + U32.from_nat(index) * Move.directed(back,stride) : U32)) : Element}:
  match steps:
    case 0n:
      +first = row_source(outer,hit,source,target)
      +position = row_target(outer,hit,source,target)
      +ran = Held.model_run(~Element,U32.to_nat(extent),U32.is_zero(extent),extent,first,stride,back,position,holding,tree)
      +same_next = Equal.trans(U32,Tree.capacity(Element,Held.ran_tree(~Element,ran)),Tree.capacity(Element,tree),capacity,
        Held.run_capacity(~Element,U32.to_nat(extent),U32.is_zero(extent),extent,first,stride,back,position,holding,tree),same)
      +cell = Nat.add(U32.to_nat(position),index)
      +cell_below = Order.offset_lt(U32.to_nat(position),index,U32.to_nat(extent),less)
      +cell_room = Order.transitive(cell,Nat.add(U32.to_nat(position),U32.to_nat(extent)),U32.to_nat(capacity),
        Bounds.strict_le(cell,Nat.add(U32.to_nat(position),U32.to_nat(extent)),cell_below),fits)
      +cell_exact = Bounds.roundtrip(cell,capacity,capacity_ok,cell_room)
      %Equal.sym(Nat,row,hit,gap) :
        {Tree.read(Element,Held.ran_tree(~Element,model_rows(~Element,1n+later,_,outer,extent,stride,back,source,target,holding,tree)),U32.from_nat(cell)) ==
          value(context,(first + U32.from_nat(index) * Move.directed(back,stride) : U32)) : Element}
      Equal.trans(Element,
        Tree.read(Element,Held.ran_tree(~Element,model_rows(~Element,later,1n+hit,outer,extent,stride,back,source,target,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran))),
          U32.from_nat(cell)),
        Tree.read(Element,Held.ran_tree(~Element,ran),U32.from_nat(cell)),
        value(context,(first + U32.from_nat(index) * Move.directed(back,stride) : U32)),
        rows_missed(~Element,later,1n+hit,outer,extent,stride,back,source,target,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran),
          depth,Held.run_balanced(~Element,U32.to_nat(extent),U32.is_zero(extent),extent,first,stride,back,position,holding,tree,depth,balanced),depth_ok,capacity,same_next,capacity_ok,U32.from_nat(cell),
          {%Equal.sym(Bool,U32.is_lt(U32.from_nat(cell),capacity),Nat.is_lt(U32.to_nat(U32.from_nat(cell)),U32.to_nat(capacity)),Comparison.lt(U32.from_nat(cell),capacity)) :
              {_ == True{} : Bool}
            {%Equal.sym(Nat,U32.to_nat(U32.from_nat(cell)),cell,cell_exact) : {Nat.is_lt(_,U32.to_nat(capacity)) == True{} : Bool}
              Bounds.nat_trans(cell,Nat.add(U32.to_nat(position),U32.to_nat(extent)),U32.to_nat(capacity),cell_below,fits) :
              {Nat.is_lt(U32.to_nat(U32.from_nat(cell)),U32.to_nat(capacity)) == True{} : Bool}} : {U32.is_lt(U32.from_nat(cell),capacity) == True{} : Bool}},
          {%Equal.sym(Nat,U32.to_nat(U32.from_nat(cell)),cell,cell_exact) : RowsMiss(later,1n+hit,outer,extent,source,target,capacity,_) miss :
            RowsMiss(later,1n+hit,outer,extent,source,target,capacity,U32.to_nat(U32.from_nat(cell)))}),
        Held.run_written(~Element,~Context,~value,context,U32.to_nat(extent),U32.is_zero(extent),extent,first,stride,back,position,holding,tree,stack,ordered,start,end,
          source_capacity,source_ok,bounded,word,word_ok,stride_room,sorted,values,run_held,Order.reflexive(U32.to_nat(extent)),{==},depth,balanced,depth_ok,
          {%Equal.sym(U32,Tree.capacity(Element,tree),capacity,same) : {U32.is_le(_,2147483648) == True{} : Bool} capacity_ok :
            {U32.is_le(Tree.capacity(Element,tree),2147483648) == True{} : Bool}},
          {%Equal.sym(U32,Tree.capacity(Element,tree),capacity,same) : {Nat.is_le(Nat.add(U32.to_nat(position),U32.to_nat(extent)),U32.to_nat(_)) == True{} : Bool} fits :
            {Nat.is_le(Nat.add(U32.to_nat(position),U32.to_nat(extent)),U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool}},
          index,less))
    case 1n+ +more:
      +first = row_source(outer,row,source,target)
      +position = row_target(outer,row,source,target)
      +ran = Held.model_run(~Element,U32.to_nat(extent),U32.is_zero(extent),extent,first,stride,back,position,holding,tree)
      +same_next = Equal.trans(U32,Tree.capacity(Element,Held.ran_tree(~Element,ran)),Tree.capacity(Element,tree),capacity,
        Held.run_capacity(~Element,U32.to_nat(extent),U32.is_zero(extent),extent,first,stride,back,position,holding,tree),same)
      rows_written(~Element,~Context,~value,context,more,later,1n+row,hit,
        Equal.trans(Nat,Nat.add(more,1n+row),1n+Nat.add(more,row),hit,Addition.successor_right(more,row),gap),
        outer,extent,stride,back,source,target,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran),stack,
        Equal.trans(Held.Stack<Element>,Held.order(~Element,Held.ran_holding(~Element,ran)),Held.order(~Element,holding),stack,
          Held.run_order(~Element,U32.to_nat(extent),U32.is_zero(extent),extent,first,stride,back,position,holding,tree),ordered),
        start,end,source_capacity,source_ok,bounded,word,word_ok,stride_room,sorted,
        Held.run_values(~Element,~Context,~value,context,U32.to_nat(extent),U32.is_zero(extent),extent,first,stride,back,position,holding,tree,values),
        run_held,depth,Held.run_balanced(~Element,U32.to_nat(extent),U32.is_zero(extent),extent,first,stride,back,position,holding,tree,depth,balanced),depth_ok,capacity,same_next,capacity_ok,fits,index,less,miss)

# ---- Offsets of a box index ----

def axis_extent(+axis: Move.Axis) -> U32:
  Move.Axis{+extent,source,back,target} = axis
  extent

def and_left(+left: Bool,+right: Bool,+both: {left && right == True{} : Bool}) -> {left == True{} : Bool}:
  match left:
    case True{}: {==}
    case False{}: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(both))

def and_right(+left: Bool,+right: Bool,+both: {left && right == True{} : Bool}) -> {right == True{} : Bool}:
  match left right:
    case True{} True{}: {==}
    case True{} False{}: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(both))
    case False{} right: Empty.absurd({right == True{} : Bool},WordEncoding.false_true(both))

# The cells of the box of the axes.
def total(axes: List<&2,Move.Axis>) -> Nat:
  match axes:
    case Nil{}: 1n
    case Con{Move.Axis{+extent,source,back,target},rest}: Nat.mul(U32.to_nat(extent),total(rest))

def rows_exact(+axes: List<&2,Move.Axis>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+positive: {Nat.is_lt(0n,total(axes)) == True{} : Bool},
  +room: {Nat.is_le(total(axes),U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat(Move.rows_of(axes)) == total(axes) : Nat}:
  match axes:
    case Nil{}: {==}
    case Con{Move.Axis{+extent,+source,+back,+target},+rest}:
      +rest_positive = DivisionOrder.right_factor_positive(U32.to_nat(extent),total(rest),positive)
      +extent_positive = RowMajor.left_factor_positive(U32.to_nat(extent),total(rest),positive)
      +rest_room = Order.transitive(total(rest),Nat.mul(U32.to_nat(extent),total(rest)),U32.to_nat(cap),
        DivisionOrder.le_multiple(U32.to_nat(extent),total(rest),extent_positive),room)
      Arithmetic.product_exact(cap,cap_ok,extent,Move.rows_of(rest),U32.to_nat(extent),total(rest),{==},rows_exact(rest,cap,cap_ok,rest_positive,rest_room),room)

# A box index of quotient times the cells of the rest plus remainder steps
# the first axis by quotient and is the remainder of the rest.
def peel(+extent: U32,+stride: U32,+back: Bool,+step: U32,+rest: List<&2,Move.Axis>,+index: U32,+source: U32,+target: U32,+quotient: Nat,+remainder: Nat,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+cells: Nat,+inner: {U32.to_nat(Move.rows_of(rest)) == cells : Nat},
  +exact: {U32.to_nat(index) == Nat.add(Nat.mul(quotient,cells),remainder) : Nat},+remainder_in: {Nat.is_lt(remainder,cells) == True{} : Bool},
  +quotient_in: {Nat.is_lt(quotient,U32.to_nat(extent)) == True{} : Bool},
  +inner_room: {Nat.is_le(Nat.double(cells),U32.to_nat(cap)) == True{} : Bool},+extent_room: {Nat.is_le(Nat.double(U32.to_nat(extent)),U32.to_nat(cap)) == True{} : Bool}) ->
  {Move.row_offsets(Con{Move.Axis{extent,stride,back,step},rest},index,source,target) ==
    Move.row_offsets(rest,U32.from_nat(remainder),(source + U32.from_nat(quotient) * Move.directed(back,stride) : U32),(target + U32.from_nat(quotient) * step : U32)) : Move.Offsets}:
  +width = U32.max(Move.rows_of(rest),1)
  +positive = Order.positive_bound(remainder,cells,remainder_in)
  +width_exact = Extrema.max_one_exact(Move.rows_of(rest),cells,inner,positive)
  +quotient_exact = Radix.quotient_word(index,width,quotient,remainder,cells,cap,cap_ok,exact,width_exact,positive,remainder_in,inner_room)
  +remainder_exact = Radix.remainder_word(index,width,quotient,remainder,cells,cap,cap_ok,exact,width_exact,positive,remainder_in,inner_room)
  +extent_positive = Order.positive_bound(quotient,U32.to_nat(extent),quotient_in)
  +extent_width = Extrema.max_one_exact(extent,U32.to_nat(extent),{==},extent_positive)
  +digit_exact = Radix.remainder_word((index / width : U32),U32.max(extent,1),0n,quotient,U32.to_nat(extent),cap,cap_ok,quotient_exact,extent_width,extent_positive,quotient_in,extent_room)
  +quotient_room = Order.transitive(quotient,U32.to_nat(extent),U32.to_nat(cap),Bounds.strict_le(quotient,U32.to_nat(extent),quotient_in),
    Order.transitive(U32.to_nat(extent),Nat.double(U32.to_nat(extent)),U32.to_nat(cap),Order.le_double(U32.to_nat(extent)),extent_room))
  +remainder_room = Order.transitive(remainder,cells,U32.to_nat(cap),Bounds.strict_le(remainder,cells,remainder_in),
    Order.transitive(cells,Nat.double(cells),U32.to_nat(cap),Order.le_double(cells),inner_room))
  +digit = Arithmetic.to_nat_injective(U32.from_nat(quotient),((index / width : U32) % U32.max(extent,1) : U32),
    Equal.trans(Nat,U32.to_nat(U32.from_nat(quotient)),quotient,U32.to_nat(((index / width : U32) % U32.max(extent,1) : U32)),
      Bounds.roundtrip(quotient,cap,cap_ok,quotient_room),Equal.sym(Nat,U32.to_nat(((index / width : U32) % U32.max(extent,1) : U32)),quotient,digit_exact)))
  +later = Arithmetic.to_nat_injective(U32.from_nat(remainder),(index % width : U32),
    Equal.trans(Nat,U32.to_nat(U32.from_nat(remainder)),remainder,U32.to_nat((index % width : U32)),
      Bounds.roundtrip(remainder,cap,cap_ok,remainder_room),Equal.sym(Nat,U32.to_nat((index % width : U32)),remainder,remainder_exact)))
  %digit : {Move.row_offsets(rest,(index % width : U32),(source + _ * Move.directed(back,stride) : U32),(target + _ * step : U32)) ==
    Move.row_offsets(rest,U32.from_nat(remainder),(source + U32.from_nat(quotient) * Move.directed(back,stride) : U32),(target + U32.from_nat(quotient) * step : U32)) : Move.Offsets}
  %later : {Move.row_offsets(rest,_,(source + U32.from_nat(quotient) * Move.directed(back,stride) : U32),(target + U32.from_nat(quotient) * step : U32)) ==
    Move.row_offsets(rest,U32.from_nat(remainder),(source + U32.from_nat(quotient) * Move.directed(back,stride) : U32),(target + U32.from_nat(quotient) * step : U32)) : Move.Offsets}
  {==}

# ---- The run and the rows of a part ----

def axis_stride(+axis: Move.Axis) -> U32:
  Move.Axis{extent,+stride,back,target} = axis
  stride

def axis_back(+axis: Move.Axis) -> Bool:
  Move.Axis{extent,stride,+back,target} = axis
  back

def axis_step(+axis: Move.Axis) -> U32:
  Move.Axis{extent,stride,back,+target} = axis
  target

def split_outer(+split: Move.Split) -> List<&2,Move.Axis>:
  Move.Split{+outer,run} = split
  outer

def split_run(+split: Move.Split) -> Move.Axis:
  Move.Split{outer,+run} = split
  run

def split_eta(+split: Move.Split) -> {Move.Split{split_outer(split),split_run(split)} == split : Move.Split}:
  match split:
    case Move.Split{+outer,+run}: {==}

def outer_of(+axes: List<&2,Move.Axis>) -> List<&2,Move.Axis>:
  split_outer(Move.split_last(axes))

def run_of(+axes: List<&2,Move.Axis>) -> Move.Axis:
  split_run(Move.split_last(axes))

def outer_cons(+axis: Move.Axis,+next: Move.Axis,+more: List<&2,Move.Axis>) ->
  {outer_of(Con{axis,Con{next,more}}) == Con{axis,outer_of(Con{next,more})} : List<&2,Move.Axis>}:
  %split_eta(Move.split_last(Con{next,more})) : {split_outer(Move.split_prepended(axis,_)) == Con{axis,outer_of(Con{next,more})} : List<&2,Move.Axis>}
  {==}

def run_cons(+axis: Move.Axis,+next: Move.Axis,+more: List<&2,Move.Axis>) -> {run_of(Con{axis,Con{next,more}}) == run_of(Con{next,more}) : Move.Axis}:
  %split_eta(Move.split_last(Con{next,more})) : {split_run(Move.split_prepended(axis,_)) == run_of(Con{next,more}) : Move.Axis}
  {==}

# A single axis is the run when it steps one target cell, otherwise one row
# axis over a run of one cell.
def single_total(dense: Bool,+extent: U32,+stride: U32,+back: Bool,+step: U32) ->
  {total(Con{Move.Axis{extent,stride,back,step},Nil{}}) == Nat.mul(total(split_outer(Move.split_dense(dense,Move.Axis{extent,stride,back,step}))),U32.to_nat(axis_extent(split_run(Move.split_dense(dense,Move.Axis{extent,stride,back,step}))))) : Nat}:
  match dense:
    case True{}:
      Equal.trans(Nat,Nat.mul(U32.to_nat(extent),1n),U32.to_nat(extent),Nat.add(U32.to_nat(extent),0n),Order.times_one(U32.to_nat(extent)),
        Equal.sym(Nat,Nat.add(U32.to_nat(extent),0n),U32.to_nat(extent),Addition.zero_right(U32.to_nat(extent))))
    case False{}:
      Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(extent),1n),1n),Nat.mul(U32.to_nat(extent),1n),Order.times_one(Nat.mul(U32.to_nat(extent),1n)))

# The box of the axes is the box of the rows times the run.
def split_total(+axes: List<&2,Move.Axis>) -> {total(axes) == Nat.mul(total(outer_of(axes)),U32.to_nat(axis_extent(run_of(axes)))) : Nat}:
  match axes:
    case Nil{}: {==}
    case Con{Move.Axis{+extent,+stride,+back,+step},Nil{}}:
      single_total(U32.is_eq(step,1),extent,stride,back,step)
    case Con{Move.Axis{+extent,+stride,+back,+step},Con{+next,+more}}:
      +axis = {Move.Axis{extent,stride,back,step} : Move.Axis}
      %Equal.sym(List<&2,Move.Axis>,outer_of(Con{axis,Con{next,more}}),Con{axis,outer_of(Con{next,more})},outer_cons(axis,next,more)) :
        {Nat.mul(U32.to_nat(extent),total(Con{next,more})) == Nat.mul(total(_),U32.to_nat(axis_extent(run_of(Con{axis,Con{next,more}})))) : Nat}
      %Equal.sym(Move.Axis,run_of(Con{axis,Con{next,more}}),run_of(Con{next,more}),run_cons(axis,next,more)) :
        {Nat.mul(U32.to_nat(extent),total(Con{next,more})) == Nat.mul(Nat.mul(U32.to_nat(extent),total(outer_of(Con{next,more}))),U32.to_nat(axis_extent(_))) : Nat}
      %Equal.sym(Nat,total(Con{next,more}),Nat.mul(total(outer_of(Con{next,more})),U32.to_nat(axis_extent(run_of(Con{next,more})))),split_total(Con{next,more})) :
        {Nat.mul(U32.to_nat(extent),_) == Nat.mul(Nat.mul(U32.to_nat(extent),total(outer_of(Con{next,more}))),U32.to_nat(axis_extent(run_of(Con{next,more})))) : Nat}
      Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(extent),total(outer_of(Con{next,more}))),U32.to_nat(axis_extent(run_of(Con{next,more})))),
        Nat.mul(U32.to_nat(extent),Nat.mul(total(outer_of(Con{next,more})),U32.to_nat(axis_extent(run_of(Con{next,more}))))),
        RowMajor.assoc(U32.to_nat(extent),total(outer_of(Con{next,more})),U32.to_nat(axis_extent(run_of(Con{next,more})))))

def regroup(+quotient: Nat,+inner: Nat,+remainder: Nat,+extent: Nat,+cell: Nat) ->
  {Nat.add(Nat.mul(Nat.add(Nat.mul(quotient,inner),remainder),extent),cell) == Nat.add(Nat.mul(quotient,Nat.mul(inner,extent)),Nat.add(Nat.mul(remainder,extent),cell)) : Nat}:
  %Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(quotient,inner),remainder),extent),Nat.add(Nat.mul(Nat.mul(quotient,inner),extent),Nat.mul(remainder,extent)),
    RowMajor.distrib(Nat.mul(quotient,inner),remainder,extent)) :
    {Nat.add(_,cell) == Nat.add(Nat.mul(quotient,Nat.mul(inner,extent)),Nat.add(Nat.mul(remainder,extent),cell)) : Nat}
  %Equal.sym(Nat,Nat.mul(Nat.mul(quotient,inner),extent),Nat.mul(quotient,Nat.mul(inner,extent)),RowMajor.assoc(quotient,inner,extent)) :
    {Nat.add(Nat.add(_,Nat.mul(remainder,extent)),cell) == Nat.add(Nat.mul(quotient,Nat.mul(inner,extent)),Nat.add(Nat.mul(remainder,extent),cell)) : Nat}
  Addition.associative(Nat.mul(quotient,Nat.mul(inner,extent)),Nat.mul(remainder,extent),cell)

def zero_of_one(+value: Nat,+below: {Nat.is_lt(value,1n) == True{} : Bool}) -> {value == 0n : Nat}:
  Order.zero_below(value,Division.lt_succ_le(value,0n,below))

def double_room(+small: Nat,+large: Nat,+cap: U32,+order: {Nat.is_le(small,large) == True{} : Bool},+room: {Nat.is_le(Nat.double(large),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(Nat.double(small),U32.to_nat(cap)) == True{} : Bool}:
  Order.transitive(Nat.double(small),Nat.double(large),U32.to_nat(cap),Order.double_le(small,large,order),room)

def single_room(+value: Nat,+cap: U32,+room: {Nat.is_le(Nat.double(value),U32.to_nat(cap)) == True{} : Bool}) -> {Nat.is_le(value,U32.to_nat(cap)) == True{} : Bool}:
  Order.transitive(value,Nat.double(value),U32.to_nat(cap),Order.le_double(value),room)

def zero_step(+base: U32,+step: U32) -> {(base + U32.from_nat(0n) * step : U32) == base : U32}:
  Equal.trans(U32,(base + 0 * step : U32),(base + 0 : U32),base,
    Equal.cong(U32,U32,word => (base + word : U32),(0 * step : U32),0,Arithmetic.multiply_zero_left(step)),Arithmetic.add_zero(base))

def last_offsets(dense: Bool,+extent: U32,+stride: U32,+back: Bool,+step: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(total(Con{Move.Axis{extent,stride,back,step},Nil{}})),U32.to_nat(cap)) == True{} : Bool},+index: U32,+row: Nat,+cell: Nat,
  +exact: {U32.to_nat(index) == Nat.add(Nat.mul(row,U32.to_nat(axis_extent(split_run(Move.split_dense(dense,Move.Axis{extent,stride,back,step}))))),cell) : Nat},
  +row_in: {Nat.is_lt(row,total(split_outer(Move.split_dense(dense,Move.Axis{extent,stride,back,step})))) == True{} : Bool},+cell_in: {Nat.is_lt(cell,U32.to_nat(axis_extent(split_run(Move.split_dense(dense,Move.Axis{extent,stride,back,step}))))) == True{} : Bool},
  +source: U32,+target: U32) ->
  {Move.row_offsets(Con{Move.Axis{extent,stride,back,step},Nil{}},index,source,target) ==
    Move.Offsets{(row_source(split_outer(Move.split_dense(dense,Move.Axis{extent,stride,back,step})),row,source,target) + U32.from_nat(cell) * Move.directed(axis_back(split_run(Move.split_dense(dense,Move.Axis{extent,stride,back,step}))),axis_stride(split_run(Move.split_dense(dense,Move.Axis{extent,stride,back,step})))) : U32),
      (row_target(split_outer(Move.split_dense(dense,Move.Axis{extent,stride,back,step})),row,source,target) + U32.from_nat(cell) * axis_step(split_run(Move.split_dense(dense,Move.Axis{extent,stride,back,step}))) : U32)} : Move.Offsets}:
  match dense:
    case True{}:
        +zero = zero_of_one(row,row_in)
        +extent_room = {%Order.times_one(U32.to_nat(extent)) : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool} room :
          {Nat.is_le(Nat.double(U32.to_nat(extent)),U32.to_nat(cap)) == True{} : Bool}}
        +cells = {%zero : {U32.to_nat(index) == Nat.add(Nat.mul(_,U32.to_nat(extent)),cell) : Nat} exact : {U32.to_nat(index) == cell : Nat}}
        peel(extent,stride,back,step,Nil{},index,source,target,cell,0n,cap,cap_ok,total(Nil{}),{==},
          Equal.trans(Nat,U32.to_nat(index),cell,Nat.add(Nat.mul(cell,1n),0n),cells,
            Equal.sym(Nat,Nat.add(Nat.mul(cell,1n),0n),cell,Equal.trans(Nat,Nat.add(Nat.mul(cell,1n),0n),Nat.mul(cell,1n),cell,Addition.zero_right(Nat.mul(cell,1n)),Order.times_one(cell)))),
          {==},cell_in,double_room(1n,U32.to_nat(extent),cap,Order.positive_one(U32.to_nat(extent),Order.positive_bound(cell,U32.to_nat(extent),cell_in)),extent_room),extent_room)
    case False{}:
      +zero = zero_of_one(cell,cell_in)
      +row_exact = Equal.trans(Nat,U32.to_nat(index),Nat.add(Nat.mul(row,1n),cell),row,exact,
        {%Equal.sym(Nat,cell,0n,zero) : {Nat.add(Nat.mul(row,1n),_) == row : Nat}
          Equal.trans(Nat,Nat.add(Nat.mul(row,1n),0n),Nat.mul(row,1n),row,Addition.zero_right(Nat.mul(row,1n)),Order.times_one(row)) :
          {Nat.add(Nat.mul(row,1n),cell) == row : Nat}})
      +same = Equal.trans(U32,index,U32.from_nat(U32.to_nat(index)),U32.from_nat(row),Equal.sym(U32,U32.from_nat(U32.to_nat(index)),index,Arithmetic.native_roundtrip(index)),
        Equal.cong(Nat,U32,U32.from_nat,U32.to_nat(index),row,row_exact))
      +rows = Move.row_offsets(Con{Move.Axis{extent,stride,back,step},Nil{}},U32.from_nat(row),source,target)
      %Equal.sym(Nat,cell,0n,zero) :
        {Move.row_offsets(Con{Move.Axis{extent,stride,back,step},Nil{}},index,source,target) ==
          Move.Offsets{(offsets_source(rows) + U32.from_nat(_) * Move.directed(False{},0) : U32),(offsets_target(rows) + U32.from_nat(_) * 1 : U32)} : Move.Offsets}
      %Equal.sym(U32,(offsets_source(rows) + U32.from_nat(0n) * Move.directed(False{},0) : U32),offsets_source(rows),zero_step(offsets_source(rows),Move.directed(False{},0))) :
        {Move.row_offsets(Con{Move.Axis{extent,stride,back,step},Nil{}},index,source,target) == Move.Offsets{_,(offsets_target(rows) + U32.from_nat(0n) * 1 : U32)} : Move.Offsets}
      %Equal.sym(U32,(offsets_target(rows) + U32.from_nat(0n) * 1 : U32),offsets_target(rows),zero_step(offsets_target(rows),1)) :
        {Move.row_offsets(Con{Move.Axis{extent,stride,back,step},Nil{}},index,source,target) == Move.Offsets{offsets_source(rows),_} : Move.Offsets}
      %Equal.sym(Move.Offsets,Move.Offsets{offsets_source(rows),offsets_target(rows)},rows,offsets_eta(rows)) :
        {Move.row_offsets(Con{Move.Axis{extent,stride,back,step},Nil{}},index,source,target) == _ : Move.Offsets}
      Equal.cong(U32,Move.Offsets,word => Move.row_offsets(Con{Move.Axis{extent,stride,back,step},Nil{}},word,source,target),index,U32.from_nat(row),same)

# Box index row times the run plus cell is cell of the run of row row.
def split_offsets(+axes: List<&2,Move.Axis>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(total(axes)),U32.to_nat(cap)) == True{} : Bool},+index: U32,+row: Nat,+cell: Nat,
  +exact: {U32.to_nat(index) == Nat.add(Nat.mul(row,U32.to_nat(axis_extent(run_of(axes)))),cell) : Nat},
  +row_in: {Nat.is_lt(row,total(outer_of(axes))) == True{} : Bool},+cell_in: {Nat.is_lt(cell,U32.to_nat(axis_extent(run_of(axes)))) == True{} : Bool},
  +source: U32,+target: U32) ->
  {Move.row_offsets(axes,index,source,target) ==
    Move.Offsets{(row_source(outer_of(axes),row,source,target) + U32.from_nat(cell) * Move.directed(axis_back(run_of(axes)),axis_stride(run_of(axes))) : U32),
      (row_target(outer_of(axes),row,source,target) + U32.from_nat(cell) * axis_step(run_of(axes)) : U32)} : Move.Offsets}:
  match axes:
    case Nil{}:
      +zero = zero_of_one(cell,cell_in)
      %Equal.sym(Nat,cell,0n,zero) :
        {Move.Offsets{source,target} == Move.Offsets{(source + U32.from_nat(_) * Move.directed(False{},0) : U32),(target + U32.from_nat(_) * 1 : U32)} : Move.Offsets}
      %Equal.sym(U32,(source + 0 : U32),source,Arithmetic.add_zero(source)) : {Move.Offsets{source,target} == Move.Offsets{_,(target + 0 : U32)} : Move.Offsets}
      %Equal.sym(U32,(target + 0 : U32),target,Arithmetic.add_zero(target)) : {Move.Offsets{source,target} == Move.Offsets{source,_} : Move.Offsets}
      {==}
    case Con{Move.Axis{+extent,+stride,+back,+step},Nil{}}:
      last_offsets(U32.is_eq(step,1),extent,stride,back,step,cap,cap_ok,room,index,row,cell,exact,row_in,cell_in,source,target)
    case Con{Move.Axis{+extent,+stride,+back,+step},Con{+next,+more}}:
      +axis = {Move.Axis{extent,stride,back,step} : Move.Axis}
      +rest = {Con{next,more} : List<&2,Move.Axis>}
      +outer = outer_of(rest)
      +run = run_of(rest)
      +outer_same = outer_cons(axis,next,more)
      +run_same = run_cons(axis,next,more)
      +inner = total(outer)
      +width = U32.to_nat(axis_extent(run))
      +rest_total = split_total(rest)
      +row_inside = {%outer_same : {Nat.is_lt(row,total(_)) == True{} : Bool} row_in : {Nat.is_lt(row,Nat.mul(U32.to_nat(extent),inner)) == True{} : Bool}}
      +cell_inside = {%run_same : {Nat.is_lt(cell,U32.to_nat(axis_extent(_))) == True{} : Bool} cell_in : {Nat.is_lt(cell,width) == True{} : Bool}}
      +cells = {%run_same : {U32.to_nat(index) == Nat.add(Nat.mul(row,U32.to_nat(axis_extent(_))),cell) : Nat} exact : {U32.to_nat(index) == Nat.add(Nat.mul(row,width),cell) : Nat}}
      +inner_positive = DivisionOrder.right_factor_positive(U32.to_nat(extent),inner,Order.positive_bound(row,Nat.mul(U32.to_nat(extent),inner),row_inside))
      +quotient = Nat.div(row,inner)
      +remainder = Nat.mod(row,inner)
      +row_exact = DivisionOrder.quotient_exact(row,inner,inner_positive)
      +quotient_in = DivisionOrder.quotient_below(row,U32.to_nat(extent),inner,inner_positive,row_inside)
      +remainder_in = DivisionOrder.remainder_small(row,inner,inner_positive)
      +extent_positive = Order.positive_bound(quotient,U32.to_nat(extent),quotient_in)
      +width_positive = Order.positive_bound(cell,width,cell_inside)
      +rest_positive = {%Equal.sym(Nat,total(rest),Nat.mul(inner,width),rest_total) : {Nat.is_lt(0n,_) == True{} : Bool}
        Order.product_positive(inner,width,inner_positive,width_positive) : {Nat.is_lt(0n,total(rest)) == True{} : Bool}}
      +rest_room = double_room(total(rest),Nat.mul(U32.to_nat(extent),total(rest)),cap,DivisionOrder.le_multiple(U32.to_nat(extent),total(rest),extent_positive),room)
      +extent_room = double_room(U32.to_nat(extent),Nat.mul(U32.to_nat(extent),total(rest)),cap,RowMajor.product_covers(U32.to_nat(extent),total(rest),rest_positive),room)
      +inner_le = {%Equal.sym(Nat,total(rest),Nat.mul(inner,width),rest_total) : {Nat.is_le(inner,_) == True{} : Bool}
        RowMajor.product_covers(inner,width,width_positive) : {Nat.is_le(inner,total(rest)) == True{} : Bool}}
      +inner_room = double_room(inner,total(rest),cap,inner_le,rest_room)
      +later = Nat.add(Nat.mul(remainder,width),cell)
      +later_in = {%Equal.sym(Nat,total(rest),Nat.mul(inner,width),rest_total) : {Nat.is_lt(later,_) == True{} : Bool}
        RowMajor.radix(remainder,inner,cell,width,remainder_in,cell_inside) : {Nat.is_lt(later,total(rest)) == True{} : Bool}}
      +later_room = Order.transitive(later,total(rest),U32.to_nat(cap),Bounds.strict_le(later,total(rest),later_in),single_room(total(rest),cap,rest_room))
      +index_exact = Equal.trans(Nat,U32.to_nat(index),Nat.add(Nat.mul(row,width),cell),Nat.add(Nat.mul(quotient,total(rest)),later),cells,
        {%Equal.sym(Nat,total(rest),Nat.mul(inner,width),rest_total) : {Nat.add(Nat.mul(row,width),cell) == Nat.add(Nat.mul(quotient,_),later) : Nat}
          {%Equal.sym(Nat,row,Nat.add(Nat.mul(quotient,inner),remainder),row_exact) : {Nat.add(Nat.mul(_,width),cell) == Nat.add(Nat.mul(quotient,Nat.mul(inner,width)),later) : Nat}
            regroup(quotient,inner,remainder,width,cell) : {Nat.add(Nat.mul(row,width),cell) == Nat.add(Nat.mul(quotient,Nat.mul(inner,width)),later) : Nat}} :
          {Nat.add(Nat.mul(row,width),cell) == Nat.add(Nat.mul(quotient,total(rest)),later) : Nat}})
      +row_room = Order.transitive(row,Nat.mul(U32.to_nat(extent),total(rest)),U32.to_nat(cap),
        Order.transitive(row,Nat.mul(U32.to_nat(extent),inner),Nat.mul(U32.to_nat(extent),total(rest)),Bounds.strict_le(row,Nat.mul(U32.to_nat(extent),inner),row_inside),
          DivisionOrder.multiply_left_le(U32.to_nat(extent),inner,total(rest),inner_le)),
        single_room(Nat.mul(U32.to_nat(extent),total(rest)),cap,room))
      +moved_source = (source + U32.from_nat(quotient) * Move.directed(back,stride) : U32)
      +moved_target = (target + U32.from_nat(quotient) * step : U32)
      +rows_peeled = peel(extent,stride,back,step,outer,U32.from_nat(row),source,target,quotient,remainder,cap,cap_ok,total(outer),
        rows_exact(outer,cap,cap_ok,inner_positive,single_room(inner,cap,inner_room)),
        Equal.trans(Nat,U32.to_nat(U32.from_nat(row)),row,Nat.add(Nat.mul(quotient,inner),remainder),Bounds.roundtrip(row,cap,cap_ok,row_room),row_exact),
        remainder_in,quotient_in,inner_room,extent_room)
      +later_offsets = split_offsets(rest,cap,cap_ok,rest_room,U32.from_nat(later),remainder,cell,
        Bounds.roundtrip(later,cap,cap_ok,later_room),remainder_in,cell_inside,moved_source,moved_target)
      %Equal.sym(List<&2,Move.Axis>,outer_of(Con{axis,rest}),Con{axis,outer},outer_same) :
        {Move.row_offsets(Con{axis,rest},index,source,target) ==
          Move.Offsets{(row_source(_,row,source,target) + U32.from_nat(cell) * Move.directed(axis_back(run_of(Con{axis,rest})),axis_stride(run_of(Con{axis,rest}))) : U32),
            (row_target(_,row,source,target) + U32.from_nat(cell) * axis_step(run_of(Con{axis,rest})) : U32)} : Move.Offsets}
      %Equal.sym(Move.Axis,run_of(Con{axis,rest}),run,run_same) :
        {Move.row_offsets(Con{axis,rest},index,source,target) ==
          Move.Offsets{(row_source(Con{axis,outer},row,source,target) + U32.from_nat(cell) * Move.directed(axis_back(_),axis_stride(_)) : U32),
            (row_target(Con{axis,outer},row,source,target) + U32.from_nat(cell) * axis_step(_) : U32)} : Move.Offsets}
      %Equal.sym(Move.Offsets,Move.row_offsets(Con{axis,outer},U32.from_nat(row),source,target),
        Move.row_offsets(outer,U32.from_nat(remainder),moved_source,moved_target),rows_peeled) :
        {Move.row_offsets(Con{axis,rest},index,source,target) ==
          Move.Offsets{(offsets_source(_) + U32.from_nat(cell) * Move.directed(axis_back(run),axis_stride(run)) : U32),
            (offsets_target(_) + U32.from_nat(cell) * axis_step(run) : U32)} : Move.Offsets}
      Equal.trans(Move.Offsets,Move.row_offsets(Con{axis,rest},index,source,target),Move.row_offsets(rest,U32.from_nat(later),moved_source,moved_target),
        Move.Offsets{(row_source(outer,remainder,moved_source,moved_target) + U32.from_nat(cell) * Move.directed(axis_back(run),axis_stride(run)) : U32),
          (row_target(outer,remainder,moved_source,moved_target) + U32.from_nat(cell) * axis_step(run) : U32)},
        peel(extent,stride,back,step,rest,index,source,target,quotient,later,cap,cap_ok,total(rest),
          rows_exact(rest,cap,cap_ok,rest_positive,single_room(total(rest),cap,rest_room)),index_exact,later_in,quotient_in,rest_room,extent_room),
        later_offsets)

# ---- The nest of the axes before the run ----

def nest_outer(+nest: Move.Nest) -> List<&2,Move.Axis>:
  Move.Nest{+outer,inner} = nest
  outer

def nest_inner(+nest: Move.Nest) -> Move.Axis:
  Move.Nest{outer,+inner} = nest
  inner

def nest_eta(+nest: Move.Nest) -> {Move.Nest{nest_outer(nest),nest_inner(nest)} == nest : Move.Nest}:
  match nest:
    case Move.Nest{+outer,+inner}: {==}

# The axes of the blocks and the inner axis of the rows of axes.
def nest_outer_of(+axes: List<&2,Move.Axis>) -> List<&2,Move.Axis>:
  nest_outer(Move.nest_of(axes))

def nest_inner_of(+axes: List<&2,Move.Axis>) -> Move.Axis:
  nest_inner(Move.nest_of(axes))

def nest_outer_cons(+axis: Move.Axis,+next: Move.Axis,+more: List<&2,Move.Axis>) ->
  {nest_outer_of(Con{axis,Con{next,more}}) == Con{axis,nest_outer_of(Con{next,more})} : List<&2,Move.Axis>}:
  %nest_eta(Move.nest_of(Con{next,more})) : {nest_outer(Move.nest_prepended(axis,_)) == Con{axis,nest_outer_of(Con{next,more})} : List<&2,Move.Axis>}
  {==}

def nest_inner_cons(+axis: Move.Axis,+next: Move.Axis,+more: List<&2,Move.Axis>) -> {nest_inner_of(Con{axis,Con{next,more}}) == nest_inner_of(Con{next,more}) : Move.Axis}:
  %nest_eta(Move.nest_of(Con{next,more})) : {nest_inner(Move.nest_prepended(axis,_)) == nest_inner_of(Con{next,more}) : Move.Axis}
  {==}

# The box of the axes is the box of the blocks times the inner axis.
def nest_total(+axes: List<&2,Move.Axis>) -> {total(axes) == Nat.mul(total(nest_outer_of(axes)),U32.to_nat(axis_extent(nest_inner_of(axes)))) : Nat}:
  match axes:
    case Nil{}: {==}
    case Con{Move.Axis{+extent,+stride,+back,+step},Nil{}}:
      Equal.trans(Nat,Nat.mul(U32.to_nat(extent),1n),U32.to_nat(extent),Nat.add(U32.to_nat(extent),0n),Order.times_one(U32.to_nat(extent)),
        Equal.sym(Nat,Nat.add(U32.to_nat(extent),0n),U32.to_nat(extent),Addition.zero_right(U32.to_nat(extent))))
    case Con{Move.Axis{+extent,+stride,+back,+step},Con{+next,+more}}:
      +axis = {Move.Axis{extent,stride,back,step} : Move.Axis}
      %Equal.sym(List<&2,Move.Axis>,nest_outer_of(Con{axis,Con{next,more}}),Con{axis,nest_outer_of(Con{next,more})},nest_outer_cons(axis,next,more)) :
        {Nat.mul(U32.to_nat(extent),total(Con{next,more})) == Nat.mul(total(_),U32.to_nat(axis_extent(nest_inner_of(Con{axis,Con{next,more}})))) : Nat}
      %Equal.sym(Move.Axis,nest_inner_of(Con{axis,Con{next,more}}),nest_inner_of(Con{next,more}),nest_inner_cons(axis,next,more)) :
        {Nat.mul(U32.to_nat(extent),total(Con{next,more})) == Nat.mul(Nat.mul(U32.to_nat(extent),total(nest_outer_of(Con{next,more}))),U32.to_nat(axis_extent(_))) : Nat}
      %Equal.sym(Nat,total(Con{next,more}),Nat.mul(total(nest_outer_of(Con{next,more})),U32.to_nat(axis_extent(nest_inner_of(Con{next,more})))),nest_total(Con{next,more})) :
        {Nat.mul(U32.to_nat(extent),_) == Nat.mul(Nat.mul(U32.to_nat(extent),total(nest_outer_of(Con{next,more}))),U32.to_nat(axis_extent(nest_inner_of(Con{next,more})))) : Nat}
      Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(extent),total(nest_outer_of(Con{next,more}))),U32.to_nat(axis_extent(nest_inner_of(Con{next,more})))),
        Nat.mul(U32.to_nat(extent),Nat.mul(total(nest_outer_of(Con{next,more})),U32.to_nat(axis_extent(nest_inner_of(Con{next,more}))))),
        RowMajor.assoc(U32.to_nat(extent),total(nest_outer_of(Con{next,more})),U32.to_nat(axis_extent(nest_inner_of(Con{next,more})))))

# The rows of one axis: one block, the axis inner.
def single_nest_offsets(+extent: U32,+stride: U32,+back: Bool,+step: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(total(Con{Move.Axis{extent,stride,back,step},Nil{}})),U32.to_nat(cap)) == True{} : Bool},+index: U32,+row: Nat,+cell: Nat,
  +exact: {U32.to_nat(index) == Nat.add(Nat.mul(row,U32.to_nat(extent)),cell) : Nat},
  +row_in: {Nat.is_lt(row,1n) == True{} : Bool},+cell_in: {Nat.is_lt(cell,U32.to_nat(extent)) == True{} : Bool},+source: U32,+target: U32) ->
  {Move.row_offsets(Con{Move.Axis{extent,stride,back,step},Nil{}},index,source,target) ==
    Move.Offsets{(source + U32.from_nat(cell) * Move.directed(back,stride) : U32),(target + U32.from_nat(cell) * step : U32)} : Move.Offsets}:
      +zero = zero_of_one(row,row_in)
      +extent_room = {%Order.times_one(U32.to_nat(extent)) : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool} room :
        {Nat.is_le(Nat.double(U32.to_nat(extent)),U32.to_nat(cap)) == True{} : Bool}}
      +cells = {%zero : {U32.to_nat(index) == Nat.add(Nat.mul(_,U32.to_nat(extent)),cell) : Nat} exact : {U32.to_nat(index) == cell : Nat}}
      peel(extent,stride,back,step,Nil{},index,source,target,cell,0n,cap,cap_ok,total(Nil{}),{==},
        Equal.trans(Nat,U32.to_nat(index),cell,Nat.add(Nat.mul(cell,1n),0n),cells,
          Equal.sym(Nat,Nat.add(Nat.mul(cell,1n),0n),cell,Equal.trans(Nat,Nat.add(Nat.mul(cell,1n),0n),Nat.mul(cell,1n),cell,Addition.zero_right(Nat.mul(cell,1n)),Order.times_one(cell)))),
        {==},cell_in,double_room(1n,U32.to_nat(extent),cap,Order.positive_one(U32.to_nat(extent),Order.positive_bound(cell,U32.to_nat(extent),cell_in)),extent_room),extent_room)

# Box index row times the inner extent plus cell is cell of the inner axis of
# block row.
def nest_offsets(+axes: List<&2,Move.Axis>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(total(axes)),U32.to_nat(cap)) == True{} : Bool},+index: U32,+row: Nat,+cell: Nat,
  +exact: {U32.to_nat(index) == Nat.add(Nat.mul(row,U32.to_nat(axis_extent(nest_inner_of(axes)))),cell) : Nat},
  +row_in: {Nat.is_lt(row,total(nest_outer_of(axes))) == True{} : Bool},+cell_in: {Nat.is_lt(cell,U32.to_nat(axis_extent(nest_inner_of(axes)))) == True{} : Bool},
  +source: U32,+target: U32) ->
  {Move.row_offsets(axes,index,source,target) ==
    Move.Offsets{(row_source(nest_outer_of(axes),row,source,target) + U32.from_nat(cell) * Move.directed(axis_back(nest_inner_of(axes)),axis_stride(nest_inner_of(axes))) : U32),
      (row_target(nest_outer_of(axes),row,source,target) + U32.from_nat(cell) * axis_step(nest_inner_of(axes)) : U32)} : Move.Offsets}:
  match axes:
    case Nil{}:
      +zero = zero_of_one(cell,cell_in)
      %Equal.sym(Nat,cell,0n,zero) :
        {Move.Offsets{source,target} == Move.Offsets{(source + U32.from_nat(_) * Move.directed(False{},0) : U32),(target + U32.from_nat(_) * 0 : U32)} : Move.Offsets}
      %Equal.sym(U32,(source + 0 : U32),source,Arithmetic.add_zero(source)) : {Move.Offsets{source,target} == Move.Offsets{_,(target + 0 : U32)} : Move.Offsets}
      %Equal.sym(U32,(target + 0 : U32),target,Arithmetic.add_zero(target)) : {Move.Offsets{source,target} == Move.Offsets{source,_} : Move.Offsets}
      {==}
    case Con{Move.Axis{+extent,+stride,+back,+step},Nil{}}:
      single_nest_offsets(extent,stride,back,step,cap,cap_ok,room,index,row,cell,exact,row_in,cell_in,source,target)
    case Con{Move.Axis{+extent,+stride,+back,+step},Con{+next,+more}}:
      +axis = {Move.Axis{extent,stride,back,step} : Move.Axis}
      +rest = {Con{next,more} : List<&2,Move.Axis>}
      +outer = nest_outer_of(rest)
      +run = nest_inner_of(rest)
      +outer_same = nest_outer_cons(axis,next,more)
      +run_same = nest_inner_cons(axis,next,more)
      +inner = total(outer)
      +width = U32.to_nat(axis_extent(run))
      +rest_total = nest_total(rest)
      +row_inside = {%outer_same : {Nat.is_lt(row,total(_)) == True{} : Bool} row_in : {Nat.is_lt(row,Nat.mul(U32.to_nat(extent),inner)) == True{} : Bool}}
      +cell_inside = {%run_same : {Nat.is_lt(cell,U32.to_nat(axis_extent(_))) == True{} : Bool} cell_in : {Nat.is_lt(cell,width) == True{} : Bool}}
      +cells = {%run_same : {U32.to_nat(index) == Nat.add(Nat.mul(row,U32.to_nat(axis_extent(_))),cell) : Nat} exact : {U32.to_nat(index) == Nat.add(Nat.mul(row,width),cell) : Nat}}
      +inner_positive = DivisionOrder.right_factor_positive(U32.to_nat(extent),inner,Order.positive_bound(row,Nat.mul(U32.to_nat(extent),inner),row_inside))
      +quotient = Nat.div(row,inner)
      +remainder = Nat.mod(row,inner)
      +row_exact = DivisionOrder.quotient_exact(row,inner,inner_positive)
      +quotient_in = DivisionOrder.quotient_below(row,U32.to_nat(extent),inner,inner_positive,row_inside)
      +remainder_in = DivisionOrder.remainder_small(row,inner,inner_positive)
      +extent_positive = Order.positive_bound(quotient,U32.to_nat(extent),quotient_in)
      +width_positive = Order.positive_bound(cell,width,cell_inside)
      +rest_positive = {%Equal.sym(Nat,total(rest),Nat.mul(inner,width),rest_total) : {Nat.is_lt(0n,_) == True{} : Bool}
        Order.product_positive(inner,width,inner_positive,width_positive) : {Nat.is_lt(0n,total(rest)) == True{} : Bool}}
      +rest_room = double_room(total(rest),Nat.mul(U32.to_nat(extent),total(rest)),cap,DivisionOrder.le_multiple(U32.to_nat(extent),total(rest),extent_positive),room)
      +extent_room = double_room(U32.to_nat(extent),Nat.mul(U32.to_nat(extent),total(rest)),cap,RowMajor.product_covers(U32.to_nat(extent),total(rest),rest_positive),room)
      +inner_le = {%Equal.sym(Nat,total(rest),Nat.mul(inner,width),rest_total) : {Nat.is_le(inner,_) == True{} : Bool}
        RowMajor.product_covers(inner,width,width_positive) : {Nat.is_le(inner,total(rest)) == True{} : Bool}}
      +inner_room = double_room(inner,total(rest),cap,inner_le,rest_room)
      +later = Nat.add(Nat.mul(remainder,width),cell)
      +later_in = {%Equal.sym(Nat,total(rest),Nat.mul(inner,width),rest_total) : {Nat.is_lt(later,_) == True{} : Bool}
        RowMajor.radix(remainder,inner,cell,width,remainder_in,cell_inside) : {Nat.is_lt(later,total(rest)) == True{} : Bool}}
      +later_room = Order.transitive(later,total(rest),U32.to_nat(cap),Bounds.strict_le(later,total(rest),later_in),single_room(total(rest),cap,rest_room))
      +index_exact = Equal.trans(Nat,U32.to_nat(index),Nat.add(Nat.mul(row,width),cell),Nat.add(Nat.mul(quotient,total(rest)),later),cells,
        {%Equal.sym(Nat,total(rest),Nat.mul(inner,width),rest_total) : {Nat.add(Nat.mul(row,width),cell) == Nat.add(Nat.mul(quotient,_),later) : Nat}
          {%Equal.sym(Nat,row,Nat.add(Nat.mul(quotient,inner),remainder),row_exact) : {Nat.add(Nat.mul(_,width),cell) == Nat.add(Nat.mul(quotient,Nat.mul(inner,width)),later) : Nat}
            regroup(quotient,inner,remainder,width,cell) : {Nat.add(Nat.mul(row,width),cell) == Nat.add(Nat.mul(quotient,Nat.mul(inner,width)),later) : Nat}} :
          {Nat.add(Nat.mul(row,width),cell) == Nat.add(Nat.mul(quotient,total(rest)),later) : Nat}})
      +row_room = Order.transitive(row,Nat.mul(U32.to_nat(extent),total(rest)),U32.to_nat(cap),
        Order.transitive(row,Nat.mul(U32.to_nat(extent),inner),Nat.mul(U32.to_nat(extent),total(rest)),Bounds.strict_le(row,Nat.mul(U32.to_nat(extent),inner),row_inside),
          DivisionOrder.multiply_left_le(U32.to_nat(extent),inner,total(rest),inner_le)),
        single_room(Nat.mul(U32.to_nat(extent),total(rest)),cap,room))
      +moved_source = (source + U32.from_nat(quotient) * Move.directed(back,stride) : U32)
      +moved_target = (target + U32.from_nat(quotient) * step : U32)
      +rows_peeled = peel(extent,stride,back,step,outer,U32.from_nat(row),source,target,quotient,remainder,cap,cap_ok,total(outer),
        rows_exact(outer,cap,cap_ok,inner_positive,single_room(inner,cap,inner_room)),
        Equal.trans(Nat,U32.to_nat(U32.from_nat(row)),row,Nat.add(Nat.mul(quotient,inner),remainder),Bounds.roundtrip(row,cap,cap_ok,row_room),row_exact),
        remainder_in,quotient_in,inner_room,extent_room)
      +later_offsets = nest_offsets(rest,cap,cap_ok,rest_room,U32.from_nat(later),remainder,cell,
        Bounds.roundtrip(later,cap,cap_ok,later_room),remainder_in,cell_inside,moved_source,moved_target)
      %Equal.sym(List<&2,Move.Axis>,nest_outer_of(Con{axis,rest}),Con{axis,outer},outer_same) :
        {Move.row_offsets(Con{axis,rest},index,source,target) ==
          Move.Offsets{(row_source(_,row,source,target) + U32.from_nat(cell) * Move.directed(axis_back(nest_inner_of(Con{axis,rest})),axis_stride(nest_inner_of(Con{axis,rest}))) : U32),
            (row_target(_,row,source,target) + U32.from_nat(cell) * axis_step(nest_inner_of(Con{axis,rest})) : U32)} : Move.Offsets}
      %Equal.sym(Move.Axis,nest_inner_of(Con{axis,rest}),run,run_same) :
        {Move.row_offsets(Con{axis,rest},index,source,target) ==
          Move.Offsets{(row_source(Con{axis,outer},row,source,target) + U32.from_nat(cell) * Move.directed(axis_back(_),axis_stride(_)) : U32),
            (row_target(Con{axis,outer},row,source,target) + U32.from_nat(cell) * axis_step(_) : U32)} : Move.Offsets}
      %Equal.sym(Move.Offsets,Move.row_offsets(Con{axis,outer},U32.from_nat(row),source,target),
        Move.row_offsets(outer,U32.from_nat(remainder),moved_source,moved_target),rows_peeled) :
        {Move.row_offsets(Con{axis,rest},index,source,target) ==
          Move.Offsets{(offsets_source(_) + U32.from_nat(cell) * Move.directed(axis_back(run),axis_stride(run)) : U32),
            (offsets_target(_) + U32.from_nat(cell) * axis_step(run) : U32)} : Move.Offsets}
      Equal.trans(Move.Offsets,Move.row_offsets(Con{axis,rest},index,source,target),Move.row_offsets(rest,U32.from_nat(later),moved_source,moved_target),
        Move.Offsets{(row_source(outer,remainder,moved_source,moved_target) + U32.from_nat(cell) * Move.directed(axis_back(run),axis_stride(run)) : U32),
          (row_target(outer,remainder,moved_source,moved_target) + U32.from_nat(cell) * axis_step(run) : U32)},
        peel(extent,stride,back,step,rest,index,source,target,quotient,later,cap,cap_ok,total(rest),
          rows_exact(rest,cap,cap_ok,rest_positive,single_room(total(rest),cap,rest_room)),index_exact,later_in,quotient_in,rest_room,extent_room),
        later_offsets)

# ---- Coalesced axes ----

def same_back(+left: Bool,+right: Bool,+same: {Move.same_direction(left,right) == True{} : Bool}) -> {left == right : Bool}:
  match left right:
    case True{} True{}: {==}
    case False{} False{}: {==}
    case True{} False{}: Empty.absurd({True{} == False{} : Bool},WordEncoding.false_true(same))
    case False{} True{}: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(same))

def directed_scaled(+back: Bool,+stride: U32,+extent: U32) -> {Move.directed(back,(stride * extent : U32)) == (Move.directed(back,stride) * extent : U32) : U32}:
  match back:
    case False{}: {==}
    case True{}:
      Equal.sym(U32,((0 - stride : U32) * extent : U32),(0 - (stride * extent : U32) : U32),
        Equal.trans(U32,((0 - stride : U32) * extent : U32),(0 * extent - stride * extent : U32),(0 - (stride * extent : U32) : U32),
          Arithmetic.multiply_subtractive(0,stride,extent),
          Equal.cong(U32,U32,word => (word - stride * extent : U32),(0 * extent : U32),0,Arithmetic.multiply_zero_left(extent))))

# Stepping high indices of an axis of step outer_step = step times extent and
# low of step step is stepping high times extent plus low of step step.
def merge_ring(+base: U32,+high: Nat,+low: Nat,+extent: U32,+step: U32,+outer_step: U32,+outer_same: {outer_step == (step * extent : U32) : U32}) ->
  {(base + U32.from_nat(Nat.add(Nat.mul(high,U32.to_nat(extent)),low)) * step : U32) == ((base + U32.from_nat(high) * outer_step : U32) + U32.from_nat(low) * step : U32) : U32}:
  +upper = U32.from_nat(high)
  +lower = U32.from_nat(low)
  +scaled = {%Arithmetic.native_roundtrip(extent) : {(upper * _ : U32) == U32.from_nat(Nat.mul(high,U32.to_nat(extent))) : U32}
    Multiplication.native_encoded_multiply(high,U32.to_nat(extent)) : {(upper * extent : U32) == U32.from_nat(Nat.mul(high,U32.to_nat(extent))) : U32}}
  +index = Equal.trans(U32,U32.from_nat(Nat.add(Nat.mul(high,U32.to_nat(extent)),low)),(U32.from_nat(Nat.mul(high,U32.to_nat(extent))) + lower : U32),(upper * extent + lower : U32),
    Equal.sym(U32,(U32.from_nat(Nat.mul(high,U32.to_nat(extent))) + lower : U32),U32.from_nat(Nat.add(Nat.mul(high,U32.to_nat(extent)),low)),
      WordArithmetic.native_add(Nat.mul(high,U32.to_nat(extent)),low)),
    Equal.cong(U32,U32,word => (word + lower : U32),U32.from_nat(Nat.mul(high,U32.to_nat(extent))),(upper * extent : U32),
      Equal.sym(U32,(upper * extent : U32),U32.from_nat(Nat.mul(high,U32.to_nat(extent))),scaled)))
  +outer = Equal.trans(U32,(upper * extent * step : U32),(upper * (extent * step : U32) : U32),(upper * outer_step : U32),
    Arithmetic.multiply_associative(upper,extent,step),
    Equal.cong(U32,U32,word => (upper * word : U32),(extent * step : U32),outer_step,
      Equal.trans(U32,(extent * step : U32),(step * extent : U32),outer_step,Arithmetic.multiply_commutative(extent,step),Equal.sym(U32,outer_step,(step * extent : U32),outer_same))))
  %Equal.sym(U32,U32.from_nat(Nat.add(Nat.mul(high,U32.to_nat(extent)),low)),(upper * extent + lower : U32),index) :
    {(base + _ * step : U32) == ((base + upper * outer_step : U32) + lower * step : U32) : U32}
  %Equal.sym(U32,((upper * extent + lower : U32) * step : U32),(upper * extent * step + lower * step : U32),Arithmetic.multiply_additive((upper * extent : U32),lower,step)) :
    {(base + _ : U32) == ((base + upper * outer_step : U32) + lower * step : U32) : U32}
  %Equal.sym(U32,(upper * extent * step : U32),(upper * outer_step : U32),outer) :
    {(base + (_ + lower * step : U32) : U32) == ((base + upper * outer_step : U32) + lower * step : U32) : U32}
  Equal.sym(U32,((base + upper * outer_step : U32) + lower * step : U32),(base + (upper * outer_step + lower * step : U32) : U32),
    Arithmetic.add_associative(base,(upper * outer_step : U32),(lower * step : U32)))

# Two chained axes merged into one keep the box.
def merged_total(+extent: U32,+stride: U32,+back: Bool,+step: U32,+inner_extent: U32,+inner_stride: U32,+inner_back: Bool,+inner_step: U32,
  +tail: List<&2,Move.Axis>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(total(Con{Move.Axis{extent,stride,back,step},Con{Move.Axis{inner_extent,inner_stride,inner_back,inner_step},tail}})),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,total(Con{Move.Axis{extent,stride,back,step},Con{Move.Axis{inner_extent,inner_stride,inner_back,inner_step},tail}})) == True{} : Bool}) ->
  {total(Con{Move.merged(Move.Axis{extent,stride,back,step},Move.Axis{inner_extent,inner_stride,inner_back,inner_step}),tail}) == total(Con{Move.Axis{extent,stride,back,step},Con{Move.Axis{inner_extent,inner_stride,inner_back,inner_step},tail}}) : Nat}:
  +outer_size = U32.to_nat(extent)
  +inner_size = U32.to_nat(inner_extent)
  +last = total(tail)
  +whole = Nat.mul(outer_size,Nat.mul(inner_size,last))
  +below_positive = DivisionOrder.right_factor_positive(outer_size,Nat.mul(inner_size,last),positive)
  +last_positive = DivisionOrder.right_factor_positive(inner_size,last,below_positive)
  +merged_room = {%Equal.sym(Nat,Nat.mul(Nat.mul(outer_size,inner_size),last),Nat.mul(outer_size,Nat.mul(inner_size,last)),RowMajor.assoc(outer_size,inner_size,last)) :
    {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} single_room(whole,cap,room) : {Nat.is_le(Nat.mul(Nat.mul(outer_size,inner_size),last),U32.to_nat(cap)) == True{} : Bool}}
  +pair_room = Order.transitive(Nat.mul(outer_size,inner_size),Nat.mul(Nat.mul(outer_size,inner_size),last),U32.to_nat(cap),
    RowMajor.product_covers(Nat.mul(outer_size,inner_size),last,last_positive),merged_room)
  +pair_exact = Arithmetic.product_exact(cap,cap_ok,extent,inner_extent,outer_size,inner_size,{==},{==},pair_room)
  Equal.trans(Nat,Nat.mul(U32.to_nat((extent * inner_extent : U32)),last),Nat.mul(Nat.mul(outer_size,inner_size),last),whole,
    Equal.cong(Nat,Nat,value => Nat.mul(value,last),U32.to_nat((extent * inner_extent : U32)),Nat.mul(outer_size,inner_size),pair_exact),
    RowMajor.assoc(outer_size,inner_size,last))

# Two chained axes merged into one keep the offsets of every box index.
def merged_offsets(+extent: U32,+stride: U32,+back: Bool,+step: U32,+inner_extent: U32,+inner_stride: U32,+inner_back: Bool,+inner_step: U32,
  +tail: List<&2,Move.Axis>,+found: {Move.chains(Move.Axis{extent,stride,back,step},Move.Axis{inner_extent,inner_stride,inner_back,inner_step}) == True{} : Bool},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(total(Con{Move.Axis{extent,stride,back,step},Con{Move.Axis{inner_extent,inner_stride,inner_back,inner_step},tail}})),U32.to_nat(cap)) == True{} : Bool},
  +index: U32,+inside: {Nat.is_lt(U32.to_nat(index),total(Con{Move.Axis{extent,stride,back,step},Con{Move.Axis{inner_extent,inner_stride,inner_back,inner_step},tail}})) == True{} : Bool},
  +source: U32,+target: U32) ->
  {Move.row_offsets(Con{Move.merged(Move.Axis{extent,stride,back,step},Move.Axis{inner_extent,inner_stride,inner_back,inner_step}),tail},index,source,target) ==
    Move.row_offsets(Con{Move.Axis{extent,stride,back,step},Con{Move.Axis{inner_extent,inner_stride,inner_back,inner_step},tail}},index,source,target) : Move.Offsets}:
  +outer_size = U32.to_nat(extent)
  +inner_size = U32.to_nat(inner_extent)
  +last = total(tail)
  +whole = Nat.mul(outer_size,Nat.mul(inner_size,last))
  +strides = and_left((U32.is_eq(step,(inner_step * inner_extent : U32)) && U32.is_eq(stride,(inner_stride * inner_extent : U32))),Move.same_direction(back,inner_back),found)
  +direction = and_right((U32.is_eq(step,(inner_step * inner_extent : U32)) && U32.is_eq(stride,(inner_stride * inner_extent : U32))),Move.same_direction(back,inner_back),found)
  +step_check = and_left(U32.is_eq(step,(inner_step * inner_extent : U32)),U32.is_eq(stride,(inner_stride * inner_extent : U32)),strides)
  +stride_check = and_right(U32.is_eq(step,(inner_step * inner_extent : U32)),U32.is_eq(stride,(inner_stride * inner_extent : U32)),strides)
  +step_same = Arithmetic.equal_of_is_eq(step,(inner_step * inner_extent : U32),step_check)
  +stride_same = Arithmetic.equal_of_is_eq(stride,(inner_stride * inner_extent : U32),stride_check)
  +back_same = same_back(back,inner_back,direction)
  +positive = Order.positive_bound(U32.to_nat(index),whole,inside)
  +outer_positive = RowMajor.left_factor_positive(outer_size,Nat.mul(inner_size,last),positive)
  +below_positive = DivisionOrder.right_factor_positive(outer_size,Nat.mul(inner_size,last),positive)
  +inner_positive = RowMajor.left_factor_positive(inner_size,last,below_positive)
  +last_positive = DivisionOrder.right_factor_positive(inner_size,last,below_positive)
  +whole_room = single_room(whole,cap,room)
  +below_le = DivisionOrder.le_multiple(outer_size,Nat.mul(inner_size,last),outer_positive)
  +last_le = Order.transitive(last,Nat.mul(inner_size,last),whole,DivisionOrder.le_multiple(inner_size,last,inner_positive),below_le)
  +outer_le = RowMajor.product_covers(outer_size,Nat.mul(inner_size,last),below_positive)
  +inner_le = Order.transitive(inner_size,Nat.mul(inner_size,last),whole,RowMajor.product_covers(inner_size,last,last_positive),below_le)
  +merged_room = {%Equal.sym(Nat,Nat.mul(Nat.mul(outer_size,inner_size),last),Nat.mul(outer_size,Nat.mul(inner_size,last)),RowMajor.assoc(outer_size,inner_size,last)) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} whole_room :
    {Nat.is_le(Nat.mul(Nat.mul(outer_size,inner_size),last),U32.to_nat(cap)) == True{} : Bool}}
  +pair_room = Order.transitive(Nat.mul(outer_size,inner_size),Nat.mul(Nat.mul(outer_size,inner_size),last),U32.to_nat(cap),
    RowMajor.product_covers(Nat.mul(outer_size,inner_size),last,last_positive),merged_room)
  +pair_exact = Arithmetic.product_exact(cap,cap_ok,extent,inner_extent,outer_size,inner_size,{==},{==},pair_room)
  +rest_exact = rows_exact(tail,cap,cap_ok,last_positive,Order.transitive(last,whole,U32.to_nat(cap),last_le,whole_room))
  +inner_cells = Nat.mul(inner_size,last)
  +high = Nat.div(U32.to_nat(index),inner_cells)
  +middle = Nat.mod(U32.to_nat(index),inner_cells)
  +low = Nat.div(middle,last)
  +cell = Nat.mod(middle,last)
  +inner_cells_positive = Order.product_positive(inner_size,last,inner_positive,last_positive)
  +index_exact = DivisionOrder.quotient_exact(U32.to_nat(index),inner_cells,inner_cells_positive)
  +middle_exact = DivisionOrder.quotient_exact(middle,last,last_positive)
  +high_in = DivisionOrder.quotient_below(U32.to_nat(index),outer_size,inner_cells,inner_cells_positive,inside)
  +middle_in = DivisionOrder.remainder_small(U32.to_nat(index),inner_cells,inner_cells_positive)
  +low_in = DivisionOrder.quotient_below(middle,inner_size,last,last_positive,middle_in)
  +cell_in = DivisionOrder.remainder_small(middle,last,last_positive)
  +joined = Nat.add(Nat.mul(high,inner_size),low)
  +joined_in = RowMajor.radix(high,outer_size,low,inner_size,high_in,low_in)
  +joined_exact = Equal.trans(Nat,U32.to_nat(index),Nat.add(Nat.mul(high,inner_cells),middle),Nat.add(Nat.mul(joined,last),cell),index_exact,
    {%Equal.sym(Nat,middle,Nat.add(Nat.mul(low,last),cell),middle_exact) : {Nat.add(Nat.mul(high,inner_cells),_) == Nat.add(Nat.mul(joined,last),cell) : Nat}
      Equal.sym(Nat,Nat.add(Nat.mul(joined,last),cell),Nat.add(Nat.mul(high,inner_cells),Nat.add(Nat.mul(low,last),cell)),regroup(high,inner_size,low,last,cell)) :
      {Nat.add(Nat.mul(high,inner_cells),middle) == Nat.add(Nat.mul(joined,last),cell) : Nat}})
  +middle_room = Order.transitive(middle,inner_cells,U32.to_nat(cap),Bounds.strict_le(middle,inner_cells,middle_in),Order.transitive(inner_cells,whole,U32.to_nat(cap),below_le,whole_room))
  +moved_source = (source + U32.from_nat(high) * Move.directed(back,stride) : U32)
  +moved_target = (target + U32.from_nat(high) * step : U32)
  +source_same = merge_ring(source,high,low,inner_extent,Move.directed(inner_back,inner_stride),Move.directed(back,stride),
    {%back_same : {Move.directed(back,stride) == (Move.directed(_,inner_stride) * inner_extent : U32) : U32}
      {%Equal.sym(U32,stride,(inner_stride * inner_extent : U32),stride_same) : {Move.directed(back,_) == (Move.directed(back,inner_stride) * inner_extent : U32) : U32}
        directed_scaled(back,inner_stride,inner_extent) : {Move.directed(back,stride) == (Move.directed(back,inner_stride) * inner_extent : U32) : U32}} :
      {Move.directed(back,stride) == (Move.directed(inner_back,inner_stride) * inner_extent : U32) : U32}})
  +target_same = merge_ring(target,high,low,inner_extent,inner_step,step,step_same)
  +merged_side = peel((extent * inner_extent : U32),inner_stride,inner_back,inner_step,tail,index,source,target,joined,cell,cap,cap_ok,total(tail),rest_exact,joined_exact,cell_in,
    {%Equal.sym(Nat,U32.to_nat((extent * inner_extent : U32)),Nat.mul(outer_size,inner_size),pair_exact) : {Nat.is_lt(joined,_) == True{} : Bool} joined_in : {Nat.is_lt(joined,U32.to_nat((extent * inner_extent : U32))) == True{} : Bool}},
    double_room(last,whole,cap,last_le,room),
    {%Equal.sym(Nat,U32.to_nat((extent * inner_extent : U32)),Nat.mul(outer_size,inner_size),pair_exact) : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool}
      double_room(Nat.mul(outer_size,inner_size),Nat.mul(Nat.mul(outer_size,inner_size),last),cap,RowMajor.product_covers(Nat.mul(outer_size,inner_size),last,last_positive),
        {%Equal.sym(Nat,Nat.mul(Nat.mul(outer_size,inner_size),last),Nat.mul(outer_size,Nat.mul(inner_size,last)),RowMajor.assoc(outer_size,inner_size,last)) : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool} room :
          {Nat.is_le(Nat.double(Nat.mul(Nat.mul(outer_size,inner_size),last)),U32.to_nat(cap)) == True{} : Bool}}) :
      {Nat.is_le(Nat.double(U32.to_nat((extent * inner_extent : U32))),U32.to_nat(cap)) == True{} : Bool}})
  +outer_side = peel(extent,stride,back,step,Con{Move.Axis{inner_extent,inner_stride,inner_back,inner_step},tail},index,source,target,high,middle,cap,cap_ok,total(Con{Move.Axis{inner_extent,inner_stride,inner_back,inner_step},tail}),
    rows_exact(Con{Move.Axis{inner_extent,inner_stride,inner_back,inner_step},tail},cap,cap_ok,inner_cells_positive,Order.transitive(inner_cells,whole,U32.to_nat(cap),below_le,whole_room)),
    index_exact,middle_in,high_in,double_room(inner_cells,whole,cap,below_le,room),double_room(outer_size,whole,cap,outer_le,room))
  +inner_side = peel(inner_extent,inner_stride,inner_back,inner_step,tail,U32.from_nat(middle),moved_source,moved_target,low,cell,cap,cap_ok,total(tail),rest_exact,
    Equal.trans(Nat,U32.to_nat(U32.from_nat(middle)),middle,Nat.add(Nat.mul(low,last),cell),Bounds.roundtrip(middle,cap,cap_ok,middle_room),middle_exact),
    cell_in,low_in,double_room(last,whole,cap,last_le,room),double_room(inner_size,whole,cap,inner_le,room))
  Equal.trans(Move.Offsets,
      Move.row_offsets(Con{Move.merged(Move.Axis{extent,stride,back,step},Move.Axis{inner_extent,inner_stride,inner_back,inner_step}),tail},index,source,target),
      Move.row_offsets(tail,U32.from_nat(cell),(source + U32.from_nat(joined) * Move.directed(inner_back,inner_stride) : U32),(target + U32.from_nat(joined) * inner_step : U32)),
      Move.row_offsets(Con{Move.Axis{extent,stride,back,step},Con{Move.Axis{inner_extent,inner_stride,inner_back,inner_step},tail}},index,source,target),
      merged_side,
      Equal.sym(Move.Offsets,
        Move.row_offsets(Con{Move.Axis{extent,stride,back,step},Con{Move.Axis{inner_extent,inner_stride,inner_back,inner_step},tail}},index,source,target),
        Move.row_offsets(tail,U32.from_nat(cell),(source + U32.from_nat(joined) * Move.directed(inner_back,inner_stride) : U32),(target + U32.from_nat(joined) * inner_step : U32)),
        Equal.trans(Move.Offsets,
          Move.row_offsets(Con{Move.Axis{extent,stride,back,step},Con{Move.Axis{inner_extent,inner_stride,inner_back,inner_step},tail}},index,source,target),
          Move.row_offsets(Con{Move.Axis{inner_extent,inner_stride,inner_back,inner_step},tail},U32.from_nat(middle),moved_source,moved_target),
          Move.row_offsets(tail,U32.from_nat(cell),(source + U32.from_nat(joined) * Move.directed(inner_back,inner_stride) : U32),(target + U32.from_nat(joined) * inner_step : U32)),
          outer_side,
          {%Equal.sym(U32,(source + U32.from_nat(joined) * Move.directed(inner_back,inner_stride) : U32),
              (moved_source + U32.from_nat(low) * Move.directed(inner_back,inner_stride) : U32),source_same) :
            {Move.row_offsets(Con{Move.Axis{inner_extent,inner_stride,inner_back,inner_step},tail},U32.from_nat(middle),moved_source,moved_target) ==
              Move.row_offsets(tail,U32.from_nat(cell),_,(target + U32.from_nat(joined) * inner_step : U32)) : Move.Offsets}
            {%Equal.sym(U32,(target + U32.from_nat(joined) * inner_step : U32),(moved_target + U32.from_nat(low) * inner_step : U32),target_same) :
              {Move.row_offsets(Con{Move.Axis{inner_extent,inner_stride,inner_back,inner_step},tail},U32.from_nat(middle),moved_source,moved_target) ==
                Move.row_offsets(tail,U32.from_nat(cell),(moved_source + U32.from_nat(low) * Move.directed(inner_back,inner_stride) : U32),_) : Move.Offsets}
              inner_side :
              {Move.row_offsets(Con{Move.Axis{inner_extent,inner_stride,inner_back,inner_step},tail},U32.from_nat(middle),moved_source,moved_target) ==
                Move.row_offsets(tail,U32.from_nat(cell),(moved_source + U32.from_nat(low) * Move.directed(inner_back,inner_stride) : U32),(target + U32.from_nat(joined) * inner_step : U32)) : Move.Offsets}} :
            {Move.row_offsets(Con{Move.Axis{inner_extent,inner_stride,inner_back,inner_step},tail},U32.from_nat(middle),moved_source,moved_target) ==
              Move.row_offsets(tail,U32.from_nat(cell),(source + U32.from_nat(joined) * Move.directed(inner_back,inner_stride) : U32),(target + U32.from_nat(joined) * inner_step : U32)) : Move.Offsets}})))

def chained_offsets(+axis: Move.Axis,+inner: Move.Axis,+tail: List<&2,Move.Axis>,+chain: Bool,+found: {Move.chains(axis,inner) == chain : Bool},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(total(Con{axis,Con{inner,tail}})),U32.to_nat(cap)) == True{} : Bool},
  +index: U32,+inside: {Nat.is_lt(U32.to_nat(index),total(Con{axis,Con{inner,tail}})) == True{} : Bool},+source: U32,+target: U32) ->
  {Move.row_offsets(Move.joined_axis(chain,axis,inner,tail),index,source,target) == Move.row_offsets(Con{axis,Con{inner,tail}},index,source,target) : Move.Offsets}:
  match axis inner chain:
    case Move.Axis{+extent,+stride,+back,+step} Move.Axis{+inner_extent,+inner_stride,+inner_back,+inner_step} True{}:
      merged_offsets(extent,stride,back,step,inner_extent,inner_stride,inner_back,inner_step,tail,found,cap,cap_ok,room,index,inside,source,target)
    case axis inner False{}: {==}

def chained_total(+axis: Move.Axis,+inner: Move.Axis,+tail: List<&2,Move.Axis>,+chain: Bool,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(total(Con{axis,Con{inner,tail}})),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,total(Con{axis,Con{inner,tail}})) == True{} : Bool}) ->
  {total(Move.joined_axis(chain,axis,inner,tail)) == total(Con{axis,Con{inner,tail}}) : Nat}:
  match axis inner chain:
    case Move.Axis{+extent,+stride,+back,+step} Move.Axis{+inner_extent,+inner_stride,+inner_back,+inner_step} True{}:
      merged_total(extent,stride,back,step,inner_extent,inner_stride,inner_back,inner_step,tail,cap,cap_ok,room,positive)
    case axis inner False{}: {==}

# An axis joined to coalesced axes keeps the offsets and the box.
def joined_offsets(+axis: Move.Axis,+list: List<&2,Move.Axis>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(total(Con{axis,list})),U32.to_nat(cap)) == True{} : Bool},
  +index: U32,+inside: {Nat.is_lt(U32.to_nat(index),total(Con{axis,list})) == True{} : Bool},+source: U32,+target: U32) ->
  {Move.row_offsets(Move.joined_coalesced(axis,list),index,source,target) == Move.row_offsets(Con{axis,list},index,source,target) : Move.Offsets}:
  match list:
    case Nil{}: {==}
    case Con{+inner,+tail}: chained_offsets(axis,inner,tail,Move.chains(axis,inner),{==},cap,cap_ok,room,index,inside,source,target)

def joined_total(+axis: Move.Axis,+list: List<&2,Move.Axis>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(total(Con{axis,list})),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,total(Con{axis,list})) == True{} : Bool}) ->
  {total(Move.joined_coalesced(axis,list)) == total(Con{axis,list}) : Nat}:
  match list:
    case Nil{}: {==}
    case Con{+inner,+tail}: chained_total(axis,inner,tail,Move.chains(axis,inner),cap,cap_ok,room,positive)

# An axis of one index left out keeps the offsets and the box.
def single_offsets(+stride: U32,+back: Bool,+step: U32,+list: List<&2,Move.Axis>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(total(list)),U32.to_nat(cap)) == True{} : Bool},+index: U32,+inside: {Nat.is_lt(U32.to_nat(index),total(list)) == True{} : Bool},
  +source: U32,+target: U32) -> {Move.row_offsets(list,index,source,target) == Move.row_offsets(Con{Move.Axis{1,stride,back,step},list},index,source,target) : Move.Offsets}:
  +positive = Order.positive_bound(U32.to_nat(index),total(list),inside)
  +peeled = peel(1,stride,back,step,list,index,source,target,0n,U32.to_nat(index),cap,cap_ok,total(list),
    rows_exact(list,cap,cap_ok,positive,single_room(total(list),cap,room)),{==},inside,{==},room,
    double_room(1n,total(list),cap,Order.positive_one(total(list),positive),room))
  +source_same = Equal.trans(U32,(source + 0 * Move.directed(back,stride) : U32),(source + 0 : U32),source,
    Equal.cong(U32,U32,word => (source + word : U32),(0 * Move.directed(back,stride) : U32),0,Arithmetic.multiply_zero_left(Move.directed(back,stride))),Arithmetic.add_zero(source))
  +target_same = Equal.trans(U32,(target + 0 * step : U32),(target + 0 : U32),target,
    Equal.cong(U32,U32,word => (target + word : U32),(0 * step : U32),0,Arithmetic.multiply_zero_left(step)),Arithmetic.add_zero(target))
  %Arithmetic.native_roundtrip(index) : {Move.row_offsets(list,_,source,target) == Move.row_offsets(Con{Move.Axis{1,stride,back,step},list},index,source,target) : Move.Offsets}
  %source_same : {Move.row_offsets(list,U32.from_nat(U32.to_nat(index)),_,target) == Move.row_offsets(Con{Move.Axis{1,stride,back,step},list},index,source,target) : Move.Offsets}
  %target_same : {Move.row_offsets(list,U32.from_nat(U32.to_nat(index)),(source + 0 * Move.directed(back,stride) : U32),_) ==
    Move.row_offsets(Con{Move.Axis{1,stride,back,step},list},index,source,target) : Move.Offsets}
  Equal.sym(Move.Offsets,Move.row_offsets(Con{Move.Axis{1,stride,back,step},list},index,source,target),
    Move.row_offsets(list,U32.from_nat(U32.to_nat(index)),(source + 0 * Move.directed(back,stride) : U32),(target + 0 * step : U32)),peeled)

def kept_offsets(+extent: U32,+stride: U32,+back: Bool,+step: U32,+list: List<&2,Move.Axis>,+single: Bool,+found: {U32.is_eq(extent,1) == single : Bool},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(total(Con{Move.Axis{extent,stride,back,step},list})),U32.to_nat(cap)) == True{} : Bool},
  +index: U32,+inside: {Nat.is_lt(U32.to_nat(index),total(Con{Move.Axis{extent,stride,back,step},list})) == True{} : Bool},+source: U32,+target: U32) ->
  {Move.row_offsets(Move.kept_coalesced(single,Move.Axis{extent,stride,back,step},list),index,source,target) == Move.row_offsets(Con{Move.Axis{extent,stride,back,step},list},index,source,target) : Move.Offsets}:
  match single:
    case True{}:
      +one = Arithmetic.equal_of_is_eq(extent,1,found)
      +room_one = {%one : {Nat.is_le(Nat.double(Nat.mul(U32.to_nat(_),total(list))),U32.to_nat(cap)) == True{} : Bool} room :
        {Nat.is_le(Nat.double(Nat.add(total(list),0n)),U32.to_nat(cap)) == True{} : Bool}}
      +inside_one = {%one : {Nat.is_lt(U32.to_nat(index),Nat.mul(U32.to_nat(_),total(list))) == True{} : Bool} inside :
        {Nat.is_lt(U32.to_nat(index),Nat.add(total(list),0n)) == True{} : Bool}}
      %Equal.sym(U32,extent,1,one) : {Move.row_offsets(list,index,source,target) == Move.row_offsets(Con{Move.Axis{_,stride,back,step},list},index,source,target) : Move.Offsets}
      single_offsets(stride,back,step,list,cap,cap_ok,
        {%Addition.zero_right(total(list)) : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool} room_one : {Nat.is_le(Nat.double(total(list)),U32.to_nat(cap)) == True{} : Bool}},
        index,{%Addition.zero_right(total(list)) : {Nat.is_lt(U32.to_nat(index),_) == True{} : Bool} inside_one : {Nat.is_lt(U32.to_nat(index),total(list)) == True{} : Bool}},
        source,target)
    case False{}: joined_offsets(Move.Axis{extent,stride,back,step},list,cap,cap_ok,room,index,inside,source,target)

def kept_total(+extent: U32,+stride: U32,+back: Bool,+step: U32,+list: List<&2,Move.Axis>,+single: Bool,+found: {U32.is_eq(extent,1) == single : Bool},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(total(Con{Move.Axis{extent,stride,back,step},list})),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,total(Con{Move.Axis{extent,stride,back,step},list})) == True{} : Bool}) ->
  {total(Move.kept_coalesced(single,Move.Axis{extent,stride,back,step},list)) == total(Con{Move.Axis{extent,stride,back,step},list}) : Nat}:
  match single:
    case True{}:
      +one = Arithmetic.equal_of_is_eq(extent,1,found)
      %Equal.sym(U32,extent,1,one) : {total(list) == Nat.mul(U32.to_nat(_),total(list)) : Nat}
      Equal.sym(Nat,Nat.add(total(list),0n),total(list),Addition.zero_right(total(list)))
    case False{}: joined_total(Move.Axis{extent,stride,back,step},list,cap,cap_ok,room,positive)

# Coalescing keeps the box of the axes.
def coalesced_total(+axes: List<&2,Move.Axis>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+positive: {Nat.is_lt(0n,total(axes)) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(total(axes)),U32.to_nat(cap)) == True{} : Bool}) -> {total(Move.coalesced(axes)) == total(axes) : Nat}:
  match axes:
    case Nil{}: {==}
    case Con{Move.Axis{+extent,+stride,+back,+step},+rest}:
      +axis = {Move.Axis{extent,stride,back,step} : Move.Axis}
      +list = Move.coalesced(rest)
      +extent_positive = RowMajor.left_factor_positive(U32.to_nat(extent),total(rest),positive)
      +rest_positive = DivisionOrder.right_factor_positive(U32.to_nat(extent),total(rest),positive)
      +rest_room = double_room(total(rest),Nat.mul(U32.to_nat(extent),total(rest)),cap,DivisionOrder.le_multiple(U32.to_nat(extent),total(rest),extent_positive),room)
      +same = coalesced_total(rest,cap,cap_ok,rest_positive,rest_room)
      +joined_room = {%Equal.sym(Nat,total(list),total(rest),same) : {Nat.is_le(Nat.double(Nat.mul(U32.to_nat(extent),_)),U32.to_nat(cap)) == True{} : Bool} room :
        {Nat.is_le(Nat.double(total(Con{axis,list})),U32.to_nat(cap)) == True{} : Bool}}
      +joined_positive = {%Equal.sym(Nat,total(list),total(rest),same) : {Nat.is_lt(0n,Nat.mul(U32.to_nat(extent),_)) == True{} : Bool} positive :
        {Nat.is_lt(0n,total(Con{axis,list})) == True{} : Bool}}
      Equal.trans(Nat,total(Move.kept_coalesced(U32.is_eq(extent,1),axis,list)),total(Con{axis,list}),Nat.mul(U32.to_nat(extent),total(rest)),
        kept_total(extent,stride,back,step,list,U32.is_eq(extent,1),{==},cap,cap_ok,joined_room,joined_positive),
        Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(extent),value),total(list),total(rest),same))

# Coalescing keeps the offsets of every box index.
def coalesced_offsets(+axes: List<&2,Move.Axis>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(total(axes)),U32.to_nat(cap)) == True{} : Bool},+index: U32,+inside: {Nat.is_lt(U32.to_nat(index),total(axes)) == True{} : Bool},
  +source: U32,+target: U32) -> {Move.row_offsets(Move.coalesced(axes),index,source,target) == Move.row_offsets(axes,index,source,target) : Move.Offsets}:
  match axes:
    case Nil{}: {==}
    case Con{Move.Axis{+extent,+stride,+back,+step},+rest}:
      +axis = {Move.Axis{extent,stride,back,step} : Move.Axis}
      +list = Move.coalesced(rest)
      +positive = Order.positive_bound(U32.to_nat(index),Nat.mul(U32.to_nat(extent),total(rest)),inside)
      +extent_positive = RowMajor.left_factor_positive(U32.to_nat(extent),total(rest),positive)
      +rest_positive = DivisionOrder.right_factor_positive(U32.to_nat(extent),total(rest),positive)
      +rest_room = double_room(total(rest),Nat.mul(U32.to_nat(extent),total(rest)),cap,DivisionOrder.le_multiple(U32.to_nat(extent),total(rest),extent_positive),room)
      +extent_room = double_room(U32.to_nat(extent),Nat.mul(U32.to_nat(extent),total(rest)),cap,RowMajor.product_covers(U32.to_nat(extent),total(rest),rest_positive),room)
      +same = coalesced_total(rest,cap,cap_ok,rest_positive,rest_room)
      +joined_room = {%Equal.sym(Nat,total(list),total(rest),same) : {Nat.is_le(Nat.double(Nat.mul(U32.to_nat(extent),_)),U32.to_nat(cap)) == True{} : Bool} room :
        {Nat.is_le(Nat.double(total(Con{axis,list})),U32.to_nat(cap)) == True{} : Bool}}
      +joined_inside = {%Equal.sym(Nat,total(list),total(rest),same) : {Nat.is_lt(U32.to_nat(index),Nat.mul(U32.to_nat(extent),_)) == True{} : Bool} inside :
        {Nat.is_lt(U32.to_nat(index),total(Con{axis,list})) == True{} : Bool}}
      +quotient = Nat.div(U32.to_nat(index),total(rest))
      +remainder = Nat.mod(U32.to_nat(index),total(rest))
      +exact = DivisionOrder.quotient_exact(U32.to_nat(index),total(rest),rest_positive)
      +quotient_in = DivisionOrder.quotient_below(U32.to_nat(index),U32.to_nat(extent),total(rest),rest_positive,inside)
      +remainder_in = DivisionOrder.remainder_small(U32.to_nat(index),total(rest),rest_positive)
      +rest_cells = rows_exact(rest,cap,cap_ok,rest_positive,single_room(total(rest),cap,rest_room))
      +list_cells = Equal.trans(Nat,U32.to_nat(Move.rows_of(list)),total(list),total(rest),
        rows_exact(list,cap,cap_ok,{%Equal.sym(Nat,total(list),total(rest),same) : {Nat.is_lt(0n,_) == True{} : Bool} rest_positive : {Nat.is_lt(0n,total(list)) == True{} : Bool}},
          {%Equal.sym(Nat,total(list),total(rest),same) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} single_room(total(rest),cap,rest_room) : {Nat.is_le(total(list),U32.to_nat(cap)) == True{} : Bool}}),
        same)
      +moved_source = (source + U32.from_nat(quotient) * Move.directed(back,stride) : U32)
      +moved_target = (target + U32.from_nat(quotient) * step : U32)
      +remainder_room = Order.transitive(remainder,total(rest),U32.to_nat(cap),Bounds.strict_le(remainder,total(rest),remainder_in),single_room(total(rest),cap,rest_room))
      +later = coalesced_offsets(rest,cap,cap_ok,rest_room,U32.from_nat(remainder),
        {%Equal.sym(Nat,U32.to_nat(U32.from_nat(remainder)),remainder,Bounds.roundtrip(remainder,cap,cap_ok,remainder_room)) : {Nat.is_lt(_,total(rest)) == True{} : Bool} remainder_in :
          {Nat.is_lt(U32.to_nat(U32.from_nat(remainder)),total(rest)) == True{} : Bool}},moved_source,moved_target)
      Equal.trans(Move.Offsets,Move.row_offsets(Move.kept_coalesced(U32.is_eq(extent,1),axis,list),index,source,target),Move.row_offsets(Con{axis,list},index,source,target),
        Move.row_offsets(Con{axis,rest},index,source,target),
        kept_offsets(extent,stride,back,step,list,U32.is_eq(extent,1),{==},cap,cap_ok,joined_room,index,joined_inside,source,target),
        Equal.trans(Move.Offsets,Move.row_offsets(Con{axis,list},index,source,target),Move.row_offsets(list,U32.from_nat(remainder),moved_source,moved_target),
          Move.row_offsets(Con{axis,rest},index,source,target),
          peel(extent,stride,back,step,list,index,source,target,quotient,remainder,cap,cap_ok,total(rest),list_cells,exact,remainder_in,quotient_in,rest_room,extent_room),
          Equal.trans(Move.Offsets,Move.row_offsets(list,U32.from_nat(remainder),moved_source,moved_target),Move.row_offsets(rest,U32.from_nat(remainder),moved_source,moved_target),
            Move.row_offsets(Con{axis,rest},index,source,target),later,
            Equal.sym(Move.Offsets,Move.row_offsets(Con{axis,rest},index,source,target),Move.row_offsets(rest,U32.from_nat(remainder),moved_source,moved_target),
              peel(extent,stride,back,step,rest,index,source,target,quotient,remainder,cap,cap_ok,total(rest),rest_cells,exact,remainder_in,quotient_in,rest_room,extent_room)))))

# ---- Strides and targets of a part ----

# Every axis of more than one index steps at most half a word.
def StridesRoom(axes: List<&2,Move.Axis>,+word: U32) -> Data:
  match axes:
    case Nil{}: Unit
    case Con{Move.Axis{+extent,+stride,+back,+step},rest}:
      Held.Both<Logic.When(U32.is_lt(1,extent),{Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool}),StridesRoom(rest,word)>

# Every axis steps at most half a word.
def AllRoom(axes: List<&2,Move.Axis>,+word: U32) -> Data:
  match axes:
    case Nil{}: Unit
    case Con{Move.Axis{+extent,+stride,+back,+step},rest}:
      Held.Both<{Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool},AllRoom(rest,word)>

def above_zero(+rest: Nat,+other: {Nat.is_eq(1n+rest,1n) == False{} : Bool}) -> {Nat.is_lt(1n,1n+rest) == True{} : Bool}:
  match rest:
    case 0n: Empty.absurd({Nat.is_lt(1n,1n) == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},other)))
    case 1n+more: {==}

def above_one(+value: Nat,+positive: {Nat.is_lt(0n,value) == True{} : Bool},+other: {Nat.is_eq(value,1n) == False{} : Bool}) -> {Nat.is_lt(1n,value) == True{} : Bool}:
  match value:
    case 0n: Empty.absurd({Nat.is_lt(1n,0n) == True{} : Bool},WordEncoding.false_true(positive))
    case 1n+ +rest: above_zero(rest,other)

def chained_room(+axis: Move.Axis,+inner: Move.Axis,+tail: List<&2,Move.Axis>,+chain: Bool,+word: U32,
  +own: {Nat.is_le(Nat.double(U32.to_nat(axis_stride(axis))),U32.to_nat(word)) == True{} : Bool},+later: AllRoom(Con{inner,tail},word)) ->
  AllRoom(Move.joined_axis(chain,axis,inner,tail),word):
  match axis inner chain:
    case Move.Axis{+extent,+stride,+back,+step} Move.Axis{+inner_extent,+inner_stride,+inner_back,+inner_step} True{}:
      later
    case Move.Axis{+extent,+stride,+back,+step} inner False{}:
      Held.Both{own,later}

def joined_room(+axis: Move.Axis,+list: List<&2,Move.Axis>,+word: U32,+own: {Nat.is_le(Nat.double(U32.to_nat(axis_stride(axis))),U32.to_nat(word)) == True{} : Bool},
  +later: AllRoom(list,word)) -> AllRoom(Move.joined_coalesced(axis,list),word):
  match axis list:
    case Move.Axis{+extent,+stride,+back,+step} Nil{}: Held.Both{own,Unit{}}
    case axis Con{+inner,+tail}: chained_room(axis,inner,tail,Move.chains(axis,inner),word,own,later)

def kept_room(+extent: U32,+stride: U32,+back: Bool,+step: U32,+list: List<&2,Move.Axis>,+single: Bool,+found: {U32.is_eq(extent,1) == single : Bool},+word: U32,
  +positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},+own: Logic.When(U32.is_lt(1,extent),{Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool}),
  +later: AllRoom(list,word)) -> AllRoom(Move.kept_coalesced(single,Move.Axis{extent,stride,back,step},list),word):
  match single:
    case True{}: later
    case False{}:
      +other = {%Comparison.eq(extent,1) : {_ == False{} : Bool} found : {Nat.is_eq(U32.to_nat(extent),1n) == False{} : Bool}}
      +more = {%Equal.sym(Bool,U32.is_lt(1,extent),Nat.is_lt(1n,U32.to_nat(extent)),Comparison.lt(1,extent)) : {_ == True{} : Bool}
        above_one(U32.to_nat(extent),positive,other) : {U32.is_lt(1,extent) == True{} : Bool}}
      joined_room(Move.Axis{extent,stride,back,step},list,word,Logic.required({Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool},U32.is_lt(1,extent),own,more),later)

# The coalesced axes of a box with cells step at most half a word.
def coalesced_room(+axes: List<&2,Move.Axis>,+word: U32,+positive: {Nat.is_lt(0n,total(axes)) == True{} : Bool},+strides: StridesRoom(axes,word)) ->
  AllRoom(Move.coalesced(axes),word):
  match axes:
    case Nil{}: Unit{}
    case Con{Move.Axis{+extent,+stride,+back,+step},+rest}:
      Held.Both{+own,+later} = strides
      kept_room(extent,stride,back,step,Move.coalesced(rest),U32.is_eq(extent,1),{==},word,
        RowMajor.left_factor_positive(U32.to_nat(extent),total(rest),positive),own,
        coalesced_room(rest,word,DivisionOrder.right_factor_positive(U32.to_nat(extent),total(rest),positive),later))

def single_room_run(dense: Bool,+extent: U32,+stride: U32,+back: Bool,+step: U32,+word: U32,
  +own: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool}) ->
  {Nat.is_le(Nat.double(U32.to_nat(axis_stride(split_run(Move.split_dense(dense,Move.Axis{extent,stride,back,step}))))),U32.to_nat(word)) == True{} : Bool}:
  match dense:
    case True{}: own
    case False{}: Order.zero_le(U32.to_nat(word))

def single_step(dense: Bool,+extent: U32,+stride: U32,+back: Bool,+step: U32,+found: {U32.is_eq(step,1) == dense : Bool}) ->
  {axis_step(split_run(Move.split_dense(dense,Move.Axis{extent,stride,back,step}))) == 1 : U32}:
  match dense:
    case True{}: Arithmetic.equal_of_is_eq(step,1,found)
    case False{}: {==}

# The run of any axes steps one target cell.
def run_step(+axes: List<&2,Move.Axis>) -> {axis_step(run_of(axes)) == 1 : U32}:
  match axes:
    case Nil{}: {==}
    case Con{Move.Axis{+extent,+stride,+back,+step},Nil{}}:
      single_step(U32.is_eq(step,1),extent,stride,back,step,{==})
    case Con{Move.Axis{+extent,+stride,+back,+step},Con{+next,+more}}:
      %Equal.sym(Move.Axis,run_of(Con{Move.Axis{extent,stride,back,step},Con{next,more}}),run_of(Con{next,more}),run_cons(Move.Axis{extent,stride,back,step},next,more)) : {axis_step(_) == 1 : U32}
      run_step(Con{next,more})

# The run of axes that all step at most half a word steps at most half a word.
def run_room(+axes: List<&2,Move.Axis>,+word: U32,+room: AllRoom(axes,word)) -> {Nat.is_le(Nat.double(U32.to_nat(axis_stride(run_of(axes)))),U32.to_nat(word)) == True{} : Bool}:
  match axes:
    case Nil{}: Order.zero_le(U32.to_nat(word))
    case Con{Move.Axis{+extent,+stride,+back,+step},Nil{}}:
      Held.Both{+own,+later} = room
      single_room_run(U32.is_eq(step,1),extent,stride,back,step,word,own)
    case Con{Move.Axis{+extent,+stride,+back,+step},Con{+next,+more}}:
      Held.Both{+own,+later} = room
      %Equal.sym(Move.Axis,run_of(Con{Move.Axis{extent,stride,back,step},Con{next,more}}),run_of(Con{next,more}),run_cons(Move.Axis{extent,stride,back,step},next,more)) :
        {Nat.is_le(Nat.double(U32.to_nat(axis_stride(_))),U32.to_nat(word)) == True{} : Bool}
      run_room(Con{next,more},word,later)

# Every axis steps in the target by the cells of the axes after it.
def DenseSteps(axes: List<&2,Move.Axis>) -> Data:
  match axes:
    case Nil{}: Unit
    case Con{Move.Axis{+extent,+stride,+back,+step},rest}: Held.Both<{step == Move.rows_of(rest) : U32},DenseSteps(rest)>

# Dense target steps put box index index at target plus index.
def dense_targets(+axes: List<&2,Move.Axis>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(total(axes)),U32.to_nat(cap)) == True{} : Bool},+dense: DenseSteps(axes),+index: U32,
  +inside: {Nat.is_lt(U32.to_nat(index),total(axes)) == True{} : Bool},+source: U32,+target: U32) ->
  {offsets_target(Move.row_offsets(axes,index,source,target)) == (target + index : U32) : U32}:
  match axes:
    case Nil{}:
      +zero = Arithmetic.to_nat_injective(index,0,zero_of_one(U32.to_nat(index),inside))
      %Equal.sym(U32,index,0,zero) : {target == (target + _ : U32) : U32}
      Equal.sym(U32,(target + 0 : U32),target,Arithmetic.add_zero(target))
    case Con{Move.Axis{+extent,+stride,+back,+step},+rest}:
      Held.Both{+own,+later} = dense
      +positive = Order.positive_bound(U32.to_nat(index),Nat.mul(U32.to_nat(extent),total(rest)),inside)
      +extent_positive = RowMajor.left_factor_positive(U32.to_nat(extent),total(rest),positive)
      +rest_positive = DivisionOrder.right_factor_positive(U32.to_nat(extent),total(rest),positive)
      +rest_room = double_room(total(rest),Nat.mul(U32.to_nat(extent),total(rest)),cap,DivisionOrder.le_multiple(U32.to_nat(extent),total(rest),extent_positive),room)
      +extent_room = double_room(U32.to_nat(extent),Nat.mul(U32.to_nat(extent),total(rest)),cap,RowMajor.product_covers(U32.to_nat(extent),total(rest),rest_positive),room)
      +quotient = Nat.div(U32.to_nat(index),total(rest))
      +remainder = Nat.mod(U32.to_nat(index),total(rest))
      +exact = DivisionOrder.quotient_exact(U32.to_nat(index),total(rest),rest_positive)
      +quotient_in = DivisionOrder.quotient_below(U32.to_nat(index),U32.to_nat(extent),total(rest),rest_positive,inside)
      +remainder_in = DivisionOrder.remainder_small(U32.to_nat(index),total(rest),rest_positive)
      +cells = rows_exact(rest,cap,cap_ok,rest_positive,single_room(total(rest),cap,rest_room))
      +remainder_room = Order.transitive(remainder,total(rest),U32.to_nat(cap),Bounds.strict_le(remainder,total(rest),remainder_in),single_room(total(rest),cap,rest_room))
      +moved_source = (source + U32.from_nat(quotient) * Move.directed(back,stride) : U32)
      +moved_target = (target + U32.from_nat(quotient) * step : U32)
      +later_target = dense_targets(rest,cap,cap_ok,rest_room,later,U32.from_nat(remainder),
        {%Equal.sym(Nat,U32.to_nat(U32.from_nat(remainder)),remainder,Bounds.roundtrip(remainder,cap,cap_ok,remainder_room)) : {Nat.is_lt(_,total(rest)) == True{} : Bool}
          remainder_in : {Nat.is_lt(U32.to_nat(U32.from_nat(remainder)),total(rest)) == True{} : Bool}},moved_source,moved_target)
      +step_word = Equal.trans(U32,step,Move.rows_of(rest),U32.from_nat(total(rest)),own,
        Equal.trans(U32,Move.rows_of(rest),U32.from_nat(U32.to_nat(Move.rows_of(rest))),U32.from_nat(total(rest)),
          Equal.sym(U32,U32.from_nat(U32.to_nat(Move.rows_of(rest))),Move.rows_of(rest),Arithmetic.native_roundtrip(Move.rows_of(rest))),
          Equal.cong(Nat,U32,U32.from_nat,U32.to_nat(Move.rows_of(rest)),total(rest),cells)))
      +index_word = Equal.trans(U32,index,U32.from_nat(U32.to_nat(index)),((U32.from_nat(quotient) * step : U32) + U32.from_nat(remainder) : U32),
        Equal.sym(U32,U32.from_nat(U32.to_nat(index)),index,Arithmetic.native_roundtrip(index)),
        Equal.trans(U32,U32.from_nat(U32.to_nat(index)),U32.from_nat(Nat.add(Nat.mul(quotient,total(rest)),remainder)),((U32.from_nat(quotient) * step : U32) + U32.from_nat(remainder) : U32),
          Equal.cong(Nat,U32,U32.from_nat,U32.to_nat(index),Nat.add(Nat.mul(quotient,total(rest)),remainder),exact),
          {%Equal.sym(U32,step,U32.from_nat(total(rest)),step_word) : {U32.from_nat(Nat.add(Nat.mul(quotient,total(rest)),remainder)) == ((U32.from_nat(quotient) * _ : U32) + U32.from_nat(remainder) : U32) : U32}
            Equal.trans(U32,U32.from_nat(Nat.add(Nat.mul(quotient,total(rest)),remainder)),(U32.from_nat(Nat.mul(quotient,total(rest))) + U32.from_nat(remainder) : U32),
              ((U32.from_nat(quotient) * U32.from_nat(total(rest)) : U32) + U32.from_nat(remainder) : U32),
              Equal.sym(U32,(U32.from_nat(Nat.mul(quotient,total(rest))) + U32.from_nat(remainder) : U32),U32.from_nat(Nat.add(Nat.mul(quotient,total(rest)),remainder)),
                WordArithmetic.native_add(Nat.mul(quotient,total(rest)),remainder)),
              Equal.cong(U32,U32,word => (word + U32.from_nat(remainder) : U32),U32.from_nat(Nat.mul(quotient,total(rest))),(U32.from_nat(quotient) * U32.from_nat(total(rest)) : U32),
                Equal.sym(U32,(U32.from_nat(quotient) * U32.from_nat(total(rest)) : U32),U32.from_nat(Nat.mul(quotient,total(rest))),Multiplication.native_encoded_multiply(quotient,total(rest))))) :
            {U32.from_nat(Nat.add(Nat.mul(quotient,total(rest)),remainder)) == ((U32.from_nat(quotient) * step : U32) + U32.from_nat(remainder) : U32) : U32}}))
      +peeled = peel(extent,stride,back,step,rest,index,source,target,quotient,remainder,cap,cap_ok,total(rest),cells,exact,remainder_in,quotient_in,rest_room,extent_room)
      %Equal.sym(Move.Offsets,Move.row_offsets(Con{Move.Axis{extent,stride,back,step},rest},index,source,target),
        Move.row_offsets(rest,U32.from_nat(remainder),moved_source,moved_target),peeled) : {offsets_target(_) == (target + index : U32) : U32}
      %Equal.sym(U32,index,((U32.from_nat(quotient) * step : U32) + U32.from_nat(remainder) : U32),index_word) :
        {offsets_target(Move.row_offsets(rest,U32.from_nat(remainder),moved_source,moved_target)) == (target + _ : U32) : U32}
      Equal.trans(U32,offsets_target(Move.row_offsets(rest,U32.from_nat(remainder),moved_source,moved_target)),(moved_target + U32.from_nat(remainder) : U32),
        (target + ((U32.from_nat(quotient) * step : U32) + U32.from_nat(remainder) : U32) : U32),later_target,
        Arithmetic.add_associative(target,(U32.from_nat(quotient) * step : U32),U32.from_nat(remainder)))

# ---- What a part writes ----

def part_outer(+axes: List<&2,Move.Axis>) -> List<&2,Move.Axis>:
  outer_of(Move.coalesced(axes))

def part_run(+axes: List<&2,Move.Axis>) -> Move.Axis:
  run_of(Move.coalesced(axes))

def part_width(+axes: List<&2,Move.Axis>) -> Nat:
  U32.to_nat(axis_extent(part_run(axes)))

def part_step(+axes: List<&2,Move.Axis>) -> U32:
  Move.directed(axis_back(part_run(axes)),axis_stride(part_run(axes)))

# The source cell of box index index.
def cell_source(+axes: List<&2,Move.Axis>,+index: Nat,+source: U32,+target: U32) -> U32:
  offsets_source(Move.row_offsets(axes,U32.from_nat(index),source,target))

def axis_eta(+axis: Move.Axis) -> {Move.Axis{axis_extent(axis),axis_stride(axis),axis_back(axis),axis_step(axis)} == axis : Move.Axis}:
  match axis:
    case Move.Axis{+extent,+stride,+back,+step}: {==}

# The rows of a part times its run are its box.
def part_total(+axes: List<&2,Move.Axis>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+positive: {Nat.is_lt(0n,total(axes)) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(total(axes)),U32.to_nat(cap)) == True{} : Bool}) -> {Nat.mul(total(part_outer(axes)),part_width(axes)) == total(axes) : Nat}:
  Equal.trans(Nat,Nat.mul(total(part_outer(axes)),part_width(axes)),total(Move.coalesced(axes)),total(axes),
    Equal.sym(Nat,total(Move.coalesced(axes)),Nat.mul(total(part_outer(axes)),part_width(axes)),split_total(Move.coalesced(axes))),
    coalesced_total(axes,cap,cap_ok,positive,room))

# Box index row times the run plus cell is cell of the run of row row.
def box_offsets(+axes: List<&2,Move.Axis>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(total(axes)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,total(axes)) == True{} : Bool},+row: Nat,+cell: Nat,
  +row_in: {Nat.is_lt(row,total(part_outer(axes))) == True{} : Bool},+cell_in: {Nat.is_lt(cell,part_width(axes)) == True{} : Bool},+source: U32,+target: U32) ->
  {Move.row_offsets(axes,U32.from_nat(Nat.add(Nat.mul(row,part_width(axes)),cell)),source,target) ==
    Move.Offsets{(row_source(part_outer(axes),row,source,target) + U32.from_nat(cell) * part_step(axes) : U32),
      (row_target(part_outer(axes),row,source,target) + U32.from_nat(cell) * axis_step(part_run(axes)) : U32)} : Move.Offsets}:
  +index = Nat.add(Nat.mul(row,part_width(axes)),cell)
  +below = RowMajor.radix(row,total(part_outer(axes)),cell,part_width(axes),row_in,cell_in)
  +same = part_total(axes,cap,cap_ok,positive,room)
  +inside = {%same : {Nat.is_lt(index,_) == True{} : Bool} below : {Nat.is_lt(index,total(axes)) == True{} : Bool}}
  +index_room = Order.transitive(index,total(axes),U32.to_nat(cap),Bounds.strict_le(index,total(axes),inside),single_room(total(axes),cap,room))
  +exact = Bounds.roundtrip(index,cap,cap_ok,index_room)
  Equal.trans(Move.Offsets,Move.row_offsets(axes,U32.from_nat(index),source,target),Move.row_offsets(Move.coalesced(axes),U32.from_nat(index),source,target),
    Move.Offsets{(row_source(part_outer(axes),row,source,target) + U32.from_nat(cell) * part_step(axes) : U32),
      (row_target(part_outer(axes),row,source,target) + U32.from_nat(cell) * axis_step(part_run(axes)) : U32)},
    Equal.sym(Move.Offsets,Move.row_offsets(Move.coalesced(axes),U32.from_nat(index),source,target),Move.row_offsets(axes,U32.from_nat(index),source,target),
      coalesced_offsets(axes,cap,cap_ok,room,U32.from_nat(index),{%Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,exact) : {Nat.is_lt(_,total(axes)) == True{} : Bool} inside :
        {Nat.is_lt(U32.to_nat(U32.from_nat(index)),total(axes)) == True{} : Bool}},source,target)),
    split_offsets(Move.coalesced(axes),cap,cap_ok,
      {%Equal.sym(Nat,total(Move.coalesced(axes)),total(axes),coalesced_total(axes,cap,cap_ok,positive,room)) : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool} room :
        {Nat.is_le(Nat.double(total(Move.coalesced(axes))),U32.to_nat(cap)) == True{} : Bool}},
      U32.from_nat(index),row,cell,exact,row_in,cell_in,source,target))

# Every box index from first on, count of them, reads a held cell.
def CellsHeld(+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+ranges: Held.Spans,+first: Nat,count: Nat) -> Data:
  match count:
    case 0n: Unit
    case 1n+rest: Held.Both<Held.SpanHeld(ranges,U32.to_nat(cell_source(axes,first,source,target))),CellsHeld(axes,source,target,ranges,1n+first,rest)>

def cells_drop(+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+ranges: Held.Spans,steps: Nat,+first: Nat,+rest: Nat,
  +cells: CellsHeld(axes,source,target,ranges,first,Nat.add(steps,rest))) -> CellsHeld(axes,source,target,ranges,Nat.add(first,steps),rest):
  match steps:
    case 0n:
      %Equal.sym(Nat,Nat.add(first,0n),first,Addition.zero_right(first)) : CellsHeld(axes,source,target,ranges,_,rest)
      cells
    case 1n+ +more:
      Held.Both{+head,+tail} = cells
      %Equal.sym(Nat,Nat.add(first,1n+more),1n+Nat.add(first,more),Addition.successor_right(first,more)) : CellsHeld(axes,source,target,ranges,_,rest)
      cells_drop(axes,source,target,ranges,more,1n+first,rest,tail)

def next_offset(+base: U32,+cell: Nat,+step: U32) -> {((base + U32.from_nat(cell) * step : U32) + step : U32) == (base + U32.from_nat(1n+cell) * step : U32) : U32}:
  Equal.trans(U32,((base + U32.from_nat(cell) * step : U32) + step : U32),(base + (U32.from_nat(cell) * step + step : U32) : U32),(base + U32.from_nat(1n+cell) * step : U32),
    Arithmetic.add_associative(base,(U32.from_nat(cell) * step : U32),step),
    Equal.cong(U32,U32,word => (base + word : U32),(U32.from_nat(cell) * step + step : U32),(U32.from_nat(1n+cell) * step : U32),
      Equal.trans(U32,(U32.from_nat(cell) * step + step : U32),((U32.from_nat(cell) + 1 : U32) * step : U32),(U32.from_nat(1n+cell) * step : U32),
        Equal.sym(U32,((U32.from_nat(cell) + 1 : U32) * step : U32),(U32.from_nat(cell) * step + step : U32),Arithmetic.multiply_successor(U32.from_nat(cell),step)),
        Equal.cong(U32,U32,word => (word * step : U32),(U32.from_nat(cell) + 1 : U32),U32.from_nat(1n+cell),Bounds.increment_encoded(cell)))))

# The held cells of a row of a part, from cell cell on, n of them.
def run_cells(+axes: List<&2,Move.Axis>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(total(axes)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,total(axes)) == True{} : Bool},
  +source: U32,+target: U32,+ranges: Held.Spans,+row: Nat,+row_in: {Nat.is_lt(row,total(part_outer(axes))) == True{} : Bool},count: Nat,+cell: Nat,+extra: Nat,
  +within: {Nat.is_le(Nat.add(cell,count),part_width(axes)) == True{} : Bool},
  +cells: CellsHeld(axes,source,target,ranges,Nat.add(Nat.mul(row,part_width(axes)),cell),Nat.add(count,extra))) ->
  Held.RunHeld(ranges,(row_source(part_outer(axes),row,source,target) + U32.from_nat(cell) * part_step(axes) : U32),part_step(axes),count):
  match count:
    case 0n: Unit{}
    case 1n+ +more:
      Held.Both{+head,+tail} = cells
      +first = row_source(part_outer(axes),row,source,target)
      +cell_in = Order.room_index(cell,more,part_width(axes),within)
      +offsets = box_offsets(axes,cap,cap_ok,room,positive,row,cell,row_in,cell_in,source,target)
      +same = Equal.cong(Move.Offsets,U32,offsets_source,Move.row_offsets(axes,U32.from_nat(Nat.add(Nat.mul(row,part_width(axes)),cell)),source,target),
        Move.Offsets{(first + U32.from_nat(cell) * part_step(axes) : U32),
          (row_target(part_outer(axes),row,source,target) + U32.from_nat(cell) * axis_step(part_run(axes)) : U32)},offsets)
      +later = run_cells(axes,cap,cap_ok,room,positive,source,target,ranges,row,row_in,more,1n+cell,extra,
        {%Addition.successor_right(cell,more) : {Nat.is_le(_,part_width(axes)) == True{} : Bool} within : {Nat.is_le(Nat.add(1n+cell,more),part_width(axes)) == True{} : Bool}},
        {%Equal.sym(Nat,Nat.add(Nat.mul(row,part_width(axes)),1n+cell),1n+Nat.add(Nat.mul(row,part_width(axes)),cell),Addition.successor_right(Nat.mul(row,part_width(axes)),cell)) : CellsHeld(axes,source,target,ranges,_,Nat.add(more,extra)) tail :
          CellsHeld(axes,source,target,ranges,Nat.add(Nat.mul(row,part_width(axes)),1n+cell),Nat.add(more,extra))})
      Held.Both{{%same : Held.SpanHeld(ranges,U32.to_nat(_)) head : Held.SpanHeld(ranges,U32.to_nat((first + U32.from_nat(cell) * part_step(axes) : U32)))},
        {%Equal.sym(U32,((first + U32.from_nat(cell) * part_step(axes) : U32) + part_step(axes) : U32),(first + U32.from_nat(1n+cell) * part_step(axes) : U32),
            next_offset(first,cell,part_step(axes))) : Held.RunHeld(ranges,_,part_step(axes),more) later :
          Held.RunHeld(ranges,((first + U32.from_nat(cell) * part_step(axes) : U32) + part_step(axes) : U32),part_step(axes),more)}}

# ---- A part as the runtime runs it ----

# The pair after a run: the packed holding of its model and its buffer,
# remodeled.
def ran_pair(~Element: Data,-result: Move.Holding<Element> & Storage.Buffer<Element>,
  +eta: {result == (Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,result),Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,result)) : Move.Holding<Element> & Storage.Buffer<Element>},
  +back: {Buffer.pack(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,result))) == Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,result) : Storage.Buffer<Element>},
  +ran: Held.Ran<Element>,+length: Nat,+facts: Held.RunResult<Element,result,ran,length>) ->
  {result == (Held.pack(~Element,Held.ran_holding(~Element,ran)),Buffer.pack(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,result)))) : Move.Holding<Element> & Storage.Buffer<Element>}:
  %Held.run_held(~Element,result,ran,length,facts) :
    {result == (_,Buffer.pack(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,result)))) : Move.Holding<Element> & Storage.Buffer<Element>}
  %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,result))),Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,result),back) :
    {result == (Pair.fst(Move.Holding<Element>,Storage.Buffer<Element>,result),_) : Move.Holding<Element> & Storage.Buffer<Element>}
  eta

# The buffer after a run, remodeled, holds the tree of its model.
def ran_content(~Element: Data,-result: Move.Holding<Element> & Storage.Buffer<Element>,
  +content: {Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,result))) == Buffer.contents(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,result)) :
    Buffer.Content<Element>},
  +ran: Held.Ran<Element>,+length: Nat,+facts: Held.RunResult<Element,result,ran,length>) ->
  {Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,result))) == Buffer.Content{Held.ran_tree(~Element,ran),length} : Buffer.Content<Element>}:
  %Held.run_target(~Element,result,ran,length,facts) : {Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,result))) == _ : Buffer.Content<Element>}
  content

# The rows of one block, each a run one step of the inner axis after the
# last.
def inner_ran(~Element: Data,fuel: Nat,+step: U32,+inner_target: U32,+extent: U32,+stride: U32,+back: Bool,+source: U32,+position: U32,
  +holding: Held.Holding<Element>,+tree: Tree.Tree<Element>) -> Held.Ran<Element>:
  match fuel:
    case 0n: Held.Ran{holding,tree}
    case 1n+rest:
      +ran = Held.model_run(~Element,U32.to_nat(extent),U32.is_zero(extent),extent,source,stride,back,position,holding,tree)
      inner_ran(~Element,rest,step,inner_target,extent,stride,back,(source + step : U32),(position + inner_target : U32),Held.ran_holding(~Element,ran),
        Held.ran_tree(~Element,ran))

# The rows of the block at offsets.
def block_ran(~Element: Data,+inner: Move.Axis,+extent: U32,+stride: U32,+back: Bool,+offsets: Move.Offsets,+holding: Held.Holding<Element>,
  +tree: Tree.Tree<Element>) -> Held.Ran<Element>:
  inner_ran(~Element,U32.to_nat(axis_extent(inner)),Move.directed(axis_back(inner),axis_stride(inner)),axis_step(inner),extent,stride,back,
    offsets_source(offsets),offsets_target(offsets),holding,tree)

# The blocks from row on of the axes before the inner one.
def blocks_ran(~Element: Data,fuel: Nat,+row: Nat,+outer: List<&2,Move.Axis>,+inner: Move.Axis,+extent: U32,+stride: U32,+back: Bool,+source: U32,
  +target: U32,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>) -> Held.Ran<Element>:
  match fuel:
    case 0n: Held.Ran{holding,tree}
    case 1n+rest:
      +ran = block_ran(~Element,inner,extent,stride,back,Move.row_offsets(outer,U32.from_nat(row),source,target),holding,tree)
      blocks_ran(~Element,rest,1n+row,outer,inner,extent,stride,back,source,target,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran))

# The blocks of a nest of axes before the run.
def nested_ran(~Element: Data,+nest: Move.Nest,+extent: U32,+stride: U32,+back: Bool,+source: U32,+target: U32,+holding: Held.Holding<Element>,
  +tree: Tree.Tree<Element>) -> Held.Ran<Element>:
  blocks_ran(~Element,U32.to_nat(Move.rows_of(nest_outer(nest))),0n,nest_outer(nest),nest_inner(nest),extent,stride,back,source,target,holding,tree)

# A run of the one held piece, copied from local cells on into the target
# from position.
def run_copy(~Element: Data,+extent: U32,+local: U32,+stride: U32,+position: U32,+input: Tree.Tree<Element>,+tree: Tree.Tree<Element>) -> Tree.Tree<Element>:
  Writes.pure(Element,CopyValues.copy_writes(~Element,U32.is_eq(stride,1) && Bool.not(False{}),U32.to_nat(extent),
    Views.View{[Views.Axis{extent,Move.directed(False{},stride)}],local},position,input),tree)

# The rows of one block of the one held piece.
def piece_inner_tree(~Element: Data,fuel: Nat,+step: U32,+inner_target: U32,+extent: U32,+stride: U32,+local: U32,+position: U32,+input: Tree.Tree<Element>,
  +tree: Tree.Tree<Element>) -> Tree.Tree<Element>:
  match fuel:
    case 0n: tree
    case 1n+rest:
      piece_inner_tree(~Element,rest,step,inner_target,extent,stride,(local + step : U32),(position + inner_target : U32),input,
        run_copy(~Element,extent,local,stride,position,input,tree))

def piece_block_tree(~Element: Data,+inner: Move.Axis,+extent: U32,+stride: U32,+offsets: Move.Offsets,+input: Tree.Tree<Element>,+tree: Tree.Tree<Element>) ->
  Tree.Tree<Element>:
  piece_inner_tree(~Element,U32.to_nat(axis_extent(inner)),Move.directed(axis_back(inner),axis_stride(inner)),axis_step(inner),extent,stride,
    offsets_source(offsets),offsets_target(offsets),input,tree)

def piece_rows_tree(~Element: Data,fuel: Nat,+row: Nat,+outer: List<&2,Move.Axis>,+inner: Move.Axis,+extent: U32,+stride: U32,+local: U32,+target: U32,
  +input: Tree.Tree<Element>,+tree: Tree.Tree<Element>) -> Tree.Tree<Element>:
  match fuel:
    case 0n: tree
    case 1n+rest:
      piece_rows_tree(~Element,rest,1n+row,outer,inner,extent,stride,local,target,input,
        piece_block_tree(~Element,inner,extent,stride,Move.row_offsets(outer,U32.from_nat(row),local,target),input,tree))

def piece_nested_tree(~Element: Data,+nest: Move.Nest,+extent: U32,+stride: U32,+local: U32,+target: U32,+input: Tree.Tree<Element>,+tree: Tree.Tree<Element>) ->
  Tree.Tree<Element>:
  piece_rows_tree(~Element,U32.to_nat(Move.rows_of(nest_outer(nest))),0n,nest_outer(nest),nest_inner(nest),extent,stride,local,target,input,tree)

# The current piece holds the whole part, which it copies without moving, or
# the part runs through the holding.
def piece_ran(~Element: Data,inside: Bool,+nest: Move.Nest,+extent: U32,+stride: U32,+back: Bool,+source: U32,+target: U32,+behind: Held.Stack<Element>,
  +buffer: Buffer.Model<Element>,+first: U32,+span: U32,+below: Held.Stack<Element>,+tree: Tree.Tree<Element>) -> Held.Ran<Element>:
  match inside:
    case True{}:
      Held.Ran{Held.Holding{behind,Held.Held{buffer,first,span,below}},
        piece_nested_tree(~Element,nest,extent,stride,(source - first : U32),target,Buffer.tree(~Element,buffer),tree)}
    case False{}: nested_ran(~Element,nest,extent,stride,back,source,target,Held.Holding{behind,Held.Held{buffer,first,span,below}},tree)

def current_ran(~Element: Data,+outer: List<&2,Move.Axis>,+run: Move.Axis,+source: U32,+target: U32,holding: Held.Holding<Element>,+tree: Tree.Tree<Element>) ->
  Held.Ran<Element>:
  match holding:
    case Held.Holding{+behind,Held.Held{+buffer,+first,+span,+below}}:
      piece_ran(~Element,Move.fits_from(outer,run,source,first,span),Move.nest_of(outer),axis_extent(run),axis_stride(run),axis_back(run),source,target,
        behind,buffer,first,span,below,tree)
    case Held.Holding{+behind,Held.Bottom{}}:
      nested_ran(~Element,Move.nest_of(outer),axis_extent(run),axis_stride(run),axis_back(run),source,target,Held.Holding{behind,Held.Bottom{}},tree)

# The part from its held pieces into a buffer tree, after the piece of its
# first source cell is current.
def part_ran(~Element: Data,+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>) -> Held.Ran<Element>:
  current_ran(~Element,part_outer(axes),part_run(axes),source,target,Held.resolve(~Element,holding,source),tree)

# A runtime block of rows on a packed holding into a packed target is its
# model.
def inner_refines(~Element: Data,fuel: Nat,+step: U32,+inner_target: U32,+extent: U32,+stride: U32,+back: Bool,+target_stride: U32,+source: U32,+position: U32,
  +holding: Held.Holding<Element>,+model: Buffer.Model<Element>) ->
  Held.RunResult<Element,Move.inner_rows(~Element,fuel,step,inner_target,Move.Axis{extent,stride,back,target_stride},source,position,(Held.pack(~Element,holding),Buffer.pack(~Element,model))),
    inner_ran(~Element,fuel,step,inner_target,extent,stride,back,source,position,holding,Buffer.tree(~Element,model)),Buffer.length(~Element,model)>:
  match fuel:
    case 0n: Held.RunResult{{==},Held.content_parts(~Element,model)}
    case 1n+ +rest:
      +ran = Held.model_run(~Element,U32.to_nat(extent),U32.is_zero(extent),extent,source,stride,back,position,holding,Buffer.tree(~Element,model))
      +next = Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.run(~Element,U32.to_nat(extent),extent,source,stride,back,position,Held.pack(~Element,holding),Buffer.pack(~Element,model))))
      +facts = Held.run_refines(~Element,U32.to_nat(extent),U32.is_zero(extent),extent,source,stride,back,position,holding,model)
      +content = run_content(~Element,extent,source,stride,back,position,holding,model)
      +same_tree = model_tree(~Element,next,Held.ran_tree(~Element,ran),Buffer.length(~Element,model),content)
      +same_length = model_length(~Element,next,Held.ran_tree(~Element,ran),Buffer.length(~Element,model),content)
      +pair = ran_pair(~Element,Move.run(~Element,U32.to_nat(extent),extent,source,stride,back,position,Held.pack(~Element,holding),Buffer.pack(~Element,model)),Held.pair_eta(~Move.Holding<Element>,~Storage.Buffer<Element>,Move.run(~Element,U32.to_nat(extent),extent,source,stride,back,position,Held.pack(~Element,holding),Buffer.pack(~Element,model))),Buffer.remodeled(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.run(~Element,U32.to_nat(extent),extent,source,stride,back,position,Held.pack(~Element,holding),Buffer.pack(~Element,model)))),ran,Buffer.length(~Element,model),facts)
      %Equal.sym(Move.Holding<Element> & Storage.Buffer<Element>,Move.run(~Element,U32.to_nat(extent),extent,source,stride,back,position,Held.pack(~Element,holding),Buffer.pack(~Element,model)),(Held.pack(~Element,Held.ran_holding(~Element,ran)),Buffer.pack(~Element,next)),pair) :
        Held.RunResult<Element,Move.inner_rows(~Element,rest,step,inner_target,Move.Axis{extent,stride,back,target_stride},(source + step : U32),(position + inner_target : U32),_),
          inner_ran(~Element,rest,step,inner_target,extent,stride,back,(source + step : U32),(position + inner_target : U32),Held.ran_holding(~Element,ran),
            Held.ran_tree(~Element,ran)),Buffer.length(~Element,model)>
      %same_tree :
        Held.RunResult<Element,Move.inner_rows(~Element,rest,step,inner_target,Move.Axis{extent,stride,back,target_stride},(source + step : U32),(position + inner_target : U32),
            (Held.pack(~Element,Held.ran_holding(~Element,ran)),Buffer.pack(~Element,next))),
          inner_ran(~Element,rest,step,inner_target,extent,stride,back,(source + step : U32),(position + inner_target : U32),Held.ran_holding(~Element,ran),_),Buffer.length(~Element,model)>
      %same_length :
        Held.RunResult<Element,Move.inner_rows(~Element,rest,step,inner_target,Move.Axis{extent,stride,back,target_stride},(source + step : U32),(position + inner_target : U32),
            (Held.pack(~Element,Held.ran_holding(~Element,ran)),Buffer.pack(~Element,next))),
          inner_ran(~Element,rest,step,inner_target,extent,stride,back,(source + step : U32),(position + inner_target : U32),Held.ran_holding(~Element,ran),
            Buffer.tree(~Element,next)),_>
      inner_refines(~Element,rest,step,inner_target,extent,stride,back,target_stride,(source + step : U32),(position + inner_target : U32),
        Held.ran_holding(~Element,ran),next)

# The runtime blocks of rows from row on are their model.
def blocks_refines(~Element: Data,fuel: Nat,+row: Nat,+outer: List<&2,Move.Axis>,+inner_extent: U32,+inner_stride: U32,+inner_back: Bool,+inner_step: U32,+extent: U32,+stride: U32,+back: Bool,+target_stride: U32,+source: U32,+target: U32,
  +holding: Held.Holding<Element>,+model: Buffer.Model<Element>) ->
  Held.RunResult<Element,Move.rows(~Element,fuel,U32.from_nat(row),outer,Move.Axis{inner_extent,inner_stride,inner_back,inner_step},Move.Axis{extent,stride,back,target_stride},source,target,(Held.pack(~Element,holding),Buffer.pack(~Element,model))),
    blocks_ran(~Element,fuel,row,outer,Move.Axis{inner_extent,inner_stride,inner_back,inner_step},extent,stride,back,source,target,holding,Buffer.tree(~Element,model)),Buffer.length(~Element,model)>:
  match fuel:
    case 0n: Held.RunResult{{==},Held.content_parts(~Element,model)}
    case 1n+ +rest:
      +offsets = Move.row_offsets(outer,U32.from_nat(row),source,target)
      +first = offsets_source(offsets)
      +position = offsets_target(offsets)
      +ran = inner_ran(~Element,U32.to_nat(inner_extent),Move.directed(inner_back,inner_stride),inner_step,extent,stride,back,first,position,holding,Buffer.tree(~Element,model))
      +facts = inner_refines(~Element,U32.to_nat(inner_extent),Move.directed(inner_back,inner_stride),inner_step,extent,stride,back,target_stride,first,position,
        holding,model)
      +next = Buffer.remodel(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.inner_rows(~Element,U32.to_nat(inner_extent),Move.directed(inner_back,inner_stride),inner_step,Move.Axis{extent,stride,back,target_stride},first,position,(Held.pack(~Element,holding),Buffer.pack(~Element,model)))))
      +content = ran_content(~Element,Move.inner_rows(~Element,U32.to_nat(inner_extent),Move.directed(inner_back,inner_stride),inner_step,Move.Axis{extent,stride,back,target_stride},first,position,(Held.pack(~Element,holding),Buffer.pack(~Element,model))),Buffer.remodel_content(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.inner_rows(~Element,U32.to_nat(inner_extent),Move.directed(inner_back,inner_stride),inner_step,Move.Axis{extent,stride,back,target_stride},first,position,(Held.pack(~Element,holding),Buffer.pack(~Element,model))))),ran,Buffer.length(~Element,model),facts)
      +same_tree = model_tree(~Element,next,Held.ran_tree(~Element,ran),Buffer.length(~Element,model),content)
      +same_length = model_length(~Element,next,Held.ran_tree(~Element,ran),Buffer.length(~Element,model),content)
      +pair = ran_pair(~Element,Move.inner_rows(~Element,U32.to_nat(inner_extent),Move.directed(inner_back,inner_stride),inner_step,Move.Axis{extent,stride,back,target_stride},first,position,(Held.pack(~Element,holding),Buffer.pack(~Element,model))),Held.pair_eta(~Move.Holding<Element>,~Storage.Buffer<Element>,Move.inner_rows(~Element,U32.to_nat(inner_extent),Move.directed(inner_back,inner_stride),inner_step,Move.Axis{extent,stride,back,target_stride},first,position,(Held.pack(~Element,holding),Buffer.pack(~Element,model)))),Buffer.remodeled(~Element,Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.inner_rows(~Element,U32.to_nat(inner_extent),Move.directed(inner_back,inner_stride),inner_step,Move.Axis{extent,stride,back,target_stride},first,position,(Held.pack(~Element,holding),Buffer.pack(~Element,model))))),ran,Buffer.length(~Element,model),facts)
      %offsets_eta(offsets) :
        Held.RunResult<Element,Move.rows(~Element,rest,(U32.from_nat(row) + 1 : U32),outer,Move.Axis{inner_extent,inner_stride,inner_back,inner_step},Move.Axis{extent,stride,back,target_stride},source,target,Move.block_rows(~Element,Move.Axis{inner_extent,inner_stride,inner_back,inner_step},Move.Axis{extent,stride,back,target_stride},_,(Held.pack(~Element,holding),Buffer.pack(~Element,model)))),
          blocks_ran(~Element,1n+rest,row,outer,Move.Axis{inner_extent,inner_stride,inner_back,inner_step},extent,stride,back,source,target,holding,Buffer.tree(~Element,model)),Buffer.length(~Element,model)>
      %Equal.sym(Move.Holding<Element> & Storage.Buffer<Element>,Move.inner_rows(~Element,U32.to_nat(inner_extent),Move.directed(inner_back,inner_stride),inner_step,Move.Axis{extent,stride,back,target_stride},first,position,(Held.pack(~Element,holding),Buffer.pack(~Element,model))),(Held.pack(~Element,Held.ran_holding(~Element,ran)),Buffer.pack(~Element,next)),pair) :
        Held.RunResult<Element,Move.rows(~Element,rest,(U32.from_nat(row) + 1 : U32),outer,Move.Axis{inner_extent,inner_stride,inner_back,inner_step},Move.Axis{extent,stride,back,target_stride},source,target,_),
          blocks_ran(~Element,1n+rest,row,outer,Move.Axis{inner_extent,inner_stride,inner_back,inner_step},extent,stride,back,source,target,holding,Buffer.tree(~Element,model)),Buffer.length(~Element,model)>
      %Equal.sym(U32,(U32.from_nat(row) + 1 : U32),U32.from_nat(1n+row),Bounds.increment_encoded(row)) :
        Held.RunResult<Element,Move.rows(~Element,rest,_,outer,Move.Axis{inner_extent,inner_stride,inner_back,inner_step},Move.Axis{extent,stride,back,target_stride},source,target,(Held.pack(~Element,Held.ran_holding(~Element,ran)),Buffer.pack(~Element,next))),
          blocks_ran(~Element,rest,1n+row,outer,Move.Axis{inner_extent,inner_stride,inner_back,inner_step},extent,stride,back,source,target,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran)),Buffer.length(~Element,model)>
      %same_tree :
        Held.RunResult<Element,Move.rows(~Element,rest,U32.from_nat(1n+row),outer,Move.Axis{inner_extent,inner_stride,inner_back,inner_step},Move.Axis{extent,stride,back,target_stride},source,target,(Held.pack(~Element,Held.ran_holding(~Element,ran)),Buffer.pack(~Element,next))),
          blocks_ran(~Element,rest,1n+row,outer,Move.Axis{inner_extent,inner_stride,inner_back,inner_step},extent,stride,back,source,target,Held.ran_holding(~Element,ran),_),Buffer.length(~Element,model)>
      %same_length :
        Held.RunResult<Element,Move.rows(~Element,rest,U32.from_nat(1n+row),outer,Move.Axis{inner_extent,inner_stride,inner_back,inner_step},Move.Axis{extent,stride,back,target_stride},source,target,(Held.pack(~Element,Held.ran_holding(~Element,ran)),Buffer.pack(~Element,next))),
          blocks_ran(~Element,rest,1n+row,outer,Move.Axis{inner_extent,inner_stride,inner_back,inner_step},extent,stride,back,source,target,Held.ran_holding(~Element,ran),Buffer.tree(~Element,next)),_>
      blocks_refines(~Element,rest,1n+row,outer,inner_extent,inner_stride,inner_back,inner_step,extent,stride,back,target_stride,source,target,
        Held.ran_holding(~Element,ran),next)

# The runtime nest of rows is its model.
def nested_refines(~Element: Data,+outer: List<&2,Move.Axis>,+inner_extent: U32,+inner_stride: U32,+inner_back: Bool,+inner_step: U32,+extent: U32,+stride: U32,+back: Bool,+target_stride: U32,+source: U32,+target: U32,
  +holding: Held.Holding<Element>,+model: Buffer.Model<Element>) ->
  Held.RunResult<Element,Move.nested_rows(~Element,Move.Nest{outer,Move.Axis{inner_extent,inner_stride,inner_back,inner_step}},Move.Axis{extent,stride,back,target_stride},source,target,(Held.pack(~Element,holding),Buffer.pack(~Element,model))),
    nested_ran(~Element,Move.Nest{outer,Move.Axis{inner_extent,inner_stride,inner_back,inner_step}},extent,stride,back,source,target,holding,Buffer.tree(~Element,model)),Buffer.length(~Element,model)>:
  blocks_refines(~Element,U32.to_nat(Move.rows_of(outer)),0n,outer,inner_extent,inner_stride,inner_back,inner_step,extent,stride,back,target_stride,source,target,
    holding,model)

# A copy from one buffer into another: the source buffer back and the target
# holding a tree.
type CopyResult<-Element: Data,-result: Storage.Buffer<Element> & Storage.Buffer<Element>,-input: Buffer.Model<Element>,-tree: Tree.Tree<Element>,-length: Nat> is Data:
  CopyResult{input: {Pair.fst(Storage.Buffer<Element>,Storage.Buffer<Element>,result) == Buffer.pack(~Element,input) : Storage.Buffer<Element>},
    target: {Buffer.contents(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,result)) == Buffer.Content{tree,length} : Buffer.Content<Element>}}

def copy_input(~Element: Data,-result: Storage.Buffer<Element> & Storage.Buffer<Element>,-input: Buffer.Model<Element>,-tree: Tree.Tree<Element>,-length: Nat,+facts: CopyResult<Element,result,input,tree,length>) ->
  {Pair.fst(Storage.Buffer<Element>,Storage.Buffer<Element>,result) == Buffer.pack(~Element,input) : Storage.Buffer<Element>}:
  CopyResult{+same,+target} = facts
  same

def copy_target(~Element: Data,-result: Storage.Buffer<Element> & Storage.Buffer<Element>,-input: Buffer.Model<Element>,-tree: Tree.Tree<Element>,-length: Nat,+facts: CopyResult<Element,result,input,tree,length>) ->
  {Buffer.contents(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,result)) == Buffer.Content{tree,length} : Buffer.Content<Element>}:
  CopyResult{+same,+target} = facts
  target

# The pair after a copy: the source buffer and the target, remodeled.
def copy_pair(~Element: Data,-result: Storage.Buffer<Element> & Storage.Buffer<Element>,+eta: {result == (Pair.fst(Storage.Buffer<Element>,Storage.Buffer<Element>,result),Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,result)) : Storage.Buffer<Element> & Storage.Buffer<Element>},
  +back: {Buffer.pack(~Element,Buffer.remodel(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,result))) == Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,result) : Storage.Buffer<Element>},
  +input: Buffer.Model<Element>,+tree: Tree.Tree<Element>,+length: Nat,+facts: CopyResult<Element,result,input,tree,length>) ->
  {result == (Buffer.pack(~Element,input),Buffer.pack(~Element,Buffer.remodel(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,result)))) : Storage.Buffer<Element> & Storage.Buffer<Element>}:
  %copy_input(~Element,result,input,tree,length,facts) : {result == (_,Buffer.pack(~Element,Buffer.remodel(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,result)))) : Storage.Buffer<Element> & Storage.Buffer<Element>}
  %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,Buffer.remodel(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,result))),Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,result),back) :
    {result == (Pair.fst(Storage.Buffer<Element>,Storage.Buffer<Element>,result),_) : Storage.Buffer<Element> & Storage.Buffer<Element>}
  eta

def copy_content(~Element: Data,-result: Storage.Buffer<Element> & Storage.Buffer<Element>,
  +content: {Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,result))) == Buffer.contents(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,result)) :
    Buffer.Content<Element>},
  +input: Buffer.Model<Element>,+tree: Tree.Tree<Element>,+length: Nat,+facts: CopyResult<Element,result,input,tree,length>) ->
  {Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,result))) == Buffer.Content{tree,length} : Buffer.Content<Element>}:
  %copy_target(~Element,result,input,tree,length,facts) : {Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,result))) == _ : Buffer.Content<Element>}
  content

# One run copied from the held piece.
def run_copied(~Element: Data,+extent: U32,+local: U32,+stride: U32,+position: U32,+input: Buffer.Model<Element>,+model: Buffer.Model<Element>) ->
  CopyResult<Element,Move.copied_run(~Element,extent,local,stride,False{},position,Buffer.pack(~Element,input),Buffer.pack(~Element,model)),input,run_copy(~Element,extent,local,stride,position,Buffer.tree(~Element,input),Buffer.tree(~Element,model)),Buffer.length(~Element,model)>:
  CopyResult{Equal.cong(Storage.Buffer<Element> & Storage.Buffer<Element>,Storage.Buffer<Element>,pair => Pair.fst(Storage.Buffer<Element>,Storage.Buffer<Element>,pair),Move.copied_run(~Element,extent,local,stride,False{},position,Buffer.pack(~Element,input),Buffer.pack(~Element,model)),(Buffer.pack(~Element,input),Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,Move.copied_run(~Element,extent,local,stride,False{},position,Buffer.pack(~Element,input),Buffer.pack(~Element,model)))),
      Held.copied_step_pair(~Element,extent,local,stride,False{},position,input,model)),
    Held.copy_step(~Element,extent,local,stride,False{},position,input,model)}

# The runtime rows of one block of the held piece are their model.
def piece_inner_refines(~Element: Data,fuel: Nat,+step: U32,+inner_target: U32,+extent: U32,+stride: U32,+back: Bool,+target_stride: U32,+local: U32,+position: U32,
  +input: Buffer.Model<Element>,+model: Buffer.Model<Element>) ->
  CopyResult<Element,Move.piece_inner(~Element,fuel,step,inner_target,Move.Axis{extent,stride,back,target_stride},local,position,(Buffer.pack(~Element,input),Buffer.pack(~Element,model))),input,
    piece_inner_tree(~Element,fuel,step,inner_target,extent,stride,local,position,Buffer.tree(~Element,input),Buffer.tree(~Element,model)),Buffer.length(~Element,model)>:
  match fuel:
    case 0n: CopyResult{{==},Held.content_parts(~Element,model)}
    case 1n+ +rest:
      +copied = run_copy(~Element,extent,local,stride,position,Buffer.tree(~Element,input),Buffer.tree(~Element,model))
      +facts = run_copied(~Element,extent,local,stride,position,input,model)
      +next = Buffer.remodel(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,Move.copied_run(~Element,extent,local,stride,False{},position,Buffer.pack(~Element,input),Buffer.pack(~Element,model))))
      +content = copy_content(~Element,Move.copied_run(~Element,extent,local,stride,False{},position,Buffer.pack(~Element,input),Buffer.pack(~Element,model)),Buffer.remodel_content(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,Move.copied_run(~Element,extent,local,stride,False{},position,Buffer.pack(~Element,input),Buffer.pack(~Element,model)))),input,copied,Buffer.length(~Element,model),facts)
      +same_tree = model_tree(~Element,next,copied,Buffer.length(~Element,model),content)
      +same_length = model_length(~Element,next,copied,Buffer.length(~Element,model),content)
      +pair = copy_pair(~Element,Move.copied_run(~Element,extent,local,stride,False{},position,Buffer.pack(~Element,input),Buffer.pack(~Element,model)),Held.pair_eta(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Move.copied_run(~Element,extent,local,stride,False{},position,Buffer.pack(~Element,input),Buffer.pack(~Element,model))),Buffer.remodeled(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,Move.copied_run(~Element,extent,local,stride,False{},position,Buffer.pack(~Element,input),Buffer.pack(~Element,model)))),input,copied,Buffer.length(~Element,model),facts)
      %Equal.sym(Storage.Buffer<Element> & Storage.Buffer<Element>,Move.copied_run(~Element,extent,local,stride,False{},position,Buffer.pack(~Element,input),Buffer.pack(~Element,model)),(Buffer.pack(~Element,input),Buffer.pack(~Element,next)),pair) :
        CopyResult<Element,Move.piece_inner(~Element,rest,step,inner_target,Move.Axis{extent,stride,back,target_stride},(local + step : U32),(position + inner_target : U32),_),input,
          piece_inner_tree(~Element,rest,step,inner_target,extent,stride,(local + step : U32),(position + inner_target : U32),Buffer.tree(~Element,input),copied),Buffer.length(~Element,model)>
      %same_tree :
        CopyResult<Element,Move.piece_inner(~Element,rest,step,inner_target,Move.Axis{extent,stride,back,target_stride},(local + step : U32),(position + inner_target : U32),(Buffer.pack(~Element,input),Buffer.pack(~Element,next))),input,
          piece_inner_tree(~Element,rest,step,inner_target,extent,stride,(local + step : U32),(position + inner_target : U32),Buffer.tree(~Element,input),_),Buffer.length(~Element,model)>
      %same_length :
        CopyResult<Element,Move.piece_inner(~Element,rest,step,inner_target,Move.Axis{extent,stride,back,target_stride},(local + step : U32),(position + inner_target : U32),(Buffer.pack(~Element,input),Buffer.pack(~Element,next))),input,
          piece_inner_tree(~Element,rest,step,inner_target,extent,stride,(local + step : U32),(position + inner_target : U32),Buffer.tree(~Element,input),Buffer.tree(~Element,next)),_>
      piece_inner_refines(~Element,rest,step,inner_target,extent,stride,back,target_stride,(local + step : U32),(position + inner_target : U32),input,next)

# The runtime blocks of the held piece from row on are their model.
def piece_rows_refines(~Element: Data,fuel: Nat,+row: Nat,+outer: List<&2,Move.Axis>,+inner_extent: U32,+inner_stride: U32,+inner_back: Bool,+inner_step: U32,+extent: U32,+stride: U32,+back: Bool,+target_stride: U32,+local: U32,+target: U32,
  +input: Buffer.Model<Element>,+model: Buffer.Model<Element>) ->
  CopyResult<Element,Move.piece_rows(~Element,fuel,U32.from_nat(row),outer,Move.Axis{inner_extent,inner_stride,inner_back,inner_step},Move.Axis{extent,stride,back,target_stride},local,target,(Buffer.pack(~Element,input),Buffer.pack(~Element,model))),input,
    piece_rows_tree(~Element,fuel,row,outer,Move.Axis{inner_extent,inner_stride,inner_back,inner_step},extent,stride,local,target,Buffer.tree(~Element,input),Buffer.tree(~Element,model)),Buffer.length(~Element,model)>:
  match fuel:
    case 0n: CopyResult{{==},Held.content_parts(~Element,model)}
    case 1n+ +rest:
      +offsets = Move.row_offsets(outer,U32.from_nat(row),local,target)
      +first = offsets_source(offsets)
      +position = offsets_target(offsets)
      +copied = piece_inner_tree(~Element,U32.to_nat(inner_extent),Move.directed(inner_back,inner_stride),inner_step,extent,stride,first,position,Buffer.tree(~Element,input),Buffer.tree(~Element,model))
      +facts = piece_inner_refines(~Element,U32.to_nat(inner_extent),Move.directed(inner_back,inner_stride),inner_step,extent,stride,back,target_stride,first,position,
        input,model)
      +next = Buffer.remodel(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,Move.piece_inner(~Element,U32.to_nat(inner_extent),Move.directed(inner_back,inner_stride),inner_step,Move.Axis{extent,stride,back,target_stride},first,position,(Buffer.pack(~Element,input),Buffer.pack(~Element,model)))))
      +content = copy_content(~Element,Move.piece_inner(~Element,U32.to_nat(inner_extent),Move.directed(inner_back,inner_stride),inner_step,Move.Axis{extent,stride,back,target_stride},first,position,(Buffer.pack(~Element,input),Buffer.pack(~Element,model))),Buffer.remodel_content(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,Move.piece_inner(~Element,U32.to_nat(inner_extent),Move.directed(inner_back,inner_stride),inner_step,Move.Axis{extent,stride,back,target_stride},first,position,(Buffer.pack(~Element,input),Buffer.pack(~Element,model))))),input,copied,Buffer.length(~Element,model),facts)
      +same_tree = model_tree(~Element,next,copied,Buffer.length(~Element,model),content)
      +same_length = model_length(~Element,next,copied,Buffer.length(~Element,model),content)
      +pair = copy_pair(~Element,Move.piece_inner(~Element,U32.to_nat(inner_extent),Move.directed(inner_back,inner_stride),inner_step,Move.Axis{extent,stride,back,target_stride},first,position,(Buffer.pack(~Element,input),Buffer.pack(~Element,model))),Held.pair_eta(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Move.piece_inner(~Element,U32.to_nat(inner_extent),Move.directed(inner_back,inner_stride),inner_step,Move.Axis{extent,stride,back,target_stride},first,position,(Buffer.pack(~Element,input),Buffer.pack(~Element,model)))),Buffer.remodeled(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,Move.piece_inner(~Element,U32.to_nat(inner_extent),Move.directed(inner_back,inner_stride),inner_step,Move.Axis{extent,stride,back,target_stride},first,position,(Buffer.pack(~Element,input),Buffer.pack(~Element,model))))),input,copied,Buffer.length(~Element,model),facts)
      %offsets_eta(offsets) :
        CopyResult<Element,Move.piece_rows(~Element,rest,(U32.from_nat(row) + 1 : U32),outer,Move.Axis{inner_extent,inner_stride,inner_back,inner_step},Move.Axis{extent,stride,back,target_stride},local,target,Move.piece_block(~Element,Move.Axis{inner_extent,inner_stride,inner_back,inner_step},Move.Axis{extent,stride,back,target_stride},_,(Buffer.pack(~Element,input),Buffer.pack(~Element,model)))),input,
          piece_rows_tree(~Element,1n+rest,row,outer,Move.Axis{inner_extent,inner_stride,inner_back,inner_step},extent,stride,local,target,Buffer.tree(~Element,input),Buffer.tree(~Element,model)),Buffer.length(~Element,model)>
      %Equal.sym(Storage.Buffer<Element> & Storage.Buffer<Element>,Move.piece_inner(~Element,U32.to_nat(inner_extent),Move.directed(inner_back,inner_stride),inner_step,Move.Axis{extent,stride,back,target_stride},first,position,(Buffer.pack(~Element,input),Buffer.pack(~Element,model))),(Buffer.pack(~Element,input),Buffer.pack(~Element,next)),pair) :
        CopyResult<Element,Move.piece_rows(~Element,rest,(U32.from_nat(row) + 1 : U32),outer,Move.Axis{inner_extent,inner_stride,inner_back,inner_step},Move.Axis{extent,stride,back,target_stride},local,target,_),input,
          piece_rows_tree(~Element,1n+rest,row,outer,Move.Axis{inner_extent,inner_stride,inner_back,inner_step},extent,stride,local,target,Buffer.tree(~Element,input),Buffer.tree(~Element,model)),Buffer.length(~Element,model)>
      %Equal.sym(U32,(U32.from_nat(row) + 1 : U32),U32.from_nat(1n+row),Bounds.increment_encoded(row)) :
        CopyResult<Element,Move.piece_rows(~Element,rest,_,outer,Move.Axis{inner_extent,inner_stride,inner_back,inner_step},Move.Axis{extent,stride,back,target_stride},local,target,(Buffer.pack(~Element,input),Buffer.pack(~Element,next))),input,
          piece_rows_tree(~Element,rest,1n+row,outer,Move.Axis{inner_extent,inner_stride,inner_back,inner_step},extent,stride,local,target,Buffer.tree(~Element,input),copied),Buffer.length(~Element,model)>
      %same_tree :
        CopyResult<Element,Move.piece_rows(~Element,rest,U32.from_nat(1n+row),outer,Move.Axis{inner_extent,inner_stride,inner_back,inner_step},Move.Axis{extent,stride,back,target_stride},local,target,(Buffer.pack(~Element,input),Buffer.pack(~Element,next))),input,
          piece_rows_tree(~Element,rest,1n+row,outer,Move.Axis{inner_extent,inner_stride,inner_back,inner_step},extent,stride,local,target,Buffer.tree(~Element,input),_),Buffer.length(~Element,model)>
      %same_length :
        CopyResult<Element,Move.piece_rows(~Element,rest,U32.from_nat(1n+row),outer,Move.Axis{inner_extent,inner_stride,inner_back,inner_step},Move.Axis{extent,stride,back,target_stride},local,target,(Buffer.pack(~Element,input),Buffer.pack(~Element,next))),input,
          piece_rows_tree(~Element,rest,1n+row,outer,Move.Axis{inner_extent,inner_stride,inner_back,inner_step},extent,stride,local,target,Buffer.tree(~Element,input),Buffer.tree(~Element,next)),_>
      piece_rows_refines(~Element,rest,1n+row,outer,inner_extent,inner_stride,inner_back,inner_step,extent,stride,back,target_stride,local,target,input,next)

# The runtime nest of the held piece is its model.
def piece_nested_refines(~Element: Data,+outer: List<&2,Move.Axis>,+inner_extent: U32,+inner_stride: U32,+inner_back: Bool,+inner_step: U32,+extent: U32,+stride: U32,+back: Bool,+target_stride: U32,+local: U32,+target: U32,
  +input: Buffer.Model<Element>,+model: Buffer.Model<Element>) ->
  CopyResult<Element,Move.piece_nested(~Element,Move.Nest{outer,Move.Axis{inner_extent,inner_stride,inner_back,inner_step}},Move.Axis{extent,stride,back,target_stride},local,target,(Buffer.pack(~Element,input),Buffer.pack(~Element,model))),input,
    piece_nested_tree(~Element,Move.Nest{outer,Move.Axis{inner_extent,inner_stride,inner_back,inner_step}},extent,stride,local,target,Buffer.tree(~Element,input),Buffer.tree(~Element,model)),Buffer.length(~Element,model)>:
  piece_rows_refines(~Element,U32.to_nat(Move.rows_of(outer)),0n,outer,inner_extent,inner_stride,inner_back,inner_step,extent,stride,back,target_stride,local,target,
    input,model)

# The runtime rows of any nest of the axes before the run are their model.
def nest_refines(~Element: Data,+outer: List<&2,Move.Axis>,+extent: U32,+stride: U32,+back: Bool,+target_stride: U32,+source: U32,+target: U32,+holding: Held.Holding<Element>,
  +model: Buffer.Model<Element>) ->
  Held.RunResult<Element,Move.nested_rows(~Element,Move.nest_of(outer),Move.Axis{extent,stride,back,target_stride},source,target,(Held.pack(~Element,holding),Buffer.pack(~Element,model))),
        nested_ran(~Element,Move.nest_of(outer),extent,stride,back,source,target,holding,Buffer.tree(~Element,model)),Buffer.length(~Element,model)>:
  %nest_eta(Move.nest_of(outer)) :
    Held.RunResult<Element,Move.nested_rows(~Element,_,Move.Axis{extent,stride,back,target_stride},source,target,(Held.pack(~Element,holding),Buffer.pack(~Element,model))),
        nested_ran(~Element,_,extent,stride,back,source,target,holding,Buffer.tree(~Element,model)),Buffer.length(~Element,model)>
  %axis_eta(nest_inner(Move.nest_of(outer))) :
    Held.RunResult<Element,Move.nested_rows(~Element,Move.Nest{nest_outer(Move.nest_of(outer)),_},Move.Axis{extent,stride,back,target_stride},source,target,(Held.pack(~Element,holding),Buffer.pack(~Element,model))),
        nested_ran(~Element,Move.Nest{nest_outer(Move.nest_of(outer)),_},extent,stride,back,source,target,holding,Buffer.tree(~Element,model)),Buffer.length(~Element,model)>
  nested_refines(~Element,nest_outer(Move.nest_of(outer)),axis_extent(nest_inner(Move.nest_of(outer))),axis_stride(nest_inner(Move.nest_of(outer))),axis_back(nest_inner(Move.nest_of(outer))),axis_step(nest_inner(Move.nest_of(outer))),extent,stride,back,target_stride,source,target,
    holding,model)

# The runtime piece path is its model when the current piece holds the
# part.
def piece_refines(~Element: Data,inside: Bool,+outer: List<&2,Move.Axis>,+extent: U32,+stride: U32,+back: Bool,+target_stride: U32,+source: U32,+target: U32,+behind: Held.Stack<Element>,
  +buffer: Buffer.Model<Element>,+first: U32,+span: U32,+below: Held.Stack<Element>,+model: Buffer.Model<Element>) ->
  Held.RunResult<Element,Move.piece_part(~Element,inside,Move.nest_of(outer),Move.Axis{extent,stride,back,target_stride},source,target,Held.pack_stack(~Element,behind),Buffer.pack(~Element,buffer),first,span,
          Held.pack_stack(~Element,below),Buffer.pack(~Element,model)),
        piece_ran(~Element,inside,Move.nest_of(outer),extent,stride,back,source,target,behind,buffer,first,span,below,Buffer.tree(~Element,model)),Buffer.length(~Element,model)>:
  match inside:
    case True{}:
      +facts = piece_nested_refines(~Element,nest_outer(Move.nest_of(outer)),axis_extent(nest_inner(Move.nest_of(outer))),axis_stride(nest_inner(Move.nest_of(outer))),axis_back(nest_inner(Move.nest_of(outer))),axis_step(nest_inner(Move.nest_of(outer))),extent,stride,back,target_stride,
        (source - first : U32),target,buffer,model)
      +eta = Equal.trans(Storage.Buffer<Element> & Storage.Buffer<Element>,Move.piece_nested(~Element,Move.Nest{nest_outer(Move.nest_of(outer)),Move.Axis{axis_extent(nest_inner(Move.nest_of(outer))),axis_stride(nest_inner(Move.nest_of(outer))),axis_back(nest_inner(Move.nest_of(outer))),axis_step(nest_inner(Move.nest_of(outer)))}},Move.Axis{extent,stride,back,target_stride},(source - first : U32),target,(Buffer.pack(~Element,buffer),Buffer.pack(~Element,model))),(Pair.fst(Storage.Buffer<Element>,Storage.Buffer<Element>,Move.piece_nested(~Element,Move.Nest{nest_outer(Move.nest_of(outer)),Move.Axis{axis_extent(nest_inner(Move.nest_of(outer))),axis_stride(nest_inner(Move.nest_of(outer))),axis_back(nest_inner(Move.nest_of(outer))),axis_step(nest_inner(Move.nest_of(outer)))}},Move.Axis{extent,stride,back,target_stride},(source - first : U32),target,(Buffer.pack(~Element,buffer),Buffer.pack(~Element,model)))),Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,Move.piece_nested(~Element,Move.Nest{nest_outer(Move.nest_of(outer)),Move.Axis{axis_extent(nest_inner(Move.nest_of(outer))),axis_stride(nest_inner(Move.nest_of(outer))),axis_back(nest_inner(Move.nest_of(outer))),axis_step(nest_inner(Move.nest_of(outer)))}},Move.Axis{extent,stride,back,target_stride},(source - first : U32),target,(Buffer.pack(~Element,buffer),Buffer.pack(~Element,model))))),(Buffer.pack(~Element,buffer),Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,Move.piece_nested(~Element,Move.Nest{nest_outer(Move.nest_of(outer)),Move.Axis{axis_extent(nest_inner(Move.nest_of(outer))),axis_stride(nest_inner(Move.nest_of(outer))),axis_back(nest_inner(Move.nest_of(outer))),axis_step(nest_inner(Move.nest_of(outer)))}},Move.Axis{extent,stride,back,target_stride},(source - first : U32),target,(Buffer.pack(~Element,buffer),Buffer.pack(~Element,model))))),
        Held.pair_eta(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Move.piece_nested(~Element,Move.Nest{nest_outer(Move.nest_of(outer)),Move.Axis{axis_extent(nest_inner(Move.nest_of(outer))),axis_stride(nest_inner(Move.nest_of(outer))),axis_back(nest_inner(Move.nest_of(outer))),axis_step(nest_inner(Move.nest_of(outer)))}},Move.Axis{extent,stride,back,target_stride},(source - first : U32),target,(Buffer.pack(~Element,buffer),Buffer.pack(~Element,model)))),
        Equal.cong(Storage.Buffer<Element>,Storage.Buffer<Element> & Storage.Buffer<Element>,head => (head,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,Move.piece_nested(~Element,Move.Nest{nest_outer(Move.nest_of(outer)),Move.Axis{axis_extent(nest_inner(Move.nest_of(outer))),axis_stride(nest_inner(Move.nest_of(outer))),axis_back(nest_inner(Move.nest_of(outer))),axis_step(nest_inner(Move.nest_of(outer)))}},Move.Axis{extent,stride,back,target_stride},(source - first : U32),target,(Buffer.pack(~Element,buffer),Buffer.pack(~Element,model))))),Pair.fst(Storage.Buffer<Element>,Storage.Buffer<Element>,Move.piece_nested(~Element,Move.Nest{nest_outer(Move.nest_of(outer)),Move.Axis{axis_extent(nest_inner(Move.nest_of(outer))),axis_stride(nest_inner(Move.nest_of(outer))),axis_back(nest_inner(Move.nest_of(outer))),axis_step(nest_inner(Move.nest_of(outer)))}},Move.Axis{extent,stride,back,target_stride},(source - first : U32),target,(Buffer.pack(~Element,buffer),Buffer.pack(~Element,model)))),Buffer.pack(~Element,buffer),
          copy_input(~Element,Move.piece_nested(~Element,Move.Nest{nest_outer(Move.nest_of(outer)),Move.Axis{axis_extent(nest_inner(Move.nest_of(outer))),axis_stride(nest_inner(Move.nest_of(outer))),axis_back(nest_inner(Move.nest_of(outer))),axis_step(nest_inner(Move.nest_of(outer)))}},Move.Axis{extent,stride,back,target_stride},(source - first : U32),target,(Buffer.pack(~Element,buffer),Buffer.pack(~Element,model))),buffer,piece_nested_tree(~Element,Move.Nest{nest_outer(Move.nest_of(outer)),Move.Axis{axis_extent(nest_inner(Move.nest_of(outer))),axis_stride(nest_inner(Move.nest_of(outer))),axis_back(nest_inner(Move.nest_of(outer))),axis_step(nest_inner(Move.nest_of(outer)))}},extent,stride,(source - first : U32),target,Buffer.tree(~Element,buffer),Buffer.tree(~Element,model)),Buffer.length(~Element,model),facts)))
      %nest_eta(Move.nest_of(outer)) :
        Held.RunResult<Element,Move.piece_part(~Element,True{},_,Move.Axis{extent,stride,back,target_stride},source,target,Held.pack_stack(~Element,behind),Buffer.pack(~Element,buffer),first,span,
          Held.pack_stack(~Element,below),Buffer.pack(~Element,model)),
        piece_ran(~Element,True{},_,extent,stride,back,source,target,behind,buffer,first,span,below,Buffer.tree(~Element,model)),Buffer.length(~Element,model)>
      %axis_eta(nest_inner(Move.nest_of(outer))) :
        Held.RunResult<Element,Move.piece_part(~Element,True{},Move.Nest{nest_outer(Move.nest_of(outer)),_},Move.Axis{extent,stride,back,target_stride},source,target,Held.pack_stack(~Element,behind),Buffer.pack(~Element,buffer),first,span,
          Held.pack_stack(~Element,below),Buffer.pack(~Element,model)),
        piece_ran(~Element,True{},Move.Nest{nest_outer(Move.nest_of(outer)),_},extent,stride,back,source,target,behind,buffer,first,span,below,Buffer.tree(~Element,model)),Buffer.length(~Element,model)>
      %Equal.sym(Storage.Buffer<Element> & Storage.Buffer<Element>,Move.piece_nested(~Element,Move.Nest{nest_outer(Move.nest_of(outer)),Move.Axis{axis_extent(nest_inner(Move.nest_of(outer))),axis_stride(nest_inner(Move.nest_of(outer))),axis_back(nest_inner(Move.nest_of(outer))),axis_step(nest_inner(Move.nest_of(outer)))}},Move.Axis{extent,stride,back,target_stride},(source - first : U32),target,(Buffer.pack(~Element,buffer),Buffer.pack(~Element,model))),(Buffer.pack(~Element,buffer),Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,Move.piece_nested(~Element,Move.Nest{nest_outer(Move.nest_of(outer)),Move.Axis{axis_extent(nest_inner(Move.nest_of(outer))),axis_stride(nest_inner(Move.nest_of(outer))),axis_back(nest_inner(Move.nest_of(outer))),axis_step(nest_inner(Move.nest_of(outer)))}},Move.Axis{extent,stride,back,target_stride},(source - first : U32),target,(Buffer.pack(~Element,buffer),Buffer.pack(~Element,model))))),eta) :
        Held.RunResult<Element,Move.piece_returned(~Element,Held.pack_stack(~Element,behind),first,span,Held.pack_stack(~Element,below),_),
          Held.Ran{Held.Holding{behind,Held.Held{buffer,first,span,below}},piece_nested_tree(~Element,Move.Nest{nest_outer(Move.nest_of(outer)),Move.Axis{axis_extent(nest_inner(Move.nest_of(outer))),axis_stride(nest_inner(Move.nest_of(outer))),axis_back(nest_inner(Move.nest_of(outer))),axis_step(nest_inner(Move.nest_of(outer)))}},extent,stride,(source - first : U32),target,Buffer.tree(~Element,buffer),Buffer.tree(~Element,model))},Buffer.length(~Element,model)>
      Held.RunResult{{==},copy_target(~Element,Move.piece_nested(~Element,Move.Nest{nest_outer(Move.nest_of(outer)),Move.Axis{axis_extent(nest_inner(Move.nest_of(outer))),axis_stride(nest_inner(Move.nest_of(outer))),axis_back(nest_inner(Move.nest_of(outer))),axis_step(nest_inner(Move.nest_of(outer)))}},Move.Axis{extent,stride,back,target_stride},(source - first : U32),target,(Buffer.pack(~Element,buffer),Buffer.pack(~Element,model))),buffer,piece_nested_tree(~Element,Move.Nest{nest_outer(Move.nest_of(outer)),Move.Axis{axis_extent(nest_inner(Move.nest_of(outer))),axis_stride(nest_inner(Move.nest_of(outer))),axis_back(nest_inner(Move.nest_of(outer))),axis_step(nest_inner(Move.nest_of(outer)))}},extent,stride,(source - first : U32),target,Buffer.tree(~Element,buffer),Buffer.tree(~Element,model)),Buffer.length(~Element,model),facts)}
    case False{}:
      nest_refines(~Element,outer,extent,stride,back,target_stride,source,target,Held.Holding{behind,Held.Held{buffer,first,span,below}},model)

# The runtime part from the current piece on is its model.
def current_refines(~Element: Data,+outer: List<&2,Move.Axis>,+extent: U32,+stride: U32,+back: Bool,+target_stride: U32,+source: U32,+target: U32,holding: Held.Holding<Element>,
  +model: Buffer.Model<Element>) ->
  Held.RunResult<Element,Move.current_part(~Element,outer,Move.Axis{extent,stride,back,target_stride},source,target,Held.pack(~Element,holding),Buffer.pack(~Element,model)),
    current_ran(~Element,outer,Move.Axis{extent,stride,back,target_stride},source,target,holding,Buffer.tree(~Element,model)),Buffer.length(~Element,model)>:
  match holding:
    case Held.Holding{+behind,Held.Held{+buffer,+first,+span,+below}}:
      piece_refines(~Element,Move.fits_from(outer,Move.Axis{extent,stride,back,target_stride},source,first,span),outer,extent,stride,back,target_stride,source,target,behind,buffer,first,span,below,model)
    case Held.Holding{+behind,Held.Bottom{}}:
      nest_refines(~Element,outer,extent,stride,back,target_stride,source,target,Held.Holding{behind,Held.Bottom{}},model)

# A part holds and writes what its model runs.
def part_refines(~Element: Data,+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+holding: Held.Holding<Element>,+model: Buffer.Model<Element>) ->
  Held.RunResult<Element,Move.part(~Element,Move.Part{axes,source,target},Held.pack(~Element,holding),Buffer.pack(~Element,model)),
    part_ran(~Element,axes,source,target,holding,Buffer.tree(~Element,model)),Buffer.length(~Element,model)>:
  +run = part_run(axes)
  %split_eta(Move.split_last(Move.coalesced(axes))) :
    Held.RunResult<Element,Move.split_part(~Element,_,source,target,Held.pack(~Element,holding),Buffer.pack(~Element,model)),
      part_ran(~Element,axes,source,target,holding,Buffer.tree(~Element,model)),Buffer.length(~Element,model)>
  %axis_eta(run) :
    Held.RunResult<Element,Move.current_part(~Element,part_outer(axes),_,source,target,Move.resolve(~Element,Held.pack(~Element,holding),source),Buffer.pack(~Element,model)),
      current_ran(~Element,part_outer(axes),_,source,target,Held.resolve(~Element,holding,source),Buffer.tree(~Element,model)),Buffer.length(~Element,model)>
  %Equal.sym(Move.Holding<Element>,Move.resolve(~Element,Held.pack(~Element,holding),source),Held.pack(~Element,Held.resolve(~Element,holding,source)),
      Held.resolve_packed(~Element,holding,source)) :
    Held.RunResult<Element,Move.current_part(~Element,part_outer(axes),Move.Axis{axis_extent(run),axis_stride(run),axis_back(run),axis_step(run)},source,target,_,Buffer.pack(~Element,model)),
      current_ran(~Element,part_outer(axes),Move.Axis{axis_extent(run),axis_stride(run),axis_back(run),axis_step(run)},source,target,Held.resolve(~Element,holding,source),Buffer.tree(~Element,model)),Buffer.length(~Element,model)>
  current_refines(~Element,part_outer(axes),axis_extent(run),axis_stride(run),axis_back(run),axis_step(run),source,target,Held.resolve(~Element,holding,source),model)

# ---- What the runtime model of a part keeps ----

# Runs over the rows keep the order of the held pieces.
def inner_order(~Element: Data,fuel: Nat,+step: U32,+inner_target: U32,+extent: U32,+stride: U32,+back: Bool,+source: U32,+position: U32,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>) ->
  {Held.order(~Element,Held.ran_holding(~Element,inner_ran(~Element,fuel,step,inner_target,extent,stride,back,source,position,holding,tree))) == Held.order(~Element,holding) : Held.Stack<Element>}:
  match fuel:
    case 0n: {==}
    case 1n+ +rest:
      +ran = Held.model_run(~Element,U32.to_nat(extent),U32.is_zero(extent),extent,source,stride,back,position,holding,tree)
      Equal.trans(Held.Stack<Element>,Held.order(~Element,Held.ran_holding(~Element,inner_ran(~Element,rest,step,inner_target,extent,stride,back,(source + step : U32),(position + inner_target : U32),Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran)))),Held.order(~Element,Held.ran_holding(~Element,ran)),Held.order(~Element,holding),
        inner_order(~Element,rest,step,inner_target,extent,stride,back,(source + step : U32),(position + inner_target : U32),Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran)),
        Held.run_order(~Element,U32.to_nat(extent),U32.is_zero(extent),extent,source,stride,back,position,holding,tree))

def blocks_order(~Element: Data,fuel: Nat,+row: Nat,+outer: List<&2,Move.Axis>,+inner: Move.Axis,+extent: U32,+stride: U32,+back: Bool,+source: U32,+target: U32,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>) ->
  {Held.order(~Element,Held.ran_holding(~Element,blocks_ran(~Element,fuel,row,outer,inner,extent,stride,back,source,target,holding,tree))) == Held.order(~Element,holding) : Held.Stack<Element>}:
  match fuel:
    case 0n: {==}
    case 1n+ +rest:
      +ran = inner_ran(~Element,U32.to_nat(axis_extent(inner)),Move.directed(axis_back(inner),axis_stride(inner)),axis_step(inner),extent,stride,back,offsets_source(Move.row_offsets(outer,U32.from_nat(row),source,target)),offsets_target(Move.row_offsets(outer,U32.from_nat(row),source,target)),holding,tree)
      Equal.trans(Held.Stack<Element>,Held.order(~Element,Held.ran_holding(~Element,blocks_ran(~Element,rest,1n+row,outer,inner,extent,stride,back,source,target,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran)))),Held.order(~Element,Held.ran_holding(~Element,ran)),Held.order(~Element,holding),
        blocks_order(~Element,rest,1n+row,outer,inner,extent,stride,back,source,target,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran)),
        inner_order(~Element,U32.to_nat(axis_extent(inner)),Move.directed(axis_back(inner),axis_stride(inner)),axis_step(inner),extent,stride,back,offsets_source(Move.row_offsets(outer,U32.from_nat(row),source,target)),offsets_target(Move.row_offsets(outer,U32.from_nat(row),source,target)),holding,tree))

def nested_order(~Element: Data,+nest: Move.Nest,+extent: U32,+stride: U32,+back: Bool,+source: U32,+target: U32,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>) ->
  {Held.order(~Element,Held.ran_holding(~Element,nested_ran(~Element,nest,extent,stride,back,source,target,holding,tree))) == Held.order(~Element,holding) : Held.Stack<Element>}:
  blocks_order(~Element,U32.to_nat(Move.rows_of(nest_outer(nest))),0n,nest_outer(nest),nest_inner(nest),extent,stride,back,source,target,holding,tree)

def piece_order(~Element: Data,inside: Bool,+nest: Move.Nest,+extent: U32,+stride: U32,+back: Bool,+source: U32,+target: U32,+behind: Held.Stack<Element>,+buffer: Buffer.Model<Element>,+first: U32,+span: U32,+below: Held.Stack<Element>,+tree: Tree.Tree<Element>) ->
  {Held.order(~Element,Held.ran_holding(~Element,piece_ran(~Element,inside,nest,extent,stride,back,source,target,behind,buffer,first,span,below,tree))) == Held.order(~Element,Held.Holding{behind,Held.Held{buffer,first,span,below}}) : Held.Stack<Element>}:
  match inside:
    case True{}: {==}
    case False{}: nested_order(~Element,nest,extent,stride,back,source,target,Held.Holding{behind,Held.Held{buffer,first,span,below}},tree)

def current_order(~Element: Data,+outer: List<&2,Move.Axis>,+run: Move.Axis,+source: U32,+target: U32,holding: Held.Holding<Element>,+tree: Tree.Tree<Element>) ->
  {Held.order(~Element,Held.ran_holding(~Element,current_ran(~Element,outer,run,source,target,holding,tree))) == Held.order(~Element,holding) : Held.Stack<Element>}:
  match holding:
    case Held.Holding{+behind,Held.Held{+buffer,+first,+span,+below}}:
      piece_order(~Element,Move.fits_from(outer,run,source,first,span),Move.nest_of(outer),axis_extent(run),axis_stride(run),axis_back(run),source,target,behind,buffer,first,span,below,tree)
    case Held.Holding{+behind,Held.Bottom{}}:
      nested_order(~Element,Move.nest_of(outer),axis_extent(run),axis_stride(run),axis_back(run),source,target,Held.Holding{behind,Held.Bottom{}},tree)

def part_order(~Element: Data,+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>) ->
  {Held.order(~Element,Held.ran_holding(~Element,part_ran(~Element,axes,source,target,holding,tree))) == Held.order(~Element,holding) : Held.Stack<Element>}:
  Equal.trans(Held.Stack<Element>,Held.order(~Element,Held.ran_holding(~Element,part_ran(~Element,axes,source,target,holding,tree))),Held.order(~Element,Held.resolve(~Element,holding,source)),Held.order(~Element,holding),
    current_order(~Element,part_outer(axes),part_run(axes),source,target,Held.resolve(~Element,holding,source),tree),Held.resolve_order(~Element,holding,source))

# Runs over the rows keep the values of the held pieces.
def inner_values(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,fuel: Nat,+step: U32,+inner_target: U32,+extent: U32,+stride: U32,+back: Bool,+source: U32,+position: U32,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>,
  +values: Held.HoldingValues(~Element,~Context,~value,context,holding)) -> Held.HoldingValues(~Element,~Context,~value,context,Held.ran_holding(~Element,inner_ran(~Element,fuel,step,inner_target,extent,stride,back,source,position,holding,tree))):
  match fuel:
    case 0n: values
    case 1n+ +rest:
      +ran = Held.model_run(~Element,U32.to_nat(extent),U32.is_zero(extent),extent,source,stride,back,position,holding,tree)
      inner_values(~Element,~Context,~value,context,rest,step,inner_target,extent,stride,back,(source + step : U32),(position + inner_target : U32),Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran),
        Held.run_values(~Element,~Context,~value,context,U32.to_nat(extent),U32.is_zero(extent),extent,source,stride,back,position,holding,tree,values))

def blocks_values(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,fuel: Nat,+row: Nat,+outer: List<&2,Move.Axis>,+inner: Move.Axis,+extent: U32,+stride: U32,+back: Bool,+source: U32,+target: U32,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>,
  +values: Held.HoldingValues(~Element,~Context,~value,context,holding)) -> Held.HoldingValues(~Element,~Context,~value,context,Held.ran_holding(~Element,blocks_ran(~Element,fuel,row,outer,inner,extent,stride,back,source,target,holding,tree))):
  match fuel:
    case 0n: values
    case 1n+ +rest:
      +ran = inner_ran(~Element,U32.to_nat(axis_extent(inner)),Move.directed(axis_back(inner),axis_stride(inner)),axis_step(inner),extent,stride,back,offsets_source(Move.row_offsets(outer,U32.from_nat(row),source,target)),offsets_target(Move.row_offsets(outer,U32.from_nat(row),source,target)),holding,tree)
      blocks_values(~Element,~Context,~value,context,rest,1n+row,outer,inner,extent,stride,back,source,target,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran),
        inner_values(~Element,~Context,~value,context,U32.to_nat(axis_extent(inner)),Move.directed(axis_back(inner),axis_stride(inner)),axis_step(inner),extent,stride,back,offsets_source(Move.row_offsets(outer,U32.from_nat(row),source,target)),offsets_target(Move.row_offsets(outer,U32.from_nat(row),source,target)),holding,tree,values))

def nested_values(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+nest: Move.Nest,+extent: U32,+stride: U32,+back: Bool,+source: U32,+target: U32,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>,+values: Held.HoldingValues(~Element,~Context,~value,context,holding)) -> Held.HoldingValues(~Element,~Context,~value,context,Held.ran_holding(~Element,nested_ran(~Element,nest,extent,stride,back,source,target,holding,tree))):
  blocks_values(~Element,~Context,~value,context,U32.to_nat(Move.rows_of(nest_outer(nest))),0n,nest_outer(nest),nest_inner(nest),extent,stride,back,source,target,holding,tree,values)

def piece_values(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,inside: Bool,+nest: Move.Nest,+extent: U32,+stride: U32,+back: Bool,+source: U32,+target: U32,+behind: Held.Stack<Element>,+buffer: Buffer.Model<Element>,+first: U32,+span: U32,+below: Held.Stack<Element>,+tree: Tree.Tree<Element>,+values: Held.HoldingValues(~Element,~Context,~value,context,Held.Holding{behind,Held.Held{buffer,first,span,below}})) -> Held.HoldingValues(~Element,~Context,~value,context,Held.ran_holding(~Element,piece_ran(~Element,inside,nest,extent,stride,back,source,target,behind,buffer,first,span,below,tree))):
  match inside:
    case True{}: values
    case False{}: nested_values(~Element,~Context,~value,context,nest,extent,stride,back,source,target,Held.Holding{behind,Held.Held{buffer,first,span,below}},tree,values)

def current_values(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+outer: List<&2,Move.Axis>,+run: Move.Axis,+source: U32,+target: U32,holding: Held.Holding<Element>,+tree: Tree.Tree<Element>,+values: Held.HoldingValues(~Element,~Context,~value,context,holding)) -> Held.HoldingValues(~Element,~Context,~value,context,Held.ran_holding(~Element,current_ran(~Element,outer,run,source,target,holding,tree))):
  match holding:
    case Held.Holding{+behind,Held.Held{+buffer,+first,+span,+below}}:
      piece_values(~Element,~Context,~value,context,Move.fits_from(outer,run,source,first,span),Move.nest_of(outer),axis_extent(run),axis_stride(run),axis_back(run),source,target,behind,buffer,first,span,below,tree,values)
    case Held.Holding{+behind,Held.Bottom{}}:
      nested_values(~Element,~Context,~value,context,Move.nest_of(outer),axis_extent(run),axis_stride(run),axis_back(run),source,target,Held.Holding{behind,Held.Bottom{}},tree,values)

def part_values(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>,+values: Held.HoldingValues(~Element,~Context,~value,context,holding)) -> Held.HoldingValues(~Element,~Context,~value,context,Held.ran_holding(~Element,part_ran(~Element,axes,source,target,holding,tree))):
  current_values(~Element,~Context,~value,context,part_outer(axes),part_run(axes),source,target,Held.resolve(~Element,holding,source),tree,Held.resolve_values(~Element,~Context,~value,context,holding,source,values))

# The rows keep a balanced tree balanced.
def inner_balanced(~Element: Data,fuel: Nat,+step: U32,+inner_target: U32,+extent: U32,+stride: U32,+back: Bool,+source: U32,+position: U32,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>,+depth: Nat,
  +balanced: Shape.Balanced(Element,depth,tree)) -> Shape.Balanced(Element,depth,Held.ran_tree(~Element,inner_ran(~Element,fuel,step,inner_target,extent,stride,back,source,position,holding,tree))):
  match fuel:
    case 0n: balanced
    case 1n+ +rest:
      +ran = Held.model_run(~Element,U32.to_nat(extent),U32.is_zero(extent),extent,source,stride,back,position,holding,tree)
      inner_balanced(~Element,rest,step,inner_target,extent,stride,back,(source + step : U32),(position + inner_target : U32),Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran),depth,
        Held.run_balanced(~Element,U32.to_nat(extent),U32.is_zero(extent),extent,source,stride,back,position,holding,tree,depth,balanced))

def blocks_balanced(~Element: Data,fuel: Nat,+row: Nat,+outer: List<&2,Move.Axis>,+inner: Move.Axis,+extent: U32,+stride: U32,+back: Bool,+source: U32,+target: U32,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>,+depth: Nat,
  +balanced: Shape.Balanced(Element,depth,tree)) -> Shape.Balanced(Element,depth,Held.ran_tree(~Element,blocks_ran(~Element,fuel,row,outer,inner,extent,stride,back,source,target,holding,tree))):
  match fuel:
    case 0n: balanced
    case 1n+ +rest:
      +ran = inner_ran(~Element,U32.to_nat(axis_extent(inner)),Move.directed(axis_back(inner),axis_stride(inner)),axis_step(inner),extent,stride,back,offsets_source(Move.row_offsets(outer,U32.from_nat(row),source,target)),offsets_target(Move.row_offsets(outer,U32.from_nat(row),source,target)),holding,tree)
      blocks_balanced(~Element,rest,1n+row,outer,inner,extent,stride,back,source,target,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran),depth,
        inner_balanced(~Element,U32.to_nat(axis_extent(inner)),Move.directed(axis_back(inner),axis_stride(inner)),axis_step(inner),extent,stride,back,offsets_source(Move.row_offsets(outer,U32.from_nat(row),source,target)),offsets_target(Move.row_offsets(outer,U32.from_nat(row),source,target)),holding,tree,depth,balanced))

def nested_balanced(~Element: Data,+nest: Move.Nest,+extent: U32,+stride: U32,+back: Bool,+source: U32,+target: U32,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>,+depth: Nat,+balanced: Shape.Balanced(Element,depth,tree)) -> Shape.Balanced(Element,depth,Held.ran_tree(~Element,nested_ran(~Element,nest,extent,stride,back,source,target,holding,tree))):
  blocks_balanced(~Element,U32.to_nat(Move.rows_of(nest_outer(nest))),0n,nest_outer(nest),nest_inner(nest),extent,stride,back,source,target,holding,tree,depth,balanced)

def piece_inner_balanced(~Element: Data,fuel: Nat,+step: U32,+inner_target: U32,+extent: U32,+stride: U32,+local: U32,+position: U32,+input: Tree.Tree<Element>,+tree: Tree.Tree<Element>,+depth: Nat,+balanced: Shape.Balanced(Element,depth,tree)) ->
  Shape.Balanced(Element,depth,piece_inner_tree(~Element,fuel,step,inner_target,extent,stride,local,position,input,tree)):
  match fuel:
    case 0n: balanced
    case 1n+ +rest:
      piece_inner_balanced(~Element,rest,step,inner_target,extent,stride,(local + step : U32),(position + inner_target : U32),input,run_copy(~Element,extent,local,stride,position,input,tree),depth,
        Held.pure_balanced(~Element,CopyValues.copy_writes(~Element,U32.is_eq(stride,1) && Bool.not(False{}),U32.to_nat(extent),Views.View{[Views.Axis{extent,Move.directed(False{},stride)}],local},position,input),depth,tree,balanced))

def piece_rows_balanced(~Element: Data,fuel: Nat,+row: Nat,+outer: List<&2,Move.Axis>,+inner: Move.Axis,+extent: U32,+stride: U32,+local: U32,+target: U32,+input: Tree.Tree<Element>,+tree: Tree.Tree<Element>,+depth: Nat,+balanced: Shape.Balanced(Element,depth,tree)) ->
  Shape.Balanced(Element,depth,piece_rows_tree(~Element,fuel,row,outer,inner,extent,stride,local,target,input,tree)):
  match fuel:
    case 0n: balanced
    case 1n+ +rest:
      +copied = piece_inner_tree(~Element,U32.to_nat(axis_extent(inner)),Move.directed(axis_back(inner),axis_stride(inner)),axis_step(inner),extent,stride,offsets_source(Move.row_offsets(outer,U32.from_nat(row),local,target)),offsets_target(Move.row_offsets(outer,U32.from_nat(row),local,target)),input,tree)
      piece_rows_balanced(~Element,rest,1n+row,outer,inner,extent,stride,local,target,input,copied,depth,piece_inner_balanced(~Element,U32.to_nat(axis_extent(inner)),Move.directed(axis_back(inner),axis_stride(inner)),axis_step(inner),extent,stride,offsets_source(Move.row_offsets(outer,U32.from_nat(row),local,target)),offsets_target(Move.row_offsets(outer,U32.from_nat(row),local,target)),input,tree,depth,balanced))

def piece_nested_balanced(~Element: Data,+nest: Move.Nest,+extent: U32,+stride: U32,+local: U32,+target: U32,+input: Tree.Tree<Element>,+tree: Tree.Tree<Element>,+depth: Nat,+balanced: Shape.Balanced(Element,depth,tree)) -> Shape.Balanced(Element,depth,piece_nested_tree(~Element,nest,extent,stride,local,target,input,tree)):
  piece_rows_balanced(~Element,U32.to_nat(Move.rows_of(nest_outer(nest))),0n,nest_outer(nest),nest_inner(nest),extent,stride,local,target,input,tree,depth,balanced)

def piece_balanced(~Element: Data,inside: Bool,+nest: Move.Nest,+extent: U32,+stride: U32,+back: Bool,+source: U32,+target: U32,+behind: Held.Stack<Element>,+buffer: Buffer.Model<Element>,+first: U32,+span: U32,+below: Held.Stack<Element>,+tree: Tree.Tree<Element>,+depth: Nat,+balanced: Shape.Balanced(Element,depth,tree)) -> Shape.Balanced(Element,depth,Held.ran_tree(~Element,piece_ran(~Element,inside,nest,extent,stride,back,source,target,behind,buffer,first,span,below,tree))):
  match inside:
    case True{}:
      piece_nested_balanced(~Element,nest,extent,stride,(source - first : U32),target,Buffer.tree(~Element,buffer),tree,depth,balanced)
    case False{}: nested_balanced(~Element,nest,extent,stride,back,source,target,Held.Holding{behind,Held.Held{buffer,first,span,below}},tree,depth,balanced)

def current_balanced(~Element: Data,+outer: List<&2,Move.Axis>,+run: Move.Axis,+source: U32,+target: U32,holding: Held.Holding<Element>,+tree: Tree.Tree<Element>,+depth: Nat,+balanced: Shape.Balanced(Element,depth,tree)) -> Shape.Balanced(Element,depth,Held.ran_tree(~Element,current_ran(~Element,outer,run,source,target,holding,tree))):
  match holding:
    case Held.Holding{+behind,Held.Held{+buffer,+first,+span,+below}}:
      piece_balanced(~Element,Move.fits_from(outer,run,source,first,span),Move.nest_of(outer),axis_extent(run),axis_stride(run),axis_back(run),source,target,behind,buffer,first,span,below,tree,depth,balanced)
    case Held.Holding{+behind,Held.Bottom{}}:
      nested_balanced(~Element,Move.nest_of(outer),axis_extent(run),axis_stride(run),axis_back(run),source,target,Held.Holding{behind,Held.Bottom{}},tree,depth,balanced)

def part_balanced(~Element: Data,+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>,+depth: Nat,+balanced: Shape.Balanced(Element,depth,tree)) -> Shape.Balanced(Element,depth,Held.ran_tree(~Element,part_ran(~Element,axes,source,target,holding,tree))):
  current_balanced(~Element,part_outer(axes),part_run(axes),source,target,Held.resolve(~Element,holding,source),tree,depth,balanced)

# The rows keep the capacity of the tree.
def inner_capacity(~Element: Data,fuel: Nat,+step: U32,+inner_target: U32,+extent: U32,+stride: U32,+back: Bool,+source: U32,+position: U32,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>) ->
  {Tree.capacity(Element,Held.ran_tree(~Element,inner_ran(~Element,fuel,step,inner_target,extent,stride,back,source,position,holding,tree))) == Tree.capacity(Element,tree) : U32}:
  match fuel:
    case 0n: {==}
    case 1n+ +rest:
      +ran = Held.model_run(~Element,U32.to_nat(extent),U32.is_zero(extent),extent,source,stride,back,position,holding,tree)
      Equal.trans(U32,Tree.capacity(Element,Held.ran_tree(~Element,inner_ran(~Element,rest,step,inner_target,extent,stride,back,(source + step : U32),(position + inner_target : U32),Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran)))),Tree.capacity(Element,Held.ran_tree(~Element,ran)),Tree.capacity(Element,tree),
        inner_capacity(~Element,rest,step,inner_target,extent,stride,back,(source + step : U32),(position + inner_target : U32),Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran)),
        Held.run_capacity(~Element,U32.to_nat(extent),U32.is_zero(extent),extent,source,stride,back,position,holding,tree))

def blocks_capacity(~Element: Data,fuel: Nat,+row: Nat,+outer: List<&2,Move.Axis>,+inner: Move.Axis,+extent: U32,+stride: U32,+back: Bool,+source: U32,+target: U32,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>) ->
  {Tree.capacity(Element,Held.ran_tree(~Element,blocks_ran(~Element,fuel,row,outer,inner,extent,stride,back,source,target,holding,tree))) == Tree.capacity(Element,tree) : U32}:
  match fuel:
    case 0n: {==}
    case 1n+ +rest:
      +ran = inner_ran(~Element,U32.to_nat(axis_extent(inner)),Move.directed(axis_back(inner),axis_stride(inner)),axis_step(inner),extent,stride,back,offsets_source(Move.row_offsets(outer,U32.from_nat(row),source,target)),offsets_target(Move.row_offsets(outer,U32.from_nat(row),source,target)),holding,tree)
      Equal.trans(U32,Tree.capacity(Element,Held.ran_tree(~Element,blocks_ran(~Element,rest,1n+row,outer,inner,extent,stride,back,source,target,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran)))),Tree.capacity(Element,Held.ran_tree(~Element,ran)),Tree.capacity(Element,tree),
        blocks_capacity(~Element,rest,1n+row,outer,inner,extent,stride,back,source,target,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran)),
        inner_capacity(~Element,U32.to_nat(axis_extent(inner)),Move.directed(axis_back(inner),axis_stride(inner)),axis_step(inner),extent,stride,back,offsets_source(Move.row_offsets(outer,U32.from_nat(row),source,target)),offsets_target(Move.row_offsets(outer,U32.from_nat(row),source,target)),holding,tree))

def nested_capacity(~Element: Data,+nest: Move.Nest,+extent: U32,+stride: U32,+back: Bool,+source: U32,+target: U32,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>) ->
  {Tree.capacity(Element,Held.ran_tree(~Element,nested_ran(~Element,nest,extent,stride,back,source,target,holding,tree))) == Tree.capacity(Element,tree) : U32}:
  blocks_capacity(~Element,U32.to_nat(Move.rows_of(nest_outer(nest))),0n,nest_outer(nest),nest_inner(nest),extent,stride,back,source,target,holding,tree)

def piece_inner_capacity(~Element: Data,fuel: Nat,+step: U32,+inner_target: U32,+extent: U32,+stride: U32,+local: U32,+position: U32,+input: Tree.Tree<Element>,+tree: Tree.Tree<Element>) ->
  {Tree.capacity(Element,piece_inner_tree(~Element,fuel,step,inner_target,extent,stride,local,position,input,tree)) == Tree.capacity(Element,tree) : U32}:
  match fuel:
    case 0n: {==}
    case 1n+ +rest:
      +copied = run_copy(~Element,extent,local,stride,position,input,tree)
      Equal.trans(U32,Tree.capacity(Element,piece_inner_tree(~Element,rest,step,inner_target,extent,stride,(local + step : U32),(position + inner_target : U32),input,copied)),Tree.capacity(Element,copied),Tree.capacity(Element,tree),
        piece_inner_capacity(~Element,rest,step,inner_target,extent,stride,(local + step : U32),(position + inner_target : U32),input,copied),Held.pure_capacity(~Element,CopyValues.copy_writes(~Element,U32.is_eq(stride,1) && Bool.not(False{}),U32.to_nat(extent),Views.View{[Views.Axis{extent,Move.directed(False{},stride)}],local},position,input),tree))

def piece_rows_capacity(~Element: Data,fuel: Nat,+row: Nat,+outer: List<&2,Move.Axis>,+inner: Move.Axis,+extent: U32,+stride: U32,+local: U32,+target: U32,+input: Tree.Tree<Element>,+tree: Tree.Tree<Element>) ->
  {Tree.capacity(Element,piece_rows_tree(~Element,fuel,row,outer,inner,extent,stride,local,target,input,tree)) == Tree.capacity(Element,tree) : U32}:
  match fuel:
    case 0n: {==}
    case 1n+ +rest:
      +copied = piece_inner_tree(~Element,U32.to_nat(axis_extent(inner)),Move.directed(axis_back(inner),axis_stride(inner)),axis_step(inner),extent,stride,offsets_source(Move.row_offsets(outer,U32.from_nat(row),local,target)),offsets_target(Move.row_offsets(outer,U32.from_nat(row),local,target)),input,tree)
      Equal.trans(U32,Tree.capacity(Element,piece_rows_tree(~Element,rest,1n+row,outer,inner,extent,stride,local,target,input,copied)),Tree.capacity(Element,copied),Tree.capacity(Element,tree),
        piece_rows_capacity(~Element,rest,1n+row,outer,inner,extent,stride,local,target,input,copied),piece_inner_capacity(~Element,U32.to_nat(axis_extent(inner)),Move.directed(axis_back(inner),axis_stride(inner)),axis_step(inner),extent,stride,offsets_source(Move.row_offsets(outer,U32.from_nat(row),local,target)),offsets_target(Move.row_offsets(outer,U32.from_nat(row),local,target)),input,tree))

def piece_nested_capacity(~Element: Data,+nest: Move.Nest,+extent: U32,+stride: U32,+local: U32,+target: U32,+input: Tree.Tree<Element>,+tree: Tree.Tree<Element>) -> {Tree.capacity(Element,piece_nested_tree(~Element,nest,extent,stride,local,target,input,tree)) == Tree.capacity(Element,tree) : U32}:
  piece_rows_capacity(~Element,U32.to_nat(Move.rows_of(nest_outer(nest))),0n,nest_outer(nest),nest_inner(nest),extent,stride,local,target,input,tree)

def piece_capacity(~Element: Data,inside: Bool,+nest: Move.Nest,+extent: U32,+stride: U32,+back: Bool,+source: U32,+target: U32,+behind: Held.Stack<Element>,+buffer: Buffer.Model<Element>,+first: U32,+span: U32,+below: Held.Stack<Element>,+tree: Tree.Tree<Element>) -> {Tree.capacity(Element,Held.ran_tree(~Element,piece_ran(~Element,inside,nest,extent,stride,back,source,target,behind,buffer,first,span,below,tree))) == Tree.capacity(Element,tree) : U32}:
  match inside:
    case True{}: piece_nested_capacity(~Element,nest,extent,stride,(source - first : U32),target,Buffer.tree(~Element,buffer),tree)
    case False{}: nested_capacity(~Element,nest,extent,stride,back,source,target,Held.Holding{behind,Held.Held{buffer,first,span,below}},tree)

def current_capacity(~Element: Data,+outer: List<&2,Move.Axis>,+run: Move.Axis,+source: U32,+target: U32,holding: Held.Holding<Element>,+tree: Tree.Tree<Element>) -> {Tree.capacity(Element,Held.ran_tree(~Element,current_ran(~Element,outer,run,source,target,holding,tree))) == Tree.capacity(Element,tree) : U32}:
  match holding:
    case Held.Holding{+behind,Held.Held{+buffer,+first,+span,+below}}:
      piece_capacity(~Element,Move.fits_from(outer,run,source,first,span),Move.nest_of(outer),axis_extent(run),axis_stride(run),axis_back(run),source,target,behind,buffer,first,span,below,tree)
    case Held.Holding{+behind,Held.Bottom{}}:
      nested_capacity(~Element,Move.nest_of(outer),axis_extent(run),axis_stride(run),axis_back(run),source,target,Held.Holding{behind,Held.Bottom{}},tree)

def part_capacity(~Element: Data,+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>) -> {Tree.capacity(Element,Held.ran_tree(~Element,part_ran(~Element,axes,source,target,holding,tree))) == Tree.capacity(Element,tree) : U32}:
  current_capacity(~Element,part_outer(axes),part_run(axes),source,target,Held.resolve(~Element,holding,source),tree)

# ---- The nested rows are the rows of a part ----

# Rows a then b from row are rows a from row, then rows b after them.
def rows_append(~Element: Data,a: Nat,+b: Nat,+row: Nat,+outer: List<&2,Move.Axis>,+extent: U32,+stride: U32,+back: Bool,+source: U32,+target: U32,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>) ->
  {model_rows(~Element,Nat.add(a,b),row,outer,extent,stride,back,source,target,holding,tree) ==
    model_rows(~Element,b,Nat.add(a,row),outer,extent,stride,back,source,target,
      Held.ran_holding(~Element,model_rows(~Element,a,row,outer,extent,stride,back,source,target,holding,tree)),
      Held.ran_tree(~Element,model_rows(~Element,a,row,outer,extent,stride,back,source,target,holding,tree))) : Held.Ran<Element>}:
  match a:
    case 0n: {==}
    case 1n+ +p:
      +ran = Held.model_run(~Element,U32.to_nat(extent),U32.is_zero(extent),extent,row_source(outer,row,source,target),stride,back,row_target(outer,row,source,target),
        holding,tree)
      +later = model_rows(~Element,p,1n+row,outer,extent,stride,back,source,target,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran))
      %Addition.successor_right(p,row) :
        {model_rows(~Element,Nat.add(p,b),1n+row,outer,extent,stride,back,source,target,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran)) ==
          model_rows(~Element,b,_,outer,extent,stride,back,source,target,Held.ran_holding(~Element,later),Held.ran_tree(~Element,later)) : Held.Ran<Element>}
      rows_append(~Element,p,b,1n+row,outer,extent,stride,back,source,target,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran))

# The rows of one block from cell k on are the rows of the box from block q
# times the inner extent plus k.
def inner_flat(~Element: Data,count: Nat,+cell: Nat,+q: Nat,+outer: List<&2,Move.Axis>,+extent: U32,+stride: U32,+back: Bool,+source: U32,+target: U32,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(total(outer)),U32.to_nat(cap)) == True{} : Bool},+q_in: {Nat.is_lt(q,total(nest_outer_of(outer))) == True{} : Bool},+within: {Nat.is_le(Nat.add(cell,count),U32.to_nat(axis_extent(nest_inner_of(outer)))) == True{} : Bool}) ->
  {inner_ran(~Element,count,Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))),axis_step(nest_inner_of(outer)),extent,stride,back,(row_source(nest_outer_of(outer),q,source,target) + U32.from_nat(cell) * Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))) : U32),(row_target(nest_outer_of(outer),q,source,target) + U32.from_nat(cell) * axis_step(nest_inner_of(outer)) : U32),holding,tree) ==
    model_rows(~Element,count,Nat.add(Nat.mul(q,U32.to_nat(axis_extent(nest_inner_of(outer)))),cell),outer,extent,stride,back,source,target,holding,tree) : Held.Ran<Element>}:
  match count:
    case 0n: {==}
    case 1n+ +p:
      +width = U32.to_nat(axis_extent(nest_inner_of(outer)))
      +index = Nat.add(Nat.mul(q,width),cell)
      +cell_in = Order.room_index(cell,p,width,within)
      +product = nest_total(outer)
      +below = RowMajor.radix(q,total(nest_outer_of(outer)),cell,width,q_in,cell_in)
      +inside = {%Equal.sym(Nat,total(outer),Nat.mul(total(nest_outer_of(outer)),width),product) : {Nat.is_lt(index,_) == True{} : Bool} below :
        {Nat.is_lt(index,total(outer)) == True{} : Bool}}
      +index_room = Order.transitive(index,total(outer),U32.to_nat(cap),Bounds.strict_le(index,total(outer),inside),single_room(total(outer),cap,room))
      +exact = Bounds.roundtrip(index,cap,cap_ok,index_room)
      +first = (row_source(nest_outer_of(outer),q,source,target) + U32.from_nat(cell) * Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))) : U32)
      +position = (row_target(nest_outer_of(outer),q,source,target) + U32.from_nat(cell) * axis_step(nest_inner_of(outer)) : U32)
      +offsets = nest_offsets(outer,cap,cap_ok,room,U32.from_nat(index),q,cell,exact,q_in,cell_in,source,target)
      +ran = Held.model_run(~Element,U32.to_nat(extent),U32.is_zero(extent),extent,first,stride,back,position,holding,tree)
      %Equal.sym(Move.Offsets,Move.row_offsets(outer,U32.from_nat(index),source,target),Move.Offsets{first,position},offsets) :
        {inner_ran(~Element,p,Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))),axis_step(nest_inner_of(outer)),extent,stride,back,(first + Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))) : U32),(position + axis_step(nest_inner_of(outer)) : U32),Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran)) ==
          model_rows(~Element,p,1n+index,outer,extent,stride,back,source,target,
            Held.ran_holding(~Element,Held.model_run(~Element,U32.to_nat(extent),U32.is_zero(extent),extent,offsets_source(_),stride,back,offsets_target(_),holding,tree)),
            Held.ran_tree(~Element,Held.model_run(~Element,U32.to_nat(extent),U32.is_zero(extent),extent,offsets_source(_),stride,back,offsets_target(_),holding,tree))) : Held.Ran<Element>}
      %Equal.sym(U32,(first + Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))) : U32),(row_source(nest_outer_of(outer),q,source,target) + U32.from_nat(1n+cell) * Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))) : U32),next_offset(row_source(nest_outer_of(outer),q,source,target),cell,Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))))) :
        {inner_ran(~Element,p,Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))),axis_step(nest_inner_of(outer)),extent,stride,back,_,(position + axis_step(nest_inner_of(outer)) : U32),Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran)) ==
          model_rows(~Element,p,1n+index,outer,extent,stride,back,source,target,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran)) : Held.Ran<Element>}
      %Equal.sym(U32,(position + axis_step(nest_inner_of(outer)) : U32),(row_target(nest_outer_of(outer),q,source,target) + U32.from_nat(1n+cell) * axis_step(nest_inner_of(outer)) : U32),next_offset(row_target(nest_outer_of(outer),q,source,target),cell,axis_step(nest_inner_of(outer)))) :
        {inner_ran(~Element,p,Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))),axis_step(nest_inner_of(outer)),extent,stride,back,(row_source(nest_outer_of(outer),q,source,target) + U32.from_nat(1n+cell) * Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))) : U32),_,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran)) ==
          model_rows(~Element,p,1n+index,outer,extent,stride,back,source,target,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran)) : Held.Ran<Element>}
      %Addition.successor_right(Nat.mul(q,width),cell) :
        {inner_ran(~Element,p,Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))),axis_step(nest_inner_of(outer)),extent,stride,back,(row_source(nest_outer_of(outer),q,source,target) + U32.from_nat(1n+cell) * Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))) : U32),(row_target(nest_outer_of(outer),q,source,target) + U32.from_nat(1n+cell) * axis_step(nest_inner_of(outer)) : U32),
            Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran)) ==
          model_rows(~Element,p,_,outer,extent,stride,back,source,target,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran)) : Held.Ran<Element>}
      inner_flat(~Element,p,1n+cell,q,outer,extent,stride,back,source,target,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran),cap,cap_ok,room,q_in,
        {%Addition.successor_right(cell,p) : {Nat.is_le(_,width) == True{} : Bool} within : {Nat.is_le(Nat.add(1n+cell,p),width) == True{} : Bool}})

# The rows of one block are the rows of the box from the block times the
# inner extent on.
def block_flat(~Element: Data,+q: Nat,+outer: List<&2,Move.Axis>,+extent: U32,+stride: U32,+back: Bool,+source: U32,+target: U32,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(total(outer)),U32.to_nat(cap)) == True{} : Bool},+q_in: {Nat.is_lt(q,total(nest_outer_of(outer))) == True{} : Bool}) ->
  {model_rows(~Element,U32.to_nat(axis_extent(nest_inner_of(outer))),Nat.mul(q,U32.to_nat(axis_extent(nest_inner_of(outer)))),outer,extent,stride,back,source,target,holding,tree) ==
    block_ran(~Element,nest_inner_of(outer),extent,stride,back,Move.row_offsets(nest_outer_of(outer),U32.from_nat(q),source,target),holding,tree) : Held.Ran<Element>}:
  +width = U32.to_nat(axis_extent(nest_inner_of(outer)))
  %Addition.zero_right(Nat.mul(q,width)) :
    {model_rows(~Element,width,_,outer,extent,stride,back,source,target,holding,tree) ==
      block_ran(~Element,nest_inner_of(outer),extent,stride,back,Move.row_offsets(nest_outer_of(outer),U32.from_nat(q),source,target),holding,tree) : Held.Ran<Element>}
  %zero_step(row_source(nest_outer_of(outer),q,source,target),Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer)))) :
    {model_rows(~Element,width,Nat.add(Nat.mul(q,width),0n),outer,extent,stride,back,source,target,holding,tree) ==
      inner_ran(~Element,width,Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))),axis_step(nest_inner_of(outer)),extent,stride,back,_,row_target(nest_outer_of(outer),q,source,target),holding,tree) : Held.Ran<Element>}
  %zero_step(row_target(nest_outer_of(outer),q,source,target),axis_step(nest_inner_of(outer))) :
    {model_rows(~Element,width,Nat.add(Nat.mul(q,width),0n),outer,extent,stride,back,source,target,holding,tree) ==
      inner_ran(~Element,width,Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))),axis_step(nest_inner_of(outer)),extent,stride,back,(row_source(nest_outer_of(outer),q,source,target) + U32.from_nat(0n) * Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))) : U32),_,holding,tree) : Held.Ran<Element>}
  Equal.sym(Held.Ran<Element>,
    inner_ran(~Element,width,Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))),axis_step(nest_inner_of(outer)),extent,stride,back,(row_source(nest_outer_of(outer),q,source,target) + U32.from_nat(0n) * Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))) : U32),(row_target(nest_outer_of(outer),q,source,target) + U32.from_nat(0n) * axis_step(nest_inner_of(outer)) : U32),holding,tree),
    model_rows(~Element,width,Nat.add(Nat.mul(q,width),0n),outer,extent,stride,back,source,target,holding,tree),
    inner_flat(~Element,width,0n,q,outer,extent,stride,back,source,target,holding,tree,cap,cap_ok,room,q_in,Order.reflexive(width)))

# The blocks from q on are the rows of the box from block q on.
def blocks_flat(~Element: Data,count: Nat,+q: Nat,+outer: List<&2,Move.Axis>,+extent: U32,+stride: U32,+back: Bool,+source: U32,+target: U32,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(total(outer)),U32.to_nat(cap)) == True{} : Bool},+within: {Nat.is_le(Nat.add(q,count),total(nest_outer_of(outer))) == True{} : Bool}) ->
  {blocks_ran(~Element,count,q,nest_outer_of(outer),nest_inner_of(outer),extent,stride,back,source,target,holding,tree) ==
    model_rows(~Element,Nat.mul(count,U32.to_nat(axis_extent(nest_inner_of(outer)))),Nat.mul(q,U32.to_nat(axis_extent(nest_inner_of(outer)))),outer,extent,stride,back,source,target,holding,tree) : Held.Ran<Element>}:
  match count:
    case 0n: {==}
    case 1n+ +p:
      +width = U32.to_nat(axis_extent(nest_inner_of(outer)))
      +q_in = Order.room_index(q,p,total(nest_outer_of(outer)),within)
      +ran = block_ran(~Element,nest_inner_of(outer),extent,stride,back,Move.row_offsets(nest_outer_of(outer),U32.from_nat(q),source,target),holding,tree)
      +first = model_rows(~Element,width,Nat.mul(q,width),outer,extent,stride,back,source,target,holding,tree)
      %Equal.sym(Held.Ran<Element>,model_rows(~Element,Nat.add(width,Nat.mul(p,width)),Nat.mul(q,width),outer,extent,stride,back,source,target,holding,tree),
          model_rows(~Element,Nat.mul(p,width),Nat.add(width,Nat.mul(q,width)),outer,extent,stride,back,source,target,Held.ran_holding(~Element,first),
            Held.ran_tree(~Element,first)),
          rows_append(~Element,width,Nat.mul(p,width),Nat.mul(q,width),outer,extent,stride,back,source,target,holding,tree)) :
        {blocks_ran(~Element,p,1n+q,nest_outer_of(outer),nest_inner_of(outer),extent,stride,back,source,target,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran)) == _ : Held.Ran<Element>}
      %Equal.sym(Held.Ran<Element>,first,ran,block_flat(~Element,q,outer,extent,stride,back,source,target,holding,tree,cap,cap_ok,room,q_in)) :
        {blocks_ran(~Element,p,1n+q,nest_outer_of(outer),nest_inner_of(outer),extent,stride,back,source,target,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran)) ==
          model_rows(~Element,Nat.mul(p,width),Nat.add(width,Nat.mul(q,width)),outer,extent,stride,back,source,target,Held.ran_holding(~Element,_),
            Held.ran_tree(~Element,_)) : Held.Ran<Element>}
      blocks_flat(~Element,p,1n+q,outer,extent,stride,back,source,target,Held.ran_holding(~Element,ran),Held.ran_tree(~Element,ran),cap,cap_ok,room,
        {%Addition.successor_right(q,p) : {Nat.is_le(_,total(nest_outer_of(outer))) == True{} : Bool} within : {Nat.is_le(Nat.add(1n+q,p),total(nest_outer_of(outer))) == True{} : Bool}})

# The nested rows of the axes before the run are their rows.
def nested_flat(~Element: Data,+outer: List<&2,Move.Axis>,+extent: U32,+stride: U32,+back: Bool,+source: U32,+target: U32,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(total(outer)),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,total(outer)) == True{} : Bool}) ->
  {nested_ran(~Element,Move.nest_of(outer),extent,stride,back,source,target,holding,tree) ==
    model_rows(~Element,U32.to_nat(Move.rows_of(outer)),0n,outer,extent,stride,back,source,target,holding,tree) : Held.Ran<Element>}:
  +width = U32.to_nat(axis_extent(nest_inner_of(outer)))
  +product = nest_total(outer)
  +back_product = Equal.sym(Nat,total(outer),Nat.mul(total(nest_outer_of(outer)),width),product)
  +nest_positive = {%product : {Nat.is_lt(0n,_) == True{} : Bool} positive : {Nat.is_lt(0n,Nat.mul(total(nest_outer_of(outer)),width)) == True{} : Bool}}
  +blocks_positive = RowMajor.left_factor_positive(total(nest_outer_of(outer)),width,nest_positive)
  +width_positive = DivisionOrder.right_factor_positive(total(nest_outer_of(outer)),width,nest_positive)
  +blocks_room = Order.transitive(total(nest_outer_of(outer)),total(outer),U32.to_nat(cap),
    {%back_product : {Nat.is_le(total(nest_outer_of(outer)),_) == True{} : Bool} RowMajor.product_covers(total(nest_outer_of(outer)),width,width_positive) :
      {Nat.is_le(total(nest_outer_of(outer)),total(outer)) == True{} : Bool}},
    single_room(total(outer),cap,room))
  +blocks_exact = rows_exact(nest_outer_of(outer),cap,cap_ok,blocks_positive,blocks_room)
  +rows_all = rows_exact(outer,cap,cap_ok,positive,single_room(total(outer),cap,room))
  %Equal.sym(Nat,U32.to_nat(Move.rows_of(outer)),total(outer),rows_all) :
    {nested_ran(~Element,Move.nest_of(outer),extent,stride,back,source,target,holding,tree) ==
      model_rows(~Element,_,0n,outer,extent,stride,back,source,target,holding,tree) : Held.Ran<Element>}
  %back_product :
    {nested_ran(~Element,Move.nest_of(outer),extent,stride,back,source,target,holding,tree) ==
      model_rows(~Element,_,0n,outer,extent,stride,back,source,target,holding,tree) : Held.Ran<Element>}
  %blocks_exact :
    {nested_ran(~Element,Move.nest_of(outer),extent,stride,back,source,target,holding,tree) ==
      model_rows(~Element,Nat.mul(_,width),0n,outer,extent,stride,back,source,target,holding,tree) : Held.Ran<Element>}
  blocks_flat(~Element,U32.to_nat(Move.rows_of(nest_outer_of(outer))),0n,outer,extent,stride,back,source,target,holding,tree,cap,cap_ok,room,
    {%Equal.sym(Nat,U32.to_nat(Move.rows_of(nest_outer_of(outer))),total(nest_outer_of(outer)),blocks_exact) : {Nat.is_le(_,total(nest_outer_of(outer))) == True{} : Bool} Order.reflexive(total(nest_outer_of(outer))) : {Nat.is_le(Nat.add(0n,U32.to_nat(Move.rows_of(nest_outer_of(outer)))),total(nest_outer_of(outer))) == True{} : Bool}})

# ---- The piece path ----

def not_true(+flag: Bool,+negated: {Bool.not(flag) == True{} : Bool}) -> {flag == False{} : Bool}:
  match flag:
    case True{}: Empty.absurd({True{} == False{} : Bool},WordEncoding.false_true(negated))
    case False{}: {==}

# A word less than or equal to zero is no positive extent.
def zero_extent(+extent: U32,+positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},+none: {U32.is_le(extent,0) == True{} : Bool}) -> Empty:
  Held.lt_le_absurd(0n,U32.to_nat(extent),positive,{%Comparison.le(extent,0) : {_ == True{} : Bool} none : {Nat.is_le(U32.to_nat(extent),0n) == True{} : Bool}})

def sub_self(+value: U32) -> {(value - value : U32) == 0 : U32}:
  %Arithmetic.add_zero(value) : {(_ - value : U32) == 0 : U32}
  Arithmetic.subtract_first(value,0)

# Subtracting before or after adding is the same.
def shift_swap(+value: U32,+step: U32,+shift: U32) -> {((value - shift : U32) + step : U32) == ((value + step : U32) - shift : U32) : U32}:
  Equal.trans(U32,((value - shift : U32) + step : U32),(step + (value - shift : U32) : U32),((value + step : U32) - shift : U32),
    Arithmetic.add_commutative((value - shift : U32),step),
    Equal.trans(U32,(step + (value - shift : U32) : U32),((step + value : U32) - shift : U32),((value + step : U32) - shift : U32),
      Equal.sym(U32,((step + value : U32) - shift : U32),(step + (value - shift : U32) : U32),Arithmetic.subtract_from_sum(step,value,shift)),
      Equal.cong(U32,U32,sum => (sum - shift : U32),(step + value : U32),(value + step : U32),Arithmetic.add_commutative(step,value))))

# Rows from a source shifted back read the source cells shifted back.
def offsets_shift(axes: List<&2,Move.Axis>,+row: U32,+source: U32,+shift: U32,+target: U32) ->
  {Move.row_offsets(axes,row,(source - shift : U32),target) ==
    Move.Offsets{(offsets_source(Move.row_offsets(axes,row,source,target)) - shift : U32),offsets_target(Move.row_offsets(axes,row,source,target))} : Move.Offsets}:
  match axes:
    case Nil{}: {==}
    case Con{Move.Axis{+extent,+stride,+back,+step},+rest}:
      +inner = U32.max(Move.rows_of(rest),1)
      +index = (row / inner % U32.max(extent,1) : U32)
      +moved = (index * Move.directed(back,stride) : U32)
      %Equal.sym(U32,((source - shift : U32) + moved : U32),((source + moved : U32) - shift : U32),shift_swap(source,moved,shift)) :
        {Move.row_offsets(rest,(row % inner : U32),_,(target + index * step : U32)) ==
          Move.Offsets{(offsets_source(Move.row_offsets(rest,(row % inner : U32),(source + moved : U32),(target + index * step : U32))) - shift : U32),
            offsets_target(Move.row_offsets(rest,(row % inner : U32),(source + moved : U32),(target + index * step : U32)))} : Move.Offsets}
      offsets_shift(rest,(row % inner : U32),(source + moved : U32),shift,(target + index * step : U32))

# A step from a cell the current piece holds keeps it current.
def advanced_stop(~Element: Data,below: Held.Stack<Element>,+index: U32,+buffer: Buffer.Model<Element>,+first: U32,+span: U32,+behind: Held.Stack<Element>) ->
  {Held.advanced(~Element,below,False{},index,buffer,first,span,behind) == Held.Holding{behind,Held.Held{buffer,first,span,below}} : Held.Holding<Element>}:
  match below:
    case Held.Held{+next,+next_first,+next_count,+rest}: {==}
    case Held.Bottom{}: {==}

def resolve_inside(~Element: Data,+behind: Held.Stack<Element>,+buffer: Buffer.Model<Element>,+first: U32,+span: U32,+below: Held.Stack<Element>,+index: U32,
  +after: {U32.is_le(first,index) == True{} : Bool},+before: {U32.is_lt(index,(first + span : U32)) == True{} : Bool}) ->
  {Held.resolve(~Element,Held.Holding{behind,Held.Held{buffer,first,span,below}},index) == Held.Holding{behind,Held.Held{buffer,first,span,below}} : Held.Holding<Element>}:
  +not_before = {%Equal.sym(Bool,U32.is_lt(index,first),Nat.is_lt(U32.to_nat(index),U32.to_nat(first)),Comparison.lt(index,first)) : {_ == False{} : Bool}
    Order.not_below(U32.to_nat(first),U32.to_nat(index),Bounds.as_nat_le(first,index,after)) : {U32.is_lt(index,first) == False{} : Bool}}
  +not_after = {%Equal.sym(Bool,U32.is_le((first + span : U32),index),Nat.is_le(U32.to_nat((first + span : U32)),U32.to_nat(index)),Comparison.le((first + span : U32),index)) :
      {_ == False{} : Bool}
    Held.lt_not_le(U32.to_nat(index),U32.to_nat((first + span : U32)),Bounds.as_nat_lt(index,(first + span : U32),before)) :
    {U32.is_le((first + span : U32),index) == False{} : Bool}}
  %Equal.sym(Bool,U32.is_lt(index,first),False{},not_before) : {Held.located(~Element,_,index,behind,buffer,first,span,below) == Held.Holding{behind,Held.Held{buffer,first,span,below}} : Held.Holding<Element>}
  %Equal.sym(Bool,U32.is_le((first + span : U32),index),False{},not_after) : {Held.advanced(~Element,below,_,index,buffer,first,span,behind) == Held.Holding{behind,Held.Held{buffer,first,span,below}} : Held.Holding<Element>}
  advanced_stop(~Element,below,index,buffer,first,span,behind)

# A run whose cells the piece holds starts in the piece.
def fits_holds(+e: U32,+o: U32,+s: U32,+first: U32,+span: U32,+positive: {Nat.is_lt(0n,U32.to_nat(e)) == True{} : Bool},
  +le: {U32.is_le(e,Move.inside(e,o,s,False{},first,span)) == True{} : Bool}) -> Held.Both<{U32.is_le(first,o) == True{} : Bool},{U32.is_lt(o,(first + span : U32)) == True{} : Bool}>:
  +inner = Bool.pick(U32,U32.is_zero(s),e,U32.min(e,((Held.step_reach(False{},o,first,span) / U32.max(s,1) : U32) + 1 : U32)))
  Held.by_cases(Held.Both<{U32.is_le(first,o) == True{} : Bool},{U32.is_lt(o,(first + span : U32)) == True{} : Bool}>,U32.is_le(first,o),
    +after => Held.by_cases(Held.Both<{U32.is_le(first,o) == True{} : Bool},{U32.is_lt(o,(first + span : U32)) == True{} : Bool}>,U32.is_lt(o,(first + span : U32)),
      +before => Held.Both{after,before},
      +outside => Empty.absurd(Held.Both<{U32.is_le(first,o) == True{} : Bool},{U32.is_lt(o,(first + span : U32)) == True{} : Bool}>,zero_extent(e,positive,
        {%after : {U32.is_le(e,Bool.pick(U32,_ && False{},inner,0)) == True{} : Bool}
          {%outside : {U32.is_le(e,Bool.pick(U32,U32.is_le(first,o) && _,inner,0)) == True{} : Bool} le :
            {U32.is_le(e,Bool.pick(U32,U32.is_le(first,o) && False{},inner,0)) == True{} : Bool}} :
          {U32.is_le(e,0) == True{} : Bool}}))),
    +early => Empty.absurd(Held.Both<{U32.is_le(first,o) == True{} : Bool},{U32.is_lt(o,(first + span : U32)) == True{} : Bool}>,zero_extent(e,positive,
      {%early : {U32.is_le(e,Bool.pick(U32,_ && U32.is_lt(o,(first + span : U32)),inner,0)) == True{} : Bool} le : {U32.is_le(e,0) == True{} : Bool}})))

# A run the piece holds copies all its cells from it.
def inside_whole(+extent: U32,+offset: U32,+stride: U32,+first: U32,+span: U32,+le: {U32.is_le(extent,Move.inside(extent,offset,stride,False{},first,span)) == True{} : Bool}) -> {Move.inside(extent,offset,stride,False{},first,span) == extent : U32}:
  Arithmetic.to_nat_injective(Move.inside(extent,offset,stride,False{},first,span),extent,
    Order.antisymmetric(U32.to_nat(Move.inside(extent,offset,stride,False{},first,span)),U32.to_nat(extent),Held.cells_most(extent,offset,stride,False{},first,span),Bounds.as_nat_le(extent,Move.inside(extent,offset,stride,False{},first,span),le)))

# A copy of no cells writes nothing.
def copy_none(~Element: Data,fast: Bool,source: Views.View,+destination: U32,+input: Tree.Tree<Element>) ->
  {CopyValues.copy_writes(~Element,fast,0n,source,destination,input) == [] : List<&2,Writes.Assignment<Element>>}:
  match fast source:
    case True{} Views.View{+axes,+offset}: {==}
    case False{} +source: {==}

# A run that is done keeps the holding and the tree.
def run_done(~Element: Data,fuel: Nat,+count: U32,+offset: U32,+stride: U32,+back: Bool,+position: U32,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>) ->
  {Held.model_run(~Element,fuel,True{},count,offset,stride,back,position,holding,tree) == Held.Ran{holding,tree} : Held.Ran<Element>}:
  match fuel:
    case 0n: {==}
    case 1n+ +rest: {==}

# A run the current piece holds whole copies its cells from the piece and
# keeps the piece current.
def row_piece_with(~Element: Data,fuel: Nat,done: Bool,+extent: U32,+stride: U32,+back: Bool,+target_stride: U32,+offset: U32,+position: U32,+behind: Held.Stack<Element>,+buffer: Buffer.Model<Element>,+first: U32,+span: U32,+below: Held.Stack<Element>,
  +tree: Tree.Tree<Element>,+fuel_exact: {U32.to_nat(extent) == fuel : Nat},+done_exact: {U32.is_zero(extent) == done : Bool},
  +fits: {Move.run_fits(Move.Axis{extent,stride,back,target_stride},offset,first,span) == True{} : Bool}) ->
  {Held.model_run(~Element,fuel,done,extent,offset,stride,back,position,Held.Holding{behind,Held.Held{buffer,first,span,below}},tree) ==
    Held.Ran{Held.Holding{behind,Held.Held{buffer,first,span,below}},run_copy(~Element,extent,(offset - first : U32),stride,position,Buffer.tree(~Element,buffer),tree)} : Held.Ran<Element>}:
  match fuel done:
    case 0n done:
      %Equal.sym(Nat,U32.to_nat(extent),0n,fuel_exact) :
        {Held.Ran{Held.Holding{behind,Held.Held{buffer,first,span,below}},tree} == Held.Ran{Held.Holding{behind,Held.Held{buffer,first,span,below}},Writes.pure(Element,CopyValues.copy_writes(~Element,U32.is_eq(stride,1) && Bool.not(False{}),_,
          Views.View{[Views.Axis{extent,Move.directed(False{},stride)}],(offset - first : U32)},position,Buffer.tree(~Element,buffer)),tree)} : Held.Ran<Element>}
      %Equal.sym(List<&2,Writes.Assignment<Element>>,CopyValues.copy_writes(~Element,U32.is_eq(stride,1) && Bool.not(False{}),0n,Views.View{[Views.Axis{extent,Move.directed(False{},stride)}],(offset - first : U32)},
          position,Buffer.tree(~Element,buffer)),[],copy_none(~Element,U32.is_eq(stride,1) && Bool.not(False{}),Views.View{[Views.Axis{extent,Move.directed(False{},stride)}],(offset - first : U32)},position,Buffer.tree(~Element,buffer))) :
        {Held.Ran{Held.Holding{behind,Held.Held{buffer,first,span,below}},tree} == Held.Ran{Held.Holding{behind,Held.Held{buffer,first,span,below}},Writes.pure(Element,_,tree)} : Held.Ran<Element>}
      {==}
    case 1n+ +rest True{}:
      +positive = {%Equal.sym(Nat,U32.to_nat(extent),1n+rest,fuel_exact) : {Nat.is_lt(0n,_) == True{} : Bool} {==} : {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool}}
      Empty.absurd({Held.model_run(~Element,1n+rest,True{},extent,offset,stride,back,position,Held.Holding{behind,Held.Held{buffer,first,span,below}},tree) ==
          Held.Ran{Held.Holding{behind,Held.Held{buffer,first,span,below}},run_copy(~Element,extent,(offset - first : U32),stride,position,Buffer.tree(~Element,buffer),tree)} : Held.Ran<Element>},
        WordEncoding.false_true(Equal.trans(Bool,False{},U32.is_zero(extent),True{},
          Equal.sym(Bool,U32.is_zero(extent),False{},Arithmetic.word_positive_nonzero(extent,positive)),done_exact)))
    case 1n+ +rest False{}:
      +positive = {%Equal.sym(Nat,U32.to_nat(extent),1n+rest,fuel_exact) : {Nat.is_lt(0n,_) == True{} : Bool} {==} : {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool}}
      +back_false = not_true(back,and_left(Bool.not(back),U32.is_le(extent,Move.inside(extent,offset,stride,False{},first,span)),fits))
      +le = and_right(Bool.not(back),U32.is_le(extent,Move.inside(extent,offset,stride,False{},first,span)),fits)
      +holds = fits_holds(extent,offset,stride,first,span,positive,le)
      +after = Held.first_of({U32.is_le(first,offset) == True{} : Bool},{U32.is_lt(offset,(first + span : U32)) == True{} : Bool},holds)
      +before = Held.second_of({U32.is_le(first,offset) == True{} : Bool},{U32.is_lt(offset,(first + span : U32)) == True{} : Bool},holds)
      +same = resolve_inside(~Element,behind,buffer,first,span,below,offset,after,before)
      +whole = inside_whole(extent,offset,stride,first,span,le)
      +copied = run_copy(~Element,extent,(offset - first : U32),stride,position,Buffer.tree(~Element,buffer),tree)
      %Equal.sym(Bool,back,False{},back_false) :
        {Held.model_run(~Element,1n+rest,False{},extent,offset,stride,_,position,Held.Holding{behind,Held.Held{buffer,first,span,below}},tree) == Held.Ran{Held.Holding{behind,Held.Held{buffer,first,span,below}},copied} : Held.Ran<Element>}
      %Equal.sym(Held.Holding<Element>,Held.resolve(~Element,Held.Holding{behind,Held.Held{buffer,first,span,below}},offset),Held.Holding{behind,Held.Held{buffer,first,span,below}},same) :
        {Held.model_run(~Element,rest,U32.is_zero((extent - Held.step_cells(~Element,_,extent,offset,stride,False{}) : U32)),(extent - Held.step_cells(~Element,_,extent,offset,stride,False{}) : U32),(offset + Held.step_cells(~Element,_,extent,offset,stride,False{}) * Move.directed(False{},stride) : U32),stride,False{},(position + Held.step_cells(~Element,_,extent,offset,stride,False{}) : U32),_,Held.step_tree(~Element,_,extent,offset,stride,False{},position,tree)) == Held.Ran{Held.Holding{behind,Held.Held{buffer,first,span,below}},copied} : Held.Ran<Element>}
      %Equal.sym(U32,Move.inside(extent,offset,stride,False{},first,span),extent,whole) :
        {Held.model_run(~Element,rest,U32.is_zero((extent - _ : U32)),(extent - _ : U32),(offset + _ * Move.directed(False{},stride) : U32),stride,False{},(position + _ : U32),Held.Holding{behind,Held.Held{buffer,first,span,below}},Writes.pure(Element,CopyValues.copy_writes(~Element,U32.is_eq(stride,1) && Bool.not(False{}),U32.to_nat(_),Views.View{[Views.Axis{_,Move.directed(False{},stride)}],(offset - first : U32)},position,Buffer.tree(~Element,buffer)),tree)) == Held.Ran{Held.Holding{behind,Held.Held{buffer,first,span,below}},copied} : Held.Ran<Element>}
      %Equal.sym(U32,(extent - extent : U32),0,sub_self(extent)) :
        {Held.model_run(~Element,rest,U32.is_zero(_),_,(offset + extent * Move.directed(False{},stride) : U32),stride,False{},(position + extent : U32),Held.Holding{behind,Held.Held{buffer,first,span,below}},copied) == Held.Ran{Held.Holding{behind,Held.Held{buffer,first,span,below}},copied} : Held.Ran<Element>}
      run_done(~Element,rest,0,(offset + extent * Move.directed(False{},stride) : U32),stride,False{},(position + extent : U32),Held.Holding{behind,Held.Held{buffer,first,span,below}},copied)

def row_piece(~Element: Data,+extent: U32,+stride: U32,+back: Bool,+target_stride: U32,+offset: U32,+position: U32,+behind: Held.Stack<Element>,+buffer: Buffer.Model<Element>,+first: U32,+span: U32,+below: Held.Stack<Element>,+tree: Tree.Tree<Element>,
  +fits: {Move.run_fits(Move.Axis{extent,stride,back,target_stride},offset,first,span) == True{} : Bool}) ->
  {Held.model_run(~Element,U32.to_nat(extent),U32.is_zero(extent),extent,offset,stride,back,position,Held.Holding{behind,Held.Held{buffer,first,span,below}},tree) ==
    Held.Ran{Held.Holding{behind,Held.Held{buffer,first,span,below}},run_copy(~Element,extent,(offset - first : U32),stride,position,Buffer.tree(~Element,buffer),tree)} : Held.Ran<Element>}:
  row_piece_with(~Element,U32.to_nat(extent),U32.is_zero(extent),extent,stride,back,target_stride,offset,position,behind,buffer,first,span,below,tree,{==},{==},fits)

# Every run of a block, a step apart, that the piece holds whole.
def RowsFit(count: Nat,+step: U32,+run: Move.Axis,+offset: U32,+first: U32,+span: U32) -> Data:
  match count:
    case 0n: Unit
    case 1n+rest: Held.Both<{Move.run_fits(run,offset,first,span) == True{} : Bool},RowsFit(rest,step,run,(offset + step : U32),first,span)>

def BlocksFit(count: Nat,+row: Nat,+outer: List<&2,Move.Axis>,+inner: Move.Axis,+run: Move.Axis,+source: U32,+target: U32,+first: U32,+span: U32) -> Data:
  match count:
    case 0n: Unit
    case 1n+rest:
      Held.Both<RowsFit(U32.to_nat(axis_extent(inner)),Move.directed(axis_back(inner),axis_stride(inner)),run,row_source(outer,row,source,target),first,span),
        BlocksFit(rest,1n+row,outer,inner,run,source,target,first,span)>

# Rows of a block the piece holds are copies from the piece.
def inner_piece(~Element: Data,count: Nat,+step: U32,+inner_target: U32,+extent: U32,+stride: U32,+back: Bool,+target_stride: U32,+offset: U32,+position: U32,+behind: Held.Stack<Element>,+buffer: Buffer.Model<Element>,+first: U32,+span: U32,+below: Held.Stack<Element>,
  +tree: Tree.Tree<Element>,+fit: RowsFit(count,step,Move.Axis{extent,stride,back,target_stride},offset,first,span)) ->
  {inner_ran(~Element,count,step,inner_target,extent,stride,back,offset,position,Held.Holding{behind,Held.Held{buffer,first,span,below}},tree) ==
    Held.Ran{Held.Holding{behind,Held.Held{buffer,first,span,below}},piece_inner_tree(~Element,count,step,inner_target,extent,stride,(offset - first : U32),position,Buffer.tree(~Element,buffer),tree)} : Held.Ran<Element>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      Held.Both{+here,+later} = fit
      +run = Held.model_run(~Element,U32.to_nat(extent),U32.is_zero(extent),extent,offset,stride,back,position,Held.Holding{behind,Held.Held{buffer,first,span,below}},tree)
      +copied = run_copy(~Element,extent,(offset - first : U32),stride,position,Buffer.tree(~Element,buffer),tree)
      %Equal.sym(Held.Ran<Element>,run,Held.Ran{Held.Holding{behind,Held.Held{buffer,first,span,below}},copied},row_piece(~Element,extent,stride,back,target_stride,offset,position,behind,buffer,first,span,below,tree,here)) :
        {inner_ran(~Element,rest,step,inner_target,extent,stride,back,(offset + step : U32),(position + inner_target : U32),Held.ran_holding(~Element,_),Held.ran_tree(~Element,_)) ==
          Held.Ran{Held.Holding{behind,Held.Held{buffer,first,span,below}},piece_inner_tree(~Element,rest,step,inner_target,extent,stride,((offset - first : U32) + step : U32),(position + inner_target : U32),Buffer.tree(~Element,buffer),copied)} : Held.Ran<Element>}
      %Equal.sym(U32,((offset - first : U32) + step : U32),((offset + step : U32) - first : U32),shift_swap(offset,step,first)) :
        {inner_ran(~Element,rest,step,inner_target,extent,stride,back,(offset + step : U32),(position + inner_target : U32),Held.Holding{behind,Held.Held{buffer,first,span,below}},copied) ==
          Held.Ran{Held.Holding{behind,Held.Held{buffer,first,span,below}},piece_inner_tree(~Element,rest,step,inner_target,extent,stride,_,(position + inner_target : U32),Buffer.tree(~Element,buffer),copied)} : Held.Ran<Element>}
      inner_piece(~Element,rest,step,inner_target,extent,stride,back,target_stride,(offset + step : U32),(position + inner_target : U32),behind,buffer,first,span,below,copied,later)

# Blocks the piece holds are copies from the piece.
def blocks_piece(~Element: Data,count: Nat,+row: Nat,+outer: List<&2,Move.Axis>,+inner: Move.Axis,+extent: U32,+stride: U32,+back: Bool,+target_stride: U32,
  +source: U32,+target: U32,+behind: Held.Stack<Element>,+buffer: Buffer.Model<Element>,+first: U32,+span: U32,+below: Held.Stack<Element>,+tree: Tree.Tree<Element>,+fit: BlocksFit(count,row,outer,inner,Move.Axis{extent,stride,back,target_stride},source,target,first,span)) ->
  {blocks_ran(~Element,count,row,outer,inner,extent,stride,back,source,target,Held.Holding{behind,Held.Held{buffer,first,span,below}},tree) ==
    Held.Ran{Held.Holding{behind,Held.Held{buffer,first,span,below}},piece_rows_tree(~Element,count,row,outer,inner,extent,stride,(source - first : U32),target,Buffer.tree(~Element,buffer),tree)} : Held.Ran<Element>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      Held.Both{+here,+later} = fit
      +offsets = Move.row_offsets(outer,U32.from_nat(row),source,target)
      +width = U32.to_nat(axis_extent(inner))
      +step = Move.directed(axis_back(inner),axis_stride(inner))
      +inner_target = axis_step(inner)
      +copied = piece_inner_tree(~Element,width,step,inner_target,extent,stride,(offsets_source(offsets) - first : U32),offsets_target(offsets),Buffer.tree(~Element,buffer),tree)
      +block = inner_ran(~Element,width,step,inner_target,extent,stride,back,offsets_source(offsets),offsets_target(offsets),Held.Holding{behind,Held.Held{buffer,first,span,below}},tree)
      %Equal.sym(Held.Ran<Element>,block,Held.Ran{Held.Holding{behind,Held.Held{buffer,first,span,below}},copied},
          inner_piece(~Element,width,step,inner_target,extent,stride,back,target_stride,offsets_source(offsets),offsets_target(offsets),behind,buffer,first,span,below,tree,here)) :
        {blocks_ran(~Element,rest,1n+row,outer,inner,extent,stride,back,source,target,Held.ran_holding(~Element,_),Held.ran_tree(~Element,_)) ==
          Held.Ran{Held.Holding{behind,Held.Held{buffer,first,span,below}},piece_rows_tree(~Element,rest,1n+row,outer,inner,extent,stride,(source - first : U32),target,Buffer.tree(~Element,buffer),
            piece_block_tree(~Element,inner,extent,stride,Move.row_offsets(outer,U32.from_nat(row),(source - first : U32),target),Buffer.tree(~Element,buffer),tree))} : Held.Ran<Element>}
      %Equal.sym(Move.Offsets,Move.row_offsets(outer,U32.from_nat(row),(source - first : U32),target),
          Move.Offsets{(offsets_source(offsets) - first : U32),offsets_target(offsets)},offsets_shift(outer,U32.from_nat(row),source,first,target)) :
        {blocks_ran(~Element,rest,1n+row,outer,inner,extent,stride,back,source,target,Held.Holding{behind,Held.Held{buffer,first,span,below}},copied) ==
          Held.Ran{Held.Holding{behind,Held.Held{buffer,first,span,below}},piece_rows_tree(~Element,rest,1n+row,outer,inner,extent,stride,(source - first : U32),target,Buffer.tree(~Element,buffer),
            piece_block_tree(~Element,inner,extent,stride,_,Buffer.tree(~Element,buffer),tree))} : Held.Ran<Element>}
      blocks_piece(~Element,rest,1n+row,outer,inner,extent,stride,back,target_stride,source,target,behind,buffer,first,span,below,copied,later)

def nested_piece(~Element: Data,+nest: Move.Nest,+extent: U32,+stride: U32,+back: Bool,+target_stride: U32,+source: U32,+target: U32,+behind: Held.Stack<Element>,+buffer: Buffer.Model<Element>,+first: U32,+span: U32,+below: Held.Stack<Element>,+tree: Tree.Tree<Element>,
  +fit: BlocksFit(U32.to_nat(Move.rows_of(nest_outer(nest))),0n,nest_outer(nest),nest_inner(nest),Move.Axis{extent,stride,back,target_stride},source,target,first,span)) ->
  {nested_ran(~Element,nest,extent,stride,back,source,target,Held.Holding{behind,Held.Held{buffer,first,span,below}},tree) ==
    Held.Ran{Held.Holding{behind,Held.Held{buffer,first,span,below}},piece_nested_tree(~Element,nest,extent,stride,(source - first : U32),target,Buffer.tree(~Element,buffer),tree)} : Held.Ran<Element>}:
  blocks_piece(~Element,U32.to_nat(Move.rows_of(nest_outer(nest))),0n,nest_outer(nest),nest_inner(nest),extent,stride,back,target_stride,source,target,behind,buffer,first,span,below,tree,fit)

# ---- A part the current piece holds ----

# Cell low lies in the piece from first and at most at cell high.
type Lower<-first: U32,-low: U32,-high: U32> is Data:
  Lower{after: {U32.is_le(first,low) == True{} : Bool},order: {Nat.is_le(U32.to_nat(low),U32.to_nat(high)) == True{} : Bool}}

def lower_after(-first: U32,-low: U32,-high: U32,+lower: Lower<first,low,high>) -> {U32.is_le(first,low) == True{} : Bool}:
  Lower{+after,+order} = lower
  after

def lower_order(-first: U32,-low: U32,-high: U32,+lower: Lower<first,low,high>) -> {Nat.is_le(U32.to_nat(low),U32.to_nat(high)) == True{} : Bool}:
  Lower{+after,+order} = lower
  order

# Both bounds of a cell, with the room of the piece, are a held cell.
def holds_from(+offset: U32,+first: U32,+span: U32,+capacity: U32,+both: Held.Both<{U32.is_le(first,offset) == True{} : Bool},{U32.is_lt(offset,(first + span : U32)) == True{} : Bool}>,
  +room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat(capacity)) == True{} : Bool}) -> Held.Holds<first,span,offset,capacity>:
  Held.Both{+after,+before} = both
  Held.Holds{after,before,room}

# A held cell lies before the end of the piece.
def holds_before(+offset: U32,+first: U32,+span: U32,+capacity: U32,+holds: Held.Holds<first,span,offset,capacity>) -> {U32.is_lt(offset,(first + span : U32)) == True{} : Bool}:
  Held.Holds{+after,+before,+inside} = holds
  before

# A run of some cells that the piece holds starts in the piece.
def holds_of(+extent: U32,+offset: U32,+stride: U32,+first: U32,+span: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat(capacity)) == True{} : Bool},+word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},+le: {U32.is_le(extent,Move.inside(extent,offset,stride,False{},first,span)) == True{} : Bool}) -> Held.Holds<first,span,offset,capacity>:
  holds_from(offset,first,span,capacity,fits_holds(extent,offset,stride,first,span,positive,le),room)

# Forward from a cell of the piece, the cell and its reach lie in the piece.
def reach_end(+offset: U32,+first: U32,+span: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat(capacity)) == True{} : Bool},+word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+holds: Held.Holds<first,span,offset,capacity>) -> {Nat.is_lt(Nat.add(U32.to_nat(offset),U32.to_nat(Held.step_reach(False{},offset,first,span))),Nat.add(U32.to_nat(first),U32.to_nat(span))) == True{} : Bool}:
  +cell = U32.to_nat(offset)
  +far = U32.to_nat(Held.step_reach(False{},offset,first,span))
  +whole = Held.forward_reach(offset,first,span,capacity,capacity_ok,holds)
  +below = Held.offset_below(offset,first,span,capacity,capacity_ok,holds)
  +sum = Equal.trans(Nat,1n+Nat.add(cell,far),Nat.add(cell,1n+far),Nat.add(U32.to_nat(first),U32.to_nat(span)),
    Equal.sym(Nat,Nat.add(cell,1n+far),1n+Nat.add(cell,far),Addition.successor_right(cell,far)),
    Equal.trans(Nat,Nat.add(cell,1n+far),Nat.add(cell,Nat.sub(Nat.add(U32.to_nat(first),U32.to_nat(span)),cell)),Nat.add(U32.to_nat(first),U32.to_nat(span)),
      Equal.cong(Nat,Nat,tail => Nat.add(cell,tail),1n+far,Nat.sub(Nat.add(U32.to_nat(first),U32.to_nat(span)),cell),whole),
      Order.restore_shift(cell,Nat.add(U32.to_nat(first),U32.to_nat(span)),Bounds.strict_le(cell,Nat.add(U32.to_nat(first),U32.to_nat(span)),below))))
  {%sum : {Nat.is_lt(Nat.add(cell,far),_) == True{} : Bool} Order.lt_succ(Nat.add(cell,far),Nat.add(cell,far),Order.reflexive(Nat.add(cell,far))) :
    {Nat.is_lt(Nat.add(cell,far),Nat.add(U32.to_nat(first),U32.to_nat(span))) == True{} : Bool}}

# With a stride, a run copies the cells of its reach over the stride, at most
# its extent.
def wide_cells(+extent: U32,+offset: U32,+stride: U32,+first: U32,+span: U32,+capacity: U32,+holds: Held.Holds<first,span,offset,capacity>,
  +nonzero: {U32.is_zero(stride) == False{} : Bool}) -> {Move.inside(extent,offset,stride,False{},first,span) == U32.min(extent,((Held.step_reach(False{},offset,first,span) / U32.max(stride,1) : U32) + 1 : U32)) : U32}:
  Held.Holds{+after,+before,+inside} = holds
  Equal.trans(U32,Move.inside(extent,offset,stride,False{},first,span),Bool.pick(U32,U32.is_zero(stride),extent,U32.min(extent,((Held.step_reach(False{},offset,first,span) / U32.max(stride,1) : U32) + 1 : U32))),U32.min(extent,((Held.step_reach(False{},offset,first,span) / U32.max(stride,1) : U32) + 1 : U32)),
    Held.held_cells(extent,offset,stride,False{},first,span,after,before),
    {%Equal.sym(Bool,U32.is_zero(stride),False{},nonzero) : {Bool.pick(U32,_,extent,U32.min(extent,((Held.step_reach(False{},offset,first,span) / U32.max(stride,1) : U32) + 1 : U32))) == U32.min(extent,((Held.step_reach(False{},offset,first,span) / U32.max(stride,1) : U32) + 1 : U32)) : U32} {==} :
      {Bool.pick(U32,U32.is_zero(stride),extent,U32.min(extent,((Held.step_reach(False{},offset,first,span) / U32.max(stride,1) : U32) + 1 : U32))) == U32.min(extent,((Held.step_reach(False{},offset,first,span) / U32.max(stride,1) : U32) + 1 : U32)) : U32}})

# Without a stride, a run copies its extent.
def still_cells(+extent: U32,+offset: U32,+stride: U32,+first: U32,+span: U32,+capacity: U32,+holds: Held.Holds<first,span,offset,capacity>,
  +zero: {U32.is_zero(stride) == True{} : Bool}) -> {Move.inside(extent,offset,stride,False{},first,span) == extent : U32}:
  Held.Holds{+after,+before,+inside} = holds
  Equal.trans(U32,Move.inside(extent,offset,stride,False{},first,span),Bool.pick(U32,U32.is_zero(stride),extent,U32.min(extent,((Held.step_reach(False{},offset,first,span) / U32.max(stride,1) : U32) + 1 : U32))),extent,
    Held.held_cells(extent,offset,stride,False{},first,span,after,before),
    {%Equal.sym(Bool,U32.is_zero(stride),True{},zero) : {Bool.pick(U32,_,extent,U32.min(extent,((Held.step_reach(False{},offset,first,span) / U32.max(stride,1) : U32) + 1 : U32))) == extent : U32} {==} :
      {Bool.pick(U32,U32.is_zero(stride),extent,U32.min(extent,((Held.step_reach(False{},offset,first,span) / U32.max(stride,1) : U32) + 1 : U32))) == extent : U32}})

# One before a positive count lies below the count.
def before_count(+count: Nat,+positive: {Nat.is_lt(0n,count) == True{} : Bool}) -> {Nat.is_lt(Nat.sub(count,1n),count) == True{} : Bool}:
  {%Order.positive_has_last(count,positive) : {Nat.is_lt(Nat.sub(count,1n),_) == True{} : Bool}
    Order.lt_succ(Nat.sub(count,1n),Nat.sub(count,1n),Order.reflexive(Nat.sub(count,1n))) : {Nat.is_lt(Nat.sub(count,1n),count) == True{} : Bool}}

# A run of some cells that the piece holds ends in the piece: its last cell
# lies below the end of the piece.
def fit_reach(+extent: U32,+stride: U32,+back: Bool,+step: U32,+offset: U32,+first: U32,+span: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat(capacity)) == True{} : Bool},+word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},+fits: {Move.run_fits(Move.Axis{extent,stride,back,step},offset,first,span) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(U32.to_nat(offset),Nat.mul(Nat.sub(U32.to_nat(extent),1n),U32.to_nat(stride))),Nat.add(U32.to_nat(first),U32.to_nat(span))) == True{} : Bool}:
  +le = and_right(Bool.not(back),U32.is_le(extent,Move.inside(extent,offset,stride,False{},first,span)),fits)
  +holds = holds_of(extent,offset,stride,first,span,capacity,capacity_ok,room,word,word_ok,positive,le)
  +cell = U32.to_nat(offset)
  +last = Nat.sub(U32.to_nat(extent),1n)
  +far = reach_end(offset,first,span,capacity,capacity_ok,room,word,word_ok,holds)
  Held.by_cases({Nat.is_lt(Nat.add(cell,Nat.mul(last,U32.to_nat(stride))),Nat.add(U32.to_nat(first),U32.to_nat(span))) == True{} : Bool},U32.is_zero(stride),
    +zero => {%Equal.sym(Nat,U32.to_nat(stride),0n,Equal.cong(U32,Nat,U32.to_nat,stride,0,Arithmetic.word_zero(stride,zero))) :
        {Nat.is_lt(Nat.add(cell,Nat.mul(last,_)),Nat.add(U32.to_nat(first),U32.to_nat(span))) == True{} : Bool}
      {%Equal.sym(Nat,Nat.mul(last,0n),0n,RowMajor.multiply_zero_right(last)) : {Nat.is_lt(Nat.add(cell,_),Nat.add(U32.to_nat(first),U32.to_nat(span))) == True{} : Bool}
        {%Equal.sym(Nat,Nat.add(cell,0n),cell,Addition.zero_right(cell)) : {Nat.is_lt(_,Nat.add(U32.to_nat(first),U32.to_nat(span))) == True{} : Bool}
          Held.offset_below(offset,first,span,capacity,capacity_ok,holds) : {Nat.is_lt(Nat.add(cell,0n),Nat.add(U32.to_nat(first),U32.to_nat(span))) == True{} : Bool}} :
        {Nat.is_lt(Nat.add(cell,Nat.mul(last,0n)),Nat.add(U32.to_nat(first),U32.to_nat(span))) == True{} : Bool}} :
      {Nat.is_lt(Nat.add(cell,Nat.mul(last,U32.to_nat(stride))),Nat.add(U32.to_nat(first),U32.to_nat(span))) == True{} : Bool}},
    +nonzero =>
      Order.le_lt_trans(Nat.add(cell,Nat.mul(last,U32.to_nat(stride))),Nat.add(cell,U32.to_nat(Held.step_reach(False{},offset,first,span))),Nat.add(U32.to_nat(first),U32.to_nat(span)),
        Order.offset_le(cell,Nat.mul(last,U32.to_nat(stride)),U32.to_nat(Held.step_reach(False{},offset,first,span)),
          Held.reach_quotient(extent,Held.step_reach(False{},offset,first,span),stride,capacity,capacity_ok,word,word_ok,stride_room,nonzero,
            Held.reach_room(False{},offset,first,span,capacity,capacity_ok,holds),last,
            Order.lt_le(last,U32.to_nat(extent),U32.to_nat(U32.min(extent,((Held.step_reach(False{},offset,first,span) / U32.max(stride,1) : U32) + 1 : U32))),before_count(U32.to_nat(extent),positive),
              Bounds.as_nat_le(extent,U32.min(extent,((Held.step_reach(False{},offset,first,span) / U32.max(stride,1) : U32) + 1 : U32)),
                {%wide_cells(extent,offset,stride,first,span,capacity,holds,nonzero) : {U32.is_le(extent,_) == True{} : Bool} le :
                  {U32.is_le(extent,U32.min(extent,((Held.step_reach(False{},offset,first,span) / U32.max(stride,1) : U32) + 1 : U32))) == True{} : Bool}})))),
        far))

# A word no larger than zero.
def zero_least(+value: U32) -> {U32.is_le(0,value) == True{} : Bool}:
  U32Division.nat_le_native(0,value,Order.zero_le(U32.to_nat(value)))

# The reach of a cell no later is no shorter.
def reach_order(+low: U32,+high: U32,+first: U32,+span: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat(capacity)) == True{} : Bool},+word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+low_holds: Held.Holds<first,span,low,capacity>,+high_holds: Held.Holds<first,span,high,capacity>,
  +order: {Nat.is_le(U32.to_nat(low),U32.to_nat(high)) == True{} : Bool}) -> {Nat.is_le(U32.to_nat(Held.step_reach(False{},high,first,span)),U32.to_nat(Held.step_reach(False{},low,first,span))) == True{} : Bool}:
  +low_whole = Held.forward_reach(low,first,span,capacity,capacity_ok,low_holds)
  +high_whole = Held.forward_reach(high,first,span,capacity,capacity_ok,high_holds)
  {%Equal.sym(Nat,1n+U32.to_nat(Held.step_reach(False{},high,first,span)),Nat.sub(Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat(high)),high_whole) : {Nat.is_le(_,1n+U32.to_nat(Held.step_reach(False{},low,first,span))) == True{} : Bool}
    {%Equal.sym(Nat,1n+U32.to_nat(Held.step_reach(False{},low,first,span)),Nat.sub(Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat(low)),low_whole) : {Nat.is_le(Nat.sub(Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat(high)),_) == True{} : Bool}
      Order.sub_antitone(Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat(low),U32.to_nat(high),order) :
      {Nat.is_le(Nat.sub(Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat(high)),1n+U32.to_nat(Held.step_reach(False{},low,first,span))) == True{} : Bool}} :
    {Nat.is_le(1n+U32.to_nat(Held.step_reach(False{},high,first,span)),1n+U32.to_nat(Held.step_reach(False{},low,first,span))) == True{} : Bool}}

# With a stride, a run from a cell no later copies at least the cells the run
# from the later cell copies whole.
def wide_lower(+extent: U32,+stride: U32,+low: U32,+high: U32,+first: U32,+span: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat(capacity)) == True{} : Bool},+word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool},+low_holds: Held.Holds<first,span,low,capacity>,+high_holds: Held.Holds<first,span,high,capacity>,
  +order: {Nat.is_le(U32.to_nat(low),U32.to_nat(high)) == True{} : Bool},+nonzero: {U32.is_zero(stride) == False{} : Bool},
  +high_le: {U32.is_le(extent,U32.min(extent,((Held.step_reach(False{},high,first,span) / U32.max(stride,1) : U32) + 1 : U32))) == True{} : Bool}) -> {U32.is_le(extent,U32.min(extent,((Held.step_reach(False{},low,first,span) / U32.max(stride,1) : U32) + 1 : U32))) == True{} : Bool}:
  +positive = Arithmetic.word_nonzero_positive(stride,nonzero)
  +low_whole = Held.reach_whole(Held.step_reach(False{},low,first,span),stride,capacity,capacity_ok,word,word_ok,stride_room,nonzero,Held.reach_room(False{},low,first,span,capacity,capacity_ok,low_holds))
  +high_whole = Held.reach_whole(Held.step_reach(False{},high,first,span),stride,capacity,capacity_ok,word,word_ok,stride_room,nonzero,Held.reach_room(False{},high,first,span,capacity,capacity_ok,high_holds))
  +high_bound = Order.transitive(U32.to_nat(extent),U32.to_nat(U32.min(extent,((Held.step_reach(False{},high,first,span) / U32.max(stride,1) : U32) + 1 : U32))),U32.to_nat(((Held.step_reach(False{},high,first,span) / U32.max(stride,1) : U32) + 1 : U32)),Bounds.as_nat_le(extent,U32.min(extent,((Held.step_reach(False{},high,first,span) / U32.max(stride,1) : U32) + 1 : U32)),high_le),
    Bounds.as_nat_le(U32.min(extent,((Held.step_reach(False{},high,first,span) / U32.max(stride,1) : U32) + 1 : U32)),((Held.step_reach(False{},high,first,span) / U32.max(stride,1) : U32) + 1 : U32),Extrema.minimum_right(extent,((Held.step_reach(False{},high,first,span) / U32.max(stride,1) : U32) + 1 : U32))))
  +quotients = DivisionOrder.quotient_le(U32.to_nat(Held.step_reach(False{},high,first,span)),U32.to_nat(Held.step_reach(False{},low,first,span)),U32.to_nat(stride),positive,reach_order(low,high,first,span,capacity,capacity_ok,room,word,word_ok,low_holds,high_holds,order))
  +low_bound = Order.transitive(U32.to_nat(extent),U32.to_nat(((Held.step_reach(False{},high,first,span) / U32.max(stride,1) : U32) + 1 : U32)),U32.to_nat(((Held.step_reach(False{},low,first,span) / U32.max(stride,1) : U32) + 1 : U32)),high_bound,
    {%Equal.sym(Nat,U32.to_nat(((Held.step_reach(False{},high,first,span) / U32.max(stride,1) : U32) + 1 : U32)),1n+Nat.div(U32.to_nat(Held.step_reach(False{},high,first,span)),U32.to_nat(stride)),high_whole) : {Nat.is_le(_,U32.to_nat(((Held.step_reach(False{},low,first,span) / U32.max(stride,1) : U32) + 1 : U32))) == True{} : Bool}
      {%Equal.sym(Nat,U32.to_nat(((Held.step_reach(False{},low,first,span) / U32.max(stride,1) : U32) + 1 : U32)),1n+Nat.div(U32.to_nat(Held.step_reach(False{},low,first,span)),U32.to_nat(stride)),low_whole) : {Nat.is_le(1n+Nat.div(U32.to_nat(Held.step_reach(False{},high,first,span)),U32.to_nat(stride)),_) == True{} : Bool} quotients :
        {Nat.is_le(1n+Nat.div(U32.to_nat(Held.step_reach(False{},high,first,span)),U32.to_nat(stride)),U32.to_nat(((Held.step_reach(False{},low,first,span) / U32.max(stride,1) : U32) + 1 : U32))) == True{} : Bool}} :
      {Nat.is_le(U32.to_nat(((Held.step_reach(False{},high,first,span) / U32.max(stride,1) : U32) + 1 : U32)),U32.to_nat(((Held.step_reach(False{},low,first,span) / U32.max(stride,1) : U32) + 1 : U32))) == True{} : Bool}})
  U32Division.nat_le_native(extent,U32.min(extent,((Held.step_reach(False{},low,first,span) / U32.max(stride,1) : U32) + 1 : U32)),
    {%Equal.sym(Nat,U32.to_nat(U32.min(extent,((Held.step_reach(False{},low,first,span) / U32.max(stride,1) : U32) + 1 : U32))),Nat.min(U32.to_nat(extent),U32.to_nat(((Held.step_reach(False{},low,first,span) / U32.max(stride,1) : U32) + 1 : U32))),Extrema.minimum(extent,((Held.step_reach(False{},low,first,span) / U32.max(stride,1) : U32) + 1 : U32))) : {Nat.is_le(U32.to_nat(extent),_) == True{} : Bool}
      {%Equal.sym(Nat,Nat.min(U32.to_nat(extent),U32.to_nat(((Held.step_reach(False{},low,first,span) / U32.max(stride,1) : U32) + 1 : U32))),U32.to_nat(extent),NatExtrema.minimum_when_ordered(U32.to_nat(extent),U32.to_nat(((Held.step_reach(False{},low,first,span) / U32.max(stride,1) : U32) + 1 : U32)),low_bound)) : {Nat.is_le(U32.to_nat(extent),_) == True{} : Bool}
        Order.reflexive(U32.to_nat(extent)) : {Nat.is_le(U32.to_nat(extent),Nat.min(U32.to_nat(extent),U32.to_nat(((Held.step_reach(False{},low,first,span) / U32.max(stride,1) : U32) + 1 : U32)))) == True{} : Bool}} :
      {Nat.is_le(U32.to_nat(extent),U32.to_nat(U32.min(extent,((Held.step_reach(False{},low,first,span) / U32.max(stride,1) : U32) + 1 : U32)))) == True{} : Bool}})

# A run of some cells from a cell no later than a run the piece holds whole,
# and in the piece, is held whole too.
def some_lower(+extent: U32,+stride: U32,+low: U32,+high: U32,+first: U32,+span: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat(capacity)) == True{} : Bool},+word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},+lower: Lower<first,low,high>,
  +high_le: {U32.is_le(extent,Move.inside(extent,high,stride,False{},first,span)) == True{} : Bool}) -> {U32.is_le(extent,Move.inside(extent,low,stride,False{},first,span)) == True{} : Bool}:
  +order = lower_order(first,low,high,lower)
  +high_holds = holds_of(extent,high,stride,first,span,capacity,capacity_ok,room,word,word_ok,positive,high_le)
  +high_before = Bounds.as_nat_lt(high,(first + span : U32),holds_before(high,first,span,capacity,high_holds))
  +low_holds = {Held.Holds{lower_after(first,low,high,lower),
    U32Division.nat_lt_native(low,(first + span : U32),Order.le_lt_trans(U32.to_nat(low),U32.to_nat(high),U32.to_nat((first + span : U32)),order,high_before)),room} : Held.Holds<first,span,low,capacity>}
  Held.by_cases({U32.is_le(extent,Move.inside(extent,low,stride,False{},first,span)) == True{} : Bool},U32.is_zero(stride),
    +zero => {%Equal.sym(U32,Move.inside(extent,low,stride,False{},first,span),extent,still_cells(extent,low,stride,first,span,capacity,low_holds,zero)) : {U32.is_le(extent,_) == True{} : Bool}
      U32Division.nat_le_native(extent,extent,Order.reflexive(U32.to_nat(extent))) : {U32.is_le(extent,Move.inside(extent,low,stride,False{},first,span)) == True{} : Bool}},
    +nonzero => {%Equal.sym(U32,Move.inside(extent,low,stride,False{},first,span),U32.min(extent,((Held.step_reach(False{},low,first,span) / U32.max(stride,1) : U32) + 1 : U32)),wide_cells(extent,low,stride,first,span,capacity,low_holds,nonzero)) : {U32.is_le(extent,_) == True{} : Bool}
      wide_lower(extent,stride,low,high,first,span,capacity,capacity_ok,room,word,word_ok,stride_room,low_holds,high_holds,order,nonzero,
        {%wide_cells(extent,high,stride,first,span,capacity,high_holds,nonzero) : {U32.is_le(extent,_) == True{} : Bool} high_le :
          {U32.is_le(extent,U32.min(extent,((Held.step_reach(False{},high,first,span) / U32.max(stride,1) : U32) + 1 : U32))) == True{} : Bool}}) :
      {U32.is_le(extent,Move.inside(extent,low,stride,False{},first,span)) == True{} : Bool}})

# A run from a cell in the piece no later than a run the piece holds whole
# is held whole.
def fit_lower(+extent: U32,+stride: U32,+back: Bool,+step: U32,+low: U32,+high: U32,+first: U32,+span: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat(capacity)) == True{} : Bool},+word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool},+fits: {Move.run_fits(Move.Axis{extent,stride,back,step},high,first,span) == True{} : Bool},+lower: Lower<first,low,high>) ->
  {Move.run_fits(Move.Axis{extent,stride,back,step},low,first,span) == True{} : Bool}:
  +not_back = and_left(Bool.not(back),U32.is_le(extent,Move.inside(extent,high,stride,False{},first,span)),fits)
  +high_le = and_right(Bool.not(back),U32.is_le(extent,Move.inside(extent,high,stride,False{},first,span)),fits)
  %Equal.sym(Bool,Bool.not(back),True{},not_back) : {_ && U32.is_le(extent,Move.inside(extent,low,stride,False{},first,span)) == True{} : Bool}
  Held.by_cases({U32.is_le(extent,Move.inside(extent,low,stride,False{},first,span)) == True{} : Bool},U32.is_zero(extent),
    +zero => {%Equal.sym(U32,extent,0,Arithmetic.word_zero(extent,zero)) : {U32.is_le(_,Move.inside(_,low,stride,False{},first,span)) == True{} : Bool}
      zero_least(Move.inside(0,low,stride,False{},first,span)) : {U32.is_le(extent,Move.inside(extent,low,stride,False{},first,span)) == True{} : Bool}},
    +some => some_lower(extent,stride,low,high,first,span,capacity,capacity_ok,room,word,word_ok,stride_room,Arithmetic.word_nonzero_positive(extent,some),lower,high_le))

# ---- Every row of a part the piece holds ----

# Cells equal to cells the piece holds lie in the piece in the same order.
def lower_moved(+first: U32,+low: U32,+high: U32,+moved_low: U32,+moved_high: U32,+low_same: {low == moved_low : U32},+high_same: {high == moved_high : U32},
  +lower: Lower<first,low,high>) -> Lower<first,moved_low,moved_high>:
  %low_same : Lower<first,_,moved_high>
  %high_same : Lower<first,low,_>
  lower

# An axis of no index leaves both cells in place.
def still_lower(+extent: U32,+stride: U32,+low: U32,+high: U32,+index: U32,+first: U32,+zero: {U32.is_zero(extent) == True{} : Bool},+lower: Lower<first,low,high>,
  +index_in: {Nat.is_lt(U32.to_nat(index),U32.to_nat(U32.max(extent,1))) == True{} : Bool}) -> Lower<first,(low + index * stride : U32),(high + (U32.max(extent,1) - 1) * stride : U32)>:
  +empty = Arithmetic.word_zero(extent,zero)
  +index_one = {%empty : {Nat.is_lt(U32.to_nat(index),U32.to_nat(U32.max(_,1))) == True{} : Bool} index_in : {Nat.is_lt(U32.to_nat(index),1n) == True{} : Bool}}
  +index_zero = Equal.trans(U32,index,U32.from_nat(U32.to_nat(index)),0,Equal.sym(U32,U32.from_nat(U32.to_nat(index)),index,Arithmetic.native_roundtrip(index)),
    Equal.cong(Nat,U32,U32.from_nat,U32.to_nat(index),0n,zero_of_one(U32.to_nat(index),index_one)))
  +low_same = Equal.trans(U32,(low + index * stride : U32),(low + 0 * stride : U32),low,Equal.cong(U32,U32,word => (low + word * stride : U32),index,0,index_zero),zero_step(low,stride))
  +high_same = Equal.trans(U32,(high + (U32.max(extent,1) - 1) * stride : U32),(high + (U32.max(0,1) - 1) * stride : U32),high,
    Equal.cong(U32,U32,word => (high + (U32.max(word,1) - 1) * stride : U32),extent,0,empty),zero_step(high,stride))
  lower_moved(first,low,high,(low + index * stride : U32),(high + (U32.max(extent,1) - 1) * stride : U32),Equal.sym(U32,(low + index * stride : U32),low,low_same),Equal.sym(U32,(high + (U32.max(extent,1) - 1) * stride : U32),high,high_same),lower)

# A forward axis the piece holds from the high cell moves the low cell by an
# index of the axis to no further than the high cell moved across the axis.
def some_step(+extent: U32,+stride: U32,+step: U32,+low: U32,+high: U32,+index: U32,+first: U32,+span: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat(capacity)) == True{} : Bool},+word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+extent_room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(cap)) == True{} : Bool},+stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool},
  +nonzero: {U32.is_zero(extent) == False{} : Bool},+fits: {Move.run_fits(Move.Axis{extent,stride,False{},step},high,first,span) == True{} : Bool},+lower: Lower<first,low,high>,
  +index_in: {Nat.is_lt(U32.to_nat(index),U32.to_nat(U32.max(extent,1))) == True{} : Bool}) -> Lower<first,(low + index * stride : U32),(high + (U32.max(extent,1) - 1) * stride : U32)>:
  +positive = Arithmetic.word_nonzero_positive(extent,nonzero)
  +count = U32.to_nat(extent)
  +last = Nat.sub(count,1n)
  +gap = U32.to_nat(stride)
  +near = U32.to_nat(low)
  +far = U32.to_nat(high)
  +moved = U32.to_nat(index)
  +reach = fit_reach(extent,stride,False{},step,high,first,span,capacity,capacity_ok,room,word,word_ok,stride_room,positive,fits)
  +far_room = Order.transitive(Nat.add(far,Nat.mul(last,gap)),Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat(capacity),Bounds.strict_le(Nat.add(far,Nat.mul(last,gap)),Nat.add(U32.to_nat(first),U32.to_nat(span)),reach),room)
  +widest = Extrema.max_one_exact(extent,count,{==},positive)
  +last_exact = Arithmetic.difference_exact(cap,cap_ok,U32.max(extent,1),1,count,1n,widest,{==},Order.strict_follow(0n,count,positive),extent_room)
  +span_exact = Arithmetic.product_exact(capacity,capacity_ok,(U32.max(extent,1) - 1 : U32),stride,last,gap,last_exact,{==},
    Order.transitive(Nat.mul(last,gap),Nat.add(far,Nat.mul(last,gap)),U32.to_nat(capacity),Order.right_le(far,Nat.mul(last,gap)),far_room))
  +far_exact = Arithmetic.sum_exact(capacity,capacity_ok,high,((U32.max(extent,1) - 1 : U32) * stride : U32),far,Nat.mul(last,gap),{==},span_exact,far_room)
  +index_below = {%widest : {Nat.is_lt(moved,_) == True{} : Bool} index_in : {Nat.is_lt(moved,count) == True{} : Bool}}
  +index_last = Division.lt_succ_le(moved,last,
    {%Equal.sym(Nat,1n+last,count,Order.positive_has_last(count,positive)) : {Nat.is_lt(moved,_) == True{} : Bool} index_below : {Nat.is_lt(moved,1n+last) == True{} : Bool}})
  +ordered = Order.transitive(Nat.add(near,Nat.mul(moved,gap)),Nat.add(far,Nat.mul(moved,gap)),Nat.add(far,Nat.mul(last,gap)),
    Order.add_le(near,far,Nat.mul(moved,gap),lower_order(first,low,high,lower)),Order.offset_le(far,Nat.mul(moved,gap),Nat.mul(last,gap),RowMajor.mul_le(moved,last,gap,index_last)))
  +near_room = Order.transitive(Nat.add(near,Nat.mul(moved,gap)),Nat.add(far,Nat.mul(last,gap)),U32.to_nat(capacity),ordered,far_room)
  +step_exact = Arithmetic.product_exact(capacity,capacity_ok,index,stride,moved,gap,{==},{==},
    Order.transitive(Nat.mul(moved,gap),Nat.add(near,Nat.mul(moved,gap)),U32.to_nat(capacity),Order.right_le(near,Nat.mul(moved,gap)),near_room))
  +near_exact = Arithmetic.sum_exact(capacity,capacity_ok,low,(index * stride : U32),near,Nat.mul(moved,gap),{==},step_exact,near_room)
  +after = U32Division.nat_le_native(first,(low + index * stride : U32),
    {%Equal.sym(Nat,U32.to_nat((low + index * stride : U32)),Nat.add(near,Nat.mul(moved,gap)),near_exact) : {Nat.is_le(U32.to_nat(first),_) == True{} : Bool}
      Order.transitive(U32.to_nat(first),near,Nat.add(near,Nat.mul(moved,gap)),Bounds.as_nat_le(first,low,lower_after(first,low,high,lower)),Order.le_add(near,Nat.mul(moved,gap))) :
      {Nat.is_le(U32.to_nat(first),U32.to_nat((low + index * stride : U32))) == True{} : Bool}})
  +order = {%Equal.sym(Nat,U32.to_nat((low + index * stride : U32)),Nat.add(near,Nat.mul(moved,gap)),near_exact) : {Nat.is_le(_,U32.to_nat((high + (U32.max(extent,1) - 1) * stride : U32))) == True{} : Bool}
    {%Equal.sym(Nat,U32.to_nat((high + (U32.max(extent,1) - 1) * stride : U32)),Nat.add(far,Nat.mul(last,gap)),far_exact) : {Nat.is_le(Nat.add(near,Nat.mul(moved,gap)),_) == True{} : Bool} ordered :
      {Nat.is_le(Nat.add(near,Nat.mul(moved,gap)),U32.to_nat((high + (U32.max(extent,1) - 1) * stride : U32))) == True{} : Bool}} :
    {Nat.is_le(U32.to_nat((low + index * stride : U32)),U32.to_nat((high + (U32.max(extent,1) - 1) * stride : U32))) == True{} : Bool}}
  Lower{after,order}

# An axis the piece holds from the high cell moves the low cell by an index of
# the axis to no further than the high cell moved across the axis.
def step_lower(+extent: U32,+stride: U32,+back: Bool,+step: U32,+low: U32,+high: U32,+index: U32,+first: U32,+span: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat(capacity)) == True{} : Bool},+word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+extent_room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(cap)) == True{} : Bool},+stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool},
  +fits: {Move.run_fits(Move.Axis{extent,stride,back,step},high,first,span) == True{} : Bool},+lower: Lower<first,low,high>,
  +index_in: {Nat.is_lt(U32.to_nat(index),U32.to_nat(U32.max(extent,1))) == True{} : Bool}) -> Lower<first,(low + index * Move.directed(back,stride) : U32),(high + (U32.max(extent,1) - 1) * stride : U32)>:
  match back:
    case True{}:
      Empty.absurd(Lower<first,(low + index * Move.directed(True{},stride) : U32),(high + (U32.max(extent,1) - 1) * stride : U32)>,
        WordEncoding.false_true(and_left(Bool.not(True{}),U32.is_le(extent,Move.inside(extent,high,stride,False{},first,span)),fits)))
    case False{}:
      Held.by_cases(Lower<first,(low + index * stride : U32),(high + (U32.max(extent,1) - 1) * stride : U32)>,U32.is_zero(extent),
        +zero => still_lower(extent,stride,low,high,index,first,zero,lower,index_in),
        +some => some_step(extent,stride,step,low,high,index,first,span,capacity,capacity_ok,room,word,word_ok,cap,cap_ok,extent_room,stride_room,some,fits,lower,index_in))

# When the far corner of the axes fits the piece, the run of every row from a
# cell no later fits it too.
def flat_fits(outer: List<&2,Move.Axis>,+extent: U32,+stride: U32,+back: Bool,+step: U32,+low: U32,+high: U32,+target: U32,+first: U32,+span: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat(capacity)) == True{} : Bool},+word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room_total: {Nat.is_le(Nat.double(total(outer)),U32.to_nat(cap)) == True{} : Bool},+positive_total: {Nat.is_lt(0n,total(outer)) == True{} : Bool},+all_room: AllRoom(outer,word),+stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool},
  +lower: Lower<first,low,high>,+fits: {Move.fits_from(outer,Move.Axis{extent,stride,back,step},high,first,span) == True{} : Bool},+row: U32) ->
  {Move.run_fits(Move.Axis{extent,stride,back,step},offsets_source(Move.row_offsets(outer,row,low,target)),first,span) == True{} : Bool}:
  match outer:
    case Nil{}:
      fit_lower(extent,stride,back,step,low,high,first,span,capacity,capacity_ok,room,word,word_ok,stride_room,fits,lower)
    case Con{Move.Axis{+size,+gap,+reverse,+jump},+rest}:
      Held.Both{+own,+later} = all_room
      +here_fits = and_left(Move.run_fits(Move.Axis{size,gap,reverse,jump},high,first,span),Move.fits_from(rest,Move.Axis{extent,stride,back,step},(high + (U32.max(size,1) - 1) * gap : U32),first,span),fits)
      +rest_fits = and_right(Move.run_fits(Move.Axis{size,gap,reverse,jump},high,first,span),Move.fits_from(rest,Move.Axis{extent,stride,back,step},(high + (U32.max(size,1) - 1) * gap : U32),first,span),fits)
      +size_positive = RowMajor.left_factor_positive(U32.to_nat(size),total(rest),positive_total)
      +rest_positive = DivisionOrder.right_factor_positive(U32.to_nat(size),total(rest),positive_total)
      +size_room = double_room(U32.to_nat(size),Nat.mul(U32.to_nat(size),total(rest)),cap,RowMajor.product_covers(U32.to_nat(size),total(rest),rest_positive),room_total)
      +rest_room = double_room(total(rest),Nat.mul(U32.to_nat(size),total(rest)),cap,DivisionOrder.le_multiple(U32.to_nat(size),total(rest),size_positive),room_total)
      +widest = Extrema.max_one_exact(size,U32.to_nat(size),{==},size_positive)
      +blocks = (row / U32.max(Move.rows_of(rest),1) : U32)
      +index = (blocks % U32.max(size,1) : U32)
      +widest_positive = {%Equal.sym(Nat,U32.to_nat(U32.max(size,1)),U32.to_nat(size),widest) : {Nat.is_lt(0n,_) == True{} : Bool} size_positive :
        {Nat.is_lt(0n,U32.to_nat(U32.max(size,1))) == True{} : Bool}}
      +widest_room = {%Equal.sym(Nat,U32.to_nat(U32.max(size,1)),U32.to_nat(size),widest) : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool} size_room :
        {Nat.is_le(Nat.double(U32.to_nat(U32.max(size,1))),U32.to_nat(cap)) == True{} : Bool}}
      +index_exact = U32Division.modulo(blocks,U32.max(size,1),cap,cap_ok,widest_room,widest_positive)
      +index_in = {%Equal.sym(Nat,U32.to_nat(index),Nat.mod(U32.to_nat(blocks),U32.to_nat(U32.max(size,1))),index_exact) : {Nat.is_lt(_,U32.to_nat(U32.max(size,1))) == True{} : Bool}
        DivisionOrder.remainder_small(U32.to_nat(blocks),U32.to_nat(U32.max(size,1)),widest_positive) : {Nat.is_lt(U32.to_nat(index),U32.to_nat(U32.max(size,1))) == True{} : Bool}}
      +moved = step_lower(size,gap,reverse,jump,low,high,index,first,span,capacity,capacity_ok,room,word,word_ok,cap,cap_ok,single_room(U32.to_nat(size),cap,size_room),own,here_fits,lower,index_in)
      flat_fits(rest,extent,stride,back,step,(low + index * Move.directed(reverse,gap) : U32),(high + (U32.max(size,1) - 1) * gap : U32),(target + index * jump : U32),first,span,capacity,capacity_ok,room,word,word_ok,cap,cap_ok,
        rest_room,rest_positive,later,stride_room,moved,rest_fits,(row % U32.max(Move.rows_of(rest),1) : U32))

# The runs of the rows of a block from a cell on fit the piece.
def rows_fit(count: Nat,+cell: Nat,+q: Nat,+outer: List<&2,Move.Axis>,+extent: U32,+stride: U32,+back: Bool,+step: U32,+source: U32,+target: U32,+first: U32,+span: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat(capacity)) == True{} : Bool},+word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room_total: {Nat.is_le(Nat.double(total(outer)),U32.to_nat(cap)) == True{} : Bool},+positive_total: {Nat.is_lt(0n,total(outer)) == True{} : Bool},+all_room: AllRoom(outer,word),+stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool},+lower: Lower<first,source,source>,+fits: {Move.fits_from(outer,Move.Axis{extent,stride,back,step},source,first,span) == True{} : Bool},
  +q_in: {Nat.is_lt(q,total(nest_outer_of(outer))) == True{} : Bool},+within: {Nat.is_le(Nat.add(cell,count),U32.to_nat(axis_extent(nest_inner_of(outer)))) == True{} : Bool}) ->
  RowsFit(count,Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))),Move.Axis{extent,stride,back,step},(row_source(nest_outer_of(outer),q,source,target) + U32.from_nat(cell) * Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))) : U32),first,span):
  match count:
    case 0n: Unit{}
    case 1n+ +p:
      +index = Nat.add(Nat.mul(q,U32.to_nat(axis_extent(nest_inner_of(outer)))),cell)
      +cell_in = Order.room_index(cell,p,U32.to_nat(axis_extent(nest_inner_of(outer))),within)
      +below = RowMajor.radix(q,total(nest_outer_of(outer)),cell,U32.to_nat(axis_extent(nest_inner_of(outer))),q_in,cell_in)
      +inside = {%Equal.sym(Nat,total(outer),Nat.mul(total(nest_outer_of(outer)),U32.to_nat(axis_extent(nest_inner_of(outer)))),nest_total(outer)) : {Nat.is_lt(index,_) == True{} : Bool} below :
        {Nat.is_lt(index,total(outer)) == True{} : Bool}}
      +index_room = Order.transitive(index,total(outer),U32.to_nat(cap),Bounds.strict_le(index,total(outer),inside),single_room(total(outer),cap,room_total))
      +exact = Bounds.roundtrip(index,cap,cap_ok,index_room)
      +offsets = nest_offsets(outer,cap,cap_ok,room_total,U32.from_nat(index),q,cell,exact,q_in,cell_in,source,target)
      +here = {%offsets : {Move.run_fits(Move.Axis{extent,stride,back,step},offsets_source(_),first,span) == True{} : Bool}
        flat_fits(outer,extent,stride,back,step,source,source,target,first,span,capacity,capacity_ok,room,word,word_ok,cap,cap_ok,room_total,positive_total,all_room,stride_room,lower,fits,U32.from_nat(index)) :
        {Move.run_fits(Move.Axis{extent,stride,back,step},(row_source(nest_outer_of(outer),q,source,target) + U32.from_nat(cell) * Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))) : U32),first,span) == True{} : Bool}}
      +later = rows_fit(p,1n+cell,q,outer,extent,stride,back,step,source,target,first,span,capacity,capacity_ok,room,word,word_ok,cap,cap_ok,room_total,positive_total,all_room,stride_room,lower,fits,q_in,
        {%Addition.successor_right(cell,p) : {Nat.is_le(_,U32.to_nat(axis_extent(nest_inner_of(outer)))) == True{} : Bool} within : {Nat.is_le(Nat.add(1n+cell,p),U32.to_nat(axis_extent(nest_inner_of(outer)))) == True{} : Bool}})
      Held.Both{here,{%Equal.sym(U32,((row_source(nest_outer_of(outer),q,source,target) + U32.from_nat(cell) * Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))) : U32) + Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))) : U32),(row_source(nest_outer_of(outer),q,source,target) + U32.from_nat(1n+cell) * Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))) : U32),next_offset(row_source(nest_outer_of(outer),q,source,target),cell,Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))))) :
        RowsFit(p,Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))),Move.Axis{extent,stride,back,step},_,first,span) later : RowsFit(p,Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))),Move.Axis{extent,stride,back,step},((row_source(nest_outer_of(outer),q,source,target) + U32.from_nat(cell) * Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))) : U32) + Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))) : U32),first,span)}}

# The runs of the rows of blocks from a block on fit the piece.
def blocks_fit(count: Nat,+q: Nat,+outer: List<&2,Move.Axis>,+extent: U32,+stride: U32,+back: Bool,+step: U32,+source: U32,+target: U32,+first: U32,+span: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat(capacity)) == True{} : Bool},+word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room_total: {Nat.is_le(Nat.double(total(outer)),U32.to_nat(cap)) == True{} : Bool},+positive_total: {Nat.is_lt(0n,total(outer)) == True{} : Bool},+all_room: AllRoom(outer,word),+stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool},+lower: Lower<first,source,source>,+fits: {Move.fits_from(outer,Move.Axis{extent,stride,back,step},source,first,span) == True{} : Bool},
  +within: {Nat.is_le(Nat.add(q,count),total(nest_outer_of(outer))) == True{} : Bool}) ->
  BlocksFit(count,q,nest_outer_of(outer),nest_inner_of(outer),Move.Axis{extent,stride,back,step},source,target,first,span):
  match count:
    case 0n: Unit{}
    case 1n+ +p:
      +q_in = Order.room_index(q,p,total(nest_outer_of(outer)),within)
      +rows = rows_fit(U32.to_nat(axis_extent(nest_inner_of(outer))),0n,q,outer,extent,stride,back,step,source,target,first,span,capacity,capacity_ok,room,word,word_ok,cap,cap_ok,room_total,positive_total,all_room,stride_room,lower,fits,q_in,Order.reflexive(U32.to_nat(axis_extent(nest_inner_of(outer)))))
      Held.Both{{%zero_step(row_source(nest_outer_of(outer),q,source,target),Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer)))) : RowsFit(U32.to_nat(axis_extent(nest_inner_of(outer))),Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))),Move.Axis{extent,stride,back,step},_,first,span) rows : RowsFit(U32.to_nat(axis_extent(nest_inner_of(outer))),Move.directed(axis_back(nest_inner_of(outer)),axis_stride(nest_inner_of(outer))),Move.Axis{extent,stride,back,step},row_source(nest_outer_of(outer),q,source,target),first,span)},
        blocks_fit(p,1n+q,outer,extent,stride,back,step,source,target,first,span,capacity,capacity_ok,room,word,word_ok,cap,cap_ok,room_total,positive_total,all_room,stride_room,lower,fits,
          {%Addition.successor_right(q,p) : {Nat.is_le(_,total(nest_outer_of(outer))) == True{} : Bool} within : {Nat.is_le(Nat.add(1n+q,p),total(nest_outer_of(outer))) == True{} : Bool}})}

# When the far corner of the axes fits the piece, the runs of every row fit it.
def piece_fits(+outer: List<&2,Move.Axis>,+extent: U32,+stride: U32,+back: Bool,+step: U32,+source: U32,+target: U32,+first: U32,+span: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat(capacity)) == True{} : Bool},+word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room_total: {Nat.is_le(Nat.double(total(outer)),U32.to_nat(cap)) == True{} : Bool},+positive_total: {Nat.is_lt(0n,total(outer)) == True{} : Bool},+all_room: AllRoom(outer,word),+stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool},+lower: Lower<first,source,source>,+fits: {Move.fits_from(outer,Move.Axis{extent,stride,back,step},source,first,span) == True{} : Bool}) ->
  BlocksFit(U32.to_nat(Move.rows_of(nest_outer_of(outer))),0n,nest_outer_of(outer),nest_inner_of(outer),Move.Axis{extent,stride,back,step},source,target,first,span):
  +product = nest_total(outer)
  +nest_positive = {%product : {Nat.is_lt(0n,_) == True{} : Bool} positive_total : {Nat.is_lt(0n,Nat.mul(total(nest_outer_of(outer)),U32.to_nat(axis_extent(nest_inner_of(outer))))) == True{} : Bool}}
  +blocks_positive = RowMajor.left_factor_positive(total(nest_outer_of(outer)),U32.to_nat(axis_extent(nest_inner_of(outer))),nest_positive)
  +width_positive = DivisionOrder.right_factor_positive(total(nest_outer_of(outer)),U32.to_nat(axis_extent(nest_inner_of(outer))),nest_positive)
  +blocks_room = Order.transitive(total(nest_outer_of(outer)),total(outer),U32.to_nat(cap),
    {%Equal.sym(Nat,total(outer),Nat.mul(total(nest_outer_of(outer)),U32.to_nat(axis_extent(nest_inner_of(outer)))),product) : {Nat.is_le(total(nest_outer_of(outer)),_) == True{} : Bool}
      RowMajor.product_covers(total(nest_outer_of(outer)),U32.to_nat(axis_extent(nest_inner_of(outer))),width_positive) : {Nat.is_le(total(nest_outer_of(outer)),total(outer)) == True{} : Bool}},
    single_room(total(outer),cap,room_total))
  +blocks_exact = rows_exact(nest_outer_of(outer),cap,cap_ok,blocks_positive,blocks_room)
  blocks_fit(U32.to_nat(Move.rows_of(nest_outer_of(outer))),0n,outer,extent,stride,back,step,source,target,first,span,capacity,capacity_ok,room,word,word_ok,cap,cap_ok,room_total,positive_total,all_room,stride_room,lower,fits,
    {%Equal.sym(Nat,U32.to_nat(Move.rows_of(nest_outer_of(outer))),total(nest_outer_of(outer)),blocks_exact) : {Nat.is_le(_,total(nest_outer_of(outer))) == True{} : Bool}
      Order.reflexive(total(nest_outer_of(outer))) : {Nat.is_le(Nat.add(0n,U32.to_nat(Move.rows_of(nest_outer_of(outer)))),total(nest_outer_of(outer))) == True{} : Bool}})

# ---- A part as rows ----

# The rows of a part, every row one run, from its held pieces into a buffer
# tree.
def flat_ran(~Element: Data,+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>) -> Held.Ran<Element>:
  model_rows(~Element,U32.to_nat(Move.rows_of(part_outer(axes))),0n,part_outer(axes),axis_extent(part_run(axes)),axis_stride(part_run(axes)),axis_back(part_run(axes)),
    source,target,holding,tree)

# ---- The cells a part writes and the cells it leaves ----

# The target cell of box index index.
def cell_target(+axes: List<&2,Move.Axis>,+index: Nat,+source: U32,+target: U32) -> U32:
  offsets_target(Move.row_offsets(axes,U32.from_nat(index),source,target))

# Box indices from first on, count of them, whose target cells lie below
# capacity.
def CellsBelow(+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+capacity: U32,+first: Nat,count: Nat) -> Data:
  match count:
    case 0n: Unit
    case 1n+rest:
      Held.Both<{Nat.is_lt(U32.to_nat(cell_target(axes,first,source,target)),U32.to_nat(capacity)) == True{} : Bool},CellsBelow(axes,source,target,capacity,1n+first,rest)>

# Box indices from first on, count of them, whose target cells are not query.
def CellsMiss(+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+query: U32,+first: Nat,count: Nat) -> Data:
  match count:
    case 0n: Unit
    case 1n+rest:
      Held.Both<{U32.is_eq(cell_target(axes,first,source,target),query) == False{} : Bool},CellsMiss(axes,source,target,query,1n+first,rest)>

# Box indices from first on, count of them, whose target cell is the target
# cell of index only at index.
def CellsApart(+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+index: Nat,+first: Nat,count: Nat) -> Data:
  match count:
    case 0n: Unit
    case 1n+rest:
      Held.Both<{U32.is_eq(cell_target(axes,first,source,target),cell_target(axes,index,source,target)) == Nat.is_eq(first,index) : Bool},CellsApart(axes,source,target,index,1n+first,rest)>

# The first head cells of a run of head and tail cells.
def below_front(head: Nat,+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+capacity: U32,+first: Nat,+tail: Nat,
  +cells: CellsBelow(axes,source,target,capacity,first,Nat.add(head,tail))) -> CellsBelow(axes,source,target,capacity,first,head):
  match head:
    case 0n: Unit{}
    case 1n+ +more:
      Held.Both{+fact,+rest} = cells
      Held.Both{fact,below_front(more,axes,source,target,capacity,1n+first,tail,rest)}

# The last tail cells of a run of head and tail cells.
def below_back(head: Nat,+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+capacity: U32,+first: Nat,+tail: Nat,
  +cells: CellsBelow(axes,source,target,capacity,first,Nat.add(head,tail))) -> CellsBelow(axes,source,target,capacity,Nat.add(first,head),tail):
  match head:
    case 0n:
      %Equal.sym(Nat,Nat.add(first,0n),first,Addition.zero_right(first)) : CellsBelow(axes,source,target,capacity,_,tail)
      cells
    case 1n+ +more:
      Held.Both{+fact,+rest} = cells
      %Equal.sym(Nat,Nat.add(first,1n+more),1n+Nat.add(first,more),Addition.successor_right(first,more)) : CellsBelow(axes,source,target,capacity,_,tail)
      below_back(more,axes,source,target,capacity,1n+first,tail,rest)

# The first head cells of a run of head and tail cells.
def miss_front(head: Nat,+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+query: U32,+first: Nat,+tail: Nat,
  +cells: CellsMiss(axes,source,target,query,first,Nat.add(head,tail))) -> CellsMiss(axes,source,target,query,first,head):
  match head:
    case 0n: Unit{}
    case 1n+ +more:
      Held.Both{+fact,+rest} = cells
      Held.Both{fact,miss_front(more,axes,source,target,query,1n+first,tail,rest)}

# The last tail cells of a run of head and tail cells.
def miss_back(head: Nat,+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+query: U32,+first: Nat,+tail: Nat,
  +cells: CellsMiss(axes,source,target,query,first,Nat.add(head,tail))) -> CellsMiss(axes,source,target,query,Nat.add(first,head),tail):
  match head:
    case 0n:
      %Equal.sym(Nat,Nat.add(first,0n),first,Addition.zero_right(first)) : CellsMiss(axes,source,target,query,_,tail)
      cells
    case 1n+ +more:
      Held.Both{+fact,+rest} = cells
      %Equal.sym(Nat,Nat.add(first,1n+more),1n+Nat.add(first,more),Addition.successor_right(first,more)) : CellsMiss(axes,source,target,query,_,tail)
      miss_back(more,axes,source,target,query,1n+first,tail,rest)

def apart_back(head: Nat,+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+index: Nat,+first: Nat,+tail: Nat,
  +cells: CellsApart(axes,source,target,index,first,Nat.add(head,tail))) -> CellsApart(axes,source,target,index,Nat.add(first,head),tail):
  match head:
    case 0n:
      %Equal.sym(Nat,Nat.add(first,0n),first,Addition.zero_right(first)) : CellsApart(axes,source,target,index,_,tail)
      cells
    case 1n+ +more:
      Held.Both{+fact,+rest} = cells
      %Equal.sym(Nat,Nat.add(first,1n+more),1n+Nat.add(first,more),Addition.successor_right(first,more)) : CellsApart(axes,source,target,index,_,tail)
      apart_back(more,axes,source,target,index,1n+first,tail,rest)

# The first cell of a nonempty run of cells below capacity.
def below_first(count: Nat,+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+capacity: U32,+first: Nat,+positive: {Nat.is_lt(0n,count) == True{} : Bool},
  +cells: CellsBelow(axes,source,target,capacity,first,count)) -> {Nat.is_lt(U32.to_nat(cell_target(axes,first,source,target)),U32.to_nat(capacity)) == True{} : Bool}:
  match count:
    case 0n: Empty.absurd({Nat.is_lt(U32.to_nat(cell_target(axes,first,source,target)),U32.to_nat(capacity)) == True{} : Bool},Order.below_zero(0n,positive))
    case 1n+ +rest:
      Held.Both{+fact,+later} = cells
      fact

# Cell offset of a run of cells that are not query.
def miss_at(count: Nat,+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+query: U32,+first: Nat,offset: Nat,+inside: {Nat.is_lt(offset,count) == True{} : Bool},
  +cells: CellsMiss(axes,source,target,query,first,count)) -> {U32.is_eq(cell_target(axes,Nat.add(first,offset),source,target),query) == False{} : Bool}:
  match count offset:
    case 0n offset:
      Empty.absurd({U32.is_eq(cell_target(axes,Nat.add(first,offset),source,target),query) == False{} : Bool},Order.below_zero(offset,inside))
    case 1n+ +rest 0n:
      Held.Both{+fact,+later} = cells
      %Equal.sym(Nat,Nat.add(first,0n),first,Addition.zero_right(first)) : {U32.is_eq(cell_target(axes,_,source,target),query) == False{} : Bool}
      fact
    case 1n+ +rest 1n+ +step:
      Held.Both{+fact,+later} = cells
      %Equal.sym(Nat,Nat.add(first,1n+step),1n+Nat.add(first,step),Addition.successor_right(first,step)) :
        {U32.is_eq(cell_target(axes,_,source,target),query) == False{} : Bool}
      miss_at(rest,axes,source,target,query,1n+first,step,inside,later)

# Cells after index are not the target cell of index.
def apart_miss(count: Nat,+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+index: Nat,+first: Nat,+after: {Nat.is_lt(index,first) == True{} : Bool},
  +cells: CellsApart(axes,source,target,index,first,count)) -> CellsMiss(axes,source,target,cell_target(axes,index,source,target),first,count):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      Held.Both{+fact,+later} = cells
      +differ = Equal.trans(Bool,Nat.is_eq(first,index),Nat.is_eq(index,first),False{},Order.equal_symmetric(first,index),Order.less_not_equal(index,first,after))
      Held.Both{Equal.trans(Bool,U32.is_eq(cell_target(axes,first,source,target),cell_target(axes,index,source,target)),Nat.is_eq(first,index),False{},fact,differ),
        apart_miss(rest,axes,source,target,index,1n+first,Bounds.nat_trans(index,first,1n+first,after,Order.successor(first,first,Order.reflexive(first))),later)}

# The rows of a nonempty part.
def rows_count(+axes: List<&2,Move.Axis>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(total(axes)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,total(axes)) == True{} : Bool}) -> {U32.to_nat(Move.rows_of(part_outer(axes))) == total(part_outer(axes)) : Nat}:
  +outer = part_outer(axes)
  +width = part_width(axes)
  +product = part_total(axes,cap,cap_ok,positive,room)
  +product_positive = {%Equal.sym(Nat,Nat.mul(total(outer),width),total(axes),product) : {Nat.is_lt(0n,_) == True{} : Bool} positive :
    {Nat.is_lt(0n,Nat.mul(total(outer),width)) == True{} : Bool}}
  +outer_positive = RowMajor.left_factor_positive(total(outer),width,product_positive)
  +width_positive = DivisionOrder.right_factor_positive(total(outer),width,product_positive)
  +outer_le = {%product : {Nat.is_le(total(outer),_) == True{} : Bool} RowMajor.product_covers(total(outer),width,width_positive) :
    {Nat.is_le(total(outer),total(axes)) == True{} : Bool}}
  rows_exact(outer,cap,cap_ok,outer_positive,Order.transitive(total(outer),total(axes),U32.to_nat(cap),outer_le,single_room(total(axes),cap,room)))

# The run of a nonempty part is not empty.
def run_positive(+axes: List<&2,Move.Axis>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(total(axes)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,total(axes)) == True{} : Bool}) -> {Nat.is_lt(0n,part_width(axes)) == True{} : Bool}:
  +product = part_total(axes,cap,cap_ok,positive,room)
  DivisionOrder.right_factor_positive(total(part_outer(axes)),part_width(axes),
    {%Equal.sym(Nat,Nat.mul(total(part_outer(axes)),part_width(axes)),total(axes),product) : {Nat.is_lt(0n,_) == True{} : Bool} positive :
      {Nat.is_lt(0n,Nat.mul(total(part_outer(axes)),part_width(axes))) == True{} : Bool}})

# Cell cell of row row of a part is box index row times the run plus cell.
def row_cell(+axes: List<&2,Move.Axis>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(total(axes)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,total(axes)) == True{} : Bool},+row: Nat,+cell: Nat,+row_in: {Nat.is_lt(row,total(part_outer(axes))) == True{} : Bool},+cell_in: {Nat.is_lt(cell,part_width(axes)) == True{} : Bool},+source: U32,+target: U32) ->
  {cell_target(axes,Nat.add(Nat.mul(row,part_width(axes)),cell),source,target) == (row_target(part_outer(axes),row,source,target) + U32.from_nat(cell) : U32) : U32}:
  +position = row_target(part_outer(axes),row,source,target)
  %Equal.cong(U32,U32,word => (position + word : U32),(U32.from_nat(cell) * 1 : U32),U32.from_nat(cell),Arithmetic.multiply_one(U32.from_nat(cell))) :
    {cell_target(axes,Nat.add(Nat.mul(row,part_width(axes)),cell),source,target) == _ : U32}
  %run_step(Move.coalesced(axes)) : {cell_target(axes,Nat.add(Nat.mul(row,part_width(axes)),cell),source,target) == (position + U32.from_nat(cell) * _ : U32) : U32}
  Equal.cong(Move.Offsets,U32,offsets_target,Move.row_offsets(axes,U32.from_nat(Nat.add(Nat.mul(row,part_width(axes)),cell)),source,target),
    Move.Offsets{(row_source(part_outer(axes),row,source,target) + U32.from_nat(cell) * part_step(axes) : U32),(position + U32.from_nat(cell) * axis_step(part_run(axes)) : U32)},
    box_offsets(axes,cap,cap_ok,room,positive,row,cell,row_in,cell_in,source,target))

# A row of a part whose cells from done on lie below capacity fits it.
def row_fits_from(remaining: Nat,+done: Nat,+axes: List<&2,Move.Axis>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(total(axes)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,total(axes)) == True{} : Bool},+row: Nat,+row_in: {Nat.is_lt(row,total(part_outer(axes))) == True{} : Bool},+source: U32,+target: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +within: {Nat.is_le(Nat.add(done,remaining),part_width(axes)) == True{} : Bool},
  +start: {Nat.is_le(Nat.add(U32.to_nat(row_target(part_outer(axes),row,source,target)),done),U32.to_nat(capacity)) == True{} : Bool},
  +cells: CellsBelow(axes,source,target,capacity,Nat.add(Nat.mul(row,part_width(axes)),done),remaining)) ->
  {Nat.is_le(Nat.add(U32.to_nat(row_target(part_outer(axes),row,source,target)),Nat.add(done,remaining)),U32.to_nat(capacity)) == True{} : Bool}:
  match remaining:
    case 0n:
      %Equal.sym(Nat,Nat.add(done,0n),done,Addition.zero_right(done)) : {Nat.is_le(Nat.add(U32.to_nat(row_target(part_outer(axes),row,source,target)),_),U32.to_nat(capacity)) == True{} : Bool}
      start
    case 1n+ +more:
      Held.Both{+head,+tail} = cells
      +position = row_target(part_outer(axes),row,source,target)
      +done_in = Order.room_index(done,more,part_width(axes),within)
      +done_room = Order.transitive(done,Nat.add(U32.to_nat(position),done),U32.to_nat(capacity),
        {%Order.commutative(done,U32.to_nat(position)) : {Nat.is_le(done,_) == True{} : Bool} Order.le_add(done,U32.to_nat(position)) :
          {Nat.is_le(done,Nat.add(U32.to_nat(position),done)) == True{} : Bool}},start)
      +exact = Arithmetic.sum_exact(capacity,capacity_ok,position,U32.from_nat(done),U32.to_nat(position),done,{==},Bounds.roundtrip(done,capacity,capacity_ok,done_room),start)
      +below = {%row_cell(axes,cap,cap_ok,room,positive,row,done,row_in,done_in,source,target) : {Nat.is_lt(U32.to_nat(_),U32.to_nat(capacity)) == True{} : Bool} head :
        {Nat.is_lt(U32.to_nat((position + U32.from_nat(done) : U32)),U32.to_nat(capacity)) == True{} : Bool}}
      +sum_below = {%exact : {Nat.is_lt(_,U32.to_nat(capacity)) == True{} : Bool} below : {Nat.is_lt(Nat.add(U32.to_nat(position),done),U32.to_nat(capacity)) == True{} : Bool}}
      +next = {%Equal.sym(Nat,Nat.add(U32.to_nat(position),1n+done),1n+Nat.add(U32.to_nat(position),done),Addition.successor_right(U32.to_nat(position),done)) :
          {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
        Order.strict_follow(Nat.add(U32.to_nat(position),done),U32.to_nat(capacity),sum_below) :
        {Nat.is_le(Nat.add(U32.to_nat(position),1n+done),U32.to_nat(capacity)) == True{} : Bool}}
      +shift = Equal.sym(Nat,Nat.add(done,1n+more),1n+Nat.add(done,more),Addition.successor_right(done,more))
      +result = row_fits_from(more,1n+done,axes,cap,cap_ok,room,positive,row,row_in,source,target,capacity,capacity_ok,
        {%Addition.successor_right(done,more) : {Nat.is_le(_,part_width(axes)) == True{} : Bool} within : {Nat.is_le(Nat.add(1n+done,more),part_width(axes)) == True{} : Bool}},next,
        {%Equal.sym(Nat,Nat.add(Nat.mul(row,part_width(axes)),1n+done),1n+Nat.add(Nat.mul(row,part_width(axes)),done),Addition.successor_right(Nat.mul(row,part_width(axes)),done)) :
          CellsBelow(axes,source,target,capacity,_,more) tail : CellsBelow(axes,source,target,capacity,Nat.add(Nat.mul(row,part_width(axes)),1n+done),more)})
      {%shift : {Nat.is_le(Nat.add(U32.to_nat(position),_),U32.to_nat(capacity)) == True{} : Bool} result :
        {Nat.is_le(Nat.add(U32.to_nat(position),Nat.add(done,1n+more)),U32.to_nat(capacity)) == True{} : Bool}}

# A row of a part whose cells lie below capacity fits it.
def row_fits(+axes: List<&2,Move.Axis>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(total(axes)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,total(axes)) == True{} : Bool},+row: Nat,+row_in: {Nat.is_lt(row,total(part_outer(axes))) == True{} : Bool},+source: U32,+target: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +cells: CellsBelow(axes,source,target,capacity,Nat.mul(row,part_width(axes)),part_width(axes))) ->
  {Nat.is_le(Nat.add(U32.to_nat(row_target(part_outer(axes),row,source,target)),part_width(axes)),U32.to_nat(capacity)) == True{} : Bool}:
  +width = part_width(axes)
  +position = row_target(part_outer(axes),row,source,target)
  +width_positive = run_positive(axes,cap,cap_ok,room,positive)
  +first = below_first(width,axes,source,target,capacity,Nat.mul(row,width),width_positive,cells)
  +zero_cell = Equal.trans(U32,cell_target(axes,Nat.mul(row,width),source,target),(position + U32.from_nat(0n) : U32),position,
    {%Addition.zero_right(Nat.mul(row,width)) : {cell_target(axes,_,source,target) == (position + U32.from_nat(0n) : U32) : U32}
      row_cell(axes,cap,cap_ok,room,positive,row,0n,row_in,width_positive,source,target) :
      {cell_target(axes,Nat.mul(row,width),source,target) == (position + U32.from_nat(0n) : U32) : U32}},
    Arithmetic.add_zero(position))
  +start = {%Equal.sym(Nat,Nat.add(U32.to_nat(position),0n),U32.to_nat(position),Addition.zero_right(U32.to_nat(position))) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    Bounds.strict_le(U32.to_nat(position),U32.to_nat(capacity),
      {%zero_cell : {Nat.is_lt(U32.to_nat(_),U32.to_nat(capacity)) == True{} : Bool} first : {Nat.is_lt(U32.to_nat(position),U32.to_nat(capacity)) == True{} : Bool}}) :
    {Nat.is_le(Nat.add(U32.to_nat(position),0n),U32.to_nat(capacity)) == True{} : Bool}}
  row_fits_from(width,0n,axes,cap,cap_ok,room,positive,row,row_in,source,target,capacity,capacity_ok,Order.reflexive(width),start,
    {%Equal.sym(Nat,Nat.add(Nat.mul(row,width),0n),Nat.mul(row,width),Addition.zero_right(Nat.mul(row,width))) : CellsBelow(axes,source,target,capacity,_,width) cells :
      CellsBelow(axes,source,target,capacity,Nat.add(Nat.mul(row,width),0n),width)})

def outside_cases(before: Bool,after: Bool,+axes: List<&2,Move.Axis>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(total(axes)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,total(axes)) == True{} : Bool},+row: Nat,+row_in: {Nat.is_lt(row,total(part_outer(axes))) == True{} : Bool},+source: U32,+target: U32,+query: U32,
  +found_before: {Nat.is_lt(U32.to_nat(query),U32.to_nat(row_target(part_outer(axes),row,source,target))) == before : Bool},
  +found_after: {Nat.is_le(Nat.add(U32.to_nat(row_target(part_outer(axes),row,source,target)),part_width(axes)),U32.to_nat(query)) == after : Bool},
  +miss: CellsMiss(axes,source,target,query,Nat.mul(row,part_width(axes)),part_width(axes))) -> Interval.Outside<U32.to_nat(row_target(part_outer(axes),row,source,target)),part_width(axes),U32.to_nat(query)>:
  match before after:
    case True{} after: Interval.Before{found_before}
    case False{} True{}: Interval.After{found_after}
    case False{} False{}:
      +position = row_target(part_outer(axes),row,source,target)
      +start = U32.to_nat(position)
      +width = part_width(axes)
      +reached = Order.not_less(U32.to_nat(query),start,found_before)
      +short = Held.le_false_lt(Nat.add(start,width),U32.to_nat(query),found_after)
      +cell = Nat.sub(U32.to_nat(query),start)
      +cell_in = Order.sub_lt(U32.to_nat(query),start,width,reached,short)
      +restored = Order.restore_shift(start,U32.to_nat(query),reached)
      +index = Nat.add(Nat.mul(row,width),cell)
      +same = Equal.trans(U32,cell_target(axes,index,source,target),(position + U32.from_nat(cell) : U32),query,
        row_cell(axes,cap,cap_ok,room,positive,row,cell,row_in,cell_in,source,target),
        Equal.trans(U32,(position + U32.from_nat(cell) : U32),U32.from_nat(Nat.add(start,cell)),query,
          {%Arithmetic.native_roundtrip(position) : {(_ + U32.from_nat(cell) : U32) == U32.from_nat(Nat.add(start,cell)) : U32}
            WordArithmetic.native_add(start,cell) : {(position + U32.from_nat(cell) : U32) == U32.from_nat(Nat.add(start,cell)) : U32}},
          {%Equal.sym(Nat,Nat.add(start,cell),U32.to_nat(query),restored) : {U32.from_nat(_) == query : U32}
            Arithmetic.native_roundtrip(query) : {U32.from_nat(Nat.add(start,cell)) == query : U32}}))
      +fact = {%same : {U32.is_eq(_,query) == False{} : Bool} miss_at(width,axes,source,target,query,Nat.mul(row,width),cell,cell_in,miss) :
        {U32.is_eq(query,query) == False{} : Bool}}
      +truth = Equal.trans(Bool,U32.is_eq(query,query),Nat.is_eq(U32.to_nat(query),U32.to_nat(query)),True{},Comparison.eq(query,query),Order.equal_reflexive(U32.to_nat(query)))
      Empty.absurd(Interval.Outside<start,width,U32.to_nat(query)>,
        WordEncoding.false_true(Equal.trans(Bool,False{},U32.is_eq(query,query),True{},Equal.sym(Bool,U32.is_eq(query,query),False{},fact),truth)))

# A row leaves alone a cell that none of its cells is.
def row_outside(+axes: List<&2,Move.Axis>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(total(axes)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,total(axes)) == True{} : Bool},+row: Nat,+row_in: {Nat.is_lt(row,total(part_outer(axes))) == True{} : Bool},+source: U32,+target: U32,+query: U32,
  +miss: CellsMiss(axes,source,target,query,Nat.mul(row,part_width(axes)),part_width(axes))) -> Interval.Outside<U32.to_nat(row_target(part_outer(axes),row,source,target)),part_width(axes),U32.to_nat(query)>:
  +position = row_target(part_outer(axes),row,source,target)
  outside_cases(Nat.is_lt(U32.to_nat(query),U32.to_nat(position)),Nat.is_le(Nat.add(U32.to_nat(position),part_width(axes)),U32.to_nat(query)),axes,cap,cap_ok,room,positive,row,row_in,source,target,query,
    {==},{==},miss)

# Every row of a part from row on fits a target whose cells lie below
# capacity, and leaves alone a cell that none of their cells is.
def rows_miss_of(fuel: Nat,+row: Nat,+axes: List<&2,Move.Axis>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(total(axes)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,total(axes)) == True{} : Bool},+source: U32,+target: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +left: {Nat.add(row,fuel) == total(part_outer(axes)) : Nat},
  +below: CellsBelow(axes,source,target,capacity,Nat.mul(row,part_width(axes)),Nat.mul(fuel,part_width(axes))),
  +query: U32,+miss: CellsMiss(axes,source,target,query,Nat.mul(row,part_width(axes)),Nat.mul(fuel,part_width(axes)))) ->
  RowsMiss(fuel,row,part_outer(axes),axis_extent(part_run(axes)),source,target,capacity,U32.to_nat(query)):
  match fuel:
    case 0n: Unit{}
    case 1n+ +rest:
      +width = part_width(axes)
      +row_in = Order.below_complement(row,rest,total(part_outer(axes)),left)
      +next = Order.commutative(Nat.mul(row,width),width)
      +own_below = below_front(width,axes,source,target,capacity,Nat.mul(row,width),Nat.mul(rest,width),below)
      +later_below = below_back(width,axes,source,target,capacity,Nat.mul(row,width),Nat.mul(rest,width),below)
      +own_miss = miss_front(width,axes,source,target,query,Nat.mul(row,width),Nat.mul(rest,width),miss)
      +later_miss = miss_back(width,axes,source,target,query,Nat.mul(row,width),Nat.mul(rest,width),miss)
      Held.Both{row_fits(axes,cap,cap_ok,room,positive,row,row_in,source,target,capacity,capacity_ok,own_below),
        Held.Both{row_outside(axes,cap,cap_ok,room,positive,row,row_in,source,target,query,own_miss),
          rows_miss_of(rest,1n+row,axes,cap,cap_ok,room,positive,source,target,capacity,capacity_ok,
            Equal.trans(Nat,Nat.add(1n+row,rest),Nat.add(row,1n+rest),total(part_outer(axes)),
              Equal.sym(Nat,Nat.add(row,1n+rest),1n+Nat.add(row,rest),Addition.successor_right(row,rest)),left),
            {%next : CellsBelow(axes,source,target,capacity,_,Nat.mul(rest,width)) later_below : CellsBelow(axes,source,target,capacity,Nat.mul(1n+row,width),Nat.mul(rest,width))},
            query,
            {%next : CellsMiss(axes,source,target,query,_,Nat.mul(rest,width)) later_miss : CellsMiss(axes,source,target,query,Nat.mul(1n+row,width),Nat.mul(rest,width))})}}

# A part leaves every cell of the target that none of its cells is.
def flat_kept(~Element: Data,+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>,+depth: Nat,
  +balanced: Shape.Balanced(Element,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity_ok: {U32.is_le(Tree.capacity(Element,tree),2147483648) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(total(axes)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,total(axes)) == True{} : Bool},
  +below: CellsBelow(axes,source,target,Tree.capacity(Element,tree),0n,total(axes)),
  +query: U32,+query_inside: {U32.is_lt(query,Tree.capacity(Element,tree)) == True{} : Bool},+miss: CellsMiss(axes,source,target,query,0n,total(axes))) ->
  {Tree.read(Element,Held.ran_tree(~Element,flat_ran(~Element,axes,source,target,holding,tree)),query) == Tree.read(Element,tree,query) : Element}:
  +outer = part_outer(axes)
  +run = part_run(axes)
  +width = part_width(axes)
  +capacity = Tree.capacity(Element,tree)
  +product = part_total(axes,cap,cap_ok,positive,room)
  +back = Equal.sym(Nat,Nat.mul(total(outer),width),total(axes),product)
  %Equal.sym(Nat,U32.to_nat(Move.rows_of(outer)),total(outer),rows_count(axes,cap,cap_ok,room,positive)) :
    {Tree.read(Element,Held.ran_tree(~Element,model_rows(~Element,_,0n,outer,axis_extent(run),axis_stride(run),axis_back(run),source,target,holding,tree)),query) ==
      Tree.read(Element,tree,query) : Element}
  rows_missed(~Element,total(outer),0n,outer,axis_extent(run),axis_stride(run),axis_back(run),source,target,holding,tree,depth,balanced,depth_ok,capacity,{==},capacity_ok,
    query,query_inside,
    rows_miss_of(total(outer),0n,axes,cap,cap_ok,room,positive,source,target,capacity,capacity_ok,{==},
      {%back : CellsBelow(axes,source,target,capacity,0n,_) below : CellsBelow(axes,source,target,capacity,0n,Nat.mul(total(outer),width))},query,
      {%back : CellsMiss(axes,source,target,query,0n,_) miss : CellsMiss(axes,source,target,query,0n,Nat.mul(total(outer),width))}))

# A part from held source pieces writes target cell T(index) of every box
# index with the value of source cell S(index), when no other box index has
# that target cell.
def flat_written(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+axes: List<&2,Move.Axis>,+source: U32,+target: U32,
  +holding: Held.Holding<Element>,+tree: Tree.Tree<Element>,+depth: Nat,+balanced: Shape.Balanced(Element,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity_ok: {U32.is_le(Tree.capacity(Element,tree),2147483648) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(total(axes)),U32.to_nat(cap)) == True{} : Bool},
  +word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+strides: StridesRoom(axes,word),
  +stack: Held.Stack<Element>,+ordered: {Held.order(~Element,holding) == stack : Held.Stack<Element>},+start: Nat,+end: Nat,+source_capacity: U32,
  +source_ok: {U32.is_le(source_capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(end,U32.to_nat(source_capacity)) == True{} : Bool},
  +sorted: Held.Sorted(~Element,stack,start,end),+values: Held.HoldingValues(~Element,~Context,~value,context,holding),
  +cells: CellsHeld(axes,source,target,Held.spans(~Element,stack),0n,total(axes)),
  +below: CellsBelow(axes,source,target,Tree.capacity(Element,tree),0n,total(axes)),
  +index: Nat,+inside: {Nat.is_lt(index,total(axes)) == True{} : Bool},+apart: CellsApart(axes,source,target,index,0n,total(axes))) ->
  {Tree.read(Element,Held.ran_tree(~Element,flat_ran(~Element,axes,source,target,holding,tree)),cell_target(axes,index,source,target)) ==
    value(context,cell_source(axes,index,source,target)) : Element}:
  +positive = Order.positive_bound(index,total(axes),inside)
  +outer = part_outer(axes)
  +run = part_run(axes)
  +width = part_width(axes)
  +capacity = Tree.capacity(Element,tree)
  +ranges = Held.spans(~Element,stack)
  +product = part_total(axes,cap,cap_ok,positive,room)
  +width_positive = run_positive(axes,cap,cap_ok,room,positive)
  +row = Nat.div(index,width)
  +cell = Nat.mod(index,width)
  +inner = Nat.add(Nat.mul(row,width),cell)
  +index_exact = DivisionOrder.quotient_exact(index,width,width_positive)
  +row_in = DivisionOrder.quotient_below(index,total(outer),width,width_positive,
    {%Equal.sym(Nat,Nat.mul(total(outer),width),total(axes),product) : {Nat.is_lt(index,_) == True{} : Bool} inside : {Nat.is_lt(index,Nat.mul(total(outer),width)) == True{} : Bool}})
  +cell_in = DivisionOrder.remainder_small(index,width,width_positive)
  +position = row_target(outer,row,source,target)
  +later = Nat.sub(total(outer),1n+row)
  +split = Equal.trans(Nat,Nat.add(row,1n+later),1n+Nat.add(row,later),total(outer),Addition.successor_right(row,later),
    Order.restore_shift(1n+row,total(outer),Order.strict_follow(row,total(outer),row_in)))
  +rest_cells = Nat.add(width,Nat.mul(later,width))
  +box_total = Equal.trans(Nat,total(axes),Nat.mul(total(outer),width),Nat.add(Nat.mul(row,width),rest_cells),
    Equal.sym(Nat,Nat.mul(total(outer),width),total(axes),product),
    {%split : {Nat.mul(_,width) == Nat.add(Nat.mul(row,width),rest_cells) : Nat} RowMajor.distrib(row,1n+later,width) :
      {Nat.mul(total(outer),width) == Nat.add(Nat.mul(row,width),rest_cells) : Nat}})
  +held_from = cells_drop(axes,source,target,ranges,Nat.mul(row,width),0n,rest_cells,
    {%box_total : CellsHeld(axes,source,target,ranges,0n,_) cells : CellsHeld(axes,source,target,ranges,0n,Nat.add(Nat.mul(row,width),rest_cells))})
  +run_cells_held = run_cells(axes,cap,cap_ok,room,positive,source,target,ranges,row,row_in,width,0n,Nat.mul(later,width),Order.reflexive(width),
    {%Equal.sym(Nat,Nat.add(Nat.mul(row,width),0n),Nat.mul(row,width),Addition.zero_right(Nat.mul(row,width))) : CellsHeld(axes,source,target,ranges,_,rest_cells) held_from :
      CellsHeld(axes,source,target,ranges,Nat.add(Nat.mul(row,width),0n),rest_cells)})
  +run_held = {%zero_step(row_source(outer,row,source,target),part_step(axes)) : Held.RunHeld(ranges,_,part_step(axes),width) run_cells_held :
    Held.RunHeld(ranges,row_source(outer,row,source,target),part_step(axes),width)}
  +below_all = {%box_total : CellsBelow(axes,source,target,capacity,0n,_) below : CellsBelow(axes,source,target,capacity,0n,Nat.add(Nat.mul(row,width),rest_cells))}
  +from_below = below_back(Nat.mul(row,width),axes,source,target,capacity,0n,rest_cells,below_all)
  +own_below = below_front(width,axes,source,target,capacity,Nat.mul(row,width),Nat.mul(later,width),from_below)
  +after_below = below_back(width,axes,source,target,capacity,Nat.mul(row,width),Nat.mul(later,width),from_below)
  +apart_all = {%box_total : CellsApart(axes,source,target,index,0n,_) apart : CellsApart(axes,source,target,index,0n,Nat.add(Nat.mul(row,width),rest_cells))}
  +from_apart = apart_back(Nat.mul(row,width),axes,source,target,index,0n,rest_cells,apart_all)
  +after_apart = apart_back(width,axes,source,target,index,Nat.mul(row,width),Nat.mul(later,width),from_apart)
  +fits = row_fits(axes,cap,cap_ok,room,positive,row,row_in,source,target,capacity,capacity_ok,own_below)
  +next = Order.commutative(Nat.mul(row,width),width)
  +index_before = {%Equal.sym(Nat,index,inner,index_exact) : {Nat.is_lt(_,Nat.add(Nat.mul(row,width),width)) == True{} : Bool}
    {%Equal.sym(Nat,Nat.add(Nat.mul(row,width),width),Nat.mul(1n+row,width),next) : {Nat.is_lt(inner,_) == True{} : Bool}
      RowMajor.radix(row,1n+row,cell,width,Order.lt_succ(row,row,Order.reflexive(row)),cell_in) : {Nat.is_lt(inner,Nat.add(Nat.mul(row,width),width)) == True{} : Bool}} :
    {Nat.is_lt(index,Nat.add(Nat.mul(row,width),width)) == True{} : Bool}}
  +after_miss = apart_miss(Nat.mul(later,width),axes,source,target,index,Nat.add(Nat.mul(row,width),width),index_before,after_apart)
  +cell_n = Nat.add(U32.to_nat(position),cell)
  +query_exact = Equal.trans(U32,cell_target(axes,index,source,target),cell_target(axes,inner,source,target),U32.from_nat(cell_n),
    Equal.cong(Nat,U32,place => cell_target(axes,place,source,target),index,inner,index_exact),
    Equal.trans(U32,cell_target(axes,inner,source,target),(position + U32.from_nat(cell) : U32),U32.from_nat(cell_n),
      row_cell(axes,cap,cap_ok,room,positive,row,cell,row_in,cell_in,source,target),
      {%Arithmetic.native_roundtrip(position) : {(_ + U32.from_nat(cell) : U32) == U32.from_nat(cell_n) : U32}
        WordArithmetic.native_add(U32.to_nat(position),cell) : {(position + U32.from_nat(cell) : U32) == U32.from_nat(cell_n) : U32}}))
  +cell_below = Order.offset_lt(U32.to_nat(position),cell,width,cell_in)
  +cell_room = Order.transitive(cell_n,Nat.add(U32.to_nat(position),width),U32.to_nat(capacity),Bounds.strict_le(cell_n,Nat.add(U32.to_nat(position),width),cell_below),fits)
  +cell_exact = Bounds.roundtrip(cell_n,capacity,capacity_ok,cell_room)
  +miss = rows_miss_of(later,1n+row,axes,cap,cap_ok,room,positive,source,target,capacity,capacity_ok,
    Equal.trans(Nat,1n+Nat.add(row,later),Nat.add(row,1n+later),total(outer),Equal.sym(Nat,Nat.add(row,1n+later),1n+Nat.add(row,later),Addition.successor_right(row,later)),split),
    {%next : CellsBelow(axes,source,target,capacity,_,Nat.mul(later,width)) after_below : CellsBelow(axes,source,target,capacity,Nat.mul(1n+row,width),Nat.mul(later,width))},
    U32.from_nat(cell_n),
    {%next : CellsMiss(axes,source,target,U32.from_nat(cell_n),_,Nat.mul(later,width))
      {%query_exact : CellsMiss(axes,source,target,_,Nat.add(Nat.mul(row,width),width),Nat.mul(later,width)) after_miss :
        CellsMiss(axes,source,target,U32.from_nat(cell_n),Nat.add(Nat.mul(row,width),width),Nat.mul(later,width))} :
      CellsMiss(axes,source,target,U32.from_nat(cell_n),Nat.mul(1n+row,width),Nat.mul(later,width))})
  +offsets = box_offsets(axes,cap,cap_ok,room,positive,row,cell,row_in,cell_in,source,target)
  +read_source = Equal.trans(U32,cell_source(axes,index,source,target),cell_source(axes,inner,source,target),
    (row_source(outer,row,source,target) + U32.from_nat(cell) * part_step(axes) : U32),
    Equal.cong(Nat,U32,place => cell_source(axes,place,source,target),index,inner,index_exact),
    Equal.cong(Move.Offsets,U32,offsets_source,Move.row_offsets(axes,U32.from_nat(inner),source,target),
      Move.Offsets{(row_source(outer,row,source,target) + U32.from_nat(cell) * part_step(axes) : U32),
        (position + U32.from_nat(cell) * axis_step(run) : U32)},offsets))
  +stride_room = run_room(Move.coalesced(axes),word,coalesced_room(axes,word,positive,strides))
  +fuel_split = Equal.trans(Nat,U32.to_nat(Move.rows_of(outer)),total(outer),Nat.add(row,1n+later),rows_count(axes,cap,cap_ok,room,positive),Equal.sym(Nat,Nat.add(row,1n+later),total(outer),split))
  %Equal.sym(Nat,U32.to_nat(Move.rows_of(outer)),Nat.add(row,1n+later),fuel_split) :
    {Tree.read(Element,Held.ran_tree(~Element,model_rows(~Element,_,0n,outer,axis_extent(run),axis_stride(run),axis_back(run),source,target,holding,tree)),
        cell_target(axes,index,source,target)) == value(context,cell_source(axes,index,source,target)) : Element}
  %Equal.sym(U32,cell_target(axes,index,source,target),U32.from_nat(cell_n),query_exact) :
    {Tree.read(Element,Held.ran_tree(~Element,model_rows(~Element,Nat.add(row,1n+later),0n,outer,axis_extent(run),axis_stride(run),axis_back(run),source,target,holding,tree)),_) ==
      value(context,cell_source(axes,index,source,target)) : Element}
  %Equal.sym(U32,cell_source(axes,index,source,target),(row_source(outer,row,source,target) + U32.from_nat(cell) * part_step(axes) : U32),read_source) :
    {Tree.read(Element,Held.ran_tree(~Element,model_rows(~Element,Nat.add(row,1n+later),0n,outer,axis_extent(run),axis_stride(run),axis_back(run),source,target,holding,tree)),
        U32.from_nat(cell_n)) == value(context,_) : Element}
  rows_written(~Element,~Context,~value,context,row,later,0n,row,Addition.zero_right(row),outer,axis_extent(run),axis_stride(run),axis_back(run),source,target,
    holding,tree,stack,ordered,start,end,source_capacity,source_ok,bounded,word,word_ok,stride_room,sorted,values,run_held,depth,balanced,depth_ok,
    capacity,{==},capacity_ok,fits,cell,cell_in,
    {%cell_exact : RowsMiss(later,1n+row,outer,axis_extent(run),source,target,capacity,_) miss :
      RowsMiss(later,1n+row,outer,axis_extent(run),source,target,capacity,cell_n)})

# ---- A part as the runtime runs it is the part as rows ----

# Zero over a positive word is zero.
def word_zero_div(+divisor: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+divisor_room: {Nat.is_le(Nat.double(U32.to_nat(divisor)),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(divisor)) == True{} : Bool}) -> {(0 / divisor : U32) == 0 : U32}:
  +exact = U32Division.divide(0,divisor,cap,cap_ok,divisor_room,positive)
  +none = Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(0n,U32.to_nat(divisor)),(0n,0n),Division.small(0n,U32.to_nat(divisor),positive))
  Equal.trans(U32,(0 / divisor : U32),U32.from_nat(U32.to_nat((0 / divisor : U32))),0,
    Equal.sym(U32,U32.from_nat(U32.to_nat((0 / divisor : U32))),(0 / divisor : U32),Arithmetic.native_roundtrip((0 / divisor : U32))),
    Equal.cong(Nat,U32,U32.from_nat,U32.to_nat((0 / divisor : U32)),0n,Equal.trans(Nat,U32.to_nat((0 / divisor : U32)),Nat.div(0n,U32.to_nat(divisor)),0n,exact,none)))

# Zero modulo a positive word is zero.
def word_zero_mod(+divisor: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+divisor_room: {Nat.is_le(Nat.double(U32.to_nat(divisor)),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(divisor)) == True{} : Bool}) -> {(0 % divisor : U32) == 0 : U32}:
  +exact = U32Division.modulo(0,divisor,cap,cap_ok,divisor_room,positive)
  +none = Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(0n,U32.to_nat(divisor)),(0n,0n),Division.small(0n,U32.to_nat(divisor),positive))
  Equal.trans(U32,(0 % divisor : U32),U32.from_nat(U32.to_nat((0 % divisor : U32))),0,
    Equal.sym(U32,U32.from_nat(U32.to_nat((0 % divisor : U32))),(0 % divisor : U32),Arithmetic.native_roundtrip((0 % divisor : U32))),
    Equal.cong(Nat,U32,U32.from_nat,U32.to_nat((0 % divisor : U32)),0n,Equal.trans(Nat,U32.to_nat((0 % divisor : U32)),Nat.mod(0n,U32.to_nat(divisor)),0n,exact,none)))

# The first row of nonempty axes starts at the source.
def offsets_zero(axes: List<&2,Move.Axis>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room_total: {Nat.is_le(Nat.double(total(axes)),U32.to_nat(cap)) == True{} : Bool},
  +positive_total: {Nat.is_lt(0n,total(axes)) == True{} : Bool},+source: U32,+target: U32) -> {offsets_source(Move.row_offsets(axes,0,source,target)) == source : U32}:
  match axes:
    case Nil{}: {==}
    case Con{Move.Axis{+size,+gap,+reverse,+jump},+rest}:
      +size_positive = RowMajor.left_factor_positive(U32.to_nat(size),total(rest),positive_total)
      +rest_positive = DivisionOrder.right_factor_positive(U32.to_nat(size),total(rest),positive_total)
      +size_room = double_room(U32.to_nat(size),Nat.mul(U32.to_nat(size),total(rest)),cap,RowMajor.product_covers(U32.to_nat(size),total(rest),rest_positive),room_total)
      +rest_room = double_room(total(rest),Nat.mul(U32.to_nat(size),total(rest)),cap,DivisionOrder.le_multiple(U32.to_nat(size),total(rest),size_positive),room_total)
      +widest = Extrema.max_one_exact(size,U32.to_nat(size),{==},size_positive)
      +block = U32.max(Move.rows_of(rest),1)
      +block_exact = Extrema.max_one_exact(Move.rows_of(rest),total(rest),rows_exact(rest,cap,cap_ok,rest_positive,single_room(total(rest),cap,rest_room)),rest_positive)
      +block_positive = {%Equal.sym(Nat,U32.to_nat(block),total(rest),block_exact) : {Nat.is_lt(0n,_) == True{} : Bool} rest_positive : {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool}}
      +block_room = {%Equal.sym(Nat,U32.to_nat(block),total(rest),block_exact) : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool} rest_room :
        {Nat.is_le(Nat.double(U32.to_nat(block)),U32.to_nat(cap)) == True{} : Bool}}
      +widest_positive = {%Equal.sym(Nat,U32.to_nat(U32.max(size,1)),U32.to_nat(size),widest) : {Nat.is_lt(0n,_) == True{} : Bool} size_positive :
        {Nat.is_lt(0n,U32.to_nat(U32.max(size,1))) == True{} : Bool}}
      +widest_room = {%Equal.sym(Nat,U32.to_nat(U32.max(size,1)),U32.to_nat(size),widest) : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool} size_room :
        {Nat.is_le(Nat.double(U32.to_nat(U32.max(size,1))),U32.to_nat(cap)) == True{} : Bool}}
      +later_target = (target + (0 / block % U32.max(size,1)) * jump : U32)
      %Equal.sym(U32,(0 / block : U32),0,word_zero_div(block,cap,cap_ok,block_room,block_positive)) :
        {offsets_source(Move.row_offsets(rest,(0 % block : U32),(source + (_ % U32.max(size,1)) * Move.directed(reverse,gap) : U32),later_target)) == source : U32}
      %Equal.sym(U32,(0 % U32.max(size,1) : U32),0,word_zero_mod(U32.max(size,1),cap,cap_ok,widest_room,widest_positive)) :
        {offsets_source(Move.row_offsets(rest,(0 % block : U32),(source + _ * Move.directed(reverse,gap) : U32),later_target)) == source : U32}
      %Equal.sym(U32,(source + U32.from_nat(0n) * Move.directed(reverse,gap) : U32),source,zero_step(source,Move.directed(reverse,gap))) :
        {offsets_source(Move.row_offsets(rest,(0 % block : U32),_,later_target)) == source : U32}
      %Equal.sym(U32,(0 % block : U32),0,word_zero_mod(block,cap,cap_ok,block_room,block_positive)) :
        {offsets_source(Move.row_offsets(rest,_,source,later_target)) == source : U32}
      offsets_zero(rest,cap,cap_ok,rest_room,rest_positive,source,later_target)

# The axes before the run of one axis step at most half a word when the axis does.
def dense_room(dense: Bool,+extent: U32,+stride: U32,+back: Bool,+step: U32,+word: U32,
  +own: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool}) -> AllRoom(split_outer(Move.split_dense(dense,Move.Axis{extent,stride,back,step})),word):
  match dense:
    case True{}: Unit{}
    case False{}: Held.Both{own,Unit{}}

# The axes before the run of axes that all step at most half a word step at
# most half a word.
def outer_room(axes: List<&2,Move.Axis>,+word: U32,+room: AllRoom(axes,word)) -> AllRoom(outer_of(axes),word):
  match axes:
    case Nil{}: Unit{}
    case Con{Move.Axis{+extent,+stride,+back,+step},Nil{}}:
      Held.Both{+own,+later} = room
      dense_room(U32.is_eq(step,1),extent,stride,back,step,word,own)
    case Con{Move.Axis{+extent,+stride,+back,+step},Con{+next,+more}}:
      Held.Both{+own,+later} = room
      %Equal.sym(List<&2,Move.Axis>,outer_of(Con{Move.Axis{extent,stride,back,step},Con{next,more}}),Con{Move.Axis{extent,stride,back,step},outer_of(Con{next,more})},
        outer_cons(Move.Axis{extent,stride,back,step},next,more)) : AllRoom(_,word)
      Held.Both{own,outer_room(Con{next,more},word,later)}

# A part from a piece runs as its rows, through the piece when its far
# corner fits it.
def piece_flat(~Element: Data,inside: Bool,+outer: List<&2,Move.Axis>,+extent: U32,+stride: U32,+back: Bool,+step: U32,+source: U32,+target: U32,+behind: Held.Stack<Element>,+buffer: Buffer.Model<Element>,+first: U32,+span: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(span)),U32.to_nat(capacity)) == True{} : Bool},+word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},
  +below: Held.Stack<Element>,+tree: Tree.Tree<Element>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room_total: {Nat.is_le(Nat.double(total(outer)),U32.to_nat(cap)) == True{} : Bool},+positive_total: {Nat.is_lt(0n,total(outer)) == True{} : Bool},+all_room: AllRoom(outer,word),+stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool},+lower: Lower<first,source,source>,
  +found: {Move.fits_from(outer,Move.Axis{extent,stride,back,step},source,first,span) == inside : Bool}) ->
  {piece_ran(~Element,inside,Move.nest_of(outer),extent,stride,back,source,target,behind,buffer,first,span,below,tree) ==
    model_rows(~Element,U32.to_nat(Move.rows_of(outer)),0n,outer,extent,stride,back,source,target,Held.Holding{behind,Held.Held{buffer,first,span,below}},tree) : Held.Ran<Element>}:
  match inside:
    case True{}:
      Equal.trans(Held.Ran<Element>,
        Held.Ran{Held.Holding{behind,Held.Held{buffer,first,span,below}},piece_nested_tree(~Element,Move.nest_of(outer),extent,stride,(source - first : U32),target,Buffer.tree(~Element,buffer),tree)},
        nested_ran(~Element,Move.nest_of(outer),extent,stride,back,source,target,Held.Holding{behind,Held.Held{buffer,first,span,below}},tree),
        model_rows(~Element,U32.to_nat(Move.rows_of(outer)),0n,outer,extent,stride,back,source,target,Held.Holding{behind,Held.Held{buffer,first,span,below}},tree),
        Equal.sym(Held.Ran<Element>,nested_ran(~Element,Move.nest_of(outer),extent,stride,back,source,target,Held.Holding{behind,Held.Held{buffer,first,span,below}},tree),
          Held.Ran{Held.Holding{behind,Held.Held{buffer,first,span,below}},piece_nested_tree(~Element,Move.nest_of(outer),extent,stride,(source - first : U32),target,Buffer.tree(~Element,buffer),tree)},
          nested_piece(~Element,Move.nest_of(outer),extent,stride,back,step,source,target,behind,buffer,first,span,below,tree,
            piece_fits(outer,extent,stride,back,step,source,target,first,span,capacity,capacity_ok,room,word,word_ok,cap,cap_ok,room_total,positive_total,all_room,stride_room,lower,found))),
        nested_flat(~Element,outer,extent,stride,back,source,target,Held.Holding{behind,Held.Held{buffer,first,span,below}},tree,cap,cap_ok,room_total,positive_total))
    case False{}:
      nested_flat(~Element,outer,extent,stride,back,source,target,Held.Holding{behind,Held.Held{buffer,first,span,below}},tree,cap,cap_ok,room_total,positive_total)

# A part from the current piece runs as its rows.
def current_flat(~Element: Data,+outer: List<&2,Move.Axis>,+extent: U32,+stride: U32,+back: Bool,+step: U32,+source: U32,+target: U32,holding: Held.Holding<Element>,+tree: Tree.Tree<Element>,
  +word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room_total: {Nat.is_le(Nat.double(total(outer)),U32.to_nat(cap)) == True{} : Bool},+positive_total: {Nat.is_lt(0n,total(outer)) == True{} : Bool},+all_room: AllRoom(outer,word),+stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(word)) == True{} : Bool},
  +capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+current: Held.Current(~Element,holding,source,capacity)) ->
  {current_ran(~Element,outer,Move.Axis{extent,stride,back,step},source,target,holding,tree) ==
    model_rows(~Element,U32.to_nat(Move.rows_of(outer)),0n,outer,extent,stride,back,source,target,holding,tree) : Held.Ran<Element>}:
  match holding:
    case Held.Holding{+behind,Held.Held{+buffer,+first,+span,+below}}:
      Held.Holds{+after,+before,+inside} = current
      piece_flat(~Element,Move.fits_from(outer,Move.Axis{extent,stride,back,step},source,first,span),outer,extent,stride,back,step,source,target,behind,buffer,first,span,capacity,capacity_ok,inside,word,word_ok,
        below,tree,cap,cap_ok,room_total,positive_total,all_room,stride_room,Lower{after,Order.reflexive(U32.to_nat(source))},{==})
    case Held.Holding{+behind,Held.Bottom{}}:
      nested_flat(~Element,outer,extent,stride,back,source,target,Held.Holding{behind,Held.Bottom{}},tree,cap,cap_ok,room_total,positive_total)

# A part runs as its rows from the piece it resolves at its source.
def part_flat(~Element: Data,+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(total(axes)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,total(axes)) == True{} : Bool},
  +word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+strides: StridesRoom(axes,word),
  +capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+current: Held.Current(~Element,Held.resolve(~Element,holding,source),source,capacity)) ->
  {part_ran(~Element,axes,source,target,holding,tree) == flat_ran(~Element,axes,source,target,Held.resolve(~Element,holding,source),tree) : Held.Ran<Element>}:
  +outer = part_outer(axes)
  +width = part_width(axes)
  +run = part_run(axes)
  +resolved = Held.resolve(~Element,holding,source)
  +product = part_total(axes,cap,cap_ok,positive,room)
  +product_positive = {%Equal.sym(Nat,Nat.mul(total(outer),width),total(axes),product) : {Nat.is_lt(0n,_) == True{} : Bool} positive :
    {Nat.is_lt(0n,Nat.mul(total(outer),width)) == True{} : Bool}}
  +outer_positive = RowMajor.left_factor_positive(total(outer),width,product_positive)
  +width_positive = DivisionOrder.right_factor_positive(total(outer),width,product_positive)
  +outer_le = {%product : {Nat.is_le(total(outer),_) == True{} : Bool} RowMajor.product_covers(total(outer),width,width_positive) :
    {Nat.is_le(total(outer),total(axes)) == True{} : Bool}}
  +coalesced = coalesced_room(axes,word,positive,strides)
  %axis_eta(run) : {current_ran(~Element,outer,_,source,target,resolved,tree) == flat_ran(~Element,axes,source,target,resolved,tree) : Held.Ran<Element>}
  current_flat(~Element,outer,axis_extent(run),axis_stride(run),axis_back(run),axis_step(run),source,target,resolved,tree,word,word_ok,cap,cap_ok,
    double_room(total(outer),total(axes),cap,outer_le,room),outer_positive,outer_room(Move.coalesced(axes),word,coalesced),run_room(Move.coalesced(axes),word,coalesced),
    capacity,capacity_ok,current)

# The first cell of nonempty held cells is held.
def first_held(+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+ranges: Held.Spans,count: Nat,+positive: {Nat.is_lt(0n,count) == True{} : Bool},
  +cells: CellsHeld(axes,source,target,ranges,0n,count)) -> Held.SpanHeld(ranges,U32.to_nat(cell_source(axes,0n,source,target))):
  match count:
    case 0n: Empty.absurd(Held.SpanHeld(ranges,U32.to_nat(cell_source(axes,0n,source,target))),WordEncoding.false_true(positive))
    case 1n+ +rest:
      Held.Both{+here,+later} = cells
      here

# A part whose cells are held resolves the piece of its source.
def part_current(~Element: Data,+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+holding: Held.Holding<Element>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(total(axes)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,total(axes)) == True{} : Bool},+stack: Held.Stack<Element>,+ordered: {Held.order(~Element,holding) == stack : Held.Stack<Element>},+start: Nat,+end: Nat,+source_capacity: U32,+source_ok: {U32.is_le(source_capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(end,U32.to_nat(source_capacity)) == True{} : Bool},+sorted: Held.Sorted(~Element,stack,start,end),+cells: CellsHeld(axes,source,target,Held.spans(~Element,stack),0n,total(axes))) ->
  Held.Current(~Element,Held.resolve(~Element,holding,source),source,source_capacity):
  +held = first_held(axes,source,target,Held.spans(~Element,stack),total(axes),positive,cells)
  +start_held = {%offsets_zero(axes,cap,cap_ok,room,positive,source,target) :
    Held.SpanHeld(Held.spans(~Element,stack),U32.to_nat(_)) held : Held.SpanHeld(Held.spans(~Element,stack),U32.to_nat(source))}
  Held.resolve_current(~Element,holding,source,start,end,source_capacity,source_ok,bounded,
    {%Equal.sym(Held.Stack<Element>,Held.order(~Element,holding),stack,ordered) : Held.Sorted(~Element,_,start,end) sorted : Held.Sorted(~Element,Held.order(~Element,holding),start,end)},
    {%Equal.sym(Held.Stack<Element>,Held.order(~Element,holding),stack,ordered) : Held.SpanHeld(Held.spans(~Element,_),U32.to_nat(source)) start_held :
      Held.SpanHeld(Held.spans(~Element,Held.order(~Element,holding)),U32.to_nat(source))})

# A part whose cells are held leaves every target cell it misses.
def part_kept(~Element: Data,+axes: List<&2,Move.Axis>,+source: U32,+target: U32,+holding: Held.Holding<Element>,+tree: Tree.Tree<Element>,+depth: Nat,
  +balanced: Shape.Balanced(Element,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity_ok: {U32.is_le(Tree.capacity(Element,tree),2147483648) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(total(axes)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,total(axes)) == True{} : Bool},
  +word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+strides: StridesRoom(axes,word),+stack: Held.Stack<Element>,+ordered: {Held.order(~Element,holding) == stack : Held.Stack<Element>},+start: Nat,+end: Nat,+source_capacity: U32,+source_ok: {U32.is_le(source_capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(end,U32.to_nat(source_capacity)) == True{} : Bool},+sorted: Held.Sorted(~Element,stack,start,end),+cells: CellsHeld(axes,source,target,Held.spans(~Element,stack),0n,total(axes)),
  +below: CellsBelow(axes,source,target,Tree.capacity(Element,tree),0n,total(axes)),
  +query: U32,+query_inside: {U32.is_lt(query,Tree.capacity(Element,tree)) == True{} : Bool},+miss: CellsMiss(axes,source,target,query,0n,total(axes))) ->
  {Tree.read(Element,Held.ran_tree(~Element,part_ran(~Element,axes,source,target,holding,tree)),query) == Tree.read(Element,tree,query) : Element}:
  +current = part_current(~Element,axes,source,target,holding,cap,cap_ok,room,positive,stack,ordered,start,end,source_capacity,source_ok,bounded,sorted,cells)
  %Equal.sym(Held.Ran<Element>,part_ran(~Element,axes,source,target,holding,tree),flat_ran(~Element,axes,source,target,Held.resolve(~Element,holding,source),tree),
    part_flat(~Element,axes,source,target,holding,tree,cap,cap_ok,room,positive,word,word_ok,strides,source_capacity,source_ok,current)) :
    {Tree.read(Element,Held.ran_tree(~Element,_),query) == Tree.read(Element,tree,query) : Element}
  flat_kept(~Element,axes,source,target,Held.resolve(~Element,holding,source),tree,depth,balanced,depth_ok,capacity_ok,cap,cap_ok,room,positive,below,query,query_inside,miss)

# A part whose cells are held writes each target cell it reaches with the
# value of its source cell.
def part_written(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+axes: List<&2,Move.Axis>,+source: U32,+target: U32,
  +holding: Held.Holding<Element>,+tree: Tree.Tree<Element>,+depth: Nat,+balanced: Shape.Balanced(Element,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity_ok: {U32.is_le(Tree.capacity(Element,tree),2147483648) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(total(axes)),U32.to_nat(cap)) == True{} : Bool},
  +word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+strides: StridesRoom(axes,word),
  +stack: Held.Stack<Element>,+ordered: {Held.order(~Element,holding) == stack : Held.Stack<Element>},+start: Nat,+end: Nat,+source_capacity: U32,
  +source_ok: {U32.is_le(source_capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(end,U32.to_nat(source_capacity)) == True{} : Bool},
  +sorted: Held.Sorted(~Element,stack,start,end),+values: Held.HoldingValues(~Element,~Context,~value,context,holding),
  +cells: CellsHeld(axes,source,target,Held.spans(~Element,stack),0n,total(axes)),
  +below: CellsBelow(axes,source,target,Tree.capacity(Element,tree),0n,total(axes)),
  +index: Nat,+inside: {Nat.is_lt(index,total(axes)) == True{} : Bool},+apart: CellsApart(axes,source,target,index,0n,total(axes))) ->
  {Tree.read(Element,Held.ran_tree(~Element,part_ran(~Element,axes,source,target,holding,tree)),cell_target(axes,index,source,target)) ==
    value(context,cell_source(axes,index,source,target)) : Element}:
  +positive = Order.le_lt_trans(0n,index,total(axes),Order.zero_le(index),inside)
  +current = part_current(~Element,axes,source,target,holding,cap,cap_ok,room,positive,stack,ordered,start,end,source_capacity,source_ok,bounded,sorted,cells)
  %Equal.sym(Held.Ran<Element>,part_ran(~Element,axes,source,target,holding,tree),flat_ran(~Element,axes,source,target,Held.resolve(~Element,holding,source),tree),
    part_flat(~Element,axes,source,target,holding,tree,cap,cap_ok,room,positive,word,word_ok,strides,source_capacity,source_ok,current)) :
    {Tree.read(Element,Held.ran_tree(~Element,_),cell_target(axes,index,source,target)) == value(context,cell_source(axes,index,source,target)) : Element}
  flat_written(~Element,~Context,~value,context,axes,source,target,Held.resolve(~Element,holding,source),tree,depth,balanced,depth_ok,capacity_ok,cap,cap_ok,room,word,word_ok,strides,
    stack,Equal.trans(Held.Stack<Element>,Held.order(~Element,Held.resolve(~Element,holding,source)),Held.order(~Element,holding),stack,Held.resolve_order(~Element,holding,source),ordered),
    start,end,source_capacity,source_ok,bounded,sorted,Held.resolve_values(~Element,~Context,~value,context,holding,source,values),cells,below,index,inside,apart)

# ---- A part with dense target steps ----

def DenseRoom(+axes: List<&2,Move.Axis>,+target: U32,+bound: U32) -> Data:
  {Nat.is_le(Nat.add(U32.to_nat(target),total(axes)),U32.to_nat(bound)) == True{} : Bool}

# A box index as a word.
def index_exact(+axes: List<&2,Move.Axis>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(total(axes)),U32.to_nat(cap)) == True{} : Bool},+index: Nat,+inside: {Nat.is_lt(index,total(axes)) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(index)) == index : Nat}:
  Bounds.roundtrip(index,cap,cap_ok,Order.transitive(index,total(axes),U32.to_nat(cap),Bounds.strict_le(index,total(axes),inside),single_room(total(axes),cap,room)))

# With dense steps box index index has target cell target + index.
def dense_cell(+axes: List<&2,Move.Axis>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(total(axes)),U32.to_nat(cap)) == True{} : Bool},+dense: DenseSteps(axes),+source: U32,+target: U32,
  +index: Nat,+inside: {Nat.is_lt(index,total(axes)) == True{} : Bool}) ->
  {cell_target(axes,index,source,target) == (target + U32.from_nat(index) : U32) : U32}:
  +exact = index_exact(axes,cap,cap_ok,room,index,inside)
  dense_targets(axes,cap,cap_ok,room,dense,U32.from_nat(index),
    {%Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,exact) : {Nat.is_lt(_,total(axes)) == True{} : Bool} inside :
      {Nat.is_lt(U32.to_nat(U32.from_nat(index)),total(axes)) == True{} : Bool}},source,target)

# The value of a dense target cell.
def dense_value(+axes: List<&2,Move.Axis>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(total(axes)),U32.to_nat(cap)) == True{} : Bool},+dense: DenseSteps(axes),+source: U32,+target: U32,
  +target_room: DenseRoom(axes,target,cap),+index: Nat,+inside: {Nat.is_lt(index,total(axes)) == True{} : Bool}) ->
  {U32.to_nat(cell_target(axes,index,source,target)) == Nat.add(U32.to_nat(target),index) : Nat}:
  +at = dense_cell(axes,cap,cap_ok,room,dense,source,target,index,inside)
  +exact = index_exact(axes,cap,cap_ok,room,index,inside)
  %Equal.sym(U32,cell_target(axes,index,source,target),(target + U32.from_nat(index) : U32),at) : {U32.to_nat(_) == Nat.add(U32.to_nat(target),index) : Nat}
  Arithmetic.sum_exact(cap,cap_ok,target,U32.from_nat(index),U32.to_nat(target),index,{==},exact,
    Order.transitive(Nat.add(U32.to_nat(target),index),Nat.add(U32.to_nat(target),total(axes)),U32.to_nat(cap),
      Order.offset_le(U32.to_nat(target),index,total(axes),Bounds.strict_le(index,total(axes),inside)),target_room))

# Dense cells from first on lie below a capacity the box fits.
def dense_below(count: Nat,+axes: List<&2,Move.Axis>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(total(axes)),U32.to_nat(cap)) == True{} : Bool},+dense: DenseSteps(axes),+source: U32,+target: U32,
  +target_room: DenseRoom(axes,target,cap),+capacity: U32,+fits: DenseRoom(axes,target,capacity),+first: Nat,
  +within: {Nat.is_le(Nat.add(first,count),total(axes)) == True{} : Bool}) -> CellsBelow(axes,source,target,capacity,first,count):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      +inside = Order.room_index(first,rest,total(axes),within)
      Held.Both{{%Equal.sym(Nat,U32.to_nat(cell_target(axes,first,source,target)),Nat.add(U32.to_nat(target),first),
            dense_value(axes,cap,cap_ok,room,dense,source,target,target_room,first,inside)) : {Nat.is_lt(_,U32.to_nat(capacity)) == True{} : Bool}
          Bounds.nat_trans(Nat.add(U32.to_nat(target),first),Nat.add(U32.to_nat(target),total(axes)),U32.to_nat(capacity),
            Order.offset_lt(U32.to_nat(target),first,total(axes),inside),fits) :
          {Nat.is_lt(U32.to_nat(cell_target(axes,first,source,target)),U32.to_nat(capacity)) == True{} : Bool}},
        dense_below(rest,axes,cap,cap_ok,room,dense,source,target,target_room,capacity,fits,1n+first,
          {%Addition.successor_right(first,rest) : {Nat.is_le(_,total(axes)) == True{} : Bool} within : {Nat.is_le(Nat.add(1n+first,rest),total(axes)) == True{} : Bool}})}

# Numbers equal after a common offset are equal.
def offset_equal(+offset: Nat,+left: Nat,+right: Nat) -> {Nat.is_eq(Nat.add(offset,left),Nat.add(offset,right)) == Nat.is_eq(left,right) : Bool}:
  match offset:
    case 0n: {==}
    case 1n+ +rest: offset_equal(rest,left,right)

# Dense cells from first on are the cell of index only at index.
def dense_apart(count: Nat,+axes: List<&2,Move.Axis>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(total(axes)),U32.to_nat(cap)) == True{} : Bool},+dense: DenseSteps(axes),+source: U32,+target: U32,
  +target_room: DenseRoom(axes,target,cap),+index: Nat,+index_in: {Nat.is_lt(index,total(axes)) == True{} : Bool},+first: Nat,
  +within: {Nat.is_le(Nat.add(first,count),total(axes)) == True{} : Bool}) -> CellsApart(axes,source,target,index,first,count):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      +inside = Order.room_index(first,rest,total(axes),within)
      +at_first = dense_value(axes,cap,cap_ok,room,dense,source,target,target_room,first,inside)
      +at_index = dense_value(axes,cap,cap_ok,room,dense,source,target,target_room,index,index_in)
      Held.Both{Equal.trans(Bool,U32.is_eq(cell_target(axes,first,source,target),cell_target(axes,index,source,target)),
          Nat.is_eq(U32.to_nat(cell_target(axes,first,source,target)),U32.to_nat(cell_target(axes,index,source,target))),Nat.is_eq(first,index),
          Comparison.eq(cell_target(axes,first,source,target),cell_target(axes,index,source,target)),
          {%Equal.sym(Nat,U32.to_nat(cell_target(axes,first,source,target)),Nat.add(U32.to_nat(target),first),at_first) :
              {Nat.is_eq(_,U32.to_nat(cell_target(axes,index,source,target))) == Nat.is_eq(first,index) : Bool}
            {%Equal.sym(Nat,U32.to_nat(cell_target(axes,index,source,target)),Nat.add(U32.to_nat(target),index),at_index) :
                {Nat.is_eq(Nat.add(U32.to_nat(target),first),_) == Nat.is_eq(first,index) : Bool}
              offset_equal(U32.to_nat(target),first,index) :
              {Nat.is_eq(Nat.add(U32.to_nat(target),first),U32.to_nat(cell_target(axes,index,source,target))) == Nat.is_eq(first,index) : Bool}} :
            {Nat.is_eq(U32.to_nat(cell_target(axes,first,source,target)),U32.to_nat(cell_target(axes,index,source,target))) == Nat.is_eq(first,index) : Bool}}),
        dense_apart(rest,axes,cap,cap_ok,room,dense,source,target,target_room,index,index_in,1n+first,
          {%Addition.successor_right(first,rest) : {Nat.is_le(_,total(axes)) == True{} : Bool} within : {Nat.is_le(Nat.add(1n+first,rest),total(axes)) == True{} : Bool}})}

# A part with dense target steps from held source pieces writes target cell
# target + index with the value of the source cell of box index index.
def dense_written(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+axes: List<&2,Move.Axis>,+source: U32,+target: U32,
  +holding: Held.Holding<Element>,+tree: Tree.Tree<Element>,+depth: Nat,+balanced: Shape.Balanced(Element,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity_ok: {U32.is_le(Tree.capacity(Element,tree),2147483648) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(total(axes)),U32.to_nat(cap)) == True{} : Bool},+dense: DenseSteps(axes),
  +target_room: DenseRoom(axes,target,cap),+fits: DenseRoom(axes,target,Tree.capacity(Element,tree)),
  +word: U32,+word_ok: {U32.is_le(word,2147483648) == True{} : Bool},+strides: StridesRoom(axes,word),
  +stack: Held.Stack<Element>,+ordered: {Held.order(~Element,holding) == stack : Held.Stack<Element>},+start: Nat,+end: Nat,+source_capacity: U32,
  +source_ok: {U32.is_le(source_capacity,2147483648) == True{} : Bool},+bounded: {Nat.is_le(end,U32.to_nat(source_capacity)) == True{} : Bool},
  +sorted: Held.Sorted(~Element,stack,start,end),+values: Held.HoldingValues(~Element,~Context,~value,context,holding),
  +cells: CellsHeld(axes,source,target,Held.spans(~Element,stack),0n,total(axes)),
  +index: Nat,+inside: {Nat.is_lt(index,total(axes)) == True{} : Bool}) ->
  {Tree.read(Element,Held.ran_tree(~Element,part_ran(~Element,axes,source,target,holding,tree)),(target + U32.from_nat(index) : U32)) ==
    value(context,cell_source(axes,index,source,target)) : Element}:
  %dense_cell(axes,cap,cap_ok,room,dense,source,target,index,inside) : {Tree.read(Element,Held.ran_tree(~Element,part_ran(~Element,axes,source,target,holding,tree)),_) == value(context,cell_source(axes,index,source,target)) : Element}
  part_written(~Element,~Context,~value,context,axes,source,target,holding,tree,depth,balanced,depth_ok,capacity_ok,cap,cap_ok,room,word,word_ok,strides,
    stack,ordered,start,end,source_capacity,source_ok,bounded,sorted,values,cells,
    dense_below(total(axes),axes,cap,cap_ok,room,dense,source,target,target_room,Tree.capacity(Element,tree),fits,0n,Order.reflexive(total(axes))),
    index,inside,dense_apart(total(axes),axes,cap,cap_ok,room,dense,source,target,target_room,index,inside,0n,Order.reflexive(total(axes))))
