# The order of Partials as pure functions. A pass of a reduced axis merges the
# runs of its lines in a binary counter, which builds the left-complete tree
# of every line. The partials of the rows lie in a tree model, so the merge of
# one output is the tree of the values the model holds at the addresses of
# its rows.
import Base
import ../tensor_partials.bend as Partials
import ./u32_bit_length.bend as Bits
import ../tensor_pieces.bend as Pieces
import ./natural_addition.bend as Addition
import ./nat_order.bend as Order
import ./nat_division.bend as Quotients
import ./word_encoding.bend as WordEncoding
import ./tensor_grid_ranges.bend as GridRanges
import ./u32_comparison.bend as Comparison
import ./word_multiplication.bend as WordMultiplication
import ./binary_division.bend as BinaryDivision
import ./row_major_index.bend as RowMajor
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./encoded_coordinates.bend as Encoded
import ./nat_to_u32_bounds.bend as Bounds
import ./tensor_line_chain_values.bend as LineChainValues
import ./tensor_move_readback.bend as Held
import ./u32_full_range.bend as FullRange
import ../kernel_gemm.bend as KernelGemm
import ../tensor_view.bend as Views
import ../tensor_grid.bend as Grid
import ../storage_buffer.bend as Storage
import ../machine_limits.bend as Machine
import ./storage_tree_model.bend as Tree
import ./list_storage_readback.bend as ListReadback
import ./storage_shape.bend as Shape
import ./tensor_allocation_depth.bend as AllocationDepth
import ./tensor_grid_pieces.bend as GridPieces
import ./machine_limit_bounds.bend as Limits
import ./u32_division.bend as U32Division
import ./nat_division_order.bend as DivisionOrder
import ./kernel_gemm_spec.bend as GemmSpec
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_box_digits.bend as Digits
import ../tensor.bend as Tensors
import ../tensor_layout.bend as Layout
import ../tensor_line_chain.bend as LineChain
import ../tensor_operations.bend as Operations
import ./tensor_reduce_values.bend as Reduce
import ./storage_buffer_model.bend as Buffer
import ../traversal_partition.bend as Partition
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_pieces_content.bend as Content
import ./tensor_pieces_plan.bend as PiecesPlan
import ./tensor_global_shape_count.bend as Global
import ./tensor_count_completion.bend as CountCompletion
import ../tensor_shape.bend as Shapes
import ./boolean_logic.bend as Boolean
import ./tensor_contract.bend as Contract
import ./tensor_list_values.bend as Listed
import ./tensor_tiled_public.bend as TiledPublic
import ./tensor_storage_count.bend as StorageCount
import ../default_machine_limits.bend as Default
import ./tensor_tile_coordinates.bend as Coordinates

def odd_of(n: Nat) -> Bool:
  match n:
    case 0n: False{}
    case 1n+m:
      match m:
        case 0n: True{}
        case 1n+k: odd_of(k)

def half_of(n: Nat) -> Nat:
  match n:
    case 0n: 0n
    case 1n+m:
      match m:
        case 0n: 0n
        case 1n+k: 1n+half_of(k)

def ones_nat(remaining: Nat,odd: Bool,+value: Nat) -> Nat:
  match remaining odd:
    case 1n+more True{}: 1n+ones_nat(more,odd_of(half_of(value)),half_of(value))
    case remaining odd: 0n

def odd_double(k: Nat) -> {odd_of(Nat.double(k)) == False{} : Bool}:
  match k:
    case 0n: {==}
    case 1n+ +p: odd_double(p)

def odd_successor(k: Nat) -> {odd_of(1n+Nat.double(k)) == True{} : Bool}:
  match k:
    case 0n: {==}
    case 1n+ +p: odd_successor(p)

def half_double(k: Nat) -> {half_of(Nat.double(k)) == k : Nat}:
  match k:
    case 0n: {==}
    case 1n+ +p: Equal.cong(Nat,Nat,value => 1n+value,half_of(Nat.double(p)),p,half_double(p))

def half_successor(k: Nat) -> {half_of(1n+Nat.double(k)) == k : Nat}:
  match k:
    case 0n: {==}
    case 1n+ +p: Equal.cong(Nat,Nat,value => 1n+value,half_of(1n+Nat.double(p)),p,half_successor(p))

def and_zero(n: Nat,w: Word(n)) -> {Word.and(n,w,Word.zero(n)) == Word.zero(n) : Word(n)}:
  match n:
    case 0n: {==}
    case 1n+ +p:
      match w:
        case WCon{b,t}:
          match b:
            case True{}: Equal.cong(Word(p),Word(1n+p),tail => WCon{False{},tail},Word.and(p,t,Word.zero(p)),Word.zero(p),and_zero(p,t))
            case False{}: Equal.cong(Word(p),Word(1n+p),tail => WCon{False{},tail},Word.and(p,t,Word.zero(p)),Word.zero(p),and_zero(p,t))

def low_odd(x: U32) -> {U32.is_eq(U32.and(x,1),1) == odd_of(U32.to_nat(x)) : Bool}:
  match x:
    case U32{w}:
      match w:
        case WCon{b,+t}:
          match b:
            case True{}:
              %Equal.sym(Bool,odd_of(1n+Nat.double(Word.to_nat(31n,t))),True{},odd_successor(Word.to_nat(31n,t))) : {U32.is_eq(U32.and(U32{WCon{True{},t}},1),1) == _ : Bool}
              %Equal.sym(Word(31n),Word.and(31n,t,Word.zero(31n)),Word.zero(31n),and_zero(31n,t)) : {Cmp.is_eq(Word.cmp.fin(True{},True{},Word.cmp(31n,_,Word.zero(31n)))) == True{} : Bool}
              {==}
            case False{}:
              %Equal.sym(Bool,odd_of(Nat.double(Word.to_nat(31n,t))),False{},odd_double(Word.to_nat(31n,t))) : {U32.is_eq(U32.and(U32{WCon{False{},t}},1),1) == _ : Bool}
              %Equal.sym(Word(31n),Word.and(31n,t,Word.zero(31n)),Word.zero(31n),and_zero(31n,t)) : {Cmp.is_eq(Word.cmp.fin(False{},True{},Word.cmp(31n,_,Word.zero(31n)))) == False{} : Bool}
              {==}

def shifted_half(x: U32) -> {U32.to_nat(U32.shr(x)) == half_of(U32.to_nat(x)) : Nat}:
  match x:
    case U32{w}:
      match w:
        case WCon{b,+t}:
          match b:
            case True{}:
              %Equal.sym(Nat,half_of(1n+Nat.double(Word.to_nat(31n,t))),Word.to_nat(31n,t),half_successor(Word.to_nat(31n,t))) : {U32.to_nat(U32.shr(U32{WCon{True{},t}})) == _ : Nat}
              Bits.pad_value(31n,t)
            case False{}:
              %Equal.sym(Nat,half_of(Nat.double(Word.to_nat(31n,t))),Word.to_nat(31n,t),half_double(Word.to_nat(31n,t))) : {U32.to_nat(U32.shr(U32{WCon{False{},t}})) == _ : Nat}
              Bits.pad_value(31n,t)

# The trailing ones of a word, read by halving its value.
def ones_value(remaining: Nat,odd: Bool,+x: U32,+low: {U32.is_eq(U32.and(x,1),1) == odd : Bool}) ->
  {Partials.ones_below(remaining,odd,x) == ones_nat(remaining,odd_of(U32.to_nat(x)),U32.to_nat(x)) : Nat}:
  match remaining odd:
    case 1n+ +more True{}:
      +next = U32.shr(x)
      %low_odd(x) : {1n+Partials.ones_below(more,U32.is_eq(U32.and(next,1),1),next) == ones_nat(1n+more,_,U32.to_nat(x)) : Nat}
      %Equal.sym(Bool,U32.is_eq(U32.and(x,1),1),True{},low) : {1n+Partials.ones_below(more,U32.is_eq(U32.and(next,1),1),next) == ones_nat(1n+more,_,U32.to_nat(x)) : Nat}
      %shifted_half(x) : {1n+Partials.ones_below(more,U32.is_eq(U32.and(next,1),1),next) == 1n+ones_nat(more,odd_of(_),_) : Nat}
      Equal.cong(Nat,Nat,value => 1n+value,Partials.ones_below(more,U32.is_eq(U32.and(next,1),1),next),ones_nat(more,odd_of(U32.to_nat(next)),U32.to_nat(next)),
        ones_value(more,U32.is_eq(U32.and(next,1),1),next,{==}))
    case 0n odd: {==}
    case 1n+ +more False{}:
      %low_odd(x) : {0n == ones_nat(1n+more,_,U32.to_nat(x)) : Nat}
      %Equal.sym(Bool,U32.is_eq(U32.and(x,1),1),False{},low) : {0n == ones_nat(1n+more,_,U32.to_nat(x)) : Nat}
      {==}

# ---- The binary counter of a line ----

# The counter after the runs of a line from index on.
def counted_runs(~Partial: Data,~merge: Partial -> Partial -> Partial,runs: List<&2,List<&2,Partial>>,counter: List<&2,List<&2,Partial>>,+fuel: Nat,+index: Nat) -> List<&2,List<&2,Partial>>:
  match runs:
    case Nil{}: counter
    case Con{run,rest}: counted_runs(~Partial,~merge,rest,Partials.counted(~Partial,~merge,counter,ones_nat(fuel,odd_of(index),index),run),fuel,1n+index)

# The tree of a line over its runs: a left-complete tree with the largest power
# of two below the count on the left.
def line_tree(~Partial: Data,~merge: Partial -> Partial -> Partial,depth: Nat,single: Bool,+count: U32,+runs: List<&2,List<&2,Partial>>) -> List<&2,Partial>:
  match depth single:
    case 0n single: []
    case 1n+rest True{}: Partials.only_run(~Partial,runs)
    case 1n+ +rest False{}:
      +left = Pieces.below(count)
      +right = (count - left : U32)
      Partials.merged_runs(~Partial,~merge,line_tree(~Partial,~merge,rest,U32.is_le(left,1),left,List.take(&2,List<&2,Partial>,runs,U32.to_nat(left))),
        line_tree(~Partial,~merge,rest,U32.is_le(right,1),right,List.drop(&2,List<&2,Partial>,runs,U32.to_nat(left))))

def counted_split(~Partial: Data,~merge: Partial -> Partial -> Partial,+runs: List<&2,List<&2,Partial>>,+n: Nat,+counter: List<&2,List<&2,Partial>>,+fuel: Nat,+index: Nat) ->
  {counted_runs(~Partial,~merge,runs,counter,fuel,index) ==
    counted_runs(~Partial,~merge,List.drop(&2,List<&2,Partial>,runs,n),counted_runs(~Partial,~merge,List.take(&2,List<&2,Partial>,runs,n),counter,fuel,index),fuel,Nat.add(index,n)) : List<&2,List<&2,Partial>>}:
  match runs n:
    case [] n: {==}
    case +run <> +rest 0n:
      %Equal.sym(Nat,Nat.add(index,0n),index,Addition.zero_right(index)) :
        {counted_runs(~Partial,~merge,run <> rest,counter,fuel,index) == counted_runs(~Partial,~merge,run <> rest,counter,fuel,_) : List<&2,List<&2,Partial>>}
      {==}
    case +run <> +rest 1n+ +previous:
      %Equal.sym(Nat,Nat.add(index,1n+previous),1n+Nat.add(index,previous),Addition.successor_right(index,previous)) :
        {counted_runs(~Partial,~merge,rest,Partials.counted(~Partial,~merge,counter,ones_nat(fuel,odd_of(index),index),run),fuel,1n+index) ==
          counted_runs(~Partial,~merge,List.drop(&2,List<&2,Partial>,rest,previous),counted_runs(~Partial,~merge,List.take(&2,List<&2,Partial>,rest,previous),Partials.counted(~Partial,~merge,counter,ones_nat(fuel,odd_of(index),index),run),fuel,1n+index),fuel,_) : List<&2,List<&2,Partial>>}
      counted_split(~Partial,~merge,rest,previous,Partials.counted(~Partial,~merge,counter,ones_nat(fuel,odd_of(index),index),run),fuel,1n+index)

def counted_zero(~Partial: Data,~merge: Partial -> Partial -> Partial,counter: List<&2,List<&2,Partial>>,+run: List<&2,Partial>) -> {Partials.counted(~Partial,~merge,counter,0n,run) == run <> counter : List<&2,List<&2,Partial>>}:
  match counter:
    case []: {==}
    case latest <> earlier: {==}

# The total of a counter after a run, however many merges it counts.
def total_counted(~Partial: Data,~merge: Partial -> Partial -> Partial,counter: List<&2,List<&2,Partial>>,merges: Nat,+run: List<&2,Partial>) ->
  {Partials.counter_total(~Partial,~merge,Partials.counted(~Partial,~merge,counter,merges,run)) == Partials.counter_merged(~Partial,~merge,counter,run) : List<&2,Partial>}:
  match counter:
    case []: {==}
    case +latest <> +earlier:
      match merges:
        case 0n: {==}
        case 1n+ +more: total_counted(~Partial,~merge,earlier,more,Partials.merged_runs(~Partial,~merge,latest,run))

def ones_false(remaining: Nat,+value: Nat) -> {ones_nat(remaining,False{},value) == 0n : Nat}:
  match remaining:
    case 0n: {==}
    case 1n+more: {==}

def ones_even(+remaining: Nat,+k: Nat) -> {ones_nat(remaining,odd_of(Nat.double(k)),Nat.double(k)) == 0n : Nat}:
  %Equal.sym(Bool,odd_of(Nat.double(k)),False{},odd_double(k)) : {ones_nat(remaining,_,Nat.double(k)) == 0n : Nat}
  ones_false(remaining,Nat.double(k))

def ones_odd(+more: Nat,+k: Nat) -> {ones_nat(1n+more,odd_of(1n+Nat.double(k)),1n+Nat.double(k)) == 1n+ones_nat(more,odd_of(k),k) : Nat}:
  %Equal.sym(Bool,odd_of(1n+Nat.double(k)),True{},odd_successor(k)) : {ones_nat(1n+more,_,1n+Nat.double(k)) == 1n+ones_nat(more,odd_of(k),k) : Nat}
  %Equal.sym(Nat,half_of(1n+Nat.double(k)),k,half_successor(k)) : {1n+ones_nat(more,odd_of(_),_) == 1n+ones_nat(more,odd_of(k),k) : Nat}
  {==}

def fuel_step(+fuel: Nat,+level: Nat,+below: {Nat.is_lt(level,fuel) == True{} : Bool}) -> {Nat.sub(fuel,level) == 1n+Nat.sub(fuel,1n+level) : Nat}:
  match fuel:
    case 0n: Empty.absurd({Nat.sub(0n,level) == 1n+Nat.sub(0n,1n+level) : Nat},Order.below_zero(level,below))
    case 1n+ +more:
      match level:
        case 0n: Equal.cong(Nat,Nat,value => 1n+value,more,Nat.sub(more,0n),Equal.sym(Nat,Nat.sub(more,0n),more,Order.subtract_zero(more)))
        case 1n+ +previous: fuel_step(more,previous,below)

# ---- Powers of two ----

def power_add(+x: Nat,+y: Nat) -> {Bits.power(Nat.add(x,y)) == Nat.mul(Bits.power(x),Bits.power(y)) : Nat}:
  match x:
    case 0n: Equal.sym(Nat,Nat.add(Bits.power(y),0n),Bits.power(y),Addition.zero_right(Bits.power(y)))
    case 1n+ +p:
      %Equal.sym(Nat,Bits.power(Nat.add(p,y)),Nat.mul(Bits.power(p),Bits.power(y)),power_add(p,y)) :
        {Nat.double(_) == Nat.mul(Nat.double(Bits.power(p)),Bits.power(y)) : Nat}
      Equal.sym(Nat,Nat.mul(Nat.double(Bits.power(p)),Bits.power(y)),Nat.double(Nat.mul(Bits.power(p),Bits.power(y))),BinaryDivision.product_double(Bits.power(p),Bits.power(y)))

def double_le_inv(+a: Nat,+b: Nat,+ordered: {Nat.is_le(Nat.double(a),Nat.double(b)) == True{} : Bool}) -> {Nat.is_le(a,b) == True{} : Bool}:
  match a:
    case 0n: Order.zero_le(b)
    case 1n+ +p:
      match b:
        case 0n: Empty.absurd({Nat.is_le(1n+p,0n) == True{} : Bool},WordEncoding.false_true(ordered))
        case 1n+ +q: double_le_inv(p,q,ordered)

def power_le_inv(+a: Nat,+b: Nat,+ordered: {Nat.is_le(Bits.power(a),Bits.power(b)) == True{} : Bool}) -> {Nat.is_le(a,b) == True{} : Bool}:
  match a:
    case 0n: Order.zero_le(b)
    case 1n+ +p:
      match b:
        case 0n:
          Empty.absurd({Nat.is_le(1n+p,0n) == True{} : Bool},WordEncoding.false_true(
            Equal.trans(Bool,False{},Nat.is_le(Nat.double(Bits.power(p)),1n),True{},
              Equal.cong(Nat,Bool,value => Nat.is_le(Nat.double(value),1n),1n+Nat.sub(Bits.power(p),1n),Bits.power(p),Order.positive_has_last(Bits.power(p),GridRanges.power_positive(p))),
              ordered)))
        case 1n+ +q: power_le_inv(p,q,double_le_inv(Bits.power(p),Bits.power(q),ordered))

def power_lt_inv(+a: Nat,+b: Nat,+less: {Nat.is_lt(Bits.power(a),Bits.power(b)) == True{} : Bool}) -> {Nat.is_lt(a,b) == True{} : Bool}:
  Bits.true_unless(Nat.is_lt(a,b),not_less => WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(Bits.power(a),Bits.power(b)),True{},
    Equal.sym(Bool,Nat.is_lt(Bits.power(a),Bits.power(b)),False{},Order.not_below(Bits.power(b),Bits.power(a),Bits.power_monotone(b,a,Order.not_less(a,b,not_less)))),less)))

def power_above(+g: Nat) -> {Nat.is_lt(1n,Bits.power(1n+g)) == True{} : Bool}:
  Bounds.predecessor_strict(1n,Bits.power(1n+g),Order.double_le(1n,Bits.power(g),Order.positive_one(Bits.power(g),GridRanges.power_positive(g))))

def le_one_false(+value: Nat,+more: {Nat.is_lt(1n,value) == True{} : Bool}) -> {Nat.is_le(value,1n) == False{} : Bool}:
  match value:
    case 0n: Empty.absurd({Nat.is_le(0n,1n) == False{} : Bool},WordEncoding.false_true(more))
    case 1n+ +rest:
      match rest:
        case 0n: Empty.absurd({Nat.is_le(1n,1n) == False{} : Bool},WordEncoding.false_true(more))
        case 1n+ +others: {==}

def one_le(+count: U32,+exact: {U32.to_nat(count) == 1n : Nat}) -> {U32.is_le(count,1) == True{} : Bool}:
  Equal.trans(Bool,U32.is_le(count,1),Nat.is_le(U32.to_nat(count),U32.to_nat(1)),True{},Comparison.le(count,1),
    {%Equal.sym(Nat,U32.to_nat(count),1n,exact) : {Nat.is_le(_,U32.to_nat(1)) == True{} : Bool} {==} : {Nat.is_le(U32.to_nat(count),U32.to_nat(1)) == True{} : Bool}})

def above_one(+count: U32,+more: {Nat.is_lt(1n,U32.to_nat(count)) == True{} : Bool}) -> {U32.is_le(count,1) == False{} : Bool}:
  Equal.trans(Bool,U32.is_le(count,1),Nat.is_le(U32.to_nat(count),U32.to_nat(1)),False{},Comparison.le(count,1),le_one_false(U32.to_nat(count),more))

def big_count(+count: U32,+test: {U32.is_le(count,1) == False{} : Bool}) -> {Nat.is_lt(1n,U32.to_nat(count)) == True{} : Bool}:
  Bits.true_unless(Nat.is_lt(1n,U32.to_nat(count)),small => WordEncoding.false_true(Equal.trans(Bool,False{},U32.is_le(count,1),True{},Equal.sym(Bool,U32.is_le(count,1),False{},test),
    Equal.trans(Bool,U32.is_le(count,1),Nat.is_le(U32.to_nat(count),U32.to_nat(1)),True{},Comparison.le(count,1),Order.not_less(1n,U32.to_nat(count),small)))))

def word_within(+part: U32,+count: U32,+ordered: {Nat.is_le(U32.to_nat(part),U32.to_nat(count)) == True{} : Bool},+within: {U32.is_le(count,1073741824) == True{} : Bool}) ->
  {U32.is_le(part,1073741824) == True{} : Bool}:
  Order.word_le(part,count,1073741824,Equal.trans(Bool,U32.is_le(part,count),Nat.is_le(U32.to_nat(part),U32.to_nat(count)),True{},Comparison.le(part,count),ordered),within)

def half_rest(+half: Nat,+whole: Nat,+part: Nat,+whole_is: {whole == Nat.double(half) : Nat},+part_is: {part == half : Nat}) -> {Nat.sub(whole,part) == half : Nat}:
  %Equal.sym(Nat,whole,Nat.double(half),whole_is) : {Nat.sub(_,part) == half : Nat}
  %Equal.sym(Nat,part,half,part_is) : {Nat.sub(Nat.double(half),_) == half : Nat}
  %Addition.double(half) : {Nat.sub(_,half) == half : Nat}
  Order.subtract_shift(half,half)

# The left side of a count of a power of two is its half.
def half_exact(+g: Nat,+count: U32,+exact: {U32.to_nat(count) == Bits.power(1n+g) : Nat},+within: {U32.is_le(count,1073741824) == True{} : Bool}) -> {U32.to_nat(Pieces.below(count)) == Bits.power(g) : Nat}:
  +more = {%Equal.sym(Nat,U32.to_nat(count),Bits.power(1n+g),exact) : {Nat.is_lt(1n,_) == True{} : Bool} power_above(g) : {Nat.is_lt(1n,U32.to_nat(count)) == True{} : Bool}}
  +halves = GridRanges.below_facts(count,more,within)
  +under = {%exact : {Nat.is_lt(Bits.power(GridRanges.half_depth(count)),_) == True{} : Bool} GridRanges.below_under(count,halves) : {Nat.is_lt(Bits.power(GridRanges.half_depth(count)),Bits.power(1n+g)) == True{} : Bool}}
  +cover = {%exact : {Nat.is_le(_,Nat.double(Bits.power(GridRanges.half_depth(count)))) == True{} : Bool} GridRanges.below_cover(count,halves) : {Nat.is_le(Bits.power(1n+g),Nat.double(Bits.power(GridRanges.half_depth(count)))) == True{} : Bool}}
  +same = Order.antisymmetric(GridRanges.half_depth(count),g,Order.strict_follow(GridRanges.half_depth(count),1n+g,power_lt_inv(GridRanges.half_depth(count),1n+g,under)),power_le_inv(1n+g,1n+GridRanges.half_depth(count),cover))
  Equal.trans(Nat,U32.to_nat(Pieces.below(count)),Bits.power(GridRanges.half_depth(count)),Bits.power(g),GridRanges.below_exact(count,halves),Equal.cong(Nat,Nat,value => Bits.power(value),GridRanges.half_depth(count),g,same))

def right_exact(+g: Nat,+count: U32,+exact: {U32.to_nat(count) == Bits.power(1n+g) : Nat},+within: {U32.is_le(count,1073741824) == True{} : Bool}) -> {U32.to_nat((count - Pieces.below(count) : U32)) == Bits.power(g) : Nat}:
  +half = half_exact(g,count,exact,within)
  +ordered = Held.nat_le_moved(U32.to_nat(Pieces.below(count)),Bits.power(g),U32.to_nat(count),half,Held.nat_le_right_moved(Bits.power(g),U32.to_nat(count),Nat.double(Bits.power(g)),exact,Order.le_double(Bits.power(g))))
  Equal.trans(Nat,U32.to_nat((count - Pieces.below(count) : U32)),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count))),Bits.power(g),
    Arithmetic.subtract(count,Pieces.below(count),count,Encoded.native_le_transitive(count,1073741824,2147483648,within,{==}),ordered,Order.sub_le(U32.to_nat(count),U32.to_nat(Pieces.below(count)))),
    half_rest(Bits.power(g),U32.to_nat(count),U32.to_nat(Pieces.below(count)),exact,half))

def next_index(+step: Nat,+start: Nat,+moved: Nat,+same: {moved == step : Nat}) -> {Nat.add(step,start) == Nat.add(start,moved) : Nat}:
  Equal.trans(Nat,Nat.add(step,start),Nat.add(start,step),Nat.add(start,moved),Order.commutative(step,start),
    Equal.cong(Nat,Nat,value => Nat.add(start,value),step,moved,Equal.sym(Nat,moved,step,same)))

# The start of a line of at most two to the top plus one runs, as an even
# number of blocks of two to the low.
def index_split(+j: Nat,+top: Nat,+low: Nat,+short: {Nat.is_le(low,top) == True{} : Bool}) ->
  {Nat.mul(Nat.double(Nat.mul(j,Bits.power(Nat.sub(top,low)))),Bits.power(low)) == Nat.mul(j,Bits.power(1n+top)) : Nat}:
  %Equal.sym(Nat,Nat.mul(j,Nat.double(Bits.power(top))),Nat.double(Nat.mul(j,Bits.power(top))),BinaryDivision.product_right(j,Bits.power(top))) :
    {Nat.mul(Nat.double(Nat.mul(j,Bits.power(Nat.sub(top,low)))),Bits.power(low)) == _ : Nat}
  %Equal.sym(Nat,Nat.mul(Nat.double(Nat.mul(j,Bits.power(Nat.sub(top,low)))),Bits.power(low)),Nat.double(Nat.mul(Nat.mul(j,Bits.power(Nat.sub(top,low))),Bits.power(low))),
    BinaryDivision.product_double(Nat.mul(j,Bits.power(Nat.sub(top,low))),Bits.power(low))) :
    {_ == Nat.double(Nat.mul(j,Bits.power(top))) : Nat}
  Equal.cong(Nat,Nat,value => Nat.double(value),Nat.mul(Nat.mul(j,Bits.power(Nat.sub(top,low))),Bits.power(low)),Nat.mul(j,Bits.power(top)),
    Equal.trans(Nat,Nat.mul(Nat.mul(j,Bits.power(Nat.sub(top,low))),Bits.power(low)),Nat.mul(j,Nat.mul(Bits.power(Nat.sub(top,low)),Bits.power(low))),Nat.mul(j,Bits.power(top)),
      RowMajor.assoc(j,Bits.power(Nat.sub(top,low)),Bits.power(low)),
      Equal.cong(Nat,Nat,value => Nat.mul(j,value),Nat.mul(Bits.power(Nat.sub(top,low)),Bits.power(low)),Bits.power(top),
        Equal.trans(Nat,Nat.mul(Bits.power(Nat.sub(top,low)),Bits.power(low)),Bits.power(Nat.add(Nat.sub(top,low),low)),Bits.power(top),
          Equal.sym(Nat,Bits.power(Nat.add(Nat.sub(top,low),low)),Nat.mul(Bits.power(Nat.sub(top,low)),Bits.power(low)),power_add(Nat.sub(top,low),low)),
          Equal.cong(Nat,Nat,value => Bits.power(value),Nat.add(Nat.sub(top,low),low),top,
            Equal.trans(Nat,Nat.add(Nat.sub(top,low),low),Nat.add(low,Nat.sub(top,low)),top,Order.commutative(Nat.sub(top,low),low),Order.restore_shift(low,top,short)))))))

# ---- One line through the counter ----

def single_block(~Partial: Data,~merge: Partial -> Partial -> Partial,+runs: List<&2,List<&2,Partial>>,+counter: List<&2,List<&2,Partial>>,+fuel: Nat,+index: Nat,
  +positive: {Nat.is_lt(0n,List.length(&2,List<&2,Partial>,runs)) == True{} : Bool},
  +short: {Nat.is_le(List.length(&2,List<&2,Partial>,runs),1n) == True{} : Bool}) ->
  {counted_runs(~Partial,~merge,runs,counter,fuel,index) == Partials.counted(~Partial,~merge,counter,ones_nat(fuel,odd_of(index),index),Partials.only_run(~Partial,runs)) : List<&2,List<&2,Partial>>}:
  match runs:
    case []:
      Empty.absurd({counted_runs(~Partial,~merge,[],counter,fuel,index) == Partials.counted(~Partial,~merge,counter,ones_nat(fuel,odd_of(index),index),Partials.only_run(~Partial,[])) : List<&2,List<&2,Partial>>},
        WordEncoding.false_true(positive))
    case +run <> +rest:
      match rest:
        case []: {==}
        case +next <> +others:
          Empty.absurd({counted_runs(~Partial,~merge,run <> next <> others,counter,fuel,index) ==
            Partials.counted(~Partial,~merge,counter,ones_nat(fuel,odd_of(index),index),Partials.only_run(~Partial,run <> next <> others)) : List<&2,List<&2,Partial>>},WordEncoding.false_true(short))

def aligned_single(~Partial: Data,~merge: Partial -> Partial -> Partial,+runs: List<&2,List<&2,Partial>>,+counter: List<&2,List<&2,Partial>>,+fuel: Nat,+index: Nat,
  +positive: {Nat.is_lt(0n,List.length(&2,List<&2,Partial>,runs)) == True{} : Bool},
  +short: {Nat.is_le(List.length(&2,List<&2,Partial>,runs),1n) == True{} : Bool}) ->
  {Partials.counter_total(~Partial,~merge,counted_runs(~Partial,~merge,runs,counter,fuel,index)) == Partials.counter_merged(~Partial,~merge,counter,Partials.only_run(~Partial,runs)) : List<&2,Partial>}:
  %Equal.sym(List<&2,List<&2,Partial>>,counted_runs(~Partial,~merge,runs,counter,fuel,index),Partials.counted(~Partial,~merge,counter,ones_nat(fuel,odd_of(index),index),Partials.only_run(~Partial,runs)),
    single_block(~Partial,~merge,runs,counter,fuel,index,positive,short)) :
    {Partials.counter_total(~Partial,~merge,_) == Partials.counter_merged(~Partial,~merge,counter,Partials.only_run(~Partial,runs)) : List<&2,Partial>}
  total_counted(~Partial,~merge,counter,ones_nat(fuel,odd_of(index),index),Partials.only_run(~Partial,runs))

# A whole block of two to the h runs, aligned to its size, adds its tree to
# the counter with the trailing ones of its place.
def perfect_block(~Partial: Data,~merge: Partial -> Partial -> Partial,+depth: Nat,+h: Nat,+count: U32,+runs: List<&2,List<&2,Partial>>,+counter: List<&2,List<&2,Partial>>,+fuel: Nat,+k: Nat,
  +exact: {U32.to_nat(count) == Bits.power(h) : Nat},
  +length: {List.length(&2,List<&2,Partial>,runs) == Bits.power(h) : Nat},
  +within: {U32.is_le(count,1073741824) == True{} : Bool},
  +deep: {Nat.is_lt(h,depth) == True{} : Bool},
  +fueled: {Nat.is_le(h,fuel) == True{} : Bool}) ->
  {counted_runs(~Partial,~merge,runs,counter,fuel,Nat.mul(k,Bits.power(h))) ==
    Partials.counted(~Partial,~merge,counter,ones_nat(Nat.sub(fuel,h),odd_of(k),k),line_tree(~Partial,~merge,depth,U32.is_le(count,1),count,runs)) : List<&2,List<&2,Partial>>}:
  match depth:
    case 0n:
      Empty.absurd({counted_runs(~Partial,~merge,runs,counter,fuel,Nat.mul(k,Bits.power(h))) ==
        Partials.counted(~Partial,~merge,counter,ones_nat(Nat.sub(fuel,h),odd_of(k),k),line_tree(~Partial,~merge,0n,U32.is_le(count,1),count,runs)) : List<&2,List<&2,Partial>>},Order.below_zero(h,deep))
    case 1n+ +rest:
      match h:
        case 0n:
          %Equal.sym(Bool,U32.is_le(count,1),True{},one_le(count,exact)) :
            {counted_runs(~Partial,~merge,runs,counter,fuel,Nat.mul(k,1n)) == Partials.counted(~Partial,~merge,counter,ones_nat(Nat.sub(fuel,0n),odd_of(k),k),line_tree(~Partial,~merge,1n+rest,_,count,runs)) : List<&2,List<&2,Partial>>}
          %Equal.sym(Nat,Nat.mul(k,1n),k,Order.times_one(k)) :
            {counted_runs(~Partial,~merge,runs,counter,fuel,_) == Partials.counted(~Partial,~merge,counter,ones_nat(Nat.sub(fuel,0n),odd_of(k),k),Partials.only_run(~Partial,runs)) : List<&2,List<&2,Partial>>}
          %Equal.sym(Nat,Nat.sub(fuel,0n),fuel,Order.subtract_zero(fuel)) :
            {counted_runs(~Partial,~merge,runs,counter,fuel,k) == Partials.counted(~Partial,~merge,counter,ones_nat(_,odd_of(k),k),Partials.only_run(~Partial,runs)) : List<&2,List<&2,Partial>>}
          single_block(~Partial,~merge,runs,counter,fuel,k,
            {%Equal.sym(Nat,List.length(&2,List<&2,Partial>,runs),1n,length) : {Nat.is_lt(0n,_) == True{} : Bool} {==} : {Nat.is_lt(0n,List.length(&2,List<&2,Partial>,runs)) == True{} : Bool}},
            {%Equal.sym(Nat,List.length(&2,List<&2,Partial>,runs),1n,length) : {Nat.is_le(_,1n) == True{} : Bool} {==} : {Nat.is_le(List.length(&2,List<&2,Partial>,runs),1n) == True{} : Bool}})
        case 1n+ +g:
          +more = {%Equal.sym(Nat,U32.to_nat(count),Bits.power(1n+g),exact) : {Nat.is_lt(1n,_) == True{} : Bool} power_above(g) : {Nat.is_lt(1n,U32.to_nat(count)) == True{} : Bool}}
          +half = half_exact(g,count,exact,within)
          +right = right_exact(g,count,exact,within)
          +inside = Held.nat_le_moved(U32.to_nat(Pieces.below(count)),Bits.power(g),List.length(&2,List<&2,Partial>,runs),half,Held.nat_le_right_moved(Bits.power(g),List.length(&2,List<&2,Partial>,runs),Nat.double(Bits.power(g)),length,Order.le_double(Bits.power(g))))
          +lower = Order.transitive(g,1n+g,fuel,Order.successor(g,g,Order.reflexive(g)),fueled)
          +left = Equal.trans(List<&2,List<&2,Partial>>,counted_runs(~Partial,~merge,List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))),counter,fuel,Nat.mul(Nat.double(k),Bits.power(g))),Partials.counted(~Partial,~merge,counter,ones_nat(Nat.sub(fuel,g),odd_of(Nat.double(k)),Nat.double(k)),line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))))),line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))) <> counter,
            perfect_block(~Partial,~merge,rest,g,Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))),counter,fuel,Nat.double(k),half,
              Equal.trans(Nat,List.length(&2,List<&2,Partial>,List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))),U32.to_nat(Pieces.below(count)),Bits.power(g),LineChainValues.take_length(~List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)),inside),half),
              word_within(Pieces.below(count),count,Held.nat_le_moved(U32.to_nat(Pieces.below(count)),Bits.power(g),U32.to_nat(count),half,Held.nat_le_right_moved(Bits.power(g),U32.to_nat(count),Nat.double(Bits.power(g)),exact,Order.le_double(Bits.power(g)))),within),
              deep,lower),
            Equal.trans(List<&2,List<&2,Partial>>,Partials.counted(~Partial,~merge,counter,ones_nat(Nat.sub(fuel,g),odd_of(Nat.double(k)),Nat.double(k)),line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))))),Partials.counted(~Partial,~merge,counter,0n,line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))))),line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))) <> counter,
              Equal.cong(Nat,List<&2,List<&2,Partial>>,merges => Partials.counted(~Partial,~merge,counter,merges,line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))))),ones_nat(Nat.sub(fuel,g),odd_of(Nat.double(k)),Nat.double(k)),0n,ones_even(Nat.sub(fuel,g),k)),
              counted_zero(~Partial,~merge,counter,line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))))))
          +after = perfect_block(~Partial,~merge,rest,g,(count - Pieces.below(count) : U32),List.drop(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))),line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))) <> counter,fuel,1n+Nat.double(k),right,
            Equal.trans(Nat,List.length(&2,List<&2,Partial>,List.drop(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))),Nat.sub(List.length(&2,List<&2,Partial>,runs),U32.to_nat(Pieces.below(count))),Bits.power(g),LineChainValues.drop_length(~List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))),half_rest(Bits.power(g),List.length(&2,List<&2,Partial>,runs),U32.to_nat(Pieces.below(count)),length,half)),
            word_within((count - Pieces.below(count) : U32),count,Held.nat_le_moved(U32.to_nat((count - Pieces.below(count) : U32)),Bits.power(g),U32.to_nat(count),right,Held.nat_le_right_moved(Bits.power(g),U32.to_nat(count),Nat.double(Bits.power(g)),exact,Order.le_double(Bits.power(g)))),within),
            deep,lower)
          %Equal.sym(Bool,U32.is_le(count,1),False{},above_one(count,more)) :
            {counted_runs(~Partial,~merge,runs,counter,fuel,Nat.mul(k,Nat.double(Bits.power(g)))) == Partials.counted(~Partial,~merge,counter,ones_nat(Nat.sub(fuel,1n+g),odd_of(k),k),line_tree(~Partial,~merge,1n+rest,_,count,runs)) : List<&2,List<&2,Partial>>}
          %WordMultiplication.double_mul(k,Bits.power(g)) :
            {counted_runs(~Partial,~merge,runs,counter,fuel,_) == Partials.counted(~Partial,~merge,counter,ones_nat(Nat.sub(fuel,1n+g),odd_of(k),k),Partials.merged_runs(~Partial,~merge,line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))),line_tree(~Partial,~merge,rest,U32.is_le((count - Pieces.below(count) : U32),1),(count - Pieces.below(count) : U32),List.drop(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))))) : List<&2,List<&2,Partial>>}
          %Equal.sym(List<&2,List<&2,Partial>>,counted_runs(~Partial,~merge,runs,counter,fuel,Nat.mul(Nat.double(k),Bits.power(g))),counted_runs(~Partial,~merge,List.drop(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))),counted_runs(~Partial,~merge,List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))),counter,fuel,Nat.mul(Nat.double(k),Bits.power(g))),fuel,Nat.add(Nat.mul(Nat.double(k),Bits.power(g)),U32.to_nat(Pieces.below(count)))),
            counted_split(~Partial,~merge,runs,U32.to_nat(Pieces.below(count)),counter,fuel,Nat.mul(Nat.double(k),Bits.power(g)))) :
            {_ == Partials.counted(~Partial,~merge,counter,ones_nat(Nat.sub(fuel,1n+g),odd_of(k),k),Partials.merged_runs(~Partial,~merge,line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))),line_tree(~Partial,~merge,rest,U32.is_le((count - Pieces.below(count) : U32),1),(count - Pieces.below(count) : U32),List.drop(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))))) : List<&2,List<&2,Partial>>}
          %Equal.sym(List<&2,List<&2,Partial>>,counted_runs(~Partial,~merge,List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))),counter,fuel,Nat.mul(Nat.double(k),Bits.power(g))),line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))) <> counter,left) :
            {counted_runs(~Partial,~merge,List.drop(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))),_,fuel,Nat.add(Nat.mul(Nat.double(k),Bits.power(g)),U32.to_nat(Pieces.below(count)))) == Partials.counted(~Partial,~merge,counter,ones_nat(Nat.sub(fuel,1n+g),odd_of(k),k),Partials.merged_runs(~Partial,~merge,line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))),line_tree(~Partial,~merge,rest,U32.is_le((count - Pieces.below(count) : U32),1),(count - Pieces.below(count) : U32),List.drop(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))))) : List<&2,List<&2,Partial>>}
          %next_index(Bits.power(g),Nat.mul(Nat.double(k),Bits.power(g)),U32.to_nat(Pieces.below(count)),half) :
            {counted_runs(~Partial,~merge,List.drop(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))),line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))) <> counter,fuel,_) == Partials.counted(~Partial,~merge,counter,ones_nat(Nat.sub(fuel,1n+g),odd_of(k),k),Partials.merged_runs(~Partial,~merge,line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))),line_tree(~Partial,~merge,rest,U32.is_le((count - Pieces.below(count) : U32),1),(count - Pieces.below(count) : U32),List.drop(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))))) : List<&2,List<&2,Partial>>}
          %Equal.sym(List<&2,List<&2,Partial>>,counted_runs(~Partial,~merge,List.drop(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))),line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))) <> counter,fuel,Nat.mul(1n+Nat.double(k),Bits.power(g))),Partials.counted(~Partial,~merge,line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))) <> counter,ones_nat(Nat.sub(fuel,g),odd_of(1n+Nat.double(k)),1n+Nat.double(k)),line_tree(~Partial,~merge,rest,U32.is_le((count - Pieces.below(count) : U32),1),(count - Pieces.below(count) : U32),List.drop(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))))),after) :
            {_ == Partials.counted(~Partial,~merge,counter,ones_nat(Nat.sub(fuel,1n+g),odd_of(k),k),Partials.merged_runs(~Partial,~merge,line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))),line_tree(~Partial,~merge,rest,U32.is_le((count - Pieces.below(count) : U32),1),(count - Pieces.below(count) : U32),List.drop(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))))) : List<&2,List<&2,Partial>>}
          %Equal.sym(Nat,Nat.sub(fuel,g),1n+Nat.sub(fuel,1n+g),fuel_step(fuel,g,Order.succ_le_lt(g,fuel,fueled))) :
            {Partials.counted(~Partial,~merge,line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))) <> counter,ones_nat(_,odd_of(1n+Nat.double(k)),1n+Nat.double(k)),line_tree(~Partial,~merge,rest,U32.is_le((count - Pieces.below(count) : U32),1),(count - Pieces.below(count) : U32),List.drop(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))))) == Partials.counted(~Partial,~merge,counter,ones_nat(Nat.sub(fuel,1n+g),odd_of(k),k),Partials.merged_runs(~Partial,~merge,line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))),line_tree(~Partial,~merge,rest,U32.is_le((count - Pieces.below(count) : U32),1),(count - Pieces.below(count) : U32),List.drop(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))))) : List<&2,List<&2,Partial>>}
          %Equal.sym(Nat,ones_nat(1n+Nat.sub(fuel,1n+g),odd_of(1n+Nat.double(k)),1n+Nat.double(k)),1n+ones_nat(Nat.sub(fuel,1n+g),odd_of(k),k),ones_odd(Nat.sub(fuel,1n+g),k)) :
            {Partials.counted(~Partial,~merge,line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))) <> counter,_,line_tree(~Partial,~merge,rest,U32.is_le((count - Pieces.below(count) : U32),1),(count - Pieces.below(count) : U32),List.drop(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))))) == Partials.counted(~Partial,~merge,counter,ones_nat(Nat.sub(fuel,1n+g),odd_of(k),k),Partials.merged_runs(~Partial,~merge,line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))),line_tree(~Partial,~merge,rest,U32.is_le((count - Pieces.below(count) : U32),1),(count - Pieces.below(count) : U32),List.drop(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))))) : List<&2,List<&2,Partial>>}
          {==}

# A line of at most two to the b runs from an index aligned to that size
# leaves its tree merged after the counter.
def aligned_line(~Partial: Data,~merge: Partial -> Partial -> Partial,+depth: Nat,+single: Bool,+count: U32,+runs: List<&2,List<&2,Partial>>,+counter: List<&2,List<&2,Partial>>,+fuel: Nat,+j: Nat,+b: Nat,
  +test: {U32.is_le(count,1) == single : Bool},
  +length: {List.length(&2,List<&2,Partial>,runs) == U32.to_nat(count) : Nat},
  +positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
  +fits: {Nat.is_le(U32.to_nat(count),Bits.power(b)) == True{} : Bool},
  +within: {U32.is_le(count,1073741824) == True{} : Bool},
  +deep: {Nat.is_lt(b,depth) == True{} : Bool},
  +fueled: {Nat.is_le(b,fuel) == True{} : Bool}) ->
  {Partials.counter_total(~Partial,~merge,counted_runs(~Partial,~merge,runs,counter,fuel,Nat.mul(j,Bits.power(b)))) ==
    Partials.counter_merged(~Partial,~merge,counter,line_tree(~Partial,~merge,depth,single,count,runs)) : List<&2,Partial>}:
  match depth:
    case 0n:
      Empty.absurd({Partials.counter_total(~Partial,~merge,counted_runs(~Partial,~merge,runs,counter,fuel,Nat.mul(j,Bits.power(b)))) ==
        Partials.counter_merged(~Partial,~merge,counter,line_tree(~Partial,~merge,0n,single,count,runs)) : List<&2,Partial>},Order.below_zero(b,deep))
    case 1n+ +rest:
      match single:
        case True{}:
          aligned_single(~Partial,~merge,runs,counter,fuel,Nat.mul(j,Bits.power(b)),
            Held.nat_lt_moved(0n,List.length(&2,List<&2,Partial>,runs),U32.to_nat(count),length,positive),
            Held.nat_le_moved(List.length(&2,List<&2,Partial>,runs),U32.to_nat(count),1n,length,
              Equal.trans(Bool,Nat.is_le(U32.to_nat(count),1n),U32.is_le(count,1),True{},Equal.sym(Bool,U32.is_le(count,1),Nat.is_le(U32.to_nat(count),U32.to_nat(1)),Comparison.le(count,1)),test)))
        case False{}:
          match b:
            case 0n:
              +more = big_count(count,test)
              +halves = GridRanges.below_facts(count,more,within)
              Empty.absurd({Partials.counter_total(~Partial,~merge,counted_runs(~Partial,~merge,runs,counter,fuel,Nat.mul(j,1n))) ==
                Partials.counter_merged(~Partial,~merge,counter,line_tree(~Partial,~merge,1n+rest,False{},count,runs)) : List<&2,Partial>},
                Order.below_zero(GridRanges.half_depth(count),power_lt_inv(GridRanges.half_depth(count),0n,Order.lt_le(Bits.power(GridRanges.half_depth(count)),U32.to_nat(count),1n,GridRanges.below_under(count,halves),fits))))
            case 1n+ +previous:
              +more = big_count(count,test)
              +halves = GridRanges.below_facts(count,more,within)
              +exact = GridRanges.below_exact(count,halves)
              +under = GridRanges.below_under(count,halves)
              +cover = GridRanges.below_cover(count,halves)
              +short = Order.strict_follow(GridRanges.half_depth(count),1n+previous,power_lt_inv(GridRanges.half_depth(count),1n+previous,Order.lt_le(Bits.power(GridRanges.half_depth(count)),U32.to_nat(count),Bits.power(1n+previous),under,fits)))
              +shallow = Order.le_lt_trans(GridRanges.half_depth(count),previous,rest,short,deep)
              +lower = Order.transitive(GridRanges.half_depth(count),previous,fuel,short,Order.transitive(previous,1n+previous,fuel,Order.successor(previous,previous,Order.reflexive(previous)),fueled))
              +nl_under = {%Equal.sym(Nat,U32.to_nat(Pieces.below(count)),Bits.power(GridRanges.half_depth(count)),exact) : {Nat.is_lt(_,U32.to_nat(count)) == True{} : Bool} under : {Nat.is_lt(U32.to_nat(Pieces.below(count)),U32.to_nat(count)) == True{} : Bool}}
              +ordered = Bounds.strict_le(U32.to_nat(Pieces.below(count)),U32.to_nat(count),nl_under)
              +right = Arithmetic.subtract(count,Pieces.below(count),count,Encoded.native_le_transitive(count,1073741824,2147483648,within,{==}),ordered,Order.sub_le(U32.to_nat(count),U32.to_nat(Pieces.below(count))))
              +inside = Held.nat_le_right_moved(U32.to_nat(Pieces.below(count)),List.length(&2,List<&2,Partial>,runs),U32.to_nat(count),length,ordered)
              +left = Equal.trans(List<&2,List<&2,Partial>>,counted_runs(~Partial,~merge,List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))),counter,fuel,Nat.mul(Nat.double(Nat.mul(j,Bits.power(Nat.sub(previous,GridRanges.half_depth(count))))),Bits.power(GridRanges.half_depth(count)))),Partials.counted(~Partial,~merge,counter,ones_nat(Nat.sub(fuel,GridRanges.half_depth(count)),odd_of(Nat.double(Nat.mul(j,Bits.power(Nat.sub(previous,GridRanges.half_depth(count)))))),Nat.double(Nat.mul(j,Bits.power(Nat.sub(previous,GridRanges.half_depth(count)))))),line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))))),line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))) <> counter,
                perfect_block(~Partial,~merge,rest,GridRanges.half_depth(count),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))),counter,fuel,Nat.double(Nat.mul(j,Bits.power(Nat.sub(previous,GridRanges.half_depth(count))))),exact,
                  Equal.trans(Nat,List.length(&2,List<&2,Partial>,List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))),U32.to_nat(Pieces.below(count)),Bits.power(GridRanges.half_depth(count)),LineChainValues.take_length(~List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)),inside),exact),
                  word_within(Pieces.below(count),count,ordered,within),shallow,lower),
                Equal.trans(List<&2,List<&2,Partial>>,Partials.counted(~Partial,~merge,counter,ones_nat(Nat.sub(fuel,GridRanges.half_depth(count)),odd_of(Nat.double(Nat.mul(j,Bits.power(Nat.sub(previous,GridRanges.half_depth(count)))))),Nat.double(Nat.mul(j,Bits.power(Nat.sub(previous,GridRanges.half_depth(count)))))),line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))))),Partials.counted(~Partial,~merge,counter,0n,line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))))),line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))) <> counter,
                  Equal.cong(Nat,List<&2,List<&2,Partial>>,merges => Partials.counted(~Partial,~merge,counter,merges,line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))))),ones_nat(Nat.sub(fuel,GridRanges.half_depth(count)),odd_of(Nat.double(Nat.mul(j,Bits.power(Nat.sub(previous,GridRanges.half_depth(count)))))),Nat.double(Nat.mul(j,Bits.power(Nat.sub(previous,GridRanges.half_depth(count)))))),0n,ones_even(Nat.sub(fuel,GridRanges.half_depth(count)),Nat.mul(j,Bits.power(Nat.sub(previous,GridRanges.half_depth(count)))))),
                  counted_zero(~Partial,~merge,counter,line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))))))
              +after = aligned_line(~Partial,~merge,rest,U32.is_le((count - Pieces.below(count) : U32),1),(count - Pieces.below(count) : U32),List.drop(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))),line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))) <> counter,fuel,1n+Nat.double(Nat.mul(j,Bits.power(Nat.sub(previous,GridRanges.half_depth(count))))),GridRanges.half_depth(count),{==},
                Equal.trans(Nat,List.length(&2,List<&2,Partial>,List.drop(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))),Nat.sub(List.length(&2,List<&2,Partial>,runs),U32.to_nat(Pieces.below(count))),U32.to_nat((count - Pieces.below(count) : U32)),LineChainValues.drop_length(~List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))),
                  Equal.trans(Nat,Nat.sub(List.length(&2,List<&2,Partial>,runs),U32.to_nat(Pieces.below(count))),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count))),U32.to_nat((count - Pieces.below(count) : U32)),
                    Equal.cong(Nat,Nat,whole => Nat.sub(whole,U32.to_nat(Pieces.below(count))),List.length(&2,List<&2,Partial>,runs),U32.to_nat(count),length),
                    Equal.sym(Nat,U32.to_nat((count - Pieces.below(count) : U32)),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count))),right))),
                Held.nat_lt_moved(0n,U32.to_nat((count - Pieces.below(count) : U32)),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count))),right,Order.positive_difference(U32.to_nat(count),U32.to_nat(Pieces.below(count)),nl_under)),
                Held.nat_le_moved(U32.to_nat((count - Pieces.below(count) : U32)),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count))),Bits.power(GridRanges.half_depth(count)),right,
                  {%Equal.sym(Nat,U32.to_nat(Pieces.below(count)),Bits.power(GridRanges.half_depth(count)),exact) : {Nat.is_le(Nat.sub(U32.to_nat(count),_),Bits.power(GridRanges.half_depth(count))) == True{} : Bool}
                    GridRanges.past_half(U32.to_nat(count),Bits.power(GridRanges.half_depth(count)),Bounds.strict_le(Bits.power(GridRanges.half_depth(count)),U32.to_nat(count),under),cover) : {Nat.is_le(Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count))),Bits.power(GridRanges.half_depth(count))) == True{} : Bool}}),
                word_within((count - Pieces.below(count) : U32),count,Held.nat_le_moved(U32.to_nat((count - Pieces.below(count) : U32)),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count))),U32.to_nat(count),right,Order.sub_le(U32.to_nat(count),U32.to_nat(Pieces.below(count)))),within),
                shallow,lower)
              %index_split(j,previous,GridRanges.half_depth(count),short) :
                {Partials.counter_total(~Partial,~merge,counted_runs(~Partial,~merge,runs,counter,fuel,_)) == Partials.counter_merged(~Partial,~merge,counter,Partials.merged_runs(~Partial,~merge,line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))),line_tree(~Partial,~merge,rest,U32.is_le((count - Pieces.below(count) : U32),1),(count - Pieces.below(count) : U32),List.drop(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))))) : List<&2,Partial>}
              %Equal.sym(List<&2,List<&2,Partial>>,counted_runs(~Partial,~merge,runs,counter,fuel,Nat.mul(Nat.double(Nat.mul(j,Bits.power(Nat.sub(previous,GridRanges.half_depth(count))))),Bits.power(GridRanges.half_depth(count)))),counted_runs(~Partial,~merge,List.drop(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))),counted_runs(~Partial,~merge,List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))),counter,fuel,Nat.mul(Nat.double(Nat.mul(j,Bits.power(Nat.sub(previous,GridRanges.half_depth(count))))),Bits.power(GridRanges.half_depth(count)))),fuel,Nat.add(Nat.mul(Nat.double(Nat.mul(j,Bits.power(Nat.sub(previous,GridRanges.half_depth(count))))),Bits.power(GridRanges.half_depth(count))),U32.to_nat(Pieces.below(count)))),
                counted_split(~Partial,~merge,runs,U32.to_nat(Pieces.below(count)),counter,fuel,Nat.mul(Nat.double(Nat.mul(j,Bits.power(Nat.sub(previous,GridRanges.half_depth(count))))),Bits.power(GridRanges.half_depth(count))))) :
                {Partials.counter_total(~Partial,~merge,_) == Partials.counter_merged(~Partial,~merge,counter,Partials.merged_runs(~Partial,~merge,line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))),line_tree(~Partial,~merge,rest,U32.is_le((count - Pieces.below(count) : U32),1),(count - Pieces.below(count) : U32),List.drop(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))))) : List<&2,Partial>}
              %Equal.sym(List<&2,List<&2,Partial>>,counted_runs(~Partial,~merge,List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))),counter,fuel,Nat.mul(Nat.double(Nat.mul(j,Bits.power(Nat.sub(previous,GridRanges.half_depth(count))))),Bits.power(GridRanges.half_depth(count)))),line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))) <> counter,left) :
                {Partials.counter_total(~Partial,~merge,counted_runs(~Partial,~merge,List.drop(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))),_,fuel,Nat.add(Nat.mul(Nat.double(Nat.mul(j,Bits.power(Nat.sub(previous,GridRanges.half_depth(count))))),Bits.power(GridRanges.half_depth(count))),U32.to_nat(Pieces.below(count))))) == Partials.counter_merged(~Partial,~merge,counter,Partials.merged_runs(~Partial,~merge,line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))),line_tree(~Partial,~merge,rest,U32.is_le((count - Pieces.below(count) : U32),1),(count - Pieces.below(count) : U32),List.drop(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))))) : List<&2,Partial>}
              %next_index(Bits.power(GridRanges.half_depth(count)),Nat.mul(Nat.double(Nat.mul(j,Bits.power(Nat.sub(previous,GridRanges.half_depth(count))))),Bits.power(GridRanges.half_depth(count))),U32.to_nat(Pieces.below(count)),exact) :
                {Partials.counter_total(~Partial,~merge,counted_runs(~Partial,~merge,List.drop(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count))),line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))) <> counter,fuel,_)) == Partials.counter_merged(~Partial,~merge,counter,Partials.merged_runs(~Partial,~merge,line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),List.take(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))),line_tree(~Partial,~merge,rest,U32.is_le((count - Pieces.below(count) : U32),1),(count - Pieces.below(count) : U32),List.drop(&2,List<&2,Partial>,runs,U32.to_nat(Pieces.below(count)))))) : List<&2,Partial>}
              after

# ---- The passes of the runtime ----

# Lists joined in order.
def joined(~Element: Data,lists: List<&2,List<&2,Element>>) -> List<&2,Element>:
  match lists:
    case Nil{}: []
    case Con{list,rest}: List.append(&2,Element,list,joined(~Element,rest))

# Every list of the given length.
def Lengths(~Element: Data,lists: List<&2,List<&2,Element>>,+length: Nat) -> Data:
  match lists:
    case Nil{}: Unit
    case Con{list,rest}: KernelGemm.Both<{List.length(&2,Element,list) == length : Nat},Lengths(~Element,rest,length)>

# The tree of each line of a reduced axis through the counter.
def line_totals(~Partial: Data,~merge: Partial -> Partial -> Partial,lines: List<&2,List<&2,List<&2,Partial>>>) -> List<&2,List<&2,Partial>>:
  match lines:
    case Nil{}: []
    case Con{line,rest}: Partials.counter_total(~Partial,~merge,counted_runs(~Partial,~merge,line,[],32n,0n)) <> line_totals(~Partial,~merge,rest)

# Each line of a kept axis joined.
def line_joins(~Partial: Data,lines: List<&2,List<&2,List<&2,Partial>>>) -> List<&2,List<&2,Partial>>:
  match lines:
    case Nil{}: []
    case Con{line,rest}: joined(~Partial,line) <> line_joins(~Partial,rest)

def trailing_word(+index: Nat,+capacity: U32,+inside: {Nat.is_le(index,U32.to_nat(capacity)) == True{} : Bool}) ->
  {Partials.trailing_ones(U32.from_nat(index)) == ones_nat(32n,odd_of(index),index) : Nat}:
  %FullRange.roundtrip(index,capacity,inside) : {Partials.trailing_ones(U32.from_nat(index)) == ones_nat(32n,odd_of(_),_) : Nat}
  ones_value(32n,U32.is_eq(U32.and(U32.from_nat(index),1),1),U32.from_nat(index),{==})

def last_word(+index: Nat,+extent: U32,+ends: {Nat.add(index,1n) == U32.to_nat(extent) : Nat}) -> {U32.is_eq(U32.from_nat(1n+index),extent) == True{} : Bool}:
  +exact = Equal.trans(Nat,1n+index,Nat.add(index,1n),U32.to_nat(extent),Equal.sym(Nat,Nat.add(index,1n),1n+index,Addition.successor_width(index)),ends)
  %Equal.sym(Bool,U32.is_eq(U32.from_nat(1n+index),extent),Nat.is_eq(U32.to_nat(U32.from_nat(1n+index)),U32.to_nat(extent)),Comparison.eq(U32.from_nat(1n+index),extent)) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(1n+index)),1n+index,
    FullRange.roundtrip(1n+index,extent,Held.nat_le_right_moved(1n+index,U32.to_nat(extent),1n+index,Equal.sym(Nat,1n+index,U32.to_nat(extent),exact),Order.reflexive(1n+index)))) :
    {Nat.is_eq(_,U32.to_nat(extent)) == True{} : Bool}
  %exact : {Nat.is_eq(1n+index,_) == True{} : Bool}
  Order.equal_reflexive(1n+index)

def inner_word(+index: Nat,+extent: U32,+more: Nat,+ends: {Nat.add(index,2n+more) == U32.to_nat(extent) : Nat}) -> {U32.is_eq(U32.from_nat(1n+index),extent) == False{} : Bool}:
  +place = Equal.trans(Nat,Nat.add(1n+index,1n+more),Nat.add(index,2n+more),U32.to_nat(extent),
    Equal.sym(Nat,Nat.add(index,2n+more),1n+Nat.add(index,1n+more),Addition.successor_right(index,1n+more)),ends)
  +less = Order.below_complement(1n+index,more,U32.to_nat(extent),place)
  %Equal.sym(Bool,U32.is_eq(U32.from_nat(1n+index),extent),Nat.is_eq(U32.to_nat(U32.from_nat(1n+index)),U32.to_nat(extent)),Comparison.eq(U32.from_nat(1n+index),extent)) : {_ == False{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(1n+index)),1n+index,FullRange.roundtrip(1n+index,extent,Bounds.strict_le(1n+index,U32.to_nat(extent),less))) :
    {Nat.is_eq(_,U32.to_nat(extent)) == False{} : Bool}
  Order.less_not_equal(1n+index,U32.to_nat(extent),less)

def first_word(+extent: U32) -> {U32.is_eq(U32.from_nat(1n),extent) == U32.is_eq(extent,1) : Bool}:
  Equal.trans(Bool,U32.is_eq(U32.from_nat(1n),extent),Nat.is_eq(1n,U32.to_nat(extent)),U32.is_eq(extent,1),Comparison.eq(U32.from_nat(1n),extent),
    Equal.trans(Bool,Nat.is_eq(1n,U32.to_nat(extent)),Nat.is_eq(U32.to_nat(extent),1n),U32.is_eq(extent,1),Order.equal_symmetric(1n,U32.to_nat(extent)),
      Equal.sym(Bool,U32.is_eq(extent,1),Nat.is_eq(U32.to_nat(extent),U32.to_nat(1)),Comparison.eq(extent,1))))

def plus_two(+index: Nat) -> {(U32.from_nat(index) + 2 : U32) == U32.from_nat(2n+index) : U32}:
  Equal.trans(U32,(U32.from_nat(index) + 2 : U32),((U32.from_nat(index) + 1 : U32) + 1 : U32),U32.from_nat(2n+index),
    Equal.sym(U32,((U32.from_nat(index) + 1 : U32) + 1 : U32),(U32.from_nat(index) + (1 + 1 : U32) : U32),Arithmetic.add_associative(U32.from_nat(index),1,1)),
    Equal.trans(U32,((U32.from_nat(index) + 1 : U32) + 1 : U32),(U32.from_nat(1n+index) + 1 : U32),U32.from_nat(2n+index),
      Equal.cong(U32,U32,word => (word + 1 : U32),(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),Bounds.increment_encoded(index)),
      Bounds.increment_encoded(1n+index)))

def index_room(+index: Nat,+extent: U32,+length: Nat,+ends: {Nat.add(index,length) == U32.to_nat(extent) : Nat}) -> {Nat.is_le(index,U32.to_nat(extent)) == True{} : Bool}:
  Held.nat_le_right_moved(index,U32.to_nat(extent),Nat.add(index,length),Equal.sym(Nat,Nat.add(index,length),U32.to_nat(extent),ends),Order.le_add(index,length))

def ends_after(+index: Nat,+more: Nat,+extent: U32,+ends: {Nat.add(index,1n+more) == U32.to_nat(extent) : Nat}) -> {Nat.add(1n+index,more) == U32.to_nat(extent) : Nat}:
  Equal.trans(Nat,Nat.add(1n+index,more),Nat.add(index,1n+more),U32.to_nat(extent),Equal.sym(Nat,Nat.add(index,1n+more),1n+Nat.add(index,more),Addition.successor_right(index,more)),ends)

# One line of a reduced axis from index on ends in the total of its counter.
def reduced_line(~Partial: Data,~merge: Partial -> Partial -> Partial,+line: List<&2,List<&2,Partial>>,+rest: List<&2,List<&2,Partial>>,+extent: U32,+index: Nat,+last: Bool,+counter: List<&2,List<&2,Partial>>,+done: List<&2,List<&2,Partial>>,
  +test: {U32.is_eq(U32.from_nat(1n+index),extent) == last : Bool},
  +ends: {Nat.add(index,List.length(&2,List<&2,Partial>,line)) == U32.to_nat(extent) : Nat},
  +nonempty: {Nat.is_lt(0n,List.length(&2,List<&2,Partial>,line)) == True{} : Bool}) ->
  {Partials.reduced_pass(~Partial,~merge,List.append(&2,List<&2,Partial>,line,rest),last,extent,U32.from_nat(index),counter,done) ==
    Partials.reduced_pass(~Partial,~merge,rest,U32.is_eq(extent,1),extent,0,[],Partials.counter_total(~Partial,~merge,counted_runs(~Partial,~merge,line,counter,32n,index)) <> done) : List<&2,List<&2,Partial>>}:
  match line:
    case []:
      Empty.absurd({Partials.reduced_pass(~Partial,~merge,List.append(&2,List<&2,Partial>,[],rest),last,extent,U32.from_nat(index),counter,done) ==
        Partials.reduced_pass(~Partial,~merge,rest,U32.is_eq(extent,1),extent,0,[],Partials.counter_total(~Partial,~merge,counted_runs(~Partial,~merge,[],counter,32n,index)) <> done) : List<&2,List<&2,Partial>>},
        WordEncoding.false_true(nonempty))
    case +run <> +others:
      match others:
        case []:
          %Equal.sym(Bool,last,True{},Equal.trans(Bool,last,U32.is_eq(U32.from_nat(1n+index),extent),True{},Equal.sym(Bool,U32.is_eq(U32.from_nat(1n+index),extent),last,test),last_word(index,extent,ends))) :
            {Partials.reduced_pass(~Partial,~merge,run <> rest,_,extent,U32.from_nat(index),counter,done) ==
              Partials.reduced_pass(~Partial,~merge,rest,U32.is_eq(extent,1),extent,0,[],Partials.counter_total(~Partial,~merge,Partials.counted(~Partial,~merge,counter,ones_nat(32n,odd_of(index),index),run)) <> done) : List<&2,List<&2,Partial>>}
          %Equal.sym(Nat,Partials.trailing_ones(U32.from_nat(index)),ones_nat(32n,odd_of(index),index),trailing_word(index,extent,index_room(index,extent,1n,ends))) :
            {Partials.reduced_pass(~Partial,~merge,rest,U32.is_eq(extent,1),extent,0,[],Partials.counter_total(~Partial,~merge,Partials.counted(~Partial,~merge,counter,_,run)) <> done) ==
              Partials.reduced_pass(~Partial,~merge,rest,U32.is_eq(extent,1),extent,0,[],Partials.counter_total(~Partial,~merge,Partials.counted(~Partial,~merge,counter,ones_nat(32n,odd_of(index),index),run)) <> done) : List<&2,List<&2,Partial>>}
          {==}
        case +next <> +more:
          %Equal.sym(Bool,last,False{},Equal.trans(Bool,last,U32.is_eq(U32.from_nat(1n+index),extent),False{},Equal.sym(Bool,U32.is_eq(U32.from_nat(1n+index),extent),last,test),
            inner_word(index,extent,List.length(&2,List<&2,Partial>,more),ends))) :
            {Partials.reduced_pass(~Partial,~merge,run <> List.append(&2,List<&2,Partial>,next <> more,rest),_,extent,U32.from_nat(index),counter,done) == Partials.reduced_pass(~Partial,~merge,rest,U32.is_eq(extent,1),extent,0,[],Partials.counter_total(~Partial,~merge,counted_runs(~Partial,~merge,next <> more,Partials.counted(~Partial,~merge,counter,ones_nat(32n,odd_of(index),index),run),32n,1n+index)) <> done) : List<&2,List<&2,Partial>>}
          %Equal.sym(U32,(U32.from_nat(index) + 2 : U32),U32.from_nat(2n+index),plus_two(index)) :
            {Partials.reduced_pass(~Partial,~merge,List.append(&2,List<&2,Partial>,next <> more,rest),U32.is_eq(_,extent),extent,(U32.from_nat(index) + 1 : U32),
              Partials.counted(~Partial,~merge,counter,Partials.trailing_ones(U32.from_nat(index)),run),done) == Partials.reduced_pass(~Partial,~merge,rest,U32.is_eq(extent,1),extent,0,[],Partials.counter_total(~Partial,~merge,counted_runs(~Partial,~merge,next <> more,Partials.counted(~Partial,~merge,counter,ones_nat(32n,odd_of(index),index),run),32n,1n+index)) <> done) : List<&2,List<&2,Partial>>}
          %Equal.sym(U32,(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),Bounds.increment_encoded(index)) :
            {Partials.reduced_pass(~Partial,~merge,List.append(&2,List<&2,Partial>,next <> more,rest),U32.is_eq(U32.from_nat(2n+index),extent),extent,_,
              Partials.counted(~Partial,~merge,counter,Partials.trailing_ones(U32.from_nat(index)),run),done) == Partials.reduced_pass(~Partial,~merge,rest,U32.is_eq(extent,1),extent,0,[],Partials.counter_total(~Partial,~merge,counted_runs(~Partial,~merge,next <> more,Partials.counted(~Partial,~merge,counter,ones_nat(32n,odd_of(index),index),run),32n,1n+index)) <> done) : List<&2,List<&2,Partial>>}
          %Equal.sym(Nat,Partials.trailing_ones(U32.from_nat(index)),ones_nat(32n,odd_of(index),index),trailing_word(index,extent,index_room(index,extent,2n+List.length(&2,List<&2,Partial>,more),ends))) :
            {Partials.reduced_pass(~Partial,~merge,List.append(&2,List<&2,Partial>,next <> more,rest),U32.is_eq(U32.from_nat(2n+index),extent),extent,U32.from_nat(1n+index),
              Partials.counted(~Partial,~merge,counter,_,run),done) == Partials.reduced_pass(~Partial,~merge,rest,U32.is_eq(extent,1),extent,0,[],Partials.counter_total(~Partial,~merge,counted_runs(~Partial,~merge,next <> more,Partials.counted(~Partial,~merge,counter,ones_nat(32n,odd_of(index),index),run),32n,1n+index)) <> done) : List<&2,List<&2,Partial>>}
          reduced_line(~Partial,~merge,next <> more,rest,extent,1n+index,U32.is_eq(U32.from_nat(2n+index),extent),Partials.counted(~Partial,~merge,counter,ones_nat(32n,odd_of(index),index),run),done,{==},
            ends_after(index,1n+List.length(&2,List<&2,Partial>,more),extent,ends),{==})

def reversed_same(~Partial: Data,run: List<&2,Partial>,+joined: List<&2,Partial>) -> {Partials.reversed_onto(~Partial,run,joined) == LineChainValues.onto_values(~Partial,run,joined) : List<&2,Partial>}:
  match run:
    case []: {==}
    case +partial <> +rest: reversed_same(~Partial,rest,partial <> joined)

def back_again(~Element: Data,+items: List<&2,Element>) -> {List.reverse(&2,Element,LineChainValues.onto_values(~Element,items,[])) == items : List<&2,Element>}:
  Equal.trans(List<&2,Element>,List.reverse(&2,Element,LineChainValues.onto_values(~Element,items,[])),List.append(&2,Element,items,[]),items,
    LineChainValues.reverse_onto(~Element,items,[],[]),ListReadback.append_empty(~Element,items))

# One line of a kept axis from index on ends joined.
def kept_line(~Partial: Data,+line: List<&2,List<&2,Partial>>,+rest: List<&2,List<&2,Partial>>,+extent: U32,+index: Nat,+last: Bool,+front: List<&2,Partial>,+done: List<&2,List<&2,Partial>>,
  +test: {U32.is_eq(U32.from_nat(1n+index),extent) == last : Bool},
  +ends: {Nat.add(index,List.length(&2,List<&2,Partial>,line)) == U32.to_nat(extent) : Nat},
  +nonempty: {Nat.is_lt(0n,List.length(&2,List<&2,Partial>,line)) == True{} : Bool}) ->
  {Partials.kept_pass(~Partial,List.append(&2,List<&2,Partial>,line,rest),last,extent,U32.from_nat(index),front,done) ==
    Partials.kept_pass(~Partial,rest,U32.is_eq(extent,1),extent,0,[],List.reverse(&2,Partial,LineChainValues.onto_values(~Partial,joined(~Partial,line),front)) <> done) : List<&2,List<&2,Partial>>}:
  match line:
    case []:
      Empty.absurd({Partials.kept_pass(~Partial,List.append(&2,List<&2,Partial>,[],rest),last,extent,U32.from_nat(index),front,done) ==
        Partials.kept_pass(~Partial,rest,U32.is_eq(extent,1),extent,0,[],List.reverse(&2,Partial,LineChainValues.onto_values(~Partial,joined(~Partial,[]),front)) <> done) : List<&2,List<&2,Partial>>},
        WordEncoding.false_true(nonempty))
    case +run <> +others:
      match others:
        case []:
          %Equal.sym(Bool,last,True{},Equal.trans(Bool,last,U32.is_eq(U32.from_nat(1n+index),extent),True{},Equal.sym(Bool,U32.is_eq(U32.from_nat(1n+index),extent),last,test),last_word(index,extent,ends))) :
            {Partials.kept_pass(~Partial,run <> rest,_,extent,U32.from_nat(index),front,done) ==
              Partials.kept_pass(~Partial,rest,U32.is_eq(extent,1),extent,0,[],List.reverse(&2,Partial,LineChainValues.onto_values(~Partial,List.append(&2,Partial,run,[]),front)) <> done) : List<&2,List<&2,Partial>>}
          Equal.cong(List<&2,Partial>,List<&2,List<&2,Partial>>,values => Partials.kept_pass(~Partial,rest,U32.is_eq(extent,1),extent,0,[],List.reverse(&2,Partial,values) <> done),
            Partials.reversed_onto(~Partial,run,front),LineChainValues.onto_values(~Partial,List.append(&2,Partial,run,[]),front),
            Equal.trans(List<&2,Partial>,Partials.reversed_onto(~Partial,run,front),LineChainValues.onto_values(~Partial,run,front),LineChainValues.onto_values(~Partial,List.append(&2,Partial,run,[]),front),
              reversed_same(~Partial,run,front),
              Equal.cong(List<&2,Partial>,List<&2,Partial>,values => LineChainValues.onto_values(~Partial,values,front),run,List.append(&2,Partial,run,[]),
                Equal.sym(List<&2,Partial>,List.append(&2,Partial,run,[]),run,ListReadback.append_empty(~Partial,run)))))
        case +next <> +more:
          %Equal.sym(Bool,last,False{},Equal.trans(Bool,last,U32.is_eq(U32.from_nat(1n+index),extent),False{},Equal.sym(Bool,U32.is_eq(U32.from_nat(1n+index),extent),last,test),
            inner_word(index,extent,List.length(&2,List<&2,Partial>,more),ends))) :
            {Partials.kept_pass(~Partial,run <> List.append(&2,List<&2,Partial>,next <> more,rest),_,extent,U32.from_nat(index),front,done) == Partials.kept_pass(~Partial,rest,U32.is_eq(extent,1),extent,0,[],List.reverse(&2,Partial,LineChainValues.onto_values(~Partial,List.append(&2,Partial,run,joined(~Partial,next <> more)),front)) <> done) : List<&2,List<&2,Partial>>}
          %Equal.sym(U32,(U32.from_nat(index) + 2 : U32),U32.from_nat(2n+index),plus_two(index)) :
            {Partials.kept_pass(~Partial,List.append(&2,List<&2,Partial>,next <> more,rest),U32.is_eq(_,extent),extent,(U32.from_nat(index) + 1 : U32),Partials.reversed_onto(~Partial,run,front),done) == Partials.kept_pass(~Partial,rest,U32.is_eq(extent,1),extent,0,[],List.reverse(&2,Partial,LineChainValues.onto_values(~Partial,List.append(&2,Partial,run,joined(~Partial,next <> more)),front)) <> done) : List<&2,List<&2,Partial>>}
          %Equal.sym(U32,(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),Bounds.increment_encoded(index)) :
            {Partials.kept_pass(~Partial,List.append(&2,List<&2,Partial>,next <> more,rest),U32.is_eq(U32.from_nat(2n+index),extent),extent,_,Partials.reversed_onto(~Partial,run,front),done) == Partials.kept_pass(~Partial,rest,U32.is_eq(extent,1),extent,0,[],List.reverse(&2,Partial,LineChainValues.onto_values(~Partial,List.append(&2,Partial,run,joined(~Partial,next <> more)),front)) <> done) : List<&2,List<&2,Partial>>}
          Equal.trans(List<&2,List<&2,Partial>>,Partials.kept_pass(~Partial,List.append(&2,List<&2,Partial>,next <> more,rest),U32.is_eq(U32.from_nat(2n+index),extent),extent,U32.from_nat(1n+index),Partials.reversed_onto(~Partial,run,front),done),
            Partials.kept_pass(~Partial,rest,U32.is_eq(extent,1),extent,0,[],List.reverse(&2,Partial,LineChainValues.onto_values(~Partial,joined(~Partial,next <> more),Partials.reversed_onto(~Partial,run,front))) <> done),
            Partials.kept_pass(~Partial,rest,U32.is_eq(extent,1),extent,0,[],List.reverse(&2,Partial,LineChainValues.onto_values(~Partial,List.append(&2,Partial,run,joined(~Partial,next <> more)),front)) <> done),
            kept_line(~Partial,next <> more,rest,extent,1n+index,U32.is_eq(U32.from_nat(2n+index),extent),Partials.reversed_onto(~Partial,run,front),done,{==},
              ends_after(index,1n+List.length(&2,List<&2,Partial>,more),extent,ends),{==}),
            Equal.cong(List<&2,Partial>,List<&2,List<&2,Partial>>,values => Partials.kept_pass(~Partial,rest,U32.is_eq(extent,1),extent,0,[],List.reverse(&2,Partial,values) <> done),
              LineChainValues.onto_values(~Partial,joined(~Partial,next <> more),Partials.reversed_onto(~Partial,run,front)),
              LineChainValues.onto_values(~Partial,List.append(&2,Partial,run,joined(~Partial,next <> more)),front),
              Equal.trans(List<&2,Partial>,LineChainValues.onto_values(~Partial,joined(~Partial,next <> more),Partials.reversed_onto(~Partial,run,front)),
                LineChainValues.onto_values(~Partial,joined(~Partial,next <> more),LineChainValues.onto_values(~Partial,run,front)),
                LineChainValues.onto_values(~Partial,List.append(&2,Partial,run,joined(~Partial,next <> more)),front),
                Equal.cong(List<&2,Partial>,List<&2,Partial>,values => LineChainValues.onto_values(~Partial,joined(~Partial,next <> more),values),
                  Partials.reversed_onto(~Partial,run,front),LineChainValues.onto_values(~Partial,run,front),reversed_same(~Partial,run,front)),
                Equal.sym(List<&2,Partial>,LineChainValues.onto_values(~Partial,List.append(&2,Partial,run,joined(~Partial,next <> more)),front),
                  LineChainValues.onto_values(~Partial,joined(~Partial,next <> more),LineChainValues.onto_values(~Partial,run,front)),
                  LineChainValues.onto_append(~Partial,run,joined(~Partial,next <> more),front)))))

def reduced_lines(~Partial: Data,~merge: Partial -> Partial -> Partial,lines: List<&2,List<&2,List<&2,Partial>>>,+extent: U32,+done: List<&2,List<&2,Partial>>,+lengths: Lengths(~List<&2,Partial>,lines,U32.to_nat(extent)),
  +positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool}) ->
  {Partials.reduced_pass(~Partial,~merge,joined(~List<&2,Partial>,lines),U32.is_eq(extent,1),extent,0,[],done) == LineChainValues.onto_values(~List<&2,Partial>,line_totals(~Partial,~merge,lines),done) : List<&2,List<&2,Partial>>}:
  match lines:
    case []: {==}
    case +line <> +rest:
      KernelGemm.Both{+length,+others} = lengths
      Equal.trans(List<&2,List<&2,Partial>>,Partials.reduced_pass(~Partial,~merge,List.append(&2,List<&2,Partial>,line,joined(~List<&2,Partial>,rest)),U32.is_eq(extent,1),extent,0,[],done),
        Partials.reduced_pass(~Partial,~merge,joined(~List<&2,Partial>,rest),U32.is_eq(extent,1),extent,0,[],Partials.counter_total(~Partial,~merge,counted_runs(~Partial,~merge,line,[],32n,0n)) <> done),
        LineChainValues.onto_values(~List<&2,Partial>,line_totals(~Partial,~merge,line <> rest),done),
        reduced_line(~Partial,~merge,line,joined(~List<&2,Partial>,rest),extent,0n,U32.is_eq(extent,1),[],done,first_word(extent),length,Held.nat_lt_moved(0n,List.length(&2,List<&2,Partial>,line),U32.to_nat(extent),length,positive)),
        reduced_lines(~Partial,~merge,rest,extent,Partials.counter_total(~Partial,~merge,counted_runs(~Partial,~merge,line,[],32n,0n)) <> done,others,positive))

def kept_lines(~Partial: Data,lines: List<&2,List<&2,List<&2,Partial>>>,+extent: U32,+done: List<&2,List<&2,Partial>>,+lengths: Lengths(~List<&2,Partial>,lines,U32.to_nat(extent)),
  +positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool}) ->
  {Partials.kept_pass(~Partial,joined(~List<&2,Partial>,lines),U32.is_eq(extent,1),extent,0,[],done) == LineChainValues.onto_values(~List<&2,Partial>,line_joins(~Partial,lines),done) : List<&2,List<&2,Partial>>}:
  match lines:
    case []: {==}
    case +line <> +rest:
      KernelGemm.Both{+length,+others} = lengths
      Equal.trans(List<&2,List<&2,Partial>>,Partials.kept_pass(~Partial,List.append(&2,List<&2,Partial>,line,joined(~List<&2,Partial>,rest)),U32.is_eq(extent,1),extent,0,[],done),
        Partials.kept_pass(~Partial,joined(~List<&2,Partial>,rest),U32.is_eq(extent,1),extent,0,[],List.reverse(&2,Partial,LineChainValues.onto_values(~Partial,joined(~Partial,line),[])) <> done),
        LineChainValues.onto_values(~List<&2,Partial>,line_joins(~Partial,line <> rest),done),
        kept_line(~Partial,line,joined(~List<&2,Partial>,rest),extent,0n,U32.is_eq(extent,1),[],done,first_word(extent),length,Held.nat_lt_moved(0n,List.length(&2,List<&2,Partial>,line),U32.to_nat(extent),length,positive)),
        Equal.trans(List<&2,List<&2,Partial>>,Partials.kept_pass(~Partial,joined(~List<&2,Partial>,rest),U32.is_eq(extent,1),extent,0,[],List.reverse(&2,Partial,LineChainValues.onto_values(~Partial,joined(~Partial,line),[])) <> done),
          Partials.kept_pass(~Partial,joined(~List<&2,Partial>,rest),U32.is_eq(extent,1),extent,0,[],joined(~Partial,line) <> done),
          LineChainValues.onto_values(~List<&2,Partial>,line_joins(~Partial,line <> rest),done),
          Equal.cong(List<&2,Partial>,List<&2,List<&2,Partial>>,values => Partials.kept_pass(~Partial,joined(~List<&2,Partial>,rest),U32.is_eq(extent,1),extent,0,[],values <> done),
            List.reverse(&2,Partial,LineChainValues.onto_values(~Partial,joined(~Partial,line),[])),joined(~Partial,line),back_again(~Partial,joined(~Partial,line))),
          kept_lines(~Partial,rest,extent,joined(~Partial,line) <> done,others,positive)))

# The pass of an axis over its lines.
def reduced_axis_pass(~Partial: Data,~merge: Partial -> Partial -> Partial,+lines: List<&2,List<&2,List<&2,Partial>>>,+extent: U32,+lengths: Lengths(~List<&2,Partial>,lines,U32.to_nat(extent)),
  +positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool}) ->
  {Partials.axis_pass(~Partial,~merge,True{},extent,joined(~List<&2,Partial>,lines)) == line_totals(~Partial,~merge,lines) : List<&2,List<&2,Partial>>}:
  Equal.trans(List<&2,List<&2,Partial>>,Partials.axis_pass(~Partial,~merge,True{},extent,joined(~List<&2,Partial>,lines)),List.reverse(&2,List<&2,Partial>,LineChainValues.onto_values(~List<&2,Partial>,line_totals(~Partial,~merge,lines),[])),line_totals(~Partial,~merge,lines),
    Equal.cong(List<&2,List<&2,Partial>>,List<&2,List<&2,Partial>>,runs => List.reverse(&2,List<&2,Partial>,runs),Partials.reduced_pass(~Partial,~merge,joined(~List<&2,Partial>,lines),U32.is_eq(extent,1),extent,0,[],[]),
      LineChainValues.onto_values(~List<&2,Partial>,line_totals(~Partial,~merge,lines),[]),reduced_lines(~Partial,~merge,lines,extent,[],lengths,positive)),
    back_again(~List<&2,Partial>,line_totals(~Partial,~merge,lines)))

def kept_axis_pass(~Partial: Data,~merge: Partial -> Partial -> Partial,+lines: List<&2,List<&2,List<&2,Partial>>>,+extent: U32,+lengths: Lengths(~List<&2,Partial>,lines,U32.to_nat(extent)),
  +positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool}) ->
  {Partials.axis_pass(~Partial,~merge,False{},extent,joined(~List<&2,Partial>,lines)) == line_joins(~Partial,lines) : List<&2,List<&2,Partial>>}:
  Equal.trans(List<&2,List<&2,Partial>>,Partials.axis_pass(~Partial,~merge,False{},extent,joined(~List<&2,Partial>,lines)),List.reverse(&2,List<&2,Partial>,LineChainValues.onto_values(~List<&2,Partial>,line_joins(~Partial,lines),[])),line_joins(~Partial,lines),
    Equal.cong(List<&2,List<&2,Partial>>,List<&2,List<&2,Partial>>,runs => List.reverse(&2,List<&2,Partial>,runs),Partials.kept_pass(~Partial,joined(~List<&2,Partial>,lines),U32.is_eq(extent,1),extent,0,[],[]),
      LineChainValues.onto_values(~List<&2,Partial>,line_joins(~Partial,lines),[]),kept_lines(~Partial,lines,extent,[],lengths,positive)),
    back_again(~List<&2,Partial>,line_joins(~Partial,lines)))

# ---- The tree of one output ----

def first_extent(axes: List<&2,Views.Axis>) -> U32:
  match axes:
    case Nil{}: 1
    case Con{Views.Axis{extent,stride},rest}: extent

# More than the levels of a tree over the axes.
def depth_of(axes: List<&2,Views.Axis>) -> Nat:
  1n+Nat.mul(33n,List.length(&2,Views.Axis,axes))

# The place of output index among axes, from offset on.
def address(axes: List<&2,Views.Axis>,+index: U32,+offset: U32) -> U32:
  match axes:
    case Nil{}: offset
    case Con{Views.Axis{+extent,+stride},+rest}:
      +inner = U32.max(Grid.product(Views.axis_extents(rest)),1)
      address(rest,(index % inner : U32),(offset + index / inner % U32.max(extent,1) * stride : U32))

def same_flag(flag: Bool,wanted: Bool) -> Bool:
  match flag wanted:
    case True{} True{}: True{}
    case False{} False{}: True{}
    case flag wanted: False{}

def kept_axis(keep: Bool,axis: Views.Axis,rest: List<&2,Views.Axis>) -> List<&2,Views.Axis>:
  match keep:
    case True{}: axis <> rest
    case False{}: rest

# The axes of extents whose flag is wanted, each with its stride in the
# row-major order of extents.
def axes_with(extents: List<&2,U32>,flags: List<&2,Bool>,+wanted: Bool) -> List<&2,Views.Axis>:
  match extents flags:
    case Con{+extent,+rest} Con{flag,others}:
      kept_axis(same_flag(flag,wanted),Views.Axis{extent,U32.max(Grid.product(rest),1)},axes_with(rest,others,wanted))
    case extents flags: []

# The tree over count indices from first on of the first axis, every index
# the tree over the axes after it, the partial at base plus the strides of
# the coordinates read from the context. fuel exceeds the depth of the tree.
def tree_of(~Partial: Data,~merge: Partial -> Partial -> Partial,~Context: Data,~read: Context -> U32 -> Partial,+context: Context,fuel: Nat,single: Bool,axes: List<&2,Views.Axis>,
  +first: U32,+count: U32,+base: U32,+zero: Partial) -> Partial:
  match fuel single axes:
    case 0n single axes: zero
    case 1n+rest single Nil{}: read(context,base)
    case 1n+rest True{} Con{Views.Axis{extent,+stride},+tail}:
      +next = first_extent(tail)
      tree_of(~Partial,~merge,~Context,~read,context,rest,U32.is_le(next,1),tail,0,next,(base + first * stride : U32),zero)
    case 1n+ +rest False{} Con{Views.Axis{+extent,+stride},+tail}:
      +left = Pieces.below(count)
      +right = (count - left : U32)
      merge(tree_of(~Partial,~merge,~Context,~read,context,rest,U32.is_le(left,1),Con{Views.Axis{extent,stride},tail},first,left,base,zero),
        tree_of(~Partial,~merge,~Context,~read,context,rest,U32.is_le(right,1),Con{Views.Axis{extent,stride},tail},(first + left : U32),right,base,zero))

def model_read(~Partial: Data,model: Tree.Tree<Partial>,index: U32) -> Partial:
  Tree.read(Partial,model,index)

# The partials of a reduction in a tree model, the kept axes with the strides
# of their outputs among the partials and the reduced axes with theirs.
type Box<-Partial: Data> is Data:
  Box{model: Tree.Tree<Partial>,kept: List<&2,Views.Axis>,rows: List<&2,Views.Axis>,offset: U32,zero: Partial}

# The merged partial of output index: the tree of its rows from the address
# of the index among the kept axes.
def box_output(~Partial: Data,~merge: Partial -> Partial -> Partial,box: Box<Partial>,index: Nat) -> Partial:
  Box{+model,+kept,+rows,+offset,+zero} = box
  +count = first_extent(rows)
  tree_of(~Partial,~merge,~Tree.Tree<Partial>,~model_read(~Partial),model,depth_of(rows),U32.is_le(count,1),rows,0,count,address(kept,U32.from_nat(index),offset),zero)

# The partials of a list, written in order into fresh storage.
def written(~Partial: Data,+partials: List<&2,Partial>,+zero: Partial) -> Tree.Tree<Partial>:
  Tree.write_sequence(~Partial,partials,0,Tree.fresh(Partial,Storage.depth_for(U32.from_nat(List.length(&2,Partial,partials))),zero))

# A partial written in order reads back at its place.
def written_at(~Partial: Data,+partials: List<&2,Partial>,+zero: Partial,+index: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,1073741824) == True{} : Bool},
  +room: {Nat.is_le(List.length(&2,Partial,partials),U32.to_nat(cap)) == True{} : Bool},+inside: {Nat.is_lt(index,List.length(&2,Partial,partials)) == True{} : Bool}) ->
  {model_read(~Partial,written(~Partial,partials,zero),U32.from_nat(index)) == ListReadback.at(~Partial,zero,index,partials) : Partial}:
  +count = List.length(&2,Partial,partials)
  +word = U32.from_nat(count)
  +exact = FullRange.roundtrip(count,cap,room)
  +sized = AllocationDepth.bounded(word,Encoded.native_le_transitive(word,cap,1073741824,
    Equal.trans(Bool,U32.is_le(word,cap),Nat.is_le(U32.to_nat(word),U32.to_nat(cap)),True{},Comparison.le(word,cap),
      {%Equal.sym(Nat,U32.to_nat(word),count,exact) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} room : {Nat.is_le(U32.to_nat(word),U32.to_nat(cap)) == True{} : Bool}}),cap_ok))
  +depth = Storage.depth_for(word)
  +tree = Tree.fresh(Partial,depth,zero)
  +capacity = {%Equal.sym(U32,Tree.capacity(Partial,tree),U32.shln(1,depth),Shape.capacity(Partial,depth,tree,Shape.fresh(Partial,depth,zero))) : {Nat.is_le(count,U32.to_nat(_)) == True{} : Bool}
    %exact : {Nat.is_le(_,U32.to_nat(U32.shln(1,depth))) == True{} : Bool} AllocationDepth.allocation_room(word,sized) : {Nat.is_le(count,U32.to_nat(Tree.capacity(Partial,tree))) == True{} : Bool}}
  ListReadback.listed(~Partial,partials,index,0n,depth,tree,zero,Shape.fresh(Partial,depth,zero),AllocationDepth.allocation_depth(word,sized),capacity,inside)

# ---- The outputs of a box of rows by its axes ----

# The outputs of the kept axes of a box from offset on, its tree with fuel.
type Sub<-Partial: Data> is Data:
  Sub{model: Tree.Tree<Partial>,kept: List<&2,Views.Axis>,rows: List<&2,Views.Axis>,offset: U32,zero: Partial,fuel: Nat}

def sub_output(~Partial: Data,~merge: Partial -> Partial -> Partial,sub: Sub<Partial>,index: Nat) -> Partial:
  Sub{+model,+kept,+rows,+offset,+zero,+fuel} = sub
  +count = first_extent(rows)
  tree_of(~Partial,~merge,~Tree.Tree<Partial>,~model_read(~Partial),model,fuel,U32.is_le(count,1),rows,0,count,address(kept,U32.from_nat(index),offset),zero)


# The close of the fold of a line among the closed folds of lines.
def closed_at(~Element: Data,~Row: Data,~Partial: Data,~step: Row -> Element -> Row,~close: Row -> Partial,~Context: Data,~value: Context -> U32 -> Element,+context: Context,
  lines: Nat,+size: Nat,+position: Nat,+state: Row,index: Nat,+fill: Partial,+inside: {Nat.is_lt(index,lines) == True{} : Bool}) ->
  {ListReadback.at(~Partial,fill,index,LineChainValues.turned_values(~Row,~Partial,~close,Reduce.line_folds(~Element,~Row,~step,~Context,~value,context,lines,size,position,state))) ==
    close(Reduce.line_fold(~Element,~Row,~step,~Context,~value,context,size,Nat.add(position,Nat.mul(index,size)),state)) : Partial}:
  match lines index:
    case 0n +index: Empty.absurd({ListReadback.at(~Partial,fill,index,[]) == close(Reduce.line_fold(~Element,~Row,~step,~Context,~value,context,size,Nat.add(position,Nat.mul(index,size)),state)) : Partial},
      WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(index,0n),True{},Equal.sym(Bool,Nat.is_lt(index,0n),False{},Order.not_below(0n,index,Order.zero_le(index))),inside)))
    case 1n+ +rest 0n:
      %Equal.sym(Nat,Nat.add(position,0n),position,Addition.zero_right(position)) :
        {close(Reduce.line_fold(~Element,~Row,~step,~Context,~value,context,size,position,state)) == close(Reduce.line_fold(~Element,~Row,~step,~Context,~value,context,size,_,state)) : Partial}
      {==}
    case 1n+ +rest 1n+ +previous:
      %Addition.associative(position,size,Nat.mul(previous,size)) :
        {ListReadback.at(~Partial,fill,previous,LineChainValues.turned_values(~Row,~Partial,~close,Reduce.line_folds(~Element,~Row,~step,~Context,~value,context,rest,size,Nat.add(position,size),state))) ==
          close(Reduce.line_fold(~Element,~Row,~step,~Context,~value,context,size,_,state)) : Partial}
      closed_at(~Element,~Row,~Partial,~step,~close,~Context,~value,context,rest,size,Nat.add(position,size),state,previous,fill,inside)

def closed_length(~Element: Data,~Row: Data,~Partial: Data,~step: Row -> Element -> Row,~close: Row -> Partial,~Context: Data,~value: Context -> U32 -> Element,+context: Context,
  lines: Nat,+size: Nat,+position: Nat,+state: Row) ->
  {List.length(&2,Partial,LineChainValues.turned_values(~Row,~Partial,~close,Reduce.line_folds(~Element,~Row,~step,~Context,~value,context,lines,size,position,state))) == lines : Nat}:
  match lines:
    case 0n: {==}
    case 1n+ +rest:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Partial,LineChainValues.turned_values(~Row,~Partial,~close,Reduce.line_folds(~Element,~Row,~step,~Context,~value,context,rest,size,Nat.add(position,size),state))),rest,
        closed_length(~Element,~Row,~Partial,~step,~close,~Context,~value,context,rest,size,Nat.add(position,size),state))

# A row of a box of folded lines reads the close of the fold of its line.
def row_read(~Element: Data,~Row: Data,~Partial: Data,~step: Row -> Element -> Row,~close: Row -> Partial,~Context: Data,~value: Context -> U32 -> Element,+context: Context,
  +lines: Nat,+size: Nat,+state: Row,+zero: Partial,+row: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,1073741824) == True{} : Bool},+room: {Nat.is_le(lines,U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(row,lines) == True{} : Bool}) ->
  {model_read(~Partial,written(~Partial,LineChainValues.turned_values(~Row,~Partial,~close,Reduce.line_folds(~Element,~Row,~step,~Context,~value,context,lines,size,0n,state)),zero),U32.from_nat(row)) ==
    close(Reduce.line_fold(~Element,~Row,~step,~Context,~value,context,size,Nat.mul(row,size),state)) : Partial}:
  +partials = LineChainValues.turned_values(~Row,~Partial,~close,Reduce.line_folds(~Element,~Row,~step,~Context,~value,context,lines,size,0n,state))
  +length = closed_length(~Element,~Row,~Partial,~step,~close,~Context,~value,context,lines,size,0n,state)
  +listed_room = {%Equal.sym(Nat,List.length(&2,Partial,partials),lines,length) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} room : {Nat.is_le(List.length(&2,Partial,partials),U32.to_nat(cap)) == True{} : Bool}}
  +listed_inside = {%Equal.sym(Nat,List.length(&2,Partial,partials),lines,length) : {Nat.is_lt(row,_) == True{} : Bool} inside : {Nat.is_lt(row,List.length(&2,Partial,partials)) == True{} : Bool}}
  Equal.trans(Partial,model_read(~Partial,written(~Partial,partials,zero),U32.from_nat(row)),ListReadback.at(~Partial,zero,row,partials),
    close(Reduce.line_fold(~Element,~Row,~step,~Context,~value,context,size,Nat.mul(row,size),state)),
    written_at(~Partial,partials,zero,row,cap,cap_ok,listed_room,listed_inside),
    closed_at(~Element,~Row,~Partial,~step,~close,~Context,~value,context,lines,size,0n,state,row,zero,inside))

def kept_count(kept: List<&2,Views.Axis>) -> Nat:
  U32.to_nat(Grid.product(Views.axis_extents(kept)))

# The runs of the outer indices of a box: run index holds the outputs of the
# inner kept axes from index times stride on.
type Outer<-Partial: Data> is Data:
  Outer{model: Tree.Tree<Partial>,kept: List<&2,Views.Axis>,rows: List<&2,Views.Axis>,stride: U32,zero: Partial,fuel: Nat}

def outer_run(~Partial: Data,~merge: Partial -> Partial -> Partial,outer: Outer<Partial>,index: Nat) -> List<&2,Partial>:
  Outer{+model,+kept,+rows,+stride,+zero,+fuel} = outer
  ListReadback.enumerate(~Partial,~Sub<Partial>,~sub_output(~Partial,~merge),kept_count(kept),0n,Sub{model,kept,rows,(U32.from_nat(index) * stride : U32),zero,fuel})

# ---- Enumerations ----

def enumerate_split(~Element: Data,~Context: Data,~value: Context -> Nat -> Element,before: Nat,+after: Nat,+first: Nat,+context: Context) ->
  {ListReadback.enumerate(~Element,~Context,~value,Nat.add(before,after),first,context) ==
    List.append(&2,Element,ListReadback.enumerate(~Element,~Context,~value,before,first,context),ListReadback.enumerate(~Element,~Context,~value,after,Nat.add(first,before),context)) : List<&2,Element>}:
  match before:
    case 0n:
      %Equal.sym(Nat,Nat.add(first,0n),first,Addition.zero_right(first)) :
        {ListReadback.enumerate(~Element,~Context,~value,after,first,context) == ListReadback.enumerate(~Element,~Context,~value,after,_,context) : List<&2,Element>}
      {==}
    case 1n+ +more:
      %Equal.sym(Nat,Nat.add(first,1n+more),1n+Nat.add(first,more),Addition.successor_right(first,more)) :
        {value(context,first) <> ListReadback.enumerate(~Element,~Context,~value,Nat.add(more,after),1n+first,context) ==
          value(context,first) <> List.append(&2,Element,ListReadback.enumerate(~Element,~Context,~value,more,1n+first,context),ListReadback.enumerate(~Element,~Context,~value,after,_,context)) : List<&2,Element>}
      Equal.cong(List<&2,Element>,List<&2,Element>,tail => value(context,first) <> tail,ListReadback.enumerate(~Element,~Context,~value,Nat.add(more,after),1n+first,context),
        List.append(&2,Element,ListReadback.enumerate(~Element,~Context,~value,more,1n+first,context),ListReadback.enumerate(~Element,~Context,~value,after,Nat.add(1n+first,more),context)),
        enumerate_split(~Element,~Context,~value,more,after,1n+first,context))

# Lines of width values of an enumeration.
type Lined<-Context: Data> is Data:
  Lined{context: Context,width: Nat}

def line_of(~Element: Data,~Context: Data,~value: Context -> Nat -> Element,lined: Lined<Context>,index: Nat) -> List<&2,Element>:
  Lined{+context,+width} = lined
  ListReadback.enumerate(~Element,~Context,~value,width,Nat.mul(index,width),context)

def enumerate_lines(~Element: Data,~Context: Data,~value: Context -> Nat -> Element,count: Nat,+first: Nat,+width: Nat,+context: Context) ->
  {ListReadback.enumerate(~Element,~Context,~value,Nat.mul(count,width),Nat.mul(first,width),context) ==
    joined(~Element,ListReadback.enumerate(~List<&2,Element>,~Lined<Context>,~line_of(~Element,~Context,~value),count,first,Lined{context,width})) : List<&2,Element>}:
  match count:
    case 0n: {==}
    case 1n+ +more:
      Equal.trans(List<&2,Element>,ListReadback.enumerate(~Element,~Context,~value,Nat.add(width,Nat.mul(more,width)),Nat.mul(first,width),context),
        List.append(&2,Element,ListReadback.enumerate(~Element,~Context,~value,width,Nat.mul(first,width),context),
          ListReadback.enumerate(~Element,~Context,~value,Nat.mul(more,width),Nat.add(Nat.mul(first,width),width),context)),
        joined(~Element,ListReadback.enumerate(~List<&2,Element>,~Lined<Context>,~line_of(~Element,~Context,~value),1n+more,first,Lined{context,width})),
        enumerate_split(~Element,~Context,~value,width,Nat.mul(more,width),Nat.mul(first,width),context),
        Equal.cong(List<&2,Element>,List<&2,Element>,tail => List.append(&2,Element,ListReadback.enumerate(~Element,~Context,~value,width,Nat.mul(first,width),context),tail),
          ListReadback.enumerate(~Element,~Context,~value,Nat.mul(more,width),Nat.add(Nat.mul(first,width),width),context),
          joined(~Element,ListReadback.enumerate(~List<&2,Element>,~Lined<Context>,~line_of(~Element,~Context,~value),more,1n+first,Lined{context,width})),
          Equal.trans(List<&2,Element>,ListReadback.enumerate(~Element,~Context,~value,Nat.mul(more,width),Nat.add(Nat.mul(first,width),width),context),
            ListReadback.enumerate(~Element,~Context,~value,Nat.mul(more,width),Nat.add(width,Nat.mul(first,width)),context),
            joined(~Element,ListReadback.enumerate(~List<&2,Element>,~Lined<Context>,~line_of(~Element,~Context,~value),more,1n+first,Lined{context,width})),
            Equal.cong(Nat,List<&2,Element>,start => ListReadback.enumerate(~Element,~Context,~value,Nat.mul(more,width),start,context),
              Nat.add(Nat.mul(first,width),width),Nat.add(width,Nat.mul(first,width)),Order.commutative(Nat.mul(first,width),width)),
            enumerate_lines(~Element,~Context,~value,more,1n+first,width,context))))

def line_lengths(~Element: Data,~Context: Data,~value: Context -> Nat -> Element,count: Nat,+first: Nat,+width: Nat,+context: Context) ->
  Lengths(~Element,ListReadback.enumerate(~List<&2,Element>,~Lined<Context>,~line_of(~Element,~Context,~value),count,first,Lined{context,width}),width):
  match count:
    case 0n: Unit{}
    case 1n+ +more:
      KernelGemm.Both{ListReadback.enumerate_length(~Element,~Context,~value,width,Nat.mul(first,width),context),
        line_lengths(~Element,~Context,~value,more,1n+first,width,context)}

def cons_same(~Element: Data,+left: Element,+right: Element,+left_rest: List<&2,Element>,+right_rest: List<&2,Element>,+heads: {left == right : Element},
  +rests: {left_rest == right_rest : List<&2,Element>}) -> {left <> left_rest == right <> right_rest : List<&2,Element>}:
  Equal.trans(List<&2,Element>,left <> left_rest,right <> left_rest,right <> right_rest,
    Equal.cong(Element,List<&2,Element>,head => head <> left_rest,left,right,heads),
    Equal.cong(List<&2,Element>,List<&2,Element>,tail => right <> tail,left_rest,right_rest,rests))

# Two enumerations agree where their values do.
def enumerate_same(~Element: Data,~Left: Data,~left: Left -> Nat -> Element,~Right: Data,~right: Right -> Nat -> Element,~left_context: Left,~right_context: Right,
  ~left_first: Nat,~right_first: Nat,~total: Nat,
  ~agree: @+index: Nat -> @+inside: {Nat.is_lt(index,total) == True{} : Bool} -> {left(left_context,Nat.add(left_first,index)) == right(right_context,Nat.add(right_first,index)) : Element},
  remaining: Nat,+done: Nat,+room: {Nat.is_le(Nat.add(done,remaining),total) == True{} : Bool}) ->
  {ListReadback.enumerate(~Element,~Left,~left,remaining,Nat.add(left_first,done),left_context) ==
    ListReadback.enumerate(~Element,~Right,~right,remaining,Nat.add(right_first,done),right_context) : List<&2,Element>}:
  match remaining:
    case 0n: {==}
    case 1n+ +more:
      %Addition.successor_right(left_first,done) :
        {left(left_context,Nat.add(left_first,done)) <> ListReadback.enumerate(~Element,~Left,~left,more,_,left_context) ==
          right(right_context,Nat.add(right_first,done)) <> ListReadback.enumerate(~Element,~Right,~right,more,1n+Nat.add(right_first,done),right_context) : List<&2,Element>}
      %Addition.successor_right(right_first,done) :
        {left(left_context,Nat.add(left_first,done)) <> ListReadback.enumerate(~Element,~Left,~left,more,Nat.add(left_first,1n+done),left_context) ==
          right(right_context,Nat.add(right_first,done)) <> ListReadback.enumerate(~Element,~Right,~right,more,_,right_context) : List<&2,Element>}
      cons_same(~Element,left(left_context,Nat.add(left_first,done)),right(right_context,Nat.add(right_first,done)),
        ListReadback.enumerate(~Element,~Left,~left,more,Nat.add(left_first,1n+done),left_context),ListReadback.enumerate(~Element,~Right,~right,more,Nat.add(right_first,1n+done),right_context),
        agree(done,Order.room_index(done,more,total,room)),
        enumerate_same(~Element,~Left,~left,~Right,~right,~left_context,~right_context,~left_first,~right_first,~total,~agree,more,1n+done,Order.advance_room(done,more,total,room)))

# ---- Addresses ----

def address_shift(kept: List<&2,Views.Axis>,+index: U32,+offset: U32,+step: U32) ->
  {address(kept,index,(offset + step : U32)) == (address(kept,index,offset) + step : U32) : U32}:
  match kept:
    case Nil{}: {==}
    case Con{Views.Axis{+extent,+stride},+rest}:
      +inner = U32.max(Grid.product(Views.axis_extents(rest)),1)
      +digit = (index / inner % U32.max(extent,1) * stride : U32)
      Equal.trans(U32,address(rest,(index % inner : U32),((offset + step : U32) + digit : U32)),address(rest,(index % inner : U32),((offset + digit : U32) + step : U32)),
        (address(rest,(index % inner : U32),(offset + digit : U32)) + step : U32),
        Equal.cong(U32,U32,word => address(rest,(index % inner : U32),word),((offset + step : U32) + digit : U32),((offset + digit : U32) + step : U32),Arithmetic.add_swap(offset,step,digit)),
        address_shift(rest,(index % inner : U32),(offset + digit : U32),step))

def half_room(+limits: Machine.Limits,+value: Nat,+room: {Nat.is_le(value,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {Nat.is_le(value,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  Order.transitive(value,U32.to_nat(Machine.capacity(limits)),U32.to_nat(Machine.half_word(limits)),room,Bounds.as_nat_le(Machine.capacity(limits),Machine.half_word(limits),Limits.capacity_within_half(limits)))

def word_quotient(+limits: Machine.Limits,+value: Nat,+divisor: Nat,+value_room: {Nat.is_le(value,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+divisor_room: {Nat.is_le(divisor,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) -> {(U32.from_nat(value) / U32.from_nat(divisor) : U32) == U32.from_nat(Nat.div(value,divisor)) : U32}:
  U32Division.encoded_divide(value,divisor,Machine.half_word(limits),Limits.half_word_fits(limits),half_room(limits,value,value_room),half_room(limits,divisor,divisor_room),
    Limits.double_bound(limits,divisor,divisor_room),positive)

def word_remainder(+limits: Machine.Limits,+value: Nat,+divisor: Nat,+value_room: {Nat.is_le(value,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+divisor_room: {Nat.is_le(divisor,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) -> {(U32.from_nat(value) % U32.from_nat(divisor) : U32) == U32.from_nat(Nat.mod(value,divisor)) : U32}:
  U32Division.encoded_modulo(value,divisor,Machine.half_word(limits),Limits.half_word_fits(limits),half_room(limits,value,value_room),half_room(limits,divisor,divisor_room),
    Limits.double_bound(limits,divisor,divisor_room),positive)

# A word at least one is its own maximum with one, and the word of its value.
def word_at_least(+word: U32,+positive: {Nat.is_lt(0n,U32.to_nat(word)) == True{} : Bool}) -> {U32.max(word,1) == U32.from_nat(U32.to_nat(word)) : U32}:
  Equal.trans(U32,U32.max(word,1),word,U32.from_nat(U32.to_nat(word)),GridPieces.word_kept(word,U32.to_nat(word),{==},positive),
    Equal.sym(U32,U32.from_nat(U32.to_nat(word)),word,Arithmetic.native_roundtrip(word)))

# The place of an output index below the kept axes after a first one.
def address_head(+limits: Machine.Limits,+extent: U32,+stride: U32,+kept: List<&2,Views.Axis>,+offset: U32,+outer: Nat,+inner: Nat,
  +inside: {Nat.is_lt(inner,kept_count(kept)) == True{} : Bool},+outer_inside: {Nat.is_lt(outer,U32.to_nat(extent)) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(extent),kept_count(kept)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {address(Views.Axis{extent,stride} <> kept,U32.from_nat(Nat.add(Nat.mul(outer,kept_count(kept)),inner)),offset) ==
    address(kept,U32.from_nat(inner),(offset + U32.from_nat(outer) * stride : U32)) : U32}:
  +value = Nat.add(Nat.mul(outer,kept_count(kept)),inner)
  +positive = Order.positive_bound(inner,kept_count(kept),inside)
  +extent_positive = Order.positive_bound(outer,U32.to_nat(extent),outer_inside)
  +below = Order.lt_le(value,Nat.add(Nat.mul(outer,kept_count(kept)),kept_count(kept)),Nat.mul(U32.to_nat(extent),kept_count(kept)),Order.offset_lt(Nat.mul(outer,kept_count(kept)),inner,kept_count(kept),inside),
    Held.nat_le_moved(Nat.add(Nat.mul(outer,kept_count(kept)),kept_count(kept)),Nat.mul(1n+outer,kept_count(kept)),Nat.mul(U32.to_nat(extent),kept_count(kept)),Order.commutative(Nat.mul(outer,kept_count(kept)),kept_count(kept)),
      RowMajor.mul_le(1n+outer,U32.to_nat(extent),kept_count(kept),Order.strict_follow(outer,U32.to_nat(extent),outer_inside))))
  +value_room = Order.transitive(value,Nat.mul(U32.to_nat(extent),kept_count(kept)),U32.to_nat(Machine.capacity(limits)),Bounds.strict_le(value,Nat.mul(U32.to_nat(extent),kept_count(kept)),below),room)
  +count_room = Order.transitive(kept_count(kept),Nat.mul(U32.to_nat(extent),kept_count(kept)),U32.to_nat(Machine.capacity(limits)),DivisionOrder.le_multiple(U32.to_nat(extent),kept_count(kept),extent_positive),room)
  +extent_room = Order.transitive(U32.to_nat(extent),Nat.mul(U32.to_nat(extent),kept_count(kept)),U32.to_nat(Machine.capacity(limits)),
    Held.nat_le_right_moved(U32.to_nat(extent),Nat.mul(U32.to_nat(extent),kept_count(kept)),Nat.mul(kept_count(kept),U32.to_nat(extent)),RowMajor.multiply_commutative(U32.to_nat(extent),kept_count(kept)),DivisionOrder.le_multiple(kept_count(kept),U32.to_nat(extent),positive)),room)
  +outer_room = Order.transitive(outer,U32.to_nat(extent),U32.to_nat(Machine.capacity(limits)),Bounds.strict_le(outer,U32.to_nat(extent),outer_inside),extent_room)
  +remainder = Equal.trans(U32,(U32.from_nat(value) % U32.from_nat(kept_count(kept)) : U32),U32.from_nat(Nat.mod(value,kept_count(kept))),U32.from_nat(inner),
    word_remainder(limits,value,kept_count(kept),value_room,count_room,positive),
    Equal.cong(Nat,U32,number => U32.from_nat(number),Nat.mod(value,kept_count(kept)),inner,
      Equal.trans(Nat,Nat.mod(value,kept_count(kept)),Nat.mod(inner,kept_count(kept)),inner,DivisionOrder.remainder_shift(inner,kept_count(kept),outer,positive),DivisionOrder.small_remainder(inner,kept_count(kept),inside))))
  +quotient = Equal.trans(U32,(U32.from_nat(value) / U32.from_nat(kept_count(kept)) : U32),U32.from_nat(Nat.div(value,kept_count(kept))),U32.from_nat(outer),
    word_quotient(limits,value,kept_count(kept),value_room,count_room,positive),
    Equal.cong(Nat,U32,number => U32.from_nat(number),Nat.div(value,kept_count(kept)),outer,
      Equal.trans(Nat,Nat.div(value,kept_count(kept)),Nat.add(outer,Nat.div(inner,kept_count(kept))),outer,DivisionOrder.quotient_shift(inner,kept_count(kept),outer,positive),
        Equal.trans(Nat,Nat.add(outer,Nat.div(inner,kept_count(kept))),Nat.add(outer,0n),outer,
          Equal.cong(Nat,Nat,number => Nat.add(outer,number),Nat.div(inner,kept_count(kept)),0n,DivisionOrder.small_quotient(inner,kept_count(kept),inside)),Addition.zero_right(outer)))))
  +digit = Equal.trans(U32,(U32.from_nat(outer) % U32.max(extent,1) : U32),(U32.from_nat(outer) % U32.from_nat(U32.to_nat(extent)) : U32),U32.from_nat(outer),
    Equal.cong(U32,U32,word => (U32.from_nat(outer) % word : U32),U32.max(extent,1),U32.from_nat(U32.to_nat(extent)),word_at_least(extent,extent_positive)),
    Equal.trans(U32,(U32.from_nat(outer) % U32.from_nat(U32.to_nat(extent)) : U32),U32.from_nat(Nat.mod(outer,U32.to_nat(extent))),U32.from_nat(outer),
      word_remainder(limits,outer,U32.to_nat(extent),outer_room,extent_room,extent_positive),
      Equal.cong(Nat,U32,number => U32.from_nat(number),Nat.mod(outer,U32.to_nat(extent)),outer,DivisionOrder.small_remainder(outer,U32.to_nat(extent),outer_inside))))
  %Equal.sym(U32,U32.max(Grid.product(Views.axis_extents(kept)),1),U32.from_nat(kept_count(kept)),word_at_least(Grid.product(Views.axis_extents(kept)),positive)) :
    {address(kept,(U32.from_nat(value) % _ : U32),(offset + U32.from_nat(value) / _ % U32.max(extent,1) * stride : U32)) ==
      address(kept,U32.from_nat(inner),(offset + U32.from_nat(outer) * stride : U32)) : U32}
  %Equal.sym(U32,(U32.from_nat(value) % U32.from_nat(kept_count(kept)) : U32),U32.from_nat(inner),remainder) :
    {address(kept,_,(offset + U32.from_nat(value) / U32.from_nat(kept_count(kept)) % U32.max(extent,1) * stride : U32)) ==
      address(kept,U32.from_nat(inner),(offset + U32.from_nat(outer) * stride : U32)) : U32}
  %Equal.sym(U32,(U32.from_nat(value) / U32.from_nat(kept_count(kept)) : U32),U32.from_nat(outer),quotient) :
    {address(kept,U32.from_nat(inner),(offset + _ % U32.max(extent,1) * stride : U32)) ==
      address(kept,U32.from_nat(inner),(offset + U32.from_nat(outer) * stride : U32)) : U32}
  %Equal.sym(U32,(U32.from_nat(outer) % U32.max(extent,1) : U32),U32.from_nat(outer),digit) :
    {address(kept,U32.from_nat(inner),(offset + _ * stride : U32)) ==
      address(kept,U32.from_nat(inner),(offset + U32.from_nat(outer) * stride : U32)) : U32}
  {==}

# The offset of an outer index of a line is the offset of the line plus its
# steps.
def offset_step(+outer: Nat,+inner: Nat,+extent: U32,+stride: U32) ->
  {(U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(extent)),inner)) * stride : U32) == ((U32.from_nat(outer) * (extent * stride : U32) : U32) + U32.from_nat(inner) * stride : U32) : U32}:
  %Equal.sym(U32,U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(extent)),inner)),(U32.from_nat(Nat.mul(outer,U32.to_nat(extent))) + U32.from_nat(inner) : U32),GemmSpec.encoded_sum(Nat.mul(outer,U32.to_nat(extent)),inner)) :
    {(_ * stride : U32) == ((U32.from_nat(outer) * (extent * stride : U32) : U32) + U32.from_nat(inner) * stride : U32) : U32}
  %Equal.sym(U32,U32.from_nat(Nat.mul(outer,U32.to_nat(extent))),(U32.from_nat(outer) * U32.from_nat(U32.to_nat(extent)) : U32),GemmSpec.encoded_product(outer,U32.to_nat(extent))) :
    {((_ + U32.from_nat(inner) : U32) * stride : U32) == ((U32.from_nat(outer) * (extent * stride : U32) : U32) + U32.from_nat(inner) * stride : U32) : U32}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(extent)),extent,Arithmetic.native_roundtrip(extent)) :
    {(((U32.from_nat(outer) * _ : U32) + U32.from_nat(inner) : U32) * stride : U32) == ((U32.from_nat(outer) * (extent * stride : U32) : U32) + U32.from_nat(inner) * stride : U32) : U32}
  Equal.trans(U32,(((U32.from_nat(outer) * extent : U32) + U32.from_nat(inner) : U32) * stride : U32),
    (U32.from_nat(outer) * extent * stride + U32.from_nat(inner) * stride : U32),
    ((U32.from_nat(outer) * (extent * stride : U32) : U32) + U32.from_nat(inner) * stride : U32),
    Arithmetic.multiply_additive((U32.from_nat(outer) * extent : U32),U32.from_nat(inner),stride),
    Equal.cong(U32,U32,word => (word + U32.from_nat(inner) * stride : U32),(U32.from_nat(outer) * extent * stride : U32),(U32.from_nat(outer) * (extent * stride : U32) : U32),
      Arithmetic.multiply_associative(U32.from_nat(outer),extent,stride)))

def line_offset(+outer: Nat,+inner: Nat,+extent: U32,+stride: U32,+outer_stride: U32,+scaled: {outer_stride == (extent * stride : U32) : U32}) ->
  {(U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(extent)),inner)) * stride : U32) == ((U32.from_nat(outer) * outer_stride : U32) + U32.from_nat(inner) * stride : U32) : U32}:
  Equal.trans(U32,(U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(extent)),inner)) * stride : U32),((U32.from_nat(outer) * (extent * stride : U32) : U32) + U32.from_nat(inner) * stride : U32),
    ((U32.from_nat(outer) * outer_stride : U32) + U32.from_nat(inner) * stride : U32),offset_step(outer,inner,extent,stride),
    Equal.cong(U32,U32,word => ((U32.from_nat(outer) * word : U32) + U32.from_nat(inner) * stride : U32),(extent * stride : U32),outer_stride,
      Equal.sym(U32,outer_stride,(extent * stride : U32),scaled)))

# The runs of one line of a kept axis joined are the outputs with the axis
# among the kept ones.
def kept_line_runs(~Partial: Data,~merge: Partial -> Partial -> Partial,+limits: Machine.Limits,+model: Tree.Tree<Partial>,+extent: U32,+stride: U32,+outer_stride: U32,+kept: List<&2,Views.Axis>,+rows: List<&2,Views.Axis>,
  +zero: Partial,+fuel: Nat,+outer: Nat,
  +scaled: {outer_stride == (extent * stride : U32) : U32},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(extent),kept_count(kept)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {joined(~Partial,ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat(extent),Nat.mul(outer,U32.to_nat(extent)),Outer{model,kept,rows,stride,zero,fuel})) ==
    outer_run(~Partial,~merge,Outer{model,Views.Axis{extent,stride} <> kept,rows,outer_stride,zero,fuel},outer) : List<&2,Partial>}:
  +offset = (U32.from_nat(outer) * outer_stride : U32)
  +whole = {Sub{model,Views.Axis{extent,stride} <> kept,rows,offset,zero,fuel} : Sub<Partial>}
  +count = Arithmetic.product_exact(Machine.capacity(limits),Limits.capacity_fits(limits),extent,Grid.product(Views.axis_extents(kept)),U32.to_nat(extent),kept_count(kept),{==},{==},room)
  +lines = {%Addition.zero_right(Nat.mul(outer,U32.to_nat(extent))) :
    {ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat(extent),_,Outer{model,kept,rows,stride,zero,fuel}) ==
      ListReadback.enumerate(~List<&2,Partial>,~Lined<Sub<Partial>>,~line_of(~Partial,~Sub<Partial>,~sub_output(~Partial,~merge)),U32.to_nat(extent),0n,Lined{whole,kept_count(kept)}) : List<&2,List<&2,Partial>>}
    enumerate_same(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),~Lined<Sub<Partial>>,~line_of(~Partial,~Sub<Partial>,~sub_output(~Partial,~merge)),~Outer{model,kept,rows,stride,zero,fuel},~Lined{whole,kept_count(kept)},
      ~Nat.mul(outer,U32.to_nat(extent)),~0n,~U32.to_nat(extent),
      ~(line => line_inside =>
        {%Addition.zero_right(Nat.mul(line,kept_count(kept))) :
          {ListReadback.enumerate(~Partial,~Sub<Partial>,~sub_output(~Partial,~merge),kept_count(kept),0n,Sub{model,kept,rows,(U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(extent)),line)) * stride : U32),zero,fuel}) ==
            ListReadback.enumerate(~Partial,~Sub<Partial>,~sub_output(~Partial,~merge),kept_count(kept),_,whole) : List<&2,Partial>}
          enumerate_same(~Partial,~Sub<Partial>,~sub_output(~Partial,~merge),~Sub<Partial>,~sub_output(~Partial,~merge),~Sub{model,kept,rows,(U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(extent)),line)) * stride : U32),zero,fuel},~whole,
            ~0n,~Nat.mul(line,kept_count(kept)),~kept_count(kept),
            ~(cell => cell_inside =>
              Equal.cong(U32,Partial,base => tree_of(~Partial,~merge,~Tree.Tree<Partial>,~model_read(~Partial),model,fuel,U32.is_le(first_extent(rows),1),rows,0,first_extent(rows),base,zero),
                address(kept,U32.from_nat(cell),(U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(extent)),line)) * stride : U32)),
                address(Views.Axis{extent,stride} <> kept,U32.from_nat(Nat.add(Nat.mul(line,kept_count(kept)),cell)),offset),
                Equal.trans(U32,address(kept,U32.from_nat(cell),(U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(extent)),line)) * stride : U32)),
                  address(kept,U32.from_nat(cell),(offset + U32.from_nat(line) * stride : U32)),
                  address(Views.Axis{extent,stride} <> kept,U32.from_nat(Nat.add(Nat.mul(line,kept_count(kept)),cell)),offset),
                  Equal.cong(U32,U32,word => address(kept,U32.from_nat(cell),word),(U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(extent)),line)) * stride : U32),(offset + U32.from_nat(line) * stride : U32),
                    line_offset(outer,line,extent,stride,outer_stride,scaled)),
                  Equal.sym(U32,address(Views.Axis{extent,stride} <> kept,U32.from_nat(Nat.add(Nat.mul(line,kept_count(kept)),cell)),offset),
                    address(kept,U32.from_nat(cell),(offset + U32.from_nat(line) * stride : U32)),
                    address_head(limits,extent,stride,kept,offset,line,cell,cell_inside,line_inside,room))))),
            kept_count(kept),0n,Order.reflexive(kept_count(kept)))
          : {ListReadback.enumerate(~Partial,~Sub<Partial>,~sub_output(~Partial,~merge),kept_count(kept),0n,Sub{model,kept,rows,(U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(extent)),line)) * stride : U32),zero,fuel}) ==
            ListReadback.enumerate(~Partial,~Sub<Partial>,~sub_output(~Partial,~merge),kept_count(kept),Nat.mul(line,kept_count(kept)),whole) : List<&2,Partial>}}),
      U32.to_nat(extent),0n,Order.reflexive(U32.to_nat(extent)))
    : {ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat(extent),Nat.mul(outer,U32.to_nat(extent)),Outer{model,kept,rows,stride,zero,fuel}) ==
      ListReadback.enumerate(~List<&2,Partial>,~Lined<Sub<Partial>>,~line_of(~Partial,~Sub<Partial>,~sub_output(~Partial,~merge)),U32.to_nat(extent),0n,Lined{whole,kept_count(kept)}) : List<&2,List<&2,Partial>>}}
  %Equal.sym(Nat,kept_count(Views.Axis{extent,stride} <> kept),Nat.mul(U32.to_nat(extent),kept_count(kept)),count) :
    {joined(~Partial,ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat(extent),Nat.mul(outer,U32.to_nat(extent)),Outer{model,kept,rows,stride,zero,fuel})) ==
      ListReadback.enumerate(~Partial,~Sub<Partial>,~sub_output(~Partial,~merge),_,0n,whole) : List<&2,Partial>}
  Equal.trans(List<&2,Partial>,joined(~Partial,ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat(extent),Nat.mul(outer,U32.to_nat(extent)),Outer{model,kept,rows,stride,zero,fuel})),
    joined(~Partial,ListReadback.enumerate(~List<&2,Partial>,~Lined<Sub<Partial>>,~line_of(~Partial,~Sub<Partial>,~sub_output(~Partial,~merge)),U32.to_nat(extent),0n,Lined{whole,kept_count(kept)})),
    ListReadback.enumerate(~Partial,~Sub<Partial>,~sub_output(~Partial,~merge),Nat.mul(U32.to_nat(extent),kept_count(kept)),0n,whole),
    Equal.cong(List<&2,List<&2,Partial>>,List<&2,Partial>,runs => joined(~Partial,runs),ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat(extent),Nat.mul(outer,U32.to_nat(extent)),Outer{model,kept,rows,stride,zero,fuel}),
      ListReadback.enumerate(~List<&2,Partial>,~Lined<Sub<Partial>>,~line_of(~Partial,~Sub<Partial>,~sub_output(~Partial,~merge)),U32.to_nat(extent),0n,Lined{whole,kept_count(kept)}),lines),
    Equal.sym(List<&2,Partial>,ListReadback.enumerate(~Partial,~Sub<Partial>,~sub_output(~Partial,~merge),Nat.mul(U32.to_nat(extent),kept_count(kept)),0n,whole),
      joined(~Partial,ListReadback.enumerate(~List<&2,Partial>,~Lined<Sub<Partial>>,~line_of(~Partial,~Sub<Partial>,~sub_output(~Partial,~merge)),U32.to_nat(extent),0n,Lined{whole,kept_count(kept)})),
      enumerate_lines(~Partial,~Sub<Partial>,~sub_output(~Partial,~merge),U32.to_nat(extent),0n,kept_count(kept),whole)))

# ---- Fuel ----

# Extents none zero, none past the words a split of the tree takes.
def Within(extents: List<&2,U32>,+cap: U32) -> Data:
  match extents:
    case Nil{}: Unit
    case Con{+extent,rest}: KernelGemm.Both<{Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},KernelGemm.Both<{Nat.is_le(U32.to_nat(extent),U32.to_nat(cap)) == True{} : Bool},Within(rest,cap)>>

# The fuel a tree takes over axes, its first count at most two to the bits.
def fuel_needed(axes: List<&2,Views.Axis>,+bits: Nat) -> Nat:
  match axes:
    case Nil{}: 1n
    case Con{axis,tail}: Nat.add(bits,1n+fuel_needed(tail,30n))

def needed_positive(axes: List<&2,Views.Axis>,+bits: Nat) -> {Nat.is_lt(0n,fuel_needed(axes,bits)) == True{} : Bool}:
  match axes:
    case Nil{}: {==}
    case Con{+axis,+tail}:
      %Equal.sym(Nat,Nat.add(bits,1n+fuel_needed(tail,30n)),1n+Nat.add(bits,fuel_needed(tail,30n)),Addition.successor_right(bits,fuel_needed(tail,30n))) :
        {Nat.is_lt(0n,_) == True{} : Bool}
      {==}

def fuel_after(+bits: Nat,+tail: Nat,+fuel: Nat,+enough: {Nat.is_le(Nat.add(bits,1n+tail),1n+fuel) == True{} : Bool}) -> {Nat.is_le(tail,fuel) == True{} : Bool}:
  Order.transitive(tail,Nat.add(bits,tail),fuel,Order.right_le(bits,tail),
    Held.nat_le_moved(1n+Nat.add(bits,tail),Nat.add(bits,1n+tail),1n+fuel,Equal.sym(Nat,Nat.add(bits,1n+tail),1n+Nat.add(bits,tail),Addition.successor_right(bits,tail)),enough))

def fuel_split(+low: Nat,+bits: Nat,+tail: Nat,+fuel: Nat,+ordered: {Nat.is_le(low,bits) == True{} : Bool},+enough: {Nat.is_le(Nat.add(1n+bits,1n+tail),1n+fuel) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(low,1n+tail),fuel) == True{} : Bool}:
  Order.transitive(Nat.add(low,1n+tail),Nat.add(bits,1n+tail),fuel,Order.add_le(low,bits,1n+tail,ordered),enough)

def power_room(+limits: Machine.Limits,+value: Nat,+inside: {Nat.is_le(value,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {Nat.is_le(value,Bits.power(30n)) == True{} : Bool}:
  Held.nat_le_right_moved(value,Bits.power(30n),U32.to_nat(U32.shln(1,30n)),Equal.sym(Nat,U32.to_nat(U32.shln(1,30n)),Bits.power(30n),GridRanges.shifted_power(30n,{==})),
    Order.transitive(value,U32.to_nat(Machine.capacity(limits)),U32.to_nat(U32.shln(1,30n)),inside,Bounds.as_nat_le(Machine.capacity(limits),U32.shln(1,30n),Limits.capacity_allocatable(limits))))

def word_room(+limits: Machine.Limits,+count: U32,+small: {Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {U32.is_le(count,1073741824) == True{} : Bool}:
  Order.word_le(count,Machine.capacity(limits),1073741824,
    Equal.trans(Bool,U32.is_le(count,Machine.capacity(limits)),Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(limits))),True{},Comparison.le(count,Machine.capacity(limits)),small),
    Limits.capacity_allocatable(limits))

def first_positive(axes: List<&2,Views.Axis>,+cap: U32,+within: Within(Views.axis_extents(axes),cap)) -> {Nat.is_lt(0n,U32.to_nat(first_extent(axes))) == True{} : Bool}:
  match axes:
    case Nil{}: {==}
    case Con{Views.Axis{+extent,+stride},+rest}:
      KernelGemm.Both{+positive,+others} = within
      positive

def first_small(axes: List<&2,Views.Axis>,+limits: Machine.Limits,+within: Within(Views.axis_extents(axes),Machine.capacity(limits))) ->
  {Nat.is_le(U32.to_nat(first_extent(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  match axes:
    case Nil{}: Bounds.as_nat_le(1,Machine.capacity(limits),Order.word_le(1,16,Machine.capacity(limits),{==},Limits.sixteen_within_capacity(limits)))
    case Con{Views.Axis{+extent,+stride},+rest}:
      KernelGemm.Both{+positive,+others} = within
      KernelGemm.Both{+small,+later} = others
      small

# A tree over axes reads the same with any fuel past what it takes.
def tree_fuel(~Partial: Data,~merge: Partial -> Partial -> Partial,~Context: Data,~read: Context -> U32 -> Partial,+context: Context,+low: Nat,+high: Nat,+single: Bool,+axes: List<&2,Views.Axis>,
  +limits: Machine.Limits,+first: U32,+count: U32,+base: U32,+zero: Partial,+bits: Nat,
  +test: {U32.is_le(count,1) == single : Bool},
  +fits: {Nat.is_le(U32.to_nat(count),Bits.power(bits)) == True{} : Bool},
  +small: {Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +within: Within(Views.axis_extents(axes),Machine.capacity(limits)),
  +low_enough: {Nat.is_le(fuel_needed(axes,bits),low) == True{} : Bool},
  +high_enough: {Nat.is_le(fuel_needed(axes,bits),high) == True{} : Bool}) ->
  {tree_of(~Partial,~merge,~Context,~read,context,low,single,axes,first,count,base,zero) == tree_of(~Partial,~merge,~Context,~read,context,high,single,axes,first,count,base,zero) : Partial}:
  match low:
    case 0n:
      Empty.absurd({tree_of(~Partial,~merge,~Context,~read,context,0n,single,axes,first,count,base,zero) == tree_of(~Partial,~merge,~Context,~read,context,high,single,axes,first,count,base,zero) : Partial},
        WordEncoding.false_true(Order.lt_le(0n,fuel_needed(axes,bits),0n,needed_positive(axes,bits),low_enough)))
    case 1n+ +lower:
      match high:
        case 0n:
          Empty.absurd({tree_of(~Partial,~merge,~Context,~read,context,1n+lower,single,axes,first,count,base,zero) == tree_of(~Partial,~merge,~Context,~read,context,0n,single,axes,first,count,base,zero) : Partial},
            WordEncoding.false_true(Order.lt_le(0n,fuel_needed(axes,bits),0n,needed_positive(axes,bits),high_enough)))
        case 1n+ +higher:
          match single:
            case True{}:
              match axes:
                case []: {==}
                case Con{Views.Axis{+extent,+stride},+tail}:
                  KernelGemm.Both{+head_positive,+others} = within
                  KernelGemm.Both{+head_small,+inner} = others
                  +next = first_extent(tail)
                  tree_fuel(~Partial,~merge,~Context,~read,context,lower,higher,U32.is_le(next,1),tail,limits,0,next,(base + first * stride : U32),zero,30n,{==},
                    power_room(limits,U32.to_nat(next),first_small(tail,limits,inner)),first_small(tail,limits,inner),inner,
                    fuel_after(bits,fuel_needed(tail,30n),lower,low_enough),fuel_after(bits,fuel_needed(tail,30n),higher,high_enough))
            case False{}:
              match axes:
                case []: {==}
                case Con{Views.Axis{+extent,+stride},+tail}:
                  match bits:
                    case 0n:
                      +more = big_count(count,test)
                      Empty.absurd({tree_of(~Partial,~merge,~Context,~read,context,1n+lower,False{},Views.Axis{extent,stride} <> tail,first,count,base,zero) ==
                        tree_of(~Partial,~merge,~Context,~read,context,1n+higher,False{},Views.Axis{extent,stride} <> tail,first,count,base,zero) : Partial},
                        WordEncoding.false_true(Order.lt_le(1n,U32.to_nat(count),1n,more,fits)))
                    case 1n+ +previous:
                      +more = big_count(count,test)
                      +halves = GridRanges.below_facts(count,more,word_room(limits,count,small))
                      +exact = GridRanges.below_exact(count,halves)
                      +under = GridRanges.below_under(count,halves)
                      +cover = GridRanges.below_cover(count,halves)
                      +short = Order.strict_follow(GridRanges.half_depth(count),1n+previous,power_lt_inv(GridRanges.half_depth(count),1n+previous,Order.lt_le(Bits.power(GridRanges.half_depth(count)),U32.to_nat(count),Bits.power(1n+previous),under,fits)))
                      +nl_under = {%Equal.sym(Nat,U32.to_nat(Pieces.below(count)),Bits.power(GridRanges.half_depth(count)),exact) : {Nat.is_lt(_,U32.to_nat(count)) == True{} : Bool} under : {Nat.is_lt(U32.to_nat(Pieces.below(count)),U32.to_nat(count)) == True{} : Bool}}
                      +ordered = Bounds.strict_le(U32.to_nat(Pieces.below(count)),U32.to_nat(count),nl_under)
                      +right = Arithmetic.subtract(count,Pieces.below(count),count,Encoded.native_le_transitive(count,1073741824,2147483648,word_room(limits,count,small),{==}),ordered,Order.sub_le(U32.to_nat(count),U32.to_nat(Pieces.below(count))))
                      +right_fits = Held.nat_le_moved(U32.to_nat((count - Pieces.below(count) : U32)),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count))),Bits.power(GridRanges.half_depth(count)),right,
                        {%Equal.sym(Nat,U32.to_nat(Pieces.below(count)),Bits.power(GridRanges.half_depth(count)),exact) : {Nat.is_le(Nat.sub(U32.to_nat(count),_),Bits.power(GridRanges.half_depth(count))) == True{} : Bool}
                          GridRanges.past_half(U32.to_nat(count),Bits.power(GridRanges.half_depth(count)),Bounds.strict_le(Bits.power(GridRanges.half_depth(count)),U32.to_nat(count),under),cover) : {Nat.is_le(Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count))),Bits.power(GridRanges.half_depth(count))) == True{} : Bool}})
                      +left_small = Order.transitive(U32.to_nat(Pieces.below(count)),U32.to_nat(count),U32.to_nat(Machine.capacity(limits)),ordered,small)
                      +right_small = Order.transitive(U32.to_nat((count - Pieces.below(count) : U32)),U32.to_nat(count),U32.to_nat(Machine.capacity(limits)),Held.nat_le_moved(U32.to_nat((count - Pieces.below(count) : U32)),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count))),U32.to_nat(count),right,Order.sub_le(U32.to_nat(count),U32.to_nat(Pieces.below(count)))),small)
                      Equal.trans(Partial,
                        merge(tree_of(~Partial,~merge,~Context,~read,context,lower,U32.is_le(Pieces.below(count),1),Views.Axis{extent,stride} <> tail,first,Pieces.below(count),base,zero),
                          tree_of(~Partial,~merge,~Context,~read,context,lower,U32.is_le((count - Pieces.below(count) : U32),1),Views.Axis{extent,stride} <> tail,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32),base,zero)),
                        merge(tree_of(~Partial,~merge,~Context,~read,context,higher,U32.is_le(Pieces.below(count),1),Views.Axis{extent,stride} <> tail,first,Pieces.below(count),base,zero),
                          tree_of(~Partial,~merge,~Context,~read,context,lower,U32.is_le((count - Pieces.below(count) : U32),1),Views.Axis{extent,stride} <> tail,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32),base,zero)),
                        merge(tree_of(~Partial,~merge,~Context,~read,context,higher,U32.is_le(Pieces.below(count),1),Views.Axis{extent,stride} <> tail,first,Pieces.below(count),base,zero),
                          tree_of(~Partial,~merge,~Context,~read,context,higher,U32.is_le((count - Pieces.below(count) : U32),1),Views.Axis{extent,stride} <> tail,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32),base,zero)),
                        Equal.cong(Partial,Partial,low_tree => merge(low_tree,tree_of(~Partial,~merge,~Context,~read,context,lower,U32.is_le((count - Pieces.below(count) : U32),1),Views.Axis{extent,stride} <> tail,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32),base,zero)),
                          tree_of(~Partial,~merge,~Context,~read,context,lower,U32.is_le(Pieces.below(count),1),Views.Axis{extent,stride} <> tail,first,Pieces.below(count),base,zero),
                          tree_of(~Partial,~merge,~Context,~read,context,higher,U32.is_le(Pieces.below(count),1),Views.Axis{extent,stride} <> tail,first,Pieces.below(count),base,zero),
                          tree_fuel(~Partial,~merge,~Context,~read,context,lower,higher,U32.is_le(Pieces.below(count),1),Views.Axis{extent,stride} <> tail,limits,first,Pieces.below(count),base,zero,GridRanges.half_depth(count),{==},
                            Held.nat_le_moved(U32.to_nat(Pieces.below(count)),Bits.power(GridRanges.half_depth(count)),Bits.power(GridRanges.half_depth(count)),exact,Order.reflexive(Bits.power(GridRanges.half_depth(count)))),left_small,within,
                            fuel_split(GridRanges.half_depth(count),previous,fuel_needed(tail,30n),lower,short,low_enough),fuel_split(GridRanges.half_depth(count),previous,fuel_needed(tail,30n),higher,short,high_enough))),
                        Equal.cong(Partial,Partial,high_tree => merge(tree_of(~Partial,~merge,~Context,~read,context,higher,U32.is_le(Pieces.below(count),1),Views.Axis{extent,stride} <> tail,first,Pieces.below(count),base,zero),high_tree),
                          tree_of(~Partial,~merge,~Context,~read,context,lower,U32.is_le((count - Pieces.below(count) : U32),1),Views.Axis{extent,stride} <> tail,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32),base,zero),
                          tree_of(~Partial,~merge,~Context,~read,context,higher,U32.is_le((count - Pieces.below(count) : U32),1),Views.Axis{extent,stride} <> tail,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32),base,zero),
                          tree_fuel(~Partial,~merge,~Context,~read,context,lower,higher,U32.is_le((count - Pieces.below(count) : U32),1),Views.Axis{extent,stride} <> tail,limits,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32),base,zero,GridRanges.half_depth(count),{==},
                            right_fits,right_small,within,
                            fuel_split(GridRanges.half_depth(count),previous,fuel_needed(tail,30n),lower,short,low_enough),fuel_split(GridRanges.half_depth(count),previous,fuel_needed(tail,30n),higher,short,high_enough))))

def take_enumerate(~Element: Data,~Context: Data,~value: Context -> Nat -> Element,remaining: Nat,taken: Nat,+first: Nat,+context: Context,+inside: {Nat.is_le(taken,remaining) == True{} : Bool}) ->
  {List.take(&2,Element,ListReadback.enumerate(~Element,~Context,~value,remaining,first,context),taken) == ListReadback.enumerate(~Element,~Context,~value,taken,first,context) : List<&2,Element>}:
  match remaining:
    case 0n:
      match taken:
        case 0n: {==}
        case 1n+ +more:
          Empty.absurd({List.take(&2,Element,[],1n+more) == ListReadback.enumerate(~Element,~Context,~value,1n+more,first,context) : List<&2,Element>},WordEncoding.false_true(inside))
    case 1n+ +rest:
      match taken:
        case 0n: {==}
        case 1n+ +more:
          Equal.cong(List<&2,Element>,List<&2,Element>,tail => value(context,first) <> tail,
            List.take(&2,Element,ListReadback.enumerate(~Element,~Context,~value,rest,1n+first,context),more),ListReadback.enumerate(~Element,~Context,~value,more,1n+first,context),
            take_enumerate(~Element,~Context,~value,rest,more,1n+first,context,inside))

def drop_enumerate(~Element: Data,~Context: Data,~value: Context -> Nat -> Element,remaining: Nat,dropped: Nat,+first: Nat,+context: Context) ->
  {List.drop(&2,Element,ListReadback.enumerate(~Element,~Context,~value,remaining,first,context),dropped) ==
    ListReadback.enumerate(~Element,~Context,~value,Nat.sub(remaining,dropped),Nat.add(first,dropped),context) : List<&2,Element>}:
  match remaining:
    case 0n:
      %Equal.sym(Nat,Nat.sub(0n,dropped),0n,Order.zero_subtract(dropped)) :
        {List.drop(&2,Element,[],dropped) == ListReadback.enumerate(~Element,~Context,~value,_,Nat.add(first,dropped),context) : List<&2,Element>}
      {==}
    case 1n+ +rest:
      match dropped:
        case 0n:
          %Equal.sym(Nat,Nat.add(first,0n),first,Addition.zero_right(first)) :
            {ListReadback.enumerate(~Element,~Context,~value,1n+rest,first,context) == ListReadback.enumerate(~Element,~Context,~value,1n+rest,_,context) : List<&2,Element>}
          {==}
        case 1n+ +more:
          %Equal.sym(Nat,Nat.add(first,1n+more),1n+Nat.add(first,more),Addition.successor_right(first,more)) :
            {List.drop(&2,Element,ListReadback.enumerate(~Element,~Context,~value,rest,1n+first,context),more) == ListReadback.enumerate(~Element,~Context,~value,Nat.sub(rest,more),_,context) : List<&2,Element>}
          drop_enumerate(~Element,~Context,~value,rest,more,1n+first,context)

# The outputs of a split of the tree over the first axis.
type Split<-Partial: Data> is Data:
  Split{model: Tree.Tree<Partial>,kept: List<&2,Views.Axis>,axes: List<&2,Views.Axis>,offset: U32,zero: Partial,fuel: Nat,single: Bool,first: U32,count: U32}

def split_output(~Partial: Data,~merge: Partial -> Partial -> Partial,split: Split<Partial>,index: Nat) -> Partial:
  Split{+model,+kept,+axes,+offset,+zero,+fuel,+single,+first,+count} = split
  tree_of(~Partial,~merge,~Tree.Tree<Partial>,~model_read(~Partial),model,fuel,single,axes,first,count,address(kept,U32.from_nat(index),offset),zero)

def merged_splits(~Partial: Data,~merge: Partial -> Partial -> Partial,remaining: Nat,+index: Nat,+model: Tree.Tree<Partial>,+kept: List<&2,Views.Axis>,+extent: U32,+stride: U32,+rows: List<&2,Views.Axis>,+offset: U32,+zero: Partial,
  +fuel: Nat,+first: U32,+count: U32) ->
  {Partials.merged_runs(~Partial,~merge,
      ListReadback.enumerate(~Partial,~Split<Partial>,~split_output(~Partial,~merge),remaining,index,Split{model,kept,Con{Views.Axis{extent,stride},rows},offset,zero,fuel,U32.is_le(Pieces.below(count),1),first,Pieces.below(count)}),
      ListReadback.enumerate(~Partial,~Split<Partial>,~split_output(~Partial,~merge),remaining,index,Split{model,kept,Con{Views.Axis{extent,stride},rows},offset,zero,fuel,U32.is_le((count - Pieces.below(count) : U32),1),(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)})) ==
    ListReadback.enumerate(~Partial,~Split<Partial>,~split_output(~Partial,~merge),remaining,index,Split{model,kept,Con{Views.Axis{extent,stride},rows},offset,zero,1n+fuel,False{},first,count}) : List<&2,Partial>}:
  match remaining:
    case 0n: {==}
    case 1n+ +more:
      Equal.cong(List<&2,Partial>,List<&2,Partial>,tail => split_output(~Partial,~merge,Split{model,kept,Con{Views.Axis{extent,stride},rows},offset,zero,1n+fuel,False{},first,count},index) <> tail,
        Partials.merged_runs(~Partial,~merge,
          ListReadback.enumerate(~Partial,~Split<Partial>,~split_output(~Partial,~merge),more,1n+index,Split{model,kept,Con{Views.Axis{extent,stride},rows},offset,zero,fuel,U32.is_le(Pieces.below(count),1),first,Pieces.below(count)}),
          ListReadback.enumerate(~Partial,~Split<Partial>,~split_output(~Partial,~merge),more,1n+index,Split{model,kept,Con{Views.Axis{extent,stride},rows},offset,zero,fuel,U32.is_le((count - Pieces.below(count) : U32),1),(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)})),
        ListReadback.enumerate(~Partial,~Split<Partial>,~split_output(~Partial,~merge),more,1n+index,Split{model,kept,Con{Views.Axis{extent,stride},rows},offset,zero,1n+fuel,False{},first,count}),
        merged_splits(~Partial,~merge,more,1n+index,model,kept,extent,stride,rows,offset,zero,fuel,first,count))

def one_count(+count: U32,+test: {U32.is_le(count,1) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}) -> {U32.to_nat(count) == 1n : Nat}:
  Order.antisymmetric(U32.to_nat(count),1n,Equal.trans(Bool,Nat.is_le(U32.to_nat(count),1n),U32.is_le(count,1),True{},Equal.sym(Bool,U32.is_le(count,1),Nat.is_le(U32.to_nat(count),U32.to_nat(1)),Comparison.le(count,1)),test),
    Order.positive_one(U32.to_nat(count),positive))

# The right half of a split ends where the whole does.
def right_range(+first: Nat,+left: Nat,+count: Nat,+extent: Nat,+left_le: {Nat.is_le(left,count) == True{} : Bool},+range: {Nat.is_le(Nat.add(first,count),extent) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Nat.add(first,left),Nat.sub(count,left)),extent) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(Nat.add(first,left),Nat.sub(count,left)),Nat.add(first,Nat.add(left,Nat.sub(count,left))),Addition.associative(first,left,Nat.sub(count,left))) :
    {Nat.is_le(_,extent) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(left,Nat.sub(count,left)),count,Order.restore_shift(left,count,left_le)) : {Nat.is_le(Nat.add(first,_),extent) == True{} : Bool}
  range

# The tree of a line of runs through the counter is the tree of the axis of
# the line over its outputs.
def line_tree_of(~Partial: Data,~merge: Partial -> Partial -> Partial,+depth: Nat,+single: Bool,+fuel: Nat,+bits: Nat,+limits: Machine.Limits,+model: Tree.Tree<Partial>,+kept: List<&2,Views.Axis>,+rows: List<&2,Views.Axis>,
  +extent: U32,+stride: U32,+outer_stride: U32,+zero: Partial,+outer: Nat,+settled: Nat,+count: U32,+first: U32,
  +test: {U32.is_le(count,1) == single : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
  +fits: {Nat.is_le(U32.to_nat(count),Bits.power(bits)) == True{} : Bool},
  +range: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(extent)) == True{} : Bool},
  +deep: {Nat.is_lt(bits,depth) == True{} : Bool},
  +enough: {Nat.is_le(Nat.add(bits,1n+fuel_needed(rows,30n)),fuel) == True{} : Bool},
  +settled_enough: {Nat.is_le(fuel_needed(rows,30n),settled) == True{} : Bool},
  +scaled: {outer_stride == (extent * stride : U32) : U32},
  +within: Within(Views.axis_extents(rows),Machine.capacity(limits)),
  +room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {line_tree(~Partial,~merge,depth,single,count,ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat(count),Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat(first)),Outer{model,kept,rows,stride,zero,settled})) ==
    ListReadback.enumerate(~Partial,~Split<Partial>,~split_output(~Partial,~merge),kept_count(kept),0n,Split{model,kept,Con{Views.Axis{extent,stride},rows},(U32.from_nat(outer) * outer_stride : U32),zero,fuel,single,first,count}) : List<&2,Partial>}:
  match depth:
    case 0n:
      Empty.absurd({line_tree(~Partial,~merge,0n,single,count,ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat(count),Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat(first)),Outer{model,kept,rows,stride,zero,settled})) ==
        ListReadback.enumerate(~Partial,~Split<Partial>,~split_output(~Partial,~merge),kept_count(kept),0n,Split{model,kept,Con{Views.Axis{extent,stride},rows},(U32.from_nat(outer) * outer_stride : U32),zero,fuel,single,first,count}) : List<&2,Partial>},Order.below_zero(bits,deep))
    case 1n+ +rest:
      match single:
        case True{}:
          match fuel:
            case 0n:
              Empty.absurd({line_tree(~Partial,~merge,1n+rest,True{},count,ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat(count),Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat(first)),Outer{model,kept,rows,stride,zero,settled})) ==
                ListReadback.enumerate(~Partial,~Split<Partial>,~split_output(~Partial,~merge),kept_count(kept),0n,Split{model,kept,Con{Views.Axis{extent,stride},rows},(U32.from_nat(outer) * outer_stride : U32),zero,0n,True{},first,count}) : List<&2,Partial>},
                WordEncoding.false_true(Order.lt_le(0n,fuel_needed(Con{Views.Axis{extent,stride},rows},bits),0n,needed_positive(Con{Views.Axis{extent,stride},rows},bits),enough)))
            case 1n+ +lower:
              +next = first_extent(rows)
              +offset = (U32.from_nat(outer) * outer_stride : U32)
              %Equal.sym(Nat,U32.to_nat(count),1n,one_count(count,test,positive)) :
                {line_tree(~Partial,~merge,1n+rest,True{},count,ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),_,Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat(first)),Outer{model,kept,rows,stride,zero,settled})) ==
                  ListReadback.enumerate(~Partial,~Split<Partial>,~split_output(~Partial,~merge),kept_count(kept),0n,Split{model,kept,Con{Views.Axis{extent,stride},rows},offset,zero,1n+lower,True{},first,count}) : List<&2,Partial>}
              enumerate_same(~Partial,~Sub<Partial>,~sub_output(~Partial,~merge),~Split<Partial>,~split_output(~Partial,~merge),~Sub{model,kept,rows,(U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat(first))) * stride : U32),zero,settled},
                ~Split{model,kept,Con{Views.Axis{extent,stride},rows},offset,zero,1n+lower,True{},first,count},~0n,~0n,~kept_count(kept),
                ~(cell => cell_inside =>
                  Equal.trans(Partial,
                    tree_of(~Partial,~merge,~Tree.Tree<Partial>,~model_read(~Partial),model,settled,U32.is_le(next,1),rows,0,next,address(kept,U32.from_nat(cell),(U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat(first))) * stride : U32)),zero),
                    tree_of(~Partial,~merge,~Tree.Tree<Partial>,~model_read(~Partial),model,settled,U32.is_le(next,1),rows,0,next,(address(kept,U32.from_nat(cell),offset) + first * stride : U32),zero),
                    tree_of(~Partial,~merge,~Tree.Tree<Partial>,~model_read(~Partial),model,lower,U32.is_le(next,1),rows,0,next,(address(kept,U32.from_nat(cell),offset) + first * stride : U32),zero),
                    Equal.cong(U32,Partial,base => tree_of(~Partial,~merge,~Tree.Tree<Partial>,~model_read(~Partial),model,settled,U32.is_le(next,1),rows,0,next,base,zero),
                      address(kept,U32.from_nat(cell),(U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat(first))) * stride : U32)),(address(kept,U32.from_nat(cell),offset) + first * stride : U32),
                      Equal.trans(U32,address(kept,U32.from_nat(cell),(U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat(first))) * stride : U32)),address(kept,U32.from_nat(cell),(offset + first * stride : U32)),
                        (address(kept,U32.from_nat(cell),offset) + first * stride : U32),
                        Equal.cong(U32,U32,word => address(kept,U32.from_nat(cell),word),(U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat(first))) * stride : U32),(offset + first * stride : U32),
                          Equal.trans(U32,(U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat(first))) * stride : U32),(offset + U32.from_nat(U32.to_nat(first)) * stride : U32),(offset + first * stride : U32),
                            line_offset(outer,U32.to_nat(first),extent,stride,outer_stride,scaled),
                            Equal.cong(U32,U32,word => (offset + word * stride : U32),U32.from_nat(U32.to_nat(first)),first,Arithmetic.native_roundtrip(first)))),
                        address_shift(kept,U32.from_nat(cell),offset,(first * stride : U32)))),
                    tree_fuel(~Partial,~merge,~Tree.Tree<Partial>,~model_read(~Partial),model,settled,lower,U32.is_le(next,1),rows,limits,0,next,(address(kept,U32.from_nat(cell),offset) + first * stride : U32),zero,30n,{==},
                      power_room(limits,U32.to_nat(next),first_small(rows,limits,within)),first_small(rows,limits,within),within,settled_enough,
                      fuel_after(bits,fuel_needed(rows,30n),lower,enough)))),
                kept_count(kept),0n,Order.reflexive(kept_count(kept)))
        case False{}:
          match fuel:
            case 0n:
              Empty.absurd({line_tree(~Partial,~merge,1n+rest,False{},count,ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat(count),Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat(first)),Outer{model,kept,rows,stride,zero,settled})) ==
                ListReadback.enumerate(~Partial,~Split<Partial>,~split_output(~Partial,~merge),kept_count(kept),0n,Split{model,kept,Con{Views.Axis{extent,stride},rows},(U32.from_nat(outer) * outer_stride : U32),zero,0n,False{},first,count}) : List<&2,Partial>},
                WordEncoding.false_true(Order.lt_le(0n,fuel_needed(Con{Views.Axis{extent,stride},rows},bits),0n,needed_positive(Con{Views.Axis{extent,stride},rows},bits),enough)))
            case 1n+ +lower:
              match bits:
                case 0n:
                  +more = big_count(count,test)
                  Empty.absurd({line_tree(~Partial,~merge,1n+rest,False{},count,ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat(count),Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat(first)),Outer{model,kept,rows,stride,zero,settled})) ==
                    ListReadback.enumerate(~Partial,~Split<Partial>,~split_output(~Partial,~merge),kept_count(kept),0n,Split{model,kept,Con{Views.Axis{extent,stride},rows},(U32.from_nat(outer) * outer_stride : U32),zero,1n+lower,False{},first,count}) : List<&2,Partial>},
                    WordEncoding.false_true(Order.lt_le(1n,U32.to_nat(count),1n,more,fits)))
                case 1n+ +previous:
                  +more = big_count(count,test)
                  +small = Order.transitive(U32.to_nat(count),Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(Machine.capacity(limits)),Order.right_le(U32.to_nat(first),U32.to_nat(count)),Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(extent),U32.to_nat(Machine.capacity(limits)),range,room))
                  +halves = GridRanges.below_facts(count,more,word_room(limits,count,small))
                  +exact = GridRanges.below_exact(count,halves)
                  +under = GridRanges.below_under(count,halves)
                  +cover = GridRanges.below_cover(count,halves)
                  +short = Order.strict_follow(GridRanges.half_depth(count),1n+previous,power_lt_inv(GridRanges.half_depth(count),1n+previous,Order.lt_le(Bits.power(GridRanges.half_depth(count)),U32.to_nat(count),Bits.power(1n+previous),under,fits)))
                  +shallow = Order.le_lt_trans(GridRanges.half_depth(count),previous,rest,short,deep)
                  +nl_under = {%Equal.sym(Nat,U32.to_nat(Pieces.below(count)),Bits.power(GridRanges.half_depth(count)),exact) : {Nat.is_lt(_,U32.to_nat(count)) == True{} : Bool} under : {Nat.is_lt(U32.to_nat(Pieces.below(count)),U32.to_nat(count)) == True{} : Bool}}
                  +ordered = Bounds.strict_le(U32.to_nat(Pieces.below(count)),U32.to_nat(count),nl_under)
                  +right = Arithmetic.subtract(count,Pieces.below(count),count,Encoded.native_le_transitive(count,1073741824,2147483648,word_room(limits,count,small),{==}),ordered,Order.sub_le(U32.to_nat(count),U32.to_nat(Pieces.below(count))))
                  +right_fits = Held.nat_le_moved(U32.to_nat((count - Pieces.below(count) : U32)),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count))),Bits.power(GridRanges.half_depth(count)),right,
                    {%Equal.sym(Nat,U32.to_nat(Pieces.below(count)),Bits.power(GridRanges.half_depth(count)),exact) : {Nat.is_le(Nat.sub(U32.to_nat(count),_),Bits.power(GridRanges.half_depth(count))) == True{} : Bool}
                      GridRanges.past_half(U32.to_nat(count),Bits.power(GridRanges.half_depth(count)),Bounds.strict_le(Bits.power(GridRanges.half_depth(count)),U32.to_nat(count),under),cover) : {Nat.is_le(Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count))),Bits.power(GridRanges.half_depth(count))) == True{} : Bool}})
                  +left_range = Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(extent),Order.offset_le(U32.to_nat(first),U32.to_nat(Pieces.below(count)),U32.to_nat(count),ordered),range)
                  +left_end = Arithmetic.add(first,Pieces.below(count),Machine.capacity(limits),Limits.capacity_fits(limits),Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),U32.to_nat(extent),U32.to_nat(Machine.capacity(limits)),left_range,room))
                  +right_range_nat = right_range(U32.to_nat(first),U32.to_nat(Pieces.below(count)),U32.to_nat(count),U32.to_nat(extent),ordered,range)
                  +left_tree = line_tree_of(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),lower,GridRanges.half_depth(count),limits,model,kept,rows,extent,stride,outer_stride,zero,outer,settled,Pieces.below(count),first,{==},
                    Held.nat_lt_moved(0n,U32.to_nat(Pieces.below(count)),Bits.power(GridRanges.half_depth(count)),exact,GridRanges.power_positive(GridRanges.half_depth(count))),Held.nat_le_moved(U32.to_nat(Pieces.below(count)),Bits.power(GridRanges.half_depth(count)),Bits.power(GridRanges.half_depth(count)),exact,Order.reflexive(Bits.power(GridRanges.half_depth(count)))),left_range,shallow,
                    fuel_split(GridRanges.half_depth(count),previous,fuel_needed(rows,30n),lower,short,enough),settled_enough,scaled,within,room)
                  +right_tree = line_tree_of(~Partial,~merge,rest,U32.is_le((count - Pieces.below(count) : U32),1),lower,GridRanges.half_depth(count),limits,model,kept,rows,extent,stride,outer_stride,zero,outer,settled,(count - Pieces.below(count) : U32),(first + Pieces.below(count) : U32),{==},
                    Held.nat_lt_moved(0n,U32.to_nat((count - Pieces.below(count) : U32)),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count))),right,Order.positive_difference(U32.to_nat(count),U32.to_nat(Pieces.below(count)),nl_under)),right_fits,
                    Held.nat_le_moved(Nat.add(U32.to_nat((first + Pieces.below(count) : U32)),U32.to_nat((count - Pieces.below(count) : U32))),Nat.add(Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count)))),U32.to_nat(extent),
                      Equal.trans(Nat,Nat.add(U32.to_nat((first + Pieces.below(count) : U32)),U32.to_nat((count - Pieces.below(count) : U32))),Nat.add(Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),U32.to_nat((count - Pieces.below(count) : U32))),Nat.add(Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count)))),
                        Equal.cong(Nat,Nat,start => Nat.add(start,U32.to_nat((count - Pieces.below(count) : U32))),U32.to_nat((first + Pieces.below(count) : U32)),Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),left_end),
                        Equal.cong(Nat,Nat,size => Nat.add(Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),size),U32.to_nat((count - Pieces.below(count) : U32)),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count))),right)),
                      right_range_nat),
                    shallow,fuel_split(GridRanges.half_depth(count),previous,fuel_needed(rows,30n),lower,short,enough),settled_enough,scaled,within,room)
                  +dropped = Equal.trans(List<&2,List<&2,Partial>>,List.drop(&2,List<&2,Partial>,ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat(count),Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat(first)),Outer{model,kept,rows,stride,zero,settled}),U32.to_nat(Pieces.below(count))),
                    ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count))),Nat.add(Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat(first)),U32.to_nat(Pieces.below(count))),Outer{model,kept,rows,stride,zero,settled}),
                    ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat((count - Pieces.below(count) : U32)),Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat((first + Pieces.below(count) : U32))),Outer{model,kept,rows,stride,zero,settled}),
                    drop_enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat(count),U32.to_nat(Pieces.below(count)),Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat(first)),Outer{model,kept,rows,stride,zero,settled}),
                    Equal.trans(List<&2,List<&2,Partial>>,ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count))),Nat.add(Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat(first)),U32.to_nat(Pieces.below(count))),Outer{model,kept,rows,stride,zero,settled}),
                      ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat((count - Pieces.below(count) : U32)),Nat.add(Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat(first)),U32.to_nat(Pieces.below(count))),Outer{model,kept,rows,stride,zero,settled}),
                      ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat((count - Pieces.below(count) : U32)),Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat((first + Pieces.below(count) : U32))),Outer{model,kept,rows,stride,zero,settled}),
                      Equal.cong(Nat,List<&2,List<&2,Partial>>,size => ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),size,Nat.add(Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat(first)),U32.to_nat(Pieces.below(count))),Outer{model,kept,rows,stride,zero,settled}),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count))),U32.to_nat((count - Pieces.below(count) : U32)),
                        Equal.sym(Nat,U32.to_nat((count - Pieces.below(count) : U32)),Nat.sub(U32.to_nat(count),U32.to_nat(Pieces.below(count))),right)),
                      Equal.cong(Nat,List<&2,List<&2,Partial>>,start => ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat((count - Pieces.below(count) : U32)),start,Outer{model,kept,rows,stride,zero,settled}),
                        Nat.add(Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat(first)),U32.to_nat(Pieces.below(count))),Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat((first + Pieces.below(count) : U32))),
                        Equal.trans(Nat,Nat.add(Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat(first)),U32.to_nat(Pieces.below(count))),Nat.add(Nat.mul(outer,U32.to_nat(extent)),Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count)))),Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat((first + Pieces.below(count) : U32))),
                          Addition.associative(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat(first),U32.to_nat(Pieces.below(count))),
                          Equal.cong(Nat,Nat,start => Nat.add(Nat.mul(outer,U32.to_nat(extent)),start),Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),U32.to_nat((first + Pieces.below(count) : U32)),
                            Equal.sym(Nat,U32.to_nat((first + Pieces.below(count) : U32)),Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),left_end))))))
                  %Equal.sym(List<&2,List<&2,Partial>>,List.take(&2,List<&2,Partial>,ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat(count),Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat(first)),Outer{model,kept,rows,stride,zero,settled}),U32.to_nat(Pieces.below(count))),ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat(Pieces.below(count)),Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat(first)),Outer{model,kept,rows,stride,zero,settled}),
                    take_enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat(count),U32.to_nat(Pieces.below(count)),Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat(first)),Outer{model,kept,rows,stride,zero,settled},ordered)) :
                    {Partials.merged_runs(~Partial,~merge,line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),_),line_tree(~Partial,~merge,rest,U32.is_le((count - Pieces.below(count) : U32),1),(count - Pieces.below(count) : U32),List.drop(&2,List<&2,Partial>,ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat(count),Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat(first)),Outer{model,kept,rows,stride,zero,settled}),U32.to_nat(Pieces.below(count))))) ==
                      ListReadback.enumerate(~Partial,~Split<Partial>,~split_output(~Partial,~merge),kept_count(kept),0n,Split{model,kept,Con{Views.Axis{extent,stride},rows},(U32.from_nat(outer) * outer_stride : U32),zero,1n+lower,False{},first,count}) : List<&2,Partial>}
                  %Equal.sym(List<&2,List<&2,Partial>>,List.drop(&2,List<&2,Partial>,ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat(count),Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat(first)),Outer{model,kept,rows,stride,zero,settled}),U32.to_nat(Pieces.below(count))),
                    ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat((count - Pieces.below(count) : U32)),Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat((first + Pieces.below(count) : U32))),Outer{model,kept,rows,stride,zero,settled}),dropped) :
                    {Partials.merged_runs(~Partial,~merge,line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat(Pieces.below(count)),Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat(first)),Outer{model,kept,rows,stride,zero,settled})),line_tree(~Partial,~merge,rest,U32.is_le((count - Pieces.below(count) : U32),1),(count - Pieces.below(count) : U32),_)) ==
                      ListReadback.enumerate(~Partial,~Split<Partial>,~split_output(~Partial,~merge),kept_count(kept),0n,Split{model,kept,Con{Views.Axis{extent,stride},rows},(U32.from_nat(outer) * outer_stride : U32),zero,1n+lower,False{},first,count}) : List<&2,Partial>}
                  Equal.trans(List<&2,Partial>,Partials.merged_runs(~Partial,~merge,line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat(Pieces.below(count)),Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat(first)),Outer{model,kept,rows,stride,zero,settled})),line_tree(~Partial,~merge,rest,U32.is_le((count - Pieces.below(count) : U32),1),(count - Pieces.below(count) : U32),ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat((count - Pieces.below(count) : U32)),Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat((first + Pieces.below(count) : U32))),Outer{model,kept,rows,stride,zero,settled}))),
                    Partials.merged_runs(~Partial,~merge,ListReadback.enumerate(~Partial,~Split<Partial>,~split_output(~Partial,~merge),kept_count(kept),0n,Split{model,kept,Con{Views.Axis{extent,stride},rows},(U32.from_nat(outer) * outer_stride : U32),zero,lower,U32.is_le(Pieces.below(count),1),first,Pieces.below(count)}),line_tree(~Partial,~merge,rest,U32.is_le((count - Pieces.below(count) : U32),1),(count - Pieces.below(count) : U32),ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat((count - Pieces.below(count) : U32)),Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat((first + Pieces.below(count) : U32))),Outer{model,kept,rows,stride,zero,settled}))),
                    ListReadback.enumerate(~Partial,~Split<Partial>,~split_output(~Partial,~merge),kept_count(kept),0n,Split{model,kept,Con{Views.Axis{extent,stride},rows},(U32.from_nat(outer) * outer_stride : U32),zero,1n+lower,False{},first,count}),
                    Equal.cong(List<&2,Partial>,List<&2,Partial>,low_run => Partials.merged_runs(~Partial,~merge,low_run,line_tree(~Partial,~merge,rest,U32.is_le((count - Pieces.below(count) : U32),1),(count - Pieces.below(count) : U32),ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat((count - Pieces.below(count) : U32)),Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat((first + Pieces.below(count) : U32))),Outer{model,kept,rows,stride,zero,settled}))),line_tree(~Partial,~merge,rest,U32.is_le(Pieces.below(count),1),Pieces.below(count),ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat(Pieces.below(count)),Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat(first)),Outer{model,kept,rows,stride,zero,settled})),ListReadback.enumerate(~Partial,~Split<Partial>,~split_output(~Partial,~merge),kept_count(kept),0n,Split{model,kept,Con{Views.Axis{extent,stride},rows},(U32.from_nat(outer) * outer_stride : U32),zero,lower,U32.is_le(Pieces.below(count),1),first,Pieces.below(count)}),left_tree),
                    Equal.trans(List<&2,Partial>,Partials.merged_runs(~Partial,~merge,ListReadback.enumerate(~Partial,~Split<Partial>,~split_output(~Partial,~merge),kept_count(kept),0n,Split{model,kept,Con{Views.Axis{extent,stride},rows},(U32.from_nat(outer) * outer_stride : U32),zero,lower,U32.is_le(Pieces.below(count),1),first,Pieces.below(count)}),line_tree(~Partial,~merge,rest,U32.is_le((count - Pieces.below(count) : U32),1),(count - Pieces.below(count) : U32),ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat((count - Pieces.below(count) : U32)),Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat((first + Pieces.below(count) : U32))),Outer{model,kept,rows,stride,zero,settled}))),
                      Partials.merged_runs(~Partial,~merge,ListReadback.enumerate(~Partial,~Split<Partial>,~split_output(~Partial,~merge),kept_count(kept),0n,Split{model,kept,Con{Views.Axis{extent,stride},rows},(U32.from_nat(outer) * outer_stride : U32),zero,lower,U32.is_le(Pieces.below(count),1),first,Pieces.below(count)}),ListReadback.enumerate(~Partial,~Split<Partial>,~split_output(~Partial,~merge),kept_count(kept),0n,Split{model,kept,Con{Views.Axis{extent,stride},rows},(U32.from_nat(outer) * outer_stride : U32),zero,lower,U32.is_le((count - Pieces.below(count) : U32),1),(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)})),
                      ListReadback.enumerate(~Partial,~Split<Partial>,~split_output(~Partial,~merge),kept_count(kept),0n,Split{model,kept,Con{Views.Axis{extent,stride},rows},(U32.from_nat(outer) * outer_stride : U32),zero,1n+lower,False{},first,count}),
                      Equal.cong(List<&2,Partial>,List<&2,Partial>,high_run => Partials.merged_runs(~Partial,~merge,ListReadback.enumerate(~Partial,~Split<Partial>,~split_output(~Partial,~merge),kept_count(kept),0n,Split{model,kept,Con{Views.Axis{extent,stride},rows},(U32.from_nat(outer) * outer_stride : U32),zero,lower,U32.is_le(Pieces.below(count),1),first,Pieces.below(count)}),high_run),
                        line_tree(~Partial,~merge,rest,U32.is_le((count - Pieces.below(count) : U32),1),(count - Pieces.below(count) : U32),ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat((count - Pieces.below(count) : U32)),Nat.add(Nat.mul(outer,U32.to_nat(extent)),U32.to_nat((first + Pieces.below(count) : U32))),Outer{model,kept,rows,stride,zero,settled})),ListReadback.enumerate(~Partial,~Split<Partial>,~split_output(~Partial,~merge),kept_count(kept),0n,Split{model,kept,Con{Views.Axis{extent,stride},rows},(U32.from_nat(outer) * outer_stride : U32),zero,lower,U32.is_le((count - Pieces.below(count) : U32),1),(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)}),right_tree),
                      merged_splits(~Partial,~merge,kept_count(kept),0n,model,kept,extent,stride,rows,(U32.from_nat(outer) * outer_stride : U32),zero,lower,first,count)))

# The runs of one line of a reduced axis through the counter are the outputs
# with the axis among the reduced ones.
def reduced_line_runs(~Partial: Data,~merge: Partial -> Partial -> Partial,+limits: Machine.Limits,+model: Tree.Tree<Partial>,+extent: U32,+stride: U32,+outer_stride: U32,+kept: List<&2,Views.Axis>,+rows: List<&2,Views.Axis>,
  +zero: Partial,+outer: Nat,+settled: Nat,+fuel: Nat,
  +scaled: {outer_stride == (extent * stride : U32) : U32},
  +positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +within: Within(Views.axis_extents(rows),Machine.capacity(limits)),
  +settled_enough: {Nat.is_le(fuel_needed(rows,30n),settled) == True{} : Bool},
  +enough: {Nat.is_le(fuel_needed(Con{Views.Axis{extent,stride},rows},30n),fuel) == True{} : Bool}) ->
  {Partials.counter_total(~Partial,~merge,counted_runs(~Partial,~merge,ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat(extent),Nat.mul(outer,U32.to_nat(extent)),Outer{model,kept,rows,stride,zero,settled}),[],32n,0n)) ==
    outer_run(~Partial,~merge,Outer{model,kept,Con{Views.Axis{extent,stride},rows},outer_stride,zero,fuel},outer) : List<&2,Partial>}:
  +line = ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat(extent),Nat.mul(outer,U32.to_nat(extent)),Outer{model,kept,rows,stride,zero,settled})
  +whole = {Split{model,kept,Con{Views.Axis{extent,stride},rows},(U32.from_nat(outer) * outer_stride : U32),zero,fuel,U32.is_le(extent,1),0,extent} : Split<Partial>}
  Equal.trans(List<&2,Partial>,Partials.counter_total(~Partial,~merge,counted_runs(~Partial,~merge,line,[],32n,0n)),line_tree(~Partial,~merge,31n,U32.is_le(extent,1),extent,line),
    outer_run(~Partial,~merge,Outer{model,kept,Con{Views.Axis{extent,stride},rows},outer_stride,zero,fuel},outer),
    aligned_line(~Partial,~merge,31n,U32.is_le(extent,1),extent,line,[],32n,0n,30n,{==},ListReadback.enumerate_length(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat(extent),Nat.mul(outer,U32.to_nat(extent)),Outer{model,kept,rows,stride,zero,settled}),
      positive,power_room(limits,U32.to_nat(extent),room),word_room(limits,extent,room),{==},{==}),
    Equal.trans(List<&2,Partial>,line_tree(~Partial,~merge,31n,U32.is_le(extent,1),extent,line),ListReadback.enumerate(~Partial,~Split<Partial>,~split_output(~Partial,~merge),kept_count(kept),0n,whole),
      outer_run(~Partial,~merge,Outer{model,kept,Con{Views.Axis{extent,stride},rows},outer_stride,zero,fuel},outer),
      {%Addition.zero_right(Nat.mul(outer,U32.to_nat(extent))) :
        {line_tree(~Partial,~merge,31n,U32.is_le(extent,1),extent,ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat(extent),_,Outer{model,kept,rows,stride,zero,settled})) ==
          ListReadback.enumerate(~Partial,~Split<Partial>,~split_output(~Partial,~merge),kept_count(kept),0n,whole) : List<&2,Partial>}
        line_tree_of(~Partial,~merge,31n,U32.is_le(extent,1),fuel,30n,limits,model,kept,rows,extent,stride,outer_stride,zero,outer,settled,extent,0,{==},positive,power_room(limits,U32.to_nat(extent),room),
          Order.reflexive(U32.to_nat(extent)),{==},enough,settled_enough,scaled,within,room)
        : {line_tree(~Partial,~merge,31n,U32.is_le(extent,1),extent,line) == ListReadback.enumerate(~Partial,~Split<Partial>,~split_output(~Partial,~merge),kept_count(kept),0n,whole) : List<&2,Partial>}},
      enumerate_same(~Partial,~Split<Partial>,~split_output(~Partial,~merge),~Sub<Partial>,~sub_output(~Partial,~merge),~whole,~Sub{model,kept,Con{Views.Axis{extent,stride},rows},(U32.from_nat(outer) * outer_stride : U32),zero,fuel},
        ~0n,~0n,~kept_count(kept),~(cell => cell_inside => {==}),kept_count(kept),0n,Order.reflexive(kept_count(kept)))))

# ---- Shapes of a box ----

def within_head(+extent: U32,+rest: List<&2,U32>,+cap: U32,+within: Within(extent <> rest,cap)) -> {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool}:
  KernelGemm.Both{+positive,+more} = within
  positive

def within_bound(+extent: U32,+rest: List<&2,U32>,+cap: U32,+within: Within(extent <> rest,cap)) -> {Nat.is_le(U32.to_nat(extent),U32.to_nat(cap)) == True{} : Bool}:
  KernelGemm.Both{+positive,+more} = within
  KernelGemm.Both{+bounded,+later} = more
  bounded

def within_tail(+extent: U32,+rest: List<&2,U32>,+cap: U32,+within: Within(extent <> rest,cap)) -> Within(rest,cap):
  KernelGemm.Both{+positive,+more} = within
  KernelGemm.Both{+bounded,+later} = more
  later

def within_kept(keep: Bool,+extent: U32,+stride: U32,+rest: List<&2,Views.Axis>,+cap: U32,+positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +bounded: {Nat.is_le(U32.to_nat(extent),U32.to_nat(cap)) == True{} : Bool},+others: Within(Views.axis_extents(rest),cap)) ->
  Within(Views.axis_extents(kept_axis(keep,Views.Axis{extent,stride},rest)),cap):
  match keep:
    case True{}: KernelGemm.Both{positive,KernelGemm.Both{bounded,others}}
    case False{}: others

# The extents of the axes of wanted flags lie within the capacity too.
def within_with(extents: List<&2,U32>,flags: List<&2,Bool>,+wanted: Bool,+cap: U32,+within: Within(extents,cap)) -> Within(Views.axis_extents(axes_with(extents,flags,wanted)),cap):
  match extents flags:
    case Nil{} Nil{}: Unit{}
    case Nil{} Con{+flag,+others}: Unit{}
    case Con{+extent,+rest} Nil{}: Unit{}
    case Con{+extent,+rest} Con{+flag,+others}:
      KernelGemm.Both{+positive,+more} = within
      KernelGemm.Both{+bounded,+later} = more
      within_kept(same_flag(flag,wanted),extent,U32.max(Grid.product(rest),1),axes_with(rest,others,wanted),cap,positive,bounded,within_with(rest,others,wanted,cap,later))

def elements_positive(extents: List<&2,U32>,+cap: U32,+within: Within(extents,cap)) -> {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}:
  match extents:
    case Nil{}: {==}
    case Con{+extent,+rest}:
      KernelGemm.Both{+positive,+more} = within
      KernelGemm.Both{+bounded,+later} = more
      Order.product_positive(U32.to_nat(extent),ShapeLaws.elements(rest),positive,elements_positive(rest,cap,later))

def product_word(extents: List<&2,U32>,+limits: Machine.Limits,+within: Within(extents,Machine.capacity(limits)),+room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat(Grid.product(extents)) == ShapeLaws.elements(extents) : Nat}:
  match extents:
    case Nil{}: {==}
    case Con{+extent,+rest}:
      KernelGemm.Both{+positive,+more} = within
      KernelGemm.Both{+bounded,+later} = more
      +rest_room = Order.transitive(ShapeLaws.elements(rest),ShapeLaws.elements(extent <> rest),U32.to_nat(Machine.capacity(limits)),DivisionOrder.le_multiple(U32.to_nat(extent),ShapeLaws.elements(rest),positive),room)
      Arithmetic.product_exact(Machine.capacity(limits),Limits.capacity_fits(limits),extent,Grid.product(rest),U32.to_nat(extent),ShapeLaws.elements(rest),{==},product_word(rest,limits,later,rest_room),room)

def kept_le(keep: Bool,+extent: U32,+stride: U32,+rest: List<&2,Views.Axis>,+others: List<&2,U32>,+positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +inner: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(rest)),ShapeLaws.elements(others)) == True{} : Bool}) ->
  {Nat.is_le(ShapeLaws.elements(Views.axis_extents(kept_axis(keep,Views.Axis{extent,stride},rest))),Nat.mul(U32.to_nat(extent),ShapeLaws.elements(others))) == True{} : Bool}:
  match keep:
    case True{}: RowMajor.mul_le_left(U32.to_nat(extent),ShapeLaws.elements(Views.axis_extents(rest)),ShapeLaws.elements(others),inner)
    case False{}:
      Order.transitive(ShapeLaws.elements(Views.axis_extents(rest)),ShapeLaws.elements(others),Nat.mul(U32.to_nat(extent),ShapeLaws.elements(others)),inner,
        DivisionOrder.le_multiple(U32.to_nat(extent),ShapeLaws.elements(others),positive))

def with_le(extents: List<&2,U32>,flags: List<&2,Bool>,+wanted: Bool,+cap: U32,+within: Within(extents,cap)) ->
  {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes_with(extents,flags,wanted))),ShapeLaws.elements(extents)) == True{} : Bool}:
  match extents flags:
    case Nil{} Nil{}: {==}
    case Nil{} Con{+flag,+others}: {==}
    case Con{+extent,+rest} Nil{}: Order.positive_one(ShapeLaws.elements(extent <> rest),elements_positive(extent <> rest,cap,within))
    case Con{+extent,+rest} Con{+flag,+others}:
      KernelGemm.Both{+positive,+more} = within
      KernelGemm.Both{+bounded,+later} = more
      kept_le(same_flag(flag,wanted),extent,U32.max(Grid.product(rest),1),axes_with(rest,others,wanted),rest,positive,with_le(rest,others,wanted,cap,later))

def onto_within(xs: List<&2,U32>,+extents: List<&2,U32>,+cap: U32,+within: Within(LineChainValues.onto_values(~U32,xs,extents),cap)) -> Within(extents,cap):
  match xs:
    case []: within
    case +extent <> +rest: within_tail(extent,extents,cap,onto_within(rest,extent <> extents,cap,within))

def onto_le(xs: List<&2,U32>,+extents: List<&2,U32>,+cap: U32,+within: Within(LineChainValues.onto_values(~U32,xs,extents),cap)) ->
  {Nat.is_le(ShapeLaws.elements(extents),ShapeLaws.elements(LineChainValues.onto_values(~U32,xs,extents))) == True{} : Bool}:
  match xs:
    case []: Order.reflexive(ShapeLaws.elements(extents))
    case +extent <> +rest:
      Order.transitive(ShapeLaws.elements(extents),ShapeLaws.elements(extent <> extents),ShapeLaws.elements(LineChainValues.onto_values(~U32,rest,extent <> extents)),
        DivisionOrder.le_multiple(U32.to_nat(extent),ShapeLaws.elements(extents),within_head(extent,extents,cap,onto_within(rest,extent <> extents,cap,within))),
        onto_le(rest,extent <> extents,cap,within))

def stride_scaled(+limits: Machine.Limits,+extent: U32,+extents: List<&2,U32>,+within: Within(extent <> extents,Machine.capacity(limits)),+room: {Nat.is_le(ShapeLaws.elements(extent <> extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.max(Grid.product(extent <> extents),1) == (extent * U32.max(Grid.product(extents),1) : U32) : U32}:
  +later = within_tail(extent,extents,Machine.capacity(limits),within)
  +rest_room = Order.transitive(ShapeLaws.elements(extents),ShapeLaws.elements(extent <> extents),U32.to_nat(Machine.capacity(limits)),
    DivisionOrder.le_multiple(U32.to_nat(extent),ShapeLaws.elements(extents),within_head(extent,extents,Machine.capacity(limits),within)),room)
  Equal.trans(U32,U32.max(Grid.product(extent <> extents),1),Grid.product(extent <> extents),(extent * U32.max(Grid.product(extents),1) : U32),
    GridPieces.word_kept(Grid.product(extent <> extents),ShapeLaws.elements(extent <> extents),product_word(extent <> extents,limits,within,room),elements_positive(extent <> extents,Machine.capacity(limits),within)),
    Equal.cong(U32,U32,word => (extent * word : U32),Grid.product(extents),U32.max(Grid.product(extents),1),
      Equal.sym(U32,U32.max(Grid.product(extents),1),Grid.product(extents),
        GridPieces.word_kept(Grid.product(extents),ShapeLaws.elements(extents),product_word(extents,limits,later,rest_room),elements_positive(extents,Machine.capacity(limits),later)))))

def kept_room(+limits: Machine.Limits,+extent: U32,+extents: List<&2,U32>,+flags: List<&2,Bool>,+within: Within(extent <> extents,Machine.capacity(limits)),
  +room: {Nat.is_le(ShapeLaws.elements(extent <> extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(U32.to_nat(extent),kept_count(axes_with(extents,flags,False{}))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  +later = within_tail(extent,extents,Machine.capacity(limits),within)
  +kept_extents = Views.axis_extents(axes_with(extents,flags,False{}))
  +inner = with_le(extents,flags,False{},Machine.capacity(limits),later)
  +rest_room = Order.transitive(ShapeLaws.elements(extents),ShapeLaws.elements(extent <> extents),U32.to_nat(Machine.capacity(limits)),
    DivisionOrder.le_multiple(U32.to_nat(extent),ShapeLaws.elements(extents),within_head(extent,extents,Machine.capacity(limits),within)),room)
  +exact = product_word(kept_extents,limits,within_with(extents,flags,False{},Machine.capacity(limits),later),
    Order.transitive(ShapeLaws.elements(kept_extents),ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits)),inner,rest_room))
  Held.nat_le_moved(Nat.mul(U32.to_nat(extent),kept_count(axes_with(extents,flags,False{}))),Nat.mul(U32.to_nat(extent),ShapeLaws.elements(kept_extents)),U32.to_nat(Machine.capacity(limits)),
    Equal.cong(Nat,Nat,count => Nat.mul(U32.to_nat(extent),count),kept_count(axes_with(extents,flags,False{})),ShapeLaws.elements(kept_extents),exact),
    Order.transitive(Nat.mul(U32.to_nat(extent),ShapeLaws.elements(kept_extents)),ShapeLaws.elements(extent <> extents),U32.to_nat(Machine.capacity(limits)),RowMajor.mul_le_left(U32.to_nat(extent),ShapeLaws.elements(kept_extents),ShapeLaws.elements(extents),inner),room))

def needed_depth(rows: List<&2,Views.Axis>) -> {Nat.is_le(fuel_needed(rows,30n),depth_of(rows)) == True{} : Bool}:
  match rows:
    case Nil{}: {==}
    case Con{+axis,+tail}:
      %Equal.sym(Nat,Nat.mul(33n,1n+List.length(&2,Views.Axis,tail)),Nat.add(33n,Nat.mul(33n,List.length(&2,Views.Axis,tail))),RowMajor.multiply_successor_right(33n,List.length(&2,Views.Axis,tail))) :
        {Nat.is_le(fuel_needed(axis <> tail,30n),1n+_) == True{} : Bool}
      Order.successor(fuel_needed(tail,30n),2n+Nat.mul(33n,List.length(&2,Views.Axis,tail)),
        Order.successor(fuel_needed(tail,30n),1n+Nat.mul(33n,List.length(&2,Views.Axis,tail)),needed_depth(tail)))

# ---- The runs of the passes ----

# The count of outer indices of the axes left to pass.
def outer_count(extents: List<&2,U32>) -> Nat:
  match extents:
    case Nil{}: 1n
    case Con{extent,rest}: Nat.mul(outer_count(rest),U32.to_nat(extent))

# The runs of a box whose passes went over extents.
def box_outer(~Partial: Data,+model: Tree.Tree<Partial>,+extents: List<&2,U32>,+flags: List<&2,Bool>,+zero: Partial) -> Outer<Partial>:
  Outer{model,axes_with(extents,flags,False{}),axes_with(extents,flags,True{}),U32.max(Grid.product(extents),1),zero,depth_of(axes_with(extents,flags,True{}))}

def totals_runs(~Partial: Data,~merge: Partial -> Partial -> Partial,count: Nat,+first: Nat,+limits: Machine.Limits,+model: Tree.Tree<Partial>,+zero: Partial,+extent: U32,+extents: List<&2,U32>,+flags: List<&2,Bool>,
  +scaled: {U32.max(Grid.product(extent <> extents),1) == (extent * U32.max(Grid.product(extents),1) : U32) : U32},
  +positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +within: Within(Views.axis_extents(axes_with(extents,flags,True{})),Machine.capacity(limits))) ->
  {line_totals(~Partial,~merge,ListReadback.enumerate(~List<&2,List<&2,Partial>>,~Lined<Outer<Partial>>,~line_of(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge)),count,first,Lined{box_outer(~Partial,model,extents,flags,zero),U32.to_nat(extent)})) ==
    ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),count,first,box_outer(~Partial,model,extent <> extents,True{} <> flags,zero)) : List<&2,List<&2,Partial>>}:
  match count:
    case 0n: {==}
    case 1n+ +more:
      cons_same(~List<&2,Partial>,Partials.counter_total(~Partial,~merge,counted_runs(~Partial,~merge,ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat(extent),Nat.mul(first,U32.to_nat(extent)),box_outer(~Partial,model,extents,flags,zero)),[],32n,0n)),
        outer_run(~Partial,~merge,box_outer(~Partial,model,extent <> extents,True{} <> flags,zero),first),
        line_totals(~Partial,~merge,ListReadback.enumerate(~List<&2,List<&2,Partial>>,~Lined<Outer<Partial>>,~line_of(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge)),more,1n+first,Lined{box_outer(~Partial,model,extents,flags,zero),U32.to_nat(extent)})),
        ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),more,1n+first,box_outer(~Partial,model,extent <> extents,True{} <> flags,zero)),
        reduced_line_runs(~Partial,~merge,limits,model,extent,U32.max(Grid.product(extents),1),U32.max(Grid.product(extent <> extents),1),axes_with(extents,flags,False{}),axes_with(extents,flags,True{}),
          zero,first,depth_of(axes_with(extents,flags,True{})),depth_of(Con{Views.Axis{extent,U32.max(Grid.product(extents),1)},axes_with(extents,flags,True{})}),
          scaled,positive,room,within,needed_depth(axes_with(extents,flags,True{})),needed_depth(Con{Views.Axis{extent,U32.max(Grid.product(extents),1)},axes_with(extents,flags,True{})})),
        totals_runs(~Partial,~merge,more,1n+first,limits,model,zero,extent,extents,flags,scaled,positive,room,within))

def joins_runs(~Partial: Data,~merge: Partial -> Partial -> Partial,count: Nat,+first: Nat,+limits: Machine.Limits,+model: Tree.Tree<Partial>,+zero: Partial,+extent: U32,+extents: List<&2,U32>,+flags: List<&2,Bool>,
  +scaled: {U32.max(Grid.product(extent <> extents),1) == (extent * U32.max(Grid.product(extents),1) : U32) : U32},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(extent),kept_count(axes_with(extents,flags,False{}))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {line_joins(~Partial,ListReadback.enumerate(~List<&2,List<&2,Partial>>,~Lined<Outer<Partial>>,~line_of(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge)),count,first,Lined{box_outer(~Partial,model,extents,flags,zero),U32.to_nat(extent)})) ==
    ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),count,first,box_outer(~Partial,model,extent <> extents,False{} <> flags,zero)) : List<&2,List<&2,Partial>>}:
  match count:
    case 0n: {==}
    case 1n+ +more:
      cons_same(~List<&2,Partial>,joined(~Partial,ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),U32.to_nat(extent),Nat.mul(first,U32.to_nat(extent)),box_outer(~Partial,model,extents,flags,zero))),
        outer_run(~Partial,~merge,box_outer(~Partial,model,extent <> extents,False{} <> flags,zero),first),
        line_joins(~Partial,ListReadback.enumerate(~List<&2,List<&2,Partial>>,~Lined<Outer<Partial>>,~line_of(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge)),more,1n+first,Lined{box_outer(~Partial,model,extents,flags,zero),U32.to_nat(extent)})),
        ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),more,1n+first,box_outer(~Partial,model,extent <> extents,False{} <> flags,zero)),
        kept_line_runs(~Partial,~merge,limits,model,extent,U32.max(Grid.product(extents),1),U32.max(Grid.product(extent <> extents),1),axes_with(extents,flags,False{}),axes_with(extents,flags,True{}),
          zero,depth_of(axes_with(extents,flags,True{})),first,scaled,room),
        joins_runs(~Partial,~merge,more,1n+first,limits,model,zero,extent,extents,flags,scaled,room))

def predecessor(value: Nat) -> Nat:
  match value:
    case 0n: 0n
    case 1n+previous: previous


# One pass over the runs of a box, the axis of the pass in front of the
# axes passed.
def axis_step(~Partial: Data,~merge: Partial -> Partial -> Partial,flag: Bool,+limits: Machine.Limits,+model: Tree.Tree<Partial>,+zero: Partial,+extent: U32,+extents: List<&2,U32>,+flags: List<&2,Bool>,+count: Nat,
  +positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +scaled: {U32.max(Grid.product(extent <> extents),1) == (extent * U32.max(Grid.product(extents),1) : U32) : U32},
  +within: Within(extent <> extents,Machine.capacity(limits)),
  +room: {Nat.is_le(ShapeLaws.elements(extent <> extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Partials.axis_pass(~Partial,~merge,flag,extent,joined(~List<&2,Partial>,ListReadback.enumerate(~List<&2,List<&2,Partial>>,~Lined<Outer<Partial>>,~line_of(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge)),count,0n,Lined{box_outer(~Partial,model,extents,flags,zero),U32.to_nat(extent)}))) ==
    ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),count,0n,box_outer(~Partial,model,extent <> extents,flag <> flags,zero)) : List<&2,List<&2,Partial>>}:
  match flag:
    case True{}:
      Equal.trans(List<&2,List<&2,Partial>>,Partials.axis_pass(~Partial,~merge,True{},extent,joined(~List<&2,Partial>,ListReadback.enumerate(~List<&2,List<&2,Partial>>,~Lined<Outer<Partial>>,~line_of(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge)),count,0n,Lined{box_outer(~Partial,model,extents,flags,zero),U32.to_nat(extent)}))),line_totals(~Partial,~merge,ListReadback.enumerate(~List<&2,List<&2,Partial>>,~Lined<Outer<Partial>>,~line_of(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge)),count,0n,Lined{box_outer(~Partial,model,extents,flags,zero),U32.to_nat(extent)})),
        ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),count,0n,box_outer(~Partial,model,extent <> extents,True{} <> flags,zero)),
        reduced_axis_pass(~Partial,~merge,ListReadback.enumerate(~List<&2,List<&2,Partial>>,~Lined<Outer<Partial>>,~line_of(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge)),count,0n,Lined{box_outer(~Partial,model,extents,flags,zero),U32.to_nat(extent)}),extent,line_lengths(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),count,0n,U32.to_nat(extent),box_outer(~Partial,model,extents,flags,zero)),positive),
        totals_runs(~Partial,~merge,count,0n,limits,model,zero,extent,extents,flags,scaled,positive,within_bound(extent,extents,Machine.capacity(limits),within),
          within_with(extents,flags,True{},Machine.capacity(limits),within_tail(extent,extents,Machine.capacity(limits),within))))
    case False{}:
      Equal.trans(List<&2,List<&2,Partial>>,Partials.axis_pass(~Partial,~merge,False{},extent,joined(~List<&2,Partial>,ListReadback.enumerate(~List<&2,List<&2,Partial>>,~Lined<Outer<Partial>>,~line_of(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge)),count,0n,Lined{box_outer(~Partial,model,extents,flags,zero),U32.to_nat(extent)}))),line_joins(~Partial,ListReadback.enumerate(~List<&2,List<&2,Partial>>,~Lined<Outer<Partial>>,~line_of(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge)),count,0n,Lined{box_outer(~Partial,model,extents,flags,zero),U32.to_nat(extent)})),
        ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),count,0n,box_outer(~Partial,model,extent <> extents,False{} <> flags,zero)),
        kept_axis_pass(~Partial,~merge,ListReadback.enumerate(~List<&2,List<&2,Partial>>,~Lined<Outer<Partial>>,~line_of(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge)),count,0n,Lined{box_outer(~Partial,model,extents,flags,zero),U32.to_nat(extent)}),extent,line_lengths(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),count,0n,U32.to_nat(extent),box_outer(~Partial,model,extents,flags,zero)),positive),
        joins_runs(~Partial,~merge,count,0n,limits,model,zero,extent,extents,flags,scaled,kept_room(limits,extent,extents,flags,within,room)))

# The passes over the axes left, the last first, end in one run of every
# output.
def passes_runs(~Partial: Data,~merge: Partial -> Partial -> Partial,xs: List<&2,U32>,fs: List<&2,Bool>,+limits: Machine.Limits,+model: Tree.Tree<Partial>,+zero: Partial,+extents: List<&2,U32>,+flags: List<&2,Bool>,+runs: List<&2,List<&2,Partial>>,
  +same: {runs == ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),outer_count(xs),0n,box_outer(~Partial,model,extents,flags,zero)) : List<&2,List<&2,Partial>>},
  +lengths: {List.length(&2,U32,xs) == List.length(&2,Bool,fs) : Nat},
  +within: Within(LineChainValues.onto_values(~U32,xs,extents),Machine.capacity(limits)),
  +room: {Nat.is_le(ShapeLaws.elements(LineChainValues.onto_values(~U32,xs,extents)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Partials.passes(~Partial,~merge,xs,fs,runs) ==
    ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),1n,0n,box_outer(~Partial,model,LineChainValues.onto_values(~U32,xs,extents),LineChainValues.onto_values(~Bool,fs,flags),zero)) : List<&2,List<&2,Partial>>}:
  match xs:
    case []:
      match fs:
        case []: same
        case +flag <> +others:
          Empty.absurd({Partials.passes(~Partial,~merge,[],flag <> others,runs) ==
            ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),1n,0n,box_outer(~Partial,model,extents,LineChainValues.onto_values(~Bool,flag <> others,flags),zero)) : List<&2,List<&2,Partial>>},
            Digits.zero_successor(List.length(&2,Bool,others),lengths))
    case +extent <> +rest:
      match fs:
        case []:
          Empty.absurd({Partials.passes(~Partial,~merge,extent <> rest,[],runs) ==
            ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),1n,0n,box_outer(~Partial,model,LineChainValues.onto_values(~U32,extent <> rest,extents),flags,zero)) : List<&2,List<&2,Partial>>},
            Digits.zero_successor(List.length(&2,U32,rest),Equal.sym(Nat,1n+List.length(&2,U32,rest),0n,lengths)))
        case +flag <> +others:
          +whole_within = onto_within(rest,extent <> extents,Machine.capacity(limits),within)
          +whole_room = Order.transitive(ShapeLaws.elements(extent <> extents),ShapeLaws.elements(LineChainValues.onto_values(~U32,rest,extent <> extents)),U32.to_nat(Machine.capacity(limits)),
            onto_le(rest,extent <> extents,Machine.capacity(limits),within),room)
          +positive = within_head(extent,extents,Machine.capacity(limits),whole_within)
          +later = within_tail(extent,extents,Machine.capacity(limits),whole_within)
          +scaled = stride_scaled(limits,extent,extents,whole_within,whole_room)
          +lines = ListReadback.enumerate(~List<&2,List<&2,Partial>>,~Lined<Outer<Partial>>,~line_of(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge)),outer_count(rest),0n,Lined{box_outer(~Partial,model,extents,flags,zero),U32.to_nat(extent)})
          +split = Equal.trans(List<&2,List<&2,Partial>>,runs,ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),Nat.mul(outer_count(rest),U32.to_nat(extent)),0n,box_outer(~Partial,model,extents,flags,zero)),joined(~List<&2,Partial>,lines),
            same,enumerate_lines(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),outer_count(rest),0n,U32.to_nat(extent),box_outer(~Partial,model,extents,flags,zero)))
          +step = Equal.trans(List<&2,List<&2,Partial>>,Partials.axis_pass(~Partial,~merge,flag,extent,runs),Partials.axis_pass(~Partial,~merge,flag,extent,joined(~List<&2,Partial>,lines)),
            ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),outer_count(rest),0n,box_outer(~Partial,model,extent <> extents,flag <> flags,zero)),
            Equal.cong(List<&2,List<&2,Partial>>,List<&2,List<&2,Partial>>,each => Partials.axis_pass(~Partial,~merge,flag,extent,each),runs,joined(~List<&2,Partial>,lines),split),
            axis_step(~Partial,~merge,flag,limits,model,zero,extent,extents,flags,outer_count(rest),positive,scaled,whole_within,whole_room))
          passes_runs(~Partial,~merge,rest,others,limits,model,zero,extent <> extents,flag <> flags,Partials.axis_pass(~Partial,~merge,flag,extent,runs),step,
            Equal.cong(Nat,Nat,value => predecessor(value),1n+List.length(&2,U32,rest),1n+List.length(&2,Bool,others),lengths),within,room)

# ---- The first runs ----

def drop_zero(~Element: Data,items: List<&2,Element>) -> {List.drop(&2,Element,items,0n) == items : List<&2,Element>}:
  match items:
    case []: {==}
    case head <> tail: {==}

def list_tail(~Element: Data,items: List<&2,Element>) -> List<&2,Element>:
  match items:
    case []: []
    case head <> tail: tail

def at_head(~Element: Data,+fallback: Element,items: List<&2,Element>,index: Nat,+head: Element,+tail: List<&2,Element>,
  +dropped: {List.drop(&2,Element,items,index) == head <> tail : List<&2,Element>}) -> {ListReadback.at(~Element,fallback,index,items) == head : Element}:
  match items:
    case []:
      Empty.absurd({ListReadback.at(~Element,fallback,index,[]) == head : Element},
        Digits.zero_successor(List.length(&2,Element,tail),Equal.cong(List<&2,Element>,Nat,list => List.length(&2,Element,list),[],head <> tail,dropped)))
    case +item <> +rest:
      match index:
        case 0n: Equal.cong(List<&2,Element>,Element,list => ListReadback.at(~Element,fallback,0n,list),item <> rest,head <> tail,dropped)
        case 1n+ +previous: at_head(~Element,fallback,rest,previous,head,tail,dropped)

def drop_next(~Element: Data,items: List<&2,Element>,index: Nat,+head: Element,+tail: List<&2,Element>,
  +dropped: {List.drop(&2,Element,items,index) == head <> tail : List<&2,Element>}) -> {List.drop(&2,Element,items,1n+index) == tail : List<&2,Element>}:
  match items:
    case []:
      Empty.absurd({List.drop(&2,Element,[],1n+index) == tail : List<&2,Element>},
        Digits.zero_successor(List.length(&2,Element,tail),Equal.cong(List<&2,Element>,Nat,list => List.length(&2,Element,list),[],head <> tail,dropped)))
    case +item <> +rest:
      match index:
        case 0n:
          Equal.trans(List<&2,Element>,List.drop(&2,Element,rest,0n),rest,tail,drop_zero(~Element,rest),
            Equal.cong(List<&2,Element>,List<&2,Element>,list => list_tail(~Element,list),item <> rest,head <> tail,dropped))
        case 1n+ +previous: drop_next(~Element,rest,previous,head,tail,dropped)

def drop_inside(~Element: Data,+items: List<&2,Element>,+index: Nat,+head: Element,+tail: List<&2,Element>,
  +dropped: {List.drop(&2,Element,items,index) == head <> tail : List<&2,Element>}) -> {Nat.is_lt(index,List.length(&2,Element,items)) == True{} : Bool}:
  Order.difference_positive(index,List.length(&2,Element,items),
    Held.nat_lt_moved(0n,Nat.sub(List.length(&2,Element,items),index),1n+List.length(&2,Element,tail),
      Equal.trans(Nat,Nat.sub(List.length(&2,Element,items),index),List.length(&2,Element,List.drop(&2,Element,items,index)),1n+List.length(&2,Element,tail),
        Equal.sym(Nat,List.length(&2,Element,List.drop(&2,Element,items,index)),Nat.sub(List.length(&2,Element,items),index),LineChainValues.drop_length(~Element,items,index)),
        Equal.cong(List<&2,Element>,Nat,list => List.length(&2,Element,list),List.drop(&2,Element,items,index),head <> tail,dropped)),
      {==}))

# Every partial a run of its own, read from the partials written in order.
def singles_runs(~Partial: Data,~merge: Partial -> Partial -> Partial,rest: List<&2,Partial>,+limits: Machine.Limits,+partials: List<&2,Partial>,+zero: Partial,+index: Nat,
  +dropped: {List.drop(&2,Partial,partials,index) == rest : List<&2,Partial>},
  +room: {Nat.is_le(List.length(&2,Partial,partials),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Partials.singles(~Partial,rest) == ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),List.length(&2,Partial,rest),index,box_outer(~Partial,written(~Partial,partials,zero),[],[],zero)) : List<&2,List<&2,Partial>>}:
  match rest:
    case []: {==}
    case +head <> +tail:
      +read = Equal.trans(Partial,model_read(~Partial,written(~Partial,partials,zero),(U32.from_nat(index) * 1 : U32)),model_read(~Partial,written(~Partial,partials,zero),U32.from_nat(index)),head,
        Equal.cong(U32,Partial,word => model_read(~Partial,written(~Partial,partials,zero),word),(U32.from_nat(index) * 1 : U32),U32.from_nat(index),Arithmetic.multiply_one(U32.from_nat(index))),
        Equal.trans(Partial,model_read(~Partial,written(~Partial,partials,zero),U32.from_nat(index)),ListReadback.at(~Partial,zero,index,partials),head,
          written_at(~Partial,partials,zero,index,Machine.capacity(limits),Limits.capacity_allocatable(limits),room,drop_inside(~Partial,partials,index,head,tail,dropped)),
          at_head(~Partial,zero,partials,index,head,tail,dropped)))
      cons_same(~List<&2,Partial>,[head],outer_run(~Partial,~merge,box_outer(~Partial,written(~Partial,partials,zero),[],[],zero),index),Partials.singles(~Partial,tail),
        ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),List.length(&2,Partial,tail),1n+index,box_outer(~Partial,written(~Partial,partials,zero),[],[],zero)),
        Equal.cong(Partial,List<&2,Partial>,partial => [partial],head,model_read(~Partial,written(~Partial,partials,zero),(U32.from_nat(index) * 1 : U32)),
          Equal.sym(Partial,model_read(~Partial,written(~Partial,partials,zero),(U32.from_nat(index) * 1 : U32)),head,read)),
        singles_runs(~Partial,~merge,tail,limits,partials,zero,1n+index,drop_next(~Partial,partials,index,head,tail,dropped),room))

# ---- Reversed lists ----

def onto_go(~Element: Data,items: List<&2,Element>,+done: List<&2,Element>) ->
  {LineChainValues.onto_values(~Element,items,done) == List.reverse.go(&2,Element,items,done) : List<&2,Element>}:
  match items:
    case []: {==}
    case +head <> +tail: onto_go(~Element,tail,head <> done)

def onto_reverse(~Element: Data,+items: List<&2,Element>) -> {LineChainValues.onto_values(~Element,List.reverse(&2,Element,items),[]) == items : List<&2,Element>}:
  Equal.trans(List<&2,Element>,LineChainValues.onto_values(~Element,List.reverse(&2,Element,items),[]),List.reverse.go(&2,Element,List.reverse(&2,Element,items),[]),items,
    onto_go(~Element,List.reverse(&2,Element,items),[]),
    Equal.trans(List<&2,Element>,List.reverse.go(&2,Element,List.reverse(&2,Element,items),[]),List.reverse.go(&2,Element,LineChainValues.onto_values(~Element,items,[]),[]),items,
      Equal.cong(List<&2,Element>,List<&2,Element>,list => List.reverse.go(&2,Element,list,[]),List.reverse(&2,Element,items),LineChainValues.onto_values(~Element,items,[]),
        Equal.sym(List<&2,Element>,LineChainValues.onto_values(~Element,items,[]),List.reverse(&2,Element,items),onto_go(~Element,items,[]))),
      Equal.trans(List<&2,Element>,List.reverse.go(&2,Element,LineChainValues.onto_values(~Element,items,[]),[]),List.append(&2,Element,items,[]),items,
        LineChainValues.reverse_onto(~Element,items,[],[]),ListReadback.append_empty(~Element,items))))

def go_length(~Element: Data,items: List<&2,Element>,+done: List<&2,Element>) ->
  {List.length(&2,Element,List.reverse.go(&2,Element,items,done)) == Nat.add(List.length(&2,Element,items),List.length(&2,Element,done)) : Nat}:
  match items:
    case []: {==}
    case +head <> +tail:
      Equal.trans(Nat,List.length(&2,Element,List.reverse.go(&2,Element,tail,head <> done)),Nat.add(List.length(&2,Element,tail),1n+List.length(&2,Element,done)),
        1n+Nat.add(List.length(&2,Element,tail),List.length(&2,Element,done)),
        go_length(~Element,tail,head <> done),Addition.successor_right(List.length(&2,Element,tail),List.length(&2,Element,done)))

def reverse_length(~Element: Data,+items: List<&2,Element>) -> {List.length(&2,Element,List.reverse(&2,Element,items)) == List.length(&2,Element,items) : Nat}:
  Equal.trans(Nat,List.length(&2,Element,List.reverse(&2,Element,items)),Nat.add(List.length(&2,Element,items),0n),List.length(&2,Element,items),
    go_length(~Element,items,[]),Addition.zero_right(List.length(&2,Element,items)))

def count_go(extents: List<&2,U32>,+done: List<&2,U32>) -> {outer_count(List.reverse.go(&2,U32,extents,done)) == Nat.mul(outer_count(done),ShapeLaws.elements(extents)) : Nat}:
  match extents:
    case []: Equal.sym(Nat,Nat.mul(outer_count(done),1n),outer_count(done),Order.times_one(outer_count(done)))
    case +extent <> +rest:
      Equal.trans(Nat,outer_count(List.reverse.go(&2,U32,rest,extent <> done)),Nat.mul(Nat.mul(outer_count(done),U32.to_nat(extent)),ShapeLaws.elements(rest)),
        Nat.mul(outer_count(done),Nat.mul(U32.to_nat(extent),ShapeLaws.elements(rest))),count_go(rest,extent <> done),RowMajor.assoc(outer_count(done),U32.to_nat(extent),ShapeLaws.elements(rest)))

def count_reverse(+extents: List<&2,U32>) -> {outer_count(List.reverse(&2,U32,extents)) == ShapeLaws.elements(extents) : Nat}:
  Equal.trans(Nat,outer_count(List.reverse(&2,U32,extents)),Nat.mul(1n,ShapeLaws.elements(extents)),ShapeLaws.elements(extents),count_go(extents,[]),
    Addition.zero_right(ShapeLaws.elements(extents)))

# ---- The outputs of a box ----

# The outputs of a box of rows: every output the tree of the listed partials
# of its rows.
def box_read(~Partial: Data,~merge: Partial -> Partial -> Partial,+limits: Machine.Limits,+extents: List<&2,U32>,+flags: List<&2,Bool>,+zero: Partial,+partials: List<&2,Partial>,
  +within: Within(extents,Machine.capacity(limits)),
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +length: {List.length(&2,Partial,partials) == ShapeLaws.elements(extents) : Nat},
  +lengths: {List.length(&2,U32,extents) == List.length(&2,Bool,flags) : Nat}) ->
  {Partials.box_outputs(~Partial,~merge,extents,flags,partials) ==
    ListReadback.enumerate(~Partial,~Box<Partial>,~box_output(~Partial,~merge),kept_count(axes_with(extents,flags,False{})),0n,
      Box{written(~Partial,partials,zero),axes_with(extents,flags,False{}),axes_with(extents,flags,True{}),0,zero}) : List<&2,Partial>}:
  +reversed = List.reverse(&2,U32,extents)
  +turned = List.reverse(&2,Bool,flags)
  +back = onto_reverse(~U32,extents)
  +flags_back = onto_reverse(~Bool,flags)
  +kept = axes_with(extents,flags,False{})
  +rows = axes_with(extents,flags,True{})
  +base = Equal.trans(List<&2,List<&2,Partial>>,Partials.singles(~Partial,partials),
    ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),List.length(&2,Partial,partials),0n,box_outer(~Partial,written(~Partial,partials,zero),[],[],zero)),
    ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),outer_count(reversed),0n,box_outer(~Partial,written(~Partial,partials,zero),[],[],zero)),
    singles_runs(~Partial,~merge,partials,limits,partials,zero,0n,drop_zero(~Partial,partials),Held.nat_le_moved(List.length(&2,Partial,partials),ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits)),length,room)),
    Equal.cong(Nat,List<&2,List<&2,Partial>>,count => ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),count,0n,box_outer(~Partial,written(~Partial,partials,zero),[],[],zero)),List.length(&2,Partial,partials),outer_count(reversed),
      Equal.trans(Nat,List.length(&2,Partial,partials),ShapeLaws.elements(extents),outer_count(reversed),length,Equal.sym(Nat,outer_count(reversed),ShapeLaws.elements(extents),count_reverse(extents)))))
  +passed = passes_runs(~Partial,~merge,reversed,turned,limits,written(~Partial,partials,zero),zero,[],[],Partials.singles(~Partial,partials),base,
    Equal.trans(Nat,List.length(&2,U32,reversed),List.length(&2,U32,extents),List.length(&2,Bool,turned),reverse_length(~U32,extents),
      Equal.trans(Nat,List.length(&2,U32,extents),List.length(&2,Bool,flags),List.length(&2,Bool,turned),lengths,Equal.sym(Nat,List.length(&2,Bool,turned),List.length(&2,Bool,flags),reverse_length(~Bool,flags)))),
    {%Equal.sym(List<&2,U32>,LineChainValues.onto_values(~U32,reversed,[]),extents,back) : Within(_,Machine.capacity(limits)) within : Within(LineChainValues.onto_values(~U32,reversed,[]),Machine.capacity(limits))},
    {%Equal.sym(List<&2,U32>,LineChainValues.onto_values(~U32,reversed,[]),extents,back) : {Nat.is_le(ShapeLaws.elements(_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} room :
      {Nat.is_le(ShapeLaws.elements(LineChainValues.onto_values(~U32,reversed,[])),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}})
  +whole = Equal.trans(List<&2,List<&2,Partial>>,Partials.passes(~Partial,~merge,reversed,turned,Partials.singles(~Partial,partials)),
    ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),1n,0n,box_outer(~Partial,written(~Partial,partials,zero),LineChainValues.onto_values(~U32,reversed,[]),LineChainValues.onto_values(~Bool,turned,[]),zero)),
    ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),1n,0n,box_outer(~Partial,written(~Partial,partials,zero),extents,flags,zero)),
    passed,
    Equal.trans(List<&2,List<&2,Partial>>,ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),1n,0n,box_outer(~Partial,written(~Partial,partials,zero),LineChainValues.onto_values(~U32,reversed,[]),LineChainValues.onto_values(~Bool,turned,[]),zero)),
      ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),1n,0n,box_outer(~Partial,written(~Partial,partials,zero),extents,LineChainValues.onto_values(~Bool,turned,[]),zero)),
      ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),1n,0n,box_outer(~Partial,written(~Partial,partials,zero),extents,flags,zero)),
      Equal.cong(List<&2,U32>,List<&2,List<&2,Partial>>,shape => ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),1n,0n,box_outer(~Partial,written(~Partial,partials,zero),shape,LineChainValues.onto_values(~Bool,turned,[]),zero)),
        LineChainValues.onto_values(~U32,reversed,[]),extents,back),
      Equal.cong(List<&2,Bool>,List<&2,List<&2,Partial>>,marks => ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),1n,0n,box_outer(~Partial,written(~Partial,partials,zero),extents,marks,zero)),
        LineChainValues.onto_values(~Bool,turned,[]),flags,flags_back)))
  Equal.trans(List<&2,Partial>,Partials.only_run(~Partial,Partials.passes(~Partial,~merge,reversed,turned,Partials.singles(~Partial,partials))),
    Partials.only_run(~Partial,ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),1n,0n,box_outer(~Partial,written(~Partial,partials,zero),extents,flags,zero))),
    ListReadback.enumerate(~Partial,~Box<Partial>,~box_output(~Partial,~merge),kept_count(kept),0n,Box{written(~Partial,partials,zero),kept,rows,0,zero}),
    Equal.cong(List<&2,List<&2,Partial>>,List<&2,Partial>,runs => Partials.only_run(~Partial,runs),Partials.passes(~Partial,~merge,reversed,turned,Partials.singles(~Partial,partials)),
      ListReadback.enumerate(~List<&2,Partial>,~Outer<Partial>,~outer_run(~Partial,~merge),1n,0n,box_outer(~Partial,written(~Partial,partials,zero),extents,flags,zero)),whole),
    Equal.trans(List<&2,Partial>,ListReadback.enumerate(~Partial,~Sub<Partial>,~sub_output(~Partial,~merge),kept_count(kept),0n,Sub{written(~Partial,partials,zero),kept,rows,(U32.from_nat(0n) * U32.max(Grid.product(extents),1) : U32),zero,depth_of(rows)}),
      ListReadback.enumerate(~Partial,~Sub<Partial>,~sub_output(~Partial,~merge),kept_count(kept),0n,Sub{written(~Partial,partials,zero),kept,rows,0,zero,depth_of(rows)}),
      ListReadback.enumerate(~Partial,~Box<Partial>,~box_output(~Partial,~merge),kept_count(kept),0n,Box{written(~Partial,partials,zero),kept,rows,0,zero}),
      Equal.cong(U32,List<&2,Partial>,offset => ListReadback.enumerate(~Partial,~Sub<Partial>,~sub_output(~Partial,~merge),kept_count(kept),0n,Sub{written(~Partial,partials,zero),kept,rows,offset,zero,depth_of(rows)}),
        (U32.from_nat(0n) * U32.max(Grid.product(extents),1) : U32),0,Arithmetic.multiply_zero_left(U32.max(Grid.product(extents),1))),
      enumerate_same(~Partial,~Sub<Partial>,~sub_output(~Partial,~merge),~Box<Partial>,~box_output(~Partial,~merge),~Sub{written(~Partial,partials,zero),kept,rows,0,zero,depth_of(rows)},~Box{written(~Partial,partials,zero),kept,rows,0,zero},
        ~0n,~0n,~kept_count(kept),~(cell => cell_inside => {==}),kept_count(kept),0n,Order.reflexive(kept_count(kept)))))

# ---- The rows of a read ----

# A read hands back the pieces regrouped and the partials of the box of its
# rows.
def read_pair(~Block: Type,~Fill: Data,~Partial: Data,~merge: Partial -> Partial -> Partial,~vacant: Fill -> Block,+grid: Grid.Grid,+fill: Fill,+extents: List<&2,U32>,+flags: List<&2,Bool>,folded: Partition.Tree<&1,Block,Unit> & List<&2,Partial>) ->
  {Partials.regrouped_back(~Block,~Fill,~Partial,~vacant,grid,fill,Partials.carried_joined(~Block,~Partial,~merge,extents,flags,folded)) ==
    (Pieces.blocks_regrouped(~Block,~Fill,~vacant,Pieces.ranges_of_grid(grid),fill,Pair.fst(Partition.Tree<&1,Block,Unit>,List<&2,Partial>,folded)),Partials.box_outputs(~Partial,~merge,extents,flags,Pair.snd(Partition.Tree<&1,Block,Unit>,List<&2,Partial>,folded))) : Partition.Tree<&1,Block,Unit> & List<&2,Partial>}:
  match folded:
    case (tree,partials): {==}

def per_outer(grid: Grid.Grid) -> {Partials.cells_per_outer(grid) == U32.max(Layout.cells_per_outer(grid),1) : U32}:
  match grid:
    case Grid.Grid{+count,+prefix,+outer,+extent,+unit_cells,+piece_units,+per_outer,+pieces}: {==}

# A refilled list of contexts is not empty.
def refilled_ready(~Start: Data,+all: List<&2,Start>,later: List<&2,Start>,+filled: {Nat.is_lt(0n,List.length(&2,Start,all)) == True{} : Bool}) ->
  {Nat.is_lt(0n,List.length(&2,Start,LineChain.refilled(~Start,all,later))) == True{} : Bool}:
  match later:
    case Nil{}: filled
    case Con{+head,+tail}: {==}

# The contexts drawn for lines are one a line.
def drawn_length(~Start: Data,~Value: Data,~begin_value: Start -> Value,remaining: Nat,+all: List<&2,Start>,upcoming: List<&2,Start>,
  +filled: {Nat.is_lt(0n,List.length(&2,Start,all)) == True{} : Bool},+ready: {Nat.is_lt(0n,List.length(&2,Start,upcoming)) == True{} : Bool}) ->
  {List.length(&2,Value,LineChainValues.drawn(~Start,~Value,~begin_value,remaining,all,upcoming)) == remaining : Nat}:
  match remaining upcoming:
    case 0n upcoming: {==}
    case 1n+ +rest Nil{}:
      Empty.absurd({List.length(&2,Value,LineChainValues.drawn(~Start,~Value,~begin_value,1n+rest,all,[])) == 1n+rest : Nat},WordEncoding.false_true(ready))
    case 1n+ +rest Con{+head,+later}:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Value,LineChainValues.drawn(~Start,~Value,~begin_value,rest,all,LineChain.refilled(~Start,all,later))),rest,
        drawn_length(~Start,~Value,~begin_value,rest,all,LineChain.refilled(~Start,all,later),filled,refilled_ready(~Start,all,later,filled)))

# The lines taken from enough starts are as many as asked.
def lines_count(~Cell: Data,~Value: Data,~step: Value -> Cell -> Value,remaining: Nat,+line: Nat,+inner: Nat,+group: List<&2,Cell>,starts: List<&2,Value>,
  +enough: {Nat.is_le(remaining,List.length(&2,Value,starts)) == True{} : Bool}) ->
  {List.length(&2,Value,LineChainValues.lines_taken(~Cell,~Value,~step,remaining,line,inner,group,starts)) == remaining : Nat}:
  match remaining starts:
    case 0n starts: {==}
    case 1n+ +rest Nil{}:
      Empty.absurd({List.length(&2,Value,LineChainValues.lines_taken(~Cell,~Value,~step,1n+rest,line,inner,group,[])) == 1n+rest : Nat},WordEncoding.false_true(enough))
    case 1n+ +rest Con{+start,+tail}:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Value,LineChainValues.lines_taken(~Cell,~Value,~step,rest,1n+line,inner,group,tail)),rest,
        lines_count(~Cell,~Value,~step,rest,1n+line,inner,group,tail,enough))

# The finals of blocks are inner a block.
def block_count(~Cell: Data,~Value: Data,~Final: Data,~step: Value -> Cell -> Value,~turn_value: Value -> Final,groups: Nat,+length: U32,+inner: U32,+list: List<&2,Cell>,+starts: List<&2,Value>,
  +enough: {Nat.is_le(U32.to_nat(inner),List.length(&2,Value,starts)) == True{} : Bool}) ->
  {List.length(&2,Final,Reduce.block_folds(~Cell,~Value,~Final,~step,~turn_value,groups,length,inner,list,starts)) == Nat.mul(groups,U32.to_nat(inner)) : Nat}:
  match groups:
    case 0n: {==}
    case 1n+ +rest:
      +group = List.take(&2,Cell,list,U32.to_nat((length * inner : U32)))
      +taken = LineChainValues.lines_taken(~Cell,~Value,~step,U32.to_nat(inner),0n,U32.to_nat(inner),group,starts)
      +lines = LineChainValues.turned_values(~Value,~Final,~turn_value,taken)
      +later = Reduce.block_folds(~Cell,~Value,~Final,~step,~turn_value,rest,length,inner,List.drop(&2,Cell,list,U32.to_nat((length * inner : U32))),starts)
      +first = Equal.trans(Nat,List.length(&2,Final,lines),List.length(&2,Value,taken),U32.to_nat(inner),Reduce.turned_length(~Final,~Value,~turn_value,taken),
        lines_count(~Cell,~Value,~step,U32.to_nat(inner),0n,U32.to_nat(inner),group,starts,enough))
      Equal.trans(Nat,List.length(&2,Final,List.append(&2,Final,lines,later)),Nat.add(List.length(&2,Final,lines),List.length(&2,Final,later)),Nat.mul(1n+rest,U32.to_nat(inner)),
        ListReadback.append_length(~Final,lines,later),
        Equal.trans(Nat,Nat.add(List.length(&2,Final,lines),List.length(&2,Final,later)),Nat.add(U32.to_nat(inner),List.length(&2,Final,later)),Nat.add(U32.to_nat(inner),Nat.mul(rest,U32.to_nat(inner))),
          Equal.cong(Nat,Nat,count => Nat.add(count,List.length(&2,Final,later)),List.length(&2,Final,lines),U32.to_nat(inner),first),
          Equal.cong(Nat,Nat,count => Nat.add(U32.to_nat(inner),count),List.length(&2,Final,later),Nat.mul(rest,U32.to_nat(inner)),
            block_count(~Cell,~Value,~Final,~step,~turn_value,rest,length,inner,List.drop(&2,Cell,list,U32.to_nat((length * inner : U32))),starts,enough))))

# ---- The lines of blocks one by one ----


# The final at an index of turned values is the turn of the value there.
def turned_at(~Value: Data,~Final: Data,~turn_value: Value -> Final,+zero: Final,+fallback: Value,values: List<&2,Value>,index: Nat,
  +inside: {Nat.is_lt(index,List.length(&2,Value,values)) == True{} : Bool}) ->
  {ListReadback.at(~Final,zero,index,LineChainValues.turned_values(~Value,~Final,~turn_value,values)) == turn_value(ListReadback.at(~Value,fallback,index,values)) : Final}:
  match values index:
    case Nil{} index:
      Empty.absurd({ListReadback.at(~Final,zero,index,[]) == turn_value(ListReadback.at(~Value,fallback,index,[])) : Final},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(index,0n),True{},Equal.sym(Bool,Nat.is_lt(index,0n),False{},Quotients.lt_zero(index)),inside)))
    case Con{+head,+tail} 0n: {==}
    case Con{+head,+tail} 1n+ +previous: turned_at(~Value,~Final,~turn_value,zero,fallback,tail,previous,inside)

# The value at an index of the lines of a block folds the cells of its line
# from its start.
def lines_taken_at(~Cell: Data,~Value: Data,~step: Value -> Cell -> Value,remaining: Nat,+line: Nat,+inner: Nat,+group: List<&2,Cell>,starts: List<&2,Value>,+fallback: Value,index: Nat,
  +inside: {Nat.is_lt(index,remaining) == True{} : Bool},+enough: {Nat.is_lt(index,List.length(&2,Value,starts)) == True{} : Bool}) ->
  {ListReadback.at(~Value,fallback,index,LineChainValues.lines_taken(~Cell,~Value,~step,remaining,line,inner,group,starts)) ==
    LineChainValues.taken_list(~Cell,~Value,~step,group,Nat.add(line,index),inner,ListReadback.at(~Value,fallback,index,starts)) : Value}:
  match remaining starts index:
    case 0n starts index:
      Empty.absurd({ListReadback.at(~Value,fallback,index,LineChainValues.lines_taken(~Cell,~Value,~step,0n,line,inner,group,starts)) ==
          LineChainValues.taken_list(~Cell,~Value,~step,group,Nat.add(line,index),inner,ListReadback.at(~Value,fallback,index,starts)) : Value},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(index,0n),True{},Equal.sym(Bool,Nat.is_lt(index,0n),False{},Quotients.lt_zero(index)),inside)))
    case 1n+ +rest Nil{} index:
      Empty.absurd({ListReadback.at(~Value,fallback,index,LineChainValues.lines_taken(~Cell,~Value,~step,1n+rest,line,inner,group,[])) ==
          LineChainValues.taken_list(~Cell,~Value,~step,group,Nat.add(line,index),inner,ListReadback.at(~Value,fallback,index,[])) : Value},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(index,0n),True{},Equal.sym(Bool,Nat.is_lt(index,0n),False{},Quotients.lt_zero(index)),enough)))
    case 1n+ +rest Con{+start,+tail} 0n:
      %Equal.sym(Nat,Nat.add(line,0n),line,Addition.zero_right(line)) :
        {LineChainValues.taken_list(~Cell,~Value,~step,group,line,inner,start) == LineChainValues.taken_list(~Cell,~Value,~step,group,_,inner,start) : Value}
      {==}
    case 1n+ +rest Con{+start,+tail} 1n+ +previous:
      %Equal.sym(Nat,Nat.add(line,1n+previous),1n+Nat.add(line,previous),Addition.successor_right(line,previous)) :
        {ListReadback.at(~Value,fallback,previous,LineChainValues.lines_taken(~Cell,~Value,~step,rest,1n+line,inner,group,tail)) ==
          LineChainValues.taken_list(~Cell,~Value,~step,group,_,inner,ListReadback.at(~Value,fallback,previous,tail)) : Value}
      lines_taken_at(~Cell,~Value,~step,rest,1n+line,inner,group,tail,fallback,previous,inside,enough)

# The final of a line of the first block of blocks of values is the turn of
# the walk of the block that takes the values of the line from its start.
def block_first(~Cell: Data,~Value: Data,~Final: Data,~step: Value -> Cell -> Value,~turn_value: Value -> Final,+rest: Nat,+length: U32,+inner: U32,+list: List<&2,Cell>,
  +starts: List<&2,Value>,+zero: Final,+fallback: Value,+line: Nat,+line_inside: {Nat.is_lt(line,U32.to_nat(inner)) == True{} : Bool},
  +enough: {Nat.is_le(U32.to_nat(inner),List.length(&2,Value,starts)) == True{} : Bool}) ->
  {ListReadback.at(~Final,zero,line,Reduce.block_folds(~Cell,~Value,~Final,~step,~turn_value,1n+rest,length,inner,list,starts)) ==
    turn_value(LineChainValues.taken_list(~Cell,~Value,~step,List.take(&2,Cell,list,U32.to_nat((length * inner : U32))),line,U32.to_nat(inner),ListReadback.at(~Value,fallback,line,starts))) : Final}:
  +lines = U32.to_nat(inner)
  +group = List.take(&2,Cell,list,U32.to_nat((length * inner : U32)))
  +taken = LineChainValues.lines_taken(~Cell,~Value,~step,lines,0n,lines,group,starts)
  +finals = LineChainValues.turned_values(~Value,~Final,~turn_value,taken)
  +later = Reduce.block_folds(~Cell,~Value,~Final,~step,~turn_value,rest,length,inner,List.drop(&2,Cell,list,U32.to_nat((length * inner : U32))),starts)
  +taken_count = Equal.sym(Nat,List.length(&2,Value,taken),lines,lines_count(~Cell,~Value,~step,lines,0n,lines,group,starts,enough))
  +taken_inside = {%taken_count : {Nat.is_lt(line,_) == True{} : Bool} line_inside : {Nat.is_lt(line,List.length(&2,Value,taken)) == True{} : Bool}}
  +finals_inside = {%Equal.sym(Nat,List.length(&2,Final,finals),List.length(&2,Value,taken),Reduce.turned_length(~Final,~Value,~turn_value,taken)) : {Nat.is_lt(line,_) == True{} : Bool} taken_inside :
    {Nat.is_lt(line,List.length(&2,Final,finals)) == True{} : Bool}}
  Equal.trans(Final,ListReadback.at(~Final,zero,line,List.append(&2,Final,finals,later)),ListReadback.at(~Final,zero,line,finals),
    turn_value(LineChainValues.taken_list(~Cell,~Value,~step,group,line,lines,ListReadback.at(~Value,fallback,line,starts))),
    ListReadback.appended_before(~Final,zero,finals,later,line,finals_inside),
    Equal.trans(Final,ListReadback.at(~Final,zero,line,finals),turn_value(ListReadback.at(~Value,fallback,line,taken)),
      turn_value(LineChainValues.taken_list(~Cell,~Value,~step,group,line,lines,ListReadback.at(~Value,fallback,line,starts))),
      turned_at(~Value,~Final,~turn_value,zero,fallback,taken,line,taken_inside),
      Equal.cong(Value,Final,folded => turn_value(folded),ListReadback.at(~Value,fallback,line,taken),
        LineChainValues.taken_list(~Cell,~Value,~step,group,line,lines,ListReadback.at(~Value,fallback,line,starts)),
        lines_taken_at(~Cell,~Value,~step,lines,0n,lines,group,starts,fallback,line,line_inside,Order.lt_le(line,lines,List.length(&2,Value,starts),line_inside,enough)))))

# The final of line line of block block of blocks of values read in a row is
# the turn of the walk of the block that takes the values of the line from its
# start.
def block_read(~Cell: Data,~Value: Data,~Final: Data,~step: Value -> Cell -> Value,~turn_value: Value -> Final,~Context: Data,~value: Context -> U32 -> Cell,+context: Context,
  groups: Nat,+length: U32,+inner: U32,+position: Nat,+starts: List<&2,Value>,+zero: Final,+fallback: Value,block: Nat,+line: Nat,
  +block_inside: {Nat.is_lt(block,groups) == True{} : Bool},+line_inside: {Nat.is_lt(line,U32.to_nat(inner)) == True{} : Bool},
  +enough: {Nat.is_le(U32.to_nat(inner),List.length(&2,Value,starts)) == True{} : Bool}) ->
  {ListReadback.at(~Final,zero,Nat.add(Nat.mul(block,U32.to_nat(inner)),line),Reduce.block_folds(~Cell,~Value,~Final,~step,~turn_value,groups,length,inner,
      Reduce.values_from(~Cell,~Context,~value,context,Nat.mul(groups,U32.to_nat((length * inner : U32))),position),starts)) ==
    turn_value(LineChainValues.taken_list(~Cell,~Value,~step,Reduce.values_from(~Cell,~Context,~value,context,U32.to_nat((length * inner : U32)),Nat.add(position,Nat.mul(block,U32.to_nat((length * inner : U32))))),
      line,U32.to_nat(inner),ListReadback.at(~Value,fallback,line,starts))) : Final}:
  match groups block:
    case 0n block:
      Empty.absurd({ListReadback.at(~Final,zero,Nat.add(Nat.mul(block,U32.to_nat(inner)),line),Reduce.block_folds(~Cell,~Value,~Final,~step,~turn_value,0n,length,inner,
          Reduce.values_from(~Cell,~Context,~value,context,0n,position),starts)) ==
        turn_value(LineChainValues.taken_list(~Cell,~Value,~step,Reduce.values_from(~Cell,~Context,~value,context,U32.to_nat((length * inner : U32)),Nat.add(position,Nat.mul(block,U32.to_nat((length * inner : U32))))),
          line,U32.to_nat(inner),ListReadback.at(~Value,fallback,line,starts))) : Final},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(block,0n),True{},Equal.sym(Bool,Nat.is_lt(block,0n),False{},Quotients.lt_zero(block)),block_inside)))
    case 1n+ +rest 0n:
      +size = U32.to_nat((length * inner : U32))
      +start = ListReadback.at(~Value,fallback,line,starts)
      +all = Reduce.values_from(~Cell,~Context,~value,context,Nat.add(size,Nat.mul(rest,size)),position)
      Equal.trans(Final,ListReadback.at(~Final,zero,line,Reduce.block_folds(~Cell,~Value,~Final,~step,~turn_value,1n+rest,length,inner,all,starts)),
        turn_value(LineChainValues.taken_list(~Cell,~Value,~step,List.take(&2,Cell,all,size),line,U32.to_nat(inner),start)),
        turn_value(LineChainValues.taken_list(~Cell,~Value,~step,Reduce.values_from(~Cell,~Context,~value,context,size,Nat.add(position,0n)),line,U32.to_nat(inner),start)),
        block_first(~Cell,~Value,~Final,~step,~turn_value,rest,length,inner,all,starts,zero,fallback,line,line_inside,enough),
        Equal.cong(List<&2,Cell>,Final,list => turn_value(LineChainValues.taken_list(~Cell,~Value,~step,list,line,U32.to_nat(inner),start)),
          List.take(&2,Cell,all,size),Reduce.values_from(~Cell,~Context,~value,context,size,Nat.add(position,0n)),
          %Equal.sym(Nat,Nat.add(position,0n),position,Addition.zero_right(position)) :
            {List.take(&2,Cell,all,size) == Reduce.values_from(~Cell,~Context,~value,context,size,_) : List<&2,Cell>}
          Reduce.values_take(~Cell,~Context,~value,context,size,Nat.mul(rest,size),position)))
    case 1n+ +rest 1n+ +previous:
      +size = U32.to_nat((length * inner : U32))
      +lines = U32.to_nat(inner)
      +all = Reduce.values_from(~Cell,~Context,~value,context,Nat.add(size,Nat.mul(rest,size)),position)
      +taken = LineChainValues.lines_taken(~Cell,~Value,~step,lines,0n,lines,List.take(&2,Cell,all,size),starts)
      +finals = LineChainValues.turned_values(~Value,~Final,~turn_value,taken)
      +later_values = Reduce.values_from(~Cell,~Context,~value,context,Nat.mul(rest,size),Nat.add(position,size))
      +later = Reduce.block_folds(~Cell,~Value,~Final,~step,~turn_value,rest,length,inner,List.drop(&2,Cell,all,size),starts)
      +start = ListReadback.at(~Value,fallback,line,starts)
      +index = Nat.add(Nat.mul(previous,lines),line)
      +finals_count = Equal.trans(Nat,List.length(&2,Final,finals),List.length(&2,Value,taken),lines,Reduce.turned_length(~Final,~Value,~turn_value,taken),
        lines_count(~Cell,~Value,~step,lines,0n,lines,List.take(&2,Cell,all,size),starts,enough))
      +index_exact = Equal.trans(Nat,Nat.add(Nat.add(lines,Nat.mul(previous,lines)),line),Nat.add(lines,index),Nat.add(List.length(&2,Final,finals),index),
        Addition.associative(lines,Nat.mul(previous,lines),line),
        Equal.cong(Nat,Nat,first => Nat.add(first,index),lines,List.length(&2,Final,finals),Equal.sym(Nat,List.length(&2,Final,finals),lines,finals_count)))
      +walk = turn_value(LineChainValues.taken_list(~Cell,~Value,~step,Reduce.values_from(~Cell,~Context,~value,context,size,Nat.add(position,Nat.add(size,Nat.mul(previous,size)))),line,lines,start))
      Equal.trans(Final,ListReadback.at(~Final,zero,Nat.add(Nat.add(lines,Nat.mul(previous,lines)),line),List.append(&2,Final,finals,later)),
        ListReadback.at(~Final,zero,Nat.add(List.length(&2,Final,finals),index),List.append(&2,Final,finals,later)),walk,
        Equal.cong(Nat,Final,at => ListReadback.at(~Final,zero,at,List.append(&2,Final,finals,later)),Nat.add(Nat.add(lines,Nat.mul(previous,lines)),line),Nat.add(List.length(&2,Final,finals),index),index_exact),
        Equal.trans(Final,ListReadback.at(~Final,zero,Nat.add(List.length(&2,Final,finals),index),List.append(&2,Final,finals,later)),ListReadback.at(~Final,zero,index,later),walk,
          ListReadback.appended_after(~Final,zero,finals,later,index),
          Equal.trans(Final,ListReadback.at(~Final,zero,index,later),
            ListReadback.at(~Final,zero,index,Reduce.block_folds(~Cell,~Value,~Final,~step,~turn_value,rest,length,inner,later_values,starts)),walk,
            Equal.cong(List<&2,Cell>,Final,list => ListReadback.at(~Final,zero,index,Reduce.block_folds(~Cell,~Value,~Final,~step,~turn_value,rest,length,inner,list,starts)),
              List.drop(&2,Cell,all,size),later_values,Reduce.values_drop(~Cell,~Context,~value,context,size,Nat.mul(rest,size),position)),
            Equal.trans(Final,ListReadback.at(~Final,zero,index,Reduce.block_folds(~Cell,~Value,~Final,~step,~turn_value,rest,length,inner,later_values,starts)),
              turn_value(LineChainValues.taken_list(~Cell,~Value,~step,Reduce.values_from(~Cell,~Context,~value,context,size,Nat.add(Nat.add(position,size),Nat.mul(previous,size))),line,lines,start)),walk,
              block_read(~Cell,~Value,~Final,~step,~turn_value,~Context,~value,context,rest,length,inner,Nat.add(position,size),starts,zero,fallback,previous,line,block_inside,line_inside,enough),
              Equal.cong(Nat,Final,at => turn_value(LineChainValues.taken_list(~Cell,~Value,~step,Reduce.values_from(~Cell,~Context,~value,context,size,at),line,lines,start)),
                Nat.add(Nat.add(position,size),Nat.mul(previous,size)),Nat.add(position,Nat.add(size,Nat.mul(previous,size))),Addition.associative(position,size,Nat.mul(previous,size)))))))

# The rows of a line of length cells, inner apart, after the kept leading
# axes, read from a reader's values, give the pieces back and the outputs of
# the box of their partials.
def read_lines(~Block: Type,~Model: Data,~Element: Data,~Start: Data,~State: Type,~Value: Data,~Partial: Data,~represent: Model -> Block,~cell: Model -> U32 -> Element,~operation: Value -> Element -> Value,~begin: Start -> State,~begin_value: Start -> Value,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~close: State -> Partial,~close_value: Value -> Partial,~merge: Partial -> Partial -> Partial,~Fill: Data,~vacant: Fill -> Block,~vacant_model: Fill -> Model,~vacant_same: @+fill: Fill -> {vacant(fill) == represent(vacant_model(fill)) : Block},~chain_same: LineChainValues.ChainSame(~Block,~Model,~represent,~Start,~State,~Value,~Partial,~begin,~begin_value,~fold,~LineChainValues.strided_model(~Model,~Element,~Value,~cell,~operation),~close,~close_value,~Fill,~vacant,~vacant_model),~Context: Data,~value: Context -> U32 -> Element,+limits: Machine.Limits,+shape: List<&2,U32>,+kept: Nat,+length: U32,+inner: U32,+contexts: List<&2,Start>,+bytes: F32,+operations: F32,+fill: Fill,+workers: U32,+context: Context,pieces: Partition.Tree<&1,Block,Unit>,+box_extents: List<&2,U32>,+box_flags: List<&2,Bool>,+zero: Partial,+room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+length_exact: {U32.to_nat((length * inner : U32)) == ShapeLaws.elements(List.drop(&2,U32,shape,kept)) : Nat},+turns: Reduce.Turns<Start,contexts,inner>,+inner_positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},+inner_le: {Nat.is_le(U32.to_nat(inner),U32.to_nat((length * inner : U32))) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},+source: Reduce.Streamed<Block,Model,Element,represent,cell,pieces,Pieces.ranges_of_grid(Tensors.grid_of(shape)),Reduce.values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n)>,+within: Within(box_extents,Machine.capacity(limits)),+box_exact: {ShapeLaws.elements(box_extents) == Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,kept)),U32.to_nat(inner)) : Nat},+lengths: {List.length(&2,U32,box_extents) == List.length(&2,Bool,box_flags) : Nat}) ->
  {Partials.regrouped_back(~Block,~Fill,~Partial,~vacant,Tensors.grid_of(shape),fill,Partials.carried_joined(~Block,~Partial,~merge,box_extents,box_flags,LineChain.folds(~Block,~Fill,~Start,~State,~Partial,~begin,~fold,~close,~vacant,contexts,LineChain.Line{length,inner},((length * inner : U32) / Partials.cells_per_outer(Tensors.grid_of(shape)) : U32),bytes,operations,Tensors.grid_of(shape),fill,workers,pieces))) == (pieces,ListReadback.enumerate(~Partial,~Box<Partial>,~box_output(~Partial,~merge),kept_count(axes_with(box_extents,box_flags,False{})),0n,Box{written(~Partial,Reduce.block_folds(~Element,~Value,~Partial,~operation,~close_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,inner,Reduce.values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n),Reduce.block_starts(~Start,~Value,~begin_value,contexts,inner)),zero),axes_with(box_extents,box_flags,False{}),axes_with(box_extents,box_flags,True{}),0,zero})) : Partition.Tree<&1,Block,Unit> & List<&2,Partial>}:
  Reduce.Turns{+laps,+positive,+inner_exact} = turns
  +folded = {%Equal.sym(U32,Partials.cells_per_outer(Tensors.grid_of(shape)),U32.max(Layout.cells_per_outer(Tensors.grid_of(shape)),1),per_outer(Tensors.grid_of(shape))) :
      {Pair.snd(Partition.Tree<&1,Block,Unit>,List<&2,Partial>,LineChain.folds(~Block,~Fill,~Start,~State,~Partial,~begin,~fold,~close,~vacant,contexts,LineChain.Line{length,inner},((length * inner : U32) / _ : U32),bytes,operations,Tensors.grid_of(shape),fill,workers,pieces)) == Reduce.block_folds(~Element,~Value,~Partial,~operation,~close_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,inner,Reduce.values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n),Reduce.block_starts(~Start,~Value,~begin_value,contexts,inner)) : List<&2,Partial>}
    Reduce.trailing_folds(~Block,~Model,~Element,~Start,~State,~Value,~Partial,~represent,~cell,~operation,~begin,~begin_value,~fold,~close,~close_value,~Fill,~vacant,~vacant_model,~vacant_same,~chain_same,~Context,~value,limits,shape,kept,Grid.piece_target(),length,inner,contexts,bytes,operations,fill,workers,context,pieces,room,nonempty,length_exact,turns,inner_positive,inner_le,cap,cap_ok,count_room,source) : {Pair.snd(Partition.Tree<&1,Block,Unit>,List<&2,Partial>,LineChain.folds(~Block,~Fill,~Start,~State,~Partial,~begin,~fold,~close,~vacant,contexts,LineChain.Line{length,inner},((length * inner : U32) / Partials.cells_per_outer(Tensors.grid_of(shape)) : U32),bytes,operations,Tensors.grid_of(shape),fill,workers,pieces)) == Reduce.block_folds(~Element,~Value,~Partial,~operation,~close_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,inner,Reduce.values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n),Reduce.block_starts(~Start,~Value,~begin_value,contexts,inner)) : List<&2,Partial>}}
  +kept_pieces = {%Equal.sym(U32,Partials.cells_per_outer(Tensors.grid_of(shape)),U32.max(Layout.cells_per_outer(Tensors.grid_of(shape)),1),per_outer(Tensors.grid_of(shape))) :
      {Pieces.blocks_regrouped(~Block,~Fill,~vacant,Pieces.ranges_of_grid(Tensors.grid_of(shape)),fill,Pair.fst(Partition.Tree<&1,Block,Unit>,List<&2,Partial>,LineChain.folds(~Block,~Fill,~Start,~State,~Partial,~begin,~fold,~close,~vacant,contexts,LineChain.Line{length,inner},((length * inner : U32) / _ : U32),bytes,operations,Tensors.grid_of(shape),fill,workers,pieces))) == pieces : Partition.Tree<&1,Block,Unit>}
    Reduce.trailing_kept(~Block,~Model,~Element,~Start,~State,~Value,~Partial,~represent,~cell,~operation,~begin,~begin_value,~fold,~close,~close_value,~Fill,~vacant,~vacant_model,~vacant_same,~chain_same,~Context,~value,limits,shape,kept,Grid.piece_target(),length,inner,contexts,bytes,operations,fill,workers,context,pieces,room,nonempty,length_exact,turns,inner_positive,inner_le,cap,cap_ok,count_room,source) : {Pieces.blocks_regrouped(~Block,~Fill,~vacant,Pieces.ranges_of_grid(Tensors.grid_of(shape)),fill,Pair.fst(Partition.Tree<&1,Block,Unit>,List<&2,Partial>,LineChain.folds(~Block,~Fill,~Start,~State,~Partial,~begin,~fold,~close,~vacant,contexts,LineChain.Line{length,inner},((length * inner : U32) / Partials.cells_per_outer(Tensors.grid_of(shape)) : U32),bytes,operations,Tensors.grid_of(shape),fill,workers,pieces))) == pieces : Partition.Tree<&1,Block,Unit>}}
  +starts_length = drawn_length(~Start,~Value,~begin_value,U32.to_nat(inner),contexts,contexts,positive,positive)
  +enough = {%Equal.sym(Nat,List.length(&2,Value,Reduce.block_starts(~Start,~Value,~begin_value,contexts,inner)),U32.to_nat(inner),starts_length) : {Nat.is_le(U32.to_nat(inner),_) == True{} : Bool}
    Order.reflexive(U32.to_nat(inner)) : {Nat.is_le(U32.to_nat(inner),List.length(&2,Value,Reduce.block_starts(~Start,~Value,~begin_value,contexts,inner))) == True{} : Bool}}
  +rows = block_count(~Element,~Value,~Partial,~operation,~close_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,inner,Reduce.values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n),Reduce.block_starts(~Start,~Value,~begin_value,contexts,inner),enough)
  +rows_length = Equal.trans(Nat,List.length(&2,Partial,Reduce.block_folds(~Element,~Value,~Partial,~operation,~close_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,inner,Reduce.values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n),Reduce.block_starts(~Start,~Value,~begin_value,contexts,inner))),Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,kept)),U32.to_nat(inner)),ShapeLaws.elements(box_extents),rows,
    Equal.sym(Nat,ShapeLaws.elements(box_extents),Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,kept)),U32.to_nat(inner)),box_exact))
  +split = CountCompletion.elements_split(shape,kept)
  +whole = Equal.trans(Nat,Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,kept)),U32.to_nat((length * inner : U32))),Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept))),ShapeLaws.elements(shape),
    Equal.cong(Nat,Nat,count => Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,kept)),count),U32.to_nat((length * inner : U32)),ShapeLaws.elements(List.drop(&2,U32,shape,kept)),length_exact),
    Equal.sym(Nat,ShapeLaws.elements(shape),Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept))),split))
  +fewer = Held.nat_le_right_moved(Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,kept)),U32.to_nat(inner)),ShapeLaws.elements(shape),Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,kept)),U32.to_nat((length * inner : U32))),
    Equal.sym(Nat,Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,kept)),U32.to_nat((length * inner : U32))),ShapeLaws.elements(shape),whole),
    RowMajor.mul_le_left(ShapeLaws.elements(List.take(&2,U32,shape,kept)),U32.to_nat(inner),U32.to_nat((length * inner : U32)),inner_le))
  +box_room = Held.nat_le_moved(ShapeLaws.elements(box_extents),Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,kept)),U32.to_nat(inner)),U32.to_nat(Machine.capacity(limits)),box_exact,
    Order.transitive(Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,kept)),U32.to_nat(inner)),ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits)),fewer,room))
  +outputs = box_read(~Partial,~merge,limits,box_extents,box_flags,zero,Reduce.block_folds(~Element,~Value,~Partial,~operation,~close_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,inner,Reduce.values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n),Reduce.block_starts(~Start,~Value,~begin_value,contexts,inner)),within,box_room,rows_length,lengths)
  Equal.trans(Partition.Tree<&1,Block,Unit> & List<&2,Partial>,Partials.regrouped_back(~Block,~Fill,~Partial,~vacant,Tensors.grid_of(shape),fill,Partials.carried_joined(~Block,~Partial,~merge,box_extents,box_flags,LineChain.folds(~Block,~Fill,~Start,~State,~Partial,~begin,~fold,~close,~vacant,contexts,LineChain.Line{length,inner},((length * inner : U32) / Partials.cells_per_outer(Tensors.grid_of(shape)) : U32),bytes,operations,Tensors.grid_of(shape),fill,workers,pieces))),
    (Pieces.blocks_regrouped(~Block,~Fill,~vacant,Pieces.ranges_of_grid(Tensors.grid_of(shape)),fill,Pair.fst(Partition.Tree<&1,Block,Unit>,List<&2,Partial>,LineChain.folds(~Block,~Fill,~Start,~State,~Partial,~begin,~fold,~close,~vacant,contexts,LineChain.Line{length,inner},((length * inner : U32) / Partials.cells_per_outer(Tensors.grid_of(shape)) : U32),bytes,operations,Tensors.grid_of(shape),fill,workers,pieces))),Partials.box_outputs(~Partial,~merge,box_extents,box_flags,Pair.snd(Partition.Tree<&1,Block,Unit>,List<&2,Partial>,LineChain.folds(~Block,~Fill,~Start,~State,~Partial,~begin,~fold,~close,~vacant,contexts,LineChain.Line{length,inner},((length * inner : U32) / Partials.cells_per_outer(Tensors.grid_of(shape)) : U32),bytes,operations,Tensors.grid_of(shape),fill,workers,pieces)))),(pieces,ListReadback.enumerate(~Partial,~Box<Partial>,~box_output(~Partial,~merge),kept_count(axes_with(box_extents,box_flags,False{})),0n,Box{written(~Partial,Reduce.block_folds(~Element,~Value,~Partial,~operation,~close_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,inner,Reduce.values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n),Reduce.block_starts(~Start,~Value,~begin_value,contexts,inner)),zero),axes_with(box_extents,box_flags,False{}),axes_with(box_extents,box_flags,True{}),0,zero})),
    read_pair(~Block,~Fill,~Partial,~merge,~vacant,Tensors.grid_of(shape),fill,box_extents,box_flags,LineChain.folds(~Block,~Fill,~Start,~State,~Partial,~begin,~fold,~close,~vacant,contexts,LineChain.Line{length,inner},((length * inner : U32) / Partials.cells_per_outer(Tensors.grid_of(shape)) : U32),bytes,operations,Tensors.grid_of(shape),fill,workers,pieces)),
    {%Equal.sym(Partition.Tree<&1,Block,Unit>,Pieces.blocks_regrouped(~Block,~Fill,~vacant,Pieces.ranges_of_grid(Tensors.grid_of(shape)),fill,Pair.fst(Partition.Tree<&1,Block,Unit>,List<&2,Partial>,LineChain.folds(~Block,~Fill,~Start,~State,~Partial,~begin,~fold,~close,~vacant,contexts,LineChain.Line{length,inner},((length * inner : U32) / Partials.cells_per_outer(Tensors.grid_of(shape)) : U32),bytes,operations,Tensors.grid_of(shape),fill,workers,pieces))),pieces,kept_pieces) : {(_,Partials.box_outputs(~Partial,~merge,box_extents,box_flags,Pair.snd(Partition.Tree<&1,Block,Unit>,List<&2,Partial>,LineChain.folds(~Block,~Fill,~Start,~State,~Partial,~begin,~fold,~close,~vacant,contexts,LineChain.Line{length,inner},((length * inner : U32) / Partials.cells_per_outer(Tensors.grid_of(shape)) : U32),bytes,operations,Tensors.grid_of(shape),fill,workers,pieces)))) == (pieces,ListReadback.enumerate(~Partial,~Box<Partial>,~box_output(~Partial,~merge),kept_count(axes_with(box_extents,box_flags,False{})),0n,Box{written(~Partial,Reduce.block_folds(~Element,~Value,~Partial,~operation,~close_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,inner,Reduce.values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n),Reduce.block_starts(~Start,~Value,~begin_value,contexts,inner)),zero),axes_with(box_extents,box_flags,False{}),axes_with(box_extents,box_flags,True{}),0,zero})) : Partition.Tree<&1,Block,Unit> & List<&2,Partial>}
      %Equal.sym(List<&2,Partial>,Pair.snd(Partition.Tree<&1,Block,Unit>,List<&2,Partial>,LineChain.folds(~Block,~Fill,~Start,~State,~Partial,~begin,~fold,~close,~vacant,contexts,LineChain.Line{length,inner},((length * inner : U32) / Partials.cells_per_outer(Tensors.grid_of(shape)) : U32),bytes,operations,Tensors.grid_of(shape),fill,workers,pieces)),Reduce.block_folds(~Element,~Value,~Partial,~operation,~close_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,inner,Reduce.values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n),Reduce.block_starts(~Start,~Value,~begin_value,contexts,inner)),folded) : {(pieces,Partials.box_outputs(~Partial,~merge,box_extents,box_flags,_)) == (pieces,ListReadback.enumerate(~Partial,~Box<Partial>,~box_output(~Partial,~merge),kept_count(axes_with(box_extents,box_flags,False{})),0n,Box{written(~Partial,Reduce.block_folds(~Element,~Value,~Partial,~operation,~close_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,inner,Reduce.values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n),Reduce.block_starts(~Start,~Value,~begin_value,contexts,inner)),zero),axes_with(box_extents,box_flags,False{}),axes_with(box_extents,box_flags,True{}),0,zero})) : Partition.Tree<&1,Block,Unit> & List<&2,Partial>}
      %Equal.sym(List<&2,Partial>,Partials.box_outputs(~Partial,~merge,box_extents,box_flags,Reduce.block_folds(~Element,~Value,~Partial,~operation,~close_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,inner,Reduce.values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n),Reduce.block_starts(~Start,~Value,~begin_value,contexts,inner))),ListReadback.enumerate(~Partial,~Box<Partial>,~box_output(~Partial,~merge),kept_count(axes_with(box_extents,box_flags,False{})),0n,Box{written(~Partial,Reduce.block_folds(~Element,~Value,~Partial,~operation,~close_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,inner,Reduce.values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n),Reduce.block_starts(~Start,~Value,~begin_value,contexts,inner)),zero),axes_with(box_extents,box_flags,False{}),axes_with(box_extents,box_flags,True{}),0,zero}),outputs) : {(pieces,_) == (pieces,ListReadback.enumerate(~Partial,~Box<Partial>,~box_output(~Partial,~merge),kept_count(axes_with(box_extents,box_flags,False{})),0n,Box{written(~Partial,Reduce.block_folds(~Element,~Value,~Partial,~operation,~close_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,inner,Reduce.values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n),Reduce.block_starts(~Start,~Value,~begin_value,contexts,inner)),zero),axes_with(box_extents,box_flags,False{}),axes_with(box_extents,box_flags,True{}),0,zero})) : Partition.Tree<&1,Block,Unit> & List<&2,Partial>}
      {==} : {(Pieces.blocks_regrouped(~Block,~Fill,~vacant,Pieces.ranges_of_grid(Tensors.grid_of(shape)),fill,Pair.fst(Partition.Tree<&1,Block,Unit>,List<&2,Partial>,LineChain.folds(~Block,~Fill,~Start,~State,~Partial,~begin,~fold,~close,~vacant,contexts,LineChain.Line{length,inner},((length * inner : U32) / Partials.cells_per_outer(Tensors.grid_of(shape)) : U32),bytes,operations,Tensors.grid_of(shape),fill,workers,pieces))),Partials.box_outputs(~Partial,~merge,box_extents,box_flags,Pair.snd(Partition.Tree<&1,Block,Unit>,List<&2,Partial>,LineChain.folds(~Block,~Fill,~Start,~State,~Partial,~begin,~fold,~close,~vacant,contexts,LineChain.Line{length,inner},((length * inner : U32) / Partials.cells_per_outer(Tensors.grid_of(shape)) : U32),bytes,operations,Tensors.grid_of(shape),fill,workers,pieces)))) == (pieces,ListReadback.enumerate(~Partial,~Box<Partial>,~box_output(~Partial,~merge),kept_count(axes_with(box_extents,box_flags,False{})),0n,Box{written(~Partial,Reduce.block_folds(~Element,~Value,~Partial,~operation,~close_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,inner,Reduce.values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n),Reduce.block_starts(~Start,~Value,~begin_value,contexts,inner)),zero),axes_with(box_extents,box_flags,False{}),axes_with(box_extents,box_flags,True{}),0,zero})) : Partition.Tree<&1,Block,Unit> & List<&2,Partial>}})
def successor_back(+first: Nat,+second: Nat,+same: {1n+first == 1n+second : Nat}) -> {first == second : Nat}:
  Equal.trans(Nat,first,Nat.sub(1n+first,1n),second,Equal.sym(Nat,Nat.sub(1n+first,1n),first,Order.subtract_zero(first)),
    Equal.trans(Nat,Nat.sub(1n+first,1n),Nat.sub(1n+second,1n),second,Equal.cong(Nat,Nat,value => Nat.sub(value,1n),1n+first,1n+second,same),Order.subtract_zero(second)))

def successor_after(n: Nat,+positive: {Nat.is_lt(0n,n) == True{} : Bool}) -> {Nat.add(Nat.sub(n,1n),1n) == n : Nat}:
  match n:
    case 0n: Empty.absurd({Nat.add(Nat.sub(0n,1n),1n) == 0n : Nat},WordEncoding.false_true(positive))
    case 1n+ +rest:
      %Equal.sym(Nat,Nat.sub(rest,0n),rest,Order.subtract_zero(rest)) : {Nat.add(_,1n) == 1n+rest : Nat}
      Equal.trans(Nat,Nat.add(rest,1n),1n+Nat.add(rest,0n),1n+rest,Addition.successor_right(rest,0n),
        Equal.cong(Nat,Nat,value => 1n+value,Nat.add(rest,0n),rest,Addition.zero_right(rest)))

# The extents from the last axis on are the last extent alone.
def dropped_last(shape: List<&2,U32>,index: Nat,+exact: {Nat.add(index,1n) == List.length(&2,U32,shape) : Nat}) ->
  {List.drop(&2,U32,shape,index) == [Grid.lanes_at(shape,index)] : List<&2,U32>}:
  match shape index:
    case [] 0n: Empty.absurd({List.drop(&2,U32,[],0n) == [Grid.lanes_at([],0n)] : List<&2,U32>},Digits.zero_successor(0n,Equal.sym(Nat,1n,0n,exact)))
    case [] 1n+rest:
      Empty.absurd({List.drop(&2,U32,[],1n+rest) == [Grid.lanes_at([],1n+rest)] : List<&2,U32>},Digits.zero_successor(Nat.add(rest,1n),Equal.sym(Nat,1n+Nat.add(rest,1n),0n,exact)))
    case +extent <> +tail 0n:
      %Equal.sym(List<&2,U32>,tail,[],LineChainValues.empty_of_length(~U32,tail,Equal.sym(Nat,0n,List.length(&2,U32,tail),successor_back(0n,List.length(&2,U32,tail),exact)))) :
        {extent <> _ == [extent] : List<&2,U32>}
      {==}
    case +extent <> +tail 1n+ +rest:
      dropped_last(tail,rest,successor_back(Nat.add(rest,1n),List.length(&2,U32,tail),exact))

# A reduced last axis lies in a shape of one axis or more.
def flagged_positive(rank: Nat,+axes: List<&2,Nat>,+flagged: {Partials.flag_at(Layout.reduced_flags(rank,0n,axes),Nat.sub(rank,1n)) == True{} : Bool}) ->
  {Nat.is_lt(0n,rank) == True{} : Bool}:
  match rank:
    case 0n: Empty.absurd({Nat.is_lt(0n,0n) == True{} : Bool},WordEncoding.false_true(flagged))
    case 1n+rest: {==}

# The values after the leading axes of a shape with values are some.
def tail_positive(+shape: List<&2,U32>,+kept: Nat,+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool}) -> {Nat.is_lt(0n,ShapeLaws.elements(List.drop(&2,U32,shape,kept))) == True{} : Bool}:
  +split = CountCompletion.elements_split(shape,kept)
  GridRanges.right_factor_positive(ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept)),Held.nat_lt_moved(0n,Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept))),ShapeLaws.elements(shape),Equal.sym(Nat,ShapeLaws.elements(shape),Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept))),split),nonempty))

# The pieces of a tensor's model under a contract, as a stream of its values.
def contract_streamed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+model: PiecesModel.Model<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  +contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers})) ->
  Reduce.Streamed<Storage.Buffer<Element>,Buffer.Model<Element>,Element,Buffer.pack(~Element),Reduce.cell(~Element),PiecesModel.pack(~Element,model),Pieces.ranges_of_grid(Tensors.grid_of(shape)),
    Reduce.values_from(~Element,~Logical,~original,logical,ShapeLaws.elements(shape),0n)>:
  KernelGemm.Both{KernelGemm.Both{Content.WellFormed{+fields,+ready_pieces,+coverage},KernelGemm.Both{+counted,+canonical}},+values} = contract
  +packed = PiecesModel.packed_contents(~Element,model)
  Reduce.model_streamed(~Element,~Logical,~original,logical,model,Pieces.ranges_of_grid(Tensors.grid_of(shape)),ShapeLaws.elements(shape),
    {%packed : {PiecesModel.content_layout(~Element,_) == Pieces.ranges_of_grid(Tensors.grid_of(shape)) : Pieces.Ranges} canonical : {PiecesModel.content_layout(~Element,PiecesModel.content(~Element,model)) == Pieces.ranges_of_grid(Tensors.grid_of(shape)) : Pieces.Ranges}},
    {%packed : PiecesPlan.Coverage(PiecesModel.content_layout(~Element,_),0n,ShapeLaws.elements(shape)) coverage : PiecesPlan.Coverage(PiecesModel.content_layout(~Element,PiecesModel.content(~Element,model)),0n,ShapeLaws.elements(shape))},
    {%packed : Content.Values(~Element,~Logical,~original,logical,_) values : Content.Values(~Element,~Logical,~original,logical,PiecesModel.content(~Element,model))})

# Positive extents within a capacity lie within it each.
def within_of(extents: List<&2,U32>,+cap: U32,+nonzero: {Shapes.has_zero(extents) == False{} : Bool},+room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(cap)) == True{} : Bool}) -> Within(extents,cap):
  match extents:
    case Nil{}: Unit{}
    case Con{+extent,+rest}:
      +head_nonzero = Boolean.or_false_left(U32.is_zero(extent),Shapes.has_zero(rest),nonzero)
      +rest_nonzero = Boolean.or_false_right(U32.is_zero(extent),Shapes.has_zero(rest),nonzero)
      +positive = Arithmetic.word_nonzero_positive(extent,head_nonzero)
      +rest_positive = CountCompletion.positive_elements(rest,rest_nonzero)
      +head_room = Order.transitive(U32.to_nat(extent),ShapeLaws.elements(Con{extent,rest}),U32.to_nat(cap),RowMajor.product_covers(U32.to_nat(extent),ShapeLaws.elements(rest),rest_positive),room)
      +rest_room = Order.transitive(ShapeLaws.elements(rest),ShapeLaws.elements(Con{extent,rest}),U32.to_nat(cap),DivisionOrder.le_multiple(U32.to_nat(extent),ShapeLaws.elements(rest),positive),room)
      KernelGemm.Both{positive,KernelGemm.Both{head_room,within_of(rest,cap,rest_nonzero,rest_room)}}

def within_removed(extents: List<&2,U32>,axis: Nat,+cap: U32,+within: Within(extents,cap)) -> Within(Grid.removed(extents,axis),cap):
  match extents axis:
    case Nil{} axis: Unit{}
    case Con{+extent,+rest} 0n:
      KernelGemm.Both{+positive,+more} = within
      KernelGemm.Both{+bounded,+later} = more
      later
    case Con{+extent,+rest} 1n+ +next:
      KernelGemm.Both{+positive,+more} = within
      KernelGemm.Both{+bounded,+later} = more
      KernelGemm.Both{positive,KernelGemm.Both{bounded,within_removed(rest,next,cap,later)}}

# The value of output index of a box: the finish of the merged partial.
def box_value(~Element: Data,~Partial: Data,~merge: Partial -> Partial -> Partial,~finish: Partial -> Element,box: Box<Partial>,index: U32) -> Element:
  finish(box_output(~Partial,~merge,box,U32.to_nat(index)))

def box_finished(~Element: Data,~Partial: Data,~merge: Partial -> Partial -> Partial,~finish: Partial -> Element,box: Box<Partial>,index: Nat) -> Element:
  finish(box_output(~Partial,~merge,box,index))

def finished_enumerate(~Element: Data,~Partial: Data,~merge: Partial -> Partial -> Partial,~finish: Partial -> Element,remaining: Nat,+first: Nat,+box: Box<Partial>) ->
  {Partials.finished(~Element,~Partial,~finish,ListReadback.enumerate(~Partial,~Box<Partial>,~box_output(~Partial,~merge),remaining,first,box)) == ListReadback.enumerate(~Element,~Box<Partial>,~box_finished(~Element,~Partial,~merge,~finish),remaining,first,box) : List<&2,Element>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      Equal.cong(List<&2,Element>,List<&2,Element>,tail => finish(box_output(~Partial,~merge,box,first)) <> tail,
        Partials.finished(~Element,~Partial,~finish,ListReadback.enumerate(~Partial,~Box<Partial>,~box_output(~Partial,~merge),rest,1n+first,box)),ListReadback.enumerate(~Element,~Box<Partial>,~box_finished(~Element,~Partial,~merge,~finish),rest,1n+first,box),
        finished_enumerate(~Element,~Partial,~merge,~finish,rest,1n+first,box))

# An output read through a word is the output of its index.
def box_word(~Element: Data,~Partial: Data,~merge: Partial -> Partial -> Partial,~finish: Partial -> Element,box: Box<Partial>,+index: Nat) -> {box_value(~Element,~Partial,~merge,~finish,box,U32.from_nat(index)) == box_finished(~Element,~Partial,~merge,~finish,box,index) : Element}:
  match box:
    case Box{+model,+kept,+rows,+offset,+zero}:
      Equal.cong(U32,Element,word => finish(tree_of(~Partial,~merge,~Tree.Tree<Partial>,~model_read(~Partial),model,depth_of(rows),U32.is_le(first_extent(rows),1),rows,0,
          first_extent(rows),address(kept,word,offset),zero)),
        U32.from_nat(U32.to_nat(U32.from_nat(index))),U32.from_nat(index),Arithmetic.native_roundtrip(U32.from_nat(index)))

# A list of the finished outputs of a box holds the value of the box at every
# index below its length.
def listed_at(~Element: Data,~Partial: Data,~merge: Partial -> Partial -> Partial,~finish: Partial -> Element,+fill: Element,+values: List<&2,Element>,+outputs: Nat,+box: Box<Partial>,+same: {values == ListReadback.enumerate(~Element,~Box<Partial>,~box_finished(~Element,~Partial,~merge,~finish),outputs,0n,box) : List<&2,Element>},
  +total: U32,+equal: {List.length(&2,Element,values) == U32.to_nat(total) : Nat},+index: Nat,+inside: {Nat.is_lt(index,U32.to_nat(total)) == True{} : Bool}) ->
  {ListReadback.at(~Element,fill,index,values) == box_value(~Element,~Partial,~merge,~finish,box,U32.from_nat(index)) : Element}:
  +counted = Equal.trans(Nat,outputs,List.length(&2,Element,ListReadback.enumerate(~Element,~Box<Partial>,~box_finished(~Element,~Partial,~merge,~finish),outputs,0n,box)),U32.to_nat(total),
    Equal.sym(Nat,List.length(&2,Element,ListReadback.enumerate(~Element,~Box<Partial>,~box_finished(~Element,~Partial,~merge,~finish),outputs,0n,box)),outputs,ListReadback.enumerate_length(~Element,~Box<Partial>,~box_finished(~Element,~Partial,~merge,~finish),outputs,0n,box)),
    Equal.trans(Nat,List.length(&2,Element,ListReadback.enumerate(~Element,~Box<Partial>,~box_finished(~Element,~Partial,~merge,~finish),outputs,0n,box)),List.length(&2,Element,values),U32.to_nat(total),
      Equal.cong(List<&2,Element>,Nat,list => List.length(&2,Element,list),ListReadback.enumerate(~Element,~Box<Partial>,~box_finished(~Element,~Partial,~merge,~finish),outputs,0n,box),values,Equal.sym(List<&2,Element>,values,ListReadback.enumerate(~Element,~Box<Partial>,~box_finished(~Element,~Partial,~merge,~finish),outputs,0n,box),same)),equal))
  +within = Held.nat_lt_moved(index,outputs,U32.to_nat(total),counted,inside)
  Equal.trans(Element,ListReadback.at(~Element,fill,index,values),ListReadback.at(~Element,fill,index,ListReadback.enumerate(~Element,~Box<Partial>,~box_finished(~Element,~Partial,~merge,~finish),outputs,0n,box)),box_value(~Element,~Partial,~merge,~finish,box,U32.from_nat(index)),
    Equal.cong(List<&2,Element>,Element,list => ListReadback.at(~Element,fill,index,list),values,ListReadback.enumerate(~Element,~Box<Partial>,~box_finished(~Element,~Partial,~merge,~finish),outputs,0n,box),same),
    Equal.trans(Element,ListReadback.at(~Element,fill,index,ListReadback.enumerate(~Element,~Box<Partial>,~box_finished(~Element,~Partial,~merge,~finish),outputs,0n,box)),box_finished(~Element,~Partial,~merge,~finish,box,index),box_value(~Element,~Partial,~merge,~finish,box,U32.from_nat(index)),
      ListReadback.enumerate_at(~Element,~Box<Partial>,~box_finished(~Element,~Partial,~merge,~finish),fill,outputs,index,0n,box,within),
      Equal.sym(Element,box_value(~Element,~Partial,~merge,~finish,box,U32.from_nat(index)),box_finished(~Element,~Partial,~merge,~finish,box,index),box_word(~Element,~Partial,~merge,~finish,box,index))))

# A list of the finished outputs of a box, in order, shaped and with workers,
# is a tensor of the contract of the box.
def listed_contract(~Element: Data,~Partial: Data,~merge: Partial -> Partial -> Partial,~finish: Partial -> Element,+box: Box<Partial>,+extents: List<&2,U32>,
  +values: List<&2,Element>,+fill: Element,+workers: U32,+outputs: Nat,
  +same: {values == ListReadback.enumerate(~Element,~Box<Partial>,~box_finished(~Element,~Partial,~merge,~finish),outputs,0n,box) : List<&2,Element>},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  ready: {Contract.tensor_ready(~Element,Tensors.with_workers(~Element,Tensors.from_list(~Element,extents,values,fill),workers)) == True{} : Bool}) ->
  Global.Contract(~Element,~Box<Partial>,~box_value(~Element,~Partial,~merge,~finish),box,Tensors.with_workers(~Element,Tensors.from_list(~Element,extents,values,fill),workers)):
  Global.with_workers(~Element,~Box<Partial>,~box_value(~Element,~Partial,~merge,~finish),box,Tensors.from_list(~Element,extents,values,fill),workers,ready,
    inner => Listed.from_list_read(~Element,~Box<Partial>,~box_value(~Element,~Partial,~merge,~finish),box,extents,values,fill,
      count => counted => equal => index => inside => listed_at(~Element,~Partial,~merge,~finish,fill,values,outputs,box,same,count,equal,index,inside),
      nonempty,inner))

# The same, with its extents.
def listed_shaped(~Element: Data,~Partial: Data,~merge: Partial -> Partial -> Partial,~finish: Partial -> Element,+box: Box<Partial>,+extents: List<&2,U32>,
  +values: List<&2,Element>,+fill: Element,+workers: U32,+outputs: Nat,
  +same: {values == ListReadback.enumerate(~Element,~Box<Partial>,~box_finished(~Element,~Partial,~merge,~finish),outputs,0n,box) : List<&2,Element>},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool},
  +ready: {Contract.tensor_ready(~Element,Tensors.with_workers(~Element,Tensors.from_list(~Element,extents,values,fill),workers)) == True{} : Bool}) ->
  Global.Shaped(~Element,~Box<Partial>,~box_value(~Element,~Partial,~merge,~finish),box,extents,Tensors.with_workers(~Element,Tensors.from_list(~Element,extents,values,fill),workers)):
  Global.shaped(~Element,~Box<Partial>,~box_value(~Element,~Partial,~merge,~finish),box,extents,Tensors.with_workers(~Element,Tensors.from_list(~Element,extents,values,fill),workers),ready,
    Equal.trans(List<&2,U32>,Contract.extents(~Element,Tensors.with_workers(~Element,Tensors.from_list(~Element,extents,values,fill),workers)),Contract.extents(~Element,Tensors.from_list(~Element,extents,values,fill)),extents,
      Reduce.workers_extents(~Element,Tensors.from_list(~Element,extents,values,fill),workers),Listed.from_list_extents(~Element,extents,values,fill)),
    listed_contract(~Element,~Partial,~merge,~finish,box,extents,values,fill,workers,outputs,same,nonempty,ready))

def kept_head(flag: Bool,+extent: U32,+rest: List<&2,U32>,+head: {U32.is_zero(extent) == False{} : Bool},+others: {Shapes.has_zero(rest) == False{} : Bool}) ->
  {Shapes.has_zero(Partials.kept_extent(flag,extent,rest)) == False{} : Bool}:
  match flag:
    case True{}: others
    case False{}:
      %Equal.sym(Bool,U32.is_zero(extent),False{},head) : {_ || Shapes.has_zero(rest) == False{} : Bool}
      %Equal.sym(Bool,Shapes.has_zero(rest),False{},others) : {False{} || _ == False{} : Bool}
      {==}

# The kept extents of a shape of no zero extent have none.
def kept_nonzero(shape: List<&2,U32>,flags: List<&2,Bool>,+nonzero: {Shapes.has_zero(shape) == False{} : Bool}) -> {Shapes.has_zero(Partials.kept_extents(shape,flags)) == False{} : Bool}:
  match shape flags:
    case Nil{} Nil{}: {==}
    case Nil{} Con{+flag,+others}: {==}
    case Con{+extent,+rest} Nil{}: {==}
    case Con{+extent,+rest} Con{+flag,+others}:
      kept_head(flag,extent,Partials.kept_extents(rest,others),Boolean.or_false_left(U32.is_zero(extent),Shapes.has_zero(rest),nonzero),
        kept_nonzero(rest,others,Boolean.or_false_right(U32.is_zero(extent),Shapes.has_zero(rest),nonzero)))

# The finished outputs of a reduction of a tensor of the contract of a reader
# are those of the box of the folds of the lines of the reader.
# The kept axes of extents hold the kept extents.
def kept_axes_extents(extents: List<&2,U32>,flags: List<&2,Bool>) ->
  {Views.axis_extents(axes_with(extents,flags,False{})) == Partials.kept_extents(extents,flags) : List<&2,U32>}:
  match extents flags:
    case +extent <> +rest flag <> +others:
      match flag:
        case True{}: kept_axes_extents(rest,others)
        case False{}:
          Equal.cong(List<&2,U32>,List<&2,U32>,tail => extent <> tail,Views.axis_extents(axes_with(rest,others,False{})),Partials.kept_extents(rest,others),kept_axes_extents(rest,others))
    case Nil{} flags: {==}
    case Con{extent,rest} Nil{}: {==}

# Extents without a reduced axis keep the same kept extents.
def removed_kept(extents: List<&2,U32>,flags: List<&2,Bool>,axis: Nat,+flagged: {Partials.flag_at(flags,axis) == True{} : Bool}) ->
  {Partials.kept_extents(Grid.removed(extents,axis),Partials.flag_removed(flags,axis)) == Partials.kept_extents(extents,flags) : List<&2,U32>}:
  match extents flags axis:
    case Nil{} flags axis: {==}
    case Con{+extent,+rest} Nil{} +axis:
      Empty.absurd({Partials.kept_extents(Grid.removed(extent <> rest,axis),Partials.flag_removed([],axis)) == Partials.kept_extents(extent <> rest,[]) : List<&2,U32>},WordEncoding.false_true(flagged))
    case Con{+extent,+rest} Con{True{},+others} 0n: {==}
    case Con{+extent,+rest} Con{False{},+others} 0n:
      Empty.absurd({Partials.kept_extents(rest,others) == Partials.kept_extents(extent <> rest,False{} <> others) : List<&2,U32>},WordEncoding.false_true(flagged))
    case Con{+extent,+rest} Con{+flag,+others} 1n+ +next:
      Equal.cong(List<&2,U32>,List<&2,U32>,tail => Partials.kept_extent(flag,extent,tail),Partials.kept_extents(Grid.removed(rest,next),Partials.flag_removed(others,next)),Partials.kept_extents(rest,others),
        removed_kept(rest,others,next,flagged))

# The rows of a reduction keep the kept extents of its values.
def rows_kept(reduced: Bool,+shape: List<&2,U32>,+flags: List<&2,Bool>,+axis: Nat,+flagged: {Partials.flag_at(flags,axis) == reduced : Bool}) ->
  {Partials.kept_extents(Partials.row_extents(reduced,shape,axis,1n),Partials.row_flags(reduced,flags,axis,1n)) == Partials.kept_extents(shape,flags) : List<&2,U32>}:
  match reduced:
    case True{}: removed_kept(shape,flags,axis,flagged)
    case False{}: {==}

# The kept extents of a nonempty shape hold at most its elements.
def kept_fewer(extents: List<&2,U32>,flags: List<&2,Bool>,+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  {Nat.is_le(ShapeLaws.elements(Partials.kept_extents(extents,flags)),ShapeLaws.elements(extents)) == True{} : Bool}:
  match extents flags:
    case +extent <> +rest flag <> +others:
      match flag:
        case True{}:
          Order.transitive(ShapeLaws.elements(Partials.kept_extents(rest,others)),ShapeLaws.elements(rest),Nat.mul(U32.to_nat(extent),ShapeLaws.elements(rest)),
            kept_fewer(rest,others,GridRanges.right_factor_positive(U32.to_nat(extent),ShapeLaws.elements(rest),nonempty)),
            DivisionOrder.le_multiple(U32.to_nat(extent),ShapeLaws.elements(rest),RowMajor.left_factor_positive(U32.to_nat(extent),ShapeLaws.elements(rest),nonempty)))
        case False{}:
          RowMajor.mul_le_left(U32.to_nat(extent),ShapeLaws.elements(Partials.kept_extents(rest,others)),ShapeLaws.elements(rest),
            kept_fewer(rest,others,GridRanges.right_factor_positive(U32.to_nat(extent),ShapeLaws.elements(rest),nonempty)))
    case Nil{} flags: Order.reflexive(1n)
    case Con{+extent,+rest} Nil{}: Order.positive_one(Nat.mul(U32.to_nat(extent),ShapeLaws.elements(rest)),nonempty)

# The outputs of the box of a reduction are the elements of its kept extents.
def outputs_kept(+reduced: Bool,+shape: List<&2,U32>,+flags: List<&2,Bool>,+axis: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +flagged: {Partials.flag_at(flags,axis) == reduced : Bool},+bound: {Nat.is_le(ShapeLaws.elements(Partials.kept_extents(shape,flags)),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat(Grid.product(Views.axis_extents(axes_with(Partials.row_extents(reduced,shape,axis,1n),Partials.row_flags(reduced,flags,axis,1n),False{})))) == ShapeLaws.elements(Partials.kept_extents(shape,flags)) : Nat}:
  +same = Equal.trans(List<&2,U32>,Views.axis_extents(axes_with(Partials.row_extents(reduced,shape,axis,1n),Partials.row_flags(reduced,flags,axis,1n),False{})),
    Partials.kept_extents(Partials.row_extents(reduced,shape,axis,1n),Partials.row_flags(reduced,flags,axis,1n)),Partials.kept_extents(shape,flags),
    kept_axes_extents(Partials.row_extents(reduced,shape,axis,1n),Partials.row_flags(reduced,flags,axis,1n)),rows_kept(reduced,shape,flags,axis,flagged))
  %Equal.sym(List<&2,U32>,Views.axis_extents(axes_with(Partials.row_extents(reduced,shape,axis,1n),Partials.row_flags(reduced,flags,axis,1n),False{})),Partials.kept_extents(shape,flags),same) :
    {U32.to_nat(Grid.product(_)) == ShapeLaws.elements(Partials.kept_extents(shape,flags)) : Nat}
  TiledPublic.product_exact(cap,cap_ok,Partials.kept_extents(shape,flags),bound)

# A read of the rows of the spanned axes, its line and its box as given.
def read_values(~Block: Type,~Model: Data,~Element: Data,~Start: Data,~State: Type,~Value: Data,~Partial: Data,~represent: Model -> Block,~cell: Model -> U32 -> Element,~operation: Value -> Element -> Value,~begin: Start -> State,~begin_value: Start -> Value,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~close: State -> Partial,~close_value: Value -> Partial,~merge: Partial -> Partial -> Partial,~Fill: Data,~vacant: Fill -> Block,~vacant_model: Fill -> Model,~vacant_same: @+fill: Fill -> {vacant(fill) == represent(vacant_model(fill)) : Block},~chain_same: LineChainValues.ChainSame(~Block,~Model,~represent,~Start,~State,~Value,~Partial,~begin,~begin_value,~fold,~LineChainValues.strided_model(~Model,~Element,~Value,~cell,~operation),~close,~close_value,~Fill,~vacant,~vacant_model),~Context: Data,~value: Context -> U32 -> Element,+limits: Machine.Limits,+shape: List<&2,U32>,+kept: Nat,+length: U32,+inner: U32,+contexts: List<&2,Start>,+bytes: F32,+operations: F32,+fill: Fill,+workers: U32,+context: Context,pieces: Partition.Tree<&1,Block,Unit>,+box_extents: List<&2,U32>,+box_flags: List<&2,Bool>,+zero: Partial,+flags: List<&2,Bool>,+trailing: Nat,+spanned: Nat,+reduced: Bool,
  +flagged: {Partials.flag_at(flags,Nat.sub(List.length(&2,U32,shape),Nat.add(trailing,spanned))) == reduced : Bool},+line_same: {Partials.row_line(reduced,shape,Nat.sub(List.length(&2,U32,shape),Nat.add(trailing,spanned)),spanned) == LineChain.Line{length,inner} : LineChain.Line},
  +extents_same: {Partials.row_extents(reduced,shape,Nat.sub(List.length(&2,U32,shape),Nat.add(trailing,spanned)),spanned) == box_extents : List<&2,U32>},+flags_same: {Partials.row_flags(reduced,flags,Nat.sub(List.length(&2,U32,shape),Nat.add(trailing,spanned)),spanned) == box_flags : List<&2,Bool>},
  +room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+length_exact: {U32.to_nat((length * inner : U32)) == ShapeLaws.elements(List.drop(&2,U32,shape,kept)) : Nat},+turns: Reduce.Turns<Start,contexts,inner>,+inner_positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},+inner_le: {Nat.is_le(U32.to_nat(inner),U32.to_nat((length * inner : U32))) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},+source: Reduce.Streamed<Block,Model,Element,represent,cell,pieces,Pieces.ranges_of_grid(Tensors.grid_of(shape)),Reduce.values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n)>,+within: Within(box_extents,Machine.capacity(limits)),+box_exact: {ShapeLaws.elements(box_extents) == Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,kept)),U32.to_nat(inner)) : Nat},+lengths: {List.length(&2,U32,box_extents) == List.length(&2,Bool,box_flags) : Nat}) ->
  {Partials.read(~Block,~Fill,~Start,~State,~Partial,~begin,~fold,~close,~merge,~vacant,contexts,bytes,operations,shape,flags,trailing,spanned,fill,workers,pieces) == (pieces,ListReadback.enumerate(~Partial,~Box<Partial>,~box_output(~Partial,~merge),kept_count(axes_with(box_extents,box_flags,False{})),0n,Box{written(~Partial,Reduce.block_folds(~Element,~Value,~Partial,~operation,~close_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,inner,Reduce.values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n),Reduce.block_starts(~Start,~Value,~begin_value,contexts,inner)),zero),axes_with(box_extents,box_flags,False{}),axes_with(box_extents,box_flags,True{}),0,zero})) : Partition.Tree<&1,Block,Unit> & List<&2,Partial>}:
  %Equal.sym(Bool,Partials.flag_at(flags,Nat.sub(List.length(&2,U32,shape),Nat.add(trailing,spanned))),reduced,flagged) :
    {Partials.regrouped_back(~Block,~Fill,~Partial,~vacant,Tensors.grid_of(shape),fill,Partials.carried_joined(~Block,~Partial,~merge,Partials.row_extents(_,shape,Nat.sub(List.length(&2,U32,shape),Nat.add(trailing,spanned)),spanned),Partials.row_flags(_,flags,Nat.sub(List.length(&2,U32,shape),Nat.add(trailing,spanned)),spanned),LineChain.folds(~Block,~Fill,~Start,~State,~Partial,~begin,~fold,~close,~vacant,contexts,Partials.row_line(_,shape,Nat.sub(List.length(&2,U32,shape),Nat.add(trailing,spanned)),spanned),(LineChain.cells_of(Partials.row_line(_,shape,Nat.sub(List.length(&2,U32,shape),Nat.add(trailing,spanned)),spanned)) / Partials.cells_per_outer(Tensors.grid_of(shape)) : U32),bytes,operations,Tensors.grid_of(shape),fill,workers,pieces))) == (pieces,ListReadback.enumerate(~Partial,~Box<Partial>,~box_output(~Partial,~merge),kept_count(axes_with(box_extents,box_flags,False{})),0n,Box{written(~Partial,Reduce.block_folds(~Element,~Value,~Partial,~operation,~close_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,inner,Reduce.values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n),Reduce.block_starts(~Start,~Value,~begin_value,contexts,inner)),zero),axes_with(box_extents,box_flags,False{}),axes_with(box_extents,box_flags,True{}),0,zero})) : Partition.Tree<&1,Block,Unit> & List<&2,Partial>}
  %Equal.sym(LineChain.Line,Partials.row_line(reduced,shape,Nat.sub(List.length(&2,U32,shape),Nat.add(trailing,spanned)),spanned),LineChain.Line{length,inner},line_same) :
    {Partials.regrouped_back(~Block,~Fill,~Partial,~vacant,Tensors.grid_of(shape),fill,Partials.carried_joined(~Block,~Partial,~merge,Partials.row_extents(reduced,shape,Nat.sub(List.length(&2,U32,shape),Nat.add(trailing,spanned)),spanned),Partials.row_flags(reduced,flags,Nat.sub(List.length(&2,U32,shape),Nat.add(trailing,spanned)),spanned),LineChain.folds(~Block,~Fill,~Start,~State,~Partial,~begin,~fold,~close,~vacant,contexts,_,(LineChain.cells_of(_) / Partials.cells_per_outer(Tensors.grid_of(shape)) : U32),bytes,operations,Tensors.grid_of(shape),fill,workers,pieces))) == (pieces,ListReadback.enumerate(~Partial,~Box<Partial>,~box_output(~Partial,~merge),kept_count(axes_with(box_extents,box_flags,False{})),0n,Box{written(~Partial,Reduce.block_folds(~Element,~Value,~Partial,~operation,~close_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,inner,Reduce.values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n),Reduce.block_starts(~Start,~Value,~begin_value,contexts,inner)),zero),axes_with(box_extents,box_flags,False{}),axes_with(box_extents,box_flags,True{}),0,zero})) : Partition.Tree<&1,Block,Unit> & List<&2,Partial>}
  %Equal.sym(List<&2,U32>,Partials.row_extents(reduced,shape,Nat.sub(List.length(&2,U32,shape),Nat.add(trailing,spanned)),spanned),box_extents,extents_same) :
    {Partials.regrouped_back(~Block,~Fill,~Partial,~vacant,Tensors.grid_of(shape),fill,Partials.carried_joined(~Block,~Partial,~merge,_,Partials.row_flags(reduced,flags,Nat.sub(List.length(&2,U32,shape),Nat.add(trailing,spanned)),spanned),LineChain.folds(~Block,~Fill,~Start,~State,~Partial,~begin,~fold,~close,~vacant,contexts,LineChain.Line{length,inner},((length * inner : U32) / Partials.cells_per_outer(Tensors.grid_of(shape)) : U32),bytes,operations,Tensors.grid_of(shape),fill,workers,pieces))) == (pieces,ListReadback.enumerate(~Partial,~Box<Partial>,~box_output(~Partial,~merge),kept_count(axes_with(box_extents,box_flags,False{})),0n,Box{written(~Partial,Reduce.block_folds(~Element,~Value,~Partial,~operation,~close_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,inner,Reduce.values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n),Reduce.block_starts(~Start,~Value,~begin_value,contexts,inner)),zero),axes_with(box_extents,box_flags,False{}),axes_with(box_extents,box_flags,True{}),0,zero})) : Partition.Tree<&1,Block,Unit> & List<&2,Partial>}
  %Equal.sym(List<&2,Bool>,Partials.row_flags(reduced,flags,Nat.sub(List.length(&2,U32,shape),Nat.add(trailing,spanned)),spanned),box_flags,flags_same) :
    {Partials.regrouped_back(~Block,~Fill,~Partial,~vacant,Tensors.grid_of(shape),fill,Partials.carried_joined(~Block,~Partial,~merge,box_extents,_,LineChain.folds(~Block,~Fill,~Start,~State,~Partial,~begin,~fold,~close,~vacant,contexts,LineChain.Line{length,inner},((length * inner : U32) / Partials.cells_per_outer(Tensors.grid_of(shape)) : U32),bytes,operations,Tensors.grid_of(shape),fill,workers,pieces))) == (pieces,ListReadback.enumerate(~Partial,~Box<Partial>,~box_output(~Partial,~merge),kept_count(axes_with(box_extents,box_flags,False{})),0n,Box{written(~Partial,Reduce.block_folds(~Element,~Value,~Partial,~operation,~close_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,inner,Reduce.values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n),Reduce.block_starts(~Start,~Value,~begin_value,contexts,inner)),zero),axes_with(box_extents,box_flags,False{}),axes_with(box_extents,box_flags,True{}),0,zero})) : Partition.Tree<&1,Block,Unit> & List<&2,Partial>}
  read_lines(~Block,~Model,~Element,~Start,~State,~Value,~Partial,~represent,~cell,~operation,~begin,~begin_value,~fold,~close,~close_value,~merge,~Fill,~vacant,~vacant_model,~vacant_same,~chain_same,~Context,~value,limits,shape,kept,length,inner,contexts,bytes,operations,fill,workers,context,pieces,box_extents,box_flags,zero,room,nonempty,length_exact,turns,inner_positive,inner_le,cap,cap_ok,count_room,source,within,box_exact,lengths)

# ---- The contract of a reduction ----

# The extents before the last axis are the extents without it.
def removed_last(shape: List<&2,U32>,index: Nat,+exact: {Nat.add(index,1n) == List.length(&2,U32,shape) : Nat}) ->
  {Grid.removed(shape,index) == List.take(&2,U32,shape,index) : List<&2,U32>}:
  match shape index:
    case [] 0n: {==}
    case [] 1n+rest: {==}
    case +extent <> +tail 0n:
      LineChainValues.empty_of_length(~U32,tail,Equal.sym(Nat,0n,List.length(&2,U32,tail),successor_back(0n,List.length(&2,U32,tail),exact)))
    case +extent <> +tail 1n+ +rest:
      Equal.cong(List<&2,U32>,List<&2,U32>,items => extent <> items,Grid.removed(tail,rest),List.take(&2,U32,tail,rest),
        removed_last(tail,rest,successor_back(Nat.add(rest,1n),List.length(&2,U32,tail),exact)))

# The flags of a rank are one an axis.
def flags_length(remaining: Nat,+index: Nat,+reduced: List<&2,Nat>) -> {List.length(&2,Bool,Layout.reduced_flags(remaining,index,reduced)) == remaining : Nat}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Bool,Layout.reduced_flags(rest,1n+index,reduced)),rest,flags_length(rest,1n+index,reduced))

# An axis removed from extents and flags of one length leaves them of one
# length.
def removed_length(extents: List<&2,U32>,flags: List<&2,Bool>,axis: Nat,+same: {List.length(&2,U32,extents) == List.length(&2,Bool,flags) : Nat}) ->
  {List.length(&2,U32,Grid.removed(extents,axis)) == List.length(&2,Bool,Partials.flag_removed(flags,axis)) : Nat}:
  match extents flags axis:
    case Nil{} Nil{} axis: {==}
    case Nil{} Con{+flag,+others} axis:
      Empty.absurd({List.length(&2,U32,Grid.removed([],axis)) == List.length(&2,Bool,Partials.flag_removed(flag <> others,axis)) : Nat},Digits.zero_successor(List.length(&2,Bool,others),same))
    case Con{+extent,+rest} Nil{} axis:
      Empty.absurd({List.length(&2,U32,Grid.removed(extent <> rest,axis)) == List.length(&2,Bool,Partials.flag_removed([],axis)) : Nat},
        Digits.zero_successor(List.length(&2,U32,rest),Equal.sym(Nat,1n+List.length(&2,U32,rest),0n,same)))
    case Con{+extent,+rest} Con{+flag,+others} 0n: successor_back(List.length(&2,U32,rest),List.length(&2,Bool,others),same)
    case Con{+extent,+rest} Con{+flag,+others} 1n+ +next:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,U32,Grid.removed(rest,next)),List.length(&2,Bool,Partials.flag_removed(others,next)),
        removed_length(rest,others,next,successor_back(List.length(&2,U32,rest),List.length(&2,Bool,others),same)))

# The partials of the rows of a reduction, one line of the values after the
# kept leading axes a row, are the partials of the folds of those lines.
def rows_lines(~Element: Data,~Value: Data,~Partial: Data,~operation: Value -> Element -> Value,~close_value: Value -> Partial,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,+kept: Nat,+length: U32,+initial: Value,
  +length_exact: {U32.to_nat((length * 1 : U32)) == ShapeLaws.elements(List.drop(&2,U32,shape,kept)) : Nat}) ->
  {Reduce.block_folds(~Element,~Value,~Partial,~operation,~close_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,1,Reduce.values_from(~Element,~Logical,~original,logical,ShapeLaws.elements(shape),0n),[initial]) == LineChainValues.turned_values(~Value,~Partial,~close_value,Reduce.line_folds(~Element,~Value,~operation,~Logical,~original,logical,ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept)),0n,initial)) : List<&2,Partial>}:
  +single = Reduce.single_blocks(~Element,~Value,~Partial,~operation,~close_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,Reduce.values_from(~Element,~Logical,~original,logical,ShapeLaws.elements(shape),0n),initial)
  +sized = {%length_exact : {Reduce.block_folds(~Element,~Value,~Partial,~operation,~close_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,1,Reduce.values_from(~Element,~Logical,~original,logical,ShapeLaws.elements(shape),0n),[initial]) == LineChainValues.turned_values(~Value,~Partial,~close_value,Reduce.chunk_folds(~Element,~Value,~operation,ShapeLaws.elements(List.take(&2,U32,shape,kept)),_,Reduce.values_from(~Element,~Logical,~original,logical,ShapeLaws.elements(shape),0n),initial)) : List<&2,Partial>}
    single : {Reduce.block_folds(~Element,~Value,~Partial,~operation,~close_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,1,Reduce.values_from(~Element,~Logical,~original,logical,ShapeLaws.elements(shape),0n),[initial]) == LineChainValues.turned_values(~Value,~Partial,~close_value,Reduce.chunk_folds(~Element,~Value,~operation,ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept)),Reduce.values_from(~Element,~Logical,~original,logical,ShapeLaws.elements(shape),0n),initial)) : List<&2,Partial>}}
  +split = CountCompletion.elements_split(shape,kept)
  +chunks = {%Equal.sym(Nat,ShapeLaws.elements(shape),Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept))),split) :
      {Reduce.chunk_folds(~Element,~Value,~operation,ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept)),Reduce.values_from(~Element,~Logical,~original,logical,_,0n),initial) == Reduce.line_folds(~Element,~Value,~operation,~Logical,~original,logical,ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept)),0n,initial) : List<&2,Value>}
    Reduce.chunk_lines(~Element,~Value,~operation,~Logical,~original,logical,ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept)),0n,initial) :
      {Reduce.chunk_folds(~Element,~Value,~operation,ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept)),Reduce.values_from(~Element,~Logical,~original,logical,ShapeLaws.elements(shape),0n),initial) == Reduce.line_folds(~Element,~Value,~operation,~Logical,~original,logical,ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept)),0n,initial) : List<&2,Value>}}
  Equal.trans(List<&2,Partial>,Reduce.block_folds(~Element,~Value,~Partial,~operation,~close_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,1,Reduce.values_from(~Element,~Logical,~original,logical,ShapeLaws.elements(shape),0n),[initial]),
    LineChainValues.turned_values(~Value,~Partial,~close_value,Reduce.chunk_folds(~Element,~Value,~operation,ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept)),Reduce.values_from(~Element,~Logical,~original,logical,ShapeLaws.elements(shape),0n),initial)),LineChainValues.turned_values(~Value,~Partial,~close_value,Reduce.line_folds(~Element,~Value,~operation,~Logical,~original,logical,ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept)),0n,initial)),sized,
    Equal.cong(List<&2,Value>,List<&2,Partial>,values => LineChainValues.turned_values(~Value,~Partial,~close_value,values),Reduce.chunk_folds(~Element,~Value,~operation,ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept)),Reduce.values_from(~Element,~Logical,~original,logical,ShapeLaws.elements(shape),0n),initial),
      Reduce.line_folds(~Element,~Value,~operation,~Logical,~original,logical,ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept)),0n,initial),chunks))

# The box of a reduction of the listed axes of the values of a reader of
# shape: the partials of the lines of the last spanned axes when they are
# reduced, otherwise of every value, and the kept and the reduced axes of the
# rows.
def box_at(~Element: Data,~Value: Data,~Partial: Data,~operation: Value -> Element -> Value,~close_value: Value -> Partial,~Logical: Data,~original: Logical -> U32 -> Element,+reduced: Bool,+logical: Logical,+initial: Value,+zero: Partial,
  +shape: List<&2,U32>,+axes: List<&2,Nat>,+spanned: Nat) -> Box<Partial>:
  +extents = Partials.row_extents(reduced,shape,Nat.sub(List.length(&2,U32,shape),spanned),spanned)
  +flags = Partials.row_flags(reduced,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),spanned),spanned)
  +kept = Bool.pick(Nat,reduced,Nat.sub(List.length(&2,U32,shape),spanned),List.length(&2,U32,shape))
  Box{written(~Partial,LineChainValues.turned_values(~Value,~Partial,~close_value,Reduce.line_folds(~Element,~Value,~operation,~Logical,~original,logical,ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept)),0n,initial)),zero),axes_with(extents,flags,False{}),axes_with(extents,flags,True{}),0,zero}

# The box of a reduction of values, its last axis reduced when listed.
def reduced_box(~Element: Data,~Value: Data,~Partial: Data,~operation: Value -> Element -> Value,~close_value: Value -> Partial,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+initial: Value,+zero: Partial,
  +shape: List<&2,U32>,+axes: List<&2,Nat>) -> Box<Partial>:
  box_at(~Element,~Value,~Partial,~operation,~close_value,~Logical,~original,Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n)),logical,initial,zero,shape,axes,1n)

# The partials of a reduction of a tensor of the contract of a reader, its
# rows of length values, are the outputs of the box of the folds of the lines.
def reduce_rows_at(~Element: Data,~Start: Data,~Row: Type,~Value: Data,~Partial: Data,~operation: Value -> Element -> Value,~begin: Start -> Row,~begin_value: Start -> Value,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: Row -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Row,~close: Row -> Partial,~close_value: Value -> Partial,~merge: Partial -> Partial -> Partial,~finish: Partial -> Element,~chain_same: LineChainValues.ChainSame(~Storage.Buffer<Element>,~Buffer.Model<Element>,~Buffer.pack(~Element),~Start,~Row,~Value,~Partial,~begin,~begin_value,~fold,~LineChainValues.strided_model(~Buffer.Model<Element>,~Element,~Value,~Reduce.cell(~Element),~operation),~close,~close_value,~Element,~Pieces.vacant_buffer(~Element),~Reduce.vacant_model(~Element)),~Logical: Data,~original: Logical -> U32 -> Element,+limits: Machine.Limits,+operations: F32,+start: Start,+zero: Partial,+model: PiecesModel.Model<Element>,+shape: List<&2,U32>,+axes: List<&2,Nat>,+fill: Element,+workers: U32,+logical: Logical,+contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers}),+room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},+reduced: Bool,+kept: Nat,+length: U32,+box_extents: List<&2,U32>,+box_flags: List<&2,Bool>,
  +flagged: {Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n)) == reduced : Bool},+line_same: {Partials.row_line(reduced,shape,Nat.sub(List.length(&2,U32,shape),1n),1n) == LineChain.Line{length,1} : LineChain.Line},
  +extents_same: {Partials.row_extents(reduced,shape,Nat.sub(List.length(&2,U32,shape),1n),1n) == box_extents : List<&2,U32>},+flags_same: {Partials.row_flags(reduced,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n),1n) == box_flags : List<&2,Bool>},
  +length_exact: {U32.to_nat((length * 1 : U32)) == ShapeLaws.elements(List.drop(&2,U32,shape,kept)) : Nat},+inner_le: {Nat.is_le(1n,U32.to_nat((length * 1 : U32))) == True{} : Bool},
  +within: Within(box_extents,Machine.capacity(limits)),+box_exact: {ShapeLaws.elements(box_extents) == Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,kept)),1n) : Nat},+lengths: {List.length(&2,U32,box_extents) == List.length(&2,Bool,box_flags) : Nat}) ->
  {Partials.partials_only(~Storage.Buffer<Element>,~Partial,Partials.read(~Storage.Buffer<Element>,~Element,~Start,~Row,~Partial,~begin,~fold,~close,~merge,~Pieces.vacant_buffer(~Element),[start],1.0,operations,shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),0n,1n,fill,workers,PiecesModel.pack(~Element,model))) == ListReadback.enumerate(~Partial,~Box<Partial>,~box_output(~Partial,~merge),kept_count(axes_with(box_extents,box_flags,False{})),0n,Box{written(~Partial,LineChainValues.turned_values(~Value,~Partial,~close_value,Reduce.line_folds(~Element,~Value,~operation,~Logical,~original,logical,ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept)),0n,begin_value(start))),zero),axes_with(box_extents,box_flags,False{}),axes_with(box_extents,box_flags,True{}),0,zero}) : List<&2,Partial>}:
  +source = contract_streamed(~Element,~Logical,~original,logical,model,shape,fill,workers,contract)
  +read = read_values(~Storage.Buffer<Element>,~Buffer.Model<Element>,~Element,~Start,~Row,~Value,~Partial,~Buffer.pack(~Element),~Reduce.cell(~Element),~operation,~begin,~begin_value,~fold,~close,~close_value,~merge,~Element,~Pieces.vacant_buffer(~Element),~Reduce.vacant_model(~Element),~Reduce.vacant_same(~Element),~chain_same,~Logical,~original,limits,shape,kept,length,1,[start],1.0,operations,fill,workers,logical,PiecesModel.pack(~Element,model),box_extents,box_flags,zero,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),0n,1n,reduced,flagged,line_same,extents_same,flags_same,
    room,nonempty,length_exact,{Reduce.Turns{1n,{==},{==}} : Reduce.Turns<Start,[start],1>},{==},inner_le,cap,cap_ok,count_room,source,within,box_exact,lengths)
  +rows = rows_lines(~Element,~Value,~Partial,~operation,~close_value,~Logical,~original,logical,shape,kept,length,begin_value(start),length_exact)
  Equal.trans(List<&2,Partial>,Partials.partials_only(~Storage.Buffer<Element>,~Partial,Partials.read(~Storage.Buffer<Element>,~Element,~Start,~Row,~Partial,~begin,~fold,~close,~merge,~Pieces.vacant_buffer(~Element),[start],1.0,operations,shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),0n,1n,fill,workers,PiecesModel.pack(~Element,model))),
    ListReadback.enumerate(~Partial,~Box<Partial>,~box_output(~Partial,~merge),kept_count(axes_with(box_extents,box_flags,False{})),0n,Box{written(~Partial,Reduce.block_folds(~Element,~Value,~Partial,~operation,~close_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,1,Reduce.values_from(~Element,~Logical,~original,logical,ShapeLaws.elements(shape),0n),[begin_value(start)]),zero),axes_with(box_extents,box_flags,False{}),axes_with(box_extents,box_flags,True{}),0,zero}),
    ListReadback.enumerate(~Partial,~Box<Partial>,~box_output(~Partial,~merge),kept_count(axes_with(box_extents,box_flags,False{})),0n,Box{written(~Partial,LineChainValues.turned_values(~Value,~Partial,~close_value,Reduce.line_folds(~Element,~Value,~operation,~Logical,~original,logical,ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept)),0n,begin_value(start))),zero),axes_with(box_extents,box_flags,False{}),axes_with(box_extents,box_flags,True{}),0,zero}),
    Equal.cong(Partition.Tree<&1,Storage.Buffer<Element>,Unit> & List<&2,Partial>,List<&2,Partial>,pair => Partials.partials_only(~Storage.Buffer<Element>,~Partial,pair),Partials.read(~Storage.Buffer<Element>,~Element,~Start,~Row,~Partial,~begin,~fold,~close,~merge,~Pieces.vacant_buffer(~Element),[start],1.0,operations,shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),0n,1n,fill,workers,PiecesModel.pack(~Element,model)),
      (PiecesModel.pack(~Element,model),ListReadback.enumerate(~Partial,~Box<Partial>,~box_output(~Partial,~merge),kept_count(axes_with(box_extents,box_flags,False{})),0n,Box{written(~Partial,Reduce.block_folds(~Element,~Value,~Partial,~operation,~close_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,1,Reduce.values_from(~Element,~Logical,~original,logical,ShapeLaws.elements(shape),0n),[begin_value(start)]),zero),axes_with(box_extents,box_flags,False{}),axes_with(box_extents,box_flags,True{}),0,zero})),read),
    Equal.cong(List<&2,Partial>,List<&2,Partial>,partials => ListReadback.enumerate(~Partial,~Box<Partial>,~box_output(~Partial,~merge),kept_count(axes_with(box_extents,box_flags,False{})),0n,Box{written(~Partial,partials,zero),axes_with(box_extents,box_flags,False{}),axes_with(box_extents,box_flags,True{}),0,zero}),
      Reduce.block_folds(~Element,~Value,~Partial,~operation,~close_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,1,Reduce.values_from(~Element,~Logical,~original,logical,ShapeLaws.elements(shape),0n),[begin_value(start)]),LineChainValues.turned_values(~Value,~Partial,~close_value,Reduce.line_folds(~Element,~Value,~operation,~Logical,~original,logical,ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept)),0n,begin_value(start))),rows))

# The partials of a reduction of a tensor of the contract of a reader are the
# outputs of its box.
def reduce_rows(~Element: Data,~Start: Data,~Row: Type,~Value: Data,~Partial: Data,~operation: Value -> Element -> Value,~begin: Start -> Row,~begin_value: Start -> Value,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: Row -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Row,~close: Row -> Partial,~close_value: Value -> Partial,~merge: Partial -> Partial -> Partial,~finish: Partial -> Element,~chain_same: LineChainValues.ChainSame(~Storage.Buffer<Element>,~Buffer.Model<Element>,~Buffer.pack(~Element),~Start,~Row,~Value,~Partial,~begin,~begin_value,~fold,~LineChainValues.strided_model(~Buffer.Model<Element>,~Element,~Value,~Reduce.cell(~Element),~operation),~close,~close_value,~Element,~Pieces.vacant_buffer(~Element),~Reduce.vacant_model(~Element)),~Logical: Data,~original: Logical -> U32 -> Element,reduced: Bool,+limits: Machine.Limits,+operations: F32,+start: Start,+zero: Partial,+model: PiecesModel.Model<Element>,+shape: List<&2,U32>,+axes: List<&2,Nat>,+fill: Element,+workers: U32,+logical: Logical,+contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers}),+room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},+flagged: {Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n)) == reduced : Bool}) ->
  {Partials.partials_only(~Storage.Buffer<Element>,~Partial,Partials.read(~Storage.Buffer<Element>,~Element,~Start,~Row,~Partial,~begin,~fold,~close,~merge,~Pieces.vacant_buffer(~Element),[start],1.0,operations,shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),0n,1n,fill,workers,PiecesModel.pack(~Element,model))) == ListReadback.enumerate(~Partial,~Box<Partial>,~box_output(~Partial,~merge),kept_count(axes_with(Partials.row_extents(reduced,shape,Nat.sub(List.length(&2,U32,shape),1n),1n),Partials.row_flags(reduced,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n),1n),False{})),0n,
    box_at(~Element,~Value,~Partial,~operation,~close_value,~Logical,~original,reduced,logical,begin_value(start),zero,shape,axes,1n)) : List<&2,Partial>}:
  match reduced:
    case True{}:
      +positive = flagged_positive(List.length(&2,U32,shape),axes,flagged)
      +exact = successor_after(List.length(&2,U32,shape),positive)
      +last_drop = dropped_last(shape,Nat.sub(List.length(&2,U32,shape),1n),exact)
      +none = Equal.trans(List<&2,U32>,List.drop(&2,U32,shape,Nat.add(Nat.sub(List.length(&2,U32,shape),1n),1n)),List.drop(&2,U32,shape,List.length(&2,U32,shape)),[],
        Equal.cong(Nat,List<&2,U32>,count => List.drop(&2,U32,shape,count),Nat.add(Nat.sub(List.length(&2,U32,shape),1n),1n),List.length(&2,U32,shape),exact),Reduce.dropped_all(~U32,shape))
      +line = Equal.cong(List<&2,U32>,LineChain.Line,items => LineChain.Line{Grid.product(List.take(&2,U32,List.drop(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),1n)),1n)),Grid.product(items)},List.drop(&2,U32,shape,Nat.add(Nat.sub(List.length(&2,U32,shape),1n),1n)),[],none)
      +lanes = Grid.lanes_at(shape,Nat.sub(List.length(&2,U32,shape),1n))
      +length_exact = {%Equal.sym(List<&2,U32>,List.drop(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),1n)),[lanes],last_drop) :
          {U32.to_nat((Grid.product(List.take(&2,U32,_,1n)) * 1 : U32)) == ShapeLaws.elements(_) : Nat}
        Equal.trans(Nat,U32.to_nat(((lanes * 1 : U32) * 1 : U32)),U32.to_nat(lanes),Nat.mul(U32.to_nat(lanes),1n),
          Equal.cong(U32,Nat,word => U32.to_nat(word),((lanes * 1 : U32) * 1 : U32),lanes,
            Equal.trans(U32,((lanes * 1 : U32) * 1 : U32),(lanes * 1 : U32),lanes,Arithmetic.multiply_one((lanes * 1 : U32)),Arithmetic.multiply_one(lanes))),
          Equal.sym(Nat,Nat.mul(U32.to_nat(lanes),1n),U32.to_nat(lanes),Order.times_one(U32.to_nat(lanes)))) :
          {U32.to_nat((Grid.product(List.take(&2,U32,List.drop(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),1n)),1n)) * 1 : U32)) == ShapeLaws.elements(List.drop(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),1n))) : Nat}}
      +drop_positive = tail_positive(shape,Nat.sub(List.length(&2,U32,shape),1n),nonempty)
      +inner_le = Held.nat_le_right_moved(1n,U32.to_nat((Grid.product(List.take(&2,U32,List.drop(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),1n)),1n)) * 1 : U32)),ShapeLaws.elements(List.drop(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),1n))),length_exact,
        Order.positive_one(ShapeLaws.elements(List.drop(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),1n))),drop_positive))
      +box_exact = Equal.trans(Nat,ShapeLaws.elements(Grid.removed(shape,Nat.sub(List.length(&2,U32,shape),1n))),ShapeLaws.elements(List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),1n))),Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),1n))),1n),
        Equal.cong(List<&2,U32>,Nat,items => ShapeLaws.elements(items),Grid.removed(shape,Nat.sub(List.length(&2,U32,shape),1n)),List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),1n)),removed_last(shape,Nat.sub(List.length(&2,U32,shape),1n),exact)),
        Equal.sym(Nat,Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),1n))),1n),ShapeLaws.elements(List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),1n))),Order.times_one(ShapeLaws.elements(List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),1n))))))
      +within = within_removed(shape,Nat.sub(List.length(&2,U32,shape),1n),Machine.capacity(limits),within_of(shape,Machine.capacity(limits),CountCompletion.positive_not_zero(shape,nonempty),room))
      +lengths = removed_length(shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n),Equal.sym(Nat,List.length(&2,Bool,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes)),List.length(&2,U32,shape),flags_length(List.length(&2,U32,shape),0n,axes)))
      reduce_rows_at(~Element,~Start,~Row,~Value,~Partial,~operation,~begin,~begin_value,~fold,~close,~close_value,~merge,~finish,~chain_same,~Logical,~original,limits,operations,start,zero,model,shape,axes,fill,workers,logical,contract,room,nonempty,cap,cap_ok,count_room,True{},Nat.sub(List.length(&2,U32,shape),1n),Grid.product(List.take(&2,U32,List.drop(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),1n)),1n)),Grid.removed(shape,Nat.sub(List.length(&2,U32,shape),1n)),Partials.flag_removed(Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n)),flagged,line,{==},{==},length_exact,inner_le,within,box_exact,lengths)
    case False{}:
      +none = Reduce.dropped_all(~U32,shape)
      +length_exact = {%Equal.sym(List<&2,U32>,List.drop(&2,U32,shape,List.length(&2,U32,shape)),[],none) : {U32.to_nat((1 * 1 : U32)) == ShapeLaws.elements(_) : Nat}
        {==} : {U32.to_nat((1 * 1 : U32)) == ShapeLaws.elements(List.drop(&2,U32,shape,List.length(&2,U32,shape))) : Nat}}
      +split = CountCompletion.elements_split(shape,List.length(&2,U32,shape))
      +box_exact = {%none : {ShapeLaws.elements(shape) == Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,List.length(&2,U32,shape))),ShapeLaws.elements(_)) : Nat}
        split : {ShapeLaws.elements(shape) == Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,List.length(&2,U32,shape))),1n) : Nat}}
      +within = within_of(shape,Machine.capacity(limits),CountCompletion.positive_not_zero(shape,nonempty),room)
      +lengths = Equal.sym(Nat,List.length(&2,Bool,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes)),List.length(&2,U32,shape),flags_length(List.length(&2,U32,shape),0n,axes))
      reduce_rows_at(~Element,~Start,~Row,~Value,~Partial,~operation,~begin,~begin_value,~fold,~close,~close_value,~merge,~finish,~chain_same,~Logical,~original,limits,operations,start,zero,model,shape,axes,fill,workers,logical,contract,room,nonempty,cap,cap_ok,count_room,False{},List.length(&2,U32,shape),1,shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),flagged,{==},{==},{==},length_exact,{==},within,box_exact,lengths)

# The finished outputs of a reduction of a tensor of the contract of a reader
# are those of its box.
def sized_values(~Element: Data,~Start: Data,~Row: Type,~Value: Data,~Partial: Data,~operation: Value -> Element -> Value,~begin: Start -> Row,~begin_value: Start -> Value,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: Row -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Row,~close: Row -> Partial,~close_value: Value -> Partial,~merge: Partial -> Partial -> Partial,~finish: Partial -> Element,~chain_same: LineChainValues.ChainSame(~Storage.Buffer<Element>,~Buffer.Model<Element>,~Buffer.pack(~Element),~Start,~Row,~Value,~Partial,~begin,~begin_value,~fold,~LineChainValues.strided_model(~Buffer.Model<Element>,~Element,~Value,~Reduce.cell(~Element),~operation),~close,~close_value,~Element,~Pieces.vacant_buffer(~Element),~Reduce.vacant_model(~Element)),~Logical: Data,~original: Logical -> U32 -> Element,+limits: Machine.Limits,+operations: F32,+start: Start,+zero: Partial,+model: PiecesModel.Model<Element>,+shape: List<&2,U32>,+axes: List<&2,Nat>,+fill: Element,+workers: U32,+logical: Logical,+contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers}),+room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool}) -> {Partials.finished(~Element,~Partial,~finish,Partials.partials_only(~Storage.Buffer<Element>,~Partial,Partials.read(~Storage.Buffer<Element>,~Element,~Start,~Row,~Partial,~begin,~fold,~close,~merge,~Pieces.vacant_buffer(~Element),[start],1.0,operations,shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),0n,1n,fill,workers,PiecesModel.pack(~Element,model)))) == ListReadback.enumerate(~Element,~Box<Partial>,~box_finished(~Element,~Partial,~merge,~finish),kept_count(axes_with(Partials.row_extents(Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n)),shape,Nat.sub(List.length(&2,U32,shape),1n),1n),Partials.row_flags(Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n)),Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n),1n),False{})),0n,reduced_box(~Element,~Value,~Partial,~operation,~close_value,~Logical,~original,logical,begin_value(start),zero,shape,axes)) : List<&2,Element>}:
  +rows = reduce_rows(~Element,~Start,~Row,~Value,~Partial,~operation,~begin,~begin_value,~fold,~close,~close_value,~merge,~finish,~chain_same,~Logical,~original,Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n)),limits,operations,start,zero,model,shape,axes,fill,workers,logical,contract,room,nonempty,cap,cap_ok,count_room,{==})
  Equal.trans(List<&2,Element>,Partials.finished(~Element,~Partial,~finish,Partials.partials_only(~Storage.Buffer<Element>,~Partial,Partials.read(~Storage.Buffer<Element>,~Element,~Start,~Row,~Partial,~begin,~fold,~close,~merge,~Pieces.vacant_buffer(~Element),[start],1.0,operations,shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),0n,1n,fill,workers,PiecesModel.pack(~Element,model)))),Partials.finished(~Element,~Partial,~finish,ListReadback.enumerate(~Partial,~Box<Partial>,~box_output(~Partial,~merge),kept_count(axes_with(Partials.row_extents(Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n)),shape,Nat.sub(List.length(&2,U32,shape),1n),1n),Partials.row_flags(Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n)),Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n),1n),False{})),0n,reduced_box(~Element,~Value,~Partial,~operation,~close_value,~Logical,~original,logical,begin_value(start),zero,shape,axes))),ListReadback.enumerate(~Element,~Box<Partial>,~box_finished(~Element,~Partial,~merge,~finish),kept_count(axes_with(Partials.row_extents(Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n)),shape,Nat.sub(List.length(&2,U32,shape),1n),1n),Partials.row_flags(Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n)),Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n),1n),False{})),0n,reduced_box(~Element,~Value,~Partial,~operation,~close_value,~Logical,~original,logical,begin_value(start),zero,shape,axes)),
    Equal.cong(List<&2,Partial>,List<&2,Element>,list => Partials.finished(~Element,~Partial,~finish,list),Partials.partials_only(~Storage.Buffer<Element>,~Partial,Partials.read(~Storage.Buffer<Element>,~Element,~Start,~Row,~Partial,~begin,~fold,~close,~merge,~Pieces.vacant_buffer(~Element),[start],1.0,operations,shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),0n,1n,fill,workers,PiecesModel.pack(~Element,model))),ListReadback.enumerate(~Partial,~Box<Partial>,~box_output(~Partial,~merge),kept_count(axes_with(Partials.row_extents(Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n)),shape,Nat.sub(List.length(&2,U32,shape),1n),1n),Partials.row_flags(Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n)),Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n),1n),False{})),0n,reduced_box(~Element,~Value,~Partial,~operation,~close_value,~Logical,~original,logical,begin_value(start),zero,shape,axes)),rows),
    finished_enumerate(~Element,~Partial,~merge,~finish,kept_count(axes_with(Partials.row_extents(Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n)),shape,Nat.sub(List.length(&2,U32,shape),1n),1n),Partials.row_flags(Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n)),Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n),1n),False{})),0n,reduced_box(~Element,~Value,~Partial,~operation,~close_value,~Logical,~original,logical,begin_value(start),zero,shape,axes)))

# A Ready reduction of the listed axes of a tensor of the contract of a
# reader holds at each kept index the finish of the tree of its box.
def sized_shaped(~Element: Data,~Start: Data,~Row: Type,~Value: Data,~Partial: Data,~operation: Value -> Element -> Value,~begin: Start -> Row,~begin_value: Start -> Value,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: Row -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Row,~close: Row -> Partial,~close_value: Value -> Partial,~merge: Partial -> Partial -> Partial,~finish: Partial -> Element,~chain_same: LineChainValues.ChainSame(~Storage.Buffer<Element>,~Buffer.Model<Element>,~Buffer.pack(~Element),~Start,~Row,~Value,~Partial,~begin,~begin_value,~fold,~LineChainValues.strided_model(~Buffer.Model<Element>,~Element,~Value,~Reduce.cell(~Element),~operation),~close,~close_value,~Element,~Pieces.vacant_buffer(~Element),~Reduce.vacant_model(~Element)),~Logical: Data,~original: Logical -> U32 -> Element,+limits: Machine.Limits,+operations: F32,+start: Start,+zero: Partial,+model: PiecesModel.Model<Element>,+shape: List<&2,U32>,+axes: List<&2,Nat>,+fill: Element,+workers: U32,+logical: Logical,+contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers}),+room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32}) ->
  Global.Shaped(~Element,~Box<Partial>,~box_value(~Element,~Partial,~merge,~finish),reduced_box(~Element,~Value,~Partial,~operation,~close_value,~Logical,~original,logical,begin_value(start),zero,shape,axes),Partials.kept_extents(shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes)),Tensors.with_workers(~Element,Tensors.from_list(~Element,Partials.kept_extents(shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes)),Partials.finished(~Element,~Partial,~finish,Partials.partials_only(~Storage.Buffer<Element>,~Partial,Partials.read(~Storage.Buffer<Element>,~Element,~Start,~Row,~Partial,~begin,~fold,~close,~merge,~Pieces.vacant_buffer(~Element),[start],1.0,operations,shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),0n,1n,fill,workers,PiecesModel.pack(~Element,model)))),fill),workers)):
  +values = sized_values(~Element,~Start,~Row,~Value,~Partial,~operation,~begin,~begin_value,~fold,~close,~close_value,~merge,~finish,~chain_same,~Logical,~original,limits,operations,start,zero,model,shape,axes,fill,workers,logical,contract,room,nonempty,cap,cap_ok,count_room)
  +outputs_nonempty = CountCompletion.positive_elements(Partials.kept_extents(shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes)),kept_nonzero(shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),CountCompletion.positive_not_zero(shape,nonempty)))
  +kept_room = Order.transitive(ShapeLaws.elements(Partials.kept_extents(shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes))),ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits)),kept_fewer(shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),nonempty),room)
  +outputs_exact = outputs_kept(Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n)),shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n),Machine.capacity(limits),Limits.capacity_fits(limits),{==},kept_room)
  +values_length = Equal.trans(Nat,List.length(&2,Element,Partials.finished(~Element,~Partial,~finish,Partials.partials_only(~Storage.Buffer<Element>,~Partial,Partials.read(~Storage.Buffer<Element>,~Element,~Start,~Row,~Partial,~begin,~fold,~close,~merge,~Pieces.vacant_buffer(~Element),[start],1.0,operations,shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),0n,1n,fill,workers,PiecesModel.pack(~Element,model))))),List.length(&2,Element,ListReadback.enumerate(~Element,~Box<Partial>,~box_finished(~Element,~Partial,~merge,~finish),kept_count(axes_with(Partials.row_extents(Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n)),shape,Nat.sub(List.length(&2,U32,shape),1n),1n),Partials.row_flags(Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n)),Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n),1n),False{})),0n,reduced_box(~Element,~Value,~Partial,~operation,~close_value,~Logical,~original,logical,begin_value(start),zero,shape,axes))),ShapeLaws.elements(Partials.kept_extents(shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes))),
    Equal.cong(List<&2,Element>,Nat,list => List.length(&2,Element,list),Partials.finished(~Element,~Partial,~finish,Partials.partials_only(~Storage.Buffer<Element>,~Partial,Partials.read(~Storage.Buffer<Element>,~Element,~Start,~Row,~Partial,~begin,~fold,~close,~merge,~Pieces.vacant_buffer(~Element),[start],1.0,operations,shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),0n,1n,fill,workers,PiecesModel.pack(~Element,model)))),ListReadback.enumerate(~Element,~Box<Partial>,~box_finished(~Element,~Partial,~merge,~finish),kept_count(axes_with(Partials.row_extents(Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n)),shape,Nat.sub(List.length(&2,U32,shape),1n),1n),Partials.row_flags(Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n)),Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n),1n),False{})),0n,reduced_box(~Element,~Value,~Partial,~operation,~close_value,~Logical,~original,logical,begin_value(start),zero,shape,axes)),values),
    Equal.trans(Nat,List.length(&2,Element,ListReadback.enumerate(~Element,~Box<Partial>,~box_finished(~Element,~Partial,~merge,~finish),kept_count(axes_with(Partials.row_extents(Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n)),shape,Nat.sub(List.length(&2,U32,shape),1n),1n),Partials.row_flags(Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n)),Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n),1n),False{})),0n,reduced_box(~Element,~Value,~Partial,~operation,~close_value,~Logical,~original,logical,begin_value(start),zero,shape,axes))),kept_count(axes_with(Partials.row_extents(Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n)),shape,Nat.sub(List.length(&2,U32,shape),1n),1n),Partials.row_flags(Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n)),Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n),1n),False{})),ShapeLaws.elements(Partials.kept_extents(shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes))),
      ListReadback.enumerate_length(~Element,~Box<Partial>,~box_finished(~Element,~Partial,~merge,~finish),kept_count(axes_with(Partials.row_extents(Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n)),shape,Nat.sub(List.length(&2,U32,shape),1n),1n),Partials.row_flags(Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n)),Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n),1n),False{})),0n,reduced_box(~Element,~Value,~Partial,~operation,~close_value,~Logical,~original,logical,begin_value(start),zero,shape,axes)),outputs_exact))
  +ready = Listed.listed_ready(~Element,Partials.kept_extents(shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes)),Partials.finished(~Element,~Partial,~finish,Partials.partials_only(~Storage.Buffer<Element>,~Partial,Partials.read(~Storage.Buffer<Element>,~Element,~Start,~Row,~Partial,~begin,~fold,~close,~merge,~Pieces.vacant_buffer(~Element),[start],1.0,operations,shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),0n,1n,fill,workers,PiecesModel.pack(~Element,model)))),fill,workers,StorageCount.counted_within(limits,Partials.kept_extents(shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes)),kept_room,default_capacity),values_length)
  listed_shaped(~Element,~Partial,~merge,~finish,reduced_box(~Element,~Value,~Partial,~operation,~close_value,~Logical,~original,logical,begin_value(start),zero,shape,axes),Partials.kept_extents(shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes)),Partials.finished(~Element,~Partial,~finish,Partials.partials_only(~Storage.Buffer<Element>,~Partial,Partials.read(~Storage.Buffer<Element>,~Element,~Start,~Row,~Partial,~begin,~fold,~close,~merge,~Pieces.vacant_buffer(~Element),[start],1.0,operations,shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),0n,1n,fill,workers,PiecesModel.pack(~Element,model)))),fill,workers,kept_count(axes_with(Partials.row_extents(Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n)),shape,Nat.sub(List.length(&2,U32,shape),1n),1n),Partials.row_flags(Partials.flag_at(Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n)),Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),Nat.sub(List.length(&2,U32,shape),1n),1n),False{})),values,outputs_nonempty,ready)

# A reduction of values that are there holds the box.
def sized_cases(~Element: Data,~Start: Data,~Row: Type,~Value: Data,~Partial: Data,~operation: Value -> Element -> Value,~begin: Start -> Row,~begin_value: Start -> Value,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: Row -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Row,~close: Row -> Partial,~close_value: Value -> Partial,~merge: Partial -> Partial -> Partial,~finish: Partial -> Element,~chain_same: LineChainValues.ChainSame(~Storage.Buffer<Element>,~Buffer.Model<Element>,~Buffer.pack(~Element),~Start,~Row,~Value,~Partial,~begin,~begin_value,~fold,~LineChainValues.strided_model(~Buffer.Model<Element>,~Element,~Value,~Reduce.cell(~Element),~operation),~close,~close_value,~Element,~Pieces.vacant_buffer(~Element),~Reduce.vacant_model(~Element)),~Logical: Data,~original: Logical -> U32 -> Element,+limits: Machine.Limits,+operations: F32,+start: Start,+zero: Partial,+model: PiecesModel.Model<Element>,+shape: List<&2,U32>,+axes: List<&2,Nat>,+fill: Element,+workers: U32,+logical: Logical,+contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers}),+room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},empty: Bool,unread: Bool,+empty_false: {empty == False{} : Bool},+unread_false: {unread == False{} : Bool},+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32}) ->
  Global.Shaped(~Element,~Box<Partial>,~box_value(~Element,~Partial,~merge,~finish),reduced_box(~Element,~Value,~Partial,~operation,~close_value,~Logical,~original,logical,begin_value(start),zero,shape,axes),Partials.kept_extents(shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes)),Partials.reduce_sized(~Element,~Start,~Row,~Partial,~begin,~fold,~close,~merge,~finish,empty,unread,operations,start,zero,PiecesModel.pack(~Element,model),shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),fill,workers)):
  match empty unread:
    case False{} False{}: sized_shaped(~Element,~Start,~Row,~Value,~Partial,~operation,~begin,~begin_value,~fold,~close,~close_value,~merge,~finish,~chain_same,~Logical,~original,limits,operations,start,zero,model,shape,axes,fill,workers,logical,contract,room,nonempty,cap,cap_ok,count_room,default_capacity)
    case False{} True{}:
      Empty.absurd(Global.Shaped(~Element,~Box<Partial>,~box_value(~Element,~Partial,~merge,~finish),reduced_box(~Element,~Value,~Partial,~operation,~close_value,~Logical,~original,logical,begin_value(start),zero,shape,axes),Partials.kept_extents(shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes)),Partials.reduce_sized(~Element,~Start,~Row,~Partial,~begin,~fold,~close,~merge,~finish,False{},True{},operations,start,zero,PiecesModel.pack(~Element,model),shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),fill,workers)),WordEncoding.false_true(Equal.sym(Bool,True{},False{},unread_false)))
    case True{} unread:
      Empty.absurd(Global.Shaped(~Element,~Box<Partial>,~box_value(~Element,~Partial,~merge,~finish),reduced_box(~Element,~Value,~Partial,~operation,~close_value,~Logical,~original,logical,begin_value(start),zero,shape,axes),Partials.kept_extents(shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes)),Partials.reduce_sized(~Element,~Start,~Row,~Partial,~begin,~fold,~close,~merge,~finish,True{},unread,operations,start,zero,PiecesModel.pack(~Element,model),shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),fill,workers)),WordEncoding.false_true(Equal.sym(Bool,True{},False{},empty_false)))

def listed_valid(~Element: Data,~Start: Data,~Row: Type,~Value: Data,~Partial: Data,~operation: Value -> Element -> Value,~begin: Start -> Row,~begin_value: Start -> Value,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: Row -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Row,~close: Row -> Partial,~close_value: Value -> Partial,~merge: Partial -> Partial -> Partial,~finish: Partial -> Element,~chain_same: LineChainValues.ChainSame(~Storage.Buffer<Element>,~Buffer.Model<Element>,~Buffer.pack(~Element),~Start,~Row,~Value,~Partial,~begin,~begin_value,~fold,~LineChainValues.strided_model(~Buffer.Model<Element>,~Element,~Value,~Reduce.cell(~Element),~operation),~close,~close_value,~Element,~Pieces.vacant_buffer(~Element),~Reduce.vacant_model(~Element)),~Logical: Data,~original: Logical -> U32 -> Element,+limits: Machine.Limits,+operations: F32,+start: Start,+zero: Partial,+model: PiecesModel.Model<Element>,+shape: List<&2,U32>,+axes: List<&2,Nat>,+fill: Element,+workers: U32,+logical: Logical,+contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers}),+room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},valid: Bool,+name: String,+decided: {valid == True{} : Bool},+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32}) ->
  Global.Shaped(~Element,~Box<Partial>,~box_value(~Element,~Partial,~merge,~finish),reduced_box(~Element,~Value,~Partial,~operation,~close_value,~Logical,~original,logical,begin_value(start),zero,shape,axes),Partials.kept_extents(shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes)),Partials.reduce_listed(~Element,~Start,~Row,~Partial,~begin,~fold,~close,~merge,~finish,name,Tensors.Usable{},valid,operations,start,zero,PiecesModel.pack(~Element,model),shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),fill,workers)):
  match valid:
    case True{}:
      sized_cases(~Element,~Start,~Row,~Value,~Partial,~operation,~begin,~begin_value,~fold,~close,~close_value,~merge,~finish,~chain_same,~Logical,~original,limits,operations,start,zero,model,shape,axes,fill,workers,logical,contract,room,nonempty,cap,cap_ok,count_room,Shapes.has_zero(Partials.kept_extents(shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes))),Shapes.has_zero(shape),kept_nonzero(shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),CountCompletion.positive_not_zero(shape,nonempty)),
        CountCompletion.positive_not_zero(shape,nonempty),default_capacity)
    case False{}: Empty.absurd(Global.Shaped(~Element,~Box<Partial>,~box_value(~Element,~Partial,~merge,~finish),reduced_box(~Element,~Value,~Partial,~operation,~close_value,~Logical,~original,logical,begin_value(start),zero,shape,axes),Partials.kept_extents(shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes)),Partials.reduce_listed(~Element,~Start,~Row,~Partial,~begin,~fold,~close,~merge,~finish,name,Tensors.Usable{},False{},operations,start,zero,PiecesModel.pack(~Element,model),shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,axes),fill,workers)),WordEncoding.false_true(decided))

# A Ready reduction of distinct axes below the rank of a tensor of the
# contract of a reader holds the box of the reader.
def reduce_contract(~Element: Data,~Start: Data,~Row: Type,~Value: Data,~Partial: Data,~operation: Value -> Element -> Value,~begin: Start -> Row,~begin_value: Start -> Value,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: Row -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Row,~close: Row -> Partial,~close_value: Value -> Partial,~merge: Partial -> Partial -> Partial,~finish: Partial -> Element,~chain_same: LineChainValues.ChainSame(~Storage.Buffer<Element>,~Buffer.Model<Element>,~Buffer.pack(~Element),~Start,~Row,~Value,~Partial,~begin,~begin_value,~fold,~LineChainValues.strided_model(~Buffer.Model<Element>,~Element,~Value,~Reduce.cell(~Element),~operation),~close,~close_value,~Element,~Pieces.vacant_buffer(~Element),~Reduce.vacant_model(~Element)),~Logical: Data,~original: Logical -> U32 -> Element,+limits: Machine.Limits,+operations: F32,+start: Start,+zero: Partial,+model: PiecesModel.Model<Element>,+shape: List<&2,U32>,+axes: List<&2,U32>,+fill: Element,+workers: U32,+logical: Logical,+contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers}),+room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},+name: String,
  +unique: {Views.unique_axes(Layout.naturals(axes)) == True{} : Bool},+below: {Layout.all_below(axes,List.length(&2,U32,shape)) == True{} : Bool},+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32}) ->
  Global.Shaped(~Element,~Box<Partial>,~box_value(~Element,~Partial,~merge,~finish),reduced_box(~Element,~Value,~Partial,~operation,~close_value,~Logical,~original,logical,begin_value(start),zero,shape,Layout.naturals(axes)),
    Partials.kept_extents(shape,Layout.reduced_flags(List.length(&2,U32,shape),0n,Layout.naturals(axes))),
    Partials.reduce(~Element,~Start,~Row,~Partial,~begin,~fold,~close,~merge,~finish,name,operations,start,zero,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers},axes)):
  +decided = {%Equal.sym(Bool,Views.unique_axes(Layout.naturals(axes)),True{},unique) : {_ && Layout.all_below(axes,List.length(&2,U32,shape)) == True{} : Bool}
    %Equal.sym(Bool,Layout.all_below(axes,List.length(&2,U32,shape)),True{},below) : {True{} && _ == True{} : Bool}
    {==} : {Views.unique_axes(Layout.naturals(axes)) && Layout.all_below(axes,List.length(&2,U32,shape)) == True{} : Bool}}
  listed_valid(~Element,~Start,~Row,~Value,~Partial,~operation,~begin,~begin_value,~fold,~close,~close_value,~merge,~finish,~chain_same,~Logical,~original,limits,operations,start,zero,model,shape,Layout.naturals(axes),fill,workers,logical,contract,room,nonempty,cap,cap_ok,count_room,
    Views.unique_axes(Layout.naturals(axes)) && Layout.all_below(axes,List.length(&2,U32,shape)),name,decided,default_capacity)
def decrement_encoded(+position: Nat) -> {(U32.from_nat(1n+position) - 1 : U32) == U32.from_nat(position) : U32}:
  %Bounds.increment_encoded(position) : {(_ - 1 : U32) == U32.from_nat(position) : U32}
  Equal.trans(U32,((U32.from_nat(position) + 1 : U32) - 1 : U32),(U32.from_nat(position) + (1 - 1 : U32) : U32),U32.from_nat(position),
    Arithmetic.subtract_from_sum(U32.from_nat(position),1,1),Arithmetic.add_zero(U32.from_nat(position)))

def first_of(extents: List<&2,U32>) -> U32:
  match extents:
    case Nil{}: 1
    case Con{extent,rest}: extent

def first_same(axes: List<&2,Views.Axis>) -> {first_extent(axes) == first_of(Views.axis_extents(axes)) : U32}:
  match axes:
    case Nil{}: {==}
    case Con{Views.Axis{+extent,+stride},+rest}: {==}

# The address of digits of axes from base on, as the tree of Partials steps.
def digits_address(axes: List<&2,Views.Axis>,+base: U32,digits: List<&2,Nat>) -> U32:
  match axes digits:
    case Con{Views.Axis{extent,+stride},+rest} Con{+digit,+more}: digits_address(rest,(base + U32.from_nat(digit) * stride : U32),more)
    case axes digits: base

def address_append(done: List<&2,Views.Axis>,+axes: List<&2,Views.Axis>,+base: U32,prefix: List<&2,Nat>,+digits: List<&2,Nat>,+below: Coordinates.DigitsBelow(Views.axis_extents(done),prefix)) ->
  {digits_address(List.append(&2,Views.Axis,done,axes),base,List.append(&2,Nat,prefix,digits)) == digits_address(axes,digits_address(done,base,prefix),digits) : U32}:
  match done prefix:
    case Nil{} Nil{}: {==}
    case Con{Views.Axis{+extent,+stride},+rest} Con{+digit,+more}:
      KernelGemm.Both{+here,+later} = below
      address_append(rest,axes,(base + U32.from_nat(digit) * stride : U32),more,digits,later)
    case Nil{} Con{+digit,+more}:
      Empty.absurd({digits_address(List.append(&2,Views.Axis,[],axes),base,List.append(&2,Nat,digit <> more,digits)) == digits_address(axes,digits_address([],base,digit <> more),digits) : U32},below)
    case Con{Views.Axis{+extent,+stride},+rest} Nil{}:
      Empty.absurd({digits_address(List.append(&2,Views.Axis,Views.Axis{extent,stride} <> rest,axes),base,List.append(&2,Nat,[],digits)) ==
        digits_address(axes,digits_address(Views.Axis{extent,stride} <> rest,base,[]),digits) : U32},below)

def extents_append(left: List<&2,Views.Axis>,+right: List<&2,Views.Axis>) -> {Views.axis_extents(List.append(&2,Views.Axis,left,right)) == List.append(&2,U32,Views.axis_extents(left),Views.axis_extents(right)) : List<&2,U32>}:
  match left:
    case Nil{}: {==}
    case Con{Views.Axis{+extent,+stride},+rest}:
      Equal.cong(List<&2,U32>,List<&2,U32>,tail => extent <> tail,Views.axis_extents(List.append(&2,Views.Axis,rest,right)),List.append(&2,U32,Views.axis_extents(rest),Views.axis_extents(right)),extents_append(rest,right))

def below_append(extents: List<&2,U32>,digits: List<&2,Nat>,+more_extents: List<&2,U32>,+more: List<&2,Nat>,+below: Coordinates.DigitsBelow(extents,digits),
  +more_below: Coordinates.DigitsBelow(more_extents,more)) -> Coordinates.DigitsBelow(List.append(&2,U32,extents,more_extents),List.append(&2,Nat,digits,more)):
  match extents digits:
    case Nil{} Nil{}: more_below
    case Con{+extent,+rest} Con{+digit,+tail}:
      KernelGemm.Both{+here,+later} = below
      KernelGemm.Both{here,below_append(rest,tail,more_extents,more,later,more_below)}
    case Nil{} Con{+digit,+tail}: Empty.absurd(Coordinates.DigitsBelow(List.append(&2,U32,[],more_extents),List.append(&2,Nat,digit <> tail,more)),below)
    case Con{+extent,+rest} Nil{}: Empty.absurd(Coordinates.DigitsBelow(List.append(&2,U32,extent <> rest,more_extents),List.append(&2,Nat,[],more)),below)

def nil_cons(+extent: U32,+rest: List<&2,U32>,+same: {[] == extent <> rest : List<&2,U32>}) -> Empty:
  Digits.zero_successor(List.length(&2,U32,rest),Equal.cong(List<&2,U32>,Nat,items => List.length(&2,U32,items),[],extent <> rest,same))

def head_of(items: List<&2,U32>) -> U32:
  match items:
    case Nil{}: 0
    case Con{head,tail}: head

def tail_of(items: List<&2,U32>) -> List<&2,U32>:
  match items:
    case Nil{}: []
    case Con{head,tail}: tail

def below_added(+value: Nat,count: Nat,+positive: {Nat.is_lt(0n,count) == True{} : Bool}) -> {Nat.is_lt(value,Nat.add(value,count)) == True{} : Bool}:
  match count:
    case 0n: Empty.absurd({Nat.is_lt(value,Nat.add(value,0n)) == True{} : Bool},WordEncoding.false_true(positive))
    case 1n+ +more: Order.below_sum(value,more)

# Two trees of Partials over axes of the same extents, from addresses that
# step from roots as the trees do, agree when their readers agree at the
# addresses of every digits below the extents of the roots.
def trees_agree(~Partial: Data,~merge: Partial -> Partial -> Partial,~left_read: Unit -> U32 -> Partial,~right_read: Unit -> U32 -> Partial,~extents: List<&2,U32>,~left_root: List<&2,Views.Axis>,~right_root: List<&2,Views.Axis>,~left_base: U32,~right_base: U32,~agree: @+digits: List<&2,Nat> -> @+below: Coordinates.DigitsBelow(extents,digits) -> {left_read(Unit{},digits_address(left_root,left_base,digits)) == right_read(Unit{},digits_address(right_root,right_base,digits)) : Partial},
  fuel: Nat,single: Bool,left_axes: List<&2,Views.Axis>,right_axes: List<&2,Views.Axis>,+first: U32,+count: U32,+left_at: U32,+right_at: U32,+zero: Partial,
  +done_left: List<&2,Views.Axis>,+done_right: List<&2,Views.Axis>,+prefix: List<&2,Nat>,
  +left_whole: {left_root == List.append(&2,Views.Axis,done_left,left_axes) : List<&2,Views.Axis>},+right_whole: {right_root == List.append(&2,Views.Axis,done_right,right_axes) : List<&2,Views.Axis>},
  +left_extents: {Views.axis_extents(left_root) == extents : List<&2,U32>},+same: {Views.axis_extents(left_axes) == Views.axis_extents(right_axes) : List<&2,U32>},
  +left_below: Coordinates.DigitsBelow(Views.axis_extents(done_left),prefix),+right_below: Coordinates.DigitsBelow(Views.axis_extents(done_right),prefix),
  +left_address: {left_at == digits_address(done_left,left_base,prefix) : U32},+right_address: {right_at == digits_address(done_right,right_base,prefix) : U32},
  +cap: U32,+cap_ok: {U32.is_le(cap,1073741824) == True{} : Bool},+within: Within(Views.axis_extents(left_axes),cap),
  +range: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(first_extent(left_axes))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},+test: {U32.is_le(count,1) == single : Bool}) ->
  {tree_of(~Partial,~merge,~Unit,~left_read,Unit{},fuel,single,left_axes,first,count,left_at,zero) == tree_of(~Partial,~merge,~Unit,~right_read,Unit{},fuel,single,right_axes,first,count,right_at,zero) : Partial}:
  match fuel single left_axes right_axes:
    case 0n single left_axes right_axes: {==}
    case 1n+rest single Nil{} Nil{}:
      +left_done = Equal.trans(List<&2,Views.Axis>,left_root,List.append(&2,Views.Axis,done_left,[]),done_left,left_whole,ListReadback.append_empty(~Views.Axis,done_left))
      +right_done = Equal.trans(List<&2,Views.Axis>,right_root,List.append(&2,Views.Axis,done_right,[]),done_right,right_whole,ListReadback.append_empty(~Views.Axis,done_right))
      +below = {%Equal.trans(List<&2,U32>,Views.axis_extents(done_left),Views.axis_extents(left_root),extents,Equal.cong(List<&2,Views.Axis>,List<&2,U32>,axes => Views.axis_extents(axes),done_left,left_root,Equal.sym(List<&2,Views.Axis>,left_root,done_left,left_done)),left_extents) :
        Coordinates.DigitsBelow(_,prefix) left_below : Coordinates.DigitsBelow(extents,prefix)}
      +left_same = Equal.trans(U32,left_at,digits_address(done_left,left_base,prefix),digits_address(left_root,left_base,prefix),left_address,
        Equal.cong(List<&2,Views.Axis>,U32,axes => digits_address(axes,left_base,prefix),done_left,left_root,Equal.sym(List<&2,Views.Axis>,left_root,done_left,left_done)))
      +right_same = Equal.trans(U32,right_at,digits_address(done_right,right_base,prefix),digits_address(right_root,right_base,prefix),right_address,
        Equal.cong(List<&2,Views.Axis>,U32,axes => digits_address(axes,right_base,prefix),done_right,right_root,Equal.sym(List<&2,Views.Axis>,right_root,done_right,right_done)))
      Equal.trans(Partial,left_read(Unit{},left_at),left_read(Unit{},digits_address(left_root,left_base,prefix)),right_read(Unit{},right_at),
        Equal.cong(U32,Partial,word => left_read(Unit{},word),left_at,digits_address(left_root,left_base,prefix),left_same),
        Equal.trans(Partial,left_read(Unit{},digits_address(left_root,left_base,prefix)),right_read(Unit{},digits_address(right_root,right_base,prefix)),right_read(Unit{},right_at),
          agree(prefix,below),
          Equal.cong(U32,Partial,word => right_read(Unit{},word),digits_address(right_root,right_base,prefix),right_at,Equal.sym(U32,right_at,digits_address(right_root,right_base,prefix),right_same))))
    case 1n+rest single Nil{} Con{Views.Axis{+extent,+stride},+tail}:
      Empty.absurd({tree_of(~Partial,~merge,~Unit,~left_read,Unit{},1n+rest,single,[],first,count,left_at,zero) == tree_of(~Partial,~merge,~Unit,~right_read,Unit{},1n+rest,single,Views.Axis{extent,stride} <> tail,first,count,right_at,zero) : Partial},
        nil_cons(extent,Views.axis_extents(tail),same))
    case 1n+rest single Con{Views.Axis{+extent,+stride},+tail} Nil{}:
      Empty.absurd({tree_of(~Partial,~merge,~Unit,~left_read,Unit{},1n+rest,single,Views.Axis{extent,stride} <> tail,first,count,left_at,zero) == tree_of(~Partial,~merge,~Unit,~right_read,Unit{},1n+rest,single,[],first,count,right_at,zero) : Partial},
        nil_cons(extent,Views.axis_extents(tail),Equal.sym(List<&2,U32>,extent <> Views.axis_extents(tail),[],same)))
    case 1n+rest True{} Con{Views.Axis{+left_extent,+left_stride},+left_tail} Con{Views.Axis{+right_extent,+right_stride},+right_tail}:
      KernelGemm.Both{+head_positive,KernelGemm.Both{+head_bound,+tail_within}} = within
      +heads = Equal.cong(List<&2,U32>,U32,items => head_of(items),left_extent <> Views.axis_extents(left_tail),right_extent <> Views.axis_extents(right_tail),same)
      +tails = Equal.cong(List<&2,U32>,List<&2,U32>,items => tail_of(items),left_extent <> Views.axis_extents(left_tail),right_extent <> Views.axis_extents(right_tail),same)
      +next = first_extent(left_tail)
      +next_same = Equal.trans(U32,next,first_of(Views.axis_extents(left_tail)),first_extent(right_tail),first_same(left_tail),
        Equal.trans(U32,first_of(Views.axis_extents(left_tail)),first_of(Views.axis_extents(right_tail)),first_extent(right_tail),
          Equal.cong(List<&2,U32>,U32,items => first_of(items),Views.axis_extents(left_tail),Views.axis_extents(right_tail),tails),Equal.sym(U32,first_extent(right_tail),first_of(Views.axis_extents(right_tail)),first_same(right_tail))))
      +digit = U32.to_nat(first)
      +left_inside = Bounds.nat_trans(digit,Nat.add(digit,U32.to_nat(count)),U32.to_nat(left_extent),below_added(digit,U32.to_nat(count),positive),range)
      +right_inside = {%heads : {Nat.is_lt(digit,U32.to_nat(_)) == True{} : Bool} left_inside : {Nat.is_lt(digit,U32.to_nat(right_extent)) == True{} : Bool}}
      +left_axis = {Views.Axis{left_extent,left_stride} : Views.Axis}
      +right_axis = {Views.Axis{right_extent,right_stride} : Views.Axis}
      +left_whole_next = Equal.trans(List<&2,Views.Axis>,left_root,List.append(&2,Views.Axis,done_left,left_axis <> left_tail),List.append(&2,Views.Axis,List.append(&2,Views.Axis,done_left,[left_axis]),left_tail),left_whole,
        Equal.sym(List<&2,Views.Axis>,List.append(&2,Views.Axis,List.append(&2,Views.Axis,done_left,[left_axis]),left_tail),List.append(&2,Views.Axis,done_left,left_axis <> left_tail),ListReadback.append_associative(~Views.Axis,done_left,[left_axis],left_tail)))
      +right_whole_next = Equal.trans(List<&2,Views.Axis>,right_root,List.append(&2,Views.Axis,done_right,right_axis <> right_tail),List.append(&2,Views.Axis,List.append(&2,Views.Axis,done_right,[right_axis]),right_tail),right_whole,
        Equal.sym(List<&2,Views.Axis>,List.append(&2,Views.Axis,List.append(&2,Views.Axis,done_right,[right_axis]),right_tail),List.append(&2,Views.Axis,done_right,right_axis <> right_tail),ListReadback.append_associative(~Views.Axis,done_right,[right_axis],right_tail)))
      +left_below_next = {%Equal.sym(List<&2,U32>,Views.axis_extents(List.append(&2,Views.Axis,done_left,[left_axis])),List.append(&2,U32,Views.axis_extents(done_left),[left_extent]),extents_append(done_left,[left_axis])) :
          Coordinates.DigitsBelow(_,List.append(&2,Nat,prefix,[digit]))
        below_append(Views.axis_extents(done_left),prefix,[left_extent],[digit],left_below,KernelGemm.Both{left_inside,Unit{}}) :
        Coordinates.DigitsBelow(Views.axis_extents(List.append(&2,Views.Axis,done_left,[left_axis])),List.append(&2,Nat,prefix,[digit]))}
      +right_below_next = {%Equal.sym(List<&2,U32>,Views.axis_extents(List.append(&2,Views.Axis,done_right,[right_axis])),List.append(&2,U32,Views.axis_extents(done_right),[right_extent]),extents_append(done_right,[right_axis])) :
          Coordinates.DigitsBelow(_,List.append(&2,Nat,prefix,[digit]))
        below_append(Views.axis_extents(done_right),prefix,[right_extent],[digit],right_below,KernelGemm.Both{right_inside,Unit{}}) :
        Coordinates.DigitsBelow(Views.axis_extents(List.append(&2,Views.Axis,done_right,[right_axis])),List.append(&2,Nat,prefix,[digit]))}
      +left_address_next = Equal.trans(U32,(left_at + first * left_stride : U32),(digits_address(done_left,left_base,prefix) + first * left_stride : U32),
        digits_address(List.append(&2,Views.Axis,done_left,[left_axis]),left_base,List.append(&2,Nat,prefix,[digit])),
        Equal.cong(U32,U32,word => (word + first * left_stride : U32),left_at,digits_address(done_left,left_base,prefix),left_address),
        Equal.trans(U32,(digits_address(done_left,left_base,prefix) + first * left_stride : U32),(digits_address(done_left,left_base,prefix) + U32.from_nat(digit) * left_stride : U32),
          digits_address(List.append(&2,Views.Axis,done_left,[left_axis]),left_base,List.append(&2,Nat,prefix,[digit])),
          Equal.cong(U32,U32,word => (digits_address(done_left,left_base,prefix) + word * left_stride : U32),first,U32.from_nat(digit),Equal.sym(U32,U32.from_nat(digit),first,Arithmetic.native_roundtrip(first))),
          Equal.sym(U32,digits_address(List.append(&2,Views.Axis,done_left,[left_axis]),left_base,List.append(&2,Nat,prefix,[digit])),(digits_address(done_left,left_base,prefix) + U32.from_nat(digit) * left_stride : U32),
            address_append(done_left,[left_axis],left_base,prefix,[digit],left_below))))
      +right_address_next = Equal.trans(U32,(right_at + first * right_stride : U32),(digits_address(done_right,right_base,prefix) + first * right_stride : U32),
        digits_address(List.append(&2,Views.Axis,done_right,[right_axis]),right_base,List.append(&2,Nat,prefix,[digit])),
        Equal.cong(U32,U32,word => (word + first * right_stride : U32),right_at,digits_address(done_right,right_base,prefix),right_address),
        Equal.trans(U32,(digits_address(done_right,right_base,prefix) + first * right_stride : U32),(digits_address(done_right,right_base,prefix) + U32.from_nat(digit) * right_stride : U32),
          digits_address(List.append(&2,Views.Axis,done_right,[right_axis]),right_base,List.append(&2,Nat,prefix,[digit])),
          Equal.cong(U32,U32,word => (digits_address(done_right,right_base,prefix) + word * right_stride : U32),first,U32.from_nat(digit),Equal.sym(U32,U32.from_nat(digit),first,Arithmetic.native_roundtrip(first))),
          Equal.sym(U32,digits_address(List.append(&2,Views.Axis,done_right,[right_axis]),right_base,List.append(&2,Nat,prefix,[digit])),(digits_address(done_right,right_base,prefix) + U32.from_nat(digit) * right_stride : U32),
            address_append(done_right,[right_axis],right_base,prefix,[digit],right_below))))
      Equal.trans(Partial,tree_of(~Partial,~merge,~Unit,~left_read,Unit{},rest,U32.is_le(next,1),left_tail,0,next,(left_at + first * left_stride : U32),zero),
        tree_of(~Partial,~merge,~Unit,~right_read,Unit{},rest,U32.is_le(next,1),right_tail,0,next,(right_at + first * right_stride : U32),zero),
        tree_of(~Partial,~merge,~Unit,~right_read,Unit{},rest,U32.is_le(first_extent(right_tail),1),right_tail,0,first_extent(right_tail),(right_at + first * right_stride : U32),zero),
        trees_agree(~Partial,~merge,~left_read,~right_read,~extents,~left_root,~right_root,~left_base,~right_base,~agree,rest,U32.is_le(next,1),left_tail,right_tail,0,next,(left_at + first * left_stride : U32),(right_at + first * right_stride : U32),zero,
          List.append(&2,Views.Axis,done_left,[left_axis]),List.append(&2,Views.Axis,done_right,[right_axis]),List.append(&2,Nat,prefix,[digit]),left_whole_next,right_whole_next,left_extents,tails,left_below_next,right_below_next,
          left_address_next,right_address_next,cap,cap_ok,tail_within,Order.reflexive(U32.to_nat(next)),first_positive(left_tail,cap,tail_within),{==}),
        Equal.cong(U32,Partial,word => tree_of(~Partial,~merge,~Unit,~right_read,Unit{},rest,U32.is_le(word,1),right_tail,0,word,(right_at + first * right_stride : U32),zero),next,first_extent(right_tail),next_same))
    case 1n+ +rest False{} Con{Views.Axis{+left_extent,+left_stride},+left_tail} Con{Views.Axis{+right_extent,+right_stride},+right_tail}:
      KernelGemm.Both{+head_positive,KernelGemm.Both{+head_bound,+tail_within}} = within
      +left_axes_now = {Views.Axis{left_extent,left_stride} <> left_tail : List<&2,Views.Axis>}
      +right_axes_now = {Views.Axis{right_extent,right_stride} <> right_tail : List<&2,Views.Axis>}
      +left = Pieces.below(count)
      +right = (count - left : U32)
      +inside = Order.transitive(U32.to_nat(count),Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap),Order.right_le(U32.to_nat(first),U32.to_nat(count)),
        Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(left_extent),U32.to_nat(cap),range,head_bound))
      +halves = GridRanges.halves_of(cap,cap_ok,GridRanges.half_depth(count),count,test,positive,
        GridRanges.below_cover(count,GridRanges.below_facts(count,GridRanges.more_than_one(count,test),Encoded.native_le_transitive(count,cap,1073741824,Held.u32_le(count,cap,inside),cap_ok))),inside)
      +left_positive = GridRanges.halves_left_positive(count,GridRanges.half_depth(count),halves)
      +left_count = GridRanges.halves_left_count(count,GridRanges.half_depth(count),halves)
      +right_exact = GridRanges.halves_right_exact(count,GridRanges.half_depth(count),halves)
      +right_positive = GridRanges.halves_right_positive(count,GridRanges.half_depth(count),halves)
      +left_range = Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(left)),Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(left_extent),
        Equal.trans(Bool,Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(left)),Nat.add(U32.to_nat(first),U32.to_nat(count))),Nat.is_le(U32.to_nat(left),U32.to_nat(count)),True{},
          Order.le_offset(U32.to_nat(first),U32.to_nat(left),U32.to_nat(count)),left_count),range)
      +start = (first + left : U32)
      +start_exact = Arithmetic.add(first,left,cap,Encoded.native_le_transitive(cap,1073741824,2147483648,cap_ok,{==}),
        Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(left)),U32.to_nat(left_extent),U32.to_nat(cap),left_range,head_bound))
      +right_range_now = {%Equal.sym(Nat,U32.to_nat(start),Nat.add(U32.to_nat(first),U32.to_nat(left)),start_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(right)),U32.to_nat(left_extent)) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(right),Nat.sub(U32.to_nat(count),U32.to_nat(left)),right_exact) : {Nat.is_le(Nat.add(Nat.add(U32.to_nat(first),U32.to_nat(left)),_),U32.to_nat(left_extent)) == True{} : Bool}
        right_range(U32.to_nat(first),U32.to_nat(left),U32.to_nat(count),U32.to_nat(left_extent),left_count,range) :
        {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(right)),U32.to_nat(left_extent)) == True{} : Bool}}
      +low = trees_agree(~Partial,~merge,~left_read,~right_read,~extents,~left_root,~right_root,~left_base,~right_base,~agree,rest,U32.is_le(left,1),left_axes_now,right_axes_now,first,left,left_at,right_at,zero,done_left,done_right,prefix,left_whole,right_whole,left_extents,same,
        left_below,right_below,left_address,right_address,cap,cap_ok,within,left_range,left_positive,{==})
      +high = trees_agree(~Partial,~merge,~left_read,~right_read,~extents,~left_root,~right_root,~left_base,~right_base,~agree,rest,U32.is_le(right,1),left_axes_now,right_axes_now,start,right,left_at,right_at,zero,done_left,done_right,prefix,left_whole,right_whole,left_extents,same,
        left_below,right_below,left_address,right_address,cap,cap_ok,within,right_range_now,right_positive,{==})
      +left_low = tree_of(~Partial,~merge,~Unit,~left_read,Unit{},rest,U32.is_le(left,1),left_axes_now,first,left,left_at,zero)
      +right_low = tree_of(~Partial,~merge,~Unit,~right_read,Unit{},rest,U32.is_le(left,1),right_axes_now,first,left,right_at,zero)
      +left_high = tree_of(~Partial,~merge,~Unit,~left_read,Unit{},rest,U32.is_le(right,1),left_axes_now,start,right,left_at,zero)
      +right_high = tree_of(~Partial,~merge,~Unit,~right_read,Unit{},rest,U32.is_le(right,1),right_axes_now,start,right,right_at,zero)
      Equal.trans(Partial,merge(left_low,left_high),merge(right_low,left_high),merge(right_low,right_high),
        Equal.cong(Partial,Partial,part => merge(part,left_high),left_low,right_low,low),
        Equal.cong(Partial,Partial,part => merge(right_low,part),left_high,right_high,high))

def extents_count(axes: List<&2,Views.Axis>) -> {List.length(&2,U32,Views.axis_extents(axes)) == List.length(&2,Views.Axis,axes) : Nat}:
  match axes:
    case Nil{}: {==}
    case Con{Views.Axis{+extent,+stride},+rest}:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,U32,Views.axis_extents(rest)),List.length(&2,Views.Axis,rest),extents_count(rest))

def depth_same(+left: List<&2,Views.Axis>,+right: List<&2,Views.Axis>,+same: {Views.axis_extents(left) == Views.axis_extents(right) : List<&2,U32>}) ->
  {depth_of(left) == depth_of(right) : Nat}:
  +right_count = {%Equal.sym(List<&2,U32>,Views.axis_extents(left),Views.axis_extents(right),same) : {List.length(&2,U32,_) == List.length(&2,Views.Axis,right) : Nat} extents_count(right) :
    {List.length(&2,U32,Views.axis_extents(left)) == List.length(&2,Views.Axis,right) : Nat}}
  +counted = Equal.trans(Nat,List.length(&2,Views.Axis,left),List.length(&2,U32,Views.axis_extents(left)),List.length(&2,Views.Axis,right),
    Equal.sym(Nat,List.length(&2,U32,Views.axis_extents(left)),List.length(&2,Views.Axis,left),extents_count(left)),right_count)
  Equal.cong(Nat,Nat,count => 1n+Nat.mul(33n,count),List.length(&2,Views.Axis,left),List.length(&2,Views.Axis,right),counted)

# Two trees of the same extents from their roots agree when their readers
# agree at every digits.
def roots_agree(~Partial: Data,~merge: Partial -> Partial -> Partial,~left_read: Unit -> U32 -> Partial,~right_read: Unit -> U32 -> Partial,~left_root: List<&2,Views.Axis>,~right_root: List<&2,Views.Axis>,~left_base: U32,~right_base: U32,
  ~agree: @+digits: List<&2,Nat> -> @+below: Coordinates.DigitsBelow(Views.axis_extents(left_root),digits) -> {left_read(Unit{},digits_address(left_root,left_base,digits)) == right_read(Unit{},digits_address(right_root,right_base,digits)) : Partial},
  +zero: Partial,+cap: U32,+cap_ok: {U32.is_le(cap,1073741824) == True{} : Bool},+within: Within(Views.axis_extents(left_root),cap),
  +same: {Views.axis_extents(left_root) == Views.axis_extents(right_root) : List<&2,U32>}) ->
  {tree_of(~Partial,~merge,~Unit,~left_read,Unit{},depth_of(left_root),U32.is_le(first_extent(left_root),1),left_root,0,first_extent(left_root),left_base,zero) ==
    tree_of(~Partial,~merge,~Unit,~right_read,Unit{},depth_of(right_root),U32.is_le(first_extent(right_root),1),right_root,0,first_extent(right_root),right_base,zero) : Partial}:
  +left_first = first_extent(left_root)
  +first_eq = Equal.trans(U32,left_first,first_of(Views.axis_extents(left_root)),first_extent(right_root),first_same(left_root),
    Equal.trans(U32,first_of(Views.axis_extents(left_root)),first_of(Views.axis_extents(right_root)),first_extent(right_root),
      Equal.cong(List<&2,U32>,U32,extents => first_of(extents),Views.axis_extents(left_root),Views.axis_extents(right_root),same),
      Equal.sym(U32,first_extent(right_root),first_of(Views.axis_extents(right_root)),first_same(right_root))))
  %first_eq : {tree_of(~Partial,~merge,~Unit,~left_read,Unit{},depth_of(left_root),U32.is_le(left_first,1),left_root,0,left_first,left_base,zero) ==
    tree_of(~Partial,~merge,~Unit,~right_read,Unit{},depth_of(right_root),U32.is_le(_,1),right_root,0,first_extent(right_root),right_base,zero) : Partial}
  %first_eq : {tree_of(~Partial,~merge,~Unit,~left_read,Unit{},depth_of(left_root),U32.is_le(left_first,1),left_root,0,left_first,left_base,zero) ==
    tree_of(~Partial,~merge,~Unit,~right_read,Unit{},depth_of(right_root),U32.is_le(left_first,1),right_root,0,_,right_base,zero) : Partial}
  %depth_same(left_root,right_root,same) : {tree_of(~Partial,~merge,~Unit,~left_read,Unit{},depth_of(left_root),U32.is_le(left_first,1),left_root,0,left_first,left_base,zero) ==
    tree_of(~Partial,~merge,~Unit,~right_read,Unit{},_,U32.is_le(left_first,1),right_root,0,left_first,right_base,zero) : Partial}
  trees_agree(~Partial,~merge,~left_read,~right_read,~Views.axis_extents(left_root),~left_root,~right_root,~left_base,~right_base,~agree,
    depth_of(left_root),U32.is_le(left_first,1),left_root,right_root,0,left_first,left_base,right_base,zero,[],[],[],{==},{==},{==},same,Unit{},Unit{},{==},{==},
    cap,cap_ok,within,Order.reflexive(U32.to_nat(left_first)),first_positive(left_root,cap,within),{==})

# A map that takes merges to merges takes the tree of a reader to the tree
# of the mapped reader.
def tree_mapped(~Partial: Data,~merge: Partial -> Partial -> Partial,~Mapped: Data,~joined: Mapped -> Mapped -> Mapped,~map: Partial -> Mapped,
  ~homomorphic: @+left: Partial -> @+right: Partial -> {map(merge(left,right)) == joined(map(left),map(right)) : Mapped},
  ~Context: Data,~read: Context -> U32 -> Partial,fuel: Nat,single: Bool,axes: List<&2,Views.Axis>,+context: Context,+first: U32,+count: U32,+base: U32,+zero: Partial) ->
  {map(tree_of(~Partial,~merge,~Context,~read,context,fuel,single,axes,first,count,base,zero)) ==
    tree_of(~Mapped,~joined,~Unit,~(unit => index => map(read(context,index))),Unit{},fuel,single,axes,first,count,base,map(zero)) : Mapped}:
  match fuel single axes:
    case 0n single axes: {==}
    case 1n+rest single Nil{}: {==}
    case 1n+rest True{} Con{Views.Axis{extent,+stride},+tail}:
      tree_mapped(~Partial,~merge,~Mapped,~joined,~map,~homomorphic,~Context,~read,rest,U32.is_le(first_extent(tail),1),tail,context,0,first_extent(tail),(base + first * stride : U32),zero)
    case 1n+ +rest False{} Con{Views.Axis{+extent,+stride},+tail}:
      +left = Pieces.below(count)
      +right = (count - left : U32)
      +whole = {Con{Views.Axis{extent,stride},tail} : List<&2,Views.Axis>}
      +low = tree_of(~Partial,~merge,~Context,~read,context,rest,U32.is_le(left,1),whole,first,left,base,zero)
      +high = tree_of(~Partial,~merge,~Context,~read,context,rest,U32.is_le(right,1),whole,(first + left : U32),right,base,zero)
      +high_mapped = tree_of(~Mapped,~joined,~Unit,~(unit => index => map(read(context,index))),Unit{},rest,U32.is_le(right,1),whole,(first + left : U32),right,base,map(zero))
      %tree_mapped(~Partial,~merge,~Mapped,~joined,~map,~homomorphic,~Context,~read,rest,U32.is_le(left,1),whole,context,first,left,base,zero) :
        {map(merge(low,high)) == joined(_,high_mapped) : Mapped}
      %tree_mapped(~Partial,~merge,~Mapped,~joined,~map,~homomorphic,~Context,~read,rest,U32.is_le(right,1),whole,context,(first + left : U32),right,base,zero) :
        {map(merge(low,high)) == joined(map(low),_) : Mapped}
      homomorphic(low,high)

# The extents of the rows of a box lie within the capacity.
def within_rows(reduced: Bool,+extents: List<&2,U32>,+axis: Nat,+cap: U32,+within: Within(extents,cap)) -> Within(Partials.row_extents(reduced,extents,axis,1n),cap):
  match reduced:
    case True{}: within_removed(extents,axis,cap,within)
    case False{}: within

# The last extent is the count of the values from the last axis on.
def last_elements(+shape: List<&2,U32>,+positive: {Nat.is_lt(0n,List.length(&2,U32,shape)) == True{} : Bool}) ->
  {U32.to_nat(Grid.lanes_at(shape,Nat.sub(List.length(&2,U32,shape),1n))) == ShapeLaws.elements(List.drop(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),1n))) : Nat}:
  %Equal.sym(List<&2,U32>,List.drop(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),1n)),[Grid.lanes_at(shape,Nat.sub(List.length(&2,U32,shape),1n))],dropped_last(shape,Nat.sub(List.length(&2,U32,shape),1n),successor_after(List.length(&2,U32,shape),positive))) :
    {U32.to_nat(Grid.lanes_at(shape,Nat.sub(List.length(&2,U32,shape),1n))) == ShapeLaws.elements(_) : Nat}
  Equal.sym(Nat,Nat.mul(U32.to_nat(Grid.lanes_at(shape,Nat.sub(List.length(&2,U32,shape),1n))),1n),U32.to_nat(Grid.lanes_at(shape,Nat.sub(List.length(&2,U32,shape),1n))),Order.times_one(U32.to_nat(Grid.lanes_at(shape,Nat.sub(List.length(&2,U32,shape),1n)))))

# The extents from an axis on are its extent and the extents after it.
def drop_lanes(shape: List<&2,U32>,axis: Nat,+inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool}) ->
  {List.drop(&2,U32,shape,axis) == Grid.lanes_at(shape,axis) <> List.drop(&2,U32,shape,Nat.add(axis,1n)) : List<&2,U32>}:
  match shape axis:
    case [] +index:
      Empty.absurd({List.drop(&2,U32,[],index) == Grid.lanes_at([],index) <> List.drop(&2,U32,[],Nat.add(index,1n)) : List<&2,U32>},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(index,0n),True{},Equal.sym(Bool,Nat.is_lt(index,0n),False{},Order.not_below(0n,index,Order.zero_le(index))),inside)))
    case +extent <> +tail 0n:
      Equal.cong(List<&2,U32>,List<&2,U32>,rest => extent <> rest,tail,List.drop(&2,U32,tail,0n),Equal.sym(List<&2,U32>,List.drop(&2,U32,tail,0n),tail,LineChainValues.drop_zero(~U32,tail)))
    case +extent <> +tail 1n+ +rest: drop_lanes(tail,rest,inside)

# The values from an axis on are its extent times the values after it.
def drop_elements(+shape: List<&2,U32>,+axis: Nat,+inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool}) -> {ShapeLaws.elements(List.drop(&2,U32,shape,axis)) == Nat.mul(U32.to_nat(Grid.lanes_at(shape,axis)),ShapeLaws.elements(List.drop(&2,U32,shape,Nat.add(axis,1n)))) : Nat}:
  Equal.cong(List<&2,U32>,Nat,extents => ShapeLaws.elements(extents),List.drop(&2,U32,shape,axis),Grid.lanes_at(shape,axis) <> List.drop(&2,U32,shape,Nat.add(axis,1n)),drop_lanes(shape,axis,inside))

# The values after the leading axes of a shape with values are no more than
# all.
def tail_room(+shape: List<&2,U32>,+kept: Nat,+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool}) -> {Nat.is_le(ShapeLaws.elements(List.drop(&2,U32,shape,kept)),ShapeLaws.elements(shape)) == True{} : Bool}:
  +split = CountCompletion.elements_split(shape,kept)
  +take_positive = RowMajor.left_factor_positive(ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept)),Held.nat_lt_moved(0n,Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept))),ShapeLaws.elements(shape),Equal.sym(Nat,ShapeLaws.elements(shape),Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept))),split),nonempty))
  Held.nat_le_right_moved(ShapeLaws.elements(List.drop(&2,U32,shape,kept)),ShapeLaws.elements(shape),Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept))),split,
    Held.nat_le_right_moved(ShapeLaws.elements(List.drop(&2,U32,shape,kept)),Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept))),Nat.mul(ShapeLaws.elements(List.drop(&2,U32,shape,kept)),ShapeLaws.elements(List.take(&2,U32,shape,kept))),RowMajor.multiply_commutative(ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept))),RowMajor.product_covers(ShapeLaws.elements(List.drop(&2,U32,shape,kept)),ShapeLaws.elements(List.take(&2,U32,shape,kept)),take_positive)))

# The word of the values after the leading axes of a shape that fits.
def tail_word(+limits: Machine.Limits,+shape: List<&2,U32>,+kept: Nat,+room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool}) -> {U32.to_nat(Grid.product(List.drop(&2,U32,shape,kept))) == ShapeLaws.elements(List.drop(&2,U32,shape,kept)) : Nat}:
  Equal.trans(Nat,U32.to_nat(Grid.product(List.drop(&2,U32,shape,kept))),U32.to_nat(U32.from_nat(ShapeLaws.elements(List.drop(&2,U32,shape,kept)))),ShapeLaws.elements(List.drop(&2,U32,shape,kept)),
    Equal.cong(U32,Nat,word => U32.to_nat(word),Grid.product(List.drop(&2,U32,shape,kept)),U32.from_nat(ShapeLaws.elements(List.drop(&2,U32,shape,kept))),
      Equal.sym(U32,U32.from_nat(ShapeLaws.elements(List.drop(&2,U32,shape,kept))),Grid.product(List.drop(&2,U32,shape,kept)),Digits.product_from(List.drop(&2,U32,shape,kept)))),
    FullRange.roundtrip(ShapeLaws.elements(List.drop(&2,U32,shape,kept)),Machine.capacity(limits),Order.transitive(ShapeLaws.elements(List.drop(&2,U32,shape,kept)),ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits)),tail_room(shape,kept,nonempty),room)))

# The cells of a block of lines of an axis, the cells after it interleaved.
def block_length(+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},+room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool}) -> {U32.to_nat((Grid.lanes_at(shape,axis) * Grid.product(List.drop(&2,U32,shape,Nat.add(axis,1n))) : U32)) == ShapeLaws.elements(List.drop(&2,U32,shape,axis)) : Nat}:
  %drop_lanes(shape,axis,inside) : {U32.to_nat(Grid.product(_)) == ShapeLaws.elements(List.drop(&2,U32,shape,axis)) : Nat}
  tail_word(limits,shape,axis,room,nonempty)

def block_inner_positive(+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool}) -> {Nat.is_lt(0n,U32.to_nat(Grid.product(List.drop(&2,U32,shape,Nat.add(axis,1n))))) == True{} : Bool}:
  Held.nat_lt_moved(0n,U32.to_nat(Grid.product(List.drop(&2,U32,shape,Nat.add(axis,1n)))),ShapeLaws.elements(List.drop(&2,U32,shape,Nat.add(axis,1n))),tail_word(limits,shape,Nat.add(axis,1n),room,nonempty),tail_positive(shape,Nat.add(axis,1n),nonempty))

def block_inner_le(+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},+room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool}) -> {Nat.is_le(U32.to_nat(Grid.product(List.drop(&2,U32,shape,Nat.add(axis,1n)))),U32.to_nat((Grid.lanes_at(shape,axis) * Grid.product(List.drop(&2,U32,shape,Nat.add(axis,1n))) : U32))) == True{} : Bool}:
  +split = drop_elements(shape,axis,inside)
  +lanes_positive = RowMajor.left_factor_positive(U32.to_nat(Grid.lanes_at(shape,axis)),ShapeLaws.elements(List.drop(&2,U32,shape,Nat.add(axis,1n))),Held.nat_lt_moved(0n,Nat.mul(U32.to_nat(Grid.lanes_at(shape,axis)),ShapeLaws.elements(List.drop(&2,U32,shape,Nat.add(axis,1n)))),ShapeLaws.elements(List.drop(&2,U32,shape,axis)),Equal.sym(Nat,ShapeLaws.elements(List.drop(&2,U32,shape,axis)),Nat.mul(U32.to_nat(Grid.lanes_at(shape,axis)),ShapeLaws.elements(List.drop(&2,U32,shape,Nat.add(axis,1n)))),split),tail_positive(shape,axis,nonempty)))
  +covered = Held.nat_le_right_moved(ShapeLaws.elements(List.drop(&2,U32,shape,Nat.add(axis,1n))),ShapeLaws.elements(List.drop(&2,U32,shape,axis)),Nat.mul(U32.to_nat(Grid.lanes_at(shape,axis)),ShapeLaws.elements(List.drop(&2,U32,shape,Nat.add(axis,1n)))),split,
    Held.nat_le_right_moved(ShapeLaws.elements(List.drop(&2,U32,shape,Nat.add(axis,1n))),Nat.mul(U32.to_nat(Grid.lanes_at(shape,axis)),ShapeLaws.elements(List.drop(&2,U32,shape,Nat.add(axis,1n)))),Nat.mul(ShapeLaws.elements(List.drop(&2,U32,shape,Nat.add(axis,1n))),U32.to_nat(Grid.lanes_at(shape,axis))),RowMajor.multiply_commutative(U32.to_nat(Grid.lanes_at(shape,axis)),ShapeLaws.elements(List.drop(&2,U32,shape,Nat.add(axis,1n)))),RowMajor.product_covers(ShapeLaws.elements(List.drop(&2,U32,shape,Nat.add(axis,1n))),U32.to_nat(Grid.lanes_at(shape,axis)),lanes_positive)))
  Held.nat_le_moved(U32.to_nat(Grid.product(List.drop(&2,U32,shape,Nat.add(axis,1n)))),ShapeLaws.elements(List.drop(&2,U32,shape,Nat.add(axis,1n))),U32.to_nat((Grid.lanes_at(shape,axis) * Grid.product(List.drop(&2,U32,shape,Nat.add(axis,1n))) : U32)),tail_word(limits,shape,Nat.add(axis,1n),room,nonempty),
    Held.nat_le_right_moved(ShapeLaws.elements(List.drop(&2,U32,shape,Nat.add(axis,1n))),U32.to_nat((Grid.lanes_at(shape,axis) * Grid.product(List.drop(&2,U32,shape,Nat.add(axis,1n))) : U32)),ShapeLaws.elements(List.drop(&2,U32,shape,axis)),block_length(limits,shape,axis,inside,room,nonempty),covered))
