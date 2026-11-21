# Each cell of a block is the ordered dot of its taps: step k reads the
# lane tile and the row tile the reader's cursor points to, and multiplies lane
# by row.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ../ordered_matrix.bend as OrderedMatrix
import ./storage_tree_model.bend as Tree
import ./kernel_tile_refinement.bend as TileLaws
import ./traversal_reduction_refinement.bend as ReductionLaws
import ./kernel_gemm_refinement.bend as GemmLaws
import ./word_encoding.bend as WordEncoding
import ./nat_division.bend as Division
import ./natural_addition.bend as Addition
import ./ordered_ranges.bend as Ranges
import ./ordered_reduction_split.bend as Ordered
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_arithmetic.bend as WordArithmetic
import ./word_multiplication.bend as Multiplication
import ./u32_full_range.bend as FullRange
import ./u32_comparison.bend as Comparison
import ./nat_division_order.bend as DivisionOrder
import ./nat_order.bend as Order
import ./row_major_index.bend as RowMajor

type Source is Data:
  Source{row_tiles: Tree.Tree<Tile.Tile8>,lane_tiles: Tree.Tree<Tile.Tile8>}

def lane_element(source: Source,lane: Nat,tap: ReductionLaws.AddressPair) -> F32:
  Source{row_tiles,lane_tiles} = source
  ReductionLaws.AddressPair{lane_address,row_address} = tap
  TileLaws.select_lane(lane,Tree.read(Tile.Tile8,lane_tiles,lane_address))

def row_element(source: Source,row: Nat,tap: ReductionLaws.AddressPair) -> F32:
  Source{row_tiles,lane_tiles} = source
  ReductionLaws.AddressPair{lane_address,row_address} = tap
  TileLaws.select_lane(row,Tree.read(Tile.Tile8,row_tiles,row_address))

# The ordered dot of a cell. Lane first multiplies the lane value by the row
# value, row first the row value by the lane value.
def dot(+order: GemmLaws.Order,taps: List<&2,ReductionLaws.AddressPair>,+row: Nat,+lane: Nat,+source: Source,initial: F32) -> F32:
  match order:
    case GemmLaws.LaneFirst{}: OrderedMatrix.dot(~Nat,~ReductionLaws.AddressPair,~Nat,~Source,~lane_element,~row_element,taps,row,lane,source,initial)
    case GemmLaws.RowFirst{}: OrderedMatrix.dot(~Nat,~ReductionLaws.AddressPair,~Nat,~Source,~row_element,~lane_element,taps,lane,row,source,initial)

# The taps of count steps, whose first read is the argument.
def taps_from(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,count: Nat,read: Gemm.Step<Cursor>) -> List<&2,ReductionLaws.AddressPair>:
  match count:
    case 0n: []
    case 1n+rest:
      Gemm.Step{lane_address,row_address,cursor} = read
      ReductionLaws.AddressPair{lane_address,row_address} <> taps_from(~Cursor,~advance,rest,advance(cursor))

def taps(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,count: Nat,cursor: Cursor) -> List<&2,ReductionLaws.AddressPair>:
  taps_from(~Cursor,~advance,count,advance(cursor))

# The direct reader walks both operands one address per step.
def direct_taps(count: Nat,+lane_address: U32,+row_address: U32) ->
  {taps(~Gemm.Direct(),~Gemm.directs,count,Gemm.Both{lane_address,row_address}) == ReductionLaws.addresses(count,lane_address,row_address) : List<&2,ReductionLaws.AddressPair>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +first = {ReductionLaws.AddressPair{lane_address,row_address} : ReductionLaws.AddressPair}
      Equal.cong(List<&2,ReductionLaws.AddressPair>,List<&2,ReductionLaws.AddressPair>,later => first <> later,
        taps(~Gemm.Direct(),~Gemm.directs,rest,Gemm.Both{(lane_address + 1 : U32),(row_address + 1 : U32)}),
        ReductionLaws.addresses(rest,(lane_address + 1 : U32),(row_address + 1 : U32)),direct_taps(rest,(lane_address + 1 : U32),(row_address + 1 : U32)))

# ---- Readers ----

# The addresses of count reads of a reader from a cursor.
def reads(~Reader: Data,~read: Reader -> U32 & Reader,count: Nat,cursor: Reader) -> List<&2,U32>:
  match count:
    case 0n: []
    case 1n+rest:
      +cursor = cursor
      Pair.fst(U32,Reader,read(cursor)) <> reads(~Reader,~read,rest,Pair.snd(U32,Reader,read(cursor)))

# Two address lists step by step.
def paired(inputs: List<&2,U32>,row_tiles: List<&2,U32>) -> List<&2,ReductionLaws.AddressPair>:
  match inputs row_tiles:
    case Nil{} Nil{}: []
    case Nil{} row_address <> later: []
    case lane_address <> rest Nil{}: []
    case lane_address <> rest row_address <> later: ReductionLaws.AddressPair{lane_address,row_address} <> paired(rest,later)

# Two readers that step together pair their addresses.
def both_taps(~Input: Data,~Weight: Data,~lane_address: Input -> U32 & Input,~row_address: Weight -> U32 & Weight,
  ~advance: Gemm.Both<Input,Weight> -> Gemm.Step<Gemm.Both<Input,Weight>>,
  ~advanced: @+first: Input -> @+second: Weight ->
    {advance(Gemm.Both{first,second}) == Gemm.Step{Pair.fst(U32,Input,lane_address(first)),Pair.fst(U32,Weight,row_address(second)),
      Gemm.Both{Pair.snd(U32,Input,lane_address(first)),Pair.snd(U32,Weight,row_address(second))}} : Gemm.Step<Gemm.Both<Input,Weight>>},
  count: Nat,+first: Input,+second: Weight) ->
  {taps(~Gemm.Both<Input,Weight>,~advance,count,Gemm.Both{first,second}) ==
    paired(reads(~Input,~lane_address,count,first),reads(~Weight,~row_address,count,second)) : List<&2,ReductionLaws.AddressPair>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +a = Pair.fst(U32,Input,lane_address(first))
      +b = Pair.fst(U32,Weight,row_address(second))
      +i = Pair.snd(U32,Input,lane_address(first))
      +w = Pair.snd(U32,Weight,row_address(second))
      %Equal.sym(Gemm.Step<Gemm.Both<Input,Weight>>,advance(Gemm.Both{first,second}),Gemm.Step{a,b,Gemm.Both{i,w}},advanced(first,second)) :
        {taps_from(~Gemm.Both<Input,Weight>,~advance,1n+rest,_) ==
          paired(reads(~Input,~lane_address,1n+rest,first),reads(~Weight,~row_address,1n+rest,second)) : List<&2,ReductionLaws.AddressPair>}
      Equal.cong(List<&2,ReductionLaws.AddressPair>,List<&2,ReductionLaws.AddressPair>,tail => ReductionLaws.AddressPair{a,b} <> tail,
        taps(~Gemm.Both<Input,Weight>,~advance,rest,Gemm.Both{i,w}),paired(reads(~Input,~lane_address,rest,i),reads(~Weight,~row_address,rest,w)),
        both_taps(~Input,~Weight,~lane_address,~row_address,~advance,~advanced,rest,i,w))

# Two key maps paired key by key.
def both_key(~Key: Data,~Left: Data,~Right: Data,~left: Left -> Key -> U32,~right: Right -> Key -> U32,context: Gemm.Both<Left,Right>,key: Key) -> ReductionLaws.AddressPair:
  Gemm.Both{first,second} = context
  +key = key
  ReductionLaws.AddressPair{left(first,key),right(second,key)}

def paired_keys(~Key: Data,~Left: Data,~Right: Data,~left: Left -> Key -> U32,~right: Right -> Key -> U32,keys: List<&2,Key>,+first: Left,+second: Right) ->
  {paired(Ordered.map_keys(~Key,~U32,~Left,~left,keys,first),Ordered.map_keys(~Key,~U32,~Right,~right,keys,second)) ==
    Ordered.map_keys(~Key,~ReductionLaws.AddressPair,~Gemm.Both<Left,Right>,~both_key(~Key,~Left,~Right,~left,~right),keys,Gemm.Both{first,second}) :
    List<&2,ReductionLaws.AddressPair>}:
  match keys:
    case Nil{}: {==}
    case +key <> +rest:
      Equal.cong(List<&2,ReductionLaws.AddressPair>,List<&2,ReductionLaws.AddressPair>,tail => ReductionLaws.AddressPair{left(first,key),right(second,key)} <> tail,
        paired(Ordered.map_keys(~Key,~U32,~Left,~left,rest,first),Ordered.map_keys(~Key,~U32,~Right,~right,rest,second)),
        Ordered.map_keys(~Key,~ReductionLaws.AddressPair,~Gemm.Both<Left,Right>,~both_key(~Key,~Left,~Right,~left,~right),rest,Gemm.Both{first,second}),
        paired_keys(~Key,~Left,~Right,~left,~right,rest,first,second))

# The direct reader from base + step reads base + step, base + step + 1, ...
def offset_word(base: Nat,step: Nat) -> U32:
  U32.from_nat(Nat.add(base,step))

def direct_reads(count: Nat,+base: Nat,+step: Nat) ->
  {reads(~U32,~Gemm.direct,count,U32.from_nat(Nat.add(base,step))) == Ordered.map_keys(~Nat,~U32,~Nat,~offset_word,Ranges.range(count,step),base) : List<&2,U32>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +here = U32.from_nat(Nat.add(base,step))
      %Equal.sym(U32,(here + 1 : U32),U32.from_nat(1n+Nat.add(base,step)),Bounds.increment_encoded(Nat.add(base,step))) :
        {here <> reads(~U32,~Gemm.direct,rest,_) == here <> Ordered.map_keys(~Nat,~U32,~Nat,~offset_word,Ranges.range(rest,1n+step),base) : List<&2,U32>}
      %Addition.successor_right(base,step) :
        {here <> reads(~U32,~Gemm.direct,rest,U32.from_nat(_)) == here <> Ordered.map_keys(~Nat,~U32,~Nat,~offset_word,Ranges.range(rest,1n+step),base) : List<&2,U32>}
      Equal.cong(List<&2,U32>,List<&2,U32>,tail => here <> tail,
        reads(~U32,~Gemm.direct,rest,U32.from_nat(Nat.add(base,1n+step))),Ordered.map_keys(~Nat,~U32,~Nat,~offset_word,Ranges.range(rest,1n+step),base),
        direct_reads(rest,base,1n+step))

# A grid reader walks planes of rows of columns entries from position first:
# rows are band addresses apart, and jump more after the last entry of a
# plane reaches the first entry of the next. Position p is row p / columns of
# the planes and column p % columns; row r is row r % rows of plane r / rows.
type Grid is Data:
  Grid{base: U32,band: Nat,columns: U32,rows: U32,jump: U32,first: Nat}

# The distance from a plane to the next.
def grid_span(+band: Nat,+columns: U32,+rows: U32,+jump: U32) -> Nat:
  Nat.add(Nat.add(Nat.mul(Nat.sub(U32.to_nat(rows),1n),band),Nat.sub(U32.to_nat(columns),1n)),U32.to_nat(jump))

def grid_address(+base: U32,+band: Nat,+span: Nat,+plane: Nat,+row: Nat,+column: Nat) -> U32:
  (base + U32.from_nat(Nat.add(Nat.mul(plane,span),Nat.add(Nat.mul(row,band),column))) : U32)

def grid_width(+grid: Grid) -> Nat:
  Grid{base,band,+columns,rows,jump,first} = grid
  U32.to_nat(columns)

def grid_rows(+grid: Grid) -> Nat:
  Grid{base,band,columns,+rows,jump,first} = grid
  U32.to_nat(rows)

# Entry (plane, row, column) of a grid.
def grid_at(+grid: Grid,+plane: Nat,+row: Nat,+column: Nat) -> Gemm.Entries:
  Grid{+base,+band,+columns,+rows,+jump,first} = grid
  Gemm.Entries{grid_address(base,band,grid_span(band,columns,rows,jump),plane,row,column),U32.from_nat(column),U32.from_nat(row),columns,rows,
    (U32.from_nat(band) - columns + 1 : U32),jump}

def grid_at_word(+grid: Grid,+plane: Nat,+row: Nat,+column: Nat) -> U32:
  Grid{+base,+band,+columns,+rows,+jump,first} = grid
  grid_address(base,band,grid_span(band,columns,rows,jump),plane,row,column)

def grid_placed(+grid: Grid,place: Nat & Nat,+column: Nat) -> Gemm.Entries:
  (+plane,+row) = place
  grid_at(grid,plane,row,column)

def grid_placed_word(+grid: Grid,place: Nat & Nat,+column: Nat) -> U32:
  (+plane,+row) = place
  grid_at_word(grid,plane,row,column)

# Entry (row, column) of the rows of all planes.
def grid_cell(+grid: Grid,cell: Nat & Nat) -> Gemm.Entries:
  (+row,+column) = cell
  grid_placed(grid,Nat.divmod(row,grid_rows(grid)),column)

def grid_word(+grid: Grid,cell: Nat & Nat) -> U32:
  (+row,+column) = cell
  grid_placed_word(grid,Nat.divmod(row,grid_rows(grid)),column)

def grid_position(+grid: Grid,+step: Nat) -> Nat:
  Grid{base,band,columns,rows,jump,+first} = grid
  Nat.add(first,step)

def grid_entry(grid: Grid,step: Nat) -> Gemm.Entries:
  +grid = grid
  +step = step
  grid_cell(grid,Nat.divmod(grid_position(grid,step),grid_width(grid)))

def grid_read(grid: Grid,step: Nat) -> U32:
  +grid = grid
  +step = step
  grid_word(grid,Nat.divmod(grid_position(grid,step),grid_width(grid)))

# In words, an entry is its plane, row and column from the base; the plane
# step is the last row and column of a plane and the jump.
def plane_word(+band: U32,+columns: U32,+rows: U32,+jump: U32) -> U32:
  ((rows - 1) * band + (columns - 1) + jump : U32)

def encoded_sum(+left: Nat,+right: Nat) -> {U32.from_nat(Nat.add(left,right)) == (U32.from_nat(left) + U32.from_nat(right) : U32) : U32}:
  Equal.sym(U32,(U32.from_nat(left) + U32.from_nat(right) : U32),U32.from_nat(Nat.add(left,right)),WordArithmetic.native_add(left,right))

def encoded_product(+left: Nat,+right: Nat) -> {U32.from_nat(Nat.mul(left,right)) == (U32.from_nat(left) * U32.from_nat(right) : U32) : U32}:
  Equal.sym(U32,(U32.from_nat(left) * U32.from_nat(right) : U32),U32.from_nat(Nat.mul(left,right)),Multiplication.native_encoded_multiply(left,right))

def encoded_less(+value: U32,+positive: {Nat.is_lt(0n,U32.to_nat(value)) == True{} : Bool}) ->
  {U32.from_nat(Nat.sub(U32.to_nat(value),1n)) == (value - 1 : U32) : U32}:
  Equal.sym(U32,(value - 1 : U32),U32.from_nat(Nat.sub(U32.to_nat(value),1n)),Arithmetic.native_subtract(value,1,Order.strict_follow(0n,U32.to_nat(value),positive)))

def row_offset(+band: U32,+row: Nat,+column: Nat) ->
  {U32.from_nat(Nat.add(Nat.mul(row,U32.to_nat(band)),column)) == (U32.from_nat(row) * band + U32.from_nat(column) : U32) : U32}:
  +bn = U32.to_nat(band)
  Equal.trans(U32,U32.from_nat(Nat.add(Nat.mul(row,bn),column)),(U32.from_nat(Nat.mul(row,bn)) + U32.from_nat(column) : U32),(U32.from_nat(row) * band + U32.from_nat(column) : U32),
    encoded_sum(Nat.mul(row,bn),column),
    Equal.cong(U32,U32,value => (value + U32.from_nat(column) : U32),U32.from_nat(Nat.mul(row,bn)),(U32.from_nat(row) * band : U32),
      Equal.trans(U32,U32.from_nat(Nat.mul(row,bn)),(U32.from_nat(row) * U32.from_nat(bn) : U32),(U32.from_nat(row) * band : U32),encoded_product(row,bn),
        Equal.cong(U32,U32,value => (U32.from_nat(row) * value : U32),U32.from_nat(bn),band,Arithmetic.native_roundtrip(band)))))

def span_word(+band: U32,+columns: U32,+rows: U32,+jump: U32,
  +columns_positive: {Nat.is_lt(0n,U32.to_nat(columns)) == True{} : Bool},+rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool}) ->
  {U32.from_nat(grid_span(U32.to_nat(band),columns,rows,jump)) == plane_word(band,columns,rows,jump) : U32}:
  +bn = U32.to_nat(band)
  +r1 = Nat.sub(U32.to_nat(rows),1n)
  +c1 = Nat.sub(U32.to_nat(columns),1n)
  Equal.trans(U32,U32.from_nat(Nat.add(Nat.add(Nat.mul(r1,bn),c1),U32.to_nat(jump))),(U32.from_nat(Nat.add(Nat.mul(r1,bn),c1)) + U32.from_nat(U32.to_nat(jump)) : U32),
    plane_word(band,columns,rows,jump),encoded_sum(Nat.add(Nat.mul(r1,bn),c1),U32.to_nat(jump)),
    Equal.trans(U32,(U32.from_nat(Nat.add(Nat.mul(r1,bn),c1)) + U32.from_nat(U32.to_nat(jump)) : U32),(U32.from_nat(Nat.add(Nat.mul(r1,bn),c1)) + jump : U32),plane_word(band,columns,rows,jump),
      Equal.cong(U32,U32,value => (U32.from_nat(Nat.add(Nat.mul(r1,bn),c1)) + value : U32),U32.from_nat(U32.to_nat(jump)),jump,Arithmetic.native_roundtrip(jump)),
      Equal.cong(U32,U32,value => (value + jump : U32),U32.from_nat(Nat.add(Nat.mul(r1,bn),c1)),((rows - 1) * band + (columns - 1) : U32),
        Equal.trans(U32,U32.from_nat(Nat.add(Nat.mul(r1,bn),c1)),(U32.from_nat(r1) * band + U32.from_nat(c1) : U32),((rows - 1) * band + (columns - 1) : U32),row_offset(band,r1,c1),
          Equal.trans(U32,(U32.from_nat(r1) * band + U32.from_nat(c1) : U32),((rows - 1) * band + U32.from_nat(c1) : U32),((rows - 1) * band + (columns - 1) : U32),
            Equal.cong(U32,U32,value => (value * band + U32.from_nat(c1) : U32),U32.from_nat(r1),(rows - 1 : U32),encoded_less(rows,rows_positive)),
            Equal.cong(U32,U32,value => ((rows - 1) * band + value : U32),U32.from_nat(c1),(columns - 1 : U32),encoded_less(columns,columns_positive)))))))

def grid_offset(+base: U32,+band: U32,+columns: U32,+rows: U32,+jump: U32,+first: Nat,+plane: Nat,+row: Nat,+column: Nat,
  +columns_positive: {Nat.is_lt(0n,U32.to_nat(columns)) == True{} : Bool},+rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool}) ->
  {grid_at_word(Grid{base,U32.to_nat(band),columns,rows,jump,first},plane,row,column) ==
    (base + (U32.from_nat(plane) * plane_word(band,columns,rows,jump) + (U32.from_nat(row) * band + U32.from_nat(column))) : U32) : U32}:
  +bn = U32.to_nat(band)
  +span = grid_span(bn,columns,rows,jump)
  +inner = Nat.add(Nat.mul(row,bn),column)
  Equal.cong(U32,U32,value => (base + value : U32),U32.from_nat(Nat.add(Nat.mul(plane,span),inner)),
    (U32.from_nat(plane) * plane_word(band,columns,rows,jump) + (U32.from_nat(row) * band + U32.from_nat(column)) : U32),
    Equal.trans(U32,U32.from_nat(Nat.add(Nat.mul(plane,span),inner)),(U32.from_nat(Nat.mul(plane,span)) + U32.from_nat(inner) : U32),
      (U32.from_nat(plane) * plane_word(band,columns,rows,jump) + (U32.from_nat(row) * band + U32.from_nat(column)) : U32),
      encoded_sum(Nat.mul(plane,span),inner),
      Equal.trans(U32,(U32.from_nat(Nat.mul(plane,span)) + U32.from_nat(inner) : U32),(U32.from_nat(plane) * plane_word(band,columns,rows,jump) + U32.from_nat(inner) : U32),
        (U32.from_nat(plane) * plane_word(band,columns,rows,jump) + (U32.from_nat(row) * band + U32.from_nat(column)) : U32),
        Equal.cong(U32,U32,value => (value + U32.from_nat(inner) : U32),U32.from_nat(Nat.mul(plane,span)),(U32.from_nat(plane) * plane_word(band,columns,rows,jump) : U32),
          Equal.trans(U32,U32.from_nat(Nat.mul(plane,span)),(U32.from_nat(plane) * U32.from_nat(span) : U32),(U32.from_nat(plane) * plane_word(band,columns,rows,jump) : U32),
            encoded_product(plane,span),
            Equal.cong(U32,U32,value => (U32.from_nat(plane) * value : U32),U32.from_nat(span),plane_word(band,columns,rows,jump),span_word(band,columns,rows,jump,columns_positive,rows_positive)))),
        Equal.cong(U32,U32,value => (U32.from_nat(plane) * plane_word(band,columns,rows,jump) + value : U32),U32.from_nat(inner),(U32.from_nat(row) * band + U32.from_nat(column) : U32),
          row_offset(band,row,column)))))

# The next column of a row is the next address.
def next_address(+base: U32,+low: U32,+next: U32,+step: {(low + 1 : U32) == next : U32}) -> {((base + low : U32) + 1 : U32) == (base + next : U32) : U32}:
  Equal.trans(U32,((base + low : U32) + 1 : U32),(base + (low + 1 : U32) : U32),(base + next : U32),Arithmetic.add_associative(base,low,1),
    Equal.cong(U32,U32,tail => (base + tail : U32),(low + 1 : U32),next,step))

# After the last column, band - width + 1 more reaches the next row.
def wrapped_address(+base: U32,+low: U32,+width: U32,+band: U32,+high: U32,+step: {(low + 1 : U32) == (high + width : U32) : U32}) ->
  {((base + low : U32) + ((band - width : U32) + 1 : U32) : U32) == (base + (band + high : U32) : U32) : U32}:
  +rest = (band - width : U32)
  +inner = Equal.trans(U32,(low + (rest + 1 : U32) : U32),(low + (1 + rest : U32) : U32),(band + high : U32),
    Equal.cong(U32,U32,tail => (low + tail : U32),(rest + 1 : U32),(1 + rest : U32),Arithmetic.add_commutative(rest,1)),
    Equal.trans(U32,(low + (1 + rest : U32) : U32),((low + 1 : U32) + rest : U32),(band + high : U32),
      Equal.sym(U32,((low + 1 : U32) + rest : U32),(low + (1 + rest : U32) : U32),Arithmetic.add_associative(low,1,rest)),
      Equal.trans(U32,((low + 1 : U32) + rest : U32),((high + width : U32) + rest : U32),(band + high : U32),
        Equal.cong(U32,U32,head => (head + rest : U32),(low + 1 : U32),(high + width : U32),step),
        Equal.trans(U32,((high + width : U32) + rest : U32),(high + (width + rest : U32) : U32),(band + high : U32),
          Arithmetic.add_associative(high,width,rest),
          Equal.trans(U32,(high + (width + rest : U32) : U32),(high + band : U32),(band + high : U32),
            Equal.cong(U32,U32,tail => (high + tail : U32),(width + rest : U32),band,
              Equal.trans(U32,(width + rest : U32),((width + band : U32) - width : U32),band,
                Equal.sym(U32,((width + band : U32) - width : U32),(width + (band - width : U32) : U32),Arithmetic.subtract_from_sum(width,band,width)),
                Arithmetic.subtract_first(width,band))),
            Arithmetic.add_commutative(high,band))))))
  Equal.trans(U32,((base + low : U32) + (rest + 1 : U32) : U32),(base + (low + (rest + 1 : U32) : U32) : U32),(base + (band + high : U32) : U32),
    Arithmetic.add_associative(base,low,(rest + 1 : U32)),
    Equal.cong(U32,U32,tail => (base + tail : U32),(low + (rest + 1 : U32) : U32),(band + high : U32),inner))

# A U32 comparison of a small successor is the natural one.
def successor_below(+value: Nat,+bound: U32,inside: Bool,+room: {Nat.is_le(1n+value,U32.to_nat(bound)) == True{} : Bool},
  +compared: {Nat.is_lt(1n+value,U32.to_nat(bound)) == inside : Bool}) -> {U32.is_lt((U32.from_nat(value) + 1 : U32),bound) == inside : Bool}:
  %Equal.sym(U32,(U32.from_nat(value) + 1 : U32),U32.from_nat(1n+value),Bounds.increment_encoded(value)) : {U32.is_lt(_,bound) == inside : Bool}
  %Equal.sym(Bool,U32.is_lt(U32.from_nat(1n+value),bound),Nat.is_lt(U32.to_nat(U32.from_nat(1n+value)),U32.to_nat(bound)),Comparison.lt(U32.from_nat(1n+value),bound)) : {_ == inside : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(1n+value)),1n+value,FullRange.roundtrip(1n+value,bound,room)) : {Nat.is_lt(_,U32.to_nat(bound)) == inside : Bool}
  compared

# The next column of a plane row.
def column_address(+base: U32,+band: Nat,+span: Nat,+plane: Nat,+row: Nat,+column: Nat) ->
  {(grid_address(base,band,span,plane,row,column) + 1 : U32) == grid_address(base,band,span,plane,row,1n+column) : U32}:
  +head = Nat.mul(plane,span)
  +line = Nat.mul(row,band)
  +low = Nat.add(head,Nat.add(line,column))
  +next = Nat.add(head,Nat.add(line,1n+column))
  +moved = Equal.trans(Nat,next,Nat.add(head,1n+Nat.add(line,column)),1n+low,
    Equal.cong(Nat,Nat,tail => Nat.add(head,tail),Nat.add(line,1n+column),1n+Nat.add(line,column),Addition.successor_right(line,column)),
    Addition.successor_right(head,Nat.add(line,column)))
  next_address(base,U32.from_nat(low),U32.from_nat(next),
    Equal.trans(U32,(U32.from_nat(low) + 1 : U32),U32.from_nat(1n+low),U32.from_nat(next),Bounds.increment_encoded(low),
      Equal.cong(Nat,U32,U32.from_nat,1n+low,next,Equal.sym(Nat,next,1n+low,moved))))

# After the last column of a plane row, the row jump reaches the next row.
def next_row_address(+base: U32,+band: Nat,+span: Nat,+plane: Nat,+row: Nat,+column: Nat,+columns: U32,+last: {1n+column == U32.to_nat(columns) : Nat}) ->
  {(grid_address(base,band,span,plane,row,column) + ((U32.from_nat(band) - columns : U32) + 1 : U32) : U32) == grid_address(base,band,span,plane,1n+row,0n) : U32}:
  +head = Nat.mul(plane,span)
  +line = Nat.mul(row,band)
  +width = U32.to_nat(columns)
  +low = Nat.add(head,Nat.add(line,column))
  +high = Nat.add(head,line)
  +step = Equal.trans(U32,(U32.from_nat(low) + 1 : U32),U32.from_nat(Nat.add(high,width)),(U32.from_nat(high) + columns : U32),
    Equal.trans(U32,(U32.from_nat(low) + 1 : U32),U32.from_nat(1n+low),U32.from_nat(Nat.add(high,width)),Bounds.increment_encoded(low),
      Equal.cong(Nat,U32,U32.from_nat,1n+low,Nat.add(high,width),
        Equal.trans(Nat,1n+low,Nat.add(head,Nat.add(line,1n+column)),Nat.add(high,width),
          Equal.sym(Nat,Nat.add(head,Nat.add(line,1n+column)),1n+low,
            Equal.trans(Nat,Nat.add(head,Nat.add(line,1n+column)),Nat.add(head,1n+Nat.add(line,column)),1n+low,
              Equal.cong(Nat,Nat,tail => Nat.add(head,tail),Nat.add(line,1n+column),1n+Nat.add(line,column),Addition.successor_right(line,column)),
              Addition.successor_right(head,Nat.add(line,column)))),
          Equal.trans(Nat,Nat.add(head,Nat.add(line,1n+column)),Nat.add(head,Nat.add(line,width)),Nat.add(high,width),
            Equal.cong(Nat,Nat,last_column => Nat.add(head,Nat.add(line,last_column)),1n+column,width,last),
            Equal.sym(Nat,Nat.add(high,width),Nat.add(head,Nat.add(line,width)),Addition.associative(head,line,width)))))),
    Equal.trans(U32,U32.from_nat(Nat.add(high,width)),(U32.from_nat(high) + U32.from_nat(width) : U32),(U32.from_nat(high) + columns : U32),
      Equal.sym(U32,U32.add(U32.from_nat(high),U32.from_nat(width)),U32.from_nat(Nat.add(high,width)),WordArithmetic.native_add(high,width)),
      Equal.cong(U32,U32,tail => (U32.from_nat(high) + tail : U32),U32.from_nat(width),columns,Arithmetic.native_roundtrip(columns))))
  +target = Equal.trans(Nat,Nat.add(band,high),Nat.add(head,Nat.add(band,line)),Nat.add(head,Nat.add(Nat.add(band,line),0n)),
    RowMajor.add_swap(band,head,line),
    Equal.cong(Nat,Nat,tail => Nat.add(head,tail),Nat.add(band,line),Nat.add(Nat.add(band,line),0n),Equal.sym(Nat,Nat.add(Nat.add(band,line),0n),Nat.add(band,line),Addition.zero_right(Nat.add(band,line)))))
  Equal.trans(U32,(grid_address(base,band,span,plane,row,column) + ((U32.from_nat(band) - columns : U32) + 1 : U32) : U32),
    (base + (U32.from_nat(band) + U32.from_nat(high) : U32) : U32),grid_address(base,band,span,plane,1n+row,0n),
    wrapped_address(base,U32.from_nat(low),columns,U32.from_nat(band),U32.from_nat(high),step),
    Equal.cong(U32,U32,tail => (base + tail : U32),(U32.from_nat(band) + U32.from_nat(high) : U32),U32.from_nat(Nat.add(head,Nat.add(Nat.add(band,line),0n))),
      Equal.trans(U32,(U32.from_nat(band) + U32.from_nat(high) : U32),U32.from_nat(Nat.add(band,high)),U32.from_nat(Nat.add(head,Nat.add(Nat.add(band,line),0n))),
        WordArithmetic.native_add(band,high),Equal.cong(Nat,U32,U32.from_nat,Nat.add(band,high),Nat.add(head,Nat.add(Nat.add(band,line),0n)),target))))

# The span of a plane whose last entry is (row, column).
def span_last(+band: Nat,+columns: U32,+rows: U32,+jump: U32,+row: Nat,+column: Nat,+row_last: {1n+row == U32.to_nat(rows) : Nat},
  +column_last: {1n+column == U32.to_nat(columns) : Nat}) ->
  {grid_span(band,columns,rows,jump) == Nat.add(Nat.add(Nat.mul(row,band),column),U32.to_nat(jump)) : Nat}:
  +extra = U32.to_nat(jump)
  +wanted = Nat.add(Nat.add(Nat.mul(row,band),column),extra)
  %row_last : {Nat.add(Nat.add(Nat.mul(Nat.sub(_,1n),band),Nat.sub(U32.to_nat(columns),1n)),extra) == wanted : Nat}
  %column_last : {Nat.add(Nat.add(Nat.mul(Nat.sub(1n+row,1n),band),Nat.sub(_,1n)),extra) == wanted : Nat}
  %Equal.sym(Nat,Nat.sub(row,0n),row,Order.subtract_zero(row)) : {Nat.add(Nat.add(Nat.mul(_,band),Nat.sub(column,0n)),extra) == wanted : Nat}
  %Equal.sym(Nat,Nat.sub(column,0n),column,Order.subtract_zero(column)) : {Nat.add(Nat.add(Nat.mul(row,band),_),extra) == wanted : Nat}
  {==}

# After the last entry of a plane, the jump reaches the next plane.
def plane_address(+base: U32,+band: Nat,+columns: U32,+rows: U32,+jump: U32,+plane: Nat,+row: Nat,+column: Nat,
  +row_last: {1n+row == U32.to_nat(rows) : Nat},+column_last: {1n+column == U32.to_nat(columns) : Nat}) ->
  {(grid_address(base,band,grid_span(band,columns,rows,jump),plane,row,column) + jump : U32) == grid_address(base,band,grid_span(band,columns,rows,jump),1n+plane,0n,0n) : U32}:
  +span = grid_span(band,columns,rows,jump)
  +extra = U32.to_nat(jump)
  +head = Nat.mul(plane,span)
  +inner = Nat.add(Nat.mul(row,band),column)
  +low = Nat.add(head,inner)
  +total = Equal.trans(Nat,Nat.add(low,extra),Nat.add(head,Nat.add(inner,extra)),Nat.add(Nat.add(span,head),0n),
    Addition.associative(head,inner,extra),
    Equal.trans(Nat,Nat.add(head,Nat.add(inner,extra)),Nat.add(head,span),Nat.add(Nat.add(span,head),0n),
      Equal.cong(Nat,Nat,tail => Nat.add(head,tail),Nat.add(inner,extra),span,Equal.sym(Nat,span,Nat.add(inner,extra),span_last(band,columns,rows,jump,row,column,row_last,column_last))),
      Equal.trans(Nat,Nat.add(head,span),Nat.add(span,head),Nat.add(Nat.add(span,head),0n),Order.commutative(head,span),
        Equal.sym(Nat,Nat.add(Nat.add(span,head),0n),Nat.add(span,head),Addition.zero_right(Nat.add(span,head))))))
  Equal.trans(U32,((base + U32.from_nat(low) : U32) + jump : U32),(base + (U32.from_nat(low) + jump : U32) : U32),(base + U32.from_nat(Nat.add(Nat.add(span,head),0n)) : U32),
    Arithmetic.add_associative(base,U32.from_nat(low),jump),
    Equal.cong(U32,U32,tail => (base + tail : U32),(U32.from_nat(low) + jump : U32),U32.from_nat(Nat.add(Nat.add(span,head),0n)),
      Equal.trans(U32,(U32.from_nat(low) + jump : U32),(U32.from_nat(low) + U32.from_nat(extra) : U32),U32.from_nat(Nat.add(Nat.add(span,head),0n)),
        Equal.cong(U32,U32,tail => (U32.from_nat(low) + tail : U32),jump,U32.from_nat(extra),Equal.sym(U32,U32.from_nat(extra),jump,Arithmetic.native_roundtrip(jump))),
        Equal.trans(U32,(U32.from_nat(low) + U32.from_nat(extra) : U32),U32.from_nat(Nat.add(low,extra)),U32.from_nat(Nat.add(Nat.add(span,head),0n)),
          WordArithmetic.native_add(low,extra),Equal.cong(Nat,U32,U32.from_nat,Nat.add(low,extra),Nat.add(Nat.add(span,head),0n),total)))))

# One step of a grid reader at (plane, row, column) before the last column.
def column_entry(+grid: Grid,+plane: Nat,+row: Nat,+column: Nat,+next_column: {Nat.is_lt(1n+column,grid_width(grid)) == True{} : Bool},
  +column_small: {Nat.is_lt(column,grid_width(grid)) == True{} : Bool}) ->
  {Gemm.entry(grid_at(grid,plane,row,column)) == (grid_at_word(grid,plane,row,column),grid_at(grid,plane,row,1n+column)) : U32 & Gemm.Entries}:
  Grid{+base,+band,+columns,+rows,+jump,+first} = grid
  +span = grid_span(band,columns,rows,jump)
  +address = grid_address(base,band,span,plane,row,column)
  +turn = (U32.from_nat(band) - columns + 1 : U32)
  +column_word = U32.from_nat(column)
  +row_word = U32.from_nat(row)
  %Equal.sym(Bool,U32.is_lt((column_word + 1 : U32),columns),True{},
      successor_below(column,columns,True{},Order.strict_follow(column,U32.to_nat(columns),column_small),next_column)) :
    {(address,Gemm.entry_next(_,U32.is_lt((row_word + 1 : U32),rows),address,column_word,row_word,columns,rows,turn,jump)) ==
      (address,Gemm.Entries{grid_address(base,band,span,plane,row,1n+column),U32.from_nat(1n+column),row_word,columns,rows,turn,jump}) : U32 & Gemm.Entries}
  %Equal.sym(U32,(column_word + 1 : U32),U32.from_nat(1n+column),Bounds.increment_encoded(column)) :
    {(address,Gemm.Entries{(address + 1 : U32),_,row_word,columns,rows,turn,jump}) ==
      (address,Gemm.Entries{grid_address(base,band,span,plane,row,1n+column),U32.from_nat(1n+column),row_word,columns,rows,turn,jump}) : U32 & Gemm.Entries}
  %Equal.sym(U32,(address + 1 : U32),grid_address(base,band,span,plane,row,1n+column),column_address(base,band,span,plane,row,column)) :
    {(address,Gemm.Entries{_,U32.from_nat(1n+column),row_word,columns,rows,turn,jump}) ==
      (address,Gemm.Entries{grid_address(base,band,span,plane,row,1n+column),U32.from_nat(1n+column),row_word,columns,rows,turn,jump}) : U32 & Gemm.Entries}
  {==}

# After the last column of a row before the last row of a plane.
def row_entry(+grid: Grid,+plane: Nat,+row: Nat,+column: Nat,+next_column: {Nat.is_lt(1n+column,grid_width(grid)) == False{} : Bool},
  +next_row: {Nat.is_lt(1n+row,grid_rows(grid)) == True{} : Bool},
  +column_small: {Nat.is_lt(column,grid_width(grid)) == True{} : Bool},+row_small: {Nat.is_lt(row,grid_rows(grid)) == True{} : Bool}) ->
  {Gemm.entry(grid_at(grid,plane,row,column)) == (grid_at_word(grid,plane,row,column),grid_at(grid,plane,1n+row,0n)) : U32 & Gemm.Entries}:
  Grid{+base,+band,+columns,+rows,+jump,+first} = grid
  +span = grid_span(band,columns,rows,jump)
  +address = grid_address(base,band,span,plane,row,column)
  +turn = (U32.from_nat(band) - columns + 1 : U32)
  +column_word = U32.from_nat(column)
  +row_word = U32.from_nat(row)
  %Equal.sym(Bool,U32.is_lt((column_word + 1 : U32),columns),False{},
      successor_below(column,columns,False{},Order.strict_follow(column,U32.to_nat(columns),column_small),next_column)) :
    {(address,Gemm.entry_next(_,U32.is_lt((row_word + 1 : U32),rows),address,column_word,row_word,columns,rows,turn,jump)) ==
      (address,Gemm.Entries{grid_address(base,band,span,plane,1n+row,0n),U32.from_nat(0n),U32.from_nat(1n+row),columns,rows,turn,jump}) : U32 & Gemm.Entries}
  %Equal.sym(Bool,U32.is_lt((row_word + 1 : U32),rows),True{},successor_below(row,rows,True{},Order.strict_follow(row,U32.to_nat(rows),row_small),next_row)) :
    {(address,Gemm.entry_next(False{},_,address,column_word,row_word,columns,rows,turn,jump)) ==
      (address,Gemm.Entries{grid_address(base,band,span,plane,1n+row,0n),U32.from_nat(0n),U32.from_nat(1n+row),columns,rows,turn,jump}) : U32 & Gemm.Entries}
  %Equal.sym(U32,(row_word + 1 : U32),U32.from_nat(1n+row),Bounds.increment_encoded(row)) :
    {(address,Gemm.Entries{(address + turn : U32),0,_,columns,rows,turn,jump}) ==
      (address,Gemm.Entries{grid_address(base,band,span,plane,1n+row,0n),U32.from_nat(0n),U32.from_nat(1n+row),columns,rows,turn,jump}) : U32 & Gemm.Entries}
  %Equal.sym(U32,(address + turn : U32),grid_address(base,band,span,plane,1n+row,0n),
      next_row_address(base,band,span,plane,row,column,columns,DivisionOrder.not_below(column,U32.to_nat(columns),column_small,next_column))) :
    {(address,Gemm.Entries{_,0,U32.from_nat(1n+row),columns,rows,turn,jump}) ==
      (address,Gemm.Entries{grid_address(base,band,span,plane,1n+row,0n),U32.from_nat(0n),U32.from_nat(1n+row),columns,rows,turn,jump}) : U32 & Gemm.Entries}
  {==}

# After the last entry of a plane.
def plane_entry(+grid: Grid,+plane: Nat,+row: Nat,+column: Nat,+next_column: {Nat.is_lt(1n+column,grid_width(grid)) == False{} : Bool},
  +next_row: {Nat.is_lt(1n+row,grid_rows(grid)) == False{} : Bool},
  +column_small: {Nat.is_lt(column,grid_width(grid)) == True{} : Bool},+row_small: {Nat.is_lt(row,grid_rows(grid)) == True{} : Bool}) ->
  {Gemm.entry(grid_at(grid,plane,row,column)) == (grid_at_word(grid,plane,row,column),grid_at(grid,1n+plane,0n,0n)) : U32 & Gemm.Entries}:
  Grid{+base,+band,+columns,+rows,+jump,+first} = grid
  +span = grid_span(band,columns,rows,jump)
  +address = grid_address(base,band,span,plane,row,column)
  +turn = (U32.from_nat(band) - columns + 1 : U32)
  +column_word = U32.from_nat(column)
  +row_word = U32.from_nat(row)
  %Equal.sym(Bool,U32.is_lt((column_word + 1 : U32),columns),False{},
      successor_below(column,columns,False{},Order.strict_follow(column,U32.to_nat(columns),column_small),next_column)) :
    {(address,Gemm.entry_next(_,U32.is_lt((row_word + 1 : U32),rows),address,column_word,row_word,columns,rows,turn,jump)) ==
      (address,Gemm.Entries{grid_address(base,band,span,1n+plane,0n,0n),U32.from_nat(0n),U32.from_nat(0n),columns,rows,turn,jump}) : U32 & Gemm.Entries}
  %Equal.sym(Bool,U32.is_lt((row_word + 1 : U32),rows),False{},successor_below(row,rows,False{},Order.strict_follow(row,U32.to_nat(rows),row_small),next_row)) :
    {(address,Gemm.entry_next(False{},_,address,column_word,row_word,columns,rows,turn,jump)) ==
      (address,Gemm.Entries{grid_address(base,band,span,1n+plane,0n,0n),U32.from_nat(0n),U32.from_nat(0n),columns,rows,turn,jump}) : U32 & Gemm.Entries}
  %Equal.sym(U32,(address + jump : U32),grid_address(base,band,span,1n+plane,0n,0n),
      plane_address(base,band,columns,rows,jump,plane,row,column,DivisionOrder.not_below(row,U32.to_nat(rows),row_small,next_row),
        DivisionOrder.not_below(column,U32.to_nat(columns),column_small,next_column))) :
    {(address,Gemm.Entries{_,0,0,columns,rows,turn,jump}) ==
      (address,Gemm.Entries{grid_address(base,band,span,1n+plane,0n,0n),U32.from_nat(0n),U32.from_nat(0n),columns,rows,turn,jump}) : U32 & Gemm.Entries}
  {==}

# One step of a grid reader at (plane, row, column): the next column, the
# first column of the next row, or the first entry of the next plane.
def entry_at(+grid: Grid,+plane: Nat,+row: Nat,+column: Nat,more_columns: Bool,more_rows: Bool,
  +next_column: {Nat.is_lt(1n+column,grid_width(grid)) == more_columns : Bool},+next_row: {Nat.is_lt(1n+row,grid_rows(grid)) == more_rows : Bool},
  +column_small: {Nat.is_lt(column,grid_width(grid)) == True{} : Bool},+row_small: {Nat.is_lt(row,grid_rows(grid)) == True{} : Bool}) ->
  {Gemm.entry(grid_at(grid,plane,row,column)) == (grid_at_word(grid,plane,row,column),
    Bool.pick(Gemm.Entries,more_columns,grid_at(grid,plane,row,1n+column),Bool.pick(Gemm.Entries,more_rows,grid_at(grid,plane,1n+row,0n),grid_at(grid,1n+plane,0n,0n)))) :
    U32 & Gemm.Entries}:
  match more_columns more_rows:
    case True{} more_rows: column_entry(grid,plane,row,column,next_column,column_small)
    case False{} True{}: row_entry(grid,plane,row,column,next_column,next_row,column_small,row_small)
    case False{} False{}: plane_entry(grid,plane,row,column,next_column,next_row,column_small,row_small)

# A position of a grid is its row and column.
def grid_split(+grid: Grid,+step: Nat,+positive: {Nat.is_lt(0n,grid_width(grid)) == True{} : Bool}) ->
  {Nat.divmod(grid_position(grid,step),grid_width(grid)) ==
    (Nat.div(grid_position(grid,step),grid_width(grid)),Nat.mod(grid_position(grid,step),grid_width(grid))) : Nat & Nat}:
  +position = grid_position(grid,step)
  +width = grid_width(grid)
  Division.unique(position,width,Nat.div(position,width),Nat.mod(position,width),DivisionOrder.quotient_exact(position,width,positive),positive,
    DivisionOrder.remainder_small(position,width,positive))

def grid_next(+grid: Grid,+step: Nat) -> {grid_position(grid,1n+step) == 1n+grid_position(grid,step) : Nat}:
  Grid{+base,+band,+columns,+rows,+jump,+first} = grid
  Addition.successor_right(first,step)

# The row after the last column of a line of the planes.
def next_line(+grid: Grid,+line: Nat,+plane: Nat,+row: Nat,more_rows: Bool,
  +rows_split: {Nat.divmod(1n+line,grid_rows(grid)) == Bool.pick(Nat & Nat,more_rows,(plane,1n+row),(1n+plane,0n)) : Nat & Nat}) ->
  {grid_cell(grid,(1n+line,0n)) == Bool.pick(Gemm.Entries,more_rows,grid_at(grid,plane,1n+row,0n),grid_at(grid,1n+plane,0n,0n)) : Gemm.Entries}:
  match more_rows:
    case True{}:
      %Equal.sym(Nat & Nat,Nat.divmod(1n+line,grid_rows(grid)),(plane,1n+row),rows_split) : {grid_placed(grid,_,0n) == grid_at(grid,plane,1n+row,0n) : Gemm.Entries}
      {==}
    case False{}:
      %Equal.sym(Nat & Nat,Nat.divmod(1n+line,grid_rows(grid)),(1n+plane,0n),rows_split) : {grid_placed(grid,_,0n) == grid_at(grid,1n+plane,0n,0n) : Gemm.Entries}
      {==}

# The entry after (line, column) of the planes.
def next_cell(+grid: Grid,+line: Nat,+column: Nat,+plane: Nat,+row: Nat,more_columns: Bool,+more_rows: Bool,
  +split: {Nat.divmod(line,grid_rows(grid)) == (plane,row) : Nat & Nat},
  +rows_split: {Nat.divmod(1n+line,grid_rows(grid)) == Bool.pick(Nat & Nat,more_rows,(plane,1n+row),(1n+plane,0n)) : Nat & Nat}) ->
  {grid_cell(grid,Bool.pick(Nat & Nat,more_columns,(line,1n+column),(1n+line,0n))) ==
    Bool.pick(Gemm.Entries,more_columns,grid_at(grid,plane,row,1n+column),Bool.pick(Gemm.Entries,more_rows,grid_at(grid,plane,1n+row,0n),grid_at(grid,1n+plane,0n,0n))) :
    Gemm.Entries}:
  match more_columns:
    case True{}:
      %Equal.sym(Nat & Nat,Nat.divmod(line,grid_rows(grid)),(plane,row),split) : {grid_placed(grid,_,1n+column) == grid_at(grid,plane,row,1n+column) : Gemm.Entries}
      {==}
    case False{}: next_line(grid,line,plane,row,more_rows,rows_split)

# A grid reader reads its position and moves to the next.
def entry_step(+grid: Grid,+step: Nat,+columns_positive: {Nat.is_lt(0n,grid_width(grid)) == True{} : Bool},
  +rows_positive: {Nat.is_lt(0n,grid_rows(grid)) == True{} : Bool}) ->
  {Gemm.entry(grid_entry(grid,step)) == (grid_read(grid,step),grid_entry(grid,1n+step)) : U32 & Gemm.Entries}:
  +position = grid_position(grid,step)
  +width = grid_width(grid)
  +height = grid_rows(grid)
  +line = Nat.div(position,width)
  +column = Nat.mod(position,width)
  +plane = Nat.div(line,height)
  +row = Nat.mod(line,height)
  +more_columns = Nat.is_lt(1n+column,width)
  +more_rows = Nat.is_lt(1n+row,height)
  +column_small = DivisionOrder.remainder_small(position,width,columns_positive)
  +row_small = DivisionOrder.remainder_small(line,height,rows_positive)
  +line_exact = DivisionOrder.quotient_exact(position,width,columns_positive)
  +plane_exact = DivisionOrder.quotient_exact(line,height,rows_positive)
  +split = Division.unique(line,height,plane,row,plane_exact,rows_positive,row_small)
  +rows_split = DivisionOrder.advanced_pair(line,height,plane,row,more_rows,plane_exact,row_small,rows_positive,{==})
  +after = Bool.pick(Gemm.Entries,more_columns,grid_at(grid,plane,row,1n+column),Bool.pick(Gemm.Entries,more_rows,grid_at(grid,plane,1n+row,0n),grid_at(grid,1n+plane,0n,0n)))
  %Equal.sym(Nat & Nat,Nat.divmod(position,width),(line,column),grid_split(grid,step,columns_positive)) :
    {Gemm.entry(grid_cell(grid,_)) == (grid_word(grid,_),grid_cell(grid,Nat.divmod(grid_position(grid,1n+step),width))) : U32 & Gemm.Entries}
  %Equal.sym(Nat & Nat,Nat.divmod(line,height),(plane,row),split) :
    {Gemm.entry(grid_placed(grid,_,column)) == (grid_placed_word(grid,_,column),grid_cell(grid,Nat.divmod(grid_position(grid,1n+step),width))) : U32 & Gemm.Entries}
  %Equal.sym(Nat,grid_position(grid,1n+step),1n+position,grid_next(grid,step)) :
    {Gemm.entry(grid_at(grid,plane,row,column)) == (grid_at_word(grid,plane,row,column),grid_cell(grid,Nat.divmod(_,width))) : U32 & Gemm.Entries}
  %Equal.sym(Nat & Nat,Nat.divmod(1n+position,width),Bool.pick(Nat & Nat,more_columns,(line,1n+column),(1n+line,0n)),
      DivisionOrder.advanced_pair(position,width,line,column,more_columns,line_exact,column_small,columns_positive,{==})) :
    {Gemm.entry(grid_at(grid,plane,row,column)) == (grid_at_word(grid,plane,row,column),grid_cell(grid,_)) : U32 & Gemm.Entries}
  %Equal.sym(Gemm.Entries,grid_cell(grid,Bool.pick(Nat & Nat,more_columns,(line,1n+column),(1n+line,0n))),after,
      next_cell(grid,line,column,plane,row,more_columns,more_rows,split,rows_split)) :
    {Gemm.entry(grid_at(grid,plane,row,column)) == (grid_at_word(grid,plane,row,column),_) : U32 & Gemm.Entries}
  entry_at(grid,plane,row,column,more_columns,more_rows,{==},{==},column_small,row_small)

# A grid reader from step reads the positions of its steps.
def entry_reads(count: Nat,+grid: Grid,+step: Nat,+columns_positive: {Nat.is_lt(0n,grid_width(grid)) == True{} : Bool},
  +rows_positive: {Nat.is_lt(0n,grid_rows(grid)) == True{} : Bool}) ->
  {reads(~Gemm.Entries,~Gemm.entry,count,grid_entry(grid,step)) == Ordered.map_keys(~Nat,~U32,~Grid,~grid_read,Ranges.range(count,step),grid) : List<&2,U32>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(U32 & Gemm.Entries,Gemm.entry(grid_entry(grid,step)),(grid_read(grid,step),grid_entry(grid,1n+step)),entry_step(grid,step,columns_positive,rows_positive)) :
        {Pair.fst(U32,Gemm.Entries,_) <> reads(~Gemm.Entries,~Gemm.entry,rest,Pair.snd(U32,Gemm.Entries,_)) ==
          Ordered.map_keys(~Nat,~U32,~Grid,~grid_read,Ranges.range(1n+rest,step),grid) : List<&2,U32>}
      Equal.cong(List<&2,U32>,List<&2,U32>,tail => grid_read(grid,step) <> tail,
        reads(~Gemm.Entries,~Gemm.entry,rest,grid_entry(grid,1n+step)),Ordered.map_keys(~Nat,~U32,~Grid,~grid_read,Ranges.range(rest,1n+step),grid),
        entry_reads(rest,grid,1n+step,columns_positive,rows_positive))

# Step (plane * rows + row) * columns + column of a grid from position 0 is
# entry (plane, row, column).
def grid_read_index(+base: U32,+band: Nat,+columns: U32,+rows: U32,+jump: U32,+plane: Nat,+row: Nat,+column: Nat,
  +row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(columns)) == True{} : Bool}) ->
  {grid_read(Grid{base,band,columns,rows,jump,0n},RowMajor.index(plane,U32.to_nat(rows),U32.to_nat(columns),row,column)) ==
    grid_at_word(Grid{base,band,columns,rows,jump,0n},plane,row,column) : U32}:
  +grid = {Grid{base,band,columns,rows,jump,0n} : Grid}
  +r = U32.to_nat(rows)
  +c = U32.to_nat(columns)
  +index = RowMajor.index(plane,r,c,row,column)
  +line = Nat.add(Nat.mul(plane,r),row)
  +cells = Division.unique(index,c,line,column,Equal.sym(Nat,Nat.add(Nat.mul(line,c),column),index,RowMajor.mixed(plane,r,c,row,column)),
    Order.positive_bound(column,c,column_inside),column_inside)
  +planes = Division.unique(line,r,plane,row,{==},Order.positive_bound(row,r,row_inside),row_inside)
  %Equal.sym(Nat & Nat,Nat.divmod(index,c),(line,column),cells) : {grid_word(grid,_) == grid_at_word(grid,plane,row,column) : U32}
  %Equal.sym(Nat & Nat,Nat.divmod(line,r),(plane,row),planes) : {grid_placed_word(grid,_,column) == grid_at_word(grid,plane,row,column) : U32}
  {==}

# A row of the first plane is at its row of the band.

def first_plane(+grid: Grid,+row: Nat,+column: Nat,+inside: {Nat.is_lt(row,grid_rows(grid)) == True{} : Bool}) ->
  {grid_word(grid,(row,column)) == grid_at_word(grid,0n,row,column) : U32}:
  %Equal.sym(Nat & Nat,Nat.divmod(row,grid_rows(grid)),(0n,row),Division.small(row,grid_rows(grid),inside)) :
    {grid_placed_word(grid,_,column) == grid_at_word(grid,0n,row,column) : U32}
  {==}

def select_row(row: Nat,accumulators: Gemm.Accumulators) -> Tile.Tile8:
  match row:
    case 0n: Gemm.row0(accumulators)
    case 1n+a:
      match a:
        case 0n: Gemm.row1(accumulators)
        case 1n+b:
          match b:
            case 0n: Gemm.row2(accumulators)
            case 1n+c:
              match c:
                case 0n: Gemm.row3(accumulators)
                case 1n+d:
                  match d:
                    case 0n: Gemm.row4(accumulators)
                    case 1n+e:
                      match e:
                        case 0n: Gemm.row5(accumulators)
                        case 1n+f:
                          match f:
                            case 0n: Gemm.row6(accumulators)
                            case 1n+g: Gemm.row7(accumulators)

def cell(+row: Nat,+lane: Nat,+accumulators: Gemm.Accumulators) -> F32:
  TileLaws.select_lane(lane,select_row(row,accumulators))

# A row of the block (below eight) takes the multiply-add of its value in the row tile.
def updated_row(~times: F32 -> F32 -> F32,+row: Nat,+accumulators: Gemm.Accumulators,+lane_tile: Tile.Tile8,+row_tile: Tile.Tile8,small: {Nat.is_lt(row,8n) == True{} : Bool}) ->
  {select_row(row,Gemm.update(~times,accumulators,lane_tile,row_tile)) == Tile.update(~times,select_row(row,accumulators),lane_tile,TileLaws.select_lane(row,row_tile)) : Tile.Tile8}:
  match row:
    case 0n:
      Gemm.Accumulators{+a0,+a1,+a2,+a3,+a4,+a5,+a6,+a7} = accumulators
      Tile.Tile{+w0,+w1,+w2,+w3,+w4,+w5,+w6,+w7} = row_tile
      {==}
    case 1n+ +a:
      match a:
        case 0n:
          Gemm.Accumulators{+a0,+a1,+a2,+a3,+a4,+a5,+a6,+a7} = accumulators
          Tile.Tile{+w0,+w1,+w2,+w3,+w4,+w5,+w6,+w7} = row_tile
          {==}
        case 1n+ +b:
          match b:
            case 0n:
              Gemm.Accumulators{+a0,+a1,+a2,+a3,+a4,+a5,+a6,+a7} = accumulators
              Tile.Tile{+w0,+w1,+w2,+w3,+w4,+w5,+w6,+w7} = row_tile
              {==}
            case 1n+ +c:
              match c:
                case 0n:
                  Gemm.Accumulators{+a0,+a1,+a2,+a3,+a4,+a5,+a6,+a7} = accumulators
                  Tile.Tile{+w0,+w1,+w2,+w3,+w4,+w5,+w6,+w7} = row_tile
                  {==}
                case 1n+ +d:
                  match d:
                    case 0n:
                      Gemm.Accumulators{+a0,+a1,+a2,+a3,+a4,+a5,+a6,+a7} = accumulators
                      Tile.Tile{+w0,+w1,+w2,+w3,+w4,+w5,+w6,+w7} = row_tile
                      {==}
                    case 1n+ +e:
                      match e:
                        case 0n:
                          Gemm.Accumulators{+a0,+a1,+a2,+a3,+a4,+a5,+a6,+a7} = accumulators
                          Tile.Tile{+w0,+w1,+w2,+w3,+w4,+w5,+w6,+w7} = row_tile
                          {==}
                        case 1n+ +f:
                          match f:
                            case 0n:
                              Gemm.Accumulators{+a0,+a1,+a2,+a3,+a4,+a5,+a6,+a7} = accumulators
                              Tile.Tile{+w0,+w1,+w2,+w3,+w4,+w5,+w6,+w7} = row_tile
                              {==}
                            case 1n+ +g:
                              match g:
                                case 0n:
                                  Gemm.Accumulators{+a0,+a1,+a2,+a3,+a4,+a5,+a6,+a7} = accumulators
                                  Tile.Tile{+w0,+w1,+w2,+w3,+w4,+w5,+w6,+w7} = row_tile
                                  {==}
                                case 1n+ +h:
                                  Empty.absurd({select_row(8n+h,Gemm.update(~times,accumulators,lane_tile,row_tile)) == Tile.update(~times,select_row(8n+h,accumulators),lane_tile,TileLaws.select_lane(8n+h,row_tile)) : Tile.Tile8},
                                    WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(h,0n),True{},Equal.sym(Bool,Nat.is_lt(h,0n),False{},Division.lt_zero(h)),small)))

def updated_cell(~times: F32 -> F32 -> F32,+row: Nat,+lane: Nat,+accumulators: Gemm.Accumulators,+lane_tile: Tile.Tile8,+row_tile: Tile.Tile8,+small: {Nat.is_lt(row,8n) == True{} : Bool}) ->
  {cell(row,lane,Gemm.update(~times,accumulators,lane_tile,row_tile)) ==
    (cell(row,lane,accumulators) + times(TileLaws.select_lane(lane,lane_tile),TileLaws.select_lane(row,row_tile)) : F32) : F32}:
  %Equal.sym(Tile.Tile8,select_row(row,Gemm.update(~times,accumulators,lane_tile,row_tile)),
    Tile.update(~times,select_row(row,accumulators),lane_tile,TileLaws.select_lane(row,row_tile)),updated_row(~times,row,accumulators,lane_tile,row_tile,small)) :
    {TileLaws.select_lane(lane,_) == (cell(row,lane,accumulators) + times(TileLaws.select_lane(lane,lane_tile),TileLaws.select_lane(row,row_tile)) : F32) : F32}
  TileLaws.updated_by_value(~times,select_row(row,accumulators),lane_tile,TileLaws.select_lane(row,row_tile),lane)

# A cell after one update in the order.
def ordered_cell(+order: GemmLaws.Order,+row: Nat,+lane: Nat,+accumulators: Gemm.Accumulators,+lane_tile: Tile.Tile8,+row_tile: Tile.Tile8,
  +small: {Nat.is_lt(row,8n) == True{} : Bool}) ->
  {cell(row,lane,GemmLaws.ordered_update(order,accumulators,lane_tile,row_tile)) ==
    (cell(row,lane,accumulators) + GemmLaws.ordered_times(order,TileLaws.select_lane(lane,lane_tile),TileLaws.select_lane(row,row_tile)) : F32) : F32}:
  match order:
    case GemmLaws.LaneFirst{}: updated_cell(~Gemm.lane_times,row,lane,accumulators,lane_tile,row_tile,small)
    case GemmLaws.RowFirst{}: updated_cell(~Gemm.row_times,row,lane,accumulators,lane_tile,row_tile,small)

# The dot over no taps is its initial value; over a tap and more it continues
# from the initial value and the tap's product.
def dot_nil(+order: GemmLaws.Order,+row: Nat,+lane: Nat,+source: Source,+initial: F32) -> {dot(order,[],row,lane,source,initial) == initial : F32}:
  match order:
    case GemmLaws.LaneFirst{}: {==}
    case GemmLaws.RowFirst{}: {==}

def dot_cons(+order: GemmLaws.Order,+tap: ReductionLaws.AddressPair,+rest: List<&2,ReductionLaws.AddressPair>,+row: Nat,+lane: Nat,+source: Source,+initial: F32) ->
  {dot(order,tap <> rest,row,lane,source,initial) ==
    dot(order,rest,row,lane,source,(initial + GemmLaws.ordered_times(order,lane_element(source,lane,tap),row_element(source,row,tap)) : F32)) : F32}:
  match order:
    case GemmLaws.LaneFirst{}: {==}
    case GemmLaws.RowFirst{}: {==}

# The k loop: a cell continues its ordered dot over the remaining taps.
def reduced_cell_from(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,+count: Nat,+row_tiles: Tree.Tree<Tile.Tile8>,+lane_tiles: Tree.Tree<Tile.Tile8>,
  +accumulators: Gemm.Accumulators,+row: Nat,+lane: Nat,+small: {Nat.is_lt(row,8n) == True{} : Bool},read: Gemm.Step<Cursor>) ->
  {cell(row,lane,GemmLaws.reduced_accumulators(~Cursor,GemmLaws.reduce(~Cursor,~advance,~order,count,row_tiles,lane_tiles,
      GemmLaws.reduced_with(~Cursor,~order,row_tiles,lane_tiles,accumulators,read)))) ==
    dot(order,taps_from(~Cursor,~advance,1n+count,read),row,lane,Source{row_tiles,lane_tiles},cell(row,lane,accumulators)) : F32}:
  match count:
    case 0n:
      Gemm.Step{+lane_address,+row_address,+cursor} = read
      +samples = Tree.read(Tile.Tile8,lane_tiles,lane_address)
      +column = Tree.read(Tile.Tile8,row_tiles,row_address)
      +source = {Source{row_tiles,lane_tiles} : Source}
      +tap = {ReductionLaws.AddressPair{lane_address,row_address} : ReductionLaws.AddressPair}
      +start = cell(row,lane,accumulators)
      +after = (start + GemmLaws.ordered_times(order,TileLaws.select_lane(lane,samples),TileLaws.select_lane(row,column)) : F32)
      Equal.trans(F32,cell(row,lane,GemmLaws.ordered_update(order,accumulators,samples,column)),after,dot(order,tap <> [],row,lane,source,start),
        ordered_cell(order,row,lane,accumulators,samples,column,small),
        Equal.sym(F32,dot(order,tap <> [],row,lane,source,start),after,
          Equal.trans(F32,dot(order,tap <> [],row,lane,source,start),dot(order,[],row,lane,source,after),after,
            dot_cons(order,tap,[],row,lane,source,start),dot_nil(order,row,lane,source,after))))
    case 1n+ +rest:
      Gemm.Step{+lane_address,+row_address,+cursor} = read
      +samples = Tree.read(Tile.Tile8,lane_tiles,lane_address)
      +column = Tree.read(Tile.Tile8,row_tiles,row_address)
      +source = {Source{row_tiles,lane_tiles} : Source}
      +tap = {ReductionLaws.AddressPair{lane_address,row_address} : ReductionLaws.AddressPair}
      +start = cell(row,lane,accumulators)
      +next = GemmLaws.ordered_update(order,accumulators,samples,column)
      +after = (start + GemmLaws.ordered_times(order,TileLaws.select_lane(lane,samples),TileLaws.select_lane(row,column)) : F32)
      +remaining = taps_from(~Cursor,~advance,1n+rest,advance(cursor))
      Equal.trans(F32,cell(row,lane,GemmLaws.reduced_accumulators(~Cursor,GemmLaws.reduce(~Cursor,~advance,~order,1n+rest,row_tiles,lane_tiles,GemmLaws.Reduced{cursor,next}))),
        dot(order,remaining,row,lane,source,cell(row,lane,next)),dot(order,tap <> remaining,row,lane,source,start),
        reduced_cell_from(~Cursor,~advance,~order,rest,row_tiles,lane_tiles,next,row,lane,small,advance(cursor)),
        Equal.trans(F32,dot(order,remaining,row,lane,source,cell(row,lane,next)),dot(order,remaining,row,lane,source,after),dot(order,tap <> remaining,row,lane,source,start),
          Equal.cong(F32,F32,value => dot(order,remaining,row,lane,source,value),cell(row,lane,next),after,
            ordered_cell(order,row,lane,accumulators,samples,column,small)),
          Equal.sym(F32,dot(order,tap <> remaining,row,lane,source,start),dot(order,remaining,row,lane,source,after),
            dot_cons(order,tap,remaining,row,lane,source,start))))

def reduced_cell(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,+count: Nat,+row_tiles: Tree.Tree<Tile.Tile8>,+lane_tiles: Tree.Tree<Tile.Tile8>,
  +cursor: Cursor,+accumulators: Gemm.Accumulators,+row: Nat,+lane: Nat,+small: {Nat.is_lt(row,8n) == True{} : Bool}) ->
  {cell(row,lane,GemmLaws.reduced_accumulators(~Cursor,GemmLaws.reduce(~Cursor,~advance,~order,count,row_tiles,lane_tiles,GemmLaws.Reduced{cursor,accumulators}))) ==
    dot(order,taps(~Cursor,~advance,count,cursor),row,lane,Source{row_tiles,lane_tiles},cell(row,lane,accumulators)) : F32}:
  match count:
    case 0n: Equal.sym(F32,dot(order,[],row,lane,Source{row_tiles,lane_tiles},cell(row,lane,accumulators)),cell(row,lane,accumulators),
      dot_nil(order,row,lane,Source{row_tiles,lane_tiles},cell(row,lane,accumulators)))
    case 1n+ +rest: reduced_cell_from(~Cursor,~advance,~order,rest,row_tiles,lane_tiles,accumulators,row,lane,small,advance(cursor))

# The cells of the zero accumulators are +0.
def row_zero(+row: Nat) -> {select_row(row,Gemm.zero()) == Tile.filled(0.0) : Tile.Tile8}:
  match row:
    case 0n: {==}
    case 1n+ +a:
      match a:
        case 0n: {==}
        case 1n+ +b:
          match b:
            case 0n: {==}
            case 1n+ +c:
              match c:
                case 0n: {==}
                case 1n+ +d:
                  match d:
                    case 0n: {==}
                    case 1n+ +e:
                      match e:
                        case 0n: {==}
                        case 1n+ +f:
                          match f:
                            case 0n: {==}
                            case 1n+ +g: {==}

def cell_zero(+row: Nat,+lane: Nat) -> {cell(row,lane,Gemm.zero()) == 0.0 : F32}:
  %Equal.sym(Tile.Tile8,select_row(row,Gemm.zero()),Tile.filled(0.0),row_zero(row)) : {TileLaws.select_lane(lane,_) == 0.0 : F32}
  TileLaws.filled_lane(lane,0.0)

# A row of dots from +0: lane l is the dot of cell (row, l).
def dot_row(+order: GemmLaws.Order,+taps: List<&2,ReductionLaws.AddressPair>,+row: Nat,+source: Source) -> Tile.Tile8:
  Tile.Tile{dot(order,taps,row,0n,source,0.0),dot(order,taps,row,1n,source,0.0),dot(order,taps,row,2n,source,0.0),dot(order,taps,row,3n,source,0.0),
    dot(order,taps,row,4n,source,0.0),dot(order,taps,row,5n,source,0.0),dot(order,taps,row,6n,source,0.0),dot(order,taps,row,7n,source,0.0)}

def dot_lane(+order: GemmLaws.Order,+taps: List<&2,ReductionLaws.AddressPair>,+row: Nat,+lane: Nat,+source: Source,+small: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,dot_row(order,taps,row,source)) == dot(order,taps,row,lane,source,0.0) : F32}:
  match lane:
    case 0n: {==}
    case 1n+ +a:
      match a:
        case 0n: {==}
        case 1n+ +b:
          match b:
            case 0n: {==}
            case 1n+ +c:
              match c:
                case 0n: {==}
                case 1n+ +d:
                  match d:
                    case 0n: {==}
                    case 1n+ +e:
                      match e:
                        case 0n: {==}
                        case 1n+ +f:
                          match f:
                            case 0n: {==}
                            case 1n+ +g:
                              match g:
                                case 0n: {==}
                                case 1n+ +h:
                                  Empty.absurd({TileLaws.select_lane(8n+h,dot_row(order,taps,row,source)) == dot(order,taps,row,8n+h,source,0.0) : F32},
                                    WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(h,0n),True{},Equal.sym(Bool,Nat.is_lt(h,0n),False{},Division.lt_zero(h)),small)))

# A lane of a row after the k loop from +0 is its dot.
def reduced_lane(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,+count: Nat,+row_tiles: Tree.Tree<Tile.Tile8>,+lane_tiles: Tree.Tree<Tile.Tile8>,
  +cursor: Cursor,+row: Nat,+lane: Nat,+small: {Nat.is_lt(row,8n) == True{} : Bool}) ->
  {cell(row,lane,GemmLaws.reduced_accumulators(~Cursor,GemmLaws.reduce(~Cursor,~advance,~order,count,row_tiles,lane_tiles,GemmLaws.Reduced{cursor,Gemm.zero()}))) ==
    dot(order,taps(~Cursor,~advance,count,cursor),row,lane,Source{row_tiles,lane_tiles},0.0) : F32}:
  +steps = taps(~Cursor,~advance,count,cursor)
  Equal.trans(F32,cell(row,lane,GemmLaws.reduced_accumulators(~Cursor,GemmLaws.reduce(~Cursor,~advance,~order,count,row_tiles,lane_tiles,GemmLaws.Reduced{cursor,Gemm.zero()}))),
    dot(order,steps,row,lane,Source{row_tiles,lane_tiles},cell(row,lane,Gemm.zero())),dot(order,steps,row,lane,Source{row_tiles,lane_tiles},0.0),
    reduced_cell(~Cursor,~advance,~order,count,row_tiles,lane_tiles,cursor,Gemm.zero(),row,lane,small),
    Equal.cong(F32,F32,initial => dot(order,steps,row,lane,Source{row_tiles,lane_tiles},initial),cell(row,lane,Gemm.zero()),0.0,cell_zero(row,lane)))

# A row after the k loop from +0 is its row of dots.
def reduced_row(~Cursor: Data,~advance: Cursor -> Gemm.Step<Cursor>,~order: GemmLaws.Order,+count: Nat,+row_tiles: Tree.Tree<Tile.Tile8>,+lane_tiles: Tree.Tree<Tile.Tile8>,
  +cursor: Cursor,+row: Nat,+small: {Nat.is_lt(row,8n) == True{} : Bool}) ->
  {select_row(row,GemmLaws.reduced_accumulators(~Cursor,GemmLaws.reduce(~Cursor,~advance,~order,count,row_tiles,lane_tiles,GemmLaws.Reduced{cursor,Gemm.zero()}))) ==
    dot_row(order,taps(~Cursor,~advance,count,cursor),row,Source{row_tiles,lane_tiles}) : Tile.Tile8}:
  +values = select_row(row,GemmLaws.reduced_accumulators(~Cursor,GemmLaws.reduce(~Cursor,~advance,~order,count,row_tiles,lane_tiles,GemmLaws.Reduced{cursor,Gemm.zero()})))
  +steps = taps(~Cursor,~advance,count,cursor)
  +source = {Source{row_tiles,lane_tiles} : Source}
  -d0 = dot(order,steps,row,0n,source,0.0)
  -d1 = dot(order,steps,row,1n,source,0.0)
  -d2 = dot(order,steps,row,2n,source,0.0)
  -d3 = dot(order,steps,row,3n,source,0.0)
  -d4 = dot(order,steps,row,4n,source,0.0)
  -d5 = dot(order,steps,row,5n,source,0.0)
  -d6 = dot(order,steps,row,6n,source,0.0)
  -d7 = dot(order,steps,row,7n,source,0.0)
  Equal.trans(Tile.Tile8,values,Tile.Tile{Tile.lane0(values),Tile.lane1(values),Tile.lane2(values),Tile.lane3(values),Tile.lane4(values),Tile.lane5(values),Tile.lane6(values),
      Tile.lane7(values)},dot_row(order,steps,row,source),TileLaws.tile_eta(values),
    %Equal.sym(F32,Tile.lane0(values),d0,reduced_lane(~Cursor,~advance,~order,count,row_tiles,lane_tiles,cursor,row,0n,small)) :
      {Tile.Tile{_,Tile.lane1(values),Tile.lane2(values),Tile.lane3(values),Tile.lane4(values),Tile.lane5(values),Tile.lane6(values),Tile.lane7(values)} ==
        dot_row(order,steps,row,source) : Tile.Tile8}
    %Equal.sym(F32,Tile.lane1(values),d1,reduced_lane(~Cursor,~advance,~order,count,row_tiles,lane_tiles,cursor,row,1n,small)) :
      {Tile.Tile{d0,_,Tile.lane2(values),Tile.lane3(values),Tile.lane4(values),Tile.lane5(values),Tile.lane6(values),Tile.lane7(values)} == dot_row(order,steps,row,source) : Tile.Tile8}
    %Equal.sym(F32,Tile.lane2(values),d2,reduced_lane(~Cursor,~advance,~order,count,row_tiles,lane_tiles,cursor,row,2n,small)) :
      {Tile.Tile{d0,d1,_,Tile.lane3(values),Tile.lane4(values),Tile.lane5(values),Tile.lane6(values),Tile.lane7(values)} == dot_row(order,steps,row,source) : Tile.Tile8}
    %Equal.sym(F32,Tile.lane3(values),d3,reduced_lane(~Cursor,~advance,~order,count,row_tiles,lane_tiles,cursor,row,3n,small)) :
      {Tile.Tile{d0,d1,d2,_,Tile.lane4(values),Tile.lane5(values),Tile.lane6(values),Tile.lane7(values)} == dot_row(order,steps,row,source) : Tile.Tile8}
    %Equal.sym(F32,Tile.lane4(values),d4,reduced_lane(~Cursor,~advance,~order,count,row_tiles,lane_tiles,cursor,row,4n,small)) :
      {Tile.Tile{d0,d1,d2,d3,_,Tile.lane5(values),Tile.lane6(values),Tile.lane7(values)} == dot_row(order,steps,row,source) : Tile.Tile8}
    %Equal.sym(F32,Tile.lane5(values),d5,reduced_lane(~Cursor,~advance,~order,count,row_tiles,lane_tiles,cursor,row,5n,small)) :
      {Tile.Tile{d0,d1,d2,d3,d4,_,Tile.lane6(values),Tile.lane7(values)} == dot_row(order,steps,row,source) : Tile.Tile8}
    %Equal.sym(F32,Tile.lane6(values),d6,reduced_lane(~Cursor,~advance,~order,count,row_tiles,lane_tiles,cursor,row,6n,small)) :
      {Tile.Tile{d0,d1,d2,d3,d4,d5,_,Tile.lane7(values)} == dot_row(order,steps,row,source) : Tile.Tile8}
    %Equal.sym(F32,Tile.lane7(values),d7,reduced_lane(~Cursor,~advance,~order,count,row_tiles,lane_tiles,cursor,row,7n,small)) :
      {Tile.Tile{d0,d1,d2,d3,d4,d5,d6,_} == dot_row(order,steps,row,source) : Tile.Tile8}
    {==})

# The sample a GEMM step reads from its input and its weights at a key.
def gemm_input() -> Gemm.Both<Gemm.Both<Nat,Grid>,Source> -> Nat -> Nat -> F32:
  Ordered.mapped_sample(~Nat,~ReductionLaws.AddressPair,~Nat,~Source,~Gemm.Both<Nat,Grid>,
    ~both_key(~Nat,~Nat,~Grid,~offset_word,~grid_read),~lane_element)

def gemm_weight() -> Gemm.Both<Gemm.Both<Nat,Grid>,Source> -> Nat -> Nat -> F32:
  Ordered.mapped_sample(~Nat,~ReductionLaws.AddressPair,~Nat,~Source,~Gemm.Both<Nat,Grid>,
    ~both_key(~Nat,~Nat,~Grid,~offset_word,~grid_read),~row_element)
