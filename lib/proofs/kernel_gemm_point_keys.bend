# The taps of a pointwise job named by the lane they read: lane `lane` of
# cell group `cell` is step 8 cell + lane, at address address + 8 cell + lane
# and at cell first + cell.
import Base
import ../kernel_gemm.bend as Gemm
import ../ordered_matrix.bend as OrderedMatrix
import ./kernel_gemm_point_spec.bend as PointSpec
import ./ordered_reduction_split.bend as Ordered
import ./nat_to_u32_bounds.bend as Bounds
import ./word_arithmetic.bend as WordArithmetic
import ./natural_addition.bend as Addition
import ./nat_order.bend as Order
import ./sequence_membership.bend as Members
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_multiplication.bend as Multiplication
import ./word_encoding.bend as WordEncoding
import ./kernel_tile_refinement.bend as TileLaws
import ./storage_tree_model.bend as Tree
import ../kernel_tile.bend as Tile
import ./kernel_gemm_point_refinement.bend as PointLaws
import ./kernel_gemm_point_jobs.bend as PointJobs
import ./kernel_gemm_point_readback.bend as PointReadback

type Key is Data:
  LaneKey{cell: Nat,lane: Nat}

# The address and the cell of the first tap of a job.
type Origin is Data:
  TapOrigin{address: Nat,first: Nat}

def key_tap(origin: Origin,key: Key) -> PointSpec.Tap:
  TapOrigin{address,first} = origin
  LaneKey{+cell,+lane} = key
  PointSpec.Tap{U32.from_nat(Nat.add(lane,Nat.add(Nat.mul(cell,8n),address))),U32.from_nat(Nat.add(cell,first)),lane}

# The lanes from lane on of a cell group.
def lane_keys(count: Nat,+lane: Nat,+cell: Nat) -> List<&2,Key>:
  match count:
    case 0n: []
    case 1n+rest: LaneKey{cell,lane} <> lane_keys(rest,1n+lane,cell)

# The keys of count groups of sixteen from cell group cell on.
def keys(count: Nat,+cell: Nat) -> List<&2,Key>:
  match count:
    case 0n: []
    case 1n+rest: OrderedMatrix.cat(Key,OrderedMatrix.cat(Key,lane_keys(8n,0n,cell),lane_keys(8n,0n,1n+cell)),keys(rest,1n+1n+cell))

def keyed_taps(+count: Nat,+cell: Nat,+origin: Origin) -> List<&2,PointSpec.Tap>:
  Ordered.map_keys(~Key,~PointSpec.Tap,~Origin,~key_tap,keys(count,cell),origin)

# ---- Half a group ----

# count taps of one cell group from lane lane on.
def half_taps(~at: U32 -> U32,count: Nat,+lane: Nat,+cursor: Gemm.Points) -> List<&2,PointSpec.Tap>:
  match count:
    case 0n: []
    case 1n+rest: PointSpec.tap_at(~at,lane,cursor) <> half_taps(~at,rest,1n+lane,PointSpec.advanced(cursor))

def advanced_by(count: Nat,cursor: Gemm.Points) -> Gemm.Points:
  match count:
    case 0n: cursor
    case 1n+rest: advanced_by(rest,PointSpec.advanced(cursor))

def advanced_word(count: Nat,+position: Nat,+cell: U32,+stride: U32) ->
  {advanced_by(count,Gemm.Points{U32.from_nat(position),cell,stride}) == Gemm.Points{U32.from_nat(Nat.add(count,position)),cell,stride} : Gemm.Points}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.increment_encoded(position)) :
        {advanced_by(rest,Gemm.Points{_,cell,stride}) == Gemm.Points{U32.from_nat(1n+Nat.add(rest,position)),cell,stride} : Gemm.Points}
      %Addition.successor_right(rest,position) :
        {advanced_by(rest,Gemm.Points{U32.from_nat(1n+position),cell,stride}) == Gemm.Points{U32.from_nat(_),cell,stride} : Gemm.Points}
      advanced_word(rest,1n+position,cell,stride)

def half_keyed(~at: U32 -> U32,count: Nat,+lane: Nat,+cell: Nat,+address: Nat,+first: Nat,+word: U32,+stride: U32,
  +word_exact: {at(word) == U32.from_nat(Nat.add(cell,first)) : U32}) ->
  {half_taps(~at,count,lane,Gemm.Points{U32.from_nat(Nat.add(lane,Nat.add(Nat.mul(cell,8n),address))),word,stride}) ==
    Ordered.map_keys(~Key,~PointSpec.Tap,~Origin,~key_tap,lane_keys(count,lane,cell),TapOrigin{address,first}) : List<&2,PointSpec.Tap>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +position = Nat.add(lane,Nat.add(Nat.mul(cell,8n),address))
      +later = Ordered.map_keys(~Key,~PointSpec.Tap,~Origin,~key_tap,lane_keys(rest,1n+lane,cell),TapOrigin{address,first})
      +tap = {PointSpec.Tap{U32.from_nat(position),U32.from_nat(Nat.add(cell,first)),lane} : PointSpec.Tap}
      %Equal.sym(U32,(U32.from_nat(position) + 1 : U32),U32.from_nat(1n+position),Bounds.increment_encoded(position)) :
        {PointSpec.Tap{U32.from_nat(position),at(word),lane} <> half_taps(~at,rest,1n+lane,Gemm.Points{_,word,stride}) == tap <> later : List<&2,PointSpec.Tap>}
      %Equal.sym(U32,at(word),U32.from_nat(Nat.add(cell,first)),word_exact) :
        {PointSpec.Tap{U32.from_nat(position),_,lane} <> half_taps(~at,rest,1n+lane,Gemm.Points{U32.from_nat(1n+position),word,stride}) == tap <> later : List<&2,PointSpec.Tap>}
      Equal.cong(List<&2,PointSpec.Tap>,List<&2,PointSpec.Tap>,tail => tap <> tail,
        half_taps(~at,rest,1n+lane,Gemm.Points{U32.from_nat(1n+position),word,stride}),later,
        half_keyed(~at,rest,1n+lane,cell,address,first,word,stride,word_exact))

# ---- A group and the taps of a job ----

def add_two(+at: Nat) -> {(U32.from_nat(at) + 2 : U32) == U32.from_nat(1n+1n+at) : U32}:
  %Order.commutative(at,2n) : {(U32.from_nat(at) + 2 : U32) == U32.from_nat(_) : U32}
  WordArithmetic.native_add(at,2n)

# A group is the taps of two cell groups.
def group_keyed(+cell: Nat,+address: Nat,+first: Nat,+stride: U32) ->
  {PointSpec.group_taps(Gemm.Points{U32.from_nat(Nat.add(Nat.mul(cell,8n),address)),U32.from_nat(Nat.add(cell,first)),stride}) ==
    OrderedMatrix.cat(PointSpec.Tap,Ordered.map_keys(~Key,~PointSpec.Tap,~Origin,~key_tap,lane_keys(8n,0n,cell),TapOrigin{address,first}),
      Ordered.map_keys(~Key,~PointSpec.Tap,~Origin,~key_tap,lane_keys(8n,0n,1n+cell),TapOrigin{address,first})) : List<&2,PointSpec.Tap>}:
  +start = Nat.add(Nat.mul(cell,8n),address)
  +at = Nat.add(cell,first)
  +cursor = {Gemm.Points{U32.from_nat(start),U32.from_nat(at),stride} : Gemm.Points}
  -lower = Ordered.map_keys(~Key,~PointSpec.Tap,~Origin,~key_tap,lane_keys(8n,0n,cell),TapOrigin{address,first})
  -upper = Ordered.map_keys(~Key,~PointSpec.Tap,~Origin,~key_tap,lane_keys(8n,0n,1n+cell),TapOrigin{address,first})
  %Equal.sym(Gemm.Points,advanced_by(8n,cursor),Gemm.Points{U32.from_nat(Nat.add(8n,start)),U32.from_nat(at),stride},advanced_word(8n,start,U32.from_nat(at),stride)) :
    {OrderedMatrix.cat(PointSpec.Tap,half_taps(~Gemm.first_cell,8n,0n,cursor),half_taps(~Gemm.second_cell,8n,0n,_)) == OrderedMatrix.cat(PointSpec.Tap,lower,upper) : List<&2,PointSpec.Tap>}
  %Equal.sym(List<&2,PointSpec.Tap>,half_taps(~Gemm.second_cell,8n,0n,Gemm.Points{U32.from_nat(Nat.add(8n,start)),U32.from_nat(at),stride}),upper,
    half_keyed(~Gemm.second_cell,8n,0n,1n+cell,address,first,U32.from_nat(at),stride,Bounds.increment_encoded(at))) :
    {OrderedMatrix.cat(PointSpec.Tap,half_taps(~Gemm.first_cell,8n,0n,cursor),_) == OrderedMatrix.cat(PointSpec.Tap,lower,upper) : List<&2,PointSpec.Tap>}
  %Equal.sym(List<&2,PointSpec.Tap>,half_taps(~Gemm.first_cell,8n,0n,cursor),lower,half_keyed(~Gemm.first_cell,8n,0n,cell,address,first,U32.from_nat(at),stride,{==})) :
    {OrderedMatrix.cat(PointSpec.Tap,_,upper) == OrderedMatrix.cat(PointSpec.Tap,lower,upper) : List<&2,PointSpec.Tap>}
  {==}

# The cursor after a group: sixteen steps on, two cell groups on.
def group_cursor(+start: Nat,+at: Nat,+stride: U32) ->
  {PointSpec.group_after(Gemm.Points{U32.from_nat(start),U32.from_nat(at),stride}) == Gemm.Points{U32.from_nat(Nat.add(16n,start)),U32.from_nat(1n+1n+at),stride} : Gemm.Points}:
  %Equal.sym(Gemm.Points,advanced_by(16n,Gemm.Points{U32.from_nat(start),U32.from_nat(at),stride}),Gemm.Points{U32.from_nat(Nat.add(16n,start)),U32.from_nat(at),stride},
    advanced_word(16n,start,U32.from_nat(at),stride)) :
    {PointSpec.cells_after(_) == Gemm.Points{U32.from_nat(Nat.add(16n,start)),U32.from_nat(1n+1n+at),stride} : Gemm.Points}
  Equal.cong(U32,Gemm.Points,word => Gemm.Points{U32.from_nat(Nat.add(16n,start)),word,stride},(U32.from_nat(at) + 2 : U32),U32.from_nat(1n+1n+at),add_two(at))

# The taps of count groups are the keyed taps of their cell groups.
def taps_keyed(count: Nat,+cell: Nat,+address: Nat,+first: Nat,+stride: U32) ->
  {PointSpec.taps(count,Gemm.Points{U32.from_nat(Nat.add(Nat.mul(cell,8n),address)),U32.from_nat(Nat.add(cell,first)),stride}) ==
    keyed_taps(count,cell,TapOrigin{address,first}) : List<&2,PointSpec.Tap>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +start = Nat.add(Nat.mul(cell,8n),address)
      +at = Nat.add(cell,first)
      +cursor = {Gemm.Points{U32.from_nat(start),U32.from_nat(at),stride} : Gemm.Points}
      +next = {Gemm.Points{U32.from_nat(Nat.add(16n,start)),U32.from_nat(1n+1n+at),stride} : Gemm.Points}
      +later = keyed_taps(rest,1n+1n+cell,TapOrigin{address,first})
      +lower = Ordered.map_keys(~Key,~PointSpec.Tap,~Origin,~key_tap,lane_keys(8n,0n,cell),TapOrigin{address,first})
      +upper = Ordered.map_keys(~Key,~PointSpec.Tap,~Origin,~key_tap,lane_keys(8n,0n,1n+cell),TapOrigin{address,first})
      -wanted = OrderedMatrix.cat(PointSpec.Tap,OrderedMatrix.cat(PointSpec.Tap,lower,upper),later)
      %Equal.sym(Gemm.Points,PointSpec.group_after(cursor),next,group_cursor(start,at,stride)) :
        {OrderedMatrix.cat(PointSpec.Tap,PointSpec.group_taps(cursor),PointSpec.taps(rest,_)) == wanted : List<&2,PointSpec.Tap>}
      %Equal.sym(List<&2,PointSpec.Tap>,PointSpec.taps(rest,next),later,taps_keyed(rest,1n+1n+cell,address,first,stride)) :
        {OrderedMatrix.cat(PointSpec.Tap,PointSpec.group_taps(cursor),_) == wanted : List<&2,PointSpec.Tap>}
      Equal.cong(List<&2,PointSpec.Tap>,List<&2,PointSpec.Tap>,group => OrderedMatrix.cat(PointSpec.Tap,group,later),
        PointSpec.group_taps(cursor),OrderedMatrix.cat(PointSpec.Tap,lower,upper),group_keyed(cell,address,first,stride))

# ---- The dot of a job over its keys ----

# The channel a key reads: lane lane of cell group cell.
def key_channel(key: Key) -> Nat:
  LaneKey{cell,lane} = key
  Nat.add(lane,Nat.mul(cell,8n))

def key_input() -> Gemm.Both<Origin,PointSpec.Sampled> -> Nat -> Key -> F32:
  Ordered.mapped_sample(~Key,~PointSpec.Tap,~Nat,~PointSpec.Sampled,~Origin,~key_tap,~PointSpec.input_sample)

def key_weight() -> Gemm.Both<Origin,PointSpec.Sampled> -> Nat -> Key -> F32:
  Ordered.mapped_sample(~Key,~PointSpec.Tap,~Nat,~PointSpec.Sampled,~Origin,~key_tap,~PointSpec.weight_sample)

# The ordered dot of the taps of count groups from a job's first tap is the
# dot over their keys, in the same order.
def keyed_dot(+count: Nat,+address: Nat,+first: Nat,+stride: U32,+row: Nat,+lane: Nat,+sampled: PointSpec.Sampled,+initial: F32) ->
  {PointSpec.dot(PointSpec.taps(count,Gemm.Points{U32.from_nat(address),U32.from_nat(first),stride}),row,lane,sampled,initial) ==
    OrderedMatrix.dot(~Nat,~Key,~Nat,~Gemm.Both<Origin,PointSpec.Sampled>,~key_input(),~key_weight(),keys(count,0n),lane,row,Gemm.Both{TapOrigin{address,first},sampled},initial) : F32}:
  %Equal.sym(List<&2,PointSpec.Tap>,PointSpec.taps(count,Gemm.Points{U32.from_nat(address),U32.from_nat(first),stride}),keyed_taps(count,0n,TapOrigin{address,first}),
    taps_keyed(count,0n,address,first,stride)) :
    {PointSpec.dot(_,row,lane,sampled,initial) ==
      OrderedMatrix.dot(~Nat,~Key,~Nat,~Gemm.Both<Origin,PointSpec.Sampled>,~key_input(),~key_weight(),keys(count,0n),lane,row,Gemm.Both{TapOrigin{address,first},sampled},initial) : F32}
  Ordered.dot_reindexed(~Key,~PointSpec.Tap,~Nat,~Nat,~PointSpec.Sampled,~Origin,~key_tap,~PointSpec.input_sample,~PointSpec.weight_sample,
    keys(count,0n),TapOrigin{address,first},lane,row,sampled,initial)

# ---- Evidence for every key ----

def lane_all(~Context: Data,~Property: @+context: Context -> @+key: Key -> Data,~cells: Context -> Nat,
  ~rule: @+context: Context -> @+cell: Nat -> @+lane: Nat -> @+cell_inside: {Nat.is_lt(cell,cells(context)) == True{} : Bool} ->
    @+lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool} -> Property(context,LaneKey{cell,lane}),
  count: Nat,+lane: Nat,+cell: Nat,+context: Context,+cell_inside: {Nat.is_lt(cell,cells(context)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(lane,count),8n) == True{} : Bool}) ->
  Members.All(Key,{key => Property(context,key) : Key -> Data},lane_keys(count,lane,cell)):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      Gemm.Both{rule(context,cell,lane,cell_inside,Order.room_index(lane,rest,8n,room)),
        lane_all(~Context,~Property,~cells,~rule,rest,1n+lane,cell,context,cell_inside,Order.advance_room(lane,rest,8n,room))}

# Evidence for each key of count groups from cell group cell on.
def keys_all(~Context: Data,~Property: @+context: Context -> @+key: Key -> Data,~cells: Context -> Nat,
  ~rule: @+context: Context -> @+cell: Nat -> @+lane: Nat -> @+cell_inside: {Nat.is_lt(cell,cells(context)) == True{} : Bool} ->
    @+lane_inside: {Nat.is_lt(lane,8n) == True{} : Bool} -> Property(context,LaneKey{cell,lane}),
  count: Nat,+cell: Nat,+context: Context,+room: {Nat.is_le(Nat.add(cell,Nat.mul(count,2n)),cells(context)) == True{} : Bool}) ->
  Members.All(Key,{key => Property(context,key) : Key -> Data},keys(count,cell)):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      +size = cells(context)
      +twice = Nat.mul(rest,2n)
      +lower_room = Order.advance_room(cell,1n+twice,size,room)
      +lower = lane_all(~Context,~Property,~cells,~rule,8n,0n,cell,context,Order.room_index(cell,1n+twice,size,room),Order.reflexive(8n))
      +upper = lane_all(~Context,~Property,~cells,~rule,8n,0n,1n+cell,context,Order.room_index(1n+cell,twice,size,lower_room),Order.reflexive(8n))
      Members.all_concatenated(Key,{key => Property(context,key) : Key -> Data},OrderedMatrix.cat(Key,lane_keys(8n,0n,cell),lane_keys(8n,0n,1n+cell)),keys(rest,1n+1n+cell),
        Members.all_concatenated(Key,{key => Property(context,key) : Key -> Data},lane_keys(8n,0n,cell),lane_keys(8n,0n,1n+cell),lower,upper),
        keys_all(~Context,~Property,~cells,~rule,rest,1n+1n+cell,context,Order.advance_room(1n+cell,twice,size,lower_room)))

# ---- The words and samples of a job ----

def step_word(+start: Nat,+count: Nat,+stride: U32) ->
  {(U32.from_nat(start) + U32.from_nat(count) * stride : U32) == U32.from_nat(Nat.add(start,Nat.mul(count,U32.to_nat(stride)))) : U32}:
  %Arithmetic.native_roundtrip(stride) :
    {(U32.from_nat(start) + U32.from_nat(count) * _ : U32) == U32.from_nat(Nat.add(start,Nat.mul(count,U32.to_nat(stride)))) : U32}
  %Equal.sym(U32,(U32.from_nat(count) * U32.from_nat(U32.to_nat(stride)) : U32),U32.from_nat(Nat.mul(count,U32.to_nat(stride))),
    Multiplication.native_encoded_multiply(count,U32.to_nat(stride))) :
    {(U32.from_nat(start) + _ : U32) == U32.from_nat(Nat.add(start,Nat.mul(count,U32.to_nat(stride)))) : U32}
  WordArithmetic.native_add(start,Nat.mul(count,U32.to_nat(stride)))

# Row row of a job reads the lane of its key row strides past the key's cell.
def key_input_value(+address: Nat,+first: Nat,+cells: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+stride: U32,+row: Nat,+cell: Nat,+lane: Nat,
  +small: {Nat.is_le(row,3n) == True{} : Bool}) ->
  {key_input()(Gemm.Both{TapOrigin{address,first},PointSpec.Sampled{cells,pairs,stride}},row,LaneKey{cell,lane}) ==
    TileLaws.select_lane(lane,Tree.read(Tile.Tile8,cells,U32.from_nat(Nat.add(Nat.add(cell,first),Nat.mul(row,U32.to_nat(stride)))))) : F32}:
  match row:
    case 0n:
      +at = Nat.add(cell,first)
      %Equal.sym(Nat,Nat.add(at,0n),at,Addition.zero_right(at)) :
        {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,cells,U32.from_nat(at))) == TileLaws.select_lane(lane,Tree.read(Tile.Tile8,cells,U32.from_nat(_))) : F32}
      {==}
    case 1n+ +a:
      match a:
        case 0n:
          +at = Nat.add(cell,first)
          Equal.cong(U32,F32,word => TileLaws.select_lane(lane,Tree.read(Tile.Tile8,cells,word)),(U32.from_nat(at) + U32.from_nat(1n) * stride : U32),
            U32.from_nat(Nat.add(at,Nat.mul(1n,U32.to_nat(stride)))),step_word(at,1n,stride))
        case 1n+ +b:
          match b:
            case 0n:
              +at = Nat.add(cell,first)
              Equal.cong(U32,F32,word => TileLaws.select_lane(lane,Tree.read(Tile.Tile8,cells,word)),(U32.from_nat(at) + U32.from_nat(2n) * stride : U32),
                U32.from_nat(Nat.add(at,Nat.mul(2n,U32.to_nat(stride)))),step_word(at,2n,stride))
            case 1n+ +c:
              match c:
                case 0n:
                  +at = Nat.add(cell,first)
                  Equal.cong(U32,F32,word => TileLaws.select_lane(lane,Tree.read(Tile.Tile8,cells,word)),(U32.from_nat(at) + U32.from_nat(3n) * stride : U32),
                    U32.from_nat(Nat.add(at,Nat.mul(3n,U32.to_nat(stride)))),step_word(at,3n,stride))
                case 1n+d:
                  Empty.absurd({key_input()(Gemm.Both{TapOrigin{address,first},PointSpec.Sampled{cells,pairs,stride}},4n+d,LaneKey{cell,lane}) ==
                    TileLaws.select_lane(lane,Tree.read(Tile.Tile8,cells,U32.from_nat(Nat.add(Nat.add(cell,first),Nat.mul(4n+d,U32.to_nat(stride)))))) : F32},
                    WordEncoding.false_true(small))

# A lane of a sum of pairs is the sum of their lanes.
def half_plus(+low: Bool,+lane: Nat,+a: Tile.Tile8,+b: Tile.Tile8,+c: Tile.Tile8,+d: Tile.Tile8) ->
  {PointSpec.half_lane(low,lane,Tile.plus(a,c),Tile.plus(b,d)) == (PointSpec.half_lane(low,lane,a,b) + PointSpec.half_lane(low,lane,c,d) : F32) : F32}:
  match low:
    case True{}: TileLaws.plus_lane(a,c,lane)
    case False{}: TileLaws.plus_lane(b,d,Nat.sub(lane,8n))

def pair_plus(+left: Tile.Tile16,+right: Tile.Tile16,+lane: Nat) ->
  {PointSpec.pair_lane(lane,Tile.plus_pair(left,right)) == (PointSpec.pair_lane(lane,left) + PointSpec.pair_lane(lane,right) : F32) : F32}:
  Tile.Tiles{+a,+b} = left
  Tile.Tiles{+c,+d} = right
  half_plus(Nat.is_lt(lane,8n),lane,a,b,c,d)

# ---- The value a job stores ----

# The cell of the first row of a unit.
def job_first(+cell: U32,+unit: Nat,+stride: U32) -> Nat:
  Nat.add(U32.to_nat(cell),Nat.mul(Nat.mul(unit,4n),U32.to_nat(stride)))

def address_word(+pair: Nat,+inner: U32) -> {(U32.from_nat(pair) * inner : U32) == U32.from_nat(Nat.mul(pair,U32.to_nat(inner))) : U32}:
  %Arithmetic.native_roundtrip(inner) : {(U32.from_nat(pair) * _ : U32) == U32.from_nat(Nat.mul(pair,U32.to_nat(inner))) : U32}
  Multiplication.native_encoded_multiply(pair,U32.to_nat(inner))

def first_word(+cell: U32,+unit: Nat,+stride: U32) -> {(cell + U32.from_nat(unit) * 4 * stride : U32) == U32.from_nat(job_first(cell,unit,stride)) : U32}:
  %Arithmetic.native_roundtrip(cell) : {(_ + U32.from_nat(unit) * 4 * stride : U32) == U32.from_nat(job_first(cell,unit,stride)) : U32}
  %Equal.sym(U32,(U32.from_nat(unit) * 4 : U32),U32.from_nat(Nat.mul(unit,4n)),Multiplication.native_encoded_multiply(unit,4n)) :
    {(U32.from_nat(U32.to_nat(cell)) + _ * stride : U32) == U32.from_nat(job_first(cell,unit,stride)) : U32}
  step_word(U32.to_nat(cell),Nat.mul(unit,4n),stride)

# The dot of the job of a pair and a unit at a row and a lane, over its keys.
def job_dot(+channels: U32,+positions: U32,+inner: U32,+start: U32,+output_positions: U32,+cell: U32,+stride: U32,
  +cells: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+pair: Nat,+unit: Nat,+row: Nat,+lane: Nat) ->
  {PointSpec.cell_value(row,lane,PointJobs.job_accumulators(Gemm.PointMatrix{Gemm.TileMatrix{channels,positions,inner,start,output_positions},cell,stride},
      Gemm.Place{U32.from_nat(pair),U32.from_nat(unit)},cells,pairs)) ==
    OrderedMatrix.dot(~Nat,~Key,~Nat,~Gemm.Both<Origin,PointSpec.Sampled>,~key_input(),~key_weight(),keys(U32.to_nat((inner / 16 : U32)),0n),lane,row,
      Gemm.Both{TapOrigin{Nat.mul(pair,U32.to_nat(inner)),job_first(cell,unit,stride)},PointSpec.Sampled{cells,pairs,stride}},0.0) : F32}:
  +count = U32.to_nat((inner / 16 : U32))
  +address = Nat.mul(pair,U32.to_nat(inner))
  +first = job_first(cell,unit,stride)
  +sampled = {PointSpec.Sampled{cells,pairs,stride} : PointSpec.Sampled}
  +wanted = OrderedMatrix.dot(~Nat,~Key,~Nat,~Gemm.Both<Origin,PointSpec.Sampled>,~key_input(),~key_weight(),keys(count,0n),lane,row,Gemm.Both{TapOrigin{address,first},sampled},0.0)
  +reached = PointSpec.cell_value(row,lane,PointSpec.accumulators_of(PointLaws.groups(~Gemm.row_times,count,PointLaws.Sources{cells,pairs},
    PointLaws.Reduced{Gemm.Points{U32.from_nat(address),U32.from_nat(first),stride},Gemm.quad_zero()})))
  %Equal.sym(U32,(U32.from_nat(pair) * inner : U32),U32.from_nat(address),address_word(pair,inner)) :
    {PointSpec.cell_value(row,lane,PointSpec.accumulators_of(PointLaws.groups(~Gemm.row_times,count,PointLaws.Sources{cells,pairs},
      PointLaws.Reduced{Gemm.Points{_,(cell + U32.from_nat(unit) * 4 * stride : U32),stride},Gemm.quad_zero()}))) == wanted : F32}
  %Equal.sym(U32,(cell + U32.from_nat(unit) * 4 * stride : U32),U32.from_nat(first),first_word(cell,unit,stride)) :
    {PointSpec.cell_value(row,lane,PointSpec.accumulators_of(PointLaws.groups(~Gemm.row_times,count,PointLaws.Sources{cells,pairs},
      PointLaws.Reduced{Gemm.Points{U32.from_nat(address),_,stride},Gemm.quad_zero()}))) == wanted : F32}
  Equal.trans(F32,reached,PointSpec.dot(PointSpec.taps(count,Gemm.Points{U32.from_nat(address),U32.from_nat(first),stride}),row,lane,sampled,0.0),wanted,
    %PointSpec.zero_cell(row,lane) : {reached == PointSpec.dot(PointSpec.taps(count,Gemm.Points{U32.from_nat(address),U32.from_nat(first),stride}),row,lane,sampled,_) : F32}
    PointSpec.groups_cell(count,cells,pairs,U32.from_nat(address),U32.from_nat(first),stride,Gemm.quad_zero(),row,lane),
    keyed_dot(count,address,first,stride,row,lane,sampled,0.0))

# Lane lane of what the job of a pair and a unit stores at a row: the dot over
# its keys plus the lane of the biases of the pair.
def stored_lane(+channels: U32,+positions: U32,+inner: U32,+start: U32,+output_positions: U32,+cell: U32,+stride: U32,
  +bias: Tree.Tree<Tile.Tile8>,+cells: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+pair: Nat,+unit: Nat,+row: Nat,+lane: Nat) ->
  {PointSpec.pair_lane(lane,PointReadback.stored_value(PointReadback.Target{Gemm.PointMatrix{Gemm.TileMatrix{channels,positions,inner,start,output_positions},cell,stride},
      PointJobs.Readers{bias,cells,pairs}},U32.from_nat(pair),U32.from_nat(unit),row)) ==
    (OrderedMatrix.dot(~Nat,~Key,~Nat,~Gemm.Both<Origin,PointSpec.Sampled>,~key_input(),~key_weight(),keys(U32.to_nat((inner / 16 : U32)),0n),lane,row,
      Gemm.Both{TapOrigin{Nat.mul(pair,U32.to_nat(inner)),job_first(cell,unit,stride)},PointSpec.Sampled{cells,pairs,stride}},0.0) +
      PointSpec.pair_lane(lane,PointReadback.biases(bias,PointReadback.target_groups(PointReadback.Target{Gemm.PointMatrix{Gemm.TileMatrix{channels,positions,inner,start,output_positions},cell,stride},
        PointJobs.Readers{bias,cells,pairs}}),U32.from_nat(pair))) : F32) : F32}:
  +matrix = {Gemm.PointMatrix{Gemm.TileMatrix{channels,positions,inner,start,output_positions},cell,stride} : Gemm.PointMatrix}
  +accumulators = PointJobs.job_accumulators(matrix,Gemm.Place{U32.from_nat(pair),U32.from_nat(unit)},cells,pairs)
  +biases = PointReadback.biases(bias,PointReadback.target_groups(PointReadback.Target{matrix,PointJobs.Readers{bias,cells,pairs}}),U32.from_nat(pair))
  +dotted = OrderedMatrix.dot(~Nat,~Key,~Nat,~Gemm.Both<Origin,PointSpec.Sampled>,~key_input(),~key_weight(),keys(U32.to_nat((inner / 16 : U32)),0n),lane,row,
    Gemm.Both{TapOrigin{Nat.mul(pair,U32.to_nat(inner)),job_first(cell,unit,stride)},PointSpec.Sampled{cells,pairs,stride}},0.0)
  Equal.trans(F32,PointSpec.pair_lane(lane,Tile.plus_pair(PointSpec.row_of(row,accumulators),biases)),
    (PointSpec.cell_value(row,lane,accumulators) + PointSpec.pair_lane(lane,biases) : F32),(dotted + PointSpec.pair_lane(lane,biases) : F32),
    pair_plus(PointSpec.row_of(row,accumulators),biases,lane),
    Equal.cong(F32,F32,value => (value + PointSpec.pair_lane(lane,biases) : F32),PointSpec.cell_value(row,lane,accumulators),dotted,
      job_dot(channels,positions,inner,start,output_positions,cell,stride,cells,pairs,pair,unit,row,lane)))

# ---- Firsts of the same word ----

# Two firsts of the same word give each key the same tap.
def tap_same(+address: Nat,+first: Nat,+other: Nat,+same: {U32.from_nat(first) == U32.from_nat(other) : U32},+key: Key) ->
  {key_tap(TapOrigin{address,first},key) == key_tap(TapOrigin{address,other},key) : PointSpec.Tap}:
  LaneKey{+cell,+lane} = key
  +tap_address = U32.from_nat(Nat.add(lane,Nat.add(Nat.mul(cell,8n),address)))
  %WordArithmetic.native_add(cell,first) : {PointSpec.Tap{tap_address,_,lane} == PointSpec.Tap{tap_address,U32.from_nat(Nat.add(cell,other)),lane} : PointSpec.Tap}
  %WordArithmetic.native_add(cell,other) :
    {PointSpec.Tap{tap_address,(U32.from_nat(cell) + U32.from_nat(first) : U32),lane} == PointSpec.Tap{tap_address,_,lane} : PointSpec.Tap}
  %Equal.sym(U32,U32.from_nat(first),U32.from_nat(other),same) :
    {PointSpec.Tap{tap_address,(U32.from_nat(cell) + _ : U32),lane} == PointSpec.Tap{tap_address,(U32.from_nat(cell) + U32.from_nat(other) : U32),lane} : PointSpec.Tap}
  {==}

def origin_matching(keys: List<&2,Key>,+address: Nat,+first: Nat,+other: Nat,+same: {U32.from_nat(first) == U32.from_nat(other) : U32},
  +sampled: PointSpec.Sampled,+lane: Nat,+row: Nat) ->
  Ordered.MatchingSamples(~Key,~Nat,~Nat,~Gemm.Both<Origin,PointSpec.Sampled>,~Nat,~Nat,~Gemm.Both<Origin,PointSpec.Sampled>,~key_input(),~key_weight(),~key_input(),~key_weight(),
    keys,lane,row,Gemm.Both{TapOrigin{address,first},sampled},lane,row,Gemm.Both{TapOrigin{address,other},sampled}):
  match keys:
    case Nil{}: Unit{}
    case +key <> +rest:
      Gemm.Both{
        Gemm.Both{
          Equal.cong(PointSpec.Tap,F32,tap => PointSpec.input_sample(sampled,row,tap),key_tap(TapOrigin{address,first},key),key_tap(TapOrigin{address,other},key),
            tap_same(address,first,other,same,key)),
          Equal.cong(PointSpec.Tap,F32,tap => PointSpec.weight_sample(sampled,lane,tap),key_tap(TapOrigin{address,first},key),key_tap(TapOrigin{address,other},key),
            tap_same(address,first,other,same,key))},
        origin_matching(rest,address,first,other,same,sampled,lane,row)}

# The keyed dot of a first is the keyed dot of any first of the same word.
def origin_dot(+keys: List<&2,Key>,+address: Nat,+first: Nat,+other: Nat,+same: {U32.from_nat(first) == U32.from_nat(other) : U32},
  +sampled: PointSpec.Sampled,+lane: Nat,+row: Nat,+initial: F32) ->
  {OrderedMatrix.dot(~Nat,~Key,~Nat,~Gemm.Both<Origin,PointSpec.Sampled>,~key_input(),~key_weight(),keys,lane,row,Gemm.Both{TapOrigin{address,first},sampled},initial) ==
    OrderedMatrix.dot(~Nat,~Key,~Nat,~Gemm.Both<Origin,PointSpec.Sampled>,~key_input(),~key_weight(),keys,lane,row,Gemm.Both{TapOrigin{address,other},sampled},initial) : F32}:
  Ordered.dot_congruence(~Key,~Nat,~Nat,~Gemm.Both<Origin,PointSpec.Sampled>,~Nat,~Nat,~Gemm.Both<Origin,PointSpec.Sampled>,~key_input(),~key_weight(),~key_input(),~key_weight(),
    keys,lane,row,Gemm.Both{TapOrigin{address,first},sampled},lane,row,Gemm.Both{TapOrigin{address,other},sampled},initial,
    origin_matching(keys,address,first,other,same,sampled,lane,row))
