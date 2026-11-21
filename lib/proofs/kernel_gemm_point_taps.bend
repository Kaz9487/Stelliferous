# The ordered taps of either point-product kernel, including the short suffix.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ../machine_limits.bend as Machine
import ../ordered_matrix.bend as Matrix
import ./kernel_gemm_point_jobs.bend as Jobs
import ./kernel_gemm_point_keys.bend as Keys
import ./kernel_gemm_point_readback.bend as Readback
import ./kernel_gemm_point_refinement.bend as Laws
import ./kernel_gemm_point_spec.bend as Spec
import ./kernel_tile_refinement.bend as TileLaws
import ./machine_limit_bounds.bend as Limits
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./binary_radix.bend as Radix
import ./nat_extrema.bend as Extrema
import ./nat_division_order.bend as DivisionOrder
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./nat_order.bend as Order
import ./ordered_reduction_split.bend as Ordered
import ./ordered_ranges.bend as Ranges
import ./row_major_index.bend as Index
import ./sequence_membership.bend as Members
import ./storage_tree_model.bend as Tree
import ./u32_extrema.bend as WordExtrema
import ./u32_division.bend as U32Division
import ./word_arithmetic.bend as Words

def same_group(+sources: Laws.Sources,+state: Laws.Reduced) ->
  {Laws.grouped(~Gemm.panel_times,sources,state) == Laws.grouped(~Gemm.row_times,sources,state) : Laws.Reduced}:
  Laws.Sources{+input,+pairs} = sources
  Laws.Reduced{Gemm.Points{+address,+cell,+stride},+accumulators} = state
  {==}

def same_groups(count: Nat,+sources: Laws.Sources,+state: Laws.Reduced) ->
  {Laws.groups(~Gemm.panel_times,count,sources,state) == Laws.groups(~Gemm.row_times,count,sources,state) : Laws.Reduced}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +panel = Laws.grouped(~Gemm.panel_times,sources,state)
      +row = Laws.grouped(~Gemm.row_times,sources,state)
      Equal.trans(Laws.Reduced,Laws.groups(~Gemm.panel_times,rest,sources,panel),Laws.groups(~Gemm.row_times,rest,sources,panel),Laws.groups(~Gemm.row_times,rest,sources,row),
        same_groups(rest,sources,panel),Equal.cong(Laws.Reduced,Laws.Reduced,next => Laws.groups(~Gemm.row_times,rest,sources,next),panel,row,same_group(sources,state)))

def reached(+matrix: Gemm.PointMatrix,+place: Gemm.Place,+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>) -> Laws.Reduced:
  Laws.groups(~Gemm.row_times,Gemm.point_groups_of(matrix),Laws.Sources{input,pairs},Laws.Reduced{Spec.job_start(matrix,place),Gemm.quad_zero()})

def whole_taps(+matrix: Gemm.PointMatrix,+place: Gemm.Place) -> List<&2,Spec.Tap>:
  Spec.taps(Gemm.point_groups_of(matrix),Spec.job_start(matrix,place))

def group_stride(+sources: Laws.Sources,+state: Laws.Reduced) ->
  {Spec.stride_of(Laws.grouped(~Gemm.row_times,sources,state)) == Spec.stride_of(state) : U32}:
  Laws.Sources{+input,+pairs} = sources
  Laws.Reduced{Gemm.Points{+address,+cell,+stride},+accumulators} = state
  {==}

def groups_stride(count: Nat,+sources: Laws.Sources,+state: Laws.Reduced) ->
  {Spec.stride_of(Laws.groups(~Gemm.row_times,count,sources,state)) == Spec.stride_of(state) : U32}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +next = Laws.grouped(~Gemm.row_times,sources,state)
      Equal.trans(U32,Spec.stride_of(Laws.groups(~Gemm.row_times,rest,sources,next)),Spec.stride_of(next),Spec.stride_of(state),
        groups_stride(rest,sources,next),group_stride(sources,state))

def reached_stride(+matrix: Gemm.PointMatrix,+place: Gemm.Place,+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>) ->
  {Spec.stride_of(reached(matrix,place,input,pairs)) == Spec.matrix_stride(matrix) : U32}:
  Gemm.PointMatrix{Gemm.TileMatrix{+channels,+positions,+inner,+start,+end},+cell,+stride} = matrix
  Gemm.Place{+pair,+unit} = place
  groups_stride(Gemm.point_groups_of(matrix),Laws.Sources{input,pairs},Laws.Reduced{Spec.job_start(matrix,place),Gemm.quad_zero()})

def remaining_taps(+matrix: Gemm.PointMatrix,+place: Gemm.Place,+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>) -> List<&2,Spec.Tap>:
  Spec.remainder_taps(Gemm.point_remainder(matrix),Spec.cursor_of(reached(matrix,place,input,pairs)))

def whole_dot(+matrix: Gemm.PointMatrix,+place: Gemm.Place,+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+row: Nat,+lane: Nat) ->
  {Spec.cell_value(row,lane,Jobs.job_accumulators(matrix,place,input,pairs)) ==
    Spec.dot(whole_taps(matrix,place),row,lane,Spec.Sampled{input,pairs,Spec.matrix_stride(matrix)},0.0) : F32}:
  Gemm.PointMatrix{Gemm.TileMatrix{+channels,+positions,+inner,+start,+end},+cell,+stride} = matrix
  Gemm.Place{+pair,+unit} = place
  +address = (pair * inner : U32)
  +first = (cell + unit * 4 * stride : U32)
  +cursor = {Gemm.Points{address,first,stride} : Gemm.Points}
  +count = U32.to_nat((inner / 16 : U32))
  +result = Spec.cell_value(row,lane,Jobs.job_accumulators(matrix,place,input,pairs))
  %Spec.zero_cell(row,lane) : {result == Spec.dot(Spec.taps(count,cursor),row,lane,Spec.Sampled{input,pairs,stride},_) : F32}
  Spec.groups_cell(count,input,pairs,address,first,stride,Gemm.quad_zero(),row,lane)

def suffix_dot(+matrix: Gemm.PointMatrix,+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+row: Nat,+lane: Nat,+state: Laws.Reduced) ->
  {Spec.cell_value(row,lane,Spec.accumulators_of(Laws.tail_reduced(matrix,Laws.Sources{input,pairs},state))) ==
    Spec.dot(Spec.remainder_taps(Gemm.point_remainder(matrix),Spec.cursor_of(state)),row,lane,Spec.Sampled{input,pairs,Spec.stride_of(state)},
      Spec.cell_value(row,lane,Spec.accumulators_of(state))) : F32}:
  Laws.Reduced{Gemm.Points{+address,+cell,+stride},+accumulators} = state
  Spec.remainder_model_cell(Gemm.point_remainder(matrix),input,pairs,address,cell,stride,accumulators,row,lane)

def tail_dot(+matrix: Gemm.PointMatrix,+place: Gemm.Place,+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+row: Nat,+lane: Nat) ->
  {Spec.cell_value(row,lane,Jobs.job_accumulators_with(~Gemm.panel_times,~Laws.tail_reduced,matrix,place,input,pairs)) ==
    Spec.dot(Matrix.cat(Spec.Tap,whole_taps(matrix,place),remaining_taps(matrix,place,input,pairs)),row,lane,Spec.Sampled{input,pairs,Spec.matrix_stride(matrix)},0.0) : F32}:
  +sources = {Laws.Sources{input,pairs} : Laws.Sources}
  +start = {Laws.Reduced{Spec.job_start(matrix,place),Gemm.quad_zero()} : Laws.Reduced}
  +count = Gemm.point_groups_of(matrix)
  +finish = reached(matrix,place,input,pairs)
  +old = Laws.groups(~Gemm.panel_times,count,sources,start)
  +initial = whole_taps(matrix,place)
  +last = remaining_taps(matrix,place,input,pairs)
  +sample = {Spec.Sampled{input,pairs,Spec.matrix_stride(matrix)} : Spec.Sampled}
  +folded = Spec.cell_value(row,lane,Spec.accumulators_of(finish))
  +wanted = Spec.dot(Matrix.cat(Spec.Tap,initial,last),row,lane,sample,0.0)
  %Equal.sym(Laws.Reduced,old,finish,same_groups(count,sources,start)) :
    {Spec.cell_value(row,lane,Spec.accumulators_of(Laws.tail_reduced(matrix,sources,_))) == wanted : F32}
  Equal.trans(F32,Spec.cell_value(row,lane,Spec.accumulators_of(Laws.tail_reduced(matrix,sources,finish))),Spec.dot(last,row,lane,sample,folded),wanted,
    %reached_stride(matrix,place,input,pairs) : {Spec.cell_value(row,lane,Spec.accumulators_of(Laws.tail_reduced(matrix,sources,finish))) ==
      Spec.dot(last,row,lane,Spec.Sampled{input,pairs,_},folded) : F32}
    suffix_dot(matrix,input,pairs,row,lane,finish),
    Equal.trans(F32,Spec.dot(last,row,lane,sample,folded),Spec.dot(last,row,lane,sample,Spec.dot(initial,row,lane,sample,0.0)),wanted,
      Equal.cong(F32,F32,value => Spec.dot(last,row,lane,sample,value),folded,Spec.dot(initial,row,lane,sample,0.0),
        whole_dot(matrix,place,input,pairs,row,lane)),
      Equal.sym(F32,wanted,Spec.dot(last,row,lane,sample,Spec.dot(initial,row,lane,sample,0.0)),
        Ordered.reduction_split(~Nat,~Spec.Tap,~Nat,~Spec.Sampled,~Spec.input_sample,~Spec.weight_sample,initial,last,lane,row,sample,0.0))))

def job_whole_taps(+matrix: Gemm.PointMatrix,+place: Gemm.Place,+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>) -> List<&2,Spec.Tap>:
  whole_taps(matrix,place)

def job_tail_taps(+matrix: Gemm.PointMatrix,+place: Gemm.Place,+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>) -> List<&2,Spec.Tap>:
  Matrix.cat(Spec.Tap,whole_taps(matrix,place),remaining_taps(matrix,place,input,pairs))

def Taps() -> Type:
  @+matrix: Gemm.PointMatrix -> @+place: Gemm.Place -> @+input: Tree.Tree<Tile.Tile8> -> @+pairs: Tree.Tree<Tile.Tile16> -> List<&2,Spec.Tap>

def LaneReduction(~accumulate: Jobs.Accumulate(),~taps: Taps()) -> Type:
  @+matrix: Gemm.PointMatrix -> @+place: Gemm.Place -> @+input: Tree.Tree<Tile.Tile8> -> @+pairs: Tree.Tree<Tile.Tile16> -> @+row: Nat -> @+lane: Nat ->
    {Spec.cell_value(row,lane,accumulate(matrix,place,input,pairs)) ==
      Spec.dot(taps(matrix,place,input,pairs),row,lane,Spec.Sampled{input,pairs,Spec.matrix_stride(matrix)},0.0) : F32}

def input_of(target: Readback.Target) -> Tree.Tree<Tile.Tile8>:
  Readback.Target{matrix,Jobs.Readers{bias,input,pairs}} = target
  input

def pairs_of(target: Readback.Target) -> Tree.Tree<Tile.Tile16>:
  Readback.Target{matrix,Jobs.Readers{bias,input,pairs}} = target
  pairs

def bias_of(target: Readback.Target) -> Tree.Tree<Tile.Tile8>:
  Readback.Target{matrix,Jobs.Readers{bias,input,pairs}} = target
  bias

def stored_dot(~accumulate: Jobs.Accumulate(),~taps: Taps(),~reduces: LaneReduction(~accumulate,~taps),
  +target: Readback.Target,+pair: U32,+unit: U32,+row: Nat,+lane: Nat) ->
  {Spec.pair_lane(lane,Readback.stored_value_by(~accumulate,target,pair,unit,row)) ==
    (Spec.dot(taps(Readback.target_matrix(target),Gemm.Place{pair,unit},input_of(target),pairs_of(target)),row,lane,
      Spec.Sampled{input_of(target),pairs_of(target),Spec.matrix_stride(Readback.target_matrix(target))},0.0) +
      Spec.pair_lane(lane,Readback.biases(bias_of(target),Readback.target_groups(target),pair)) : F32) : F32}:
  Readback.Target{+matrix,Jobs.Readers{+bias,+input,+pairs}} = target
  +place = {Gemm.Place{pair,unit} : Gemm.Place}
  +acc = accumulate(matrix,place,input,pairs)
  +stored = Spec.row_of(row,acc)
  +biased = Readback.biases(bias,Readback.target_groups(target),pair)
  +offset = Spec.pair_lane(lane,biased)
  +dot = Spec.dot(taps(matrix,place,input,pairs),row,lane,Spec.Sampled{input,pairs,Spec.matrix_stride(matrix)},0.0)
  Equal.trans(F32,Spec.pair_lane(lane,Tile.plus_pair(stored,biased)),(Spec.cell_value(row,lane,acc) + offset : F32),(dot + offset : F32),
    Keys.pair_plus(stored,biased,lane),
    Equal.cong(F32,F32,value => (value + offset : F32),Spec.cell_value(row,lane,acc),dot,reduces(matrix,place,input,pairs,row,lane)))

def group_cursor(+sources: Laws.Sources,+state: Laws.Reduced) ->
  {Spec.cursor_of(Laws.grouped(~Gemm.row_times,sources,state)) == Spec.group_after(Spec.cursor_of(state)) : Gemm.Points}:
  Laws.Sources{+input,+pairs} = sources
  Laws.Reduced{Gemm.Points{+address,+cell,+stride},+accumulators} = state
  {==}

def advance(+rest: Nat,+stride: Nat,+base: Nat) ->
  {Nat.add(Nat.mul(rest,stride),Nat.add(stride,base)) == Nat.add(Nat.mul(1n+rest,stride),base) : Nat}:
  +shift = Nat.mul(rest,stride)
  Equal.trans(Nat,Nat.add(shift,Nat.add(stride,base)),Nat.add(Nat.add(shift,stride),base),Nat.add(Nat.add(stride,shift),base),
    Equal.sym(Nat,Nat.add(Nat.add(shift,stride),base),Nat.add(shift,Nat.add(stride,base)),Addition.associative(shift,stride,base)),
    Equal.cong(Nat,Nat,sum => Nat.add(sum,base),Nat.add(shift,stride),Nat.add(stride,shift),Order.commutative(shift,stride)))

def advanced_state(+state: Laws.Reduced,+cursor: Gemm.Points,+exact: {Spec.cursor_of(state) == cursor : Gemm.Points}) ->
  {state == Laws.Reduced{cursor,Spec.accumulators_of(state)} : Laws.Reduced}:
  Laws.Reduced{+from,+accumulators} = state
  %exact : {Laws.Reduced{from,accumulators} == Laws.Reduced{_,accumulators} : Laws.Reduced}
  {==}

def cursor_after(count: Nat,+sources: Laws.Sources,+address: Nat,+first: Nat,+stride: U32,+accumulators: Gemm.Quad) ->
  {Spec.cursor_of(Laws.groups(~Gemm.row_times,count,sources,Laws.Reduced{Gemm.Points{U32.from_nat(address),U32.from_nat(first),stride},accumulators})) ==
    Gemm.Points{U32.from_nat(Nat.add(Nat.mul(count,16n),address)),U32.from_nat(Nat.add(Nat.mul(count,2n),first)),stride} : Gemm.Points}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +cursor = {Gemm.Points{U32.from_nat(address),U32.from_nat(first),stride} : Gemm.Points}
      +state = {Laws.Reduced{cursor,accumulators} : Laws.Reduced}
      +next = Laws.grouped(~Gemm.row_times,sources,state)
      +next_address = Nat.add(16n,address)
      +next_first = Nat.add(2n,first)
      +after = {Gemm.Points{U32.from_nat(next_address),U32.from_nat(next_first),stride} : Gemm.Points}
      +exact = Equal.trans(Gemm.Points,Spec.cursor_of(next),Spec.group_after(cursor),after,group_cursor(sources,state),Keys.group_cursor(address,first,stride))
      +moved = advanced_state(next,after,exact)
      +wanted = {Gemm.Points{U32.from_nat(Nat.add(Nat.mul(1n+rest,16n),address)),U32.from_nat(Nat.add(Nat.mul(1n+rest,2n),first)),stride} : Gemm.Points}
      %Equal.sym(Laws.Reduced,next,Laws.Reduced{after,Spec.accumulators_of(next)},moved) :
        {Spec.cursor_of(Laws.groups(~Gemm.row_times,rest,sources,_)) == wanted : Gemm.Points}
      %advance(rest,16n,address) :
        {Spec.cursor_of(Laws.groups(~Gemm.row_times,rest,sources,Laws.Reduced{after,Spec.accumulators_of(next)})) ==
          Gemm.Points{U32.from_nat(_),U32.from_nat(Nat.add(Nat.mul(1n+rest,2n),first)),stride} : Gemm.Points}
      %advance(rest,2n,first) :
        {Spec.cursor_of(Laws.groups(~Gemm.row_times,rest,sources,Laws.Reduced{after,Spec.accumulators_of(next)})) ==
          Gemm.Points{U32.from_nat(Nat.add(Nat.mul(rest,16n),next_address)),U32.from_nat(_),stride} : Gemm.Points}
      cursor_after(rest,sources,next_address,next_first,stride,Spec.accumulators_of(next))

def key_number(context: Unit,key: Keys.Key) -> Nat:
  Keys.key_channel(key)

def numbered(count: Nat,+cell: Nat) ->
  {Ordered.map_keys(~Keys.Key,~Nat,~Unit,~key_number,Keys.keys(count,cell),Unit{}) == Ranges.range(Nat.mul(count,16n),Nat.mul(cell,8n)) : List<&2,Nat>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +prefix = Ordered.map_keys(~Keys.Key,~Nat,~Unit,~key_number,Matrix.cat(Keys.Key,Keys.lane_keys(8n,0n,cell),Keys.lane_keys(8n,0n,1n+cell)),Unit{})
      Equal.cong(List<&2,Nat>,List<&2,Nat>,later => Matrix.cat(Nat,prefix,later),
        Ordered.map_keys(~Keys.Key,~Nat,~Unit,~key_number,Keys.keys(rest,1n+1n+cell),Unit{}),Ranges.range(Nat.mul(rest,16n),Nat.mul(1n+1n+cell,8n)),
        numbered(rest,1n+1n+cell))

def job_address(+matrix: Gemm.PointMatrix,+pair: Nat) -> Nat:
  Gemm.PointMatrix{Gemm.TileMatrix{channels,positions,+inner,start,end},base,stride} = matrix
  Nat.mul(pair,U32.to_nat(inner))

def job_cell(+matrix: Gemm.PointMatrix,+unit: Nat) -> Nat:
  Gemm.PointMatrix{product,+base,+stride} = matrix
  Keys.job_first(base,unit,stride)

def whole_keyed(+matrix: Gemm.PointMatrix,+pair: Nat,+unit: Nat) ->
  {whole_taps(matrix,Gemm.Place{U32.from_nat(pair),U32.from_nat(unit)}) ==
    Keys.keyed_taps(Gemm.point_groups_of(matrix),0n,Keys.TapOrigin{job_address(matrix,pair),job_cell(matrix,unit)}) : List<&2,Spec.Tap>}:
  Gemm.PointMatrix{Gemm.TileMatrix{channels,positions,+inner,start,end},+base,+stride} = matrix
  +count = Gemm.point_groups_of(matrix)
  +address = job_address(matrix,pair)
  +first = job_cell(matrix,unit)
  +keyed = Keys.keyed_taps(count,0n,Keys.TapOrigin{address,first})
  %Equal.sym(U32,(U32.from_nat(pair) * inner : U32),U32.from_nat(address),Keys.address_word(pair,inner)) :
    {Spec.taps(count,Gemm.Points{_,(base + U32.from_nat(unit) * 4 * stride : U32),stride}) == keyed : List<&2,Spec.Tap>}
  %Equal.sym(U32,(base + U32.from_nat(unit) * 4 * stride : U32),U32.from_nat(first),Keys.first_word(base,unit,stride)) :
    {Spec.taps(count,Gemm.Points{U32.from_nat(address),_,stride}) == keyed : List<&2,Spec.Tap>}
  Keys.taps_keyed(count,0n,address,first,stride)

def job_cursor(+matrix: Gemm.PointMatrix,+pair: Nat,+unit: Nat,+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>) ->
  {Spec.cursor_of(reached(matrix,Gemm.Place{U32.from_nat(pair),U32.from_nat(unit)},input,pairs)) ==
    Gemm.Points{U32.from_nat(Nat.add(Nat.mul(Gemm.point_groups_of(matrix),16n),job_address(matrix,pair))),
      U32.from_nat(Nat.add(Nat.mul(Gemm.point_groups_of(matrix),2n),job_cell(matrix,unit))),Spec.matrix_stride(matrix)} : Gemm.Points}:
  Gemm.PointMatrix{Gemm.TileMatrix{channels,positions,+inner,start,end},+base,+stride} = matrix
  +count = Gemm.point_groups_of(matrix)
  +address = job_address(matrix,pair)
  +first = job_cell(matrix,unit)
  +sources = {Laws.Sources{input,pairs} : Laws.Sources}
  +acc = Gemm.quad_zero()
  +wanted = {Gemm.Points{U32.from_nat(Nat.add(Nat.mul(count,16n),address)),U32.from_nat(Nat.add(Nat.mul(count,2n),first)),stride} : Gemm.Points}
  %Equal.sym(U32,(U32.from_nat(pair) * inner : U32),U32.from_nat(address),Keys.address_word(pair,inner)) :
    {Spec.cursor_of(Laws.groups(~Gemm.row_times,count,sources,Laws.Reduced{Gemm.Points{_,(base + U32.from_nat(unit) * 4 * stride : U32),stride},acc})) == wanted : Gemm.Points}
  %Equal.sym(U32,(base + U32.from_nat(unit) * 4 * stride : U32),U32.from_nat(first),Keys.first_word(base,unit,stride)) :
    {Spec.cursor_of(Laws.groups(~Gemm.row_times,count,sources,Laws.Reduced{Gemm.Points{U32.from_nat(address),_,stride},acc})) == wanted : Gemm.Points}
  cursor_after(count,sources,address,first,stride,acc)

def suffix_natural(~at: U32 -> U32,count: Nat,+lane: Nat,+cursor: Gemm.Points,
  +inside: {Nat.is_le(Nat.add(lane,count),8n) == True{} : Bool}) ->
  {Spec.suffix_taps(~at,count,U32.from_nat(lane),cursor) == Keys.half_taps(~at,count,lane,cursor) : List<&2,Spec.Tap>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +word = U32.from_nat(lane)
      +exact = Bounds.roundtrip(lane,8,{==},Order.transitive(lane,Nat.add(lane,1n+rest),8n,Order.le_add(lane,1n+rest),inside))
      +next = Spec.advanced(cursor)
      %Equal.sym(Nat,U32.to_nat(word),lane,exact) :
        {Spec.tap_at(~at,_,cursor) <> Spec.suffix_taps(~at,rest,(word + 1 : U32),next) ==
          Spec.tap_at(~at,lane,cursor) <> Keys.half_taps(~at,rest,1n+lane,next) : List<&2,Spec.Tap>}
      %Equal.sym(U32,(word + 1 : U32),U32.from_nat(1n+lane),Bounds.increment_encoded(lane)) :
        {Spec.tap_at(~at,lane,cursor) <> Spec.suffix_taps(~at,rest,_,next) ==
          Spec.tap_at(~at,lane,cursor) <> Keys.half_taps(~at,rest,1n+lane,next) : List<&2,Spec.Tap>}
      Equal.cong(List<&2,Spec.Tap>,List<&2,Spec.Tap>,tail => Spec.tap_at(~at,lane,cursor) <> tail,
        Spec.suffix_taps(~at,rest,U32.from_nat(1n+lane),next),Keys.half_taps(~at,rest,1n+lane,next),
        suffix_natural(~at,rest,1n+lane,next,Order.advance_room(lane,rest,8n,inside)))

def suffix_lower(+remainder: U32) -> Nat:
  U32.to_nat(U32.min(remainder,8))

def suffix_upper(+remainder: U32) -> Nat:
  U32.to_nat((remainder - U32.min(remainder,8) : U32))

def LowerFull(upper: Nat,+lower: Nat) -> Data:
  match upper:
    case 0n: Unit
    case 1n+rest: {lower == 8n : Nat}

def lower_full(upper: Nat,+lower: Nat,+full: {lower == 8n : Nat}) -> LowerFull(upper,lower):
  match upper:
    case 0n: Unit{}
    case 1n+rest: full

type Suffix<-remainder: U32> is Data:
  Suffix{lower: {Nat.is_le(suffix_lower(remainder),8n) == True{} : Bool},
    upper: {Nat.is_le(suffix_upper(remainder),8n) == True{} : Bool},
    total: {Nat.add(suffix_lower(remainder),suffix_upper(remainder)) == U32.to_nat(remainder) : Nat},
    contiguous: LowerFull(suffix_upper(remainder),suffix_lower(remainder))}

def remainder_split(small: Bool,+remainder: U32,+room: {Nat.is_le(U32.to_nat(remainder),16n) == True{} : Bool},
  +compared: {Nat.is_le(U32.to_nat(remainder),8n) == small : Bool}) -> Suffix<remainder>:
  match small:
    case True{}:
      +count = U32.to_nat(remainder)
      +first = suffix_lower(remainder)
      +second = suffix_upper(remainder)
      +word = U32.min(remainder,8)
      +same = Equal.trans(Nat,first,Nat.min(count,8n),count,WordExtrema.minimum(remainder,8),Extrema.minimum_when_ordered(count,8n,compared))
      +word_same = Arithmetic.to_nat_injective(word,remainder,same)
      +equal = {%Equal.sym(U32,word,remainder,word_same) : {U32.to_nat((remainder - _ : U32)) == 0n : Nat}
        %Order.subtract_below(count,count,Order.reflexive(count)) : {U32.to_nat((remainder - remainder : U32)) == _ : Nat}
        Arithmetic.subtract(remainder,remainder,16,{==},Order.reflexive(count),Order.transitive(Nat.sub(count,count),count,16n,Order.sub_le(count,count),room)) :
        {second == 0n : Nat}}
      Suffix{
        %Equal.sym(Nat,first,count,same) : {Nat.is_le(_,8n) == True{} : Bool} compared,
        %Equal.sym(Nat,second,0n,equal) : {Nat.is_le(_,8n) == True{} : Bool} {==},
        %Equal.sym(Nat,first,count,same) : {Nat.add(_,second) == count : Nat}
        %Equal.sym(Nat,second,0n,equal) : {Nat.add(count,_) == count : Nat}
        Addition.zero_right(count),
        %Equal.sym(Nat,second,0n,equal) : LowerFull(_,first)
        Unit{}}
    case False{}:
      +count = U32.to_nat(remainder)
      +first = suffix_lower(remainder)
      +second = suffix_upper(remainder)
      +word = U32.min(remainder,8)
      +more = {%Equal.sym(Bool,Nat.is_lt(8n,count),Bool.not(Nat.is_le(count,8n)),Order.strict_complement(8n,count)) : {_ == True{} : Bool}
        %Equal.sym(Bool,Nat.is_le(count,8n),False{},compared) : {Bool.not(_) == True{} : Bool}
        {==} : {Nat.is_lt(8n,count) == True{} : Bool}}
      +ge = Bounds.strict_le(8n,count,more)
      +same = Equal.trans(Nat,first,Nat.min(count,8n),8n,WordExtrema.minimum(remainder,8),
        Equal.trans(Nat,Nat.min(count,8n),Nat.min(8n,count),8n,Extrema.minimum_commutative(count,8n),Extrema.minimum_when_ordered(8n,count,ge)))
      +word_same = Arithmetic.to_nat_injective(word,8,same)
      +equal = {%Equal.sym(U32,word,8,word_same) : {U32.to_nat((remainder - _ : U32)) == Nat.sub(count,8n) : Nat}
        Arithmetic.subtract(remainder,8,16,{==},ge,Order.transitive(Nat.sub(count,8n),count,16n,Order.sub_le(count,8n),room)) :
        {second == Nat.sub(count,8n) : Nat}}
      Suffix{
        %Equal.sym(Nat,first,8n,same) : {Nat.is_le(_,8n) == True{} : Bool} {==},
        %Equal.sym(Nat,second,Nat.sub(count,8n),equal) : {Nat.is_le(_,8n) == True{} : Bool}
        %Equal.sym(Bool,Nat.is_le(Nat.sub(count,8n),8n),Nat.is_le(count,16n),Order.subtraction_le(count,8n,8n)) : {_ == True{} : Bool}
        room,
        %Equal.sym(Nat,first,8n,same) : {Nat.add(_,second) == count : Nat}
        %Equal.sym(Nat,second,Nat.sub(count,8n),equal) : {Nat.add(8n,_) == count : Nat}
        Order.restore_shift(8n,count,ge),lower_full(second,first,same)}

def suffix(+remainder: U32,+room: {Nat.is_le(U32.to_nat(remainder),16n) == True{} : Bool}) -> Suffix<remainder>:
  remainder_split(Nat.is_le(U32.to_nat(remainder),8n),remainder,room,{==})

def suffix_keys(+remainder: U32,+cell: Nat) -> List<&2,Keys.Key>:
  Matrix.cat(Keys.Key,Keys.lane_keys(suffix_lower(remainder),0n,cell),Keys.lane_keys(suffix_upper(remainder),0n,1n+cell))

def suffix_address(count: Nat,+base: Nat) ->
  {Spec.suffix_address(count,U32.from_nat(base)) == U32.from_nat(Nat.add(count,base)) : U32}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(U32,(U32.from_nat(base) + 1 : U32),U32.from_nat(1n+base),Bounds.increment_encoded(base)) :
        {Spec.suffix_address(rest,_) == U32.from_nat(1n+Nat.add(rest,base)) : U32}
      %Addition.successor_right(rest,base) : {Spec.suffix_address(rest,U32.from_nat(1n+base)) == U32.from_nat(_) : U32}
      suffix_address(rest,1n+base)

def lower_keyed(+count: Nat,+cell: Nat,+address: Nat,+first: Nat,+stride: U32,+small: {Nat.is_le(count,8n) == True{} : Bool}) ->
  {Spec.suffix_taps(~Gemm.first_cell,count,0,Gemm.Points{U32.from_nat(Nat.add(Nat.mul(cell,8n),address)),U32.from_nat(Nat.add(cell,first)),stride}) ==
    Ordered.map_keys(~Keys.Key,~Spec.Tap,~Keys.Origin,~Keys.key_tap,Keys.lane_keys(count,0n,cell),Keys.TapOrigin{address,first}) : List<&2,Spec.Tap>}:
  +cursor = {Gemm.Points{U32.from_nat(Nat.add(Nat.mul(cell,8n),address)),U32.from_nat(Nat.add(cell,first)),stride} : Gemm.Points}
  +numbered = Ordered.map_keys(~Keys.Key,~Spec.Tap,~Keys.Origin,~Keys.key_tap,Keys.lane_keys(count,0n,cell),Keys.TapOrigin{address,first})
  Equal.trans(List<&2,Spec.Tap>,Spec.suffix_taps(~Gemm.first_cell,count,0,cursor),Keys.half_taps(~Gemm.first_cell,count,0n,cursor),numbered,
    suffix_natural(~Gemm.first_cell,count,0n,cursor,small),
    Keys.half_keyed(~Gemm.first_cell,count,0n,cell,address,first,U32.from_nat(Nat.add(cell,first)),stride,{==}))

def upper_keyed(count: Nat,+lower: Nat,+cell: Nat,+address: Nat,+first: Nat,+stride: U32,
  +small: {Nat.is_le(count,8n) == True{} : Bool},+full: LowerFull(count,lower)) ->
  {Spec.suffix_taps(~Gemm.second_cell,count,0,Gemm.Points{Spec.suffix_address(lower,U32.from_nat(Nat.add(Nat.mul(cell,8n),address))),
      U32.from_nat(Nat.add(cell,first)),stride}) ==
    Ordered.map_keys(~Keys.Key,~Spec.Tap,~Keys.Origin,~Keys.key_tap,Keys.lane_keys(count,0n,1n+cell),Keys.TapOrigin{address,first}) : List<&2,Spec.Tap>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +here = Nat.add(Nat.mul(cell,8n),address)
      +next = Nat.add(Nat.mul(1n+cell,8n),address)
      +word = U32.from_nat(Nat.add(cell,first))
      +cursor = {Gemm.Points{U32.from_nat(next),word,stride} : Gemm.Points}
      +numbered = Ordered.map_keys(~Keys.Key,~Spec.Tap,~Keys.Origin,~Keys.key_tap,Keys.lane_keys(1n+rest,0n,1n+cell),Keys.TapOrigin{address,first})
      %Equal.sym(Nat,lower,8n,full) : {Spec.suffix_taps(~Gemm.second_cell,1n+rest,0,Gemm.Points{Spec.suffix_address(_,U32.from_nat(here)),word,stride}) == numbered : List<&2,Spec.Tap>}
      %Equal.sym(U32,Spec.suffix_address(8n,U32.from_nat(here)),U32.from_nat(next),suffix_address(8n,here)) :
        {Spec.suffix_taps(~Gemm.second_cell,1n+rest,0,Gemm.Points{_,word,stride}) == numbered : List<&2,Spec.Tap>}
      Equal.trans(List<&2,Spec.Tap>,Spec.suffix_taps(~Gemm.second_cell,1n+rest,0,cursor),Keys.half_taps(~Gemm.second_cell,1n+rest,0n,cursor),numbered,
        suffix_natural(~Gemm.second_cell,1n+rest,0n,cursor,small),
        Keys.half_keyed(~Gemm.second_cell,1n+rest,0n,1n+cell,address,first,word,stride,Bounds.increment_encoded(Nat.add(cell,first))))

def suffix_keyed(+remainder: U32,+cell: Nat,+address: Nat,+first: Nat,+stride: U32,+span: Suffix<remainder>) ->
  {Spec.remainder_taps(remainder,Gemm.Points{U32.from_nat(Nat.add(Nat.mul(cell,8n),address)),U32.from_nat(Nat.add(cell,first)),stride}) ==
    Ordered.map_keys(~Keys.Key,~Spec.Tap,~Keys.Origin,~Keys.key_tap,suffix_keys(remainder,cell),Keys.TapOrigin{address,first}) : List<&2,Spec.Tap>}:
  Suffix{+low,+high,+total,+full} = span
  +lower = suffix_lower(remainder)
  +upper = suffix_upper(remainder)
  +word = U32.from_nat(Nat.add(cell,first))
  +base = U32.from_nat(Nat.add(Nat.mul(cell,8n),address))
  +here = {Gemm.Points{base,word,stride} : Gemm.Points}
  +next = {Gemm.Points{Spec.suffix_address(lower,base),word,stride} : Gemm.Points}
  +origin = {Keys.TapOrigin{address,first} : Keys.Origin}
  +left = Keys.lane_keys(lower,0n,cell)
  +right = Keys.lane_keys(upper,0n,1n+cell)
  +mapped_left = Ordered.map_keys(~Keys.Key,~Spec.Tap,~Keys.Origin,~Keys.key_tap,left,origin)
  +mapped_right = Ordered.map_keys(~Keys.Key,~Spec.Tap,~Keys.Origin,~Keys.key_tap,right,origin)
  %Ordered.mapped_cat(~Keys.Key,~Spec.Tap,~Keys.Origin,~Keys.key_tap,left,right,origin) :
    {Spec.remainder_taps(remainder,here) == _ : List<&2,Spec.Tap>}
  %lower_keyed(lower,cell,address,first,stride,low) :
    {Spec.remainder_taps(remainder,here) == Matrix.cat(Spec.Tap,_,mapped_right) : List<&2,Spec.Tap>}
  %upper_keyed(upper,lower,cell,address,first,stride,high,full) :
    {Spec.remainder_taps(remainder,here) == Matrix.cat(Spec.Tap,Spec.suffix_taps(~Gemm.first_cell,lower,0,here),_) : List<&2,Spec.Tap>}
  {==}

def numbered_lanes(count: Nat,+lane: Nat,+cell: Nat) ->
  {Ordered.map_keys(~Keys.Key,~Nat,~Unit,~key_number,Keys.lane_keys(count,lane,cell),Unit{}) == Ranges.range(count,Nat.add(lane,Nat.mul(cell,8n))) : List<&2,Nat>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      Equal.cong(List<&2,Nat>,List<&2,Nat>,tail => Nat.add(lane,Nat.mul(cell,8n)) <> tail,
        Ordered.map_keys(~Keys.Key,~Nat,~Unit,~key_number,Keys.lane_keys(rest,1n+lane,cell),Unit{}),
        Ranges.range(rest,Nat.add(1n+lane,Nat.mul(cell,8n))),numbered_lanes(rest,1n+lane,cell))

def numbered_upper(upper: Nat,+lower: Nat,+cell: Nat,+full: LowerFull(upper,lower)) ->
  {Ranges.range(upper,Nat.mul(1n+cell,8n)) == Ranges.range(upper,Nat.add(Nat.mul(cell,8n),lower)) : List<&2,Nat>}:
  match upper:
    case 0n: {==}
    case 1n+rest:
      +first = Nat.mul(cell,8n)
      %Equal.sym(Nat,lower,8n,full) : {Ranges.range(1n+rest,Nat.add(8n,first)) == Ranges.range(1n+rest,Nat.add(first,_)) : List<&2,Nat>}
      Equal.cong(Nat,List<&2,Nat>,position => Ranges.range(1n+rest,position),Nat.add(8n,first),Nat.add(first,8n),Order.commutative(8n,first))

def numbered_suffix(+remainder: U32,+cell: Nat,+span: Suffix<remainder>) ->
  {Ordered.map_keys(~Keys.Key,~Nat,~Unit,~key_number,suffix_keys(remainder,cell),Unit{}) ==
    Ranges.range(U32.to_nat(remainder),Nat.mul(cell,8n)) : List<&2,Nat>}:
  Suffix{+low,+high,+total,+full} = span
  +start = Nat.mul(cell,8n)
  +lower = suffix_lower(remainder)
  +upper = suffix_upper(remainder)
  +left = Keys.lane_keys(lower,0n,cell)
  +right = Keys.lane_keys(upper,0n,1n+cell)
  +more = Ordered.map_keys(~Keys.Key,~Nat,~Unit,~key_number,right,Unit{})
  +left_range = Ranges.range(lower,start)
  +right_range = Ranges.range(upper,Nat.add(start,lower))
  +digits = Matrix.cat(Nat,left_range,right_range)
  +right_exact = Equal.trans(List<&2,Nat>,more,Ranges.range(upper,Nat.mul(1n+cell,8n)),right_range,
    numbered_lanes(upper,0n,1n+cell),numbered_upper(upper,lower,cell,full))
  %Ordered.mapped_cat(~Keys.Key,~Nat,~Unit,~key_number,left,right,Unit{}) : {_ == Ranges.range(U32.to_nat(remainder),start) : List<&2,Nat>}
  %Equal.sym(List<&2,Nat>,Ordered.map_keys(~Keys.Key,~Nat,~Unit,~key_number,left,Unit{}),left_range,numbered_lanes(lower,0n,cell)) :
    {Matrix.cat(Nat,_,more) == Ranges.range(U32.to_nat(remainder),start) : List<&2,Nat>}
  %Equal.sym(List<&2,Nat>,more,right_range,right_exact) : {Matrix.cat(Nat,left_range,_) == Ranges.range(U32.to_nat(remainder),start) : List<&2,Nat>}
  %total : {digits == Ranges.range(_,start) : List<&2,Nat>}
  Equal.sym(List<&2,Nat>,Ranges.range(Nat.add(lower,upper),start),digits,Ranges.split(lower,upper,start))

def inner_of(+matrix: Gemm.PointMatrix) -> U32:
  Gemm.PointMatrix{Gemm.TileMatrix{channels,positions,inner,start,end},base,stride} = matrix
  inner

type InnerSpan<-matrix: Gemm.PointMatrix> is Data:
  InnerSpan{split: {U32.to_nat(inner_of(matrix)) == Nat.add(Nat.mul(Gemm.point_groups_of(matrix),16n),U32.to_nat(Gemm.point_remainder(matrix))) : Nat},
    suffix: Suffix<Gemm.point_remainder(matrix)>}

def inner_span(+limits: Machine.Limits,+matrix: Gemm.PointMatrix) -> InnerSpan<matrix>:
  Gemm.PointMatrix{Gemm.TileMatrix{channels,positions,+inner,start,end},base,stride} = matrix
  +width = U32.to_nat(inner)
  +count = Gemm.point_groups_of(matrix)
  +extra = Gemm.point_remainder(matrix)
  +half = Machine.half_word(limits)
  +half_ok = Limits.half_word_fits(limits)
  +sixteen = Limits.double_bound(limits,16n,Bounds.as_nat_le(16,Machine.capacity(limits),Limits.sixteen_within_capacity(limits)))
  +groups_exact = U32Division.divide(inner,16,half,half_ok,sixteen,{==})
  +extra_exact = U32Division.modulo(inner,16,half,half_ok,sixteen,{==})
  +extra_room = {%Equal.sym(Nat,U32.to_nat(extra),Nat.mod(width,16n),extra_exact) : {Nat.is_le(_,16n) == True{} : Bool}
    Bounds.strict_le(Nat.mod(width,16n),16n,DivisionOrder.remainder_small(width,16n,{==})) :
    {Nat.is_le(U32.to_nat(extra),16n) == True{} : Bool}}
  InnerSpan{
    %Equal.sym(Nat,count,Nat.div(width,16n),groups_exact) : {width == Nat.add(Nat.mul(_,16n),U32.to_nat(extra)) : Nat}
    %Equal.sym(Nat,U32.to_nat(extra),Nat.mod(width,16n),extra_exact) : {width == Nat.add(Nat.mul(Nat.div(width,16n),16n),_) : Nat}
    DivisionOrder.quotient_exact(width,16n,{==}),suffix(extra,extra_room)}

def tail_keys(+matrix: Gemm.PointMatrix) -> List<&2,Keys.Key>:
  +count = Gemm.point_groups_of(matrix)
  Matrix.cat(Keys.Key,Keys.keys(count,0n),suffix_keys(Gemm.point_remainder(matrix),Nat.mul(count,2n)))

def tail_keyed(+matrix: Gemm.PointMatrix,+pair: Nat,+unit: Nat,+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,+span: InnerSpan<matrix>) ->
  {job_tail_taps(matrix,Gemm.Place{U32.from_nat(pair),U32.from_nat(unit)},input,pairs) ==
    Ordered.map_keys(~Keys.Key,~Spec.Tap,~Keys.Origin,~Keys.key_tap,tail_keys(matrix),Keys.TapOrigin{job_address(matrix,pair),job_cell(matrix,unit)}) : List<&2,Spec.Tap>}:
  InnerSpan{+split,+extra} = span
  +count = Gemm.point_groups_of(matrix)
  +cells = Nat.mul(count,2n)
  +address = job_address(matrix,pair)
  +first = job_cell(matrix,unit)
  +stride = Spec.matrix_stride(matrix)
  +place = {Gemm.Place{U32.from_nat(pair),U32.from_nat(unit)} : Gemm.Place}
  +begin = whole_taps(matrix,place)
  +tail = remaining_taps(matrix,place,input,pairs)
  +left = Keys.keys(count,0n)
  +right = suffix_keys(Gemm.point_remainder(matrix),cells)
  +origin = {Keys.TapOrigin{address,first} : Keys.Origin}
  +mapped_left = Ordered.map_keys(~Keys.Key,~Spec.Tap,~Keys.Origin,~Keys.key_tap,left,origin)
  +mapped_right = Ordered.map_keys(~Keys.Key,~Spec.Tap,~Keys.Origin,~Keys.key_tap,right,origin)
  +remainder = Gemm.point_remainder(matrix)
  +cursor = Spec.cursor_of(reached(matrix,place,input,pairs))
  +after = {Gemm.Points{U32.from_nat(Nat.add(Nat.mul(count,16n),address)),U32.from_nat(Nat.add(cells,first)),stride} : Gemm.Points}
  +tail_exact = {%Equal.sym(Gemm.Points,cursor,after,job_cursor(matrix,pair,unit,input,pairs)) :
      {Spec.remainder_taps(remainder,_) == mapped_right : List<&2,Spec.Tap>}
    %Index.assoc(count,2n,8n) : {Spec.remainder_taps(remainder,Gemm.Points{U32.from_nat(Nat.add(_,address)),U32.from_nat(Nat.add(cells,first)),stride}) == mapped_right : List<&2,Spec.Tap>}
    suffix_keyed(remainder,cells,address,first,stride,extra) : {tail == mapped_right : List<&2,Spec.Tap>}}
  %Ordered.mapped_cat(~Keys.Key,~Spec.Tap,~Keys.Origin,~Keys.key_tap,left,right,origin) :
    {Matrix.cat(Spec.Tap,begin,tail) == _ : List<&2,Spec.Tap>}
  %whole_keyed(matrix,pair,unit) : {Matrix.cat(Spec.Tap,begin,tail) == Matrix.cat(Spec.Tap,_,mapped_right) : List<&2,Spec.Tap>}
  %tail_exact : {Matrix.cat(Spec.Tap,begin,tail) == Matrix.cat(Spec.Tap,begin,_) : List<&2,Spec.Tap>}
  {==}

def numbered_tail(+matrix: Gemm.PointMatrix,+span: InnerSpan<matrix>) ->
  {Ordered.map_keys(~Keys.Key,~Nat,~Unit,~key_number,tail_keys(matrix),Unit{}) == Ranges.range(U32.to_nat(inner_of(matrix)),0n) : List<&2,Nat>}:
  InnerSpan{+split,+extra} = span
  +count = Gemm.point_groups_of(matrix)
  +cells = Nat.mul(count,2n)
  +head = Nat.mul(count,16n)
  +tail = U32.to_nat(Gemm.point_remainder(matrix))
  +left = Keys.keys(count,0n)
  +right = suffix_keys(Gemm.point_remainder(matrix),cells)
  +left_range = Ranges.range(head,0n)
  +right_range = Ranges.range(tail,head)
  +total = Nat.add(head,tail)
  +right_map = Ordered.map_keys(~Keys.Key,~Nat,~Unit,~key_number,right,Unit{})
  +right_exact = {%Index.assoc(count,2n,8n) : {right_map == Ranges.range(tail,_) : List<&2,Nat>}
    numbered_suffix(Gemm.point_remainder(matrix),cells,extra) : {right_map == right_range : List<&2,Nat>}}
  %Ordered.mapped_cat(~Keys.Key,~Nat,~Unit,~key_number,left,right,Unit{}) : {_ == Ranges.range(U32.to_nat(inner_of(matrix)),0n) : List<&2,Nat>}
  %Equal.sym(List<&2,Nat>,Ordered.map_keys(~Keys.Key,~Nat,~Unit,~key_number,left,Unit{}),left_range,numbered(count,0n)) :
    {Matrix.cat(Nat,_,right_map) == Ranges.range(U32.to_nat(inner_of(matrix)),0n) : List<&2,Nat>}
  %Equal.sym(List<&2,Nat>,right_map,right_range,right_exact) : {Matrix.cat(Nat,left_range,_) == Ranges.range(U32.to_nat(inner_of(matrix)),0n) : List<&2,Nat>}
  %Equal.sym(Nat,U32.to_nat(inner_of(matrix)),total,split) : {Matrix.cat(Nat,left_range,right_range) == Ranges.range(_,0n) : List<&2,Nat>}
  Equal.sym(List<&2,Nat>,Ranges.range(total,0n),Matrix.cat(Nat,left_range,right_range),Ranges.split(head,tail,0n))

def numbered_whole(+matrix: Gemm.PointMatrix,+span: InnerSpan<matrix>,+whole: {U32.is_zero(Gemm.point_remainder(matrix)) == True{} : Bool}) ->
  {Ordered.map_keys(~Keys.Key,~Nat,~Unit,~key_number,Keys.keys(Gemm.point_groups_of(matrix),0n),Unit{}) == Ranges.range(U32.to_nat(inner_of(matrix)),0n) : List<&2,Nat>}:
  InnerSpan{+split,+extra} = span
  +width = U32.to_nat(inner_of(matrix))
  +count = Gemm.point_groups_of(matrix)
  +head = Nat.mul(count,16n)
  +rem = Gemm.point_remainder(matrix)
  +exact = Equal.trans(Nat,width,Nat.add(head,U32.to_nat(rem)),head,split,
    %Equal.sym(U32,rem,0,Arithmetic.word_zero(rem,whole)) : {Nat.add(head,U32.to_nat(_)) == head : Nat}
    Addition.zero_right(head))
  %Equal.sym(Nat,width,head,exact) :
    {Ordered.map_keys(~Keys.Key,~Nat,~Unit,~key_number,Keys.keys(count,0n),Unit{}) == Ranges.range(_,0n) : List<&2,Nat>}
  numbered(count,0n)

def LaneInside(key: Keys.Key) -> Data:
  Keys.LaneKey{cell,+lane} = key
  {Nat.is_lt(lane,8n) == True{} : Bool}

def safe_lanes(count: Nat,+lane: Nat,+cell: Nat,+inside: {Nat.is_le(Nat.add(lane,count),8n) == True{} : Bool}) ->
  Members.All(Keys.Key,LaneInside,Keys.lane_keys(count,lane,cell)):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      Gemm.Both{Order.room_index(lane,rest,8n,inside),safe_lanes(rest,1n+lane,cell,Order.advance_room(lane,rest,8n,inside))}

def safe_groups(count: Nat,+cell: Nat) -> Members.All(Keys.Key,LaneInside,Keys.keys(count,cell)):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      +left = Keys.lane_keys(8n,0n,cell)
      +right = Keys.lane_keys(8n,0n,1n+cell)
      Members.all_concatenated(Keys.Key,LaneInside,Matrix.cat(Keys.Key,left,right),Keys.keys(rest,1n+1n+cell),
        Members.all_concatenated(Keys.Key,LaneInside,left,right,safe_lanes(8n,0n,cell,{==}),safe_lanes(8n,0n,1n+cell,{==})),
        safe_groups(rest,1n+1n+cell))

def safe_suffix(+remainder: U32,+cell: Nat,+span: Suffix<remainder>) -> Members.All(Keys.Key,LaneInside,suffix_keys(remainder,cell)):
  Suffix{+low,+high,+total,+full} = span
  +lower = suffix_lower(remainder)
  +upper = suffix_upper(remainder)
  Members.all_concatenated(Keys.Key,LaneInside,Keys.lane_keys(lower,0n,cell),Keys.lane_keys(upper,0n,1n+cell),
    safe_lanes(lower,0n,cell,low),safe_lanes(upper,0n,1n+cell,high))

def safe_tail(+matrix: Gemm.PointMatrix,+span: InnerSpan<matrix>) -> Members.All(Keys.Key,LaneInside,tail_keys(matrix)):
  InnerSpan{+mixed,+suffix} = span
  +prefix = Gemm.point_groups_of(matrix)
  +cell = Nat.mul(prefix,2n)
  Members.all_concatenated(Keys.Key,LaneInside,Keys.keys(prefix,0n),suffix_keys(Gemm.point_remainder(matrix),cell),
    safe_groups(prefix,0n),safe_suffix(Gemm.point_remainder(matrix),cell,suffix))

type KeyOrder<-taps: Taps(),-matrix: Gemm.PointMatrix,-pair: Nat,-unit: Nat,-input: Tree.Tree<Tile.Tile8>,-pairs: Tree.Tree<Tile.Tile16>> is Data:
  KeyOrder{keys: List<&2,Keys.Key>,safe: Members.All(Keys.Key,LaneInside,keys),
    ordered: {Ordered.map_keys(~Keys.Key,~Nat,~Unit,~key_number,keys,Unit{}) == Ranges.range(U32.to_nat(inner_of(matrix)),0n) : List<&2,Nat>},
    tapped: {taps(matrix,Gemm.Place{U32.from_nat(pair),U32.from_nat(unit)},input,pairs) ==
      Ordered.map_keys(~Keys.Key,~Spec.Tap,~Keys.Origin,~Keys.key_tap,keys,Keys.TapOrigin{job_address(matrix,pair),job_cell(matrix,unit)}) : List<&2,Spec.Tap>}}

def whole_order(+limits: Machine.Limits,+matrix: Gemm.PointMatrix,+pair: Nat,+unit: Nat,+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>,
  +whole: {U32.is_zero(Gemm.point_remainder(matrix)) == True{} : Bool}) ->
  KeyOrder<job_whole_taps,matrix,pair,unit,input,pairs>:
  KeyOrder{Keys.keys(Gemm.point_groups_of(matrix),0n),safe_groups(Gemm.point_groups_of(matrix),0n),numbered_whole(matrix,inner_span(limits,matrix),whole),whole_keyed(matrix,pair,unit)}

def tail_order(+limits: Machine.Limits,+matrix: Gemm.PointMatrix,+pair: Nat,+unit: Nat,+input: Tree.Tree<Tile.Tile8>,+pairs: Tree.Tree<Tile.Tile16>) ->
  KeyOrder<job_tail_taps,matrix,pair,unit,input,pairs>:
  +span = inner_span(limits,matrix)
  KeyOrder{tail_keys(matrix),safe_tail(matrix,span),numbered_tail(matrix,span),tail_keyed(matrix,pair,unit,input,pairs,span)}

def sample_origin(+target: Readback.Target,+pair: Nat,+unit: Nat) -> Gemm.Both<Keys.Origin,Spec.Sampled>:
  +matrix = Readback.target_matrix(target)
  Gemm.Both{Keys.TapOrigin{job_address(matrix,pair),job_cell(matrix,unit)},Spec.Sampled{input_of(target),pairs_of(target),Spec.matrix_stride(matrix)}}

def bias_lane(+target: Readback.Target,+pair: Nat,+lane: Nat) -> F32:
  Spec.pair_lane(lane,Readback.biases(bias_of(target),Readback.target_groups(target),U32.from_nat(pair)))

def keyed_dot(+keys: List<&2,Keys.Key>,+target: Readback.Target,+pair: Nat,+unit: Nat,+row: Nat,+lane: Nat) -> F32:
  Matrix.dot(~Nat,~Keys.Key,~Nat,~Gemm.Both<Keys.Origin,Spec.Sampled>,~Keys.key_input(),~Keys.key_weight(),keys,lane,row,sample_origin(target,pair,unit),0.0)

type OrderedLane<-accumulate: Jobs.Accumulate(),-target: Readback.Target,-pair: Nat,-unit: Nat,-row: Nat,-lane: Nat> is Data:
  OrderedLane{keys: List<&2,Keys.Key>,safe: Members.All(Keys.Key,LaneInside,keys),
    sequence: {Ordered.map_keys(~Keys.Key,~Nat,~Unit,~key_number,keys,Unit{}) == Ranges.range(U32.to_nat(inner_of(Readback.target_matrix(target))),0n) : List<&2,Nat>},
    exact: {Spec.pair_lane(lane,Readback.stored_value_by(~accumulate,target,U32.from_nat(pair),U32.from_nat(unit),row)) ==
      (keyed_dot(keys,target,pair,unit,row,lane) + bias_lane(target,pair,lane) : F32) : F32}}

def ordered_lane(~accumulate: Jobs.Accumulate(),~taps: Taps(),~reduces: LaneReduction(~accumulate,~taps),
  +target: Readback.Target,+pair: Nat,+unit: Nat,+row: Nat,+lane: Nat,
  +order: KeyOrder<taps,Readback.target_matrix(target),pair,unit,input_of(target),pairs_of(target)>) ->
  OrderedLane<accumulate,target,pair,unit,row,lane>:
  Readback.Target{+matrix,Jobs.Readers{+bias,+input,+pairs}} = target
  KeyOrder{+keys,+safe,+sequence,+tapped} = order
  +place = {Gemm.Place{U32.from_nat(pair),U32.from_nat(unit)} : Gemm.Place}
  +samples = {Spec.Sampled{input,pairs,Spec.matrix_stride(matrix)} : Spec.Sampled}
  +origin = {Keys.TapOrigin{job_address(matrix,pair),job_cell(matrix,unit)} : Keys.Origin}
  +actual = Spec.pair_lane(lane,Readback.stored_value_by(~accumulate,target,U32.from_nat(pair),U32.from_nat(unit),row))
  +accumulated = Spec.dot(taps(matrix,place,input,pairs),row,lane,samples,0.0)
  +dotted = keyed_dot(keys,target,pair,unit,row,lane)
  +added = bias_lane(target,pair,lane)
  +reindexed = {%Equal.sym(List<&2,Spec.Tap>,taps(matrix,place,input,pairs),Ordered.map_keys(~Keys.Key,~Spec.Tap,~Keys.Origin,~Keys.key_tap,keys,origin),tapped) :
      {Spec.dot(_,row,lane,samples,0.0) == dotted : F32}
    Ordered.dot_reindexed(~Keys.Key,~Spec.Tap,~Nat,~Nat,~Spec.Sampled,~Keys.Origin,~Keys.key_tap,~Spec.input_sample,~Spec.weight_sample,
      keys,origin,lane,row,samples,0.0) : {accumulated == dotted : F32}}
  OrderedLane{keys,safe,sequence,
    Equal.trans(F32,actual,(accumulated + added : F32),(dotted + added : F32),
      stored_dot(~accumulate,~taps,~reduces,target,U32.from_nat(pair),U32.from_nat(unit),row,lane),
      Equal.cong(F32,F32,value => (value + added : F32),accumulated,dotted,reindexed))}

def input_address(+target: Readback.Target,+unit: Nat,+row: Nat,+key: Nat) -> Nat:
  +matrix = Readback.target_matrix(target)
  Nat.add(Nat.add(Nat.div(key,8n),job_cell(matrix,unit)),Nat.mul(row,U32.to_nat(Spec.matrix_stride(matrix))))

def physical_input(+target: Readback.Target,+unit: Nat,+row: Nat,+key: Nat) -> F32:
  TileLaws.select_lane(Nat.mod(key,8n),Tree.read(Tile.Tile8,input_of(target),U32.from_nat(input_address(target,unit,row,key))))

def physical_weight(+target: Readback.Target,+pair: Nat,+lane: Nat,+key: Nat) -> F32:
  Spec.pair_lane(lane,Tree.read(Tile.Tile16,pairs_of(target),U32.from_nat(Nat.add(job_address(Readback.target_matrix(target),pair),key))))

def input_key(+target: Readback.Target,+pair: Nat,+unit: Nat,+row: Nat,+key: Keys.Key,
  +row_inside: {Nat.is_le(row,3n) == True{} : Bool},+lane_inside: LaneInside(key)) ->
  {Keys.key_input()(sample_origin(target,pair,unit),row,key) == physical_input(target,unit,row,Keys.key_channel(key)) : F32}:
  Keys.LaneKey{+cell,+lane} = key
  +matrix = Readback.target_matrix(target)
  +stride = Spec.matrix_stride(matrix)
  +first = job_cell(matrix,unit)
  +number = Keys.key_channel(key)
  +column = Nat.mul(cell,8n)
  +normal = Nat.add(column,lane)
  +equal = Order.commutative(lane,column)
  +input = input_of(target)
  +actual = Keys.key_input()(sample_origin(target,pair,unit),row,key)
  %Equal.sym(Nat,number,normal,equal) :
    {actual == TileLaws.select_lane(Nat.mod(_,8n),Tree.read(Tile.Tile8,input,U32.from_nat(input_address(target,unit,row,_)))) : F32}
  %Equal.sym(Nat,Nat.div(normal,8n),cell,Radix.quotient(cell,lane,8n,{==},lane_inside)) :
    {actual == TileLaws.select_lane(Nat.mod(normal,8n),Tree.read(Tile.Tile8,input,U32.from_nat(Nat.add(Nat.add(_,first),Nat.mul(row,U32.to_nat(stride)))))) : F32}
  %Equal.sym(Nat,Nat.mod(normal,8n),lane,Radix.remainder(cell,lane,8n,{==},lane_inside)) :
    {actual == TileLaws.select_lane(_,Tree.read(Tile.Tile8,input,U32.from_nat(Nat.add(Nat.add(cell,first),Nat.mul(row,U32.to_nat(stride)))))) : F32}
  Keys.key_input_value(job_address(matrix,pair),first,input,pairs_of(target),stride,row,cell,lane,row_inside)

def weight_key(+target: Readback.Target,+pair: Nat,+unit: Nat,+lane: Nat,+key: Keys.Key) ->
  {Keys.key_weight()(sample_origin(target,pair,unit),lane,key) == physical_weight(target,pair,lane,Keys.key_channel(key)) : F32}:
  Keys.LaneKey{+cell,+part} = key
  +base = job_address(Readback.target_matrix(target),pair)
  +offset = Nat.mul(cell,8n)
  +number = Nat.add(part,offset)
  +source = pairs_of(target)
  +actual = Nat.add(part,Nat.add(offset,base))
  +expected = Nat.add(base,number)
  +at = Equal.trans(Nat,actual,Nat.add(number,base),expected,
    Equal.sym(Nat,Nat.add(number,base),actual,Addition.associative(part,offset,base)),Order.commutative(number,base))
  Equal.cong(Nat,F32,number => Spec.pair_lane(lane,Tree.read(Tile.Tile16,source,U32.from_nat(number))),actual,expected,at)
