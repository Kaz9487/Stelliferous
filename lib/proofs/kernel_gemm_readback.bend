# Every in-range output of the actual four-row-unit GEMM is written by exactly
# one job, row and lane; its value is that job's ordered accumulator plus bias.
# Jobs are input-major: index = group * units + unit.
import Base
import ../math_sequence.bend as Sequence
import ./kernel_coordinate_refinement.bend as CoordinateLaws
import ./kernel_tile_refinement.bend as TileLaws
import ../kernel_gemm.bend as Gemm
import ../kernel_rows.bend as Rows
import ../kernel_tile.bend as Tile
import ../traversal_reduction.bend as Reduction
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./write_program.bend as Writes
import ./kernel_gemm_refinement.bend as GemmLaws
import ./kernel_gemm_writes.bend as GemmWrites
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./binary_radix.bend as Radix
import ./row_major_index.bend as RowMajor
import ./u32_comparison.bend as Comparison
import ./u32_division.bend as U32Division
import ./natural_addition.bend as Addition
import ./boolean_logic.bend as Boolean
import ./word_encoding.bend as WordEncoding
import ./nat_division.bend as Division
import ./write_program_readback.bend as WriteReadback
import ./u32_shift.bend as U32Shift
import ./kernel_accumulator.bend as Accumulator
import ./kernel_weight_gradient_packing.bend as PackedRows
import ./kernel_weight_gradient_group.bend as MicroDot
import ./storage_readback.bend as TreeRead
import ./kernel_reduction_model.bend as ReductionModel
import ../tensor_view.bend as Views
import ./view_writes.bend as ViewWrites
import ./tensor_copy_values.bend as Copy
import ./write_program_every.bend as Every

type Query is Data:
  Query{rows: U32,columns: U32,inner: U32,groups: Nat,limit: U32,
    input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>,channel: Nat,column: Nat,
    limit_fits: {U32.is_le(limit,2147483648) == True{} : Bool},
    columns_exact: {U32.to_nat(columns) == Nat.mul(groups,8n) : Nat},
    rows_room: {Nat.is_le(Nat.add(U32.to_nat(rows),3n),U32.to_nat(limit)) == True{} : Bool},
    area_room: {Nat.is_le(Nat.mul(U32.to_nat(rows),U32.to_nat(columns)),U32.to_nat(limit)) == True{} : Bool},
    channel_inside: {Nat.is_lt(channel,U32.to_nat(rows)) == True{} : Bool},
    column_inside: {Nat.is_lt(column,U32.to_nat(columns)) == True{} : Bool}}

def rows(query: Query) -> U32:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  rows

def columns(query: Query) -> U32:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  columns

def groups(query: Query) -> Nat:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  groups

def limit(query: Query) -> U32:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  limit

def layout(query: Query) -> Gemm.Layout:
  Query{rows,+columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  Gemm.Layout{rows,columns,inner,0,columns}

def input(query: Query) -> Tree.Tree<Tile.Tile8>:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  input

def weights(query: Query) -> Tree.Tree<Tile.Tile8>:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  weights

def bias(query: Query) -> Tree.Tree<F32>:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  bias

def channel(query: Query) -> Nat:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  channel

def column(query: Query) -> Nat:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  column

def inner(query: Query) -> U32:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  inner

def limit_fits(+query: Query) -> {U32.is_le(limit(query),2147483648) == True{} : Bool}:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  limit_fits

def columns_exact(+query: Query) -> {U32.to_nat(columns(query)) == Nat.mul(groups(query),8n) : Nat}:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  columns_exact

def rows_room(+query: Query) -> {Nat.is_le(Nat.add(U32.to_nat(rows(query)),3n),U32.to_nat(limit(query))) == True{} : Bool}:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  rows_room

def area_room(+query: Query) -> {Nat.is_le(Nat.mul(U32.to_nat(rows(query)),U32.to_nat(columns(query))),U32.to_nat(limit(query))) == True{} : Bool}:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  area_room

def channel_inside(+query: Query) -> {Nat.is_lt(channel(query),U32.to_nat(rows(query))) == True{} : Bool}:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  channel_inside

def column_inside(+query: Query) -> {Nat.is_lt(column(query),U32.to_nat(columns(query))) == True{} : Bool}:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  column_inside

def select_lane(lane: Nat,values: Tile.Tile8) -> F32:
  TileLaws.select_lane(lane,values)

# Units of four output rows, as Gemm.row_units counts them.
def units(query: Query) -> Nat:
  Nat.div(Nat.add(U32.to_nat(rows(query)),3n),4n)

def job_index(unit: Nat,group: Nat,units: Nat) -> Nat:
  Nat.add(Nat.mul(group,units),unit)

def accumulators(job: Gemm.Job,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>) -> Rows.Rows<Tile.Tile8>:
  Gemm.Job{+context,upper,count} = job
  Reduction.accumulator(~&2,~Tree.Tree<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~U32,~Rows.Rows<Tile.Tile8>,
    GemmLaws.reduce(upper,count,input,weights,context))

def unit_accumulators(+query: Query,unit: Nat,group: Nat) -> Rows.Rows<Tile.Tile8>:
  accumulators(Gemm.job(layout(query),U32.from_nat(job_index(unit,group,units(query)))),input(query),weights(query))

def output_value(~biased: Bool,+query: Query,+unit: Nat,+row: Nat,+group: Nat,+lane: Nat,row_word: U32) -> F32:
  Bool.pick(F32,biased,(select_lane(lane,Accumulator.select_row(~Tile.Tile8,row,unit_accumulators(query,unit,group))) + Tree.read(F32,bias(query),row_word) : F32),select_lane(lane,Accumulator.select_row(~Tile.Tile8,row,unit_accumulators(query,unit,group))))

def address(+query: Query) -> Nat:
  Nat.add(Nat.mul(channel(query),U32.to_nat(columns(query))),column(query))

def wanted(~biased: Bool,+query: Query) -> F32:
  output_value(~biased,query,Nat.div(channel(query),4n),Nat.mod(channel(query),4n),Nat.div(column(query),8n),Nat.mod(column(query),8n),U32.from_nat(channel(query)))

def row_number(unit: Nat,row: Nat) -> Nat:
  Nat.add(Nat.mul(unit,4n),row)

def lane_number(group: Nat,lane: Nat) -> Nat:
  Nat.add(Nat.mul(group,8n),lane)

def columns_positive(+query: Query) -> {Nat.is_lt(0n,U32.to_nat(columns(query))) == True{} : Bool}:
  Order.le_lt_trans(0n,column(query),U32.to_nat(columns(query)),Order.zero_le(column(query)),column_inside(query))

def rows_positive(+query: Query) -> {Nat.is_lt(0n,U32.to_nat(rows(query))) == True{} : Bool}:
  Order.le_lt_trans(0n,channel(query),U32.to_nat(rows(query)),Order.zero_le(channel(query)),channel_inside(query))

def lane_inside(+query: Query,+group: Nat,+lane: Nat,group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool},lane_small: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {Nat.is_lt(lane_number(group,lane),U32.to_nat(columns(query))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(columns(query)),Nat.mul(groups(query),8n),columns_exact(query)) : {Nat.is_lt(lane_number(group,lane),_) == True{} : Bool}
  RowMajor.radix(group,groups(query),lane,8n,group_inside,lane_small)

# If the query address equals an actual store address, the store is the one
# selected by the query: same unit, row, group and lane, hence same value.
def selected_value(~biased: Bool,+query: Query,+unit: Nat,+row: Nat,+group: Nat,+lane: Nat,+row_word: U32,
  +same: {address(query) == Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,lane)) : Nat},
  +row_exact: {U32.to_nat(row_word) == row_number(unit,row) : Nat},
  +row_small: {Nat.is_lt(row,4n) == True{} : Bool},+lane_small: {Nat.is_lt(lane,8n) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool}) ->
  {output_value(~biased,query,unit,row,group,lane,row_word) == wanted(~biased,query) : F32}:
  +width = U32.to_nat(columns(query))
  +positive = columns_positive(query)
  +lane_bound = lane_inside(query,group,lane,group_inside,lane_small)
  +channel_same = Radix.row_unique(channel(query),row_number(unit,row),width,column(query),lane_number(group,lane),positive,column_inside(query),lane_bound,same)
  +column_same = Radix.col_unique(channel(query),row_number(unit,row),width,column(query),lane_number(group,lane),positive,column_inside(query),lane_bound,same)
  %Equal.sym(Nat,channel(query),row_number(unit,row),channel_same) :
    {output_value(~biased,query,unit,row,group,lane,row_word) ==
      output_value(~biased,query,Nat.div(_,4n),Nat.mod(_,4n),Nat.div(column(query),8n),Nat.mod(column(query),8n),U32.from_nat(_)) : F32}
  %Equal.sym(Nat,column(query),lane_number(group,lane),column_same) :
    {output_value(~biased,query,unit,row,group,lane,row_word) ==
      output_value(~biased,query,Nat.div(row_number(unit,row),4n),Nat.mod(row_number(unit,row),4n),Nat.div(_,8n),Nat.mod(_,8n),U32.from_nat(row_number(unit,row))) : F32}
  %Equal.sym(Nat,Nat.div(row_number(unit,row),4n),unit,Radix.quotient(unit,row,4n,{==},row_small)) :
    {output_value(~biased,query,unit,row,group,lane,row_word) ==
      output_value(~biased,query,_,Nat.mod(row_number(unit,row),4n),Nat.div(lane_number(group,lane),8n),Nat.mod(lane_number(group,lane),8n),U32.from_nat(row_number(unit,row))) : F32}
  %Equal.sym(Nat,Nat.mod(row_number(unit,row),4n),row,Radix.remainder(unit,row,4n,{==},row_small)) :
    {output_value(~biased,query,unit,row,group,lane,row_word) ==
      output_value(~biased,query,unit,_,Nat.div(lane_number(group,lane),8n),Nat.mod(lane_number(group,lane),8n),U32.from_nat(row_number(unit,row))) : F32}
  %Equal.sym(Nat,Nat.div(lane_number(group,lane),8n),group,Radix.quotient(group,lane,8n,{==},lane_small)) :
    {output_value(~biased,query,unit,row,group,lane,row_word) ==
      output_value(~biased,query,unit,row,_,Nat.mod(lane_number(group,lane),8n),U32.from_nat(row_number(unit,row))) : F32}
  %Equal.sym(Nat,Nat.mod(lane_number(group,lane),8n),lane,Radix.remainder(group,lane,8n,{==},lane_small)) :
    {output_value(~biased,query,unit,row,group,lane,row_word) ==
      output_value(~biased,query,unit,row,group,_,U32.from_nat(row_number(unit,row))) : F32}
  %Equal.sym(U32,U32.from_nat(row_number(unit,row)),row_word,Arithmetic.from_nat_exact(row_word,row_number(unit,row),row_exact)) :
    {output_value(~biased,query,unit,row,group,lane,row_word) == output_value(~biased,query,unit,row,group,lane,_) : F32}
  {==}

def columns_within(+query: Query) -> {Nat.is_le(U32.to_nat(columns(query)),U32.to_nat(limit(query))) == True{} : Bool}:
  Order.transitive(U32.to_nat(columns(query)),Nat.mul(U32.to_nat(rows(query)),U32.to_nat(columns(query))),U32.to_nat(limit(query)),
    %RowMajor.multiply_commutative(U32.to_nat(columns(query)),U32.to_nat(rows(query))) : {Nat.is_le(U32.to_nat(columns(query)),_) == True{} : Bool}
    RowMajor.product_covers(U32.to_nat(columns(query)),U32.to_nat(rows(query)),rows_positive(query)),area_room(query))

def row_inside(+query: Query,+unit: Nat,+row: Nat,+row_word: U32,
  +row_exact: {U32.to_nat(row_word) == row_number(unit,row) : Nat},valid: {U32.is_lt(row_word,rows(query)) == True{} : Bool}) ->
  {Nat.is_lt(row_number(unit,row),U32.to_nat(rows(query))) == True{} : Bool}:
  %row_exact : {Nat.is_lt(_,U32.to_nat(rows(query))) == True{} : Bool}
  %Comparison.lt(row_word,rows(query)) : {_ == True{} : Bool}
  valid

def double_within_four(+value: Nat) -> {Nat.is_le(Nat.double(value),Nat.mul(value,4n)) == True{} : Bool}:
  match value:
    case 0n: {==}
    case 1n+ +rest:
      Order.successor(Nat.double(rest),1n+Nat.mul(rest,4n),Order.successor(Nat.double(rest),Nat.mul(rest,4n),double_within_four(rest)))

def groups_within_columns(+query: Query) -> {Nat.is_le(groups(query),U32.to_nat(columns(query))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(columns(query)),Nat.mul(groups(query),8n),columns_exact(query)) : {Nat.is_le(groups(query),_) == True{} : Bool}
  RowMajor.product_covers(groups(query),8n,{==})

def groups_word_exact(+query: Query) -> {U32.to_nat(Gemm.column_groups(columns(query))) == groups(query) : Nat}:
  Equal.trans(Nat,U32.to_nat(Gemm.column_groups(columns(query))),Nat.div(U32.to_nat(columns(query)),8n),groups(query),
    U32Division.divide(columns(query),8,2147483648,{==},{==},{==}),
    %Equal.sym(Nat,U32.to_nat(columns(query)),Nat.mul(groups(query),8n),columns_exact(query)) : {Nat.div(_,8n) == groups(query) : Nat}
    %Addition.zero_right(Nat.mul(groups(query),8n)) : {Nat.div(_,8n) == groups(query) : Nat}
    Radix.quotient(groups(query),0n,8n,{==},{==}))

def units_word_exact(+query: Query) -> {U32.to_nat(Gemm.row_units(rows(query))) == units(query) : Nat}:
  Equal.trans(Nat,U32.to_nat(Gemm.row_units(rows(query))),Nat.div(U32.to_nat((rows(query) + 3 : U32)),4n),units(query),
    U32Division.divide((rows(query) + 3 : U32),4,2147483648,{==},{==},{==}),
    Equal.cong(Nat,Nat,value => Nat.div(value,4n),U32.to_nat((rows(query) + 3 : U32)),Nat.add(U32.to_nat(rows(query)),3n),
      Arithmetic.add(rows(query),3,limit(query),limit_fits(query),rows_room(query))))

def decomposed(+value: Nat,+divisor: Nat,+limit: U32,+fits: {U32.is_le(limit,2147483648) == True{} : Bool},
  +small: {Nat.is_le(value,U32.to_nat(limit)) == True{} : Bool},+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {value == Nat.add(Nat.mul(Nat.div(value,divisor),divisor),Nat.mod(value,divisor)) : Nat}:
  %Bounds.roundtrip(value,limit,fits,small) :
    {_ == Nat.add(Nat.mul(Nat.div(_,divisor),divisor),Nat.mod(_,divisor)) : Nat}
  Radix.division_identity(U32.from_nat(value),divisor,positive)

def remainder_small(+value: Nat,+divisor: Nat,+limit: U32,+fits: {U32.is_le(limit,2147483648) == True{} : Bool},
  +small: {Nat.is_le(value,U32.to_nat(limit)) == True{} : Bool},+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {Nat.is_lt(Nat.mod(value,divisor),divisor) == True{} : Bool}:
  %Bounds.roundtrip(value,limit,fits,small) : {Nat.is_lt(Nat.mod(_,divisor),divisor) == True{} : Bool}
  Radix.remainder_inside(U32.from_nat(value),divisor,positive)

def units_room(+query: Query) -> {Nat.is_le(Nat.mul(units(query),4n),Nat.add(U32.to_nat(rows(query)),3n)) == True{} : Bool}:
  +padded = Nat.add(U32.to_nat(rows(query)),3n)
  %Equal.sym(Nat,padded,Nat.add(Nat.mul(units(query),4n),Nat.mod(padded,4n)),decomposed(padded,4n,limit(query),limit_fits(query),rows_room(query),{==})) :
    {Nat.is_le(Nat.mul(units(query),4n),_) == True{} : Bool}
  Order.le_add(Nat.mul(units(query),4n),Nat.mod(padded,4n))

# A unit of four fits below the channel count whenever units*4 <= rows+3.
def unit_row_room(+rows: Nat,+unit: Nat,room: {Nat.is_le(Nat.mul(1n+unit,4n),Nat.add(rows,3n)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.mul(unit,4n),rows) == True{} : Bool}:
  Bounds.predecessor_strict(Nat.mul(unit,4n),rows,
    %Order.commutative(rows,3n) : {Nat.is_le(Nat.mul(1n+unit,4n),_) == True{} : Bool}
    room)

def units_within_rows(+rows: Nat,+count: Nat,room: {Nat.is_le(Nat.mul(count,4n),Nat.add(rows,3n)) == True{} : Bool}) ->
  {Nat.is_le(count,rows) == True{} : Bool}:
  match count:
    case 0n: Order.zero_le(rows)
    case 1n+ +unit:
      Order.strict_follow(unit,rows,Order.le_lt_trans(unit,Nat.mul(unit,4n),rows,RowMajor.product_covers(unit,4n,{==}),unit_row_room(rows,unit,room)))

def unit_bound(+query: Query,+unit: Nat,inside: {Nat.is_lt(unit,units(query)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.mul(unit,4n),U32.to_nat(rows(query))) == True{} : Bool}:
  unit_row_room(U32.to_nat(rows(query)),unit,
    Order.transitive(Nat.mul(1n+unit,4n),Nat.mul(units(query),4n),Nat.add(U32.to_nat(rows(query)),3n),
      RowMajor.mul_le(1n+unit,units(query),4n,Order.strict_follow(unit,units(query),inside)),units_room(query)))

def unit_before(+rows: Nat,+count: Nat,+unit: Nat,+remainder: Nat,
  +unit_inside: {Nat.is_lt(Nat.mul(unit,4n),rows) == True{} : Bool},
  +identity: {Nat.add(rows,3n) == Nat.add(Nat.mul(count,4n),remainder) : Nat},+remainder_small: {Nat.is_lt(remainder,4n) == True{} : Bool}) ->
  {Nat.is_lt(unit,count) == True{} : Bool}:
  RowMajor.cancel_multiply_lt(1n+unit,1n+count,4n,
    Order.le_lt_trans(Nat.mul(1n+unit,4n),Nat.add(rows,3n),Nat.mul(1n+count,4n),
      %Order.commutative(3n,rows) : {Nat.is_le(Nat.mul(1n+unit,4n),_) == True{} : Bool}
      Order.strict_follow(Nat.mul(unit,4n),rows,unit_inside),
      %Equal.sym(Nat,Nat.add(rows,3n),Nat.add(Nat.mul(count,4n),remainder),identity) : {Nat.is_lt(_,Nat.mul(1n+count,4n)) == True{} : Bool}
      %Order.commutative(remainder,Nat.mul(count,4n)) : {Nat.is_lt(_,Nat.mul(1n+count,4n)) == True{} : Bool}
      Order.add_lt(remainder,4n,Nat.mul(count,4n),remainder_small)))

def unit_counted(+query: Query,+unit: Nat,+unit_rows: {Nat.is_lt(Nat.mul(unit,4n),U32.to_nat(rows(query))) == True{} : Bool}) ->
  {Nat.is_lt(unit,units(query)) == True{} : Bool}:
  +padded = Nat.add(U32.to_nat(rows(query)),3n)
  unit_before(U32.to_nat(rows(query)),units(query),unit,Nat.mod(padded,4n),unit_rows,
    decomposed(padded,4n,limit(query),limit_fits(query),rows_room(query),{==}),
    remainder_small(padded,4n,limit(query),limit_fits(query),rows_room(query),{==}))

def units_positive(+query: Query) -> {Nat.is_lt(0n,units(query)) == True{} : Bool}:
  +padded = Nat.add(U32.to_nat(rows(query)),3n)
  unit_before(U32.to_nat(rows(query)),units(query),0n,Nat.mod(padded,4n),rows_positive(query),
    decomposed(padded,4n,limit(query),limit_fits(query),rows_room(query),{==}),
    remainder_small(padded,4n,limit(query),limit_fits(query),rows_room(query),{==}))

def double_units(+query: Query) -> {Nat.is_le(Nat.double(units(query)),U32.to_nat(limit(query))) == True{} : Bool}:
  Order.transitive(Nat.double(units(query)),Nat.mul(units(query),4n),U32.to_nat(limit(query)),double_within_four(units(query)),
    Order.transitive(Nat.mul(units(query),4n),Nat.add(U32.to_nat(rows(query)),3n),U32.to_nat(limit(query)),units_room(query),rows_room(query)))

def jobs_within(+query: Query) -> {Nat.is_le(Nat.mul(groups(query),units(query)),U32.to_nat(limit(query))) == True{} : Bool}:
  +rows_nat = U32.to_nat(rows(query))
  +width = U32.to_nat(columns(query))
  +count = units(query)
  Order.transitive(Nat.mul(groups(query),count),Nat.mul(rows_nat,width),U32.to_nat(limit(query)),
    Order.transitive(Nat.mul(groups(query),count),Nat.mul(width,count),Nat.mul(rows_nat,width),
      RowMajor.mul_le(groups(query),width,count,groups_within_columns(query)),
      %RowMajor.multiply_commutative(count,width) : {Nat.is_le(_,Nat.mul(rows_nat,width)) == True{} : Bool}
      RowMajor.mul_le(count,rows_nat,width,units_within_rows(rows_nat,count,units_room(query)))),
    area_room(query))

def index_exact(+query: Query,+unit: Nat,+group: Nat,
  unit_inside: {Nat.is_lt(unit,units(query)) == True{} : Bool},group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(job_index(unit,group,units(query)))) == job_index(unit,group,units(query)) : Nat}:
  Bounds.roundtrip(job_index(unit,group,units(query)),limit(query),limit_fits(query),
    Order.transitive(job_index(unit,group,units(query)),Nat.mul(groups(query),units(query)),U32.to_nat(limit(query)),
      Bounds.strict_le(job_index(unit,group,units(query)),Nat.mul(groups(query),units(query)),
        RowMajor.radix(group,groups(query),unit,units(query),group_inside,unit_inside)),
      jobs_within(query)))

def unit_word(+query: Query,unit: Nat,group: Nat) -> U32:
  (U32.from_nat(job_index(unit,group,units(query))) % Gemm.row_units(rows(query)) : U32)

def group_word(+query: Query,unit: Nat,group: Nat) -> U32:
  (U32.from_nat(job_index(unit,group,units(query))) / Gemm.row_units(rows(query)) : U32)

def unit_exact(+query: Query,+unit: Nat,+group: Nat,
  +unit_inside: {Nat.is_lt(unit,units(query)) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool}) ->
  {U32.to_nat(unit_word(query,unit,group)) == unit : Nat}:
  Radix.remainder_word(U32.from_nat(job_index(unit,group,units(query))),Gemm.row_units(rows(query)),group,unit,units(query),limit(query),limit_fits(query),
    index_exact(query,unit,group,unit_inside,group_inside),units_word_exact(query),units_positive(query),unit_inside,double_units(query))

def group_exact(+query: Query,+unit: Nat,+group: Nat,
  +unit_inside: {Nat.is_lt(unit,units(query)) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool}) ->
  {U32.to_nat(group_word(query,unit,group)) == group : Nat}:
  Radix.quotient_word(U32.from_nat(job_index(unit,group,units(query))),Gemm.row_units(rows(query)),group,unit,units(query),limit(query),limit_fits(query),
    index_exact(query,unit,group,unit_inside,group_inside),units_word_exact(query),units_positive(query),unit_inside,double_units(query))

def first_row_exact(+query: Query,+unit: Nat,+word: U32,+exact: {U32.to_nat(word) == unit : Nat},
  +unit_rows: {Nat.is_lt(Nat.mul(unit,4n),U32.to_nat(rows(query))) == True{} : Bool}) ->
  {U32.to_nat((word * 4 : U32)) == Nat.mul(unit,4n) : Nat}:
  %exact : {U32.to_nat((word * 4 : U32)) == Nat.mul(_,4n) : Nat}
  Arithmetic.multiply(word,4,limit(query),limit_fits(query),
    %Equal.sym(Nat,U32.to_nat(word),unit,exact) : {Nat.is_le(Nat.mul(_,4n),U32.to_nat(limit(query))) == True{} : Bool}
    Order.transitive(Nat.mul(unit,4n),U32.to_nat(rows(query)),U32.to_nat(limit(query)),Bounds.strict_le(Nat.mul(unit,4n),U32.to_nat(rows(query)),unit_rows),
      Order.transitive(U32.to_nat(rows(query)),Nat.add(U32.to_nat(rows(query)),3n),U32.to_nat(limit(query)),Order.le_add(U32.to_nat(rows(query)),3n),rows_room(query))))

def row_zero_exact(+query: Query,+unit: Nat,+word: U32,+exact: {U32.to_nat(word) == unit : Nat},
  +unit_rows: {Nat.is_lt(Nat.mul(unit,4n),U32.to_nat(rows(query))) == True{} : Bool}) ->
  {U32.to_nat((word * 4 : U32)) == row_number(unit,0n) : Nat}:
  %Equal.sym(Nat,Nat.add(Nat.mul(unit,4n),0n),Nat.mul(unit,4n),Addition.zero_right(Nat.mul(unit,4n))) : {U32.to_nat((word * 4 : U32)) == _ : Nat}
  first_row_exact(query,unit,word,exact,unit_rows)

def row_offset_exact(+query: Query,+unit: Nat,+word: U32,+row: Nat,+exact: {U32.to_nat(word) == unit : Nat},
  +unit_rows: {Nat.is_lt(Nat.mul(unit,4n),U32.to_nat(rows(query))) == True{} : Bool},+row_bound: {Nat.is_le(row,3n) == True{} : Bool}) ->
  {U32.to_nat(((word * 4 : U32) + U32.from_nat(row) : U32)) == row_number(unit,row) : Nat}:
  +rows_nat = U32.to_nat(rows(query))
  +product = (word * 4 : U32)
  +row_exact = Bounds.roundtrip(row,limit(query),limit_fits(query),
    Order.transitive(row,3n,U32.to_nat(limit(query)),row_bound,
      Order.transitive(3n,Nat.add(rows_nat,3n),U32.to_nat(limit(query)),
        %Order.commutative(3n,rows_nat) : {Nat.is_le(3n,_) == True{} : Bool}
        Order.le_add(3n,rows_nat),rows_room(query))))
  +product_exact = first_row_exact(query,unit,word,exact,unit_rows)
  %product_exact : {U32.to_nat((product + U32.from_nat(row) : U32)) == Nat.add(_,row) : Nat}
  %row_exact : {U32.to_nat((product + U32.from_nat(row) : U32)) == Nat.add(U32.to_nat(product),_) : Nat}
  Arithmetic.add(product,U32.from_nat(row),limit(query),limit_fits(query),
    %Equal.sym(Nat,U32.to_nat(product),Nat.mul(unit,4n),product_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(U32.from_nat(row))),U32.to_nat(limit(query))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(row)),row,row_exact) : {Nat.is_le(Nat.add(Nat.mul(unit,4n),_),U32.to_nat(limit(query))) == True{} : Bool}
    Order.transitive(Nat.add(Nat.mul(unit,4n),row),Nat.add(rows_nat,row),U32.to_nat(limit(query)),
      Order.add_le(Nat.mul(unit,4n),rows_nat,row,Bounds.strict_le(Nat.mul(unit,4n),rows_nat,unit_rows)),
      Order.transitive(Nat.add(rows_nat,row),Nat.add(rows_nat,3n),U32.to_nat(limit(query)),Order.offset_le(rows_nat,row,3n,row_bound),rows_room(query))))

def position_exact(+query: Query,+group: Nat,+word: U32,+exact: {U32.to_nat(word) == group : Nat},
  +group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool}) ->
  {U32.to_nat((word * 8 : U32)) == Nat.mul(group,8n) : Nat}:
  %exact : {U32.to_nat((word * 8 : U32)) == Nat.mul(_,8n) : Nat}
  Arithmetic.multiply(word,8,limit(query),limit_fits(query),
    %Equal.sym(Nat,U32.to_nat(word),group,exact) : {Nat.is_le(Nat.mul(_,8n),U32.to_nat(limit(query))) == True{} : Bool}
    Order.transitive(Nat.mul(group,8n),U32.to_nat(columns(query)),U32.to_nat(limit(query)),
      %Equal.sym(Nat,U32.to_nat(columns(query)),Nat.mul(groups(query),8n),columns_exact(query)) : {Nat.is_le(Nat.mul(group,8n),_) == True{} : Bool}
      RowMajor.mul_le(group,groups(query),8n,Bounds.strict_le(group,groups(query),group_inside)),columns_within(query)))

def job_writes(~biased: Bool,+query: Query,+unit: Nat,+group: Nat) -> List<&2,Writes.Assignment<F32>>:
  GemmWrites.job_writes(~biased,Gemm.job(layout(query),U32.from_nat(job_index(unit,group,units(query)))),input(query),weights(query),bias(query))

type IndexedJob<-query: Query,-index: Nat> is Data:
  IndexedJob{unit: Nat,group: Nat,
    unit_rows: {Nat.is_lt(Nat.mul(unit,4n),U32.to_nat(rows(query))) == True{} : Bool},
    unit_inside: {Nat.is_lt(unit,units(query)) == True{} : Bool},
    group_inside: {Nat.is_lt(group,groups(query)) == True{} : Bool},
    index_exact: {index == job_index(unit,group,units(query)) : Nat}}

def indexed_job(+query: Query,+index: Nat,
  +inside: {Nat.is_lt(index,Nat.mul(groups(query),units(query))) == True{} : Bool}) -> IndexedJob<query,index>:
  +count = units(query)
  +small = Order.transitive(index,Nat.mul(groups(query),count),U32.to_nat(limit(query)),Bounds.strict_le(index,Nat.mul(groups(query),count),inside),jobs_within(query))
  +unit = Nat.mod(index,count)
  +group = Nat.div(index,count)
  +identity = decomposed(index,count,limit(query),limit_fits(query),small,units_positive(query))
  +unit_inside = remainder_small(index,count,limit(query),limit_fits(query),small,units_positive(query))
  +group_inside = RowMajor.row_inside(group,groups(query),count,unit,
    %identity : {Nat.is_lt(_,Nat.mul(groups(query),count)) == True{} : Bool}
    inside)
  IndexedJob{unit,group,unit_bound(query,unit,unit_inside),unit_inside,group_inside,identity}

def step_after(+first: Nat,+target: Nat,after: {Nat.is_le(first,target) == True{} : Bool},differ: {Nat.is_eq(first,target) == False{} : Bool}) ->
  {Nat.is_le(1n+first,target) == True{} : Bool}:
  match first target:
    case 0n 0n: Empty.absurd({Nat.is_le(1n,0n) == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},differ)))
    case 0n 1n+ +rest: Order.zero_le(rest)
    case 1n+ +rest 0n: Empty.absurd({Nat.is_le(2n+rest,0n) == True{} : Bool},WordEncoding.false_true(after))
    case 1n+ +rest 1n+ +other: step_after(rest,other,after,differ)

def layout_exact(+query: Query) -> {layout(query) == Gemm.Layout{rows(query),columns(query),inner(query),0,columns(query)} : Gemm.Layout}:
  Query{rows,columns,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside} = query
  {==}

def split_address(+query: Query,+unit: Nat,+row: Nat,+group: Nat,+lane: Nat,
  +channel_split: {channel(query) == row_number(unit,row) : Nat},+column_split: {column(query) == lane_number(group,lane) : Nat}) ->
  {address(query) == Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,lane)) : Nat}:
  %Equal.sym(Nat,channel(query),row_number(unit,row),channel_split) :
    {Nat.add(Nat.mul(_,U32.to_nat(columns(query))),column(query)) == Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,lane)) : Nat}
  %Equal.sym(Nat,column(query),lane_number(group,lane),column_split) :
    {Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),_) == Nat.add(Nat.mul(row_number(unit,row),U32.to_nat(columns(query))),lane_number(group,lane)) : Nat}
  {==}

def split_row_inside(+query: Query,+unit: Nat,+row: Nat,+channel_split: {channel(query) == row_number(unit,row) : Nat}) ->
  {Nat.is_lt(row_number(unit,row),U32.to_nat(rows(query))) == True{} : Bool}:
  %channel_split : {Nat.is_lt(_,U32.to_nat(rows(query))) == True{} : Bool}
  channel_inside(query)

def split_unit_inside(+query: Query,+unit: Nat,+row: Nat,+channel_split: {channel(query) == row_number(unit,row) : Nat}) ->
  {Nat.is_lt(Nat.mul(unit,4n),U32.to_nat(rows(query))) == True{} : Bool}:
  Order.le_lt_trans(Nat.mul(unit,4n),channel(query),U32.to_nat(rows(query)),
    %Equal.sym(Nat,channel(query),row_number(unit,row),channel_split) : {Nat.is_le(Nat.mul(unit,4n),_) == True{} : Bool}
    Order.le_add(Nat.mul(unit,4n),row),channel_inside(query))

def split_group_inside(+query: Query,+group: Nat,+lane: Nat,+column_split: {column(query) == lane_number(group,lane) : Nat}) ->
  {Nat.is_lt(group,groups(query)) == True{} : Bool}:
  RowMajor.row_inside(group,groups(query),8n,lane,
    %column_split : {Nat.is_lt(_,Nat.mul(groups(query),8n)) == True{} : Bool}
    %columns_exact(query) : {Nat.is_lt(column(query),_) == True{} : Bool}
    column_inside(query))

def count_exact(+query: Query) -> {GemmWrites.job_count(rows(query),columns(query)) == Nat.mul(groups(query),units(query)) : Nat}:
  +group_count = Gemm.column_groups(columns(query))
  +unit_count = Gemm.row_units(rows(query))
  %groups_word_exact(query) : {GemmWrites.job_count(rows(query),columns(query)) == Nat.mul(_,units(query)) : Nat}
  %units_word_exact(query) : {GemmWrites.job_count(rows(query),columns(query)) == Nat.mul(U32.to_nat(group_count),_) : Nat}
  Arithmetic.multiply(group_count,unit_count,limit(query),limit_fits(query),
    %Equal.sym(Nat,U32.to_nat(group_count),groups(query),groups_word_exact(query)) : {Nat.is_le(Nat.mul(_,U32.to_nat(unit_count)),U32.to_nat(limit(query))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(unit_count),units(query),units_word_exact(query)) : {Nat.is_le(Nat.mul(groups(query),_),U32.to_nat(limit(query))) == True{} : Bool}
    jobs_within(query))

# A job's selected scalar is its existing ordered reduction at the same bases.
def job_dot(job: Gemm.Job,row: Accumulator.Row,+lane: Nat,input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>) -> F32:
  Gemm.Job{Gemm.Context{first,position,rows,columns,input_base,weight_base},upper,count} = job
  ReductionModel.ordered_selected(upper,ReductionModel.RowLane{row,TileLaws.lane_of(lane)},count,ReductionModel.initial(input,weights,input_base,0,weight_base))

def job_dot_matches(+job: Gemm.Job,+row: Accumulator.Row,+lane: Nat,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>) ->
  {select_lane(lane,Accumulator.row(~Tile.Tile8,row,accumulators(job,input,weights))) == job_dot(job,row,lane,input,weights) : F32}:
  Gemm.Job{Gemm.Context{first,position,rows,columns,+input_base,+weight_base},+upper,+count} = job
  +key = {ReductionModel.RowLane{row,TileLaws.lane_of(lane)} : ReductionModel.RowLane}
  +state = ReductionModel.initial(input,weights,input_base,0,weight_base)
  Equal.trans(F32,select_lane(lane,Accumulator.row(~Tile.Tile8,row,accumulators(job,input,weights))),ReductionModel.accumulator_at(key,Reduction.accumulator(~&2,~Tree.Tree<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~U32,~Rows.Rows<Tile.Tile8>,ReductionModel.selected(upper,count,state))),job_dot(job,row,lane,input,weights),
    TileLaws.selected_lane(lane,Accumulator.row(~Tile.Tile8,row,accumulators(job,input,weights))),ReductionModel.selected_ordered(upper,key,count,state))

def row_dot(+job: Gemm.Job,+row: Nat,+lane: Nat,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>) -> F32:
  job_dot(job,Accumulator.row_of(row),lane,input,weights)

def row_dot_matches(+job: Gemm.Job,+row: Nat,+lane: Nat,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>) ->
  {select_lane(lane,Accumulator.select_row(~Tile.Tile8,row,accumulators(job,input,weights))) == row_dot(job,row,lane,input,weights) : F32}:
  %Equal.sym(Tile.Tile8,Accumulator.select_row(~Tile.Tile8,row,accumulators(job,input,weights)),Accumulator.row(~Tile.Tile8,Accumulator.row_of(row),accumulators(job,input,weights)),Accumulator.selection_matches(~Tile.Tile8,row,accumulators(job,input,weights))) :
    {select_lane(lane,_) == row_dot(job,row,lane,input,weights) : F32}
  job_dot_matches(job,Accumulator.row_of(row),lane,input,weights)

def wanted_ordered(~biased: Bool,+query: Query) -> F32:
  +job = Gemm.job(layout(query),U32.from_nat(job_index(Nat.div(channel(query),4n),Nat.div(column(query),8n),units(query))))
  +dot = row_dot(job,Nat.mod(channel(query),4n),Nat.mod(column(query),8n),input(query),weights(query))
  Bool.pick(F32,biased,(dot + Tree.read(F32,bias(query),U32.from_nat(channel(query))) : F32),dot)

def wanted_as_ordered(~biased: Bool,+query: Query) -> {wanted(~biased,query) == wanted_ordered(~biased,query) : F32}:
  +job = Gemm.job(layout(query),U32.from_nat(job_index(Nat.div(channel(query),4n),Nat.div(column(query),8n),units(query))))
  Equal.cong(F32,F32,value => Bool.pick(F32,biased,(value + Tree.read(F32,bias(query),U32.from_nat(channel(query))) : F32),value),
    select_lane(Nat.mod(column(query),8n),Accumulator.select_row(~Tile.Tile8,Nat.mod(channel(query),4n),accumulators(job,input(query),weights(query)))),
    row_dot(job,Nat.mod(channel(query),4n),Nat.mod(column(query),8n),input(query),weights(query)),row_dot_matches(job,Nat.mod(channel(query),4n),Nat.mod(column(query),8n),input(query),weights(query)))

def row_number_of(+row: Nat,small: {Nat.is_lt(row,4n) == True{} : Bool}) -> {CoordinateLaws.row_number(Accumulator.row_of(row)) == row : Nat}:
  match row:
    case 0n: {==}
    case 1n+a:
      match a:
        case 0n: {==}
        case 1n+b:
          match b:
            case 0n: {==}
            case 1n+c:
              match c:
                case 0n: {==}
                case 1n+d:
                  Empty.absurd({CoordinateLaws.row_number(Accumulator.row_of(4n+d)) == 4n+d : Nat},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(d,0n),True{},Equal.sym(Bool,Nat.is_lt(d,0n),False{},Division.lt_zero(d)),small)))

def lane_value_of(+lane: Nat,small: {Nat.is_lt(lane,8n) == True{} : Bool}) -> {Sequence.index_value(8n,TileLaws.lane_of(lane)) == lane : Nat}:
  match lane:
    case 0n: {==}
    case 1n+a:
      match a:
        case 0n: {==}
        case 1n+b:
          match b:
            case 0n: {==}
            case 1n+c:
              match c:
                case 0n: {==}
                case 1n+d:
                  match d:
                    case 0n: {==}
                    case 1n+e:
                      match e:
                        case 0n: {==}
                        case 1n+f:
                          match f:
                            case 0n: {==}
                            case 1n+g:
                              match g:
                                case 0n: {==}
                                case 1n+h:
                                  Empty.absurd({Sequence.index_value(8n,TileLaws.lane_of(8n+h)) == 8n+h : Nat},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(h,0n),True{},Equal.sym(Bool,Nat.is_lt(h,0n),False{},Division.lt_zero(h)),small)))

def eight_split(+channel: Nat,+unit: Nat,+row: Nat,+block: Nat,+parity: Nat,
  +channel_split: {channel == Nat.add(Nat.mul(unit,4n),row) : Nat},+unit_split: {unit == Nat.add(Nat.mul(block,2n),parity) : Nat}) ->
  {channel == Nat.add(Nat.mul(block,8n),Nat.add(Nat.mul(parity,4n),row)) : Nat}:
  %Addition.associative(Nat.mul(block,8n),Nat.mul(parity,4n),row) : {channel == _ : Nat}
  %RowMajor.assoc(block,2n,4n) : {channel == Nat.add(Nat.add(_,Nat.mul(parity,4n)),row) : Nat}
  %RowMajor.distrib(Nat.mul(block,2n),parity,4n) : {channel == Nat.add(_,row) : Nat}
  %unit_split : {channel == Nat.add(Nat.mul(_,4n),row) : Nat}
  channel_split

def parity_exact(+unit_word: U32,+unit: Nat,+unit_exact: {U32.to_nat(unit_word) == unit : Nat}) ->
  {U32.is_eq((unit_word % 2 : U32),1) == Nat.is_eq(Nat.mod(unit,2n),1n) : Bool}:
  Equal.trans(Bool,U32.is_eq((unit_word % 2 : U32),1),Nat.is_eq(U32.to_nat((unit_word % 2 : U32)),1n),Nat.is_eq(Nat.mod(unit,2n),1n),
    Comparison.eq((unit_word % 2 : U32),1),
    Equal.cong(Nat,Bool,value => Nat.is_eq(value,1n),U32.to_nat((unit_word % 2 : U32)),Nat.mod(unit,2n),
      Equal.trans(Nat,U32.to_nat((unit_word % 2 : U32)),Nat.mod(U32.to_nat(unit_word),2n),Nat.mod(unit,2n),
        U32Division.modulo(unit_word,2,2147483648,{==},{==},{==}),
        Equal.cong(Nat,Nat,value => Nat.mod(value,2n),U32.to_nat(unit_word),unit,unit_exact))))

# Concrete coordinates of the one job containing this query's point.
def selected_channel_room(+query: Query) -> {Nat.is_le(channel(query),U32.to_nat(limit(query))) == True{} : Bool}:
  Order.transitive(channel(query),U32.to_nat(rows(query)),U32.to_nat(limit(query)),Bounds.strict_le(channel(query),U32.to_nat(rows(query)),channel_inside(query)),
    Order.transitive(U32.to_nat(rows(query)),Nat.add(U32.to_nat(rows(query)),3n),U32.to_nat(limit(query)),Order.le_add(U32.to_nat(rows(query)),3n),rows_room(query)))

def selected_column_room(+query: Query) -> {Nat.is_le(column(query),U32.to_nat(limit(query))) == True{} : Bool}:
  Order.transitive(column(query),U32.to_nat(columns(query)),U32.to_nat(limit(query)),Bounds.strict_le(column(query),U32.to_nat(columns(query)),column_inside(query)),columns_within(query))

def selected_channel_split(+query: Query) -> {channel(query) == row_number(Nat.div(channel(query),4n),Nat.mod(channel(query),4n)) : Nat}:
  decomposed(channel(query),4n,limit(query),limit_fits(query),selected_channel_room(query),{==})

def selected_column_split(+query: Query) -> {column(query) == lane_number(Nat.div(column(query),8n),Nat.mod(column(query),8n)) : Nat}:
  decomposed(column(query),8n,limit(query),limit_fits(query),selected_column_room(query),{==})

def selected_unit_inside(+query: Query) -> {Nat.is_lt(Nat.div(channel(query),4n),units(query)) == True{} : Bool}:
  unit_counted(query,Nat.div(channel(query),4n),split_unit_inside(query,Nat.div(channel(query),4n),Nat.mod(channel(query),4n),selected_channel_split(query)))

def selected_group_inside(+query: Query) -> {Nat.is_lt(Nat.div(column(query),8n),groups(query)) == True{} : Bool}:
  split_group_inside(query,Nat.div(column(query),8n),Nat.mod(column(query),8n),selected_column_split(query))

def selected_unit_exact(+query: Query) -> {U32.to_nat(unit_word(query,Nat.div(channel(query),4n),Nat.div(column(query),8n))) == Nat.div(channel(query),4n) : Nat}:
  unit_exact(query,Nat.div(channel(query),4n),Nat.div(column(query),8n),selected_unit_inside(query),selected_group_inside(query))

def selected_group_exact(+query: Query) -> {U32.to_nat(group_word(query,Nat.div(channel(query),4n),Nat.div(column(query),8n))) == Nat.div(column(query),8n) : Nat}:
  group_exact(query,Nat.div(channel(query),4n),Nat.div(column(query),8n),selected_unit_inside(query),selected_group_inside(query))

def selected_row_small(+query: Query) -> {Nat.is_lt(Nat.mod(channel(query),4n),4n) == True{} : Bool}:
  remainder_small(channel(query),4n,limit(query),limit_fits(query),selected_channel_room(query),{==})

def selected_lane_small(+query: Query) -> {Nat.is_lt(Nat.mod(column(query),8n),8n) == True{} : Bool}:
  remainder_small(column(query),8n,limit(query),limit_fits(query),selected_column_room(query),{==})

def selected_unit_room(+query: Query) -> {Nat.is_le(Nat.div(channel(query),4n),U32.to_nat(limit(query))) == True{} : Bool}:
  Order.transitive(Nat.div(channel(query),4n),channel(query),U32.to_nat(limit(query)),
    Order.transitive(Nat.div(channel(query),4n),Nat.mul(Nat.div(channel(query),4n),4n),channel(query),RowMajor.product_covers(Nat.div(channel(query),4n),4n,{==}),
      %Equal.sym(Nat,channel(query),row_number(Nat.div(channel(query),4n),Nat.mod(channel(query),4n)),selected_channel_split(query)) : {Nat.is_le(Nat.mul(Nat.div(channel(query),4n),4n),_) == True{} : Bool}
      Order.le_add(Nat.mul(Nat.div(channel(query),4n),4n),Nat.mod(channel(query),4n))),selected_channel_room(query))

def selected_unit_split(+query: Query) -> {Nat.div(channel(query),4n) == Nat.add(Nat.mul(Nat.div(Nat.div(channel(query),4n),2n),2n),Nat.mod(Nat.div(channel(query),4n),2n)) : Nat}:
  decomposed(Nat.div(channel(query),4n),2n,limit(query),limit_fits(query),selected_unit_room(query),{==})

def selected_parity_small(+query: Query) -> {Nat.is_lt(Nat.mod(Nat.div(channel(query),4n),2n),2n) == True{} : Bool}:
  remainder_small(Nat.div(channel(query),4n),2n,limit(query),limit_fits(query),selected_unit_room(query),{==})

def selected_eight_split(+query: Query) -> {channel(query) == Nat.add(Nat.mul(Nat.div(Nat.div(channel(query),4n),2n),8n),Nat.add(Nat.mul(Nat.mod(Nat.div(channel(query),4n),2n),4n),Nat.mod(channel(query),4n))) : Nat}:
  eight_split(channel(query),Nat.div(channel(query),4n),Nat.mod(channel(query),4n),Nat.div(Nat.div(channel(query),4n),2n),Nat.mod(Nat.div(channel(query),4n),2n),selected_channel_split(query),selected_unit_split(query))

def half_row_inside(+parity: Nat,+row: Nat,parity_small: {Nat.is_lt(parity,2n) == True{} : Bool},row_small: {Nat.is_lt(row,4n) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(Nat.mul(parity,4n),row),8n) == True{} : Bool}:
  match parity:
    case 0n: Order.le_lt_trans(row,4n,8n,Bounds.strict_le(row,4n,row_small),{==})
    case 1n+rest:
      match rest:
        case 0n: row_small
        case 1n+more: Empty.absurd({Nat.is_lt(Nat.add(Nat.mul(2n+more,4n),row),8n) == True{} : Bool},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(more,0n),True{},Equal.sym(Bool,Nat.is_lt(more,0n),False{},Division.lt_zero(more)),parity_small)))

def selected_block_exact(+query: Query) -> {Nat.div(channel(query),8n) == Nat.div(Nat.div(channel(query),4n),2n) : Nat}:
  %Equal.sym(Nat,channel(query),Nat.add(Nat.mul(Nat.div(Nat.div(channel(query),4n),2n),8n),Nat.add(Nat.mul(Nat.mod(Nat.div(channel(query),4n),2n),4n),Nat.mod(channel(query),4n))),selected_eight_split(query)) :
    {Nat.div(_,8n) == Nat.div(Nat.div(channel(query),4n),2n) : Nat}
  Radix.quotient(Nat.div(Nat.div(channel(query),4n),2n),Nat.add(Nat.mul(Nat.mod(Nat.div(channel(query),4n),2n),4n),Nat.mod(channel(query),4n)),8n,{==},half_row_inside(Nat.mod(Nat.div(channel(query),4n),2n),Nat.mod(channel(query),4n),selected_parity_small(query),selected_row_small(query)))

def selected_channel_remainder(+query: Query) -> {Nat.mod(channel(query),8n) == Nat.add(Nat.mul(Nat.mod(Nat.div(channel(query),4n),2n),4n),Nat.mod(channel(query),4n)) : Nat}:
  %Equal.sym(Nat,channel(query),Nat.add(Nat.mul(Nat.div(Nat.div(channel(query),4n),2n),8n),Nat.add(Nat.mul(Nat.mod(Nat.div(channel(query),4n),2n),4n),Nat.mod(channel(query),4n))),selected_eight_split(query)) :
    {Nat.mod(_,8n) == Nat.add(Nat.mul(Nat.mod(Nat.div(channel(query),4n),2n),4n),Nat.mod(channel(query),4n)) : Nat}
  Radix.remainder(Nat.div(Nat.div(channel(query),4n),2n),Nat.add(Nat.mul(Nat.mod(Nat.div(channel(query),4n),2n),4n),Nat.mod(channel(query),4n)),8n,{==},half_row_inside(Nat.mod(Nat.div(channel(query),4n),2n),Nat.mod(channel(query),4n),selected_parity_small(query),selected_row_small(query)))

def parity_lane(+upper: Bool,+parity: Nat,+row: Nat,+upper_exact: {upper == Nat.is_eq(parity,1n) : Bool},+parity_small: {Nat.is_lt(parity,2n) == True{} : Bool},+row_small: {Nat.is_lt(row,4n) == True{} : Bool}) ->
  {CoordinateLaws.weight_lane(upper,Accumulator.row_of(row)) == TileLaws.lane_of(Nat.add(Nat.mul(parity,4n),row)) : Sequence.Index(8n)}:
  match parity:
    case 0n:
      %Equal.sym(Bool,upper,False{},upper_exact) : {CoordinateLaws.weight_lane(_,Accumulator.row_of(row)) == TileLaws.lane_of(row) : Sequence.Index(8n)}
      %row_number_of(row,row_small) : {CoordinateLaws.weight_lane(False{},Accumulator.row_of(row)) == TileLaws.lane_of(_) : Sequence.Index(8n)}
      CoordinateLaws.weight_lane_matches(False{},Accumulator.row_of(row))
    case 1n+rest:
      match rest:
        case 0n:
          %Equal.sym(Bool,upper,True{},upper_exact) : {CoordinateLaws.weight_lane(_,Accumulator.row_of(row)) == TileLaws.lane_of(4n+row) : Sequence.Index(8n)}
          %row_number_of(row,row_small) : {CoordinateLaws.weight_lane(True{},Accumulator.row_of(row)) == TileLaws.lane_of(4n+_) : Sequence.Index(8n)}
          CoordinateLaws.weight_lane_matches(True{},Accumulator.row_of(row))
        case 1n+more: Empty.absurd({CoordinateLaws.weight_lane(upper,Accumulator.row_of(row)) == TileLaws.lane_of(Nat.add(Nat.mul(2n+more,4n),row)) : Sequence.Index(8n)},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(more,0n),True{},Equal.sym(Bool,Nat.is_lt(more,0n),False{},Division.lt_zero(more)),parity_small)))

def selected_weight_lane(+query: Query) ->
  {CoordinateLaws.weight_lane(U32.is_eq((unit_word(query,Nat.div(channel(query),4n),Nat.div(column(query),8n)) % 2 : U32),1),Accumulator.row_of(Nat.mod(channel(query),4n))) == TileLaws.lane_of(Nat.mod(channel(query),8n)) : Sequence.Index(8n)}:
  %Equal.sym(Nat,Nat.mod(channel(query),8n),Nat.add(Nat.mul(Nat.mod(Nat.div(channel(query),4n),2n),4n),Nat.mod(channel(query),4n)),selected_channel_remainder(query)) :
    {CoordinateLaws.weight_lane(U32.is_eq((unit_word(query,Nat.div(channel(query),4n),Nat.div(column(query),8n)) % 2 : U32),1),Accumulator.row_of(Nat.mod(channel(query),4n))) == TileLaws.lane_of(_) : Sequence.Index(8n)}
  parity_lane(U32.is_eq((unit_word(query,Nat.div(channel(query),4n),Nat.div(column(query),8n)) % 2 : U32),1),Nat.mod(Nat.div(channel(query),4n),2n),Nat.mod(channel(query),4n),parity_exact(unit_word(query,Nat.div(channel(query),4n),Nat.div(column(query),8n)),Nat.div(channel(query),4n),selected_unit_exact(query)),selected_parity_small(query),selected_row_small(query))

# ---- Readback through the job views ----

# A run that stores its columns from start on, in an output of out columns.
# The last group may run up to seven padding lanes past the output.
type OutputWindow is Data:
  OutputWindow{query: Query,start: U32,out: U32,
    end_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(columns(query))),Nat.add(U32.to_nat(out),7n)) == True{} : Bool},
    out_room: {Nat.is_le(U32.to_nat(out),U32.to_nat(limit(query))) == True{} : Bool}}
def window_query(+window: OutputWindow) -> Query:
  OutputWindow{query,start,out,end_room,out_room} = window
  query
def window_start(+window: OutputWindow) -> U32:
  OutputWindow{query,start,out,end_room,out_room} = window
  start
def window_out(+window: OutputWindow) -> U32:
  OutputWindow{query,start,out,end_room,out_room} = window
  out
def window_end_room(+window: OutputWindow) ->
  {Nat.is_le(Nat.add(U32.to_nat(window_start(window)),U32.to_nat(columns(window_query(window)))),Nat.add(U32.to_nat(window_out(window)),7n)) == True{} : Bool}:
  OutputWindow{query,start,out,end_room,out_room} = window
  end_room
def window_out_room(+window: OutputWindow) -> {Nat.is_le(U32.to_nat(window_out(window)),U32.to_nat(limit(window_query(window)))) == True{} : Bool}:
  OutputWindow{query,start,out,end_room,out_room} = window
  out_room
def window_layout(+window: OutputWindow) -> Gemm.Layout:
  +query = window_query(window)
  Gemm.Layout{rows(query),columns(query),inner(query),window_start(window),window_out(window)}

# An output index of row channel and the given column, and an output
# capacity that holds every write of the run.
type Probe is Data:
  Probe{window: OutputWindow,capacity: U32,capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
    output_room: {Nat.is_le(Nat.mul(U32.to_nat(rows(window_query(window))),U32.to_nat(window_out(window))),U32.to_nat(capacity)) == True{} : Bool},
    column: Nat,column_inside: {Nat.is_lt(column,U32.to_nat(window_out(window))) == True{} : Bool}}
def probe_window(+probe: Probe) -> OutputWindow:
  Probe{window,capacity,capacity_ok,output_room,column,column_inside} = probe
  window
def probe_capacity(+probe: Probe) -> U32:
  Probe{window,capacity,capacity_ok,output_room,column,column_inside} = probe
  capacity
def probe_capacity_ok(+probe: Probe) -> {U32.is_le(probe_capacity(probe),2147483648) == True{} : Bool}:
  Probe{window,capacity,capacity_ok,output_room,column,column_inside} = probe
  capacity_ok
def probe_output_room(+probe: Probe) ->
  {Nat.is_le(Nat.mul(U32.to_nat(rows(window_query(probe_window(probe)))),U32.to_nat(window_out(probe_window(probe)))),U32.to_nat(probe_capacity(probe))) == True{} : Bool}:
  Probe{window,capacity,capacity_ok,output_room,column,column_inside} = probe
  output_room
def probe_column(+probe: Probe) -> Nat:
  Probe{window,capacity,capacity_ok,output_room,column,column_inside} = probe
  column
def probe_column_inside(+probe: Probe) -> {Nat.is_lt(probe_column(probe),U32.to_nat(window_out(probe_window(probe)))) == True{} : Bool}:
  Probe{window,capacity,capacity_ok,output_room,column,column_inside} = probe
  column_inside
def probe_query(+probe: Probe) -> Query:
  window_query(probe_window(probe))
def probe_number(+probe: Probe) -> Nat:
  Nat.add(Nat.mul(channel(probe_query(probe)),U32.to_nat(window_out(probe_window(probe)))),probe_column(probe))
def probe_word(+probe: Probe) -> U32:
  U32.from_nat(probe_number(probe))
def probe_inside(+probe: Probe) -> {Nat.is_lt(probe_number(probe),U32.to_nat(probe_capacity(probe))) == True{} : Bool}:
  +query = probe_query(probe)
  +width = U32.to_nat(window_out(probe_window(probe)))
  Bounds.nat_trans(probe_number(probe),Nat.mul(U32.to_nat(rows(query)),width),U32.to_nat(probe_capacity(probe)),
    RowMajor.radix(channel(query),U32.to_nat(rows(query)),probe_column(probe),width,channel_inside(query),probe_column_inside(probe)),probe_output_room(probe))
def probe_exact(+probe: Probe) -> {U32.to_nat(probe_word(probe)) == probe_number(probe) : Nat}:
  Bounds.roundtrip(probe_number(probe),probe_capacity(probe),probe_capacity_ok(probe),
    Bounds.strict_le(probe_number(probe),U32.to_nat(probe_capacity(probe)),probe_inside(probe)))
def probe_bound(+probe: Probe) -> {U32.is_lt(probe_word(probe),probe_capacity(probe)) == True{} : Bool}:
  Bounds.native_bound(probe_number(probe),probe_capacity(probe),probe_exact(probe),probe_inside(probe))

# One job of a run, by unit and group.
type Job is Data:
  Job{probe: Probe,unit: Nat,group: Nat,
    unit_rows: {Nat.is_lt(Nat.mul(unit,4n),U32.to_nat(rows(probe_query(probe)))) == True{} : Bool},
    unit_inside: {Nat.is_lt(unit,units(probe_query(probe))) == True{} : Bool},
    group_inside: {Nat.is_lt(group,groups(probe_query(probe))) == True{} : Bool}}
def job_probe(+job: Job) -> Probe:
  Job{probe,unit,group,unit_rows,unit_inside,group_inside} = job
  probe
def job_unit(+job: Job) -> Nat:
  Job{probe,unit,group,unit_rows,unit_inside,group_inside} = job
  unit
def job_group(+job: Job) -> Nat:
  Job{probe,unit,group,unit_rows,unit_inside,group_inside} = job
  group
def job_unit_rows(+job: Job) -> {Nat.is_lt(Nat.mul(job_unit(job),4n),U32.to_nat(rows(probe_query(job_probe(job))))) == True{} : Bool}:
  Job{probe,unit,group,unit_rows,unit_inside,group_inside} = job
  unit_rows
def job_unit_inside(+job: Job) -> {Nat.is_lt(job_unit(job),units(probe_query(job_probe(job)))) == True{} : Bool}:
  Job{probe,unit,group,unit_rows,unit_inside,group_inside} = job
  unit_inside
def job_group_inside(+job: Job) -> {Nat.is_lt(job_group(job),groups(probe_query(job_probe(job)))) == True{} : Bool}:
  Job{probe,unit,group,unit_rows,unit_inside,group_inside} = job
  group_inside
def job_query(+job: Job) -> Query:
  probe_query(job_probe(job))
def job_window(+job: Job) -> OutputWindow:
  probe_window(job_probe(job))
def job_first(+job: Job) -> U32:
  (unit_word(job_query(job),job_unit(job),job_group(job)) * 4 : U32)
def job_position(+job: Job) -> U32:
  (window_start(job_window(job)) + group_word(job_query(job),job_unit(job),job_group(job)) * 8 : U32)
def job_context(+job: Job) -> Gemm.Context:
  +query = job_query(job)
  +word = unit_word(query,job_unit(job),job_group(job))
  Gemm.Context{job_first(job),job_position(job),rows(query),window_out(job_window(job)),
    (group_word(query,job_unit(job),job_group(job)) * inner(query) : U32),(word / 2 * inner(query) : U32)}
def job_block(+job: Job) -> GemmWrites.Block:
  +query = job_query(job)
  GemmWrites.Block{job_context(job),unit_accumulators(query,job_unit(job),job_group(job)),bias(query)}
def job_view(+job: Job) -> Views.View:
  GemmWrites.store_view(job_context(job))
def job_rows(+job: Job) -> Nat:
  GemmWrites.stored_rows(job_context(job))
def job_lanes(+job: Job) -> Nat:
  GemmWrites.stored_lanes(job_context(job))

def job_first_exact(+job: Job) -> {U32.to_nat(job_first(job)) == Nat.mul(job_unit(job),4n) : Nat}:
  +query = job_query(job)
  first_row_exact(query,job_unit(job),unit_word(query,job_unit(job),job_group(job)),
    unit_exact(query,job_unit(job),job_group(job),job_unit_inside(job),job_group_inside(job)),job_unit_rows(job))

# The first column a job stores lies before the end of the output.
def job_position_before(+job: Job) ->
  {Nat.is_lt(Nat.add(U32.to_nat(window_start(job_window(job))),Nat.mul(job_group(job),8n)),U32.to_nat(window_out(job_window(job)))) == True{} : Bool}:
  +query = job_query(job)
  +start = U32.to_nat(window_start(job_window(job)))
  +out = U32.to_nat(window_out(job_window(job)))
  +group = job_group(job)
  +position = Nat.add(start,Nat.mul(group,8n))
  +next_room = Order.offset_le(start,Nat.mul(1n+group,8n),Nat.mul(groups(query),8n),
    RowMajor.mul_le(1n+group,groups(query),8n,Order.strict_follow(group,groups(query),job_group_inside(job))))
  +room = Order.transitive(Nat.add(start,Nat.mul(1n+group,8n)),Nat.add(start,Nat.mul(groups(query),8n)),Nat.add(out,7n),next_room,
    %columns_exact(query) : {Nat.is_le(Nat.add(start,_),Nat.add(out,7n)) == True{} : Bool} window_end_room(job_window(job)))
  Bounds.predecessor_strict(position,out,
    Order.cancel_offset_le(7n,1n+position,out,
      %RowMajor.add_swap(start,8n,Nat.mul(group,8n)) : {Nat.is_le(_,Nat.add(7n,out)) == True{} : Bool}
      %Order.commutative(out,7n) : {Nat.is_le(Nat.add(start,Nat.mul(1n+group,8n)),_) == True{} : Bool}
      room))
def job_position_exact(+job: Job) -> {U32.to_nat(job_position(job)) == Nat.add(U32.to_nat(window_start(job_window(job))),Nat.mul(job_group(job),8n)) : Nat}:
  +query = job_query(job)
  +start = window_start(job_window(job))
  +out = U32.to_nat(window_out(job_window(job)))
  +local = (group_word(query,job_unit(job),job_group(job)) * 8 : U32)
  +local_exact = position_exact(query,job_group(job),group_word(query,job_unit(job),job_group(job)),
    group_exact(query,job_unit(job),job_group(job),job_unit_inside(job),job_group_inside(job)),job_group_inside(job))
  %local_exact : {U32.to_nat(job_position(job)) == Nat.add(U32.to_nat(start),_) : Nat}
  Arithmetic.add(start,local,limit(query),limit_fits(query),
    %Equal.sym(Nat,U32.to_nat(local),Nat.mul(job_group(job),8n),local_exact) : {Nat.is_le(Nat.add(U32.to_nat(start),_),U32.to_nat(limit(query))) == True{} : Bool}
    Order.transitive(Nat.add(U32.to_nat(start),Nat.mul(job_group(job),8n)),out,U32.to_nat(limit(query)),
      Bounds.strict_le(Nat.add(U32.to_nat(start),Nat.mul(job_group(job),8n)),out,job_position_before(job)),window_out_room(job_window(job))))
def job_ordered(+job: Job) -> {Nat.is_le(U32.to_nat(job_first(job)),U32.to_nat(rows(job_query(job)))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(job_first(job)),Nat.mul(job_unit(job),4n),job_first_exact(job)) : {Nat.is_le(_,U32.to_nat(rows(job_query(job)))) == True{} : Bool}
  Bounds.strict_le(Nat.mul(job_unit(job),4n),U32.to_nat(rows(job_query(job))),job_unit_rows(job))

# A job's writes are the writes of its view.
def job_writes_exact(~biased: Bool,+job: Job) ->
  {GemmWrites.job_writes(~biased,Gemm.job(window_layout(job_window(job)),U32.from_nat(job_index(job_unit(job),job_group(job),units(job_query(job))))),
      input(job_query(job)),weights(job_query(job)),bias(job_query(job))) ==
    GemmWrites.context_writes(~biased,job_context(job),unit_accumulators(job_query(job),job_unit(job),job_group(job)),bias(job_query(job))) : List<&2,Writes.Assignment<F32>>}:
  Job{Probe{OutputWindow{Query{rows,span,inner,groups,limit,input,weights,bias,channel,column,limit_fits,columns_exact,rows_room,area_room,channel_inside,column_inside},
    start,out,end_room,out_room},capacity,capacity_ok,output_room,probed,probed_inside},unit,group,unit_rows,unit_inside,group_inside} = job
  {==}
def job_view_writes(~biased: Bool,+job: Job) ->
  {GemmWrites.context_writes(~biased,job_context(job),unit_accumulators(job_query(job),job_unit(job),job_group(job)),bias(job_query(job))) ==
    ViewWrites.writes(~F32,~GemmWrites.Block,~GemmWrites.tile_value(~biased),Nat.mul(job_rows(job),job_lanes(job)),job_view(job),job_block(job)) : List<&2,Writes.Assignment<F32>>}:
  +query = job_query(job)
  GemmWrites.view_writes(~biased,GemmWrites.Placed{job_block(job),limit(query),limit_fits(query),job_ordered(job),rows_room(query)})
def job_writes_view(~biased: Bool,+job: Job) ->
  {GemmWrites.job_writes(~biased,Gemm.job(window_layout(job_window(job)),U32.from_nat(job_index(job_unit(job),job_group(job),units(job_query(job))))),
      input(job_query(job)),weights(job_query(job)),bias(job_query(job))) ==
    ViewWrites.writes(~F32,~GemmWrites.Block,~GemmWrites.tile_value(~biased),Nat.mul(job_rows(job),job_lanes(job)),job_view(job),job_block(job)) : List<&2,Writes.Assignment<F32>>}:
  Equal.trans(List<&2,Writes.Assignment<F32>>,
    GemmWrites.job_writes(~biased,Gemm.job(window_layout(job_window(job)),U32.from_nat(job_index(job_unit(job),job_group(job),units(job_query(job))))),
      input(job_query(job)),weights(job_query(job)),bias(job_query(job))),
    GemmWrites.context_writes(~biased,job_context(job),unit_accumulators(job_query(job),job_unit(job),job_group(job)),bias(job_query(job))),
    ViewWrites.writes(~F32,~GemmWrites.Block,~GemmWrites.tile_value(~biased),Nat.mul(job_rows(job),job_lanes(job)),job_view(job),job_block(job)),
    job_writes_exact(~biased,job),job_view_writes(~biased,job))

# A row or lane is stored exactly when its output row or column lies inside
# the output.
def row_bound(+job: Job,+row: Nat) ->
  {Nat.is_lt(row,U32.to_nat((rows(job_query(job)) - job_first(job) : U32))) == Nat.is_lt(row_number(job_unit(job),row),U32.to_nat(rows(job_query(job)))) : Bool}:
  +query = job_query(job)
  +first = job_first(job)
  +total = U32.to_nat(rows(query))
  +ordered = job_ordered(job)
  +difference = Arithmetic.subtract(rows(query),first,limit(query),limit_fits(query),ordered,
    Order.transitive(Nat.sub(total,U32.to_nat(first)),total,U32.to_nat(limit(query)),Order.sub_le(total,U32.to_nat(first)),
      Order.transitive(total,Nat.add(total,3n),U32.to_nat(limit(query)),Order.le_add(total,3n),rows_room(query))))
  %Equal.sym(Nat,U32.to_nat((rows(query) - first : U32)),Nat.sub(total,U32.to_nat(first)),difference) :
    {Nat.is_lt(row,_) == Nat.is_lt(row_number(job_unit(job),row),total) : Bool}
  %job_first_exact(job) : {Nat.is_lt(row,Nat.sub(total,U32.to_nat(first))) == Nat.is_lt(Nat.add(_,row),total) : Bool}
  Order.below_difference(U32.to_nat(first),row,total,ordered)
def lane_bound(+job: Job,+lane: Nat) ->
  {Nat.is_lt(lane,U32.to_nat((window_out(job_window(job)) - job_position(job) : U32))) ==
    Nat.is_lt(Nat.add(U32.to_nat(window_start(job_window(job))),lane_number(job_group(job),lane)),U32.to_nat(window_out(job_window(job)))) : Bool}:
  +query = job_query(job)
  +window = job_window(job)
  +position = job_position(job)
  +start = U32.to_nat(window_start(window))
  +eight = Nat.mul(job_group(job),8n)
  +total = U32.to_nat(window_out(window))
  +position_exact = job_position_exact(job)
  +ordered = {
    %Equal.sym(Nat,U32.to_nat(position),Nat.add(start,eight),position_exact) : {Nat.is_le(_,total) == True{} : Bool}
    Bounds.strict_le(Nat.add(start,eight),total,job_position_before(job)) : {Nat.is_le(U32.to_nat(position),total) == True{} : Bool}}
  +difference = Arithmetic.subtract(window_out(window),position,limit(query),limit_fits(query),ordered,
    Order.transitive(Nat.sub(total,U32.to_nat(position)),total,U32.to_nat(limit(query)),Order.sub_le(total,U32.to_nat(position)),window_out_room(window)))
  %Equal.sym(Nat,U32.to_nat((window_out(window) - position : U32)),Nat.sub(total,U32.to_nat(position)),difference) :
    {Nat.is_lt(lane,_) == Nat.is_lt(Nat.add(start,Nat.add(eight,lane)),total) : Bool}
  %Addition.associative(start,eight,lane) : {Nat.is_lt(lane,Nat.sub(total,U32.to_nat(position))) == Nat.is_lt(_,total) : Bool}
  %position_exact : {Nat.is_lt(lane,Nat.sub(total,U32.to_nat(position))) == Nat.is_lt(Nat.add(_,lane),total) : Bool}
  Order.below_difference(U32.to_nat(position),lane,total,ordered)

# A stored cell lies inside the rows and the output columns.
def cell_row_below(+job: Job,+row: Nat,+row_inside: {Nat.is_lt(row,job_rows(job)) == True{} : Bool}) ->
  {Nat.is_lt(row_number(job_unit(job),row),U32.to_nat(rows(job_query(job)))) == True{} : Bool}:
  +left = (rows(job_query(job)) - job_first(job) : U32)
  Equal.trans(Bool,Nat.is_lt(row_number(job_unit(job),row),U32.to_nat(rows(job_query(job)))),Nat.is_lt(row,U32.to_nat(left)),True{},
    Equal.sym(Bool,Nat.is_lt(row,U32.to_nat(left)),Nat.is_lt(row_number(job_unit(job),row),U32.to_nat(rows(job_query(job)))),row_bound(job,row)),
    ViewWrites.clipped_below(row,4,left,row_inside))
def cell_column_below(+job: Job,+lane: Nat,+lane_inside: {Nat.is_lt(lane,job_lanes(job)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(U32.to_nat(window_start(job_window(job))),lane_number(job_group(job),lane)),U32.to_nat(window_out(job_window(job)))) == True{} : Bool}:
  +left = (window_out(job_window(job)) - job_position(job) : U32)
  +column = Nat.add(U32.to_nat(window_start(job_window(job))),lane_number(job_group(job),lane))
  Equal.trans(Bool,Nat.is_lt(column,U32.to_nat(window_out(job_window(job)))),Nat.is_lt(lane,U32.to_nat(left)),True{},
    Equal.sym(Bool,Nat.is_lt(lane,U32.to_nat(left)),Nat.is_lt(column,U32.to_nat(window_out(job_window(job)))),lane_bound(job,lane)),
    ViewWrites.clipped_below(lane,8,left,lane_inside))

# The output index of a stored cell.
def cell_number(+job: Job,+row: Nat,+lane: Nat) -> Nat:
  Nat.add(Nat.mul(row_number(job_unit(job),row),U32.to_nat(window_out(job_window(job)))),Nat.add(U32.to_nat(window_start(job_window(job))),lane_number(job_group(job),lane)))
def cell_inside(+job: Job,+row: Nat,+lane: Nat,+row_inside: {Nat.is_lt(row,job_rows(job)) == True{} : Bool},+lane_inside: {Nat.is_lt(lane,job_lanes(job)) == True{} : Bool}) ->
  {Nat.is_lt(cell_number(job,row,lane),U32.to_nat(probe_capacity(job_probe(job)))) == True{} : Bool}:
  +query = job_query(job)
  +width = U32.to_nat(window_out(job_window(job)))
  Bounds.nat_trans(cell_number(job,row,lane),Nat.mul(U32.to_nat(rows(query)),width),U32.to_nat(probe_capacity(job_probe(job))),
    RowMajor.radix(row_number(job_unit(job),row),U32.to_nat(rows(query)),Nat.add(U32.to_nat(window_start(job_window(job))),lane_number(job_group(job),lane)),width,
      cell_row_below(job,row,row_inside),cell_column_below(job,lane,lane_inside)),
    probe_output_room(job_probe(job)))
def cell_exact(+job: Job,+row: Nat,+lane: Nat,+row_inside: {Nat.is_lt(row,job_rows(job)) == True{} : Bool},+lane_inside: {Nat.is_lt(lane,job_lanes(job)) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(cell_number(job,row,lane))) == cell_number(job,row,lane) : Nat}:
  +capacity = probe_capacity(job_probe(job))
  Bounds.roundtrip(cell_number(job,row,lane),capacity,probe_capacity_ok(job_probe(job)),
    Bounds.strict_le(cell_number(job,row,lane),U32.to_nat(capacity),cell_inside(job,row,lane,row_inside,lane_inside)))
def cell_bound(+job: Job,+row: Nat,+lane: Nat,+row_inside: {Nat.is_lt(row,job_rows(job)) == True{} : Bool},+lane_inside: {Nat.is_lt(lane,job_lanes(job)) == True{} : Bool}) ->
  {U32.is_lt(U32.from_nat(cell_number(job,row,lane)),probe_capacity(job_probe(job))) == True{} : Bool}:
  Bounds.native_bound(cell_number(job,row,lane),probe_capacity(job_probe(job)),cell_exact(job,row,lane,row_inside,lane_inside),cell_inside(job,row,lane,row_inside,lane_inside))

# A cell's view address: lane plus row times the output columns plus the
# job's first address, rearranged as the cell's output index.
def cell_sum(+lane: Nat,+row: Nat,+first: Nat,+width: Nat,+start: Nat,+offset: Nat) ->
  {Nat.add(Nat.mul(lane,1n),Nat.add(Nat.mul(row,width),Nat.add(Nat.mul(first,width),Nat.add(start,offset)))) ==
    Nat.add(Nat.mul(Nat.add(first,row),width),Nat.add(start,Nat.add(offset,lane))) : Nat}:
  +a = Nat.mul(row,width)
  +b = Nat.mul(first,width)
  +c = Nat.add(start,offset)
  %Equal.sym(Nat,Nat.mul(lane,1n),lane,Order.times_one(lane)) :
    {Nat.add(_,Nat.add(a,Nat.add(b,c))) == Nat.add(Nat.mul(Nat.add(first,row),width),Nat.add(start,Nat.add(offset,lane))) : Nat}
  %Equal.sym(Nat,Nat.mul(Nat.add(first,row),width),Nat.add(b,a),RowMajor.distrib(first,row,width)) :
    {Nat.add(lane,Nat.add(a,Nat.add(b,c))) == Nat.add(_,Nat.add(start,Nat.add(offset,lane))) : Nat}
  Equal.trans(Nat,Nat.add(lane,Nat.add(a,Nat.add(b,c))),Nat.add(Nat.add(a,Nat.add(b,c)),lane),Nat.add(Nat.add(b,a),Nat.add(start,Nat.add(offset,lane))),
    Order.commutative(lane,Nat.add(a,Nat.add(b,c))),
    Equal.trans(Nat,Nat.add(Nat.add(a,Nat.add(b,c)),lane),Nat.add(Nat.add(Nat.add(b,a),c),lane),Nat.add(Nat.add(b,a),Nat.add(start,Nat.add(offset,lane))),
      Equal.cong(Nat,Nat,sum => Nat.add(sum,lane),Nat.add(a,Nat.add(b,c)),Nat.add(Nat.add(b,a),c),
        Equal.trans(Nat,Nat.add(a,Nat.add(b,c)),Nat.add(Nat.add(a,b),c),Nat.add(Nat.add(b,a),c),
          Equal.sym(Nat,Nat.add(Nat.add(a,b),c),Nat.add(a,Nat.add(b,c)),Addition.associative(a,b,c)),
          Equal.cong(Nat,Nat,sum => Nat.add(sum,c),Nat.add(a,b),Nat.add(b,a),Order.commutative(a,b)))),
      Equal.trans(Nat,Nat.add(Nat.add(Nat.add(b,a),c),lane),Nat.add(Nat.add(b,a),Nat.add(c,lane)),Nat.add(Nat.add(b,a),Nat.add(start,Nat.add(offset,lane))),
        Addition.associative(Nat.add(b,a),c,lane),
        Equal.cong(Nat,Nat,tail => Nat.add(Nat.add(b,a),tail),Nat.add(c,lane),Nat.add(start,Nat.add(offset,lane)),Addition.associative(start,offset,lane)))))
def cell_word(+job: Job,+row: Nat,+lane: Nat,+row_inside: {Nat.is_lt(row,job_rows(job)) == True{} : Bool},+lane_inside: {Nat.is_lt(lane,job_lanes(job)) == True{} : Bool}) ->
  {Copy.address(job_view(job),Nat.add(Nat.mul(row,job_lanes(job)),lane)) == U32.from_nat(cell_number(job,row,lane)) : U32}:
  +probe = job_probe(job)
  +context = job_context(job)
  +first = job_first(job)
  +position = job_position(job)
  +out = window_out(job_window(job))
  +width = U32.to_nat(out)
  +start = U32.to_nat(window_start(job_window(job)))
  +unit_first = Nat.mul(job_unit(job),4n)
  +eight = Nat.mul(job_group(job),8n)
  +capacity = probe_capacity(probe)
  +capacity_ok = probe_capacity_ok(probe)
  +first_exact = job_first_exact(job)
  +position_exact = job_position_exact(job)
  +origin = Nat.add(Nat.mul(unit_first,width),Nat.add(start,eight))
  +origin_inside = Bounds.nat_trans(origin,Nat.mul(U32.to_nat(rows(job_query(job))),width),U32.to_nat(capacity),
    RowMajor.radix(unit_first,U32.to_nat(rows(job_query(job))),Nat.add(start,eight),width,job_unit_rows(job),job_position_before(job)),
    probe_output_room(probe))
  +origin_small = Bounds.strict_le(origin,U32.to_nat(capacity),origin_inside)
  +product_exact = Arithmetic.multiply(first,out,capacity,capacity_ok,
    %Equal.sym(Nat,U32.to_nat(first),unit_first,first_exact) : {Nat.is_le(Nat.mul(_,width),U32.to_nat(capacity)) == True{} : Bool}
    Order.transitive(Nat.mul(unit_first,width),origin,U32.to_nat(capacity),Order.le_add(Nat.mul(unit_first,width),Nat.add(start,eight)),origin_small))
  +target_exact = {
    %position_exact : {U32.to_nat(GemmWrites.store_target(context)) == Nat.add(Nat.mul(unit_first,width),_) : Nat}
    %first_exact : {U32.to_nat(GemmWrites.store_target(context)) == Nat.add(Nat.mul(_,width),U32.to_nat(position)) : Nat}
    %product_exact : {U32.to_nat(GemmWrites.store_target(context)) == Nat.add(_,U32.to_nat(position)) : Nat}
    Arithmetic.add((first * out : U32),position,capacity,capacity_ok,
      %Equal.sym(Nat,U32.to_nat((first * out : U32)),Nat.mul(U32.to_nat(first),width),product_exact) :
        {Nat.is_le(Nat.add(_,U32.to_nat(position)),U32.to_nat(capacity)) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(first),unit_first,first_exact) : {Nat.is_le(Nat.add(Nat.mul(_,width),U32.to_nat(position)),U32.to_nat(capacity)) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(position),Nat.add(start,eight),position_exact) : {Nat.is_le(Nat.add(Nat.mul(unit_first,width),_),U32.to_nat(capacity)) == True{} : Bool}
      origin_small) :
    {U32.to_nat(GemmWrites.store_target(context)) == origin : Nat}}
  Equal.trans(U32,Copy.address(job_view(job),Nat.add(Nat.mul(row,job_lanes(job)),lane)),
    U32.from_nat(Nat.add(Nat.mul(lane,1n),Nat.add(Nat.mul(row,width),U32.to_nat(GemmWrites.store_target(context))))),U32.from_nat(cell_number(job,row,lane)),
    ViewWrites.plane_digits(GemmWrites.store_target(context),GemmWrites.store_rows(context),out,row,GemmWrites.store_lanes(context),1,lane,row_inside,lane_inside),
    Equal.cong(Nat,U32,U32.from_nat,Nat.add(Nat.mul(lane,1n),Nat.add(Nat.mul(row,width),U32.to_nat(GemmWrites.store_target(context)))),cell_number(job,row,lane),
      %Equal.sym(Nat,U32.to_nat(GemmWrites.store_target(context)),origin,target_exact) :
        {Nat.add(Nat.mul(lane,1n),Nat.add(Nat.mul(row,width),_)) == cell_number(job,row,lane) : Nat}
      cell_sum(lane,row,unit_first,width,start,eight)))

# A stored cell's value is the output value of its unit, row, group and lane.
def cell_value_exact(~biased: Bool,+job: Job,+row: Nat,+lane: Nat,+lane_inside: {Nat.is_lt(lane,job_lanes(job)) == True{} : Bool}) ->
  {GemmWrites.tile_value(~biased,job_block(job),Nat.add(Nat.mul(row,job_lanes(job)),lane)) ==
    output_value(~biased,job_query(job),job_unit(job),row,job_group(job),lane,GemmWrites.shifted(job_first(job),row)) : F32}:
  Equal.sym(F32,GemmWrites.cell_value(~biased,job_block(job),row,lane),GemmWrites.tile_value(~biased,job_block(job),Nat.add(Nat.mul(row,job_lanes(job)),lane)),
    GemmWrites.cell_element(~biased,job_block(job),row,lane,lane_inside))
def cell_row_exact(+job: Job,+row: Nat,+row_inside: {Nat.is_lt(row,job_rows(job)) == True{} : Bool}) ->
  {U32.to_nat(GemmWrites.shifted(job_first(job),row)) == row_number(job_unit(job),row) : Nat}:
  +query = job_query(job)
  +small = ViewWrites.clipped_small(row,4,(rows(query) - job_first(job) : U32),row_inside)
  %Equal.sym(U32,GemmWrites.shifted(job_first(job),row),(job_first(job) + U32.from_nat(row) : U32),GemmWrites.shifted_exact(job_first(job),row)) :
    {U32.to_nat(_) == row_number(job_unit(job),row) : Nat}
  row_offset_exact(query,job_unit(job),unit_word(query,job_unit(job),job_group(job)),row,
    unit_exact(query,job_unit(job),job_group(job),job_unit_inside(job),job_group_inside(job)),job_unit_rows(job),
    {Order.strict_follow(row,4n,small) : {Nat.is_le(row,3n) == True{} : Bool}})

# A cell at the probed index is the probed cell: same row, and the same
# column of the output.
def cell_numbers(+job: Job,+row: Nat,+lane: Nat,+row_inside: {Nat.is_lt(row,job_rows(job)) == True{} : Bool},+lane_inside: {Nat.is_lt(lane,job_lanes(job)) == True{} : Bool},
  equal: {U32.is_eq(U32.from_nat(cell_number(job,row,lane)),probe_word(job_probe(job))) == True{} : Bool}) ->
  {probe_number(job_probe(job)) == cell_number(job,row,lane) : Nat}:
  +probe = job_probe(job)
  +words = Arithmetic.equal_of_is_eq(U32.from_nat(cell_number(job,row,lane)),probe_word(probe),equal)
  Equal.trans(Nat,probe_number(probe),U32.to_nat(probe_word(probe)),cell_number(job,row,lane),
    Equal.sym(Nat,U32.to_nat(probe_word(probe)),probe_number(probe),probe_exact(probe)),
    Equal.trans(Nat,U32.to_nat(probe_word(probe)),U32.to_nat(U32.from_nat(cell_number(job,row,lane))),cell_number(job,row,lane),
      Equal.cong(U32,Nat,U32.to_nat,probe_word(probe),U32.from_nat(cell_number(job,row,lane)),Equal.sym(U32,U32.from_nat(cell_number(job,row,lane)),probe_word(probe),words)),
      cell_exact(job,row,lane,row_inside,lane_inside)))
def probe_positive(+probe: Probe) -> {Nat.is_lt(0n,U32.to_nat(window_out(probe_window(probe)))) == True{} : Bool}:
  Order.le_lt_trans(0n,probe_column(probe),U32.to_nat(window_out(probe_window(probe))),Order.zero_le(probe_column(probe)),probe_column_inside(probe))
def cell_channel(+job: Job,+row: Nat,+lane: Nat,+row_inside: {Nat.is_lt(row,job_rows(job)) == True{} : Bool},+lane_inside: {Nat.is_lt(lane,job_lanes(job)) == True{} : Bool},
  equal: {U32.is_eq(U32.from_nat(cell_number(job,row,lane)),probe_word(job_probe(job))) == True{} : Bool}) ->
  {channel(job_query(job)) == row_number(job_unit(job),row) : Nat}:
  +probe = job_probe(job)
  Radix.row_unique(channel(job_query(job)),row_number(job_unit(job),row),U32.to_nat(window_out(job_window(job))),probe_column(probe),
    Nat.add(U32.to_nat(window_start(job_window(job))),lane_number(job_group(job),lane)),probe_positive(probe),probe_column_inside(probe),
    cell_column_below(job,lane,lane_inside),cell_numbers(job,row,lane,row_inside,lane_inside,equal))
def cell_column(+job: Job,+row: Nat,+lane: Nat,+row_inside: {Nat.is_lt(row,job_rows(job)) == True{} : Bool},+lane_inside: {Nat.is_lt(lane,job_lanes(job)) == True{} : Bool},
  equal: {U32.is_eq(U32.from_nat(cell_number(job,row,lane)),probe_word(job_probe(job))) == True{} : Bool}) ->
  {probe_column(job_probe(job)) == Nat.add(U32.to_nat(window_start(job_window(job))),lane_number(job_group(job),lane)) : Nat}:
  +probe = job_probe(job)
  Radix.col_unique(channel(job_query(job)),row_number(job_unit(job),row),U32.to_nat(window_out(job_window(job))),probe_column(probe),
    Nat.add(U32.to_nat(window_start(job_window(job))),lane_number(job_group(job),lane)),probe_positive(probe),probe_column_inside(probe),
    cell_column_below(job,lane,lane_inside),cell_numbers(job,row,lane,row_inside,lane_inside,equal))

# ---- Every job ----

# The writes of all jobs from first on hold a property when each job's
# writes hold it.
def every_indexed(~biased: Bool,~Target: Data,~Holds: @+target: Target -> U32 -> F32 -> Data,~window: @+target: Target -> OutputWindow,
  ~job_holds: @+target: Target -> @+unit: Nat -> @+group: Nat ->
    @+unit_rows: {Nat.is_lt(Nat.mul(unit,4n),U32.to_nat(rows(window_query(window(target))))) == True{} : Bool} ->
    @+unit_inside: {Nat.is_lt(unit,units(window_query(window(target)))) == True{} : Bool} ->
    @+group_inside: {Nat.is_lt(group,groups(window_query(window(target)))) == True{} : Bool} ->
    Every.Every(~F32,~Target,~Holds,target,GemmWrites.job_writes(~biased,Gemm.job(window_layout(window(target)),U32.from_nat(job_index(unit,group,units(window_query(window(target)))))),
      input(window_query(window(target))),weights(window_query(window(target))),bias(window_query(window(target))))),
  +target: Target,+index: Nat,decoded: IndexedJob<window_query(window(target)),index>) ->
  Every.Every(~F32,~Target,~Holds,target,GemmWrites.job_writes(~biased,Gemm.job(window_layout(window(target)),U32.from_nat(index)),
    input(window_query(window(target))),weights(window_query(window(target))),bias(window_query(window(target))))):
  IndexedJob{+unit,+group,+unit_rows,+unit_inside,+group_inside,+exact} = decoded
  +query = window_query(window(target))
  %Equal.sym(Nat,index,job_index(unit,group,units(query)),exact) :
    Every.Every(~F32,~Target,~Holds,target,GemmWrites.job_writes(~biased,Gemm.job(window_layout(window(target)),U32.from_nat(_)),input(query),weights(query),bias(query)))
  job_holds(target,unit,group,unit_rows,unit_inside,group_inside)
def all_every(~biased: Bool,~Target: Data,~Holds: @+target: Target -> U32 -> F32 -> Data,~window: @+target: Target -> OutputWindow,
  ~job_holds: @+target: Target -> @+unit: Nat -> @+group: Nat ->
    @+unit_rows: {Nat.is_lt(Nat.mul(unit,4n),U32.to_nat(rows(window_query(window(target))))) == True{} : Bool} ->
    @+unit_inside: {Nat.is_lt(unit,units(window_query(window(target)))) == True{} : Bool} ->
    @+group_inside: {Nat.is_lt(group,groups(window_query(window(target)))) == True{} : Bool} ->
    Every.Every(~F32,~Target,~Holds,target,GemmWrites.job_writes(~biased,Gemm.job(window_layout(window(target)),U32.from_nat(job_index(unit,group,units(window_query(window(target)))))),
      input(window_query(window(target))),weights(window_query(window(target))),bias(window_query(window(target))))),
  +target: Target,+count: Nat,+first: Nat,
  +room: {Nat.is_le(Nat.add(first,count),Nat.mul(groups(window_query(window(target))),units(window_query(window(target))))) == True{} : Bool}) ->
  Every.Every(~F32,~Target,~Holds,target,GemmWrites.all_writes(~biased,count,U32.from_nat(first),window_layout(window(target)),
    input(window_query(window(target))),weights(window_query(window(target))),bias(window_query(window(target))))):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      +query = window_query(window(target))
      +layout = window_layout(window(target))
      +here = GemmWrites.job_writes(~biased,Gemm.job(layout,U32.from_nat(first)),input(query),weights(query),bias(query))
      +later = GemmWrites.all_writes(~biased,rest,U32.from_nat(1n+first),layout,input(query),weights(query),bias(query))
      %Equal.sym(U32,(U32.from_nat(first) + 1 : U32),U32.from_nat(1n+first),Bounds.add_one(U32.from_nat(first))) :
        Every.Every(~F32,~Target,~Holds,target,List.append(&2,Writes.Assignment<F32>,here,GemmWrites.all_writes(~biased,rest,_,layout,input(query),weights(query),bias(query))))
      Every.appended(~F32,~Target,~Holds,target,here,later,
        every_indexed(~biased,~Target,~Holds,~window,~job_holds,target,first,
          indexed_job(query,first,Order.room_index(first,rest,Nat.mul(groups(query),units(query)),room))),
        all_every(~biased,~Target,~Holds,~window,~job_holds,target,rest,1n+first,Order.advance_room(first,rest,Nat.mul(groups(query),units(query)),room)))

# ---- Reads at a query's column ----

# A probe at the query's own column.
type WindowRead is Data:
  WindowRead{probe: Probe,at: {probe_column(probe) == Nat.add(U32.to_nat(window_start(probe_window(probe))),column(probe_query(probe))) : Nat}}
def read_probe(+read: WindowRead) -> Probe:
  WindowRead{probe,at} = read
  probe
def read_self(+read: WindowRead) -> WindowRead:
  read
def read_window(+read: WindowRead) -> OutputWindow:
  probe_window(read_probe(read))
def read_capacity(+read: WindowRead) -> U32:
  probe_capacity(read_probe(read))
def read_word(+read: WindowRead) -> U32:
  probe_word(read_probe(read))
def read_wanted(~biased: Bool,+read: WindowRead) -> F32:
  wanted(~biased,probe_query(read_probe(read)))

# Every write of a run stays inside the output, and a write at the read
# index stores the wanted value.
def Written(~biased: Bool,+read: WindowRead,address: U32,value: F32) -> Data:
  Shape.Both<{U32.is_lt(address,read_capacity(read)) == True{} : Bool},Boolean.When(U32.is_eq(address,read_word(read)),{value == read_wanted(~biased,read) : F32})>
def written_inside(~biased: Bool,+read: WindowRead,+address: U32,+value: F32,holds: Written(~biased,read,address,value)) ->
  {U32.is_lt(address,read_capacity(read)) == True{} : Bool}:
  Shape.Both{inside,matches} = holds
  inside
def written_matches(~biased: Bool,+read: WindowRead,+address: U32,+value: F32,holds: Written(~biased,read,address,value)) ->
  Boolean.When(U32.is_eq(address,read_word(read)),{value == read_wanted(~biased,read) : F32}):
  Shape.Both{inside,matches} = holds
  matches

# A job of a read.
type ReadJob is Data:
  ReadJob{job: Job,at: {probe_column(job_probe(job)) == Nat.add(U32.to_nat(window_start(job_window(job))),column(job_query(job))) : Nat}}
def read_job(+cells: ReadJob) -> Job:
  ReadJob{job,at} = cells
  job
def read_of(+cells: ReadJob) -> WindowRead:
  ReadJob{job,at} = cells
  WindowRead{job_probe(job),at}
def read_view(+cells: ReadJob) -> Views.View:
  job_view(read_job(cells))
def read_block(+cells: ReadJob) -> GemmWrites.Block:
  job_block(read_job(cells))
def read_rows(+cells: ReadJob) -> Nat:
  job_rows(read_job(cells))
def read_lanes(+cells: ReadJob) -> Nat:
  job_lanes(read_job(cells))

def written_value(~biased: Bool,+job: Job,+at: {probe_column(job_probe(job)) == Nat.add(U32.to_nat(window_start(job_window(job))),column(job_query(job))) : Nat},
  +row: Nat,+lane: Nat,+row_inside: {Nat.is_lt(row,job_rows(job)) == True{} : Bool},+lane_inside: {Nat.is_lt(lane,job_lanes(job)) == True{} : Bool},equal: Bool,
  +compared: {U32.is_eq(U32.from_nat(cell_number(job,row,lane)),probe_word(job_probe(job))) == equal : Bool}) ->
  Boolean.When(equal,{GemmWrites.tile_value(~biased,job_block(job),Nat.add(Nat.mul(row,job_lanes(job)),lane)) == wanted(~biased,job_query(job)) : F32}):
  match equal:
    case False{}: Unit{}
    case True{}:
      +query = job_query(job)
      +start = U32.to_nat(window_start(job_window(job)))
      +small = ViewWrites.clipped_small(row,4,(rows(query) - job_first(job) : U32),row_inside)
      +lane_small = ViewWrites.clipped_small(lane,8,(window_out(job_window(job)) - job_position(job) : U32),lane_inside)
      +channel_same = cell_channel(job,row,lane,row_inside,lane_inside,compared)
      +lane_same = Order.cancel_sum(start,column(query),lane_number(job_group(job),lane),
        Equal.trans(Nat,Nat.add(start,column(query)),probe_column(job_probe(job)),Nat.add(start,lane_number(job_group(job),lane)),
          Equal.sym(Nat,probe_column(job_probe(job)),Nat.add(start,column(query)),at),cell_column(job,row,lane,row_inside,lane_inside,compared)))
      Equal.trans(F32,GemmWrites.tile_value(~biased,job_block(job),Nat.add(Nat.mul(row,job_lanes(job)),lane)),
        output_value(~biased,query,job_unit(job),row,job_group(job),lane,GemmWrites.shifted(job_first(job),row)),wanted(~biased,query),
        cell_value_exact(~biased,job,row,lane,lane_inside),
        selected_value(~biased,query,job_unit(job),row,job_group(job),lane,GemmWrites.shifted(job_first(job),row),
          split_address(query,job_unit(job),row,job_group(job),lane,channel_same,lane_same),
          cell_row_exact(job,row,row_inside),small,lane_small,job_group_inside(job)))
def written_cell(~biased: Bool,+cells: ReadJob,+row: Nat,+lane: Nat,+row_inside: {Nat.is_lt(row,read_rows(cells)) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,read_lanes(cells)) == True{} : Bool}) ->
  Written(~biased,read_of(cells),Copy.address(read_view(cells),Nat.add(Nat.mul(row,read_lanes(cells)),lane)),
    GemmWrites.tile_value(~biased,read_block(cells),Nat.add(Nat.mul(row,read_lanes(cells)),lane))):
  ReadJob{+job,+at} = cells
  +probe = job_probe(job)
  +number = cell_number(job,row,lane)
  +value = GemmWrites.tile_value(~biased,job_block(job),Nat.add(Nat.mul(row,job_lanes(job)),lane))
  %Equal.sym(U32,Copy.address(job_view(job),Nat.add(Nat.mul(row,job_lanes(job)),lane)),U32.from_nat(number),cell_word(job,row,lane,row_inside,lane_inside)) :
    Written(~biased,WindowRead{probe,at},_,value)
  Shape.Both{cell_bound(job,row,lane,row_inside,lane_inside),
    written_value(~biased,job,at,row,lane,row_inside,lane_inside,U32.is_eq(U32.from_nat(number),probe_word(probe)),{==})}
def read_job_every(~biased: Bool,+read: WindowRead,+unit: Nat,+group: Nat,
  +unit_rows: {Nat.is_lt(Nat.mul(unit,4n),U32.to_nat(rows(window_query(read_window(read))))) == True{} : Bool},
  +unit_inside: {Nat.is_lt(unit,units(window_query(read_window(read)))) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,groups(window_query(read_window(read)))) == True{} : Bool}) ->
  Every.Every(~F32,~WindowRead,~Written(~biased),read,GemmWrites.job_writes(~biased,Gemm.job(window_layout(read_window(read)),U32.from_nat(job_index(unit,group,units(window_query(read_window(read)))))),
    input(window_query(read_window(read))),weights(window_query(read_window(read))),bias(window_query(read_window(read))))):
  WindowRead{+probe,+at} = read
  +job = {Job{probe,unit,group,unit_rows,unit_inside,group_inside} : Job}
  %Equal.sym(List<&2,Writes.Assignment<F32>>,
    GemmWrites.job_writes(~biased,Gemm.job(window_layout(job_window(job)),U32.from_nat(job_index(unit,group,units(job_query(job))))),input(job_query(job)),weights(job_query(job)),bias(job_query(job))),
    ViewWrites.writes(~F32,~GemmWrites.Block,~GemmWrites.tile_value(~biased),Nat.mul(job_rows(job),job_lanes(job)),job_view(job),job_block(job)),
    job_writes_view(~biased,job)) :
    Every.Every(~F32,~WindowRead,~Written(~biased),WindowRead{probe,at},_)
  ViewWrites.every_plane(~F32,~GemmWrites.Block,~GemmWrites.tile_value(~biased),~ReadJob,~WindowRead,~read_of,~Written(~biased),~read_view,~read_block,~read_rows,~read_lanes,
    ~written_cell(~biased),{ReadJob{job,at} : ReadJob})

# The job that stores a read's cell.
def read_unit(+read: WindowRead) -> Nat:
  Nat.div(channel(probe_query(read_probe(read))),4n)
def read_group(+read: WindowRead) -> Nat:
  Nat.div(column(probe_query(read_probe(read))),8n)
def read_index(+read: WindowRead) -> Nat:
  job_index(read_unit(read),read_group(read),units(probe_query(read_probe(read))))
def read_covered(~biased: Bool,+read: WindowRead) ->
  {WriteReadback.contains(~F32,read_word(read),GemmWrites.job_writes(~biased,Gemm.job(window_layout(read_window(read)),U32.from_nat(read_index(read))),
    input(window_query(read_window(read))),weights(window_query(read_window(read))),bias(window_query(read_window(read))))) == True{} : Bool}:
  WindowRead{+probe,+at} = read
  +query = probe_query(probe)
  +start = U32.to_nat(window_start(probe_window(probe)))
  +out = U32.to_nat(window_out(probe_window(probe)))
  +unit = Nat.div(channel(query),4n)
  +row = Nat.mod(channel(query),4n)
  +group = Nat.div(column(query),8n)
  +lane = Nat.mod(column(query),8n)
  +channel_split = selected_channel_split(query)
  +column_split = selected_column_split(query)
  +job = {Job{probe,unit,group,split_unit_inside(query,unit,row,channel_split),selected_unit_inside(query),selected_group_inside(query)} : Job}
  +row_inside = ViewWrites.clipped(row,4,(rows(query) - job_first(job) : U32),selected_row_small(query),
    Equal.trans(Bool,Nat.is_lt(row,U32.to_nat((rows(query) - job_first(job) : U32))),Nat.is_lt(row_number(unit,row),U32.to_nat(rows(query))),True{},
      row_bound(job,row),split_row_inside(query,unit,row,channel_split)))
  +column_at = Equal.trans(Nat,probe_column(probe),Nat.add(start,column(query)),Nat.add(start,lane_number(group,lane)),at,
    Equal.cong(Nat,Nat,value => Nat.add(start,value),column(query),lane_number(group,lane),column_split))
  +lane_inside = ViewWrites.clipped(lane,8,(window_out(probe_window(probe)) - job_position(job) : U32),selected_lane_small(query),
    Equal.trans(Bool,Nat.is_lt(lane,U32.to_nat((window_out(probe_window(probe)) - job_position(job) : U32))),Nat.is_lt(Nat.add(start,lane_number(group,lane)),out),True{},
      lane_bound(job,lane),
      %column_at : {Nat.is_lt(_,out) == True{} : Bool} probe_column_inside(probe)))
  +numbers = {
    %Equal.sym(Nat,channel(query),row_number(unit,row),channel_split) : {Nat.add(Nat.mul(_,out),probe_column(probe)) == cell_number(job,row,lane) : Nat}
    %Equal.sym(Nat,probe_column(probe),Nat.add(start,lane_number(group,lane)),column_at) : {Nat.add(Nat.mul(row_number(unit,row),out),_) == cell_number(job,row,lane) : Nat}
    {==} : {probe_number(probe) == cell_number(job,row,lane) : Nat}}
  +writes = ViewWrites.writes(~F32,~GemmWrites.Block,~GemmWrites.tile_value(~biased),Nat.mul(job_rows(job),job_lanes(job)),job_view(job),job_block(job))
  %Equal.sym(List<&2,Writes.Assignment<F32>>,
    GemmWrites.job_writes(~biased,Gemm.job(window_layout(job_window(job)),U32.from_nat(job_index(unit,group,units(query)))),input(query),weights(query),bias(query)),writes,
    job_writes_view(~biased,job)) :
    {WriteReadback.contains(~F32,probe_word(probe),_) == True{} : Bool}
  %Equal.sym(Nat,probe_number(probe),cell_number(job,row,lane),numbers) : {WriteReadback.contains(~F32,U32.from_nat(_),writes) == True{} : Bool}
  %cell_word(job,row,lane,row_inside,lane_inside) : {WriteReadback.contains(~F32,_,writes) == True{} : Bool}
  ViewWrites.covered_plane(~F32,~GemmWrites.Block,~GemmWrites.tile_value(~biased),job_rows(job),job_lanes(job),row,lane,job_view(job),job_block(job),row_inside,lane_inside)

# The writes of all jobs from first on contain an address that the writes
# of a later job contain.
def contains_step(~biased: Bool,+word: U32,+layout: Gemm.Layout,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,
  +rest: Nat,+first: Nat,+target: Nat,+equal: Bool,exact: {Nat.is_eq(first,target) == equal : Bool},
  +job: {WriteReadback.contains(~F32,word,GemmWrites.job_writes(~biased,Gemm.job(layout,U32.from_nat(target)),input,weights,bias)) == True{} : Bool},
  +after: {Nat.is_le(first,target) == True{} : Bool},
  later: {Nat.is_le(1n+first,target) == True{} : Bool} -> {WriteReadback.contains(~F32,word,GemmWrites.all_writes(~biased,rest,U32.from_nat(1n+first),layout,input,weights,bias)) == True{} : Bool}) ->
  {WriteReadback.contains(~F32,word,List.append(&2,Writes.Assignment<F32>,GemmWrites.job_writes(~biased,Gemm.job(layout,U32.from_nat(first)),input,weights,bias),
    GemmWrites.all_writes(~biased,rest,U32.from_nat(1n+first),layout,input,weights,bias))) == True{} : Bool}:
  match equal:
    case True{}:
      +here = GemmWrites.job_writes(~biased,Gemm.job(layout,U32.from_nat(first)),input,weights,bias)
      +rest_writes = GemmWrites.all_writes(~biased,rest,U32.from_nat(1n+first),layout,input,weights,bias)
      WriteReadback.covered_left(~F32,word,here,rest_writes,
        %Equal.sym(Nat,first,target,Order.equal_of_is_eq(first,target,exact)) :
          {WriteReadback.contains(~F32,word,GemmWrites.job_writes(~biased,Gemm.job(layout,U32.from_nat(_)),input,weights,bias)) == True{} : Bool}
        job)
    case False{}:
      +here = GemmWrites.job_writes(~biased,Gemm.job(layout,U32.from_nat(first)),input,weights,bias)
      +rest_writes = GemmWrites.all_writes(~biased,rest,U32.from_nat(1n+first),layout,input,weights,bias)
      WriteReadback.covered_right(~F32,word,here,rest_writes,later(step_after(first,target,after,exact)))
def all_contains(~biased: Bool,+word: U32,+layout: Gemm.Layout,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<F32>,
  +count: Nat,+first: Nat,+target: Nat,
  +job: {WriteReadback.contains(~F32,word,GemmWrites.job_writes(~biased,Gemm.job(layout,U32.from_nat(target)),input,weights,bias)) == True{} : Bool},
  +after: {Nat.is_le(first,target) == True{} : Bool},+before: {Nat.is_lt(target,Nat.add(first,count)) == True{} : Bool}) ->
  {WriteReadback.contains(~F32,word,GemmWrites.all_writes(~biased,count,U32.from_nat(first),layout,input,weights,bias)) == True{} : Bool}:
  match count:
    case 0n:
      +below = {%Addition.zero_right(first) : {Nat.is_lt(target,_) == True{} : Bool}
        before : {Nat.is_lt(target,first) == True{} : Bool}}
      Empty.absurd({WriteReadback.contains(~F32,word,[]) == True{} : Bool},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_eq(first,first),True{},
        Equal.sym(Bool,Nat.is_eq(first,first),False{},Order.less_not_equal(first,first,Order.le_lt_trans(first,target,first,after,below))),
        Order.equal_reflexive(first))))
    case 1n+ +rest:
      %Equal.sym(U32,(U32.from_nat(first) + 1 : U32),U32.from_nat(1n+first),Bounds.add_one(U32.from_nat(first))) :
        {WriteReadback.contains(~F32,word,List.append(&2,Writes.Assignment<F32>,GemmWrites.job_writes(~biased,Gemm.job(layout,U32.from_nat(first)),input,weights,bias),
          GemmWrites.all_writes(~biased,rest,_,layout,input,weights,bias))) == True{} : Bool}
      contains_step(~biased,word,layout,input,weights,bias,rest,first,target,Nat.is_eq(first,target),{==},job,after,
        next => all_contains(~biased,word,layout,input,weights,bias,rest,1n+first,target,job,next,
          %Addition.successor_right(first,rest) : {Nat.is_lt(target,_) == True{} : Bool}
          before))

# ---- Reads before the window ----

# A probe at a column before the window's start.
type Keep is Data:
  Keep{probe: Probe,before: {Nat.is_lt(probe_column(probe),U32.to_nat(window_start(probe_window(probe)))) == True{} : Bool}}
def keep_probe(+keep: Keep) -> Probe:
  Keep{probe,before} = keep
  probe
def keep_self(+keep: Keep) -> Keep:
  keep
def keep_window(+keep: Keep) -> OutputWindow:
  probe_window(keep_probe(keep))
def keep_capacity(+keep: Keep) -> U32:
  probe_capacity(keep_probe(keep))
def keep_word(+keep: Keep) -> U32:
  probe_word(keep_probe(keep))

# Every write of a run stays inside the output and away from the kept index.
def Kept(+keep: Keep,address: U32,value: F32) -> Data:
  Shape.Both<{U32.is_lt(address,keep_capacity(keep)) == True{} : Bool},{U32.is_eq(address,keep_word(keep)) == False{} : Bool}>
def kept_inside(+keep: Keep,+address: U32,+value: F32,holds: Kept(keep,address,value)) -> {U32.is_lt(address,keep_capacity(keep)) == True{} : Bool}:
  Shape.Both{inside,different} = holds
  inside
def kept_different(+keep: Keep,+address: U32,+value: F32,holds: Kept(keep,address,value)) -> {U32.is_eq(address,keep_word(keep)) == False{} : Bool}:
  Shape.Both{inside,different} = holds
  different

# A job of a kept read.
type KeepJob is Data:
  KeepJob{job: Job,before: {Nat.is_lt(probe_column(job_probe(job)),U32.to_nat(window_start(job_window(job)))) == True{} : Bool}}
def keep_job(+cells: KeepJob) -> Job:
  KeepJob{job,before} = cells
  job
def keep_of(+cells: KeepJob) -> Keep:
  KeepJob{job,before} = cells
  Keep{job_probe(job),before}
def keep_view(+cells: KeepJob) -> Views.View:
  job_view(keep_job(cells))
def keep_block(+cells: KeepJob) -> GemmWrites.Block:
  job_block(keep_job(cells))
def keep_rows(+cells: KeepJob) -> Nat:
  job_rows(keep_job(cells))
def keep_lanes(+cells: KeepJob) -> Nat:
  job_lanes(keep_job(cells))

def kept_compared(+job: Job,+before: {Nat.is_lt(probe_column(job_probe(job)),U32.to_nat(window_start(job_window(job)))) == True{} : Bool},
  +row: Nat,+lane: Nat,+row_inside: {Nat.is_lt(row,job_rows(job)) == True{} : Bool},+lane_inside: {Nat.is_lt(lane,job_lanes(job)) == True{} : Bool},equal: Bool,
  compared: {U32.is_eq(U32.from_nat(cell_number(job,row,lane)),probe_word(job_probe(job))) == equal : Bool}) ->
  {U32.is_eq(U32.from_nat(cell_number(job,row,lane)),probe_word(job_probe(job))) == False{} : Bool}:
  match equal:
    case False{}: compared
    case True{}:
      +start = U32.to_nat(window_start(job_window(job)))
      +column = probe_column(job_probe(job))
      +column_same = cell_column(job,row,lane,row_inside,lane_inside,compared)
      +reached = {
        %Equal.sym(Nat,column,Nat.add(start,lane_number(job_group(job),lane)),column_same) : {Nat.is_le(start,_) == True{} : Bool}
        Order.le_add(start,lane_number(job_group(job),lane)) : {Nat.is_le(start,column) == True{} : Bool}}
      Empty.absurd({U32.is_eq(U32.from_nat(cell_number(job,row,lane)),probe_word(job_probe(job))) == False{} : Bool},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_eq(start,start),True{},
          Equal.sym(Bool,Nat.is_eq(start,start),False{},Order.less_not_equal(start,start,Order.le_lt_trans(start,column,start,reached,before))),
          Order.equal_reflexive(start))))
def kept_cell(~biased: Bool,+cells: KeepJob,+row: Nat,+lane: Nat,+row_inside: {Nat.is_lt(row,keep_rows(cells)) == True{} : Bool},
  +lane_inside: {Nat.is_lt(lane,keep_lanes(cells)) == True{} : Bool}) ->
  Kept(keep_of(cells),Copy.address(keep_view(cells),Nat.add(Nat.mul(row,keep_lanes(cells)),lane)),
    GemmWrites.tile_value(~biased,keep_block(cells),Nat.add(Nat.mul(row,keep_lanes(cells)),lane))):
  KeepJob{+job,+before} = cells
  +probe = job_probe(job)
  +number = cell_number(job,row,lane)
  %Equal.sym(U32,Copy.address(job_view(job),Nat.add(Nat.mul(row,job_lanes(job)),lane)),U32.from_nat(number),cell_word(job,row,lane,row_inside,lane_inside)) :
    Kept(Keep{probe,before},_,GemmWrites.tile_value(~biased,job_block(job),Nat.add(Nat.mul(row,job_lanes(job)),lane)))
  Shape.Both{cell_bound(job,row,lane,row_inside,lane_inside),
    kept_compared(job,before,row,lane,row_inside,lane_inside,U32.is_eq(U32.from_nat(number),probe_word(probe)),{==})}
def keep_job_every(~biased: Bool,+keep: Keep,+unit: Nat,+group: Nat,
  +unit_rows: {Nat.is_lt(Nat.mul(unit,4n),U32.to_nat(rows(window_query(keep_window(keep))))) == True{} : Bool},
  +unit_inside: {Nat.is_lt(unit,units(window_query(keep_window(keep)))) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,groups(window_query(keep_window(keep)))) == True{} : Bool}) ->
  Every.Every(~F32,~Keep,~Kept,keep,GemmWrites.job_writes(~biased,Gemm.job(window_layout(keep_window(keep)),U32.from_nat(job_index(unit,group,units(window_query(keep_window(keep)))))),
    input(window_query(keep_window(keep))),weights(window_query(keep_window(keep))),bias(window_query(keep_window(keep))))):
  Keep{+probe,+before} = keep
  +job = {Job{probe,unit,group,unit_rows,unit_inside,group_inside} : Job}
  %Equal.sym(List<&2,Writes.Assignment<F32>>,
    GemmWrites.job_writes(~biased,Gemm.job(window_layout(job_window(job)),U32.from_nat(job_index(unit,group,units(job_query(job))))),input(job_query(job)),weights(job_query(job)),bias(job_query(job))),
    ViewWrites.writes(~F32,~GemmWrites.Block,~GemmWrites.tile_value(~biased),Nat.mul(job_rows(job),job_lanes(job)),job_view(job),job_block(job)),
    job_writes_view(~biased,job)) :
    Every.Every(~F32,~Keep,~Kept,Keep{probe,before},_)
  ViewWrites.every_plane(~F32,~GemmWrites.Block,~GemmWrites.tile_value(~biased),~KeepJob,~Keep,~keep_of,~Kept,~keep_view,~keep_block,~keep_rows,~keep_lanes,
    ~kept_cell(~biased),{KeepJob{job,before} : KeepJob})

# ---- Readback of a run ----

def capacity_fits(+output: Tree.Tree<F32>,+depth: Nat,+balanced: Shape.Balanced(F32,depth,output),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool}) ->
  {U32.is_le(Tree.capacity(F32,output),2147483648) == True{} : Bool}:
  %Equal.sym(U32,Tree.capacity(F32,output),U32.shln(1,depth),Shape.capacity(F32,depth,output,balanced)) : {U32.is_le(_,2147483648) == True{} : Bool}
  U32Shift.power_fits(depth,depth_ok)

# The writes of a window's jobs store the wanted value at an output index in
# the window.
def window_readback(~biased: Bool,+window: OutputWindow,+output: Tree.Tree<F32>,+depth: Nat,+balanced: Shape.Balanced(F32,depth,output),
  +depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +output_room: {Nat.is_le(Nat.mul(U32.to_nat(rows(window_query(window))),U32.to_nat(window_out(window))),U32.to_nat(Tree.capacity(F32,output))) == True{} : Bool},
  +inside: {Nat.is_lt(Nat.add(U32.to_nat(window_start(window)),column(window_query(window))),U32.to_nat(window_out(window))) == True{} : Bool}) ->
  {Tree.read(F32,Writes.pure(F32,GemmWrites.all_writes(~biased,Nat.mul(groups(window_query(window)),units(window_query(window))),0,window_layout(window),
      input(window_query(window)),weights(window_query(window)),bias(window_query(window))),output),
    U32.from_nat(Nat.add(Nat.mul(channel(window_query(window)),U32.to_nat(window_out(window))),Nat.add(U32.to_nat(window_start(window)),column(window_query(window)))))) ==
    wanted(~biased,window_query(window)) : F32}:
  +query = window_query(window)
  +jobs = Nat.mul(groups(query),units(query))
  +probe = {Probe{window,Tree.capacity(F32,output),capacity_fits(output,depth,balanced,depth_ok),output_room,Nat.add(U32.to_nat(window_start(window)),column(query)),inside} : Probe}
  +read = {WindowRead{probe,{==}} : WindowRead}
  +layout = window_layout(window)
  +writes = GemmWrites.all_writes(~biased,jobs,U32.from_nat(0n),layout,input(query),weights(query),bias(query))
  +holds = all_every(~biased,~WindowRead,~Written(~biased),~read_window,~read_job_every(~biased),read,jobs,0n,Order.reflexive(jobs))
  WriteReadback.covered(~F32,writes,depth,output,probe_word(probe),wanted(~biased,query),balanced,depth_ok,probe_bound(probe),
    Every.inside(~F32,~WindowRead,~Written(~biased),~WindowRead,~read_self,~read_capacity,~written_inside(~biased),read,writes,holds),
    Every.matching(~F32,~WindowRead,~Written(~biased),~WindowRead,~read_self,~read_word,~read_wanted(~biased),~written_matches(~biased),read,writes,holds),
    all_contains(~biased,probe_word(probe),layout,input(query),weights(query),bias(query),jobs,0n,read_index(read),read_covered(~biased,read),
      Order.zero_le(read_index(read)),RowMajor.radix(read_group(read),groups(query),read_unit(read),units(query),selected_group_inside(query),selected_unit_inside(query))))

# The writes of a window's jobs leave an output index before the window.
def window_preserves_before(~biased: Bool,+window: OutputWindow,+output: Tree.Tree<F32>,+depth: Nat,+column: Nat,+balanced: Shape.Balanced(F32,depth,output),
  +depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +output_room: {Nat.is_le(Nat.mul(U32.to_nat(rows(window_query(window))),U32.to_nat(window_out(window))),U32.to_nat(Tree.capacity(F32,output))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(window_out(window))) == True{} : Bool},+before: {Nat.is_lt(column,U32.to_nat(window_start(window))) == True{} : Bool}) ->
  {Tree.read(F32,Writes.pure(F32,GemmWrites.all_writes(~biased,Nat.mul(groups(window_query(window)),units(window_query(window))),0,window_layout(window),
      input(window_query(window)),weights(window_query(window)),bias(window_query(window))),output),
    U32.from_nat(Nat.add(Nat.mul(channel(window_query(window)),U32.to_nat(window_out(window))),column))) ==
    Tree.read(F32,output,U32.from_nat(Nat.add(Nat.mul(channel(window_query(window)),U32.to_nat(window_out(window))),column))) : F32}:
  +query = window_query(window)
  +jobs = Nat.mul(groups(query),units(query))
  +probe = {Probe{window,Tree.capacity(F32,output),capacity_fits(output,depth,balanced,depth_ok),output_room,column,column_inside} : Probe}
  +keep = {Keep{probe,before} : Keep}
  +writes = GemmWrites.all_writes(~biased,jobs,U32.from_nat(0n),window_layout(window),input(query),weights(query),bias(query))
  +holds = all_every(~biased,~Keep,~Kept,~keep_window,~keep_job_every(~biased),keep,jobs,0n,Order.reflexive(jobs))
  WriteReadback.excluded_readback(writes,output,depth,probe_word(probe),balanced,depth_ok,
    Every.inside(~F32,~Keep,~Kept,~Keep,~keep_self,~keep_capacity,~kept_inside,keep,writes,holds),probe_bound(probe),
    Every.excludes(~Keep,~Kept,~Keep,~keep_self,~keep_word,~kept_different,keep,writes,holds))

# A run stored from a start column into an output of out columns: an
# output in the window reads back its job value.
def run_at_readback(~biased: Bool,+query: Query,+start: U32,+out: U32,+output: Tree.Tree<F32>,+depth: Nat,
  +balanced: Shape.Balanced(F32,depth,output),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +output_room: {Nat.is_le(Nat.mul(U32.to_nat(rows(query)),U32.to_nat(out)),U32.to_nat(Tree.capacity(F32,output))) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(columns(query))),Nat.add(U32.to_nat(out),7n)) == True{} : Bool},
  +out_room: {Nat.is_le(U32.to_nat(out),U32.to_nat(limit(query))) == True{} : Bool},
  +query_inside: {Nat.is_lt(Nat.add(U32.to_nat(start),column(query)),U32.to_nat(out)) == True{} : Bool}) ->
  {Tree.read(F32,GemmLaws.run_at(~biased,rows(query),input(query),weights(query),bias(query),output,columns(query),inner(query),start,out),
    U32.from_nat(Nat.add(Nat.mul(channel(query),U32.to_nat(out)),Nat.add(U32.to_nat(start),column(query))))) == wanted(~biased,query) : F32}:
  +jobs = Nat.mul(groups(query),units(query))
  +layout = {Gemm.Layout{rows(query),columns(query),inner(query),start,out} : Gemm.Layout}
  +target = U32.from_nat(Nat.add(Nat.mul(channel(query),U32.to_nat(out)),Nat.add(U32.to_nat(start),column(query))))
  %Equal.sym(Tree.Tree<F32>,GemmLaws.run_at(~biased,rows(query),input(query),weights(query),bias(query),output,columns(query),inner(query),start,out),
    Writes.pure(F32,GemmWrites.all_writes(~biased,GemmWrites.job_count(rows(query),columns(query)),0,layout,input(query),weights(query),bias(query)),output),
    GemmWrites.run_at_writes(~biased,rows(query),input(query),weights(query),bias(query),output,columns(query),inner(query),start,out)) :
    {Tree.read(F32,_,target) == wanted(~biased,query) : F32}
  %Equal.sym(Nat,GemmWrites.job_count(rows(query),columns(query)),jobs,count_exact(query)) :
    {Tree.read(F32,Writes.pure(F32,GemmWrites.all_writes(~biased,_,0,layout,input(query),weights(query),bias(query)),output),target) == wanted(~biased,query) : F32}
  window_readback(~biased,OutputWindow{query,start,out,end_room,out_room},output,depth,balanced,depth_ok,output_room,query_inside)

# The same run leaves an output before its start column unchanged.
def run_at_preserves_before(~biased: Bool,+query: Query,+start: U32,+out: U32,+output: Tree.Tree<F32>,+depth: Nat,+column: Nat,
  +balanced: Shape.Balanced(F32,depth,output),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +output_room: {Nat.is_le(Nat.mul(U32.to_nat(rows(query)),U32.to_nat(out)),U32.to_nat(Tree.capacity(F32,output))) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(columns(query))),Nat.add(U32.to_nat(out),7n)) == True{} : Bool},
  +out_room: {Nat.is_le(U32.to_nat(out),U32.to_nat(limit(query))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,U32.to_nat(out)) == True{} : Bool},+before: {Nat.is_lt(column,U32.to_nat(start)) == True{} : Bool}) ->
  {Tree.read(F32,GemmLaws.run_at(~biased,rows(query),input(query),weights(query),bias(query),output,columns(query),inner(query),start,out),
      U32.from_nat(Nat.add(Nat.mul(channel(query),U32.to_nat(out)),column))) ==
    Tree.read(F32,output,U32.from_nat(Nat.add(Nat.mul(channel(query),U32.to_nat(out)),column))) : F32}:
  +jobs = Nat.mul(groups(query),units(query))
  +layout = {Gemm.Layout{rows(query),columns(query),inner(query),start,out} : Gemm.Layout}
  +target = U32.from_nat(Nat.add(Nat.mul(channel(query),U32.to_nat(out)),column))
  %Equal.sym(Tree.Tree<F32>,GemmLaws.run_at(~biased,rows(query),input(query),weights(query),bias(query),output,columns(query),inner(query),start,out),
    Writes.pure(F32,GemmWrites.all_writes(~biased,GemmWrites.job_count(rows(query),columns(query)),0,layout,input(query),weights(query),bias(query)),output),
    GemmWrites.run_at_writes(~biased,rows(query),input(query),weights(query),bias(query),output,columns(query),inner(query),start,out)) :
    {Tree.read(F32,_,target) == Tree.read(F32,output,target) : F32}
  %Equal.sym(Nat,GemmWrites.job_count(rows(query),columns(query)),jobs,count_exact(query)) :
    {Tree.read(F32,Writes.pure(F32,GemmWrites.all_writes(~biased,_,0,layout,input(query),weights(query),bias(query)),output),target) == Tree.read(F32,output,target) : F32}
  window_preserves_before(~biased,OutputWindow{query,start,out,end_room,out_room},output,depth,column,balanced,depth_ok,output_room,column_inside,before)

# A run into an output of its own columns: any output reads back its job
# value.
def run_readback(~biased: Bool,+query: Query,+output: Tree.Tree<F32>,+depth: Nat,+balanced: Shape.Balanced(F32,depth,output),
  +depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +output_room: {Nat.is_le(Nat.mul(U32.to_nat(rows(query)),U32.to_nat(columns(query))),U32.to_nat(Tree.capacity(F32,output))) == True{} : Bool}) ->
  {Tree.read(F32,GemmLaws.run(~biased,rows(query),input(query),weights(query),bias(query),output,columns(query),inner(query)),U32.from_nat(address(query))) ==
    wanted(~biased,query) : F32}:
  +jobs = Nat.mul(groups(query),units(query))
  +layout = {Gemm.Layout{rows(query),columns(query),inner(query),0,columns(query)} : Gemm.Layout}
  %Equal.sym(Tree.Tree<F32>,GemmLaws.run(~biased,rows(query),input(query),weights(query),bias(query),output,columns(query),inner(query)),
    Writes.pure(F32,GemmWrites.all_writes(~biased,GemmWrites.job_count(rows(query),columns(query)),0,layout,input(query),weights(query),bias(query)),output),
    GemmWrites.run_writes(~biased,rows(query),input(query),weights(query),bias(query),output,columns(query),inner(query))) :
    {Tree.read(F32,_,U32.from_nat(address(query))) == wanted(~biased,query) : F32}
  %Equal.sym(Nat,GemmWrites.job_count(rows(query),columns(query)),jobs,count_exact(query)) :
    {Tree.read(F32,Writes.pure(F32,GemmWrites.all_writes(~biased,_,0,layout,input(query),weights(query),bias(query)),output),U32.from_nat(address(query))) == wanted(~biased,query) : F32}
  window_readback(~biased,OutputWindow{query,0,columns(query),Order.le_add(U32.to_nat(columns(query)),7n),columns_within(query)},output,depth,balanced,depth_ok,output_room,
    column_inside(query))

# A complete eight-by-eight tile with zero bias uses the same job schedule.
def micro_query(+upper: Bool,+row: Accumulator.Row,+lane: Nat,+inner: U32,input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,+within: {Nat.is_lt(lane,8n) == True{} : Bool}) -> Query:
  Query{8,8,inner,1n,64,input,weights,Tree.fresh(F32,3n,0.0),PackedRows.patch_key(upper,row),lane,{==},{==},{==},{==},PackedRows.patch_key_inside(upper,row),within}

def micro_row_remainder(upper: Bool,row: Accumulator.Row) ->
  {Nat.mod(PackedRows.patch_key(upper,row),4n) == U32.to_nat(GemmWrites.row_number(row)) : Nat}:
  match upper row:
    case False{} Accumulator.R0{}: {==}
    case False{} Accumulator.R1{}: {==}
    case False{} Accumulator.R2{}: {==}
    case False{} Accumulator.R3{}: {==}
    case True{} Accumulator.R0{}: {==}
    case True{} Accumulator.R1{}: {==}
    case True{} Accumulator.R2{}: {==}
    case True{} Accumulator.R3{}: {==}

def micro_zero_bases(+upper: Bool,+inner: U32,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>) ->
  {Reduction.accumulator(~&2,~Tree.Tree<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~U32,~Rows.Rows<Tile.Tile8>,
    ReductionModel.selected(upper,U32.to_nat(inner),Reduction.State{input,weights,(0 * inner : U32),0,(0 * inner : U32),Rows.filled(~Tile.Tile8,Tile.filled(0.0))})) ==
    GemmWrites.half_accumulators(upper,U32.to_nat(inner),input,weights,Rows.filled(~Tile.Tile8,Tile.filled(0.0))) : Rows.Rows<Tile.Tile8>}:
  %Equal.sym(U32,(0 * inner : U32),0,Arithmetic.multiply_zero_left(inner)) :
    {Reduction.accumulator(~&2,~Tree.Tree<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~U32,~Rows.Rows<Tile.Tile8>,
      ReductionModel.selected(upper,U32.to_nat(inner),Reduction.State{input,weights,_,0,_,Rows.filled(~Tile.Tile8,Tile.filled(0.0))})) ==
      GemmWrites.half_accumulators(upper,U32.to_nat(inner),input,weights,Rows.filled(~Tile.Tile8,Tile.filled(0.0))) : Rows.Rows<Tile.Tile8>}
  {==}

def micro_accumulators_zero(upper: Bool,row: Accumulator.Row,+lane: Nat,+inner: U32,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+within: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {unit_accumulators(micro_query(upper,row,lane,inner,input,weights,within),Nat.div(PackedRows.patch_key(upper,row),4n),0n) ==
    GemmWrites.half_accumulators(upper,U32.to_nat(inner),input,weights,Rows.filled(~Tile.Tile8,Tile.filled(0.0))) : Rows.Rows<Tile.Tile8>}:
  match upper row:
    case False{} Accumulator.R0{}: micro_zero_bases(False{},inner,input,weights)
    case False{} Accumulator.R1{}: micro_zero_bases(False{},inner,input,weights)
    case False{} Accumulator.R2{}: micro_zero_bases(False{},inner,input,weights)
    case False{} Accumulator.R3{}: micro_zero_bases(False{},inner,input,weights)
    case True{} Accumulator.R0{}: micro_zero_bases(True{},inner,input,weights)
    case True{} Accumulator.R1{}: micro_zero_bases(True{},inner,input,weights)
    case True{} Accumulator.R2{}: micro_zero_bases(True{},inner,input,weights)
    case True{} Accumulator.R3{}: micro_zero_bases(True{},inner,input,weights)

def micro_accumulators(+upper: Bool,+row: Accumulator.Row,+lane: Nat,+inner: U32,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+within: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {unit_accumulators(micro_query(upper,row,lane,inner,input,weights,within),Nat.div(PackedRows.patch_key(upper,row),4n),Nat.div(lane,8n)) ==
    GemmWrites.half_accumulators(upper,U32.to_nat(inner),input,weights,Rows.filled(~Tile.Tile8,Tile.filled(0.0))) : Rows.Rows<Tile.Tile8>}:
  %Equal.sym(Nat,Nat.div(lane,8n),0n,Radix.quotient(0n,lane,8n,{==},within)) :
    {unit_accumulators(micro_query(upper,row,lane,inner,input,weights,within),Nat.div(PackedRows.patch_key(upper,row),4n),_) ==
      GemmWrites.half_accumulators(upper,U32.to_nat(inner),input,weights,Rows.filled(~Tile.Tile8,Tile.filled(0.0))) : Rows.Rows<Tile.Tile8>}
  micro_accumulators_zero(upper,row,lane,inner,input,weights,within)

def micro_selected_row(row: Accumulator.Row,+values: Rows.Rows<Tile.Tile8>) ->
  {Accumulator.select_row(~Tile.Tile8,U32.to_nat(GemmWrites.row_number(row)),values) == Accumulator.row(~Tile.Tile8,row,values) : Tile.Tile8}:
  match row values:
    case Accumulator.R0{} Rows.Rows{a,b,c,d}: {==}
    case Accumulator.R1{} Rows.Rows{a,b,c,d}: {==}
    case Accumulator.R2{} Rows.Rows{a,b,c,d}: {==}
    case Accumulator.R3{} Rows.Rows{a,b,c,d}: {==}

def micro_wanted(+upper: Bool,+row: Accumulator.Row,+lane: Nat,+inner: U32,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+within: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {wanted(~True{},micro_query(upper,row,lane,inner,input,weights,within)) ==
    (TileLaws.select_lane(lane,Accumulator.row(~Tile.Tile8,row,GemmWrites.half_accumulators(upper,U32.to_nat(inner),input,weights,Rows.filled(~Tile.Tile8,Tile.filled(0.0))))) + 0.0 : F32) : F32}:
  +query = micro_query(upper,row,lane,inner,input,weights,within)
  +values = GemmWrites.half_accumulators(upper,U32.to_nat(inner),input,weights,Rows.filled(~Tile.Tile8,Tile.filled(0.0)))
  %Equal.sym(Nat,Nat.mod(lane,8n),lane,Radix.remainder(0n,lane,8n,{==},within)) :
    {(select_lane(_,Accumulator.select_row(~Tile.Tile8,Nat.mod(PackedRows.patch_key(upper,row),4n),unit_accumulators(query,Nat.div(PackedRows.patch_key(upper,row),4n),Nat.div(lane,8n)))) + Tree.read(F32,Tree.fresh(F32,3n,0.0),U32.from_nat(PackedRows.patch_key(upper,row))) : F32) ==
      (TileLaws.select_lane(lane,Accumulator.row(~Tile.Tile8,row,values)) + 0.0 : F32) : F32}
  %Equal.sym(Nat,Nat.mod(PackedRows.patch_key(upper,row),4n),U32.to_nat(GemmWrites.row_number(row)),micro_row_remainder(upper,row)) :
    {(select_lane(lane,Accumulator.select_row(~Tile.Tile8,_,unit_accumulators(query,Nat.div(PackedRows.patch_key(upper,row),4n),Nat.div(lane,8n)))) + Tree.read(F32,Tree.fresh(F32,3n,0.0),U32.from_nat(PackedRows.patch_key(upper,row))) : F32) ==
      (TileLaws.select_lane(lane,Accumulator.row(~Tile.Tile8,row,values)) + 0.0 : F32) : F32}
  %Equal.sym(Rows.Rows<Tile.Tile8>,unit_accumulators(query,Nat.div(PackedRows.patch_key(upper,row),4n),Nat.div(lane,8n)),values,micro_accumulators(upper,row,lane,inner,input,weights,within)) :
    {(select_lane(lane,Accumulator.select_row(~Tile.Tile8,U32.to_nat(GemmWrites.row_number(row)),_)) + Tree.read(F32,Tree.fresh(F32,3n,0.0),U32.from_nat(PackedRows.patch_key(upper,row))) : F32) ==
      (TileLaws.select_lane(lane,Accumulator.row(~Tile.Tile8,row,values)) + 0.0 : F32) : F32}
  %Equal.sym(Tile.Tile8,Accumulator.select_row(~Tile.Tile8,U32.to_nat(GemmWrites.row_number(row)),values),Accumulator.row(~Tile.Tile8,row,values),micro_selected_row(row,values)) :
    {(select_lane(lane,_) + Tree.read(F32,Tree.fresh(F32,3n,0.0),U32.from_nat(PackedRows.patch_key(upper,row))) : F32) ==
      (TileLaws.select_lane(lane,Accumulator.row(~Tile.Tile8,row,values)) + 0.0 : F32) : F32}
  Equal.cong(F32,F32,bias => (TileLaws.select_lane(lane,Accumulator.row(~Tile.Tile8,row,values)) + bias : F32),Tree.read(F32,Tree.fresh(F32,3n,0.0),U32.from_nat(PackedRows.patch_key(upper,row))),0.0,
    TreeRead.read_fresh(~F32,3n,0.0,U32.from_nat(PackedRows.patch_key(upper,row))))

# The observable tile is the ordered dot followed by one addition of +0.
def micro_dot_readback(+upper: Bool,+row: Accumulator.Row,+lane: Nat,+inner: U32,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+within: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {Tree.read(F32,GemmLaws.run(~True{},8,input,weights,Tree.fresh(F32,3n,0.0),Tree.fresh(F32,6n,0.0),8,inner),U32.from_nat(MicroDot.micro_index(upper,row,lane))) ==
    (MicroDot.natural_dot(upper,row,lane,U32.to_nat(inner),input,weights,Rows.filled(~Tile.Tile8,Tile.filled(0.0))) + 0.0 : F32) : F32}:
  +query = micro_query(upper,row,lane,inner,input,weights,within)
  +values = GemmWrites.half_accumulators(upper,U32.to_nat(inner),input,weights,Rows.filled(~Tile.Tile8,Tile.filled(0.0)))
  Equal.trans(F32,Tree.read(F32,GemmLaws.run(~True{},8,input,weights,Tree.fresh(F32,3n,0.0),Tree.fresh(F32,6n,0.0),8,inner),U32.from_nat(MicroDot.micro_index(upper,row,lane))),wanted(~True{},query),
    (MicroDot.natural_dot(upper,row,lane,U32.to_nat(inner),input,weights,Rows.filled(~Tile.Tile8,Tile.filled(0.0))) + 0.0 : F32),
    run_readback(~True{},query,Tree.fresh(F32,6n,0.0),6n,Shape.fresh(F32,6n,0.0),{==},{==}),
    Equal.trans(F32,wanted(~True{},query),(TileLaws.select_lane(lane,Accumulator.row(~Tile.Tile8,row,values)) + 0.0 : F32),
      (MicroDot.natural_dot(upper,row,lane,U32.to_nat(inner),input,weights,Rows.filled(~Tile.Tile8,Tile.filled(0.0))) + 0.0 : F32),micro_wanted(upper,row,lane,inner,input,weights,within),
      Equal.cong(F32,F32,value => (value + 0.0 : F32),TileLaws.select_lane(lane,Accumulator.row(~Tile.Tile8,row,values)),MicroDot.natural_dot(upper,row,lane,U32.to_nat(inner),input,weights,Rows.filled(~Tile.Tile8,Tile.filled(0.0))),
        Equal.trans(F32,TileLaws.select_lane(lane,Accumulator.row(~Tile.Tile8,row,values)),MicroDot.half_dot(upper,row,lane,U32.to_nat(inner),input,weights,Rows.filled(~Tile.Tile8,Tile.filled(0.0))),MicroDot.natural_dot(upper,row,lane,U32.to_nat(inner),input,weights,Rows.filled(~Tile.Tile8,Tile.filled(0.0))),
          MicroDot.half_dot_matches(upper,row,lane,U32.to_nat(inner),input,weights,Rows.filled(~Tile.Tile8,Tile.filled(0.0))),MicroDot.half_dot_as_natural(upper,row,lane,U32.to_nat(inner),input,weights,Rows.filled(~Tile.Tile8,Tile.filled(0.0)))))))

