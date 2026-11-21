# A tile product reads back, at every tile of its window, the ordered dots of
# its taps finished with the channel biases, and leaves the columns before
# the window as they were.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./write_program.bend as Writes
import ./kernel_gemm_refinement.bend as GemmLaws
import ./kernel_gemm_jobs.bend as GemmJobs
import ./kernel_gemm_spec.bend as GemmSpec
import ./view_writes.bend as ViewWrites
import ./binary_radix.bend as Radix
import ./row_major_index.bend as RowMajor
import ./natural_addition.bend as Addition
import ./nat_order.bend as Order
import ./u32_comparison.bend as Comparison
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_arithmetic.bend as WordArithmetic
import ./word_multiplication.bend as Multiplication
import ./word_encoding.bend as WordEncoding
import ./u32_division.bend as U32Division
import ./nat_division_order.bend as DivisionOrder
import ./nat_division.bend as NaturalDivision
import ./nat_extrema.bend as Extrema
import ./u32_extrema.bend as U32Extrema
import ./kernel_tile_refinement.bend as TileLaws
import ./write_program_every.bend as Every
import ./write_program_fold.bend as Fold
import ./write_program_readback.bend as WriteReadback

# ---- The cells of a product ----

# A product of rows channels by columns positions (columns a multiple of
# eight), stored from column start of rows output columns wide. Its word
# arithmetic stays below limit.
type Product is Data:
  Product{rows: U32,columns: U32,inner: U32,start: U32,output: U32,row_tiles: Tree.Tree<Tile.Tile8>,lane_tiles: Tree.Tree<Tile.Tile8>,bias: Tree.Tree<F32>,
    limit: U32,groups: Nat,
    limit_ok: {U32.is_le(limit,2147483648) == True{} : Bool},
    columns_exact: {U32.to_nat(columns) == Nat.mul(groups,8n) : Nat},
    rows_room: {Nat.is_le(Nat.add(U32.to_nat(rows),7n),U32.to_nat(limit)) == True{} : Bool},
    area_room: {Nat.is_le(Nat.mul(U32.to_nat(rows),U32.to_nat(output)),U32.to_nat(limit)) == True{} : Bool},
    end_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(columns)),Nat.add(U32.to_nat(output),7n)) == True{} : Bool},
    output_room: {Nat.is_le(U32.to_nat(output),U32.to_nat(limit)) == True{} : Bool}}

def product_rows(+product: Product) -> U32:
  Product{rows,columns,inner,start,output,row_tiles,lane_tiles,bias,limit,groups,limit_ok,columns_exact,rows_room,area_room,end_room,output_room} = product
  rows

def product_columns(+product: Product) -> U32:
  Product{rows,columns,inner,start,output,row_tiles,lane_tiles,bias,limit,groups,limit_ok,columns_exact,rows_room,area_room,end_room,output_room} = product
  columns

def product_inner(+product: Product) -> U32:
  Product{rows,columns,inner,start,output,row_tiles,lane_tiles,bias,limit,groups,limit_ok,columns_exact,rows_room,area_room,end_room,output_room} = product
  inner

def product_start(+product: Product) -> U32:
  Product{rows,columns,inner,start,output,row_tiles,lane_tiles,bias,limit,groups,limit_ok,columns_exact,rows_room,area_room,end_room,output_room} = product
  start

def product_output(+product: Product) -> U32:
  Product{rows,columns,inner,start,output,row_tiles,lane_tiles,bias,limit,groups,limit_ok,columns_exact,rows_room,area_room,end_room,output_room} = product
  output

def product_row_tiles(+product: Product) -> Tree.Tree<Tile.Tile8>:
  Product{rows,columns,inner,start,output,row_tiles,lane_tiles,bias,limit,groups,limit_ok,columns_exact,rows_room,area_room,end_room,output_room} = product
  row_tiles

def product_lane_tiles(+product: Product) -> Tree.Tree<Tile.Tile8>:
  Product{rows,columns,inner,start,output,row_tiles,lane_tiles,bias,limit,groups,limit_ok,columns_exact,rows_room,area_room,end_room,output_room} = product
  lane_tiles

def product_limit(+product: Product) -> U32:
  Product{rows,columns,inner,start,output,row_tiles,lane_tiles,bias,limit,groups,limit_ok,columns_exact,rows_room,area_room,end_room,output_room} = product
  limit

def product_groups(+product: Product) -> Nat:
  Product{rows,columns,inner,start,output,row_tiles,lane_tiles,bias,limit,groups,limit_ok,columns_exact,rows_room,area_room,end_room,output_room} = product
  groups

def limit_fits(+product: Product) -> {U32.is_le(product_limit(product),2147483648) == True{} : Bool}:
  Product{rows,columns,inner,start,output,row_tiles,lane_tiles,bias,limit,groups,limit_ok,columns_exact,rows_room,area_room,end_room,output_room} = product
  limit_ok

def columns_grouped(+product: Product) -> {U32.to_nat(product_columns(product)) == Nat.mul(product_groups(product),8n) : Nat}:
  Product{rows,columns,inner,start,output,row_tiles,lane_tiles,bias,limit,groups,limit_ok,columns_exact,rows_room,area_room,end_room,output_room} = product
  columns_exact

def rows_fit(+product: Product) -> {Nat.is_le(Nat.add(U32.to_nat(product_rows(product)),7n),U32.to_nat(product_limit(product))) == True{} : Bool}:
  Product{rows,columns,inner,start,output,row_tiles,lane_tiles,bias,limit,groups,limit_ok,columns_exact,rows_room,area_room,end_room,output_room} = product
  rows_room

def area_fits(+product: Product) -> {Nat.is_le(Nat.mul(U32.to_nat(product_rows(product)),U32.to_nat(product_output(product))),U32.to_nat(product_limit(product))) == True{} : Bool}:
  Product{rows,columns,inner,start,output,row_tiles,lane_tiles,bias,limit,groups,limit_ok,columns_exact,rows_room,area_room,end_room,output_room} = product
  area_room

def end_fits(+product: Product) ->
  {Nat.is_le(Nat.add(U32.to_nat(product_start(product)),U32.to_nat(product_columns(product))),Nat.add(U32.to_nat(product_output(product)),7n)) == True{} : Bool}:
  Product{rows,columns,inner,start,output,row_tiles,lane_tiles,bias,limit,groups,limit_ok,columns_exact,rows_room,area_room,end_room,output_room} = product
  end_room

def output_fits(+product: Product) -> {Nat.is_le(U32.to_nat(product_output(product)),U32.to_nat(product_limit(product))) == True{} : Bool}:
  Product{rows,columns,inner,start,output,row_tiles,lane_tiles,bias,limit,groups,limit_ok,columns_exact,rows_room,area_room,end_room,output_room} = product
  output_room

def row_count(+product: Product) -> Nat:
  U32.to_nat(product_rows(product))

def output_width(+product: Product) -> Nat:
  U32.to_nat(product_output(product))

def start_column(+product: Product) -> Nat:
  U32.to_nat(product_start(product))

def limit_count(+product: Product) -> Nat:
  U32.to_nat(product_limit(product))

def units(+product: Product) -> Nat:
  U32.to_nat(Gemm.blocks(product_rows(product)))

def area(+product: Product) -> Nat:
  Nat.mul(row_count(product),output_width(product))

# ---- Units of eight rows and groups of eight columns ----

def eight_room(+value: Nat) -> {Nat.is_le(Nat.double(value),Nat.mul(value,8n)) == True{} : Bool}:
  match value:
    case 0n: {==}
    case 1n+ +rest:
      Order.successor(Nat.double(rest),5n+Nat.mul(rest,8n),Order.successor(Nat.double(rest),4n+Nat.mul(rest,8n),Order.successor(Nat.double(rest),3n+Nat.mul(rest,8n),
        Order.successor(Nat.double(rest),2n+Nat.mul(rest,8n),Order.successor(Nat.double(rest),1n+Nat.mul(rest,8n),Order.successor(Nat.double(rest),Nat.mul(rest,8n),eight_room(rest)))))))

# A count of eights that fits in total plus seven lies below the total.
def eighth_below(+total: Nat,+count: Nat,room: {Nat.is_le(Nat.mul(1n+count,8n),Nat.add(total,7n)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.mul(count,8n),total) == True{} : Bool}:
  Bounds.predecessor_strict(Nat.mul(count,8n),total,
    %Order.commutative(total,7n) : {Nat.is_le(Nat.mul(1n+count,8n),_) == True{} : Bool}
    room)

def eighths_within(+total: Nat,+count: Nat,room: {Nat.is_le(Nat.mul(count,8n),Nat.add(total,7n)) == True{} : Bool}) -> {Nat.is_le(count,total) == True{} : Bool}:
  match count:
    case 0n: Order.zero_le(total)
    case 1n+ +rest:
      Order.strict_follow(rest,total,Order.le_lt_trans(rest,Nat.mul(rest,8n),total,RowMajor.product_covers(rest,8n,{==}),eighth_below(total,rest,room)))

def padded_rows(+product: Product) -> {U32.to_nat((product_rows(product) + 7 : U32)) == Nat.add(row_count(product),7n) : Nat}:
  Arithmetic.add(product_rows(product),7,product_limit(product),limit_fits(product),rows_fit(product))

def units_exact(+product: Product) -> {units(product) == Nat.div(Nat.add(row_count(product),7n),8n) : Nat}:
  Equal.trans(Nat,units(product),Nat.div(U32.to_nat((product_rows(product) + 7 : U32)),8n),Nat.div(Nat.add(row_count(product),7n),8n),
    U32Division.divide((product_rows(product) + 7 : U32),8,2147483648,{==},{==},{==}),
    Equal.cong(Nat,Nat,value => Nat.div(value,8n),U32.to_nat((product_rows(product) + 7 : U32)),Nat.add(row_count(product),7n),padded_rows(product)))

def padded_identity(+product: Product) ->
  {Nat.add(row_count(product),7n) == Nat.add(Nat.mul(Nat.div(Nat.add(row_count(product),7n),8n),8n),Nat.mod(Nat.add(row_count(product),7n),8n)) : Nat}:
  %padded_rows(product) : {_ == Nat.add(Nat.mul(Nat.div(_,8n),8n),Nat.mod(_,8n)) : Nat}
  Radix.division_identity((product_rows(product) + 7 : U32),8n,{==})

def units_room(+product: Product) -> {Nat.is_le(Nat.mul(units(product),8n),Nat.add(row_count(product),7n)) == True{} : Bool}:
  +total = Nat.add(row_count(product),7n)
  +quotient = Nat.div(total,8n)
  %Equal.sym(Nat,units(product),quotient,units_exact(product)) : {Nat.is_le(Nat.mul(_,8n),total) == True{} : Bool}
  %Equal.sym(Nat,total,Nat.add(Nat.mul(quotient,8n),Nat.mod(total,8n)),padded_identity(product)) : {Nat.is_le(Nat.mul(quotient,8n),_) == True{} : Bool}
  Order.le_add(Nat.mul(quotient,8n),Nat.mod(total,8n))

def unit_rows(+product: Product,+unit: Nat,+inside: {Nat.is_lt(unit,units(product)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.mul(unit,8n),row_count(product)) == True{} : Bool}:
  eighth_below(row_count(product),unit,
    Order.transitive(Nat.mul(1n+unit,8n),Nat.mul(units(product),8n),Nat.add(row_count(product),7n),
      RowMajor.mul_le(1n+unit,units(product),8n,Order.strict_follow(unit,units(product),inside)),units_room(product)))

def double_units(+product: Product) -> {Nat.is_le(Nat.double(units(product)),limit_count(product)) == True{} : Bool}:
  Order.transitive(Nat.double(units(product)),Nat.mul(units(product),8n),limit_count(product),eight_room(units(product)),
    Order.transitive(Nat.mul(units(product),8n),Nat.add(row_count(product),7n),limit_count(product),units_room(product),rows_fit(product)))

def groups_room(+product: Product) -> {Nat.is_le(Nat.mul(product_groups(product),8n),Nat.add(output_width(product),7n)) == True{} : Bool}:
  +columns = U32.to_nat(product_columns(product))
  %columns_grouped(product) : {Nat.is_le(_,Nat.add(output_width(product),7n)) == True{} : Bool}
  Order.transitive(columns,Nat.add(start_column(product),columns),Nat.add(output_width(product),7n),
    %Order.commutative(columns,start_column(product)) : {Nat.is_le(columns,_) == True{} : Bool}
    Order.le_add(columns,start_column(product)),end_fits(product))

# Every group of a product starts inside the output row.
def group_start(+product: Product,+group: Nat,+inside: {Nat.is_lt(group,product_groups(product)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(start_column(product),Nat.mul(group,8n)),output_width(product)) == True{} : Bool}:
  +first = start_column(product)
  +width = output_width(product)
  +offset = Nat.mul(group,8n)
  +columns = U32.to_nat(product_columns(product))
  +covered = Order.transitive(Nat.add(Nat.mul(1n+group,8n),first),Nat.add(columns,first),Nat.add(width,7n),
    Order.add_le(Nat.mul(1n+group,8n),columns,first,
      %Equal.sym(Nat,columns,Nat.mul(product_groups(product),8n),columns_grouped(product)) : {Nat.is_le(Nat.mul(1n+group,8n),_) == True{} : Bool}
      RowMajor.mul_le(1n+group,product_groups(product),8n,Order.strict_follow(group,product_groups(product),inside))),
    %Order.commutative(first,columns) : {Nat.is_le(_,Nat.add(width,7n)) == True{} : Bool}
    end_fits(product))
  %Order.commutative(offset,first) : {Nat.is_lt(_,width) == True{} : Bool}
  Bounds.predecessor_strict(Nat.add(offset,first),width,
    %Order.commutative(width,7n) : {Nat.is_le(8n+Nat.add(offset,first),_) == True{} : Bool}
    covered)

def jobs_within(+product: Product) -> {Nat.is_le(Nat.mul(product_groups(product),units(product)),limit_count(product)) == True{} : Bool}:
  +rows = row_count(product)
  +width = output_width(product)
  +count = units(product)
  Order.transitive(Nat.mul(product_groups(product),count),Nat.mul(rows,width),limit_count(product),
    Order.transitive(Nat.mul(product_groups(product),count),Nat.mul(width,count),Nat.mul(rows,width),
      RowMajor.mul_le(product_groups(product),width,count,eighths_within(width,product_groups(product),groups_room(product))),
      %RowMajor.multiply_commutative(count,width) : {Nat.is_le(_,Nat.mul(rows,width)) == True{} : Bool}
      RowMajor.mul_le(count,rows,width,eighths_within(rows,count,units_room(product)))),
    area_fits(product))

# ---- The job of a unit and a group ----

# The number of the job of a unit of eight rows and a group of eight columns:
# units within groups, or groups within units when the weights pass once.
def index_of(once: Bool,+unit: Nat,+group: Nat,+count: Nat,+groups: Nat) -> Nat:
  match once:
    case True{}: Nat.add(Nat.mul(unit,groups),group)
    case False{}: Nat.add(Nat.mul(group,count),unit)

def index_below(once: Bool,+unit: Nat,+group: Nat,+count: Nat,+groups: Nat,+unit_inside: {Nat.is_lt(unit,count) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups) == True{} : Bool}) ->
  {Nat.is_lt(index_of(once,unit,group,count,groups),Nat.mul(groups,count)) == True{} : Bool}:
  match once:
    case True{}:
      %RowMajor.multiply_commutative(count,groups) : {Nat.is_lt(Nat.add(Nat.mul(unit,groups),group),_) == True{} : Bool}
      RowMajor.radix(unit,count,group,groups,unit_inside,group_inside)
    case False{}: RowMajor.radix(group,groups,unit,count,group_inside,unit_inside)

# The unit a job number names.
def job_group_exact(once: Bool,+word: U32,+count_word: U32,+groups_word: U32,+unit: Nat,+group: Nat,+count: Nat,+groups: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +number: {U32.to_nat(word) == index_of(once,unit,group,count,groups) : Nat},+count_exact: {U32.to_nat(count_word) == count : Nat},+groups_exact: {U32.to_nat(groups_word) == groups : Nat},
  +unit_inside: {Nat.is_lt(unit,count) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups) == True{} : Bool},
  +double_count: {Nat.is_le(Nat.double(count),U32.to_nat(cap)) == True{} : Bool},+double_groups: {Nat.is_le(Nat.double(groups),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat(Gemm.job_group(once,word,count_word,groups_word)) == unit : Nat}:
  match once:
    case True{}: Radix.quotient_word(word,groups_word,unit,group,groups,cap,cap_ok,number,groups_exact,Order.positive_bound(group,groups,group_inside),group_inside,double_groups)
    case False{}: Radix.remainder_word(word,count_word,group,unit,count,cap,cap_ok,number,count_exact,Order.positive_bound(unit,count,unit_inside),unit_inside,double_count)

# The group a job number names.
def job_unit_exact(once: Bool,+word: U32,+count_word: U32,+groups_word: U32,+unit: Nat,+group: Nat,+count: Nat,+groups: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +number: {U32.to_nat(word) == index_of(once,unit,group,count,groups) : Nat},+count_exact: {U32.to_nat(count_word) == count : Nat},+groups_exact: {U32.to_nat(groups_word) == groups : Nat},
  +unit_inside: {Nat.is_lt(unit,count) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups) == True{} : Bool},
  +double_count: {Nat.is_le(Nat.double(count),U32.to_nat(cap)) == True{} : Bool},+double_groups: {Nat.is_le(Nat.double(groups),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat(Gemm.job_unit(once,word,count_word,groups_word)) == group : Nat}:
  match once:
    case True{}: Radix.remainder_word(word,groups_word,unit,group,groups,cap,cap_ok,number,groups_exact,Order.positive_bound(group,groups,group_inside),group_inside,double_groups)
    case False{}: Radix.quotient_word(word,count_word,group,unit,count,cap,cap_ok,number,count_exact,Order.positive_bound(unit,count,unit_inside),unit_inside,double_count)

# Twice a count whose eighths fit seven above a total of at least seven.
def eighths_doubled(+count: Nat,+total: Nat,+seven: {Nat.is_le(7n,total) == True{} : Bool},room: {Nat.is_le(Nat.mul(count,8n),Nat.add(total,7n)) == True{} : Bool}) ->
  {Nat.is_le(Nat.double(count),total) == True{} : Bool}:
  match count:
    case 0n: Order.zero_le(total)
    case 1n: Order.transitive(2n,7n,total,{==},seven)
    case 2n+ +rest:
      +within = {%Order.commutative(total,7n) : {Nat.is_le(Nat.mul(2n+rest,8n),_) == True{} : Bool} room : {Nat.is_le(9n+Nat.mul(rest,8n),total) == True{} : Bool}}
      Order.transitive(4n+Nat.double(rest),9n+Nat.mul(rest,8n),total,
        Order.successor(Nat.double(rest),4n+Nat.mul(rest,8n),Order.successor(Nat.double(rest),3n+Nat.mul(rest,8n),Order.successor(Nat.double(rest),2n+Nat.mul(rest,8n),
          Order.successor(Nat.double(rest),1n+Nat.mul(rest,8n),Order.successor(Nat.double(rest),Nat.mul(rest,8n),eight_room(rest)))))),within)

def double_groups(+product: Product) -> {Nat.is_le(Nat.double(product_groups(product)),limit_count(product)) == True{} : Bool}:
  eighths_doubled(product_groups(product),limit_count(product),
    Order.transitive(7n,Nat.add(row_count(product),7n),limit_count(product),
      %Order.commutative(7n,row_count(product)) : {Nat.is_le(7n,_) == True{} : Bool} Order.le_add(7n,row_count(product)),rows_fit(product)),
    Order.transitive(Nat.mul(product_groups(product),8n),Nat.add(output_width(product),7n),Nat.add(limit_count(product),7n),groups_room(product),
      Order.add_le(output_width(product),limit_count(product),7n,output_fits(product))))

def groups_exact(+product: Product) -> {U32.to_nat((product_columns(product) / 8 : U32)) == product_groups(product) : Nat}:
  Equal.trans(Nat,U32.to_nat((product_columns(product) / 8 : U32)),Nat.div(U32.to_nat(product_columns(product)),8n),product_groups(product),
    U32Division.divide(product_columns(product),8,2147483648,{==},{==},{==}),
    %Equal.sym(Nat,U32.to_nat(product_columns(product)),Nat.mul(product_groups(product),8n),columns_grouped(product)) : {Nat.div(_,8n) == product_groups(product) : Nat}
    %Addition.zero_right(Nat.mul(product_groups(product),8n)) : {Nat.div(_,8n) == product_groups(product) : Nat}
    Radix.quotient(product_groups(product),0n,8n,{==},{==}))

def once(+product: Product) -> Bool:
  Gemm.streamed(Gemm.blocks(product_rows(product)),product_inner(product))

def index_number(+product: Product,+unit: Nat,+group: Nat) -> Nat:
  index_of(once(product),unit,group,units(product),product_groups(product))

def job_index(+product: Product,+unit: Nat,+group: Nat) -> U32:
  U32.from_nat(index_number(product,unit,group))

def unit_word(+product: Product,+index: U32) -> U32:
  Gemm.job_group(once(product),index,Gemm.blocks(product_rows(product)),(product_columns(product) / 8 : U32))

def group_word(+product: Product,+index: U32) -> U32:
  Gemm.job_unit(once(product),index,Gemm.blocks(product_rows(product)),(product_columns(product) / 8 : U32))

def position_word(+product: Product,+index: U32) -> U32:
  (product_start(product) + group_word(product,index) * 8 : U32)

def index_exact(+product: Product,+unit: Nat,+group: Nat,+unit_inside: {Nat.is_lt(unit,units(product)) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,product_groups(product)) == True{} : Bool}) -> {U32.to_nat(job_index(product,unit,group)) == index_number(product,unit,group) : Nat}:
  +jobs = Nat.mul(product_groups(product),units(product))
  Bounds.roundtrip(index_number(product,unit,group),product_limit(product),limit_fits(product),
    Order.transitive(index_number(product,unit,group),jobs,limit_count(product),
      Bounds.strict_le(index_number(product,unit,group),jobs,index_below(once(product),unit,group,units(product),product_groups(product),unit_inside,group_inside)),jobs_within(product)))

def unit_exact(+product: Product,+unit: Nat,+group: Nat,+unit_inside: {Nat.is_lt(unit,units(product)) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,product_groups(product)) == True{} : Bool}) -> {U32.to_nat(unit_word(product,job_index(product,unit,group))) == unit : Nat}:
  job_group_exact(once(product),job_index(product,unit,group),Gemm.blocks(product_rows(product)),(product_columns(product) / 8 : U32),unit,group,units(product),product_groups(product),product_limit(product),limit_fits(product),
    index_exact(product,unit,group,unit_inside,group_inside),{==},groups_exact(product),unit_inside,group_inside,double_units(product),double_groups(product))

def group_exact(+product: Product,+unit: Nat,+group: Nat,+unit_inside: {Nat.is_lt(unit,units(product)) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,product_groups(product)) == True{} : Bool}) -> {U32.to_nat(group_word(product,job_index(product,unit,group))) == group : Nat}:
  job_unit_exact(once(product),job_index(product,unit,group),Gemm.blocks(product_rows(product)),(product_columns(product) / 8 : U32),unit,group,units(product),product_groups(product),product_limit(product),limit_fits(product),
    index_exact(product,unit,group,unit_inside,group_inside),{==},groups_exact(product),unit_inside,group_inside,double_units(product),double_groups(product))

def unit_value(+product: Product,+unit: Nat,+group: Nat,+unit_inside: {Nat.is_lt(unit,units(product)) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,product_groups(product)) == True{} : Bool}) -> {unit_word(product,job_index(product,unit,group)) == U32.from_nat(unit) : U32}:
  +word = unit_word(product,job_index(product,unit,group))
  Equal.sym(U32,U32.from_nat(unit),word,Arithmetic.from_nat_exact(word,unit,unit_exact(product,unit,group,unit_inside,group_inside)))

def group_value(+product: Product,+unit: Nat,+group: Nat,+unit_inside: {Nat.is_lt(unit,units(product)) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,product_groups(product)) == True{} : Bool}) -> {group_word(product,job_index(product,unit,group)) == U32.from_nat(group) : U32}:
  +word = group_word(product,job_index(product,unit,group))
  Equal.sym(U32,U32.from_nat(group),word,Arithmetic.from_nat_exact(word,group,group_exact(product,unit,group,unit_inside,group_inside)))

def position_exact(+product: Product,+unit: Nat,+group: Nat,+unit_inside: {Nat.is_lt(unit,units(product)) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,product_groups(product)) == True{} : Bool}) ->
  {U32.to_nat(position_word(product,job_index(product,unit,group))) == Nat.add(start_column(product),Nat.mul(group,8n)) : Nat}:
  +word = group_word(product,job_index(product,unit,group))
  +exact = group_exact(product,unit,group,unit_inside,group_inside)
  +first = start_column(product)
  +offset = Nat.mul(group,8n)
  +room = Order.transitive(Nat.add(first,offset),output_width(product),limit_count(product),
    Bounds.strict_le(Nat.add(first,offset),output_width(product),group_start(product,group,group_inside)),output_fits(product))
  +offset_room = Order.transitive(offset,Nat.add(first,offset),limit_count(product),
    %Order.commutative(offset,first) : {Nat.is_le(offset,_) == True{} : Bool} Order.le_add(offset,first),room)
  +scaled = (word * 8 : U32)
  +scaled_exact = Equal.trans(Nat,U32.to_nat(scaled),Nat.mul(U32.to_nat(word),8n),offset,
    Multiplication.bounded_multiply(word,8,product_limit(product),limit_fits(product),
      %Equal.sym(Nat,U32.to_nat(word),group,exact) : {Nat.is_le(Nat.mul(_,8n),limit_count(product)) == True{} : Bool} offset_room),
    Equal.cong(Nat,Nat,value => Nat.mul(value,8n),U32.to_nat(word),group,exact))
  Equal.trans(Nat,U32.to_nat((product_start(product) + scaled : U32)),Nat.add(first,U32.to_nat(scaled)),Nat.add(first,offset),
    Arithmetic.add(product_start(product),scaled,product_limit(product),limit_fits(product),
      %Equal.sym(Nat,U32.to_nat(scaled),offset,scaled_exact) : {Nat.is_le(Nat.add(first,_),limit_count(product)) == True{} : Bool} room),
    Equal.cong(Nat,Nat,value => Nat.add(first,value),U32.to_nat(scaled),offset,scaled_exact))

def lanes_left(+product: Product,+unit: Nat,+group: Nat,+unit_inside: {Nat.is_lt(unit,units(product)) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,product_groups(product)) == True{} : Bool}) ->
  {U32.to_nat((product_output(product) - position_word(product,job_index(product,unit,group)) : U32)) ==
    Nat.sub(output_width(product),Nat.add(start_column(product),Nat.mul(group,8n))) : Nat}:
  +position = position_word(product,job_index(product,unit,group))
  +exact = position_exact(product,unit,group,unit_inside,group_inside)
  +width = output_width(product)
  +first = Nat.add(start_column(product),Nat.mul(group,8n))
  Equal.trans(Nat,U32.to_nat((product_output(product) - position : U32)),Nat.sub(width,U32.to_nat(position)),Nat.sub(width,first),
    Arithmetic.subtract(product_output(product),position,product_limit(product),limit_fits(product),
      %Equal.sym(Nat,U32.to_nat(position),first,exact) : {Nat.is_le(_,width) == True{} : Bool}
      Bounds.strict_le(first,width,group_start(product,group,group_inside)),
      Order.transitive(Nat.sub(width,U32.to_nat(position)),width,limit_count(product),Order.sub_le(width,U32.to_nat(position)),output_fits(product))),
    Equal.cong(Nat,Nat,value => Nat.sub(width,value),U32.to_nat(position),first,exact))

# ---- The cells a job stores ----

# A job of a product: its unit of eight rows and its group of eight columns.
type Placement is Data:
  Placement{product: Product,unit: Nat,group: Nat,unit_inside: {Nat.is_lt(unit,units(product)) == True{} : Bool},
    group_inside: {Nat.is_lt(group,product_groups(product)) == True{} : Bool}}

def placement_product(+placement: Placement) -> Product:
  Placement{product,unit,group,unit_inside,group_inside} = placement
  product

def placement_unit(+placement: Placement) -> Nat:
  Placement{product,unit,group,unit_inside,group_inside} = placement
  unit

def placement_group(+placement: Placement) -> Nat:
  Placement{product,unit,group,unit_inside,group_inside} = placement
  group

def tile_matrix(+product: Product) -> Gemm.TileMatrix:
  Gemm.TileMatrix{product_rows(product),product_columns(product),product_inner(product),product_start(product),product_output(product)}

def tile_job(+placement: Placement) -> Gemm.Job<Gemm.Direct()>:
  Placement{product,unit,group,unit_inside,group_inside} = placement
  Gemm.tile_major(tile_matrix(product),job_index(product,unit,group))

def cell_column(+placement: Placement,+lane: Nat) -> Nat:
  Nat.add(Nat.mul(placement_group(placement),8n),lane)

def column_inside(+placement: Placement,+lane: Nat,+lane_inside: {Nat.is_lt(lane,GemmJobs.job_rows(~Gemm.Direct(),tile_job(placement))) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(start_column(placement_product(placement)),cell_column(placement,lane)),output_width(placement_product(placement))) == True{} : Bool}:
  Placement{+product,+unit,+group,+unit_inside,+group_inside} = placement
  +left = (product_output(product) - position_word(product,job_index(product,unit,group)) : U32)
  +width = output_width(product)
  +first = Nat.add(start_column(product),Nat.mul(group,8n))
  +below = {
    %lanes_left(product,unit,group,unit_inside,group_inside) : {Nat.is_lt(lane,_) == True{} : Bool}
    ViewWrites.clipped_below(lane,8,left,lane_inside) : {Nat.is_lt(lane,Nat.sub(width,first)) == True{} : Bool}}
  %Addition.associative(start_column(product),Nat.mul(group,8n),lane) : {Nat.is_lt(_,width) == True{} : Bool}
  %Order.restore_shift(first,width,Bounds.strict_le(first,width,group_start(product,group,group_inside))) : {Nat.is_lt(Nat.add(first,lane),_) == True{} : Bool}
  Equal.trans(Bool,Nat.is_lt(Nat.add(first,lane),Nat.add(first,Nat.sub(width,first))),Nat.is_lt(lane,Nat.sub(width,first)),True{},
    Order.lt_offset(first,lane,Nat.sub(width,first)),below)

def column_written(+placement: Placement,+lane: Nat,+lane_inside: {Nat.is_lt(lane,GemmJobs.job_rows(~Gemm.Direct(),tile_job(placement))) == True{} : Bool}) ->
  {Nat.is_lt(cell_column(placement,lane),U32.to_nat(product_columns(placement_product(placement)))) == True{} : Bool}:
  Placement{+product,+unit,+group,+unit_inside,+group_inside} = placement
  +lane_small = ViewWrites.clipped_small(lane,8,(product_output(product) - position_word(product,job_index(product,unit,group)) : U32),lane_inside)
  %Equal.sym(Nat,U32.to_nat(product_columns(product)),Nat.mul(product_groups(product),8n),columns_grouped(product)) :
    {Nat.is_lt(Nat.add(Nat.mul(group,8n),lane),_) == True{} : Bool}
  RowMajor.radix(group,product_groups(product),lane,8n,group_inside,lane_small)

# ---- Every job ----

def jobs(+product: Product) -> Nat:
  Nat.mul(product_groups(product),units(product))

# A number below the jobs is the number of one unit and one group.
type Split<-once: Bool,-first: Nat,-count: Nat,-groups: Nat> is Data:
  Split{unit: Nat,group: Nat,unit_inside: {Nat.is_lt(unit,count) == True{} : Bool},group_inside: {Nat.is_lt(group,groups) == True{} : Bool},
    exact: {first == index_of(once,unit,group,count,groups) : Nat}}

def split_of(once: Bool,+first: Nat,+count: Nat,+groups: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+small: {Nat.is_le(first,U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(first,Nat.mul(groups,count)) == True{} : Bool}) -> Split<once,first,count,groups>:
  match once:
    case True{}:
      +roundtrip = Bounds.roundtrip(first,cap,cap_ok,small)
      +positive = RowMajor.left_factor_positive(groups,count,Order.le_lt_trans(0n,first,Nat.mul(groups,count),Order.zero_le(first),inside))
      +unit = Nat.div(first,groups)
      +group = Nat.mod(first,groups)
      +identity = {
        %roundtrip : {_ == Nat.add(Nat.mul(Nat.div(_,groups),groups),Nat.mod(_,groups)) : Nat}
        Radix.division_identity(U32.from_nat(first),groups,positive) : {first == Nat.add(Nat.mul(unit,groups),group) : Nat}}
      +group_inside = {
        %roundtrip : {Nat.is_lt(Nat.mod(_,groups),groups) == True{} : Bool}
        Radix.remainder_inside(U32.from_nat(first),groups,positive) : {Nat.is_lt(group,groups) == True{} : Bool}}
      +unit_inside = RowMajor.cancel_multiply_lt(unit,count,groups,
        Order.le_lt_trans(Nat.mul(unit,groups),first,Nat.mul(count,groups),
          %Equal.sym(Nat,first,Nat.add(Nat.mul(unit,groups),group),identity) : {Nat.is_le(Nat.mul(unit,groups),_) == True{} : Bool} Order.le_add(Nat.mul(unit,groups),group),
          %RowMajor.multiply_commutative(groups,count) : {Nat.is_lt(first,_) == True{} : Bool} inside))
      Split{unit,group,unit_inside,group_inside,identity}
    case False{}:
      +roundtrip = Bounds.roundtrip(first,cap,cap_ok,small)
      +positive = RowMajor.left_factor_positive(count,groups,
        %RowMajor.multiply_commutative(groups,count) : {Nat.is_lt(0n,_) == True{} : Bool}
        Order.le_lt_trans(0n,first,Nat.mul(groups,count),Order.zero_le(first),inside))
      +group = Nat.div(first,count)
      +unit = Nat.mod(first,count)
      +identity = {
        %roundtrip : {_ == Nat.add(Nat.mul(Nat.div(_,count),count),Nat.mod(_,count)) : Nat}
        Radix.division_identity(U32.from_nat(first),count,positive) : {first == Nat.add(Nat.mul(group,count),unit) : Nat}}
      +unit_inside = {
        %roundtrip : {Nat.is_lt(Nat.mod(_,count),count) == True{} : Bool}
        Radix.remainder_inside(U32.from_nat(first),count,positive) : {Nat.is_lt(unit,count) == True{} : Bool}}
      +group_inside = RowMajor.cancel_multiply_lt(group,groups,count,
        Order.le_lt_trans(Nat.mul(group,count),first,Nat.mul(groups,count),
          %Equal.sym(Nat,first,Nat.add(Nat.mul(group,count),unit),identity) : {Nat.is_le(Nat.mul(group,count),_) == True{} : Bool} Order.le_add(Nat.mul(group,count),unit),inside))
      Split{unit,group,unit_inside,group_inside,identity}

# Job number first of a product is its unit and group.
type Numbered<-product: Product,-first: Nat> is Data:
  Numbered{unit: Nat,group: Nat,unit_inside: {Nat.is_lt(unit,units(product)) == True{} : Bool},group_inside: {Nat.is_lt(group,product_groups(product)) == True{} : Bool},
    exact: {first == index_number(product,unit,group) : Nat}}

def numbered_of(+product: Product,+first: Nat,+split: Split<once(product),first,units(product),product_groups(product)>) -> Numbered<product,first>:
  Split{+unit,+group,+unit_inside,+group_inside,+exact} = split
  Numbered{unit,group,unit_inside,group_inside,exact}

def numbered(+product: Product,+first: Nat,+inside: {Nat.is_lt(first,jobs(product)) == True{} : Bool}) -> Numbered<product,first>:
  numbered_of(product,first,split_of(once(product),first,units(product),product_groups(product),product_limit(product),limit_fits(product),
    Order.transitive(first,jobs(product),limit_count(product),Bounds.strict_le(first,jobs(product),inside),jobs_within(product)),inside))

# ---- The queried cell is stored ----

# A number below a limit is its quotient and remainder by a positive divisor.
def split_number(+value: Nat,+divisor: Nat,+limit: U32,+limit_ok: {U32.is_le(limit,2147483648) == True{} : Bool},
  +small: {Nat.is_le(value,U32.to_nat(limit)) == True{} : Bool},+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {value == Nat.add(Nat.mul(Nat.div(value,divisor),divisor),Nat.mod(value,divisor)) : Nat}:
  %Bounds.roundtrip(value,limit,limit_ok,small) : {_ == Nat.add(Nat.mul(Nat.div(_,divisor),divisor),Nat.mod(_,divisor)) : Nat}
  Radix.division_identity(U32.from_nat(value),divisor,positive)

def split_remainder(+value: Nat,+divisor: Nat,+limit: U32,+limit_ok: {U32.is_le(limit,2147483648) == True{} : Bool},
  +small: {Nat.is_le(value,U32.to_nat(limit)) == True{} : Bool},+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {Nat.is_lt(Nat.mod(value,divisor),divisor) == True{} : Bool}:
  %Bounds.roundtrip(value,limit,limit_ok,small) : {Nat.is_lt(Nat.mod(_,divisor),divisor) == True{} : Bool}
  Radix.remainder_inside(U32.from_nat(value),divisor,positive)

# A unit whose first row is below the rows is a unit of the product.
def unit_counted(+product: Product,+unit: Nat,+unit_rows: {Nat.is_lt(Nat.mul(unit,8n),row_count(product)) == True{} : Bool}) ->
  {Nat.is_lt(unit,units(product)) == True{} : Bool}:
  +rows = row_count(product)
  +padded = Nat.add(rows,7n)
  +count = units(product)
  +remainder = Nat.mod(padded,8n)
  +identity = {
    %Equal.sym(Nat,count,Nat.div(padded,8n),units_exact(product)) : {padded == Nat.add(Nat.mul(_,8n),remainder) : Nat}
    padded_identity(product) : {padded == Nat.add(Nat.mul(count,8n),remainder) : Nat}}
  +remainder_small = split_remainder(padded,8n,product_limit(product),limit_fits(product),rows_fit(product),{==})
  RowMajor.cancel_multiply_lt(1n+unit,1n+count,8n,
    Order.le_lt_trans(Nat.mul(1n+unit,8n),padded,Nat.mul(1n+count,8n),
      %Order.commutative(7n,rows) : {Nat.is_le(Nat.mul(1n+unit,8n),_) == True{} : Bool}
      Order.strict_follow(Nat.mul(unit,8n),rows,unit_rows),
      %Equal.sym(Nat,padded,Nat.add(Nat.mul(count,8n),remainder),identity) : {Nat.is_lt(_,Nat.mul(1n+count,8n)) == True{} : Bool}
      %Order.commutative(remainder,Nat.mul(count,8n)) : {Nat.is_lt(_,Nat.mul(1n+count,8n)) == True{} : Bool}
      Order.add_lt(remainder,8n,Nat.mul(count,8n),remainder_small)))

def channel_small(+product: Product,+channel: Nat,+channel_inside: {Nat.is_lt(channel,row_count(product)) == True{} : Bool}) ->
  {Nat.is_le(channel,limit_count(product)) == True{} : Bool}:
  Order.transitive(channel,row_count(product),limit_count(product),Bounds.strict_le(channel,row_count(product),channel_inside),
    Order.transitive(row_count(product),Nat.add(row_count(product),7n),limit_count(product),Order.le_add(row_count(product),7n),rows_fit(product)))

def column_small(+product: Product,+column: Nat,+column_inside: {Nat.is_lt(Nat.add(start_column(product),column),output_width(product)) == True{} : Bool}) ->
  {Nat.is_le(column,limit_count(product)) == True{} : Bool}:
  Order.transitive(column,Nat.add(start_column(product),column),limit_count(product),
    %Order.commutative(column,start_column(product)) : {Nat.is_le(column,_) == True{} : Bool} Order.le_add(column,start_column(product)),
    Order.transitive(Nat.add(start_column(product),column),output_width(product),limit_count(product),
      Bounds.strict_le(Nat.add(start_column(product),column),output_width(product),column_inside),output_fits(product)))

def channel_unit(+product: Product,+channel: Nat,+channel_inside: {Nat.is_lt(channel,row_count(product)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.div(channel,8n),units(product)) == True{} : Bool}:
  +unit = Nat.div(channel,8n)
  +identity = split_number(channel,8n,product_limit(product),limit_fits(product),channel_small(product,channel,channel_inside),{==})
  unit_counted(product,unit,
    Order.le_lt_trans(Nat.mul(unit,8n),channel,row_count(product),
      %Equal.sym(Nat,channel,Nat.add(Nat.mul(unit,8n),Nat.mod(channel,8n)),identity) : {Nat.is_le(Nat.mul(unit,8n),_) == True{} : Bool}
      Order.le_add(Nat.mul(unit,8n),Nat.mod(channel,8n)),channel_inside))

def column_group(+product: Product,+column: Nat,+column_inside: {Nat.is_lt(Nat.add(start_column(product),column),output_width(product)) == True{} : Bool},
  +written: {Nat.is_lt(column,U32.to_nat(product_columns(product))) == True{} : Bool}) -> {Nat.is_lt(Nat.div(column,8n),product_groups(product)) == True{} : Bool}:
  +group = Nat.div(column,8n)
  +identity = split_number(column,8n,product_limit(product),limit_fits(product),column_small(product,column,column_inside),{==})
  RowMajor.cancel_multiply_lt(group,product_groups(product),8n,
    Order.le_lt_trans(Nat.mul(group,8n),column,Nat.mul(product_groups(product),8n),
      %Equal.sym(Nat,column,Nat.add(Nat.mul(group,8n),Nat.mod(column,8n)),identity) : {Nat.is_le(Nat.mul(group,8n),_) == True{} : Bool}
      Order.le_add(Nat.mul(group,8n),Nat.mod(column,8n)),
      %columns_grouped(product) : {Nat.is_lt(column,_) == True{} : Bool} written))

def below_difference(+first: Nat,+offset: Nat,+total: Nat,+ordered: {Nat.is_le(first,total) == True{} : Bool},
  +below: {Nat.is_lt(Nat.add(first,offset),total) == True{} : Bool}) -> {Nat.is_lt(offset,Nat.sub(total,first)) == True{} : Bool}:
  Equal.trans(Bool,Nat.is_lt(offset,Nat.sub(total,first)),Nat.is_lt(Nat.add(first,offset),Nat.add(first,Nat.sub(total,first))),True{},
    Equal.sym(Bool,Nat.is_lt(Nat.add(first,offset),Nat.add(first,Nat.sub(total,first))),Nat.is_lt(offset,Nat.sub(total,first)),Order.lt_offset(first,offset,Nat.sub(total,first))),
    %Equal.sym(Nat,Nat.add(first,Nat.sub(total,first)),total,Order.restore_shift(first,total,ordered)) : {Nat.is_lt(Nat.add(first,offset),_) == True{} : Bool}
    below)

# ---- Job counts ----

def jobs_exact(+product: Product) -> {U32.to_nat(Gemm.tile_jobs(tile_matrix(product))) == jobs(product) : Nat}:
  +groups = (product_columns(product) / 8 : U32)
  Equal.trans(Nat,U32.to_nat((groups * Gemm.blocks(product_rows(product)) : U32)),Nat.mul(U32.to_nat(groups),units(product)),jobs(product),
    Multiplication.bounded_multiply(groups,Gemm.blocks(product_rows(product)),product_limit(product),limit_fits(product),
      %Equal.sym(Nat,U32.to_nat(groups),product_groups(product),groups_exact(product)) : {Nat.is_le(Nat.mul(_,units(product)),limit_count(product)) == True{} : Bool}
      jobs_within(product)),
    Equal.cong(Nat,Nat,value => Nat.mul(value,units(product)),U32.to_nat(groups),product_groups(product),groups_exact(product)))

# ---- The tiles of a product ----

# A tile of the window: its channel group and column.
type TileSpot is Data:
  TileSpot{group: Nat,column: Nat}

def TileWritten(+product: Product,+spot: TileSpot) -> Data:
  TileSpot{+group,+column} = spot
  Gemm.Both<{Nat.is_lt(group,units(product)) == True{} : Bool},Gemm.Both<{Nat.is_lt(Nat.add(start_column(product),column),output_width(product)) == True{} : Bool},
    {Nat.is_lt(column,U32.to_nat(product_columns(product))) == True{} : Bool}>>

# Tile (group, column) of the window is output tile tile_number(...): the
# groups of a position side by side.
def tile_number(+product: Product,+group: Nat,+column: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.add(start_column(product),column),units(product)),group)

def tile_word(+product: Product,+spot: TileSpot) -> U32:
  TileSpot{+group,+column} = spot
  U32.from_nat(tile_number(product,group,column))

def tile_area(+product: Product) -> Nat:
  Nat.mul(output_width(product),units(product))

def units_within(+product: Product) -> {Nat.is_le(units(product),row_count(product)) == True{} : Bool}:
  eighths_within(row_count(product),units(product),units_room(product))

def tile_area_fits(+product: Product) -> {Nat.is_le(tile_area(product),limit_count(product)) == True{} : Bool}:
  %RowMajor.multiply_commutative(units(product),output_width(product)) : {Nat.is_le(_,limit_count(product)) == True{} : Bool}
  Order.transitive(Nat.mul(units(product),output_width(product)),area(product),limit_count(product),
    RowMajor.mul_le(units(product),row_count(product),output_width(product),units_within(product)),area_fits(product))

def tile_number_bound(+product: Product,+group: Nat,+column: Nat,+group_inside: {Nat.is_lt(group,units(product)) == True{} : Bool},
  +column_inside: {Nat.is_lt(Nat.add(start_column(product),column),output_width(product)) == True{} : Bool}) ->
  {Nat.is_lt(tile_number(product,group,column),tile_area(product)) == True{} : Bool}:
  RowMajor.radix(Nat.add(start_column(product),column),output_width(product),group,units(product),column_inside,group_inside)

def tile_number_exact(+product: Product,+group: Nat,+column: Nat,+group_inside: {Nat.is_lt(group,units(product)) == True{} : Bool},
  +column_inside: {Nat.is_lt(Nat.add(start_column(product),column),output_width(product)) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(tile_number(product,group,column))) == tile_number(product,group,column) : Nat}:
  +number = tile_number(product,group,column)
  Bounds.roundtrip(number,product_limit(product),limit_fits(product),
    Bounds.strict_le(number,limit_count(product),Bounds.nat_trans(number,tile_area(product),limit_count(product),
      tile_number_bound(product,group,column,group_inside,column_inside),tile_area_fits(product))))

# Every written tile lies below the tile area.
def tile_below(+product: Product,+spot: TileSpot,+valid: TileWritten(product,spot)) -> {Nat.is_lt(U32.to_nat(tile_word(product,spot)),tile_area(product)) == True{} : Bool}:
  TileSpot{+group,+column} = spot
  Gemm.Both{+group_inside,Gemm.Both{+column_inside,+inside}} = valid
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(tile_number(product,group,column))),tile_number(product,group,column),tile_number_exact(product,group,column,group_inside,column_inside)) :
    {Nat.is_lt(_,tile_area(product)) == True{} : Bool}
  tile_number_bound(product,group,column,group_inside,column_inside)

# Two tiles at one address are the same tile.
def tile_spot_same(+product: Product,+spot: TileSpot,+other: TileSpot,+valid: TileWritten(product,spot),+other_valid: TileWritten(product,other),
  +equal: {tile_word(product,spot) == tile_word(product,other) : U32}) -> {spot == other : TileSpot}:
  TileSpot{+group,+column} = spot
  TileSpot{+other_group,+other_column} = other
  Gemm.Both{+group_inside,Gemm.Both{+column_inside,+inside}} = valid
  Gemm.Both{+other_group_inside,Gemm.Both{+other_inside,+other_written}} = other_valid
  +count = units(product)
  +first = start_column(product)
  +number = tile_number(product,group,column)
  +other_number = tile_number(product,other_group,other_column)
  +numbers = Equal.trans(Nat,number,U32.to_nat(U32.from_nat(number)),other_number,
    Equal.sym(Nat,U32.to_nat(U32.from_nat(number)),number,tile_number_exact(product,group,column,group_inside,column_inside)),
    Equal.trans(Nat,U32.to_nat(U32.from_nat(number)),U32.to_nat(U32.from_nat(other_number)),other_number,
      Equal.cong(U32,Nat,U32.to_nat,U32.from_nat(number),U32.from_nat(other_number),equal),
      tile_number_exact(product,other_group,other_column,other_group_inside,other_inside)))
  +positive = Order.positive_bound(group,count,group_inside)
  +groups_same = Equal.trans(Nat,group,Nat.mod(number,count),other_group,
    Equal.sym(Nat,Nat.mod(number,count),group,Radix.remainder(Nat.add(first,column),group,count,positive,group_inside)),
    Equal.trans(Nat,Nat.mod(number,count),Nat.mod(other_number,count),other_group,
      Equal.cong(Nat,Nat,total => Nat.mod(total,count),number,other_number,numbers),
      Radix.remainder(Nat.add(first,other_column),other_group,count,positive,other_group_inside)))
  +positions_same = Equal.trans(Nat,Nat.add(first,column),Nat.div(number,count),Nat.add(first,other_column),
    Equal.sym(Nat,Nat.div(number,count),Nat.add(first,column),Radix.quotient(Nat.add(first,column),group,count,positive,group_inside)),
    Equal.trans(Nat,Nat.div(number,count),Nat.div(other_number,count),Nat.add(first,other_column),
      Equal.cong(Nat,Nat,total => Nat.div(total,count),number,other_number,numbers),
      Radix.quotient(Nat.add(first,other_column),other_group,count,positive,other_group_inside)))
  +columns_same = Order.cancel_sum(first,column,other_column,positions_same)
  %groups_same : {TileSpot{group,column} == TileSpot{_,other_column} : TileSpot}
  %columns_same : {TileSpot{group,column} == TileSpot{group,_} : TileSpot}
  {==}

# ---- The tiles a job stores ----

# The word of a tile as the words of its group, column group and row.
def tile_word_split(+product: Product,+group: Nat,+unit: Nat,+row: Nat) ->
  {tile_word(product,TileSpot{group,Nat.add(Nat.mul(unit,8n),row)}) ==
    ((product_start(product) + (U32.from_nat(unit) * 8 + U32.from_nat(row) : U32) : U32) * Gemm.blocks(product_rows(product)) + U32.from_nat(group) : U32) : U32}:
  +column = Nat.add(Nat.mul(unit,8n),row)
  +count = units(product)
  +first = start_column(product)
  +position = Nat.add(first,column)
  +groups = Gemm.blocks(product_rows(product))
  +wanted_word = ((product_start(product) + (U32.from_nat(unit) * 8 + U32.from_nat(row) : U32) : U32) * groups + U32.from_nat(group) : U32)
  %WordArithmetic.native_add(Nat.mul(position,count),group) : {_ == wanted_word : U32}
  %Multiplication.native_encoded_multiply(position,count) : {(_ + U32.from_nat(group) : U32) == wanted_word : U32}
  %Equal.sym(U32,U32.from_nat(count),groups,Arithmetic.native_roundtrip(groups)) :
    {(U32.from_nat(position) * _ + U32.from_nat(group) : U32) == wanted_word : U32}
  %WordArithmetic.native_add(first,column) : {(_ * groups + U32.from_nat(group) : U32) == wanted_word : U32}
  %Equal.sym(U32,U32.from_nat(first),product_start(product),Arithmetic.native_roundtrip(product_start(product))) :
    {((_ + U32.from_nat(column) : U32) * groups + U32.from_nat(group) : U32) == wanted_word : U32}
  %WordArithmetic.native_add(Nat.mul(unit,8n),row) :
    {((product_start(product) + _ : U32) * groups + U32.from_nat(group) : U32) == wanted_word : U32}
  %Multiplication.native_encoded_multiply(unit,8n) :
    {((product_start(product) + (_ + U32.from_nat(row) : U32) : U32) * groups + U32.from_nat(group) : U32) == wanted_word : U32}
  {==}

def tile_identity(+start: U32,+offset: U32,+row: U32,+groups: U32,+unit: U32) ->
  {((start + offset : U32) * groups + unit + row * groups + U32.from_nat(0n) * 1 : U32) == ((start + (offset + row : U32) : U32) * groups + unit : U32) : U32}:
  +base = (start + offset : U32)
  +wanted = ((start + (offset + row : U32) : U32) * groups + unit : U32)
  Equal.trans(U32,(base * groups + unit + row * groups + U32.from_nat(0n) * 1 : U32),(base * groups + unit + row * groups : U32),wanted,
    Arithmetic.add_zero((base * groups + unit + row * groups : U32)),
    Equal.trans(U32,(base * groups + unit + row * groups : U32),(base * groups + row * groups + unit : U32),wanted,
      Arithmetic.add_swap((base * groups : U32),unit,(row * groups : U32)),
      Equal.cong(U32,U32,scaled => (scaled + unit : U32),(base * groups + row * groups : U32),((start + (offset + row : U32) : U32) * groups : U32),
        Equal.trans(U32,(base * groups + row * groups : U32),((base + row : U32) * groups : U32),((start + (offset + row : U32) : U32) * groups : U32),
          Equal.sym(U32,((base + row : U32) * groups : U32),(base * groups + row * groups : U32),Arithmetic.multiply_additive(base,row,groups)),
          Equal.cong(U32,U32,sum => (sum * groups : U32),(base + row : U32),(start + (offset + row : U32) : U32),Arithmetic.add_associative(start,offset,row))))))

# Row row of the job of a placement stores the tile of its group and column.
def tile_address(+placement: Placement,+row: Nat) ->
  {GemmJobs.cell_address(~Gemm.Direct(),tile_job(placement),row,0n) ==
    tile_word(placement_product(placement),TileSpot{placement_unit(placement),cell_column(placement,row)}) : U32}:
  Placement{+product,+unit,+group,+unit_inside,+group_inside} = placement
  +index = job_index(product,unit,group)
  +unit_part = unit_word(product,index)
  +group_part = group_word(product,index)
  +groups = Gemm.blocks(product_rows(product))
  +start = product_start(product)
  +row_number = U32.from_nat(row)
  +unit_number = U32.from_nat(unit)
  +group_number = U32.from_nat(group)
  -target = ((start + (group_number * 8 + row_number : U32) : U32) * groups + unit_number : U32)
  %Equal.sym(U32,tile_word(product,TileSpot{unit,Nat.add(Nat.mul(group,8n),row)}),target,tile_word_split(product,unit,group,row)) :
    {GemmJobs.cell_address(~Gemm.Direct(),Gemm.tile_major(tile_matrix(product),index),row,0n) == _ : U32}
  %Equal.sym(U32,unit_part,unit_number,unit_value(product,unit,group,unit_inside,group_inside)) :
    {((start + group_part * 8 : U32) * groups + _ + row_number * groups + U32.from_nat(0n) * 1 : U32) == target : U32}
  %Equal.sym(U32,group_part,group_number,group_value(product,unit,group,unit_inside,group_inside)) :
    {((start + _ * 8 : U32) * groups + unit_number + row_number * groups + U32.from_nat(0n) * 1 : U32) == target : U32}
  tile_identity(start,(group_number * 8 : U32),row_number,groups,unit_number)

# ---- Every tile job ----

def tile_rows_small(+product: Product,+index: Nat,+inside: {Nat.is_lt(index,jobs(product)) == True{} : Bool}) ->
  {Nat.is_le(GemmJobs.job_rows(~Gemm.Direct(),Gemm.tile_major(tile_matrix(product),U32.from_nat(index))),8n) == True{} : Bool}:
  U32Extrema.minimum_below(8,(product_output(product) - position_word(product,U32.from_nat(index)) : U32))

# A written tile is row column % 8 of the job of its group and column group.
def tile_found(+product: Product,+spot: TileSpot,+valid: TileWritten(product,spot)) ->
  GemmJobs.Locate(~Gemm.TileMatrix,~Gemm.Direct(),~Gemm.tile_major,~GemmLaws.Tiles{},~Product,~tile_matrix,~jobs,~TileSpot,~tile_word,product,spot):
  TileSpot{+group,+column} = spot
  Gemm.Both{+group_inside,Gemm.Both{+column_inside,+inside}} = valid
  +block_inside = column_group(product,column,column_inside,inside)
  +block = Nat.div(column,8n)
  +row = Nat.mod(column,8n)
  +column_identity = split_number(column,8n,product_limit(product),limit_fits(product),column_small(product,column,column_inside),{==})
  +row_small = split_remainder(column,8n,product_limit(product),limit_fits(product),column_small(product,column,column_inside),{==})
  +placement = {Placement{product,group,block,group_inside,block_inside} : Placement}
  +remaining = (product_output(product) - position_word(product,job_index(product,group,block)) : U32)
  +first_column = Nat.add(start_column(product),Nat.mul(block,8n))
  +row_below = {%column_identity : {Nat.is_lt(Nat.add(start_column(product),_),output_width(product)) == True{} : Bool} column_inside :
    {Nat.is_lt(Nat.add(start_column(product),Nat.add(Nat.mul(block,8n),row)),output_width(product)) == True{} : Bool}}
  +row_left = below_difference(first_column,row,output_width(product),Bounds.strict_le(first_column,output_width(product),group_start(product,block,block_inside)),
    %Equal.sym(Nat,Nat.add(first_column,row),Nat.add(start_column(product),Nat.add(Nat.mul(block,8n),row)),Addition.associative(start_column(product),Nat.mul(block,8n),row)) :
      {Nat.is_lt(_,output_width(product)) == True{} : Bool}
    row_below)
  +row_inside = ViewWrites.clipped(row,8,remaining,row_small,
    %Equal.sym(Nat,U32.to_nat(remaining),Nat.sub(output_width(product),first_column),lanes_left(product,group,block,group_inside,block_inside)) : {Nat.is_lt(row,_) == True{} : Bool}
    row_left)
  +address = {
    %Equal.sym(Nat,column,Nat.add(Nat.mul(block,8n),row),column_identity) :
      {GemmJobs.cell_address(~Gemm.Direct(),tile_job(placement),row,0n) == tile_word(product,TileSpot{group,_}) : U32}
    tile_address(placement,row) : {GemmJobs.cell_address(~Gemm.Direct(),tile_job(placement),row,0n) == tile_word(product,TileSpot{group,column}) : U32}}
  GemmJobs.Found{index_number(product,group,block),row,0n,index_below(once(product),group,block,units(product),product_groups(product),group_inside,block_inside),
    row_inside,{==},address}

# A channel splits into its group of eight rows and its lane in the group.
type ChannelParts<-product: Product,-channel: Nat> is Data:
  ChannelParts{split: {channel == Nat.add(Nat.mul(Nat.div(channel,8n),8n),Nat.mod(channel,8n)) : Nat},
    small: {Nat.is_lt(Nat.mod(channel,8n),8n) == True{} : Bool},
    first_exact: {U32.to_nat((U32.from_nat(Nat.div(channel,8n)) * 8 : U32)) == Nat.mul(Nat.div(channel,8n),8n) : Nat},
    lane_exact: {U32.to_nat(U32.from_nat(Nat.mod(channel,8n))) == Nat.mod(channel,8n) : Nat},
    channel_exact: {U32.to_nat(U32.from_nat(channel)) == channel : Nat}}

def channel_parts(+product: Product,+channel: Nat,+inside: {Nat.is_lt(channel,row_count(product)) == True{} : Bool}) -> ChannelParts<product,channel>:
  +group = Nat.div(channel,8n)
  +lane = Nat.mod(channel,8n)
  +limit = product_limit(product)
  +split = DivisionOrder.quotient_exact(channel,8n,{==})
  +channel_room = channel_small(product,channel,inside)
  +first_room = Order.transitive(Nat.mul(group,8n),channel,limit_count(product),DivisionOrder.product_below(channel,8n,{==}),channel_room)
  +group_exact = Bounds.roundtrip(group,limit,limit_fits(product),Order.transitive(group,Nat.mul(group,8n),limit_count(product),RowMajor.product_covers(group,8n,{==}),first_room))
  +lane_room = {
    %Equal.sym(Nat,channel,Nat.add(Nat.mul(group,8n),lane),split) : {Nat.is_le(lane,_) == True{} : Bool}
    %Order.commutative(lane,Nat.mul(group,8n)) : {Nat.is_le(lane,_) == True{} : Bool}
    Order.le_add(lane,Nat.mul(group,8n)) : {Nat.is_le(lane,channel) == True{} : Bool}}
  ChannelParts{split,DivisionOrder.remainder_small(channel,8n,{==}),
    Equal.trans(Nat,U32.to_nat((U32.from_nat(group) * 8 : U32)),Nat.mul(U32.to_nat(U32.from_nat(group)),8n),Nat.mul(group,8n),
      Multiplication.bounded_multiply(U32.from_nat(group),8,limit,limit_fits(product),
        %Equal.sym(Nat,U32.to_nat(U32.from_nat(group)),group,group_exact) : {Nat.is_le(Nat.mul(_,8n),limit_count(product)) == True{} : Bool} first_room),
      Equal.cong(Nat,Nat,value => Nat.mul(value,8n),U32.to_nat(U32.from_nat(group)),group,group_exact)),
    Bounds.roundtrip(lane,limit,limit_fits(product),Order.transitive(lane,channel,limit_count(product),lane_room,channel_room)),
    Bounds.roundtrip(channel,limit,limit_fits(product),channel_room)}

# The lane of a channel lies below the lanes of its group.
def channel_lane_of(+product: Product,+channel: Nat,+inside: {Nat.is_lt(channel,row_count(product)) == True{} : Bool},+parts: ChannelParts<product,channel>) ->
  {U32.is_lt(U32.from_nat(Nat.mod(channel,8n)),U32.min(8,(product_rows(product) - (U32.from_nat(Nat.div(channel,8n)) * 8 : U32) : U32))) == True{} : Bool}:
  ChannelParts{+split,+small,+first_exact,+lane_exact,+channel_exact} = parts
  +group = Nat.div(channel,8n)
  +lane = Nat.mod(channel,8n)
  +rows = row_count(product)
  +first = (U32.from_nat(group) * 8 : U32)
  +rest = (product_rows(product) - first : U32)
  +first_rows = {
    %Equal.sym(Nat,U32.to_nat(first),Nat.mul(group,8n),first_exact) : {Nat.is_le(_,rows) == True{} : Bool}
    Bounds.strict_le(Nat.mul(group,8n),rows,Order.le_lt_trans(Nat.mul(group,8n),channel,rows,DivisionOrder.product_below(channel,8n,{==}),inside)) :
    {Nat.is_le(U32.to_nat(first),rows) == True{} : Bool}}
  +rest_exact = Arithmetic.subtract(product_rows(product),first,product_limit(product),limit_fits(product),first_rows,
    Order.transitive(Nat.sub(rows,U32.to_nat(first)),rows,limit_count(product),Order.sub_le(rows,U32.to_nat(first)),
      Order.transitive(rows,Nat.add(rows,7n),limit_count(product),Order.le_add(rows,7n),rows_fit(product))))
  %Equal.sym(Bool,U32.is_lt(U32.from_nat(lane),U32.min(8,rest)),Nat.is_lt(U32.to_nat(U32.from_nat(lane)),U32.to_nat(U32.min(8,rest))),Comparison.lt(U32.from_nat(lane),U32.min(8,rest))) :
    {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(lane)),lane,lane_exact) : {Nat.is_lt(_,U32.to_nat(U32.min(8,rest))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.min(8,rest)),Nat.min(8n,U32.to_nat(rest)),U32Extrema.minimum(8,rest)) : {Nat.is_lt(lane,_) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(rest),Nat.sub(rows,U32.to_nat(first)),rest_exact) : {Nat.is_lt(lane,Nat.min(8n,_)) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(first),Nat.mul(group,8n),first_exact) : {Nat.is_lt(lane,Nat.min(8n,Nat.sub(rows,_))) == True{} : Bool}
  Extrema.minimum_strict_lower(8n,Nat.sub(rows,Nat.mul(group,8n)),lane,small,
    Equal.trans(Bool,Nat.is_lt(lane,Nat.sub(rows,Nat.mul(group,8n))),Nat.is_lt(Nat.add(Nat.mul(group,8n),lane),rows),True{},
      Order.below_difference(Nat.mul(group,8n),lane,rows,
        %first_exact : {Nat.is_le(_,rows) == True{} : Bool} first_rows),
      %split : {Nat.is_lt(_,rows) == True{} : Bool} inside))

def channel_lane(+product: Product,+channel: Nat,+inside: {Nat.is_lt(channel,row_count(product)) == True{} : Bool}) ->
  {U32.is_lt(U32.from_nat(Nat.mod(channel,8n)),U32.min(8,(product_rows(product) - (U32.from_nat(Nat.div(channel,8n)) * 8 : U32) : U32))) == True{} : Bool}:
  channel_lane_of(product,channel,inside,channel_parts(product,channel,inside))

# The ordered dot of a channel and column on the tile writer over row tiles:
# the column's row by the channel's lane.
def row_dot(+product: Product,+rows: Tree.Tree<Tile.Tile8>,+channel: Nat,+column: Nat) -> F32:
  +inner = product_inner(product)
  GemmSpec.dot(GemmLaws.RowFirst{},GemmSpec.taps(~Gemm.Direct(),~Gemm.directs,U32.to_nat(inner),Gemm.Both{(U32.from_nat(Nat.div(channel,8n)) * inner : U32),(U32.from_nat(Nat.div(column,8n)) * inner : U32)}),
    Nat.mod(column,8n),Nat.mod(channel,8n),GemmSpec.Source{rows,product_lane_tiles(product)},0.0)

# A lane below the lanes of a unit of eight rows is a channel of the product.
def lane_channel(+product: Product,+unit: Nat,+unit_inside: {Nat.is_lt(unit,units(product)) == True{} : Bool},+lane: Nat,+small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +inside: {U32.is_lt(U32.from_nat(lane),U32.min(8,(product_rows(product) - (U32.from_nat(unit) * 8 : U32) : U32))) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(Nat.mul(unit,8n),lane),row_count(product)) == True{} : Bool}:
  +rows = row_count(product)
  +limit = product_limit(product)
  +first = (U32.from_nat(unit) * 8 : U32)
  +rest = (product_rows(product) - first : U32)
  +below = unit_rows(product,unit,unit_inside)
  +rows_room = Order.transitive(rows,Nat.add(rows,7n),limit_count(product),Order.le_add(rows,7n),rows_fit(product))
  +first_room = Bounds.strict_le(Nat.mul(unit,8n),limit_count(product),Bounds.nat_trans(Nat.mul(unit,8n),rows,limit_count(product),below,rows_room))
  +unit_exact = Bounds.roundtrip(unit,limit,limit_fits(product),Order.transitive(unit,Nat.mul(unit,8n),limit_count(product),RowMajor.product_covers(unit,8n,{==}),first_room))
  +first_exact = Equal.trans(Nat,U32.to_nat(first),Nat.mul(U32.to_nat(U32.from_nat(unit)),8n),Nat.mul(unit,8n),
    Multiplication.bounded_multiply(U32.from_nat(unit),8,limit,limit_fits(product),
      %Equal.sym(Nat,U32.to_nat(U32.from_nat(unit)),unit,unit_exact) : {Nat.is_le(Nat.mul(_,8n),limit_count(product)) == True{} : Bool} first_room),
    Equal.cong(Nat,Nat,value => Nat.mul(value,8n),U32.to_nat(U32.from_nat(unit)),unit,unit_exact))
  +first_rows = {
    %Equal.sym(Nat,U32.to_nat(first),Nat.mul(unit,8n),first_exact) : {Nat.is_le(_,rows) == True{} : Bool}
    Bounds.strict_le(Nat.mul(unit,8n),rows,below) : {Nat.is_le(U32.to_nat(first),rows) == True{} : Bool}}
  +rest_exact = Arithmetic.subtract(product_rows(product),first,limit,limit_fits(product),first_rows,
    Order.transitive(Nat.sub(rows,U32.to_nat(first)),rows,limit_count(product),Order.sub_le(rows,U32.to_nat(first)),rows_room))
  +lane_exact = Bounds.roundtrip(lane,8,{==},Bounds.strict_le(lane,8n,small))
  +remaining = Nat.sub(rows,Nat.mul(unit,8n))
  +decoded = {
    %Equal.sym(Bool,U32.is_lt(U32.from_nat(lane),U32.min(8,rest)),Nat.is_lt(U32.to_nat(U32.from_nat(lane)),U32.to_nat(U32.min(8,rest))),Comparison.lt(U32.from_nat(lane),U32.min(8,rest))) :
      {Nat.is_lt(lane,Nat.min(8n,remaining)) == _ : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(lane)),lane,lane_exact) : {Nat.is_lt(lane,Nat.min(8n,remaining)) == Nat.is_lt(_,U32.to_nat(U32.min(8,rest))) : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.min(8,rest)),Nat.min(8n,U32.to_nat(rest)),U32Extrema.minimum(8,rest)) : {Nat.is_lt(lane,Nat.min(8n,remaining)) == Nat.is_lt(lane,_) : Bool}
    %Equal.sym(Nat,U32.to_nat(rest),Nat.sub(rows,U32.to_nat(first)),rest_exact) : {Nat.is_lt(lane,Nat.min(8n,remaining)) == Nat.is_lt(lane,Nat.min(8n,_)) : Bool}
    %Equal.sym(Nat,U32.to_nat(first),Nat.mul(unit,8n),first_exact) : {Nat.is_lt(lane,Nat.min(8n,remaining)) == Nat.is_lt(lane,Nat.min(8n,Nat.sub(rows,_))) : Bool}
    {==} : {Nat.is_lt(lane,Nat.min(8n,remaining)) == U32.is_lt(U32.from_nat(lane),U32.min(8,rest)) : Bool}}
  +bounded = Equal.trans(Bool,Nat.is_lt(lane,Nat.min(8n,remaining)),U32.is_lt(U32.from_nat(lane),U32.min(8,rest)),True{},decoded,inside)
  Equal.trans(Bool,Nat.is_lt(Nat.add(Nat.mul(unit,8n),lane),rows),Nat.is_lt(lane,remaining),True{},
    Equal.sym(Bool,Nat.is_lt(lane,remaining),Nat.is_lt(Nat.add(Nat.mul(unit,8n),lane),rows),Order.below_difference(Nat.mul(unit,8n),lane,rows,Bounds.strict_le(Nat.mul(unit,8n),rows,below))),
    Bounds.nat_trans(lane,Nat.min(8n,remaining),remaining,bounded,Extrema.minimum_right(8n,remaining)))

# A tile of the output positions before the window's start column.
type TileKeep is Data:
  TileKeep{product: Product,group: Nat,offset: Nat,group_inside: {Nat.is_lt(group,units(product)) == True{} : Bool},
    offset_inside: {Nat.is_lt(offset,output_width(product)) == True{} : Bool},before: {Nat.is_lt(offset,start_column(product)) == True{} : Bool}}

def tile_keep_product(+keep: TileKeep) -> Product:
  TileKeep{product,group,offset,group_inside,offset_inside,before} = keep
  product

def tile_keep_number(+keep: TileKeep) -> Nat:
  TileKeep{product,group,offset,group_inside,offset_inside,before} = keep
  Nat.add(Nat.mul(offset,units(product)),group)

def tile_keep_word(+keep: TileKeep) -> U32:
  U32.from_nat(tile_keep_number(keep))

def tile_kept_compared(+keep: TileKeep,+spot_group: Nat,+column: Nat,
  +spot_inside: {Nat.is_lt(spot_group,units(tile_keep_product(keep))) == True{} : Bool},
  +column_inside: {Nat.is_lt(Nat.add(start_column(tile_keep_product(keep)),column),output_width(tile_keep_product(keep))) == True{} : Bool},
  equal: Bool,compared: {U32.is_eq(tile_word(tile_keep_product(keep),TileSpot{spot_group,column}),tile_keep_word(keep)) == equal : Bool}) ->
  {U32.is_eq(tile_word(tile_keep_product(keep),TileSpot{spot_group,column}),tile_keep_word(keep)) == False{} : Bool}:
  TileKeep{+product,+group,+offset,+group_inside,+offset_inside,+before} = keep
  match equal:
    case False{}: compared
    case True{}:
      +first = start_column(product)
      +count = units(product)
      +number = tile_number(product,spot_group,column)
      +kept = Nat.add(Nat.mul(offset,count),group)
      +kept_exact = Bounds.roundtrip(kept,product_limit(product),limit_fits(product),
        Bounds.strict_le(kept,limit_count(product),Bounds.nat_trans(kept,tile_area(product),limit_count(product),
          RowMajor.radix(offset,output_width(product),group,count,offset_inside,group_inside),tile_area_fits(product))))
      +words = Arithmetic.equal_of_is_eq(U32.from_nat(number),U32.from_nat(kept),compared)
      +numbers = Equal.trans(Nat,number,U32.to_nat(U32.from_nat(number)),kept,
        Equal.sym(Nat,U32.to_nat(U32.from_nat(number)),number,tile_number_exact(product,spot_group,column,spot_inside,column_inside)),
        Equal.trans(Nat,U32.to_nat(U32.from_nat(number)),U32.to_nat(U32.from_nat(kept)),kept,
          Equal.cong(U32,Nat,U32.to_nat,U32.from_nat(number),U32.from_nat(kept),words),kept_exact))
      +positive = Order.positive_bound(group,count,group_inside)
      +positions = Equal.trans(Nat,Nat.add(first,column),Nat.div(number,count),offset,
        Equal.sym(Nat,Nat.div(number,count),Nat.add(first,column),Radix.quotient(Nat.add(first,column),spot_group,count,positive,spot_inside)),
        Equal.trans(Nat,Nat.div(number,count),Nat.div(kept,count),offset,
          Equal.cong(Nat,Nat,total => Nat.div(total,count),number,kept,numbers),
          Radix.quotient(offset,group,count,positive,group_inside)))
      +reached = {
        %positions : {Nat.is_le(first,_) == True{} : Bool}
        Order.le_add(first,column) : {Nat.is_le(first,offset) == True{} : Bool}}
      Empty.absurd({U32.is_eq(tile_word(product,TileSpot{spot_group,column}),tile_keep_word(TileKeep{product,group,offset,group_inside,offset_inside,before})) == False{} : Bool},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_eq(first,first),True{},
          Equal.sym(Bool,Nat.is_eq(first,first),False{},Order.less_not_equal(first,first,Order.le_lt_trans(first,offset,first,reached,before))),
          Order.equal_reflexive(first))))

# No written tile is at a kept address.
def tile_apart(+keep: TileKeep,+spot: TileSpot,+valid: TileWritten(tile_keep_product(keep),spot)) ->
  {U32.is_eq(tile_word(tile_keep_product(keep),spot),tile_keep_word(keep)) == False{} : Bool}:
  TileSpot{+spot_group,+column} = spot
  Gemm.Both{+spot_inside,Gemm.Both{+column_inside,+inside}} = valid
  tile_kept_compared(keep,spot_group,column,spot_inside,column_inside,
    U32.is_eq(tile_word(tile_keep_product(keep),TileSpot{spot_group,column}),tile_keep_word(keep)),{==})

def tile_keep_inside(+keep: TileKeep,+capacity: U32,
  +room: {Nat.is_le(tile_area(tile_keep_product(keep)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.is_lt(tile_keep_word(keep),capacity) == True{} : Bool}:
  TileKeep{+product,+group,+offset,+group_inside,+offset_inside,+before} = keep
  +kept = Nat.add(Nat.mul(offset,units(product)),group)
  +below = RowMajor.radix(offset,output_width(product),group,units(product),offset_inside,group_inside)
  +exact = Bounds.roundtrip(kept,product_limit(product),limit_fits(product),
    Bounds.strict_le(kept,limit_count(product),Bounds.nat_trans(kept,tile_area(product),limit_count(product),below,tile_area_fits(product))))
  %Equal.sym(Bool,U32.is_lt(U32.from_nat(kept),capacity),Nat.is_lt(U32.to_nat(U32.from_nat(kept)),U32.to_nat(capacity)),Comparison.lt(U32.from_nat(kept),capacity)) :
    {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(kept)),kept,exact) : {Nat.is_lt(_,U32.to_nat(capacity)) == True{} : Bool}
  Bounds.nat_trans(kept,tile_area(product),U32.to_nat(capacity),below,room)

# ---- The native tile product ----

# The product's geometry and packed sources with one bias cell per group.
type CellProduct is Data:
  CellProduct{product: Product,bias: Tree.Tree<Tile.Tile8>}

def cell_product(layout: CellProduct) -> Product:
  CellProduct{product,bias} = layout
  product

def cell_bias(layout: CellProduct) -> Tree.Tree<Tile.Tile8>:
  CellProduct{product,bias} = layout
  bias

def cell_matrix(+layout: CellProduct) -> Gemm.TileMatrix:
  tile_matrix(cell_product(layout))

def cell_jobs(+layout: CellProduct) -> Nat:
  jobs(cell_product(layout))

def cell_readers(layout: CellProduct) -> GemmJobs.CellReaders:
  CellProduct{+product,bias} = layout
  GemmJobs.CellReaders{product_row_tiles(product),product_lane_tiles(product),bias,Tree.Leaf{0.0}}

def CellWritten(+layout: CellProduct,+spot: TileSpot) -> Data:
  TileWritten(cell_product(layout),spot)

def cell_word(+layout: CellProduct,+spot: TileSpot) -> U32:
  tile_word(cell_product(layout),spot)

def cell_formed(+layout: CellProduct,+group: U32,+unit: U32,+row: Nat) -> Tile.Tile8:
  +product = cell_product(layout)
  +inner = product_inner(product)
  +first = (group * 8 : U32)
  GemmLaws.finished_tile(~GemmLaws.biased_value,~Gemm.plus,0,U32.min(8,(product_rows(product) - first : U32)),
    GemmLaws.bias_tree(Tree.read(Tile.Tile8,cell_bias(layout),(first / 8 : U32))),
    GemmSpec.dot_row(GemmLaws.RowFirst{},GemmSpec.taps(~Gemm.Direct(),~Gemm.directs,U32.to_nat(inner),Gemm.Both{(group * inner : U32),(unit * inner : U32)}),row,
      GemmSpec.Source{product_row_tiles(product),product_lane_tiles(product)}))

def cell_wanted(+layout: CellProduct,+spot: TileSpot) -> Tile.Tile8:
  TileSpot{group,+column} = spot
  cell_formed(layout,U32.from_nat(group),U32.from_nat(Nat.div(column,8n)),Nat.mod(column,8n))

def CellLocated(+layout: CellProduct,address: U32,value: Tile.Tile8) -> Data:
  GemmJobs.Located(~Tile.Tile8,~CellProduct,~TileSpot,~CellWritten,~cell_word,~cell_wanted,layout,address,value)

# The local bias view changes no row, lane, address or dot product.
def cell_tile_value(+placement: Placement,+bias: Tree.Tree<Tile.Tile8>,+row: Nat,
  +row_inside: {Nat.is_lt(row,GemmJobs.job_rows(~Gemm.Direct(),tile_job(placement))) == True{} : Bool}) ->
  {GemmJobs.cell_job_value(tile_job(placement),cell_readers(CellProduct{placement_product(placement),bias}),row,0n) ==
    cell_wanted(CellProduct{placement_product(placement),bias},TileSpot{placement_unit(placement),cell_column(placement,row)}) : Tile.Tile8}:
  Placement{+product,+unit,+group,+unit_inside,+group_inside} = placement
  +layout = {CellProduct{product,bias} : CellProduct}
  +index = job_index(product,unit,group)
  +column = Nat.add(Nat.mul(group,8n),row)
  +small = ViewWrites.clipped_small(row,8,(product_output(product) - position_word(product,index) : U32),row_inside)
  +observed = GemmJobs.cell_job_value(Gemm.tile_major(tile_matrix(product),index),cell_readers(layout),row,0n)
  %Equal.sym(Nat,Nat.div(column,8n),group,Radix.quotient(group,row,8n,{==},small)) :
    {observed == cell_formed(layout,U32.from_nat(unit),U32.from_nat(_),Nat.mod(column,8n)) : Tile.Tile8}
  %Equal.sym(Nat,Nat.mod(column,8n),row,Radix.remainder(group,row,8n,{==},small)) :
    {observed == cell_formed(layout,U32.from_nat(unit),U32.from_nat(group),_) : Tile.Tile8}
  %unit_value(product,unit,group,unit_inside,group_inside) : {observed == cell_formed(layout,_,U32.from_nat(group),row) : Tile.Tile8}
  %group_value(product,unit,group,unit_inside,group_inside) : {observed == cell_formed(layout,unit_word(product,index),_,row) : Tile.Tile8}
  {==}

def cell_tile_placed(+placement: Placement,+bias: Tree.Tree<Tile.Tile8>,+row: Nat,
  +row_inside: {Nat.is_lt(row,GemmJobs.job_rows(~Gemm.Direct(),tile_job(placement))) == True{} : Bool}) ->
  CellLocated(CellProduct{placement_product(placement),bias},GemmJobs.cell_address(~Gemm.Direct(),tile_job(placement),row,0n),
    GemmJobs.cell_job_value(tile_job(placement),cell_readers(CellProduct{placement_product(placement),bias}),row,0n)):
  Placement{+product,+unit,+group,+unit_inside,+group_inside} = placement
  +whole = {Placement{product,unit,group,unit_inside,group_inside} : Placement}
  GemmJobs.Keyed{TileSpot{unit,cell_column(whole,row)},Gemm.Both{unit_inside,Gemm.Both{column_inside(whole,row,row_inside),column_written(whole,row,row_inside)}},
    tile_address(whole,row),cell_tile_value(whole,bias,row,row_inside)}

def cell_tile_numbered(+product: Product,+bias: Tree.Tree<Tile.Tile8>,+index: Nat,+number: Numbered<product,index>,+row: Nat,
  +row_inside: {Nat.is_lt(row,GemmJobs.job_rows(~Gemm.Direct(),Gemm.tile_major(tile_matrix(product),U32.from_nat(index)))) == True{} : Bool}) ->
  CellLocated(CellProduct{product,bias},GemmJobs.cell_address(~Gemm.Direct(),Gemm.tile_major(tile_matrix(product),U32.from_nat(index)),row,0n),
    GemmJobs.cell_job_value(Gemm.tile_major(tile_matrix(product),U32.from_nat(index)),cell_readers(CellProduct{product,bias}),row,0n)):
  Numbered{+unit,+group,+unit_inside,+group_inside,+exact} = number
  +placement = {Placement{product,unit,group,unit_inside,group_inside} : Placement}
  %Equal.sym(Nat,index,index_number(product,unit,group),exact) :
    CellLocated(CellProduct{product,bias},GemmJobs.cell_address(~Gemm.Direct(),Gemm.tile_major(tile_matrix(product),U32.from_nat(_)),row,0n),
      GemmJobs.cell_job_value(Gemm.tile_major(tile_matrix(product),U32.from_nat(_)),cell_readers(CellProduct{product,bias}),row,0n))
  cell_tile_placed(placement,bias,row,%exact : {Nat.is_lt(row,GemmJobs.job_rows(~Gemm.Direct(),Gemm.tile_major(tile_matrix(product),U32.from_nat(_)))) == True{} : Bool} row_inside)

def cell_tile(+layout: CellProduct,+index: Nat,+inside: {Nat.is_lt(index,cell_jobs(layout)) == True{} : Bool},+row: Nat,+column: Nat,
  +row_inside: {Nat.is_lt(row,GemmJobs.job_rows(~Gemm.Direct(),Gemm.tile_major(cell_matrix(layout),U32.from_nat(index)))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,1n) == True{} : Bool}) ->
  CellLocated(layout,GemmJobs.cell_address(~Gemm.Direct(),Gemm.tile_major(cell_matrix(layout),U32.from_nat(index)),row,column),
    GemmJobs.cell_job_value(Gemm.tile_major(cell_matrix(layout),U32.from_nat(index)),cell_readers(layout),row,column)):
  CellProduct{+product,+bias} = layout
  %Equal.sym(Nat,column,0n,GemmJobs.only_cell(column,column_inside)) :
    CellLocated(layout,GemmJobs.cell_address(~Gemm.Direct(),Gemm.tile_major(cell_matrix(layout),U32.from_nat(index)),row,_),
      GemmJobs.cell_job_value(Gemm.tile_major(cell_matrix(layout),U32.from_nat(index)),cell_readers(layout),row,_))
  cell_tile_numbered(product,bias,index,numbered(product,index,inside),row,row_inside)

def cell_bound(+layout: CellProduct) -> Nat:
  tile_area(cell_product(layout))

def cell_below(+layout: CellProduct,+spot: TileSpot,+valid: CellWritten(layout,spot)) -> {Nat.is_lt(U32.to_nat(cell_word(layout,spot)),cell_bound(layout)) == True{} : Bool}:
  tile_below(cell_product(layout),spot,valid)

def cell_same(+layout: CellProduct,+spot: TileSpot,+other: TileSpot,+valid: CellWritten(layout,spot),+other_valid: CellWritten(layout,other),
  +equal: {cell_word(layout,spot) == cell_word(layout,other) : U32}) -> {cell_wanted(layout,spot) == cell_wanted(layout,other) : Tile.Tile8}:
  Equal.cong(TileSpot,Tile.Tile8,key => cell_wanted(layout,key),spot,other,tile_spot_same(cell_product(layout),spot,other,valid,other_valid,equal))

# Each cell-bias job is checked with the shared write-view theorem.
def CellJob() -> Data:
  GemmJobs.Placed(~CellProduct,~cell_jobs)

def cell_job_layout(+placed: CellJob()) -> CellProduct:
  GemmJobs.Placement{layout,index,inside} = placed
  layout

def cell_job_native(+placed: CellJob()) -> Gemm.Job<Gemm.Direct()>:
  GemmJobs.Placement{+layout,index,inside} = placed
  Gemm.tile_major(cell_matrix(layout),U32.from_nat(index))

def cell_job_virtual(+placed: CellJob()) -> Gemm.Job<Gemm.Direct()>:
  GemmJobs.cell_virtual_job(cell_job_native(placed))

def cell_job_sources(+placed: CellJob()) -> GemmJobs.Readers:
  GemmJobs.cell_virtual_readers(cell_job_native(placed),cell_readers(cell_job_layout(placed)))

def cell_job_rows(+placed: CellJob()) -> {Nat.is_le(GemmJobs.job_rows(~Gemm.Direct(),cell_job_virtual(placed)),8n) == True{} : Bool}:
  GemmJobs.Placement{+layout,+index,+inside} = placed
  tile_rows_small(cell_product(layout),index,inside)

def cell_job_columns(+placed: CellJob()) -> {Nat.is_le(GemmJobs.job_columns(~Gemm.Direct(),GemmLaws.Tiles{},cell_job_virtual(placed)),8n) == True{} : Bool}:
  GemmJobs.Placement{layout,index,inside} = placed
  {==}

def cell_job_cell(+placed: CellJob(),+row: Nat,+column: Nat,
  +row_inside: {Nat.is_lt(row,GemmJobs.job_rows(~Gemm.Direct(),cell_job_virtual(placed))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,GemmJobs.job_columns(~Gemm.Direct(),GemmLaws.Tiles{},cell_job_virtual(placed))) == True{} : Bool}) ->
  CellLocated(cell_job_layout(placed),GemmJobs.cell_address(~Gemm.Direct(),cell_job_virtual(placed),row,column),
    GemmJobs.cell_value(~Gemm.Direct(),~Gemm.directs,~GemmLaws.RowFirst{},~GemmLaws.biased_value,~Gemm.plus,GemmLaws.Tiles{},cell_job_virtual(placed),cell_job_sources(placed),row,column)):
  GemmJobs.Placement{+layout,+index,+inside} = placed
  cell_tile(layout,index,inside,row,column,row_inside,column_inside)

def cell_every_job(+layout: CellProduct,+index: Nat,+inside: {Nat.is_lt(index,cell_jobs(layout)) == True{} : Bool}) ->
  Every.Every(~Tile.Tile8,~CellProduct,~GemmJobs.Located(~Tile.Tile8,~CellProduct,~TileSpot,~CellWritten,~cell_word,~cell_wanted),layout,GemmJobs.cell_step_writes(U32.from_nat(index),cell_matrix(layout),cell_readers(layout))):
  +placed = {GemmJobs.Placement{layout,index,inside} : CellJob()}
  +job = cell_job_native(placed)
  +readers = cell_readers(layout)
  %Equal.sym(List<&2,Writes.Assignment<Tile.Tile8>>,GemmJobs.cell_job_writes(job,readers),
    GemmJobs.job_writes(~Gemm.Direct(),~Gemm.directs,~GemmLaws.RowFirst{},~GemmLaws.Tiles{},~GemmLaws.biased_value,~Gemm.plus,GemmJobs.cell_virtual_job(job),GemmJobs.cell_virtual_readers(job,readers)),
    GemmJobs.cell_job_as_virtual(job,readers)) : Every.Every(~Tile.Tile8,~CellProduct,~GemmJobs.Located(~Tile.Tile8,~CellProduct,~TileSpot,~CellWritten,~cell_word,~cell_wanted),layout,_)
  GemmJobs.job_every(~Gemm.Direct(),~Gemm.directs,~GemmLaws.RowFirst{},~GemmLaws.Tiles{},~GemmLaws.biased_value,~Gemm.plus,
    ~CellJob(),~CellProduct,~cell_job_layout,~GemmJobs.Located(~Tile.Tile8,~CellProduct,~TileSpot,~CellWritten,~cell_word,~cell_wanted),~cell_job_virtual,~cell_job_sources,~cell_job_rows,~cell_job_columns,~cell_job_cell,placed)

def CellRemaining(+layout: CellProduct,rest: Nat,position: Nat,matrix: Gemm.TileMatrix,readers: GemmJobs.CellReaders) -> Data:
  Gemm.Both<{Nat.add(position,rest) == cell_jobs(layout) : Nat},Gemm.Both<{matrix == cell_matrix(layout) : Gemm.TileMatrix},{readers == cell_readers(layout) : GemmJobs.CellReaders}>>

def cell_target(+layout: CellProduct) -> CellProduct:
  layout

def cell_every_step(+layout: CellProduct,+rest: Nat,+position: Nat,+matrix: Gemm.TileMatrix,+readers: GemmJobs.CellReaders,
  left: CellRemaining(layout,1n+rest,position,matrix,readers)) ->
  Every.Every(~Tile.Tile8,~CellProduct,~GemmJobs.Located(~Tile.Tile8,~CellProduct,~TileSpot,~CellWritten,~cell_word,~cell_wanted),cell_target(layout),GemmJobs.cell_step_writes(U32.from_nat(position),matrix,readers)):
  Gemm.Both{room,Gemm.Both{same_matrix,same_readers}} = left
  %Equal.sym(Gemm.TileMatrix,matrix,cell_matrix(layout),same_matrix) :
    Every.Every(~Tile.Tile8,~CellProduct,~GemmJobs.Located(~Tile.Tile8,~CellProduct,~TileSpot,~CellWritten,~cell_word,~cell_wanted),layout,GemmJobs.cell_step_writes(U32.from_nat(position),_,readers))
  %Equal.sym(GemmJobs.CellReaders,readers,cell_readers(layout),same_readers) :
    Every.Every(~Tile.Tile8,~CellProduct,~GemmJobs.Located(~Tile.Tile8,~CellProduct,~TileSpot,~CellWritten,~cell_word,~cell_wanted),layout,GemmJobs.cell_step_writes(U32.from_nat(position),cell_matrix(layout),_))
  cell_every_job(layout,position,Order.below_complement(position,rest,cell_jobs(layout),room))

def cell_remaining_kept(+layout: CellProduct,+rest: Nat,+position: Nat,+matrix: Gemm.TileMatrix,+readers: GemmJobs.CellReaders,
  left: CellRemaining(layout,1n+rest,position,matrix,readers)) -> CellRemaining(layout,rest,1n+position,matrix,GemmJobs.cell_kept(U32.from_nat(position),matrix,readers)):
  Gemm.Both{room,same} = left
  Gemm.Both{Equal.trans(Nat,1n+Nat.add(position,rest),Nat.add(position,1n+rest),cell_jobs(layout),
    Equal.sym(Nat,Nat.add(position,1n+rest),1n+Nat.add(position,rest),Addition.successor_right(position,rest)),room),same}

def cell_writes(+layout: CellProduct) -> List<&2,Writes.Assignment<Tile.Tile8>>:
  GemmJobs.cell_all_writes(cell_jobs(layout),0,cell_matrix(layout),cell_readers(layout))

def cell_all_every(+layout: CellProduct) -> Every.Every(~Tile.Tile8,~CellProduct,~GemmJobs.Located(~Tile.Tile8,~CellProduct,~TileSpot,~CellWritten,~cell_word,~cell_wanted),layout,cell_writes(layout)):
  Fold.every(~Tile.Tile8,~Gemm.TileMatrix,~GemmJobs.CellReaders,~GemmJobs.cell_step_writes,~GemmJobs.cell_kept,~CellProduct,~CellProduct,~cell_target,~GemmJobs.Located(~Tile.Tile8,~CellProduct,~TileSpot,~CellWritten,~cell_word,~cell_wanted),
    ~CellRemaining,~cell_every_step,~cell_remaining_kept,layout,cell_jobs(layout),0n,cell_matrix(layout),cell_readers(layout),Gemm.Both{{==},Gemm.Both{{==},{==}}})

# The shared tile schedule contains every requested output position.
def cell_found(+layout: CellProduct,+spot: TileSpot,+valid: CellWritten(layout,spot)) ->
  GemmJobs.Locate(~Gemm.TileMatrix,~Gemm.Direct(),~Gemm.tile_major,~GemmLaws.Tiles{},~CellProduct,~cell_matrix,~cell_jobs,~TileSpot,~cell_word,layout,spot):
  tile_found(cell_product(layout),spot,valid)

def CellQuery() -> Data:
  GemmJobs.Asked(~CellProduct,~TileSpot,~CellWritten)

def cell_query_word(+query: CellQuery()) -> U32:
  GemmJobs.query_word(~CellProduct,~TileSpot,~CellWritten,~cell_word,query)

def cell_query_index(+query: CellQuery()) -> Nat:
  GemmJobs.query_index(~Gemm.TileMatrix,~Gemm.Direct(),~Gemm.tile_major,~GemmLaws.Tiles{},~CellProduct,~cell_matrix,~cell_jobs,~TileSpot,~CellWritten,~cell_word,~cell_found,query)

def CellScheduled(+query: CellQuery(),rest: Nat,position: Nat,matrix: Gemm.TileMatrix,readers: GemmJobs.CellReaders) -> Data:
  +layout = GemmJobs.query_layout(~CellProduct,~TileSpot,~CellWritten,query)
  Gemm.Both<{matrix == cell_matrix(layout) : Gemm.TileMatrix},{readers == cell_readers(layout) : GemmJobs.CellReaders}>

def cell_scheduled_kept(+query: CellQuery(),+rest: Nat,+position: Nat,+matrix: Gemm.TileMatrix,+readers: GemmJobs.CellReaders,
  schedule: CellScheduled(query,1n+rest,position,matrix,readers)) -> CellScheduled(query,rest,1n+position,matrix,GemmJobs.cell_kept(U32.from_nat(position),matrix,readers)):
  schedule

def cell_found_contains(+layout: CellProduct,+spot: TileSpot,+located: GemmJobs.Locate(~Gemm.TileMatrix,~Gemm.Direct(),~Gemm.tile_major,~GemmLaws.Tiles{},~CellProduct,~cell_matrix,~cell_jobs,~TileSpot,~cell_word,layout,spot)) ->
  {WriteReadback.contains(~Tile.Tile8,cell_word(layout,spot),GemmJobs.cell_step_writes(
    U32.from_nat(GemmJobs.found_index(~Gemm.TileMatrix,~Gemm.Direct(),~Gemm.tile_major,~GemmLaws.Tiles{},~CellProduct,~cell_matrix,~cell_jobs,~TileSpot,~cell_word,layout,spot,located)),cell_matrix(layout),cell_readers(layout))) == True{} : Bool}:
  GemmJobs.Found{+index,+row,+column,+inside,+row_inside,+column_inside,+address_exact} = located
  +job = Gemm.tile_major(cell_matrix(layout),U32.from_nat(index))
  %address_exact : {WriteReadback.contains(~Tile.Tile8,_,GemmJobs.cell_job_writes(job,cell_readers(layout))) == True{} : Bool}
  GemmJobs.cell_job_contains(job,cell_readers(layout),row,column,row_inside,column_inside,tile_rows_small(cell_product(layout),index,inside))

def cell_scheduled_found(+query: CellQuery(),+rest: Nat,+matrix: Gemm.TileMatrix,+readers: GemmJobs.CellReaders,
  schedule: CellScheduled(query,1n+rest,cell_query_index(query),matrix,readers)) ->
  {WriteReadback.contains(~Tile.Tile8,cell_query_word(query),GemmJobs.cell_step_writes(U32.from_nat(cell_query_index(query)),matrix,readers)) == True{} : Bool}:
  GemmJobs.Query{+layout,+spot,+valid} = query
  Gemm.Both{same_matrix,same_readers} = schedule
  +index = cell_query_index(query)
  %Equal.sym(Gemm.TileMatrix,matrix,cell_matrix(layout),same_matrix) :
    {WriteReadback.contains(~Tile.Tile8,cell_word(layout,spot),GemmJobs.cell_step_writes(U32.from_nat(index),_,readers)) == True{} : Bool}
  %Equal.sym(GemmJobs.CellReaders,readers,cell_readers(layout),same_readers) :
    {WriteReadback.contains(~Tile.Tile8,cell_word(layout,spot),GemmJobs.cell_step_writes(U32.from_nat(index),cell_matrix(layout),_)) == True{} : Bool}
  cell_found_contains(layout,spot,cell_found(layout,spot,valid))

def cell_all_contains(+layout: CellProduct,+spot: TileSpot,+valid: CellWritten(layout,spot)) ->
  {WriteReadback.contains(~Tile.Tile8,cell_word(layout,spot),cell_writes(layout)) == True{} : Bool}:
  +query = {GemmJobs.Query{layout,spot,valid} : CellQuery()}
  Fold.contains_at(~Tile.Tile8,~Gemm.TileMatrix,~GemmJobs.CellReaders,~GemmJobs.cell_step_writes,~GemmJobs.cell_kept,
    ~CellQuery(),~cell_query_word,~cell_query_index,~CellScheduled,~cell_scheduled_kept,~cell_scheduled_found,query,cell_jobs(layout),cell_query_index(query),0n,cell_matrix(layout),cell_readers(layout),Gemm.Both{{==},{==}},
    GemmJobs.found_inside(~Gemm.TileMatrix,~Gemm.Direct(),~Gemm.tile_major,~GemmLaws.Tiles{},~CellProduct,~cell_matrix,~cell_jobs,~TileSpot,~cell_word,layout,spot,cell_found(layout,spot,valid)),{==})

def cell_run(+matrix: Gemm.TileMatrix,+rows: Tree.Tree<Tile.Tile8>,+lanes: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<Tile.Tile8>,+tree: Tree.Tree<Tile.Tile8>) -> GemmLaws.CellOperands:
  GemmLaws.cell_run(U32.to_nat(Gemm.tile_jobs(matrix)),matrix,GemmLaws.CellOperands{bias,GemmLaws.Operands{rows,lanes,Tree.Leaf{0.0},Writes.Initial{tree}}})

def cell_tree(+matrix: Gemm.TileMatrix,+rows: Tree.Tree<Tile.Tile8>,+lanes: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<Tile.Tile8>,+tree: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  GemmJobs.cell_output(cell_run(matrix,rows,lanes,bias,tree))

def cell_product_tree(+layout: CellProduct,+tree: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  +product = cell_product(layout)
  cell_tree(cell_matrix(layout),product_row_tiles(product),product_lane_tiles(product),cell_bias(layout),tree)

def cell_run_tree(+layout: CellProduct,+tree: Tree.Tree<Tile.Tile8>) ->
  {cell_product_tree(layout,tree) == Writes.pure(Tile.Tile8,cell_writes(layout),tree) : Tree.Tree<Tile.Tile8>}:
  CellProduct{+product,+bias} = layout
  %Equal.sym(Nat,U32.to_nat(Gemm.tile_jobs(tile_matrix(product))),jobs(product),jobs_exact(product)) :
    {GemmJobs.cell_output(GemmLaws.cell_run(_,tile_matrix(product),GemmLaws.CellOperands{bias,GemmLaws.Operands{product_row_tiles(product),product_lane_tiles(product),Tree.Leaf{0.0},Writes.Initial{tree}}})) ==
      Writes.pure(Tile.Tile8,cell_writes(layout),tree) : Tree.Tree<Tile.Tile8>}
  GemmJobs.cell_run_writes(jobs(product),0,tile_matrix(product),GemmLaws.CellOperands{bias,GemmLaws.Operands{product_row_tiles(product),product_lane_tiles(product),Tree.Leaf{0.0},Writes.Initial{tree}}})

def cell_tile_readback(+layout: CellProduct,+spot: TileSpot,+valid: CellWritten(layout,spot),+depth: Nat,+tree: Tree.Tree<Tile.Tile8>,
  +balanced: Shape.Balanced(Tile.Tile8,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +room: {Nat.is_le(cell_bound(layout),U32.to_nat(Tree.capacity(Tile.Tile8,tree))) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,cell_product_tree(layout,tree),cell_word(layout,spot)) == cell_wanted(layout,spot) : Tile.Tile8}:
  %Equal.sym(Tree.Tree<Tile.Tile8>,cell_product_tree(layout,tree),Writes.pure(Tile.Tile8,cell_writes(layout),tree),cell_run_tree(layout,tree)) :
    {Tree.read(Tile.Tile8,_,cell_word(layout,spot)) == cell_wanted(layout,spot) : Tile.Tile8}
  GemmJobs.located_readback(~Tile.Tile8,~CellProduct,~TileSpot,~CellWritten,~cell_word,~cell_wanted,~cell_bound,~cell_below,~cell_same,
    layout,spot,valid,depth,tree,balanced,depth_ok,room,cell_writes(layout),cell_all_every(layout),cell_all_contains(layout,spot,valid))

# The first channel of a group selects that group's bias cell.
def channel_group_word(+product: Product,+channel: Nat,+parts: ChannelParts<product,channel>) ->
  {((U32.from_nat(Nat.div(channel,8n)) * 8 : U32) / 8 : U32) == U32.from_nat(Nat.div(channel,8n)) : U32}:
  ChannelParts{split,small,+first_exact,lane_exact,channel_exact} = parts
  +group = Nat.div(channel,8n)
  +first = (U32.from_nat(group) * 8 : U32)
  +exact = Equal.trans(Nat,U32.to_nat((first / 8 : U32)),Nat.div(U32.to_nat(first),8n),group,
    U32Division.divide(first,8,16,{==},{==},{==}),
    %Equal.sym(Nat,U32.to_nat(first),Nat.mul(group,8n),first_exact) : {Nat.div(_,8n) == group : Nat} NaturalDivision.exact_quotient(group,8n,{==}))
  %exact : {(first / 8 : U32) == U32.from_nat(_) : U32}
  Equal.sym(U32,U32.from_nat(U32.to_nat((first / 8 : U32))),(first / 8 : U32),Arithmetic.native_roundtrip((first / 8 : U32)))

def cell_lane_wanted(+layout: CellProduct,+channel: Nat,+column: Nat) -> F32:
  +product = cell_product(layout)
  Gemm.plus(row_dot(product,product_row_tiles(product),channel,column),
    TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,cell_bias(layout),U32.from_nat(Nat.div(channel,8n)))))

def cell_lane_readback(+layout: CellProduct,+channel: Nat,+column: Nat,
  +channel_inside: {Nat.is_lt(channel,row_count(cell_product(layout))) == True{} : Bool},
  +column_inside: {Nat.is_lt(Nat.add(start_column(cell_product(layout)),column),output_width(cell_product(layout))) == True{} : Bool},
  +written: {Nat.is_lt(column,U32.to_nat(product_columns(cell_product(layout)))) == True{} : Bool},+depth: Nat,+tree: Tree.Tree<Tile.Tile8>,
  +balanced: Shape.Balanced(Tile.Tile8,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +room: {Nat.is_le(cell_bound(layout),U32.to_nat(Tree.capacity(Tile.Tile8,tree))) == True{} : Bool}) ->
  {TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,cell_product_tree(layout,tree),tile_word(cell_product(layout),TileSpot{Nat.div(channel,8n),column}))) ==
    cell_lane_wanted(layout,channel,column) : F32}:
  +product = cell_product(layout)
  +group = Nat.div(channel,8n)
  +lane = Nat.mod(channel,8n)
  +inner = product_inner(product)
  +first = (U32.from_nat(group) * 8 : U32)
  +lanes = U32.min(8,(product_rows(product) - first : U32))
  +biases = Tree.read(Tile.Tile8,cell_bias(layout),(first / 8 : U32))
  +bias = GemmLaws.bias_tree(biases)
  +taps = GemmSpec.taps(~Gemm.Direct(),~Gemm.directs,U32.to_nat(inner),Gemm.Both{(U32.from_nat(group) * inner : U32),(U32.from_nat(Nat.div(column,8n)) * inner : U32)})
  +source = {GemmSpec.Source{product_row_tiles(product),product_lane_tiles(product)} : GemmSpec.Source}
  +values = GemmSpec.dot_row(GemmLaws.RowFirst{},taps,Nat.mod(column,8n),source)
  +small = DivisionOrder.remainder_small(channel,8n,{==})
  +wanted = cell_lane_wanted(layout,channel,column)
  %Equal.sym(Tile.Tile8,Tree.read(Tile.Tile8,cell_product_tree(layout,tree),tile_word(product,TileSpot{group,column})),cell_wanted(layout,TileSpot{group,column}),
    cell_tile_readback(layout,TileSpot{group,column},Gemm.Both{channel_unit(product,channel,channel_inside),Gemm.Both{column_inside,written}},depth,tree,balanced,depth_ok,room)) :
    {TileLaws.select_lane(lane,_) == wanted : F32}
  Equal.trans(F32,TileLaws.select_lane(lane,GemmLaws.finished_tile(~GemmLaws.biased_value,~Gemm.plus,0,lanes,bias,values)),
    Gemm.plus(TileLaws.select_lane(lane,values),Tree.read(F32,bias,GemmLaws.shifted(0,lane))),wanted,
    GemmLaws.finished_lane(~GemmLaws.biased_value,~Gemm.plus,0,lanes,bias,values,lane,channel_lane(product,channel,channel_inside),small),
    %Equal.sym(F32,TileLaws.select_lane(lane,values),GemmSpec.dot(GemmLaws.RowFirst{},taps,Nat.mod(column,8n),lane,source,0.0),GemmSpec.dot_lane(GemmLaws.RowFirst{},taps,Nat.mod(column,8n),lane,source,small)) :
      {Gemm.plus(_,Tree.read(F32,bias,GemmLaws.shifted(0,lane))) == wanted : F32}
    %Equal.sym(F32,Tree.read(F32,bias,GemmLaws.shifted(0,lane)),TileLaws.select_lane(lane,biases),GemmLaws.bias_tree_lane(biases,lane,small)) :
      {Gemm.plus(row_dot(product,product_row_tiles(product),channel,column),_) == wanted : F32}
    %Equal.sym(U32,(first / 8 : U32),U32.from_nat(group),channel_group_word(product,channel,channel_parts(product,channel,channel_inside))) :
      {Gemm.plus(row_dot(product,product_row_tiles(product),channel,column),TileLaws.select_lane(lane,Tree.read(Tile.Tile8,cell_bias(layout),_))) == wanted : F32}
    {==})

type CellKeep is Data:
  CellKeep{keep: TileKeep,bias: Tree.Tree<Tile.Tile8>}

def cell_keep_layout(+kept: CellKeep) -> CellProduct:
  CellKeep{keep,bias} = kept
  CellProduct{tile_keep_product(keep),bias}

def cell_keep_word(+kept: CellKeep) -> U32:
  CellKeep{keep,bias} = kept
  tile_keep_word(keep)

def cell_keep_apart(+kept: CellKeep,+spot: TileSpot,+valid: CellWritten(cell_keep_layout(kept),spot)) ->
  {U32.is_eq(cell_word(cell_keep_layout(kept),spot),cell_keep_word(kept)) == False{} : Bool}:
  CellKeep{+keep,bias} = kept
  tile_apart(keep,spot,valid)

def cell_keeps(+keep: TileKeep,+bias: Tree.Tree<Tile.Tile8>,+depth: Nat,+tree: Tree.Tree<Tile.Tile8>,
  +balanced: Shape.Balanced(Tile.Tile8,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +room: {Nat.is_le(tile_area(tile_keep_product(keep)),U32.to_nat(Tree.capacity(Tile.Tile8,tree))) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,cell_product_tree(CellProduct{tile_keep_product(keep),bias},tree),tile_keep_word(keep)) == Tree.read(Tile.Tile8,tree,tile_keep_word(keep)) : Tile.Tile8}:
  +layout = {CellProduct{tile_keep_product(keep),bias} : CellProduct}
  +kept = {CellKeep{keep,bias} : CellKeep}
  +writes = cell_writes(layout)
  +holds = cell_all_every(layout)
  +output = {GemmJobs.Output{layout,Tree.capacity(Tile.Tile8,tree),room} : GemmJobs.Bounded(~CellProduct,~cell_bound)}
  %Equal.sym(Tree.Tree<Tile.Tile8>,cell_product_tree(layout,tree),Writes.pure(Tile.Tile8,writes,tree),cell_run_tree(layout,tree)) :
    {Tree.read(Tile.Tile8,_,tile_keep_word(keep)) == Tree.read(Tile.Tile8,tree,tile_keep_word(keep)) : Tile.Tile8}
  WriteReadback.excluded_readback(~Tile.Tile8,writes,tree,depth,tile_keep_word(keep),balanced,depth_ok,
    Every.inside(~Tile.Tile8,~CellProduct,~GemmJobs.Located(~Tile.Tile8,~CellProduct,~TileSpot,~CellWritten,~cell_word,~cell_wanted),
      ~GemmJobs.Bounded(~CellProduct,~cell_bound),~GemmJobs.output_layout(~CellProduct,~cell_bound),~GemmJobs.output_capacity(~CellProduct,~cell_bound),
      ~GemmJobs.located_inside(~Tile.Tile8,~CellProduct,~TileSpot,~CellWritten,~cell_word,~cell_wanted,~cell_bound,~cell_below),output,writes,holds),
    tile_keep_inside(keep,Tree.capacity(Tile.Tile8,tree),room),
    Every.excludes(~Tile.Tile8,~CellProduct,~GemmJobs.Located(~Tile.Tile8,~CellProduct,~TileSpot,~CellWritten,~cell_word,~cell_wanted),~CellKeep,~cell_keep_layout,~cell_keep_word,
      ~GemmJobs.located_apart(~Tile.Tile8,~CellProduct,~TileSpot,~CellWritten,~cell_word,~cell_wanted,~CellKeep,~cell_keep_layout,~cell_keep_word,~cell_keep_apart),kept,writes,holds))

def cell_native(+matrix: Gemm.TileMatrix,+rows: Tree.Tree<Tile.Tile8>,+lanes: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<Tile.Tile8>,+tree: Tree.Tree<Tile.Tile8>) ->
  {Gemm.cell_run(U32.to_nat(Gemm.tile_jobs(matrix)),matrix,GemmLaws.cell_represent(GemmLaws.CellOperands{bias,GemmLaws.Operands{rows,lanes,Tree.Leaf{0.0},Writes.Initial{tree}}})) ==
    GemmLaws.cell_represent(cell_run(matrix,rows,lanes,bias,tree)) : Gemm.CellOperands}:
  GemmLaws.cell_run_matches(U32.to_nat(Gemm.tile_jobs(matrix)),matrix,GemmLaws.CellOperands{bias,GemmLaws.Operands{rows,lanes,Tree.Leaf{0.0},Writes.Initial{tree}}})

def cell_tree_writes(+matrix: Gemm.TileMatrix,+rows: Tree.Tree<Tile.Tile8>,+lanes: Tree.Tree<Tile.Tile8>,+bias: Tree.Tree<Tile.Tile8>,+tree: Tree.Tree<Tile.Tile8>) ->
  {cell_tree(matrix,rows,lanes,bias,tree) ==
    Writes.pure(Tile.Tile8,GemmJobs.cell_all_writes(U32.to_nat(Gemm.tile_jobs(matrix)),0,matrix,GemmJobs.CellReaders{rows,lanes,bias,Tree.Leaf{0.0}}),tree) : Tree.Tree<Tile.Tile8>}:
  GemmJobs.cell_run_writes(U32.to_nat(Gemm.tile_jobs(matrix)),0,matrix,GemmLaws.CellOperands{bias,GemmLaws.Operands{rows,lanes,Tree.Leaf{0.0},Writes.Initial{tree}}})

def cell_output_matches(+state: GemmLaws.CellOperands) ->
  {Gemm.cell_output(GemmLaws.cell_represent(state)) == Tree.pack(Tile.Tile8,GemmJobs.cell_output(state)) : Array<Tile.Tile8>}:
  GemmLaws.CellOperands{bias,GemmLaws.Operands{rows,lanes,unused_bias,output}} = state
  {==}
