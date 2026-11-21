# The staged output of a pointwise run reads back, at pair p of position q,
# the stored row of the job of p and the unit of q, plus the biases of p:
# exactly one job stores there.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ./storage_tree_model.bend as Tree
import ./write_program.bend as Writes
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_multiplication.bend as Multiplication
import ./nat_order.bend as Order
import ./row_major_index.bend as RowMajor
import ./natural_addition.bend as Addition
import ./word_encoding.bend as WordEncoding
import ./u32_comparison.bend as Comparison
import ./u32_extrema.bend as U32Extrema
import ./nat_to_u32_bounds.bend as Bounds
import ./binary_radix.bend as BinaryRadix
import ./kernel_gemm_readback.bend as GemmReadback
import ./boolean_logic.bend as Boolean
import ./write_program_every.bend as Every
import ./write_program_fold.bend as Fold
import ./kernel_gemm_refinement.bend as GemmLaws
import ./kernel_gemm_point_spec.bend as PointSpec
import ./kernel_gemm_point_jobs.bend as PointJobs
import ./kernel_gemm_jobs.bend as GemmJobs
import ./write_program_readback.bend as WriteReadback
import ./storage_shape.bend as Shape
import ./nat_extrema.bend as NatExtrema
import ./traversal_update_readback.bend as Updates
import ./nat_division_order.bend as DivisionOrder
import ./nat_division.bend as Division
import ./u32_division.bend as U32Division

# ---- The address of a stored row ----

# Row row of a block stored from base, pairs apart, as the job writes it.
def row_offset(row: Nat,+base: U32,+pairs: U32) -> U32:
  match row:
    case 0n: base
    case 1n+a:
      match a:
        case 0n: (base + pairs : U32)
        case 1n+b:
          match b:
            case 0n: (base + 2 * pairs : U32)
            case 1n+c: (base + 3 * pairs : U32)

def block_base(+start: U32,+unit: U32,+pairs: U32,+pair: U32) -> U32:
  ((start + unit * 4 : U32) * pairs + pair : U32)

# The first position of a unit and the number of a stored pair.
def first_position(+start: U32,+unit: U32) -> Nat:
  Nat.add(U32.to_nat(start),Nat.mul(U32.to_nat(unit),4n))

def stored_number(+position: Nat,+pairs: U32,+pair: U32) -> Nat:
  Nat.add(Nat.mul(position,U32.to_nat(pairs)),U32.to_nat(pair))

# The words of a block stay below cap: its last row's number does.
type Room<-start: U32,-unit: U32,-pairs: U32,-pair: U32,-cap: U32> is Data:
  Room{cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},positive: {Nat.is_lt(0n,U32.to_nat(pairs)) == True{} : Bool},
    last: {Nat.is_le(stored_number(Nat.add(first_position(start,unit),3n),pairs,pair),U32.to_nat(cap)) == True{} : Bool}}

def right_offset_le(+left: Nat,+right: Nat,+offset: Nat,ordered: {Nat.is_le(left,right) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(left,offset),Nat.add(right,offset)) == True{} : Bool}:
  %Order.commutative(offset,left) : {Nat.is_le(_,Nat.add(right,offset)) == True{} : Bool}
  %Order.commutative(offset,right) : {Nat.is_le(Nat.add(offset,left),_) == True{} : Bool}
  Order.offset_le(offset,left,right,ordered)

# A number of a row is at most the last row's.
def row_room(+position: Nat,+row: Nat,+pairs: U32,+pair: U32,+cap: U32,small: {Nat.is_le(row,3n) == True{} : Bool},
  last: {Nat.is_le(stored_number(Nat.add(position,3n),pairs,pair),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(stored_number(Nat.add(position,row),pairs,pair),U32.to_nat(cap)) == True{} : Bool}:
  Order.transitive(stored_number(Nat.add(position,row),pairs,pair),stored_number(Nat.add(position,3n),pairs,pair),U32.to_nat(cap),
    right_offset_le(Nat.mul(Nat.add(position,row),U32.to_nat(pairs)),Nat.mul(Nat.add(position,3n),U32.to_nat(pairs)),U32.to_nat(pair),
      RowMajor.mul_le(Nat.add(position,row),Nat.add(position,3n),U32.to_nat(pairs),Order.offset_le(position,row,3n,small))),
    last)

# The base of a block plus k pairs is the number of its row k.
def shifted_number(+position: Nat,+pairs: Nat,+pair: Nat,+row: Nat,+added: Nat,added_exact: {added == Nat.mul(row,pairs) : Nat}) ->
  {Nat.add(Nat.add(Nat.mul(position,pairs),pair),added) == Nat.add(Nat.mul(Nat.add(position,row),pairs),pair) : Nat}:
  %Equal.sym(Nat,Nat.mul(Nat.add(position,row),pairs),Nat.add(Nat.mul(position,pairs),Nat.mul(row,pairs)),RowMajor.distrib(position,row,pairs)) :
    {Nat.add(Nat.add(Nat.mul(position,pairs),pair),added) == Nat.add(_,pair) : Nat}
  %added_exact : {Nat.add(Nat.add(Nat.mul(position,pairs),pair),added) == Nat.add(Nat.add(Nat.mul(position,pairs),_),pair) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(position,pairs),pair),added),Nat.add(Nat.mul(position,pairs),Nat.add(pair,added)),Addition.associative(Nat.mul(position,pairs),pair,added)) : {_ == Nat.add(Nat.add(Nat.mul(position,pairs),added),pair) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(position,pairs),added),pair),Nat.add(Nat.mul(position,pairs),Nat.add(added,pair)),Addition.associative(Nat.mul(position,pairs),added,pair)) : {Nat.add(Nat.mul(position,pairs),Nat.add(pair,added)) == _ : Nat}
  Equal.cong(Nat,Nat,tail => Nat.add(Nat.mul(position,pairs),tail),Nat.add(pair,added),Nat.add(added,pair),Order.commutative(pair,added))

# The words of a block are their numbers: the first position, the block base
# and each row's word.
type Words<-start: U32,-unit: U32,-pairs: U32,-pair: U32> is Data:
  Words{position: {U32.to_nat((start + unit * 4 : U32)) == first_position(start,unit) : Nat},
    base: {U32.to_nat(block_base(start,unit,pairs,pair)) == stored_number(first_position(start,unit),pairs,pair) : Nat}}

def words(+start: U32,+unit: U32,+pairs: U32,+pair: U32,+cap: U32,+room: Room<start,unit,pairs,pair,cap>) -> Words<start,unit,pairs,pair>:
  Room{+cap_ok,+positive,+last} = room
  +s = U32.to_nat(start)
  +u = U32.to_nat(unit)
  +pn = U32.to_nat(pairs)
  +an = U32.to_nat(pair)
  +first = first_position(start,unit)
  +top = Nat.add(first,3n)
  +limit = U32.to_nat(cap)
  +first_room = Order.transitive(first,stored_number(top,pairs,pair),limit,
    Order.transitive(first,Nat.mul(top,pn),stored_number(top,pairs,pair),
      Order.transitive(first,top,Nat.mul(top,pn),Order.le_add(first,3n),RowMajor.product_covers(top,pn,positive)),Order.le_add(Nat.mul(top,pn),an)),last)
  +times_four = Multiplication.bounded_multiply(unit,4,cap,cap_ok,Order.transitive(Nat.mul(u,4n),first,limit,Order.right_le(s,Nat.mul(u,4n)),first_room))
  +position = {
    %times_four : {U32.to_nat((start + unit * 4 : U32)) == Nat.add(s,_) : Nat}
    Arithmetic.add(start,(unit * 4 : U32),cap,cap_ok,
      %Equal.sym(Nat,U32.to_nat((unit * 4 : U32)),Nat.mul(u,4n),times_four) : {Nat.is_le(Nat.add(s,_),limit) == True{} : Bool}
      first_room) : {U32.to_nat((start + unit * 4 : U32)) == first : Nat}}
  +product_room = Order.transitive(Nat.mul(first,pn),stored_number(top,pairs,pair),limit,
    Order.transitive(Nat.mul(first,pn),Nat.mul(top,pn),stored_number(top,pairs,pair),RowMajor.mul_le(first,top,pn,Order.le_add(first,3n)),Order.le_add(Nat.mul(top,pn),an)),last)
  +product = {
    %position : {U32.to_nat(((start + unit * 4 : U32) * pairs : U32)) == Nat.mul(_,pn) : Nat}
    Multiplication.bounded_multiply((start + unit * 4 : U32),pairs,cap,cap_ok,
      %Equal.sym(Nat,U32.to_nat((start + unit * 4 : U32)),first,position) : {Nat.is_le(Nat.mul(_,pn),limit) == True{} : Bool}
      product_room) : {U32.to_nat(((start + unit * 4 : U32) * pairs : U32)) == Nat.mul(first,pn) : Nat}}
  +base_room = Order.transitive(stored_number(first,pairs,pair),stored_number(top,pairs,pair),limit,
    right_offset_le(Nat.mul(first,pn),Nat.mul(top,pn),an,RowMajor.mul_le(first,top,pn,Order.le_add(first,3n))),last)
  +base = {
    %product :
      {U32.to_nat(block_base(start,unit,pairs,pair)) == Nat.add(_,an) : Nat}
    Arithmetic.add(((start + unit * 4 : U32) * pairs : U32),pair,cap,cap_ok,
      %Equal.sym(Nat,U32.to_nat(((start + unit * 4 : U32) * pairs : U32)),Nat.mul(first,pn),product) : {Nat.is_le(Nat.add(_,an),limit) == True{} : Bool}
      base_room) : {U32.to_nat(block_base(start,unit,pairs,pair)) == stored_number(first,pairs,pair) : Nat}}
  Words{position,base}

# The base word plus row pairs is the number of row row.
def added_number(+start: U32,+unit: U32,+pairs: U32,+pair: U32,+cap: U32,+room: Room<start,unit,pairs,pair,cap>,+row: Nat,+small: {Nat.is_le(row,3n) == True{} : Bool},
  +added: U32,+exact: {U32.to_nat(added) == Nat.mul(row,U32.to_nat(pairs)) : Nat},+known: Words<start,unit,pairs,pair>) ->
  {U32.to_nat((block_base(start,unit,pairs,pair) + added : U32)) == stored_number(Nat.add(first_position(start,unit),row),pairs,pair) : Nat}:
  Room{+cap_ok,+positive,+last} = room
  Words{+position,+base} = known
  +first = first_position(start,unit)
  +base_word = block_base(start,unit,pairs,pair)
  +shifted = shifted_number(first,U32.to_nat(pairs),U32.to_nat(pair),row,U32.to_nat(added),exact)
  +bound = {
    %Equal.sym(Nat,U32.to_nat(base_word),stored_number(first,pairs,pair),base) : {Nat.is_le(Nat.add(_,U32.to_nat(added)),U32.to_nat(cap)) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(stored_number(first,pairs,pair),U32.to_nat(added)),stored_number(Nat.add(first,row),pairs,pair),shifted) :
      {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
    row_room(first,row,pairs,pair,cap,small,last) : {Nat.is_le(Nat.add(U32.to_nat(base_word),U32.to_nat(added)),U32.to_nat(cap)) == True{} : Bool}}
  Equal.trans(Nat,U32.to_nat((base_word + added : U32)),Nat.add(U32.to_nat(base_word),U32.to_nat(added)),stored_number(Nat.add(first,row),pairs,pair),
    Arithmetic.add(base_word,added,cap,cap_ok,bound),
    Equal.trans(Nat,Nat.add(U32.to_nat(base_word),U32.to_nat(added)),Nat.add(stored_number(first,pairs,pair),U32.to_nat(added)),stored_number(Nat.add(first,row),pairs,pair),
      Equal.cong(Nat,Nat,value => Nat.add(value,U32.to_nat(added)),U32.to_nat(base_word),stored_number(first,pairs,pair),base),shifted))

# k pairs for k of 2 and 3 stay below cap.
def pairs_room(+start: U32,+unit: U32,+pairs: U32,+pair: U32,+cap: U32,+room: Room<start,unit,pairs,pair,cap>,+factor: Nat,+small: {Nat.is_le(factor,3n) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(factor,U32.to_nat(pairs)),U32.to_nat(cap)) == True{} : Bool}:
  Room{+cap_ok,+positive,+last} = room
  +top = Nat.add(first_position(start,unit),3n)
  Order.transitive(Nat.mul(factor,U32.to_nat(pairs)),stored_number(top,pairs,pair),U32.to_nat(cap),
    Order.transitive(Nat.mul(factor,U32.to_nat(pairs)),Nat.mul(top,U32.to_nat(pairs)),stored_number(top,pairs,pair),
      RowMajor.mul_le(factor,top,U32.to_nat(pairs),Order.transitive(factor,3n,top,small,Order.right_le(first_position(start,unit),3n))),
      Order.le_add(Nat.mul(top,U32.to_nat(pairs)),U32.to_nat(pair))),
    last)

def room_cap(+start: U32,+unit: U32,+pairs: U32,+pair: U32,+cap: U32,+room: Room<start,unit,pairs,pair,cap>) -> {U32.is_le(cap,2147483648) == True{} : Bool}:
  Room{+cap_ok,positive,last} = room
  cap_ok

def known_base(+start: U32,+unit: U32,+pairs: U32,+pair: U32,+known: Words<start,unit,pairs,pair>) ->
  {U32.to_nat(block_base(start,unit,pairs,pair)) == stored_number(first_position(start,unit),pairs,pair) : Nat}:
  Words{position,+base} = known
  base

def words_base(+start: U32,+unit: U32,+pairs: U32,+pair: U32,+cap: U32,+room: Room<start,unit,pairs,pair,cap>) ->
  {U32.to_nat(block_base(start,unit,pairs,pair)) == stored_number(first_position(start,unit),pairs,pair) : Nat}:
  known_base(start,unit,pairs,pair,words(start,unit,pairs,pair,cap,room))

# The word of each row of a block is its number.
def row_number(+row: Nat,+start: U32,+unit: U32,+pairs: U32,+pair: U32,+cap: U32,+room: Room<start,unit,pairs,pair,cap>,+small: {Nat.is_le(row,3n) == True{} : Bool}) ->
  {U32.to_nat(row_offset(row,block_base(start,unit,pairs,pair),pairs)) == stored_number(Nat.add(first_position(start,unit),row),pairs,pair) : Nat}:
  match row:
    case 0n:
      %Equal.sym(Nat,Nat.add(first_position(start,unit),0n),first_position(start,unit),Addition.zero_right(first_position(start,unit))) :
        {U32.to_nat(block_base(start,unit,pairs,pair)) == stored_number(_,pairs,pair) : Nat}
      Equal.trans(Nat,U32.to_nat(block_base(start,unit,pairs,pair)),stored_number(first_position(start,unit),pairs,pair),stored_number(first_position(start,unit),pairs,pair),
        words_base(start,unit,pairs,pair,cap,room),{==})
    case 1n+ +a:
      match a:
        case 0n:
          added_number(start,unit,pairs,pair,cap,room,1n,small,pairs,Equal.sym(Nat,Nat.add(U32.to_nat(pairs),0n),U32.to_nat(pairs),Addition.zero_right(U32.to_nat(pairs))),
            words(start,unit,pairs,pair,cap,room))
        case 1n+ +b:
          match b:
            case 0n:
              added_number(start,unit,pairs,pair,cap,room,2n,small,(2 * pairs : U32),
                Multiplication.bounded_multiply(2,pairs,cap,room_cap(start,unit,pairs,pair,cap,room),pairs_room(start,unit,pairs,pair,cap,room,2n,{==})),
                words(start,unit,pairs,pair,cap,room))
            case 1n+ +c:
              match c:
                case 0n:
                  added_number(start,unit,pairs,pair,cap,room,3n,small,(3 * pairs : U32),
                    Multiplication.bounded_multiply(3,pairs,cap,room_cap(start,unit,pairs,pair,cap,room),pairs_room(start,unit,pairs,pair,cap,room,3n,{==})),
                    words(start,unit,pairs,pair,cap,room))
                case 1n+d:
                  Empty.absurd({U32.to_nat(row_offset(4n+d,block_base(start,unit,pairs,pair),pairs)) == stored_number(Nat.add(first_position(start,unit),4n+d),pairs,pair) : Nat},
                    WordEncoding.false_true(small))

# ---- What every store of a run is ----

# A run and the trees its jobs read.
type Target is Data:
  Target{matrix: Gemm.PointMatrix,readers: PointJobs.Readers}

def target_start(target: Target) -> U32:
  Target{matrix,readers} = target
  Gemm.PointMatrix{Gemm.TileMatrix{channels,positions,inner,start,output_positions},cell,stride} = matrix
  start

def target_end(target: Target) -> U32:
  Target{matrix,readers} = target
  Gemm.PointMatrix{Gemm.TileMatrix{channels,positions,inner,start,output_positions},cell,stride} = matrix
  output_positions

def target_pairs(target: Target) -> U32:
  Target{matrix,readers} = target
  Gemm.PointMatrix{Gemm.TileMatrix{+channels,positions,inner,start,output_positions},cell,stride} = matrix
  Gemm.pairs_of(channels)

def target_groups(target: Target) -> U32:
  Target{matrix,readers} = target
  Gemm.PointMatrix{Gemm.TileMatrix{channels,positions,inner,start,output_positions},cell,stride} = matrix
  Gemm.blocks(channels)

# The biases of a pair: of its two groups, the lower group again for a
# missing upper one.
def biases(+bias: Tree.Tree<Tile.Tile8>,+groups: U32,+pair: U32) -> Tile.Tile16:
  Tile.Tiles{Tree.read(Tile.Tile8,bias,(pair * 2 : U32)),Tree.read(Tile.Tile8,bias,U32.min((pair * 2 + 1 : U32),(groups - 1 : U32)))}

# The value the job of a pair and a unit stores at row row.
def stored_value_by(~accumulate: PointJobs.Accumulate(),+target: Target,+pair: U32,+unit: U32,+row: Nat) -> Tile.Tile16:
  Target{+matrix,+readers} = target
  PointJobs.Readers{+bias,+cells,+pairs} = readers
  Tile.plus_pair(PointSpec.row_of(row,accumulate(matrix,Gemm.Place{pair,unit},cells,pairs)),biases(bias,target_groups(Target{matrix,readers}),pair))

def stored_value(+target: Target,+pair: U32,+unit: U32,+row: Nat) -> Tile.Tile16:
  stored_value_by(~PointJobs.job_accumulators_with(~Gemm.row_times,~PointJobs.whole_inner),target,pair,unit,row)

def target_matrix(target: Target) -> Gemm.PointMatrix:
  Target{matrix,readers} = target
  matrix

def target_readers(target: Target) -> PointJobs.Readers:
  Target{matrix,readers} = target
  readers

def target_units(target: Target) -> U32:
  Target{matrix,readers} = target
  Gemm.PointMatrix{Gemm.TileMatrix{channels,+positions,inner,start,output_positions},cell,stride} = matrix
  ((positions + 3) / 4 : U32)

def target_once(target: Target) -> Bool:
  Target{matrix,readers} = target
  Gemm.PointMatrix{Gemm.TileMatrix{+channels,positions,+inner,start,output_positions},cell,stride} = matrix
  Gemm.streamed(Gemm.blocks(channels),inner)

# ---- A checked run ----

# What a run needs of its words: the job numbers, the pairs and units
# doubled and the stores of the last unit stay below cap, and the units
# start before the end and reach it.
type Run<-target: Target,-cap: U32> is Data:
  Run{cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},positive: {Nat.is_lt(0n,U32.to_nat(target_pairs(target))) == True{} : Bool},
    double_pairs: {Nat.is_le(Nat.double(U32.to_nat(target_pairs(target))),U32.to_nat(cap)) == True{} : Bool},
    double_units: {Nat.is_le(Nat.double(U32.to_nat(target_units(target))),U32.to_nat(cap)) == True{} : Bool},
    jobs_room: {Nat.is_le(Nat.mul(U32.to_nat(target_units(target)),U32.to_nat(target_pairs(target))),U32.to_nat(cap)) == True{} : Bool},
    end_room: {Nat.is_le(Nat.add(Nat.mul(Nat.add(U32.to_nat(target_end(target)),3n),U32.to_nat(target_pairs(target))),U32.to_nat(target_pairs(target))),U32.to_nat(cap)) == True{} : Bool},
    last_unit: {Nat.is_le(Nat.add(U32.to_nat(target_start(target)),Nat.mul(U32.to_nat(target_units(target)),4n)),Nat.add(4n,U32.to_nat(target_end(target)))) == True{} : Bool},
    reach: {Nat.is_le(U32.to_nat(target_end(target)),Nat.add(U32.to_nat(target_start(target)),Nat.mul(U32.to_nat(target_units(target)),4n))) == True{} : Bool}}

def run_cap(+target: Target,+cap: U32,+run: Run<target,cap>) -> {U32.is_le(cap,2147483648) == True{} : Bool}:
  Run{+cap_ok,positive,double_pairs,double_units,jobs_room,end_room,last_unit,reach} = run
  cap_ok

def run_jobs(+target: Target,+cap: U32,+run: Run<target,cap>) ->
  {Nat.is_le(Nat.mul(U32.to_nat(target_units(target)),U32.to_nat(target_pairs(target))),U32.to_nat(cap)) == True{} : Bool}:
  Run{cap_ok,positive,double_pairs,double_units,+jobs_room,end_room,last_unit,reach} = run
  jobs_room

def run_reach(+target: Target,+cap: U32,+run: Run<target,cap>) ->
  {Nat.is_le(U32.to_nat(target_end(target)),Nat.add(U32.to_nat(target_start(target)),Nat.mul(U32.to_nat(target_units(target)),4n))) == True{} : Bool}:
  Run{cap_ok,positive,double_pairs,double_units,jobs_room,end_room,last_unit,+reach} = run
  reach

# A run with its words checked.
type Checked is Data:
  Checked{target: Target,cap: U32,run: Run<target,cap>}

def checked_target(+checked: Checked) -> Target:
  Checked{target,cap,run} = checked
  target

def checked_cap(+checked: Checked) -> U32:
  Checked{target,cap,run} = checked
  cap

def checked_layout(+checked: Checked) -> Checked:
  checked

def checked_jobs(+checked: Checked) -> Nat:
  Checked{+target,cap,run} = checked
  Nat.mul(U32.to_nat(target_units(target)),U32.to_nat(target_pairs(target)))

def checked_cap_ok(+checked: Checked) -> {U32.is_le(checked_cap(checked),2147483648) == True{} : Bool}:
  Checked{+target,+cap,+run} = checked
  run_cap(target,cap,run)

def checked_positive(+checked: Checked) -> {Nat.is_lt(0n,U32.to_nat(target_pairs(checked_target(checked)))) == True{} : Bool}:
  Checked{+target,+cap,+run} = checked
  Run{cap_ok,+positive,double_pairs,double_units,jobs_room,end_room,last_unit,reach} = run
  positive

# The stores of a run lie below the end times the pairs.
def checked_bound(+checked: Checked) -> Nat:
  Nat.mul(U32.to_nat(target_end(checked_target(checked))),U32.to_nat(target_pairs(checked_target(checked))))

def bound_room(+checked: Checked) -> {Nat.is_le(checked_bound(checked),U32.to_nat(checked_cap(checked))) == True{} : Bool}:
  Checked{+target,+cap,+run} = checked
  Run{cap_ok,positive,double_pairs,double_units,jobs_room,+end_room,last_unit,reach} = run
  +e = U32.to_nat(target_end(target))
  +pn = U32.to_nat(target_pairs(target))
  Order.transitive(Nat.mul(e,pn),Nat.add(Nat.mul(Nat.add(e,3n),pn),pn),U32.to_nat(cap),
    Order.transitive(Nat.mul(e,pn),Nat.mul(Nat.add(e,3n),pn),Nat.add(Nat.mul(Nat.add(e,3n),pn),pn),
      RowMajor.mul_le(e,Nat.add(e,3n),pn,Order.le_add(e,3n)),Order.le_add(Nat.mul(Nat.add(e,3n),pn),pn)),
    end_room)

# ---- What every store of a run is ----

# A store of a run: row row of the block of a pair and a unit.
type Spot is Data:
  Spot{pair: Nat,unit: Nat,row: Nat}

def spot_position(+checked: Checked,+unit: Nat,+row: Nat) -> Nat:
  Nat.add(Nat.add(U32.to_nat(target_start(checked_target(checked))),Nat.mul(unit,4n)),row)

# A spot of a run: its pair is a pair of the run, its row a row of a block
# and its position lies before the end.
def Placed(+checked: Checked,+spot: Spot) -> Data:
  Spot{+pair,+unit,+row} = spot
  Gemm.Both<{Nat.is_lt(pair,U32.to_nat(target_pairs(checked_target(checked)))) == True{} : Bool},
    Gemm.Both<{Nat.is_le(row,3n) == True{} : Bool},{Nat.is_lt(spot_position(checked,unit,row),U32.to_nat(target_end(checked_target(checked)))) == True{} : Bool}>>

def spot_number(+checked: Checked,+spot: Spot) -> Nat:
  Spot{+pair,+unit,+row} = spot
  Nat.add(Nat.mul(spot_position(checked,unit,row),U32.to_nat(target_pairs(checked_target(checked)))),pair)

def spot_word(+checked: Checked,+spot: Spot) -> U32:
  U32.from_nat(spot_number(checked,spot))

def spot_wanted(~accumulate: PointJobs.Accumulate(),+checked: Checked,+spot: Spot) -> Tile.Tile16:
  Spot{+pair,+unit,+row} = spot
  stored_value_by(~accumulate,checked_target(checked),U32.from_nat(pair),U32.from_nat(unit),row)

# A put stores what it is given when it stores.
def put_every(~accumulate: PointJobs.Accumulate(),+checked: Checked,+inside: Bool,+offset: U32,+value: Tile.Tile16,
  facts: Boolean.When(inside,GemmJobs.Located(~Tile.Tile16,~Checked,~Spot,~Placed,~spot_word,~spot_wanted(~accumulate),checked,offset,value))) ->
  Every.Every(~Tile.Tile16,~Checked,~GemmJobs.Located(~Tile.Tile16,~Checked,~Spot,~Placed,~spot_word,~spot_wanted(~accumulate)),checked,GemmLaws.put(~Tile.Tile16,inside,offset,value)):
  match inside:
    case True{}: Gemm.Both{facts,Unit{}}
    case False{}: Unit{}

# A row below the rows of a block lies before the end.
def row_before_end(+row: Nat,+first: Nat,+row_word: U32,+position: U32,+end: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +row_exact: {U32.to_nat(row_word) == row : Nat},+position_exact: {U32.to_nat(position) == first : Nat},
  +ordered: {Nat.is_le(first,U32.to_nat(end)) == True{} : Bool},+end_room: {Nat.is_le(U32.to_nat(end),U32.to_nat(cap)) == True{} : Bool},
  inside: {U32.is_lt(row_word,U32.min(4,(end - position : U32))) == True{} : Bool}) -> {Nat.is_lt(Nat.add(first,row),U32.to_nat(end)) == True{} : Bool}:
  +last = U32.to_nat(end)
  +gap = (end - position : U32)
  +bound = U32.min(4,gap)
  +below = {
    %row_exact : {Nat.is_lt(_,U32.to_nat(bound)) == True{} : Bool}
    %Comparison.lt(row_word,bound) : {_ == True{} : Bool}
    inside : {Nat.is_lt(row,U32.to_nat(bound)) == True{} : Bool}}
  +within = {
    %Comparison.le(bound,gap) : {_ == True{} : Bool}
    U32Extrema.minimum_right(4,gap) : {Nat.is_le(U32.to_nat(bound),U32.to_nat(gap)) == True{} : Bool}}
  +gap_exact = Equal.trans(Nat,U32.to_nat(gap),Nat.sub(last,U32.to_nat(position)),Nat.sub(last,first),
    Arithmetic.subtract(end,position,cap,cap_ok,
      %Equal.sym(Nat,U32.to_nat(position),first,position_exact) : {Nat.is_le(_,last) == True{} : Bool}
      ordered,
      Order.transitive(Nat.sub(last,U32.to_nat(position)),last,U32.to_nat(cap),Order.sub_le(last,U32.to_nat(position)),end_room)),
    Equal.cong(Nat,Nat,value => Nat.sub(last,value),U32.to_nat(position),first,position_exact))
  +reach = {
    %gap_exact : {Nat.is_le(U32.to_nat(bound),_) == True{} : Bool}
    within : {Nat.is_le(U32.to_nat(bound),Nat.sub(last,first)) == True{} : Bool}}
  %Order.restore_shift(first,last,ordered) : {Nat.is_lt(Nat.add(first,row),_) == True{} : Bool}
  Order.offset_lt(first,row,Nat.sub(last,first),Order.lt_le(row,U32.to_nat(bound),Nat.sub(last,first),below,reach))

def known_position(+start: U32,+unit: U32,+pairs: U32,+pair: U32,+known: Words<start,unit,pairs,pair>) ->
  {U32.to_nat((start + unit * 4 : U32)) == first_position(start,unit) : Nat}:
  Words{+position,base} = known
  position

def room_last(+start: U32,+unit: U32,+pairs: U32,+pair: U32,+cap: U32,+room: Room<start,unit,pairs,pair,cap>) ->
  {Nat.is_le(stored_number(Nat.add(first_position(start,unit),3n),pairs,pair),U32.to_nat(cap)) == True{} : Bool}:
  Room{cap_ok,positive,+last} = room
  last

# What a block of a pair and a unit needs: its words stay below cap, its
# pair is a pair of the run and its first position is not past the end.
type Block<-target: Target,-pair: U32,-unit: U32,-cap: U32> is Data:
  Block{room: Room<target_start(target),unit,target_pairs(target),pair,cap>,pair_inside: {Nat.is_lt(U32.to_nat(pair),U32.to_nat(target_pairs(target))) == True{} : Bool},
    ordered: {Nat.is_le(first_position(target_start(target),unit),U32.to_nat(target_end(target))) == True{} : Bool},
    end_room: {Nat.is_le(U32.to_nat(target_end(target)),U32.to_nat(cap)) == True{} : Bool}}

def block_room(+target: Target,+pair: U32,+unit: U32,+cap: U32,+block: Block<target,pair,unit,cap>) -> Room<target_start(target),unit,target_pairs(target),pair,cap>:
  Block{+room,pair_inside,ordered,end_room} = block
  room

def block_rows(+target: Target,+unit: U32) -> U32:
  U32.min(4,(target_end(target) - (target_start(target) + unit * 4 : U32) : U32))

# A stored pair and unit name the same value as their numbers.
def word_value(~accumulate: PointJobs.Accumulate(),+target: Target,+pair: U32,+unit: U32,+row: Nat) ->
  {stored_value_by(~accumulate,target,pair,unit,row) == stored_value_by(~accumulate,target,U32.from_nat(U32.to_nat(pair)),U32.from_nat(U32.to_nat(unit)),row) : Tile.Tile16}:
  %Equal.sym(U32,U32.from_nat(U32.to_nat(pair)),pair,Arithmetic.native_roundtrip(pair)) :
    {stored_value_by(~accumulate,target,pair,unit,row) == stored_value_by(~accumulate,target,_,U32.from_nat(U32.to_nat(unit)),row) : Tile.Tile16}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(unit)),unit,Arithmetic.native_roundtrip(unit)) :
    {stored_value_by(~accumulate,target,pair,unit,row) == stored_value_by(~accumulate,target,pair,_,row) : Tile.Tile16}
  {==}

# Row row of a block is a store of the run when it is below the block's rows.
def row_holds(~accumulate: PointJobs.Accumulate(),+checked: Checked,+pair: U32,+unit: U32,+block: Block<checked_target(checked),pair,unit,checked_cap(checked)>,+row: Nat,+row_word: U32,
  +small: {Nat.is_le(row,3n) == True{} : Bool},+row_exact: {U32.to_nat(row_word) == row : Nat},+inside: Bool,
  +inside_exact: {U32.is_lt(row_word,block_rows(checked_target(checked),unit)) == inside : Bool}) ->
  Boolean.When(inside,GemmJobs.Located(~Tile.Tile16,~Checked,~Spot,~Placed,~spot_word,~spot_wanted(~accumulate),checked,row_offset(row,block_base(target_start(checked_target(checked)),unit,target_pairs(checked_target(checked)),pair),target_pairs(checked_target(checked))),
    stored_value_by(~accumulate,checked_target(checked),pair,unit,row))):
  Block{+room,+pair_inside,+ordered,+end_room} = block
  match inside:
    case False{}: Unit{}
    case True{}:
      +target = checked_target(checked)
      +start = target_start(target)
      +pairs = target_pairs(target)
      +cap = checked_cap(checked)
      +cap_ok = room_cap(start,unit,pairs,pair,cap,room)
      +first = first_position(start,unit)
      +number = stored_number(Nat.add(first,row),pairs,pair)
      +exact = Bounds.roundtrip(number,cap,cap_ok,row_room(first,row,pairs,pair,cap,small,room_last(start,unit,pairs,pair,cap,room)))
      +offset = row_offset(row,block_base(start,unit,pairs,pair),pairs)
      GemmJobs.Keyed{Spot{U32.to_nat(pair),U32.to_nat(unit),row},
        Gemm.Both{pair_inside,Gemm.Both{small,
          row_before_end(row,first,row_word,(start + unit * 4 : U32),target_end(target),cap,cap_ok,row_exact,
            known_position(start,unit,pairs,pair,words(start,unit,pairs,pair,cap,room)),ordered,end_room,inside_exact)}},
        Arithmetic.to_nat_injective(offset,U32.from_nat(number),
          Equal.trans(Nat,U32.to_nat(offset),number,U32.to_nat(U32.from_nat(number)),row_number(row,start,unit,pairs,pair,cap,room,small),
            Equal.sym(Nat,U32.to_nat(U32.from_nat(number)),number,exact))),
        word_value(~accumulate,target,pair,unit,row)}

# Every store of the block of a pair and a unit is a store of the run.
def block_every(~accumulate: PointJobs.Accumulate(),+checked: Checked,+pair: U32,+unit: U32,+block: Block<checked_target(checked),pair,unit,checked_cap(checked)>) ->
  Every.Every(~Tile.Tile16,~Checked,~GemmJobs.Located(~Tile.Tile16,~Checked,~Spot,~Placed,~spot_word,~spot_wanted(~accumulate)),checked,
    PointJobs.place_writes_by(~accumulate,target_matrix(checked_target(checked)),Gemm.Place{pair,unit},target_readers(checked_target(checked)))):
  Checked{+target,+cap,+run} = checked
  Target{+matrix,+readers} = target
  Gemm.PointMatrix{Gemm.TileMatrix{+channels,+positions,+inner,+start,+output_positions},+cell,+stride} = matrix
  PointJobs.Readers{+bias,+cells,+pair_tiles} = readers
  +goal = {Target{Gemm.PointMatrix{Gemm.TileMatrix{channels,positions,inner,start,output_positions},cell,stride},PointJobs.Readers{bias,cells,pair_tiles}} : Target}
  +whole = {Checked{goal,cap,run} : Checked}
  +pairs = Gemm.pairs_of(channels)
  +base = block_base(start,unit,pairs,pair)
  +rows = block_rows(goal,unit)
  +v0 = stored_value_by(~accumulate,goal,pair,unit,0n)
  +v1 = stored_value_by(~accumulate,goal,pair,unit,1n)
  +v2 = stored_value_by(~accumulate,goal,pair,unit,2n)
  +v3 = stored_value_by(~accumulate,goal,pair,unit,3n)
  +p0 = GemmLaws.put(~Tile.Tile16,U32.is_lt(0,rows),base,v0)
  +p1 = GemmLaws.put(~Tile.Tile16,U32.is_lt(1,rows),(base + pairs : U32),v1)
  +p2 = GemmLaws.put(~Tile.Tile16,U32.is_lt(2,rows),(base + 2 * pairs : U32),v2)
  +p3 = GemmLaws.put(~Tile.Tile16,U32.is_lt(3,rows),(base + 3 * pairs : U32),v3)
  +w1 = List.append(&2,Writes.Assignment<Tile.Tile16>,p0,p1)
  +w2 = List.append(&2,Writes.Assignment<Tile.Tile16>,w1,p2)
  +e0 = put_every(~accumulate,whole,U32.is_lt(0,rows),base,v0,row_holds(~accumulate,whole,pair,unit,block,0n,0,{==},{==},U32.is_lt(0,rows),{==}))
  +e1 = put_every(~accumulate,whole,U32.is_lt(1,rows),(base + pairs : U32),v1,row_holds(~accumulate,whole,pair,unit,block,1n,1,{==},{==},U32.is_lt(1,rows),{==}))
  +e2 = put_every(~accumulate,whole,U32.is_lt(2,rows),(base + 2 * pairs : U32),v2,row_holds(~accumulate,whole,pair,unit,block,2n,2,{==},{==},U32.is_lt(2,rows),{==}))
  +e3 = put_every(~accumulate,whole,U32.is_lt(3,rows),(base + 3 * pairs : U32),v3,row_holds(~accumulate,whole,pair,unit,block,3n,3,{==},{==},U32.is_lt(3,rows),{==}))
  Every.appended(~Tile.Tile16,~Checked,~GemmJobs.Located(~Tile.Tile16,~Checked,~Spot,~Placed,~spot_word,~spot_wanted(~accumulate)),whole,w2,p3,
    Every.appended(~Tile.Tile16,~Checked,~GemmJobs.Located(~Tile.Tile16,~Checked,~Spot,~Placed,~spot_word,~spot_wanted(~accumulate)),whole,w1,p2,
      Every.appended(~Tile.Tile16,~Checked,~GemmJobs.Located(~Tile.Tile16,~Checked,~Spot,~Placed,~spot_word,~spot_wanted(~accumulate)),whole,p0,p1,e0,e1),e2),e3)

# ---- The jobs of a run ----

def job_pair(+target: Target,+index: Nat) -> U32:
  Gemm.job_group(target_once(target),U32.from_nat(index),target_pairs(target),target_units(target))

def job_unit(+target: Target,+index: Nat) -> U32:
  Gemm.job_unit(target_once(target),U32.from_nat(index),target_pairs(target),target_units(target))

def add_le_both(+left: Nat,+right: Nat,+low: Nat,+high: Nat,first: {Nat.is_le(left,right) == True{} : Bool},second: {Nat.is_le(low,high) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(left,low),Nat.add(right,high)) == True{} : Bool}:
  Order.transitive(Nat.add(left,low),Nat.add(right,low),Nat.add(right,high),right_offset_le(left,right,low,first),Order.offset_le(right,low,high,second))

# A unit below the units starts before the end.
def unit_ordered(+start: Nat,+end: Nat,+units: Nat,+unit: Nat,+inside: {Nat.is_lt(unit,units) == True{} : Bool},
  +last: {Nat.is_le(Nat.add(start,Nat.mul(units,4n)),Nat.add(4n,end)) == True{} : Bool}) -> {Nat.is_le(Nat.add(start,Nat.mul(unit,4n)),end) == True{} : Bool}:
  Order.cancel_offset_le(4n,Nat.add(start,Nat.mul(unit,4n)),end,
    %RowMajor.add_swap(start,4n,Nat.mul(unit,4n)) : {Nat.is_le(_,Nat.add(4n,end)) == True{} : Bool}
    Order.transitive(Nat.add(start,Nat.mul(1n+unit,4n)),Nat.add(start,Nat.mul(units,4n)),Nat.add(4n,end),
      Order.offset_le(start,Nat.mul(1n+unit,4n),Nat.mul(units,4n),RowMajor.mul_le(1n+unit,units,4n,Order.strict_follow(unit,units,inside))),last))

# The block of a job number named by its split.
def split_block(+target: Target,+cap: U32,+run: Run<target,cap>,+index: Nat,
  +split: GemmReadback.Split<target_once(target),index,U32.to_nat(target_pairs(target)),U32.to_nat(target_units(target))>) ->
  Block<target,job_pair(target,index),job_unit(target,index),cap>:
  Run{+cap_ok,+positive,+double_pairs,+double_units,+jobs_room,+end_room,+last_unit,reach} = run
  GemmReadback.Split{+pair,+unit,+pair_inside,+unit_inside,+exact} = split
  +pairs = target_pairs(target)
  +units = target_units(target)
  +pn = U32.to_nat(pairs)
  +un = U32.to_nat(units)
  +s = U32.to_nat(target_start(target))
  +e = U32.to_nat(target_end(target))
  +limit = U32.to_nat(cap)
  +small = Order.transitive(index,Nat.mul(un,pn),limit,
    Bounds.strict_le(index,Nat.mul(un,pn),%Equal.sym(Nat,index,GemmReadback.index_of(target_once(target),pair,unit,pn,un),exact) : {Nat.is_lt(_,Nat.mul(un,pn)) == True{} : Bool} GemmReadback.index_below(target_once(target),pair,unit,pn,un,pair_inside,unit_inside)),jobs_room)
  +number = Equal.trans(Nat,U32.to_nat(U32.from_nat(index)),index,GemmReadback.index_of(target_once(target),pair,unit,pn,un),Bounds.roundtrip(index,cap,cap_ok,small),exact)
  +pair_exact = GemmReadback.job_group_exact(target_once(target),U32.from_nat(index),pairs,units,pair,unit,pn,un,cap,cap_ok,number,{==},{==},pair_inside,unit_inside,double_pairs,double_units)
  +unit_exact = GemmReadback.job_unit_exact(target_once(target),U32.from_nat(index),pairs,units,pair,unit,pn,un,cap,cap_ok,number,{==},{==},pair_inside,unit_inside,double_pairs,double_units)
  +ordered = unit_ordered(s,e,un,unit,unit_inside,last_unit)
  +last = {
    %Equal.sym(Nat,U32.to_nat(job_unit(target,index)),unit,unit_exact) : {Nat.is_le(Nat.add(Nat.mul(Nat.add(Nat.add(s,Nat.mul(_,4n)),3n),pn),U32.to_nat(job_pair(target,index))),limit) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(job_pair(target,index)),pair,pair_exact) : {Nat.is_le(Nat.add(Nat.mul(Nat.add(Nat.add(s,Nat.mul(unit,4n)),3n),pn),_),limit) == True{} : Bool}
    Order.transitive(Nat.add(Nat.mul(Nat.add(Nat.add(s,Nat.mul(unit,4n)),3n),pn),pair),Nat.add(Nat.mul(Nat.add(e,3n),pn),pn),limit,
      add_le_both(Nat.mul(Nat.add(Nat.add(s,Nat.mul(unit,4n)),3n),pn),Nat.mul(Nat.add(e,3n),pn),pair,pn,
        RowMajor.mul_le(Nat.add(Nat.add(s,Nat.mul(unit,4n)),3n),Nat.add(e,3n),pn,right_offset_le(Nat.add(s,Nat.mul(unit,4n)),e,3n,ordered)),
        Bounds.strict_le(pair,pn,pair_inside)),
      end_room) :
      {Nat.is_le(stored_number(Nat.add(first_position(target_start(target),job_unit(target,index)),3n),pairs,job_pair(target,index)),limit) == True{} : Bool}}
  +end_below = Order.transitive(e,Nat.add(Nat.mul(Nat.add(e,3n),pn),pn),limit,
    Order.transitive(e,Nat.mul(Nat.add(e,3n),pn),Nat.add(Nat.mul(Nat.add(e,3n),pn),pn),
      Order.transitive(e,Nat.add(e,3n),Nat.mul(Nat.add(e,3n),pn),Order.le_add(e,3n),RowMajor.product_covers(Nat.add(e,3n),pn,positive)),
      Order.le_add(Nat.mul(Nat.add(e,3n),pn),pn)),
    end_room)
  Block{Room{cap_ok,positive,last},
    %Equal.sym(Nat,U32.to_nat(job_pair(target,index)),pair,pair_exact) : {Nat.is_lt(_,pn) == True{} : Bool} pair_inside,
    %Equal.sym(Nat,U32.to_nat(job_unit(target,index)),unit,unit_exact) : {Nat.is_le(Nat.add(s,Nat.mul(_,4n)),e) == True{} : Bool} ordered,
    end_below}

# Every store of a job of the run is a store of the run.
def job_every(~accumulate: PointJobs.Accumulate(),+checked: Checked,+index: Nat,+inside: {Nat.is_lt(index,checked_jobs(checked)) == True{} : Bool}) ->
  Every.Every(~Tile.Tile16,~Checked,~GemmJobs.Located(~Tile.Tile16,~Checked,~Spot,~Placed,~spot_word,~spot_wanted(~accumulate)),checked,
    PointJobs.step_writes_by(~accumulate,U32.from_nat(index),target_matrix(checked_target(checked)),target_readers(checked_target(checked)))):
  Checked{+target,+cap,+run} = checked
  Target{+matrix,+readers} = target
  Gemm.PointMatrix{Gemm.TileMatrix{+channels,+positions,+inner,+start,+output_positions},+cell,+stride} = matrix
  PointJobs.Readers{+bias,+cells,+pair_tiles} = readers
  +goal = {Target{Gemm.PointMatrix{Gemm.TileMatrix{channels,positions,inner,start,output_positions},cell,stride},PointJobs.Readers{bias,cells,pair_tiles}} : Target}
  +pn = U32.to_nat(target_pairs(goal))
  +un = U32.to_nat(target_units(goal))
  +small = Order.transitive(index,Nat.mul(un,pn),U32.to_nat(cap),Bounds.strict_le(index,Nat.mul(un,pn),inside),run_jobs(goal,cap,run))
  block_every(~accumulate,Checked{goal,cap,run},job_pair(goal,index),job_unit(goal,index),
    split_block(goal,cap,run,index,GemmReadback.split_of(target_once(goal),index,pn,un,cap,run_cap(goal,cap,run),small,inside)))

# ---- Every store of a run ----

# The jobs from position on, rest of them, are jobs of the run, of its
# product and readers.
type Pending<-checked: Checked,-rest: Nat,-position: Nat,-matrix: Gemm.PointMatrix,-readers: PointJobs.Readers> is Data:
  Pending{within: {Nat.is_le(Nat.add(position,rest),checked_jobs(checked)) == True{} : Bool},matrix_is: {matrix == target_matrix(checked_target(checked)) : Gemm.PointMatrix},
    readers_is: {readers == target_readers(checked_target(checked)) : PointJobs.Readers}}

def Invariant(+checked: Checked,rest: Nat,position: Nat,matrix: Gemm.PointMatrix,readers: PointJobs.Readers) -> Data:
  Pending<checked,rest,position,matrix,readers>

def pending_step(~accumulate: PointJobs.Accumulate(),+checked: Checked,+rest: Nat,+position: Nat,+matrix: Gemm.PointMatrix,+readers: PointJobs.Readers,pending: Invariant(checked,1n+rest,position,matrix,readers)) ->
  Every.Every(~Tile.Tile16,~Checked,~GemmJobs.Located(~Tile.Tile16,~Checked,~Spot,~Placed,~spot_word,~spot_wanted(~accumulate)),checked_layout(checked),
    PointJobs.step_writes_by(~accumulate,U32.from_nat(position),matrix,readers)):
  Pending{+within,+matrix_is,+readers_is} = pending
  +target = checked_target(checked)
  %Equal.sym(Gemm.PointMatrix,matrix,target_matrix(target),matrix_is) :
    Every.Every(~Tile.Tile16,~Checked,~GemmJobs.Located(~Tile.Tile16,~Checked,~Spot,~Placed,~spot_word,~spot_wanted(~accumulate)),checked,PointJobs.step_writes_by(~accumulate,U32.from_nat(position),_,readers))
  %Equal.sym(PointJobs.Readers,readers,target_readers(target),readers_is) :
    Every.Every(~Tile.Tile16,~Checked,~GemmJobs.Located(~Tile.Tile16,~Checked,~Spot,~Placed,~spot_word,~spot_wanted(~accumulate)),checked,PointJobs.step_writes_by(~accumulate,U32.from_nat(position),target_matrix(target),_))
  job_every(~accumulate,checked,position,
    Order.lt_le(position,Nat.add(position,1n+rest),checked_jobs(checked),
      %Addition.zero_right(position) : {Nat.is_lt(_,Nat.add(position,1n+rest)) == True{} : Bool}
      Order.offset_lt(position,0n,1n+rest,{==}),
      within))

def pending_kept(+checked: Checked,+rest: Nat,+position: Nat,+matrix: Gemm.PointMatrix,+readers: PointJobs.Readers,pending: Invariant(checked,1n+rest,position,matrix,readers)) ->
  Invariant(checked,rest,1n+position,matrix,PointJobs.kept(U32.from_nat(position),matrix,readers)):
  Pending{+within,+matrix_is,+readers_is} = pending
  Pending{%Addition.successor_right(position,rest) : {Nat.is_le(_,checked_jobs(checked)) == True{} : Bool} within,matrix_is,readers_is}

# The stores of a run, its jobs in order.
def checked_writes_by(~accumulate: PointJobs.Accumulate(),+checked: Checked) -> List<&2,Writes.Assignment<Tile.Tile16>>:
  PointJobs.all_writes_by(~accumulate,checked_jobs(checked),0,target_matrix(checked_target(checked)),target_readers(checked_target(checked)))

def checked_writes(+checked: Checked) -> List<&2,Writes.Assignment<Tile.Tile16>>:
  checked_writes_by(~PointJobs.job_accumulators_with(~Gemm.row_times,~PointJobs.whole_inner),checked)

# Every store of the run is a store of the run's blocks.
def all_every(~accumulate: PointJobs.Accumulate(),+checked: Checked) ->
  Every.Every(~Tile.Tile16,~Checked,~GemmJobs.Located(~Tile.Tile16,~Checked,~Spot,~Placed,~spot_word,~spot_wanted(~accumulate)),checked,checked_writes_by(~accumulate,checked)):
  Fold.every(~Tile.Tile16,~Gemm.PointMatrix,~PointJobs.Readers,~PointJobs.step_writes_by(~accumulate),~PointJobs.kept,~Checked,~Checked,~checked_layout,
    ~GemmJobs.Located(~Tile.Tile16,~Checked,~Spot,~Placed,~spot_word,~spot_wanted(~accumulate)),~Invariant,~pending_step(~accumulate),~pending_kept,
    checked,checked_jobs(checked),0n,target_matrix(checked_target(checked)),target_readers(checked_target(checked)),Pending{Order.reflexive(checked_jobs(checked)),{==},{==}})

# ---- One store at each spot ----

def spot_room(+checked: Checked,+spot: Spot,+placed: Placed(checked,spot)) -> {Nat.is_le(spot_number(checked,spot),U32.to_nat(checked_cap(checked))) == True{} : Bool}:
  Spot{+pair,+unit,+row} = spot
  Gemm.Both{+pair_inside,+rest} = placed
  Gemm.Both{row_small,+before_end} = rest
  +pn = U32.to_nat(target_pairs(checked_target(checked)))
  +e = U32.to_nat(target_end(checked_target(checked)))
  +position = spot_position(checked,unit,row)
  +number = Nat.add(Nat.mul(position,pn),pair)
  Order.transitive(number,Nat.mul(e,pn),U32.to_nat(checked_cap(checked)),
    Bounds.strict_le(number,Nat.mul(e,pn),RowMajor.radix(position,e,pair,pn,before_end,pair_inside)),bound_room(checked))

# The word of a spot lies below the bound.
def spot_below(+checked: Checked,+spot: Spot,+placed: Placed(checked,spot)) -> {Nat.is_lt(U32.to_nat(spot_word(checked,spot)),checked_bound(checked)) == True{} : Bool}:
  Spot{+pair,+unit,+row} = spot
  Gemm.Both{+pair_inside,+rest} = placed
  Gemm.Both{+row_small,+before_end} = rest
  +pn = U32.to_nat(target_pairs(checked_target(checked)))
  +e = U32.to_nat(target_end(checked_target(checked)))
  +position = spot_position(checked,unit,row)
  +number = Nat.add(Nat.mul(position,pn),pair)
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(number)),number,
    Bounds.roundtrip(number,checked_cap(checked),checked_cap_ok(checked),spot_room(checked,Spot{pair,unit,row},Gemm.Both{pair_inside,Gemm.Both{row_small,before_end}}))) :
    {Nat.is_lt(_,Nat.mul(e,pn)) == True{} : Bool}
  RowMajor.radix(position,e,pair,pn,before_end,pair_inside)

# Two stores at one number are one pair of one row of one unit.
def same_store(+start: Nat,+pairs: Nat,+positive: {Nat.is_lt(0n,pairs) == True{} : Bool},+unit: Nat,+row: Nat,+pair: Nat,+other_unit: Nat,+other_row: Nat,+other_pair: Nat,
  +row_small: {Nat.is_le(row,3n) == True{} : Bool},+other_small: {Nat.is_le(other_row,3n) == True{} : Bool},
  +pair_inside: {Nat.is_lt(pair,pairs) == True{} : Bool},+other_inside: {Nat.is_lt(other_pair,pairs) == True{} : Bool},
  +equal: {Nat.add(Nat.mul(Nat.add(Nat.add(start,Nat.mul(unit,4n)),row),pairs),pair) == Nat.add(Nat.mul(Nat.add(Nat.add(start,Nat.mul(other_unit,4n)),other_row),pairs),other_pair) : Nat}) ->
  Gemm.Both<{unit == other_unit : Nat},Gemm.Both<{row == other_row : Nat},{pair == other_pair : Nat}>>:
  +position = BinaryRadix.row_unique(Nat.add(Nat.add(start,Nat.mul(unit,4n)),row),Nat.add(Nat.add(start,Nat.mul(other_unit,4n)),other_row),pairs,pair,other_pair,
    positive,pair_inside,other_inside,equal)
  +local = Order.cancel_sum(start,Nat.add(Nat.mul(unit,4n),row),Nat.add(Nat.mul(other_unit,4n),other_row),
    %Addition.associative(start,Nat.mul(unit,4n),row) : {_ == Nat.add(start,Nat.add(Nat.mul(other_unit,4n),other_row)) : Nat}
    %Addition.associative(start,Nat.mul(other_unit,4n),other_row) : {Nat.add(Nat.add(start,Nat.mul(unit,4n)),row) == _ : Nat}
    position)
  Gemm.Both{BinaryRadix.row_unique(unit,other_unit,4n,row,other_row,{==},Order.succ_le_lt(row,4n,row_small),Order.succ_le_lt(other_row,4n,other_small),local),
    Gemm.Both{BinaryRadix.col_unique(unit,other_unit,4n,row,other_row,{==},Order.succ_le_lt(row,4n,row_small),Order.succ_le_lt(other_row,4n,other_small),local),
      BinaryRadix.col_unique(Nat.add(Nat.add(start,Nat.mul(unit,4n)),row),Nat.add(Nat.add(start,Nat.mul(other_unit,4n)),other_row),pairs,pair,other_pair,
        positive,pair_inside,other_inside,equal)}}

def same_wanted(~accumulate: PointJobs.Accumulate(),+checked: Checked,+pair: Nat,+unit: Nat,+row: Nat,+other_pair: Nat,+other_unit: Nat,+other_row: Nat,
  +same: Gemm.Both<{unit == other_unit : Nat},Gemm.Both<{row == other_row : Nat},{pair == other_pair : Nat}>>) ->
  {spot_wanted(~accumulate,checked,Spot{pair,unit,row}) == spot_wanted(~accumulate,checked,Spot{other_pair,other_unit,other_row}) : Tile.Tile16}:
  Gemm.Both{+unit_eq,+rest} = same
  Gemm.Both{+row_eq,+pair_eq} = rest
  %unit_eq : {spot_wanted(~accumulate,checked,Spot{pair,unit,row}) == spot_wanted(~accumulate,checked,Spot{other_pair,_,other_row}) : Tile.Tile16}
  %row_eq : {spot_wanted(~accumulate,checked,Spot{pair,unit,row}) == spot_wanted(~accumulate,checked,Spot{other_pair,unit,_}) : Tile.Tile16}
  %pair_eq : {spot_wanted(~accumulate,checked,Spot{pair,unit,row}) == spot_wanted(~accumulate,checked,Spot{_,unit,row}) : Tile.Tile16}
  {==}

# Two spots of a run at one word ask for one value.
def spot_same(~accumulate: PointJobs.Accumulate(),+checked: Checked,+spot: Spot,+other: Spot,+placed: Placed(checked,spot),+other_placed: Placed(checked,other),
  +equal: {spot_word(checked,spot) == spot_word(checked,other) : U32}) ->
  {spot_wanted(~accumulate,checked,spot) == spot_wanted(~accumulate,checked,other) : Tile.Tile16}:
  Spot{+pair,+unit,+row} = spot
  Spot{+other_pair,+other_unit,+other_row} = other
  Gemm.Both{+pair_inside,+rest} = placed
  Gemm.Both{+row_small,+before_end} = rest
  Gemm.Both{+other_inside,+other_rest} = other_placed
  Gemm.Both{+other_small,+other_end} = other_rest
  +cap = checked_cap(checked)
  +cap_ok = checked_cap_ok(checked)
  +key = {Spot{pair,unit,row} : Spot}
  +other_key = {Spot{other_pair,other_unit,other_row} : Spot}
  +number = spot_number(checked,key)
  +other_number = spot_number(checked,other_key)
  +numbers = Equal.trans(Nat,number,U32.to_nat(U32.from_nat(number)),other_number,
    Equal.sym(Nat,U32.to_nat(U32.from_nat(number)),number,Bounds.roundtrip(number,cap,cap_ok,spot_room(checked,key,Gemm.Both{pair_inside,Gemm.Both{row_small,before_end}}))),
    Equal.trans(Nat,U32.to_nat(U32.from_nat(number)),U32.to_nat(U32.from_nat(other_number)),other_number,
      Equal.cong(U32,Nat,word => U32.to_nat(word),U32.from_nat(number),U32.from_nat(other_number),equal),
      Bounds.roundtrip(other_number,cap,cap_ok,spot_room(checked,other_key,Gemm.Both{other_inside,Gemm.Both{other_small,other_end}}))))
  same_wanted(~accumulate,checked,pair,unit,row,other_pair,other_unit,other_row,
    same_store(U32.to_nat(target_start(checked_target(checked))),U32.to_nat(target_pairs(checked_target(checked))),checked_positive(checked),
      unit,row,pair,other_unit,other_row,other_pair,row_small,other_small,pair_inside,other_inside,numbers))

# ---- Every spot is stored ----

# A put that stores holds its offset.
def put_contains(~E: Data,inside: Bool,+offset: U32,+value: E,stored: {inside == True{} : Bool}) ->
  {WriteReadback.contains(~E,offset,GemmLaws.put(~E,inside,offset,value)) == True{} : Bool}:
  match inside:
    case True{}:
      %Equal.sym(Bool,U32.is_eq(offset,offset),True{},Arithmetic.equal_reflexive(offset)) : {_ || WriteReadback.contains(~E,offset,[]) == True{} : Bool}
      {==}
    case False{}: Empty.absurd({WriteReadback.contains(~E,offset,[]) == True{} : Bool},WordEncoding.false_true(stored))

# A row before the end is below the rows of its block.
def row_inside(+target: Target,+pair: U32,+unit: U32,+cap: U32,+block: Block<target,pair,unit,cap>,+row: Nat,+row_word: U32,
  +small: {Nat.is_le(row,3n) == True{} : Bool},+row_exact: {U32.to_nat(row_word) == row : Nat},
  +inside: {Nat.is_lt(Nat.add(first_position(target_start(target),unit),row),U32.to_nat(target_end(target))) == True{} : Bool}) ->
  {U32.is_lt(row_word,block_rows(target,unit)) == True{} : Bool}:
  Block{+room,pair_inside,+ordered,+end_room} = block
  +start = target_start(target)
  +pairs = target_pairs(target)
  +end = target_end(target)
  +position = (start + unit * 4 : U32)
  +first = first_position(start,unit)
  +last = U32.to_nat(end)
  +gap = (end - position : U32)
  +bound = U32.min(4,gap)
  +cap_ok = room_cap(start,unit,pairs,pair,cap,room)
  +position_exact = known_position(start,unit,pairs,pair,words(start,unit,pairs,pair,cap,room))
  +gap_exact = Equal.trans(Nat,U32.to_nat(gap),Nat.sub(last,U32.to_nat(position)),Nat.sub(last,first),
    Arithmetic.subtract(end,position,cap,cap_ok,
      %Equal.sym(Nat,U32.to_nat(position),first,position_exact) : {Nat.is_le(_,last) == True{} : Bool}
      ordered,
      Order.transitive(Nat.sub(last,U32.to_nat(position)),last,U32.to_nat(cap),Order.sub_le(last,U32.to_nat(position)),end_room)),
    Equal.cong(Nat,Nat,value => Nat.sub(last,value),U32.to_nat(position),first,position_exact))
  +reach = Order.cancel_offset_lt(first,row,Nat.sub(last,first),
    %Equal.sym(Nat,Nat.add(first,Nat.sub(last,first)),last,Order.restore_shift(first,last,ordered)) : {Nat.is_lt(Nat.add(first,row),_) == True{} : Bool}
    inside)
  %Equal.sym(Bool,U32.is_lt(row_word,bound),Nat.is_lt(U32.to_nat(row_word),U32.to_nat(bound)),Comparison.lt(row_word,bound)) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(row_word),row,row_exact) : {Nat.is_lt(_,U32.to_nat(bound)) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(bound),Nat.min(U32.to_nat(4),U32.to_nat(gap)),U32Extrema.minimum(4,gap)) : {Nat.is_lt(row,_) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(gap),Nat.sub(last,first),gap_exact) : {Nat.is_lt(row,Nat.min(U32.to_nat(4),_)) == True{} : Bool}
  NatExtrema.minimum_strict_lower(U32.to_nat(4),Nat.sub(last,first),row,Order.succ_le_lt(row,4n,small),reach)

# The stores of a block hold each row before the end.
def stores_cover(+target: Target,+pair: U32,+unit: U32,+cap: U32,+block: Block<target,pair,unit,cap>,+biases: Tile.Tile16,+accumulators: Gemm.Quad,+row: Nat,
  +small: {Nat.is_le(row,3n) == True{} : Bool},
  +inside: {Nat.is_lt(Nat.add(first_position(target_start(target),unit),row),U32.to_nat(target_end(target))) == True{} : Bool}) ->
  {WriteReadback.contains(~Tile.Tile16,row_offset(row,block_base(target_start(target),unit,target_pairs(target),pair),target_pairs(target)),
    PointJobs.stores(block_base(target_start(target),unit,target_pairs(target),pair),target_pairs(target),block_rows(target,unit),biases,accumulators)) == True{} : Bool}:
  match row:
    case 0n:
      +pairs = target_pairs(target)
      +base = block_base(target_start(target),unit,pairs,pair)
      +rows = block_rows(target,unit)
      +v0 = Tile.plus_pair(Gemm.quad0(accumulators),biases)
      +p0 = GemmLaws.put(~Tile.Tile16,U32.is_lt(0,rows),base,v0)
      +p1 = GemmLaws.put(~Tile.Tile16,U32.is_lt(1,rows),(base + pairs : U32),Tile.plus_pair(Gemm.quad1(accumulators),biases))
      +p2 = GemmLaws.put(~Tile.Tile16,U32.is_lt(2,rows),(base + 2 * pairs : U32),Tile.plus_pair(Gemm.quad2(accumulators),biases))
      +p3 = GemmLaws.put(~Tile.Tile16,U32.is_lt(3,rows),(base + 3 * pairs : U32),Tile.plus_pair(Gemm.quad3(accumulators),biases))
      +w1 = List.append(&2,Writes.Assignment<Tile.Tile16>,p0,p1)
      +w2 = List.append(&2,Writes.Assignment<Tile.Tile16>,w1,p2)
      WriteReadback.covered_left(~Tile.Tile16,base,w2,p3,WriteReadback.covered_left(~Tile.Tile16,base,w1,p2,WriteReadback.covered_left(~Tile.Tile16,base,p0,p1,
        put_contains(~Tile.Tile16,U32.is_lt(0,rows),base,v0,row_inside(target,pair,unit,cap,block,0n,0,small,{==},inside)))))
    case 1n+ +a:
      match a:
        case 0n:
          +pairs = target_pairs(target)
          +base = block_base(target_start(target),unit,pairs,pair)
          +rows = block_rows(target,unit)
          +q = (base + pairs : U32)
          +v1 = Tile.plus_pair(Gemm.quad1(accumulators),biases)
          +p0 = GemmLaws.put(~Tile.Tile16,U32.is_lt(0,rows),base,Tile.plus_pair(Gemm.quad0(accumulators),biases))
          +p1 = GemmLaws.put(~Tile.Tile16,U32.is_lt(1,rows),q,v1)
          +p2 = GemmLaws.put(~Tile.Tile16,U32.is_lt(2,rows),(base + 2 * pairs : U32),Tile.plus_pair(Gemm.quad2(accumulators),biases))
          +p3 = GemmLaws.put(~Tile.Tile16,U32.is_lt(3,rows),(base + 3 * pairs : U32),Tile.plus_pair(Gemm.quad3(accumulators),biases))
          +w1 = List.append(&2,Writes.Assignment<Tile.Tile16>,p0,p1)
          +w2 = List.append(&2,Writes.Assignment<Tile.Tile16>,w1,p2)
          WriteReadback.covered_left(~Tile.Tile16,q,w2,p3,WriteReadback.covered_left(~Tile.Tile16,q,w1,p2,WriteReadback.covered_right(~Tile.Tile16,q,p0,p1,
            put_contains(~Tile.Tile16,U32.is_lt(1,rows),q,v1,row_inside(target,pair,unit,cap,block,1n,1,small,{==},inside)))))
        case 1n+ +b:
          match b:
            case 0n:
              +pairs = target_pairs(target)
              +base = block_base(target_start(target),unit,pairs,pair)
              +rows = block_rows(target,unit)
              +q = (base + 2 * pairs : U32)
              +v2 = Tile.plus_pair(Gemm.quad2(accumulators),biases)
              +p0 = GemmLaws.put(~Tile.Tile16,U32.is_lt(0,rows),base,Tile.plus_pair(Gemm.quad0(accumulators),biases))
              +p1 = GemmLaws.put(~Tile.Tile16,U32.is_lt(1,rows),(base + pairs : U32),Tile.plus_pair(Gemm.quad1(accumulators),biases))
              +p2 = GemmLaws.put(~Tile.Tile16,U32.is_lt(2,rows),q,v2)
              +p3 = GemmLaws.put(~Tile.Tile16,U32.is_lt(3,rows),(base + 3 * pairs : U32),Tile.plus_pair(Gemm.quad3(accumulators),biases))
              +w1 = List.append(&2,Writes.Assignment<Tile.Tile16>,p0,p1)
              +w2 = List.append(&2,Writes.Assignment<Tile.Tile16>,w1,p2)
              WriteReadback.covered_left(~Tile.Tile16,q,w2,p3,WriteReadback.covered_right(~Tile.Tile16,q,w1,p2,
                put_contains(~Tile.Tile16,U32.is_lt(2,rows),q,v2,row_inside(target,pair,unit,cap,block,2n,2,small,{==},inside))))
            case 1n+ +c:
              match c:
                case 0n:
                  +pairs = target_pairs(target)
                  +base = block_base(target_start(target),unit,pairs,pair)
                  +rows = block_rows(target,unit)
                  +q = (base + 3 * pairs : U32)
                  +v3 = Tile.plus_pair(Gemm.quad3(accumulators),biases)
                  +p0 = GemmLaws.put(~Tile.Tile16,U32.is_lt(0,rows),base,Tile.plus_pair(Gemm.quad0(accumulators),biases))
                  +p1 = GemmLaws.put(~Tile.Tile16,U32.is_lt(1,rows),(base + pairs : U32),Tile.plus_pair(Gemm.quad1(accumulators),biases))
                  +p2 = GemmLaws.put(~Tile.Tile16,U32.is_lt(2,rows),(base + 2 * pairs : U32),Tile.plus_pair(Gemm.quad2(accumulators),biases))
                  +w1 = List.append(&2,Writes.Assignment<Tile.Tile16>,p0,p1)
                  +w2 = List.append(&2,Writes.Assignment<Tile.Tile16>,w1,p2)
                  WriteReadback.covered_right(~Tile.Tile16,q,w2,GemmLaws.put(~Tile.Tile16,U32.is_lt(3,rows),q,v3),
                    put_contains(~Tile.Tile16,U32.is_lt(3,rows),q,v3,row_inside(target,pair,unit,cap,block,3n,3,small,{==},inside)))
                case 1n+d:
                  Empty.absurd({WriteReadback.contains(~Tile.Tile16,row_offset(4n+d,block_base(target_start(target),unit,target_pairs(target),pair),target_pairs(target)),
                    PointJobs.stores(block_base(target_start(target),unit,target_pairs(target),pair),target_pairs(target),block_rows(target,unit),biases,accumulators)) == True{} : Bool},
                    WordEncoding.false_true(small))

# The writes of a block hold each of its rows before the end.
def block_contains(~accumulate: PointJobs.Accumulate(),+target: Target,+pair: U32,+unit: U32,+cap: U32,+block: Block<target,pair,unit,cap>,+row: Nat,+small: {Nat.is_le(row,3n) == True{} : Bool},
  +inside: {Nat.is_lt(Nat.add(first_position(target_start(target),unit),row),U32.to_nat(target_end(target))) == True{} : Bool}) ->
  {WriteReadback.contains(~Tile.Tile16,row_offset(row,block_base(target_start(target),unit,target_pairs(target),pair),target_pairs(target)),
    PointJobs.place_writes_by(~accumulate,target_matrix(target),Gemm.Place{pair,unit},target_readers(target))) == True{} : Bool}:
  Target{+matrix,+readers} = target
  Gemm.PointMatrix{Gemm.TileMatrix{+channels,+positions,+inner,+start,+output_positions},+cell,+stride} = matrix
  PointJobs.Readers{+bias,+cells,+pair_tiles} = readers
  +goal = {Target{Gemm.PointMatrix{Gemm.TileMatrix{channels,positions,inner,start,output_positions},cell,stride},PointJobs.Readers{bias,cells,pair_tiles}} : Target}
  stores_cover(goal,pair,unit,cap,block,biases(bias,Gemm.blocks(channels),pair),
    accumulate(Gemm.PointMatrix{Gemm.TileMatrix{channels,positions,inner,start,output_positions},cell,stride},Gemm.Place{pair,unit},cells,pair_tiles),row,small,inside)

# A unit with a row before the end, which the units reach, is a unit of
# the run.
def unit_below(+start: Nat,+end: Nat,+units: Nat,+unit: Nat,+row: Nat,+before: {Nat.is_lt(Nat.add(Nat.add(start,Nat.mul(unit,4n)),row),end) == True{} : Bool},
  +reach: {Nat.is_le(end,Nat.add(start,Nat.mul(units,4n))) == True{} : Bool}) -> {Nat.is_lt(unit,units) == True{} : Bool}:
  +local = Order.cancel_offset_lt(start,Nat.add(Nat.mul(unit,4n),row),Nat.mul(units,4n),
    %Addition.associative(start,Nat.mul(unit,4n),row) : {Nat.is_lt(_,Nat.add(start,Nat.mul(units,4n))) == True{} : Bool}
    Order.lt_le(Nat.add(Nat.add(start,Nat.mul(unit,4n)),row),end,Nat.add(start,Nat.mul(units,4n)),before,reach))
  RowMajor.cancel_multiply_lt(unit,units,4n,Order.le_lt_trans(Nat.mul(unit,4n),Nat.add(Nat.mul(unit,4n),row),Nat.mul(units,4n),Order.le_add(Nat.mul(unit,4n),row),local))

# The job of a spot.
def spot_index(+checked: Checked,+spot: Spot) -> Nat:
  Spot{+pair,+unit,row} = spot
  +target = checked_target(checked)
  GemmReadback.index_of(target_once(target),pair,unit,U32.to_nat(target_pairs(target)),U32.to_nat(target_units(target)))

def spot_inside(+checked: Checked,+spot: Spot,+placed: Placed(checked,spot)) -> {Nat.is_lt(spot_index(checked,spot),checked_jobs(checked)) == True{} : Bool}:
  Checked{+target,+cap,+run} = checked
  Spot{+pair,+unit,+row} = spot
  Gemm.Both{+pair_inside,+rest} = placed
  Gemm.Both{small,+before_end} = rest
  +un = U32.to_nat(target_units(target))
  GemmReadback.index_below(target_once(target),pair,unit,U32.to_nat(target_pairs(target)),un,pair_inside,
    unit_below(U32.to_nat(target_start(target)),U32.to_nat(target_end(target)),un,unit,row,before_end,run_reach(target,cap,run)))

# The job of a spot stores at the spot's word.
def spot_contains(~accumulate: PointJobs.Accumulate(),+checked: Checked,+spot: Spot,+placed: Placed(checked,spot)) ->
  {WriteReadback.contains(~Tile.Tile16,spot_word(checked,spot),PointJobs.step_writes_by(~accumulate,U32.from_nat(spot_index(checked,spot)),target_matrix(checked_target(checked)),
    target_readers(checked_target(checked)))) == True{} : Bool}:
  Checked{+target,+cap,+run} = checked
  Target{+matrix,+readers} = target
  Gemm.PointMatrix{Gemm.TileMatrix{+channels,+positions,+inner,+start,+output_positions},+cell,+stride} = matrix
  PointJobs.Readers{+bias,+cells,+pair_tiles} = readers
  Run{+cap_ok,+positive,+double_pairs,+double_units,+jobs_room,+end_room,+last_unit,+reach} = run
  Spot{+pair,+unit,+row} = spot
  Gemm.Both{+pair_inside,+rest} = placed
  Gemm.Both{+small,+before_end} = rest
  +goal = {Target{Gemm.PointMatrix{Gemm.TileMatrix{channels,positions,inner,start,output_positions},cell,stride},PointJobs.Readers{bias,cells,pair_tiles}} : Target}
  +checks = {Run{cap_ok,positive,double_pairs,double_units,jobs_room,end_room,last_unit,reach} : Run<goal,cap>}
  +pairs = target_pairs(goal)
  +units = target_units(goal)
  +pn = U32.to_nat(pairs)
  +un = U32.to_nat(units)
  +s = U32.to_nat(start)
  +once = target_once(goal)
  +unit_inside = unit_below(s,U32.to_nat(output_positions),un,unit,row,before_end,reach)
  +index = GemmReadback.index_of(once,pair,unit,pn,un)
  +inside = GemmReadback.index_below(once,pair,unit,pn,un,pair_inside,unit_inside)
  +index_room = Order.transitive(index,Nat.mul(un,pn),U32.to_nat(cap),Bounds.strict_le(index,Nat.mul(un,pn),inside),jobs_room)
  +number = Bounds.roundtrip(index,cap,cap_ok,index_room)
  +pair_word = job_pair(goal,index)
  +unit_word = job_unit(goal,index)
  +pair_exact = GemmReadback.job_group_exact(once,U32.from_nat(index),pairs,units,pair,unit,pn,un,cap,cap_ok,number,{==},{==},pair_inside,unit_inside,double_pairs,double_units)
  +unit_exact = GemmReadback.job_unit_exact(once,U32.from_nat(index),pairs,units,pair,unit,pn,un,cap,cap_ok,number,{==},{==},pair_inside,unit_inside,double_pairs,double_units)
  +block = split_block(goal,cap,checks,index,GemmReadback.split_of(once,index,pn,un,cap,cap_ok,index_room,inside))
  +offset = row_offset(row,block_base(start,unit_word,pairs,pair_word),pairs)
  +wanted = Nat.add(Nat.mul(Nat.add(Nat.add(s,Nat.mul(unit,4n)),row),pn),pair)
  +stored = Equal.trans(Nat,U32.to_nat(offset),stored_number(Nat.add(first_position(start,unit_word),row),pairs,pair_word),wanted,
    row_number(row,start,unit_word,pairs,pair_word,cap,block_room(goal,pair_word,unit_word,cap,block),small),
    Equal.trans(Nat,stored_number(Nat.add(first_position(start,unit_word),row),pairs,pair_word),Nat.add(Nat.mul(Nat.add(Nat.add(s,Nat.mul(unit,4n)),row),pn),U32.to_nat(pair_word)),wanted,
      Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(Nat.add(Nat.add(s,Nat.mul(value,4n)),row),pn),U32.to_nat(pair_word)),U32.to_nat(unit_word),unit,unit_exact),
      Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(Nat.add(Nat.add(s,Nat.mul(unit,4n)),row),pn),value),U32.to_nat(pair_word),pair,pair_exact)))
  +room = spot_room(Checked{goal,cap,checks},Spot{pair,unit,row},Gemm.Both{pair_inside,Gemm.Both{small,before_end}})
  +address = Arithmetic.to_nat_injective(offset,U32.from_nat(wanted),Equal.trans(Nat,U32.to_nat(offset),wanted,U32.to_nat(U32.from_nat(wanted)),stored,
    Equal.sym(Nat,U32.to_nat(U32.from_nat(wanted)),wanted,Bounds.roundtrip(wanted,cap,cap_ok,room))))
  %address : {WriteReadback.contains(~Tile.Tile16,_,PointJobs.step_writes_by(~accumulate,U32.from_nat(index),target_matrix(goal),target_readers(goal))) == True{} : Bool}
  block_contains(~accumulate,goal,pair_word,unit_word,cap,block,row,small,
    %Equal.sym(Nat,U32.to_nat(unit_word),unit,unit_exact) : {Nat.is_lt(Nat.add(Nat.add(s,Nat.mul(_,4n)),row),U32.to_nat(output_positions)) == True{} : Bool}
    before_end)

# ---- Readback ----

def Asked() -> Data:
  GemmJobs.Asked(~Checked,~Spot,~Placed)

def asked_word(+query: Asked()) -> U32:
  GemmJobs.query_word(~Checked,~Spot,~Placed,~spot_word,query)

def asked_index(+query: Asked()) -> Nat:
  GemmJobs.Query{+checked,+spot,placed} = query
  spot_index(checked,spot)

def Scheduled(+query: Asked(),rest: Nat,position: Nat,matrix: Gemm.PointMatrix,readers: PointJobs.Readers) -> Data:
  +checked = GemmJobs.query_layout(~Checked,~Spot,~Placed,query)
  Gemm.Both<{matrix == target_matrix(checked_target(checked)) : Gemm.PointMatrix},{readers == target_readers(checked_target(checked)) : PointJobs.Readers}>

def scheduled_kept(+query: Asked(),+rest: Nat,+position: Nat,+matrix: Gemm.PointMatrix,+readers: PointJobs.Readers,schedule: Scheduled(query,1n+rest,position,matrix,readers)) ->
  Scheduled(query,rest,1n+position,matrix,PointJobs.kept(U32.from_nat(position),matrix,readers)):
  schedule

def scheduled_found(~accumulate: PointJobs.Accumulate(),+query: Asked(),+rest: Nat,+matrix: Gemm.PointMatrix,+readers: PointJobs.Readers,schedule: Scheduled(query,1n+rest,asked_index(query),matrix,readers)) ->
  {WriteReadback.contains(~Tile.Tile16,asked_word(query),PointJobs.step_writes_by(~accumulate,U32.from_nat(asked_index(query)),matrix,readers)) == True{} : Bool}:
  GemmJobs.Query{+checked,+spot,+placed} = query
  Gemm.Both{same_matrix,same_readers} = schedule
  +index = spot_index(checked,spot)
  %Equal.sym(Gemm.PointMatrix,matrix,target_matrix(checked_target(checked)),same_matrix) :
    {WriteReadback.contains(~Tile.Tile16,spot_word(checked,spot),PointJobs.step_writes_by(~accumulate,U32.from_nat(index),_,readers)) == True{} : Bool}
  %Equal.sym(PointJobs.Readers,readers,target_readers(checked_target(checked)),same_readers) :
    {WriteReadback.contains(~Tile.Tile16,spot_word(checked,spot),PointJobs.step_writes_by(~accumulate,U32.from_nat(index),target_matrix(checked_target(checked)),_)) == True{} : Bool}
  spot_contains(~accumulate,checked,spot,placed)

# The stores of a run hold the word of each spot.
def all_contains(~accumulate: PointJobs.Accumulate(),+checked: Checked,+spot: Spot,+placed: Placed(checked,spot)) ->
  {WriteReadback.contains(~Tile.Tile16,spot_word(checked,spot),checked_writes_by(~accumulate,checked)) == True{} : Bool}:
  +query = {GemmJobs.Query{checked,spot,placed} : Asked()}
  Fold.contains_at(~Tile.Tile16,~Gemm.PointMatrix,~PointJobs.Readers,~PointJobs.step_writes_by(~accumulate),~PointJobs.kept,~Asked(),~asked_word,~asked_index,~Scheduled,~scheduled_kept,~scheduled_found(~accumulate),
    query,checked_jobs(checked),asked_index(query),0n,target_matrix(checked_target(checked)),target_readers(checked_target(checked)),Gemm.Both{{==},{==}},
    spot_inside(checked,spot,placed),{==})

# The stores of a run read back, at the word of each spot, the value of the
# spot.
def spot_readback(~accumulate: PointJobs.Accumulate(),+checked: Checked,+spot: Spot,+placed: Placed(checked,spot),+depth: Nat,+tree: Tree.Tree<Tile.Tile16>,+balanced: Shape.Balanced(Tile.Tile16,depth,tree),
  +depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},+room: {Nat.is_le(checked_bound(checked),U32.to_nat(Tree.capacity(Tile.Tile16,tree))) == True{} : Bool}) ->
  {Tree.read(Tile.Tile16,Writes.pure(Tile.Tile16,checked_writes_by(~accumulate,checked),tree),spot_word(checked,spot)) == spot_wanted(~accumulate,checked,spot) : Tile.Tile16}:
  GemmJobs.located_readback(~Tile.Tile16,~Checked,~Spot,~Placed,~spot_word,~spot_wanted(~accumulate),~checked_bound,~spot_below,~spot_same(~accumulate),
    checked,spot,placed,depth,tree,balanced,depth_ok,room,checked_writes_by(~accumulate,checked),all_every(~accumulate,checked),all_contains(~accumulate,checked,spot,placed))

# ---- The address of the biases of a pair ----

# The biases of pair pair are the bias tiles of the groups 2 pair and
# 2 pair + 1, or the last group for a missing upper one.
def pair_group_words(+groups: U32,+group_count: Nat,+groups_exact: {U32.to_nat(groups) == group_count : Nat},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(group_count,U32.to_nat(cap)) == True{} : Bool},
  +pair: Nat,+pair_inside: {Nat.is_le(1n+Nat.mul(pair,2n),group_count) == True{} : Bool}) ->
  Gemm.Both<{U32.from_nat(Nat.mul(pair,2n)) == (U32.from_nat(pair) * 2 : U32) : U32},
    {U32.from_nat(Nat.min(1n+Nat.mul(pair,2n),Nat.sub(group_count,1n))) ==
      U32.min((U32.from_nat(pair) * 2 + 1 : U32),(groups - 1 : U32)) : U32}>:
  +double = Nat.mul(pair,2n)
  +last = Nat.sub(group_count,1n)
  +upper = Nat.min(1n+double,last)
  +groups_positive = {Order.lt_le(0n,1n+double,group_count,{==},pair_inside) : {Nat.is_lt(0n,group_count) == True{} : Bool}}
  +next_cap = {Order.transitive(1n+double,group_count,U32.to_nat(cap),pair_inside,room) : {Nat.is_le(1n+double,U32.to_nat(cap)) == True{} : Bool}}
  +double_cap = {Order.transitive(double,1n+double,U32.to_nat(cap),Order.successor(double,double,Order.reflexive(double)),next_cap) :
    {Nat.is_le(double,U32.to_nat(cap)) == True{} : Bool}}
  +pair_cap = {Order.transitive(pair,double,U32.to_nat(cap),RowMajor.product_covers(pair,2n,{==}),double_cap) : {Nat.is_le(pair,U32.to_nat(cap)) == True{} : Bool}}
  +pair_word = U32.from_nat(pair)
  +double_word = (pair_word * 2 : U32)
  +next_word = (double_word + 1 : U32)
  +last_word = (groups - 1 : U32)
  +upper_word = U32.min(next_word,last_word)
  +double_exact = Arithmetic.product_exact(cap,cap_ok,pair_word,2,pair,2n,Bounds.roundtrip(pair,cap,cap_ok,pair_cap),{==},double_cap)
  +next_exact = {Equal.trans(Nat,U32.to_nat(next_word),Nat.add(double,1n),1n+double,
      Arithmetic.sum_exact(cap,cap_ok,double_word,1,double,1n,double_exact,{==},%Order.commutative(1n,double) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} next_cap),
      Order.commutative(double,1n)) : {U32.to_nat(next_word) == 1n+double : Nat}}
  +last_exact = Arithmetic.difference_exact(cap,cap_ok,groups,1,group_count,1n,groups_exact,{==},Order.strict_follow(0n,group_count,groups_positive),room)
  +upper_exact = {%next_exact : {U32.to_nat(upper_word) == Nat.min(_,last) : Nat}
    %last_exact : {U32.to_nat(upper_word) == Nat.min(U32.to_nat(next_word),_) : Nat}
    U32Extrema.minimum(next_word,last_word) : {U32.to_nat(upper_word) == upper : Nat}}
  Gemm.Both{Arithmetic.from_nat_exact(double_word,double,double_exact),Arithmetic.from_nat_exact(upper_word,upper,upper_exact)}

def bias_pair_address(+bias: Tree.Tree<Tile.Tile8>,+groups: U32,+group_count: Nat,+pair: Nat,
  +group_words: Gemm.Both<{U32.from_nat(Nat.mul(pair,2n)) == (U32.from_nat(pair) * 2 : U32) : U32},
    {U32.from_nat(Nat.min(1n+Nat.mul(pair,2n),Nat.sub(group_count,1n))) ==
      U32.min((U32.from_nat(pair) * 2 + 1 : U32),(groups - 1 : U32)) : U32}>) ->
  {biases(bias,groups,U32.from_nat(pair)) ==
    Tile.Tiles{Tree.read(Tile.Tile8,bias,U32.from_nat(Nat.mul(pair,2n))),Tree.read(Tile.Tile8,bias,U32.from_nat(Nat.min(1n+Nat.mul(pair,2n),Nat.sub(group_count,1n))))} : Tile.Tile16}:
  Gemm.Both{+lower,+higher} = group_words
  +double = Nat.mul(pair,2n)
  +upper = Nat.min(1n+double,Nat.sub(group_count,1n))
  +pair_word = U32.from_nat(pair)

  %Equal.sym(U32,U32.from_nat(double),(pair_word * 2 : U32),lower) :
    {biases(bias,groups,pair_word) == Tile.Tiles{Tree.read(Tile.Tile8,bias,_),Tree.read(Tile.Tile8,bias,U32.from_nat(upper))} : Tile.Tile16}
  %Equal.sym(U32,U32.from_nat(upper),U32.min((pair_word * 2 + 1 : U32),(groups - 1 : U32)),higher) :
    {biases(bias,groups,pair_word) == Tile.Tiles{Tree.read(Tile.Tile8,bias,(pair_word * 2 : U32)),Tree.read(Tile.Tile8,bias,_)} : Tile.Tile16}
  {==}


def biases_address(+bias: Tree.Tree<Tile.Tile8>,+groups: U32,+group_count: Nat,+groups_exact: {U32.to_nat(groups) == group_count : Nat},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(group_count,U32.to_nat(cap)) == True{} : Bool},
  +pair: Nat,+pair_inside: {Nat.is_le(1n+Nat.mul(pair,2n),group_count) == True{} : Bool}) ->
  {biases(bias,groups,U32.from_nat(pair)) ==
    Tile.Tiles{Tree.read(Tile.Tile8,bias,U32.from_nat(Nat.mul(pair,2n))),Tree.read(Tile.Tile8,bias,U32.from_nat(Nat.min(1n+Nat.mul(pair,2n),Nat.sub(group_count,1n))))} : Tile.Tile16}:
  bias_pair_address(bias,groups,group_count,pair,pair_group_words(groups,group_count,groups_exact,cap,cap_ok,room,pair,pair_inside))

# ---- The cells before a run ----

# A cell before the stores of the first position of a run.
type SpotKeep is Data:
  SpotKeep{checked: Checked,query: Nat,
    before: {Nat.is_lt(query,Nat.mul(U32.to_nat(target_start(checked_target(checked))),U32.to_nat(target_pairs(checked_target(checked))))) == True{} : Bool}}

def keep_checked(+kept: SpotKeep) -> Checked:
  SpotKeep{checked,query,before} = kept
  checked

def keep_word(+kept: SpotKeep) -> U32:
  SpotKeep{checked,query,+before} = kept
  U32.from_nat(query)

# Every store of a run lies at or after the stores of its first position.
def spot_after(+checked: Checked,+spot: Spot) ->
  {Nat.is_le(Nat.mul(U32.to_nat(target_start(checked_target(checked))),U32.to_nat(target_pairs(checked_target(checked)))),spot_number(checked,spot)) == True{} : Bool}:
  Spot{+pair,+unit,+row} = spot
  +start = U32.to_nat(target_start(checked_target(checked)))
  +pairs = U32.to_nat(target_pairs(checked_target(checked)))
  +position = spot_position(checked,unit,row)
  Order.transitive(Nat.mul(start,pairs),Nat.mul(position,pairs),Nat.add(Nat.mul(position,pairs),pair),
    RowMajor.mul_le(start,position,pairs,
      Order.transitive(start,Nat.add(start,Nat.mul(unit,4n)),position,Order.le_add(start,Nat.mul(unit,4n)),Order.le_add(Nat.add(start,Nat.mul(unit,4n)),row))),
    Order.le_add(Nat.mul(position,pairs),pair))

def keep_apart(+kept: SpotKeep,+spot: Spot,+placed: Placed(keep_checked(kept),spot)) ->
  {U32.is_eq(spot_word(keep_checked(kept),spot),keep_word(kept)) == False{} : Bool}:
  SpotKeep{+checked,+query,+before} = kept
  +cap = checked_cap(checked)
  +number = spot_number(checked,spot)
  +number_room = spot_room(checked,spot,placed)
  +query_below = Order.lt_le(query,Nat.mul(U32.to_nat(target_start(checked_target(checked))),U32.to_nat(target_pairs(checked_target(checked)))),number,before,
    spot_after(checked,spot))
  Updates.words_apart(number,query,cap,checked_cap_ok(checked),number_room,
    Order.transitive(query,number,U32.to_nat(cap),Bounds.strict_le(query,number,query_below),number_room),
    %Equal.sym(Bool,Nat.is_eq(number,query),Nat.is_eq(query,number),Order.equal_symmetric(number,query)) : {_ == False{} : Bool}
    Order.less_not_equal(query,number,query_below))

# A run keeps the cells before the stores of its first position.
def spot_keeps_by(~accumulate: PointJobs.Accumulate(),+checked: Checked,+query: Nat,
  +before: {Nat.is_lt(query,Nat.mul(U32.to_nat(target_start(checked_target(checked))),U32.to_nat(target_pairs(checked_target(checked))))) == True{} : Bool},
  +depth: Nat,+tree: Tree.Tree<Tile.Tile16>,+balanced: Shape.Balanced(Tile.Tile16,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +room: {Nat.is_le(checked_bound(checked),U32.to_nat(Tree.capacity(Tile.Tile16,tree))) == True{} : Bool},
  +inside: {U32.is_lt(U32.from_nat(query),Tree.capacity(Tile.Tile16,tree)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile16,Writes.pure(Tile.Tile16,checked_writes_by(~accumulate,checked),tree),U32.from_nat(query)) == Tree.read(Tile.Tile16,tree,U32.from_nat(query)) : Tile.Tile16}:
  +kept = {SpotKeep{checked,query,before} : SpotKeep}
  +writes = checked_writes_by(~accumulate,checked)
  +holds = all_every(~accumulate,checked)
  +output = {GemmJobs.Output{checked,Tree.capacity(Tile.Tile16,tree),room} : GemmJobs.Bounded(~Checked,~checked_bound)}
  WriteReadback.excluded_readback(~Tile.Tile16,writes,tree,depth,U32.from_nat(query),balanced,depth_ok,
    Every.inside(~Tile.Tile16,~Checked,~GemmJobs.Located(~Tile.Tile16,~Checked,~Spot,~Placed,~spot_word,~spot_wanted(~accumulate)),
      ~GemmJobs.Bounded(~Checked,~checked_bound),~GemmJobs.output_layout(~Checked,~checked_bound),~GemmJobs.output_capacity(~Checked,~checked_bound),
      ~GemmJobs.located_inside(~Tile.Tile16,~Checked,~Spot,~Placed,~spot_word,~spot_wanted(~accumulate),~checked_bound,~spot_below),output,writes,holds),
    inside,
    Every.excludes(~Tile.Tile16,~Checked,~GemmJobs.Located(~Tile.Tile16,~Checked,~Spot,~Placed,~spot_word,~spot_wanted(~accumulate)),~SpotKeep,~keep_checked,~keep_word,
      ~GemmJobs.located_apart(~Tile.Tile16,~Checked,~Spot,~Placed,~spot_word,~spot_wanted(~accumulate),~SpotKeep,~keep_checked,~keep_word,~keep_apart),kept,writes,holds))

def spot_keeps(+checked: Checked,+query: Nat,
  +before: {Nat.is_lt(query,Nat.mul(U32.to_nat(target_start(checked_target(checked))),U32.to_nat(target_pairs(checked_target(checked))))) == True{} : Bool},
  +depth: Nat,+tree: Tree.Tree<Tile.Tile16>,+balanced: Shape.Balanced(Tile.Tile16,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +room: {Nat.is_le(checked_bound(checked),U32.to_nat(Tree.capacity(Tile.Tile16,tree))) == True{} : Bool},
  +inside: {U32.is_lt(U32.from_nat(query),Tree.capacity(Tile.Tile16,tree)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile16,Writes.pure(Tile.Tile16,checked_writes(checked),tree),U32.from_nat(query)) == Tree.read(Tile.Tile16,tree,U32.from_nat(query)) : Tile.Tile16}:
  spot_keeps_by(~PointJobs.job_accumulators_with(~Gemm.row_times,~PointJobs.whole_inner),checked,query,before,depth,tree,balanced,depth_ok,room,inside)

# ---- The words of a run ----

# A positive count of groups makes at least one pair.
def pairs_positive(+groups: Nat,+positive: {Nat.is_lt(0n,groups) == True{} : Bool}) -> {Nat.is_lt(0n,Nat.div(Nat.add(groups,1n),2n)) == True{} : Bool}:
  match groups:
    case 0n: Empty.absurd({Nat.is_lt(0n,Nat.div(Nat.add(0n,1n),2n)) == True{} : Bool},Order.below_zero(0n,positive))
    case 1n+ +rest:
      %Order.commutative(1n,rest) : {Nat.is_lt(0n,Nat.div(1n+_,2n)) == True{} : Bool}
      %Equal.sym(Nat,Nat.div(Nat.add(Nat.mul(1n,2n),rest),2n),Nat.add(1n,Nat.div(rest,2n)),DivisionOrder.quotient_shift(rest,2n,1n,{==})) :
        {Nat.is_lt(0n,_) == True{} : Bool}
      {==}

# The pairs of groups are at most the groups.
def pairs_below(+groups: Nat) -> {Nat.is_le(Nat.div(Nat.add(groups,1n),2n),groups) == True{} : Bool}:
  %Order.commutative(1n,groups) : {Nat.is_le(Nat.div(_,2n),groups) == True{} : Bool}
  Division.lt_succ_le(Nat.div(1n+groups,2n),groups,
    DivisionOrder.quotient_below(1n+groups,1n+groups,2n,{==},Order.lt_succ(groups,Nat.mul(groups,2n),RowMajor.product_covers(groups,2n,{==}))))

# The units of a count are at most the count, and four times them cover it.
def units_below(+count: Nat) -> {Nat.is_le(Nat.div(Nat.add(count,3n),4n),count) == True{} : Bool}:
  %Order.commutative(3n,count) : {Nat.is_le(Nat.div(_,4n),count) == True{} : Bool}
  Division.lt_succ_le(Nat.div(Nat.add(3n,count),4n),count,
    DivisionOrder.quotient_below(Nat.add(3n,count),1n+count,4n,{==},Order.lt_succ(count,Nat.mul(count,4n),RowMajor.product_covers(count,4n,{==}))))

def units_cover(+count: Nat) -> {Nat.is_le(count,Nat.mul(Nat.div(Nat.add(count,3n),4n),4n)) == True{} : Bool}:
  +units = Nat.div(Nat.add(count,3n),4n)
  Division.lt_succ_le(count,Nat.mul(units,4n),
    %Order.commutative(count,3n) : {Nat.is_lt(_,Nat.add(4n,Nat.mul(units,4n))) == True{} : Bool}
    DivisionOrder.next_above(Nat.add(count,3n),4n,{==}))

# Four pairs fit where the doubled groups and two do.
def four_pairs(+groups: Nat) -> {Nat.is_le(Nat.mul(4n,Nat.div(Nat.add(groups,1n),2n)),Nat.add(Nat.mul(groups,2n),2n)) == True{} : Bool}:
  +pairs = Nat.div(Nat.add(groups,1n),2n)
  %RowMajor.multiply_commutative(pairs,4n) : {Nat.is_le(_,Nat.add(Nat.mul(groups,2n),2n)) == True{} : Bool}
  %RowMajor.assoc(pairs,2n,2n) : {Nat.is_le(_,Nat.add(Nat.mul(groups,2n),2n)) == True{} : Bool}
  %RowMajor.distrib(groups,1n,2n) : {Nat.is_le(Nat.mul(Nat.mul(pairs,2n),2n),_) == True{} : Bool}
  RowMajor.mul_le(Nat.mul(pairs,2n),Nat.add(groups,1n),2n,DivisionOrder.product_below(Nat.add(groups,1n),2n,{==}))

# A run of the positions [start, end) of channels in groups whose words stay
# below a capacity, twice of which fit the cap.
def call_run(+channels: U32,+positions: U32,+inner: U32,+start: U32,+end: U32,+cell: U32,+stride: U32,+readers: PointJobs.Readers,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+capacity: Nat,+cap_exact: {Nat.is_le(Nat.double(capacity),U32.to_nat(cap)) == True{} : Bool},
  +groups: Nat,+groups_exact: {U32.to_nat(Gemm.blocks(channels)) == groups : Nat},+groups_positive: {Nat.is_lt(0n,groups) == True{} : Bool},
  +group_room: {Nat.is_le(Nat.add(Nat.mul(groups,2n),2n),capacity) == True{} : Bool},+four_room: {Nat.is_le(4n,capacity) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(positions)) == True{} : Bool},
  +end_exact: {U32.to_nat(end) == Nat.add(U32.to_nat(start),U32.to_nat(positions)) : Nat},
  +span: {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(start),U32.to_nat(positions)),groups),capacity) == True{} : Bool}) ->
  Run<Target{Gemm.PointMatrix{Gemm.TileMatrix{channels,positions,inner,start,end},cell,stride},readers},cap>:
  +count = U32.to_nat(positions)
  +first = U32.to_nat(start)
  +last = Nat.add(first,count)
  +twice = {%Equal.sym(Nat,Nat.add(capacity,capacity),Nat.double(capacity),Addition.double(capacity)) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} cap_exact : {Nat.is_le(Nat.add(capacity,capacity),U32.to_nat(cap)) == True{} : Bool}}
  +capacity_cap = Order.transitive(capacity,Nat.add(capacity,capacity),U32.to_nat(cap),Order.le_add(capacity,capacity),twice)
  +last_positive = Order.lt_le(0n,count,last,positive,Order.right_le(first,count))
  +groups_capacity = Order.transitive(groups,Nat.mul(last,groups),capacity,DivisionOrder.le_multiple(last,groups,last_positive),span)
  +one_capacity = Order.transitive(1n,4n,capacity,{==},four_room)
  +sum_room = Order.transitive(Nat.add(groups,1n),Nat.add(capacity,capacity),U32.to_nat(cap),
    Order.transitive(Nat.add(groups,1n),Nat.add(capacity,1n),Nat.add(capacity,capacity),Order.add_le(groups,capacity,1n,groups_capacity),
      Order.offset_le(capacity,1n,capacity,one_capacity)),twice)
  +divisor_two = Order.transitive(Nat.double(2n),Nat.double(capacity),U32.to_nat(cap),Order.double_le(2n,capacity,Order.transitive(2n,4n,capacity,{==},four_room)),cap_exact)
  +divisor_four = Order.transitive(Nat.double(4n),Nat.double(capacity),U32.to_nat(cap),Order.double_le(4n,capacity,four_room),cap_exact)
  +pairs = Nat.div(Nat.add(groups,1n),2n)
  +pairs_exact = {U32Division.quotient_exact(cap,cap_ok,(Gemm.blocks(channels) + 1 : U32),2,Nat.add(groups,1n),2n,
      Arithmetic.sum_exact(cap,cap_ok,Gemm.blocks(channels),1,groups,1n,groups_exact,{==},sum_room),{==},divisor_two,{==}) :
    {U32.to_nat(Gemm.pairs_of(channels)) == pairs : Nat}}
  +pairs_back = Equal.sym(Nat,U32.to_nat(Gemm.pairs_of(channels)),pairs,pairs_exact)
  +pairs_groups = pairs_below(groups)
  +pairs_capacity = Order.transitive(pairs,groups,capacity,pairs_groups,groups_capacity)
  +count_capacity = Order.transitive(count,last,capacity,Order.right_le(first,count),
    Order.transitive(last,Nat.mul(last,groups),capacity,RowMajor.product_covers(last,groups,groups_positive),span))
  +count_room = Order.transitive(Nat.add(count,3n),Nat.add(capacity,capacity),U32.to_nat(cap),
    Order.transitive(Nat.add(count,3n),Nat.add(capacity,3n),Nat.add(capacity,capacity),Order.add_le(count,capacity,3n,count_capacity),
      Order.offset_le(capacity,3n,capacity,Order.transitive(3n,4n,capacity,{==},four_room))),twice)
  +units = Nat.div(Nat.add(count,3n),4n)
  +units_exact = {U32Division.quotient_exact(cap,cap_ok,(positions + 3 : U32),4,Nat.add(count,3n),4n,
      Arithmetic.sum_exact(cap,cap_ok,positions,3,count,3n,{==},{==},count_room),{==},divisor_four,{==}) :
    {U32.to_nat(((positions + 3) / 4 : U32)) == units : Nat}}
  +units_back = Equal.sym(Nat,U32.to_nat(((positions + 3) / 4 : U32)),units,units_exact)
  +products = {Order.transitive(Nat.mul(last,pairs),Nat.mul(last,groups),capacity,RowMajor.mul_le_left(last,pairs,groups,pairs_groups),span) :
    {Nat.is_le(Nat.mul(last,pairs),capacity) == True{} : Bool}}
  +fours = Order.transitive(Nat.mul(4n,pairs),Nat.add(Nat.mul(groups,2n),2n),capacity,four_pairs(groups),group_room)
  +end_value = {Equal.trans(Nat,Nat.add(Nat.mul(Nat.add(last,3n),pairs),pairs),Nat.add(Nat.add(Nat.mul(last,pairs),Nat.mul(3n,pairs)),pairs),Nat.add(Nat.mul(last,pairs),Nat.mul(4n,pairs)),
      Equal.cong(Nat,Nat,value => Nat.add(value,pairs),Nat.mul(Nat.add(last,3n),pairs),Nat.add(Nat.mul(last,pairs),Nat.mul(3n,pairs)),RowMajor.distrib(last,3n,pairs)),
      Equal.trans(Nat,Nat.add(Nat.add(Nat.mul(last,pairs),Nat.mul(3n,pairs)),pairs),Nat.add(Nat.mul(last,pairs),Nat.add(Nat.mul(3n,pairs),pairs)),
        Nat.add(Nat.mul(last,pairs),Nat.mul(4n,pairs)),Addition.associative(Nat.mul(last,pairs),Nat.mul(3n,pairs),pairs),
        Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(last,pairs),value),Nat.add(Nat.mul(3n,pairs),pairs),Nat.mul(4n,pairs),Order.commutative(Nat.mul(3n,pairs),pairs)))) :
    {Nat.add(Nat.mul(Nat.add(last,3n),pairs),pairs) == Nat.add(Nat.mul(last,pairs),Nat.mul(4n,pairs)) : Nat}}
  +end_room = {%Equal.sym(Nat,U32.to_nat(end),last,end_exact) : {Nat.is_le(Nat.add(Nat.mul(Nat.add(_,3n),pairs),pairs),U32.to_nat(cap)) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(Nat.mul(Nat.add(last,3n),pairs),pairs),Nat.add(Nat.mul(last,pairs),Nat.mul(4n,pairs)),end_value) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
    Order.transitive(Nat.add(Nat.mul(last,pairs),Nat.mul(4n,pairs)),Nat.add(capacity,capacity),U32.to_nat(cap),
      Order.transitive(Nat.add(Nat.mul(last,pairs),Nat.mul(4n,pairs)),Nat.add(capacity,Nat.mul(4n,pairs)),Nat.add(capacity,capacity),
        Order.add_le(Nat.mul(last,pairs),capacity,Nat.mul(4n,pairs),products),Order.offset_le(capacity,Nat.mul(4n,pairs),capacity,fours)),twice) :
    {Nat.is_le(Nat.add(Nat.mul(Nat.add(U32.to_nat(end),3n),pairs),pairs),U32.to_nat(cap)) == True{} : Bool}}
  +units_cap = Order.transitive(units,capacity,U32.to_nat(cap),Order.transitive(units,count,capacity,units_below(count),count_capacity),capacity_cap)
  Run{cap_ok,
    %pairs_back : {Nat.is_lt(0n,_) == True{} : Bool} pairs_positive(groups,groups_positive),
    %pairs_back : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool}
      Order.transitive(Nat.double(pairs),Nat.double(capacity),U32.to_nat(cap),Order.double_le(pairs,capacity,pairs_capacity),cap_exact),
    %units_back : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool}
      Order.transitive(Nat.double(units),Nat.double(capacity),U32.to_nat(cap),Order.double_le(units,capacity,Order.transitive(units,count,capacity,units_below(count),count_capacity)),cap_exact),
    %units_back : {Nat.is_le(Nat.mul(_,U32.to_nat(Gemm.pairs_of(channels))),U32.to_nat(cap)) == True{} : Bool}
    %pairs_back : {Nat.is_le(Nat.mul(units,_),U32.to_nat(cap)) == True{} : Bool}
      Order.transitive(Nat.mul(units,pairs),Nat.mul(last,pairs),U32.to_nat(cap),
        RowMajor.mul_le(units,last,pairs,Order.transitive(units,count,last,units_below(count),Order.right_le(first,count))),
        Order.transitive(Nat.mul(last,pairs),capacity,U32.to_nat(cap),products,capacity_cap)),
    %pairs_back : {Nat.is_le(Nat.add(Nat.mul(Nat.add(U32.to_nat(end),3n),_),_),U32.to_nat(cap)) == True{} : Bool} end_room,
    %units_back : {Nat.is_le(Nat.add(first,Nat.mul(_,4n)),Nat.add(4n,U32.to_nat(end))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(end),last,end_exact) : {Nat.is_le(Nat.add(first,Nat.mul(units,4n)),Nat.add(4n,_)) == True{} : Bool}
      Order.transitive(Nat.add(first,Nat.mul(units,4n)),Nat.add(first,Nat.add(count,3n)),Nat.add(4n,last),
        Order.offset_le(first,Nat.mul(units,4n),Nat.add(count,3n),DivisionOrder.product_below(Nat.add(count,3n),4n,{==})),
        %Addition.associative(first,count,3n) : {Nat.is_le(_,Nat.add(4n,last)) == True{} : Bool}
        %Order.commutative(3n,last) : {Nat.is_le(_,Nat.add(4n,last)) == True{} : Bool}
        Order.add_le(3n,4n,last,{==})),
    %units_back : {Nat.is_le(U32.to_nat(end),Nat.add(first,Nat.mul(_,4n))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(end),last,end_exact) : {Nat.is_le(_,Nat.add(first,Nat.mul(units,4n))) == True{} : Bool}
      Order.offset_le(first,count,Nat.mul(units,4n),units_cover(count))}

# ---- The stores of a run by position ----

# The jobs of a matrix are its units times its pairs.
def count_word(+target: Target) -> {Gemm.point_count(target_matrix(target)) == (target_units(target) * target_pairs(target) : U32) : U32}:
  Target{+matrix,+readers} = target
  Gemm.PointMatrix{+product,+cell,+stride} = matrix
  Gemm.TileMatrix{+channels,+positions,+inner,+start,+end} = product
  {==}

def checked_count(+checked: Checked) ->
  {U32.to_nat(Gemm.point_count(target_matrix(checked_target(checked)))) == checked_jobs(checked) : Nat}:
  Checked{+target,+cap,+run} = checked
  %Equal.sym(U32,Gemm.point_count(target_matrix(target)),(target_units(target) * target_pairs(target) : U32),count_word(target)) :
    {U32.to_nat(_) == Nat.mul(U32.to_nat(target_units(target)),U32.to_nat(target_pairs(target))) : Nat}
  Arithmetic.product_exact(cap,run_cap(target,cap,run),target_units(target),target_pairs(target),U32.to_nat(target_units(target)),U32.to_nat(target_pairs(target)),{==},{==},
    run_jobs(target,cap,run))

# A checked run stores what the jobs of its matrix store.
def checked_stores_by(~accumulate: PointJobs.Accumulate(),+checked: Checked) ->
  {PointJobs.all_writes_by(~accumulate,U32.to_nat(Gemm.point_count(target_matrix(checked_target(checked)))),0,target_matrix(checked_target(checked)),target_readers(checked_target(checked))) ==
    checked_writes_by(~accumulate,checked) : List<&2,Writes.Assignment<Tile.Tile16>>}:
  Equal.cong(Nat,List<&2,Writes.Assignment<Tile.Tile16>>,count => PointJobs.all_writes_by(~accumulate,count,0,target_matrix(checked_target(checked)),target_readers(checked_target(checked))),
    U32.to_nat(Gemm.point_count(target_matrix(checked_target(checked)))),checked_jobs(checked),checked_count(checked))

def checked_stores(+checked: Checked) ->
  {PointJobs.all_writes(U32.to_nat(Gemm.point_count(target_matrix(checked_target(checked)))),0,target_matrix(checked_target(checked)),target_readers(checked_target(checked))) ==
    checked_writes(checked) : List<&2,Writes.Assignment<Tile.Tile16>>}:
  checked_stores_by(~PointJobs.job_accumulators_with(~Gemm.row_times,~PointJobs.whole_inner),checked)

# The position of the row of a unit is the position of its step.
def step_position(+start: Nat,+step: Nat) ->
  {Nat.add(Nat.add(start,Nat.mul(Nat.div(step,4n),4n)),Nat.mod(step,4n)) == Nat.add(start,step) : Nat}:
  Equal.trans(Nat,Nat.add(Nat.add(start,Nat.mul(Nat.div(step,4n),4n)),Nat.mod(step,4n)),Nat.add(start,Nat.add(Nat.mul(Nat.div(step,4n),4n),Nat.mod(step,4n))),Nat.add(start,step),
    Addition.associative(start,Nat.mul(Nat.div(step,4n),4n),Nat.mod(step,4n)),
    Equal.cong(Nat,Nat,value => Nat.add(start,value),Nat.add(Nat.mul(Nat.div(step,4n),4n),Nat.mod(step,4n)),step,
      Equal.sym(Nat,step,Nat.add(Nat.mul(Nat.div(step,4n),4n),Nat.mod(step,4n)),DivisionOrder.quotient_exact(step,4n,{==}))))

# A run stores at position start + step and pair pair the value of the unit
# and row of that step.
def stored_at_by(~accumulate: PointJobs.Accumulate(),+checked: Checked,+step: Nat,+pair: Nat,+pair_inside: {Nat.is_lt(pair,U32.to_nat(target_pairs(checked_target(checked)))) == True{} : Bool},
  +before_end: {Nat.is_lt(Nat.add(U32.to_nat(target_start(checked_target(checked))),step),U32.to_nat(target_end(checked_target(checked)))) == True{} : Bool},
  +depth: Nat,+tree: Tree.Tree<Tile.Tile16>,+balanced: Shape.Balanced(Tile.Tile16,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +room: {Nat.is_le(checked_bound(checked),U32.to_nat(Tree.capacity(Tile.Tile16,tree))) == True{} : Bool}) ->
  {Tree.read(Tile.Tile16,Writes.pure(Tile.Tile16,checked_writes_by(~accumulate,checked),tree),
      U32.from_nat(Nat.add(Nat.mul(Nat.add(U32.to_nat(target_start(checked_target(checked))),step),U32.to_nat(target_pairs(checked_target(checked)))),pair))) ==
    stored_value_by(~accumulate,checked_target(checked),U32.from_nat(pair),U32.from_nat(Nat.div(step,4n)),Nat.mod(step,4n)) : Tile.Tile16}:
  +start = U32.to_nat(target_start(checked_target(checked)))
  +pairs = U32.to_nat(target_pairs(checked_target(checked)))
  +spot = {Spot{pair,Nat.div(step,4n),Nat.mod(step,4n)} : Spot}
  +same = step_position(start,step)
  +placed = {Gemm.Both{pair_inside,Gemm.Both{Order.strict_follow(Nat.mod(step,4n),4n,DivisionOrder.remainder_small(step,4n,{==})),
      %Equal.sym(Nat,Nat.add(Nat.add(start,Nat.mul(Nat.div(step,4n),4n)),Nat.mod(step,4n)),Nat.add(start,step),same) :
        {Nat.is_lt(_,U32.to_nat(target_end(checked_target(checked)))) == True{} : Bool} before_end}} : Placed(checked,spot)}
  %same : {Tree.read(Tile.Tile16,Writes.pure(Tile.Tile16,checked_writes_by(~accumulate,checked),tree),U32.from_nat(Nat.add(Nat.mul(_,pairs),pair))) ==
    stored_value_by(~accumulate,checked_target(checked),U32.from_nat(pair),U32.from_nat(Nat.div(step,4n)),Nat.mod(step,4n)) : Tile.Tile16}
  spot_readback(~accumulate,checked,spot,placed,depth,tree,balanced,depth_ok,room)

def stored_at(+checked: Checked,+step: Nat,+pair: Nat,+pair_inside: {Nat.is_lt(pair,U32.to_nat(target_pairs(checked_target(checked)))) == True{} : Bool},
  +before_end: {Nat.is_lt(Nat.add(U32.to_nat(target_start(checked_target(checked))),step),U32.to_nat(target_end(checked_target(checked)))) == True{} : Bool},
  +depth: Nat,+tree: Tree.Tree<Tile.Tile16>,+balanced: Shape.Balanced(Tile.Tile16,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +room: {Nat.is_le(checked_bound(checked),U32.to_nat(Tree.capacity(Tile.Tile16,tree))) == True{} : Bool}) ->
  {Tree.read(Tile.Tile16,Writes.pure(Tile.Tile16,checked_writes(checked),tree),
      U32.from_nat(Nat.add(Nat.mul(Nat.add(U32.to_nat(target_start(checked_target(checked))),step),U32.to_nat(target_pairs(checked_target(checked)))),pair))) ==
    stored_value(checked_target(checked),U32.from_nat(pair),U32.from_nat(Nat.div(step,4n)),Nat.mod(step,4n)) : Tile.Tile16}:
  stored_at_by(~PointJobs.job_accumulators_with(~Gemm.row_times,~PointJobs.whole_inner),checked,step,pair,pair_inside,before_end,depth,tree,balanced,depth_ok,room)
