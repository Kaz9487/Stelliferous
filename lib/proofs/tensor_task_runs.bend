# The tasks of an operation are runs of adjacent ranges, each a run of whole units.
import Base
import ../tensor_pieces.bend as Pieces
import ../tensor_tasks.bend as Tasks
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as LimitBounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_order.bend as Order
import ./u32_extrema.bend as Extrema
import ../math_choice.bend as Choice
import ./boolean_logic.bend as Boolean
import ./u32_comparison.bend as Comparison
import ./nat_division.bend as Division
import ./row_major_index.bend as RowMajor
import ./nat_division_order.bend as DivisionOrder
import ./u32_division.bend as U32Division
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./tensor_move_readback.bend as Held
import ../tensor_grid.bend as Grid
import ../kernel_cost.bend as Cost
import ../default_kernel_cost.bend as DefaultCost
import ./tensor_pieces_plan.bend as PiecesPlan
import ./storage_buffer_model.bend as Buffer
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_pieces_content.bend as Content
import ../kernel_gemm.bend as KernelGemm

# The first cell of the first range and the end of the last one.
def range_low(ranges: Pieces.Ranges) -> Nat:
  match ranges:
    case Pieces.Range{first,count}: U32.to_nat(first)
    case Pieces.Split{left,right}: range_low(left)

def range_high(ranges: Pieces.Ranges) -> Nat:
  match ranges:
    case Pieces.Range{first,count}: Nat.add(U32.to_nat(first),U32.to_nat(count))
    case Pieces.Split{left,right}: range_high(right)

# The cells from low to high: a nonempty run of whole units ending at or before bound.
type Fits<-unit: Nat,-low: Nat,-high: Nat,-bound: Nat> is Data:
  Fits{low_units: Nat,high_units: Nat,low_exact: {low == Nat.mul(low_units,unit) : Nat},high_exact: {high == Nat.mul(high_units,unit) : Nat},
    less: {Nat.is_lt(low,high) == True{} : Bool},inside: {Nat.is_le(high,bound) == True{} : Bool}}

def fits_inside(+unit: Nat,+low: Nat,+high: Nat,+bound: Nat,+fits: Fits<unit,low,high,bound>) -> {Nat.is_le(high,bound) == True{} : Bool}:
  Fits{+low_units,+high_units,+low_exact,+high_exact,+less,+inside} = fits
  inside

# Ranges whose leaves fit and whose halves meet.
def Run(ranges: Pieces.Ranges,+unit: Nat,+bound: Nat) -> Data:
  match ranges:
    case Pieces.Range{+first,+count}: Fits<unit,U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),bound>
    case Pieces.Split{+left,+right}: Held.Both<Run(left,unit,bound),Held.Both<Run(right,unit,bound),{range_high(left) == range_low(right) : Nat}>>

def fits_wider(+unit: Nat,+low: Nat,+high: Nat,+bound: Nat,+wider: Nat,+fits: Fits<unit,low,high,bound>,
  +widened: {Nat.is_le(bound,wider) == True{} : Bool}) -> Fits<unit,low,high,wider>:
  Fits{+low_units,+high_units,+low_exact,+high_exact,+less,+inside} = fits
  Fits{low_units,high_units,low_exact,high_exact,less,Order.transitive(high,bound,wider,inside,widened)}

def fits_joined(+unit: Nat,+low: Nat,+left_high: Nat,+right_low: Nat,+high: Nat,+bound: Nat,+left: Fits<unit,low,left_high,bound>,
  +right: Fits<unit,right_low,high,bound>,+glue: {left_high == right_low : Nat}) -> Fits<unit,low,high,bound>:
  Fits{+low_units,+left_units,+low_exact,+left_exact,+left_less,+left_inside} = left
  Fits{+right_units,+high_units,+right_exact,+high_exact,+right_less,+right_inside} = right
  Fits{low_units,high_units,low_exact,high_exact,
    Bounds.nat_trans(low,left_high,high,left_less,Held.nat_le_moved(left_high,right_low,high,glue,Bounds.strict_le(right_low,high,right_less))),right_inside}

# A run fits as a whole.
def run_fits(ranges: Pieces.Ranges,+unit: Nat,+bound: Nat,+run: Run(ranges,unit,bound)) -> Fits<unit,range_low(ranges),range_high(ranges),bound>:
  match ranges:
    case Pieces.Range{+first,+count}: run
    case Pieces.Split{+left,+right}:
      Held.Both{+left_run,+rest} = run
      Held.Both{+right_run,+glue} = rest
      fits_joined(unit,range_low(left),range_high(left),range_low(right),range_high(right),bound,run_fits(left,unit,bound,left_run),
        run_fits(right,unit,bound,right_run),glue)

# A group from low to high whose layout is a run with the same ends.
def GroupRun(group: Tasks.Group,+low: Nat,+high: Nat,+unit: Nat,+bound: Nat) -> Data:
  match group:
    case Tasks.Group{+first,+count,+layout}:
      Held.Both<{U32.to_nat(first) == low : Nat},Held.Both<{Nat.add(U32.to_nat(first),U32.to_nat(count)) == high : Nat},
        Held.Both<Run(layout,unit,bound),Held.Both<{range_low(layout) == low : Nat},{range_high(layout) == high : Nat}>>>>

def layout_joined(+first: U32,+left_count: U32,+right_count: U32,+left: Pieces.Ranges,+right: Pieces.Ranges,+low: Nat,+middle: Nat,+high: Nat,+unit: Nat,+bound: Nat,
  +choice: Bool,+left_run: Run(left,unit,bound),+right_run: Run(right,unit,bound),+first_low: {U32.to_nat(first) == low : Nat},
  +left_low: {range_low(left) == low : Nat},+left_high: {range_high(left) == middle : Nat},+right_low: {range_low(right) == middle : Nat},
  +right_high: {range_high(right) == high : Nat},+ended: {Nat.add(U32.to_nat(first),U32.to_nat((left_count + right_count : U32))) == high : Nat}) ->
  Held.Both<Run(Bool.pick(Pieces.Ranges,choice,Pieces.Range{first,(left_count + right_count : U32)},Pieces.Split{left,right}),unit,bound),
    Held.Both<{range_low(Bool.pick(Pieces.Ranges,choice,Pieces.Range{first,(left_count + right_count : U32)},Pieces.Split{left,right})) == low : Nat},
      {range_high(Bool.pick(Pieces.Ranges,choice,Pieces.Range{first,(left_count + right_count : U32)},Pieces.Split{left,right})) == high : Nat}>>:
  match choice:
    case True{}:
      +glue = Equal.trans(Nat,range_high(left),middle,range_low(right),left_high,Equal.sym(Nat,range_low(right),middle,right_low))
      +whole = fits_joined(unit,range_low(left),range_high(left),range_low(right),range_high(right),bound,run_fits(left,unit,bound,left_run),
        run_fits(right,unit,bound,right_run),glue)
      +starts = Equal.trans(Nat,range_low(left),low,U32.to_nat(first),left_low,Equal.sym(Nat,U32.to_nat(first),low,first_low))
      +ends = Equal.trans(Nat,range_high(right),high,Nat.add(U32.to_nat(first),U32.to_nat((left_count + right_count : U32))),right_high,
        Equal.sym(Nat,Nat.add(U32.to_nat(first),U32.to_nat((left_count + right_count : U32))),high,ended))
      +moved = {%starts : Fits<unit,_,Nat.add(U32.to_nat(first),U32.to_nat((left_count + right_count : U32))),bound>
        {%ends : Fits<unit,range_low(left),_,bound> whole : Fits<unit,range_low(left),Nat.add(U32.to_nat(first),U32.to_nat((left_count + right_count : U32))),bound>} :
        Fits<unit,U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat((left_count + right_count : U32))),bound>}
      Held.Both{moved,Held.Both{first_low,ended}}
    case False{}:
      +glue = Equal.trans(Nat,range_high(left),middle,range_low(right),left_high,Equal.sym(Nat,range_low(right),middle,right_low))
      Held.Both{Held.Both{left_run,Held.Both{right_run,glue}},Held.Both{left_low,right_high}}

# Joining two adjacent groups keeps the facts.
def joined(+limits: Machine.Limits,+limit: U32,+left_ranges: Pieces.Ranges,+right_ranges: Pieces.Ranges,+left: Tasks.Group,+right: Tasks.Group,+low: Nat,+middle: Nat,
  +high: Nat,+unit: Nat,+bound: Nat,+left_facts: GroupRun(left,low,middle,unit,bound),+right_facts: GroupRun(right,middle,high,unit,bound),
  +room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  GroupRun(Tasks.group_join(limit,left_ranges,right_ranges,left,right),low,high,unit,bound):
  match left right:
    case Tasks.Group{+left_first,+left_count,+left_layout} Tasks.Group{+right_first,+right_count,+right_layout}:
      Held.Both{+first_low,+left_rest} = left_facts
      Held.Both{+left_end,+left_more} = left_rest
      Held.Both{+left_run,+left_ends} = left_more
      Held.Both{+layout_low,+layout_middle} = left_ends
      Held.Both{+right_first_middle,+right_rest} = right_facts
      Held.Both{+right_end,+right_more} = right_rest
      Held.Both{+right_run,+right_ends} = right_more
      Held.Both{+right_layout_middle,+layout_high} = right_ends
      +a = U32.to_nat(left_first)
      +b = U32.to_nat(left_count)
      +c = U32.to_nat(right_count)
      +high_inside = Held.nat_le_moved(high,range_high(right_layout),bound,Equal.sym(Nat,range_high(right_layout),high,layout_high),
        fits_inside(unit,range_low(right_layout),range_high(right_layout),bound,run_fits(right_layout,unit,bound,right_run)))
      +reached = Equal.trans(Nat,Nat.add(a,Nat.add(b,c)),Nat.add(Nat.add(a,b),c),high,Equal.sym(Nat,Nat.add(Nat.add(a,b),c),Nat.add(a,Nat.add(b,c)),Addition.associative(a,b,c)),
        Equal.trans(Nat,Nat.add(Nat.add(a,b),c),Nat.add(U32.to_nat(right_first),c),high,
          Equal.trans(Nat,Nat.add(Nat.add(a,b),c),Nat.add(middle,c),Nat.add(U32.to_nat(right_first),c),Equal.cong(Nat,Nat,value => Nat.add(value,c),Nat.add(a,b),middle,left_end),
            Equal.cong(Nat,Nat,value => Nat.add(value,c),middle,U32.to_nat(right_first),Equal.sym(Nat,U32.to_nat(right_first),middle,right_first_middle))),right_end))
      +sum_room = Order.transitive(Nat.add(b,c),Nat.add(a,Nat.add(b,c)),U32.to_nat(Machine.capacity(limits)),Order.right_le(a,Nat.add(b,c)),
        Held.nat_le_moved(Nat.add(a,Nat.add(b,c)),high,U32.to_nat(Machine.capacity(limits)),reached,Order.transitive(high,bound,U32.to_nat(Machine.capacity(limits)),high_inside,room)))
      +sum = Arithmetic.add(left_count,right_count,Machine.capacity(limits),LimitBounds.capacity_fits(limits),sum_room)
      +ended = Equal.trans(Nat,Nat.add(a,U32.to_nat((left_count + right_count : U32))),Nat.add(a,Nat.add(b,c)),high,
        Equal.cong(Nat,Nat,value => Nat.add(a,value),U32.to_nat((left_count + right_count : U32)),Nat.add(b,c),sum),reached)
      Held.Both{first_low,Held.Both{ended,layout_joined(left_first,left_count,right_count,left_layout,right_layout,low,middle,high,unit,bound,
        U32.is_le((left_count + right_count : U32),limit),left_run,right_run,first_low,layout_low,layout_middle,right_layout_middle,layout_high,ended)}}

# Folding a run into groups keeps the run and its ends.
def grouped(+limits: Machine.Limits,+limit: U32,ranges: Pieces.Ranges,+unit: Nat,+bound: Nat,+run: Run(ranges,unit,bound),
  +room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  GroupRun(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,ranges,limit),range_low(ranges),range_high(ranges),unit,bound):
  match ranges:
    case Pieces.Range{+first,+count}: Held.Both{{==},Held.Both{{==},Held.Both{run,Held.Both{{==},{==}}}}}
    case Pieces.Split{+left,+right}:
      Held.Both{+left_run,+rest} = run
      Held.Both{+right_run,+glue} = rest
      +right_group = Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,right,limit)
      joined(limits,limit,left,right,Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,left,limit),right_group,
        range_low(left),range_high(left),range_high(right),unit,bound,grouped(limits,limit,left,unit,bound,left_run,room),
        {%Equal.sym(Nat,range_high(left),range_low(right),glue) : GroupRun(right_group,_,range_high(right),unit,bound)
          grouped(limits,limit,right,unit,bound,right_run,room) : GroupRun(right_group,range_high(left),range_high(right),unit,bound)},room)

def layout_of(+group: Tasks.Group,+low: Nat,+high: Nat,+unit: Nat,+bound: Nat,+facts: GroupRun(group,low,high,unit,bound)) -> Run(Tasks.group_layout(group),unit,bound):
  match group:
    case Tasks.Group{+first,+count,+layout}:
      Held.Both{+first_low,+rest} = facts
      Held.Both{+ended,+more} = rest
      Held.Both{+run,+ends} = more
      run

# The tasks of a size over a run are a run.
def sized_run(+limits: Machine.Limits,+layout: Pieces.Ranges,+count: U32,+pool: U32,+whole: U32,+found: Tasks.Size,+unit: Nat,+bound: Nat,+run: Run(layout,unit,bound),
  +room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> Run(Tasks.tasks_sized(layout,count,pool,whole,found),unit,bound):
  match found:
    case Tasks.Size{+ranges,+largest}:
      +limit = U32.max(U32.max(Tasks.divided_up(count,pool),whole),(Tasks.divided_up(ranges,pool) * largest : U32))
      layout_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,layout,limit),range_low(layout),range_high(layout),unit,bound,
        grouped(limits,limit,layout,unit,bound,run,room))

# The tasks over a run are a run.
def tasks_run(+limits: Machine.Limits,+layout: Pieces.Ranges,+count: U32,+workers: U32,+whole: U32,+unit: Nat,+bound: Nat,+run: Run(layout,unit,bound),
  +room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> Run(Tasks.tasks_over(layout,count,workers,whole),unit,bound):
  sized_run(limits,layout,count,U32.max(workers,1),whole,Tasks.size(layout),unit,bound,run,room)

# The tasks of the pool a decision picks over a run are a run.
def waved_run(+limits: Machine.Limits,+layout: Pieces.Ranges,+count: U32,+workers: U32,+whole: U32,+cell_ns: F32,+config: Cost.Configuration,+extra: Tasks.Extra,
  +unit: Nat,+bound: Nat,+run: Run(layout,unit,bound),+room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Run(Tasks.tasks_waved(layout,count,workers,whole,cell_ns,config,extra),unit,bound):
  sized_run(limits,layout,count,Tasks.waved_pool(workers,cell_ns,config,extra,count,whole,Tasks.size(layout)),whole,Tasks.size(layout),unit,bound,run,room)

# A run fits inside any larger bound.
def run_within(ranges: Pieces.Ranges,+unit: Nat,+bound: Nat,+wider: Nat,+widened: {Nat.is_le(bound,wider) == True{} : Bool},+run: Run(ranges,unit,bound)) ->
  Run(ranges,unit,wider):
  match ranges:
    case Pieces.Range{+first,+count}:
      Fits{+low_units,+high_units,+low_exact,+high_exact,+less,+inside} = run
      Fits{low_units,high_units,low_exact,high_exact,less,Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(count)),bound,wider,inside,widened)}
    case Pieces.Split{+left,+right}:
      Held.Both{+left_run,+rest} = run
      Held.Both{+right_run,+glue} = rest
      Held.Both{run_within(left,unit,bound,wider,widened,left_run),Held.Both{run_within(right,unit,bound,wider,widened,right_run),glue}}

# Spans that fit, each starting where the one before ends, from low on.
def Chain(spans: Held.Spans,+low: Nat,+unit: Nat,+bound: Nat) -> Data:
  match spans:
    case Held.Ended{}: {low == low : Nat}
    case Held.Spanned{+first,+count,+rest}:
      Held.Both<Fits<unit,U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),bound>,
        Held.Both<{U32.to_nat(first) == low : Nat},Chain(rest,Nat.add(U32.to_nat(first),U32.to_nat(count)),unit,bound)>>

def part_run(+ranges: Pieces.Ranges,+rest: Held.Spans,+low: Nat,+unit: Nat,+bound: Nat,+facts: Held.Both<Run(ranges,unit,bound),Held.Both<{range_low(ranges) == low : Nat},Chain(rest,range_high(ranges),unit,bound)>>) -> Run(ranges,unit,bound):
  Held.Both{+run,+more} = facts
  run

def part_low(+ranges: Pieces.Ranges,+rest: Held.Spans,+low: Nat,+unit: Nat,+bound: Nat,+facts: Held.Both<Run(ranges,unit,bound),Held.Both<{range_low(ranges) == low : Nat},Chain(rest,range_high(ranges),unit,bound)>>) -> {range_low(ranges) == low : Nat}:
  Held.Both{+run,+more} = facts
  Held.Both{+starts,+next} = more
  starts

def part_chain(+ranges: Pieces.Ranges,+rest: Held.Spans,+low: Nat,+unit: Nat,+bound: Nat,+facts: Held.Both<Run(ranges,unit,bound),Held.Both<{range_low(ranges) == low : Nat},Chain(rest,range_high(ranges),unit,bound)>>) -> Chain(rest,range_high(ranges),unit,bound):
  Held.Both{+run,+more} = facts
  Held.Both{+starts,+next} = more
  next

# A run of ranges before a chain from its end is a chain from its start.
def run_chain(ranges: Pieces.Ranges,+rest: Held.Spans,+unit: Nat,+bound: Nat,+run: Run(ranges,unit,bound),+after: Chain(rest,range_high(ranges),unit,bound)) ->
  Chain(Held.layout_spans(ranges,rest),range_low(ranges),unit,bound):
  match ranges:
    case Pieces.Range{+first,+count}: Held.Both{run,Held.Both{{==},after}}
    case Pieces.Split{+left,+right}:
      Held.Both{+left_run,+more} = run
      Held.Both{+right_run,+glue} = more
      run_chain(left,Held.layout_spans(right,rest),unit,bound,left_run,
        {%Equal.sym(Nat,range_high(left),range_low(right),glue) : Chain(Held.layout_spans(right,rest),_,unit,bound)
          run_chain(right,rest,unit,bound,right_run,after) : Chain(Held.layout_spans(right,rest),range_high(left),unit,bound)})

# Ranges whose spans in order form a chain are a run.
def tree_run(ranges: Pieces.Ranges,+rest: Held.Spans,+low: Nat,+unit: Nat,+bound: Nat,+chain: Chain(Held.layout_spans(ranges,rest),low,unit,bound)) ->
  Held.Both<Run(ranges,unit,bound),Held.Both<{range_low(ranges) == low : Nat},Chain(rest,range_high(ranges),unit,bound)>>:
  match ranges:
    case Pieces.Range{+first,+count}:
      Held.Both{+fits,+more} = chain
      Held.Both{+starts,+next} = more
      Held.Both{fits,Held.Both{starts,next}}
    case Pieces.Split{+left,+right}:
      +left_facts = tree_run(left,Held.layout_spans(right,rest),low,unit,bound,chain)
      +middle = range_high(left)
      +right_facts = tree_run(right,rest,middle,unit,bound,part_chain(left,Held.layout_spans(right,rest),low,unit,bound,left_facts))
      Held.Both{Held.Both{part_run(left,Held.layout_spans(right,rest),low,unit,bound,left_facts),Held.Both{part_run(right,rest,middle,unit,bound,right_facts),
        Equal.sym(Nat,range_low(right),middle,part_low(right,rest,middle,unit,bound,right_facts))}},
        Held.Both{part_low(left,Held.layout_spans(right,rest),low,unit,bound,left_facts),part_chain(right,rest,middle,unit,bound,right_facts)}}

# ---- The decision ----

def first_low(ranges: Pieces.Ranges) -> {U32.to_nat(Pieces.first(ranges)) == range_low(ranges) : Nat}:
  match ranges:
    case Pieces.Range{+first,+count}: {==}
    case Pieces.Split{+left,+right}: first_low(left)

def last_high(+limits: Machine.Limits,ranges: Pieces.Ranges,+unit: Nat,+bound: Nat,+run: Run(ranges,unit,bound),
  +room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {U32.to_nat(Pieces.last(ranges)) == range_high(ranges) : Nat}:
  match ranges:
    case Pieces.Range{+first,+count}:
      Arithmetic.add(first,count,Machine.capacity(limits),LimitBounds.capacity_fits(limits),
        Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(count)),bound,U32.to_nat(Machine.capacity(limits)),
          fits_inside(unit,U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),bound,run),room))
    case Pieces.Split{+left,+right}:
      Held.Both{+left_run,+rest} = run
      Held.Both{+right_run,+glue} = rest
      last_high(limits,right,unit,bound,right_run,room)

def fits_less(+unit: Nat,+low: Nat,+high: Nat,+bound: Nat,+fits: Fits<unit,low,high,bound>) -> {Nat.is_lt(low,high) == True{} : Bool}:
  Fits{+low_units,+high_units,+low_exact,+high_exact,+less,+inside} = fits
  less

# The one range from the first cell of a run to its end.
# The single range from the first to the last cell of a run ends where the run ends.
def whole_end(+limits: Machine.Limits,+layout: Pieces.Ranges,+unit: Nat,+bound: Nat,+run: Run(layout,unit,bound),
  +room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.add(U32.to_nat(Pieces.first(layout)),U32.to_nat((Pieces.last(layout) - Pieces.first(layout) : U32))) == range_high(layout) : Nat}:
  +low = U32.to_nat(Pieces.first(layout))
  +high = U32.to_nat(Pieces.last(layout))
  +fits = run_fits(layout,unit,bound,run)
  +low_exact = first_low(layout)
  +high_exact = last_high(limits,layout,unit,bound,run,room)
  +ordered = Bounds.strict_le(low,high,Held.nat_lt_left(low,range_low(layout),high,low_exact,
    Held.nat_lt_moved(range_low(layout),high,range_high(layout),high_exact,fits_less(unit,range_low(layout),range_high(layout),bound,fits))))
  +high_room = Held.nat_le_moved(high,range_high(layout),U32.to_nat(Machine.capacity(limits)),high_exact,
    Order.transitive(range_high(layout),bound,U32.to_nat(Machine.capacity(limits)),fits_inside(unit,range_low(layout),range_high(layout),bound,fits),room))
  +difference = Arithmetic.subtract(Pieces.last(layout),Pieces.first(layout),Machine.capacity(limits),LimitBounds.capacity_fits(limits),ordered,
    Order.transitive(Nat.sub(high,low),high,U32.to_nat(Machine.capacity(limits)),Order.sub_le(high,low),high_room))
  Equal.trans(Nat,Nat.add(low,U32.to_nat((Pieces.last(layout) - Pieces.first(layout) : U32))),Nat.add(low,Nat.sub(high,low)),range_high(layout),
    Equal.cong(Nat,Nat,value => Nat.add(low,value),U32.to_nat((Pieces.last(layout) - Pieces.first(layout) : U32)),Nat.sub(high,low),difference),
    Equal.trans(Nat,Nat.add(low,Nat.sub(high,low)),high,range_high(layout),Order.restore_shift(low,high,ordered),high_exact))

def whole_run(+limits: Machine.Limits,+layout: Pieces.Ranges,+unit: Nat,+bound: Nat,+run: Run(layout,unit,bound),
  +room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> Run(Tasks.whole_bounds(Pieces.bounds(layout)),unit,bound):
  +low = U32.to_nat(Pieces.first(layout))
  +fits = run_fits(layout,unit,bound,run)
  +low_exact = first_low(layout)
  +ended = whole_end(limits,layout,unit,bound,run,room)
  {%Equal.sym(Nat,low,range_low(layout),low_exact) : Fits<unit,_,Nat.add(low,U32.to_nat((Pieces.last(layout) - Pieces.first(layout) : U32))),bound>
    {%Equal.sym(Nat,Nat.add(low,U32.to_nat((Pieces.last(layout) - Pieces.first(layout) : U32))),range_high(layout),ended) : Fits<unit,range_low(layout),_,bound>
      fits : Fits<unit,range_low(layout),Nat.add(low,U32.to_nat((Pieces.last(layout) - Pieces.first(layout) : U32))),bound>} :
    Fits<unit,low,Nat.add(low,U32.to_nat((Pieces.last(layout) - Pieces.first(layout) : U32))),bound>}

def decision_tasks(+decision: Tasks.Decision) -> Pieces.Ranges:
  Tasks.Decision{+parallel,+tasks} = decision
  tasks

def planned_run(+limits: Machine.Limits,+parallel: Bool,+layout: Pieces.Ranges,+unit: Nat,+bound: Nat,+run: Run(layout,unit,bound),
  +room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> Run(decision_tasks(Tasks.planned(parallel,layout)),unit,bound):
  match parallel:
    case True{}: run
    case False{}: whole_run(limits,layout,unit,bound,run,room)

# The tasks an operation decides over a run of the pieces of a grid are a run.
def decided_run(+limits: Machine.Limits,+grid: Grid.Grid,+workers: U32,+whole: U32,+bytes_per_cell: F32,+operations: F32,+extra: Tasks.Extra,+unit: Nat,+bound: Nat,
  +run: Run(Pieces.ranges_of_grid(grid),unit,bound),+room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Run(decision_tasks(Tasks.decide_whole(grid,workers,whole,bytes_per_cell,operations,extra)),unit,bound):
  Grid.Grid{+count,+prefix,+outer,+extent,+unit_cells,+piece_units,+per_outer,+pieces} = grid
  +layout = Tasks.tasks_waved(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,piece_units,per_outer,pieces}),count,workers,whole,Cost.cell_ns(bytes_per_cell,operations,DefaultCost.configuration()),DefaultCost.configuration(),extra)
  planned_run(limits,Tasks.worth(workers,Cost.cell_ns(bytes_per_cell,operations,DefaultCost.configuration()),DefaultCost.configuration(),extra,count,Tasks.size(layout)),
    layout,unit,bound,waved_run(limits,Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,piece_units,per_outer,pieces}),count,workers,whole,Cost.cell_ns(bytes_per_cell,operations,DefaultCost.configuration()),DefaultCost.configuration(),extra,unit,bound,run,room),room)

# The ranges of a run cover the cells from its low to its high once each.
def run_coverage(ranges: Pieces.Ranges,+unit: Nat,+bound: Nat,+run: Run(ranges,unit,bound)) ->
  PiecesPlan.Coverage(ranges,range_low(ranges),Nat.sub(range_high(ranges),range_low(ranges))):
  match ranges:
    case Pieces.Range{+first,+count}:
      PiecesPlan.LeafCoverage{{==},Equal.sym(Nat,Nat.sub(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(first)),U32.to_nat(count),
        Order.subtract_shift(U32.to_nat(first),U32.to_nat(count)))}
    case Pieces.Split{+left,+right}:
      Held.Both{+left_run,+run_rest} = run
      Held.Both{+right_run,+glue} = run_rest
      +left_low = range_low(left)
      +left_high = range_high(left)
      +right_low = range_low(right)
      +right_high = range_high(right)
      +left_ordered = Bounds.strict_le(left_low,left_high,fits_less(unit,left_low,left_high,bound,run_fits(left,unit,bound,left_run)))
      +right_ordered = Bounds.strict_le(right_low,right_high,fits_less(unit,right_low,right_high,bound,run_fits(right,unit,bound,right_run)))
      +left_rows = Nat.sub(left_high,left_low)
      +right_rows = Nat.sub(right_high,right_low)
      +middle = Equal.trans(Nat,Nat.add(left_low,left_rows),left_high,right_low,Order.restore_shift(left_low,left_high,left_ordered),glue)
      +whole = Equal.trans(Nat,Nat.add(Nat.add(left_low,left_rows),right_rows),Nat.add(right_low,right_rows),right_high,
        Equal.cong(Nat,Nat,value => Nat.add(value,right_rows),Nat.add(left_low,left_rows),right_low,middle),Order.restore_shift(right_low,right_high,right_ordered))
      +rows = Equal.trans(Nat,Nat.sub(right_high,left_low),Nat.sub(Nat.add(left_low,Nat.add(left_rows,right_rows)),left_low),Nat.add(left_rows,right_rows),
        Equal.cong(Nat,Nat,value => Nat.sub(value,left_low),right_high,Nat.add(left_low,Nat.add(left_rows,right_rows)),
          Equal.sym(Nat,Nat.add(left_low,Nat.add(left_rows,right_rows)),right_high,
            Equal.trans(Nat,Nat.add(left_low,Nat.add(left_rows,right_rows)),Nat.add(Nat.add(left_low,left_rows),right_rows),right_high,
              Equal.sym(Nat,Nat.add(Nat.add(left_low,left_rows),right_rows),Nat.add(left_low,Nat.add(left_rows,right_rows)),Addition.associative(left_low,left_rows,right_rows)),whole))),
        Order.subtract_shift(left_low,Nat.add(left_rows,right_rows)))
      PiecesPlan.ForkCoverage{left_rows,right_rows,rows,run_coverage(left,unit,bound,left_run),
        {%Equal.sym(Nat,Nat.add(left_low,left_rows),right_low,middle) : PiecesPlan.Coverage(right,_,right_rows)
          run_coverage(right,unit,bound,right_run) : PiecesPlan.Coverage(right,Nat.add(left_low,left_rows),right_rows)}}

# A piece whose buffer holds as many cells as its rows.
def piece_sized(~Element: Data,+first: U32,+rows: U32,+buffer: Buffer.Content<Element>) -> Data:
  {Buffer.content_length(~Element,buffer) == U32.to_nat(rows) : Nat}

# Pieces laid as a run, each holding as many cells as its rows, are pieces
# of the cells up to the bound of the run.
def content_fields(~Element: Data,content: PiecesModel.Content<Element>,+unit: Nat,+bound: Nat,+run: Run(PiecesModel.content_layout(~Element,content),unit,bound),
  +sized: PiecesModel.ContentAll(~Element,~piece_sized(~Element),content)) -> Content.Contents(~Element,bound,content):
  match content:
    case PiecesModel.PieceContent{+buffer,+first,+rows}:
      Content.Fields{sized,fits_inside(unit,U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(rows)),bound,run)}
    case PiecesModel.Joined{+left,+right}:
      Held.Both{+left_run,+run_rest} = run
      Held.Both{+right_run,+glue} = run_rest
      KernelGemm.Both{+left_sized,+right_sized} = sized
      KernelGemm.Both{content_fields(~Element,left,unit,bound,left_run,left_sized),content_fields(~Element,right,unit,bound,right_run,right_sized)}

# ---- Pieces cut between tasks ----

# The left part of a cut range of at least two units: whole units, not empty
# and less than the range.
type CutLeft<-count: U32,-cut_unit: U32> is Data:
  CutLeft{quotient: Nat,exact: {U32.to_nat((count / cut_unit / 2 * cut_unit : U32)) == Nat.mul(quotient,U32.to_nat(cut_unit)) : Nat},
    positive: {Nat.is_lt(0n,Nat.mul(quotient,U32.to_nat(cut_unit))) == True{} : Bool},less: {Nat.is_lt(Nat.mul(quotient,U32.to_nat(cut_unit)),U32.to_nat(count)) == True{} : Bool}}

def cut_left(+limits: Machine.Limits,+count: U32,+cut_unit: U32,+unit_positive: {Nat.is_lt(0n,U32.to_nat(cut_unit)) == True{} : Bool},
  +unit_room: {Nat.is_le(U32.to_nat(cut_unit),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+count_room: {Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +two: {Nat.is_le(Nat.add(U32.to_nat(cut_unit),U32.to_nat(cut_unit)),U32.to_nat(count)) == True{} : Bool}) -> CutLeft<count,cut_unit>:
  +quotient_exact = U32Division.divide(count,cut_unit,Machine.half_word(limits),LimitBounds.half_word_fits(limits),LimitBounds.double_bound(limits,U32.to_nat(cut_unit),unit_room),unit_positive)
  +half_exact = Equal.trans(Nat,U32.to_nat((count / cut_unit / 2 : U32)),Nat.div(U32.to_nat((count / cut_unit : U32)),2n),Nat.div(Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),2n),U32Division.divide((count / cut_unit : U32),2,4,{==},{==},{==}),
    Equal.cong(Nat,Nat,number => Nat.div(number,2n),U32.to_nat((count / cut_unit : U32)),Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),quotient_exact))
  +doubled = {%Equal.cong(Nat,Nat,number => Nat.add(U32.to_nat(cut_unit),number),U32.to_nat(cut_unit),Nat.add(U32.to_nat(cut_unit),0n),Equal.sym(Nat,Nat.add(U32.to_nat(cut_unit),0n),U32.to_nat(cut_unit),Addition.zero_right(U32.to_nat(cut_unit)))) :
      {Nat.is_le(_,U32.to_nat(count)) == True{} : Bool}
    two : {Nat.is_le(Nat.mul(2n,U32.to_nat(cut_unit)),U32.to_nat(count)) == True{} : Bool}}
  +two_quotient = Division.lt_succ_le(2n,Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),RowMajor.cancel_multiply_lt(2n,1n+Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),U32.to_nat(cut_unit),
    Order.le_lt_trans(Nat.mul(2n,U32.to_nat(cut_unit)),U32.to_nat(count),Nat.mul(1n+Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),U32.to_nat(cut_unit)),doubled,DivisionOrder.next_above(U32.to_nat(count),U32.to_nat(cut_unit),unit_positive))))
  +half_positive = Order.succ_le_lt(0n,Nat.div(Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),2n),DivisionOrder.quotient_le(2n,Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),2n,{==},two_quotient))
  +half_below = Order.lt_le(Nat.div(Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),2n),Nat.mul(Nat.div(Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),2n),2n),Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),
    {%RowMajor.multiply_commutative(2n,Nat.div(Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),2n)) : {Nat.is_lt(Nat.div(Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),2n),_) == True{} : Bool}
      %Addition.zero_right(Nat.div(Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),2n)) : {Nat.is_lt(_,Nat.mul(2n,Nat.div(Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),2n))) == True{} : Bool}
      Order.offset_lt(Nat.div(Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),2n),0n,Nat.add(Nat.div(Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),2n),0n),
        {%Equal.sym(Nat,Nat.add(Nat.div(Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),2n),0n),Nat.div(Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),2n),Addition.zero_right(Nat.div(Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),2n))) : {Nat.is_lt(0n,_) == True{} : Bool}
          half_positive : {Nat.is_lt(0n,Nat.add(Nat.div(Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),2n),0n)) == True{} : Bool}}) :
      {Nat.is_lt(Nat.div(Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),2n),Nat.mul(Nat.div(Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),2n),2n)) == True{} : Bool}},
    DivisionOrder.product_below(Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),2n,{==}))
  +less = Order.lt_le(Nat.mul(Nat.div(Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),2n),U32.to_nat(cut_unit)),Nat.mul(Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),U32.to_nat(cut_unit)),U32.to_nat(count),RowMajor.multiply_lt(Nat.div(Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),2n),Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),U32.to_nat(cut_unit),half_below,unit_positive),DivisionOrder.product_below(U32.to_nat(count),U32.to_nat(cut_unit),unit_positive))
  CutLeft{Nat.div(Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),2n),
    Arithmetic.product_exact(Machine.capacity(limits),LimitBounds.capacity_fits(limits),(count / cut_unit / 2 : U32),cut_unit,Nat.div(Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),2n),U32.to_nat(cut_unit),half_exact,{==},
      Order.transitive(Nat.mul(Nat.div(Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),2n),U32.to_nat(cut_unit)),U32.to_nat(count),U32.to_nat(Machine.capacity(limits)),Bounds.strict_le(Nat.mul(Nat.div(Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),2n),U32.to_nat(cut_unit)),U32.to_nat(count),less),count_room)),
    RowMajor.multiply_lt(0n,Nat.div(Nat.div(U32.to_nat(count),U32.to_nat(cut_unit)),2n),U32.to_nat(cut_unit),half_positive,unit_positive),less}

# The ranges of a cut and where they start and end.
def CutRun(ranges: Pieces.Ranges,+first: U32,+count: U32,+unit: Nat,+bound: Nat) -> Data:
  Held.Both<Run(ranges,unit,bound),Held.Both<{range_low(ranges) == U32.to_nat(first) : Nat},{range_high(ranges) == Nat.add(U32.to_nat(first),U32.to_nat(count)) : Nat}>>

def next_exact_left(+first: U32,+left: U32,+next: U32,+value: Nat,+next_exact: {U32.to_nat(next) == Nat.add(U32.to_nat(first),value) : Nat},
  +left_exact: {U32.to_nat(left) == value : Nat}) -> {U32.to_nat(next) == Nat.add(U32.to_nat(first),U32.to_nat(left)) : Nat}:
  Equal.trans(Nat,U32.to_nat(next),Nat.add(U32.to_nat(first),value),Nat.add(U32.to_nat(first),U32.to_nat(left)),next_exact,
    Equal.cong(Nat,Nat,number => Nat.add(U32.to_nat(first),number),value,U32.to_nat(left),Equal.sym(Nat,U32.to_nat(left),value,left_exact)))

# Two cut runs that meet are one run from the first one's start to the second
# one's end.
def cut_joined(left_ranges: Pieces.Ranges,right_ranges: Pieces.Ranges,+first: U32,+left: U32,+next: U32,+right: U32,+unit: Nat,+bound: Nat,+high: Nat,
  +next_exact: {U32.to_nat(next) == Nat.add(U32.to_nat(first),U32.to_nat(left)) : Nat},+ends: {Nat.add(U32.to_nat(next),U32.to_nat(right)) == high : Nat},
  +left_run: CutRun(left_ranges,first,left,unit,bound),+right_run: CutRun(right_ranges,next,right,unit,bound)) ->
  Held.Both<Run(Pieces.Split{left_ranges,right_ranges},unit,bound),Held.Both<{range_low(Pieces.Split{left_ranges,right_ranges}) == U32.to_nat(first) : Nat},
    {range_high(Pieces.Split{left_ranges,right_ranges}) == high : Nat}>>:
  Held.Both{+left_ranges_run,+left_ends} = left_run
  Held.Both{+left_low,+left_high} = left_ends
  Held.Both{+right_ranges_run,+right_ends} = right_run
  Held.Both{+right_low,+right_high} = right_ends
  Held.Both{Held.Both{left_ranges_run,Held.Both{right_ranges_run,
      Equal.trans(Nat,range_high(left_ranges),Nat.add(U32.to_nat(first),U32.to_nat(left)),range_low(right_ranges),left_high,
        Equal.sym(Nat,range_low(right_ranges),Nat.add(U32.to_nat(first),U32.to_nat(left)),Equal.trans(Nat,range_low(right_ranges),U32.to_nat(next),Nat.add(U32.to_nat(first),U32.to_nat(left)),right_low,next_exact)))}},
    Held.Both{left_low,Equal.trans(Nat,range_high(right_ranges),Nat.add(U32.to_nat(next),U32.to_nat(right)),high,right_high,ends)}}

# The facts of the two halves of a cut range.
type Halves<-first: U32,-count: U32,-cut_unit: U32,-unit: Nat,-bound: Nat> is Data:
  Halves{left: Fits<unit,U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat((count / cut_unit / 2 * cut_unit : U32))),bound>,right: Fits<unit,U32.to_nat((first + (count / cut_unit / 2 * cut_unit : U32) : U32)),Nat.add(U32.to_nat((first + (count / cut_unit / 2 * cut_unit : U32) : U32)),U32.to_nat((count - (count / cut_unit / 2 * cut_unit : U32) : U32))),bound>,
    next: {U32.to_nat((first + (count / cut_unit / 2 * cut_unit : U32) : U32)) == Nat.add(U32.to_nat(first),U32.to_nat((count / cut_unit / 2 * cut_unit : U32))) : Nat},ends: {Nat.add(U32.to_nat((first + (count / cut_unit / 2 * cut_unit : U32) : U32)),U32.to_nat((count - (count / cut_unit / 2 * cut_unit : U32) : U32))) == Nat.add(U32.to_nat(first),U32.to_nat(count)) : Nat}}

# The two halves of a cut range, from its left part.
def halves_of(+limits: Machine.Limits,+cut_unit: U32,+cell_ns: F32,+grain: F32,+unit: Nat,+bound: Nat,+multiple: Nat,
  +unit_exact: {U32.to_nat(cut_unit) == Nat.mul(multiple,unit) : Nat},+unit_positive: {Nat.is_lt(0n,U32.to_nat(cut_unit)) == True{} : Bool},
  +unit_room: {Nat.is_le(U32.to_nat(cut_unit),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+first: U32,+count: U32,+low_units: Nat,+high_units: Nat,+low_exact: {U32.to_nat(first) == Nat.mul(low_units,unit) : Nat},
  +high_exact: {Nat.add(U32.to_nat(first),U32.to_nat(count)) == Nat.mul(high_units,unit) : Nat},+inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),bound) == True{} : Bool},
  +count_room: {Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+left: CutLeft<count,cut_unit>) -> Halves<first,count,cut_unit,unit,bound>:
  CutLeft{+half,+left_exact,+left_positive,+left_less} = left
  +cap = U32.to_nat(Machine.capacity(limits))
  +value = Nat.mul(half,U32.to_nat(cut_unit))
  +left_room = Order.transitive(Nat.add(U32.to_nat(first),value),Nat.add(U32.to_nat(first),U32.to_nat(count)),cap,Bounds.strict_le(Nat.add(U32.to_nat(first),value),Nat.add(U32.to_nat(first),U32.to_nat(count)),Order.offset_lt(U32.to_nat(first),value,U32.to_nat(count),left_less)),
    Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(count)),bound,cap,inside,room))
  +next_exact = Equal.trans(Nat,U32.to_nat((first + (count / cut_unit / 2 * cut_unit : U32) : U32)),Nat.add(U32.to_nat(first),U32.to_nat((count / cut_unit / 2 * cut_unit : U32))),Nat.add(U32.to_nat(first),value),
    Arithmetic.add(first,(count / cut_unit / 2 * cut_unit : U32),Machine.capacity(limits),LimitBounds.capacity_fits(limits),
      {%Equal.sym(Nat,U32.to_nat((count / cut_unit / 2 * cut_unit : U32)),value,left_exact) : {Nat.is_le(Nat.add(U32.to_nat(first),_),cap) == True{} : Bool} left_room : {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat((count / cut_unit / 2 * cut_unit : U32))),cap) == True{} : Bool}}),
    Equal.cong(Nat,Nat,number => Nat.add(U32.to_nat(first),number),U32.to_nat((count / cut_unit / 2 * cut_unit : U32)),value,left_exact))
  +ordered = Bounds.strict_le(value,U32.to_nat(count),left_less)
  +right_exact = Arithmetic.subtract(count,(count / cut_unit / 2 * cut_unit : U32),Machine.capacity(limits),LimitBounds.capacity_fits(limits),
    {%Equal.sym(Nat,U32.to_nat((count / cut_unit / 2 * cut_unit : U32)),value,left_exact) : {Nat.is_le(_,U32.to_nat(count)) == True{} : Bool} ordered : {Nat.is_le(U32.to_nat((count / cut_unit / 2 * cut_unit : U32)),U32.to_nat(count)) == True{} : Bool}},
    Order.transitive(Nat.sub(U32.to_nat(count),U32.to_nat((count / cut_unit / 2 * cut_unit : U32))),U32.to_nat(count),cap,Order.sub_le(U32.to_nat(count),U32.to_nat((count / cut_unit / 2 * cut_unit : U32))),count_room))
  +right_value = Equal.trans(Nat,U32.to_nat((count - (count / cut_unit / 2 * cut_unit : U32) : U32)),Nat.sub(U32.to_nat(count),U32.to_nat((count / cut_unit / 2 * cut_unit : U32))),Nat.sub(U32.to_nat(count),value),right_exact,
    Equal.cong(Nat,Nat,number => Nat.sub(U32.to_nat(count),number),U32.to_nat((count / cut_unit / 2 * cut_unit : U32)),value,left_exact))
  +middle_units = Nat.add(low_units,Nat.mul(half,multiple))
  +middle_exact = Equal.trans(Nat,Nat.add(U32.to_nat(first),value),Nat.add(Nat.mul(low_units,unit),Nat.mul(Nat.mul(half,multiple),unit)),Nat.mul(middle_units,unit),
    Equal.trans(Nat,Nat.add(U32.to_nat(first),value),Nat.add(Nat.mul(low_units,unit),value),Nat.add(Nat.mul(low_units,unit),Nat.mul(Nat.mul(half,multiple),unit)),
      Equal.cong(Nat,Nat,number => Nat.add(number,value),U32.to_nat(first),Nat.mul(low_units,unit),low_exact),
      Equal.cong(Nat,Nat,number => Nat.add(Nat.mul(low_units,unit),number),value,Nat.mul(Nat.mul(half,multiple),unit),
        Equal.trans(Nat,value,Nat.mul(half,Nat.mul(multiple,unit)),Nat.mul(Nat.mul(half,multiple),unit),
          Equal.cong(Nat,Nat,number => Nat.mul(half,number),U32.to_nat(cut_unit),Nat.mul(multiple,unit),unit_exact),
          Equal.sym(Nat,Nat.mul(Nat.mul(half,multiple),unit),Nat.mul(half,Nat.mul(multiple,unit)),RowMajor.assoc(half,multiple,unit))))),
    Equal.sym(Nat,Nat.mul(middle_units,unit),Nat.add(Nat.mul(low_units,unit),Nat.mul(Nat.mul(half,multiple),unit)),RowMajor.distrib(low_units,Nat.mul(half,multiple),unit)))
  +restored = Equal.trans(Nat,Nat.add(Nat.add(U32.to_nat(first),value),Nat.sub(U32.to_nat(count),value)),Nat.add(U32.to_nat(first),Nat.add(value,Nat.sub(U32.to_nat(count),value))),Nat.add(U32.to_nat(first),U32.to_nat(count)),
    Addition.associative(U32.to_nat(first),value,Nat.sub(U32.to_nat(count),value)),
    Equal.cong(Nat,Nat,number => Nat.add(U32.to_nat(first),number),Nat.add(value,Nat.sub(U32.to_nat(count),value)),U32.to_nat(count),Order.restore_shift(value,U32.to_nat(count),ordered)))
  +right_end = Equal.trans(Nat,Nat.add(U32.to_nat((first + (count / cut_unit / 2 * cut_unit : U32) : U32)),U32.to_nat((count - (count / cut_unit / 2 * cut_unit : U32) : U32))),Nat.add(Nat.add(U32.to_nat(first),value),Nat.sub(U32.to_nat(count),value)),Nat.add(U32.to_nat(first),U32.to_nat(count)),
    Equal.trans(Nat,Nat.add(U32.to_nat((first + (count / cut_unit / 2 * cut_unit : U32) : U32)),U32.to_nat((count - (count / cut_unit / 2 * cut_unit : U32) : U32))),Nat.add(Nat.add(U32.to_nat(first),value),U32.to_nat((count - (count / cut_unit / 2 * cut_unit : U32) : U32))),Nat.add(Nat.add(U32.to_nat(first),value),Nat.sub(U32.to_nat(count),value)),
      Equal.cong(Nat,Nat,number => Nat.add(number,U32.to_nat((count - (count / cut_unit / 2 * cut_unit : U32) : U32))),U32.to_nat((first + (count / cut_unit / 2 * cut_unit : U32) : U32)),Nat.add(U32.to_nat(first),value),next_exact),
      Equal.cong(Nat,Nat,number => Nat.add(Nat.add(U32.to_nat(first),value),number),U32.to_nat((count - (count / cut_unit / 2 * cut_unit : U32) : U32)),Nat.sub(U32.to_nat(count),value),right_value)),
    restored)
  +left_fits = {%Equal.sym(Nat,U32.to_nat((count / cut_unit / 2 * cut_unit : U32)),value,left_exact) : Fits<unit,U32.to_nat(first),Nat.add(U32.to_nat(first),_),bound>
    Fits{low_units,middle_units,low_exact,middle_exact,
      {%Addition.zero_right(U32.to_nat(first)) : {Nat.is_lt(_,Nat.add(U32.to_nat(first),value)) == True{} : Bool} Order.offset_lt(U32.to_nat(first),0n,value,left_positive) :
        {Nat.is_lt(U32.to_nat(first),Nat.add(U32.to_nat(first),value)) == True{} : Bool}},
      Order.transitive(Nat.add(U32.to_nat(first),value),Nat.add(U32.to_nat(first),U32.to_nat(count)),bound,Bounds.strict_le(Nat.add(U32.to_nat(first),value),Nat.add(U32.to_nat(first),U32.to_nat(count)),Order.offset_lt(U32.to_nat(first),value,U32.to_nat(count),left_less)),inside)} :
    Fits<unit,U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat((count / cut_unit / 2 * cut_unit : U32))),bound>}
  +right_fits = {%Equal.sym(Nat,U32.to_nat((first + (count / cut_unit / 2 * cut_unit : U32) : U32)),Nat.add(U32.to_nat(first),value),next_exact) : Fits<unit,_,Nat.add(U32.to_nat((first + (count / cut_unit / 2 * cut_unit : U32) : U32)),U32.to_nat((count - (count / cut_unit / 2 * cut_unit : U32) : U32))),bound>
    %Equal.sym(Nat,Nat.add(U32.to_nat((first + (count / cut_unit / 2 * cut_unit : U32) : U32)),U32.to_nat((count - (count / cut_unit / 2 * cut_unit : U32) : U32))),Nat.add(U32.to_nat(first),U32.to_nat(count)),right_end) : Fits<unit,Nat.add(U32.to_nat(first),value),_,bound>
    Fits{middle_units,high_units,middle_exact,high_exact,Order.offset_lt(U32.to_nat(first),value,U32.to_nat(count),left_less),inside} :
    Fits<unit,U32.to_nat((first + (count / cut_unit / 2 * cut_unit : U32) : U32)),Nat.add(U32.to_nat((first + (count / cut_unit / 2 * cut_unit : U32) : U32)),U32.to_nat((count - (count / cut_unit / 2 * cut_unit : U32) : U32))),bound>}
  Halves{left_fits,right_fits,next_exact_left(first,(count / cut_unit / 2 * cut_unit : U32),(first + (count / cut_unit / 2 * cut_unit : U32) : U32),value,next_exact,left_exact),right_end}

def halves_left(-first: U32,-count: U32,-cut_unit: U32,-unit: Nat,-bound: Nat,+halves: Halves<first,count,cut_unit,unit,bound>) -> Fits<unit,U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat((count / cut_unit / 2 * cut_unit : U32))),bound>:
  Halves{+left,+right,+next,+ends} = halves
  left

def halves_right(-first: U32,-count: U32,-cut_unit: U32,-unit: Nat,-bound: Nat,+halves: Halves<first,count,cut_unit,unit,bound>) -> Fits<unit,U32.to_nat((first + (count / cut_unit / 2 * cut_unit : U32) : U32)),Nat.add(U32.to_nat((first + (count / cut_unit / 2 * cut_unit : U32) : U32)),U32.to_nat((count - (count / cut_unit / 2 * cut_unit : U32) : U32))),bound>:
  Halves{+left,+right,+next,+ends} = halves
  right

def halves_next(-first: U32,-count: U32,-cut_unit: U32,-unit: Nat,-bound: Nat,+halves: Halves<first,count,cut_unit,unit,bound>) -> {U32.to_nat((first + (count / cut_unit / 2 * cut_unit : U32) : U32)) == Nat.add(U32.to_nat(first),U32.to_nat((count / cut_unit / 2 * cut_unit : U32))) : Nat}:
  Halves{+left,+right,+next,+ends} = halves
  next

def halves_end(-first: U32,-count: U32,-cut_unit: U32,-unit: Nat,-bound: Nat,+halves: Halves<first,count,cut_unit,unit,bound>) -> {Nat.add(U32.to_nat((first + (count / cut_unit / 2 * cut_unit : U32) : U32)),U32.to_nat((count - (count / cut_unit / 2 * cut_unit : U32) : U32))) == Nat.add(U32.to_nat(first),U32.to_nat(count)) : Nat}:
  Halves{+left,+right,+next,+ends} = halves
  ends


# A range of whole units cut in halves of whole units is a run of the same
# cells.
def cut_run(fuel: Nat,cut: Bool,+limits: Machine.Limits,+cut_unit: U32,+cell_ns: F32,+grain: F32,+unit: Nat,+bound: Nat,+multiple: Nat,
  +unit_exact: {U32.to_nat(cut_unit) == Nat.mul(multiple,unit) : Nat},+unit_positive: {Nat.is_lt(0n,U32.to_nat(cut_unit)) == True{} : Bool},
  +unit_room: {Nat.is_le(U32.to_nat(cut_unit),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+first: U32,+count: U32,+fits: Fits<unit,U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),bound>,+cut_exact: {Tasks.cuts(Tasks.Cutting{cut_unit,cell_ns,grain},count) == cut : Bool}) ->
  CutRun(Tasks.cut_range(fuel,cut,Tasks.Cutting{cut_unit,cell_ns,grain},first,count),first,count,unit,bound):
  match fuel:
    case 0n: Held.Both{fits,Held.Both{{==},{==}}}
    case 1n+ +rest:
      match cut:
        case False{}: Held.Both{fits,Held.Both{{==},{==}}}
        case True{}:
          Fits{+low_units,+high_units,+low_exact,+high_exact,+range_less,+inside} = fits
          +cap = U32.to_nat(Machine.capacity(limits))
          +count_room = Order.transitive(U32.to_nat(count),Nat.add(U32.to_nat(first),U32.to_nat(count)),cap,Order.right_le(U32.to_nat(first),U32.to_nat(count)),Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(count)),bound,cap,inside,room))
          +sum = Arithmetic.add(cut_unit,cut_unit,Machine.half_word(limits),LimitBounds.half_word_fits(limits),
            {%Equal.sym(Nat,Nat.add(U32.to_nat(cut_unit),U32.to_nat(cut_unit)),Nat.double(U32.to_nat(cut_unit)),Addition.double(U32.to_nat(cut_unit))) : {Nat.is_le(_,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
              LimitBounds.double_bound(limits,U32.to_nat(cut_unit),unit_room) : {Nat.is_le(Nat.add(U32.to_nat(cut_unit),U32.to_nat(cut_unit)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}})
          +two = {%sum : {Nat.is_le(_,U32.to_nat(count)) == True{} : Bool}
            %Comparison.le((cut_unit + cut_unit : U32),count) :
              {_ == True{} : Bool}
            Boolean.and_true_left(U32.is_le((cut_unit + cut_unit : U32),count),F32.is_le((grain + grain : F32),(U32.to_f32(count) * cell_ns : F32)),cut_exact) :
            {Nat.is_le(Nat.add(U32.to_nat(cut_unit),U32.to_nat(cut_unit)),U32.to_nat(count)) == True{} : Bool}}
          +halves = halves_of(limits,cut_unit,cell_ns,grain,unit,bound,multiple,unit_exact,unit_positive,unit_room,room,first,count,low_units,high_units,low_exact,high_exact,inside,count_room,cut_left(limits,count,cut_unit,unit_positive,unit_room,count_room,two))
          cut_joined(Tasks.cut_range(rest,Tasks.cuts(Tasks.Cutting{cut_unit,cell_ns,grain},(count / cut_unit / 2 * cut_unit : U32)),Tasks.Cutting{cut_unit,cell_ns,grain},first,(count / cut_unit / 2 * cut_unit : U32)),Tasks.cut_range(rest,Tasks.cuts(Tasks.Cutting{cut_unit,cell_ns,grain},(count - (count / cut_unit / 2 * cut_unit : U32) : U32)),Tasks.Cutting{cut_unit,cell_ns,grain},(first + (count / cut_unit / 2 * cut_unit : U32) : U32),(count - (count / cut_unit / 2 * cut_unit : U32) : U32)),
            first,(count / cut_unit / 2 * cut_unit : U32),(first + (count / cut_unit / 2 * cut_unit : U32) : U32),(count - (count / cut_unit / 2 * cut_unit : U32) : U32),unit,bound,Nat.add(U32.to_nat(first),U32.to_nat(count)),halves_next(first,count,cut_unit,unit,bound,halves),halves_end(first,count,cut_unit,unit,bound,halves),
            cut_run(rest,Tasks.cuts(Tasks.Cutting{cut_unit,cell_ns,grain},(count / cut_unit / 2 * cut_unit : U32)),limits,cut_unit,cell_ns,grain,unit,bound,multiple,unit_exact,unit_positive,unit_room,room,first,(count / cut_unit / 2 * cut_unit : U32),halves_left(first,count,cut_unit,unit,bound,halves),{==}),
            cut_run(rest,Tasks.cuts(Tasks.Cutting{cut_unit,cell_ns,grain},(count - (count / cut_unit / 2 * cut_unit : U32) : U32)),limits,cut_unit,cell_ns,grain,unit,bound,multiple,unit_exact,unit_positive,unit_room,room,(first + (count / cut_unit / 2 * cut_unit : U32) : U32),(count - (count / cut_unit / 2 * cut_unit : U32) : U32),halves_right(first,count,cut_unit,unit,bound,halves),{==}))

# ---- The decision that may cut pieces ----

# Two runs of cut ranges that meet are a run of their split.
def fold_joined(left_ranges: Pieces.Ranges,right_ranges: Pieces.Ranges,+left: Pieces.Ranges,+right: Pieces.Ranges,+unit: Nat,+bound: Nat,
  +glue: {range_high(left) == range_low(right) : Nat},
  +left_run: Held.Both<Run(left_ranges,unit,bound),Held.Both<{range_low(left_ranges) == range_low(left) : Nat},{range_high(left_ranges) == range_high(left) : Nat}>>,
  +right_run: Held.Both<Run(right_ranges,unit,bound),Held.Both<{range_low(right_ranges) == range_low(right) : Nat},{range_high(right_ranges) == range_high(right) : Nat}>>) ->
  Held.Both<Run(Pieces.Split{left_ranges,right_ranges},unit,bound),Held.Both<{range_low(left_ranges) == range_low(left) : Nat},{range_high(right_ranges) == range_high(right) : Nat}>>:
  Held.Both{+left_ranges_run,+left_ends} = left_run
  Held.Both{+left_low,+left_high} = left_ends
  Held.Both{+right_ranges_run,+right_ends} = right_run
  Held.Both{+right_low,+right_high} = right_ends
  Held.Both{Held.Both{left_ranges_run,Held.Both{right_ranges_run,
      Equal.trans(Nat,range_high(left_ranges),range_high(left),range_low(right_ranges),left_high,
        Equal.trans(Nat,range_high(left),range_low(right),range_low(right_ranges),glue,Equal.sym(Nat,range_low(right_ranges),range_low(right),right_low)))}},
    Held.Both{left_low,right_high}}

# The ranges of a run with every range cut are a run of the same cells.
def cut_fold_run(+limits: Machine.Limits,+cut_unit: U32,+cell_ns: F32,+grain: F32,+unit: Nat,+bound: Nat,+multiple: Nat,
  +unit_exact: {U32.to_nat(cut_unit) == Nat.mul(multiple,unit) : Nat},+unit_positive: {Nat.is_lt(0n,U32.to_nat(cut_unit)) == True{} : Bool},
  +unit_room: {Nat.is_le(U32.to_nat(cut_unit),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},ranges: Pieces.Ranges,+run: Run(ranges,unit,bound)) ->
  Held.Both<Run(Pieces.ranges_fold(~Pieces.Ranges,~Tasks.Cutting,~Tasks.cut_leaf,~Tasks.cut_join,ranges,Tasks.Cutting{cut_unit,cell_ns,grain}),unit,bound),Held.Both<{range_low(Pieces.ranges_fold(~Pieces.Ranges,~Tasks.Cutting,~Tasks.cut_leaf,~Tasks.cut_join,ranges,Tasks.Cutting{cut_unit,cell_ns,grain})) == range_low(ranges) : Nat},{range_high(Pieces.ranges_fold(~Pieces.Ranges,~Tasks.Cutting,~Tasks.cut_leaf,~Tasks.cut_join,ranges,Tasks.Cutting{cut_unit,cell_ns,grain})) == range_high(ranges) : Nat}>>:
  match ranges:
    case Pieces.Range{+first,+count}:
      cut_run(32n,Tasks.cuts(Tasks.Cutting{cut_unit,cell_ns,grain},count),limits,cut_unit,cell_ns,grain,unit,bound,multiple,unit_exact,unit_positive,unit_room,room,first,count,run,{==})
    case Pieces.Split{+left,+right}:
      Held.Both{+left_run,+rest} = run
      Held.Both{+right_run,+glue} = rest
      fold_joined(Pieces.ranges_fold(~Pieces.Ranges,~Tasks.Cutting,~Tasks.cut_leaf,~Tasks.cut_join,left,Tasks.Cutting{cut_unit,cell_ns,grain}),Pieces.ranges_fold(~Pieces.Ranges,~Tasks.Cutting,~Tasks.cut_leaf,~Tasks.cut_join,right,Tasks.Cutting{cut_unit,cell_ns,grain}),left,right,unit,bound,glue,cut_fold_run(limits,cut_unit,cell_ns,grain,unit,bound,multiple,unit_exact,unit_positive,unit_room,room,left,left_run),cut_fold_run(limits,cut_unit,cell_ns,grain,unit,bound,multiple,unit_exact,unit_positive,unit_room,room,right,right_run))

def fold_run_of(+ranges: Pieces.Ranges,+folded: Pieces.Ranges,+unit: Nat,+bound: Nat,
  +facts: Held.Both<Run(folded,unit,bound),Held.Both<{range_low(folded) == range_low(ranges) : Nat},{range_high(folded) == range_high(ranges) : Nat}>>) -> Run(folded,unit,bound):
  Held.Both{+run,+ends} = facts
  run

# A run of the cells of another layout, from its first cell to its end.
def RunOf(ranges: Pieces.Ranges,+of: Pieces.Ranges,+unit: Nat,+bound: Nat) -> Data:
  Held.Both<Run(ranges,unit,bound),Held.Both<{range_low(ranges) == range_low(of) : Nat},{range_high(ranges) == range_high(of) : Nat}>>

def run_of_moved(+ranges: Pieces.Ranges,+middle: Pieces.Ranges,+of: Pieces.Ranges,+unit: Nat,+bound: Nat,+facts: RunOf(ranges,middle,unit,bound),
  +low_same: {range_low(middle) == range_low(of) : Nat},+high_same: {range_high(middle) == range_high(of) : Nat}) -> RunOf(ranges,of,unit,bound):
  Held.Both{+run,+ends} = facts
  Held.Both{+low,+high} = ends
  Held.Both{run,Held.Both{Equal.trans(Nat,range_low(ranges),range_low(middle),range_low(of),low,low_same),Equal.trans(Nat,range_high(ranges),range_high(middle),range_high(of),high,high_same)}}

def run_of_chain(+ranges: Pieces.Ranges,+middle: Pieces.Ranges,+of: Pieces.Ranges,+unit: Nat,+bound: Nat,+first: RunOf(ranges,middle,unit,bound),
  +second: RunOf(middle,of,unit,bound)) -> RunOf(ranges,of,unit,bound):
  Held.Both{+run,+ends} = second
  Held.Both{+low,+high} = ends
  run_of_moved(ranges,middle,of,unit,bound,first,low,high)

# The tasks of a group of ranges run over the cells of the group.
def group_ends(+group: Tasks.Group,+low: Nat,+high: Nat,+unit: Nat,+bound: Nat,+facts: GroupRun(group,low,high,unit,bound)) ->
  Held.Both<Run(Tasks.group_layout(group),unit,bound),Held.Both<{range_low(Tasks.group_layout(group)) == low : Nat},{range_high(Tasks.group_layout(group)) == high : Nat}>>:
  match group:
    case Tasks.Group{+first,+count,+layout}:
      Held.Both{+first_low,+rest} = facts
      Held.Both{+ended,+more} = rest
      more

def sized_ends(+limits: Machine.Limits,+layout: Pieces.Ranges,+count: U32,+pool: U32,+whole: U32,+found: Tasks.Size,+unit: Nat,+bound: Nat,+run: Run(layout,unit,bound),
  +room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> RunOf(Tasks.tasks_sized(layout,count,pool,whole,found),layout,unit,bound):
  match found:
    case Tasks.Size{+ranges,+largest}:
      +limit = U32.max(U32.max(Tasks.divided_up(count,pool),whole),(Tasks.divided_up(ranges,pool) * largest : U32))
      group_ends(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,layout,limit),range_low(layout),range_high(layout),unit,bound,
        grouped(limits,limit,layout,unit,bound,run,room))

# The tasks over a run run over its cells.
def tasks_ends(+limits: Machine.Limits,+layout: Pieces.Ranges,+count: U32,+workers: U32,+whole: U32,+unit: Nat,+bound: Nat,+run: Run(layout,unit,bound),
  +room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> RunOf(Tasks.tasks_over(layout,count,workers,whole),layout,unit,bound):
  sized_ends(limits,layout,count,U32.max(workers,1),whole,Tasks.size(layout),unit,bound,run,room)

# The tasks of the pool a decision picks over a run run over its cells.
def waved_ends(+limits: Machine.Limits,+layout: Pieces.Ranges,+count: U32,+workers: U32,+whole: U32,+cell_ns: F32,+config: Cost.Configuration,+extra: Tasks.Extra,
  +unit: Nat,+bound: Nat,+run: Run(layout,unit,bound),+room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  RunOf(Tasks.tasks_waved(layout,count,workers,whole,cell_ns,config,extra),layout,unit,bound):
  sized_ends(limits,layout,count,Tasks.waved_pool(workers,cell_ns,config,extra,count,whole,Tasks.size(layout)),whole,Tasks.size(layout),unit,bound,run,room)

def whole_ends(+limits: Machine.Limits,+layout: Pieces.Ranges,+unit: Nat,+bound: Nat,+run: Run(layout,unit,bound),+room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  RunOf(Tasks.whole_bounds(Pieces.bounds(layout)),layout,unit,bound):
  Held.Both{whole_run(limits,layout,unit,bound,run,room),Held.Both{first_low(layout),whole_end(limits,layout,unit,bound,run,room)}}

def planned_ends(+limits: Machine.Limits,+parallel: Bool,+layout: Pieces.Ranges,+of: Pieces.Ranges,+unit: Nat,+bound: Nat,+facts: RunOf(layout,of,unit,bound),+room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  RunOf(decision_tasks(Tasks.planned(parallel,layout)),of,unit,bound):
  match parallel:
    case True{}: facts
    case False{}:
      Held.Both{+run,+ends} = facts
      Held.Both{+low,+high} = ends
      run_of_moved(Tasks.whole_bounds(Pieces.bounds(layout)),layout,of,unit,bound,whole_ends(limits,layout,unit,bound,run,room),low,high)

def chosen_ends(flag: Bool,+yes: Pieces.Ranges,+no: Pieces.Ranges,+of: Pieces.Ranges,+unit: Nat,+bound: Nat,+yes_facts: RunOf(yes,of,unit,bound),+no_facts: RunOf(no,of,unit,bound)) ->
  RunOf(Choice.chosen(~Pieces.Ranges,flag,yes,no),of,unit,bound):
  match flag:
    case True{}: yes_facts
    case False{}: no_facts

# The tasks picked among a serial plan, whole pieces and cut pieces run over
# the cells of both.
def picked_ends(+limits: Machine.Limits,+workers: U32,+serial_ns: F32,+whole_ns: F32,+cut_ns: F32,+whole_layout: Pieces.Ranges,+cut_layout: Pieces.Ranges,+of: Pieces.Ranges,
  +unit: Nat,+bound: Nat,+whole_facts: RunOf(whole_layout,of,unit,bound),+cut_facts: RunOf(cut_layout,of,unit,bound),+room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  RunOf(decision_tasks(Tasks.picked(workers,serial_ns,whole_ns,cut_ns,whole_layout,cut_layout)),of,unit,bound):
  planned_ends(limits,U32.is_lt(1,workers) && F32.is_lt(Choice.chosen(~F32,F32.is_lt(cut_ns,whole_ns),cut_ns,whole_ns),serial_ns),Choice.chosen(~Pieces.Ranges,F32.is_lt(cut_ns,whole_ns),cut_layout,whole_layout),of,unit,bound,
    chosen_ends(F32.is_lt(cut_ns,whole_ns),cut_layout,whole_layout,of,unit,bound,cut_facts,whole_facts),room)

def decided_cut_with_ends(+limits: Machine.Limits,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+piece_units: U32,+per_outer: U32,+pieces: U32,
  +workers: U32,+whole: U32,+bytes_per_cell: F32,+operations: F32,+cut_cell_ns: F32,+configuration: Cost.Configuration,+extra: Tasks.Extra,+unit: Nat,+bound: Nat,+multiple: Nat,
  +unit_exact: {U32.to_nat(U32.max(U32.max(unit_cells,whole),1)) == Nat.mul(multiple,unit) : Nat},+unit_room: {Nat.is_le(U32.to_nat(U32.max(U32.max(unit_cells,whole),1)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +run: Run(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,piece_units,per_outer,pieces}),unit,bound),+room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  RunOf(decision_tasks(Tasks.decide_cut_with(Grid.Grid{count,prefix,outer,extent,unit_cells,piece_units,per_outer,pieces},workers,whole,Cost.cell_ns(bytes_per_cell,operations,configuration),cut_cell_ns,configuration,extra)),Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,piece_units,per_outer,pieces}),unit,bound):
  match configuration:
    case Cost.Configuration{+row_ns,+byte_ns,+step_ns,+move_ns,+grain_ns,+wake_round_ns,+dispatch_ns}:
      +config = {Cost.Configuration{row_ns,byte_ns,step_ns,move_ns,grain_ns,wake_round_ns,dispatch_ns} : Cost.Configuration}
      +cell_ns = Cost.cell_ns(bytes_per_cell,operations,configuration)
      +ranges = Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,piece_units,per_outer,pieces})
      +cut_unit = U32.max(U32.max(unit_cells,whole),1)
      +context = {Tasks.Cutting{cut_unit,cell_ns,grain_ns} : Tasks.Cutting}
      +whole_range = Tasks.whole_bounds(Pieces.bounds(ranges))
      +cut_ranges = Pieces.ranges_fold(~Pieces.Ranges,~Tasks.Cutting,~Tasks.cut_leaf,~Tasks.cut_join,whole_range,context)
      +whole_layout = Tasks.tasks_waved(ranges,count,workers,whole,cell_ns,config,extra)
      +cut_layout = Tasks.tasks_waved(cut_ranges,count,workers,whole,cell_ns,config,extra)
      +unit_positive = Order.succ_le_lt(0n,U32.to_nat(cut_unit),
        {%Comparison.le(1,cut_unit) : {_ == True{} : Bool} Extrema.maximum_right(U32.max(unit_cells,whole),1) : {Nat.is_le(1n,U32.to_nat(cut_unit)) == True{} : Bool}})
      +folded = cut_fold_run(limits,cut_unit,cell_ns,grain_ns,unit,bound,multiple,unit_exact,unit_positive,unit_room,room,whole_range,whole_run(limits,ranges,unit,bound,run,room))
      +cut_facts = run_of_chain(cut_layout,cut_ranges,ranges,unit,bound,
        waved_ends(limits,cut_ranges,count,workers,whole,cell_ns,config,extra,unit,bound,fold_run_of(whole_range,cut_ranges,unit,bound,folded),room),
        run_of_chain(cut_ranges,whole_range,ranges,unit,bound,folded,whole_ends(limits,ranges,unit,bound,run,room)))
      picked_ends(limits,workers,(U32.to_f32(count) * cell_ns : F32),Tasks.plan_ns(workers,cell_ns,config,extra,count,Tasks.size(whole_layout)),
        (Tasks.plan_ns(workers,cell_ns,config,extra,count,Tasks.size(cut_layout)) + U32.to_f32(Pieces.ranges_fold(~U32,~Pieces.Ranges,~Tasks.crossed_leaf,~Tasks.crossed_join,ranges,cut_layout)) * cut_cell_ns : F32),
        whole_layout,cut_layout,ranges,unit,bound,waved_ends(limits,ranges,count,workers,whole,cell_ns,config,extra,unit,bound,run,room),cut_facts,room)

# The unit the decision over the pieces of a grid cuts them in.
def cut_unit_of(+grid: Grid.Grid,+whole: U32) -> U32:
  Grid.Grid{count,prefix,outer,extent,+unit_cells,piece_units,per_outer,pieces} = grid
  U32.max(U32.max(unit_cells,whole),1)

# The tasks of the decision over the pieces of a grid that may cut them run
# over the cells of the grid.
def decided_cut_ends(+limits: Machine.Limits,+grid: Grid.Grid,+workers: U32,+whole: U32,+bytes_per_cell: F32,+operations: F32,+cut_cell_ns: F32,+extra: Tasks.Extra,
  +unit: Nat,+bound: Nat,+multiple: Nat,+unit_exact: {U32.to_nat(cut_unit_of(grid,whole)) == Nat.mul(multiple,unit) : Nat},
  +unit_room: {Nat.is_le(U32.to_nat(cut_unit_of(grid,whole)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +run: Run(Pieces.ranges_of_grid(grid),unit,bound),+room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  RunOf(decision_tasks(Tasks.decide_cut(grid,workers,whole,bytes_per_cell,operations,cut_cell_ns,extra)),Pieces.ranges_of_grid(grid),unit,bound):
  Grid.Grid{+count,+prefix,+outer,+extent,+unit_cells,+piece_units,+per_outer,+pieces} = grid
  decided_cut_with_ends(limits,count,prefix,outer,extent,unit_cells,piece_units,per_outer,pieces,workers,whole,bytes_per_cell,operations,cut_cell_ns,DefaultCost.configuration(),extra,
    unit,bound,multiple,unit_exact,unit_room,run,room)
