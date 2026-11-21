# Each panel reads and writes one image, whether its task runs one image or several.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_gemm_buffer.bend as GemmBuffer
import ../machine_limits.bend as Machine
import ../kernel_tile.bend as Tile
import ../storage_clone.bend as Clone
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_halo.bend as Halo
import ../tensor_move.bend as Move
import ../tensor_pieces.bend as Pieces
import ../tensor_point_products.bend as Products
import ../tensor_tasks.bend as Tasks
import ../traversal_partition.bend as Partition
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./kernel_window_windows.bend as KernelWindows
import ./kernel_gemm_point_jobs.bend as PointJobs
import ./machine_limit_bounds.bend as Limits
import ./nat_division_order.bend as DivisionOrder
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./storage_buffer_model.bend as Buffer
import ./tensor_pieces_model.bend as Model
import ./storage_clone_refinement.bend as Cloned
import ./tensor_grid_pieces.bend as GridLaws
import ./u32_bit_length.bend as Bits
import ./tensor_grid_ranges.bend as GridRanges
import ./tensor_halo_windows.bend as Windows
import ./tensor_move_readback.bend as Held
import ./tensor_point_product_tasks.bend as Shared
import ./tensor_point_products.bend as Point
import ./tensor_pieces_walk.bend as Walk
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_task_runs.bend as Runs
import ./row_major_index.bend as Index
import ./u32_comparison.bend as Comparison
import ./word_multiplication.bend as Multiply

def Singletons(ranges: Pieces.Ranges) -> Data:
  match ranges:
    case Pieces.Range{first,+count}: {count == 1 : U32}
    case Pieces.Split{+left,+right}: Held.Both<Singletons(left),Singletons(right)>

def ImageRun(+ranges: Pieces.Ranges,+first: U32,+count: U32,+bound: Nat) -> Data:
  Held.Both<Runs.RunOf(ranges,Pieces.Range{first,count},1n,bound),Singletons(ranges)>

def single(+first: U32,+count: U32,+bound: Nat,+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
  +at_most_one: {Nat.is_le(U32.to_nat(count),1n) == True{} : Bool},
  +inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),bound) == True{} : Bool}) ->
  ImageRun(Pieces.Range{first,count},first,count,bound):
  +low = U32.to_nat(first)
  +high = Nat.add(low,U32.to_nat(count))
  +one = Order.antisymmetric(U32.to_nat(count),1n,at_most_one,Order.positive_one(U32.to_nat(count),positive))
  +word_one = Equal.sym(U32,1,count,Arithmetic.from_nat_exact(count,1n,one))
  Held.Both{Held.Both{Runs.Fits{low,high,Equal.sym(Nat,Nat.mul(low,1n),low,Order.times_one(low)),
    Equal.sym(Nat,Nat.mul(high,1n),high,Order.times_one(high)),
    %Equal.sym(Nat,U32.to_nat(count),1n,one) : {Nat.is_lt(low,Nat.add(low,_)) == True{} : Bool}
    Order.below_sum(low,0n),inside},Held.Both{{==},{==}}},word_one}

def joined_image_run(+left: Pieces.Ranges,+right: Pieces.Ranges,+first: U32,+count: U32,+left_count: U32,+next: U32,+right_count: U32,+bound: Nat,
  +glue: {U32.to_nat(next) == Nat.add(U32.to_nat(first),U32.to_nat(left_count)) : Nat},
  +end: {Nat.add(U32.to_nat(next),U32.to_nat(right_count)) == Nat.add(U32.to_nat(first),U32.to_nat(count)) : Nat},
  +left_checked: ImageRun(left,first,left_count,bound),+right_checked: ImageRun(right,next,right_count,bound)) ->
  ImageRun(Pieces.Split{left,right},first,count,bound):
  Held.Both{+left_plan,+left_single} = left_checked
  Held.Both{+left_run,+left_ends} = left_plan
  Held.Both{+left_low,+left_high} = left_ends
  Held.Both{+right_plan,+right_single} = right_checked
  Held.Both{+right_run,+right_ends} = right_plan
  Held.Both{+right_low,+right_high} = right_ends
  +same = Equal.trans(Nat,Runs.range_high(left),Nat.add(U32.to_nat(first),U32.to_nat(left_count)),Runs.range_low(right),left_high,
    Equal.sym(Nat,Runs.range_low(right),Nat.add(U32.to_nat(first),U32.to_nat(left_count)),
      Equal.trans(Nat,Runs.range_low(right),U32.to_nat(next),Nat.add(U32.to_nat(first),U32.to_nat(left_count)),right_low,glue)))
  Held.Both{Held.Both{Held.Both{left_run,Held.Both{right_run,same}},
    Held.Both{left_low,Equal.trans(Nat,Runs.range_high(right),Nat.add(U32.to_nat(next),U32.to_nat(right_count)),
      Nat.add(U32.to_nat(first),U32.to_nat(count)),right_high,end)}},Held.Both{left_single,right_single}}

def index_run(+limits: Machine.Limits,depth: Nat,one: Bool,+first: U32,+count: U32,+bound: Nat,
  +test: {U32.is_le(count,1) == one : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
  +size: {Nat.is_le(U32.to_nat(count),Bits.power(depth)) == True{} : Bool},
  +inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),bound) == True{} : Bool},
  +room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  ImageRun(Tasks.index_ranges(depth,one,first,count),first,count,bound):
  match depth one:
    case 0n one: single(first,count,bound,positive,size,inside)
    case 1n+rest True{}:
      single(first,count,bound,positive,
        %Comparison.le(count,1) : {_ == True{} : Bool} test,inside)
    case 1n+ +rest False{}:
      +capacity = Machine.capacity(limits)
      +start = U32.to_nat(first)
      +length = U32.to_nat(count)
      +count_room = Order.transitive(length,Nat.add(start,length),U32.to_nat(capacity),Order.right_le(start,length),
        Order.transitive(Nat.add(start,length),bound,U32.to_nat(capacity),inside,room))
      +halves = GridRanges.halves_of(capacity,Limits.capacity_allocatable(limits),rest,count,test,positive,size,count_room)
      +left = Pieces.below(count)
      +right = (count - left : U32)
      +next = (first + left : U32)
      +left_count = GridRanges.halves_left_count(count,rest,halves)
      +left_inside = Order.transitive(Nat.add(start,U32.to_nat(left)),Nat.add(start,length),bound,Order.offset_le(start,U32.to_nat(left),length,left_count),inside)
      +glue = Arithmetic.add(first,left,capacity,Limits.capacity_fits(limits),
        Order.transitive(Nat.add(start,U32.to_nat(left)),bound,U32.to_nat(capacity),left_inside,room))
      +end = GridRanges.split_sum(first,count,glue,GridRanges.halves_right_exact(count,rest,halves),left_count)
      joined_image_run(Tasks.index_ranges(rest,U32.is_le(left,1),first,left),Tasks.index_ranges(rest,U32.is_le(right,1),next,right),
        first,count,left,next,right,bound,glue,end,
        index_run(limits,rest,U32.is_le(left,1),first,left,bound,{==},GridRanges.halves_left_positive(count,rest,halves),GridRanges.halves_left_bound(count,rest,halves),left_inside,room),
        index_run(limits,rest,U32.is_le(right,1),next,right,bound,{==},GridRanges.halves_right_positive(count,rest,halves),GridRanges.halves_right_bound(count,rest,halves),
          %Equal.sym(Nat,Nat.add(U32.to_nat(next),U32.to_nat(right)),Nat.add(start,length),end) : {Nat.is_le(_,bound) == True{} : Bool} inside,room))

def components(+limits: Machine.Limits,+count: U32,+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  ImageRun(Tasks.component_ranges(count),0,count,U32.to_nat(count)):
  index_run(limits,32n,U32.is_le(count,1),0,count,U32.to_nat(count),{==},positive,
    Order.transitive(U32.to_nat(count),Bits.power(31n),Bits.power(32n),GridRanges.capacity_power(limits,U32.to_nat(count),room),Bits.power_monotone(31n,32n,{==})),
    Order.reflexive(U32.to_nat(count)),room)

def left_width(+work: Products.Shape) -> U32:
  Products.Shape{+rows,+inner,channels,grid} = work
  (rows * Gemm.blocks(inner) : U32)

def right_width(+work: Products.Shape) -> U32:
  Products.Shape{rows,+inner,+channels,grid} = work
  (Gemm.blocks(channels) * inner : U32)

def left_scaled(+work: Products.Shape,+first: U32,+count: U32) ->
  {Products.panel_left(work,first,count) == Tasks.Span{(first * left_width(work) : U32),(count * left_width(work) : U32)} : Tasks.Span}:
  Products.Shape{+rows,+inner,channels,grid} = work
  {==}

def right_scaled(+work: Products.Shape,+first: U32,+count: U32) ->
  {Products.panel_right(work,first,count) == Tasks.Span{(first * right_width(work) : U32),(count * right_width(work) : U32)} : Tasks.Span}:
  Products.Shape{rows,+inner,+channels,grid} = work
  {==}

def panel_reaches(~reach: Halo.Reach(~Products.Shape),~width: @+work: Products.Shape -> U32,
  ~scales: @+work: Products.Shape -> @+first: U32 -> @+count: U32 ->
    {reach(work,first,count) == Tasks.Span{(first * width(work) : U32),(count * width(work) : U32)} : Tasks.Span},
  +work: Products.Shape,+images: U32,+total: Nat,+capacity: U32,+cap_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +exact: {total == Nat.mul(U32.to_nat(images),U32.to_nat(width(work))) : Nat},+room: {Nat.is_le(total,U32.to_nat(capacity)) == True{} : Bool},
  tasks: Pieces.Ranges,+run: Runs.Run(tasks,1n,U32.to_nat(images))) ->
  Windows.Reaches(~Products.Shape,~reach,work,tasks,total):
  match tasks:
    case Pieces.Range{+first,+count}:
      +span = {Tasks.Span{(first * width(work) : U32),(count * width(work) : U32)} : Tasks.Span}
      +first_cell = U32.to_nat(first)
      +end = Nat.add(first_cell,U32.to_nat(count))
      +scale = U32.to_nat(width(work))
      %Equal.sym(Tasks.Span,reach(work,first,count),span,scales(work,first,count)) :
        Held.Both<{Nat.is_le(Nat.add(U32.to_nat(Windows.span_start(_)),Nat.sub(U32.to_nat(Windows.span_end(_)),U32.to_nat(Windows.span_start(_)))),total) == True{} : Bool},
          {Nat.is_le(U32.to_nat(Windows.span_start(_)),U32.to_nat(Windows.span_end(_))) == True{} : Bool}>
      Shared.scaled_span(first,count,width(work),total,capacity,cap_ok,
        %Equal.sym(Nat,total,Nat.mul(U32.to_nat(images),scale),exact) : {Nat.is_le(Nat.mul(end,scale),_) == True{} : Bool}
        Index.mul_le(end,U32.to_nat(images),scale,Runs.fits_inside(1n,first_cell,end,U32.to_nat(images),run)),room)
    case Pieces.Split{+left,+right}:
      Held.Both{+left_run,rest} = run
      Held.Both{+right_run,meets} = rest
      Held.Both{panel_reaches(~reach,~width,~scales,work,images,total,capacity,cap_ok,exact,room,left,left_run),
        panel_reaches(~reach,~width,~scales,work,images,total,capacity,cap_ok,exact,room,right,right_run)}

def panel_windows(~Source: Data,~value: Source -> U32 -> Tile.Tile8,~reach: Halo.Reach(~Products.Shape),~width: @+work: Products.Shape -> U32,
  ~scales: @+work: Products.Shape -> @+first: U32 -> @+count: U32 ->
    {reach(work,first,count) == Tasks.Span{(first * width(work) : U32),(count * width(work) : U32)} : Tasks.Span},
  +source: Source,+work: Products.Shape,+limits: Machine.Limits,+images: U32,+shape: List<&2,U32>,
  +exact: {ShapeLaws.elements(shape) == Nat.mul(U32.to_nat(images),U32.to_nat(width(work))) : Nat},
  +room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},
  +run: Runs.Run(Tasks.component_ranges(images),1n,U32.to_nat(images)),pieces: Pieces.Pieces(~Tile.Tile8),
  +covered: Held.Covered<Tile.Tile8,Source,value,source,Move.held(~Tile.Tile8,pieces),0n,ShapeLaws.elements(shape),Held.layout_spans(Pieces.ranges_of_grid(Tensors.grid_of(shape)),Held.Ended{})>) ->
  Windows.Windowed<Tile.Tile8,Source,value,source,Products.Shape,reach,work,ShapeLaws.elements(shape),Tasks.component_ranges(images),
    Halo.windows(~Tile.Tile8,~Products.Shape,~reach,work,Tensors.grid_of(shape),Products.zero(),Tasks.component_ranges(images),pieces)>:
  +total = ShapeLaws.elements(shape)
  +tasks = Tasks.component_ranges(images)
  +grid = Tensors.grid_of(shape)
  Windows.covered_windows(~Tile.Tile8,~Source,~value,~Products.Shape,~reach,limits,grid,Pieces.ranges_of_grid(grid),total,
    KernelWindows.input_laid(limits,shape,room,nonempty),work,source,Products.zero(),Machine.capacity(limits),Limits.capacity_allocatable(limits),room,
    GridRanges.grid_pieces_word(limits,shape,Grid.piece_target(),room,nonempty),tasks,
    panel_reaches(~reach,~width,~scales,work,images,total,Machine.capacity(limits),Limits.capacity_fits(limits),exact,room,tasks,run),pieces,covered)

type Operands is Data:
  Operands{left: Held.Holding<Tile.Tile8>,right: Held.Holding<Tile.Tile8>}

def NativeOperand() -> Type:
  Pieces.Paired<Move.Holding<Tile.Tile8>,Move.Holding<Tile.Tile8>>

def NativeOperands() -> Type:
  Partition.Tree<&1,NativeOperand(),Unit>

def OperandTree() -> Data:
  Partition.Tree<&2,Operands,Unit>

def pack_operands(operands: Operands) -> NativeOperand():
  Operands{left,right} = operands
  Pieces.Both{Held.pack(~Tile.Tile8,left),Held.pack(~Tile.Tile8,right)}

def pack_tree(tree: OperandTree()) -> NativeOperands():
  match tree:
    case Partition.Leaf{operands,first,count,details}:
      Partition.Leaf{pack_operands(operands),first,count,Unit{}}
    case Partition.Fork{left,right}: Partition.Fork{pack_tree(left),pack_tree(right)}

def zipped(+left: Windows.WindowModel<Tile.Tile8>,+right: Windows.WindowModel<Tile.Tile8>) -> NativeOperands():
  Pieces.zip(~Move.Holding<Tile.Tile8>,~Move.Holding<Tile.Tile8>,Windows.pack_windows(~Tile.Tile8,left),Windows.pack_windows(~Tile.Tile8,right))

def operand_ranges(tree: OperandTree()) -> Pieces.Ranges:
  match tree:
    case Partition.Leaf{operands,first,count,details}: Pieces.Range{first,count}
    case Partition.Fork{left,right}: Pieces.Split{operand_ranges(left),operand_ranges(right)}

def input_holding(operands: Operands) -> Held.Holding<Tile.Tile8>:
  Operands{input,weights} = operands
  input

def weight_holding(operands: Operands) -> Held.Holding<Tile.Tile8>:
  Operands{input,weights} = operands
  weights

def operand_windows(~holding: Operands -> Held.Holding<Tile.Tile8>,tree: OperandTree()) -> Windows.WindowModel<Tile.Tile8>:
  match tree:
    case Partition.Leaf{operands,+first,+count,details}: Windows.WindowLeaf{holding(operands),first,count}
    case Partition.Fork{left,right}: Windows.WindowFork{operand_windows(~holding,left),operand_windows(~holding,right)}

type Paired<-ranges: Pieces.Ranges,-native: NativeOperands(),-inputs: Windows.WindowModel<Tile.Tile8>,-weights: Windows.WindowModel<Tile.Tile8>> is Data:
  Paired{operands: OperandTree(),exact: {native == pack_tree(operands) : NativeOperands()},laid: {operand_ranges(operands) == ranges : Pieces.Ranges},
    input_windows: {operand_windows(~input_holding,operands) == inputs : Windows.WindowModel<Tile.Tile8>},
    weight_windows: {operand_windows(~weight_holding,operands) == weights : Windows.WindowModel<Tile.Tile8>}}

def joined_windows(+left: Windows.WindowModel<Tile.Tile8>,+right: Windows.WindowModel<Tile.Tile8>,+inputs: Windows.WindowModel<Tile.Tile8>,+weights: Windows.WindowModel<Tile.Tile8>,
  +first: {left == inputs : Windows.WindowModel<Tile.Tile8>},+second: {right == weights : Windows.WindowModel<Tile.Tile8>}) ->
  {Windows.WindowFork{left,right} == Windows.WindowFork{inputs,weights} : Windows.WindowModel<Tile.Tile8>}:
  %first : {Windows.WindowFork{left,right} == Windows.WindowFork{_,weights} : Windows.WindowModel<Tile.Tile8>}
  %second : {Windows.WindowFork{left,right} == Windows.WindowFork{left,_} : Windows.WindowModel<Tile.Tile8>}
  {==}

def paired_fork(+left_ranges: Pieces.Ranges,+right_ranges: Pieces.Ranges,+left_windows: Windows.WindowModel<Tile.Tile8>,+right_windows: Windows.WindowModel<Tile.Tile8>,
  +left_weights: Windows.WindowModel<Tile.Tile8>,+right_weights: Windows.WindowModel<Tile.Tile8>,
  +left: Paired<left_ranges,zipped(left_windows,left_weights),left_windows,left_weights>,+right: Paired<right_ranges,zipped(right_windows,right_weights),right_windows,right_weights>) ->
  Paired<Pieces.Split{left_ranges,right_ranges},zipped(Windows.WindowFork{left_windows,right_windows},Windows.WindowFork{left_weights,right_weights}),
    Windows.WindowFork{left_windows,right_windows},Windows.WindowFork{left_weights,right_weights}>:
  Paired{+left_tree,+left_exact,+left_laid,+left_inputs,+left_weights_equation} = left
  Paired{+right_tree,+right_exact,+right_laid,+right_inputs,+right_weights_equation} = right
  Paired{Partition.Fork{left_tree,right_tree},
    %left_exact : {Partition.Fork{zipped(left_windows,left_weights),zipped(right_windows,right_weights)} == Partition.Fork{_,pack_tree(right_tree)} : NativeOperands()}
    %right_exact : {Partition.Fork{zipped(left_windows,left_weights),zipped(right_windows,right_weights)} == Partition.Fork{zipped(left_windows,left_weights),_} : NativeOperands()}
    {==},
    %left_laid : {operand_ranges(Partition.Fork{left_tree,right_tree}) == Pieces.Split{_,right_ranges} : Pieces.Ranges}
    %right_laid : {operand_ranges(Partition.Fork{left_tree,right_tree}) == Pieces.Split{operand_ranges(left_tree),_} : Pieces.Ranges}
    {==},
    joined_windows(operand_windows(~input_holding,left_tree),operand_windows(~input_holding,right_tree),left_windows,right_windows,left_inputs,right_inputs),
    joined_windows(operand_windows(~weight_holding,left_tree),operand_windows(~weight_holding,right_tree),left_weights,right_weights,left_weights_equation,right_weights_equation)}

# Both input windows are built from the same image ranges, so every leaf pairs.
def paired(~Left: Data,~left_value: Left -> U32 -> Tile.Tile8,~Right: Data,~right_value: Right -> U32 -> Tile.Tile8,
  +input: Left,+weights: Right,+input_total: Nat,+weight_total: Nat,+work: Products.Shape,
  ranges: Pieces.Ranges,left: Windows.WindowModel<Tile.Tile8>,right: Windows.WindowModel<Tile.Tile8>,
  left_ok: Windows.WindowsOk(~Tile.Tile8,~Left,~left_value,~Products.Shape,~Products.panel_left,input,work,input_total,ranges,left),
  right_ok: Windows.WindowsOk(~Tile.Tile8,~Right,~right_value,~Products.Shape,~Products.panel_right,weights,work,weight_total,ranges,right)) ->
  Paired<ranges,zipped(left,right),left,right>:
  match ranges left right:
    case Pieces.Range{+first,+count} Windows.WindowLeaf{+left_words,+left_first,+left_count} Windows.WindowLeaf{+right_words,+right_first,+right_count}:
      Held.Both{+same_first,left_rest} = left_ok
      Held.Both{+same_count,left_held} = left_rest
      Held.Both{+other_first,right_rest} = right_ok
      Held.Both{+other_count,right_held} = right_rest
      Paired{Partition.Leaf{Operands{left_words,right_words},left_first,left_count,Unit{}},{==},
        %same_first : {Pieces.Range{left_first,left_count} == Pieces.Range{_,count} : Pieces.Ranges}
        %same_count : {Pieces.Range{left_first,left_count} == Pieces.Range{left_first,_} : Pieces.Ranges}
        {==},{==},
        %Equal.sym(U32,right_first,first,other_first) :
          {Windows.WindowLeaf{right_words,left_first,left_count} == Windows.WindowLeaf{right_words,_,right_count} : Windows.WindowModel<Tile.Tile8>}
        %Equal.sym(U32,right_count,count,other_count) :
          {Windows.WindowLeaf{right_words,left_first,left_count} == Windows.WindowLeaf{right_words,first,_} : Windows.WindowModel<Tile.Tile8>}
        %same_first : {Windows.WindowLeaf{right_words,left_first,left_count} == Windows.WindowLeaf{right_words,_,count} : Windows.WindowModel<Tile.Tile8>}
        %same_count : {Windows.WindowLeaf{right_words,left_first,left_count} == Windows.WindowLeaf{right_words,left_first,_} : Windows.WindowModel<Tile.Tile8>}
        {==}}
    case Pieces.Split{+lower,+upper} Windows.WindowFork{+left_low,+left_high} Windows.WindowFork{+right_low,+right_high}:
      Held.Both{left_low_ok,left_high_ok} = left_ok
      Held.Both{right_low_ok,right_high_ok} = right_ok
      paired_fork(lower,upper,left_low,left_high,right_low,right_high,
        paired(~Left,~left_value,~Right,~right_value,input,weights,input_total,weight_total,work,lower,left_low,right_low,left_low_ok,right_low_ok),
        paired(~Left,~left_value,~Right,~right_value,input,weights,input_total,weight_total,work,upper,left_high,right_high,left_high_ok,right_high_ok))
    case Pieces.Range{+first,+count} Windows.WindowFork{+left_low,+left_high} right:
      Empty.absurd(Paired<Pieces.Range{first,count},zipped(Windows.WindowFork{left_low,left_high},right),Windows.WindowFork{left_low,left_high},right>,left_ok)
    case Pieces.Range{+first,+count} left Windows.WindowFork{+right_low,+right_high}:
      Empty.absurd(Paired<Pieces.Range{first,count},zipped(left,Windows.WindowFork{right_low,right_high}),left,Windows.WindowFork{right_low,right_high}>,right_ok)
    case Pieces.Split{+lower,+upper} Windows.WindowLeaf{+left_words,+left_first,+left_count} right:
      Empty.absurd(Paired<Pieces.Split{lower,upper},zipped(Windows.WindowLeaf{left_words,left_first,left_count},right),Windows.WindowLeaf{left_words,left_first,left_count},right>,left_ok)
    case Pieces.Split{+lower,+upper} left Windows.WindowLeaf{+right_words,+right_first,+right_count}:
      Empty.absurd(Paired<Pieces.Split{lower,upper},zipped(left,Windows.WindowLeaf{right_words,right_first,right_count}),left,Windows.WindowLeaf{right_words,right_first,right_count}>,right_ok)

def Output() -> Type:
  Pieces.Pieces(~Tile.Tile8)

def NativeResult() -> Type:
  Point.TaskOutput() & Products.Cells()

def Lefts() -> Type:
  Pieces.Lefts<Output()>

def NativeWalk() -> Type:
  Pieces.Walked<Output(),Products.Cells()>

type Observed<-native: NativeResult()> is Data:
  Observed{output: Shared.TaskModel(),bias: Buffer.Model<Tile.Tile8>,
    exact: {native == (Shared.pack_task(output),Buffer.pack(~Tile.Tile8,bias)) : NativeResult()}}

def observed_result(-task: Point.TaskOutput(),-bias: Products.Cells(),seen_task: Shared.ObservedTask<task>,seen_bias: Buffer.Observation<Tile.Tile8,bias>) ->
  Observed<(task,bias)>:
  Shared.ObservedTask{+model,+task_exact} = seen_task
  Buffer.Observation{+bias_model,+bias_exact} = seen_bias
  Observed{model,bias_model,
    %Equal.sym(Point.TaskOutput(),Shared.pack_task(model),task,task_exact) :
      {(task,bias) == (_,Buffer.pack(~Tile.Tile8,bias_model)) : NativeResult()}
    %Equal.sym(Products.Cells(),Buffer.pack(~Tile.Tile8,bias_model),bias,bias_exact) : {(task,bias) == (task,_) : NativeResult()}
    {==}}

def observe(native: NativeResult()) -> Observed<native>:
  (task,bias) = native
  observed_result(task,bias,Shared.observe_task(task),Buffer.observe(~Tile.Tile8,bias))

def leaf(~kernel: Products.Kernel(),+work: Products.Shape,+first: U32,+count: U32,+operands: Operands,+bias: Buffer.Model<Tile.Tile8>) -> NativeResult():
  Products.panel_leaf(~kernel,work,first,count,pack_operands(operands),Buffer.pack(~Tile.Tile8,bias))

def walked(~kernel: Products.Kernel(),+work: Products.Shape,+tree: OperandTree(),lefts: Lefts(),+bias: Buffer.Model<Tile.Tile8>) -> NativeWalk():
  Pieces.walked(~NativeOperand(),~Output(),~Products.Cells(),~Products.Shape,~Products.panel_leaf(~kernel),work,
    pack_tree(tree),Pieces.Walking{lefts,Buffer.pack(~Tile.Tile8,bias)})

def panel_regions(+work: Products.Shape,tree: OperandTree()) -> Pieces.Ranges:
  match tree:
    case Partition.Leaf{operands,+first,count,details}: Point.clipped_rectangle(Point.panel_geometry(work,first))
    case Partition.Fork{+left,+right}: Pieces.Split{panel_regions(work,left),panel_regions(work,right)}

def emitted_cells(~run: Point.Run(),+work: Products.Shape,+first: U32,+operands: Operands,+bias: Buffer.Model<Tile.Tile8>) -> Model.Content<Tile.Tile8>:
  Operands{+input,+weights} = operands
  Point.panel_work_model(~run,work,first,Held.pack(~Tile.Tile8,input),Held.pack(~Tile.Tile8,weights),bias)

type Emitted<-run: Point.Run(),-work: Products.Shape,-first: U32,-operands: Operands,-output: Shared.TaskModel(),-origin: Buffer.Content<Tile.Tile8>> is Data:
  Emitted{bias: Buffer.Model<Tile.Tile8>,contents: {Point.task_contents(Shared.pack_task(output)) == emitted_cells(~run,work,first,operands,bias) : Model.Content<Tile.Tile8>},
    same_bias: {Buffer.content(~Tile.Tile8,bias) == origin : Buffer.Content<Tile.Tile8>}}

def Emissions(~run: Point.Run(),+work: Products.Shape,tree: OperandTree(),output: Shared.TaskModel(),+origin: Buffer.Content<Tile.Tile8>) -> Data:
  match tree output:
    case Partition.Leaf{+operands,+first,+count,+details} output:
      Emitted<run,work,first,operands,output,origin>
    case Partition.Fork{+left,+right} Partition.Fork{+left_output,+right_output}:
      Held.Both<Emissions(~run,work,left,left_output,origin),Emissions(~run,work,right,right_output,origin)>
    case Partition.Fork{left,right} Partition.Leaf{pieces,first,count,details}: Empty

type Walked<-run: Point.Run(),-work: Products.Shape,-tree: OperandTree(),-lefts: Lefts(),-native: NativeWalk(),-origin: Buffer.Content<Tile.Tile8>> is Data:
  Walked{output: Shared.TaskModel(),bias: Buffer.Model<Tile.Tile8>,
    exact: {native == Pieces.Walked{Shared.pack_task(output),lefts,Buffer.pack(~Tile.Tile8,bias)} : NativeWalk()},
    regions: {Point.task_regions(Shared.pack_task(output)) == panel_regions(work,tree) : Pieces.Ranges},emitted: Emissions(~run,work,tree,output,origin),
    same_bias: {Buffer.content(~Tile.Tile8,bias) == origin : Buffer.Content<Tile.Tile8>}}

def walked_leaf(~kernel: Products.Kernel(),~run: Point.Run(),~agrees: Point.BlockAgreement(~kernel,~run),~keeps: Point.BiasStep(~run),
  +work: Products.Shape,+first: U32,+count: U32,+operands: Operands,-lefts: Lefts(),+bias: Buffer.Model<Tile.Tile8>,
  seen: Observed<leaf(~kernel,work,first,count,operands,bias)>) ->
  Walked<run,work,Partition.Leaf{operands,first,count,Unit{}},lefts,walked(~kernel,work,Partition.Leaf{operands,first,count,Unit{}},lefts,bias),Buffer.content(~Tile.Tile8,bias)>:
  Operands{+input,+weights} = operands
  Observed{+output,+rest,+exact} = seen
  -native = leaf(~kernel,work,first,count,operands,bias)
  -packed = {(Shared.pack_task(output),Buffer.pack(~Tile.Tile8,rest)) : NativeResult()}
  +computed = {%exact : {Point.result_contents(~Products.Cells(),_) == emitted_cells(~run,work,first,operands,bias) : Model.Content<Tile.Tile8>}
    Point.panel_leaf_contents(~kernel,~run,~agrees,work,first,count,input,weights,bias) :
    {Point.task_contents(Shared.pack_task(output)) == emitted_cells(~run,work,first,operands,bias) : Model.Content<Tile.Tile8>}}
  Walked{output,rest,
    %Equal.sym(NativeResult(),native,packed,exact) :
      {Pieces.leaf_walked(~Output(),~Products.Cells(),lefts,_) == Pieces.Walked{Shared.pack_task(output),lefts,Buffer.pack(~Tile.Tile8,rest)} : NativeWalk()}
    {==},
    %exact : {Point.result_regions(~Products.Cells(),_) == Point.clipped_rectangle(Point.panel_geometry(work,first)) : Pieces.Ranges}
    Point.panel_leaf_layout(~kernel,~run,~agrees,work,first,count,input,weights,bias),Emitted{bias,computed,{==}},
    %Buffer.packed_contents(~Tile.Tile8,rest) :
      {_ == Buffer.content(~Tile.Tile8,bias) : Buffer.Content<Tile.Tile8>}
    %exact : {Point.returned_bias(_) == Buffer.content(~Tile.Tile8,bias) : Buffer.Content<Tile.Tile8>}
    Point.panel_leaf_bias(~kernel,~run,~agrees,~keeps,work,first,count,input,weights,bias)}

def fork_regions(+left: Shared.TaskModel(),+right: Shared.TaskModel(),+left_region: Pieces.Ranges,+right_region: Pieces.Ranges,
  +left_exact: {Point.task_regions(Shared.pack_task(left)) == left_region : Pieces.Ranges},
  +right_exact: {Point.task_regions(Shared.pack_task(right)) == right_region : Pieces.Ranges}) ->
  {Point.task_regions(Shared.pack_task(Partition.Fork{left,right})) == Pieces.Split{left_region,right_region} : Pieces.Ranges}:
  +l = Point.task_regions(Shared.pack_task(left))
  +r = Point.task_regions(Shared.pack_task(right))
  %left_exact : {Pieces.Split{l,r} == Pieces.Split{_,right_region} : Pieces.Ranges}
  %right_exact : {Pieces.Split{l,r} == Pieces.Split{l,_} : Pieces.Ranges}
  {==}

def walked_joined(~kernel: Products.Kernel(),~run: Point.Run(),+work: Products.Shape,+left: OperandTree(),+right: OperandTree(),-lefts: Lefts(),
  +bias: Buffer.Model<Tile.Tile8>,+left_out: Shared.TaskModel(),+left_bias: Buffer.Model<Tile.Tile8>,
  +left_exact: {walked(~kernel,work,left,lefts,bias) == Pieces.Walked{Shared.pack_task(left_out),lefts,Buffer.pack(~Tile.Tile8,left_bias)} : NativeWalk()},
  +left_regions: {Point.task_regions(Shared.pack_task(left_out)) == panel_regions(work,left) : Pieces.Ranges},
  +left_emitted: Emissions(~run,work,left,left_out,Buffer.content(~Tile.Tile8,bias)),
  +left_same: {Buffer.content(~Tile.Tile8,left_bias) == Buffer.content(~Tile.Tile8,bias) : Buffer.Content<Tile.Tile8>},
  right_seen: Walked<run,work,right,Pieces.Left{Shared.pack_task(left_out),lefts},walked(~kernel,work,right,Pieces.Left{Shared.pack_task(left_out),lefts},left_bias),Buffer.content(~Tile.Tile8,left_bias)>) ->
  Walked<run,work,Partition.Fork{left,right},lefts,walked(~kernel,work,Partition.Fork{left,right},lefts,bias),Buffer.content(~Tile.Tile8,bias)>:
  Walked{+right_out,+right_bias,+right_exact,+right_regions,+right_emitted,+right_same} = right_seen
  -after_left = {Pieces.Walked{Shared.pack_task(left_out),lefts,Buffer.pack(~Tile.Tile8,left_bias)} : NativeWalk()}
  -right_walked = walked(~kernel,work,right,Pieces.Left{Shared.pack_task(left_out),lefts},left_bias)
  -after_right = {Pieces.Walked{Shared.pack_task(right_out),Pieces.Left{Shared.pack_task(left_out),lefts},Buffer.pack(~Tile.Tile8,right_bias)} : NativeWalk()}
  -result = {Pieces.Walked{Shared.pack_task(Partition.Fork{left_out,right_out}),lefts,Buffer.pack(~Tile.Tile8,right_bias)} : NativeWalk()}
  Walked{Partition.Fork{left_out,right_out},right_bias,
    %Equal.sym(NativeWalk(),walked(~kernel,work,left,lefts,bias),after_left,left_exact) :
      {Pieces.joined(~Output(),~Products.Cells(),
        Pieces.walked(~NativeOperand(),~Output(),~Products.Cells(),~Products.Shape,~Products.panel_leaf(~kernel),work,
          pack_tree(right),Pieces.pushed(~Output(),~Products.Cells(),_))) == result : NativeWalk()}
    %Equal.sym(NativeWalk(),right_walked,after_right,right_exact) :
      {Pieces.joined(~Output(),~Products.Cells(),_) == result : NativeWalk()}
    {==},fork_regions(left_out,right_out,panel_regions(work,left),panel_regions(work,right),left_regions,right_regions),Held.Both{left_emitted,
      %left_same : Emissions(~run,work,right,right_out,_) right_emitted},
    Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,right_bias),Buffer.content(~Tile.Tile8,left_bias),Buffer.content(~Tile.Tile8,bias),right_same,left_same)}

def walk_right(~kernel: Products.Kernel(),~run: Point.Run(),+work: Products.Shape,+left: OperandTree(),+right: OperandTree(),-lefts: Lefts(),
  +bias: Buffer.Model<Tile.Tile8>,seen: Walked<run,work,left,lefts,walked(~kernel,work,left,lefts,bias),Buffer.content(~Tile.Tile8,bias)>,
  continue: @-later: Lefts() -> @+state: Buffer.Model<Tile.Tile8> -> Walked<run,work,right,later,walked(~kernel,work,right,later,state),Buffer.content(~Tile.Tile8,state)>) ->
  Walked<run,work,Partition.Fork{left,right},lefts,walked(~kernel,work,Partition.Fork{left,right},lefts,bias),Buffer.content(~Tile.Tile8,bias)>:
  Walked{+left_out,+left_bias,+left_exact,+left_regions,+left_emitted,+left_same} = seen
  walked_joined(~kernel,~run,work,left,right,lefts,bias,left_out,left_bias,left_exact,left_regions,left_emitted,left_same,continue(Pieces.Left{Shared.pack_task(left_out),lefts},left_bias))

def serial(~kernel: Products.Kernel(),~run: Point.Run(),~agrees: Point.BlockAgreement(~kernel,~run),~keeps: Point.BiasStep(~run),
  +work: Products.Shape,tree: OperandTree(),-lefts: Lefts(),+bias: Buffer.Model<Tile.Tile8>) ->
  Walked<run,work,tree,lefts,walked(~kernel,work,tree,lefts,bias),Buffer.content(~Tile.Tile8,bias)>:
  match tree:
    case Partition.Leaf{+operands,+first,+count,Unit{}}:
      walked_leaf(~kernel,~run,~agrees,~keeps,work,first,count,operands,lefts,bias,observe(leaf(~kernel,work,first,count,operands,bias)))
    case Partition.Fork{+left,+right}:
      walk_right(~kernel,~run,work,left,right,lefts,bias,serial(~kernel,~run,~agrees,~keeps,work,left,lefts,bias),
        later => state => serial(~kernel,~run,~agrees,~keeps,work,right,later,state))

def State() -> Type:
  Pieces.Pieced(~NativeOperand(),~Products.Cells())

def each(~kernel: Products.Kernel(),+work: Products.Shape,+tasks: Pieces.Ranges,+mode: Bool,+enabled: Bool,state: State()) -> Point.TaskOutput():
  Pieces.each_part(~State(),~Output(),~Unit,~Products.Shape,~Bool,~Pieces.in_parallel,
    ~Pieces.piece_divide(~NativeOperand(),~Products.Cells(),~Products.Shape,~Clone.clone(~Tile.Tile8)),
    ~Pieces.task_walk(~NativeOperand(),~Output(),~Products.Cells(),~Products.Shape,~Products.panel_leaf(~kernel)),
    ~Pieces.task_kept(~NativeOperand(),~Output(),~Products.Cells(),~Products.Shape,~Products.panel_leaf(~kernel)),tasks,mode,enabled,work,state)

def serial_output(~kernel: Products.Kernel(),+work: Products.Shape,+tree: OperandTree(),+bias: Buffer.Model<Tile.Tile8>) -> Point.TaskOutput():
  Pair.fst(Point.TaskOutput(),Products.Cells(),Pieces.walk_result(~Output(),~Products.Cells(),walked(~kernel,work,tree,Pieces.NoLefts{},bias)))

type Grouped<-run: Point.Run(),-work: Products.Shape,-tree: OperandTree(),-native: Point.TaskOutput(),-origin: Buffer.Content<Tile.Tile8>> is Data:
  Grouped{model: Shared.TaskModel(),exact: {native == Shared.pack_task(model) : Point.TaskOutput()},
    regions: {Point.task_regions(Shared.pack_task(model)) == panel_regions(work,tree) : Pieces.Ranges},emitted: Emissions(~run,work,tree,model,origin)}

def observed_serial(~kernel: Products.Kernel(),~run: Point.Run(),+work: Products.Shape,+tree: OperandTree(),+bias: Buffer.Model<Tile.Tile8>,
  seen: Walked<run,work,tree,Pieces.NoLefts{},walked(~kernel,work,tree,Pieces.NoLefts{},bias),Buffer.content(~Tile.Tile8,bias)>) ->
  Grouped<run,work,tree,serial_output(~kernel,work,tree,bias),Buffer.content(~Tile.Tile8,bias)>:
  Walked{+model,+rest,+exact,+regions,+emitted,same_bias} = seen
  -native = walked(~kernel,work,tree,Pieces.NoLefts{},bias)
  -result = {Pieces.Walked{Shared.pack_task(model),Pieces.NoLefts{},Buffer.pack(~Tile.Tile8,rest)} : NativeWalk()}
  Grouped{model,
    %Equal.sym(NativeWalk(),native,result,exact) :
      {Pair.fst(Point.TaskOutput(),Products.Cells(),Pieces.walk_result(~Output(),~Products.Cells(),_)) == Shared.pack_task(model) : Point.TaskOutput()}
    {==},regions,emitted}

def ordered_fork(~kernel: Products.Kernel(),+work: Products.Shape,+left: Pieces.Ranges,+right: Pieces.Ranges,+mode: Bool,+enabled: Bool,-state: State()) ->
  {each(~kernel,work,Pieces.Split{left,right},mode,enabled,state) ==
    Walk.sequential(~State(),~Output(),~Unit,~Products.Shape,~Bool,~Pieces.in_parallel,
      ~Pieces.piece_divide(~NativeOperand(),~Products.Cells(),~Products.Shape,~Clone.clone(~Tile.Tile8)),
      ~Pieces.task_walk(~NativeOperand(),~Output(),~Products.Cells(),~Products.Shape,~Products.panel_leaf(~kernel)),
      ~Pieces.task_kept(~NativeOperand(),~Output(),~Products.Cells(),~Products.Shape,~Products.panel_leaf(~kernel)),left,right,enabled,work,
      Pieces.piece_divide(~NativeOperand(),~Products.Cells(),~Products.Shape,~Clone.clone(~Tile.Tile8),work,left,right,state)) : Point.TaskOutput()}:
  Walk.fork(~State(),~Output(),~Unit,~Products.Shape,~Bool,~Pieces.in_parallel,
    ~Pieces.piece_divide(~NativeOperand(),~Products.Cells(),~Products.Shape,~Clone.clone(~Tile.Tile8)),
    ~Pieces.task_walk(~NativeOperand(),~Output(),~Products.Cells(),~Products.Shape,~Products.panel_leaf(~kernel)),
    ~Pieces.task_kept(~NativeOperand(),~Output(),~Products.Cells(),~Products.Shape,~Products.panel_leaf(~kernel)),left,right,mode,enabled,work,state)

def joined_groups(~run: Point.Run(),+work: Products.Shape,+left: OperandTree(),+right: OperandTree(),-left_native: Point.TaskOutput(),-right_native: Point.TaskOutput(),
  +origin: Buffer.Content<Tile.Tile8>,+low: Buffer.Content<Tile.Tile8>,+high: Buffer.Content<Tile.Tile8>,
  +low_same: {low == origin : Buffer.Content<Tile.Tile8>},+high_same: {high == origin : Buffer.Content<Tile.Tile8>},
  +left_seen: Grouped<run,work,left,left_native,low>,+right_seen: Grouped<run,work,right,right_native,high>) ->
  Grouped<run,work,Partition.Fork{left,right},Partition.Fork{left_native,right_native},origin>:
  Grouped{+left_out,+left_exact,+left_regions,+left_emitted} = left_seen
  Grouped{+right_out,+right_exact,+right_regions,+right_emitted} = right_seen
  Grouped{Partition.Fork{left_out,right_out},
    %left_exact : {Partition.Fork{left_native,right_native} == Partition.Fork{_,Shared.pack_task(right_out)} : Point.TaskOutput()}
    %right_exact : {Partition.Fork{left_native,right_native} == Partition.Fork{left_native,_} : Point.TaskOutput()}
    {==},fork_regions(left_out,right_out,panel_regions(work,left),panel_regions(work,right),left_regions,right_regions),Held.Both{
      %low_same : Emissions(~run,work,left,left_out,_) left_emitted,
      %high_same : Emissions(~run,work,right,right_out,_) right_emitted}}

def NativeBiasPair() -> Type:
  Products.Cells() & Products.Cells()

def joined_workers(~kernel: Products.Kernel(),~run: Point.Run(),+work: Products.Shape,+left: OperandTree(),+right: OperandTree(),+left_plan: Pieces.Ranges,+right_plan: Pieces.Ranges,
  +mode: Bool,+enabled: Bool,+bias: Buffer.Model<Tile.Tile8>,+copies: Cloned.Observation<Tile.Tile8,Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,bias))>,
  continue: @+lower: Buffer.Model<Tile.Tile8> -> @+upper: Buffer.Model<Tile.Tile8> ->
    @+left_same: {Buffer.content(~Tile.Tile8,lower) == Buffer.content(~Tile.Tile8,bias) : Buffer.Content<Tile.Tile8>} ->
    @+right_same: {Buffer.content(~Tile.Tile8,upper) == Buffer.content(~Tile.Tile8,bias) : Buffer.Content<Tile.Tile8>} ->
    Grouped<run,work,Partition.Fork{left,right},Partition.Fork{each(~kernel,work,left_plan,enabled,enabled,(pack_tree(left),Buffer.pack(~Tile.Tile8,lower))),
      each(~kernel,work,right_plan,enabled,enabled,(pack_tree(right),Buffer.pack(~Tile.Tile8,upper)))},Buffer.content(~Tile.Tile8,bias)>) ->
  Grouped<run,work,Partition.Fork{left,right},each(~kernel,work,Pieces.Split{left_plan,right_plan},mode,enabled,
    (pack_tree(Partition.Fork{left,right}),Buffer.pack(~Tile.Tile8,bias))),Buffer.content(~Tile.Tile8,bias)>:
  Cloned.Observation{+lower,+upper,+exact} = copies
  +tasks = {Pieces.Split{left_plan,right_plan} : Pieces.Ranges}
  +tree = {Partition.Fork{left,right} : OperandTree()}
  -state = {(pack_tree(tree),Buffer.pack(~Tile.Tile8,bias)) : State()}
  -clones = Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,bias))
  -packed = {(Buffer.pack(~Tile.Tile8,lower),Buffer.pack(~Tile.Tile8,upper)) : NativeBiasPair()}
  -sequential = Walk.sequential(~State(),~Output(),~Unit,~Products.Shape,~Bool,~Pieces.in_parallel,
    ~Pieces.piece_divide(~NativeOperand(),~Products.Cells(),~Products.Shape,~Clone.clone(~Tile.Tile8)),
    ~Pieces.task_walk(~NativeOperand(),~Output(),~Products.Cells(),~Products.Shape,~Products.panel_leaf(~kernel)),
    ~Pieces.task_kept(~NativeOperand(),~Output(),~Products.Cells(),~Products.Shape,~Products.panel_leaf(~kernel)),left_plan,right_plan,enabled,work,
    Pieces.shared_halves(~NativeOperand(),~Products.Cells(),pack_tree(left),pack_tree(right),clones))
  %Equal.sym(Point.TaskOutput(),each(~kernel,work,tasks,mode,enabled,state),sequential,ordered_fork(~kernel,work,left_plan,right_plan,mode,enabled,state)) :
    Grouped<run,work,tree,_,Buffer.content(~Tile.Tile8,bias)>
  %exact : Grouped<run,work,tree,Walk.sequential(~State(),~Output(),~Unit,~Products.Shape,~Bool,~Pieces.in_parallel,
    ~Pieces.piece_divide(~NativeOperand(),~Products.Cells(),~Products.Shape,~Clone.clone(~Tile.Tile8)),
    ~Pieces.task_walk(~NativeOperand(),~Output(),~Products.Cells(),~Products.Shape,~Products.panel_leaf(~kernel)),
    ~Pieces.task_kept(~NativeOperand(),~Output(),~Products.Cells(),~Products.Shape,~Products.panel_leaf(~kernel)),left_plan,right_plan,enabled,work,
    Pieces.shared_halves(~NativeOperand(),~Products.Cells(),pack_tree(left),pack_tree(right),_)),Buffer.content(~Tile.Tile8,bias)>
  continue(lower,upper,Cloned.left_contents(~Tile.Tile8,bias,lower,upper,exact),Cloned.right_contents(~Tile.Tile8,bias,lower,upper,exact))

# A task may keep a whole image subtree or divide it. Both traverse its images.
def grouped(~kernel: Products.Kernel(),~run: Point.Run(),~agrees: Point.BlockAgreement(~kernel,~run),~keeps: Point.BiasStep(~run),
  +work: Products.Shape,tasks: Pieces.Ranges,+mode: Bool,+enabled: Bool,tree: OperandTree(),+bias: Buffer.Model<Tile.Tile8>) ->
  Grouped<run,work,tree,each(~kernel,work,tasks,mode,enabled,(pack_tree(tree),Buffer.pack(~Tile.Tile8,bias))),Buffer.content(~Tile.Tile8,bias)>:
  match tasks tree:
    case Pieces.Range{first,count} +tree:
      observed_serial(~kernel,~run,work,tree,bias,serial(~kernel,~run,~agrees,~keeps,work,tree,Pieces.NoLefts{},bias))
    case Pieces.Split{+left_plan,+right_plan} Partition.Leaf{+operands,+first,+count,Unit{}}:
      +tree = {Partition.Leaf{operands,first,count,Unit{}} : OperandTree()}
      -state = {(pack_tree(tree),Buffer.pack(~Tile.Tile8,bias)) : State()}
      -sequential = Walk.sequential(~State(),~Output(),~Unit,~Products.Shape,~Bool,~Pieces.in_parallel,
        ~Pieces.piece_divide(~NativeOperand(),~Products.Cells(),~Products.Shape,~Clone.clone(~Tile.Tile8)),
        ~Pieces.task_walk(~NativeOperand(),~Output(),~Products.Cells(),~Products.Shape,~Products.panel_leaf(~kernel)),
        ~Pieces.task_kept(~NativeOperand(),~Output(),~Products.Cells(),~Products.Shape,~Products.panel_leaf(~kernel)),
        left_plan,right_plan,enabled,work,Pieces.Whole{state})
      %Equal.sym(Point.TaskOutput(),each(~kernel,work,Pieces.Split{left_plan,right_plan},mode,enabled,state),sequential,
        ordered_fork(~kernel,work,left_plan,right_plan,mode,enabled,state)) : Grouped<run,work,tree,_,Buffer.content(~Tile.Tile8,bias)>
      observed_serial(~kernel,~run,work,tree,bias,serial(~kernel,~run,~agrees,~keeps,work,tree,Pieces.NoLefts{},bias))
    case Pieces.Split{+left_plan,+right_plan} Partition.Fork{+left,+right}:
      joined_workers(~kernel,~run,work,left,right,left_plan,right_plan,mode,enabled,bias,Cloned.observe_pair(~Tile.Tile8,Clone.clone(~Tile.Tile8,Buffer.pack(~Tile.Tile8,bias))),
        lower => upper => left_same => right_same =>
          joined_groups(~run,work,left,right,
            each(~kernel,work,left_plan,enabled,enabled,(pack_tree(left),Buffer.pack(~Tile.Tile8,lower))),
            each(~kernel,work,right_plan,enabled,enabled,(pack_tree(right),Buffer.pack(~Tile.Tile8,upper))),
            Buffer.content(~Tile.Tile8,bias),Buffer.content(~Tile.Tile8,lower),Buffer.content(~Tile.Tile8,upper),left_same,right_same,
            grouped(~kernel,~run,~agrees,~keeps,work,left_plan,enabled,enabled,left,lower),
            grouped(~kernel,~run,~agrees,~keeps,work,right_plan,enabled,enabled,right,upper)))

def image_width(+work: Products.Shape) -> U32:
  Products.Shape{+rows,inner,+channels,grid} = work
  (rows * Gemm.blocks(channels) : U32)

def panel_grid(+work: Products.Shape) -> Grid.Grid:
  Products.Shape{rows,inner,channels,grid} = work
  grid

def image_leaf_laid(+limits: Machine.Limits,+work: Products.Shape,+images: U32,+grid_facts: GridLaws.GridFacts<limits,panel_grid(work)>,
  +grid_exact: {GridLaws.grid_cells(panel_grid(work)) == Nat.mul(U32.to_nat(images),U32.to_nat(image_width(work))) : Nat},
  +width_positive: {Nat.is_lt(0n,U32.to_nat(image_width(work))) == True{} : Bool},
  +room: {Nat.is_le(GridLaws.grid_cells(panel_grid(work)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +output_run: Runs.Run(Pieces.ranges_of_grid(panel_grid(work)),1n,GridLaws.grid_cells(panel_grid(work))),
  +first: U32,+count: U32,+one: {count == 1 : U32},
  +inside: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(images)) == True{} : Bool}) ->
  Shared.Laid(Pieces.ranges_of_grid(panel_grid(work)),Point.clipped_rectangle(Point.panel_geometry(work,first)),
    Nat.mul(U32.to_nat(first),U32.to_nat(image_width(work))),Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(image_width(work)))):
  Products.Shape{+rows,+inner,+channels,+grid} = work
  +groups = Gemm.blocks(channels)
  +width_word = image_width(work)
  +width = U32.to_nat(width_word)
  +first_cell = Nat.mul(U32.to_nat(first),width)
  +last_cell = Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(count)),width)
  +grid_total = GridLaws.grid_cells(grid)
  +last_inside = {%Equal.sym(Nat,grid_total,Nat.mul(U32.to_nat(images),width),grid_exact) : {Nat.is_le(last_cell,_) == True{} : Bool}
    Index.mul_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(images),width,inside) : {Nat.is_le(last_cell,grid_total) == True{} : Bool}}
  +first_inside = Order.transitive(first_cell,last_cell,grid_total,Index.mul_le(U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),width,
    Order.le_add(U32.to_nat(first),U32.to_nat(count))),last_inside)
  +base = (first * width_word : U32)
  +base_exact = Multiply.bounded_multiply(first,width_word,Machine.capacity(limits),Limits.capacity_fits(limits),
    Order.transitive(first_cell,grid_total,U32.to_nat(Machine.capacity(limits)),first_inside,room))
  +end_exact = Equal.trans(Nat,last_cell,Nat.add(first_cell,Nat.mul(U32.to_nat(count),width)),Nat.add(first_cell,width),
    Index.distrib(U32.to_nat(first),U32.to_nat(count),width),
    %Equal.sym(U32,count,1,one) : {Nat.add(first_cell,Nat.mul(U32.to_nat(_),width)) == Nat.add(first_cell,width) : Nat}
    Equal.cong(Nat,Nat,value => Nat.add(first_cell,value),Nat.add(width,0n),width,Addition.zero_right(width)))
  +native_end = Nat.add(U32.to_nat(base),width)
  +native_exact = Equal.trans(Nat,last_cell,Nat.add(first_cell,width),native_end,end_exact,
    Equal.cong(Nat,Nat,value => Nat.add(value,width),first_cell,U32.to_nat(base),Equal.sym(Nat,U32.to_nat(base),first_cell,base_exact)))
  %Equal.sym(U32,(first * rows * groups : U32),base,Arithmetic.multiply_associative(first,rows,groups)) :
    Shared.Laid(Pieces.ranges_of_grid(grid),Shared.clipped_region(grid,_,width_word),first_cell,last_cell)
  %base_exact : Shared.Laid(Pieces.ranges_of_grid(grid),Shared.clipped_region(grid,base,width_word),_,last_cell)
  %Equal.sym(Nat,last_cell,native_end,native_exact) :
    Shared.Laid(Pieces.ranges_of_grid(grid),Shared.clipped_region(grid,base,width_word),U32.to_nat(base),_)
  Shared.clipped_laid(limits,grid,grid_facts,1n,grid_total,output_run,base,width_word,width_positive,
    %native_exact : {Nat.is_le(_,grid_total) == True{} : Bool} last_inside)

def panel_tree_laid(+limits: Machine.Limits,+work: Products.Shape,+images: U32,+grid_facts: GridLaws.GridFacts<limits,panel_grid(work)>,
  +grid_exact: {GridLaws.grid_cells(panel_grid(work)) == Nat.mul(U32.to_nat(images),U32.to_nat(image_width(work))) : Nat},
  +width_positive: {Nat.is_lt(0n,U32.to_nat(image_width(work))) == True{} : Bool},
  +room: {Nat.is_le(GridLaws.grid_cells(panel_grid(work)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +output_run: Runs.Run(Pieces.ranges_of_grid(panel_grid(work)),1n,GridLaws.grid_cells(panel_grid(work))),
  tree: OperandTree(),+run: Runs.Run(operand_ranges(tree),1n,U32.to_nat(images)),+ones: Singletons(operand_ranges(tree))) ->
  Shared.Laid(Pieces.ranges_of_grid(panel_grid(work)),panel_regions(work,tree),
    Nat.mul(Runs.range_low(operand_ranges(tree)),U32.to_nat(image_width(work))),
    Nat.mul(Runs.range_high(operand_ranges(tree)),U32.to_nat(image_width(work)))):
  match tree:
    case Partition.Leaf{+operands,+first,+count,Unit{}}:
      image_leaf_laid(limits,work,images,grid_facts,grid_exact,width_positive,room,output_run,first,count,ones,
        Runs.fits_inside(1n,U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(images),run))
    case Partition.Fork{+left,+right}:
      Held.Both{+left_run,rest} = run
      Held.Both{+right_run,+glue} = rest
      Held.Both{left_ones,right_ones} = ones
      +width = U32.to_nat(image_width(work))
      +left_low = Runs.range_low(operand_ranges(left))
      +left_high = Runs.range_high(operand_ranges(left))
      +right_low = Runs.range_low(operand_ranges(right))
      +right_high = Runs.range_high(operand_ranges(right))
      Shared.joined_laid(Pieces.ranges_of_grid(panel_grid(work)),panel_regions(work,left),panel_regions(work,right),
        Nat.mul(left_low,width),Nat.mul(left_high,width),Nat.mul(right_low,width),Nat.mul(right_high,width),
        Equal.cong(Nat,Nat,index => Nat.mul(index,width),left_high,right_low,glue),
        panel_tree_laid(limits,work,images,grid_facts,grid_exact,width_positive,room,output_run,left,left_run,left_ones),
        panel_tree_laid(limits,work,images,grid_facts,grid_exact,width_positive,room,output_run,right,right_run,right_ones))

def model_laid(+limits: Machine.Limits,+work: Products.Shape,+images: U32,+grid_facts: GridLaws.GridFacts<limits,panel_grid(work)>,
  +grid_exact: {GridLaws.grid_cells(panel_grid(work)) == Nat.mul(U32.to_nat(images),U32.to_nat(image_width(work))) : Nat},
  +width_positive: {Nat.is_lt(0n,U32.to_nat(image_width(work))) == True{} : Bool},
  +room: {Nat.is_le(GridLaws.grid_cells(panel_grid(work)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +output_run: Runs.Run(Pieces.ranges_of_grid(panel_grid(work)),1n,GridLaws.grid_cells(panel_grid(work))),
  +tree: OperandTree(),+images_run: ImageRun(operand_ranges(tree),0,images,U32.to_nat(images))) ->
  Shared.Laid(Pieces.ranges_of_grid(panel_grid(work)),panel_regions(work,tree),0n,GridLaws.grid_cells(panel_grid(work))):
  Held.Both{+plan,+ones} = images_run
  Held.Both{+run,+ends} = plan
  Held.Both{+first,+last} = ends
  +width = U32.to_nat(image_width(work))
  +target = Pieces.ranges_of_grid(panel_grid(work))
  +ranges = operand_ranges(tree)
  %first : Shared.Laid(target,panel_regions(work,tree),Nat.mul(_,width),GridLaws.grid_cells(panel_grid(work)))
  %Equal.sym(Nat,GridLaws.grid_cells(panel_grid(work)),Nat.mul(U32.to_nat(images),width),grid_exact) :
    Shared.Laid(target,panel_regions(work,tree),Nat.mul(Runs.range_low(ranges),width),_)
  %last : Shared.Laid(target,panel_regions(work,tree),Nat.mul(Runs.range_low(ranges),width),Nat.mul(_,width))
  panel_tree_laid(limits,work,images,grid_facts,grid_exact,width_positive,room,output_run,tree,run,ones)

type PanelOutput<-run: Point.Run(),-work: Products.Shape,-ranges: Pieces.Ranges,-native: Pieces.Pieces(~Tile.Tile8),-origin: Buffer.Content<Tile.Tile8>,
  -inputs: Windows.WindowModel<Tile.Tile8>,-weights: Windows.WindowModel<Tile.Tile8>> is Data:
  PanelOutput{model: Shared.TaskModel(),placed: {native == Tensors.placed(~Tile.Tile8,Pieces.ranges_of_grid(panel_grid(work)),Products.zero(),Shared.pack_task(model)) : Pieces.Pieces(~Tile.Tile8)},
    writes: Shared.Laid(Pieces.ranges_of_grid(panel_grid(work)),Point.task_regions(Shared.pack_task(model)),0n,GridLaws.grid_cells(panel_grid(work))),
    operands: OperandTree(),arranged: {operand_ranges(operands) == ranges : Pieces.Ranges},emitted: Emissions(~run,work,operands,model,origin),
    input_windows: {operand_windows(~input_holding,operands) == inputs : Windows.WindowModel<Tile.Tile8>},
    weight_windows: {operand_windows(~weight_holding,operands) == weights : Windows.WindowModel<Tile.Tile8>}}
def placed_panels(~run: Point.Run(),+work: Products.Shape,+tree: OperandTree(),+ranges: Pieces.Ranges,
  +ranges_exact: {operand_ranges(tree) == ranges : Pieces.Ranges},-native: Point.TaskOutput(),
  +laid: Shared.Laid(Pieces.ranges_of_grid(panel_grid(work)),panel_regions(work,tree),0n,GridLaws.grid_cells(panel_grid(work))),
  +inputs: Windows.WindowModel<Tile.Tile8>,+weights: Windows.WindowModel<Tile.Tile8>,
  +input_windows: {operand_windows(~input_holding,tree) == inputs : Windows.WindowModel<Tile.Tile8>},
  +weight_windows: {operand_windows(~weight_holding,tree) == weights : Windows.WindowModel<Tile.Tile8>},
  +origin: Buffer.Content<Tile.Tile8>,+checked: Grouped<run,work,tree,native,origin>) ->
  PanelOutput<run,work,ranges,Tensors.placed(~Tile.Tile8,Pieces.ranges_of_grid(panel_grid(work)),Products.zero(),native),origin,inputs,weights>:
  Grouped{+model,+exact,+regions,+emitted} = checked
  +grid = panel_grid(work)
  PanelOutput{model,
    %exact : {Tensors.placed(~Tile.Tile8,Pieces.ranges_of_grid(grid),Products.zero(),native) == Tensors.placed(~Tile.Tile8,Pieces.ranges_of_grid(grid),Products.zero(),_) : Pieces.Pieces(~Tile.Tile8)}
    {==},
    %Equal.sym(Pieces.Ranges,Point.task_regions(Shared.pack_task(model)),panel_regions(work,tree),regions) :
      Shared.Laid(Pieces.ranges_of_grid(grid),_,0n,GridLaws.grid_cells(grid))
    laid,tree,ranges_exact,emitted,input_windows,weight_windows}

def placed_joint(~kernel: Products.Kernel(),~run: Point.Run(),~agrees: Point.BlockAgreement(~kernel,~run),~keeps: Point.BiasStep(~run),+limits: Machine.Limits,+work: Products.Shape,
  +images: U32,+grid_facts: GridLaws.GridFacts<limits,panel_grid(work)>,
  +grid_exact: {GridLaws.grid_cells(panel_grid(work)) == Nat.mul(U32.to_nat(images),U32.to_nat(image_width(work))) : Nat},
  +width_positive: {Nat.is_lt(0n,U32.to_nat(image_width(work))) == True{} : Bool},
  +room: {Nat.is_le(GridLaws.grid_cells(panel_grid(work)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +output_run: Runs.Run(Pieces.ranges_of_grid(panel_grid(work)),1n,GridLaws.grid_cells(panel_grid(work))),
  +ranges: Pieces.Ranges,+image_run: ImageRun(ranges,0,images,U32.to_nat(images)),
  +tasks: Pieces.Ranges,+enabled: Bool,+bias: Buffer.Model<Tile.Tile8>,-native: NativeOperands(),
  +inputs: Windows.WindowModel<Tile.Tile8>,+weights: Windows.WindowModel<Tile.Tile8>,+pair: Paired<ranges,native,inputs,weights>) ->
  PanelOutput<run,work,ranges,Tensors.placed(~Tile.Tile8,Pieces.ranges_of_grid(panel_grid(work)),Products.zero(),
    each(~kernel,work,tasks,enabled,enabled,(native,Buffer.pack(~Tile.Tile8,bias)))),Buffer.content(~Tile.Tile8,bias),inputs,weights>:
  Paired{+tree,+exact,+ranges_exact,+input_windows,+weight_windows} = pair
  +laid = model_laid(limits,work,images,grid_facts,grid_exact,width_positive,room,output_run,tree,
    %Equal.sym(Pieces.Ranges,operand_ranges(tree),ranges,ranges_exact) : ImageRun(_,0,images,U32.to_nat(images)) image_run)
  -packed = pack_tree(tree)
  %Equal.sym(NativeOperands(),native,packed,exact) :
    PanelOutput<run,work,ranges,Tensors.placed(~Tile.Tile8,Pieces.ranges_of_grid(panel_grid(work)),Products.zero(),each(~kernel,work,tasks,enabled,enabled,(_,Buffer.pack(~Tile.Tile8,bias)))),Buffer.content(~Tile.Tile8,bias),inputs,weights>
  placed_panels(~run,work,tree,ranges,ranges_exact,each(~kernel,work,tasks,enabled,enabled,(packed,Buffer.pack(~Tile.Tile8,bias))),laid,
    inputs,weights,input_windows,weight_windows,Buffer.content(~Tile.Tile8,bias),
    grouped(~kernel,~run,~agrees,~keeps,work,tasks,enabled,enabled,tree,bias))

def input_shape(+images: U32,+rows: U32,+inner: U32) -> List<&2,U32>:
  [images,rows,Gemm.blocks(inner)]

def weights_shape(+images: U32,+inner: U32,+channels: U32) -> List<&2,U32>:
  [images,inner,Gemm.blocks(channels)]

def output_shape(+images: U32,+rows: U32,+channels: U32) -> List<&2,U32>:
  [images,rows,Gemm.blocks(channels)]

def work_of(+images: U32,+rows: U32,+inner: U32,+channels: U32) -> Products.Shape:
  Products.Shape{rows,inner,channels,Tensors.grid_of(output_shape(images,rows,channels))}

def decision_of(+images: U32,+workers: U32,+work: Products.Shape) -> Tasks.Decision:
  Products.Shape{+rows,+inner,+channels,grid} = work
  Tasks.decide_over(Tasks.component_ranges(images),images,workers,1,U32.to_f32((rows * Gemm.blocks(channels) * inner : U32)),0.0)

def Reads(~Source: Data,~value: Source -> U32 -> Tile.Tile8,~reach: Halo.Reach(~Products.Shape),
  +source: Source,+work: Products.Shape,+images: U32,+shape: List<&2,U32>,-pieces: Pieces.Pieces(~Tile.Tile8)) -> Data:
  Windows.Windowed<Tile.Tile8,Source,value,source,Products.Shape,reach,work,ShapeLaws.elements(shape),Tasks.component_ranges(images),
    Halo.windows(~Tile.Tile8,~Products.Shape,~reach,work,Tensors.grid_of(shape),Products.zero(),Tasks.component_ranges(images),pieces)>

def read_windows(~Source: Data,~value: Source -> U32 -> Tile.Tile8,~reach: Halo.Reach(~Products.Shape),
  +source: Source,+work: Products.Shape,+images: U32,+shape: List<&2,U32>,-pieces: Pieces.Pieces(~Tile.Tile8),
  reads: Reads(~Source,~value,~reach,source,work,images,shape,pieces)) -> Windows.WindowModel<Tile.Tile8>:
  Windows.Windowed{windows,exact,holds} = reads
  windows

type PanelChecked<-run: Point.Run(),-Left: Data,-left_value: Left -> U32 -> Tile.Tile8,-Right: Data,-right_value: Right -> U32 -> Tile.Tile8,
  -input: Left,-weights: Right,-images: U32,-rows: U32,-inner: U32,-channels: U32,
  -left: Pieces.Pieces(~Tile.Tile8),-right: Pieces.Pieces(~Tile.Tile8),-native: Pieces.Pieces(~Tile.Tile8),-origin: Buffer.Content<Tile.Tile8>> is Data:
  PanelChecked{left_reads: Reads(~Left,~left_value,~Products.panel_left,input,work_of(images,rows,inner,channels),images,input_shape(images,rows,inner),left),
    right_reads: Reads(~Right,~right_value,~Products.panel_right,weights,work_of(images,rows,inner,channels),images,weights_shape(images,inner,channels),right),
    output: PanelOutput<run,work_of(images,rows,inner,channels),Tasks.component_ranges(images),native,origin,
      read_windows(~Left,~left_value,~Products.panel_left,input,work_of(images,rows,inner,channels),images,input_shape(images,rows,inner),left,left_reads),
      read_windows(~Right,~right_value,~Products.panel_right,weights,work_of(images,rows,inner,channels),images,weights_shape(images,inner,channels),right,right_reads)>}

def panel_tasks_output(~kernel: Products.Kernel(),~run: Point.Run(),~agrees: Point.BlockAgreement(~kernel,~run),~keeps: Point.BiasStep(~run),
  ~Left: Data,~left_value: Left -> U32 -> Tile.Tile8,~Right: Data,~right_value: Right -> U32 -> Tile.Tile8,
  +input: Left,+weights: Right,+limits: Machine.Limits,+images: U32,+rows: U32,+inner: U32,+channels: U32,+bias: Buffer.Model<Tile.Tile8>,
  +grid_facts: GridLaws.GridFacts<limits,Tensors.grid_of(output_shape(images,rows,channels))>,
  +grid_exact: {GridLaws.grid_cells(Tensors.grid_of(output_shape(images,rows,channels))) == Nat.mul(U32.to_nat(images),U32.to_nat(image_width(work_of(images,rows,inner,channels)))) : Nat},
  +width_positive: {Nat.is_lt(0n,U32.to_nat(image_width(work_of(images,rows,inner,channels)))) == True{} : Bool},
  +room: {Nat.is_le(GridLaws.grid_cells(Tensors.grid_of(output_shape(images,rows,channels))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +out_run: Runs.Run(Pieces.ranges_of_grid(Tensors.grid_of(output_shape(images,rows,channels))),1n,GridLaws.grid_cells(Tensors.grid_of(output_shape(images,rows,channels)))),
  +image_run: ImageRun(Tasks.component_ranges(images),0,images,U32.to_nat(images)),
  decision: Tasks.Decision,-left: Pieces.Pieces(~Tile.Tile8),-right: Pieces.Pieces(~Tile.Tile8),
  +left_reads: Reads(~Left,~left_value,~Products.panel_left,input,work_of(images,rows,inner,channels),images,input_shape(images,rows,inner),left),
  +right_reads: Reads(~Right,~right_value,~Products.panel_right,weights,work_of(images,rows,inner,channels),images,weights_shape(images,inner,channels),right)) ->
  PanelChecked<run,Left,left_value,Right,right_value,input,weights,images,rows,inner,channels,left,right,
    Products.panel_tasks(~kernel,work_of(images,rows,inner,channels),Tasks.component_ranges(images),Tensors.grid_of(input_shape(images,rows,inner)),
      Tensors.grid_of(weights_shape(images,inner,channels)),decision,left,right,Buffer.pack(~Tile.Tile8,bias)),Buffer.content(~Tile.Tile8,bias)>:
  Tasks.Decision{+enabled,+tasks} = decision
  Windows.Windowed{+left_windows,+left_exact,+left_ok} = left_reads
  Windows.Windowed{+right_windows,+right_exact,+right_ok} = right_reads
  +ranges = Tasks.component_ranges(images)
  +work = work_of(images,rows,inner,channels)
  +grid = panel_grid(work)
  +in_shape = input_shape(images,rows,inner)
  +w_shape = weights_shape(images,inner,channels)
  -left_native = Halo.windows(~Tile.Tile8,~Products.Shape,~Products.panel_left,work,Tensors.grid_of(in_shape),Products.zero(),ranges,left)
  -right_native = Halo.windows(~Tile.Tile8,~Products.Shape,~Products.panel_right,work,Tensors.grid_of(w_shape),Products.zero(),ranges,right)
  -left_packed = Windows.pack_windows(~Tile.Tile8,left_windows)
  -right_packed = Windows.pack_windows(~Tile.Tile8,right_windows)
  PanelChecked{left_reads,right_reads,
    %Equal.sym(Partition.Tree<&1,Move.Holding<Tile.Tile8>,Unit>,left_native,left_packed,left_exact) :
      PanelOutput<run,work,ranges,Tensors.placed(~Tile.Tile8,Pieces.ranges_of_grid(grid),Products.zero(),each(~kernel,work,tasks,enabled,enabled,
        (Pieces.zip(~Move.Holding<Tile.Tile8>,~Move.Holding<Tile.Tile8>,_,right_native),Buffer.pack(~Tile.Tile8,bias)))),Buffer.content(~Tile.Tile8,bias),left_windows,right_windows>
    %Equal.sym(Partition.Tree<&1,Move.Holding<Tile.Tile8>,Unit>,right_native,right_packed,right_exact) :
      PanelOutput<run,work,ranges,Tensors.placed(~Tile.Tile8,Pieces.ranges_of_grid(grid),Products.zero(),each(~kernel,work,tasks,enabled,enabled,
        (Pieces.zip(~Move.Holding<Tile.Tile8>,~Move.Holding<Tile.Tile8>,left_packed,_),Buffer.pack(~Tile.Tile8,bias)))),Buffer.content(~Tile.Tile8,bias),left_windows,right_windows>
    placed_joint(~kernel,~run,~agrees,~keeps,limits,work,images,grid_facts,grid_exact,width_positive,room,out_run,ranges,image_run,tasks,enabled,bias,zipped(left_windows,right_windows),left_windows,right_windows,
      paired(~Left,~left_value,~Right,~right_value,input,weights,ShapeLaws.elements(in_shape),ShapeLaws.elements(w_shape),work,ranges,left_windows,right_windows,left_ok,right_ok))}

type SizedPanel<-limits: Machine.Limits,-images: U32,-shape: List<&2,U32>,-width: U32> is Data:
  SizedPanel{exact: {ShapeLaws.elements(shape) == Nat.mul(U32.to_nat(images),U32.to_nat(width)) : Nat},
    room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool}}

def Cover(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+shape: List<&2,U32>,-pieces: Pieces.Pieces(~Tile.Tile8)) -> Data:
  Held.Covered<Tile.Tile8,Source,value,source,Move.held(~Tile.Tile8,pieces),0n,ShapeLaws.elements(shape),
    Held.layout_spans(Pieces.ranges_of_grid(Tensors.grid_of(shape)),Held.Ended{})>

# Read and write coverage of the actual panel schedule, at any worker count.
def panel_product_coverage(~kernel: Products.Kernel(),~run: Point.Run(),~agrees: Point.BlockAgreement(~kernel,~run),~keeps: Point.BiasStep(~run),
  ~Left: Data,~left_value: Left -> U32 -> Tile.Tile8,~Right: Data,~right_value: Right -> U32 -> Tile.Tile8,
  +input: Left,+weights: Right,+limits: Machine.Limits,+images: U32,+rows: U32,+inner: U32,+channels: U32,+workers: U32,+bias: Buffer.Model<Tile.Tile8>,
  +images_positive: {Nat.is_lt(0n,U32.to_nat(images)) == True{} : Bool},
  +images_room: {Nat.is_le(U32.to_nat(images),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +input_size: SizedPanel<limits,images,input_shape(images,rows,inner),left_width(work_of(images,rows,inner,channels))>,
  +weights_size: SizedPanel<limits,images,weights_shape(images,inner,channels),right_width(work_of(images,rows,inner,channels))>,
  +output_size: SizedPanel<limits,images,output_shape(images,rows,channels),image_width(work_of(images,rows,inner,channels))>,
  left: Pieces.Pieces(~Tile.Tile8),right: Pieces.Pieces(~Tile.Tile8),
  +left_cover: Cover(~Left,~left_value,input,input_shape(images,rows,inner),left),
  +right_cover: Cover(~Right,~right_value,weights,weights_shape(images,inner,channels),right)) ->
  PanelChecked<run,Left,left_value,Right,right_value,input,weights,images,rows,inner,channels,left,right,
    Products.panels(~kernel,images,rows,inner,channels,workers,left,right,Buffer.pack(~Tile.Tile8,bias)),Buffer.content(~Tile.Tile8,bias)>:
  SizedPanel{+input_exact,+input_room,+input_nonempty} = input_size
  SizedPanel{+weights_exact,+weights_room,+weights_nonempty} = weights_size
  SizedPanel{+output_exact,+output_room,+output_nonempty} = output_size
  +work = work_of(images,rows,inner,channels)
  +grid = panel_grid(work)
  +out_shape = output_shape(images,rows,channels)
  +out_cells = ShapeLaws.elements(out_shape)
  +width = U32.to_nat(image_width(work))
  +target = Grid.piece_target()
  +ranges = Tasks.component_ranges(images)
  +image_run = components(limits,images,images_positive,images_room)
  +image_plan = Held.first_of(Runs.RunOf(ranges,Pieces.Range{0,images},1n,U32.to_nat(images)),Singletons(ranges),image_run)
  +reads_run = Runs.fold_run_of(Pieces.Range{0,images},ranges,1n,U32.to_nat(images),image_plan)
  +last = GridRanges.last_extent(out_shape)
  +divides = {GridRanges.Multiple{last,Equal.sym(Nat,Nat.mul(last,1n),last,Order.times_one(last))} : GridRanges.Multiple<last,1n>}
  +counted = GridRanges.grid_count(limits,out_shape,target,output_room,output_nonempty)
  +out_layout = GridRanges.grid_layout(limits,out_shape,target,output_room,output_nonempty,1n,divides)
  +width_positive = DivisionOrder.right_factor_positive(U32.to_nat(images),width,
    %output_exact : {Nat.is_lt(0n,_) == True{} : Bool} output_nonempty)
  +out_run = {%Equal.sym(Nat,GridLaws.grid_cells(grid),out_cells,counted) : Runs.Run(Pieces.ranges_of_grid(grid),1n,_)
    GridRanges.layout_run(out_shape,target,1n,out_layout) : Runs.Run(Pieces.ranges_of_grid(grid),1n,GridLaws.grid_cells(grid))}
  +left_reads = panel_windows(~Left,~left_value,~Products.panel_left,~left_width,~left_scaled,input,work,limits,images,input_shape(images,rows,inner),
    input_exact,input_room,input_nonempty,reads_run,left,left_cover)
  +right_reads = panel_windows(~Right,~right_value,~Products.panel_right,~right_width,~right_scaled,weights,work,limits,images,weights_shape(images,inner,channels),
    weights_exact,weights_room,weights_nonempty,reads_run,right,right_cover)
  panel_tasks_output(~kernel,~run,~agrees,~keeps,~Left,~left_value,~Right,~right_value,input,weights,limits,images,rows,inner,channels,bias,
    GridRanges.grid_facts(limits,out_shape,target,output_room,output_nonempty),
    Equal.trans(Nat,GridLaws.grid_cells(grid),out_cells,Nat.mul(U32.to_nat(images),width),counted,output_exact),width_positive,
    %Equal.sym(Nat,GridLaws.grid_cells(grid),out_cells,counted) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} output_room,
    out_run,image_run,decision_of(images,workers,work),left,right,left_reads,right_reads)
