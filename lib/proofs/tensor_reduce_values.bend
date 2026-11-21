# The values of a reduction over the pieces of a tensor. Output k is finish of
# the fold, from the initial value, of the values of the reduced axes at the
# kept index k, in row-major order of the reduced axes.
import Base
import ../storage_buffer.bend as Storage
import ../tensor_operations.bend as Operations
import ../tensor_pieces.bend as Pieces
import ../tensor_line_chain.bend as LineChain
import ../traversal_partition.bend as Partition
import ../kernel_gemm.bend as KernelGemm
import ../tensor_grid.bend as Grid
import ../tensor_tasks.bend as Tasks
import ./tensor_pieces_regrouped.bend as Regrouped
import ../traversal_array.bend as Traversal
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./ordered_fold_refinement.bend as Fold
import ./traversal_array_refinement.bend as ArrayLaws
import ./tensor_operations_refinement.bend as OperationsLaws
import ./tensor_line_chain_values.bend as LineChainValues
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_order.bend as Order
import ./list_storage_readback.bend as ListReadback
import ./row_major_index.bend as RowMajor
import ./natural_addition.bend as Addition
import ./binary_radix.bend as Radix
import ./u32_division.bend as U32Division
import ./tensor_grid_pieces.bend as GridPieces
import ./tensor_task_runs.bend as Runs
import ./tensor_move_readback.bend as Held
import ./word_subtraction.bend as Subtraction
import ./word_encoding.bend as WordEncoding
import ./tensor_grid_ranges.bend as GridRanges
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_bit_length.bend as Bits
import ./u32_comparison.bend as Comparison
import ./encoded_coordinates.bend as Coordinates
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as LimitBounds
import ./word_multiplication.bend as Multiplication
import ./u32_extrema.bend as Extrema
import ../kernel_cost.bend as Cost
import ../default_kernel_cost.bend as DefaultCost
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_pieces_each.bend as Each
import ./tensor_pieces_content.bend as Content
import ./tensor_pieces_plan.bend as PiecesPlan
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_tiled_public.bend as TiledPublic
import ./tensor_count_completion.bend as CountCompletion
import ../tensor_layout.bend as Layout
import ../tensor.bend as Tensors
import ./tensor_global_shape_count.bend as Global
import ./tensor_contract.bend as Contract
import ./tensor_count_refinement.bend as Count
import ./u32_full_range.bend as FullRange
import ./tensor_list_values.bend as Listed
import ./tensor_storage_count.bend as StorageCount
import ../tensor_view.bend as Views
import ./tensor_view_gather.bend as ViewGather
import ./tensor_view_part.bend as ViewPart
import ./tensor_grid_box.bend as Box
import ./nat_division.bend as Division
import ./nat_division_order.bend as NatDivision
import ./boolean_logic.bend as Boolean
import ./tensor_view_moved.bend as ViewMoved
import ./tensor_view_bounds.bend as ViewBounds
import ./tensor_contiguous_bounds.bend as ContiguousBounds
import ../default_machine_limits.bend as Default

# ---- A fold over a range of a buffer is the fold of its model ----

# The value of a buffer model at a position.
def cell(~Element: Data,model: Buffer.Model<Element>,position: U32) -> Element:
  Tree.read(Element,Buffer.tree(~Element,model),position)

# The cells of a source read step apart from position, the cells of its model
# at the positions they read.
def strided_sampled(~Element: Data,+model: Buffer.Model<Element>,+position: U32,+step: U32,+index: U32) ->
  {ArrayLaws.sample_strided(~Element,Buffer.tree(~Element,model),Traversal.Strided{position,step},index) == cell(~Element,model,(position + index * step : U32)) : Element}:
  {==}

# The values of a source read step apart from position, from index on, are
# the cells of its model step apart from the position of index.
def values_strided(~Source: Data,~Model: Data,~Element: Data,~Accumulator: Data,~sample: Source -> Traversal.Strided -> U32 -> Element,~cell: Model -> U32 -> Element,
  ~source_of: Model -> Source,
  ~sampled: @+model: Model -> @+position: U32 -> @+step: U32 -> @+index: U32 -> {sample(source_of(model),Traversal.Strided{position,step},index) == cell(model,(position + index * step : U32)) : Element},
  ~operation: Accumulator -> Element -> Accumulator,count: Nat,+index: U32,+position: U32,+step: U32,+start: U32,+initial: Accumulator,+model: Model,
  +start_value: {(position + index * step : U32) == start : U32}) ->
  {Fold.values(~Source,~Element,~Accumulator,~Traversal.Strided,~sample,~operation,count,index,source_of(model),Traversal.Strided{position,step},initial) ==
    LineChainValues.strided(~Model,~Element,~Accumulator,~cell,~operation,count,start,step,initial,model) : Accumulator}:
  match count:
    case 0n: {==}
    case 1n+rest:
      %start_value :
        {Fold.values(~Source,~Element,~Accumulator,~Traversal.Strided,~sample,~operation,1n+rest,index,source_of(model),Traversal.Strided{position,step},initial) ==
          LineChainValues.strided(~Model,~Element,~Accumulator,~cell,~operation,1n+rest,_,step,initial,model) : Accumulator}
      %Equal.sym(Element,sample(source_of(model),Traversal.Strided{position,step},index),cell(model,(position + index * step : U32)),sampled(model,position,step,index)) :
        {Fold.values(~Source,~Element,~Accumulator,~Traversal.Strided,~sample,~operation,rest,(index + 1 : U32),source_of(model),Traversal.Strided{position,step},operation(initial,_)) ==
          LineChainValues.strided(~Model,~Element,~Accumulator,~cell,~operation,rest,((position + index * step : U32) + step : U32),step,
            operation(initial,cell(model,(position + index * step : U32))),model) : Accumulator}
      values_strided(~Source,~Model,~Element,~Accumulator,~sample,~cell,~source_of,~sampled,~operation,rest,(index + 1 : U32),position,step,((position + index * step : U32) + step : U32),
        operation(initial,cell(model,(position + index * step : U32))),model,
        Equal.trans(U32,(position + (index + 1 : U32) * step : U32),(position + (index * step + step : U32) : U32),((position + index * step : U32) + step : U32),
          Equal.cong(U32,U32,word => (position + word : U32),((index + 1 : U32) * step : U32),(index * step + step : U32),Arithmetic.multiply_successor(index,step)),
          Equal.sym(U32,((position + index * step : U32) + step : U32),(position + (index * step + step : U32) : U32),Arithmetic.add_associative(position,(index * step : U32),step))))

# The first of the cells step apart from position is at position.
def start_zero(+position: U32,+step: U32) -> {(position + 0 * step : U32) == position : U32}:
  Equal.trans(U32,(position + 0 * step : U32),(position + 0 : U32),position,
    Equal.cong(U32,U32,word => (position + word : U32),(0 * step : U32),0,Arithmetic.multiply_zero_left(step)),Arithmetic.add_zero(position))

# A fold over a range of a buffer model keeps the buffer and folds its cells.
def range_same(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,+count: Nat,+position: U32,+step: U32,+initial: Accumulator,
  +model: Buffer.Model<Element>) ->
  {Operations.fold_range(~Element,~Accumulator,~operation,count,position,step,initial,Buffer.pack(~Element,model)) ==
    (Buffer.pack(~Element,model),LineChainValues.strided_model(~Buffer.Model<Element>,~Element,~Accumulator,~cell(~Element),~operation,count,position,step,initial,model)) :
    Storage.Buffer<Element> & Accumulator}:
  +values = Fold.values(~Tree.Tree<Element>,~Element,~Accumulator,~Traversal.Strided,~ArrayLaws.sample_strided(~Element),~operation,count,0,Buffer.tree(~Element,model),
    Traversal.Strided{position,step},initial)
  Equal.trans(Storage.Buffer<Element> & Accumulator,
    Operations.fold_range(~Element,~Accumulator,~operation,count,position,step,initial,Buffer.pack(~Element,model)),
    (Buffer.pack(~Element,model),values),
    (Buffer.pack(~Element,model),LineChainValues.strided_model(~Buffer.Model<Element>,~Element,~Accumulator,~cell(~Element),~operation,count,position,step,initial,model)),
    OperationsLaws.fold_range_model(~Element,~Accumulator,~operation,count,position,step,initial,model),
    Equal.cong(Accumulator,Storage.Buffer<Element> & Accumulator,value => (Buffer.pack(~Element,model),value),values,
      LineChainValues.strided_model(~Buffer.Model<Element>,~Element,~Accumulator,~cell(~Element),~operation,count,position,step,initial,model),
      values_strided(~Tree.Tree<Element>,~Buffer.Model<Element>,~Element,~Accumulator,~ArrayLaws.sample_strided(~Element),~cell(~Element),~Buffer.tree(~Element),~strided_sampled(~Element),
        ~operation,count,0,position,step,position,initial,model,start_zero(position,step))))

# The buffer a vacant piece holds, as a model.
def vacant_model(~Element: Data,fill: Element) -> Buffer.Model<Element>:
  Buffer.remodel(~Element,Pieces.vacant_buffer(~Element,fill))

def vacant_same(~Element: Data,+fill: Element) -> {Pieces.vacant_buffer(~Element,fill) == Buffer.pack(~Element,vacant_model(~Element,fill)) : Storage.Buffer<Element>}:
  Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,vacant_model(~Element,fill)),Pieces.vacant_buffer(~Element,fill),
    Buffer.remodeled(~Element,Pieces.vacant_buffer(~Element,fill)))

# ---- Chunks of lists ----

# The folds of groups chunks of size cells of a list, each from initial.
def chunk_folds(~Cell: Data,~State: Data,~step: State -> Cell -> State,groups: Nat,+size: Nat,+list: List<&2,Cell>,+initial: State) -> List<&2,State>:
  match groups:
    case 0n: []
    case 1n+rest:
      LineChainValues.taken_list(~Cell,~State,~step,List.take(&2,Cell,list,size),0n,1n,initial) <>
        chunk_folds(~Cell,~State,~step,rest,size,List.drop(&2,Cell,list,size),initial)

def take_append(~Element: Data,left: List<&2,Element>,count: Nat,+right: List<&2,Element>,
  +inside: {Nat.is_le(count,List.length(&2,Element,left)) == True{} : Bool}) ->
  {List.take(&2,Element,List.append(&2,Element,left,right),count) == List.take(&2,Element,left,count) : List<&2,Element>}:
  match left count:
    case [] 0n:
      match right:
        case []: {==}
        case +head <> +tail: {==}
    case [] 1n+previous: Empty.absurd({List.take(&2,Element,right,1n+previous) == [] : List<&2,Element>},WordEncoding.false_true(inside))
    case +head <> +tail 0n: {==}
    case +head <> +tail 1n+previous:
      Equal.cong(List<&2,Element>,List<&2,Element>,others => head <> others,List.take(&2,Element,List.append(&2,Element,tail,right),previous),List.take(&2,Element,tail,previous),
        take_append(~Element,tail,previous,right,inside))

def drop_append(~Element: Data,left: List<&2,Element>,count: Nat,+right: List<&2,Element>,
  +inside: {Nat.is_le(count,List.length(&2,Element,left)) == True{} : Bool}) ->
  {List.drop(&2,Element,List.append(&2,Element,left,right),count) == List.append(&2,Element,List.drop(&2,Element,left,count),right) : List<&2,Element>}:
  match left count:
    case [] 0n: LineChainValues.drop_zero(~Element,right)
    case [] 1n+previous:
      Empty.absurd({List.drop(&2,Element,right,1n+previous) == right : List<&2,Element>},WordEncoding.false_true(inside))
    case +head <> +tail 0n: {==}
    case +head <> +tail 1n+previous: drop_append(~Element,tail,previous,right,inside)

# ---- Task layouts over regrouped trees ----

# A task layout over a tree, as tasks_model cuts it: every task streams as
# many cells as its range counts.
def Streams(~Model: Data,~Cell: Data,~cell: Model -> U32 -> Cell,layout: Pieces.Ranges,tree: Partition.Tree<&2,Model,Unit>) -> Data:
  match layout:
    case Pieces.Range{+first,+count}: {List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,tree)) == U32.to_nat(count) : Nat}
    case Pieces.Split{+left,+right}:
      match tree:
        case Partition.Leaf{+model,+first,+count,+details}: Empty
        case Partition.Fork{+low,+high}: KernelGemm.Both<Streams(~Model,~Cell,~cell,left,low),Streams(~Model,~Cell,~cell,right,high)>

# A group folded from ranges over a tree: its layout streams and its count
# is the cells of the tree.
def Folded(~Model: Data,~Cell: Data,~cell: Model -> U32 -> Cell,group: Tasks.Group,tree: Partition.Tree<&2,Model,Unit>) -> Data:
  match group:
    case Tasks.Group{+first,+count,+layout}: KernelGemm.Both<Streams(~Model,~Cell,~cell,layout,tree),{List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,tree)) == U32.to_nat(count) : Nat}>

def picked_streams(~Model: Data,~Cell: Data,~cell: Model -> U32 -> Cell,choice: Bool,+first: U32,+count: U32,+left: Pieces.Ranges,+right: Pieces.Ranges,+low: Partition.Tree<&2,Model,Unit>,
  +high: Partition.Tree<&2,Model,Unit>,+total: {List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,Partition.Fork{low,high})) == U32.to_nat(count) : Nat},+left_streams: Streams(~Model,~Cell,~cell,left,low),
  +right_streams: Streams(~Model,~Cell,~cell,right,high)) ->
  Streams(~Model,~Cell,~cell,Bool.pick(Pieces.Ranges,choice,Pieces.Range{first,count},Pieces.Split{left,right}),Partition.Fork{low,high}):
  match choice:
    case True{}: total
    case False{}: KernelGemm.Both{left_streams,right_streams}

# Two folded groups over the halves of a tree join over the tree.
def joined_folded(~Model: Data,~Cell: Data,~cell: Model -> U32 -> Cell,+limit: U32,+left_ranges: Pieces.Ranges,+right_ranges: Pieces.Ranges,left: Tasks.Group,right: Tasks.Group,
  +low: Partition.Tree<&2,Model,Unit>,+high: Partition.Tree<&2,Model,Unit>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,Partition.Fork{low,high})),U32.to_nat(cap)) == True{} : Bool},+left_folded: Folded(~Model,~Cell,~cell,left,low),+right_folded: Folded(~Model,~Cell,~cell,right,high)) ->
  Folded(~Model,~Cell,~cell,Tasks.group_join(limit,left_ranges,right_ranges,left,right),Partition.Fork{low,high}):
  match left right:
    case Tasks.Group{+left_first,+left_count,+left_layout} Tasks.Group{+right_first,+right_count,+right_layout}:
      KernelGemm.Both{+left_streams,+left_length} = left_folded
      KernelGemm.Both{+right_streams,+right_length} = right_folded
      +length_sum = ListReadback.append_length(~Cell,LineChainValues.stream(~Model,~Cell,~cell,low),LineChainValues.stream(~Model,~Cell,~cell,high))
      +bounded = {%length_sum : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
        room : {Nat.is_le(Nat.add(List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,low)),List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,high))),U32.to_nat(cap)) == True{} : Bool}}
      +count_exact = Arithmetic.sum_exact(cap,cap_ok,left_count,right_count,List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,low)),List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,high)),
        Equal.sym(Nat,List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,low)),U32.to_nat(left_count),left_length),Equal.sym(Nat,List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,high)),U32.to_nat(right_count),right_length),bounded)
      +total = Equal.trans(Nat,List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,Partition.Fork{low,high})),Nat.add(List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,low)),List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,high))),U32.to_nat((left_count + right_count : U32)),length_sum,
        Equal.sym(Nat,U32.to_nat((left_count + right_count : U32)),Nat.add(List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,low)),List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,high))),count_exact))
      KernelGemm.Both{picked_streams(~Model,~Cell,~cell,U32.is_le((left_count + right_count : U32),limit),left_first,(left_count + right_count : U32),left_layout,right_layout,
        low,high,total,left_streams,right_streams),total}

# The groups folded from ranges stream over the tree built in their shape.
def folded_streams(~Model: Data,~Cell: Data,~cell: Model -> U32 -> Cell,~Fill: Data,~vacant: Fill -> Model,layout: Pieces.Ranges,+limit: U32,+fill: Fill,+queue: List<&2,Model>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,Regrouped.built(~Model,~Fill,~vacant,layout,fill,queue))),U32.to_nat(cap)) == True{} : Bool}) ->
  Folded(~Model,~Cell,~cell,Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,layout,limit),Regrouped.built(~Model,~Fill,~vacant,layout,fill,queue)):
  match layout:
    case Pieces.Range{+first,+count}:
      +same = LineChainValues.cells_length(~Model,~Cell,~cell,U32.to_nat(count),0,Regrouped.head(~Model,~Fill,~vacant,fill,queue))
      KernelGemm.Both{same,same}
    case Pieces.Split{+left,+right}:
      +later = Regrouped.taken(~Model,left,queue)
      +low = Regrouped.built(~Model,~Fill,~vacant,left,fill,queue)
      +high = Regrouped.built(~Model,~Fill,~vacant,right,fill,later)
      +length_sum = ListReadback.append_length(~Cell,LineChainValues.stream(~Model,~Cell,~cell,low),LineChainValues.stream(~Model,~Cell,~cell,high))
      +sum_room = {%length_sum : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
        room : {Nat.is_le(Nat.add(List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,low)),List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,high))),U32.to_nat(cap)) == True{} : Bool}}
      joined_folded(~Model,~Cell,~cell,limit,left,right,Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,left,limit),Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,right,limit),low,high,cap,cap_ok,room,
        folded_streams(~Model,~Cell,~cell,~Fill,~vacant,left,limit,fill,queue,cap,cap_ok,
          Order.transitive(List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,low)),Nat.add(List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,low)),List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,high))),U32.to_nat(cap),Order.le_add(List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,low)),List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,high))),sum_room)),
        folded_streams(~Model,~Cell,~cell,~Fill,~vacant,right,limit,fill,later,cap,cap_ok,
          Order.transitive(List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,high)),Nat.add(List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,low)),List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,high))),U32.to_nat(cap),Order.right_le(List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,low)),List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,high))),sum_room)))

# ---- Groups folded inside one outer index ----

def self_difference(+value: U32) -> {(value - value : U32) == 0 : U32}:
  %Arithmetic.add_zero_left(value) : {(_ - value : U32) == 0 : U32}
  Subtraction.subtract_add(0,value)

def difference_back(+value: U32,+part: U32) -> {(value - (value - part : U32) : U32) == part : U32}:
  Equal.trans(U32,(value - (value - part : U32) : U32),((value - value : U32) + part : U32),part,
    Equal.sym(U32,((value - value : U32) + part : U32),(value - (value - part : U32) : U32),Arithmetic.subtract_difference(value,value,part)),
    Equal.trans(U32,((value - value : U32) + part : U32),(0 + part : U32),part,
      Equal.cong(U32,U32,word => (word + part : U32),(value - value : U32),0,self_difference(value)),Arithmetic.add_zero_left(part)))

# Differences of words chain.
def telescoped(+low: U32,+middle: U32,+high: U32) -> {((middle - low : U32) + (high - middle : U32) : U32) == (high - low : U32) : U32}:
  Equal.trans(U32,((middle - low : U32) + (high - middle : U32) : U32),((high - middle : U32) + (middle - low : U32) : U32),(high - low : U32),
    Arithmetic.add_commutative((middle - low : U32),(high - middle : U32)),
    Equal.trans(U32,((high - middle : U32) + (middle - low : U32) : U32),(high - (middle - (middle - low : U32) : U32) : U32),(high - low : U32),
      Arithmetic.subtract_difference(high,middle,(middle - low : U32)),
      Equal.cong(U32,U32,word => (high - word : U32),(middle - (middle - low : U32) : U32),low,difference_back(middle,low))))

def split_end(+first: U32,+left: U32,+count: U32) -> {((first + left : U32) + (count - left : U32) : U32) == (first + count : U32) : U32}:
  Equal.trans(U32,((first + left : U32) + (count - left : U32) : U32),(first + (left + (count - left : U32) : U32) : U32),(first + count : U32),
    Arithmetic.add_associative(first,left,(count - left : U32)),
    Equal.cong(U32,U32,word => (first + word : U32),(left + (count - left : U32) : U32),count,
      Equal.trans(U32,(left + (count - left : U32) : U32),((count - left : U32) + left : U32),count,Arithmetic.add_commutative(left,(count - left : U32)),
        Subtraction.add_subtract(count,left))))

# The cells of a split run of pieces are the cells of its halves.
def cells_split(+grid: Grid.Grid,+first: U32,+left: U32,+count: U32) ->
  {(Pieces.cells_of(grid,first,left) + Pieces.cells_of(grid,(first + left : U32),(count - left : U32)) : U32) == Pieces.cells_of(grid,first,count) : U32}:
  Grid.Grid{+cells,+prefix,+outer,+extent,+unit_cells,+piece_units,+per_outer,+pieces} = grid
  %Equal.sym(U32,((first + left : U32) + (count - left : U32) : U32),(first + count : U32),split_end(first,left,count)) :
    {((U32.min(Grid.start(Grid.Grid{cells,prefix,outer,extent,unit_cells,piece_units,per_outer,pieces},(first + left : U32)),cells) - U32.min(Grid.start(Grid.Grid{cells,prefix,outer,extent,unit_cells,piece_units,per_outer,pieces},first),cells) : U32) + (U32.min(Grid.start(Grid.Grid{cells,prefix,outer,extent,unit_cells,piece_units,per_outer,pieces},_),cells) - U32.min(Grid.start(Grid.Grid{cells,prefix,outer,extent,unit_cells,piece_units,per_outer,pieces},(first + left : U32)),cells) : U32) : U32) == (U32.min(Grid.start(Grid.Grid{cells,prefix,outer,extent,unit_cells,piece_units,per_outer,pieces},(first + count : U32)),cells) - U32.min(Grid.start(Grid.Grid{cells,prefix,outer,extent,unit_cells,piece_units,per_outer,pieces},first),cells) : U32) : U32}
  telescoped(U32.min(Grid.start(Grid.Grid{cells,prefix,outer,extent,unit_cells,piece_units,per_outer,pieces},first),cells),U32.min(Grid.start(Grid.Grid{cells,prefix,outer,extent,unit_cells,piece_units,per_outer,pieces},(first + left : U32)),cells),U32.min(Grid.start(Grid.Grid{cells,prefix,outer,extent,unit_cells,piece_units,per_outer,pieces},(first + count : U32)),cells))

def first_of(group: Tasks.Group) -> U32:
  Tasks.Group{first,count,layout} = group
  first

def count_of(group: Tasks.Group) -> U32:
  Tasks.Group{first,count,layout} = group
  count

def joined_first(+limit: U32,+left_ranges: Pieces.Ranges,+right_ranges: Pieces.Ranges,left: Tasks.Group,right: Tasks.Group) ->
  {first_of(Tasks.group_join(limit,left_ranges,right_ranges,left,right)) == first_of(left) : U32}:
  match left right:
    case Tasks.Group{+left_first,+left_count,+left_layout} Tasks.Group{+right_first,+right_count,+right_layout}: {==}

def joined_count(+limit: U32,+left_ranges: Pieces.Ranges,+right_ranges: Pieces.Ranges,left: Tasks.Group,right: Tasks.Group) ->
  {count_of(Tasks.group_join(limit,left_ranges,right_ranges,left,right)) == (count_of(left) + count_of(right) : U32) : U32}:
  match left right:
    case Tasks.Group{+left_first,+left_count,+left_layout} Tasks.Group{+right_first,+right_count,+right_layout}: {==}

def joined_layout(+limit: U32,+left_ranges: Pieces.Ranges,+right_ranges: Pieces.Ranges,left: Tasks.Group,right: Tasks.Group) ->
  {Tasks.group_layout(Tasks.group_join(limit,left_ranges,right_ranges,left,right)) ==
    Bool.pick(Pieces.Ranges,U32.is_le((count_of(left) + count_of(right) : U32),limit),Pieces.Range{first_of(left),(count_of(left) + count_of(right) : U32)},
      Pieces.Split{Tasks.group_layout(left),Tasks.group_layout(right)}) : Pieces.Ranges}:
  match left right:
    case Tasks.Group{+left_first,+left_count,+left_layout} Tasks.Group{+right_first,+right_count,+right_layout}: {==}

# The pieces of one outer index fold into a group from their first cell
# that counts their cells.
def inner_first(+grid: Grid.Grid,+limit: U32,depth: Nat,single: Bool,+first: U32,+count: U32) ->
  {first_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(depth,single,grid,first,count),limit)) == Grid.start(grid,first) : U32}:
  match depth single:
    case 0n single: {==}
    case 1n+rest True{}: {==}
    case 1n+ +rest False{}:
      Equal.trans(U32,first_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(1n+rest,False{},grid,first,count),limit)),first_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le(Pieces.below(count),1),grid,first,Pieces.below(count)),limit)),Grid.start(grid,first),
        joined_first(limit,Pieces.inner_ranges(rest,U32.is_le(Pieces.below(count),1),grid,first,Pieces.below(count)),Pieces.inner_ranges(rest,U32.is_le((count - Pieces.below(count) : U32),1),grid,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le(Pieces.below(count),1),grid,first,Pieces.below(count)),limit),Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le((count - Pieces.below(count) : U32),1),grid,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),limit)),
        inner_first(grid,limit,rest,U32.is_le(Pieces.below(count),1),first,Pieces.below(count)))

def inner_count(+grid: Grid.Grid,+limit: U32,depth: Nat,single: Bool,+first: U32,+count: U32) ->
  {count_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(depth,single,grid,first,count),limit)) == Pieces.cells_of(grid,first,count) : U32}:
  match depth single:
    case 0n single: {==}
    case 1n+rest True{}: {==}
    case 1n+ +rest False{}:
      +left = Pieces.below(count)
      +left_cells = Pieces.cells_of(grid,first,left)
      +right_cells = Pieces.cells_of(grid,(first + left : U32),(count - left : U32))
      Equal.trans(U32,count_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(1n+rest,False{},grid,first,count),limit)),(count_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le(Pieces.below(count),1),grid,first,Pieces.below(count)),limit)) + count_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le((count - Pieces.below(count) : U32),1),grid,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),limit)) : U32),Pieces.cells_of(grid,first,count),
        joined_count(limit,Pieces.inner_ranges(rest,U32.is_le(Pieces.below(count),1),grid,first,Pieces.below(count)),Pieces.inner_ranges(rest,U32.is_le((count - Pieces.below(count) : U32),1),grid,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le(Pieces.below(count),1),grid,first,Pieces.below(count)),limit),Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le((count - Pieces.below(count) : U32),1),grid,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),limit)),
        Equal.trans(U32,(count_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le(Pieces.below(count),1),grid,first,Pieces.below(count)),limit)) + count_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le((count - Pieces.below(count) : U32),1),grid,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),limit)) : U32),(left_cells + count_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le((count - Pieces.below(count) : U32),1),grid,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),limit)) : U32),Pieces.cells_of(grid,first,count),
          Equal.cong(U32,U32,word => (word + count_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le((count - Pieces.below(count) : U32),1),grid,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),limit)) : U32),count_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le(Pieces.below(count),1),grid,first,Pieces.below(count)),limit)),left_cells,
            inner_count(grid,limit,rest,U32.is_le(left,1),first,left)),
          Equal.trans(U32,(left_cells + count_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le((count - Pieces.below(count) : U32),1),grid,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),limit)) : U32),(left_cells + right_cells : U32),Pieces.cells_of(grid,first,count),
            Equal.cong(U32,U32,word => (left_cells + word : U32),count_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le((count - Pieces.below(count) : U32),1),grid,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),limit)),right_cells,
              inner_count(grid,limit,rest,U32.is_le((count - left : U32),1),(first + left : U32),(count - left : U32))),
            cells_split(grid,first,left,count))))

# When its cells fit the limit, the pieces of one outer index fold into one range.
def inner_layout(+grid: Grid.Grid,+limit: U32,depth: Nat,single: Bool,+first: U32,+count: U32,
  +fits: {U32.is_le(Pieces.cells_of(grid,first,count),limit) == True{} : Bool}) ->
  {Tasks.group_layout(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(depth,single,grid,first,count),limit)) == Pieces.Range{Grid.start(grid,first),Pieces.cells_of(grid,first,count)} : Pieces.Ranges}:
  match depth single:
    case 0n single: {==}
    case 1n+rest True{}: {==}
    case 1n+ +rest False{}:
      +sum_exact = Equal.trans(U32,(count_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le(Pieces.below(count),1),grid,first,Pieces.below(count)),limit)) + count_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le((count - Pieces.below(count) : U32),1),grid,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),limit)) : U32),count_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(1n+rest,False{},grid,first,count),limit)),Pieces.cells_of(grid,first,count),
        Equal.sym(U32,count_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(1n+rest,False{},grid,first,count),limit)),(count_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le(Pieces.below(count),1),grid,first,Pieces.below(count)),limit)) + count_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le((count - Pieces.below(count) : U32),1),grid,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),limit)) : U32),joined_count(limit,Pieces.inner_ranges(rest,U32.is_le(Pieces.below(count),1),grid,first,Pieces.below(count)),Pieces.inner_ranges(rest,U32.is_le((count - Pieces.below(count) : U32),1),grid,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le(Pieces.below(count),1),grid,first,Pieces.below(count)),limit),Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le((count - Pieces.below(count) : U32),1),grid,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),limit))),
        inner_count(grid,limit,1n+rest,False{},first,count))
      +first_exact = inner_first(grid,limit,rest,U32.is_le(Pieces.below(count),1),first,Pieces.below(count))
      %Equal.sym(Pieces.Ranges,Tasks.group_layout(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(1n+rest,False{},grid,first,count),limit)),
          Bool.pick(Pieces.Ranges,U32.is_le((count_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le(Pieces.below(count),1),grid,first,Pieces.below(count)),limit)) + count_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le((count - Pieces.below(count) : U32),1),grid,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),limit)) : U32),limit),Pieces.Range{first_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le(Pieces.below(count),1),grid,first,Pieces.below(count)),limit)),(count_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le(Pieces.below(count),1),grid,first,Pieces.below(count)),limit)) + count_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le((count - Pieces.below(count) : U32),1),grid,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),limit)) : U32)},Pieces.Split{Tasks.group_layout(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le(Pieces.below(count),1),grid,first,Pieces.below(count)),limit)),Tasks.group_layout(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le((count - Pieces.below(count) : U32),1),grid,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),limit))}),
          joined_layout(limit,Pieces.inner_ranges(rest,U32.is_le(Pieces.below(count),1),grid,first,Pieces.below(count)),Pieces.inner_ranges(rest,U32.is_le((count - Pieces.below(count) : U32),1),grid,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le(Pieces.below(count),1),grid,first,Pieces.below(count)),limit),Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le((count - Pieces.below(count) : U32),1),grid,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),limit))) :
        {_ == Pieces.Range{Grid.start(grid,first),Pieces.cells_of(grid,first,count)} : Pieces.Ranges}
      %Equal.sym(U32,(count_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le(Pieces.below(count),1),grid,first,Pieces.below(count)),limit)) + count_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le((count - Pieces.below(count) : U32),1),grid,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),limit)) : U32),Pieces.cells_of(grid,first,count),sum_exact) :
        {Bool.pick(Pieces.Ranges,U32.is_le(_,limit),Pieces.Range{first_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le(Pieces.below(count),1),grid,first,Pieces.below(count)),limit)),_},Pieces.Split{Tasks.group_layout(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le(Pieces.below(count),1),grid,first,Pieces.below(count)),limit)),Tasks.group_layout(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le((count - Pieces.below(count) : U32),1),grid,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),limit))}) ==
          Pieces.Range{Grid.start(grid,first),Pieces.cells_of(grid,first,count)} : Pieces.Ranges}
      %Equal.sym(U32,first_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le(Pieces.below(count),1),grid,first,Pieces.below(count)),limit)),Grid.start(grid,first),first_exact) :
        {Bool.pick(Pieces.Ranges,U32.is_le(Pieces.cells_of(grid,first,count),limit),Pieces.Range{_,Pieces.cells_of(grid,first,count)},
            Pieces.Split{Tasks.group_layout(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le(Pieces.below(count),1),grid,first,Pieces.below(count)),limit)),Tasks.group_layout(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le((count - Pieces.below(count) : U32),1),grid,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),limit))}) ==
          Pieces.Range{Grid.start(grid,first),Pieces.cells_of(grid,first,count)} : Pieces.Ranges}
      %Equal.sym(Bool,U32.is_le(Pieces.cells_of(grid,first,count),limit),True{},fits) :
        {Bool.pick(Pieces.Ranges,_,Pieces.Range{Grid.start(grid,first),Pieces.cells_of(grid,first,count)},
            Pieces.Split{Tasks.group_layout(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le(Pieces.below(count),1),grid,first,Pieces.below(count)),limit)),Tasks.group_layout(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(rest,U32.is_le((count - Pieces.below(count) : U32),1),grid,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),limit))}) ==
          Pieces.Range{Grid.start(grid,first),Pieces.cells_of(grid,first,count)} : Pieces.Ranges}
      {==}

# ---- Groups of whole outer indices ----

# A grid whose groups of size outer indices are pieces step apart, each unit
# cells, groups of them inside the grid.
type Alignment<-grid: Grid.Grid,-step: U32,-unit: Nat,-groups: Nat> is Data:
  Alignment{limits: Machine.Limits,count: U32,prefix: U32,outer: U32,extent: U32,unit_cells: U32,block: U32,per: U32,size: U32,
    shape: GridPieces.Shape<limits,outer,extent,unit_cells,block,per>,grid_exact: {grid == Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid},
    cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
    step_exact: {step == (U32.max(per,1) * size : U32) : U32},
    unit_exact: {unit == Nat.mul(U32.to_nat(size),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))) : Nat},
    groups_inside: {Nat.is_le(Nat.mul(groups,U32.to_nat(size)),U32.to_nat(outer)) == True{} : Bool}}

# The piece q * per starts at q outer indices.
def multiple_start(+limits: Machine.Limits,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: GridPieces.Shape<limits,outer,extent,unit_cells,block,per>,
  +piece: U32,+q: Nat,+piece_exact: {U32.to_nat(piece) == Nat.mul(q,U32.to_nat(per)) : Nat},+room: {Nat.is_le(q,U32.to_nat(outer)) == True{} : Bool}) ->
  {U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)) == Nat.mul(q,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))) : Nat}:
  +bound = {%Equal.sym(Nat,Nat.add(Nat.mul(q,U32.to_nat(extent)),0n),Nat.mul(q,U32.to_nat(extent)),Addition.zero_right(Nat.mul(q,U32.to_nat(extent)))) :
      {Nat.is_le(Nat.mul(_,U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    Order.transitive(Nat.mul(Nat.mul(q,U32.to_nat(extent)),U32.to_nat(unit_cells)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits)),
      RowMajor.mul_le(Nat.mul(q,U32.to_nat(extent)),Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells),RowMajor.mul_le(q,U32.to_nat(outer),U32.to_nat(extent),room)),
      GridPieces.shape_room(limits,outer,extent,unit_cells,block,per,shape)) :
    {Nat.is_le(Nat.mul(Nat.add(Nat.mul(q,U32.to_nat(extent)),0n),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +placed = GridPieces.start_place(limits,count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32),piece,q,0n,GridPieces.per_positive(limits,outer,extent,unit_cells,block,per,shape),
    GridPieces.per_room(limits,outer,extent,unit_cells,block,per,shape),GridPieces.shape_unit(limits,outer,extent,unit_cells,block,per,shape),
    Equal.trans(Nat,U32.to_nat(piece),Nat.mul(q,U32.to_nat(per)),Nat.add(Nat.mul(q,U32.to_nat(per)),0n),piece_exact,Equal.sym(Nat,Nat.add(Nat.mul(q,U32.to_nat(per)),0n),Nat.mul(q,U32.to_nat(per)),Addition.zero_right(Nat.mul(q,U32.to_nat(per))))),
    GridPieces.per_positive(limits,outer,extent,unit_cells,block,per,shape),bound)
  Equal.trans(Nat,U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},piece)),Nat.add(Nat.mul(q,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),0n),Nat.mul(q,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),placed,Addition.zero_right(Nat.mul(q,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)))))

# The group index of size outer indices starts at index groups of cells and
# holds one group of cells.
def aligned_core(+limits: Machine.Limits,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+size: U32,+shape: GridPieces.Shape<limits,outer,extent,unit_cells,block,per>,
  +cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+groups: Nat,+groups_inside: {Nat.is_le(Nat.mul(groups,U32.to_nat(size)),U32.to_nat(outer)) == True{} : Bool},+index: U32,
  +below: {Nat.is_lt(U32.to_nat(index),groups) == True{} : Bool}) ->
  Held.Both<{U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(index * (per * size : U32) : U32))) == Nat.mul(U32.to_nat(index),Nat.mul(U32.to_nat(size),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)))) : Nat},{U32.to_nat(Pieces.cells_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(index * (per * size : U32) : U32),(per * size : U32))) == Nat.mul(U32.to_nat(size),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))) : Nat}>:
  +cap = Machine.capacity(limits)
  +cap_ok = LimitBounds.capacity_fits(limits)
  +q1 = Nat.mul(U32.to_nat(index),U32.to_nat(size))
  +q2 = Nat.add(q1,U32.to_nat(size))
  +q2_room = Held.nat_le_moved(q2,Nat.mul(1n+U32.to_nat(index),U32.to_nat(size)),U32.to_nat(outer),GridPieces.next_outer(U32.to_nat(index),U32.to_nat(size)),
    Order.transitive(Nat.mul(1n+U32.to_nat(index),U32.to_nat(size)),Nat.mul(groups,U32.to_nat(size)),U32.to_nat(outer),RowMajor.mul_le(1n+U32.to_nat(index),groups,U32.to_nat(size),Order.strict_follow(U32.to_nat(index),groups,below)),groups_inside))
  +q1_room = Order.transitive(q1,q2,U32.to_nat(outer),Order.le_add(q1,U32.to_nat(size)),q2_room)
  +pieces_room = GridPieces.pieces_room(limits,outer,extent,unit_cells,block,per,shape)
  +high_inside = RowMajor.mul_le(q2,U32.to_nat(outer),U32.to_nat(per),q2_room)
  +low_inside = RowMajor.mul_le(q1,U32.to_nat(outer),U32.to_nat(per),q1_room)
  +step_word = Multiplication.multiply_exact(per,size,U32.to_nat(per),U32.to_nat(size),cap,cap_ok,{==},{==},
    Held.nat_le_moved(Nat.mul(U32.to_nat(per),U32.to_nat(size)),Nat.mul(U32.to_nat(size),U32.to_nat(per)),U32.to_nat(cap),RowMajor.multiply_commutative(U32.to_nat(per),U32.to_nat(size)),
      Order.transitive(Nat.mul(U32.to_nat(size),U32.to_nat(per)),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat(cap),RowMajor.mul_le(U32.to_nat(size),U32.to_nat(outer),U32.to_nat(per),Order.transitive(U32.to_nat(size),q2,U32.to_nat(outer),Order.right_le(q1,U32.to_nat(size)),q2_room)),pieces_room)))
  +swapped = Equal.trans(Nat,Nat.mul(U32.to_nat(index),Nat.mul(U32.to_nat(per),U32.to_nat(size))),Nat.mul(U32.to_nat(index),Nat.mul(U32.to_nat(size),U32.to_nat(per))),Nat.mul(q1,U32.to_nat(per)),
    Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(index),value),Nat.mul(U32.to_nat(per),U32.to_nat(size)),Nat.mul(U32.to_nat(size),U32.to_nat(per)),RowMajor.multiply_commutative(U32.to_nat(per),U32.to_nat(size))),
    Equal.sym(Nat,Nat.mul(q1,U32.to_nat(per)),Nat.mul(U32.to_nat(index),Nat.mul(U32.to_nat(size),U32.to_nat(per))),RowMajor.assoc(U32.to_nat(index),U32.to_nat(size),U32.to_nat(per))))
  +piece_word = Equal.trans(Nat,U32.to_nat((index * (per * size : U32) : U32)),Nat.mul(U32.to_nat(index),Nat.mul(U32.to_nat(per),U32.to_nat(size))),Nat.mul(q1,U32.to_nat(per)),
    Multiplication.multiply_exact(index,(per * size : U32),U32.to_nat(index),Nat.mul(U32.to_nat(per),U32.to_nat(size)),cap,cap_ok,{==},step_word,
      Held.nat_le_moved(Nat.mul(U32.to_nat(index),Nat.mul(U32.to_nat(per),U32.to_nat(size))),Nat.mul(q1,U32.to_nat(per)),U32.to_nat(cap),swapped,Order.transitive(Nat.mul(q1,U32.to_nat(per)),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat(cap),low_inside,pieces_room))),
    swapped)
  +step_turned = Equal.trans(Nat,U32.to_nat((per * size : U32)),Nat.mul(U32.to_nat(per),U32.to_nat(size)),Nat.mul(U32.to_nat(size),U32.to_nat(per)),step_word,RowMajor.multiply_commutative(U32.to_nat(per),U32.to_nat(size)))
  +sum_exact = Equal.trans(Nat,Nat.add(U32.to_nat((index * (per * size : U32) : U32)),U32.to_nat((per * size : U32))),Nat.add(Nat.mul(q1,U32.to_nat(per)),Nat.mul(U32.to_nat(size),U32.to_nat(per))),Nat.mul(q2,U32.to_nat(per)),
    Equal.trans(Nat,Nat.add(U32.to_nat((index * (per * size : U32) : U32)),U32.to_nat((per * size : U32))),Nat.add(Nat.mul(q1,U32.to_nat(per)),U32.to_nat((per * size : U32))),Nat.add(Nat.mul(q1,U32.to_nat(per)),Nat.mul(U32.to_nat(size),U32.to_nat(per))),
      Equal.cong(Nat,Nat,value => Nat.add(value,U32.to_nat((per * size : U32))),U32.to_nat((index * (per * size : U32) : U32)),Nat.mul(q1,U32.to_nat(per)),piece_word),
      Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(q1,U32.to_nat(per)),value),U32.to_nat((per * size : U32)),Nat.mul(U32.to_nat(size),U32.to_nat(per)),step_turned)),
    Equal.sym(Nat,Nat.mul(q2,U32.to_nat(per)),Nat.add(Nat.mul(q1,U32.to_nat(per)),Nat.mul(U32.to_nat(size),U32.to_nat(per))),RowMajor.distrib(q1,U32.to_nat(size),U32.to_nat(per))))
  +next_word = Equal.trans(Nat,U32.to_nat(((index * (per * size : U32) : U32) + (per * size : U32) : U32)),Nat.add(U32.to_nat((index * (per * size : U32) : U32)),U32.to_nat((per * size : U32))),Nat.mul(q2,U32.to_nat(per)),
    Arithmetic.add((index * (per * size : U32) : U32),(per * size : U32),cap,cap_ok,Held.nat_le_moved(Nat.add(U32.to_nat((index * (per * size : U32) : U32)),U32.to_nat((per * size : U32))),Nat.mul(q2,U32.to_nat(per)),U32.to_nat(cap),sum_exact,
      Order.transitive(Nat.mul(q2,U32.to_nat(per)),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat(cap),high_inside,pieces_room))),
    sum_exact)
  +low_start = multiple_start(limits,count,prefix,outer,extent,unit_cells,block,per,shape,(index * (per * size : U32) : U32),q1,piece_word,q1_room)
  +high_start = multiple_start(limits,count,prefix,outer,extent,unit_cells,block,per,shape,((index * (per * size : U32) : U32) + (per * size : U32) : U32),q2,next_word,q2_room)
  +low_kept = Equal.trans(Nat,U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(index * (per * size : U32) : U32)),count)),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(index * (per * size : U32) : U32))),Nat.mul(q1,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),
    GridRanges.minimum_start(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,(index * (per * size : U32) : U32),
      Held.nat_le_moved(U32.to_nat((index * (per * size : U32) : U32)),Nat.mul(q1,U32.to_nat(per)),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),piece_word,low_inside)),low_start)
  +high_kept = Equal.trans(Nat,U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},((index * (per * size : U32) : U32) + (per * size : U32) : U32)),count)),U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},((index * (per * size : U32) : U32) + (per * size : U32) : U32))),Nat.mul(q2,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),
    GridRanges.minimum_start(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,((index * (per * size : U32) : U32) + (per * size : U32) : U32),
      Held.nat_le_moved(U32.to_nat(((index * (per * size : U32) : U32) + (per * size : U32) : U32)),Nat.mul(q2,U32.to_nat(per)),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),next_word,high_inside)),high_start)
  +ordered = Held.nat_le_moved(U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(index * (per * size : U32) : U32)),count)),Nat.mul(q1,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},((index * (per * size : U32) : U32) + (per * size : U32) : U32)),count)),low_kept,
    Held.nat_le_right_moved(Nat.mul(q1,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},((index * (per * size : U32) : U32) + (per * size : U32) : U32)),count)),Nat.mul(q2,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),high_kept,RowMajor.mul_le(q1,q2,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Order.le_add(q1,U32.to_nat(size)))))
  +whole_room = Held.nat_le_moved(Nat.mul(U32.to_nat(outer),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(cap),
    Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(outer),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),RowMajor.assoc(U32.to_nat(outer),U32.to_nat(extent),U32.to_nat(unit_cells))),
    GridPieces.shape_room(limits,outer,extent,unit_cells,block,per,shape))
  +high_room = Held.nat_le_moved(U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},((index * (per * size : U32) : U32) + (per * size : U32) : U32)),count)),Nat.mul(q2,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(cap),high_kept,Order.transitive(Nat.mul(q2,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(outer),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(cap),RowMajor.mul_le(q2,U32.to_nat(outer),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),q2_room),whole_room))
  +difference = Arithmetic.subtract(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},((index * (per * size : U32) : U32) + (per * size : U32) : U32)),count),U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(index * (per * size : U32) : U32)),count),cap,cap_ok,ordered,Order.transitive(Nat.sub(U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},((index * (per * size : U32) : U32) + (per * size : U32) : U32)),count)),U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(index * (per * size : U32) : U32)),count))),U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},((index * (per * size : U32) : U32) + (per * size : U32) : U32)),count)),U32.to_nat(cap),Order.sub_le(U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},((index * (per * size : U32) : U32) + (per * size : U32) : U32)),count)),U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(index * (per * size : U32) : U32)),count))),high_room))
  +cells_exact = Equal.trans(Nat,U32.to_nat(Pieces.cells_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(index * (per * size : U32) : U32),(per * size : U32))),Nat.sub(U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},((index * (per * size : U32) : U32) + (per * size : U32) : U32)),count)),U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(index * (per * size : U32) : U32)),count))),Nat.mul(U32.to_nat(size),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),difference,
    Equal.trans(Nat,Nat.sub(U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},((index * (per * size : U32) : U32) + (per * size : U32) : U32)),count)),U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(index * (per * size : U32) : U32)),count))),Nat.sub(Nat.mul(q2,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(index * (per * size : U32) : U32)),count))),Nat.mul(U32.to_nat(size),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),
      Equal.cong(Nat,Nat,value => Nat.sub(value,U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(index * (per * size : U32) : U32)),count))),U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},((index * (per * size : U32) : U32) + (per * size : U32) : U32)),count)),Nat.mul(q2,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),high_kept),
      Equal.trans(Nat,Nat.sub(Nat.mul(q2,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(index * (per * size : U32) : U32)),count))),Nat.sub(Nat.mul(q2,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(q1,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)))),Nat.mul(U32.to_nat(size),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),
        Equal.cong(Nat,Nat,value => Nat.sub(Nat.mul(q2,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),value),U32.to_nat(U32.min(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(index * (per * size : U32) : U32)),count)),Nat.mul(q1,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),low_kept),
        Equal.trans(Nat,Nat.sub(Nat.mul(q2,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(q1,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)))),Nat.sub(Nat.add(Nat.mul(q1,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(size),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)))),Nat.mul(q1,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)))),Nat.mul(U32.to_nat(size),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),
          Equal.cong(Nat,Nat,value => Nat.sub(value,Nat.mul(q1,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)))),Nat.mul(q2,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.add(Nat.mul(q1,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(size),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)))),RowMajor.distrib(q1,U32.to_nat(size),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)))),
          Order.subtract_shift(Nat.mul(q1,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(size),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))))))))
  Held.Both{Equal.trans(Nat,U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(index * (per * size : U32) : U32))),Nat.mul(q1,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(index),Nat.mul(U32.to_nat(size),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)))),low_start,RowMajor.assoc(U32.to_nat(index),U32.to_nat(size),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)))),cells_exact}

# Every group of an alignment starts at its units and holds one unit.
def aligned_at(+grid: Grid.Grid,+step: U32,+unit: Nat,+groups: Nat,+alignment: Alignment<grid,step,unit,groups>,+index: U32,
  +below: {Nat.is_lt(U32.to_nat(index),groups) == True{} : Bool}) -> Held.Both<{U32.to_nat(Grid.start(grid,(index * step : U32))) == Nat.mul(U32.to_nat(index),unit) : Nat},{U32.to_nat(Pieces.cells_of(grid,(index * step : U32),step)) == unit : Nat}>:
  Alignment{+limits,+count,+prefix,+outer,+extent,+unit_cells,+block,+per_outer,+size,+shape,+grid_exact,+cells,+step_exact,+unit_exact,+groups_inside} = alignment
  %Equal.sym(Grid.Grid,grid,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per_outer,(outer * per_outer : U32)},grid_exact) : Held.Both<{U32.to_nat(Grid.start(_,(index * step : U32))) == Nat.mul(U32.to_nat(index),unit) : Nat},{U32.to_nat(Pieces.cells_of(_,(index * step : U32),step)) == unit : Nat}>
  %Equal.sym(U32,step,(U32.max(per_outer,1) * size : U32),step_exact) : Held.Both<{U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per_outer,(outer * per_outer : U32)},(index * _ : U32))) == Nat.mul(U32.to_nat(index),unit) : Nat},{U32.to_nat(Pieces.cells_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per_outer,(outer * per_outer : U32)},(index * _ : U32),_)) == unit : Nat}>
  %Equal.sym(Nat,unit,Nat.mul(U32.to_nat(size),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),unit_exact) : Held.Both<{U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per_outer,(outer * per_outer : U32)},(index * (U32.max(per_outer,1) * size : U32) : U32))) == Nat.mul(U32.to_nat(index),_) : Nat},{U32.to_nat(Pieces.cells_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per_outer,(outer * per_outer : U32)},(index * (U32.max(per_outer,1) * size : U32) : U32),(U32.max(per_outer,1) * size : U32))) == _ : Nat}>
  %Equal.sym(U32,U32.max(per_outer,1),per_outer,GridPieces.maximum_kept(per_outer,1,Order.positive_one(U32.to_nat(per_outer),
      GridPieces.per_positive(limits,outer,extent,unit_cells,block,per_outer,shape)))) : Held.Both<{U32.to_nat(Grid.start(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per_outer,(outer * per_outer : U32)},(index * (_ * size : U32) : U32))) == Nat.mul(U32.to_nat(index),Nat.mul(U32.to_nat(size),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)))) : Nat},{U32.to_nat(Pieces.cells_of(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per_outer,(outer * per_outer : U32)},(index * (_ * size : U32) : U32),(_ * size : U32))) == Nat.mul(U32.to_nat(size),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))) : Nat}>
  aligned_core(limits,count,prefix,outer,extent,unit_cells,block,per_outer,size,shape,cells,groups,groups_inside,index,below)

# A group whose first cell, cells and layout are one range is that range.
def group_run_from(+group: Tasks.Group,+first: U32,+count: U32,+low: Nat,+high: Nat,+unit: Nat,+bound: Nat,+first_is: {first_of(group) == first : U32},
  +count_is: {count_of(group) == count : U32},+layout_is: {Tasks.group_layout(group) == Pieces.Range{first,count} : Pieces.Ranges},
  +facts: Runs.GroupRun(Tasks.Group{first,count,Pieces.Range{first,count}},low,high,unit,bound)) -> Runs.GroupRun(group,low,high,unit,bound):
  match group:
    case Tasks.Group{+group_first,+group_count,+layout}:
      %Equal.sym(U32,group_first,first,first_is) : Runs.GroupRun(Tasks.Group{_,group_count,layout},low,high,unit,bound)
      %Equal.sym(U32,group_count,count,count_is) : Runs.GroupRun(Tasks.Group{first,_,layout},low,high,unit,bound)
      %Equal.sym(Pieces.Ranges,layout,Pieces.Range{first,count},layout_is) : Runs.GroupRun(Tasks.Group{first,count,_},low,high,unit,bound)
      facts

# One range of one unit after low_units of them.
def leaf_group(+first: U32,+count: U32,+low_units: Nat,+units: Nat,+unit: Nat,+bound: Nat,+low_exact: {U32.to_nat(first) == Nat.mul(low_units,unit) : Nat},
  +count_exact: {U32.to_nat(count) == unit : Nat},+units_one: {units == 1n : Nat},+unit_positive: {Nat.is_lt(0n,unit) == True{} : Bool},
  +inside: {Nat.is_le(Nat.mul(Nat.add(low_units,units),unit),bound) == True{} : Bool}) ->
  Runs.GroupRun(Tasks.Group{first,count,Pieces.Range{first,count}},Nat.mul(low_units,unit),Nat.mul(Nat.add(low_units,units),unit),unit,bound):
  +high = Nat.add(U32.to_nat(first),U32.to_nat(count))
  +high_exact = Equal.trans(Nat,high,Nat.add(Nat.mul(low_units,unit),unit),Nat.mul(Nat.add(low_units,units),unit),
    Equal.trans(Nat,high,Nat.add(Nat.mul(low_units,unit),U32.to_nat(count)),Nat.add(Nat.mul(low_units,unit),unit),
      Equal.cong(Nat,Nat,value => Nat.add(value,U32.to_nat(count)),U32.to_nat(first),Nat.mul(low_units,unit),low_exact),
      Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(low_units,unit),value),U32.to_nat(count),unit,count_exact)),
    Equal.trans(Nat,Nat.add(Nat.mul(low_units,unit),unit),Nat.add(Nat.mul(low_units,unit),Nat.mul(1n,unit)),Nat.mul(Nat.add(low_units,units),unit),
      Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(low_units,unit),value),unit,Nat.mul(1n,unit),Equal.sym(Nat,Nat.mul(1n,unit),unit,LineChainValues.one_times(unit))),
      Equal.trans(Nat,Nat.add(Nat.mul(low_units,unit),Nat.mul(1n,unit)),Nat.mul(Nat.add(low_units,1n),unit),Nat.mul(Nat.add(low_units,units),unit),
        Equal.sym(Nat,Nat.mul(Nat.add(low_units,1n),unit),Nat.add(Nat.mul(low_units,unit),Nat.mul(1n,unit)),RowMajor.distrib(low_units,1n,unit)),
        Equal.cong(Nat,Nat,value => Nat.mul(Nat.add(low_units,value),unit),1n,units,Equal.sym(Nat,units,1n,units_one)))))
  +count_positive = {%Equal.sym(Nat,U32.to_nat(count),unit,count_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    unit_positive : {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}}
  +less = {%Addition.zero_right(U32.to_nat(first)) : {Nat.is_lt(_,high) == True{} : Bool}
    Order.offset_lt(U32.to_nat(first),0n,U32.to_nat(count),count_positive) : {Nat.is_lt(U32.to_nat(first),high) == True{} : Bool}}
  +within = Held.nat_le_moved(high,Nat.mul(Nat.add(low_units,units),unit),bound,high_exact,inside)
  Held.Both{low_exact,Held.Both{high_exact,Held.Both{Runs.Fits{low_units,Nat.add(low_units,units),low_exact,high_exact,less,within},Held.Both{low_exact,high_exact}}}}

def aligned_leaf(+grid: Grid.Grid,+limit: U32,+per: U32,+unit: Nat,+bound: Nat,+groups: Nat,+unit_positive: {Nat.is_lt(0n,unit) == True{} : Bool},
  +fits: {Nat.is_le(unit,U32.to_nat(limit)) == True{} : Bool},+bound_inside: {Nat.is_le(Nat.mul(groups,unit),bound) == True{} : Bool},+first: U32,+count: U32,
  +count_one: {U32.to_nat(count) == 1n : Nat},+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),groups) == True{} : Bool},
  +facts: Held.Both<{U32.to_nat(Grid.start(grid,(first * per : U32))) == Nat.mul(U32.to_nat(first),unit) : Nat},{U32.to_nat(Pieces.cells_of(grid,(first * per : U32),per)) == unit : Nat}>) ->
  Runs.GroupRun(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(32n,U32.is_le(per,1),grid,(first * per : U32),per),limit),Nat.mul(U32.to_nat(first),unit),Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(count)),unit),unit,bound):
  Held.Both{+start_exact,+cells_exact} = facts
  +piece = (first * per : U32)
  +cells = Pieces.cells_of(grid,piece,per)
  +limit_fits = Equal.trans(Bool,U32.is_le(cells,limit),Nat.is_le(U32.to_nat(cells),U32.to_nat(limit)),True{},Comparison.le(cells,limit),
    Held.nat_le_moved(U32.to_nat(cells),unit,U32.to_nat(limit),cells_exact,fits))
  +inner_bound = Order.transitive(Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(count)),unit),Nat.mul(groups,unit),bound,RowMajor.mul_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),groups,unit,room),bound_inside)
  group_run_from(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(32n,U32.is_le(per,1),grid,(first * per : U32),per),limit),Grid.start(grid,piece),cells,Nat.mul(U32.to_nat(first),unit),Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(count)),unit),unit,bound,
    inner_first(grid,limit,32n,U32.is_le(per,1),piece,per),inner_count(grid,limit,32n,U32.is_le(per,1),piece,per),
    inner_layout(grid,limit,32n,U32.is_le(per,1),piece,per,limit_fits),
    leaf_group(Grid.start(grid,piece),cells,U32.to_nat(first),U32.to_nat(count),unit,bound,start_exact,cells_exact,count_one,unit_positive,inner_bound))

# The pieces of one outer index whose cells are one unit fold into one range
# of that unit.
def outer_leaf_run(+grid: Grid.Grid,+limit: U32,+per: U32,+unit: Nat,+bound: Nat,+groups: Nat,
  +alignment: Alignment<grid,per,unit,groups>,
  +unit_positive: {Nat.is_lt(0n,unit) == True{} : Bool},+fits: {Nat.is_le(unit,U32.to_nat(limit)) == True{} : Bool},
  +bound_inside: {Nat.is_le(Nat.mul(groups,unit),bound) == True{} : Bool},
  +first: U32,+count: U32,+test: {U32.is_le(count,1) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),groups) == True{} : Bool}) ->
  Runs.GroupRun(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.inner_ranges(32n,U32.is_le(per,1),grid,(first * per : U32),per),limit),Nat.mul(U32.to_nat(first),unit),Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(count)),unit),unit,bound):
  +count_one = Order.antisymmetric(U32.to_nat(count),1n,Bounds.as_nat_le(count,1,test),Order.positive_one(U32.to_nat(count),positive))
  +index_room = Held.nat_le_moved(Nat.add(U32.to_nat(first),1n),Nat.add(U32.to_nat(first),U32.to_nat(count)),groups,
    Equal.cong(Nat,Nat,value => Nat.add(U32.to_nat(first),value),1n,U32.to_nat(count),Equal.sym(Nat,U32.to_nat(count),1n,count_one)),room)
  aligned_leaf(grid,limit,per,unit,bound,groups,unit_positive,fits,bound_inside,first,count,count_one,room,
    aligned_at(grid,per,unit,groups,alignment,first,Order.room_index(U32.to_nat(first),0n,groups,index_room)))

# The groups of outer indices first .. first + count, each one unit of whole
# lines, fold into a run of units.
def outer_group_run(+limits: Machine.Limits,+cap: U32,+cap_ok: {U32.is_le(cap,1073741824) == True{} : Bool},
  +grid: Grid.Grid,+limit: U32,+per: U32,+unit: Nat,+bound: Nat,+groups: Nat,
  +alignment: Alignment<grid,per,unit,groups>,
  +unit_positive: {Nat.is_lt(0n,unit) == True{} : Bool},+fits: {Nat.is_le(unit,U32.to_nat(limit)) == True{} : Bool},
  +bound_inside: {Nat.is_le(Nat.mul(groups,unit),bound) == True{} : Bool},
  +groups_room: {Nat.is_le(groups,U32.to_nat(cap)) == True{} : Bool},+bound_room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  depth: Nat,single: Bool,+first: U32,+count: U32,+test: {U32.is_le(count,1) == single : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
  +bounded: {Nat.is_le(U32.to_nat(count),Bits.power(depth)) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),groups) == True{} : Bool}) ->
  Runs.GroupRun(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.outer_ranges(1n+depth,single,grid,per,first,count),limit),Nat.mul(U32.to_nat(first),unit),Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(count)),unit),unit,bound):
  match depth single:
    case 0n True{}: outer_leaf_run(grid,limit,per,unit,bound,groups,alignment,unit_positive,fits,bound_inside,first,count,test,positive,room)
    case 0n False{}:
      Empty.absurd(Runs.GroupRun(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.outer_ranges(1n,False{},grid,per,first,count),limit),Nat.mul(U32.to_nat(first),unit),Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(count)),unit),unit,bound),Held.lt_le_absurd(1n,U32.to_nat(count),GridRanges.more_than_one(count,test),bounded))
    case 1n+ +more True{}: outer_leaf_run(grid,limit,per,unit,bound,groups,alignment,unit_positive,fits,bound_inside,first,count,test,positive,room)
    case 1n+ +more False{}:
      +inside = Order.transitive(U32.to_nat(count),Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap),Order.right_le(U32.to_nat(first),U32.to_nat(count)),Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(count)),groups,U32.to_nat(cap),room,groups_room))
      +halves = GridRanges.halves_of(cap,cap_ok,more,count,test,positive,bounded,inside)
      +left_positive = GridRanges.halves_left_positive(count,more,halves)
      +left_bound = GridRanges.halves_left_bound(count,more,halves)
      +left_count = GridRanges.halves_left_count(count,more,halves)
      +right_exact = GridRanges.halves_right_exact(count,more,halves)
      +right_positive = GridRanges.halves_right_positive(count,more,halves)
      +right_bound = GridRanges.halves_right_bound(count,more,halves)
      +left_room = Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),Nat.add(U32.to_nat(first),U32.to_nat(count)),groups,Order.offset_le(U32.to_nat(first),U32.to_nat(Pieces.below(count)),U32.to_nat(count),left_count),room)
      +first_exact = Arithmetic.add(first,Pieces.below(count),cap,Coordinates.native_le_transitive(cap,1073741824,2147483648,cap_ok,{==}),
        Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),groups,U32.to_nat(cap),left_room,groups_room))
      +right_sum = GridRanges.split_sum(first,count,first_exact,right_exact,left_count)
      +right_room = Held.nat_le_moved(Nat.add(U32.to_nat((first + Pieces.below(count) : U32)),U32.to_nat((count - Pieces.below(count) : U32))),Nat.add(U32.to_nat(first),U32.to_nat(count)),groups,right_sum,room)
      +left_run = outer_group_run(limits,cap,cap_ok,grid,limit,per,unit,bound,groups,alignment,unit_positive,fits,bound_inside,groups_room,bound_room,more,U32.is_le(Pieces.below(count),1),first,Pieces.below(count),{==},left_positive,left_bound,left_room)
      +right_run = outer_group_run(limits,cap,cap_ok,grid,limit,per,unit,bound,groups,alignment,unit_positive,fits,bound_inside,groups_room,bound_room,more,U32.is_le((count - Pieces.below(count) : U32),1),(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32),{==},right_positive,right_bound,right_room)
      +right_moved = {%Equal.cong(Nat,Nat,value => Nat.mul(value,unit),U32.to_nat((first + Pieces.below(count) : U32)),Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),first_exact) :
          Runs.GroupRun(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.outer_ranges(1n+more,U32.is_le((count - Pieces.below(count) : U32),1),grid,per,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),limit),_,Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(count)),unit),unit,bound)
        {%Equal.cong(Nat,Nat,value => Nat.mul(value,unit),Nat.add(U32.to_nat((first + Pieces.below(count) : U32)),U32.to_nat((count - Pieces.below(count) : U32))),Nat.add(U32.to_nat(first),U32.to_nat(count)),right_sum) : Runs.GroupRun(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.outer_ranges(1n+more,U32.is_le((count - Pieces.below(count) : U32),1),grid,per,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),limit),Nat.mul(U32.to_nat((first + Pieces.below(count) : U32)),unit),_,unit,bound)
          right_run : Runs.GroupRun(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.outer_ranges(1n+more,U32.is_le((count - Pieces.below(count) : U32),1),grid,per,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),limit),Nat.mul(U32.to_nat((first + Pieces.below(count) : U32)),unit),Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(count)),unit),unit,bound)} :
        Runs.GroupRun(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.outer_ranges(1n+more,U32.is_le((count - Pieces.below(count) : U32),1),grid,per,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),limit),Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),unit),Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(count)),unit),unit,bound)}
      Runs.joined(limits,limit,Pieces.outer_ranges(1n+more,U32.is_le(Pieces.below(count),1),grid,per,first,Pieces.below(count)),Pieces.outer_ranges(1n+more,U32.is_le((count - Pieces.below(count) : U32),1),grid,per,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.outer_ranges(1n+more,U32.is_le(Pieces.below(count),1),grid,per,first,Pieces.below(count)),limit),Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.outer_ranges(1n+more,U32.is_le((count - Pieces.below(count) : U32),1),grid,per,(first + Pieces.below(count) : U32),(count - Pieces.below(count) : U32)),limit),Nat.mul(U32.to_nat(first),unit),Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(Pieces.below(count))),unit),Nat.mul(Nat.add(U32.to_nat(first),U32.to_nat(count)),unit),unit,bound,
        left_run,right_moved,bound_room)

# A word that is times multiples of a divisor divides into times.
def exact_quotient(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+value: U32,+divisor: U32,+times: Nat,
  +exact: {U32.to_nat(value) == Nat.mul(times,U32.to_nat(divisor)) : Nat},+positive: {Nat.is_lt(0n,U32.to_nat(divisor)) == True{} : Bool},
  +divisor_room: {Nat.is_le(Nat.double(U32.to_nat(divisor)),U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat((value / divisor : U32)) == times : Nat}:
  +product = Nat.mul(times,U32.to_nat(divisor))
  Equal.trans(Nat,U32.to_nat((value / divisor : U32)),Nat.div(product,U32.to_nat(divisor)),times,
    U32Division.quotient_exact(cap,cap_ok,value,divisor,product,U32.to_nat(divisor),exact,{==},divisor_room,positive),
    Equal.trans(Nat,Nat.div(product,U32.to_nat(divisor)),Nat.div(Nat.add(product,0n),U32.to_nat(divisor)),times,
      Equal.cong(Nat,Nat,number => Nat.div(number,U32.to_nat(divisor)),product,Nat.add(product,0n),Equal.sym(Nat,Nat.add(product,0n),product,Addition.zero_right(product))),
      Radix.quotient(times,0n,U32.to_nat(divisor),positive,positive)))

# ---- Lines of whole groups of outer indices ----

def outer_room(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: GridPieces.Shape<limits,outer,extent,unit_cells,block,per>) -> {Nat.is_le(U32.to_nat(outer),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Order.transitive(U32.to_nat(outer),Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(Machine.capacity(limits)),
    RowMajor.product_covers(U32.to_nat(outer),U32.to_nat(extent),GridPieces.extent_positive(limits,outer,extent,unit_cells,block,per,shape)),
    Order.transitive(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits)),
      RowMajor.product_covers(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells),GridPieces.shape_unit(limits,outer,extent,unit_cells,block,per,shape)),GridPieces.shape_room(limits,outer,extent,unit_cells,block,per,shape)))

def size_positive(+group: U32) -> {Nat.is_lt(0n,U32.to_nat(U32.max(group,1))) == True{} : Bool}:
  Bounds.predecessor_strict(0n,U32.to_nat(U32.max(group,1)),Bounds.as_nat_le(1,U32.max(group,1),Extrema.maximum_right(group,1)))

def size_room(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: GridPieces.Shape<limits,outer,extent,unit_cells,block,per>,+group: U32,+lines: Nat,+outer_exact: {U32.to_nat(outer) == Nat.mul(lines,U32.to_nat(U32.max(group,1))) : Nat},+lines_positive: {Nat.is_lt(0n,lines) == True{} : Bool}) -> {Nat.is_le(U32.to_nat(U32.max(group,1)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Order.transitive(U32.to_nat(U32.max(group,1)),U32.to_nat(outer),U32.to_nat(Machine.capacity(limits)),
    Held.nat_le_right_moved(U32.to_nat(U32.max(group,1)),U32.to_nat(outer),Nat.mul(lines,U32.to_nat(U32.max(group,1))),outer_exact,
      Held.nat_le_right_moved(U32.to_nat(U32.max(group,1)),Nat.mul(lines,U32.to_nat(U32.max(group,1))),Nat.mul(U32.to_nat(U32.max(group,1)),lines),RowMajor.multiply_commutative(lines,U32.to_nat(U32.max(group,1))),RowMajor.product_covers(U32.to_nat(U32.max(group,1)),lines,lines_positive))),
    outer_room(limits,outer,extent,unit_cells,block,per,shape))

def lines_bound(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: GridPieces.Shape<limits,outer,extent,unit_cells,block,per>,+group: U32,+lines: Nat,+outer_exact: {U32.to_nat(outer) == Nat.mul(lines,U32.to_nat(U32.max(group,1))) : Nat},+lines_positive: {Nat.is_lt(0n,lines) == True{} : Bool}) -> {Nat.is_le(lines,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Order.transitive(lines,U32.to_nat(outer),U32.to_nat(Machine.capacity(limits)),
    Held.nat_le_right_moved(lines,U32.to_nat(outer),Nat.mul(lines,U32.to_nat(U32.max(group,1))),outer_exact,RowMajor.product_covers(lines,U32.to_nat(U32.max(group,1)),size_positive(group))),outer_room(limits,outer,extent,unit_cells,block,per,shape))

# The groups of size outer indices of a grid are its lines.
def group_lines(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: GridPieces.Shape<limits,outer,extent,unit_cells,block,per>,+group: U32,+lines: Nat,+outer_exact: {U32.to_nat(outer) == Nat.mul(lines,U32.to_nat(U32.max(group,1))) : Nat},+lines_positive: {Nat.is_lt(0n,lines) == True{} : Bool}) -> {U32.to_nat(U32.max((outer / U32.max(group,1) : U32),1)) == lines : Nat}:
  Extrema.max_one_exact((outer / U32.max(group,1) : U32),lines,
    exact_quotient(Machine.half_word(limits),LimitBounds.half_word_fits(limits),outer,U32.max(group,1),lines,outer_exact,size_positive(group),
      LimitBounds.double_bound(limits,U32.to_nat(U32.max(group,1)),size_room(limits,outer,extent,unit_cells,block,per,shape,group,lines,outer_exact,lines_positive))),
    lines_positive)

# The pieces between groups are per of each outer index.
def group_step(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: GridPieces.Shape<limits,outer,extent,unit_cells,block,per>,+group: U32,+lines: Nat,+outer_exact: {U32.to_nat(outer) == Nat.mul(lines,U32.to_nat(U32.max(group,1))) : Nat},+lines_positive: {Nat.is_lt(0n,lines) == True{} : Bool}) -> {U32.to_nat((U32.max(per,1) * U32.max(group,1) : U32)) == Nat.mul(U32.to_nat(per),U32.to_nat(U32.max(group,1))) : Nat}:
  +kept = GridPieces.maximum_kept(per,1,Order.positive_one(U32.to_nat(per),GridPieces.per_positive(limits,outer,extent,unit_cells,block,per,shape)))
  %Equal.sym(U32,U32.max(per,1),per,kept) : {U32.to_nat((_ * U32.max(group,1) : U32)) == Nat.mul(U32.to_nat(per),U32.to_nat(U32.max(group,1))) : Nat}
  Multiplication.multiply_exact(per,U32.max(group,1),U32.to_nat(per),U32.to_nat(U32.max(group,1)),Machine.capacity(limits),LimitBounds.capacity_fits(limits),{==},{==},
    Held.nat_le_moved(Nat.mul(U32.to_nat(per),U32.to_nat(U32.max(group,1))),Nat.mul(U32.to_nat(U32.max(group,1)),U32.to_nat(per)),U32.to_nat(Machine.capacity(limits)),RowMajor.multiply_commutative(U32.to_nat(per),U32.to_nat(U32.max(group,1))),
      Order.transitive(Nat.mul(U32.to_nat(U32.max(group,1)),U32.to_nat(per)),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat(Machine.capacity(limits)),
        RowMajor.mul_le(U32.to_nat(U32.max(group,1)),U32.to_nat(outer),U32.to_nat(per),Held.nat_le_right_moved(U32.to_nat(U32.max(group,1)),U32.to_nat(outer),Nat.mul(lines,U32.to_nat(U32.max(group,1))),outer_exact,
          Held.nat_le_right_moved(U32.to_nat(U32.max(group,1)),Nat.mul(lines,U32.to_nat(U32.max(group,1))),Nat.mul(U32.to_nat(U32.max(group,1)),lines),RowMajor.multiply_commutative(lines,U32.to_nat(U32.max(group,1))),RowMajor.product_covers(U32.to_nat(U32.max(group,1)),lines,lines_positive)))),
        GridPieces.pieces_room(limits,outer,extent,unit_cells,block,per,shape))))

# Grouping whole outer indices lists the pieces of a grid in the same order.
def grouped_spans(+limits: Machine.Limits,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: GridPieces.Shape<limits,outer,extent,unit_cells,block,per>,+count: U32,+prefix: U32,+group: U32,+lines: Nat,+outer_exact: {U32.to_nat(outer) == Nat.mul(lines,U32.to_nat(U32.max(group,1))) : Nat},+lines_positive: {Nat.is_lt(0n,lines) == True{} : Bool}) ->
  {Held.layout_spans(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),Held.Ended{}) == Held.layout_spans(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),Held.Ended{}) : Held.Spans}:
  +step = group_step(limits,outer,extent,unit_cells,block,per,shape,group,lines,outer_exact,lines_positive)
  +groups = group_lines(limits,outer,extent,unit_cells,block,per,shape,group,lines,outer_exact,lines_positive)
  +swapped = Equal.trans(Nat,Nat.mul(lines,Nat.mul(U32.to_nat(per),U32.to_nat(U32.max(group,1)))),Nat.mul(lines,Nat.mul(U32.to_nat(U32.max(group,1)),U32.to_nat(per))),Nat.mul(Nat.mul(lines,U32.to_nat(U32.max(group,1))),U32.to_nat(per)),
    Equal.cong(Nat,Nat,value => Nat.mul(lines,value),Nat.mul(U32.to_nat(per),U32.to_nat(U32.max(group,1))),Nat.mul(U32.to_nat(U32.max(group,1)),U32.to_nat(per)),RowMajor.multiply_commutative(U32.to_nat(per),U32.to_nat(U32.max(group,1)))),
    Equal.sym(Nat,Nat.mul(Nat.mul(lines,U32.to_nat(U32.max(group,1))),U32.to_nat(per)),Nat.mul(lines,Nat.mul(U32.to_nat(U32.max(group,1)),U32.to_nat(per))),RowMajor.assoc(lines,U32.to_nat(U32.max(group,1)),U32.to_nat(per))))
  +total = Equal.trans(Nat,Nat.mul(U32.to_nat(U32.max((outer / U32.max(group,1) : U32),1)),U32.to_nat((U32.max(per,1) * U32.max(group,1) : U32))),Nat.mul(lines,Nat.mul(U32.to_nat(per),U32.to_nat(U32.max(group,1)))),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),
    Equal.trans(Nat,Nat.mul(U32.to_nat(U32.max((outer / U32.max(group,1) : U32),1)),U32.to_nat((U32.max(per,1) * U32.max(group,1) : U32))),Nat.mul(lines,U32.to_nat((U32.max(per,1) * U32.max(group,1) : U32))),Nat.mul(lines,Nat.mul(U32.to_nat(per),U32.to_nat(U32.max(group,1)))),
      Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat((U32.max(per,1) * U32.max(group,1) : U32))),U32.to_nat(U32.max((outer / U32.max(group,1) : U32),1)),lines,groups),
      Equal.cong(Nat,Nat,value => Nat.mul(lines,value),U32.to_nat((U32.max(per,1) * U32.max(group,1) : U32)),Nat.mul(U32.to_nat(per),U32.to_nat(U32.max(group,1))),step)),
    Equal.trans(Nat,Nat.mul(lines,Nat.mul(U32.to_nat(per),U32.to_nat(U32.max(group,1)))),Nat.mul(Nat.mul(lines,U32.to_nat(U32.max(group,1))),U32.to_nat(per)),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),swapped,
      Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(per)),Nat.mul(lines,U32.to_nat(U32.max(group,1))),U32.to_nat(outer),Equal.sym(Nat,U32.to_nat(outer),Nat.mul(lines,U32.to_nat(U32.max(group,1))),outer_exact))))
  +step_positive = {%Equal.sym(Nat,U32.to_nat((U32.max(per,1) * U32.max(group,1) : U32)),Nat.mul(U32.to_nat(per),U32.to_nat(U32.max(group,1))),step) : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.product_positive(U32.to_nat(per),U32.to_nat(U32.max(group,1)),GridPieces.per_positive(limits,outer,extent,unit_cells,block,per,shape),size_positive(group)) : {Nat.is_lt(0n,U32.to_nat((U32.max(per,1) * U32.max(group,1) : U32))) == True{} : Bool}}
  +lines_cap = lines_bound(limits,outer,extent,unit_cells,block,per,shape,group,lines,outer_exact,lines_positive)
  +room = Held.nat_le_moved(Nat.mul(U32.to_nat(U32.max((outer / U32.max(group,1) : U32),1)),U32.to_nat((U32.max(per,1) * U32.max(group,1) : U32))),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),U32.to_nat(Machine.capacity(limits)),total,GridPieces.pieces_room(limits,outer,extent,unit_cells,block,per,shape))
  Equal.trans(Held.Spans,Held.layout_spans(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),Held.Ended{}),GridRanges.piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Nat.mul(U32.to_nat(U32.max((outer / U32.max(group,1) : U32),1)),U32.to_nat((U32.max(per,1) * U32.max(group,1) : U32))),Nat.mul(U32.to_nat(0),U32.to_nat((U32.max(per,1) * U32.max(group,1) : U32))),Held.Ended{}),
    Held.layout_spans(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),Held.Ended{}),
    GridRanges.outer_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Machine.capacity(limits),LimitBounds.capacity_allocatable(limits),(U32.max(per,1) * U32.max(group,1) : U32),step_positive,31n,U32.is_le(U32.max((outer / U32.max(group,1) : U32),1),1),0,U32.max((outer / U32.max(group,1) : U32),1),Held.Ended{},{==},
      Held.nat_lt_moved(0n,U32.to_nat(U32.max((outer / U32.max(group,1) : U32),1)),lines,groups,lines_positive),
      GridRanges.capacity_power(limits,U32.to_nat(U32.max((outer / U32.max(group,1) : U32),1)),Held.nat_le_moved(U32.to_nat(U32.max((outer / U32.max(group,1) : U32),1)),lines,U32.to_nat(Machine.capacity(limits)),groups,lines_cap)),room),
    Equal.trans(Held.Spans,GridRanges.piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Nat.mul(U32.to_nat(U32.max((outer / U32.max(group,1) : U32),1)),U32.to_nat((U32.max(per,1) * U32.max(group,1) : U32))),0n,Held.Ended{}),GridRanges.piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Nat.mul(U32.to_nat(outer),U32.to_nat(per)),0n,Held.Ended{}),
      Held.layout_spans(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),Held.Ended{}),
      Equal.cong(Nat,Held.Spans,pieces => GridRanges.piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},pieces,0n,Held.Ended{}),Nat.mul(U32.to_nat(U32.max((outer / U32.max(group,1) : U32),1)),U32.to_nat((U32.max(per,1) * U32.max(group,1) : U32))),Nat.mul(U32.to_nat(outer),U32.to_nat(per)),total),
      Equal.sym(Held.Spans,Held.layout_spans(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),Held.Ended{}),GridRanges.piece_spans(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},Nat.mul(U32.to_nat(outer),U32.to_nat(per)),0n,Held.Ended{}),
        GridRanges.grid_spans(limits,outer,extent,unit_cells,block,per,count,prefix,shape))))

# The longest task the tasks of a layout may hold: an even share of the
# cells, a whole block, or an even share of the ranges at their largest.
def task_limit(+count: U32,+pool: U32,+whole: U32,found: Tasks.Size) -> U32:
  Tasks.Size{+ranges,+largest} = found
  U32.max(U32.max(Tasks.divided_up(count,pool),whole),(Tasks.divided_up(ranges,pool) * largest : U32))

def sized_tasks(+layout: Pieces.Ranges,+count: U32,+pool: U32,+whole: U32,+found: Tasks.Size) ->
  {Tasks.tasks_sized(layout,count,pool,whole,found) == Tasks.group_layout(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,layout,task_limit(count,pool,whole,found))) : Pieces.Ranges}:
  match found:
    case Tasks.Size{+ranges,+largest}: {==}

def whole_within(+count: U32,+pool: U32,+whole: U32,+found: Tasks.Size) -> {Nat.is_le(U32.to_nat(whole),U32.to_nat(task_limit(count,pool,whole,found))) == True{} : Bool}:
  match found:
    case Tasks.Size{+ranges,+largest}:
      +shared = U32.max(Tasks.divided_up(count,pool),whole)
      +spread = (Tasks.divided_up(ranges,pool) * largest : U32)
      Order.transitive(U32.to_nat(whole),U32.to_nat(shared),U32.to_nat(U32.max(shared,spread)),Bounds.as_nat_le(whole,shared,Extrema.maximum_right(Tasks.divided_up(count,pool),whole)),
        Bounds.as_nat_le(shared,U32.max(shared,spread),Extrema.maximum_left(shared,spread)))

# The groups of lines of a grid, each group whole outer indices of one line,
# fold into a run of whole lines.
def carried_group_run(+limits: Machine.Limits,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +shape: GridPieces.Shape<limits,outer,extent,unit_cells,block,per>,+cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},
  +group: U32,+workers: U32,+whole: U32,+bytes: F32,+operations: F32,+lines: Nat,+unit: Nat,+bound: Nat,
  +outer_exact: {U32.to_nat(outer) == Nat.mul(lines,U32.to_nat(U32.max(group,1))) : Nat},+lines_positive: {Nat.is_lt(0n,lines) == True{} : Bool},
  +unit_exact: {unit == Nat.mul(U32.to_nat(U32.max(group,1)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))) : Nat},+unit_positive: {Nat.is_lt(0n,unit) == True{} : Bool},
  +whole_fits: {Nat.is_le(unit,U32.to_nat(whole)) == True{} : Bool},+bound_inside: {Nat.is_le(Nat.mul(lines,unit),bound) == True{} : Bool},
  +bound_room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Runs.GroupRun(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),task_limit(count,U32.max(workers,1),whole,Tasks.size(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group)))),Nat.mul(U32.to_nat(0),unit),Nat.mul(Nat.add(U32.to_nat(0),U32.to_nat(U32.max((outer / U32.max(group,1) : U32),1))),unit),unit,bound):
  +groups_word = group_lines(limits,outer,extent,unit_cells,block,per,shape,group,lines,outer_exact,lines_positive)
  +lines_room = lines_bound(limits,outer,extent,unit_cells,block,per,shape,group,lines,outer_exact,lines_positive)
  +groups_inside = Held.nat_le_right_moved(Nat.mul(lines,U32.to_nat(U32.max(group,1))),U32.to_nat(outer),Nat.mul(lines,U32.to_nat(U32.max(group,1))),outer_exact,Order.reflexive(Nat.mul(lines,U32.to_nat(U32.max(group,1)))))
  +alignment = {Alignment{limits,count,prefix,outer,extent,unit_cells,block,per,U32.max(group,1),shape,{==},cells,{==},unit_exact,groups_inside} : Alignment<Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(U32.max(per,1) * U32.max(group,1) : U32),unit,lines>}
  +fits = Order.transitive(unit,U32.to_nat(whole),U32.to_nat(task_limit(count,U32.max(workers,1),whole,Tasks.size(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group)))),whole_fits,
    whole_within(count,U32.max(workers,1),whole,Tasks.size(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group))))
  +positive = Held.nat_lt_moved(0n,U32.to_nat(U32.max((outer / U32.max(group,1) : U32),1)),lines,groups_word,lines_positive)
  +bounded = GridRanges.capacity_power(limits,U32.to_nat(U32.max((outer / U32.max(group,1) : U32),1)),Held.nat_le_moved(U32.to_nat(U32.max((outer / U32.max(group,1) : U32),1)),lines,U32.to_nat(Machine.capacity(limits)),groups_word,lines_room))
  +room = Held.nat_le_moved(U32.to_nat(U32.max((outer / U32.max(group,1) : U32),1)),lines,lines,groups_word,Order.reflexive(lines))
  +group_run = outer_group_run(limits,Machine.capacity(limits),LimitBounds.capacity_allocatable(limits),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},task_limit(count,U32.max(workers,1),whole,Tasks.size(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group))),(U32.max(per,1) * U32.max(group,1) : U32),unit,bound,lines,alignment,
    unit_positive,fits,bound_inside,lines_room,bound_room,
    31n,U32.is_le(U32.max((outer / U32.max(group,1) : U32),1),1),0,U32.max((outer / U32.max(group,1) : U32),1),{==},positive,bounded,room)
  group_run

# The tasks planned from a folded run stream over its tree, in parallel or as
# one task.
def planned_streams(~Model: Data,~Cell: Data,~cell: Model -> U32 -> Cell,+limits: Machine.Limits,parallel: Bool,group: Tasks.Group,+tree: Partition.Tree<&2,Model,Unit>,
  +low: Nat,+high: Nat,+unit: Nat,+bound: Nat,+facts: Runs.GroupRun(group,low,high,unit,bound),+folded: Folded(~Model,~Cell,~cell,group,tree),
  +room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Streams(~Model,~Cell,~cell,Runs.decision_tasks(Tasks.planned(parallel,Tasks.group_layout(group))),tree):
  match parallel group:
    case True{} Tasks.Group{+first,+count,+layout}:
      KernelGemm.Both{+streams,+length} = folded
      streams
    case False{} Tasks.Group{+first,+count,+layout}:
      Held.Both{+first_low,+rest} = facts
      Held.Both{+ended,+more} = rest
      Held.Both{+run,+ends} = more
      Held.Both{+layout_low,+layout_high} = ends
      KernelGemm.Both{+streams,+length} = folded
      +a = U32.to_nat(Pieces.first(layout))
      +x = U32.to_nat((Pieces.last(layout) - Pieces.first(layout) : U32))
      +c = U32.to_nat(count)
      +start_same = Equal.trans(Nat,a,low,U32.to_nat(first),Equal.trans(Nat,a,Runs.range_low(layout),low,Runs.first_low(layout),layout_low),Equal.sym(Nat,U32.to_nat(first),low,first_low))
      +sums = Equal.trans(Nat,Nat.add(a,x),high,Nat.add(a,c),
        Equal.trans(Nat,Nat.add(a,x),Runs.range_high(layout),high,Runs.whole_end(limits,layout,unit,bound,run,room),layout_high),
        Equal.trans(Nat,high,Nat.add(U32.to_nat(first),c),Nat.add(a,c),Equal.sym(Nat,Nat.add(U32.to_nat(first),c),high,ended),
          Equal.cong(Nat,Nat,value => Nat.add(value,c),U32.to_nat(first),a,Equal.sym(Nat,a,U32.to_nat(first),start_same))))
      Equal.trans(Nat,List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,tree)),c,x,length,Equal.sym(Nat,x,c,Order.cancel_sum(a,x,c,sums)))

# ---- The regrouped tree streams the cells of the tree ----

# The cells of spans, each from the next model of a queue.
def zipped(~Model: Data,~Cell: Data,~cell: Model -> U32 -> Cell,~Fill: Data,~vacant: Fill -> Model,spans: Held.Spans,+fill: Fill,+queue: List<&2,Model>) -> List<&2,Cell>:
  match spans:
    case Held.Ended{}: []
    case Held.Spanned{first,+count,rest}:
      List.append(&2,Cell,LineChainValues.cells(~Model,~Cell,~cell,U32.to_nat(count),0,Regrouped.head(~Model,~Fill,~vacant,fill,queue)),zipped(~Model,~Cell,~cell,~Fill,~vacant,rest,fill,Regrouped.tail(~Model,queue)))

# The ranges of the leaves of a tree in order onto spans after them.
def tree_spans(~Model: Data,tree: Partition.Tree<&2,Model,Unit>,rest: Held.Spans) -> Held.Spans:
  match tree:
    case Partition.Leaf{model,first,count,details}: Held.Spanned{first,count,rest}
    case Partition.Fork{left,right}: tree_spans(~Model,left,tree_spans(~Model,right,rest))

def built_zipped(~Model: Data,~Cell: Data,~cell: Model -> U32 -> Cell,~Fill: Data,~vacant: Fill -> Model,layout: Pieces.Ranges,+fill: Fill,+queue: List<&2,Model>,+rest: Held.Spans) ->
  {List.append(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,Regrouped.built(~Model,~Fill,~vacant,layout,fill,queue)),zipped(~Model,~Cell,~cell,~Fill,~vacant,rest,fill,Regrouped.taken(~Model,layout,queue))) == zipped(~Model,~Cell,~cell,~Fill,~vacant,Held.layout_spans(layout,rest),fill,queue) : List<&2,Cell>}:
  match layout:
    case Pieces.Range{+first,+count}: {==}
    case Pieces.Split{+left,+right}:
      +later = Regrouped.taken(~Model,left,queue)
      Equal.trans(List<&2,Cell>,List.append(&2,Cell,List.append(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,Regrouped.built(~Model,~Fill,~vacant,left,fill,queue)),LineChainValues.stream(~Model,~Cell,~cell,Regrouped.built(~Model,~Fill,~vacant,right,fill,later))),zipped(~Model,~Cell,~cell,~Fill,~vacant,rest,fill,Regrouped.taken(~Model,right,later))),List.append(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,Regrouped.built(~Model,~Fill,~vacant,left,fill,queue)),List.append(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,Regrouped.built(~Model,~Fill,~vacant,right,fill,later)),zipped(~Model,~Cell,~cell,~Fill,~vacant,rest,fill,Regrouped.taken(~Model,right,later)))),zipped(~Model,~Cell,~cell,~Fill,~vacant,Held.layout_spans(left,Held.layout_spans(right,rest)),fill,queue),
        ListReadback.append_associative(~Cell,LineChainValues.stream(~Model,~Cell,~cell,Regrouped.built(~Model,~Fill,~vacant,left,fill,queue)),LineChainValues.stream(~Model,~Cell,~cell,Regrouped.built(~Model,~Fill,~vacant,right,fill,later)),zipped(~Model,~Cell,~cell,~Fill,~vacant,rest,fill,Regrouped.taken(~Model,right,later))),
        Equal.trans(List<&2,Cell>,List.append(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,Regrouped.built(~Model,~Fill,~vacant,left,fill,queue)),List.append(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,Regrouped.built(~Model,~Fill,~vacant,right,fill,later)),zipped(~Model,~Cell,~cell,~Fill,~vacant,rest,fill,Regrouped.taken(~Model,right,later)))),List.append(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,Regrouped.built(~Model,~Fill,~vacant,left,fill,queue)),zipped(~Model,~Cell,~cell,~Fill,~vacant,Held.layout_spans(right,rest),fill,later)),zipped(~Model,~Cell,~cell,~Fill,~vacant,Held.layout_spans(left,Held.layout_spans(right,rest)),fill,queue),
          Equal.cong(List<&2,Cell>,List<&2,Cell>,cells => List.append(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,Regrouped.built(~Model,~Fill,~vacant,left,fill,queue)),cells),List.append(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,Regrouped.built(~Model,~Fill,~vacant,right,fill,later)),zipped(~Model,~Cell,~cell,~Fill,~vacant,rest,fill,Regrouped.taken(~Model,right,later))),zipped(~Model,~Cell,~cell,~Fill,~vacant,Held.layout_spans(right,rest),fill,later),built_zipped(~Model,~Cell,~cell,~Fill,~vacant,right,fill,later,rest)),
          built_zipped(~Model,~Cell,~cell,~Fill,~vacant,left,fill,queue,Held.layout_spans(right,rest))))

def tree_zipped(~Model: Data,~Cell: Data,~cell: Model -> U32 -> Cell,~Fill: Data,~vacant: Fill -> Model,tree: Partition.Tree<&2,Model,Unit>,+fill: Fill,+queue: List<&2,Model>,+rest: Held.Spans) ->
  {List.append(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,tree),zipped(~Model,~Cell,~cell,~Fill,~vacant,rest,fill,queue)) == zipped(~Model,~Cell,~cell,~Fill,~vacant,tree_spans(~Model,tree,rest),fill,Regrouped.queued(~Model,tree,queue)) : List<&2,Cell>}:
  match tree:
    case Partition.Leaf{+model,+first,+count,+details}: {==}
    case Partition.Fork{+left,+right}:
      Equal.trans(List<&2,Cell>,List.append(&2,Cell,List.append(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,left),LineChainValues.stream(~Model,~Cell,~cell,right)),zipped(~Model,~Cell,~cell,~Fill,~vacant,rest,fill,queue)),List.append(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,left),List.append(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,right),zipped(~Model,~Cell,~cell,~Fill,~vacant,rest,fill,queue))),zipped(~Model,~Cell,~cell,~Fill,~vacant,tree_spans(~Model,left,tree_spans(~Model,right,rest)),fill,Regrouped.queued(~Model,left,Regrouped.queued(~Model,right,queue))),
        ListReadback.append_associative(~Cell,LineChainValues.stream(~Model,~Cell,~cell,left),LineChainValues.stream(~Model,~Cell,~cell,right),zipped(~Model,~Cell,~cell,~Fill,~vacant,rest,fill,queue)),
        Equal.trans(List<&2,Cell>,List.append(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,left),List.append(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,right),zipped(~Model,~Cell,~cell,~Fill,~vacant,rest,fill,queue))),List.append(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,left),zipped(~Model,~Cell,~cell,~Fill,~vacant,tree_spans(~Model,right,rest),fill,Regrouped.queued(~Model,right,queue))),zipped(~Model,~Cell,~cell,~Fill,~vacant,tree_spans(~Model,left,tree_spans(~Model,right,rest)),fill,Regrouped.queued(~Model,left,Regrouped.queued(~Model,right,queue))),
          Equal.cong(List<&2,Cell>,List<&2,Cell>,cells => List.append(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,left),cells),List.append(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,right),zipped(~Model,~Cell,~cell,~Fill,~vacant,rest,fill,queue)),zipped(~Model,~Cell,~cell,~Fill,~vacant,tree_spans(~Model,right,rest),fill,Regrouped.queued(~Model,right,queue)),tree_zipped(~Model,~Cell,~cell,~Fill,~vacant,right,fill,queue,rest)),
          tree_zipped(~Model,~Cell,~cell,~Fill,~vacant,left,fill,Regrouped.queued(~Model,right,queue),tree_spans(~Model,right,rest))))

# Ranges with the spans of the leaves of a tree regroup it into a tree that
# streams the same cells.
def regrouped_stream(~Model: Data,~Cell: Data,~cell: Model -> U32 -> Cell,~Fill: Data,~vacant: Fill -> Model,+layout: Pieces.Ranges,+fill: Fill,+tree: Partition.Tree<&2,Model,Unit>,
  +same: {Held.layout_spans(layout,Held.Ended{}) == tree_spans(~Model,tree,Held.Ended{}) : Held.Spans}) -> {LineChainValues.stream(~Model,~Cell,~cell,Regrouped.regrouped(~Model,~Fill,~vacant,layout,fill,tree)) == LineChainValues.stream(~Model,~Cell,~cell,tree) : List<&2,Cell>}:
  Equal.trans(List<&2,Cell>,LineChainValues.stream(~Model,~Cell,~cell,Regrouped.regrouped(~Model,~Fill,~vacant,layout,fill,tree)),List.append(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,Regrouped.regrouped(~Model,~Fill,~vacant,layout,fill,tree)),[]),LineChainValues.stream(~Model,~Cell,~cell,tree),
    Equal.sym(List<&2,Cell>,List.append(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,Regrouped.regrouped(~Model,~Fill,~vacant,layout,fill,tree)),[]),LineChainValues.stream(~Model,~Cell,~cell,Regrouped.regrouped(~Model,~Fill,~vacant,layout,fill,tree)),ListReadback.append_empty(~Cell,LineChainValues.stream(~Model,~Cell,~cell,Regrouped.regrouped(~Model,~Fill,~vacant,layout,fill,tree)))),
    Equal.trans(List<&2,Cell>,List.append(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,Regrouped.regrouped(~Model,~Fill,~vacant,layout,fill,tree)),[]),zipped(~Model,~Cell,~cell,~Fill,~vacant,Held.layout_spans(layout,Held.Ended{}),fill,Regrouped.queued(~Model,tree,[])),LineChainValues.stream(~Model,~Cell,~cell,tree),
      built_zipped(~Model,~Cell,~cell,~Fill,~vacant,layout,fill,Regrouped.queued(~Model,tree,[]),Held.Ended{}),
      Equal.trans(List<&2,Cell>,zipped(~Model,~Cell,~cell,~Fill,~vacant,Held.layout_spans(layout,Held.Ended{}),fill,Regrouped.queued(~Model,tree,[])),zipped(~Model,~Cell,~cell,~Fill,~vacant,tree_spans(~Model,tree,Held.Ended{}),fill,Regrouped.queued(~Model,tree,[])),LineChainValues.stream(~Model,~Cell,~cell,tree),
        Equal.cong(Held.Spans,List<&2,Cell>,spans => zipped(~Model,~Cell,~cell,~Fill,~vacant,spans,fill,Regrouped.queued(~Model,tree,[])),Held.layout_spans(layout,Held.Ended{}),tree_spans(~Model,tree,Held.Ended{}),same),
        Equal.trans(List<&2,Cell>,zipped(~Model,~Cell,~cell,~Fill,~vacant,tree_spans(~Model,tree,Held.Ended{}),fill,Regrouped.queued(~Model,tree,[])),List.append(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,tree),[]),LineChainValues.stream(~Model,~Cell,~cell,tree),
          Equal.sym(List<&2,Cell>,List.append(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,tree),[]),zipped(~Model,~Cell,~cell,~Fill,~vacant,tree_spans(~Model,tree,Held.Ended{}),fill,Regrouped.queued(~Model,tree,[])),tree_zipped(~Model,~Cell,~cell,~Fill,~vacant,tree,fill,[],Held.Ended{})),
          ListReadback.append_empty(~Cell,LineChainValues.stream(~Model,~Cell,~cell,tree))))))

# ---- The ranges of a regrouping ----

def span_count(spans: Held.Spans) -> Nat:
  match spans:
    case Held.Ended{}: 0n
    case Held.Spanned{first,count,rest}: 1n+span_count(rest)

def layout_span_count(+layout: Pieces.Ranges,+rest: Held.Spans) ->
  {span_count(Held.layout_spans(layout,rest)) == Nat.add(Regrouped.ranges(layout),span_count(rest)) : Nat}:
  match layout:
    case Pieces.Range{first,count}: {==}
    case Pieces.Split{+left,+right}:
      +left_ranges = Regrouped.ranges(left)
      +right_ranges = Regrouped.ranges(right)
      Equal.trans(Nat,span_count(Held.layout_spans(left,Held.layout_spans(right,rest))),Nat.add(left_ranges,span_count(Held.layout_spans(right,rest))),Nat.add(Nat.add(left_ranges,right_ranges),span_count(rest)),
        layout_span_count(left,Held.layout_spans(right,rest)),
        Equal.trans(Nat,Nat.add(left_ranges,span_count(Held.layout_spans(right,rest))),Nat.add(left_ranges,Nat.add(right_ranges,span_count(rest))),Nat.add(Nat.add(left_ranges,right_ranges),span_count(rest)),
          Equal.cong(Nat,Nat,number => Nat.add(left_ranges,number),span_count(Held.layout_spans(right,rest)),Nat.add(right_ranges,span_count(rest)),layout_span_count(right,rest)),
          Equal.sym(Nat,Nat.add(Nat.add(left_ranges,right_ranges),span_count(rest)),Nat.add(left_ranges,Nat.add(right_ranges,span_count(rest))),Addition.associative(left_ranges,right_ranges,span_count(rest)))))

# The spans of the leaves of a tree are those of its ranges.
def tree_layout_spans(~Model: Data,+tree: Partition.Tree<&2,Model,Unit>,+rest: Held.Spans) ->
  {tree_spans(~Model,tree,rest) == Held.layout_spans(Regrouped.tree_layout(~Model,tree),rest) : Held.Spans}:
  match tree:
    case Partition.Leaf{model,first,count,details}: {==}
    case Partition.Fork{+left,+right}:
      Equal.trans(Held.Spans,tree_spans(~Model,left,tree_spans(~Model,right,rest)),Held.layout_spans(Regrouped.tree_layout(~Model,left),tree_spans(~Model,right,rest)),
        Held.layout_spans(Regrouped.tree_layout(~Model,left),Held.layout_spans(Regrouped.tree_layout(~Model,right),rest)),
        tree_layout_spans(~Model,left,tree_spans(~Model,right,rest)),
        Equal.cong(Held.Spans,Held.Spans,after => Held.layout_spans(Regrouped.tree_layout(~Model,left),after),tree_spans(~Model,right,rest),
          Held.layout_spans(Regrouped.tree_layout(~Model,right),rest),tree_layout_spans(~Model,right,rest)))

# Ranges with the spans of the ranges of a tree are one for each of its
# blocks.
def ranges_counted(~Model: Data,+layout: Pieces.Ranges,+tree: Partition.Tree<&2,Model,Unit>,
  +spans: {Held.layout_spans(layout,Held.Ended{}) == Held.layout_spans(Regrouped.tree_layout(~Model,tree),Held.Ended{}) : Held.Spans}) ->
  {Regrouped.ranges(layout) == List.length(&2,Model,Regrouped.queued(~Model,tree,[])) : Nat}:
  +tree_ranges = Regrouped.ranges(Regrouped.tree_layout(~Model,tree))
  +blocks = List.length(&2,Model,Regrouped.queued(~Model,tree,[]))
  Equal.trans(Nat,Regrouped.ranges(layout),Nat.add(Regrouped.ranges(layout),0n),blocks,
    Equal.sym(Nat,Nat.add(Regrouped.ranges(layout),0n),Regrouped.ranges(layout),Addition.zero_right(Regrouped.ranges(layout))),
    Equal.trans(Nat,Nat.add(Regrouped.ranges(layout),0n),span_count(Held.layout_spans(layout,Held.Ended{})),blocks,
      Equal.sym(Nat,span_count(Held.layout_spans(layout,Held.Ended{})),Nat.add(Regrouped.ranges(layout),0n),layout_span_count(layout,Held.Ended{})),
      Equal.trans(Nat,span_count(Held.layout_spans(layout,Held.Ended{})),span_count(Held.layout_spans(Regrouped.tree_layout(~Model,tree),Held.Ended{})),blocks,
        Equal.cong(Held.Spans,Nat,listed => span_count(listed),Held.layout_spans(layout,Held.Ended{}),Held.layout_spans(Regrouped.tree_layout(~Model,tree),Held.Ended{}),spans),
        Equal.trans(Nat,span_count(Held.layout_spans(Regrouped.tree_layout(~Model,tree),Held.Ended{})),Nat.add(tree_ranges,0n),blocks,
          layout_span_count(Regrouped.tree_layout(~Model,tree),Held.Ended{}),
          Equal.sym(Nat,blocks,Nat.add(tree_ranges,0n),Regrouped.queued_count(~Model,tree,[]))))))

# What a fold of the lines of pieces makes: the finals of its lines in order,
# and the pieces, as they were once regrouped into the pieces of the grid.
def LinesFolded(~Block: Type,~Fill: Data,~Start: Data,~State: Type,~Final: Data,~begin: Start -> State,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~turn: State -> Final,~vacant: Fill -> Block,
  +contexts: List<&2,Start>,+line: LineChain.Line,+group: U32,+bytes: F32,+operations: F32,+grid: Grid.Grid,+fill: Fill,+workers: U32,-pieces: Partition.Tree<&1,Block,Unit>,+finals: List<&2,Final>) -> Data:
  KernelGemm.Both<{Pair.snd(Partition.Tree<&1,Block,Unit>,List<&2,Final>,LineChain.folds(~Block,~Fill,~Start,~State,~Final,~begin,~fold,~turn,~vacant,contexts,line,group,bytes,operations,grid,fill,workers,pieces)) == finals : List<&2,Final>},
    {Pieces.blocks_regrouped(~Block,~Fill,~vacant,Pieces.ranges_of_grid(grid),fill,Pair.fst(Partition.Tree<&1,Block,Unit>,List<&2,Final>,LineChain.folds(~Block,~Fill,~Start,~State,~Final,~begin,~fold,~turn,~vacant,contexts,line,group,bytes,operations,grid,fill,workers,pieces))) == pieces : Partition.Tree<&1,Block,Unit>}>

def folded_states(~Block: Type,~Fill: Data,~Start: Data,~State: Type,~Final: Data,~begin: Start -> State,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~turn: State -> Final,~vacant: Fill -> Block,
  +contexts: List<&2,Start>,+line: LineChain.Line,+group: U32,+bytes: F32,+operations: F32,+grid: Grid.Grid,+fill: Fill,+workers: U32,-pieces: Partition.Tree<&1,Block,Unit>,+finals: List<&2,Final>,
  +folded: LinesFolded(~Block,~Fill,~Start,~State,~Final,~begin,~fold,~turn,~vacant,contexts,line,group,bytes,operations,grid,fill,workers,pieces,finals)) ->
  {Pair.snd(Partition.Tree<&1,Block,Unit>,List<&2,Final>,LineChain.folds(~Block,~Fill,~Start,~State,~Final,~begin,~fold,~turn,~vacant,contexts,line,group,bytes,operations,grid,fill,workers,pieces)) == finals : List<&2,Final>}:
  KernelGemm.Both{+same,kept} = folded
  same

def folded_kept(~Block: Type,~Fill: Data,~Start: Data,~State: Type,~Final: Data,~begin: Start -> State,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~turn: State -> Final,~vacant: Fill -> Block,
  +contexts: List<&2,Start>,+line: LineChain.Line,+group: U32,+bytes: F32,+operations: F32,+grid: Grid.Grid,+fill: Fill,+workers: U32,-pieces: Partition.Tree<&1,Block,Unit>,+finals: List<&2,Final>,
  +folded: LinesFolded(~Block,~Fill,~Start,~State,~Final,~begin,~fold,~turn,~vacant,contexts,line,group,bytes,operations,grid,fill,workers,pieces,finals)) ->
  {Pieces.blocks_regrouped(~Block,~Fill,~vacant,Pieces.ranges_of_grid(grid),fill,Pair.fst(Partition.Tree<&1,Block,Unit>,List<&2,Final>,LineChain.folds(~Block,~Fill,~Start,~State,~Final,~begin,~fold,~turn,~vacant,contexts,line,group,bytes,operations,grid,fill,workers,pieces))) == pieces : Partition.Tree<&1,Block,Unit>}:
  KernelGemm.Both{same,+kept} = folded
  kept

# ---- The stream of a tensor's pieces ----

# The values of count cells from start.
def values_from(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,count: Nat,+start: Nat) -> List<&2,Element>:
  match count:
    case 0n: []
    case 1n+rest: value(context,U32.from_nat(start)) <> values_from(~Element,~Context,~value,context,rest,1n+start)

def values_append(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,count: Nat,+more: Nat,+start: Nat) ->
  {List.append(&2,Element,values_from(~Element,~Context,~value,context,count,start),values_from(~Element,~Context,~value,context,more,Nat.add(start,count))) == values_from(~Element,~Context,~value,context,Nat.add(count,more),start) : List<&2,Element>}:
  match count:
    case 0n:
      Equal.cong(Nat,List<&2,Element>,begin => values_from(~Element,~Context,~value,context,more,begin),Nat.add(start,0n),start,Addition.zero_right(start))
    case 1n+ +rest:
      %Equal.sym(Nat,Nat.add(start,1n+rest),1n+Nat.add(start,rest),Addition.successor_right(start,rest)) :
        {value(context,U32.from_nat(start)) <> List.append(&2,Element,values_from(~Element,~Context,~value,context,rest,1n+start),values_from(~Element,~Context,~value,context,more,_)) ==
          value(context,U32.from_nat(start)) <> values_from(~Element,~Context,~value,context,Nat.add(rest,more),1n+start) : List<&2,Element>}
      Equal.cong(List<&2,Element>,List<&2,Element>,tail => value(context,U32.from_nat(start)) <> tail,
        List.append(&2,Element,values_from(~Element,~Context,~value,context,rest,1n+start),values_from(~Element,~Context,~value,context,more,Nat.add(1n+start,rest))),values_from(~Element,~Context,~value,context,Nat.add(rest,more),1n+start),
        values_append(~Element,~Context,~value,context,rest,more,1n+start))

# The pieces of a model as a model tree.
def model_tree(~Element: Data,model: PiecesModel.Model<Element>) -> Partition.Tree<&2,Buffer.Model<Element>,Unit>:
  match model:
    case PiecesModel.Leaf{buffer,first,rows}: Partition.Leaf{buffer,first,rows,Unit{}}
    case PiecesModel.Fork{left,right}: Partition.Fork{model_tree(~Element,left),model_tree(~Element,right)}

def model_tree_packed(~Element: Data,+model: PiecesModel.Model<Element>) ->
  {Regrouped.tree_of(~Storage.Buffer<Element>,~Buffer.Model<Element>,~Buffer.pack(~Element),model_tree(~Element,model)) == PiecesModel.pack(~Element,model) : Pieces.Pieces(~Element)}:
  match model:
    case PiecesModel.Leaf{buffer,first,rows}: {==}
    case PiecesModel.Fork{+left,+right}:
      %model_tree_packed(~Element,left) :
        {Partition.Fork{Regrouped.tree_of(~Storage.Buffer<Element>,~Buffer.Model<Element>,~Buffer.pack(~Element),model_tree(~Element,left)),
          Regrouped.tree_of(~Storage.Buffer<Element>,~Buffer.Model<Element>,~Buffer.pack(~Element),model_tree(~Element,right))} == Partition.Fork{_,PiecesModel.pack(~Element,right)} : Pieces.Pieces(~Element)}
      %model_tree_packed(~Element,right) :
        {Partition.Fork{Regrouped.tree_of(~Storage.Buffer<Element>,~Buffer.Model<Element>,~Buffer.pack(~Element),model_tree(~Element,left)),
          Regrouped.tree_of(~Storage.Buffer<Element>,~Buffer.Model<Element>,~Buffer.pack(~Element),model_tree(~Element,right))} ==
          Partition.Fork{Regrouped.tree_of(~Storage.Buffer<Element>,~Buffer.Model<Element>,~Buffer.pack(~Element),model_tree(~Element,left)),_} : Pieces.Pieces(~Element)}
      {==}

def model_tree_layout(~Element: Data,+model: PiecesModel.Model<Element>) ->
  {Regrouped.tree_layout(~Buffer.Model<Element>,model_tree(~Element,model)) == PiecesModel.content_layout(~Element,PiecesModel.content(~Element,model)) : Pieces.Ranges}:
  match model:
    case PiecesModel.Leaf{buffer,first,rows}: {==}
    case PiecesModel.Fork{+left,+right}:
      %Equal.sym(Pieces.Ranges,Regrouped.tree_layout(~Buffer.Model<Element>,model_tree(~Element,left)),PiecesModel.content_layout(~Element,PiecesModel.content(~Element,left)),model_tree_layout(~Element,left)) :
        {Pieces.Split{_,Regrouped.tree_layout(~Buffer.Model<Element>,model_tree(~Element,right))} ==
          Pieces.Split{PiecesModel.content_layout(~Element,PiecesModel.content(~Element,left)),PiecesModel.content_layout(~Element,PiecesModel.content(~Element,right))} : Pieces.Ranges}
      %Equal.sym(Pieces.Ranges,Regrouped.tree_layout(~Buffer.Model<Element>,model_tree(~Element,right)),PiecesModel.content_layout(~Element,PiecesModel.content(~Element,right)),model_tree_layout(~Element,right)) :
        {Pieces.Split{PiecesModel.content_layout(~Element,PiecesModel.content(~Element,left)),_} ==
          Pieces.Split{PiecesModel.content_layout(~Element,PiecesModel.content(~Element,left)),PiecesModel.content_layout(~Element,PiecesModel.content(~Element,right))} : Pieces.Ranges}
      {==}

# The parts of a shaped tree as a model tree.
def shaped_tree(~Part: Data,tree: PiecesModel.Shaped<Part>) -> Partition.Tree<&2,Part,Unit>:
  match tree:
    case PiecesModel.Piece{part,first,rows}: Partition.Leaf{part,first,rows,Unit{}}
    case PiecesModel.Pieces{left,right}: Partition.Fork{shaped_tree(~Part,left),shaped_tree(~Part,right)}

def shaped_tree_packed(~Part: Data,~Block: Type,~pack: Part -> Block,+tree: PiecesModel.Shaped<Part>) ->
  {Regrouped.tree_of(~Block,~Part,~pack,shaped_tree(~Part,tree)) == PiecesModel.shaped_pack(~Part,~Block,~pack,tree) : Partition.Tree<&1,Block,Unit>}:
  match tree:
    case PiecesModel.Piece{part,first,rows}: {==}
    case PiecesModel.Pieces{+left,+right}:
      %shaped_tree_packed(~Part,~Block,~pack,left) :
        {Partition.Fork{Regrouped.tree_of(~Block,~Part,~pack,shaped_tree(~Part,left)),Regrouped.tree_of(~Block,~Part,~pack,shaped_tree(~Part,right))} ==
          Partition.Fork{_,PiecesModel.shaped_pack(~Part,~Block,~pack,right)} : Partition.Tree<&1,Block,Unit>}
      %shaped_tree_packed(~Part,~Block,~pack,right) :
        {Partition.Fork{Regrouped.tree_of(~Block,~Part,~pack,shaped_tree(~Part,left)),Regrouped.tree_of(~Block,~Part,~pack,shaped_tree(~Part,right))} ==
          Partition.Fork{Regrouped.tree_of(~Block,~Part,~pack,shaped_tree(~Part,left)),_} : Partition.Tree<&1,Block,Unit>}
      {==}

def shaped_tree_layout(~Part: Data,+tree: PiecesModel.Shaped<Part>) ->
  {Regrouped.tree_layout(~Part,shaped_tree(~Part,tree)) == PiecesModel.shaped_plan(~Part,~Each.piece_rows(~Part),tree) : Pieces.Ranges}:
  match tree:
    case PiecesModel.Piece{part,first,rows}: {==}
    case PiecesModel.Pieces{+left,+right}:
      %Equal.sym(Pieces.Ranges,Regrouped.tree_layout(~Part,shaped_tree(~Part,left)),PiecesModel.shaped_plan(~Part,~Each.piece_rows(~Part),left),shaped_tree_layout(~Part,left)) :
        {Pieces.Split{_,Regrouped.tree_layout(~Part,shaped_tree(~Part,right))} ==
          Pieces.Split{PiecesModel.shaped_plan(~Part,~Each.piece_rows(~Part),left),PiecesModel.shaped_plan(~Part,~Each.piece_rows(~Part),right)} : Pieces.Ranges}
      %Equal.sym(Pieces.Ranges,Regrouped.tree_layout(~Part,shaped_tree(~Part,right)),PiecesModel.shaped_plan(~Part,~Each.piece_rows(~Part),right),shaped_tree_layout(~Part,right)) :
        {Pieces.Split{PiecesModel.shaped_plan(~Part,~Each.piece_rows(~Part),left),_} ==
          Pieces.Split{PiecesModel.shaped_plan(~Part,~Each.piece_rows(~Part),left),PiecesModel.shaped_plan(~Part,~Each.piece_rows(~Part),right)} : Pieces.Ranges}
      {==}

def at_read(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+first: U32,+input: Tree.Tree<Element>,+length: Nat,+index: Nat,
  +at: Content.At(~Element,~Context,~value,context,first,input,length,index)) ->
  {Tree.read(Element,input,U32.from_nat(index)) == value(context,U32.from_nat(Nat.add(U32.to_nat(first),index))) : Element}:
  KernelGemm.Both{+held,+read} = at
  read

# The cells of a piece from index on hold the values from its first cell plus
# index.
def leaf_cells(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+first: U32,+rows: U32,+buffer: Buffer.Model<Element>,count: Nat,+index: Nat,
  +values: Content.Leaf(~Element,~Context,~value,context,first,rows,Buffer.tree(~Element,buffer),Buffer.length(~Element,buffer)),
  +inside: {Nat.is_le(Nat.add(index,count),U32.to_nat(rows)) == True{} : Bool}) ->
  {LineChainValues.cells(~Buffer.Model<Element>,~Element,~cell(~Element),count,U32.from_nat(index),buffer) == values_from(~Element,~Context,~value,context,count,Nat.add(U32.to_nat(first),index)) : List<&2,Element>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +below = Order.room_index(index,rest,U32.to_nat(rows),inside)
      +read = at_read(~Element,~Context,~value,context,first,Buffer.tree(~Element,buffer),Buffer.length(~Element,buffer),index,
        Content.leaf_cell(~Element,~Context,~value,context,first,rows,Buffer.tree(~Element,buffer),Buffer.length(~Element,buffer),index,values,below))
      +next_room = Held.nat_le_moved(1n+Nat.add(index,rest),Nat.add(index,1n+rest),U32.to_nat(rows),
        Equal.sym(Nat,Nat.add(index,1n+rest),1n+Nat.add(index,rest),Addition.successor_right(index,rest)),inside)
      +tail = leaf_cells(~Element,~Context,~value,context,first,rows,buffer,rest,1n+index,values,next_room)
      %Equal.sym(U32,(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),Bounds.increment_encoded(index)) :
        {cell(~Element,buffer,U32.from_nat(index)) <> LineChainValues.cells(~Buffer.Model<Element>,~Element,~cell(~Element),rest,_,buffer) ==
          value(context,U32.from_nat(Nat.add(U32.to_nat(first),index))) <> values_from(~Element,~Context,~value,context,rest,1n+Nat.add(U32.to_nat(first),index)) : List<&2,Element>}
      %Addition.successor_right(U32.to_nat(first),index) :
        {cell(~Element,buffer,U32.from_nat(index)) <> LineChainValues.cells(~Buffer.Model<Element>,~Element,~cell(~Element),rest,U32.from_nat(1n+index),buffer) ==
          value(context,U32.from_nat(Nat.add(U32.to_nat(first),index))) <> values_from(~Element,~Context,~value,context,rest,_) : List<&2,Element>}
      Equal.trans(List<&2,Element>,cell(~Element,buffer,U32.from_nat(index)) <> LineChainValues.cells(~Buffer.Model<Element>,~Element,~cell(~Element),rest,U32.from_nat(1n+index),buffer),
        cell(~Element,buffer,U32.from_nat(index)) <> values_from(~Element,~Context,~value,context,rest,Nat.add(U32.to_nat(first),1n+index)),
        value(context,U32.from_nat(Nat.add(U32.to_nat(first),index))) <> values_from(~Element,~Context,~value,context,rest,Nat.add(U32.to_nat(first),1n+index)),
        Equal.cong(List<&2,Element>,List<&2,Element>,cells => cell(~Element,buffer,U32.from_nat(index)) <> cells,
          LineChainValues.cells(~Buffer.Model<Element>,~Element,~cell(~Element),rest,U32.from_nat(1n+index),buffer),values_from(~Element,~Context,~value,context,rest,Nat.add(U32.to_nat(first),1n+index)),tail),
        Equal.cong(Element,List<&2,Element>,head => head <> values_from(~Element,~Context,~value,context,rest,Nat.add(U32.to_nat(first),1n+index)),
          cell(~Element,buffer,U32.from_nat(index)),value(context,U32.from_nat(Nat.add(U32.to_nat(first),index))),read))

# The cells of a piece hold the values from its first cell.
def leaf_stream(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+first: U32,+rows: U32,+buffer: Buffer.Model<Element>,
  +values: Content.Leaf(~Element,~Context,~value,context,first,rows,Buffer.tree(~Element,buffer),Buffer.length(~Element,buffer))) ->
  {LineChainValues.cells(~Buffer.Model<Element>,~Element,~cell(~Element),U32.to_nat(rows),0,buffer) == values_from(~Element,~Context,~value,context,U32.to_nat(rows),U32.to_nat(first)) : List<&2,Element>}:
  Equal.trans(List<&2,Element>,LineChainValues.cells(~Buffer.Model<Element>,~Element,~cell(~Element),U32.to_nat(rows),0,buffer),
    values_from(~Element,~Context,~value,context,U32.to_nat(rows),Nat.add(U32.to_nat(first),0n)),values_from(~Element,~Context,~value,context,U32.to_nat(rows),U32.to_nat(first)),
    leaf_cells(~Element,~Context,~value,context,first,rows,buffer,U32.to_nat(rows),0n,values,Order.reflexive(U32.to_nat(rows))),
    Equal.cong(Nat,List<&2,Element>,start => values_from(~Element,~Context,~value,context,U32.to_nat(rows),start),Nat.add(U32.to_nat(first),0n),U32.to_nat(first),Addition.zero_right(U32.to_nat(first))))

# The pieces of a model that cover rows cells from first and hold a reader's
# values stream those values in order.
def model_stream(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,model: PiecesModel.Model<Element>,+first: Nat,+rows: Nat,
  +coverage: PiecesPlan.Coverage(PiecesModel.content_layout(~Element,PiecesModel.content(~Element,model)),first,rows),+values: Content.Values(~Element,~Context,~value,context,PiecesModel.content(~Element,model))) ->
  {LineChainValues.stream(~Buffer.Model<Element>,~Element,~cell(~Element),model_tree(~Element,model)) == values_from(~Element,~Context,~value,context,rows,first) : List<&2,Element>}:
  match model:
    case PiecesModel.Leaf{Buffer.Model{+tree,+length,+certificate},+piece_first,+piece_rows}:
      PiecesPlan.LeafCoverage{+first_equal,+rows_equal} = coverage
      %rows_equal : {LineChainValues.cells(~Buffer.Model<Element>,~Element,~cell(~Element),U32.to_nat(piece_rows),0,Buffer.Model{tree,length,certificate}) == values_from(~Element,~Context,~value,context,_,first) : List<&2,Element>}
      %first_equal : {LineChainValues.cells(~Buffer.Model<Element>,~Element,~cell(~Element),U32.to_nat(piece_rows),0,Buffer.Model{tree,length,certificate}) == values_from(~Element,~Context,~value,context,U32.to_nat(piece_rows),_) : List<&2,Element>}
      leaf_stream(~Element,~Context,~value,context,piece_first,piece_rows,Buffer.Model{tree,length,certificate},values)
    case PiecesModel.Fork{+left,+right}:
      PiecesPlan.ForkCoverage{+left_rows,+right_rows,+rows_equal,+left_coverage,+right_coverage} = coverage
      KernelGemm.Both{+left_values,+right_values} = values
      %Equal.sym(Nat,rows,Nat.add(left_rows,right_rows),rows_equal) :
        {List.append(&2,Element,LineChainValues.stream(~Buffer.Model<Element>,~Element,~cell(~Element),model_tree(~Element,left)),LineChainValues.stream(~Buffer.Model<Element>,~Element,~cell(~Element),model_tree(~Element,right))) == values_from(~Element,~Context,~value,context,_,first) : List<&2,Element>}
      Equal.trans(List<&2,Element>,List.append(&2,Element,LineChainValues.stream(~Buffer.Model<Element>,~Element,~cell(~Element),model_tree(~Element,left)),LineChainValues.stream(~Buffer.Model<Element>,~Element,~cell(~Element),model_tree(~Element,right))),List.append(&2,Element,values_from(~Element,~Context,~value,context,left_rows,first),LineChainValues.stream(~Buffer.Model<Element>,~Element,~cell(~Element),model_tree(~Element,right))),values_from(~Element,~Context,~value,context,Nat.add(left_rows,right_rows),first),
        Equal.cong(List<&2,Element>,List<&2,Element>,low => List.append(&2,Element,low,LineChainValues.stream(~Buffer.Model<Element>,~Element,~cell(~Element),model_tree(~Element,right))),LineChainValues.stream(~Buffer.Model<Element>,~Element,~cell(~Element),model_tree(~Element,left)),values_from(~Element,~Context,~value,context,left_rows,first),
          model_stream(~Element,~Context,~value,context,left,first,left_rows,left_coverage,left_values)),
        Equal.trans(List<&2,Element>,List.append(&2,Element,values_from(~Element,~Context,~value,context,left_rows,first),LineChainValues.stream(~Buffer.Model<Element>,~Element,~cell(~Element),model_tree(~Element,right))),List.append(&2,Element,values_from(~Element,~Context,~value,context,left_rows,first),values_from(~Element,~Context,~value,context,right_rows,Nat.add(first,left_rows))),
          values_from(~Element,~Context,~value,context,Nat.add(left_rows,right_rows),first),
          Equal.cong(List<&2,Element>,List<&2,Element>,high => List.append(&2,Element,values_from(~Element,~Context,~value,context,left_rows,first),high),LineChainValues.stream(~Buffer.Model<Element>,~Element,~cell(~Element),model_tree(~Element,right)),values_from(~Element,~Context,~value,context,right_rows,Nat.add(first,left_rows)),
            model_stream(~Element,~Context,~value,context,right,Nat.add(first,left_rows),right_rows,right_coverage,right_values)),
          values_append(~Element,~Context,~value,context,left_rows,right_rows,first)))

def values_length(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,count: Nat,+start: Nat) ->
  {List.length(&2,Element,values_from(~Element,~Context,~value,context,count,start)) == count : Nat}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,Element,values_from(~Element,~Context,~value,context,rest,1n+start)),rest,values_length(~Element,~Context,~value,context,rest,1n+start))

# The pieces of a tree of models: the blocks they represent, the ranges of
# their leaves and the cells they stream.
type Streamed<-Block: Type,-Model: Data,-Element: Data,-represent: Model -> Block,-cell: Model -> U32 -> Element,-pieces: Partition.Tree<&1,Block,Unit>,
  -layout: Pieces.Ranges,-values: List<&2,Element>> is Data:
  Streamed{tree: Partition.Tree<&2,Model,Unit>,packed: {Regrouped.tree_of(~Block,~Model,~represent,tree) == pieces : Partition.Tree<&1,Block,Unit>},
    laid: {Regrouped.tree_layout(~Model,tree) == layout : Pieces.Ranges},stream: {LineChainValues.stream(~Model,~Element,~cell,tree) == values : List<&2,Element>}}

# The pieces of a model laid out as layout that cover total cells and hold a
# reader's values stream those values.
def model_streamed(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+model: PiecesModel.Model<Element>,+layout: Pieces.Ranges,+total: Nat,
  +canonical: {PiecesModel.content_layout(~Element,PiecesModel.content(~Element,model)) == layout : Pieces.Ranges},
  +coverage: PiecesPlan.Coverage(PiecesModel.content_layout(~Element,PiecesModel.content(~Element,model)),0n,total),+values: Content.Values(~Element,~Context,~value,context,PiecesModel.content(~Element,model))) ->
  Streamed<Storage.Buffer<Element>,Buffer.Model<Element>,Element,Buffer.pack(~Element),cell(~Element),PiecesModel.pack(~Element,model),layout,values_from(~Element,~Context,~value,context,total,0n)>:
  Streamed{model_tree(~Element,model),model_tree_packed(~Element,model),
    Equal.trans(Pieces.Ranges,Regrouped.tree_layout(~Buffer.Model<Element>,model_tree(~Element,model)),PiecesModel.content_layout(~Element,PiecesModel.content(~Element,model)),layout,
      model_tree_layout(~Element,model),canonical),
    model_stream(~Element,~Context,~value,context,model,0n,total,coverage,values)}

# ---- Trailing axes of a grid ----

def units_outer(found: Grid.Units) -> U32:
  Grid.Units{prefix,outer,extent,unit_cells} = found
  outer

def units_extent(found: Grid.Units) -> U32:
  Grid.Units{prefix,outer,extent,unit_cells} = found
  extent

# The trailing axes of a grid's cells, lines of length cells after lines of
# them: they lie in its units, or they hold the units of whole outer indices,
# size of them a line.
type Trailing<-found: Grid.Units,-lines: Nat,-length: Nat> is Data:
  Within{units: Nat,exact: {U32.to_nat(GridRanges.unit_cells_of(found)) == Nat.mul(units,length) : Nat}}
  Carried{size: Nat,outer_exact: {U32.to_nat(units_outer(found)) == Nat.mul(lines,size) : Nat},
    length_exact: {length == Nat.mul(size,Nat.mul(U32.to_nat(units_extent(found)),U32.to_nat(GridRanges.unit_cells_of(found)))) : Nat}}

# Once the walk over the axes passes the first trailing axis, the outer
# indices of a line multiply.
def carried_walk(+limits: Machine.Limits,extents: List<&2,U32>,fits: Bool,+prefix: U32,+outer: U32,+extent: U32,+target: U32,+lines: Nat,+size: Nat,+outer_exact: {U32.to_nat(outer) == Nat.mul(lines,size) : Nat},
  +room: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(extents)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+outer_positive: {Nat.is_lt(0n,U32.to_nat(outer)) == True{} : Bool},
  +extent_positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  Trailing<Grid.units_from(extents,Grid.strides(extents),fits,prefix,outer,extent,target),lines,Nat.mul(size,Nat.mul(U32.to_nat(extent),ShapeLaws.elements(extents)))>:
  match extents fits:
    case extents True{}:
      +exact = TiledPublic.product_exact(Machine.capacity(limits),LimitBounds.capacity_fits(limits),extents,
        Order.transitive(ShapeLaws.elements(extents),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(extents)),U32.to_nat(Machine.capacity(limits)),Held.nat_le_right_moved(ShapeLaws.elements(extents),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(extents)),Nat.mul(ShapeLaws.elements(extents),Nat.mul(U32.to_nat(outer),U32.to_nat(extent))),RowMajor.multiply_commutative(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(extents)),
          RowMajor.product_covers(ShapeLaws.elements(extents),Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),Order.product_positive(U32.to_nat(outer),U32.to_nat(extent),outer_positive,extent_positive))),room))
      %Equal.sym(Grid.Units,Grid.units_from(extents,Grid.strides(extents),True{},prefix,outer,extent,target),Grid.Units{prefix,outer,extent,Grid.product(extents)},GridPieces.units_fit(extents,prefix,outer,extent,target)) :
        Trailing<_,lines,Nat.mul(size,Nat.mul(U32.to_nat(extent),ShapeLaws.elements(extents)))>
      Carried{size,outer_exact,Equal.cong(Nat,Nat,cells => Nat.mul(size,Nat.mul(U32.to_nat(extent),cells)),ShapeLaws.elements(extents),U32.to_nat(Grid.product(extents)),Equal.sym(Nat,U32.to_nat(Grid.product(extents)),ShapeLaws.elements(extents),exact))}
    case Nil{} False{}: Carried{size,outer_exact,{==}}
    case Con{+axis,Nil{}} False{}:
      Carried{size,outer_exact,Equal.cong(Nat,Nat,cells => Nat.mul(size,Nat.mul(U32.to_nat(extent),cells)),Nat.mul(U32.to_nat(axis),1n),U32.to_nat(axis),Order.times_one(U32.to_nat(axis)))}
    case Con{+axis,Con{+next,+tail}} False{}:
      +axis_positive = RowMajor.left_factor_positive(U32.to_nat(axis),ShapeLaws.elements(Con{next,tail}),nonempty)
      +others_positive = GridRanges.right_factor_positive(U32.to_nat(axis),ShapeLaws.elements(Con{next,tail}),nonempty)
      +outer_room = Order.transitive(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(Con{axis,Con{next,tail}})),U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(Con{axis,Con{next,tail}}),nonempty),room)
      +outer_word = Multiplication.bounded_multiply(outer,extent,Machine.capacity(limits),LimitBounds.capacity_fits(limits),outer_room)
      +regrouped = Equal.trans(Nat,Nat.mul(Nat.mul(U32.to_nat((outer * extent : U32)),U32.to_nat(axis)),ShapeLaws.elements(Con{next,tail})),Nat.mul(Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(axis)),ShapeLaws.elements(Con{next,tail})),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(Con{axis,Con{next,tail}})),
        Equal.cong(Nat,Nat,value => Nat.mul(Nat.mul(value,U32.to_nat(axis)),ShapeLaws.elements(Con{next,tail})),U32.to_nat((outer * extent : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),outer_word),RowMajor.assoc(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(axis),ShapeLaws.elements(Con{next,tail})))
      +next_room = Held.nat_le_moved(Nat.mul(Nat.mul(U32.to_nat((outer * extent : U32)),U32.to_nat(axis)),ShapeLaws.elements(Con{next,tail})),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(Con{axis,Con{next,tail}})),U32.to_nat(Machine.capacity(limits)),regrouped,room)
      +next_positive = {%Equal.sym(Nat,U32.to_nat((outer * extent : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),outer_word) : {Nat.is_lt(0n,_) == True{} : Bool}
        Order.product_positive(U32.to_nat(outer),U32.to_nat(extent),outer_positive,extent_positive) : {Nat.is_lt(0n,U32.to_nat((outer * extent : U32))) == True{} : Bool}}
      +next_exact = Equal.trans(Nat,U32.to_nat((outer * extent : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),Nat.mul(lines,Nat.mul(size,U32.to_nat(extent))),outer_word,
        Equal.trans(Nat,Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),Nat.mul(Nat.mul(lines,size),U32.to_nat(extent)),Nat.mul(lines,Nat.mul(size,U32.to_nat(extent))),
          Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(extent)),U32.to_nat(outer),Nat.mul(lines,size),outer_exact),RowMajor.assoc(lines,size,U32.to_nat(extent))))
      %Equal.sym(List<&2,U32>,Grid.strides(Con{axis,Con{next,tail}}),Con{Grid.product(Con{next,tail}),Grid.strides(Con{next,tail})},GridPieces.strides_con(axis,Con{next,tail})) :
        Trailing<Grid.units_from(Con{axis,Con{next,tail}},_,False{},prefix,outer,extent,target),lines,Nat.mul(size,Nat.mul(U32.to_nat(extent),ShapeLaws.elements(Con{axis,Con{next,tail}})))>
      %RowMajor.assoc(size,U32.to_nat(extent),ShapeLaws.elements(Con{axis,Con{next,tail}})) : Trailing<Grid.units_from(Con{next,tail},Grid.strides(Con{next,tail}),U32.is_le(Grid.product(Con{next,tail}),target),(prefix + 1 : U32),(outer * extent : U32),axis,target),lines,_>
      carried_walk(limits,Con{next,tail},U32.is_le(Grid.product(Con{next,tail}),target),(prefix + 1 : U32),(outer * extent : U32),axis,target,lines,Nat.mul(size,U32.to_nat(extent)),next_exact,next_room,next_positive,axis_positive,others_positive)

# Before the first trailing axis the walk counts the outer indices of the
# lines.
def kept_walk(+limits: Machine.Limits,extents: List<&2,U32>,fits: Bool,+prefix: U32,+outer: U32,+extent: U32,+target: U32,kept: Nat,+done: Nat,+done_exact: {Nat.mul(U32.to_nat(outer),U32.to_nat(extent)) == done : Nat},
  +room: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(extents)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+outer_positive: {Nat.is_lt(0n,U32.to_nat(outer)) == True{} : Bool},
  +extent_positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  Trailing<Grid.units_from(extents,Grid.strides(extents),fits,prefix,outer,extent,target),Nat.mul(done,ShapeLaws.elements(List.take(&2,U32,extents,kept))),ShapeLaws.elements(List.drop(&2,U32,extents,kept))>:
  match extents fits kept:
    case extents True{} +kept:
      +exact = TiledPublic.product_exact(Machine.capacity(limits),LimitBounds.capacity_fits(limits),extents,
        Order.transitive(ShapeLaws.elements(extents),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(extents)),U32.to_nat(Machine.capacity(limits)),Held.nat_le_right_moved(ShapeLaws.elements(extents),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(extents)),Nat.mul(ShapeLaws.elements(extents),Nat.mul(U32.to_nat(outer),U32.to_nat(extent))),RowMajor.multiply_commutative(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(extents)),
          RowMajor.product_covers(ShapeLaws.elements(extents),Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),Order.product_positive(U32.to_nat(outer),U32.to_nat(extent),outer_positive,extent_positive))),room))
      %Equal.sym(Grid.Units,Grid.units_from(extents,Grid.strides(extents),True{},prefix,outer,extent,target),Grid.Units{prefix,outer,extent,Grid.product(extents)},GridPieces.units_fit(extents,prefix,outer,extent,target)) :
        Trailing<_,Nat.mul(done,ShapeLaws.elements(List.take(&2,U32,extents,kept))),ShapeLaws.elements(List.drop(&2,U32,extents,kept))>
      Within{ShapeLaws.elements(List.take(&2,U32,extents,kept)),
        Equal.trans(Nat,U32.to_nat(Grid.product(extents)),ShapeLaws.elements(extents),Nat.mul(ShapeLaws.elements(List.take(&2,U32,extents,kept)),ShapeLaws.elements(List.drop(&2,U32,extents,kept))),
          exact,CountCompletion.elements_split(extents,kept))}
    case Nil{} False{} +kept: Within{ShapeLaws.elements(List.take(&2,U32,[],kept)),CountCompletion.elements_split([],kept)}
    case Con{+axis,Nil{}} False{} +kept:
      Within{ShapeLaws.elements(List.take(&2,U32,[axis],kept)),
        Equal.trans(Nat,U32.to_nat(axis),Nat.mul(U32.to_nat(axis),1n),Nat.mul(ShapeLaws.elements(List.take(&2,U32,[axis],kept)),ShapeLaws.elements(List.drop(&2,U32,[axis],kept))),
          Equal.sym(Nat,Nat.mul(U32.to_nat(axis),1n),U32.to_nat(axis),Order.times_one(U32.to_nat(axis))),CountCompletion.elements_split([axis],kept))}
    case Con{+axis,Con{+next,+tail}} False{} 0n:
      +axis_positive = RowMajor.left_factor_positive(U32.to_nat(axis),ShapeLaws.elements(Con{next,tail}),nonempty)
      +others_positive = GridRanges.right_factor_positive(U32.to_nat(axis),ShapeLaws.elements(Con{next,tail}),nonempty)
      +outer_room = Order.transitive(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(Con{axis,Con{next,tail}})),U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(Con{axis,Con{next,tail}}),nonempty),room)
      +outer_word = Multiplication.bounded_multiply(outer,extent,Machine.capacity(limits),LimitBounds.capacity_fits(limits),outer_room)
      +regrouped = Equal.trans(Nat,Nat.mul(Nat.mul(U32.to_nat((outer * extent : U32)),U32.to_nat(axis)),ShapeLaws.elements(Con{next,tail})),Nat.mul(Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(axis)),ShapeLaws.elements(Con{next,tail})),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(Con{axis,Con{next,tail}})),
        Equal.cong(Nat,Nat,value => Nat.mul(Nat.mul(value,U32.to_nat(axis)),ShapeLaws.elements(Con{next,tail})),U32.to_nat((outer * extent : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),outer_word),RowMajor.assoc(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(axis),ShapeLaws.elements(Con{next,tail})))
      +next_room = Held.nat_le_moved(Nat.mul(Nat.mul(U32.to_nat((outer * extent : U32)),U32.to_nat(axis)),ShapeLaws.elements(Con{next,tail})),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(Con{axis,Con{next,tail}})),U32.to_nat(Machine.capacity(limits)),regrouped,room)
      +next_positive = {%Equal.sym(Nat,U32.to_nat((outer * extent : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),outer_word) : {Nat.is_lt(0n,_) == True{} : Bool}
        Order.product_positive(U32.to_nat(outer),U32.to_nat(extent),outer_positive,extent_positive) : {Nat.is_lt(0n,U32.to_nat((outer * extent : U32))) == True{} : Bool}}
      +outer_done = Equal.trans(Nat,U32.to_nat((outer * extent : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),done,outer_word,done_exact)
      %Equal.sym(List<&2,U32>,Grid.strides(Con{axis,Con{next,tail}}),Con{Grid.product(Con{next,tail}),Grid.strides(Con{next,tail})},GridPieces.strides_con(axis,Con{next,tail})) :
        Trailing<Grid.units_from(Con{axis,Con{next,tail}},_,False{},prefix,outer,extent,target),Nat.mul(done,ShapeLaws.elements(List.take(&2,U32,Con{axis,Con{next,tail}},0n))),ShapeLaws.elements(List.drop(&2,U32,Con{axis,Con{next,tail}},0n))>
      %Equal.sym(Nat,Nat.mul(done,1n),done,Order.times_one(done)) :
        Trailing<Grid.units_from(Con{next,tail},Grid.strides(Con{next,tail}),U32.is_le(Grid.product(Con{next,tail}),target),(prefix + 1 : U32),(outer * extent : U32),axis,target),_,ShapeLaws.elements(Con{axis,Con{next,tail}})>
      %LineChainValues.one_times(ShapeLaws.elements(Con{axis,Con{next,tail}})) :
        Trailing<Grid.units_from(Con{next,tail},Grid.strides(Con{next,tail}),U32.is_le(Grid.product(Con{next,tail}),target),(prefix + 1 : U32),(outer * extent : U32),axis,target),done,_>
      carried_walk(limits,Con{next,tail},U32.is_le(Grid.product(Con{next,tail}),target),(prefix + 1 : U32),(outer * extent : U32),axis,target,done,1n,Equal.trans(Nat,U32.to_nat((outer * extent : U32)),done,Nat.mul(done,1n),outer_done,Equal.sym(Nat,Nat.mul(done,1n),done,Order.times_one(done))),
        next_room,next_positive,axis_positive,others_positive)
    case Con{+axis,Con{+next,+tail}} False{} 1n+ +rest:
      +axis_positive = RowMajor.left_factor_positive(U32.to_nat(axis),ShapeLaws.elements(Con{next,tail}),nonempty)
      +others_positive = GridRanges.right_factor_positive(U32.to_nat(axis),ShapeLaws.elements(Con{next,tail}),nonempty)
      +outer_room = Order.transitive(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(Con{axis,Con{next,tail}})),U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(Con{axis,Con{next,tail}}),nonempty),room)
      +outer_word = Multiplication.bounded_multiply(outer,extent,Machine.capacity(limits),LimitBounds.capacity_fits(limits),outer_room)
      +regrouped = Equal.trans(Nat,Nat.mul(Nat.mul(U32.to_nat((outer * extent : U32)),U32.to_nat(axis)),ShapeLaws.elements(Con{next,tail})),Nat.mul(Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(axis)),ShapeLaws.elements(Con{next,tail})),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(Con{axis,Con{next,tail}})),
        Equal.cong(Nat,Nat,value => Nat.mul(Nat.mul(value,U32.to_nat(axis)),ShapeLaws.elements(Con{next,tail})),U32.to_nat((outer * extent : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),outer_word),RowMajor.assoc(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(axis),ShapeLaws.elements(Con{next,tail})))
      +next_room = Held.nat_le_moved(Nat.mul(Nat.mul(U32.to_nat((outer * extent : U32)),U32.to_nat(axis)),ShapeLaws.elements(Con{next,tail})),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),ShapeLaws.elements(Con{axis,Con{next,tail}})),U32.to_nat(Machine.capacity(limits)),regrouped,room)
      +next_positive = {%Equal.sym(Nat,U32.to_nat((outer * extent : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),outer_word) : {Nat.is_lt(0n,_) == True{} : Bool}
        Order.product_positive(U32.to_nat(outer),U32.to_nat(extent),outer_positive,extent_positive) : {Nat.is_lt(0n,U32.to_nat((outer * extent : U32))) == True{} : Bool}}
      +outer_done = Equal.trans(Nat,U32.to_nat((outer * extent : U32)),Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),done,outer_word,done_exact)
      %Equal.sym(List<&2,U32>,Grid.strides(Con{axis,Con{next,tail}}),Con{Grid.product(Con{next,tail}),Grid.strides(Con{next,tail})},GridPieces.strides_con(axis,Con{next,tail})) :
        Trailing<Grid.units_from(Con{axis,Con{next,tail}},_,False{},prefix,outer,extent,target),Nat.mul(done,ShapeLaws.elements(List.take(&2,U32,Con{axis,Con{next,tail}},1n+rest))),ShapeLaws.elements(List.drop(&2,U32,Con{axis,Con{next,tail}},1n+rest))>
      %RowMajor.assoc(done,U32.to_nat(axis),ShapeLaws.elements(List.take(&2,U32,Con{next,tail},rest))) :
        Trailing<Grid.units_from(Con{next,tail},Grid.strides(Con{next,tail}),U32.is_le(Grid.product(Con{next,tail}),target),(prefix + 1 : U32),(outer * extent : U32),axis,target),_,ShapeLaws.elements(List.drop(&2,U32,Con{next,tail},rest))>
      kept_walk(limits,Con{next,tail},U32.is_le(Grid.product(Con{next,tail}),target),(prefix + 1 : U32),(outer * extent : U32),axis,target,rest,Nat.mul(done,U32.to_nat(axis)),Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(axis)),U32.to_nat((outer * extent : U32)),done,outer_done),
        next_room,next_positive,axis_positive,others_positive)

# ---- Trailing lines of a grid ----

# A part of a divisor holds it once at most.
def once_at_most(+part: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool},bound: Order.UpperBound<part,divisor>) ->
  {Nat.max(Nat.div(part,divisor),1n) == 1n : Nat}:
  match bound:
    case Order.Before{inside}:
      %Equal.sym(Nat,Nat.div(part,divisor),0n,Radix.quotient(0n,part,divisor,positive,inside)) : {Nat.max(_,1n) == 1n : Nat}
      {==}
    case Order.At{equal}:
      +whole = Equal.trans(Nat,Nat.add(Nat.mul(1n,divisor),0n),Nat.mul(1n,divisor),divisor,Addition.zero_right(Nat.mul(1n,divisor)),LineChainValues.one_times(divisor))
      %Equal.sym(Nat,part,divisor,equal) : {Nat.max(Nat.div(_,divisor),1n) == 1n : Nat}
      %whole : {Nat.max(Nat.div(_,divisor),1n) == 1n : Nat}
      %Equal.sym(Nat,Nat.div(Nat.add(Nat.mul(1n,divisor),0n),divisor),1n,Radix.quotient(1n,0n,divisor,positive,positive)) : {Nat.max(_,1n) == 1n : Nat}
      {==}

# Lines no longer than the cells of an outer index form groups of one.
def single_group(+length: U32,+cells: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+within: {Nat.is_le(U32.to_nat(length),U32.to_nat(cells)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(cells)) == True{} : Bool},+divisor_bound: {Nat.is_le(Nat.double(U32.to_nat(cells)),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.max((length / cells : U32),1) == 1 : U32}:
  +quotient = U32Division.quotient_exact(cap,cap_ok,length,cells,U32.to_nat(length),U32.to_nat(cells),{==},{==},divisor_bound,positive)
  Arithmetic.to_nat_injective(U32.max((length / cells : U32),1),1,
    Equal.trans(Nat,U32.to_nat(U32.max((length / cells : U32),1)),Nat.max(U32.to_nat((length / cells : U32)),1n),1n,Extrema.maximum((length / cells : U32),1),
      Equal.trans(Nat,Nat.max(U32.to_nat((length / cells : U32)),1n),Nat.max(Nat.div(U32.to_nat(length),U32.to_nat(cells)),1n),1n,
        Equal.cong(Nat,Nat,number => Nat.max(number,1n),U32.to_nat((length / cells : U32)),Nat.div(U32.to_nat(length),U32.to_nat(cells)),quotient),
        once_at_most(U32.to_nat(length),U32.to_nat(cells),positive,Order.upper_bound_cases(U32.to_nat(length),U32.to_nat(cells),within)))))

# The cells of an outer index fit the machine.
def per_outer_room(+limits: Machine.Limits,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+target: U32,+outer_positive: {Nat.is_lt(0n,U32.to_nat(outer)) == True{} : Bool},+extent_positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},+unit_positive: {Nat.is_lt(0n,U32.to_nat(unit_cells)) == True{} : Bool},+room: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {Nat.is_le(Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Order.transitive(Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits)),
    Held.nat_le_right_moved(Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),Nat.mul(Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),U32.to_nat(outer)),
      Equal.trans(Nat,Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(outer),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),U32.to_nat(outer)),RowMajor.assoc(U32.to_nat(outer),U32.to_nat(extent),U32.to_nat(unit_cells)),RowMajor.multiply_commutative(U32.to_nat(outer),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)))),
      RowMajor.product_covers(Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),U32.to_nat(outer),outer_positive)),
    room)

# The cells of an outer index of the grid of units.
def per_outer_word(+limits: Machine.Limits,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+target: U32,+outer_positive: {Nat.is_lt(0n,U32.to_nat(outer)) == True{} : Bool},+extent_positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},+unit_positive: {Nat.is_lt(0n,U32.to_nat(unit_cells)) == True{} : Bool},+room: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {U32.to_nat(U32.max(Layout.cells_per_outer(Grid.grid_of_units(Grid.Units{prefix,outer,extent,unit_cells},target)),1)) == Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)) : Nat}:
  +kept = GridPieces.maximum_kept(outer,1,Order.positive_one(U32.to_nat(outer),outer_positive))
  +spread = Equal.trans(Nat,U32.to_nat((outer * extent * unit_cells : U32)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),Nat.mul(Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),U32.to_nat(outer)),GridRanges.units_word(limits,outer,extent,unit_cells,unit_positive,room),
    Equal.trans(Nat,Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(outer),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),U32.to_nat(outer)),RowMajor.assoc(U32.to_nat(outer),U32.to_nat(extent),U32.to_nat(unit_cells)),RowMajor.multiply_commutative(U32.to_nat(outer),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)))))
  +outer_room = Order.transitive(U32.to_nat(outer),Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(U32.to_nat(outer),U32.to_nat(extent),extent_positive),
    Order.transitive(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells),unit_positive),room))
  +quotient = exact_quotient(Machine.half_word(limits),LimitBounds.half_word_fits(limits),(outer * extent * unit_cells : U32),outer,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),spread,outer_positive,LimitBounds.double_bound(limits,U32.to_nat(outer),outer_room))
  %Equal.sym(U32,U32.max(outer,1),outer,kept) : {U32.to_nat(U32.max(((outer * extent * unit_cells : U32) / _ : U32),1)) == Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)) : Nat}
  Extrema.max_one_exact(((outer * extent * unit_cells : U32) / outer : U32),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),quotient,Order.product_positive(U32.to_nat(extent),U32.to_nat(unit_cells),extent_positive,unit_positive))

def per_outer_positive(+limits: Machine.Limits,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+target: U32,+outer_positive: {Nat.is_lt(0n,U32.to_nat(outer)) == True{} : Bool},+extent_positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},+unit_positive: {Nat.is_lt(0n,U32.to_nat(unit_cells)) == True{} : Bool},+room: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {Nat.is_lt(0n,U32.to_nat(U32.max(Layout.cells_per_outer(Grid.grid_of_units(Grid.Units{prefix,outer,extent,unit_cells},target)),1))) == True{} : Bool}:
  Held.nat_lt_moved(0n,U32.to_nat(U32.max(Layout.cells_per_outer(Grid.grid_of_units(Grid.Units{prefix,outer,extent,unit_cells},target)),1)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),per_outer_word(limits,prefix,outer,extent,unit_cells,target,outer_positive,extent_positive,unit_positive,room),Order.product_positive(U32.to_nat(extent),U32.to_nat(unit_cells),extent_positive,unit_positive))

def per_outer_double(+limits: Machine.Limits,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+target: U32,+outer_positive: {Nat.is_lt(0n,U32.to_nat(outer)) == True{} : Bool},+extent_positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},+unit_positive: {Nat.is_lt(0n,U32.to_nat(unit_cells)) == True{} : Bool},+room: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {Nat.is_le(Nat.double(U32.to_nat(U32.max(Layout.cells_per_outer(Grid.grid_of_units(Grid.Units{prefix,outer,extent,unit_cells},target)),1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  LimitBounds.double_bound(limits,U32.to_nat(U32.max(Layout.cells_per_outer(Grid.grid_of_units(Grid.Units{prefix,outer,extent,unit_cells},target)),1)),Held.nat_le_moved(U32.to_nat(U32.max(Layout.cells_per_outer(Grid.grid_of_units(Grid.Units{prefix,outer,extent,unit_cells},target)),1)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits)),per_outer_word(limits,prefix,outer,extent,unit_cells,target,outer_positive,extent_positive,unit_positive,room),per_outer_room(limits,prefix,outer,extent,unit_cells,target,outer_positive,extent_positive,unit_positive,room)))

# ---- Lines that begin from contexts in turn, inner lines a block ----

# The values the inner lines of a block begin from: the contexts in turn from
# the first.
def block_starts(~Start: Data,~Value: Data,~begin_value: Start -> Value,+contexts: List<&2,Start>,+inner: U32) -> List<&2,Value>:
  LineChainValues.drawn(~Start,~Value,~begin_value,U32.to_nat(inner),contexts,contexts)

# The finals of groups blocks of a list, a block length * inner cells and its
# inner lines inner cells apart, every line from its start.
def block_folds(~Cell: Data,~Value: Data,~Final: Data,~step: Value -> Cell -> Value,~turn_value: Value -> Final,groups: Nat,+length: U32,+inner: U32,+list: List<&2,Cell>,+starts: List<&2,Value>) -> List<&2,Final>:
  match groups:
    case 0n: []
    case 1n+rest:
      List.append(&2,Final,LineChainValues.group_lines(~Cell,~Value,~Final,~step,~turn_value,U32.to_nat(inner),0n,U32.to_nat(inner),List.take(&2,Cell,list,U32.to_nat((length * inner : U32))),starts),
        block_folds(~Cell,~Value,~Final,~step,~turn_value,rest,length,inner,List.drop(&2,Cell,list,U32.to_nat((length * inner : U32))),starts))

# The inner lines of a block as whole turns of the contexts.
type Turns<-Start: Data,-contexts: List<&2,Start>,-inner: U32> is Data:
  Turns{laps: Nat,positive: {Nat.is_lt(0n,List.length(&2,Start,contexts)) == True{} : Bool},inner_exact: {U32.to_nat(inner) == Nat.mul(laps,List.length(&2,Start,contexts)) : Nat}}

def zero_remainder(+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) -> {Nat.mod(0n,divisor) == 0n : Nat}:
  Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(0n,divisor),(0n,0n),Division.small(0n,divisor,positive))

# Inner lines a whole number of turns of the contexts draw them round to the
# first again.
def drawn_round(~Start: Data,+contexts: List<&2,Start>,+inner: U32,+laps: Nat,+positive: {Nat.is_lt(0n,List.length(&2,Start,contexts)) == True{} : Bool},
  +inner_exact: {U32.to_nat(inner) == Nat.mul(laps,List.length(&2,Start,contexts)) : Nat}) ->
  {LineChainValues.after_drawn(~Start,U32.to_nat(inner),contexts,contexts) == contexts : List<&2,Start>}:
  match contexts:
    case Nil{}: Empty.absurd({LineChainValues.after_drawn(~Start,U32.to_nat(inner),[],[]) == [] : List<&2,Start>},WordEncoding.false_true(positive))
    case Con{+head,+tail}:
      +all = {head <> tail : List<&2,Start>}
      +kinds = List.length(&2,Start,all)
      +multiple = Equal.trans(Nat,U32.to_nat(inner),Nat.mul(laps,kinds),Nat.add(Nat.mul(laps,kinds),0n),inner_exact,
        Equal.sym(Nat,Nat.add(Nat.mul(laps,kinds),0n),Nat.mul(laps,kinds),Addition.zero_right(Nat.mul(laps,kinds))))
      +whole_laps = {%Equal.sym(Nat,U32.to_nat(inner),Nat.add(Nat.mul(laps,kinds),0n),multiple) : {Nat.mod(_,kinds) == 0n : Nat}
          Equal.trans(Nat,Nat.mod(Nat.add(Nat.mul(laps,kinds),0n),kinds),Nat.mod(0n,kinds),0n,NatDivision.remainder_shift(0n,kinds,laps,positive),zero_remainder(kinds,positive)) :
        {Nat.mod(U32.to_nat(inner),kinds) == 0n : Nat}}
      +after = {%zero_remainder(kinds,positive) : {Nat.mod(U32.to_nat(inner),kinds) == LineChainValues.residue_after(U32.to_nat(inner),_,kinds) : Nat}
          LineChainValues.mod_after(U32.to_nat(inner),0n,kinds,positive) :
        {Nat.mod(U32.to_nat(inner),kinds) == LineChainValues.residue_after(U32.to_nat(inner),0n,kinds) : Nat}}
      +turned = Equal.trans(Nat,LineChainValues.residue_after(U32.to_nat(inner),0n,kinds),Nat.mod(U32.to_nat(inner),kinds),0n,
        Equal.sym(Nat,Nat.mod(U32.to_nat(inner),kinds),LineChainValues.residue_after(U32.to_nat(inner),0n,kinds),after),whole_laps)
      %LineChainValues.drop_zero(~Start,all) : {LineChainValues.after_drawn(~Start,U32.to_nat(inner),all,_) == all : List<&2,Start>}
      Equal.trans(List<&2,Start>,LineChainValues.after_drawn(~Start,U32.to_nat(inner),all,List.drop(&2,Start,all,0n)),List.drop(&2,Start,all,LineChainValues.residue_after(U32.to_nat(inner),0n,kinds)),all,
        LineChainValues.after_drawn_residue(~Start,U32.to_nat(inner),0n,all,head,positive),
        Equal.trans(List<&2,Start>,List.drop(&2,Start,all,LineChainValues.residue_after(U32.to_nat(inner),0n,kinds)),List.drop(&2,Start,all,0n),all,
          Equal.cong(Nat,List<&2,Start>,index => List.drop(&2,Start,all,index),LineChainValues.residue_after(U32.to_nat(inner),0n,kinds),0n,turned),LineChainValues.drop_zero(~Start,all)))

# Blocks whose contexts come round again every block begin every block from
# the same values.
def grouped_blocks(~Cell: Data,~Start: Data,~Value: Data,~Final: Data,~step: Value -> Cell -> Value,~begin_value: Start -> Value,~turn_value: Value -> Final,groups: Nat,+length: U32,+inner: U32,
  +list: List<&2,Cell>,+contexts: List<&2,Start>,+round: {LineChainValues.after_drawn(~Start,U32.to_nat(inner),contexts,contexts) == contexts : List<&2,Start>}) ->
  {LineChainValues.grouped(~Cell,~Start,~Value,~Final,~step,~begin_value,~turn_value,contexts,LineChain.Line{length,inner},groups,list,contexts) ==
    block_folds(~Cell,~Value,~Final,~step,~turn_value,groups,length,inner,list,block_starts(~Start,~Value,~begin_value,contexts,inner)) : List<&2,Final>}:
  match groups:
    case 0n: {==}
    case 1n+ +rest:
      +starts = block_starts(~Start,~Value,~begin_value,contexts,inner)
      +size = U32.to_nat((length * inner : U32))
      +lines = LineChainValues.group_lines(~Cell,~Value,~Final,~step,~turn_value,U32.to_nat(inner),0n,U32.to_nat(inner),List.take(&2,Cell,list,size),starts)
      +later = List.drop(&2,Cell,list,size)
      %Equal.sym(List<&2,Start>,LineChainValues.after_drawn(~Start,U32.to_nat(inner),contexts,contexts),contexts,round) :
        {List.append(&2,Final,lines,LineChainValues.grouped(~Cell,~Start,~Value,~Final,~step,~begin_value,~turn_value,contexts,LineChain.Line{length,inner},rest,later,_)) ==
          List.append(&2,Final,lines,block_folds(~Cell,~Value,~Final,~step,~turn_value,rest,length,inner,later,starts)) : List<&2,Final>}
      Equal.cong(List<&2,Final>,List<&2,Final>,others => List.append(&2,Final,lines,others),
        LineChainValues.grouped(~Cell,~Start,~Value,~Final,~step,~begin_value,~turn_value,contexts,LineChain.Line{length,inner},rest,later,contexts),block_folds(~Cell,~Value,~Final,~step,~turn_value,rest,length,inner,later,starts),
        grouped_blocks(~Cell,~Start,~Value,~Final,~step,~begin_value,~turn_value,rest,length,inner,later,contexts,round))

# The finals of two lists of whole blocks, one after the other, are the finals
# of the lists appended.
def block_append(~Cell: Data,~Value: Data,~Final: Data,~step: Value -> Cell -> Value,~turn_value: Value -> Final,first: Nat,+second: Nat,+length: U32,+inner: U32,+left: List<&2,Cell>,+right: List<&2,Cell>,+starts: List<&2,Value>,
  +whole: {List.length(&2,Cell,left) == Nat.mul(first,U32.to_nat((length * inner : U32))) : Nat}) ->
  {List.append(&2,Final,block_folds(~Cell,~Value,~Final,~step,~turn_value,first,length,inner,left,starts),block_folds(~Cell,~Value,~Final,~step,~turn_value,second,length,inner,right,starts)) ==
    block_folds(~Cell,~Value,~Final,~step,~turn_value,Nat.add(first,second),length,inner,List.append(&2,Cell,left,right),starts) : List<&2,Final>}:
  match first:
    case 0n:
      %Equal.sym(List<&2,Cell>,left,[],LineChainValues.empty_of_length(~Cell,left,whole)) :
        {block_folds(~Cell,~Value,~Final,~step,~turn_value,second,length,inner,right,starts) == block_folds(~Cell,~Value,~Final,~step,~turn_value,second,length,inner,List.append(&2,Cell,_,right),starts) : List<&2,Final>}
      {==}
    case 1n+ +rest:
      +size = U32.to_nat((length * inner : U32))
      +inside = {%Equal.sym(Nat,List.length(&2,Cell,left),Nat.add(size,Nat.mul(rest,size)),whole) : {Nat.is_le(size,_) == True{} : Bool}
        Order.le_add(size,Nat.mul(rest,size)) : {Nat.is_le(size,List.length(&2,Cell,left)) == True{} : Bool}}
      +later = List.drop(&2,Cell,left,size)
      +lines = LineChainValues.group_lines(~Cell,~Value,~Final,~step,~turn_value,U32.to_nat(inner),0n,U32.to_nat(inner),List.take(&2,Cell,left,size),starts)
      +left_rest = block_folds(~Cell,~Value,~Final,~step,~turn_value,rest,length,inner,later,starts)
      +right_all = block_folds(~Cell,~Value,~Final,~step,~turn_value,second,length,inner,right,starts)
      %Equal.sym(List<&2,Cell>,List.take(&2,Cell,List.append(&2,Cell,left,right),size),List.take(&2,Cell,left,size),take_append(~Cell,left,size,right,inside)) :
        {List.append(&2,Final,List.append(&2,Final,lines,left_rest),right_all) ==
          List.append(&2,Final,LineChainValues.group_lines(~Cell,~Value,~Final,~step,~turn_value,U32.to_nat(inner),0n,U32.to_nat(inner),_,starts),
            block_folds(~Cell,~Value,~Final,~step,~turn_value,Nat.add(rest,second),length,inner,List.drop(&2,Cell,List.append(&2,Cell,left,right),size),starts)) : List<&2,Final>}
      %Equal.sym(List<&2,Cell>,List.drop(&2,Cell,List.append(&2,Cell,left,right),size),List.append(&2,Cell,later,right),drop_append(~Cell,left,size,right,inside)) :
        {List.append(&2,Final,List.append(&2,Final,lines,left_rest),right_all) ==
          List.append(&2,Final,lines,block_folds(~Cell,~Value,~Final,~step,~turn_value,Nat.add(rest,second),length,inner,_,starts)) : List<&2,Final>}
      %Equal.sym(List<&2,Final>,List.append(&2,Final,List.append(&2,Final,lines,left_rest),right_all),List.append(&2,Final,lines,List.append(&2,Final,left_rest,right_all)),
          ListReadback.append_associative(~Final,lines,left_rest,right_all)) :
        {_ == List.append(&2,Final,lines,block_folds(~Cell,~Value,~Final,~step,~turn_value,Nat.add(rest,second),length,inner,List.append(&2,Cell,later,right),starts)) : List<&2,Final>}
      Equal.cong(List<&2,Final>,List<&2,Final>,others => List.append(&2,Final,lines,others),List.append(&2,Final,left_rest,right_all),
        block_folds(~Cell,~Value,~Final,~step,~turn_value,Nat.add(rest,second),length,inner,List.append(&2,Cell,later,right),starts),
        block_append(~Cell,~Value,~Final,~step,~turn_value,rest,second,length,inner,later,right,starts,
          Equal.trans(Nat,List.length(&2,Cell,later),Nat.sub(List.length(&2,Cell,left),size),Nat.mul(rest,size),LineChainValues.drop_length(~Cell,left,size),
            %Equal.sym(Nat,List.length(&2,Cell,left),Nat.add(size,Nat.mul(rest,size)),whole) : {Nat.sub(_,size) == Nat.mul(rest,size) : Nat}
            Order.subtract_shift(size,Nat.mul(rest,size)))))

# With one line a block from one start, the blocks are the turned chunks.
def single_blocks(~Cell: Data,~Value: Data,~Final: Data,~step: Value -> Cell -> Value,~turn_value: Value -> Final,groups: Nat,+length: U32,+list: List<&2,Cell>,+initial: Value) ->
  {block_folds(~Cell,~Value,~Final,~step,~turn_value,groups,length,1,list,[initial]) == LineChainValues.turned_values(~Value,~Final,~turn_value,chunk_folds(~Cell,~Value,~step,groups,U32.to_nat((length * 1 : U32)),list,initial)) : List<&2,Final>}:
  match groups:
    case 0n: {==}
    case 1n+ +rest:
      Equal.cong(List<&2,Final>,List<&2,Final>,others => turn_value(LineChainValues.taken_list(~Cell,~Value,~step,List.take(&2,Cell,list,U32.to_nat((length * 1 : U32))),0n,1n,initial)) <> others,
        block_folds(~Cell,~Value,~Final,~step,~turn_value,rest,length,1,List.drop(&2,Cell,list,U32.to_nat((length * 1 : U32))),[initial]),
        LineChainValues.turned_values(~Value,~Final,~turn_value,chunk_folds(~Cell,~Value,~step,rest,U32.to_nat((length * 1 : U32)),List.drop(&2,Cell,list,U32.to_nat((length * 1 : U32))),initial)),
        single_blocks(~Cell,~Value,~Final,~step,~turn_value,rest,length,List.drop(&2,Cell,list,U32.to_nat((length * 1 : U32))),initial))

# ---- Tasks of whole blocks ----

# The blocks of count cells of whole blocks of length * inner cells.
def blocks_of(+length: U32,+inner: U32,+count: U32) -> Nat:
  U32.to_nat((count / U32.max((length * inner : U32),1) : U32))

# The lines of count cells of whole blocks, inner lines a block.
def line_count(+length: U32,+inner: U32,+count: U32) -> Nat:
  U32.to_nat((count / U32.max((length * inner : U32),1) * inner : U32))

# A task of count cells from first over a tree that a walk can count: its
# cells are whole blocks, they and its lines fit, and its first line begins a
# block.
def WholeBlocks(~Model: Data,~Cell: Data,~cell: Model -> U32 -> Cell,+length: U32,+inner: U32,+cap: U32,+first: U32,+count: U32,+tree: Partition.Tree<&2,Model,Unit>) -> Data:
  KernelGemm.Both<{List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,tree)) == Nat.mul(blocks_of(length,inner,count),U32.to_nat((length * inner : U32))) : Nat},
    KernelGemm.Both<{Nat.is_le(Nat.add(Nat.double(List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,tree))),4n),U32.to_nat(cap)) == True{} : Bool},
      KernelGemm.Both<{line_count(length,inner,count) == Nat.mul(blocks_of(length,inner,count),U32.to_nat(inner)) : Nat},
        KernelGemm.Both<{Nat.is_lt(line_count(length,inner,count),U32.to_nat(cap)) == True{} : Bool},
          {line_count(length,inner,first) == Nat.mul(blocks_of(length,inner,first),U32.to_nat(inner)) : Nat}>>>>

# Every task of a layout over a tree, as tasks_model cuts it, is whole.
def TasksBlocks(~Model: Data,~Cell: Data,~cell: Model -> U32 -> Cell,+length: U32,+inner: U32,+cap: U32,layout: Pieces.Ranges,tree: Partition.Tree<&2,Model,Unit>) -> Data:
  match layout:
    case Pieces.Range{+first,+count}: WholeBlocks(~Model,~Cell,~cell,length,inner,cap,first,count,tree)
    case Pieces.Split{+left,+right}:
      match tree:
        case Partition.Leaf{+model,+first,+count,+details}:
          WholeBlocks(~Model,~Cell,~cell,length,inner,cap,Pieces.first(Pieces.Split{left,right}),(Pieces.last(Pieces.Split{left,right}) - Pieces.first(Pieces.Split{left,right}) : U32),
            Partition.Leaf{model,first,count,details})
        case Partition.Fork{+low,+high}:
          KernelGemm.Both<TasksBlocks(~Model,~Cell,~cell,length,inner,cap,left,low),TasksBlocks(~Model,~Cell,~cell,length,inner,cap,right,high)>

# The blocks of every task of a layout over a tree.
def task_blocks(~Model: Data,+length: U32,+inner: U32,layout: Pieces.Ranges,tree: Partition.Tree<&2,Model,Unit>) -> Nat:
  match layout:
    case Pieces.Range{+first,+count}: blocks_of(length,inner,count)
    case Pieces.Split{+left,+right}:
      match tree:
        case Partition.Leaf{+model,+first,+count,+details}: blocks_of(length,inner,(Pieces.last(Pieces.Split{left,right}) - Pieces.first(Pieces.Split{left,right}) : U32))
        case Partition.Fork{+low,+high}: Nat.add(task_blocks(~Model,length,inner,left,low),task_blocks(~Model,length,inner,right,high))

# The cells of whole tasks are their blocks times the block.
def blocks_length(~Model: Data,~Cell: Data,~cell: Model -> U32 -> Cell,+length: U32,+inner: U32,+cap: U32,layout: Pieces.Ranges,tree: Partition.Tree<&2,Model,Unit>,
  +whole: TasksBlocks(~Model,~Cell,~cell,length,inner,cap,layout,tree)) ->
  {List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,tree)) == Nat.mul(task_blocks(~Model,length,inner,layout,tree),U32.to_nat((length * inner : U32))) : Nat}:
  match layout:
    case Pieces.Range{+first,+count}:
      KernelGemm.Both{+equation,+others} = whole
      equation
    case Pieces.Split{+left,+right}:
      match tree:
        case Partition.Leaf{+model,+first,+count,+details}:
          KernelGemm.Both{+equation,+others} = whole
          equation
        case Partition.Fork{+low,+high}:
          KernelGemm.Both{+left_whole,+right_whole} = whole
          +low_blocks = task_blocks(~Model,length,inner,left,low)
          +high_blocks = task_blocks(~Model,length,inner,right,high)
          +size = U32.to_nat((length * inner : U32))
          %Equal.sym(Nat,List.length(&2,Cell,List.append(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,low),LineChainValues.stream(~Model,~Cell,~cell,high))),
              Nat.add(List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,low)),List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,high))),
              ListReadback.append_length(~Cell,LineChainValues.stream(~Model,~Cell,~cell,low),LineChainValues.stream(~Model,~Cell,~cell,high))) :
            {_ == Nat.mul(Nat.add(low_blocks,high_blocks),size) : Nat}
          %Equal.sym(Nat,List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,low)),Nat.mul(low_blocks,size),blocks_length(~Model,~Cell,~cell,length,inner,cap,left,low,left_whole)) :
            {Nat.add(_,List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,high))) == Nat.mul(Nat.add(low_blocks,high_blocks),size) : Nat}
          %Equal.sym(Nat,List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,high)),Nat.mul(high_blocks,size),blocks_length(~Model,~Cell,~cell,length,inner,cap,right,high,right_whole)) :
            {Nat.add(Nat.mul(low_blocks,size),_) == Nat.mul(Nat.add(low_blocks,high_blocks),size) : Nat}
          Equal.sym(Nat,Nat.mul(Nat.add(low_blocks,high_blocks),size),Nat.add(Nat.mul(low_blocks,size),Nat.mul(high_blocks,size)),RowMajor.distrib(low_blocks,high_blocks,size))

# The blocks of count cells of groups whole blocks.
def blocks_exact(+length: U32,+inner: U32,+count: U32,+groups: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat((length * inner : U32))) == True{} : Bool},+divisor_room: {Nat.is_le(Nat.double(U32.to_nat((length * inner : U32))),U32.to_nat(cap)) == True{} : Bool},
  +count_value: {U32.to_nat(count) == Nat.mul(groups,U32.to_nat((length * inner : U32))) : Nat}) ->
  {blocks_of(length,inner,count) == groups : Nat}:
  +word = (length * inner : U32)
  Equal.trans(Nat,blocks_of(length,inner,count),U32.to_nat((count / word : U32)),groups,
    Equal.cong(U32,Nat,value => U32.to_nat((count / value : U32)),U32.max(word,1),word,GridPieces.maximum_kept(word,1,Order.positive_one(U32.to_nat((length * inner : U32)),positive))),
    Equal.trans(Nat,U32.to_nat((count / word : U32)),Nat.div(U32.to_nat(count),U32.to_nat((length * inner : U32))),groups,
      U32Division.divide(count,word,cap,cap_ok,divisor_room,positive),
      Equal.trans(Nat,Nat.div(U32.to_nat(count),U32.to_nat((length * inner : U32))),Nat.div(Nat.mul(groups,U32.to_nat((length * inner : U32))),U32.to_nat((length * inner : U32))),groups,
        Equal.cong(Nat,Nat,value => Nat.div(value,U32.to_nat((length * inner : U32))),U32.to_nat(count),Nat.mul(groups,U32.to_nat((length * inner : U32))),count_value),
        Division.exact_quotient(groups,U32.to_nat((length * inner : U32)),positive))))

# The lines of count cells of groups whole blocks.
def lines_exact(+length: U32,+inner: U32,+count: U32,+groups: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat((length * inner : U32))) == True{} : Bool},+divisor_room: {Nat.is_le(Nat.double(U32.to_nat((length * inner : U32))),U32.to_nat(cap)) == True{} : Bool},
  +count_value: {U32.to_nat(count) == Nat.mul(groups,U32.to_nat((length * inner : U32))) : Nat},
  +inner_le: {Nat.is_le(U32.to_nat(inner),U32.to_nat((length * inner : U32))) == True{} : Bool},+count_room: {Nat.is_le(U32.to_nat(count),U32.to_nat(cap)) == True{} : Bool}) ->
  {line_count(length,inner,count) == Nat.mul(groups,U32.to_nat(inner)) : Nat}:
  +bounded = Order.transitive(Nat.mul(groups,U32.to_nat(inner)),Nat.mul(groups,U32.to_nat((length * inner : U32))),U32.to_nat(cap),
    NatDivision.multiply_left_le(groups,U32.to_nat(inner),U32.to_nat((length * inner : U32)),inner_le),
    {%count_value : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} count_room : {Nat.is_le(Nat.mul(groups,U32.to_nat((length * inner : U32))),U32.to_nat(cap)) == True{} : Bool}})
  Arithmetic.product_exact(cap,cap_ok,(count / U32.max((length * inner : U32),1) : U32),inner,groups,U32.to_nat(inner),
    blocks_exact(length,inner,count,groups,cap,cap_ok,positive,divisor_room,count_value),{==},bounded)

# A task layout whose ranges are runs of whole blocks and stream over a tree
# holds whole tasks.
def tasks_blocks(~Model: Data,~Cell: Data,~cell: Model -> U32 -> Cell,+length: U32,+inner: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat((length * inner : U32))) == True{} : Bool},+divisor_room: {Nat.is_le(Nat.double(U32.to_nat((length * inner : U32))),U32.to_nat(cap)) == True{} : Bool},
  +inner_le: {Nat.is_le(U32.to_nat(inner),U32.to_nat((length * inner : U32))) == True{} : Bool},+bound: Nat,
  +bound_room: {Nat.is_le(Nat.add(Nat.double(bound),4n),U32.to_nat(cap)) == True{} : Bool},+bound_below: {Nat.is_lt(bound,U32.to_nat(cap)) == True{} : Bool},
  layout: Pieces.Ranges,+tree: Partition.Tree<&2,Model,Unit>,+run: Runs.Run(layout,U32.to_nat((length * inner : U32)),bound),+streams: Streams(~Model,~Cell,~cell,layout,tree)) ->
  TasksBlocks(~Model,~Cell,~cell,length,inner,cap,layout,tree):
  match layout:
    case Pieces.Range{+first,+count}:
      Runs.Fits{+low_units,+high_units,+low_exact,+high_exact,+less,+inside} = run
      +size = U32.to_nat((length * inner : U32))
      +units = Nat.sub(high_units,low_units)
      +count_value = Equal.trans(Nat,U32.to_nat(count),Nat.sub(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(first)),Nat.mul(units,size),
        Equal.sym(Nat,Nat.sub(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(first)),U32.to_nat(count),Order.subtract_shift(U32.to_nat(first),U32.to_nat(count))),
        Equal.trans(Nat,Nat.sub(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(first)),Nat.sub(Nat.mul(high_units,size),Nat.mul(low_units,size)),Nat.mul(units,size),
          Equal.trans(Nat,Nat.sub(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(first)),Nat.sub(Nat.mul(high_units,size),U32.to_nat(first)),Nat.sub(Nat.mul(high_units,size),Nat.mul(low_units,size)),
            Equal.cong(Nat,Nat,value => Nat.sub(value,U32.to_nat(first)),Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(high_units,size),high_exact),
            Equal.cong(Nat,Nat,value => Nat.sub(Nat.mul(high_units,size),value),U32.to_nat(first),Nat.mul(low_units,size),low_exact)),
          RowMajor.subtract_products(high_units,low_units,size)))
      +blocks = blocks_exact(length,inner,count,units,cap,cap_ok,positive,divisor_room,count_value)
      +bound_le = LineChainValues.lt_le(bound,U32.to_nat(cap),bound_below)
      +count_le = Order.transitive(U32.to_nat(count),Nat.add(U32.to_nat(first),U32.to_nat(count)),bound,Order.right_le(U32.to_nat(first),U32.to_nat(count)),inside)
      +first_le = Order.transitive(U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),bound,Order.le_add(U32.to_nat(first),U32.to_nat(count)),inside)
      +lines = lines_exact(length,inner,count,units,cap,cap_ok,positive,divisor_room,count_value,inner_le,Order.transitive(U32.to_nat(count),bound,U32.to_nat(cap),count_le,bound_le))
      +first_lines = Equal.trans(Nat,line_count(length,inner,first),Nat.mul(low_units,U32.to_nat(inner)),Nat.mul(blocks_of(length,inner,first),U32.to_nat(inner)),
        lines_exact(length,inner,first,low_units,cap,cap_ok,positive,divisor_room,low_exact,inner_le,Order.transitive(U32.to_nat(first),bound,U32.to_nat(cap),first_le,bound_le)),
        Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(inner)),low_units,blocks_of(length,inner,first),
          Equal.sym(Nat,blocks_of(length,inner,first),low_units,blocks_exact(length,inner,first,low_units,cap,cap_ok,positive,divisor_room,low_exact))))
      +length_le = LineChainValues.le_by(List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,tree)),U32.to_nat(count),bound,streams,count_le)
      +lines_le = Order.transitive(Nat.mul(units,U32.to_nat(inner)),Nat.mul(units,size),bound,NatDivision.multiply_left_le(units,U32.to_nat(inner),size,inner_le),
        LineChainValues.le_by(Nat.mul(units,size),U32.to_nat(count),bound,Equal.sym(Nat,U32.to_nat(count),Nat.mul(units,size),count_value),count_le))
      +units_blocks = Equal.sym(Nat,blocks_of(length,inner,count),units,blocks)
      KernelGemm.Both{
        Equal.trans(Nat,List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,tree)),U32.to_nat(count),Nat.mul(blocks_of(length,inner,count),size),streams,
          Equal.trans(Nat,U32.to_nat(count),Nat.mul(units,size),Nat.mul(blocks_of(length,inner,count),size),count_value,
            Equal.cong(Nat,Nat,value => Nat.mul(value,size),units,blocks_of(length,inner,count),units_blocks))),
        KernelGemm.Both{
          Order.transitive(Nat.add(Nat.double(List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,tree))),4n),Nat.add(Nat.double(bound),4n),U32.to_nat(cap),
            Order.add_le(Nat.double(List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,tree))),Nat.double(bound),4n,Order.double_le(List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,tree)),bound,length_le)),bound_room),
          KernelGemm.Both{
            Equal.trans(Nat,line_count(length,inner,count),Nat.mul(units,U32.to_nat(inner)),Nat.mul(blocks_of(length,inner,count),U32.to_nat(inner)),lines,
              Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(inner)),units,blocks_of(length,inner,count),units_blocks)),
            KernelGemm.Both{
              LineChainValues.lt_by(line_count(length,inner,count),Nat.mul(units,U32.to_nat(inner)),U32.to_nat(cap),lines,Order.le_lt_trans(Nat.mul(units,U32.to_nat(inner)),bound,U32.to_nat(cap),lines_le,bound_below)),
              first_lines}}}}
    case Pieces.Split{+left,+right}:
      match tree:
        case Partition.Leaf{+model,+first,+count,+details}:
          Empty.absurd(TasksBlocks(~Model,~Cell,~cell,length,inner,cap,Pieces.Split{left,right},Partition.Leaf{model,first,count,details}),streams)
        case Partition.Fork{+low,+high}:
          Held.Both{+left_run,+run_rest} = run
          Held.Both{+right_run,+glue} = run_rest
          KernelGemm.Both{+left_streams,+right_streams} = streams
          KernelGemm.Both{tasks_blocks(~Model,~Cell,~cell,length,inner,cap,cap_ok,positive,divisor_room,inner_le,bound,bound_room,bound_below,left,low,left_run,left_streams),
            tasks_blocks(~Model,~Cell,~cell,length,inner,cap,cap_ok,positive,divisor_room,inner_le,bound,bound_room,bound_below,right,high,right_run,right_streams)}

# Lines from a block start begin from no turn of the contexts.
def first_residue(+blocks: Nat,+inner: U32,+laps: Nat,+kinds: Nat,+positive: {Nat.is_lt(0n,kinds) == True{} : Bool},+inner_exact: {U32.to_nat(inner) == Nat.mul(laps,kinds) : Nat}) ->
  {Nat.mod(Nat.mul(blocks,U32.to_nat(inner)),kinds) == 0n : Nat}:
  %Equal.sym(Nat,U32.to_nat(inner),Nat.mul(laps,kinds),inner_exact) : {Nat.mod(Nat.mul(blocks,_),kinds) == 0n : Nat}
  %RowMajor.assoc(blocks,laps,kinds) : {Nat.mod(_,kinds) == 0n : Nat}
  %Addition.zero_right(Nat.mul(Nat.mul(blocks,laps),kinds)) : {Nat.mod(_,kinds) == 0n : Nat}
  Equal.trans(Nat,Nat.mod(Nat.add(Nat.mul(Nat.mul(blocks,laps),kinds),0n),kinds),Nat.mod(0n,kinds),0n,NatDivision.remainder_shift(0n,kinds,Nat.mul(blocks,laps),positive),zero_remainder(kinds,positive))

# Rooms for the lines of blocks of length * inner cells.
type BlockRooms<-length: U32,-inner: U32,-cap: U32,-kinds: Nat> is Data:
  BlockRooms{kinds_room: {Nat.is_le(Nat.double(kinds),U32.to_nat(cap)) == True{} : Bool},room: {Nat.is_le(Nat.double(U32.to_nat(inner)),U32.to_nat(cap)) == True{} : Bool},
    positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},block_room: {Nat.is_le(Nat.add(U32.to_nat((length * inner : U32)),U32.to_nat(inner)),U32.to_nat(cap)) == True{} : Bool},
    cells_room: {Nat.is_le(Nat.double(U32.to_nat((length * inner : U32))),U32.to_nat(cap)) == True{} : Bool},filled: {Nat.is_lt(0n,U32.to_nat((length * inner : U32))) == True{} : Bool}}

def half_room(+size: Nat,+cap: Nat,+room: {Nat.is_le(Nat.add(Nat.double(size),4n),cap) == True{} : Bool}) -> {Nat.is_le(Nat.add(size,2n),cap) == True{} : Bool}:
  +doubled = Held.nat_le_right_moved(size,Nat.double(size),Nat.add(size,size),GridRanges.double_add(size),Order.le_add(size,size))
  Order.transitive(Nat.add(size,2n),Nat.add(Nat.double(size),4n),cap,
    Order.transitive(Nat.add(size,2n),Nat.add(Nat.double(size),2n),Nat.add(Nat.double(size),4n),Order.add_le(size,Nat.double(size),2n,doubled),Order.offset_le(Nat.double(size),2n,4n,{==})),
    room)

# One task of whole blocks, its lines begun from the contexts in turn, turns
# the lines of the blocks of its cells.
def task_block_folds(~Model: Data,~Element: Data,~Start: Data,~Value: Data,~Final: Data,~cell: Model -> U32 -> Element,~operation: Value -> Element -> Value,~begin_value: Start -> Value,~turn_value: Value -> Final,+length: U32,+inner: U32,+contexts: List<&2,Start>,+first: U32,+count: U32,+tree: Partition.Tree<&2,Model,Unit>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+turns: Turns<Start,contexts,inner>,
  +rooms: BlockRooms<length,inner,cap,List.length(&2,Start,contexts)>,+whole: WholeBlocks(~Model,~Element,~cell,length,inner,cap,first,count,tree)) ->
  {LineChainValues.task_finals(~Model,~Start,~Value,~Final,LineChainValues.task_model(~Model,~Start,~Value,~Final,~begin_value,~LineChainValues.strided_model(~Model,~Element,~Value,~cell,~operation),~turn_value,LineChain.Lines{contexts,LineChain.Line{length,inner}},first,count,tree)) == block_folds(~Element,~Value,~Final,~operation,~turn_value,blocks_of(length,inner,count),length,inner,LineChainValues.stream(~Model,~Element,~cell,tree),block_starts(~Start,~Value,~begin_value,contexts,inner)) : List<&2,Final>}:
  Turns{+laps,+kinds_positive,+inner_exact} = turns
  BlockRooms{+kinds_room,+room,+positive,+block_room,+cells_room,+filled} = rooms
  KernelGemm.Both{+equation,+facts} = whole
  KernelGemm.Both{+tree_room,+counted} = facts
  KernelGemm.Both{+line_total,+bounded} = counted
  KernelGemm.Both{+lines_room,+first_lines} = bounded
  +blocks = blocks_of(length,inner,count)
  +kinds = List.length(&2,Start,contexts)
  +line = {LineChain.Line{length,inner} : LineChain.Line}
  +residue_zero = {%Equal.sym(Nat,line_count(length,inner,first),Nat.mul(blocks_of(length,inner,first),U32.to_nat(inner)),first_lines) : {Nat.mod(_,kinds) == 0n : Nat}
      first_residue(blocks_of(length,inner,first),inner,laps,kinds,kinds_positive,inner_exact) :
    {Nat.mod(line_count(length,inner,first),kinds) == 0n : Nat}}
  +upcoming = Equal.trans(List<&2,Start>,LineChain.upcoming_of(~Start,contexts,line,first),List.drop(&2,Start,contexts,Nat.mod(line_count(length,inner,first),kinds)),contexts,
    LineChainValues.upcoming_residue(~Start,contexts,line,first,cap,cap_ok,kinds_positive,kinds_room),
    %Equal.sym(Nat,Nat.mod(line_count(length,inner,first),kinds),0n,residue_zero) : {List.drop(&2,Start,contexts,_) == contexts : List<&2,Start>}
    LineChainValues.drop_zero(~Start,contexts))
  +stream_room = half_room(List.length(&2,Element,LineChainValues.stream(~Model,~Element,~cell,tree)),U32.to_nat(cap),tree_room)
  +starts = block_starts(~Start,~Value,~begin_value,contexts,inner)
  Equal.trans(List<&2,Final>,LineChainValues.task_finals(~Model,~Start,~Value,~Final,LineChainValues.task_model(~Model,~Start,~Value,~Final,~begin_value,~LineChainValues.strided_model(~Model,~Element,~Value,~cell,~operation),~turn_value,LineChain.Lines{contexts,LineChain.Line{length,inner}},first,count,tree)),
    LineChainValues.grouped(~Element,~Start,~Value,~Final,~operation,~begin_value,~turn_value,contexts,line,blocks,LineChainValues.stream(~Model,~Element,~cell,tree),LineChain.upcoming_of(~Start,contexts,line,first)),block_folds(~Element,~Value,~Final,~operation,~turn_value,blocks,length,inner,LineChainValues.stream(~Model,~Element,~cell,tree),starts),
    LineChainValues.task_lines(~Model,~Element,~Start,~Value,~Final,~cell,~operation,~begin_value,~turn_value,contexts,line,first,tree,blocks,cap,cap_ok,room,positive,block_room,filled,cells_room,stream_room,equation),
    %Equal.sym(List<&2,Start>,LineChain.upcoming_of(~Start,contexts,line,first),contexts,upcoming) :
      {LineChainValues.grouped(~Element,~Start,~Value,~Final,~operation,~begin_value,~turn_value,contexts,line,blocks,LineChainValues.stream(~Model,~Element,~cell,tree),_) == block_folds(~Element,~Value,~Final,~operation,~turn_value,blocks,length,inner,LineChainValues.stream(~Model,~Element,~cell,tree),starts) : List<&2,Final>}
    grouped_blocks(~Element,~Start,~Value,~Final,~operation,~begin_value,~turn_value,blocks,length,inner,LineChainValues.stream(~Model,~Element,~cell,tree),contexts,drawn_round(~Start,contexts,inner,laps,kinds_positive,inner_exact)))

# Whole tasks, their lines begun from the contexts in turn, turn the lines of
# the blocks of all cells in order.
def tasks_block_folds(~Model: Data,~Element: Data,~Start: Data,~Value: Data,~Final: Data,~cell: Model -> U32 -> Element,~operation: Value -> Element -> Value,~begin_value: Start -> Value,~turn_value: Value -> Final,+length: U32,+inner: U32,+contexts: List<&2,Start>,layout: Pieces.Ranges,tree: Partition.Tree<&2,Model,Unit>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+turns: Turns<Start,contexts,inner>,
  +rooms: BlockRooms<length,inner,cap,List.length(&2,Start,contexts)>,+whole: TasksBlocks(~Model,~Element,~cell,length,inner,cap,layout,tree)) ->
  {LineChainValues.task_finals(~Model,~Start,~Value,~Final,LineChainValues.tasks_model(~Model,~Start,~Value,~Final,~begin_value,~LineChainValues.strided_model(~Model,~Element,~Value,~cell,~operation),~turn_value,layout,LineChain.Lines{contexts,LineChain.Line{length,inner}},tree)) == block_folds(~Element,~Value,~Final,~operation,~turn_value,task_blocks(~Model,length,inner,layout,tree),length,inner,LineChainValues.stream(~Model,~Element,~cell,tree),block_starts(~Start,~Value,~begin_value,contexts,inner)) : List<&2,Final>}:
  match layout:
    case Pieces.Range{+first,+count}:
      task_block_folds(~Model,~Element,~Start,~Value,~Final,~cell,~operation,~begin_value,~turn_value,length,inner,contexts,first,count,tree,cap,cap_ok,turns,rooms,whole)
    case Pieces.Split{+left,+right}:
      match tree:
        case Partition.Leaf{+model,+first,+count,+details}:
          task_block_folds(~Model,~Element,~Start,~Value,~Final,~cell,~operation,~begin_value,~turn_value,length,inner,contexts,Pieces.first(Pieces.Split{left,right}),(Pieces.last(Pieces.Split{left,right}) - Pieces.first(Pieces.Split{left,right}) : U32),
            Partition.Leaf{model,first,count,details},cap,cap_ok,turns,rooms,whole)
        case Partition.Fork{+low,+high}:
          KernelGemm.Both{+left_whole,+right_whole} = whole
          +low_tasks = LineChainValues.tasks_model(~Model,~Start,~Value,~Final,~begin_value,~LineChainValues.strided_model(~Model,~Element,~Value,~cell,~operation),~turn_value,left,LineChain.Lines{contexts,LineChain.Line{length,inner}},low)
          +high_tasks = LineChainValues.tasks_model(~Model,~Start,~Value,~Final,~begin_value,~LineChainValues.strided_model(~Model,~Element,~Value,~cell,~operation),~turn_value,right,LineChain.Lines{contexts,LineChain.Line{length,inner}},high)
          +low_blocks = task_blocks(~Model,length,inner,left,low)
          +high_blocks = task_blocks(~Model,length,inner,right,high)
          +starts = block_starts(~Start,~Value,~begin_value,contexts,inner)
          %Equal.sym(List<&2,Final>,LineChainValues.task_finals(~Model,~Start,~Value,~Final,low_tasks),block_folds(~Element,~Value,~Final,~operation,~turn_value,low_blocks,length,inner,LineChainValues.stream(~Model,~Element,~cell,low),starts),tasks_block_folds(~Model,~Element,~Start,~Value,~Final,~cell,~operation,~begin_value,~turn_value,length,inner,contexts,left,low,cap,cap_ok,turns,rooms,left_whole)) :
            {List.append(&2,Final,_,LineChainValues.task_finals(~Model,~Start,~Value,~Final,high_tasks)) == block_folds(~Element,~Value,~Final,~operation,~turn_value,Nat.add(low_blocks,high_blocks),length,inner,List.append(&2,Element,LineChainValues.stream(~Model,~Element,~cell,low),LineChainValues.stream(~Model,~Element,~cell,high)),starts) : List<&2,Final>}
          %Equal.sym(List<&2,Final>,LineChainValues.task_finals(~Model,~Start,~Value,~Final,high_tasks),block_folds(~Element,~Value,~Final,~operation,~turn_value,high_blocks,length,inner,LineChainValues.stream(~Model,~Element,~cell,high),starts),tasks_block_folds(~Model,~Element,~Start,~Value,~Final,~cell,~operation,~begin_value,~turn_value,length,inner,contexts,right,high,cap,cap_ok,turns,rooms,right_whole)) :
            {List.append(&2,Final,block_folds(~Element,~Value,~Final,~operation,~turn_value,low_blocks,length,inner,LineChainValues.stream(~Model,~Element,~cell,low),starts),_) == block_folds(~Element,~Value,~Final,~operation,~turn_value,Nat.add(low_blocks,high_blocks),length,inner,List.append(&2,Element,LineChainValues.stream(~Model,~Element,~cell,low),LineChainValues.stream(~Model,~Element,~cell,high)),starts) : List<&2,Final>}
          block_append(~Element,~Value,~Final,~operation,~turn_value,low_blocks,high_blocks,length,inner,LineChainValues.stream(~Model,~Element,~cell,low),LineChainValues.stream(~Model,~Element,~cell,high),starts,blocks_length(~Model,~Element,~cell,length,inner,cap,left,low,left_whole))

# ---- Blocks of interleaved lines ----

# The contexts of whole turns fit as the inner lines of a block do.
def turns_room(~State: Data,+contexts: List<&2,State>,+inner: U32,+cap: U32,+turns: Turns<State,contexts,inner>,+positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(U32.to_nat(inner)),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(Nat.double(List.length(&2,State,contexts)),U32.to_nat(cap)) == True{} : Bool}:
  Turns{+laps,+kinds_positive,+inner_exact} = turns
  +kinds = List.length(&2,State,contexts)
  +product_positive = Held.nat_lt_moved(0n,Nat.mul(laps,kinds),U32.to_nat(inner),Equal.sym(Nat,U32.to_nat(inner),Nat.mul(laps,kinds),inner_exact),positive)
  +covered = Held.nat_le_right_moved(kinds,Nat.mul(laps,kinds),Nat.mul(kinds,laps),RowMajor.multiply_commutative(laps,kinds),RowMajor.product_covers(kinds,laps,RowMajor.left_factor_positive(laps,kinds,product_positive)))
  +kinds_le = Held.nat_le_right_moved(kinds,U32.to_nat(inner),Nat.mul(laps,kinds),inner_exact,covered)
  Order.transitive(Nat.double(kinds),Nat.double(U32.to_nat(inner)),U32.to_nat(cap),Order.double_le(kinds,U32.to_nat(inner),kinds_le),room)

# The lines a chain carries across the blocks of a tree, every block length
# lines of inner cells begun from the contexts in turn, turn into the finals of
# the blocks of the cells of the regrouped tree, when every task holds whole
# blocks.
def carried_lines(~Block: Type,~Model: Data,~Element: Data,~Start: Data,~State: Type,~Value: Data,~Final: Data,~represent: Model -> Block,~cell: Model -> U32 -> Element,~operation: Value -> Element -> Value,~begin: Start -> State,~begin_value: Start -> Value,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~turn: State -> Final,~turn_value: Value -> Final,~Fill: Data,~vacant: Fill -> Block,~vacant_model: Fill -> Model,~vacant_same: @+fill: Fill -> {vacant(fill) == represent(vacant_model(fill)) : Block},~chain_same: LineChainValues.ChainSame(~Block,~Model,~represent,~Start,~State,~Value,~Final,~begin,~begin_value,~fold,~LineChainValues.strided_model(~Model,~Element,~Value,~cell,~operation),~turn,~turn_value,~Fill,~vacant,~vacant_model),+length: U32,+inner: U32,+contexts: List<&2,Start>,+group: U32,+bytes: F32,+operations: F32,+grid: Grid.Grid,+fill: Fill,+workers: U32,+tree: Partition.Tree<&2,Model,Unit>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +turns: Turns<Start,contexts,inner>,+rooms: BlockRooms<length,inner,cap,List.length(&2,Start,contexts)>,+whole: TasksBlocks(~Model,~Element,~cell,length,inner,cap,LineChainValues.decided_tasks(Tasks.decide_over(Pieces.ranges_in(grid,group),Grid.count_of(grid),workers,LineChain.cells_of(LineChain.Line{length,inner}),bytes,operations)),Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(grid,group),fill,tree))) ->
  {Pair.snd(Partition.Tree<&1,Block,Unit>,List<&2,Final>,LineChain.folds(~Block,~Fill,~Start,~State,~Final,~begin,~fold,~turn,~vacant,contexts,LineChain.Line{length,inner},group,bytes,operations,grid,fill,workers,Regrouped.tree_of(~Block,~Model,~represent,tree))) ==
    block_folds(~Element,~Value,~Final,~operation,~turn_value,task_blocks(~Model,length,inner,LineChainValues.decided_tasks(Tasks.decide_over(Pieces.ranges_in(grid,group),Grid.count_of(grid),workers,LineChain.cells_of(LineChain.Line{length,inner}),bytes,operations)),Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(grid,group),fill,tree)),length,inner,LineChainValues.stream(~Model,~Element,~cell,Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(grid,group),fill,tree)),block_starts(~Start,~Value,~begin_value,contexts,inner)) : List<&2,Final>}:
  +decided = LineChainValues.decided_tasks(Tasks.decide_over(Pieces.ranges_in(grid,group),Grid.count_of(grid),workers,LineChain.cells_of(LineChain.Line{length,inner}),bytes,operations))
  +regrouped = Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(grid,group),fill,tree)
  +work = LineChainValues.tasks_model(~Model,~Start,~Value,~Final,~begin_value,~LineChainValues.strided_model(~Model,~Element,~Value,~cell,~operation),~turn_value,decided,LineChain.Lines{contexts,LineChain.Line{length,inner}},regrouped)
  +model = LineChainValues.folds_model(~Model,~Start,~Value,~Final,~begin_value,~LineChainValues.strided_model(~Model,~Element,~Value,~cell,~operation),~turn_value,~Fill,~vacant_model,contexts,LineChain.Line{length,inner},group,bytes,operations,grid,fill,workers,tree)
  +finals = block_folds(~Element,~Value,~Final,~operation,~turn_value,task_blocks(~Model,length,inner,decided,regrouped),length,inner,LineChainValues.stream(~Model,~Element,~cell,regrouped),block_starts(~Start,~Value,~begin_value,contexts,inner))
  Equal.trans(List<&2,Final>,Pair.snd(Partition.Tree<&1,Block,Unit>,List<&2,Final>,LineChain.folds(~Block,~Fill,~Start,~State,~Final,~begin,~fold,~turn,~vacant,contexts,LineChain.Line{length,inner},group,bytes,operations,grid,fill,workers,Regrouped.tree_of(~Block,~Model,~represent,tree))),LineChainValues.unzipped_finals(~Model,~Final,model),finals,
    Equal.cong(Partition.Tree<&1,Block,Unit> & List<&2,Final>,List<&2,Final>,pair => Pair.snd(Partition.Tree<&1,Block,Unit>,List<&2,Final>,pair),LineChain.folds(~Block,~Fill,~Start,~State,~Final,~begin,~fold,~turn,~vacant,contexts,LineChain.Line{length,inner},group,bytes,operations,grid,fill,workers,Regrouped.tree_of(~Block,~Model,~represent,tree)),
      LineChainValues.unzipped_pair(~Block,~Model,~represent,~Final,model),
      chain_same(contexts,LineChain.Line{length,inner},group,bytes,operations,grid,fill,workers,tree)),
    Equal.trans(List<&2,Final>,LineChainValues.unzipped_finals(~Model,~Final,model),List.append(&2,Final,LineChainValues.task_finals(~Model,~Start,~Value,~Final,work),[]),finals,
      LineChainValues.unzipped_in_order(~Model,~Start,~Value,~Final,work,[]),
      Equal.trans(List<&2,Final>,List.append(&2,Final,LineChainValues.task_finals(~Model,~Start,~Value,~Final,work),[]),LineChainValues.task_finals(~Model,~Start,~Value,~Final,work),finals,
        ListReadback.append_empty(~Final,LineChainValues.task_finals(~Model,~Start,~Value,~Final,work)),
        tasks_block_folds(~Model,~Element,~Start,~Value,~Final,~cell,~operation,~begin_value,~turn_value,length,inner,contexts,decided,regrouped,cap,cap_ok,turns,rooms,whole))))

# The tasks of a grid hold whole blocks when the fold of the ranges they are
# decided over is a run of whole blocks.
def lined_blocks(~Model: Data,~Cell: Data,~cell: Model -> U32 -> Cell,~Fill: Data,~vacant: Fill -> Model,+limits: Machine.Limits,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,
  +group: U32,+workers: U32,+length: U32,+inner: U32,+bytes: F32,+operations: F32,+bound: Nat,+fill: Fill,+tree: Partition.Tree<&2,Model,Unit>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +low: Nat,+high: Nat,+group_run: Runs.GroupRun(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),task_limit(count,U32.max(workers,1),(length * inner : U32),Tasks.size(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group)))),low,high,U32.to_nat((length * inner : U32)),bound),
  +positive: {Nat.is_lt(0n,U32.to_nat((length * inner : U32))) == True{} : Bool},+divisor_room: {Nat.is_le(Nat.double(U32.to_nat((length * inner : U32))),U32.to_nat(cap)) == True{} : Bool},+inner_le: {Nat.is_le(U32.to_nat(inner),U32.to_nat((length * inner : U32))) == True{} : Bool},
  +bound_limits: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+bound_room: {Nat.is_le(Nat.add(Nat.double(bound),4n),U32.to_nat(cap)) == True{} : Bool},
  +bound_below: {Nat.is_lt(bound,U32.to_nat(cap)) == True{} : Bool},
  +stream_room: {Nat.is_le(List.length(&2,Cell,LineChainValues.stream(~Model,~Cell,~cell,Regrouped.regrouped(~Model,~Fill,~vacant,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree))),U32.to_nat(cap)) == True{} : Bool}) ->
  TasksBlocks(~Model,~Cell,~cell,length,inner,cap,LineChainValues.decided_tasks(Tasks.decide_over(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),count,workers,(length * inner : U32),bytes,operations)),Regrouped.regrouped(~Model,~Fill,~vacant,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree)):
  +worth = Tasks.worth(workers,Cost.cell_ns(bytes,operations,DefaultCost.configuration()),DefaultCost.configuration(),Tasks.no_extra(),count,Tasks.size(Tasks.tasks_over(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),count,workers,(length * inner : U32))))
  +run = Runs.planned_run(limits,worth,Tasks.tasks_over(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),count,workers,(length * inner : U32)),U32.to_nat((length * inner : U32)),bound,{%Equal.sym(Pieces.Ranges,Tasks.tasks_sized(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),count,U32.max(workers,1),(length * inner : U32),Tasks.size(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group))),Tasks.group_layout(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),task_limit(count,U32.max(workers,1),(length * inner : U32),Tasks.size(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group))))),sized_tasks(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),count,U32.max(workers,1),(length * inner : U32),Tasks.size(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group)))) : Runs.Run(_,U32.to_nat((length * inner : U32)),bound) Runs.layout_of(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),task_limit(count,U32.max(workers,1),(length * inner : U32),Tasks.size(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group)))),low,high,U32.to_nat((length * inner : U32)),bound,group_run) : Runs.Run(Tasks.tasks_over(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),count,workers,(length * inner : U32)),U32.to_nat((length * inner : U32)),bound)},bound_limits)
  +folded = folded_streams(~Model,~Cell,~cell,~Fill,~vacant,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),task_limit(count,U32.max(workers,1),(length * inner : U32),Tasks.size(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group))),fill,Regrouped.queued(~Model,tree,[]),cap,cap_ok,stream_room)
  +streams = {%Equal.sym(Pieces.Ranges,Tasks.tasks_sized(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),count,U32.max(workers,1),(length * inner : U32),Tasks.size(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group))),Tasks.group_layout(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),task_limit(count,U32.max(workers,1),(length * inner : U32),Tasks.size(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group))))),sized_tasks(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),count,U32.max(workers,1),(length * inner : U32),Tasks.size(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group)))) : Streams(~Model,~Cell,~cell,Runs.decision_tasks(Tasks.planned(worth,_)),Regrouped.regrouped(~Model,~Fill,~vacant,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree))
    planned_streams(~Model,~Cell,~cell,limits,worth,Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),task_limit(count,U32.max(workers,1),(length * inner : U32),Tasks.size(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group)))),Regrouped.regrouped(~Model,~Fill,~vacant,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree),low,high,U32.to_nat((length * inner : U32)),bound,group_run,folded,bound_limits) : Streams(~Model,~Cell,~cell,Runs.decision_tasks(Tasks.planned(worth,Tasks.tasks_over(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),count,workers,(length * inner : U32)))),Regrouped.regrouped(~Model,~Fill,~vacant,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree))}
  tasks_blocks(~Model,~Cell,~cell,length,inner,cap,cap_ok,positive,divisor_room,inner_le,bound,bound_room,bound_below,LineChainValues.decided_tasks(Tasks.decide_over(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),count,workers,(length * inner : U32),bytes,operations)),Regrouped.regrouped(~Model,~Fill,~vacant,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree),run,streams)

# Blocks of length lines of inner cells, last in the cells of a tensor, their
# lines begun from the contexts in turn, fold into the block folds of the
# tensor's values in order, when the fold of the ranges the tasks are decided
# over is a run of whole blocks that lists the pieces of the grid.
def lined_states(~Block: Type,~Model: Data,~Element: Data,~Start: Data,~State: Type,~Value: Data,~Final: Data,~represent: Model -> Block,~cell: Model -> U32 -> Element,~operation: Value -> Element -> Value,~begin: Start -> State,~begin_value: Start -> Value,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~turn: State -> Final,~turn_value: Value -> Final,~Fill: Data,~vacant: Fill -> Block,~vacant_model: Fill -> Model,~vacant_same: @+fill: Fill -> {vacant(fill) == represent(vacant_model(fill)) : Block},~chain_same: LineChainValues.ChainSame(~Block,~Model,~represent,~Start,~State,~Value,~Final,~begin,~begin_value,~fold,~LineChainValues.strided_model(~Model,~Element,~Value,~cell,~operation),~turn,~turn_value,~Fill,~vacant,~vacant_model),~Context: Data,~value: Context -> U32 -> Element,+limits: Machine.Limits,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+length: U32,+inner: U32,+contexts: List<&2,Start>,+bytes: F32,+operations: F32,+fill: Fill,+workers: U32,+context: Context,-pieces: Partition.Tree<&1,Block,Unit>,
  +group: U32,+lines: Nat,+count_lines: {U32.to_nat(count) == Nat.mul(lines,U32.to_nat((length * inner : U32))) : Nat},+lines_positive: {Nat.is_lt(0n,lines) == True{} : Bool},
  +filled: {Nat.is_lt(0n,U32.to_nat((length * inner : U32))) == True{} : Bool},+turns: Turns<Start,contexts,inner>,+inner_positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},+inner_le: {Nat.is_le(U32.to_nat(inner),U32.to_nat((length * inner : U32))) == True{} : Bool},+bound_limits: {Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +low: Nat,+high: Nat,+group_run: Runs.GroupRun(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),task_limit(count,U32.max(workers,1),(length * inner : U32),Tasks.size(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group)))),low,high,U32.to_nat((length * inner : U32)),U32.to_nat(count)),
  +grid_spans: {Held.layout_spans(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),Held.Ended{}) == Held.layout_spans(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),Held.Ended{}) : Held.Spans},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(U32.to_nat(count)),4n),U32.to_nat(cap)) == True{} : Bool},+total: Nat,+total_exact: {total == U32.to_nat(count) : Nat},+source: Streamed<Block,Model,Element,represent,cell,pieces,Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),values_from(~Element,~Context,~value,context,total,0n)>) ->
  {Pair.snd(Partition.Tree<&1,Block,Unit>,List<&2,Final>,LineChain.folds(~Block,~Fill,~Start,~State,~Final,~begin,~fold,~turn,~vacant,contexts,LineChain.Line{length,inner}, group,bytes,operations,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},fill,workers,pieces)) == block_folds(~Element,~Value,~Final,~operation,~turn_value,lines,length,inner,values_from(~Element,~Context,~value,context,total,0n),block_starts(~Start,~Value,~begin_value,contexts,inner)) : List<&2,Final>}:
  Streamed{+tree,+packed,+laid,+stream} = source
  +spans = Equal.trans(Held.Spans,tree_spans(~Model,tree,Held.Ended{}),Held.layout_spans(Regrouped.tree_layout(~Model,tree),Held.Ended{}),Held.layout_spans(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),Held.Ended{}),
    tree_layout_spans(~Model,tree,Held.Ended{}),Equal.cong(Pieces.Ranges,Held.Spans,laid_out => Held.layout_spans(laid_out,Held.Ended{}),Regrouped.tree_layout(~Model,tree),Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),laid))
  +big = Nat.add(Nat.double(U32.to_nat(count)),4n)
  +count_below = Bounds.nat_trans(U32.to_nat(count),big,U32.to_nat(cap),
    Order.le_lt_trans(U32.to_nat(count),Nat.double(U32.to_nat(count)),big,Held.nat_le_right_moved(U32.to_nat(count),Nat.double(U32.to_nat(count)),Nat.add(U32.to_nat(count),U32.to_nat(count)),GridRanges.double_add(U32.to_nat(count)),Order.le_add(U32.to_nat(count),U32.to_nat(count))),
      Order.below_sum(Nat.double(U32.to_nat(count)),3n)),count_room)
  +line_inside = Held.nat_le_right_moved(U32.to_nat((length * inner : U32)),U32.to_nat(count),Nat.mul(lines,U32.to_nat((length * inner : U32))),count_lines,
    Held.nat_le_right_moved(U32.to_nat((length * inner : U32)),Nat.mul(lines,U32.to_nat((length * inner : U32))),Nat.mul(U32.to_nat((length * inner : U32)),lines),RowMajor.multiply_commutative(lines,U32.to_nat((length * inner : U32))),RowMajor.product_covers(U32.to_nat((length * inner : U32)),lines,lines_positive)))
  +cells_room = Order.transitive(Nat.double(U32.to_nat((length * inner : U32))),big,U32.to_nat(cap),Order.transitive(Nat.double(U32.to_nat((length * inner : U32))),Nat.double(U32.to_nat(count)),big,Order.double_le(U32.to_nat((length * inner : U32)),U32.to_nat(count),line_inside),Order.le_add(Nat.double(U32.to_nat(count)),4n)),count_room)
  +room = Order.transitive(Nat.double(U32.to_nat(inner)),Nat.double(U32.to_nat((length * inner : U32))),U32.to_nat(cap),Order.double_le(U32.to_nat(inner),U32.to_nat((length * inner : U32)),inner_le),cells_room)
  +block_room = Order.transitive(Nat.add(U32.to_nat((length * inner : U32)),U32.to_nat(inner)),Nat.double(U32.to_nat((length * inner : U32))),U32.to_nat(cap),
    Held.nat_le_right_moved(Nat.add(U32.to_nat((length * inner : U32)),U32.to_nat(inner)),Nat.double(U32.to_nat((length * inner : U32))),Nat.add(U32.to_nat((length * inner : U32)),U32.to_nat((length * inner : U32))),GridRanges.double_add(U32.to_nat((length * inner : U32))),Order.offset_le(U32.to_nat((length * inner : U32)),U32.to_nat(inner),U32.to_nat((length * inner : U32)),inner_le)),cells_room)
  +rooms = {BlockRooms{turns_room(~Start,contexts,inner,cap,turns,inner_positive,room),room,inner_positive,block_room,cells_room,filled} : BlockRooms<length,inner,cap,List.length(&2,Start,contexts)>}
  +same_spans = Equal.trans(Held.Spans,Held.layout_spans(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),Held.Ended{}),Held.layout_spans(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),Held.Ended{}),
    tree_spans(~Model,tree,Held.Ended{}),grid_spans,Equal.sym(Held.Spans,tree_spans(~Model,tree,Held.Ended{}),Held.layout_spans(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),Held.Ended{}),spans))
  +stream_exact = Equal.trans(List<&2,Element>,LineChainValues.stream(~Model,~Element,~cell,Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree)),LineChainValues.stream(~Model,~Element,~cell,tree),values_from(~Element,~Context,~value,context,total,0n),
    regrouped_stream(~Model,~Element,~cell,~Fill,~vacant_model,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree,same_spans),stream)
  +stream_length = Equal.trans(Nat,List.length(&2,Element,LineChainValues.stream(~Model,~Element,~cell,Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree))),List.length(&2,Element,values_from(~Element,~Context,~value,context,total,0n)),total,
    Equal.cong(List<&2,Element>,Nat,list => List.length(&2,Element,list),LineChainValues.stream(~Model,~Element,~cell,Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree)),values_from(~Element,~Context,~value,context,total,0n),stream_exact),values_length(~Element,~Context,~value,context,total,0n))
  +stream_room = Held.nat_le_moved(List.length(&2,Element,LineChainValues.stream(~Model,~Element,~cell,Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree))),total,U32.to_nat(cap),stream_length,
    Held.nat_le_moved(total,U32.to_nat(count),U32.to_nat(cap),total_exact,Bounds.strict_le(U32.to_nat(count),U32.to_nat(cap),count_below)))
  +whole = lined_blocks(~Model,~Element,~cell,~Fill,~vacant_model,limits,count,prefix,outer,extent,unit_cells,block,per,group,workers,length,inner,bytes,operations,U32.to_nat(count),fill,tree,cap,cap_ok,low,high,group_run,
    filled,cells_room,inner_le,bound_limits,count_room,count_below,stream_room)
  +states = carried_lines(~Block,~Model,~Element,~Start,~State,~Value,~Final,~represent,~cell,~operation,~begin,~begin_value,~fold,~turn,~turn_value,~Fill,~vacant,~vacant_model,~vacant_same,~chain_same,length,inner,contexts,group,bytes,operations,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},fill,workers,tree,cap,cap_ok,turns,rooms,whole)
  +grouped = Equal.trans(Nat,Nat.mul(task_blocks(~Model,length,inner,LineChainValues.decided_tasks(Tasks.decide_over(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),count,workers,(length * inner : U32),bytes,operations)),Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree)),U32.to_nat((length * inner : U32))),List.length(&2,Element,LineChainValues.stream(~Model,~Element,~cell,Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree))),Nat.mul(lines,U32.to_nat((length * inner : U32))),
    Equal.sym(Nat,List.length(&2,Element,LineChainValues.stream(~Model,~Element,~cell,Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree))),Nat.mul(task_blocks(~Model,length,inner,LineChainValues.decided_tasks(Tasks.decide_over(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),count,workers,(length * inner : U32),bytes,operations)),Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree)),U32.to_nat((length * inner : U32))),blocks_length(~Model,~Element,~cell,length,inner,cap,LineChainValues.decided_tasks(Tasks.decide_over(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),count,workers,(length * inner : U32),bytes,operations)),Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree),whole)),
    Equal.trans(Nat,List.length(&2,Element,LineChainValues.stream(~Model,~Element,~cell,Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree))),U32.to_nat(count),Nat.mul(lines,U32.to_nat((length * inner : U32))),Equal.trans(Nat,List.length(&2,Element,LineChainValues.stream(~Model,~Element,~cell,Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree))),total,U32.to_nat(count),stream_length,total_exact),count_lines))
  +groups_exact = Order.antisymmetric(task_blocks(~Model,length,inner,LineChainValues.decided_tasks(Tasks.decide_over(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),count,workers,(length * inner : U32),bytes,operations)),Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree)),lines,
    RowMajor.cancel_multiply_le(task_blocks(~Model,length,inner,LineChainValues.decided_tasks(Tasks.decide_over(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),count,workers,(length * inner : U32),bytes,operations)),Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree)),lines,U32.to_nat((length * inner : U32)),filled,Held.nat_le_moved(Nat.mul(task_blocks(~Model,length,inner,LineChainValues.decided_tasks(Tasks.decide_over(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),count,workers,(length * inner : U32),bytes,operations)),Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree)),U32.to_nat((length * inner : U32))),Nat.mul(lines,U32.to_nat((length * inner : U32))),Nat.mul(lines,U32.to_nat((length * inner : U32))),grouped,Order.reflexive(Nat.mul(lines,U32.to_nat((length * inner : U32)))))),
    RowMajor.cancel_multiply_le(lines,task_blocks(~Model,length,inner,LineChainValues.decided_tasks(Tasks.decide_over(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),count,workers,(length * inner : U32),bytes,operations)),Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree)),U32.to_nat((length * inner : U32)),filled,Held.nat_le_moved(Nat.mul(lines,U32.to_nat((length * inner : U32))),Nat.mul(task_blocks(~Model,length,inner,LineChainValues.decided_tasks(Tasks.decide_over(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),count,workers,(length * inner : U32),bytes,operations)),Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree)),U32.to_nat((length * inner : U32))),Nat.mul(task_blocks(~Model,length,inner,LineChainValues.decided_tasks(Tasks.decide_over(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),count,workers,(length * inner : U32),bytes,operations)),Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree)),U32.to_nat((length * inner : U32))),Equal.sym(Nat,Nat.mul(task_blocks(~Model,length,inner,LineChainValues.decided_tasks(Tasks.decide_over(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),count,workers,(length * inner : U32),bytes,operations)),Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree)),U32.to_nat((length * inner : U32))),Nat.mul(lines,U32.to_nat((length * inner : U32))),grouped),Order.reflexive(Nat.mul(task_blocks(~Model,length,inner,LineChainValues.decided_tasks(Tasks.decide_over(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),count,workers,(length * inner : U32),bytes,operations)),Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree)),U32.to_nat((length * inner : U32)))))))
  %packed : {Pair.snd(Partition.Tree<&1,Block,Unit>,List<&2,Final>,LineChain.folds(~Block,~Fill,~Start,~State,~Final,~begin,~fold,~turn,~vacant,contexts,LineChain.Line{length,inner}, group,bytes,operations,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},fill,workers,_)) == block_folds(~Element,~Value,~Final,~operation,~turn_value,lines,length,inner,values_from(~Element,~Context,~value,context,total,0n),block_starts(~Start,~Value,~begin_value,contexts,inner)) : List<&2,Final>}
  Equal.trans(List<&2,Final>,Pair.snd(Partition.Tree<&1,Block,Unit>,List<&2,Final>,LineChain.folds(~Block,~Fill,~Start,~State,~Final,~begin,~fold,~turn,~vacant,contexts,LineChain.Line{length,inner}, group,bytes,operations,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},fill,workers,Regrouped.tree_of(~Block,~Model,~represent,tree))),
    block_folds(~Element,~Value,~Final,~operation,~turn_value,task_blocks(~Model,length,inner,LineChainValues.decided_tasks(Tasks.decide_over(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),count,workers,(length * inner : U32),bytes,operations)),Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree)),length,inner,LineChainValues.stream(~Model,~Element,~cell,Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree)),block_starts(~Start,~Value,~begin_value,contexts,inner)),block_folds(~Element,~Value,~Final,~operation,~turn_value,lines,length,inner,values_from(~Element,~Context,~value,context,total,0n),block_starts(~Start,~Value,~begin_value,contexts,inner)),states,
    Equal.trans(List<&2,Final>,block_folds(~Element,~Value,~Final,~operation,~turn_value,task_blocks(~Model,length,inner,LineChainValues.decided_tasks(Tasks.decide_over(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),count,workers,(length * inner : U32),bytes,operations)),Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree)),length,inner,LineChainValues.stream(~Model,~Element,~cell,Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree)),block_starts(~Start,~Value,~begin_value,contexts,inner)),block_folds(~Element,~Value,~Final,~operation,~turn_value,lines,length,inner,LineChainValues.stream(~Model,~Element,~cell,Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree)),block_starts(~Start,~Value,~begin_value,contexts,inner)),block_folds(~Element,~Value,~Final,~operation,~turn_value,lines,length,inner,values_from(~Element,~Context,~value,context,total,0n),block_starts(~Start,~Value,~begin_value,contexts,inner)),
      Equal.cong(Nat,List<&2,Final>,groups => block_folds(~Element,~Value,~Final,~operation,~turn_value,groups,length,inner,LineChainValues.stream(~Model,~Element,~cell,Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree)),block_starts(~Start,~Value,~begin_value,contexts,inner)),task_blocks(~Model,length,inner,LineChainValues.decided_tasks(Tasks.decide_over(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),count,workers,(length * inner : U32),bytes,operations)),Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree)),lines,groups_exact),
      Equal.cong(List<&2,Element>,List<&2,Final>,list => block_folds(~Element,~Value,~Final,~operation,~turn_value,lines,length,inner,list,block_starts(~Start,~Value,~begin_value,contexts,inner)),LineChainValues.stream(~Model,~Element,~cell,Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),fill,tree)),values_from(~Element,~Context,~value,context,total,0n),stream_exact)))
# The blocks of the lines come back as they were once regrouped into the
# pieces of the grid.
def lined_kept(~Block: Type,~Model: Data,~Element: Data,~Start: Data,~State: Type,~Value: Data,~Final: Data,~represent: Model -> Block,~cell: Model -> U32 -> Element,~operation: Value -> Element -> Value,~begin: Start -> State,~begin_value: Start -> Value,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~turn: State -> Final,~turn_value: Value -> Final,~Fill: Data,~vacant: Fill -> Block,~vacant_model: Fill -> Model,~vacant_same: @+fill: Fill -> {vacant(fill) == represent(vacant_model(fill)) : Block},~chain_same: LineChainValues.ChainSame(~Block,~Model,~represent,~Start,~State,~Value,~Final,~begin,~begin_value,~fold,~LineChainValues.strided_model(~Model,~Element,~Value,~cell,~operation),~turn,~turn_value,~Fill,~vacant,~vacant_model),~Context: Data,~value: Context -> U32 -> Element,+limits: Machine.Limits,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+length: U32,+inner: U32,+contexts: List<&2,Start>,+bytes: F32,+operations: F32,+fill: Fill,+workers: U32,+context: Context,-pieces: Partition.Tree<&1,Block,Unit>,
  +group: U32,+lines: Nat,+count_lines: {U32.to_nat(count) == Nat.mul(lines,U32.to_nat((length * inner : U32))) : Nat},+lines_positive: {Nat.is_lt(0n,lines) == True{} : Bool},
  +filled: {Nat.is_lt(0n,U32.to_nat((length * inner : U32))) == True{} : Bool},+turns: Turns<Start,contexts,inner>,+inner_positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},+inner_le: {Nat.is_le(U32.to_nat(inner),U32.to_nat((length * inner : U32))) == True{} : Bool},+bound_limits: {Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +low: Nat,+high: Nat,+group_run: Runs.GroupRun(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),task_limit(count,U32.max(workers,1),(length * inner : U32),Tasks.size(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group)))),low,high,U32.to_nat((length * inner : U32)),U32.to_nat(count)),
  +grid_spans: {Held.layout_spans(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),Held.Ended{}) == Held.layout_spans(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),Held.Ended{}) : Held.Spans},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(U32.to_nat(count)),4n),U32.to_nat(cap)) == True{} : Bool},+total: Nat,+total_exact: {total == U32.to_nat(count) : Nat},+source: Streamed<Block,Model,Element,represent,cell,pieces,Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),values_from(~Element,~Context,~value,context,total,0n)>) ->
  {Pieces.blocks_regrouped(~Block,~Fill,~vacant,Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),fill,Pair.fst(Partition.Tree<&1,Block,Unit>,List<&2,Final>,LineChain.folds(~Block,~Fill,~Start,~State,~Final,~begin,~fold,~turn,~vacant,contexts,LineChain.Line{length,inner},group,bytes,operations,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},fill,workers,pieces))) == pieces : Partition.Tree<&1,Block,Unit>}:
  Streamed{+tree,+packed,+laid,+stream} = source
  +grid = {Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)} : Grid.Grid}
  +whole = Pieces.ranges_of_grid(grid)
  +tasks = Pieces.ranges_in(grid,group)
  +counted = ranges_counted(~Model,tasks,tree,Equal.trans(Held.Spans,Held.layout_spans(tasks,Held.Ended{}),Held.layout_spans(whole,Held.Ended{}),Held.layout_spans(Regrouped.tree_layout(~Model,tree),Held.Ended{}),grid_spans,
    Equal.cong(Pieces.Ranges,Held.Spans,laid_out => Held.layout_spans(laid_out,Held.Ended{}),whole,Regrouped.tree_layout(~Model,tree),Equal.sym(Pieces.Ranges,Regrouped.tree_layout(~Model,tree),whole,laid))))
  +regrouped = Regrouped.regrouped(~Model,~Fill,~vacant_model,tasks,fill,tree)
  +twice = {%laid : {Regrouped.regrouped(~Model,~Fill,~vacant_model,_,fill,regrouped) == tree : Partition.Tree<&2,Model,Unit>}
    Regrouped.regrouped_twice(~Model,~Fill,~vacant_model,tasks,fill,tree,counted) : {Regrouped.regrouped(~Model,~Fill,~vacant_model,whole,fill,regrouped) == tree : Partition.Tree<&2,Model,Unit>}}
  %packed : {Pieces.blocks_regrouped(~Block,~Fill,~vacant,whole,fill,Pair.fst(Partition.Tree<&1,Block,Unit>,List<&2,Final>,LineChain.folds(~Block,~Fill,~Start,~State,~Final,~begin,~fold,~turn,~vacant,contexts,LineChain.Line{length,inner},group,bytes,operations,grid,fill,workers,_))) == _ : Partition.Tree<&1,Block,Unit>}
  %Equal.sym(Partition.Tree<&1,Block,Unit>,Pair.fst(Partition.Tree<&1,Block,Unit>,List<&2,Final>,LineChain.folds(~Block,~Fill,~Start,~State,~Final,~begin,~fold,~turn,~vacant,contexts,LineChain.Line{length,inner},group,bytes,operations,grid,fill,workers,Regrouped.tree_of(~Block,~Model,~represent,tree))),
      Regrouped.tree_of(~Block,~Model,~represent,regrouped),
      LineChainValues.folds_kept(~Block,~Model,~represent,~Start,~State,~Value,~Final,~begin,~begin_value,~fold,~LineChainValues.strided_model(~Model,~Element,~Value,~cell,~operation),~turn,~turn_value,~Fill,~vacant,~vacant_model,~chain_same,
        contexts,LineChain.Line{length,inner},group,bytes,operations,grid,fill,workers,tree)) :
    {Pieces.blocks_regrouped(~Block,~Fill,~vacant,whole,fill,_) == Regrouped.tree_of(~Block,~Model,~represent,tree) : Partition.Tree<&1,Block,Unit>}
  Equal.trans(Partition.Tree<&1,Block,Unit>,Pieces.blocks_regrouped(~Block,~Fill,~vacant,whole,fill,Regrouped.tree_of(~Block,~Model,~represent,regrouped)),
    Regrouped.tree_of(~Block,~Model,~represent,Regrouped.regrouped(~Model,~Fill,~vacant_model,whole,fill,regrouped)),Regrouped.tree_of(~Block,~Model,~represent,tree),
    Regrouped.regrouped_same(~Block,~Model,~Fill,~vacant,~represent,~vacant_model,~vacant_same,whole,fill,regrouped),
    Equal.cong(Partition.Tree<&2,Model,Unit>,Partition.Tree<&1,Block,Unit>,blocks => Regrouped.tree_of(~Block,~Model,~represent,blocks),Regrouped.regrouped(~Model,~Fill,~vacant_model,whole,fill,regrouped),tree,twice))

# The lines fold into the block folds of the tensor's values in order, and
# the pieces come back as they were.
def lined_folds(~Block: Type,~Model: Data,~Element: Data,~Start: Data,~State: Type,~Value: Data,~Final: Data,~represent: Model -> Block,~cell: Model -> U32 -> Element,~operation: Value -> Element -> Value,~begin: Start -> State,~begin_value: Start -> Value,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~turn: State -> Final,~turn_value: Value -> Final,~Fill: Data,~vacant: Fill -> Block,~vacant_model: Fill -> Model,~vacant_same: @+fill: Fill -> {vacant(fill) == represent(vacant_model(fill)) : Block},~chain_same: LineChainValues.ChainSame(~Block,~Model,~represent,~Start,~State,~Value,~Final,~begin,~begin_value,~fold,~LineChainValues.strided_model(~Model,~Element,~Value,~cell,~operation),~turn,~turn_value,~Fill,~vacant,~vacant_model),~Context: Data,~value: Context -> U32 -> Element,+limits: Machine.Limits,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+length: U32,+inner: U32,+contexts: List<&2,Start>,+bytes: F32,+operations: F32,+fill: Fill,+workers: U32,+context: Context,-pieces: Partition.Tree<&1,Block,Unit>,
  +group: U32,+lines: Nat,+count_lines: {U32.to_nat(count) == Nat.mul(lines,U32.to_nat((length * inner : U32))) : Nat},+lines_positive: {Nat.is_lt(0n,lines) == True{} : Bool},
  +filled: {Nat.is_lt(0n,U32.to_nat((length * inner : U32))) == True{} : Bool},+turns: Turns<Start,contexts,inner>,+inner_positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},+inner_le: {Nat.is_le(U32.to_nat(inner),U32.to_nat((length * inner : U32))) == True{} : Bool},+bound_limits: {Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +low: Nat,+high: Nat,+group_run: Runs.GroupRun(Pieces.ranges_fold(~Tasks.Group,~U32,~Tasks.group_range,~Tasks.group_join,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),task_limit(count,U32.max(workers,1),(length * inner : U32),Tasks.size(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group)))),low,high,U32.to_nat((length * inner : U32)),U32.to_nat(count)),
  +grid_spans: {Held.layout_spans(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),Held.Ended{}) == Held.layout_spans(Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),Held.Ended{}) : Held.Spans},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(U32.to_nat(count)),4n),U32.to_nat(cap)) == True{} : Bool},+total: Nat,+total_exact: {total == U32.to_nat(count) : Nat},+source: Streamed<Block,Model,Element,represent,cell,pieces,Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),values_from(~Element,~Context,~value,context,total,0n)>) ->
  LinesFolded(~Block,~Fill,~Start,~State,~Final,~begin,~fold,~turn,~vacant,contexts,LineChain.Line{length,inner},group,bytes,operations,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},fill,workers,pieces,block_folds(~Element,~Value,~Final,~operation,~turn_value,lines,length,inner,values_from(~Element,~Context,~value,context,total,0n),block_starts(~Start,~Value,~begin_value,contexts,inner))):
  KernelGemm.Both{lined_states(~Block,~Model,~Element,~Start,~State,~Value,~Final,~represent,~cell,~operation,~begin,~begin_value,~fold,~turn,~turn_value,~Fill,~vacant,~vacant_model,~vacant_same,~chain_same,~Context,~value,limits,count,prefix,outer,extent,unit_cells,block,per,length,inner,contexts,bytes,operations,fill,workers,context,pieces,group,lines,count_lines,lines_positive,filled,turns,inner_positive,inner_le,bound_limits,low,high,group_run,grid_spans,cap,cap_ok,count_room,total,total_exact,source),lined_kept(~Block,~Model,~Element,~Start,~State,~Value,~Final,~represent,~cell,~operation,~begin,~begin_value,~fold,~turn,~turn_value,~Fill,~vacant,~vacant_model,~vacant_same,~chain_same,~Context,~value,limits,count,prefix,outer,extent,unit_cells,block,per,length,inner,contexts,bytes,operations,fill,workers,context,pieces,group,lines,count_lines,lines_positive,filled,turns,inner_positive,inner_le,bound_limits,low,high,group_run,grid_spans,cap,cap_ok,count_room,total,total_exact,source)}

# Blocks of whole outer indices, size of them each, fold into the block folds
# of the tensor's values in order.
def carried_folds(~Block: Type,~Model: Data,~Element: Data,~Start: Data,~State: Type,~Value: Data,~Final: Data,~represent: Model -> Block,~cell: Model -> U32 -> Element,~operation: Value -> Element -> Value,~begin: Start -> State,~begin_value: Start -> Value,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~turn: State -> Final,~turn_value: Value -> Final,~Fill: Data,~vacant: Fill -> Block,~vacant_model: Fill -> Model,~vacant_same: @+fill: Fill -> {vacant(fill) == represent(vacant_model(fill)) : Block},~chain_same: LineChainValues.ChainSame(~Block,~Model,~represent,~Start,~State,~Value,~Final,~begin,~begin_value,~fold,~LineChainValues.strided_model(~Model,~Element,~Value,~cell,~operation),~turn,~turn_value,~Fill,~vacant,~vacant_model),~Context: Data,~value: Context -> U32 -> Element,+limits: Machine.Limits,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: GridPieces.Shape<limits,outer,extent,unit_cells,block,per>,+cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+length: U32,+inner: U32,+contexts: List<&2,Start>,+bytes: F32,+operations: F32,+fill: Fill,+workers: U32,+context: Context,-pieces: Partition.Tree<&1,Block,Unit>,
  +group: U32,+lines: Nat,+outer_exact: {U32.to_nat(outer) == Nat.mul(lines,U32.to_nat(U32.max(group,1))) : Nat},+lines_positive: {Nat.is_lt(0n,lines) == True{} : Bool},
  +length_exact: {U32.to_nat((length * inner : U32)) == Nat.mul(U32.to_nat(U32.max(group,1)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))) : Nat},+filled: {Nat.is_lt(0n,U32.to_nat((length * inner : U32))) == True{} : Bool},+turns: Turns<Start,contexts,inner>,+inner_positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},+inner_le: {Nat.is_le(U32.to_nat(inner),U32.to_nat((length * inner : U32))) == True{} : Bool},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(U32.to_nat(count)),4n),U32.to_nat(cap)) == True{} : Bool},+total: Nat,+total_exact: {total == U32.to_nat(count) : Nat},+source: Streamed<Block,Model,Element,represent,cell,pieces,Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),values_from(~Element,~Context,~value,context,total,0n)>) ->
  LinesFolded(~Block,~Fill,~Start,~State,~Final,~begin,~fold,~turn,~vacant,contexts,LineChain.Line{length,inner},group,bytes,operations,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},fill,workers,pieces,block_folds(~Element,~Value,~Final,~operation,~turn_value,lines,length,inner,values_from(~Element,~Context,~value,context,total,0n),block_starts(~Start,~Value,~begin_value,contexts,inner))):
  +count_lines = Equal.trans(Nat,U32.to_nat(count),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),Nat.mul(lines,U32.to_nat((length * inner : U32))),cells,
    Equal.trans(Nat,Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(outer),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(lines,U32.to_nat((length * inner : U32))),RowMajor.assoc(U32.to_nat(outer),U32.to_nat(extent),U32.to_nat(unit_cells)),
      Equal.trans(Nat,Nat.mul(U32.to_nat(outer),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(Nat.mul(lines,U32.to_nat(U32.max(group,1))),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(lines,U32.to_nat((length * inner : U32))),
        Equal.cong(Nat,Nat,number => Nat.mul(number,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat(outer),Nat.mul(lines,U32.to_nat(U32.max(group,1))),outer_exact),
        Equal.trans(Nat,Nat.mul(Nat.mul(lines,U32.to_nat(U32.max(group,1))),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(lines,Nat.mul(U32.to_nat(U32.max(group,1)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)))),Nat.mul(lines,U32.to_nat((length * inner : U32))),RowMajor.assoc(lines,U32.to_nat(U32.max(group,1)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),
          Equal.cong(Nat,Nat,number => Nat.mul(lines,number),Nat.mul(U32.to_nat(U32.max(group,1)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),U32.to_nat((length * inner : U32)),Equal.sym(Nat,U32.to_nat((length * inner : U32)),Nat.mul(U32.to_nat(U32.max(group,1)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),length_exact))))))
  +bound_inside = Held.nat_le_moved(Nat.mul(lines,U32.to_nat((length * inner : U32))),U32.to_nat(count),U32.to_nat(count),Equal.sym(Nat,U32.to_nat(count),Nat.mul(lines,U32.to_nat((length * inner : U32))),count_lines),Order.reflexive(U32.to_nat(count)))
  +bound_limits = Held.nat_le_moved(U32.to_nat(count),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits)),cells,GridPieces.shape_room(limits,outer,extent,unit_cells,block,per,shape))
  +group_run = carried_group_run(limits,count,prefix,outer,extent,unit_cells,block,per,shape,cells,group,workers,(length * inner : U32),bytes,operations,lines,U32.to_nat((length * inner : U32)),U32.to_nat(count),outer_exact,lines_positive,length_exact,filled,
    Order.reflexive(U32.to_nat((length * inner : U32))),bound_inside,bound_limits)
  lined_folds(~Block,~Model,~Element,~Start,~State,~Value,~Final,~represent,~cell,~operation,~begin,~begin_value,~fold,~turn,~turn_value,~Fill,~vacant,~vacant_model,~vacant_same,~chain_same,~Context,~value,limits,count,prefix,outer,extent,unit_cells,block,per,length,inner,contexts,bytes,operations,fill,workers,context,pieces,group,lines,count_lines,lines_positive,filled,turns,inner_positive,inner_le,bound_limits,
    Nat.mul(U32.to_nat(0),U32.to_nat((length * inner : U32))),Nat.mul(Nat.add(U32.to_nat(0),U32.to_nat(U32.max((outer / U32.max(group,1) : U32),1))),U32.to_nat((length * inner : U32))),group_run,
    grouped_spans(limits,outer,extent,unit_cells,block,per,shape,count,prefix,group,lines,outer_exact,lines_positive),cap,cap_ok,count_room,total,total_exact,source)

# Blocks inside the units of the pieces, groups of one outer index, fold into
# the block folds of the tensor's values in order.
def within_folds(~Block: Type,~Model: Data,~Element: Data,~Start: Data,~State: Type,~Value: Data,~Final: Data,~represent: Model -> Block,~cell: Model -> U32 -> Element,~operation: Value -> Element -> Value,~begin: Start -> State,~begin_value: Start -> Value,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~turn: State -> Final,~turn_value: Value -> Final,~Fill: Data,~vacant: Fill -> Block,~vacant_model: Fill -> Model,~vacant_same: @+fill: Fill -> {vacant(fill) == represent(vacant_model(fill)) : Block},~chain_same: LineChainValues.ChainSame(~Block,~Model,~represent,~Start,~State,~Value,~Final,~begin,~begin_value,~fold,~LineChainValues.strided_model(~Model,~Element,~Value,~cell,~operation),~turn,~turn_value,~Fill,~vacant,~vacant_model),~Context: Data,~value: Context -> U32 -> Element,+limits: Machine.Limits,+count: U32,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+block: U32,+per: U32,+shape: GridPieces.Shape<limits,outer,extent,unit_cells,block,per>,+cells: {U32.to_nat(count) == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+length: U32,+inner: U32,+contexts: List<&2,Start>,+bytes: F32,+operations: F32,+fill: Fill,+workers: U32,+context: Context,-pieces: Partition.Tree<&1,Block,Unit>,
  +group: U32,+size_one: {U32.max(group,1) == 1 : U32},+units: Nat,+unit_exact: {U32.to_nat(unit_cells) == Nat.mul(units,U32.to_nat((length * inner : U32))) : Nat},
  +lines: Nat,+count_lines: {U32.to_nat(count) == Nat.mul(lines,U32.to_nat((length * inner : U32))) : Nat},+lines_positive: {Nat.is_lt(0n,lines) == True{} : Bool},
  +filled: {Nat.is_lt(0n,U32.to_nat((length * inner : U32))) == True{} : Bool},+turns: Turns<Start,contexts,inner>,+inner_positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},+inner_le: {Nat.is_le(U32.to_nat(inner),U32.to_nat((length * inner : U32))) == True{} : Bool},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(U32.to_nat(count)),4n),U32.to_nat(cap)) == True{} : Bool},+total: Nat,+total_exact: {total == U32.to_nat(count) : Nat},+source: Streamed<Block,Model,Element,represent,cell,pieces,Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),values_from(~Element,~Context,~value,context,total,0n)>) ->
  LinesFolded(~Block,~Fill,~Start,~State,~Final,~begin,~fold,~turn,~vacant,contexts,LineChain.Line{length,inner},group,bytes,operations,Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},fill,workers,pieces,block_folds(~Element,~Value,~Final,~operation,~turn_value,lines,length,inner,values_from(~Element,~Context,~value,context,total,0n),block_starts(~Start,~Value,~begin_value,contexts,inner))):
  +same = Equal.cong(U32,Pieces.Ranges,size => Pieces.outer_ranges(32n,U32.is_le(U32.max((outer / size : U32),1),1),Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},(U32.max(per,1) * size : U32),0,U32.max((outer / size : U32),1)),
    U32.max(group,1),1,size_one)
  +bound_limits = Held.nat_le_moved(U32.to_nat(count),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits)),cells,GridPieces.shape_room(limits,outer,extent,unit_cells,block,per,shape))
  +run = {%Equal.sym(Pieces.Ranges,Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),same) : Runs.Run(_,U32.to_nat((length * inner : U32)),U32.to_nat(count))
    GridRanges.grid_run(limits,outer,extent,unit_cells,block,per,count,prefix,shape,cells,U32.to_nat((length * inner : U32)),units,unit_exact) : Runs.Run(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),U32.to_nat((length * inner : U32)),U32.to_nat(count))}
  lined_folds(~Block,~Model,~Element,~Start,~State,~Value,~Final,~represent,~cell,~operation,~begin,~begin_value,~fold,~turn,~turn_value,~Fill,~vacant,~vacant_model,~vacant_same,~chain_same,~Context,~value,limits,count,prefix,outer,extent,unit_cells,block,per,length,inner,contexts,bytes,operations,fill,workers,context,pieces,group,lines,count_lines,lines_positive,filled,turns,inner_positive,inner_le,bound_limits,
    Runs.range_low(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group)),Runs.range_high(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group)),Runs.grouped(limits,task_limit(count,U32.max(workers,1),(length * inner : U32),Tasks.size(Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group))),Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),U32.to_nat((length * inner : U32)),U32.to_nat(count),run,bound_limits),
    Equal.cong(Pieces.Ranges,Held.Spans,layout => Held.layout_spans(layout,Held.Ended{}),Pieces.ranges_in(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)},group),Pieces.ranges_of_grid(Grid.Grid{count,prefix,outer,extent,unit_cells,block,per,(outer * per : U32)}),same),cap,cap_ok,count_room,total,total_exact,source)

# The trailing blocks of a grid of units fold into the block folds of the
# tensor's values in order, whether they lie within a unit or across whole
# outer indices.
def units_folds(~Block: Type,~Model: Data,~Element: Data,~Start: Data,~State: Type,~Value: Data,~Final: Data,~represent: Model -> Block,~cell: Model -> U32 -> Element,~operation: Value -> Element -> Value,~begin: Start -> State,~begin_value: Start -> Value,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~turn: State -> Final,~turn_value: Value -> Final,~Fill: Data,~vacant: Fill -> Block,~vacant_model: Fill -> Model,~vacant_same: @+fill: Fill -> {vacant(fill) == represent(vacant_model(fill)) : Block},~chain_same: LineChainValues.ChainSame(~Block,~Model,~represent,~Start,~State,~Value,~Final,~begin,~begin_value,~fold,~LineChainValues.strided_model(~Model,~Element,~Value,~cell,~operation),~turn,~turn_value,~Fill,~vacant,~vacant_model),~Context: Data,~value: Context -> U32 -> Element,+limits: Machine.Limits,+prefix: U32,+outer: U32,+extent: U32,+unit_cells: U32,+target: U32,+outer_positive: {Nat.is_lt(0n,U32.to_nat(outer)) == True{} : Bool},+extent_positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},+unit_positive: {Nat.is_lt(0n,U32.to_nat(unit_cells)) == True{} : Bool},+room: {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+length: U32,+inner: U32,+contexts: List<&2,Start>,+bytes: F32,+operations: F32,+fill: Fill,+workers: U32,+context: Context,-pieces: Partition.Tree<&1,Block,Unit>,
  +total: Nat,+total_exact: {total == Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)) : Nat},+lines: Nat,+lines_exact: {total == Nat.mul(lines,U32.to_nat((length * inner : U32))) : Nat},+lines_positive: {Nat.is_lt(0n,lines) == True{} : Bool},
  trailing: Trailing<Grid.Units{prefix,outer,extent,unit_cells},lines,U32.to_nat((length * inner : U32))>,+turns: Turns<Start,contexts,inner>,+inner_positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},+inner_le: {Nat.is_le(U32.to_nat(inner),U32.to_nat((length * inner : U32))) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(total),4n),U32.to_nat(cap)) == True{} : Bool},
  +source: Streamed<Block,Model,Element,represent,cell,pieces,Pieces.ranges_of_grid(Grid.grid_of_units(Grid.Units{prefix,outer,extent,unit_cells},target)),values_from(~Element,~Context,~value,context,total,0n)>) ->
  LinesFolded(~Block,~Fill,~Start,~State,~Final,~begin,~fold,~turn,~vacant,contexts,LineChain.Line{length,inner},((length * inner : U32) / U32.max(Layout.cells_per_outer(Grid.grid_of_units(Grid.Units{prefix,outer,extent,unit_cells},target)),1) : U32),bytes,operations,Grid.grid_of_units(Grid.Units{prefix,outer,extent,unit_cells},target),fill,workers,pieces,block_folds(~Element,~Value,~Final,~operation,~turn_value,lines,length,inner,values_from(~Element,~Context,~value,context,total,0n),block_starts(~Start,~Value,~begin_value,contexts,inner))):
  match trailing:
    case Within{+units,+exact}:
      +shape = GridRanges.units_shape(limits,outer,extent,unit_cells,target,outer_positive,extent_positive,unit_positive,room)
      +count_word = GridRanges.units_word(limits,outer,extent,unit_cells,unit_positive,room)
      +total_word = Equal.trans(Nat,total,Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat((outer * extent * unit_cells : U32)),total_exact,Equal.sym(Nat,U32.to_nat((outer * extent * unit_cells : U32)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),count_word))
      +count_lines = Equal.trans(Nat,U32.to_nat((outer * extent * unit_cells : U32)),total,Nat.mul(lines,U32.to_nat((length * inner : U32))),Equal.sym(Nat,total,U32.to_nat((outer * extent * unit_cells : U32)),total_word),lines_exact)
      +count_room_word = Held.nat_le_moved(Nat.add(Nat.double(U32.to_nat((outer * extent * unit_cells : U32))),4n),Nat.add(Nat.double(total),4n),U32.to_nat(cap),
        Equal.cong(Nat,Nat,number => Nat.add(Nat.double(number),4n),U32.to_nat((outer * extent * unit_cells : U32)),total,Equal.sym(Nat,total,U32.to_nat((outer * extent * unit_cells : U32)),total_word)),count_room)
      +units_filled = Held.nat_lt_moved(0n,Nat.mul(units,U32.to_nat((length * inner : U32))),U32.to_nat(unit_cells),Equal.sym(Nat,U32.to_nat(unit_cells),Nat.mul(units,U32.to_nat((length * inner : U32))),exact),unit_positive)
      +line_in_unit = Held.nat_le_right_moved(U32.to_nat((length * inner : U32)),U32.to_nat(unit_cells),Nat.mul(U32.to_nat((length * inner : U32)),units),Equal.trans(Nat,U32.to_nat(unit_cells),Nat.mul(units,U32.to_nat((length * inner : U32))),Nat.mul(U32.to_nat((length * inner : U32)),units),exact,RowMajor.multiply_commutative(units,U32.to_nat((length * inner : U32)))),
        RowMajor.product_covers(U32.to_nat((length * inner : U32)),units,RowMajor.left_factor_positive(units,U32.to_nat((length * inner : U32)),units_filled)))
      +unit_in_outer = Held.nat_le_right_moved(U32.to_nat(unit_cells),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),Nat.mul(U32.to_nat(unit_cells),U32.to_nat(extent)),RowMajor.multiply_commutative(U32.to_nat(extent),U32.to_nat(unit_cells)),RowMajor.product_covers(U32.to_nat(unit_cells),U32.to_nat(extent),extent_positive))
      +short = Held.nat_le_right_moved(U32.to_nat((length * inner : U32)),U32.to_nat(U32.max(Layout.cells_per_outer(Grid.grid_of_units(Grid.Units{prefix,outer,extent,unit_cells},target)),1)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),per_outer_word(limits,prefix,outer,extent,unit_cells,target,outer_positive,extent_positive,unit_positive,room),Order.transitive(U32.to_nat((length * inner : U32)),U32.to_nat(unit_cells),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),line_in_unit,unit_in_outer))
      +size_one = single_group((length * inner : U32),U32.max(Layout.cells_per_outer(Grid.grid_of_units(Grid.Units{prefix,outer,extent,unit_cells},target)),1),Machine.half_word(limits),LimitBounds.half_word_fits(limits),short,per_outer_positive(limits,prefix,outer,extent,unit_cells,target,outer_positive,extent_positive,unit_positive,room),per_outer_double(limits,prefix,outer,extent,unit_cells,target,outer_positive,extent_positive,unit_positive,room))
      within_folds(~Block,~Model,~Element,~Start,~State,~Value,~Final,~represent,~cell,~operation,~begin,~begin_value,~fold,~turn,~turn_value,~Fill,~vacant,~vacant_model,~vacant_same,~chain_same,~Context,~value,limits,(outer * extent * unit_cells : U32),prefix,outer,extent,unit_cells,Grid.piece_units(unit_cells,extent,target),Grid.divided_up(extent,Grid.piece_units(unit_cells,extent,target)),shape,count_word,length,inner,contexts,bytes,operations,fill,workers,context,pieces,((length * inner : U32) / U32.max(Layout.cells_per_outer(Grid.grid_of_units(Grid.Units{prefix,outer,extent,unit_cells},target)),1) : U32),size_one,units,exact,
        lines,count_lines,lines_positive,GridRanges.right_factor_positive(units,U32.to_nat((length * inner : U32)),units_filled),turns,inner_positive,inner_le,cap,cap_ok,count_room_word,total,total_word,source)
    case Carried{+size,+outer_exact,+length_exact}:
      +shape = GridRanges.units_shape(limits,outer,extent,unit_cells,target,outer_positive,extent_positive,unit_positive,room)
      +count_word = GridRanges.units_word(limits,outer,extent,unit_cells,unit_positive,room)
      +total_word = Equal.trans(Nat,total,Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),U32.to_nat((outer * extent * unit_cells : U32)),total_exact,Equal.sym(Nat,U32.to_nat((outer * extent * unit_cells : U32)),Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),count_word))
      +count_room_word = Held.nat_le_moved(Nat.add(Nat.double(U32.to_nat((outer * extent * unit_cells : U32))),4n),Nat.add(Nat.double(total),4n),U32.to_nat(cap),
        Equal.cong(Nat,Nat,number => Nat.add(Nat.double(number),4n),U32.to_nat((outer * extent * unit_cells : U32)),total,Equal.sym(Nat,total,U32.to_nat((outer * extent * unit_cells : U32)),total_word)),count_room)
      +size_positive = GridRanges.right_factor_positive(lines,size,Held.nat_lt_moved(0n,Nat.mul(lines,size),U32.to_nat(outer),Equal.sym(Nat,U32.to_nat(outer),Nat.mul(lines,size),outer_exact),outer_positive))
      +spread = Equal.trans(Nat,U32.to_nat((length * inner : U32)),Nat.mul(size,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(size,U32.to_nat(U32.max(Layout.cells_per_outer(Grid.grid_of_units(Grid.Units{prefix,outer,extent,unit_cells},target)),1))),length_exact,
        Equal.cong(Nat,Nat,number => Nat.mul(size,number),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),U32.to_nat(U32.max(Layout.cells_per_outer(Grid.grid_of_units(Grid.Units{prefix,outer,extent,unit_cells},target)),1)),Equal.sym(Nat,U32.to_nat(U32.max(Layout.cells_per_outer(Grid.grid_of_units(Grid.Units{prefix,outer,extent,unit_cells},target)),1)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),per_outer_word(limits,prefix,outer,extent,unit_cells,target,outer_positive,extent_positive,unit_positive,room))))
      +group_size = Extrema.max_one_exact(((length * inner : U32) / U32.max(Layout.cells_per_outer(Grid.grid_of_units(Grid.Units{prefix,outer,extent,unit_cells},target)),1) : U32),size,
        exact_quotient(Machine.half_word(limits),LimitBounds.half_word_fits(limits),(length * inner : U32),U32.max(Layout.cells_per_outer(Grid.grid_of_units(Grid.Units{prefix,outer,extent,unit_cells},target)),1),size,spread,per_outer_positive(limits,prefix,outer,extent,unit_cells,target,outer_positive,extent_positive,unit_positive,room),per_outer_double(limits,prefix,outer,extent,unit_cells,target,outer_positive,extent_positive,unit_positive,room)),size_positive)
      +outer_group = Equal.trans(Nat,U32.to_nat(outer),Nat.mul(lines,size),Nat.mul(lines,U32.to_nat(U32.max(((length * inner : U32) / U32.max(Layout.cells_per_outer(Grid.grid_of_units(Grid.Units{prefix,outer,extent,unit_cells},target)),1) : U32),1))),outer_exact,
        Equal.cong(Nat,Nat,number => Nat.mul(lines,number),size,U32.to_nat(U32.max(((length * inner : U32) / U32.max(Layout.cells_per_outer(Grid.grid_of_units(Grid.Units{prefix,outer,extent,unit_cells},target)),1) : U32),1)),Equal.sym(Nat,U32.to_nat(U32.max(((length * inner : U32) / U32.max(Layout.cells_per_outer(Grid.grid_of_units(Grid.Units{prefix,outer,extent,unit_cells},target)),1) : U32),1)),size,group_size)))
      +length_group = Equal.trans(Nat,U32.to_nat((length * inner : U32)),Nat.mul(size,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),Nat.mul(U32.to_nat(U32.max(((length * inner : U32) / U32.max(Layout.cells_per_outer(Grid.grid_of_units(Grid.Units{prefix,outer,extent,unit_cells},target)),1) : U32),1)),Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),length_exact,
        Equal.cong(Nat,Nat,number => Nat.mul(number,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),size,U32.to_nat(U32.max(((length * inner : U32) / U32.max(Layout.cells_per_outer(Grid.grid_of_units(Grid.Units{prefix,outer,extent,unit_cells},target)),1) : U32),1)),Equal.sym(Nat,U32.to_nat(U32.max(((length * inner : U32) / U32.max(Layout.cells_per_outer(Grid.grid_of_units(Grid.Units{prefix,outer,extent,unit_cells},target)),1) : U32),1)),size,group_size)))
      +filled = Held.nat_lt_moved(0n,U32.to_nat((length * inner : U32)),Nat.mul(size,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells))),length_exact,Order.product_positive(size,Nat.mul(U32.to_nat(extent),U32.to_nat(unit_cells)),size_positive,Order.product_positive(U32.to_nat(extent),U32.to_nat(unit_cells),extent_positive,unit_positive)))
      carried_folds(~Block,~Model,~Element,~Start,~State,~Value,~Final,~represent,~cell,~operation,~begin,~begin_value,~fold,~turn,~turn_value,~Fill,~vacant,~vacant_model,~vacant_same,~chain_same,~Context,~value,limits,(outer * extent * unit_cells : U32),prefix,outer,extent,unit_cells,Grid.piece_units(unit_cells,extent,target),Grid.divided_up(extent,Grid.piece_units(unit_cells,extent,target)),shape,count_word,length,inner,contexts,bytes,operations,fill,workers,context,pieces,((length * inner : U32) / U32.max(Layout.cells_per_outer(Grid.grid_of_units(Grid.Units{prefix,outer,extent,unit_cells},target)),1) : U32),lines,outer_group,lines_positive,length_group,filled,turns,inner_positive,inner_le,
        cap,cap_ok,count_room_word,total,total_word,source)

# The trailing blocks of the units a walk finds fold into the block folds of
# the tensor's values in order.
def found_folds(~Block: Type,~Model: Data,~Element: Data,~Start: Data,~State: Type,~Value: Data,~Final: Data,~represent: Model -> Block,~cell: Model -> U32 -> Element,~operation: Value -> Element -> Value,~begin: Start -> State,~begin_value: Start -> Value,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~turn: State -> Final,~turn_value: Value -> Final,~Fill: Data,~vacant: Fill -> Block,~vacant_model: Fill -> Model,~vacant_same: @+fill: Fill -> {vacant(fill) == represent(vacant_model(fill)) : Block},~chain_same: LineChainValues.ChainSame(~Block,~Model,~represent,~Start,~State,~Value,~Final,~begin,~begin_value,~fold,~LineChainValues.strided_model(~Model,~Element,~Value,~cell,~operation),~turn,~turn_value,~Fill,~vacant,~vacant_model),~Context: Data,~value: Context -> U32 -> Element,+limits: Machine.Limits,+found: Grid.Units,+target: U32,+length: U32,+inner: U32,+contexts: List<&2,Start>,+bytes: F32,+operations: F32,+fill: Fill,+workers: U32,+context: Context,-pieces: Partition.Tree<&1,Block,Unit>,
  +total: Nat,+facts: GridRanges.UnitsFacts<limits,found,total>,+lines: Nat,+trailing: Trailing<found,lines,U32.to_nat((length * inner : U32))>,
  +lines_exact: {total == Nat.mul(lines,U32.to_nat((length * inner : U32))) : Nat},+lines_positive: {Nat.is_lt(0n,lines) == True{} : Bool},+total_room: {Nat.is_le(total,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +turns: Turns<Start,contexts,inner>,+inner_positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},+inner_le: {Nat.is_le(U32.to_nat(inner),U32.to_nat((length * inner : U32))) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(total),4n),U32.to_nat(cap)) == True{} : Bool},
  +source: Streamed<Block,Model,Element,represent,cell,pieces,Pieces.ranges_of_grid(Grid.grid_of_units(found,target)),values_from(~Element,~Context,~value,context,total,0n)>) ->
  LinesFolded(~Block,~Fill,~Start,~State,~Final,~begin,~fold,~turn,~vacant,contexts,LineChain.Line{length,inner},((length * inner : U32) / U32.max(Layout.cells_per_outer(Grid.grid_of_units(found,target)),1) : U32),bytes,operations,Grid.grid_of_units(found,target),fill,workers,pieces,block_folds(~Element,~Value,~Final,~operation,~turn_value,lines,length,inner,values_from(~Element,~Context,~value,context,total,0n),block_starts(~Start,~Value,~begin_value,contexts,inner))):
  GridRanges.UnitsFacts{+prefix,+outer,+extent,+unit_cells,+exact,+outer_positive,+extent_positive,+unit_positive,+cells} = facts
  +room = Held.nat_le_moved(Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),total,U32.to_nat(Machine.capacity(limits)),cells,total_room)
  %Equal.sym(Grid.Units,found,Grid.Units{prefix,outer,extent,unit_cells},exact) : LinesFolded(~Block,~Fill,~Start,~State,~Final,~begin,~fold,~turn,~vacant,contexts,LineChain.Line{length,inner},((length * inner : U32) / U32.max(Layout.cells_per_outer(Grid.grid_of_units(_,target)),1) : U32),bytes,operations,Grid.grid_of_units(_,target),fill,workers,pieces,block_folds(~Element,~Value,~Final,~operation,~turn_value,lines,length,inner,values_from(~Element,~Context,~value,context,total,0n),block_starts(~Start,~Value,~begin_value,contexts,inner)))
  units_folds(~Block,~Model,~Element,~Start,~State,~Value,~Final,~represent,~cell,~operation,~begin,~begin_value,~fold,~turn,~turn_value,~Fill,~vacant,~vacant_model,~vacant_same,~chain_same,~Context,~value,limits,prefix,outer,extent,unit_cells,target,outer_positive,extent_positive,unit_positive,room,length,inner,contexts,bytes,operations,fill,workers,context,pieces,total,Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(extent)),U32.to_nat(unit_cells)),total,cells),lines,lines_exact,lines_positive,
    {%exact : Trailing<_,lines,U32.to_nat((length * inner : U32))> trailing : Trailing<Grid.Units{prefix,outer,extent,unit_cells},lines,U32.to_nat((length * inner : U32))>},turns,inner_positive,inner_le,cap,cap_ok,count_room,
    {%exact : Streamed<Block,Model,Element,represent,cell,pieces,Pieces.ranges_of_grid(Grid.grid_of_units(_,target)),values_from(~Element,~Context,~value,context,total,0n)> source : Streamed<Block,Model,Element,represent,cell,pieces,Pieces.ranges_of_grid(Grid.grid_of_units(Grid.Units{prefix,outer,extent,unit_cells},target)),values_from(~Element,~Context,~value,context,total,0n)>})

# The trailing axes of a shape, all but its kept leading axes, blocks of
# length lines of inner cells, fold on the grid of the shape into the block
# folds of the tensor's values in order, every line begun from the contexts in
# turn.
def trailing_folded(~Block: Type,~Model: Data,~Element: Data,~Start: Data,~State: Type,~Value: Data,~Final: Data,~represent: Model -> Block,~cell: Model -> U32 -> Element,~operation: Value -> Element -> Value,~begin: Start -> State,~begin_value: Start -> Value,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~turn: State -> Final,~turn_value: Value -> Final,~Fill: Data,~vacant: Fill -> Block,~vacant_model: Fill -> Model,~vacant_same: @+fill: Fill -> {vacant(fill) == represent(vacant_model(fill)) : Block},~chain_same: LineChainValues.ChainSame(~Block,~Model,~represent,~Start,~State,~Value,~Final,~begin,~begin_value,~fold,~LineChainValues.strided_model(~Model,~Element,~Value,~cell,~operation),~turn,~turn_value,~Fill,~vacant,~vacant_model),~Context: Data,~value: Context -> U32 -> Element,+limits: Machine.Limits,+shape: List<&2,U32>,+kept: Nat,+target: U32,+length: U32,+inner: U32,+contexts: List<&2,Start>,+bytes: F32,+operations: F32,+fill: Fill,+workers: U32,+context: Context,-pieces: Partition.Tree<&1,Block,Unit>,
  +room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+length_exact: {U32.to_nat((length * inner : U32)) == ShapeLaws.elements(List.drop(&2,U32,shape,kept)) : Nat},
  +turns: Turns<Start,contexts,inner>,+inner_positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},+inner_le: {Nat.is_le(U32.to_nat(inner),U32.to_nat((length * inner : U32))) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},
  +source: Streamed<Block,Model,Element,represent,cell,pieces,Pieces.ranges_of_grid(Grid.grid(shape,target)),values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n)>) ->
  LinesFolded(~Block,~Fill,~Start,~State,~Final,~begin,~fold,~turn,~vacant,contexts,LineChain.Line{length,inner},((length * inner : U32) / U32.max(Layout.cells_per_outer(Grid.grid(shape,target)),1) : U32),bytes,operations,Grid.grid(shape,target),fill,workers,pieces,block_folds(~Element,~Value,~Final,~operation,~turn_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,inner,values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n),block_starts(~Start,~Value,~begin_value,contexts,inner))):
  +total_room = Held.nat_le_moved(Nat.add(ShapeLaws.elements(shape),0n),ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits)),Addition.zero_right(ShapeLaws.elements(shape)),room)
  +facts = {%Addition.zero_right(ShapeLaws.elements(shape)) : GridRanges.UnitsFacts<limits,Grid.units_from(shape,Grid.strides(shape),U32.is_le(Grid.product(shape),target),0,1,1,target),_>
    GridRanges.units_facts(limits,shape,U32.is_le(Grid.product(shape),target),0,1,1,target,total_room,{==},{==},nonempty) : GridRanges.UnitsFacts<limits,Grid.units_from(shape,Grid.strides(shape),U32.is_le(Grid.product(shape),target),0,1,1,target),ShapeLaws.elements(shape)>}
  +walked = kept_walk(limits,shape,U32.is_le(Grid.product(shape),target),0,1,1,target,kept,1n,{==},total_room,{==},{==},nonempty)
  +lined = {%LineChainValues.one_times(ShapeLaws.elements(List.take(&2,U32,shape,kept))) : Trailing<Grid.units_from(shape,Grid.strides(shape),U32.is_le(Grid.product(shape),target),0,1,1,target),_,ShapeLaws.elements(List.drop(&2,U32,shape,kept))> walked : Trailing<Grid.units_from(shape,Grid.strides(shape),U32.is_le(Grid.product(shape),target),0,1,1,target),ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept))>}
  +trailing = {%Equal.sym(Nat,U32.to_nat((length * inner : U32)),ShapeLaws.elements(List.drop(&2,U32,shape,kept)),length_exact) : Trailing<Grid.units_from(shape,Grid.strides(shape),U32.is_le(Grid.product(shape),target),0,1,1,target),ShapeLaws.elements(List.take(&2,U32,shape,kept)),_> lined : Trailing<Grid.units_from(shape,Grid.strides(shape),U32.is_le(Grid.product(shape),target),0,1,1,target),ShapeLaws.elements(List.take(&2,U32,shape,kept)),U32.to_nat((length * inner : U32))>}
  +split = CountCompletion.elements_split(shape,kept)
  +lines_exact = Equal.trans(Nat,ShapeLaws.elements(shape),Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept))),Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,kept)),U32.to_nat((length * inner : U32))),split,Equal.cong(Nat,Nat,number => Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,kept)),number),ShapeLaws.elements(List.drop(&2,U32,shape,kept)),U32.to_nat((length * inner : U32)),Equal.sym(Nat,U32.to_nat((length * inner : U32)),ShapeLaws.elements(List.drop(&2,U32,shape,kept)),length_exact)))
  +lines_positive = RowMajor.left_factor_positive(ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept)),Held.nat_lt_moved(0n,Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept))),ShapeLaws.elements(shape),Equal.sym(Nat,ShapeLaws.elements(shape),Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept))),split),nonempty))
  %Equal.sym(Grid.Grid,Grid.grid(shape,target),Grid.grid_of_units(Grid.units_from(shape,Grid.strides(shape),U32.is_le(Grid.product(shape),target),0,1,1,target),target),GridPieces.grid_units(shape,target)) :
    LinesFolded(~Block,~Fill,~Start,~State,~Final,~begin,~fold,~turn,~vacant,contexts,LineChain.Line{length,inner},((length * inner : U32) / U32.max(Layout.cells_per_outer(_),1) : U32),bytes,operations,_,fill,workers,pieces,block_folds(~Element,~Value,~Final,~operation,~turn_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,inner,values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n),block_starts(~Start,~Value,~begin_value,contexts,inner)))
  found_folds(~Block,~Model,~Element,~Start,~State,~Value,~Final,~represent,~cell,~operation,~begin,~begin_value,~fold,~turn,~turn_value,~Fill,~vacant,~vacant_model,~vacant_same,~chain_same,~Context,~value,limits,Grid.units_from(shape,Grid.strides(shape),U32.is_le(Grid.product(shape),target),0,1,1,target),target,length,inner,contexts,bytes,operations,fill,workers,context,pieces,ShapeLaws.elements(shape),facts,ShapeLaws.elements(List.take(&2,U32,shape,kept)),trailing,lines_exact,lines_positive,room,turns,inner_positive,inner_le,cap,cap_ok,count_room,{%GridPieces.grid_units(shape,target) : Streamed<Block,Model,Element,represent,cell,pieces,Pieces.ranges_of_grid(_),values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n)> source : Streamed<Block,Model,Element,represent,cell,pieces,Pieces.ranges_of_grid(Grid.grid_of_units(Grid.units_from(shape,Grid.strides(shape),U32.is_le(Grid.product(shape),target),0,1,1,target),target)),values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n)>})
def trailing_folds(~Block: Type,~Model: Data,~Element: Data,~Start: Data,~State: Type,~Value: Data,~Final: Data,~represent: Model -> Block,~cell: Model -> U32 -> Element,~operation: Value -> Element -> Value,~begin: Start -> State,~begin_value: Start -> Value,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~turn: State -> Final,~turn_value: Value -> Final,~Fill: Data,~vacant: Fill -> Block,~vacant_model: Fill -> Model,~vacant_same: @+fill: Fill -> {vacant(fill) == represent(vacant_model(fill)) : Block},~chain_same: LineChainValues.ChainSame(~Block,~Model,~represent,~Start,~State,~Value,~Final,~begin,~begin_value,~fold,~LineChainValues.strided_model(~Model,~Element,~Value,~cell,~operation),~turn,~turn_value,~Fill,~vacant,~vacant_model),~Context: Data,~value: Context -> U32 -> Element,+limits: Machine.Limits,+shape: List<&2,U32>,+kept: Nat,+target: U32,+length: U32,+inner: U32,+contexts: List<&2,Start>,+bytes: F32,+operations: F32,+fill: Fill,+workers: U32,+context: Context,-pieces: Partition.Tree<&1,Block,Unit>,
  +room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+length_exact: {U32.to_nat((length * inner : U32)) == ShapeLaws.elements(List.drop(&2,U32,shape,kept)) : Nat},
  +turns: Turns<Start,contexts,inner>,+inner_positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},+inner_le: {Nat.is_le(U32.to_nat(inner),U32.to_nat((length * inner : U32))) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},
  +source: Streamed<Block,Model,Element,represent,cell,pieces,Pieces.ranges_of_grid(Grid.grid(shape,target)),values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n)>) ->
  {Pair.snd(Partition.Tree<&1,Block,Unit>,List<&2,Final>,LineChain.folds(~Block,~Fill,~Start,~State,~Final,~begin,~fold,~turn,~vacant,contexts,LineChain.Line{length,inner}, ((length * inner : U32) / U32.max(Layout.cells_per_outer(Grid.grid(shape,target)),1) : U32),bytes,operations,Grid.grid(shape,target),fill,workers,pieces)) ==
    block_folds(~Element,~Value,~Final,~operation,~turn_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,inner,values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n),block_starts(~Start,~Value,~begin_value,contexts,inner)) : List<&2,Final>}:
  folded_states(~Block,~Fill,~Start,~State,~Final,~begin,~fold,~turn,~vacant,contexts,LineChain.Line{length,inner},((length * inner : U32) / U32.max(Layout.cells_per_outer(Grid.grid(shape,target)),1) : U32),bytes,operations,Grid.grid(shape,target),fill,workers,pieces,block_folds(~Element,~Value,~Final,~operation,~turn_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,inner,values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n),block_starts(~Start,~Value,~begin_value,contexts,inner)),trailing_folded(~Block,~Model,~Element,~Start,~State,~Value,~Final,~represent,~cell,~operation,~begin,~begin_value,~fold,~turn,~turn_value,~Fill,~vacant,~vacant_model,~vacant_same,~chain_same,~Context,~value,limits,shape,kept,target,length,inner,contexts,bytes,operations,fill,workers,context,pieces,room,nonempty,length_exact,turns,inner_positive,inner_le,cap,cap_ok,count_room,source))

# The pieces of the trailing axes come back as they were.
def trailing_kept(~Block: Type,~Model: Data,~Element: Data,~Start: Data,~State: Type,~Value: Data,~Final: Data,~represent: Model -> Block,~cell: Model -> U32 -> Element,~operation: Value -> Element -> Value,~begin: Start -> State,~begin_value: Start -> Value,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~turn: State -> Final,~turn_value: Value -> Final,~Fill: Data,~vacant: Fill -> Block,~vacant_model: Fill -> Model,~vacant_same: @+fill: Fill -> {vacant(fill) == represent(vacant_model(fill)) : Block},~chain_same: LineChainValues.ChainSame(~Block,~Model,~represent,~Start,~State,~Value,~Final,~begin,~begin_value,~fold,~LineChainValues.strided_model(~Model,~Element,~Value,~cell,~operation),~turn,~turn_value,~Fill,~vacant,~vacant_model),~Context: Data,~value: Context -> U32 -> Element,+limits: Machine.Limits,+shape: List<&2,U32>,+kept: Nat,+target: U32,+length: U32,+inner: U32,+contexts: List<&2,Start>,+bytes: F32,+operations: F32,+fill: Fill,+workers: U32,+context: Context,-pieces: Partition.Tree<&1,Block,Unit>,
  +room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+length_exact: {U32.to_nat((length * inner : U32)) == ShapeLaws.elements(List.drop(&2,U32,shape,kept)) : Nat},
  +turns: Turns<Start,contexts,inner>,+inner_positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},+inner_le: {Nat.is_le(U32.to_nat(inner),U32.to_nat((length * inner : U32))) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},
  +source: Streamed<Block,Model,Element,represent,cell,pieces,Pieces.ranges_of_grid(Grid.grid(shape,target)),values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n)>) ->
  {Pieces.blocks_regrouped(~Block,~Fill,~vacant,Pieces.ranges_of_grid(Grid.grid(shape,target)),fill,Pair.fst(Partition.Tree<&1,Block,Unit>,List<&2,Final>,LineChain.folds(~Block,~Fill,~Start,~State,~Final,~begin,~fold,~turn,~vacant,contexts,LineChain.Line{length,inner},((length * inner : U32) / U32.max(Layout.cells_per_outer(Grid.grid(shape,target)),1) : U32),bytes,operations,Grid.grid(shape,target),fill,workers,pieces))) == pieces : Partition.Tree<&1,Block,Unit>}:
  folded_kept(~Block,~Fill,~Start,~State,~Final,~begin,~fold,~turn,~vacant,contexts,LineChain.Line{length,inner},((length * inner : U32) / U32.max(Layout.cells_per_outer(Grid.grid(shape,target)),1) : U32),bytes,operations,Grid.grid(shape,target),fill,workers,pieces,block_folds(~Element,~Value,~Final,~operation,~turn_value,ShapeLaws.elements(List.take(&2,U32,shape,kept)),length,inner,values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n),block_starts(~Start,~Value,~begin_value,contexts,inner)),trailing_folded(~Block,~Model,~Element,~Start,~State,~Value,~Final,~represent,~cell,~operation,~begin,~begin_value,~fold,~turn,~turn_value,~Fill,~vacant,~vacant_model,~vacant_same,~chain_same,~Context,~value,limits,shape,kept,target,length,inner,contexts,bytes,operations,fill,workers,context,pieces,room,nonempty,length_exact,turns,inner_positive,inner_le,cap,cap_ok,count_room,source))

# The trailing axes of a shape, all but its kept leading axes, fold on the grid
# of the shape the lines of the tensor's values in order.
# A line of a reduction begins from its accumulator as it is.
def unchanged_begins(~Accumulator: Data,+context: Accumulator) ->
  {LineChain.unchanged(~Accumulator,context) == LineChain.unchanged(~Accumulator,LineChain.unchanged(~Accumulator,context)) : Accumulator}:
  {==}

# A line of a reduction ends in the finish of its accumulator.
def finish_turns(~Accumulator: Data,~Element: Data,~finish: Accumulator -> Element,+value: Accumulator) ->
  {finish(LineChain.unchanged(~Accumulator,value)) == finish(value) : Element}:
  {==}

def trailing_values(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,~Context: Data,~value: Context -> U32 -> Element,+limits: Machine.Limits,+shape: List<&2,U32>,+kept: Nat,+target: U32,+length: U32,+initial: Accumulator,+bytes: F32,+operations: F32,+fill: Element,+workers: U32,+context: Context,+model: PiecesModel.Model<Element>,
  +room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+length_exact: {U32.to_nat(length) == ShapeLaws.elements(List.drop(&2,U32,shape,kept)) : Nat},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},
  +canonical: {PiecesModel.content_layout(~Element,PiecesModel.content(~Element,model)) == Pieces.ranges_of_grid(Grid.grid(shape,target)) : Pieces.Ranges},
  +coverage: PiecesPlan.Coverage(PiecesModel.content_layout(~Element,PiecesModel.content(~Element,model)),0n,ShapeLaws.elements(shape)),+values: Content.Values(~Element,~Context,~value,context,PiecesModel.content(~Element,model))) ->
  {Pair.snd(Partition.Tree<&1,Storage.Buffer<Element>,Unit>,List<&2,Element>,LineChain.folds(~Storage.Buffer<Element>,~Element,~Accumulator,~Accumulator,~Element,~LineChain.unchanged(~Accumulator),
      ~Operations.fold_range(~Element,~Accumulator,~operation),~finish,~Pieces.vacant_buffer(~Element),[initial],LineChain.Line{length,1},(length / U32.max(Layout.cells_per_outer(Grid.grid(shape,target)),1) : U32),bytes,operations,
      Grid.grid(shape,target),fill,workers,PiecesModel.pack(~Element,model))) ==
    LineChainValues.turned_values(~Accumulator,~Element,~finish,chunk_folds(~Element,~Accumulator,~operation,ShapeLaws.elements(List.take(&2,U32,shape,kept)),U32.to_nat((length * 1 : U32)),values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n),initial)) : List<&2,Element>}:
  +split = CountCompletion.elements_split(shape,kept)
  +length_word = Equal.trans(Nat,U32.to_nat((length * 1 : U32)),U32.to_nat(length),ShapeLaws.elements(List.drop(&2,U32,shape,kept)),Equal.cong(U32,Nat,word => U32.to_nat(word),(length * 1 : U32),length,Arithmetic.multiply_one(length)),length_exact)
  +drop_positive = GridRanges.right_factor_positive(ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept)),Held.nat_lt_moved(0n,Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept))),ShapeLaws.elements(shape),Equal.sym(Nat,ShapeLaws.elements(shape),Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,kept)),ShapeLaws.elements(List.drop(&2,U32,shape,kept))),split),nonempty))
  +inner_le = Held.nat_le_right_moved(1n,U32.to_nat((length * 1 : U32)),ShapeLaws.elements(List.drop(&2,U32,shape,kept)),length_word,Order.positive_one(ShapeLaws.elements(List.drop(&2,U32,shape,kept)),drop_positive))
  +list = values_from(~Element,~Context,~value,context,ShapeLaws.elements(shape),0n)
  +lines = ShapeLaws.elements(List.take(&2,U32,shape,kept))
  +folded = trailing_folds(~Storage.Buffer<Element>,~Buffer.Model<Element>,~Element,~Accumulator,~Accumulator,~Accumulator,~Element,~Buffer.pack(~Element),~cell(~Element),~operation,~LineChain.unchanged(~Accumulator),~LineChain.unchanged(~Accumulator),~Operations.fold_range(~Element,~Accumulator,~operation),~finish,~finish,~Element,~Pieces.vacant_buffer(~Element),~vacant_model(~Element),~vacant_same(~Element),~LineChainValues.folds_same(~Storage.Buffer<Element>,~Buffer.Model<Element>,~Buffer.pack(~Element),~Accumulator,~Accumulator,~Accumulator,~LineChain.unchanged(~Accumulator),~Element,~LineChain.unchanged(~Accumulator),~LineChain.unchanged(~Accumulator),~Operations.fold_range(~Element,~Accumulator,~operation),~LineChainValues.strided_model(~Buffer.Model<Element>,~Element,~Accumulator,~cell(~Element),~operation),~finish,~finish,~range_same(~Element,~Accumulator,~operation),~unchanged_begins(~Accumulator),~finish_turns(~Accumulator,~Element,~finish),~Element,~Pieces.vacant_buffer(~Element),~vacant_model(~Element),~vacant_same(~Element)),~Context,~value,limits,shape,kept,target,length,1,[initial],bytes,operations,fill,workers,context,PiecesModel.pack(~Element,model),room,nonempty,length_word,
    {Turns{1n,{==},{==}} : Turns<Accumulator,[initial],1>},{==},inner_le,cap,cap_ok,count_room,model_streamed(~Element,~Context,~value,context,model,Pieces.ranges_of_grid(Grid.grid(shape,target)),ShapeLaws.elements(shape),canonical,coverage,values))
  %Arithmetic.multiply_one(length) : {Pair.snd(Partition.Tree<&1,Storage.Buffer<Element>,Unit>,List<&2,Element>,LineChain.folds(~Storage.Buffer<Element>,~Element,~Accumulator,~Accumulator,~Element,~LineChain.unchanged(~Accumulator),
      ~Operations.fold_range(~Element,~Accumulator,~operation),~finish,~Pieces.vacant_buffer(~Element),[initial],LineChain.Line{length,1},(_ / U32.max(Layout.cells_per_outer(Grid.grid(shape,target)),1) : U32),bytes,operations,
      Grid.grid(shape,target),fill,workers,PiecesModel.pack(~Element,model))) ==
    LineChainValues.turned_values(~Accumulator,~Element,~finish,chunk_folds(~Element,~Accumulator,~operation,lines,U32.to_nat((length * 1 : U32)),list,initial)) : List<&2,Element>}
  Equal.trans(List<&2,Element>,Pair.snd(Partition.Tree<&1,Storage.Buffer<Element>,Unit>,List<&2,Element>,LineChain.folds(~Storage.Buffer<Element>,~Element,~Accumulator,~Accumulator,~Element,~LineChain.unchanged(~Accumulator),
      ~Operations.fold_range(~Element,~Accumulator,~operation),~finish,~Pieces.vacant_buffer(~Element),[initial],LineChain.Line{length,1},((length * 1 : U32) / U32.max(Layout.cells_per_outer(Grid.grid(shape,target)),1) : U32),bytes,operations,
      Grid.grid(shape,target),fill,workers,PiecesModel.pack(~Element,model))),
    block_folds(~Element,~Accumulator,~Element,~operation,~finish,lines,length,1,list,[initial]),LineChainValues.turned_values(~Accumulator,~Element,~finish,chunk_folds(~Element,~Accumulator,~operation,lines,U32.to_nat((length * 1 : U32)),list,initial)),
    folded,single_blocks(~Element,~Accumulator,~Element,~operation,~finish,lines,length,list,initial))

# ---- The contract of a reduction of trailing axes ----

# The fold of count values of a reader from position on.
def line_fold(~Element: Data,~Accumulator: Data,~step: Accumulator -> Element -> Accumulator,~Context: Data,~value: Context -> U32 -> Element,+context: Context,count: Nat,+position: Nat,+state: Accumulator) -> Accumulator:
  match count:
    case 0n: state
    case 1n+rest: line_fold(~Element,~Accumulator,~step,~Context,~value,context,rest,1n+position,step(state,value(context,U32.from_nat(position))))

def taken_fold(~Element: Data,~Accumulator: Data,~step: Accumulator -> Element -> Accumulator,~Context: Data,~value: Context -> U32 -> Element,+context: Context,count: Nat,+position: Nat,+state: Accumulator) ->
  {LineChainValues.taken_list(~Element,~Accumulator,~step,values_from(~Element,~Context,~value,context,count,position),0n,1n,state) == line_fold(~Element,~Accumulator,~step,~Context,~value,context,count,position,state) : Accumulator}:
  match count:
    case 0n: {==}
    case 1n+ +rest: taken_fold(~Element,~Accumulator,~step,~Context,~value,context,rest,1n+position,step(state,value(context,U32.from_nat(position))))

# The folds of lines of size values each, from position on.
def line_folds(~Element: Data,~Accumulator: Data,~step: Accumulator -> Element -> Accumulator,~Context: Data,~value: Context -> U32 -> Element,+context: Context,lines: Nat,+size: Nat,+position: Nat,+state: Accumulator) -> List<&2,Accumulator>:
  match lines:
    case 0n: []
    case 1n+rest: line_fold(~Element,~Accumulator,~step,~Context,~value,context,size,position,state) <> line_folds(~Element,~Accumulator,~step,~Context,~value,context,rest,size,Nat.add(position,size),state)

def values_take(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,count: Nat,more: Nat,+start: Nat) ->
  {List.take(&2,Element,values_from(~Element,~Context,~value,context,Nat.add(count,more),start),count) == values_from(~Element,~Context,~value,context,count,start) : List<&2,Element>}:
  match count more:
    case 0n 0n: {==}
    case 0n 1n+rest: {==}
    case 1n+ +previous more: Equal.cong(List<&2,Element>,List<&2,Element>,tail => value(context,U32.from_nat(start)) <> tail,
      List.take(&2,Element,values_from(~Element,~Context,~value,context,Nat.add(previous,more),1n+start),previous),values_from(~Element,~Context,~value,context,previous,1n+start),values_take(~Element,~Context,~value,context,previous,more,1n+start))

def values_drop(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,count: Nat,more: Nat,+start: Nat) ->
  {List.drop(&2,Element,values_from(~Element,~Context,~value,context,Nat.add(count,more),start),count) == values_from(~Element,~Context,~value,context,more,Nat.add(start,count)) : List<&2,Element>}:
  match count more:
    case 0n 0n: {==}
    case 0n 1n+ +rest:
      %Equal.sym(Nat,Nat.add(start,0n),start,Addition.zero_right(start)) : {values_from(~Element,~Context,~value,context,1n+rest,start) == values_from(~Element,~Context,~value,context,1n+rest,_) : List<&2,Element>}
      {==}
    case 1n+ +previous +more:
      %Equal.sym(Nat,Nat.add(start,1n+previous),1n+Nat.add(start,previous),Addition.successor_right(start,previous)) : {List.drop(&2,Element,values_from(~Element,~Context,~value,context,Nat.add(previous,more),1n+start),previous) == values_from(~Element,~Context,~value,context,more,_) : List<&2,Element>}
      values_drop(~Element,~Context,~value,context,previous,more,1n+start)

# Chunks of size values of a reader fold as its lines.
def chunk_lines(~Element: Data,~Accumulator: Data,~step: Accumulator -> Element -> Accumulator,~Context: Data,~value: Context -> U32 -> Element,+context: Context,lines: Nat,+size: Nat,+position: Nat,+state: Accumulator) ->
  {chunk_folds(~Element,~Accumulator,~step,lines,size,values_from(~Element,~Context,~value,context,Nat.mul(lines,size),position),state) == line_folds(~Element,~Accumulator,~step,~Context,~value,context,lines,size,position,state) : List<&2,Accumulator>}:
  match lines:
    case 0n: {==}
    case 1n+ +rest:
      +whole = values_from(~Element,~Context,~value,context,Nat.add(size,Nat.mul(rest,size)),position)
      %Equal.sym(List<&2,Element>,List.take(&2,Element,whole,size),values_from(~Element,~Context,~value,context,size,position),values_take(~Element,~Context,~value,context,size,Nat.mul(rest,size),position)) :
        {LineChainValues.taken_list(~Element,~Accumulator,~step,_,0n,1n,state) <> chunk_folds(~Element,~Accumulator,~step,rest,size,List.drop(&2,Element,whole,size),state) == line_folds(~Element,~Accumulator,~step,~Context,~value,context,1n+rest,size,position,state) : List<&2,Accumulator>}
      %Equal.sym(List<&2,Element>,List.drop(&2,Element,whole,size),values_from(~Element,~Context,~value,context,Nat.mul(rest,size),Nat.add(position,size)),values_drop(~Element,~Context,~value,context,size,Nat.mul(rest,size),position)) :
        {LineChainValues.taken_list(~Element,~Accumulator,~step,values_from(~Element,~Context,~value,context,size,position),0n,1n,state) <> chunk_folds(~Element,~Accumulator,~step,rest,size,_,state) == line_folds(~Element,~Accumulator,~step,~Context,~value,context,1n+rest,size,position,state) : List<&2,Accumulator>}
      %Equal.sym(Accumulator,LineChainValues.taken_list(~Element,~Accumulator,~step,values_from(~Element,~Context,~value,context,size,position),0n,1n,state),line_fold(~Element,~Accumulator,~step,~Context,~value,context,size,position,state),taken_fold(~Element,~Accumulator,~step,~Context,~value,context,size,position,state)) :
        {_ <> chunk_folds(~Element,~Accumulator,~step,rest,size,values_from(~Element,~Context,~value,context,Nat.mul(rest,size),Nat.add(position,size)),state) == line_folds(~Element,~Accumulator,~step,~Context,~value,context,1n+rest,size,position,state) : List<&2,Accumulator>}
      %Equal.sym(List<&2,Accumulator>,chunk_folds(~Element,~Accumulator,~step,rest,size,values_from(~Element,~Context,~value,context,Nat.mul(rest,size),Nat.add(position,size)),state),line_folds(~Element,~Accumulator,~step,~Context,~value,context,rest,size,Nat.add(position,size),state),
          chunk_lines(~Element,~Accumulator,~step,~Context,~value,context,rest,size,Nat.add(position,size),state)) :
        {line_fold(~Element,~Accumulator,~step,~Context,~value,context,size,position,state) <> _ == line_folds(~Element,~Accumulator,~step,~Context,~value,context,1n+rest,size,position,state) : List<&2,Accumulator>}
      {==}

# The finish of line index of the folds.
def line_at(~Element: Data,~Accumulator: Data,~step: Accumulator -> Element -> Accumulator,~Context: Data,~value: Context -> U32 -> Element,~finish: Accumulator -> Element,+context: Context,lines: Nat,+size: Nat,+position: Nat,+state: Accumulator,index: Nat,+fill: Element,
  +inside: {Nat.is_lt(index,lines) == True{} : Bool}) ->
  {ListReadback.at(~Element,fill,index,LineChainValues.turned_values(~Accumulator,~Element,~finish,line_folds(~Element,~Accumulator,~step,~Context,~value,context,lines,size,position,state))) ==
    finish(line_fold(~Element,~Accumulator,~step,~Context,~value,context,size,Nat.add(position,Nat.mul(index,size)),state)) : Element}:
  match lines index:
    case 0n +index: Empty.absurd({ListReadback.at(~Element,fill,index,[]) == finish(line_fold(~Element,~Accumulator,~step,~Context,~value,context,size,Nat.add(position,Nat.mul(index,size)),state)) : Element},
      WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(index,0n),True{},Equal.sym(Bool,Nat.is_lt(index,0n),False{},Order.not_below(0n,index,Order.zero_le(index))),inside)))
    case 1n+ +rest 0n:
      %Equal.sym(Nat,Nat.add(position,0n),position,Addition.zero_right(position)) : {finish(line_fold(~Element,~Accumulator,~step,~Context,~value,context,size,position,state)) == finish(line_fold(~Element,~Accumulator,~step,~Context,~value,context,size,_,state)) : Element}
      {==}
    case 1n+ +rest 1n+ +previous:
      %Addition.associative(position,size,Nat.mul(previous,size)) : {ListReadback.at(~Element,fill,previous,LineChainValues.turned_values(~Accumulator,~Element,~finish,line_folds(~Element,~Accumulator,~step,~Context,~value,context,rest,size,Nat.add(position,size),state))) ==
        finish(line_fold(~Element,~Accumulator,~step,~Context,~value,context,size,_,state)) : Element}
      line_at(~Element,~Accumulator,~step,~Context,~value,~finish,context,rest,size,Nat.add(position,size),state,previous,fill,inside)

# What a reduction of trailing axes holds: at index k of the kept axes, the
# finish of the fold from the initial accumulator of the length values of
# the reader from k * length on.
type Lines<-Accumulator: Data,-Logical: Data> is Data:
  Lines{initial: Accumulator,logical: Logical,length: Nat}

def line_value(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,~Logical: Data,~original: Logical -> U32 -> Element,lines: Lines<Accumulator,Logical>,index: U32) -> Element:
  Lines{+initial,+logical,+length} = lines
  finish(line_fold(~Element,~Accumulator,~operation,~Logical,~original,logical,length,Nat.mul(U32.to_nat(index),length),initial))

def trailing_agrees(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+initial: Accumulator,+fill: Element,+lines: Nat,+size: Nat,+length: Nat,+same: {size == length : Nat},
  +outputs: List<&2,U32>,+outputs_exact: {ShapeLaws.elements(outputs) == lines : Nat},+count: U32,+counted: Count.Count<ShapeLaws.elements(outputs),count>,+index: Nat,
  +inside: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {ListReadback.at(~Element,fill,index,LineChainValues.turned_values(~Accumulator,~Element,~finish,line_folds(~Element,~Accumulator,~operation,~Logical,~original,logical,lines,size,0n,initial))) ==
    line_value(~Element,~Accumulator,~operation,~finish,~Logical,~original,Lines{initial,logical,length},U32.from_nat(index)) : Element}:
  +below = Held.nat_lt_moved(index,lines,U32.to_nat(count),
    Equal.trans(Nat,lines,ShapeLaws.elements(outputs),U32.to_nat(count),Equal.sym(Nat,ShapeLaws.elements(outputs),lines,outputs_exact),Count.count_exact(ShapeLaws.elements(outputs),count,counted)),
    inside)
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,FullRange.roundtrip(index,count,Bounds.strict_le(index,U32.to_nat(count),inside))) :
    {ListReadback.at(~Element,fill,index,LineChainValues.turned_values(~Accumulator,~Element,~finish,line_folds(~Element,~Accumulator,~operation,~Logical,~original,logical,lines,size,0n,initial))) ==
      finish(line_fold(~Element,~Accumulator,~operation,~Logical,~original,logical,length,Nat.mul(_,length),initial)) : Element}
  %same : {ListReadback.at(~Element,fill,index,LineChainValues.turned_values(~Accumulator,~Element,~finish,line_folds(~Element,~Accumulator,~operation,~Logical,~original,logical,lines,size,0n,initial))) ==
      finish(line_fold(~Element,~Accumulator,~operation,~Logical,~original,logical,_,Nat.mul(index,_),initial)) : Element}
  line_at(~Element,~Accumulator,~operation,~Logical,~original,~finish,logical,lines,size,0n,initial,index,fill,below)

# The turned folds are as many as the folds.
def turned_length(~Element: Data,~Accumulator: Data,~finish: Accumulator -> Element,values: List<&2,Accumulator>) ->
  {List.length(&2,Element,LineChainValues.turned_values(~Accumulator,~Element,~finish,values)) == List.length(&2,Accumulator,values) : Nat}:
  match values:
    case []: {==}
    case value <> +rest:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Element,LineChainValues.turned_values(~Accumulator,~Element,~finish,rest)),List.length(&2,Accumulator,rest),
        turned_length(~Element,~Accumulator,~finish,rest))

# The folds of lines are one a line.
def line_folds_length(~Element: Data,~Accumulator: Data,~step: Accumulator -> Element -> Accumulator,~Context: Data,~value: Context -> U32 -> Element,+context: Context,lines: Nat,+size: Nat,+position: Nat,+state: Accumulator) ->
  {List.length(&2,Accumulator,line_folds(~Element,~Accumulator,~step,~Context,~value,context,lines,size,position,state)) == lines : Nat}:
  match lines:
    case 0n: {==}
    case 1n+ +rest:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Accumulator,line_folds(~Element,~Accumulator,~step,~Context,~value,context,rest,size,Nat.add(position,size),state)),rest,
        line_folds_length(~Element,~Accumulator,~step,~Context,~value,context,rest,size,Nat.add(position,size),state))

# The from_list of the finals of the lines holds the lines of the reader.
def trailing_listed(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+operations: F32,+length: U32,+initial: Accumulator,+model: PiecesModel.Model<Element>,+shape: List<&2,U32>,+kept: Nat,+outputs: List<&2,U32>,+fill: Element,+workers: U32,
  +folded: {Pair.snd(Pieces.Pieces(~Element),List<&2,Element>,LineChain.folds(~Storage.Buffer<Element>,~Element,~Accumulator,~Accumulator,~Element,~LineChain.unchanged(~Accumulator),~Operations.fold_range(~Element,~Accumulator,~operation),~finish,~Pieces.vacant_buffer(~Element),[initial],LineChain.Line{length,1},(length / U32.max(Layout.cells_per_outer(Tensors.grid_of(shape)),1) : U32),1.0,operations,Tensors.grid_of(shape),fill,workers,PiecesModel.pack(~Element,model))) == LineChainValues.turned_values(~Accumulator,~Element,~finish,line_folds(~Element,~Accumulator,~operation,~Logical,~original,logical,ShapeLaws.elements(List.take(&2,U32,shape,kept)),U32.to_nat((length * 1 : U32)),0n,initial)) : List<&2,Element>},
  +same: {U32.to_nat((length * 1 : U32)) == ShapeLaws.elements(List.drop(&2,U32,shape,kept)) : Nat},+outputs_exact: {ShapeLaws.elements(outputs) == ShapeLaws.elements(List.take(&2,U32,shape,kept)) : Nat},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(outputs)) == True{} : Bool},
  ready: {Contract.tensor_ready(~Element,Tensors.from_list(~Element,outputs,Pair.snd(Pieces.Pieces(~Element),List<&2,Element>,LineChain.folds(~Storage.Buffer<Element>,~Element,~Accumulator,~Accumulator,~Element,~LineChain.unchanged(~Accumulator),~Operations.fold_range(~Element,~Accumulator,~operation),~finish,~Pieces.vacant_buffer(~Element),[initial],LineChain.Line{length,1},(length / U32.max(Layout.cells_per_outer(Tensors.grid_of(shape)),1) : U32),1.0,operations,Tensors.grid_of(shape),fill,workers,PiecesModel.pack(~Element,model))),fill)) == True{} : Bool}) ->
  Global.Contract(~Element,~Lines<Accumulator,Logical>,~line_value(~Element,~Accumulator,~operation,~finish,~Logical,~original),Lines{initial,logical,ShapeLaws.elements(List.drop(&2,U32,shape,kept))},Tensors.from_list(~Element,outputs,Pair.snd(Pieces.Pieces(~Element),List<&2,Element>,LineChain.folds(~Storage.Buffer<Element>,~Element,~Accumulator,~Accumulator,~Element,~LineChain.unchanged(~Accumulator),~Operations.fold_range(~Element,~Accumulator,~operation),~finish,~Pieces.vacant_buffer(~Element),[initial],LineChain.Line{length,1},(length / U32.max(Layout.cells_per_outer(Tensors.grid_of(shape)),1) : U32),1.0,operations,Tensors.grid_of(shape),fill,workers,PiecesModel.pack(~Element,model))),fill)):
  Listed.from_list_read(~Element,~Lines<Accumulator,Logical>,~line_value(~Element,~Accumulator,~operation,~finish,~Logical,~original),Lines{initial,logical,ShapeLaws.elements(List.drop(&2,U32,shape,kept))},outputs,Pair.snd(Pieces.Pieces(~Element),List<&2,Element>,LineChain.folds(~Storage.Buffer<Element>,~Element,~Accumulator,~Accumulator,~Element,~LineChain.unchanged(~Accumulator),~Operations.fold_range(~Element,~Accumulator,~operation),~finish,~Pieces.vacant_buffer(~Element),[initial],LineChain.Line{length,1},(length / U32.max(Layout.cells_per_outer(Tensors.grid_of(shape)),1) : U32),1.0,operations,Tensors.grid_of(shape),fill,workers,PiecesModel.pack(~Element,model))),fill,
    count => counted => equal => index => inside => Equal.trans(Element,ListReadback.at(~Element,fill,index,Pair.snd(Pieces.Pieces(~Element),List<&2,Element>,LineChain.folds(~Storage.Buffer<Element>,~Element,~Accumulator,~Accumulator,~Element,~LineChain.unchanged(~Accumulator),~Operations.fold_range(~Element,~Accumulator,~operation),~finish,~Pieces.vacant_buffer(~Element),[initial],LineChain.Line{length,1},(length / U32.max(Layout.cells_per_outer(Tensors.grid_of(shape)),1) : U32),1.0,operations,Tensors.grid_of(shape),fill,workers,PiecesModel.pack(~Element,model)))),ListReadback.at(~Element,fill,index,LineChainValues.turned_values(~Accumulator,~Element,~finish,line_folds(~Element,~Accumulator,~operation,~Logical,~original,logical,ShapeLaws.elements(List.take(&2,U32,shape,kept)),U32.to_nat((length * 1 : U32)),0n,initial))),
      line_value(~Element,~Accumulator,~operation,~finish,~Logical,~original,Lines{initial,logical,ShapeLaws.elements(List.drop(&2,U32,shape,kept))},U32.from_nat(index)),
      Equal.cong(List<&2,Element>,Element,items => ListReadback.at(~Element,fill,index,items),Pair.snd(Pieces.Pieces(~Element),List<&2,Element>,LineChain.folds(~Storage.Buffer<Element>,~Element,~Accumulator,~Accumulator,~Element,~LineChain.unchanged(~Accumulator),~Operations.fold_range(~Element,~Accumulator,~operation),~finish,~Pieces.vacant_buffer(~Element),[initial],LineChain.Line{length,1},(length / U32.max(Layout.cells_per_outer(Tensors.grid_of(shape)),1) : U32),1.0,operations,Tensors.grid_of(shape),fill,workers,PiecesModel.pack(~Element,model))),LineChainValues.turned_values(~Accumulator,~Element,~finish,line_folds(~Element,~Accumulator,~operation,~Logical,~original,logical,ShapeLaws.elements(List.take(&2,U32,shape,kept)),U32.to_nat((length * 1 : U32)),0n,initial)),folded),
      trailing_agrees(~Element,~Accumulator,~operation,~finish,~Logical,~original,logical,initial,fill,ShapeLaws.elements(List.take(&2,U32,shape,kept)),U32.to_nat((length * 1 : U32)),ShapeLaws.elements(List.drop(&2,U32,shape,kept)),same,outputs,outputs_exact,count,counted,index,inside)),
    nonempty,ready)

# A worker count and from_list keep the shape they are given.
def workers_extents(~Element: Data,tensor: Tensors.Tensor<Element>,+count: U32) ->
  {Contract.extents(~Element,Tensors.with_workers(~Element,tensor,count)) == Contract.extents(~Element,tensor) : List<&2,U32>}:
  match tensor:
    case Tensors.Tensor{pieces,shape,fill,status,workers}: {==}

def trailing_extents(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,+operations: F32,+length: U32,
  +initial: Accumulator,pieces: Pieces.Pieces(~Element),+shape: List<&2,U32>,+outputs: List<&2,U32>,+fill: Element,+workers: U32) ->
  {Contract.extents(~Element,Layout.reduce_trailing(~Element,~Accumulator,~operation,~finish,operations,length,initial,pieces,shape,outputs,fill,workers)) == outputs : List<&2,U32>}:
  +values = Pair.snd(Pieces.Pieces(~Element),List<&2,Element>,
    LineChain.folds(~Storage.Buffer<Element>,~Element,~Accumulator,~Accumulator,~Element,~LineChain.unchanged(~Accumulator),~Operations.fold_range(~Element,~Accumulator,~operation),
      ~finish,~Pieces.vacant_buffer(~Element),[initial],LineChain.Line{length,1},(length / U32.max(Layout.cells_per_outer(Tensors.grid_of(shape)),1) : U32),1.0,operations,
      Tensors.grid_of(shape),fill,workers,pieces))
  Equal.trans(List<&2,U32>,Contract.extents(~Element,Tensors.with_workers(~Element,Tensors.from_list(~Element,outputs,values,fill),workers)),
    Contract.extents(~Element,Tensors.from_list(~Element,outputs,values,fill)),outputs,
    workers_extents(~Element,Tensors.from_list(~Element,outputs,values,fill),workers),Listed.from_list_extents(~Element,outputs,values,fill))

# The reduction of the trailing axes of a tensor of the contract of a
# reader, every axis after the kept ones, is Ready and holds at each index of
# the kept axes the finish of the fold of its line.

def trailing_contract(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+limits: Machine.Limits,+operations: F32,+length: U32,+initial: Accumulator,+model: PiecesModel.Model<Element>,+shape: List<&2,U32>,+kept: Nat,+outputs: List<&2,U32>,+fill: Element,+workers: U32,
  +contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers}),
  +room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},+length_exact: {U32.to_nat(length) == ShapeLaws.elements(List.drop(&2,U32,shape,kept)) : Nat},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},
  +outputs_exact: {ShapeLaws.elements(outputs) == ShapeLaws.elements(List.take(&2,U32,shape,kept)) : Nat},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32}) ->
  Global.Shaped(~Element,~Lines<Accumulator,Logical>,~line_value(~Element,~Accumulator,~operation,~finish,~Logical,~original),Lines{initial,logical,ShapeLaws.elements(List.drop(&2,U32,shape,kept))},outputs,Layout.reduce_trailing(~Element,~Accumulator,~operation,~finish,operations,length,initial,PiecesModel.pack(~Element,model),shape,outputs,fill,workers)):
  KernelGemm.Both{KernelGemm.Both{Content.WellFormed{+fields,+ready_pieces,+coverage},KernelGemm.Both{+counted,+canonical}},+values} = contract
  +packed = PiecesModel.packed_contents(~Element,model)
  +same_length = Equal.cong(U32,Nat,word => U32.to_nat(word),(length * 1 : U32),length,Arithmetic.multiply_one(length))
  +length_word = Equal.trans(Nat,U32.to_nat((length * 1 : U32)),U32.to_nat(length),ShapeLaws.elements(List.drop(&2,U32,shape,kept)),same_length,length_exact)
  +split = CountCompletion.elements_split(shape,kept)
  +taken = ShapeLaws.elements(List.take(&2,U32,shape,kept))
  +dropped = ShapeLaws.elements(List.drop(&2,U32,shape,kept))
  +size = U32.to_nat((length * 1 : U32))
  +lines_exact = Equal.trans(Nat,ShapeLaws.elements(shape),Nat.mul(taken,dropped),Nat.mul(taken,size),split,
    Equal.cong(Nat,Nat,number => Nat.mul(taken,number),dropped,size,Equal.sym(Nat,size,dropped,length_word)))
  +product_positive = Held.nat_lt_moved(0n,Nat.mul(taken,dropped),ShapeLaws.elements(shape),Equal.sym(Nat,ShapeLaws.elements(shape),Nat.mul(taken,dropped),split),nonempty)
  +lines_positive = RowMajor.left_factor_positive(taken,dropped,product_positive)
  +folded = trailing_values(~Element,~Accumulator,~operation,~finish,~Logical,~original,limits,shape,kept,Grid.piece_target(),length,initial,1.0,operations,fill,workers,logical,model,
    room,nonempty,length_exact,cap,cap_ok,count_room,
    {%packed : {PiecesModel.content_layout(~Element,_) == Pieces.ranges_of_grid(Tensors.grid_of(shape)) : Pieces.Ranges} canonical :
      {PiecesModel.content_layout(~Element,PiecesModel.content(~Element,model)) == Pieces.ranges_of_grid(Tensors.grid_of(shape)) : Pieces.Ranges}},
    {%packed : PiecesPlan.Coverage(PiecesModel.content_layout(~Element,_),0n,ShapeLaws.elements(shape)) coverage : PiecesPlan.Coverage(PiecesModel.content_layout(~Element,PiecesModel.content(~Element,model)),0n,ShapeLaws.elements(shape))},
    {%packed : Content.Values(~Element,~Logical,~original,logical,_) values : Content.Values(~Element,~Logical,~original,logical,PiecesModel.content(~Element,model))})
  +lined = {%lines_exact : {Pair.snd(Pieces.Pieces(~Element),List<&2,Element>,LineChain.folds(~Storage.Buffer<Element>,~Element,~Accumulator,~Accumulator,~Element,~LineChain.unchanged(~Accumulator),~Operations.fold_range(~Element,~Accumulator,~operation),~finish,~Pieces.vacant_buffer(~Element),[initial],LineChain.Line{length,1},(length / U32.max(Layout.cells_per_outer(Tensors.grid_of(shape)),1) : U32),1.0,operations,Tensors.grid_of(shape),fill,workers,PiecesModel.pack(~Element,model))) == LineChainValues.turned_values(~Accumulator,~Element,~finish,chunk_folds(~Element,~Accumulator,~operation,taken,size,values_from(~Element,~Logical,~original,logical,_,0n),initial)) : List<&2,Element>}
    folded : {Pair.snd(Pieces.Pieces(~Element),List<&2,Element>,LineChain.folds(~Storage.Buffer<Element>,~Element,~Accumulator,~Accumulator,~Element,~LineChain.unchanged(~Accumulator),~Operations.fold_range(~Element,~Accumulator,~operation),~finish,~Pieces.vacant_buffer(~Element),[initial],LineChain.Line{length,1},(length / U32.max(Layout.cells_per_outer(Tensors.grid_of(shape)),1) : U32),1.0,operations,Tensors.grid_of(shape),fill,workers,PiecesModel.pack(~Element,model))) == LineChainValues.turned_values(~Accumulator,~Element,~finish,chunk_folds(~Element,~Accumulator,~operation,taken,size,values_from(~Element,~Logical,~original,logical,Nat.mul(taken,size),0n),initial)) : List<&2,Element>}}
  +chunks = Equal.trans(List<&2,Element>,Pair.snd(Pieces.Pieces(~Element),List<&2,Element>,LineChain.folds(~Storage.Buffer<Element>,~Element,~Accumulator,~Accumulator,~Element,~LineChain.unchanged(~Accumulator),~Operations.fold_range(~Element,~Accumulator,~operation),~finish,~Pieces.vacant_buffer(~Element),[initial],LineChain.Line{length,1},(length / U32.max(Layout.cells_per_outer(Tensors.grid_of(shape)),1) : U32),1.0,operations,Tensors.grid_of(shape),fill,workers,PiecesModel.pack(~Element,model))),LineChainValues.turned_values(~Accumulator,~Element,~finish,chunk_folds(~Element,~Accumulator,~operation,taken,size,values_from(~Element,~Logical,~original,logical,Nat.mul(taken,size),0n),initial)),LineChainValues.turned_values(~Accumulator,~Element,~finish,line_folds(~Element,~Accumulator,~operation,~Logical,~original,logical,ShapeLaws.elements(List.take(&2,U32,shape,kept)),U32.to_nat((length * 1 : U32)),0n,initial)),
    lined,Equal.cong(List<&2,Accumulator>,List<&2,Element>,items => LineChainValues.turned_values(~Accumulator,~Element,~finish,items),chunk_folds(~Element,~Accumulator,~operation,taken,size,values_from(~Element,~Logical,~original,logical,Nat.mul(taken,size),0n),initial),line_folds(~Element,~Accumulator,~operation,~Logical,~original,logical,ShapeLaws.elements(List.take(&2,U32,shape,kept)),U32.to_nat((length * 1 : U32)),0n,initial),
      chunk_lines(~Element,~Accumulator,~operation,~Logical,~original,logical,taken,size,0n,initial)))
  +outputs_positive = Held.nat_lt_moved(0n,ShapeLaws.elements(outputs),taken,outputs_exact,lines_positive)
  +outputs_room = Held.nat_le_moved(ShapeLaws.elements(outputs),taken,U32.to_nat(Machine.capacity(limits)),outputs_exact,
    Order.transitive(taken,Nat.mul(taken,dropped),U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(taken,dropped,GridRanges.right_factor_positive(taken,dropped,product_positive)),
      Held.nat_le_moved(Nat.mul(taken,dropped),ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits)),Equal.sym(Nat,ShapeLaws.elements(shape),Nat.mul(taken,dropped),split),room)))
  +listed_length = Equal.trans(Nat,List.length(&2,Element,Pair.snd(Pieces.Pieces(~Element),List<&2,Element>,LineChain.folds(~Storage.Buffer<Element>,~Element,~Accumulator,~Accumulator,~Element,~LineChain.unchanged(~Accumulator),~Operations.fold_range(~Element,~Accumulator,~operation),~finish,~Pieces.vacant_buffer(~Element),[initial],LineChain.Line{length,1},(length / U32.max(Layout.cells_per_outer(Tensors.grid_of(shape)),1) : U32),1.0,operations,Tensors.grid_of(shape),fill,workers,PiecesModel.pack(~Element,model)))),List.length(&2,Element,LineChainValues.turned_values(~Accumulator,~Element,~finish,line_folds(~Element,~Accumulator,~operation,~Logical,~original,logical,ShapeLaws.elements(List.take(&2,U32,shape,kept)),U32.to_nat((length * 1 : U32)),0n,initial))),ShapeLaws.elements(outputs),
    Equal.cong(List<&2,Element>,Nat,items => List.length(&2,Element,items),Pair.snd(Pieces.Pieces(~Element),List<&2,Element>,LineChain.folds(~Storage.Buffer<Element>,~Element,~Accumulator,~Accumulator,~Element,~LineChain.unchanged(~Accumulator),~Operations.fold_range(~Element,~Accumulator,~operation),~finish,~Pieces.vacant_buffer(~Element),[initial],LineChain.Line{length,1},(length / U32.max(Layout.cells_per_outer(Tensors.grid_of(shape)),1) : U32),1.0,operations,Tensors.grid_of(shape),fill,workers,PiecesModel.pack(~Element,model))),LineChainValues.turned_values(~Accumulator,~Element,~finish,line_folds(~Element,~Accumulator,~operation,~Logical,~original,logical,ShapeLaws.elements(List.take(&2,U32,shape,kept)),U32.to_nat((length * 1 : U32)),0n,initial)),chunks),
    Equal.trans(Nat,List.length(&2,Element,LineChainValues.turned_values(~Accumulator,~Element,~finish,line_folds(~Element,~Accumulator,~operation,~Logical,~original,logical,ShapeLaws.elements(List.take(&2,U32,shape,kept)),U32.to_nat((length * 1 : U32)),0n,initial))),List.length(&2,Accumulator,line_folds(~Element,~Accumulator,~operation,~Logical,~original,logical,ShapeLaws.elements(List.take(&2,U32,shape,kept)),U32.to_nat((length * 1 : U32)),0n,initial)),ShapeLaws.elements(outputs),turned_length(~Element,~Accumulator,~finish,line_folds(~Element,~Accumulator,~operation,~Logical,~original,logical,ShapeLaws.elements(List.take(&2,U32,shape,kept)),U32.to_nat((length * 1 : U32)),0n,initial)),
      Equal.trans(Nat,List.length(&2,Accumulator,line_folds(~Element,~Accumulator,~operation,~Logical,~original,logical,ShapeLaws.elements(List.take(&2,U32,shape,kept)),U32.to_nat((length * 1 : U32)),0n,initial)),taken,ShapeLaws.elements(outputs),
        line_folds_length(~Element,~Accumulator,~operation,~Logical,~original,logical,taken,size,0n,initial),
        Equal.sym(Nat,ShapeLaws.elements(outputs),taken,outputs_exact))))
  +ready = Listed.listed_ready(~Element,outputs,Pair.snd(Pieces.Pieces(~Element),List<&2,Element>,LineChain.folds(~Storage.Buffer<Element>,~Element,~Accumulator,~Accumulator,~Element,~LineChain.unchanged(~Accumulator),~Operations.fold_range(~Element,~Accumulator,~operation),~finish,~Pieces.vacant_buffer(~Element),[initial],LineChain.Line{length,1},(length / U32.max(Layout.cells_per_outer(Tensors.grid_of(shape)),1) : U32),1.0,operations,Tensors.grid_of(shape),fill,workers,PiecesModel.pack(~Element,model))),fill,workers,StorageCount.counted_within(limits,outputs,outputs_room,default_capacity),listed_length)
  Global.shaped(~Element,~Lines<Accumulator,Logical>,~line_value(~Element,~Accumulator,~operation,~finish,~Logical,~original),Lines{initial,logical,ShapeLaws.elements(List.drop(&2,U32,shape,kept))},outputs,
    Layout.reduce_trailing(~Element,~Accumulator,~operation,~finish,operations,length,initial,PiecesModel.pack(~Element,model),shape,outputs,fill,workers),ready,
    trailing_extents(~Element,~Accumulator,~operation,~finish,operations,length,initial,PiecesModel.pack(~Element,model),shape,outputs,fill,workers),
    Global.with_workers(~Element,~Lines<Accumulator,Logical>,~line_value(~Element,~Accumulator,~operation,~finish,~Logical,~original),Lines{initial,logical,ShapeLaws.elements(List.drop(&2,U32,shape,kept))},Tensors.from_list(~Element,outputs,Pair.snd(Pieces.Pieces(~Element),List<&2,Element>,LineChain.folds(~Storage.Buffer<Element>,~Element,~Accumulator,~Accumulator,~Element,~LineChain.unchanged(~Accumulator),~Operations.fold_range(~Element,~Accumulator,~operation),~finish,~Pieces.vacant_buffer(~Element),[initial],LineChain.Line{length,1},(length / U32.max(Layout.cells_per_outer(Tensors.grid_of(shape)),1) : U32),1.0,operations,Tensors.grid_of(shape),fill,workers,PiecesModel.pack(~Element,model))),fill),workers,ready,
      inner => trailing_listed(~Element,~Accumulator,~operation,~finish,~Logical,~original,logical,operations,length,initial,model,shape,kept,outputs,fill,workers,chunks,length_word,outputs_exact,outputs_positive,inner)))

# ---- Strided lines of a block ----

# The fold of count values inner apart from position.
def strided_fold(~Element: Data,~Accumulator: Data,~step: Accumulator -> Element -> Accumulator,~Context: Data,~value: Context -> U32 -> Element,+context: Context,count: Nat,+position: Nat,+inner: Nat,+state: Accumulator) -> Accumulator:
  match count:
    case 0n: state
    case 1n+rest: strided_fold(~Element,~Accumulator,~step,~Context,~value,context,rest,Nat.add(position,inner),inner,step(state,value(context,U32.from_nat(position))))

# The values a walk that takes one of every inner, after gap more, takes from
# count values.
def taken_count(count: Nat,+gap: Nat,+inner: Nat) -> Nat:
  match count:
    case 0n: 0n
    case 1n+rest:
      match gap:
        case 0n: 1n+taken_count(rest,Nat.sub(inner,1n),inner)
        case 1n+before: taken_count(rest,before,inner)

# One value on, the next taken value lies a whole stride on.
def turn_position(+position: Nat,inner: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool}) -> {Nat.add(1n+position,Nat.sub(inner,1n)) == Nat.add(position,inner) : Nat}:
  match inner:
    case 0n: Empty.absurd({Nat.add(1n+position,Nat.sub(0n,1n)) == Nat.add(position,0n) : Nat},WordEncoding.false_true(positive))
    case 1n+ +rest:
      %Equal.sym(Nat,Nat.sub(rest,0n),rest,Order.subtract_zero(rest)) : {Nat.add(1n+position,_) == Nat.add(position,1n+rest) : Nat}
      Equal.sym(Nat,Nat.add(position,1n+rest),1n+Nat.add(position,rest),Addition.successor_right(position,rest))

# The values a walk takes from a list of values, one of every inner after a
# gap, fold inner apart from the first taken.
def taken_strided(~Element: Data,~Accumulator: Data,~step: Accumulator -> Element -> Accumulator,~Context: Data,~value: Context -> U32 -> Element,+context: Context,count: Nat,+position: Nat,gap: Nat,+inner: Nat,+state: Accumulator,+positive: {Nat.is_lt(0n,inner) == True{} : Bool}) ->
  {LineChainValues.taken_list(~Element,~Accumulator,~step,values_from(~Element,~Context,~value,context,count,position),gap,inner,state) == strided_fold(~Element,~Accumulator,~step,~Context,~value,context,taken_count(count,gap,inner),Nat.add(position,gap),inner,state) : Accumulator}:
  match count gap:
    case 0n gap: {==}
    case 1n+ +rest 0n:
      +taken = taken_count(rest,Nat.sub(inner,1n),inner)
      +stepped = step(state,value(context,U32.from_nat(position)))
      %Equal.sym(Nat,Nat.add(position,0n),position,Addition.zero_right(position)) : {LineChainValues.taken_list(~Element,~Accumulator,~step,values_from(~Element,~Context,~value,context,1n+rest,position),0n,inner,state) == strided_fold(~Element,~Accumulator,~step,~Context,~value,context,1n+taken,_,inner,state) : Accumulator}
      %turn_position(position,inner,positive) : {LineChainValues.taken_list(~Element,~Accumulator,~step,values_from(~Element,~Context,~value,context,1n+rest,position),0n,inner,state) == strided_fold(~Element,~Accumulator,~step,~Context,~value,context,taken,_,inner,stepped) : Accumulator}
      taken_strided(~Element,~Accumulator,~step,~Context,~value,context,rest,1n+position,Nat.sub(inner,1n),inner,stepped,positive)
    case 1n+ +rest 1n+ +before:
      %Equal.sym(Nat,Nat.add(position,1n+before),1n+Nat.add(position,before),Addition.successor_right(position,before)) :
        {LineChainValues.taken_list(~Element,~Accumulator,~step,values_from(~Element,~Context,~value,context,1n+rest,position),1n+before,inner,state) == strided_fold(~Element,~Accumulator,~step,~Context,~value,context,taken_count(rest,before,inner),_,inner,state) : Accumulator}
      taken_strided(~Element,~Accumulator,~step,~Context,~value,context,rest,1n+position,before,inner,state,positive)

# A walk skips as many values as its gap.
def taken_skipped(count: Nat,+rest: Nat,+gap: Nat,+inner: Nat) -> {taken_count(Nat.add(count,rest),Nat.add(count,gap),inner) == taken_count(rest,gap,inner) : Nat}:
  match count:
    case 0n: {==}
    case 1n+ +before: taken_skipped(before,rest,gap,inner)

# A whole stride of values from a gap below the stride takes one value and
# comes back to the gap.
def taken_period(+gap: Nat,+more: Nat,+rest: Nat) ->
  {taken_count(Nat.add(Nat.add(gap,1n+more),rest),gap,Nat.add(gap,1n+more)) == 1n+taken_count(rest,gap,Nat.add(gap,1n+more)) : Nat}:
  +inner = Nat.add(gap,1n+more)
  +whole = Nat.add(1n+more,rest)
  +back = Equal.trans(Nat,Nat.sub(inner,1n),Nat.sub(1n+Nat.add(gap,more),1n),Nat.add(more,gap),
    Equal.cong(Nat,Nat,value => Nat.sub(value,1n),inner,1n+Nat.add(gap,more),Addition.successor_right(gap,more)),
    Equal.trans(Nat,Nat.sub(1n+Nat.add(gap,more),1n),Nat.add(gap,more),Nat.add(more,gap),Order.subtract_zero(Nat.add(gap,more)),Order.commutative(gap,more)))
  %Equal.sym(Nat,Nat.add(Nat.add(gap,1n+more),rest),Nat.add(gap,whole),Addition.associative(gap,1n+more,rest)) : {taken_count(_,gap,inner) == 1n+taken_count(rest,gap,inner) : Nat}
  Equal.trans(Nat,taken_count(Nat.add(gap,whole),gap,inner),taken_count(whole,0n,inner),1n+taken_count(rest,gap,inner),
    {%Addition.zero_right(gap) : {taken_count(Nat.add(gap,whole),_,inner) == taken_count(whole,0n,inner) : Nat}
      taken_skipped(gap,whole,0n,inner) : {taken_count(Nat.add(gap,whole),gap,inner) == taken_count(whole,0n,inner) : Nat}},
    Equal.cong(Nat,Nat,count => 1n+count,taken_count(Nat.add(more,rest),Nat.sub(inner,1n),inner),taken_count(rest,gap,inner),
      {%Equal.sym(Nat,Nat.sub(inner,1n),Nat.add(more,gap),back) : {taken_count(Nat.add(more,rest),_,inner) == taken_count(rest,gap,inner) : Nat}
        taken_skipped(more,rest,gap,inner) : {taken_count(Nat.add(more,rest),Nat.sub(inner,1n),inner) == taken_count(rest,gap,inner) : Nat}}))

# Whole strides of values from a gap below the stride take one value each.
def taken_whole(blocks: Nat,+gap: Nat,+more: Nat) -> {taken_count(Nat.mul(blocks,Nat.add(gap,1n+more)),gap,Nat.add(gap,1n+more)) == blocks : Nat}:
  match blocks:
    case 0n: {==}
    case 1n+ +rest:
      +inner = Nat.add(gap,1n+more)
      Equal.trans(Nat,taken_count(Nat.add(inner,Nat.mul(rest,inner)),gap,inner),1n+taken_count(Nat.mul(rest,inner),gap,inner),1n+rest,
        taken_period(gap,more,Nat.mul(rest,inner)),Equal.cong(Nat,Nat,count => 1n+count,taken_count(Nat.mul(rest,inner),gap,inner),rest,taken_whole(rest,gap,more)))

# Two readers agree on count values, the left inner apart from left_position,
# the right in a row from right_position.
def StridedAgrees(~Element: Data,~Left: Data,~left_value: Left -> U32 -> Element,~Right: Data,~right_value: Right -> U32 -> Element,+left_context: Left,+right_context: Right,count: Nat,
  +left_position: Nat,+inner: Nat,+right_position: Nat) -> Data:
  match count:
    case 0n: Unit
    case 1n+rest:
      KernelGemm.Both<{left_value(left_context,U32.from_nat(left_position)) == right_value(right_context,U32.from_nat(right_position)) : Element},
        StridedAgrees(~Element,~Left,~left_value,~Right,~right_value,left_context,right_context,rest,Nat.add(left_position,inner),inner,1n+right_position)>

# A fold inner apart over one reader is the fold in a row over another when
# they agree value by value.
def strided_line(~Element: Data,~Accumulator: Data,~step: Accumulator -> Element -> Accumulator,~Left: Data,~left_value: Left -> U32 -> Element,~Right: Data,~right_value: Right -> U32 -> Element,+left_context: Left,+right_context: Right,count: Nat,+left_position: Nat,+inner: Nat,+right_position: Nat,+state: Accumulator,
  +agrees: StridedAgrees(~Element,~Left,~left_value,~Right,~right_value,left_context,right_context,count,left_position,inner,right_position)) ->
  {strided_fold(~Element,~Accumulator,~step,~Left,~left_value,left_context,count,left_position,inner,state) == line_fold(~Element,~Accumulator,~step,~Right,~right_value,right_context,count,right_position,state) : Accumulator}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      KernelGemm.Both{+same,+others} = agrees
      +stepped = step(state,right_value(right_context,U32.from_nat(right_position)))
      %Equal.sym(Element,left_value(left_context,U32.from_nat(left_position)),right_value(right_context,U32.from_nat(right_position)),same) :
        {strided_fold(~Element,~Accumulator,~step,~Left,~left_value,left_context,rest,Nat.add(left_position,inner),inner,step(state,_)) == line_fold(~Element,~Accumulator,~step,~Right,~right_value,right_context,rest,1n+right_position,stepped) : Accumulator}
      strided_line(~Element,~Accumulator,~step,~Left,~left_value,~Right,~right_value,left_context,right_context,rest,Nat.add(left_position,inner),inner,1n+right_position,stepped,others)

# ---- The order of the axes ----

def appended_extents(+shape: List<&2,U32>,left: List<&2,Nat>,+right: List<&2,Nat>) ->
  {Layout.extents_at(shape,List.append(&2,Nat,left,right)) == List.append(&2,U32,Layout.extents_at(shape,left),Layout.extents_at(shape,right)) : List<&2,U32>}:
  match left:
    case []: {==}
    case +index <> +rest:
      Equal.cong(List<&2,U32>,List<&2,U32>,tail => Layout.extent_at(shape,index) <> tail,Layout.extents_at(shape,List.append(&2,Nat,rest,right)),
        List.append(&2,U32,Layout.extents_at(shape,rest),Layout.extents_at(shape,right)),appended_extents(shape,rest,right))

def extents_length(+shape: List<&2,U32>,indices: List<&2,Nat>) -> {List.length(&2,U32,Layout.extents_at(shape,indices)) == List.length(&2,Nat,indices) : Nat}:
  match indices:
    case []: {==}
    case index <> +rest:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,U32,Layout.extents_at(shape,rest)),List.length(&2,Nat,rest),extents_length(shape,rest))

def take_prefix(~Element: Data,left: List<&2,Element>,+right: List<&2,Element>) ->
  {List.take(&2,Element,List.append(&2,Element,left,right),List.length(&2,Element,left)) == left : List<&2,Element>}:
  match left:
    case []:
      match right:
        case []: {==}
        case head <> tail: {==}
    case +head <> +tail:
      Equal.cong(List<&2,Element>,List<&2,Element>,others => head <> others,
        List.take(&2,Element,List.append(&2,Element,tail,right),List.length(&2,Element,tail)),tail,take_prefix(~Element,tail,right))

def dropped_none(~Element: Data,items: List<&2,Element>) -> {List.drop(&2,Element,items,0n) == items : List<&2,Element>}:
  match items:
    case []: {==}
    case head <> tail: {==}

def taken_none(~Element: Data,items: List<&2,Element>) -> {List.take(&2,Element,items,0n) == [] : List<&2,Element>}:
  match items:
    case []: {==}
    case head <> tail: {==}

def dropped_all(~Element: Data,items: List<&2,Element>) -> {List.drop(&2,Element,items,List.length(&2,Element,items)) == [] : List<&2,Element>}:
  match items:
    case []: {==}
    case head <> +tail: dropped_all(~Element,tail)

def drop_prefix(~Element: Data,left: List<&2,Element>,+right: List<&2,Element>) ->
  {List.drop(&2,Element,List.append(&2,Element,left,right),List.length(&2,Element,left)) == right : List<&2,Element>}:
  match left:
    case []: dropped_none(~Element,right)
    case head <> +tail: drop_prefix(~Element,tail,right)

def every_length(remaining: Nat,+index: Nat) -> {List.length(&2,Nat,Layout.every_index(remaining,index)) == remaining : Nat}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Nat,Layout.every_index(rest,1n+index)),rest,every_length(rest,1n+index))

# The indices from start on hold no axis below start.
def every_after(remaining: Nat,+start: Nat,+axis: Nat,+below: {Nat.is_lt(axis,start) == True{} : Bool}) ->
  {Views.contains_index(Layout.every_index(remaining,start),axis) == False{} : Bool}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +different = Equal.trans(Bool,Nat.is_eq(start,axis),Nat.is_eq(axis,start),False{},Order.equal_symmetric(start,axis),Order.less_not_equal(axis,start,below))
      %Equal.sym(Bool,Nat.is_eq(start,axis),False{},different) : {_ || Views.contains_index(Layout.every_index(rest,1n+start),axis) == False{} : Bool}
      every_after(rest,1n+start,axis,Order.lt_succ(axis,start,Bounds.strict_le(axis,start,below)))

def every_unique(remaining: Nat,+start: Nat) -> {Views.unique_axes(Layout.every_index(remaining,start)) == True{} : Bool}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(Bool,Views.contains_index(Layout.every_index(rest,1n+start),start),False{},
          every_after(rest,1n+start,start,Order.lt_succ(start,start,Order.reflexive(start)))) :
        {Bool.not(_) && Views.unique_axes(Layout.every_index(rest,1n+start)) == True{} : Bool}
      every_unique(rest,1n+start)

# The axis at start leads the axes from start on.
def picked(axes: List<&2,Views.Axis>,start: Nat,+offset: U32,inside: {Nat.is_lt(start,List.length(&2,Views.Axis,axes)) == True{} : Bool}) ->
  {Views.prepend_axis(List.get(&2,Views.Axis,axes,start),Some{Views.View{List.drop(&2,Views.Axis,axes,1n+start),offset}}) ==
    Some{Views.View{List.drop(&2,Views.Axis,axes,start),offset}} : Maybe<&2,Views.View>}:
  match axes start:
    case [] start:
      Empty.absurd({Views.prepend_axis(List.get(&2,Views.Axis,[],start),Some{Views.View{List.drop(&2,Views.Axis,[],1n+start),offset}}) ==
        Some{Views.View{List.drop(&2,Views.Axis,[],start),offset}} : Maybe<&2,Views.View>},Order.below_zero(start,inside))
    case +head <> +tail 0n:
      Equal.cong(List<&2,Views.Axis>,Maybe<&2,Views.View>,rest => Some{Views.View{head <> rest,offset}},List.drop(&2,Views.Axis,tail,0n),tail,dropped_none(~Views.Axis,tail))
    case head <> +tail 1n+ +previous: picked(tail,previous,offset,inside)

# The indices from start on pick the axes from start on.
def every_reorder(remaining: Nat,+start: Nat,+axes: List<&2,Views.Axis>,+offset: U32,
  +sized: {Nat.add(start,remaining) == List.length(&2,Views.Axis,axes) : Nat}) ->
  {Views.reorder(Layout.every_index(remaining,start),axes,offset) == Some{Views.View{List.drop(&2,Views.Axis,axes,start),offset}} : Maybe<&2,Views.View>}:
  match remaining:
    case 0n:
      +at_end = Equal.trans(Nat,start,Nat.add(start,0n),List.length(&2,Views.Axis,axes),Equal.sym(Nat,Nat.add(start,0n),start,Addition.zero_right(start)),sized)
      %Equal.sym(Nat,start,List.length(&2,Views.Axis,axes),at_end) :
        {Some{Views.View{[],offset}} == Some{Views.View{List.drop(&2,Views.Axis,axes,_),offset}} : Maybe<&2,Views.View>}
      Equal.cong(List<&2,Views.Axis>,Maybe<&2,Views.View>,rest => Some{Views.View{rest,offset}},[],List.drop(&2,Views.Axis,axes,List.length(&2,Views.Axis,axes)),
        Equal.sym(List<&2,Views.Axis>,List.drop(&2,Views.Axis,axes,List.length(&2,Views.Axis,axes)),[],dropped_all(~Views.Axis,axes)))
    case 1n+ +rest:
      +room = {%Addition.successor_right(start,rest) : {_ == List.length(&2,Views.Axis,axes) : Nat} sized :
        {Nat.add(1n+start,rest) == List.length(&2,Views.Axis,axes) : Nat}}
      %Equal.sym(Maybe<&2,Views.View>,Views.reorder(Layout.every_index(rest,1n+start),axes,offset),Some{Views.View{List.drop(&2,Views.Axis,axes,1n+start),offset}},
          every_reorder(rest,1n+start,axes,offset,room)) :
        {Views.prepend_axis(List.get(&2,Views.Axis,axes,start),_) == Some{Views.View{List.drop(&2,Views.Axis,axes,start),offset}} : Maybe<&2,Views.View>}
      picked(axes,start,offset,Order.below_complement(start,rest,List.length(&2,Views.Axis,axes),sized))

def drop_next(xs: List<&2,U32>,index: Nat,inside: {Nat.is_lt(index,List.length(&2,U32,xs)) == True{} : Bool}) ->
  {List.drop(&2,U32,xs,index) == Layout.extent_at(xs,index) <> List.drop(&2,U32,xs,1n+index) : List<&2,U32>}:
  match xs index:
    case [] index: Empty.absurd({List.drop(&2,U32,[],index) == Layout.extent_at([],index) <> List.drop(&2,U32,[],1n+index) : List<&2,U32>},Order.below_zero(index,inside))
    case +head <> +tail 0n:
      %Equal.sym(List<&2,U32>,List.drop(&2,U32,tail,0n),tail,dropped_none(~U32,tail)) : {head <> tail == head <> _ : List<&2,U32>}
      {==}
    case head <> +tail 1n+ +previous: drop_next(tail,previous,inside)

# Every index from start reads the extents of a shape from start.
def every_extents(remaining: Nat,+start: Nat,+shape: List<&2,U32>,+sized: {Nat.add(start,remaining) == List.length(&2,U32,shape) : Nat}) ->
  {Layout.extents_at(shape,Layout.every_index(remaining,start)) == List.drop(&2,U32,shape,start) : List<&2,U32>}:
  match remaining:
    case 0n:
      +at_end = Equal.trans(Nat,start,Nat.add(start,0n),List.length(&2,U32,shape),Equal.sym(Nat,Nat.add(start,0n),start,Addition.zero_right(start)),sized)
      %Equal.sym(Nat,start,List.length(&2,U32,shape),at_end) : {[] == List.drop(&2,U32,shape,_) : List<&2,U32>}
      Equal.sym(List<&2,U32>,List.drop(&2,U32,shape,List.length(&2,U32,shape)),[],dropped_all(~U32,shape))
    case 1n+ +rest:
      +room = {%Addition.successor_right(start,rest) : {_ == List.length(&2,U32,shape) : Nat} sized :
        {Nat.add(1n+start,rest) == List.length(&2,U32,shape) : Nat}}
      %Equal.sym(List<&2,U32>,Layout.extents_at(shape,Layout.every_index(rest,1n+start)),List.drop(&2,U32,shape,1n+start),every_extents(rest,1n+start,shape,room)) :
        {Layout.extent_at(shape,start) <> _ == List.drop(&2,U32,shape,start) : List<&2,U32>}
      Equal.sym(List<&2,U32>,List.drop(&2,U32,shape,start),Layout.extent_at(shape,start) <> List.drop(&2,U32,shape,1n+start),
        drop_next(shape,start,Order.below_complement(start,rest,List.length(&2,U32,shape),sized)))

# The first count indices from start read the extents taken from start.
def every_take(+xs: List<&2,U32>,count: Nat,+start: Nat,+room: {Nat.is_le(Nat.add(start,count),List.length(&2,U32,xs)) == True{} : Bool}) ->
  {Layout.extents_at(xs,Layout.every_index(count,start)) == List.take(&2,U32,List.drop(&2,U32,xs,start),count) : List<&2,U32>}:
  match count:
    case 0n: Equal.sym(List<&2,U32>,List.take(&2,U32,List.drop(&2,U32,xs,start),0n),[],taken_none(~U32,List.drop(&2,U32,xs,start)))
    case 1n+ +rest:
      +inside = Bounds.nat_trans(start,Nat.add(start,1n+rest),List.length(&2,U32,xs),
        Held.nat_lt_moved(start,Nat.add(start,1n+rest),1n+Nat.add(start,rest),Addition.successor_right(start,rest),Order.lt_succ(start,Nat.add(start,rest),Order.le_add(start,rest))),
        room)
      +later = Held.nat_le_moved(Nat.add(1n+start,rest),Nat.add(start,1n+rest),List.length(&2,U32,xs),
        Equal.sym(Nat,Nat.add(start,1n+rest),1n+Nat.add(start,rest),Addition.successor_right(start,rest)),room)
      %Equal.sym(List<&2,U32>,List.drop(&2,U32,xs,start),Layout.extent_at(xs,start) <> List.drop(&2,U32,xs,1n+start),drop_next(xs,start,inside)) :
        {Layout.extent_at(xs,start) <> Layout.extents_at(xs,Layout.every_index(rest,1n+start)) == List.take(&2,U32,_,1n+rest) : List<&2,U32>}
      Equal.cong(List<&2,U32>,List<&2,U32>,tail => Layout.extent_at(xs,start) <> tail,Layout.extents_at(xs,Layout.every_index(rest,1n+start)),
        List.take(&2,U32,List.drop(&2,U32,xs,1n+start),rest),every_take(xs,rest,1n+start,later))

def axes_length(axes: List<&2,Views.Axis>) -> {List.length(&2,U32,Views.axis_extents(axes)) == List.length(&2,Views.Axis,axes) : Nat}:
  match axes:
    case []: {==}
    case Views.Axis{extent,stride} <> +rest:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,U32,Views.axis_extents(rest)),List.length(&2,Views.Axis,rest),axes_length(rest))

def cells_extents(+extents: List<&2,U32>) -> {Views.axis_extents(Tasks.piece_cells(extents)) == extents : List<&2,U32>}:
  match extents:
    case []: {==}
    case +extent <> +rest:
      %Equal.sym(List<&2,Views.Axis>,Tasks.piece_cells(extent <> rest),Views.Axis{extent,Grid.product(rest)} <> Tasks.piece_cells(rest),GridPieces.dense_con(extent,rest)) :
        {Views.axis_extents(_) == extent <> rest : List<&2,U32>}
      Equal.cong(List<&2,U32>,List<&2,U32>,tail => extent <> tail,Views.axis_extents(Tasks.piece_cells(rest)),rest,cells_extents(rest))

# Indices in order from start are the indices from start.
def in_order_every(order: List<&2,Nat>,+start: Nat,+ordered: {Layout.in_order(order,start) == True{} : Bool}) ->
  {order == Layout.every_index(List.length(&2,Nat,order),start) : List<&2,Nat>}:
  match order:
    case []: {==}
    case +axis <> +rest:
      +first = Order.equal_of_is_eq(axis,start,Boolean.left(Nat.is_eq(axis,start),Layout.in_order(rest,1n+start),ordered))
      +later = in_order_every(rest,1n+start,Boolean.right(Nat.is_eq(axis,start),Layout.in_order(rest,1n+start),ordered))
      Equal.trans(List<&2,Nat>,axis <> rest,start <> rest,start <> Layout.every_index(List.length(&2,Nat,rest),1n+start),
        Equal.cong(Nat,List<&2,Nat>,head => head <> rest,axis,start,first),
        Equal.cong(List<&2,Nat>,List<&2,Nat>,tail => start <> tail,rest,Layout.every_index(List.length(&2,Nat,rest),1n+start),later))

def ordered_left(left: List<&2,Nat>,+right: List<&2,Nat>,+start: Nat,+ordered: {Layout.in_order(List.append(&2,Nat,left,right),start) == True{} : Bool}) ->
  {Layout.in_order(left,start) == True{} : Bool}:
  match left:
    case []: {==}
    case +axis <> +rest:
      +head = Boolean.left(Nat.is_eq(axis,start),Layout.in_order(List.append(&2,Nat,rest,right),1n+start),ordered)
      +tail = ordered_left(rest,right,1n+start,Boolean.right(Nat.is_eq(axis,start),Layout.in_order(List.append(&2,Nat,rest,right),1n+start),ordered))
      %Equal.sym(Bool,Nat.is_eq(axis,start),True{},head) : {_ && Layout.in_order(rest,1n+start) == True{} : Bool}
      %Equal.sym(Bool,Layout.in_order(rest,1n+start),True{},tail) : {True{} && _ == True{} : Bool}
      {==}

def ordered_right(left: List<&2,Nat>,+right: List<&2,Nat>,+start: Nat,+ordered: {Layout.in_order(List.append(&2,Nat,left,right),start) == True{} : Bool}) ->
  {Layout.in_order(right,Nat.add(start,List.length(&2,Nat,left))) == True{} : Bool}:
  match left:
    case []:
      %Equal.sym(Nat,Nat.add(start,0n),start,Addition.zero_right(start)) : {Layout.in_order(right,_) == True{} : Bool}
      ordered
    case +axis <> +rest:
      %Equal.sym(Nat,Nat.add(start,1n+List.length(&2,Nat,rest)),1n+Nat.add(start,List.length(&2,Nat,rest)),Addition.successor_right(start,List.length(&2,Nat,rest))) : {Layout.in_order(right,_) == True{} : Bool}
      ordered_right(rest,right,1n+start,Boolean.right(Nat.is_eq(axis,start),Layout.in_order(List.append(&2,Nat,rest,right),1n+start),ordered))

# Every axis in order reads the cells of a shape as they lie.
def line_identity(+shape: List<&2,U32>,+order: List<&2,Nat>,+ordered: {Layout.in_order(order,0n) == True{} : Bool},
  +sized: {List.length(&2,Nat,order) == List.length(&2,U32,shape) : Nat}) ->
  {Layout.line_view(shape,order) == Some{Views.View{Tasks.piece_cells(shape),0}} : Maybe<&2,Views.View>}:
  +cells = Tasks.piece_cells(shape)
  +rank = List.length(&2,U32,shape)
  +every = Layout.every_index(rank,0n)
  +order_every = Equal.trans(List<&2,Nat>,order,Layout.every_index(List.length(&2,Nat,order),0n),every,in_order_every(order,0n,ordered),
    Equal.cong(Nat,List<&2,Nat>,count => Layout.every_index(count,0n),List.length(&2,Nat,order),rank,sized))
  +ranked = Equal.cong(List<&2,U32>,Nat,items => List.length(&2,U32,items),Views.axis_extents(cells),shape,cells_extents(shape))
  +cells_rank = Equal.trans(Nat,rank,List.length(&2,U32,Views.axis_extents(cells)),List.length(&2,Views.Axis,cells),Equal.sym(Nat,List.length(&2,U32,Views.axis_extents(cells)),rank,ranked),
    axes_length(cells))
  +valid = {
    %Equal.sym(Nat,List.length(&2,U32,Views.axis_extents(cells)),rank,ranked) : {Nat.is_eq(_,List.length(&2,Nat,every)) && Views.unique_axes(every) == True{} : Bool}
    %Equal.sym(Nat,List.length(&2,Nat,every),rank,every_length(rank,0n)) : {Nat.is_eq(rank,_) && Views.unique_axes(every) == True{} : Bool}
    %Equal.sym(Bool,Nat.is_eq(rank,rank),True{},Order.equal_reflexive(rank)) : {_ && Views.unique_axes(every) == True{} : Bool}
    every_unique(rank,0n) : {Nat.is_eq(Views.rank(Views.View{cells,0}),List.length(&2,Nat,every)) && Views.unique_axes(every) == True{} : Bool}}
  %Equal.sym(List<&2,Nat>,order,every,order_every) : {Views.permute(Views.View{cells,0},_) == Some{Views.View{cells,0}} : Maybe<&2,Views.View>}
  %Equal.sym(Bool,Nat.is_eq(Views.rank(Views.View{cells,0}),List.length(&2,Nat,every)) && Views.unique_axes(every),True{},valid) :
    {Views.permute_when(_,Views.View{cells,0},every) == Some{Views.View{cells,0}} : Maybe<&2,Views.View>}
  %dropped_none(~Views.Axis,cells) : {Views.reorder(every,cells,0) == Some{Views.View{_,0}} : Maybe<&2,Views.View>}
  every_reorder(rank,0n,cells,0,cells_rank)

# ---- The cells in row-major order ----

# The row-major steps of a shape weight the coordinates of an index back into
# the index.
def row_major_sum(+limits: Machine.Limits,extents: List<&2,U32>,+index: Nat,+inside: {Nat.is_lt(index,ShapeLaws.elements(extents)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat(Box.stepped_sum(extents,ViewPart.view_steps(Tasks.piece_cells(extents)),index)) == index : Nat}:
  match extents:
    case []:
      match index:
        case 0n: {==}
        case 1n+ +previous:
          Empty.absurd({U32.to_nat(Box.stepped_sum([],ViewPart.view_steps(Tasks.piece_cells([])),1n+previous)) == 1n+previous : Nat},
            WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(previous,0n),True{},
              Equal.sym(Bool,Nat.is_lt(previous,0n),False{},Order.not_below(0n,previous,Order.zero_le(previous))),inside)))
    case +extent <> +rest:
      +inner = ShapeLaws.elements(rest)
      +outer = U32.to_nat(extent)
      +total = Nat.mul(outer,inner)
      +positive = Order.le_lt_trans(0n,index,total,Order.zero_le(index),inside)
      +inner_positive = GridRanges.right_factor_positive(outer,inner,positive)
      +outer_positive = RowMajor.left_factor_positive(outer,inner,positive)
      +quotient = Nat.div(index,inner)
      +remainder = Nat.mod(index,inner)
      +exact = NatDivision.quotient_exact(index,inner,inner_positive)
      +inner_room = Order.transitive(inner,total,U32.to_nat(Machine.capacity(limits)),
        Held.nat_le_right_moved(inner,total,Nat.mul(inner,outer),RowMajor.multiply_commutative(outer,inner),RowMajor.product_covers(inner,outer,outer_positive)),room)
      +index_room = Order.transitive(index,total,U32.to_nat(Machine.capacity(limits)),Bounds.strict_le(index,total,inside),room)
      +sum_room = Held.nat_le_moved(Nat.add(Nat.mul(quotient,inner),remainder),index,U32.to_nat(Machine.capacity(limits)),Equal.sym(Nat,index,Nat.add(Nat.mul(quotient,inner),remainder),exact),index_room)
      +scaled_room = Order.transitive(Nat.mul(quotient,inner),Nat.add(Nat.mul(quotient,inner),remainder),U32.to_nat(Machine.capacity(limits)),Order.le_add(Nat.mul(quotient,inner),remainder),sum_room)
      +quotient_room = Order.transitive(quotient,Nat.mul(quotient,inner),U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(quotient,inner,inner_positive),scaled_room)
      +quotient_word = FullRange.roundtrip(quotient,Machine.capacity(limits),quotient_room)
      +product_word = TiledPublic.product_exact(Machine.capacity(limits),LimitBounds.capacity_fits(limits),rest,inner_room)
      +scaled = Arithmetic.product_exact(Machine.capacity(limits),LimitBounds.capacity_fits(limits),U32.from_nat(quotient),Grid.product(rest),quotient,inner,
        quotient_word,product_word,scaled_room)
      +rest_word = row_major_sum(limits,rest,remainder,NatDivision.remainder_small(index,inner,inner_positive),inner_room)
      %Equal.sym(List<&2,Views.Axis>,Tasks.piece_cells(extent <> rest),Views.Axis{extent,Grid.product(rest)} <> Tasks.piece_cells(rest),GridPieces.dense_con(extent,rest)) :
        {U32.to_nat(Box.stepped_sum(extent <> rest,ViewPart.view_steps(_),index)) == index : Nat}
      Equal.trans(Nat,U32.to_nat((U32.from_nat(quotient) * Grid.product(rest) + Box.stepped_sum(rest,ViewPart.view_steps(Tasks.piece_cells(rest)),remainder) : U32)),
        Nat.add(Nat.mul(quotient,inner),remainder),index,
        Arithmetic.sum_exact(Machine.capacity(limits),LimitBounds.capacity_fits(limits),(U32.from_nat(quotient) * Grid.product(rest) : U32),
          Box.stepped_sum(rest,ViewPart.view_steps(Tasks.piece_cells(rest)),remainder),Nat.mul(quotient,inner),remainder,scaled,rest_word,sum_room),
        Equal.sym(Nat,index,Nat.add(Nat.mul(quotient,inner),remainder),exact))

# The row-major view of a shape reads every cell of the shape at its own
# index.
def row_major_read(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+limits: Machine.Limits,+shape: List<&2,U32>,+index: U32,
  +inside: {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(shape)) == True{} : Bool},+room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {ViewGather.viewed(~Element,~Logical,~original,ViewGather.Viewing{logical,Tasks.piece_cells(shape),0,[]},index) == original(logical,index) : Element}:
  +cells = Tasks.piece_cells(shape)
  +sum = Box.stepped_sum(shape,ViewPart.view_steps(cells),U32.to_nat(index))
  +word = Arithmetic.to_nat_injective(sum,index,row_major_sum(limits,shape,U32.to_nat(index),inside,room))
  %Equal.sym(U32,Views.mirrored_offset(cells,[],0),0,ViewGather.forward_offset(cells,[],0,{==})) :
    {original(logical,(_ + Box.stepped_sum(Views.axis_extents(cells),ViewPart.view_steps(Views.mirrored_axes(cells,[])),U32.to_nat(index)) : U32)) == original(logical,index) : Element}
  %Equal.sym(List<&2,Views.Axis>,Views.mirrored_axes(cells,[]),cells,ViewGather.forward_axes(cells,[],{==})) :
    {original(logical,(0 + Box.stepped_sum(Views.axis_extents(cells),ViewPart.view_steps(_),U32.to_nat(index)) : U32)) == original(logical,index) : Element}
  %Equal.sym(List<&2,U32>,Views.axis_extents(cells),shape,cells_extents(shape)) :
    {original(logical,(0 + Box.stepped_sum(_,ViewPart.view_steps(cells),U32.to_nat(index)) : U32)) == original(logical,index) : Element}
  %Equal.sym(U32,(0 + sum : U32),sum,Arithmetic.add_zero_left(sum)) : {original(logical,_) == original(logical,index) : Element}
  %Equal.sym(U32,sum,index,word) : {original(logical,_) == original(logical,index) : Element}
  {==}

# ---- The extents of a view of the axes in an order ----

def view_extents(view: Maybe<&2,Views.View>,+fallback: List<&2,U32>) -> List<&2,U32>:
  match view:
    case Some{Views.View{axes,offset}}: Views.axis_extents(axes)
    case None{}: fallback

def axis_extent(found: Maybe<&2,Views.Axis>) -> U32:
  match found:
    case Some{Views.Axis{extent,stride}}: extent
    case None{}: 0

def got_extent(axes: List<&2,Views.Axis>,index: Nat) -> {Layout.extent_at(Views.axis_extents(axes),index) == axis_extent(List.get(&2,Views.Axis,axes,index)) : U32}:
  match axes index:
    case [] index: {==}
    case Views.Axis{extent,stride} <> tail 0n: {==}
    case Views.Axis{extent,stride} <> tail 1n+previous: got_extent(tail,previous)

def prepended(found: Maybe<&2,Views.Axis>,rest: Maybe<&2,Views.View>,+head: U32,+tail: List<&2,U32>,
  head_exact: {axis_extent(found) == head : U32},tail_exact: {view_extents(rest,tail) == tail : List<&2,U32>}) ->
  {view_extents(Views.prepend_axis(found,rest),head <> tail) == head <> tail : List<&2,U32>}:
  match found rest:
    case Some{Views.Axis{+extent,stride}} Some{Views.View{+axes,offset}}:
      %Equal.sym(U32,extent,head,head_exact) : {_ <> Views.axis_extents(axes) == head <> tail : List<&2,U32>}
      %Equal.sym(List<&2,U32>,Views.axis_extents(axes),tail,tail_exact) : {head <> _ == head <> tail : List<&2,U32>}
      {==}
    case Some{axis} None{}: {==}
    case None{} Some{view}: {==}
    case None{} None{}: {==}

def reordered_extents(order: List<&2,Nat>,+axes: List<&2,Views.Axis>,+offset: U32) ->
  {view_extents(Views.reorder(order,axes,offset),Layout.extents_at(Views.axis_extents(axes),order)) == Layout.extents_at(Views.axis_extents(axes),order) : List<&2,U32>}:
  match order:
    case []: {==}
    case +index <> +rest:
      prepended(List.get(&2,Views.Axis,axes,index),Views.reorder(rest,axes,offset),Layout.extent_at(Views.axis_extents(axes),index),
        Layout.extents_at(Views.axis_extents(axes),rest),
        Equal.sym(U32,Layout.extent_at(Views.axis_extents(axes),index),axis_extent(List.get(&2,Views.Axis,axes,index)),got_extent(axes,index)),
        reordered_extents(rest,axes,offset))

def checked_extents(valid: Bool,+axes: List<&2,Views.Axis>,+offset: U32,+order: List<&2,Nat>,+shape: List<&2,U32>,
  +exact: {Views.axis_extents(axes) == shape : List<&2,U32>}) ->
  {view_extents(Views.permute_when(valid,Views.View{axes,offset},order),Layout.extents_at(shape,order)) == Layout.extents_at(shape,order) : List<&2,U32>}:
  match valid:
    case True{}:
      %exact : {view_extents(Views.reorder(order,axes,offset),Layout.extents_at(_,order)) == Layout.extents_at(_,order) : List<&2,U32>}
      reordered_extents(order,axes,offset)
    case False{}: {==}

# The view of the cells of a shape in an order has the extents of the shape
# in that order.
def line_extents(+shape: List<&2,U32>,+order: List<&2,Nat>) ->
  {view_extents(Layout.line_view(shape,order),Layout.extents_at(shape,order)) == Layout.extents_at(shape,order) : List<&2,U32>}:
  +cells = Tasks.piece_cells(shape)
  checked_extents(Nat.is_eq(Views.rank(Views.View{cells,0}),List.length(&2,Nat,order)) && Views.unique_axes(order),cells,0,order,shape,cells_extents(shape))

# ---- The reduction of axes in an order ----

# The input read through the view of its cells with the kept axes first and
# the reduced axes last.
type Ordered<-Logical: Data> is Data:
  Ordered{logical: Logical,view: Maybe<&2,Views.View>}

def viewed_at(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,view: Maybe<&2,Views.View>,+logical: Logical,index: U32) -> Element:
  match view:
    case Some{Views.View{axes,offset}}: ViewGather.viewed(~Element,~Logical,~original,ViewGather.Viewing{logical,axes,offset,[]},index)
    case None{}: original(logical,index)

def through(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,ordered: Ordered<Logical>,index: U32) -> Element:
  Ordered{+logical,view} = ordered
  viewed_at(~Element,~Logical,~original,view,logical,index)

# The reduction of the trailing axes of a tensor of a shape, the first kept
# of its axes kept, is Ready and holds the finished folds of its lines.
def arranged_contract(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+limits: Machine.Limits,+operations: F32,+length: U32,+initial: Accumulator,+kept: Nat,-arranged: Tensors.Tensor<Element>,
  +shape: List<&2,U32>,+shaped: Global.Shaped(~Element,~Logical,~original,logical,shape,arranged),
  +room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},
  +length_exact: {U32.to_nat(length) == ShapeLaws.elements(List.drop(&2,U32,shape,kept)) : Nat},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},
  +kept_room: {Nat.is_le(kept,List.length(&2,U32,shape)) == True{} : Bool},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32}) ->
  Global.Shaped(~Element,~Lines<Accumulator,Logical>,~line_value(~Element,~Accumulator,~operation,~finish,~Logical,~original),Lines{initial,logical,ShapeLaws.elements(List.drop(&2,U32,shape,kept))},
    Layout.extents_at(shape,Layout.every_index(kept,0n)),Layout.reduce_arranged(~Element,~Accumulator,~operation,~finish,operations,length,initial,kept,arranged)):
  KernelGemm.Both{Global.Usable{+model,+fill,+workers,+equation},contract} = shaped
  +outputs = Layout.extents_at(shape,Layout.every_index(kept,0n))
  +outputs_exact = Equal.cong(List<&2,U32>,Nat,items => ShapeLaws.elements(items),outputs,List.take(&2,U32,shape,kept),
    Equal.trans(List<&2,U32>,outputs,List.take(&2,U32,List.drop(&2,U32,shape,0n),kept),List.take(&2,U32,shape,kept),every_take(shape,kept,0n,kept_room),
      Equal.cong(List<&2,U32>,List<&2,U32>,items => List.take(&2,U32,items,kept),List.drop(&2,U32,shape,0n),shape,dropped_none(~U32,shape))))
  %equation : Global.Shaped(~Element,~Lines<Accumulator,Logical>,~line_value(~Element,~Accumulator,~operation,~finish,~Logical,~original),Lines{initial,logical,ShapeLaws.elements(List.drop(&2,U32,shape,kept))},outputs,
    Layout.reduce_arranged(~Element,~Accumulator,~operation,~finish,operations,length,initial,kept,_))
  trailing_contract(~Element,~Accumulator,~operation,~finish,~Logical,~original,logical,limits,operations,length,initial,model,shape,kept,outputs,fill,workers,
    %Equal.sym(Tensors.Tensor<Element>,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers},arranged,equation) : Global.Contract(~Element,~Logical,~original,logical,_) contract,
    room,nonempty,length_exact,cap,cap_ok,count_room,outputs_exact,default_capacity)

# The kept axes first and the reduced axes last, every axis in order: the
# view of the cells is the cells as they lie.
def identity_contract(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+limits: Machine.Limits,+operations: F32,+length: U32,+initial: Accumulator,+model: PiecesModel.Model<Element>,+shape: List<&2,U32>,+kept: List<&2,Nat>,+reduced: List<&2,Nat>,+fill: Element,+workers: U32,
  +contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers}),
  +ordered: {Layout.in_order(List.append(&2,Nat,kept,reduced),0n) == True{} : Bool},+sized: {List.length(&2,Nat,List.append(&2,Nat,kept,reduced)) == List.length(&2,U32,shape) : Nat},
  +room: {Nat.is_le(Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced)))) == True{} : Bool},+length_exact: {U32.to_nat(length) == ShapeLaws.elements(Layout.extents_at(shape,reduced)) : Nat},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced)))),4n),U32.to_nat(cap)) == True{} : Bool},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32}) ->
  Global.Shaped(~Element,~Lines<Accumulator,Ordered<Logical>>,~line_value(~Element,~Accumulator,~operation,~finish,~Ordered<Logical>,~through(~Element,~Logical,~original)),Lines{initial,Ordered{logical,Some{Views.View{Tasks.piece_cells(shape),0}}},ShapeLaws.elements(Layout.extents_at(shape,reduced))},Layout.extents_at(shape,kept),Layout.reduce_arranged(~Element,~Accumulator,~operation,~finish,operations,length,initial,List.length(&2,Nat,kept),Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers})):
  +sizes = Equal.trans(Nat,Nat.add(List.length(&2,Nat,kept),List.length(&2,Nat,reduced)),List.length(&2,Nat,List.append(&2,Nat,kept,reduced)),List.length(&2,U32,shape),
    Equal.sym(Nat,List.length(&2,Nat,List.append(&2,Nat,kept,reduced)),Nat.add(List.length(&2,Nat,kept),List.length(&2,Nat,reduced)),ListReadback.append_length(~Nat,kept,reduced)),sized)
  +kept_room = Held.nat_le_right_moved(List.length(&2,Nat,kept),List.length(&2,U32,shape),Nat.add(List.length(&2,Nat,kept),List.length(&2,Nat,reduced)),Equal.sym(Nat,Nat.add(List.length(&2,Nat,kept),List.length(&2,Nat,reduced)),List.length(&2,U32,shape),sizes),Order.le_add(List.length(&2,Nat,kept),List.length(&2,Nat,reduced)))
  +kept_every = in_order_every(kept,0n,ordered_left(kept,reduced,0n,ordered))
  +reduced_every = in_order_every(reduced,List.length(&2,Nat,kept),ordered_right(kept,reduced,0n,ordered))
  +kept_take = Equal.trans(List<&2,U32>,Layout.extents_at(shape,kept),Layout.extents_at(shape,Layout.every_index(List.length(&2,Nat,kept),0n)),List.take(&2,U32,shape,List.length(&2,Nat,kept)),
    Equal.cong(List<&2,Nat>,List<&2,U32>,axes => Layout.extents_at(shape,axes),kept,Layout.every_index(List.length(&2,Nat,kept),0n),kept_every),
    Equal.trans(List<&2,U32>,Layout.extents_at(shape,Layout.every_index(List.length(&2,Nat,kept),0n)),List.take(&2,U32,List.drop(&2,U32,shape,0n),List.length(&2,Nat,kept)),List.take(&2,U32,shape,List.length(&2,Nat,kept)),
      every_take(shape,List.length(&2,Nat,kept),0n,kept_room),
      Equal.cong(List<&2,U32>,List<&2,U32>,items => List.take(&2,U32,items,List.length(&2,Nat,kept)),List.drop(&2,U32,shape,0n),shape,dropped_none(~U32,shape))))
  +reduced_drop = Equal.trans(List<&2,U32>,Layout.extents_at(shape,reduced),Layout.extents_at(shape,Layout.every_index(List.length(&2,Nat,reduced),List.length(&2,Nat,kept))),List.drop(&2,U32,shape,List.length(&2,Nat,kept)),
    Equal.cong(List<&2,Nat>,List<&2,U32>,axes => Layout.extents_at(shape,axes),reduced,Layout.every_index(List.length(&2,Nat,reduced),List.length(&2,Nat,kept)),reduced_every),
    every_extents(List.length(&2,Nat,reduced),List.length(&2,Nat,kept),shape,sizes))
  +elements_exact = {
    %Equal.sym(List<&2,U32>,Layout.extents_at(shape,kept),List.take(&2,U32,shape,List.length(&2,Nat,kept)),kept_take) : {ShapeLaws.elements(shape) == Nat.mul(ShapeLaws.elements(_),ShapeLaws.elements(Layout.extents_at(shape,reduced))) : Nat}
    %Equal.sym(List<&2,U32>,Layout.extents_at(shape,reduced),List.drop(&2,U32,shape,List.length(&2,Nat,kept)),reduced_drop) :
      {ShapeLaws.elements(shape) == Nat.mul(ShapeLaws.elements(List.take(&2,U32,shape,List.length(&2,Nat,kept))),ShapeLaws.elements(_)) : Nat}
    CountCompletion.elements_split(shape,List.length(&2,Nat,kept)) : {ShapeLaws.elements(shape) == Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced))) : Nat}}
  +room_shape = Held.nat_le_moved(ShapeLaws.elements(shape),Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced))),U32.to_nat(Machine.capacity(limits)),elements_exact,room)
  +nonempty_shape = Held.nat_lt_moved(0n,ShapeLaws.elements(shape),Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced))),elements_exact,nonempty)
  +count_room_shape = Held.nat_le_moved(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),Nat.add(Nat.double(Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced)))),4n),U32.to_nat(cap),
    Equal.cong(Nat,Nat,number => Nat.add(Nat.double(number),4n),ShapeLaws.elements(shape),Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced))),elements_exact),count_room)
  +lengthed = Equal.cong(List<&2,U32>,Nat,items => ShapeLaws.elements(items),List.drop(&2,U32,shape,List.length(&2,Nat,kept)),Layout.extents_at(shape,reduced),Equal.sym(List<&2,U32>,Layout.extents_at(shape,reduced),List.drop(&2,U32,shape,List.length(&2,Nat,kept)),reduced_drop))
  +length_word = Equal.trans(Nat,U32.to_nat(length),ShapeLaws.elements(Layout.extents_at(shape,reduced)),ShapeLaws.elements(List.drop(&2,U32,shape,List.length(&2,Nat,kept))),length_exact,
    Equal.sym(Nat,ShapeLaws.elements(List.drop(&2,U32,shape,List.length(&2,Nat,kept))),ShapeLaws.elements(Layout.extents_at(shape,reduced)),lengthed))
  +cells = Tasks.piece_cells(shape)
  +identity = {Ordered{logical,Some{Views.View{cells,0}}} : Ordered<Logical>}
  +shaped = Global.restated_below(~Element,~Logical,~Ordered<Logical>,~original,~through(~Element,~Logical,~original),~ShapeLaws.elements(shape),logical,identity,
    index => inside => Equal.sym(Element,ViewGather.viewed(~Element,~Logical,~original,ViewGather.Viewing{logical,cells,0,[]},index),original(logical,index),
      row_major_read(~Element,~Logical,~original,logical,limits,shape,index,inside,room_shape)),
    shape,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers},Global.shaped(~Element,~Logical,~original,logical,shape,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers},{==},{==},contract),Order.reflexive(ShapeLaws.elements(shape)),Machine.capacity(limits),room_shape)
  +kept_outputs = Equal.cong(List<&2,Nat>,List<&2,U32>,axes => Layout.extents_at(shape,axes),kept,Layout.every_index(List.length(&2,Nat,kept),0n),kept_every)
  %lengthed : Global.Shaped(~Element,~Lines<Accumulator,Ordered<Logical>>,~line_value(~Element,~Accumulator,~operation,~finish,~Ordered<Logical>,~through(~Element,~Logical,~original)),Lines{initial,identity,_},Layout.extents_at(shape,kept),Layout.reduce_arranged(~Element,~Accumulator,~operation,~finish,operations,length,initial,List.length(&2,Nat,kept),Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers}))
  %Equal.sym(List<&2,U32>,Layout.extents_at(shape,kept),Layout.extents_at(shape,Layout.every_index(List.length(&2,Nat,kept),0n)),kept_outputs) :
    Global.Shaped(~Element,~Lines<Accumulator,Ordered<Logical>>,~line_value(~Element,~Accumulator,~operation,~finish,~Ordered<Logical>,~through(~Element,~Logical,~original)),Lines{initial,identity,ShapeLaws.elements(List.drop(&2,U32,shape,List.length(&2,Nat,kept)))},_,Layout.reduce_arranged(~Element,~Accumulator,~operation,~finish,operations,length,initial,List.length(&2,Nat,kept),Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers}))
  arranged_contract(~Element,~Accumulator,~operation,~finish,~Ordered<Logical>,~through(~Element,~Logical,~original),identity,limits,operations,length,initial,List.length(&2,Nat,kept),Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers},shape,shaped,
    room_shape,nonempty_shape,length_word,cap,cap_ok,count_room_shape,kept_room,default_capacity)

# ---- The kept and the reduced axes ----

def chosen_length(member: Bool,+index: Nat,+kept: List<&2,Nat>,+reduced: List<&2,Nat>,+rest: Nat,
  +before: {Nat.add(List.length(&2,Nat,kept),List.length(&2,Nat,reduced)) == rest : Nat}) ->
  {Nat.add(List.length(&2,Nat,Layout.chosen(member,False{},index,kept)),List.length(&2,Nat,Layout.chosen(member,True{},index,reduced))) == 1n+rest : Nat}:
  match member:
    case True{}:
      Equal.trans(Nat,Nat.add(List.length(&2,Nat,kept),1n+List.length(&2,Nat,reduced)),1n+Nat.add(List.length(&2,Nat,kept),List.length(&2,Nat,reduced)),1n+rest,
        Addition.successor_right(List.length(&2,Nat,kept),List.length(&2,Nat,reduced)),
        Equal.cong(Nat,Nat,count => 1n+count,Nat.add(List.length(&2,Nat,kept),List.length(&2,Nat,reduced)),rest,before))
    case False{}:
      Equal.cong(Nat,Nat,count => 1n+count,Nat.add(List.length(&2,Nat,kept),List.length(&2,Nat,reduced)),rest,before)

# Every axis is kept or reduced.
def partition_length(remaining: Nat,+index: Nat,+axes: List<&2,Nat>) ->
  {Nat.add(List.length(&2,Nat,Layout.axes_where(remaining,index,axes,False{})),List.length(&2,Nat,Layout.axes_where(remaining,index,axes,True{}))) == remaining : Nat}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      chosen_length(Views.contains_index(axes,index),index,Layout.axes_where(rest,1n+index,axes,False{}),Layout.axes_where(rest,1n+index,axes,True{}),rest,
        partition_length(rest,1n+index,axes))

def chosen_elements(member: Bool,+shape: List<&2,U32>,+index: Nat,+kept: List<&2,Nat>,+reduced: List<&2,Nat>,
  +before: {Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced))) == ShapeLaws.elements(List.drop(&2,U32,shape,1n+index)) : Nat},
  +next: {List.drop(&2,U32,shape,index) == Layout.extent_at(shape,index) <> List.drop(&2,U32,shape,1n+index) : List<&2,U32>}) ->
  {Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,Layout.chosen(member,False{},index,kept))),ShapeLaws.elements(Layout.extents_at(shape,Layout.chosen(member,True{},index,reduced)))) ==
    ShapeLaws.elements(List.drop(&2,U32,shape,index)) : Nat}:
  match member:
    case True{}:
      +extent = U32.to_nat(Layout.extent_at(shape,index))
      +left = ShapeLaws.elements(Layout.extents_at(shape,kept))
      +right = ShapeLaws.elements(Layout.extents_at(shape,reduced))
      +rest = ShapeLaws.elements(List.drop(&2,U32,shape,1n+index))
      Equal.trans(Nat,Nat.mul(left,Nat.mul(extent,right)),Nat.mul(extent,Nat.mul(left,right)),ShapeLaws.elements(List.drop(&2,U32,shape,index)),RowMajor.swap_factors(left,extent,right),
        Equal.trans(Nat,Nat.mul(extent,Nat.mul(left,right)),Nat.mul(extent,rest),ShapeLaws.elements(List.drop(&2,U32,shape,index)),
          Equal.cong(Nat,Nat,number => Nat.mul(extent,number),Nat.mul(left,right),rest,before),
          Equal.sym(Nat,ShapeLaws.elements(List.drop(&2,U32,shape,index)),Nat.mul(extent,rest),
            Equal.cong(List<&2,U32>,Nat,items => ShapeLaws.elements(items),List.drop(&2,U32,shape,index),Layout.extent_at(shape,index) <> List.drop(&2,U32,shape,1n+index),next))))
    case False{}:
      +extent = U32.to_nat(Layout.extent_at(shape,index))
      +left = ShapeLaws.elements(Layout.extents_at(shape,kept))
      +right = ShapeLaws.elements(Layout.extents_at(shape,reduced))
      +rest = ShapeLaws.elements(List.drop(&2,U32,shape,1n+index))
      Equal.trans(Nat,Nat.mul(Nat.mul(extent,left),right),Nat.mul(extent,Nat.mul(left,right)),ShapeLaws.elements(List.drop(&2,U32,shape,index)),RowMajor.assoc(extent,left,right),
        Equal.trans(Nat,Nat.mul(extent,Nat.mul(left,right)),Nat.mul(extent,rest),ShapeLaws.elements(List.drop(&2,U32,shape,index)),
          Equal.cong(Nat,Nat,number => Nat.mul(extent,number),Nat.mul(left,right),rest,before),
          Equal.sym(Nat,ShapeLaws.elements(List.drop(&2,U32,shape,index)),Nat.mul(extent,rest),
            Equal.cong(List<&2,U32>,Nat,items => ShapeLaws.elements(items),List.drop(&2,U32,shape,index),Layout.extent_at(shape,index) <> List.drop(&2,U32,shape,1n+index),next))))

# The kept and the reduced axes from an index on hold the cells of the shape
# from there.
def partition_elements(+shape: List<&2,U32>,remaining: Nat,+index: Nat,+axes: List<&2,Nat>,+sized: {Nat.add(index,remaining) == List.length(&2,U32,shape) : Nat}) ->
  {Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,Layout.axes_where(remaining,index,axes,False{}))),ShapeLaws.elements(Layout.extents_at(shape,Layout.axes_where(remaining,index,axes,True{})))) ==
    ShapeLaws.elements(List.drop(&2,U32,shape,index)) : Nat}:
  match remaining:
    case 0n:
      +at_end = Equal.trans(Nat,index,Nat.add(index,0n),List.length(&2,U32,shape),Equal.sym(Nat,Nat.add(index,0n),index,Addition.zero_right(index)),sized)
      %Equal.sym(Nat,index,List.length(&2,U32,shape),at_end) : {1n == ShapeLaws.elements(List.drop(&2,U32,shape,_)) : Nat}
      %Equal.sym(List<&2,U32>,List.drop(&2,U32,shape,List.length(&2,U32,shape)),[],dropped_all(~U32,shape)) : {1n == ShapeLaws.elements(_) : Nat}
      {==}
    case 1n+ +rest:
      +later = {%Addition.successor_right(index,rest) : {_ == List.length(&2,U32,shape) : Nat} sized : {Nat.add(1n+index,rest) == List.length(&2,U32,shape) : Nat}}
      chosen_elements(Views.contains_index(axes,index),shape,index,Layout.axes_where(rest,1n+index,axes,False{}),Layout.axes_where(rest,1n+index,axes,True{}),
        partition_elements(shape,rest,1n+index,axes,later),drop_next(shape,index,Order.below_complement(index,rest,List.length(&2,U32,shape),sized)))

def kept_axes(+shape: List<&2,U32>,+axes: List<&2,U32>) -> List<&2,Nat>:
  Layout.axes_where(List.length(&2,U32,shape),0n,Layout.naturals(axes),False{})

def reduced_axes(+shape: List<&2,U32>,+axes: List<&2,U32>) -> List<&2,Nat>:
  Layout.axes_where(List.length(&2,U32,shape),0n,Layout.naturals(axes),True{})

# ---- The view of the kept and the reduced axes ----

def contains_append(left: List<&2,Nat>,+right: List<&2,Nat>,+axis: Nat) ->
  {Views.contains_index(List.append(&2,Nat,left,right),axis) == Views.contains_index(left,axis) || Views.contains_index(right,axis) : Bool}:
  match left:
    case []: {==}
    case +head <> +tail:
      %Equal.sym(Bool,Views.contains_index(List.append(&2,Nat,tail,right),axis),Views.contains_index(tail,axis) || Views.contains_index(right,axis),contains_append(tail,right,axis)) :
        {Nat.is_eq(head,axis) || _ == (Nat.is_eq(head,axis) || Views.contains_index(tail,axis)) || Views.contains_index(right,axis) : Bool}
      Boolean.or_associative(Nat.is_eq(head,axis),Views.contains_index(tail,axis),Views.contains_index(right,axis))

def chosen_absent(member: Bool,listed: Bool,+index: Nat,+rest: List<&2,Nat>,+axis: Nat,+different: {Nat.is_eq(index,axis) == False{} : Bool},
  +absent: {Views.contains_index(rest,axis) == False{} : Bool}) -> {Views.contains_index(Layout.chosen(member,listed,index,rest),axis) == False{} : Bool}:
  match member listed:
    case True{} True{}:
      %Equal.sym(Bool,Nat.is_eq(index,axis),False{},different) : {_ || Views.contains_index(rest,axis) == False{} : Bool}
      absent
    case False{} False{}:
      %Equal.sym(Bool,Nat.is_eq(index,axis),False{},different) : {_ || Views.contains_index(rest,axis) == False{} : Bool}
      absent
    case True{} False{}: absent
    case False{} True{}: absent

# The axes from an index on hold no axis below it.
def where_above(remaining: Nat,+index: Nat,+axes: List<&2,Nat>,+listed: Bool,+axis: Nat,+below: {Nat.is_lt(axis,index) == True{} : Bool}) ->
  {Views.contains_index(Layout.axes_where(remaining,index,axes,listed),axis) == False{} : Bool}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      chosen_absent(Views.contains_index(axes,index),listed,index,Layout.axes_where(rest,1n+index,axes,listed),axis,
        Equal.trans(Bool,Nat.is_eq(index,axis),Nat.is_eq(axis,index),False{},Order.equal_symmetric(index,axis),Order.less_not_equal(axis,index,below)),
        where_above(rest,1n+index,axes,listed,axis,Order.lt_succ(axis,index,Bounds.strict_le(axis,index,below))))

# An axis absent from two lists that hold no axis twice may stand between them.
def unique_inserted(left: List<&2,Nat>,+axis: Nat,+right: List<&2,Nat>,+unique: {Views.unique_axes(List.append(&2,Nat,left,right)) == True{} : Bool},
  +absent: {Views.contains_index(List.append(&2,Nat,left,right),axis) == False{} : Bool}) -> {Views.unique_axes(List.append(&2,Nat,left,axis <> right)) == True{} : Bool}:
  match left:
    case []:
      %Equal.sym(Bool,Views.contains_index(right,axis),False{},absent) : {Bool.not(_) && Views.unique_axes(right) == True{} : Bool}
      unique
    case +head <> +tail:
      +rest = List.append(&2,Nat,tail,right)
      +head_absent = Boolean.not_true(Views.contains_index(rest,head),Boolean.left(Bool.not(Views.contains_index(rest,head)),Views.unique_axes(rest),unique))
      +rest_unique = Boolean.right(Bool.not(Views.contains_index(rest,head)),Views.unique_axes(rest),unique)
      +head_other = Boolean.or_false_left(Nat.is_eq(head,axis),Views.contains_index(rest,axis),absent)
      +rest_absent = Boolean.or_false_right(Nat.is_eq(head,axis),Views.contains_index(rest,axis),absent)
      +parts_absent = Equal.trans(Bool,Views.contains_index(tail,head) || Views.contains_index(right,head),Views.contains_index(rest,head),False{},
        Equal.sym(Bool,Views.contains_index(rest,head),Views.contains_index(tail,head) || Views.contains_index(right,head),contains_append(tail,right,head)),head_absent)
      +tail_absent = Boolean.or_false_left(Views.contains_index(tail,head),Views.contains_index(right,head),parts_absent)
      +right_absent = Boolean.or_false_right(Views.contains_index(tail,head),Views.contains_index(right,head),parts_absent)
      +axis_other = Equal.trans(Bool,Nat.is_eq(axis,head),Nat.is_eq(head,axis),False{},Order.equal_symmetric(axis,head),head_other)
      +inserted_absent = Equal.trans(Bool,Views.contains_index(List.append(&2,Nat,tail,axis <> right),head),Views.contains_index(tail,head) || (Nat.is_eq(axis,head) || Views.contains_index(right,head)),False{},
        contains_append(tail,axis <> right,head),
        {%Equal.sym(Bool,Views.contains_index(tail,head),False{},tail_absent) : {_ || (Nat.is_eq(axis,head) || Views.contains_index(right,head)) == False{} : Bool}
          %Equal.sym(Bool,Nat.is_eq(axis,head),False{},axis_other) : {_ || Views.contains_index(right,head) == False{} : Bool}
          right_absent : {Views.contains_index(tail,head) || (Nat.is_eq(axis,head) || Views.contains_index(right,head)) == False{} : Bool}})
      %Equal.sym(Bool,Views.contains_index(List.append(&2,Nat,tail,axis <> right),head),False{},inserted_absent) : {Bool.not(_) && Views.unique_axes(List.append(&2,Nat,tail,axis <> right)) == True{} : Bool}
      unique_inserted(tail,axis,right,rest_unique,rest_absent)

def chosen_unique(member: Bool,+index: Nat,+kept: List<&2,Nat>,+reduced: List<&2,Nat>,+unique: {Views.unique_axes(List.append(&2,Nat,kept,reduced)) == True{} : Bool},
  +kept_absent: {Views.contains_index(kept,index) == False{} : Bool},+reduced_absent: {Views.contains_index(reduced,index) == False{} : Bool}) ->
  {Views.unique_axes(List.append(&2,Nat,Layout.chosen(member,False{},index,kept),Layout.chosen(member,True{},index,reduced))) == True{} : Bool}:
  match member:
    case True{}:
      unique_inserted(kept,index,reduced,unique,
        Equal.trans(Bool,Views.contains_index(List.append(&2,Nat,kept,reduced),index),Views.contains_index(kept,index) || Views.contains_index(reduced,index),False{},contains_append(kept,reduced,index),
          {%Equal.sym(Bool,Views.contains_index(kept,index),False{},kept_absent) : {_ || Views.contains_index(reduced,index) == False{} : Bool} reduced_absent :
            {Views.contains_index(kept,index) || Views.contains_index(reduced,index) == False{} : Bool}}))
    case False{}:
      +absent = Equal.trans(Bool,Views.contains_index(List.append(&2,Nat,kept,reduced),index),Views.contains_index(kept,index) || Views.contains_index(reduced,index),False{},contains_append(kept,reduced,index),
        {%Equal.sym(Bool,Views.contains_index(kept,index),False{},kept_absent) : {_ || Views.contains_index(reduced,index) == False{} : Bool} reduced_absent :
          {Views.contains_index(kept,index) || Views.contains_index(reduced,index) == False{} : Bool}})
      %Equal.sym(Bool,Views.contains_index(List.append(&2,Nat,kept,reduced),index),False{},absent) : {Bool.not(_) && Views.unique_axes(List.append(&2,Nat,kept,reduced)) == True{} : Bool}
      unique

# The kept axes and then the reduced axes hold no axis twice.
def partition_unique(remaining: Nat,+index: Nat,+axes: List<&2,Nat>) -> {Views.unique_axes(List.append(&2,Nat,Layout.axes_where(remaining,index,axes,False{}),Layout.axes_where(remaining,index,axes,True{}))) == True{} : Bool}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +above = Order.lt_succ(index,index,Order.reflexive(index))
      chosen_unique(Views.contains_index(axes,index),index,Layout.axes_where(rest,1n+index,axes,False{}),Layout.axes_where(rest,1n+index,axes,True{}),partition_unique(rest,1n+index,axes),
        where_above(rest,1n+index,axes,False{},index,above),where_above(rest,1n+index,axes,True{},index,above))

# Every axis below a bound.
def below_all(order: List<&2,Nat>,+bound: Nat) -> Bool:
  match order:
    case []: True{}
    case axis <> rest: Nat.is_lt(axis,bound) && below_all(rest,bound)

def chosen_below(member: Bool,listed: Bool,+index: Nat,+rest: List<&2,Nat>,+bound: Nat,+inside: {Nat.is_lt(index,bound) == True{} : Bool},
  +rest_below: {below_all(rest,bound) == True{} : Bool}) -> {below_all(Layout.chosen(member,listed,index,rest),bound) == True{} : Bool}:
  match member listed:
    case True{} True{}: Boolean.and_true(Nat.is_lt(index,bound),below_all(rest,bound),inside,rest_below)
    case False{} False{}: Boolean.and_true(Nat.is_lt(index,bound),below_all(rest,bound),inside,rest_below)
    case True{} False{}: rest_below
    case False{} True{}: rest_below

def where_below(remaining: Nat,+index: Nat,+axes: List<&2,Nat>,+listed: Bool) -> {below_all(Layout.axes_where(remaining,index,axes,listed),Nat.add(index,remaining)) == True{} : Bool}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      chosen_below(Views.contains_index(axes,index),listed,index,Layout.axes_where(rest,1n+index,axes,listed),Nat.add(index,1n+rest),Order.below_sum(index,rest),
        %Equal.sym(Nat,Nat.add(index,1n+rest),1n+Nat.add(index,rest),Addition.successor_right(index,rest)) :
          {below_all(Layout.axes_where(rest,1n+index,axes,listed),_) == True{} : Bool}
        where_below(rest,1n+index,axes,listed))

def below_appended(left: List<&2,Nat>,+right: List<&2,Nat>,+bound: Nat,+left_below: {below_all(left,bound) == True{} : Bool},+right_below: {below_all(right,bound) == True{} : Bool}) ->
  {below_all(List.append(&2,Nat,left,right),bound) == True{} : Bool}:
  match left:
    case []: right_below
    case +axis <> +rest:
      Boolean.and_true(Nat.is_lt(axis,bound),below_all(List.append(&2,Nat,rest,right),bound),Boolean.left(Nat.is_lt(axis,bound),below_all(rest,bound),left_below),
        below_appended(rest,right,bound,Boolean.right(Nat.is_lt(axis,bound),below_all(rest,bound),left_below),right_below))

# The axes of the cells of a shape at an order.
def order_axes(order: List<&2,Nat>,+shape: List<&2,U32>) -> List<&2,Views.Axis>:
  match order:
    case []: []
    case +axis <> rest: Views.Axis{Layout.extent_at(shape,axis),Grid.product(List.drop(&2,U32,shape,1n+axis))} <> order_axes(rest,shape)

def cell_get(shape: List<&2,U32>,axis: Nat,+inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool}) ->
  {List.get(&2,Views.Axis,Tasks.piece_cells(shape),axis) == Some{Views.Axis{Layout.extent_at(shape,axis),Grid.product(List.drop(&2,U32,shape,1n+axis))}} : Maybe<&2,Views.Axis>}:
  match shape axis:
    case [] axis:
      Empty.absurd({List.get(&2,Views.Axis,Tasks.piece_cells([]),axis) == Some{Views.Axis{Layout.extent_at([],axis),Grid.product(List.drop(&2,U32,[],1n+axis))}} : Maybe<&2,Views.Axis>},
        Order.below_zero(axis,inside))
    case +extent <> +rest 0n:
      %Equal.sym(List<&2,Views.Axis>,Tasks.piece_cells(extent <> rest),Views.Axis{extent,Grid.product(rest)} <> Tasks.piece_cells(rest),GridPieces.dense_con(extent,rest)) :
        {List.get(&2,Views.Axis,_,0n) == Some{Views.Axis{Layout.extent_at(extent <> rest,0n),Grid.product(List.drop(&2,U32,extent <> rest,1n))}} : Maybe<&2,Views.Axis>}
      %Equal.sym(List<&2,U32>,List.drop(&2,U32,rest,0n),rest,dropped_none(~U32,rest)) :
        {Some{Views.Axis{extent,Grid.product(rest)}} == Some{Views.Axis{extent,Grid.product(_)}} : Maybe<&2,Views.Axis>}
      {==}
    case +extent <> +rest 1n+ +previous:
      %Equal.sym(List<&2,Views.Axis>,Tasks.piece_cells(extent <> rest),Views.Axis{extent,Grid.product(rest)} <> Tasks.piece_cells(rest),GridPieces.dense_con(extent,rest)) :
        {List.get(&2,Views.Axis,_,1n+previous) == Some{Views.Axis{Layout.extent_at(extent <> rest,1n+previous),Grid.product(List.drop(&2,U32,extent <> rest,2n+previous))}} : Maybe<&2,Views.Axis>}
      cell_get(rest,previous,inside)

def reorder_axes_view(order: List<&2,Nat>,+shape: List<&2,U32>,+below: {below_all(order,List.length(&2,U32,shape)) == True{} : Bool}) ->
  {Views.reorder(order,Tasks.piece_cells(shape),0) == Some{Views.View{order_axes(order,shape),0}} : Maybe<&2,Views.View>}:
  match order:
    case []: {==}
    case +axis <> +rest:
      +rank = List.length(&2,U32,shape)
      +axis_view = {Views.Axis{Layout.extent_at(shape,axis),Grid.product(List.drop(&2,U32,shape,1n+axis))} : Views.Axis}
      %Equal.sym(Maybe<&2,Views.Axis>,List.get(&2,Views.Axis,Tasks.piece_cells(shape),axis),Some{axis_view},
          cell_get(shape,axis,Boolean.left(Nat.is_lt(axis,rank),below_all(rest,rank),below))) :
        {Views.prepend_axis(_,Views.reorder(rest,Tasks.piece_cells(shape),0)) == Some{Views.View{axis_view <> order_axes(rest,shape),0}} : Maybe<&2,Views.View>}
      %Equal.sym(Maybe<&2,Views.View>,Views.reorder(rest,Tasks.piece_cells(shape),0),Some{Views.View{order_axes(rest,shape),0}},
          reorder_axes_view(rest,shape,Boolean.right(Nat.is_lt(axis,rank),below_all(rest,rank),below))) :
        {Views.prepend_axis(Some{axis_view},_) == Some{Views.View{axis_view <> order_axes(rest,shape),0}} : Maybe<&2,Views.View>}
      {==}

def order_axes_extents(order: List<&2,Nat>,+shape: List<&2,U32>) -> {Views.axis_extents(order_axes(order,shape)) == Layout.extents_at(shape,order) : List<&2,U32>}:
  match order:
    case []: {==}
    case +axis <> +rest:
      Equal.cong(List<&2,U32>,List<&2,U32>,tail => Layout.extent_at(shape,axis) <> tail,Views.axis_extents(order_axes(rest,shape)),Layout.extents_at(shape,rest),order_axes_extents(rest,shape))

# The view of the cells of a shape with the kept axes first and the reduced
# axes last is their axes in that order.
def line_picked(+shape: List<&2,U32>,+axes: List<&2,Nat>) ->
  {Layout.line_view(shape,List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{}))) == Some{Views.View{order_axes(List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{})),shape),0}} : Maybe<&2,Views.View>}:
  +rank = List.length(&2,U32,shape)
  +cells = Tasks.piece_cells(shape)
  +order = List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{}))
  +ranked = Equal.cong(List<&2,U32>,Nat,items => List.length(&2,U32,items),Views.axis_extents(cells),shape,cells_extents(shape))
  +sized = Equal.trans(Nat,List.length(&2,Nat,order),Nat.add(List.length(&2,Nat,Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{})),List.length(&2,Nat,Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{}))),rank,ListReadback.append_length(~Nat,Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{})),
    partition_length(rank,0n,axes))
  +valid = {
    %Equal.sym(Nat,List.length(&2,U32,Views.axis_extents(cells)),rank,ranked) : {Nat.is_eq(_,List.length(&2,Nat,order)) && Views.unique_axes(order) == True{} : Bool}
    %Equal.sym(Nat,List.length(&2,Nat,order),rank,sized) : {Nat.is_eq(rank,_) && Views.unique_axes(order) == True{} : Bool}
    %Equal.sym(Bool,Nat.is_eq(rank,rank),True{},Order.equal_reflexive(rank)) : {_ && Views.unique_axes(order) == True{} : Bool}
    partition_unique(rank,0n,axes) : {Nat.is_eq(Views.rank(Views.View{cells,0}),List.length(&2,Nat,order)) && Views.unique_axes(order) == True{} : Bool}}
  %Equal.sym(Bool,Nat.is_eq(Views.rank(Views.View{cells,0}),List.length(&2,Nat,order)) && Views.unique_axes(order),True{},valid) :
    {Views.permute_when(_,Views.View{cells,0},order) == Some{Views.View{order_axes(order,shape),0}} : Maybe<&2,Views.View>}
  reorder_axes_view(order,shape,below_appended(Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{}),rank,where_below(rank,0n,axes,False{}),where_below(rank,0n,axes,True{})))

# ---- The span of the cells in the order of the kept and the reduced axes ----

def appended_span(left: List<&2,Nat>,+right: List<&2,Nat>,+shape: List<&2,U32>) ->
  {ViewBounds.last_span(order_axes(List.append(&2,Nat,left,right),shape)) == Nat.add(ViewBounds.last_span(order_axes(left,shape)),ViewBounds.last_span(order_axes(right,shape))) : Nat}:
  match left:
    case []: {==}
    case +axis <> +rest:
      +term = Nat.mul(U32.to_nat((Layout.extent_at(shape,axis) - 1 : U32)),U32.to_nat(Grid.product(List.drop(&2,U32,shape,1n+axis))))
      Equal.trans(Nat,Nat.add(term,ViewBounds.last_span(order_axes(List.append(&2,Nat,rest,right),shape))),Nat.add(term,Nat.add(ViewBounds.last_span(order_axes(rest,shape)),ViewBounds.last_span(order_axes(right,shape)))),Nat.add(Nat.add(term,ViewBounds.last_span(order_axes(rest,shape))),ViewBounds.last_span(order_axes(right,shape))),
        Equal.cong(Nat,Nat,number => Nat.add(term,number),ViewBounds.last_span(order_axes(List.append(&2,Nat,rest,right),shape)),Nat.add(ViewBounds.last_span(order_axes(rest,shape)),ViewBounds.last_span(order_axes(right,shape))),appended_span(rest,right,shape)),
        Equal.sym(Nat,Nat.add(Nat.add(term,ViewBounds.last_span(order_axes(rest,shape))),ViewBounds.last_span(order_axes(right,shape))),Nat.add(term,Nat.add(ViewBounds.last_span(order_axes(rest,shape)),ViewBounds.last_span(order_axes(right,shape)))),Addition.associative(term,ViewBounds.last_span(order_axes(rest,shape)),ViewBounds.last_span(order_axes(right,shape)))))

def chosen_span(member: Bool,+shape: List<&2,U32>,+index: Nat,+kept: List<&2,Nat>,+reduced: List<&2,Nat>,+rest: Nat,
  +before: {Nat.add(ViewBounds.last_span(order_axes(kept,shape)),ViewBounds.last_span(order_axes(reduced,shape))) == rest : Nat}) ->
  {Nat.add(ViewBounds.last_span(order_axes(Layout.chosen(member,False{},index,kept),shape)),ViewBounds.last_span(order_axes(Layout.chosen(member,True{},index,reduced),shape))) == Nat.add(Nat.mul(U32.to_nat((Layout.extent_at(shape,index) - 1 : U32)),U32.to_nat(Grid.product(List.drop(&2,U32,shape,1n+index)))),rest) : Nat}:
  match member:
    case True{}:
      +term = Nat.mul(U32.to_nat((Layout.extent_at(shape,index) - 1 : U32)),U32.to_nat(Grid.product(List.drop(&2,U32,shape,1n+index))))
      +left = ViewBounds.last_span(order_axes(kept,shape))
      +right = ViewBounds.last_span(order_axes(reduced,shape))
      Equal.trans(Nat,Nat.add(left,Nat.add(term,right)),Nat.add(Nat.add(left,term),right),Nat.add(term,rest),
        Equal.sym(Nat,Nat.add(Nat.add(left,term),right),Nat.add(left,Nat.add(term,right)),Addition.associative(left,term,right)),
        Equal.trans(Nat,Nat.add(Nat.add(left,term),right),Nat.add(Nat.add(term,left),right),Nat.add(term,rest),
          Equal.cong(Nat,Nat,number => Nat.add(number,right),Nat.add(left,term),Nat.add(term,left),Order.commutative(left,term)),
          Equal.trans(Nat,Nat.add(Nat.add(term,left),right),Nat.add(term,Nat.add(left,right)),Nat.add(term,rest),Addition.associative(term,left,right),
            Equal.cong(Nat,Nat,number => Nat.add(term,number),Nat.add(left,right),rest,before))))
    case False{}:
      +term = Nat.mul(U32.to_nat((Layout.extent_at(shape,index) - 1 : U32)),U32.to_nat(Grid.product(List.drop(&2,U32,shape,1n+index))))
      +left = ViewBounds.last_span(order_axes(kept,shape))
      +right = ViewBounds.last_span(order_axes(reduced,shape))
      Equal.trans(Nat,Nat.add(Nat.add(term,left),right),Nat.add(term,Nat.add(left,right)),Nat.add(term,rest),Addition.associative(term,left,right),
        Equal.cong(Nat,Nat,number => Nat.add(term,number),Nat.add(left,right),rest,before))

# The kept and the reduced axes from an index on span the axes from there.
def partition_span(remaining: Nat,+index: Nat,+axes: List<&2,Nat>,+shape: List<&2,U32>) ->
  {Nat.add(ViewBounds.last_span(order_axes(Layout.axes_where(remaining,index,axes,False{}),shape)),ViewBounds.last_span(order_axes(Layout.axes_where(remaining,index,axes,True{}),shape))) == ViewBounds.last_span(order_axes(Layout.every_index(remaining,index),shape)) : Nat}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      chosen_span(Views.contains_index(axes,index),shape,index,Layout.axes_where(rest,1n+index,axes,False{}),Layout.axes_where(rest,1n+index,axes,True{}),
        ViewBounds.last_span(order_axes(Layout.every_index(rest,1n+index),shape)),partition_span(rest,1n+index,axes,shape))

def shifted_axes(count: Nat,+index: Nat,+extent: U32,+rest: List<&2,U32>) ->
  {order_axes(Layout.every_index(count,1n+index),extent <> rest) == order_axes(Layout.every_index(count,index),rest) : List<&2,Views.Axis>}:
  match count:
    case 0n: {==}
    case 1n+ +more:
      Equal.cong(List<&2,Views.Axis>,List<&2,Views.Axis>,tail => Views.Axis{Layout.extent_at(rest,index),Grid.product(List.drop(&2,U32,rest,1n+index))} <> tail,
        order_axes(Layout.every_index(more,2n+index),extent <> rest),order_axes(Layout.every_index(more,1n+index),rest),shifted_axes(more,1n+index,extent,rest))

# Every axis in order is the cells of the shape.
def every_cells(+shape: List<&2,U32>) -> {order_axes(Layout.every_index(List.length(&2,U32,shape),0n),shape) == Tasks.piece_cells(shape) : List<&2,Views.Axis>}:
  match shape:
    case []: {==}
    case +extent <> +rest:
      %Equal.sym(List<&2,Views.Axis>,Tasks.piece_cells(extent <> rest),Views.Axis{extent,Grid.product(rest)} <> Tasks.piece_cells(rest),GridPieces.dense_con(extent,rest)) :
        {order_axes(Layout.every_index(List.length(&2,U32,extent <> rest),0n),extent <> rest) == _ : List<&2,Views.Axis>}
      %Equal.sym(List<&2,U32>,List.drop(&2,U32,rest,0n),rest,dropped_none(~U32,rest)) :
        {Views.Axis{extent,Grid.product(_)} <> order_axes(Layout.every_index(List.length(&2,U32,rest),1n),extent <> rest) == Views.Axis{extent,Grid.product(rest)} <> Tasks.piece_cells(rest) : List<&2,Views.Axis>}
      Equal.cong(List<&2,Views.Axis>,List<&2,Views.Axis>,tail => Views.Axis{extent,Grid.product(rest)} <> tail,order_axes(Layout.every_index(List.length(&2,U32,rest),1n),extent <> rest),Tasks.piece_cells(rest),
        Equal.trans(List<&2,Views.Axis>,order_axes(Layout.every_index(List.length(&2,U32,rest),1n),extent <> rest),order_axes(Layout.every_index(List.length(&2,U32,rest),0n),rest),Tasks.piece_cells(rest),
          shifted_axes(List.length(&2,U32,rest),0n,extent,rest),every_cells(rest)))

# The last address of the cells of a nonempty shape within a capacity is one
# below its elements.
def cells_span(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+shape: List<&2,U32>,+bound: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(cap)) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool}) -> {1n+ViewBounds.last_span(Tasks.piece_cells(shape)) == ShapeLaws.elements(shape) : Nat}:
  match shape:
    case []: {==}
    case +extent <> +rest:
      +others = ShapeLaws.elements(rest)
      +span = ViewBounds.last_span(Tasks.piece_cells(rest))
      +extent_positive = RowMajor.left_factor_positive(U32.to_nat(extent),others,nonempty)
      +rest_positive = GridRanges.right_factor_positive(U32.to_nat(extent),others,nonempty)
      +rest_bound = Order.transitive(others,Nat.mul(U32.to_nat(extent),others),U32.to_nat(cap),NatDivision.le_multiple(U32.to_nat(extent),others,extent_positive),bound)
      +extent_bound = Order.transitive(U32.to_nat(extent),Nat.mul(U32.to_nat(extent),others),U32.to_nat(cap),RowMajor.product_covers(U32.to_nat(extent),others,rest_positive),bound)
      +minus = Arithmetic.subtract(extent,1,cap,cap_ok,Order.positive_one(U32.to_nat(extent),extent_positive),
        Order.transitive(Nat.sub(U32.to_nat(extent),1n),U32.to_nat(extent),U32.to_nat(cap),Order.sub_le(U32.to_nat(extent),1n),extent_bound))
      %Equal.sym(List<&2,Views.Axis>,Tasks.piece_cells(extent <> rest),Views.Axis{extent,Grid.product(rest)} <> Tasks.piece_cells(rest),GridPieces.dense_con(extent,rest)) :
        {1n+ViewBounds.last_span(_) == Nat.mul(U32.to_nat(extent),others) : Nat}
      %Equal.sym(Nat,U32.to_nat((extent - 1 : U32)),Nat.sub(U32.to_nat(extent),1n),minus) :
        {1n+Nat.add(Nat.mul(_,U32.to_nat(Grid.product(rest))),span) == Nat.mul(U32.to_nat(extent),others) : Nat}
      %Equal.sym(Nat,U32.to_nat(Grid.product(rest)),others,TiledPublic.product_exact(cap,cap_ok,rest,rest_bound)) :
        {1n+Nat.add(Nat.mul(Nat.sub(U32.to_nat(extent),1n),_),span) == Nat.mul(U32.to_nat(extent),others) : Nat}
      ContiguousBounds.head_span(U32.to_nat(extent),others,others,span,extent_positive,
        %Equal.sym(Bool,Nat.is_eq(others,others),True{},Order.equal_reflexive(others)) : {Nat.is_le(U32.to_nat(extent),1n) || _ == True{} : Bool}
        Boolean.or_true_right(Nat.is_le(U32.to_nat(extent),1n)),
        cells_span(cap,cap_ok,rest,rest_bound,rest_positive))

# The cells of a nonempty shape within a capacity in the order of its kept
# and reduced axes reach below its elements.
def line_spanned(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+shape: List<&2,U32>,+axes: List<&2,Nat>,
  +bound: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(cap)) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool}) ->
  {Nat.is_lt(ViewBounds.last_span(order_axes(List.append(&2,Nat,Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{}),Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{})),shape)),ShapeLaws.elements(shape)) == True{} : Bool}:
  +rank = List.length(&2,U32,shape)
  +kept = Layout.axes_where(List.length(&2,U32,shape),0n,axes,False{})
  +reduced = Layout.axes_where(List.length(&2,U32,shape),0n,axes,True{})
  +cells = ViewBounds.last_span(Tasks.piece_cells(shape))
  +spans = Equal.trans(Nat,ViewBounds.last_span(order_axes(List.append(&2,Nat,kept,reduced),shape)),Nat.add(ViewBounds.last_span(order_axes(kept,shape)),ViewBounds.last_span(order_axes(reduced,shape))),cells,appended_span(kept,reduced,shape),
    Equal.trans(Nat,Nat.add(ViewBounds.last_span(order_axes(kept,shape)),ViewBounds.last_span(order_axes(reduced,shape))),ViewBounds.last_span(order_axes(Layout.every_index(rank,0n),shape)),cells,partition_span(rank,0n,axes,shape),
      Equal.cong(List<&2,Views.Axis>,Nat,items => ViewBounds.last_span(items),order_axes(Layout.every_index(rank,0n),shape),Tasks.piece_cells(shape),every_cells(shape))))
  %Equal.sym(Nat,ViewBounds.last_span(order_axes(List.append(&2,Nat,kept,reduced),shape)),cells,spans) : {Nat.is_lt(_,ShapeLaws.elements(shape)) == True{} : Bool}
  Held.nat_lt_moved(cells,ShapeLaws.elements(shape),1n+cells,Equal.sym(Nat,1n+cells,ShapeLaws.elements(shape),cells_span(cap,cap_ok,shape,bound,nonempty)),
    Order.lt_succ(cells,cells,Order.reflexive(cells)))

# A view of the cells checked when moved: it fits the cells and counts its
# extents.
type Moving<-axes: List<&2,Views.Axis>,-offset: U32,-shape: List<&2,U32>> is Data:
  Moving{count: U32,fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(shape)) == True{} : Bool},
    counted: {Tensors.count_of(Views.axis_extents(axes)) == Some{count} : Maybe<&2,U32>}}

# The kept and the reduced axes of a counted shape in that order fit its cells
# and count their extents.
def moving(+limits: Machine.Limits,+shape: List<&2,U32>,+kept: List<&2,Nat>,+reduced: List<&2,Nat>,+counted: Global.Counted<shape>,
  +spanned: {Nat.is_lt(ViewBounds.last_span(order_axes(List.append(&2,Nat,kept,reduced),shape)),ShapeLaws.elements(shape)) == True{} : Bool},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +room: {Nat.is_le(Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced)))) == True{} : Bool}) ->
  Moving<order_axes(List.append(&2,Nat,kept,reduced),shape),0,shape>:
  +axes = order_axes(List.append(&2,Nat,kept,reduced),shape)
  +exact = Equal.trans(List<&2,U32>,Views.axis_extents(axes),Layout.extents_at(shape,List.append(&2,Nat,kept,reduced)),List.append(&2,U32,Layout.extents_at(shape,kept),Layout.extents_at(shape,reduced)),
    order_axes_extents(List.append(&2,Nat,kept,reduced),shape),appended_extents(shape,kept,reduced))
  +elements_exact = Equal.trans(Nat,ShapeLaws.elements(Views.axis_extents(axes)),ShapeLaws.elements(List.append(&2,U32,Layout.extents_at(shape,kept),Layout.extents_at(shape,reduced))),Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced))),
    Equal.cong(List<&2,U32>,Nat,items => ShapeLaws.elements(items),Views.axis_extents(axes),List.append(&2,U32,Layout.extents_at(shape,kept),Layout.extents_at(shape,reduced)),exact),
    ShapeLaws.elements_appended(Layout.extents_at(shape,kept),Layout.extents_at(shape,reduced)))
  +moved_counted = StorageCount.counted_within(limits,Views.axis_extents(axes),Held.nat_le_moved(ShapeLaws.elements(Views.axis_extents(axes)),Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced))),U32.to_nat(Machine.capacity(limits)),elements_exact,room),default_capacity)
  +nonzero = ViewMoved.none_zero(Views.axis_extents(axes),Held.nat_lt_moved(0n,ShapeLaws.elements(Views.axis_extents(axes)),Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced))),elements_exact,nonempty))
  +bound = {%Equal.sym(Nat,U32.to_nat(Tensors.count_or_zero(shape)),ShapeLaws.elements(shape),ViewMoved.counted_cells(shape,Global.count(shape,counted),Global.accepted(shape,counted))) :
      {Nat.is_lt(ViewBounds.last_span(axes),_) == True{} : Bool}
    spanned : {Nat.is_lt(Nat.add(U32.to_nat(0),ViewBounds.last_span(axes)),U32.to_nat(Tensors.count_or_zero(shape))) == True{} : Bool}}
  Moving{Global.count(Views.axis_extents(axes),moved_counted),ViewBounds.fits_span(axes,0,Tensors.count_or_zero(shape),nonzero,bound),Global.accepted(Views.axis_extents(axes),moved_counted)}

# The kept axes first and the reduced axes last through a view of the cells
# moved into place.
def moved_contract(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+limits: Machine.Limits,+operations: F32,+length: U32,+initial: Accumulator,+model: PiecesModel.Model<Element>,+shape: List<&2,U32>,+kept: List<&2,Nat>,+reduced: List<&2,Nat>,+fill: Element,+workers: U32,+axes: List<&2,Views.Axis>,+offset: U32,+facts: Moving<axes,offset,shape>,
  +contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers}),
  +viewed: {Layout.line_view(shape,List.append(&2,Nat,kept,reduced)) == Some{Views.View{axes,offset}} : Maybe<&2,Views.View>},
  +sized: {List.length(&2,Nat,List.append(&2,Nat,kept,reduced)) == List.length(&2,U32,shape) : Nat},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced)))) == True{} : Bool},+length_exact: {U32.to_nat(length) == ShapeLaws.elements(Layout.extents_at(shape,reduced)) : Nat},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced)))),4n),U32.to_nat(cap)) == True{} : Bool}) ->
  Global.Shaped(~Element,~Lines<Accumulator,Ordered<Logical>>,~line_value(~Element,~Accumulator,~operation,~finish,~Ordered<Logical>,~through(~Element,~Logical,~original)),Lines{initial,Ordered{logical,Some{Views.View{axes,offset}}},ShapeLaws.elements(Layout.extents_at(shape,reduced))},Layout.extents_at(shape,kept),Layout.reduce_arranged(~Element,~Accumulator,~operation,~finish,operations,length,initial,List.length(&2,Nat,kept),Tensors.moved(~Element,"reduce",Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers},axes,offset,[]))):
  Moving{+count,+fits,+counted} = facts
  +order = List.append(&2,Nat,kept,reduced)
  +line_exact = {%viewed : {view_extents(_,Layout.extents_at(shape,order)) == Layout.extents_at(shape,order) : List<&2,U32>} line_extents(shape,order) :
    {Views.axis_extents(axes) == Layout.extents_at(shape,order) : List<&2,U32>}}
  +exact = Equal.trans(List<&2,U32>,Views.axis_extents(axes),Layout.extents_at(shape,order),List.append(&2,U32,Layout.extents_at(shape,kept),Layout.extents_at(shape,reduced)),line_exact,appended_extents(shape,kept,reduced))
  +ranked = Equal.trans(Nat,List.length(&2,U32,Views.axis_extents(axes)),List.length(&2,U32,Layout.extents_at(shape,order)),List.length(&2,U32,shape),
    Equal.cong(List<&2,U32>,Nat,items => List.length(&2,U32,items),Views.axis_extents(axes),Layout.extents_at(shape,order),line_exact),
    Equal.trans(Nat,List.length(&2,U32,Layout.extents_at(shape,order)),List.length(&2,Nat,order),List.length(&2,U32,shape),extents_length(shape,order),sized))
  +elements_exact = Equal.trans(Nat,ShapeLaws.elements(Views.axis_extents(axes)),ShapeLaws.elements(List.append(&2,U32,Layout.extents_at(shape,kept),Layout.extents_at(shape,reduced))),Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced))),
    Equal.cong(List<&2,U32>,Nat,items => ShapeLaws.elements(items),Views.axis_extents(axes),List.append(&2,U32,Layout.extents_at(shape,kept),Layout.extents_at(shape,reduced)),exact),ShapeLaws.elements_appended(Layout.extents_at(shape,kept),Layout.extents_at(shape,reduced)))
  +positive = Held.nat_lt_moved(0n,ShapeLaws.elements(Views.axis_extents(axes)),Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced))),elements_exact,nonempty)
  +count_word = Count.count_exact(ShapeLaws.elements(Views.axis_extents(axes)),count,Count.count_of(Views.axis_extents(axes),count,counted))
  +nonzero = U32Division.nonzero(count,Held.nat_lt_moved(0n,U32.to_nat(count),ShapeLaws.elements(Views.axis_extents(axes)),Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(count),count_word),positive))
  +target_rank = Held.nat_le_moved(List.length(&2,U32,Views.axis_extents(axes)),List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits)),ranked,rank_room)
  +viewing = {ViewGather.Viewing{logical,axes,offset,[]} : ViewGather.Viewing<Logical>}
  +moved = {Ordered{logical,Some{Views.View{axes,offset}}} : Ordered<Logical>}
  +shaped = Global.restated(~Element,~ViewGather.Viewing<Logical>,~Ordered<Logical>,~ViewGather.viewed(~Element,~Logical,~original),~through(~Element,~Logical,~original),
    viewing,moved,index => {==},Views.axis_extents(axes),Tensors.moved(~Element,"reduce",Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers},axes,offset,[]),
    ViewMoved.view_moved(~Element,~Logical,~original,logical,limits,"reduce",model,shape,axes,offset,[],fill,workers,contract,default_capacity,rank_room,target_rank,count,counted,nonzero,fits))
  +dropped = {
    %Equal.sym(List<&2,U32>,Views.axis_extents(axes),List.append(&2,U32,Layout.extents_at(shape,kept),Layout.extents_at(shape,reduced)),exact) : {List.drop(&2,U32,_,List.length(&2,Nat,kept)) == Layout.extents_at(shape,reduced) : List<&2,U32>}
    %extents_length(shape,kept) : {List.drop(&2,U32,List.append(&2,U32,Layout.extents_at(shape,kept),Layout.extents_at(shape,reduced)),_) == Layout.extents_at(shape,reduced) : List<&2,U32>}
    drop_prefix(~U32,Layout.extents_at(shape,kept),Layout.extents_at(shape,reduced)) : {List.drop(&2,U32,Views.axis_extents(axes),List.length(&2,Nat,kept)) == Layout.extents_at(shape,reduced) : List<&2,U32>}}
  +lengthed = Equal.cong(List<&2,U32>,Nat,items => ShapeLaws.elements(items),List.drop(&2,U32,Views.axis_extents(axes),List.length(&2,Nat,kept)),Layout.extents_at(shape,reduced),dropped)
  +length_word = Equal.trans(Nat,U32.to_nat(length),ShapeLaws.elements(Layout.extents_at(shape,reduced)),ShapeLaws.elements(List.drop(&2,U32,Views.axis_extents(axes),List.length(&2,Nat,kept))),length_exact,
    Equal.sym(Nat,ShapeLaws.elements(List.drop(&2,U32,Views.axis_extents(axes),List.length(&2,Nat,kept))),ShapeLaws.elements(Layout.extents_at(shape,reduced)),lengthed))
  +widths = Equal.trans(Nat,List.length(&2,U32,Views.axis_extents(axes)),List.length(&2,U32,shape),Nat.add(List.length(&2,Nat,kept),List.length(&2,Nat,reduced)),ranked,
    Equal.trans(Nat,List.length(&2,U32,shape),List.length(&2,Nat,order),Nat.add(List.length(&2,Nat,kept),List.length(&2,Nat,reduced)),Equal.sym(Nat,List.length(&2,Nat,order),List.length(&2,U32,shape),sized),
      ListReadback.append_length(~Nat,kept,reduced)))
  +kept_room = Held.nat_le_right_moved(List.length(&2,Nat,kept),List.length(&2,U32,Views.axis_extents(axes)),Nat.add(List.length(&2,Nat,kept),List.length(&2,Nat,reduced)),widths,Order.le_add(List.length(&2,Nat,kept),List.length(&2,Nat,reduced)))
  +taken = {
    %Equal.sym(List<&2,U32>,Views.axis_extents(axes),List.append(&2,U32,Layout.extents_at(shape,kept),Layout.extents_at(shape,reduced)),exact) : {List.take(&2,U32,_,List.length(&2,Nat,kept)) == Layout.extents_at(shape,kept) : List<&2,U32>}
    %extents_length(shape,kept) : {List.take(&2,U32,List.append(&2,U32,Layout.extents_at(shape,kept),Layout.extents_at(shape,reduced)),_) == Layout.extents_at(shape,kept) : List<&2,U32>}
    take_prefix(~U32,Layout.extents_at(shape,kept),Layout.extents_at(shape,reduced)) : {List.take(&2,U32,Views.axis_extents(axes),List.length(&2,Nat,kept)) == Layout.extents_at(shape,kept) : List<&2,U32>}}
  +kept_outputs = Equal.trans(List<&2,U32>,Layout.extents_at(Views.axis_extents(axes),Layout.every_index(List.length(&2,Nat,kept),0n)),List.take(&2,U32,List.drop(&2,U32,Views.axis_extents(axes),0n),List.length(&2,Nat,kept)),Layout.extents_at(shape,kept),every_take(Views.axis_extents(axes),List.length(&2,Nat,kept),0n,kept_room),
    Equal.trans(List<&2,U32>,List.take(&2,U32,List.drop(&2,U32,Views.axis_extents(axes),0n),List.length(&2,Nat,kept)),List.take(&2,U32,Views.axis_extents(axes),List.length(&2,Nat,kept)),Layout.extents_at(shape,kept),
      Equal.cong(List<&2,U32>,List<&2,U32>,items => List.take(&2,U32,items,List.length(&2,Nat,kept)),List.drop(&2,U32,Views.axis_extents(axes),0n),Views.axis_extents(axes),dropped_none(~U32,Views.axis_extents(axes))),taken))
  %lengthed : Global.Shaped(~Element,~Lines<Accumulator,Ordered<Logical>>,~line_value(~Element,~Accumulator,~operation,~finish,~Ordered<Logical>,~through(~Element,~Logical,~original)),Lines{initial,moved,_},Layout.extents_at(shape,kept),Layout.reduce_arranged(~Element,~Accumulator,~operation,~finish,operations,length,initial,List.length(&2,Nat,kept),Tensors.moved(~Element,"reduce",Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers},axes,offset,[])))
  %kept_outputs : Global.Shaped(~Element,~Lines<Accumulator,Ordered<Logical>>,~line_value(~Element,~Accumulator,~operation,~finish,~Ordered<Logical>,~through(~Element,~Logical,~original)),Lines{initial,moved,ShapeLaws.elements(List.drop(&2,U32,Views.axis_extents(axes),List.length(&2,Nat,kept)))},_,Layout.reduce_arranged(~Element,~Accumulator,~operation,~finish,operations,length,initial,List.length(&2,Nat,kept),Tensors.moved(~Element,"reduce",Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers},axes,offset,[])))
  arranged_contract(~Element,~Accumulator,~operation,~finish,~Ordered<Logical>,~through(~Element,~Logical,~original),moved,limits,operations,length,initial,List.length(&2,Nat,kept),Tensors.moved(~Element,"reduce",Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers},axes,offset,[]),Views.axis_extents(axes),shaped,
    Held.nat_le_moved(ShapeLaws.elements(Views.axis_extents(axes)),Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced))),U32.to_nat(Machine.capacity(limits)),elements_exact,room),positive,length_word,cap,cap_ok,
    Held.nat_le_moved(Nat.add(Nat.double(ShapeLaws.elements(Views.axis_extents(axes))),4n),Nat.add(Nat.double(Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced)))),4n),U32.to_nat(cap),
      Equal.cong(Nat,Nat,number => Nat.add(Nat.double(number),4n),ShapeLaws.elements(Views.axis_extents(axes)),Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced))),elements_exact),count_room),
    kept_room,default_capacity)

# The kept and the reduced axes in this order: the view of the cells they
# pick and the last address of that view below the cells.
type Lined<-shape: List<&2,U32>,-order: List<&2,Nat>> is Data:
  Lined{picked: {Layout.line_view(shape,order) == Some{Views.View{order_axes(order,shape),0}} : Maybe<&2,Views.View>},
    spanned: {Nat.is_lt(ViewBounds.last_span(order_axes(order,shape)),ShapeLaws.elements(shape)) == True{} : Bool}}

# The reduction of the kept axes first and the reduced axes last is Ready and
# holds, at the row-major index of the kept axes, the finished fold of the
# line of the reduced axes read through the view in that order.
def ordered_contract(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,~Logical: Data,~original: Logical -> U32 -> Element,trailing: Bool,+logical: Logical,+limits: Machine.Limits,+operations: F32,+length: U32,+initial: Accumulator,+model: PiecesModel.Model<Element>,+shape: List<&2,U32>,+kept: List<&2,Nat>,+reduced: List<&2,Nat>,+fill: Element,+workers: U32,
  +contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers}),
  +ordered: {Layout.in_order(List.append(&2,Nat,kept,reduced),0n) == trailing : Bool},+lined: Lined<shape,List.append(&2,Nat,kept,reduced)>,
  +sized: {List.length(&2,Nat,List.append(&2,Nat,kept,reduced)) == List.length(&2,U32,shape) : Nat},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced)))) == True{} : Bool},+length_exact: {U32.to_nat(length) == ShapeLaws.elements(Layout.extents_at(shape,reduced)) : Nat},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced)))),4n),U32.to_nat(cap)) == True{} : Bool}) ->
  Global.Shaped(~Element,~Lines<Accumulator,Ordered<Logical>>,~line_value(~Element,~Accumulator,~operation,~finish,~Ordered<Logical>,~through(~Element,~Logical,~original)),Lines{initial,Ordered{logical,Layout.line_view(shape,List.append(&2,Nat,kept,reduced))},ShapeLaws.elements(Layout.extents_at(shape,reduced))},Layout.extents_at(shape,kept),Layout.reduce_moved(~Element,~Accumulator,~operation,~finish,operations,length,initial,trailing,Layout.line_view(shape,List.append(&2,Nat,kept,reduced)),PiecesModel.pack(~Element,model),shape,kept,fill,workers)):
  match trailing:
    case True{}:
      +cells_view = {Some{Views.View{Tasks.piece_cells(shape),0}} : Maybe<&2,Views.View>}
      %Equal.sym(Maybe<&2,Views.View>,Layout.line_view(shape,List.append(&2,Nat,kept,reduced)),cells_view,line_identity(shape,List.append(&2,Nat,kept,reduced),ordered,sized)) :
        Global.Shaped(~Element,~Lines<Accumulator,Ordered<Logical>>,~line_value(~Element,~Accumulator,~operation,~finish,~Ordered<Logical>,~through(~Element,~Logical,~original)),Lines{initial,Ordered{logical,_},ShapeLaws.elements(Layout.extents_at(shape,reduced))},Layout.extents_at(shape,kept),Layout.reduce_moved(~Element,~Accumulator,~operation,~finish,operations,length,initial,True{},Layout.line_view(shape,List.append(&2,Nat,kept,reduced)),PiecesModel.pack(~Element,model),shape,kept,fill,workers))
      identity_contract(~Element,~Accumulator,~operation,~finish,~Logical,~original,logical,limits,operations,length,initial,model,shape,kept,reduced,fill,workers,contract,ordered,sized,room,nonempty,length_exact,cap,cap_ok,count_room,default_capacity)
    case False{}:
      Lined{+picked,+spanned} = lined
      +moved_view = {Some{Views.View{order_axes(List.append(&2,Nat,kept,reduced),shape),0}} : Maybe<&2,Views.View>}
      %Equal.sym(Maybe<&2,Views.View>,Layout.line_view(shape,List.append(&2,Nat,kept,reduced)),moved_view,picked) :
        Global.Shaped(~Element,~Lines<Accumulator,Ordered<Logical>>,~line_value(~Element,~Accumulator,~operation,~finish,~Ordered<Logical>,~through(~Element,~Logical,~original)),Lines{initial,Ordered{logical,_},ShapeLaws.elements(Layout.extents_at(shape,reduced))},Layout.extents_at(shape,kept),Layout.reduce_moved(~Element,~Accumulator,~operation,~finish,operations,length,initial,False{},_,PiecesModel.pack(~Element,model),shape,kept,fill,workers))
      moved_contract(~Element,~Accumulator,~operation,~finish,~Logical,~original,logical,limits,operations,length,initial,model,shape,kept,reduced,fill,workers,order_axes(List.append(&2,Nat,kept,reduced),shape),0,
        moving(limits,shape,kept,reduced,Global.contract_counted(~Element,~Logical,~original,logical,PiecesModel.pack(~Element,model),shape,fill,workers,contract),spanned,default_capacity,room,nonempty),
        contract,picked,sized,default_capacity,rank_room,room,nonempty,length_exact,cap,cap_ok,count_room)

# ---- Reductions over listed axes ----

# A reduction over the reduced axes holds, at the row-major index of the kept
# axes, the finish of the fold of the values of the reduced axes there in
# row-major order.
def Listed(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+initial: Accumulator,+shape: List<&2,U32>,+kept: List<&2,Nat>,+reduced: List<&2,Nat>,-tensor: Tensors.Tensor<Element>) -> Data:
  Global.Shaped(~Element,~Lines<Accumulator,Ordered<Logical>>,~line_value(~Element,~Accumulator,~operation,~finish,~Ordered<Logical>,~through(~Element,~Logical,~original)),Lines{initial,Ordered{logical,Layout.line_view(shape,List.append(&2,Nat,kept,reduced))},ShapeLaws.elements(Layout.extents_at(shape,reduced))},Layout.extents_at(shape,kept),tensor)

def sized_contract(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,~Logical: Data,~original: Logical -> U32 -> Element,empty: Bool,unread: Bool,+count: U32,+length: U32,+logical: Logical,+limits: Machine.Limits,+operations: F32,+initial: Accumulator,+model: PiecesModel.Model<Element>,+shape: List<&2,U32>,+kept: List<&2,Nat>,+reduced: List<&2,Nat>,+fill: Element,+workers: U32,+contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers}),
  +emptied: {U32.is_zero(count) == empty : Bool},+unreads: {U32.is_zero(length) == unread : Bool},
  +counts: {Tensors.count_of(Layout.extents_at(shape,kept)) == Some{count} : Maybe<&2,U32>},+lengths: {Tensors.count_of(Layout.extents_at(shape,reduced)) == Some{length} : Maybe<&2,U32>},
  +lined: Lined<shape,List.append(&2,Nat,kept,reduced)>,+sized: {List.length(&2,Nat,List.append(&2,Nat,kept,reduced)) == List.length(&2,U32,shape) : Nat},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced)))) == True{} : Bool},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced)))),4n),U32.to_nat(cap)) == True{} : Bool}) ->
  Listed(~Element,~Accumulator,~operation,~finish,~Logical,~original,logical,initial,shape,kept,reduced,Layout.reduce_sized(~Element,~Accumulator,~operation,~finish,operations,empty,unread,count,length,initial,PiecesModel.pack(~Element,model),shape,kept,reduced,fill,workers)):
  match empty unread:
    case True{} unread:
      +positive = Held.nat_lt_moved(0n,U32.to_nat(count),ShapeLaws.elements(Layout.extents_at(shape,kept)),
        Equal.sym(Nat,ShapeLaws.elements(Layout.extents_at(shape,kept)),U32.to_nat(count),Count.count_exact(ShapeLaws.elements(Layout.extents_at(shape,kept)),count,Count.count_of(Layout.extents_at(shape,kept),count,counts))),
        RowMajor.left_factor_positive(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced)),nonempty))
      Empty.absurd(Listed(~Element,~Accumulator,~operation,~finish,~Logical,~original,logical,initial,shape,kept,reduced,Layout.reduce_sized(~Element,~Accumulator,~operation,~finish,operations,True{},unread,count,length,initial,PiecesModel.pack(~Element,model),shape,kept,reduced,fill,workers)),
        WordEncoding.false_true(Equal.trans(Bool,False{},U32.is_zero(count),True{},Equal.sym(Bool,U32.is_zero(count),False{},U32Division.nonzero(count,positive)),emptied)))
    case False{} True{}:
      +positive = Held.nat_lt_moved(0n,U32.to_nat(length),ShapeLaws.elements(Layout.extents_at(shape,reduced)),
        Equal.sym(Nat,ShapeLaws.elements(Layout.extents_at(shape,reduced)),U32.to_nat(length),Count.count_exact(ShapeLaws.elements(Layout.extents_at(shape,reduced)),length,Count.count_of(Layout.extents_at(shape,reduced),length,lengths))),
        GridRanges.right_factor_positive(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced)),nonempty))
      Empty.absurd(Listed(~Element,~Accumulator,~operation,~finish,~Logical,~original,logical,initial,shape,kept,reduced,Layout.reduce_sized(~Element,~Accumulator,~operation,~finish,operations,False{},True{},count,length,initial,PiecesModel.pack(~Element,model),shape,kept,reduced,fill,workers)),
        WordEncoding.false_true(Equal.trans(Bool,False{},U32.is_zero(length),True{},Equal.sym(Bool,U32.is_zero(length),False{},U32Division.nonzero(length,positive)),unreads)))
    case False{} False{}:
      ordered_contract(~Element,~Accumulator,~operation,~finish,~Logical,~original,Layout.in_order(List.append(&2,Nat,kept,reduced),0n),logical,limits,operations,length,initial,model,shape,kept,reduced,fill,workers,
        contract,{==},lined,sized,default_capacity,rank_room,room,nonempty,
        Equal.sym(Nat,ShapeLaws.elements(Layout.extents_at(shape,reduced)),U32.to_nat(length),Count.count_exact(ShapeLaws.elements(Layout.extents_at(shape,reduced)),length,Count.count_of(Layout.extents_at(shape,reduced),length,lengths))),
        cap,cap_ok,count_room)

def counted_contract(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,~Logical: Data,~original: Logical -> U32 -> Element,+name: String,+logical: Logical,+limits: Machine.Limits,+operations: F32,+initial: Accumulator,+model: PiecesModel.Model<Element>,+shape: List<&2,U32>,+kept: List<&2,Nat>,+reduced: List<&2,Nat>,+fill: Element,+workers: U32,+contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers}),
  +lined: Lined<shape,List.append(&2,Nat,kept,reduced)>,+sized: {List.length(&2,Nat,List.append(&2,Nat,kept,reduced)) == List.length(&2,U32,shape) : Nat},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced)))) == True{} : Bool},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced)))),4n),U32.to_nat(cap)) == True{} : Bool}) ->
  Listed(~Element,~Accumulator,~operation,~finish,~Logical,~original,logical,initial,shape,kept,reduced,Layout.reduce_counted(~Element,~Accumulator,~operation,~finish,name,operations,Tensors.count_of(Layout.extents_at(shape,kept)),Tensors.count_of(Layout.extents_at(shape,reduced)),initial,PiecesModel.pack(~Element,model),shape,kept,reduced,fill,workers)):
  +kept_counted = StorageCount.counted_within(limits,Layout.extents_at(shape,kept),
    Order.transitive(ShapeLaws.elements(Layout.extents_at(shape,kept)),Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced))),U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced)),GridRanges.right_factor_positive(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced)),nonempty)),room),default_capacity)
  +reduced_counted = StorageCount.counted_within(limits,Layout.extents_at(shape,reduced),
    Order.transitive(ShapeLaws.elements(Layout.extents_at(shape,reduced)),Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced))),U32.to_nat(Machine.capacity(limits)),NatDivision.le_multiple(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced)),RowMajor.left_factor_positive(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced)),nonempty)),room),default_capacity)
  +count = Global.count(Layout.extents_at(shape,kept),kept_counted)
  +length = Global.count(Layout.extents_at(shape,reduced),reduced_counted)
  +counts = Global.accepted(Layout.extents_at(shape,kept),kept_counted)
  +lengths = Global.accepted(Layout.extents_at(shape,reduced),reduced_counted)
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(Layout.extents_at(shape,kept)),Some{count},counts) :
    Listed(~Element,~Accumulator,~operation,~finish,~Logical,~original,logical,initial,shape,kept,reduced,Layout.reduce_counted(~Element,~Accumulator,~operation,~finish,name,operations,_,Tensors.count_of(Layout.extents_at(shape,reduced)),initial,PiecesModel.pack(~Element,model),shape,kept,reduced,fill,workers))
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(Layout.extents_at(shape,reduced)),Some{length},lengths) :
    Listed(~Element,~Accumulator,~operation,~finish,~Logical,~original,logical,initial,shape,kept,reduced,Layout.reduce_counted(~Element,~Accumulator,~operation,~finish,name,operations,Some{count},_,initial,PiecesModel.pack(~Element,model),shape,kept,reduced,fill,workers))
  sized_contract(~Element,~Accumulator,~operation,~finish,~Logical,~original,U32.is_zero(count),U32.is_zero(length),count,length,logical,limits,operations,initial,model,shape,kept,reduced,fill,workers,contract,{==},{==},counts,lengths,
    lined,sized,default_capacity,rank_room,room,nonempty,cap,cap_ok,count_room)

# reduce_axes of a tensor of the contract of a reader over valid axes is
# Ready and holds at the row-major index of the kept axes the finish of the
# fold of the values of the listed axes there in row-major order.
def reduce_axes(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+limits: Machine.Limits,+name: String,+operations: F32,+initial: Accumulator,+axes: List<&2,U32>,
  +model: PiecesModel.Model<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,+contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers}),
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap)) == True{} : Bool},
  +valid: {Views.unique_axes(Layout.naturals(axes)) && Layout.all_below(axes,List.length(&2,U32,shape)) == True{} : Bool}) ->
  Listed(~Element,~Accumulator,~operation,~finish,~Logical,~original,logical,initial,shape,kept_axes(shape,axes),reduced_axes(shape,axes),Layout.reduce_axes(~Element,~Accumulator,~operation,~finish,name,operations,initial,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers},axes)):
  +rank = List.length(&2,U32,shape)
  +listed = Layout.naturals(axes)
  +kept = kept_axes(shape,axes)
  +reduced = reduced_axes(shape,axes)
  +sized = Equal.trans(Nat,List.length(&2,Nat,List.append(&2,Nat,kept,reduced)),Nat.add(List.length(&2,Nat,kept),List.length(&2,Nat,reduced)),rank,ListReadback.append_length(~Nat,kept,reduced),
    partition_length(rank,0n,listed))
  +product = Equal.trans(Nat,Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced))),ShapeLaws.elements(List.drop(&2,U32,shape,0n)),ShapeLaws.elements(shape),partition_elements(shape,rank,0n,listed,{==}),
    Equal.cong(List<&2,U32>,Nat,items => ShapeLaws.elements(items),List.drop(&2,U32,shape,0n),shape,dropped_none(~U32,shape)))
  +shape_room = StorageCount.counted_room(limits,shape,Global.contract_counted(~Element,~Logical,~original,logical,PiecesModel.pack(~Element,model),shape,fill,workers,contract),default_capacity)
  +room = Held.nat_le_moved(Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced))),ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits)),product,shape_room)
  +positive = Held.nat_lt_moved(0n,Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced))),ShapeLaws.elements(shape),product,nonempty)
  +lines_room = Held.nat_le_moved(Nat.add(Nat.double(Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced)))),4n),Nat.add(Nat.double(ShapeLaws.elements(shape)),4n),U32.to_nat(cap),
    Equal.cong(Nat,Nat,number => Nat.add(Nat.double(number),4n),Nat.mul(ShapeLaws.elements(Layout.extents_at(shape,kept)),ShapeLaws.elements(Layout.extents_at(shape,reduced))),ShapeLaws.elements(shape),product),count_room)
  +lined = {Lined{line_picked(shape,listed),line_spanned(Machine.capacity(limits),LimitBounds.capacity_fits(limits),shape,listed,shape_room,nonempty)} : Lined<shape,List.append(&2,Nat,kept,reduced)>}
  %Equal.sym(Bool,Views.unique_axes(listed) && Layout.all_below(axes,rank),True{},valid) :
    Listed(~Element,~Accumulator,~operation,~finish,~Logical,~original,logical,initial,shape,kept,reduced,Layout.reduce_listed(~Element,~Accumulator,~operation,~finish,name,operations,initial,Tensors.Tensor{PiecesModel.pack(~Element,model),shape,fill,Tensors.Usable{},workers},kept,reduced,_))
  counted_contract(~Element,~Accumulator,~operation,~finish,~Logical,~original,name,logical,limits,operations,initial,model,shape,kept,reduced,fill,workers,contract,lined,sized,default_capacity,rank_room,room,positive,cap,cap_ok,lines_room)
