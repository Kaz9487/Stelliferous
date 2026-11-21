# The shared-group task tree places the cells of its actual rectangles.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_tile.bend as Tile
import ../machine_limits.bend as Machine
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_move.bend as Move
import ../tensor_pieces.bend as Pieces
import ../tensor_point_products.bend as Products
import ../tensor_tasks.bend as Tasks
import ../traversal_partition.bend as Partition
import ./kernel_gemm_output_groups.bend as Groups
import ./kernel_gemm_point_jobs.bend as Jobs
import ./machine_limit_bounds.bend as Limits
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./nat_to_u32_bounds.bend as Bounds
import ./storage_buffer_model.bend as Buffer
import ./tensor_grid_pieces.bend as GridLaws
import ./tensor_grid_ranges.bend as GridRanges
import ./tensor_halo_windows.bend as Windows
import ./tensor_move_readback.bend as Held
import ./tensor_pieces_grouped_with.bend as Grouped
import ./tensor_pieces_model.bend as Model
import ./tensor_point_product_fragments.bend as Fragments
import ./tensor_point_product_placement.bend as Placed
import ./tensor_point_product_tasks.bend as Shared
import ./tensor_point_products.bend as Point
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_task_runs.bend as Runs
import ./traversal_partition_refinement.bend as PartitionLaws

def output_cells(+rows: U32,+channels: U32) -> Nat:
  Nat.mul(U32.to_nat(Gemm.blocks(channels)),U32.to_nat(rows))

type Layout<-limits: Machine.Limits,-rows: U32,-channels: U32,-grid: Grid.Grid> is Data:
  Layout{grid_facts: GridLaws.GridFacts<limits,grid>,
    channel_room: {Nat.is_le(U32.to_nat(channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
    rows_room: {Nat.is_le(U32.to_nat(rows),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    cells: {GridLaws.grid_cells(grid) == output_cells(rows,channels) : Nat},
    room: {Nat.is_le(output_cells(rows,channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    run: Runs.Run(Pieces.ranges_of_grid(grid),U32.to_nat(rows),output_cells(rows,channels))}

def leaf_stage(+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,+first: U32,+count: U32,
  +holding: Held.Holding<Tile.Tile8>,+resource: Point.SharedResource) -> Buffer.Content<Tile.Tile16>:
  Point.SharedResource{+input,+bias} = resource
  +rectangle = Point.shared_geometry(Products.Shape{rows,inner,channels,grid},first,count)
  +base = (first / rows * inner : U32)
  Point.computed_staged(~Jobs.run,rectangle,0,Buffer.content(~Tile.Tile8,input),
    Point.shared_held_pairs(rectangle,base,holding),Buffer.content(~Tile.Tile8,Point.shared_bias_at(resource,rows,first,count)))

def layout_inside(+limits: Machine.Limits,+rows: U32,+channels: U32,+grid: Grid.Grid,
  +layout: Layout<limits,rows,channels,grid>,+first: U32,+count: U32,
  +fits: Runs.Fits<U32.to_nat(rows),U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),output_cells(rows,channels)>) ->
  {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Layout{+grid_facts,+channel_room,+rows_positive,+rows_room,+cells,+room,+run} = layout
  Runs.Fits{+low,+high,+first_exact,+last_exact,+less,+inside} = fits
  Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(count)),output_cells(rows,channels),U32.to_nat(Machine.capacity(limits)),inside,room)

def leaf_space(+limits: Machine.Limits,+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,+first: U32,+count: U32,
  +rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +rows_room: {Nat.is_le(U32.to_nat(rows),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +length: {U32.to_nat(count) == Fragments.rectangle_cells(Point.shared_geometry(Products.Shape{rows,inner,channels,grid},first,count)) : Nat},
  +clip: Shared.Clipped<limits,rows,channels,first,count>) -> Fragments.RectangleReadRoom<limits,Point.shared_geometry(Products.Shape{rows,inner,channels,grid},first,count)>:
  Shared.Clipped{+exact,+extent_room,+extent_positive,+groups_exact} = clip
  +extent = Shared.clipped_extent(rows,channels,first,count)
  +groups = Groups.groups_fit(limits,extent,extent_room)
  Fragments.RectangleReadRoom{rows_positive,rows_room,Groups.groups_positive(limits,extent,extent_room,extent_positive),groups,Groups.pairs_fewer(limits,extent,groups),
    %length : {Nat.is_le(Nat.add(U32.to_nat(first),_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} room}

def LeafModel(+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,+first: U32,+count: U32,
  +holding: Held.Holding<Tile.Tile8>,+resource: Point.SharedResource,
  fact: Shared.SharedLeaf(Products.Shape{rows,inner,channels,grid},holding,first,count,resource)) -> Data:
  Grouped.TaskFact{+result,+rest,+exact,+regions} = fact
  {Model.content(~Tile.Tile8,Placed.task_pieces(result)) ==
    Point.rectangle_model(Point.shared_geometry(Products.Shape{rows,inner,channels,grid},first,count),leaf_stage(rows,inner,channels,grid,first,count,holding,resource)) :
    Model.Content<Tile.Tile8>}

def leaf_contents(+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,+first: U32,+count: U32,
  +holding: Held.Holding<Tile.Tile8>,+resource: Point.SharedResource,
  +fact: Shared.SharedLeaf(Products.Shape{rows,inner,channels,grid},holding,first,count,resource)) ->
  LeafModel(rows,inner,channels,grid,first,count,holding,resource,fact):
  Point.SharedResource{+input,+bias} = resource
  Grouped.TaskFact{+result,+rest,+exact,+regions} = fact
  +work = {Products.Shape{rows,inner,channels,grid} : Products.Shape}
  +rectangle = Point.shared_geometry(work,first,count)
  +stage = leaf_stage(rows,inner,channels,grid,first,count,holding,resource)
  -native = Products.shared_leaf(work,first,count,Held.pack(~Tile.Tile8,holding),Point.pack_resource(resource))
  -packed = {(Shared.pack_task(result),Point.pack_resource(rest)) : Point.TaskOutput() & Point.Resources()}
  +content = Point.task_contents(Shared.pack_task(result))
  Equal.trans(Model.Content<Tile.Tile8>,Model.content(~Tile.Tile8,Placed.task_pieces(result)),content,Point.rectangle_model(rectangle,stage),
    Equal.sym(Model.Content<Tile.Tile8>,content,Model.content(~Tile.Tile8,Placed.task_pieces(result)),Placed.task_contents(result)),
    %exact : {Point.result_contents(~Point.Resources(),_) == Point.rectangle_model(rectangle,stage) : Model.Content<Tile.Tile8>}
    Point.shared_leaf_contents(work,first,count,holding,resource))

def leaf_plan(+limits: Machine.Limits,+rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,+first: U32,+count: U32,
  +holding: Held.Holding<Tile.Tile8>,+resource: Point.SharedResource,
  +layout: Layout<limits,rows,channels,grid>,
  +fits: Runs.Fits<U32.to_nat(rows),U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),output_cells(rows,channels)>,
  +result: Shared.TaskModel(),
  +holds: {Model.content(~Tile.Tile8,Placed.task_pieces(result)) ==
    Point.rectangle_model(Point.shared_geometry(Products.Shape{rows,inner,channels,grid},first,count),leaf_stage(rows,inner,channels,grid,first,count,holding,resource)) :
    Model.Content<Tile.Tile8>}) ->
  Placed.ObservedPlan<limits,Pieces.ranges_of_grid(grid),U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),result>:
  Layout{+grid_facts,+channel_room,+rows_positive,+rows_room,+cells,+room,+run} = layout
  +work = {Products.Shape{rows,inner,channels,grid} : Products.Shape}
  +rectangle = Point.shared_geometry(work,first,count)
  +stage = leaf_stage(rows,inner,channels,grid,first,count,holding,resource)
  +interval = Shared.group_interval(rows,channels,Machine.half_word(limits),rows_positive,Limits.double_bound(limits,U32.to_nat(rows),rows_room),
    Limits.half_word_fits(limits),first,count,fits)
  +clip = Shared.clipped_facts(limits,rows,channels,first,count,channel_room,interval)
  +length = Shared.clipped_cells(limits,rows,channels,first,count,interval,clip)
  +shape = leaf_space(limits,rows,inner,channels,grid,first,count,rows_positive,rows_room,layout_inside(limits,rows,channels,grid,layout,first,count,fits),length,clip)
  +fragments = Shared.shared_leaf_laid(limits,rows,inner,channels,grid,grid_facts,channel_room,rows_positive,rows_room,cells,room,run,first,count,fits)
  Placed.ObservedPlan{Placed.RectanglePlan{rectangle,stage,result,first,count,{==},length,{==},{==},shape,
    holds,
    %length : Shared.Laid(Pieces.ranges_of_grid(grid),Point.clipped_rectangle(rectangle),U32.to_nat(first),Nat.add(U32.to_nat(first),_))
    fragments},{==}}

def run_order(+ranges: Pieces.Ranges,+unit: Nat,+bound: Nat,+fits: Runs.Fits<unit,Runs.range_low(ranges),Runs.range_high(ranges),bound>) ->
  {Nat.is_le(Runs.range_low(ranges),Runs.range_high(ranges)) == True{} : Bool}:
  Runs.Fits{+low,+high,+start,+end,+positive,+inside} = fits
  Bounds.strict_le(Runs.range_low(ranges),Runs.range_high(ranges),positive)

def joined_plan(+limits: Machine.Limits,+layout: Pieces.Ranges,+left: Pieces.Ranges,+right: Pieces.Ranges,+left_tasks: Shared.TaskModel(),+right_tasks: Shared.TaskModel(),
  +glue: {Runs.range_high(left) == Runs.range_low(right) : Nat},
  +left_order: {Nat.is_le(Runs.range_low(left),Runs.range_high(left)) == True{} : Bool},
  +right_order: {Nat.is_le(Runs.range_low(right),Runs.range_high(right)) == True{} : Bool},
  +left_seen: Placed.ObservedPlan<limits,layout,Runs.range_low(left),Runs.range_high(left),left_tasks>,
  +right_seen: Placed.ObservedPlan<limits,layout,Runs.range_low(right),Runs.range_high(right),right_tasks>) ->
  Placed.ObservedPlan<limits,layout,Runs.range_low(left),Runs.range_high(right),Partition.Fork{left_tasks,right_tasks}>:
  +middle = Runs.range_high(left)
  +aligned = Equal.sym(Nat,middle,Runs.range_low(right),glue)
  Placed.join_plans(limits,layout,Runs.range_low(left),middle,Runs.range_high(right),left_tasks,right_tasks,left_order,
    %aligned : {Nat.is_le(_,Runs.range_high(right)) == True{} : Bool} right_order,
    left_seen,%aligned : Placed.ObservedPlan<limits,layout,_,Runs.range_high(right),right_tasks> right_seen)
def tree_plan(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+total: Nat,+limits: Machine.Limits,+rows: U32,+inner: U32,+channels: U32,
  +grid: Grid.Grid,+layout: Layout<limits,rows,channels,grid>,
  tasks: Pieces.Ranges,windows: Windows.WindowModel<Tile.Tile8>,+resource: Point.SharedResource,
  run: Runs.Run(tasks,U32.to_nat(rows),output_cells(rows,channels)),
  reads: Windows.WindowsOk(~Tile.Tile8,~Source,~value,~Products.Shape,~Products.shared_reach,source,Products.Shape{rows,inner,channels,grid},total,tasks,windows),
  facts: Shared.SharedTree(Products.Shape{rows,inner,channels,grid},tasks,windows,resource)) ->
  Placed.ObservedPlan<limits,Pieces.ranges_of_grid(grid),Runs.range_low(tasks),Runs.range_high(tasks),Shared.shared_results(Products.Shape{rows,inner,channels,grid},tasks,windows,resource,facts)>:
  match tasks windows:
    case Pieces.Range{+first,+count} Windows.WindowLeaf{+holding,+window_first,+window_count}:
      Held.Both{+first_same,+rest} = reads
      Held.Both{+count_same,+window_ok} = rest
      Grouped.TaskFact{+result,+returned,+exact,+regions} = facts
      +work = {Products.Shape{rows,inner,channels,grid} : Products.Shape}
      leaf_plan(limits,rows,inner,channels,grid,first,count,holding,resource,layout,run,result,
        %first_same : {Model.content(~Tile.Tile8,Placed.task_pieces(result)) ==
          Point.rectangle_model(Point.shared_geometry(work,_,count),leaf_stage(rows,inner,channels,grid,_,count,holding,resource)) : Model.Content<Tile.Tile8>}
        %count_same : {Model.content(~Tile.Tile8,Placed.task_pieces(result)) ==
          Point.rectangle_model(Point.shared_geometry(work,window_first,_),leaf_stage(rows,inner,channels,grid,window_first,_,holding,resource)) : Model.Content<Tile.Tile8>}
        leaf_contents(rows,inner,channels,grid,window_first,window_count,holding,resource,Grouped.TaskFact{result,returned,exact,regions}))
    case Pieces.Split{+left,+right} Windows.WindowFork{+left_windows,+right_windows}:
      Held.Both{+left_run,+next} = run
      Held.Both{+right_run,+glue} = next
      Held.Both{+left_reads,+right_reads} = reads
      Held.Both{+left_facts,+right_facts} = facts
      +work = {Products.Shape{rows,inner,channels,grid} : Products.Shape}
      +split = Shared.split_resources(resource)
      +left_resource = PartitionLaws.first(~Point.SharedResource,split)
      +right_resource = PartitionLaws.second(~Point.SharedResource,split)
      +unit = U32.to_nat(rows)
      +bound = output_cells(rows,channels)
      joined_plan(limits,Pieces.ranges_of_grid(grid),left,right,Shared.shared_results(work,left,left_windows,left_resource,left_facts),
        Shared.shared_results(work,right,right_windows,right_resource,right_facts),glue,
        run_order(left,unit,bound,Runs.run_fits(left,unit,bound,left_run)),run_order(right,unit,bound,Runs.run_fits(right,unit,bound,right_run)),
        tree_plan(~Source,~value,source,total,limits,rows,inner,channels,grid,layout,left,left_windows,left_resource,left_run,left_reads,left_facts),
        tree_plan(~Source,~value,source,total,limits,rows,inner,channels,grid,layout,right,right_windows,right_resource,right_run,right_reads,right_facts))
    case Pieces.Range{+first,+count} Windows.WindowFork{+left,+right}:
      Empty.absurd(Placed.ObservedPlan<limits,Pieces.ranges_of_grid(grid),U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),
        Shared.shared_results(Products.Shape{rows,inner,channels,grid},Pieces.Range{first,count},Windows.WindowFork{left,right},resource,facts)>,reads)
    case Pieces.Split{+left,+right} Windows.WindowLeaf{+holding,+first,+count}:
      Empty.absurd(Placed.ObservedPlan<limits,Pieces.ranges_of_grid(grid),Runs.range_low(left),Runs.range_high(right),
        Shared.shared_results(Products.Shape{rows,inner,channels,grid},Pieces.Split{left,right},Windows.WindowLeaf{holding,first,count},resource,facts)>,reads)

type Actual<-limits: Machine.Limits,-grid: Grid.Grid,-tasks: Pieces.Ranges,-native: Pieces.Pieces(~Tile.Tile8)> is Data:
  Actual{plan: Placed.Plan<limits,Pieces.ranges_of_grid(grid),Runs.range_low(tasks),Runs.range_high(tasks)>,
    exact: {native == Tensors.placed(~Tile.Tile8,Pieces.ranges_of_grid(grid),Products.zero(),
      Shared.pack_task(Placed.planned_tasks(limits,Pieces.ranges_of_grid(grid),Runs.range_low(tasks),Runs.range_high(tasks),plan))) : Pieces.Pieces(~Tile.Tile8)}}

def actual_plan(+limits: Machine.Limits,+grid: Grid.Grid,+tasks: Pieces.Ranges,-native: Pieces.Pieces(~Tile.Tile8),+model: Shared.TaskModel(),
  +placed: {native == Tensors.placed(~Tile.Tile8,Pieces.ranges_of_grid(grid),Products.zero(),Shared.pack_task(model)) : Pieces.Pieces(~Tile.Tile8)},
  +observed: Placed.ObservedPlan<limits,Pieces.ranges_of_grid(grid),Runs.range_low(tasks),Runs.range_high(tasks),model>) ->
  Actual<limits,grid,tasks,native>:
  Placed.ObservedPlan{+plan,+same} = observed
  +layout = Pieces.ranges_of_grid(grid)
  +results = Placed.planned_tasks(limits,layout,Runs.range_low(tasks),Runs.range_high(tasks),plan)
  Actual{plan,%Equal.sym(Shared.TaskModel(),results,model,same) : {native == Tensors.placed(~Tile.Tile8,layout,Products.zero(),Shared.pack_task(_)) : Pieces.Pieces(~Tile.Tile8)} placed}

def scheduled_plan(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+total: Nat,+limits: Machine.Limits,
  +rows: U32,+inner: U32,+channels: U32,+grid: Grid.Grid,+tasks: Pieces.Ranges,+resource: Point.SharedResource,-native: Pieces.Pieces(~Tile.Tile8),
  +layout: Layout<limits,rows,channels,grid>,+run: Runs.Run(tasks,U32.to_nat(rows),output_cells(rows,channels)),
  +scheduled: Shared.ScheduledShared<Source,value,source,total,Products.Shape{rows,inner,channels,grid},tasks,resource,grid,native>) ->
  Actual<limits,grid,tasks,native>:
  Shared.ScheduledShared{+windows,+reads,+facts,+placed} = scheduled
  +work = {Products.Shape{rows,inner,channels,grid} : Products.Shape}
  actual_plan(limits,grid,tasks,native,Shared.shared_results(work,tasks,windows,resource,facts),placed,
    tree_plan(~Source,~value,source,total,limits,rows,inner,channels,grid,layout,tasks,windows,resource,run,reads,facts))

def prepared_layout(+limits: Machine.Limits,+rows: U32,+channels: U32,
  +channel_room: {Nat.is_le(U32.to_nat(channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +rows_room: {Nat.is_le(U32.to_nat(rows),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +out_room: {Nat.is_le(ShapeLaws.elements([Gemm.blocks(channels),1,rows]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +out_nonempty: {Nat.is_lt(0n,ShapeLaws.elements([Gemm.blocks(channels),1,rows])) == True{} : Bool}) ->
  Layout<limits,rows,channels,Shared.shared_grid(rows,channels)>:
  +shape = {[Gemm.blocks(channels),1,rows] : List<&2,U32>}
  +grid = Shared.shared_grid(rows,channels)
  +target = Grid.piece_target()
  +unit = U32.to_nat(rows)
  +bound = output_cells(rows,channels)
  +shape_exact = Shared.shared_cells(rows,channels)
  +divides = {GridRanges.Multiple{1n,Equal.sym(Nat,Nat.add(unit,0n),unit,Addition.zero_right(unit))} : GridRanges.Multiple<unit,unit>}
  +room = {%shape_exact : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} out_room :
    {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +layout = GridRanges.grid_layout(limits,shape,target,out_room,out_nonempty,unit,divides)
  Layout{GridRanges.grid_facts(limits,shape,target,out_room,out_nonempty),channel_room,rows_positive,rows_room,
    Equal.trans(Nat,GridLaws.grid_cells(grid),ShapeLaws.elements(shape),bound,GridRanges.grid_count(limits,shape,target,out_room,out_nonempty),shape_exact),room,
    %shape_exact : Runs.Run(Pieces.ranges_of_grid(grid),unit,_)
    GridRanges.layout_run(shape,target,unit,layout)}

def decided_plan(~Source: Data,~value: Source -> U32 -> Tile.Tile8,+source: Source,+limits: Machine.Limits,
  +rows: U32,+inner: U32,+channels: U32,+workers: U32,+resource: Point.SharedResource,
  +channel_room: {Nat.is_le(U32.to_nat(channels),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +rows_room: {Nat.is_le(U32.to_nat(rows),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +out_room: {Nat.is_le(ShapeLaws.elements([Gemm.blocks(channels),1,rows]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +out_nonempty: {Nat.is_lt(0n,ShapeLaws.elements([Gemm.blocks(channels),1,rows])) == True{} : Bool},
  +weights_room: {Nat.is_le(ShapeLaws.elements(Shared.weight_shape(inner,channels)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +weights_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Shared.weight_shape(inner,channels))) == True{} : Bool},weights: Pieces.Pieces(~Tile.Tile8),
  +covered: Held.Covered<Tile.Tile8,Source,value,source,Move.held(~Tile.Tile8,weights),0n,ShapeLaws.elements(Shared.weight_shape(inner,channels)),
    Held.layout_spans(Pieces.ranges_of_grid(Tensors.grid_of(Shared.weight_shape(inner,channels))),Held.Ended{})>) ->
  Actual<limits,Shared.shared_grid(rows,channels),Runs.decision_tasks(Shared.shared_decision(rows,inner,channels,workers)),
    Shared.shared_execution(Products.Shape{rows,inner,channels,Shared.shared_grid(rows,channels)},Tensors.grid_of(Shared.weight_shape(inner,channels)),Shared.shared_decision(rows,inner,channels,workers),resource,weights)>:
  +grid = Shared.shared_grid(rows,channels)
  +weight_grid = Tensors.grid_of(Shared.weight_shape(inner,channels))
  +total = ShapeLaws.elements(Shared.weight_shape(inner,channels))
  +decision = Shared.shared_decision(rows,inner,channels,workers)
  +tasks = Runs.decision_tasks(decision)
  +work = {Products.Shape{rows,inner,channels,grid} : Products.Shape}
  +unit = U32.to_nat(rows)
  +bound = output_cells(rows,channels)
  +target = Pieces.ranges_of_grid(grid)
  +layout = prepared_layout(limits,rows,channels,channel_room,rows_positive,rows_room,out_room,out_nonempty)
  +plan_run = Shared.shared_run(limits,rows,inner,channels,workers,rows_room,out_room,out_nonempty)
  +windowed = Shared.actual_shared_windows(~Source,~value,source,limits,rows,inner,channels,workers,rows_positive,rows_room,out_room,out_nonempty,
    weights_room,weights_nonempty,weights,covered)
  scheduled_plan(~Source,~value,source,total,limits,rows,inner,channels,grid,tasks,resource,Shared.shared_execution(work,weight_grid,decision,resource,weights),layout,
    Held.first_of(Runs.Run(tasks,unit,bound),Held.Both<{Runs.range_low(tasks) == Runs.range_low(target) : Nat},{Runs.range_high(tasks) == Runs.range_high(target) : Nat}>,plan_run),
    Shared.scheduled_decision(~Source,~value,source,total,work,weight_grid,decision,resource,weights,windowed))
